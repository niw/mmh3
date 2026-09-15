//! NVFP4 linear layers through cuBLASLt's block-scaled FP4 GEMM, with weights requantized from the
//! INT8 ConvRot checkpoints and activations rotated like ConvRot activations.
//!
//! The tensor scale of the activations comes from their largest magnitude. The first call of a
//! layer finds it in a pass of its own, and later calls take the previous call's value with a
//! margin, so that the quantization can fuse into the pass that produces the rows.

use crate::{CudaError, DeviceBuffer, check};
use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_nvfp4_quantize(
        input: *const c_void,
        swiglu: c_int,
        values: *mut c_void,
        scales: *mut c_void,
        tensor_scale: *mut c_void,
        reference: *mut c_void,
        margin: f32,
        observed: *mut c_void,
        exact: c_int,
        m: c_int,
        k: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_nvfp4_alpha(
        tensor_scale: *const c_void,
        weight_scale: f32,
        alpha_beta: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_nvfp4_quantize_weights(
        weights: *const c_void,
        row_scales: *const c_void,
        values: *mut c_void,
        scales: *mut c_void,
        tensor_scale: f32,
        n: c_int,
        k: c_int,
        deinterleave: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_cublaslt_nvfp4(
        weights: *const c_void,
        weight_scales: *const c_void,
        activations: *const c_void,
        activation_scales: *const c_void,
        alpha_beta: *const c_void,
        output: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut c_void,
    ) -> c_int;
}

/// How far the activations may grow over the previous call's largest magnitude before their block
/// scales saturate.
pub const DELAYED_MARGIN: f32 = 2.0;

/// Bytes of the block scales of `rows` rows of `features` values: one per 16 values, with the rows
/// padded to a multiple of 128.
fn scale_bytes(rows: usize, features: usize) -> usize {
    rows.next_multiple_of(128) * features / 16
}

/// A weight `[outputs, features]` in NVFP4.
pub struct Nvfp4Weight {
    values: DeviceBuffer,
    scales: DeviceBuffer,
    tensor_scale: f32,
    pub outputs: usize,
    pub features: usize,
}

impl Nvfp4Weight {
    /// Requantizes an INT8 ConvRot weight `[outputs, features]` with one f32 scale per row. With
    /// `deinterleave_swiglu`, the rows went through `interleave_swiglu_rows` and come back in their
    /// checkpoint order, gates first.
    pub fn from_int8(
        weights: &DeviceBuffer,
        row_scales: &DeviceBuffer,
        outputs: usize,
        features: usize,
        deinterleave_swiglu: bool,
    ) -> Result<Self, CudaError> {
        assert!(
            weights.bytes() >= outputs * features && row_scales.bytes() >= outputs * 4,
            "the INT8 weight is smaller than outputs × features"
        );
        // The rows' largest magnitudes are at most 128 steps of their scales.
        let largest_scale = row_scales
            .to_f32_range(0, outputs)?
            .into_iter()
            .fold(0.0f32, f32::max);
        let tensor_scale = (128.0 * largest_scale / (6.0 * 448.0)).max(f32::MIN_POSITIVE);
        let values = DeviceBuffer::new(outputs * features / 2)?;
        let scales = DeviceBuffer::zeroed(scale_bytes(outputs, features))?;
        // SAFETY: the buffers cover the extents the kernel touches, checked above and allocated
        // here.
        check(unsafe {
            mmh3_nvfp4_quantize_weights(
                weights.pointer(),
                row_scales.pointer(),
                values.pointer(),
                scales.pointer(),
                tensor_scale,
                outputs as c_int,
                features as c_int,
                deinterleave_swiglu as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(Nvfp4Weight {
            values,
            scales,
            tensor_scale,
            outputs,
            features,
        })
    }
}

/// The largest activation magnitudes of one layer's last two calls, which give the next call its
/// tensor scale.
pub struct Nvfp4Scale {
    maxima: DeviceBuffer,
    calls: Cell<u64>,
}

/// Where one call reads and writes its largest magnitudes.
pub(crate) struct ScaleSlots {
    /// The magnitude that gives the tensor scale, with `margin`.
    pub(crate) reference: *mut c_void,
    pub(crate) margin: f32,
    /// Where the call's own largest magnitude goes, or null when `exact`.
    pub(crate) observed: *mut c_void,
    /// Whether the call finds its own largest magnitude first and uses it.
    pub(crate) exact: bool,
}

impl Nvfp4Scale {
    pub fn new() -> Result<Self, CudaError> {
        Ok(Nvfp4Scale {
            maxima: DeviceBuffer::zeroed(2 * 4)?,
            calls: Cell::new(0),
        })
    }

    /// Whether a previous call left its largest magnitude for the next one.
    pub fn is_calibrated(&self) -> bool {
        self.calls.get() > 0
    }

    /// The slots of the next call. The first call is exact, and later calls alternate between the
    /// two slots.
    pub(crate) fn next_call(&self) -> ScaleSlots {
        let call = self.calls.get();
        self.calls.set(call + 1);
        let slot = |index: u64| {
            // SAFETY: the buffer holds two u32 slots.
            unsafe {
                self.maxima
                    .pointer()
                    .cast::<u32>()
                    .add((index % 2) as usize)
                    .cast()
            }
        };
        if call == 0 {
            ScaleSlots {
                reference: slot(0),
                margin: 1.0,
                observed: ptr::null_mut(),
                exact: true,
            }
        } else {
            ScaleSlots {
                reference: slot(call - 1),
                margin: DELAYED_MARGIN,
                observed: slot(call),
                exact: false,
            }
        }
    }
}

/// Buffers for the NVFP4 activations of up to `rows` rows of up to `features` values.
pub struct Nvfp4Activations {
    pub(crate) values: DeviceBuffer,
    pub(crate) scales: DeviceBuffer,
    /// The tensor scale of the quantized activations, then alpha and beta for the GEMM.
    pub(crate) scratch: DeviceBuffer,
    rows: usize,
    features: usize,
}

impl Nvfp4Activations {
    pub fn new(rows: usize, features: usize) -> Result<Self, CudaError> {
        Ok(Nvfp4Activations {
            values: DeviceBuffer::new(rows * features / 2)?,
            scales: DeviceBuffer::zeroed(scale_bytes(rows, features))?,
            scratch: DeviceBuffer::new(3 * 4)?,
            rows,
            features,
        })
    }

    /// Checks that `rows` rows of `features` values fit.
    pub(crate) fn check_fits(&self, rows: usize, features: usize) {
        assert!(
            rows <= self.rows && features <= self.features,
            "the NVFP4 activation buffers are too small"
        );
    }
}

/// What the activations of a layer are made from.
#[derive(Clone, Copy)]
pub enum Nvfp4Input {
    /// BF16 rows `[rows, features]`.
    Rows(*const c_void),
    /// BF16 gates and up projections `[rows, 2 × features]`, whose SwiGLU is the input.
    SwiGlu(*const c_void),
}

/// Quantizes `rows` rows of `features` inputs into `activations` with the next tensor scale of
/// `scale`.
///
/// # Safety
/// The input must hold its `rows` rows.
pub unsafe fn quantize_pointers(
    input: Nvfp4Input,
    rows: usize,
    features: usize,
    scale: &Nvfp4Scale,
    activations: &Nvfp4Activations,
) -> Result<(), CudaError> {
    activations.check_fits(rows, features);
    let (pointer, swiglu) = match input {
        Nvfp4Input::Rows(pointer) => (pointer, false),
        Nvfp4Input::SwiGlu(pointer) => (pointer, true),
    };
    let slots = scale.next_call();
    // SAFETY: the caller guarantees the input extent, and the activation buffers cover `rows ×
    // features`, checked above.
    check(unsafe {
        mmh3_nvfp4_quantize(
            pointer,
            swiglu as c_int,
            activations.values.pointer(),
            activations.scales.pointer(),
            activations.scratch.pointer(),
            slots.reference,
            slots.margin,
            slots.observed,
            slots.exact as c_int,
            rows as c_int,
            features as c_int,
            ptr::null_mut(),
        )
    })
}

/// `output[rows, outputs] = activations · weightᵀ` in BF16 for activations quantized for this
/// weight's input size.
///
/// # Safety
/// `output` must hold `rows × outputs` BF16 values.
pub unsafe fn gemm_pointers(
    weight: &Nvfp4Weight,
    activations: &Nvfp4Activations,
    output: *mut c_void,
    rows: usize,
) -> Result<(), CudaError> {
    activations.check_fits(rows, weight.features);
    let scratch = activations.scratch.pointer().cast::<f32>();
    // SAFETY: the caller guarantees the output extent, and the scratch holds three floats.
    unsafe {
        check(mmh3_nvfp4_alpha(
            scratch.cast(),
            weight.tensor_scale,
            scratch.add(1).cast(),
            ptr::null_mut(),
        ))?;
        check(mmh3_cublaslt_nvfp4(
            weight.values.pointer(),
            weight.scales.pointer(),
            activations.values.pointer(),
            activations.scales.pointer(),
            scratch.add(1).cast(),
            output,
            rows as i64,
            weight.outputs as i64,
            weight.features as i64,
            ptr::null_mut(),
        ))
    }
}

/// Quantizes BF16 rows `input` with `scale` and multiplies them with `weight` into `output`, both
/// holding `rows` rows.
pub fn linear(
    weight: &Nvfp4Weight,
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    rows: usize,
    scale: &Nvfp4Scale,
    activations: &Nvfp4Activations,
) -> Result<(), CudaError> {
    assert!(
        input.bytes() >= rows * weight.features * 2 && output.bytes() >= rows * weight.outputs * 2,
        "the input or output is smaller than its rows"
    );
    // SAFETY: both buffers cover their rows, checked above.
    unsafe {
        quantize_pointers(
            Nvfp4Input::Rows(input.pointer()),
            rows,
            weight.features,
            scale,
            activations,
        )?;
        gemm_pointers(weight, activations, output.pointer(), rows)
    }
}

/// `linear` for the SwiGLU of gates and up projections `[rows, 2 × features]`.
pub fn linear_swiglu(
    weight: &Nvfp4Weight,
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    rows: usize,
    scale: &Nvfp4Scale,
    activations: &Nvfp4Activations,
) -> Result<(), CudaError> {
    assert!(
        input.bytes() >= rows * weight.features * 4 && output.bytes() >= rows * weight.outputs * 2,
        "the input or output is smaller than its rows"
    );
    // SAFETY: both buffers cover their rows, checked above.
    unsafe {
        quantize_pointers(
            Nvfp4Input::SwiGlu(input.pointer()),
            rows,
            weight.features,
            scale,
            activations,
        )?;
        gemm_pointers(weight, activations, output.pointer(), rows)
    }
}
