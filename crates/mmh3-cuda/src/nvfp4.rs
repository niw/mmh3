//! NVFP4 linear layers through cuBLASLt's block-scaled FP4 GEMM, with weights requantized from the
//! INT8 ConvRot checkpoints and activations rotated like ConvRot activations.
//!
//! The tensor scale of the activations comes from their largest magnitude. The first call of a
//! layer finds it in a pass of its own, and later calls take the previous call's value with a
//! margin, so that the quantization can fuse into the pass that produces the rows.
//!
//! A low-rank adapter runs inside the layer's GEMM as extra columns, `[x | z] · [W | B]ᵀ` with
//! `z = x · Aᵀ`: its up projection B follows the weight's inputs in every row, and before the GEMM
//! its down projection A runs as an NVFP4 GEMM of its own on the quantized input and writes z into
//! the activations' columns after the inputs.

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
        columns: c_int,
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
        columns: c_int,
        deinterleave: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_nvfp4_quantize_columns(
        input: *const c_void,
        multiplier: f32,
        values: *mut c_void,
        scales: *mut c_void,
        tensor_scale: *const c_void,
        fixed_tensor_scale: f32,
        deinterleave: c_int,
        m: c_int,
        count: c_int,
        offset: c_int,
        columns: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_nvfp4_adapter_ranges(
        down: *const c_void,
        down_scales: *const c_void,
        rank: c_int,
        k: c_int,
        up: *const c_void,
        up_count: usize,
        maxima: *mut c_void,
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

/// Norm of the longest row of an adapter's down projection, which is scaled to it while its up
/// projection takes the inverse. The adapter's activations share the tensor scale of the layer's
/// inputs, and with this they are about a sixteenth of the inputs' root mean square, well inside
/// the range of the block scales both when rows line up with the largest inputs and when they
/// miss them.
pub const ADAPTER_DOWN_NORM: f32 = 1.0 / 16.0;

/// Bytes of the block scales of `rows` rows of `columns` values: one per 16 values, with the rows
/// padded to a multiple of 128.
fn scale_bytes(rows: usize, columns: usize) -> usize {
    rows.next_multiple_of(128) * columns / 16
}

/// A weight `[outputs, features]` in NVFP4, with room for more columns in each row.
pub struct Nvfp4Weight {
    values: DeviceBuffer,
    scales: DeviceBuffer,
    tensor_scale: f32,
    pub outputs: usize,
    /// Inputs of the layer.
    pub features: usize,
    /// Values in each row: the inputs, then the up projection of the adapter, and zeros.
    pub columns: usize,
    /// The adapter's down projection `[rank, columns]`, with zeros after the inputs.
    down: Option<Box<Nvfp4Weight>>,
}

/// A low-rank adapter `scale · up · down` for `Nvfp4Weight::from_int8_with`.
pub struct AdapterSource<'a> {
    /// INT8 `[rank, features]`, rotated like the activations, with one f32 scale per row in
    /// `down_scales`.
    pub down: &'a DeviceBuffer,
    pub down_scales: &'a DeviceBuffer,
    /// BF16 `[outputs, rank]`, in the order of the weight's rows.
    pub up: &'a DeviceBuffer,
    /// A multiple of 64.
    pub rank: usize,
    pub scale: f32,
}

/// What follows the inputs in each row of an NVFP4 weight.
pub enum Columns<'a> {
    /// Nothing.
    Inputs,
    /// Zeros up to this many columns in all, for a layer that takes the activations of another
    /// layer, together with that layer's adapter columns.
    Zeros(usize),
    /// The up projection of an adapter, then zeros up to a multiple of 256 columns.
    Adapter(AdapterSource<'a>),
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
        Self::from_int8_with(
            weights,
            row_scales,
            outputs,
            features,
            deinterleave_swiglu,
            Columns::Inputs,
        )
    }

    /// `from_int8` with `columns` after the inputs. With `deinterleave_swiglu`, an adapter's up
    /// rows went through `interleave_swiglu_rows` too.
    pub fn from_int8_with(
        weights: &DeviceBuffer,
        row_scales: &DeviceBuffer,
        outputs: usize,
        features: usize,
        deinterleave_swiglu: bool,
        columns: Columns,
    ) -> Result<Self, CudaError> {
        assert!(
            weights.bytes() >= outputs * features && row_scales.bytes() >= outputs * 4,
            "the INT8 weight is smaller than outputs × features"
        );
        // NOTE: cuBLASLt's FP4 GEMM runs K in tiles of 256 values, so an adapter costs a whole tile
        // anyway, and rows padded to the tile take about 0.15 s less per step at 768p than rows
        // that end inside one.
        let total = match &columns {
            Columns::Inputs => features,
            Columns::Zeros(total) => *total,
            Columns::Adapter(adapter) => (features + adapter.rank).next_multiple_of(256),
        };
        assert!(
            total >= features && total.is_multiple_of(64),
            "the columns of an NVFP4 weight must cover its inputs and be a multiple of 64"
        );
        // The rows' largest magnitudes are at most 128 steps of their scales.
        let mut largest = 128.0
            * row_scales
                .to_f32_range(0, outputs)?
                .into_iter()
                .fold(0.0f32, f32::max);
        let adapter = match columns {
            Columns::Adapter(adapter) => {
                let (down_norm, up_maximum) = adapter.ranges(outputs, features)?;
                let balance = if down_norm > 0.0 {
                    ADAPTER_DOWN_NORM / down_norm
                } else {
                    1.0
                };
                let multiplier = adapter.scale / balance;
                largest = largest.max(up_maximum * multiplier.abs());
                Some((adapter, balance, multiplier))
            }
            _ => None,
        };
        let tensor_scale = (largest / (6.0 * 448.0)).max(f32::MIN_POSITIVE);
        let values = DeviceBuffer::zeroed(outputs * total / 2)?;
        let scales = DeviceBuffer::zeroed(scale_bytes(outputs, total))?;
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
                total as c_int,
                deinterleave_swiglu as c_int,
                ptr::null_mut(),
            )
        })?;
        let down = match adapter {
            Some((adapter, balance, multiplier)) => {
                // SAFETY: up holds `outputs × rank` values, checked in `ranges`, and the adapter's
                // columns lie inside the rows allocated here.
                check(unsafe {
                    mmh3_nvfp4_quantize_columns(
                        adapter.up.pointer(),
                        multiplier,
                        values.pointer(),
                        scales.pointer(),
                        ptr::null(),
                        tensor_scale,
                        deinterleave_swiglu as c_int,
                        outputs as c_int,
                        adapter.rank as c_int,
                        features as c_int,
                        total as c_int,
                        ptr::null_mut(),
                    )
                })?;
                let mut down = Self::from_int8_with(
                    adapter.down,
                    adapter.down_scales,
                    adapter.rank,
                    features,
                    false,
                    Columns::Zeros(total),
                )?;
                down.tensor_scale *= balance;
                Some(Box::new(down))
            }
            None => None,
        };
        Ok(Nvfp4Weight {
            values,
            scales,
            tensor_scale,
            outputs,
            features,
            columns: total,
            down,
        })
    }

    /// Rank of the adapter, or 0 without one.
    pub fn adapter_rank(&self) -> usize {
        self.down.as_ref().map_or(0, |down| down.outputs)
    }
}

impl AdapterSource<'_> {
    /// The norm of the longest down row and the largest up magnitude.
    fn ranges(&self, outputs: usize, features: usize) -> Result<(f32, f32), CudaError> {
        assert!(
            self.rank.is_multiple_of(64)
                && self.down.bytes() >= self.rank * features
                && self.down_scales.bytes() >= self.rank * 4
                && self.up.bytes() >= outputs * self.rank * 2,
            "the adapter does not fit the layer"
        );
        let maxima = DeviceBuffer::new(2 * 4)?;
        // SAFETY: the buffers cover the extents, checked above.
        check(unsafe {
            mmh3_nvfp4_adapter_ranges(
                self.down.pointer(),
                self.down_scales.pointer(),
                self.rank as c_int,
                features as c_int,
                self.up.pointer(),
                outputs * self.rank,
                maxima.pointer(),
                ptr::null_mut(),
            )
        })?;
        let maxima = maxima.to_f32()?;
        Ok((maxima[0], maxima[1]))
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

/// Buffers for the NVFP4 activations of up to `rows` rows of up to `columns` values, for layers
/// with adapters of up to `adapter_rank`.
pub struct Nvfp4Activations {
    pub(crate) values: DeviceBuffer,
    pub(crate) scales: DeviceBuffer,
    /// The tensor scale of the quantized activations, then alpha and beta for the GEMM.
    pub(crate) scratch: DeviceBuffer,
    /// BF16 down projections of an adapter `[rows, rank]`.
    adapter: DeviceBuffer,
    rows: usize,
    columns: usize,
    adapter_rank: usize,
}

impl Nvfp4Activations {
    pub fn new(rows: usize, columns: usize, adapter_rank: usize) -> Result<Self, CudaError> {
        Ok(Nvfp4Activations {
            values: DeviceBuffer::new(rows * columns / 2)?,
            scales: DeviceBuffer::zeroed(scale_bytes(rows, columns))?,
            scratch: DeviceBuffer::new(3 * 4)?,
            adapter: DeviceBuffer::new(rows * adapter_rank.max(1) * 2)?,
            rows,
            columns,
            adapter_rank,
        })
    }

    /// Checks that `rows` rows of `columns` values fit.
    pub(crate) fn check_fits(&self, rows: usize, columns: usize) {
        assert!(
            rows <= self.rows && columns <= self.columns,
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

/// Quantizes `rows` rows of inputs of `weight` into `activations` with the next tensor scale of
/// `scale`.
///
/// # Safety
/// The input must hold its `rows` rows.
pub unsafe fn quantize_pointers(
    input: Nvfp4Input,
    rows: usize,
    weight: &Nvfp4Weight,
    scale: &Nvfp4Scale,
    activations: &Nvfp4Activations,
) -> Result<(), CudaError> {
    activations.check_fits(rows, weight.columns);
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
            weight.features as c_int,
            weight.columns as c_int,
            ptr::null_mut(),
        )
    })
}

/// `output[rows, outputs] = activations · weightᵀ` in BF16 for activations quantized for this
/// weight's columns. With an adapter, its down projection first fills the activations' columns
/// after the inputs.
///
/// # Safety
/// `output` must hold `rows × outputs` BF16 values.
pub unsafe fn gemm_pointers(
    weight: &Nvfp4Weight,
    activations: &Nvfp4Activations,
    output: *mut c_void,
    rows: usize,
) -> Result<(), CudaError> {
    activations.check_fits(rows, weight.columns);
    if let Some(down) = &weight.down {
        assert!(
            down.outputs <= activations.adapter_rank,
            "the NVFP4 adapter buffer is too small"
        );
        // SAFETY: the adapter buffer holds `rows × rank` values, checked above, and the columns
        // after the inputs lie inside the activation rows.
        unsafe {
            gemm_pointers(down, activations, activations.adapter.pointer(), rows)?;
            check(mmh3_nvfp4_quantize_columns(
                activations.adapter.pointer(),
                1.0,
                activations.values.pointer(),
                activations.scales.pointer(),
                activations.scratch.pointer(),
                0.0,
                0,
                rows as c_int,
                down.outputs as c_int,
                weight.features as c_int,
                weight.columns as c_int,
                ptr::null_mut(),
            ))?;
        }
    }
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
            weight.columns as i64,
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
            weight,
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
            weight,
            scale,
            activations,
        )?;
        gemm_pointers(weight, activations, output.pointer(), rows)
    }
}
