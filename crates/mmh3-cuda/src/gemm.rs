//! GEMM kernels for the DiT linear layers.

use crate::{CudaError, DeviceBuffer, check};
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_rotate_quantize(
        input: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        tokens: c_int,
        columns: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_int8_gemm_config_count() -> c_int;
    fn mmh3_int8_gemm_bf16_adapter(
        config: c_int,
        activations: *const c_void,
        weights: *const c_void,
        activation_scales: *const c_void,
        weight_scales: *const c_void,
        output: *mut c_void,
        m: c_int,
        n: c_int,
        k: c_int,
        adapter_down: *const c_void,
        adapter_up: *const c_void,
        rank: c_int,
        adapter_scale: f32,
        stream: *mut c_void,
    ) -> c_int;
}

/// Raw form of `rotate_quantize`.
///
/// # Safety
/// `input` must hold `rows × columns` BF16 values, `output` `rows × columns` bytes and `scales` `rows` f32 values.
pub(crate) unsafe fn rotate_quantize_pointers(
    input: *const c_void,
    output: *mut c_void,
    scales: *mut c_void,
    rows: usize,
    columns: usize,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    check(unsafe { mmh3_rotate_quantize(input, output, scales, rows as c_int, columns as c_int, ptr::null_mut()) })
}

/// Prepares the activations of an INT8 ConvRot layer: rotates every group of 256 columns of the BF16 rows by the
/// normalized regular Hadamard matrix and quantizes each row to INT8 with scale max |x| / 127, rounding half to even.
/// Columns must be a multiple of 256, up to 32,768.
pub fn rotate_quantize(
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    scales: &mut DeviceBuffer,
    rows: usize,
    columns: usize,
) -> Result<(), CudaError> {
    assert!(input.bytes() >= rows * columns * 2, "input is smaller than rows × columns");
    assert!(output.bytes() >= rows * columns, "output is smaller than rows × columns");
    assert!(scales.bytes() >= rows * 4, "scales are smaller than rows");
    // SAFETY: every buffer covers the extent the kernel touches, checked above.
    unsafe { rotate_quantize_pointers(input.pointer(), output.pointer(), scales.pointer(), rows, columns) }
}

/// Number of tile configurations the INT8 GEMM kernel is compiled for.
pub fn int8_config_count() -> usize {
    // SAFETY: no arguments.
    unsafe { mmh3_int8_gemm_config_count() as usize }
}

/// A low-rank adapter added to an INT8 GEMM: `scale · down · upᵀ` with `down` `[m, rank]` and `up` `[n, rank]`, both
/// BF16, and the rank a multiple of 64.
pub struct Adapter<'a> {
    pub down: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub rank: usize,
    pub scale: f32,
}

/// Raw form of `Adapter`.
#[derive(Clone, Copy)]
pub(crate) struct AdapterPointers {
    pub(crate) down: *const c_void,
    pub(crate) up: *const c_void,
    pub(crate) rank: usize,
    pub(crate) scale: f32,
}

/// Raw form of `int8_bf16` that picks the tile shape: 128 × 256 for inputs shorter than 256 rows, such as prompts,
/// and 256 × 128 otherwise.
///
/// # Safety
/// The pointers must cover the extents described for `int8_bf16`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_bf16_pointers(
    activations: *const c_void,
    weights: *const c_void,
    activation_scales: *const c_void,
    weight_scales: *const c_void,
    output: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<AdapterPointers>,
) -> Result<(), CudaError> {
    let config = if m < 256 && n % 256 == 0 { 1 } else { 0 };
    // SAFETY: the caller guarantees the extents.
    unsafe { int8_bf16_config(config, activations, weights, activation_scales, weight_scales, output, m, n, k, adapter) }
}

/// # Safety
/// The pointers must cover the extents described for `int8_bf16`.
#[allow(clippy::too_many_arguments)]
unsafe fn int8_bf16_config(
    config: usize,
    activations: *const c_void,
    weights: *const c_void,
    activation_scales: *const c_void,
    weight_scales: *const c_void,
    output: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<AdapterPointers>,
) -> Result<(), CudaError> {
    let adapter = adapter.unwrap_or(AdapterPointers { down: ptr::null(), up: ptr::null(), rank: 0, scale: 0.0 });
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_int8_gemm_bf16_adapter(
            config as c_int,
            activations,
            weights,
            activation_scales,
            weight_scales,
            output,
            m as c_int,
            n as c_int,
            k as c_int,
            adapter.down,
            adapter.up,
            adapter.rank as c_int,
            adapter.scale,
            ptr::null_mut(),
        )
    })
}

/// output[m, n] = bf16(Σₖ activations[m, k] · weights[n, k] · activation_scales[m] · weight_scales[n] + the adapter).
///
/// Activations and weights are row-major INT8 with K contiguous, scales are f32. K must be a multiple of 128 and N
/// a multiple of the config's tile width, 128 for config 0 and 256 for config 1.
#[allow(clippy::too_many_arguments)]
pub fn int8_bf16(
    config: usize,
    activations: &DeviceBuffer,
    weights: &DeviceBuffer,
    activation_scales: &DeviceBuffer,
    weight_scales: &DeviceBuffer,
    output: &mut DeviceBuffer,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<&Adapter>,
) -> Result<(), CudaError> {
    assert!(activations.bytes() >= m * k, "activations are smaller than m × k");
    assert!(weights.bytes() >= n * k, "weights are smaller than n × k");
    assert!(activation_scales.bytes() >= m * 4, "activation scales are smaller than m");
    assert!(weight_scales.bytes() >= n * 4, "weight scales are smaller than n");
    assert!(output.bytes() >= m * n * 2, "output is smaller than m × n");
    if let Some(adapter) = adapter {
        assert!(adapter.down.bytes() >= m * adapter.rank * 2, "adapter down activations are smaller than m × rank");
        assert!(adapter.up.bytes() >= n * adapter.rank * 2, "adapter up weights are smaller than n × rank");
    }
    let adapter = adapter.map(|adapter| AdapterPointers {
        down: adapter.down.pointer(),
        up: adapter.up.pointer(),
        rank: adapter.rank,
        scale: adapter.scale,
    });
    // SAFETY: every buffer covers the extent the kernel touches, checked above.
    unsafe {
        int8_bf16_config(
            config,
            activations.pointer(),
            weights.pointer(),
            activation_scales.pointer(),
            weight_scales.pointer(),
            output.pointer(),
            m,
            n,
            k,
            adapter,
        )
    }
}
