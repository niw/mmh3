//! GEMM kernels for the DiT linear layers.

use crate::{CudaError, DeviceBuffer, check};
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_rotate_quantize(
        input: *const c_void,
        input_is_f16: c_int,
        output: *mut c_void,
        scales: *mut c_void,
        tokens: c_int,
        columns: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_int8_gemm_config_count() -> c_int;
    fn mmh3_int8_gemm_group_heights(heights: *mut c_int, capacity: c_int) -> c_int;
    fn mmh3_int8_gemm(
        config: c_int,
        group_m: c_int,
        activations: *const c_void,
        weights: *const c_void,
        activation_scales: *const c_void,
        weight_scales: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        output_is_f16: c_int,
        swiglu: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        output_stride: c_int,
        adapter_down: *const c_void,
        adapter_up: *const c_void,
        rank: c_int,
        adapter_down_stride: c_int,
        adapter_scale: f32,
        column_maxima: *mut c_void,
        maxima_first_column: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_merge_low_rank(
        weights: *mut c_void,
        scales: *mut c_void,
        up: *const c_void,
        down: *mut c_void,
        n: c_int,
        k: c_int,
        rank: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_interleave_swiglu_rows(
        source: *const c_void,
        destination: *mut c_void,
        rows: c_int,
        row_bytes: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

/// Raw form of `rotate_quantize`, for BF16 or FP16 input.
///
/// # Safety
/// `input` must hold `rows × columns` 16-bit values, `output` `rows × columns` bytes and `scales`
/// `rows` f32 values.
pub(crate) unsafe fn rotate_quantize_pointers(
    input: *const c_void,
    input_is_f16: bool,
    output: *mut c_void,
    scales: *mut c_void,
    rows: usize,
    columns: usize,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_rotate_quantize(
            input,
            input_is_f16 as c_int,
            output,
            scales,
            rows as c_int,
            columns as c_int,
            ptr::null_mut(),
        )
    })
}

/// Prepares the activations of an INT8 ConvRot layer: rotates every group of 256 columns of the
/// BF16 rows by the normalized regular Hadamard matrix and quantizes each row to INT8 with scale
/// max |x| / 127, rounding half to even. Columns must be a multiple of 256, up to 32,768.
pub fn rotate_quantize(
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    scales: &mut DeviceBuffer,
    rows: usize,
    columns: usize,
) -> Result<(), CudaError> {
    assert!(
        input.bytes() >= rows * columns * 2,
        "input is smaller than rows × columns"
    );
    assert!(
        output.bytes() >= rows * columns,
        "output is smaller than rows × columns"
    );
    assert!(scales.bytes() >= rows * 4, "scales are smaller than rows");
    // SAFETY: every buffer covers the extent the kernel touches, checked above.
    unsafe {
        rotate_quantize_pointers(
            input.pointer(),
            false,
            output.pointer(),
            scales.pointer(),
            rows,
            columns,
        )
    }
}

/// Number of tile configurations the INT8 GEMM kernel is compiled for.
pub fn int8_config_count() -> usize {
    // SAFETY: no arguments.
    unsafe { mmh3_int8_gemm_config_count() as usize }
}

/// The heights of the bands of tiles the INT8 GEMM tries for a new shape.
pub fn int8_group_heights() -> Vec<usize> {
    // SAFETY: a capacity of 0 writes nothing.
    let count = unsafe { mmh3_int8_gemm_group_heights(ptr::null_mut(), 0) };
    let mut heights = vec![0; count as usize];
    // SAFETY: the vector holds `count` values.
    unsafe { mmh3_int8_gemm_group_heights(heights.as_mut_ptr(), count) };
    heights.into_iter().map(|height| height as usize).collect()
}

/// How the INT8 GEMM covers its output: the tile configuration, 256 × 128 for config 0 and
/// 128 × 256 for config 1, and how many rows of tiles it walks down before moving right. Every
/// tiling gives the same output bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Tiling {
    pub config: usize,
    pub group_m: usize,
}

/// A low-rank adapter added to an INT8 GEMM: `scale · down · upᵀ` with `down` `[m, rank]`, its rows
/// `down_stride` elements apart (a multiple of 8), and `up` `[n, rank]`, both BF16, and the rank a
/// multiple of 64.
pub struct Adapter<'a> {
    pub down: &'a DeviceBuffer,
    pub up: &'a DeviceBuffer,
    pub rank: usize,
    pub down_stride: usize,
    pub scale: f32,
}

/// Raw form of `Adapter`.
#[derive(Clone, Copy)]
pub(crate) struct AdapterPointers {
    pub(crate) down: *const c_void,
    pub(crate) up: *const c_void,
    pub(crate) rank: usize,
    pub(crate) down_stride: usize,
    pub(crate) scale: f32,
}

/// Where the INT8 GEMM writes its result: BF16 or FP16 `[m, n]`, after adding an optional f32 bias
/// `[n]`, or `silu(gate) · up` as `[m, n / 2]` with `swiglu`.
#[derive(Clone, Copy)]
pub(crate) struct Int8Output {
    pub(crate) pointer: *mut c_void,
    pub(crate) f16: bool,
    pub(crate) bias: *const c_void,
    pub(crate) swiglu: bool,
    /// Elements between the rows of the output, or zero for rows side by side. A shared-out step
    /// writes one rank's heads into a row that holds every rank's.
    pub(crate) stride: usize,
    /// Zeroed FP32 values that receive the maximum |output| of each 128 columns from
    /// `maxima_first_column` on, or null. BF16 output without `swiglu` only.
    pub(crate) column_maxima: *mut c_void,
    pub(crate) maxima_first_column: usize,
}

impl Int8Output {
    pub(crate) fn bf16(pointer: *mut c_void) -> Self {
        Int8Output {
            pointer,
            f16: false,
            bias: ptr::null(),
            swiglu: false,
            stride: 0,
            column_maxima: ptr::null_mut(),
            maxima_first_column: 0,
        }
    }
}

/// Merges a low-rank update into an INT8 ConvRot weight `[outputs, features]` with one scale per
/// row: every row becomes `weight · scale + up · down` and is quantized again with scale
/// max |w| / 127, rounding half to even. `up` is `[outputs, rank]` and `down` `[rank, features]`,
/// both FP32. `down` is in the layer's input space and is rotated in place like the activations.
pub fn merge_low_rank(
    weights: &mut DeviceBuffer,
    scales: &mut DeviceBuffer,
    up: &DeviceBuffer,
    down: &mut DeviceBuffer,
    outputs: usize,
    features: usize,
    rank: usize,
) -> Result<(), CudaError> {
    assert!(
        weights.bytes() >= outputs * features && scales.bytes() >= outputs * 4,
        "the weights are smaller than outputs × features"
    );
    assert!(
        up.bytes() >= outputs * rank * 4 && down.bytes() >= rank * features * 4,
        "the update is smaller than its rank"
    );
    // SAFETY: every buffer covers the extent the kernels touch, checked above.
    check(unsafe {
        mmh3_merge_low_rank(
            weights.pointer(),
            scales.pointer(),
            up.pointer(),
            down.pointer(),
            outputs as c_int,
            features as c_int,
            rank as c_int,
            ptr::null_mut(),
        )
    })
}

/// Returns the rows of a `[rows, row_bytes]` matrix whose first half are SwiGLU gates and second
/// half the matching up projections, reordered so that each group of eight rows holds the gates of
/// four features followed by their up projections. The INT8 GEMM's SwiGLU output needs the weights,
/// weight scales, bias and adapter up weights of the layer in this order.
pub fn interleave_swiglu_rows(
    source: &DeviceBuffer,
    rows: usize,
    row_bytes: usize,
) -> Result<DeviceBuffer, CudaError> {
    assert!(
        source.bytes() >= rows * row_bytes,
        "the matrix is smaller than rows × row_bytes"
    );
    let destination = DeviceBuffer::new(rows * row_bytes)?;
    // SAFETY: both buffers hold `rows × row_bytes` bytes, checked above.
    check(unsafe {
        mmh3_interleave_swiglu_rows(
            source.pointer(),
            destination.pointer(),
            rows as c_int,
            row_bytes as c_int,
            ptr::null_mut(),
        )
    })?;
    Ok(destination)
}

/// Raw form of `int8` with the tiling chosen for the shape, which the first call of a shape times.
///
/// # Safety
/// The pointers must cover the extents described for `int8`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn int8_pointers(
    activations: *const c_void,
    weights: *const c_void,
    activation_scales: *const c_void,
    weight_scales: *const c_void,
    output: Int8Output,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<AdapterPointers>,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    unsafe {
        int8_tiling(
            None,
            activations,
            weights,
            activation_scales,
            weight_scales,
            output,
            m,
            n,
            k,
            adapter,
        )
    }
}

/// Runs `tiling`, or the one chosen for the shape without it.
///
/// # Safety
/// The pointers must cover the extents described for `int8`.
#[allow(clippy::too_many_arguments)]
unsafe fn int8_tiling(
    tiling: Option<Int8Tiling>,
    activations: *const c_void,
    weights: *const c_void,
    activation_scales: *const c_void,
    weight_scales: *const c_void,
    output: Int8Output,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<AdapterPointers>,
) -> Result<(), CudaError> {
    let adapter = adapter.unwrap_or(AdapterPointers {
        down: ptr::null(),
        up: ptr::null(),
        rank: 0,
        down_stride: 0,
        scale: 0.0,
    });
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_int8_gemm(
            tiling.map_or(-1, |tiling| tiling.config as c_int),
            tiling.map_or(0, |tiling| tiling.group_m as c_int),
            activations,
            weights,
            activation_scales,
            weight_scales,
            output.bias,
            output.pointer,
            output.f16 as c_int,
            output.swiglu as c_int,
            m as c_int,
            n as c_int,
            k as c_int,
            output.stride as c_int,
            adapter.down,
            adapter.up,
            adapter.rank as c_int,
            adapter.down_stride as c_int,
            adapter.scale,
            output.column_maxima,
            output.maxima_first_column as c_int,
            ptr::null_mut(),
        )
    })
}

/// Where `int8` writes its result, `[m, n]` in BF16 or FP16, after adding an optional f32 bias
/// `[n]`, or `silu(gate) · up` as `[m, n / 2]` with `swiglu` for operands in the order of
/// `interleave_swiglu_rows`.
pub struct Output<'a> {
    pub buffer: &'a mut DeviceBuffer,
    pub f16: bool,
    pub bias: Option<&'a DeviceBuffer>,
    pub swiglu: bool,
    /// BF16 output without `swiglu` only.
    pub column_maxima: Option<ColumnMaxima<'a>>,
}

/// Zeroed FP32 values that receive the maximum |output| of each 128 columns from `first_column`, a
/// multiple of 128, on.
pub struct ColumnMaxima<'a> {
    pub values: &'a DeviceBuffer,
    pub first_column: usize,
}

/// output[m, n] = round(Σₖ activations[m, k] · weights[n, k] · activation_scales[m] ·
/// weight_scales[n] + the adapter + the bias).
///
/// Activations and weights are row-major INT8 with K contiguous, scales are f32. K must be a
/// multiple of 128 and N a multiple of the tile width, 128 for config 0 and 256 for config 1.
/// Without a tiling, it runs the one chosen for the shape.
#[allow(clippy::too_many_arguments)]
pub fn int8(
    tiling: Option<Int8Tiling>,
    activations: &DeviceBuffer,
    weights: &DeviceBuffer,
    activation_scales: &DeviceBuffer,
    weight_scales: &DeviceBuffer,
    output: Output,
    m: usize,
    n: usize,
    k: usize,
    adapter: Option<&Adapter>,
) -> Result<(), CudaError> {
    assert!(
        activations.bytes() >= m * k,
        "activations are smaller than m × k"
    );
    assert!(weights.bytes() >= n * k, "weights are smaller than n × k");
    assert!(
        activation_scales.bytes() >= m * 4,
        "activation scales are smaller than m"
    );
    assert!(
        weight_scales.bytes() >= n * 4,
        "weight scales are smaller than n"
    );
    assert!(
        output.buffer.bytes() >= m * n * if output.swiglu { 1 } else { 2 },
        "output is smaller than its m rows"
    );
    if let Some(bias) = output.bias {
        assert!(bias.bytes() >= n * 4, "bias is smaller than n");
    }
    if let Some(maxima) = &output.column_maxima {
        assert!(
            maxima.first_column <= n
                && maxima.values.bytes() >= (n - maxima.first_column) / 128 * 4,
            "column maxima are smaller than the groups of 128 columns from first_column"
        );
    }
    if let Some(adapter) = adapter {
        assert!(
            adapter.down_stride >= adapter.rank
                && adapter.down.bytes() >= m * adapter.down_stride * 2,
            "adapter down activations are smaller than m × down_stride"
        );
        assert!(
            adapter.up.bytes() >= n * adapter.rank * 2,
            "adapter up weights are smaller than n × rank"
        );
    }
    let adapter = adapter.map(|adapter| AdapterPointers {
        down: adapter.down.pointer(),
        up: adapter.up.pointer(),
        rank: adapter.rank,
        down_stride: adapter.down_stride,
        scale: adapter.scale,
    });
    let output = Int8Output {
        pointer: output.buffer.pointer(),
        f16: output.f16,
        bias: output
            .bias
            .map_or(ptr::null(), |bias| bias.pointer().cast_const()),
        swiglu: output.swiglu,
        stride: 0,
        column_maxima: output
            .column_maxima
            .as_ref()
            .map_or(ptr::null_mut(), |maxima| maxima.values.pointer()),
        maxima_first_column: output
            .column_maxima
            .as_ref()
            .map_or(0, |maxima| maxima.first_column),
    };
    // SAFETY: every buffer covers the extent the kernel touches, checked above.
    unsafe {
        int8_tiling(
            tiling,
            activations.pointer(),
            weights.pointer(),
            activation_scales.pointer(),
            weight_scales.pointer(),
            output,
            m,
            n,
            k,
            adapter,
        )
    }
}
