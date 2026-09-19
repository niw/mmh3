//! Attention over the DiT's fused qkv projection output.

use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::dit::sparse::{SPARSE_BLOCK, SparseSinks};
use mmh3_core::dit::vsa::{VSA_TILE, VsaPlan};
use std::ffi::{c_int, c_void};
use std::ptr;

pub use mmh3_core::shard::HEAD_DIM;

/// Strides in elements of the query, key, value and output tensors, in that order, and the
/// attention pattern.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionLayout {
    pub token_stride: [i64; 4],
    pub head_stride: [i64; 4],
    pub batch_stride: [i64; 4],
    /// Query heads that share one key and value head, where zero means one.
    pub heads_per_key_value: i32,
    /// Nonzero masks the keys after each query's own position.
    pub causal: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Bf16 = 0,
    F16 = 1,
}

/// Where query, key, value and output start, in elements from the start of their buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionOffsets {
    pub query: usize,
    pub key: usize,
    pub value: usize,
    pub output: usize,
}

/// Arithmetic used by the DiT's attention. Quantization changes generated details.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttentionPrecision {
    #[default]
    Bf16,
    /// INT8 QK, FP8 E4M3 PV, FP32 softmax and accumulation, BF16 output.
    Int8Fp8,
}

#[repr(C)]
struct RawQuantizedWorkspace {
    query: *mut c_void,
    key: *mut c_void,
    value: *mut c_void,
    query_scales: *mut c_void,
    key_scales: *mut c_void,
    value_scales: *mut c_void,
    value_maxima: *mut c_void,
}

/// Scratch for one sequence of INT8/FP8 attention. Reused across layers.
pub struct QuantizedWorkspace {
    tokens: usize,
    heads: usize,
    query: DeviceBuffer,
    key: DeviceBuffer,
    value: DeviceBuffer,
    query_scales: DeviceBuffer,
    key_scales: DeviceBuffer,
    value_scales: DeviceBuffer,
    value_maxima: DeviceBuffer,
}

impl QuantizedWorkspace {
    pub fn new(tokens: usize, heads: usize) -> Result<Self, CudaError> {
        Self::with_blocks(tokens, heads, tokens.div_ceil(SPARSE_BLOCK))
    }

    /// Scratch for key blocks of 64 columns each, such as VSA tiles.
    fn with_blocks(tokens: usize, heads: usize, blocks: usize) -> Result<Self, CudaError> {
        assert!(
            tokens > 0
                && tokens <= (i32::MAX - 63) as usize
                && heads > 0
                && heads <= u16::MAX as usize,
            "invalid attention shape"
        );
        let rows = tokens
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(HEAD_DIM))
            .expect("attention shape overflow");
        let scales = heads
            .checked_mul(blocks)
            .and_then(|n| n.checked_mul(4))
            .expect("attention scale shape overflow");
        Ok(Self {
            tokens,
            heads,
            query: DeviceBuffer::new(rows)?,
            key: DeviceBuffer::new(rows)?,
            value: DeviceBuffer::new(blocks * SPARSE_BLOCK * heads * HEAD_DIM)?,
            query_scales: DeviceBuffer::new(scales)?,
            key_scales: DeviceBuffer::new(scales)?,
            value_scales: DeviceBuffer::new(heads * 4)?,
            value_maxima: DeviceBuffer::new(scales)?,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    fn raw(&self) -> RawQuantizedWorkspace {
        RawQuantizedWorkspace {
            query: self.query.pointer(),
            key: self.key.pointer(),
            value: self.value.pointer(),
            query_scales: self.query_scales.pointer(),
            key_scales: self.key_scales.pointer(),
            value_scales: self.value_scales.pointer(),
            value_maxima: self.value_maxima.pointer(),
        }
    }
}

/// Device pointers of the Sol-Attn scratch buffers, see kernels/sparse_attention.cu.
#[repr(C)]
struct RawSparseWorkspace {
    centroids: *mut c_void,
    block_keys: *mut c_void,
    value_sums: *mut c_void,
    key_mean: *mut c_void,
    key_variance: *mut c_void,
    row_offsets: *mut c_void,
    routes: *mut c_void,
    route_counts: *mut c_void,
    tail_max: *mut c_void,
    tail_sum: *mut c_void,
    tail_values: *mut c_void,
}

unsafe extern "C" {
    fn mmh3_attention_quantized(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        layout: *const AttentionLayout,
        scale: f32,
        sparse: *const RawSparseWorkspace,
        workspace: *const RawQuantizedWorkspace,
        inputs_ready: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_qk_norm_rope(
        qkv: *mut c_void,
        query_weight: *const c_void,
        key_weight: *const c_void,
        angles: *const c_void,
        pairs: c_int,
        tokens: c_int,
        query_heads: c_int,
        key_heads: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn mmh3_attention_inputs(
        qkv: *mut c_void,
        query_weight: *const c_void,
        key_weight: *const c_void,
        angles: *const c_void,
        pairs: c_int,
        tokens: c_int,
        heads: c_int,
        epsilon: f32,
        sparse: *const RawSparseWorkspace,
        quantized: *const RawQuantizedWorkspace,
        tile_starts: *const c_void,
        tile_lengths: *const c_void,
        tiles: c_int,
        stream: *mut c_void,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn mmh3_sparse_attention(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        layout: *const AttentionLayout,
        scale: f32,
        tau: f32,
        sink_key_start: c_int,
        sink_key_end: c_int,
        sink_query_start: c_int,
        sink_query_end: c_int,
        workspace: *const RawSparseWorkspace,
        quantized: *const RawQuantizedWorkspace,
        inputs_ready: c_int,
        stream: *mut c_void,
    ) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn mmh3_vsa_attention(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        gate: *const c_void,
        gate_stride: i64,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        layout: *const AttentionLayout,
        scale: f32,
        tiles: c_int,
        prefix_tiles: c_int,
        kept: c_int,
        workspace: *const RawVsaWorkspace,
        quantized: *const RawQuantizedWorkspace,
        inputs_ready: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_attention(
        element_type: c_int,
        head_dim: c_int,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        batch: c_int,
        layout: *const AttentionLayout,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_attention_roped(
        element_type: c_int,
        head_dim: c_int,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        batch: c_int,
        layout: *const AttentionLayout,
        scale: f32,
        angles: *const c_void,
        pairs: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_attention_bf16(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        query_stride: i64,
        key_stride: i64,
        value_stride: i64,
        output_stride: i64,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;
}

/// Raw form of `attention` for pointers the caller has already checked.
///
/// # Safety
/// Every element the layout addresses for `batch`, `tokens` and `heads` must lie inside the
/// pointed-to buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention_pointers(
    element: Element,
    head_dim: usize,
    query: *const c_void,
    key: *const c_void,
    value: *const c_void,
    output: *mut c_void,
    tokens: usize,
    heads: usize,
    batch: usize,
    layout: &AttentionLayout,
    scale: f32,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_attention(
            element as c_int,
            head_dim as c_int,
            query,
            key,
            value,
            output,
            tokens as c_int,
            heads as c_int,
            batch as c_int,
            layout,
            scale,
            ptr::null_mut(),
        )
    })
}

/// `attention_pointers` with the normalization and the rotation that the video VAE's decoder puts
/// between its projection and its attention: as each tile of q and k reaches shared memory, its
/// rows take an RMS norm over the head dimension and a RoPE from `angles`, [tokens, pairs].
///
/// # Safety
/// As `attention_pointers`, and `angles` must hold `tokens × pairs` floats.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn attention_roped_pointers(
    element: Element,
    head_dim: usize,
    query: *const c_void,
    key: *const c_void,
    value: *const c_void,
    output: *mut c_void,
    tokens: usize,
    heads: usize,
    batch: usize,
    layout: &AttentionLayout,
    scale: f32,
    angles: *const c_void,
    pairs: usize,
    epsilon: f32,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_attention_roped(
            element as c_int,
            head_dim as c_int,
            query,
            key,
            value,
            output,
            tokens as c_int,
            heads as c_int,
            batch as c_int,
            layout,
            scale,
            angles,
            pairs as c_int,
            epsilon,
            ptr::null_mut(),
        )
    })
}

/// Softmax attention for head dimension 64 or 128 over tensors described by `layout`, reading
/// query, key and value from `input` and writing `output`, each starting at its offset. Query head
/// h reads key and value head h / heads_per_key_value.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    element: Element,
    head_dim: usize,
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    offsets: AttentionOffsets,
    tokens: usize,
    heads: usize,
    batch: usize,
    layout: &AttentionLayout,
    scale: f32,
) -> Result<(), CudaError> {
    let group = layout.heads_per_key_value.max(1) as usize;
    assert_eq!(
        heads % group,
        0,
        "heads must be a multiple of the heads per key and value head"
    );
    let last = |offset: usize, operand: usize| {
        let operand_heads = if operand == 1 || operand == 2 {
            heads / group
        } else {
            heads
        };
        offset as i64
            + (batch as i64 - 1) * layout.batch_stride[operand]
            + (tokens as i64 - 1) * layout.token_stride[operand]
            + (operand_heads as i64 - 1) * layout.head_stride[operand]
            + head_dim as i64
    };
    let input_elements = (input.bytes() / 2) as i64;
    for (operand, offset) in [offsets.query, offsets.key, offsets.value]
        .into_iter()
        .enumerate()
    {
        assert!(
            last(offset, operand) <= input_elements,
            "operand {operand} reaches past the input buffer"
        );
    }
    assert!(
        last(offsets.output, 3) <= (output.bytes() / 2) as i64,
        "the output reaches past its buffer"
    );
    // SAFETY: every addressed element lies inside the buffers, checked above.
    unsafe {
        attention_pointers(
            element,
            head_dim,
            input.pointer_at(offsets.query * 2),
            input.pointer_at(offsets.key * 2),
            input.pointer_at(offsets.value * 2),
            output.pointer_at(offsets.output * 2),
            tokens,
            heads,
            batch,
            layout,
            scale,
        )
    }
}

/// Raw form of `dense_bf16` for buffers the caller has already checked.
///
/// # Safety
/// `qkv` must hold `tokens × 3 × heads × 128` BF16 values and `output` `tokens × heads × 128`.
pub(crate) unsafe fn dense_bf16_pointers(
    qkv: *const c_void,
    output: *mut c_void,
    tokens: usize,
    heads: usize,
    scale: f32,
) -> Result<(), CudaError> {
    let inner = heads * HEAD_DIM;
    let base = qkv.cast::<u16>();
    // SAFETY: q, k and v are the three interleaved thirds of qkv, as the caller guarantees.
    check(unsafe {
        mmh3_attention_bf16(
            base.cast(),
            base.add(inner).cast(),
            base.add(2 * inner).cast(),
            output,
            tokens as c_int,
            heads as c_int,
            (3 * inner) as i64,
            (3 * inner) as i64,
            (3 * inner) as i64,
            inner as i64,
            scale,
            ptr::null_mut(),
        )
    })
}

/// Dense bidirectional attention with head dimension 128.
///
/// `qkv` is BF16 `[tokens, 3, heads, 128]` as produced by the fused qkv projection, and `output`
/// receives BF16 `[tokens, heads, 128]`.
pub fn dense_bf16(
    qkv: &DeviceBuffer,
    output: &mut DeviceBuffer,
    tokens: usize,
    heads: usize,
    scale: f32,
) -> Result<(), CudaError> {
    let inner = heads * HEAD_DIM;
    assert!(
        qkv.bytes() >= tokens * 3 * inner * 2,
        "qkv is smaller than tokens × 3 × heads × 128"
    );
    assert!(
        output.bytes() >= tokens * inner * 2,
        "output is smaller than tokens × heads × 128"
    );
    // SAFETY: both buffers cover every token, checked above.
    unsafe { dense_bf16_pointers(qkv.pointer(), output.pointer(), tokens, heads, scale) }
}

/// Where the attention finds its per-block inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionInputs {
    /// The attention computes them from its BF16 q, k and v.
    Raw,
    /// `prepare_inputs` has left them in the workspace.
    Prepared,
}

impl AttentionInputs {
    fn ready(self) -> c_int {
        c_int::from(self == AttentionInputs::Prepared)
    }
}

/// Per-head RMSNorm weights of q and k, and the rotary angles of every token, of a DiT block.
pub struct HeadNorm<'a> {
    pub query_weight: &'a DeviceBuffer,
    pub key_weight: &'a DeviceBuffer,
    /// FP32 `[tokens, pairs]`, or None for no rotary embedding.
    pub angles: Option<&'a DeviceBuffer>,
    /// Rotated dimension pairs, which split the first `2 × pairs` dimensions of a head in halves.
    pub pairs: usize,
    pub epsilon: f32,
}

impl HeadNorm<'_> {
    fn check(&self, qkv: &DeviceBuffer, tokens: usize, heads: usize) {
        assert!(
            qkv.bytes() >= tokens * 3 * heads * HEAD_DIM * 2,
            "qkv is smaller than tokens × 3 × heads × 128"
        );
        assert!(
            self.query_weight.bytes() >= HEAD_DIM * 2 && self.key_weight.bytes() >= HEAD_DIM * 2,
            "the norm weights hold 128 BF16 values"
        );
        assert!(
            2 * self.pairs <= HEAD_DIM,
            "the rotary pairs exceed the head"
        );
        if let Some(angles) = self.angles {
            assert!(
                angles.bytes() >= tokens * self.pairs * 4,
                "the angles do not cover every token"
            );
        }
    }

    fn angles(&self) -> *const c_void {
        self.angles
            .map_or(ptr::null(), |angles| angles.pointer().cast_const())
    }
}

/// Per-head RMSNorm of q and k in the DiT's BF16 `[tokens, 3, heads, 128]` qkv rows, then the
/// split-half rotary embedding of their first `2 × pairs` dimensions.
pub fn qk_norm_rope(
    qkv: &mut DeviceBuffer,
    norm: &HeadNorm,
    tokens: usize,
    heads: usize,
) -> Result<(), CudaError> {
    norm.check(qkv, tokens, heads);
    // SAFETY: the buffers cover every token and head, checked above.
    check(unsafe {
        mmh3_qk_norm_rope(
            qkv.pointer(),
            norm.query_weight.pointer(),
            norm.key_weight.pointer(),
            norm.angles(),
            norm.pairs as c_int,
            tokens as c_int,
            heads as c_int,
            heads as c_int,
            norm.epsilon,
            ptr::null_mut(),
        )
    })
}

/// The attention that `prepare_inputs` prepares.
#[derive(Clone, Copy)]
pub enum PreparedAttention<'a> {
    DenseQuantized(&'a QuantizedWorkspace),
    Sparse(&'a SparseWorkspace),
    Vsa(&'a VsaWorkspace),
}

/// `qk_norm_rope` fused with the per-block work of the attention that follows, which then runs with
/// `AttentionInputs::Prepared`: Sol-Attn's block statistics, VSA's pooled tiles, and INT8 q and k
/// with the block maxima of |v| for quantized attention. Normalized k stays unwritten in qkv when the attention reads
/// only the INT8 keys, and so does normalized q for dense quantized attention.
pub fn prepare_inputs(
    qkv: &mut DeviceBuffer,
    norm: &HeadNorm,
    tokens: usize,
    heads: usize,
    attention: PreparedAttention,
) -> Result<(), CudaError> {
    norm.check(qkv, tokens, heads);
    // SAFETY: the buffers cover every token and head, checked above.
    unsafe {
        prepare_inputs_pointers(
            qkv.pointer(),
            norm.query_weight.pointer(),
            norm.key_weight.pointer(),
            norm.angles(),
            norm.pairs,
            tokens,
            heads,
            norm.epsilon,
            attention,
        )
    }
}

/// Raw form of `prepare_inputs`.
///
/// # Safety
/// `qkv` must hold `tokens × 3 × heads × 128` BF16 values, the weights 128 BF16 values each and
/// `angles`, unless null, `tokens × pairs` FP32 values.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn prepare_inputs_pointers(
    qkv: *mut c_void,
    query_weight: *const c_void,
    key_weight: *const c_void,
    angles: *const c_void,
    pairs: usize,
    tokens: usize,
    heads: usize,
    epsilon: f32,
    attention: PreparedAttention,
) -> Result<(), CudaError> {
    let mut tiles = (ptr::null(), ptr::null(), 0);
    let (sparse, quantized) = match attention {
        PreparedAttention::DenseQuantized(workspace) => {
            assert!(
                workspace.tokens == tokens && workspace.heads == heads,
                "the quantized workspace has another shape"
            );
            (None, Some(workspace.raw()))
        }
        PreparedAttention::Sparse(workspace) => {
            assert!(
                workspace.tokens == tokens && workspace.heads == heads,
                "the sparse workspace has another shape"
            );
            (
                Some(workspace.raw()),
                workspace.quantized.as_ref().map(QuantizedWorkspace::raw),
            )
        }
        PreparedAttention::Vsa(workspace) => {
            assert!(
                workspace.tokens == tokens && workspace.heads == heads,
                "the VSA workspace has another shape"
            );
            tiles = (
                workspace.tile_starts.pointer().cast_const(),
                workspace.tile_lengths.pointer().cast_const(),
                workspace.tiles as c_int,
            );
            // The pooled tiles take the places of the block statistics.
            let pooled = RawSparseWorkspace {
                centroids: workspace.pooled_query.pointer(),
                block_keys: workspace.pooled_key.pointer(),
                value_sums: workspace.pooled_value.pointer(),
                key_mean: ptr::null_mut(),
                key_variance: ptr::null_mut(),
                row_offsets: ptr::null_mut(),
                routes: ptr::null_mut(),
                route_counts: ptr::null_mut(),
                tail_max: ptr::null_mut(),
                tail_sum: ptr::null_mut(),
                tail_values: ptr::null_mut(),
            };
            (
                Some(pooled),
                workspace.quantized.as_ref().map(QuantizedWorkspace::raw),
            )
        }
    };
    // SAFETY: the caller guarantees the extents, and the workspaces match the shape, checked above.
    check(unsafe {
        mmh3_attention_inputs(
            qkv,
            query_weight,
            key_weight,
            angles,
            pairs as c_int,
            tokens as c_int,
            heads as c_int,
            epsilon,
            sparse.as_ref().map_or(ptr::null(), |raw| raw),
            quantized.as_ref().map_or(ptr::null(), |raw| raw),
            tiles.0,
            tiles.1,
            tiles.2,
            ptr::null_mut(),
        )
    })
}

/// Dense bidirectional attention over the DiT's BF16 qkv layout, with INT8 QK and FP8 PV.
/// This approximation retains FP32 softmax/accumulation and writes BF16 output.
pub fn dense_quantized(
    qkv: &DeviceBuffer,
    output: &mut DeviceBuffer,
    scale: f32,
    workspace: &QuantizedWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    let elements = workspace.tokens * workspace.heads * HEAD_DIM;
    assert!(
        qkv.bytes() >= elements * 6 && output.bytes() >= elements * 2,
        "attention buffers are too small"
    );
    // SAFETY: both buffers cover the workspace's sequence, checked above.
    unsafe { dense_quantized_pointers(qkv.pointer(), output.pointer(), scale, workspace, inputs) }
}

/// # Safety
/// qkv and output must cover the workspace's sequence in the fused BF16 qkv and contiguous output
/// layouts.
pub(crate) unsafe fn dense_quantized_pointers(
    qkv: *const c_void,
    output: *mut c_void,
    scale: f32,
    workspace: &QuantizedWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    let inner = workspace.heads * HEAD_DIM;
    let layout = AttentionLayout {
        token_stride: [
            (3 * inner) as i64,
            (3 * inner) as i64,
            (3 * inner) as i64,
            inner as i64,
        ],
        head_stride: [HEAD_DIM as i64; 4],
        ..AttentionLayout::default()
    };
    let raw = workspace.raw();
    let base = qkv.cast::<u16>();
    // SAFETY: the caller guarantees the buffer extents and the workspace owns all quantization
    // scratch.
    check(unsafe {
        mmh3_attention_quantized(
            base.cast(),
            base.add(inner).cast(),
            base.add(2 * inner).cast(),
            output,
            workspace.tokens as c_int,
            workspace.heads as c_int,
            &layout,
            scale,
            ptr::null(),
            &raw,
            inputs.ready(),
            ptr::null_mut(),
        )
    })
}

/// Scratch buffers of Sol-Attn for one sequence length and head count of 128-wide heads.
pub struct SparseWorkspace {
    quantized: Option<QuantizedWorkspace>,
    tokens: usize,
    heads: usize,
    blocks: usize,
    centroids: DeviceBuffer,
    block_keys: DeviceBuffer,
    value_sums: DeviceBuffer,
    key_mean: DeviceBuffer,
    key_variance: DeviceBuffer,
    row_offsets: DeviceBuffer,
    routes: DeviceBuffer,
    route_counts: DeviceBuffer,
    tail_max: DeviceBuffer,
    tail_sum: DeviceBuffer,
    tail_values: DeviceBuffer,
}

impl SparseWorkspace {
    pub fn new(tokens: usize, heads: usize) -> Result<Self, CudaError> {
        Self::with_precision(tokens, heads, AttentionPrecision::Bf16)
    }

    /// Keeps BF16 routing and the FP32 pooled tail. Quantizes the routed products when requested.
    pub fn with_precision(
        tokens: usize,
        heads: usize,
        precision: AttentionPrecision,
    ) -> Result<Self, CudaError> {
        let blocks = tokens.div_ceil(SPARSE_BLOCK);
        let per_block = heads * blocks * HEAD_DIM * 4;
        Ok(SparseWorkspace {
            quantized: match precision {
                AttentionPrecision::Bf16 => None,
                AttentionPrecision::Int8Fp8 => Some(QuantizedWorkspace::new(tokens, heads)?),
            },
            tokens,
            heads,
            blocks,
            centroids: DeviceBuffer::new(per_block)?,
            block_keys: DeviceBuffer::new(per_block)?,
            value_sums: DeviceBuffer::new(per_block)?,
            key_mean: DeviceBuffer::new(heads * HEAD_DIM * 4)?,
            key_variance: DeviceBuffer::new(heads * HEAD_DIM * 4)?,
            row_offsets: DeviceBuffer::new(heads * tokens * 4)?,
            routes: DeviceBuffer::new(heads * blocks * blocks * 2)?,
            route_counts: DeviceBuffer::new(heads * blocks * 4)?,
            tail_max: DeviceBuffer::new(heads * blocks * 4)?,
            tail_sum: DeviceBuffer::new(heads * blocks * 4)?,
            tail_values: DeviceBuffer::new(per_block)?,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    fn raw(&self) -> RawSparseWorkspace {
        RawSparseWorkspace {
            centroids: self.centroids.pointer(),
            block_keys: self.block_keys.pointer(),
            value_sums: self.value_sums.pointer(),
            key_mean: self.key_mean.pointer(),
            key_variance: self.key_variance.pointer(),
            row_offsets: self.row_offsets.pointer(),
            routes: self.routes.pointer(),
            route_counts: self.route_counts.pointer(),
            tail_max: self.tail_max.pointer(),
            tail_sum: self.tail_sum.pointer(),
            tail_values: self.tail_values.pointer(),
        }
    }

    /// Fraction of (head, query block, key block) triples the last call routed token by token.
    pub fn routed_fraction(&self) -> Result<f64, CudaError> {
        let mut bytes = vec![0u8; self.route_counts.bytes()];
        self.route_counts.copy_to_host(&mut bytes)?;
        let routed: u64 = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&count| i32::from_le_bytes(count) as u64)
            .sum();
        Ok(routed as f64 / (self.heads * self.blocks * self.blocks) as f64)
    }
}

/// Raw form of `sparse` for pointers the caller has already checked.
///
/// # Safety
/// Every element the layout addresses for `tokens` and `heads` must lie inside the pointed-to
/// buffers, and the workspace must be sized for `tokens` and `heads`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sparse_pointers(
    query: *const c_void,
    key: *const c_void,
    value: *const c_void,
    output: *mut c_void,
    tokens: usize,
    heads: usize,
    layout: &AttentionLayout,
    scale: f32,
    tau: f32,
    sinks: SparseSinks,
    workspace: &SparseWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    assert!(
        workspace.tokens == tokens && workspace.heads == heads,
        "the sparse workspace has another shape"
    );
    let raw = workspace.raw();
    let quantized = workspace.quantized.as_ref().map(QuantizedWorkspace::raw);
    // SAFETY: the caller guarantees the extents, and the workspace matches the shape, checked
    // above.
    check(unsafe {
        mmh3_sparse_attention(
            query,
            key,
            value,
            output,
            tokens as c_int,
            heads as c_int,
            layout,
            scale,
            tau,
            sinks.key_blocks.0 as c_int,
            sinks.key_blocks.1 as c_int,
            sinks.query_blocks.0 as c_int,
            sinks.query_blocks.1 as c_int,
            &raw,
            quantized.as_ref().map_or(ptr::null(), |raw| raw),
            inputs.ready(),
            ptr::null_mut(),
        )
    })
}

/// Device pointers of the VSA scratch buffers, see kernels/vsa_attention.cu.
#[repr(C)]
struct RawVsaWorkspace {
    tile_starts: *const c_void,
    tile_lengths: *const c_void,
    pooled_query: *mut c_void,
    pooled_key: *mut c_void,
    pooled_value: *mut c_void,
    routes: *mut c_void,
    route_counts: *mut c_void,
    coarse: *mut c_void,
}

/// The tiles of one VSA plan and scratch buffers for its sequence and head count.
pub struct VsaWorkspace {
    quantized: Option<QuantizedWorkspace>,
    tokens: usize,
    heads: usize,
    tiles: usize,
    prefix_tiles: usize,
    tile_starts: DeviceBuffer,
    tile_lengths: DeviceBuffer,
    pooled_query: DeviceBuffer,
    pooled_key: DeviceBuffer,
    pooled_value: DeviceBuffer,
    routes: DeviceBuffer,
    route_counts: DeviceBuffer,
    coarse: DeviceBuffer,
}

/// What two neighbouring query tiles select, as a block covering both would see it.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteOverlap {
    /// Neighbouring pairs counted, over every head.
    pub pairs: usize,
    /// Mean tiles the two blocks load between them today.
    pub separate: f64,
    /// Mean distinct tiles, which is what one block covering both rows would load.
    pub merged: f64,
}

impl VsaWorkspace {
    /// Downloads the selections and measures how much neighbouring query tiles share, which is the
    /// key and value traffic a block over two query tiles would save. Diagnostic, not a hot path.
    pub fn route_overlap(&self) -> Result<RouteOverlap, CudaError> {
        let tiles = self.tiles;
        let mut counts = vec![0u8; self.route_counts.bytes()];
        self.route_counts.copy_to_host(&mut counts)?;
        let mut routes = vec![0u8; self.routes.bytes()];
        self.routes.copy_to_host(&mut routes)?;
        let count = |head: usize, tile: usize| -> usize {
            let offset = (head * tiles + tile) * 4;
            i32::from_le_bytes(counts[offset..offset + 4].try_into().unwrap()).max(0) as usize
        };
        let route = |head: usize, tile: usize, entry: usize| -> usize {
            let offset = ((head * tiles + tile) * tiles + entry) * 2;
            u16::from_le_bytes(routes[offset..offset + 2].try_into().unwrap()) as usize
        };
        let mut overlap = RouteOverlap::default();
        let mut selected = vec![false; tiles];
        for head in 0..self.heads {
            for tile in (0..tiles.saturating_sub(1)).step_by(2) {
                let (first, second) = (count(head, tile), count(head, tile + 1));
                if first == 0 && second == 0 {
                    continue;
                }
                selected.iter_mut().for_each(|value| *value = false);
                let mut distinct = 0;
                for (other, length) in [(tile, first), (tile + 1, second)] {
                    for entry in 0..length {
                        let target = route(head, other, entry);
                        if target < tiles && !selected[target] {
                            selected[target] = true;
                            distinct += 1;
                        }
                    }
                }
                overlap.pairs += 1;
                overlap.separate += (first + second) as f64;
                overlap.merged += distinct as f64;
            }
        }
        if overlap.pairs > 0 {
            overlap.separate /= overlap.pairs as f64;
            overlap.merged /= overlap.pairs as f64;
        }
        Ok(overlap)
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn new(plan: &VsaPlan, tokens: usize, heads: usize) -> Result<Self, CudaError> {
        Self::with_precision(plan, tokens, heads, AttentionPrecision::Bf16)
    }

    /// Quantizes the token attention when requested, which then needs prepared inputs.
    pub fn with_precision(
        plan: &VsaPlan,
        tokens: usize,
        heads: usize,
        precision: AttentionPrecision,
    ) -> Result<Self, CudaError> {
        let tiles = plan.tiles();
        assert!(
            tiles > 0 && tiles <= u16::MAX as usize + 1,
            "invalid tile count {tiles}"
        );
        assert!(
            plan.tile_lengths
                .iter()
                .all(|&length| (1..=VSA_TILE).contains(&length))
                && plan
                    .tile_starts
                    .iter()
                    .zip(&plan.tile_lengths)
                    .all(|(&start, &length)| start + length <= tokens),
            "the plan's tiles do not fit the sequence"
        );
        let int32 = |values: &[usize]| {
            DeviceBuffer::from_bytes(
                &values
                    .iter()
                    .flat_map(|&value| (value as i32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        let per_tile = heads * tiles * HEAD_DIM * 4;
        Ok(VsaWorkspace {
            quantized: match precision {
                AttentionPrecision::Bf16 => None,
                AttentionPrecision::Int8Fp8 => {
                    Some(QuantizedWorkspace::with_blocks(tokens, heads, tiles)?)
                }
            },
            tokens,
            heads,
            tiles,
            prefix_tiles: plan.prefix_tiles,
            tile_starts: int32(&plan.tile_starts)?,
            tile_lengths: int32(&plan.tile_lengths)?,
            pooled_query: DeviceBuffer::new(per_tile)?,
            pooled_key: DeviceBuffer::new(per_tile)?,
            pooled_value: DeviceBuffer::new(per_tile)?,
            routes: DeviceBuffer::new(heads * tiles * tiles * 2)?,
            route_counts: DeviceBuffer::new(heads * tiles * 4)?,
            coarse: DeviceBuffer::new(per_tile)?,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    fn raw(&self) -> RawVsaWorkspace {
        RawVsaWorkspace {
            tile_starts: self.tile_starts.pointer().cast_const(),
            tile_lengths: self.tile_lengths.pointer().cast_const(),
            pooled_query: self.pooled_query.pointer(),
            pooled_key: self.pooled_key.pointer(),
            pooled_value: self.pooled_value.pointer(),
            routes: self.routes.pointer(),
            route_counts: self.route_counts.pointer(),
            coarse: self.coarse.pointer(),
        }
    }

    /// Mean fraction of the tiles the query tiles of the last call attended token by token.
    pub fn selected_fraction(&self) -> Result<f64, CudaError> {
        let mut bytes = vec![0u8; self.route_counts.bytes()];
        self.route_counts.copy_to_host(&mut bytes)?;
        let selected: u64 = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&count| i32::from_le_bytes(count) as u64)
            .sum();
        Ok(selected as f64 / (self.heads * self.tiles * self.tiles) as f64)
    }
}

/// Raw form of `vsa` for pointers the caller has already checked.
///
/// # Safety
/// Every element the layout addresses for the workspace's tokens and heads must lie inside the
/// pointed-to buffers, and a non-null `gate` must hold `tokens` rows of `heads × 128` BF16 values
/// `gate_stride` elements apart.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn vsa_pointers(
    query: *const c_void,
    key: *const c_void,
    value: *const c_void,
    gate: *const c_void,
    gate_stride: usize,
    output: *mut c_void,
    layout: &AttentionLayout,
    scale: f32,
    kept: usize,
    workspace: &VsaWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    assert!(
        workspace.quantized.is_none() || inputs == AttentionInputs::Prepared,
        "quantized VSA needs prepared inputs"
    );
    let raw = workspace.raw();
    let quantized = workspace.quantized.as_ref().map(QuantizedWorkspace::raw);
    // SAFETY: the caller guarantees the extents, and the workspace's tiles lie inside its sequence.
    check(unsafe {
        mmh3_vsa_attention(
            query,
            key,
            value,
            gate,
            gate_stride as i64,
            output,
            workspace.tokens as c_int,
            workspace.heads as c_int,
            layout,
            scale,
            workspace.tiles as c_int,
            workspace.prefix_tiles as c_int,
            kept as c_int,
            &raw,
            quantized.as_ref().map_or(ptr::null(), |raw| raw),
            inputs.ready(),
            ptr::null_mut(),
        )
    })
}

/// VSA over BF16 heads of 128 for one sequence in the workspace's tile order, reading query, key and
/// value from `input` and writing `output` at their offsets with the token and head strides of
/// `layout`. Video query tiles keep `kept` video tiles, and `gate`, BF16 `[tokens, heads, 128]`,
/// adds the coarse branch.
#[allow(clippy::too_many_arguments)]
pub fn vsa(
    input: &DeviceBuffer,
    gate: Option<&DeviceBuffer>,
    output: &mut DeviceBuffer,
    offsets: AttentionOffsets,
    layout: &AttentionLayout,
    scale: f32,
    kept: usize,
    workspace: &VsaWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    let (tokens, heads) = (workspace.tokens, workspace.heads);
    let last = |offset: usize, operand: usize| {
        offset as i64
            + (tokens as i64 - 1) * layout.token_stride[operand]
            + (heads as i64 - 1) * layout.head_stride[operand]
            + HEAD_DIM as i64
    };
    for (operand, offset) in [offsets.query, offsets.key, offsets.value]
        .into_iter()
        .enumerate()
    {
        assert!(
            last(offset, operand) <= (input.bytes() / 2) as i64,
            "operand {operand} reaches past the input buffer"
        );
    }
    assert!(
        last(offsets.output, 3) <= (output.bytes() / 2) as i64,
        "the output reaches past its buffer"
    );
    if let Some(gate) = gate {
        assert!(
            gate.bytes() >= tokens * heads * HEAD_DIM * 2,
            "the gate is smaller than tokens × heads × 128"
        );
    }
    // SAFETY: every addressed element lies inside the buffers, checked above.
    unsafe {
        vsa_pointers(
            input.pointer_at(offsets.query * 2),
            input.pointer_at(offsets.key * 2),
            input.pointer_at(offsets.value * 2),
            gate.map_or(ptr::null(), |gate| gate.pointer().cast_const()),
            heads * HEAD_DIM,
            output.pointer_at(offsets.output * 2),
            layout,
            scale,
            kept,
            workspace,
            inputs,
        )
    }
}

/// Sol-Attn over BF16 heads of 128 for one sequence, reading query, key and value from `input` and
/// writing `output` at their offsets with the token and head strides of `layout`.
#[allow(clippy::too_many_arguments)]
pub fn sparse(
    input: &DeviceBuffer,
    output: &mut DeviceBuffer,
    offsets: AttentionOffsets,
    tokens: usize,
    heads: usize,
    layout: &AttentionLayout,
    scale: f32,
    tau: f32,
    sinks: SparseSinks,
    workspace: &SparseWorkspace,
    inputs: AttentionInputs,
) -> Result<(), CudaError> {
    let last = |offset: usize, operand: usize| {
        offset as i64
            + (tokens as i64 - 1) * layout.token_stride[operand]
            + (heads as i64 - 1) * layout.head_stride[operand]
            + HEAD_DIM as i64
    };
    for (operand, offset) in [offsets.query, offsets.key, offsets.value]
        .into_iter()
        .enumerate()
    {
        assert!(
            last(offset, operand) <= (input.bytes() / 2) as i64,
            "operand {operand} reaches past the input buffer"
        );
    }
    assert!(
        last(offsets.output, 3) <= (output.bytes() / 2) as i64,
        "the output reaches past its buffer"
    );
    // SAFETY: every addressed element lies inside the buffers, checked above.
    unsafe {
        sparse_pointers(
            input.pointer_at(offsets.query * 2),
            input.pointer_at(offsets.key * 2),
            input.pointer_at(offsets.value * 2),
            output.pointer_at(offsets.output * 2),
            tokens,
            heads,
            layout,
            scale,
            tau,
            sinks,
            workspace,
            inputs,
        )
    }
}
