//! Attention over the DiT's fused qkv projection output.

use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::dit::sparse::{SPARSE_BLOCK, SparseSinks};
use std::ffi::{c_int, c_void};
use std::ptr;

pub const HEAD_DIM: usize = 128;

/// Strides in elements of the query, key, value and output tensors, in that order, and the attention pattern.
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
/// Every element the layout addresses for `batch`, `tokens` and `heads` must lie inside the pointed-to buffers.
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

/// Softmax attention for head dimension 64 or 128 over tensors described by `layout`, reading query, key and value
/// from `input` and writing `output`, each starting at its offset. Query head h reads key and value head
/// h / heads_per_key_value.
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
    assert_eq!(heads % group, 0, "heads must be a multiple of the heads per key and value head");
    let last = |offset: usize, operand: usize| {
        let operand_heads = if operand == 1 || operand == 2 { heads / group } else { heads };
        offset as i64
            + (batch as i64 - 1) * layout.batch_stride[operand]
            + (tokens as i64 - 1) * layout.token_stride[operand]
            + (operand_heads as i64 - 1) * layout.head_stride[operand]
            + head_dim as i64
    };
    let input_elements = (input.bytes() / 2) as i64;
    for (operand, offset) in [offsets.query, offsets.key, offsets.value].into_iter().enumerate() {
        assert!(last(offset, operand) <= input_elements, "operand {operand} reaches past the input buffer");
    }
    assert!(last(offsets.output, 3) <= (output.bytes() / 2) as i64, "the output reaches past its buffer");
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
/// `qkv` is BF16 `[tokens, 3, heads, 128]` as produced by the fused qkv projection, and `output` receives BF16
/// `[tokens, heads, 128]`.
pub fn dense_bf16(
    qkv: &DeviceBuffer,
    output: &mut DeviceBuffer,
    tokens: usize,
    heads: usize,
    scale: f32,
) -> Result<(), CudaError> {
    let inner = heads * HEAD_DIM;
    assert!(qkv.bytes() >= tokens * 3 * inner * 2, "qkv is smaller than tokens × 3 × heads × 128");
    assert!(output.bytes() >= tokens * inner * 2, "output is smaller than tokens × heads × 128");
    // SAFETY: both buffers cover every token, checked above.
    unsafe { dense_bf16_pointers(qkv.pointer(), output.pointer(), tokens, heads, scale) }
}

/// Scratch buffers of Sol-Attn for one sequence length and head count of 128-wide heads.
pub struct SparseWorkspace {
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
        let blocks = tokens.div_ceil(SPARSE_BLOCK);
        let per_block = heads * blocks * HEAD_DIM * 4;
        Ok(SparseWorkspace {
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
        let routed: u64 = bytes.chunks_exact(4).map(|count| i32::from_le_bytes(count.try_into().unwrap()) as u64).sum();
        Ok(routed as f64 / (self.heads * self.blocks * self.blocks) as f64)
    }
}

/// Raw form of `sparse` for pointers the caller has already checked.
///
/// # Safety
/// Every element the layout addresses for `tokens` and `heads` must lie inside the pointed-to buffers, and the
/// workspace must be sized for `tokens` and `heads`.
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
) -> Result<(), CudaError> {
    assert!(workspace.tokens == tokens && workspace.heads == heads, "the sparse workspace has another shape");
    let raw = workspace.raw();
    // SAFETY: the caller guarantees the extents, and the workspace matches the shape, checked above.
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
            ptr::null_mut(),
        )
    })
}

/// Sol-Attn over BF16 heads of 128 for one sequence, reading query, key and value from `input` and writing `output`
/// at their offsets with the token and head strides of `layout`.
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
) -> Result<(), CudaError> {
    let last = |offset: usize, operand: usize| {
        offset as i64 + (tokens as i64 - 1) * layout.token_stride[operand] + (heads as i64 - 1) * layout.head_stride[operand] + HEAD_DIM as i64
    };
    for (operand, offset) in [offsets.query, offsets.key, offsets.value].into_iter().enumerate() {
        assert!(last(offset, operand) <= (input.bytes() / 2) as i64, "operand {operand} reaches past the input buffer");
    }
    assert!(last(offsets.output, 3) <= (output.bytes() / 2) as i64, "the output reaches past its buffer");
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
        )
    }
}
