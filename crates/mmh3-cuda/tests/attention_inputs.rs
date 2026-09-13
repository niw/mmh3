//! Preparing the attention inputs in the pass that normalizes q and k must change nothing: every attention path has to
//! match qk_norm_rope followed by the attention's own preparation in every bit.

use mmh3_core::dit::sparse::SparseSinks;
use mmh3_core::numeric::f32_to_bf16;
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::attention::{
    self, AttentionInputs, AttentionLayout, AttentionOffsets, AttentionPrecision, HEAD_DIM, HeadNorm, PreparedAttention, QuantizedWorkspace,
    SparseWorkspace,
};

const PAIRS: usize = 48;

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn bf16_buffer(values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|&value| f32_to_bf16(value).to_le_bytes()).collect();
    DeviceBuffer::from_bytes(&bytes).unwrap()
}

fn download(buffer: &DeviceBuffer) -> Vec<u8> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
}

struct Inputs {
    qkv: Vec<f32>,
    query_weight: DeviceBuffer,
    key_weight: DeviceBuffer,
    angles: DeviceBuffer,
}

impl Inputs {
    /// Blocks of tokens share a direction, so that Sol-Attn routes selectively.
    fn new(tokens: usize, heads: usize) -> Self {
        let inner = heads * HEAD_DIM;
        let mut random = Random(tokens as u64 * 31 + heads as u64);
        let directions: Vec<f32> = (0..tokens.div_ceil(64) * 3 * inner).map(|_| random.uniform(-2.0, 2.0)).collect();
        let qkv = (0..tokens * 3 * inner).map(|index| directions[(index / (3 * inner) / 64) * 3 * inner + index % (3 * inner)] + random.uniform(-1.0, 1.0)).collect();
        let query_weight = bf16_buffer(&(0..HEAD_DIM).map(|_| random.uniform(0.5, 1.5)).collect::<Vec<_>>());
        let key_weight = bf16_buffer(&(0..HEAD_DIM).map(|_| random.uniform(0.5, 1.5)).collect::<Vec<_>>());
        let angles = DeviceBuffer::from_f32(&(0..tokens * PAIRS).map(|_| random.uniform(-40.0, 40.0)).collect::<Vec<_>>()).unwrap();
        Inputs { qkv, query_weight, key_weight, angles }
    }

    fn norm(&self) -> HeadNorm<'_> {
        HeadNorm { query_weight: &self.query_weight, key_weight: &self.key_weight, angles: Some(&self.angles), pairs: PAIRS, epsilon: 1e-6 }
    }
}

/// The first `columns` of every qkv row.
fn columns(qkv: &DeviceBuffer, tokens: usize, heads: usize, columns: usize) -> Vec<u8> {
    let row = 3 * heads * HEAD_DIM * 2;
    download(qkv).chunks_exact(row).take(tokens).flat_map(|values| values[..columns * 2].to_vec()).collect()
}

fn check_sparse(tokens: usize, heads: usize, precision: AttentionPrecision) {
    let inner = heads * HEAD_DIM;
    let inputs = Inputs::new(tokens, heads);
    let layout = AttentionLayout {
        token_stride: [(3 * inner) as i64, (3 * inner) as i64, (3 * inner) as i64, inner as i64],
        head_stride: [HEAD_DIM as i64; 4],
        ..AttentionLayout::default()
    };
    let offsets = AttentionOffsets { query: 0, key: inner, value: 2 * inner, output: 0 };
    let sinks = SparseSinks { key_blocks: (0, 1), query_blocks: (0, 0) };
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut separate = bf16_buffer(&inputs.qkv);
    let mut separate_output = DeviceBuffer::new(tokens * inner * 2).unwrap();
    let separate_workspace = SparseWorkspace::with_precision(tokens, heads, precision).unwrap();
    attention::qk_norm_rope(&mut separate, &inputs.norm(), tokens, heads).unwrap();
    attention::sparse(&separate, &mut separate_output, offsets, tokens, heads, &layout, scale, 1.0, sinks, &separate_workspace, AttentionInputs::Raw).unwrap();

    let mut fused = bf16_buffer(&inputs.qkv);
    let mut fused_output = DeviceBuffer::new(tokens * inner * 2).unwrap();
    let fused_workspace = SparseWorkspace::with_precision(tokens, heads, precision).unwrap();
    attention::prepare_inputs(&mut fused, &inputs.norm(), tokens, heads, PreparedAttention::Sparse(&fused_workspace)).unwrap();
    attention::sparse(&fused, &mut fused_output, offsets, tokens, heads, &layout, scale, 1.0, sinks, &fused_workspace, AttentionInputs::Prepared).unwrap();

    let fraction = separate_workspace.routed_fraction().unwrap();
    assert_eq!(fraction, fused_workspace.routed_fraction().unwrap(), "{tokens} tokens, {precision:?}: routing differs");
    assert!(download(&separate_output) == download(&fused_output), "{tokens} tokens, {precision:?}: outputs differ");
    // Quantized attention reads only INT8 keys, so the fused pass leaves k as it was.
    let normalized = if precision == AttentionPrecision::Bf16 { 2 * inner } else { inner };
    assert!(columns(&separate, tokens, heads, normalized) == columns(&fused, tokens, heads, normalized), "{tokens} tokens, {precision:?}: normalized q or k differs");
    eprintln!("{tokens} tokens, {heads} heads, {precision:?}: identical, routed {fraction:.3}");
}

#[test]
fn sparse_attention_with_prepared_inputs_matches_separate_passes() {
    for tokens in [1, 63, 64, 65, 200, 1000] {
        for precision in [AttentionPrecision::Bf16, AttentionPrecision::Int8Fp8] {
            check_sparse(tokens, 3, precision);
        }
    }
}

#[test]
fn dense_quantized_attention_with_prepared_inputs_matches_separate_passes() {
    let heads = 2;
    let inner = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for tokens in [1, 63, 64, 65, 200, 1000] {
        let inputs = Inputs::new(tokens, heads);

        let mut separate = bf16_buffer(&inputs.qkv);
        let mut separate_output = DeviceBuffer::new(tokens * inner * 2).unwrap();
        let separate_workspace = QuantizedWorkspace::new(tokens, heads).unwrap();
        attention::qk_norm_rope(&mut separate, &inputs.norm(), tokens, heads).unwrap();
        attention::dense_quantized(&separate, &mut separate_output, scale, &separate_workspace, AttentionInputs::Raw).unwrap();

        let mut fused = bf16_buffer(&inputs.qkv);
        let mut fused_output = DeviceBuffer::new(tokens * inner * 2).unwrap();
        let fused_workspace = QuantizedWorkspace::new(tokens, heads).unwrap();
        attention::prepare_inputs(&mut fused, &inputs.norm(), tokens, heads, PreparedAttention::DenseQuantized(&fused_workspace)).unwrap();
        attention::dense_quantized(&fused, &mut fused_output, scale, &fused_workspace, AttentionInputs::Prepared).unwrap();

        assert!(download(&separate_output) == download(&fused_output), "{tokens} tokens: outputs differ");
    }
}
