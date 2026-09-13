use mmh3_cuda::DeviceBuffer;
use mmh3_core::numeric::f16_to_f32;
use mmh3_cuda::attention::{self, AttentionLayout, AttentionOffsets, Element, HEAD_DIM};

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn check_against_reference(tokens: usize, heads: usize) {
    let inner = heads * HEAD_DIM;
    let mut random = Random(tokens as u64 * 31 + heads as u64);
    let qkv: Vec<u16> = (0..tokens * 3 * inner).map(|_| to_bf16(random.uniform(-2.0, 2.0))).collect();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut qkv_buffer = DeviceBuffer::new(qkv.len() * 2).unwrap();
    qkv_buffer.copy_from_host(&qkv.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let mut output_buffer = DeviceBuffer::new(tokens * inner * 2).unwrap();
    attention::dense_bf16(&qkv_buffer, &mut output_buffer, tokens, heads, scale).unwrap();
    let mut output_bytes = vec![0; tokens * inner * 2];
    output_buffer.copy_to_host(&mut output_bytes).unwrap();

    let element = |token: usize, part: usize, head: usize, dimension: usize| {
        from_bf16(qkv[token * 3 * inner + part * inner + head * HEAD_DIM + dimension]) as f64
    };
    for head in 0..heads {
        for query in 0..tokens {
            let scores: Vec<f64> = (0..tokens)
                .map(|key| {
                    (0..HEAD_DIM).map(|dimension| element(query, 0, head, dimension) * element(key, 1, head, dimension)).sum::<f64>()
                        * scale as f64
                })
                .collect();
            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for dimension in 0..HEAD_DIM {
                let expected: f64 =
                    (0..tokens).map(|key| weights[key] * element(key, 2, head, dimension)).sum::<f64>() / total;
                let offset = (query * inner + head * HEAD_DIM + dimension) * 2;
                let actual = from_bf16(u16::from_le_bytes([output_bytes[offset], output_bytes[offset + 1]])) as f64;
                assert!(
                    (actual - expected).abs() <= 0.02 + 0.01 * expected.abs(),
                    "tokens {tokens}, head {head}, query {query}, dimension {dimension}: expected {expected}, got {actual}"
                );
            }
        }
    }
}

#[test]
fn matches_cpu_reference() {
    for (tokens, heads) in [(1, 1), (64, 2), (200, 3)] {
        check_against_reference(tokens, heads);
    }
}

fn to_f16(value: f32) -> u16 {
    // Round to nearest even through f64, adequate for test data in [-2, 2].
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let magnitude = value.abs();
    if magnitude < 6.1035156e-5 {
        return sign | (magnitude / 5.9604645e-8).round() as u16;
    }
    let exponent = magnitude.log2().floor() as i32;
    let mantissa = (magnitude / 2f32.powi(exponent) - 1.0) * 1024.0;
    let rounded = mantissa.round_ties_even() as u32;
    let (exponent, rounded) = if rounded == 1024 { (exponent + 1, 0) } else { (exponent, rounded) };
    sign | (((exponent + 15) as u16) << 10) | rounded as u16
}

/// The VAE decoder layout: FP16, heads of 64 with [q, k, v] interleaved per head, several independent tiles.
#[test]
fn matches_cpu_reference_for_interleaved_half_precision_heads() {
    let (batch, tokens, heads, head_dim) = (2, 97, 3, 64);
    let per_token = heads * 3 * head_dim;
    let mut random = Random(99);
    let values: Vec<u16> = (0..batch * tokens * per_token).map(|_| to_f16(random.uniform(-2.0, 2.0))).collect();
    let input = {
        let mut buffer = DeviceBuffer::new(values.len() * 2).unwrap();
        buffer.copy_from_host(&values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        buffer
    };
    let output_width = heads * head_dim;
    let mut output = DeviceBuffer::new(batch * tokens * output_width * 2).unwrap();
    let layout = AttentionLayout {
        token_stride: [per_token as i64, per_token as i64, per_token as i64, output_width as i64],
        head_stride: [(3 * head_dim) as i64, (3 * head_dim) as i64, (3 * head_dim) as i64, head_dim as i64],
        batch_stride: [(tokens * per_token) as i64, (tokens * per_token) as i64, (tokens * per_token) as i64, (tokens * output_width) as i64],
        ..AttentionLayout::default()
    };
    let offsets = AttentionOffsets { query: 0, key: head_dim, value: 2 * head_dim, output: 0 };
    let scale = 1.0 / (head_dim as f32).sqrt();
    attention::attention(Element::F16, head_dim, &input, &mut output, offsets, tokens, heads, batch, &layout, scale).unwrap();
    let mut bytes = vec![0; batch * tokens * output_width * 2];
    output.copy_to_host(&mut bytes).unwrap();

    let element = |tile: usize, token: usize, head: usize, part: usize, dimension: usize| {
        f16_to_f32(values[tile * tokens * per_token + token * per_token + head * 3 * head_dim + part * head_dim + dimension]) as f64
    };
    for tile in 0..batch {
        for head in 0..heads {
            for query in 0..tokens {
                let scores: Vec<f64> = (0..tokens)
                    .map(|key| (0..head_dim).map(|d| element(tile, query, head, 0, d) * element(tile, key, head, 1, d)).sum::<f64>() * scale as f64)
                    .collect();
                let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for dimension in 0..head_dim {
                    let expected: f64 = (0..tokens).map(|key| weights[key] * element(tile, key, head, 2, dimension)).sum::<f64>() / total;
                    let offset = ((tile * tokens + query) * output_width + head * head_dim + dimension) * 2;
                    let actual = f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])) as f64;
                    assert!((actual - expected).abs() <= 0.01 + 0.005 * expected.abs(), "tile {tile}, head {head}, query {query}, dimension {dimension}: expected {expected}, got {actual}");
                }
            }
        }
    }
}

/// The text encoder layout: BF16 [q | k | v] rows with eight query heads per key and value head, causal.
#[test]
fn matches_cpu_reference_for_causal_grouped_heads() {
    let (tokens, query_heads, key_heads) = (150, 8, 2);
    let group = query_heads / key_heads;
    let row = (query_heads + 2 * key_heads) * HEAD_DIM;
    let mut random = Random(7);
    let values: Vec<u16> = (0..tokens * row).map(|_| to_bf16(random.uniform(-2.0, 2.0))).collect();
    let input = {
        let mut buffer = DeviceBuffer::new(values.len() * 2).unwrap();
        buffer.copy_from_host(&values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        buffer
    };
    let output_width = query_heads * HEAD_DIM;
    let mut output = DeviceBuffer::new(tokens * output_width * 2).unwrap();
    let layout = AttentionLayout {
        token_stride: [row as i64, row as i64, row as i64, output_width as i64],
        head_stride: [HEAD_DIM as i64; 4],
        heads_per_key_value: group as i32,
        causal: 1,
        ..AttentionLayout::default()
    };
    let offsets = AttentionOffsets { query: 0, key: query_heads * HEAD_DIM, value: (query_heads + key_heads) * HEAD_DIM, output: 0 };
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    attention::attention(Element::Bf16, HEAD_DIM, &input, &mut output, offsets, tokens, query_heads, 1, &layout, scale).unwrap();
    let mut bytes = vec![0; tokens * output_width * 2];
    output.copy_to_host(&mut bytes).unwrap();

    let element = |token: usize, column: usize| from_bf16(values[token * row + column]) as f64;
    for head in 0..query_heads {
        let (key_column, value_column) = ((query_heads + head / group) * HEAD_DIM, (query_heads + key_heads + head / group) * HEAD_DIM);
        for query in 0..tokens {
            let scores: Vec<f64> = (0..=query)
                .map(|key| (0..HEAD_DIM).map(|d| element(query, head * HEAD_DIM + d) * element(key, key_column + d)).sum::<f64>() * scale as f64)
                .collect();
            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for dimension in 0..HEAD_DIM {
                let expected: f64 = (0..=query).map(|key| weights[key] * element(key, value_column + dimension)).sum::<f64>() / total;
                let offset = (query * output_width + head * HEAD_DIM + dimension) * 2;
                let actual = from_bf16(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])) as f64;
                assert!((actual - expected).abs() <= 0.02 + 0.01 * expected.abs(), "head {head}, query {query}, dimension {dimension}: expected {expected}, got {actual}");
            }
        }
    }
}
