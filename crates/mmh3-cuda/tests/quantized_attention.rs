//! Check approximate attention against the BF16 kernel, including ragged tiles and scratch reuse.
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::attention::{self, HEAD_DIM, QuantizedWorkspace};

fn download(buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.bytes()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
}

#[test]
fn bounded_error_with_ragged_tiles_and_reused_scales() {
    let heads = 3;
    let inner = heads * HEAD_DIM;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut random = 17u64;
    for tokens in [1usize, 63, 64, 65, 127, 129, 200] {
        let workspace = QuantizedWorkspace::new(tokens, heads).unwrap();
        let mut input = DeviceBuffer::new(tokens * inner * 6).unwrap();
        let mut output = DeviceBuffer::new(tokens * inner * 2).unwrap();
        let mut reference = DeviceBuffer::new(tokens * inner * 2).unwrap();
        // Reuse every scratch buffer while changing per-head value scales, then clear all inputs.
        for pass in 0..3 {
            let values: Vec<u8> = (0..tokens * 3 * inner).flat_map(|index| {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let value = 4.0 * ((random >> 40) as f32 / (1u64 << 24) as f32) - 2.0;
                let head = (index % inner) / HEAD_DIM;
                let factor = if (index / inner) % 3 == 2 { 2f32.powi((head as i32 - 1) * (pass + 1) * 3) } else { 1.0 };
                f32_to_bf16(if pass == 2 { 0.0 } else { value * factor }).to_le_bytes()
            }).collect();
            input.copy_from_host(&values).unwrap();
            attention::dense_bf16(&input, &mut reference, tokens, heads, scale).unwrap();
            attention::dense_quantized(&input, &mut output, scale, &workspace).unwrap();
            let expected = download(&reference);
            let actual = download(&output);
            assert!(actual.iter().all(|v| v.is_finite()));
            if pass == 2 {
                assert!(actual.iter().all(|&v| v == 0.0), "zero input must clear prior results");
                continue;
            }
            for head in 0..heads {
                let mut squared_error = 0.0f64;
                let mut squared_norm = 0.0f64;
                for token in 0..tokens {
                    for dimension in 0..HEAD_DIM {
                        let index = token * inner + head * HEAD_DIM + dimension;
                        squared_error += (actual[index] as f64 - expected[index] as f64).powi(2);
                        squared_norm += (expected[index] as f64).powi(2);
                    }
                }
                let relative = (squared_error / squared_norm).sqrt();
                assert!(relative < 0.05, "tokens {tokens}, head {head}, pass {pass}: relative L2 {relative}");
            }
        }
    }
}

#[test]
fn zero_queries_produce_uniform_means_including_the_last_key() {
    // Values +/-448 are exactly representable in E4M3 and force a scale of one.
    // Alternating values cancel; an odd sequence leaves a known nonzero mean.
    for tokens in [1usize, 63, 64, 65, 129] {
        let mut qkv = vec![0u16; tokens * 3 * HEAD_DIM];
        for token in 0..tokens {
            for dimension in 0..HEAD_DIM {
                qkv[(token * 3 + 2) * HEAD_DIM + dimension] = f32_to_bf16(
                    if (token + dimension) % 2 == 0 { 448.0 } else { -448.0 });
            }
        }
        let mut input = DeviceBuffer::new(qkv.len() * 2).unwrap();
        input.copy_from_host(&qkv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
        let mut output = DeviceBuffer::new(tokens * HEAD_DIM * 2).unwrap();
        let workspace = QuantizedWorkspace::new(tokens, 1).unwrap();
        attention::dense_quantized(&input, &mut output, 1.0 / (HEAD_DIM as f32).sqrt(), &workspace).unwrap();
        for (index, actual) in download(&output).into_iter().enumerate() {
            let mean = if tokens % 2 == 0 { 0.0 } else { 448.0 / tokens as f32 };
            let expected = bf16_to_f32(f32_to_bf16(if index % 2 == 0 { mean } else { -mean }));
            assert_eq!(actual, expected, "tokens {tokens}, element {index}");
        }
    }
}
