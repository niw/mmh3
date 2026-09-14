use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::gemm::{self, Output};

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 33) as f32 / (1u64 << 31) as f32)
    }

    /// Values rounded to BF16.
    fn bf16_values(&mut self, count: usize, bound: f32) -> Vec<f32> {
        (0..count)
            .map(|_| bf16_to_f32(f32_to_bf16(self.uniform(-bound, bound))))
            .collect()
    }
}

fn bf16_buffer(values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
        .collect();
    DeviceBuffer::from_bytes(&bytes).unwrap()
}

/// Quantizes BF16 rows like an INT8 ConvRot weight or activation.
fn quantize(values: &[f32], rows: usize, columns: usize) -> (DeviceBuffer, DeviceBuffer) {
    let mut quantized = DeviceBuffer::new(rows * columns).unwrap();
    let mut scales = DeviceBuffer::new(rows * 4).unwrap();
    gemm::rotate_quantize(
        &bf16_buffer(values),
        &mut quantized,
        &mut scales,
        rows,
        columns,
    )
    .unwrap();
    (quantized, scales)
}

fn product(
    activations: &(DeviceBuffer, DeviceBuffer),
    weights: &(DeviceBuffer, DeviceBuffer),
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    gemm::int8(
        0,
        &activations.0,
        &weights.0,
        &activations.1,
        &weights.1,
        Output {
            buffer: &mut output,
            f16: false,
            bias: None,
            swiglu: false,
        },
        m,
        n,
        k,
        None,
    )
    .unwrap();
    let mut bytes = vec![0; m * n * 2];
    output.copy_to_host(&mut bytes).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| bf16_to_f32(u16::from_le_bytes(pair)))
        .collect()
}

fn relative_error(actual: &[f32], expected: &[f64]) -> f64 {
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (actual as f64 - expected).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|value| value * value).sum();
    (difference / norm).sqrt()
}

#[test]
fn merges_like_quantizing_the_merged_weight() {
    let (m, n, k, rank) = (64, 256, 512, 20);
    let mut random = Random(3);
    let activations = random.bf16_values(m * k, 1.0);
    let weights = random.bf16_values(n * k, 1.0);
    let down = random.bf16_values(rank * k, 0.5);
    let up = random.bf16_values(n * rank, 0.5);
    let scale = 1.5f32;

    let merged_weights: Vec<f64> = (0..n * k)
        .map(|index| {
            let (row, column) = (index / k, index % k);
            weights[index] as f64
                + scale as f64
                    * (0..rank)
                        .map(|rank_index| {
                            up[row * rank + rank_index] as f64
                                * down[rank_index * k + column] as f64
                        })
                        .sum::<f64>()
        })
        .collect();
    let exact = |weights: &dyn Fn(usize) -> f64| -> Vec<f64> {
        (0..m * n)
            .map(|index| {
                let (row, column) = (index / n, index % n);
                (0..k)
                    .map(|inner| activations[row * k + inner] as f64 * weights(column * k + inner))
                    .sum()
            })
            .collect()
    };
    let expected = exact(&|index| merged_weights[index]);
    let base = exact(&|index| weights[index] as f64);
    let base_f32: Vec<f32> = base.iter().map(|&value| value as f32).collect();
    let change = relative_error(&base_f32, &expected);
    assert!(
        change > 0.3,
        "the update changes the output by only {change}"
    );

    let quantized_activations = quantize(&activations, m, k);
    let mut merged = quantize(&weights, n, k);
    let mut down_buffer =
        DeviceBuffer::from_f32(&down.iter().map(|&value| value * scale).collect::<Vec<_>>())
            .unwrap();
    gemm::merge_low_rank(
        &mut merged.0,
        &mut merged.1,
        &DeviceBuffer::from_f32(&up).unwrap(),
        &mut down_buffer,
        n,
        k,
        rank,
    )
    .unwrap();
    let merged_error = relative_error(
        &product(&quantized_activations, &merged, m, n, k),
        &expected,
    );
    // Quantizing the exact merged weight in BF16 is as good as a merge can get.
    let requantized: Vec<f32> = merged_weights
        .iter()
        .map(|&value| bf16_to_f32(f32_to_bf16(value as f32)))
        .collect();
    let requantized_error = relative_error(
        &product(
            &quantized_activations,
            &quantize(&requantized, n, k),
            m,
            n,
            k,
        ),
        &expected,
    );
    eprintln!("merged {merged_error:.3e}, requantized {requantized_error:.3e}");
    assert!(
        merged_error < 1.5 * requantized_error + 1e-3,
        "merged {merged_error}, requantized {requantized_error}"
    );
}
