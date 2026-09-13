use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::gemm::{self, Adapter};

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn int8(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }

    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 31) as f32)
    }

    fn bf16(&mut self, low: f32, high: f32) -> u16 {
        f32_to_bf16(self.uniform(low, high))
    }
}

fn upload(bytes: &[u8]) -> DeviceBuffer {
    let mut buffer = DeviceBuffer::new(bytes.len()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Runs every config on random operands, with a random adapter of `rank` when it is not zero, and compares the output
/// with an f64 reference within BF16 rounding.
fn check_every_config(m: usize, n: usize, k: usize, rank: usize, seed: u64) {
    let mut random = Random(seed);
    let activations: Vec<i8> = (0..m * k).map(|_| random.int8()).collect();
    let weights: Vec<i8> = (0..n * k).map(|_| random.int8()).collect();
    let activation_scales: Vec<f32> = (0..m).map(|_| random.uniform(0.5, 1.5)).collect();
    // Small weight scales keep the INT8 product about as large as the adapter's contribution.
    let weight_scales: Vec<f32> = (0..n).map(|_| random.uniform(0.0001, 0.001)).collect();
    let down: Vec<u16> = (0..m * rank).map(|_| random.bf16(-2.0, 2.0)).collect();
    let up: Vec<u16> = (0..n * rank).map(|_| random.bf16(-2.0, 2.0)).collect();
    let adapter_scale = 1.5f32;

    let activation_buffer = upload(&activations.iter().map(|&value| value as u8).collect::<Vec<_>>());
    let weight_buffer = upload(&weights.iter().map(|&value| value as u8).collect::<Vec<_>>());
    let activation_scale_buffer = upload(&f32_bytes(&activation_scales));
    let weight_scale_buffer = upload(&f32_bytes(&weight_scales));
    let (down_buffer, up_buffer) = (upload(&u16_bytes(&down)), upload(&u16_bytes(&up)));
    let adapter = Adapter { down: &down_buffer, up: &up_buffer, rank, scale: adapter_scale };

    let expected: Vec<f64> = (0..m)
        .flat_map(|row| (0..n).map(move |column| (row, column)))
        .map(|(row, column)| {
            let dot: i32 = (0..k).map(|index| activations[row * k + index] as i32 * weights[column * k + index] as i32).sum();
            let low_rank: f64 = (0..rank)
                .map(|index| bf16_to_f32(down[row * rank + index]) as f64 * bf16_to_f32(up[column * rank + index]) as f64)
                .sum();
            dot as f64 * activation_scales[row] as f64 * weight_scales[column] as f64 + adapter_scale as f64 * low_rank
        })
        .collect();

    for config in 0..gemm::int8_config_count() {
        let mut output = DeviceBuffer::new(m * n * 2).unwrap();
        gemm::int8_bf16(
            config,
            &activation_buffer,
            &weight_buffer,
            &activation_scale_buffer,
            &weight_scale_buffer,
            &mut output,
            m,
            n,
            k,
            (rank > 0).then_some(&adapter),
        )
        .unwrap();
        let mut output_bytes = vec![0; m * n * 2];
        output.copy_to_host(&mut output_bytes).unwrap();

        for (index, &expected) in expected.iter().enumerate() {
            let actual = bf16_to_f32(u16::from_le_bytes([output_bytes[index * 2], output_bytes[index * 2 + 1]])) as f64;
            let tolerance = expected.abs() / 256.0 + 1e-4;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{m} × {n} × {k}, rank {rank}, config {config}, row {}, column {}: expected {expected}, got {actual}",
                index / n,
                index % n
            );
        }
    }
}

#[test]
fn matches_cpu_reference_for_every_config() {
    check_every_config(300, 512, 384, 0, 7);
}

#[test]
fn matches_cpu_reference_over_several_tiles_per_block() {
    // 192 tiles, more than a Blackwell GPU has SMs, with a ragged last row of tiles and an odd number of K blocks.
    check_every_config(2000, 3072, 384, 0, 8);
}

#[test]
fn adds_a_low_rank_adapter() {
    check_every_config(300, 512, 256, 128, 9);
    // The rank of the fused qkv adapters, and more tiles than SMs.
    check_every_config(600, 3072, 256, 384, 10);
}
