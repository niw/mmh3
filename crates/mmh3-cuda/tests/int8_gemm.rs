use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::gemm;

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
}

fn upload(bytes: &[u8]) -> DeviceBuffer {
    let mut buffer = DeviceBuffer::new(bytes.len()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn matches_cpu_reference_for_every_config() {
    let (m, n, k) = (300, 512, 320);
    let mut random = Random(7);
    let activations: Vec<i8> = (0..m * k).map(|_| random.int8()).collect();
    let weights: Vec<i8> = (0..n * k).map(|_| random.int8()).collect();
    let activation_scales: Vec<f32> = (0..m).map(|_| random.uniform(0.5, 1.5)).collect();
    let weight_scales: Vec<f32> = (0..n).map(|_| random.uniform(0.001, 0.01)).collect();

    let activation_buffer = upload(&activations.iter().map(|&value| value as u8).collect::<Vec<_>>());
    let weight_buffer = upload(&weights.iter().map(|&value| value as u8).collect::<Vec<_>>());
    let activation_scale_buffer = upload(&f32_bytes(&activation_scales));
    let weight_scale_buffer = upload(&f32_bytes(&weight_scales));

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
        )
        .unwrap();
        let mut output_bytes = vec![0; m * n * 2];
        output.copy_to_host(&mut output_bytes).unwrap();

        for row in 0..m {
            for column in 0..n {
                let dot: i32 = (0..k)
                    .map(|index| activations[row * k + index] as i32 * weights[column * k + index] as i32)
                    .sum();
                let expected = dot as f64 * activation_scales[row] as f64 * weight_scales[column] as f64;
                let offset = (row * n + column) * 2;
                let bits = u16::from_le_bytes([output_bytes[offset], output_bytes[offset + 1]]);
                let actual = f32::from_bits((bits as u32) << 16) as f64;
                let tolerance = expected.abs() / 256.0 + 1e-6;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "config {config}, row {row}, column {column}: expected {expected}, got {actual}"
                );
            }
        }
    }
}
