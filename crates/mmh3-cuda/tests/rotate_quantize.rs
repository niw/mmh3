#![cfg(target_os = "linux")]
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::gemm;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// Mostly in [-2, 2), with an outlier up to 60 times larger about once in 256 draws, like DiT
    /// activations.
    fn activation(&mut self) -> f32 {
        let bits = self.next();
        let value = ((bits & 0xFFFFFF) as f32 / 16777216.0 - 0.5) * 4.0;
        if bits >> 24 == 0 { value * 60.0 } else { value }
    }
}

/// The ConvRot rotation of one row in the kernel's order of operations: radix-4 stages with strides
/// 1, 4, 16 and 64 inside every group of 256, then a factor of 1/16.
fn rotate(row: &mut [f32]) {
    for group in row.chunks_mut(256) {
        let mut stride = 1;
        while stride < 256 {
            for butterfly in 0..64 {
                let first = butterfly / stride * stride * 4 + butterfly % stride;
                let (x0, x1, x2, x3) = (
                    group[first],
                    group[first + stride],
                    group[first + 2 * stride],
                    group[first + 3 * stride],
                );
                group[first] = x0 + x1 + x2 - x3;
                group[first + stride] = x0 + x1 - x2 + x3;
                group[first + 2 * stride] = x0 - x1 + x2 + x3;
                group[first + 3 * stride] = -x0 + x1 + x2 + x3;
            }
            stride *= 4;
        }
        for value in group.iter_mut() {
            *value *= 1.0 / 16.0;
        }
    }
}

#[test]
fn matches_the_reference_bit_for_bit() {
    let mut random = Random(11);
    for (rows, columns) in [(37, 256), (29, 5376), (13, 14336), (5, 25600)] {
        let input: Vec<u16> = (0..rows * columns)
            .map(|_| f32_to_bf16(random.activation()))
            .collect();
        let mut input_buffer = DeviceBuffer::new(rows * columns * 2).unwrap();
        input_buffer
            .copy_from_host(
                &input
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut output_buffer = DeviceBuffer::new(rows * columns).unwrap();
        let mut scale_buffer = DeviceBuffer::new(rows * 4).unwrap();
        gemm::rotate_quantize(
            &input_buffer,
            &mut output_buffer,
            &mut scale_buffer,
            rows,
            columns,
        )
        .unwrap();
        let mut output = vec![0u8; rows * columns];
        output_buffer.copy_to_host(&mut output).unwrap();
        let mut scale_bytes = vec![0u8; rows * 4];
        scale_buffer.copy_to_host(&mut scale_bytes).unwrap();

        for row in 0..rows {
            let mut values: Vec<f32> = input[row * columns..(row + 1) * columns]
                .iter()
                .map(|&value| bf16_to_f32(value))
                .collect();
            rotate(&mut values);
            let maximum = values
                .iter()
                .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
            let scale = (maximum / 127.0).max(1e-30);
            let actual_scale =
                f32::from_le_bytes(scale_bytes[row * 4..row * 4 + 4].try_into().unwrap());
            assert_eq!(
                actual_scale.to_bits(),
                scale.to_bits(),
                "{rows} × {columns}, scale of row {row}"
            );
            for (column, value) in values.iter().enumerate() {
                let expected = (value / scale).round_ties_even().clamp(-128.0, 127.0) as i8;
                let actual = output[row * columns + column] as i8;
                assert_eq!(
                    actual, expected,
                    "{rows} × {columns}, row {row}, column {column}"
                );
            }
        }
    }
}
