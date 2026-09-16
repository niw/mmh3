#![cfg(target_os = "linux")]
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::gemm::{self, Output};
use mmh3_cuda::nvfp4::{
    self, ADAPTER_DOWN_NORM, AdapterSource, Columns, Nvfp4Activations, Nvfp4Scale, Nvfp4Weight,
};
use std::ffi::{c_int, c_void};

unsafe extern "C" {
    fn mmh3_swiglu(
        input: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        ffn: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 33) as f32 / (1u64 << 31) as f32)
    }

    /// Roughly Gaussian values with a few large outliers, rounded to BF16.
    fn bf16_values(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let value: f32 = (0..4).map(|_| self.uniform(-1.0, 1.0)).sum();
                let value = if index % 97 == 0 { value * 20.0 } else { value };
                bf16_to_f32(f32_to_bf16(value))
            })
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

fn download_bf16(buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
    let mut bytes = vec![0; count * 2];
    buffer.copy_to_host(&mut bytes).unwrap();
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
fn matches_the_exact_product_within_fp4_rounding() {
    let (m, n, k) = (300, 512, 1024);
    let mut random = Random(5);
    let activations = random.bf16_values(m * k);
    let weights = random.bf16_values(n * k);
    let expected: Vec<f64> = (0..m * n)
        .map(|index| {
            let (row, column) = (index / n, index % n);
            (0..k)
                .map(|inner| {
                    activations[row * k + inner] as f64 * weights[column * k + inner] as f64
                })
                .sum()
        })
        .collect();

    let (int8_weights, row_scales) = quantize(&weights, n, k);
    let input = bf16_buffer(&activations);
    let buffers = Nvfp4Activations::new(m, k, 0).unwrap();
    let weight = Nvfp4Weight::from_int8(&int8_weights, &row_scales, n, k, false).unwrap();
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear(
        &weight,
        &input,
        &mut output,
        m,
        &Nvfp4Scale::new().unwrap(),
        &buffers,
    )
    .unwrap();
    let nvfp4_error = relative_error(&download_bf16(&output, m * n), &expected);

    let quantized_activations = quantize(&activations, m, k);
    let mut int8_output = DeviceBuffer::new(m * n * 2).unwrap();
    gemm::int8(
        0,
        &quantized_activations.0,
        &int8_weights,
        &quantized_activations.1,
        &row_scales,
        Output {
            buffer: &mut int8_output,
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
    let int8_error = relative_error(&download_bf16(&int8_output, m * n), &expected);
    eprintln!("NVFP4 {nvfp4_error:.3e}, INT8 {int8_error:.3e}");
    // FP4 E2M1 keeps about one significant bit per value, so each operand loses several percent.
    assert!(nvfp4_error < 0.15, "NVFP4 relative error {nvfp4_error}");
}

#[test]
fn undoes_the_swiglu_interleave() {
    let (m, n, k, rank) = (130, 256, 512, 64);
    let mut random = Random(6);
    let activations = random.bf16_values(m * k);
    let weights = random.bf16_values(n * k);
    let (int8_weights, row_scales) = quantize(&weights, n, k);
    let interleaved_weights = gemm::interleave_swiglu_rows(&int8_weights, n, k).unwrap();
    let interleaved_scales = gemm::interleave_swiglu_rows(&row_scales, n, 4).unwrap();
    let (down, down_scales) = quantize(&random.bf16_values(rank * k), rank, k);
    let up = bf16_buffer(&random.bf16_values(n * rank));
    let interleaved_up = gemm::interleave_swiglu_rows(&up, n, rank * 2).unwrap();
    let input = bf16_buffer(&activations);
    let buffers = Nvfp4Activations::new(m, k + 256, rank).unwrap();

    let mut outputs = Vec::new();
    for (weights, scales, up, deinterleave) in [
        (&int8_weights, &row_scales, &up, false),
        (
            &interleaved_weights,
            &interleaved_scales,
            &interleaved_up,
            true,
        ),
    ] {
        let adapter = AdapterSource {
            down: &down,
            down_scales: &down_scales,
            up,
            rank,
            scale: 0.01,
        };
        let weight = Nvfp4Weight::from_int8_with(
            weights,
            scales,
            n,
            k,
            deinterleave,
            Columns::Adapter(adapter),
        )
        .unwrap();
        let mut output = DeviceBuffer::new(m * n * 2).unwrap();
        nvfp4::linear(
            &weight,
            &input,
            &mut output,
            m,
            &Nvfp4Scale::new().unwrap(),
            &buffers,
        )
        .unwrap();
        outputs.push(download_bf16(&output, m * n));
    }
    assert_eq!(outputs[0], outputs[1]);
}

/// Rotates each 256 group of a row by the normalized regular Hadamard matrix of ConvRot.
fn rotate(row: &mut [f64]) {
    for group in row.as_chunks_mut::<256>().0 {
        for stride in [1, 4, 16, 64] {
            for base in 0..256 {
                if (base / stride) % 4 != 0 {
                    continue;
                }
                let x: [f64; 4] = std::array::from_fn(|index| group[base + index * stride]);
                let y = [
                    x[0] + x[1] + x[2] - x[3],
                    x[0] + x[1] - x[2] + x[3],
                    x[0] - x[1] + x[2] + x[3],
                    -x[0] + x[1] + x[2] + x[3],
                ];
                for index in 0..4 {
                    group[base + index * stride] = y[index];
                }
            }
        }
        group.iter_mut().for_each(|value| *value /= 16.0);
    }
}

/// The nearest value in `table`, ties to the even index, saturating at the largest.
fn nearest(value: f64, table: &[f64]) -> f64 {
    let magnitude = value.abs();
    let mut best = 0;
    for index in 1..table.len() {
        let (distance, best_distance) = (
            (table[index] - magnitude).abs(),
            (table[best] - magnitude).abs(),
        );
        if distance < best_distance || (distance == best_distance && index % 2 == 0) {
            best = index;
        }
    }
    table[best].copysign(value)
}

/// Positive finite E4M3 values in code order.
fn e4m3_table() -> Vec<f64> {
    (0..0x7Fu32)
        .map(|code| {
            let (exponent, mantissa) = ((code >> 3) as i32, (code & 7) as f64);
            if exponent == 0 {
                mantissa / 8.0 * 2f64.powi(-6)
            } else {
                (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
            }
        })
        .collect()
}

/// Quantizes and dequantizes rows like the NVFP4 kernels with `tensor_scale`, in blocks of 16
/// values.
fn fake_quantize(rows: &mut [f64], tensor_scale: f64) {
    let (e4m3, e2m1) = (e4m3_table(), [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
    for block in rows.as_chunks_mut::<16>().0 {
        let maximum = block
            .iter()
            .fold(0.0f64, |maximum, value| maximum.max(value.abs()));
        // The kernels compute the block scale and its factor in f32.
        let block_scale = nearest(
            (maximum as f32 / 6.0 * (1.0 / tensor_scale as f32)) as f64,
            &e4m3,
        );
        let factor = if block_scale > 0.0 {
            ((1.0 / tensor_scale as f32) / block_scale as f32) as f64
        } else {
            0.0
        };
        for value in block.iter_mut() {
            *value =
                nearest((*value as f32 * factor as f32) as f64, &e2m1) * block_scale * tensor_scale;
        }
    }
}

#[test]
fn matches_a_host_model_of_the_quantization() {
    let (m, n, k) = (200, 256, 512);
    let mut random = Random(7);
    let activations = random.bf16_values(m * k);
    let weights = random.bf16_values(n * k);
    let (int8_weights, row_scales) = quantize(&weights, n, k);
    let input = bf16_buffer(&activations);
    let buffers = Nvfp4Activations::new(m, k, 0).unwrap();
    let weight = Nvfp4Weight::from_int8(&int8_weights, &row_scales, n, k, false).unwrap();
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear(
        &weight,
        &input,
        &mut output,
        m,
        &Nvfp4Scale::new().unwrap(),
        &buffers,
    )
    .unwrap();
    let actual = download_bf16(&output, m * n);

    let mut weight_bytes = vec![0u8; n * k];
    int8_weights.copy_to_host(&mut weight_bytes).unwrap();
    let scales = row_scales.to_f32().unwrap();
    let mut model_weights: Vec<f64> = weight_bytes
        .iter()
        .enumerate()
        .map(|(index, &byte)| byte as i8 as f64 * scales[index / k] as f64)
        .collect();
    let weight_tensor_scale =
        (128.0 * scales.iter().fold(0.0f32, |a, &b| a.max(b)) / (6.0 * 448.0)) as f64;
    fake_quantize(&mut model_weights, (weight_tensor_scale as f32) as f64);
    let mut model_activations: Vec<f64> = activations.iter().map(|&value| value as f64).collect();
    model_activations.chunks_exact_mut(k).for_each(rotate);
    let activation_maximum = model_activations
        .iter()
        .fold(0.0f64, |maximum, value| maximum.max(value.abs()))
        as f32;
    fake_quantize(
        &mut model_activations,
        (activation_maximum / (6.0 * 448.0)) as f64,
    );
    let expected: Vec<f64> = (0..m * n)
        .map(|index| {
            let (row, column) = (index / n, index % n);
            (0..k)
                .map(|inner| model_activations[row * k + inner] * model_weights[column * k + inner])
                .sum()
        })
        .collect();
    let error = relative_error(&actual, &expected);
    eprintln!("against the host model {error:.3e}");
    assert!(
        error < 1e-2,
        "relative error {error} against the host model"
    );
}

#[test]
fn keeps_the_result_with_the_delayed_scale() {
    let (m, n, k) = (200, 256, 512);
    let mut random = Random(8);
    let activations = random.bf16_values(m * k);
    let weights = random.bf16_values(n * k);
    let (int8_weights, row_scales) = quantize(&weights, n, k);
    let weight = Nvfp4Weight::from_int8(&int8_weights, &row_scales, n, k, false).unwrap();
    let input = bf16_buffer(&activations);
    let buffers = Nvfp4Activations::new(m, k, 0).unwrap();
    let scale = Nvfp4Scale::new().unwrap();
    // The first call finds the tensor scale, and the later ones use it with a margin of a power
    // of two, which moves the block scales by whole exponents.
    let outputs: Vec<Vec<f32>> = (0..3)
        .map(|_| {
            let mut output = DeviceBuffer::new(m * n * 2).unwrap();
            nvfp4::linear(&weight, &input, &mut output, m, &scale, &buffers).unwrap();
            download_bf16(&output, m * n)
        })
        .collect();
    assert!(scale.is_calibrated());
    let expected: Vec<f64> = outputs[0].iter().map(|&value| value as f64).collect();
    for output in &outputs[1..] {
        let difference = relative_error(output, &expected);
        assert!(
            difference < 1e-3,
            "delayed scale changes the result by {difference}"
        );
    }
}

#[test]
fn quantizes_the_swiglu_of_its_input() {
    let (m, n, ffn) = (130, 256, 512);
    let mut random = Random(9);
    let expanded = random.bf16_values(m * 2 * ffn);
    let weights = random.bf16_values(n * ffn);
    let (int8_weights, row_scales) = quantize(&weights, n, ffn);
    let weight = Nvfp4Weight::from_int8(&int8_weights, &row_scales, n, ffn, false).unwrap();
    let input = bf16_buffer(&expanded);
    let activated = DeviceBuffer::new(m * ffn * 2).unwrap();
    // SAFETY: input holds `m × 2 × ffn` values and activated `m × ffn`.
    let status = unsafe {
        mmh3_swiglu(
            input.pointer(),
            activated.pointer(),
            m as c_int,
            ffn as c_int,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, 0);
    let buffers = Nvfp4Activations::new(m, ffn, 0).unwrap();
    let mut separate = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear(
        &weight,
        &activated,
        &mut separate,
        m,
        &Nvfp4Scale::new().unwrap(),
        &buffers,
    )
    .unwrap();
    let mut fused = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear_swiglu(
        &weight,
        &input,
        &mut fused,
        m,
        &Nvfp4Scale::new().unwrap(),
        &buffers,
    )
    .unwrap();
    assert_eq!(
        download_bf16(&separate, m * n),
        download_bf16(&fused, m * n)
    );
}

/// Dequantizes INT8 rows of `columns` values with one scale per row.
fn dequantize(
    quantized: &DeviceBuffer,
    scales: &DeviceBuffer,
    rows: usize,
    columns: usize,
) -> Vec<f64> {
    let mut bytes = vec![0u8; rows * columns];
    quantized.copy_to_host(&mut bytes).unwrap();
    let scales = scales.to_f32_range(0, rows).unwrap();
    bytes
        .iter()
        .enumerate()
        .map(|(index, &byte)| byte as i8 as f64 * scales[index / columns] as f64)
        .collect()
}

fn largest(values: impl IntoIterator<Item = f32>) -> f32 {
    values
        .into_iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
}

/// `[rows, inner] · [columns, inner]ᵀ`.
fn product(left: &[f64], right: &[f64], rows: usize, columns: usize, inner: usize) -> Vec<f64> {
    (0..rows * columns)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            (0..inner)
                .map(|position| left[row * inner + position] * right[column * inner + position])
                .sum()
        })
        .collect()
}

#[test]
fn matches_a_host_model_with_an_adapter() {
    let (m, n, k, rank, scale) = (200, 256, 512, 64, 0.05f32);
    let mut random = Random(10);
    let activations = random.bf16_values(m * k);
    let (int8_weights, row_scales) = quantize(&random.bf16_values(n * k), n, k);
    let (down, down_scales) = quantize(&random.bf16_values(rank * k), rank, k);
    let up_values = random.bf16_values(n * rank);
    let up = bf16_buffer(&up_values);
    let adapter = AdapterSource {
        down: &down,
        down_scales: &down_scales,
        up: &up,
        rank,
        scale,
    };
    let weight = Nvfp4Weight::from_int8_with(
        &int8_weights,
        &row_scales,
        n,
        k,
        false,
        Columns::Adapter(adapter),
    )
    .unwrap();
    assert_eq!((weight.columns, weight.adapter_rank()), (k + 256, rank));
    let buffers = Nvfp4Activations::new(m, weight.columns, rank).unwrap();
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear(
        &weight,
        &bf16_buffer(&activations),
        &mut output,
        m,
        &Nvfp4Scale::new().unwrap(),
        &buffers,
    )
    .unwrap();
    let actual = download_bf16(&output, m * n);

    let mut model_activations: Vec<f64> = activations.iter().map(|&value| value as f64).collect();
    model_activations.chunks_exact_mut(k).for_each(rotate);
    let activation_scale =
        (largest(model_activations.iter().map(|&value| value as f32)) / (6.0 * 448.0)) as f64;
    fake_quantize(&mut model_activations, activation_scale);

    // The down projection is scaled so that its longest row has norm ADAPTER_DOWN_NORM, and the
    // up projection by the inverse.
    let mut model_down = dequantize(&down, &down_scales, rank, k);
    let down_norm = model_down
        .chunks_exact(k)
        .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt() as f32)
        .fold(0.0f32, f32::max);
    let balance = ADAPTER_DOWN_NORM / down_norm;
    let down_row_scales = down_scales.to_f32_range(0, rank).unwrap();
    fake_quantize(
        &mut model_down,
        (128.0 * largest(down_row_scales) / (6.0 * 448.0)) as f64,
    );
    let mut adapter_activations: Vec<f64> = product(&model_activations, &model_down, m, rank, k)
        .into_iter()
        .map(|value| bf16_to_f32(f32_to_bf16((value * balance as f64) as f32)) as f64)
        .collect();
    fake_quantize(&mut adapter_activations, activation_scale);

    let multiplier = scale / balance;
    let mut model_weights = dequantize(&int8_weights, &row_scales, n, k);
    let mut model_up: Vec<f64> = up_values
        .iter()
        .map(|&value| (value * multiplier) as f64)
        .collect();
    let weight_scale = (128.0 * largest(row_scales.to_f32().unwrap()))
        .max(largest(up_values.iter().copied()) * multiplier)
        / (6.0 * 448.0);
    fake_quantize(&mut model_weights, weight_scale as f64);
    fake_quantize(&mut model_up, weight_scale as f64);
    let expected: Vec<f64> = product(&model_activations, &model_weights, m, n, k)
        .into_iter()
        .zip(product(&adapter_activations, &model_up, m, n, rank))
        .map(|(layer, adapter)| layer + adapter)
        .collect();
    let error = relative_error(&actual, &expected);
    let adapter_share = relative_error(
        &product(&model_activations, &model_weights, m, n, k)
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>(),
        &expected,
    );
    eprintln!("against the host model {error:.3e}, the adapter's share {adapter_share:.3e}");
    assert!(
        adapter_share > 0.1,
        "the adapter changes too little to check"
    );
    assert!(
        error < 1e-2,
        "relative error {error} against the host model"
    );
}

#[test]
fn ignores_zero_columns() {
    let (m, n, k) = (200, 256, 512);
    let mut random = Random(11);
    let input = bf16_buffer(&random.bf16_values(m * k));
    let (int8_weights, row_scales) = quantize(&random.bf16_values(n * k), n, k);
    let buffers = Nvfp4Activations::new(m, k + 128, 0).unwrap();
    let outputs: Vec<Vec<f32>> = [Columns::Inputs, Columns::Zeros(k + 128)]
        .into_iter()
        .map(|columns| {
            let weight =
                Nvfp4Weight::from_int8_with(&int8_weights, &row_scales, n, k, false, columns)
                    .unwrap();
            let mut output = DeviceBuffer::new(m * n * 2).unwrap();
            nvfp4::linear(
                &weight,
                &input,
                &mut output,
                m,
                &Nvfp4Scale::new().unwrap(),
                &buffers,
            )
            .unwrap();
            download_bf16(&output, m * n)
        })
        .collect();
    let expected: Vec<f64> = outputs[0].iter().map(|&value| value as f64).collect();
    let difference = relative_error(&outputs[1], &expected);
    assert!(
        difference < 1e-3,
        "zero columns change the result by {difference}"
    );
}

#[test]
fn saves_and_loads_the_chosen_algorithms() {
    let (m, n, k) = (72, 128, 256);
    let mut random = Random(12);
    let (int8_weights, row_scales) = quantize(&random.bf16_values(n * k), n, k);
    let weight = Nvfp4Weight::from_int8(&int8_weights, &row_scales, n, k, false).unwrap();
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    nvfp4::linear(
        &weight,
        &bf16_buffer(&random.bf16_values(m * k)),
        &mut output,
        m,
        &Nvfp4Scale::new().unwrap(),
        &Nvfp4Activations::new(m, k, 0).unwrap(),
    )
    .unwrap();

    let directory = std::env::temp_dir().join(format!("mmh3-nvfp4-{}", std::process::id()));
    let path = directory.join("algorithms.txt");
    assert!(nvfp4::save_algorithms(&path).unwrap());
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.lines()
            .any(|line| line.starts_with(&format!("{m} {n} {k} ")))
    );
    let saved = text.lines().count() - 2;
    assert!(nvfp4::load_algorithms(&path).unwrap() >= saved);

    std::fs::write(&path, text.replacen("format", "other format", 1)).unwrap();
    assert_eq!(nvfp4::load_algorithms(&path).unwrap(), 0);
    std::fs::remove_dir_all(&directory).unwrap();
}
