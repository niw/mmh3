#![cfg(target_os = "macos")]
use mmh3_metal::{Device, ops::Array};

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + salt * 31) % 101) as f32 / 50.0 - 1.0)
        .collect()
}

#[test]
fn grouped_causal_attention_matches_f64_reference() {
    for (queries, tokens, heads, kv_heads, dim) in [
        (7, 7, 4, 2, 19),
        (17, 35, 4, 1, 32),
        (33, 33, 4, 2, 37),
        (17, 17, 2, 2, 64),
        (35, 35, 4, 1, 128),
        (9, 19, 2, 1, 129),
        (17, 17, 2, 1, 256),
    ] {
        check_attention(queries, tokens, heads, kv_heads, dim);
    }
}

fn check_attention(queries: usize, tokens: usize, heads: usize, kv_heads: usize, dim: usize) {
    let device = Device::new().unwrap();
    // Non-power-of-two head width and uneven token count exercise bounds and inactive lanes.
    let q = values(queries * heads * dim, 1);
    let k = values(tokens * kv_heads * dim, 2);
    let v = values(tokens * kv_heads * dim, 3);
    let query = Array::from_f32(&device, queries, heads * dim, &q).unwrap();
    let key = Array::from_f32(&device, tokens, kv_heads * dim, &k).unwrap();
    let value = Array::from_f32(&device, tokens, kv_heads * dim, &v).unwrap();

    for causal in [false, true] {
        if causal && queries != tokens {
            continue;
        }

        let actual = query
            .attention(&key, &value, heads, kv_heads, causal)
            .unwrap()
            .to_f32()
            .unwrap();
        for row in 0..queries {
            for head in 0..heads {
                let kh = head / (heads / kv_heads);
                let scores: Vec<f64> = (0..if causal { row + 1 } else { tokens })
                    .map(|t| {
                        (0..dim)
                            .map(|c| {
                                q[(row * heads + head) * dim + c] as f64
                                    * k[(t * kv_heads + kh) * dim + c] as f64
                            })
                            .sum::<f64>()
                            / (dim as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f64 = probabilities.iter().sum();

                for c in 0..dim {
                    let expected = probabilities
                        .iter()
                        .enumerate()
                        .map(|(t, &p)| p * v[(t * kv_heads + kh) * dim + c] as f64)
                        .sum::<f64>()
                        / total;
                    assert!(
                        (actual[(row * heads + head) * dim + c] as f64 - expected).abs() < 1e-6
                    );
                }
            }
        }
    }
}

#[test]
fn products_and_norms_cover_unaligned_dimensions_and_invalid_shapes() {
    let device = Device::new().unwrap();
    let x = values(3 * 37, 4);
    let w = values(5 * 37, 5);
    let a = Array::from_f32(&device, 3, 37, &x).unwrap();
    let b = Array::from_f32(&device, 5, 37, &w).unwrap();
    let product = a.linear(&b).unwrap().to_f32().unwrap();

    for r in 0..3 {
        for o in 0..5 {
            let expected: f64 = (0..37)
                .map(|i| x[r * 37 + i] as f64 * w[o * 37 + i] as f64)
                .sum();
            assert!((product[r * 5 + o] as f64 - expected).abs() < 1e-5);
        }
    }

    let scales = values(37, 6);
    let scale = Array::from_f32(&device, 1, 37, &scales).unwrap();
    for center in [true, false] {
        let actual = a.norm(&scale, 1e-5, center).unwrap().to_f32().unwrap();
        for r in 0..3 {
            let row = &x[r * 37..(r + 1) * 37];
            let mean = if center {
                row.iter().map(|&x| x as f64).sum::<f64>() / 37.0
            } else {
                0.0
            };

            let variance = row.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / 37.0;
            for c in 0..37 {
                let expected = (row[c] as f64 - mean) / (variance + 1e-5).sqrt() * scales[c] as f64;
                assert!((actual[r * 37 + c] as f64 - expected).abs() < 1e-6);
            }
        }
    }

    assert!(Array::from_f32(&device, 0, 37, &[]).is_err());
    assert!(a.slice(usize::MAX, 1, 0, 1).is_err());
    assert!(a.add(&b).is_err());
    assert!(a.norm(&b, 1e-5, false).is_err());
}

/// Φ(x) in f64 by Simpson's rule over the normal density from 0, far finer than FP32.
fn normal_cdf(x: f64) -> f64 {
    // Past 10 it is 0 or 1 to far below f64's precision near them.
    if x.abs() > 10.0 {
        return (x > 0.0) as u8 as f64;
    }
    let steps = 20_000;
    let h = x / steps as f64;
    let density = |t: f64| (-t * t / 2.0).exp();
    let inner: f64 = (1..steps)
        .map(|i| density(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 })
        .sum();
    0.5 + (density(0.0) + inner + density(x)) * h / 3.0 / (2.0 * std::f64::consts::PI).sqrt()
}

#[test]
fn softmax_and_transpose_cover_rows_wider_than_a_threadgroup() {
    let device = Device::new().unwrap();
    let (rows, cols) = (3, 601);
    let x = values(rows * cols, 7);
    let a = Array::from_f32(&device, rows, cols, &x).unwrap();
    let softmax = a.softmax(2.5).unwrap().to_f32().unwrap();
    for (row, actual) in x.chunks(cols).zip(softmax.chunks(cols)) {
        let exponentials: Vec<f64> = row.iter().map(|&v| (2.5 * v as f64).exp()).collect();
        let sum: f64 = exponentials.iter().sum();
        for (&value, exponential) in actual.iter().zip(&exponentials) {
            assert!((value as f64 - exponential / sum).abs() < 1e-8);
        }
    }

    let transposed = a.transpose().unwrap();
    assert_eq!(transposed.shape(), [cols, rows]);
    let transposed = transposed.to_f32().unwrap();
    for row in 0..rows {
        for col in 0..cols {
            assert_eq!(transposed[col * rows + row], x[row * cols + col]);
        }
    }
}

#[test]
fn gelu_matches_f64_reference_on_both_tails() {
    let device = Device::new().unwrap();
    // Fine steps over ±10, and the far tails, where tanh and the error function saturate.
    let x: Vec<f32> = (0..801)
        .map(|i| (i as f32 - 400.0) / 40.0)
        .chain([-1e5, -1e3, -50.0, 50.0, 1e3, 1e5])
        .collect();
    let a = Array::from_f32(&device, 1, x.len(), &x).unwrap();
    let exact = a.gelu(false).unwrap().to_f32().unwrap();
    let approximate = a.gelu(true).unwrap().to_f32().unwrap();
    for (i, &v) in x.iter().enumerate() {
        let v = v as f64;
        let expected = v * normal_cdf(v);
        // Relative to the value, with an absolute floor where GELU nears 0 in the negative tail.
        let tolerance = 1e-6 * expected.abs().max(1e-3);
        assert!(
            (exact[i] as f64 - expected).abs() <= tolerance,
            "GELU({v}) = {} != {expected}",
            exact[i]
        );
        let tanh = 0.5
            * v
            * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh());
        // The approximation cancels in FP32 where tanh nears −1, as it does on CUDA.
        assert!(
            (approximate[i] as f64 - tanh).abs() <= 1e-6 * tanh.abs().max(0.3),
            "tanh GELU({v}) = {} != {tanh}",
            approximate[i]
        );
    }
}

#[test]
fn batches_retain_dropped_inputs_and_recycle_only_completed_buffers() {
    let device = Device::new().unwrap();
    let mut retained = Vec::new();
    for i in 0..160 {
        let data = vec![i as f32; 257];
        let input = Array::from_f32(&device, 1, 257, &data).unwrap();
        let output = input.unary(0, 2.0).unwrap().unary(0, 0.25).unwrap();
        drop(input);
        retained.push(output);
    }

    device.synchronize().unwrap();
    assert!(device.stats().command_buffers < 10);
    for (i, output) in retained.iter().enumerate() {
        assert_eq!(output.to_f32().unwrap(), vec![i as f32 * 0.5; 257]);
    }

    drop(retained);
    for i in 0..64 {
        let data = vec![i as f32; 257];
        let input = Array::from_f32(&device, 1, 257, &data).unwrap();
        assert_eq!(
            input.unary(0, 2.0).unwrap().to_f32().unwrap(),
            vec![i as f32 * 2.0; 257]
        );
    }

    assert!(device.stats().buffer_reuses > 0);
}
