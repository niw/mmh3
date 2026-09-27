#![cfg(target_os = "linux")]
use mmh3_core::numeric::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::algorithms;
use mmh3_cuda::gemm::{self, Adapter, ColumnMaxima, Int8Tiling, Output};

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Every tiling that can run `n` outputs, and none for the one chosen for the shape.
fn every_tiling(n: usize) -> Vec<Option<Int8Tiling>> {
    let mut tilings = vec![None];
    for config in 0..gemm::int8_config_count() {
        if n.is_multiple_of(128 << config) {
            for group_m in gemm::int8_group_heights() {
                tilings.push(Some(Int8Tiling { config, group_m }));
            }
        }
    }
    tilings
}

/// How a run copies its tiles and multiplies them.
#[derive(Clone, Copy, Debug)]
struct Path {
    tma: bool,
    wgmma: bool,
}

impl Path {
    /// Runs `run` on this path, and leaves the thread on the default one after it.
    fn run<T>(self, run: impl FnOnce() -> T) -> T {
        mmh3_cuda::use_tma_on_this_thread(self.tma);
        mmh3_cuda::use_wgmma_on_this_thread(self.wgmma);
        let result = run();
        mmh3_cuda::use_tma_on_this_thread(true);
        mmh3_cuda::use_wgmma_on_this_thread(true);
        result
    }
}

fn is_hopper() -> bool {
    let device = mmh3_cuda::current_device().unwrap();
    mmh3_cuda::device_info(device).unwrap().compute_capability == (9, 0)
}

/// Every tiling of `every_tiling` on every path this card can take: on an sm_90 card wgmma first,
/// then mma.sync with TMA copies, and the cp.async copies of a card without TMA (sm_89).
fn every_tiling_and_path(n: usize) -> Vec<(Option<Int8Tiling>, Path)> {
    let mut paths = Vec::new();
    if is_hopper() {
        paths.push(Path {
            tma: true,
            wgmma: true,
        });
    }
    paths.push(Path {
        tma: true,
        wgmma: false,
    });
    paths.push(Path {
        tma: false,
        wgmma: false,
    });
    paths
        .into_iter()
        .flat_map(|path| {
            every_tiling(n)
                .into_iter()
                .map(move |tiling| (tiling, path))
        })
        .collect()
}

/// Runs every tiling on every path on random operands, with a random adapter of `rank` when it is
/// not zero, compares the output with an f64 reference within BF16 rounding, and checks that every
/// run gives the same bits. The wgmma and mma.sync runs add an adapter's BF16 products in their own
/// orders, so with an adapter each only has to agree with itself. The adapter's down rows lie
/// `down_stride` values apart, with values the kernel must skip in between. `f16_with_bias`
/// switches to FP16 output with a random bias.
fn check_every_config(
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    down_stride: usize,
    f16_with_bias: bool,
    seed: u64,
) {
    let mut random = Random(seed);
    let activations: Vec<i8> = (0..m * k).map(|_| random.int8()).collect();
    let weights: Vec<i8> = (0..n * k).map(|_| random.int8()).collect();
    let activation_scales: Vec<f32> = (0..m).map(|_| random.uniform(0.5, 1.5)).collect();
    // Small weight scales keep the INT8 product about as large as the adapter's contribution.
    let weight_scales: Vec<f32> = (0..n).map(|_| random.uniform(0.0001, 0.001)).collect();
    let down: Vec<u16> = (0..m * down_stride)
        .map(|index| {
            if index % down_stride < rank {
                random.bf16(-2.0, 2.0)
            } else {
                f32_to_bf16(1000.0)
            }
        })
        .collect();
    let up: Vec<u16> = (0..n * rank).map(|_| random.bf16(-2.0, 2.0)).collect();
    let adapter_scale = 1.5f32;
    let bias: Vec<f32> = (0..n)
        .map(|_| {
            if f16_with_bias {
                random.uniform(-20.0, 20.0)
            } else {
                0.0
            }
        })
        .collect();

    let activation_buffer = upload(
        &activations
            .iter()
            .map(|&value| value as u8)
            .collect::<Vec<_>>(),
    );
    let weight_buffer = upload(&weights.iter().map(|&value| value as u8).collect::<Vec<_>>());
    let activation_scale_buffer = upload(&f32_bytes(&activation_scales));
    let weight_scale_buffer = upload(&f32_bytes(&weight_scales));
    let (down_buffer, up_buffer) = (upload(&u16_bytes(&down)), upload(&u16_bytes(&up)));
    let adapter = Adapter {
        down: &down_buffer,
        up: &up_buffer,
        rank,
        down_stride,
        scale: adapter_scale,
    };
    let bias_buffer = upload(&f32_bytes(&bias));

    let expected: Vec<f64> = (0..m)
        .flat_map(|row| (0..n).map(move |column| (row, column)))
        .map(|(row, column)| {
            let dot: i32 = (0..k)
                .map(|index| {
                    activations[row * k + index] as i32 * weights[column * k + index] as i32
                })
                .sum();
            let low_rank: f64 = (0..rank)
                .map(|index| {
                    bf16_to_f32(down[row * down_stride + index]) as f64
                        * bf16_to_f32(up[column * rank + index]) as f64
                })
                .sum();
            dot as f64 * activation_scales[row] as f64 * weight_scales[column] as f64
                + adapter_scale as f64 * low_rank
                + bias[column] as f64
        })
        .collect();

    // BF16 output also reports the maxima of the groups of 128 columns after the first.
    let first_column = 128;
    let groups = if f16_with_bias {
        0
    } else {
        (n - first_column) / 128
    };
    // The first output of the wgmma runs and of the mma.sync ones.
    let mut first_outputs: [Option<Vec<u8>>; 2] = [None, None];
    for (tiling, path) in every_tiling_and_path(n) {
        let mut output = DeviceBuffer::new(m * n * 2).unwrap();
        let maxima = DeviceBuffer::zeroed(groups.max(1) * 4).unwrap();
        path.run(|| {
            gemm::int8(
                tiling,
                &activation_buffer,
                &weight_buffer,
                &activation_scale_buffer,
                &weight_scale_buffer,
                Output {
                    buffer: &mut output,
                    f16: f16_with_bias,
                    bias: f16_with_bias.then_some(&bias_buffer),
                    swiglu: false,
                    column_maxima: (!f16_with_bias).then_some(ColumnMaxima {
                        values: &maxima,
                        first_column,
                    }),
                },
                m,
                n,
                k,
                (rank > 0).then_some(&adapter),
            )
        })
        .unwrap();
        let mut output_bytes = vec![0; m * n * 2];
        output.copy_to_host(&mut output_bytes).unwrap();
        let mut expected_maxima = vec![0.0f32; groups];
        for (index, pair) in output_bytes.as_chunks::<2>().0.iter().enumerate() {
            let column = index % n;
            if column >= first_column && groups > 0 {
                let group = (column - first_column) / 128;
                let magnitude = bf16_to_f32(u16::from_le_bytes(*pair)).abs();
                expected_maxima[group] = expected_maxima[group].max(magnitude);
            }
        }
        assert_eq!(
            maxima.to_f32().unwrap()[..groups],
            expected_maxima,
            "{m} × {n} × {k}, rank {rank}, {tiling:?}, {path:?}: column maxima"
        );

        for (index, &expected) in expected.iter().enumerate() {
            let bits = u16::from_le_bytes([output_bytes[index * 2], output_bytes[index * 2 + 1]]);
            let actual = if f16_with_bias {
                f16_to_f32(bits)
            } else {
                bf16_to_f32(bits)
            } as f64;
            let tolerance = expected.abs() / 256.0 + 1e-4;
            assert!(
                (actual - expected).abs() <= tolerance,
                "{m} × {n} × {k}, rank {rank}, {tiling:?}, {path:?}, row {}, column {}: expected {expected}, got {actual}",
                index / n,
                index % n
            );
        }
        // Without an adapter every product is an integer and the FP32 steps are the same, so the
        // two families agree too.
        let family = usize::from(!path.wgmma);
        let compared = if rank == 0 {
            first_outputs.iter().flatten().next()
        } else {
            first_outputs[family].as_ref()
        };
        if let Some(first) = compared {
            assert!(
                *first == output_bytes,
                "{m} × {n} × {k}, rank {rank}: {tiling:?} on {path:?} differs from the first run"
            );
        }
        if first_outputs[family].is_none() {
            first_outputs[family] = Some(output_bytes);
        }
    }
}

#[test]
fn matches_cpu_reference_for_every_config() {
    check_every_config(300, 512, 384, 0, 0, false, 7);
}

#[test]
fn matches_cpu_reference_over_several_tiles_per_block() {
    // 192 tiles, more than a Blackwell GPU has SMs, with a ragged last row of tiles and an odd
    // number of K blocks.
    check_every_config(2000, 3072, 384, 0, 0, false, 8);
}

#[test]
fn adds_a_low_rank_adapter() {
    check_every_config(300, 512, 256, 128, 128, false, 9);
    // The rank of the fused qkv adapters, and more tiles than SMs.
    check_every_config(600, 3072, 256, 384, 384, false, 10);
}

#[test]
fn reads_adapter_rows_at_their_stride() {
    // The down projections of INT8 layers come out 128 columns wide for ranks of 64.
    check_every_config(300, 512, 256, 64, 128, false, 12);
}

#[test]
fn writes_fp16_with_a_bias() {
    check_every_config(600, 3072, 256, 0, 0, true, 11);
}

/// Rounds an f64 to BF16 or FP16 and back.
fn round_output(value: f64, f16: bool) -> f32 {
    if f16 {
        f16_to_f32(f32_to_f16(value as f32))
    } else {
        bf16_to_f32(f32_to_bf16(value as f32))
    }
}

#[test]
fn writes_swiglu_of_interleaved_rows() {
    let (m, features, k, rank) = (300, 1024, 256, 128);
    let n = 2 * features;
    for (f16, seed) in [(false, 12), (true, 13)] {
        let mut random = Random(seed);
        let activations: Vec<i8> = (0..m * k).map(|_| random.int8()).collect();
        let weights: Vec<i8> = (0..n * k).map(|_| random.int8()).collect();
        let activation_scales: Vec<f32> = (0..m).map(|_| random.uniform(0.5, 1.5)).collect();
        let weight_scales: Vec<f32> = (0..n).map(|_| random.uniform(0.00002, 0.0001)).collect();
        let bias: Vec<f32> = (0..n).map(|_| random.uniform(-2.0, 2.0)).collect();
        let down: Vec<u16> = (0..m * rank).map(|_| random.bf16(-0.5, 0.5)).collect();
        let up: Vec<u16> = (0..n * rank).map(|_| random.bf16(-0.5, 0.5)).collect();
        let adapter_scale = 0.75f32;

        let interleave = |buffer: DeviceBuffer, row_bytes: usize| {
            gemm::interleave_swiglu_rows(&buffer, n, row_bytes).unwrap()
        };
        let activation_buffer = upload(
            &activations
                .iter()
                .map(|&value| value as u8)
                .collect::<Vec<_>>(),
        );
        let weight_buffer = interleave(
            upload(&weights.iter().map(|&value| value as u8).collect::<Vec<_>>()),
            k,
        );
        let activation_scale_buffer = upload(&f32_bytes(&activation_scales));
        let weight_scale_buffer = interleave(upload(&f32_bytes(&weight_scales)), 4);
        let bias_buffer = interleave(upload(&f32_bytes(&bias)), 4);
        let down_buffer = upload(&u16_bytes(&down));
        let up_buffer = interleave(upload(&u16_bytes(&up)), rank * 2);
        let adapter = Adapter {
            down: &down_buffer,
            up: &up_buffer,
            rank,
            down_stride: rank,
            scale: adapter_scale,
        };

        // The runs of each family, wgmma and mma.sync, which add the adapter in their own orders.
        let mut families: [Vec<_>; 2] = [Vec::new(), Vec::new()];
        for (tiling, path) in every_tiling_and_path(n) {
            let mut output = DeviceBuffer::new(m * features * 2).unwrap();
            path.run(|| {
                gemm::int8(
                    tiling,
                    &activation_buffer,
                    &weight_buffer,
                    &activation_scale_buffer,
                    &weight_scale_buffer,
                    Output {
                        buffer: &mut output,
                        f16,
                        bias: Some(&bias_buffer),
                        swiglu: true,
                        column_maxima: None,
                    },
                    m,
                    n,
                    k,
                    Some(&adapter),
                )
            })
            .unwrap();
            let mut output_bytes = vec![0; m * features * 2];
            output.copy_to_host(&mut output_bytes).unwrap();
            families[usize::from(!path.wgmma)].push((tiling, path, output_bytes));
        }
        for runs in &families {
            for (tiling, path, other) in runs.iter().skip(1) {
                assert!(
                    *other == runs[0].2,
                    "f16 {f16}: {tiling:?} on {path:?} differs from the first run"
                );
            }
        }

        let product = |row: usize, column: usize| -> f32 {
            let dot: i32 = (0..k)
                .map(|index| {
                    activations[row * k + index] as i32 * weights[column * k + index] as i32
                })
                .sum();
            let low_rank: f64 = (0..rank)
                .map(|index| {
                    bf16_to_f32(down[row * rank + index]) as f64
                        * bf16_to_f32(up[column * rank + index]) as f64
                })
                .sum();
            let value = dot as f64 * activation_scales[row] as f64 * weight_scales[column] as f64
                + adapter_scale as f64 * low_rank
                + bias[column] as f64;
            round_output(value, f16)
        };
        for (_, path, output_bytes) in families.iter().filter_map(|runs| runs.first()) {
            for row in 0..m {
                for feature in 0..features {
                    let (gate, up) = (product(row, feature), product(row, features + feature));
                    let expected =
                        round_output((gate / (1.0 + (-gate).exp()) * up) as f64, f16) as f64;
                    let index = row * features + feature;
                    let bits =
                        u16::from_le_bytes([output_bytes[index * 2], output_bytes[index * 2 + 1]]);
                    let actual = if f16 {
                        f16_to_f32(bits)
                    } else {
                        bf16_to_f32(bits)
                    } as f64;
                    assert!(
                        (actual - expected).abs() <= expected.abs() / 64.0 + 1e-3,
                        "f16 {f16}, {path:?}, row {row}, feature {feature}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }
}

#[test]
fn keeps_the_chosen_tilings() {
    // A shape no other test runs, with more rows of tiles than the shortest band.
    let (m, n, k) = (1300, 1024, 256);
    let mut random = Random(14);
    let activations: Vec<u8> = (0..m * k).map(|_| random.int8() as u8).collect();
    let weights: Vec<u8> = (0..n * k).map(|_| random.int8() as u8).collect();
    let mut output = DeviceBuffer::new(m * n * 2).unwrap();
    gemm::int8(
        None,
        &upload(&activations),
        &upload(&weights),
        &upload(&f32_bytes(&vec![1.0; m])),
        &upload(&f32_bytes(&vec![0.001; n])),
        Output {
            buffer: &mut output,
            f16: false,
            bias: None,
            swiglu: false,
            column_maxima: None,
        },
        m,
        n,
        k,
        None,
    )
    .unwrap();

    let directory = std::env::temp_dir().join(format!("mmh3-int8-{}", std::process::id()));
    let path = directory.join("tilings.txt");
    assert!(algorithms::save_int8(&path).unwrap());
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        text.lines()
            .any(|line| line.starts_with(&format!("{m} {n} {k} 0 ")))
    );
    let saved = text.lines().count() - 2;
    assert!(algorithms::load_int8(&path).unwrap() >= saved);

    std::fs::write(&path, text.replacen("format", "other format", 1)).unwrap();
    assert_eq!(algorithms::load_int8(&path).unwrap(), 0);
    std::fs::remove_dir_all(&directory).unwrap();
}
