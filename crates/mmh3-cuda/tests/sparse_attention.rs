//! Compares the CUDA Sol-Attn with the f64 reference in mmh3-core on random BF16 heads in the DiT's
//! qkv layout.
#![cfg(target_os = "linux")]

use mmh3_core::dit::sparse::{SPARSE_BLOCK, SparseSinks, reference};
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::attention::{
    self, AttentionInputs, AttentionLayout, AttentionOffsets, AttentionPrecision, HEAD_DIM,
    SparseWorkspace,
};

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

fn check(tokens: usize, heads: usize, tau: f32, sinks: SparseSinks, precision: AttentionPrecision) {
    let inner = heads * HEAD_DIM;
    let mut random = Random(tokens as u64);
    // Every block of keys and queries shares a random direction, so pooled scores spread and
    // routing is selective.
    let blocks = tokens.div_ceil(SPARSE_BLOCK);
    let directions: Vec<f32> = (0..blocks * 3 * inner)
        .map(|_| random.uniform(-1.0, 1.0))
        .collect();
    let mut qkv = vec![0u16; tokens * 3 * inner];
    for token in 0..tokens {
        for column in 0..3 * inner {
            let value =
                directions[(token / SPARSE_BLOCK) * 3 * inner + column] + random.uniform(-1.0, 1.0);
            qkv[token * 3 * inner + column] = f32_to_bf16(value);
        }
    }
    let part = |offset: usize| -> Vec<f32> {
        (0..tokens * inner)
            .map(|index| bf16_to_f32(qkv[(index / inner) * 3 * inner + offset + index % inner]))
            .collect()
    };
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let (expected, expected_fraction) = reference(
        &part(0),
        &part(inner),
        &part(2 * inner),
        tokens,
        heads,
        HEAD_DIM,
        tau,
        scale,
        sinks,
    );

    let input = {
        let mut buffer = DeviceBuffer::new(qkv.len() * 2).unwrap();
        buffer
            .copy_from_host(
                &qkv.iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        buffer
    };
    let mut output = DeviceBuffer::new(tokens * inner * 2).unwrap();
    let layout = AttentionLayout {
        token_stride: [
            (3 * inner) as i64,
            (3 * inner) as i64,
            (3 * inner) as i64,
            inner as i64,
        ],
        head_stride: [HEAD_DIM as i64; 4],
        ..AttentionLayout::default()
    };
    let workspace = SparseWorkspace::with_precision(tokens, heads, precision).unwrap();
    let offsets = AttentionOffsets {
        query: 0,
        key: inner,
        value: 2 * inner,
        output: 0,
    };
    attention::sparse(
        &input,
        &mut output,
        offsets,
        tokens,
        heads,
        &layout,
        scale,
        tau,
        sinks,
        &workspace,
        AttentionInputs::Raw,
    )
    .unwrap();
    let fraction = workspace.routed_fraction().unwrap();
    let mut bytes = vec![0u8; tokens * inner * 2];
    output.copy_to_host(&mut bytes).unwrap();
    let actual: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| bf16_to_f32(u16::from_le_bytes(pair)))
        .collect();

    let difference: f64 = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = expected
        .iter()
        .map(|&value| (value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let worst = actual
        .iter()
        .zip(&expected)
        .fold(0.0f32, |maximum, (&a, &b)| maximum.max((a - b).abs()));
    eprintln!(
        "{tokens} tokens, {heads} heads, tau {tau}: routed {fraction:.3} (reference {expected_fraction:.3}), relative error {:.3e}, max error {worst:.3e}",
        difference / norm
    );
    assert!(
        (fraction - expected_fraction).abs() < 0.01,
        "routed fraction {fraction} against {expected_fraction}"
    );
    assert!(
        fraction > 0.1 && fraction < 0.9,
        "routing is not selective: {fraction}"
    );
    assert!(
        difference / norm < 0.01 && worst < 0.05,
        "relative error {}, max error {worst}",
        difference / norm
    );
}

#[test]
fn matches_the_reference() {
    check(
        1000,
        2,
        0.5,
        SparseSinks {
            key_blocks: (0, 2),
            query_blocks: (1, 2),
        },
        AttentionPrecision::Bf16,
    );
    check(
        777,
        3,
        1.3,
        SparseSinks {
            key_blocks: (0, 1),
            query_blocks: (0, 0),
        },
        AttentionPrecision::Bf16,
    );
}

#[test]
fn quantized_matches_the_same_reference_and_error_bounds() {
    check(
        1000,
        2,
        0.5,
        SparseSinks {
            key_blocks: (0, 2),
            query_blocks: (1, 2),
        },
        AttentionPrecision::Int8Fp8,
    );
    check(
        777,
        3,
        1.3,
        SparseSinks {
            key_blocks: (0, 1),
            query_blocks: (0, 0),
        },
        AttentionPrecision::Int8Fp8,
    );
}
