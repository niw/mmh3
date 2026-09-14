//! Compares the CUDA VSA with the f64 reference in mmh3-core on random BF16 heads in the DiT's qkv
//! layout.

use mmh3_core::dit::layout::PackedLayout;
use mmh3_core::dit::vsa::{VsaPlan, reference};
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::attention::{self, AttentionLayout, AttentionOffsets, HEAD_DIM, VsaWorkspace};

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

fn upload(values: &[u16]) -> DeviceBuffer {
    let mut buffer = DeviceBuffer::new(values.len() * 2).unwrap();
    buffer
        .copy_from_host(
            &values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    buffer
}

fn check(layout: &PackedLayout, heads: usize, sparsity: f64, gated: bool) {
    let plan = VsaPlan::for_layout(layout);
    let tokens = layout.len();
    let inner = heads * HEAD_DIM;
    let kept = plan.kept_video_tiles(sparsity);
    let mut random = Random(tokens as u64 + heads as u64);
    // The tokens of a tile share a random direction, so pooled scores spread.
    let directions: Vec<f32> = (0..plan.tiles() * 3 * inner)
        .map(|_| random.uniform(-1.0, 1.0))
        .collect();
    let mut tile_of = vec![0; tokens];
    for (tile, (&start, &length)) in plan.tile_starts.iter().zip(&plan.tile_lengths).enumerate() {
        tile_of[start..start + length].fill(tile);
    }
    let mut qkv = vec![0u16; tokens * 3 * inner];
    for token in 0..tokens {
        for column in 0..3 * inner {
            let value = directions[tile_of[token] * 3 * inner + column] + random.uniform(-1.0, 1.0);
            qkv[token * 3 * inner + column] = f32_to_bf16(value);
        }
    }
    let gate: Vec<u16> = (0..tokens * inner)
        .map(|_| f32_to_bf16(random.uniform(-2.0, 2.0)))
        .collect();
    let part = |offset: usize| -> Vec<f32> {
        (0..tokens * inner)
            .map(|index| bf16_to_f32(qkv[(index / inner) * 3 * inner + offset + index % inner]))
            .collect()
    };
    let gate_f32: Vec<f32> = gate.iter().map(|&value| bf16_to_f32(value)).collect();
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let expected = reference(
        &part(0),
        &part(inner),
        &part(2 * inner),
        gated.then_some(gate_f32.as_slice()),
        &plan,
        heads,
        HEAD_DIM,
        scale,
        kept,
    );

    let input = upload(&qkv);
    let gate_buffer = upload(&gate);
    let mut output = DeviceBuffer::new(tokens * inner * 2).unwrap();
    let attention_layout = AttentionLayout {
        token_stride: [
            (3 * inner) as i64,
            (3 * inner) as i64,
            (3 * inner) as i64,
            inner as i64,
        ],
        head_stride: [HEAD_DIM as i64; 4],
        ..AttentionLayout::default()
    };
    let workspace = VsaWorkspace::new(&plan, tokens, heads).unwrap();
    let offsets = AttentionOffsets {
        query: 0,
        key: inner,
        value: 2 * inner,
        output: 0,
    };
    attention::vsa(
        &input,
        gated.then_some(&gate_buffer),
        &mut output,
        offsets,
        &attention_layout,
        scale,
        kept,
        &workspace,
    )
    .unwrap();
    let fraction = workspace.selected_fraction().unwrap();
    let mut bytes = vec![0u8; tokens * inner * 2];
    output.copy_to_host(&mut bytes).unwrap();
    let actual: Vec<f32> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| bf16_to_f32(u16::from_le_bytes(pair)))
        .collect();

    let (prefix, video) = (plan.prefix_tiles, plan.video_tiles());
    let expected_fraction = if kept >= video {
        1.0
    } else {
        (prefix * plan.tiles() + video * (prefix + kept)) as f64
            / (plan.tiles() * plan.tiles()) as f64
    };
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
        "{tokens} tokens, {heads} heads, {} tiles, kept {kept} of {video} video tiles, gated {gated}: selected {fraction:.3}, relative error {:.3e}, max error {worst:.3e}",
        plan.tiles(),
        difference / norm
    );
    assert!(
        (fraction - expected_fraction).abs() < 1e-9,
        "selected fraction {fraction} against {expected_fraction}"
    );
    assert!(
        difference / norm < 0.01 && worst < 0.05,
        "relative error {}, max error {worst}",
        difference / norm
    );
}

#[test]
fn matches_the_reference() {
    // 150 prefix tokens, then a 9 × 8 × 12 patch grid in 3 × 2 × 3 cubes, some of them partial.
    let layout = PackedLayout::text_to_video(70, 9, 16, 24, 40);
    check(&layout, 2, 0.7, false);
    check(&layout, 3, 0.7, true);
    check(&layout, 2, 0.0, true);
}
