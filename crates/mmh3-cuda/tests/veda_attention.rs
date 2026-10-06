//! Compares the CUDA Veda with the reference in mmh3-core on BF16 heads in the DiT's qkv layout.
#![cfg(target_os = "linux")]

use mmh3_core::dit::layout::{KeyframeShape, PackedLayout};
use mmh3_core::dit::veda::{
    self, Projection, TilePlan, TileShape, VedaPredictor, VedaTiling, column_blocks,
};
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::attention::{AttentionLayout, AttentionOffsets, HEAD_DIM};
use mmh3_cuda::veda::{VedaPass, VedaWeights};

/// A predictor of two blocks of three heads, which share a shape in the first block and not in the
/// second, with FP8 projections of no infinities or NaNs.
fn predictor() -> VedaPredictor {
    let (layers, heads) = (2, 3);
    let projection = |seed: usize| Projection {
        fp8: (0..heads * 3 * HEAD_DIM * HEAD_DIM)
            .map(|index| {
                let bits = (index * 2_654_435_761 + seed * 40_503) >> 7;
                (bits % 0x60) as u8 | if bits & 0x100 != 0 { 0x80 } else { 0 }
            })
            .collect(),
        scales: (0..heads).map(|head| 0.002 + 0.001 * head as f32).collect(),
    };
    VedaPredictor {
        layers,
        heads,
        head_dim: HEAD_DIM,
        keep_ratio: 0.5,
        plans: vec![TilePlan {
            name: "test".into(),
            grid: [5, 8, 12],
            shapes: vec![
                TileShape::new(4, 4, 8).unwrap(),
                TileShape::new(2, 8, 8).unwrap(),
            ],
            head_shape: vec![vec![1, 1, 1], vec![0, 1, 0]],
        }],
        projections: (0..layers)
            .map(|layer| [projection(2 * layer), projection(2 * layer + 1)])
            .collect(),
    }
}

/// BF16 values that spread the scores.
fn values(tokens: usize, inner: usize, seed: usize) -> Vec<f32> {
    (0..tokens * inner)
        .map(|index| {
            let token = index / inner;
            let wave = ((index * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5;
            bf16_to_f32(f32_to_bf16(
                wave * 2.5 + (token as f32 / 37.0 + seed as f32).sin(),
            ))
        })
        .collect()
}

#[test]
fn matches_the_reference() {
    let predictor = predictor();
    let weights = VedaWeights::new(&predictor).unwrap();
    let heads = predictor.heads;
    let inner = heads * HEAD_DIM;
    // 70 text tokens, 2 × 40 audio rows and a video grid of 5 × 8 × 12 patches, alone and after a
    // keyframe, whose tiles the reference block keeps.
    let keyframe = [KeyframeShape {
        frame_index: 0,
        latent_frames: 1,
        audio_frames: 0,
    }];
    for keyframes in [&[][..], &keyframe[..]] {
        let layout = PackedLayout::new(70, 5, 16, 24, 40, keyframes, &[]);
        let tokens = layout.len();
        let (query, key, value) = (
            values(tokens, inner, 1),
            values(tokens, inner, 2),
            values(tokens, inner, 3),
        );
        let qkv: Vec<u8> = (0..tokens)
            .flat_map(|token| {
                [&query, &key, &value]
                    .into_iter()
                    .flat_map(move |part| part[token * inner..(token + 1) * inner].iter())
            })
            .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
            .collect();
        let input = DeviceBuffer::from_bytes(&qkv).unwrap();
        let mut output = DeviceBuffer::new(tokens * inner * 2).unwrap();
        let layout_strides = AttentionLayout {
            token_stride: [
                (3 * inner) as i64,
                (3 * inner) as i64,
                (3 * inner) as i64,
                inner as i64,
            ],
            head_stride: [HEAD_DIM as i64; 4],
            ..AttentionLayout::default()
        };
        let offsets = AttentionOffsets {
            query: 0,
            key: inner,
            value: 2 * inner,
            output: 0,
        };
        let plan = &predictor.plans[0];
        let tilings: Vec<Option<VedaTiling>> = plan
            .shapes
            .iter()
            .map(|&shape| Some(VedaTiling::new(&layout, shape, true)))
            .collect();
        let pass = VedaPass::new(&weights, &layout, 0.5, 0.5).unwrap();
        let mut kept = 0;
        let mut pairs = 0;
        for layer in 0..predictor.layers {
            let (expected, kept_here) = veda::reference(
                &query,
                &key,
                &value,
                heads,
                HEAD_DIM,
                &plan.head_shape[layer],
                &tilings,
                &predictor.projections[layer],
                0.5,
                0.5,
            );
            kept += kept_here;
            pairs += plan.head_shape[layer]
                .iter()
                .map(|&shape| tilings[shape].as_ref().unwrap().video_tiles.pow(2))
                .sum::<usize>();
            pass.attend(
                &weights,
                &input,
                &mut output,
                offsets,
                &layout_strides,
                layer,
            )
            .unwrap();
            let mut bytes = vec![0u8; tokens * inner * 2];
            output.copy_to_host(&mut bytes).unwrap();
            let worst = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| bf16_to_f32(u16::from_le_bytes(pair)))
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            eprintln!(
                "{} keyframes, block {layer}: max error {worst:.3e}",
                keyframes.len()
            );
            assert!(
                worst < 1e-2,
                "{} keyframes, block {layer}: {worst} from the reference",
                keyframes.len()
            );
        }
        let fraction = pass.kept_fraction().unwrap().unwrap();
        assert_eq!((fraction * pairs as f64).round() as usize, kept);
        assert!(
            fraction < 0.9,
            "kept {fraction} of the tiles, which tests little"
        );
        // The reference block keeps its own budget.
        let tiling = tilings[1].as_ref().unwrap();
        assert_eq!(
            column_blocks(tiling, 0.5, 0.5).len(),
            if keyframes.is_empty() { 1 } else { 2 }
        );
    }
}
