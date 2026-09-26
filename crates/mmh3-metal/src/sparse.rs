//! Block-sparse attention over the DiT's heads of 128: Sol-Attn and VSA, as mmh3-core's
//! `dit::sparse` and `dit::vsa` describe them. ops.metal says what each kernel does.

use crate::{
    Buffer, Device, Error, Result,
    ops::{Array, AttentionInputs},
};
use mmh3_core::dit::{
    sparse::{SPARSE_BLOCK, SparseSinks},
    vsa::VsaPlan,
};

const HEAD: usize = 128;
/// A query tile's tail row: 128 values, and for Sol-Attn the tail's maximum and sum.
const TAIL: usize = HEAD + 2;
/// The most tiles a routing kernel scores in threadgroup memory, `SPARSE_MAX_TILES` in ops.metal.
const MAX_TILES: usize = 4096;

/// How a query tile picks the tiles it attends token by token.
#[derive(Clone, Copy, Debug)]
enum Routing {
    /// Sol-Attn over 64-token blocks, with its threshold in standard deviations.
    Sol { tau: f32, sinks: SparseSinks },
    /// VSA: every tile before the video, and the `kept` best video tiles for a video query tile.
    Vsa { prefix_tiles: usize, kept: usize },
}

/// The tiles of one call's sequence and the memory its blocks route them in, which every block of
/// the call reuses.
pub struct SparsePass {
    routing: Routing,
    tokens: usize,
    heads: usize,
    tiles: usize,
    starts: Buffer,
    lengths: Buffer,
    pooled_queries: Buffer,
    pooled_keys: Buffer,
    pooled_values: Buffer,
    key_mean: Buffer,
    key_variance: Buffer,
    routes: Buffer,
    counts: Buffer,
    tails: Buffer,
    /// Tiles Sol-Attn routed exactly, over the blocks of the call.
    routed: Buffer,
    blocks: std::cell::Cell<usize>,
}

impl SparsePass {
    /// Sol-Attn over `tokens` rows of `heads` heads, in 64-token blocks.
    pub fn sol(
        device: &Device,
        tokens: usize,
        heads: usize,
        tau: f32,
        sinks: SparseSinks,
    ) -> Result<Self> {
        let starts: Vec<usize> = (0..tokens).step_by(SPARSE_BLOCK).collect();
        let lengths: Vec<usize> = starts
            .iter()
            .map(|&start| (tokens - start).min(SPARSE_BLOCK))
            .collect();
        Self::new(
            device,
            Routing::Sol { tau, sinks },
            tokens,
            heads,
            &starts,
            &lengths,
        )
    }

    /// VSA over a sequence of `heads` heads already in `plan`'s order, keeping `kept` video tiles.
    pub fn vsa(device: &Device, plan: &VsaPlan, heads: usize, kept: usize) -> Result<Self> {
        let tokens = plan
            .tile_starts
            .iter()
            .zip(&plan.tile_lengths)
            .map(|(start, length)| start + length)
            .max()
            .unwrap_or(0);
        Self::new(
            device,
            Routing::Vsa {
                prefix_tiles: plan.prefix_tiles,
                kept,
            },
            tokens,
            heads,
            &plan.tile_starts,
            &plan.tile_lengths,
        )
    }

    fn new(
        device: &Device,
        routing: Routing,
        tokens: usize,
        heads: usize,
        starts: &[usize],
        lengths: &[usize],
    ) -> Result<Self> {
        let tiles = starts.len();
        if tiles == 0 || heads == 0 {
            return Err(Error::new("sparse attention over nothing".into()));
        }
        if tiles > MAX_TILES {
            return Err(Error::new(format!(
                "sparse attention takes up to {MAX_TILES} tiles of 64 tokens, not {tiles}"
            )));
        }
        let table = |values: &[usize]| -> Result<Buffer> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&value| (value as u32).to_ne_bytes())
                .collect();
            device.alloc(bytes.len(), Some(&bytes))
        };
        let floats = |count: usize| device.alloc(count * 4, None);
        Ok(Self {
            routing,
            tokens,
            heads,
            tiles,
            starts: table(starts)?,
            lengths: table(lengths)?,
            pooled_queries: floats(heads * tiles * HEAD)?,
            pooled_keys: floats(heads * tiles * HEAD)?,
            pooled_values: floats(heads * tiles * HEAD)?,
            key_mean: floats(heads * HEAD)?,
            key_variance: floats(heads * HEAD)?,
            routes: device.alloc(heads * tiles * tiles * 2, None)?,
            counts: device.alloc(heads * tiles * 4, None)?,
            tails: floats(heads * tiles * TAIL)?,
            routed: device.alloc(4, Some(&[0; 4]))?,
            blocks: std::cell::Cell::new(0),
        })
    }

    pub fn is_vsa(&self) -> bool {
        matches!(self.routing, Routing::Vsa { .. })
    }

    /// Attention of `inputs`, in the sequence's order and in their precision: FP16 on the matrix
    /// units or FP32. VSA adds `gate`, `[tokens, heads × 128]`, times its coarse output.
    pub fn attend(&self, inputs: &AttentionInputs, gate: Option<&Array>) -> Result<Array> {
        let width = self.heads * HEAD;
        if inputs.rows != self.tokens
            || inputs.heads != self.heads
            || inputs.dim != HEAD
            || gate.is_some_and(|gate| gate.shape() != [self.tokens, width])
        {
            return Err(Error::new(
                "sparse attention inputs do not match its sequence".into(),
            ));
        }
        if gate.is_some() && !self.is_vsa() {
            return Err(Error::new("only VSA takes a gate".into()));
        }
        let device = &inputs.query.0.device;
        let (tiles, heads) = (self.tiles as u32, self.heads as u32);
        let pairs = self.tiles * self.heads.div_ceil(2);
        device.run(
            "sparse_pool",
            &[
                &inputs.query,
                &inputs.key,
                &inputs.value,
                &self.starts,
                &self.lengths,
                &self.pooled_queries,
                &self.pooled_keys,
                &self.pooled_values,
            ],
            &[tiles, heads, inputs.half as u32],
            pairs,
            true,
        )?;
        let scale = (HEAD as f32).sqrt().recip();
        let scale_log2 = scale * std::f32::consts::LOG2_E;
        match self.routing {
            Routing::Sol { tau, sinks } => {
                device.run(
                    "sparse_center_keys",
                    &[&self.pooled_keys, &self.key_mean, &self.key_variance],
                    &[tiles, heads],
                    self.heads.div_ceil(2),
                    true,
                )?;
                device.run(
                    "sol_route",
                    &[
                        &self.pooled_queries,
                        &self.pooled_keys,
                        &self.pooled_values,
                        &self.key_variance,
                        &self.lengths,
                        &self.routes,
                        &self.counts,
                        &self.tails,
                        &self.routed,
                    ],
                    &[
                        tiles,
                        heads,
                        tau.to_bits(),
                        scale_log2.to_bits(),
                        sinks.key_blocks.0 as u32,
                        sinks.key_blocks.1 as u32,
                        sinks.query_blocks.0 as u32,
                        sinks.query_blocks.1 as u32,
                    ],
                    self.tiles * self.heads,
                    true,
                )?;
                self.blocks.set(self.blocks.get() + 1);
            }
            Routing::Vsa { prefix_tiles, kept } => device.run(
                "vsa_select",
                &[
                    &self.pooled_queries,
                    &self.pooled_keys,
                    &self.pooled_values,
                    &self.lengths,
                    &self.routes,
                    &self.counts,
                    &self.tails,
                ],
                &[
                    tiles,
                    heads,
                    prefix_tiles as u32,
                    kept as u32,
                    scale.to_bits(),
                    gate.is_some() as u32,
                ],
                self.tiles * self.heads,
                true,
            )?,
        }

        let out = Array::empty(device, self.tokens, width)?;
        let parameters = [
            tiles,
            heads,
            self.is_vsa() as u32,
            gate.is_some() as u32,
            scale_log2.to_bits(),
        ];
        let gate = gate.map_or(&self.starts, |gate| &gate.buffer);
        if let Some(quantized) = &inputs.quantized {
            let mut parameters = parameters.to_vec();
            parameters.push(self.tokens as u32);
            device.run(
                "mpp_sparse_attention_int8_128",
                &[
                    &inputs.query,
                    &inputs.key,
                    &inputs.value,
                    &out.buffer,
                    &self.starts,
                    &self.lengths,
                    &self.routes,
                    &self.counts,
                    &self.tails,
                    &self.key_mean,
                    gate,
                    &quantized.query,
                    &quantized.key,
                    &quantized.scales,
                ],
                &parameters,
                self.tiles * self.heads,
                true,
            )?;
        } else if inputs.half {
            device.run(
                "mpp_sparse_attention_128",
                &[
                    &inputs.query,
                    &inputs.key,
                    &inputs.value,
                    &out.buffer,
                    &self.starts,
                    &self.lengths,
                    &self.routes,
                    &self.counts,
                    &self.tails,
                    &self.key_mean,
                    gate,
                ],
                &parameters,
                self.tiles * self.heads,
                true,
            )?;
        } else {
            device.run(
                "attention_sparse_128",
                &[
                    &inputs.query,
                    &inputs.key,
                    &inputs.value,
                    &out.buffer,
                    &self.starts,
                    &self.lengths,
                    &self.routes,
                    &self.counts,
                    &self.tails,
                    &self.key_mean,
                    gate,
                ],
                &parameters,
                self.tiles * self.heads * SPARSE_BLOCK.div_ceil(8),
                true,
            )?;
        }
        Ok(out)
    }

    /// The mean fraction of (head, query block, key block) triples Sol-Attn routed exactly over
    /// the blocks attended so far, or None for VSA or before any block.
    pub fn routed_fraction(&self) -> Result<Option<f64>> {
        let blocks = self.blocks.get();
        if !matches!(self.routing, Routing::Sol { .. }) || blocks == 0 {
            return Ok(None);
        }
        let bytes = self.routed.to_bytes()?;
        let routed = u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes"));
        Ok(Some(
            routed as f64 / (blocks * self.heads * self.tiles * self.tiles) as f64,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmh3_core::dit::layout::PackedLayout;

    /// Values of a head's rows that spread their scores, so that routing has something to decide.
    fn inputs(tokens: usize, heads: usize, seed: usize) -> Vec<f32> {
        (0..tokens * heads * HEAD)
            .map(|index| {
                let token = index / (heads * HEAD);
                let wave = ((index * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5;
                wave * 2.5 + (token as f32 / 97.0).sin()
            })
            .collect()
    }

    fn largest_difference(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    /// Whether the inputs are FP16, and how far from the reference the output may be.
    fn precisions(device: &Device) -> Vec<(bool, f32)> {
        let mut precisions = vec![(false, 1e-4)];
        if device.supports_tensor_ops() {
            precisions.push((true, 5e-3));
        }
        precisions
    }

    /// Queries, keys and values as a block's qkv projection yields them, with INT8 queries and
    /// keys for the scores, and the FP32 values of what the attention reads, for a reference.
    fn quantized_inputs(
        device: &Device,
        tokens: usize,
        heads: usize,
        seeds: [usize; 3],
    ) -> (AttentionInputs, [Vec<f32>; 3]) {
        let inner = heads * HEAD;
        let [q, k, v] = seeds.map(|seed| inputs(tokens, heads, seed));
        let qkv: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                [&q, &k, &v]
                    .into_iter()
                    .flat_map(move |x| x[t * inner..(t + 1) * inner].iter().copied())
            })
            .collect();
        let qkv = Array::from_f32(device, tokens, 3 * inner, &qkv).unwrap();
        let ones = Array::from_f32(device, 1, HEAD, &[1.0; HEAD]).unwrap();
        let inputs = qkv
            .attention_inputs(heads, (&ones, &ones), 1e-5, None, true, false, true)
            .unwrap();
        let read = |buffer: &Buffer| -> Vec<f32> {
            buffer
                .to_bytes()
                .unwrap()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| mmh3_core::numeric::f16_to_f32(u16::from_le_bytes(*bytes)))
                .collect()
        };
        let values = [read(&inputs.query), read(&inputs.key), read(&inputs.value)];
        (inputs, values)
    }

    #[test]
    fn int8_scores_stay_near_the_references() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let scale = (HEAD as f32).sqrt().recip();
        let heads = 2;

        let layout = PackedLayout::text_to_video(100, 3, 16, 24, 20);
        let tokens = layout.len();
        let sinks = SparseSinks::for_layout(&layout);
        let (inputs, [q, k, v]) = quantized_inputs(&device, tokens, heads, [1, 2, 3]);
        let (expected, _) =
            mmh3_core::dit::sparse::reference(&q, &k, &v, tokens, heads, HEAD, 1.3, scale, sinks);
        let out = SparsePass::sol(&device, tokens, heads, 1.3, sinks)
            .unwrap()
            .attend(&inputs, None)
            .unwrap()
            .to_f32()
            .unwrap();
        let difference = largest_difference(&out, &expected);
        assert!(
            difference < 3e-2,
            "Sol-Attn: {difference} from the reference"
        );

        let layout = PackedLayout::text_to_video(70, 5, 12, 24, 40);
        let plan = VsaPlan::for_layout(&layout);
        let tokens = layout.len();
        let kept = plan.kept_video_tiles(0.5);
        let (inputs, [q, k, v]) = quantized_inputs(&device, tokens, heads, [4, 5, 6]);
        let expected =
            mmh3_core::dit::vsa::reference(&q, &k, &v, None, &plan, heads, HEAD, scale, kept);
        let out = SparsePass::vsa(&device, &plan, heads, kept)
            .unwrap()
            .attend(&inputs, None)
            .unwrap()
            .to_f32()
            .unwrap();
        let difference = largest_difference(&out, &expected);
        assert!(difference < 3e-2, "VSA: {difference} from the reference");
    }

    #[test]
    fn sol_attn_matches_the_reference() {
        let device = Device::new().unwrap();
        let layout = PackedLayout::text_to_video(100, 3, 16, 24, 20);
        let (tokens, heads) = (layout.len(), 2);
        let sinks = SparseSinks::for_layout(&layout);
        let (q, k, v) = (
            inputs(tokens, heads, 1),
            inputs(tokens, heads, 2),
            inputs(tokens, heads, 3),
        );
        let scale = (HEAD as f32).sqrt().recip();
        let (expected, routed) =
            mmh3_core::dit::sparse::reference(&q, &k, &v, tokens, heads, HEAD, 1.3, scale, sinks);
        assert!(routed < 1.0, "every block routed, which tests nothing");
        let upload = |x: &[f32]| Array::from_f32(&device, tokens, heads * HEAD, x).unwrap();
        for (half, tolerance) in precisions(&device) {
            let sparse = SparsePass::sol(&device, tokens, heads, 1.3, sinks).unwrap();
            let inputs =
                AttentionInputs::from_arrays(&upload(&q), &upload(&k), &upload(&v), heads, half)
                    .unwrap();
            let out = sparse.attend(&inputs, None).unwrap().to_f32().unwrap();
            let difference = largest_difference(&out, &expected);
            assert!(
                difference < tolerance,
                "FP16 {half}: {difference} from the reference"
            );
            let fraction = sparse.routed_fraction().unwrap().unwrap();
            assert!(
                (fraction - routed).abs() < 1e-9,
                "{fraction} against {routed}"
            );
        }
    }

    #[test]
    fn vsa_matches_the_reference_with_and_without_a_gate() {
        let device = Device::new().unwrap();
        // 70 text tokens, 2 × 40 audio rows and a video grid of 5 × 6 × 12 patches, which cuts
        // tiles shorter than 64 at the edges of the grid.
        let layout = PackedLayout::text_to_video(70, 5, 12, 24, 40);
        let plan = VsaPlan::for_layout(&layout);
        let (tokens, heads) = (layout.len(), 2);
        let kept = plan.kept_video_tiles(0.5);
        assert!(
            kept < plan.video_tiles(),
            "every tile kept, which tests nothing"
        );
        let (q, k, v, g) = (
            inputs(tokens, heads, 4),
            inputs(tokens, heads, 5),
            inputs(tokens, heads, 6),
            inputs(tokens, heads, 7),
        );
        let scale = (HEAD as f32).sqrt().recip();
        let upload = |x: &[f32]| Array::from_f32(&device, tokens, heads * HEAD, x).unwrap();
        for gate in [None, Some(&g)] {
            let expected = mmh3_core::dit::vsa::reference(
                &q,
                &k,
                &v,
                gate.map(Vec::as_slice),
                &plan,
                heads,
                HEAD,
                scale,
                kept,
            );
            for (half, tolerance) in precisions(&device) {
                let sparse = SparsePass::vsa(&device, &plan, heads, kept).unwrap();
                let inputs = AttentionInputs::from_arrays(
                    &upload(&q),
                    &upload(&k),
                    &upload(&v),
                    heads,
                    half,
                )
                .unwrap();
                let out = sparse
                    .attend(&inputs, gate.map(|g| upload(g)).as_ref())
                    .unwrap()
                    .to_f32()
                    .unwrap();
                let difference = largest_difference(&out, &expected);
                assert!(
                    difference < tolerance,
                    "FP16 {half} with a gate {}: {difference} from the reference",
                    gate.is_some()
                );
            }
        }
    }
}
