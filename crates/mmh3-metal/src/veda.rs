//! Veda's block-sparse attention over the DiT's heads of 128, as mmh3-core's `dit::veda`
//! describes it, on the attention kernels of `sparse`. ops.metal says what each kernel does.

use crate::{
    Buffer, Device, Error, Result,
    ops::{Array, AttentionInputs},
};
use mmh3_core::dit::{
    layout::PackedLayout,
    veda::{self, TilePlan, VEDA_TILE, VedaPredictor, VedaTiling},
};

const HEAD: usize = 128;
/// The most tiles of 128 a selection scores in threadgroup memory, `SPARSE_MAX_TILES` in ops.metal.
const MAX_TILES: usize = 4096;
/// Video tiles one threadgroup of veda_project projects, `VEDA_PROJECT_TILES` in ops.metal.
const PROJECT_TILES: usize = 8;

/// A predictor on the device: every layer's projections and the plans they were trained with.
pub struct VedaWeights {
    layers: usize,
    heads: usize,
    plans: Vec<TilePlan>,
    query: Buffer,
    key: Buffer,
    query_scales: Buffer,
    key_scales: Buffer,
    /// The target grid whose plan was last announced.
    announced: std::cell::Cell<Option<[usize; 3]>>,
}

impl VedaWeights {
    pub fn new(device: &Device, predictor: &VedaPredictor) -> Result<Self> {
        if predictor.head_dim != HEAD {
            return Err(Error::new(format!(
                "Veda on Metal takes heads of {HEAD}, not {}",
                predictor.head_dim
            )));
        }
        let upload = |which: usize| -> Result<(Buffer, Buffer)> {
            let weights: Vec<u8> = predictor
                .projections
                .iter()
                .flat_map(|layer| layer[which].fp8.iter().copied())
                .collect();
            let scales: Vec<u8> = predictor
                .projections
                .iter()
                .flat_map(|layer| layer[which].scales.iter().flat_map(|s| s.to_ne_bytes()))
                .collect();
            Ok((
                device.alloc(weights.len(), Some(&weights))?,
                device.alloc(scales.len(), Some(&scales))?,
            ))
        };
        let (query, query_scales) = upload(0)?;
        let (key, key_scales) = upload(1)?;
        Ok(Self {
            layers: predictor.layers,
            heads: predictor.heads,
            plans: predictor.plans.clone(),
            query,
            key,
            query_scales,
            key_scales,
            announced: std::cell::Cell::new(None),
        })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

/// The tables of one tile shape of the target: the sequence's order, its tiles of 128 and the
/// tiles of 64 the attention runs on, and each video query tile's budgets.
struct Shape {
    tiles: usize,
    video: usize,
    reference: usize,
    order: Buffer,
    starts: Buffer,
    lengths: Buffer,
    small_starts: Buffer,
    small_lengths: Buffer,
    allowed: Buffer,
}

impl Shape {
    fn new(
        device: &Device,
        tiling: &VedaTiling,
        sparsity: f64,
        reference_sparsity: f64,
    ) -> Result<Self> {
        let tiles = tiling.tiles();
        if tiling.video_tiles > MAX_TILES || 2 * tiles > u16::MAX as usize {
            return Err(Error::new(format!(
                "Veda takes up to {MAX_TILES} video tiles of {VEDA_TILE} tokens, not {}",
                tiling.video_tiles
            )));
        }
        let table = |values: &mut dyn Iterator<Item = usize>| -> Result<Buffer> {
            let bytes: Vec<u8> = values
                .flat_map(|value| (value as u32).to_ne_bytes())
                .collect();
            device.alloc(
                bytes.len().max(4),
                (!bytes.is_empty()).then_some(&bytes[..]),
            )
        };
        let blocks = veda::column_blocks(tiling, sparsity, reference_sparsity);
        // The reference block's count, then the generated block's, for each video query tile.
        let allowed = (0..tiling.video_tiles).flat_map(|row| {
            let generated = blocks.last().expect("a generated block");
            let reference = if blocks.len() == 2 {
                blocks[0].allowed(row)
            } else {
                0
            };
            [reference, generated.allowed(row)]
        });
        Ok(Self {
            tiles,
            video: tiling.video_tiles,
            reference: tiling.reference_tiles,
            order: table(&mut tiling.order.iter().copied())?,
            starts: table(&mut tiling.tile_starts.iter().copied())?,
            lengths: table(&mut tiling.tile_lengths.iter().copied())?,
            small_starts: table(
                &mut tiling
                    .tile_starts
                    .iter()
                    .flat_map(|&start| [start, start + VEDA_TILE / 2]),
            )?,
            small_lengths: table(&mut tiling.tile_lengths.iter().flat_map(|&length| {
                [
                    length.min(VEDA_TILE / 2),
                    length.saturating_sub(VEDA_TILE / 2),
                ]
            }))?,
            allowed: table(&mut allowed.into_iter())?,
        })
    }
}

/// A head group of a layer: its shape and its heads.
struct Group {
    shape: usize,
    members: usize,
    heads: Buffer,
}

/// A head group's copies of a block's attention inputs in tile order and its attention's output,
/// FP16 or FP32 as the inputs are, and the INT8 queries and keys with their scales when the inputs
/// have them.
struct Copies {
    half: bool,
    query: Buffer,
    key: Buffer,
    value: Buffer,
    attended: Buffer,
    quantized: Option<[Buffer; 3]>,
}

/// Veda over one sequence, with the plan of its target grid. Its scratch memory serves every block
/// and head group of a call in turn, since fresh memory for each would stay taken until the GPU had
/// run the blocks.
pub struct VedaPass<'a> {
    weights: &'a VedaWeights,
    rows: usize,
    shapes: Vec<Option<Shape>>,
    groups: Vec<Vec<Group>>,
    pooled_queries: Buffer,
    pooled_keys: Buffer,
    projected_queries: Buffer,
    projected_keys: Buffer,
    routes: Buffer,
    counts: Buffer,
    kept: Buffer,
    /// The most heads a group has.
    members: usize,
    copies: std::cell::RefCell<Option<Copies>>,
    /// Video tile pairs the attended blocks could have kept, over their heads.
    pairs: std::cell::Cell<usize>,
}

impl<'a> VedaPass<'a> {
    pub fn new(
        weights: &'a VedaWeights,
        layout: &PackedLayout,
        sparsity: f64,
        reference_sparsity: f64,
    ) -> Result<Self> {
        let device = &weights.query.0.device;
        let grid = veda::target_grid(layout);
        let (plan, exact) = veda::select_plan(&weights.plans, grid)
            .ok_or_else(|| Error::new("a Veda predictor without tile plans".into()))?;
        if weights.announced.replace(Some(grid)) != Some(grid) {
            let trained = if exact {
                "which the predictor was trained for"
            } else {
                "the nearest the predictor was trained for"
            };
            println!(
                "Veda tiles the {}×{}×{} video grid with plan {}, {trained}",
                grid[0], grid[1], grid[2], plan.name
            );
        }
        let tile_references = reference_sparsity > 0.0;
        let mut shapes: Vec<Option<Shape>> = (0..plan.shapes.len()).map(|_| None).collect();
        let mut groups = Vec::with_capacity(weights.layers);
        for layer in 0..weights.layers {
            let mut layer_groups = Vec::new();
            for (shape, heads) in plan.head_groups(layer) {
                if shapes[shape].is_none() {
                    let tiling = VedaTiling::new(layout, plan.shapes[shape], tile_references);
                    shapes[shape] =
                        Some(Shape::new(device, &tiling, sparsity, reference_sparsity)?);
                }
                let bytes: Vec<u8> = heads
                    .iter()
                    .flat_map(|&head| (head as u32).to_ne_bytes())
                    .collect();
                layer_groups.push(Group {
                    shape,
                    members: heads.len(),
                    heads: device.alloc(bytes.len(), Some(&bytes))?,
                });
            }
            groups.push(layer_groups);
        }

        let built = || shapes.iter().flatten();
        let members = groups
            .iter()
            .flatten()
            .map(|group| group.members)
            .max()
            .unwrap_or(1);
        let video = built().map(|shape| shape.video).max().unwrap_or(1).max(1);
        let small = built().map(|shape| 2 * shape.tiles).max().unwrap_or(1);
        let floats = |count: usize| device.alloc(count * 4, None);
        Ok(Self {
            weights,
            rows: layout.len(),
            pooled_queries: floats(members * video * 3 * HEAD)?,
            pooled_keys: floats(members * video * 3 * HEAD)?,
            projected_queries: floats(members * video * HEAD)?,
            projected_keys: floats(members * video * HEAD)?,
            routes: device.alloc(members * small * small * 2, None)?,
            counts: device.alloc(members * small * 4, None)?,
            kept: device.alloc(4, Some(&[0; 4]))?,
            members,
            copies: std::cell::RefCell::new(None),
            pairs: std::cell::Cell::new(0),
            shapes,
            groups,
        })
    }

    /// Block `layer`'s attention of `inputs`, in the packed order and in their precision: FP16 on
    /// the matrix units or FP32.
    pub fn attend(&self, inputs: &AttentionInputs, layer: usize) -> Result<Array> {
        let heads = self.weights.heads;
        if inputs.rows != self.rows
            || inputs.heads != heads
            || inputs.dim != HEAD
            || layer >= self.groups.len()
        {
            return Err(Error::new(
                "Veda attention inputs do not match its sequence".into(),
            ));
        }
        let device = &inputs.query.0.device;
        let rows = self.rows;
        let out = Array::empty(device, rows, heads * HEAD)?;
        let scale = (HEAD as f32).sqrt().recip();
        let mut copies = self.copies.borrow_mut();
        if copies.as_ref().is_none_or(|c| {
            c.half != inputs.half || c.quantized.is_some() != inputs.quantized.is_some()
        }) {
            let element = if inputs.half { 2 } else { 4 };
            let copy = |element: usize| device.alloc(rows * self.members * HEAD * element, None);
            *copies = Some(Copies {
                half: inputs.half,
                query: copy(element)?,
                key: copy(element)?,
                value: copy(element)?,
                attended: copy(4)?,
                quantized: inputs
                    .quantized
                    .as_ref()
                    .map(|_| -> Result<[Buffer; 3]> {
                        Ok([
                            copy(1)?,
                            copy(1)?,
                            device.alloc(2 * rows * self.members * 4, None)?,
                        ])
                    })
                    .transpose()?,
            });
        }
        let Copies {
            query,
            key,
            value,
            attended,
            quantized: quantized_copies,
            ..
        } = copies.as_ref().expect("copies for these inputs");
        for group in &self.groups[layer] {
            let shape = self.shapes[group.shape]
                .as_ref()
                .expect("a shape for every group");
            let members = group.members;
            let gather = |source: &Buffer, words: usize, offsets: [usize; 2], copy: &Buffer| {
                device.run(
                    "veda_gather",
                    &[source, copy, &shape.order, &group.heads],
                    &[
                        rows as u32,
                        heads as u32,
                        members as u32,
                        words as u32,
                        offsets[0] as u32,
                        offsets[1] as u32,
                    ],
                    rows * members * words,
                    false,
                )
            };
            let words = HEAD * if inputs.half { 2 } else { 4 } / 4;
            gather(&inputs.query, words, [0, 0], query)?;
            gather(&inputs.key, words, [0, 0], key)?;
            gather(&inputs.value, words, [0, 0], value)?;

            let (tiles, video) = (shape.tiles as u32, shape.video as u32);
            device.run(
                "veda_pool",
                &[
                    query,
                    key,
                    &shape.starts,
                    &shape.lengths,
                    &self.pooled_queries,
                    &self.pooled_keys,
                ],
                &[video, members as u32, inputs.half as u32],
                shape.video * members.div_ceil(2),
                true,
            )?;
            for (pooled, weights, scales, projected) in [
                (
                    &self.pooled_queries,
                    &self.weights.query,
                    &self.weights.query_scales,
                    &self.projected_queries,
                ),
                (
                    &self.pooled_keys,
                    &self.weights.key,
                    &self.weights.key_scales,
                    &self.projected_keys,
                ),
            ] {
                device.run(
                    "veda_project",
                    &[pooled, weights, scales, &group.heads, projected],
                    &[video, members as u32, layer as u32, heads as u32],
                    members * shape.video.div_ceil(PROJECT_TILES),
                    true,
                )?;
            }
            device.run(
                "veda_select",
                &[
                    &self.projected_queries,
                    &self.projected_keys,
                    &shape.lengths,
                    &shape.allowed,
                    &self.routes,
                    &self.counts,
                    &self.kept,
                ],
                &[tiles, video, shape.reference as u32, scale.to_bits()],
                shape.tiles * members,
                true,
            )?;
            self.pairs
                .set(self.pairs.get() + members * shape.video * shape.video);

            let parameters = [
                2 * tiles,
                members as u32,
                1,
                0,
                (scale * std::f32::consts::LOG2_E).to_bits(),
            ];
            let tables = [
                &shape.small_starts,
                &shape.small_lengths,
                &self.routes,
                &self.counts,
            ];
            // VSA without a gate reads neither the tails nor the key means nor the gate.
            let unused = &self.counts;
            if let (Some(quantized), Some([query8, key8, scales])) =
                (&inputs.quantized, quantized_copies)
            {
                gather(&quantized.query, HEAD / 4, [0, 0], query8)?;
                gather(&quantized.key, HEAD / 4, [0, 0], key8)?;
                gather(&quantized.scales, 1, [0, 0], scales)?;
                gather(&quantized.scales, 1, [rows, rows], scales)?;
                let mut parameters = parameters.to_vec();
                parameters.push(rows as u32);
                device.run(
                    "mpp_sparse_attention_int8_128",
                    &[
                        query, key, value, attended, tables[0], tables[1], tables[2], tables[3],
                        unused, unused, unused, query8, key8, scales,
                    ],
                    &parameters,
                    2 * shape.tiles * members,
                    true,
                )?;
            } else {
                let (name, threadgroups) = if inputs.half {
                    ("mpp_sparse_attention_128", 2 * shape.tiles * members)
                } else {
                    ("attention_sparse_128", 2 * shape.tiles * members * 8)
                };
                device.run(
                    name,
                    &[
                        query, key, value, attended, tables[0], tables[1], tables[2], tables[3],
                        unused, unused, unused,
                    ],
                    &parameters,
                    threadgroups,
                    true,
                )?;
            }
            device.run(
                "veda_scatter",
                &[attended, &out.buffer, &shape.order, &group.heads],
                &[rows as u32, heads as u32, members as u32],
                rows * members * HEAD,
                false,
            )?;
        }
        Ok(out)
    }

    /// The fraction of the video tile pairs the query tiles kept, over the blocks attended so far,
    /// or None before any.
    pub fn kept_fraction(&self) -> Result<Option<f64>> {
        let pairs = self.pairs.get();
        if pairs == 0 {
            return Ok(None);
        }
        let bytes = self.kept.to_bytes()?;
        let kept = u32::from_ne_bytes(bytes[..4].try_into().expect("four bytes"));
        Ok(Some(kept as f64 / pairs as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmh3_core::dit::{
        layout::KeyframeShape,
        veda::{Projection, TileShape},
    };
    use mmh3_core::numeric::{f16_to_f32, f32_to_f16};

    /// Values that spread the scores, rounded to FP16 so that both precisions read the same ones.
    fn inputs(tokens: usize, heads: usize, seed: usize) -> Vec<f32> {
        (0..tokens * heads * HEAD)
            .map(|index| {
                let token = index / (heads * HEAD);
                let wave = ((index * 7919 + seed * 104_729) % 1009) as f32 / 1009.0 - 0.5;
                f16_to_f32(f32_to_f16(
                    wave * 2.5 + (token as f32 / 37.0 + seed as f32).sin(),
                ))
            })
            .collect()
    }

    /// A predictor of two blocks of three heads, which share a shape in the first block and not in
    /// the second, with FP8 projections of no infinities or NaNs.
    fn predictor() -> VedaPredictor {
        let (layers, heads) = (2, 3);
        let projection = |seed: usize| Projection {
            fp8: (0..heads * 3 * HEAD * HEAD)
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
            head_dim: HEAD,
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

    #[test]
    fn veda_matches_the_reference() {
        let device = Device::new().unwrap();
        let predictor = predictor();
        let weights = VedaWeights::new(&device, &predictor).unwrap();
        // 70 text tokens, 2 × 40 audio rows and a video grid of 5 × 8 × 12 patches, alone and
        // after a keyframe, whose tiles the reference block keeps.
        let keyframe = [KeyframeShape {
            frame_index: 0,
            latent_frames: 1,
            audio_frames: 0,
        }];
        for keyframes in [&[][..], &keyframe[..]] {
            let layout = PackedLayout::new(70, 5, 16, 24, 40, keyframes, &[]);
            let (tokens, heads) = (layout.len(), predictor.heads);
            let (q, k, v) = (
                inputs(tokens, heads, 1),
                inputs(tokens, heads, 2),
                inputs(tokens, heads, 3),
            );
            let upload = |x: &[f32]| Array::from_f32(&device, tokens, heads * HEAD, x).unwrap();
            let plan = &predictor.plans[0];
            let tilings: Vec<Option<VedaTiling>> = plan
                .shapes
                .iter()
                .map(|&shape| Some(VedaTiling::new(&layout, shape, true)))
                .collect();
            let mut precisions = vec![(false, 1e-4)];
            if device.supports_tensor_ops() {
                precisions.push((true, 5e-3));
            }
            for (half, tolerance) in precisions {
                let pass = VedaPass::new(&weights, &layout, 0.5, 0.5).unwrap();
                let mut kept = 0;
                for layer in 0..predictor.layers {
                    let (expected, kept_here) = veda::reference(
                        &q,
                        &k,
                        &v,
                        heads,
                        HEAD,
                        &plan.head_shape[layer],
                        &tilings,
                        &predictor.projections[layer],
                        0.5,
                        0.5,
                    );
                    kept += kept_here;
                    let inputs = AttentionInputs::from_arrays(
                        &upload(&q),
                        &upload(&k),
                        &upload(&v),
                        heads,
                        half,
                    )
                    .unwrap();
                    let out = pass.attend(&inputs, layer).unwrap().to_f32().unwrap();
                    let difference = out
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        difference < tolerance,
                        "FP16 {half}, {} keyframes, block {layer}: {difference} from the reference",
                        keyframes.len()
                    );
                }
                let pairs = pass.pairs.get() as f64;
                let fraction = pass.kept_fraction().unwrap().unwrap();
                assert_eq!((fraction * pairs).round() as usize, kept);
                assert!(
                    fraction < 0.9,
                    "kept {fraction} of the tiles, which tests little"
                );
            }
        }
    }
}
