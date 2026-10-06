//! Veda's block-sparse attention over the DiT's BF16 heads of 128, as mmh3-core's `dit::veda`
//! describes it. kernels/veda_attention.cu says what each kernel does.

use crate::attention::{AttentionLayout, AttentionOffsets, HEAD_DIM};
use crate::model::{Error, i32_buffer};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::dit::layout::{PackedLayout, Segment};
use mmh3_core::dit::veda::{self, TilePlan, VEDA_TILE, VedaPredictor, VedaTiling};
use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::ptr;

#[repr(C)]
struct RawVedaShape {
    order: *const c_void,
    starts: *const c_void,
    lengths: *const c_void,
    allowed: *const c_void,
    tiles: i32,
    video: i32,
    reference: i32,
}

#[repr(C)]
struct RawVedaWorkspace {
    pooled_query: *mut c_void,
    pooled_key: *mut c_void,
    projected_query: *mut c_void,
    projected_key: *mut c_void,
    routes: *mut c_void,
    counts: *mut c_void,
    kept: *mut c_void,
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn mmh3_veda_attention(
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        layout: *const AttentionLayout,
        shape: *const RawVedaShape,
        heads: *const c_void,
        members: c_int,
        query_weights: *const c_void,
        query_scales: *const c_void,
        key_weights: *const c_void,
        key_scales: *const c_void,
        scale: f32,
        workspace: *const RawVedaWorkspace,
        stream: *mut c_void,
    ) -> c_int;
}

/// A predictor on the device: every layer's projections and the plans they were trained with.
pub struct VedaWeights {
    layers: usize,
    heads: usize,
    plans: Vec<TilePlan>,
    query: DeviceBuffer,
    key: DeviceBuffer,
    query_scales: DeviceBuffer,
    key_scales: DeviceBuffer,
    /// The target grid whose plan was last announced.
    announced: Cell<Option<[usize; 3]>>,
}

impl VedaWeights {
    pub fn new(predictor: &VedaPredictor) -> Result<Self, Error> {
        if predictor.head_dim != HEAD_DIM {
            return Err(Error::Model(format!(
                "Veda on CUDA takes heads of {HEAD_DIM}, not {}",
                predictor.head_dim
            )));
        }
        let upload = |which: usize| -> Result<(DeviceBuffer, DeviceBuffer), CudaError> {
            let weights: Vec<u8> = predictor
                .projections
                .iter()
                .flat_map(|layer| layer[which].fp8.iter().copied())
                .collect();
            let scales: Vec<u8> = predictor
                .projections
                .iter()
                .flat_map(|layer| layer[which].scales.iter().flat_map(|s| s.to_le_bytes()))
                .collect();
            Ok((
                DeviceBuffer::from_bytes(&weights)?,
                DeviceBuffer::from_bytes(&scales)?,
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
            announced: Cell::new(None),
        })
    }

    /// A copy on the device this thread computes on.
    pub fn copied(&self) -> Result<Self, CudaError> {
        Ok(Self {
            layers: self.layers,
            heads: self.heads,
            plans: self.plans.clone(),
            query: self.query.copied()?,
            key: self.key.copied()?,
            query_scales: self.query_scales.copied()?,
            key_scales: self.key_scales.copied()?,
            announced: Cell::new(None),
        })
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn heads(&self) -> usize {
        self.heads
    }
}

/// The tables of one tile shape of the target: the sequence's order, its tiles of 128 and each
/// video query tile's budgets.
struct Shape {
    tiles: usize,
    video: usize,
    reference: usize,
    order: DeviceBuffer,
    starts: DeviceBuffer,
    lengths: DeviceBuffer,
    allowed: DeviceBuffer,
}

impl Shape {
    fn new(tiling: &VedaTiling, sparsity: f64, reference_sparsity: f64) -> Result<Self, Error> {
        let tiles = tiling.tiles();
        if 2 * tiles > u16::MAX as usize {
            return Err(Error::Model(format!(
                "Veda takes up to {} tiles of {VEDA_TILE} tokens, not {tiles}",
                u16::MAX / 2
            )));
        }
        let blocks = veda::column_blocks(tiling, sparsity, reference_sparsity);
        let generated = blocks.last().expect("a generated block");
        // The reference block's count, then the generated block's, for each video query tile.
        let allowed: Vec<usize> = (0..tiling.video_tiles)
            .flat_map(|row| {
                let reference = if blocks.len() == 2 {
                    blocks[0].allowed(row)
                } else {
                    0
                };
                [reference, generated.allowed(row)]
            })
            .collect();
        let table = |values: &[usize]| {
            if values.is_empty() {
                DeviceBuffer::new(4)
            } else {
                i32_buffer(values)
            }
        };
        Ok(Self {
            tiles,
            video: tiling.video_tiles,
            reference: tiling.reference_tiles,
            order: table(&tiling.order)?,
            starts: table(&tiling.tile_starts)?,
            lengths: table(&tiling.tile_lengths)?,
            allowed: table(&allowed)?,
        })
    }

    fn raw(&self) -> RawVedaShape {
        RawVedaShape {
            order: self.order.pointer().cast_const(),
            starts: self.starts.pointer().cast_const(),
            lengths: self.lengths.pointer().cast_const(),
            allowed: self.allowed.pointer().cast_const(),
            tiles: self.tiles as i32,
            video: self.video as i32,
            reference: self.reference as i32,
        }
    }
}

/// A head group of a layer: its shape and its heads.
struct Group {
    shape: usize,
    members: usize,
    heads: DeviceBuffer,
}

/// What a pass was made for, which the steps of a generation share.
#[derive(Clone, Debug, PartialEq)]
struct Key {
    segments: Vec<Segment>,
    grid: [usize; 3],
    sparsity: f64,
    reference_sparsity: f64,
}

impl Key {
    fn new(layout: &PackedLayout, sparsity: f64, reference_sparsity: f64) -> Self {
        Key {
            segments: layout.segments.clone(),
            grid: veda::target_grid(layout),
            sparsity,
            reference_sparsity,
        }
    }
}

/// Veda over one sequence, with the plan of its target grid. Its scratch memory serves every block
/// and head group in turn.
pub struct VedaPass {
    key: Key,
    rows: usize,
    heads: usize,
    shapes: Vec<Option<Shape>>,
    groups: Vec<Vec<Group>>,
    pooled_query: DeviceBuffer,
    pooled_key: DeviceBuffer,
    projected_query: DeviceBuffer,
    projected_key: DeviceBuffer,
    routes: DeviceBuffer,
    counts: DeviceBuffer,
    kept: DeviceBuffer,
    /// Video tile pairs the attended blocks could have kept, over their heads.
    pairs: Cell<usize>,
}

impl VedaPass {
    pub fn new(
        weights: &VedaWeights,
        layout: &PackedLayout,
        sparsity: f64,
        reference_sparsity: f64,
    ) -> Result<Self, Error> {
        let grid = veda::target_grid(layout);
        let (plan, exact) = veda::select_plan(&weights.plans, grid)
            .ok_or_else(|| Error::Model("a Veda predictor without tile plans".to_owned()))?;
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
                    shapes[shape] = Some(Shape::new(&tiling, sparsity, reference_sparsity)?);
                }
                layer_groups.push(Group {
                    shape,
                    members: heads.len(),
                    heads: i32_buffer(&heads)?,
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
        let tiles = built().map(|shape| shape.tiles).max().unwrap_or(1);
        let floats = |count: usize| DeviceBuffer::new(count * 4);
        Ok(Self {
            key: Key::new(layout, sparsity, reference_sparsity),
            rows: layout.len(),
            heads: weights.heads,
            pooled_query: floats(members * video * 3 * HEAD_DIM)?,
            pooled_key: floats(members * video * 3 * HEAD_DIM)?,
            projected_query: floats(members * video * HEAD_DIM)?,
            projected_key: floats(members * video * HEAD_DIM)?,
            routes: DeviceBuffer::new(members * tiles * 2 * tiles * 2)?,
            counts: DeviceBuffer::new(members * tiles * 4)?,
            kept: DeviceBuffer::zeroed(8)?,
            pairs: Cell::new(0),
            shapes,
            groups,
        })
    }

    /// Whether the pass serves `layout` at these sparsities.
    pub fn serves(&self, layout: &PackedLayout, sparsity: f64, reference_sparsity: f64) -> bool {
        self.key == Key::new(layout, sparsity, reference_sparsity)
    }

    /// Forgets the tiles kept so far.
    pub fn reset(&self) -> Result<(), CudaError> {
        self.pairs.set(0);
        self.kept.clear()
    }

    /// Raw form of `attend` for pointers the caller has already checked.
    ///
    /// # Safety
    /// Every element the layout addresses for the pass's rows and the predictor's heads must lie
    /// inside the pointed-to buffers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attend_pointers(
        &self,
        weights: &VedaWeights,
        query: *const c_void,
        key: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        layout: &AttentionLayout,
        layer: usize,
    ) -> Result<(), CudaError> {
        assert!(
            layer < self.groups.len() && weights.heads == self.heads,
            "the Veda pass was not made for this predictor"
        );
        let workspace = RawVedaWorkspace {
            pooled_query: self.pooled_query.pointer(),
            pooled_key: self.pooled_key.pointer(),
            projected_query: self.projected_query.pointer(),
            projected_key: self.projected_key.pointer(),
            routes: self.routes.pointer(),
            counts: self.counts.pointer(),
            kept: self.kept.pointer(),
        };
        let projection = self.heads * 3 * HEAD_DIM * HEAD_DIM;
        let scale = (HEAD_DIM as f32).sqrt().recip();
        for group in &self.groups[layer] {
            let shape = self.shapes[group.shape]
                .as_ref()
                .expect("a shape for every group");
            // SAFETY: the caller guarantees the extents, the tables lie inside the sequence, and the
            // weights hold every layer's projections.
            check(unsafe {
                mmh3_veda_attention(
                    query,
                    key,
                    value,
                    output,
                    layout,
                    &shape.raw(),
                    group.heads.pointer().cast_const(),
                    group.members as c_int,
                    weights.query.pointer_at(layer * projection).cast_const(),
                    weights
                        .query_scales
                        .pointer_at(layer * self.heads * 4)
                        .cast_const(),
                    weights.key.pointer_at(layer * projection).cast_const(),
                    weights
                        .key_scales
                        .pointer_at(layer * self.heads * 4)
                        .cast_const(),
                    scale,
                    &workspace,
                    ptr::null_mut(),
                )
            })?;
            self.pairs
                .set(self.pairs.get() + group.members * shape.video * shape.video);
        }
        Ok(())
    }

    /// Block `layer`'s attention, reading query, key and value from `input` and writing `output` at
    /// their offsets with the token and head strides of `layout`.
    pub fn attend(
        &self,
        weights: &VedaWeights,
        input: &DeviceBuffer,
        output: &mut DeviceBuffer,
        offsets: AttentionOffsets,
        layout: &AttentionLayout,
        layer: usize,
    ) -> Result<(), CudaError> {
        let (rows, heads) = (self.rows, self.heads);
        let last = |offset: usize, operand: usize| {
            offset as i64
                + (rows as i64 - 1) * layout.token_stride[operand]
                + (heads as i64 - 1) * layout.head_stride[operand]
                + HEAD_DIM as i64
        };
        for (operand, offset) in [offsets.query, offsets.key, offsets.value]
            .into_iter()
            .enumerate()
        {
            assert!(
                last(offset, operand) <= (input.bytes() / 2) as i64,
                "operand {operand} reaches past the input buffer"
            );
        }
        assert!(
            last(offsets.output, 3) <= (output.bytes() / 2) as i64,
            "the output reaches past its buffer"
        );
        // SAFETY: every addressed element lies inside the buffers, checked above.
        unsafe {
            self.attend_pointers(
                weights,
                input.pointer_at(offsets.query * 2),
                input.pointer_at(offsets.key * 2),
                input.pointer_at(offsets.value * 2),
                output.pointer_at(offsets.output * 2),
                layout,
                layer,
            )
        }
    }

    /// The fraction of the video tile pairs the query tiles kept, over the blocks attended since
    /// the last reset, or None before any.
    pub fn kept_fraction(&self) -> Result<Option<f64>, CudaError> {
        let pairs = self.pairs.get();
        if pairs == 0 {
            return Ok(None);
        }
        let mut bytes = [0u8; 8];
        self.kept.copy_to_host(&mut bytes)?;
        Ok(Some(u64::from_le_bytes(bytes) as f64 / pairs as f64))
    }
}
