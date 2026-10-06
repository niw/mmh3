//! Veda, a block-sparse attention whose tiles a small distilled predictor picks (arXiv
//! 2605.30325), as the Veda-Sparse predictor for MiniMax H3 was trained: the tiling of the packed
//! sequence, the predictor bundle, the selection rules and a CPU reference of the attention.
//!
//! A tile shape (t, h, w) of 128 tokens cuts a video span's (frames, patch rows, patch columns)
//! grid into boxes, ordered by row box, column box and frame box from outer to inner, with the
//! tokens of a box in their packed order. Boxes past the grid's edges hold fewer tokens. The target
//! video is one span, and the video rows of keyframes and references are spans of their own before
//! it, each cut by the shape with the least padding on its grid. Every other row (text, audio) is
//! global: global rows are cut into tiles of 128 in sequence order, after the tiles of the spans.
//! Each layer's heads use the tile shapes of a plan, at most two shapes a layer, searched for one
//! target grid and shipped in the bundle.
//!
//! Per head, the predictor pools the queries and the keys of every video tile into their mean,
//! maximum and minimum, maps them to `pooled · P + mean` with a learned projection P of that layer
//! and head, and scores every video query tile against every video key tile with the dot products
//! of those. The selection then keeps for each video query tile:
//!
//! 1. Every global key tile. Global query tiles keep every tile.
//! 2. In the video columns, the best-scoring key tiles of each column block on its own: the
//!    reference block (the spans before the target) and the generated block (the target).
//! 3. Per block, `(1 - sparsity) · n² / columns` tiles, where n is the block's tokens over 128,
//!    so that padded tiles do not raise the budget. The fraction is spread over the query tiles by
//!    a Bresenham pattern, so the mean kept count is the budget.
//! 4. The query tile's own tile, which counts toward the budget, and never an empty tile.
//!
//! The attention is then exact over the kept tiles, token by token.

use crate::dit::layout::{PackedLayout, Segment, SegmentKind};
use crate::json;
use crate::safetensors::{DType, SafeTensors};
use std::path::Path;

pub const VEDA_TILE: usize = 128;
/// The fraction of the video tiles the released predictor was trained to drop.
pub const VEDA_SPARSITY: f64 = 0.9;
/// The predictor that `--attention veda` reads from the models directory by default.
pub const VEDA_PREDICTOR_FILE: &str =
    "veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors";
const BUNDLE_FORMAT: &str = "miowtion-veda-predictor-v1";

/// A box of `VEDA_TILE` tokens: (frames, patch rows, patch columns).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileShape {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl TileShape {
    pub fn new(t: usize, h: usize, w: usize) -> Option<Self> {
        (t * h * w == VEDA_TILE).then_some(TileShape { t, h, w })
    }

    /// A shape written as `8x4x4`.
    pub fn parse(text: &str) -> Option<Self> {
        let mut extents = text.split('x').map(|value| value.parse::<usize>().ok());
        let shape = Self::new(extents.next()??, extents.next()??, extents.next()??)?;
        extents.next().is_none().then_some(shape)
    }

    pub fn transposed(self) -> Self {
        TileShape {
            t: self.t,
            h: self.w,
            w: self.h,
        }
    }

    fn extents(self) -> [usize; 3] {
        [self.t, self.h, self.w]
    }

    /// Tiles of this shape on `grid`, counting the boxes past its edges.
    pub fn tiles(self, grid: [usize; 3]) -> usize {
        grid.iter()
            .zip(self.extents())
            .map(|(&size, extent)| size.div_ceil(extent))
            .product()
    }

    fn spread(self) -> f64 {
        let extents = self.extents();
        *extents.iter().max().unwrap() as f64 / *extents.iter().min().unwrap() as f64
    }
}

/// Every power-of-two shape, in (t, h, w) order.
fn all_shapes() -> Vec<TileShape> {
    let exponent = VEDA_TILE.trailing_zeros();
    let mut shapes = Vec::new();
    for i in 0..=exponent {
        for j in 0..=exponent - i {
            shapes.push(TileShape::new(1 << i, 1 << j, 1 << (exponent - i - j)).unwrap());
        }
    }
    shapes.sort();
    shapes
}

/// The shape with the fewest tiles on `grid`, the most cubic among those, then the first. Shapes
/// that do not fit inside the grid only count when none does.
pub fn least_padding_shape(grid: [usize; 3]) -> TileShape {
    let shapes = all_shapes();
    let fitting: Vec<TileShape> = shapes
        .iter()
        .copied()
        .filter(|shape| shape.t <= grid[0] && shape.h <= grid[1] && shape.w <= grid[2])
        .collect();
    let candidates = if fitting.is_empty() { shapes } else { fitting };
    candidates
        .into_iter()
        .min_by(|left, right| {
            left.tiles(grid)
                .cmp(&right.tiles(grid))
                .then(left.spread().total_cmp(&right.spread()))
                .then(left.cmp(right))
        })
        .unwrap()
}

/// The tile shape of every layer and head for one target grid.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan {
    pub name: String,
    /// The target grid (frames, patch rows, patch columns) the plan was searched for.
    pub grid: [usize; 3],
    pub shapes: Vec<TileShape>,
    /// For each layer and head, an index into `shapes`.
    pub head_shape: Vec<Vec<usize>>,
}

impl TilePlan {
    /// The plan of the grid with its rows and columns swapped.
    pub fn transposed(&self) -> Self {
        TilePlan {
            name: format!("{}_T", self.name),
            grid: [self.grid[0], self.grid[2], self.grid[1]],
            shapes: self.shapes.iter().map(|shape| shape.transposed()).collect(),
            head_shape: self.head_shape.clone(),
        }
    }

    /// The distinct shapes of a layer, ascending, with the heads that use each.
    pub fn head_groups(&self, layer: usize) -> Vec<(usize, Vec<usize>)> {
        let row = &self.head_shape[layer];
        let mut used: Vec<usize> = row.clone();
        used.sort_unstable();
        used.dedup();
        used.into_iter()
            .map(|shape| {
                let heads = (0..row.len()).filter(|&head| row[head] == shape).collect();
                (shape, heads)
            })
            .collect()
    }
}

/// The plan for a target grid: the nearest aspect ratio, then the nearest number of frames, then
/// the least padding, among the plans and their transposes. Also says whether the plan was searched
/// for exactly this grid; any other plan still tiles it, outside what the predictor was trained on.
pub fn select_plan(plans: &[TilePlan], grid: [usize; 3]) -> Option<(TilePlan, bool)> {
    let aspect = |grid: [usize; 3]| (grid[2] as f64 / grid[1] as f64).ln();
    let target = aspect(grid);
    let cost = |plan: &TilePlan| {
        let distance = ((aspect(plan.grid) - target).abs() * 1000.0).round() as u64;
        let padding: usize = plan.shapes.iter().map(|shape| shape.tiles(grid)).sum();
        (distance, plan.grid[0].abs_diff(grid[0]), padding)
    };
    let candidates = plans
        .iter()
        .cloned()
        .chain(plans.iter().map(TilePlan::transposed));
    // The first of equal costs wins.
    let mut best: Option<TilePlan> = None;
    for plan in candidates {
        if best.as_ref().is_none_or(|best| cost(&plan) < cost(best)) {
            best = Some(plan);
        }
    }
    best.map(|plan| {
        let exact = plan.grid == grid;
        (plan, exact)
    })
}

/// One layer's query or key projection, `[heads, 3 × dim, dim]` FP8 E4M3 values with a scale
/// for each head.
#[derive(Clone, Debug)]
pub struct Projection {
    pub fp8: Vec<u8>,
    pub scales: Vec<f32>,
}

impl Projection {
    /// A weight as the predictor computes with it: the FP8 value times its head's scale, rounded to
    /// BF16.
    pub fn weight(&self, dim: usize, head: usize, row: usize, column: usize) -> f32 {
        let value = self.fp8[(head * 3 * dim + row) * dim + column];
        let scaled = fp8_e4m3_to_f32(value) * self.scales[head];
        crate::numeric::bf16_to_f32(crate::numeric::f32_to_bf16(scaled))
    }
}

/// The value of an FP8 E4M3 (fn) byte, which has no infinities.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 3) & 0xF) as i32;
    let mantissa = (bits & 0x7) as f32;
    if exponent == 0xF && bits & 0x7 == 0x7 {
        return f32::NAN;
    }
    if exponent == 0 {
        return sign * mantissa / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
}

/// A predictor bundle: the projections and the tile plans they were trained with.
#[derive(Clone, Debug)]
pub struct VedaPredictor {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// The fraction of the video tiles the predictor was trained to keep.
    pub keep_ratio: f64,
    pub plans: Vec<TilePlan>,
    /// Each layer's query and key projections.
    pub projections: Vec<[Projection; 2]>,
}

impl VedaPredictor {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = SafeTensors::open(path).map_err(|error| error.to_string())?;
        Self::from_safetensors(&file).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_safetensors(file: &SafeTensors) -> Result<Self, String> {
        let metadata = |key: &str| -> Result<&str, String> {
            file.metadata()
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value.as_str())
                .ok_or_else(|| format!("no {key} in the metadata of a Veda predictor"))
        };
        if metadata("format").ok() != Some(BUNDLE_FORMAT) {
            return Err(format!("not a Veda predictor ({BUNDLE_FORMAT})"));
        }
        let number = |key: &str| -> Result<usize, String> {
            metadata(key)?
                .parse()
                .map_err(|_| format!("{key} is not a number"))
        };
        let (layers, heads, head_dim) = (
            number("num_layers")?,
            number("num_heads")?,
            number("head_dim")?,
        );
        let keep_ratio: f64 = metadata("keep_ratio")?
            .parse()
            .map_err(|_| "keep_ratio is not a number".to_owned())?;
        if metadata("dtype")? != "float8_e4m3fn" {
            return Err(format!(
                "mmh3 reads FP8 Veda predictors, not {}",
                metadata("dtype")?
            ));
        }

        let plans = parse_plans(metadata("plans")?)?;
        if plans.is_empty() {
            return Err("a Veda predictor without tile plans".into());
        }
        for plan in &plans {
            if plan.head_shape.len() != layers
                || plan.head_shape.iter().any(|row| {
                    row.len() != heads || row.iter().any(|&shape| shape >= plan.shapes.len())
                })
            {
                return Err(format!(
                    "plan {} does not cover {layers} × {heads} heads",
                    plan.name
                ));
            }
        }

        let tensor = |name: &str, dtype: DType, shape: &[usize]| -> Result<&[u8], String> {
            let info = file
                .get(name)
                .filter(|info| info.dtype == dtype && info.shape == shape)
                .ok_or_else(|| format!("{name} is missing or not {} {shape:?}", dtype.name()))?;
            Ok(file.data(info))
        };
        let mut projections = Vec::with_capacity(layers);
        for layer in 0..layers {
            let projection = |which: &str| -> Result<Projection, String> {
                let name = format!("layers.{layer}.{which}");
                let fp8 = tensor(&name, DType::F8E4M3, &[heads, 3 * head_dim, head_dim])?;
                let scales = tensor(&format!("{name}.__scale"), DType::F32, &[heads])?;
                Ok(Projection {
                    fp8: fp8.to_vec(),
                    scales: scales
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|bytes| f32::from_le_bytes(*bytes))
                        .collect(),
                })
            };
            projections.push([projection("proj_q")?, projection("proj_k")?]);
        }
        if file.tensors().len() != 4 * layers {
            return Err("unexpected tensors in a Veda predictor".into());
        }
        Ok(VedaPredictor {
            layers,
            heads,
            head_dim,
            keep_ratio,
            plans,
            projections,
        })
    }
}

fn parse_plans(text: &str) -> Result<Vec<TilePlan>, String> {
    let invalid = |name: &str| format!("tile plan {name} is invalid");
    let value = json::parse(text).map_err(|error| format!("tile plans: {error}"))?;
    let members = value.as_object().ok_or("tile plans are not an object")?;
    members
        .iter()
        .map(|(name, plan)| {
            let grid: Vec<usize> = plan
                .get("grid")
                .and_then(json::Value::as_array)
                .ok_or_else(|| invalid(name))?
                .iter()
                .map(|size| size.as_u64().map(|size| size as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| invalid(name))?;
            let shapes = plan
                .get("shapes")
                .and_then(json::Value::as_array)
                .ok_or_else(|| invalid(name))?
                .iter()
                .map(|shape| shape.as_str().and_then(TileShape::parse))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid(name))?;
            let head_shape = plan
                .get("head_shape")
                .and_then(json::Value::as_array)
                .ok_or_else(|| invalid(name))?
                .iter()
                .map(|row| {
                    row.as_array()?
                        .iter()
                        .map(|shape| shape.as_u64().map(|shape| shape as usize))
                        .collect::<Option<Vec<_>>>()
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid(name))?;
            let name = plan
                .get("geometry")
                .and_then(json::Value::as_str)
                .unwrap_or(name)
                .to_owned();
            Ok(TilePlan {
                grid: grid.try_into().map_err(|_| invalid(&name))?,
                name,
                shapes,
                head_shape,
            })
        })
        .collect()
}

/// The target grid of a layout: (latent frames, patch rows, patch columns).
pub fn target_grid(layout: &PackedLayout) -> [usize; 3] {
    [
        layout.latent_frames,
        layout.latent_height / 2,
        layout.latent_width / 2,
    ]
}

/// The (frames, patch rows, patch columns) grid of a segment whose positions run over one, row by
/// row, or None.
fn segment_grid(layout: &PackedLayout, segment: &Segment) -> Option<[usize; 3]> {
    let positions = &layout.positions[segment.start..segment.end];
    let axis = |index: usize| {
        let mut values: Vec<f64> = positions.iter().map(|position| position[index]).collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        values
    };
    let axes = [axis(0), axis(1), axis(2)];
    let (frames, rows, columns) = (axes[0].len(), axes[1].len(), axes[2].len());
    if frames * rows * columns != positions.len() {
        return None;
    }
    let row_major = positions.iter().enumerate().all(|(index, position)| {
        position[0] == axes[0][index / (rows * columns)]
            && position[1] == axes[1][index / columns % rows]
            && position[2] == axes[2][index % columns]
    });
    row_major.then_some([frames, rows, columns])
}

/// A video span of the packed sequence and the shape that tiles it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: usize,
    grid: [usize; 3],
    shape: TileShape,
}

/// The packed sequence cut into Veda's tiles for one tile shape of the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VedaTiling {
    /// The packed row of each position of the sequence in tile order.
    pub order: Vec<usize>,
    /// First position of each tile in the sequence in tile order.
    pub tile_starts: Vec<usize>,
    /// Rows in each tile, at most `VEDA_TILE` and possibly none.
    pub tile_lengths: Vec<usize>,
    /// The tiles of the reference spans, which come first.
    pub reference_tiles: usize,
    /// The tiles of every span, references then the target, before the global tiles.
    pub video_tiles: usize,
    pub reference_tokens: usize,
    pub target_tokens: usize,
}

impl VedaTiling {
    /// The tiling of `layout` with the target in `shape`, and with the video rows of keyframes and
    /// references as spans of their own when `tile_references`, global otherwise.
    pub fn new(layout: &PackedLayout, shape: TileShape, tile_references: bool) -> Self {
        let mut spans = Vec::new();
        if tile_references {
            for segment in &layout.segments {
                if !matches!(
                    segment.kind,
                    SegmentKind::KeyframeVideo(_) | SegmentKind::ReferenceVideo(_)
                ) || segment.is_empty()
                {
                    continue;
                }
                // A segment that is not a grid stays global, which is only slower.
                if let Some(grid) = segment_grid(layout, segment) {
                    spans.push(Span {
                        start: segment.start,
                        grid,
                        shape: least_padding_shape(grid),
                    });
                }
            }
        }
        let video = layout.segment(SegmentKind::Video);
        let grid = target_grid(layout);
        assert_eq!(
            grid.iter().product::<usize>(),
            video.len(),
            "the video segment does not match the latent grid"
        );
        let reference_spans = spans.len();
        spans.push(Span {
            start: video.start,
            grid,
            shape,
        });

        let mut tiling = VedaTiling {
            order: Vec::with_capacity(layout.len()),
            tile_starts: Vec::new(),
            tile_lengths: Vec::new(),
            reference_tiles: 0,
            video_tiles: 0,
            reference_tokens: spans[..reference_spans]
                .iter()
                .map(|span| span.grid.iter().product::<usize>())
                .sum(),
            target_tokens: video.len(),
        };
        let mut covered = vec![false; layout.len()];
        for (index, span) in spans.iter().enumerate() {
            let [frames, rows, columns] = span.grid;
            let TileShape { t, h, w } = span.shape;
            for row_box in (0..rows).step_by(h) {
                for column_box in (0..columns).step_by(w) {
                    for frame_box in (0..frames).step_by(t) {
                        let first = tiling.order.len();
                        for frame in frame_box..(frame_box + t).min(frames) {
                            for row in row_box..(row_box + h).min(rows) {
                                for column in column_box..(column_box + w).min(columns) {
                                    let packed =
                                        span.start + (frame * rows + row) * columns + column;
                                    tiling.order.push(packed);
                                    covered[packed] = true;
                                }
                            }
                        }
                        tiling.tile_starts.push(first);
                        tiling.tile_lengths.push(tiling.order.len() - first);
                    }
                }
            }
            if index + 1 == reference_spans {
                tiling.reference_tiles = tiling.tile_starts.len();
            }
        }
        tiling.video_tiles = tiling.tile_starts.len();
        let global: Vec<usize> = (0..layout.len()).filter(|&row| !covered[row]).collect();
        for chunk in global.chunks(VEDA_TILE) {
            tiling.tile_starts.push(tiling.order.len());
            tiling.tile_lengths.push(chunk.len());
            tiling.order.extend_from_slice(chunk);
        }
        tiling
    }

    pub fn tiles(&self) -> usize {
        self.tile_starts.len()
    }

    pub fn global_tiles(&self) -> usize {
        self.tiles() - self.video_tiles
    }

    fn rows(&self, tile: usize) -> std::ops::Range<usize> {
        self.tile_starts[tile]..self.tile_starts[tile] + self.tile_lengths[tile]
    }
}

/// How many key tiles of a column block each video query tile keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Keep {
    /// Every tile of the block that holds a row.
    All,
    /// `low` tiles, and `high` for the query tiles the Bresenham pattern of `fraction` picks.
    Budget {
        low: usize,
        high: usize,
        fraction: f64,
    },
}

/// The video key tiles `[start, end)` that one top-k covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnBlock {
    pub start: usize,
    pub end: usize,
    pub keep: Keep,
}

impl ColumnBlock {
    /// Tiles video query tile `row` keeps here, at most; empty tiles are never kept.
    pub fn allowed(&self, row: usize) -> usize {
        match self.keep {
            Keep::All => self.end - self.start,
            Keep::Budget {
                low,
                high,
                fraction,
            } => {
                let extra =
                    (((row + 1) as f64) * fraction).floor() > (row as f64 * fraction).floor();
                if extra { high } else { low }
            }
        }
    }
}

/// The keep of a block of `columns` tiles over `tokens` tokens at `sparsity`.
fn keep(sparsity: f64, tokens: usize, columns: usize) -> Keep {
    let ratio = 1.0 - sparsity;
    if ratio >= 1.0 {
        return Keep::All;
    }
    let ideal = tokens.div_ceil(VEDA_TILE) as f64;
    let budget = ratio * ideal * ideal / columns as f64;
    if budget >= columns as f64 {
        return Keep::All;
    }
    let low = (budget.floor() as usize).clamp(1, columns);
    // Rounded so that 2.3 - 2 gives 0.3, as Veda does.
    let fraction = ((budget - low as f64).clamp(0.0, 1.0) * 1e12).round() / 1e12;
    Keep::Budget {
        low,
        high: (low + 1).min(columns),
        fraction,
    }
}

/// The column blocks of a tiling: the references when they are tiled, then the target.
pub fn column_blocks(
    tiling: &VedaTiling,
    sparsity: f64,
    reference_sparsity: f64,
) -> Vec<ColumnBlock> {
    let mut blocks = Vec::new();
    if tiling.reference_tiles > 0 {
        blocks.push(ColumnBlock {
            start: 0,
            end: tiling.reference_tiles,
            keep: keep(
                reference_sparsity,
                tiling.reference_tokens,
                tiling.reference_tiles,
            ),
        });
    }
    let columns = tiling.video_tiles - tiling.reference_tiles;
    blocks.push(ColumnBlock {
        start: tiling.reference_tiles,
        end: tiling.video_tiles,
        keep: keep(sparsity, tiling.target_tokens, columns),
    });
    blocks
}

/// The video key tiles video query tile `row` keeps from its `scores` against every video tile,
/// ascending. Within a block the best scores win, ties going to the lower tile.
pub fn select_video_tiles(
    scores: &[f32],
    row: usize,
    tiling: &VedaTiling,
    blocks: &[ColumnBlock],
) -> Vec<usize> {
    let mut kept = Vec::new();
    for block in blocks {
        let mut candidates: Vec<(f32, usize)> = (block.start..block.end)
            .filter_map(|tile| {
                if tile == row {
                    Some((f32::INFINITY, tile))
                } else if tiling.tile_lengths[tile] > 0 {
                    Some((scores[tile], tile))
                } else {
                    None
                }
            })
            .collect();
        candidates.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
        candidates.truncate(block.allowed(row));
        kept.extend(candidates.into_iter().map(|(_, tile)| tile));
    }
    kept.sort_unstable();
    kept
}

/// Every tile query tile `query` attends token by token, ascending, given the video tiles it keeps.
pub fn attended_tiles(tiling: &VedaTiling, query: usize, kept_video: &[usize]) -> Vec<usize> {
    let all_rows = |tile: &usize| tiling.tile_lengths[*tile] > 0;
    if query >= tiling.video_tiles {
        return (0..tiling.tiles()).filter(all_rows).collect();
    }
    kept_video
        .iter()
        .copied()
        .chain((tiling.video_tiles..tiling.tiles()).filter(all_rows))
        .collect()
}

/// The predictor's view of one head's video tiles, `[video tiles, dim]`: the mean, maximum and
/// minimum of the tile's rows of `x` (`[tokens, heads, dim]` in packed order), projected.
pub fn predicted_tiles(
    x: &[f32],
    heads: usize,
    head: usize,
    tiling: &VedaTiling,
    projection: &Projection,
) -> Vec<f32> {
    let dim = x.len() / (tiling.order.len() * heads);
    let mut out = Vec::with_capacity(tiling.video_tiles * dim);
    for tile in 0..tiling.video_tiles {
        let mut pooled = vec![0.0f32; 3 * dim];
        let rows = tiling.rows(tile);
        if !rows.is_empty() {
            for index in 0..dim {
                let values = rows
                    .clone()
                    .map(|position| x[(tiling.order[position] * heads + head) * dim + index]);
                let sum: f64 = values.clone().map(f64::from).sum();
                pooled[index] = (sum / rows.len() as f64) as f32;
                pooled[dim + index] = values.clone().fold(f32::MIN, f32::max);
                pooled[2 * dim + index] = values.fold(f32::MAX, f32::min);
            }
        }
        for column in 0..dim {
            let projected: f64 = (0..3 * dim)
                .map(|row| pooled[row] as f64 * projection.weight(dim, head, row, column) as f64)
                .sum();
            out.push((projected + pooled[column] as f64) as f32);
        }
    }
    out
}

/// Veda over `[tokens, heads, dim]` query, key and value in packed order for one layer, computed
/// in f64 but for the scores. `tilings` holds a tiling for each of the plan's shapes the layer
/// uses, by shape index. Returns the output and how many video tiles the query tiles kept.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    heads: usize,
    dim: usize,
    head_shape: &[usize],
    tilings: &[Option<VedaTiling>],
    projections: &[Projection; 2],
    sparsity: f64,
    reference_sparsity: f64,
) -> (Vec<f32>, usize) {
    let tokens = query.len() / (heads * dim);
    let at = |tensor: &[f32], token: usize, head: usize, index: usize| {
        tensor[(token * heads + head) * dim + index] as f64
    };
    let scale = (dim as f64).sqrt().recip();
    let mut output = vec![0.0f32; tokens * heads * dim];
    let mut kept_total = 0;
    for head in 0..heads {
        let tiling = tilings[head_shape[head]]
            .as_ref()
            .expect("a tiling for every shape");
        let blocks = column_blocks(tiling, sparsity, reference_sparsity);
        let queries = predicted_tiles(query, heads, head, tiling, &projections[0]);
        let keys = predicted_tiles(key, heads, head, tiling, &projections[1]);
        for query_tile in 0..tiling.tiles() {
            let kept = if query_tile < tiling.video_tiles {
                let pooled = &queries[query_tile * dim..(query_tile + 1) * dim];
                let scores: Vec<f32> = keys
                    .chunks_exact(dim)
                    .map(|other| {
                        let dot: f32 = pooled.iter().zip(other).map(|(a, b)| a * b).sum();
                        dot * scale as f32
                    })
                    .collect();
                select_video_tiles(&scores, query_tile, tiling, &blocks)
            } else {
                Vec::new()
            };
            kept_total += kept.len();
            let keys: Vec<usize> = attended_tiles(tiling, query_tile, &kept)
                .into_iter()
                .flat_map(|tile| tiling.rows(tile).map(|position| tiling.order[position]))
                .collect();
            for position in tiling.rows(query_tile) {
                let token = tiling.order[position];
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|&other| {
                        (0..dim)
                            .map(|index| {
                                at(query, token, head, index) * at(key, other, head, index)
                            })
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let maximum = logits.iter().cloned().fold(f64::MIN, f64::max);
                let weights: Vec<f64> =
                    logits.iter().map(|logit| (logit - maximum).exp()).collect();
                let total: f64 = weights.iter().sum();
                for index in 0..dim {
                    let sum: f64 = keys
                        .iter()
                        .zip(&weights)
                        .map(|(&other, weight)| weight * at(value, other, head, index))
                        .sum();
                    output[(token * heads + head) * dim + index] = (sum / total) as f32;
                }
            }
        }
    }
    (output, kept_total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::layout::{KeyframeShape, ReferenceShape};

    #[test]
    fn decodes_fp8_e4m3() {
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xC0), -2.0);
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
        assert!(fp8_e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn picks_shapes_with_the_least_padding() {
        assert_eq!(all_shapes().len(), 36);
        assert_eq!(
            least_padding_shape([1, 16, 16]),
            TileShape::new(1, 8, 16).unwrap()
        );
        assert_eq!(
            least_padding_shape([2, 12, 21]),
            TileShape::new(2, 8, 8).unwrap()
        );
        assert_eq!(TileShape::parse("8x4x4"), TileShape::new(8, 4, 4));
        assert_eq!(TileShape::parse("8x4x2"), None);
    }

    #[test]
    fn tiles_the_spans_then_the_global_rows() {
        // 70 text tokens, 2 × 40 audio rows and a video grid of 5 × 3 × 6 patches.
        let layout = PackedLayout::text_to_video(70, 5, 6, 12, 40);
        let tiling = VedaTiling::new(&layout, TileShape::new(4, 4, 8).unwrap(), true);
        // Frame boxes of 4 and 1 in one row box and one column box.
        assert_eq!(tiling.video_tiles, 2);
        assert_eq!(&tiling.tile_lengths[..2], &[72, 18]);
        // The first box runs over frames 0 to 3 of the whole 3 × 6 plane.
        let video = layout.segment(SegmentKind::Video).start;
        assert_eq!(
            &tiling.order[..7],
            &[
                video,
                video + 1,
                video + 2,
                video + 3,
                video + 4,
                video + 5,
                video + 6
            ]
        );
        assert_eq!(tiling.order[18], video + 18);
        // 150 text and audio rows in global tiles of 128 and 22.
        assert_eq!(tiling.global_tiles(), 2);
        assert_eq!(&tiling.tile_lengths[2..], &[128, 22]);
        assert_eq!(tiling.order[90], 0);
        let mut sorted = tiling.order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..layout.len()).collect::<Vec<_>>());
    }

    #[test]
    fn tiles_keyframes_and_references_as_spans() {
        let layout = PackedLayout::new(
            20,
            5,
            8,
            8,
            12,
            &[KeyframeShape {
                frame_index: 0,
                latent_frames: 1,
                audio_frames: 0,
            }],
            &[ReferenceShape::Picture {
                latent_height: 32,
                latent_width: 32,
            }],
        );
        let tiling = VedaTiling::new(&layout, TileShape::new(2, 8, 8).unwrap(), true);
        // A 1 × 4 × 4 keyframe and a 1 × 16 × 16 picture, then 5 × 4 × 4 of target in 3 tiles.
        assert_eq!(tiling.reference_tiles, 3);
        assert_eq!(&tiling.tile_lengths[..3], &[16, 128, 128]);
        assert_eq!(tiling.video_tiles, 6);
        assert_eq!(tiling.reference_tokens, 16 + 256);
        assert_eq!(tiling.target_tokens, 80);
        let untiled = VedaTiling::new(&layout, TileShape::new(2, 8, 8).unwrap(), false);
        assert_eq!(untiled.reference_tiles, 0);
        assert_eq!(untiled.video_tiles, 3);
    }

    #[test]
    fn budgets_spread_the_fraction_and_keep_the_own_tile() {
        // 10 full tiles at 75 % sparsity keep 2.5 tiles a row.
        let block = ColumnBlock {
            start: 0,
            end: 10,
            keep: keep(0.75, 1280, 10),
        };
        assert_eq!(
            block.keep,
            Keep::Budget {
                low: 2,
                high: 3,
                fraction: 0.5
            }
        );
        let kept: usize = (0..10).map(|row| block.allowed(row)).sum();
        assert_eq!(kept, 25);
        // Padded tiles do not raise the budget: 1280 tokens over 20 tiles keep 1.25 a row.
        assert_eq!(
            keep(0.75, 1280, 20),
            Keep::Budget {
                low: 1,
                high: 2,
                fraction: 0.25
            }
        );
        assert_eq!(keep(0.0, 1280, 10), Keep::All);

        let layout = PackedLayout::text_to_video(10, 1, 8, 256, 5);
        let tiling = VedaTiling::new(&layout, TileShape::new(1, 4, 32).unwrap(), true);
        let blocks = [ColumnBlock {
            start: 0,
            end: 4,
            keep: Keep::Budget {
                low: 2,
                high: 2,
                fraction: 0.0,
            },
        }];
        // The own tile wins over a better score, then the best other.
        let scores = [0.0, 5.0, 1.0, 9.0];
        assert_eq!(select_video_tiles(&scores, 2, &tiling, &blocks), vec![2, 3]);
        assert_eq!(attended_tiles(&tiling, 2, &[2, 3]), vec![2, 3, 4]);
        assert_eq!(attended_tiles(&tiling, 4, &[]), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn plans_match_the_aspect_then_the_length() {
        let plan = |name: &str, grid: [usize; 3]| TilePlan {
            name: name.into(),
            grid,
            shapes: vec![TileShape::new(8, 4, 4).unwrap()],
            head_shape: vec![vec![0]],
        };
        let plans = [
            plan("16x9_t37", [37, 24, 42]),
            plan("16x9_t72", [72, 24, 42]),
            plan("1x1_t37", [37, 24, 24]),
        ];
        let (chosen, exact) = select_plan(&plans, [37, 24, 42]).unwrap();
        assert!(exact);
        assert_eq!(chosen.name, "16x9_t37");
        let (chosen, exact) = select_plan(&plans, [19, 12, 21]).unwrap();
        assert!(!exact);
        assert_eq!(chosen.name, "16x9_t37");
        // Portrait takes the transposed landscape plan.
        let (chosen, _) = select_plan(&plans, [70, 42, 24]).unwrap();
        assert_eq!(chosen.name, "16x9_t72_T");
        assert_eq!(chosen.grid, [72, 42, 24]);
    }

    #[test]
    fn keeping_every_tile_is_dense_attention() {
        let layout = PackedLayout::text_to_video(10, 2, 8, 16, 5);
        let (tokens, heads, dim) = (layout.len(), 2, 4);
        let values: Vec<f32> = (0..tokens * heads * dim)
            .map(|index| ((index * 37 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let tilings = [
            Some(VedaTiling::new(
                &layout,
                TileShape::new(2, 4, 16).unwrap(),
                true,
            )),
            Some(VedaTiling::new(
                &layout,
                TileShape::new(1, 8, 16).unwrap(),
                true,
            )),
        ];
        let projection = Projection {
            fp8: vec![0x38; heads * 3 * dim * dim],
            scales: vec![0.5; heads],
        };
        let (output, _) = reference(
            &values,
            &values,
            &values,
            heads,
            dim,
            &[0, 1],
            &tilings,
            &[projection.clone(), projection],
            0.0,
            0.0,
        );
        let scale = 0.5;
        for (token, head) in [(0, 0), (12, 1), (tokens - 1, 0)] {
            let at =
                |token: usize, index: usize| values[(token * heads + head) * dim + index] as f64;
            let scores: Vec<f64> = (0..tokens)
                .map(|other| {
                    (0..dim)
                        .map(|index| at(token, index) * at(other, index))
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for index in 0..dim {
                let expected: f64 = (0..tokens)
                    .map(|other| weights[other] * at(other, index))
                    .sum::<f64>()
                    / total;
                let actual = output[(token * heads + head) * dim + index] as f64;
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }
}
