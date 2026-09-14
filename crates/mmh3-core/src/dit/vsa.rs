//! Video Sparse Attention as FastVideo trained FastH3 with it (VSA-H3, 64-token tiles): the tiling
//! of the packed sequence and a CPU reference of the attention.
//!
//! The text and audio segments are cut into tiles of up to 64 consecutive tokens each. The video
//! patches are grouped into 4 × 4 × 4 cubes of (latent frame, patch row, patch column), smaller at
//! the far edges of the grid, and the DiT runs on the sequence with the video in cube order. Each
//! head scores every query tile against every key tile with the mean query and the mean key of the
//! tiles. A video query tile attends token by token to every text and audio tile and to the
//! best-scoring fraction of the video tiles, and a text or audio query tile attends to everything.
//! A coarse branch adds `gate · softmax(scores) · mean values` to every row of the query tile,
//! where the gate is a learned projection of the block input.

use crate::dit::layout::{PackedLayout, SegmentKind};

pub const VSA_TILE: usize = 64;
/// Latent frames, patch rows and patch columns of a video tile.
pub const VSA_CUBE: [usize; 3] = [4, 4, 4];
/// Fraction of the video tiles FastH3 was trained to drop.
pub const FASTH3_SPARSITY: f64 = 0.9;

/// The packed sequence cut into tiles, with the video in cube order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VsaPlan {
    /// First token of each tile in the reordered sequence.
    pub tile_starts: Vec<usize>,
    /// Tokens in each tile, at most `VSA_TILE`.
    pub tile_lengths: Vec<usize>,
    /// Text and audio tiles, which come first.
    pub prefix_tiles: usize,
    /// First token of the video.
    pub video_start: usize,
    /// For each video token in the reordered sequence, its index among the video tokens in the
    /// packed layout.
    pub video_order: Vec<usize>,
}

impl VsaPlan {
    pub fn for_layout(layout: &PackedLayout) -> Self {
        let mut tile_starts = Vec::new();
        let mut tile_lengths = Vec::new();
        for kind in [SegmentKind::Text, SegmentKind::Audio] {
            let segment = layout.segment(kind);
            for start in (segment.start..segment.end).step_by(VSA_TILE) {
                tile_starts.push(start);
                tile_lengths.push((segment.end - start).min(VSA_TILE));
            }
        }
        let prefix_tiles = tile_starts.len();

        let video = layout.segment(SegmentKind::Video);
        let grid = [
            layout.latent_frames,
            layout.latent_height / 2,
            layout.latent_width / 2,
        ];
        assert_eq!(
            grid.iter().product::<usize>(),
            video.len(),
            "the video segment does not match the latent grid"
        );
        let mut video_order = Vec::with_capacity(video.len());
        let mut start = video.start;
        for frame_tile in (0..grid[0]).step_by(VSA_CUBE[0]) {
            for row_tile in (0..grid[1]).step_by(VSA_CUBE[1]) {
                for column_tile in (0..grid[2]).step_by(VSA_CUBE[2]) {
                    let before = video_order.len();
                    for frame in frame_tile..(frame_tile + VSA_CUBE[0]).min(grid[0]) {
                        for row in row_tile..(row_tile + VSA_CUBE[1]).min(grid[1]) {
                            for column in column_tile..(column_tile + VSA_CUBE[2]).min(grid[2]) {
                                video_order.push((frame * grid[1] + row) * grid[2] + column);
                            }
                        }
                    }
                    let length = video_order.len() - before;
                    tile_starts.push(start);
                    tile_lengths.push(length);
                    start += length;
                }
            }
        }
        VsaPlan {
            tile_starts,
            tile_lengths,
            prefix_tiles,
            video_start: video.start,
            video_order,
        }
    }

    pub fn tiles(&self) -> usize {
        self.tile_starts.len()
    }

    pub fn video_tiles(&self) -> usize {
        self.tiles() - self.prefix_tiles
    }

    /// Video tiles each video query tile keeps at `sparsity`, as FastVideo counts them.
    pub fn kept_video_tiles(&self, sparsity: f64) -> usize {
        let tiles = self.video_tiles();
        (((1.0 - sparsity) * tiles as f64).ceil() as usize).clamp(1, tiles.max(1))
    }

    /// The packed sequence position of each position of the reordered sequence.
    pub fn sequence_order(&self, tokens: usize) -> Vec<usize> {
        (0..self.video_start)
            .chain(
                self.video_order
                    .iter()
                    .map(|&index| self.video_start + index),
            )
            .chain(self.video_start + self.video_order.len()..tokens)
            .collect()
    }

    /// Video rows of `width` values from the packed order into cube order.
    pub fn reorder_video<T: Copy>(&self, rows: &[T], width: usize) -> Vec<T> {
        assert_eq!(rows.len(), self.video_order.len() * width);
        self.video_order
            .iter()
            .flat_map(|&index| rows[index * width..(index + 1) * width].iter().copied())
            .collect()
    }

    /// Video rows of `width` values from cube order back into the packed order.
    pub fn restore_video<T: Copy + Default>(&self, rows: &[T], width: usize) -> Vec<T> {
        assert_eq!(rows.len(), self.video_order.len() * width);
        let mut restored = vec![T::default(); rows.len()];
        for (position, &index) in self.video_order.iter().enumerate() {
            restored[index * width..(index + 1) * width]
                .copy_from_slice(&rows[position * width..(position + 1) * width]);
        }
        restored
    }
}

/// Tiles every query tile attends token by token: all tiles for text and audio query tiles, and
/// for video query tiles the text and audio tiles plus the `kept` best-scoring video tiles in
/// ascending order. `scores` holds one query tile's scores against every tile.
pub fn select_tiles(
    scores: &[f32],
    query_tile: usize,
    prefix_tiles: usize,
    kept: usize,
) -> Vec<usize> {
    let tiles = scores.len();
    if query_tile < prefix_tiles || kept >= tiles - prefix_tiles {
        return (0..tiles).collect();
    }
    let mut video: Vec<usize> = (prefix_tiles..tiles).collect();
    // Ties go to the lower tile index.
    video.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then(left.cmp(&right))
    });
    video.truncate(kept);
    video.sort_unstable();
    (0..prefix_tiles).chain(video).collect()
}

/// VSA over `[tokens, heads, dim]` query, key and value in the plan's order, computed in f64.
/// `gate`, in the same layout, enables the coarse branch.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    gate: Option<&[f32]>,
    plan: &VsaPlan,
    heads: usize,
    dim: usize,
    scale: f32,
    kept: usize,
) -> Vec<f32> {
    let tokens = query.len() / (heads * dim);
    let tiles = plan.tiles();
    let at = |tensor: &[f32], token: usize, head: usize, index: usize| {
        tensor[(token * heads + head) * dim + index] as f64
    };
    let dot = |left: &[f64], right: &[f64]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let mut output = vec![0.0f32; tokens * heads * dim];
    for head in 0..heads {
        let pool = |tensor: &[f32]| -> Vec<Vec<f64>> {
            (0..tiles)
                .map(|tile| {
                    let rows =
                        plan.tile_starts[tile]..plan.tile_starts[tile] + plan.tile_lengths[tile];
                    (0..dim)
                        .map(|index| {
                            rows.clone()
                                .map(|token| at(tensor, token, head, index))
                                .sum::<f64>()
                                / plan.tile_lengths[tile] as f64
                        })
                        .collect()
                })
                .collect()
        };
        let (pooled_query, pooled_key, pooled_value) = (pool(query), pool(key), pool(value));
        for (query_tile, pooled_query) in pooled_query.iter().enumerate() {
            let scores: Vec<f64> = pooled_key
                .iter()
                .map(|pooled| dot(pooled_query, pooled) * scale as f64)
                .collect();
            let scores_f32: Vec<f32> = scores.iter().map(|&score| score as f32).collect();
            let selected = select_tiles(&scores_f32, query_tile, plan.prefix_tiles, kept);

            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            let coarse: Vec<f64> = (0..dim)
                .map(|index| {
                    weights
                        .iter()
                        .zip(&pooled_value)
                        .map(|(weight, pooled)| weight * pooled[index])
                        .sum::<f64>()
                        / total
                })
                .collect();

            let start = plan.tile_starts[query_tile];
            for token in start..start + plan.tile_lengths[query_tile] {
                let row: Vec<f64> = (0..dim)
                    .map(|index| at(query, token, head, index))
                    .collect();
                let keys: Vec<usize> = selected
                    .iter()
                    .flat_map(|&tile| {
                        plan.tile_starts[tile]..plan.tile_starts[tile] + plan.tile_lengths[tile]
                    })
                    .collect();
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|&other| {
                        let key_row: Vec<f64> =
                            (0..dim).map(|index| at(key, other, head, index)).collect();
                        dot(&row, &key_row) * scale as f64
                    })
                    .collect();
                let maximum = logits.iter().cloned().fold(f64::MIN, f64::max);
                let probabilities: Vec<f64> =
                    logits.iter().map(|logit| (logit - maximum).exp()).collect();
                let total: f64 = probabilities.iter().sum();
                for index in 0..dim {
                    let mut sum: f64 = keys
                        .iter()
                        .zip(&probabilities)
                        .map(|(&other, probability)| probability * at(value, other, head, index))
                        .sum::<f64>()
                        / total;
                    if let Some(gate) = gate {
                        sum += at(gate, token, head, index) * coarse[index];
                    }
                    output[(token * heads + head) * dim + index] = sum as f32;
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_the_prefix_and_the_video_cubes() {
        // 70 text tokens, 2 × 40 audio rows and a video grid of 5 × 3 × 6 patches.
        let layout = PackedLayout::text_to_video(70, 5, 6, 12, 40);
        let plan = VsaPlan::for_layout(&layout);
        assert_eq!(plan.prefix_tiles, 4);
        assert_eq!(&plan.tile_starts[..4], &[0, 64, 70, 134]);
        assert_eq!(&plan.tile_lengths[..4], &[64, 6, 64, 16]);
        // Frame tiles of 4 and 1, one row tile of 3, column tiles of 4 and 2.
        assert_eq!(plan.video_tiles(), 4);
        assert_eq!(&plan.tile_lengths[4..], &[48, 24, 12, 6]);
        assert_eq!(plan.tile_starts[4], 150);
        assert_eq!(plan.tile_starts[7], 150 + 48 + 24 + 12);
        // The first cube starts with frame 0, row 0, columns 0 to 3, then row 1.
        assert_eq!(&plan.video_order[..6], &[0, 1, 2, 3, 6, 7]);
        let order = plan.sequence_order(layout.len());
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..layout.len()).collect::<Vec<_>>());

        let rows: Vec<u32> = (0..90 * 2).collect();
        assert_eq!(plan.restore_video(&plan.reorder_video(&rows, 2), 2), rows);
    }

    #[test]
    fn counts_kept_tiles_like_fastvideo() {
        let layout = PackedLayout::text_to_video(36, 32, 48, 84, 124);
        let plan = VsaPlan::for_layout(&layout);
        // 8 frame tiles × 6 row tiles × 11 column tiles.
        assert_eq!(plan.video_tiles(), 528);
        assert_eq!(plan.kept_video_tiles(FASTH3_SPARSITY), 53);
        assert_eq!(plan.kept_video_tiles(0.0), 528);
    }

    #[test]
    fn keeping_every_tile_without_a_gate_is_dense_attention() {
        let layout = PackedLayout::text_to_video(10, 1, 4, 8, 5);
        let plan = VsaPlan::for_layout(&layout);
        let (tokens, heads, dim) = (layout.len(), 2, 4);
        let values: Vec<f32> = (0..tokens * heads * dim)
            .map(|index| ((index * 37 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let output = reference(
            &values,
            &values,
            &values,
            None,
            &plan,
            heads,
            dim,
            0.5,
            plan.video_tiles(),
        );
        for (token, head) in [(0, 0), (12, 1), (tokens - 1, 0)] {
            let at =
                |token: usize, index: usize| values[(token * heads + head) * dim + index] as f64;
            let scores: Vec<f64> = (0..tokens)
                .map(|other| {
                    (0..dim)
                        .map(|index| at(token, index) * at(other, index))
                        .sum::<f64>()
                        * 0.5
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

    #[test]
    fn selects_the_prefix_and_the_best_video_tiles() {
        let scores = [0.0, 0.0, 3.0, 1.0, 5.0, 1.0];
        assert_eq!(select_tiles(&scores, 3, 2, 2), vec![0, 1, 2, 4]);
        // Ties go to the lower index, and prefix queries see everything.
        assert_eq!(select_tiles(&scores, 3, 2, 3), vec![0, 1, 2, 3, 4]);
        assert_eq!(select_tiles(&scores, 1, 2, 1), vec![0, 1, 2, 3, 4, 5]);
    }
}
