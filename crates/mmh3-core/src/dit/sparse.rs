//! Sol-Attn, a training-free block-sparse attention for video diffusion (arXiv 2607.24027):
//! settings for the DiT and a CPU reference of the algorithm.
//!
//! Tokens form blocks of 64. Each query block attends a routed subset of key blocks token by token.
//! Every other key block contributes one pooled term, its mean key scored against the query block's
//! mean query, weighted by its length and carrying its summed values, so the softmax still covers
//! the whole sequence. A key block is routed when its pooled score clears tau standard deviations
//! of the query block's pooled score distribution, and the diagonal blocks, the conditioning prefix
//! and the dense query rows are always routed.

use crate::dit::layout::{PackedLayout, SegmentKind};

pub const SPARSE_BLOCK: usize = 64;

/// When and how sparsely the DiT uses Sol-Attn. The defaults follow ComfyUI's Model Sparse
/// Attention node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseAttention {
    /// Routing threshold in standard deviations of the pooled scores. Higher is sparser.
    pub tau: f32,
    /// Fraction of the sampling steps that run dense before sparse attention starts.
    pub start_fraction: f32,
    /// Sequences shorter than this stay dense.
    pub min_tokens: usize,
}

impl Default for SparseAttention {
    fn default() -> Self {
        SparseAttention {
            tau: 1.3,
            start_fraction: 0.2,
            min_tokens: 12_288,
        }
    }
}

impl SparseAttention {
    /// Whether step `step` of `steps` runs sparse.
    pub fn applies_to_step(&self, step: usize, steps: usize) -> bool {
        step as f32 >= self.start_fraction * steps as f32
    }
}

/// Block ranges `[start, end)` that are always exact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseSinks {
    /// Key blocks every query attends token by token.
    pub key_blocks: (usize, usize),
    /// Query blocks that attend every key token by token.
    pub query_blocks: (usize, usize),
}

impl SparseSinks {
    /// H3's conditioning: every query attends the text and audio prefix exactly, and the audio
    /// queries run dense.
    pub fn for_layout(layout: &PackedLayout) -> Self {
        let video = layout.segment(SegmentKind::Video);
        let audio = layout.segment(SegmentKind::Audio);
        let prefix_blocks = video.start.div_ceil(SPARSE_BLOCK);
        SparseSinks {
            key_blocks: (0, prefix_blocks),
            query_blocks: (audio.start / SPARSE_BLOCK, prefix_blocks),
        }
    }
}

/// Sol-Attn over `[tokens, heads, dim]` query, key and value, computed densely in f64. Returns the
/// output in the same layout and the fraction of (head, query block, key block) triples routed
/// exactly.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    dim: usize,
    tau: f32,
    scale: f32,
    sinks: SparseSinks,
) -> (Vec<f32>, f64) {
    let blocks = tokens.div_ceil(SPARSE_BLOCK);
    let length = |block: usize| (tokens - block * SPARSE_BLOCK).min(SPARSE_BLOCK) as f64;
    let at = |tensor: &[f32], token: usize, head: usize, index: usize| {
        tensor[(token * heads + head) * dim + index] as f64
    };
    let log2_scale = scale as f64 * std::f64::consts::LOG2_E;
    let mut output = vec![0.0f32; tokens * heads * dim];
    let mut routed = 0usize;
    for head in 0..heads {
        // Block means of the queries and keys and block sums of the values.
        let pool = |tensor: &[f32], mean: bool| -> Vec<Vec<f64>> {
            (0..blocks)
                .map(|block| {
                    let rows =
                        block * SPARSE_BLOCK..(block * SPARSE_BLOCK + SPARSE_BLOCK).min(tokens);
                    let divisor = if mean { length(block) } else { 1.0 };
                    (0..dim)
                        .map(|index| {
                            rows.clone()
                                .map(|token| at(tensor, token, head, index))
                                .sum::<f64>()
                                / divisor
                        })
                        .collect()
                })
                .collect()
        };
        let centroids = pool(query, true);
        let mut block_keys = pool(key, true);
        let value_sums = pool(value, false);
        let key_mean: Vec<f64> = (0..dim)
            .map(|index| block_keys.iter().map(|row| row[index]).sum::<f64>() / blocks as f64)
            .collect();
        for row in &mut block_keys {
            for (value, mean) in row.iter_mut().zip(&key_mean) {
                *value -= mean;
            }
        }
        let key_variance: Vec<f64> = (0..dim)
            .map(|index| {
                block_keys
                    .iter()
                    .map(|row| row[index] * row[index])
                    .sum::<f64>()
                    / blocks as f64
            })
            .collect();
        let dot =
            |left: &[f64], right: &[f64]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();

        for query_block in 0..blocks {
            let centroid = &centroids[query_block];
            let spread: f64 = centroid
                .iter()
                .zip(&key_variance)
                .map(|(c, variance)| c * c * variance)
                .sum();
            let threshold = tau as f64 * (spread * log2_scale * log2_scale + 1e-6).sqrt();
            let pooled: Vec<f64> = block_keys
                .iter()
                .map(|block_key| dot(centroid, block_key) * log2_scale)
                .collect();
            let exact: Vec<bool> = (0..blocks)
                .map(|key_block| {
                    pooled[key_block] > threshold
                        || query_block.abs_diff(key_block) <= 1
                        || (sinks.key_blocks.0..sinks.key_blocks.1).contains(&key_block)
                        || (sinks.query_blocks.0..sinks.query_blocks.1).contains(&query_block)
                })
                .collect();
            routed += exact.iter().filter(|&&flag| flag).count();
            for token in
                query_block * SPARSE_BLOCK..(query_block * SPARSE_BLOCK + SPARSE_BLOCK).min(tokens)
            {
                let row: Vec<f64> = (0..dim)
                    .map(|index| at(query, token, head, index))
                    .collect();
                let mut terms: Vec<(f64, f64, Vec<f64>)> = Vec::new();
                for key_block in 0..blocks {
                    if exact[key_block] {
                        for other in key_block * SPARSE_BLOCK
                            ..(key_block * SPARSE_BLOCK + SPARSE_BLOCK).min(tokens)
                        {
                            let centered: Vec<f64> = (0..dim)
                                .map(|index| at(key, other, head, index) - key_mean[index])
                                .collect();
                            let values = (0..dim)
                                .map(|index| at(value, other, head, index))
                                .collect();
                            terms.push((dot(&row, &centered) * log2_scale, 1.0, values));
                        }
                    } else {
                        terms.push((
                            pooled[key_block],
                            length(key_block),
                            value_sums[key_block].clone(),
                        ));
                    }
                }
                let maximum = terms.iter().map(|term| term.0).fold(f64::MIN, f64::max);
                let mut numerator = vec![0.0f64; dim];
                let mut denominator = 0.0f64;
                for (score, weight, values) in &terms {
                    let probability = (score - maximum).exp2();
                    denominator += probability * weight;
                    for (sum, value) in numerator.iter_mut().zip(values) {
                        *sum += probability * value;
                    }
                }
                for index in 0..dim {
                    output[(token * heads + head) * dim + index] =
                        (numerator[index] / denominator) as f32;
                }
            }
        }
    }
    (output, routed as f64 / (heads * blocks * blocks) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_everything_like_dense_attention() {
        // With every block exact the pooled terms vanish and the result is plain softmax attention.
        let (tokens, heads, dim) = (70, 1, 4);
        let values: Vec<f32> = (0..tokens * heads * dim)
            .map(|index| ((index * 37 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let sinks = SparseSinks {
            key_blocks: (0, 2),
            query_blocks: (0, 0),
        };
        let (output, routed) = reference(
            &values, &values, &values, tokens, heads, dim, 0.0, 0.5, sinks,
        );
        assert_eq!(routed, 1.0);
        for token in [0, 33, 69] {
            let scores: Vec<f64> = (0..tokens)
                .map(|other| {
                    (0..dim)
                        .map(|index| {
                            values[token * dim + index] as f64 * values[other * dim + index] as f64
                        })
                        .sum::<f64>()
                        * 0.5
                })
                .collect();
            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for index in 0..dim {
                let expected: f64 = (0..tokens)
                    .map(|other| weights[other] * values[other * dim + index] as f64)
                    .sum::<f64>()
                    / total;
                assert!((output[token * dim + index] as f64 - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn places_the_h3_sinks_on_the_conditioning() {
        let layout = PackedLayout::text_to_video(100, 7, 16, 28, 37);
        let sinks = SparseSinks::for_layout(&layout);
        // 100 text tokens and 74 audio rows put the video at token 174.
        assert_eq!(
            sinks,
            SparseSinks {
                key_blocks: (0, 3),
                query_blocks: (1, 3)
            }
        );
        let settings = SparseAttention::default();
        assert!(!settings.applies_to_step(0, 4) && settings.applies_to_step(1, 4));
        assert!(!settings.applies_to_step(3, 20) && settings.applies_to_step(4, 20));
    }
}
