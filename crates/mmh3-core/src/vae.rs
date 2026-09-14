//! Video VAE decoding plans shared by every backend: spatial tiles and temporal chunks.
//!
//! The decoder turns each latent token into a 4 × 16 × 16 block of pixels. It runs on 256-pixel
//! tiles that overlap and on chunks of 7 latent frames that overlap by 2, then blends the overlaps
//! linearly.

/// Pixels per latent position along height and width.
pub const SPATIAL_RATIO: usize = 16;
/// Frames per latent position along time.
pub const TEMPORAL_RATIO: usize = 4;
/// Frames the encoder consumed per clip.
const CLIP_LENGTH: usize = 17;
/// Latent frames the encoder dropped from the end.
const TOKEN_DROP: usize = 3;
/// Latent frames that start each chunk: ceil(CLIP_LENGTH / TEMPORAL_RATIO).
pub const CHUNK_TOKENS: usize = CLIP_LENGTH.div_ceil(TEMPORAL_RATIO);
/// Extra latent frames each chunk decodes past its start.
pub const CHUNK_OVERLAP_TOKENS: usize = (CHUNK_TOKENS - TOKEN_DROP % CHUNK_TOKENS) % CHUNK_TOKENS;
/// Leading frames of each decoded part that are discarded.
pub const FRAME_PRE_PADDING: usize =
    (TEMPORAL_RATIO - CLIP_LENGTH % TEMPORAL_RATIO) % TEMPORAL_RATIO;
/// Frames blended between consecutive chunks.
pub const FRAME_OVERLAP: usize = CHUNK_OVERLAP_TOKENS * TEMPORAL_RATIO - FRAME_PRE_PADDING;
/// Frames in the first decoded part of a chunk, before the pre-padding is dropped.
pub const CHUNK_FRAMES: usize = CHUNK_TOKENS * TEMPORAL_RATIO;

/// Tiles along one axis, in pixels: every tile has the same length and neighbors overlap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileAxis {
    pub starts: Vec<usize>,
    pub length: usize,
    /// Overlap between tile i and tile i + 1.
    pub overlaps: Vec<usize>,
}

pub fn split_tiles(length: usize, tile_size: usize, overlap_min: usize) -> TileAxis {
    if tile_size >= length {
        return TileAxis {
            starts: vec![0],
            length,
            overlaps: Vec::new(),
        };
    }
    let mut count = length.div_ceil(tile_size);
    let (mut overlaps, remaining) = loop {
        let overlaps = vec![overlap_min; count - 1];
        let covered = tile_size * count - overlaps.iter().sum::<usize>();
        if covered >= length {
            break (overlaps, covered - length);
        }
        count += 1;
    };
    for unit in 0..remaining / SPATIAL_RATIO {
        let index = unit % (count - 1);
        overlaps[index] += SPATIAL_RATIO;
    }
    let mut starts = vec![0];
    for overlap in &overlaps {
        starts.push(starts.last().unwrap() + tile_size - overlap);
    }
    TileAxis {
        starts,
        length: tile_size,
        overlaps,
    }
}

/// How a latent of `latent_frames` frames decodes in chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemporalPlan {
    /// Copies of the last latent frame appended so the chunks cover the latent.
    pub pad_tokens: usize,
    pub chunks: usize,
    pub frames: usize,
}

impl TemporalPlan {
    pub fn new(latent_frames: usize) -> Self {
        let pseudo_total = latent_frames + TOKEN_DROP;
        let mut pad_tokens = (CHUNK_TOKENS - pseudo_total % CHUNK_TOKENS) % CHUNK_TOKENS;
        let mut chunks = (pseudo_total + pad_tokens) / CHUNK_TOKENS - 1;
        if chunks < 1 {
            pad_tokens += CHUNK_TOKENS;
            chunks += 1;
        }
        let padded = latent_frames + pad_tokens;
        let mut frames = 0;
        let mut final_overlap = 0;
        for chunk in 0..chunks {
            let (first, last) = Self::chunk_tokens(chunk, padded);
            let clip_frames = (last - first) * TEMPORAL_RATIO;
            for part in 0..2 {
                let start = part * CHUNK_FRAMES;
                let end = (start + CHUNK_FRAMES).min(clip_frames);
                let kept = end.saturating_sub(start).saturating_sub(FRAME_PRE_PADDING);
                if part == 0 {
                    frames += kept;
                } else {
                    final_overlap = kept;
                }
            }
        }
        frames += final_overlap;
        let padded_frames: usize = (0..pad_tokens)
            .map(|index| {
                let intra_tail = CLIP_LENGTH % TEMPORAL_RATIO;
                if intra_tail != 0 && (latent_frames + index) % CHUNK_TOKENS == 0 {
                    intra_tail
                } else {
                    TEMPORAL_RATIO
                }
            })
            .sum();
        TemporalPlan {
            pad_tokens,
            chunks,
            frames: frames - padded_frames,
        }
    }

    /// Latent frames `[first, last)` of the padded latent that chunk `chunk` decodes.
    pub fn chunk_tokens(chunk: usize, padded_frames: usize) -> (usize, usize) {
        let first = chunk * CHUNK_TOKENS;
        (
            (first).min(padded_frames),
            (first + CHUNK_TOKENS + CHUNK_OVERLAP_TOKENS).min(padded_frames),
        )
    }
}

/// Rotary angles `[tokens + suffix, 3 × frequencies]` of one tile of `frames × height × width`
/// latent tokens. Coordinates span (−1, 1) per axis and the suffix tokens sit at the origin.
pub fn rope_angles(
    frames: usize,
    height: usize,
    width: usize,
    suffix: usize,
    frequencies: usize,
    base: f32,
) -> Vec<f32> {
    let inverse: Vec<f32> = (0..frequencies)
        .map(|index| base.powf(-(index as f32) / frequencies as f32))
        .collect();
    let coordinate = |index: usize, size: usize| ((index as f32 + 0.5) / size as f32) * 2.0 - 1.0;
    let mut angles = Vec::with_capacity((frames * height * width + suffix) * 3 * frequencies);
    for frame in 0..frames {
        for y in 0..height {
            for x in 0..width {
                for position in [
                    coordinate(frame, frames),
                    coordinate(y, height),
                    coordinate(x, width),
                ] {
                    angles.extend(
                        inverse
                            .iter()
                            .map(|&frequency| 2.0 * std::f32::consts::PI * position * frequency),
                    );
                }
            }
        }
    }
    angles.resize(angles.len() + suffix * 3 * frequencies, 0.0);
    angles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_the_reference() {
        assert_eq!(
            split_tiles(200, 256, 64),
            TileAxis {
                starts: vec![0],
                length: 200,
                overlaps: vec![]
            }
        );
        assert_eq!(
            split_tiles(448, 256, 64),
            TileAxis {
                starts: vec![0, 192],
                length: 256,
                overlaps: vec![64]
            }
        );
        assert_eq!(
            split_tiles(768, 256, 64),
            TileAxis {
                starts: vec![0, 160, 336, 512],
                length: 256,
                overlaps: vec![96, 80, 80]
            }
        );
        assert_eq!(split_tiles(1344, 256, 64).starts.len(), 7);
        assert_eq!(
            split_tiles(96, 32, 16),
            TileAxis {
                starts: vec![0, 16, 32, 48, 64],
                length: 32,
                overlaps: vec![16; 4]
            }
        );
    }

    #[test]
    fn plans_chunks_like_the_reference() {
        assert_eq!(
            (
                CHUNK_TOKENS,
                CHUNK_OVERLAP_TOKENS,
                FRAME_PRE_PADDING,
                FRAME_OVERLAP
            ),
            (5, 2, 3, 5)
        );
        assert_eq!(
            TemporalPlan::new(7),
            TemporalPlan {
                pad_tokens: 0,
                chunks: 1,
                frames: 22
            }
        );
        assert_eq!(
            TemporalPlan::new(12),
            TemporalPlan {
                pad_tokens: 0,
                chunks: 2,
                frames: 39
            }
        );
        assert_eq!(
            TemporalPlan::new(37),
            TemporalPlan {
                pad_tokens: 0,
                chunks: 7,
                frames: 124
            }
        );
        assert_eq!(TemporalPlan::chunk_tokens(1, 12), (5, 12));
    }
}
