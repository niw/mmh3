//! Video VAE decoding plans shared by every backend: spatial tiles and temporal chunks.
//!
//! The decoder turns each latent token into a 4 × 16 × 16 block of pixels. It runs on 256-pixel
//! tiles that overlap and on chunks of 7 latent frames that overlap by 2, then blends the overlaps
//! linearly.

/// Pixels per latent position along height and width.
pub const SPATIAL_RATIO: usize = 16;
/// Frames per latent position along time.
pub const TEMPORAL_RATIO: usize = 4;
/// Frames the encoder consumes per clip.
pub const CLIP_LENGTH: usize = 17;
/// Latent frames the encoder drops from the end of a run.
pub const TOKEN_DROP: usize = 3;
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

/// Side of the square tiles a decode takes, the one the decoder was trained on. Its RoPE spans a
/// tile whatever the tile's size, so a larger tile spaces its tokens closer than the decoder knows,
/// and it decodes in blocks.
pub const DEFAULT_TILE_SIZE: usize = 256;

/// Pixels neighboring tiles overlap by at least, on every backend, so that a canvas is cut and
/// blended the same way whichever decodes it.
///
/// The grid this and a tile size produce, not the overlap itself, sets the decode cost: 32 gives
/// 6 × 4 tiles of 256 at 1344 × 768 where 48 and above give 7 × 4, and two rows rather than three
/// at 480.
pub const DEFAULT_TILE_OVERLAP_MIN: usize = 32;

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

/// Joins the encoder outputs of the tiles of a picture, `[tile latent pixels, channels]` each in
/// row-major tile order, into `[latent pixels, channels]`, as the reference does: each tile blends
/// linearly with the unblended tile above over their overlap, then with the unblended tile to its
/// left, and keeps its rows and columns up to the next tile.
pub fn blend_encoded_tiles(
    tiles: &[&[f32]],
    rows: &TileAxis,
    columns: &TileAxis,
    channels: usize,
) -> Vec<f32> {
    let (tile_height, tile_width) = (rows.length / SPATIAL_RATIO, columns.length / SPATIAL_RATIO);
    let height = rows.starts.last().unwrap() / SPATIAL_RATIO + tile_height;
    let width = columns.starts.last().unwrap() / SPATIAL_RATIO + tile_width;
    assert_eq!(tiles.len(), rows.starts.len() * columns.starts.len());
    let at = |tile: &[f32], y: usize, x: usize, channel: usize| {
        tile[(y * tile_width + x) * channels + channel]
    };
    let mut output = vec![0.0f32; height * width * channels];
    for (row, &top) in rows.starts.iter().enumerate() {
        for (column, &left) in columns.starts.iter().enumerate() {
            let tile = tiles[row * columns.starts.len() + column];
            let mut blended = tile.to_vec();
            if row > 0 {
                let above = tiles[(row - 1) * columns.starts.len() + column];
                let extent = (rows.overlaps[row - 1] / SPATIAL_RATIO).min(tile_height);
                for y in 0..extent {
                    let weight = y as f32 / extent as f32;
                    for x in 0..tile_width {
                        for channel in 0..channels {
                            blended[(y * tile_width + x) * channels + channel] =
                                at(above, tile_height - extent + y, x, channel) * (1.0 - weight)
                                    + at(tile, y, x, channel) * weight;
                        }
                    }
                }
            }
            if column > 0 {
                let left_tile = tiles[row * columns.starts.len() + column - 1];
                let extent = (columns.overlaps[column - 1] / SPATIAL_RATIO).min(tile_width);
                let vertical = blended.clone();
                for y in 0..tile_height {
                    for x in 0..extent {
                        let weight = x as f32 / extent as f32;
                        for channel in 0..channels {
                            blended[(y * tile_width + x) * channels + channel] =
                                at(left_tile, y, tile_width - extent + x, channel) * (1.0 - weight)
                                    + at(&vertical, y, x, channel) * weight;
                        }
                    }
                }
            }
            let kept_height = rows
                .overlaps
                .get(row)
                .map_or(tile_height, |overlap| tile_height - overlap / SPATIAL_RATIO);
            let kept_width = columns
                .overlaps
                .get(column)
                .map_or(tile_width, |overlap| tile_width - overlap / SPATIAL_RATIO);
            let (top, left) = (top / SPATIAL_RATIO, left / SPATIAL_RATIO);
            for y in 0..kept_height {
                for x in 0..kept_width {
                    let destination = ((top + y) * width + left + x) * channels;
                    let source = (y * tile_width + x) * channels;
                    output[destination..destination + channels]
                        .copy_from_slice(&blended[source..source + channels]);
                }
            }
        }
    }
    output
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
                if intra_tail != 0 && (latent_frames + index).is_multiple_of(CHUNK_TOKENS) {
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

/// Rotary angles `[tokens + suffix, 2 × frequencies]` of one tile of `height × width` latent tokens
/// of a picture decoder, whose rotary embedding has no time axis. Coordinates and suffix tokens are
/// those of `rope_angles`.
pub fn picture_rope_angles(
    height: usize,
    width: usize,
    suffix: usize,
    frequencies: usize,
    base: f32,
) -> Vec<f32> {
    let angles = rope_angles(1, height, width, suffix, frequencies, base);
    angles
        .chunks_exact(3 * frequencies)
        .flat_map(|token| &token[frequencies..])
        .copied()
        .collect()
}

/// Per-channel mean and standard deviation of the latents of MiniMax H3 Image's picture VAE, from
/// the `vae/config.json` of MiniMaxAI/MiniMax-H3-Image. Comfy-Org's FP16 file does not carry them.
/// They keep the digits the config writes them with, which round to the same `f32`.
#[allow(clippy::excessive_precision)]
pub const PICTURE_LATENTS_MEAN: [f32; 24] = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
];

#[allow(clippy::excessive_precision)]
pub const PICTURE_LATENTS_STD: [f32; 24] = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
];

/// Pixels neighboring tiles of the picture decoder overlap by at least, the reference's.
pub const PICTURE_TILE_OVERLAP_MIN: usize = 64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_the_time_axis_of_picture_angles() {
        let angles = picture_rope_angles(2, 3, 1, 2, 100.0);
        assert_eq!(angles.len(), (2 * 3 + 1) * 4);
        // Token (1, 2): height coordinate 0.5, width coordinate 2/3, both at frequency 1 first.
        let token = &angles[5 * 4..6 * 4];
        let pi = std::f32::consts::PI;
        assert!((token[0] - 2.0 * pi * 0.5).abs() < 1e-6);
        assert!((token[2] - 2.0 * pi * (2.0 / 3.0)).abs() < 1e-5);
        assert_eq!(&angles[6 * 4..], &[0.0; 4]);
    }

    #[test]
    fn blends_encoded_tiles_over_their_overlap() {
        // 448 pixels make two tiles 192 pixels apart, overlapping by 64 pixels, 4 latent columns.
        let (rows, columns) = (split_tiles(256, 256, 64), split_tiles(448, 256, 64));
        assert_eq!(
            (columns.starts.clone(), columns.overlaps.clone()),
            (vec![0, 192], vec![64])
        );
        let tile = |value: f32| vec![value; 16 * 16];
        let (first, second) = (tile(1.0), tile(3.0));
        let blended = blend_encoded_tiles(&[&first, &second], &rows, &columns, 1);
        assert_eq!(blended.len(), 16 * 28);
        let row: Vec<f32> = blended[..28].to_vec();
        assert_eq!(&row[..12], &[1.0; 12]);
        // The second tile starts at latent column 12 and ramps in over 4 columns.
        assert_eq!(&row[12..16], &[1.0, 1.5, 2.0, 2.5]);
        assert_eq!(&row[16..], &[3.0; 12]);
    }

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
