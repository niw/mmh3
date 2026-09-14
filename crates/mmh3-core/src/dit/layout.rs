//! The packed token sequence `[text | audio | video]` and the 3D positions its rotary embedding
//! uses.

use crate::dit::timestep::{Modality, StepTimesteps};

/// Video latent frame k covers FRAME_RESCALE × FRAME_PER_TOKEN[k % 5] units on the time axis.
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Spatial coordinates span this many units across a frame of equal height and width.
const SPATIAL_EXTENT: f64 = 32.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Text,
    Audio,
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub kind: SegmentKind,
    pub start: usize,
    pub end: usize,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[derive(Clone, Debug)]
pub struct PackedLayout {
    pub segments: Vec<Segment>,
    /// (time, height, width) per token.
    pub positions: Vec<[f64; 3]>,
    pub latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub audio_frames: usize,
}

/// Area-normalized coordinates of the 2 × 2 patches along one latent axis.
fn patch_axis(dimension: usize, square_root_area: f64) -> Vec<f64> {
    let ratio = dimension as f64 / square_root_area;
    let count = dimension / 2;
    (0..count)
        .map(|index| (index as f64 * (ratio / count as f64) + (1.0 - ratio) / 2.0) * SPATIAL_EXTENT)
        .collect()
}

impl PackedLayout {
    /// Layout for text-to-video with audio: text, then stereo audio rows (channel-major), then
    /// video patches.
    pub fn text_to_video(
        text_tokens: usize,
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_frames: usize,
    ) -> Self {
        assert!(
            latent_height.is_multiple_of(2) && latent_width.is_multiple_of(2),
            "latent height and width must be even"
        );
        let square_root_area = ((latent_height * latent_width) as f64).sqrt();
        let height_axis = patch_axis(latent_height, square_root_area);
        let width_axis = patch_axis(latent_width, square_root_area);

        let mut positions = Vec::new();
        positions.extend((0..text_tokens).map(|index| [index as f64, 0.0, 0.0]));

        let cursor = text_tokens as f64;
        let width_extremes = [width_axis[0], width_axis[width_axis.len() - 1]];
        for width in width_extremes {
            positions.extend((0..audio_frames).map(|frame| [cursor + frame as f64, 0.0, width]));
        }

        let mut frame_time = cursor;
        for frame in 0..latent_frames {
            for &height in &height_axis {
                for &width in &width_axis {
                    positions.push([frame_time, height, width]);
                }
            }
            frame_time += FRAME_RESCALE * FRAME_PER_TOKEN[frame % FRAME_PER_TOKEN.len()];
        }

        let audio_start = text_tokens;
        let video_start = audio_start + 2 * audio_frames;
        let segments = vec![
            Segment {
                kind: SegmentKind::Text,
                start: 0,
                end: audio_start,
            },
            Segment {
                kind: SegmentKind::Audio,
                start: audio_start,
                end: video_start,
            },
            Segment {
                kind: SegmentKind::Video,
                start: video_start,
                end: positions.len(),
            },
        ];
        PackedLayout {
            segments,
            positions,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn segment(&self, kind: SegmentKind) -> Segment {
        *self
            .segments
            .iter()
            .find(|segment| segment.kind == kind)
            .expect("segment is present")
    }

    /// Rotation angles per token, flattened `[tokens, 3 × frequencies]`: time, height and width
    /// axes in order. Positions are rounded to f32 before scaling, as the reference does.
    pub fn rope_angles(&self, inverse_frequencies: &[f32]) -> Vec<f32> {
        self.positions
            .iter()
            .flat_map(|position| {
                position.iter().flat_map(|&coordinate| {
                    inverse_frequencies
                        .iter()
                        .map(move |&frequency| coordinate as f32 * frequency)
                })
            })
            .collect()
    }

    /// Per-token row of the block AdaLN modulation table.
    pub fn modulation_rows(&self, timesteps: &StepTimesteps) -> Vec<usize> {
        self.segments
            .iter()
            .flat_map(|segment| {
                let row = timesteps
                    .modulation_row(timesteps.index_of(segment.kind), segment.kind.modality());
                std::iter::repeat_n(row, segment.len())
            })
            .collect()
    }
}

impl SegmentKind {
    pub fn modality(self) -> Modality {
        match self {
            SegmentKind::Text => Modality::Text,
            SegmentKind::Audio => Modality::Audio,
            SegmentKind::Video => Modality::Video,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_segments_and_positions() {
        let layout = PackedLayout::text_to_video(3, 6, 4, 6, 2);
        assert_eq!(layout.len(), 3 + 4 + 6 * 2 * 3);
        assert_eq!(
            layout.segment(SegmentKind::Audio),
            Segment {
                kind: SegmentKind::Audio,
                start: 3,
                end: 7
            }
        );
        assert_eq!(layout.positions[2], [2.0, 0.0, 0.0]);
        assert_eq!(layout.positions[4][0], 4.0);
        assert!(layout.positions[3][2] < 0.0 && layout.positions[5][2] > 0.0);
        let video = layout.segment(SegmentKind::Video);
        let frame_times: Vec<f64> = (0..6)
            .map(|frame| layout.positions[video.start + frame * 6][0])
            .collect();
        let expected = [
            3.0,
            3.0 + 5.0 / 3.0,
            3.0 + 25.0 / 3.0,
            3.0 + 15.0,
            3.0 + 65.0 / 3.0,
            3.0 + 85.0 / 3.0,
        ];
        for (actual, expected) in frame_times.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{frame_times:?}");
        }
    }
}
