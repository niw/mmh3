//! The packed token sequence `[text | conditions | audio | video]` and the 3D positions its rotary
//! embedding uses.
//!
//! Conditions are clean latents the DiT sees next to the target and never denoises, such as the
//! keyframes of first and last frame generation. Each keyframe brings video rows on the target's
//! latent grid, audio rows, or both, placed on the time axis at the pixel frame they align with.

use crate::dit::inputs::DitInputs;
use crate::dit::timestep::{Modality, StepTimesteps};

/// Video latent frame k covers FRAME_RESCALE × FRAME_PER_TOKEN[k % 5] units on the time axis.
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;
/// Spatial coordinates span this many units across a frame of equal height and width.
const SPATIAL_EXTENT: f64 = 32.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Text,
    /// Video rows of keyframe `n`.
    KeyframeVideo(usize),
    /// Stereo audio rows of keyframe `n`.
    KeyframeAudio(usize),
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

/// What a keyframe adds to the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyframeShape {
    /// Pixel frame of the target the keyframe's first frame aligns with.
    pub frame_index: usize,
    /// Latent frames of video on the target's latent grid, 0 without video.
    pub latent_frames: usize,
    /// Audio latent frames, 0 without audio.
    pub audio_frames: usize,
}

#[derive(Clone, Debug)]
pub struct PackedLayout {
    pub segments: Vec<Segment>,
    /// (time, height, width) per token.
    pub positions: Vec<[f64; 3]>,
    /// AdaLN branch of each text token: video for the vision blocks of pictures in the prompt.
    pub text_modalities: Vec<Modality>,
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

/// Positions of `frames` latent frames of patches from `start` on the time axis.
fn video_positions(
    start: f64,
    frames: usize,
    height_axis: &[f64],
    width_axis: &[f64],
    positions: &mut Vec<[f64; 3]>,
) {
    let mut frame_time = start;
    for frame in 0..frames {
        for &height in height_axis {
            for &width in width_axis {
                positions.push([frame_time, height, width]);
            }
        }
        frame_time += FRAME_RESCALE * FRAME_PER_TOKEN[frame % FRAME_PER_TOKEN.len()];
    }
}

/// Positions of stereo audio rows, channel-major, from `start` on the time axis, with the two
/// channels at the width extremes.
fn audio_positions(start: f64, frames: usize, width_axis: &[f64], positions: &mut Vec<[f64; 3]>) {
    let width_extremes = [width_axis[0], width_axis[width_axis.len() - 1]];
    for width in width_extremes {
        positions.extend((0..frames).map(|frame| [start + frame as f64, 0.0, width]));
    }
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
        Self::new(
            text_tokens,
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
            &[],
        )
    }

    /// Layout for text-to-video with audio from keyframes: text, then per keyframe its video rows
    /// and its audio rows, then the target's audio and video rows.
    pub fn new(
        text_tokens: usize,
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_frames: usize,
        keyframes: &[KeyframeShape],
    ) -> Self {
        assert!(
            latent_height.is_multiple_of(2) && latent_width.is_multiple_of(2),
            "latent height and width must be even"
        );
        let square_root_area = ((latent_height * latent_width) as f64).sqrt();
        let height_axis = patch_axis(latent_height, square_root_area);
        let width_axis = patch_axis(latent_width, square_root_area);

        let mut positions: Vec<[f64; 3]> = (0..text_tokens)
            .map(|index| [index as f64, 0.0, 0.0])
            .collect();
        let mut segments = vec![Segment {
            kind: SegmentKind::Text,
            start: 0,
            end: text_tokens,
        }];
        let mut add = |kind, positions: &Vec<[f64; 3]>, start: usize, required: bool| {
            if required || positions.len() > start {
                segments.push(Segment {
                    kind,
                    start,
                    end: positions.len(),
                });
            }
        };

        let cursor = text_tokens as f64;
        for (index, keyframe) in keyframes.iter().enumerate() {
            let time = cursor + FRAME_RESCALE * keyframe.frame_index as f64;
            let start = positions.len();
            video_positions(
                time,
                keyframe.latent_frames,
                &height_axis,
                &width_axis,
                &mut positions,
            );
            add(SegmentKind::KeyframeVideo(index), &positions, start, false);
            let start = positions.len();
            audio_positions(time, keyframe.audio_frames, &width_axis, &mut positions);
            add(SegmentKind::KeyframeAudio(index), &positions, start, false);
        }

        let start = positions.len();
        audio_positions(cursor, audio_frames, &width_axis, &mut positions);
        add(SegmentKind::Audio, &positions, start, true);
        let start = positions.len();
        video_positions(
            cursor,
            latent_frames,
            &height_axis,
            &width_axis,
            &mut positions,
        );
        add(SegmentKind::Video, &positions, start, true);
        PackedLayout {
            segments,
            positions,
            text_modalities: vec![Modality::Text; text_tokens],
            latent_frames,
            latent_height,
            latent_width,
            audio_frames,
        }
    }

    /// The layout of a DiT call's inputs.
    pub fn for_inputs(inputs: &DitInputs) -> Self {
        let video_shape = &inputs.video.shape;
        let keyframes: Vec<KeyframeShape> = inputs
            .keyframes
            .iter()
            .map(|keyframe| KeyframeShape {
                frame_index: keyframe.frame_index,
                latent_frames: keyframe.video.as_ref().map_or(0, |video| video.shape[1]),
                audio_frames: keyframe.audio.as_ref().map_or(0, |audio| audio.shape[2]),
            })
            .collect();
        let mut layout = Self::new(
            inputs.context.shape[0],
            video_shape[1],
            video_shape[2],
            video_shape[3],
            inputs.audio.shape[2],
            &keyframes,
        );
        if !inputs.context_modalities.is_empty() {
            assert_eq!(
                inputs.context_modalities.len(),
                inputs.context.shape[0],
                "one modality per context token"
            );
            layout.text_modalities = inputs.context_modalities.clone();
        }
        layout
    }

    /// Whether the sequence holds conditions, whose rows run at their own timesteps.
    pub fn has_conditions(&self) -> bool {
        self.segments.iter().any(|segment| {
            matches!(
                segment.kind,
                SegmentKind::KeyframeVideo(_) | SegmentKind::KeyframeAudio(_)
            )
        })
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
                let timestep = timesteps.index_of(segment.kind);
                (0..segment.len()).map(move |offset| {
                    let modality = match segment.kind {
                        SegmentKind::Text => self.text_modalities[offset],
                        kind => kind.modality(),
                    };
                    timesteps.modulation_row(timestep, modality)
                })
            })
            .collect()
    }
}

impl SegmentKind {
    pub fn modality(self) -> Modality {
        match self {
            SegmentKind::Text => Modality::Text,
            SegmentKind::KeyframeAudio(_) | SegmentKind::Audio => Modality::Audio,
            SegmentKind::KeyframeVideo(_) | SegmentKind::Video => Modality::Video,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_keyframes_between_the_text_and_the_target() {
        let first = KeyframeShape {
            frame_index: 0,
            latent_frames: 1,
            audio_frames: 0,
        };
        let last = KeyframeShape {
            frame_index: 191,
            latent_frames: 1,
            audio_frames: 3,
        };
        let layout = PackedLayout::new(5, 57, 4, 6, 320, &[first, last]);
        let kinds: Vec<(SegmentKind, usize)> = layout
            .segments
            .iter()
            .map(|segment| (segment.kind, segment.len()))
            .collect();
        assert_eq!(
            kinds,
            [
                (SegmentKind::Text, 5),
                (SegmentKind::KeyframeVideo(0), 6),
                (SegmentKind::KeyframeVideo(1), 6),
                (SegmentKind::KeyframeAudio(1), 6),
                (SegmentKind::Audio, 640),
                (SegmentKind::Video, 57 * 6),
            ]
        );
        assert!(layout.has_conditions());
        // The first keyframe sits on the target's first frame, the last one 5/3 · 191 later, which
        // no target latent frame shares.
        let video = layout.segment(SegmentKind::Video);
        assert_eq!(layout.positions[5], layout.positions[video.start]);
        let last_time = layout.positions[layout.segment(SegmentKind::KeyframeVideo(1)).start][0];
        assert!((last_time - (5.0 + 5.0 / 3.0 * 191.0)).abs() < 1e-9);
        let audio = layout.segment(SegmentKind::KeyframeAudio(1));
        assert_eq!(layout.positions[audio.start + 1][0], last_time + 1.0);
        assert_eq!(
            layout.positions[audio.start][2],
            layout.positions[layout.segment(SegmentKind::Audio).start][2]
        );
    }

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
