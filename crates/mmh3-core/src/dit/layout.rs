//! The packed token sequence `[text | conditions | audio | video]` and the 3D positions its rotary
//! embedding uses.
//!
//! Conditions are clean latents the DiT sees next to the target and never denoises. Keyframes, as
//! first and last frame generation takes them, bring video rows on the target's latent grid, audio
//! rows, or both, placed on the time axis at the pixel frame they align with. References, as
//! reference to video generation takes them, bring rows on latent grids of their own, each on a
//! time span of its own between the text and the target.

use crate::dit::inputs::{DitInputs, Reference};
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
    /// Video rows of reference `n`, a picture or a clip.
    ReferenceVideo(usize),
    /// Stereo audio rows of reference `n`, a sound or a clip's soundtrack.
    ReferenceAudio(usize),
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

/// What a reference adds to the sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceShape {
    /// One latent frame of a picture.
    Picture {
        latent_height: usize,
        latent_width: usize,
    },
    /// Latent frames of a clip and the audio latent frames of its soundtrack, 0 without one.
    Video {
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_frames: usize,
    },
    /// Audio latent frames of a sound.
    Audio { audio_frames: usize },
}

impl ReferenceShape {
    pub fn of(reference: &Reference) -> Self {
        match reference {
            Reference::Picture(video) => ReferenceShape::Picture {
                latent_height: video.shape[2],
                latent_width: video.shape[3],
            },
            Reference::Video { video, audio } => ReferenceShape::Video {
                latent_frames: video.shape[1],
                latent_height: video.shape[2],
                latent_width: video.shape[3],
                audio_frames: audio.as_ref().map_or(0, |audio| audio.shape[2]),
            },
            Reference::Audio(audio) => ReferenceShape::Audio {
                audio_frames: audio.shape[2],
            },
        }
    }

    /// Units of the time axis the reference takes: 1 for a picture, one per audio latent frame,
    /// and for a clip the longer of its video and its soundtrack.
    fn span(&self) -> f64 {
        match *self {
            ReferenceShape::Picture { .. } => 1.0,
            ReferenceShape::Video {
                latent_frames,
                audio_frames,
                ..
            } => (audio_frames as f64).max(video_span(latent_frames)),
            ReferenceShape::Audio { audio_frames } => audio_frames as f64,
        }
    }
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

/// Units of the time axis `frames` latent frames of video take.
fn video_span(frames: usize) -> f64 {
    (0..frames)
        .map(|frame| FRAME_RESCALE * FRAME_PER_TOKEN[frame % FRAME_PER_TOKEN.len()])
        .sum()
}

/// The coordinates of the 2 × 2 patches along the height and width of a latent grid.
fn patch_axes(latent_height: usize, latent_width: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(
        latent_height.is_multiple_of(2) && latent_width.is_multiple_of(2),
        "latent height and width must be even"
    );
    let square_root_area = ((latent_height * latent_width) as f64).sqrt();
    (
        patch_axis(latent_height, square_root_area),
        patch_axis(latent_width, square_root_area),
    )
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
            &[],
        )
    }

    /// Layout for text-to-video with audio from keyframes and references: text, then per keyframe
    /// its video rows and its audio rows, then per reference its audio rows and its video rows,
    /// then the target's audio and video rows. The references take the time axis after the text
    /// in their order, and the target and its keyframes start after them.
    pub fn new(
        text_tokens: usize,
        latent_frames: usize,
        latent_height: usize,
        latent_width: usize,
        audio_frames: usize,
        keyframes: &[KeyframeShape],
        references: &[ReferenceShape],
    ) -> Self {
        let (height_axis, width_axis) = patch_axes(latent_height, latent_width);

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

        let mut cursor = text_tokens as f64;
        for reference in references {
            cursor += reference.span();
        }
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

        let mut reference_time = text_tokens as f64;
        for (index, reference) in references.iter().enumerate() {
            match *reference {
                ReferenceShape::Picture {
                    latent_height,
                    latent_width,
                } => {
                    let (height_axis, width_axis) = patch_axes(latent_height, latent_width);
                    let start = positions.len();
                    video_positions(reference_time, 1, &height_axis, &width_axis, &mut positions);
                    add(SegmentKind::ReferenceVideo(index), &positions, start, true);
                }
                ReferenceShape::Video {
                    latent_frames,
                    latent_height,
                    latent_width,
                    audio_frames,
                } => {
                    let (height_axis, width_axis) = patch_axes(latent_height, latent_width);
                    let start = positions.len();
                    audio_positions(reference_time, audio_frames, &width_axis, &mut positions);
                    add(SegmentKind::ReferenceAudio(index), &positions, start, false);
                    let start = positions.len();
                    video_positions(
                        reference_time,
                        latent_frames,
                        &height_axis,
                        &width_axis,
                        &mut positions,
                    );
                    add(SegmentKind::ReferenceVideo(index), &positions, start, true);
                }
                ReferenceShape::Audio { audio_frames } => {
                    let start = positions.len();
                    audio_positions(reference_time, audio_frames, &width_axis, &mut positions);
                    add(SegmentKind::ReferenceAudio(index), &positions, start, true);
                }
            }
            reference_time += reference.span();
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
        let references: Vec<ReferenceShape> =
            inputs.references.iter().map(ReferenceShape::of).collect();
        let mut layout = Self::new(
            inputs.context.shape[0],
            video_shape[1],
            video_shape[2],
            video_shape[3],
            inputs.audio.shape[2],
            &keyframes,
            &references,
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
                SegmentKind::KeyframeVideo(_)
                    | SegmentKind::KeyframeAudio(_)
                    | SegmentKind::ReferenceVideo(_)
                    | SegmentKind::ReferenceAudio(_)
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
            SegmentKind::KeyframeAudio(_) | SegmentKind::ReferenceAudio(_) | SegmentKind::Audio => {
                Modality::Audio
            }
            SegmentKind::KeyframeVideo(_) | SegmentKind::ReferenceVideo(_) | SegmentKind::Video => {
                Modality::Video
            }
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
        let layout = PackedLayout::new(5, 57, 4, 6, 320, &[first, last], &[]);
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
    fn places_references_on_their_own_grids_and_time_spans() {
        let picture = ReferenceShape::Picture {
            latent_height: 8,
            latent_width: 2,
        };
        let clip = ReferenceShape::Video {
            latent_frames: 7,
            latent_height: 4,
            latent_width: 4,
            audio_frames: 5,
        };
        let sound = ReferenceShape::Audio { audio_frames: 30 };
        let layout = PackedLayout::new(5, 2, 4, 6, 3, &[], &[picture, clip, sound]);
        let kinds: Vec<(SegmentKind, usize)> = layout
            .segments
            .iter()
            .map(|segment| (segment.kind, segment.len()))
            .collect();
        assert_eq!(
            kinds,
            [
                (SegmentKind::Text, 5),
                (SegmentKind::ReferenceVideo(0), 4),
                (SegmentKind::ReferenceAudio(1), 10),
                (SegmentKind::ReferenceVideo(1), 7 * 4),
                (SegmentKind::ReferenceAudio(2), 60),
                (SegmentKind::Audio, 6),
                (SegmentKind::Video, 2 * 6),
            ]
        );
        assert!(layout.has_conditions());
        let time = |kind| layout.positions[layout.segment(kind).start][0];
        // The picture takes one unit after the text, the clip the longer of its 7 latent frames
        // (85/3 units) and its 5 audio frames, and the sound one unit per frame.
        let clip_span = 5.0 / 3.0 * (1.0 + 4.0 * 4.0 + 1.0 + 4.0);
        assert_eq!(time(SegmentKind::ReferenceVideo(0)), 5.0);
        assert_eq!(time(SegmentKind::ReferenceAudio(1)), 6.0);
        assert_eq!(time(SegmentKind::ReferenceVideo(1)), 6.0);
        assert!((time(SegmentKind::ReferenceAudio(2)) - (6.0 + clip_span)).abs() < 1e-9);
        assert!((time(SegmentKind::Video) - (36.0 + clip_span)).abs() < 1e-9);
        assert_eq!(time(SegmentKind::Audio), time(SegmentKind::Video));
        // A picture's rows span its own grid, 4 patches tall and 1 wide, normalized by its own area.
        let rows = layout.segment(SegmentKind::ReferenceVideo(0));
        let heights: Vec<f64> = layout.positions[rows.start..rows.end]
            .iter()
            .map(|position| position[1])
            .collect();
        assert_eq!(heights, [-16.0, 0.0, 16.0, 32.0]);
        assert!(
            layout.positions[rows.start..rows.end]
                .iter()
                .all(|position| position[2] == 8.0)
        );
        // The soundtrack sits at the clip's width extremes, the sound at the target's.
        let soundtrack = layout.segment(SegmentKind::ReferenceAudio(1));
        assert_eq!(layout.positions[soundtrack.start][2], 0.0);
        assert_eq!(layout.positions[soundtrack.end - 1][2], 16.0);
        let sound = layout.segment(SegmentKind::ReferenceAudio(2));
        let target_audio = layout.segment(SegmentKind::Audio);
        assert_eq!(
            layout.positions[sound.start][2],
            layout.positions[target_audio.start][2]
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
