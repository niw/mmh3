//! Per-step timesteps of the video and audio streams and the AdaLN rows they select.

use crate::dit::layout::{PackedLayout, SegmentKind};
use crate::tensor::Tensor;

/// Maps a sigma from one exponential flow shift to another through the unshifted grid.
pub fn time_shift_sigma(sigma: f32, from_shift: f32, to_shift: f32) -> f32 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// AdaLN branch selected by a token. The projections hold one set of vectors per modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Video = 0,
    Text = 1,
    Audio = 2,
}

pub const MODALITY_COUNT: usize = 3;

/// Timestep of the video rows of conditions, at least: the rows are nearly clean, with 0.1% of
/// noise mixed in.
pub const VIDEO_CONDITION_TIMESTEP: f32 = 0.999;
/// Timestep of the audio rows of conditions, at least.
pub const AUDIO_CONDITION_TIMESTEP: f32 = 1.0;

/// The distinct timesteps t = 1 − sigma of one model call, in ascending order.
#[derive(Clone, Debug, PartialEq)]
pub struct StepTimesteps {
    pub values: Vec<f32>,
    /// Index into `values` for video and text tokens.
    pub video: usize,
    /// Index into `values` for audio tokens.
    pub audio: usize,
    /// Indices into `values` for the video and audio rows of conditions, when a call has them.
    pub conditions: Option<(usize, usize)>,
}

impl StepTimesteps {
    /// Text follows the video timestep. Audio runs on its own shifted schedule derived from the
    /// video sigma.
    pub fn new(sigma_video: f32, shift_video: f32, shift_audio: f32) -> Self {
        Self::build(sigma_video, shift_video, shift_audio, false)
    }

    /// `new` with the timesteps of conditions, which never go below `VIDEO_CONDITION_TIMESTEP`
    /// and `AUDIO_CONDITION_TIMESTEP`.
    pub fn with_conditions(sigma_video: f32, shift_video: f32, shift_audio: f32) -> Self {
        Self::build(sigma_video, shift_video, shift_audio, true)
    }

    /// The timesteps of a call with `layout`, with those of conditions when it has them.
    pub fn for_layout(
        layout: &PackedLayout,
        sigma_video: f32,
        shift_video: f32,
        shift_audio: f32,
    ) -> Self {
        Self::build(
            sigma_video,
            shift_video,
            shift_audio,
            layout.has_conditions(),
        )
    }

    fn build(sigma_video: f32, shift_video: f32, shift_audio: f32, conditions: bool) -> Self {
        let sigma_video = sigma_video.max(1e-6);
        let video_t = 1.0 - sigma_video;
        let audio_t = 1.0 - time_shift_sigma(sigma_video, shift_video, shift_audio);
        let (video_condition_t, audio_condition_t) = (
            video_t.max(VIDEO_CONDITION_TIMESTEP),
            audio_t.max(AUDIO_CONDITION_TIMESTEP),
        );
        let mut values = vec![video_t, audio_t];
        if conditions {
            values.extend([video_condition_t, audio_condition_t]);
        }
        values.sort_by(f32::total_cmp);
        values.dedup();
        let index_of = |value: f32| {
            values
                .iter()
                .position(|&candidate| candidate == value)
                .unwrap()
        };
        StepTimesteps {
            video: index_of(video_t),
            audio: index_of(audio_t),
            conditions: conditions
                .then(|| (index_of(video_condition_t), index_of(audio_condition_t))),
            values,
        }
    }

    /// Row of the AdaLN modulation table: timestep index × modalities + modality.
    pub fn modulation_row(&self, timestep_index: usize, modality: Modality) -> usize {
        timestep_index * MODALITY_COUNT + modality as usize
    }

    /// Timestep index of a segment. Text follows the video timestep.
    pub fn index_of(&self, kind: SegmentKind) -> usize {
        let conditions = || {
            self.conditions
                .expect("the timesteps of a layout with conditions include theirs")
        };
        match kind {
            SegmentKind::Audio => self.audio,
            SegmentKind::KeyframeVideo(_) | SegmentKind::ReferenceVideo(_) => conditions().0,
            SegmentKind::KeyframeAudio(_) | SegmentKind::ReferenceAudio(_) => conditions().1,
            SegmentKind::Text | SegmentKind::Video => self.video,
        }
    }

    /// Time-embedding coordinates `[timesteps, rank]`, linearly interpolated from the AdaLN curve
    /// table.
    pub fn time_embedding(&self, table: &Tensor) -> Vec<f32> {
        let (grid, rank) = (table.shape[0], table.shape[1]);
        let mut coordinates = Vec::with_capacity(self.values.len() * rank);
        for &value in &self.values {
            let position = value.clamp(0.0, 1.0) * (grid - 1) as f32;
            let lower = (position.floor() as usize).min(grid - 2);
            let fraction = position - lower as f32;
            for column in 0..rank {
                let (low, high) = (
                    table.data[lower * rank + column],
                    table.data[(lower + 1) * rank + column],
                );
                coordinates.push(low + fraction * (high - low));
            }
        }
        coordinates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_between_schedules() {
        assert!((time_shift_sigma(0.5, 1.0, 12.0) - 12.0 / 13.0).abs() < 1e-6);
        assert!((time_shift_sigma(time_shift_sigma(0.3, 1.0, 12.0), 12.0, 1.0) - 0.3).abs() < 1e-6);
        assert_eq!(time_shift_sigma(1.0, 12.0, 3.0), 1.0);
    }

    #[test]
    fn orders_distinct_timesteps() {
        let timesteps = StepTimesteps::new(0.7, 12.0, 3.0);
        assert_eq!(timesteps.values.len(), 2);
        assert!(timesteps.values[0] < timesteps.values[1]);
        assert_eq!(timesteps.video, 0);
        assert_eq!(timesteps.modulation_row(1, Modality::Audio), 5);
        let equal = StepTimesteps::new(1.0, 3.0, 3.0);
        assert_eq!((equal.values.len(), equal.video, equal.audio), (1, 0, 0));
    }

    #[test]
    fn keeps_conditions_nearly_clean() {
        let timesteps = StepTimesteps::with_conditions(0.7, 12.0, 3.0);
        assert_eq!(timesteps.values.len(), 4);
        let (video, audio) = timesteps.conditions.unwrap();
        assert_eq!(timesteps.values[video], VIDEO_CONDITION_TIMESTEP);
        assert_eq!(timesteps.values[audio], AUDIO_CONDITION_TIMESTEP);
        assert_eq!(timesteps.index_of(SegmentKind::KeyframeVideo(1)), video);
        // Late in sampling the video rows of the target are as clean as the conditions.
        let late = StepTimesteps::with_conditions(0.0005, 12.0, 3.0);
        assert_eq!(late.conditions.unwrap().0, late.video);
    }
}
