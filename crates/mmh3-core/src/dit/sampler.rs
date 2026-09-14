//! Sampling schedules and the Euler update of the rectified flow.

use crate::dit::timestep::time_shift_sigma;

/// Video and audio sigma schedules that share one unshifted grid, ending at zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

impl Schedule {
    /// `steps` model calls on the unshifted grid linspace(1, 0, steps + 1), shifted separately per
    /// stream. This is the official scheduler's grid and the q grid of the Turbo LoRAs.
    pub fn uniform(steps: usize, shift_video: f32, shift_audio: f32) -> Self {
        let indices: Vec<usize> = (0..=steps).collect();
        Self::retained(steps + 1, &indices, shift_video, shift_audio)
    }

    /// The states at `indices` of the shifted `points`-point grid linspace(1, 0, points). The first
    /// index must be 0 and the last `points − 1`, so that sampling starts from noise and ends at
    /// zero.
    pub fn retained(points: usize, indices: &[usize], shift_video: f32, shift_audio: f32) -> Self {
        assert!(
            indices.len() >= 2
                && indices[0] == 0
                && indices[indices.len() - 1] == points - 1
                && indices.windows(2).all(|pair| pair[0] < pair[1]),
            "the retained states must rise from 0 to {}",
            points - 1
        );
        let base: Vec<f32> = indices
            .iter()
            .map(|&index| 1.0 - index as f32 / (points - 1) as f32)
            .collect();
        Schedule {
            video: base
                .iter()
                .map(|&value| time_shift_sigma(value, 1.0, shift_video))
                .collect(),
            audio: base
                .iter()
                .map(|&value| time_shift_sigma(value, 1.0, shift_audio))
                .collect(),
        }
    }

    /// The three steps TaoMate-H3's LoRA was distilled for: states 0, 16, 33 and 49 of the
    /// 50-point grid.
    pub fn taomate(shift_video: f32, shift_audio: f32) -> Self {
        Self::retained(50, &[0, 16, 33, 49], shift_video, shift_audio)
    }

    pub fn steps(&self) -> usize {
        self.video.len() - 1
    }
}

/// One Euler step: x ← x + (sigma_next − sigma) · v, where v is the flow velocity
/// (x0 = x − sigma · v).
pub fn euler_step(latent: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    let delta = sigma_next - sigma;
    for (value, &slope) in latent.iter_mut().zip(velocity) {
        *value += delta * slope;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_turbo_grid() {
        let schedule = Schedule::uniform(4, 12.0, 3.0);
        let expected_video = [1.0, 0.9730, 0.9231, 0.8000, 0.0];
        let expected_audio = [1.0, 0.9000, 0.7500, 0.5000, 0.0];
        for (actual, expected) in schedule.video.iter().zip(expected_video) {
            assert!((actual - expected).abs() < 1e-4, "{:?}", schedule.video);
        }
        for (actual, expected) in schedule.audio.iter().zip(expected_audio) {
            assert!((actual - expected).abs() < 1e-4, "{:?}", schedule.audio);
        }
    }

    #[test]
    fn matches_the_taomate_states() {
        let schedule = Schedule::taomate(12.0, 3.0);
        let expected_video = [1.0, 0.961165, 0.853333, 0.0];
        let expected_audio = [1.0, 0.860870, 0.592593, 0.0];
        for (actual, expected) in schedule.video.iter().zip(expected_video) {
            assert!((actual - expected).abs() < 1e-5, "{:?}", schedule.video);
        }
        for (actual, expected) in schedule.audio.iter().zip(expected_audio) {
            assert!((actual - expected).abs() < 1e-5, "{:?}", schedule.audio);
        }
    }

    #[test]
    fn lands_on_the_clean_latent() {
        let (clean, noise) = (2.0f32, -1.0f32);
        let mut latent = [noise];
        let schedule = Schedule::uniform(3, 12.0, 3.0);
        for step in 0..schedule.steps() {
            let (sigma, next) = (schedule.video[step], schedule.video[step + 1]);
            euler_step(&mut latent, &[noise - clean], sigma, next);
        }
        assert!((latent[0] - clean).abs() < 1e-6);
    }
}
