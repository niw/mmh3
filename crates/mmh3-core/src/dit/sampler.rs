//! Sampling schedules and the Euler update of the rectified flow.

use crate::dit::timestep::time_shift_sigma;

/// Video and audio sigma schedules that share one unshifted grid, ending at zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Schedule {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

impl Schedule {
    /// `steps` model calls on the unshifted grid linspace(1, 0, steps + 1), shifted separately per stream.
    /// This is the official scheduler's grid and the q grid of the Turbo LoRAs.
    pub fn uniform(steps: usize, shift_video: f32, shift_audio: f32) -> Self {
        let base: Vec<f32> = (0..=steps).map(|step| 1.0 - step as f32 / steps as f32).collect();
        Schedule {
            video: base.iter().map(|&value| time_shift_sigma(value, 1.0, shift_video)).collect(),
            audio: base.iter().map(|&value| time_shift_sigma(value, 1.0, shift_audio)).collect(),
        }
    }

    pub fn steps(&self) -> usize {
        self.video.len() - 1
    }
}

/// One Euler step: x ← x + (sigma_next − sigma) · v, where v is the flow velocity (x0 = x − sigma · v).
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
