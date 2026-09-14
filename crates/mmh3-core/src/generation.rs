//! Shapes of one generation request: the frame grid the video VAE decodes and the latent sizes that
//! match it.

use crate::vae::SPATIAL_RATIO;

pub const FPS: usize = 24;
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const VIDEO_LATENT_CHANNELS: usize = 24;
pub const AUDIO_LATENT_CHANNELS: usize = 32;
pub const AUDIO_CHANNELS: usize = 2;
/// Width and height are multiples of the VAE's 16 pixels times the DiT's 2 × 2 patch.
pub const CANVAS_MULTIPLE: usize = 32;
/// Frame counts are `17n + 5`: the VAE encodes clips of 17 frames and keeps 5 latent frames of
/// each.
const FRAMES_PER_CHUNK: usize = 17;
const LATENTS_PER_CHUNK: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationShape {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub latent_frames: usize,
    pub audio_frames: usize,
}

impl GenerationShape {
    /// The shape for a canvas of `width × height` and at least `frames` frames, snapped up to the
    /// `17n + 5` grid.
    pub fn new(width: usize, height: usize, frames: usize) -> Result<Self, String> {
        if width == 0
            || height == 0
            || width % CANVAS_MULTIPLE != 0
            || height % CANVAS_MULTIPLE != 0
        {
            return Err(format!(
                "width and height must be positive multiples of {CANVAS_MULTIPLE}, got {width}×{height}"
            ));
        }
        let mut frames = frames.max(LATENTS_PER_CHUNK);
        while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
            frames += 1;
        }
        let latent_frames = (frames - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2;
        let audio_frames = (frames as f64 / FPS as f64 * AUDIO_LATENTS_PER_SECOND as f64)
            .round_ties_even() as usize;
        Ok(GenerationShape {
            width,
            height,
            frames,
            latent_frames,
            audio_frames,
        })
    }

    /// `[channels, frames, height, width]` of the video latent.
    pub fn video_latent_shape(&self) -> Vec<usize> {
        vec![
            VIDEO_LATENT_CHANNELS,
            self.latent_frames,
            self.height / SPATIAL_RATIO,
            self.width / SPATIAL_RATIO,
        ]
    }

    /// `[channels, stereo channels, frames]` of the audio latent.
    pub fn audio_latent_shape(&self) -> Vec<usize> {
        vec![AUDIO_LATENT_CHANNELS, AUDIO_CHANNELS, self.audio_frames]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_to_the_reference_grid() {
        let shape = GenerationShape::new(1344, 768, 120).unwrap();
        assert_eq!(
            (shape.frames, shape.latent_frames, shape.audio_frames),
            (124, 37, 207)
        );
        assert_eq!(shape.video_latent_shape(), vec![24, 37, 48, 84]);
        assert_eq!(shape.audio_latent_shape(), vec![32, 2, 207]);
        let small = GenerationShape::new(448, 256, 22).unwrap();
        assert_eq!(
            (small.frames, small.latent_frames, small.audio_frames),
            (22, 7, 37)
        );
        assert_eq!(GenerationShape::new(1344, 768, 1).unwrap().frames, 5);
        assert!(GenerationShape::new(1000, 768, 124).is_err());
    }
}
