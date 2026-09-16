//! Geometry of the audio VAE's latents.

/// Sample rate of the waveforms the audio VAE encodes and decodes.
pub const SAMPLE_RATE: usize = 32_000;
/// Samples per latent frame: the product of the encoder's strides and of the decoder's upsampling
/// rates.
pub const SAMPLES_PER_LATENT: usize = 800;
/// Latent frames per second.
pub const LATENT_RATE: usize = SAMPLE_RATE / SAMPLES_PER_LATENT;
