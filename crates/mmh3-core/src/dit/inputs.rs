use crate::tensor::Tensor;

/// Inputs of one DiT call for text-to-video with audio.
pub struct DitInputs {
    /// Normalized video latent `[channels, frames, height, width]`.
    pub video: Tensor,
    /// Normalized audio latent `[channels, 2, frames]`.
    pub audio: Tensor,
    /// Text encoder hidden states `[tokens, text_dim]`.
    pub context: Tensor,
    pub sigma: f32,
    pub shift_video: f32,
    pub shift_audio: f32,
}
