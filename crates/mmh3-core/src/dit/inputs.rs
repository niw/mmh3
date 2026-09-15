use crate::dit::timestep::Modality;
use crate::tensor::Tensor;

/// A clean condition anchored at a frame of the target, which the DiT sees next to it and never
/// denoises, such as the first or the last frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    /// Pixel frame of the target the condition's first frame aligns with.
    pub frame_index: usize,
    /// Normalized video latent `[channels, frames, height, width]` on the target's latent grid,
    /// with its noise already mixed in.
    pub video: Option<Tensor>,
    /// Normalized audio latent `[channels, 2, frames]`.
    pub audio: Option<Tensor>,
}

/// Inputs of one DiT call for text-to-video with audio.
pub struct DitInputs {
    /// Normalized video latent `[channels, frames, height, width]`.
    pub video: Tensor,
    /// Normalized audio latent `[channels, 2, frames]`.
    pub audio: Tensor,
    /// Text encoder hidden states `[tokens, text_dim]`.
    pub context: Tensor,
    /// AdaLN branch of each context token, video for the vision blocks of pictures in the prompt.
    /// Empty when every token is text.
    pub context_modalities: Vec<Modality>,
    pub keyframes: Vec<Keyframe>,
    pub sigma: f32,
    pub shift_video: f32,
    pub shift_audio: f32,
}
