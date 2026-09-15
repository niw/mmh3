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

/// A clean reference the DiT sees before the target on a time span of its own and never denoises,
/// as reference to video generation takes them.
#[derive(Clone, Debug, PartialEq)]
pub enum Reference {
    /// Normalized video latent `[channels, 1, height, width]` of a picture on its own latent grid,
    /// with its noise already mixed in.
    Picture(Tensor),
    /// Normalized video latent `[channels, frames, height, width]` of a clip on its own latent grid,
    /// with its noise already mixed in, and the normalized audio latent `[channels, 2, frames]` of
    /// its soundtrack.
    Video {
        video: Tensor,
        audio: Option<Tensor>,
    },
    /// Normalized audio latent `[channels, 2, frames]` of a sound.
    Audio(Tensor),
}

impl Reference {
    pub fn video(&self) -> Option<&Tensor> {
        match self {
            Reference::Picture(video) | Reference::Video { video, .. } => Some(video),
            Reference::Audio(_) => None,
        }
    }

    pub fn audio(&self) -> Option<&Tensor> {
        match self {
            Reference::Video { audio, .. } => audio.as_ref(),
            Reference::Audio(audio) => Some(audio),
            Reference::Picture(_) => None,
        }
    }
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
    pub references: Vec<Reference>,
    pub sigma: f32,
    pub shift_video: f32,
    pub shift_audio: f32,
}
