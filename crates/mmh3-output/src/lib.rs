//! Portable output backends, encoder interfaces and muxers, independent of CUDA and model loading.

mod ffmpeg;
#[cfg(feature = "webm")]
mod webm;
#[cfg(feature = "webm")]
mod webm_mux;

pub use ffmpeg::FfmpegOutput;
#[cfg(feature = "webm")]
pub use webm::WebmOutput;

use mmh3_core::{media::Yuv420, tensor::Tensor};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Dimensions and timing shared by output paths. Native adapters determine input
/// storage and pixel format. Audio has one or two channels.
#[derive(Clone, Copy, Debug)]
pub struct MediaSpec {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub sample_rate: usize,
    pub channels: usize,
}

impl MediaSpec {
    pub fn validate(&self) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(2)
            || !self.height.is_multiple_of(2)
        {
            return Err("video dimensions must be positive and even".into());
        }
        if self.frames == 0
            || self.fps == 0
            || self.sample_rate == 0
            || !(1..=2).contains(&self.channels)
        {
            return Err(
                "media needs frames, a positive frame/sample rate, and mono or stereo audio".into(),
            );
        }
        self.width
            .checked_mul(self.height)
            .and_then(|pixels| pixels.checked_mul(3))
            .and_then(|bytes| bytes.checked_mul(self.frames))
            .ok_or("video dimensions overflow")?;
        u32::try_from(self.sample_rate)?;
        Ok(())
    }
}

/// Host input for WebM and ffmpeg: 8-bit BT.709 limited YUV420 (centered chroma)
/// and planar float PCM. Native MP4 adapters use their own typed device frames.
pub struct DecodedMedia<'a> {
    pub spec: MediaSpec,
    pub video: &'a Yuv420,
    pub audio: &'a Tensor,
}

impl DecodedMedia<'_> {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        let spec = self.spec;
        if (self.video.width, self.video.height, self.video.frames)
            != (spec.width, spec.height, spec.frames)
            || self.video.data.len() != spec.frames * Yuv420::frame_bytes(spec.height, spec.width)
        {
            return Err("decoded video does not match the media specification".into());
        }
        validate_audio(&spec, self.audio)
    }
}

pub(crate) fn validate_audio(spec: &MediaSpec, audio: &Tensor) -> Result<()> {
    if audio.shape.len() != 2
        || audio.shape[0] != spec.channels
        || audio.shape[1] == 0
        || audio.shape[1].checked_mul(spec.channels) != Some(audio.data.len())
        || audio.data.iter().any(|sample| !sample.is_finite())
    {
        return Err("audio must contain finite planar PCM [channels, samples]".into());
    }
    Ok(())
}

/// Add a backend by implementing this trait and selecting it at the CLI boundary.
pub trait OutputBackend {
    /// Called before expensive generation to catch unsupported settings and missing tools.
    fn validate(&self, spec: &MediaSpec, destination: &Path) -> Result<()>;
    /// Writes a complete file. Implementations preserve an existing destination on failure.
    fn write(&self, media: &DecodedMedia<'_>, destination: &Path) -> Result<()>;
}

pub(crate) fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

pub(crate) fn validate_destination(path: &Path) -> Result<()> {
    if path.file_name().is_none() || path.is_dir() {
        return Err("--out must name an output file".into());
    }
    // Check writability without truncating an existing result.
    tempfile::NamedTempFile::new_in(output_parent(path))?;
    Ok(())
}

#[cfg(feature = "mp4")]
mod aac;
#[cfg(feature = "mp4")]
pub mod mp4;
#[cfg(feature = "mp4")]
pub use aac::AacOutput;
