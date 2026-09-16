//! Select output by explicit CLI override, then container extension. Native frame
//! adapters live at the application boundary. The MP4 writer is platform-independent.
use mmh3_core::{media::Yuv420, tensor::Tensor};
use mmh3_output::{DecodedMedia, FfmpegOutput, MediaSpec, OutputBackend, Result};
use std::path::{Path, PathBuf};

pub struct HostSession {
    backend: Box<dyn OutputBackend>,
    spec: MediaSpec,
    destination: PathBuf,
    video: Option<Yuv420>,
    audio: Option<Tensor>,
}

pub enum PreparedOutput {
    Host(HostSession),
    #[cfg(feature = "metal")]
    Apple(
        mmh3_output::mp4::Mp4Session<
            mmh3_output_videotoolbox::MetalVideoToolboxEncoder,
            mmh3_output::AacOutput,
        >,
    ),
    #[cfg(feature = "mp4")]
    Nvidia(mmh3_output::mp4::Mp4Session<mmh3_output_nvenc::NvencEncoder, mmh3_output::AacOutput>),
}

pub fn prepare(
    ffmpeg_arguments: Option<&[String]>,
    spec: MediaSpec,
    destination: &Path,
) -> Result<PreparedOutput> {
    spec.validate()?;
    let backend: Box<dyn OutputBackend> = if let Some(arguments) = ffmpeg_arguments {
        Box::new(FfmpegOutput::new(
            arguments.iter().map(Into::into).collect(),
        ))
    } else {
        let extension = destination
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match extension.as_str() {
            "mp4" => {
                #[cfg(feature = "metal")]
                {
                    return Ok(PreparedOutput::Apple(
                        mmh3_output::mp4::Mp4Session::prepare(
                            spec,
                            destination,
                            mmh3_output::AacOutput,
                            || mmh3_output_videotoolbox::MetalVideoToolboxEncoder::new(spec),
                        )?,
                    ));
                }
                #[cfg(feature = "mp4")]
                {
                    return Ok(PreparedOutput::Nvidia(
                        mmh3_output::mp4::Mp4Session::prepare(
                            spec,
                            destination,
                            mmh3_output::AacOutput,
                            || mmh3_output_nvenc::NvencEncoder::new(spec),
                        )?,
                    ));
                }
                #[cfg(not(any(feature = "mp4", feature = "metal")))]
                return Err(
                    "native MP4 needs --features mp4. Use --out FILE.webm or --ffmpeg".into(),
                );
            }
            "webm" => {
                #[cfg(feature = "webm")]
                {
                    Box::new(mmh3_output::WebmOutput::default())
                }
                #[cfg(not(feature = "webm"))]
                return Err(
                    "this build has no WebM backend. Rebuild with --features webm or pass --ffmpeg"
                        .into(),
                );
            }
            _ => {
                return Err(
                    "choose --out FILE.mp4 or FILE.webm, or pass --ffmpeg for other formats".into(),
                );
            }
        }
    };
    backend.validate(&spec, destination)?;
    Ok(PreparedOutput::Host(HostSession {
        backend,
        spec,
        destination: destination.into(),
        video: None,
        audio: None,
    }))
}
impl PreparedOutput {
    /// CPU-readable video, including frames decoded by the Metal backend.
    pub fn write_host_video(&mut self, frames: Yuv420) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(_) => Err("VideoToolbox output requires Metal frames".into()),
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(frames);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(_) => {
                Err("NVENC output requires CUDA frames. Use --ffmpeg for host frames".into())
            }
        }
    }
    #[cfg(feature = "metal")]
    pub fn streams_metal_video(&self) -> bool {
        matches!(self, Self::Apple(_))
    }
    #[cfg(feature = "metal")]
    pub fn write_metal_video_chunk(
        &mut self,
        frames: &mmh3_metal::vae::MetalVideoFrames,
    ) -> Result<()> {
        match self {
            Self::Apple(session) => session.write_video_chunk(frames, frames.frame_range()),
            Self::Host(_) => Err("host output requires the complete video".into()),
        }
    }
    #[cfg(feature = "metal")]
    pub fn write_metal_video(&mut self, frames: &mmh3_metal::vae::MetalVideoFrames) -> Result<()> {
        match self {
            Self::Apple(session) => session.write_video(frames),
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(Yuv420::from_pixels(&frames.to_pixels()?)?);
                Ok(())
            }
        }
    }
    #[cfg(feature = "cuda")]
    pub fn write_cuda_video(&mut self, frames: &mmh3_cuda::vae::CudaVideoFrames) -> Result<()> {
        match self {
            Self::Host(session) => {
                if session.video.is_some() {
                    return Err("video was already submitted".into());
                }
                session.video = Some(frames.to_yuv420()?);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.write_video(frames),
        }
    }
    pub fn write_audio(&mut self, audio: Tensor) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(session) => session.write_audio(&audio),
            Self::Host(session) => {
                if session.audio.is_some() {
                    return Err("audio was already submitted".into());
                }
                session.audio = Some(audio);
                Ok(())
            }
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.write_audio(&audio),
        }
    }
    pub fn finish(self) -> Result<()> {
        match self {
            #[cfg(feature = "metal")]
            Self::Apple(session) => session.finish(),
            Self::Host(session) => session.backend.write(
                &DecodedMedia {
                    spec: session.spec,
                    video: session.video.as_ref().ok_or("missing output video")?,
                    audio: session.audio.as_ref().ok_or("missing output audio")?,
                },
                &session.destination,
            ),
            #[cfg(feature = "mp4")]
            Self::Nvidia(session) => session.finish(),
        }
    }
}
