//! Platform-independent H.264/AAC MP4 output. Native encoders supply owned samples.
//! Device handles, input buffer lifetimes and synchronization stay in their adapters.
use crate::{MediaSpec, Result, output_parent, validate_destination};
use mmh3_core::tensor::Tensor;
use std::io::Write;
use std::path::{Path, PathBuf};

mod mux;

#[derive(Clone, Debug)]
pub struct H264Config {
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

/// H.264 data uses four-byte NAL lengths. AAC data is a raw access unit (no ADTS).
/// Timestamps and durations are in the track's timebase.
pub struct Sample {
    pub data: Vec<u8>,
    pub dts: u64,
    pub pts: i64,
    pub duration: u32,
    pub keyframe: bool,
}

/// Implement for NVENC or a future VideoToolbox adapter. The associated input type
/// preserves native ownership. There is no untyped device pointer in this interface.
pub trait VideoEncoder {
    type Frames;
    fn config(&self) -> &H264Config;
    fn encode_frame(&mut self, frames: &Self::Frames, index: usize) -> Result<Sample>;
    fn finish(&mut self) -> Result<Vec<Sample>>;
}

pub struct AudioTrack {
    pub sample_rate: u32,
    /// Average and maximum bitrate recorded in the decoder configuration, in bits per second.
    pub bitrate: u32,
    pub channels: u16,
    pub config: Vec<u8>,
    pub priming: u32,
    pub samples: u64,
    pub packets: Vec<Sample>,
}

/// Future platform audio encoders (e.g. AudioToolbox) can supply the same AudioTrack.
pub trait AudioEncoder {
    fn validate(&self, spec: &MediaSpec) -> Result<()>;
    fn encode(&self, spec: &MediaSpec, audio: &Tensor) -> Result<AudioTrack>;
}

/// Prepared before model loading. An incomplete session never replaces the destination.
pub struct Mp4Session<V: VideoEncoder, A: AudioEncoder> {
    video_encoder: V,
    audio_encoder: A,
    spec: MediaSpec,
    destination: PathBuf,
    video: Vec<Sample>,
    audio: Option<AudioTrack>,
    video_written: bool,
    video_complete: bool,
}

impl<V: VideoEncoder, A: AudioEncoder> Mp4Session<V, A> {
    /// Validates the output, then opens the video encoder, which may hold native resources.
    pub fn prepare(
        spec: MediaSpec,
        destination: &Path,
        audio_encoder: A,
        video_encoder: impl FnOnce() -> Result<V>,
    ) -> Result<Self> {
        validate(&spec, destination)?;
        audio_encoder.validate(&spec)?;
        Ok(Self {
            video_encoder: video_encoder()?,
            audio_encoder,
            spec,
            destination: destination.into(),
            video: Vec::new(),
            audio: None,
            video_written: false,
            video_complete: false,
        })
    }
    pub fn write_video(&mut self, frames: &V::Frames) -> Result<()> {
        if self.video_written {
            return Err("video was already submitted".into());
        }
        self.video_written = true;
        for index in 0..self.spec.frames {
            self.video.push(self.video_encoder.encode_frame(frames, index)?);
        }
        self.video.extend(self.video_encoder.finish()?);
        self.video_complete = true;
        Ok(())
    }
    pub fn write_audio(&mut self, audio: &Tensor) -> Result<()> {
        if self.audio.is_some() {
            return Err("audio was already submitted".into());
        }
        self.audio = Some(self.audio_encoder.encode(&self.spec, audio)?);
        Ok(())
    }
    pub fn finish(self) -> Result<()> {
        let audio = self.audio.ok_or("missing MP4 audio")?;
        if !self.video_complete || self.video.len() != self.spec.frames {
            return Err("incomplete MP4 video".into());
        }
        let config = self.video_encoder.config();
        if config.sps.len() < 4 || config.sps[0] & 31 != 7 || config.pps.first().is_none_or(|header| header & 31 != 8) {
            return Err("invalid H.264 codec configuration".into());
        }
        if audio.sample_rate == 0
            || audio.sample_rate > 65535
            || audio.channels as usize != self.spec.channels
            || audio.config.is_empty()
            || audio.packets.is_empty()
            || u128::from(audio.samples)
                != self.spec.frames as u128 * u128::from(audio.sample_rate) / self.spec.fps as u128
        {
            return Err("invalid AAC track configuration or duration".into());
        }
        let mut output = tempfile::NamedTempFile::new_in(output_parent(&self.destination))?;
        mux::write(
            output.as_file_mut(),
            &self.spec,
            self.video_encoder.config(),
            &self.video,
            &audio,
        )?;
        output.flush()?;
        output.persist(self.destination)?;
        Ok(())
    }
}

fn validate(spec: &MediaSpec, destination: &Path) -> Result<()> {
    spec.validate()?;
    u16::try_from(spec.width)?;
    u16::try_from(spec.height)?;
    u32::try_from(spec.fps)?;
    u64::try_from(spec.frames as u128 * 1_000_000 / spec.fps as u128)?;
    if !destination.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("mp4")) {
        return Err("MP4 output requires --out FILE.mp4".into());
    }
    validate_destination(destination)
}

fn nal_units(bytes: &[u8]) -> Vec<&[u8]> {
    let mut start_codes = Vec::new();
    let mut position = 0;
    while position + 3 <= bytes.len() {
        let size = if bytes[position..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if bytes[position..].starts_with(&[0, 0, 1]) {
            3
        } else {
            position += 1;
            continue;
        };
        start_codes.push((position, position + size));
        position += size;
    }
    start_codes
        .iter()
        .enumerate()
        .filter_map(|(index, &(_, start))| {
            let mut end = start_codes.get(index + 1).map_or(bytes.len(), |next| next.0);
            while end > start && bytes[end - 1] == 0 {
                end -= 1;
            }
            (end > start).then_some(&bytes[start..end])
        })
        .collect()
}
impl H264Config {
    pub fn from_annex_b(bytes: &[u8]) -> Result<Self> {
        let units = nal_units(bytes);
        let sps = units
            .iter()
            .find(|nal| nal[0] & 31 == 7)
            .ok_or("NVENC did not provide H.264 SPS")?
            .to_vec();
        let pps = units
            .iter()
            .find(|nal| nal[0] & 31 == 8)
            .ok_or("NVENC did not provide H.264 PPS")?
            .to_vec();
        if sps.len() < 4 {
            return Err("invalid H.264 SPS".into());
        }
        Ok(Self { sps, pps })
    }
}

pub fn annex_b_sample(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut has_slice = false;
    for nal in nal_units(bytes) {
        let kind = nal[0] & 31;
        if matches!(kind, 7..=9) {
            continue;
        }
        has_slice |= matches!(kind, 1..=5);
        output.extend(u32::try_from(nal.len())?.to_be_bytes());
        output.extend(nal);
    }
    if !has_slice {
        return Err("H.264 packet contains no coded frame".into());
    }
    Ok(output)
}
