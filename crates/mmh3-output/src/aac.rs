use crate::mp4::{AudioEncoder, AudioTrack, Sample};
use crate::{MediaSpec, Result, validate_audio};
use mmh3_core::tensor::Tensor;
use rusty_aac::encode::{AacEncoder, AacEncoderConfig, audio_specific_config_bytes};

const BITRATE: u32 = 192_000;

/// CPU AAC-LC encoder. Input rate is preserved, so the 32 kHz audio VAE needs no resampling.
pub struct AacOutput;
impl AudioEncoder for AacOutput {
    fn validate(&self, spec: &MediaSpec) -> Result<()> {
        spec.validate()?;
        if spec.sample_rate > 65535 || rusty_aac::sf_index_for_rate(spec.sample_rate as u32).is_none() {
            return Err("unsupported AAC sample rate".into());
        }
        Ok(())
    }
    fn encode(&self, spec: &MediaSpec, audio: &Tensor) -> Result<AudioTrack> {
        self.validate(spec)?;
        validate_audio(spec, audio)?;
        let samples = usize::try_from(spec.frames as u128 * spec.sample_rate as u128 / spec.fps as u128)?;
        if samples == 0 {
            return Err("video is shorter than one audio sample".into());
        }
        let mut encoder = AacEncoder::new(AacEncoderConfig {
            bitrate_bps: BITRATE,
            ..Default::default()
        });
        let retained = samples.min(audio.shape[1]);
        let planes: Vec<&[f32]> = audio
            .data
            .chunks_exact(audio.shape[1])
            .map(|channel| &channel[..retained])
            .collect();
        encoder.push_pcm_planar(&planes, spec.sample_rate as u32)?;
        if retained < samples {
            let silence = vec![0.0; samples - retained];
            encoder.push_pcm_planar(&vec![silence.as_slice(); spec.channels], spec.sample_rate as u32)?;
        }
        encoder.finish();
        let mut packets = Vec::new();
        loop {
            match encoder.next_packet() {
                Ok(packet) => packets.push(Sample {
                    data: packet.data,
                    dts: u64::try_from(packet.pts)?,
                    pts: packet.pts,
                    duration: packet.duration,
                    keyframe: true,
                }),
                Err(rusty_aac::Error::Eof) => break,
                Err(error) => return Err(error.into()),
            }
        }
        // This encoder's MDCT overlap adds exactly one 1024-sample priming frame.
        // MP4's edit list excludes priming and trims the final padded frame from playback.
        let priming = 1024;
        let end = samples as u64 + u64::from(priming);
        let last = packets.last_mut().ok_or("AAC produced no packets")?;
        if last.dts >= end || last.dts + u64::from(last.duration) < end {
            return Err("unexpected AAC encoder delay".into());
        }
        last.duration = u32::try_from(end - last.dts)?;
        Ok(AudioTrack {
            sample_rate: spec.sample_rate as u32,
            bitrate: BITRATE,
            channels: spec.channels as u16,
            config: audio_specific_config_bytes(spec.sample_rate as u32, spec.channels as u16),
            priming,
            samples: samples as u64,
            packets,
        })
    }
}
