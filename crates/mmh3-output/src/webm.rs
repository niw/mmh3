use crate::webm_mux::{self, Packet};
use crate::{DecodedMedia, MediaSpec, OutputBackend, Result, output_parent, validate_destination};
use opus::{Application, Bitrate, Channels};
use rubato::{FftFixedInOut, Resampler};
use shiguredo_libvpx::{
    CodecConfig, EncodeOptions, Encoder, EncoderConfig, ImageData, ImageFormat, Vp9Config,
};
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::Path;

const OPUS_RATE: usize = 48_000;
const OPUS_FRAME: usize = 960; // 20 ms

/// Built-in VP9 (libvpx) + Opus output. Audio is trimmed or silence-padded to the video duration.
pub struct WebmOutput {
    /// libvpx CPU-used setting, 0..=9 (higher is faster).
    pub speed: u8,
    /// Fixed VP9 quantizer, 0..=63 (lower is higher quality).
    pub quantizer: usize,
    pub audio_bitrate: i32,
}

impl Default for WebmOutput {
    fn default() -> Self {
        Self {
            speed: 4,
            quantizer: 30,
            audio_bitrate: 192_000,
        }
    }
}

impl WebmOutput {
    fn encoder(&self, spec: &MediaSpec) -> Result<Encoder> {
        if self.speed > 9 || self.quantizer > 63 {
            return Err("VP9 speed must be 0..=9 and quantizer 0..=63".into());
        }
        // The binding converts dimensions to unsigned C ints and the timebase to signed ints.
        u32::try_from(spec.width)?;
        u32::try_from(spec.height)?;
        i32::try_from(spec.fps)?;
        let mut config = EncoderConfig::new(
            spec.width,
            spec.height,
            ImageFormat::I420,
            CodecConfig::Vp9(Vp9Config {
                row_mt: true,
                ..Default::default()
            }),
        );
        config.fps_numerator = spec.fps;
        config.cpu_used = Some(self.speed as usize);
        // Fix the quantizer so quality does not depend on a resolution-specific bitrate target.
        config.min_quantizer = self.quantizer;
        config.max_quantizer = self.quantizer;
        config.cq_level = self.quantizer;
        // One frame of lookahead prevents invisible alternate-reference frames. This keeps
        // packets in presentation order, with exactly one visible frame per packet.
        // None would leave libvpx's default lookahead enabled in this binding.
        config.lag_in_frames = NonZeroUsize::new(1);
        config.frame_drop_threshold = Some(0);
        config.threads = std::thread::available_parallelism().ok();
        config.keyframe_interval = NonZeroUsize::new(spec.fps * 2);
        Ok(Encoder::new(config)?)
    }

    fn audio_encoder(&self, channels: usize) -> Result<opus::Encoder> {
        let channels = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        let mut encoder = opus::Encoder::new(OPUS_RATE as u32, channels, Application::Audio)?;
        encoder.set_bitrate(Bitrate::Bits(self.audio_bitrate))?;
        Ok(encoder)
    }
}

impl OutputBackend for WebmOutput {
    fn validate(&self, spec: &MediaSpec, destination: &Path) -> Result<()> {
        spec.validate()?;
        if !destination
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("webm"))
        {
            return Err(
                "built-in output requires --out FILE.webm. Use --ffmpeg for other formats".into(),
            );
        }
        validate_destination(destination)?;
        self.encoder(spec)?;
        self.audio_encoder(spec.channels)?;
        Ok(())
    }

    fn write(&self, media: &DecodedMedia<'_>, destination: &Path) -> Result<()> {
        media.validate()?;
        let mut encoder = self.encoder(&media.spec)?;
        let mut packets = Vec::new();
        let luma = media.spec.width * media.spec.height;
        for (index, data) in media.video.data.chunks_exact(luma * 3 / 2).enumerate() {
            encoder.encode(
                &ImageData::I420 {
                    y: &data[..luma],
                    u: &data[luma..luma * 5 / 4],
                    v: &data[luma * 5 / 4..],
                },
                &EncodeOptions {
                    force_keyframe: index.is_multiple_of(media.spec.fps * 2),
                },
            )?;
            drain_video(&mut encoder, &mut packets, media.spec.fps)?;
        }
        // libvpx may need multiple null-input calls to release all delayed packets.
        loop {
            encoder.finish()?;
            if drain_video(&mut encoder, &mut packets, media.spec.fps)? == 0 {
                break;
            }
        }
        if packets.len() != media.spec.frames {
            return Err("VP9 encoder did not output every frame".into());
        }

        let mut encoder = self.audio_encoder(media.spec.channels)?;
        let lookahead = encoder.get_lookahead()? as usize;
        let samples = resample(media)?;
        let frames = samples[0].len();
        let packet_count = (frames + lookahead).div_ceil(OPUS_FRAME);
        let padding = packet_count * OPUS_FRAME - lookahead - frames;
        let mut input = vec![0.0f32; OPUS_FRAME * media.spec.channels];
        let mut compressed = vec![0u8; 4000];
        for packet_index in 0..packet_count {
            for sample in 0..OPUS_FRAME {
                for channel in 0..media.spec.channels {
                    input[sample * media.spec.channels + channel] = samples[channel]
                        .get(packet_index * OPUS_FRAME + sample)
                        .copied()
                        .unwrap_or(0.0)
                        .clamp(-1.0, 1.0);
                }
            }
            let size = encoder.encode_float(&input, &mut compressed)?;
            packets.push(Packet {
                track: 2,
                timestamp_ns: timestamp((packet_index * OPUS_FRAME) as u64, OPUS_RATE as u64),
                keyframe: true,
                data: compressed[..size].to_vec(),
                discard_padding_ns: if packet_index + 1 == packet_count {
                    timestamp(padding as u64, OPUS_RATE as u64)
                } else {
                    0
                },
            });
        }
        // Audio precedes video at equal timestamps. CodecDelay accounts for the Opus lookahead.
        packets.sort_by_key(|packet| (packet.timestamp_ns, std::cmp::Reverse(packet.track)));
        let output = tempfile::NamedTempFile::new_in(output_parent(destination))?;
        {
            let mut writer = BufWriter::new(output.as_file());
            webm_mux::write(&mut writer, &media.spec, lookahead, &packets)?;
            writer.flush()?;
        }
        output.persist(destination)?;
        Ok(())
    }
}

/// Frame dropping and alternate-reference frames are disabled, so the packet index is its PTS.
fn drain_video(encoder: &mut Encoder, packets: &mut Vec<Packet>, fps: usize) -> Result<usize> {
    let start = packets.len();
    while let Some(frame) = encoder.next_frame() {
        let mut data = frame.data().to_vec();
        set_vp9_color(&mut data, frame.is_keyframe())?;
        packets.push(Packet {
            track: 1,
            timestamp_ns: timestamp(packets.len() as u64, fps as u64),
            keyframe: frame.is_keyframe(),
            data,
            discard_padding_ns: 0,
        });
    }
    Ok(packets.len() - start)
}

/// The binding does not expose VP9E_SET_COLOR_SPACE. Set the keyframe's uncompressed
/// color_config to BT.709 limited range, matching the input YUV and WebM Colour element.
/// This changes only metadata, not the compressed image. Inter frames inherit this config.
/// Layout: https://www.webmproject.org/vp9/ (VP9 Bitstream & Decoding Process Specification).
fn set_vp9_color(data: &mut [u8], keyframe: bool) -> Result<()> {
    // Accept only profile 0, one visible coded frame (no show_existing_frame).
    let header = *data.first().ok_or("empty VP9 packet")?;
    if header & 0xfa != 0x82 || (header & 0x04 == 0) != keyframe {
        return Err("expected a visible VP9 profile 0 frame".into());
    }
    if keyframe {
        // One header byte, three sync bytes, then 3-bit color_space + 1-bit color_range.
        if data.len() < 5 || data[1..4] != [0x49, 0x83, 0x42] {
            return Err("invalid VP9 keyframe header".into());
        }
        data[4] = (data[4] & 0x0f) | (2 << 5); // BT.709 = 2, studio range = 0.
    }
    Ok(())
}

pub(crate) fn timestamp(samples: u64, rate: u64) -> u64 {
    (u128::from(samples) * 1_000_000_000 / u128::from(rate)) as u64
}

/// FFT resampling preserves the original band limit. Remove the resampler's delay,
/// flush its tail, then trim/pad audio to match the video's exact duration at 48 kHz.
fn resample(media: &DecodedMedia<'_>) -> Result<Vec<Vec<f32>>> {
    let channels = media.spec.channels;
    let source_frames = media.audio.shape[1];
    let target_frames =
        usize::try_from(media.spec.frames as u128 * OPUS_RATE as u128 / media.spec.fps as u128)?;
    if target_frames == 0 {
        return Err("video duration is shorter than one audio sample".into());
    }
    if media.spec.sample_rate == OPUS_RATE {
        return Ok(media
            .audio
            .data
            .chunks_exact(source_frames)
            .map(|channel| {
                let mut channel = channel[..source_frames.min(target_frames)].to_vec();
                channel.resize(target_frames, 0.0);
                channel
            })
            .collect());
    }
    let mut resampler =
        FftFixedInOut::<f32>::new(media.spec.sample_rate, OPUS_RATE, 1024, channels)?;
    let delay = resampler.output_delay();
    let resampled_frames = usize::try_from(
        source_frames as u128 * OPUS_RATE as u128 / media.spec.sample_rate as u128,
    )?;
    let retained = resampled_frames.min(target_frames);
    let mut output = vec![Vec::new(); channels];
    let mut offset = 0;
    while output[0].len() < delay + retained {
        let needed = resampler.input_frames_next();
        let mut input = vec![vec![0.0f32; needed]; channels];
        let count = needed.min(source_frames.saturating_sub(offset));
        for (channel, input) in input.iter_mut().enumerate() {
            let start = channel * source_frames + offset.min(source_frames);
            input[..count].copy_from_slice(&media.audio.data[start..start + count]);
        }
        offset += count;
        for (channel, chunk) in output.iter_mut().zip(resampler.process(&input, None)?) {
            channel.extend(chunk);
        }
    }
    for channel in &mut output {
        channel.drain(..delay);
        channel.truncate(retained);
        channel.resize(target_frames, 0.0);
    }
    Ok(output)
}
