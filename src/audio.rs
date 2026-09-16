//! Audio files, decoded with Symphonia and resampled to the audio VAE's sample rate.

use mmh3_core::audio::SAMPLE_RATE;
use mmh3_core::tensor::Tensor;
use rubato::{FftFixedInOut, Resampler};
use std::error::Error;
use std::fs::File;
use std::io;
use std::path::Path;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

/// Channels of a waveform the audio VAE encodes, which are its latent's stereo axis.
const CHANNELS: usize = 2;
/// Output frames of one resampler chunk.
const RESAMPLER_CHUNK: usize = 1024;

/// One channel per plane, as the file holds them.
struct Decoded {
    planes: Vec<Vec<f32>>,
    rate: usize,
}

/// Reads an audio file as a waveform `[2, samples]` in [−1, 1] at the audio VAE's sample rate.
/// Symphonia reads WAV, FLAC, MP3, AAC, ALAC, Ogg Vorbis, CAF, MP4 and Matroska files. A mono file
/// fills both channels and a file with more channels keeps the first two.
pub fn load_audio(path: &Path) -> Result<Tensor, Box<dyn Error>> {
    let decoded = decode(path)?;
    let waveform = resample(stereo(decoded.planes), decoded.rate)?;
    let samples = waveform[0].len();
    let mut data = Vec::with_capacity(CHANNELS * samples);
    for channel in &waveform {
        // Float files may carry samples outside the range the VAE was trained on.
        data.extend(channel.iter().map(|value| value.clamp(-1.0, 1.0)));
    }
    Ok(Tensor::new(vec![CHANNELS, samples], data))
}

/// Decodes the file's default audio track into one vector per channel, with its sample rate.
fn decode(path: &Path) -> Result<Decoded, Box<dyn Error>> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("{}: the file has no audio track", path.display()))?;
    let track_id = track.id;
    let parameters = track
        .codec_params
        .as_ref()
        .and_then(|parameters| parameters.audio())
        .ok_or_else(|| {
            format!(
                "{}: the audio track has no codec parameters",
                path.display()
            )
        })?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(parameters, &AudioDecoderOptions::default())
        .map_err(|error| format!("{}: {error}", path.display()))?;

    let mut planes: Vec<Vec<f32>> = Vec::new();
    let mut decoded = Vec::new();
    let mut rate = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // A truncated file ends where its samples end, as it would for a player.
            Err(SymphoniaError::IoError(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(error) => return Err(format!("{}: {error}", path.display()).into()),
        };
        if packet.track_id != track_id {
            continue;
        }
        let audio = match decoder.decode(&packet) {
            Ok(audio) => audio,
            // A damaged packet is skipped and the rest of the file still decodes.
            Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => continue,
            Err(error) => return Err(format!("{}: {error}", path.display()).into()),
        };
        if audio.frames() == 0 {
            continue;
        }
        let packet_rate = audio.spec().rate() as usize;
        audio.copy_to_vecs_planar(&mut decoded);
        if planes.is_empty() {
            planes = vec![Vec::new(); decoded.len()];
            rate = packet_rate;
        } else if packet_rate != rate || decoded.len() != planes.len() {
            return Err(format!(
                "{}: the sample rate or the channel count changes inside the file",
                path.display()
            )
            .into());
        }
        for (channel, decoded) in planes.iter_mut().zip(&decoded) {
            channel.extend_from_slice(decoded);
        }
    }
    if rate == 0 || planes.iter().all(|channel| channel.is_empty()) {
        return Err(format!("{}: the audio track has no samples", path.display()).into());
    }
    Ok(Decoded { planes, rate })
}

/// The latent's stereo axis holds two channels, so a mono file fills both and a file with more
/// channels keeps the first two.
fn stereo(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if planes.len() == 1 {
        planes.push(planes[0].clone());
    } else {
        planes.truncate(CHANNELS);
    }
    planes
}

/// Resamples to the VAE's rate with the FFT resampler, whose delay is removed from the output.
fn resample(planes: Vec<Vec<f32>>, rate: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    if rate == SAMPLE_RATE {
        return Ok(planes);
    }
    let source = planes[0].len();
    let target = usize::try_from(source as u128 * SAMPLE_RATE as u128 / rate as u128)?;
    if target == 0 {
        return Err(format!(
            "{source} samples at {rate} Hz are shorter than one sample at the VAE's rate"
        )
        .into());
    }
    let mut resampler = FftFixedInOut::<f32>::new(rate, SAMPLE_RATE, RESAMPLER_CHUNK, CHANNELS)?;
    let delay = resampler.output_delay();
    let mut output = vec![Vec::new(); CHANNELS];
    let mut offset = 0;
    while output[0].len() < delay + target {
        let needed = resampler.input_frames_next();
        let mut input = vec![vec![0.0f32; needed]; CHANNELS];
        let count = needed.min(source - offset);
        for (channel, input) in input.iter_mut().enumerate() {
            input[..count].copy_from_slice(&planes[channel][offset..offset + count]);
        }
        offset += count;
        for (channel, chunk) in output.iter_mut().zip(resampler.process(&input, None)?) {
            channel.extend(chunk);
        }
    }
    for channel in &mut output {
        channel.drain(..delay);
        channel.truncate(target);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmh3_core::media::write_wav;
    use std::f32::consts::TAU;
    use std::io::BufWriter;

    /// A sine of `frequency` Hz per channel, the second one at half the amplitude.
    fn sine(channels: usize, samples: usize, rate: usize, frequency: f32) -> Tensor {
        let mut data = Vec::with_capacity(channels * samples);
        for channel in 0..channels {
            let amplitude = 0.8 / (channel + 1) as f32;
            data.extend(
                (0..samples).map(|sample| {
                    amplitude * (TAU * frequency * sample as f32 / rate as f32).sin()
                }),
            );
        }
        Tensor::new(vec![channels, samples], data)
    }

    fn write(directory: &Path, waveform: &Tensor, rate: usize) -> std::path::PathBuf {
        let path = directory.join("audio.wav");
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        write_wav(&mut writer, waveform, rate).unwrap();
        path
    }

    #[test]
    fn reads_a_stereo_file_at_the_vae_rate() {
        let directory = tempfile::tempdir().unwrap();
        let waveform = sine(2, 4_000, SAMPLE_RATE, 1_000.0);
        let path = write(directory.path(), &waveform, SAMPLE_RATE);
        let read = load_audio(&path).unwrap();
        assert_eq!(read.shape, vec![2, 4_000]);
        // 16-bit PCM quantizes the samples, everything else is untouched.
        for (read, written) in read.data.iter().zip(&waveform.data) {
            assert!((read - written).abs() < 1e-4, "{read} against {written}");
        }
    }

    #[test]
    fn fills_both_channels_from_a_mono_file() {
        let directory = tempfile::tempdir().unwrap();
        let waveform = sine(1, 1_000, SAMPLE_RATE, 500.0);
        let path = write(directory.path(), &waveform, SAMPLE_RATE);
        let read = load_audio(&path).unwrap();
        assert_eq!(read.shape, vec![2, 1_000]);
        assert_eq!(read.data[..1_000], read.data[1_000..]);
    }

    #[test]
    fn resamples_a_48_khz_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = write(directory.path(), &sine(2, 24_000, 48_000, 1_000.0), 48_000);
        let read = load_audio(&path).unwrap();
        assert_eq!(read.shape, vec![2, 16_000]);
        // The same tone at the VAE's rate, in phase. The FFT resampler rings at both edges.
        let expected = sine(2, 16_000, SAMPLE_RATE, 1_000.0);
        for channel in 0..2 {
            for sample in 500..15_500 {
                let index = channel * 16_000 + sample;
                let (read, expected) = (read.data[index], expected.data[index]);
                assert!(
                    (read - expected).abs() < 0.02,
                    "channel {channel} sample {sample}: {read} against {expected}"
                );
            }
        }
    }
}
