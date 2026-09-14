//! Requires NVIDIA NVENC, ffmpeg and ffprobe. No model weights are needed.
use mmh3_core::{media::Yuv420, tensor::Tensor};
use mmh3_cuda::{DeviceBuffer, vae::CudaVideoFrames};
use mmh3_output::{AacOutput, MediaSpec, mp4::Mp4Session};
use mmh3_output_nvenc::NvencEncoder;
use std::path::Path;
use std::process::Command;

fn decode(path: &Path, kind: &str, seek: bool) -> Vec<u8> {
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error"]);
    if seek {
        command.args(["-ss", "2.1"]);
    }
    command.arg("-i").arg(path);
    if kind == "video" {
        command.args([
            "-map", "0:v:0", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ]);
    } else {
        command.args(["-map", "0:a:0", "-f", "f32le", "-"]);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
#[ignore = "requires NVENC GPU, ffmpeg and ffprobe"]
fn nvenc_mp4_preserves_frames_color_audio_timing_and_seeking() {
    let dir = tempfile::tempdir().unwrap();
    for (width, height, frames, channels, rate, source_samples) in [
        (322, 242, 60, 2, 32_000, 80_123),
        (320, 240, 1, 1, 48_000, 2_500),
        (1344, 768, 6, 2, 32_000, 4_000),
    ] {
        let spec = MediaSpec {
            width,
            height,
            frames,
            fps: 24,
            sample_rate: rate,
            channels,
        };
        let path = dir.path().join(format!("{frames}.mp4"));
        // Deliberately non-grey frames and changing luminance expose plane swaps, padding and
        // reordering.
        let data: Vec<f32> = (0..3)
            .flat_map(|channel| {
                (0..frames).flat_map(move |frame| {
                    let value = 0.15 + frame as f32 * 0.007 + channel as f32 * 0.12;
                    std::iter::repeat_n(value, width * height)
                })
            })
            .collect();
        let reference =
            Yuv420::from_pixels(&Tensor::new(vec![3, frames, height, width], data.clone()))
                .unwrap();
        let video = CudaVideoFrames::from_rgb(
            DeviceBuffer::from_f32(&data).unwrap(),
            frames,
            height,
            width,
        )
        .unwrap();
        let waveform = (0..channels)
            .flat_map(|channel| {
                (0..source_samples).map(move |i| {
                    0.3 * (std::f32::consts::TAU * (440 * (channel + 1)) as f32 * i as f32
                        / rate as f32)
                        .sin()
                })
            })
            .collect();
        let audio = Tensor::new(vec![channels, source_samples], waveform);
        let mut session =
            Mp4Session::prepare(spec, &path, AacOutput, || NvencEncoder::new(spec)).unwrap();
        session.write_video(&video).unwrap();
        session.write_audio(&audio).unwrap();
        session.finish().unwrap();
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
            ])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let info: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let video_stream = &info["streams"][0];
        let audio_stream = &info["streams"][1];
        assert_eq!(video_stream["codec_name"], "h264");
        assert_eq!(video_stream["nb_read_frames"], frames.to_string());
        assert_eq!(video_stream["width"], width);
        assert_eq!(video_stream["height"], height);
        assert_eq!(video_stream["pix_fmt"], "yuv420p");
        for key in ["color_space", "color_primaries", "color_transfer"] {
            assert_eq!(video_stream[key], "bt709");
        }
        assert_eq!(video_stream["color_range"], "tv");
        assert_eq!(audio_stream["codec_name"], "aac");
        assert_eq!(audio_stream["profile"], "LC");
        assert_eq!(audio_stream["sample_rate"], rate.to_string());
        assert_eq!(audio_stream["channels"], channels);
        let duration: f64 = info["format"]["duration"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((duration - frames as f64 / 24.0).abs() < 0.001);
        let audio_duration: f64 = audio_stream["duration"].as_str().unwrap().parse().unwrap();
        assert!((audio_duration - (frames * rate / 24) as f64 / rate as f64).abs() < 0.0001);
        assert_eq!(
            audio_stream["start_time"], "0.000000",
            "AAC priming must be excluded"
        );
        let decoded = decode(&path, "video", false);
        assert_eq!(decoded.len(), reference.data.len());
        let plane = width * height;
        for (frame, expected) in decoded
            .chunks_exact(plane * 3 / 2)
            .zip(reference.data.chunks_exact(plane * 3 / 2))
        {
            for range in [0..plane, plane..plane * 5 / 4, plane * 5 / 4..plane * 3 / 2] {
                let mae = frame[range.clone()]
                    .iter()
                    .zip(&expected[range.clone()])
                    .map(|(&actual, &wanted)| (actual as f64 - wanted as f64).abs())
                    .sum::<f64>()
                    / range.len() as f64;
                assert!(mae < 3.0, "video plane error: {mae}");
            }
        }
        let pcm = decode(&path, "audio", false);
        let decoded: Vec<f32> = pcm
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        let wanted = frames * rate / 24;
        eprintln!(
            "{frames} frames: AAC decoded {} samples, wanted {wanted}",
            decoded.len() / channels
        );
        // Some decoders return the final AAC block's padding. The MP4 edit list and duration limit
        // its presentation.
        assert!(decoded.len() / channels >= wanted && decoded.len() / channels < wanted + 1024);
        let end = wanted.min(source_samples).min(rate / 2);
        for channel in 0..channels {
            let mut signal = 0.0;
            let mut error = 0.0;
            for i in 400..end.saturating_sub(400) {
                let expected = audio.data[channel * source_samples + i];
                signal += expected * expected;
                error += (expected - decoded[i * channels + channel]).powi(2);
            }
            let snr = 10.0f32 * (signal / error).log10();
            assert!(snr > 18.0, "AAC phase-sensitive SNR: {snr}");
        }
        if frames == 6 {
            assert!(
                decoded[6500 * channels..wanted * channels]
                    .iter()
                    .all(|sample| sample.abs() < 0.01)
            );
        }
        if frames == 60 {
            let seek = decode(&path, "video", true);
            let start = 51 * plane * 3 / 2;
            let expected = &reference.data[start..start + plane];
            let mae = seek[..plane]
                .iter()
                .zip(expected)
                .map(|(&actual, &wanted)| (actual as f64 - wanted as f64).abs())
                .sum::<f64>()
                / plane as f64;
            assert!(mae < 3.0, "seek must reach requested frame: {mae}");
            let seek = decode(&path, "audio", true);
            let seek: Vec<f32> = seek
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            let offset = rate * 21 / 10 * channels;
            let mse = seek
                .iter()
                .zip(&decoded[offset..])
                .skip(1024 * channels)
                .map(|(seeked, original)| (seeked - original).powi(2))
                .sum::<f32>()
                / seek.len() as f32;
            assert!(mse < 0.0002, "AAC seek must preserve phase: {mse}");
        }
        let bytes = std::fs::read(&path).unwrap();
        let moov = bytes.windows(4).position(|kind| kind == b"moov").unwrap();
        let mdat = bytes.windows(4).position(|kind| kind == b"mdat").unwrap();
        assert!(moov < mdat, "fast start");
    }
}

#[test]
#[ignore = "requires NVENC GPU"]
fn incomplete_session_preserves_destination_and_releases_encoder() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.mp4");
    std::fs::write(&path, b"original").unwrap();
    let spec = MediaSpec {
        width: 320,
        height: 240,
        frames: 1,
        fps: 24,
        sample_rate: 32_000,
        channels: 2,
    };
    for _ in 0..3 {
        let session =
            Mp4Session::prepare(spec, &path, AacOutput, || NvencEncoder::new(spec)).unwrap();
        assert!(session.finish().is_err());
    }
    assert_eq!(std::fs::read(&path).unwrap(), b"original");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}
