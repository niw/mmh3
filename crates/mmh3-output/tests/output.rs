//! Independent decoder checks. Run explicitly with ffmpeg and ffprobe installed:
//! cargo test -p mmh3-output --test output -- --ignored
use mmh3_core::{media::Yuv420, tensor::Tensor};
use mmh3_output::{DecodedMedia, FfmpegOutput, MediaSpec, OutputBackend};
use std::path::Path;
use std::process::Command;

fn fixture(
    frames: usize,
    channels: usize,
    rate: usize,
    samples: usize,
) -> (MediaSpec, Yuv420, Tensor) {
    let spec = MediaSpec {
        width: 64,
        height: 64,
        frames,
        fps: 24,
        channels,
        sample_rate: rate,
    };
    let mut data = Vec::new();
    for frame in 0..frames {
        data.extend(vec![32 + (frame % 100) as u8; 64 * 64]);
        data.extend(vec![96; 64 * 64 / 4]);
        data.extend(vec![160; 64 * 64 / 4]);
    }
    let waveform = (0..channels)
        .flat_map(|channel| {
            (0..samples).map(move |i| {
                0.3 * (std::f32::consts::TAU * (440 * (channel + 1)) as f32 * i as f32
                    / rate as f32)
                    .sin()
            })
        })
        .collect();
    (
        spec,
        Yuv420 {
            frames,
            width: 64,
            height: 64,
            data,
        },
        Tensor::new(vec![channels, samples], waveform),
    )
}

fn probe(path: &Path) -> serde_json::Value {
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
        .arg(path)
        .output()
        .expect("install ffprobe to run these tests");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[cfg(feature = "webm")]
fn decode_audio(path: &Path) -> Vec<f32> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-map", "0:a:0", "-f", "f32le", "-c:a", "pcm_f32le", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

#[test]
#[ignore = "requires ffmpeg and ffprobe"]
#[cfg(feature = "webm")]
fn webm_decodes_with_correct_frames_color_audio_alignment_duration_and_seeking() {
    use mmh3_output::WebmOutput;
    let dir = tempfile::tempdir().unwrap();
    // Full resampling + flush, a single-frame file, and short audio requiring silence padding.
    for (frames, channels, rate, samples) in [
        (60, 2, 32_000, 80_123),
        (1, 1, 32_000, 2_000),
        (6, 1, 48_000, 8_000),
    ] {
        // Extensions are case-insensitive, as in output selection.
        let path = dir.path().join(format!(
            "{frames}.{}",
            if frames == 1 { "WEBM" } else { "webm" }
        ));
        let (spec, video, audio) = fixture(frames, channels, rate, samples);
        let backend = WebmOutput::default();
        backend.validate(&spec, &path).unwrap();
        backend
            .write(
                &DecodedMedia {
                    spec,
                    video: &video,
                    audio: &audio,
                },
                &path,
            )
            .unwrap();
        let info = probe(&path);
        let video_stream = &info["streams"][0];
        let audio_stream = &info["streams"][1];
        assert_eq!(video_stream["codec_name"], "vp9");
        assert_eq!(video_stream["nb_read_frames"], frames.to_string());
        assert_eq!(video_stream["profile"], "Profile 0");
        let video = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&path)
            .args([
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
            ])
            .output()
            .unwrap();
        assert!(
            video.status.success(),
            "{}",
            String::from_utf8_lossy(&video.stderr)
        );
        assert_eq!(video.stdout.len(), frames * 64 * 64 * 3 / 2);
        for (index, frame) in video
            .stdout
            .as_chunks::<{ 64 * 64 * 3 / 2 }>()
            .0
            .iter()
            .enumerate()
        {
            for (plane, expected) in [
                (&frame[..4096], 32.0 + index as f64),
                (&frame[4096..5120], 96.0),
                (&frame[5120..], 160.0),
            ] {
                let mean = plane.iter().map(|&x| f64::from(x)).sum::<f64>() / plane.len() as f64;
                assert!(
                    (mean - expected).abs() < 4.0,
                    "frame {index}: {mean} != {expected}"
                );
            }
        }
        assert_eq!(video_stream["width"], 64);
        assert_eq!(video_stream["height"], 64);
        assert_eq!(video_stream["color_range"], "tv");
        assert_eq!(video_stream["color_space"], "bt709");
        assert_eq!(video_stream["color_transfer"], "bt709");
        assert_eq!(video_stream["color_primaries"], "bt709");
        assert_eq!(audio_stream["codec_name"], "opus");
        assert_eq!(audio_stream["sample_rate"], "48000");
        assert_eq!(audio_stream["channels"], channels);
        let duration: f64 = info["format"]["duration"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((duration - frames as f64 / 24.0).abs() < 0.001);
        let decoded = decode_audio(&path);
        assert_eq!(
            decoded.len(),
            frames * 48_000 / 24 * channels,
            "Opus pre-skip and discard padding"
        );
        // A phase-sensitive comparison catches both channel swaps and uncompensated codec/resampler
        // delay.
        let compare_frames = (decoded.len() / channels)
            .min(samples * 48_000 / rate)
            .min(24_000);
        for channel in 0..channels {
            let mut signal = 0.0;
            let mut error = 0.0;
            for i in 400..compare_frames.saturating_sub(400) {
                let expected = 0.3
                    * (std::f32::consts::TAU * (440 * (channel + 1)) as f32 * i as f32 / 48_000.0)
                        .sin();
                signal += expected * expected;
                error += (decoded[i * channels + channel] - expected).powi(2);
            }
            assert!(
                10.0 * (signal / error).log10() > 18.0,
                "audio must remain in phase (signal={signal}, error={error})"
            );
        }
        if rate == 48_000 {
            assert!(
                decoded[10_000..].iter().all(|x| x.abs() < 0.002),
                "short audio ends in silence"
            );
        }
        if frames == 60 {
            // Opus priming has a negative start timestamp. Seek to video timeline time,
            // not an offset relative to that negative start time.
            let output = Command::new("ffmpeg")
                .args(["-v", "error", "-seek_timestamp", "1", "-ss", "2.1", "-i"])
                .arg(&path)
                .args([
                    "-map",
                    "0:v:0",
                    "-frames:v",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "yuv420p",
                    "-",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                output.stdout.len(),
                64 * 64 * 3 / 2,
                "seek through cues must return a frame"
            );
            let luma = output.stdout[..64 * 64]
                .iter()
                .map(|&x| f64::from(x))
                .sum::<f64>()
                / (64 * 64) as f64;
            assert!(
                (80.0..87.0).contains(&luma),
                "seek must reach the requested video frame: {luma}"
            );
            let output = Command::new("ffmpeg")
                .args(["-v", "error", "-seek_timestamp", "1", "-ss", "2.1", "-i"])
                .arg(&path)
                .args(["-map", "0:a:0", "-f", "f32le", "-c:a", "pcm_f32le", "-"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let seek_audio: Vec<f32> = output
                .stdout
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            assert_eq!(
                seek_audio.len(),
                19_200 * channels,
                "audio seek must preserve the remaining duration"
            );
            let reference = &decoded[100_800 * channels..];
            let error = seek_audio
                .iter()
                .zip(reference)
                .skip(1000)
                .map(|(seeked, original)| (seeked - original).powi(2))
                .sum::<f32>()
                / seek_audio.len() as f32;
            assert!(error < 0.0002, "audio seek must stay in phase: {error}");
        }
    }
}

#[test]
#[ignore = "requires ffmpeg and ffprobe"]
fn ffmpeg_custom_arguments_templates_and_failed_output_are_handled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a b;$HOME.mkv");
    let (spec, video, audio) = fixture(4, 2, 32_000, 8_000);
    let media = DecodedMedia {
        spec,
        video: &video,
        audio: &audio,
    };
    let args = ["-v", "error", "-c:v", "ffv1", "-c:a", "pcm_s16le"];
    let backend = FfmpegOutput::new(args.map(Into::into).into());
    backend.validate(&spec, &path).unwrap();
    backend.write(&media, &path).unwrap();
    assert_eq!(probe(&path)["streams"][0]["codec_name"], "ffv1");
    let original = std::fs::read(&path).unwrap();
    let broken = FfmpegOutput::new(
        ["-v", "error", "-c:v", "no-such-codec"]
            .map(Into::into)
            .into(),
    );
    assert!(
        broken.validate(&spec, &path).is_err(),
        "an unknown codec fails the trial encode"
    );
    assert!(broken.write(&media, &path).is_err());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        original,
        "failed encoding must not clobber existing output"
    );
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "temporary output is cleaned up"
    );

    let template = FfmpegOutput::new(
        [
            "-v", "error", "-i", "{video}", "-an", "-c:v", "ffv1", "{out}",
        ]
        .map(Into::into)
        .into(),
    );
    template.validate(&spec, &path).unwrap();
    template.write(&media, &path).unwrap();
    assert_eq!(probe(&path)["streams"].as_array().unwrap().len(), 1);

    // The default H.264 + AAC recipe cannot be muxed into WebM. The trial encode reports it.
    assert!(
        FfmpegOutput::new(vec![])
            .validate(&spec, &dir.path().join("out.webm"))
            .is_err()
    );
    FfmpegOutput::new(vec![])
        .validate(&spec, &dir.path().join("out.mp4"))
        .unwrap();
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        1,
        "validation writes only temporary files"
    );
}

#[test]
fn invalid_media_and_missing_ffmpeg_fail_before_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let (mut spec, video, audio) = fixture(1, 2, 32_000, 2_000);
    spec.fps = 0;
    assert!(
        DecodedMedia {
            spec,
            video: &video,
            audio: &audio
        }
        .validate()
        .is_err()
    );
    spec.fps = 24;
    let backend = FfmpegOutput {
        executable: dir.path().join("missing-ffmpeg"),
        arguments: vec![],
    };
    assert!(
        backend
            .validate(&spec, &dir.path().join("out.mp4"))
            .is_err()
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}
