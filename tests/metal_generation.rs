#![cfg(all(feature = "metal", target_os = "macos"))]
//! Exercises the real CLI with the existing small golden models and an independent MP4 decoder.
use mmh3_core::{
    safetensors::{SafeTensors, write_f32},
    tensor::Tensor,
};
use std::{path::Path, process::Command};

fn extract_model(fixture: &str, path: &Path) {
    let file = SafeTensors::open(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture),
    )
    .unwrap();
    let tensors: Vec<_> = file
        .tensors()
        .iter()
        .filter_map(|info| {
            info.name
                .strip_prefix("weight.")
                .map(|name| (name, Tensor::load(&file, info).unwrap()))
        })
        .collect();
    let values: Vec<_> = tensors
        .iter()
        .map(|(name, tensor)| (*name, tensor.shape.as_slice(), tensor.data.as_slice()))
        .collect();
    write_f32(path, &values, &[]).unwrap();
}

#[test]
fn samples_decodes_and_writes_native_mp4_without_ffmpeg() {
    generate(false, "5");
}

#[test]
fn samples_decodes_and_writes_mp4_with_ffmpeg_override() {
    generate(true, "5");
}

#[test]
fn streams_multiple_temporal_chunks_to_native_mp4() {
    generate(false, "39");
}

fn generate(ffmpeg: bool, frames: &str) {
    let dir = tempfile::tempdir().unwrap();
    let dit = dir.path().join("dit.safetensors");
    let video = dir.path().join("video.safetensors");
    let audio = dir.path().join("audio.safetensors");
    extract_model("dit_tiny.safetensors", &dit);
    extract_model("vae_tiny.safetensors", &video);
    extract_model("audio_vae_tiny.safetensors", &audio);
    let fixture = SafeTensors::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dit_tiny.safetensors"),
    )
    .unwrap();
    let context = Tensor::load(&fixture, fixture.get("input.context").unwrap()).unwrap();
    let context_path = dir.path().join("context.safetensors");
    write_f32(
        &context_path,
        &[("context", &context.shape, &context.data)],
        &[],
    )
    .unwrap();
    let destination = dir.path().join("clip.mp4");
    let mut command = Command::new(env!("CARGO_BIN_EXE_mmh3"));
    command
        .arg("generate")
        .arg("--context")
        .arg(&context_path)
        .arg("--dit")
        .arg(&dit)
        .arg("--video-vae")
        .arg(&video)
        .arg("--audio-vae")
        .arg(&audio)
        .args([
            "--width", "32", "--height", "32", "--frames", frames, "--steps", "1",
        ])
        .arg("--out")
        .arg(&destination);
    if ffmpeg {
        command.arg("--ffmpeg");
    } else {
        // The generation process cannot locate ffmpeg or any other external executable.
        command.env("PATH", dir.path().join("no-executables"));
    }

    let run = command.output().unwrap();
    assert!(
        run.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,nb_read_frames,sample_rate,channels",
            "-of",
            "default",
        ])
        .arg(&destination)
        .output()
        .expect("ffprobe is required for output verification");
    assert!(
        probe.status.success(),
        "{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let details = String::from_utf8(probe.stdout).unwrap();

    for expected in [
        "codec_type=video",
        "width=32",
        "height=32",
        &format!("nb_read_frames={frames}"),
        "codec_type=audio",
        "sample_rate=32000",
        "channels=2",
    ] {
        assert!(details.contains(expected), "missing {expected}: {details}");
    }

    let decoded = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&destination)
        .args(["-f", "null", "-"])
        .output()
        .unwrap();
    assert!(
        decoded.status.success() && decoded.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );
}
