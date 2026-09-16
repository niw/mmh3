//! Runs the CUDA video encoder on the tiny golden fixture from tools/golden/video_encoder_tiny.py.
#![cfg(target_os = "linux")]

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::video_encoder::{CudaVideoEncoder, Temporal};
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/video_encoder_tiny.safetensors"
);
const TILE_SIZE: usize = 32;
const TILE_OVERLAP_MIN: usize = 16;

fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap()
}

/// Cosine similarity and the largest absolute error against the largest expected magnitude.
fn compare(label: &str, actual: &Tensor, expected: &Tensor) {
    assert_eq!(actual.shape, expected.shape, "{label} shape");
    let dot: f64 = actual
        .data
        .iter()
        .zip(&expected.data)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|&value| value as f64 * value as f64)
            .sum::<f64>()
            .sqrt()
    };
    let cosine = dot / (norm(&actual.data) * norm(&expected.data));
    let scale = expected
        .data
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = actual
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    eprintln!("{label}: cosine {cosine:.7}, max error {worst:.3e}, scale {scale:.3}");
    // The reference computes in FP32, the encoder in FP16.
    assert!(
        cosine > 0.9999 && worst <= scale * 2e-2,
        "{label}: cosine {cosine}, max error {worst}, scale {scale}"
    );
}

#[test]
fn matches_comfyui_golden_clip() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let frames = tensor(&file, "input.pixels");
    let encoder =
        CudaVideoEncoder::load(&file, Temporal::Clip, TILE_SIZE, TILE_OVERLAP_MIN).unwrap();
    let latent = encoder.encode_clip(&frames).unwrap();
    assert_eq!(
        CudaVideoEncoder::latent_frames(frames.shape[0]),
        latent.mean_latent().shape[1]
    );
    compare(
        "clip",
        &latent.mean_latent(),
        &tensor(&file, "output.latent"),
    );
}

#[test]
fn matches_comfyui_golden_picture() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let frames = tensor(&file, "input.pixels");
    let pixels: usize = frames.shape[1..].iter().product();
    let picture = Tensor::new(frames.shape[1..].to_vec(), frames.data[..pixels].to_vec());
    let encoder =
        CudaVideoEncoder::load(&file, Temporal::Frame, TILE_SIZE, TILE_OVERLAP_MIN).unwrap();
    let latent = encoder.encode_picture(&picture).unwrap();
    compare(
        "picture",
        &latent.mean_latent(),
        &tensor(&file, "output.picture_latent"),
    );
}
