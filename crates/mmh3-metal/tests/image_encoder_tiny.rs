//! Runs the Metal video encoder with a picture VAE on the tiny golden fixture from
//! tools/golden/image_encoder_tiny.py.
#![cfg(target_os = "macos")]

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};
use mmh3_metal::Device;
use mmh3_metal::video_encoder::{MetalVideoEncoder, Temporal};
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/image_encoder_tiny.safetensors"
);

fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap()
}

/// Whether the GPU has the matrix units that the video encoder needs. A GPU without them, such as
/// the paravirtual GPU of a virtual machine, skips the test and says so.
fn has_matrix_units() -> bool {
    let supported = Device::shared().unwrap().supports_tensor_ops();
    if !supported {
        eprintln!("skipped: the video encoder needs the matrix units, which this GPU lacks");
    }
    supported
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
fn matches_diffusers_golden_picture() {
    if !has_matrix_units() {
        return;
    }
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    // Without latent statistics, as in Comfy-Org's file, the encoder takes the release's.
    assert!(file.get("latents_mean").is_none());
    let encoder = MetalVideoEncoder::load(
        &file,
        Temporal::Frame,
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    )
    .unwrap();
    // 2 × 3 tiles of 256 pixels whose overlaps take the slack.
    let posterior = encoder
        .encode_picture(&tensor(&file, "input.pixels"))
        .unwrap();
    compare("mean", &posterior.mean, &tensor(&file, "output.mean"));
    compare(
        "deviation",
        &posterior.deviation,
        &tensor(&file, "output.deviation"),
    );
    compare(
        "latent",
        &posterior.mean_latent(),
        &tensor(&file, "output.latent"),
    );
    compare(
        "sampled latent",
        &posterior.sampled_latent(&tensor(&file, "input.noise").data),
        &tensor(&file, "output.sampled_latent"),
    );
}
