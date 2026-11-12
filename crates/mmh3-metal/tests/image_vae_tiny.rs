#![cfg(target_os = "macos")]

//! Runs the Metal video decoder with a picture VAE on the tiny golden fixture from
//! tools/golden/image_vae_tiny.py.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{
    DEFAULT_TILE_SIZE, PICTURE_LATENTS_MEAN, PICTURE_LATENTS_STD, PICTURE_TILE_OVERLAP_MIN,
};
use mmh3_metal::vae::MetalVideoDecoder;
use mmh3_metal::{AttentionPrecision, Device, LinearPrecision};
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/image_vae_tiny.safetensors"
);

/// A tensor of the fixture without its batch axis, and with a frame axis for a latent or pixels.
fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    let mut tensor = Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap();
    if tensor.shape.len() == 4 {
        tensor.shape[0] = tensor.shape[1];
        tensor.shape[1] = 1;
    }
    tensor
}

fn worst_error(actual: &Tensor, expected: &Tensor) -> f32 {
    assert_eq!(actual.shape, expected.shape);
    actual
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        })
}

#[test]
fn carries_the_released_latent_statistics() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    assert_eq!(
        tensor(&file, "reference.latents_mean").data,
        PICTURE_LATENTS_MEAN
    );
    assert_eq!(
        tensor(&file, "reference.latents_std").data,
        PICTURE_LATENTS_STD
    );
    // The decoder has to fall back to them, as it does for Comfy-Org's file.
    assert!(file.get("weight.latents_mean").is_none());
}

#[test]
fn matches_diffusers_golden_decode() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut decoder = MetalVideoDecoder::load(
        &file,
        "weight.",
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    )
    .unwrap();
    // FP32 throughout differs from the reference by summation order alone. FP16 attention and
    // products round their inputs to 11 bits, which the ImageNet scale of about a quarter shrinks
    // in the pixels. The products take FP16 at the INT8 precision too, since the weights are FP16.
    let mut precisions = vec![(LinearPrecision::Fp32, AttentionPrecision::Fp32, 1e-5)];
    if Device::shared().unwrap().supports_tensor_ops() {
        precisions.push((LinearPrecision::Fp32, AttentionPrecision::Fp16, 5e-4));
        precisions.push((LinearPrecision::Fp16, AttentionPrecision::Fp16, 5e-4));
        precisions.push((LinearPrecision::Int8, AttentionPrecision::Fp16, 5e-4));
    }
    for (linear, attention, tolerance) in precisions {
        decoder.set_precision(linear, attention).unwrap();
        // 2 × 3 tiles whose overlaps take the slack, and 1 × 2 tiles of 192 × 256 pixels.
        for suffix in ["", "_small"] {
            let latent = tensor(&file, &format!("input.latent{suffix}"));
            let expected = tensor(&file, &format!("output.pixels{suffix}"));
            let pixels = decoder.decode(&latent, false).unwrap().pixels;
            let worst = worst_error(&pixels, &expected);
            eprintln!(
                "{linear:?}/{attention:?} latent{suffix} {:?}: max error {worst:.3e}",
                latent.shape
            );
            assert!(worst <= tolerance, "max error {worst}");
        }
    }

    let mut frames = tensor(&file, "input.latent_small");
    frames.shape[1] = 2;
    frames.data.extend_from_within(..);
    assert!(decoder.decode(&frames, false).is_err());
}
