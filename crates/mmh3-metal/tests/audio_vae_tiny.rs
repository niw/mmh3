#![cfg(target_os = "macos")]

//! Runs the Metal audio decoder on the tiny golden fixture from tools/golden/audio_vae_tiny.py.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_metal::audio_vae::MetalAudioDecoder;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/audio_vae_tiny.safetensors"
);

#[test]
fn matches_comfyui_golden_decode() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let tensor = |name: &str| {
        let mut tensor = Tensor::load(
            &file,
            file.get(name).unwrap_or_else(|| panic!("missing {name}")),
        )
        .unwrap();
        tensor.shape.remove(0);
        tensor
    };

    let decoder = MetalAudioDecoder::load(&file, "weight.").unwrap();
    let waveform = decoder.decode(&tensor("input.latent")).unwrap();
    let expected = tensor("output.waveform");
    assert_eq!(waveform.shape, expected.shape);
    let dot: f64 = waveform
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

    let cosine = dot / (norm(&waveform.data) * norm(&expected.data));
    let scale = expected
        .data
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = waveform
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    eprintln!("waveform: cosine {cosine:.7}, max error {worst:.3e}, scale {scale:.3}");
    assert!(
        cosine > 0.99999 && worst <= scale * 1e-3,
        "cosine {cosine}, max error {worst}, scale {scale}"
    );
}
