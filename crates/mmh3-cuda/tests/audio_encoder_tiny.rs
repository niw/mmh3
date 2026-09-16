//! Runs the CUDA audio encoder on the tiny golden fixture from tools/golden/audio_encoder_tiny.py.
#![cfg(target_os = "linux")]

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::audio_encoder::CudaAudioEncoder;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/audio_encoder_tiny.safetensors"
);

#[test]
fn matches_comfyui_golden_encode() {
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
    let encoder = CudaAudioEncoder::load(&file, "weight.").unwrap();
    let latent = encoder.encode(&tensor("input.waveform")).unwrap();
    let expected = tensor("output.latent");
    assert_eq!(latent.shape, expected.shape);
    let dot: f64 = latent
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
    let cosine = dot / (norm(&latent.data) * norm(&expected.data));
    let scale = expected
        .data
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = latent
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    eprintln!("latent: cosine {cosine:.7}, max error {worst:.3e}, scale {scale:.3}");
    assert!(
        cosine > 0.99999 && worst <= scale * 1e-3,
        "cosine {cosine}, max error {worst}, scale {scale}"
    );
}
