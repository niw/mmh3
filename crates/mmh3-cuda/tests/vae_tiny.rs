//! Runs the CUDA video decoder on the tiny golden fixture from tools/golden/vae_tiny.py.
#![cfg(target_os = "linux")]

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::vae::CudaVideoDecoder;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/vae_tiny.safetensors"
);

fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    let mut tensor = Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap();
    tensor.shape.remove(0);
    tensor
}

fn metadata_number(file: &SafeTensors, key: &str) -> usize {
    file.metadata()
        .iter()
        .find(|(name, _)| name == key)
        .unwrap()
        .1
        .parse()
        .unwrap()
}

fn assert_close(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(actual.shape, expected.shape, "{name}: shape");
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
    let worst = actual
        .data
        .iter()
        .zip(&expected.data)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    eprintln!("{name}: cosine {cosine:.7}, max error {worst:.3e}");
    assert!(
        cosine > 0.9999 && worst <= tolerance,
        "{name}: cosine {cosine}, max error {worst}"
    );
}

#[test]
fn matches_comfyui_golden_decode() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let decoder = CudaVideoDecoder::load(
        &file,
        "weight.",
        metadata_number(&file, "tile_size"),
        metadata_number(&file, "tile_overlap_min"),
    )
    .unwrap();
    let decoding = decoder
        .decode(&tensor(&file, "input.latent"), true)
        .unwrap();
    let first_tile = tensor(&file, "intermediate.first_tile");
    let scale = first_tile
        .data
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    assert_close(
        "first tile",
        decoding.first_tile.as_ref().unwrap(),
        &first_tile,
        scale * 0.02,
    );
    assert_close(
        "pixels",
        &decoding.pixels,
        &tensor(&file, "output.pixels"),
        0.02,
    );
}
