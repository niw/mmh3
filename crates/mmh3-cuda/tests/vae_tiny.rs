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

/// A chunk that was handed out and never came back ends the decode rather than being decoded here
/// after all. What it asked for is the point: a decode that went on to the next chunk would have
/// asked for that one too.
#[test]
fn a_chunk_that_never_came_back_ends_the_decode() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let decoder = CudaVideoDecoder::to_assemble(
        &file,
        "weight.",
        metadata_number(&file, "tile_size"),
        metadata_number(&file, "tile_overlap_min"),
    )
    .unwrap();
    // Enough frames for more than one chunk, so that stopping at the first is visible.
    let shape = tensor(&file, "input.latent").shape;
    let (channels, frames) = (shape[0], 12);
    let count = channels * frames * shape[2] * shape[3];
    let latent = Tensor::new(
        vec![channels, frames, shape[2], shape[3]],
        (0..count)
            .map(|index| ((index * 2_654_435_761) % 1013) as f32 / 1013.0 - 0.5)
            .collect(),
    );
    assert!(decoder.plan(&latent).unwrap().chunks > 1);

    let mut asked = Vec::new();
    let decoded = decoder.decode_device_with(&latent, &mut |chunk| {
        asked.push(chunk);
        Err(format!("chunk {chunk} never came back"))
    });
    let Err(error) = decoded else {
        panic!("the decode answered with frames for a chunk it never had");
    };
    assert!(
        error.to_string().contains("never came back"),
        "the decode ends with the reason the chunk gave, not {error}"
    );
    assert_eq!(asked, vec![0]);
}
