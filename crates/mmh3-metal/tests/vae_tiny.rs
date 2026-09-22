#![cfg(target_os = "macos")]

//! Runs the Metal video decoder on the tiny golden fixture from tools/golden/vae_tiny.py.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_metal::vae::MetalVideoDecoder;
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
    let decoder = MetalVideoDecoder::load(
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
    let latent = tensor(&file, "input.latent");
    let [_, frames, height, width] = decoding.pixels.shape.as_slice() else {
        panic!("shape")
    };

    let plane = height * width;
    let mut pixels = vec![0.0; decoding.pixels.data.len()];
    let mut next = 0;
    decoder
        .decode_stream(&latent, |part| {
            let range = part.frame_range();
            assert_eq!(range.start, next);
            assert!(range.len() <= 17);
            let host = part.to_pixels()?;

            for c in 0..3 {
                pixels[(c * frames + range.start) * plane..(c * frames + range.end) * plane]
                    .copy_from_slice(
                        &host.data[c * range.len() * plane..(c + 1) * range.len() * plane],
                    );
            }

            next = range.end;
            Ok(())
        })
        .unwrap();
    assert_eq!(next, *frames);
    assert_eq!(pixels, decoding.pixels.data);
    let mut calls = 0;
    let error = decoder
        .decode_stream(&latent, |_| {
            calls += 1;
            Err(mmh3_metal::Error::new("stop output".into()))
        })
        .unwrap_err();
    assert_eq!(calls, 1);
    assert_eq!(error.message, "stop output");
    let [channels, latent_frames, lh, lw] = latent.shape.as_slice() else {
        panic!("shape")
    };

    let single = Tensor::new(
        vec![*channels, 1, *lh, *lw],
        (0..*channels)
            .flat_map(|c| {
                latent.data[c * latent_frames * lh * lw..c * latent_frames * lh * lw + lh * lw]
                    .iter()
                    .copied()
            })
            .collect(),
    );
    let expected = decoder.decode(&single, false).unwrap().pixels;
    let mut calls = 0;
    decoder
        .decode_stream(&single, |part| {
            calls += 1;
            assert_eq!(part.frame_range(), 0..1);
            assert_eq!(part.to_pixels()?, expected);
            Ok(())
        })
        .unwrap();
    assert_eq!(calls, 1);
}
