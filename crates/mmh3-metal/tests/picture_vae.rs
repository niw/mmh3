#![cfg(target_os = "macos")]

//! MiniMax H3 Image's picture VAE on Metal, when the models directory has it: a picture encoded and
//! decoded again comes back close to itself.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};
use mmh3_metal::vae::MetalVideoDecoder;
use mmh3_metal::video_encoder::{MetalVideoEncoder, Temporal};
use std::path::Path;

const CHECKPOINT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/vae/minimax_h3_image_vae_fp16.safetensors"
);

/// A picture `[height, width, 3]` of soft gradients and a few discs, smooth enough for a VAE to
/// keep, and with edges that cross the tiles.
fn picture(height: usize, width: usize) -> Tensor {
    let mut data = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (x as f32 / width as f32, y as f32 / height as f32);
            let disc = |cx: f32, cy: f32, r: f32| ((u - cx).powi(2) + (v - cy).powi(2)) < r * r;
            let mut rgb = [0.2 + 0.6 * u, 0.3 + 0.5 * v, 0.8 - 0.6 * u * v];
            if disc(0.3, 0.35, 0.18) {
                rgb = [0.9, 0.2, 0.15];
            }
            if disc(0.7, 0.6, 0.22) {
                rgb = [0.1, 0.5, 0.25];
            }
            data.extend(rgb);
        }
    }
    Tensor::new(vec![height, width, 3], data)
}

#[test]
fn decodes_what_it_encodes() {
    let path = Path::new(CHECKPOINT);
    if !path.exists() {
        eprintln!("no checkpoint at {}, skipping", path.display());
        return;
    }
    let file = SafeTensors::open(path).unwrap();
    let encoder = MetalVideoEncoder::load(
        &file,
        Temporal::Frame,
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    );
    let encoder = match encoder {
        Ok(encoder) => encoder,
        Err(error) => {
            eprintln!("{error}, skipping");
            return;
        }
    };
    let decoder =
        MetalVideoDecoder::load(&file, "", DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN).unwrap();

    // 640 × 512 pixels decode in tiles that overlap across both axes.
    let (height, width) = (512, 640);
    let original = picture(height, width);
    let latent = encoder.encode_picture(&original).unwrap().mean_latent();
    assert_eq!(latent.shape, [24, 1, height / 16, width / 16]);
    let decoded = decoder.decode(&latent, false).unwrap().pixels;
    assert_eq!(decoded.shape, [3, 1, height, width]);

    let plane = height * width;
    let squared: f64 = (0..plane * 3)
        .map(|index| {
            let (pixel, channel) = (index / 3, index % 3);
            let error = decoded.data[channel * plane + pixel] - original.data[index];
            f64::from(error * error)
        })
        .sum();
    let psnr = 10.0 * (1.0 / (squared / (plane * 3) as f64)).log10();
    eprintln!("decoded {width}×{height} {psnr:.1} dB PSNR from the picture it encoded");
    assert!(psnr > 30.0, "{psnr:.1} dB");

    // The picture decoder takes one latent frame.
    let frames = Tensor::new(
        vec![24, 2, height / 16, width / 16],
        [latent.data.clone(), latent.data].concat(),
    );
    assert!(decoder.decode(&frames, false).is_err());
}
