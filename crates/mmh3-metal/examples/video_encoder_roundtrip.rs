//! Encodes raw RGB frames with the video VAE's encoder, decodes the mean latent back and reports
//! the timings and the round trip's PSNR. Run with
//! `cargo run -p mmh3-metal --release --example video_encoder_roundtrip -- VAE RGB WIDTH HEIGHT
//! FRAMES`, where RGB holds FRAMES frames of 8-bit RGB, as `ffmpeg -f rawvideo -pix_fmt rgb24`
//! writes them. One frame is encoded as a picture, and more as a clip.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use mmh3_core::{safetensors::SafeTensors, tensor::Tensor};
    use mmh3_metal::{
        AttentionPrecision, LinearPrecision,
        vae::MetalVideoDecoder,
        video_encoder::{DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE, MetalVideoEncoder, Temporal},
    };
    use std::{path::Path, time::Instant};

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [vae, rgb, width, height, frames] = &arguments[..] else {
        return Err("usage: video_encoder_roundtrip VAE RGB WIDTH HEIGHT FRAMES".into());
    };
    let (width, height, frames): (usize, usize, usize) =
        (width.parse()?, height.parse()?, frames.parse()?);
    let bytes = std::fs::read(rgb)?;
    if bytes.len() < frames * height * width * 3 {
        return Err(format!("{rgb} holds fewer than {frames} frames").into());
    }
    let pixels: Vec<f32> = bytes[..frames * height * width * 3]
        .iter()
        .map(|&value| value as f32 / 255.0)
        .collect();
    let file = SafeTensors::open(Path::new(vae))?;
    let temporal = if frames == 1 {
        Temporal::Frame
    } else {
        Temporal::Clip
    };

    let started = Instant::now();
    let encoder =
        MetalVideoEncoder::load(&file, temporal, DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?;
    println!(
        "loaded the encoder in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    let mut posterior = None;
    for trial in 0..3 {
        let started = Instant::now();
        posterior = Some(if frames == 1 {
            encoder.encode_picture(&Tensor::new(vec![height, width, 3], pixels.clone()))?
        } else {
            encoder.encode_clip(&Tensor::new(vec![frames, height, width, 3], pixels.clone()))?
        });
        println!(
            "encode {trial}: {width}×{height}×{frames} in {:.3} s",
            started.elapsed().as_secs_f64()
        );
    }
    println!(
        "device peak {:.2} GB",
        mmh3_metal::Device::shared()?.stats().peak_allocated_bytes as f64 / 1e9
    );
    drop(encoder);
    let latent = posterior.unwrap().mean_latent();
    println!("latent {:?}", latent.shape);
    if frames > 1 {
        // The causal encoder's first latent frame sees the first frame alone, so it is that
        // frame's picture latent.
        let encoder = MetalVideoEncoder::load(
            &file,
            Temporal::Frame,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_OVERLAP_MIN,
        )?;
        let picture = encoder
            .encode_picture(&Tensor::new(
                vec![height, width, 3],
                pixels[..height * width * 3].to_vec(),
            ))?
            .mean_latent();
        let plane = picture.data.len() / latent.shape[0];
        let first: Vec<f32> = (0..picture.data.len())
            .map(|index| latent.data[(index / plane * latent.shape[1]) * plane + index % plane])
            .collect();
        let dot: f64 = first
            .iter()
            .zip(&picture.data)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let norm = |values: &[f32]| {
            values
                .iter()
                .map(|&v| v as f64 * v as f64)
                .sum::<f64>()
                .sqrt()
        };
        println!(
            "first latent frame against the picture's: cosine {:.6}",
            dot / (norm(&first) * norm(&picture.data))
        );
    }

    let mut decoder = MetalVideoDecoder::load(
        &file,
        "",
        mmh3_core::vae::DEFAULT_TILE_SIZE,
        mmh3_core::vae::DEFAULT_TILE_OVERLAP_MIN,
    )?;
    decoder.set_precision(LinearPrecision::Int8, AttentionPrecision::Fp16)?;
    // The decoder takes a lone latent frame poorly, so a picture's decodes as a still clip of
    // two, of which the first frame is compared.
    let latent = if frames == 1 {
        let mut shape = latent.shape.clone();
        shape[1] = 2;
        let plane = latent.data.len() / shape[0];
        let data = latent
            .data
            .chunks(plane)
            .flat_map(|channel| channel.iter().chain(channel).copied())
            .collect();
        Tensor::new(shape, data)
    } else {
        latent
    };
    let decoded = decoder.decode(&latent, false)?.pixels;
    let [3, decoded_frames, decoded_height, decoded_width] = decoded.shape[..] else {
        return Err("unexpected decoded shape".into());
    };
    assert_eq!((decoded_height, decoded_width), (height, width));
    println!("decoded {decoded_frames} frames");
    let plane = height * width;
    let psnr = |error: f64, count: usize| 10.0 * (1.0 / (error / count as f64)).log10();
    let mut total = 0.0;
    for frame in 0..frames.min(decoded_frames) {
        let mut error = 0.0;
        for pixel in 0..plane {
            for channel in 0..3 {
                let original = pixels[(frame * plane + pixel) * 3 + channel] as f64;
                let value = decoded.data[(channel * decoded_frames + frame) * plane + pixel] as f64;
                error += (original - value).powi(2);
            }
        }
        total += error;
        if frames > 1 {
            println!("frame {frame}: PSNR {:.2} dB", psnr(error, plane * 3));
        }
    }
    println!(
        "PSNR {:.2} dB over {} frames",
        psnr(total, plane * 3 * frames.min(decoded_frames)),
        frames.min(decoded_frames)
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
