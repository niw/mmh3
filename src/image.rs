//! Pictures generated with MiniMax H3 Image from text, or by editing reference pictures, written
//! to a PNG file.
//!
//! MiniMax H3 Image runs the MiniMax H3 DiT without its audio layers on one latent frame, with the
//! H3 text encoder unchanged, and decodes with a picture VAE of its own. Its checkpoint is distilled
//! to six steps on fixed sigmas with an output head per step and no guidance. An edit puts the
//! references in the prompt, through the vision tower, and in front of the generated rows, through
//! the VAE's encoder, as nearly clean conditions.

use mmh3_core::generation::CANVAS_MULTIPLE;
use mmh3_core::picture::Picture;
use std::error::Error;
#[cfg(feature = "cuda")]
use {
    crate::cli::{option_number, option_values, parse_options},
    crate::models::{TEXT_ENCODER_FILE, option_path},
    mmh3_core::safetensors::SafeTensors,
    mmh3_core::tensor::Tensor,
    std::path::Path,
    std::time::Instant,
};

pub const USAGE: &str = "usage:
  mmh3 image (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <image.png> [--models DIR]
             [--reference FILE]... [--width N] [--height N] [--min-generation-area PIXELS] [--seed N]
             [--dit FILE] [--text-encoder FILE] [--vae FILE]
             [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--vram-budget GB] [--consistent]

image generates a picture with MiniMax H3 Image from a prompt, used as given. The model expects a detailed English
description of the whole picture. Checkpoints default to their ComfyUI names inside a models directory laid out like
ComfyUI's models folder and the Comfy-Org/MiniMax-H3-Image repository, given by --models or MMH3_MODELS:
  diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_image_vae_fp16.safetensors
--width and --height default to 2048 and round to multiples of 32. The model works best at about 16 megapixels,
such as 4096x4096 or 5472x3072. A picture smaller than --min-generation-area pixels, 4194304 (2048x2048) by default,
is generated on a larger canvas of the same aspect ratio and resized to the requested size with Lanczos.
--reference takes a PNG or JPEG picture to edit, which the prompt refers to as <Picture 1>, <Picture 2> and so on,
in the order of the options, up to 16. The size of the result is still --width and --height.";

/// The DiT of MiniMax H3 Image inside the models directory.
pub const IMAGE_DIT_FILE: &str =
    "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors";
/// The picture VAE of MiniMax H3 Image inside the models directory.
pub const IMAGE_VAE_FILE: &str = "vae/minimax_h3_image_vae_fp16.safetensors";
/// Prompt tokens the reference keeps.
#[cfg(feature = "cuda")]
const MAX_PROMPT_TOKENS: usize = 4096;
#[cfg(feature = "cuda")]
const DEFAULT_SIDE: usize = 2048;
#[cfg(feature = "cuda")]
const DEFAULT_MIN_GENERATION_AREA: usize = 2048 * 2048;
/// The most reference pictures an edit takes.
#[cfg(feature = "cuda")]
const MAX_REFERENCES: usize = 16;
/// Bounds of a reference's short edge for the VAE's encoder and for the vision tower.
const REFERENCE_SHORT_EDGE: (usize, usize) = (1024, 2048);
const VISION_SHORT_EDGE: usize = 1024;
/// The area bounds of the Qwen2-VL image processor of the release.
const VISION_MIN_PIXELS: usize = 65_536;
const VISION_MAX_PIXELS: usize = 16_777_216;
/// Noise streams of the references, apart from the seed's noise of the generated picture.
#[cfg(feature = "cuda")]
const REFERENCE_POSTERIOR_STREAM: u64 = 0x5265_6650_6f73_7465;
#[cfg(feature = "cuda")]
const REFERENCE_NOISE_STREAM: u64 = 0x5265_664e_6f69_7365;

/// The canvas a `width × height` picture is generated on: scaled up at the same aspect ratio to at
/// least `min_area` pixels, with both sides rounded to the nearest multiple of 32.
pub fn canvas_size(width: usize, height: usize, min_area: usize) -> (usize, usize) {
    let scale = (min_area as f64 / (width * height) as f64).sqrt().max(1.0);
    let round = |side: usize| {
        let multiples = (side as f64 * scale / CANVAS_MULTIPLE as f64).round_ties_even() as usize;
        multiples.max(1) * CANVAS_MULTIPLE
    };
    (round(width), round(height))
}

/// A reference picture as an edit reads it: the copy for the VAE's encoder, its short edge within
/// 1024 to 2048 pixels and cropped at the top left to whole latent pixels, and the copy for the
/// vision tower, with a 1024-pixel short edge and sides that are multiples of 32.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditReference {
    pub encoded: Picture,
    pub vision: Picture,
}

impl EditReference {
    pub fn new(picture: &Picture) -> Result<Self, String> {
        use mmh3_core::picture::{bound_short_edge, vision_size};
        use mmh3_core::vae::SPATIAL_RATIO;

        if picture.width.min(picture.height) < CANVAS_MULTIPLE {
            return Err(format!(
                "a reference needs at least {CANVAS_MULTIPLE} pixels on each side, not {}×{}",
                picture.width, picture.height
            ));
        }
        let resized = |(width, height): (usize, usize)| match (width, height)
            == (picture.width, picture.height)
        {
            true => picture.clone(),
            false => picture.resize(width, height),
        };
        let (minimum, maximum) = REFERENCE_SHORT_EDGE;
        let encoded = resized(bound_short_edge(
            picture.width,
            picture.height,
            minimum,
            maximum,
            CANVAS_MULTIPLE,
        ));
        let encoded = encoded.crop(
            encoded.width / SPATIAL_RATIO * SPATIAL_RATIO,
            encoded.height / SPATIAL_RATIO * SPATIAL_RATIO,
        );
        let (width, height) = bound_short_edge(
            picture.width,
            picture.height,
            VISION_SHORT_EDGE,
            VISION_SHORT_EDGE,
            CANVAS_MULTIPLE,
        );
        // NOTE: the Qwen2-VL processor resizes a copy whose sides it rounds once more, which only
        // a picture with a 1024-pixel short edge and another side off the 32-pixel grid needs,
        // with bicubic interpolation. This resizes it from the original with Lanczos instead.
        let vision = resized(vision_size(
            width,
            height,
            CANVAS_MULTIPLE,
            VISION_MIN_PIXELS,
            VISION_MAX_PIXELS,
        ));
        Ok(EditReference { encoded, vision })
    }
}

/// The sigmas a DiT with an output head per step samples on, ending at 0, each rounded through its
/// FP32 timestep `1 − sigma` and back as the released sampler does.
pub fn step_schedule(step_sigmas: &[f32]) -> Vec<f32> {
    step_sigmas
        .iter()
        .map(|&sigma| 1.0 - (1.0 - sigma))
        .chain([0.0])
        .collect()
}

#[cfg(feature = "cuda")]
pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::euler_step;
    use mmh3_core::generation::VIDEO_LATENT_CHANNELS;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN, SPATIAL_RATIO};
    use mmh3_core::vision::{VisionGrid, picture_prompt};
    use mmh3_cuda::text_encoder::CudaTextEncoder;
    use mmh3_cuda::vae::CudaVideoDecoder;
    use mmh3_cuda::vision::CudaVisionEncoder;

    let options = parse_options(
        arguments,
        &[
            "prompt",
            "prompt-file",
            "context",
            "reference",
            "out",
            "models",
            "width",
            "height",
            "min-generation-area",
            "seed",
            "dit",
            "text-encoder",
            "vae",
            "attention-precision",
            "linear-precision",
            "vram-budget",
        ],
        &["consistent"],
        USAGE,
    )?;
    mmh3_cuda::check_device()?;
    crate::resident::take_budget(&options)?;
    mmh3_cuda::algorithms::set_consistent(options.contains_key("consistent"));
    if options.get("linear-precision").copied() == Some("nvfp4") {
        mmh3_cuda::nvfp4::check_supported()?;
    }
    crate::models::load_algorithm_cache();

    let out = Path::new(options.get("out").ok_or(USAGE)?);
    if !out
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    {
        return Err("--out must name a .png file".into());
    }
    let width = option_number(&options, "width", DEFAULT_SIDE)?;
    let height = option_number(&options, "height", DEFAULT_SIDE)?;
    let min_area = option_number(&options, "min-generation-area", DEFAULT_MIN_GENERATION_AREA)?;
    let seed = option_number(&options, "seed", 0)? as u64;
    if width < CANVAS_MULTIPLE || height < CANVAS_MULTIPLE {
        return Err(format!("--width and --height must be at least {CANVAS_MULTIPLE}").into());
    }
    if !(1024..=16_777_216).contains(&min_area) {
        return Err("--min-generation-area must be from 1024 to 16777216".into());
    }
    let (canvas_width, canvas_height) = canvas_size(width, height, min_area);

    let prompt = match (options.get("prompt"), options.get("prompt-file")) {
        (Some(prompt), None) => Some((*prompt).to_owned()),
        (None, Some(path)) => Some(std::fs::read_to_string(path)?),
        (None, None) => None,
        _ => return Err("pass either --prompt or --prompt-file".into()),
    };
    let references = option_values(arguments, "reference")
        .into_iter()
        .map(|path| -> Result<EditReference, Box<dyn Error>> {
            let picture = crate::pictures::load_picture(Path::new(path))?;
            Ok(EditReference::new(&picture).map_err(|error| format!("{path}: {error}"))?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if references.len() > MAX_REFERENCES {
        return Err(format!("pass at most {MAX_REFERENCES} --reference pictures").into());
    }
    let (context, context_modalities) = match (prompt, options.get("context")) {
        (Some(prompt), None) => {
            let started = Instant::now();
            let path = option_path(&options, "text-encoder", TEXT_ENCODER_FILE)?;
            let file = SafeTensors::open(Path::new(&path))?;
            let encoder = CudaTextEncoder::load_fitting(&file)?;
            let (context, modalities) = if references.is_empty() {
                let mut ids = Tokenizer::h3().encode(&prompt);
                ids.truncate(MAX_PROMPT_TOKENS);
                (encoder.encode(&ids, &[])?.context, Vec::new())
            } else {
                // The tower is gone before the prompt encodes.
                let embeddings = {
                    let vision = CudaVisionEncoder::load(&file)?;
                    references
                        .iter()
                        .map(|reference| vision.encode(&reference.vision.to_tensor()))
                        .collect::<Result<Vec<_>, _>>()?
                };
                let grids: Vec<VisionGrid> = references
                    .iter()
                    .map(|reference| {
                        VisionGrid::for_picture(reference.vision.height, reference.vision.width)
                    })
                    .collect();
                let prompt = picture_prompt(&Tokenizer::h3(), &prompt, &grids, MAX_PROMPT_TOKENS);
                let context = encoder.encode_prompt(&prompt, &embeddings, &[])?.context;
                (context, prompt.modalities)
            };
            println!(
                "encoded {} prompt tokens with {} pictures in {:.1} s",
                context.shape[0],
                references.len(),
                started.elapsed().as_secs_f64()
            );
            (context, modalities)
        }
        (None, Some(path)) => {
            let file = SafeTensors::open(Path::new(path))?;
            let mut context = Tensor::load(
                &file,
                file.get("context")
                    .ok_or("the context file has no context tensor")?,
            )?;
            if context.shape.len() == 3 && context.shape[0] == 1 {
                context.shape.remove(0);
            }
            if !references.is_empty() {
                return Err("references need a prompt, not --context".into());
            }
            (context, Vec::new())
        }
        _ => return Err("pass a prompt or --context, not both".into()),
    };
    let references = encode_references(&options, seed, &references)?;

    let dit = crate::models::load_dit(&options, "dit", IMAGE_DIT_FILE)?;
    let config = dit.config();
    if config.has_audio() || config.step_sigmas.is_empty() {
        return Err(
            "image needs the DiT of MiniMax H3 Image, which has no audio layers and an output \
             head per step"
                .into(),
        );
    }
    let sigmas = step_schedule(&config.step_sigmas);
    let latent_shape = vec![
        VIDEO_LATENT_CHANNELS,
        1,
        canvas_height / SPATIAL_RATIO,
        canvas_width / SPATIAL_RATIO,
    ];
    let mut latent = Tensor::new(
        latent_shape.clone(),
        NormalSampler::new(seed).samples(latent_shape.iter().product()),
    );
    let silence = Tensor::new(vec![0, 2, 0], Vec::new());
    println!(
        "sampling {canvas_width}×{canvas_height} in {} steps",
        sigmas.len() - 1
    );
    for step in 0..sigmas.len() - 1 {
        let started = Instant::now();
        let inputs = DitInputs {
            video: latent.clone(),
            audio: silence.clone(),
            context: context.clone(),
            context_modalities: context_modalities.clone(),
            keyframes: Vec::new(),
            references: references.clone(),
            sigma: sigmas[step],
            shift_video: 1.0,
            shift_audio: 1.0,
        };
        let outputs = dit.forward(&inputs, &[], None)?;
        euler_step(
            &mut latent.data,
            &outputs.video,
            sigmas[step],
            sigmas[step + 1],
        );
        println!(
            "step {}/{} at sigma {:.4} in {:.1} s",
            step + 1,
            sigmas.len() - 1,
            sigmas[step],
            started.elapsed().as_secs_f64()
        );
    }
    drop(dit);
    crate::models::save_algorithm_cache();

    let started = Instant::now();
    let path = option_path(&options, "vae", IMAGE_VAE_FILE)?;
    let decoder = CudaVideoDecoder::load(
        &SafeTensors::open(Path::new(&path))?,
        "",
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    )?;
    let pixels = decoder.decode(&latent, false)?.pixels;
    drop(decoder);
    println!(
        "decoded the picture with {path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    write_png(&pixels, width, height, out)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// The references as the DiT sees them: a sample of each one's posterior from the picture VAE's
/// encoder, rounded to FP16, cropped at the top left to whole patches and held just short of clean
/// with noise mixed in at the conditions' timestep.
#[cfg(feature = "cuda")]
fn encode_references(
    options: &std::collections::HashMap<&str, &str>,
    seed: u64,
    references: &[EditReference],
) -> Result<Vec<mmh3_core::dit::inputs::Reference>, Box<dyn Error>> {
    use mmh3_core::dit::inputs::Reference;
    use mmh3_core::dit::timestep::VIDEO_CONDITION_TIMESTEP;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};
    use mmh3_cuda::video_encoder::{CudaVideoEncoder, Temporal};

    if references.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = option_path(options, "vae", IMAGE_VAE_FILE)?;
    let encoder = CudaVideoEncoder::load(
        &SafeTensors::open(Path::new(&path))?,
        Temporal::Frame,
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    )?;
    let mut posterior_noise = NormalSampler::new(seed ^ REFERENCE_POSTERIOR_STREAM);
    let mut noise = NormalSampler::new(seed ^ REFERENCE_NOISE_STREAM);
    let latents = references
        .iter()
        .map(|reference| {
            let posterior = encoder.encode_picture(&reference.encoded.to_tensor())?;
            let count = posterior.mean.data.len();
            let mut latent =
                whole_patches(&posterior.sampled_latent(&posterior_noise.samples(count)));
            let count = latent.data.len();
            for (value, noise) in latent.data.iter_mut().zip(noise.samples(count)) {
                *value =
                    VIDEO_CONDITION_TIMESTEP * *value + (1.0 - VIDEO_CONDITION_TIMESTEP) * noise;
            }
            Ok(Reference::Picture(latent))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    println!(
        "encoded {} references with {path} in {:.1} s",
        latents.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(latents)
}

/// The top left of a latent `[channels, 1, height, width]` in whole 2 × 2 patches.
#[cfg(feature = "cuda")]
fn whole_patches(latent: &Tensor) -> Tensor {
    let (channels, height, width) = (latent.shape[0], latent.shape[2], latent.shape[3]);
    let (kept_height, kept_width) = (height / 2 * 2, width / 2 * 2);
    let mut data = Vec::with_capacity(channels * kept_height * kept_width);
    for channel in 0..channels {
        for row in 0..kept_height {
            let start = (channel * height + row) * width;
            data.extend_from_slice(&latent.data[start..start + kept_width]);
        }
    }
    Tensor::new(vec![channels, 1, kept_height, kept_width], data)
}

#[cfg(not(feature = "cuda"))]
pub fn run(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("image runs on CUDA only for now".into())
}

/// Writes pixels `[3, 1, height, width]` in [0, 1] as an 8-bit RGB PNG of `width × height`,
/// resized with Lanczos when the canvas differs. The values are truncated to 8 bits, as the
/// reference pipeline quantizes them, before any resize.
#[cfg(feature = "cuda")]
fn write_png(
    pixels: &Tensor,
    width: usize,
    height: usize,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    use image::{RgbImage, imageops};

    let (canvas_height, canvas_width) = (pixels.shape[2], pixels.shape[3]);
    let plane = canvas_height * canvas_width;
    let mut bytes = vec![0u8; 3 * plane];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let (pixel, channel) = (index / 3, index % 3);
        *byte = (pixels.data[channel * plane + pixel] * 255.0).clamp(0.0, 255.0) as u8;
    }
    let mut picture = RgbImage::from_raw(canvas_width as u32, canvas_height as u32, bytes)
        .ok_or("the decoded picture has the wrong size")?;
    if (canvas_width, canvas_height) != (width, height) {
        picture = imageops::resize(
            &picture,
            width as u32,
            height as u32,
            imageops::FilterType::Lanczos3,
        );
    }
    picture.save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_small_pictures_on_a_larger_canvas() {
        assert_eq!(canvas_size(4096, 4096, 2048 * 2048), (4096, 4096));
        assert_eq!(canvas_size(1024, 1024, 2048 * 2048), (2048, 2048));
        assert_eq!(canvas_size(1000, 500, 1024), (992, 512));
        assert_eq!(canvas_size(1920, 1080, 2048 * 2048), (2720, 1536));
    }

    #[test]
    fn sizes_the_copies_of_a_reference() {
        let picture = |width: usize, height: usize| Picture {
            width,
            height,
            pixels: vec![128; width * height * 3],
        };
        // A small picture grows to a 1024-pixel short edge for both.
        let small = EditReference::new(&picture(800, 600)).unwrap();
        assert_eq!((small.encoded.width, small.encoded.height), (1376, 1024));
        assert_eq!((small.vision.width, small.vision.height), (1376, 1024));
        // One within the VAE's bounds keeps its size there, cropped to whole latent pixels.
        let medium = EditReference::new(&picture(1500, 1100)).unwrap();
        assert_eq!(
            (medium.encoded.width, medium.encoded.height),
            (1488, 1100 / 16 * 16)
        );
        assert_eq!((medium.vision.width, medium.vision.height), (1408, 1024));
        // A 1024-pixel short edge with a side off the grid is rounded for the vision tower.
        let edge = EditReference::new(&picture(1500, 1024)).unwrap();
        assert_eq!((edge.vision.width, edge.vision.height), (1504, 1024));
        assert!(EditReference::new(&picture(20, 600)).is_err());
    }

    #[test]
    fn ends_the_schedule_at_zero() {
        let schedule = step_schedule(&[1.0, 0.75, 0.23200887]);
        assert_eq!(schedule.len(), 4);
        assert_eq!(schedule[0], 1.0);
        assert_eq!(schedule[3], 0.0);
        assert!((schedule[2] - 0.23200887).abs() < 1e-7);
    }
}
