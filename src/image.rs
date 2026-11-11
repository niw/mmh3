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
#[cfg(any(feature = "cuda", feature = "metal"))]
use {
    crate::cli::{option_float, option_number, option_values, parse_options},
    crate::models::{TEXT_ENCODER_FILE, option_path},
    mmh3_core::safetensors::SafeTensors,
    mmh3_core::tensor::Tensor,
    std::path::Path,
    std::time::Instant,
};

pub const USAGE: &str = "usage:
  mmh3 image (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <image.png> [--models DIR]
             [--reference FILE]... [--width N] [--height N] [--min-generation-area PIXELS] [--seed N]
             [--dit FILE] [--text-encoder FILE] [--vae FILE] [--save-latent <latent.safetensors>]
             [--save-context <text.safetensors>]
             [--attention-precision PRECISION] [--linear-precision PRECISION] [--vram-budget GB] [--consistent]
             [--attention dense|sol] [--sparse-tau X] [--sparse-start X] [--draft-steps N] [--latent-upscaler FILE]
             [--renoise-sigma X] [--steps N] [--shift X] [--patch FILE] [--lora FILE] [--lora-strength X] [--lora-mode adapter|merge]

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
in the order of the options, up to 16. The size of the result is still --width and --height.
--save-latent also writes the final latent, which tools/models/video_vae_int8.py calibrates an INT8 picture VAE on.
--attention-precision takes bf16 or int8-fp8 on CUDA and fp32, fp16 or int8-fp16 on Metal, and --linear-precision
int8 or nvfp4 on CUDA and fp32, fp16, int8 or mps-fp16 on Metal.
--attention sol runs Sol-Attn, a training-free block-sparse attention, as mmh3 generate does.
--draft-steps runs the first N steps on a canvas of half the width and height, which must then be multiples of 64,
and enlarges the latent they predict with the latent upscaler of --latent-upscaler, or bilinearly without one.
--renoise-sigma noises the enlarged draft to X rather than to the next sigma, and the steps at full size run on the
sigmas below X. The draft may then take every step.
--steps samples in N steps on the uniform grid shifted by --shift, 12 by default, rather than on the six sigmas the
checkpoint was distilled for, as a few-step LoRA such as PDMD's wants. A step then projects through the output heads
of the distilled sigmas it spans, weighted by how much of it each one covers. --patch and --lora apply as they do to
mmh3 generate.
--save-context also writes the text encoder's states of the prompt, which --context takes in place of the prompt
and its encoder on the next run.";

/// The DiT of MiniMax H3 Image inside the models directory.
pub const IMAGE_DIT_FILE: &str =
    "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors";
/// The picture VAE of MiniMax H3 Image inside the models directory.
pub const IMAGE_VAE_FILE: &str = "vae/minimax_h3_image_vae_fp16.safetensors";
/// Prompt tokens the reference keeps.
#[cfg(any(feature = "cuda", feature = "metal"))]
const MAX_PROMPT_TOKENS: usize = 4096;
#[cfg(any(feature = "cuda", feature = "metal"))]
const DEFAULT_SIDE: usize = 2048;
#[cfg(any(feature = "cuda", feature = "metal"))]
const DEFAULT_MIN_GENERATION_AREA: usize = 2048 * 2048;
/// The shift of the grid of `--steps`, the one PDMD's LoRA was distilled on.
#[cfg(any(feature = "cuda", feature = "metal"))]
const DEFAULT_SHIFT: f32 = 12.0;
/// The most reference pictures an edit takes.
#[cfg(any(feature = "cuda", feature = "metal"))]
const MAX_REFERENCES: usize = 16;
/// Bounds of a reference's short edge for the VAE's encoder and for the vision tower.
const REFERENCE_SHORT_EDGE: (usize, usize) = (1024, 2048);
const VISION_SHORT_EDGE: usize = 1024;
/// The area bounds of the Qwen2-VL image processor of the release.
const VISION_MIN_PIXELS: usize = 65_536;
const VISION_MAX_PIXELS: usize = 16_777_216;
/// Noise streams of the references, apart from the seed's noise of the generated picture.
#[cfg(any(feature = "cuda", feature = "metal"))]
const REFERENCE_POSTERIOR_STREAM: u64 = 0x5265_6650_6f73_7465;
#[cfg(any(feature = "cuda", feature = "metal"))]
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

/// The schedule of a run whose `draft_steps` first steps of `sigmas` run on the draft canvas, with
/// the enlarged draft noised to `renoise` and the steps at full size on the sigmas below it. The
/// draft may take every step of `sigmas`, which leaves one full-size step from `renoise` to 0.
pub fn renoised_schedule(
    sigmas: &[f32],
    draft_steps: usize,
    renoise: f32,
) -> Result<Vec<f32>, String> {
    let steps = sigmas.len() - 1;
    if draft_steps > steps {
        return Err(format!(
            "--draft-steps takes at most the {steps} steps there are"
        ));
    }
    let last_draft = sigmas[draft_steps - 1];
    if !(renoise > 0.0 && renoise < last_draft) {
        return Err(format!(
            "--renoise-sigma must be above 0 and below the last draft step's sigma {last_draft}"
        ));
    }
    Ok(sigmas[..draft_steps]
        .iter()
        .copied()
        .chain([renoise])
        .chain(
            sigmas[draft_steps..]
                .iter()
                .copied()
                .filter(|&sigma| sigma < renoise),
        )
        .collect())
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

#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::{Schedule, euler_step};
    use mmh3_core::generation::VIDEO_LATENT_CHANNELS;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_core::vae::SPATIAL_RATIO;
    use mmh3_core::vision::{VisionGrid, picture_prompt};
    #[cfg(feature = "cuda")]
    use mmh3_cuda::vision::CudaVisionEncoder as VisionEncoder;
    #[cfg(feature = "metal")]
    use mmh3_metal::vision::MetalVisionEncoder as VisionEncoder;

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
            "save-latent",
            "save-context",
            "attention-precision",
            "linear-precision",
            "vram-budget",
            "attention",
            "sparse-tau",
            "sparse-start",
            "draft-steps",
            "latent-upscaler",
            "renoise-sigma",
            "steps",
            "shift",
            "patch",
            "lora",
            "lora-strength",
            "lora-mode",
        ],
        &["consistent"],
        USAGE,
    )?;
    #[cfg(feature = "cuda")]
    mmh3_cuda::check_device()?;
    crate::resident::take_budget(&options)?;
    #[cfg(feature = "cuda")]
    {
        mmh3_cuda::algorithms::set_consistent(options.contains_key("consistent"));
        if options.get("linear-precision").copied() == Some("nvfp4") {
            mmh3_cuda::nvfp4::check_supported()?;
        }
        crate::models::load_algorithm_cache();
    }
    // A wrong precision is refused before anything is read.
    #[cfg(feature = "metal")]
    {
        crate::metal::attention_precision(&options)?;
        crate::metal::linear_precision(&options)?;
        if let Some(mode) = options.get("lora-mode")
            && *mode != "adapter"
        {
            return Err(format!("Metal runs --lora-mode adapter, not {mode}").into());
        }
    }

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
    let draft_steps = option_number(&options, "draft-steps", 0)?;
    if draft_steps > 0
        && (canvas_width % (2 * CANVAS_MULTIPLE) != 0 || canvas_height % (2 * CANVAS_MULTIPLE) != 0)
    {
        return Err(format!(
            "--draft-steps needs a canvas of multiples of {}, not {canvas_width}×{canvas_height}",
            2 * CANVAS_MULTIPLE
        )
        .into());
    }
    if draft_steps == 0 && options.contains_key("latent-upscaler") {
        return Err("--latent-upscaler enlarges the draft of --draft-steps".into());
    }
    let renoise_sigma = match options.contains_key("renoise-sigma") {
        true if draft_steps == 0 => {
            return Err("--renoise-sigma noises the draft of --draft-steps".into());
        }
        true => Some(option_float(&options, "renoise-sigma", 0.0)?),
        false => None,
    };
    let sparse = crate::models::sparse_attention(&options, false)?;
    let uniform_steps = match options.contains_key("steps") {
        true => Some(option_number(&options, "steps", 0)?),
        false => None,
    };
    if uniform_steps == Some(0) {
        return Err("--steps must be at least 1".into());
    }
    if uniform_steps.is_none() && options.contains_key("shift") {
        return Err("--shift shifts the grid of --steps".into());
    }
    let shift = option_float(&options, "shift", DEFAULT_SHIFT)?;

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
    let mut held = match crate::generation::keeps_models() {
        true => Held::Kept(crate::resident::Table::new(
            crate::resident::current_device(),
        )),
        false => Held::Alone,
    };
    if options.contains_key("save-context") && !references.is_empty() {
        return Err("--save-context saves a prompt without references".into());
    }
    let (context, context_modalities) = match (prompt, options.get("context")) {
        (Some(prompt), None) => {
            let started = Instant::now();
            let path = option_path(&options, "text-encoder", TEXT_ENCODER_FILE)?;
            let mut ids = Tokenizer::h3().encode(&prompt);
            ids.truncate(MAX_PROMPT_TOKENS);
            let key = crate::resident::ContextKey {
                encoder: crate::worker::digest(&SafeTensors::open(Path::new(&path))?),
                ids: ids.clone(),
                references: references.iter().map(reference_digest).collect(),
            };
            let (context, modalities) = match held.context(&key) {
                Some(kept) => {
                    println!(
                        "found the {} prompt tokens with {} pictures encoded",
                        kept.0.shape[0],
                        references.len()
                    );
                    kept
                }
                None => {
                    // A server's states are kept under the encoder, the prompt and the
                    // references, so its tower runs at the default precision, as its kept
                    // encoder does, rather than at whatever the request asked.
                    #[cfg(feature = "metal")]
                    let vision_precision = match &held {
                        Held::Alone => crate::metal::attention_precision(&options)?,
                        Held::Kept(_) => crate::metal::default_attention_precision(),
                    };
                    let (context, modalities) = held.text_encoder(&path, &options, |encoder| {
                        if references.is_empty() {
                            return Ok((encoder.encode(&ids, &[])?.context, Vec::new()));
                        }
                        // The tower is gone before the prompt encodes.
                        let embeddings = {
                            #[allow(unused_mut)]
                            let mut vision =
                                VisionEncoder::load(&SafeTensors::open(Path::new(&path))?)?;
                            #[cfg(feature = "metal")]
                            vision.set_precision(
                                crate::metal::default_linear_precision(),
                                vision_precision,
                            )?;
                            references
                                .iter()
                                .map(|reference| vision.encode(&reference.vision.to_tensor()))
                                .collect::<Result<Vec<_>, _>>()?
                        };
                        let grids: Vec<VisionGrid> = references
                            .iter()
                            .map(|reference| {
                                VisionGrid::for_picture(
                                    reference.vision.height,
                                    reference.vision.width,
                                )
                            })
                            .collect();
                        let prompt =
                            picture_prompt(&Tokenizer::h3(), &prompt, &grids, MAX_PROMPT_TOKENS);
                        let context = encoder.encode_prompt(&prompt, &embeddings, &[])?.context;
                        Ok((context, prompt.modalities))
                    })?;
                    println!(
                        "encoded {} prompt tokens with {} pictures in {:.1} s",
                        context.shape[0],
                        references.len(),
                        started.elapsed().as_secs_f64()
                    );
                    held.keep_context(key, &context, &modalities);
                    (context, modalities)
                }
            };
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
    if draft_steps > 0 && !references.is_empty() {
        return Err("--draft-steps cannot be combined with references".into());
    }
    if let Some(path) = options.get("save-context") {
        let mut metadata = Vec::new();
        if let Some(prompt) = options.get("prompt") {
            metadata.push(("prompt", *prompt));
        }
        mmh3_core::safetensors::write_f32(
            Path::new(path),
            &[("context", &context.shape, &context.data)],
            &metadata,
        )?;
        println!("wrote the context {:?} to {path}", context.shape);
    }
    let references = encode_references(&options, seed, &references)?;

    let (mut dit, dit_key) = held.dit(&options)?;
    let config = dit.config();
    if config.has_audio() || config.step_sigmas.is_empty() {
        return Err(
            "image needs the DiT of MiniMax H3 Image, which has no audio layers and an output \
             head per step"
                .into(),
        );
    }
    let mut sigmas = match uniform_steps {
        Some(steps) => {
            let uniform = Schedule::uniform(steps, shift, 1.0).video;
            step_schedule(&uniform[..steps])
        }
        None => step_schedule(&config.step_sigmas),
    };
    match renoise_sigma {
        Some(renoise) => sigmas = renoised_schedule(&sigmas, draft_steps, renoise)?,
        None if draft_steps + 1 >= sigmas.len() => {
            return Err(format!(
                "--draft-steps must leave at least one of the {} steps at full size",
                sigmas.len() - 1
            )
            .into());
        }
        None => {}
    }
    dit.set_sample_sigmas(&sigmas);
    let steps = sigmas.len() - 1;
    let (latent_height, latent_width) =
        (canvas_height / SPATIAL_RATIO, canvas_width / SPATIAL_RATIO);
    let latent_shape = match draft_steps {
        0 => vec![VIDEO_LATENT_CHANNELS, 1, latent_height, latent_width],
        _ => vec![
            VIDEO_LATENT_CHANNELS,
            1,
            latent_height / 2,
            latent_width / 2,
        ],
    };
    let mut noise = NormalSampler::new(seed);
    let mut latent = Tensor::new(
        latent_shape.clone(),
        noise.samples(latent_shape.iter().product()),
    );
    let mut inputs = DitInputs {
        video: latent.clone(),
        audio: Tensor::new(vec![0, 2, 0], Vec::new()),
        context,
        context_modalities,
        keyframes: Vec::new(),
        references,
        sigma: sigmas[0],
        shift_video: 1.0,
        shift_audio: 1.0,
    };
    println!(
        "sampling {canvas_width}×{canvas_height} in {steps} steps, {draft_steps} at half size"
    );
    let vae_path = option_path(&options, "vae", IMAGE_VAE_FILE)?;
    let vae_ahead = held.read_decoder_ahead(&vae_path, &options)?;
    // Metal refines the prompt once for every step.
    #[cfg(feature = "metal")]
    let prepared = held.with_room(|| dit.prepare_text(&inputs.context))?;
    for step in 0..steps {
        let started = Instant::now();
        // Draft steps fall short of the length below which Sol-Attn stays dense. Their layout is
        // redrawn at full size, so they go sparse anyway.
        let step_sparse = sparse
            .map(|sparse| match step < draft_steps {
                true => mmh3_core::dit::sparse::SparseAttention {
                    min_tokens: 0,
                    ..sparse
                },
                false => sparse,
            })
            .filter(|sparse| sparse.applies_to_step(step, steps));
        inputs.video = latent.clone();
        inputs.sigma = sigmas[step];
        let outputs = held.with_room(|| {
            #[cfg(feature = "cuda")]
            return dit.forward(&inputs, &[], step_sparse.as_ref());
            #[cfg(feature = "metal")]
            return prepared.forward(&inputs, &[], step_sparse.as_ref());
        })?;
        let (sigma, sigma_next) = (sigmas[step], sigmas[step + 1]);
        // The enlargement after the last draft step says its own time, so the step leaves it out.
        let mut enlarging = 0.0;
        if step + 1 == draft_steps {
            // The clean latent the draft predicts, enlarged to the full canvas and noised to the
            // next sigma with fresh noise of the seed, starts the steps at full size.
            for (value, &slope) in latent.data.iter_mut().zip(&outputs.video) {
                *value -= sigma * slope;
            }
            let enlarged = Instant::now();
            latent =
                crate::generation::enlarge_draft(&options, &latent, latent_height, latent_width)?;
            enlarging = enlarged.elapsed().as_secs_f64();
            let fresh = noise.samples(latent.data.len());
            for (value, fresh) in latent.data.iter_mut().zip(fresh) {
                *value = (1.0 - sigma_next) * *value + sigma_next * fresh;
            }
        } else {
            euler_step(&mut latent.data, &outputs.video, sigma, sigma_next);
        }
        let routing = outputs.routed_fraction.map_or(String::new(), |fraction| {
            format!(", Sol-Attn routed {:.1}%", 100.0 * fraction)
        });
        let draft = if step < draft_steps { " (draft)" } else { "" };
        let seconds = started.elapsed().as_secs_f64() - enlarging;
        println!(
            "step {}/{steps}{draft} at sigma {sigma:.4} in {seconds:.1} s{routing}",
            step + 1,
        );
        crate::generation::reached(crate::generation::Progress {
            step: step + 1,
            steps,
            seconds,
        });
    }
    #[cfg(feature = "metal")]
    drop(prepared);
    held.keep_dit(dit, dit_key);
    #[cfg(feature = "cuda")]
    crate::models::save_algorithm_cache();
    if let Some(path) = options.get("save-latent") {
        let values = [width.to_string(), height.to_string(), seed.to_string()];
        let mut metadata: Vec<(&str, &str)> = ["width", "height", "seed"]
            .into_iter()
            .zip(values.iter().map(String::as_str))
            .collect();
        if let Some(prompt) = options.get("prompt") {
            metadata.push(("prompt", prompt));
        }
        mmh3_core::safetensors::write_f32(
            Path::new(path),
            &[("video", &latent.shape, &latent.data)],
            &metadata,
        )?;
        println!("wrote the latent {:?} to {path}", latent.shape);
    }

    let started = Instant::now();
    let (pixels, timing) = held.decode(&vae_path, &options, &latent, vae_ahead)?;
    println!(
        "decoded the picture with {vae_path} in {:.1} s: {timing}",
        started.elapsed().as_secs_f64()
    );
    write_png(&pixels, width, height, out)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// Where a run takes its models from: read for it alone and let go of when it ends, as a command
/// does, or kept in this process's table for the runs after it, as a server does. The table is
/// the one the worker of a server keeps its models in, so a picture and a video share the text
/// encoder they both read.
#[cfg(any(feature = "cuda", feature = "metal"))]
enum Held {
    Alone,
    Kept(crate::resident::Table),
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Held {
    /// Runs `work` with the text encoder of `path`.
    fn text_encoder<T>(
        &mut self,
        path: &str,
        #[allow(unused_variables)] options: &std::collections::HashMap<&str, &str>,
        mut work: impl FnMut(&crate::resident::TextEncoder) -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        let file = SafeTensors::open(Path::new(path))?;
        match self {
            Held::Alone => {
                #[allow(unused_mut)]
                let mut encoder = crate::resident::TextEncoder::load_fitting(&file)?;
                // One prompt reads every layer once, so a Mac that streams the encoder keeps none
                // of its layers back.
                #[cfg(feature = "metal")]
                {
                    encoder.set_precision(
                        crate::metal::default_linear_precision(),
                        crate::metal::attention_precision(options)?,
                    )?;
                    encoder.read_once();
                }
                work(&encoder)
            }
            Held::Kept(table) => {
                let key = crate::worker::digest(&file);
                table.with_room(|models| work(models.text_encoder(Path::new(path), key)?))
            }
        }
    }

    /// The encoded prompt a server kept from an earlier picture of the same prompt and references
    /// through the same encoder, which spares it the encoder's load and the encode.
    fn context(
        &mut self,
        key: &crate::resident::ContextKey,
    ) -> Option<(Tensor, Vec<mmh3_core::dit::timestep::Modality>)> {
        match self {
            Held::Alone => None,
            Held::Kept(table) => table.borrow().context(key),
        }
    }

    /// Keeps an encoded prompt for the pictures after this one, where the models are kept.
    fn keep_context(
        &mut self,
        key: crate::resident::ContextKey,
        context: &Tensor,
        modalities: &[mmh3_core::dit::timestep::Modality],
    ) {
        if let Held::Kept(table) = self {
            table
                .borrow()
                .keep_context(key, context.clone(), modalities.to_vec());
        }
    }

    /// The DiT of `--dit` or the models directory, with the key it goes back into the table under
    /// when it is kept. A kept DiT runs INT8 linear layers on CUDA, since NVFP4 ones would stay in
    /// it for the runs after this one.
    fn dit(
        &mut self,
        options: &std::collections::HashMap<&str, &str>,
    ) -> Result<(crate::resident::Dit, Option<crate::resident::DitKey>), Box<dyn Error>> {
        use crate::resident::Dit;

        let table = match self {
            Held::Alone => {
                let dit = crate::models::load_dit(options, "dit", IMAGE_DIT_FILE)?;
                return Ok((dit, None));
            }
            Held::Kept(table) => table,
        };
        #[cfg(feature = "cuda")]
        if options
            .get("linear-precision")
            .is_some_and(|precision| *precision != "int8")
        {
            return Err("a server keeps the DiT with INT8 linear layers".into());
        }
        let path = option_path(options, "dit", IMAGE_DIT_FILE)?;
        let file = SafeTensors::open(Path::new(&path))?;
        let adapters = wanted_adapters(options)?;
        let key = crate::resident::DitKey {
            checkpoint: crate::worker::digest(&file),
            adapters: adapters
                .iter()
                .map(|(_, adapter)| (adapter.digest, adapter.strength.to_bits(), adapter.mode))
                .collect(),
        };
        let mut models = table.borrow();
        // Adapters only ever add, so a kept DiT whose adapters begin this run's takes the rest.
        let (mut dit, applied) = match models.take_dit() {
            Some((kept, dit))
                if kept.checkpoint == key.checkpoint
                    && key.adapters.starts_with(&kept.adapters) =>
            {
                (dit, kept.adapters.len())
            }
            other => {
                drop(other);
                let started = Instant::now();
                let dit = models
                    .read_fitting(|| Dit::load(&file, ""), || Dit::load_fitting(&file, ""))?;
                println!("loaded {path} in {:.1} s", started.elapsed().as_secs_f64());
                (dit, 0)
            }
        };
        for (adapter_path, adapter) in &adapters[applied..] {
            let started = Instant::now();
            let layers = crate::worker::add_adapter(
                &mut dit,
                &SafeTensors::open(Path::new(adapter_path))?,
                adapter,
            )?;
            println!(
                "added {adapter_path} to {layers} layers at strength {} in {:.1} s",
                adapter.strength,
                started.elapsed().as_secs_f64()
            );
        }
        #[cfg(feature = "cuda")]
        dit.set_attention_precision(attention_precision(options)?);
        #[cfg(feature = "metal")]
        {
            dit.set_linear_precision(crate::metal::linear_precision(options)?)?;
            dit.set_attention_precision(crate::metal::attention_precision(options)?)?;
        }
        Ok((dit, Some(key)))
    }

    /// Puts a kept DiT back into the table, and lets go of one read for this run alone.
    fn keep_dit(&mut self, dit: crate::resident::Dit, key: Option<crate::resident::DitKey>) {
        if let (Held::Kept(table), Some(key)) = (self, key) {
            table.borrow().keep_dit(key, dit);
        }
    }

    /// Runs `work`, which can run again, letting go of the kept model used least and running it
    /// again each time the device runs out of memory: a kept model may be holding the room a step
    /// or the prompt's refinement needs. A run holding nothing else fails as the device did.
    fn with_room<T, E: Into<Box<dyn Error>>>(
        &mut self,
        mut work: impl FnMut() -> Result<T, E>,
    ) -> Result<T, Box<dyn Error>> {
        loop {
            match work().map_err(Into::into) {
                Err(error) if crate::resident::out_of_memory(&*error) && self.let_go_of_one() => {}
                result => return result,
            }
        }
    }

    /// Lets go of the kept model used least, for work that ran out of memory, and says whether
    /// there was one.
    fn let_go_of_one(&mut self) -> bool {
        match self {
            Held::Alone => false,
            Held::Kept(table) => table.let_go_of_one(),
        }
    }

    /// Starts reading the picture VAE of `path` on a thread of its own, so that the read hides
    /// behind the steps: it uploads the weights between the steps' calls on the device. A run alone
    /// gets the decoder back through the handle; a server's table keeps it, where the decode finds
    /// it. A read that fails says so and leaves the decode to read again, having cost the steps
    /// nothing.
    fn read_decoder_ahead(
        &mut self,
        path: &str,
        options: &std::collections::HashMap<&str, &str>,
    ) -> Result<DecoderAhead, Box<dyn Error>> {
        use crate::resident::Table;

        let device = crate::resident::current_device();
        let path = path.to_owned();
        let read: DecoderRead = match self {
            Held::Alone => {
                let precision = decoder_precision(options)?;
                Box::new(move || {
                    let file = SafeTensors::open(Path::new(&path))?;
                    Ok(Some(load_decoder(&file, precision)?))
                })
            }
            Held::Kept(table) => {
                let device = table.device();
                Box::new(move || {
                    let file = SafeTensors::open(Path::new(&path))?;
                    let key = decoder_key(&file);
                    Table::new(device).with_room(|models| {
                        models.video_decoder(Path::new(&path), key).map(|_| ())
                    })?;
                    Ok(None)
                })
            }
        };
        let thread = std::thread::Builder::new()
            .name("picture VAE".to_owned())
            .spawn(move || {
                let started = Instant::now();
                let decoder = match crate::resident::bind_device(device).and_then(|()| read()) {
                    Ok(decoder) => decoder,
                    Err(error) => {
                        eprintln!("warning: reading the picture VAE during the steps: {error}");
                        None
                    }
                };
                (decoder, started.elapsed().as_secs_f64())
            })?;
        Ok(DecoderAhead { thread })
    }

    /// Decodes a latent with the picture VAE of `path` into pixels `[3, 1, height, width]`, with
    /// the decoder `ahead` read during the steps or one read here.
    fn decode(
        &mut self,
        path: &str,
        options: &std::collections::HashMap<&str, &str>,
        latent: &Tensor,
        ahead: DecoderAhead,
    ) -> Result<(Tensor, DecodeTiming), Box<dyn Error>> {
        let waited = Instant::now();
        let (decoder, mut timing) = match ahead.thread.join() {
            Ok((decoder, read)) => (
                decoder,
                DecodeTiming {
                    read,
                    waited: waited.elapsed().as_secs_f64(),
                    ..DecodeTiming::default()
                },
            ),
            Err(_) => (None, DecodeTiming::default()),
        };
        let read = Instant::now();
        let file = SafeTensors::open(Path::new(path))?;
        let (pixels, decoded, read_back) = match self {
            Held::Alone => {
                let decoder = match decoder {
                    Some(decoder) => decoder,
                    None => load_decoder(&file, decoder_precision(options)?)?,
                };
                timing.read += read.elapsed().as_secs_f64();
                decode_timed(&decoder, latent)?
            }
            Held::Kept(table) => {
                let key = decoder_key(&file);
                table.with_room(|models| {
                    let decoder = models.video_decoder(Path::new(path), key)?;
                    timing.read += read.elapsed().as_secs_f64();
                    decode_timed(decoder, latent)
                })?
            }
        };
        timing.decoded = decoded;
        timing.read_back = read_back;
        Ok((pixels, timing))
    }
}

/// The thread reading the picture VAE while the steps run, with the decoder it read for a run
/// alone and how long it took.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct DecoderAhead {
    thread: std::thread::JoinHandle<(Option<crate::resident::VideoDecoder>, f64)>,
}

/// The read that thread runs, which gives back the decoder for a run alone.
#[cfg(any(feature = "cuda", feature = "metal"))]
type DecoderRead =
    Box<dyn FnOnce() -> Result<Option<crate::resident::VideoDecoder>, Box<dyn Error>> + Send>;

/// Where the time of a decode went: reading the VAE, which mostly overlaps the steps and only
/// counts as waited for where it ran past them, the decode on the device and the pixels' way back.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Default)]
struct DecodeTiming {
    read: f64,
    waited: f64,
    decoded: f64,
    read_back: f64,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl std::fmt::Display for DecodeTiming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VAE read in {:.1} s during the steps and waited for {:.1} s, decoded in {:.1} s, read \
             back in {:.1} s",
            self.read, self.waited, self.decoded, self.read_back
        )
    }
}

/// What a kept picture VAE is: its checkpoint and the tile geometry a picture decodes in.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn decoder_key(file: &SafeTensors) -> crate::resident::VideoDecoderKey {
    use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};

    (
        crate::worker::digest(file),
        DEFAULT_TILE_SIZE,
        PICTURE_TILE_OVERLAP_MIN,
    )
}

/// The precisions a run alone decodes with, which on Metal follow `--attention-precision`.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn decoder_precision(
    #[allow(unused_variables)] options: &std::collections::HashMap<&str, &str>,
) -> Result<DecoderPrecision, Box<dyn Error>> {
    #[cfg(feature = "metal")]
    return Ok((
        crate::metal::default_linear_precision(),
        crate::metal::attention_precision(options)?,
    ));
    #[cfg(feature = "cuda")]
    Ok(())
}

#[cfg(feature = "metal")]
type DecoderPrecision = (mmh3_metal::LinearPrecision, mmh3_metal::AttentionPrecision);
#[cfg(feature = "cuda")]
type DecoderPrecision = ();

/// The picture VAE of `file` with its weights on the device, for a run alone.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn load_decoder(
    file: &SafeTensors,
    #[allow(unused_variables)] precision: DecoderPrecision,
) -> Result<crate::resident::VideoDecoder, Box<dyn Error>> {
    use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};

    #[allow(unused_mut)]
    let mut decoder =
        crate::resident::VideoDecoder::load(file, "", DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN)?;
    #[cfg(feature = "metal")]
    decoder.set_precision(precision.0, precision.1)?;
    Ok(decoder)
}

/// Decodes `latent`, with the seconds the device took and the seconds the pixels took to come
/// back. CUDA's decode brings them back itself, so there the first holds both.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn decode_timed(
    decoder: &crate::resident::VideoDecoder,
    latent: &Tensor,
) -> Result<(Tensor, f64, f64), Box<dyn Error>> {
    let started = Instant::now();
    #[cfg(feature = "metal")]
    {
        let frames = decoder.decode_device(latent)?;
        mmh3_metal::Device::shared()?.synchronize()?;
        let decoded = started.elapsed().as_secs_f64();
        let pixels = frames.to_pixels()?;
        Ok((pixels, decoded, started.elapsed().as_secs_f64() - decoded))
    }
    #[cfg(feature = "cuda")]
    {
        let pixels = decoder.decode(latent, false)?.pixels;
        Ok((pixels, started.elapsed().as_secs_f64(), 0.0))
    }
}

/// The patch of `--patch` and the LoRA of `--lora` with the files they come from, in the order they
/// apply, since they do not commute. A patch applies as adapters at full strength.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn wanted_adapters(
    options: &std::collections::HashMap<&str, &str>,
) -> Result<Vec<(String, mmh3_core::worker::Adapter)>, Box<dyn Error>> {
    use crate::models::model_file;
    use mmh3_core::worker::Adapter;

    let wanted = [
        (
            model_file(options, "patch", &["patches", "loras"])?,
            1.0,
            "adapter",
        ),
        (
            model_file(options, "lora", &["loras"])?,
            option_float(options, "lora-strength", 1.0)?,
            options.get("lora-mode").copied().unwrap_or("adapter"),
        ),
    ];
    let mut adapters = Vec::new();
    for (path, strength, mode) in wanted {
        let Some(path) = path else {
            continue;
        };
        let mode = match mode {
            "adapter" => Adapter::ADAPTER,
            "merge" => Adapter::MERGE,
            other => {
                return Err(format!("--lora-mode must be adapter or merge, not {other}").into());
            }
        };
        let digest = crate::worker::digest(&SafeTensors::open(Path::new(&path))?);
        adapters.push((
            path,
            Adapter {
                digest,
                strength,
                mode,
            },
        ));
    }
    Ok(adapters)
}

/// The attention precision of `--attention-precision`, BF16 by default.
#[cfg(feature = "cuda")]
fn attention_precision(
    options: &std::collections::HashMap<&str, &str>,
) -> Result<mmh3_cuda::attention::AttentionPrecision, Box<dyn Error>> {
    use mmh3_cuda::attention::AttentionPrecision;

    match options
        .get("attention-precision")
        .copied()
        .unwrap_or("bf16")
    {
        "bf16" => Ok(AttentionPrecision::Bf16),
        "int8-fp8" => Ok(AttentionPrecision::Int8Fp8),
        other => Err(format!("--attention-precision must be bf16 or int8-fp8, not {other}").into()),
    }
}

/// The references as the DiT sees them: a sample of each one's posterior from the picture VAE's
/// encoder, rounded to FP16, cropped at the top left to whole patches and held just short of clean
/// with noise mixed in at the conditions' timestep.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn encode_references(
    options: &std::collections::HashMap<&str, &str>,
    seed: u64,
    references: &[EditReference],
) -> Result<Vec<mmh3_core::dit::inputs::Reference>, Box<dyn Error>> {
    use mmh3_core::dit::inputs::Reference;
    use mmh3_core::dit::timestep::VIDEO_CONDITION_TIMESTEP;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::vae::{DEFAULT_TILE_SIZE, PICTURE_TILE_OVERLAP_MIN};
    #[cfg(feature = "cuda")]
    use mmh3_cuda::video_encoder::{CudaVideoEncoder as VideoEncoder, Temporal};
    #[cfg(feature = "metal")]
    use mmh3_metal::video_encoder::{MetalVideoEncoder as VideoEncoder, Temporal};

    if references.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = option_path(options, "vae", IMAGE_VAE_FILE)?;
    let encoder = VideoEncoder::load(
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

/// What tells one reference from another in a kept prompt: the pixels of both copies the
/// encoders read.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn reference_digest(reference: &EditReference) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for picture in [&reference.encoded, &reference.vision] {
        (picture.width, picture.height, &picture.pixels).hash(&mut hasher);
    }
    hasher.finish()
}

/// The top left of a latent `[channels, 1, height, width]` in whole 2 × 2 patches.
#[cfg(any(feature = "cuda", feature = "metal"))]
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

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn run(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("image needs a build with CUDA or Metal".into())
}

/// Writes pixels `[3, 1, height, width]` in [0, 1] as an 8-bit RGB PNG of `width × height`,
/// resized with Lanczos when the canvas differs. The values are truncated to 8 bits, as the
/// reference pipeline quantizes them, before any resize.
#[cfg(any(feature = "cuda", feature = "metal"))]
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
    fn noises_the_draft_to_the_renoise_sigma() {
        let sigmas = [1.0, 0.96, 0.857, 0.0];
        assert_eq!(
            renoised_schedule(&sigmas, 1, 0.5).unwrap(),
            vec![1.0, 0.5, 0.0]
        );
        assert_eq!(
            renoised_schedule(&sigmas, 2, 0.9).unwrap(),
            vec![1.0, 0.96, 0.9, 0.857, 0.0]
        );
        assert_eq!(
            renoised_schedule(&sigmas, 3, 0.5).unwrap(),
            vec![1.0, 0.96, 0.857, 0.5, 0.0]
        );
        assert!(renoised_schedule(&sigmas, 2, 0.96).is_err());
        assert!(renoised_schedule(&sigmas, 2, 0.0).is_err());
        assert!(renoised_schedule(&sigmas, 4, 0.5).is_err());
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
