//! Sampling shared by `mmh3 generate` and `mmh3-tools latent`: the generation settings, the
//! prompt's text states and the DiT's Euler steps from the seed's noise.

use crate::cli::{option_float, option_number};
use crate::models::{
    TEXT_ENCODER_FILE, load_dit, option_path, save_algorithm_cache, sparse_attention,
};
use mmh3_core::dit::sampler::Schedule;
use mmh3_core::generation::GenerationShape;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

/// The options that choose what is sampled and with which models.
pub const OPTIONS: &[&str] = &[
    "prompt",
    "prompt-file",
    "context",
    "models",
    "width",
    "height",
    "frames",
    "steps",
    "schedule",
    "seed",
    "shift-video",
    "shift-audio",
    "dit",
    "text-encoder",
    "patch",
    "lora",
    "lora-strength",
    "lora-mode",
    "attention",
    "attention-precision",
    "linear-precision",
    "sparse-tau",
    "sparse-start",
    "vsa-sparsity",
];

pub struct Settings {
    pub shape: GenerationShape,
    pub schedule: Schedule,
    pub steps: usize,
    pub seed: u64,
    pub shift_video: f32,
    pub shift_audio: f32,
}

impl Settings {
    /// The canvas, step schedule, seed and schedule shifts of the options, with their defaults.
    pub fn parse(options: &HashMap<&str, &str>) -> Result<Self, Box<dyn Error>> {
        let shift_video = option_float(options, "shift-video", 12.0)?;
        let shift_audio = option_float(options, "shift-audio", 3.0)?;
        let schedule = match options.get("schedule").copied().unwrap_or("uniform") {
            "uniform" => {
                let steps = option_number(options, "steps", 20)?;
                if steps == 0 {
                    return Err("--steps must be at least 1".into());
                }
                Schedule::uniform(steps, shift_video, shift_audio)
            }
            "taomate" => {
                if options.contains_key("steps") {
                    return Err("--schedule taomate always runs 3 steps, leave out --steps".into());
                }
                Schedule::taomate(shift_video, shift_audio)
            }
            other => {
                return Err(format!("--schedule must be uniform or taomate, not {other}").into());
            }
        };
        Ok(Settings {
            shape: GenerationShape::new(
                option_number(options, "width", 1344)?,
                option_number(options, "height", 768)?,
                option_number(options, "frames", 124)?,
            )?,
            steps: schedule.steps(),
            schedule,
            seed: option_number(options, "seed", 0)? as u64,
            shift_video,
            shift_audio,
        })
    }
}

/// Encodes the prompt, or reads the text states of `--context`, and runs the DiT's Euler steps
/// from the seed's noise. Returns the final video and audio latents.
pub fn sample(
    options: &HashMap<&str, &str>,
    settings: &Settings,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::euler_step;
    use mmh3_core::generation::FPS;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_cuda::text_encoder::CudaTextEncoder;
    use std::time::Instant;

    let prompt = match (options.get("prompt"), options.get("prompt-file")) {
        (Some(prompt), None) => Some((*prompt).to_owned()),
        (None, Some(path)) => Some(std::fs::read_to_string(path)?),
        (None, None) => None,
        _ => return Err("pass either --prompt or --prompt-file".into()),
    };
    let context = match (prompt, options.get("context")) {
        (Some(prompt), None) => {
            let started = Instant::now();
            let ids = Tokenizer::h3().encode(&prompt);
            let path = option_path(options, "text-encoder", TEXT_ENCODER_FILE)?;
            let encoder = CudaTextEncoder::load(&SafeTensors::open(Path::new(&path))?)?;
            let context = encoder.encode(&ids, &[])?.context;
            println!(
                "encoded {} prompt tokens in {:.1} s",
                ids.len(),
                started.elapsed().as_secs_f64()
            );
            context
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
            context
        }
        _ => return Err("pass a prompt or --context, not both".into()),
    };
    // NOTE: the video noise is drawn before the audio noise, the order of the official pipeline.
    let shape = &settings.shape;
    let mut noise = NormalSampler::new(settings.seed);
    let (video_shape, audio_shape) = (shape.video_latent_shape(), shape.audio_latent_shape());
    let mut video = Tensor::new(
        video_shape.clone(),
        noise.samples(video_shape.iter().product()),
    );
    let mut audio = Tensor::new(
        audio_shape.clone(),
        noise.samples(audio_shape.iter().product()),
    );
    let (steps, seed) = (settings.steps, settings.seed);
    println!(
        "{}×{}, {} frames ({:.2} s), {} text tokens, {steps} steps, seed {seed}",
        shape.width,
        shape.height,
        shape.frames,
        shape.frames as f64 / FPS as f64,
        context.shape[0]
    );

    let dit = load_dit(options, "dit")?;
    let sparse = sparse_attention(options, dit.has_vsa_gates())?;
    let schedule = &settings.schedule;
    for step in 0..steps {
        let started = Instant::now();
        let inputs = DitInputs {
            video: video.clone(),
            audio: audio.clone(),
            context: context.clone(),
            sigma: schedule.video[step],
            shift_video: settings.shift_video,
            shift_audio: settings.shift_audio,
        };
        let step_sparse = sparse.filter(|settings| settings.applies_to_step(step, steps));
        let outputs = dit.forward(&inputs, &[], step_sparse.as_ref())?;
        euler_step(
            &mut video.data,
            &outputs.video,
            schedule.video[step],
            schedule.video[step + 1],
        );
        euler_step(
            &mut audio.data,
            &outputs.audio,
            schedule.audio[step],
            schedule.audio[step + 1],
        );
        let routing = outputs.routed_fraction.map_or(String::new(), |fraction| {
            format!(", Sol-Attn routed {:.1}%", 100.0 * fraction)
        });
        println!(
            "step {}/{steps} at sigma {:.4} in {:.1} s{routing}",
            step + 1,
            schedule.video[step],
            started.elapsed().as_secs_f64()
        );
    }
    save_algorithm_cache();
    Ok((video, audio))
}
