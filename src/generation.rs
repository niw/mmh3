//! Sampling shared by `mmh3 generate` and `mmh3-tools latent`: the generation settings, the
//! prompt's text states and the DiT's Euler steps from the seed's noise, with the keyframes of
//! first and last frame generation.

use crate::cli::{option_float, option_number};
use crate::models::{
    DIT_FILE, TEXT_ENCODER_FILE, load_dit, option_path, save_algorithm_cache, sparse_attention,
    video_vae_path,
};
use crate::pictures::load_picture;
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
    "first-frame",
    "last-frame",
    "steps",
    "schedule",
    "seed",
    "shift-video",
    "shift-audio",
    "dit",
    "text-encoder",
    "video-vae",
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

/// The seed of the noise the keyframes sample from the video VAE's posterior, as the reference
/// pipeline fixes it.
const KEYFRAME_POSTERIOR_SEED: u64 = 42;
/// Mixed into the seed for the noise of the keyframes, which comes from its own stream so that a
/// seed keeps the target noise it has without keyframes.
const KEYFRAME_NOISE_STREAM: u64 = 0x6b65_7966_7261_6d65;

/// A picture that a frame of the generation starts from or reaches, fitted to the canvas.
pub struct KeyframePicture {
    pub frame_index: usize,
    pub picture: mmh3_core::picture::Picture,
}

pub struct Settings {
    pub shape: GenerationShape,
    /// The first frame, then the last frame, when given.
    pub keyframes: Vec<KeyframePicture>,
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
        let pictures: Vec<(bool, mmh3_core::picture::Picture)> =
            [("first-frame", true), ("last-frame", false)]
                .into_iter()
                .filter_map(|(name, first)| {
                    options
                        .get(name)
                        .map(|path| load_picture(Path::new(path)).map(|picture| (first, picture)))
                })
                .collect::<Result<_, _>>()?;
        // Without a canvas size, the first picture gives the aspect ratio.
        let (width, height) = match (
            pictures.first(),
            options.contains_key("width") || options.contains_key("height"),
        ) {
            (Some((_, picture)), false) => {
                mmh3_core::picture::canvas_for(picture.width, picture.height)
            }
            _ => (
                option_number(options, "width", 1344)?,
                option_number(options, "height", 768)?,
            ),
        };
        let shape = GenerationShape::new(width, height, option_number(options, "frames", 124)?)?;
        // The first picture stretches to the canvas and the other covers it, as the reference
        // pipeline fits them.
        let keyframes = pictures
            .into_iter()
            .enumerate()
            .map(|(index, (first, picture))| KeyframePicture {
                frame_index: if first { 0 } else { shape.frames - 1 },
                picture: picture.fit(width, height, index == 0),
            })
            .collect();
        Ok(Settings {
            shape,
            keyframes,
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
    let pictures: Vec<Tensor> = settings
        .keyframes
        .iter()
        .map(|keyframe| keyframe.picture.to_tensor())
        .collect();
    let mut context_modalities = Vec::new();
    let context = match (prompt, options.get("context")) {
        (Some(prompt), None) => {
            let started = Instant::now();
            let path = option_path(options, "text-encoder", TEXT_ENCODER_FILE)?;
            let file = SafeTensors::open(Path::new(&path))?;
            let encoder = CudaTextEncoder::load(&file)?;
            let context = if pictures.is_empty() {
                let ids = Tokenizer::h3().encode(&prompt);
                encoder.encode(&ids, &[])?.context
            } else {
                use mmh3_core::vision::{VisionGrid, vision_prompt};
                use mmh3_cuda::vision::CudaVisionEncoder;

                let vision = CudaVisionEncoder::load(&file)?;
                let embeddings = pictures
                    .iter()
                    .map(|picture| vision.encode(picture))
                    .collect::<Result<Vec<_>, _>>()?;
                drop(vision);
                let grids: Vec<VisionGrid> = pictures
                    .iter()
                    .map(|picture| VisionGrid::for_picture(picture.shape[0], picture.shape[1]))
                    .collect();
                let prompt = vision_prompt(&Tokenizer::h3(), &prompt, &grids);
                context_modalities = prompt.modalities.clone();
                encoder.encode_prompt(&prompt, &embeddings, &[])?.context
            };
            println!(
                "encoded {} prompt tokens with {} pictures in {:.1} s",
                context.shape[0],
                pictures.len(),
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
    if options.contains_key("context") && !pictures.is_empty() {
        return Err("keyframes need a prompt, not --context".into());
    }
    let keyframes = encode_keyframes(options, settings, &pictures)?;
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

    let dit = load_dit(options, "dit", DIT_FILE)?;
    let sparse = sparse_attention(options, dit.has_vsa_gates())?;
    let schedule = &settings.schedule;
    for step in 0..steps {
        let started = Instant::now();
        let inputs = DitInputs {
            video: video.clone(),
            audio: audio.clone(),
            context: context.clone(),
            context_modalities: context_modalities.clone(),
            keyframes: keyframes.clone(),
            references: Vec::new(),
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

/// The keyframe latents as the DiT sees them: a sample of the video VAE's posterior for each picture
/// with 0.1% of noise mixed in.
fn encode_keyframes(
    options: &HashMap<&str, &str>,
    settings: &Settings,
    pictures: &[Tensor],
) -> Result<Vec<mmh3_core::dit::inputs::Keyframe>, Box<dyn Error>> {
    use mmh3_core::dit::inputs::Keyframe;
    use mmh3_core::dit::timestep::VIDEO_CONDITION_TIMESTEP;
    use mmh3_core::random::NormalSampler;
    use mmh3_cuda::video_encoder::CudaVideoEncoder;
    use std::time::Instant;

    if pictures.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = video_vae_path(options)?;
    let encoder = CudaVideoEncoder::load(&SafeTensors::open(Path::new(&path))?)?;
    let mut noise = NormalSampler::new(settings.seed ^ KEYFRAME_NOISE_STREAM);
    let keyframes = settings
        .keyframes
        .iter()
        .zip(pictures)
        .map(|(keyframe, picture)| {
            let posterior = encoder.encode_picture(picture)?;
            let count = posterior.mean.data.len();
            let mut latent = posterior
                .sampled_latent(&NormalSampler::new(KEYFRAME_POSTERIOR_SEED).samples(count));
            for (value, noise) in latent.data.iter_mut().zip(noise.samples(count)) {
                *value =
                    VIDEO_CONDITION_TIMESTEP * *value + (1.0 - VIDEO_CONDITION_TIMESTEP) * noise;
            }
            Ok(Keyframe {
                frame_index: keyframe.frame_index,
                video: Some(latent),
                audio: None,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    println!(
        "encoded {} keyframes with {path} in {:.1} s",
        keyframes.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(keyframes)
}
