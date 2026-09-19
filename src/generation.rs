//! Sampling shared by `mmh3 generate` and `mmh3-tools latent`: the generation settings, the
//! prompt's text states and the DiT's Euler steps from the seed's noise, with the keyframes of
//! first and last frame generation or the reference pictures of reference to video generation.

use crate::audio::load_audio;
use crate::cli::{option_float, option_number, option_values};
#[cfg(feature = "cuda")]
use crate::models::{AUDIO_VAE_FILE, save_algorithm_cache, video_vae_path};
use crate::models::{
    DIT_FILE, REFERENCE_DIT_FILE, TEXT_ENCODER_FILE, load_dit, option_path, sparse_attention,
};
use crate::pictures::load_picture;
use crate::video::{ReferenceClip, block_seconds};
#[cfg(feature = "cuda")]
use crate::video::{block_frames, load_clip};
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
    "reference",
    "reference-audio",
    "reference-video",
    "steps",
    "schedule",
    "seed",
    "shift-video",
    "shift-audio",
    "dit",
    "text-encoder",
    "video-vae",
    "audio-vae",
    "patch",
    "lora",
    "lora-strength",
    "lora-mode",
    "worker",
    "shard-dit",
    "token",
    "attention",
    "attention-precision",
    "linear-precision",
    "sparse-tau",
    "sparse-start",
    "vsa-sparsity",
];

/// The seed of the noise keyframes and reference pictures sample from the video VAE's posterior,
/// as the reference pipeline fixes it.
#[cfg(feature = "cuda")]
const KEYFRAME_POSTERIOR_SEED: u64 = 42;
/// Mixed into the seed for the noise of keyframes and reference pictures, which comes from its own
/// stream so that a seed keeps the target noise it has without them.
#[cfg(feature = "cuda")]
const KEYFRAME_NOISE_STREAM: u64 = 0x6b65_7966_7261_6d65;
/// The most reference pictures the official pipeline takes.
const MAX_REFERENCE_PICTURES: usize = 9;
/// The most reference sounds the official pipeline takes.
const MAX_REFERENCE_SOUNDS: usize = 3;
/// The most reference clips the official pipeline takes.
const MAX_REFERENCE_CLIPS: usize = 3;
/// Mixed into the seed for the noise of reference clips, so that they take their own stream.
#[cfg(feature = "cuda")]
const CLIP_NOISE_STREAM: u64 = 0x636c_6970_6e6f_6973;

/// A picture that a frame of the generation starts from or reaches, fitted to the canvas.
pub struct KeyframePicture {
    pub frame_index: usize,
    pub picture: mmh3_core::picture::Picture,
}

pub struct Settings {
    pub shape: GenerationShape,
    /// The first frame, then the last frame, when given.
    pub keyframes: Vec<KeyframePicture>,
    /// Pictures of `--reference` in their order, each scaled to its size for the DiT.
    pub references: Vec<mmh3_core::picture::Picture>,
    /// Waveforms of `--reference-audio` in their order, `[2, samples]` at the audio VAE's rate.
    pub sounds: Vec<Tensor>,
    /// Clips of `--reference-video` in their order, each on its own canvas with its soundtrack.
    pub clips: Vec<ReferenceClip>,
    pub schedule: Schedule,
    pub steps: usize,
    pub seed: u64,
    pub shift_video: f32,
    pub shift_audio: f32,
    /// Addresses of `--worker`, machines this run may borrow.
    pub workers: Vec<String>,
    /// The shared secret of `--token`, empty when there is none.
    pub token: String,
}

impl Settings {
    /// The worker addresses as borrowed strings, for `worker::connect_all`.
    pub fn workers_borrowed(&self) -> Vec<&str> {
        self.workers.iter().map(String::as_str).collect()
    }

    /// The canvas, step schedule, seed and schedule shifts of the options, with their defaults.
    /// `arguments` gives the repeated `--reference` and `--worker` options.
    pub fn parse(
        options: &HashMap<&str, &str>,
        arguments: &[String],
    ) -> Result<Self, Box<dyn Error>> {
        #[cfg(feature = "metal")]
        crate::metal::validate_options(options)?;
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
        let references = option_values(arguments, "reference");
        if references.len() > MAX_REFERENCE_PICTURES {
            return Err(
                format!("pass at most {MAX_REFERENCE_PICTURES} --reference pictures").into(),
            );
        }
        let files = option_values(arguments, "reference-audio");
        if files.len() > MAX_REFERENCE_SOUNDS {
            return Err(
                format!("pass at most {MAX_REFERENCE_SOUNDS} --reference-audio files").into(),
            );
        }
        let sounds = files
            .into_iter()
            .map(|path| load_audio(Path::new(path)))
            .collect::<Result<Vec<_>, _>>()?;
        let files = option_values(arguments, "reference-video");
        if files.len() > MAX_REFERENCE_CLIPS {
            return Err(
                format!("pass at most {MAX_REFERENCE_CLIPS} --reference-video files").into(),
            );
        }
        let references = references
            .into_iter()
            .map(|path| load_picture(Path::new(path)))
            .collect::<Result<Vec<_>, _>>()?;
        if !pictures.is_empty() && !(references.is_empty() && sounds.is_empty() && files.is_empty())
        {
            return Err(
                "--reference, --reference-audio and --reference-video cannot be combined with --first-frame or --last-frame"
                    .into(),
            );
        }
        // Without a canvas size, the first keyframe gives the aspect ratio.
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
        // A clip is never longer than the video it is a reference for.
        #[cfg(feature = "cuda")]
        let clips = files
            .into_iter()
            .map(|path| load_clip(Path::new(path), shape.frames))
            .collect::<Result<Vec<_>, _>>()?;
        #[cfg(feature = "metal")]
        let clips: Vec<ReferenceClip> = Vec::new();
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
        // Each reference keeps its aspect ratio, stretched to its size on the 32-pixel grid.
        let references = references
            .into_iter()
            .map(|picture| {
                let (reference_width, reference_height) = mmh3_core::picture::reference_size(
                    picture.width,
                    picture.height,
                    width,
                    height,
                );
                picture.resize(reference_width, reference_height)
            })
            .collect();
        Ok(Settings {
            shape,
            keyframes,
            references,
            sounds,
            clips,
            steps: schedule.steps(),
            schedule,
            seed: option_number(options, "seed", 0)? as u64,
            shift_video,
            shift_audio,
            workers: option_values(arguments, "worker")
                .iter()
                .map(|address| (*address).to_owned())
                .collect(),
            token: match options.get("token") {
                Some(path) => std::fs::read_to_string(path)?.trim().to_owned(),
                None => String::new(),
            },
        })
    }
}

/// Encodes the prompt, or reads the text states of `--context`, and runs the DiT's Euler steps
/// from the seed's noise. Returns the final video and audio latents.
pub fn sample(
    options: &HashMap<&str, &str>,
    settings: &Settings,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    use mmh3_core::dit::inputs::{DitInputs, Keyframe, Reference};
    use mmh3_core::dit::sampler::euler_step;
    use mmh3_core::generation::FPS;
    use mmh3_core::random::NormalSampler;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_core::vision::{PromptReference, VisionGrid};
    #[cfg(feature = "cuda")]
    use mmh3_cuda::text_encoder::CudaTextEncoder as TextEncoder;
    #[cfg(feature = "metal")]
    use mmh3_metal::text_encoder::MetalTextEncoder as TextEncoder;
    use std::time::Instant;

    let prompt = match (options.get("prompt"), options.get("prompt-file")) {
        (Some(prompt), None) => Some((*prompt).to_owned()),
        (None, Some(path)) => Some(std::fs::read_to_string(path)?),
        (None, None) => None,
        _ => return Err("pass either --prompt or --prompt-file".into()),
    };
    // The prompt holds the keyframes or the references, whichever the settings have.
    let pictures: Vec<Tensor> = settings
        .keyframes
        .iter()
        .map(|keyframe| &keyframe.picture)
        .chain(&settings.references)
        .map(|picture| picture.to_tensor())
        .collect();
    #[cfg(feature = "cuda")]
    let clip_blocks: Vec<Vec<Tensor>> = settings
        .clips
        .iter()
        .map(|clip| block_frames(&clip.frames, FPS))
        .collect();

    // The pictures come first, then the clips with their soundtracks, then the standalone
    // sounds, the order of the official presentation.
    let mut prompt_references: Vec<PromptReference> = pictures
        .iter()
        .map(|picture| {
            PromptReference::Picture(VisionGrid::for_picture(picture.shape[0], picture.shape[1]))
        })
        .collect();
    for clip in &settings.clips {
        if clip.sound.is_some() {
            prompt_references.push(PromptReference::Sound);
        }
        prompt_references.push(PromptReference::Clip {
            grid: VisionGrid::for_picture(clip.frames.shape[1], clip.frames.shape[2]),
            timestamps: block_seconds(clip.frames.shape[0], FPS),
        });
    }
    prompt_references.extend(settings.sounds.iter().map(|_| PromptReference::Sound));

    #[allow(unused_mut)]
    let mut context_modalities = Vec::new();
    let context = match (prompt, options.get("context")) {
        (Some(prompt), None) => {
            let started = Instant::now();
            // A worker that holds the text encoder spares this machine 27 GB and the load, but only
            // for a prompt of plain text: pictures and clips still go through the vision tower here.
            if prompt_references.is_empty()
                && let Some(context) = encode_on_worker(settings, &Tokenizer::h3().encode(&prompt))
            {
                println!(
                    "encoded {} prompt tokens on a worker in {:.1} s",
                    context.shape[0],
                    started.elapsed().as_secs_f64()
                );
                context
            } else {
                let path = option_path(options, "text-encoder", TEXT_ENCODER_FILE)?;
                let file = SafeTensors::open(Path::new(&path))?;
                let encoder = TextEncoder::load(&file)?;
                let context = if prompt_references.is_empty() {
                    let ids = Tokenizer::h3().encode(&prompt);
                    encoder.encode(&ids, &[])?.context
                } else {
                    #[cfg(feature = "metal")]
                    return Err("picture and sound prompts are not implemented on Metal yet".into());
                    #[cfg(feature = "cuda")]
                    {
                        use mmh3_core::vision::vision_prompt;
                        use mmh3_cuda::vision::CudaVisionEncoder;

                        // One embedding per vision block: the pictures, then the pairs of every clip.
                        let embeddings = if pictures.is_empty() && clip_blocks.is_empty() {
                            Vec::new()
                        } else {
                            let vision = CudaVisionEncoder::load(&file)?;
                            let mut embeddings = pictures
                                .iter()
                                .map(|picture| vision.encode(picture))
                                .collect::<Result<Vec<_>, _>>()?;
                            for blocks in &clip_blocks {
                                for pair in blocks.chunks(2) {
                                    embeddings.push(vision.encode_frames(&pair[0], &pair[1])?);
                                }
                            }
                            embeddings
                        };
                        let prompt = vision_prompt(&Tokenizer::h3(), &prompt, &prompt_references);
                        context_modalities = prompt.modalities.clone();
                        encoder.encode_prompt(&prompt, &embeddings, &[])?.context
                    }
                };
                println!(
                    "encoded {} prompt tokens with {} pictures, {} clips and {} sounds in {:.1} s",
                    context.shape[0],
                    pictures.len(),
                    settings.clips.len(),
                    settings.sounds.len(),
                    started.elapsed().as_secs_f64()
                );
                context
            }
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
    if options.contains_key("context") && !prompt_references.is_empty() {
        return Err("keyframes and references need a prompt, not --context".into());
    }
    #[cfg(feature = "cuda")]
    let latents = encode_pictures(options, settings.seed, &pictures)?;
    #[cfg(feature = "metal")]
    let latents: Vec<Tensor> = Vec::new();
    #[cfg(feature = "cuda")]
    let clip_latents = encode_clips(options, settings.seed, &settings.clips)?;
    #[cfg(feature = "metal")]
    let clip_latents: Vec<Tensor> = Vec::new();
    #[cfg(feature = "cuda")]
    let sound_latents = encode_sounds(options, &settings.sounds)?;
    #[cfg(feature = "metal")]
    let sound_latents: Vec<Tensor> = Vec::new();
    #[cfg(feature = "cuda")]
    let soundtrack_latents = encode_sounds(
        options,
        &settings
            .clips
            .iter()
            .filter_map(|clip| clip.sound.clone())
            .collect::<Vec<_>>(),
    )?;
    #[cfg(feature = "metal")]
    let soundtrack_latents: Vec<Tensor> = Vec::new();

    let has_references = !(settings.references.is_empty()
        && settings.sounds.is_empty()
        && settings.clips.is_empty());
    let (keyframes, references) = if !has_references {
        let keyframes = settings
            .keyframes
            .iter()
            .zip(latents)
            .map(|(keyframe, latent)| Keyframe {
                frame_index: keyframe.frame_index,
                video: Some(latent),
                audio: None,
            })
            .collect();
        (keyframes, Vec::new())
    } else {
        // The DiT takes the references in the order the prompt introduces them.
        let mut soundtracks = soundtrack_latents.into_iter();
        let references = latents
            .into_iter()
            .map(Reference::Picture)
            .chain(
                settings
                    .clips
                    .iter()
                    .zip(clip_latents)
                    .map(|(clip, video)| Reference::Video {
                        video,
                        audio: clip.sound.as_ref().and_then(|_| soundtracks.next()),
                    }),
            )
            .chain(sound_latents.into_iter().map(Reference::Audio))
            .collect();
        (Vec::new(), references)
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

    let dit_file = if references.is_empty() {
        DIT_FILE
    } else {
        REFERENCE_DIT_FILE
    };
    let dit = load_dit(options, "dit", dit_file)?;
    let sparse = sparse_attention(options, dit.has_vsa_gates())?;
    let schedule = &settings.schedule;
    // A machine that can take a share of every step, and a run whose shape one can be cut out of.
    #[cfg(feature = "cuda")]
    let mut target = shard_target(
        settings,
        options,
        &dit,
        sparse.as_ref(),
        &context,
        &context_modalities,
        &video,
        &audio,
        &keyframes,
        &references,
    );
    #[cfg(feature = "cuda")]
    let mut sharing = match &mut target {
        Some(target) => {
            let shard = target.shard.clone();
            let algorithms = target
                .open
                .first()
                .map_or(0, |open| open.algorithms.len() as u32);
            match crate::worker::open_shard(
                &mut target.workers,
                &target.open,
                &target.payload,
                &shard,
                target.tokens,
                dit.config().hidden,
                target.gated,
                algorithms,
            ) {
                Ok(exchanger) => Some(Sharing::With(Box::new(exchanger), shard)),
                Err(error) => {
                    eprintln!("warning: opening a shared run: {error}");
                    None
                }
            }
        }
        // `--shard-dit 1` runs the same path with nothing to carry anywhere, which tells the cost
        // of the machinery apart from the cost of the link.
        None => alone_shard(
            options,
            &dit,
            &context,
            &video,
            &audio,
            &keyframes,
            &references,
            settings,
        )
        .map(|shard| Sharing::Alone(mmh3_cuda::shard::WholeExchange::new(), shard)),
    };
    #[cfg(feature = "metal")]
    let prepared = dit.prepare_text(&context)?;
    for step in 0..steps {
        let started = Instant::now();
        let inputs = DitInputs {
            video: video.clone(),
            audio: audio.clone(),
            context: context.clone(),
            context_modalities: context_modalities.clone(),
            keyframes: keyframes.clone(),
            references: references.clone(),
            sigma: schedule.video[step],
            shift_video: settings.shift_video,
            shift_audio: settings.shift_audio,
        };
        let step_sparse = sparse.filter(|settings| settings.applies_to_step(step, steps));
        #[cfg(feature = "cuda")]
        let outputs = match &mut sharing {
            // Every rank runs the same step over its own rows, and the exchanges inside the blocks
            // keep the attention whole. A rank that fails takes the run with it: there is no
            // halfway through a step to fall back from.
            Some(Sharing::With(exchanger, shard)) => crate::worker::step_shard(
                exchanger,
                &dit,
                &inputs,
                step_sparse.as_ref(),
                shard,
                step,
            )?,
            Some(Sharing::Alone(exchange, shard)) => {
                let began = Instant::now();
                let mut context = mmh3_cuda::shard::ShardContext {
                    shard: shard.clone(),
                    exchange,
                    timing: Default::default(),
                };
                let outputs = dit.forward_shard(&inputs, step_sparse.as_ref(), &mut context)?;
                println!(
                    "  {}",
                    crate::worker::describe_timing(&context.timing, began.elapsed())
                );
                let part = outputs.part.ok_or("a shared step returned no rows")?;
                dit.assemble_velocity(&inputs, step_sparse.as_ref(), &[part])?
            }
            None => dit.forward(&inputs, &[], step_sparse.as_ref())?,
        };
        #[cfg(feature = "metal")]
        let outputs = prepared.forward(&inputs, &[], step_sparse.as_ref())?;
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
    #[cfg(feature = "cuda")]
    save_algorithm_cache();
    Ok((video, audio))
}

/// The posterior means of the reference sounds, `[latent channels, stereo channels, frames]` each.
/// They take no condition noise, as the reference pipeline leaves reference audio clean.
#[cfg(feature = "cuda")]
/// Asks the first worker that holds the text encoder, or returns `None` so the caller encodes here.
/// A worker that fails is reported and skipped: a generation never depends on one.
fn encode_on_worker(settings: &Settings, ids: &[u32]) -> Option<Tensor> {
    use crate::worker::{TEXT_ENCODER_ROLE as ROLE, Worker};
    use mmh3_core::worker::CAPABILITY_ENCODE_TEXT;

    for address in &settings.workers {
        let mut worker = match Worker::connect(address, &settings.token) {
            Ok(worker) => worker,
            Err(error) => {
                eprintln!("warning: worker {address}: {error}");
                continue;
            }
        };
        if !worker.serves(CAPABILITY_ENCODE_TEXT) || worker.checkpoint(ROLE).is_none() {
            continue;
        }
        match worker.encode_text(ROLE, ids) {
            Ok(context) => return Some(context),
            Err(error) => eprintln!("warning: worker {address}: {error}"),
        }
    }
    None
}

fn encode_sounds(
    options: &HashMap<&str, &str>,
    sounds: &[Tensor],
) -> Result<Vec<Tensor>, Box<dyn Error>> {
    use mmh3_core::audio::LATENT_RATE;
    use mmh3_cuda::audio_encoder::CudaAudioEncoder;
    use std::time::Instant;

    if sounds.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = option_path(options, "audio-vae", AUDIO_VAE_FILE)?;
    let encoder = CudaAudioEncoder::load(&SafeTensors::open(Path::new(&path))?, "")?;
    let latents = sounds
        .iter()
        .map(|waveform| encoder.encode(waveform))
        .collect::<Result<Vec<Tensor>, _>>()?;
    let frames: usize = latents.iter().map(|latent| latent.shape[2]).sum();
    println!(
        "encoded {} sounds, {:.2} s of audio, in {:.1} s",
        latents.len(),
        frames as f64 / LATENT_RATE as f64,
        started.elapsed().as_secs_f64()
    );
    Ok(latents)
}

/// The latents of the reference clips, sampled from the video VAE's posterior and mixed with the
/// condition noise, as the reference pictures are.
#[cfg(feature = "cuda")]
fn encode_clips(
    options: &HashMap<&str, &str>,
    seed: u64,
    clips: &[ReferenceClip],
) -> Result<Vec<Tensor>, Box<dyn Error>> {
    use mmh3_core::dit::timestep::VIDEO_CONDITION_TIMESTEP;
    use mmh3_core::random::NormalSampler;
    use mmh3_cuda::video_encoder::{
        CudaVideoEncoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE, Temporal,
    };
    use std::time::Instant;

    if clips.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = video_vae_path(options)?;
    let encoder = CudaVideoEncoder::load(
        &SafeTensors::open(Path::new(&path))?,
        Temporal::Clip,
        DEFAULT_TILE_SIZE,
        DEFAULT_TILE_OVERLAP_MIN,
    )?;
    let mut noise = NormalSampler::new(seed ^ KEYFRAME_NOISE_STREAM ^ CLIP_NOISE_STREAM);
    let latents = clips
        .iter()
        .map(|clip| {
            let posterior = encoder.encode_clip(&clip.frames)?;
            let count = posterior.mean.data.len();
            let mut latent = posterior
                .sampled_latent(&NormalSampler::new(KEYFRAME_POSTERIOR_SEED).samples(count));
            for (value, noise) in latent.data.iter_mut().zip(noise.samples(count)) {
                *value =
                    VIDEO_CONDITION_TIMESTEP * *value + (1.0 - VIDEO_CONDITION_TIMESTEP) * noise;
            }
            Ok(latent)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let frames: usize = clips.iter().map(|clip| clip.frames.shape[0]).sum();
    println!(
        "encoded {} clips, {frames} frames, with {path} in {:.1} s",
        latents.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(latents)
}

/// The latents of keyframes or reference pictures as the DiT sees them: a sample of the video VAE's
/// posterior for each picture with 0.1% of noise mixed in.
#[cfg(feature = "cuda")]
fn encode_pictures(
    options: &HashMap<&str, &str>,
    seed: u64,
    pictures: &[Tensor],
) -> Result<Vec<Tensor>, Box<dyn Error>> {
    use mmh3_core::dit::timestep::VIDEO_CONDITION_TIMESTEP;
    use mmh3_core::random::NormalSampler;
    use mmh3_cuda::video_encoder::{
        CudaVideoEncoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE, Temporal,
    };
    use std::time::Instant;

    if pictures.is_empty() {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let path = video_vae_path(options)?;
    let encoder = CudaVideoEncoder::load(
        &SafeTensors::open(Path::new(&path))?,
        Temporal::Frame,
        DEFAULT_TILE_SIZE,
        DEFAULT_TILE_OVERLAP_MIN,
    )?;
    let mut noise = NormalSampler::new(seed ^ KEYFRAME_NOISE_STREAM);
    let latents = pictures
        .iter()
        .map(|picture| {
            let posterior = encoder.encode_picture(picture)?;
            let count = posterior.mean.data.len();
            let mut latent = posterior
                .sampled_latent(&NormalSampler::new(KEYFRAME_POSTERIOR_SEED).samples(count));
            for (value, noise) in latent.data.iter_mut().zip(noise.samples(count)) {
                *value =
                    VIDEO_CONDITION_TIMESTEP * *value + (1.0 - VIDEO_CONDITION_TIMESTEP) * noise;
            }
            Ok(latent)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    println!(
        "encoded {} pictures with {path} in {:.1} s",
        latents.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(latents)
}

/// A worker that will take a share of every step, ready to open. Held apart from the exchange
/// because the exchange borrows it for the whole run.
#[cfg(feature = "cuda")]
struct ShardTarget {
    workers: Vec<crate::worker::Worker>,
    shard: mmh3_cuda::shard::Shard,
    tokens: usize,
    gated: bool,
    open: Vec<mmh3_core::worker::OpenSession>,
    payload: Vec<u8>,
}

/// The first worker that can take a share of the DiT, or `None` to keep the whole step here.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn shard_target(
    settings: &Settings,
    options: &HashMap<&str, &str>,
    dit: &mmh3_cuda::dit::CudaDit,
    sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
    context: &Tensor,
    context_modalities: &[mmh3_core::dit::timestep::Modality],
    video: &Tensor,
    audio: &Tensor,
    keyframes: &[mmh3_core::dit::inputs::Keyframe],
    references: &[mmh3_core::dit::inputs::Reference],
) -> Option<ShardTarget> {
    use crate::worker::{Worker, digest, session_conditions, session_payload};
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::layout::PackedLayout;
    use mmh3_core::worker::{CAPABILITY_DIT_SHARD, Checkpoint, OpenSession};
    use mmh3_cuda::shard::Shard;

    // A worker that only encodes the prompt and decodes some chunks is worth 9% of a run, which is
    // not worth a second machine. Sharing every step is worth 35%, so it is what `--worker` does
    // unless `--shard-dit 0` says otherwise.
    let asked = options.contains_key("shard-dit");
    // Without `--shard-dit`, every worker that can take a share does: naming a machine is what
    // asks for it. With it, the number is a cap on the ranks rather than a promise of them.
    let ranks = crate::cli::option_number(options, "shard-dit", settings.workers.len() + 1).ok()?;
    if ranks < 2 {
        // Zero keeps the DiT here. One shares a step with nobody, which `alone_shard` handles.
        return None;
    }
    if settings.workers.is_empty() {
        if asked {
            println!("sharing a step needs a worker, so the DiT stays here");
        }
        return None;
    }
    let file = if references.is_empty() {
        crate::models::DIT_FILE
    } else {
        crate::models::REFERENCE_DIT_FILE
    };
    let path = crate::models::option_path(options, "dit", file).ok()?;
    // The LoRAs and patches this run added, in the order `load_dit` adds them. A worker that
    // cannot find one refuses the session rather than running a different DiT.
    let adapters = match session_adapters(options) {
        Ok(adapters) => adapters,
        Err(error) => {
            eprintln!("warning: naming the adapters for a worker: {error}");
            return None;
        }
    };
    let file = mmh3_core::safetensors::SafeTensors::open(std::path::Path::new(&path)).ok()?;
    let checkpoint = Checkpoint {
        role: "dit.h3".to_owned(),
        digest: digest(&file),
    };
    drop(file);

    // The workers that can take a share, in the order they were given, up to the rank count.
    let mut workers = Vec::new();
    for address in &settings.workers {
        if workers.len() + 1 >= ranks {
            break;
        }
        let worker = match Worker::connect(address, &settings.token) {
            Ok(worker) => worker,
            Err(error) => {
                eprintln!("warning: worker {address}: {error}");
                continue;
            }
        };
        if !worker.serves(CAPABILITY_DIT_SHARD) || !worker.reads_remotely() {
            continue;
        }
        workers.push(worker);
    }
    if workers.is_empty() {
        if asked {
            println!("no worker can take a share of a step, so the DiT stays here");
        }
        return None;
    }
    // A rank that cannot be filled is one the run does without: the ranks are what the machines
    // that answered add up to, not what `--shard-dit` asked for.
    let ranks = workers.len() + 1;
    let inputs = DitInputs {
        video: video.clone(),
        audio: audio.clone(),
        context: context.clone(),
        context_modalities: context_modalities.to_vec(),
        keyframes: keyframes.to_vec(),
        references: references.to_vec(),
        sigma: 1.0,
        shift_video: settings.shift_video,
        shift_audio: settings.shift_audio,
    };
    let tokens = PackedLayout::for_inputs(&inputs).len();
    let shape = |dimensions: &[usize]| -> Vec<u32> {
        dimensions.iter().map(|value| *value as u32).collect()
    };
    // The workers run the leader's choices rather than timing the candidates themselves, since a
    // rank that measures while the wire and the others have its device measures the load.
    let algorithm_key = mmh3_cuda::algorithms::key().unwrap_or_default();
    let algorithms = mmh3_cuda::algorithms::chosen_matmul();
    let open = |rank: usize| -> Option<OpenSession> {
        Some(OpenSession {
            checkpoint: checkpoint.clone(),
            ranks: ranks as u32,
            rank: rank as u32,
            video_shape: shape(&video.shape).try_into().ok()?,
            audio_shape: shape(&audio.shape).try_into().ok()?,
            context_shape: shape(&context.shape).try_into().ok()?,
            shift_video: settings.shift_video,
            shift_audio: settings.shift_audio,
            steps: settings.steps as u32,
            sparse: crate::worker::sparse_settings(sparse),
            conditions: session_conditions(keyframes, references),
            adapters: adapters.clone(),
            algorithm_key: algorithm_key.clone(),
            algorithms: algorithms.clone(),
            precision: match dit.attention_precision() {
                mmh3_cuda::attention::AttentionPrecision::Int8Fp8 => 1,
                _ => 0,
            },
        })
    };
    let open: Option<Vec<OpenSession>> = (1..ranks).map(open).collect();
    let open = open?;
    println!(
        "{} takes a share of every step, {tokens} tokens split {ranks} ways",
        workers
            .iter()
            .map(|worker| worker.address.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Some(ShardTarget {
        shard: Shard::even(0, ranks, tokens, dit.config().heads, 1),
        workers,
        tokens,
        gated: dit.has_vsa_gates(),
        open,
        payload: session_payload(context, context_modalities, keyframes, references),
    })
}

/// The LoRAs and patches of `--patch` and `--lora`, as a session names them: by the digest of the
/// file, since the machines name the files differently.
#[cfg(feature = "cuda")]
fn session_adapters(
    options: &HashMap<&str, &str>,
) -> Result<Vec<mmh3_core::worker::Adapter>, Box<dyn Error>> {
    use crate::models::model_file;
    use mmh3_core::worker::Adapter;

    let mut adapters = Vec::new();
    // A patch applies as adapters at full strength, before the LoRA.
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
    for (path, strength, mode) in wanted {
        let Some(path) = path else {
            continue;
        };
        adapters.push(Adapter {
            digest: crate::worker::digest(&SafeTensors::open(Path::new(&path))?),
            strength,
            mode: if mode == "merge" {
                Adapter::MERGE
            } else {
                Adapter::ADAPTER
            },
        });
    }
    Ok(adapters)
}

/// How a step is shared out, which is either with another machine or, for `--shard-dit 1`, with
/// nobody at all so that the machinery can be timed on its own.
#[cfg(feature = "cuda")]
enum Sharing<'a> {
    With(Box<crate::worker::Exchanger<'a>>, mmh3_cuda::shard::Shard),
    Alone(mmh3_cuda::shard::WholeExchange, mmh3_cuda::shard::Shard),
}

/// The shard of a run that shares a step with nobody. One rank covers the whole sequence and every
/// head, so the result is the whole step's, and what is left is the gathers and the waits.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn alone_shard(
    options: &HashMap<&str, &str>,
    dit: &mmh3_cuda::dit::CudaDit,
    context: &Tensor,
    video: &Tensor,
    audio: &Tensor,
    keyframes: &[mmh3_core::dit::inputs::Keyframe],
    references: &[mmh3_core::dit::inputs::Reference],
    settings: &Settings,
) -> Option<mmh3_cuda::shard::Shard> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::layout::PackedLayout;

    if crate::cli::option_number(options, "shard-dit", 0).ok()? != 1 {
        return None;
    }
    let inputs = DitInputs {
        video: video.clone(),
        audio: audio.clone(),
        context: context.clone(),
        context_modalities: Vec::new(),
        keyframes: keyframes.to_vec(),
        references: references.to_vec(),
        sigma: 1.0,
        shift_video: settings.shift_video,
        shift_audio: settings.shift_audio,
    };
    let tokens = PackedLayout::for_inputs(&inputs).len();
    println!("sharing every step with nobody, {tokens} tokens in one piece");
    Some(mmh3_cuda::shard::Shard::even(
        0,
        1,
        tokens,
        dit.config().heads,
        1,
    ))
}
