//! Sampling shared by `mmh3 generate` and `mmh3-tools latent`: the generation settings, the
//! prompt's text states and the DiT's Euler steps from the seed's noise, with the keyframes of
//! first and last frame generation or the reference pictures of reference to video generation.

use crate::audio::load_audio;
use crate::cli::{option_float, option_number, option_values};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::models::load_dit;
#[cfg(feature = "cuda")]
use crate::models::{AUDIO_VAE_FILE, save_algorithm_cache, video_vae_path};
use crate::models::{DIT_FILE, REFERENCE_DIT_FILE, sparse_attention};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::models::{TEXT_ENCODER_FILE, option_path};
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
    "worker-units",
    "token",
    "attention",
    "attention-precision",
    "linear-precision",
    "sparse-tau",
    "sparse-start",
    "vsa-sparsity",
    "vram-budget",
];

/// The options that stand alone, taking nothing after them.
pub const FLAGS: &[&str] = &["local-worker", "consistent"];

/// A machine this run may borrow, and what it may be asked for.
pub struct Machine {
    pub address: String,
    /// The capabilities this run will ask it for, which is every one it serves unless
    /// `--worker-units` named fewer.
    pub units: u32,
}

/// Every unit there is, for a machine that was named with no units of its own.
fn every_unit() -> u32 {
    use mmh3_core::worker::{
        CAPABILITY_DECODE_AUDIO, CAPABILITY_DECODE_VIDEO, CAPABILITY_DIT_SHARD,
        CAPABILITY_ENCODE_TEXT,
    };

    CAPABILITY_DIT_SHARD
        | CAPABILITY_ENCODE_TEXT
        | CAPABILITY_DECODE_VIDEO
        | CAPABILITY_DECODE_AUDIO
}

/// The units of one `--worker-units`, as the capabilities a worker announces.
fn units(text: &str) -> Result<u32, Box<dyn Error>> {
    use mmh3_core::worker::{
        CAPABILITY_DECODE_AUDIO, CAPABILITY_DECODE_VIDEO, CAPABILITY_DIT_SHARD,
        CAPABILITY_ENCODE_TEXT,
    };

    text.split(',')
        .try_fold(0, |units, unit| -> Result<u32, Box<dyn Error>> {
            Ok(units
                | match unit.trim() {
                    "steps" => CAPABILITY_DIT_SHARD,
                    "prompt" => CAPABILITY_ENCODE_TEXT,
                    "video" => CAPABILITY_DECODE_VIDEO,
                    "audio" => CAPABILITY_DECODE_AUDIO,
                    other => {
                        return Err(format!(
                            "--worker-units takes steps, prompt, video or audio, not {other}"
                        )
                        .into());
                    }
                })
        })
}

/// How far the generation running here has got, and how long its last step took.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    pub step: usize,
    pub steps: usize,
    pub seconds: f64,
}

/// What the run here has reached. There is one device and so one run at a time, which is what
/// makes one of these enough to say where it is.
static PROGRESS: std::sync::Mutex<Option<Progress>> = std::sync::Mutex::new(None);

/// Where the generation running here has got to, or None before one has taken a step.
pub fn progress() -> Option<Progress> {
    *PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn reached(progress: Progress) {
    *PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(progress);
}

/// Forgets where the run before this one had got to, so that a run is not reported at a step it
/// has not taken.
pub fn starting() {
    *PROGRESS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

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
    /// The machines of `--worker` and `--local-worker`, in the order they were given, which is
    /// the order their ranks are numbered in.
    pub workers: Vec<Machine>,
    /// The shared secret of `--token`, empty when there is none.
    pub token: String,
}

impl Settings {
    /// The same as `workers_for`, for a caller that keeps the addresses.
    pub fn workers_owned(&self, unit: u32) -> Vec<String> {
        self.workers_for(unit)
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// The machines this run may ask for `unit`, in the order they were given.
    pub fn workers_for(&self, unit: u32) -> Vec<&str> {
        self.workers
            .iter()
            .filter(|machine| machine.units & unit != 0)
            .map(|machine| machine.address.as_str())
            .collect()
    }

    /// The machines of `--worker` and `--local-worker`, with the units each may be asked for. A
    /// `--local-worker` is started here, so that this machine's own GPU takes part as a worker
    /// rather than as the leader: a leader reads no model.
    fn machines(
        options: &HashMap<&str, &str>,
        arguments: &[String],
        token: &str,
    ) -> Result<Vec<Machine>, Box<dyn Error>> {
        let mut machines = Vec::new();
        for borrowed in crate::cli::borrowed(arguments)? {
            let units = match borrowed.units {
                Some(text) => units(text)?,
                None => every_unit(),
            };
            let address = match borrowed.address {
                Some(address) => address.to_owned(),
                None => Self::worker_here(options, token)?,
            };
            machines.push(Machine { address, units });
        }
        Ok(machines)
    }

    /// The address of a worker started in this process for `--local-worker`.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn worker_here(options: &HashMap<&str, &str>, token: &str) -> Result<String, Box<dyn Error>> {
        let models = crate::models::models_directory(options)
            .ok_or("--local-worker needs a models directory, from --models DIR or MMH3_MODELS")?;
        crate::worker::worker_here(&models, token)
            .ok_or_else(|| "no worker could start on this machine".into())
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    fn worker_here(_options: &HashMap<&str, &str>, _token: &str) -> Result<String, Box<dyn Error>> {
        Err("this build runs nothing of its own, so it has no --local-worker to give".into())
    }

    /// The canvas, step schedule, seed and schedule shifts of the options, with their defaults.
    /// `arguments` gives the repeated `--reference` and `--worker` options.
    pub fn parse(
        options: &HashMap<&str, &str>,
        arguments: &[String],
    ) -> Result<Self, Box<dyn Error>> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        crate::resident::take_budget(options)?;
        crate::worker::set_consistent(options.contains_key("consistent"));
        let token = match options.get("token") {
            Some(path) => std::fs::read_to_string(path)?.trim().to_owned(),
            None => String::new(),
        };
        let workers = Self::machines(options, arguments, &token)?;
        #[cfg(feature = "metal")]
        {
            crate::metal::validate_options(options)?;
            // A run with no worker to hand a step to runs every one here, so what this machine
            // cannot run it can refuse now rather than after encoding a prompt. A run that means
            // to hand them out is judged by `load_dit`, which is reached only if it ends up
            // running them here after all.
            if workers
                .iter()
                .all(|machine| machine.units & mmh3_core::worker::CAPABILITY_DIT_SHARD == 0)
            {
                crate::metal::validate_step_options(options)?;
            }
        }
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
        #[cfg(not(feature = "cuda"))]
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
            workers,
            token,
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

    // Which DiT a run uses is the settings' business and not the prompt's, so the workers can be
    // told to read it, and this machine can read its own, while the prompt is still encoding.
    let has_references = !(settings.references.is_empty()
        && settings.sounds.is_empty()
        && settings.clips.is_empty());
    let dit_file = if has_references {
        REFERENCE_DIT_FILE
    } else {
        DIT_FILE
    };
    // A prompt of plain text goes to a worker, which reads nothing ahead for it.
    let prompt_on_worker =
        prompt.is_some() && options.get("context").is_none() && prompt_references.is_empty();
    let prepared_workers = prepare_workers(settings, options, dit_file, prompt_on_worker);

    let encode_context = || -> Result<(Tensor, Vec<mmh3_core::dit::timestep::Modality>), String> {
        #[allow(unused_mut)]
        let mut context_modalities = Vec::new();
        let encode = || -> Result<Tensor, Box<dyn Error>> {
            Ok(match (prompt, options.get("context")) {
                (Some(prompt), None) => {
                    let started = Instant::now();
                    // A worker that holds the text encoder spares this machine 27 GB and the load, but only
                    // for a prompt of plain text: pictures and clips still go through the vision tower here.
                    if prompt_references.is_empty()
                        && let Some(context) =
                            encode_on_worker(settings, &Tokenizer::h3().encode(&prompt))?
                    {
                        println!(
                            "encoded {} prompt tokens on a worker in {:.1} s",
                            context.shape[0],
                            started.elapsed().as_secs_f64()
                        );
                        context
                    } else {
                        // The prompt is the one thing a machine that runs no block still cannot do
                        // for itself, so a build with no backend asks for a worker by name.
                        #[cfg(not(any(feature = "cuda", feature = "metal")))]
                        return Err(
                            "this build encodes no prompt of its own: name a worker that \
                                    holds the text encoder with --worker"
                                .into(),
                        );
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        let path = option_path(options, "text-encoder", TEXT_ENCODER_FILE)?;
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        let file = SafeTensors::open(Path::new(&path))?;
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        let encoder = TextEncoder::load_fitting(&file)?;
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        let context = if prompt_references.is_empty() {
                            let ids = Tokenizer::h3().encode(&prompt);
                            encoder.encode(&ids, &[])?.context
                        } else {
                            #[cfg(feature = "metal")]
                            return Err(
                                "picture and sound prompts are not implemented on Metal yet".into(),
                            );
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
                                            embeddings
                                                .push(vision.encode_frames(&pair[0], &pair[1])?);
                                        }
                                    }
                                    embeddings
                                };
                                let prompt =
                                    vision_prompt(&Tokenizer::h3(), &prompt, &prompt_references);
                                context_modalities = prompt.modalities.clone();
                                encoder.encode_prompt(&prompt, &embeddings, &[])?.context
                            }
                        };
                        #[cfg(any(feature = "cuda", feature = "metal"))]
                        println!(
                            "encoded {} prompt tokens with {} pictures, {} clips and {} sounds in {:.1} s",
                            context.shape[0],
                            pictures.len(),
                            settings.clips.len(),
                            settings.sounds.len(),
                            started.elapsed().as_secs_f64()
                        );
                        #[cfg(any(feature = "cuda", feature = "metal"))]
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
            })
        };
        let context = encode().map_err(|error| error.to_string())?;
        Ok((context, context_modalities))
    };
    // A run that hands every step to its workers needs the DiT's shape and none of the weights
    // behind it. Opening a checkpoint maps the file, so this costs the header pages and nothing.
    let (config, gated) = dit_shape(options, dit_file)?;
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let hands_out = !prepared_workers.is_empty();
    // The prompt is encoded before the DiT is read, so the text encoder is gone by the time the
    // DiT arrives and the device holds one checkpoint at a time.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    let (context, context_modalities, mut dit) = {
        let (context, modalities) =
            encode_context().map_err(|error| -> Box<dyn Error> { error.into() })?;
        let dit = match hands_out {
            true => None,
            false => Some(load_dit(options, "dit", dit_file)?),
        };
        (context, modalities, dit)
    };
    // A build with no backend never holds a DiT. `Infallible` says so in the type: there is no
    // value this could be, which is what makes every arm that would use one unreachable here.
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let (context, context_modalities, _dit) = {
        let (context, modalities) =
            encode_context().map_err(|error| -> Box<dyn Error> { error.into() })?;
        let _dit: Option<std::convert::Infallible> = None;
        (context, modalities, _dit)
    };
    if options.contains_key("context") && !prompt_references.is_empty() {
        return Err("keyframes and references need a prompt, not --context".into());
    }
    #[cfg(feature = "cuda")]
    let latents = encode_pictures(options, settings.seed, &pictures)?;
    #[cfg(not(feature = "cuda"))]
    let latents: Vec<Tensor> = Vec::new();
    #[cfg(feature = "cuda")]
    let clip_latents = encode_clips(options, settings.seed, &settings.clips)?;
    #[cfg(not(feature = "cuda"))]
    let clip_latents: Vec<Tensor> = Vec::new();
    #[cfg(feature = "cuda")]
    let sound_latents = encode_sounds(options, &settings.sounds)?;
    #[cfg(not(feature = "cuda"))]
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
    #[cfg(not(feature = "cuda"))]
    let soundtrack_latents: Vec<Tensor> = Vec::new();

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

    let sparse = sparse_attention(options, gated)?;
    let schedule = &settings.schedule;
    // A machine that can take a share of every step, and a run whose shape one can be cut out of.
    let mut target = shard_target(
        settings,
        options,
        &config,
        sparse.as_ref(),
        &context,
        &context_modalities,
        &video,
        &audio,
        &keyframes,
        &references,
        prepared_workers,
    );
    let mut sharing = match &mut target {
        Some(target) => {
            match crate::worker::Coordinator::open(
                &mut target.workers,
                &target.open,
                &target.payload,
            ) {
                Ok(coordinator) => Some(Sharing::With(Box::new(coordinator))),
                Err(error) => {
                    eprintln!("warning: opening a shared run: {error}");
                    None
                }
            }
        }
        None => None,
    };
    // A run that meant to hand every step out and could not takes the steps itself, which is the
    // fallback everything else a worker does already has. It pays the load it had skipped.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if dit.is_none() && !matches!(sharing, Some(Sharing::With(_))) {
        dit = Some(load_dit(options, "dit", dit_file)?);
    }
    // Nothing may load a DiT past this point: a run that was going to has done it by now.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[allow(unused_variables)]
    let dit = dit;
    #[cfg(feature = "metal")]
    let prepared = match &dit {
        Some(dit) => Some(dit.prepare_text(&context)?),
        None => None,
    };
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
        // The velocity of this step and, where a whole one ran here, what Sol-Attn routed.
        let (velocity, routed): (mmh3_core::shard::Velocity, Option<f64>) = match &mut sharing {
            // Every rank runs the same step over its own rows, and the exchanges inside the blocks
            // keep the attention whole. A rank that fails takes the run with it: there is no
            // halfway through a step to fall back from.
            Some(Sharing::With(coordinator)) => {
                let velocity = crate::worker::step_shard(
                    coordinator,
                    &config,
                    &inputs,
                    step_sparse.as_ref(),
                    step,
                )?;
                (velocity, None)
            }
            // A run that shares a step with nobody takes the whole one, which on Metal goes
            // through the text this run refined once rather than through the DiT directly.
            None => {
                #[cfg(not(any(feature = "cuda", feature = "metal")))]
                {
                    return Err(
                        "this build runs no step of its own: name a worker with --worker".into(),
                    );
                }
                #[cfg(any(feature = "cuda", feature = "metal"))]
                {
                    #[cfg(feature = "cuda")]
                    let outputs = {
                        let dit = dit
                            .as_ref()
                            .ok_or("a step to take here and no DiT to take it")?;
                        dit.forward(&inputs, &[], step_sparse.as_ref())?
                    };
                    #[cfg(feature = "metal")]
                    let outputs = {
                        let prepared = prepared
                            .as_ref()
                            .ok_or("a step to take here and no DiT to take it")?;
                        prepared.forward(&inputs, &[], step_sparse.as_ref())?
                    };
                    (whole_velocity(&outputs), outputs.routed_fraction)
                }
            }
        };
        euler_step(
            &mut video.data,
            &velocity.video,
            schedule.video[step],
            schedule.video[step + 1],
        );
        euler_step(
            &mut audio.data,
            &velocity.audio,
            schedule.audio[step],
            schedule.audio[step + 1],
        );
        let routing = routed.map_or(String::new(), |fraction| {
            format!(", Sol-Attn routed {:.1}%", 100.0 * fraction)
        });
        println!(
            "step {}/{steps} at sigma {:.4} in {:.1} s{routing}",
            step + 1,
            schedule.video[step],
            started.elapsed().as_secs_f64()
        );
        reached(Progress {
            step: step + 1,
            steps,
            seconds: started.elapsed().as_secs_f64(),
        });
    }
    #[cfg(feature = "cuda")]
    save_algorithm_cache();
    Ok((video, audio))
}

/// Asks the first worker that holds the text encoder, or returns `None` so the caller encodes here.
/// A worker that fails is reported and skipped: a generation never depends on one.
///
/// Nothing in this is a backend's. It asked on one of them only because the client it uses was
/// that backend's once, and a machine with the smaller memory is the one that most wants the text
/// encoder to load somewhere else.
fn encode_on_worker(settings: &Settings, ids: &[u32]) -> Result<Option<Tensor>, Box<dyn Error>> {
    use crate::worker::{TEXT_ENCODER_ROLE as ROLE, Worker};
    use mmh3_core::worker::CAPABILITY_ENCODE_TEXT;

    for address in settings.workers_for(CAPABILITY_ENCODE_TEXT) {
        let mut worker = match Worker::connect(address, &settings.token) {
            Ok(worker) => worker,
            // A machine that cannot be reached at all offered nothing, so the next one is asked.
            Err(error) => {
                eprintln!("warning: worker {address}: {error}");
                continue;
            }
        };
        if !worker.serves(CAPABILITY_ENCODE_TEXT) || worker.checkpoint(ROLE).is_none() {
            continue;
        }
        return worker
            .encode_text(ROLE, ids)
            .map(Some)
            .map_err(|error| format!("the prompt on {address}: {error}").into());
    }
    Ok(None)
}

/// The posterior means of the reference sounds, `[latent channels, stereo channels, frames]` each.
/// They take no condition noise, as the reference pipeline leaves reference audio clean.
#[cfg(feature = "cuda")]
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

/// The velocity a whole step answered with, in the words a shared one answers in, so that the
/// step loop reads one thing whichever way the step was taken.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn whole_velocity(outputs: &crate::worker::DitStep) -> mmh3_core::shard::Velocity {
    mmh3_core::shard::Velocity {
        video: outputs.video.clone(),
        audio: outputs.audio.clone(),
    }
}

/// The DiT's shape and whether it carries VSA gates, from the checkpoint headers alone. A patch
/// may add the gates, so the patch's header is read beside the DiT's: the shape a leader hands out
/// has to be the shape its workers will load, adapters and all.
fn dit_shape(
    options: &HashMap<&str, &str>,
    dit_file: &str,
) -> Result<(mmh3_core::dit::config::DitConfig, bool), Box<dyn Error>> {
    use mmh3_core::dit::config::DitConfig;
    use mmh3_core::safetensors::SafeTensors;

    let path = crate::models::option_path(options, "dit", dit_file)?;
    let file = SafeTensors::open(std::path::Path::new(&path))?;
    let config = DitConfig::of(&file, "").map_err(|error| format!("reading {path}: {error}"))?;
    let mut gated = DitConfig::gated(&file, "");
    for name in ["patch", "lora"] {
        if let Some(path) = crate::models::model_file(options, name, &["patches", "loras"])?
            && let Ok(file) = SafeTensors::open(std::path::Path::new(&path))
        {
            // A patch names the DiT's tensors under the prefix the backends strip as they apply
            // one, so its gates are only there.
            gated = gated || DitConfig::gated(&file, "diffusion_model.");
        }
    }
    Ok((config, gated))
}

/// A worker that will take a share of every step, ready to open. Held apart from the exchange
/// because the exchange borrows it for the whole run.
struct ShardTarget {
    workers: Vec<crate::worker::Worker>,
    open: Vec<mmh3_core::worker::OpenSession>,
    payload: Vec<u8>,
}

/// Connects to the workers a shared-out run would use and names the DiT it will share, so that
/// they read it while this machine still has a prompt to encode. The connections are the ones the
/// session opens on, since what a worker reads early is only there for the connection it came on.
///
/// The worker that encodes the prompt, when `prompt_on_worker` says one does, is not told. Its
/// reads take turns, so the DiT read ahead would only hold the prompt up, and on a card without
/// the room for both it would be let go for the text encoder and read again for the steps.
fn prepare_workers(
    settings: &Settings,
    options: &HashMap<&str, &str>,
    dit_file: &str,
    prompt_on_worker: bool,
) -> Vec<crate::worker::Worker> {
    use crate::worker::{TEXT_ENCODER_ROLE, Worker, digest};
    use mmh3_core::worker::{CAPABILITY_DIT_SHARD, CAPABILITY_ENCODE_TEXT, Checkpoint};

    let ranks = settings.workers_for(CAPABILITY_DIT_SHARD);
    if ranks.is_empty() {
        return Vec::new();
    }
    let Ok(path) = crate::models::option_path(options, "dit", dit_file) else {
        return Vec::new();
    };
    let checkpoint = match mmh3_core::safetensors::SafeTensors::open(std::path::Path::new(&path)) {
        Ok(file) => Checkpoint {
            role: "dit.h3".to_owned(),
            digest: digest(&file),
        },
        Err(error) => {
            eprintln!("warning: reading {path}: {error}");
            return Vec::new();
        }
    };
    // The machine `encode_on_worker` asks first, which is the one that encodes.
    let encoder = prompt_on_worker
        .then(|| {
            settings
                .workers_for(CAPABILITY_ENCODE_TEXT)
                .first()
                .copied()
        })
        .flatten();
    let mut workers = Vec::new();
    for address in ranks {
        let mut worker = match Worker::connect(address, &settings.token) {
            Ok(worker) => worker,
            Err(error) => {
                eprintln!("warning: worker {address}: {error}");
                continue;
            }
        };
        if !worker.serves(CAPABILITY_DIT_SHARD) {
            continue;
        }
        let encodes = encoder == Some(address)
            && worker.serves(CAPABILITY_ENCODE_TEXT)
            && worker.checkpoint(TEXT_ENCODER_ROLE).is_some();
        if encodes {
            workers.push(worker);
            continue;
        }
        if let Err(error) = worker.prepare(&checkpoint) {
            eprintln!("warning: worker {address}: {error}");
            continue;
        }
        workers.push(worker);
    }
    workers
}

/// The first worker that can take a share of the DiT, or `None` to keep the whole step here.
#[allow(clippy::too_many_arguments)]
fn shard_target(
    settings: &Settings,
    options: &HashMap<&str, &str>,
    config: &mmh3_core::dit::config::DitConfig,
    sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
    context: &Tensor,
    context_modalities: &[mmh3_core::dit::timestep::Modality],
    video: &Tensor,
    audio: &Tensor,
    keyframes: &[mmh3_core::dit::inputs::Keyframe],
    references: &[mmh3_core::dit::inputs::Reference],
    workers: Vec<crate::worker::Worker>,
) -> Option<ShardTarget> {
    use crate::worker::{digest, session_conditions, session_payload};
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::layout::PackedLayout;
    use mmh3_core::shard::Shard;
    use mmh3_core::worker::{CAPABILITY_DIT_SHARD, Checkpoint, OpenSession};

    // A machine that only encodes the prompt and decodes some chunks is worth 9% of a run, which
    // is not worth a second machine. Sharing every step is worth 35%, so a machine named with no
    // units of its own takes a rank as well.
    if settings.workers_for(CAPABILITY_DIT_SHARD).is_empty() {
        // Nobody may be asked for a step, so every one of them runs here.
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

    // The workers `prepare_workers` reached, which are already reading this DiT.
    let mut workers = workers;
    // What each rank says it can do. This machine is not one of them: it hands the work out and
    // puts the answers together, and runs no block of its own.
    let mut measured: Vec<Option<mmh3_core::worker::Speed>> =
        workers.iter().map(|worker| worker.speed()).collect();
    // A machine that says nothing cannot be given a share in proportion to it. Where some
    // measured and some did not, the ones that did not take no share of a step rather than an
    // equal one on no evidence: a share too large for a machine holds every other rank at every
    // barrier of every block. They are still asked for a prompt and for chunks. Where none
    // measured, which is a cluster whose backend cannot measure itself, the shares stay even,
    // since machines that all say nothing are most likely alike.
    if measured.iter().any(Option::is_some) && measured.iter().any(Option::is_none) {
        let refused: Vec<&str> = workers
            .iter()
            .filter(|worker| worker.speed().is_none())
            .map(|worker| worker.address.as_str())
            .collect();
        if !refused.is_empty() {
            println!(
                "{} takes no share of a step, since it measures nothing to cut one by",
                refused.join(", ")
            );
        }
        let mut keep = measured.iter().map(Option::is_some).collect::<Vec<bool>>();
        keep.reverse();
        workers.retain(|_| keep.pop().unwrap_or(false));
        measured.retain(Option::is_some);
    }
    if workers.is_empty() {
        println!("no worker can take a share of a step, so the DiT stays here");
        return None;
    }
    // A rank that cannot be filled is one the run does without: the ranks are what the machines
    // that answered add up to, not what the arguments named.
    let ranks = workers.len();
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
    // A block is two kinds of work and a machine is not equally good at both. The projections
    // and the MLP follow a rank's tokens, and attention follows its heads, since a rank attends
    // its own heads over the whole sequence however few tokens it carries. So the two axes are
    // cut by their own measurements. A machine that measured no attention is priced by its
    // product on both, which is what every cut did before.
    let products: Vec<f64> = measured
        .iter()
        .map(|speed| speed.map_or(1.0, |speed| speed.gemm_tops as f64))
        .collect();
    let attentions: Vec<f64> = measured
        .iter()
        .zip(&products)
        .map(
            |(speed, product)| match speed.map(|speed| speed.attention_tops as f64) {
                Some(attention) if attention > 0.0 => attention,
                _ => *product,
            },
        )
        .collect();
    let shard = if measured.iter().all(Option::is_some) && ranks > 0 {
        let say = |what: &str, weights: &[f64]| {
            format!(
                "{what} {}",
                weights
                    .iter()
                    .map(|weight| format!("{weight:.0}"))
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        };
        println!(
            "the shares follow what the machines measured: {}, {} TOPS",
            say("tokens by", &products),
            say("heads by", &attentions)
        );
        Shard::weighted(0, &products, &attentions, tokens, config.heads, 1)
    } else {
        println!("every machine takes the same share, since none of them measures itself");
        Shard::even(0, ranks, tokens, config.heads, 1)
    };
    let spans: Vec<mmh3_core::worker::ShardSpan> = shard
        .tokens
        .iter()
        .zip(&shard.heads)
        .map(|(tokens, heads)| mmh3_core::worker::ShardSpan {
            tokens: [tokens.start as u32, tokens.end as u32],
            heads: [heads.start as u32, heads.end as u32],
        })
        .collect();
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
            shard: spans.clone(),
            precision: attention_precision(options),
        })
    };
    let open: Option<Vec<OpenSession>> = (0..ranks).map(open).collect();
    let open = open?;
    println!(
        "{} takes every step, {tokens} tokens split {ranks} ways",
        workers
            .iter()
            .map(|worker| worker.address.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Some(ShardTarget {
        workers,
        open,
        payload: session_payload(context, context_modalities, keyframes, references),
    })
}

/// The LoRAs and patches of `--patch` and `--lora`, as a session names them: by the digest of the
/// file, since the machines name the files differently.
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

/// How a step is shared out. There is one way now: every step goes to the workers, and this
/// machine puts the parts back together.
enum Sharing<'a> {
    With(Box<crate::worker::Coordinator<'a>>),
}

/// How a session says this run attends, which only one backend offers a choice about. It comes
/// from the option rather than from a loaded DiT, since a leader that hands every step out has
/// none to ask.
#[cfg(feature = "cuda")]
fn attention_precision(options: &HashMap<&str, &str>) -> u8 {
    match options.get("attention-precision").copied() {
        Some("int8-fp8") => 1,
        _ => 0,
    }
}

#[cfg(not(feature = "cuda"))]
fn attention_precision(_options: &HashMap<&str, &str>) -> u8 {
    0
}
