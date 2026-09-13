//! Text-to-video generation with a synchronized soundtrack.

use crate::USAGE;
use mmh3::cli::{option_float, option_number, parse_options};
use mmh3::models::{
    AUDIO_VAE_FILE, TEXT_ENCODER_FILE, VIDEO_VAE_FILE, VIDEO_VAE_INT8_FILE, load_dit, option_path, sparse_attention,
};
use mmh3_core::safetensors::SafeTensors;
use std::error::Error;
use std::path::Path;

/// Generates a video with its soundtrack from a prompt, or from precomputed text states. The video is written as
/// YUV4MPEG2 to `--out` and the audio as WAV next to it.
pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::{Schedule, euler_step};
    use mmh3_core::generation::{FPS, GenerationShape};
    use mmh3_core::media::{write_wav, write_y4m};
    use mmh3_core::random::NormalSampler;
    use mmh3_core::tensor::Tensor;
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_cuda::audio_vae::{CudaAudioDecoder, SAMPLE_RATE};
    use mmh3_cuda::text_encoder::CudaTextEncoder;
    use mmh3_cuda::vae::{CudaVideoDecoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE};
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &[
            "prompt",
            "prompt-file",
            "context",
            "out",
            "models",
            "width",
            "height",
            "frames",
            "steps",
            "seed",
            "shift-video",
            "shift-audio",
            "dit",
            "video-vae",
            "audio-vae",
            "text-encoder",
            "lora",
            "lora-strength",
            "attention",
            "attention-precision",
            "sparse-tau",
            "sparse-start",
        ],
        USAGE,
    )?;
    let video_path = Path::new(options.get("out").ok_or(USAGE)?).to_path_buf();
    if video_path.extension().and_then(|extension| extension.to_str()) != Some("y4m") {
        return Err("--out must name a .y4m file. The WAV file is written next to it".into());
    }
    let audio_path = video_path.with_extension("wav");
    let shape = GenerationShape::new(
        option_number(&options, "width", 1344)?,
        option_number(&options, "height", 768)?,
        option_number(&options, "frames", 124)?,
    )?;
    let steps = option_number(&options, "steps", 20)?;
    let seed = option_number(&options, "seed", 0)? as u64;
    let (shift_video, shift_audio) =
        (option_float(&options, "shift-video", 12.0)?, option_float(&options, "shift-audio", 3.0)?);
    if steps == 0 {
        return Err("--steps must be at least 1".into());
    }

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
            let path = option_path(&options, "text-encoder", TEXT_ENCODER_FILE)?;
            let encoder = CudaTextEncoder::load(&SafeTensors::open(Path::new(&path))?)?;
            let context = encoder.encode(&ids, &[])?.context;
            println!("encoded {} prompt tokens in {:.1} s", ids.len(), started.elapsed().as_secs_f64());
            context
        }
        (None, Some(path)) => {
            let file = SafeTensors::open(Path::new(path))?;
            let mut context =
                Tensor::load(&file, file.get("context").ok_or("the context file has no context tensor")?)?;
            if context.shape.len() == 3 && context.shape[0] == 1 {
                context.shape.remove(0);
            }
            context
        }
        _ => return Err("pass a prompt or --context, not both".into()),
    };
    // NOTE: the video noise is drawn before the audio noise, the order of the official pipeline.
    let mut noise = NormalSampler::new(seed);
    let (video_shape, audio_shape) = (shape.video_latent_shape(), shape.audio_latent_shape());
    let mut video = Tensor::new(video_shape.clone(), noise.samples(video_shape.iter().product()));
    let mut audio = Tensor::new(audio_shape.clone(), noise.samples(audio_shape.iter().product()));
    println!(
        "{}×{}, {} frames ({:.2} s), {} text tokens, {steps} steps, seed {seed}",
        shape.width,
        shape.height,
        shape.frames,
        shape.frames as f64 / FPS as f64,
        context.shape[0]
    );

    {
        let dit = load_dit(&options, "dit")?;
        let sparse = sparse_attention(&options)?;
        let schedule = Schedule::uniform(steps, shift_video, shift_audio);
        for step in 0..steps {
            let started = Instant::now();
            let inputs = DitInputs {
                video: video.clone(),
                audio: audio.clone(),
                context: context.clone(),
                sigma: schedule.video[step],
                shift_video,
                shift_audio,
            };
            let step_sparse = sparse.filter(|settings| settings.applies_to_step(step, steps));
            let outputs = dit.forward(&inputs, &[], step_sparse.as_ref())?;
            euler_step(&mut video.data, &outputs.video, schedule.video[step], schedule.video[step + 1]);
            euler_step(&mut audio.data, &outputs.audio, schedule.audio[step], schedule.audio[step + 1]);
            let routing = outputs
                .routed_fraction
                .map_or(String::new(), |fraction| format!(", Sol-Attn routed {:.1}%", 100.0 * fraction));
            println!(
                "step {}/{steps} at sigma {:.4} in {:.1} s{routing}",
                step + 1,
                schedule.video[step],
                started.elapsed().as_secs_f64()
            );
        }
    }

    let started = Instant::now();
    // Without --video-vae, the INT8 ConvRot VAE decodes when the models directory has it, and the FP16 one otherwise.
    let path = match options.get("video-vae") {
        Some(path) => (*path).to_owned(),
        None => {
            let int8_path = option_path(&options, "video-vae", VIDEO_VAE_INT8_FILE)?;
            if Path::new(&int8_path).exists() { int8_path } else { option_path(&options, "video-vae", VIDEO_VAE_FILE)? }
        }
    };
    let yuv =
        CudaVideoDecoder::load(&SafeTensors::open(Path::new(&path))?, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?
            .decode_yuv420(&video)?;
    println!("decoded the video with {path} in {:.1} s", started.elapsed().as_secs_f64());
    let mut writer = BufWriter::new(File::create(&video_path)?);
    write_y4m(&mut writer, &yuv, FPS)?;
    writer.flush()?;

    let started = Instant::now();
    let path = option_path(&options, "audio-vae", AUDIO_VAE_FILE)?;
    let waveform = CudaAudioDecoder::load(&SafeTensors::open(Path::new(&path))?, "")?.decode(&audio)?;
    println!("decoded the audio in {:.1} s", started.elapsed().as_secs_f64());
    let mut writer = BufWriter::new(File::create(&audio_path)?);
    write_wav(&mut writer, &waveform, SAMPLE_RATE)?;
    writer.flush()?;

    println!("wrote {} and {}", video_path.display(), audio_path.display());
    let options = "-c:v libx264 -crf 18 -pix_fmt yuv420p -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k";
    println!(
        "an mp4, for example: ffmpeg -i {} -i {} {options} -shortest {}",
        video_path.display(),
        audio_path.display(),
        video_path.with_extension("mp4").display()
    );
    Ok(())
}
