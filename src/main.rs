use mmh3_core::json;
use mmh3_core::safetensors::{DType, SafeTensors};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage:
  mmh3 generate (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <video.y4m> [--models DIR]
                [--width N] [--height N] [--frames N] [--steps N] [--seed N] [--shift-video X] [--shift-audio X]
                [--dit FILE] [--video-vae FILE] [--audio-vae FILE] [--text-encoder FILE] [--tokenizer FILE]
                [--lora FILE] [--lora-strength X] [--attention dense|sol] [--sparse-tau X] [--sparse-start X]
  mmh3 inspect <file.safetensors> [--all]
  mmh3 device
  mmh3 bench gemm [--tokens N] [--iterations N] [--kinds bf16,fp8,int8,nvfp4,int8-mmh3]
  mmh3 bench memory [--megabytes N] [--iterations N]
  mmh3 bench mma [--iterations N]
  mmh3 bench attention [--tokens N] [--heads N] [--iterations N]
  mmh3 check dit --golden <directory> [--models DIR] [--weights FILE] [--lora FILE] [--lora-strength X]
                 [--attention dense|sol] [--sparse-tau X]
  mmh3 check sample --golden <directory> [--models DIR] [--weights FILE] [--lora FILE] [--lora-strength X]
                    [--attention dense|sol] [--sparse-tau X] [--sparse-start X]
  mmh3 check video-vae --golden <directory> [--models DIR] [--weights FILE]
  mmh3 check audio-vae --golden <directory> [--models DIR] [--weights FILE]
  mmh3 check text-encoder --golden <file.safetensors> [--models DIR] [--weights FILE] [--tokenizer FILE]

Checkpoints default to their ComfyUI names inside a models directory laid out like ComfyUI's models folder and the
Comfy-Org/MiniMax-H3 repository, given by --models or MMH3_MODELS:
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
  tokenizer/tokenizer.json (from MiniMaxAI/MiniMax-H3)";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result = match arguments.first().map(String::as_str) {
        Some("generate") => generate(&arguments[1..]),
        Some("inspect") => inspect(&arguments[1..]),
        Some("device") => device(),
        Some("bench") => bench(&arguments[1..]),
        Some("check") => check(&arguments[1..]),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

struct TensorGroup {
    pattern: String,
    count: usize,
    dtype: Option<DType>,
    shape: Vec<usize>,
    uniform_shape: bool,
    elements: usize,
}

fn inspect(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut path = None;
    let mut list_all = false;
    for argument in arguments {
        match argument.as_str() {
            "--all" => list_all = true,
            _ if path.is_none() => path = Some(argument),
            _ => return Err(USAGE.into()),
        }
    }
    let path = path.ok_or(USAGE)?;
    let file = SafeTensors::open(Path::new(path))?;

    println!("file      {path}");
    println!("size      {} bytes ({})", format_count(file.file_size()), format_bytes(file.file_size()));
    println!("header    {} bytes", format_count(file.header_size()));
    println!("tensors   {}", format_count(file.tensors().len()));
    for (key, value) in file.metadata() {
        println!("metadata  {key}: {}", truncate(value, 160));
    }
    println!();

    if list_all {
        for tensor in file.tensors() {
            println!("{:<8} {:<24} {}", tensor.dtype, format_shape(&tensor.shape), tensor.name);
        }
        return Ok(());
    }

    let mut groups: Vec<TensorGroup> = Vec::new();
    let mut group_index: HashMap<String, usize> = HashMap::new();
    let mut quantization_formats: Vec<(String, usize)> = Vec::new();
    let mut bytes_by_dtype: Vec<(DType, usize)> = Vec::new();
    for tensor in file.tensors() {
        let pattern = layer_pattern(&tensor.name);
        let position = *group_index.entry(pattern.clone()).or_insert_with(|| {
            groups.push(TensorGroup {
                pattern,
                count: 0,
                dtype: Some(tensor.dtype),
                shape: tensor.shape.clone(),
                uniform_shape: true,
                elements: 0,
            });
            groups.len() - 1
        });
        let group = &mut groups[position];
        group.count += 1;
        group.elements += tensor.element_count();
        if group.dtype != Some(tensor.dtype) {
            group.dtype = None;
        }
        if group.shape != tensor.shape {
            group.uniform_shape = false;
        }

        match bytes_by_dtype.iter_mut().find(|(dtype, _)| *dtype == tensor.dtype) {
            Some((_, bytes)) => *bytes += tensor.byte_count(),
            None => bytes_by_dtype.push((tensor.dtype, tensor.byte_count())),
        }

        if tensor.name.ends_with(".comfy_quant") && tensor.dtype == DType::U8 {
            let text = std::str::from_utf8(file.data(tensor))?;
            let description = describe_quantization(text);
            match quantization_formats.iter_mut().find(|(known, _)| *known == description) {
                Some((_, count)) => *count += 1,
                None => quantization_formats.push((description, 1)),
            }
        }
    }

    println!("{:>5}  {:<8} {:<24} {:>9}  pattern", "count", "dtype", "shape", "elements");
    for group in &groups {
        let dtype = group.dtype.map_or("mixed", DType::name);
        let shape = if group.uniform_shape { format_shape(&group.shape) } else { "(varies)".to_owned() };
        println!(
            "{:>5}  {:<8} {:<24} {:>9}  {}",
            group.count,
            dtype,
            shape,
            format_elements(group.elements),
            group.pattern
        );
    }

    if !quantization_formats.is_empty() {
        println!();
        println!("quantization (comfy_quant)");
        for (description, count) in &quantization_formats {
            println!("{count:>5}  {description}");
        }
    }

    println!();
    println!("bytes by dtype");
    bytes_by_dtype.sort_by(|left, right| right.1.cmp(&left.1));
    for (dtype, bytes) in &bytes_by_dtype {
        println!("{:>8}  {}", dtype.name(), format_bytes(*bytes));
    }
    Ok(())
}

/// Replaces numeric path components with N so that per-layer tensors group together.
fn layer_pattern(name: &str) -> String {
    name.split('.')
        .map(|component| if component.bytes().all(|byte| byte.is_ascii_digit()) { "N" } else { component })
        .collect::<Vec<_>>()
        .join(".")
}

fn describe_quantization(text: &str) -> String {
    let Ok(value) = json::parse(text) else {
        return text.to_owned();
    };
    let Some(members) = value.as_object() else {
        return text.to_owned();
    };
    members
        .iter()
        .map(|(key, value)| {
            let rendered = match value {
                json::Value::String(text) => text.clone(),
                json::Value::Bool(flag) => flag.to_string(),
                json::Value::Number(number) => number.to_string(),
                other => format!("{other:?}"),
            };
            format!("{key}={rendered}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_shape(shape: &[usize]) -> String {
    let dimensions: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", dimensions.join(", "))
}

fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::new();
    for (position, digit) in digits.chars().enumerate() {
        if position > 0 && (digits.len() - position) % 3 == 0 {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

fn format_elements(value: usize) -> String {
    match value {
        1_000_000_000.. => format!("{:.3}B", value as f64 / 1e9),
        1_000_000.. => format!("{:.2}M", value as f64 / 1e6),
        1_000.. => format!("{:.1}K", value as f64 / 1e3),
        _ => value.to_string(),
    }
}

fn format_bytes(value: usize) -> String {
    match value {
        1_000_000_000.. => format!("{:.2} GB", value as f64 / 1e9),
        1_000_000.. => format!("{:.2} MB", value as f64 / 1e6),
        1_000.. => format!("{:.1} KB", value as f64 / 1e3),
        _ => format!("{value} B"),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    match text.char_indices().nth(limit) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

#[cfg(feature = "cuda")]
fn device() -> Result<(), Box<dyn Error>> {
    let count = mmh3_cuda::device_count()?;
    for device in 0..count {
        let info = mmh3_cuda::device_info(device)?;
        println!("device {device}: {}", info.name);
        println!("  compute capability     {}.{}", info.compute_capability.0, info.compute_capability.1);
        println!("  multiprocessors        {}", info.multiprocessor_count);
        println!("  shared memory / block  {} bytes (opt-in)", format_count(info.shared_memory_per_block_optin as usize));
        println!("  shared memory / SM     {} bytes", format_count(info.shared_memory_per_multiprocessor as usize));
        println!("  L2 cache               {}", format_bytes(info.l2_cache_bytes as usize));
        println!("  integrated             {}", info.integrated);
        println!("  global memory          {}", format_bytes(info.total_global_bytes as usize));
    }
    let (free_bytes, total_bytes) = mmh3_cuda::memory_info()?;
    println!("memory    {} free of {}", format_bytes(free_bytes), format_bytes(total_bytes));
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn device() -> Result<(), Box<dyn Error>> {
    Err("this build has no GPU backend. Rebuild with --features cuda".into())
}

/// Parses `--name value` pairs, rejecting names outside `allowed`.
#[cfg(feature = "cuda")]
fn parse_options<'a>(arguments: &'a [String], allowed: &[&str]) -> Result<HashMap<&'a str, &'a str>, Box<dyn Error>> {
    let mut options = HashMap::new();
    let mut remaining = arguments.iter();
    while let Some(argument) = remaining.next() {
        let name = argument.strip_prefix("--").filter(|name| allowed.contains(name)).ok_or(USAGE)?;
        let value = remaining.next().ok_or_else(|| format!("--{name} needs a value"))?;
        options.insert(name, value.as_str());
    }
    Ok(options)
}

#[cfg(feature = "cuda")]
fn option_number(options: &HashMap<&str, &str>, name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match options.get(name) {
        Some(value) => Ok(value.replace('_', "").parse().map_err(|_| format!("--{name} must be a number"))?),
        None => Ok(default),
    }
}

#[cfg(feature = "cuda")]
fn option_float(options: &HashMap<&str, &str>, name: &str, default: f32) -> Result<f32, Box<dyn Error>> {
    match options.get(name) {
        Some(value) => Ok(value.parse().map_err(|_| format!("--{name} must be a number"))?),
        None => Ok(default),
    }
}

#[cfg(feature = "cuda")]
fn bench(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    match arguments.first().map(String::as_str) {
        Some("gemm") => bench_gemm(&arguments[1..]),
        Some("memory") => bench_memory(&arguments[1..]),
        Some("mma") => bench_mma(&arguments[1..]),
        Some("attention") => bench_attention(&arguments[1..]),
        _ => Err(USAGE.into()),
    }
}

#[cfg(not(feature = "cuda"))]
fn bench(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("this build has no GPU backend. Rebuild with --features cuda".into())
}

/// Output and input features of the four linear layers in every DiT block.
#[cfg(feature = "cuda")]
const DIT_LINEAR_SHAPES: [(&str, usize, usize); 4] =
    [("qkv", 21504, 5376), ("out", 5376, 7168), ("fc1", 28672, 5376), ("fc2", 5376, 14336)];

#[cfg(feature = "cuda")]
const DIT_BLOCK_COUNT: usize = 50;

#[cfg(feature = "cuda")]
fn bench_gemm(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::bench::{GemmKind, gemm};

    let options = parse_options(arguments, &["tokens", "iterations", "kinds"])?;
    let tokens = option_number(&options, "tokens", 38_710)?;
    let iterations = option_number(&options, "iterations", 10)?;
    let kinds = match options.get("kinds") {
        Some(list) => list
            .split(',')
            .map(|name| GemmKind::from_name(name).ok_or_else(|| format!("unknown GEMM kind {name}")))
            .collect::<Result<Vec<_>, _>>()?,
        None => GemmKind::ALL.to_vec(),
    };

    println!("y[{tokens}, N] = x[{tokens}, K] · w[N, K]ᵀ, best candidate of each kind, {iterations} runs each");
    println!("{:<6} {:>6} {:>6}  {:<9} {:>10} {:>9} {:>10}", "layer", "N", "K", "kind", "ms", "TFLOPS", "candidates");
    let mut step_seconds: Vec<(GemmKind, Option<f64>)> = kinds.iter().map(|&kind| (kind, Some(0.0))).collect();
    for (layer, output_features, input_features) in DIT_LINEAR_SHAPES {
        for (kind, total) in step_seconds.iter_mut() {
            let label = format!("{layer:<6} {output_features:>6} {input_features:>6}  {:<9}", kind.name());
            match gemm(*kind, tokens, output_features, input_features, iterations) {
                Ok(timing) => {
                    let operations = 2.0 * tokens as f64 * output_features as f64 * input_features as f64;
                    let teraflops = operations / (timing.milliseconds as f64 * 1e-3) / 1e12;
                    println!("{label} {:>10.3} {:>9.1} {:>10}", timing.milliseconds, teraflops, timing.candidates_timed);
                    if let Some(seconds) = total {
                        *seconds += timing.milliseconds as f64 * DIT_BLOCK_COUNT as f64 / 1e3;
                    }
                }
                Err(error) => {
                    println!("{label} {}", error.message);
                    *total = None;
                }
            }
        }
    }
    println!();
    println!("linear time per DiT forward ({DIT_BLOCK_COUNT} blocks)");
    for (kind, total) in &step_seconds {
        match total {
            Some(seconds) => println!("  {:<9} {seconds:.2} s", kind.name()),
            None => println!("  {:<9} unavailable", kind.name()),
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_memory(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse_options(arguments, &["megabytes", "iterations"])?;
    let megabytes = option_number(&options, "megabytes", 2048)?;
    let iterations = option_number(&options, "iterations", 20)?;
    let bandwidth = mmh3_cuda::bench::memory_copy(megabytes << 20, iterations)?;
    println!("device copy of {megabytes} MiB, {iterations} runs (read + write)");
    println!("  copy kernel  {:.1} GB/s", bandwidth.copy_kernel_gigabytes_per_second);
    println!("  cudaMemcpy   {:.1} GB/s", bandwidth.memcpy_gigabytes_per_second);
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_mma(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::bench::{MmaKind, mma_peak};

    let options = parse_options(arguments, &["iterations"])?;
    let iterations = option_number(&options, "iterations", 4096)?;
    println!("register-only mma.sync throughput, {iterations} iterations of 8 independent chains per warp");
    for kind in MmaKind::ALL {
        let throughput = mma_peak(kind, iterations)?;
        println!("  {:<22} {throughput:>7.1} T/s", kind.instruction());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench_attention(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse_options(arguments, &["tokens", "heads", "iterations"])?;
    let tokens = option_number(&options, "tokens", 38_710)?;
    let heads = option_number(&options, "heads", 56)?;
    let iterations = option_number(&options, "iterations", 3)?;
    let milliseconds = mmh3_cuda::bench::attention(tokens, heads, iterations)?;
    let operations = 4.0 * (tokens as f64).powi(2) * 128.0 * heads as f64;
    let teraflops = operations / (milliseconds as f64 * 1e-3) / 1e12;
    println!("dense attention, {tokens} tokens, {heads} heads of 128, {iterations} runs");
    println!("  {milliseconds:.3} ms per call, {teraflops:.1} TFLOPS");
    println!("  {:.2} s per DiT forward ({DIT_BLOCK_COUNT} blocks)", milliseconds as f64 * DIT_BLOCK_COUNT as f64 / 1e3);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn generate(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("this build has no GPU backend. Rebuild with --features cuda".into())
}

/// Generates a video with its soundtrack from a prompt, or from precomputed text states. The video is written as
/// YUV4MPEG2 to `--out` and the audio as WAV next to it.
#[cfg(feature = "cuda")]
fn generate(arguments: &[String]) -> Result<(), Box<dyn Error>> {
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
            "tokenizer",
            "lora",
            "lora-strength",
            "attention",
            "sparse-tau",
            "sparse-start",
        ],
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
    let (shift_video, shift_audio) = (option_float(&options, "shift-video", 12.0)?, option_float(&options, "shift-audio", 3.0)?);
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
            let ids = Tokenizer::load_h3(Path::new(&option_path(&options, "tokenizer", TOKENIZER_FILE)?))?.encode(&prompt);
            let path = option_path(&options, "text-encoder", TEXT_ENCODER_FILE)?;
            let encoder = CudaTextEncoder::load(&SafeTensors::open(Path::new(&path))?)?;
            let context = encoder.encode(&ids, &[])?.context;
            println!("encoded {} prompt tokens in {:.1} s", ids.len(), started.elapsed().as_secs_f64());
            context
        }
        (None, Some(path)) => {
            let file = SafeTensors::open(Path::new(path))?;
            let mut context = Tensor::load(&file, file.get("context").ok_or("the context file has no context tensor")?)?;
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
            let routing = outputs.routed_fraction.map_or(String::new(), |fraction| format!(", Sol-Attn routed {:.1}%", 100.0 * fraction));
            println!("step {}/{steps} at sigma {:.4} in {:.1} s{routing}", step + 1, schedule.video[step], started.elapsed().as_secs_f64());
        }
    }

    let started = Instant::now();
    let path = option_path(&options, "video-vae", VIDEO_VAE_FILE)?;
    let pixels = CudaVideoDecoder::load(&SafeTensors::open(Path::new(&path))?, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?
        .decode(&video, false)?
        .pixels;
    println!("decoded the video in {:.1} s", started.elapsed().as_secs_f64());
    let mut writer = BufWriter::new(File::create(&video_path)?);
    write_y4m(&mut writer, &pixels, FPS)?;
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

#[cfg(not(feature = "cuda"))]
fn check(_arguments: &[String]) -> Result<(), Box<dyn Error>> {
    Err("this build has no GPU backend. Rebuild with --features cuda".into())
}

#[cfg(feature = "cuda")]
fn check(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    match arguments.first().map(String::as_str) {
        Some("dit") => check_dit(&arguments[1..]),
        Some("sample") => check_sample(&arguments[1..]),
        Some("video-vae") => check_video_vae(&arguments[1..]),
        Some("audio-vae") => check_audio_vae(&arguments[1..]),
        Some("text-encoder") => check_text_encoder(&arguments[1..]),
        _ => Err(USAGE.into()),
    }
}

/// Environment variable naming the models directory when `--models` is absent.
#[cfg(feature = "cuda")]
const MODELS_VARIABLE: &str = "MMH3_MODELS";

// Files inside the models directory.
#[cfg(feature = "cuda")]
const DIT_FILE: &str = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
#[cfg(feature = "cuda")]
const VIDEO_VAE_FILE: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
#[cfg(feature = "cuda")]
const AUDIO_VAE_FILE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";
#[cfg(feature = "cuda")]
const TEXT_ENCODER_FILE: &str = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors";
#[cfg(feature = "cuda")]
const TOKENIZER_FILE: &str = "tokenizer/tokenizer.json";

/// Loads the DiT from `--weights` or `--dit`, whichever `name` says, with the LoRA of `--lora` when given.
#[cfg(feature = "cuda")]
fn load_dit(options: &HashMap<&str, &str>, name: &str) -> Result<mmh3_cuda::dit::CudaDit, Box<dyn Error>> {
    use std::time::Instant;

    let started = Instant::now();
    let path = option_path(options, name, DIT_FILE)?;
    let mut dit = mmh3_cuda::dit::CudaDit::load(&SafeTensors::open(Path::new(&path))?, "")?;
    println!("loaded {path} in {:.1} s", started.elapsed().as_secs_f64());
    if let Some(lora) = options.get("lora") {
        let strength = option_float(options, "lora-strength", 1.0)?;
        let layers = dit.add_lora(&SafeTensors::open(Path::new(lora))?, strength)?;
        println!("added {lora} to {layers} layers at strength {strength}");
    }
    Ok(dit)
}

/// Sol-Attn settings from `--attention sol`, `--sparse-tau` and `--sparse-start`, or None for dense attention.
#[cfg(feature = "cuda")]
fn sparse_attention(options: &HashMap<&str, &str>) -> Result<Option<mmh3_core::dit::sparse::SparseAttention>, Box<dyn Error>> {
    use mmh3_core::dit::sparse::SparseAttention;

    match options.get("attention").copied().unwrap_or("dense") {
        "dense" => Ok(None),
        "sol" => {
            let defaults = SparseAttention::default();
            Ok(Some(SparseAttention {
                tau: option_float(options, "sparse-tau", defaults.tau)?,
                start_fraction: option_float(options, "sparse-start", defaults.start_fraction)?,
                ..defaults
            }))
        }
        other => Err(format!("--attention must be dense or sol, not {other}").into()),
    }
}

/// The path of `--name`, or `file` inside the models directory of `--models` or `MMH3_MODELS`.
#[cfg(feature = "cuda")]
fn option_path(options: &HashMap<&str, &str>, name: &str, file: &str) -> Result<String, Box<dyn Error>> {
    if let Some(path) = options.get(name) {
        return Ok((*path).to_owned());
    }
    let directory = match options.get("models") {
        Some(directory) => std::path::PathBuf::from(directory),
        None => std::env::var_os(MODELS_VARIABLE)
            .map(std::path::PathBuf::from)
            .ok_or_else(|| format!("pass --{name} FILE, or a models directory with --models DIR or {MODELS_VARIABLE}"))?,
    };
    Ok(directory.join(file).to_string_lossy().into_owned())
}

/// One file of a golden directory written by tools/golden/h3_reference.py.
#[cfg(feature = "cuda")]
struct GoldenFile(SafeTensors);

#[cfg(feature = "cuda")]
impl GoldenFile {
    fn open(directory: &Path, name: &str) -> Result<Self, Box<dyn Error>> {
        Ok(GoldenFile(SafeTensors::open(&directory.join(name))?))
    }

    fn metadata(&self, key: &str) -> Result<String, Box<dyn Error>> {
        Ok(self.0.metadata().iter().find(|(name, _)| name == key).ok_or(format!("golden metadata lacks {key}"))?.1.clone())
    }

    fn tensor(&self, name: &str) -> Result<mmh3_core::tensor::Tensor, Box<dyn Error>> {
        Ok(mmh3_core::tensor::Tensor::load(&self.0, self.0.get(name).ok_or(format!("golden data lacks {name}"))?)?)
    }

    /// The tensor without its leading batch dimension.
    fn unbatched(&self, name: &str) -> Result<mmh3_core::tensor::Tensor, Box<dyn Error>> {
        let mut tensor = self.tensor(name)?;
        tensor.shape.remove(0);
        Ok(tensor)
    }
}

/// Similarity of `actual` to `expected`: cosine, relative L2 error and the largest absolute error against the
/// largest expected magnitude.
#[cfg(feature = "cuda")]
fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64, f32, f32) {
    let dot: f64 = actual.iter().zip(expected).map(|(&left, &right)| left as f64 * right as f64).sum();
    let norm = |values: &[f32]| values.iter().map(|&value| value as f64 * value as f64).sum::<f64>().sqrt();
    let difference: f64 = actual.iter().zip(expected).map(|(&left, &right)| (left as f64 - right as f64).powi(2)).sum::<f64>().sqrt();
    let scale = expected.iter().fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = actual.iter().zip(expected).fold(0.0f32, |maximum, (&left, &right)| maximum.max((left - right).abs()));
    (dot / (norm(actual) * norm(expected)), difference / norm(expected), worst, scale)
}

/// Runs the CUDA DiT on the captured step of a golden directory written by tools/golden/h3_reference.py.
#[cfg(feature = "cuda")]
fn check_dit(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights", "lora", "lora-strength", "attention", "sparse-tau"])?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let text_file = GoldenFile::open(golden, "text.safetensors")?;
    let step: usize = dit_file.metadata("capture_step")?.parse()?;
    let stride: usize = dit_file.metadata("block_row_stride")?.parse()?;
    let sigmas = json::parse(&dit_file.metadata("sigmas_video")?)?;
    let sigma = sigmas.as_array().and_then(|values| values.get(step)).and_then(json::Value::as_f64).ok_or("bad sigmas_video")?;
    let watched: Vec<usize> = json::parse(&dit_file.metadata("watched_blocks")?)?
        .as_array()
        .ok_or("bad watched_blocks")?
        .iter()
        .filter_map(|value| value.as_u64().map(|index| index as usize))
        .collect();
    let inputs = DitInputs {
        video: dit_file.unbatched(&format!("step{step}.input.video"))?,
        audio: dit_file.unbatched(&format!("step{step}.input.audio"))?,
        context: text_file.tensor("context")?,
        sigma: sigma as f32,
        shift_video: dit_file.metadata("shift_video")?.parse()?,
        shift_audio: dit_file.metadata("shift_audio")?.parse()?,
    };

    let dit = load_dit(&options, "weights")?;
    let started = Instant::now();
    let outputs = dit.forward(&inputs, &watched, sparse_attention(&options)?.as_ref())?;
    let routing = outputs.routed_fraction.map_or(String::new(), |fraction| format!(", Sol-Attn routed {:.1}%", 100.0 * fraction));
    println!("forward at sigma {sigma:.4} in {:.1} s{routing}", started.elapsed().as_secs_f64());

    println!("{:<12} {:>11} {:>11} {:>11} {:>9}", "tensor", "cosine", "rel L2", "max error", "scale");
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, scale) = compare(actual, expected);
        println!("{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {scale:>9.3}");
    };
    let hidden = dit.config().hidden;
    let text_tokens = inputs.context.shape[0];
    let audio_rows = 2 * inputs.audio.shape[2];
    let segment_of = |token: usize| match token {
        token if token < text_tokens => "text",
        token if token < text_tokens + audio_rows => "audio",
        _ => "video",
    };
    for (index, residual) in &outputs.blocks {
        let expected = dit_file.tensor(&format!("step{step}.block.{index}"))?;
        let sampled: Vec<f32> = residual.chunks_exact(hidden).step_by(stride).flatten().copied().collect();
        report(&format!("block {index}"), &sampled, &expected.data);
        for segment in ["text", "audio", "video"] {
            let (mut actual, mut reference) = (Vec::new(), Vec::new());
            for (sample, (row, expected_row)) in sampled.chunks_exact(hidden).zip(expected.data.chunks_exact(hidden)).enumerate() {
                if segment_of(sample * stride) == segment {
                    actual.extend_from_slice(row);
                    reference.extend_from_slice(expected_row);
                }
            }
            if !actual.is_empty() {
                report(&format!("  {segment}"), &actual, &reference);
            }
        }
    }
    // The golden data stores the data-ward prediction u with x0 = x + sigma · u, the negated flow velocity.
    let negated = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        Ok(dit_file.tensor(name)?.data.into_iter().map(|value| -value).collect())
    };
    report("video", &outputs.video, &negated(&format!("step{step}.velocity.video"))?);
    report("audio", &outputs.audio, &negated(&format!("step{step}.velocity.audio"))?);
    Ok(())
}

/// Samples every step of a golden directory from its initial noise and compares the latents after each step.
#[cfg(feature = "cuda")]
fn check_sample(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::sampler::{Schedule, euler_step};
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights", "lora", "lora-strength", "attention", "sparse-tau", "sparse-start"])?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let text_file = GoldenFile::open(golden, "text.safetensors")?;
    let sparse = sparse_attention(&options)?;
    let (shift_video, shift_audio): (f32, f32) = (dit_file.metadata("shift_video")?.parse()?, dit_file.metadata("shift_audio")?.parse()?);
    let schedule = Schedule::uniform(dit_file.metadata("steps")?.parse()?, shift_video, shift_audio);

    let dit = load_dit(&options, "weights")?;
    let mut video = dit_file.unbatched("noise.video")?;
    let mut audio = dit_file.unbatched("noise.audio")?;
    let context = text_file.tensor("context")?;
    println!("{:<6} {:>8} {:>11} {:>11} {:>11} {:>11} {:>7}", "step", "sigma", "video cos", "video L2", "audio cos", "audio L2", "s");
    for step in 0..schedule.steps() {
        let started = Instant::now();
        let inputs = DitInputs {
            video: video.clone(),
            audio: audio.clone(),
            context: context.clone(),
            sigma: schedule.video[step],
            shift_video,
            shift_audio,
        };
        let step_sparse = sparse.filter(|settings| settings.applies_to_step(step, schedule.steps()));
        let outputs = dit.forward(&inputs, &[], step_sparse.as_ref())?;
        euler_step(&mut video.data, &outputs.video, schedule.video[step], schedule.video[step + 1]);
        euler_step(&mut audio.data, &outputs.audio, schedule.audio[step], schedule.audio[step + 1]);
        let (video_cosine, video_relative, _, _) = compare(&video.data, &dit_file.tensor(&format!("step{step}.latent.video"))?.data);
        let (audio_cosine, audio_relative, _, _) = compare(&audio.data, &dit_file.tensor(&format!("step{step}.latent.audio"))?.data);
        println!(
            "{step:<6} {:>8.4} {video_cosine:>11.7} {video_relative:>11.3e} {audio_cosine:>11.7} {audio_relative:>11.3e} {:>7.1}",
            schedule.video[step],
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

/// Decodes the final video latent of a golden directory and compares the pixels with ComfyUI's.
#[cfg(feature = "cuda")]
fn check_video_vae(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::vae::{CudaVideoDecoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE};
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights"])?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let weights_path = option_path(&options, "weights", VIDEO_VAE_FILE)?;
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let steps: usize = dit_file.metadata("steps")?.parse()?;
    let latent = dit_file.unbatched(&format!("step{}.latent.video", steps - 1))?;

    let started = Instant::now();
    let weights = SafeTensors::open(Path::new(&weights_path))?;
    let decoder = CudaVideoDecoder::load(&weights, "", DEFAULT_TILE_SIZE, DEFAULT_TILE_OVERLAP_MIN)?;
    println!("loaded {weights_path} in {:.1} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let pixels = decoder.decode(&latent, false)?.pixels;
    println!("decoded latent {:?} to {:?} in {:.1} s", latent.shape, pixels.shape, started.elapsed().as_secs_f64());
    if !golden.join("decode.safetensors").exists() {
        println!("no decode.safetensors in the golden directory, nothing to compare");
        return Ok(());
    }
    let expected = GoldenFile::open(golden, "decode.safetensors")?.unbatched("video")?;
    if pixels.shape != expected.shape {
        return Err(format!("decoded shape {:?} differs from the golden {:?}", pixels.shape, expected.shape).into());
    }

    let psnr = |actual: &[f32], expected: &[f32]| {
        let squared: f64 = actual.iter().zip(expected).map(|(&left, &right)| (left as f64 - right as f64).powi(2)).sum();
        10.0 * (actual.len() as f64 / squared).log10()
    };
    println!("{:<12} {:>11} {:>11} {:>11} {:>9}", "pixels", "cosine", "rel L2", "max error", "PSNR dB");
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, _) = compare(actual, expected);
        println!("{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {:>9.2}", psnr(actual, expected));
    };
    report("all", &pixels.data, &expected.data);
    let (frames, plane) = (pixels.shape[1], pixels.shape[2] * pixels.shape[3]);
    let frame_values = |data: &[f32], frame: usize| -> Vec<f32> {
        (0..pixels.shape[0]).flat_map(|channel| data[(channel * frames + frame) * plane..][..plane].to_vec()).collect()
    };
    for frame in 0..frames {
        report(&format!("frame {frame}"), &frame_values(&pixels.data, frame), &frame_values(&expected.data, frame));
    }
    Ok(())
}

/// Decodes the final audio latent of a golden directory and compares the waveform with ComfyUI's decodes.
#[cfg(feature = "cuda")]
fn check_audio_vae(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::audio_vae::CudaAudioDecoder;
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights"])?;
    let golden = Path::new(options.get("golden").ok_or(USAGE)?);
    let weights_path = option_path(&options, "weights", AUDIO_VAE_FILE)?;
    let dit_file = GoldenFile::open(golden, "dit.safetensors")?;
    let steps: usize = dit_file.metadata("steps")?.parse()?;
    let latent = dit_file.unbatched(&format!("step{}.latent.audio", steps - 1))?;

    let started = Instant::now();
    let weights = SafeTensors::open(Path::new(&weights_path))?;
    let decoder = CudaAudioDecoder::load(&weights, "")?;
    println!("loaded {weights_path} in {:.1} s", started.elapsed().as_secs_f64());
    let started = Instant::now();
    let waveform = decoder.decode(&latent)?;
    println!("decoded latent {:?} to {:?} in {:.2} s", latent.shape, waveform.shape, started.elapsed().as_secs_f64());
    if !golden.join("decode.safetensors").exists() {
        println!("no decode.safetensors in the golden directory, nothing to compare");
        return Ok(());
    }

    let decode_file = GoldenFile::open(golden, "decode.safetensors")?;
    println!("{:<16} {:>11} {:>11} {:>11} {:>8}", "against", "cosine", "rel L2", "max error", "SNR dB");
    // NOTE: ComfyUI decodes audio in BF16 by default. The golden script also stores an FP32 decode.
    for (label, name) in [("ComfyUI default", "audio"), ("ComfyUI FP32", "audio_float32")] {
        if decode_file.0.get(name).is_none() {
            continue;
        }
        let expected = decode_file.unbatched(name)?;
        if waveform.shape != expected.shape {
            return Err(format!("decoded shape {:?} differs from the golden {:?}", waveform.shape, expected.shape).into());
        }
        let (cosine, relative, worst, _) = compare(&waveform.data, &expected.data);
        println!("{label:<16} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {:>8.2}", -20.0 * relative.log10());
    }
    Ok(())
}

/// Tokenizes and encodes the prompt of a golden file written by tools/golden/text_encoder.py and compares the token
/// ids, the hidden states it holds and the conditioning.
#[cfg(feature = "cuda")]
fn check_text_encoder(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::tokenizer::Tokenizer;
    use mmh3_cuda::text_encoder::CudaTextEncoder;
    use std::time::Instant;

    let options = parse_options(arguments, &["golden", "models", "weights", "tokenizer"])?;
    let golden = GoldenFile(SafeTensors::open(Path::new(options.get("golden").ok_or(USAGE)?))?);
    let prompt = golden.metadata("prompt")?;
    let id_info = golden.0.get("token_ids").ok_or("golden data lacks token_ids")?;
    let expected_ids: Vec<u32> = golden.0.data(id_info).chunks_exact(8).map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()) as u32).collect();
    let tokenizer = Tokenizer::load_h3(Path::new(&option_path(&options, "tokenizer", TOKENIZER_FILE)?))?;
    let ids = tokenizer.encode(&prompt);
    if ids != expected_ids {
        return Err(format!("token ids differ from the golden data:\n  expected {expected_ids:?}\n  actual   {ids:?}").into());
    }
    println!("{} tokens match", ids.len());

    let started = Instant::now();
    let weights_path = option_path(&options, "weights", TEXT_ENCODER_FILE)?;
    let encoder = CudaTextEncoder::load(&SafeTensors::open(Path::new(&weights_path))?)?;
    println!("loaded {weights_path} in {:.1} s", started.elapsed().as_secs_f64());
    let captured: Vec<usize> = golden.0.tensors().iter().filter_map(|info| info.name.strip_prefix("layer.")?.parse().ok()).collect();
    let started = Instant::now();
    let encoding = encoder.encode(&ids, &captured)?;
    println!("encoded in {:.3} s", started.elapsed().as_secs_f64());

    println!("{:<12} {:>11} {:>11} {:>11} {:>11}", "tensor", "cosine", "rel L2", "max error", "scale");
    let report = |name: &str, actual: &[f32], expected: &[f32]| {
        let (cosine, relative, worst, scale) = compare(actual, expected);
        println!("{name:<12} {cosine:>11.7} {relative:>11.3e} {worst:>11.3e} {scale:>11.3}");
    };
    for (layer, hidden) in &encoding.layers {
        report(&format!("layer {layer}"), &hidden.data, &golden.tensor(&format!("layer.{layer}"))?.data);
    }
    let expected = golden.tensor("context")?;
    report("context", &encoding.context.data, &expected.data);
    // NOTE: the first token carries the model's massive activations, so it is reported apart from the rest.
    let hidden = encoder.config().hidden;
    report("  token 0", &encoding.context.data[..hidden], &expected.data[..hidden]);
    if ids.len() > 1 {
        report("  others", &encoding.context.data[hidden..], &expected.data[hidden..]);
    }
    Ok(())
}
