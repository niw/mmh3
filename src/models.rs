//! Checkpoint paths and DiT settings shared by generation and golden-data checks.

use crate::cli::option_float;
use mmh3_core::safetensors::SafeTensors;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

/// Environment variable naming the models directory when `--models` is absent.
const MODELS_VARIABLE: &str = "MMH3_MODELS";

// Files inside the models directory.
const DIT_FILE: &str = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
pub const VIDEO_VAE_FILE: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
/// The faster INT8 ConvRot video VAE, which generate prefers when the models directory has it.
pub const VIDEO_VAE_INT8_FILE: &str = "vae/minimax_h3_video_vae_int8_convrot.safetensors";
pub const AUDIO_VAE_FILE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";
pub const TEXT_ENCODER_FILE: &str = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors";

/// Loads the DiT from `--weights` or `--dit`, whichever `name` says, with the LoRA of `--lora` when
/// given.
pub fn load_dit(
    options: &HashMap<&str, &str>,
    name: &str,
) -> Result<mmh3_cuda::dit::CudaDit, Box<dyn Error>> {
    use std::time::Instant;

    let precision = match options
        .get("attention-precision")
        .copied()
        .unwrap_or("bf16")
    {
        "bf16" => mmh3_cuda::attention::AttentionPrecision::Bf16,
        "int8-fp8" => mmh3_cuda::attention::AttentionPrecision::Int8Fp8,
        other => {
            return Err(
                format!("--attention-precision must be bf16 or int8-fp8, not {other}").into(),
            );
        }
    };
    let started = Instant::now();
    let path = option_path(options, name, DIT_FILE)?;
    let mut dit = mmh3_cuda::dit::CudaDit::load(&SafeTensors::open(Path::new(&path))?, "")?;
    dit.set_attention_precision(precision);
    println!("loaded {path} in {:.1} s", started.elapsed().as_secs_f64());
    if let Some(lora) = options.get("lora") {
        let strength = option_float(options, "lora-strength", 1.0)?;
        let layers = dit.add_lora(&SafeTensors::open(Path::new(lora))?, strength)?;
        println!("added {lora} to {layers} layers at strength {strength}");
    }
    Ok(dit)
}

/// Sol-Attn settings from `--attention sol`, `--sparse-tau` and `--sparse-start`, or None for dense
/// attention.
pub fn sparse_attention(
    options: &HashMap<&str, &str>,
) -> Result<Option<mmh3_core::dit::sparse::SparseAttention>, Box<dyn Error>> {
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
pub fn option_path(
    options: &HashMap<&str, &str>,
    name: &str,
    file: &str,
) -> Result<String, Box<dyn Error>> {
    if let Some(path) = options.get(name) {
        return Ok((*path).to_owned());
    }
    let directory = match options.get("models") {
        Some(directory) => std::path::PathBuf::from(directory),
        None => std::env::var_os(MODELS_VARIABLE).map(std::path::PathBuf::from).ok_or_else(|| {
            format!("pass --{name} FILE, or a models directory with --models DIR or {MODELS_VARIABLE}")
        })?,
    };
    Ok(directory.join(file).to_string_lossy().into_owned())
}
