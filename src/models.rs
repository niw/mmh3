//! Checkpoint paths and DiT settings shared by generation and golden-data checks.

use crate::cli::option_float;
use mmh3_core::safetensors::SafeTensors;
use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Environment variable naming the models directory when `--models` is absent.
const MODELS_VARIABLE: &str = "MMH3_MODELS";

// Files inside the models directory.
/// The DiT for text, first frame and last frame to video.
pub const DIT_FILE: &str = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors";
/// The DiT for reference to video.
pub const REFERENCE_DIT_FILE: &str =
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors";
pub const VIDEO_VAE_FILE: &str = "vae/minimax_h3_video_vae_fp16.safetensors";
/// The faster INT8 ConvRot video VAE, which generate prefers when the models directory has it.
pub const VIDEO_VAE_INT8_FILE: &str = "vae/minimax_h3_video_vae_int8_convrot.safetensors";
pub const AUDIO_VAE_FILE: &str = "vae/minimax_h3_audio_vae_fp32.safetensors";
pub const TEXT_ENCODER_FILE: &str = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors";

/// Loads the DiT from `--weights` or `--dit`, whichever `name` says, or `file` in the models
/// directory, with the patch of `--patch` and the LoRA of `--lora` when given. NVFP4 layers take
/// them into their weights, so they come before `--linear-precision`.
pub fn load_dit(
    options: &HashMap<&str, &str>,
    name: &str,
    file: &str,
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
    let path = option_path(options, name, file)?;
    let mut dit = mmh3_cuda::dit::CudaDit::load(&SafeTensors::open(Path::new(&path))?, "")?;
    dit.set_attention_precision(precision);
    println!("loaded {path} in {:.1} s", started.elapsed().as_secs_f64());
    // NOTE: a patch is a LoRA file whose other tensors replace or add checkpoint tensors, and it
    // applies as adapters at full strength.
    if let Some(patch) = model_file(options, "patch", &["patches", "loras"])? {
        let layers = dit.add_lora(
            &SafeTensors::open(Path::new(&patch))?,
            1.0,
            mmh3_cuda::dit::LoraMode::Adapter,
        )?;
        println!("applied {patch} to {layers} layers and tensors");
    }
    if let Some(lora) = model_file(options, "lora", &["loras"])? {
        let started = Instant::now();
        let strength = option_float(options, "lora-strength", 1.0)?;
        let mode = match options.get("lora-mode").copied().unwrap_or("adapter") {
            "adapter" => mmh3_cuda::dit::LoraMode::Adapter,
            "merge" => mmh3_cuda::dit::LoraMode::Merge,
            other => {
                return Err(format!("--lora-mode must be adapter or merge, not {other}").into());
            }
        };
        let layers = dit.add_lora(&SafeTensors::open(Path::new(&lora))?, strength, mode)?;
        println!(
            "added {lora} to {layers} layers at strength {strength} in {:.1} s",
            started.elapsed().as_secs_f64()
        );
    }
    match options.get("linear-precision").copied().unwrap_or("int8") {
        "int8" => {}
        "nvfp4" => {
            let started = Instant::now();
            let layers = dit.use_nvfp4()?;
            println!(
                "requantized {layers} layers to NVFP4 in {:.1} s",
                started.elapsed().as_secs_f64()
            );
            if let Some(cache) = algorithm_cache() {
                match mmh3_cuda::nvfp4::load_algorithms(&cache) {
                    Ok(0) => {}
                    Ok(count) => {
                        println!("took {count} cuBLASLt algorithms from {}", cache.display())
                    }
                    Err(error) => eprintln!("warning: reading {}: {error}", cache.display()),
                }
            }
        }
        other => {
            return Err(format!("--linear-precision must be int8 or nvfp4, not {other}").into());
        }
    }
    Ok(dit)
}

/// The video VAE of `--video-vae`, or without it the INT8 ConvRot VAE when the models directory has
/// it and the FP16 one otherwise.
pub fn video_vae_path(options: &HashMap<&str, &str>) -> Result<String, Box<dyn Error>> {
    if let Some(path) = options.get("video-vae") {
        return Ok((*path).to_owned());
    }
    let int8_path = option_path(options, "video-vae", VIDEO_VAE_INT8_FILE)?;
    if Path::new(&int8_path).exists() {
        Ok(int8_path)
    } else {
        option_path(options, "video-vae", VIDEO_VAE_FILE)
    }
}

/// File that keeps the cuBLASLt algorithms chosen for the NVFP4 GEMMs between runs, in
/// `$XDG_CACHE_HOME/mmh3` or `~/.cache/mmh3`.
fn algorithm_cache() -> Option<PathBuf> {
    let directory = std::env::var_os("XDG_CACHE_HOME")
        .filter(|directory| !directory.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(directory.join("mmh3/cublaslt-nvfp4-algorithms.txt"))
}

/// Keeps the cuBLASLt algorithms this run chose for NVFP4 GEMMs for later runs.
pub fn save_algorithm_cache() {
    let Some(cache) = algorithm_cache() else {
        return;
    };
    match mmh3_cuda::nvfp4::save_algorithms(&cache) {
        Ok(true) => println!("saved the cuBLASLt algorithms to {}", cache.display()),
        Ok(false) => {}
        Err(error) => eprintln!("warning: writing {}: {error}", cache.display()),
    }
}

/// Block-sparse attention settings from `--attention sol|vsa`, `--sparse-tau`, `--sparse-start`
/// and `--vsa-sparsity`, or None for dense attention. Without `--attention`, DiTs with VSA gates
/// (`vsa_gates`) use VSA and others dense attention.
pub fn sparse_attention(
    options: &HashMap<&str, &str>,
    vsa_gates: bool,
) -> Result<Option<mmh3_core::dit::sparse::SparseAttention>, Box<dyn Error>> {
    use mmh3_core::dit::sparse::{SparseAttention, SparseMethod};
    use mmh3_core::dit::vsa::FASTH3_SPARSITY;

    let default = if vsa_gates { "vsa" } else { "dense" };
    match options.get("attention").copied().unwrap_or(default) {
        "dense" => Ok(None),
        "sol" => {
            let defaults = SparseAttention::default();
            Ok(Some(SparseAttention {
                method: SparseMethod::Sol {
                    tau: option_float(options, "sparse-tau", 1.3)?,
                },
                start_fraction: option_float(options, "sparse-start", defaults.start_fraction)?,
                ..defaults
            }))
        }
        "vsa" => {
            // NOTE: parsed as f64, so that the kept tile count rounds like FastVideo's.
            let sparsity = match options.get("vsa-sparsity") {
                Some(value) => value
                    .parse::<f64>()
                    .map_err(|_| "--vsa-sparsity must be a number")?,
                None => FASTH3_SPARSITY,
            };
            if !(0.0..1.0).contains(&sparsity) {
                return Err("--vsa-sparsity must be at least 0 and below 1".into());
            }
            Ok(Some(SparseAttention {
                start_fraction: option_float(options, "sparse-start", 0.0)?,
                ..SparseAttention::vsa(sparsity)
            }))
        }
        other => Err(format!("--attention must be dense, sol or vsa, not {other}").into()),
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
    let directory = models_directory(options).ok_or_else(|| {
        format!("pass --{name} FILE, or a models directory with --models DIR or {MODELS_VARIABLE}")
    })?;
    Ok(directory.join(file).to_string_lossy().into_owned())
}

/// The models directory of `--models` or `MMH3_MODELS`.
fn models_directory(options: &HashMap<&str, &str>) -> Option<PathBuf> {
    match options.get("models") {
        Some(directory) => Some(PathBuf::from(directory)),
        None => std::env::var_os(MODELS_VARIABLE).map(PathBuf::from),
    }
}

/// The file `--name` gives: the path itself when it exists, and otherwise the first of
/// `directories` in the models directory that has a file of that name. None without `--name`.
fn model_file(
    options: &HashMap<&str, &str>,
    name: &str,
    directories: &[&str],
) -> Result<Option<String>, Box<dyn Error>> {
    let Some(value) = options.get(name) else {
        return Ok(None);
    };
    if Path::new(value).exists() {
        return Ok(Some((*value).to_owned()));
    }
    let found = models_directory(options).and_then(|models| {
        directories
            .iter()
            .map(|directory| models.join(directory).join(value))
            .find(|path| path.exists())
    });
    match found {
        Some(path) => Ok(Some(path.to_string_lossy().into_owned())),
        None => Err(format!(
            "--{name} {value}: no such file, and none in the models directory's {}",
            directories.join(" or ")
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_patches_in_patches_then_loras() {
        let models = std::env::temp_dir().join(format!("mmh3-models-{}", std::process::id()));
        for directory in ["patches", "loras"] {
            std::fs::create_dir_all(models.join(directory)).unwrap();
        }
        std::fs::write(models.join("patches/both.safetensors"), b"").unwrap();
        std::fs::write(models.join("loras/both.safetensors"), b"").unwrap();
        std::fs::write(models.join("loras/comfy.safetensors"), b"").unwrap();
        let models_path = models.to_string_lossy().into_owned();
        let lookup = |value: &str| {
            let options = HashMap::from([("models", models_path.as_str()), ("patch", value)]);
            model_file(&options, "patch", &["patches", "loras"])
        };

        assert_eq!(
            lookup("both.safetensors").unwrap().unwrap(),
            models.join("patches/both.safetensors").to_string_lossy()
        );
        assert_eq!(
            lookup("comfy.safetensors").unwrap().unwrap(),
            models.join("loras/comfy.safetensors").to_string_lossy()
        );
        let direct = models.join("loras/comfy.safetensors");
        assert_eq!(
            lookup(&direct.to_string_lossy()).unwrap().unwrap(),
            direct.to_string_lossy()
        );
        assert!(lookup("missing.safetensors").is_err());
        let without = HashMap::from([("models", models_path.as_str())]);
        assert!(
            model_file(&without, "patch", &["patches"])
                .unwrap()
                .is_none()
        );
        std::fs::remove_dir_all(&models).unwrap();
    }
}
