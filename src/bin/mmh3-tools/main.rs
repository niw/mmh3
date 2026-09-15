mod format;
mod inspect;

#[cfg(feature = "cuda")]
mod bench;
#[cfg(feature = "cuda")]
mod check;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod latent;

use std::error::Error;
use std::process::ExitCode;

const USAGE: &str = "usage:
  mmh3-tools inspect <file.safetensors> [--all]
  mmh3-tools device
  mmh3-tools bench gemm [--tokens N] [--iterations N] [--kinds bf16,fp8,int8,nvfp4,int8-mmh3]
  mmh3-tools bench memory [--megabytes N] [--iterations N]
  mmh3-tools bench mma [--iterations N]
  mmh3-tools bench attention [--tokens N] [--heads N] [--iterations N]
  mmh3-tools bench vsa [--width N] [--height N] [--frames N] [--heads N] [--iterations N]
                   [--attention-precision bf16|int8-fp8]
  mmh3-tools check dit --golden <directory> [--models DIR] [--weights FILE] [--patch FILE] [--lora FILE] [--lora-strength X] [--lora-mode adapter|merge]
                 [--attention dense|sol|vsa] [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--sparse-tau X] [--vsa-sparsity X]
  mmh3-tools check sample --golden <directory> [--models DIR] [--weights FILE] [--patch FILE] [--lora FILE] [--lora-strength X] [--lora-mode adapter|merge]
                    [--attention dense|sol|vsa] [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--sparse-tau X] [--sparse-start X] [--vsa-sparsity X]
  mmh3-tools check video-vae --golden <directory> [--models DIR] [--weights FILE] [--reference FILE]
  mmh3-tools check keyframes --golden <directory> [--models DIR] [--weights FILE] [--reference FILE]
  mmh3-tools check audio-vae --golden <directory> [--models DIR] [--weights FILE]
  mmh3-tools check text-encoder --golden <file.safetensors> [--models DIR] [--weights FILE]
  mmh3-tools latent (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <latents.safetensors>
                    [--models DIR] [--width N] [--height N] [--frames N] [--steps N | --schedule taomate] [--seed N]
                    [--shift-video X] [--shift-audio X] [--dit FILE] [--text-encoder FILE] [--patch FILE] [--lora FILE]
                    [--lora-strength X] [--lora-mode adapter|merge] [--attention dense|sol|vsa]
                    [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--sparse-tau X]
                    [--sparse-start X] [--vsa-sparsity X]

Checkpoints default to their ComfyUI names inside the models directory given by --models or MMH3_MODELS.
latent samples like the generate command of mmh3, with the same defaults, and writes the final video and audio
latents without decoding them.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result: Result<(), Box<dyn Error>> = match arguments.first().map(String::as_str) {
        Some("inspect") => inspect::run(&arguments[1..]),
        #[cfg(feature = "cuda")]
        Some("device") => device::run(),
        #[cfg(feature = "cuda")]
        Some("bench") => bench::run(&arguments[1..]),
        #[cfg(feature = "cuda")]
        Some("check") => check::run(&arguments[1..]),
        #[cfg(feature = "cuda")]
        Some("latent") => latent::run(&arguments[1..]),
        #[cfg(not(feature = "cuda"))]
        Some("device" | "bench" | "check" | "latent") => {
            Err("this build has no GPU backend. Rebuild with --features cuda".into())
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    mmh3::cli::exit_code(result)
}
