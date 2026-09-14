#[cfg(feature = "cuda")]
mod generate;

use std::error::Error;
use std::process::ExitCode;

const USAGE: &str = "usage:
  mmh3 generate (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <video.mp4|video.webm> [--models DIR]
                [--width N] [--height N] [--frames N] [--steps N] [--seed N] [--shift-video X] [--shift-audio X]
                [--dit FILE] [--video-vae FILE] [--audio-vae FILE] [--text-encoder FILE]
                [--lora FILE] [--lora-strength X] [--attention dense|sol|vsa] [--attention-precision bf16|int8-fp8] [--sparse-tau X] [--sparse-start X] [--vsa-sparsity X]
                [--ffmpeg [FFMPEG_ARGUMENTS...]]

Checkpoints default to their ComfyUI names inside a models directory laid out like ComfyUI's models folder and the
Comfy-Org/MiniMax-H3 repository, given by --models or MMH3_MODELS:
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
MP4 uses NVENC H.264 + AAC on CUDA; WebM uses built-in VP9 + Opus.
--ffmpeg overrides native output and consumes all remaining arguments.
With no ffmpeg arguments it uses H.264 + AAC. Set --out to a .mp4 file.
A one-frame trial encode checks the ffmpeg arguments before generation.
Arguments normally go after the generated inputs and before the output path.
For a full invocation, use whole-argument {video}, {audio}, and {out} placeholders. No shell is invoked.
generate decodes with vae/minimax_h3_video_vae_int8_convrot.safetensors instead of the FP16 video VAE when the models
directory has it.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result: Result<(), Box<dyn Error>> = match arguments.first().map(String::as_str) {
        #[cfg(feature = "cuda")]
        Some("generate") => generate::run(&arguments[1..]),
        #[cfg(not(feature = "cuda"))]
        Some("generate") => {
            Err("this build has no GPU backend. Rebuild with --features cuda".into())
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    mmh3::cli::exit_code(result)
}
