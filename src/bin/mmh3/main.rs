#[cfg(any(feature = "cuda", feature = "metal"))]
mod generate;

use std::error::Error;
use std::process::ExitCode;

const USAGE: &str = "usage:
  mmh3 worker [--listen ADDR] [--models DIR] [--token FILE]
  mmh3 generate (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <video.mp4|video.webm> [--models DIR]
                [--width N] [--height N] [--frames N] [--first-frame FILE] [--last-frame FILE] [--reference FILE]...
                [--reference-audio FILE]... [--reference-video FILE]...
                [--steps N | --schedule taomate] [--seed N] [--shift-video X] [--shift-audio X]
                [--dit FILE] [--video-vae FILE] [--audio-vae FILE] [--text-encoder FILE]
                [--patch FILE] [--lora FILE] [--lora-strength X] [--lora-mode adapter|merge] [--attention dense|sol|vsa] [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--sparse-tau X] [--sparse-start X] [--vsa-sparsity X]
                [--worker HOST[:PORT]]... [--shard-dit N] [--token FILE] [--ffmpeg [FFMPEG_ARGUMENTS...]]

Metal linear precision: mps-fp16 (default), fp16 (MPP), int8 (MPP), or fp32.

Checkpoints default to their ComfyUI names inside a models directory laid out like ComfyUI's models folder and the
Comfy-Org/MiniMax-H3 repository, given by --models or MMH3_MODELS:
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_video_vae_fp16.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
--first-frame and --last-frame take PNG or JPEG pictures the video starts from and ends on. Without --width and
--height the first picture sets the canvas's aspect ratio.
--reference takes a PNG or JPEG picture the prompt refers to as <Picture 1>, <Picture 2> and so on, in the order
of the options, and switches the default DiT to diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors.
--reference-audio does the same with a sound the prompt refers to as <Audio 1>, <Audio 2> and so on, from a WAV,
FLAC, MP3, AAC, ALAC, Ogg Vorbis, MP4 or Matroska file, resampled to the audio VAE's 32 kHz.
--reference-video takes an H.264 MP4 the prompt refers to as <Video 1>, <Video 2> and so on. Its frames are decoded
with NVDEC, and its soundtrack becomes the <Audio j> that precedes it.
MP4 uses NVENC on CUDA or VideoToolbox on Metal for H.264 video, with built-in AAC audio.
WebM uses built-in VP9 + Opus.
--ffmpeg overrides native output and consumes all remaining arguments.
With no ffmpeg arguments it uses H.264 + AAC. Set --out to a .mp4 file.
A one-frame trial encode checks the ffmpeg arguments before generation.
Arguments normally go after the generated inputs and before the output path.
For a full invocation, use whole-argument {video}, {audio}, and {out} placeholders. No shell is invoked.
generate decodes with vae/minimax_h3_video_vae_int8_convrot.safetensors instead of the FP16 video VAE when the models
directory has it.
--shard-dit N splits every DiT step across N machines, Ulysses style, this one and N-1 workers. Over the 200 GbE
between two DGX Sparks a 768p step goes from 13.4 s to 12.2 s, and a step exchanges 34 GB, so it asks for a
worker that reads this machine's memory directly.
--worker borrows another machine running `mmh3 worker`, repeat it for several. A worker that holds the text
encoder encodes the prompt, so this machine never loads it. Anything a worker cannot do, or fails at, this
machine does itself.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result: Result<(), Box<dyn Error>> = match arguments.first().map(String::as_str) {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        Some("generate") => generate::run(&arguments[1..]),
        #[cfg(any(feature = "cuda", feature = "metal"))]
        Some("worker") => mmh3::worker::serve(&arguments[1..]),
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        Some("generate") => {
            Err("this build has no GPU backend. Rebuild with --features cuda on Linux or --features metal on macOS".into())
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    mmh3::cli::exit_code(result)
}
