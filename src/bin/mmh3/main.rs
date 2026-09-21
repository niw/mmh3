use std::error::Error;
use std::process::ExitCode;

const USAGE: &str = "usage:
  mmh3 server [--listen ADDR] [--jobs DIR] [--models DIR] [--worker HOST[:PORT]]... [--local-worker]
              [--vram-budget GB] [--idle-unload SECONDS]
  mmh3 worker [--listen ADDR] [--models DIR] [--token FILE] [--vram-budget GB] [--idle-unload SECONDS]
  mmh3 generate (--prompt TEXT | --prompt-file FILE | --context <text.safetensors>) --out <video.mp4|video.webm> [--models DIR]
                [--width N] [--height N] [--frames N] [--first-frame FILE] [--last-frame FILE] [--reference FILE]...
                [--reference-audio FILE]... [--reference-video FILE]...
                [--steps N | --schedule taomate] [--seed N] [--shift-video X] [--shift-audio X]
                [--dit FILE] [--video-vae FILE] [--audio-vae FILE] [--text-encoder FILE]
                [--patch FILE] [--lora FILE] [--lora-strength X] [--lora-mode adapter|merge] [--attention dense|sol|vsa] [--attention-precision bf16|int8-fp8] [--linear-precision int8|nvfp4] [--sparse-tau X] [--sparse-start X] [--vsa-sparsity X]
                [--worker HOST[:PORT] [--worker-units UNITS]]... [--local-worker [--worker-units UNITS]]
                [--token FILE] [--vram-budget GB] [--ffmpeg [FFMPEG_ARGUMENTS...]]

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
--worker borrows another machine running `mmh3 worker`, repeat it for several. --local-worker starts one in this
process, so that this machine's own GPU takes part as a worker rather than as the leader. A leader reads no
model at all: it hands out the prompt, every step and every chunk of the decode, and puts what comes back
together. What no machine is asked for, it does itself.
--worker-units names what the machine before it may be asked for, out of steps, prompt, video and audio,
separated by commas. Without it a machine may be asked for anything it serves. Every step goes to the machines
allowed steps, in the order given, which is the order their ranks are numbered in. Where none is allowed them,
every step runs here.
--vram-budget GB holds a run to that much device memory, failing an allocation past it as a device that small
would. A worker lets go of the model it has used least whenever an allocation finds no memory left.
--idle-unload SECONDS lets a worker go of every model it holds after that long with nothing to do, so that a
machine nobody is generating on is a machine with its memory back. Without it a worker holds what it loaded.
A machine that cannot be reached, or that holds no checkpoint for what it was asked, is passed over. One that
answered and then failed ends the run.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let result: Result<(), Box<dyn Error>> = match arguments.first().map(String::as_str) {
        Some("generate") => mmh3::generate::run(&arguments[1..], USAGE),
        #[cfg(any(feature = "cuda", feature = "metal"))]
        Some("worker") => mmh3::worker::serve(&arguments[1..]),
        #[cfg(feature = "server")]
        Some("server") => mmh3::server::serve(&arguments[1..]),

        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    mmh3::cli::exit_code(result)
}
