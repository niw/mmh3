# mmh3

mmh3 is an inference engine dedicated to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
which generates video with a synchronized stereo soundtrack from text. It is written in Rust with
CUDA C++ kernels and runs without PyTorch, ComfyUI or any other framework: the tokenizer, the
Qwen3-VL text encoder, the diffusion transformer, the sampler and both VAE decoders are implemented
in this repository and tuned for H3's shapes. It loads the ComfyUI checkpoints from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3). It saves MP4 directly with
NVENC H.264 video and CPU-encoded AAC audio, or WebM with VP9 and Opus. Both formats work
without an ffmpeg installation.

## Getting started

With the [requirements](#requirements) installed, run these commands from the repository directory:

```sh
make
make download-models
make generate
```

`make` builds mmh3, `make download-models` downloads the models into `models`, and `make generate`
generates a 5.2-second 1344×768 video with audio from a sample prompt and writes `out.mp4`
(NVENC H.264 video and AAC audio). A complete MP4 generation on a DGX Spark took about
**86 seconds**, including model loading and encoding ([details](#performance)). The targets stop
with a message when cargo or `hf` is missing.

`PROMPT`, `SEED`, `OUT` and `MODELS` change the prompt, the seed, the output file and the models
directory. This writes `panda.mp4`:

```sh
make generate PROMPT="A red panda sips tea on a sunny wooden porch while birds chirp in the garden." \
  SEED=2 OUT=panda.mp4
```

To save VP9 + Opus WebM, use `make generate FEATURES=cuda,webm OUT=out.webm`. The extension
selects the native encoder. Neither command launches ffmpeg or writes intermediate Y4M/WAV files.

`make generate` uses the fastest settings measured so far: the INT8 video VAE, the 4-step Turbo
LoRA, INT8/FP8 attention and Sol-Attn from the first step. Compared with mmh3's defaults, they can
change image details and the composition. See [Usage](#usage) for the other settings.

## Status

- Text to video with audio (T2VA) works end to end, one video at a time. Native MP4 and WebM
  generation and complete video/audio decoding have been verified on GB10.
- NVIDIA Blackwell GPUs with CUDA. It is developed on a DGX Spark (GB10, `sm_121`) and builds for
  `sm_120f`, so it should also run on RTX PRO 6000 and RTX 50 series GPUs, which have not been
  tested yet.
- Not yet: first and last frame conditioning (FL2VA), reference conditioning (Ref2VA), an HTTP
  server, and a Metal backend for Apple Silicon.

## Performance

The times below are from a DGX Spark at 1344×768 and 124 frames (5.2 s), with the Tokyo rain prompt
and the Turbo LoRA (37,770 tokens in all). ComfyUI's times come from random inputs with 1,000 text
tokens (38,710 tokens in all):

| Stage | mmh3 | ComfyUI |
| --- | ---: | ---: |
| DiT step, dense BF16 attention | 38.7 s | 49.7 s |
| DiT step, dense INT8/FP8 attention | 26.7 s | |
| DiT step, Sol-Attn in BF16 | 16.5 s | 23.0 s |
| DiT step, Sol-Attn in INT8/FP8 | 14.6 s | |
| Video VAE decode, FP16 VAE | 30.9 s | 59.2 s |
| Video VAE decode, INT8 ConvRot VAE | 20.2 s | |
| Audio VAE decode | 0.3 s | |
| Text encoder load and encode | 3.4 s | |
| DiT load | 2.9 s | |

A complete run with the settings of `make generate`, the garden red panda prompt (75 text tokens),
and seed 42 took **86.29 s for MP4** and **90.88 s for WebM**, including model loading and native
encoding. Both runs produced 1344×768, 124-frame clips with stereo audio, and both files were
verified by decoding the entire video and audio streams. These are individual runs, not averages.

With the Tokyo rain prompt and seed 1, the same settings took 86.76 s without the final video and
audio encoding, 98.98 s with the first step dense, and 116.20 s with BF16 attention and the first
step dense.

## Accuracy

mmh3 is checked against ComfyUI's implementation, stage by stage, with the golden data tools in
`tools/golden`:

- The DiT deviates from ComfyUI's default BF16 run by less than ComfyUI's own BF16 and FP32 runs
  deviate from each other.
- The video VAE decoder matches ComfyUI's FP16 decode at 65 dB PSNR, and the audio VAE decoder
  matches a strict FP32 decode at 108 dB SNR.
- The tokenizer gives the same token ids as Hugging Face tokenizers.
- Sol-Attn is an approximation by design, and mmh3's differs from dense attention as much as
  ComfyUI's does.

## Features

- INT8 ConvRot linear layers (activations rotated by a Hadamard transform and quantized per row) on
  a dedicated INT8 GEMM, FlashAttention-2 style attention, and an FP32 residual stream.
- Sol-Attn, a training-free block-sparse attention (arXiv:2607.24027), for the long video sequences.
- FastVideo's
  [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) 4-step
  model with its video sparse attention (VSA), as a patch on the base DiT.
- Optional INT8/FP8 attention in the DiT, with FP32 softmax and accumulation.
- Kernels fused for H3's shapes, such as the normalization and quantization of each layer input in
  one pass, and SwiGLU and LoRA inside the INT8 GEMM.
- LoRAs such as the [MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) for
  4-step generation, applied on top of the INT8 weights at run time without requantizing them.
- The video VAE decoder with FP16 weights, or with INT8 ConvRot weights in its transformer.
- The official per-stream Euler schedules for video and audio.
- Weights stream from disk to the GPU with direct reads, without filling the page cache.
- Native NVENC H.264 + AAC MP4 output, built-in VP9 + Opus WebM, or an external ffmpeg CLI.

## Requirements

- Linux with an NVIDIA Blackwell GPU (developed on aarch64).
- The CUDA toolkit with `nvcc` and cuBLASLt (developed with CUDA 13.0).
- Rust with edition 2024 support (developed with 1.98).
- The Hugging Face CLI (`hf`) to download the models, for example installed with
  `uv tool install huggingface_hub`.
- A C/C++ toolchain for the CUDA kernels and NVENC adapter.
- With the optional `webm` feature, CMake for the bundled Opus encoder, plus
  `curl`, `tar`, and `sha256sum` (Linux). The VP9 binding downloads
  a versioned libvpx static library and verifies its checksum during the build. This needs
  network access to GitHub Releases. Its Linux prebuilt libraries support Ubuntu 22.04/24.04,
  x86-64 and aarch64. A separate runtime libvpx installation is not needed.
- Native MP4 requires the `mp4` feature, H.264 NVENC and a driver supporting NVENC API 12.2.
  The driver is loaded at runtime. No separately installed Video Codec SDK is required. GPUs
  without NVENC can use WebM or ffmpeg output with a build without `mp4`.
- About 54 GB of disk for the models. mmh3 loads one model at a time. The largest is the DiT, with
  21 GB of weights plus activations.

ffmpeg is optional: install it only to use the explicit `--ffmpeg` output path. Developers
running the output integration tests also need ffmpeg and ffprobe as independent decoders.

## Build

```sh
make
```

`make` builds `mmh3` with CUDA, native MP4 output and the ffmpeg CLI output:
`cargo build --release --features cuda,mp4 --bin mmh3`. `make FEATURES=cuda,webm` builds
native WebM output instead of MP4, for GPUs without NVENC. It needs the VP9/Opus dependencies
and the libvpx download. `make FEATURES=cuda,mp4,webm` builds both.
`CUDA_HOME` points at the CUDA toolkit (default `/usr/local/cuda`) and `MMH3_CUDA_ARCH` sets the GPU
architecture (default `sm_120f`). Without `--features cuda`, both commands can build with the CPU-side
crates. In that case, only `mmh3-tools inspect` is available.

Build the inspection and development tools separately with:

```sh
cargo build --release --features cuda --bin mmh3-tools
```

To run a specific command through Cargo:

```sh
cargo run --release --features cuda,mp4 --bin mmh3 -- generate --prompt "A rainy street." --out out.mp4
cargo run --release --features cuda --bin mmh3-tools -- device
```

`cargo run` defaults to `mmh3` when `--bin` is omitted.

## Models

mmh3 finds each checkpoint under its ComfyUI name in a models directory laid out like ComfyUI's
`models` folder, so a ComfyUI installation's `models` directory works as is. `mmh3` takes the
directory with `--models DIR` or `MMH3_MODELS`, and single files with `--dit`, `--text-encoder`,
`--video-vae` and `--audio-vae`.

`make download-models` runs `tools/download-models.sh`, which downloads these files into `models`:

| File | From |
| --- | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | Kijai/MiniMax-H3-experimental |
| `loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` | lightx2v/Minimax-h3-Turbo |

With `--precision fp16`, the script downloads the FP16 video VAE,
`vae/minimax_h3_video_vae_fp16.safetensors` from Comfy-Org/MiniMax-H3, instead of the INT8 one.
`--no-lora` skips the LoRA, and `--models DIR` downloads into another directory.

Patches, such as the FastH3 patch below, go into `patches` of the models directory. `--patch` and
`--lora` take a path, or a file name that mmh3 looks up in `patches` and then `loras` of the models
directory for `--patch`, and in `loras` for `--lora`, so patches kept with ComfyUI's LoRAs work too.

`generate` decodes with the INT8 video VAE when the models directory has it, and with the FP16 one
otherwise. The INT8 one is faster, and its output is about 51 dB PSNR from the FP16 decode, as close
as ComfyUI's own INT8 and FP16 decodes are to each other.

## Usage

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 generate --out out.mp4 \
  --prompt "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."
```

This writes `out.mp4` using H.264 High profile (NVENC P4, constant QP 20, no B frames,
keyframes at most two seconds apart) and AAC-LC at 192 kb/s. The output path extension
selects the container. `--ffmpeg` explicitly overrides native output selection.

The video VAE retains RGB float32 pixels on the GPU. CUDA converts one frame at a time
directly into a pitched NV12 allocation registered with NVENC. Only compressed video is
read back to the CPU. The allocation is synchronized before submission and reused only
after NVENC has finished reading it. This path makes no assumption about shared system
memory and supports the same design on discrete GPUs such as RTX 5090. GB10 has been
tested. RTX 5090 has not yet been tested on hardware.

Audio is encoded on the CPU using `rusty_aac`, preserving the VAE's 32 kHz sample rate.
Audio is trimmed or silence-padded to the video duration. MP4 edit lists compensate for
AAC priming and end padding. The MP4 header precedes media data for progressive playback.
A decoder exporting raw PCM may still return the last AAC block's padding. The track's
presentation duration excludes it.

The encoder and output file are validated before model loading. Unsupported hardware,
missing drivers, unsupported dimensions or unavailable build features produce an early
error. There is no automatic switch from a requested MP4 to WebM or ffmpeg.

For the software path, build with the `webm` feature and use `--out out.webm`. Video is
encoded by libvpx (VP9 profile 0, CPU-used 4, fixed quantizer 30 on the 0–63 scale, 8-bit
YUV420 with BT.709 limited-range colors). These encoder settings are currently library defaults, not CLI options.
Audio is encoded by libopus at 192 kb/s after resampling from 32 to 48 kHz. Audio is trimmed
or padded with silence to the video duration. Resampler and Opus delays are compensated. The
file includes duration and keyframe seeking information.

To use the ffmpeg CLI instead, append `--ffmpeg`. With no further arguments it uses the
H.264 (libx264, CRF 18) + AAC (192 kb/s) recipe:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 --ffmpeg
```

All arguments after `--ffmpeg` belong to ffmpeg. Put every mmh3 option before it. Custom
arguments replace the default encoding recipe, and are inserted after the generated video
and audio inputs (`0:v` and `1:a`) and before the output filename:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 \
  --ffmpeg -c:v libx264 -crf 20 -preset slow -vf scale=960:-2 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 160k -shortest
```

For control of input options, mapping, and the entire argument order, use the whole-argument
placeholders `{video}`, `{audio}`, and `{out}`. A template must include `{out}`. Either input
may be omitted. In template mode, automatic input and output arguments are disabled:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 \
  --ffmpeg -i '{video}' -i '{audio}' -map 0:v -map 1:a \
  -c:v libx264 -crf 18 -c:a aac -shortest '{out}'
```

Arguments are passed directly to ffmpeg, without a shell. Quote individual arguments as
needed. Do not combine all ffmpeg arguments into one quoted string. Shell pipelines and
redirection are not interpreted. ffmpeg runs with `-nostdin -y`. Before model loading, mmh3
runs the same arguments on one black, silent frame and writes the result to a temporary file
with the output's extension. Unknown options or codecs, and codecs that the container does
not accept, are reported before generation. Generated Y4M/WAV inputs are temporary and
removed on success or failure. The output is written to a temporary file next to the
destination (with the same extension) and replaces it only after successful encoding.
Additional output paths explicitly supplied in ffmpeg arguments are managed by ffmpeg itself,
and the trial encode writes them as well.

The `mmh3-output` crate owns platform-independent `VideoEncoder` / `AudioEncoder`
interfaces, `Mp4Session`, and the MP4 writer. `VideoEncoder` has an associated native frame
type, so the common layer never casts device handles or requires a CPU YUV buffer.
`mmh3-output-nvenc` owns the NVENC input allocation, CUDA synchronization, driver session,
and conversion of the H.264 bitstream into MP4 samples. The application selects and
prepares output, submits video and audio, then finishes the file. WebM and ffmpeg output
implement the host-input `OutputBackend` interface behind the same lifecycle.

An Apple adapter can implement `VideoEncoder` using Metal-compatible CVPixelBuffer /
IOSurface input and VideoToolbox, and `AudioEncoder` using AudioToolbox. It can reuse the
same MP4 writer, sample timing, edit lists and destination publication. Apple generation
and output adapters are not implemented.

The `cuda` feature builds generation with the ffmpeg CLI output, which every generation build
has. The `mp4` and `webm` features add native output formats. `mp4` encodes video with NVENC
and implies `cuda`. No feature is enabled by default. Examples:

```sh
# Native MP4 and ffmpeg, without the VP9/Opus dependencies or the libvpx download.
cargo build --release --features cuda,mp4 --bin mmh3
# Native WebM and ffmpeg, for GPUs without NVENC.
cargo build --release --features cuda,webm --bin mmh3
# ffmpeg only.
cargo build --release --features cuda --bin mmh3
# Native MP4, native WebM and ffmpeg.
cargo build --release --features cuda,mp4,webm --bin mmh3
```

The encoder dependencies have their own licenses. `rusty_aac` is Apache-2.0, libvpx and
libopus are BSD-3-Clause, `shiguredo_libvpx` is Apache-2.0, the Rust Opus bindings are
MIT/Apache-2.0, and rubato is MIT. The vendored NVENC API header is MIT-licensed and its
notices are retained. The NVIDIA driver is loaded from the installed system. Codec patent
terms remain separate from software licenses. No libx264 or FFmpeg implementation is
linked into the native MP4 path.
See [rusty_aac](https://crates.io/crates/rusty_aac),
[libvpx](https://github.com/webmproject/libvpx),
[the VP9 Rust binding](https://github.com/shiguredo/libvpx-rs), and
[Opus](https://opus-codec.org/license/).

Output integration tests use ffmpeg/ffprobe as independent decoders and do not load models.
The portable tests need no CUDA. The native output tests require NVENC:

```sh
cargo test -p mmh3-output --features mp4,webm -- --include-ignored
# Requires an NVIDIA GPU with NVENC and also tests GPU NV12 conversion.
cargo test -p mmh3-output-nvenc --test output -- --ignored
```

The fast settings of `make generate` use the Turbo LoRA with the shifts it was trained with,
INT8/FP8 attention, and Sol-Attn from the first step:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

### FastH3

FastVideo's FastH3 VSA-DataFree generates in four steps with the base schedule and video sparse
attention (VSA), which mmh3 selects for DiTs with VSA gates. It runs as a patch on the base DiT
with `--patch`:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --attention-precision int8-fp8 \
  --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
```

[tools/models](tools/models/README.md) describes the patch and how to build it.

INT8/FP8 attention changes image details and needs about 0.76 GiB more memory. `--sparse-start 0`
can also change the composition. `--sparse-start 0.2` keeps the first step dense and took 98.98 s.
So far these settings have been compared visually on one prompt and seed only.

| Option | Default | Meaning |
| --- | --- | --- |
| `--out FILE` | required (`make generate` uses `out.mp4`) | `.mp4` uses native NVENC H.264 + AAC in an `mp4` build. `.webm` uses native VP9 + Opus in a `webm` build. |
| `--ffmpeg [ARGS...]` | off | Use an installed ffmpeg instead. All following arguments belong to ffmpeg. |
| `--prompt TEXT`, `--prompt-file FILE` | | The prompt, raw text without a chat template. |
| `--width N`, `--height N` | 1344, 768 | Canvas size, multiples of 32. H3 is trained with a 768-pixel short edge. |
| `--frames N` | 124 | Frame count at 24 fps, rounded up to the next 17n + 5. |
| `--steps N` | 20 | Model evaluations. The released checkpoint is guidance-distilled, so there is no CFG. |
| `--seed N` | 0 | Seed of the initial noise. |
| `--shift-video X`, `--shift-audio X` | 12, 3 | Sigma shifts of the two schedules. The 768p Turbo LoRA wants 6 and 3. |
| `--attention dense\|sol\|vsa` | `vsa` with VSA gates, `dense` otherwise | Sol-Attn switches to block-sparse attention. VSA is the sparse attention FastH3 was trained with. |
| `--attention-precision bf16\|int8-fp8` | `bf16` | INT8 QK / FP8 PV in the DiT, with FP32 softmax and accumulation. Changes generated details. |
| `--sparse-tau X` | 1.3 | Sol-Attn's routing threshold. Higher is sparser. |
| `--sparse-start X` | 0.2 for Sol-Attn, 0 for VSA | Fraction of the steps that stay dense before the sparse attention starts. |
| `--vsa-sparsity X` | 0.9 | Fraction of the video tiles VSA leaves out for each query tile. |
| `--patch FILE` | none | A patch for the DiT, such as the FastH3 patch, applied before a LoRA. |
| `--lora FILE`, `--lora-strength X` | none, 1.0 | A ComfyUI LoRA for the DiT. |

H3 follows long, structured prompts well, with the picture, the sound and the music described
separately, one section per line:

```text
integrated_multimodal_description: [Shot 1] Cinematic, wide shot, slow push-in. At golden hour on a rugged coastline, huge waves crash against dark cliffs and throw white spray high into the air. A white lighthouse stands on the headland as its beam begins to sweep.
overall_soundscape: The deep roar of waves breaking on the rocks, gusting wind, and distant seagull calls.
non_diegetic_music: A slow, swelling orchestral string theme.
```

Inspection and development commands are available in `mmh3-tools`:

- `mmh3-tools inspect FILE` summarizes the tensors of a safetensors file.
- `mmh3-tools device` prints the GPU.
- `mmh3-tools bench gemm|memory|mma|attention` measures kernels and the GPU.
- `mmh3-tools check dit|sample|video-vae|audio-vae|text-encoder` compares a stage with golden data written
  by `tools/golden`.

## Repository layout

- `src/bin/mmh3/`: the `mmh3` entry point and video generation.
- `src/bin/mmh3-tools/`: the `mmh3-tools` entry point, checkpoint inspection, benchmarks and reference checks.
- `src/lib.rs`: support shared by both commands, with argument parsing in `src/cli.rs` and checkpoint loading
  options in `src/models.rs`.
- `crates/mmh3-core`: everything that does not depend on a GPU backend, such as the safetensors
  reader, the tokenizer, the packed token layout, schedules, VAE tiling plans, Sol-Attn's reference
  and the media writers.
- `crates/mmh3-cuda`: the CUDA kernels (`kernels/*.cu`) and the Rust code that runs the models with
  them.
- `crates/mmh3-cpu`: an FP32 CPU reference of the DiT for tests.
- `crates/mmh3-output`: portable encoder interfaces, AAC encoding, MP4/WebM muxing and ffmpeg CLI output.
- `crates/mmh3-output-nvenc`: the NVENC adapter that accepts CUDA frames for native H.264 encoding.
- `tests/fixtures`: small random models with outputs computed by ComfyUI's implementation.
- `tools/download-models.sh`: the model downloader that `make download-models` runs.
- `tools/golden`: development tools that write golden data with a ComfyUI checkout.
- `tools/models`: development tools that build model files, such as the FastH3 patch.
- `tools/unicode`: the generator of the tokenizer's Unicode tables.

## Tests

```sh
cargo test --release --workspace --features cuda,mp4
```

The tests run the CUDA kernels and models on small fixtures and compare them with ComfyUI's results
and CPU references.

## Formatting

```sh
make format
```

`make format` formats the Rust code with rustfmt, the Python tools with ruff and the C++ and CUDA
code with clang-format. ruff and clang-format run through `uvx` of [uv](https://docs.astral.sh/uv/)
at pinned versions, so they need no separate installation.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
