# mmh3

mmh3 is an inference engine dedicated to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
which generates video with a synchronized stereo soundtrack from text. It is written in Rust with
CUDA C++ kernels and runs without PyTorch, ComfyUI or any other framework: the tokenizer, the
Qwen3-VL text encoder, the diffusion transformer, the sampler and both VAE decoders are implemented
in this repository and tuned for H3's shapes. It loads the ComfyUI checkpoints from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Getting started

With the [requirements](#requirements) installed, run these commands from the repository directory:

```sh
make
make download-models
make generate
```

`make` builds mmh3, `make download-models` downloads the models into `models`, and `make generate`
generates a 5.2-second 1344×768 video with audio from a sample prompt and writes `out.y4m`,
`out.wav` and `out.mp4`. On a DGX Spark, the generation takes about **87 seconds**
([details](#performance)). The targets stop with a message when cargo, `hf` or ffmpeg is missing.

`PROMPT`, `SEED`, `OUT` and `MODELS` change the prompt, the seed, the output files and the models
directory. This writes `panda.y4m`, `panda.wav` and `panda.mp4`:

```sh
make generate PROMPT="A red panda sips tea on a sunny wooden porch while birds chirp in the garden." \
  SEED=2 OUT=panda.y4m
```

`make generate` uses the fastest settings measured so far: the INT8 video VAE, the 4-step Turbo
LoRA, INT8/FP8 attention and Sol-Attn from the first step. Compared with mmh3's defaults, they can
change image details and the composition. See [Usage](#usage) for the other settings.

## Status

- Text to video with audio (T2VA) works end to end, one video at a time.
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

The settings of `make generate` took 86.76 s in one run with seed 1, from the prompt to the Y4M and
WAV files, including model loading. With the first step dense it took 98.98 s, and with BF16
attention and the first step dense 116.20 s.

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
- Optional INT8/FP8 attention in the DiT, with FP32 softmax and accumulation.
- Kernels fused for H3's shapes, such as the normalization and quantization of each layer input in
  one pass, and SwiGLU and LoRA inside the INT8 GEMM.
- LoRAs such as the [MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) for
  4-step generation, applied on top of the INT8 weights at run time without requantizing them.
- The video VAE decoder with FP16 weights, or with INT8 ConvRot weights in its transformer.
- The official per-stream Euler schedules for video and audio.
- Weights stream from disk to the GPU with direct reads, without filling the page cache.
- Output as YUV4MPEG2 video and WAV audio, written without external libraries.

## Requirements

- Linux with an NVIDIA Blackwell GPU (developed on aarch64).
- The CUDA toolkit with `nvcc` and cuBLASLt (developed with CUDA 13.0).
- Rust with edition 2024 support (developed with 1.98).
- The Hugging Face CLI (`hf`) to download the models, for example installed with
  `uv tool install huggingface_hub`.
- ffmpeg for the MP4 of `make generate`. mmh3 itself does not need it.
- About 54 GB of disk for the models. mmh3 loads one model at a time. The largest is the DiT, with
  21 GB of weights plus activations.

## Build

```sh
make
```

`make` runs `cargo build --release --features cuda`. `CUDA_HOME` points at the CUDA toolkit (default
`/usr/local/cuda`) and `MMH3_CUDA_ARCH` sets the GPU architecture (default `sm_120f`). Without
`--features cuda`, only the CPU-side crates build and the command offers just `inspect`.

## Models

mmh3 finds each checkpoint under its ComfyUI name in a models directory laid out like ComfyUI's
`models` folder, so a ComfyUI installation's `models` directory works as is. `mmh3` takes the
directory with `--models DIR` or `MMH3_MODELS`, and single files with `--dit`, `--text-encoder`,
`--video-vae`, `--audio-vae` and `--tokenizer`.

`make download-models` runs `tools/download-models.sh`, which downloads these files into `models`:

| File | From |
| --- | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | Kijai/MiniMax-H3-experimental |
| `loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` | lightx2v/Minimax-h3-Turbo |
| `tokenizer/tokenizer.json` | MiniMaxAI/MiniMax-H3 |

With `--precision fp16`, the script downloads the FP16 video VAE,
`vae/minimax_h3_video_vae_fp16.safetensors` from Comfy-Org/MiniMax-H3, instead of the INT8 one.
`--no-lora` skips the LoRA, and `--models DIR` downloads into another directory.

`generate` decodes with the INT8 video VAE when the models directory has it, and with the FP16 one
otherwise. The INT8 one is faster, and its output is about 51 dB PSNR from the FP16 decode, as close
as ComfyUI's own INT8 and FP16 decodes are to each other.

## Usage

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 generate --out out.y4m \
  --prompt "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."
```

This writes `out.y4m` and `out.wav`. mmh3 does not encode MP4 itself. `make generate` muxes the two
with ffmpeg like this:

```sh
ffmpeg -i out.y4m -i out.wav -c:v libx264 -crf 18 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k -shortest out.mp4
```

The fast settings of `make generate` use the Turbo LoRA with the shifts it was trained with,
INT8/FP8 attention, and Sol-Attn from the first step:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.y4m \
  --steps 4 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

INT8/FP8 attention changes image details and needs about 0.76 GiB more memory. `--sparse-start 0`
can also change the composition. `--sparse-start 0.2` keeps the first step dense and took 98.98 s.
So far these settings have been compared visually on one prompt and seed only.

| Option | Default | Meaning |
| --- | --- | --- |
| `--prompt TEXT`, `--prompt-file FILE` | | The prompt, raw text without a chat template. |
| `--width N`, `--height N` | 1344, 768 | Canvas size, multiples of 32. H3 is trained with a 768-pixel short edge. |
| `--frames N` | 124 | Frame count at 24 fps, rounded up to the next 17n + 5. |
| `--steps N` | 20 | Model evaluations. The released checkpoint is guidance-distilled, so there is no CFG. |
| `--seed N` | 0 | Seed of the initial noise. |
| `--shift-video X`, `--shift-audio X` | 12, 3 | Sigma shifts of the two schedules. The 768p Turbo LoRA wants 6 and 3. |
| `--attention dense\|sol` | `dense` | Sol-Attn switches to block-sparse attention. |
| `--attention-precision bf16\|int8-fp8` | `bf16` | INT8 QK / FP8 PV in the DiT, with FP32 softmax and accumulation. Changes generated details. |
| `--sparse-tau X` | 1.3 | Sol-Attn's routing threshold. Higher is sparser. |
| `--sparse-start X` | 0.2 | Fraction of the steps that stay dense before Sol-Attn starts. |
| `--lora FILE`, `--lora-strength X` | none, 1.0 | A ComfyUI LoRA for the DiT. |

H3 follows long, structured prompts well, with the picture, the sound and the music described
separately, one section per line:

```text
integrated_multimodal_description: [Shot 1] Cinematic, wide shot, slow push-in. At golden hour on a rugged coastline, huge waves crash against dark cliffs and throw white spray high into the air. A white lighthouse stands on the headland as its beam begins to sweep.
overall_soundscape: The deep roar of waves breaking on the rocks, gusting wind, and distant seagull calls.
non_diegetic_music: A slow, swelling orchestral string theme.
```

Other commands:

- `mmh3 inspect FILE` summarizes the tensors of a safetensors file.
- `mmh3 device` prints the GPU.
- `mmh3 bench gemm|memory|mma|attention` measures kernels and the GPU.
- `mmh3 check dit|sample|video-vae|audio-vae|text-encoder` compares a stage with golden data written
  by `tools/golden`.

## Repository layout

- `src/main.rs`: the `mmh3` command.
- `crates/mmh3-core`: everything that does not depend on a GPU backend, such as the safetensors
  reader, the tokenizer, the packed token layout, schedules, VAE tiling plans, Sol-Attn's reference
  and the media writers.
- `crates/mmh3-cuda`: the CUDA kernels (`kernels/*.cu`) and the Rust code that runs the models with
  them.
- `crates/mmh3-cpu`: an FP32 CPU reference of the DiT for tests.
- `tests/fixtures`: small random models with outputs computed by ComfyUI's implementation.
- `tools/download-models.sh`: the model downloader that `make download-models` runs.
- `tools/golden`: development tools that write golden data with a ComfyUI checkout.
- `tools/unicode`: the generator of the tokenizer's Unicode tables.

## Tests

```sh
cargo test --release --workspace --features cuda -- --test-threads=1
```

The tests run the CUDA kernels and models on small fixtures and compare them with ComfyUI's results
and CPU references. They run one at a time, because loading a model during another test's DiT
forward pass can corrupt that forward pass. The tokenizer test needs the MiniMax H3 `tokenizer.json` from
`MMH3_TOKENIZER` or the `MMH3_MODELS` directory, and is skipped without it.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
