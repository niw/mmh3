# mmh3

mmh3 is an inference engine dedicated to [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3),
which generates video with a synchronized stereo soundtrack from text. It is written in Rust with
CUDA C++ kernels and runs without PyTorch, ComfyUI or any other framework: the tokenizer, the
Qwen3-VL text encoder, the diffusion transformer, the sampler and both VAE decoders are implemented
in this repository and tuned for H3's shapes. It loads the ComfyUI checkpoints from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3).

## Status

- Text to video with audio (T2VA) works end to end, one video at a time.
- NVIDIA Blackwell GPUs with CUDA. It is developed on a DGX Spark (GB10, `sm_121`) and builds for
  `sm_120f`, so it should also run on RTX PRO 6000 and RTX 50 series GPUs, which have not been
  tested yet.
- Not yet: first and last frame conditioning (FL2VA), reference conditioning (Ref2VA), an HTTP
  server, and a Metal backend for Apple Silicon.

## Performance

On a DGX Spark at 1344×768 for 124 frames (5.2 s, 37,729 tokens). The DiT stage timings below
use BF16 attention:

| Stage | mmh3 | ComfyUI |
| --- | ---: | ---: |
| DiT step, dense attention | 38.2 s | 49.7 s |
| DiT step, Sol-Attn | 16.9 s | 23.0 s |
| Video VAE decode, FP16 VAE | 30.9 s | 59.2 s |
| Video VAE decode, INT8 ConvRot VAE | 20.1 s | |
| Audio VAE decode | 0.3 s | |
| Text encoder load and encode | 3.4 s | |
| DiT load | 2.5 s | |

With the 4-step Turbo LoRA, INT8/FP8 attention, Sol-Attn from the first step and the INT8 video VAE,
a whole generation from the prompt to the video and audio files takes about **98 seconds**. This
configuration was measured at 98.04 s in one run with the Tokyo rain prompt and seed 1, including
model loading and Y4M/WAV writing, excluding MP4 encoding. BF16 attention with the first step
dense took about 122 s.

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

- INT8 ConvRot linear layers (activations rotated by a Hadamard transform and quantized per row)
  with a dedicated INT8 GEMM that loads its operands with TMA, FlashAttention-2 style attention,
  and an FP32 residual stream.
- Sol-Attn, a training-free block-sparse attention (arXiv:2607.24027), for the long video sequences.
- LoRAs applied at run time on top of the INT8 weights without requantizing them, added inside the
  INT8 GEMM, including the [MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
  for 4-step generation.
- The video VAE decoder with FP16 weights or with the INT8 ConvRot weights of its transformer.
- The official per-stream Euler schedules for video and audio.
- Weights stream from disk to the GPU through direct reads into pinned buffers, without filling the
  page cache.
- Output as YUV4MPEG2 video and WAV audio, written without external libraries.

## Requirements

- Linux with an NVIDIA Blackwell GPU (developed on aarch64).
- The CUDA toolkit with `nvcc` and cuBLASLt (developed with CUDA 13.0).
- Rust with edition 2024 support (developed with 1.98).
- About 54 GB of disk for the checkpoints, 56 GB with the Turbo LoRA. The command loads one model at
  a time, and the largest stage is the DiT with 21 GB of weights plus activations. So far it has
  only run on the DGX Spark's 128 GB of unified memory.

## Build

```sh
cargo build --release --features cuda
```

`CUDA_HOME` points at the CUDA toolkit (default `/usr/local/cuda`) and `MMH3_CUDA_ARCH` sets the GPU
architecture (default `sm_120f`). Without `--features cuda`, only the CPU-side crates build and the
command offers just `inspect`.

## Models

mmh3 finds each checkpoint under its ComfyUI name inside a models directory laid out like ComfyUI's
`models` folder. A ComfyUI installation's `models` directory works as is. To download the files, for
example with the Hugging Face CLI:

```sh
hf download Comfy-Org/MiniMax-H3 --local-dir models \
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors
hf download MiniMaxAI/MiniMax-H3 tokenizer/tokenizer.json --local-dir models

# Optional, for 4-step generation.
hf download lightx2v/Minimax-h3-Turbo \
  minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
  --local-dir models/loras

# Optional, a faster video VAE decoder.
hf download Kijai/MiniMax-H3-experimental minimax_h3_video_vae_int8_convrot.safetensors \
  --local-dir models/vae
```

Pass the directory with `--models DIR` or set `MMH3_MODELS`. Flags such as `--dit`,
`--text-encoder`, `--video-vae`, `--audio-vae` and `--tokenizer` point at individual files instead.

When the models directory has `vae/minimax_h3_video_vae_int8_convrot.safetensors`, the video VAE
with INT8 ConvRot weights in its transformer from Kijai/MiniMax-H3-experimental, `generate` decodes
with it instead of the FP16 VAE. It is faster, and its pixels are about 51 dB PSNR from the FP16
decode, as far as ComfyUI's own INT8 and FP16 decodes are from each other. `--video-vae` picks
either file.

## Usage

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 generate --out out.y4m \
  --prompt "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."
```

This writes `out.y4m` and `out.wav`. mmh3 does not encode MP4 itself. ffmpeg, for example, can mux
the two:

```sh
ffmpeg -i out.y4m -i out.wav -c:v libx264 -crf 18 -pix_fmt yuv420p \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -c:a aac -b:a 192k -shortest out.mp4
```

The fast setting uses the Turbo LoRA with the shifts it was trained with, INT8/FP8 attention,
and Sol-Attn from the first step:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.y4m \
  --steps 4 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

INT8/FP8 attention needs about 0.76 GiB of extra scratch memory at this sequence length and changes
image details. Using `--sparse-start 0` can also change the composition. To keep the first of four
steps dense, use `--sparse-start 0.2`; that setting took 108.51 s with INT8/FP8 attention. Visual
comparison so far covers sampled frames from one prompt and seed. The text refiner, text encoder
and VAEs keep their existing attention precision.

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
- `tools/golden`: development tools that write golden data with a ComfyUI checkout.
- `tools/unicode`: the generator of the tokenizer's Unicode tables.

## Tests

```sh
cargo test --release --workspace --features cuda
```

The tests run the CUDA kernels and models on small fixtures and compare them with ComfyUI's results
and CPU references. The tokenizer test also needs the MiniMax H3 `tokenizer.json`, from
`MMH3_TOKENIZER` or from the `MMH3_MODELS` directory, and is skipped without it.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
