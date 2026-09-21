# Development

## Implementation

- INT8 ConvRot linear layers (activations rotated by a Hadamard transform and quantized per row) on
  a dedicated INT8 GEMM, FlashAttention-2 style attention, and an FP32 residual stream.
- Sol-Attn, a training-free block-sparse attention (arXiv:2607.24027), for the long video sequences.
- FastVideo's
  [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree) 4-step
  model with its video sparse attention (VSA), as a patch on the base DiT.
- Optional INT8/FP8 attention in the DiT, with FP32 softmax and accumulation.
- Kernels fused for MiniMax H3's shapes, such as the normalization and quantization of each layer
  input in one pass, and SwiGLU and LoRA inside the INT8 GEMM.
- Experimental NVFP4 linear layers on cuBLASLt's block-scaled FP4 GEMM, requantized from the INT8
  weights at load time, with LoRAs as extra GEMM columns.
- LoRAs such as the [MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) for
  4-step generation, applied on top of the INT8 weights at run time without requantizing them.
- The video VAE decoder with FP16 weights, or with INT8 ConvRot weights in its transformer.
- The official per-stream Euler schedules for video and audio.
- Weights stream from disk to the GPU with direct reads, without filling the page cache.
- Native NVENC or VideoToolbox H.264 + AAC MP4 output, built-in VP9 + Opus WebM, or an external ffmpeg CLI.

## Repository layout

- `src/bin/mmh3/`: the `mmh3` entry point and video generation.
- `src/bin/mmh3-tools/`: the `mmh3-tools` entry point, checkpoint inspection, benchmarks and
  reference checks.
- `src/lib.rs`: support shared by both commands, with argument parsing in `src/cli.rs`,
  checkpoint loading options in `src/models.rs` and the sampling in `src/generation.rs`.
- `crates/mmh3-core`: everything that does not depend on a GPU backend, such as the safetensors
  reader and writer, the tokenizer, the packed token layout, schedules, VAE tiling plans, Sol-Attn's
  reference and the media writers.
- `crates/mmh3-cuda`: the CUDA kernels (`kernels/*.cu`) and the Rust code that runs the models with
  them.
- `crates/mmh3-metal`: the Metal backend, with a Swift resource/MPS bridge and
  `kernels/*.metal`. See [Metal](metal.md) for usage and supported options.
- `crates/mmh3-cpu`: an FP32 CPU reference of the DiT for tests.
- `crates/mmh3-output`: portable encoder interfaces, AAC encoding, MP4/WebM muxing and ffmpeg CLI
  output.
- `crates/mmh3-output-nvenc`: the NVENC adapter that accepts CUDA frames for native H.264 encoding.
- `crates/mmh3-output-videotoolbox`: native H.264 encoding on macOS, with a Swift VideoToolbox
  bridge and Metal RGB-to-NV12 conversion into shared pixel buffers.
- `tests/fixtures`: small random models with outputs computed by ComfyUI's implementation.
- `tools/download-models.sh`: the model downloader that `make download-models` runs.
- `tools/golden`: development tools that write golden data with a ComfyUI checkout.
- `tools/models`: development tools that build model files, such as the FastH3 patch.
- `tools/unicode`: the generator of the tokenizer's Unicode tables.

## mmh3-tools

Inspection and development commands are available in `mmh3-tools`:

- `mmh3-tools inspect FILE` summarizes the tensors of a safetensors file.
- `mmh3-tools device` prints the GPU.
- `mmh3-tools bench gemm|memory|mma|attention` measures kernels and the GPU.
- `mmh3-tools check dit|sample|video-vae|audio-vae|text-encoder` compares a stage with golden data
  written by `tools/golden`.
- `mmh3-tools latent` samples like `mmh3 generate` and writes the final latents without decoding
  them, for example as calibration input of `tools/models/video_vae_int8.py`.

## Tests

```sh
cargo test --release --workspace --features cuda,mp4
```

The tests run the CUDA kernels and models on small fixtures and compare them with ComfyUI's results
and CPU references. The CUDA crates are empty off Linux, so `cargo test --workspace` needs no CUDA
toolchain there. On macOS, `--features metal` takes the place of `cuda,mp4`.

## Formatting

```sh
make format
```

`make format` formats the Rust code with rustfmt, the Python tools with ruff, the Swift code with
swiftformat and the C++, CUDA and Metal code with clang-format. ruff and clang-format run through
`uvx` of [uv](https://docs.astral.sh/uv/) at pinned versions, so they need no separate
installation. swiftformat only runs where it is installed, since the Swift code builds on macOS
alone.
