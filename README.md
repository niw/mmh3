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
generates a 5.2-second 1344×768 video with audio from a sample prompt and writes `out.mp4` (NVENC
H.264 video and AAC audio).

`make generate` uses FastVideo's 4-step [FastH3](docs/fasth3.md) as a patch on the base DiT,
with its video sparse attention, INT8/FP8 attention and the INT8 video VAE. Compared with mmh3's
defaults, they can change image details and the composition. mmh3 also runs two other few-step
models with the settings on their pages. On a DGX Spark, a complete generation, including model
loading and encoding, took:

| Model | Steps | Time |
| --- | ---: | ---: |
| [FastH3](docs/fasth3.md) | 4 | 87 s |
| [lightx2v Turbo LoRA](docs/lightx2v-turbo.md) | 4 | 86 s |
| [TaoMate-H3](docs/taomate.md) | 3 | 70 s |

See [Usage](docs/usage.md) for the other settings.

## Status

- Text to video with audio (T2VA) works end to end, one video at a time. Native MP4 and WebM
  generation and complete video/audio decoding have been verified on GB10.
- Videos from a first frame, a last frame or both ([FL2VA](docs/fl2va.md)) work too.
- NVIDIA Blackwell GPUs with CUDA. It is developed on a DGX Spark (GB10, `sm_121`) and builds for
  `sm_120f`, so it should also run on RTX PRO 6000 and RTX 50 series GPUs, which have not been
  tested yet.
- Videos from reference pictures, sounds and clips ([Ref2VA](docs/ref2va.md)) work with the ref2va
  DiT, reading reference clips from MP4 files.
- Not yet: an HTTP server and a Metal backend for Apple Silicon.

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
- About 55 GB of disk for the models. mmh3 loads one model at a time. The largest is the DiT, with
  21 GB of weights plus activations.

ffmpeg is optional: install it only to use the explicit `--ffmpeg` output path. Developers
running the output integration tests also need ffmpeg and ffprobe as independent decoders.

## Documentation

[docs/README.md](docs/README.md) lists the documentation pages.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
