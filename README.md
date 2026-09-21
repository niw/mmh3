# mmh3

mmh3 is an inference engine dedicated to
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), written in Rust with CUDA and Metal
kernels and running without PyTorch or ComfyUI. The tokenizer, the Qwen3-VL text encoder, the
diffusion transformer, the sampler and both VAE decoders are written in this repository and tuned
for MiniMax H3's shapes. It reads the ComfyUI checkpoints from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), and writes MP4 with NVENC or
VideoToolbox H.264 and AAC, or WebM with VP9 and Opus, without an ffmpeg installation.

It runs on one machine or [distributed across several](docs/distributed.md) running `mmh3 worker`,
splitting every diffusion step between them over RDMA where there is a path for it and ordinary
sockets where there is not. The machine that hands the work out runs none of it, so it needs no GPU
of its own.

The backends are [CUDA](docs/cuda.md) on Linux and [Metal](docs/metal.md) on macOS, the second
through a Swift bridge and MPS beside its own kernels. Each page says what that backend needs.

## Getting started

With what your backend needs installed, [CUDA](docs/cuda.md) or [Metal](docs/metal.md), run these
commands from the repository directory:

```sh
make
make download-models
make generate
```

`make` builds mmh3 with Metal on macOS and CUDA on Linux. `make download-models` downloads the
models into `models`, and `make generate` writes a video with audio to `out.mp4` from a sample
prompt. On CUDA it generates a 5.2-second 1344×768 video with NVENC H.264 and AAC. On Metal it
generates a 1.625-second 448×256 video with the four-step Turbo LoRA and VideoToolbox output.

On CUDA, `make generate` uses FastVideo's 4-step [FastH3](docs/fasth3.md) as a patch on the base DiT,
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
- Metal text-to-video support on macOS. See [Metal](docs/metal.md) for its current limits.
- [Distributed generation](docs/distributed.md) works on both backends and between them, and from a
  machine with neither. Shares follow what each machine measures itself to do. Between two DGX
  Sparks a 768p run takes about a quarter less, and a Mac that hands every step to a Spark rather
  than taking one itself generates about twenty-five times faster than it does alone.
- Not yet: an HTTP server.

## Documentation

[docs/README.md](docs/README.md) lists the documentation pages.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
