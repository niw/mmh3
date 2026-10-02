# mmh3

mmh3 is an inference engine dedicated to
[MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), written in Rust with CUDA and Metal
kernels and running without PyTorch or ComfyUI. The tokenizer, the Qwen3-VL text encoder, the
diffusion transformer, the sampler and both VAE decoders are written in this repository and tuned
for MiniMax H3's shapes.

It runs on one machine or [distributed across several](docs/distributed.md) running `mmh3 worker`,
splitting every diffusion step between them over RDMA where there is a path for it and ordinary
sockets where there is not. The machine that hands the work out runs none of it, so it needs no GPU
of its own.

The backends are [CUDA](docs/cuda.md) on Linux and Windows (WSL) and [Metal](docs/metal.md) on
macOS, each with its own kernels. Each page says what that backend needs.

## Getting started

With what your backend needs installed, [CUDA](docs/cuda.md) or [Metal](docs/metal.md), run these
commands from the repository directory:

```sh
make
make download-models
make generate
```

`make` builds mmh3 with Metal on macOS and CUDA on Linux and Windows (WSL).
`make download-models` downloads the models into `models`, and `make generate` writes a video with
audio from a sample prompt to `out.mp4`. See [Usage](docs/usage.md) for the other settings.

## Performance

These are the times of `make generate`, including model loading and encoding, with warm model
files. [Performance and accuracy](docs/performance.md) has the times of each stage.

### CUDA on Linux and Windows (WSL)

`make generate` uses FastVideo's 4-step [FastH3](docs/fasth3.md) as a patch on the base DiT, with
its video sparse attention, INT8/FP8 attention and the INT8 video VAE, and writes MP4 with NVENC
H.264 and AAC. Compared with mmh3's defaults, these can change image details and the composition.
mmh3 also runs three other few-step models with the settings on their pages.

On a DGX Spark, a complete generation took:

| Model | Steps | Time per step | Time |
| --- | ---: | ---: | ---: |
| [FastH3](docs/fasth3.md) | 4 | 14 s | 80 s |
| [lightx2v Turbo LoRA](docs/lightx2v-turbo.md) | 4 | 14 s | 81 s |
| [TaoMate-H3](docs/taomate.md) | 3 | 14 s | 66 s |
| [PDMD](docs/pdmd.md) | 2 | 14 s | 52 s |

Each generated a 1344×768, 124-frame video, 5.2 seconds at 24 fps.

### Metal on macOS

`make generate` uses the same 4-step [FastH3](docs/fasth3.md) patch with its video sparse
attention, in a smaller and shorter clip, and writes MP4 with VideoToolbox H.264 and AAC. The DiT,
the text encoder and the video VAE's transformer run on the GPU's matrix units with INT8 products
and FP16 attention, on the Neural Accelerators of an M5 or later. A Mac whose GPU cannot hold a
whole model keeps as many layers as fit and reads the others from the disk as they run.

A complete generation took:

| Mac | Memory | Time per step | Time |
| --- | ---: | ---: | ---: |
| M4 Max | 128 GB | 27 s | 153 s |
| M6 | 24 GB | 12 s | 67 s |

Each generated a 672×384, 73-frame video, 3.0 seconds at 24 fps.

## Status

- Text to video with audio (T2VA) works end to end, one video at a time. Native MP4 and WebM
  generation and complete video/audio decoding have been verified on GB10.
- Videos from a first frame, a last frame or both ([FL2VA](docs/fl2va.md)) work too.
- Videos from reference pictures, sounds and clips ([Ref2VA](docs/ref2va.md)) work with the ref2va
  DiT, reading reference clips from MP4 files.
- NVIDIA GPUs with CUDA. It is developed on a DGX Spark (GB10, `sm_121`) and builds for `sm_120f`,
  so it should also run on RTX PRO 6000 and RTX 50 series GPUs, which have not been tested yet. The
  build also targets Ada (`sm_89`), which runs on an RTX 4090.
- Apple silicon with Metal on macOS 15 or later, tested on an M4 Max and an M6. Text to video with
  audio works with LoRAs, patches such as FastH3's, and Sol-Attn and VSA sparse attention, and a
  Mac with 24 GB runs it by reading the layers that do not fit from the disk. Videos from frames and
  references work on macOS 26. See [Metal](docs/metal.md) for its current limits.
- [Distributed generation](docs/distributed.md) works on both backends and between them, and from a
  machine with neither. Shares follow what each machine measures itself to do. Between two DGX
  Sparks a 768p run takes about a quarter less, and a Mac that hands every step to a Spark rather
  than taking one itself generates about twenty-five times faster than it does alone.
- An [HTTP server](docs/server.md) takes a generation as a form and keeps its models loaded
  between the generations it runs, so the second one against a warm server starts at its first
  step. It has no authentication of its own and waits on loopback.

## Documentation

[docs/README.md](docs/README.md) lists the documentation pages.

## License

mmh3 is released under the MIT License. See [LICENSE](LICENSE), and [NOTICE](NOTICE) for third-party
material and credits. The model weights are not part of this repository and come under their own
licenses.
