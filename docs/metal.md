# Metal on macOS

The `metal` backend runs text-to-video generation on Apple silicon, using the same model
checkpoints as the CUDA backend. MP4 output uses VideoToolbox for H.264 video and AAC audio.
External encoding through ffmpeg is optional.

## Requirements

- An Apple silicon Mac with macOS 15 or later.
- Xcode 26 or later command-line tools, for the Swift bridge.
- Rust with edition 2024 support.
- About 55 GB of disk for the models. A run loads one model at a time. The largest is the DiT, with
  21 GB of weights plus activations. See [Models](models.md) for the checkpoints.

A Mac whose GPU may not hold a whole model runs it anyway. The text encoder and the DiT load whole
when they fit in the share of memory the GPU may fill. When one does not, it keeps as many of its
layers as fit beside what the run computes in, and reads the others from the disk each time they
run, while the layers before them compute. A line such as `keeping 25 of 52 DiT blocks on the
device and reading the rest as they run` says how many it kept. The text encoder also leaves its
embedding table on the disk and reads the rows of the prompt from there. A 24 GB Mac generates the
clip of `make generate` this way within 18 GB.

A worker is the exception: it keeps what it has read for the runs that follow, so a Mac serving one
holds several models at once and lets go of the one it used least when the device has no memory
left. `--vram-budget GB` holds it to less than the device would give, and `--idle-unload SECONDS`
gives the memory back when nothing is asking.

MP4 output needs nothing further, since VideoToolbox is part of the system. ffmpeg is optional and
only for the explicit `--ffmpeg` path. A machine that only hands work out to others needs none of
this. See [Distributed generation](distributed.md).

## Build and run

On macOS, the Makefile selects Metal automatically:

```sh
make download-models
make generate
```

This downloads the base models and Turbo LoRA, then generates a 448×256, 39-frame clip using
four steps. Change the prompt and output path with:

```sh
make generate PROMPT="A rainy street." OUT=rain.mp4
```

To build and run directly:

```sh
cargo build --release --features metal --bins
target/release/mmh3-tools device

target/release/mmh3 generate \
  --models models --prompt "A rainy street." \
  --width 448 --height 256 --frames 39 --steps 4 \
  --shift-video 6 --shift-audio 3 \
  --lora minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
  --out rain.mp4
```

Select either `metal` or `cuda`. Metal includes native MP4 output. The separate `mp4` Cargo
feature selects CUDA/NVENC. Use `--ffmpeg` for an external encoder override. See [Output](output.md).

## Options and current limits

The default is dense FP32 attention with MPS FP16 linear products. The DiT's INT8-weight layers
can use another product mode with `--linear-precision`:

| Value | Mode |
| --- | --- |
| `mps-fp16` | FP16 through MPS (default) |
| `fp16` | FP16 through MPP (macOS 26+) |
| `int8` | Experimental INT8 through MPP (macOS 26+) |
| `fp32` | FP32 reference mode |

Changing precision can change the generated sample even with the same seed. These options
apply to the DiT's INT8-weight layers. Other layers and LoRA adapters retain FP32 computation.

Text prompts, precomputed `--context` tensors and adapter LoRAs are supported. Image, audio and
video conditioning (`--first-frame`, `--last-frame`, `--reference`, `--reference-audio`,
`--reference-video`), sparse attention, FastH3/VSA, replacement-weight patches, NVFP4 and LoRA
merge mode are not yet supported.
The Metal CLI supports `generate`, `latent`, `device` and checkpoint inspection. The `bench` and
`check` are not yet available. See [Usage](usage.md) for general command options.

## Approximate speed

On an Apple M4 Max, the 448×256, 39-frame, four-step Turbo example takes roughly a minute with
warm model files, including loading, generation and MP4 output. Denoising takes about 11 seconds
per step. Loading uncached models takes longer, and larger clips require more time and memory.
These figures are a starting point for this configuration, not a comparison across GPUs.
