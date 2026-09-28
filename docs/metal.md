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
device and reading the rest as they run` says how many it kept. The GPU reads the layers where
they land from the disk, with no copy. A text encoder that does not fit keeps none of its layers,
since `generate` encodes one prompt and reads each layer once either way, and layers kept would
only push the rest of the system's memory into swap. It also leaves its embedding table on the
disk and reads the rows of the prompt from there. A 24 GB Mac generates the clip of
`make generate` this way within 18 GB.

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

This downloads the base models and the [FastH3](fasth3.md) patch, then generates a 672×384,
73-frame clip in four steps with FastH3's video sparse attention. Change the prompt and output path with:

```sh
make generate PROMPT="A rainy street." OUT=rain.mp4
```

To build and run directly:

```sh
cargo build --release --features metal --bins
target/release/mmh3-tools device

target/release/mmh3 generate \
  --models models --prompt "A rainy street." \
  --width 672 --height 384 --frames 73 --steps 4 \
  --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors \
  --out rain.mp4
```

Select either `metal` or `cuda`. Metal includes native MP4 output. The separate `mp4` Cargo
feature selects CUDA/NVENC. Use `--ffmpeg` for an external encoder override. See [Output](output.md).

## Options and current limits

On macOS 26 and later, the DiT, the text encoder and the video VAE's transformer run on the GPU's
matrix units by default, the Neural Accelerators on an M5 or later. Their INT8-weight layers take
INT8 activations, rotated and quantized a row at a time as on CUDA, and their dense attention
takes FP16 products with FP32 softmax and accumulation. A LoRA adapter's two products read FP16
copies of its weights. On an older macOS the defaults are FP16 products through MPS and FP32
attention.

`--linear-precision` selects the product of the DiT's INT8-weight layers:

| Value | Mode |
| --- | --- |
| `int8` | INT8 through MPP (default on macOS 26+) |
| `fp16` | FP16 through MPP (macOS 26+) |
| `mps-fp16` | FP16 through MPS (default before macOS 26) |
| `fp32` | FP32 reference mode |

`--attention-precision` selects the attention of the DiT, the text encoder and the video VAE:

| Value | Mode |
| --- | --- |
| `fp16` | FP16 products through MPP (default on macOS 26+) |
| `int8-fp16` | `fp16`, with the DiT's scores from INT8 queries and keys (macOS 26+) |
| `fp32` | FP32 reference mode (default before macOS 26) |

Changing precision can change the generated sample even with the same seed. The other layers
compute in FP32.

`int8-fp16` quantizes the DiT's queries and keys to INT8 with one scale a token and head, and
keeps the values and the probabilities in FP16, since Metal's matrix units take FP8 only from
memory. The text encoder and the video VAE attend in FP16. The scores are a small share of a step,
so it gains little. It is this machine's setting: workers asked for a step take their own default.

Text prompts, precomputed `--context` tensors, adapter LoRAs, patches such as
[FastH3](fasth3.md)'s, and Sol-Attn and VSA sparse attention are supported. Image, audio and video
conditioning (`--first-frame`, `--last-frame`, `--reference`, `--reference-audio`,
`--reference-video`), INT8/FP8 attention, NVFP4 and LoRA merge mode are not yet supported.
The Metal CLI supports `generate`, `latent`, `device` and checkpoint inspection. The `bench` and
`check` are not yet available. See [Usage](usage.md) for general command options.

## Sparse attention

`--attention sol` and `--attention vsa` run as on CUDA, with the options in [Usage](usage.md), over
the DiT's heads of 128. The routing runs in FP32 and the attention over the routed tiles in the
precision of `--attention-precision`, FP16 on the matrix units or FP32. A share of a step handed out
by a [distributed](distributed.md) run still attends densely.

Both attend over a fraction of the tiles, so a step at a large size takes much less time than with
dense attention. From the same context and noise, a step's video latent stays close to CUDA's BF16
one with Sol-Attn, whose routing follows a threshold. With VSA it moves as far as CUDA's own moves
when its attention changes to INT8/FP8: VSA keeps a tenth of the video tiles by their pooled scores,
so a small difference in the scores keeps a different tenth.

## Approximate speed

`make generate`, a 672×384, 73-frame FastH3 clip in four steps, takes about 153 seconds on an Apple
M4 Max with 128 GB, at about 27 seconds a step, and about 67 seconds on an M6 with 24 GB, at about
12 seconds a step, even though the M6 keeps only 24 of the DiT's 52 blocks on the device. Both are
with warm model files and include loading, generation and MP4 output. Loading uncached models takes
longer, and larger clips require more time and memory.
These figures are a starting point for this configuration, not a comparison across GPUs.
