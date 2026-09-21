# NVFP4 linear layers

`--linear-precision nvfp4` runs the linear layers of the 50 DiT blocks on the FP4 tensor cores of
Blackwell GPUs, through cuBLASLt's block-scaled FP4 GEMM. It is experimental. The steps take about
a quarter less time than with the INT8 default, but the results are further from the reference.
Compared by eye and ear on a few clips, videos lose some fine detail.

## Generate

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --attention-precision int8-fp8 --linear-precision nvfp4 \
  --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
```

These are the [FastH3](fasth3.md) settings of `make generate` with NVFP4. It works the same way
with the [lightx2v Turbo LoRA](lightx2v-turbo.md), [TaoMate-H3](taomate.md) and other LoRAs, and
with both `--lora-mode` values.

## How it works

- At load time, mmh3 requantizes the INT8 ConvRot weights of the attention and MLP layers to
  NVFP4: FP4 values with one scale per 16 values and one per tensor. This takes about 0.6 s. The
  NVFP4 weights need about 12 GB of GPU memory next to the INT8 weights, which stay.
- Only the video rows run in NVFP4. The text and audio rows, about 1% of the sequence at 768p, go
  through the INT8 weights, because in NVFP4 they cost most of the accuracy of the audio.
- A patch or a LoRA in adapter mode runs inside the same FP4 GEMM as extra columns of the weight,
  so its update stays whole. mmh3 applies `--patch` and `--lora` before it switches the layers to
  NVFP4.
- The first step takes 2.7 s longer than the others at 768p, against 0.6 s with INT8. Its layers
  find the scales of their inputs in a pass of their own, and every GEMM shape tries cuBLASLt's
  candidates once. mmh3 keeps the chosen candidates in
  `$XDG_CACHE_HOME/mmh3/cublaslt-nvfp4-algorithms.txt` (`~/.cache/mmh3` by default) for later runs
  with the same shapes, cuBLASLt version and GPU, which saves about 1 s of it. The GEMMs that are
  not NVFP4 keep theirs beside it, in `cublaslt-matmul-algorithms.txt`.

## Speed

On a DGX Spark at 1344×768 and 124 frames, with the Tokyo rain prompt, seed 1 and INT8/FP8
attention, a step after the first took:

| Model | INT8 | NVFP4 |
| --- | ---: | ---: |
| [FastH3](fasth3.md) with VSA | 13.9 s | 10.2 to 10.4 s |
| [lightx2v Turbo LoRA](lightx2v-turbo.md) with Sol-Attn | 14.1 to 14.2 s | 10.7 s |
| [TaoMate-H3](taomate.md) with Sol-Attn | 13.9 to 14.0 s | 10.4 to 10.5 s |

A complete FastH3 run, including model loading and encoding, took 72.0 s with NVFP4 and 84.7 s with
INT8. Its first step took 12.0 s with the saved cuBLASLt candidates.

## Accuracy

Relative L2 error of the DiT velocity at step 1 against an FP32 reference, see
[Performance and accuracy](performance.md):

| Setup | INT8 video | INT8 audio | NVFP4 video | NVFP4 audio |
| --- | ---: | ---: | ---: | ---: |
| Base model, 448×256, dense attention | 6.7e-2 | 3.9e-2 | 2.0e-1 | 9.2e-2 |
| FastH3, 448×256, VSA keeping every tile | 1.9e-1 | 8.5e-2 | 3.5e-1 | 9.5e-2 |
| FastH3, 1344×768, VSA | 1.9e-1 | 4.7e-2 | 3.8e-1 | 1.1e-1 |

At 1344×768 with VSA, NVFP4 is about as far from the FP32 reference as ComfyUI's own BF16 run
(3.7e-1 for video and 1.1e-1 for audio).
