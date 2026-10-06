# Veda

[Veda](https://veda-sparse.github.io/) is a block-sparse attention for video diffusion
transformers whose tiles a small distilled predictor picks. mmh3 runs it with
[Veda-Sparse's predictor for MiniMax H3](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
on CUDA and [Metal](metal.md). It changes no DiT weights, so it runs with any LoRA or patch.

## How it works

The DiT's video tokens are cut into tiles of 128, boxes of (frames, patch rows, patch columns) whose
shape the predictor's plan sets for each block and head, and the text and audio tokens into tiles of
128 in sequence order. Per head, the predictor pools the queries and the keys of every video tile
into their mean, maximum and minimum and projects them, and each video query tile keeps the video
tiles of the best scores between those, besides its own, and every text and audio tile. At the
default `--veda-sparsity 0.9`, it keeps a tenth of the tiles that the video's tokens fill, and the
attention is exact over the tiles it keeps. Keyframes and references are tiled as videos of their
own, with a budget of their own, `--veda-reference-sparsity`.

The plans were searched for 1344×768, 768×1344, 768×768 and 1024×768 clips of 5.2, 10.1 and
14.4 seconds. Other sizes take the plan of the nearest aspect ratio and length, which mmh3 says at
the first step, and are outside what the predictor was trained on. So are other step counts than
the 8 steps of the Turbo LoRA it was trained with, and keyframes and references.

The predictor and the selection run on the GPU, and keep the same tiles as Veda-on-ComfyUI. On
CUDA, Veda attends in BF16, so `--attention-precision int8-fp8` does not run with it.

## Download

`tools/download-models.sh --veda` downloads
`veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`, the FP8 predictor of 275 MB, from
Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview into the models directory. `--veda-predictor`
takes another path.

## Generate

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --width 672 --height 384 --frames 73 \
  --steps 4 --shift-video 6 --shift-audio 3 --attention veda --veda-sparsity 0.7 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

With the [lightx2v Turbo LoRA](lightx2v-turbo.md) as above, a step takes less time than with dense
attention, and the larger the clip, the more Veda saves. Each step says how many of the video tile
pairs Veda kept. So far it has been compared visually on one prompt and seed only.

A small clip has few tiles, and its plan was searched for another size, so at the default
`--veda-sparsity` the shapes in it can break apart. A lower `--veda-sparsity`, such as 0.7, keeps
them whole at 672×384, as above. The sizes the plans were searched for look fine at the default.

A run with Veda takes every step on the machine that leads it, since [workers](distributed.md) do
not run it.
