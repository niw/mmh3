# FastH3

FastVideo's
[FastH3 VSA-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is MiniMax H3 fine-tuned to generate in four steps with the base schedule and video sparse
attention (VSA). `make generate` uses it.

## Download

mmh3 runs FastH3 as a patch on the base DiT,
`patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors` from
[yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3). `make download-models` and
`tools/download-models.sh` download it by default, and `--no-fasth3` leaves it out. The patch holds
a rank-64 LoRA and the tensors that a LoRA cannot carry, such as the VSA gates and the fine-tuned
AdaLN, so it is 2.7 GB instead of a second DiT.

## Generate

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --attention-precision int8-fp8 \
  --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
```

`make generate` runs these settings. `--patch` takes a path, or a file name in `patches` or `loras`
of the models directory. mmh3 selects VSA for DiTs with VSA gates, which the patch adds.
`--vsa-sparsity X` (default 0.9) is the fraction of the video tiles that VSA leaves out for each
query tile, and `--attention-precision int8-fp8` runs the attention with INT8 QK and FP8 PV. With
these settings, a complete MP4 generation at 1344×768 on a DGX Spark takes about 87 seconds,
including model loading and encoding.

INT8/FP8 attention changes image details compared with `--attention-precision bf16`.

## How the patch is built

[tools/models](../tools/models/README.md#fasth3_vsa_patchpy) describes how
`tools/models/fasth3_vsa_patch.py` builds the patch from the full BF16 DiT and FastH3's checkpoint,
and how close the patched DiT is to FastH3.
