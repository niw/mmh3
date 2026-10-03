# PDMD

[PDMD](https://pdmd2026.github.io/) (Projected Distribution Matching Distillation) by UC San Diego
and ByteDance distills MiniMax H3 into few-step students. mmh3 runs its
[2-NFE LoRA](https://huggingface.co/pdmd2026/pdmd_2NFE_lora), which generates in two steps.

## Download

`tools/download-models.sh --pdmd` downloads
`loras/minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors` from
[yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3). It is the PDMD 2-NFE LoRA, the
step-4000 student of rank 128, under ComfyUI's LoRA names in BF16, so ComfyUI loads it too. Its q, k
and v updates are fused into one update and cut from rank 384 to 128, which keeps 97% of their
energy on average.

## Generate

The LoRA runs in two steps with the video and audio shifts 12 and 3 it was distilled with:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 2 --shift-video 12 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors
```

`make download-models LORA=pdmd` and `make generate LORA=pdmd` run these settings too.

At 1344×768 with Sol-Attn and INT8/FP8 attention, the two steps take 14.1 and 13.7 s on a DGX Spark,
against 13.5 and 13.3 s for the DiT without a LoRA.

On [Metal](metal.md), which has no INT8/FP8 attention, leave out `--attention-precision int8-fp8`.
On an M6 with 24 GB, a 672×384, 73-frame clip takes about 45 s.

## How the LoRA is built

[tools/models](../tools/models/README.md#pdmd_lorapy) describes how `tools/models/pdmd_lora.py`
converts the LoRA.
