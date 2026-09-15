# TaoMate-H3

[TaoMate-H3](https://huggingface.co/TaoLiveAIGC/TaoMate-H3) by the Alibaba TaoLive AIGC Team is a
LoRA distilled from MiniMax H3 for streaming generation in three steps per chunk. mmh3 generates
the whole clip at once with it, in the three steps it was distilled for. Streaming is not
implemented.

## Download

`tools/download-models.sh --taomate` downloads
`loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors` from
[yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3). It is the TaoMate-H3 adapter,
the step-3000 generator EMA of rank 128, under ComfyUI's LoRA names in BF16, so ComfyUI loads it
too.

## Generate

`--schedule taomate` samples states 0, 16, 33 and 49 of the shifted 50-step schedule, the steps
the LoRA was distilled for, in place of `--steps`:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --schedule taomate --attention sol --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors
```

At 1344×768 with Sol-Attn and INT8/FP8 attention, the three steps take 14.3, 13.8 and 13.9 s on a
DGX Spark, against 13.8, 13.5 and 13.5 s for the DiT without a LoRA.

Keep `--lora-mode adapter`, the default. Most of TaoMate's updates are smaller than the INT8 step
of their weights, so `--lora-mode merge` loses them, with visibly worse detail.

## How the LoRA is built

[tools/models](../tools/models/README.md#taomate_lorapy) describes how
`tools/models/taomate_lora.py` converts the adapter, and how smaller ranks compare.
