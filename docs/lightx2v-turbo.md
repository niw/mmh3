# lightx2v Turbo LoRA

lightx2v's [MiniMax-H3 Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) makes MiniMax
H3 generate in four steps. mmh3 runs its 768p 4-step v1.2 file in ComfyUI's format, a LoRA of the
attention and MLP layers of the 50 blocks and the text refiner, of rank 128 (384 for the fused qkv
projections). mmh3 applies it on top of the INT8 weights at run time.

## Download

`tools/download-models.sh --lightx2v-turbo` downloads
`loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` from
lightx2v/Minimax-h3-Turbo into the models directory.

## Generate

The LoRA runs in four steps with the shifts it was trained with, INT8/FP8 attention, and Sol-Attn
from the first step:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
```

INT8/FP8 attention changes image details and needs about 0.76 GiB more memory. `--sparse-start 0`
can also change the composition. `--sparse-start 0.2` keeps the first step dense and took 98.98 s.
So far these settings have been compared visually on one prompt and seed only.

[Performance and accuracy](performance.md) gives the times of these settings on a DGX Spark.
