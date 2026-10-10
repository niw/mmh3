# DMAD

[DMAD](https://yzmblog.github.io/projects/DMAD/) (Distribution Matching as Adversarial
Distillation) by Texas A&M University and ByteDance distills MiniMax H3 into
[4-step students](https://huggingface.co/ZhengmingYu/DMAD). mmh3 runs the one whose critic was
fully trained, which scores higher on AVGen-Bench than the paper's checkpoint.

## Download

`tools/download-models.sh --dmad` downloads
`loras/minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors` from
[yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3). It is that student, of rank
128, under ComfyUI's LoRA names in BF16, so ComfyUI loads it too. Its q, k and v updates are fused
into one update and cut from rank 384 to 128, which keeps 96% of their energy on average.

## Generate

The LoRA runs in four steps with the video and audio shifts 12 and 2 it was distilled with, and
with the re-noise step rule it was trained with:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --shift-video 12 --shift-audio 2 --sampler renoise --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors
```

`make download-models LORA=dmad` and `make generate LORA=dmad` run these settings too.

On [Metal](metal.md), which has no INT8/FP8 attention, leave out `--attention-precision int8-fp8`.

With these settings, a complete generation takes:

| Machine | Clip | Each step | Whole generation |
| --- | --- | ---: | ---: |
| DGX Spark | 1344×768, 124 frames | 14.1, 13.6, 13.7, 13.8 s | 80 s |
| M6 | 672×384, 73 frames | 13.5 s | 76 s |

`--sampler euler` runs the steps in the same time, with a different video for the same seed.

## How the LoRA is built

[tools/models](../tools/models/README.md#dmad_lorapy) describes how `tools/models/dmad_lora.py`
converts the LoRA.
