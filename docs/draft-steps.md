# Draft steps

`--draft-steps N` runs the first N steps of a generation on a canvas of half the width and height,
a quarter of the video tokens, and the rest at full size. The early steps of a few-step model settle
the layout and the motion, and a smaller canvas is enough for that.

## How it works

The draft steps start from noise of the half-size latent. After the last of them, mmh3 takes the
clean latent that step predicts, enlarges it to the full canvas with the
[latent upscaler](#latent-upscaler), and noises it again to the next sigma with fresh noise from the
seed. The steps after it run at full size and draw the details. The audio latent does not depend on
the canvas and goes through every step as it is.

A half-size canvas of a 768p clip is shorter than the 12,288 tokens at which Sol-Attn starts, so
the draft steps would run dense. They run Sol-Attn regardless, since their layout is redrawn at
full size anyway.

A run with [workers](distributed.md), and so the [server](server.md), opens one session for the
draft steps and another for the steps at full size, on the same connections.

## Generate

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --steps 4 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora models/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
  --draft-steps 2
```

The width and the height must be multiples of 64, so that the half-size canvas stays on the
32-pixel grid. Draft steps do not run with keyframes or references.

## Latent upscaler

[LBH-123-AI's latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) is a
3D CNN trained to double the width and height of a MiniMax H3 latent. `tools/download-models.sh
--latent-upscaler` downloads it into `latent_upscale_models` of the models directory, where mmh3
looks for it, and `--latent-upscaler FILE` takes another path. It runs on the GPU in FP16 and takes
about 1.3 s to load and 1.3 s to run at 1344×768 on a DGX Spark.

Without it, mmh3 enlarges the latent bilinearly. A latent mixed that way is not one the VAE expects,
and the steps at full size leave marks of it: specks and letter-like shapes on faces, and with
fewer steps left, a dotted grid over the lights and the background.

## How many

With a 4-step model, about 2 is the limit. At 3, only the last step runs at full size, and the
half-size canvas settles the layout and the motion, so the video changes noticeably.

The same seed gives a different composition with draft steps than without them. With the Turbo
LoRA, the videos with draft steps came out sharper, with less background blur. So far draft steps
have been compared visually on one prompt and a few seeds only.

## Performance

With the settings of each LoRA's page and seed 1, a complete generation took:

| Machine | Clip | Model | Without | With 2 draft steps |
| --- | --- | --- | ---: | ---: |
| DGX Spark | 1344×768, 124 frames | [lightx2v Turbo LoRA](lightx2v-turbo.md) | 83.2 s | 61.5 s |
| DGX Spark | 1344×768, 124 frames | [DMAD](dmad.md) | 83.5 s | 62.0 s |
| M6 | 768×448, 56 frames | [lightx2v Turbo LoRA](lightx2v-turbo.md) | 72.9 s | 61.0 s |
| M6 | 768×448, 56 frames | [DMAD](dmad.md) | 73.1 s | 58.5 s |

On the DGX Spark, a draft step took about 3.5 s, against about 14.5 s at full size, and the latent
upscaler 2.6 s. The times on the M6 are with the bilinear enlargement.
