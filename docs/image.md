# MiniMax H3 Image

[MiniMax H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax generates pictures
of about 16 megapixels from text. Its DiT is MiniMax H3's without the audio layers, distilled to six
steps with no guidance, it reads the MiniMax H3 text encoder unchanged, and it decodes with a picture
VAE of its own. `mmh3 image` generates pictures from text and edits reference pictures on
[CUDA](cuda.md) and [Metal](metal.md).

## Download

`tools/download-models.sh --image` downloads these files from
[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) next to the text
encoder that MiniMax H3 uses:

- `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors`, the DiT in INT8 ConvRot, 21 GB.
- `vae/minimax_h3_image_vae_fp16.safetensors`, the picture VAE in FP16, 1.2 GB.

[tools/models](../tools/models/README.md#the-picture-vae-of-minimax-h3-image) also builds a picture
VAE with INT8 ConvRot decoder blocks, `vae/minimax_h3_image_vae_int8_convrot.safetensors`, which
`--vae` takes. It decodes 63 dB PSNR from the FP16 one and only a little faster, so `mmh3 image`
keeps the FP16 one unless told.

## Generate

```sh
target/release/mmh3 image --out panda.png --width 4096 --height 4096 \
  --prompt "A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn, soft golden light filtering through the trees, photorealistic, detailed fur"
```

The prompt goes to the text encoder as given, with no template. The model expects a detailed English
description of the whole picture, which MiniMax's own service writes from a short prompt with a
prompt expander before generating.

- `--out FILE` (required): A `.png` file.
- `--prompt TEXT`, `--prompt-file FILE`: The prompt, cut after 4096 tokens.
- `--width N`, `--height N` (default 2048, 2048): The size of the picture, rounded to multiples
  of 32. The model works from about 4 to 20 megapixels and best at about 16, such as 4096×4096,
  4736×3552, 5024×3360, 5472×3072 or 6272×2688 and their portrait sides.
- `--min-generation-area PIXELS` (default 4194304, 2048×2048): A picture smaller than this is
  generated on a canvas of this area at the same aspect ratio and resized to the requested size
  with Lanczos. 1024 turns this off.
- `--reference FILE` (default none, repeatable): A PNG or JPEG picture to edit, see
  [Edit](#edit).
- `--seed N` (default 0): The seed of the initial noise.
- `--attention-precision bf16|int8-fp8` (default bf16): INT8/FP8 attention, as in
  [Performance and accuracy](performance.md). On Metal, `fp32|fp16|int8-fp16` as in
  [Metal](metal.md#options-and-current-limits).
- `--linear-precision int8|nvfp4` (default int8): [NVFP4 linear layers](nvfp4.md). On Metal,
  `fp32|fp16|int8|mps-fp16` as in [Metal](metal.md#options-and-current-limits).
- `--draft-steps N` (default 0): Runs the first N steps on a canvas of half the width and height,
  as in [Faster](#faster). The canvas must then be a multiple of 64 on each side, and an edit
  takes none.
- `--latent-upscaler FILE`: Enlarges the draft with
  [the latent upscaler](draft-steps.md#latent-upscaler) rather than bilinearly.
- `--attention dense|sol` (default dense), `--sparse-tau X`, `--sparse-start X`: Sol-Attn, as in
  [Usage](usage.md), from the third step by default.
- `--dit FILE`, `--text-encoder FILE`, `--vae FILE`, `--models DIR`: Checkpoints, as in
  [Models](models.md).
- `--save-latent FILE`: Also writes the final latent, which the INT8 picture VAE is calibrated
  on.
- `--context FILE`: The text encoder states of a `text.safetensors` in place of a prompt.

The six steps run on the sigmas the checkpoint was distilled for, with the output head of each
step, and the picture VAE decodes in tiles of 256 pixels that overlap by at least 64, as the
reference pipeline does. The picture is quantized to 8 bits by truncation, as the reference
pipeline does too.

A [server](server.md#asking-for-a-picture) takes the same generations with `POST /v1/images` and
keeps the models loaded between them.

## Faster

These settings generate a 4096×4096 picture in about 35 s rather than 63 s on a DGX Spark:

```sh
tools/download-models.sh --image --latent-upscaler
target/release/mmh3 image --out panda.png --width 4096 --height 4096 \
  --draft-steps 2 --latent-upscaler minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  --attention sol --attention-precision int8-fp8 \
  --prompt "A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn, soft golden light filtering through the trees, photorealistic, detailed fur"
```

- The first two steps run as [draft steps](draft-steps.md) on a 2048×2048 canvas, still within the
  sizes the model works at, and settle the composition in a quarter of the tokens. The picture
  VAE's encoder is the video VAE's, so the latent upscaler of MiniMax H3 enlarges the draft. The
  same seed gives another composition than without draft steps.
- The four steps at full size run Sol-Attn, which attends to about 17% of the picture.
- `--linear-precision nvfp4` takes it down to about 30 s, with muted colors and coarser textures.

Sol-Attn costs the most accuracy. Compared on a shop sign, a portrait and a station hall, it
rewrote the prices on a chalkboard, added stray lettering and painted a second clock, while faces
and hands stayed as they were. With `--sparse-start 0.5` it starts at the fourth step and kept the
lettering. Draft steps alone kept large lettering and the composition, and moved small lettering
at most. For a picture with text, leave `--attention sol` out, which takes about 50 s.

## Edit

With `--reference`, the prompt edits up to 16 pictures, which it refers to as `<Picture 1>`,
`<Picture 2>` and so on, in the order of the options:

```sh
target/release/mmh3 image --out night.png --width 4096 --height 4096 \
  --reference street.jpg \
  --prompt "<Picture 1> at night, the street lamps lit and the shop windows glowing warm yellow."
```

The size of the result is `--width` and `--height`, whatever the size of the references. Each
reference goes twice into the generation, as the reference pipeline takes it:

- To the text encoder, as `Picture i: ` and the vision tower's embeddings of a copy with a
  1024-pixel short edge, before the prompt.
- To the DiT, as the latent of a copy whose short edge is kept within 1024 to 2048 pixels, encoded
  by the picture VAE and held just short of clean in front of the generated picture.

## Speed

On a DGX Spark, with the prompt above and seed 1:

| Picture | Draft steps | Attention | Linear layers | Each step at full size | Whole generation |
| --- | ---: | --- | --- | ---: | ---: |
| 1024×1024 | 0 | BF16 | INT8 | 1.4 s | 18 s |
| 1024×1024 | 2 | INT8/FP8 | INT8 | 1.3 s | 16 s |
| 4096×4096 | 0 | BF16 | INT8 | 8.8 s | 63 s |
| 4096×4096 | 0 | INT8/FP8 | INT8 | 7.3 s | 54 s |
| 4096×4096 | 0 | INT8/FP8 | NVFP4 | 6.2 s | 47 s |
| 4096×4096 | 0 | Sol-Attn, BF16 | INT8 | 5.3 s | 49 s |
| 4096×4096 | 2 | BF16 | INT8 | 8.8 s | 50 s |
| 4096×4096 | 2 | Sol-Attn, BF16 | INT8 | 5.3 s | 37 s |
| 4096×4096 | 2 | Sol-Attn, INT8/FP8 | INT8 | 4.9 s | 35 s |
| 4096×4096 | 2 | Sol-Attn, INT8/FP8 | NVFP4 | 3.7 s | 30 s |

The 1024×1024 picture is generated on a 2048×2048 canvas, as `--min-generation-area` sets by
default, so its draft steps run on a 1024×1024 canvas, too short for Sol-Attn. A draft step at
4096×4096 takes about 1.5 s and the latent upscaler 2 s with its loading, and the picture VAE
decodes a 4096×4096 picture in 2.9 s. The whole generation includes loading the text encoder, the
DiT and the VAE from warm files. At 4096×4096 INT8/FP8 attention changes fine details of the
picture and keeps its composition. NVFP4 also mutes colors and coarsens textures.

On an M6 with 24 GB, with the same prompt and seed, INT8 linear layers and attention on the matrix
units, FP16 unless INT8/FP16:

| Picture | Draft steps | Attention | Each step at full size | Whole generation |
| --- | ---: | --- | ---: | ---: |
| 1024×1024 | 0 | FP16 | 7.1 s | 57 s |
| 1024×1024 | 2 | INT8/FP16 | 7.0 s | 52 s |
| 4096×4096 | 0 | FP16 | 49 s | 337 s |
| 4096×4096 | 2 | FP16 | 49 s | 255 s |
| 4096×4096 | 0 | Sol-Attn, INT8/FP16 | 26 s | 237 s |
| 4096×4096 | 2 | Sol-Attn, INT8/FP16 | 26 s | 165 s |

The last 4096×4096 row is the [faster settings](#faster) with `--attention-precision int8-fp16`,
Metal's INT8 attention. The 1024×1024 picture is generated on a 2048×2048 canvas, as on CUDA, so a
2048×2048 picture takes as long and its draft steps are too short for Sol-Attn. A draft step at
4096×4096 takes about 7 s, the latent upscaler 5 s with its loading, and the picture VAE decodes a
4096×4096 picture in 35 s. The 21 GB DiT does not fit in the GPU's share of 24 GB, so the Mac keeps
as many of its blocks as fit beside the step, 31 of the 52 at 2048×2048 and 11 at 4096×4096, and
reads the others from the disk as they run. Editing needs macOS 26, since the picture VAE's encoder
runs on the matrix units, as for the [references](ref2va.md) of a video.

## Accuracy

`tools/golden/h3_image_reference.py` writes golden data from MiniMaxAI's release through diffusers,
with the BF16 DiT and the FP32 picture VAE, and `mmh3-tools check` compares mmh3 with it. At
512×512, with the text encoder states of the golden data, the relative L2 error is:

| Check | INT8 DiT | BF16 DiT | INT8 DiT, editing two pictures |
| --- | ---: | ---: | ---: |
| DiT velocity at the second step | 3.1e-2 | 3.5e-2 | 5.5e-2 |
| Latent after all six steps from the same noise | 2.5e-1 | 1.4e-1 | 7.7e-2 |

The FP16 picture VAE decodes the final latent of the golden data 72 dB PSNR from the reference's
FP32 decode, and its encoder gives the posterior means of the two references within 4.6e-3 of the
reference's FP32 encoder.

## License

MiniMax H3 Image is released under the
[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE),
which differs from MiniMax H3's. Among other terms, a service that offers it to others, or a user
with an annualized revenue above 1 million USD, needs MiniMax's prior written authorization.
