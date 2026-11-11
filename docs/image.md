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
description of the whole picture, as in [Prompts](#prompts).

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
  [the latent upscaler](draft-steps.md#latent-upscaler) rather than bilinearly. On Metal, it runs
  on the matrix units, so it needs macOS 26.
- `--renoise-sigma X`: Noises the enlarged draft to X rather than to the next step's sigma, and
  runs the steps at full size on the sigmas below X. The draft may then take every step, as in
  [Fewer steps](#fewer-steps).
- `--attention dense|sol` (default dense), `--sparse-tau X`, `--sparse-start X`: Sol-Attn, as in
  [Usage](usage.md), from the third step by default.
- `--steps N`, `--shift X` (default 12): Samples in N steps on the uniform grid shifted by X rather
  than on the six sigmas the checkpoint was distilled for, as in [Fewer steps](#fewer-steps).
- `--patch FILE`, `--lora FILE`, `--lora-strength X`, `--lora-mode adapter|merge`: A patch and a
  LoRA, as in [Usage](usage.md).
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

These settings generate a 4096×4096 picture in about 24 s rather than 62 s on a DGX Spark:

```sh
tools/download-models.sh --image --pdmd --latent-upscaler
target/release/mmh3 image --out panda.png --width 4096 --height 4096 \
  --steps 3 --lora minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors \
  --draft-steps 3 --renoise-sigma 0.5 \
  --latent-upscaler minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors \
  --attention-precision int8-fp8 \
  --prompt "A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn, soft golden light filtering through the trees, photorealistic, detailed fur"
```

- The [PDMD](pdmd.md) LoRA generates in three steps, as in [Fewer steps](#fewer-steps).
- All three run as [draft steps](draft-steps.md) on a 2048×2048 canvas, still within the sizes the
  model works at. The picture VAE's encoder is the video VAE's, so the latent upscaler of MiniMax H3
  enlarges the draft.
- One step at full size redraws the enlarged draft from sigma 0.5.
- `--linear-precision nvfp4` takes about 1 s off, with muted colors and coarser textures.
- On Metal, `--attention-precision int8-fp16` takes the place of `int8-fp8`, which Metal does not
  take, as in [Speed](#speed).

The same seed gives another composition than the six steps without the LoRA. On a shop front these
settings kept the lettering of the sign and the chalkboard.

Without a LoRA, `--draft-steps 2` with the latent upscaler, `--attention sol` and
`--attention-precision int8-fp8` take about 35 s. Sol-Attn, which attends to about 17% of the
picture, costs the most accuracy there. Compared on a shop sign, a portrait and a station hall, it
rewrote the prices on a chalkboard, added stray lettering and painted a second clock, while faces
and hands stayed as they were. With `--sparse-start 0.5` it starts at the fourth step and kept the
lettering. Draft steps alone kept large lettering and the composition, and moved small lettering
at most.

## Prompts

MiniMax H3 Image draws best from a complete description of the finished picture: what it is for,
what is in it and where, how it looks, and every word on it in quotes. MiniMax's own service writes
one from a short request with a prompt expander before it generates, and `mmh3 image` does not, so
the prompt it takes should already be one.

- MiniMax's
  [h3-image-prompt-writing](https://github.com/MiniMax-AI/MiniMax-H3-Image/tree/main/skills/h3-image-prompt-writing)
  skill turns a short request into such a prompt, for text to image and for edits with
  `<Picture N>`. An agent such as Claude Code loads it as a skill, and its guides
  `references/t2i-en.txt` and `references/edit-en.txt` read well on their own.
- [MiniMax-H3-Image-PE](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image-PE) is the prompt expander
  of MiniMax's service for text to image, a language model of 27B parameters in FP4 that SGLang
  serves. It writes the description and a suggested aspect ratio from a short request.

## Fewer steps

The [PDMD](pdmd.md) 2-NFE LoRA, distilled on MiniMax H3, works on MiniMax H3 Image too. It runs on
the uniform grid of `--steps`, shifted by 12:

```sh
tools/download-models.sh --image --pdmd
target/release/mmh3 image --out panda.png --width 4096 --height 4096 --steps 3 \
  --lora minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors \
  --prompt "A red panda in a knitted scarf sits on a mossy log in a misty pine forest at dawn, soft golden light filtering through the trees, photorealistic, detailed fur"
```

On a DGX Spark, the 4096×4096 picture takes about 40 s in three steps and 29 s in two, against
62 s in six. Three steps keep the composition and the detail, with a faint woven texture on fur at
full size. Two steps show it plainly on knitted and flat areas and suit drafts. Leave the LoRA as an
adapter, since `--lora-mode merge` rounds most of it away in the INT8 weights.

Draft steps leave out the woven texture. The [faster settings](#faster) run all three steps on the
half-size canvas and redraw the enlarged draft from sigma 0.5 in one step at full size.
`--draft-steps 2` without `--renoise-sigma` instead runs the last of the three at full size from its
own sigma of 0.857, which takes as long and keeps less fine detail, such as the crust of bread or
single hairs. A higher `--renoise-sigma` redraws more of the draft and a lower one less.

A step off the six distilled sigmas projects through the output heads of the distilled steps it
spans, averaged by how much of it each one covers, as ComfyUI takes them.

## Edit

With `--reference`, the prompt edits up to 16 pictures, which it refers to as `<Picture 1>`,
`<Picture 2>` and so on, in the order of the options. On Metal, the picture VAE's encoder runs on
the matrix units, as for the [references](ref2va.md) of a video, so edits need macOS 26:

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

| Picture | Steps | Draft steps | Attention | Linear layers | Each step at full size | Whole generation |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 1024×1024 | 6 | 0 | BF16 | INT8 | 1.4 s | 18 s |
| 1024×1024 | 6 | 2 | INT8/FP8 | INT8 | 1.3 s | 16 s |
| 4096×4096 | 6 | 0 | BF16 | INT8 | 8.8 s | 63 s |
| 4096×4096 | 6 | 0 | INT8/FP8 | INT8 | 7.3 s | 54 s |
| 4096×4096 | 6 | 0 | INT8/FP8 | NVFP4 | 6.2 s | 47 s |
| 4096×4096 | 6 | 0 | Sol-Attn, BF16 | INT8 | 5.3 s | 49 s |
| 4096×4096 | 6 | 2 | BF16 | INT8 | 8.8 s | 50 s |
| 4096×4096 | 6 | 2 | Sol-Attn, BF16 | INT8 | 5.3 s | 37 s |
| 4096×4096 | 6 | 2 | Sol-Attn, INT8/FP8 | INT8 | 4.9 s | 35 s |
| 4096×4096 | 6 | 2 | Sol-Attn, INT8/FP8 | NVFP4 | 3.7 s | 30 s |
| 4096×4096 | 3, PDMD | 0 | BF16 | INT8 | 8.9 s | 40 s |
| 4096×4096 | 3, PDMD | 3 | INT8/FP8 | INT8 | 7.4 s | 24 s |
| 4096×4096 | 3, PDMD | 3 | INT8/FP8 | NVFP4 | 6.1 s | 23 s |

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
| 4096×4096, 3 PDMD steps | 3 | INT8/FP16 | 59 s | 140 s |

The last row is the [faster settings](#faster) with `--attention-precision int8-fp16`, Metal's INT8
attention. The 1024×1024 picture is generated on a 2048×2048 canvas, as on CUDA, so a
2048×2048 picture takes as long and its draft steps are too short for Sol-Attn. A draft step at
4096×4096 takes about 7 s, the latent upscaler 5 s with its loading, and the picture VAE decodes a
4096×4096 picture in 35 s. The 21 GB DiT does not fit in the GPU's share of 24 GB, so the Mac keeps
as many of its blocks as fit beside the step, 31 of the 52 at 2048×2048 and 11 at 4096×4096, and
reads the others from the disk as they run.

On a 14-inch MacBook Pro with an M5 Max and 128 GB, in High Power Mode on AC power, with the same
prompt and seed, INT8 linear layers and FP16 attention:

| Picture | Draft steps | Attention | Each step at full size | Whole generation |
| --- | ---: | --- | ---: | ---: |
| 2048×2048 | 0 | FP16 | 4.3 s | 38 s |
| 2048×2048 | 2 | FP16 | 4.4 s | 30 s |
| 4096×4096 | 0 | FP16 | 24 s | 165 s |
| 4096×4096 | 0 | Sol-Attn, FP16 | 18 s | 153 s |
| 4096×4096 | 2 | FP16 | 27 s | 140 s |
| 4096×4096 | 2 | Sol-Attn, FP16 | 14 s | 89 s |

The draft was enlarged bilinearly, and a draft step at 4096×4096 took about 3.5 s. These runs, like
the M6's above, decoded the picture's tiles one at a time with FP32 products, in 5 s at 2048×2048
and 18 s at 4096×4096. The picture VAE now decodes several tiles at once, as CUDA does, and at these
precisions its linear layers take its FP16 weights on the matrix units, with FP16 inputs, which
takes about 0.22 of that time at the same clock; its pictures are within one level everywhere of
those of FP32 products (68.5 dB PSNR at 4096×4096). The GPU of this MacBook Pro runs at its top
clock of about 1.4 GHz only for the first second or two of a load and then at about 700 to 800 MHz,
held there by the power and cooling of the 14-inch case whatever the adapter, so the same run took
from about two thirds to one and a half times these times as the clock moved. A 16-inch MacBook Pro
or a Mac Studio with the same chip should be faster.

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

`mmh3-tools check` runs on CUDA. On Metal, the tests check the picture VAE's decoder and encoder
against diffusers on small random models, and the DiT's steps through each output head against
the CPU reference and against a single head summed by hand.

## License

MiniMax H3 Image is released under the
[MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE),
which differs from MiniMax H3's. Among other terms, a service that offers it to others, or a user
with an annualized revenue above 1 million USD, needs MiniMax's prior written authorization.
