# MiniMax H3 Image

[MiniMax H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) by MiniMax generates pictures
of about 16 megapixels from text. Its DiT is MiniMax H3's without the audio layers, distilled to six
steps with no guidance, it reads the MiniMax H3 text encoder unchanged, and it decodes with a picture
VAE of its own. `mmh3 image` generates pictures from text and edits reference pictures on
[CUDA](cuda.md). Metal is not supported yet.

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
  [Performance and accuracy](performance.md).
- `--linear-precision int8|nvfp4` (default int8): [NVFP4 linear layers](nvfp4.md).
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

| Picture | Attention | Linear layers | Each step | Decode | Whole generation |
| --- | --- | --- | ---: | ---: | ---: |
| 1024×1024 | BF16 | INT8 | 1.4 s | 1.0 s | 18 s |
| 4096×4096 | BF16 | INT8 | 8.8 s | 2.9 s | 63 s |
| 4096×4096 | INT8/FP8 | INT8 | 7.3 s | 2.9 s | 54 s |
| 4096×4096 | INT8/FP8 | NVFP4 | 6.2 s | 2.9 s | 47 s |

The 1024×1024 picture is generated on a 2048×2048 canvas, as `--min-generation-area` sets by
default. The whole generation includes loading the text encoder, the DiT and the VAE from warm
files. At 4096×4096 INT8/FP8 attention changes fine details of the picture and keeps its
composition. NVFP4 also mutes colors and coarsens textures.

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
