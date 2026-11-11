# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "accelerate",
#     "diffusers @ git+https://github.com/huggingface/diffusers@f598e6e6a99528858b962c1399c45b59a6c25484",
#     "numpy>=2",
#     "safetensors",
#     "torch>=2.9",
#     "transformers",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# ///
"""Writes golden text-to-image data of MiniMax H3 Image from the released weights through diffusers.

Development-only tool. It reads a local copy of MiniMaxAI/MiniMax-H3-Image with the
transformer_diffusers, vae_diffusers and pdd_config.json of the repository, and the text encoder
states of a prompt from a text.safetensors written by h3_reference.py, since the DiT reads them
the same way in both models. It runs on a CUDA GPU with uv, which installs diffusers and PyTorch
for CUDA 13.0, for example:

    uv run tools/golden/h3_image_reference.py --repo /path/to/MiniMax-H3-Image \\
        --text /path/to/golden/small/text.safetensors --out /path/to/golden/image-small

The BF16 DiT takes about 66 GB of device memory. The defaults describe the small case, 512x512
with seed 2026. It runs the released six-step sampler, Euler steps on the distilled sigmas with
the output head of each step, on the noise of a CPU generator.

--reference, repeatable, adds a reference picture to edit the way the edit pipeline does: resized
to a short edge within 1024 to 2048 pixels, cropped to whole latent pixels, encoded by the picture
VAE's FP32 encoder with its released tiling, sampled from the posterior, rounded to FP16, cropped
to whole patches and noised to t = 0.999. The text file should then hold the states of a prompt
with as many pictures, such as the text.safetensors of a ref2va golden directory.

It writes three files into the output directory, in the layout mmh3-tools check reads:

- text.safetensors: a copy of the text file.
- dit.safetensors: the initial noise, the latent after every step, and the input, the
  prediction and selected block outputs of one step. The stored velocity is the DiT's own
  data-ward prediction u with x0 = x + sigma * u.
- references.safetensors, with --reference: the pictures as the encoder takes them, [height,
  width, 3] in [0, 1], and their normalized posterior means. dit.safetensors then holds the
  references as the DiT sees them.
- decode.safetensors, unless --no-decode: the pixels of the final latent in [0, 1] from the
  FP32 picture VAE with its released tiling, as "video" of one frame, before the 8-bit
  quantization of the pipeline.
"""

import argparse
import json
import os
import shutil

import numpy
import torch
from safetensors.torch import load_file, save_file

LATENT_CHANNELS = 24
SPATIAL_RATIO = 16
PATCH = (1, 2, 2)
IMAGE_TAG = 0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
REFERENCE_SHORT_EDGE = (1024, 2048)
# The timestep the reference rows are held at, just short of clean.
REFERENCE_TIMESTEP = 0.999


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo", required=True, help="local MiniMaxAI/MiniMax-H3-Image"
    )
    parser.add_argument(
        "--text", required=True, help="text.safetensors with context and token_tags"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument("--watched-blocks", default="0,25,49")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--no-decode", action="store_true")
    args = parser.parse_args()

    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import (
        AutoencoderKLMiniMaxH3Image,
    )
    from diffusers.models.transformers.transformer_minimax_h3_image import (
        MiniMaxH3ImageTransformer2DModel,
    )
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
        build_image_packed_sequence,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import (
        bound_short_edge,
    )
    from PIL import Image, ImageOps

    device = torch.device("cuda")
    os.makedirs(args.out, exist_ok=True)
    text_path = os.path.join(args.out, "text.safetensors")
    if os.path.abspath(args.text) != os.path.abspath(text_path):
        shutil.copyfile(args.text, text_path)
    text = load_file(args.text)
    context = text["context"].to(device, torch.bfloat16)
    if context.ndim == 2:
        context = context[None]
    text_tags = text["token_tags"].to(torch.long)

    with open(os.path.join(args.repo, "pdd_config.json")) as file:
        sample_sigmas = json.load(file)["sigmas"]
    # The released sampler rounds every sigma through its float32 timestep 1 - sigma and back.
    timesteps = torch.tensor(
        [1.0 - sigma for sigma in sample_sigmas[:-1]], dtype=torch.float32
    )
    sigmas = torch.cat([1.0 - timesteps, torch.zeros(1)])
    steps = len(timesteps)

    def load_vae():
        return AutoencoderKLMiniMaxH3Image.from_pretrained(
            os.path.join(args.repo, "vae_diffusers"), torch_dtype=torch.float32
        ).to(device)

    # The edit pipeline samples the posteriors off the request's generator before the noise.
    generator = torch.Generator("cpu").manual_seed(args.seed)
    vae = load_vae() if args.reference else None
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
    pictures, means, reference_latents = [], [], []
    for path in args.reference:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        size = bound_short_edge(*image.size, *REFERENCE_SHORT_EDGE, 32)
        if image.size != size:
            image = image.resize(size, resample=Image.LANCZOS)
        pixels = torch.from_numpy(numpy.array(image)).to(device).permute(2, 0, 1)[None]
        pixels = pixels[
            ...,
            : pixels.shape[-2] // SPATIAL_RATIO * SPATIAL_RATIO,
            : pixels.shape[-1] // SPATIAL_RATIO * SPATIAL_RATIO,
        ]
        pixels = pixels.to(torch.float32) / 255.0
        pictures.append(pixels[0].permute(1, 2, 0).cpu().contiguous())
        latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(
            1, -1, 1, 1
        )
        latents_std = torch.tensor(vae.config.latents_std, device=device).view(
            1, -1, 1, 1
        )
        with torch.no_grad():
            posterior = vae.encode(
                (pixels - pixel_mean) / pixel_std, return_dict=False
            )[0]
        means.append(
            ((posterior.mean - latents_mean) / latents_std)[0, :, None]
            .cpu()
            .contiguous()
        )
        latent = posterior.sample(generator=generator).to(torch.float16).float()
        latent = latent[..., : latent.shape[-2] // 2 * 2, : latent.shape[-1] // 2 * 2]
        reference_latents.append(((latent - latents_mean) / latents_std).cpu())

    latent_height, latent_width = (
        args.height // SPATIAL_RATIO,
        args.width // SPATIAL_RATIO,
    )
    position_ids, token_tags, image_indices, text_indices, reference_rows = (
        build_image_packed_sequence(
            text_tags,
            [tuple(latent.shape[-2:]) for latent in reference_latents],
            latent_height,
            latent_width,
            PATCH,
            IMAGE_TAG,
        )
    )
    sequence = position_ids.shape[0]

    noise = torch.randn(
        (1, LATENT_CHANNELS, 1, latent_height, latent_width),
        generator=generator,
        dtype=torch.float32,
    )
    augmented = []
    for latent in reference_latents:
        reference_noise = torch.randn(
            latent.shape, generator=generator, dtype=torch.float32
        )
        timestep = torch.tensor(REFERENCE_TIMESTEP, dtype=torch.float32)
        augmented.append(
            (timestep * latent + (1.0 - timestep) * reference_noise)[:, :, None]
        )

    def unpatchify(rows):
        grid = rows.reshape(
            latent_height // 2, latent_width // 2, LATENT_CHANNELS, 2, 2
        )
        latent = grid.permute(2, 0, 3, 1, 4).reshape(
            LATENT_CHANNELS, latent_height, latent_width
        )
        return latent[None, :, None].contiguous()

    transformer = MiniMaxH3ImageTransformer2DModel.from_pretrained(
        os.path.join(args.repo, "transformer_diffusers"),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    watched = [int(index) for index in args.watched_blocks.split(",") if index]
    captured = {}

    def hook_for(index):
        def hook(_module, _inputs, output):
            if capture:
                captured[index] = output[0].float().cpu()

        return hook

    for index in watched:
        transformer.transformer_blocks[index].register_forward_hook(hook_for(index))

    tensors = {"noise.video": noise}
    for index, latent in enumerate(augmented):
        tensors[f"reference.{index}.augmented"] = latent[0].contiguous()
    conditions = (
        torch.cat([patchify_video_latents(latent, PATCH) for latent in augmented]).to(
            device
        )
        if augmented
        else None
    )
    latents = patchify_video_latents(noise, PATCH).to(device)
    for step in range(steps):
        capture = step == args.capture_step
        row_timesteps = torch.full(
            (sequence,), float(timesteps[step]), dtype=torch.float32
        )
        row_timesteps[image_indices[:reference_rows]] = REFERENCE_TIMESTEP
        timestep, timestep_indices = torch.unique(
            row_timesteps, sorted=True, return_inverse=True
        )
        if capture:
            tensors[f"step{step}.input.video"] = unpatchify(latents.cpu())
        rows = latents if conditions is None else torch.cat([conditions, latents])
        with torch.no_grad():
            prediction = transformer(
                hidden_states=rows[None],
                encoder_hidden_states=context,
                timestep=timestep.to(device),
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags.to(device),
                position_ids=position_ids.to(device),
                image_indices=image_indices.to(device),
                text_indices=text_indices.to(device),
                pdd_step_index=step,
                return_dict=False,
            )[0][0, reference_rows:].float()
        if capture:
            tensors[f"step{step}.velocity.video"] = unpatchify(prediction.cpu())
            for index, residual in captured.items():
                tensors[f"step{step}.block.{index}"] = residual
        latents = latents + (sigmas[step] - sigmas[step + 1]).item() * prediction
        tensors[f"step{step}.latent.video"] = unpatchify(latents.cpu())
        print(f"step {step} at sigma {sigmas[step].item():.6f}")
    del transformer
    torch.cuda.empty_cache()

    metadata = {
        "width": str(args.width),
        "height": str(args.height),
        "frames": "1",
        "seed": str(args.seed),
        "steps": str(steps),
        "sigmas_video": json.dumps([float(sigma) for sigma in sigmas]),
        "shift_video": "1.0",
        "shift_audio": "1.0",
        "capture_step": str(args.capture_step),
        "watched_blocks": json.dumps(watched),
        "block_row_stride": "1",
        "decode": str(not args.no_decode),
        "reference_pictures": str(len(args.reference)),
    }
    save_file(tensors, os.path.join(args.out, "dit.safetensors"), metadata)
    if args.reference:
        references = {}
        for index, (picture, mean) in enumerate(zip(pictures, means)):
            references[f"reference.{index}.pixels"] = picture
            references[f"reference.{index}.latent"] = mean
        save_file(
            references, os.path.join(args.out, "references.safetensors"), metadata
        )

    if args.no_decode:
        return
    vae = vae or load_vae()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1)
    latent = tensors[f"step{steps - 1}.latent.video"][:, :, 0].to(device)
    with torch.no_grad():
        image = vae.decode(latent * std + mean, return_dict=False)[0].float()
    pixels = (image * pixel_std + pixel_mean).clamp(0, 1)
    save_file(
        {"video": pixels[:, :, None].cpu().contiguous()},
        os.path.join(args.out, "decode.safetensors"),
        metadata,
    )
    print(f"decoded {tuple(pixels.shape)}")


if __name__ == "__main__":
    main()
