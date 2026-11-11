"""Writes golden data for a tiny MiniMax H3 Image picture VAE encoder from diffusers' code.

Development-only tool. An edit puts its references through this encoder, the video VAE's with
its own weights, and the reference here is diffusers, as for image_vae_tiny.py. The latent
statistics, which the checkpoint does not carry, come from ComfyUI's source. Run it with
checkouts of both that have MiniMax H3 Image:

    uv run --with torch --with numpy --with safetensors --with huggingface_hub \\
        --with filelock --with regex --with requests --with pillow \\
        python3 tools/golden/image_encoder_tiny.py --diffusers /path/to/diffusers \\
        --comfyui /path/to/ComfyUI --out tests/fixtures/image_encoder_tiny.safetensors

AutoencoderKLMiniMaxH3Image gets a random encoder with the released levels and downsampling at
32 and 64 channels, so the 16-pixel ratio and a shortcut convolution hold, and keeps its released
tiling of 256 pixels that overlap by at least 64. A 288 × 528 picture then encodes as 2 × 3
tiles whose overlaps take the slack. The file holds the weights in F16 under the checkpoint's
names, with a picture decoder's `decoder.proj_out.weight` and no latent statistics, as in
Comfy-Org's file; the picture's pixels in [0, 1], 8-bit values rounded to F16; the noise of a
posterior sample; and the posterior's mean and deviation, its normalized mean and the normalized
sample rounded to FP16 the way the edit pipeline takes it, computed in FP32 on the CPU.
"""

import argparse
import ast
import importlib.util
import math
import os
import sys

import torch
from safetensors.torch import save_file

CHANNELS = (32, 32, 32, 32, 64, 64)
HEIGHT, WIDTH = 288, 528
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def checkpoint_name(name):
    """The checkpoint's name of a diffusers encoder parameter, which the conversion undoes."""
    parts = name.split(".")
    if parts[1] == "down_blocks":
        level = parts[2]
        if parts[3] == "resnets":
            rest = [part.replace("conv_shortcut", "nin_shortcut") for part in parts[5:]]
            parts = [parts[0], "down", level, "block", parts[4], *rest]
        else:
            parts = [parts[0], "down", level, "downsample", *parts[5:]]
    return ".".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusers", required=True)
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    sys.path.insert(0, os.path.join(arguments.diffusers, "src"))
    from diffusers import AutoencoderKLMiniMaxH3Image

    script = os.path.join(
        arguments.diffusers, "scripts", "convert_minimax_h3_to_diffusers.py"
    )
    spec = importlib.util.spec_from_file_location("convert_minimax_h3", script)
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)

    with open(
        os.path.join(arguments.comfyui, "comfy", "ldm", "minimax", "vae.py")
    ) as source:
        statistics = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source.read()).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("LATENTS_MEAN", "LATENTS_STD")
        }
    latents_mean, latents_std = statistics["LATENTS_MEAN"], statistics["LATENTS_STD"]
    model = AutoencoderKLMiniMaxH3Image(
        block_out_channels=CHANNELS,
        layers_per_block=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=64,
        latents_mean=latents_mean,
        latents_std=latents_std,
    ).eval()
    model.requires_grad_(False)

    generator = torch.Generator().manual_seed(arguments.seed)
    weights = {}
    for name, tensor in model.named_parameters():
        if not name.startswith(("encoder.", "quant_conv.")):
            continue
        if name.endswith(".bias"):
            value = 0.1 * torch.randn(tensor.shape, generator=generator)
        elif ".norm" in name:
            value = 1.0 + 0.1 * torch.randn(tensor.shape, generator=generator)
        else:
            fan_in = math.prod(tensor.shape[1:])
            value = torch.randn(tensor.shape, generator=generator) / math.sqrt(fan_in)
        value = value.to(torch.float16)
        source = checkpoint_name(name)
        # The conversion must map the checkpoint's name back onto the parameter.
        assert [
            target for target, _ in convert.convert_video_vae_key(source, value, {})
        ] == [name], name
        weights[source] = value
        tensor.copy_(value.float())
    # Comfy-Org's picture VAE carries no latent statistics, which a decoder that turns a token
    # into one frame of 16 × 16 pixels stands for.
    weights["decoder.proj_out.weight"] = torch.zeros(
        (3 * 16 * 16, 8), dtype=torch.float16
    )

    pixels = (
        torch.randint(0, 256, (HEIGHT, WIDTH, 3), generator=generator).float() / 255.0
    ).half()
    # A smooth gradient under the noise, so the tiles differ in more than their noise.
    ramp = torch.linspace(0.0, 1.0, WIDTH).view(1, WIDTH, 1)
    pixels = ((pixels.float() + ramp) / 2.0).half()
    pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    model.enable_tiling()
    with torch.inference_mode():
        normalized = (pixels.float().permute(2, 0, 1)[None] - pixel_mean) / pixel_std
        posterior = model.encode(normalized, return_dict=False)[0]
    noise = torch.randn(posterior.mean.shape, generator=generator)
    mean = torch.tensor(latents_mean).view(1, -1, 1, 1)
    std = torch.tensor(latents_std).view(1, -1, 1, 1)
    sample = (posterior.mean + posterior.std * noise).half().float()

    # A frame axis, as the encoder gives a picture's latent.
    def frames(tensor):
        return tensor[0][:, None].contiguous()

    tensors = {name: value.contiguous() for name, value in weights.items()}
    tensors["input.pixels"] = pixels.contiguous()
    tensors["input.noise"] = frames(noise)
    tensors["output.mean"] = frames(posterior.mean)
    tensors["output.deviation"] = frames(posterior.std)
    tensors["output.latent"] = frames((posterior.mean - mean) / std)
    tensors["output.sampled_latent"] = frames((sample - mean) / std)
    save_file(tensors, arguments.out)
    print(
        f"wrote {len(tensors)} tensors, latent {tuple(posterior.mean.shape)}, range "
        f"{posterior.mean.min().item():.3f}..{posterior.mean.max().item():.3f}, deviation "
        f"{posterior.std.min().item():.3f}..{posterior.std.max().item():.3f}"
    )


if __name__ == "__main__":
    main()
