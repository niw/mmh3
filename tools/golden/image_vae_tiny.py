"""Writes golden data for a tiny MiniMax H3 Image picture VAE decoder from diffusers' code.

Development-only tool. ComfyUI blends a tile against neighbors it has already blended, while the
release and diffusers blend against their unblended decodes, so the reference here is diffusers.
The latent statistics, which the checkpoint does not carry, come from ComfyUI's source. Run it
with checkouts of both that have MiniMax H3 Image:

    uv run --with torch --with numpy --with safetensors --with huggingface_hub \\
        --with filelock --with regex --with requests --with pillow \\
        python3 tools/golden/image_vae_tiny.py --diffusers /path/to/diffusers \\
        --comfyui /path/to/ComfyUI --out tests/fixtures/image_vae_tiny.safetensors

The weights are random, F16 with the checkpoint's names and layouts (`to_qkv` interleaved per
head, `ff.w1` as gate then up), and reach diffusers' AutoencoderKLMiniMaxH3Image through its own
conversion. A 24 × 40 latent decodes as 2 × 3 tiles of 256 pixels whose overlaps take the slack,
and a 12 × 20 one as 1 × 2 tiles of 192 × 256 pixels. The file holds DiT-space latents and the
pixels in [0, 1] after the pipeline's ImageNet de-normalization, computed in FP32 on the CPU,
and the latent statistics.
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import sys

import torch
from safetensors.torch import save_file

LATENT_SHAPES = {"": (1, 24, 24, 40), "_small": (1, 24, 12, 20)}
DECODER = {"num_layers": 2, "heads": 2, "dim_head": 64, "registers": 4}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 1, 1, 1),
        decoder_num_layers=DECODER["num_layers"],
        decoder_num_attention_heads=DECODER["heads"],
        decoder_attention_head_dim=DECODER["dim_head"],
        decoder_num_register_tokens=DECODER["registers"],
        latents_mean=latents_mean,
        latents_std=latents_std,
    ).eval()
    config = {
        "decoder_num_attention_heads": DECODER["heads"],
        "decoder_attention_head_dim": DECODER["dim_head"],
    }

    generator = torch.Generator().manual_seed(arguments.seed)
    dim = DECODER["heads"] * DECODER["dim_head"]
    shapes = {
        "post_quant_conv.weight": (24, 24, 1, 1),
        "post_quant_conv.bias": (24,),
        "decoder.x_embedder.weight": (dim, 24),
        "decoder.x_embedder.bias": (dim,),
        "decoder.register_tokens": (1, DECODER["registers"], dim),
        "decoder.norm_out.weight": (dim,),
        "decoder.norm_out.bias": (dim,),
        "decoder.proj_out.weight": (768, dim),
        "decoder.proj_out.bias": (768,),
    }
    for layer in range(DECODER["num_layers"]):
        prefix = f"decoder.transformer_blocks.{layer}"
        shapes |= {
            f"{prefix}.norm1.weight": (dim,),
            f"{prefix}.attn.to_qkv.weight": (3 * dim, dim),
            f"{prefix}.attn.to_qkv.bias": (3 * dim,),
            f"{prefix}.attn.to_out.weight": (dim, dim),
            f"{prefix}.attn.to_out.bias": (dim,),
            f"{prefix}.scale1": (dim,),
            f"{prefix}.norm2.weight": (dim,),
            f"{prefix}.ff.w1.weight": (8 * dim, dim),
            f"{prefix}.ff.w1.bias": (8 * dim,),
            f"{prefix}.ff.w2.weight": (dim, 4 * dim),
            f"{prefix}.ff.w2.bias": (dim,),
            f"{prefix}.scale2": (dim,),
        }

    weights = {}
    for name, shape in shapes.items():
        if name.endswith(("norm1.weight", "norm2.weight", "norm_out.weight")):
            value = 1.0 + 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith(("scale1", "scale2")):
            value = 0.5 + 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith((".bias", "register_tokens")):
            value = 0.1 * torch.randn(shape, generator=generator)
        else:
            fan_in = math.prod(shape[1:])
            value = torch.randn(shape, generator=generator) / math.sqrt(fan_in)
        # Keep most decoded pixels inside [0, 1], where the clamp hides nothing.
        if name == "decoder.proj_out.weight":
            value = value * 0.3
        weights[name] = value.to(torch.float16)

    state = {}
    for name, value in weights.items():
        for target, tensor in convert.convert_video_vae_key(
            name, value.float(), config
        ):
            state[target] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [
        name for name in missing if not name.startswith(("encoder.", "quant_conv."))
    ]
    assert not missing and not unexpected, (missing, unexpected)

    mean = torch.tensor(latents_mean).view(1, -1, 1, 1)
    std = torch.tensor(latents_std).view(1, -1, 1, 1)
    pixel_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    pixel_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensors = {f"weight.{name}": value.contiguous() for name, value in weights.items()}
    tensors["reference.latents_mean"] = mean.flatten().contiguous()
    tensors["reference.latents_std"] = std.flatten().contiguous()
    for suffix, shape in LATENT_SHAPES.items():
        latent = torch.randn(shape, generator=generator)
        with torch.inference_mode():
            decoded = model.decode(latent * std + mean).sample
        pixels = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
        clamped = ((decoded * pixel_std + pixel_mean - pixels).abs() > 0).float().mean()
        tensors[f"input.latent{suffix}"] = latent.contiguous()
        tensors[f"output.pixels{suffix}"] = pixels.contiguous()
        print(f"latent {shape}: pixels {tuple(pixels.shape)}, {clamped:.1%} clamped")
    save_file(tensors, arguments.out, metadata={"decoder": json.dumps(DECODER)})
    print(f"wrote {len(tensors)} tensors")


if __name__ == "__main__":
    main()
