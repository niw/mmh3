"""Writes golden data for a tiny MiniMax H3 video VAE decoder computed by ComfyUI's implementation.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that supports MiniMax H3, like
dit_tiny.py:

    python3 tools/golden/vae_tiny.py --comfyui /path/to/ComfyUI --out tests/fixtures/vae_tiny.safetensors

ComfyUI's MiniMaxH3VideoVAE gets a random two-layer ViT decoder and 32-pixel tiles, so a 12-frame latent of 4 × 6
decodes as 15 spatial tiles in two temporal chunks and exercises every blend. The file holds the weights with the
checkpoint's names in F16, the input latent, the raw decoder output of the first tile and the final pixels in
[0, 1], all computed in FP32 on the CPU.
"""

import argparse
import json
import math
import sys

import torch
from safetensors.torch import save_file

LATENT_SHAPE = (1, 24, 12, 4, 6)
TILE_SIZE = 32
TILE_OVERLAP_MIN = 16
DECODER = {"num_layers": 2, "heads": 2, "dim_head": 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.ops
    from comfy.ldm.minimax import vae as h3_vae

    generator = torch.Generator().manual_seed(arguments.seed)
    model = h3_vae.MiniMaxH3VideoVAE(tile_size=TILE_SIZE, tile_overlap_min=TILE_OVERLAP_MIN)
    model.decoder = h3_vae.ViT3DDecoder(**DECODER, operations=comfy.ops.disable_weight_init)
    model.requires_grad_(False)

    weights = {}
    parameters = dict(model.decoder.named_parameters(prefix="decoder"))
    parameters.update(dict(model.post_quant_conv.named_parameters(prefix="post_quant_conv")))
    for name, tensor in parameters.items():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm_out.weight"):
            value = 1.0 + 0.1 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith("scale1") or name.endswith("scale2"):
            value = 0.5 + 0.1 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith(".bias") or name.endswith("register_tokens") or name.endswith("mask_token"):
            value = 0.1 * torch.randn(tensor.shape, generator=generator)
        else:
            fan_in = tensor[0].numel()
            value = torch.randn(tensor.shape, generator=generator) / math.sqrt(fan_in)
        value = value.to(torch.float16)
        weights[name] = value
        with torch.no_grad():
            tensor.copy_(value.float())
    for name in ("latents_mean", "latents_std"):
        base = torch.randn(24, generator=generator) * 0.5 if name == "latents_mean" else 1.0 + torch.rand(24, generator=generator)
        value = base.to(torch.float16)
        weights[name] = value
        getattr(model, name).copy_(value.float())

    captured = []
    decode_pixels = model._decode_pixels

    def capture(z):
        output = decode_pixels(z)
        if not captured:
            captured.append(output.clone())
        return output

    model._decode_pixels = capture
    latent = torch.randn(LATENT_SHAPE, generator=generator)
    with torch.inference_mode():
        pixels = model.decode(latent)

    tensors = {f"weight.{name}": value.contiguous() for name, value in weights.items()}
    tensors["input.latent"] = latent.contiguous()
    tensors["intermediate.first_tile"] = captured[0].contiguous()
    tensors["output.pixels"] = pixels.to(torch.float16).contiguous()
    metadata = {"tile_size": str(TILE_SIZE), "tile_overlap_min": str(TILE_OVERLAP_MIN), "decoder": json.dumps(DECODER)}
    save_file(tensors, arguments.out, metadata=metadata)
    print(f"wrote {len(tensors)} tensors, pixels {tuple(pixels.shape)}, first tile {tuple(captured[0].shape)}")


if __name__ == "__main__":
    main()
