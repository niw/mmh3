"""Writes golden data for a tiny MiniMax H3 video VAE encoder from ComfyUI's code.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3, like vae_tiny.py:

    python3 tools/golden/video_encoder_tiny.py --comfyui /path/to/ComfyUI \\
        --out tests/fixtures/video_encoder_tiny.safetensors

ComfyUI's MiniMaxH3VideoVAE gets a random encoder with 32 channels at every level but the
released downsampling, so the 16-pixel and 4-frame ratios hold, and 32-pixel tiles. A
22-frame clip of 48 x 48 pixels then encodes as two temporal clips, the second padded by
repeating its last frame, over four spatial tiles, and gives seven latent frames after the
three dropped tokens. The file holds the weights in F16 under the checkpoint's names, which
the encoder reads without a prefix, the input frames in [0, 1], the moments of the whole
clip and of its first frame alone, and the normalized posterior means, all computed in FP32
on the CPU.
"""

import argparse
import sys

import torch
from safetensors.torch import save_file

CHANNELS = 32
FRAMES = 22
SIZE = 48
TILE_SIZE = 32
TILE_OVERLAP_MIN = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax import vae as h3_vae

    generator = torch.Generator().manual_seed(arguments.seed)
    model = h3_vae.MiniMaxH3VideoVAE(
        ch=CHANNELS,
        ch_mult=(1,) * 6,
        num_res_blocks=1,
        tile_size=TILE_SIZE,
        tile_overlap_min=TILE_OVERLAP_MIN,
    )
    model.requires_grad_(False)

    weights = {}
    for name, tensor in model.named_parameters():
        if not name.startswith(("encoder.", "quant_conv.")):
            continue
        if name.endswith(".bias"):
            value = 0.1 * torch.randn(tensor.shape, generator=generator)
        elif ".norm" in name:
            value = 1.0 + 0.1 * torch.randn(tensor.shape, generator=generator)
        else:
            fan_in = tensor[0].numel()
            value = torch.randn(tensor.shape, generator=generator) / fan_in**0.5
        weights[name] = value
        with torch.no_grad():
            tensor.copy_(value)
    for name in ("latents_mean", "latents_std"):
        weights[name] = getattr(model, name).clone()

    # Pixels in [0, 1], the layout mmh3 reads, with a moving pattern so the temporal taps
    # matter.
    frames = torch.rand((FRAMES, SIZE, SIZE, 3), generator=generator)
    for index in range(FRAMES):
        frames[index, :, :, 0] *= index / FRAMES
    pixels = frames.permute(3, 0, 1, 2).unsqueeze(0)

    with torch.inference_mode():
        clip = 2.0 * pixels - 1.0
        moments = model.encode_temporal(clip, torch.device("cpu"))
        latent = model.encode(clip)
        picture = clip[:, :, :1]
        picture_moments = model._adaptive_encode(model._normalize_pixels(picture))
        picture_latent = model.encode(picture)

    # The encoder loads the checkpoint's names as they are, without a prefix.
    tensors = {name: value.contiguous() for name, value in weights.items()}
    tensors["input.pixels"] = frames.contiguous()
    tensors["intermediate.moments"] = moments[0].contiguous()
    tensors["intermediate.picture_moments"] = picture_moments[0].contiguous()
    tensors["output.latent"] = latent[0].contiguous()
    tensors["output.picture_latent"] = picture_latent[0].contiguous()
    tensors = {
        name: value.float()
        if name.split(".")[0] in ("input", "intermediate", "output")
        else value.half()
        for name, value in tensors.items()
    }
    save_file(tensors, arguments.out)
    print(
        f"wrote {len(tensors)} tensors, latent {tuple(latent.shape)}, "
        f"moments {tuple(moments.shape)}, range "
        f"{latent.min().item():.3f}..{latent.max().item():.3f}"
    )


if __name__ == "__main__":
    main()
