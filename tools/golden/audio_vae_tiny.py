"""Writes golden data for a tiny MiniMax H3 audio VAE decoder from ComfyUI's code.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3, like dit_tiny.py:

    python3 tools/golden/audio_vae_tiny.py --comfyui /path/to/ComfyUI \\
        --out tests/fixtures/audio_vae_tiny.safetensors

ComfyUI's MiniMaxH3AudioVAE gets a BigVGAN decoder with the released upsampling rates
and kernels but 64 latent features and 128 initial channels, so the last stage has a
single channel. Small convolution weights keep the waveform inside the final clamp. The
file holds the decoder weights with the checkpoint's names, the input latent, the
outputs of conv_pre, the first upsampling layer and the first AMP block, and the stereo
waveform, all computed in FP32 on the CPU.
"""

import argparse
import math
import sys

import torch
from safetensors.torch import save_file

LATENT_SHAPE = (1, 32, 2, 6)
LATENT_FEATURES = 64
DECODER_CHANNELS = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    from comfy.ldm.minimax import audio_vae as h3_audio_vae

    generator = torch.Generator().manual_seed(arguments.seed)
    model = h3_audio_vae.MiniMaxH3AudioVAE(
        latent_dim=LATENT_FEATURES, decoder_dim=DECODER_CHANNELS
    )
    model.requires_grad_(False)

    weights = {}
    for name, tensor in model.named_parameters():
        if not name.startswith(("decoder.", "dec_in_proj.")):
            continue
        if name.endswith((".alpha", ".beta")):
            value = 0.2 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith(".bias"):
            value = 0.1 * torch.randn(tensor.shape, generator=generator)
        else:
            value = torch.randn(tensor.shape, generator=generator) / math.sqrt(
                tensor[0].numel()
            )
            if ".convs" in name:
                value *= 0.3
            elif name == "decoder.conv_post.weight":
                value *= 0.05
        weights[name] = value
        with torch.no_grad():
            tensor.copy_(value)
    for name, buffer in model.named_buffers():
        if name.endswith(".filter"):
            if name.startswith("decoder."):
                weights[name] = buffer.clone()
        elif name in ("latents_mean", "latents_std"):
            value = (
                torch.randn(32, generator=generator) * 0.5
                if name == "latents_mean"
                else 1.0 + torch.rand(32, generator=generator)
            )
            weights[name] = value
            buffer.copy_(value)

    captured = {}

    def capture(name):
        # A forward hook that returns a value replaces the module output, so this one
        # returns nothing.
        def hook(module, inputs, output):
            captured.setdefault(name, output.clone())

        return hook

    model.decoder.conv_pre.register_forward_hook(capture("conv_pre"))
    model.decoder.ups[0][0].register_forward_hook(capture("up0"))
    model.decoder.resblocks[0].register_forward_hook(capture("block0"))
    latent = torch.randn(LATENT_SHAPE, generator=generator)
    with torch.inference_mode():
        waveform = model.decode(latent)

    tensors = {f"weight.{name}": value.contiguous() for name, value in weights.items()}
    tensors["input.latent"] = latent.contiguous()
    for name, value in captured.items():
        tensors[f"intermediate.{name}"] = value.contiguous()
    tensors["output.waveform"] = waveform.contiguous()
    save_file(tensors, arguments.out)
    clipped = (waveform.abs() >= 1.0).float().mean().item()
    print(
        f"wrote {len(tensors)} tensors, waveform {tuple(waveform.shape)}, {clipped:.1%} clipped, "
        f"range {waveform.min().item():.3f}..{waveform.max().item():.3f}"
    )


if __name__ == "__main__":
    main()
