"""Writes golden data for a tiny MiniMax H3 audio VAE encoder from ComfyUI's code.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3, like audio_vae_tiny.py:

    python3 tools/golden/audio_encoder_tiny.py --comfyui /path/to/ComfyUI \\
        --out tests/fixtures/audio_encoder_tiny.safetensors

The encoder keeps the released strides, so 800 samples still give one latent frame, but
it has 4 channels at the first level and 256 features, and its head pools 32 values per
group onto 4 latent channels as the released one pools onto 32. Small weights keep the
Snake activations in a sane range. The file holds the encoder weights with the
checkpoint's names, the input waveform, the outputs of the first convolution, the first
residual unit and the first block, the features the head sees, and the normalized
posterior mean, all computed in FP32 on the CPU.
"""

import argparse
import math
import sys

import torch
from safetensors.torch import save_file

ENCODER_CHANNELS = 2
FEATURES = 256
LATENT_CHANNELS = 4
FRAMES = 6
SAMPLES_PER_LATENT = 800


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
        encoder_dim=ENCODER_CHANNELS,
        latent_dim=FEATURES,
        vae_latent_channels=LATENT_CHANNELS,
    )
    model.requires_grad_(False)

    weights = {}
    for name, tensor in model.named_parameters():
        if not name.startswith(("encoder.", "pre_block.", "mean_proj.")):
            continue
        if name.endswith(".alpha"):
            value = 0.5 + 0.2 * torch.randn(tensor.shape, generator=generator)
        elif name.endswith(".bias"):
            value = 0.1 * torch.randn(tensor.shape, generator=generator)
        elif ".norm" in name and name.endswith(".weight"):
            value = 1.0 + 0.1 * torch.randn(tensor.shape, generator=generator)
        else:
            value = torch.randn(tensor.shape, generator=generator) / math.sqrt(
                tensor[0].numel()
            )
            if ".block." in name:
                value *= 0.3
        weights[name] = value
        with torch.no_grad():
            tensor.copy_(value)
    for name, buffer in model.named_buffers():
        if name == "pre_block.attn.zero_k_bias":
            # The module registers it as an empty buffer, the released checkpoint holds zeros.
            value = torch.zeros(buffer.shape)
            weights[name] = value
            buffer.copy_(value)
        elif name in ("latents_mean", "latents_std"):
            value = (
                torch.randn(LATENT_CHANNELS, generator=generator) * 0.5
                if name == "latents_mean"
                else 1.0 + torch.rand(LATENT_CHANNELS, generator=generator)
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

    model.encoder.block[0].register_forward_hook(capture("conv_in"))
    model.encoder.block[1].block[0].register_forward_hook(capture("unit0"))
    model.encoder.block[1].register_forward_hook(capture("block0"))
    model.encoder.register_forward_hook(capture("features"))
    waveform = torch.randn(
        (1, 2, FRAMES * SAMPLES_PER_LATENT), generator=generator
    ).clamp(-1.0, 1.0)
    with torch.inference_mode():
        latent = model.encode(waveform)

    tensors = {f"weight.{name}": value.contiguous() for name, value in weights.items()}
    tensors["input.waveform"] = waveform.contiguous()
    for name, value in captured.items():
        tensors[f"intermediate.{name}"] = value.contiguous()
    tensors["output.latent"] = latent.contiguous()
    save_file(tensors, arguments.out)
    print(
        f"wrote {len(tensors)} tensors, latent {tuple(latent.shape)}, "
        f"range {latent.min().item():.3f}..{latent.max().item():.3f}"
    )


if __name__ == "__main__":
    main()
