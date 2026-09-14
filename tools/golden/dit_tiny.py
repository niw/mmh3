"""Writes golden data for a tiny MiniMax H3 DiT computed by ComfyUI's implementation.

Development-only tool. The engine and its tests never run Python. They read the file
this script writes. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3, for example:

    python3 tools/golden/dit_tiny.py --comfyui /path/to/ComfyUI \\
        --out tests/fixtures/dit_tiny.safetensors

The file holds random weights stored with the dtypes of the released pruned checkpoints,
the inputs, the refined text states, the hidden states after every block and the outputs
of one text-to-video forward pass in FP32 on the CPU.
"""

import argparse
import json
import math
import sys

import torch
from safetensors.torch import save_file

CONFIG = {
    "hidden_size": 192,
    "num_layers": 2,
    "token_refiner_num_layers": 1,
    "num_attention_heads": 2,
    "attention_head_dim": 128,
    "ffn_hidden_size": 256,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "text_dim": 64,
    "time_embed_dim": 8,
    "rope_inv_freq_len": 16,
    "adaln_curve_grid": 1025,
}
TEXT_TOKENS = 7
LATENT_FRAMES = 7
LATENT_HEIGHT = 4
LATENT_WIDTH = 6
AUDIO_FRAMES = 9
SIGMA = 0.7


def storage_dtype(name):
    """Dtype the released pruned checkpoints use for each tensor."""
    if (
        name in ("adaln_t_table", "rope.inv_freq")
        or "patch_proj" in name
        or name.startswith(("final_layer.video_out", "final_layer.audio_out"))
    ):
        return torch.float32
    if "adaln_proj" in name:
        return torch.float16
    return torch.bfloat16


def initialize(model, generator):
    state = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        shape = tensor.shape
        if name == "rope.inv_freq":
            value = 10000.0 ** (
                -2.0 * torch.arange(shape[0], dtype=torch.float32) / (2 * shape[0])
            )
        elif name == "adaln_t_table":
            grid = torch.linspace(0.0, 1.0, shape[0], dtype=torch.float32)[:, None]
            order = torch.arange(1, shape[1] + 1, dtype=torch.float32)[None, :]
            value = torch.cos(math.pi * grid * order) * 0.5 / order
        elif name.endswith(
            (
                "norm.weight",
                "norm1.weight",
                "norm2.weight",
                "q_norm.weight",
                "k_norm.weight",
            )
        ):
            value = 1.0 + 0.1 * torch.randn(shape, generator=generator)
        elif "adaln_proj" in name:
            value = 0.1 * torch.randn(shape, generator=generator)
        elif name.endswith(".bias"):
            value = 0.02 * torch.randn(shape, generator=generator)
        else:
            value = torch.randn(shape, generator=generator) / math.sqrt(shape[-1])
        value = value.to(storage_dtype(name))
        state[name] = value
        with torch.no_grad():
            tensor.copy_(value.to(torch.float32))
    return state


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
    from comfy.ldm.minimax import model as h3

    generator = torch.Generator().manual_seed(arguments.seed)
    model = h3.MiniMaxH3Model(
        **CONFIG,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    model.requires_grad_(False)
    weights = initialize(model, generator)

    video = torch.randn(
        (1, CONFIG["latents_dim"], LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH),
        generator=generator,
    )
    audio = torch.randn(
        (1, CONFIG["audio_latents_dim"], 2, AUDIO_FRAMES), generator=generator
    )
    context = torch.randn((1, TEXT_TOKENS, CONFIG["text_dim"]), generator=generator)
    timestep = torch.tensor([SIGMA * 1000.0], dtype=torch.float32)

    captured = {}
    model.token_refiner.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("text_states", output)
    )
    for index, block in enumerate(model.blocks):
        block.register_forward_hook(
            lambda module, inputs, output, index=index: captured.__setitem__(
                f"block.{index}", output.clone()
            )
        )
    with torch.inference_mode():
        video_out, audio_out = model._forward(
            [video, audio], timestep, context, transformer_options={}
        )

    tensors = {f"weight.{name}": value.contiguous() for name, value in weights.items()}
    tensors.update(
        {
            "input.video": video.contiguous(),
            "input.audio": audio.contiguous(),
            "input.context": context.contiguous(),
            "input.timestep": timestep,
            "intermediate.text_states": captured["text_states"].contiguous(),
            "output.video": video_out.contiguous(),
            "output.audio": audio_out.contiguous(),
        }
    )
    for index in range(CONFIG["num_layers"]):
        tensors[f"intermediate.block.{index}"] = captured[f"block.{index}"].contiguous()
    metadata = {
        "config": json.dumps(CONFIG),
        "sigma": str(SIGMA),
        "shift_video": "12.0",
        "shift_audio": "3.0",
    }
    save_file(tensors, arguments.out, metadata=metadata)
    print(f"wrote {len(tensors)} tensors to {arguments.out}")


if __name__ == "__main__":
    main()
