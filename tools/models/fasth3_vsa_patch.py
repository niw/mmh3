# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2", "torch>=2.9"]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# ///
"""Builds a patch that turns the MiniMax H3 FL2VA DiT into FastVideo's FastH3 VSA-DataFree.

Development tool. It reads the full BF16 FL2VA DiT in ComfyUI's layout, such as
diffusion_models/minimax_h3_fl2va_bf16.safetensors of Comfy-Org/MiniMax-H3, and the
transformer directory of FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree in
diffusers' layout, and writes one safetensors file that mmh3 applies with --patch on top of
the pruned INT8 ConvRot FL2VA DiT. It computes on a CUDA GPU. Run it with uv, which installs
numpy and PyTorch for CUDA 13.0, for example:

    uv run tools/models/fasth3_vsa_patch.py \\
        --base /path/to/minimax_h3_fl2va_bf16.safetensors \\
        --fasth3 /path/to/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/transformer \\
        --out models/patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors

The patch is a LoRA plus tensors that no low-rank update can carry. It holds, under the prefix
diffusion_model.:

- lora_A.weight, lora_B.weight and alpha in BF16 for the linear layers of the blocks and
  of the text refiner: the rank --rank truncated SVD of the fine-tuned weight minus the
  base weight, with alpha equal to the rank. FastH3 keeps FP32 master weights and ships
  BF16 ones, so these updates are the base weights moved by one or a few BF16 steps where
  the training moved them far enough, about 1e-4 of the weights in norm and far from low
  rank. --rank 0 leaves them out.
- Tensors that replace the base's or add to them, under their checkpoint names and in the
  dtypes of the pruned checkpoints: norms, the text projection, the patch and output
  projections, the pruned AdaLN and the VSA gates. The pruned AdaLN is a rank-8 basis of
  the time-embedding curve silu(time_embedder(t)) on 1025 timesteps t in [0, 1]
  (adaln_t_table, after subtracting the curve's mean) with every AdaLN projection folded
  onto it and the mean folded into its bias. The gates to_gate_compress are rotated and
  quantized to INT8 like the base's INT8 ConvRot layers.
"""

import argparse
import json
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from checkpoint import CONVROT_GROUP, SafeTensors, hadamard, write_safetensors

PREFIX = "diffusion_model."
CURVE_POINTS = 1025
CURVE_RANK = 8
DEVICE = torch.device("cuda")


class ShardedSafeTensors:
    """The shards of a diffusers model directory, through its index."""

    def __init__(self, directory):
        directory = Path(directory)
        index = json.loads(
            (directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
        )
        self.files = {}
        self.owner = {}
        for name, shard in index["weight_map"].items():
            if shard not in self.files:
                self.files[shard] = SafeTensors(directory / shard)
            self.owner[name] = self.files[shard]

    def __contains__(self, name):
        return name in self.owner

    def dtype(self, name):
        return self.owner[name].dtype(name)

    def read(self, name):
        return self.owner[name].read(name)

    def load(self, name):
        return self.owner[name].load(name)


def stored_tensor(file, name):
    """A tensor of a checkpoint in host memory, in its stored dtype."""
    values = torch.from_numpy(file.read(name))
    return values.view(torch.bfloat16) if file.dtype(name) == "BF16" else values


def read_ahead(items, read, depth=4):
    """Yields read(item) for each item while threads read the next `depth` items."""
    with ThreadPoolExecutor(max_workers=depth) as pool:
        pending = deque(pool.submit(read, item) for item in items[:depth])
        for index in range(len(items)):
            values = pending.popleft().result()
            if index + depth < len(items):
                pending.append(pool.submit(read, items[index + depth]))
            yield values


def bf16_bytes(values):
    """Little-endian bytes of a tensor rounded to BF16, to nearest even."""
    return values.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()


def encode(values, dtype):
    """Little-endian bytes of float32 values stored as `dtype`."""
    if dtype == "BF16":
        return bf16_bytes(torch.from_numpy(np.asarray(values, dtype=np.float32)))
    if dtype == "F16":
        return np.asarray(values, dtype=np.float16).tobytes()
    if dtype == "F32":
        return np.asarray(values, dtype=np.float32).tobytes()
    if dtype == "I8":
        return np.asarray(values, dtype=np.int8).tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def quantize_convrot(weight):
    """Rotates the rows of a GPU weight [n, k] in 256-column groups and quantizes each row to
    INT8 with scale max |w| / 127, rounding half to even. Returns (int8, scales [n, 1])."""
    rows, columns = weight.shape
    rotation = torch.from_numpy(hadamard()).to(DEVICE)
    rotated = (
        weight.double().reshape(rows, columns // CONVROT_GROUP, CONVROT_GROUP)
        @ rotation
    ).reshape(rows, columns)
    scales = torch.clamp(rotated.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-30)
    quantized = torch.clamp(torch.round(rotated / scales), -128, 127).to(torch.int8)
    return quantized.cpu().numpy(), scales.float().cpu().numpy()


def truncated_svd(matrix, rank, oversampling=16, iterations=3, seed=0):
    """The top `rank` singular triplets of a float32 GPU matrix by randomized SVD with power
    iterations."""
    generator = np.random.default_rng(seed)
    probe = generator.standard_normal(
        (matrix.shape[1], rank + oversampling), dtype=np.float32
    )
    basis, _ = torch.linalg.qr(matrix @ torch.from_numpy(probe).to(DEVICE))
    for _ in range(iterations):
        basis, _ = torch.linalg.qr(matrix.T @ basis)
        basis, _ = torch.linalg.qr(matrix @ basis)
    left, singular, right = torch.linalg.svd(basis.T @ matrix, full_matrices=False)
    return (basis @ left)[:, :rank], singular[:rank], right[:rank]


def linear_layers(prefix, fasth3_prefix):
    """(ComfyUI name, diffusers names, whether fc1's halves swap) of a block's linears."""
    attention = f"{fasth3_prefix}.attn"
    return [
        (
            f"{prefix}.attn.qkv_proj",
            [
                f"{attention}.to_q.weight",
                f"{attention}.to_k.weight",
                f"{attention}.to_v.weight",
            ],
            False,
        ),
        (f"{prefix}.attn.out_proj", [f"{attention}.to_out.0.weight"], False),
        # NOTE: diffusers' SwiGLU takes [up; gate] and ComfyUI's [gate; up].
        (f"{prefix}.mlp.fc1", [f"{fasth3_prefix}.ff.net.0.proj.weight"], True),
        (f"{prefix}.mlp.fc2", [f"{fasth3_prefix}.ff.net.2.weight"], False),
    ]


def replaced_tensors(layers, refiner_layers):
    """(ComfyUI name, diffusers name) of the tensors the patch replaces as they are."""
    pairs = [
        ("condition_proj.weight", "context_embedder.weight"),
        ("condition_proj.bias", "context_embedder.bias"),
        ("video_patch_proj.weight", "proj_in.weight"),
        ("video_patch_proj.bias", "proj_in.bias"),
        ("audio_patch_proj.weight", "audio_proj_in.weight"),
        ("audio_patch_proj.bias", "audio_proj_in.bias"),
        ("final_layer.video_out.weight", "proj_out.weight"),
        ("final_layer.video_out.bias", "proj_out.bias"),
        ("final_layer.audio_out.weight", "audio_proj_out.weight"),
        ("final_layer.audio_out.bias", "audio_proj_out.bias"),
        ("final_layer.norm.weight", "norm_out.norm.weight"),
        ("token_refiner.final_norm.weight", "token_refiner.final_norm.weight"),
    ]
    for prefix, fasth3_prefix, count in [
        ("blocks", "transformer_blocks", layers),
        ("token_refiner.blocks", "token_refiner.refiner_blocks", refiner_layers),
    ]:
        for index in range(count):
            block, source = f"{prefix}.{index}", f"{fasth3_prefix}.{index}"
            pairs += [
                (f"{block}.norm1.weight", f"{source}.norm1.weight"),
                (f"{block}.norm2.weight", f"{source}.norm2.weight"),
                (f"{block}.attn.q_norm.weight", f"{source}.attn.norm_q.weight"),
                (f"{block}.attn.k_norm.weight", f"{source}.attn.norm_k.weight"),
            ]
    return pairs


def time_curve(fasth3):
    """silu(time_embedder(t)) on CURVE_POINTS timesteps in [0, 1], [points, dim] float64."""
    t = np.linspace(0.0, 1.0, CURVE_POINTS)
    half = fasth3.load("time_embedder.linear_1.weight").shape[1] // 2
    frequencies = np.exp(-np.log(10000.0) * np.arange(half) / half)
    arguments = t[:, None] * frequencies[None]
    embedding = np.concatenate([np.cos(arguments), np.sin(arguments)], axis=1)

    def silu(values):
        return values / (1.0 + np.exp(-values))

    def linear(name, values):
        weight = fasth3.load(f"{name}.weight").astype(np.float64)
        return values @ weight.T + fasth3.load(f"{name}.bias").astype(np.float64)

    hidden = silu(linear("time_embedder.linear_1", embedding))
    return silu(linear("time_embedder.linear_2", hidden))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", required=True, help="full BF16 FL2VA DiT")
    parser.add_argument("--fasth3", required=True, help="FastH3 transformer directory")
    parser.add_argument("--out", required=True, help="patch file to write")
    parser.add_argument("--rank", type=int, default=64, help="a multiple of 64")
    parser.add_argument(
        "--layers", type=int, default=None, help="convert only the first N blocks"
    )
    arguments = parser.parse_args()
    if arguments.rank % 64 != 0:
        sys.exit("--rank must be a multiple of 64")

    base = SafeTensors(arguments.base)
    fasth3 = ShardedSafeTensors(arguments.fasth3)
    layers = sum(1 for name in base.header if name.endswith(".adaln_proj.linear.bias"))
    layers -= 1  # the final layer's
    refiner_layers = sum(
        1
        for name in base.header
        if name.startswith("token_refiner.blocks.") and name.endswith(".norm1.weight")
    )
    converted_layers = layers if arguments.layers is None else arguments.layers
    tensors = {}
    report = []

    # Low-rank updates of the linear layers.
    blocks = [
        (f"blocks.{index}", f"transformer_blocks.{index}")
        for index in range(converted_layers)
    ] + [
        (f"token_refiner.blocks.{index}", f"token_refiner.refiner_blocks.{index}")
        for index in range(refiner_layers)
    ]
    linears = [
        linear
        for prefix, fasth3_prefix in blocks
        for linear in linear_layers(prefix, fasth3_prefix)
    ]
    if arguments.rank == 0:
        linears = []

    def read_linear(linear):
        name, sources, _ = linear
        fine_tuned = [stored_tensor(fasth3, source) for source in sources]
        return fine_tuned, stored_tensor(base, f"{name}.weight")

    for (name, _, swap_halves), (fine_tuned, weight) in zip(
        linears, read_ahead(linears, read_linear)
    ):
        fine_tuned = torch.cat([values.to(DEVICE).float() for values in fine_tuned])
        if swap_halves:
            fine_tuned = torch.cat(fine_tuned.chunk(2)[::-1])
        weight = weight.to(DEVICE).float()
        update = fine_tuned - weight
        del fine_tuned
        relative = float(torch.linalg.norm(update) / torch.linalg.norm(weight))
        if relative > 0.5:
            sys.exit(
                f"{name}: the checkpoints differ by {relative:.2f}, wrong mapping?"
            )
        total = float(torch.sum(update.double() ** 2))
        rank = arguments.rank
        left, singular, right = truncated_svd(update, rank)
        kept = float(torch.sum(singular.double() ** 2) / total)
        root = torch.sqrt(singular)
        tensors[f"{PREFIX}{name}.lora_B.weight"] = (
            "BF16",
            (weight.shape[0], rank),
            bf16_bytes(left * root),
        )
        tensors[f"{PREFIX}{name}.lora_A.weight"] = (
            "BF16",
            (rank, weight.shape[1]),
            bf16_bytes(root[:, None] * right),
        )
        tensors[f"{PREFIX}{name}.alpha"] = (
            "BF16",
            (),
            encode([float(rank)], "BF16"),
        )
        report.append((name, relative, kept))
        print(
            f"{name:40} update {relative:.3e} of the weight, "
            f"rank {rank} keeps {kept:.3f} of it",
            flush=True,
        )

    # Tensors replaced as they are, in the base's dtypes.
    for name, source in replaced_tensors(converted_layers, refiner_layers):
        values = fasth3.load(source)
        if np.array_equal(values, base.load(name)):
            continue
        tensors[f"{PREFIX}{name}"] = (
            base.dtype(name),
            values.shape,
            encode(values, base.dtype(name)),
        )

    # The pruned AdaLN over a basis of the fine-tuned time-embedding curve.
    curve = time_curve(fasth3)
    mean = curve.mean(axis=0)
    _, singular, right = np.linalg.svd(curve - mean, full_matrices=False)
    basis = right[:CURVE_RANK].T
    table = (curve - mean) @ basis
    print(
        f"time curve: rank {CURVE_RANK} keeps "
        f"{1 - np.sum(singular[CURVE_RANK:] ** 2) / np.sum(singular**2):.9f} of the centered energy",
        flush=True,
    )
    tensors[f"{PREFIX}adaln_t_table"] = ("F32", table.shape, encode(table, "F32"))
    projections = [
        (
            f"blocks.{index}.adaln_proj.linear",
            f"transformer_blocks.{index}.adaln_proj.linear",
        )
        for index in range(converted_layers)
    ] + [("final_layer.adaln_proj.linear", "norm_out.linear")]
    worst = 0.0
    # Every eighth timestep is enough to measure the fit.
    curve_samples = torch.from_numpy(curve[::8].copy()).to(DEVICE)
    table_samples = torch.from_numpy(table[::8].copy()).to(DEVICE)
    basis = torch.from_numpy(basis.copy()).to(DEVICE)
    mean = torch.from_numpy(mean).to(DEVICE)
    for (name, _), (weight, bias) in zip(
        projections,
        read_ahead(
            projections,
            lambda projection: [
                stored_tensor(fasth3, f"{projection[1]}.{suffix}")
                for suffix in ("weight", "bias")
            ],
        ),
    ):
        weight = weight.to(DEVICE).double()
        bias = bias.to(DEVICE).double()
        folded_weight = (weight @ basis).half()
        folded_bias = (bias + weight @ mean).half()
        exact = curve_samples @ weight.T + bias
        approximate = table_samples @ folded_weight.double().T + folded_bias
        worst = max(
            worst,
            float(torch.linalg.norm(approximate - exact) / torch.linalg.norm(exact)),
        )
        tensors[f"{PREFIX}{name}.weight"] = (
            "F16",
            tuple(folded_weight.shape),
            folded_weight.cpu().numpy().tobytes(),
        )
        tensors[f"{PREFIX}{name}.bias"] = (
            "F16",
            tuple(folded_bias.shape),
            folded_bias.cpu().numpy().tobytes(),
        )
    print(f"pruned AdaLN: modulations within {worst:.3e} of the full ones", flush=True)

    # The VSA gates, which the base does not have, in INT8 ConvRot.
    gates = list(range(converted_layers))
    for index, weight in zip(
        gates,
        read_ahead(
            gates,
            lambda index: stored_tensor(
                fasth3, f"transformer_blocks.{index}.attn.to_gate_compress.weight"
            ),
        ),
    ):
        name = f"blocks.{index}.attn.to_gate_compress"
        quantized, scales = quantize_convrot(weight.to(DEVICE))
        tensors[f"{PREFIX}{name}.weight"] = ("I8", quantized.shape, quantized.tobytes())
        tensors[f"{PREFIX}{name}.weight_scale"] = (
            "F32",
            scales.shape,
            scales.tobytes(),
        )

    metadata = {
        "source": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
        "base": Path(arguments.base).name,
        "rank": str(arguments.rank),
        "note": "A patch of LoRA and replacement tensors for the pruned INT8 ConvRot FL2VA "
        "DiT, modified from the FastH3 checkpoint.",
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(arguments.out, tensors, metadata)
    print(
        f"wrote {arguments.out}: {len(report)} low-rank layers, {len(tensors)} tensors"
    )


if __name__ == "__main__":
    main()
