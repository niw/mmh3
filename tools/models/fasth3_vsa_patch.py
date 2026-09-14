# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Builds a patch that turns the MiniMax H3 FL2VA DiT into FastVideo's FastH3 VSA-DataFree.

Development tool. It reads the full BF16 FL2VA DiT in ComfyUI's layout, such as
diffusion_models/minimax_h3_fl2va_bf16.safetensors of Comfy-Org/MiniMax-H3, and the
transformer directory of FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree in
diffusers' layout, and writes one safetensors file that mmh3 applies with --patch on top of
the pruned INT8 ConvRot FL2VA DiT. Run it with uv, which installs numpy, for example:

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
import os
import struct
import sys
from pathlib import Path

import numpy as np

PREFIX = "diffusion_model."
CURVE_POINTS = 1025
CURVE_RANK = 8
CONVROT_GROUP = 256
DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "I8": np.int8,
    "U8": np.uint8,
}


class SafeTensors:
    """A safetensors file whose tensors are read one at a time.

    NOTE: The tensors are read into memory rather than mapped, and their pages are dropped
    from the page cache after each read, because a pass over two 70 GB checkpoints would
    otherwise fill the memory with mapped or cached pages."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            self.header = json.loads(file.read(length))
        self.header.pop("__metadata__", None)
        self.base = 8 + length

    def __contains__(self, name):
        return name in self.header

    def dtype(self, name):
        return self.header[name]["dtype"]

    def shape(self, name):
        return self.header[name]["shape"]

    def load(self, name):
        """The tensor as float32, or as stored for integer dtypes."""
        info = self.header[name]
        start, end = info["data_offsets"]
        dtype = np.dtype(DTYPES[info["dtype"]])
        values = np.empty((end - start) // dtype.itemsize, dtype=dtype)
        with open(self.path, "rb") as file:
            file.seek(self.base + start)
            file.readinto(memoryview(values).cast("B"))
            os.posix_fadvise(
                file.fileno(), self.base + start, end - start, os.POSIX_FADV_DONTNEED
            )
        values = values.reshape(info["shape"])
        if info["dtype"] == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        if info["dtype"] in ("F16", "F32"):
            return values.astype(np.float32)
        return values


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

    def load(self, name):
        return self.owner[name].load(name)


def to_bf16_bits(values):
    """Rounds float32 values to BF16, to nearest even, as uint16 bits."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


def encode(values, dtype):
    """Little-endian bytes of float32 values stored as `dtype`."""
    if dtype == "BF16":
        return to_bf16_bits(values).tobytes()
    if dtype == "F16":
        return np.asarray(values, dtype=np.float16).tobytes()
    if dtype == "F32":
        return np.asarray(values, dtype=np.float32).tobytes()
    if dtype == "I8":
        return np.asarray(values, dtype=np.int8).tobytes()
    raise ValueError(f"unsupported dtype {dtype}")


def write_safetensors(path, tensors, metadata):
    """Writes {name: (dtype, shape, bytes)} with string metadata."""
    header = {"__metadata__": metadata}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for _, _, data in tensors.values():
            output.write(data)


def hadamard():
    """The normalized regular Hadamard matrix of ConvRot's 256-column groups."""
    kernel = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64
    )
    matrix = kernel / 2
    for _ in range(3):
        matrix = np.kron(matrix, kernel / 2)
    return matrix


def quantize_convrot(weight):
    """Rotates the rows of a weight [n, k] in 256-column groups and quantizes each row to
    INT8 with scale max |w| / 127, rounding half to even. Returns (int8, scales [n, 1])."""
    rows, columns = weight.shape
    rotated = (
        weight.astype(np.float64).reshape(rows, columns // CONVROT_GROUP, CONVROT_GROUP)
        @ hadamard()
    ).reshape(rows, columns)
    scales = np.maximum(np.abs(rotated).max(axis=1, keepdims=True) / 127.0, 1e-30)
    quantized = np.clip(np.rint(rotated / scales), -128, 127).astype(np.int8)
    return quantized, scales.astype(np.float32)


def truncated_svd(matrix, rank, oversampling=16, iterations=3, seed=0):
    """The top `rank` singular triplets of a float32 matrix by randomized SVD with power
    iterations."""
    generator = np.random.default_rng(seed)
    probe = generator.standard_normal(
        (matrix.shape[1], rank + oversampling), dtype=np.float32
    )
    basis, _ = np.linalg.qr(matrix @ probe)
    for _ in range(iterations):
        basis, _ = np.linalg.qr(matrix.T @ basis)
        basis, _ = np.linalg.qr(matrix @ basis)
    left, singular, right = np.linalg.svd(basis.T @ matrix, full_matrices=False)
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
    for prefix, fasth3_prefix in blocks if arguments.rank > 0 else []:
        for name, sources, swap_halves in linear_layers(prefix, fasth3_prefix):
            fine_tuned = np.concatenate([fasth3.load(source) for source in sources])
            if swap_halves:
                half = fine_tuned.shape[0] // 2
                fine_tuned = np.concatenate([fine_tuned[half:], fine_tuned[:half]])
            weight = base.load(f"{name}.weight")
            update = fine_tuned - weight
            relative = np.linalg.norm(update) / np.linalg.norm(weight)
            if relative > 0.5:
                sys.exit(
                    f"{name}: the checkpoints differ by {relative:.2f}, wrong mapping?"
                )
            total = float(np.sum(update.astype(np.float64) ** 2))
            rank = arguments.rank
            left, singular, right = truncated_svd(update, rank)
            kept = float(np.sum(singular.astype(np.float64) ** 2) / total)
            root = np.sqrt(singular)
            tensors[f"{PREFIX}{name}.lora_B.weight"] = (
                "BF16",
                (weight.shape[0], rank),
                encode(left * root, "BF16"),
            )
            tensors[f"{PREFIX}{name}.lora_A.weight"] = (
                "BF16",
                (rank, weight.shape[1]),
                encode(root[:, None] * right, "BF16"),
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
    for name, source in projections:
        weight = fasth3.load(f"{source}.weight").astype(np.float64)
        bias = fasth3.load(f"{source}.bias").astype(np.float64)
        folded_weight = (weight @ basis).astype(np.float16)
        folded_bias = (bias + weight @ mean).astype(np.float16)
        # Every eighth timestep is enough to measure the fit.
        exact = curve[::8] @ weight.T + bias
        approximate = table[::8] @ folded_weight.astype(np.float64).T + folded_bias
        worst = max(worst, np.linalg.norm(approximate - exact) / np.linalg.norm(exact))
        tensors[f"{PREFIX}{name}.weight"] = (
            "F16",
            folded_weight.shape,
            folded_weight.tobytes(),
        )
        tensors[f"{PREFIX}{name}.bias"] = (
            "F16",
            folded_bias.shape,
            folded_bias.tobytes(),
        )
    print(f"pruned AdaLN: modulations within {worst:.3e} of the full ones", flush=True)

    # The VSA gates, which the base does not have, in INT8 ConvRot.
    for index in range(converted_layers):
        name = f"blocks.{index}.attn.to_gate_compress"
        quantized, scales = quantize_convrot(
            fasth3.load(f"transformer_blocks.{index}.attn.to_gate_compress.weight")
        )
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
