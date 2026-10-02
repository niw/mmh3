# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Converts the PDMD LoRA of MiniMax H3 into a ComfyUI LoRA.

Development tool. It reads lora_model_0.safetensors of pdmd2026/pdmd_2NFE_lora, a rank-128 BF16 LoRA
under the Diffusers names of the transformer whose alpha and rank are in its metadata, and writes
one safetensors file that mmh3 applies with --lora and ComfyUI loads as a LoRA. It computes on the
CPU with NumPy. Run it with uv, for example:

    uv run tools/models/pdmd_lora.py \\
        --lora /path/to/pdmd_2NFE_lora/lora_model_0.safetensors \\
        --out models/loras/minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors

The q, k and v updates go into the fused qkv_proj, of rank 384, and the two halves of the
feed-forward input update are swapped, because Diffusers puts the up half first and the DiT's fc1
the gate half. Each update is factored again by its SVD and cut to --max-rank, or with --energy to
the smallest multiple of 64 that keeps that fraction of its squared Frobenius norm. The file holds
lora_A.weight, lora_B.weight and alpha in BF16 under the prefix diffusion_model., with alpha equal
to the rank.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from checkpoint import SafeTensors, bf16_bytes, update_svd, write_safetensors

SOURCE_PREFIX = "transformer."
PREFIX = "diffusion_model."
RANK_STEP = 64
BLOCKS = {
    "transformer_blocks": "blocks",
    "token_refiner.refiner_blocks": "token_refiner.blocks",
}
LAYERS = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
}
QKV = ("attn.to_q", "attn.to_k", "attn.to_v")


def block_diagonal(matrices):
    rows = sum(matrix.shape[0] for matrix in matrices)
    columns = sum(matrix.shape[1] for matrix in matrices)
    result = np.zeros((rows, columns), dtype=np.float32)
    row = column = 0
    for matrix in matrices:
        result[row : row + matrix.shape[0], column : column + matrix.shape[1]] = matrix
        row += matrix.shape[0]
        column += matrix.shape[1]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--lora", required=True, help="the PDMD lora_model_0.safetensors"
    )
    parser.add_argument("--out", required=True, help="LoRA file to write")
    parser.add_argument(
        "--max-rank",
        type=int,
        default=128,
        help="the highest rank of a layer, a multiple of 64",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=None,
        help="the fraction of each update's energy to keep with the smallest rank",
    )
    arguments = parser.parse_args()
    if arguments.energy is not None and not 0.0 < arguments.energy <= 1.0:
        sys.exit("--energy must be in (0, 1]")
    if arguments.max_rank % RANK_STEP != 0 or arguments.max_rank <= 0:
        sys.exit(f"--max-rank must be a positive multiple of {RANK_STEP}")

    source = SafeTensors(arguments.lora)
    metadata = source.metadata
    scale = float(metadata["lora_alpha"]) / float(metadata["lora_rank"])

    pattern = re.compile(
        r"transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\."
        r"(.+)\.lora_A\.weight"
    )
    blocks = {}
    for name in source.header:
        match = pattern.fullmatch(name)
        if match is None:
            if not name.endswith(".lora_B.weight"):
                sys.exit(f"{name}: not a LoRA tensor of the transformer blocks")
            continue
        blocks.setdefault((match[1], int(match[2])), []).append(match[3])

    tensors = {}
    ranks = []
    energies = []
    lost = []
    for (group, index), layers in sorted(blocks.items()):
        source_block = f"{SOURCE_PREFIX}{group}.{index}"
        block = f"{BLOCKS[group]}.{index}"
        updates = []
        if all(layer in layers for layer in QKV):
            lora_bs = [
                source.load(f"{source_block}.{layer}.lora_B.weight") for layer in QKV
            ]
            lora_as = [
                source.load(f"{source_block}.{layer}.lora_A.weight") for layer in QKV
            ]
            updates.append(
                (
                    f"{block}.attn.qkv_proj",
                    block_diagonal(lora_bs),
                    np.concatenate(lora_as),
                )
            )
        elif any(layer in layers for layer in QKV):
            sys.exit(f"{source_block}: q, k and v are not all present")
        for layer in layers:
            if layer in QKV:
                continue
            if layer not in LAYERS:
                sys.exit(f"{source_block}.{layer}: no matching layer of the DiT")
            lora_b = source.load(f"{source_block}.{layer}.lora_B.weight")
            lora_a = source.load(f"{source_block}.{layer}.lora_A.weight")
            if layer == "ff.net.0.proj":
                half = lora_b.shape[0] // 2
                lora_b = np.concatenate([lora_b[half:], lora_b[:half]])
            updates.append((f"{block}.{LAYERS[layer]}", lora_b, lora_a))

        for name, lora_b, lora_a in updates:
            left, singular, right = update_svd(lora_b, lora_a, scale)
            kept = np.cumsum(singular**2) / np.sum(singular**2)
            highest_rank = min(len(singular), arguments.max_rank)
            rank = highest_rank
            if arguments.energy is not None:
                for candidate in range(RANK_STEP, highest_rank + 1, RANK_STEP):
                    # NOTE: the tolerance keeps a full-rank update at the smallest rank that
                    # holds all of it in float64.
                    if kept[candidate - 1] >= arguments.energy - 1e-12:
                        rank = candidate
                        break
            root = np.sqrt(singular[:rank])
            tensors[f"{PREFIX}{name}.lora_B.weight"] = (
                "BF16",
                (lora_b.shape[0], rank),
                bf16_bytes(left[:, :rank] * root),
            )
            tensors[f"{PREFIX}{name}.lora_A.weight"] = (
                "BF16",
                (rank, lora_a.shape[1]),
                bf16_bytes(root[:, None] * right[:rank]),
            )
            tensors[f"{PREFIX}{name}.alpha"] = ("BF16", (), bf16_bytes([float(rank)]))
            ranks.append(rank)
            energies.append(float(np.sum(singular**2)))
            lost.append(max(0.0, 1.0 - float(kept[rank - 1])))
            print(
                f"{name:40} rank {rank:3} keeps {kept[rank - 1]:.4f} of the update",
                flush=True,
            )

    energies = np.array(energies)
    relative = np.sqrt(np.sum(energies * np.array(lost)) / np.sum(energies))
    output_metadata = {
        "source": "PDMD LoRA of MiniMax H3",
        "tag": metadata.get("tag", ""),
        "max_rank": str(arguments.max_rank),
        "energy": "" if arguments.energy is None else str(arguments.energy),
        "note": "A ComfyUI LoRA modified from the PDMD LoRA: q, k and v are fused, the fc1 "
        "halves are swapped, and the updates are factored again by their SVD, truncated and "
        "rounded to BF16.",
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(arguments.out, tensors, output_metadata)
    print(
        f"wrote {arguments.out}: {len(ranks)} layers, mean rank {np.mean(ranks):.1f}, "
        f"relative error {relative:.3e} of all updates"
    )


if __name__ == "__main__":
    main()
