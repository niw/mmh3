"""Converts a LoRA of MiniMax H3 under the Diffusers names into a ComfyUI LoRA.

The q, k and v updates go into the fused qkv_proj, of rank 384, and the two halves of the
feed-forward input update are swapped, because Diffusers puts the up half first and the DiT's fc1
the gate half. Each update is factored again by its SVD and cut to a highest rank, or to the
smallest multiple of 64 that keeps a fraction of its squared Frobenius norm. The file holds
lora_A.weight, lora_B.weight and alpha in BF16 under the prefix diffusion_model., with alpha equal
to the rank.
"""

import re
import sys
from pathlib import Path

import numpy as np

from checkpoint import bf16_bytes, update_svd, write_safetensors

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
# NOTE: PEFT names the factors lora_A and lora_B, and some trainers lora.down and lora.up.
FACTORS = {
    "lora_A.weight": "A",
    "lora_B.weight": "B",
    "lora.down.weight": "A",
    "lora.up.weight": "B",
}
NAME = re.compile(
    r"(?:transformer\.)?(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)\."
    r"(lora_A\.weight|lora_B\.weight|lora\.down\.weight|lora\.up\.weight)"
)


def check_ranks(max_rank, energy):
    if energy is not None and not 0.0 < energy <= 1.0:
        sys.exit("--energy must be in (0, 1]")
    if max_rank % RANK_STEP != 0 or max_rank <= 0:
        sys.exit(f"--max-rank must be a positive multiple of {RANK_STEP}")


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


def block_factors(source):
    """{(group, index): {layer: {"A": name, "B": name}}} of the LoRA tensors of `source`."""
    blocks = {}
    for name in source.header:
        match = NAME.fullmatch(name)
        if match is None:
            sys.exit(f"{name}: not a LoRA tensor of the transformer blocks")
        layers = blocks.setdefault((match[1], int(match[2])), {})
        layers.setdefault(match[3], {})[FACTORS[match[4]]] = name
    return blocks


def block_updates(source, block, layers):
    """(name, lora_b, lora_a) of each layer of one block, under the DiT's names."""
    updates = []
    if all(layer in layers for layer in QKV):
        lora_bs = [source.load(layers[layer]["B"]) for layer in QKV]
        lora_as = [source.load(layers[layer]["A"]) for layer in QKV]
        updates.append(
            (f"{block}.attn.qkv_proj", block_diagonal(lora_bs), np.concatenate(lora_as))
        )
    elif any(layer in layers for layer in QKV):
        sys.exit(f"{block}: q, k and v are not all present")
    for layer, factors in layers.items():
        if layer in QKV:
            continue
        if layer not in LAYERS:
            sys.exit(f"{block}.{layer}: no matching layer of the DiT")
        lora_b = source.load(factors["B"])
        lora_a = source.load(factors["A"])
        if layer == "ff.net.0.proj":
            half = lora_b.shape[0] // 2
            lora_b = np.concatenate([lora_b[half:], lora_b[:half]])
        updates.append((f"{block}.{LAYERS[layer]}", lora_b, lora_a))
    return updates


def convert(source, scale, out, max_rank, energy, metadata):
    """Writes the LoRA of `source`, whose updates are scale · B·A, as a ComfyUI LoRA to `out`."""
    tensors = {}
    ranks = []
    energies = []
    lost = []
    for (group, index), layers in sorted(block_factors(source).items()):
        block = f"{BLOCKS[group]}.{index}"
        for name, lora_b, lora_a in block_updates(source, block, layers):
            left, singular, right = update_svd(lora_b, lora_a, scale)
            kept = np.cumsum(singular**2) / np.sum(singular**2)
            highest_rank = min(len(singular), max_rank)
            rank = highest_rank
            if energy is not None:
                for candidate in range(RANK_STEP, highest_rank + 1, RANK_STEP):
                    # NOTE: the tolerance keeps a full-rank update at the smallest rank that
                    # holds all of it in float64.
                    if kept[candidate - 1] >= energy - 1e-12:
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
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(out, tensors, metadata)
    print(
        f"wrote {out}: {len(ranks)} layers, mean rank {np.mean(ranks):.1f}, "
        f"relative error {relative:.3e} of all updates"
    )
