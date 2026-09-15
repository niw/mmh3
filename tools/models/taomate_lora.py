# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Converts the TaoMate-H3 adapter into a ComfyUI LoRA with ranks in multiples of 64.

Development tool. It reads the adapter directory of TaoLiveAIGC/TaoMate-H3, whose
adapter_model.safetensors holds rank-128 FP32 lora_a and lora_b under the names of the pruned
DiT and whose adapter_config.json holds the rank and alpha, and writes one safetensors file
that mmh3 applies with --lora and ComfyUI loads as a LoRA. It computes on the CPU with NumPy.
Run it with uv, for example:

    uv run tools/models/taomate_lora.py \\
        --adapter /path/to/TaoMate-H3 \\
        --out models/loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors

Each layer's update, B·A times alpha / rank, is factored again by its SVD and cut to --rank
singular values, or with --energy to the smallest multiple of 64 up to --rank that keeps that
fraction of the update's squared Frobenius norm. mmh3 pads LoRA ranks to multiples of 64 and the
down projections of INT8 layers to 128 rows, so smaller ranks only make the file smaller. The
file holds, under the prefix diffusion_model.,
lora_A.weight, lora_B.weight and alpha in BF16 for the linear layers of the blocks and of the
text refiner, with the singular values split evenly between the two factors and alpha equal to
the layer's rank.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from checkpoint import SafeTensors, write_safetensors

PREFIX = "diffusion_model."
RANK_STEP = 64


def bf16_bytes(values):
    """Little-endian bytes of float32 values rounded to BF16, to nearest even."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).tobytes()


def update_svd(lora_b, lora_a, scale):
    """The SVD of scale · lora_b @ lora_a through QR factors of both, in float64."""
    left, left_r = np.linalg.qr(lora_b.astype(np.float64))
    right, right_r = np.linalg.qr(lora_a.astype(np.float64).T)
    core_left, singular, core_right = np.linalg.svd(scale * left_r @ right_r.T)
    return left @ core_left, singular, core_right @ right.T


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapter", required=True, help="TaoMate-H3 adapter directory")
    parser.add_argument("--out", required=True, help="LoRA file to write")
    parser.add_argument(
        "--rank", type=int, default=128, help="a multiple of 64, at most the adapter's"
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=None,
        help="the fraction of each update's energy to keep with the smallest rank",
    )
    arguments = parser.parse_args()

    directory = Path(arguments.adapter)
    config = json.loads((directory / "adapter_config.json").read_text())
    source_rank, alpha = int(config["rank"]), float(config["alpha"])
    if arguments.rank % RANK_STEP != 0 or not 0 < arguments.rank <= source_rank:
        sys.exit(f"--rank must be a multiple of {RANK_STEP} up to {source_rank}")
    if arguments.energy is not None and not 0.0 < arguments.energy <= 1.0:
        sys.exit("--energy must be in (0, 1]")

    adapter = SafeTensors(directory / "adapter_model.safetensors")
    names = [
        name[: -len(".lora_a")] for name in adapter.header if name.endswith(".lora_a")
    ]
    tensors = {}
    ranks = []
    energies = []
    lost = []
    for name in names:
        lora_a, lora_b = adapter.load(f"{name}.lora_a"), adapter.load(f"{name}.lora_b")
        if lora_a.shape[0] != source_rank or lora_b.shape[1] != source_rank:
            sys.exit(
                f"{name}: rank {lora_a.shape[0]}, not the configured {source_rank}"
            )
        left, singular, right = update_svd(lora_b, lora_a, alpha / source_rank)
        kept = np.cumsum(singular**2) / np.sum(singular**2)
        rank = arguments.rank
        if arguments.energy is not None:
            for candidate in range(RANK_STEP, arguments.rank + 1, RANK_STEP):
                # NOTE: the tolerance keeps a full-rank update at the smallest rank that holds
                # all of it in float64.
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
        lost.append(1.0 - float(kept[rank - 1]))
        print(
            f"{name:40} rank {rank:3} keeps {kept[rank - 1]:.4f} of the update",
            flush=True,
        )

    energies = np.array(energies)
    relative = np.sqrt(np.sum(energies * np.array(lost)) / np.sum(energies))
    metadata = {
        "source": "TaoLiveAIGC/TaoMate-H3",
        "weight_source": str(config.get("weight_source", "")),
        "optimizer_step": str(config.get("optimizer_step", "")),
        "rank": str(arguments.rank),
        "energy": "" if arguments.energy is None else str(arguments.energy),
        "note": "A ComfyUI LoRA modified from the TaoMate-H3 adapter: the updates are "
        "factored again by their SVD, truncated and rounded to BF16.",
    }
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(arguments.out, tensors, metadata)
    print(
        f"wrote {arguments.out}: {len(ranks)} layers, mean rank {np.mean(ranks):.1f}, "
        f"{sum(rank > RANK_STEP for rank in ranks)} above {RANK_STEP}, "
        f"relative error {relative:.3e} of all updates"
    )


if __name__ == "__main__":
    main()
