# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2"]
# ///
"""Converts a DMAD LoRA of MiniMax H3 into a ComfyUI LoRA.

Development tool. It reads one of the 4-step students in minimax_h3 of ZhengmingYu/DMAD, a rank-128
BF16 LoRA under the Diffusers names of the transformer whose alpha and rank are in its metadata,
and writes one safetensors file that mmh3 applies with --lora and ComfyUI loads as a LoRA. It
computes on the CPU with NumPy. Run it with uv, for example:

    uv run tools/models/dmad_lora.py \\
        --lora /path/to/DMAD/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors \\
        --out models/loras/minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors

diffusers_lora.py describes the conversion. --max-rank cuts each update to that rank, and --energy
to the smallest multiple of 64 that keeps that fraction of its squared Frobenius norm.
"""

import argparse

from checkpoint import SafeTensors
from diffusers_lora import check_ranks, convert


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lora", required=True, help="a DMAD student of minimax_h3")
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
    check_ranks(arguments.max_rank, arguments.energy)

    source = SafeTensors(arguments.lora)
    metadata = source.metadata
    convert(
        source,
        float(metadata["lora_alpha"]) / float(metadata["lora_rank"]),
        arguments.out,
        arguments.max_rank,
        arguments.energy,
        {
            "source": "DMAD LoRA of MiniMax H3",
            "checkpoint": metadata.get("checkpoint", ""),
            "max_rank": str(arguments.max_rank),
            "energy": "" if arguments.energy is None else str(arguments.energy),
            "note": "A ComfyUI LoRA modified from the DMAD LoRA: q, k and v are fused, the fc1 "
            "halves are swapped, and the updates are factored again by their SVD, truncated and "
            "rounded to BF16.",
        },
    )


if __name__ == "__main__":
    main()
