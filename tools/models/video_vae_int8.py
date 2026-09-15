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
"""Builds an INT8 ConvRot video VAE from the FP16 one.

Development tool. It reads the FP16 video VAE in ComfyUI's layout,
vae/minimax_h3_video_vae_fp16.safetensors of Comfy-Org/MiniMax-H3, and calibration latents,
and writes the VAE with the linear layers of its decoder blocks in INT8 ConvRot, the format
that mmh3 and ComfyUI load from vae/minimax_h3_video_vae_int8_convrot.safetensors. It computes
on a CUDA GPU. Run it with uv, which installs numpy and PyTorch for CUDA 13.0, for example:

    uv run tools/models/video_vae_int8.py \\
        --vae /path/to/minimax_h3_video_vae_fp16.safetensors \\
        --latents /path/to/latents/*.safetensors \\
        --out models/vae/minimax_h3_video_vae_int8_convrot.safetensors

mmh3 runs these layers as W8A8: it rotates each input row in groups of 256 columns with a
Hadamard matrix, quantizes the row to INT8 with scale max |x| / 127, and multiplies it with the
rotated INT8 weights, which have one scale per output row. The tool runs the decoder in FP32 on
tiles of the calibration latents, block by block and with the blocks before already quantized,
and quantizes each layer in two steps:

- Smoothing: it divides the input channels by s = sqrt(rms(x) / |w|), where rms(x) is the
  channel's root mean square over the calibration tokens and |w| the norm of the weight column,
  and multiplies the weight columns by s. The division folds into the preceding operation, the
  RMSNorm weight for to_qkv and ff.w1, the value rows of to_qkv for to_out, and the up rows of
  ff.w1 for ff.w2, so the decoder computes the same function. The per-row activation scale then
  wastes less of the INT8 range, and this s minimizes the activation rounding error that reaches
  the output.
- GPTQ: it rounds the rotated weight columns one at a time and spreads each rounding error over
  the columns not yet rounded through the inverse Hessian of the layer's rotated calibration
  inputs.
"""

import argparse
import glob
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint import CONVROT_GROUP, SafeTensors, hadamard, write_safetensors

DEVICE = torch.device("cuda")
SPATIAL_RATIO = 16
TILE_SIZE = 256
TILE_OVERLAP_MIN = 64
CHUNK_TOKENS = 5
TOKEN_DROP = 3
CHUNK_LATENT_FRAMES = 7
HEAD_DIM = 64
ROPE_FREQUENCIES = 8
ROPE_BASE = 100.0
NORM_EPSILON = 1e-5
QUANTIZATION = json.dumps(
    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP}
).encode()
HADAMARD = torch.from_numpy(hadamard()).float().to(DEVICE)


def rotate(values):
    """Rotates the last dimension in groups of CONVROT_GROUP columns."""
    shape = values.shape
    grouped = values.reshape(*shape[:-1], shape[-1] // CONVROT_GROUP, CONVROT_GROUP)
    return (grouped @ HADAMARD.to(values.dtype)).reshape(shape)


def quantize_activations(rotated):
    """The rotated rows after INT8 rounding with their per-row absmax scales."""
    scales = torch.clamp(rotated.abs().amax(dim=-1, keepdim=True) / 127.0, min=1e-30)
    return torch.clamp(torch.round(rotated / scales), -128, 127) * scales


def gptq(rotated_weight, hessian, scales, block=128, damping=0.01):
    """INT8 values of the rotated weight [n, k] by GPTQ against the Hessian [k, k] of the
    rotated inputs, with fixed per-row scales [n, 1], in descending Hessian diagonal order."""
    weight = rotated_weight.clone()
    hessian = hessian.clone()
    columns = weight.shape[1]
    diagonal = torch.diagonal(hessian).clone()
    dead = diagonal == 0
    hessian[dead, dead] = 1
    weight[:, dead] = 0
    order = torch.argsort(diagonal, descending=True)
    weight = weight[:, order]
    hessian = hessian[order][:, order]
    hessian.diagonal().add_(damping * hessian.diagonal().mean())
    inverse = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
    )
    row_scales = scales[:, 0]
    quantized = torch.empty(weight.shape, dtype=torch.int8, device=weight.device)
    for start in range(0, columns, block):
        end = min(start + block, columns)
        current = weight[:, start:end].clone()
        errors = torch.empty_like(current)
        local = inverse[start:end, start:end]
        for index in range(end - start):
            values = torch.clamp(torch.round(current[:, index] / row_scales), -127, 127)
            quantized[:, start + index] = values.to(torch.int8)
            error = (current[:, index] - values * row_scales) / local[index, index]
            current[:, index:] -= error[:, None] * local[index, index:]
            errors[:, index] = error
        weight[:, end:] -= errors @ inverse[start:end, end:]
    result = torch.empty_like(quantized)
    result[:, order] = quantized
    return result


def split_tiles(length):
    """Tile starts along one axis in pixels, like mmh3_core::vae::split_tiles."""
    if TILE_SIZE >= length:
        return [0], length
    count = -(-length // TILE_SIZE)
    while True:
        overlaps = [TILE_OVERLAP_MIN] * (count - 1)
        covered = TILE_SIZE * count - sum(overlaps)
        if covered >= length:
            break
        count += 1
    for unit in range((covered - length) // SPATIAL_RATIO):
        overlaps[unit % (count - 1)] += SPATIAL_RATIO
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + TILE_SIZE - overlap)
    return starts, TILE_SIZE


def rope_angles(frames, height, width, suffix):
    """Rotary angles [tokens, 3 × ROPE_FREQUENCIES] of one tile, like
    mmh3_core::vae::rope_angles."""
    inverse = ROPE_BASE ** (-np.arange(ROPE_FREQUENCIES) / ROPE_FREQUENCIES)

    def coordinates(size):
        return ((np.arange(size) + 0.5) / size) * 2 - 1

    grid = np.meshgrid(
        coordinates(frames), coordinates(height), coordinates(width), indexing="ij"
    )
    positions = np.stack([axis.ravel() for axis in grid], axis=1)
    angles = (2 * np.pi * positions[:, :, None] * inverse).reshape(len(positions), -1)
    angles = np.concatenate([angles, np.zeros((suffix, angles.shape[1]))])
    return torch.from_numpy(angles).float().to(DEVICE)


def load_latent(path):
    """The video latent [channels, frames, height, width] of a safetensors file: the `video`
    tensor of mmh3-tools latent, or the last step's latent of a golden file of
    tools/golden/h3_reference.py."""
    file = SafeTensors(path)
    names = [name for name in file.header if name.endswith("latent.video")]
    if "video" in file:
        name = "video"
    elif names:
        name = max(names, key=lambda name: int(name.removeprefix("step").split(".")[0]))
    else:
        sys.exit(f"{path}: no video latent")
    latent = file.load(name)
    return latent[0] if latent.ndim == 5 else latent


class Decoder:
    """The decoder blocks of the video VAE in FP32 on the GPU, whose tensors the tool edits
    in place."""

    def __init__(self, vae):
        self.tensors = {
            name: torch.from_numpy(vae.load(name)).to(DEVICE)
            for name in vae.header
            if name.startswith("decoder.") or name.startswith("post_quant_conv.")
        }
        self.latents_mean = vae.load("latents_mean").reshape(-1)
        self.latents_std = vae.load("latents_std").reshape(-1)
        self.layers = sum(1 for name in self.tensors if name.endswith(".scale1"))
        self.registers = self.tensors["decoder.register_tokens"][0]

    def tiles(self, latent):
        """Latent rows [tokens, channels] and grid of every tile of every temporal chunk, the
        tiles mmh3 decodes."""
        channels, frames, height, width = latent.shape
        latent = (
            latent * self.latents_std[:, None, None, None]
            + self.latents_mean[:, None, None, None]
        )
        rows, tile_height = split_tiles(height * SPATIAL_RATIO)
        columns, tile_width = split_tiles(width * SPATIAL_RATIO)
        tile_height //= SPATIAL_RATIO
        tile_width //= SPATIAL_RATIO
        # The chunk count of mmh3_core::vae::TemporalPlan.
        chunks = max(1, -(-(frames + TOKEN_DROP) // CHUNK_TOKENS) - 1)
        tiles = []
        for chunk in range(chunks):
            frame_indices = [
                min(chunk * CHUNK_TOKENS + frame, frames - 1)
                for frame in range(CHUNK_LATENT_FRAMES)
            ]
            for top in rows:
                for left in columns:
                    y, x = top // SPATIAL_RATIO, left // SPATIAL_RATIO
                    part = latent[
                        :, frame_indices, y : y + tile_height, x : x + tile_width
                    ]
                    tiles.append(
                        (
                            part.reshape(channels, -1).T,
                            (CHUNK_LATENT_FRAMES, tile_height, tile_width),
                        )
                    )
        return tiles

    def embed(self, rows):
        """The residual streams [tiles, tokens, dim] of tiles of latent rows [tiles, patches,
        channels]: embedded latent rows, then the register tokens and a zero token."""
        channels = rows.shape[-1]
        post = self.tensors["post_quant_conv.weight"].reshape(channels, channels)
        rows = rows @ post.T + self.tensors["post_quant_conv.bias"]
        embedded = (
            rows @ self.tensors["decoder.x_embedder.weight"].T
            + self.tensors["decoder.x_embedder.bias"]
        )
        tiles, _, dim = embedded.shape
        suffix = torch.cat([self.registers, torch.zeros(1, dim, device=DEVICE)])
        return torch.cat([embedded, suffix.expand(tiles, -1, -1)], dim=1)

    def norm(self, residual, weight):
        mean_square = (residual * residual).mean(dim=-1, keepdim=True)
        return residual * torch.rsqrt(mean_square + NORM_EPSILON) * self.tensors[weight]

    def attention(self, qkv, angles):
        """Attention within each tile of qkv [tiles, tokens, heads × 3 × HEAD_DIM]: RMSNorm of
        q and k without weight, split-half rotary embedding, softmax over the tile."""
        tiles, tokens, _ = qkv.shape
        qkv = qkv.reshape(tiles, tokens, -1, 3, HEAD_DIM).permute(3, 0, 2, 1, 4)
        pairs = angles.shape[1]
        cosine, sine = torch.cos(angles), torch.sin(angles)

        def normalize_rotate(values):
            values = values * torch.rsqrt(
                (values * values).mean(dim=-1, keepdim=True) + NORM_EPSILON
            )
            first, second = values[..., :pairs], values[..., pairs : 2 * pairs]
            return torch.cat(
                [
                    first * cosine - second * sine,
                    first * sine + second * cosine,
                    values[..., 2 * pairs :],
                ],
                dim=-1,
            )

        attended = F.scaled_dot_product_attention(
            normalize_rotate(qkv[0]), normalize_rotate(qkv[1]), qkv[2]
        )
        return attended.transpose(1, 2).reshape(tiles, tokens, -1)


class Layer:
    """A quantized linear layer: INT8 rotated weight, per-row scales and FP32 bias."""

    def __init__(self, quantized, scales, bias):
        self.quantized = quantized
        self.scales = scales.float()
        self.bias = bias.clone()
        self.dequantized = quantized.float() * self.scales

    def scale_rows(self, rows, factors):
        """Multiplies the output rows `rows` by `factors`, which leaves the INT8 values as
        they are."""
        self.scales[rows, 0] *= factors
        self.bias[rows] *= factors
        self.dequantized[rows] *= factors[:, None]

    def __call__(self, inputs):
        return quantize_activations(rotate(inputs)) @ self.dequantized.T + self.bias


def smoothing(inputs, weight):
    """Per-channel divisors s = sqrt(rms(x) / |w|) with a geometric mean of 1."""
    rms = torch.sqrt(torch.mean(inputs.reshape(-1, inputs.shape[-1]) ** 2, dim=0))
    column_norms = torch.linalg.norm(weight, dim=0)
    divisors = torch.sqrt(
        torch.clamp(rms, min=1e-12) / torch.clamp(column_norms, min=1e-12)
    )
    return divisors / torch.exp(torch.mean(torch.log(divisors)))


def quantize(weight, bias, inputs, method):
    """The layer of weight [n, k] and bias for inputs [..., k]."""
    rotated = rotate(weight)
    scales = torch.clamp(rotated.abs().amax(dim=1, keepdim=True) / 127.0, min=1e-30)
    if method == "gptq":
        rotated_inputs = rotate(inputs.reshape(-1, inputs.shape[-1]))
        quantized = gptq(rotated, rotated_inputs.T @ rotated_inputs, scales)
    else:
        quantized = torch.clamp(torch.round(rotated / scales), -127, 127).to(torch.int8)
    return Layer(quantized, scales, bias)


def quantize_block(decoder, layer, states, angles, method, smooth):
    """Quantizes block `layer` on the residual states [tiles, tokens, dim] of the calibration
    tiles, editing the decoder's tensors for the smoothing, and returns the states after the
    quantized block and its layers."""
    prefix = f"decoder.transformer_blocks.{layer}"
    tensors = decoder.tensors
    name = f"{prefix}.attn.to_qkv"
    normalized = decoder.norm(states, f"{prefix}.norm1.weight")
    if smooth:
        divisors = smoothing(normalized, tensors[f"{name}.weight"])
        tensors[f"{prefix}.norm1.weight"] /= divisors
        tensors[f"{name}.weight"] *= divisors
        normalized /= divisors
    to_qkv = quantize(
        tensors[f"{name}.weight"], tensors[f"{name}.bias"], normalized, method
    )

    attended = decoder.attention(to_qkv(normalized), angles)
    name = f"{prefix}.attn.to_out"
    if smooth:
        divisors = smoothing(attended, tensors[f"{name}.weight"])
        heads = divisors.shape[0] // HEAD_DIM
        value_rows = (
            torch.arange(heads, device=DEVICE)[:, None] * 3 * HEAD_DIM
            + 2 * HEAD_DIM
            + torch.arange(HEAD_DIM, device=DEVICE)
        ).reshape(-1)
        to_qkv.scale_rows(value_rows, 1 / divisors)
        tensors[f"{name}.weight"] *= divisors
        attended /= divisors
    to_out = quantize(
        tensors[f"{name}.weight"], tensors[f"{name}.bias"], attended, method
    )
    states = states + tensors[f"{prefix}.scale1"] * to_out(attended)

    name = f"{prefix}.ff.w1"
    normalized = decoder.norm(states, f"{prefix}.norm2.weight")
    if smooth:
        divisors = smoothing(normalized, tensors[f"{name}.weight"])
        tensors[f"{prefix}.norm2.weight"] /= divisors
        tensors[f"{name}.weight"] *= divisors
        normalized /= divisors
    w1 = quantize(
        tensors[f"{name}.weight"], tensors[f"{name}.bias"], normalized, method
    )

    gate, up = w1(normalized).chunk(2, dim=-1)
    activated = F.silu(gate) * up
    name = f"{prefix}.ff.w2"
    if smooth:
        divisors = smoothing(activated, tensors[f"{name}.weight"])
        up_rows = torch.arange(divisors.shape[0], device=DEVICE) + divisors.shape[0]
        w1.scale_rows(up_rows, 1 / divisors)
        tensors[f"{name}.weight"] *= divisors
        activated /= divisors
    w2 = quantize(tensors[f"{name}.weight"], tensors[f"{name}.bias"], activated, method)
    states = states + tensors[f"{prefix}.scale2"] * w2(activated)
    return states, {
        f"{prefix}.attn.to_qkv": to_qkv,
        f"{prefix}.attn.to_out": to_out,
        f"{prefix}.ff.w1": w1,
        f"{prefix}.ff.w2": w2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--vae", required=True, help="FP16 video VAE in ComfyUI's layout"
    )
    parser.add_argument(
        "--latents", nargs="+", required=True, help="safetensors files of video latents"
    )
    parser.add_argument("--out", required=True, help="INT8 ConvRot video VAE to write")
    parser.add_argument(
        "--tiles",
        type=int,
        default=96,
        help="calibration tiles, drawn from all latents",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["gptq", "rtn"], default="gptq")
    parser.add_argument("--no-smoothing", action="store_true")
    arguments = parser.parse_args()

    vae = SafeTensors(arguments.vae)
    decoder = Decoder(vae)
    paths = [
        path for pattern in arguments.latents for path in sorted(glob.glob(pattern))
    ]
    tiles = [tile for path in paths for tile in decoder.tiles(load_latent(path))]
    generator = np.random.default_rng(arguments.seed)
    chosen = generator.choice(
        len(tiles), min(arguments.tiles, len(tiles)), replace=False
    )
    tiles = [tiles[index] for index in sorted(chosen)]
    grids = {grid for _, grid in tiles}
    if len(grids) != 1:
        sys.exit(f"calibration tiles of different sizes {grids}")
    angles = rope_angles(*grids.pop(), decoder.registers.shape[0] + 1)
    print(f"{len(tiles)} calibration tiles from {len(paths)} latents", flush=True)

    rows = torch.from_numpy(np.stack([rows for rows, _ in tiles])).float().to(DEVICE)
    states = decoder.embed(rows)
    layers = {}
    started = time.time()
    for layer in range(decoder.layers):
        states, quantized = quantize_block(
            decoder, layer, states, angles, arguments.method, not arguments.no_smoothing
        )
        for quantized_layer in quantized.values():
            quantized_layer.dequantized = None
        layers.update(quantized)
    print(
        f"quantized {decoder.layers} blocks in {time.time() - started:.1f} s",
        flush=True,
    )

    tensors = {}
    for name in vae.header:
        layer = name.rsplit(".", 1)[0]
        if layer in layers:
            quantized = layers[layer]
            if name.endswith(".weight"):
                values = quantized.quantized.cpu().numpy()
                scales = quantized.scales.cpu().numpy()
                tensors[name] = ("I8", values.shape, values.tobytes())
                tensors[f"{layer}.weight_scale"] = (
                    "F32",
                    scales.shape,
                    scales.tobytes(),
                )
                tensors[f"{layer}.comfy_quant"] = (
                    "U8",
                    (len(QUANTIZATION),),
                    QUANTIZATION,
                )
            else:
                bias = quantized.bias.cpu().numpy()
                tensors[name] = ("F32", bias.shape, bias.tobytes())
            continue
        if name in decoder.tensors:
            values = decoder.tensors[name].half().cpu().numpy()
        else:
            values = vae.load(name).astype(np.float16)
        tensors[name] = ("F16", values.shape, values.tobytes())
    with open(arguments.vae, "rb") as file:
        length = struct.unpack("<Q", file.read(8))[0]
        metadata = json.loads(file.read(length)).get("__metadata__", {})
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(arguments.out, tensors, metadata)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
