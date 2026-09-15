"""Writes golden text-to-video data from the released weights through ComfyUI.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3 and has the released checkpoints in its models directory, on a GPU,
for example:

    python3 tools/golden/h3_reference.py --comfyui /path/to/ComfyUI \\
        --out /path/to/golden/small

The defaults describe the small case (448x256, 22 frames). The 768p case is --width 1344
--height 768 --frames 124 --block-row-stride 16 --no-decode. --compute-dtype float32
runs the DiT with an FP32 residual stream and FP32 activations instead of ComfyUI's
default BF16, as a tighter reference. --lora applies a LoRA from models/loras to the
DiT, for example the 768p Turbo LoRA with --steps 4 --shift-video 6. ComfyUI merges a
LoRA into quantized weights and requantizes them with stochastic rounding, and
--lora-rounding nearest rounds to nearest instead. --sparse-attention sol patches in
ComfyUI's Sol-Attn from the step at --sparse-start of the run. --dit picks another DiT
from the models directory, and --sparse-attention vsa replaces the attention of its blocks
with an FP32 implementation of FastVideo's VSA-H3 on every step, for FastH3 checkpoints
such as diffusion_models/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot
.safetensors.

It writes up to three files into the output directory:

- text.safetensors: prompt token ids, the text encoder hidden states and the per-token
  modality tags. An existing file is reused, so copying it from another case skips the
  text encoder.
- dit.safetensors: initial noise, the latents after every step of a plain Euler sampler
  on the unshifted grid linspace(1, 0, steps + 1), and the prediction and selected block
  outputs of one step. The stored velocity is the data-ward prediction u with x0 = x +
  sigma · u, which is the negated flow velocity ComfyUI's model returns. Step 0 runs at
  sigma 1 where the video and audio timesteps coincide, so step 1 is captured by
  default.
- keyframes.safetensors, with --first-frame and --last-frame: the keyframe pictures on the
  canvas in [0, 1], the way ComfyUI's MiniMaxH3ImageToVideo resizes them (the first one
  stretched, the last one cover-cropped, both with Lanczos), and their normalized latents
  from the video VAE's encoder. The pictures also go to the text encoder, and
  dit.safetensors then holds the keyframe latents with the condition noise ComfyUI mixes
  in, as the DiT sees them.
- decode.safetensors: video pixels in [0, 1] and the stereo waveform decoded from the
  final latents, before ComfyUI's audio loudness normalization. The waveform comes from
  ComfyUI's default audio VAE setup as "audio" and from strict FP32, with PyTorch's TF32
  convolutions turned off, as "audio_float32". --decode-only rewrites this file from an
  existing dit.safetensors without sampling again, and --video-vae picks another video
  VAE from the models directory, such as
  vae/minimax_h3_video_vae_int8_convrot.safetensors.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import save_file

DIT = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER = "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "vae/minimax_h3_audio_vae_fp32.safetensors"
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40


def temporal_shape(frames):
    """Returns the frame counts of a generation.

    The frame count snapped to the 17k + 5 grid, the latent frame count and the audio
    latent frame count.
    """
    while frames % 17 != 5:
        frames += 1
    latent_frames = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return frames, latent_frames, round(frames / FPS * AUDIO_LATENTS_PER_SECOND)


def time_shift(base, shift):
    return shift * base / (1.0 + (shift - 1.0) * base)


VSA_TILE = 64
VSA_CUBE = (4, 4, 4)


def vsa_plan(layout, device):
    """Tiles of FastVideo's VSA-H3 for a ComfyUI packed layout.

    Returns the padded tile table [tiles, 64] of sequence rows (-1 marks padding), the
    tile lengths and the number of text and audio tiles, which come first. Every
    non-video segment is cut into tiles of up to 64 consecutive rows, and the video
    patches form 4 x 4 x 4 cubes of (latent frame, patch row, patch column).
    """
    _, latent_t, latent_h, latent_w, _ = layout.signature
    tiles = []
    for start, end, kind in layout.segments:
        if kind != "video":
            tiles += [
                list(range(row, min(row + VSA_TILE, end)))
                for row in range(start, end, VSA_TILE)
            ]
    prefix = len(tiles)
    start, end = next((a, b) for a, b, kind in layout.segments if kind == "video")
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    rows = torch.arange(start, end).view(*grid)
    for t in range(0, grid[0], VSA_CUBE[0]):
        for h in range(0, grid[1], VSA_CUBE[1]):
            for w in range(0, grid[2], VSA_CUBE[2]):
                cube = rows[
                    t : t + VSA_CUBE[0], h : h + VSA_CUBE[1], w : w + VSA_CUBE[2]
                ]
                tiles.append(cube.flatten().tolist())
    table = torch.full((len(tiles), VSA_TILE), -1, dtype=torch.int64)
    for index, tile in enumerate(tiles):
        table[index, : len(tile)] = torch.tensor(tile)
    lengths = torch.tensor([len(tile) for tile in tiles])
    return table.to(device), lengths.to(device), prefix


def vsa(query, key, value, gate, plan, sparsity, chunk_tiles=8):
    """VSA-H3 in FP32 over [tokens, heads, dim] tensors in sequence order.

    Each query tile attends to the key tiles it selects, gathered per head. The matrix
    products run in TF32, which holds the BF16 query, key and value exactly.
    """
    table, lengths, prefix = plan
    tiles = table.shape[0]
    tokens, heads, dim = query.shape
    live = (table >= 0).flatten()
    rows = table.flatten()[live]

    def tiled(x):
        padded = x.new_zeros((tiles * VSA_TILE, heads, dim), dtype=torch.float32)
        padded[live] = x[rows].float()
        return padded

    q, k, v = tiled(query), tiled(key), tiled(value)
    count = lengths.float().view(tiles, 1, 1)
    pooled = [x.view(tiles, VSA_TILE, heads, dim).sum(1) / count for x in (q, k, v)]
    scale = 1.0 / math.sqrt(dim)
    scores = torch.einsum("qhd,khd->hqk", pooled[0], pooled[1]) * scale
    video = tiles - prefix
    kept = max(1, min(math.ceil((1.0 - sparsity) * video), video))

    # [heads, tiles, 64, dim] views and the live key slots of every tile.
    q4, k4, v4 = (
        x.view(tiles, VSA_TILE, heads, dim).permute(2, 0, 1, 3) for x in (q, k, v)
    )
    live4 = table >= 0
    output4 = torch.empty_like(q4)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        # Text and audio query tiles, and every tile without sparsity, attend to all keys.
        dense = prefix if kept < video else tiles
        dense_chunk = max(1, chunk_tiles // 4)
        for first in range(0, dense, dense_chunk):
            last = min(first + dense_chunk, dense)
            logits = torch.einsum(
                "hqd,hkd->hqk", q4[:, first:last].flatten(1, 2), k4.flatten(1, 2)
            )
            logits = logits * scale
            logits.masked_fill_(~live4.view(1, 1, -1), float("-inf"))
            output4[:, first:last] = torch.einsum(
                "hqk,hkd->hqd", torch.softmax(logits, -1), v4.flatten(1, 2)
            ).view(heads, last - first, VSA_TILE, dim)
        if kept < video:
            top = scores[:, prefix:, prefix:].topk(kept, dim=-1).indices + prefix
            selected = torch.cat(
                [
                    torch.arange(prefix, device=top.device).expand(
                        heads, video, prefix
                    ),
                    top,
                ],
                -1,
            )
            head_index = torch.arange(heads, device=top.device).view(heads, 1, 1)
            for first in range(0, video, chunk_tiles):
                last = min(first + chunk_tiles, video)
                chosen = selected[:, first:last]
                keys = k4[head_index, chosen].flatten(2, 3)
                values = v4[head_index, chosen].flatten(2, 3)
                logits = torch.einsum(
                    "hcqd,hckd->hcqk", q4[:, prefix + first : prefix + last], keys
                )
                logits = logits * scale
                logits.masked_fill_(
                    ~live4[chosen].flatten(2, 3).unsqueeze(2), float("-inf")
                )
                output4[:, prefix + first : prefix + last] = torch.einsum(
                    "hcqk,hckd->hcqd", torch.softmax(logits, -1), values
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    output = output4.permute(1, 2, 0, 3).reshape(-1, heads, dim)
    if gate is not None:
        coarse = torch.einsum("hqk,khd->qhd", torch.softmax(scores, -1), pooled[2])
        output = output.view(tiles, VSA_TILE, heads, dim) + tiled(gate).view(
            tiles, VSA_TILE, heads, dim
        ) * coarse.unsqueeze(1)
        output = output.view(-1, heads, dim)
    result = torch.empty((tokens, heads, dim), dtype=torch.float32, device=query.device)
    result[rows] = output[live]
    return result


def vsa_block_patch(block, sparsity, plans):
    import comfy.model_management
    import comfy.quant_ops

    attn = block.attn

    def attention(x, rope_freqs=None, transformer_options=None):
        layout = transformer_options["minimax_h3_layout"]
        key = (layout.signature, tuple(layout.segments))
        if key not in plans:
            plans[key] = vsa_plan(layout, x.device)
        tokens, heads, dim = x.shape[0], attn.heads, attn.head_dim
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        q = q.view(1, tokens, heads, dim)
        k = k.view(1, tokens, heads, dim)
        query_weight = comfy.model_management.cast_to(
            attn.q_norm.weight, device=x.device
        )
        key_weight = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(
            q,
            k,
            rope_freqs,
            query_weight,
            key_weight,
            epsilon=attn.q_norm.eps,
            rot_dim=rope_freqs.shape[-3] * 2,
        )
        gate = None
        if attn.to_gate_compress is not None:
            gate = attn.to_gate_compress(x).view(tokens, heads, dim)
        out = vsa(q[0], k[0], v.reshape(tokens, heads, dim), gate, plans[key], sparsity)
        return attn.out_proj(out.to(x.dtype).view(tokens, heads * dim))

    def patch(args, extra):
        return extra["original_block"]({**args, "attention": attention})

    return patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prompt",
        default="A red panda sips tea on a sunny wooden porch while birds chirp in the garden.",
    )
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--shift-video", type=float, default=12.0)
    parser.add_argument("--shift-audio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--capture-step", type=int, default=1)
    parser.add_argument(
        "--block-row-stride",
        type=int,
        default=1,
        help="keep every n-th token of block outputs",
    )
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument(
        "--video-vae", default=VIDEO_VAE, help="video VAE file in the models directory"
    )
    parser.add_argument(
        "--compute-dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    parser.add_argument("--lora", help="LoRA file in models/loras applied to the DiT")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument(
        "--lora-rounding",
        choices=["stochastic", "nearest"],
        default="stochastic",
        help="how ComfyUI requantizes quantized weights after merging the LoRA",
    )
    parser.add_argument("--dit", default=DIT, help="DiT file in the models directory")
    parser.add_argument("--first-frame", help="picture file for the first frame")
    parser.add_argument("--last-frame", help="picture file for the last frame")
    parser.add_argument(
        "--sparse-attention", choices=["off", "sol", "vsa"], default="off"
    )
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--sparse-tau", type=float, default=1.3)
    parser.add_argument("--sparse-extra-tokens", type=int, default=256)
    parser.add_argument(
        "--sparse-start",
        type=float,
        default=0.2,
        help="fraction of the steps that stay dense before Sol-Attn starts",
    )
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)
    os.makedirs(arguments.out, exist_ok=True)

    import comfy.cli_args

    comfy.cli_args.args.gpu_only = True
    import comfy.sd
    import comfy.utils
    from comfy import model_management
    from safetensors.torch import load_file

    models = os.path.join(arguments.comfyui, "models")
    device = model_management.get_torch_device()
    metadata = {
        key: str(value)
        for key, value in vars(arguments).items()
        if key not in ("comfyui", "out")
    }

    if arguments.decode_only:
        from safetensors import safe_open

        with safe_open(
            os.path.join(arguments.out, "dit.safetensors"), framework="pt"
        ) as file:
            metadata = file.metadata()
            last = int(metadata["steps"]) - 1
            video = file.get_tensor(f"step{last}.latent.video").to(device)
            audio = file.get_tensor(f"step{last}.latent.audio").to(device)
        decode(arguments, models, device, video, audio, metadata)
        return

    frames, _, _ = temporal_shape(arguments.frames)
    keyframes = []
    if arguments.first_frame or arguments.last_frame:
        import numpy
        from comfy_extras.nodes_minimax_h3 import _resize
        from PIL import Image

        for path, frame_index, crop in (
            (arguments.first_frame, 0, "disabled"),
            (arguments.last_frame, frames - 1, "center"),
        ):
            if path is None:
                continue
            picture = numpy.asarray(
                Image.open(path).convert("RGB"), dtype=numpy.float32
            )
            picture = torch.from_numpy(picture / 255.0)[None]
            keyframes.append(
                {
                    "resolved_frame_index": frame_index,
                    "image": _resize(picture, arguments.width, arguments.height, crop),
                }
            )
    images = [keyframe["image"] for keyframe in keyframes]
    # Each picture becomes one vision token per 32 × 32 pixels.
    vision_tokens = (arguments.width // 32) * (arguments.height // 32)

    started = time.time()
    text_path = os.path.join(arguments.out, "text.safetensors")
    if os.path.exists(text_path):
        # Text encoding takes minutes, so an existing result for the same prompt is
        # reused.
        saved = load_file(text_path)
        context, tags = saved["context"][None], saved["token_tags"]
        print(f"text: reused {text_path}", flush=True)
    else:
        clip = comfy.sd.load_clip(
            ckpt_paths=[os.path.join(models, TEXT_ENCODER)],
            clip_type=comfy.sd.CLIPType.MINIMAX,
        )
        tokens = clip.tokenize(arguments.prompt, images=images)
        # A vision block's embeddings count as -1.
        token_ids = []
        for entry in next(iter(tokens.values()))[0]:
            if isinstance(entry[0], dict):
                token_ids.extend([-1] * vision_tokens)
            else:
                token_ids.append(entry[0])
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        context = conditioning[0][0]
        tags = conditioning[0][1].get("minimax_token_tags")
        if tags is None:
            tags = torch.ones(len(token_ids), dtype=torch.int64)
        save_file(
            {
                "token_ids": torch.tensor(token_ids, dtype=torch.int64),
                "context": context[0].float().contiguous().cpu(),
                "token_tags": tags.contiguous().cpu(),
            },
            text_path,
            metadata=metadata,
        )
        print(
            f"text: {len(token_ids)} tokens, context {tuple(context.shape)}, {time.time() - started:.1f} s",
            flush=True,
        )
        del clip
        model_management.unload_all_models()
        model_management.soft_empty_cache()

    if keyframes:
        started = time.time()
        with torch.inference_mode():
            video_vae = comfy.sd.VAE(
                sd=comfy.utils.load_torch_file(
                    os.path.join(models, arguments.video_vae)
                )
            )
            for keyframe in keyframes:
                keyframe["latent"] = video_vae.encode(keyframe["image"]).float()
        save_file(
            {
                name: value.contiguous()
                for index, keyframe in enumerate(keyframes)
                for name, value in (
                    (f"keyframe.{index}.pixels", keyframe["image"][0].float()),
                    (f"keyframe.{index}.latent", keyframe["latent"][0].cpu()),
                )
            },
            os.path.join(arguments.out, "keyframes.safetensors"),
            metadata={
                **metadata,
                "frame_indices": json.dumps(
                    [keyframe["resolved_frame_index"] for keyframe in keyframes]
                ),
            },
        )
        print(
            f"keyframes: {len(keyframes)} encoded with {arguments.video_vae}, {time.time() - started:.1f} s",
            flush=True,
        )
        del video_vae
        model_management.unload_all_models()
        model_management.soft_empty_cache()

    started = time.time()
    patcher = comfy.sd.load_diffusion_model(os.path.join(models, arguments.dit))
    if arguments.lora:
        if arguments.lora_rounding == "nearest":
            # ComfyUI seeds stochastic rounding from each weight's key, and seed 0 means
            # round to nearest.
            comfy.utils.string_to_seed = lambda key: 0
        lora = comfy.utils.load_torch_file(
            os.path.join(models, "loras", arguments.lora), safe_load=True
        )
        patcher, _ = comfy.sd.load_lora_for_models(
            patcher, None, lora, arguments.lora_strength, 0
        )
    if arguments.sparse_attention == "sol":
        from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention

        # The window opens at sigma 1, and the loop below keeps the early steps dense by
        # hiding their sigma.
        patcher = apply_block_sparse_attention(
            patcher,
            tau=arguments.sparse_tau,
            topk_ratio=0.0,
            vsa=False,
            start_percent=0.0,
            end_percent=1.0,
            min_tokens=12288,
            dense_blocks=set(),
            sink_conditioning="exact_kv_and_rows",
            extra_tokens=arguments.sparse_extra_tokens,
            verbose=False,
        )
    if arguments.sparse_attention == "vsa":
        patcher = patcher.clone()
        plans = {}
        for index, block in enumerate(patcher.model.diffusion_model.blocks):
            patcher.set_model_patch_replace(
                vsa_block_patch(block, arguments.vsa_sparsity, plans),
                "dit",
                "double_block",
                index,
            )
    model_management.load_models_gpu([patcher], force_full_load=True)
    model = patcher.model.diffusion_model
    # ComfyUI's DiT computes in the dtype of the context it receives.
    compute_dtype = (
        torch.float32
        if arguments.compute_dtype == "float32"
        else patcher.model.get_dtype_inference()
    )
    context = context.to(device=device, dtype=compute_dtype)
    _, latent_frames, audio_frames = temporal_shape(arguments.frames)
    generator = torch.Generator().manual_seed(arguments.seed)
    video = torch.randn(
        (1, 24, latent_frames, arguments.height // 16, arguments.width // 16),
        generator=generator,
    )
    audio = torch.randn((1, 32, 2, audio_frames), generator=generator)
    tensors = {"noise.video": video.clone(), "noise.audio": audio.clone()}
    video, audio = video.to(device), audio.to(device)

    watched = sorted({0, len(model.blocks) // 2, len(model.blocks) - 1})
    captured = {}

    capturing = [False]

    def capture(index):
        # A forward hook that returns a value replaces the module output, so this one
        # returns nothing.
        def hook(module, inputs, output):
            if capturing[0]:
                captured[index] = output[:: arguments.block_row_stride].float().cpu()

        return hook

    handles = [
        model.blocks[index].register_forward_hook(capture(index)) for index in watched
    ]

    base = [1.0 - step / arguments.steps for step in range(arguments.steps + 1)]
    sigmas_video = [time_shift(value, arguments.shift_video) for value in base]
    sigmas_audio = [time_shift(value, arguments.shift_audio) for value in base]
    options = {
        "minimax_h3_sigma_shift_video": arguments.shift_video,
        "minimax_h3_sigma_shift_audio": arguments.shift_audio,
    }
    payload = {"text_token_tags": tags, "seed": arguments.seed}
    if keyframes:
        payload["keyframes"] = [
            {
                "resolved_frame_index": keyframe["resolved_frame_index"],
                "latent": keyframe["latent"],
            }
            for keyframe in keyframes
        ]
        payload["cond_video_latents"] = [keyframe["latent"] for keyframe in keyframes]
        # The condition rows with ComfyUI's noise, back in the latent layout.
        rows = model._cond_video_rows(payload, device).float().cpu()
        channels, _, height, width = keyframes[0]["latent"].shape[1:]
        per_keyframe = (height // 2) * (width // 2)
        for index in range(len(keyframes)):
            part = rows[index * per_keyframe : (index + 1) * per_keyframe]
            part = part.view(1, height // 2, width // 2, channels, 1, 2, 2)
            tensors[f"keyframe.{index}.augmented"] = (
                part.permute(3, 0, 4, 1, 5, 2, 6)
                .reshape(channels, 1, height, width)
                .contiguous()
            )
        metadata["frame_indices"] = json.dumps(
            [keyframe["resolved_frame_index"] for keyframe in keyframes]
        )
    with torch.inference_mode():
        for step in range(arguments.steps):
            timestep = torch.tensor([sigmas_video[step] * 1000.0], device=device)
            capturing[0] = step == arguments.capture_step
            if capturing[0]:
                tensors[f"step{step}.input.video"] = video.float().cpu()
                tensors[f"step{step}.input.audio"] = audio.float().cpu()
            step_options = dict(options)
            if arguments.sparse_attention == "vsa":
                step_options.update(patcher.model_options["transformer_options"])
            if arguments.sparse_attention == "sol":
                step_options.update(patcher.model_options["transformer_options"])
                sparse = step >= arguments.sparse_start * arguments.steps
                step_options["sigmas"] = torch.tensor(
                    [sigmas_video[step] if sparse else 2.0]
                )
            outputs = model._forward(
                [video, audio],
                timestep,
                context,
                transformer_options=step_options,
                minimax_payload=payload,
            )
            velocity_video, velocity_audio = (
                (-outputs[0]).float(),
                (-outputs[1]).float(),
            )
            if capturing[0]:
                tensors[f"step{step}.velocity.video"] = velocity_video.cpu()
                tensors[f"step{step}.velocity.audio"] = velocity_audio.cpu()
                for index in watched:
                    tensors[f"step{step}.block.{index}"] = captured[index]
            ratio_video = sigmas_video[step + 1] / sigmas_video[step]
            ratio_audio = sigmas_audio[step + 1] / sigmas_audio[step]
            clean_video = video.float() + sigmas_video[step] * velocity_video
            clean_audio = audio.float() + sigmas_audio[step] * velocity_audio
            video = ratio_video * video.float() + (1.0 - ratio_video) * clean_video
            audio = ratio_audio * audio.float() + (1.0 - ratio_audio) * clean_audio
            tensors[f"step{step}.latent.video"] = video.cpu()
            tensors[f"step{step}.latent.audio"] = audio.cpu()
            print(
                f"dit: step {step} at sigma {sigmas_video[step]:.4f} done, {time.time() - started:.1f} s",
                flush=True,
            )
    metadata["sigmas_video"] = json.dumps(sigmas_video)
    metadata["sigmas_audio"] = json.dumps(sigmas_audio)
    metadata["watched_blocks"] = json.dumps(watched)
    for handle in handles:
        handle.remove()
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        os.path.join(arguments.out, "dit.safetensors"),
        metadata=metadata,
    )
    del patcher, model
    model_management.unload_all_models()
    model_management.soft_empty_cache()

    if arguments.decode:
        decode(arguments, models, device, video, audio, metadata)


def decode(arguments, models, device, video, audio, metadata):
    import comfy.sd
    import comfy.utils
    from comfy import model_management

    started = time.time()
    decoded = {}
    with torch.inference_mode():
        video_vae = comfy.sd.VAE(
            sd=comfy.utils.load_torch_file(os.path.join(models, arguments.video_vae))
        )
        model_management.load_models_gpu([video_vae.patcher], force_full_load=True)
        pixels = video_vae.first_stage_model.decode(
            video.to(device=device, dtype=video_vae.vae_dtype)
        )
        decoded["video"] = pixels.float().cpu().contiguous()
        print(
            f"decode: video with {arguments.video_vae} in {video_vae.vae_dtype}",
            flush=True,
        )
        del video_vae
        model_management.unload_all_models()
        audio_weights = comfy.utils.load_torch_file(os.path.join(models, AUDIO_VAE))
        allow_tf32 = torch.backends.cudnn.allow_tf32
        for name, dtype, tf32 in (
            ("audio", None, allow_tf32),
            ("audio_float32", torch.float32, False),
        ):
            torch.backends.cudnn.allow_tf32 = tf32
            audio_vae = comfy.sd.VAE(sd=dict(audio_weights), dtype=dtype)
            model_management.load_models_gpu([audio_vae.patcher], force_full_load=True)
            waveform = audio_vae.first_stage_model.decode(
                audio.to(device=device, dtype=audio_vae.vae_dtype)
            )
            decoded[name] = waveform.float().cpu().contiguous()
            print(
                f"decode: {name} with the audio VAE in {audio_vae.vae_dtype}, TF32 convolutions {tf32}",
                flush=True,
            )
            del audio_vae
            model_management.unload_all_models()
        torch.backends.cudnn.allow_tf32 = allow_tf32
    save_file(
        decoded, os.path.join(arguments.out, "decode.safetensors"), metadata=metadata
    )
    print(
        f"decode: video {tuple(decoded['video'].shape)}, audio {tuple(decoded['audio'].shape)}, "
        f"{time.time() - started:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
