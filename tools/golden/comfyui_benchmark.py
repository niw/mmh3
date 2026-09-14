"""Measures ComfyUI's MiniMax H3 speed at 768p for 5 seconds, the baseline for mmh3.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3 and has the checkpoints in its models directory, on a GPU with
nothing else running, for example:

    python3 tools/golden/comfyui_benchmark.py --comfyui /path/to/ComfyUI

It times one DiT call with ComfyUI's default dense attention and with the Sol-Attn patch
of the Model Sparse Attention node at its defaults, and one video VAE decode, all on
random inputs of the target shape.
"""

import argparse
import json
import os
import sys
import time

import torch

DIT = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VIDEO_VAE = "vae/minimax_h3_video_vae_fp16.safetensors"


def timed(function, runs):
    torch.cuda.synchronize()
    function()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(runs):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / runs


def benchmark_dit(arguments, models, device, generator, shape, results):
    """Times one DiT call with dense attention and with Sol-Attn.

    The model is released when this returns.
    """
    import comfy.sd
    from comfy import model_management
    from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention

    patcher = comfy.sd.load_diffusion_model(os.path.join(models, DIT))
    model_management.load_models_gpu([patcher], force_full_load=True)
    model = patcher.model.diffusion_model
    dtype = patcher.model.get_dtype_inference()
    video = torch.randn((1, 24) + shape, generator=generator).to(device)
    audio = torch.randn((1, 32, 2, arguments.audio_frames), generator=generator).to(
        device
    )
    context = torch.randn((1, arguments.text_tokens, 5120), generator=generator).to(
        device=device, dtype=dtype
    )
    payload = {"text_token_tags": torch.ones(arguments.text_tokens, dtype=torch.int64)}
    timestep = torch.tensor([500.0], device=device)
    tokens = (
        arguments.text_tokens
        + 2 * arguments.audio_frames
        + shape[0] * shape[1] * shape[2] // 4
    )
    results["tokens"] = tokens

    with torch.inference_mode():
        results["dit_dense_seconds"] = timed(
            lambda: model._forward(
                [video, audio],
                timestep,
                context,
                transformer_options={},
                minimax_payload=payload,
            ),
            arguments.runs,
        )
        print(json.dumps(results), flush=True)

        sparse = apply_block_sparse_attention(
            patcher,
            tau=1.3,
            topk_ratio=0.0,
            vsa=False,
            start_percent=0.0,
            end_percent=1.0,
            min_tokens=12288,
            dense_blocks=set(),
            sink_conditioning="exact_kv_and_rows",
            extra_tokens=256,
            verbose=False,
        )
        options = sparse.model_options["transformer_options"]
        results["dit_sol_attn_seconds"] = timed(
            lambda: model._forward(
                [video, audio],
                timestep,
                context,
                transformer_options=dict(options),
                minimax_payload=payload,
            ),
            arguments.runs,
        )
        print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--text-tokens", type=int, default=1000)
    parser.add_argument("--latent-frames", type=int, default=37)
    parser.add_argument("--latent-height", type=int, default=48)
    parser.add_argument("--latent-width", type=int, default=84)
    parser.add_argument("--audio-frames", type=int, default=207)
    parser.add_argument("--runs", type=int, default=2)
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)

    import comfy.cli_args

    comfy.cli_args.args.gpu_only = True
    import comfy.sd
    import comfy.utils
    from comfy import model_management
    from comfy.ldm.modules import attention

    models = os.path.join(arguments.comfyui, "models")
    device = model_management.get_torch_device()
    results = {"attention_backend": attention.optimized_attention.__name__}

    generator = torch.Generator().manual_seed(0)
    shape = (arguments.latent_frames, arguments.latent_height, arguments.latent_width)
    benchmark_dit(arguments, models, device, generator, shape, results)
    model_management.unload_all_models()
    model_management.soft_empty_cache()

    with torch.inference_mode():
        vae = comfy.sd.VAE(
            sd=comfy.utils.load_torch_file(os.path.join(models, VIDEO_VAE))
        )
        model_management.load_models_gpu([vae.patcher], force_full_load=True)
        latent = torch.randn((1, 24) + shape, generator=generator).to(
            device=device, dtype=vae.vae_dtype
        )
        results["video_vae_decode_seconds"] = timed(
            lambda: vae.first_stage_model.decode(latent), 1
        )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
