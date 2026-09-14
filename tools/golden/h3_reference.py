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
ComfyUI's Sol-Attn from the step at --sparse-start of the run.

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
    parser.add_argument("--sparse-attention", choices=["off", "sol"], default="off")
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
        tokens = clip.tokenize(arguments.prompt)
        token_ids = [entry[0] for entry in next(iter(tokens.values()))[0]]
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

    started = time.time()
    patcher = comfy.sd.load_diffusion_model(os.path.join(models, DIT))
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
    payload = {"text_token_tags": tags}
    with torch.inference_mode():
        for step in range(arguments.steps):
            timestep = torch.tensor([sigmas_video[step] * 1000.0], device=device)
            capturing[0] = step == arguments.capture_step
            if capturing[0]:
                tensors[f"step{step}.input.video"] = video.float().cpu()
                tensors[f"step{step}.input.audio"] = audio.float().cpu()
            step_options = dict(options)
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
