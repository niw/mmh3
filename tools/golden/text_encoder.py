"""Writes golden data for the MiniMax H3 text encoder computed by ComfyUI.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3 and has the text encoder checkpoint in its models directory, on a
GPU, for example:

    python3 tools/golden/text_encoder.py --comfyui /path/to/ComfyUI \\
        --out /path/to/golden/text-encoder/red-panda.safetensors

The file holds the prompt's token ids, the conditioning (the hidden state after the last
layer, without the final norm) and the hidden states after a few layers, all widened to
FP32. --text-encoder picks the checkpoint in models/text_encoders. --dtype float32 runs
the model in FP32 instead of the BF16 ComfyUI picks for the checkpoint, as a tighter
reference.

--picture, once per picture, puts pictures in the prompt the way ComfyUI's
MiniMaxH3ImageToVideo does for first and last frames: resized to --width × --height with
Lanczos, the first one stretched and the others cover-cropped. The file then also holds
the pictures, the embeddings and DeepStack features of the vision tower, and the token
tags. Vision embeddings count as -1 in the token ids.
"""

import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file

DEFAULT_PROMPT = (
    "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", help="read the prompt from this file instead")
    parser.add_argument(
        "--text-encoder", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    )
    parser.add_argument(
        "--layers",
        default="0,25,49",
        help="layers whose output hidden states are stored",
    )
    parser.add_argument("--dtype", choices=["default", "float32"], default="default")
    parser.add_argument("--picture", action="append", default=[])
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)
    if arguments.prompt_file:
        with open(arguments.prompt_file) as file:
            prompt = file.read()
    else:
        prompt = arguments.prompt

    import comfy.cli_args

    comfy.cli_args.args.gpu_only = True
    import comfy.sd
    import comfy.text_encoders.minimax
    from comfy import model_management

    if arguments.dtype == "float32":
        # ComfyUI builds the model in the dtype it detects in the checkpoint, which
        # overrides model_options.
        build = comfy.text_encoders.minimax.te
        comfy.text_encoders.minimax.te = lambda **kwargs: build(
            **{**kwargs, "dtype_llama": torch.float32}
        )

    started = time.time()
    path = os.path.join(
        arguments.comfyui, "models", "text_encoders", arguments.text_encoder
    )
    clip = comfy.sd.load_clip(ckpt_paths=[path], clip_type=comfy.sd.CLIPType.MINIMAX)
    pictures = []
    if arguments.picture:
        import numpy
        from comfy_extras.nodes_minimax_h3 import _resize
        from PIL import Image

        for index, picture_path in enumerate(arguments.picture):
            picture = numpy.asarray(
                Image.open(picture_path).convert("RGB"), dtype=numpy.float32
            )
            picture = torch.from_numpy(picture / 255.0)[None]
            crop = "disabled" if index == 0 else "center"
            pictures.append(_resize(picture, arguments.width, arguments.height, crop))
    tokens = clip.tokenize(prompt, images=pictures)
    vision_tokens = (arguments.width // 32) * (arguments.height // 32)
    token_ids = []
    for entry in next(iter(tokens.values()))[0]:
        if isinstance(entry[0], dict):
            token_ids.extend([-1] * vision_tokens)
        else:
            token_ids.append(entry[0])

    model_management.load_models_gpu([clip.patcher], force_full_load=True)
    language_model = clip.cond_stage_model.qwen3vl_32b.transformer.model
    watched = [int(layer) for layer in arguments.layers.split(",")]
    captured = {}

    def capture(index):
        # A forward hook that returns a value replaces the module output, so this one
        # returns nothing.
        def hook(module, inputs, output):
            captured[index] = output[0][0].float().cpu()

        return hook

    handles = [
        language_model.layers[index].register_forward_hook(capture(index))
        for index in watched
    ]
    vision_outputs = []
    if pictures:
        visual = clip.cond_stage_model.qwen3vl_32b.transformer.visual

        def capture_vision(module, inputs, output):
            merged, deepstack = output
            vision_outputs.append(
                (merged.float().cpu(), [value.float().cpu() for value in deepstack])
            )

        handles.append(visual.register_forward_hook(capture_vision))
    with torch.inference_mode():
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    for handle in handles:
        handle.remove()
    context = conditioning[0][0][0].float().cpu()

    tensors = {
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
        "context": context.contiguous(),
    }
    tags = conditioning[0][1].get("minimax_token_tags")
    if tags is not None:
        tensors["token_tags"] = tags.contiguous().cpu()
    for index, (picture, (merged, deepstack)) in enumerate(
        zip(pictures, vision_outputs)
    ):
        tensors[f"picture.{index}.pixels"] = picture[0].float().contiguous()
        tensors[f"picture.{index}.merged"] = merged.contiguous()
        for layer, value in enumerate(deepstack):
            tensors[f"picture.{index}.deepstack.{layer}"] = value.contiguous()
    for index, value in captured.items():
        tensors[f"layer.{index}"] = value.contiguous()
    metadata = {
        "prompt": prompt,
        "text_encoder": arguments.text_encoder,
        "dtype": str(clip.cond_stage_model.dtypes),
    }
    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    save_file(tensors, arguments.out, metadata=metadata)
    print(
        f"{len(token_ids)} tokens, context {tuple(context.shape)}, layers {sorted(captured)}, "
        f"{time.time() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
