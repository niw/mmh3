"""Writes golden data for the MiniMax H3 text encoder computed by ComfyUI.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that supports MiniMax H3 and has
the text encoder checkpoint in its models directory, on a GPU, for example:

    python3 tools/golden/text_encoder.py --comfyui /path/to/ComfyUI --out /path/to/golden/text-encoder/red-panda.safetensors

The file holds the prompt's token ids, the conditioning (the hidden state after the last layer, without the final
norm) and the hidden states after a few layers, all widened to FP32. --text-encoder picks the checkpoint in
models/text_encoders. --dtype float32 runs the model in FP32 instead of the BF16 ComfyUI picks for the checkpoint, as
a tighter reference.
"""

import argparse
import os
import sys
import time

import torch
from safetensors.torch import save_file

DEFAULT_PROMPT = "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", help="read the prompt from this file instead")
    parser.add_argument("--text-encoder", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    parser.add_argument("--layers", default="0,25,49", help="layers whose output hidden states are stored")
    parser.add_argument("--dtype", choices=["default", "float32"], default="default")
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)
    prompt = open(arguments.prompt_file).read() if arguments.prompt_file else arguments.prompt

    import comfy.cli_args

    comfy.cli_args.args.gpu_only = True
    import comfy.model_management as model_management
    import comfy.sd
    import comfy.text_encoders.minimax

    if arguments.dtype == "float32":
        # ComfyUI builds the model in the dtype it detects in the checkpoint, which overrides model_options.
        build = comfy.text_encoders.minimax.te
        comfy.text_encoders.minimax.te = lambda **kwargs: build(**{**kwargs, "dtype_llama": torch.float32})

    started = time.time()
    path = os.path.join(arguments.comfyui, "models", "text_encoders", arguments.text_encoder)
    clip = comfy.sd.load_clip(ckpt_paths=[path], clip_type=comfy.sd.CLIPType.MINIMAX)
    tokens = clip.tokenize(prompt)
    token_ids = [entry[0] for entry in next(iter(tokens.values()))[0]]

    model_management.load_models_gpu([clip.patcher], force_full_load=True)
    language_model = clip.cond_stage_model.qwen3vl_32b.transformer.model
    watched = [int(layer) for layer in arguments.layers.split(",")]
    captured = {}

    def capture(index):
        # A forward hook that returns a value replaces the module output, so this one returns nothing.
        def hook(module, inputs, output):
            captured[index] = output[0][0].float().cpu()
        return hook

    handles = [language_model.layers[index].register_forward_hook(capture(index)) for index in watched]
    with torch.inference_mode():
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    for handle in handles:
        handle.remove()
    context = conditioning[0][0][0].float().cpu()

    tensors = {"token_ids": torch.tensor(token_ids, dtype=torch.int64), "context": context.contiguous()}
    for index, value in captured.items():
        tensors[f"layer.{index}"] = value.contiguous()
    metadata = {"prompt": prompt, "text_encoder": arguments.text_encoder, "dtype": str(clip.cond_stage_model.dtypes)}
    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    save_file(tensors, arguments.out, metadata=metadata)
    print(f"{len(token_ids)} tokens, context {tuple(context.shape)}, layers {sorted(captured)}, "
          f"{time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
