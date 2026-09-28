"""Writes golden data for the Qwen3-VL vision tower of the MiniMax H3 text encoder from
ComfyUI's code.

Development-only tool. Run it with the Python environment of a ComfyUI checkout that
supports MiniMax H3; it runs on the CPU and reads only the checkpoint's visual.* tensors,
about a gigabyte:

    python3 tools/golden/vision.py --comfyui /path/to/ComfyUI \\
        --text-encoder /path/to/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \\
        --out /path/to/vision.safetensors

The tower is built as ComfyUI builds it for the text encoder, with the BF16 weights of the
checkpoint cast to the FP32 of the picture, and embeds a synthetic picture of --width x
--height, or with --frames a pair of frames of a clip. The file holds the picture (or both
frames) in [0, 1], [height, width, 3], the merged embeddings and the three DeepStack
features. crates/mmh3-metal/tests/vision.rs compares the Metal tower with it:

    MMH3_TEXT_ENCODER=... MMH3_VISION_GOLDEN=... \\
        cargo test --release -p mmh3-metal --test vision -- --ignored --nocapture
"""

import argparse
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def synthetic_picture(height, width):
    y = torch.arange(height, dtype=torch.float32)[:, None, None]
    x = torch.arange(width, dtype=torch.float32)[None, :, None]
    c = torch.arange(3, dtype=torch.float32)[None, None, :]
    smooth = 0.35 * torch.sin(x * 0.05 + c * 1.3) * torch.cos(y * 0.037 - c * 0.7)
    detail = 0.1 * torch.sin((x + 2 * y) * 0.31 + c)
    return (0.5 + smooth + detail).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", action="store_true")
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.comfyui)

    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.ops
    import comfy.text_encoders.qwen_vl
    from comfy.text_encoders.minimax import process_video_block
    from comfy.text_encoders.qwen3vl import (
        QWEN3VL_VISION,
        QWEN3VL_VISION_COMMON,
        Qwen3VLVisionModel,
    )

    tensors = {}
    with safe_open(arguments.text_encoder, "pt") as file:
        for name in file.keys():
            if name.startswith("visual."):
                tensors[name[len("visual.") :]] = file.get_tensor(name)
    width = tensors["merger.linear_fc2.weight"].shape[0]
    config = {
        **QWEN3VL_VISION_COMMON,
        **QWEN3VL_VISION["qwen3vl_32b"],
        "out_hidden_size": width,
    }
    model = Qwen3VLVisionModel(
        config, device="cpu", dtype=torch.bfloat16, ops=comfy.ops.manual_cast
    )
    model.load_state_dict(tensors, strict=True)

    picture = synthetic_picture(arguments.height, arguments.width)
    pictures = [picture]
    if arguments.frames:
        pictures.append((1 - picture).roll(7, dims=1))
    started = time.time()
    with torch.inference_mode():
        if arguments.frames:
            patches, grid = process_video_block(torch.stack(pictures))
        else:
            patches, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
                picture[None],
                patch_size=16,
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
            )
        merged, deepstack = model(patches.to(torch.float32), grid)

    output = {"merged": merged.float().contiguous()}
    for index, frame in enumerate(pictures):
        output[f"picture.{index}"] = frame.contiguous()
    for index, features in enumerate(deepstack):
        output[f"deepstack.{index}"] = features.float().contiguous()
    save_file(output, arguments.out)
    print(
        f"grid {grid.tolist()}, merged {tuple(merged.shape)}, "
        f"{time.time() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
