# tools/models

Development tools that build model files for mmh3. Generating video does not need them. Each script
lists its Python dependencies in its header, and `uv run` installs them into a cached environment,
so no other setup is needed.

## fasth3_vsa_patch.py

Builds a patch that turns the pruned INT8 ConvRot FL2VA DiT, which mmh3 loads by default, into
FastVideo's [FastH3 VSA-DataFree](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
4-step model. mmh3 applies it with `--patch`. It is 2.7 GB, because it reuses the base DiT. The
patch built with the defaults is published at
[yniw/MiniMax-H3-mmh3](https://huggingface.co/yniw/MiniMax-H3-mmh3), and
`tools/download-models.sh` downloads it.

It reads two checkpoints of about 66 GB each: the full BF16 FL2VA DiT of Comfy-Org/MiniMax-H3, and
the `transformer` directory of the FastH3 repository. The Hugging Face CLI prints where it puts
them:

```sh
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_bf16.safetensors
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree --include "transformer/*"

uv run tools/models/fasth3_vsa_patch.py \
  --base /path/to/minimax_h3_fl2va_bf16.safetensors \
  --fasth3 /path/to/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/transformer \
  --out models/patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
```

It computes on a CUDA GPU with PyTorch, which `uv run` installs for CUDA 13.0. On a DGX Spark it
takes about 70 seconds and 4 GB of host memory. It reads the tensors in threads a few layers ahead
of the computation and drops them from the page cache.

The patch is a LoRA plus the tensors that a LoRA cannot carry, under ComfyUI's names:

- LoRA updates (`lora_A`, `lora_B`, `alpha`) of the 208 linear layers of the blocks and the text
  refiner: the truncated SVD of the FastH3 weight minus the base weight, of rank `--rank` (64 by
  default, a multiple of 64).
- The VSA gates `to_gate_compress`, which the base DiT does not have, in INT8 ConvRot like its other
  linear layers (1.9 GB of the patch).
- The fine-tuned AdaLN in the pruned layout: `adaln_t_table`, a rank-8 basis of the time-embedding
  curve on 1025 timesteps after subtracting its mean, and every AdaLN projection folded onto it.
  The modulations stay within 2.1e-4 of the full ones, about as close as in Comfy-Org's pruned
  checkpoints.
- The norms and projections that FastH3 changed, as they are.

FastH3 trains FP32 master weights and ships BF16 ones, so its linear updates are the base weights
moved by one or a few BF16 steps where the training moved them far enough. They amount to about
1e-4 of the weights in norm, and rank 512 keeps only 20 to 60% of their energy. They still matter.
Against FP32 golden data of FastH3 from `tools/golden/h3_reference.py --sparse-attention vsa`, the
DiT velocity deviates as follows:

| DiT | Video | Audio |
| --- | ---: | ---: |
| Base + patch, rank 128 | 1.84e-1 | 3.63e-2 |
| Base + patch, rank 64 | 1.61e-1 | 3.81e-2 |
| Base + patch without LoRA updates (`--rank 0`) | 3.23e-1 | 6.11e-2 |

Ranks 64 and 128 are within the spread of this comparison and run at the same speed, so the
default is the smaller file.

The rank-64 numbers are those of the published patch. A patch built again gives the same LoRA
energy but not the same bits, because the randomized SVD rounds differently on another device, and
FastH3 amplifies such differences: a GPU build gives 1.79e-1 and 3.81e-2 here. Over five builds with
different SVD seeds on the CPU and the GPU, the video deviation against FP32 golden data with every
VSA tile kept (`--vsa-sparsity 0`) spreads from 1.65e-1 to 2.11e-1, with the published patch at
1.75e-1.
