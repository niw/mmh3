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

## video_vae_int8.py

Builds the INT8 ConvRot video VAE, `vae/minimax_h3_video_vae_int8_convrot.safetensors`, from the
FP16 video VAE of Comfy-Org/MiniMax-H3 and calibration latents. It converts the linear layers of the
36 decoder blocks and keeps every other tensor in FP16, so mmh3 decodes it as fast as Kijai's INT8
video VAE of the same format, and ComfyUI loads it too. The file is 2.8 GB.

The calibration latents come from `mmh3-tools latent`, which samples like `mmh3 generate` and
writes the final latents instead of decoding them. The results below use six 1344×768, 39-frame
FastH3 latents of the prompts in `video_vae_calibration_prompts.txt`, with seeds 11 to 16. With
`MMH3_MODELS` set as for `mmh3` and `mmh3-tools` built with the `cuda` feature:

```sh
seed=11
while read -r prompt; do
  target/release/mmh3-tools latent --prompt "$prompt" --seed $seed --steps 4 --frames 39 \
    --attention-precision int8-fp8 \
    --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors \
    --out latents/$seed.safetensors
  seed=$((seed + 1))
done < tools/models/video_vae_calibration_prompts.txt

hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors

uv run tools/models/video_vae_int8.py \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --latents 'latents/*.safetensors' \
  --out models/vae/minimax_h3_video_vae_int8_convrot.safetensors
```

`--latents` also takes golden files of `tools/golden/h3_reference.py`, whose last step it uses. The
tool draws `--tiles` tiles, 96 by default, from all the latents. A tile is what mmh3 decodes at
once: 7 latent frames of 16 × 16 positions. It runs the decoder on the GPU in FP32 block by block,
with the blocks before already quantized, and quantizes each layer in two steps, described in the
script:

- Smoothing divides each input channel by `sqrt(rms(x) / |w|)` and multiplies the weight column by
  it. The division folds into the RMSNorm weight or into the rows of the layer before, so the
  decoder computes the same function, and the per-row INT8 activations lose less to rounding.
- GPTQ rounds the rotated weights column by column and spreads each rounding error over the columns
  left, through the inverse Hessian of the layer's calibration inputs.

On a DGX Spark the six latents take about 4 minutes, and the tool about 4 minutes and 4.5 GB of host
memory.

`mmh3-tools check video-vae --reference` compares an INT8 decode with the FP16 one. The table gives
the PSNR of the pixels, averaged over five 1344×768 latents of FastH3 outside the calibration: four
39-frame latents of other prompts and the 124-frame golden latent. Rounding the pixels to 8 bits
alone gives about 58.9 dB.

| INT8 video VAE | PSNR |
| --- | ---: |
| Kijai/MiniMax-H3-experimental | 58.2 dB |
| Rounding to nearest (`--method rtn --no-smoothing`) | 58.2 dB |
| Smoothing only (`--method rtn`), 12 tiles | 59.5 dB |
| GPTQ only (`--no-smoothing`), 12 tiles | 59.7 dB |
| Smoothing and GPTQ, 12 tiles | 60.4 dB |
| Smoothing and GPTQ, 48 tiles | 60.8 dB |
| Smoothing and GPTQ, 96 tiles (the defaults) | 60.9 dB |
