# Models

mmh3 finds each checkpoint under its ComfyUI name in a models directory laid out like ComfyUI's
`models` folder, so a ComfyUI installation's `models` directory works as is. `mmh3` takes the
directory with `--models DIR` or `MMH3_MODELS`, and single files with `--dit`, `--text-encoder`,
`--video-vae` and `--audio-vae`.

`make download-models` runs `tools/download-models.sh`, which downloads these files into `models`:

| File | From |
| --- | --- |
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | Comfy-Org/MiniMax-H3 |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | yniw/MiniMax-H3-mmh3 |
| `patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors` | yniw/MiniMax-H3-mmh3 |

With `--video-vae fp16`, the script downloads the FP16 video VAE,
`vae/minimax_h3_video_vae_fp16.safetensors` from Comfy-Org/MiniMax-H3, instead of the INT8 one.
`--ref2va` also downloads the ref2va DiT and its 8-step Turbo LoRA for [reference
pictures](ref2va.md), `--no-fasth3` skips the [FastH3](fasth3.md) patch, `--lightx2v-turbo` also
downloads the [Turbo LoRA](lightx2v-turbo.md),
`loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors` from
lightx2v/Minimax-h3-Turbo, `--taomate` also downloads the [TaoMate-H3](taomate.md) LoRA,
`loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors` from yniw/MiniMax-H3-mmh3, and
`--models DIR` downloads into another directory.

Patches, such as the FastH3 patch, go into `patches` of the models directory. `--patch` and `--lora`
take a path, or a file name that mmh3 looks up in `patches` and then `loras` of the models directory
for `--patch`, and in `loras` for `--lora`, so patches kept with ComfyUI's LoRAs work too.

`generate` decodes with the INT8 video VAE when the models directory has it, and with the FP16
one otherwise. The INT8 one is faster, and on FastH3 videos its pixels are about 61 dB PSNR from the
FP16 decode, closer than rounding them to 8 bits. `tools/models/video_vae_int8.py` builds it.
