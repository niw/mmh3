#!/usr/bin/env bash
# Downloads the MiniMax H3 checkpoints that mmh3 loads into a models directory laid out like ComfyUI's.

set -euo pipefail

usage() {
  cat <<EOF
usage: $0 [--video-vae int8|fp16] [--fasth3 | --no-fasth3] [--lightx2v-turbo | --no-lightx2v-turbo]
       [--models DIR]

Downloads the checkpoints mmh3 loads from Hugging Face with the Hugging Face CLI (hf). By default
it downloads the ones make generate uses: the INT8 ConvRot DiT, text encoder and video VAE, the FP32
audio VAE and the FastH3 VSA-DataFree patch.

  --video-vae int8     The INT8 ConvRot video VAE from Kijai/MiniMax-H3-experimental (default).
  --video-vae fp16     The FP16 video VAE from Comfy-Org/MiniMax-H3 instead. The DiT and the text
                       encoder are INT8 ConvRot either way.
  --fasth3             The FastH3 VSA-DataFree patch from yniw/MiniMax-H3-mmh3, into patches
                       (default).
  --no-fasth3          No FastH3 patch.
  --lightx2v-turbo     The 768p 4-step Turbo LoRA from lightx2v/Minimax-h3-Turbo, into loras.
  --no-lightx2v-turbo  No Turbo LoRA (default).
  --models DIR         The models directory (default: models in the repository).
EOF
}

fail() {
  echo "error: $1" >&2
  exit 2
}

video_vae=int8
fasth3=1
turbo=0
models=$(cd "$(dirname "$0")/.." && pwd)/models

while [[ $# -gt 0 ]]; do
  case $1 in
    --video-vae)
      [[ $# -ge 2 ]] || fail "--video-vae needs int8 or fp16"
      video_vae=$2
      shift
      ;;
    --fasth3) fasth3=1 ;;
    --no-fasth3) fasth3=0 ;;
    --lightx2v-turbo) turbo=1 ;;
    --no-lightx2v-turbo) turbo=0 ;;
    --models)
      [[ $# -ge 2 ]] || fail "--models needs a directory"
      models=$2
      shift
      ;;
    -h | --help)
      usage
      exit
      ;;
    *) fail "unknown option $1, see $0 --help" ;;
  esac
  shift
done

case $video_vae in
  int8 | fp16) ;;
  *) fail "--video-vae must be int8 or fp16, not $video_vae" ;;
esac

if ! command -v hf >/dev/null; then
  cat >&2 <<EOF
error: the Hugging Face CLI (hf) is not installed. Install it, for example with uv:

  uv tool install huggingface_hub

and run this script again. To install uv, see https://docs.astral.sh/uv/getting-started/installation/.
Other ways to install hf are in https://huggingface.co/docs/huggingface_hub/guides/cli.
EOF
  exit 1
fi

files=(
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
)
if [[ $video_vae == fp16 ]]; then
  files+=(vae/minimax_h3_video_vae_fp16.safetensors)
fi
hf download Comfy-Org/MiniMax-H3 "${files[@]}" --local-dir "$models"
if [[ $video_vae == int8 ]]; then
  hf download Kijai/MiniMax-H3-experimental minimax_h3_video_vae_int8_convrot.safetensors \
    --local-dir "$models/vae"
fi
if [[ $turbo == 1 ]]; then
  hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
    --local-dir "$models/loras"
fi
if [[ $fasth3 == 1 ]]; then
  hf download yniw/MiniMax-H3-mmh3 patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors \
    --local-dir "$models"
fi
