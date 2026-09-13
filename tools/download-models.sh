#!/usr/bin/env bash
# Downloads the MiniMax H3 checkpoints that mmh3 loads into a models directory laid out like ComfyUI's.

set -euo pipefail

usage() {
  cat <<EOF
usage: $0 [--precision int8|fp16] [--lora | --no-lora] [--models DIR]

Downloads the checkpoints mmh3 loads from Hugging Face with the Hugging Face CLI (hf). By default
it downloads the ones for the fastest generation: the INT8 ConvRot DiT, text encoder and video VAE,
the FP32 audio VAE and the lightx2v 4-step Turbo LoRA.

  --precision int8  The INT8 ConvRot video VAE from Kijai/MiniMax-H3-experimental (default).
  --precision fp16  The FP16 video VAE from Comfy-Org/MiniMax-H3 instead. The DiT and the text
                    encoder are INT8 ConvRot either way.
  --lora            The 768p 4-step Turbo LoRA from lightx2v/Minimax-h3-Turbo (default).
  --no-lora         No Turbo LoRA.
  --models DIR      The models directory (default: models in the repository).
EOF
}

fail() {
  echo "error: $1" >&2
  exit 2
}

precision=int8
lora=1
models=$(cd "$(dirname "$0")/.." && pwd)/models

while [[ $# -gt 0 ]]; do
  case $1 in
    --precision)
      [[ $# -ge 2 ]] || fail "--precision needs int8 or fp16"
      precision=$2
      shift
      ;;
    --lora) lora=1 ;;
    --no-lora) lora=0 ;;
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

case $precision in
  int8 | fp16) ;;
  *) fail "--precision must be int8 or fp16, not $precision" ;;
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
if [[ $precision == fp16 ]]; then
  files+=(vae/minimax_h3_video_vae_fp16.safetensors)
fi
hf download Comfy-Org/MiniMax-H3 "${files[@]}" --local-dir "$models"
if [[ $precision == int8 ]]; then
  hf download Kijai/MiniMax-H3-experimental minimax_h3_video_vae_int8_convrot.safetensors \
    --local-dir "$models/vae"
fi
if [[ $lora == 1 ]]; then
  hf download lightx2v/Minimax-h3-Turbo minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors \
    --local-dir "$models/loras"
fi
