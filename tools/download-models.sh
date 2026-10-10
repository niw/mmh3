#!/usr/bin/env bash
# Downloads the MiniMax H3 checkpoints that mmh3 loads into a models directory laid out like ComfyUI's.

set -euo pipefail

usage() {
  cat <<EOF
usage: $0 [--video-vae int8|fp16] [--ref2va | --no-ref2va] [--fasth3 | --no-fasth3]
       [--lightx2v-turbo | --no-lightx2v-turbo] [--taomate | --no-taomate] [--pdmd | --no-pdmd]
       [--dmad | --no-dmad] [--veda | --no-veda] [--latent-upscaler | --no-latent-upscaler]
       [--models DIR]

Downloads the checkpoints mmh3 loads from Hugging Face with the Hugging Face CLI (hf), or with curl
when hf is not installed. curl sends HF_TOKEN when it is set. By default it downloads the ones make
generate uses: the INT8 ConvRot DiT, text encoder and video VAE, the FP32 audio VAE and the FastH3
VSA-DataFree patch.

  --video-vae int8     The INT8 ConvRot video VAE from yniw/MiniMax-H3-mmh3 (default).
  --video-vae fp16     The FP16 video VAE from Comfy-Org/MiniMax-H3 instead. The DiT and the text
                       encoder are INT8 ConvRot either way.
  --ref2va             The INT8 ConvRot ref2va DiT from Comfy-Org/MiniMax-H3 and the 768p 8-step
                       Ref2VA Turbo LoRA from lightx2v/Minimax-h3-Turbo, into loras.
  --no-ref2va          No ref2va DiT or LoRA (default).
  --fasth3             The FastH3 VSA-DataFree patch from yniw/MiniMax-H3-mmh3, into patches
                       (default).
  --no-fasth3          No FastH3 patch.
  --lightx2v-turbo     The 768p 4-step Turbo LoRA from lightx2v/Minimax-h3-Turbo, into loras.
  --no-lightx2v-turbo  No Turbo LoRA (default).
  --taomate            The 3-step TaoMate-H3 LoRA from yniw/MiniMax-H3-mmh3, into loras.
  --no-taomate         No TaoMate LoRA (default).
  --pdmd               The 2-step PDMD LoRA from yniw/MiniMax-H3-mmh3, into loras.
  --no-pdmd            No PDMD LoRA (default).
  --dmad               The 4-step DMAD LoRA from yniw/MiniMax-H3-mmh3, into loras.
  --no-dmad            No DMAD LoRA (default).
  --veda               Veda's sparse attention predictor from
                       Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview, into veda.
  --no-veda            No Veda predictor (default).
  --latent-upscaler    LBH-123-AI's latent upscaler for draft steps from
                       LBH-123-AI/Minimax_h3_latent_Upscaler, into latent_upscale_models.
  --no-latent-upscaler No latent upscaler (default).
  --models DIR         The models directory (default: models in the repository).
EOF
}

fail() {
  echo "error: $1" >&2
  exit 2
}

video_vae=int8
ref2va=0
fasth3=1
turbo=0
taomate=0
pdmd=0
dmad=0
veda=0
latent_upscaler=0
models=$(cd "$(dirname "$0")/.." && pwd)/models

while [[ $# -gt 0 ]]; do
  case $1 in
    --video-vae)
      [[ $# -ge 2 ]] || fail "--video-vae needs int8 or fp16"
      video_vae=$2
      shift
      ;;
    --ref2va) ref2va=1 ;;
    --no-ref2va) ref2va=0 ;;
    --fasth3) fasth3=1 ;;
    --no-fasth3) fasth3=0 ;;
    --lightx2v-turbo) turbo=1 ;;
    --no-lightx2v-turbo) turbo=0 ;;
    --taomate) taomate=1 ;;
    --no-taomate) taomate=0 ;;
    --pdmd) pdmd=1 ;;
    --no-pdmd) pdmd=0 ;;
    --dmad) dmad=1 ;;
    --no-dmad) dmad=0 ;;
    --veda) veda=1 ;;
    --no-veda) veda=0 ;;
    --latent-upscaler) latent_upscaler=1 ;;
    --no-latent-upscaler) latent_upscaler=0 ;;
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

# NOTE: Without hf, curl resumes each file from DEST/FILE.incomplete and renames it when the
# download completes, so a file that already exists is complete and is skipped.
download() {
  local repo=$1 dest=$2
  shift 2
  if command -v hf >/dev/null; then
    hf download "$repo" "$@" --local-dir "$dest"
    return
  fi
  local curl_options=(--location --fail --continue-at - --retry 5)
  if [[ -n ${HF_TOKEN:-} ]]; then
    curl_options+=(--header "Authorization: Bearer $HF_TOKEN")
  fi
  local file path
  for file in "$@"; do
    path=$dest/$file
    if [[ -f $path ]]; then
      echo "$path exists, skipping"
      continue
    fi
    mkdir -p "$(dirname "$path")"
    echo "Downloading $repo/$file"
    curl "${curl_options[@]}" --output "$path.incomplete" \
      "https://huggingface.co/$repo/resolve/main/$file"
    mv "$path.incomplete" "$path"
  done
}

if ! command -v hf >/dev/null && ! command -v curl >/dev/null; then
  fail "neither the Hugging Face CLI (hf) nor curl is installed"
fi

files=(
  diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
  text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  vae/minimax_h3_audio_vae_fp32.safetensors
)
if [[ $video_vae == fp16 ]]; then
  files+=(vae/minimax_h3_video_vae_fp16.safetensors)
fi
if [[ $ref2va == 1 ]]; then
  files+=(diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)
fi
download Comfy-Org/MiniMax-H3 "$models" "${files[@]}"
mmh3_files=()
if [[ $video_vae == int8 ]]; then
  mmh3_files+=(vae/minimax_h3_video_vae_int8_convrot.safetensors)
fi
if [[ $fasth3 == 1 ]]; then
  mmh3_files+=(patches/minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors)
fi
if [[ $taomate == 1 ]]; then
  mmh3_files+=(loras/minimax_h3_taomate_3step_lora_rank128_bf16.safetensors)
fi
if [[ $pdmd == 1 ]]; then
  mmh3_files+=(loras/minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors)
fi
if [[ $dmad == 1 ]]; then
  mmh3_files+=(loras/minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors)
fi
if [[ ${#mmh3_files[@]} -gt 0 ]]; then
  download yniw/MiniMax-H3-mmh3 "$models" "${mmh3_files[@]}"
fi
turbo_files=()
if [[ $turbo == 1 ]]; then
  turbo_files+=(minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors)
fi
if [[ $ref2va == 1 ]]; then
  turbo_files+=(minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors)
fi
if [[ ${#turbo_files[@]} -gt 0 ]]; then
  download lightx2v/Minimax-h3-Turbo "$models/loras" "${turbo_files[@]}"
fi
if [[ $veda == 1 ]]; then
  download Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview "$models/veda" \
    minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors
fi
if [[ $latent_upscaler == 1 ]]; then
  download LBH-123-AI/Minimax_h3_latent_Upscaler "$models/latent_upscale_models" \
    minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors
fi
