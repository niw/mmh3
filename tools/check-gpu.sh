#!/usr/bin/env bash
# Checks mmh3's CUDA backend on the GPU of this machine in one run, and collects what it finds.

set -uo pipefail

usage() {
  cat <<EOF
usage: $0 [--models DIR] [--golden DIR] [--out DIR] [--quick] [--no-build]

Builds mmh3, runs the CUDA tests and benchmarks, checks the golden data, generates clips and profiles
one, each step logged on its own, and packs the logs into DIR.tar.gz. A step that fails is recorded
and the run goes on, so one run on a rented card says everything it can. It is meant for a GPU
mmh3 has not run on before, such as an H100 or a B200.

  --models DIR   The models directory (default: models in the repository). Without the models,
                 the golden checks and the generations are skipped. tools/download-models.sh
                 fetches them.
  --golden DIR   The golden data, with small/ and 768p-fasth3/ in it (default:
                 \$XDG_CACHE_HOME/mmh3/golden or ~/.cache/mmh3/golden). Skipped when missing.
  --out DIR      Where the logs go (default: check-gpu-HOST-DATE in the current directory).
  --quick        Leaves out the default-size generation and the profile.
  --no-build     Uses the binaries already in target/release.
EOF
}

repository="$(cd "$(dirname "$0")/.." && pwd)"
models="$repository/models"
golden="${XDG_CACHE_HOME:-$HOME/.cache}/mmh3/golden"
out="check-gpu-$(hostname)-$(date +%Y%m%d-%H%M%S)"
quick=0
build=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) models="$2"; shift ;;
    --golden) golden="$2"; shift ;;
    --out) out="$2"; shift ;;
    --quick) quick=1 ;;
    --no-build) build=0 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

mkdir -p "$out"
out="$(cd "$out" && pwd)"
summary="$out/summary.txt"
: > "$summary"
tools="$repository/target/release/mmh3-tools"
mmh3="$repository/target/release/mmh3"
patch="minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors"
prompt="A woman in a yellow raincoat opens a clear umbrella on a neon-lit Tokyo street at night. Rain patters on the umbrella, with distant traffic and soft piano music."

# Runs a command into $out/NAME.log, and records whether it succeeded and how long it took.
step() {
  local name="$1"
  shift
  local started
  started=$(date +%s)
  echo "== $name: $*" | tee -a "$out/$name.log"
  "$@" >> "$out/$name.log" 2>&1
  local status=$?
  local seconds=$(($(date +%s) - started))
  if [[ $status -eq 0 ]]; then
    echo "ok      $name (${seconds} s)" | tee -a "$summary"
  else
    echo "FAILED  $name (${seconds} s, exit $status, see $name.log)" | tee -a "$summary"
  fi
  return $status
}

skip() {
  echo "skipped $1 ($2)" | tee -a "$summary"
}

# The environment, for reading the logs later.
{
  date
  uname -a
  nvidia-smi
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,power.limit --format=csv
  command -v nvcc > /dev/null && nvcc --version
  rustc --version
  git -C "$repository" log -1 --format='%H %s'
  git -C "$repository" status --short
  command -v ffmpeg > /dev/null && ffmpeg -version | head -1
  command -v nsys > /dev/null && nsys --version
  df -h "$repository"
} > "$out/environment.txt" 2>&1

cd "$repository" || exit 1
if [[ $build -eq 1 ]]; then
  step build cargo build --release --features cuda,mp4,server --bin mmh3 --bin mmh3-tools || {
    echo "the build failed, so nothing else can run" | tee -a "$summary"
    tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
    exit 1
  }
fi
step device "$tools" device
step tests cargo test --release -p mmh3-cuda --no-fail-fast
step bench-mma "$tools" bench mma
step bench-gemm "$tools" bench gemm --kinds int8-mmh3,int8,fp8,bf16,nvfp4
step bench-gemm-mma-sync "$tools" bench gemm --kinds int8-mmh3 --no-wgmma
step bench-gemm-cp-async "$tools" bench gemm --kinds int8-mmh3 --no-tma --no-wgmma
step bench-attention "$tools" bench attention
step bench-attention-mma-sync "$tools" bench attention --no-wgmma
step bench-attention-cp-async "$tools" bench attention --no-tma --no-wgmma

if [[ ! -f "$models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" ]]; then
  skip golden "no models in $models"
  skip generate "no models in $models"
else
  if [[ -f "$golden/small/dit.safetensors" ]]; then
    step golden-dit "$tools" check dit --golden "$golden/small" --models "$models"
    step golden-dit-int8-fp8 "$tools" check dit --golden "$golden/small" \
      --attention-precision int8-fp8 --models "$models"
    step golden-sample "$tools" check sample --golden "$golden/small" --models "$models"
    step golden-video-vae "$tools" check video-vae --golden "$golden/small" \
      --weights "$models/vae/minimax_h3_video_vae_int8_convrot.safetensors" --models "$models"
    step golden-audio-vae "$tools" check audio-vae --golden "$golden/small" --models "$models"
  else
    skip golden-small "no $golden/small"
  fi
  if [[ -f "$golden/768p-fasth3/dit.safetensors" ]]; then
    step golden-dit-768p "$tools" check dit --golden "$golden/768p-fasth3" \
      --patch "$models/patches/$patch" --attention vsa --vsa-sparsity 0.9 --models "$models"
  else
    skip golden-768p "no $golden/768p-fasth3"
  fi

  # MP4 through NVENC where the card has it, and through ffmpeg where it does not (Hopper).
  output=(--out "$out/small.mp4")
  if ! step generate-small "$mmh3" generate --models "$models" --prompt "$prompt" --seed 1 \
    --steps 4 --attention-precision int8-fp8 --patch "$patch" --width 448 --height 256 \
    --frames 39 "${output[@]}"; then
    if grep -q NVENC "$out/generate-small.log" && command -v ffmpeg > /dev/null; then
      output=(--out "$out/small.mp4" --ffmpeg)
      step generate-small-ffmpeg "$mmh3" generate --models "$models" --prompt "$prompt" --seed 1 \
        --steps 4 --attention-precision int8-fp8 --patch "$patch" --width 448 --height 256 \
        --frames 39 "${output[@]}"
    fi
  fi
  if [[ $quick -eq 0 ]]; then
    default_output=("${output[@]}")
    default_output[1]="$out/default.mp4"
    step generate-default "$mmh3" generate --models "$models" --prompt "$prompt" --seed 1 \
      --steps 4 --attention-precision int8-fp8 --patch "$patch" "${default_output[@]}"
    if command -v nsys > /dev/null; then
      profile_output=("${output[@]}")
      profile_output[1]="$out/profile.mp4"
      step profile nsys profile --trace=cuda,nvtx,osrt --force-overwrite true \
        -o "$out/profile" "$mmh3" generate --models "$models" --prompt "$prompt" --seed 1 \
        --steps 2 --attention-precision int8-fp8 --patch "$patch" "${profile_output[@]}" &&
        step profile-stats nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
          --format csv --output "$out/profile" "$out/profile.nsys-rep"
    else
      skip profile "no nsys"
    fi
  fi
fi

tar -czf "$out.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
echo "wrote $out.tar.gz"
