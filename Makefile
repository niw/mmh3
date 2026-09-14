MODELS = models
# Native output: out.mp4 (NVENC H.264 + AAC) needs the mp4 feature, and out.webm (VP9 + Opus) needs webm.
OUT = out.mp4
FEATURES = cuda,mp4
SEED = 1
export PROMPT = A woman in a yellow raincoat opens a clear umbrella on a neon-lit Tokyo street at night. Rain patters on the umbrella, with distant traffic and soft piano music.

.PHONY: build
build:
	$(if $(shell command -v cargo),,$(error cargo is not installed. Install Rust))
	cargo build --release --features "$(FEATURES)" --bin mmh3

# The fastest measured generation settings. Native output works without ffmpeg.
# NOTE: the prompt goes through the environment so that quotes in it reach mmh3 as they are.
.PHONY: generate
generate: build
	target/release/mmh3 generate \
		--models "$(MODELS)" \
		--out "$(OUT)" \
		--prompt "$$PROMPT" \
		--seed $(SEED) \
		--steps 4 \
		--shift-video 6 \
		--shift-audio 3 \
		--attention sol \
		--attention-precision int8-fp8 \
		--sparse-start 0 \
		--lora "$(MODELS)/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"

.PHONY: download-models
download-models:
	tools/download-models.sh --models "$(MODELS)"

# Formats the Rust code with rustfmt, the Python tools with ruff and the C++ and CUDA code with
# clang-format. uvx runs pinned formatter versions, so the output does not change between releases.
.PHONY: format
format:
	$(if $(shell command -v cargo),,$(error cargo is not installed. Install Rust))
	$(if $(shell command -v uvx),,$(error uvx is not installed. Install uv))
	cargo fmt --all
	uvx ruff@0.16.7 format tools
	uvx clang-format@23.1.1 -i $$(git ls-files '*.cu' '*.cuh' '*.cpp' '*.h')
