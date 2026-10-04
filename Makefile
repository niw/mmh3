MODELS = models
# Native output: out.mp4 (NVENC H.264 + AAC) needs the mp4 feature, and out.webm (VP9 + Opus) needs webm.
# Metal MP4 output uses VideoToolbox.
OUT = out.mp4
# Options that come last, after GENERATE_OPTIONS, such as --ffmpeg, which takes the rest of the line.
OUTPUT_OPTIONS =
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
FEATURES = metal,server
else
FEATURES = cuda,mp4,server
endif
SEED = 1
export PROMPT = A woman in a yellow raincoat opens a clear umbrella on a neon-lit Tokyo street at night. Rain patters on the umbrella, with distant traffic and soft piano music.

comma := ,
ifneq ($(filter metal,$(subst $(comma), ,$(FEATURES))),)
# A smaller and shorter clip on Metal.
BACKEND_OPTIONS = --width 672 --height 384 --frames 73
else
BACKEND_OPTIONS = --attention-precision int8-fp8
endif
# A few-step LoRA in place of the FastH3 patch, with the settings on its page in docs.
LORA =
LORA_OPTIONS_pdmd = --steps 2 --shift-video 12 --shift-audio 3 --attention sol --sparse-start 0 \
	--lora minimax_h3_pdmd_2step_lora_rank128_bf16.safetensors
LORA_OPTIONS_turbo = --steps 4 --shift-video 6 --shift-audio 3 --attention sol --sparse-start 0 \
	--lora minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
LORA_OPTIONS_taomate = --schedule taomate --attention sol --sparse-start 0 \
	--lora minimax_h3_taomate_3step_lora_rank128_bf16.safetensors
LORA_OPTIONS_dmad = --steps 4 --shift-video 12 --shift-audio 2 --sampler renoise --attention sol \
	--sparse-start 0 --lora minimax_h3_dmad_4step_full_critic_rank128_bf16.safetensors
LORA_DOWNLOAD_pdmd = --pdmd
LORA_DOWNLOAD_turbo = --lightx2v-turbo
LORA_DOWNLOAD_taomate = --taomate
LORA_DOWNLOAD_dmad = --dmad
ifeq ($(LORA),)
# FastVideo's FastH3 in four steps, as a patch on the base DiT, with VSA.
GENERATE_OPTIONS = --steps 4 $(BACKEND_OPTIONS) \
	--patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
else ifneq ($(LORA_OPTIONS_$(LORA)),)
GENERATE_OPTIONS = $(BACKEND_OPTIONS) $(LORA_OPTIONS_$(LORA))
else
$(error LORA must be pdmd, dmad, turbo or taomate, not $(LORA))
endif

.PHONY: build
build:
	$(if $(shell command -v cargo),,$(error cargo is not installed. Install Rust))
	cargo build --release --features "$(FEATURES)" --bin mmh3

# NOTE: the prompt goes through the environment so that quotes in it reach mmh3 as they are.
.PHONY: generate
generate: build
	target/release/mmh3 generate \
		--models "$(MODELS)" \
		--out "$(OUT)" \
		--prompt "$$PROMPT" \
		--seed $(SEED) \
		$(GENERATE_OPTIONS) \
		$(OUTPUT_OPTIONS)

.PHONY: download-models
download-models:
	tools/download-models.sh --models "$(MODELS)" $(LORA_DOWNLOAD_$(LORA))

# Formats the Rust code with rustfmt, the Python tools with ruff, the Swift code with swiftformat
# and the C++, CUDA and Metal code with clang-format. uvx runs pinned formatter versions, so the
# output does not change between releases. swiftformat only runs where it is installed, since the
# Swift code builds on macOS alone.
.PHONY: format
format:
	$(if $(shell command -v cargo),,$(error cargo is not installed. Install Rust))
	$(if $(shell command -v uvx),,$(error uvx is not installed. Install uv))
	cargo fmt --all
	uvx ruff@0.16.7 format tools
	@if command -v swiftformat >/dev/null; then \
		swiftformat $$(git ls-files '*.swift'); \
	else \
		echo "swiftformat is not installed, skipping the Swift code"; \
	fi
	uvx clang-format@23.1.1 -i $$(git ls-files '*.cu' '*.cuh' '*.cpp' '*.h' '*.metal')
