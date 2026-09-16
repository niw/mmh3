MODELS = models
# Native output: out.mp4 (NVENC H.264 + AAC) needs the mp4 feature, and out.webm (VP9 + Opus) needs webm.
# Metal MP4 output uses VideoToolbox.
OUT = out.mp4
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
FEATURES = metal
else
FEATURES = cuda,mp4
endif
SEED = 1
export PROMPT = A woman in a yellow raincoat opens a clear umbrella on a neon-lit Tokyo street at night. Rain patters on the umbrella, with distant traffic and soft piano music.

comma := ,
ifneq ($(filter metal,$(subst $(comma), ,$(FEATURES))),)
# Start with a short clip using dense attention and the four-step Turbo LoRA on Metal.
GENERATE_OPTIONS = --width 448 --height 256 --frames 39 --steps 4 \
	--shift-video 6 --shift-audio 3 --attention dense \
	--lora minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors
DOWNLOAD_OPTIONS = --no-fasth3 --lightx2v-turbo
else
# FastVideo's FastH3 in four steps, as a patch on the base DiT, with VSA and INT8/FP8 attention.
GENERATE_OPTIONS = --steps 4 --attention-precision int8-fp8 \
	--patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
DOWNLOAD_OPTIONS =
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
		$(GENERATE_OPTIONS)

.PHONY: download-models
download-models:
	tools/download-models.sh --models "$(MODELS)" $(DOWNLOAD_OPTIONS)

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
