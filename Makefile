MODELS = models
OUT = out.y4m
SEED = 1
export PROMPT = A woman in a yellow raincoat opens a clear umbrella on a neon-lit Tokyo street at night. Rain patters on the umbrella, with distant traffic and soft piano music.

.PHONY: build
build:
	$(if $(shell command -v cargo),,$(error cargo is not installed. Install Rust))
	cargo build --release --features cuda --bin mmh3

# The fastest measured settings, with the video and audio encoded as an MP4 next to them.
# NOTE: the prompt goes through the environment so that quotes in it reach mmh3 as they are.
.PHONY: generate
generate: build
	$(if $(shell command -v ffmpeg),,$(error ffmpeg, which encodes the MP4, is not installed))
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
	ffmpeg -y -loglevel error \
		-i "$(OUT)" \
		-i "$(OUT:.y4m=.wav)" \
		-c:v libx264 \
		-crf 18 \
		-colorspace bt709 \
		-color_primaries bt709 \
		-color_trc bt709 \
		-c:a aac \
		-b:a 192k \
		-shortest \
		"$(OUT:.y4m=.mp4)"

.PHONY: download-models
download-models:
	tools/download-models.sh --models "$(MODELS)"
