# Performance and accuracy

## Performance

The times below are from a DGX Spark at 1344×768 and 124 frames (5.2 s), with the Tokyo rain prompt
and the Turbo LoRA (37,770 tokens in all). ComfyUI's times come from random inputs with 1,000 text
tokens (38,710 tokens in all):

| Stage | mmh3 | ComfyUI |
| --- | ---: | ---: |
| DiT step, dense BF16 attention | 38.7 s | 49.7 s |
| DiT step, dense INT8/FP8 attention | 26.7 s | |
| DiT step, Sol-Attn in BF16 | 16.5 s | 23.0 s |
| DiT step, Sol-Attn in INT8/FP8 | 14.6 s | |
| DiT step, Sol-Attn in INT8/FP8, [NVFP4](nvfp4.md) linear layers | 10.7 s | |
| Video VAE decode, FP16 VAE | 30.9 s | 59.2 s |
| Video VAE decode, INT8 ConvRot VAE | 20.2 s | |
| Audio VAE decode | 0.3 s | |
| Text encoder load and encode | 3.4 s | |
| DiT load | 2.9 s | |

A complete run with the [Turbo LoRA](lightx2v-turbo.md) settings, the garden red panda
prompt (75 text tokens) and seed 42 took **86.29 s for MP4** and **90.88 s for WebM**, including
model loading and native encoding. Both runs produced 1344×768, 124-frame clips with stereo audio,
and both files were verified by decoding the entire video and audio streams. These are individual
runs, not averages.

With the Tokyo rain prompt and seed 1, the same settings took 86.76 s without the final video and
audio encoding, 98.98 s with the first step dense, and 116.20 s with BF16 attention and the first
step dense.

## Accuracy

mmh3 is checked against ComfyUI's implementation, stage by stage, with the golden data tools in
`tools/golden`:

- The DiT deviates from ComfyUI's default BF16 run by less than ComfyUI's own BF16 and FP32 runs
  deviate from each other.
- The video VAE decoder matches ComfyUI's FP16 decode at 65 dB PSNR, and the audio VAE decoder
  matches a strict FP32 decode at 108 dB SNR.
- The tokenizer gives the same token ids as Hugging Face tokenizers.
- Sol-Attn is an approximation by design, and mmh3's differs from dense attention as much as
  ComfyUI's does.
- The experimental [NVFP4 linear layers](nvfp4.md) are further from the reference than the INT8
  ones.
