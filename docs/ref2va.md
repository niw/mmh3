# Reference pictures, sounds and clips

MiniMax H3 has a second DiT for reference to video generation (Ref2VA): the prompt refers to
pictures, videos and sounds, and the video takes people, animals, places or styles from them. mmh3
runs it with all three.

## Download

`tools/download-models.sh --ref2va` downloads the ref2va DiT,
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` from Comfy-Org/MiniMax-H3,
and lightx2v's [Ref2VA Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo),
`loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`, which makes it generate
in eight steps. `--reference`, `--reference-audio` and `--reference-video` switch `generate` to the
ref2va DiT, and `--dit` picks another file. Reference sounds and clips need no extra file, since
the audio and video VAEs that `generate` decodes with hold their encoders too.

## Generate

`--reference` takes a PNG or JPEG file and repeats, up to nine pictures as the official pipeline
allows. The prompt calls them `<Picture 1>`, `<Picture 2>` and so on, in the order of the options.
The official prompt guide (`skills/h3-prompt-writing/references/ref-en.txt` in
MiniMaxAI/MiniMax-H3) writes Ref2VA prompts in six sections, which define what each picture brings
and then describe the video:

```text
subject_definitions:
<Subject 1> is the tabby kitten in <Picture 1>, with grey-brown striped fur and large amber-green eyes.
<Subject 2> is the rainy Tokyo street at night in <Picture 2>, with glowing shop signs and a wet pavement.

summary:
[reference generation] The target video shows <Subject 1> sitting on <Subject 2> at night, watching the rain.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - the kitten's striped fur and amber-green eyes are retained.
<Subject 2> (appears in [Shot 1]): fully_preserved - the shop signs and the wet pavement are retained.

detailed_description:
The target video is in a realistic cinematic style with warm neon light.
[Shot 1] A low medium shot shows <Subject 1> ...

overall_soundscape:
Steady rain patters on the awning and the pavement ...

non_diegetic_music:
Soft, slow piano music plays quietly throughout.
```

With the Turbo LoRA in eight steps, at the shifts it was trained with, and Sol-Attn with INT8/FP8
attention as for the [Turbo LoRA](lightx2v-turbo.md) of the other DiT:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --reference kitten.jpg --reference street.png \
  --steps 8 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
```

`--reference-audio` does the same with a sound the prompt calls `<Audio 1>`, `<Audio 2>` and so
on, up to three as the official pipeline allows. It reads WAV, FLAC, MP3, AAC, ALAC, Ogg Vorbis,
MP4 and Matroska files and resamples them to the audio VAE's 32 kHz, so a sound the video should
take a voice, an instrument or an ambience from goes in as it is:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --reference kitten.jpg --reference-audio rain.flac \
  --steps 8 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
```

`--reference-video` takes an H.264 MP4 the prompt calls `<Video 1>`, `<Video 2>` and so on, up to
three. The clip brings its own soundtrack, which becomes the `<Audio j>` right before it, so a
prompt that refers to a clip usually refers to that sound as well:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --reference-video street.mp4 \
  --steps 8 --shift-video 6 --shift-audio 3 --attention sol \
  --attention-precision int8-fp8 --sparse-start 0 \
  --lora minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
```

Without the LoRA the DiT runs in 20 steps with the default shifts. The canvas is 1344 × 768
unless `--width` and `--height` say otherwise, as the references do not set it. None of
`--reference`, `--reference-audio` and `--reference-video` combine with `--first-frame` and
`--last-frame`.

## How it works

- Each picture keeps its aspect ratio and is scaled down, never up, to the canvas's area, each side
  rounded to a multiple of 32 pixels and resized with Pillow's Lanczos filter. This is the `match`
  sizing of ComfyUI's MiniMaxH3ReferenceToVideo, which the Turbo LoRA was trained with. The
  official pipeline scales every picture to a 2048-pixel short edge instead, several times more
  rows.
- The text encoder sees the pictures as `<Picture i>: ` vision blocks before the prompt, like the
  keyframes of [first and last frame generation](fl2va.md), about 1,000 tokens each at 768p.
- The video VAE's encoder turns each picture into one latent frame, sampled from its posterior with
  0.1% of noise mixed in.
- The DiT sees these latents as rows after the text, each on a latent grid of its own and on one
  unit of the time axis of its own, and never denoises them. The target video starts on the time
  axis after them.
- A sound gets its `<Audio j>: ` label in the prompt and nothing more, since H3 has no audio tower.
  The labels follow the pictures, the order of the official presentation.
- The audio VAE's encoder turns a sound into 40 latent frames per second, the posterior mean with
  no noise mixed in. Those rows sit after the pictures, two per latent frame for the stereo
  channels, on as many units of the time axis as the sound has frames, and they are never
  denoised either.
- A sound is padded with silence to a whole latent frame, which is what the official pipeline
  does. ComfyUI's VAE wrapper crops the tail to a whole frame instead, so a sound whose length is
  not a multiple of 800 samples gives it one frame less.
- A clip's frames are demuxed by mmh3 itself and decoded with NVDEC. They go on the canvas of
  their own aspect ratio, or keep their own size on the 32-pixel grid when that is smaller, since
  a clip is never scaled up, and their count is cut to the video being generated and then down to
  the 17n + 5 frames the VAE's clips cover.
- The text encoder sees a clip at two frames per second, two frames to a vision block, each block
  after a `<T.T seconds>` label and all of them after `<Video k>: `. The two frames take a
  temporal slot each of the block's patches, where a picture repeats itself.
- The video VAE's encoder turns the clip into one latent frame per four, in groups of seventeen
  frames that give five latent frames each, less the three the reference drops from the end.
- The DiT sees a clip's soundtrack rows first and then its video rows, both from the same point of
  the time axis, and it takes as many units of that axis as the longer of the two.
- The soundtrack comes out of the same file through Symphonia and is encoded like any other
  reference sound.

## Speed and accuracy

The kitten and street example above at 1344×768 and 124 frames on a DGX Spark, with the Turbo LoRA
in eight steps: 177.9 s in all, steps 17.7 s. Without the LoRA in 20 steps, the first four of them
dense, 424.8 s. The two pictures add about 4,500 rows to the sequence with their vision blocks.
Encoding the prompt with the pictures takes 7.8 s and the pictures 2.2 s.

A sound is far cheaper than a picture: it brings no vision tokens, and a two-second one adds 160
rows, two per latent frame at 40 frames per second, against about 2,300 rows for a 768p picture.
Encoding it takes 0.2 s, and a run with one 448×256 reference picture and a two-second sound at
1344×768 and 124 frames takes 14.4 s per step, the same as without the sound.

A clip costs what its own size and length ask for. A 22-frame clip of 448×256 encodes in 0.8 s and
adds 784 video rows, 74 audio rows and one vision block, and a 1344×768 video of 124 frames still
runs at 15.1 s per step with it, about two and a half minutes for the eight steps. The same 22
frames at 1344×768 take 10.8 s to encode instead and bring about 37,000 rows, as many as the
target itself, and 124 frames of 1344×768 take 42 s, so a long reference at full size is expensive
on both counts.

Against ComfyUI in FP32 at 448×256 with two reference pictures, one of them 128×256, the DiT's
velocity is within 1.1e-2 for video and 1.9e-2 for audio, as close as with keyframes. With one
reference picture and a two-second sound, the sound's posterior mean is within 1.3e-5 of a strict
FP32 encode (2.7e-3 of ComfyUI's default, which uses TF32 convolutions), the condition rows deviate
as much per block as reference pictures alone, and the velocity is within 2.0e-2 for video and
4.0e-2 for audio.

A reference clip is right where it can be checked exactly: on a tiny model built from ComfyUI's
own code, a forward with a picture, a clip with its soundtrack and a sound matches ComfyUI's to a
cosine of 0.9999999, so the rows of every kind sit where the reference puts them. The frames mmh3
demuxes and decodes itself land within one level of 255 of ComfyUI's decode of the same file, and
the clip's latent is within 1.7e-3 of ComfyUI's encode of the same frames. With the released
checkpoint at 448×256, the velocity is within 1.4e-2 for video and 4.3e-2 for audio of ComfyUI in
FP32, about as far as ComfyUI's own BF16 sits from its FP32 on that case, 1.4e-2 and 3.2e-2.
