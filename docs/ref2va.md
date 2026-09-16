# Reference pictures and sounds

MiniMax H3 has a second DiT for reference to video generation (Ref2VA): the prompt refers to
pictures, videos and sounds, and the video takes people, animals, places or styles from them. mmh3
runs it with reference pictures and sounds so far.

## Download

`tools/download-models.sh --ref2va` downloads the ref2va DiT,
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` from Comfy-Org/MiniMax-H3,
and lightx2v's [Ref2VA Turbo LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo),
`loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`, which makes it generate
in eight steps. `--reference` and `--reference-audio` switch `generate` to the ref2va DiT, and
`--dit` picks another file. Reference sounds need no extra file, since the audio VAE that
`generate` decodes with holds the encoder too.

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

Without the LoRA the DiT runs in 20 steps with the default shifts. The canvas is 1344 × 768
unless `--width` and `--height` say otherwise, as the references do not set it. `--reference` and
`--reference-audio` do not combine with `--first-frame` and `--last-frame`.

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

## Speed and accuracy

The kitten and street example above at 1344×768 and 124 frames on a DGX Spark, with the Turbo LoRA
in eight steps: 177.9 s in all, steps 17.7 s. Without the LoRA in 20 steps, the first four of them
dense, 424.8 s. The two pictures add about 4,500 rows to the sequence with their vision blocks.
Encoding the prompt with the pictures takes 7.8 s and the pictures 2.2 s.

A sound is far cheaper than a picture: it brings no vision tokens, and a two-second one adds 160
rows, two per latent frame at 40 frames per second, against about 2,300 rows for a 768p picture.
Encoding it takes 0.2 s, and a run with one 448×256 reference picture and a two-second sound at
1344×768 and 124 frames takes 14.4 s per step, the same as without the sound.

Against ComfyUI in FP32 at 448×256 with two reference pictures, one of them 128×256, the DiT's
velocity is within 1.1e-2 for video and 1.9e-2 for audio, as close as with keyframes. With one
reference picture and a two-second sound, the sound's posterior mean is within 1.3e-5 of a strict
FP32 encode (2.7e-3 of ComfyUI's default, which uses TF32 convolutions), the condition rows deviate
as much per block as reference pictures alone, and the velocity is within 2.0e-2 for video and
4.0e-2 for audio.
