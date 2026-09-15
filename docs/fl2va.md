# First and last frames

MiniMax H3's DiT generates from pictures as well as from text: a video that starts from a picture,
ends on one, or both. mmh3 runs this first and last frame mode (FL2VA) with the same checkpoints
as text-to-video.

## Generate

`--first-frame` and `--last-frame` take PNG or JPEG files. The official prompt guide asks for a
first line that says how the pictures align with the video, followed by a blank line:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.17-second mark of the target video.

integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

With only a first frame the line reads `For the target video, at 0.00 seconds into the target
video, <Picture 1> (from [Shot 1]) is fully referenced.`, and with only a last frame `How the
reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the
S.SS-second mark of the target video.`, where S.SS is the video's length, 5.17 seconds for the
default 124 frames. Then, with the [FastH3](fasth3.md) settings of `make generate`:

```sh
target/release/mmh3 generate --prompt-file prompt.txt --out out.mp4 \
  --first-frame first.png --last-frame last.png \
  --steps 4 --attention-precision int8-fp8 \
  --patch minimax_h3_fasth3_vsa_datafree_patch_rank64.safetensors
```

Without `--width` and `--height`, the first picture sets the canvas: a 768-pixel short edge at its
aspect ratio, at most 768 × 1344 pixels, each side a multiple of 32. The first picture given is
stretched to the canvas and the other scaled to cover it and cropped around the center, with
Pillow's Lanczos filter, as the official pipeline fits them.

## How it works

- The text encoder sees the pictures too. Each one goes through the vision tower of the Qwen3-VL
  checkpoint and takes the place of a `<Picture i>: ` vision block before the prompt, about 1,000
  tokens per picture at 768p.
- The video VAE's encoder turns each picture into one latent frame, sampled from its posterior
  like the official pipeline does, with 0.1% of noise mixed in.
- The DiT sees these latents as extra rows after the text, at the time of the first and the last
  frame, and never denoises them. The generated frames at those times start from noise like the
  others.

## Speed and accuracy

Tokyo rain at 1344×768 with the FastH3 patch and INT8/FP8 attention on a DGX Spark, a first and a
last frame: 111.1 s in all, steps 19.1 s. With `--linear-precision nvfp4` 99.3 s, steps 15.5 s. The
pictures add about 4,100 rows to the sequence, which VSA attends from every video tile, so a step
takes about 5 s longer than without pictures. Encoding the prompt with the pictures takes 6.7 s
and the keyframes 2.1 s.

Against ComfyUI in FP32 at 448×256 with a first and a last frame, the DiT's velocity is within
9.1e-3 for video and 2.3e-2 for audio, and the keyframe latents within 1.3e-3. The vision tower
matches ComfyUI's FP32 run to three digits apart from ComfyUI's TF32 patch embedding.
