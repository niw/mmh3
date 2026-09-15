# Usage

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 generate --out out.mp4 \
  --prompt "A red panda sips tea on a sunny wooden porch while birds chirp in the garden."
```

This writes `out.mp4` with native NVENC H.264 video and AAC audio. The extension of `--out` selects
the container, and `--ffmpeg` hands the encoding to an installed ffmpeg instead.
[Output](output.md) describes the formats and the ffmpeg path.

## make generate

`make generate` runs the [FastH3](fasth3.md) settings. `PROMPT`, `SEED`, `OUT` and `MODELS` change
the prompt, the seed, the output file and the models directory. This writes `panda.mp4`:

```sh
make generate \
  PROMPT="A red panda sips tea on a sunny wooden porch while birds chirp in the garden." \
  SEED=2 OUT=panda.mp4
```

## Options

- `--out FILE` (required, `make generate` uses `out.mp4`): `.mp4` uses native NVENC H.264 + AAC in
  an `mp4` build. `.webm` uses native VP9 + Opus in a `webm` build.
- `--ffmpeg [ARGS...]` (default off): Use an installed ffmpeg instead. All following arguments
  belong to ffmpeg.
- `--prompt TEXT`, `--prompt-file FILE`: The prompt, raw text without a chat template.
- `--width N`, `--height N` (default 1344, 768): Canvas size, multiples of 32. H3 is trained with a
  768-pixel short edge.
- `--frames N` (default 124): Frame count at 24 fps, rounded up to the next 17n + 5.
- `--first-frame FILE`, `--last-frame FILE` (default none): PNG or JPEG pictures the video starts
  from and ends on. Without `--width` and `--height` the first picture sets the canvas. See
  [First and last frames](fl2va.md) for the prompt they need.
- `--reference FILE` (default none, repeatable): A PNG or JPEG picture the prompt refers to as
  `<Picture 1>`, `<Picture 2>` and so on, in the order of the options. Switches the default DiT to
  the ref2va one. See [Reference pictures](ref2va.md).
- `--steps N` (default 20): Model evaluations. The released checkpoint is guidance-distilled, so
  there is no CFG.
- `--schedule uniform|taomate` (default `uniform`): `taomate` runs the three steps the TaoMate-H3
  LoRA was distilled for, states 0, 16, 33 and 49 of the 50-step schedule, in place of `--steps`.
- `--seed N` (default 0): Seed of the initial noise.
- `--shift-video X`, `--shift-audio X` (default 12, 3): Sigma shifts of the two schedules. The 768p
  Turbo LoRA wants 6 and 3.
- `--attention dense|sol|vsa` (default `vsa` with VSA gates, `dense` otherwise): Sol-Attn switches
  to block-sparse attention. VSA is the sparse attention FastH3 was trained with.
- `--attention-precision bf16|int8-fp8` (default `bf16`): INT8 QK / FP8 PV in the DiT, with FP32
  softmax and accumulation. Changes generated details.
- `--sparse-tau X` (default 1.3): Sol-Attn's routing threshold. Higher is sparser.
- `--sparse-start X` (default 0.2 for Sol-Attn, 0 for VSA): Fraction of the steps that stay dense
  before the sparse attention starts.
- `--vsa-sparsity X` (default 0.9): Fraction of the video tiles VSA leaves out for each query tile.
- `--patch FILE` (default none): A patch for the DiT, such as the FastH3 patch, applied before a
  LoRA.
- `--lora FILE`, `--lora-strength X` (default none, 1.0): A ComfyUI LoRA for the DiT.
- `--lora-mode adapter|merge` (default `adapter`): `adapter` runs the LoRA next to the INT8 weights,
  its down projection on the layer's INT8 input. `merge` adds it into the INT8 weights at load time,
  like ComfyUI, so the steps run as fast as without a LoRA, but updates smaller than the INT8 step
  are lost: most of the TaoMate LoRA, with visibly worse detail.
- `--linear-precision int8|nvfp4` (default `int8`): `nvfp4` runs the video rows of the DiT blocks'
  linear layers in NVFP4, which takes about a quarter off each step and loses some detail.
  Experimental, see [NVFP4 linear layers](nvfp4.md).

## Prompts

H3 follows long, structured prompts well, with the picture, the sound and the music described
separately, one section per line:

```text
integrated_multimodal_description: [Shot 1] Cinematic, wide shot, slow push-in. At golden hour on a rugged coastline, huge waves crash against dark cliffs and throw white spray high into the air. A white lighthouse stands on the headland as its beam begins to sweep.
overall_soundscape: The deep roar of waves breaking on the rocks, gusting wind, and distant seagull calls.
non_diegetic_music: A slow, swelling orchestral string theme.
```

## Few-step models

- [FastH3](fasth3.md): FastVideo's 4-step model with video sparse attention, as a patch on the base
  DiT. `make generate` uses it.
- [lightx2v Turbo LoRA](lightx2v-turbo.md): a LoRA that generates in four steps.
- [TaoMate-H3](taomate.md): a LoRA that generates in three steps with `--schedule taomate`.
