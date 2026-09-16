# Documentation

- [Metal on macOS](metal.md): the experimental Swift/Metal backend, build instructions and limits.

- [Usage](usage.md): the options of `mmh3 generate` and prompts.
- [FastH3](fasth3.md), [lightx2v Turbo LoRA](lightx2v-turbo.md) and [TaoMate-H3](taomate.md): the
  few-step models, where to get them and how to run them.
- [First and last frames](fl2va.md): generating from pictures with `--first-frame` and
  `--last-frame`.
- [Reference pictures, sounds and clips](ref2va.md): generating with the ref2va DiT from pictures,
  sounds and clips the prompt refers to, with `--reference`, `--reference-audio` and
  `--reference-video`.
- [NVFP4 linear layers](nvfp4.md): the experimental `--linear-precision nvfp4`, its speed and
  accuracy.
- [Models](models.md): the models directory and what `make download-models` downloads.
- [Output](output.md): MP4, WebM and ffmpeg output and their encoders.
- [Build](build.md): build options and Cargo features.
- [Performance and accuracy](performance.md): times on a DGX Spark and the checks against ComfyUI.
- [Development](development.md): what mmh3 implements, the repository layout, `mmh3-tools`, tests
  and formatting.
