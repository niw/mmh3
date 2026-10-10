# Documentation

Setting a machine up:

- [CUDA on Linux and Windows (WSL)](cuda.md): what the CUDA backend needs and what it writes.
- [Metal on macOS](metal.md): what the Metal backend needs, and its limits.
- [Models](models.md): the models directory and what is downloaded into it.
- [Build](build.md): build options and Cargo features.

Running a generation:

- [Usage](usage.md): the options of `mmh3 generate`, and prompts.
- [First and last frames](fl2va.md): generating from pictures.
- [Reference pictures, sounds and clips](ref2va.md): generating from what a prompt refers to.
- [Distributed generation](distributed.md): splitting a generation across machines.
- [Server](server.md): asking for a generation over HTTP, and keeping the models loaded.
- [Output](output.md): MP4, WebM and ffmpeg, and their encoders.

Generating in fewer steps:

- [FastH3](fasth3.md): a 4-step patch, with video sparse attention.
- [lightx2v Turbo LoRA](lightx2v-turbo.md): a 4-step LoRA.
- [TaoMate-H3](taomate.md): a 3-step LoRA.
- [PDMD](pdmd.md): a 2-step LoRA.
- [DMAD](dmad.md): a 4-step LoRA, with the re-noise step rule.

Going faster, and looking underneath:

- [Draft steps](draft-steps.md): the first steps on a half-size canvas.
- [Veda](veda.md): a block-sparse attention whose tiles a predictor picks.
- [NVFP4 linear layers](nvfp4.md): a faster, coarser precision for the DiT.
- [Performance and accuracy](performance.md): times, and the checks against ComfyUI.
- [Development](development.md): the repository layout, tools and tests.
