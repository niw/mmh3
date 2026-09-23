# CUDA on Linux

The `cuda` backend runs every part of a generation on an NVIDIA Blackwell GPU, with the kernels in
`crates/mmh3-cuda`. MP4 output uses NVENC for H.264 video and a built-in AAC encoder.

## Requirements

- Linux with an NVIDIA Blackwell GPU (developed on aarch64).
- The CUDA toolkit with `nvcc` and cuBLASLt (developed with CUDA 13.0).
- Rust with edition 2024 support (developed with 1.98).
- The Hugging Face CLI (`hf`) to download the models, for example installed with
  `uv tool install huggingface_hub`.
- A C/C++ toolchain for the CUDA kernels and NVENC adapter.
- Optionally, the libibverbs headers (`libibverbs-dev`) for RDMA between machines. Without them
  mmh3 builds without RDMA and sends everything over the socket. After installing them, run
  `cargo clean -p mmh3-rdma` so that the next build finds them.
- About 55 GB of disk for the models. mmh3 loads one model at a time. The largest is the DiT, with
  21 GB of weights plus activations.

A machine that only hands work out to others needs none of these. See
[Distributed generation](distributed.md).

## Output

- Native MP4 requires the `mp4` feature, H.264 NVENC and a driver supporting NVENC API 12.2. The
  driver is loaded at runtime, so no separately installed Video Codec SDK is required. A GPU
  without NVENC can write WebM or use ffmpeg from a build without `mp4`.
- The optional `webm` feature needs CMake for the bundled Opus encoder, plus `curl`, `tar` and
  `sha256sum`. The VP9 binding downloads a versioned libvpx static library during the build and
  verifies its checksum, which needs network access to GitHub Releases. Its prebuilt libraries
  support Ubuntu 22.04 and 24.04 on x86-64 and aarch64, and no runtime libvpx is needed.
- ffmpeg is optional. Install it only to use the explicit `--ffmpeg` output path. Running the
  output integration tests also needs ffmpeg and ffprobe as independent decoders.

See [Output](output.md) for the formats and [Build](build.md) for the Cargo features.
