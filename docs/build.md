# Build

```sh
make
```

`make` builds `mmh3` with CUDA, native MP4 output and the ffmpeg CLI output:
`cargo build --release --features cuda,mp4 --bin mmh3`. `make FEATURES=cuda,webm` builds native WebM
output instead of MP4, for GPUs without NVENC. It needs the VP9/Opus dependencies and the libvpx
download. `make FEATURES=cuda,mp4,webm` builds both. `CUDA_HOME` points at the CUDA toolkit (default
`/usr/local/cuda`) and `MMH3_CUDA_ARCH` sets the GPU architecture (default `sm_120f`). Without
`--features cuda`, both commands can build with the CPU-side crates. In that case, only
`mmh3-tools inspect` is available.

Build the inspection and development tools separately with:

```sh
cargo build --release --features cuda --bin mmh3-tools
```

To run a specific command through Cargo:

```sh
cargo run --release --features cuda,mp4 --bin mmh3 -- \
  generate --prompt "A rainy street." --out out.mp4
cargo run --release --features cuda --bin mmh3-tools -- device
```

`cargo run` defaults to `mmh3` when `--bin` is omitted.

The `cuda` feature builds generation with the ffmpeg CLI output, which every generation build
has. The `mp4` and `webm` features add native output formats. `mp4` encodes video with NVENC
and implies `cuda`. No feature is enabled by default. Examples:

```sh
# Native MP4 and ffmpeg, without the VP9/Opus dependencies or the libvpx download.
cargo build --release --features cuda,mp4 --bin mmh3
# Native WebM and ffmpeg, for GPUs without NVENC.
cargo build --release --features cuda,webm --bin mmh3
# ffmpeg only.
cargo build --release --features cuda --bin mmh3
# Native MP4, native WebM and ffmpeg.
cargo build --release --features cuda,mp4,webm --bin mmh3
```
