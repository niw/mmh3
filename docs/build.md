# Build

```sh
make
```

`make` selects `FEATURES=metal` on macOS and `FEATURES=cuda,mp4` on other systems. It builds
`mmh3` in release mode. Override the selection with `make FEATURES=metal` or
`make FEATURES=cuda,mp4`. See [Metal](metal.md) for macOS requirements and supported options.

On Linux, the default builds CUDA, native MP4 output and the ffmpeg CLI output:
`cargo build --release --features cuda,mp4 --bin mmh3`. `make FEATURES=cuda,webm` builds native WebM
output instead of MP4, for GPUs without NVENC. It needs the VP9/Opus dependencies and the libvpx
download. `make FEATURES=cuda,mp4,webm` builds both. `CUDA_HOME` points at the CUDA toolkit (default
`/usr/local/cuda`) and `MMH3_CUDA_ARCH` lists the GPU architectures, separated by commas (default
`sm_120f,sm_90a,sm_89`: Blackwell, Hopper and Ada). Without either GPU feature (`cuda` or
`metal`), both commands can build with the CPU-side crates. In that case, only `mmh3-tools inspect`
is available.

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

The `cuda` and `metal` features build generation with the ffmpeg CLI output. Choose one backend.
The `metal` feature also includes native VideoToolbox H.264/AAC MP4 output without ffmpeg.
The `server` feature builds `mmh3 server`, which takes a generation over HTTP. See
[Server](server.md). `make` includes it, and a build that leaves it out has none of the crates
behind it.
The `mp4` and `webm` features add native output formats. `mp4` encodes video with NVENC
and implies `cuda`. No feature is enabled by default. Examples:

```sh
# Metal with native VideoToolbox MP4 and optional ffmpeg output on macOS.
cargo build --release --features metal --bin mmh3
# Native MP4 and ffmpeg, without the VP9/Opus dependencies or the libvpx download.
cargo build --release --features cuda,mp4 --bin mmh3
# Native WebM and ffmpeg, for GPUs without NVENC.
cargo build --release --features cuda,webm --bin mmh3
# ffmpeg only.
cargo build --release --features cuda --bin mmh3
# Native MP4, native WebM and ffmpeg.
cargo build --release --features cuda,mp4,webm --bin mmh3
```
