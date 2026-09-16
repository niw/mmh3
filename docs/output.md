# Output

`mmh3 generate` writes MP4 with native H.264 and AAC, WebM with VP9 and Opus, or hands the encoding
to an installed ffmpeg. H.264 uses NVENC on CUDA and VideoToolbox on Metal. The extension of `--out`
selects the container, and `--ffmpeg` overrides the native output.

## MP4

An `.mp4` output uses H.264 High profile with no B frames and keyframes at most two seconds apart,
plus AAC-LC at 192 kb/s. CUDA builds need the `mp4` feature and use NVENC P4 with constant QP 20.
The `metal` feature includes VideoToolbox output with hardware encoding and quality set to 0.8.
Neither native path needs ffmpeg.

On CUDA, the video VAE retains RGB float32 pixels on the GPU. CUDA converts one frame at a time
directly into a pitched NV12 allocation registered with NVENC. Only compressed video is
read back to the CPU. The allocation is synchronized before submission and reused only
after NVENC has finished reading it. This path makes no assumption about shared system
memory and supports the same design on discrete GPUs such as RTX 5090. GB10 has been
tested. RTX 5090 has not yet been tested on hardware.

On Metal, the VAE restores pixel blocks and blends spatial tiles and temporal chunks on the GPU.
Decoded RGB frames stay in a Metal buffer. A compute kernel converts each frame to BT.709
limited-range NV12 by writing directly into the two planes of an IOSurface-backed CVPixelBuffer.
VideoToolbox reads that same pixel buffer. The adapter waits for GPU conversion before encoding
and for the encoded packet before reusing the input. Only compressed video is copied to the CPU
for native MP4 output. Explicit ffmpeg and WebM output still read back pixels for their host-input
encoders.

Audio is encoded on the CPU using `rusty_aac`, preserving the VAE's 32 kHz sample rate.
Audio is trimmed or silence-padded to the video duration. MP4 edit lists compensate for
AAC priming and end padding. The MP4 header precedes media data for progressive playback.
A decoder exporting raw PCM may still return the last AAC block's padding. The track's
presentation duration excludes it.

The encoder and output file are validated before model loading. Unsupported hardware,
missing drivers, unsupported dimensions or unavailable build features produce an early
error. There is no automatic switch from a requested MP4 to WebM or ffmpeg.

## WebM

For the software path, build with the `webm` feature and use `--out out.webm`, or run
`make generate FEATURES=cuda,webm OUT=out.webm`. Video is encoded by libvpx (VP9 profile 0,
CPU-used 4, fixed quantizer 30 on the 0–63 scale, 8-bit YUV420 with BT.709 limited-range colors).
These encoder settings are currently library defaults, not CLI options. Audio is encoded by libopus
at 192 kb/s after resampling from 32 to 48 kHz. Audio is trimmed or padded with silence to the video
duration. Resampler and Opus delays are compensated. The file includes duration and keyframe seeking
information.

## ffmpeg

To use the ffmpeg CLI instead, append `--ffmpeg`. With no further arguments it uses the
H.264 (libx264, CRF 18) + AAC (192 kb/s) recipe:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 --ffmpeg
```

All arguments after `--ffmpeg` belong to ffmpeg. Put every mmh3 option before it. Custom
arguments replace the default encoding recipe, and are inserted after the generated video
and audio inputs (`0:v` and `1:a`) and before the output filename:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 \
  --ffmpeg -c:v libx264 -crf 20 -preset slow -vf scale=960:-2 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -c:a aac -b:a 160k -shortest
```

For control of input options, mapping, and the entire argument order, use the whole-argument
placeholders `{video}`, `{audio}`, and `{out}`. A template must include `{out}`. Either input
may be omitted. In template mode, automatic input and output arguments are disabled:

```sh
target/release/mmh3 generate --prompt "A rainy street." --out out.mp4 \
  --ffmpeg -i '{video}' -i '{audio}' -map 0:v -map 1:a \
  -c:v libx264 -crf 18 -c:a aac -shortest '{out}'
```

Arguments are passed directly to ffmpeg, without a shell. Quote individual arguments as
needed. Do not combine all ffmpeg arguments into one quoted string. Shell pipelines and
redirection are not interpreted. ffmpeg runs with `-nostdin -y`. Before model loading, mmh3
runs the same arguments on one black, silent frame and writes the result to a temporary file
with the output's extension. Unknown options or codecs, and codecs that the container does
not accept, are reported before generation. Generated Y4M/WAV inputs are temporary and
removed on success or failure. The output is written to a temporary file next to the
destination (with the same extension) and replaces it only after successful encoding.
Additional output paths explicitly supplied in ffmpeg arguments are managed by ffmpeg itself,
and the trial encode writes them as well.

## Encoders

The `mmh3-output` crate owns platform-independent `VideoEncoder` / `AudioEncoder`
interfaces, `Mp4Session`, and the MP4 writer. `VideoEncoder` has an associated native frame
type, so the common layer never casts device handles or requires a CPU YUV buffer.
`mmh3-output-nvenc` owns the NVENC input allocation, CUDA synchronization, driver session,
and conversion of the H.264 bitstream into MP4 samples. The application selects and
prepares output, submits video and audio, then finishes the file. WebM and ffmpeg output
implement the host-input `OutputBackend` interface behind the same lifecycle.

`mmh3-output-videotoolbox` provides a host YUV420 adapter and, with its `metal` feature, a
GPU-buffer adapter using shared CVPixelBuffers. The application selects the GPU adapter for
Metal generation. Both reuse the CPU AAC encoder, MP4 writer, sample timing, edit lists and
destination publication. AudioToolbox is not required.

The encoder dependencies have their own licenses. `rusty_aac` is Apache-2.0, libvpx and
libopus are BSD-3-Clause, `shiguredo_libvpx` is Apache-2.0, the Rust Opus bindings are
MIT/Apache-2.0, and rubato is MIT. The vendored NVENC API header is MIT-licensed and its
notices are retained. The NVIDIA driver is loaded from the installed system. Codec patent
terms remain separate from software licenses. No libx264 or FFmpeg implementation is
linked into the native MP4 path.
See [rusty_aac](https://crates.io/crates/rusty_aac),
[libvpx](https://github.com/webmproject/libvpx),
[the VP9 Rust binding](https://github.com/shiguredo/libvpx-rs), and
[Opus](https://opus-codec.org/license/).

## Tests

Output integration tests use ffmpeg/ffprobe as independent decoders and do not load models.
The portable tests need no GPU. Native output tests require the corresponding hardware encoder:

```sh
cargo test -p mmh3-output --features mp4,webm -- --include-ignored
# Requires an NVIDIA GPU with NVENC and also tests GPU NV12 conversion.
cargo test -p mmh3-output-nvenc --test output -- --ignored
# Requires a Mac with a VideoToolbox H.264 hardware encoder.
cargo test -p mmh3-output-videotoolbox --features metal --test output -- --ignored
```
