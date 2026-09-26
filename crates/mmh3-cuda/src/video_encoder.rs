//! The video VAE's encoder on the GPU, for the keyframes of first and last frame generation and
//! the reference pictures and clips of reference to video generation.
//!
//! The encoder is a causal 3D CNN of residual blocks with group norms, SiLU and 2× downsampling,
//! whose group statistics are per frame. Its 3 x 3 convolutions at stride one run as one fused
//! pass over the input, and the rest as GEMMs over im2col rows, with reflect padding in space and
//! two zero frames in front in time. For a single frame every temporal kernel reduces to its last
//! tap, since the earlier taps read those zero frames, so a `Temporal::Frame` encoder keeps only
//! that tap and runs 2D convolutions. Like the reference, it encodes 256-pixel tiles on their own
//! and blends the moments of the overlaps, and a clip is encoded in groups of 17 frames that each
//! give five latent frames.

use crate::model::{Error, LinearKind};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::numeric::f32_to_f16;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{
    CHUNK_TOKENS, CLIP_LENGTH, SPATIAL_RATIO, TEMPORAL_RATIO, TOKEN_DROP, blend_encoded_tiles,
    split_tiles,
};
use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::mem::ManuallyDrop;
use std::ptr;

unsafe extern "C" {
    fn mmh3_video_encoder_tile_input(
        canvas: *const c_void,
        canvas_frames: c_int,
        height: c_int,
        width: c_int,
        frame_offset: c_int,
        top: c_int,
        left: c_int,
        frames: c_int,
        tile_height: c_int,
        tile_width: c_int,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_video_encoder_im2col(
        input: *const c_void,
        height: c_int,
        width: c_int,
        channels: c_int,
        stride: c_int,
        pad: c_int,
        taps: c_int,
        time_stride: c_int,
        output_height: c_int,
        output_width: c_int,
        columns: c_int,
        row_offset: usize,
        rows: usize,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_video_encoder_conv3d(
        input: *const c_void,
        frames: c_int,
        height: c_int,
        width: c_int,
        channels: c_int,
        taps: c_int,
        weight: *const c_void,
        columns: c_int,
        bias: *const c_void,
        outputs: c_int,
        accumulate: c_int,
        statistics: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_video_encoder_group_norm_statistics(
        input: *const c_void,
        frames: c_int,
        pixels: c_int,
        channels: c_int,
        epsilon: f32,
        partials: *mut c_void,
        statistics: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_video_encoder_group_norm_silu(
        input: *const c_void,
        frames: c_int,
        pixels: c_int,
        channels: c_int,
        weight: *const c_void,
        bias: *const c_void,
        epsilon: f32,
        partials: *mut c_void,
        statistics: *mut c_void,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_cublaslt_matmul(
        kind: c_int,
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        alpha: f32,
        beta: f32,
        stream: *mut c_void,
    ) -> c_int;
}

pub const DEFAULT_TILE_SIZE: usize = 256;
pub const DEFAULT_TILE_OVERLAP_MIN: usize = 64;
const GROUPS: usize = 32;
/// Temporal stride of each downsampling level of the released encoder. Their product is the
/// latent's frames per token.
const TIME_STRIDES: [usize; 4] = [1, 2, 2, 1];
/// Bytes of im2col columns a convolution builds at a time. A slice that stays inside the L2 cache
/// is handed to the GEMM without a round trip through memory, which the 3 x 3 kernels' twenty-seven
/// columns per channel would otherwise dominate: two thirds of the L2 takes 448 x 256 from 3.4 to
/// 2.3 seconds on a GB10.
fn column_slice_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let cache = crate::device_info(0).map_or(0, |info| info.l2_cache_bytes.max(0) as usize);
        (cache / 3 * 2).clamp(16 << 20, 256 << 20)
    })
}
/// Channels the fused convolution stages at a time, which its inputs must be a multiple of.
const CONV_CHANNEL_CHUNK: usize = 32;
const NORM_EPSILON: f32 = 1e-6;
/// Latent channels. The encoder writes twice as many moments, the mean and the log variance.
pub const LATENT_CHANNELS: usize = 24;

fn half_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| f32_to_f16(value).to_le_bytes())
        .collect()
}

fn load(file: &SafeTensors, name: &str) -> Result<Tensor, Error> {
    let info = file
        .get(name)
        .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
    Tensor::load(file, info).map_err(Error::Model)
}

/// What an encoder keeps of the temporal kernels, which decides what it can encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Temporal {
    /// A single frame, where the causal padding leaves only the last tap.
    Frame,
    /// A clip, with every tap and the two zero frames in front.
    Clip,
}

impl Temporal {
    fn taps(self) -> usize {
        match self {
            Temporal::Frame => 1,
            Temporal::Clip => 3,
        }
    }
}

/// A convolution as a GEMM: FP16 weights `[outputs, columns]` with the columns ordered (kt, ky,
/// kx, input channel) for 3 × 3 kernels and padded with zeros to a multiple of 8.
struct Convolution {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
    kernel: usize,
    taps: usize,
    columns: usize,
}

impl Convolution {
    /// Loads a Conv3d `[outputs, inputs, t, k, k]` and keeps its last `taps` temporal taps.
    fn load(file: &SafeTensors, name: &str, taps: usize) -> Result<Self, Error> {
        let weight = load(file, &format!("{name}.weight"))?;
        let bias = load(file, &format!("{name}.bias"))?;
        let [outputs, inputs, kernel_taps, kernel, width] = weight.shape[..] else {
            return Err(Error::Model(format!("{name} is not a Conv3d")));
        };
        if kernel != width || (kernel != 1 && kernel != 3) {
            return Err(Error::Model(format!("{name}: unsupported kernel {kernel}")));
        }
        // A pointwise shortcut spans one frame however much the rest of the encoder keeps.
        let taps = taps.min(kernel_taps);
        let columns = (inputs * taps * kernel * kernel).next_multiple_of(8);
        let mut reordered = vec![0.0f32; outputs * columns];
        for output in 0..outputs {
            for input in 0..inputs {
                for tap in 0..taps {
                    for y in 0..kernel {
                        for x in 0..kernel {
                            // The taps kept are the last ones, the earlier ones reading the zero
                            // frames of the causal padding.
                            let source_tap = kernel_taps - taps + tap;
                            let source = (((output * inputs + input) * kernel_taps + source_tap)
                                * kernel
                                + y)
                                * kernel
                                + x;
                            let column = ((tap * kernel + y) * kernel + x) * inputs + input;
                            reordered[output * columns + column] = weight.data[source];
                        }
                    }
                }
            }
        }
        Ok(Convolution {
            weight: DeviceBuffer::from_bytes(&half_bytes(reordered))?,
            bias: DeviceBuffer::from_bytes(&half_bytes(bias.data))?,
            inputs,
            outputs,
            kernel,
            taps,
            columns,
        })
    }
}

struct GroupNorm {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
}

impl GroupNorm {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        Ok(GroupNorm {
            weight: DeviceBuffer::from_bytes(&half_bytes(
                load(file, &format!("{name}.weight"))?.data,
            ))?,
            bias: DeviceBuffer::from_bytes(&half_bytes(load(file, &format!("{name}.bias"))?.data))?,
        })
    }
}

struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Convolution,
    norm2: GroupNorm,
    conv2: Convolution,
    shortcut: Option<Convolution>,
}

struct Level {
    blocks: Vec<ResnetBlock>,
    /// A stride-2 convolution after the blocks.
    downsample: Option<Convolution>,
}

/// Device buffers an encode reuses. Allocating one costs a few hundred microseconds on an
/// integrated GPU, and freeing one waits for the work that reads it, so the activations of a tile
/// are handed back here rather than released.
#[derive(Default)]
struct Buffers {
    free: RefCell<Vec<(usize, DeviceBuffer)>>,
}

impl Buffers {
    fn take(&self, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        let mut free = self.free.borrow_mut();
        match free.iter().position(|&(size, _)| size == bytes) {
            Some(index) => Ok(free.swap_remove(index).1),
            None => DeviceBuffer::new(bytes),
        }
    }

    fn give(&self, bytes: usize, buffer: DeviceBuffer) {
        self.free.borrow_mut().push((bytes, buffer));
    }
}

/// FP16 activations `[frames, height, width, channels]`, from a pool of buffers.
struct Activation<'a> {
    buffer: ManuallyDrop<DeviceBuffer>,
    pool: &'a Buffers,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
}

impl Drop for Activation<'_> {
    fn drop(&mut self) {
        // SAFETY: the buffer is taken once, in this the only owner's drop.
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        self.pool.give(self.bytes(), buffer);
    }
}

impl<'a> Activation<'a> {
    fn new(
        pool: &'a Buffers,
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Self, CudaError> {
        let bytes = frames * height * width * channels * 2;
        Ok(Activation {
            buffer: ManuallyDrop::new(pool.take(bytes)?),
            pool,
            frames,
            height,
            width,
            channels,
        })
    }

    fn bytes(&self) -> usize {
        self.frames * self.height * self.width * self.channels * 2
    }

    /// Pixels of one frame.
    fn pixels(&self) -> usize {
        self.height * self.width
    }
}

/// Scratch buffers of one tile, and the pool its activations come from.
struct Scratch {
    columns: DeviceBuffer,
    partials: DeviceBuffer,
    statistics: DeviceBuffer,
    activations: Buffers,
}

/// The diagonal Gaussian the encoder gives a picture or a clip, in the VAE's raw latent units.
pub struct Posterior {
    pub mean: Tensor,
    pub deviation: Tensor,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl Posterior {
    fn normalized(&self, raw: impl Fn(usize) -> f32) -> Tensor {
        let plane: usize = self.mean.shape[1..].iter().product();
        let data = (0..self.mean.data.len())
            .map(|index| {
                let channel = index / plane;
                (raw(index) - self.latents_mean[channel]) / self.latents_std[channel]
            })
            .collect();
        Tensor::new(self.mean.shape.clone(), data)
    }

    /// The normalized latent at the mean, as ComfyUI encodes.
    pub fn mean_latent(&self) -> Tensor {
        self.normalized(|index| self.mean.data[index])
    }

    /// The normalized latent of a sample `mean + deviation · noise`, rounded to FP16 like the
    /// reference pipeline's samples.
    pub fn sampled_latent(&self, noise: &[f32]) -> Tensor {
        assert_eq!(
            noise.len(),
            self.mean.data.len(),
            "one noise value per latent value"
        );
        self.normalized(|index| {
            let value = self.mean.data[index] + self.deviation.data[index] * noise[index];
            mmh3_core::numeric::f16_to_f32(f32_to_f16(value))
        })
    }
}

pub struct CudaVideoEncoder {
    input: Convolution,
    levels: Vec<Level>,
    output_norm: GroupNorm,
    output: Convolution,
    quant: Convolution,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    temporal: Temporal,
    tile_size: usize,
    tile_overlap_min: usize,
}

impl CudaVideoEncoder {
    /// Loads the encoder of a video VAE checkpoint: `encoder.*`, `quant_conv` and the latent
    /// statistics. `temporal` decides whether it encodes single frames or clips, and `tile_size`
    /// and `tile_overlap_min` set the tile geometry, which is the reference's with
    /// `DEFAULT_TILE_SIZE` and `DEFAULT_TILE_OVERLAP_MIN`.
    pub fn load(
        file: &SafeTensors,
        temporal: Temporal,
        tile_size: usize,
        tile_overlap_min: usize,
    ) -> Result<Self, Error> {
        let taps = temporal.taps();
        let mut levels = Vec::new();
        for level in 0.. {
            let prefix = format!("encoder.down.{level}");
            if file
                .get(&format!("{prefix}.block.0.conv1.weight"))
                .is_none()
            {
                break;
            }
            let mut blocks = Vec::new();
            for block in 0.. {
                let name = format!("{prefix}.block.{block}");
                if file.get(&format!("{name}.conv1.weight")).is_none() {
                    break;
                }
                blocks.push(ResnetBlock {
                    norm1: GroupNorm::load(file, &format!("{name}.norm1"))?,
                    conv1: Convolution::load(file, &format!("{name}.conv1"), taps)?,
                    norm2: GroupNorm::load(file, &format!("{name}.norm2"))?,
                    conv2: Convolution::load(file, &format!("{name}.conv2"), taps)?,
                    shortcut: file
                        .get(&format!("{name}.nin_shortcut.weight"))
                        .map(|_| Convolution::load(file, &format!("{name}.nin_shortcut"), taps))
                        .transpose()?,
                });
            }
            let downsample = file
                .get(&format!("{prefix}.downsample.conv.weight"))
                .map(|_| Convolution::load(file, &format!("{prefix}.downsample.conv"), taps))
                .transpose()?;
            levels.push(Level { blocks, downsample });
        }
        let encoder = CudaVideoEncoder {
            input: Convolution::load(file, "encoder.conv_in", taps)?,
            levels,
            output_norm: GroupNorm::load(file, "encoder.norm_out")?,
            output: Convolution::load(file, "encoder.conv_out", taps)?,
            quant: Convolution::load(file, "quant_conv", taps)?,
            latents_mean: load(file, "latents_mean")?.data,
            latents_std: load(file, "latents_std")?.data,
            temporal,
            tile_size,
            tile_overlap_min,
        };
        let downsamples = encoder
            .levels
            .iter()
            .filter(|level| level.downsample.is_some())
            .count();
        if 1 << downsamples != SPATIAL_RATIO
            || downsamples != TIME_STRIDES.len()
            || TIME_STRIDES.iter().product::<usize>() != TEMPORAL_RATIO
            || encoder.quant.outputs != 2 * LATENT_CHANNELS
            || encoder.latents_mean.len() != LATENT_CHANNELS
        {
            return Err(Error::Model("unsupported video encoder".to_owned()));
        }
        Ok(encoder)
    }

    /// The temporal stride of each downsampling level, in the order the levels run.
    fn time_strides(&self) -> impl Iterator<Item = usize> {
        TIME_STRIDES.into_iter()
    }

    /// Encodes a picture `[height, width, 3]` in [0, 1], height and width multiples of 16, into
    /// its latent posterior `[channels, 1, height / 16, width / 16]`.
    pub fn encode_picture(&self, picture: &Tensor) -> Result<Posterior, Error> {
        let [height, width, 3] = picture.shape[..] else {
            return Err(Error::Model("a picture is [height, width, 3]".to_owned()));
        };
        if self.temporal != Temporal::Frame {
            return Err(Error::Model(
                "this encoder keeps the temporal taps of a clip".to_owned(),
            ));
        }
        self.encode(&picture.data, 1, height, width)
    }

    /// Encodes a clip `[frames, height, width, 3]` in [0, 1], height and width multiples of 16,
    /// into its latent posterior `[channels, latent frames, height / 16, width / 16]`.
    pub fn encode_clip(&self, clip: &Tensor) -> Result<Posterior, Error> {
        let [frames, height, width, 3] = clip.shape[..] else {
            return Err(Error::Model(
                "a clip is [frames, height, width, 3]".to_owned(),
            ));
        };
        if self.temporal != Temporal::Clip {
            return Err(Error::Model(
                "this encoder keeps only the last temporal tap".to_owned(),
            ));
        }
        self.encode(&clip.data, frames, height, width)
    }

    /// Latent frames a clip of `frames` frames gives: five per group of seventeen, less the three
    /// the reference drops from the end of the run.
    pub fn latent_frames(frames: usize) -> usize {
        frames.div_ceil(CLIP_LENGTH) * CHUNK_TOKENS - TOKEN_DROP
    }

    /// The posterior of `frames` frames of pixels in [0, 1], tile by tile and clip by clip.
    fn encode(
        &self,
        pixels: &[f32],
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Posterior, Error> {
        if !height.is_multiple_of(SPATIAL_RATIO) || !width.is_multiple_of(SPATIAL_RATIO) {
            return Err(Error::Model(format!(
                "{width}×{height} is not a multiple of {SPATIAL_RATIO}"
            )));
        }
        if frames == 0 || pixels.len() != frames * height * width * 3 {
            return Err(Error::Model(format!(
                "{} values are not {frames} frames of {width}×{height} pixels",
                pixels.len()
            )));
        }
        let canvas = DeviceBuffer::from_f32(pixels)?;
        let rows = split_tiles(height, self.tile_size, self.tile_overlap_min);
        let columns = split_tiles(width, self.tile_size, self.tile_overlap_min);
        let (tile_height, tile_width) = (rows.length, columns.length);
        // A single frame is its own clip and keeps its one latent frame. A clip runs in groups of
        // seventeen frames, the last padded by repeating its last frame, and drops the tokens the
        // reference drops from the end.
        let (clip_frames, tokens, dropped) = match self.temporal {
            Temporal::Frame => (1, 1, 0),
            Temporal::Clip => (CLIP_LENGTH, CHUNK_TOKENS, TOKEN_DROP),
        };
        let clips = frames.div_ceil(clip_frames);
        let latent_frames = clips * tokens - dropped;
        if latent_frames == 0 {
            return Err(Error::Model(format!(
                "{frames} frames give no latent frames"
            )));
        }
        let scratch = self.scratch(clip_frames, tile_height, tile_width)?;
        let tile_latent_pixels = (tile_height / SPATIAL_RATIO) * (tile_width / SPATIAL_RATIO);
        let (latent_height, latent_width) = (height / SPATIAL_RATIO, width / SPATIAL_RATIO);
        let plane = latent_height * latent_width;
        let mut mean = vec![0.0; LATENT_CHANNELS * latent_frames * plane];
        let mut deviation = vec![0.0; LATENT_CHANNELS * latent_frames * plane];
        for clip in 0..clips {
            let tiles = rows
                .starts
                .iter()
                .flat_map(|&top| columns.starts.iter().map(move |&left| (top, left)))
                .map(|(top, left)| {
                    self.encode_tile(
                        &canvas,
                        [frames, height, width],
                        [clip * clip_frames, clip_frames],
                        [top, left, tile_height, tile_width],
                        &scratch,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;
            for token in 0..tokens {
                let latent_frame = clip * tokens + token;
                if latent_frame >= latent_frames {
                    break;
                }
                let frame: Vec<&[f32]> = tiles
                    .iter()
                    .map(|tile| {
                        let values = tile_latent_pixels * 2 * LATENT_CHANNELS;
                        &tile[token * values..][..values]
                    })
                    .collect();
                let moments = blend_encoded_tiles(&frame, &rows, &columns, 2 * LATENT_CHANNELS);
                for (pixel, values) in moments
                    .as_chunks::<{ 2 * LATENT_CHANNELS }>()
                    .0
                    .iter()
                    .enumerate()
                {
                    for channel in 0..LATENT_CHANNELS {
                        let index = (channel * latent_frames + latent_frame) * plane + pixel;
                        mean[index] = values[channel];
                        deviation[index] =
                            (0.5 * values[LATENT_CHANNELS + channel].clamp(-30.0, 20.0)).exp();
                    }
                }
            }
        }
        let shape = vec![LATENT_CHANNELS, latent_frames, latent_height, latent_width];
        Ok(Posterior {
            mean: Tensor::new(shape.clone(), mean),
            deviation: Tensor::new(shape, deviation),
            latents_mean: self.latents_mean.clone(),
            latents_std: self.latents_std.clone(),
        })
    }

    fn scratch(&self, frames: usize, height: usize, width: usize) -> Result<Scratch, CudaError> {
        // The widest im2col rows come at full resolution or after a downsampling, and their
        // columns are built in slices, so that buffer does not grow with the frames.
        let slice = |convolution: &Convolution, rows: usize| {
            let slice_rows = (column_slice_bytes() / (convolution.columns * 2)).max(1);
            slice_rows.min(rows) * convolution.columns
        };
        let (mut level_frames, mut level_height, mut level_width) = (frames, height, width);
        let mut columns = slice(&self.input, level_frames * level_height * level_width);
        let mut strides = self.time_strides();
        for level in &self.levels {
            let rows = level_frames * level_height * level_width;
            for block in &level.blocks {
                columns = columns
                    .max(slice(&block.conv1, rows))
                    .max(slice(&block.conv2, rows));
            }
            if let Some(downsample) = &level.downsample {
                let time_stride = strides.next().unwrap_or(1);
                level_frames = (level_frames - 1) / time_stride + 1;
                level_height /= 2;
                level_width /= 2;
                columns = columns.max(slice(downsample, level_frames * level_height * level_width));
            }
        }
        columns = columns.max(slice(
            &self.output,
            level_frames * level_height * level_width,
        ));
        let partials = frames * (height * width).div_ceil(256) * GROUPS * 2;
        Ok(Scratch {
            columns: DeviceBuffer::new(columns * 2)?,
            partials: DeviceBuffer::new(partials * 4)?,
            statistics: DeviceBuffer::new(frames * GROUPS * 2 * 4)?,
            activations: Buffers::default(),
        })
    }

    /// The moments `[latent frames, tile latent pixels, 48]` of one tile of one clip. `canvas`
    /// gives the frames, height and width of the whole input, `clip` its first frame and its
    /// frames, and `tile` the tile's top, left, height and width.
    fn encode_tile(
        &self,
        canvas: &DeviceBuffer,
        [canvas_frames, height, width]: [usize; 3],
        [frame_offset, frames]: [usize; 2],
        [top, left, tile_height, tile_width]: [usize; 4],
        scratch: &Scratch,
    ) -> Result<Vec<f32>, Error> {
        let pixels = Activation::new(&scratch.activations, frames, tile_height, tile_width, 3)?;
        // SAFETY: the canvas holds its frames of the tile and the activation
        // `frames × tile_height × tile_width × 3`.
        check(unsafe {
            mmh3_video_encoder_tile_input(
                canvas.pointer(),
                canvas_frames as c_int,
                height as c_int,
                width as c_int,
                frame_offset as c_int,
                top as c_int,
                left as c_int,
                frames as c_int,
                tile_height as c_int,
                tile_width as c_int,
                pixels.buffer.pointer(),
                ptr::null_mut(),
            )
        })?;
        let mut hidden = self.convolve(&self.input, &pixels, 1, 1, None, None, scratch)?;
        let mut strides = self.time_strides();
        for level in &self.levels {
            for block in &level.blocks {
                hidden = self.resnet_block(block, &hidden, scratch)?;
            }
            if let Some(downsample) = &level.downsample {
                let time_stride = strides.next().unwrap_or(1);
                hidden = self.convolve(downsample, &hidden, 2, time_stride, None, None, scratch)?;
            }
        }
        let output =
            self.normalized_convolve(&self.output_norm, &self.output, &hidden, None, scratch)?;
        let moments = self.convolve(&self.quant, &output, 1, 1, None, None, scratch)?;
        let mut bytes = vec![0u8; moments.buffer.bytes()];
        moments.buffer.copy_to_host(&mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| mmh3_core::numeric::f16_to_f32(u16::from_le_bytes(pair)))
            .collect())
    }

    fn resnet_block<'a>(
        &self,
        block: &ResnetBlock,
        input: &Activation,
        scratch: &'a Scratch,
    ) -> Result<Activation<'a>, Error> {
        let hidden = self.normalized_convolve(&block.norm1, &block.conv1, input, None, scratch)?;
        // The second convolution adds onto the shortcut.
        let residual = match &block.shortcut {
            Some(shortcut) => self.convolve(shortcut, input, 1, 1, None, None, scratch)?,
            None => {
                let copy = Activation::new(
                    &scratch.activations,
                    input.frames,
                    input.height,
                    input.width,
                    input.channels,
                )?;
                // SAFETY: both buffers hold the same number of bytes.
                unsafe {
                    crate::copy_device(
                        copy.buffer.pointer(),
                        input.buffer.pointer(),
                        input.buffer.bytes(),
                    )?
                };
                copy
            }
        };
        self.normalized_convolve(&block.norm2, &block.conv2, &hidden, Some(residual), scratch)
    }

    /// SiLU(GroupNorm(input)) and then the convolution. A 3 x 3 kernel at stride one applies the
    /// norm as it stages its input, so the normalized activation never reaches memory.
    fn normalized_convolve<'a>(
        &self,
        norm: &GroupNorm,
        convolution: &Convolution,
        input: &Activation,
        accumulate: Option<Activation<'a>>,
        scratch: &'a Scratch,
    ) -> Result<Activation<'a>, Error> {
        if self.fuses(convolution, input, 1, 1) && input.channels.is_multiple_of(GROUPS) {
            return self.convolve(convolution, input, 1, 1, accumulate, Some(norm), scratch);
        }
        let normalized = self.group_norm_silu(norm, input, scratch)?;
        self.convolve(convolution, &normalized, 1, 1, accumulate, None, scratch)
    }

    /// Whether the fused convolution takes this kernel and input.
    fn fuses(
        &self,
        convolution: &Convolution,
        input: &Activation,
        stride: usize,
        time_stride: usize,
    ) -> bool {
        convolution.kernel == 3
            && stride == 1
            && time_stride == 1
            && input.channels.is_multiple_of(CONV_CHANNEL_CHUNK)
    }

    fn group_norm_statistics(&self, input: &Activation, scratch: &Scratch) -> Result<(), Error> {
        // SAFETY: the input holds `frames × pixels × channels` values, and the scratch buffers were
        // sized for the frames and pixels of a tile.
        check(unsafe {
            mmh3_video_encoder_group_norm_statistics(
                input.buffer.pointer(),
                input.frames as c_int,
                input.pixels() as c_int,
                input.channels as c_int,
                NORM_EPSILON,
                scratch.partials.pointer(),
                scratch.statistics.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn group_norm_silu<'a>(
        &self,
        norm: &GroupNorm,
        input: &Activation,
        scratch: &'a Scratch,
    ) -> Result<Activation<'a>, Error> {
        let output = Activation::new(
            &scratch.activations,
            input.frames,
            input.height,
            input.width,
            input.channels,
        )?;
        // SAFETY: input and output hold `frames × pixels × channels` values, and the scratch
        // buffers were sized for the frames and pixels of a tile.
        check(unsafe {
            mmh3_video_encoder_group_norm_silu(
                input.buffer.pointer(),
                input.frames as c_int,
                input.pixels() as c_int,
                input.channels as c_int,
                norm.weight.pointer(),
                norm.bias.pointer(),
                NORM_EPSILON,
                scratch.partials.pointer(),
                scratch.statistics.pointer(),
                output.buffer.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(output)
    }

    /// Applies a convolution with `stride` in space and `time_stride` in time: a 3 × 3 kernel
    /// reflect-pads one pixel on each side at stride 1, and one pixel after the input at stride 2,
    /// while the temporal taps read zeros before the clip. With `accumulate`, the result adds onto
    /// that activation.
    #[allow(clippy::too_many_arguments)]
    fn convolve<'a>(
        &self,
        convolution: &Convolution,
        input: &Activation,
        stride: usize,
        time_stride: usize,
        accumulate: Option<Activation<'a>>,
        normalize: Option<&GroupNorm>,
        scratch: &'a Scratch,
    ) -> Result<Activation<'a>, Error> {
        if input.channels != convolution.inputs {
            return Err(Error::Model("convolution inputs do not match".to_owned()));
        }
        let (height, width) = (input.height / stride, input.width / stride);
        // The causal padding covers the taps, so the frames only follow the temporal stride.
        let frames = (input.frames - 1) / time_stride + 1;
        let beta = if accumulate.is_some() { 1.0 } else { 0.0 };
        let output = match accumulate {
            Some(activation) => activation,
            None => Activation::new(
                &scratch.activations,
                frames,
                height,
                width,
                convolution.outputs,
            )?,
        };
        let rows = frames * height * width;
        // A 3 x 3 kernel at stride one runs as one pass over the input, which keeps its columns out
        // of memory. The channels of the first convolution and the strided downsamplings do not fit
        // the kernel, and take the columns.
        if self.fuses(convolution, input, stride, time_stride) {
            let statistics = match normalize {
                Some(_) => {
                    self.group_norm_statistics(input, scratch)?;
                    scratch.statistics.pointer().cast_const()
                }
                None => ptr::null(),
            };
            let (norm_weight, norm_bias) = match normalize {
                Some(norm) => (
                    norm.weight.pointer().cast_const(),
                    norm.bias.pointer().cast_const(),
                ),
                None => (ptr::null(), ptr::null()),
            };
            // SAFETY: the input holds its frames and pixels of channels, the weight `outputs ×
            // columns` and the output the same frames and pixels of `convolution.outputs`. The
            // statistics hold two floats per group of every frame.
            check(unsafe {
                mmh3_video_encoder_conv3d(
                    input.buffer.pointer(),
                    input.frames as c_int,
                    input.height as c_int,
                    input.width as c_int,
                    input.channels as c_int,
                    convolution.taps as c_int,
                    convolution.weight.pointer(),
                    convolution.columns as c_int,
                    convolution.bias.pointer(),
                    convolution.outputs as c_int,
                    c_int::from(beta != 0.0),
                    statistics,
                    norm_weight,
                    norm_bias,
                    output.buffer.pointer(),
                    ptr::null_mut(),
                )
            })?;
            return Ok(output);
        }
        if normalize.is_some() {
            return Err(Error::Model(
                "this convolution cannot normalize its input".to_owned(),
            ));
        }
        let matmul = |operand: *const c_void, first: usize, count: usize| {
            // SAFETY: the operand holds `count × columns` values, the weight `outputs × columns`
            // and the output `count × outputs` rows from `first`.
            check(unsafe {
                mmh3_cublaslt_matmul(
                    LinearKind::F16 as c_int,
                    operand,
                    convolution.weight.pointer(),
                    convolution.bias.pointer(),
                    output.buffer.pointer_at(first * convolution.outputs * 2),
                    count as i64,
                    convolution.outputs as i64,
                    convolution.columns as i64,
                    1.0,
                    beta,
                    ptr::null_mut(),
                )
            })
        };
        if convolution.kernel == 1 {
            assert_eq!(
                convolution.columns, input.channels,
                "a 1 × 1 convolution reads the activation rows as they are"
            );
            matmul(input.buffer.pointer().cast_const(), 0, rows)?;
            return Ok(output);
        }
        // The columns of every row at once would be `taps × 9` times the activation, so they are
        // built for a slice of the rows at a time.
        let slice = (column_slice_bytes() / (convolution.columns * 2)).max(1);
        assert!(
            scratch.columns.bytes() >= slice.min(rows) * convolution.columns * 2,
            "the im2col buffer is too small"
        );
        let mut first = 0;
        while first < rows {
            let count = slice.min(rows - first);
            // SAFETY: the input holds its frames and pixels and the scratch buffer the slice's
            // rows, checked above.
            check(unsafe {
                mmh3_video_encoder_im2col(
                    input.buffer.pointer(),
                    input.height as c_int,
                    input.width as c_int,
                    input.channels as c_int,
                    stride as c_int,
                    if stride == 1 { 1 } else { 0 },
                    convolution.taps as c_int,
                    time_stride as c_int,
                    height as c_int,
                    width as c_int,
                    convolution.columns as c_int,
                    first,
                    count,
                    scratch.columns.pointer(),
                    ptr::null_mut(),
                )
            })?;
            matmul(scratch.columns.pointer().cast_const(), first, count)?;
            first += count;
        }
        Ok(output)
    }
}
