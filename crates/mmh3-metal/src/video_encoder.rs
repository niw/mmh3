//! The video VAE's encoder on Metal, for the keyframes of first and last frame generation and the
//! reference pictures and clips of reference to video generation.
//!
//! The encoder is a causal 3D CNN of residual blocks with group norms, SiLU and 2× downsampling,
//! whose group statistics are per frame. Its convolutions are FP16 products on the matrix units
//! that read their taps straight from the activation, with reflect padding in space and two zero
//! frames in front in time, so no im2col columns are written. For a single frame every temporal
//! kernel reduces to its last tap, since the earlier taps read those zero frames, so a
//! `Temporal::Frame` encoder keeps only that tap. Like the reference, it encodes 256-pixel tiles
//! on their own and blends the moments of the overlaps, and a clip is encoded in groups of 17
//! frames that each give five latent frames.
//!
//! The activations are FP16 like the CUDA encoder's, and the tiles of a picture go through the
//! network several at a time: one tile's deepest levels are a few hundred pixels, which leave
//! most of the GPU idle.
use crate::{Buffer, Device, Error, Result};
use mmh3_core::numeric::{f16_to_f32, f32_to_f16};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{
    CHUNK_TOKENS, CLIP_LENGTH, SPATIAL_RATIO, TEMPORAL_RATIO, TOKEN_DROP, blend_encoded_tiles,
    split_tiles,
};

/// The reference encoder's tile geometry, which is not the decoder's: its tiles overlap by at
/// least 64 pixels.
pub const DEFAULT_TILE_SIZE: usize = 256;
pub const DEFAULT_TILE_OVERLAP_MIN: usize = 64;
const GROUPS: usize = 32;
/// Temporal stride of each downsampling level of the released encoder. Their product is the
/// latent's frames per token.
const TIME_STRIDES: [usize; 4] = [1, 2, 2, 1];
/// Channels the products take at a time, which every convolution's inputs are padded to a
/// multiple of: the first convolution's three become sixteen.
const CHANNEL_CHUNK: usize = 16;
/// The most channels a group norm sums, four for each of its threads.
const NORM_MAX_CHANNELS: usize = 1024;
/// Pixels one threadgroup of video_encoder_norm_partials sums, as ops.metal has it.
const NORM_PIXELS: usize = 1024;
const NORM_EPSILON: f32 = 1e-6;
/// Outputs a threadgroup of mpp_video_convolution takes along each axis.
const PRODUCT_TILE: usize = 128;
/// Pixels of input the tiles of one pass hold together, all frames counted. One tile of 256 is
/// 256 pixels of 1024 channels at the deepest levels, which leaves most of the GPU idle: four at a
/// time take 1344 × 768 from 0.76 to 0.63 seconds on an M6 and keep the widest activation, 128
/// channels at full resolution, at 64 MiB, where eight are no faster and hold 0.8 GB more at
/// their peak. A clip's seventeen frames go through one tile at a time.
const BATCH_PIXELS: usize = 4 * 256 * 256;
/// Latent channels. The encoder writes twice as many moments, the mean and the log variance.
pub const LATENT_CHANNELS: usize = 24;

fn half_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| f32_to_f16(value).to_le_bytes())
        .collect()
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn load(file: &SafeTensors, name: &str) -> Result<Tensor> {
    let info = file
        .get(name)
        .ok_or_else(|| Error::new(format!("missing tensor {name}")))?;
    Tensor::load(file, info).map_err(Error::new)
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

/// A convolution as a product: FP16 weights `[outputs, columns]` with the columns ordered (kt, ky,
/// kx, input channel), the input channels padded with zeros to a multiple of `CHANNEL_CHUNK`, and
/// an FP32 bias.
struct Convolution {
    weight: Buffer,
    bias: Buffer,
    /// Input channels, padded.
    inputs: usize,
    outputs: usize,
    kernel: usize,
    taps: usize,
}

impl Convolution {
    /// Loads a Conv3d `[outputs, inputs, t, k, k]` and keeps its last `taps` temporal taps.
    fn load(device: &Device, file: &SafeTensors, name: &str, taps: usize) -> Result<Self> {
        let weight = load(file, &format!("{name}.weight"))?;
        let bias = load(file, &format!("{name}.bias"))?;
        let [outputs, inputs, kernel_taps, kernel, width] = weight.shape[..] else {
            return Err(Error::new(format!("{name} is not a Conv3d")));
        };
        if kernel != width || (kernel != 1 && kernel != 3) || bias.data.len() != outputs {
            return Err(Error::new(format!("{name}: unsupported kernel {kernel}")));
        }
        // A pointwise shortcut spans one frame however much the rest of the encoder keeps.
        let taps = taps.min(kernel_taps);
        let padded = inputs.next_multiple_of(CHANNEL_CHUNK);
        let columns = taps * kernel * kernel * padded;
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
                            let column = ((tap * kernel + y) * kernel + x) * padded + input;
                            reordered[output * columns + column] = weight.data[source];
                        }
                    }
                }
            }
        }
        let weight = half_bytes(reordered);
        Ok(Convolution {
            weight: device.alloc(weight.len(), Some(&weight))?,
            bias: device.alloc(outputs * 4, Some(&float_bytes(&bias.data)))?,
            inputs: padded,
            outputs,
            kernel,
            taps,
        })
    }
}

/// A group norm's FP32 scale and shift.
struct GroupNorm {
    weight: Buffer,
    bias: Buffer,
    channels: usize,
}

impl GroupNorm {
    fn load(device: &Device, file: &SafeTensors, name: &str) -> Result<Self> {
        let weight = load(file, &format!("{name}.weight"))?.data;
        let bias = load(file, &format!("{name}.bias"))?.data;
        let channels = weight.len();
        if bias.len() != channels
            || !channels.is_multiple_of(GROUPS)
            || channels > NORM_MAX_CHANNELS
        {
            return Err(Error::new(format!("{name}: unsupported group norm")));
        }
        Ok(GroupNorm {
            weight: device.alloc(channels * 4, Some(&float_bytes(&weight)))?,
            bias: device.alloc(channels * 4, Some(&float_bytes(&bias)))?,
            channels,
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

/// FP16 activations `[sequences, frames, height, width, channels]`, a sequence being one tile of
/// one clip.
struct Activation {
    buffer: Buffer,
    sequences: usize,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
}

impl Activation {
    fn new(
        device: &Device,
        sequences: usize,
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let values = sequences * frames * height * width * channels;
        if values > u32::MAX as usize {
            return Err(Error::new(format!(
                "{values} activation values are more than a kernel indexes"
            )));
        }
        Ok(Activation {
            buffer: device.alloc(values * 2, None)?,
            sequences,
            frames,
            height,
            width,
            channels,
        })
    }

    fn values(&self) -> usize {
        self.rows() * self.channels
    }

    /// Pixels of every frame of every sequence, the rows of a product.
    fn rows(&self) -> usize {
        self.sequences * self.frames * self.height * self.width
    }
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
            f16_to_f32(f32_to_f16(value))
        })
    }
}

pub struct MetalVideoEncoder {
    device: Device,
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

impl MetalVideoEncoder {
    /// Loads the encoder of a video VAE checkpoint: `encoder.*`, `quant_conv` and the latent
    /// statistics, which both the FP16 and the INT8 checkpoint hold whole. `temporal` decides
    /// whether it encodes single frames or clips, and `tile_size` and `tile_overlap_min` set the
    /// tile geometry, which is the reference's with `DEFAULT_TILE_SIZE` and
    /// `DEFAULT_TILE_OVERLAP_MIN`.
    ///
    /// The products run on the matrix units, so this needs macOS 26.
    pub fn load(
        file: &SafeTensors,
        temporal: Temporal,
        tile_size: usize,
        tile_overlap_min: usize,
    ) -> Result<Self> {
        let device = Device::shared()?;
        if !device.supports_tensor_ops() {
            return Err(Error::new(
                "the video encoder needs the matrix units of macOS 26".into(),
            ));
        }
        if tile_size == 0
            || !tile_size.is_multiple_of(SPATIAL_RATIO)
            || !tile_overlap_min.is_multiple_of(SPATIAL_RATIO)
            || tile_overlap_min >= tile_size
        {
            return Err(Error::new(format!(
                "encoder tiles of {tile_size} overlapping by {tile_overlap_min} are not multiples \
                 of {SPATIAL_RATIO} with the overlap below the size"
            )));
        }
        let taps = temporal.taps();
        let convolution = |name: &str| Convolution::load(&device, file, name, taps);
        let norm = |name: &str| GroupNorm::load(&device, file, name);
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
                    norm1: norm(&format!("{name}.norm1"))?,
                    conv1: convolution(&format!("{name}.conv1"))?,
                    norm2: norm(&format!("{name}.norm2"))?,
                    conv2: convolution(&format!("{name}.conv2"))?,
                    shortcut: file
                        .get(&format!("{name}.nin_shortcut.weight"))
                        .map(|_| convolution(&format!("{name}.nin_shortcut")))
                        .transpose()?,
                });
            }
            let downsample = file
                .get(&format!("{prefix}.downsample.conv.weight"))
                .map(|_| convolution(&format!("{prefix}.downsample.conv")))
                .transpose()?;
            levels.push(Level { blocks, downsample });
        }
        let encoder = MetalVideoEncoder {
            input: convolution("encoder.conv_in")?,
            levels,
            output_norm: norm("encoder.norm_out")?,
            output: convolution("encoder.conv_out")?,
            quant: convolution("quant_conv")?,
            latents_mean: load(file, "latents_mean")?.data,
            latents_std: load(file, "latents_std")?.data,
            temporal,
            tile_size,
            tile_overlap_min,
            device,
        };
        let downsamples = encoder
            .levels
            .iter()
            .filter(|level| level.downsample.is_some())
            .count();
        if 1 << downsamples != SPATIAL_RATIO
            || downsamples != TIME_STRIDES.len()
            || TIME_STRIDES.iter().product::<usize>() != TEMPORAL_RATIO
            || encoder.input.inputs != CHANNEL_CHUNK
            || encoder.quant.outputs != 2 * LATENT_CHANNELS
            || encoder.latents_mean.len() != LATENT_CHANNELS
            || encoder.latents_std.len() != LATENT_CHANNELS
        {
            return Err(Error::new("unsupported video encoder".to_owned()));
        }
        Ok(encoder)
    }

    /// Encodes a picture `[height, width, 3]` in [0, 1], height and width multiples of 16, into
    /// its latent posterior `[channels, 1, height / 16, width / 16]`.
    pub fn encode_picture(&self, picture: &Tensor) -> Result<Posterior> {
        let [height, width, 3] = picture.shape[..] else {
            return Err(Error::new("a picture is [height, width, 3]".to_owned()));
        };
        if self.temporal != Temporal::Frame {
            return Err(Error::new(
                "this encoder keeps the temporal taps of a clip".to_owned(),
            ));
        }
        self.encode(&picture.data, 1, height, width)
    }

    /// Encodes a clip `[frames, height, width, 3]` in [0, 1], height and width multiples of 16,
    /// into its latent posterior `[channels, latent frames, height / 16, width / 16]`.
    pub fn encode_clip(&self, clip: &Tensor) -> Result<Posterior> {
        let [frames, height, width, 3] = clip.shape[..] else {
            return Err(Error::new(
                "a clip is [frames, height, width, 3]".to_owned(),
            ));
        };
        if self.temporal != Temporal::Clip {
            return Err(Error::new(
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

    /// The posterior of `frames` frames of pixels in [0, 1], a batch of tiles and clips at a time.
    fn encode(
        &self,
        pixels: &[f32],
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Posterior> {
        if !height.is_multiple_of(SPATIAL_RATIO) || !width.is_multiple_of(SPATIAL_RATIO) {
            return Err(Error::new(format!(
                "{width}×{height} is not a multiple of {SPATIAL_RATIO}"
            )));
        }
        if frames == 0 || pixels.len() != frames * height * width * 3 {
            return Err(Error::new(format!(
                "{} values are not {frames} frames of {width}×{height} pixels",
                pixels.len()
            )));
        }
        let canvas = self
            .device
            .alloc(pixels.len() * 4, Some(&float_bytes(pixels)))?;
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
            return Err(Error::new(format!("{frames} frames give no latent frames")));
        }
        // Every tile of every clip is a sequence of its own, each clip's tiles in the order
        // blend_encoded_tiles takes them.
        let mut sequences = Vec::new();
        for clip in 0..clips {
            for &top in &rows.starts {
                for &left in &columns.starts {
                    sequences.push([(clip * clip_frames) as u32, top as u32, left as u32]);
                }
            }
        }
        let batch = (BATCH_PIXELS / (clip_frames * tile_height * tile_width)).max(1);
        let tile_latent_pixels = (tile_height / SPATIAL_RATIO) * (tile_width / SPATIAL_RATIO);
        let tile_values = tokens * tile_latent_pixels * 2 * LATENT_CHANNELS;
        let mut moments = Vec::with_capacity(sequences.len() * tile_values);
        for part in sequences.chunks(batch) {
            let encoded = self.encode_tiles(
                &canvas,
                [frames, height, width],
                clip_frames,
                part,
                [tile_height, tile_width],
            )?;
            if encoded.values() != part.len() * tile_values {
                return Err(Error::new(
                    "the encoder's moments have the wrong shape".into(),
                ));
            }
            moments.extend(
                encoded
                    .buffer
                    .to_bytes()?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&pair| f16_to_f32(u16::from_le_bytes(pair))),
            );
        }

        let tiles_per_clip = rows.starts.len() * columns.starts.len();
        let (latent_height, latent_width) = (height / SPATIAL_RATIO, width / SPATIAL_RATIO);
        let plane = latent_height * latent_width;
        let mut mean = vec![0.0; LATENT_CHANNELS * latent_frames * plane];
        let mut deviation = vec![0.0; LATENT_CHANNELS * latent_frames * plane];
        for clip in 0..clips {
            let tiles =
                &moments[clip * tiles_per_clip * tile_values..][..tiles_per_clip * tile_values];
            for token in 0..tokens {
                let latent_frame = clip * tokens + token;
                if latent_frame >= latent_frames {
                    break;
                }
                let frame: Vec<&[f32]> = tiles
                    .chunks(tile_values)
                    .map(|tile| {
                        let values = tile_latent_pixels * 2 * LATENT_CHANNELS;
                        &tile[token * values..][..values]
                    })
                    .collect();
                let blended = blend_encoded_tiles(&frame, &rows, &columns, 2 * LATENT_CHANNELS);
                for (pixel, values) in blended
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

    /// The moments `[sequences, latent frames, tile latent pixels, 48]` of a batch of tiles.
    /// `canvas` gives the frames, height and width of the whole input, each sequence its first
    /// frame, top and left, and every sequence spans `frames` frames of a tile.
    fn encode_tiles(
        &self,
        canvas: &Buffer,
        [canvas_frames, height, width]: [usize; 3],
        frames: usize,
        sequences: &[[u32; 3]],
        [tile_height, tile_width]: [usize; 2],
    ) -> Result<Activation> {
        let device = &self.device;
        let starts: Vec<u8> = sequences
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let starts = device.alloc(starts.len(), Some(&starts))?;
        let pixels = Activation::new(
            device,
            sequences.len(),
            frames,
            tile_height,
            tile_width,
            CHANNEL_CHUNK,
        )?;
        device.run(
            "video_encoder_tile_input",
            &[canvas, &starts, &pixels.buffer],
            &[
                pixels.values() as u32,
                canvas_frames as u32,
                height as u32,
                width as u32,
                frames as u32,
                tile_height as u32,
                tile_width as u32,
                CHANNEL_CHUNK as u32,
            ],
            pixels.values(),
            false,
        )?;
        let mut hidden = self.convolve(&self.input, &pixels, 1, 1, None)?;
        drop(pixels);
        let mut strides = TIME_STRIDES.into_iter();
        for level in &self.levels {
            for block in &level.blocks {
                hidden = self.resnet_block(block, &hidden)?;
            }
            if let Some(downsample) = &level.downsample {
                let time_stride = strides.next().unwrap_or(1);
                hidden = self.convolve(downsample, &hidden, 2, time_stride, None)?;
            }
        }
        let normalized = self.group_norm_silu(&self.output_norm, &hidden)?;
        drop(hidden);
        let output = self.convolve(&self.output, &normalized, 1, 1, None)?;
        drop(normalized);
        self.convolve(&self.quant, &output, 1, 1, None)
    }

    fn resnet_block(&self, block: &ResnetBlock, input: &Activation) -> Result<Activation> {
        let normalized = self.group_norm_silu(&block.norm1, input)?;
        let hidden = self.convolve(&block.conv1, &normalized, 1, 1, None)?;
        drop(normalized);
        let normalized = self.group_norm_silu(&block.norm2, &hidden)?;
        drop(hidden);
        // The second convolution adds the shortcut as it writes.
        match &block.shortcut {
            Some(shortcut) => {
                let residual = self.convolve(shortcut, input, 1, 1, None)?;
                self.convolve(&block.conv2, &normalized, 1, 1, Some(&residual))
            }
            None => self.convolve(&block.conv2, &normalized, 1, 1, Some(input)),
        }
    }

    /// SiLU(GroupNorm(input)), each frame of each sequence with its own statistics.
    fn group_norm_silu(&self, norm: &GroupNorm, input: &Activation) -> Result<Activation> {
        let device = &self.device;
        if input.channels != norm.channels {
            return Err(Error::new("group norm channels do not match".to_owned()));
        }
        let frames = input.sequences * input.frames;
        let pixels = input.height * input.width;
        let chunks = pixels.div_ceil(NORM_PIXELS);
        let partials = device.alloc(frames * chunks * GROUPS * 2 * 4, None)?;
        device.run(
            "video_encoder_norm_partials",
            &[&input.buffer, &partials],
            &[pixels as u32, input.channels as u32, chunks as u32],
            frames * chunks,
            true,
        )?;
        let statistics = device.alloc(frames * GROUPS * 2 * 4, None)?;
        device.run(
            "video_encoder_norm_statistics",
            &[&input.buffer, &partials, &statistics],
            &[
                (frames * GROUPS) as u32,
                pixels as u32,
                input.channels as u32,
                chunks as u32,
                NORM_EPSILON.to_bits(),
            ],
            frames * GROUPS,
            false,
        )?;
        let output = Activation::new(
            device,
            input.sequences,
            input.frames,
            input.height,
            input.width,
            input.channels,
        )?;
        device.run(
            "video_encoder_norm_silu",
            &[
                &input.buffer,
                &statistics,
                &norm.weight,
                &norm.bias,
                &output.buffer,
            ],
            &[output.values() as u32, pixels as u32, input.channels as u32],
            output.values(),
            false,
        )?;
        Ok(output)
    }

    /// Applies a convolution with `stride` in space and `time_stride` in time: a 3 × 3 kernel
    /// reflect-pads one pixel on each side at stride 1, and one pixel after the input at stride 2,
    /// while the temporal taps read zeros before each sequence. With `residual`, the result adds
    /// that activation.
    fn convolve(
        &self,
        convolution: &Convolution,
        input: &Activation,
        stride: usize,
        time_stride: usize,
        residual: Option<&Activation>,
    ) -> Result<Activation> {
        if input.channels != convolution.inputs {
            return Err(Error::new("convolution inputs do not match".to_owned()));
        }
        // The causal padding covers the taps, so the frames only follow the temporal stride.
        let output = Activation::new(
            &self.device,
            input.sequences,
            (input.frames - 1) / time_stride + 1,
            input.height / stride,
            input.width / stride,
            convolution.outputs,
        )?;
        if let Some(residual) = residual
            && residual.values() != output.values()
        {
            return Err(Error::new(
                "the residual does not match the output".to_owned(),
            ));
        }
        let rows = output.rows();
        self.device.run(
            "mpp_video_convolution",
            &[
                &input.buffer,
                &convolution.weight,
                &convolution.bias,
                residual.map_or(&output.buffer, |residual| &residual.buffer),
                &output.buffer,
            ],
            &[
                rows as u32,
                convolution.outputs as u32,
                input.channels as u32,
                input.frames as u32,
                input.height as u32,
                input.width as u32,
                output.frames as u32,
                output.height as u32,
                output.width as u32,
                convolution.kernel as u32,
                convolution.taps as u32,
                stride as u32,
                time_stride as u32,
                u32::from(residual.is_some()),
            ],
            rows.div_ceil(PRODUCT_TILE) * convolution.outputs.div_ceil(PRODUCT_TILE),
            true,
        )?;
        Ok(output)
    }
}
