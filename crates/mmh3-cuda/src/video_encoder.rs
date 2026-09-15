//! The video VAE's encoder on the GPU, for pictures such as the keyframes of first and last frame
//! generation.
//!
//! The encoder is a causal 3D CNN of residual blocks with group norms, SiLU and 2× downsampling.
//! For a single frame, every temporal kernel reduces to its last tap, since the earlier taps see
//! the zero frames of the causal padding, so the encoder runs 2D convolutions with reflect padding.
//! Like the reference, it encodes 256-pixel tiles of the picture on their own and blends the
//! moments of the overlaps.

use crate::model::{Error, LinearKind};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::numeric::f32_to_f16;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vae::{SPATIAL_RATIO, blend_encoded_tiles, split_tiles};
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_video_encoder_tile_input(
        canvas: *const c_void,
        width: c_int,
        top: c_int,
        left: c_int,
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
        output_height: c_int,
        output_width: c_int,
        columns: c_int,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_video_encoder_group_norm_silu(
        input: *const c_void,
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

const TILE_SIZE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
const GROUPS: usize = 32;
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

/// A convolution as a GEMM: FP16 weights `[outputs, columns]` with the columns ordered (ky, kx,
/// input channel) for 3 × 3 kernels and padded with zeros to a multiple of 8.
struct Convolution {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
    kernel: usize,
    columns: usize,
}

impl Convolution {
    /// Loads a Conv3d `[outputs, inputs, t, k, k]` and keeps its last temporal tap.
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = load(file, &format!("{name}.weight"))?;
        let bias = load(file, &format!("{name}.bias"))?;
        let [outputs, inputs, taps, kernel, width] = weight.shape[..] else {
            return Err(Error::Model(format!("{name} is not a Conv3d")));
        };
        if kernel != width || (kernel != 1 && kernel != 3) {
            return Err(Error::Model(format!("{name}: unsupported kernel {kernel}")));
        }
        let columns = (inputs * kernel * kernel).next_multiple_of(8);
        let mut reordered = vec![0.0f32; outputs * columns];
        for output in 0..outputs {
            for input in 0..inputs {
                for y in 0..kernel {
                    for x in 0..kernel {
                        let source = (((output * inputs + input) * taps + taps - 1) * kernel + y)
                            * kernel
                            + x;
                        reordered[output * columns + (y * kernel + x) * inputs + input] =
                            weight.data[source];
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

/// FP16 activations `[height, width, channels]`.
struct Activation {
    buffer: DeviceBuffer,
    height: usize,
    width: usize,
    channels: usize,
}

impl Activation {
    fn new(height: usize, width: usize, channels: usize) -> Result<Self, CudaError> {
        Ok(Activation {
            buffer: DeviceBuffer::new(height * width * channels * 2)?,
            height,
            width,
            channels,
        })
    }

    fn pixels(&self) -> usize {
        self.height * self.width
    }
}

/// Scratch buffers of one tile.
struct Scratch {
    columns: DeviceBuffer,
    partials: DeviceBuffer,
    statistics: DeviceBuffer,
}

/// The diagonal Gaussian the encoder gives a picture, in the VAE's raw latent units.
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
}

impl CudaVideoEncoder {
    /// Loads the encoder of a video VAE checkpoint: `encoder.*`, `quant_conv` and the latent
    /// statistics.
    pub fn load(file: &SafeTensors) -> Result<Self, Error> {
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
                    conv1: Convolution::load(file, &format!("{name}.conv1"))?,
                    norm2: GroupNorm::load(file, &format!("{name}.norm2"))?,
                    conv2: Convolution::load(file, &format!("{name}.conv2"))?,
                    shortcut: file
                        .get(&format!("{name}.nin_shortcut.weight"))
                        .map(|_| Convolution::load(file, &format!("{name}.nin_shortcut")))
                        .transpose()?,
                });
            }
            let downsample = file
                .get(&format!("{prefix}.downsample.conv.weight"))
                .map(|_| Convolution::load(file, &format!("{prefix}.downsample.conv")))
                .transpose()?;
            levels.push(Level { blocks, downsample });
        }
        let encoder = CudaVideoEncoder {
            input: Convolution::load(file, "encoder.conv_in")?,
            levels,
            output_norm: GroupNorm::load(file, "encoder.norm_out")?,
            output: Convolution::load(file, "encoder.conv_out")?,
            quant: Convolution::load(file, "quant_conv")?,
            latents_mean: load(file, "latents_mean")?.data,
            latents_std: load(file, "latents_std")?.data,
        };
        let downsamples = encoder
            .levels
            .iter()
            .filter(|level| level.downsample.is_some())
            .count();
        if 1 << downsamples != SPATIAL_RATIO
            || encoder.quant.outputs != 2 * LATENT_CHANNELS
            || encoder.latents_mean.len() != LATENT_CHANNELS
        {
            return Err(Error::Model("unsupported video encoder".to_owned()));
        }
        Ok(encoder)
    }

    /// Encodes a picture `[height, width, 3]` in [0, 1], height and width multiples of 16, into
    /// its latent posterior `[channels, 1, height / 16, width / 16]`.
    pub fn encode_picture(&self, picture: &Tensor) -> Result<Posterior, Error> {
        let [height, width, 3] = picture.shape[..] else {
            return Err(Error::Model("a picture is [height, width, 3]".to_owned()));
        };
        if !height.is_multiple_of(SPATIAL_RATIO) || !width.is_multiple_of(SPATIAL_RATIO) {
            return Err(Error::Model(format!(
                "picture {width}×{height} is not a multiple of {SPATIAL_RATIO}"
            )));
        }
        let canvas = DeviceBuffer::from_f32(&picture.data)?;
        let rows = split_tiles(height, TILE_SIZE, TILE_OVERLAP_MIN);
        let columns = split_tiles(width, TILE_SIZE, TILE_OVERLAP_MIN);
        let (tile_height, tile_width) = (rows.length, columns.length);
        let scratch = self.scratch(tile_height, tile_width)?;
        let mut tiles = Vec::new();
        for &top in &rows.starts {
            for &left in &columns.starts {
                tiles.push(self.encode_tile(
                    &canvas,
                    width,
                    top,
                    left,
                    tile_height,
                    tile_width,
                    &scratch,
                )?);
            }
        }
        let moments = blend_encoded_tiles(&tiles, &rows, &columns, 2 * LATENT_CHANNELS);
        let (latent_height, latent_width) = (height / SPATIAL_RATIO, width / SPATIAL_RATIO);
        let plane = latent_height * latent_width;
        let mut mean = vec![0.0; LATENT_CHANNELS * plane];
        let mut deviation = vec![0.0; LATENT_CHANNELS * plane];
        for (pixel, values) in moments
            .as_chunks::<{ 2 * LATENT_CHANNELS }>()
            .0
            .iter()
            .enumerate()
        {
            for channel in 0..LATENT_CHANNELS {
                mean[channel * plane + pixel] = values[channel];
                deviation[channel * plane + pixel] =
                    (0.5 * values[LATENT_CHANNELS + channel].clamp(-30.0, 20.0)).exp();
            }
        }
        let shape = vec![LATENT_CHANNELS, 1, latent_height, latent_width];
        Ok(Posterior {
            mean: Tensor::new(shape.clone(), mean),
            deviation: Tensor::new(shape, deviation),
            latents_mean: self.latents_mean.clone(),
            latents_std: self.latents_std.clone(),
        })
    }

    fn scratch(&self, height: usize, width: usize) -> Result<Scratch, CudaError> {
        // The widest im2col rows come at full resolution or after a downsampling.
        let mut columns = self.input.columns * height * width;
        let (mut level_height, mut level_width) = (height, width);
        for level in &self.levels {
            for block in &level.blocks {
                columns = columns
                    .max(block.conv1.columns * level_height * level_width)
                    .max(block.conv2.columns * level_height * level_width);
            }
            if let Some(downsample) = &level.downsample {
                level_height /= 2;
                level_width /= 2;
                columns = columns.max(downsample.columns * level_height * level_width);
            }
        }
        columns = columns.max(self.output.columns * level_height * level_width);
        Ok(Scratch {
            columns: DeviceBuffer::new(columns * 2)?,
            partials: DeviceBuffer::new((height * width).div_ceil(256) * GROUPS * 2 * 4)?,
            statistics: DeviceBuffer::new(GROUPS * 2 * 4)?,
        })
    }

    /// The moments `[latent pixels, 48]` of one tile.
    #[allow(clippy::too_many_arguments)]
    fn encode_tile(
        &self,
        canvas: &DeviceBuffer,
        width: usize,
        top: usize,
        left: usize,
        tile_height: usize,
        tile_width: usize,
        scratch: &Scratch,
    ) -> Result<Vec<f32>, Error> {
        let pixels = Activation::new(tile_height, tile_width, 3)?;
        // SAFETY: the canvas holds the tile and the activation `tile_height × tile_width × 3`.
        check(unsafe {
            mmh3_video_encoder_tile_input(
                canvas.pointer(),
                width as c_int,
                top as c_int,
                left as c_int,
                tile_height as c_int,
                tile_width as c_int,
                pixels.buffer.pointer(),
                ptr::null_mut(),
            )
        })?;
        let mut hidden = self.convolve(&self.input, &pixels, 1, None, scratch)?;
        for level in &self.levels {
            for block in &level.blocks {
                hidden = self.resnet_block(block, &hidden, scratch)?;
            }
            if let Some(downsample) = &level.downsample {
                hidden = self.convolve(downsample, &hidden, 2, None, scratch)?;
            }
        }
        let normalized = self.group_norm_silu(&self.output_norm, &hidden, scratch)?;
        let output = self.convolve(&self.output, &normalized, 1, None, scratch)?;
        let moments = self.convolve(&self.quant, &output, 1, None, scratch)?;
        let mut bytes = vec![0u8; moments.buffer.bytes()];
        moments.buffer.copy_to_host(&mut bytes)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| mmh3_core::numeric::f16_to_f32(u16::from_le_bytes(pair)))
            .collect())
    }

    fn resnet_block(
        &self,
        block: &ResnetBlock,
        input: &Activation,
        scratch: &Scratch,
    ) -> Result<Activation, Error> {
        let normalized = self.group_norm_silu(&block.norm1, input, scratch)?;
        let hidden = self.convolve(&block.conv1, &normalized, 1, None, scratch)?;
        let normalized = self.group_norm_silu(&block.norm2, &hidden, scratch)?;
        // The second convolution adds onto the shortcut.
        let residual = match &block.shortcut {
            Some(shortcut) => self.convolve(shortcut, input, 1, None, scratch)?,
            None => {
                let copy = Activation::new(input.height, input.width, input.channels)?;
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
        self.convolve(&block.conv2, &normalized, 1, Some(residual), scratch)
    }

    fn group_norm_silu(
        &self,
        norm: &GroupNorm,
        input: &Activation,
        scratch: &Scratch,
    ) -> Result<Activation, Error> {
        let output = Activation::new(input.height, input.width, input.channels)?;
        // SAFETY: input and output hold `pixels × channels` values, and the scratch buffers were
        // sized for the tile's pixels.
        check(unsafe {
            mmh3_video_encoder_group_norm_silu(
                input.buffer.pointer(),
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

    /// Applies a convolution with `stride`: a 3 × 3 kernel reflect-pads one pixel on each side at
    /// stride 1, and one pixel after the input at stride 2. With `accumulate`, the result adds onto
    /// that activation.
    fn convolve(
        &self,
        convolution: &Convolution,
        input: &Activation,
        stride: usize,
        accumulate: Option<Activation>,
        scratch: &Scratch,
    ) -> Result<Activation, Error> {
        if input.channels != convolution.inputs {
            return Err(Error::Model("convolution inputs do not match".to_owned()));
        }
        let (height, width) = (input.height / stride, input.width / stride);
        let beta = if accumulate.is_some() { 1.0 } else { 0.0 };
        let output = match accumulate {
            Some(activation) => activation,
            None => Activation::new(height, width, convolution.outputs)?,
        };
        let rows = height * width;
        let operand = if convolution.kernel == 3 {
            assert!(
                scratch.columns.bytes() >= rows * convolution.columns * 2,
                "the im2col buffer is too small"
            );
            // SAFETY: the input holds its pixels and the scratch buffer the rows, checked above.
            check(unsafe {
                mmh3_video_encoder_im2col(
                    input.buffer.pointer(),
                    input.height as c_int,
                    input.width as c_int,
                    input.channels as c_int,
                    stride as c_int,
                    if stride == 1 { 1 } else { 0 },
                    height as c_int,
                    width as c_int,
                    convolution.columns as c_int,
                    scratch.columns.pointer(),
                    ptr::null_mut(),
                )
            })?;
            scratch.columns.pointer().cast_const()
        } else {
            assert_eq!(
                convolution.columns, input.channels,
                "a 1 × 1 convolution reads the activation rows as they are"
            );
            input.buffer.pointer().cast_const()
        };
        // SAFETY: the operand holds `rows × columns` values, the weight `outputs × columns` and
        // the output `rows × outputs`.
        check(unsafe {
            mmh3_cublaslt_matmul(
                LinearKind::F16 as c_int,
                operand,
                convolution.weight.pointer(),
                convolution.bias.pointer(),
                output.buffer.pointer(),
                rows as i64,
                convolution.outputs as i64,
                convolution.columns as i64,
                1.0,
                beta,
                ptr::null_mut(),
            )
        })?;
        Ok(output)
    }
}
