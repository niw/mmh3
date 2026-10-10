//! LBH-123-AI's latent upscaler on the GPU, which doubles the width and height of a video latent
//! for the steps at full size after draft steps. See `mmh3_core::latent_upscaler`.
//!
//! Activations are FP16 and pixel-major. The 3 × 3 × 3 convolutions run in the video encoder's
//! fused kernel, with zero padding and the group statistics of the whole clip, and the pointwise
//! convolutions as GEMMs.

use crate::model::{Error, LinearKind};
use crate::{DeviceBuffer, check};
use mmh3_core::latent_upscaler::{self as upscaler, BlockKind, GROUPS, NORM_EPSILON, load};
use mmh3_core::numeric::{f16_to_f32, f32_to_f16};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_video_encoder_conv3d(
        input: *const c_void,
        frames: c_int,
        height: c_int,
        width: c_int,
        channels: c_int,
        taps: c_int,
        zero_padding: c_int,
        weight: *const c_void,
        columns: c_int,
        bias: *const c_void,
        outputs: c_int,
        accumulate: c_int,
        statistics: *const c_void,
        shared_statistics: c_int,
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
    fn mmh3_latent_upscaler_temporal(
        input: *const c_void,
        frames: c_int,
        pixels: c_int,
        channels: c_int,
        weight: *const c_void,
        kernel: c_int,
        bias: *const c_void,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_latent_upscaler_resize(
        input: *const c_void,
        frames: c_int,
        height: c_int,
        width: c_int,
        channels: c_int,
        output_height: c_int,
        output_width: c_int,
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

/// Channels the fused convolution stages at a time, which its inputs must be a multiple of.
const CHANNEL_CHUNK: usize = 32;
/// Pixels one partial sum of a group norm covers.
const PARTIAL_PIXELS: usize = 256;

fn half_buffer(values: impl IntoIterator<Item = f32>) -> Result<DeviceBuffer, Error> {
    let bytes: Vec<u8> = values
        .into_iter()
        .flat_map(|value| f32_to_f16(value).to_le_bytes())
        .collect();
    Ok(DeviceBuffer::from_bytes(&bytes)?)
}

fn model(error: String) -> Error {
    Error::Model(error)
}

/// A 3 × 3 × 3 convolution, FP16 weights `[outputs, 27 · inputs]` with the columns ordered (kt, ky,
/// kx, input channel) and the inputs padded with zeros to a multiple of 32.
struct Convolution {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
}

impl Convolution {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = load(file, &format!("{name}.weight")).map_err(model)?;
        let bias = load(file, &format!("{name}.bias")).map_err(model)?;
        let [outputs, sources, 3, 3, 3] = weight.shape[..] else {
            return Err(Error::Model(format!("{name} is not a 3 × 3 × 3 Conv3d")));
        };
        let inputs = sources.next_multiple_of(CHANNEL_CHUNK);
        let columns = 27 * inputs;
        let mut reordered = vec![0.0f32; outputs * columns];
        for output in 0..outputs {
            for input in 0..sources {
                for tap in 0..27 {
                    reordered[output * columns + tap * inputs + input] =
                        weight.data[(output * sources + input) * 27 + tap];
                }
            }
        }
        Ok(Convolution {
            weight: half_buffer(reordered)?,
            bias: half_buffer(bias.data)?,
            inputs,
            outputs,
        })
    }
}

struct Norm {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
}

impl Norm {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        Self::new(
            load(file, &format!("{name}.weight")).map_err(model)?.data,
            load(file, &format!("{name}.bias")).map_err(model)?.data,
        )
    }

    fn new(weight: Vec<f32>, bias: Vec<f32>) -> Result<Self, Error> {
        Ok(Norm {
            weight: half_buffer(weight)?,
            bias: half_buffer(bias)?,
        })
    }
}

enum Block {
    Residual {
        norm1: Norm,
        conv1: Convolution,
        /// The second norm with the block's modulation folded into its affine.
        norm2: Norm,
        conv2: Convolution,
    },
    Temporal {
        norm: Norm,
        /// `[channels, kernel]`.
        depthwise: DeviceBuffer,
        depthwise_bias: DeviceBuffer,
        kernel: usize,
        /// `[channels, channels]`.
        pointwise: DeviceBuffer,
        pointwise_bias: DeviceBuffer,
    },
}

impl Block {
    fn load(
        file: &SafeTensors,
        name: &str,
        kind: BlockKind,
        embedding: &[f32],
    ) -> Result<Self, Error> {
        Ok(match kind {
            BlockKind::Residual => {
                let (weight, bias) =
                    upscaler::modulated_norm(file, name, embedding).map_err(model)?;
                Block::Residual {
                    norm1: Norm::load(file, &format!("{name}.in_layers.0"))?,
                    conv1: Convolution::load(file, &format!("{name}.in_layers.2"))?,
                    norm2: Norm::new(weight, bias)?,
                    conv2: Convolution::load(file, &format!("{name}.out_layers.2"))?,
                }
            }
            BlockKind::Temporal { kernel } => Block::Temporal {
                norm: Norm::load(file, &format!("{name}.norm"))?,
                depthwise: half_buffer(
                    load(file, &format!("{name}.dwconv.weight"))
                        .map_err(model)?
                        .data,
                )?,
                depthwise_bias: half_buffer(
                    load(file, &format!("{name}.dwconv.bias"))
                        .map_err(model)?
                        .data,
                )?,
                kernel,
                pointwise: half_buffer(
                    load(file, &format!("{name}.pwconv.weight"))
                        .map_err(model)?
                        .data,
                )?,
                pointwise_bias: half_buffer(
                    load(file, &format!("{name}.pwconv.bias"))
                        .map_err(model)?
                        .data,
                )?,
            },
        })
    }
}

/// FP16 activations `[frames, height, width, channels]`.
struct Activation {
    buffer: DeviceBuffer,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
}

impl Activation {
    fn new(frames: usize, height: usize, width: usize, channels: usize) -> Result<Self, Error> {
        Ok(Activation {
            buffer: DeviceBuffer::new(frames * height * width * channels * 2)?,
            frames,
            height,
            width,
            channels,
        })
    }

    fn like(&self, channels: usize) -> Result<Self, Error> {
        Self::new(self.frames, self.height, self.width, channels)
    }

    fn pixels(&self) -> usize {
        self.frames * self.height * self.width
    }
}

/// The group statistics of the whole clip, and the partial sums they come from.
struct Statistics {
    partials: DeviceBuffer,
    statistics: DeviceBuffer,
}

pub struct CudaLatentUpscaler {
    input: Convolution,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    output_norm: Norm,
    output: Convolution,
}

impl CudaLatentUpscaler {
    /// Whether this GPU runs the upscaler, which every one mmh3 builds for does.
    pub fn supported() -> bool {
        true
    }

    pub fn load(file: &SafeTensors) -> Result<Self, Error> {
        let embedding = upscaler::embedding(file).map_err(model)?;
        let blocks = |prefix: &str| -> Result<Vec<Block>, Error> {
            upscaler::blocks(file, prefix)
                .map_err(model)?
                .into_iter()
                .enumerate()
                .map(|(index, kind)| {
                    Block::load(file, &format!("{prefix}.{index}"), kind, &embedding)
                })
                .collect()
        };
        let input = Convolution::load(file, "conv_in")?;
        let output = Convolution::load(file, "conv_out")?;
        if output.outputs != upscaler::CHANNELS || input.outputs % GROUPS != 0 {
            return Err(Error::Model(
                "the latent upscaler does not take H3 latents".to_owned(),
            ));
        }
        Ok(CudaLatentUpscaler {
            input,
            in_blocks: blocks("in_blocks")?,
            out_blocks: blocks("out_blocks")?,
            output_norm: Norm::load(file, "norm_out")?,
            output,
        })
    }

    /// The latent `[24, frames, height, width]` at twice its width and height.
    pub fn upscale(&self, latent: &Tensor) -> Result<Tensor, Error> {
        let [channels, frames, height, width] = latent.shape[..] else {
            return Err(Error::Model("the latent is not [24, T, H, W]".to_owned()));
        };
        if channels != upscaler::CHANNELS || height < 2 || width < 2 {
            return Err(Error::Model(format!(
                "the latent upscaler takes 24 channels of at least 2 × 2, not {:?}",
                latent.shape
            )));
        }
        let (output_height, output_width) = (height * upscaler::SCALE, width * upscaler::SCALE);
        let largest = frames * output_height * output_width;
        let statistics = Statistics {
            partials: DeviceBuffer::new(largest.div_ceil(PARTIAL_PIXELS) * GROUPS * 2 * 4)?,
            statistics: DeviceBuffer::new(GROUPS * 2 * 4)?,
        };

        let rows = upscaler::rows(latent, self.input.inputs);
        let input = Activation {
            buffer: half_buffer(rows)?,
            frames,
            height,
            width,
            channels: self.input.inputs,
        };
        let mut hidden = input.like(self.input.outputs)?;
        self.convolve(&self.input, &input, None, &mut hidden, false, &statistics)?;
        drop(input);
        self.blocks(&self.in_blocks, &mut hidden, &statistics)?;

        let mut enlarged = Activation::new(frames, output_height, output_width, hidden.channels)?;
        // SAFETY: the input holds `frames × height × width × channels` values and the output the
        // same frames at the doubled size.
        check(unsafe {
            mmh3_latent_upscaler_resize(
                hidden.buffer.pointer(),
                frames as c_int,
                height as c_int,
                width as c_int,
                hidden.channels as c_int,
                output_height as c_int,
                output_width as c_int,
                enlarged.buffer.pointer(),
                ptr::null_mut(),
            )
        })?;
        drop(hidden);
        self.blocks(&self.out_blocks, &mut enlarged, &statistics)?;

        let mut output = enlarged.like(self.output.outputs)?;
        self.convolve(
            &self.output,
            &enlarged,
            Some(&self.output_norm),
            &mut output,
            false,
            &statistics,
        )?;
        let mut bytes = vec![0u8; output.buffer.bytes()];
        output.buffer.copy_to_host(&mut bytes)?;
        let rows: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| f16_to_f32(u16::from_le_bytes(pair)))
            .collect();
        Ok(upscaler::latent(&rows, frames, output_height, output_width))
    }

    fn blocks(
        &self,
        blocks: &[Block],
        hidden: &mut Activation,
        statistics: &Statistics,
    ) -> Result<(), Error> {
        let mut scratch = hidden.like(hidden.channels)?;
        let mut second = None;
        for block in blocks {
            match block {
                Block::Residual {
                    norm1,
                    conv1,
                    norm2,
                    conv2,
                } => {
                    self.convolve(conv1, hidden, Some(norm1), &mut scratch, false, statistics)?;
                    self.convolve(conv2, &scratch, Some(norm2), hidden, true, statistics)?;
                }
                Block::Temporal {
                    norm,
                    depthwise,
                    depthwise_bias,
                    kernel,
                    pointwise,
                    pointwise_bias,
                } => {
                    let temporal = match &mut second {
                        Some(buffer) => buffer,
                        None => second.insert(hidden.like(hidden.channels)?),
                    };
                    // SAFETY: the input, the scratch and the second buffer hold the same values,
                    // and the statistics were sized for the largest activation.
                    check(unsafe {
                        mmh3_video_encoder_group_norm_silu(
                            hidden.buffer.pointer(),
                            1,
                            hidden.pixels() as c_int,
                            hidden.channels as c_int,
                            norm.weight.pointer(),
                            norm.bias.pointer(),
                            NORM_EPSILON,
                            statistics.partials.pointer(),
                            statistics.statistics.pointer(),
                            scratch.buffer.pointer(),
                            ptr::null_mut(),
                        )
                    })?;
                    check(unsafe {
                        mmh3_latent_upscaler_temporal(
                            scratch.buffer.pointer(),
                            hidden.frames as c_int,
                            (hidden.height * hidden.width) as c_int,
                            hidden.channels as c_int,
                            depthwise.pointer(),
                            *kernel as c_int,
                            depthwise_bias.pointer(),
                            temporal.buffer.pointer(),
                            ptr::null_mut(),
                        )
                    })?;
                    // The pointwise convolution adds onto the block's input.
                    check(unsafe {
                        mmh3_cublaslt_matmul(
                            LinearKind::F16 as c_int,
                            temporal.buffer.pointer(),
                            pointwise.pointer(),
                            pointwise_bias.pointer(),
                            hidden.buffer.pointer(),
                            hidden.pixels() as i64,
                            hidden.channels as i64,
                            hidden.channels as i64,
                            1.0,
                            1.0,
                            ptr::null_mut(),
                        )
                    })?;
                }
            }
        }
        Ok(())
    }

    /// The convolution of `input`, normalized first when `norm` is given, into `output`, or added
    /// onto it with `accumulate`.
    fn convolve(
        &self,
        convolution: &Convolution,
        input: &Activation,
        norm: Option<&Norm>,
        output: &mut Activation,
        accumulate: bool,
        statistics: &Statistics,
    ) -> Result<(), Error> {
        if input.channels != convolution.inputs || output.channels != convolution.outputs {
            return Err(Error::Model(
                "a latent upscaler convolution does not match its activations".to_owned(),
            ));
        }
        if norm.is_some() {
            // SAFETY: the input holds its pixels of channels and the partials were sized for the
            // largest activation.
            check(unsafe {
                mmh3_video_encoder_group_norm_statistics(
                    input.buffer.pointer(),
                    1,
                    input.pixels() as c_int,
                    input.channels as c_int,
                    NORM_EPSILON,
                    statistics.partials.pointer(),
                    statistics.statistics.pointer(),
                    ptr::null_mut(),
                )
            })?;
        }
        let (norm_weight, norm_bias) = match norm {
            Some(norm) => (
                norm.weight.pointer().cast_const(),
                norm.bias.pointer().cast_const(),
            ),
            None => (ptr::null(), ptr::null()),
        };
        // SAFETY: the input holds its frames and pixels of `convolution.inputs` channels, the
        // weight `outputs × 27 · inputs` and the output the same frames and pixels of `outputs`.
        check(unsafe {
            mmh3_video_encoder_conv3d(
                input.buffer.pointer(),
                input.frames as c_int,
                input.height as c_int,
                input.width as c_int,
                input.channels as c_int,
                3,
                1,
                convolution.weight.pointer(),
                (27 * convolution.inputs) as c_int,
                convolution.bias.pointer(),
                convolution.outputs as c_int,
                c_int::from(accumulate),
                match norm {
                    Some(_) => statistics.statistics.pointer().cast_const(),
                    None => ptr::null(),
                },
                1,
                norm_weight,
                norm_bias,
                output.buffer.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}
