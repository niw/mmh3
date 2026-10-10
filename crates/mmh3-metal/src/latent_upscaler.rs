//! LBH-123-AI's latent upscaler on Metal, which doubles the width and height of a video latent for
//! the steps at full size after draft steps. See `mmh3_core::latent_upscaler`.
//!
//! Activations are FP16 and pixel-major. The convolutions are the video encoder's FP16 products on
//! the matrix units, with zero padding, and the group norms take the statistics of the whole clip.

use crate::{Buffer, Device, Error, Result};
use mmh3_core::latent_upscaler::{self as upscaler, BlockKind, GROUPS, NORM_EPSILON};
use mmh3_core::numeric::{f16_to_f32, f32_to_f16};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;

/// Channels the products take at a time, which every convolution's inputs are padded to a
/// multiple of.
const CHANNEL_CHUNK: usize = 16;
/// Pixels one threadgroup of video_encoder_norm_partials sums, as ops.metal has it.
const NORM_PIXELS: usize = 1024;
/// Outputs a threadgroup of mpp_video_convolution takes along each axis.
const PRODUCT_TILE: usize = 128;

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
    upscaler::load(file, name).map_err(Error::new)
}

/// A convolution as a product: FP16 weights `[outputs, columns]` with the columns ordered (kt, ky,
/// kx, input channel), the inputs padded with zeros to a multiple of `CHANNEL_CHUNK`, and an FP32
/// bias. `kernel` is 3 for the 3 × 3 × 3 convolutions and 1 for the pointwise ones.
struct Convolution {
    weight: Buffer,
    bias: Buffer,
    inputs: usize,
    outputs: usize,
    kernel: usize,
}

impl Convolution {
    fn load(device: &Device, file: &SafeTensors, name: &str) -> Result<Self> {
        let weight = load(file, &format!("{name}.weight"))?;
        let bias = load(file, &format!("{name}.bias"))?;
        let [outputs, sources, kernel, height, width] = weight.shape[..] else {
            return Err(Error::new(format!("{name} is not a Conv3d")));
        };
        if !(kernel == height && kernel == width && (kernel == 1 || kernel == 3)) {
            return Err(Error::new(format!("{name}: unsupported kernel {kernel}")));
        }
        let taps = kernel * kernel * kernel;
        let inputs = sources.next_multiple_of(CHANNEL_CHUNK);
        let columns = taps * inputs;
        let mut reordered = vec![0.0f32; outputs * columns];
        for output in 0..outputs {
            for input in 0..sources {
                for tap in 0..taps {
                    reordered[output * columns + tap * inputs + input] =
                        weight.data[(output * sources + input) * taps + tap];
                }
            }
        }
        let weight = half_bytes(reordered);
        Ok(Convolution {
            weight: device.alloc(weight.len(), Some(&weight))?,
            bias: device.alloc(outputs * 4, Some(&float_bytes(&bias.data)))?,
            inputs,
            outputs,
            kernel,
        })
    }
}

/// A group norm's FP32 scale and shift.
struct Norm {
    weight: Buffer,
    bias: Buffer,
}

impl Norm {
    fn load(device: &Device, file: &SafeTensors, name: &str) -> Result<Self> {
        Self::new(
            device,
            &load(file, &format!("{name}.weight"))?.data,
            &load(file, &format!("{name}.bias"))?.data,
        )
    }

    fn new(device: &Device, weight: &[f32], bias: &[f32]) -> Result<Self> {
        Ok(Norm {
            weight: device.alloc(weight.len() * 4, Some(&float_bytes(weight)))?,
            bias: device.alloc(bias.len() * 4, Some(&float_bytes(bias)))?,
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
        /// FP32 `[channels, kernel]`.
        depthwise: Buffer,
        depthwise_bias: Buffer,
        kernel: usize,
        pointwise: Convolution,
    },
}

/// FP16 activations `[frames, height, width, channels]`.
struct Activation {
    buffer: Buffer,
    frames: usize,
    height: usize,
    width: usize,
    channels: usize,
}

impl Activation {
    fn new(
        device: &Device,
        frames: usize,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Result<Self> {
        let values = frames * height * width * channels;
        if values > u32::MAX as usize {
            return Err(Error::new(format!(
                "{values} activation values are more than a kernel indexes"
            )));
        }
        Ok(Activation {
            buffer: device.alloc(values * 2, None)?,
            frames,
            height,
            width,
            channels,
        })
    }

    fn rows(&self) -> usize {
        self.frames * self.height * self.width
    }

    fn values(&self) -> usize {
        self.rows() * self.channels
    }
}

pub struct MetalLatentUpscaler {
    device: Device,
    input: Convolution,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    output_norm: Norm,
    output: Convolution,
}

impl MetalLatentUpscaler {
    /// Whether this Mac has the matrix units of macOS 26 the products run on.
    pub fn supported() -> bool {
        Device::shared().is_ok_and(|device| device.supports_tensor_ops())
    }

    pub fn load(file: &SafeTensors) -> Result<Self> {
        let device = Device::shared()?;
        if !device.supports_tensor_ops() {
            return Err(Error::new(
                "the latent upscaler needs the matrix units of macOS 26".into(),
            ));
        }
        let embedding = upscaler::embedding(file).map_err(Error::new)?;
        let blocks = |prefix: &str| -> Result<Vec<Block>> {
            upscaler::blocks(file, prefix)
                .map_err(Error::new)?
                .into_iter()
                .enumerate()
                .map(|(index, kind)| {
                    let name = format!("{prefix}.{index}");
                    Ok(match kind {
                        BlockKind::Residual => {
                            let (weight, bias) = upscaler::modulated_norm(file, &name, &embedding)
                                .map_err(Error::new)?;
                            Block::Residual {
                                norm1: Norm::load(&device, file, &format!("{name}.in_layers.0"))?,
                                conv1: Convolution::load(
                                    &device,
                                    file,
                                    &format!("{name}.in_layers.2"),
                                )?,
                                norm2: Norm::new(&device, &weight, &bias)?,
                                conv2: Convolution::load(
                                    &device,
                                    file,
                                    &format!("{name}.out_layers.2"),
                                )?,
                            }
                        }
                        BlockKind::Temporal { kernel } => {
                            let depthwise = load(file, &format!("{name}.dwconv.weight"))?.data;
                            let depthwise_bias = load(file, &format!("{name}.dwconv.bias"))?.data;
                            Block::Temporal {
                                norm: Norm::load(&device, file, &format!("{name}.norm"))?,
                                depthwise: device
                                    .alloc(depthwise.len() * 4, Some(&float_bytes(&depthwise)))?,
                                depthwise_bias: device.alloc(
                                    depthwise_bias.len() * 4,
                                    Some(&float_bytes(&depthwise_bias)),
                                )?,
                                kernel,
                                pointwise: Convolution::load(
                                    &device,
                                    file,
                                    &format!("{name}.pwconv"),
                                )?,
                            }
                        }
                    })
                })
                .collect()
        };
        let input = Convolution::load(&device, file, "conv_in")?;
        let output = Convolution::load(&device, file, "conv_out")?;
        if output.outputs != upscaler::CHANNELS || !input.outputs.is_multiple_of(GROUPS) {
            return Err(Error::new(
                "the latent upscaler does not take H3 latents".into(),
            ));
        }
        Ok(MetalLatentUpscaler {
            in_blocks: blocks("in_blocks")?,
            out_blocks: blocks("out_blocks")?,
            output_norm: Norm::load(&device, file, "norm_out")?,
            input,
            output,
            device,
        })
    }

    /// The latent `[24, frames, height, width]` at twice its width and height.
    pub fn upscale(&self, latent: &Tensor) -> Result<Tensor> {
        let device = &self.device;
        let [channels, frames, height, width] = latent.shape[..] else {
            return Err(Error::new("the latent is not [24, T, H, W]".into()));
        };
        if channels != upscaler::CHANNELS || height < 2 || width < 2 {
            return Err(Error::new(format!(
                "the latent upscaler takes 24 channels of at least 2 × 2, not {:?}",
                latent.shape
            )));
        }
        let (output_height, output_width) = (height * upscaler::SCALE, width * upscaler::SCALE);
        let rows = half_bytes(upscaler::rows(latent, self.input.inputs));
        let input = Activation {
            buffer: device.alloc(rows.len(), Some(&rows))?,
            frames,
            height,
            width,
            channels: self.input.inputs,
        };
        let mut hidden = self.convolve(&self.input, &input, None)?;
        drop(input);
        hidden = self.blocks(&self.in_blocks, hidden)?;

        let enlarged =
            Activation::new(device, frames, output_height, output_width, hidden.channels)?;
        device.run(
            "latent_upscaler_resize",
            &[&hidden.buffer, &enlarged.buffer],
            &[
                enlarged.values() as u32,
                height as u32,
                width as u32,
                hidden.channels as u32,
                output_height as u32,
                output_width as u32,
            ],
            enlarged.values(),
            false,
        )?;
        drop(hidden);
        let enlarged = self.blocks(&self.out_blocks, enlarged)?;

        let normalized = self.group_norm_silu(&self.output_norm, &enlarged)?;
        let output = self.convolve(&self.output, &normalized, None)?;
        let rows: Vec<f32> = output
            .buffer
            .to_bytes()?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| f16_to_f32(u16::from_le_bytes(pair)))
            .collect();
        Ok(upscaler::latent(
            &rows[..output.values()],
            frames,
            output_height,
            output_width,
        ))
    }

    fn blocks(&self, blocks: &[Block], mut hidden: Activation) -> Result<Activation> {
        for block in blocks {
            hidden = match block {
                Block::Residual {
                    norm1,
                    conv1,
                    norm2,
                    conv2,
                } => {
                    let normalized = self.group_norm_silu(norm1, &hidden)?;
                    let inner = self.convolve(conv1, &normalized, None)?;
                    let normalized = self.group_norm_silu(norm2, &inner)?;
                    self.convolve(conv2, &normalized, Some(&hidden))?
                }
                Block::Temporal {
                    norm,
                    depthwise,
                    depthwise_bias,
                    kernel,
                    pointwise,
                } => {
                    let normalized = self.group_norm_silu(norm, &hidden)?;
                    let temporal = Activation::new(
                        &self.device,
                        hidden.frames,
                        hidden.height,
                        hidden.width,
                        hidden.channels,
                    )?;
                    self.device.run(
                        "latent_upscaler_temporal",
                        &[
                            &normalized.buffer,
                            depthwise,
                            depthwise_bias,
                            &temporal.buffer,
                        ],
                        &[
                            temporal.values() as u32,
                            (hidden.height * hidden.width) as u32,
                            hidden.channels as u32,
                            hidden.frames as u32,
                            *kernel as u32,
                        ],
                        temporal.values(),
                        false,
                    )?;
                    self.convolve(pointwise, &temporal, Some(&hidden))?
                }
            };
        }
        Ok(hidden)
    }

    /// SiLU(GroupNorm(input)) with the statistics of the whole clip.
    fn group_norm_silu(&self, norm: &Norm, input: &Activation) -> Result<Activation> {
        let device = &self.device;
        let pixels = input.rows();
        let chunks = pixels.div_ceil(NORM_PIXELS);
        let partials = device.alloc(chunks * GROUPS * 2 * 4, None)?;
        device.run(
            "video_encoder_norm_partials",
            &[&input.buffer, &partials],
            &[pixels as u32, input.channels as u32, chunks as u32],
            chunks,
            true,
        )?;
        let statistics = device.alloc(GROUPS * 2 * 4, None)?;
        device.run(
            "video_encoder_norm_statistics",
            &[&input.buffer, &partials, &statistics],
            &[
                GROUPS as u32,
                pixels as u32,
                input.channels as u32,
                chunks as u32,
                NORM_EPSILON.to_bits(),
            ],
            GROUPS,
            false,
        )?;
        let output = Activation::new(
            device,
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

    /// The convolution of `input` with zero padding, adding `residual` when given.
    fn convolve(
        &self,
        convolution: &Convolution,
        input: &Activation,
        residual: Option<&Activation>,
    ) -> Result<Activation> {
        if input.channels != convolution.inputs {
            return Err(Error::new(
                "a latent upscaler convolution does not match its input".into(),
            ));
        }
        let output = Activation::new(
            &self.device,
            input.frames,
            input.height,
            input.width,
            convolution.outputs,
        )?;
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
                convolution.kernel as u32,
                1,
                1,
                u32::from(residual.is_some()),
                u32::from(convolution.kernel == 3),
            ],
            rows.div_ceil(PRODUCT_TILE) * convolution.outputs.div_ceil(PRODUCT_TILE),
            true,
        )?;
        Ok(output)
    }
}
