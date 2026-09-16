//! Audio VAE decoding on the GPU: the BigVGAN vocoder at 32 kHz in FP32.
//!
//! Signals are time-major, `[stereo channels, samples, channels]`, and each stereo channel decodes
//! as its own sequence. Every convolution runs as im2col and a cuBLASLt FP32 GEMM.

use crate::model::{Error, LinearKind, cublaslt_linear, host_tensor};
use crate::{DeviceBuffer, check};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_audio_im2col(
        input: *const c_void,
        columns: *mut c_void,
        sequences: c_int,
        length: c_int,
        channels: c_int,
        kernel: c_int,
        dilation: c_int,
        padding: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_conv_transpose_gather(
        products: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        sequences: c_int,
        input_length: c_int,
        output_length: c_int,
        channels: c_int,
        kernel: c_int,
        stride: c_int,
        padding: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_snake_beta(
        input: *const c_void,
        output: *mut c_void,
        parameters: *const c_void,
        up_filter: *const c_void,
        down_filter: *const c_void,
        sequences: c_int,
        length: c_int,
        channels: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_add(
        output: *mut c_void,
        first: *const c_void,
        second: *const c_void,
        count: usize,
        divisor: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_clamp(
        values: *mut c_void,
        count: usize,
        limit: f32,
        stream: *mut c_void,
    ) -> c_int;
}

pub use mmh3_core::audio::SAMPLE_RATE;
/// Upsampling rate and kernel size of each transposed convolution. Their product is 800 samples per
/// latent frame.
const UPSAMPLE: [(usize, usize); 7] = [(5, 9), (5, 9), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
const EDGE_KERNEL: usize = 7;
const SNAKE_EPSILON: f32 = 1e-9;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDecoderConfig {
    pub latent_channels: usize,
    pub latent_features: usize,
    pub initial_channels: usize,
}

impl AudioDecoderConfig {
    pub fn samples_per_latent(&self) -> usize {
        UPSAMPLE.iter().map(|&(rate, _)| rate).product()
    }
}

/// A convolution with its weight laid out for im2col rows: `[outputs, kernel, inputs]`, or
/// `[kernel, outputs, inputs]` for a transposed convolution.
struct Convolution {
    weight: DeviceBuffer,
    bias: Option<DeviceBuffer>,
    outputs: usize,
    inputs: usize,
    kernel: usize,
}

impl Convolution {
    fn load(file: &SafeTensors, name: &str, has_bias: bool) -> Result<Self, Error> {
        let weight = host_tensor(file, &format!("{name}.weight"))?;
        let &[outputs, inputs, kernel] = weight.shape.as_slice() else {
            return Err(Error::Model(format!(
                "{name}.weight has shape {:?}",
                weight.shape
            )));
        };
        let mut reordered = vec![0.0f32; weight.data.len()];
        for output in 0..outputs {
            for input in 0..inputs {
                for tap in 0..kernel {
                    reordered[(output * kernel + tap) * inputs + input] =
                        weight.data[(output * inputs + input) * kernel + tap];
                }
            }
        }
        let bias = if has_bias {
            Some(DeviceBuffer::from_f32(
                &host_tensor(file, &format!("{name}.bias"))?.data,
            )?)
        } else {
            None
        };
        Ok(Convolution {
            weight: DeviceBuffer::from_f32(&reordered)?,
            bias,
            outputs,
            inputs,
            kernel,
        })
    }

    fn load_transposed(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = host_tensor(file, &format!("{name}.weight"))?;
        let &[inputs, outputs, kernel] = weight.shape.as_slice() else {
            return Err(Error::Model(format!(
                "{name}.weight has shape {:?}",
                weight.shape
            )));
        };
        let mut reordered = vec![0.0f32; weight.data.len()];
        for input in 0..inputs {
            for output in 0..outputs {
                for tap in 0..kernel {
                    reordered[(tap * outputs + output) * inputs + input] =
                        weight.data[(input * outputs + output) * kernel + tap];
                }
            }
        }
        let bias = DeviceBuffer::from_f32(&host_tensor(file, &format!("{name}.bias"))?.data)?;
        Ok(Convolution {
            weight: DeviceBuffer::from_f32(&reordered)?,
            bias: Some(bias),
            outputs,
            inputs,
            kernel,
        })
    }
}

/// Anti-aliased SnakeBeta: α and 1 / (β + 1e-9) per channel, and the upsample and downsample
/// filters.
struct Activation {
    parameters: DeviceBuffer,
    up_filter: DeviceBuffer,
    down_filter: DeviceBuffer,
    channels: usize,
}

impl Activation {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let alpha = host_tensor(file, &format!("{name}.act.alpha"))?.data;
        let beta = host_tensor(file, &format!("{name}.act.beta"))?.data;
        let parameters: Vec<f32> = alpha
            .iter()
            .map(|value| value.exp())
            .chain(beta.iter().map(|value| 1.0 / (value.exp() + SNAKE_EPSILON)))
            .collect();
        let filter = |path: &str| -> Result<DeviceBuffer, Error> {
            let filter = host_tensor(file, &format!("{name}.{path}.filter"))?;
            if filter.data.len() != 12 {
                return Err(Error::Model(format!(
                    "{name}.{path}.filter has shape {:?}",
                    filter.shape
                )));
            }
            Ok(DeviceBuffer::from_f32(&filter.data)?)
        };
        Ok(Activation {
            parameters: DeviceBuffer::from_f32(&parameters)?,
            up_filter: filter("upsample")?,
            down_filter: filter("downsample.lowpass")?,
            channels: alpha.len(),
        })
    }
}

/// Three dilated residual units, each an activation and a dilated convolution, then an activation
/// and a plain one.
struct AmpBlock {
    dilated: Vec<Convolution>,
    plain: Vec<Convolution>,
    activations: Vec<Activation>,
}

/// Buffers sized for the longest signal of one decode call.
struct Workspace {
    signal: DeviceBuffer,
    sum: DeviceBuffer,
    running: DeviceBuffer,
    first: DeviceBuffer,
    second: DeviceBuffer,
    columns: DeviceBuffer,
}

pub struct CudaAudioDecoder {
    config: AudioDecoderConfig,
    input_projection: Convolution,
    pre: Convolution,
    upsamples: Vec<Convolution>,
    blocks: Vec<AmpBlock>,
    post_activation: Activation,
    post: Convolution,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
}

impl CudaAudioDecoder {
    /// Uploads the decoder half of an audio VAE checkpoint. `prefix` is prepended to every
    /// checkpoint tensor name.
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self, Error> {
        let input_projection = Convolution::load(file, &format!("{prefix}dec_in_proj"), true)?;
        let pre = Convolution::load(file, &format!("{prefix}decoder.conv_pre"), true)?;
        let config = AudioDecoderConfig {
            latent_channels: input_projection.inputs,
            latent_features: input_projection.outputs,
            initial_channels: pre.outputs,
        };
        let mut upsamples = Vec::new();
        let mut blocks = Vec::new();
        for (stage, &(_, kernel)) in UPSAMPLE.iter().enumerate() {
            let upsample =
                Convolution::load_transposed(file, &format!("{prefix}decoder.ups.{stage}.0"))?;
            if upsample.kernel != kernel
                || upsample.inputs != config.initial_channels >> stage
                || upsample.outputs != upsample.inputs / 2
            {
                return Err(Error::Model(format!(
                    "decoder.ups.{stage} does not match the H3 vocoder"
                )));
            }
            upsamples.push(upsample);
            for (index, &kernel) in RESBLOCK_KERNELS.iter().enumerate() {
                let name = format!(
                    "{prefix}decoder.resblocks.{}",
                    stage * RESBLOCK_KERNELS.len() + index
                );
                let block = AmpBlock {
                    dilated: (0..3)
                        .map(|unit| Convolution::load(file, &format!("{name}.convs1.{unit}"), true))
                        .collect::<Result<_, _>>()?,
                    plain: (0..3)
                        .map(|unit| Convolution::load(file, &format!("{name}.convs2.{unit}"), true))
                        .collect::<Result<_, _>>()?,
                    activations: (0..6)
                        .map(|unit| Activation::load(file, &format!("{name}.activations.{unit}")))
                        .collect::<Result<_, _>>()?,
                };
                if block
                    .dilated
                    .iter()
                    .chain(&block.plain)
                    .any(|convolution| convolution.kernel != kernel)
                {
                    return Err(Error::Model(format!(
                        "{name} does not have kernel size {kernel}"
                    )));
                }
                blocks.push(block);
            }
        }
        Ok(CudaAudioDecoder {
            post_activation: Activation::load(file, &format!("{prefix}decoder.activation_post"))?,
            post: Convolution::load(file, &format!("{prefix}decoder.conv_post"), false)?,
            latents_mean: host_tensor(file, &format!("{prefix}latents_mean"))?.data,
            latents_std: host_tensor(file, &format!("{prefix}latents_std"))?.data,
            config,
            input_projection,
            pre,
            upsamples,
            blocks,
        })
    }

    pub fn config(&self) -> &AudioDecoderConfig {
        &self.config
    }

    /// `output[sequences × length, outputs]` from `input[sequences × length, inputs]`, through
    /// im2col for kernels wider than one sample. Zero padding keeps the length.
    #[allow(clippy::too_many_arguments)]
    fn convolve(
        &self,
        convolution: &Convolution,
        dilation: usize,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        workspace: &Workspace,
        sequences: usize,
        length: usize,
    ) -> Result<(), Error> {
        let rows = sequences * length;
        let features = convolution.kernel * convolution.inputs;
        assert!(
            input.bytes() >= rows * convolution.inputs * 4
                && output.bytes() >= rows * convolution.outputs * 4,
            "signal buffers are too small"
        );
        let columns = if convolution.kernel == 1 {
            input
        } else {
            assert!(
                workspace.columns.bytes() >= rows * features * 4,
                "the im2col buffer is too small"
            );
            // SAFETY: the input holds `rows × inputs` values and the columns
            // `rows × kernel × inputs`, checked above.
            check(unsafe {
                mmh3_audio_im2col(
                    input.pointer(),
                    workspace.columns.pointer(),
                    sequences as c_int,
                    length as c_int,
                    convolution.inputs as c_int,
                    convolution.kernel as c_int,
                    dilation as c_int,
                    (dilation * (convolution.kernel - 1) / 2) as c_int,
                    ptr::null_mut(),
                )
            })?;
            &workspace.columns
        };
        let bias = convolution
            .bias
            .as_ref()
            .map_or(ptr::null(), |bias| bias.pointer().cast_const());
        // SAFETY: the columns hold `rows × features` values, the weight `outputs × features` and
        // the output `rows × outputs`, checked above.
        unsafe {
            cublaslt_linear(
                LinearKind::F32,
                columns.pointer(),
                convolution.weight.pointer(),
                bias,
                output.pointer(),
                rows,
                convolution.outputs,
                features,
            )?
        };
        Ok(())
    }

    fn activate(
        &self,
        activation: &Activation,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        sequences: usize,
        length: usize,
    ) -> Result<(), Error> {
        let count = sequences * length * activation.channels;
        assert!(
            input.bytes() >= count * 4 && output.bytes() >= count * 4,
            "signal buffers are too small"
        );
        // SAFETY: both buffers hold `sequences × length × channels` values, checked above.
        check(unsafe {
            mmh3_audio_snake_beta(
                input.pointer(),
                output.pointer(),
                activation.parameters.pointer(),
                activation.up_filter.pointer(),
                activation.down_filter.pointer(),
                sequences as c_int,
                length as c_int,
                activation.channels as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn add(
        output: &DeviceBuffer,
        first: &DeviceBuffer,
        second: &DeviceBuffer,
        count: usize,
        divisor: f32,
    ) -> Result<(), Error> {
        assert!(
            output.bytes().min(first.bytes()).min(second.bytes()) >= count * 4,
            "signal buffers are too small"
        );
        // SAFETY: every buffer holds `count` values, checked above.
        check(unsafe {
            mmh3_audio_add(
                output.pointer(),
                first.pointer(),
                second.pointer(),
                count,
                divisor,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Upsamples `workspace.signal` from `length` samples of `inputs` channels into
    /// `workspace.sum`.
    fn upsample(
        &self,
        convolution: &Convolution,
        rate: usize,
        workspace: &Workspace,
        sequences: usize,
        length: usize,
    ) -> Result<usize, Error> {
        let padding = (convolution.kernel - rate) / 2;
        let output_length = (length - 1) * rate + convolution.kernel - 2 * padding;
        let product_features = convolution.kernel * convolution.outputs;
        assert!(
            workspace.columns.bytes() >= sequences * length * product_features * 4,
            "the product buffer is too small"
        );
        assert!(
            workspace.sum.bytes() >= sequences * output_length * convolution.outputs * 4,
            "the signal buffer is too small"
        );
        // SAFETY: the signal holds `sequences × length × inputs` values and the products and output
        // the sizes checked above.
        unsafe {
            cublaslt_linear(
                LinearKind::F32,
                workspace.signal.pointer(),
                convolution.weight.pointer(),
                ptr::null(),
                workspace.columns.pointer(),
                sequences * length,
                product_features,
                convolution.inputs,
            )?;
            check(mmh3_audio_conv_transpose_gather(
                workspace.columns.pointer(),
                convolution
                    .bias
                    .as_ref()
                    .map_or(ptr::null(), |bias| bias.pointer().cast_const()),
                workspace.sum.pointer(),
                sequences as c_int,
                length as c_int,
                output_length as c_int,
                convolution.outputs as c_int,
                convolution.kernel as c_int,
                rate as c_int,
                padding as c_int,
                ptr::null_mut(),
            ))?;
        }
        Ok(output_length)
    }

    /// Averages the three AMP blocks of one stage over `workspace.signal` into `workspace.sum`.
    fn amp_blocks(
        &self,
        stage: usize,
        workspace: &Workspace,
        sequences: usize,
        length: usize,
        channels: usize,
    ) -> Result<(), Error> {
        let count = sequences * length * channels;
        let block_count = RESBLOCK_KERNELS.len();
        for (index, block) in self.blocks[stage * block_count..][..block_count]
            .iter()
            .enumerate()
        {
            let running = if index == 0 {
                &workspace.sum
            } else {
                &workspace.running
            };
            for (unit, &dilation) in RESBLOCK_DILATIONS.iter().enumerate() {
                let source = if unit == 0 {
                    &workspace.signal
                } else {
                    running
                };
                self.activate(
                    &block.activations[2 * unit],
                    source,
                    &workspace.first,
                    sequences,
                    length,
                )?;
                self.convolve(
                    &block.dilated[unit],
                    dilation,
                    &workspace.first,
                    &workspace.second,
                    workspace,
                    sequences,
                    length,
                )?;
                self.activate(
                    &block.activations[2 * unit + 1],
                    &workspace.second,
                    &workspace.first,
                    sequences,
                    length,
                )?;
                self.convolve(
                    &block.plain[unit],
                    1,
                    &workspace.first,
                    &workspace.second,
                    workspace,
                    sequences,
                    length,
                )?;
                Self::add(running, &workspace.second, source, count, 1.0)?;
            }
            if index > 0 {
                let divisor = if index + 1 == block_count {
                    block_count as f32
                } else {
                    1.0
                };
                Self::add(&workspace.sum, &workspace.sum, running, count, divisor)?;
            }
        }
        Ok(())
    }

    /// Decodes a normalized latent `[latent channels, stereo channels, frames]` into waveforms
    /// `[stereo channels, frames × 800]` in [−1, 1] at 32 kHz.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor, Error> {
        let config = &self.config;
        let &[channels, sequences, frames] = latent.shape.as_slice() else {
            return Err(Error::Model(format!(
                "latent shape {:?} is not [channels, stereo channels, frames]",
                latent.shape
            )));
        };
        if channels != config.latent_channels || frames == 0 {
            return Err(Error::Model(format!(
                "latent shape {:?} does not match the decoder",
                latent.shape
            )));
        }
        let mut rows = vec![0.0f32; sequences * frames * channels];
        for channel in 0..channels {
            for sequence in 0..sequences {
                for frame in 0..frames {
                    let value = latent.data[(channel * sequences + sequence) * frames + frame];
                    rows[(sequence * frames + frame) * channels + channel] =
                        value * self.latents_std[channel] + self.latents_mean[channel];
                }
            }
        }

        let (mut signal_size, mut columns_size) = (
            frames * config.latent_features.max(config.initial_channels),
            frames * EDGE_KERNEL * config.latent_features,
        );
        let (mut length, mut width) = (frames, config.initial_channels);
        for (&(rate, _), upsample) in UPSAMPLE.iter().zip(&self.upsamples) {
            columns_size = columns_size.max(length * upsample.kernel * upsample.outputs);
            (length, width) = (length * rate, upsample.outputs);
            signal_size = signal_size.max(length * width);
            columns_size =
                columns_size.max(length * RESBLOCK_KERNELS.iter().max().unwrap() * width);
        }
        columns_size = columns_size.max(length * EDGE_KERNEL * width);
        let mut workspace = Workspace {
            signal: DeviceBuffer::new(sequences * signal_size * 4)?,
            sum: DeviceBuffer::new(sequences * signal_size * 4)?,
            running: DeviceBuffer::new(sequences * signal_size * 4)?,
            first: DeviceBuffer::new(sequences * signal_size * 4)?,
            second: DeviceBuffer::new(sequences * signal_size * 4)?,
            columns: DeviceBuffer::new(sequences * columns_size * 4)?,
        };

        let latent_rows = DeviceBuffer::from_f32(&rows)?;
        self.convolve(
            &self.input_projection,
            1,
            &latent_rows,
            &workspace.first,
            &workspace,
            sequences,
            frames,
        )?;
        self.convolve(
            &self.pre,
            1,
            &workspace.first,
            &workspace.signal,
            &workspace,
            sequences,
            frames,
        )?;
        let mut length = frames;
        for (stage, (&(rate, _), upsample)) in UPSAMPLE.iter().zip(&self.upsamples).enumerate() {
            length = self.upsample(upsample, rate, &workspace, sequences, length)?;
            std::mem::swap(&mut workspace.signal, &mut workspace.sum);
            self.amp_blocks(stage, &workspace, sequences, length, upsample.outputs)?;
            std::mem::swap(&mut workspace.signal, &mut workspace.sum);
        }

        self.activate(
            &self.post_activation,
            &workspace.signal,
            &workspace.first,
            sequences,
            length,
        )?;
        let output = DeviceBuffer::new(sequences * length * 4)?;
        self.convolve(
            &self.post,
            1,
            &workspace.first,
            &output,
            &workspace,
            sequences,
            length,
        )?;
        // SAFETY: the output holds `sequences × length` values.
        check(unsafe {
            mmh3_audio_clamp(output.pointer(), sequences * length, 1.0, ptr::null_mut())
        })?;
        Ok(Tensor::new(vec![sequences, length], output.to_f32()?))
    }
}
