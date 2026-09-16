//! The audio VAE's encoder on the GPU, for the reference sounds of reference to video generation.
//!
//! The DAC encoder is a stack of dilated residual units and strided convolutions that turns 800
//! samples into one latent frame of 2048 features, and the attention head projects those features
//! onto the 32 latent channels. Everything runs in FP32, one stereo channel at a time, and the
//! convolutions run as GEMMs over im2col rows like the decoder's.

use crate::audio_vae::{Convolution, Walk, add, column_values, convolve, convolve_with};
use crate::model::{Error, LinearKind, cublaslt_linear, host_tensor};
use crate::{DeviceBuffer, check};
use mmh3_core::audio::SAMPLES_PER_LATENT;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_audio_encoder_snake(
        input: *const c_void,
        output: *mut c_void,
        alpha: *const c_void,
        length: c_int,
        channels: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_encoder_layer_norm(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        epsilon: f32,
        rows: c_int,
        width: c_int,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_encoder_split_qkv(
        qkv: *const c_void,
        rows: c_int,
        heads: c_int,
        dim: c_int,
        query: *mut c_void,
        key: *mut c_void,
        values: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_encoder_causal_softmax(
        scores: *mut c_void,
        rows: c_int,
        columns: c_int,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_encoder_pool_heads(
        attention: *const c_void,
        rows: c_int,
        heads: c_int,
        dim: c_int,
        outputs: c_int,
        pooled: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_audio_encoder_geglu(
        gate: *const c_void,
        value: *const c_void,
        output: *mut c_void,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
}

/// Samples one encoder block folds into one output sample. Their product is the 800 samples of a
/// latent frame.
const STRIDES: [usize; 5] = [2, 4, 4, 5, 5];
/// Dilations of the three residual units of a block.
const DILATIONS: [usize; 3] = [1, 3, 9];
/// Kernel of a residual unit's dilated convolution.
const RESIDUAL_KERNEL: usize = 7;
/// Heads of the attention head's causal attention.
const HEADS: usize = 8;
/// Hidden features of the GeGLU, twice the latent channels.
const MLP_RATIO: usize = 2;
const NORM_EPSILON: f32 = 1e-5;

/// Snake with one parameter per channel, which the encoder uses as both α and β.
struct Snake {
    alpha: DeviceBuffer,
    channels: usize,
}

impl Snake {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let alpha = host_tensor(file, &format!("{name}.alpha"))?;
        Ok(Snake {
            channels: alpha.data.len(),
            alpha: DeviceBuffer::from_f32(&alpha.data)?,
        })
    }

    fn apply(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        length: usize,
    ) -> Result<(), Error> {
        let count = length * self.channels;
        assert!(
            input.bytes() >= count * 4 && output.bytes() >= count * 4,
            "signal buffers are too small"
        );
        // SAFETY: both buffers hold `length × channels` values, checked above.
        check(unsafe {
            mmh3_audio_encoder_snake(
                input.pointer(),
                output.pointer(),
                self.alpha.pointer(),
                length as c_int,
                self.channels as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}

struct Linear {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
}

impl Linear {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = host_tensor(file, &format!("{name}.weight"))?;
        let bias = host_tensor(file, &format!("{name}.bias"))?.data;
        Linear::new(weight, bias)
    }

    fn new(weight: Tensor, bias: Vec<f32>) -> Result<Self, Error> {
        let &[outputs, inputs] = weight.shape.as_slice() else {
            return Err(Error::Model(format!(
                "a linear layer has shape {:?}",
                weight.shape
            )));
        };
        if bias.len() != outputs {
            return Err(Error::Model(format!(
                "a linear layer of {outputs} outputs has {} bias values",
                bias.len()
            )));
        }
        Ok(Linear {
            weight: DeviceBuffer::from_f32(&weight.data)?,
            bias: DeviceBuffer::from_f32(&bias)?,
            inputs,
            outputs,
        })
    }

    fn apply(&self, input: &DeviceBuffer, output: &DeviceBuffer, rows: usize) -> Result<(), Error> {
        assert!(
            input.bytes() >= rows * self.inputs * 4 && output.bytes() >= rows * self.outputs * 4,
            "the buffers are smaller than their rows"
        );
        // SAFETY: the buffers hold their rows, checked above.
        unsafe {
            cublaslt_linear(
                LinearKind::F32,
                input.pointer(),
                self.weight.pointer(),
                self.bias.pointer(),
                output.pointer(),
                rows,
                self.outputs,
                self.inputs,
            )?
        };
        Ok(())
    }
}

struct LayerNorm {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    width: usize,
}

impl LayerNorm {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = host_tensor(file, &format!("{name}.weight"))?;
        Ok(LayerNorm {
            width: weight.data.len(),
            weight: DeviceBuffer::from_f32(&weight.data)?,
            bias: DeviceBuffer::from_f32(&host_tensor(file, &format!("{name}.bias"))?.data)?,
        })
    }

    fn apply(&self, input: &DeviceBuffer, output: &DeviceBuffer, rows: usize) -> Result<(), Error> {
        assert!(
            input.bytes() >= rows * self.width * 4 && output.bytes() >= rows * self.width * 4,
            "the buffers are smaller than their rows"
        );
        // SAFETY: the buffers hold their rows, checked above.
        check(unsafe {
            mmh3_audio_encoder_layer_norm(
                input.pointer(),
                self.weight.pointer(),
                self.bias.pointer(),
                NORM_EPSILON,
                rows as c_int,
                self.width as c_int,
                output.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}

/// Two Snake activations around a dilated convolution and a pointwise one, added to its input.
struct ResidualUnit {
    activations: [Snake; 2],
    dilated: Convolution,
    pointwise: Convolution,
    dilation: usize,
}

/// Three residual units, then a Snake and a strided convolution that doubles the channels.
struct EncoderBlock {
    units: Vec<ResidualUnit>,
    activation: Snake,
    downsample: Convolution,
    stride: usize,
}

/// The posterior head: a causal attention whose heads are pooled onto the latent channels, a
/// projection of the features themselves, and a GeGLU.
struct AttentionProjection {
    norm1: LayerNorm,
    norm3: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    attention_projection: Linear,
    projection: Linear,
    mlp_norm: LayerNorm,
    gate: Linear,
    value: Linear,
    output: Linear,
}

/// Buffers for the samples of one stereo channel, sized for the widest stage.
struct Workspace {
    signal: DeviceBuffer,
    first: DeviceBuffer,
    second: DeviceBuffer,
    columns: DeviceBuffer,
}

/// Buffers of the posterior head, which works on latent frames.
struct HeadWorkspace {
    normalized: DeviceBuffer,
    qkv: DeviceBuffer,
    query: DeviceBuffer,
    key: DeviceBuffer,
    values: DeviceBuffer,
    scores: DeviceBuffer,
    attention: DeviceBuffer,
    latent: DeviceBuffer,
    first: DeviceBuffer,
    second: DeviceBuffer,
}

pub struct CudaAudioEncoder {
    input: Convolution,
    blocks: Vec<EncoderBlock>,
    post_activation: Snake,
    post: Convolution,
    head: AttentionProjection,
    mean_projection: Convolution,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    features: usize,
    latent_channels: usize,
}

impl CudaAudioEncoder {
    /// Uploads the encoder half of an audio VAE checkpoint. `prefix` is prepended to every
    /// checkpoint tensor name.
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self, Error> {
        let input = Convolution::load(file, &format!("{prefix}encoder.block.0"), true)?;
        let blocks = STRIDES
            .iter()
            .enumerate()
            .map(|(index, &stride)| {
                let name = format!("{prefix}encoder.block.{}", index + 1);
                let units = DILATIONS
                    .iter()
                    .enumerate()
                    .map(|(unit, &dilation)| {
                        let name = format!("{name}.block.{unit}.block");
                        Ok(ResidualUnit {
                            activations: [
                                Snake::load(file, &format!("{name}.0"))?,
                                Snake::load(file, &format!("{name}.2"))?,
                            ],
                            dilated: Convolution::load(file, &format!("{name}.1"), true)?,
                            pointwise: Convolution::load(file, &format!("{name}.3"), true)?,
                            dilation,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Ok(EncoderBlock {
                    units,
                    activation: Snake::load(file, &format!("{name}.block.3"))?,
                    downsample: Convolution::load(file, &format!("{name}.block.4"), true)?,
                    stride,
                })
            })
            .collect::<Result<Vec<EncoderBlock>, Error>>()?;
        let post = Convolution::load(file, &format!("{prefix}encoder.block.7"), true)?;
        let attention = format!("{prefix}pre_block.attn");
        let qkv = host_tensor(file, &format!("{attention}.qkv.weight"))?;
        let bias: Vec<f32> = ["q_bias", "zero_k_bias", "v_bias"]
            .iter()
            .map(|name| host_tensor(file, &format!("{attention}.{name}")))
            .collect::<Result<Vec<_>, Error>>()?
            .iter()
            .flat_map(|tensor| tensor.data.iter().copied())
            .collect();
        let head = AttentionProjection {
            norm1: LayerNorm::load(file, &format!("{prefix}pre_block.norm1"))?,
            norm3: LayerNorm::load(file, &format!("{prefix}pre_block.norm3"))?,
            norm2: LayerNorm::load(file, &format!("{prefix}pre_block.norm2"))?,
            qkv: Linear::new(qkv, bias)?,
            attention_projection: Linear::load(file, &format!("{attention}.proj"))?,
            projection: Linear::load(file, &format!("{prefix}pre_block.proj"))?,
            mlp_norm: LayerNorm::load(file, &format!("{prefix}pre_block.mlp.norm"))?,
            gate: Linear::load(file, &format!("{prefix}pre_block.mlp.w0"))?,
            value: Linear::load(file, &format!("{prefix}pre_block.mlp.w1"))?,
            output: Linear::load(file, &format!("{prefix}pre_block.mlp.w2"))?,
        };
        let encoder = CudaAudioEncoder {
            features: post.outputs,
            latent_channels: head.projection.outputs,
            mean_projection: Convolution::load(file, &format!("{prefix}mean_proj"), true)?,
            latents_mean: host_tensor(file, &format!("{prefix}latents_mean"))?.data,
            latents_std: host_tensor(file, &format!("{prefix}latents_std"))?.data,
            input,
            blocks,
            post_activation: Snake::load(file, &format!("{prefix}encoder.block.6"))?,
            post,
            head,
        };
        encoder.validate()?;
        Ok(encoder)
    }

    /// Checks the shapes the encode path relies on.
    fn validate(&self) -> Result<(), Error> {
        let mut channels = self.input.outputs;
        if self.input.inputs != 1 {
            return Err(Error::Model(
                "the audio encoder does not take a single channel".to_owned(),
            ));
        }
        for block in &self.blocks {
            let half = channels;
            if block.downsample.inputs != half
                || block.downsample.outputs != 2 * half
                || block.downsample.kernel != 2 * block.stride
                || block.units.iter().any(|unit| {
                    unit.dilated.kernel != RESIDUAL_KERNEL
                        || unit.dilated.inputs != half
                        || unit.pointwise.kernel != 1
                })
            {
                return Err(Error::Model(
                    "the audio encoder's blocks do not match the H3 checkpoint".to_owned(),
                ));
            }
            channels *= 2;
        }
        if self.post.inputs != channels
            || !self.features.is_multiple_of(HEADS)
            || self.head.qkv.inputs != self.features
            || self.head.qkv.outputs != 3 * self.features
            || self.head.gate.outputs != MLP_RATIO * self.latent_channels
            || self.mean_projection.kernel != 1
            || self.mean_projection.inputs != self.latent_channels
            || self.latents_mean.len() != self.latent_channels
            || self.latents_std.len() != self.latent_channels
        {
            return Err(Error::Model(
                "the audio encoder's head does not match the H3 checkpoint".to_owned(),
            ));
        }
        Ok(())
    }

    /// Latent frames of `samples` samples, which the encoder pads to a whole number of them.
    pub fn latent_frames(samples: usize) -> usize {
        samples.div_ceil(SAMPLES_PER_LATENT)
    }

    /// Encodes a waveform `[stereo channels, samples]` in [−1, 1] at 32 kHz into the posterior mean
    /// `[latent channels, stereo channels, frames]` in normalized units, the layout the DiT and the
    /// decoder take. The samples are padded with zeros on the right to a whole number of frames.
    pub fn encode(&self, waveform: &Tensor) -> Result<Tensor, Error> {
        let &[sequences, samples] = waveform.shape.as_slice() else {
            return Err(Error::Model(format!(
                "a waveform of shape {:?} is not [stereo channels, samples]",
                waveform.shape
            )));
        };
        if sequences == 0 || samples == 0 {
            return Err(Error::Model("the waveform is empty".to_owned()));
        }
        let frames = Self::latent_frames(samples);
        let padded = frames * SAMPLES_PER_LATENT;
        let workspace = self.workspace(padded)?;
        let head = self.head_workspace(frames)?;
        let mut latents = Vec::with_capacity(sequences * frames * self.latent_channels);
        for sequence in 0..sequences {
            let mut channel = vec![0.0f32; padded];
            channel[..samples].copy_from_slice(&waveform.data[sequence * samples..][..samples]);
            latents.extend(self.encode_channel(&channel, frames, &workspace, &head)?);
        }
        Ok(self.normalize(&latents, sequences, frames))
    }

    /// The features of one stereo channel, `[frames, latent channels]` in the VAE's raw units.
    fn encode_channel(
        &self,
        channel: &[f32],
        frames: usize,
        workspace: &Workspace,
        head: &HeadWorkspace,
    ) -> Result<Vec<f32>, Error> {
        let samples = DeviceBuffer::from_f32(channel)?;
        let mut length = channel.len();
        convolve(
            &self.input,
            1,
            &samples,
            &workspace.signal,
            &workspace.columns,
            1,
            length,
        )?;
        let mut channels = self.input.outputs;
        for block in &self.blocks {
            for unit in &block.units {
                unit.activations[0].apply(&workspace.signal, &workspace.first, length)?;
                convolve(
                    &unit.dilated,
                    unit.dilation,
                    &workspace.first,
                    &workspace.second,
                    &workspace.columns,
                    1,
                    length,
                )?;
                unit.activations[1].apply(&workspace.second, &workspace.first, length)?;
                convolve(
                    &unit.pointwise,
                    1,
                    &workspace.first,
                    &workspace.second,
                    &workspace.columns,
                    1,
                    length,
                )?;
                add(
                    &workspace.signal,
                    &workspace.signal,
                    &workspace.second,
                    length * channels,
                    1.0,
                )?;
            }
            block
                .activation
                .apply(&workspace.signal, &workspace.first, length)?;
            length = convolve_with(
                &block.downsample,
                Walk {
                    dilation: 1,
                    step: block.stride,
                    padding: block.stride.div_ceil(2),
                },
                &workspace.first,
                &workspace.signal,
                &workspace.columns,
                1,
                length,
            )?;
            channels = block.downsample.outputs;
        }
        if length != frames {
            return Err(Error::Model(format!(
                "the encoder gave {length} frames instead of {frames}"
            )));
        }
        self.post_activation
            .apply(&workspace.signal, &workspace.first, frames)?;
        convolve(
            &self.post,
            1,
            &workspace.first,
            &workspace.signal,
            &workspace.columns,
            1,
            frames,
        )?;
        self.project(&workspace.signal, frames, head)?;
        Ok(head.latent.to_f32_range(0, frames * self.latent_channels)?)
    }

    /// The posterior head on `features[frames, features]`, leaving the mean in `head.latent`.
    fn project(
        &self,
        features: &DeviceBuffer,
        frames: usize,
        head: &HeadWorkspace,
    ) -> Result<(), Error> {
        let (channels, head_dim) = (self.latent_channels, self.features / HEADS);
        self.head.norm1.apply(features, &head.normalized, frames)?;
        self.head.qkv.apply(&head.normalized, &head.qkv, frames)?;
        // SAFETY: qkv holds `frames × 3 × features` values and the outputs `frames × features`
        // each, allocated in `head_workspace`.
        check(unsafe {
            mmh3_audio_encoder_split_qkv(
                head.qkv.pointer(),
                frames as c_int,
                HEADS as c_int,
                head_dim as c_int,
                head.query.pointer(),
                head.key.pointer(),
                head.values.pointer(),
                ptr::null_mut(),
            )
        })?;
        let at = |buffer: &DeviceBuffer, head_index: usize| {
            buffer.pointer_at(head_index * frames * head_dim * 4)
        };
        for index in 0..HEADS {
            // SAFETY: each head holds `frames × head dimension` values and the scores
            // `frames × frames`, allocated in `head_workspace`.
            unsafe {
                cublaslt_linear(
                    LinearKind::F32,
                    at(&head.query, index),
                    at(&head.key, index),
                    ptr::null(),
                    head.scores.pointer(),
                    frames,
                    frames,
                    head_dim,
                )?;
                check(mmh3_audio_encoder_causal_softmax(
                    head.scores.pointer(),
                    frames as c_int,
                    frames as c_int,
                    1.0 / (head_dim as f32).sqrt(),
                    ptr::null_mut(),
                ))?;
                cublaslt_linear(
                    LinearKind::F32,
                    head.scores.pointer(),
                    at(&head.values, index),
                    ptr::null(),
                    at(&head.attention, index),
                    frames,
                    head_dim,
                    frames,
                )?;
            }
        }
        // SAFETY: the attention holds `heads × frames × head dimension` values and the pooled
        // rows `frames × latent channels`, allocated in `head_workspace`.
        check(unsafe {
            mmh3_audio_encoder_pool_heads(
                head.attention.pointer(),
                frames as c_int,
                HEADS as c_int,
                head_dim as c_int,
                channels as c_int,
                head.first.pointer(),
                ptr::null_mut(),
            )
        })?;
        self.head
            .attention_projection
            .apply(&head.first, &head.second, frames)?;
        self.head.norm3.apply(features, &head.normalized, frames)?;
        self.head
            .projection
            .apply(&head.normalized, &head.first, frames)?;
        add(
            &head.first,
            &head.first,
            &head.second,
            frames * channels,
            1.0,
        )?;

        self.head.norm2.apply(&head.first, &head.second, frames)?;
        self.head
            .mlp_norm
            .apply(&head.second, &head.latent, frames)?;
        self.head.gate.apply(&head.latent, &head.qkv, frames)?;
        let hidden = MLP_RATIO * channels;
        self.head.value.apply(&head.latent, &head.query, frames)?;
        // SAFETY: both halves hold `frames × hidden` values, allocated in `head_workspace`.
        check(unsafe {
            mmh3_audio_encoder_geglu(
                head.qkv.pointer(),
                head.query.pointer(),
                head.key.pointer(),
                frames * hidden,
                ptr::null_mut(),
            )
        })?;
        self.head.output.apply(&head.key, &head.second, frames)?;
        add(
            &head.second,
            &head.second,
            &head.first,
            frames * channels,
            1.0,
        )?;
        convolve(
            &self.mean_projection,
            1,
            &head.second,
            &head.latent,
            &head.scores,
            1,
            frames,
        )?;
        Ok(())
    }

    /// `[latent channels, stereo channels, frames]` from the raw `[stereo channels, frames,
    /// latent channels]` features, in normalized units.
    fn normalize(&self, latents: &[f32], sequences: usize, frames: usize) -> Tensor {
        let channels = self.latent_channels;
        let mut data = vec![0.0f32; channels * sequences * frames];
        for channel in 0..channels {
            let (mean, deviation) = (self.latents_mean[channel], self.latents_std[channel]);
            for sequence in 0..sequences {
                for frame in 0..frames {
                    let value = latents[(sequence * frames + frame) * channels + channel];
                    data[(channel * sequences + sequence) * frames + frame] =
                        (value - mean) / deviation;
                }
            }
        }
        Tensor::new(vec![channels, sequences, frames], data)
    }

    /// Signal and im2col buffers for the widest stage of `samples` samples. The convolutions build
    /// their columns in slices, so that buffer does not grow with the length of the sound.
    fn workspace(&self, samples: usize) -> Result<Workspace, Error> {
        let (mut length, mut channels) = (samples, self.input.outputs);
        let mut signal_values = length * channels;
        let mut columns = column_values(length, self.input.kernel * self.input.inputs);
        for block in &self.blocks {
            columns = columns
                .max(column_values(length, RESIDUAL_KERNEL * channels))
                .max(column_values(
                    length / block.stride,
                    block.downsample.kernel * channels,
                ));
            length /= block.stride;
            channels = block.downsample.outputs;
            signal_values = signal_values.max(length * channels);
        }
        columns = columns.max(column_values(length, self.post.kernel * channels));
        let buffer = |values: usize| DeviceBuffer::new(values * 4);
        Ok(Workspace {
            signal: buffer(signal_values)?,
            first: buffer(signal_values)?,
            second: buffer(signal_values)?,
            columns: buffer(columns)?,
        })
    }

    /// Buffers of the posterior head for `frames` latent frames.
    fn head_workspace(&self, frames: usize) -> Result<HeadWorkspace, Error> {
        let (channels, head_values) = (self.latent_channels, frames * self.features / HEADS);
        let buffer = |values: usize| DeviceBuffer::new(values * 4);
        Ok(HeadWorkspace {
            normalized: buffer(frames * self.features)?,
            // The GeGLU reuses the qkv, query and key buffers, which are wider than its hidden
            // features.
            qkv: buffer(frames * 3 * self.features)?,
            query: buffer(HEADS * head_values)?,
            key: buffer(HEADS * head_values)?,
            values: buffer(HEADS * head_values)?,
            // The pointwise mean projection reads no im2col rows, so the scores stand in for them.
            scores: buffer(frames * frames)?,
            attention: buffer(HEADS * head_values)?,
            latent: buffer(frames * channels)?,
            first: buffer(frames * channels)?,
            second: buffer(frames * channels)?,
        })
    }
}
