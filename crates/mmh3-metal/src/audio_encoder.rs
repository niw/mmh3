//! The audio VAE's encoder, for reference sounds and the soundtracks of clips.
//!
//! The DAC encoder is a stack of dilated residual units and strided convolutions that turns 800
//! samples into one latent frame of 2048 features, and the attention head projects those features
//! onto the 32 latent channels. Everything runs in FP32, one stereo channel at a time, and the
//! convolutions are MPS products over im2col rows like the decoder's.
use crate::{Error, Result, model::Weights, ops::Array};
use mmh3_core::{audio::SAMPLES_PER_LATENT, safetensors::SafeTensors, tensor::Tensor};

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
/// Bytes of im2col rows a convolution builds at a time. The rows of a whole signal are `kernel`
/// times its size, over half a gigabyte for ten seconds at the first level, while a 64 MiB slice
/// still gives the product tens of thousands of rows.
const COLUMN_SLICE_BYTES: usize = 64 << 20;

/// How a convolution walks the signal. A `step` above one shortens the output.
#[derive(Clone, Copy)]
struct Walk {
    dilation: usize,
    step: usize,
    padding: usize,
}

pub struct MetalAudioEncoder {
    weights: Weights,
    /// The qkv projection's bias, which the checkpoint holds in three parts, the keys' all zeros.
    qkv_bias: Array,
    features: usize,
    latent_channels: usize,
}

impl MetalAudioEncoder {
    /// Loads the encoder half of an audio VAE checkpoint. `prefix` is prepended to every
    /// checkpoint tensor name. The decoder's tensors are left out, so that its transposed
    /// convolutions are not repacked as the encoder's are.
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self> {
        let mut weights = Weights::load_selected(file, prefix, |n| {
            n.starts_with("encoder.")
                || n.starts_with("pre_block.")
                || n.starts_with("mean_proj.")
                || n == "latents_mean"
                || n == "latents_std"
        })?;
        weights.prepare_audio_convolutions()?;
        let bias: Vec<f32> = ["q_bias", "zero_k_bias", "v_bias"]
            .iter()
            .map(|name| Ok(weights.host(&format!("pre_block.attn.{name}"))?.data))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let encoder = Self {
            qkv_bias: Array::from_f32(&weights.device, 1, bias.len(), &bias)?,
            features: weights.shape("encoder.block.7.weight")?[0],
            latent_channels: weights.shape("pre_block.proj.weight")?[0],
            weights,
        };
        encoder.validate()?;
        Ok(encoder)
    }

    /// Checks the shapes the encode path relies on.
    fn validate(&self) -> Result<()> {
        let w = &self.weights;
        let convolution = |name: &str| -> Result<[usize; 3]> {
            let shape = w.shape(&format!("{name}.weight"))?;
            shape
                .try_into()
                .map_err(|_| Error::new(format!("{name}.weight is not a convolution")))
        };
        let [mut channels, inputs, _] = convolution("encoder.block.0")?;
        if inputs != 1 {
            return Err(Error::new(
                "the audio encoder does not take a single channel".into(),
            ));
        }
        for (index, &stride) in STRIDES.iter().enumerate() {
            let name = format!("encoder.block.{}", index + 1);
            let units = (0..DILATIONS.len())
                .map(|unit| {
                    let name = format!("{name}.block.{unit}.block");
                    Ok((
                        convolution(&format!("{name}.1"))?,
                        convolution(&format!("{name}.3"))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            if convolution(&format!("{name}.block.4"))? != [2 * channels, channels, 2 * stride]
                || units.iter().any(|&(dilated, pointwise)| {
                    dilated != [channels, channels, RESIDUAL_KERNEL]
                        || pointwise != [channels, channels, 1]
                })
            {
                return Err(Error::new(
                    "the audio encoder's blocks do not match the H3 checkpoint".into(),
                ));
            }
            channels *= 2;
        }
        let (features, latent) = (self.features, self.latent_channels);
        if convolution("encoder.block.7")?[1] != channels
            || !features.is_multiple_of(HEADS)
            || features / HEADS > 256
            || w.shape("pre_block.attn.qkv.weight")? != [3 * features, features]
            || self.qkv_bias.len() != 3 * features
            || w.shape("pre_block.mlp.w0.weight")? != [MLP_RATIO * latent, latent]
            || convolution("mean_proj")? != [latent, latent, 1]
            || w.shape("latents_mean")?.iter().product::<usize>() != latent
            || w.shape("latents_std")?.iter().product::<usize>() != latent
        {
            return Err(Error::new(
                "the audio encoder's head does not match the H3 checkpoint".into(),
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
    pub fn encode(&self, waveform: &Tensor) -> Result<Tensor> {
        let &[sequences, samples] = waveform.shape.as_slice() else {
            return Err(Error::new(format!(
                "a waveform of shape {:?} is not [stereo channels, samples]",
                waveform.shape
            )));
        };
        if sequences == 0 || samples == 0 {
            return Err(Error::new("the waveform is empty".into()));
        }
        let frames = Self::latent_frames(samples);
        let mut latents = Vec::with_capacity(sequences * frames * self.latent_channels);
        for sequence in 0..sequences {
            let mut channel = vec![0.0f32; frames * SAMPLES_PER_LATENT];
            channel[..samples].copy_from_slice(&waveform.data[sequence * samples..][..samples]);
            latents.extend(self.encode_channel(&channel, frames)?);
        }
        self.normalize(&latents, sequences, frames)
    }

    /// The features of one stereo channel, `[frames, latent channels]` in the VAE's raw units.
    fn encode_channel(&self, channel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut length = channel.len();
        let x = Array::from_f32(&self.weights.device, length, 1, channel)?;
        let mut x = self.convolve(&x, "encoder.block.0", length, 1)?;
        for (index, &stride) in STRIDES.iter().enumerate() {
            let name = format!("encoder.block.{}", index + 1);
            for (unit, &dilation) in DILATIONS.iter().enumerate() {
                let name = format!("{name}.block.{unit}.block");
                let y = self.snake(&x, &format!("{name}.0"))?;
                let y = self.convolve(&y, &format!("{name}.1"), length, dilation)?;
                let y = self.snake(&y, &format!("{name}.2"))?;
                let y = self.convolve(&y, &format!("{name}.3"), length, 1)?;
                x = x.add(&y)?;
            }
            let y = self.snake(&x, &format!("{name}.block.3"))?;
            let walk = Walk {
                dilation: 1,
                step: stride,
                padding: stride.div_ceil(2),
            };
            (x, length) = self.convolve_with(&y, &format!("{name}.block.4"), length, walk)?;
        }
        if length != frames {
            return Err(Error::new(format!(
                "the encoder gave {length} frames instead of {frames}"
            )));
        }
        let y = self.snake(&x, "encoder.block.6")?;
        let features = self.convolve(&y, "encoder.block.7", frames, 1)?;
        self.project(&features)?.to_f32()
    }

    /// The posterior head on `features[frames, features]`: a causal attention whose heads are
    /// pooled onto the latent channels, a projection of the features themselves, a GeGLU, and the
    /// projection onto the mean.
    fn project(&self, features: &Array) -> Result<Array> {
        let w = &self.weights;
        let (frames, width) = (features.rows, self.features);
        let qkv = self
            .layer_norm(features, "pre_block.norm1")?
            .linear(&w.array("pre_block.attn.qkv.weight")?)?
            .add(&self.qkv_bias)?;
        let [query, key, value] = [0, 1, 2].map(|part| qkv.slice(0, frames, part * width, width));
        let attention = query?.attention(&key?, &value?, HEADS, HEADS, true)?;
        let pooled = Array::empty(&w.device, frames, self.latent_channels)?;
        w.device.run(
            "audio_pool_heads",
            &[&attention.buffer, &pooled.buffer],
            &[
                pooled.len() as u32,
                HEADS as u32,
                (width / HEADS) as u32,
                self.latent_channels as u32,
            ],
            pooled.len(),
            false,
        )?;
        let attended = w.linear(&pooled, "pre_block.attn.proj")?;
        let x = w
            .linear(
                &self.layer_norm(features, "pre_block.norm3")?,
                "pre_block.proj",
            )?
            .add(&attended)?;

        let normalized = self.layer_norm(
            &self.layer_norm(&x, "pre_block.norm2")?,
            "pre_block.mlp.norm",
        )?;
        let gate = w.linear(&normalized, "pre_block.mlp.w0")?;
        let value = w.linear(&normalized, "pre_block.mlp.w1")?;
        let hidden = Array::empty(&w.device, gate.rows, gate.cols)?;
        w.device.run(
            "audio_geglu",
            &[&gate.buffer, &value.buffer, &hidden.buffer],
            &[hidden.len() as u32],
            hidden.len(),
            false,
        )?;
        let x = w.linear(&hidden, "pre_block.mlp.w2")?.add(&x)?;
        self.convolve(&x, "mean_proj", frames, 1)
    }

    /// `x[length, inputs]` through a convolution whose zero padding keeps the length.
    fn convolve(&self, x: &Array, name: &str, length: usize, dilation: usize) -> Result<Array> {
        let kernel = self.weights.shape(&format!("{name}.weight"))?[2];
        let walk = Walk {
            dilation,
            step: 1,
            padding: dilation * (kernel - 1) / 2,
        };
        Ok(self.convolve_with(x, name, length, walk)?.0)
    }

    /// `x[length, inputs]` through a convolution, returning `[output length, outputs]` and the
    /// output length. The im2col rows are built a slice at a time into one buffer, which the
    /// products read in the order the queue runs them.
    fn convolve_with(
        &self,
        x: &Array,
        name: &str,
        length: usize,
        walk: Walk,
    ) -> Result<(Array, usize)> {
        let w = &self.weights;
        let &[outputs, inputs, kernel] = w.shape(&format!("{name}.weight"))?.as_slice() else {
            return Err(Error::new(format!("{name}: invalid convolution shape")));
        };
        let weight = w.array(&format!("{name}.weight"))?;
        let span = walk.dilation * (kernel - 1) + 1;
        if x.shape() != [length, inputs] || length + 2 * walk.padding < span {
            return Err(Error::new(format!(
                "{name}: a signal of {length} samples does not fit the convolution"
            )));
        }
        let rows = (length + 2 * walk.padding - span) / walk.step + 1;
        let y = if kernel == 1 && walk.step == 1 {
            x.linear(&weight)?
        } else {
            let features = kernel * inputs;
            let slice = (COLUMN_SLICE_BYTES / (features * 4)).clamp(1, rows);
            let columns = Array::empty(&w.device, slice, features)?;
            let y = Array::empty(&w.device, rows, outputs)?;
            for first in (0..rows).step_by(slice) {
                let columns = Array {
                    rows: slice.min(rows - first),
                    ..columns.clone()
                };
                w.device.run(
                    "audio_columns",
                    &[&x.buffer, &columns.buffer],
                    &[
                        columns.len() as u32,
                        length as u32,
                        inputs as u32,
                        kernel as u32,
                        walk.dilation as u32,
                        walk.step as u32,
                        walk.padding as u32,
                        rows as u32,
                        first as u32,
                    ],
                    columns.len(),
                    false,
                )?;
                columns.linear_into(&weight, &y, first)?;
            }
            y
        };
        Ok((y.add(&w.vector(&format!("{name}.bias"))?)?, rows))
    }

    /// The encoder's Snake, whose one parameter a channel is both α and β.
    fn snake(&self, x: &Array, name: &str) -> Result<Array> {
        let w = &self.weights;
        let alpha = w.vector(&format!("{name}.alpha"))?;
        if alpha.len() != x.cols {
            return Err(Error::new(format!("{name}: invalid Snake parameters")));
        }
        let y = Array::empty(&w.device, x.rows, x.cols)?;
        w.device.run(
            "audio_encoder_snake",
            &[&x.buffer, &alpha.buffer, &y.buffer],
            &[y.len() as u32, y.cols as u32],
            y.len(),
            false,
        )?;
        Ok(y)
    }

    fn layer_norm(&self, x: &Array, name: &str) -> Result<Array> {
        let w = &self.weights;
        x.norm(&w.vector(&format!("{name}.weight"))?, NORM_EPSILON, true)?
            .add(&w.vector(&format!("{name}.bias"))?)
    }

    /// `[latent channels, stereo channels, frames]` from the raw `[stereo channels, frames,
    /// latent channels]` features, in normalized units.
    fn normalize(&self, latents: &[f32], sequences: usize, frames: usize) -> Result<Tensor> {
        let channels = self.latent_channels;
        let mean = self.weights.host("latents_mean")?.data;
        let deviation = self.weights.host("latents_std")?.data;
        let mut data = vec![0.0f32; channels * sequences * frames];
        for channel in 0..channels {
            for sequence in 0..sequences {
                for frame in 0..frames {
                    let value = latents[(sequence * frames + frame) * channels + channel];
                    data[(channel * sequences + sequence) * frames + frame] =
                        (value - mean[channel]) / deviation[channel];
                }
            }
        }
        Ok(Tensor::new(vec![channels, sequences, frames], data))
    }
}
