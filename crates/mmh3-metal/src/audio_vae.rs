//! BigVGAN audio decoder. Convolutions use MPS products. Resampling and SnakeBeta use Metal.
use crate::{Error, Result, model::Weights, ops::Array};
use mmh3_core::{safetensors::SafeTensors, tensor::Tensor};

pub const SAMPLE_RATE: usize = 32_000;
const UPSAMPLE: [(usize, usize); 7] = [(5, 9), (5, 9), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)];
pub struct MetalAudioDecoder {
    weights: Weights,
    channels: usize,
}

impl MetalAudioDecoder {
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self> {
        let mut weights = Weights::load_selected(file, prefix, |n| {
            n.starts_with("decoder.")
                || n.starts_with("dec_in_proj.")
                || n == "latents_mean"
                || n == "latents_std"
        })?;
        weights.prepare_audio_convolutions()?;
        let channels = weights.shape("dec_in_proj.weight")?[1];
        Ok(Self { weights, channels })
    }

    fn convolve(&self, x: &Array, name: &str, length: usize, dilation: usize) -> Result<Array> {
        let w = &self.weights;
        let shape = w.shape(&format!("{name}.weight"))?;
        let &[_, inputs, kernel] = shape.as_slice() else {
            return Err(Error::new(format!("{name}: invalid convolution shape")));
        };

        if x.cols != inputs || length == 0 || !x.rows.is_multiple_of(length) {
            return Err(Error::new("convolution input shape mismatch".into()));
        }

        let columns = if kernel == 1 {
            x.clone()
        } else {
            let cols = Array::empty(&w.device, x.rows, inputs * kernel)?;
            w.device.run(
                "audio_columns",
                &[&x.buffer, &cols.buffer],
                &[
                    cols.len() as u32,
                    length as u32,
                    inputs as u32,
                    kernel as u32,
                    dilation as u32,
                ],
                cols.len(),
                false,
            )?;
            cols
        };

        let mut y = columns.linear(&w.array(&format!("{name}.weight"))?)?;
        if w.contains(&format!("{name}.bias")) {
            y = y.add(&w.vector(&format!("{name}.bias"))?)?;
        }

        Ok(y)
    }

    fn upsample(
        &self,
        x: &Array,
        name: &str,
        length: usize,
        rate: usize,
        expected_kernel: usize,
    ) -> Result<Array> {
        let w = &self.weights;
        let shape = w.shape(&format!("{name}.weight"))?;
        let &[inputs, outputs, kernel] = shape.as_slice() else {
            return Err(Error::new("invalid transposed convolution shape".into()));
        };

        if inputs != x.cols || kernel != expected_kernel {
            return Err(Error::new("invalid vocoder upsample configuration".into()));
        }

        let product = x.linear(
            &w.array(&format!("{name}.weight"))?
                .reshape(kernel * outputs, inputs)?,
        )?;
        let out = Array::empty(&w.device, x.rows * rate, outputs)?;
        let bias = w.vector(&format!("{name}.bias"))?;
        w.device.run(
            "audio_transpose",
            &[&product.buffer, &bias.buffer, &out.buffer],
            &[
                out.len() as u32,
                length as u32,
                (length * rate) as u32,
                outputs as u32,
                kernel as u32,
                rate as u32,
            ],
            out.len(),
            false,
        )?;
        Ok(out)
    }

    fn activate(&self, x: &Array, name: &str, length: usize) -> Result<Array> {
        let w = &self.weights;
        let alpha = w.host(&format!("{name}.act.alpha"))?.data;
        let beta = w.host(&format!("{name}.act.beta"))?.data;
        if alpha.len() != x.cols || beta.len() != x.cols {
            return Err(Error::new("invalid SnakeBeta parameters".into()));
        }

        let parameters: Vec<_> = alpha
            .iter()
            .map(|a| a.exp())
            .chain(beta.iter().map(|b| 1.0 / (b.exp() + 1e-9)))
            .collect();
        let parameters = Array::from_f32(&w.device, 2, x.cols, &parameters)?;
        let up = w.vector(&format!("{name}.upsample.filter"))?;
        let down = w.vector(&format!("{name}.downsample.lowpass.filter"))?;

        if up.len() != 12 || down.len() != 12 {
            return Err(Error::new("SnakeBeta requires 12-tap filters".into()));
        }

        let out = Array::empty(&w.device, x.rows, x.cols)?;
        w.device.run(
            "audio_snake",
            &[
                &x.buffer,
                &parameters.buffer,
                &up.buffer,
                &down.buffer,
                &out.buffer,
            ],
            &[x.len() as u32, length as u32, x.cols as u32],
            x.len(),
            false,
        )?;
        Ok(out)
    }

    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let &[channels, sequences, frames] = latent.shape.as_slice() else {
            return Err(Error::new(
                "audio latent must be [channels, sequences, frames]".into(),
            ));
        };

        if channels != self.channels || frames == 0 || sequences == 0 {
            return Err(Error::new("invalid audio latent shape".into()));
        }

        let w = &self.weights;
        let mean = w.host("latents_mean")?.data;
        let std = w.host("latents_std")?.data;
        let mut rows = vec![0.0; latent.data.len()];

        for c in 0..channels {
            for s in 0..sequences {
                for t in 0..frames {
                    rows[(s * frames + t) * channels + c] =
                        latent.data[(c * sequences + s) * frames + t] * std[c] + mean[c];
                }
            }
        }

        let x = Array::from_f32(&w.device, sequences * frames, channels, &rows)?;
        let x = self.convolve(&x, "dec_in_proj", frames, 1)?;
        let mut x = self.convolve(&x, "decoder.conv_pre", frames, 1)?;
        let mut length = frames;

        for (stage, (rate, kernel)) in UPSAMPLE.into_iter().enumerate() {
            x = self.upsample(&x, &format!("decoder.ups.{stage}.0"), length, rate, kernel)?;
            length *= rate;
            let mut branches = Vec::new();
            for block in 0..3 {
                let p = format!("decoder.resblocks.{}", stage * 3 + block);
                let mut running = x.clone();
                for (unit, dilation) in [1, 3, 5].into_iter().enumerate() {
                    let first =
                        self.activate(&running, &format!("{p}.activations.{}", unit * 2), length)?;
                    let second =
                        self.convolve(&first, &format!("{p}.convs1.{unit}"), length, dilation)?;
                    let first = self.activate(
                        &second,
                        &format!("{p}.activations.{}", unit * 2 + 1),
                        length,
                    )?;
                    running = running.add(&self.convolve(
                        &first,
                        &format!("{p}.convs2.{unit}"),
                        length,
                        1,
                    )?)?;
                }

                branches.push(running);
            }

            x = branches[0]
                .add(&branches[1])?
                .add(&branches[2])?
                .unary(0, 1.0 / 3.0)?;
        }

        let x = self.activate(&x, "decoder.activation_post", length)?;
        let x = self
            .convolve(&x, "decoder.conv_post", length, 1)?
            .unary(2, 1.0)?;
        Ok(Tensor::new(vec![sequences, length], x.to_f32()?))
    }
}
