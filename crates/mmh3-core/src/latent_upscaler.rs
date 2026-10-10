//! LBH-123-AI's latent upscaler for MiniMax H3, which doubles the width and height of a video
//! latent: a 3D CNN of residual blocks and depthwise temporal convolutions on either side of a
//! trilinear resize. The layout of the network and the statistics of its input are the ones its
//! weights were trained in.
//!
//! The residual blocks modulate their second group norm by an embedding of the scale. The scale is
//! always 2 here, so the modulation folds into that norm's affine when the weights load, and every
//! backend runs plain group norms.

use crate::safetensors::SafeTensors;
use crate::tensor::Tensor;

/// Latent channels the network takes and gives.
pub const CHANNELS: usize = 24;
/// The factor of both spatial axes. The time axis keeps its length.
pub const SCALE: usize = 2;
pub const GROUPS: usize = 32;
/// The epsilon of the group norms, PyTorch's default.
pub const NORM_EPSILON: f32 = 1e-5;

/// The per-channel statistics of the latents the weights were trained against. The network takes
/// `(latent − mean) / std` and gives back the same units.
pub const MEAN: [f32; CHANNELS] = [
    0.858_090_34,
    -0.960_659_1,
    1.066_164,
    -0.509_032_6,
    -0.272_758_2,
    -1.367_541_4,
    -0.255_325_5,
    -0.269_075_55,
    -0.537_684_1,
    -0.046_409_73,
    0.665_737,
    0.196_901_28,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_341_99,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_44,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];
pub const STD: [f32; CHANNELS] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_8,
    0.799_680_95,
    0.449_889,
    0.719_74,
    0.693_629_3,
    2.961_095,
    2.769_42,
    3.049_618_5,
    2.108_805_4,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];

/// What one slot of `in_blocks` or `out_blocks` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// `x + conv(SiLU(modulated GroupNorm(conv(SiLU(GroupNorm(x))))))`, both convolutions 3 × 3 × 3.
    Residual,
    /// `x + pointwise(depthwise(SiLU(GroupNorm(x))))`, the depthwise convolution along time only.
    Temporal { kernel: usize },
}

pub fn load(file: &SafeTensors, name: &str) -> Result<Tensor, String> {
    let info = file
        .get(name)
        .ok_or_else(|| format!("the latent upscaler has no {name}"))?;
    Tensor::load(file, info)
}

/// The kinds of the blocks of `in_blocks` or `out_blocks`, read off their keys.
pub fn blocks(file: &SafeTensors, prefix: &str) -> Result<Vec<BlockKind>, String> {
    let mut kinds = Vec::new();
    loop {
        let name = format!("{prefix}.{}", kinds.len());
        if let Some(info) = file.get(&format!("{name}.dwconv.weight")) {
            kinds.push(BlockKind::Temporal {
                kernel: info.shape[2],
            });
        } else if file.get(&format!("{name}.in_layers.2.weight")).is_some() {
            kinds.push(BlockKind::Residual);
        } else {
            break;
        }
    }
    if kinds.is_empty() {
        return Err(format!("the latent upscaler has no {prefix}"));
    }
    Ok(kinds)
}

/// The embedding of the scale, `Linear(SiLU(Linear(scale − 1)))`.
pub fn embedding(file: &SafeTensors) -> Result<Vec<f32>, String> {
    let first = linear(file, "embed.0", &[SCALE as f32 - 1.0])?;
    let activated: Vec<f32> = first.iter().map(|&value| silu(value)).collect();
    linear(file, "embed.2", &activated)
}

/// The affine of a residual block's second group norm with the block's modulation folded in:
/// `(normalized · weight + bias) · (1 + scale) + shift`.
pub fn modulated_norm(
    file: &SafeTensors,
    block: &str,
    embedding: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let activated: Vec<f32> = embedding.iter().map(|&value| silu(value)).collect();
    let modulation = linear(file, &format!("{block}.emb_layers.1"), &activated)?;
    let weight = load(file, &format!("{block}.out_norm.weight"))?.data;
    let bias = load(file, &format!("{block}.out_norm.bias"))?.data;
    let channels = weight.len();
    if modulation.len() != 2 * channels {
        return Err(format!("{block}: the modulation does not match the norm"));
    }
    let (scale, shift) = modulation.split_at(channels);
    Ok((
        (0..channels)
            .map(|channel| weight[channel] * (1.0 + scale[channel]))
            .collect(),
        (0..channels)
            .map(|channel| bias[channel] * (1.0 + scale[channel]) + shift[channel])
            .collect(),
    ))
}

fn linear(file: &SafeTensors, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
    let weight = load(file, &format!("{name}.weight"))?;
    let bias = load(file, &format!("{name}.bias"))?;
    let [outputs, inputs] = weight.shape[..] else {
        return Err(format!("{name} is not a Linear"));
    };
    if inputs != input.len() {
        return Err(format!("{name} takes {inputs} values, not {}", input.len()));
    }
    Ok((0..outputs)
        .map(|output| {
            bias.data[output]
                + (0..inputs)
                    .map(|index| weight.data[output * inputs + index] * input[index])
                    .sum::<f32>()
        })
        .collect())
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// A latent `[24, frames, height, width]` normalized into rows `[frames, height, width, channels]`,
/// the channels past 24 zero.
pub fn rows(latent: &Tensor, channels: usize) -> Vec<f32> {
    let (frames, height, width) = (latent.shape[1], latent.shape[2], latent.shape[3]);
    let plane = frames * height * width;
    let mut rows = vec![0.0; plane * channels];
    for channel in 0..CHANNELS {
        for (pixel, &value) in latent.data[channel * plane..(channel + 1) * plane]
            .iter()
            .enumerate()
        {
            rows[pixel * channels + channel] = (value - MEAN[channel]) / STD[channel];
        }
    }
    rows
}

/// The inverse of `rows` for rows `[frames, height, width, 24]`.
pub fn latent(rows: &[f32], frames: usize, height: usize, width: usize) -> Tensor {
    let plane = frames * height * width;
    let mut data = vec![0.0; CHANNELS * plane];
    for pixel in 0..plane {
        for channel in 0..CHANNELS {
            data[channel * plane + pixel] =
                rows[pixel * CHANNELS + channel] * STD[channel] + MEAN[channel];
        }
    }
    Tensor::new(vec![CHANNELS, frames, height, width], data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip() {
        let latent = Tensor::new(
            vec![CHANNELS, 2, 1, 3],
            (0..CHANNELS * 6).map(|value| value as f32 * 0.1).collect(),
        );
        let rows = rows(&latent, CHANNELS);
        assert!((rows[1] - (latent.data[6] - MEAN[1]) / STD[1]).abs() < 1e-6);
        let back = super::latent(&rows, 2, 1, 3);
        for (back, original) in back.data.iter().zip(&latent.data) {
            assert!((back - original).abs() < 1e-5);
        }
    }
}
