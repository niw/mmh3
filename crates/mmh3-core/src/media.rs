//! Uncompressed output files: YUV4MPEG2 video and WAV audio, which common tools such as ffmpeg read directly.

use crate::tensor::Tensor;
use std::io::{self, Write};

/// BT.709 luma weights of red and blue.
const LUMA_RED: f32 = 0.2126;
const LUMA_BLUE: f32 = 0.0722;

fn limited_range(value: f32, offset: f32, scale: f32) -> u8 {
    (offset + scale * value).round().clamp(0.0, 255.0) as u8
}

/// Writes pixels `[3, frames, height, width]` in [0, 1] as 4:2:0 YUV4MPEG2 with BT.709 limited-range colors.
/// Chroma is the average of each 2 × 2 block. Height and width must be even.
pub fn write_y4m(writer: &mut impl Write, pixels: &Tensor, fps: usize) -> io::Result<()> {
    let &[channels, frames, height, width] = pixels.shape.as_slice() else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "pixels must be [3, frames, height, width]"));
    };
    if channels != 3 || height % 2 != 0 || width % 2 != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "pixels must be RGB with an even height and width"));
    }
    write!(writer, "YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C420jpeg XCOLORRANGE=LIMITED\n")?;
    let plane = height * width;
    let luma_green = 1.0 - LUMA_RED - LUMA_BLUE;
    let mut luma = vec![0u8; plane];
    let mut blue = vec![0u8; plane / 4];
    let mut red = vec![0u8; plane / 4];
    for frame in 0..frames {
        let channel = |index: usize| &pixels.data[(index * frames + frame) * plane..][..plane];
        let (r, g, b) = (channel(0), channel(1), channel(2));
        for index in 0..plane {
            luma[index] = limited_range(LUMA_RED * r[index] + luma_green * g[index] + LUMA_BLUE * b[index], 16.0, 219.0);
        }
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                let corners = [2 * y * width + 2 * x, 2 * y * width + 2 * x + 1, (2 * y + 1) * width + 2 * x, (2 * y + 1) * width + 2 * x + 1];
                let average = |values: &[f32]| corners.iter().map(|&index| values[index]).sum::<f32>() / 4.0;
                let (red_value, green_value, blue_value) = (average(r), average(g), average(b));
                let luma_value = LUMA_RED * red_value + luma_green * green_value + LUMA_BLUE * blue_value;
                blue[y * width / 2 + x] = limited_range((blue_value - luma_value) / (2.0 * (1.0 - LUMA_BLUE)), 128.0, 224.0);
                red[y * width / 2 + x] = limited_range((red_value - luma_value) / (2.0 * (1.0 - LUMA_RED)), 128.0, 224.0);
            }
        }
        writer.write_all(b"FRAME\n")?;
        writer.write_all(&luma)?;
        writer.write_all(&blue)?;
        writer.write_all(&red)?;
    }
    Ok(())
}

/// Writes a waveform `[channels, samples]` in [−1, 1] as interleaved 16-bit PCM WAV.
pub fn write_wav(writer: &mut impl Write, waveform: &Tensor, sample_rate: usize) -> io::Result<()> {
    let &[channels, samples] = waveform.shape.as_slice() else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "the waveform must be [channels, samples]"));
    };
    let data_bytes = channels * samples * 2;
    let block_align = channels * 2;
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_bytes as u32).to_le_bytes())?;
    writer.write_all(b"WAVEfmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&(channels as u16).to_le_bytes())?;
    writer.write_all(&(sample_rate as u32).to_le_bytes())?;
    writer.write_all(&((sample_rate * block_align) as u32).to_le_bytes())?;
    writer.write_all(&(block_align as u16).to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&(data_bytes as u32).to_le_bytes())?;
    let mut interleaved = Vec::with_capacity(data_bytes);
    for sample in 0..samples {
        for channel in 0..channels {
            let value = (waveform.data[channel * samples + sample].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            interleaved.extend_from_slice(&value.to_le_bytes());
        }
    }
    writer.write_all(&interleaved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_y4m_frames() {
        // Two frames of 2 × 2: white, then pure red.
        let mut data = vec![1.0f32; 3 * 2 * 4];
        for channel in [1, 2] {
            data[(channel * 2 + 1) * 4..][..4].fill(0.0);
        }
        let mut bytes = Vec::new();
        write_y4m(&mut bytes, &Tensor::new(vec![3, 2, 2, 2], data), 24).unwrap();
        let header = b"YUV4MPEG2 W2 H2 F24:1 Ip A1:1 C420jpeg XCOLORRANGE=LIMITED\n";
        assert_eq!(&bytes[..header.len()], header);
        let frames = &bytes[header.len()..];
        assert_eq!(frames.len(), 2 * (6 + 6));
        assert_eq!(&frames[..12], b"FRAME\n\xEB\xEB\xEB\xEB\x80\x80");
        assert_eq!(&frames[12..], b"FRAME\n\x3F\x3F\x3F\x3F\x66\xF0");
    }

    #[test]
    fn writes_pcm_wav() {
        let mut bytes = Vec::new();
        write_wav(&mut bytes, &Tensor::new(vec![2, 2], vec![0.5, -1.0, 1.0, 2.0]), 32_000).unwrap();
        assert_eq!(bytes.len(), 44 + 8);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 32_000);
        let samples: Vec<i16> = bytes[44..].chunks_exact(2).map(|pair| i16::from_le_bytes([pair[0], pair[1]])).collect();
        assert_eq!(samples, vec![16384, 32767, -32767, 32767]);
    }
}
