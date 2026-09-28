//! The colour of decoded video: NV12 pixels as RGB, with the matrix and range a stream declares.
//!
//! The hardware decoders of each backend hand frames back as NV12 in host memory, and the
//! conversion to RGB happens here so that a clip comes out the same whichever decoded it.

use mmh3_core::tensor::Tensor;

/// How a stream's luma and chroma become RGB: the weights of red and blue, and whether the values
/// use the whole 8-bit range or the studio range of 16 to 235.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Colour {
    red: f32,
    blue: f32,
    full_range: bool,
}

impl Colour {
    /// The matrix a stream declares, by the code of H.264's VUI. A stream that declares none gets
    /// the guess players make from its height: BT.709 for high definition, BT.601 below it.
    pub fn of(matrix: i32, full_range: bool, height: usize) -> Self {
        let (red, blue) = match matrix {
            // BT.709, BT.601 in its two forms, SMPTE 240M and BT.2020 non-constant luminance.
            1 => (0.2126, 0.0722),
            5 | 6 => (0.299, 0.114),
            7 => (0.212, 0.087),
            9 | 10 => (0.2627, 0.0593),
            _ if height > 576 => (0.2126, 0.0722),
            _ => (0.299, 0.114),
        };
        Colour {
            red,
            blue,
            full_range,
        }
    }
}

/// NV12 pixels as RGB `[height, width, 3]` in [0, 1], in the given colour. Chroma is shared by
/// every 2 x 2 block of pixels.
pub fn nv12_to_rgb(nv12: &[u8], width: usize, height: usize, colour: Colour) -> Tensor {
    let (red, blue) = (colour.red, colour.blue);
    let green = 1.0 - red - blue;
    // The chroma weights follow from the luma ones, as the matrix of a colour system does.
    let (to_red, to_blue) = (2.0 * (1.0 - red), 2.0 * (1.0 - blue));
    let (from_red, from_blue) = (to_red * red / green, to_blue * blue / green);
    let (luma_offset, luma_scale, chroma_scale) = if colour.full_range {
        (0.0, 255.0, 255.0)
    } else {
        (16.0, 219.0, 224.0)
    };
    let chroma = width * height;
    let mut pixels = vec![0.0f32; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let luma = (f32::from(nv12[y * width + x]) - luma_offset) / luma_scale;
            let offset = chroma + (y / 2) * width + (x / 2) * 2;
            let blue = (f32::from(nv12[offset]) - 128.0) / chroma_scale;
            let red = (f32::from(nv12[offset + 1]) - 128.0) / chroma_scale;
            let values = [
                luma + to_red * red,
                luma - from_blue * blue - from_red * red,
                luma + to_blue * blue,
            ];
            for (channel, value) in values.into_iter().enumerate() {
                pixels[(y * width + x) * 3 + channel] = value.clamp(0.0, 1.0);
            }
        }
    }
    Tensor::new(vec![height, width, 3], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 2 x 2 block of a colour, as NV12.
    fn block(luma: u8, blue: u8, red: u8) -> Vec<u8> {
        vec![luma, luma, luma, luma, blue, red]
    }

    #[test]
    fn guesses_the_matrix_a_stream_leaves_out() {
        // High definition without signalling is BT.709, standard definition BT.601.
        assert_eq!(Colour::of(2, false, 768), Colour::of(1, false, 768));
        assert_eq!(Colour::of(2, false, 256), Colour::of(5, false, 256));
        assert!(Colour::of(1, true, 768).full_range);
    }

    #[test]
    fn converts_the_limited_range_of_bt709() {
        let cases = [
            // Black and white sit at the ends of the luma range.
            (block(16, 128, 128), [0.0, 0.0, 0.0]),
            (block(235, 128, 128), [1.0, 1.0, 1.0]),
            // BT.709 puts pure red at luma 0.2126 with the chroma of its own difference.
            (block(63, 102, 240), [1.0, 0.0, 0.0]),
            (block(173, 42, 26), [0.0, 1.0, 0.0]),
            (block(32, 240, 118), [0.0, 0.0, 1.0]),
        ];
        let colour = Colour::of(1, false, 768);
        for (nv12, expected) in cases {
            let pixels = nv12_to_rgb(&nv12, 2, 2, colour);
            assert_eq!(pixels.shape, vec![2, 2, 3]);
            for pixel in 0..4 {
                let values = &pixels.data[pixel * 3..pixel * 3 + 3];
                for (value, expected) in values.iter().zip(&expected) {
                    assert!(
                        (value - expected).abs() < 0.01,
                        "{values:?} against {expected:?}"
                    );
                }
            }
        }
    }
}
