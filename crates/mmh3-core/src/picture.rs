//! Pictures for conditioning: fitting them to the canvas of a generation the way the reference
//! pipelines do, with Pillow's Lanczos filter on 8-bit channels.

use crate::generation::CANVAS_MULTIPLE;
use crate::tensor::Tensor;

/// The short edge of the canvases H3 was trained on.
const BASE_SHORT_EDGE: f64 = 768.0;
/// The largest canvas area, 768 × 1344.
const MAX_PIXELS: f64 = 768.0 * 1344.0;
/// Pillow's fixed-point precision for 8-bit resampling.
const PRECISION_BITS: u32 = 32 - 8 - 2;
const LANCZOS_SUPPORT: f64 = 3.0;

/// An 8-bit RGB picture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    /// `[height, width, 3]`.
    pub pixels: Vec<u8>,
}

/// The canvas for a picture of `width × height`: a 768-pixel short edge at the picture's aspect
/// ratio, at most 768 × 1344 pixels, each side rounded to a multiple of 32.
pub fn canvas_for(width: usize, height: usize) -> (usize, usize) {
    let ratio = width as f64 / height as f64;
    let (mut canvas_width, mut canvas_height) = if ratio >= 1.0 {
        (BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE)
    } else {
        (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio)
    };
    if canvas_width * canvas_height > MAX_PIXELS {
        let scale = (MAX_PIXELS / (canvas_width * canvas_height)).sqrt();
        canvas_width *= scale;
        canvas_height *= scale;
    }
    let snap = |side: f64| {
        let multiple = CANVAS_MULTIPLE as f64;
        ((side / multiple).round_ties_even() as usize * CANVAS_MULTIPLE).max(CANVAS_MULTIPLE)
    };
    (snap(canvas_width), snap(canvas_height))
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

fn lanczos(x: f64) -> f64 {
    if (-LANCZOS_SUPPORT..LANCZOS_SUPPORT).contains(&x) {
        sinc(x) * sinc(x / LANCZOS_SUPPORT)
    } else {
        0.0
    }
}

/// Pillow's resampling taps from `input` to `output` samples: for each output sample the first
/// input sample and the fixed-point weights of the following ones.
fn taps(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = LANCZOS_SUPPORT * filter_scale;
    (0..output)
        .map(|index| {
            let center = (index as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as i64).max(0) as usize;
            let last = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let weights: Vec<f64> = (first..last)
                .map(|source| lanczos((source as f64 - center + 0.5) / filter_scale))
                .collect();
            let sum: f64 = weights.iter().sum();
            let fixed = weights
                .into_iter()
                .map(|weight| {
                    let weight = if sum != 0.0 { weight / sum } else { weight };
                    let scaled = weight * f64::from(1u32 << PRECISION_BITS);
                    (if weight < 0.0 {
                        scaled - 0.5
                    } else {
                        scaled + 0.5
                    }) as i32
                })
                .collect();
            (first, fixed)
        })
        .collect()
}

fn clip8(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

impl Picture {
    /// Resizes with Pillow's `LANCZOS` filter, bit for bit: a horizontal pass over the rows the
    /// vertical pass reads, then the vertical pass, each rounding to 8 bits.
    pub fn resize(&self, width: usize, height: usize) -> Picture {
        let rounding = 1i32 << (PRECISION_BITS - 1);
        let horizontal = (width != self.width).then(|| taps(self.width, width));
        let vertical = (height != self.height).then(|| taps(self.height, height));
        let (first_row, last_row) = match &vertical {
            Some(taps) => {
                let (last_first, last_weights) = taps.last().unwrap();
                (taps[0].0, last_first + last_weights.len())
            }
            None => (0, self.height),
        };
        let rows = last_row - first_row;
        let stretched = match &horizontal {
            Some(taps) => {
                let mut pixels = vec![0u8; rows * width * 3];
                for row in 0..rows {
                    let source = &self.pixels[(first_row + row) * self.width * 3..];
                    for (column, (first, weights)) in taps.iter().enumerate() {
                        for channel in 0..3 {
                            let mut sum = rounding;
                            for (offset, &weight) in weights.iter().enumerate() {
                                sum += i32::from(source[(first + offset) * 3 + channel]) * weight;
                            }
                            pixels[(row * width + column) * 3 + channel] = clip8(sum);
                        }
                    }
                }
                pixels
            }
            None => self.pixels[first_row * self.width * 3..last_row * self.width * 3].to_vec(),
        };
        let pixels = match &vertical {
            Some(taps) => {
                let mut pixels = vec![0u8; height * width * 3];
                for (row, (first, weights)) in taps.iter().enumerate() {
                    for index in 0..width * 3 {
                        let mut sum = rounding;
                        for (offset, &weight) in weights.iter().enumerate() {
                            let source = (first - first_row + offset) * width * 3 + index;
                            sum += i32::from(stretched[source]) * weight;
                        }
                        pixels[row * width * 3 + index] = clip8(sum);
                    }
                }
                pixels
            }
            None => stretched,
        };
        Picture {
            width,
            height,
            pixels,
        }
    }

    /// Fits the picture to a canvas like the reference pipeline: stretched to it, or scaled to
    /// cover it and cropped around the center.
    pub fn fit(&self, width: usize, height: usize, stretch: bool) -> Picture {
        if stretch {
            return self.resize(width, height);
        }
        let scale = (width as f64 / self.width as f64).max(height as f64 / self.height as f64);
        let covered_width = ((self.width as f64 * scale).round_ties_even() as usize).max(width);
        let covered_height = ((self.height as f64 * scale).round_ties_even() as usize).max(height);
        let covered = self.resize(covered_width, covered_height);
        let (left, top) = ((covered_width - width) / 2, (covered_height - height) / 2);
        let mut pixels = Vec::with_capacity(width * height * 3);
        for row in top..top + height {
            let start = (row * covered_width + left) * 3;
            pixels.extend_from_slice(&covered.pixels[start..start + width * 3]);
        }
        Picture {
            width,
            height,
            pixels,
        }
    }

    /// `[height, width, 3]` in [0, 1].
    pub fn to_tensor(&self) -> Tensor {
        Tensor::new(
            vec![self.height, self.width, 3],
            self.pixels
                .iter()
                .map(|&value| f32::from(value) / 255.0)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_canvas_from_the_aspect_ratio() {
        assert_eq!(canvas_for(1920, 1080), (1344, 768));
        assert_eq!(canvas_for(448, 256), (1344, 768));
        assert_eq!(canvas_for(1080, 1920), (768, 1344));
        assert_eq!(canvas_for(1000, 1000), (768, 768));
        assert_eq!(canvas_for(4000, 3000), (1024, 768));
        // Wider than 1344 × 768 shrinks to its area.
        assert_eq!(canvas_for(2400, 1000), (1568, 640));
    }

    #[test]
    fn keeps_flat_pictures_flat() {
        let flat = Picture {
            width: 7,
            height: 5,
            pixels: vec![200; 7 * 5 * 3],
        };
        let resized = flat.resize(19, 3);
        assert_eq!((resized.width, resized.height), (19, 3));
        assert!(resized.pixels.iter().all(|&value| value == 200));
        assert_eq!(flat.resize(7, 5), flat);
        let cropped = flat.fit(4, 4, false);
        assert_eq!((cropped.width, cropped.height), (4, 4));
    }
}
