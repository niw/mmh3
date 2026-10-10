//! Conversions between latents and the token rows the DiT embeds.

use crate::tensor::Tensor;

/// Video latent `[channels, frames, height, width]` to rows of 2 × 2 patches, features ordered
/// (channel, y, x).
pub fn patchify_video(latent: &Tensor) -> Vec<f32> {
    let (channels, frames, height, width) = (
        latent.shape[0],
        latent.shape[1],
        latent.shape[2],
        latent.shape[3],
    );
    let mut rows = Vec::with_capacity(latent.data.len());
    for frame in 0..frames {
        for patch_y in 0..height / 2 {
            for patch_x in 0..width / 2 {
                for channel in 0..channels {
                    for offset_y in 0..2 {
                        for offset_x in 0..2 {
                            let (y, x) = (patch_y * 2 + offset_y, patch_x * 2 + offset_x);
                            rows.push(
                                latent.data[((channel * frames + frame) * height + y) * width + x],
                            );
                        }
                    }
                }
            }
        }
    }
    rows
}

/// Inverse of `patchify_video` for a latent of `shape`.
pub fn unpatchify_video(rows: &[f32], shape: &[usize]) -> Vec<f32> {
    let (channels, frames, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let mut latent = vec![0.0; rows.len()];
    let mut index = 0;
    for frame in 0..frames {
        for patch_y in 0..height / 2 {
            for patch_x in 0..width / 2 {
                for channel in 0..channels {
                    for offset_y in 0..2 {
                        for offset_x in 0..2 {
                            let (y, x) = (patch_y * 2 + offset_y, patch_x * 2 + offset_x);
                            latent[((channel * frames + frame) * height + y) * width + x] =
                                rows[index];
                            index += 1;
                        }
                    }
                }
            }
        }
    }
    latent
}

/// Audio latent `[channels, 2, frames]` to rows ordered stereo channel first, then frame.
pub fn pack_audio(latent: &Tensor) -> Vec<f32> {
    let (channels, stereo, frames) = (latent.shape[0], latent.shape[1], latent.shape[2]);
    let mut rows = Vec::with_capacity(latent.data.len());
    for side in 0..stereo {
        for frame in 0..frames {
            rows.extend(
                (0..channels)
                    .map(|channel| latent.data[(channel * stereo + side) * frames + frame]),
            );
        }
    }
    rows
}

/// Inverse of `pack_audio` for a latent of `shape`.
pub fn unpack_audio(rows: &[f32], shape: &[usize]) -> Vec<f32> {
    let (channels, stereo, frames) = (shape[0], shape[1], shape[2]);
    let mut latent = vec![0.0; rows.len()];
    for side in 0..stereo {
        for frame in 0..frames {
            for channel in 0..channels {
                latent[(channel * stereo + side) * frames + frame] =
                    rows[(side * frames + frame) * channels + channel];
            }
        }
    }
    latent
}

/// Video latent `[channels, frames, height, width]` resized bilinearly to `height × width` in every
/// frame, with pixel centers aligned as `align_corners=False` aligns them.
pub fn resize_video(latent: &Tensor, height: usize, width: usize) -> Tensor {
    let (channels, frames, source_height, source_width) = (
        latent.shape[0],
        latent.shape[1],
        latent.shape[2],
        latent.shape[3],
    );
    // For one output coordinate, the two source coordinates it lies between and the weight of the
    // second one.
    let taps = |size: usize, source_size: usize| -> Vec<(usize, usize, f32)> {
        let scale = source_size as f32 / size as f32;
        (0..size)
            .map(|index| {
                let position = ((index as f32 + 0.5) * scale - 0.5).max(0.0);
                let low = (position as usize).min(source_size - 1);
                let high = (low + 1).min(source_size - 1);
                (low, high, position - low as f32)
            })
            .collect()
    };
    let (rows, columns) = (taps(height, source_height), taps(width, source_width));
    let mut data = Vec::with_capacity(channels * frames * height * width);
    for plane in latent.data.chunks_exact(source_height * source_width) {
        for &(top, bottom, down) in &rows {
            for &(left, right, across) in &columns {
                let upper = plane[top * source_width + left] * (1.0 - across)
                    + plane[top * source_width + right] * across;
                let lower = plane[bottom * source_width + left] * (1.0 - across)
                    + plane[bottom * source_width + right] * across;
                data.push(upper * (1.0 - down) + lower * down);
            }
        }
    }
    Tensor::new(vec![channels, frames, height, width], data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_latents() {
        let video = Tensor::new(
            vec![3, 2, 4, 6],
            (0..144).map(|value| value as f32).collect(),
        );
        let rows = patchify_video(&video);
        assert_eq!(&rows[..4], &[0.0, 1.0, 6.0, 7.0]);
        assert_eq!(unpatchify_video(&rows, &video.shape), video.data);
        let audio = Tensor::new(vec![4, 2, 3], (0..24).map(|value| value as f32).collect());
        let rows = pack_audio(&audio);
        assert_eq!(&rows[..4], &[0.0, 6.0, 12.0, 18.0]);
        assert_eq!(unpack_audio(&rows, &audio.shape), audio.data);
    }

    #[test]
    fn doubles_a_frame_bilinearly() {
        let video = Tensor::new(vec![1, 1, 1, 2], vec![0.0, 4.0]);
        let resized = resize_video(&video, 2, 4);
        assert_eq!(resized.shape, vec![1, 1, 2, 4]);
        assert_eq!(resized.data, vec![0.0, 1.0, 3.0, 4.0, 0.0, 1.0, 3.0, 4.0]);
    }
}
