//! Reference clips read from MP4 files: the frames on the clip's canvas and its soundtrack.
//!
//! The demuxer of `mmh3-input` and NVDEC give the frames, which are fitted to the canvas the
//! reference pipeline puts a clip on, and the soundtrack comes from the same file through
//! Symphonia, which reads MP4 audio itself.

use crate::audio::load_audio_if_any;
use mmh3_core::picture::{Picture, canvas_for, reference_size};
use mmh3_core::tensor::Tensor;
use mmh3_input::mp4::Mp4File;
use std::error::Error;
use std::path::Path;

/// Frames the video VAE consumes per clip, which a reference clip's length is snapped to.
const CLIP_LENGTH: usize = 17;
/// The frames left over after the whole clips, which the latent keeps.
const CLIP_TAIL: usize = 5;

/// A clip the prompt refers to, with the soundtrack of its file when it has one.
pub struct ReferenceClip {
    /// The frames on the clip's canvas, `[frames, height, width, 3]` in [0, 1].
    pub frames: Tensor,
    /// The soundtrack at the audio VAE's rate.
    pub sound: Option<Tensor>,
}

/// Reads at most `limit` frames of an MP4 as a clip on its own canvas, its length snapped to the
/// 17n + 5 frames the VAE encodes, with its soundtrack.
pub fn load_clip(path: &Path, limit: usize) -> Result<ReferenceClip, Box<dyn Error>> {
    let mut file = Mp4File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let available = file.track().samples.len().min(limit);
    let frames = snap_frames(available).ok_or_else(|| {
        format!(
            "{}: a reference clip needs at least {CLIP_TAIL} frames, the file has {}",
            path.display(),
            file.track().samples.len()
        )
    })?;
    let decoded = mmh3_input_nvdec::decode_frames(&mut file, frames)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let [decoded_frames, height, width, 3] = decoded.shape[..] else {
        return Err(format!("{}: the frames have an odd shape", path.display()).into());
    };
    // The decoder may stop short of the samples the table promised.
    let frames = snap_frames(decoded_frames.min(frames)).ok_or_else(|| {
        format!(
            "{}: only {decoded_frames} frames decoded, too few for a reference clip",
            path.display()
        )
    })?;
    let (canvas_width, canvas_height) = clip_canvas(width, height);
    let mut pixels = Vec::with_capacity(frames * canvas_height * canvas_width * 3);
    let plane = height * width * 3;
    for frame in 0..frames {
        // The reference resizes through 8-bit pixels, so the frames go through them as well.
        let picture = Picture {
            width,
            height,
            pixels: decoded.data[frame * plane..(frame + 1) * plane]
                .iter()
                .map(|value| (value * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect(),
        };
        let fitted = picture.resize(canvas_width, canvas_height);
        pixels.extend(fitted.pixels.iter().map(|&value| f32::from(value) / 255.0));
    }
    Ok(ReferenceClip {
        frames: Tensor::new(vec![frames, canvas_height, canvas_width, 3], pixels),
        sound: load_audio_if_any(path)?,
    })
}

/// The largest length of at most `frames` frames that the VAE's clips cover, if there is one.
fn snap_frames(frames: usize) -> Option<usize> {
    if frames < CLIP_TAIL {
        return None;
    }
    Some(frames - (frames - CLIP_TAIL) % CLIP_LENGTH)
}

/// The canvas a clip of `width` × `height` pixels goes on: the generation canvas of its aspect
/// ratio, or its own size on the 32-pixel grid when that is smaller, since a clip is never
/// scaled up.
fn clip_canvas(width: usize, height: usize) -> (usize, usize) {
    let (canvas_width, canvas_height) = canvas_for(width, height);
    if width * height < canvas_width * canvas_height {
        return reference_size(width, height, width, height);
    }
    (canvas_width, canvas_height)
}

/// Seconds of every vision block of a clip: the frames go to the text encoder at two frames per
/// second, two frames to a block, and a block sits at the mean of their seconds. An odd count
/// repeats the last frame, as the reference pads the block.
pub fn block_seconds(frames: usize, fps: usize) -> Vec<f32> {
    let step = fps / 2;
    let sampled: Vec<f32> = (0..frames)
        .step_by(step.max(1))
        .enumerate()
        .map(|(index, _)| index as f32 / 2.0)
        .collect();
    sampled
        .chunks(2)
        .map(|pair| match pair {
            [first, second] => (first + second) / 2.0,
            [only] => *only,
            _ => unreachable!("chunks of two"),
        })
        .collect()
}

/// The frames of a clip that go to the text encoder, two frames to a block at two frames per
/// second, the last repeated when their count is odd.
pub fn block_frames(clip: &Tensor, fps: usize) -> Vec<Tensor> {
    let [frames, height, width, 3] = clip.shape[..] else {
        panic!("a clip is [frames, height, width, 3]");
    };
    let plane = height * width * 3;
    let mut sampled: Vec<Tensor> = (0..frames)
        .step_by((fps / 2).max(1))
        .map(|frame| {
            Tensor::new(
                vec![height, width, 3],
                clip.data[frame * plane..(frame + 1) * plane].to_vec(),
            )
        })
        .collect();
    if !sampled.len().is_multiple_of(2) {
        sampled.push(sampled[sampled.len() - 1].clone());
    }
    sampled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snaps_a_clip_to_the_frames_the_vae_covers() {
        assert_eq!(snap_frames(4), None);
        assert_eq!(snap_frames(5), Some(5));
        assert_eq!(snap_frames(21), Some(5));
        assert_eq!(snap_frames(22), Some(22));
        assert_eq!(snap_frames(124), Some(124));
        assert_eq!(snap_frames(130), Some(124));
    }

    #[test]
    fn keeps_a_small_clip_at_its_own_size() {
        // A 16:9 clip larger than the canvas goes on the canvas.
        assert_eq!(clip_canvas(1920, 1080), canvas_for(1920, 1080));
        // A small clip keeps its size on the 32-pixel grid.
        assert_eq!(clip_canvas(300, 200), (288, 192));
    }

    #[test]
    fn labels_the_blocks_at_two_frames_per_second() {
        // 24 frames per second give a frame every twelfth, so 22 frames make two blocks.
        assert_eq!(block_seconds(22, 24), vec![0.25]);
        // 124 frames give eleven sampled frames, so the last block repeats the eleventh.
        assert_eq!(
            block_seconds(124, 24),
            vec![0.25, 1.25, 2.25, 3.25, 4.25, 5.0]
        );
    }
}
