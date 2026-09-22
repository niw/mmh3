#![cfg(all(feature = "metal", target_os = "macos"))]
//! The per-chunk canvas a leader hands to another machine: its size, its bounds, and its agreement
//! with the whole decode it is a part of.
use mmh3_core::{
    media::Yuv420,
    safetensors::{SafeTensors, write_f32},
    tensor::Tensor,
    vae::TemporalPlan,
};
use mmh3_metal::vae::MetalVideoDecoder;
use std::path::Path;

const CHANNELS: usize = 24;
const TILE: usize = 32;
const OVERLAP: usize = 16;
const LATENT_EXTENT: usize = 2;
/// `video_write_frames` leaves the canvas normalized and denormalizes as it writes frames out.
const SCALE: [f32; 3] = [0.229, 0.224, 0.225];
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];

fn decoder(directory: &Path) -> MetalVideoDecoder {
    let fixture = SafeTensors::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vae_tiny.safetensors"),
    )
    .unwrap();
    let tensors: Vec<_> = fixture
        .tensors()
        .iter()
        .filter_map(|info| {
            info.name
                .strip_prefix("weight.")
                .map(|name| (name, Tensor::load(&fixture, info).unwrap()))
        })
        .collect();
    let values: Vec<_> = tensors
        .iter()
        .map(|(name, tensor)| (*name, tensor.shape.as_slice(), tensor.data.as_slice()))
        .collect();
    let path = directory.join("vae.safetensors");
    write_f32(&path, &values, &[]).unwrap();
    MetalVideoDecoder::load(&SafeTensors::open(&path).unwrap(), "", TILE, OVERLAP).unwrap()
}

/// A latent whose values are spread over a range the decoder does something visible with.
fn sample_latent(frames: usize) -> Tensor {
    let count = CHANNELS * frames * LATENT_EXTENT * LATENT_EXTENT;
    let data = (0..count)
        .map(|i| ((i * 2654435761) % 1013) as f32 / 1013.0 - 0.5)
        .collect();
    Tensor::new(vec![CHANNELS, frames, LATENT_EXTENT, LATENT_EXTENT], data)
}

fn canvas_bytes(canvas_frames: usize) -> usize {
    3 * canvas_frames * (LATENT_EXTENT * 16) * (LATENT_EXTENT * 16) * 4
}

#[test]
fn a_chunk_canvas_has_the_size_the_leader_expects() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());

    let latent_frames = 12;
    let chunks = TemporalPlan::new(latent_frames).chunks;
    assert!(chunks > 1, "this latent should span several chunks");
    let latent = sample_latent(latent_frames);
    for chunk in 0..chunks {
        let bytes = decoder.decode_chunk(&latent, chunk).unwrap();
        assert_eq!(bytes.len(), canvas_bytes(28), "chunk {chunk}");
    }

    // A single latent frame is one chunk of one four-frame canvas.
    let single = sample_latent(1);
    assert_eq!(TemporalPlan::new(1).chunks, 1);
    assert_eq!(
        decoder.decode_chunk(&single, 0).unwrap().len(),
        canvas_bytes(4)
    );
}

#[test]
fn a_chunk_past_the_end_is_an_error() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent_frames = 12;
    let chunks = TemporalPlan::new(latent_frames).chunks;
    let latent = sample_latent(latent_frames);
    assert!(decoder.decode_chunk(&latent, chunks).is_err());
    assert!(decoder.decode_chunk(&sample_latent(1), 1).is_err());
}

#[test]
fn a_chunk_canvas_is_the_same_bytes_every_time() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);
    assert_eq!(
        decoder.decode_chunk(&latent, 1).unwrap(),
        decoder.decode_chunk(&latent, 1).unwrap()
    );
}

/// The first chunk blends against nothing, so a whole decode's opening frames are that chunk's
/// canvas from frame 3 on, denormalized and clamped. This is what ties the two paths together.
#[test]
fn the_first_chunks_canvas_carries_the_frames_a_whole_decode_writes() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);

    let bytes = decoder.decode_chunk(&latent, 0).unwrap();
    let canvas: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect();
    let pixels = decoder.decode(&latent, false).unwrap().pixels;
    let (frames, height, width) = (pixels.shape[1], pixels.shape[2], pixels.shape[3]);
    let plane = height * width;
    let count = 17.min(frames);

    for channel in 0..3 {
        for frame in 0..count {
            for pixel in 0..plane {
                let value = canvas[(channel * 28 + 3 + frame) * plane + pixel];
                let expected = (value * SCALE[channel] + MEAN[channel]).clamp(0.0, 1.0);
                let found = pixels.data[(channel * frames + frame) * plane + pixel];
                assert!(
                    (expected - found).abs() < 1e-6,
                    "channel {channel} frame {frame} pixel {pixel}: {expected} against {found}"
                );
            }
        }
    }
}

/// A canvas handed in stands in for the one this machine would have built, so a decode fed its own
/// canvases back comes out bit for bit what it produces on its own.
#[test]
fn a_canvas_from_elsewhere_stands_in_for_the_one_built_here() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);
    let chunks = TemporalPlan::new(12).chunks;

    let expected = decoder.decode(&latent, false).unwrap().pixels;
    let canvases: Vec<Vec<u8>> = (0..chunks)
        .map(|chunk| decoder.decode_chunk(&latent, chunk).unwrap())
        .collect();
    let frames = decoder
        .decode_device_with(&latent, &mut |chunk| Ok(Some(canvases[chunk].clone())))
        .unwrap();

    assert_eq!(frames.to_pixels().unwrap().data, expected.data);
}

/// A canvas of the wrong size did not come from this latent, so the chunk decodes here instead and
/// the video is still whole.
#[test]
fn a_canvas_of_the_wrong_size_is_refused_and_the_chunk_decodes_here() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);

    let expected = decoder.decode(&latent, false).unwrap().pixels;
    let short = decoder.decode_chunk(&latent, 0).unwrap().len() - 3;
    let frames = decoder
        .decode_device_with(&latent, &mut |_| Ok(Some(vec![0u8; short])))
        .unwrap();

    assert_eq!(frames.to_pixels().unwrap().data, expected.data);
}

/// The temporal tail carries from one chunk to the next, so the chunks are asked for in order.
#[test]
fn the_chunks_are_asked_for_in_order() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);
    let chunks = TemporalPlan::new(12).chunks;

    let mut asked = Vec::new();
    decoder
        .decode_device_with(&latent, &mut |chunk| {
            asked.push(chunk);
            Ok(None)
        })
        .unwrap();

    assert_eq!(asked, (0..chunks).collect::<Vec<_>>());
}

/// A chunk handed to another machine that never came back is not this machine's to decode: it
/// gave its chunks away because it has no weights to decode them with. The decode says so rather
/// than filling the gap.
#[test]
fn a_chunk_that_never_came_back_ends_the_decode() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);

    let mut asked = Vec::new();
    let Err(error) = decoder.decode_device_with(&latent, &mut |chunk| {
        asked.push(chunk);
        Err("the worker on 127.0.0.1 went away".into())
    }) else {
        panic!("a chunk that never came back should end the decode");
    };

    assert_eq!(asked, vec![0], "the decode stopped at the first chunk");
    assert!(error.message.contains("went away"), "{}", error.message);
}

#[test]
fn the_plan_describes_the_canvases_a_leader_hands_round() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent = sample_latent(12);

    let plan = decoder.plan(&latent).unwrap();
    assert_eq!(plan.chunks, TemporalPlan::new(12).chunks);
    assert_eq!(plan.canvas_frames, 28);
    assert_eq!(plan.height, LATENT_EXTENT * 16);
    assert_eq!(plan.width, LATENT_EXTENT * 16);
    assert_eq!(
        plan.canvas_values() * 4,
        decoder.decode_chunk(&latent, 0).unwrap().len()
    );
}

/// What a machine with no device of its own asks for covers the whole video rather than a chunk of
/// it, and arrives as the planes an encoder takes. A decode wired to a chunk would answer with the
/// right kind of thing and the wrong amount of it, which the length is what catches.
#[test]
fn the_whole_video_comes_back_as_the_planes_an_encoder_takes() {
    let directory = tempfile::tempdir().unwrap();
    let decoder = decoder(directory.path());
    let latent_frames = 12;
    assert!(
        TemporalPlan::new(latent_frames).chunks > 1,
        "this latent should span several chunks, so that a chunk is not the whole"
    );
    let latent = sample_latent(latent_frames);

    let [_, frames, height, width] = decoder.decode_device(&latent).unwrap().shape();
    let yuv = decoder.decode_yuv420(&latent).unwrap();

    assert_eq!(
        (yuv.frames, yuv.height, yuv.width),
        (frames, height, width),
        "the planes should cover what the pixels do"
    );
    assert_eq!(yuv.data.len(), frames * Yuv420::frame_bytes(height, width));
}
