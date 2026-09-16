#![cfg(target_os = "macos")]

//! Runs the Metal DiT on the tiny golden fixture from tools/golden/dit_tiny.py.

use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_metal::dit::MetalDit;
use std::path::Path;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dit_tiny.safetensors"
);

fn tensor(file: &SafeTensors, name: &str) -> Tensor {
    Tensor::load(
        file,
        file.get(name).unwrap_or_else(|| panic!("missing {name}")),
    )
    .unwrap()
}

fn unbatched(mut tensor: Tensor) -> Tensor {
    tensor.shape.remove(0);
    tensor
}

fn metadata_number(file: &SafeTensors, key: &str) -> f32 {
    file.metadata()
        .iter()
        .find(|(name, _)| name == key)
        .unwrap()
        .1
        .parse()
        .unwrap()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| left as f64 * right as f64)
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|&value| value as f64 * value as f64)
            .sum::<f64>()
            .sqrt()
    };

    let cosine = dot / (norm(actual) * norm(expected));
    let scale = expected
        .iter()
        .fold(0.0f32, |maximum, &value| maximum.max(value.abs()));
    let worst = actual
        .iter()
        .zip(expected)
        .fold(0.0f32, |maximum, (&left, &right)| {
            maximum.max((left - right).abs())
        });
    eprintln!("{name}: cosine {cosine:.7}, max error {worst:.3e}, scale {scale:.3}");
    assert!(
        cosine > 0.999_999 && worst <= scale * 1e-4,
        "{name}: cosine {cosine}, max error {worst}, scale {scale}"
    );
}

fn check_forward() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let dit = MetalDit::load(&file, "weight.").unwrap();
    let inputs = DitInputs {
        video: unbatched(tensor(&file, "input.video")),
        audio: unbatched(tensor(&file, "input.audio")),
        context: unbatched(tensor(&file, "input.context")),
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references: Vec::new(),
        sigma: tensor(&file, "input.timestep").data[0] / 1000.0,
        shift_video: metadata_number(&file, "shift_video"),
        shift_audio: metadata_number(&file, "shift_audio"),
    };

    let outputs = dit.forward(&inputs, &[0, 1], None).unwrap();
    assert_close(
        "text states",
        &outputs.text_states,
        &tensor(&file, "intermediate.text_states").data,
    );
    for (index, block) in &outputs.blocks {
        assert_close(
            &format!("block {index}"),
            block,
            &tensor(&file, &format!("intermediate.block.{index}")).data,
        );
    }

    assert_close("video", &outputs.video, &tensor(&file, "output.video").data);
    assert_close("audio", &outputs.audio, &tensor(&file, "output.audio").data);

    // Repeated calls produce the same output.
    let repeated = dit.forward(&inputs, &[], None).unwrap();
    assert!(repeated.video == outputs.video && repeated.audio == outputs.audio);
    let context = inputs.context.clone();
    let prepared = dit.prepare_text(&context).unwrap();
    let cached = prepared.forward(&inputs, &[], None).unwrap();
    assert_eq!(cached.video, outputs.video);
    assert_eq!(cached.audio, outputs.audio);
    assert!(cached.text_states.is_empty());
    let mut changed = inputs;
    changed.sigma *= 0.5;
    let expected = dit.forward(&changed, &[], None).unwrap();
    let actual = prepared.forward(&changed, &[], None).unwrap();
    assert_eq!(actual.video, expected.video);
    assert_eq!(actual.audio, expected.audio);
    changed.context.data[0] += 1.0;
    assert!(prepared.forward(&changed, &[], None).is_err());
}

#[test]
fn matches_comfyui_golden_forward() {
    check_forward();
}

/// Frames `[first, first + count)` of a latent `[channels, frames, ...]`.
fn frames(latent: &Tensor, first: usize, count: usize) -> Tensor {
    let (channels, total) = (latent.shape[0], latent.shape[1]);
    let frame_values: usize = latent.shape[2..].iter().product();
    let mut data = Vec::with_capacity(channels * count * frame_values);
    for channel in 0..channels {
        let start = (channel * total + first) * frame_values;
        data.extend_from_slice(&latent.data[start..start + count * frame_values]);
    }

    let mut shape = latent.shape.clone();
    shape[1] = count;
    Tensor::new(shape, data)
}

/// The CPU reference's outputs for `inputs` with the fixture's weights.
fn cpu_forward(file: &SafeTensors, inputs: &DitInputs) -> mmh3_cpu::dit::DitTrace {
    use mmh3_core::dit::config::DitConfig;
    use std::collections::HashMap;

    let mut tensors = HashMap::new();
    for info in file.tensors() {
        if let Some(name) = info.name.strip_prefix("weight.") {
            tensors.insert(name.to_owned(), Tensor::load(file, info).unwrap());
        }
    }

    let config =
        DitConfig::from_shapes(|name| tensors.get(name).map(|tensor| tensor.shape.clone()))
            .unwrap();
    mmh3_cpu::dit::forward(&mmh3_cpu::dit::DitWeights::new(tensors), &config, inputs)
}

/// The largest difference between the video outputs of two calls.
fn video_change(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

/// The first `count` frames of a stereo audio latent `[channels, 2, frames]`.
fn audio_frames(audio: &Tensor, count: usize) -> Tensor {
    let mut shape = audio.shape.clone();
    shape[2] = count;
    let data = (0..audio.shape[0] * 2)
        .flat_map(|row| audio.data[row * audio.shape[2]..][..count].to_vec())
        .collect();
    Tensor::new(shape, data)
}

#[test]
fn follows_the_cpu_reference_with_keyframes() {
    use mmh3_core::dit::inputs::Keyframe;
    use mmh3_core::dit::timestep::Modality;

    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let video = unbatched(tensor(&file, "input.video"));
    let audio = unbatched(tensor(&file, "input.audio"));
    let context = unbatched(tensor(&file, "input.context"));
    // Keyframes made from the input's own latents: the first frame, and the last latent frame with
    // the first audio frames, anchored at the last pixel frame.
    let latent_frames = video.shape[1];
    let pixel_frames = (latent_frames - 2) / 5 * 17 + 5;
    let keyframes = vec![
        Keyframe {
            frame_index: 0,
            video: Some(frames(&video, 0, 1)),
            audio: None,
        },
        Keyframe {
            frame_index: pixel_frames - 1,
            video: Some(frames(&video, latent_frames - 1, 1)),
            audio: Some(audio_frames(&audio, 3)),
        },
    ];
    let context_modalities: Vec<Modality> = (0..context.shape[0])
        .map(|token| {
            if (2..6).contains(&token) {
                Modality::Video
            } else {
                Modality::Text
            }
        })
        .collect();
    let inputs = DitInputs {
        video,
        audio,
        context,
        context_modalities,
        keyframes,
        references: Vec::new(),
        sigma: 0.6,
        shift_video: metadata_number(&file, "shift_video"),
        shift_audio: metadata_number(&file, "shift_audio"),
    };

    let expected = cpu_forward(&file, &inputs);

    let dit = MetalDit::load(&file, "weight.").unwrap();
    let outputs = dit.forward(&inputs, &[], None).unwrap();
    assert_close("video", &outputs.video, &expected.video);
    assert_close("audio", &outputs.audio, &expected.audio);

    // The keyframes change the result.
    let plain = DitInputs {
        keyframes: Vec::new(),
        context_modalities: Vec::new(),
        ..inputs
    };

    let without = dit.forward(&plain, &[], None).unwrap();
    let change = video_change(&without.video, &outputs.video);
    assert!(change > 1e-3, "keyframes change the video by only {change}");
}

#[test]
fn follows_the_cpu_reference_with_references() {
    use mmh3_core::dit::inputs::Reference;

    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let video = unbatched(tensor(&file, "input.video"));
    let audio = unbatched(tensor(&file, "input.audio"));
    let context = unbatched(tensor(&file, "input.context"));
    // A picture on a grid of its own, 2 × 8 latents from the input's values, a clip of the
    // input's first two latent frames with a soundtrack, and a sound.
    let channels = video.shape[0];
    let picture = Tensor::new(
        vec![channels, 1, 2, 8],
        video.data[..channels * 2 * 8].to_vec(),
    );
    let references = vec![
        Reference::Picture(picture),
        Reference::Video {
            video: frames(&video, 0, 2),
            audio: Some(audio_frames(&audio, 4)),
        },
        Reference::Audio(audio_frames(&audio, 5)),
    ];
    let inputs = DitInputs {
        video,
        audio,
        context,
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references,
        sigma: 0.6,
        shift_video: metadata_number(&file, "shift_video"),
        shift_audio: metadata_number(&file, "shift_audio"),
    };

    let expected = cpu_forward(&file, &inputs);

    let dit = MetalDit::load(&file, "weight.").unwrap();
    let outputs = dit.forward(&inputs, &[], None).unwrap();
    assert_close("video", &outputs.video, &expected.video);
    assert_close("audio", &outputs.audio, &expected.audio);

    let plain = DitInputs {
        references: Vec::new(),
        ..inputs
    };

    let without = dit.forward(&plain, &[], None).unwrap();
    let change = video_change(&without.video, &outputs.video);
    assert!(
        change > 1e-3,
        "references change the video by only {change}"
    );
}

#[test]
fn adapter_lora_matches_a_cpu_weight_update() {
    use mmh3_core::{dit::config::DitConfig, safetensors::write_f32};
    use std::collections::HashMap;
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let inputs = DitInputs {
        video: unbatched(tensor(&file, "input.video")),
        audio: unbatched(tensor(&file, "input.audio")),
        context: unbatched(tensor(&file, "input.context")),
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references: Vec::new(),
        sigma: 0.6,
        shift_video: 12.0,
        shift_audio: 3.0,
    };

    let mut tensors: HashMap<_, _> = file
        .tensors()
        .iter()
        .filter_map(|info| {
            info.name
                .strip_prefix("weight.")
                .map(|name| (name.to_owned(), Tensor::load(&file, info).unwrap()))
        })
        .collect();
    let weight = tensors.get_mut("blocks.0.attn.out_proj.weight").unwrap();
    let (outputs, features, rank) = (weight.shape[0], weight.shape[1], 3);
    let down: Vec<f32> = (0..rank * features)
        .map(|i| (i as f32 * 0.37).sin() * 0.03)
        .collect();
    let up: Vec<f32> = (0..outputs * rank)
        .map(|i| (i as f32 * 0.13).cos() * 0.04)
        .collect();
    let (alpha, strength) = (2.0, 0.6);
    let path = std::env::temp_dir().join(format!(
        "mmh3-metal-lora-{}.safetensors",
        std::process::id()
    ));
    write_f32(
        &path,
        &[
            (
                "diffusion_model.blocks.0.attn.out_proj.lora_A.weight",
                &[rank, features],
                &down,
            ),
            (
                "diffusion_model.blocks.0.attn.out_proj.lora_B.weight",
                &[outputs, rank],
                &up,
            ),
            (
                "diffusion_model.blocks.0.attn.out_proj.alpha",
                &[],
                &[alpha],
            ),
        ],
        &[],
    )
    .unwrap();
    let lora = SafeTensors::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut dit = MetalDit::load(&file, "weight.").unwrap();
    let original = dit.forward(&inputs, &[], None).unwrap();
    assert_eq!(dit.add_lora(&lora, strength).unwrap(), 1);
    let actual = dit.forward(&inputs, &[], None).unwrap();

    for o in 0..outputs {
        for i in 0..features {
            weight.data[o * features + i] += (0..rank)
                .map(|r| up[o * rank + r] * down[r * features + i])
                .sum::<f32>()
                * strength
                * alpha
                / rank as f32;
        }
    }

    let config = DitConfig::from_shapes(|name| tensors.get(name).map(|t| t.shape.clone())).unwrap();
    let expected =
        mmh3_cpu::dit::forward(&mmh3_cpu::dit::DitWeights::new(tensors), &config, &inputs);
    assert_close("LoRA video", &actual.video, &expected.video);
    assert_close("LoRA audio", &actual.audio, &expected.audio);
    assert!(video_change(&original.video, &actual.video) > 1e-4);
    assert!(dit.add_lora(&lora, strength).is_err());
}
