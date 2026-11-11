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

/// The sigmas MiniMax H3 Image's bank of output heads was distilled on, one per head.
const IMAGE_SIGMAS: [f32; 6] = [
    1.0,
    0.967_523_7,
    0.887_548_5,
    0.75,
    0.532_773_9,
    0.232_008_87,
];

/// A tensor as a checkpoint stores it.
struct Stored {
    name: String,
    dtype: mmh3_core::safetensors::DType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// A checkpoint written for a test, which the DiT reads from its path while it loads.
struct Checkpoint {
    file: SafeTensors,
    path: std::path::PathBuf,
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn write_checkpoint(tensors: &[Stored], metadata: &[(&str, &str)]) -> Checkpoint {
    use mmh3_core::json::quote;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut entries = vec![format!(
        "\"__metadata__\":{{{}}}",
        metadata
            .iter()
            .map(|(key, value)| format!("{}:{}", quote(key), quote(value)))
            .collect::<Vec<_>>()
            .join(",")
    )];
    let mut offset = 0;
    for tensor in tensors {
        let end = offset + tensor.bytes.len();
        entries.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{offset},{end}]}}",
            quote(&tensor.name),
            tensor.dtype.name(),
            tensor.shape
        ));
        offset = end;
    }
    let mut header = format!("{{{}}}", entries.join(","));
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }

    static WRITTEN: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "mmh3-metal-dit-image-{}-{}.safetensors",
        std::process::id(),
        WRITTEN.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for tensor in tensors {
        file.write_all(&tensor.bytes).unwrap();
    }
    Checkpoint {
        file: SafeTensors::open(&path).unwrap(),
        path,
    }
}

/// The fixture's weights as MiniMax H3 Image's checkpoint holds its own: without the audio stream,
/// with BF16 AdaLN biases, with ComfyUI's attention hint beside a block, and with a bank of
/// `heads` output heads, the first the fixture's and the others stored as offsets from it, with
/// their sigmas in the metadata when `sigmas` is set. With `merged`, the checkpoint instead has
/// the one head that is the first plus the offset of head `merged`.
fn image_checkpoint(heads: usize, sigmas: bool, merged: Option<usize>) -> Checkpoint {
    use mmh3_core::numeric::f32_to_bf16;
    use mmh3_core::safetensors::DType;

    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut tensors = Vec::new();
    for info in file.tensors() {
        let Some(name) = info.name.strip_prefix("weight.") else {
            continue;
        };
        if name.starts_with("audio_patch_proj.") || name.starts_with("final_layer.audio_out.") {
            continue;
        }
        let tensor = Tensor::load(&file, info).unwrap();
        let (dtype, shape, values) = if name.starts_with("final_layer.video_out.") {
            let size = tensor.data.len();
            let offset = |head: usize, index: usize| {
                ((index * 7 + head * 13) as f32 * 0.61).sin() * 0.05 * head as f32
            };
            let first = &tensor.data;
            let bank: Vec<f32> = match merged {
                Some(head) => (0..size)
                    .map(|index| first[index] + offset(head, index))
                    .collect(),
                None => (0..heads)
                    .flat_map(|head| {
                        (0..size).map(move |index| match head {
                            0 => first[index],
                            _ => offset(head, index),
                        })
                    })
                    .collect(),
            };
            let mut shape = tensor.shape.clone();
            shape[0] *= bank.len() / size;
            let bytes = bank.iter().flat_map(|value| value.to_le_bytes()).collect();
            (DType::F32, shape, bytes)
        } else if name.ends_with("adaln_proj.linear.bias") {
            let bytes = tensor
                .data
                .iter()
                .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
                .collect();
            (DType::BF16, tensor.shape.clone(), bytes)
        } else {
            (info.dtype, info.shape.clone(), file.data(info).to_vec())
        };
        tensors.push(Stored {
            name: name.to_owned(),
            dtype,
            shape,
            bytes: values,
        });
    }
    let hint = br#"{"sage_attention": false}"#;
    tensors.push(Stored {
        name: "blocks.0.attn.comfy_attention.config".to_owned(),
        dtype: DType::U8,
        shape: vec![hint.len()],
        bytes: hint.to_vec(),
    });

    let config = format!(
        "{{\"transformer\": {{\"pdd_sigmas\": {:?}}}}}",
        IMAGE_SIGMAS.iter().chain(&[0.0]).collect::<Vec<_>>()
    );
    let named = [("config", config.as_str())];
    write_checkpoint(&tensors, if sigmas { &named } else { &[] })
}

/// One step of text-to-image at `sigma` on the fixture's first latent frame.
fn image_inputs(sigma: f32) -> DitInputs {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    DitInputs {
        video: frames(&unbatched(tensor(&file, "input.video")), 0, 1),
        audio: Tensor::new(vec![0, 2, 0], Vec::new()),
        context: unbatched(tensor(&file, "input.context")),
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references: Vec::new(),
        sigma,
        shift_video: 1.0,
        shift_audio: 1.0,
    }
}

#[test]
fn reads_an_image_checkpoint() {
    let checkpoint = image_checkpoint(6, true, None);
    let dit = MetalDit::load(&checkpoint.file, "").unwrap();
    let config = dit.config();
    assert_eq!(config.audio_channels, 0);
    assert_eq!(config.step_sigmas, IMAGE_SIGMAS);

    let unnamed = image_checkpoint(6, false, None);
    assert!(MetalDit::load(&unnamed.file, "").is_err());
}

#[test]
fn projects_each_step_with_its_merged_head() {
    let bank = MetalDit::load(&image_checkpoint(6, true, None).file, "").unwrap();
    let mut outputs = Vec::new();
    for head in [0, 3, 5] {
        let single = MetalDit::load(&image_checkpoint(1, false, Some(head)).file, "").unwrap();
        assert!(single.config().step_sigmas.is_empty());
        let inputs = image_inputs(IMAGE_SIGMAS[head]);
        let expected = single.forward(&inputs, &[], None).unwrap();
        let actual = bank.forward(&inputs, &[], None).unwrap();
        assert_close(&format!("head {head}"), &actual.video, &expected.video);
        assert!(actual.audio.is_empty());
        outputs.push(actual.video);
    }
    assert!(video_change(&outputs[0], &outputs[1]) > 1e-3);

    // A sigma between two of the bank's takes the nearer head.
    let near = bank.forward(&image_inputs(0.74), &[], None).unwrap();
    let single = MetalDit::load(&image_checkpoint(1, false, Some(3)).file, "").unwrap();
    let expected = single.forward(&image_inputs(0.74), &[], None).unwrap();
    assert_close("nearest head", &near.video, &expected.video);
}

#[test]
fn refuses_inputs_that_do_not_match_the_streams() {
    let image = MetalDit::load(&image_checkpoint(6, true, None).file, "").unwrap();
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut with_audio = image_inputs(0.5);
    with_audio.audio = unbatched(tensor(&file, "input.audio"));
    assert!(image.forward(&with_audio, &[], None).is_err());

    let video = MetalDit::load(&file, "weight.").unwrap();
    assert_eq!(video.config().audio_channels, 32);
    let mut silent = image_inputs(0.5);
    silent.audio = Tensor::new(vec![32, 2, 0], Vec::new());
    assert!(video.forward(&silent, &[], None).is_err());
}
