#![cfg(target_os = "macos")]

//! The Metal vision tower on a tiny random tower, and against ComfyUI's on the real checkpoint.

use mmh3_core::safetensors::{SafeTensors, write_f32};
use mmh3_core::tensor::Tensor;
use mmh3_core::vision::PATCH_VALUES;
use mmh3_metal::{Device, vision::MetalVisionEncoder};
use std::path::Path;
use std::time::Instant;

/// The tensors of a vision tower, each of smooth values around `offset`.
#[derive(Default)]
struct Tower(Vec<(String, Tensor)>);

impl Tower {
    fn add(&mut self, name: &str, shape: Vec<usize>, scale: f32, offset: f32) {
        let salt = self.0.len();
        let count = shape.iter().product::<usize>();
        let data = (0..count)
            .map(|i| offset + scale * ((i * 7 + salt * 13) as f32 * 0.37).sin())
            .collect();
        self.0
            .push((format!("visual.{name}"), Tensor::new(shape, data)));
    }

    fn linear(&mut self, name: &str, outputs: usize, inputs: usize) {
        let scale = 1.0 / (inputs as f32).sqrt();
        self.add(&format!("{name}.weight"), vec![outputs, inputs], scale, 0.0);
        self.add(&format!("{name}.bias"), vec![outputs], 0.1, 0.0);
    }

    fn norm(&mut self, name: &str, width: usize) {
        self.add(&format!("{name}.weight"), vec![width], 0.2, 1.0);
        self.add(&format!("{name}.bias"), vec![width], 0.1, 0.0);
    }
}

/// A vision tower of one 72-wide head, 25 blocks so that blocks 8, 16 and 24 feed DeepStack, and
/// an output of 40.
fn tiny_tower() -> Vec<(String, Tensor)> {
    let (hidden, ffn, output) = (72, 96, 40);
    let mut tower = Tower::default();
    let scale = 1.0 / (PATCH_VALUES as f32).sqrt();
    tower.add(
        "patch_embed.proj.weight",
        vec![hidden, 3, 2, 16, 16],
        scale,
        0.0,
    );
    tower.add("patch_embed.proj.bias", vec![hidden], 0.1, 0.0);
    tower.add("pos_embed.weight", vec![2304, hidden], 0.5, 0.0);
    for block in 0..25 {
        let p = format!("blocks.{block}");
        tower.norm(&format!("{p}.norm1"), hidden);
        tower.linear(&format!("{p}.attn.qkv"), 3 * hidden, hidden);
        tower.linear(&format!("{p}.attn.proj"), hidden, hidden);
        tower.norm(&format!("{p}.norm2"), hidden);
        tower.linear(&format!("{p}.mlp.linear_fc1"), ffn, hidden);
        tower.linear(&format!("{p}.mlp.linear_fc2"), hidden, ffn);
    }
    // The main merger norms each patch, the DeepStack mergers 2 × 2 joined patches.
    for (name, norm_width) in [
        ("merger", hidden),
        ("deepstack_merger_list.0", 4 * hidden),
        ("deepstack_merger_list.1", 4 * hidden),
        ("deepstack_merger_list.2", 4 * hidden),
    ] {
        tower.norm(&format!("{name}.norm"), norm_width);
        tower.linear(&format!("{name}.linear_fc1"), 4 * hidden, 4 * hidden);
        tower.linear(&format!("{name}.linear_fc2"), output, 4 * hidden);
    }
    tower.0
}

fn picture(height: usize, width: usize, salt: f32) -> Tensor {
    let data = (0..height * width * 3)
        .map(|i| 0.5 + 0.45 * (i as f32 * 0.013 + salt).sin())
        .collect();
    Tensor::new(vec![height, width, 3], data)
}

#[test]
fn tiny_tower_embeds_pictures_and_pairs_of_frames() {
    let tensors = tiny_tower();
    let path = std::env::temp_dir().join(format!(
        "mmh3-metal-vision-{}.safetensors",
        std::process::id()
    ));
    let write = |tensors: &[(String, Tensor)]| {
        write_f32(
            &path,
            &tensors
                .iter()
                .map(|(n, t)| (n.as_str(), t.shape.as_slice(), t.data.as_slice()))
                .collect::<Vec<_>>(),
            &[],
        )
        .unwrap();
        SafeTensors::open(&path).unwrap()
    };
    let encoder = MetalVisionEncoder::load(&write(&tensors)).unwrap();
    // Without the DeepStack mergers the tower is not Qwen3-VL's.
    let without: Vec<_> = tensors
        .iter()
        .filter(|(name, _)| !name.starts_with("visual.deepstack_merger_list.2."))
        .cloned()
        .collect();
    assert!(MetalVisionEncoder::load(&write(&without)).is_err());
    std::fs::remove_file(&path).unwrap();
    assert_eq!(encoder.output_width(), 40);

    let first = picture(64, 96, 0.0);
    let embeddings = encoder.encode(&first).unwrap();
    assert_eq!(embeddings.tokens, 2 * 3);
    assert_eq!(embeddings.merged.shape(), [6, 40]);
    assert_eq!(embeddings.deepstack.len(), 3);
    let merged = embeddings.merged.to_f32().unwrap();
    assert!(merged.iter().all(|value| value.is_finite()));
    for features in &embeddings.deepstack {
        assert_eq!(features.shape(), [6, 40]);
        assert_ne!(features.to_f32().unwrap(), merged);
    }
    // A picture is a pair of frames that are both the picture.
    let pair = encoder.encode_frames(&first, &first).unwrap();
    assert_eq!(pair.merged.to_f32().unwrap(), merged);
    let second = picture(64, 96, 1.0);
    let moving = encoder.encode_frames(&first, &second).unwrap();
    assert_ne!(moving.merged.to_f32().unwrap(), merged);
    // The attention sees the whole picture, so every token depends on every patch.
    let mut changed = first.clone();
    changed.data[0] += 0.3;
    let other = encoder.encode(&changed).unwrap().merged.to_f32().unwrap();
    for (token, (a, b)) in merged.chunks(40).zip(other.chunks(40)).enumerate() {
        assert_ne!(a, b, "token {token} does not see the first patch");
    }
}

/// Cosine similarity and the largest error relative to the largest reference value.
fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let dot: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let norm = |values: &[f32]| {
        values
            .iter()
            .map(|&v| v as f64 * v as f64)
            .sum::<f64>()
            .sqrt()
    };
    let scale = expected.iter().fold(0f32, |m, &v| m.max(v.abs())) as f64;
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (a as f64 - b as f64).abs())
        .fold(0.0, f64::max);
    (dot / (norm(actual) * norm(expected)), worst / scale)
}

/// Compares the tower on the text encoder checkpoint `MMH3_TEXT_ENCODER` with ComfyUI's outputs
/// in `MMH3_VISION_GOLDEN`, which tools/golden/vision.py writes. Only the tower is read.
#[test]
#[ignore = "needs the text encoder checkpoint and a golden file from tools/golden/vision.py"]
fn matches_comfyui_on_the_checkpoint() {
    let variable = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("set {name}"));
    let checkpoint = SafeTensors::open(Path::new(&variable("MMH3_TEXT_ENCODER"))).unwrap();
    let golden = SafeTensors::open(Path::new(&variable("MMH3_VISION_GOLDEN"))).unwrap();
    let tensor = |name: &str| Tensor::load(&golden, golden.get(name).unwrap()).unwrap();

    let encoder = MetalVisionEncoder::load(&checkpoint).unwrap();
    let first = tensor("picture.0");
    let encode = || match golden.get("picture.1") {
        Some(_) => encoder.encode_frames(&first, &tensor("picture.1")),
        None => encoder.encode(&first),
    };
    // The first run compiles and warms up the kernels.
    encode().unwrap();
    let started = Instant::now();
    let embeddings = encode().unwrap();
    let merged = embeddings.merged.to_f32().unwrap();
    println!(
        "{:?} picture in {:.3} s, {:.2} GB on the device at most",
        first.shape,
        started.elapsed().as_secs_f64(),
        Device::shared().unwrap().stats().peak_allocated_bytes as f64 / 1e9
    );
    let (cosine, worst) = compare(&merged, &tensor("merged").data);
    println!(
        "merged: 1 − cosine {:.1e}, max error {worst:.2e} of the largest value",
        1.0 - cosine
    );
    assert!(cosine > 0.99999 && worst < 1e-3);
    for (index, features) in embeddings.deepstack.iter().enumerate() {
        let (cosine, worst) = compare(
            &features.to_f32().unwrap(),
            &tensor(&format!("deepstack.{index}")).data,
        );
        println!(
            "deepstack {index}: 1 − cosine {:.1e}, max error {worst:.2e} of the largest value",
            1.0 - cosine
        );
        assert!(cosine > 0.99999 && worst < 1e-3);
    }
}
