//! Compares the reference DiT with golden data from ComfyUI's implementation
//! (tools/golden/dit_tiny.py).

use mmh3_core::dit::config::DitConfig;
use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cpu::dit::{DitWeights, forward};
use std::collections::HashMap;
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

/// Drops the leading batch dimension of size one.
fn unbatched(mut tensor: Tensor) -> Tensor {
    assert_eq!(tensor.shape[0], 1);
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
    eprintln!("{name}: cosine {cosine:.9}, max error {worst:.3e}, scale {scale:.3}");
    assert!(
        cosine > 0.999_999 && worst <= scale * 1e-4,
        "{name}: cosine {cosine}, max error {worst}, scale {scale}"
    );
}

#[test]
fn matches_comfyui_golden_forward() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut tensors = HashMap::new();
    for info in file.tensors() {
        if let Some(name) = info.name.strip_prefix("weight.") {
            tensors.insert(name.to_owned(), Tensor::load(&file, info).unwrap());
        }
    }
    let config =
        DitConfig::from_shapes(|name| tensors.get(name).map(|tensor| tensor.shape.clone()))
            .unwrap();
    assert_eq!(
        (
            config.hidden,
            config.heads,
            config.layers,
            config.refiner_layers
        ),
        (192, 2, 2, 1)
    );

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
    let trace = forward(&DitWeights::new(tensors), &config, &inputs);

    assert_close(
        "text states",
        &trace.text_states,
        &tensor(&file, "intermediate.text_states").data,
    );
    for (index, output) in trace.block_outputs.iter().enumerate() {
        assert_close(
            &format!("block {index}"),
            output,
            &tensor(&file, &format!("intermediate.block.{index}")).data,
        );
    }
    assert_close("video", &trace.video, &tensor(&file, "output.video").data);
    assert_close("audio", &trace.audio, &tensor(&file, "output.audio").data);
}
