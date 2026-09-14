//! Runs the CUDA DiT on the tiny golden fixture from tools/golden/dit_tiny.py.

use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::attention::AttentionPrecision;
use mmh3_cuda::dit::CudaDit;
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
        cosine > 0.9995 && worst <= scale * 0.03,
        "{name}: cosine {cosine}, max error {worst}, scale {scale}"
    );
}

fn check_forward(precision: AttentionPrecision) {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut dit = CudaDit::load(&file, "weight.").unwrap();
    dit.set_attention_precision(precision);
    let inputs = DitInputs {
        video: unbatched(tensor(&file, "input.video")),
        audio: unbatched(tensor(&file, "input.audio")),
        context: unbatched(tensor(&file, "input.context")),
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
}

#[test]
fn matches_comfyui_golden_forward() {
    check_forward(AttentionPrecision::Bf16);
}

#[test]
fn quantized_matches_golden_with_the_same_bounds() {
    check_forward(AttentionPrecision::Int8Fp8);
}
