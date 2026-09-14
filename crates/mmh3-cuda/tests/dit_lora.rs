//! Applies a random LoRA to the tiny DiT on the GPU and compares it with the FP32 reference on
//! merged weights.

use mmh3_core::dit::config::DitConfig;
use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::dit::{CudaDit, LoraMode};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dit_tiny.safetensors"
);
const STRENGTH: f32 = 0.8;
const ALPHA: f32 = 8.0;

struct Random(u64);

impl Random {
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        low + (high - low) * ((self.0 >> 40) as f32 / (1u64 << 24) as f32)
    }
}

/// Writes tensors to a safetensors file: (name, dtype, shape, little-endian bytes).
fn write_safetensors(path: &Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut entries = Vec::new();
    let mut offset = 0;
    for (name, dtype, shape, bytes) in tensors {
        let dimensions: Vec<String> = shape.iter().map(ToString::to_string).collect();
        entries.push(format!(
            r#""{name}": {{"dtype": "{dtype}", "shape": [{}], "data_offsets": [{offset}, {}]}}"#,
            dimensions.join(", "),
            offset + bytes.len()
        ));
        offset += bytes.len();
    }
    let mut header = format!("{{{}}}", entries.join(", ")).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    for (_, _, _, bytes) in tensors {
        file.extend(bytes);
    }
    std::fs::write(path, file).unwrap();
}

fn unbatched(file: &SafeTensors, name: &str) -> Tensor {
    let mut tensor = Tensor::load(file, file.get(name).unwrap()).unwrap();
    tensor.shape.remove(0);
    tensor
}

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
        .collect()
}

fn relative_error(actual: &[f32], expected: &[f32]) -> f64 {
    let difference: f64 = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| (left as f64 - right as f64).powi(2))
        .sum();
    let norm: f64 = expected.iter().map(|&value| (value as f64).powi(2)).sum();
    (difference / norm).sqrt()
}

#[test]
fn applies_a_lora_like_merged_weights() {
    let file = SafeTensors::open(Path::new(FIXTURE)).unwrap();
    let mut weights: HashMap<String, Tensor> = file
        .tensors()
        .iter()
        .filter_map(|info| {
            Some((
                info.name.strip_prefix("weight.")?.to_owned(),
                Tensor::load(&file, info).unwrap(),
            ))
        })
        .collect();
    let config =
        DitConfig::from_shapes(|name| weights.get(name).map(|tensor| tensor.shape.clone()))
            .unwrap();
    let inputs = DitInputs {
        video: unbatched(&file, "input.video"),
        audio: unbatched(&file, "input.audio"),
        context: unbatched(&file, "input.context"),
        sigma: Tensor::load(&file, file.get("input.timestep").unwrap())
            .unwrap()
            .data[0]
            / 1000.0,
        shift_video: 12.0,
        shift_audio: 3.0,
    };
    let base = mmh3_cpu::dit::forward(
        &mmh3_cpu::dit::DitWeights::new(weights.clone()),
        &config,
        &inputs,
    );

    // A fused qkv layer with three times the rank, as the Turbo LoRAs have, an FFN input, an FFN
    // output and a refiner layer, plus tensors replaced as they are.
    let layers = [
        ("blocks.0.attn.qkv_proj", 12),
        ("blocks.1.mlp.fc1", 4),
        ("blocks.1.mlp.fc2", 4),
        ("token_refiner.blocks.0.attn.out_proj", 4),
    ];
    let mut random = Random(5);
    let mut lora = Vec::new();
    for (layer, rank) in layers {
        let weight = weights.get_mut(&format!("{layer}.weight")).unwrap();
        let (outputs, inputs) = (weight.shape[0], weight.shape[1]);
        let mut sample = |count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| bf16_to_f32(f32_to_bf16(random.uniform(-0.5, 0.5))))
                .collect()
        };
        let (down, up) = (sample(rank * inputs), sample(outputs * rank));
        let scale = STRENGTH * ALPHA / rank as f32;
        for output in 0..outputs {
            for input in 0..inputs {
                let delta: f32 = (0..rank)
                    .map(|index| up[output * rank + index] * down[index * inputs + input])
                    .sum();
                weight.data[output * inputs + input] += scale * delta;
            }
        }
        lora.push((
            format!("diffusion_model.{layer}.lora_A.weight"),
            "BF16",
            vec![rank, inputs],
            bf16(&down),
        ));
        lora.push((
            format!("diffusion_model.{layer}.lora_B.weight"),
            "BF16",
            vec![outputs, rank],
            bf16(&up),
        ));
        lora.push((
            format!("diffusion_model.{layer}.alpha"),
            "F32",
            vec![],
            ALPHA.to_le_bytes().to_vec(),
        ));
    }
    // Tensors that the file replaces as they are: a norm weight and the AdaLN curve table.
    let norm = weights.get_mut("blocks.1.norm2.weight").unwrap();
    for value in &mut norm.data {
        *value = bf16_to_f32(f32_to_bf16(*value * random.uniform(0.5, 1.5)));
    }
    lora.push((
        "diffusion_model.blocks.1.norm2.weight".to_owned(),
        "BF16",
        norm.shape.clone(),
        bf16(&norm.data),
    ));
    let table = weights.get_mut("adaln_t_table").unwrap();
    for value in &mut table.data {
        *value *= random.uniform(0.8, 1.2);
    }
    lora.push((
        "diffusion_model.adaln_t_table".to_owned(),
        "F32",
        table.shape.clone(),
        table
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    ));
    let merged = mmh3_cpu::dit::forward(&mmh3_cpu::dit::DitWeights::new(weights), &config, &inputs);
    let change = relative_error(&merged.video, &base.video);
    assert!(
        change > 0.05,
        "the LoRA changes the video velocity by only {change}"
    );

    let path: PathBuf =
        std::env::temp_dir().join(format!("mmh3-dit-lora-{}.safetensors", std::process::id()));
    write_safetensors(&path, &lora);
    let mut dit = CudaDit::load(&file, "weight.").unwrap();
    assert_eq!(
        dit.add_lora(
            &SafeTensors::open(&path).unwrap(),
            STRENGTH,
            LoraMode::Adapter
        )
        .unwrap(),
        layers.len() + 2
    );
    std::fs::remove_file(&path).unwrap();
    let outputs = dit.forward(&inputs, &[], None).unwrap();

    for (name, actual, expected) in [
        ("video", &outputs.video, &merged.video),
        ("audio", &outputs.audio, &merged.audio),
    ] {
        let error = relative_error(actual, expected);
        eprintln!(
            "{name}: relative error {error:.3e} against merged weights, {:.3e} without the LoRA",
            relative_error(
                actual,
                if name == "video" {
                    &base.video
                } else {
                    &base.audio
                }
            )
        );
        assert!(error < 0.02, "{name}: relative error {error}");
    }
}
