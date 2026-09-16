#![cfg(target_os = "macos")]
use mmh3_core::{
    safetensors::{SafeTensors, write_f32},
    tensor::Tensor,
};
use mmh3_metal::text_encoder::MetalTextEncoder;

#[test]
fn language_tower_preserves_causal_prefixes_and_captures_final_hidden_states() {
    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    let mut weight = |name: &str, rows: usize, cols: usize| {
        let salt = tensors.len();
        let data = (0..rows * cols)
            .map(|i| ((i * 17 + salt * 19) as f32 * 0.13).sin() / (cols as f32).sqrt())
            .collect();
        tensors.push((
            format!("model.{name}.weight"),
            Tensor::new(vec![rows, cols], data),
        ));
    };

    weight("embed_tokens", 8, 32);
    for layer in 0..2 {
        for (name, rows, cols) in [
            ("self_attn.q_proj", 256, 32),
            ("self_attn.k_proj", 128, 32),
            ("self_attn.v_proj", 128, 32),
            ("self_attn.o_proj", 32, 256),
            ("mlp.gate_proj", 48, 32),
            ("mlp.up_proj", 48, 32),
            ("mlp.down_proj", 32, 48),
        ] {
            weight(&format!("layers.{layer}.{name}"), rows, cols);
        }
    }

    for layer in 0..2 {
        for (name, cols) in [
            ("input_layernorm", 32),
            ("post_attention_layernorm", 32),
            ("self_attn.q_norm", 128),
            ("self_attn.k_norm", 128),
        ] {
            tensors.push((
                format!("model.layers.{layer}.{name}.weight"),
                Tensor::new(vec![cols], vec![1.0; cols]),
            ));
        }
    }

    let path = std::env::temp_dir().join(format!(
        "mmh3-metal-text-{}.safetensors",
        std::process::id()
    ));
    write_f32(
        &path,
        &tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t.shape.as_slice(), t.data.as_slice()))
            .collect::<Vec<_>>(),
        &[],
    )
    .unwrap();
    let file = SafeTensors::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let encoder = MetalTextEncoder::load(&file).unwrap();
    let prefix = encoder.encode(&[2, 1], &[]).unwrap();
    let full = encoder.encode(&[2, 1, 5, 3], &[0, 1]).unwrap();
    assert_eq!(full.context.shape, [4, 32]);
    assert_eq!(full.layers.len(), 2);
    assert_eq!(full.layers[1].1.data, full.context.data);

    for (&a, &b) in prefix.context.data.iter().zip(&full.context.data) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 2e-5,
            "causal prefix changed: {a} != {b}"
        );
    }

    assert!(
        full.layers[0]
            .1
            .data
            .iter()
            .zip(&full.context.data)
            .any(|(&a, &b)| (a - b).abs() > 1e-3)
    );
    assert!(encoder.encode(&[], &[]).is_err());
    assert!(encoder.encode(&[8], &[]).is_err());
}
