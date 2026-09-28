#![cfg(target_os = "macos")]
use mmh3_core::dit::timestep::Modality;
use mmh3_core::vision::VisionPrompt;
use mmh3_core::{
    safetensors::{SafeTensors, write_f32},
    tensor::Tensor,
};
use mmh3_metal::{Device, ops::Array, text_encoder::MetalTextEncoder, vision::VisionEmbeddings};

/// A language tower of two layers, 32 wide, with 8 tokens, and its embedding table. `name` keeps
/// the checkpoints of tests that run at once apart.
fn tiny_encoder(name: &str) -> (MetalTextEncoder, Vec<f32>) {
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
        "mmh3-metal-text-{name}-{}.safetensors",
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
    // The encoder reads its weights from the file, so it goes once they are loaded.
    let encoder = MetalTextEncoder::load(&file).unwrap();
    std::fs::remove_file(&path).unwrap();
    (encoder, tensors[0].1.data.clone())
}

#[test]
fn language_tower_preserves_causal_prefixes_and_captures_final_hidden_states() {
    let (encoder, _) = tiny_encoder("causal");
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

#[test]
fn pictures_take_their_rows_positions_and_deepstack_features() {
    let (encoder, table) = tiny_encoder("pictures");
    let device = Device::shared().unwrap();
    let ids = [2, 1, 7, 7, 7, 5, 3];
    let (start, tokens) = (2, 3);
    let text_positions: Vec<[usize; 3]> = (0..ids.len()).map(|t| [t; 3]).collect();
    let prompt = |positions: Vec<[usize; 3]>| VisionPrompt {
        ids: ids.to_vec(),
        picture_starts: vec![start],
        modalities: vec![Modality::Text; ids.len()],
        positions,
    };
    let rows = |values: &[f32]| Array::from_f32(&device, tokens, 32, values).unwrap();
    let picture = |merged: &[f32], deepstack: &[Vec<f32>]| VisionEmbeddings {
        merged: rows(merged),
        deepstack: deepstack.iter().map(|values| rows(values)).collect(),
        tokens,
    };
    let text = encoder.encode(&ids, &[0, 1]).unwrap();

    // Without pictures, a prompt is its text.
    let plain = VisionPrompt {
        picture_starts: Vec::new(),
        ..prompt(text_positions.clone())
    };
    let encoded = encoder.encode_prompt(&plain, &[], &[0, 1]).unwrap();
    assert_eq!(encoded.context.data, text.context.data);
    assert!(
        encoder
            .encode_prompt(&plain, &[picture(&[0.0; 96], &[])], &[])
            .is_err()
    );

    // A picture whose embeddings are its tokens' own and whose features are zeros changes
    // nothing.
    let own = &table[7 * 32..8 * 32].repeat(tokens);
    let zeros = vec![vec![0.0; tokens * 32]; 3];
    let same = encoder
        .encode_prompt(
            &prompt(text_positions.clone()),
            &[picture(own, &zeros)],
            &[0, 1],
        )
        .unwrap();
    assert_eq!(same.context.data, text.context.data);

    // Other embeddings change the picture's rows and the ones after them, but not the ones
    // before, which attend to nothing later.
    let other: Vec<f32> = (0..tokens * 32).map(|i| (i as f32 * 0.7).cos()).collect();
    let embedded = encoder
        .encode_prompt(
            &prompt(text_positions.clone()),
            &[picture(&other, &zeros)],
            &[0],
        )
        .unwrap();
    let before = start * 32;
    assert_eq!(embedded.context.data[..before], text.context.data[..before]);
    for row in start..ids.len() {
        let range = row * 32..(row + 1) * 32;
        assert_ne!(
            embedded.context.data[range.clone()],
            text.context.data[range]
        );
    }

    // A layer's DeepStack features add to the picture's rows after that layer is captured.
    let features: Vec<Vec<f32>> = (0..3)
        .map(|layer| {
            (0..tokens * 32)
                .map(|i| ((i + 5 * layer) as f32 * 0.3).sin())
                .collect()
        })
        .collect();
    let stacked = encoder
        .encode_prompt(
            &prompt(text_positions.clone()),
            &[picture(&other, &features)],
            &[0, 1],
        )
        .unwrap();
    assert_eq!(stacked.layers[0].1.data, embedded.layers[0].1.data);
    // Layer 1 sees layer 0's features, and the context takes those of layer 1 on top.
    let second = &stacked.layers[1].1.data;
    assert_eq!(second[..before], embedded.context.data[..before]);
    assert_ne!(second[before..], embedded.context.data[before..]);
    let after = (start + tokens) * 32;
    let context = &stacked.context.data;
    for (index, (&with, &without)) in context[before..after]
        .iter()
        .zip(&second[before..after])
        .enumerate()
    {
        let expected = without + features[1][index];
        assert!((with - expected).abs() < 1e-6, "{with} != {expected}");
    }
    assert_eq!(context[after..], second[after..]);

    // The picture's rows spread over the height and width axes of the rotary positions.
    let mut positions = text_positions;
    for (index, position) in positions[start..start + tokens].iter_mut().enumerate() {
        *position = [start, start + index / 2, start + index % 2];
    }
    let placed = encoder
        .encode_prompt(&prompt(positions), &[picture(&other, &zeros)], &[])
        .unwrap();
    assert_eq!(placed.context.data[..before], text.context.data[..before]);
    // The picture's first token sits at [2, 2, 2] either way.
    assert_eq!(
        placed.context.data[before..before + 32],
        embedded.context.data[before..before + 32]
    );
    assert_ne!(
        placed.context.data[before + 32..],
        embedded.context.data[before + 32..]
    );
}
