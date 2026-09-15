//! Text-to-video DiT forward pass: one model call on the packed `[text | audio | video]` sequence.

use mmh3_core::dit::config::DitConfig;
use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::dit::latent::{pack_audio, patchify_video, unpack_audio, unpatchify_video};
use mmh3_core::dit::layout::{PackedLayout, SegmentKind};
use mmh3_core::dit::timestep::StepTimesteps;
use mmh3_core::tensor::Tensor;
use std::collections::HashMap;

/// Weights keyed by checkpoint tensor name.
pub struct DitWeights {
    tensors: HashMap<String, Tensor>,
}

impl DitWeights {
    pub fn new(tensors: HashMap<String, Tensor>) -> Self {
        DitWeights { tensors }
    }

    fn get(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }
}

/// Outputs of one call and the intermediate states tests compare against.
pub struct DitTrace {
    pub text_states: Vec<f32>,
    pub block_outputs: Vec<Vec<f32>>,
    /// Velocity in the layout of the video input.
    pub video: Vec<f32>,
    /// Velocity in the layout of the audio input.
    pub audio: Vec<f32>,
}

/// output[row, o] = Σᵢ input[row, i] · weight[o, i] + bias[o].
fn linear(input: &[f32], weight: &Tensor, bias: Option<&Tensor>) -> Vec<f32> {
    let (outputs, width) = (weight.shape[0], weight.shape[1]);
    assert_eq!(
        input.len() % width,
        0,
        "input width does not match the weight"
    );
    let mut result = Vec::with_capacity(input.len() / width * outputs);
    for row in input.chunks_exact(width) {
        for (output, weight_row) in weight.data.chunks_exact(width).enumerate() {
            let dot: f64 = row
                .iter()
                .zip(weight_row)
                .map(|(&left, &right)| left as f64 * right as f64)
                .sum();
            result.push(dot as f32 + bias.map_or(0.0, |bias| bias.data[output]));
        }
    }
    result
}

fn rms_norm(data: &mut [f32], width: usize, weight: &[f32], epsilon: f32) {
    for row in data.chunks_exact_mut(width) {
        let mean_square = row
            .iter()
            .map(|&value| value as f64 * value as f64)
            .sum::<f64>()
            / width as f64;
        let inverse = 1.0 / (mean_square + epsilon as f64).sqrt();
        for (value, &scale) in row.iter_mut().zip(weight) {
            *value = (*value as f64 * inverse) as f32 * scale;
        }
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// Softmax attention over all tokens. q, k and v are `[tokens, heads, head_dim]`.
fn attention(query: &[f32], key: &[f32], value: &[f32], heads: usize, head_dim: usize) -> Vec<f32> {
    let inner = heads * head_dim;
    let tokens = query.len() / inner;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut output = vec![0.0; query.len()];
    let mut scores = vec![0.0f64; tokens];
    for head in 0..heads {
        let range =
            |token: usize| token * inner + head * head_dim..token * inner + (head + 1) * head_dim;
        for query_token in 0..tokens {
            let query_row = &query[range(query_token)];
            for (key_token, score) in scores.iter_mut().enumerate() {
                let key_row = &key[range(key_token)];
                *score = query_row
                    .iter()
                    .zip(key_row)
                    .map(|(&left, &right)| left as f64 * right as f64)
                    .sum::<f64>()
                    * scale;
            }
            let maximum = scores.iter().cloned().fold(f64::MIN, f64::max);
            let mut total = 0.0;
            for score in scores.iter_mut() {
                *score = (*score - maximum).exp();
                total += *score;
            }
            let output_range = range(query_token);
            for dimension in 0..head_dim {
                let sum: f64 = (0..tokens)
                    .map(|key_token| {
                        scores[key_token]
                            * value[key_token * inner + head * head_dim + dimension] as f64
                    })
                    .sum();
                output[output_range.start + dimension] = (sum / total) as f32;
            }
        }
    }
    output
}

/// Rotates dimension pairs (i, i + half) for i < half within the first 2 × half dimensions of every
/// head. `angles` is `[tokens, half]`.
fn apply_rope(data: &mut [f32], heads: usize, head_dim: usize, angles: &[f32]) {
    let tokens = data.len() / (heads * head_dim);
    let half = angles.len() / tokens;
    for (token_index, token) in data.chunks_exact_mut(heads * head_dim).enumerate() {
        for head in token.chunks_exact_mut(head_dim) {
            for (pair, &angle) in angles[token_index * half..(token_index + 1) * half]
                .iter()
                .enumerate()
            {
                let (sine, cosine) = angle.sin_cos();
                let (first, second) = (head[pair], head[pair + half]);
                head[pair] = first * cosine - second * sine;
                head[pair + half] = first * sine + second * cosine;
            }
        }
    }
}

struct AttentionWeights<'a> {
    qkv: &'a Tensor,
    query_norm: &'a Tensor,
    key_norm: &'a Tensor,
    output: &'a Tensor,
}

fn self_attention(
    hidden: &[f32],
    weights: &AttentionWeights,
    config: &DitConfig,
    angles: Option<&[f32]>,
) -> Vec<f32> {
    let inner = config.inner();
    let qkv = linear(hidden, weights.qkv, None);
    let tokens = qkv.len() / (3 * inner);
    let mut query = Vec::with_capacity(tokens * inner);
    let mut key = Vec::with_capacity(tokens * inner);
    let mut value = Vec::with_capacity(tokens * inner);
    for row in qkv.chunks_exact(3 * inner) {
        query.extend_from_slice(&row[..inner]);
        key.extend_from_slice(&row[inner..2 * inner]);
        value.extend_from_slice(&row[2 * inner..]);
    }
    rms_norm(
        &mut query,
        config.head_dim,
        &weights.query_norm.data,
        config.norm_eps,
    );
    rms_norm(
        &mut key,
        config.head_dim,
        &weights.key_norm.data,
        config.norm_eps,
    );
    if let Some(angles) = angles {
        apply_rope(&mut query, config.heads, config.head_dim, angles);
        apply_rope(&mut key, config.heads, config.head_dim, angles);
    }
    linear(
        &attention(&query, &key, &value, config.heads, config.head_dim),
        weights.output,
        None,
    )
}

fn feed_forward(hidden: &[f32], first: &Tensor, second: &Tensor, ffn: usize) -> Vec<f32> {
    let projected = linear(hidden, first, None);
    let activated: Vec<f32> = projected
        .chunks_exact(2 * ffn)
        .flat_map(|row| {
            row[..ffn]
                .iter()
                .zip(&row[ffn..])
                .map(|(&gate, &up)| silu(gate) * up)
                .collect::<Vec<_>>()
        })
        .collect();
    linear(&activated, second, None)
}

fn attention_weights<'a>(weights: &'a DitWeights, prefix: &str) -> AttentionWeights<'a> {
    AttentionWeights {
        qkv: weights.get(&format!("{prefix}.attn.qkv_proj.weight")),
        query_norm: weights.get(&format!("{prefix}.attn.q_norm.weight")),
        key_norm: weights.get(&format!("{prefix}.attn.k_norm.weight")),
        output: weights.get(&format!("{prefix}.attn.out_proj.weight")),
    }
}

/// Text states after condition_proj and the token refiner, `[tokens, hidden]`.
fn refine_text(weights: &DitWeights, config: &DitConfig, context: &Tensor) -> Vec<f32> {
    let width = config.hidden;
    let mut hidden = linear(
        &context.data,
        weights.get("condition_proj.weight"),
        Some(weights.get("condition_proj.bias")),
    );
    for layer in 0..config.refiner_layers {
        let prefix = format!("token_refiner.blocks.{layer}");
        let mut normalized = hidden.clone();
        rms_norm(
            &mut normalized,
            width,
            &weights.get(&format!("{prefix}.norm1.weight")).data,
            config.norm_eps,
        );
        let attended = self_attention(
            &normalized,
            &attention_weights(weights, &prefix),
            config,
            None,
        );
        hidden
            .iter_mut()
            .zip(&attended)
            .for_each(|(value, &delta)| *value += delta);
        let mut normalized = hidden.clone();
        rms_norm(
            &mut normalized,
            width,
            &weights.get(&format!("{prefix}.norm2.weight")).data,
            config.norm_eps,
        );
        let mixed = feed_forward(
            &normalized,
            weights.get(&format!("{prefix}.mlp.fc1.weight")),
            weights.get(&format!("{prefix}.mlp.fc2.weight")),
            config.ffn,
        );
        hidden
            .iter_mut()
            .zip(&mixed)
            .for_each(|(value, &delta)| *value += delta);
    }
    rms_norm(
        &mut hidden,
        width,
        &weights.get("token_refiner.final_norm.weight").data,
        config.norm_eps,
    );
    hidden
}

/// Scale-and-shift modulation vectors of one AdaLN projection:
/// `[timesteps × modalities, chunks, hidden]`.
struct Modulation {
    values: Vec<f32>,
    chunks: usize,
    hidden: usize,
}

impl Modulation {
    fn new(
        time_embedding: &[f32],
        weight: &Tensor,
        bias: &Tensor,
        chunks: usize,
        hidden: usize,
    ) -> Self {
        Modulation {
            values: linear(time_embedding, weight, Some(bias)),
            chunks,
            hidden,
        }
    }

    /// `row` counts timesteps × modalities for block projections and timesteps for the final layer.
    fn vector(&self, row: usize, chunk: usize) -> &[f32] {
        let start = (row * self.chunks + chunk) * self.hidden;
        &self.values[start..start + self.hidden]
    }
}

pub fn forward(weights: &DitWeights, config: &DitConfig, inputs: &DitInputs) -> DitTrace {
    let width = config.hidden;
    let video_shape = &inputs.video.shape;
    let layout = PackedLayout::for_inputs(inputs);
    let timesteps = StepTimesteps::for_layout(
        &layout,
        inputs.sigma,
        inputs.shift_video,
        inputs.shift_audio,
    );
    let rows = layout.modulation_rows(&timesteps);

    let text_states = refine_text(weights, config, &inputs.context);
    let embed_audio = |latent: &Tensor| {
        linear(
            &pack_audio(latent),
            weights.get("audio_patch_proj.weight"),
            Some(weights.get("audio_patch_proj.bias")),
        )
    };
    let embed_video = |latent: &Tensor| {
        linear(
            &patchify_video(latent),
            weights.get("video_patch_proj.weight"),
            Some(weights.get("video_patch_proj.bias")),
        )
    };
    let mut hidden = Vec::with_capacity(layout.len() * width);
    for segment in &layout.segments {
        match segment.kind {
            SegmentKind::Text => hidden.extend_from_slice(&text_states),
            SegmentKind::KeyframeVideo(index) => {
                hidden.extend(embed_video(inputs.keyframes[index].video.as_ref().unwrap()))
            }
            SegmentKind::KeyframeAudio(index) => {
                hidden.extend(embed_audio(inputs.keyframes[index].audio.as_ref().unwrap()))
            }
            SegmentKind::Audio => hidden.extend(embed_audio(&inputs.audio)),
            SegmentKind::Video => hidden.extend(embed_video(&inputs.video)),
        }
    }
    assert_eq!(hidden.len(), layout.len() * width);

    let time_embedding = timesteps.time_embedding(weights.get("adaln_t_table"));
    let angles = layout.rope_angles(&weights.get("rope.inv_freq").data);
    let modulate = |normalized: &mut [f32],
                    modulation: &Modulation,
                    shift_chunk: usize,
                    scale_chunk: usize| {
        for (token, row) in normalized.chunks_exact_mut(width).enumerate() {
            let (shift, scale) = (
                modulation.vector(rows[token], shift_chunk),
                modulation.vector(rows[token], scale_chunk),
            );
            for index in 0..width {
                row[index] = row[index] * (1.0 + scale[index]) + shift[index];
            }
        }
    };
    let add_gated =
        |hidden: &mut [f32], delta: &[f32], modulation: &Modulation, gate_chunk: usize| {
            for (token, (row, delta_row)) in hidden
                .chunks_exact_mut(width)
                .zip(delta.chunks_exact(width))
                .enumerate()
            {
                let gate = modulation.vector(rows[token], gate_chunk);
                for index in 0..width {
                    row[index] += delta_row[index] * gate[index];
                }
            }
        };

    let mut block_outputs = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let prefix = format!("blocks.{layer}");
        let modulation = Modulation::new(
            &time_embedding,
            weights.get(&format!("{prefix}.adaln_proj.linear.weight")),
            weights.get(&format!("{prefix}.adaln_proj.linear.bias")),
            6,
            width,
        );
        let mut normalized = hidden.clone();
        rms_norm(
            &mut normalized,
            width,
            &weights.get(&format!("{prefix}.norm1.weight")).data,
            config.norm_eps,
        );
        modulate(&mut normalized, &modulation, 0, 1);
        let attended = self_attention(
            &normalized,
            &attention_weights(weights, &prefix),
            config,
            Some(&angles),
        );
        add_gated(&mut hidden, &attended, &modulation, 2);

        let mut normalized = hidden.clone();
        rms_norm(
            &mut normalized,
            width,
            &weights.get(&format!("{prefix}.norm2.weight")).data,
            config.norm_eps,
        );
        modulate(&mut normalized, &modulation, 3, 4);
        let mixed = feed_forward(
            &normalized,
            weights.get(&format!("{prefix}.mlp.fc1.weight")),
            weights.get(&format!("{prefix}.mlp.fc2.weight")),
            config.ffn,
        );
        add_gated(&mut hidden, &mixed, &modulation, 5);
        block_outputs.push(hidden.clone());
    }

    let final_modulation = Modulation::new(
        &time_embedding,
        weights.get("final_layer.adaln_proj.linear.weight"),
        weights.get("final_layer.adaln_proj.linear.bias"),
        2,
        width,
    );
    let project = |kind: SegmentKind, head: &str| {
        let segment = layout.segment(kind);
        let mut rows = hidden[segment.start * width..segment.end * width].to_vec();
        rms_norm(
            &mut rows,
            width,
            &weights.get("final_layer.norm.weight").data,
            config.norm_eps,
        );
        let timestep = timesteps.index_of(kind);
        let (shift, scale) = (
            final_modulation.vector(timestep, 0),
            final_modulation.vector(timestep, 1),
        );
        for row in rows.chunks_exact_mut(width) {
            for index in 0..width {
                row[index] = row[index] * (1.0 + scale[index]) + shift[index];
            }
        }
        linear(
            &rows,
            weights.get(&format!("final_layer.{head}.weight")),
            Some(weights.get(&format!("final_layer.{head}.bias"))),
        )
    };
    let video_rows = project(SegmentKind::Video, "video_out");
    let audio_rows = project(SegmentKind::Audio, "audio_out");
    let video = unpatchify_video(&video_rows, video_shape)
        .into_iter()
        .map(|value| -value)
        .collect();
    let audio = unpack_audio(&audio_rows, &inputs.audio.shape)
        .into_iter()
        .map(|value| -value)
        .collect();
    DitTrace {
        text_states,
        block_outputs,
        video,
        audio,
    }
}
