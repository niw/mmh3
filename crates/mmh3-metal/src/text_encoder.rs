//! Qwen3-VL language tower for text-only prompts. The output precedes the final model norm.
use crate::{Error, Result, model::Weights, ops::Array};
use mmh3_core::{safetensors::SafeTensors, tensor::Tensor};

pub struct MetalTextEncoder {
    weights: Weights,
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
}

pub struct TextEncoding {
    pub context: Tensor,
    pub layers: Vec<(usize, Tensor)>,
}

impl MetalTextEncoder {
    pub fn load(file: &SafeTensors) -> Result<Self> {
        let weights = Weights::load(file, "model.")?;
        let hidden = weights.shape("embed_tokens.weight")?[1];
        let heads = weights.shape("layers.0.self_attn.q_proj.weight")?[0] / 128;
        let kv_heads = weights.shape("layers.0.self_attn.k_proj.weight")?[0] / 128;
        let layers = (0..)
            .take_while(|i| weights.contains(&format!("layers.{i}.input_layernorm.weight")))
            .count();
        if heads == 0 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) || layers == 0 {
            return Err(Error::new("unsupported text encoder configuration".into()));
        }

        Ok(Self {
            weights,
            hidden,
            layers,
            heads,
            kv_heads,
        })
    }

    pub fn encode(&self, ids: &[u32], capture: &[usize]) -> Result<TextEncoding> {
        let w = &self.weights;
        let vocabulary = w.shape("embed_tokens.weight")?[0];
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= vocabulary) {
            return Err(Error::new("invalid prompt token ids".into()));
        }

        let mut hidden = w.embedding("embed_tokens.weight", ids)?;
        let angles: Vec<f32> = (0..ids.len())
            .flat_map(|t| (0..64).map(move |i| t as f32 / 5_000_000f32.powf(i as f32 / 64.0)))
            .collect();
        let angles = Array::from_f32(&w.device, ids.len(), 64, &angles)?;
        let mut layers = Vec::new();

        for layer in 0..self.layers {
            let p = format!("layers.{layer}");
            let normalized = w.norm(&hidden, &format!("{p}.input_layernorm"), 1e-6)?;
            let project = |name: &str, heads: usize| -> Result<Array> {
                let x = w.linear(&normalized, &format!("{p}.self_attn.{name}_proj"))?;
                w.norm(
                    &x.reshape(ids.len() * heads, 128)?,
                    &format!("{p}.self_attn.{name}_norm"),
                    1e-6,
                )?
                .reshape(ids.len(), heads * 128)?
                .rope(heads, &angles)
            };

            let q = project("q", self.heads)?;
            let k = project("k", self.kv_heads)?;
            let v = w.linear(&normalized, &format!("{p}.self_attn.v_proj"))?;
            hidden = hidden.add(&w.linear(
                &q.attention(&k, &v, self.heads, self.kv_heads, true)?,
                &format!("{p}.self_attn.o_proj"),
            )?)?;
            let normalized = w.norm(&hidden, &format!("{p}.post_attention_layernorm"), 1e-6)?;
            let gate = w
                .linear(&normalized, &format!("{p}.mlp.gate_proj"))?
                .unary(1, 1.0)?;
            let up = w.linear(&normalized, &format!("{p}.mlp.up_proj"))?;
            hidden = hidden.add(&w.linear(&gate.mul(&up)?, &format!("{p}.mlp.down_proj"))?)?;

            if capture.contains(&layer) {
                layers.push((
                    layer,
                    Tensor::new(vec![ids.len(), self.hidden], hidden.to_f32()?),
                ));
            }
        }

        Ok(TextEncoding {
            context: Tensor::new(vec![ids.len(), self.hidden], hidden.to_f32()?),
            layers,
        })
    }
}
