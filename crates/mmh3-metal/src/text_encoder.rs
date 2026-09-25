//! Qwen3-VL language tower for text-only prompts. The output precedes the final model norm.
//!
//! The layers are units, which a Mac without the room for all of them reads on the way. Such an
//! encoder also leaves its embedding table on the disk and reads the rows of the prompt from there.
use crate::{
    Error, Result,
    model::{Weight, Weights},
    ops::Array,
    streaming::{Stream, fits},
};
use mmh3_core::{
    safetensors::{DType, SafeTensors},
    streaming::{Arrangement, Files, Unit, give_up_order},
    tensor::Tensor,
};
use std::cell::RefCell;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

const PREFIX: &str = "model.";
const EMBEDDING: &str = "embed_tokens.weight";

pub struct MetalTextEncoder {
    weights: Weights,
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    /// The layers read on the way, when some are.
    stream: Option<RefCell<Stream>>,
    /// The checkpoint, the offset and the type of an embedding table left on the disk.
    embedding: Option<(PathBuf, u64, DType)>,
}

pub struct TextEncoding {
    pub context: Tensor,
    pub layers: Vec<(usize, Tensor)>,
}

/// The layer a tensor belongs to.
fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("layers.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

impl MetalTextEncoder {
    /// Loads the whole encoder, or fails as out of memory when it does not fit in the GPU's share,
    /// which Metal would fill past by paging rather than refuse.
    pub fn load(file: &SafeTensors) -> Result<Self> {
        let bytes = file
            .tensors()
            .iter()
            .filter(|info| info.name.starts_with(PREFIX))
            .map(|info| info.byte_count())
            .sum();
        if !fits(bytes)? {
            return Err(Error::out_of_memory(format!(
                "the text encoder's {bytes} bytes do not fit in the GPU's share of memory"
            )));
        }
        Self::from_weights(Weights::load(file, PREFIX)?, None, None)
    }

    /// `load`, or, without the room for the whole encoder, an encoder that reads its layers on the
    /// way and keeps back what fits once it knows the prompt.
    pub fn load_fitting(file: &SafeTensors) -> Result<Self> {
        match Self::load(file) {
            Err(error) if error.is_out_of_memory() => {}
            result => return result,
        }
        let mut files = Files::default();
        let mut units = Vec::new();
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                continue;
            };
            let Some(layer) = layer_of(name) else {
                continue;
            };
            if name.ends_with(".comfy_quant") {
                continue;
            }
            while units.len() <= layer {
                units.push(Unit::new(&format!("layers.{}", units.len())));
            }
            units[layer].insert(
                name,
                info.dtype,
                info.shape.clone(),
                vec![files.piece(file, info)],
                Arrangement::Contiguous,
            );
        }
        let embedding = file
            .get(&format!("{PREFIX}{EMBEDDING}"))
            .ok_or_else(|| Error::new(format!("missing tensor {PREFIX}{EMBEDDING}")))?;
        let embedding = (
            file.path().to_owned(),
            file.file_offset(embedding),
            embedding.dtype,
        );
        let weights = Weights::load_leaving(
            file,
            PREFIX,
            |_| true,
            |name| name == EMBEDDING || layer_of(name).is_some(),
        )?;
        let order = give_up_order(units.len());
        let stream = Stream::new("text encoder layers", files, units, order);
        Self::from_weights(weights, Some(stream), Some(embedding))
    }

    fn from_weights(
        weights: Weights,
        stream: Option<Stream>,
        embedding: Option<(PathBuf, u64, DType)>,
    ) -> Result<Self> {
        let hidden = weights.shape(EMBEDDING)?[1];
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
            stream: stream.map(RefCell::new),
            embedding,
        })
    }

    /// The prompt's rows of the embedding table.
    fn embed(&self, ids: &[u32]) -> Result<Array> {
        let w = &self.weights;
        let Some((path, offset, dtype)) = &self.embedding else {
            return w.embedding(EMBEDDING, ids);
        };
        let row_bytes = self.hidden * dtype.size_in_bytes();
        let file = std::fs::File::open(path)?;
        let mut rows = vec![0; ids.len() * row_bytes];
        for (&id, row) in ids.iter().zip(rows.chunks_exact_mut(row_bytes)) {
            file.read_exact_at(row, offset + (id as usize * row_bytes) as u64)?;
        }
        let table = Weight {
            buffer: w.device.alloc(rows.len(), Some(&rows))?,
            shape: vec![ids.len(), self.hidden],
            dtype: *dtype,
        };
        let order: Vec<u32> = (0..ids.len() as u32).collect();
        w.embedding_of(&table, &order)
    }

    pub fn encode(&self, ids: &[u32], capture: &[usize]) -> Result<TextEncoding> {
        let w = &self.weights;
        let vocabulary = w.shape(EMBEDDING)?[0];
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= vocabulary) {
            return Err(Error::new("invalid prompt token ids".into()));
        }

        let mut stream = self.stream.as_ref().map(RefCell::borrow_mut);
        if let Some(stream) = &mut stream {
            // FP32 rows of the residual, the projections and the MLP, several of each alive at once.
            let ffn = w.shape("layers.0.mlp.gate_proj.weight")?[0];
            stream.settle(w, ids.len() * (self.hidden * 64 + ffn * 16))?;
        }
        let mut hidden = self.embed(ids)?;
        let angles: Vec<f32> = (0..ids.len())
            .flat_map(|t| (0..64).map(move |i| t as f32 / 5_000_000f32.powf(i as f32 / 64.0)))
            .collect();
        let angles = Array::from_f32(&w.device, ids.len(), 64, &angles)?;
        let mut layers = Vec::new();
        let mut pass = stream
            .as_mut()
            .map(|stream| stream.pass(w, 0..self.layers))
            .transpose()?;

        for layer in 0..self.layers {
            if let Some(pass) = &mut pass {
                pass.enter(layer)?;
            }
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
            if let Some(pass) = &mut pass {
                pass.leave(layer);
            }

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
