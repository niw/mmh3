//! Qwen3-VL language tower for prompts of text and of the pictures the vision tower embeds. The
//! output precedes the final model norm.
//!
//! The layers are units, which a Mac without the room for all of them reads on the way. The
//! embedding table stays on the disk, and an encoder reads the rows of the prompt from there.
use crate::{
    Error, Result,
    model::{Concatenated, Weight, Weights},
    ops::Array,
    streaming::{Stream, fits},
    vision::VisionEmbeddings,
};
use mmh3_core::{
    safetensors::{DType, SafeTensors},
    streaming::{Arrangement, Files, give_up_order},
    tensor::Tensor,
    vision::VisionPrompt,
};
use std::cell::RefCell;
use std::fs::File;
use std::os::unix::fs::FileExt;

const PREFIX: &str = "model.";
const EMBEDDING: &str = "embed_tokens.weight";
const HEAD_DIM: usize = 128;
const ROPE_THETA: f32 = 5_000_000.0;
/// The layers of a block that read one input, read as one layer each so that the input is rotated
/// and packed once for them and one product covers their outputs.
const CONCATENATED: [(&str, &[&str]); 2] = [
    (
        "self_attn.qkv_proj",
        &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ),
    ("mlp.gate_up_proj", &["mlp.gate_proj", "mlp.up_proj"]),
];

/// The layers of the checkpoint's `layers` blocks that are read concatenated.
fn concatenations(layers: usize) -> Vec<Concatenated> {
    (0..layers)
        .flat_map(|layer| {
            CONCATENATED.iter().map(move |(name, parts)| Concatenated {
                name: format!("layers.{layer}.{name}"),
                parts: parts
                    .iter()
                    .map(|part| format!("layers.{layer}.{part}"))
                    .collect(),
            })
        })
        .collect()
}

/// How many blocks the checkpoint has.
fn layer_count(file: &SafeTensors) -> usize {
    (0..)
        .take_while(|i| {
            file.get(&format!("{PREFIX}layers.{i}.input_layernorm.weight"))
                .is_some()
        })
        .count()
}

pub struct MetalTextEncoder {
    weights: Weights,
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    /// The layers read on the way, when some are.
    stream: Option<RefCell<Stream>>,
    /// The checkpoint, the offset and the type of the embedding table.
    embedding: (File, u64, DType),
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

/// The checkpoint, the offset and the type of the embedding table, whose rows a prompt reads.
fn embedding(file: &SafeTensors) -> Result<(File, u64, DType)> {
    let embedding = file
        .get(&format!("{PREFIX}{EMBEDDING}"))
        .ok_or_else(|| Error::new(format!("missing tensor {PREFIX}{EMBEDDING}")))?;
    Ok((
        File::open(file.path())?,
        file.file_offset(embedding),
        embedding.dtype,
    ))
}

impl MetalTextEncoder {
    /// Loads the whole encoder but its embedding table, or fails as out of memory when it does not
    /// fit in the GPU's share, which Metal would fill past by paging rather than refuse.
    pub fn load(file: &SafeTensors) -> Result<Self> {
        let embedding = embedding(file)?;
        let bytes = file
            .tensors()
            .iter()
            .filter(|info| {
                info.name
                    .strip_prefix(PREFIX)
                    .is_some_and(|name| name != EMBEDDING)
            })
            .map(|info| info.byte_count())
            .sum();
        if !fits(bytes)? {
            return Err(Error::out_of_memory(format!(
                "the text encoder's {bytes} bytes do not fit in the GPU's share of memory"
            )));
        }
        let weights = Weights::load_concatenating(
            file,
            PREFIX,
            |_| true,
            |name| name == EMBEDDING,
            &concatenations(layer_count(file)),
        )?;
        Self::from_weights(file, weights, None, embedding)
    }

    /// `load`, or, without the room for the whole encoder, an encoder that reads its layers on the
    /// way and keeps back what fits once it knows the prompt.
    /// Select the precisions of the encoder's INT8 layers and of its attention.
    pub fn set_precision(
        &mut self,
        linear: crate::LinearPrecision,
        attention: crate::AttentionPrecision,
    ) -> Result<()> {
        self.weights.set_precision((linear, attention))
    }

    pub fn load_fitting(file: &SafeTensors) -> Result<Self> {
        match Self::load(file) {
            Err(error) if error.is_out_of_memory() => {}
            result => return result,
        }
        let mut files = Files::default();
        let layers = layer_count(file);
        let concatenated = concatenations(layers);
        let mut units: Vec<_> = (0..layers)
            .map(|layer| crate::streaming::unit(&format!("layers.{layer}")))
            .collect();
        let is_part = |layer: &str| {
            concatenated
                .iter()
                .any(|whole| whole.parts.iter().any(|part| part == layer))
        };
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                continue;
            };
            let Some(layer) = layer_of(name) else {
                continue;
            };
            if name.ends_with(".comfy_quant")
                || name
                    .rsplit_once('.')
                    .is_some_and(|(layer, _)| is_part(layer))
            {
                continue;
            }
            units[layer].insert(
                name,
                info.dtype,
                info.shape.clone(),
                vec![files.piece(file, info)],
                Arrangement::Contiguous,
            );
        }
        for whole in &concatenated {
            let layer = layer_of(&whole.name).expect("a block's layer");
            for (name, infos, shape) in whole.tensors(file, PREFIX)? {
                units[layer].insert(
                    &name,
                    infos[0].dtype,
                    shape,
                    infos.iter().map(|info| files.piece(file, info)).collect(),
                    Arrangement::Contiguous,
                );
            }
        }
        let embedding = embedding(file)?;
        let weights = Weights::load_concatenating(
            file,
            PREFIX,
            |_| true,
            |name| name == EMBEDDING || layer_of(name).is_some(),
            &concatenated,
        )?;
        let order = give_up_order(units.len());
        let stream = Stream::new("text encoder layers", files, units, order);
        Self::from_weights(file, weights, Some(stream), embedding)
    }

    /// Reads every layer on the way, keeping none back, for an encoder that encodes one prompt.
    /// It reads each layer once either way, and layers kept back would press the rest of the
    /// system's memory into swap and hold what the DiT wants next.
    pub fn read_once(&mut self) {
        if let Some(stream) = &self.stream {
            stream.borrow_mut().keep_none();
        }
    }

    fn from_weights(
        file: &SafeTensors,
        weights: Weights,
        stream: Option<Stream>,
        embedding: (File, u64, DType),
    ) -> Result<Self> {
        let hidden = weights.shape(EMBEDDING)?[1];
        // The query, key and value projections are read as one layer, so their widths come from
        // the checkpoint.
        let outputs = |name: &str| -> Result<usize> {
            let name = format!("{PREFIX}layers.0.self_attn.{name}_proj.weight");
            file.get(&name)
                .and_then(|info| info.shape.first().copied())
                .ok_or_else(|| Error::new(format!("missing tensor {name}")))
        };
        let heads = outputs("q")? / HEAD_DIM;
        let kv_heads = outputs("k")? / HEAD_DIM;
        let layers = layer_count(file);
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
        let (file, offset, dtype) = &self.embedding;
        let row_bytes = self.hidden * dtype.size_in_bytes();
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

    /// Rotary angles `[tokens, 64]` of (time, height, width) positions with Qwen3-VL's interleaved
    /// MRoPE: frequency i rotates by the height for i % 3 == 1 and by the width for i % 3 == 2
    /// among the first 60, and by the time otherwise. Text sits at the same position on all three
    /// axes, which makes these the plain rotary angles of its position.
    fn rope_angles(positions: &[[usize; 3]]) -> Vec<f32> {
        let axis = |index: usize| match index % 3 {
            1 if index < 60 => 1,
            2 if index < 60 => 2,
            _ => 0,
        };
        positions
            .iter()
            .flat_map(|position| {
                (0..HEAD_DIM / 2).map(move |i| {
                    position[axis(i)] as f32 / ROPE_THETA.powf(i as f32 / (HEAD_DIM / 2) as f32)
                })
            })
            .collect()
    }

    /// `rows` with each picture's rows replaced by what `replacement` gives for the picture, when
    /// it gives anything. The pictures come in order and do not overlap.
    fn with_pictures<'a>(
        rows: &Array,
        pictures: &[(usize, &'a VisionEmbeddings)],
        replacement: impl Fn(&'a VisionEmbeddings) -> Option<&'a Array>,
    ) -> Result<Array> {
        let [count, width] = rows.shape();
        let mut parts = Vec::new();
        let mut next = 0;
        for &(start, picture) in pictures {
            let Some(replacement) = replacement(picture) else {
                continue;
            };
            if start > next {
                parts.push(rows.slice(next, start - next, 0, width)?);
            }
            parts.push(replacement.clone());
            next = start + picture.tokens;
        }
        if next < count {
            parts.push(rows.slice(next, count - next, 0, width)?);
        }
        Array::concat(&parts, false)
    }

    /// Encodes token ids and returns the conditioning `[tokens, hidden]`, capturing the hidden
    /// states after the listed layers.
    pub fn encode(&self, ids: &[u32], capture: &[usize]) -> Result<TextEncoding> {
        let positions: Vec<[usize; 3]> = (0..ids.len()).map(|position| [position; 3]).collect();
        self.encode_positions(ids, &positions, &[], capture)
    }

    /// Encodes a prompt with the embeddings of its pictures, in order: the pictures' rows take
    /// their embeddings in place of the pad tokens' and add their DeepStack features after the
    /// first layers, after those layers' hidden states are captured.
    pub fn encode_prompt(
        &self,
        prompt: &VisionPrompt,
        pictures: &[VisionEmbeddings],
        capture: &[usize],
    ) -> Result<TextEncoding> {
        if pictures.len() != prompt.picture_starts.len() {
            return Err(Error::new(format!(
                "the prompt holds {} pictures, not {}",
                prompt.picture_starts.len(),
                pictures.len()
            )));
        }
        let pictures: Vec<(usize, &VisionEmbeddings)> = prompt
            .picture_starts
            .iter()
            .copied()
            .zip(pictures)
            .collect();
        self.encode_positions(&prompt.ids, &prompt.positions, &pictures, capture)
    }

    fn encode_positions(
        &self,
        ids: &[u32],
        positions: &[[usize; 3]],
        pictures: &[(usize, &VisionEmbeddings)],
        capture: &[usize],
    ) -> Result<TextEncoding> {
        let w = &self.weights;
        let vocabulary = w.shape(EMBEDDING)?[0];
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= vocabulary) {
            return Err(Error::new("invalid prompt token ids".into()));
        }
        if positions.len() != ids.len() {
            return Err(Error::new(
                "the prompt's positions do not match its ids".into(),
            ));
        }
        let mut end = 0;
        for &(start, picture) in pictures {
            let fits = |rows: &Array| rows.shape() == [picture.tokens, self.hidden];
            if start < end
                || start + picture.tokens > ids.len()
                || !fits(&picture.merged)
                || !picture.deepstack.iter().all(fits)
            {
                return Err(Error::new(
                    "a picture's embeddings do not fit the prompt".into(),
                ));
            }
            end = start + picture.tokens;
        }

        let mut stream = self.stream.as_ref().map(RefCell::borrow_mut);
        if let Some(stream) = &mut stream {
            // FP32 rows of the residual, the projections and the MLP, several of each alive at once.
            let ffn = w.shape("layers.0.mlp.gate_up_proj.weight")?[0] / 2;
            stream.settle(w, ids.len() * (self.hidden * 64 + ffn * 16))?;
        }
        let mut hidden = self.embed(ids)?;
        if !pictures.is_empty() {
            hidden = Self::with_pictures(&hidden, pictures, |picture| Some(&picture.merged))?;
        }
        let angles = Array::from_f32(
            &w.device,
            ids.len(),
            HEAD_DIM / 2,
            &Self::rope_angles(positions),
        )?;
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
            // The queries, the keys and the values side by side, from one product.
            let qkv = w.linear(&normalized, &format!("{p}.self_attn.qkv_proj"))?;
            let (q_width, kv_width) = (self.heads * HEAD_DIM, self.kv_heads * HEAD_DIM);
            let project = |name: &str, column: usize, heads: usize| -> Result<Array> {
                let x = qkv.slice(0, ids.len(), column, heads * HEAD_DIM)?;
                w.norm(
                    &x.reshape(ids.len() * heads, HEAD_DIM)?,
                    &format!("{p}.self_attn.{name}_norm"),
                    1e-6,
                )?
                .reshape(ids.len(), heads * HEAD_DIM)?
                .rope(heads, &angles)
            };

            let q = project("q", 0, self.heads)?;
            let k = project("k", q_width, self.kv_heads)?;
            let v = qkv.slice(0, ids.len(), q_width + kv_width, kv_width)?;
            hidden = hidden.add(&w.linear(
                &q.attention_at(
                    &k,
                    &v,
                    self.heads,
                    self.kv_heads,
                    true,
                    w.attention_precision,
                )?,
                &format!("{p}.self_attn.o_proj"),
            )?)?;
            let normalized = w.norm(&hidden, &format!("{p}.post_attention_layernorm"), 1e-6)?;
            // The gate and up halves from one product, gated as the down projection packs them.
            let gate_up = w.linear(&normalized, &format!("{p}.mlp.gate_up_proj"))?;
            hidden = hidden.add(&w.linear_gated(&gate_up, &format!("{p}.mlp.down_proj"))?)?;
            if let Some(pass) = &mut pass {
                pass.leave(layer);
            }

            if capture.contains(&layer) {
                layers.push((
                    layer,
                    Tensor::new(vec![ids.len(), self.hidden], hidden.to_f32()?),
                ));
            }
            // The pictures' DeepStack features for this layer, with zeros in the other rows,
            // which leave them as they are.
            if pictures
                .iter()
                .any(|(_, picture)| picture.deepstack.len() > layer)
            {
                let zeros = Array::zeros(&w.device, ids.len(), self.hidden)?;
                hidden = hidden.add(&Self::with_pictures(&zeros, pictures, |picture| {
                    picture.deepstack.get(layer)
                })?)?;
            }
        }

        Ok(TextEncoding {
            context: Tensor::new(vec![ids.len(), self.hidden], hidden.to_f32()?),
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaves_the_axes_of_the_rotary_positions() {
        let angles = MetalTextEncoder::rope_angles(&[[3; 3], [7, 30, 48]]);
        assert_eq!(angles.len(), 2 * 64);
        let frequency = |i: usize| 5_000_000f32.powf(i as f32 / 64.0);
        // Text rotates like the plain rotary embedding of its position.
        for (i, &angle) in angles[..64].iter().enumerate() {
            assert_eq!(angle, 3.0 / frequency(i));
        }
        let picture = &angles[64..];
        assert_eq!(picture[0], 7.0 / frequency(0));
        assert_eq!(picture[1], 30.0 / frequency(1));
        assert_eq!(picture[2], 48.0 / frequency(2));
        assert_eq!(picture[58], 30.0 / frequency(58));
        assert_eq!(picture[59], 48.0 / frequency(59));
        // Past the first 60 frequencies every one takes the time.
        for (i, &angle) in picture.iter().enumerate().skip(60) {
            assert_eq!(angle, 7.0 / frequency(i));
        }
    }
}
