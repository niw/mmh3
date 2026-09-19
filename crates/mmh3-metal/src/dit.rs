//! Dense DiT on Metal, including the shared keyframe and reference token layouts.
use crate::{
    Device, Error, Result,
    model::Weights,
    ops::{Array, RowMap},
    shard::Memory,
};
use mmh3_core::{
    dit::{
        config::DitConfig,
        inputs::DitInputs,
        latent::{pack_audio, patchify_video, unpack_audio, unpatchify_video},
        layout::{PackedLayout, SegmentKind},
        timestep::StepTimesteps,
    },
    safetensors::SafeTensors,
    shard::{Exchange, ExchangeError, Region, Shard, regions as shard_regions},
    tensor::Tensor,
};

/// A transport's failure in this backend's terms, since a block reports one error type.
fn shard_error(error: ExchangeError) -> Error {
    Error(error.0)
}

pub struct MetalDit {
    weights: Weights,
    config: DitConfig,
    time_table: Tensor,
    inv_freq: Vec<f32>,
}

/// Refined prompt states retained on the GPU for an entire sampling run.
/// Borrowing the model prevents changing adapters while these states are in use.
pub struct PreparedText<'a> {
    model: &'a MetalDit,
    source: &'a Tensor,
    text: Array,
}

impl PreparedText<'_> {
    /// Sampling path: block captures are optional; text states are not read back.
    pub fn forward(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
    ) -> Result<DitOutput> {
        if inputs.context != *self.source {
            return Err(Error("prepared text does not match the context".into()));
        }

        self.model
            .forward_inner(inputs, capture, sparse, &self.text, false)
    }
}

pub struct DitOutput {
    pub text_states: Vec<f32>,
    pub blocks: Vec<(usize, Vec<f32>)>,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    pub routed_fraction: Option<f64>,
}

impl MetalDit {
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self> {
        let config = DitConfig::from_shapes(|name| {
            file.get(&format!("{prefix}{name}"))
                .map(|i| i.shape.clone())
        })?;
        let weights = Weights::load(file, prefix)?;
        let time_table = weights.host("adaln_t_table")?;
        let inv_freq = weights.host("rope.inv_freq")?.data;
        Ok(Self {
            weights,
            config,
            time_table,
            inv_freq,
        })
    }

    /// Select computation for INT8 ConvRot layers before preparing the prompt.
    pub fn set_linear_precision(&mut self, precision: crate::LinearPrecision) -> Result<()> {
        if matches!(
            precision,
            crate::LinearPrecision::Fp16 | crate::LinearPrecision::Int8
        ) && !self.device().supports_tensor_ops()
        {
            return Err(crate::Error(
                "low-precision Metal products require macOS 26 and Apple silicon".into(),
            ));
        }

        self.weights.linear_precision = precision;
        Ok(())
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.weights.device
    }

    pub fn has_vsa_gates(&self) -> bool {
        self.weights
            .contains("blocks.0.attn.to_gate_compress.weight")
    }

    pub fn add_lora(&mut self, file: &SafeTensors, strength: f32) -> Result<usize> {
        self.weights.add_lora(file, strength)
    }

    fn attention(&self, x: &Array, prefix: &str, angles: Option<&Array>) -> Result<Array> {
        let c = &self.config;
        let qkv = self.weights.linear(x, &format!("{prefix}.attn.qkv_proj"))?;
        let norm = |offset, name: &str| -> Result<Array> {
            let x = qkv
                .slice(0, x.rows, offset, c.inner())?
                .reshape(x.rows * c.heads, c.head_dim)?;
            let x = self
                .weights
                .norm(&x, &format!("{prefix}.attn.{name}_norm"), c.norm_eps)?
                .reshape(qkv.rows, c.inner())?;
            if let Some(angles) = angles {
                x.rope(c.heads, angles)
            } else {
                Ok(x)
            }
        };

        let q = norm(0, "q")?;
        let k = norm(c.inner(), "k")?;
        let v = qkv.slice(0, x.rows, 2 * c.inner(), c.inner())?;
        self.weights.linear(
            &q.attention(&k, &v, c.heads, c.heads, false)?,
            &format!("{prefix}.attn.out_proj"),
        )
    }

    /// This rank's heads of a block's attention inputs, projected from the rotated INT8 rows the
    /// exchange delivered rather than from the FP32 a whole block would have had. The rows are
    /// every token's: a rank attends its own heads over the whole sequence, which is what the
    /// exchange is for.
    pub fn project_exchanged_inputs(
        &self,
        prefix: &str,
        input: &crate::shard::Memory,
        scales: &Array,
        tokens: usize,
        own_heads: std::ops::Range<usize>,
    ) -> Result<Array> {
        let c = &self.config;
        let qkv = self.weights.linear_quantized(
            input.buffer(),
            scales,
            tokens,
            &format!("{prefix}.attn.qkv_proj"),
        )?;
        crate::shard::pack(&qkv, 3, c.heads, own_heads, c.head_dim)
    }

    /// One rank's share of a block's attention.
    ///
    /// A rank carries its own run of the sequence through everything else in a block, but a query
    /// has to see every key. So the rows are exchanged as the INT8 layers consume them, this rank
    /// projects the whole sequence for its own heads alone, attends them, and the shares come back
    /// to the rows it carries. `normalized` is this rank's rows, as the block's first
    /// normalization leaves them.
    pub fn sharded_attention<E>(
        &self,
        normalized: &Array,
        prefix: &str,
        angles: Option<&Array>,
        shard: &Shard,
        exchange: &mut E,
    ) -> Result<Array>
    where
        E: Exchange<Memory = Memory>,
    {
        let c = &self.config;
        let tokens: usize = shard.tokens.iter().map(|rows| rows.len()).sum();
        let (rows, own) = (shard.own_tokens(), shard.own_heads());
        let own_inner = own.len() * c.head_dim;
        if normalized.shape() != [rows.len(), c.hidden] {
            return Err(Error(format!(
                "this rank carries {:?}, not [{}, {}]",
                normalized.shape(),
                rows.len(),
                c.hidden
            )));
        }

        // NOTE: every region a peer writes into has to exist before the barrier that carries the
        // write, since a transport has nowhere to put what arrives for a region that was never
        // made. That is why a rank makes them all up front rather than as it reaches them.
        for (region, bytes) in shard_regions(shard, tokens, c.hidden, false) {
            // This rank keeps its attention inputs in its own arrays and Metal has no VSA gate.
            // Nothing outside writes to either, so neither is made.
            if !matches!(region, Region::Inputs | Region::Gate) {
                exchange.region(region, bytes).map_err(shard_error)?;
            }
        }

        // This rank's rows go where its peers can read them, quantized as the projections take
        // them, which is an eighth of the bytes the projections would produce.
        let input = exchange
            .region(Region::Normalized, tokens * c.hidden)
            .map_err(shard_error)?;
        let scales = exchange
            .region(Region::Scales, tokens * 4)
            .map_err(shard_error)?;
        let mine = input.write_quantized(rows.start * c.hidden, normalized)?;
        scales.write_f32(rows.start * 4, &mine)?;
        exchange
            .publish(
                Region::Normalized,
                rows.start * c.hidden,
                rows.len() * c.hidden,
            )
            .map_err(shard_error)?;
        exchange
            .publish(Region::Scales, rows.start * 4, rows.len() * 4)
            .map_err(shard_error)?;
        self.exchange_rows(shard, exchange, &rows)?;

        // Every token, this rank's heads. The projection runs over the whole sequence because the
        // attention that follows does.
        let qkv = self.project_exchanged_inputs(
            prefix,
            &input,
            &scales.read_f32(0, tokens, 1)?,
            tokens,
            own.clone(),
        )?;
        let part = |offset: usize, name: &str| -> Result<Array> {
            let x = qkv
                .slice(0, tokens, offset, own_inner)?
                .reshape(tokens * own.len(), c.head_dim)?;
            let x = self
                .weights
                .norm(&x, &format!("{prefix}.attn.{name}_norm"), c.norm_eps)?
                .reshape(tokens, own_inner)?;
            match angles {
                Some(angles) => x.rope(own.len(), angles),
                None => Ok(x),
            }
        };
        let (q, k) = (part(0, "q")?, part(own_inner, "k")?);
        let v = qkv.slice(0, tokens, 2 * own_inner, own_inner)?;
        let attended = q.attention(&k, &v, own.len(), own.len(), false)?;

        // Back to "my tokens, every head". A rank's own share is scattered straight into place and
        // the peers' shares land beside it.
        let whole = Array::zeros(&self.weights.device, rows.len(), c.inner())?;
        self.exchange_attended(shard, exchange, &rows, &attended, &whole)?;
        self.weights
            .linear(&whole, &format!("{prefix}.attn.out_proj"))
    }

    /// Pushes this rank's rows of a block's input to every peer and waits for theirs.
    fn exchange_rows<E>(
        &self,
        shard: &Shard,
        exchange: &mut E,
        rows: &std::ops::Range<usize>,
    ) -> Result<()>
    where
        E: Exchange<Memory = Memory>,
    {
        let hidden = self.config.hidden;
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            for (region, width) in [(Region::Normalized, hidden), (Region::Scales, 4)] {
                exchange
                    .write(
                        peer,
                        region,
                        rows.start * width,
                        region,
                        rows.start * width,
                        rows.len() * width,
                    )
                    .map_err(shard_error)?;
            }
        }
        exchange.barrier().map_err(shard_error)?;
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            let taken = shard.tokens[peer].clone();
            for (region, width) in [(Region::Normalized, hidden), (Region::Scales, 4)] {
                exchange
                    .receive(region, taken.start * width, taken.len() * width)
                    .map_err(shard_error)?;
            }
        }
        // Nothing may be overwritten until every rank has read it.
        exchange.barrier().map_err(shard_error)
    }

    /// Turns "every token, my heads" back into "my tokens, every head".
    fn exchange_attended<E>(
        &self,
        shard: &Shard,
        exchange: &mut E,
        rows: &std::ops::Range<usize>,
        attended: &Array,
        whole: &Array,
    ) -> Result<()>
    where
        E: Exchange<Memory = Memory>,
    {
        let c = &self.config;
        let tokens: usize = shard.tokens.iter().map(|taken| taken.len()).sum();
        let own = shard.own_heads();
        let own_inner = own.len() * c.head_dim;

        let mine = exchange
            .region(Region::Attended, tokens * own_inner * 2)
            .map_err(shard_error)?;
        mine.write_bf16(0, attended)?;
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            let taken = shard.tokens[peer].clone();
            exchange
                .publish(
                    Region::Attended,
                    taken.start * own_inner * 2,
                    taken.len() * own_inner * 2,
                )
                .map_err(shard_error)?;
            exchange
                .write(
                    peer,
                    Region::Attended,
                    taken.start * own_inner * 2,
                    Region::Received(shard.rank),
                    0,
                    taken.len() * own_inner * 2,
                )
                .map_err(shard_error)?;
        }
        exchange.barrier().map_err(shard_error)?;

        for peer in 0..shard.ranks() {
            let span = shard.heads[peer].clone();
            let inner = span.len() * c.head_dim;
            let part = if peer == shard.rank {
                mine.read_bf16(rows.start * own_inner * 2, rows.len(), own_inner)?
            } else {
                let from = exchange
                    .region(Region::Received(peer), rows.len() * inner * 2)
                    .map_err(shard_error)?;
                exchange
                    .receive(Region::Received(peer), 0, rows.len() * inner * 2)
                    .map_err(shard_error)?;
                from.read_bf16(0, rows.len(), inner)?
            };
            crate::shard::unpack(&part, whole, c.heads, span, c.head_dim)?;
        }
        exchange.barrier().map_err(shard_error)
    }

    fn mlp(&self, x: &Array, prefix: &str) -> Result<Array> {
        self.weights.linear(
            &self
                .weights
                .linear(x, &format!("{prefix}.mlp.fc1"))?
                .swiglu()?,
            &format!("{prefix}.mlp.fc2"),
        )
    }

    pub fn prepare_text<'a>(&'a self, context: &'a Tensor) -> Result<PreparedText<'a>> {
        let c = &self.config;
        let w = &self.weights;
        if context.shape.len() != 2 || context.shape[0] == 0 || context.shape[1] != c.text_dim {
            return Err(Error("context does not match the model".into()));
        }

        let uploaded = Array::from_f32(&w.device, context.shape[0], c.text_dim, &context.data)?;
        let mut text = w.linear(&uploaded, "condition_proj")?;
        for layer in 0..c.refiner_layers {
            let p = format!("token_refiner.blocks.{layer}");
            text = text.add(&self.attention(
                &w.norm(&text, &format!("{p}.norm1"), c.norm_eps)?,
                &p,
                None,
            )?)?;
            text = text.add(&self.mlp(&w.norm(&text, &format!("{p}.norm2"), c.norm_eps)?, &p)?)?;
        }

        text = w.norm(&text, "token_refiner.final_norm", c.norm_eps)?;
        Ok(PreparedText {
            model: self,
            source: context,
            text,
        })
    }

    pub fn forward(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
    ) -> Result<DitOutput> {
        let prepared = self.prepare_text(&inputs.context)?;
        self.forward_inner(inputs, capture, sparse, &prepared.text, true)
    }

    fn forward_inner(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
        text: &Array,
        capture_text: bool,
    ) -> Result<DitOutput> {
        if sparse.is_some() || self.has_vsa_gates() {
            return Err(Error(
                "Metal currently supports dense attention without VSA gates".into(),
            ));
        }

        let c = &self.config;
        if inputs.context.shape.len() != 2
            || inputs.context.shape[1] != c.text_dim
            || inputs.video.shape.len() != 4
            || inputs.video.shape[0] != c.video_channels
            || inputs.video.shape[2..]
                .iter()
                .any(|&n| n == 0 || n % 2 != 0)
            || inputs.video.shape[1] == 0
            || inputs.audio.shape.len() != 3
            || inputs.audio.shape[0] != c.audio_channels
            || inputs.audio.shape[1] != 2
            || inputs.audio.shape[2] == 0
            || !inputs.sigma.is_finite()
            || !(0.0..=1.0).contains(&inputs.sigma)
        {
            return Err(Error("DiT inputs do not match the model".into()));
        }

        let w = &self.weights;
        let width = c.hidden;
        let d = &w.device;
        let upload = |values: &[f32], cols| Array::from_f32(d, values.len() / cols, cols, values);
        let text_states = if capture_text {
            text.to_f32()?
        } else {
            Vec::new()
        };

        let layout = PackedLayout::for_inputs(inputs);
        let timesteps = StepTimesteps::for_layout(
            &layout,
            inputs.sigma,
            inputs.shift_video,
            inputs.shift_audio,
        );
        let rows = RowMap::new(d, &layout.modulation_rows(&timesteps))?;
        let mut parts = Vec::new();

        for segment in &layout.segments {
            let (latent, video) = match segment.kind {
                SegmentKind::Text => {
                    parts.push(text.clone());
                    continue;
                }

                SegmentKind::Video => (&inputs.video, true),
                SegmentKind::Audio => (&inputs.audio, false),
                SegmentKind::KeyframeVideo(i) => {
                    (inputs.keyframes[i].video.as_ref().unwrap(), true)
                }

                SegmentKind::KeyframeAudio(i) => {
                    (inputs.keyframes[i].audio.as_ref().unwrap(), false)
                }

                SegmentKind::ReferenceVideo(i) => (inputs.references[i].video().unwrap(), true),
                SegmentKind::ReferenceAudio(i) => (inputs.references[i].audio().unwrap(), false),
            };

            let (values, features, name) = if video {
                (
                    patchify_video(latent),
                    c.video_patch_features(),
                    "video_patch_proj",
                )
            } else {
                (pack_audio(latent), c.audio_channels, "audio_patch_proj")
            };

            parts.push(w.linear(&upload(&values, features)?, name)?);
        }

        let mut hidden = Array::concat(&parts, false)?;
        drop(parts);
        let time = upload(&timesteps.time_embedding(&self.time_table), c.adaln_rank)?;
        let angles = upload(&layout.rope_angles(&self.inv_freq), c.rope_dims() / 2)?;
        let modulation = |name: &str, chunks: usize| -> Result<Array> {
            // The table has one row per timestep. Projections concatenate each modality's chunks.
            let a = w.linear(&time, name)?;
            let values = a.len();
            a.reshape(values / (chunks * width), chunks * width)
        };

        let mut blocks = Vec::new();
        for layer in 0..c.layers {
            let p = format!("blocks.{layer}");
            let m = modulation(&format!("{p}.adaln_proj.linear"), 6)?;
            let norm = w
                .norm(&hidden, &format!("{p}.norm1"), c.norm_eps)?
                .modulate(&m, &rows, 0, 1)?;
            hidden = hidden.add_gated(&self.attention(&norm, &p, Some(&angles))?, &m, &rows, 2)?;
            let norm = w
                .norm(&hidden, &format!("{p}.norm2"), c.norm_eps)?
                .modulate(&m, &rows, 3, 4)?;
            hidden = hidden.add_gated(&self.mlp(&norm, &p)?, &m, &rows, 5)?;

            if capture.contains(&layer) {
                blocks.push((layer, hidden.to_f32()?));
            }
        }

        let m = modulation("final_layer.adaln_proj.linear", 2)?;
        let project = |kind, name: &str| -> Result<Array> {
            let s = layout.segment(kind);
            let x = w.norm(
                &hidden.slice(s.start, s.end - s.start, 0, width)?,
                "final_layer.norm",
                c.norm_eps,
            )?;
            let m = m.slice(timesteps.index_of(kind), 1, 0, 2 * width)?;
            let rows = RowMap::new(d, &vec![0; x.rows])?;
            w.linear(&x.modulate(&m, &rows, 0, 1)?, name)
        };

        let video = project(SegmentKind::Video, "final_layer.video_out")?;
        let audio = project(SegmentKind::Audio, "final_layer.audio_out")?;
        let video = unpatchify_video(&video.to_f32()?, &inputs.video.shape)
            .into_iter()
            .map(|v| -v)
            .collect();
        let audio = unpack_audio(&audio.to_f32()?, &inputs.audio.shape)
            .into_iter()
            .map(|v| -v)
            .collect();
        Ok(DitOutput {
            text_states,
            blocks,
            video,
            audio,
            routed_fraction: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::WholeExchange;
    use std::path::Path;

    /// One rank sharing a step with nobody has to come out as a block that never went near a
    /// shard. It is not an equality: the exchange carries a block's attention output in bf16 and
    /// an ordinary block keeps it in FP32, so the two differ by that rounding and nothing else.
    ///
    /// The tiny fixture cannot reach this path — ConvRot wants a multiple of 256 features and its
    /// weights are FP32, not INT8 — so this runs against a real checkpoint when one is there.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn one_rank_attends_a_block_as_a_whole_one_does() {
        let path = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
            .join("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors");
        if !path.exists() {
            eprintln!("no checkpoint at {}, skipping", path.display());
            return;
        }

        let started = std::time::Instant::now();
        let mut dit = MetalDit::load(&SafeTensors::open(&path).unwrap(), "").unwrap();
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());

        // The turbo LoRA is in every real run here, so a rank that cannot take an adapted layer
        // cannot take a share of a real step. Loaded when it is there.
        let lora = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"))
            .join("loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors");
        if lora.exists() {
            let added = dit
                .add_lora(&SafeTensors::open(&lora).unwrap(), 1.0)
                .unwrap();
            eprintln!("{added} adapted layers");
        } else {
            eprintln!("no LoRA at {}, running without one", lora.display());
        }

        let (c, device) = (dit.config.clone(), dit.weights.device.clone());
        let tokens = 64;
        let values: Vec<f32> = (0..tokens * c.hidden)
            .map(|i| ((i * 31) % 251) as f32 / 251.0 - 0.5)
            .collect();
        let x = Array::from_f32(&device, tokens, c.hidden, &values).unwrap();

        let whole = dit
            .attention(&x, "blocks.0", None)
            .unwrap()
            .to_f32()
            .unwrap();
        let shard = Shard::even(0, 1, tokens, c.heads, 1);
        let mut exchange = WholeExchange::new(&device);
        let shared = dit
            .sharded_attention(&x, "blocks.0", None, &shard, &mut exchange)
            .unwrap()
            .to_f32()
            .unwrap();

        assert_eq!(shared.len(), whole.len());
        let largest = whole.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mut worst = 0.0f32;
        for (index, (&found, &want)) in shared.iter().zip(whole.iter()).enumerate() {
            let error = (found - want).abs();
            worst = worst.max(error);
            assert!(
                error <= largest * 0.02,
                "value {index}: {found} against {want}, largest {largest}"
            );
        }
        eprintln!(
            "worst {worst:.6} against a largest of {largest:.6}, {:.3}%",
            100.0 * worst / largest
        );
    }
}
