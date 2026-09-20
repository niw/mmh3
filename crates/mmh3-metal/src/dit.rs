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
    shard::{Exchange, ExchangeError, Region, Shard, VelocityRows, regions as shard_regions},
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
            .forward_inner(inputs, capture, sparse, &self.text, false, None)
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
        let config = DitConfig::of(file, prefix)?;
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
    pub fn sharded_attention(
        &self,
        normalized: &Array,
        prefix: &str,
        angles: Option<&Array>,
        shard: &Shard,
        exchange: &mut dyn Exchange<Memory = Memory>,
    ) -> Result<Array> {
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
    fn exchange_rows(
        &self,
        shard: &Shard,
        exchange: &mut dyn Exchange<Memory = Memory>,
        rows: &std::ops::Range<usize>,
    ) -> Result<()> {
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
    fn exchange_attended(
        &self,
        shard: &Shard,
        exchange: &mut dyn Exchange<Memory = Memory>,
        rows: &std::ops::Range<usize>,
        attended: &Array,
        whole: &Array,
    ) -> Result<()> {
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
                // NOTE: this rank's own rows never cross a wire, so they are taken from the array
                // as they are. Reading them back out of the region would put them through the
                // bf16 the exchange carries for nothing, and on a backend whose blocks run in
                // FP32 that rounding compounds: one block loses about 1%, fifty lose most of the
                // answer. Only what a peer sends is worth that.
                attended.slice(rows.start, rows.len(), 0, own_inner)?
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
        self.forward_inner(inputs, capture, sparse, &prepared.text, true, None)
    }

    /// One rank's share of a step. It carries the rows the shard gives it through every block and
    /// answers with the part of each segment those rows cover, still patchified and packed as the
    /// final projections leave it, since the rows of one rank cannot be unpatchified on their own.
    /// A leader puts the parts together; one rank's part is the whole.
    pub fn forward_shard(
        &self,
        inputs: &DitInputs,
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
        shard: &Shard,
        exchange: &mut dyn Exchange<Memory = Memory>,
    ) -> Result<VelocityRows> {
        let prepared = self.prepare_text(&inputs.context)?;
        let out = self.forward_inner(
            inputs,
            &[],
            sparse,
            &prepared.text,
            false,
            Some((shard, exchange)),
        )?;
        Ok(VelocityRows {
            rows: shard.own_tokens(),
            video: out.video,
            audio: out.audio,
        })
    }

    /// Puts the parts of a shared-out step back together, which only the rank answering to the
    /// caller needs to do. The parts may arrive in any order and must cover the sequence once.
    ///
    /// This is where the projections a rank answers with become the velocity: the negation
    /// happens here, once, and a rank that negated its own rows would be counted twice.
    ///
    /// It reads the config and the layout and nothing else — no weights, no device — so a leader
    /// can put a step together without holding a DiT it never runs.
    pub fn assemble_velocity(
        &self,
        inputs: &DitInputs,
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
        parts: &[VelocityRows],
    ) -> Result<DitOutput> {
        // A VSA run hands back rows in the order the attention wanted rather than the order the
        // video does, and this backend has nothing to put them back with. Refusing beats
        // assembling rows whose order nothing here can account for.
        if sparse.is_some() {
            return Err(Error(
                "Metal currently supports dense attention without VSA gates".into(),
            ));
        }

        let c = &self.config;
        let layout = PackedLayout::for_inputs(inputs);
        let gather = |kind: SegmentKind,
                      width: usize,
                      take: &dyn Fn(&VelocityRows) -> &Vec<f32>|
         -> Result<Vec<f32>> {
            let segment = layout.segment(kind);
            let mut values = vec![0.0f32; segment.len() * width];
            let mut covered = 0;
            for part in parts {
                let first = segment.start.max(part.rows.start);
                let last = segment.end.min(part.rows.end);
                if first >= last {
                    continue;
                }

                let rows = take(part);
                if rows.len() != (last - first) * width {
                    return Err(Error(format!(
                        "a part of {} values for {} rows",
                        rows.len(),
                        last - first
                    )));
                }
                values[(first - segment.start) * width..(last - segment.start) * width]
                    .copy_from_slice(rows);
                covered += last - first;
            }

            // A cut that leaves a gap would otherwise assemble zeros and look like a picture that
            // simply did not diffuse there, which is a long way from where the cut is.
            if covered != segment.len() {
                return Err(Error(format!(
                    "the parts cover {covered} of {} rows",
                    segment.len()
                )));
            }
            Ok(values)
        };

        let video = gather(SegmentKind::Video, c.video_patch_features(), &|part| {
            &part.video
        })?;
        let audio = gather(SegmentKind::Audio, c.audio_channels, &|part| &part.audio)?;
        Ok(DitOutput {
            text_states: Vec::new(),
            blocks: Vec::new(),
            video: unpatchify_video(&video, &inputs.video.shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            audio: unpack_audio(&audio, &inputs.audio.shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            routed_fraction: None,
        })
    }

    fn forward_inner(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
        text: &Array,
        capture_text: bool,
        mut shard: Option<(&Shard, &mut dyn Exchange<Memory = Memory>)>,
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
        // The rows this rank carries. Without a shard that is every row, and nothing below knows
        // the difference.
        let carried = match &shard {
            Some((shard, _)) => {
                let tokens = layout.len();
                if shard.tokens.iter().map(|rows| rows.len()).sum::<usize>() != tokens
                    || shard.heads.iter().map(|heads| heads.len()).sum::<usize>() != c.heads
                {
                    return Err(Error("a shard that does not cover the step".into()));
                }
                if !capture.is_empty() {
                    return Err(Error("a sharded step cannot capture blocks".into()));
                }
                // A rank exchanges the quantized block input, which only an INT8 projection has.
                if let Some(layer) = (0..c.layers).find(|layer| {
                    !self
                        .weights
                        .is_int8(&format!("blocks.{layer}.attn.qkv_proj"))
                }) {
                    return Err(Error(format!(
                        "a sharded step needs INT8 attention projections, and block {layer} has none"
                    )));
                }
                let rows = shard.own_tokens();
                // The text rows are refined once, at the front of the sequence, so they fall to
                // the first rank that carries anything. That is rank 0 while every rank takes a
                // share, and is not once a rank keeps its number and stops taking one.
                let first = shard.tokens.iter().position(|taken| !taken.is_empty());
                if !rows.is_empty()
                    && Some(shard.rank) != first
                    && rows.start < layout.segment(SegmentKind::Text).end
                {
                    return Err(Error(
                        "the text rows must fall to the first rank that carries any".into(),
                    ));
                }
                rows
            }
            None => 0..layout.len(),
        };
        let timesteps = StepTimesteps::for_layout(
            &layout,
            inputs.sigma,
            inputs.shift_video,
            inputs.shift_audio,
        );
        let rows = RowMap::new(d, &layout.modulation_rows(&timesteps)[carried.clone()])?;
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
        if carried.len() != hidden.shape()[0] {
            hidden = hidden.slice(carried.start, carried.len(), 0, width)?;
        }
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
            let attended = match &mut shard {
                Some((shard, exchange)) => {
                    self.sharded_attention(&norm, &p, Some(&angles), shard, *exchange)?
                }
                None => self.attention(&norm, &p, Some(&angles))?,
            };
            hidden = hidden.add_gated(&attended, &m, &rows, 2)?;
            let norm = w
                .norm(&hidden, &format!("{p}.norm2"), c.norm_eps)?
                .modulate(&m, &rows, 3, 4)?;
            hidden = hidden.add_gated(&self.mlp(&norm, &p)?, &m, &rows, 5)?;

            if capture.contains(&layer) {
                blocks.push((layer, hidden.to_f32()?));
            }
        }

        let m = modulation("final_layer.adaln_proj.linear", 2)?;
        let project = |kind, name: &str| -> Result<Vec<f32>> {
            let s = layout.segment(kind);
            let (first, last) = (s.start.max(carried.start), s.end.min(carried.end));
            if first >= last {
                return Ok(Vec::new());
            }

            let x = w.norm(
                &hidden.slice(first - carried.start, last - first, 0, width)?,
                "final_layer.norm",
                c.norm_eps,
            )?;
            let m = m.slice(timesteps.index_of(kind), 1, 0, 2 * width)?;
            let rows = RowMap::new(d, &vec![0; x.shape()[0]])?;
            w.linear(&x.modulate(&m, &rows, 0, 1)?, name)?.to_f32()
        };

        let video = project(SegmentKind::Video, "final_layer.video_out")?;
        let audio = project(SegmentKind::Audio, "final_layer.audio_out")?;
        // The rows of one rank cannot be unpatchified on their own, so a share answers with the
        // parts as the projections left them and a leader puts them together.
        //
        // NOTE: a share is NOT negated. A whole step answers with the velocity, which is the
        // negated projection, but `assemble_velocity` negates what the parts carry once it has
        // gathered them. Negating here as well would send this rank's rows the wrong way, and a
        // rank driven away from the data looks like a rank that never moved.
        let (video, audio) = if shard.is_some() {
            (video, audio)
        } else {
            (
                unpatchify_video(&video, &inputs.video.shape)
                    .into_iter()
                    .map(|v| -v)
                    .collect(),
                unpack_audio(&audio, &inputs.audio.shape)
                    .into_iter()
                    .map(|v| -v)
                    .collect(),
            )
        };
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

    /// The real checkpoint with the turbo LoRA on it, or nothing when the models are not here.
    /// The tiny fixture cannot stand in: ConvRot wants a multiple of 256 features and its weights
    /// are FP32 rather than INT8, so it fails the sharded path on both counts.
    fn checkpoint(with_lora: bool) -> Option<MetalDit> {
        let models = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let path = models.join("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors");
        if !path.exists() {
            eprintln!("no checkpoint at {}, skipping", path.display());
            return None;
        }

        let started = std::time::Instant::now();
        let mut dit = MetalDit::load(&SafeTensors::open(&path).unwrap(), "").unwrap();
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());

        // The turbo LoRA is in every real run here, so a rank that cannot take an adapted layer
        // cannot take a share of a real step.
        let lora =
            models.join("loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors");
        if with_lora && lora.exists() {
            let added = dit
                .add_lora(&SafeTensors::open(&lora).unwrap(), 1.0)
                .unwrap();
            eprintln!("{added} adapted layers");
        }
        Some(dit)
    }

    /// The one layer an exchanged input reaches, with an adapter on it, against the same layer
    /// given the rows a block would have had. This is where the adapted path can be checked
    /// honestly: one layer, with no fifty blocks of amplification behind it.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn an_exchanged_input_projects_to_what_the_block_would_have() {
        let Some(mut dit) = checkpoint(true) else {
            return;
        };
        dit.set_linear_precision(crate::LinearPrecision::Int8)
            .unwrap();
        let (c, device) = (dit.config.clone(), dit.weights.device.clone());
        let tokens = 8;
        let x = Array::from_f32(
            &device,
            tokens,
            c.hidden,
            &(0..tokens * c.hidden)
                .map(|i| ((i * 31) % 251) as f32 / 251.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let name = "blocks.0.attn.qkv_proj";
        let here = dit.weights.linear(&x, name).unwrap().to_f32().unwrap();

        let mut exchange = crate::shard::WholeExchange::new(&device);
        use mmh3_core::shard::Exchange as _;
        let region = exchange
            .region(Region::Normalized, tokens * c.hidden)
            .unwrap();
        let scales = region.write_quantized(0, &x).unwrap();
        let there = dit
            .weights
            .linear_quantized(region.buffer(), &scales, tokens, name)
            .unwrap()
            .to_f32()
            .unwrap();

        let largest = here.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = there
            .iter()
            .zip(here.iter())
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
        eprintln!(
            "qkv_proj: worst {worst:.6} of {largest:.6}, {:.4}%",
            100.0 * worst / largest
        );
        assert!(
            worst <= largest * 1e-5,
            "worst {worst} of a largest of {largest}"
        );
    }

    /// The smallest latent the layout accepts, to keep a fifty-block step short.
    fn sample_inputs(c: &DitConfig) -> DitInputs {
        let ramp = |count: usize, step: usize, modulus: usize| -> Vec<f32> {
            (0..count)
                .map(|i| ((i * step) % modulus) as f32 / modulus as f32 - 0.5)
                .collect()
        };
        let video_shape = vec![c.video_channels, 1, 2, 2];
        let audio_shape = vec![c.audio_channels, 2, 4];
        DitInputs {
            video: Tensor::new(
                video_shape.clone(),
                ramp(video_shape.iter().product(), 31, 251),
            ),
            audio: Tensor::new(
                audio_shape.clone(),
                ramp(audio_shape.iter().product(), 17, 173),
            ),
            context: Tensor::new(vec![8, c.text_dim], ramp(8 * c.text_dim, 11, 97)),
            context_modalities: Vec::new(),
            keyframes: Vec::new(),
            references: Vec::new(),
            sigma: 0.5,
            shift_video: 6.0,
            shift_audio: 3.0,
        }
    }

    /// How far apart two ordinary steps land when their inputs differ by a part in a million.
    /// This is here to stop anyone, including me, reading a whole step's output as a check on a
    /// path: at this sensitivity such a comparison measures the model and not the code. It asserts
    /// that the sensitivity is large, so if the model ever stops behaving this way, the reasoning
    /// the test beside it rests on stops being true and this says so.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn sensitivity_makes_a_whole_step_useless_as_a_comparison() {
        let Some(mut dit) = checkpoint(true) else {
            return;
        };
        dit.set_linear_precision(crate::LinearPrecision::Int8)
            .unwrap();
        let c = dit.config.clone();
        let inputs = sample_inputs(&c);
        let first = dit.forward(&inputs, &[], None).unwrap();
        let largest = first.video.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        let moved = |nudge: f32| -> f32 {
            let mut inputs = sample_inputs(&c);
            inputs.video.data[0] += nudge * inputs.video.data[0].abs().max(1e-3);
            let second = dit.forward(&inputs, &[], None).unwrap();
            second
                .video
                .iter()
                .zip(first.video.iter())
                .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()))
        };

        let (small, large) = (moved(1e-6), moved(1e-4));
        eprintln!(
            "of a largest of {largest:.6}: a part in a million moves {small:.6} ({:.2}%), a \
             hundred times that moves {large:.6} ({:.2}%)",
            100.0 * small / largest,
            100.0 * large / largest
        );
        assert!(
            small > largest * 0.1,
            "a part in a million no longer moves the output, so a whole step may be a fair \
             comparison after all: {small} of {largest}"
        );
        // NOTE: the discriminating check. A model that were merely ill-conditioned would answer a
        // nudge a hundred times larger with a response a hundred times larger. This one saturates,
        // because what moves is discrete: one INT8 value crosses a rounding edge and fifty blocks
        // later most of the velocity has changed, and crossing it by more changes no more. Without
        // this the test would read as "the model is delicate", which is the wrong lesson and would
        // send the next person looking for conditioning rather than for a boundary.
        assert!(
            large < small * 10.0,
            "the response scales with the nudge, so this is conditioning and not a boundary: \
             {small} against {large}"
        );
    }

    /// Two ranks against one, which is the first time the branches a single rank never enters do
    /// anything: the writes to a peer, the `Received(peer)` regions, and the offsets each rank
    /// addresses the others' rows by.
    ///
    /// NOTE: one thread runs one rank at a time, so a rank reaching a barrier cannot wait for one
    /// that has not been called yet. The sweep is run twice and the second is read: everything
    /// here depends only on the rows the ranks publish, which are fixed, so the second sweep sees
    /// what two machines would have seen at the first. What this cannot show is a transport that
    /// would deadlock, since nothing here ever waits.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn two_ranks_attend_a_block_as_one_rank_does() {
        let Some(mut dit) = checkpoint(true) else {
            return;
        };
        dit.set_linear_precision(crate::LinearPrecision::Int8)
            .unwrap();

        let (c, device) = (dit.config.clone(), dit.weights.device.clone());
        let tokens = 64;
        let x = Array::from_f32(
            &device,
            tokens,
            c.hidden,
            &(0..tokens * c.hidden)
                .map(|i| ((i * 31) % 251) as f32 / 251.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let alone = Shard::even(0, 1, tokens, c.heads, 1);
        let mut one = crate::shard::WholeExchange::new(&device);
        let whole = dit
            .sharded_attention(&x, "blocks.0", None, &alone, &mut one)
            .unwrap()
            .to_f32()
            .unwrap();

        let shards: Vec<Shard> = (0..2)
            .map(|r| Shard::even(r, 2, tokens, c.heads, 1))
            .collect();
        let mut exchanges = crate::shard::PairExchange::pair(&device, 2);
        let mut parts = vec![Vec::new(), Vec::new()];
        // Driven until it stops moving rather than a fixed number of times. How many sweeps it
        // takes is a property of the order one thread happens to call the ranks in, not of the
        // code under test, and a number here would be one nobody could justify. Three is what it
        // takes today; the assertion is that it settles at all.
        let mut sweeps = 0;
        loop {
            let previous = parts.clone();
            for rank in 0..2 {
                let rows = shards[rank].own_tokens();
                let mine = x.slice(rows.start, rows.len(), 0, c.hidden).unwrap();
                parts[rank] = dit
                    .sharded_attention(&mine, "blocks.0", None, &shards[rank], &mut exchanges[rank])
                    .unwrap()
                    .to_f32()
                    .unwrap();
            }
            sweeps += 1;
            if sweeps > 1 && parts == previous {
                break;
            }
            assert!(sweeps < 8, "the ranks never settled");
        }
        eprintln!("settled after {sweeps} sweeps");

        let together: Vec<f32> = parts.concat();
        assert_eq!(together.len(), whole.len());
        let largest = whole.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for rank in 0..2 {
            let rows = shards[rank].own_tokens();
            let want = &whole[rows.start * c.hidden..rows.end * c.hidden];
            let found = &parts[rank];
            let bad = found
                .iter()
                .zip(want.iter())
                .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
            eprintln!(
                "rank {rank}: worst {bad:.6} of {largest:.6}, {:.3}%",
                100.0 * bad / largest
            );
        }
        let worst = together
            .iter()
            .zip(whole.iter())
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
        eprintln!(
            "two ranks against one: worst {worst:.6} of {largest:.6}, {:.3}%",
            100.0 * worst / largest
        );
        // Two ranks put each other's shares through the bf16 the exchange carries them in, where
        // one rank keeps its own in FP32. That rounding is the whole of the difference.
        assert!(
            worst <= largest * 0.02,
            "worst {worst} of a largest of {largest}"
        );
    }

    /// The gather, the order and the sign, with no model behind them. The tiny fixture serves
    /// here where it cannot serve the sharded step, because assembling reads the config and the
    /// layout and never touches a weight.
    #[test]
    fn assembling_parts_rebuilds_a_whole_step() {
        let file = SafeTensors::open(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/dit_tiny.safetensors"
        )))
        .unwrap();
        let dit = MetalDit::load(&file, "weight.").unwrap();
        let c = dit.config.clone();
        let inputs = sample_inputs(&c);
        let layout = PackedLayout::for_inputs(&inputs);
        let (video, audio) = (
            layout.segment(SegmentKind::Video),
            layout.segment(SegmentKind::Audio),
        );
        let (video_width, audio_width) = (c.video_patch_features(), c.audio_channels);

        // A projection nobody computed, but one every row of which is distinguishable, so a row
        // put in the wrong place shows up as a wrong value rather than as a plausible one.
        let whole_video: Vec<f32> = (0..video.len() * video_width)
            .map(|i| i as f32 + 0.5)
            .collect();
        let whole_audio: Vec<f32> = (0..audio.len() * audio_width)
            .map(|i| -(i as f32) - 0.25)
            .collect();
        let slice = |whole: &[f32],
                     segment: &mmh3_core::dit::layout::Segment,
                     width,
                     rows: &std::ops::Range<usize>| {
            let first = segment.start.max(rows.start);
            let last = segment.end.min(rows.end);
            if first >= last {
                return Vec::new();
            }
            whole[(first - segment.start) * width..(last - segment.start) * width].to_vec()
        };
        let part = |rows: std::ops::Range<usize>| VelocityRows {
            video: slice(&whole_video, &video, video_width, &rows),
            audio: slice(&whole_audio, &audio, audio_width, &rows),
            rows,
        };

        // Cut anywhere, hand them over in any order: the answer is the whole step negated once.
        let tokens = layout.len();
        let cut = tokens / 3;
        let out = dit
            .assemble_velocity(&inputs, None, &[part(cut..tokens), part(0..cut)])
            .unwrap();
        let want_video: Vec<f32> = unpatchify_video(&whole_video, &inputs.video.shape)
            .into_iter()
            .map(|v| -v)
            .collect();
        let want_audio: Vec<f32> = unpack_audio(&whole_audio, &inputs.audio.shape)
            .into_iter()
            .map(|v| -v)
            .collect();
        assert_eq!(out.video, want_video, "video");
        assert_eq!(out.audio, want_audio, "audio");

        // A gap is refused rather than assembled as zeros, which would read as a patch of the
        // picture that simply did not diffuse.
        assert!(
            dit.assemble_velocity(&inputs, None, &[part(0..cut)])
                .is_err(),
            "parts that cover only part of the sequence should be refused"
        );
        // So is a part whose values do not match the rows it claims.
        let mut short = part(0..tokens);
        short.video.pop();
        assert!(
            dit.assemble_velocity(&inputs, None, &[short]).is_err(),
            "a part of the wrong length should be refused"
        );
    }

    /// The text is refined once and lives at the front, so the rank that carries the front of the
    /// sequence is the one that may hold it. That is rank 0 while every rank takes a share, and
    /// stops being rank 0 the moment a rank keeps its number and takes none — which is what a
    /// leader that orchestrates rather than computes does.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn the_text_rows_fall_to_the_first_rank_that_carries_any() {
        let Some(dit) = checkpoint(false) else {
            return;
        };
        let c = dit.config.clone();
        let inputs = sample_inputs(&c);
        let tokens = PackedLayout::for_inputs(&inputs).len();
        let text = PackedLayout::for_inputs(&inputs).segment(SegmentKind::Text);
        let cut = |ranks: Vec<std::ops::Range<usize>>, rank: usize| Shard {
            rank,
            tokens: ranks,
            heads: (0..2)
                .map(|r| r * c.heads / 2..(r + 1) * c.heads / 2)
                .collect(),
        };
        let run = |shard: &Shard| {
            let device = dit.weights.device.clone();
            let mut exchange = crate::shard::WholeExchange::new(&device);
            dit.forward_shard(&inputs, None, shard, &mut exchange)
                .err()
                .map(|error| error.0)
        };

        // NOTE: these shards are refused for other reasons too — one exchange cannot serve two
        // ranks — so what is asserted is whether THIS rule fired, not whether the step ran.
        const REFUSED: &str = "the text rows must fall to the first rank that carries any";
        let refused_for_text = |shard: &Shard| run(shard).as_deref() == Some(REFUSED);

        // Two ranks, both carrying. The text is at the front, so it is rank 0's alone.
        assert!(
            !refused_for_text(&cut(vec![0..text.end + 1, text.end + 1..tokens], 1)),
            "rank 1 starts past the text and should not be refused for holding it"
        );
        assert!(
            refused_for_text(&cut(vec![0..1, 1..tokens], 1)),
            "rank 1 reaching into the text should be refused"
        );

        // A rank that keeps its number and carries nothing. The front moves to rank 1, and rank 1
        // holding the text becomes correct rather than an error.
        assert!(
            !refused_for_text(&cut(vec![0..0, 0..tokens], 1)),
            "rank 1 is the first rank that carries anything, so the text is its own"
        );
        // And rank 0 is no longer privileged by its number: carrying nothing, it carries no text.
        assert!(
            !refused_for_text(&cut(vec![0..0, 0..tokens], 0)),
            "a rank with no rows holds no text rows"
        );
    }

    /// A whole step through the sharded path, one rank sharing with nobody, against the same step
    /// through the ordinary one. This is every block rather than one, so it is where a mistake in
    /// the rows a rank carries or in the tables sliced to them would show, which attending a
    /// single block cannot see.
    ///
    /// NOTE: no adapter here, and that is not laziness. Without one the two paths are the same
    /// arithmetic on the same bytes and this is an equality, which is the strongest thing it could
    /// be. With one they differ by the INT8 the exchange carries a block's input in, and
    /// `sensitivity_makes_a_whole_step_useless_as_a_comparison` beside this shows that a
    /// difference that small does not stay small. A threshold here would measure the model and not
    /// the code. What the adapted path is worth is asserted where it can be, one layer at a time.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn one_rank_runs_a_whole_step_as_a_whole_one_does() {
        let Some(mut dit) = checkpoint(false) else {
            return;
        };
        // The exchange carries a block's input as INT8. A step that is not shared runs its
        // activations at whatever road it is on, so only the INT8 road is the same arithmetic.
        dit.set_linear_precision(crate::LinearPrecision::Int8)
            .unwrap();

        let c = dit.config.clone();
        let inputs = sample_inputs(&c);

        let whole = dit.forward(&inputs, &[], None).unwrap();
        let tokens = PackedLayout::for_inputs(&inputs).len();
        let shard = Shard::even(0, 1, tokens, c.heads, 1);
        let mut exchange = crate::shard::WholeExchange::new(&dit.weights.device);
        let part = dit
            .forward_shard(&inputs, None, &shard, &mut exchange)
            .unwrap();
        assert_eq!(part.rows, 0..tokens, "one rank carries the whole sequence");

        // NOTE: assembled by the function a leader assembles with, not by a copy of it written
        // here. A test that spells out the convention itself agrees with whatever the code says,
        // which is how a negated share passed this comparison for a day.
        let assembled = dit.assemble_velocity(&inputs, None, &[part]).unwrap();

        for (name, found, want) in [
            ("video", &assembled.video, &whole.video),
            ("audio", &assembled.audio, &whole.audio),
        ] {
            assert_eq!(found.len(), want.len(), "{name} length");
            let largest = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = found
                .iter()
                .zip(want.iter())
                .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
            eprintln!(
                "{name}: worst {worst:.6} of {largest:.6}, {:.3}%",
                100.0 * worst / largest.max(f32::MIN_POSITIVE)
            );
            assert_eq!(worst, 0.0, "{name}: {worst} of a largest of {largest}");
        }
    }

    /// One rank sharing a step with nobody has to come out as a block that never went near a
    /// shard. It is not an equality: the exchange carries a block's attention output in bf16 and
    /// an ordinary block keeps it in FP32, so the two differ by that rounding and nothing else.
    ///
    /// The tiny fixture cannot reach this path — ConvRot wants a multiple of 256 features and its
    /// weights are FP32, not INT8 — so this runs against a real checkpoint when one is there.
    #[test]
    #[ignore = "loads a 20 GB checkpoint; run with --ignored when the models are present"]
    fn one_rank_attends_a_block_as_a_whole_one_does() {
        let Some(dit) = checkpoint(true) else {
            return;
        };

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
