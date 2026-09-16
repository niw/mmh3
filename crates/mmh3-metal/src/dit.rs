//! Dense DiT on Metal, including the shared keyframe and reference token layouts.
use crate::{
    Device, Error, Result,
    model::Weights,
    ops::{Array, RowMap},
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
    tensor::Tensor,
};

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
