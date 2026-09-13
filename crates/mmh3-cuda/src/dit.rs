//! One DiT call on the GPU for text-to-video with audio.
//!
//! The residual stream stays in FP32. Linear layers take BF16 inputs and run through the INT8 ConvRot GEMM for
//! quantized weights and through cuBLASLt for BF16 and FP32 weights.

use crate::attention::{AttentionLayout, HEAD_DIM, SparseWorkspace, dense_bf16_pointers, sparse_pointers};
use crate::loader::Uploader;
use crate::model::{DeviceTensors, LowRank, check_quantization, host_tensor, i32_buffer};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::dit::config::DitConfig;
use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::dit::latent::{pack_audio, patchify_video, unpack_audio, unpatchify_video};
use mmh3_core::dit::layout::{PackedLayout, SegmentKind};
use mmh3_core::dit::sparse::{SparseAttention, SparseSinks};
use mmh3_core::dit::timestep::{MODALITY_COUNT, StepTimesteps};
use mmh3_core::numeric::f32_to_bf16;
use mmh3_core::safetensors::{DType, SafeTensors};
use mmh3_core::tensor::Tensor;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::ptr;

pub use crate::model::Error;

unsafe extern "C" {
    fn mmh3_rms_norm_modulate(
        input: *const c_void,
        weight: *const c_void,
        modulation: *const c_void,
        rows: *const c_void,
        chunks: c_int,
        shift_chunk: c_int,
        scale_chunk: c_int,
        output: *mut c_void,
        output_is_f32: c_int,
        tokens: c_int,
        hidden: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_gated_residual_add(
        residual: *mut c_void,
        delta: *const c_void,
        modulation: *const c_void,
        rows: *const c_void,
        chunks: c_int,
        gate_chunk: c_int,
        tokens: c_int,
        hidden: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_qk_norm_rope(
        qkv: *mut c_void,
        query_weight: *const c_void,
        key_weight: *const c_void,
        angles: *const c_void,
        pairs: c_int,
        tokens: c_int,
        query_heads: c_int,
        key_heads: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_swiglu(input: *const c_void, output: *mut c_void, tokens: c_int, ffn: c_int, stream: *mut c_void) -> c_int;
    fn mmh3_modulation(
        time_embedding: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        steps: c_int,
        rank: c_int,
        outputs: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_bf16_to_f32(input: *const c_void, output: *mut c_void, count: usize, stream: *mut c_void) -> c_int;
}

/// Block modulation vectors per row: shift, scale and gate for attention, then for the MLP.
const BLOCK_CHUNKS: usize = 6;
/// Final layer modulation vectors per row: shift and scale.
const FINAL_CHUNKS: usize = 2;

/// Buffers sized for one sequence length.
struct Workspace {
    residual: DeviceBuffer,
    normalized: DeviceBuffer,
    projected: DeviceBuffer,
    qkv: DeviceBuffer,
    attention: DeviceBuffer,
    delta: DeviceBuffer,
    expanded: DeviceBuffer,
    activated: DeviceBuffer,
    quantized: DeviceBuffer,
    scales: DeviceBuffer,
    /// Intermediate rows of the low-rank adapters.
    adapter: DeviceBuffer,
}

impl Workspace {
    fn new(config: &DitConfig, tokens: usize, adapter_rank: usize) -> Result<Self, CudaError> {
        let (hidden, inner, ffn) = (config.hidden, config.inner(), config.ffn);
        Ok(Workspace {
            residual: DeviceBuffer::zeroed(tokens * hidden * 4)?,
            normalized: DeviceBuffer::new(tokens * hidden * 2)?,
            projected: DeviceBuffer::new(tokens * hidden * 4)?,
            qkv: DeviceBuffer::new(tokens * 3 * inner * 2)?,
            attention: DeviceBuffer::new(tokens * inner * 2)?,
            delta: DeviceBuffer::new(tokens * hidden * 2)?,
            expanded: DeviceBuffer::new(tokens * 2 * ffn * 2)?,
            activated: DeviceBuffer::new(tokens * ffn * 2)?,
            quantized: DeviceBuffer::new(tokens * hidden.max(inner).max(ffn))?,
            scales: DeviceBuffer::new(tokens * 4)?,
            adapter: DeviceBuffer::new(tokens * adapter_rank.max(1) * 2)?,
        })
    }
}

/// Outputs of one call and the states captured along the way.
pub struct DitOutputs {
    /// Refined text states `[text tokens, hidden]`.
    pub text_states: Vec<f32>,
    /// Residual stream `[tokens, hidden]` after each requested block.
    pub blocks: Vec<(usize, Vec<f32>)>,
    /// Velocity in the layout of the video input.
    pub video: Vec<f32>,
    /// Velocity in the layout of the audio input.
    pub audio: Vec<f32>,
    /// Mean fraction of key blocks Sol-Attn routed exactly, over the blocks, when it ran.
    pub routed_fraction: Option<f64>,
}

/// Sol-Attn for the blocks of one call.
struct SparsePass<'a> {
    workspace: &'a SparseWorkspace,
    tau: f32,
    sinks: SparseSinks,
}

pub struct CudaDit {
    config: DitConfig,
    tensors: DeviceTensors,
    /// Low-rank adapters by layer name.
    adapters: HashMap<String, LowRank>,
    adaln_table: Tensor,
    inverse_frequencies: Vec<f32>,
}

impl CudaDit {
    /// Uploads a pruned DiT checkpoint. `prefix` is prepended to every checkpoint tensor name.
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self, Error> {
        let config = DitConfig::from_shapes(|name| file.get(&format!("{prefix}{name}")).map(|info| info.shape.clone()))
            .map_err(Error::Model)?;
        if config.head_dim != HEAD_DIM {
            return Err(Error::Model(format!("head dimension {} is not {HEAD_DIM}", config.head_dim)));
        }
        let mut tensors = DeviceTensors::default();
        let mut uploader = Uploader::new(file);
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(prefix) else {
                continue;
            };
            if name.ends_with(".comfy_quant") {
                check_quantization(name, file.data(info))?;
                continue;
            }
            if name == "adaln_t_table" || name == "rope.inv_freq" {
                continue;
            }
            tensors.insert(name, file, info, &mut uploader)?;
        }
        uploader.run()?;
        Ok(CudaDit {
            config,
            tensors,
            adapters: HashMap::new(),
            adaln_table: host_tensor(file, &format!("{prefix}adaln_t_table"))?,
            inverse_frequencies: host_tensor(file, &format!("{prefix}rope.inv_freq"))?.data,
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    fn pointer(&self, name: &str) -> Result<*const c_void, Error> {
        self.tensors.pointer(name)
    }

    /// Adds the LoRA of a ComfyUI LoRA file at `strength`: every layer `{name}` with `{name}.lora_A.weight`,
    /// `{name}.lora_B.weight` and `{name}.alpha` gains `strength · alpha / rank · B · A` on top of its weight.
    pub fn add_lora(&mut self, file: &SafeTensors, strength: f32) -> Result<usize, Error> {
        const PREFIX: &str = "diffusion_model.";
        let mut uploader = Uploader::new(file);
        let mut adapters = Vec::new();
        for info in file.tensors() {
            let Some(layer) = info.name.strip_prefix(PREFIX).and_then(|name| name.strip_suffix(".lora_A.weight")) else {
                continue;
            };
            let weight = self.tensors.get(&format!("{layer}.weight"))?;
            let up = file.get(&format!("{PREFIX}{layer}.lora_B.weight")).ok_or_else(|| Error::Model(format!("{layer} has no lora_B")))?;
            let alpha = host_tensor(file, &format!("{PREFIX}{layer}.alpha"))?.data[0];
            let (rank, inputs, outputs) = (info.shape[0], info.shape[1], up.shape[0]);
            if info.dtype != DType::BF16 || up.dtype != DType::BF16 || up.shape[1] != rank || inputs != weight.shape[1] || outputs != weight.shape[0] {
                return Err(Error::Model(format!("the LoRA of {layer} does not fit the layer")));
            }
            if weight.dtype != DType::I8 && weight.dtype != DType::BF16 {
                return Err(Error::Model(format!("LoRA on {layer} needs BF16 activations")));
            }
            if self.adapters.contains_key(layer) {
                return Err(Error::Model(format!("{layer} already has a LoRA")));
            }
            let adapter = LowRank {
                down: uploader.allocate(file, info)?,
                up: uploader.allocate(file, up)?,
                rank,
                inputs,
                outputs,
                scale: strength * alpha / rank as f32,
            };
            adapters.push((layer.to_owned(), adapter));
        }
        uploader.run()?;
        let added = adapters.len();
        self.adapters.extend(adapters);
        Ok(added)
    }

    fn linear(&self, name: &str, input: *const c_void, output: *mut c_void, rows: usize, workspace: &Workspace) -> Result<(), Error> {
        self.tensors.linear(name, input, output, rows, &workspace.quantized, &workspace.scales)?;
        if let Some(adapter) = self.adapters.get(name) {
            // SAFETY: the layer read `rows × inputs` BF16 values from input and wrote `rows × outputs` to output.
            unsafe { adapter.apply(input, output, rows, &workspace.adapter)? };
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn normalize(
        &self,
        weight: &str,
        input: *const c_void,
        output: *mut c_void,
        output_is_f32: bool,
        tokens: usize,
        modulation: Option<(&DeviceBuffer, &DeviceBuffer, usize, usize, usize)>,
    ) -> Result<(), Error> {
        let (table, rows, chunks, shift, scale) = match modulation {
            Some((table, rows, chunks, shift, scale)) => (table.pointer().cast_const(), rows.pointer().cast_const(), chunks, shift, scale),
            None => (ptr::null(), ptr::null(), 0, 0, 0),
        };
        // SAFETY: input and output hold `tokens × hidden` values and the modulation rows index the table.
        check(unsafe {
            mmh3_rms_norm_modulate(
                input,
                self.pointer(weight)?,
                table,
                rows,
                chunks as c_int,
                shift as c_int,
                scale as c_int,
                output,
                output_is_f32 as c_int,
                tokens as c_int,
                self.config.hidden as c_int,
                self.config.norm_eps,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn add_residual(
        &self,
        workspace: &Workspace,
        tokens: usize,
        gate: Option<(&DeviceBuffer, &DeviceBuffer, usize)>,
    ) -> Result<(), Error> {
        let (table, rows, chunk) = match gate {
            Some((table, rows, chunk)) => (table.pointer().cast_const(), rows.pointer().cast_const(), chunk),
            None => (ptr::null(), ptr::null(), 0),
        };
        // SAFETY: residual and delta hold `tokens × hidden` values.
        check(unsafe {
            mmh3_gated_residual_add(
                workspace.residual.pointer(),
                workspace.delta.pointer(),
                table,
                rows,
                BLOCK_CHUNKS as c_int,
                chunk as c_int,
                tokens as c_int,
                self.config.hidden as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn modulation(&self, name: &str, time_embedding: &DeviceBuffer, steps: usize, output: &DeviceBuffer) -> Result<(), Error> {
        let weight = self.tensors.get(&format!("{name}.weight"))?;
        if weight.dtype != DType::F16 {
            return Err(Error::Model(format!("{name}: AdaLN weights must be F16 in pruned checkpoints")));
        }
        // SAFETY: output holds `steps × outputs` values and the weight is `[outputs, rank]`.
        check(unsafe {
            mmh3_modulation(
                time_embedding.pointer(),
                weight.buffer.pointer(),
                self.pointer(&format!("{name}.bias"))?,
                output.pointer(),
                steps as c_int,
                weight.shape[1] as c_int,
                weight.shape[0] as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Attention and MLP halves of a block or refiner block. `modulation` carries the block modulation and `sparse`
    /// replaces dense attention with Sol-Attn.
    fn transformer_block(
        &self,
        prefix: &str,
        workspace: &Workspace,
        tokens: usize,
        angles: Option<&DeviceBuffer>,
        modulation: Option<(&DeviceBuffer, &DeviceBuffer)>,
        sparse: Option<&SparsePass>,
    ) -> Result<(), Error> {
        let config = &self.config;
        let modulate = |shift, scale| modulation.map(|(table, rows)| (table, rows, BLOCK_CHUNKS, shift, scale));
        let gate = |chunk| modulation.map(|(table, rows)| (table, rows, chunk));

        self.normalize(&format!("{prefix}.norm1.weight"), workspace.residual.pointer(), workspace.normalized.pointer(), false, tokens, modulate(0, 1))?;
        self.linear(&format!("{prefix}.attn.qkv_proj"), workspace.normalized.pointer(), workspace.qkv.pointer(), tokens, workspace)?;
        // SAFETY: qkv holds `tokens × 3 × heads × 128` values and the angles cover every token.
        unsafe {
            check(mmh3_qk_norm_rope(
                workspace.qkv.pointer(),
                self.pointer(&format!("{prefix}.attn.q_norm.weight"))?,
                self.pointer(&format!("{prefix}.attn.k_norm.weight"))?,
                angles.map_or(ptr::null(), |angles| angles.pointer().cast_const()),
                (3 * config.rope_frequencies) as c_int,
                tokens as c_int,
                config.heads as c_int,
                config.heads as c_int,
                config.norm_eps,
                ptr::null_mut(),
            ))?;
            let scale = 1.0 / (config.head_dim as f32).sqrt();
            match sparse {
                None => dense_bf16_pointers(workspace.qkv.pointer(), workspace.attention.pointer(), tokens, config.heads, scale)?,
                Some(pass) => {
                    let inner = config.inner();
                    let layout = AttentionLayout {
                        token_stride: [(3 * inner) as i64, (3 * inner) as i64, (3 * inner) as i64, inner as i64],
                        head_stride: [HEAD_DIM as i64; 4],
                        ..AttentionLayout::default()
                    };
                    let qkv = workspace.qkv.pointer().cast::<u16>();
                    sparse_pointers(
                        qkv.cast(),
                        qkv.add(inner).cast(),
                        qkv.add(2 * inner).cast(),
                        workspace.attention.pointer(),
                        tokens,
                        config.heads,
                        &layout,
                        scale,
                        pass.tau,
                        pass.sinks,
                        pass.workspace,
                    )?
                }
            }
        }
        self.linear(&format!("{prefix}.attn.out_proj"), workspace.attention.pointer(), workspace.delta.pointer(), tokens, workspace)?;
        self.add_residual(workspace, tokens, gate(2))?;

        self.normalize(&format!("{prefix}.norm2.weight"), workspace.residual.pointer(), workspace.normalized.pointer(), false, tokens, modulate(3, 4))?;
        self.linear(&format!("{prefix}.mlp.fc1"), workspace.normalized.pointer(), workspace.expanded.pointer(), tokens, workspace)?;
        // SAFETY: expanded holds `tokens × 2 × ffn` values and activated `tokens × ffn`.
        check(unsafe {
            mmh3_swiglu(workspace.expanded.pointer(), workspace.activated.pointer(), tokens as c_int, config.ffn as c_int, ptr::null_mut())
        })?;
        self.linear(&format!("{prefix}.mlp.fc2"), workspace.activated.pointer(), workspace.delta.pointer(), tokens, workspace)?;
        self.add_residual(workspace, tokens, gate(5))
    }

    /// Writes the refined text states into the first rows of the residual stream.
    fn refine_text(&self, context: &Tensor, workspace: &Workspace) -> Result<(), Error> {
        let tokens = context.shape[0];
        let context_bytes: Vec<u8> = context.data.iter().flat_map(|&value| f32_to_bf16(value).to_le_bytes()).collect();
        let context_buffer = DeviceBuffer::from_bytes(&context_bytes)?;
        self.linear("condition_proj", context_buffer.pointer(), workspace.normalized.pointer(), tokens, workspace)?;
        // SAFETY: both buffers hold at least `tokens × hidden` values.
        check(unsafe {
            mmh3_bf16_to_f32(workspace.normalized.pointer(), workspace.residual.pointer(), tokens * self.config.hidden, ptr::null_mut())
        })?;
        for layer in 0..self.config.refiner_layers {
            self.transformer_block(&format!("token_refiner.blocks.{layer}"), workspace, tokens, None, None, None)?;
        }
        self.normalize("token_refiner.final_norm.weight", workspace.residual.pointer(), workspace.residual.pointer(), true, tokens, None)
    }

    /// Runs one DiT call and returns the velocity, capturing the residual stream after the listed blocks. `sparse`
    /// switches the blocks to Sol-Attn when the sequence is long enough.
    pub fn forward(&self, inputs: &DitInputs, capture: &[usize], sparse: Option<&SparseAttention>) -> Result<DitOutputs, Error> {
        let config = &self.config;
        let hidden = config.hidden;
        let video_shape = &inputs.video.shape;
        let text_tokens = inputs.context.shape[0];
        let layout = PackedLayout::text_to_video(text_tokens, video_shape[1], video_shape[2], video_shape[3], inputs.audio.shape[2]);
        let tokens = layout.len();
        let timesteps = StepTimesteps::new(inputs.sigma, inputs.shift_video, inputs.shift_audio);
        let steps = timesteps.values.len();
        let adapter_rank = self.adapters.values().map(|adapter| adapter.rank).max().unwrap_or(0);
        let workspace = Workspace::new(config, tokens, adapter_rank)?;

        let block_rows = i32_buffer(&layout.modulation_rows(&timesteps))?;
        let final_rows: Vec<usize> =
            layout.segments.iter().flat_map(|segment| std::iter::repeat_n(timesteps.index_of(segment.kind), segment.len())).collect();
        let final_rows = i32_buffer(&final_rows)?;
        let angles = DeviceBuffer::from_f32(&layout.rope_angles(&self.inverse_frequencies))?;
        let time_embedding = DeviceBuffer::from_f32(&timesteps.time_embedding(&self.adaln_table))?;

        self.refine_text(&inputs.context, &workspace)?;
        let text_states = workspace.residual.to_f32()?[..text_tokens * hidden].to_vec();

        let audio = layout.segment(SegmentKind::Audio);
        let video = layout.segment(SegmentKind::Video);
        let audio_rows = DeviceBuffer::from_f32(&pack_audio(&inputs.audio))?;
        let video_rows = DeviceBuffer::from_f32(&patchify_video(&inputs.video))?;
        self.linear("audio_patch_proj", audio_rows.pointer(), workspace.residual.pointer_at(audio.start * hidden * 4), audio.len(), &workspace)?;
        self.linear("video_patch_proj", video_rows.pointer(), workspace.residual.pointer_at(video.start * hidden * 4), video.len(), &workspace)?;

        let sparse_workspace = match sparse {
            Some(settings) if tokens >= settings.min_tokens => Some((SparseWorkspace::new(tokens, config.heads)?, settings.tau)),
            _ => None,
        };
        let sparse_pass = sparse_workspace.as_ref().map(|(workspace, tau)| SparsePass { workspace, tau: *tau, sinks: SparseSinks::for_layout(&layout) });
        let mut routed = 0.0;
        let modulation = DeviceBuffer::new(steps * MODALITY_COUNT * BLOCK_CHUNKS * hidden * 4)?;
        let mut blocks = Vec::new();
        for layer in 0..config.layers {
            let prefix = format!("blocks.{layer}");
            self.modulation(&format!("{prefix}.adaln_proj.linear"), &time_embedding, steps, &modulation)?;
            self.transformer_block(&prefix, &workspace, tokens, Some(&angles), Some((&modulation, &block_rows)), sparse_pass.as_ref())?;
            if let Some(pass) = &sparse_pass {
                routed += pass.workspace.routed_fraction()?;
            }
            if capture.contains(&layer) {
                blocks.push((layer, workspace.residual.to_f32()?));
            }
        }

        let final_modulation = DeviceBuffer::new(steps * FINAL_CHUNKS * hidden * 4)?;
        self.modulation("final_layer.adaln_proj.linear", &time_embedding, steps, &final_modulation)?;
        self.normalize(
            "final_layer.norm.weight",
            workspace.residual.pointer(),
            workspace.projected.pointer(),
            true,
            tokens,
            Some((&final_modulation, &final_rows, FINAL_CHUNKS, 0, 1)),
        )?;
        let project = |segment: mmh3_core::dit::layout::Segment, head: &str, width: usize| -> Result<Vec<f32>, Error> {
            let output = DeviceBuffer::new(segment.len() * width * 4)?;
            self.linear(head, workspace.projected.pointer_at(segment.start * hidden * 4), output.pointer(), segment.len(), &workspace)?;
            Ok(output.to_f32()?)
        };
        let video_velocity = project(video, "final_layer.video_out", config.video_patch_features())?;
        let audio_velocity = project(audio, "final_layer.audio_out", config.audio_channels)?;
        Ok(DitOutputs {
            text_states,
            blocks,
            video: unpatchify_video(&video_velocity, video_shape).into_iter().map(|value| -value).collect(),
            audio: unpack_audio(&audio_velocity, &inputs.audio.shape).into_iter().map(|value| -value).collect(),
            routed_fraction: sparse_pass.map(|_| routed / config.layers as f64),
        })
    }
}
