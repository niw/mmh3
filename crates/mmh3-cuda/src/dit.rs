//! One DiT call on the GPU for text-to-video with audio.
//!
//! The residual stream stays in FP32. Linear layers take BF16 inputs and run through the INT8
//! ConvRot GEMM for quantized weights and through cuBLASLt for BF16 and FP32 weights.

use crate::attention::{
    AttentionInputs, AttentionLayout, AttentionPrecision, HEAD_DIM, PreparedAttention,
    QuantizedWorkspace, SparseWorkspace, VsaWorkspace, dense_bf16_pointers,
    dense_quantized_pointers, prepare_inputs_pointers, sparse_pointers, vsa_pointers,
};
use crate::gemm::interleave_swiglu_rows;
use crate::loader::Uploader;
use crate::model::{DeviceTensors, LowRank, check_quantization, host_tensor, i32_buffer};
use crate::{CudaError, DeviceBuffer, check, copy_device};
use mmh3_core::dit::config::DitConfig;
use mmh3_core::dit::inputs::DitInputs;
use mmh3_core::dit::latent::{pack_audio, patchify_video, unpack_audio, unpatchify_video};
use mmh3_core::dit::layout::{PackedLayout, SegmentKind};
use mmh3_core::dit::sparse::{SparseAttention, SparseMethod, SparseSinks};
use mmh3_core::dit::timestep::{MODALITY_COUNT, StepTimesteps};
use mmh3_core::dit::vsa::VsaPlan;
use mmh3_core::numeric::f32_to_bf16;
use mmh3_core::safetensors::{DType, SafeTensors};
use mmh3_core::tensor::Tensor;
use std::cell::RefCell;
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
    fn mmh3_swiglu(
        input: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        ffn: c_int,
        stream: *mut c_void,
    ) -> c_int;
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
    fn mmh3_bf16_to_f32(
        input: *const c_void,
        output: *mut c_void,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_add_norm_quantize(
        residual: *mut c_void,
        delta: *const c_void,
        gate_modulation: *const c_void,
        modulation: *const c_void,
        rows: *const c_void,
        chunks: c_int,
        gate_chunk: c_int,
        shift_chunk: c_int,
        scale_chunk: c_int,
        weight: *const c_void,
        normalized: *mut c_void,
        quantized: *mut c_void,
        scales: *mut c_void,
        tokens: c_int,
        hidden: c_int,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int;
}

/// Block modulation vectors per row: shift, scale and gate for attention, then for the MLP.
const BLOCK_CHUNKS: usize = 6;
/// The chunk of the block modulation that gates the MLP output.
const MLP_GATE_CHUNK: usize = 5;
/// Final layer modulation vectors per row: shift and scale.
const FINAL_CHUNKS: usize = 2;

/// What a workspace is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorkspaceShape {
    tokens: usize,
    /// Rows of the gate and up projections, which INT8 MLPs need only for the text tokens as they
    /// write SwiGLU directly.
    expanded_rows: usize,
    adapter_rank: usize,
    /// Whether the blocks have VSA gates.
    gated: bool,
}

/// Buffers sized for one sequence length.
struct Workspace {
    shape: WorkspaceShape,
    attention_quantized: Option<QuantizedWorkspace>,
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
    /// VSA gates of the coarse branch, BF16 `[tokens, heads × 128]`.
    gate: Option<DeviceBuffer>,
}

impl Workspace {
    fn new(config: &DitConfig, shape: WorkspaceShape) -> Result<Self, CudaError> {
        let (hidden, inner, ffn) = (config.hidden, config.inner(), config.ffn);
        let WorkspaceShape {
            tokens,
            expanded_rows,
            adapter_rank,
            gated,
        } = shape;
        Ok(Workspace {
            shape,
            attention_quantized: None,
            residual: DeviceBuffer::zeroed(tokens * hidden * 4)?,
            normalized: DeviceBuffer::new(tokens * hidden * 2)?,
            projected: DeviceBuffer::new(tokens * hidden * 4)?,
            qkv: DeviceBuffer::new(tokens * 3 * inner * 2)?,
            attention: DeviceBuffer::new(tokens * inner * 2)?,
            delta: DeviceBuffer::new(tokens * hidden * 2)?,
            expanded: DeviceBuffer::new(expanded_rows.max(1) * 2 * ffn * 2)?,
            activated: DeviceBuffer::new(tokens * ffn * 2)?,
            quantized: DeviceBuffer::new(tokens * hidden.max(inner).max(ffn))?,
            scales: DeviceBuffer::new(tokens * 4)?,
            adapter: DeviceBuffer::new(tokens * adapter_rank.max(1) * 2)?,
            gate: if gated {
                Some(DeviceBuffer::new(tokens * inner * 2)?)
            } else {
                None
            },
        })
    }
}

/// Buffers that one call leaves to the next. The steps of a generation share one sequence length,
/// so they allocate these gigabytes once, which saves about 0.3 s per step at 768p.
#[derive(Default)]
struct ForwardBuffers {
    workspace: Option<Workspace>,
    sparse: Option<(AttentionPrecision, SparseWorkspace)>,
    vsa: Option<(AttentionPrecision, VsaPlan, VsaWorkspace)>,
    text: Option<TextStates>,
}

/// Refined text states `[text tokens, hidden]` in FP32 and the context they came from. The refiner
/// sees only the context, so the steps of a generation share them.
struct TextStates {
    context: Tensor,
    states: DeviceBuffer,
}

/// Outputs of one call and the states captured along the way.
pub struct DitOutputs {
    /// Residual stream `[tokens, hidden]` after each requested block.
    pub blocks: Vec<(usize, Vec<f32>)>,
    /// Velocity in the layout of the video input.
    pub video: Vec<f32>,
    /// Velocity in the layout of the audio input.
    pub audio: Vec<f32>,
    /// Mean fraction of key blocks Sol-Attn routed exactly, over the blocks, when it ran.
    pub routed_fraction: Option<f64>,
}

/// Block-sparse attention for the blocks of one call.
enum SparsePass<'a> {
    Sol {
        workspace: &'a SparseWorkspace,
        tau: f32,
        sinks: SparseSinks,
    },
    /// VSA on the sequence in the workspace's tile order, keeping `kept` video tiles.
    Vsa {
        workspace: &'a VsaWorkspace,
        kept: usize,
    },
}

pub struct CudaDit {
    attention_precision: AttentionPrecision,
    config: DitConfig,
    tensors: DeviceTensors,
    /// Low-rank adapters by layer name.
    adapters: HashMap<String, LowRank>,
    adaln_table: Tensor,
    inverse_frequencies: Vec<f32>,
    buffers: RefCell<ForwardBuffers>,
}

impl CudaDit {
    /// Uploads a pruned DiT checkpoint. `prefix` is prepended to every checkpoint tensor name.
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self, Error> {
        let config = DitConfig::from_shapes(|name| {
            file.get(&format!("{prefix}{name}"))
                .map(|info| info.shape.clone())
        })
        .map_err(Error::Model)?;
        if config.head_dim != HEAD_DIM {
            return Err(Error::Model(format!(
                "head dimension {} is not {HEAD_DIM}",
                config.head_dim
            )));
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
        for layer in 0..config.layers {
            let fc1 = format!("blocks.{layer}.mlp.fc1");
            if tensors.is_int8(&fc1) {
                tensors.interleave_swiglu(&format!("{fc1}.weight"))?;
                tensors.interleave_swiglu(&format!("{fc1}.weight_scale"))?;
            }
        }
        Ok(CudaDit {
            attention_precision: AttentionPrecision::Bf16,
            config,
            tensors,
            adapters: HashMap::new(),
            adaln_table: host_tensor(file, &format!("{prefix}adaln_t_table"))?,
            inverse_frequencies: host_tensor(file, &format!("{prefix}rope.inv_freq"))?.data,
            buffers: RefCell::default(),
        })
    }

    /// Selects quantized attention for the main DiT blocks. The text refiner stays in BF16.
    pub fn set_attention_precision(&mut self, precision: AttentionPrecision) {
        self.attention_precision = precision;
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    /// Whether the blocks carry the gates of VSA's coarse branch, which VSA-trained checkpoints
    /// such as FastH3 have.
    pub fn has_vsa_gates(&self) -> bool {
        self.tensors
            .optional("blocks.0.attn.to_gate_compress.weight")
            .is_some()
    }

    /// Refined text states `[text tokens, hidden]` of the last call's context, or `None` before the
    /// first call or after `add_lora`.
    pub fn text_states(&self) -> Result<Option<Vec<f32>>, Error> {
        let buffers = self.buffers.borrow();
        Ok(buffers
            .text
            .as_ref()
            .map(|text| text.states.to_f32())
            .transpose()?)
    }

    fn pointer(&self, name: &str) -> Result<*const c_void, Error> {
        self.tensors.pointer(name)
    }

    /// Adds the LoRA of a ComfyUI LoRA file at `strength`: every layer `{name}` with
    /// `{name}.lora_A.weight`, `{name}.lora_B.weight` and `{name}.alpha` gains
    /// `strength · alpha / rank · B · A` on top of its weight.
    pub fn add_lora(&mut self, file: &SafeTensors, strength: f32) -> Result<usize, Error> {
        const PREFIX: &str = "diffusion_model.";
        let mut uploader = Uploader::new(file);
        let mut adapters = Vec::new();
        for info in file.tensors() {
            let Some(layer) = info
                .name
                .strip_prefix(PREFIX)
                .and_then(|name| name.strip_suffix(".lora_A.weight"))
            else {
                continue;
            };
            let weight = self.tensors.get(&format!("{layer}.weight"))?;
            let up = file
                .get(&format!("{PREFIX}{layer}.lora_B.weight"))
                .ok_or_else(|| Error::Model(format!("{layer} has no lora_B")))?;
            let alpha = host_tensor(file, &format!("{PREFIX}{layer}.alpha"))?.data[0];
            let (rank, inputs, outputs) = (info.shape[0], info.shape[1], up.shape[0]);
            if info.dtype != DType::BF16
                || up.dtype != DType::BF16
                || up.shape[1] != rank
                || inputs != weight.shape[1]
                || outputs != weight.shape[0]
            {
                return Err(Error::Model(format!(
                    "the LoRA of {layer} does not fit the layer"
                )));
            }
            if weight.dtype != DType::I8 && weight.dtype != DType::BF16 {
                return Err(Error::Model(format!(
                    "LoRA on {layer} needs BF16 activations"
                )));
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
        for (layer, adapter) in &mut adapters {
            if layer.ends_with(".mlp.fc1") && self.tensors.is_int8(layer) {
                adapter.up =
                    interleave_swiglu_rows(&adapter.up, adapter.outputs, adapter.rank * 2)?;
            }
        }
        let added = adapters.len();
        self.adapters.extend(adapters);
        // The adapters may change the refiner.
        self.buffers.get_mut().text = None;
        Ok(added)
    }

    fn linear(
        &self,
        name: &str,
        input: *const c_void,
        output: *mut c_void,
        rows: usize,
        workspace: &Workspace,
    ) -> Result<(), Error> {
        let adapter = self
            .adapters
            .get(name)
            .map(|adapter| (adapter, &workspace.adapter));
        self.tensors.linear(
            name,
            input,
            output,
            rows,
            &workspace.quantized,
            &workspace.scales,
            adapter,
        )
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
            Some((table, rows, chunks, shift, scale)) => (
                table.pointer().cast_const(),
                rows.pointer().cast_const(),
                chunks,
                shift,
                scale,
            ),
            None => (ptr::null(), ptr::null(), 0, 0, 0),
        };
        // SAFETY: input and output hold `tokens × hidden` values and the modulation rows index the
        // table.
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

    /// Adds the delta to the residual, gated by a chunk of a modulation table when `gate` names
    /// one, then normalizes the residual with `weight` and the modulation's shift and scale chunks
    /// and quantizes it into the workspace for the INT8 layer `layer`. The BF16 rows stay in
    /// `normalized` only when the layer's adapter needs them.
    #[allow(clippy::too_many_arguments)]
    fn add_norm_quantize(
        &self,
        weight: &str,
        layer: &str,
        workspace: &Workspace,
        tokens: usize,
        (table, rows): (&DeviceBuffer, &DeviceBuffer),
        gate: Option<(&DeviceBuffer, usize)>,
        shift: usize,
        scale: usize,
    ) -> Result<(), Error> {
        let hidden = self.config.hidden;
        assert!(
            workspace.quantized.bytes() >= tokens * hidden
                && workspace.scales.bytes() >= tokens * 4,
            "the quantization buffers are too small"
        );
        let normalized = if self.adapters.contains_key(layer) {
            workspace.normalized.pointer()
        } else {
            ptr::null_mut()
        };
        // SAFETY: the residual, delta and normalized buffers hold `tokens × hidden` values, the
        // quantization buffers were checked above, and the modulation rows index the table.
        check(unsafe {
            mmh3_add_norm_quantize(
                workspace.residual.pointer(),
                gate.map_or(ptr::null(), |_| workspace.delta.pointer().cast_const()),
                gate.map_or(ptr::null(), |(gate_table, _)| {
                    gate_table.pointer().cast_const()
                }),
                table.pointer(),
                rows.pointer(),
                BLOCK_CHUNKS as c_int,
                gate.map_or(0, |(_, chunk)| chunk) as c_int,
                shift as c_int,
                scale as c_int,
                self.pointer(weight)?,
                normalized,
                workspace.quantized.pointer(),
                workspace.scales.pointer(),
                tokens as c_int,
                hidden as c_int,
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
            Some((table, rows, chunk)) => (
                table.pointer().cast_const(),
                rows.pointer().cast_const(),
                chunk,
            ),
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

    fn modulation(
        &self,
        name: &str,
        time_embedding: &DeviceBuffer,
        steps: usize,
        output: &DeviceBuffer,
    ) -> Result<(), Error> {
        let weight = self.tensors.get(&format!("{name}.weight"))?;
        if weight.dtype != DType::F16 {
            return Err(Error::Model(format!(
                "{name}: AdaLN weights must be F16 in pruned checkpoints"
            )));
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

    /// Attention and MLP halves of a block or refiner block. `modulation` carries the block
    /// modulation and `sparse` replaces dense attention with Sol-Attn.
    ///
    /// A modulated block leaves its MLP output in `delta` instead of adding it to the residual, so
    /// that the next block adds it in its first normalization. `pending` is the modulation table of
    /// the block that left it, and `add_pending` adds it on its own.
    #[allow(clippy::too_many_arguments)]
    fn transformer_block(
        &self,
        prefix: &str,
        workspace: &Workspace,
        tokens: usize,
        angles: Option<&DeviceBuffer>,
        modulation: Option<(&DeviceBuffer, &DeviceBuffer)>,
        pending: Option<&DeviceBuffer>,
        sparse: Option<&SparsePass>,
    ) -> Result<(), Error> {
        let config = &self.config;
        let modulate = |shift, scale| {
            modulation.map(|(table, rows)| (table, rows, BLOCK_CHUNKS, shift, scale))
        };
        let gate = |chunk| modulation.map(|(table, rows)| (table, rows, chunk));

        let qkv = format!("{prefix}.attn.qkv_proj");
        let adapter = |layer: &str| {
            self.adapters
                .get(layer)
                .map(|adapter| (adapter, &workspace.adapter))
        };
        // Modulated blocks normalize and quantize the inputs of their INT8 layers in one pass.
        if let Some(modulation) = modulation.filter(|_| self.tensors.is_int8(&qkv)) {
            let pending_gate = pending.map(|table| (table, MLP_GATE_CHUNK));
            self.add_norm_quantize(
                &format!("{prefix}.norm1.weight"),
                &qkv,
                workspace,
                tokens,
                modulation,
                pending_gate,
                0,
                1,
            )?;
            self.tensors.linear_quantized(
                &qkv,
                workspace.normalized.pointer(),
                workspace.qkv.pointer(),
                tokens,
                &workspace.quantized,
                &workspace.scales,
                adapter(&qkv),
                false,
            )?;
        } else {
            if let Some((table, rows)) = pending.zip(modulation.map(|(_, rows)| rows)) {
                self.add_pending(workspace, tokens, table, rows)?;
            }
            self.normalize(
                &format!("{prefix}.norm1.weight"),
                workspace.residual.pointer(),
                workspace.normalized.pointer(),
                false,
                tokens,
                modulate(0, 1),
            )?;
            self.linear(
                &qkv,
                workspace.normalized.pointer(),
                workspace.qkv.pointer(),
                tokens,
                workspace,
            )?;
        }
        let vsa_gate = match sparse {
            Some(SparsePass::Vsa { .. }) => {
                let fused = modulation.is_some() && self.tensors.is_int8(&qkv);
                self.vsa_gate(prefix, workspace, tokens, fused)?
            }
            _ => None,
        };
        let query_norm = self.pointer(&format!("{prefix}.attn.q_norm.weight"))?;
        let key_norm = self.pointer(&format!("{prefix}.attn.k_norm.weight"))?;
        let angles = angles.map_or(ptr::null(), |angles| angles.pointer().cast_const());
        let pairs = 3 * config.rope_frequencies;
        let quantized = workspace
            .attention_quantized
            .as_ref()
            .filter(|_| modulation.is_some());
        // Sparse and quantized attention take their per-block inputs from the pass that normalizes
        // q and k.
        let prepared = match (sparse, quantized) {
            (Some(SparsePass::Sol { workspace, .. }), _) => {
                Some(PreparedAttention::Sparse(workspace))
            }
            (Some(SparsePass::Vsa { workspace, .. }), _) => Some(PreparedAttention::Vsa(workspace)),
            (None, Some(quantized)) => Some(PreparedAttention::DenseQuantized(quantized)),
            (None, None) => None,
        };
        let inputs = if prepared.is_some() {
            AttentionInputs::Prepared
        } else {
            AttentionInputs::Raw
        };
        // SAFETY: qkv holds `tokens × 3 × heads × 128` values and the angles cover every token.
        unsafe {
            match prepared {
                Some(attention) => prepare_inputs_pointers(
                    workspace.qkv.pointer(),
                    query_norm,
                    key_norm,
                    angles,
                    pairs,
                    tokens,
                    config.heads,
                    config.norm_eps,
                    attention,
                )?,
                None => check(mmh3_qk_norm_rope(
                    workspace.qkv.pointer(),
                    query_norm,
                    key_norm,
                    angles,
                    pairs as c_int,
                    tokens as c_int,
                    config.heads as c_int,
                    config.heads as c_int,
                    config.norm_eps,
                    ptr::null_mut(),
                ))?,
            }
            let scale = 1.0 / (config.head_dim as f32).sqrt();
            match sparse {
                None => match quantized {
                    Some(quantized) => dense_quantized_pointers(
                        workspace.qkv.pointer(),
                        workspace.attention.pointer(),
                        scale,
                        quantized,
                        inputs,
                    )?,
                    None => dense_bf16_pointers(
                        workspace.qkv.pointer(),
                        workspace.attention.pointer(),
                        tokens,
                        config.heads,
                        scale,
                    )?,
                },
                Some(pass) => {
                    let inner = config.inner();
                    let layout = AttentionLayout {
                        token_stride: [
                            (3 * inner) as i64,
                            (3 * inner) as i64,
                            (3 * inner) as i64,
                            inner as i64,
                        ],
                        head_stride: [HEAD_DIM as i64; 4],
                        ..AttentionLayout::default()
                    };
                    let qkv = workspace.qkv.pointer().cast::<u16>();
                    match *pass {
                        SparsePass::Sol {
                            workspace: sparse_workspace,
                            tau,
                            sinks,
                        } => sparse_pointers(
                            qkv.cast(),
                            qkv.add(inner).cast(),
                            qkv.add(2 * inner).cast(),
                            workspace.attention.pointer(),
                            tokens,
                            config.heads,
                            &layout,
                            scale,
                            tau,
                            sinks,
                            sparse_workspace,
                            inputs,
                        )?,
                        SparsePass::Vsa {
                            workspace: vsa_workspace,
                            kept,
                        } => vsa_pointers(
                            qkv.cast(),
                            qkv.add(inner).cast(),
                            qkv.add(2 * inner).cast(),
                            vsa_gate.unwrap_or(ptr::null()),
                            inner,
                            workspace.attention.pointer(),
                            &layout,
                            scale,
                            kept,
                            vsa_workspace,
                            inputs,
                        )?,
                    }
                }
            }
        }
        self.linear(
            &format!("{prefix}.attn.out_proj"),
            workspace.attention.pointer(),
            workspace.delta.pointer(),
            tokens,
            workspace,
        )?;

        let fc1 = format!("{prefix}.mlp.fc1");
        if let Some(modulation) = modulation.filter(|_| self.tensors.is_int8(&fc1)) {
            self.add_norm_quantize(
                &format!("{prefix}.norm2.weight"),
                &fc1,
                workspace,
                tokens,
                modulation,
                Some((modulation.0, 2)),
                3,
                4,
            )?;
            self.tensors.linear_quantized(
                &fc1,
                workspace.normalized.pointer(),
                workspace.activated.pointer(),
                tokens,
                &workspace.quantized,
                &workspace.scales,
                adapter(&fc1),
                true,
            )?;
        } else if self.tensors.is_int8(&fc1) {
            return Err(Error::Model(format!(
                "{fc1}: an INT8 MLP needs the block modulation"
            )));
        } else {
            self.add_residual(workspace, tokens, gate(2))?;
            self.normalize(
                &format!("{prefix}.norm2.weight"),
                workspace.residual.pointer(),
                workspace.normalized.pointer(),
                false,
                tokens,
                modulate(3, 4),
            )?;
            self.linear(
                &fc1,
                workspace.normalized.pointer(),
                workspace.expanded.pointer(),
                tokens,
                workspace,
            )?;
            // SAFETY: expanded holds `tokens × 2 × ffn` values and activated `tokens × ffn`.
            check(unsafe {
                mmh3_swiglu(
                    workspace.expanded.pointer(),
                    workspace.activated.pointer(),
                    tokens as c_int,
                    config.ffn as c_int,
                    ptr::null_mut(),
                )
            })?;
        }
        self.linear(
            &format!("{prefix}.mlp.fc2"),
            workspace.activated.pointer(),
            workspace.delta.pointer(),
            tokens,
            workspace,
        )?;
        match modulation {
            Some(_) => Ok(()),
            None => self.add_residual(workspace, tokens, None),
        }
    }

    /// Writes the VSA gates of block `prefix` into the workspace and returns them, or None when
    /// the block has no gates. With `fused`, the INT8 input of the block's qkv projection is still
    /// in the quantization buffers, and otherwise its BF16 input is in `normalized`.
    fn vsa_gate(
        &self,
        prefix: &str,
        workspace: &Workspace,
        tokens: usize,
        fused: bool,
    ) -> Result<Option<*const c_void>, Error> {
        let name = format!("{prefix}.attn.to_gate_compress");
        let Some(gate) = workspace
            .gate
            .as_ref()
            .filter(|_| self.tensors.optional(&format!("{name}.weight")).is_some())
        else {
            return Ok(None);
        };
        if fused && self.tensors.is_int8(&name) {
            self.tensors.linear_quantized(
                &name,
                workspace.normalized.pointer(),
                gate.pointer(),
                tokens,
                &workspace.quantized,
                &workspace.scales,
                None,
                false,
            )?;
        } else {
            self.linear(
                &name,
                workspace.normalized.pointer(),
                gate.pointer(),
                tokens,
                workspace,
            )?;
        }
        Ok(Some(gate.pointer().cast_const()))
    }

    /// Adds the MLP output a modulated block left in `delta`, gated by the block's modulation
    /// table.
    fn add_pending(
        &self,
        workspace: &Workspace,
        tokens: usize,
        table: &DeviceBuffer,
        rows: &DeviceBuffer,
    ) -> Result<(), Error> {
        self.add_residual(workspace, tokens, Some((table, rows, MLP_GATE_CHUNK)))
    }

    /// Writes the refined text states into the first rows of the residual stream.
    fn refine_text(&self, context: &Tensor, workspace: &Workspace) -> Result<(), Error> {
        let tokens = context.shape[0];
        let context_bytes: Vec<u8> = context
            .data
            .iter()
            .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
            .collect();
        let context_buffer = DeviceBuffer::from_bytes(&context_bytes)?;
        self.linear(
            "condition_proj",
            context_buffer.pointer(),
            workspace.normalized.pointer(),
            tokens,
            workspace,
        )?;
        // SAFETY: both buffers hold at least `tokens × hidden` values.
        check(unsafe {
            mmh3_bf16_to_f32(
                workspace.normalized.pointer(),
                workspace.residual.pointer(),
                tokens * self.config.hidden,
                ptr::null_mut(),
            )
        })?;
        for layer in 0..self.config.refiner_layers {
            self.transformer_block(
                &format!("token_refiner.blocks.{layer}"),
                workspace,
                tokens,
                None,
                None,
                None,
                None,
            )?;
        }
        self.normalize(
            "token_refiner.final_norm.weight",
            workspace.residual.pointer(),
            workspace.residual.pointer(),
            true,
            tokens,
            None,
        )
    }

    /// Runs one DiT call and returns the velocity, capturing the residual stream after the listed
    /// blocks. `sparse` switches the blocks to Sol-Attn or VSA when the sequence is long enough.
    /// VSA runs the blocks on the sequence with the video in cube order, and uses the blocks' gates
    /// for its coarse branch when they have them.
    pub fn forward(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&SparseAttention>,
    ) -> Result<DitOutputs, Error> {
        let config = &self.config;
        let hidden = config.hidden;
        let video_shape = &inputs.video.shape;
        let text_tokens = inputs.context.shape[0];
        let layout = PackedLayout::text_to_video(
            text_tokens,
            video_shape[1],
            video_shape[2],
            video_shape[3],
            inputs.audio.shape[2],
        );
        let tokens = layout.len();
        let timesteps = StepTimesteps::new(inputs.sigma, inputs.shift_video, inputs.shift_audio);
        let steps = timesteps.values.len();
        let adapter_rank = self
            .adapters
            .values()
            .map(|adapter| adapter.rank)
            .max()
            .unwrap_or(0);
        // The refiner's MLPs still write the gate and up projections for the text tokens.
        let fused = (0..config.layers)
            .all(|layer| self.tensors.is_int8(&format!("blocks.{layer}.mlp.fc1")));
        let method = sparse
            .filter(|settings| tokens >= settings.min_tokens)
            .map(|settings| settings.method);
        let sparse_tau = match method {
            Some(SparseMethod::Sol { tau }) => Some(tau),
            _ => None,
        };
        let vsa_sparsity = match method {
            Some(SparseMethod::Vsa { sparsity }) => Some(sparsity),
            _ => None,
        };
        let plan = vsa_sparsity.map(|_| VsaPlan::for_layout(&layout));
        let quantized_dense =
            self.attention_precision == AttentionPrecision::Int8Fp8 && method.is_none();

        let mut buffers = self.buffers.borrow_mut();
        let ForwardBuffers {
            workspace: cached_workspace,
            sparse: cached_sparse,
            vsa: cached_vsa,
            text: cached_text,
        } = &mut *buffers;
        let shape = WorkspaceShape {
            tokens,
            expanded_rows: if fused { text_tokens } else { tokens },
            adapter_rank,
            gated: plan.is_some() && self.has_vsa_gates(),
        };
        let workspace = match cached_workspace.take() {
            Some(mut workspace) if workspace.shape == shape => {
                workspace.residual.fill_f32(0.0)?;
                workspace
            }
            _ => Workspace::new(config, shape)?,
        };
        let workspace = cached_workspace.insert(workspace);
        match (quantized_dense, workspace.attention_quantized.is_some()) {
            (true, false) => {
                workspace.attention_quantized = Some(QuantizedWorkspace::new(tokens, config.heads)?)
            }
            (false, true) => workspace.attention_quantized = None,
            _ => {}
        }
        let workspace = &*workspace;
        match sparse_tau {
            Some(_)
                if cached_sparse.as_ref().is_some_and(|(precision, sparse)| {
                    *precision == self.attention_precision && sparse.tokens() == tokens
                }) => {}
            Some(_) => {
                *cached_sparse = None;
                *cached_sparse = Some((
                    self.attention_precision,
                    SparseWorkspace::with_precision(
                        tokens,
                        config.heads,
                        self.attention_precision,
                    )?,
                ));
            }
            None => *cached_sparse = None,
        }
        match &plan {
            Some(plan)
                if cached_vsa.as_ref().is_some_and(|(precision, cached, _)| {
                    *precision == self.attention_precision && cached == plan
                }) => {}
            Some(plan) => {
                *cached_vsa = None;
                *cached_vsa = Some((
                    self.attention_precision,
                    plan.clone(),
                    VsaWorkspace::with_precision(
                        plan,
                        tokens,
                        config.heads,
                        self.attention_precision,
                    )?,
                ));
            }
            None => *cached_vsa = None,
        }

        let block_rows = i32_buffer(&layout.modulation_rows(&timesteps))?;
        let final_rows: Vec<usize> = layout
            .segments
            .iter()
            .flat_map(|segment| {
                std::iter::repeat_n(timesteps.index_of(segment.kind), segment.len())
            })
            .collect();
        let final_rows = i32_buffer(&final_rows)?;
        let audio = layout.segment(SegmentKind::Audio);
        let video = layout.segment(SegmentKind::Video);
        let mut angles = layout.rope_angles(&self.inverse_frequencies);
        let mut video_rows = patchify_video(&inputs.video);
        if let Some(plan) = &plan {
            let width = 3 * self.inverse_frequencies.len();
            let reordered = plan.reorder_video(&angles[video.start * width..], width);
            angles[video.start * width..].copy_from_slice(&reordered);
            video_rows = plan.reorder_video(&video_rows, config.video_patch_features());
        }
        let angles = DeviceBuffer::from_f32(&angles)?;
        let time_embedding = DeviceBuffer::from_f32(&timesteps.time_embedding(&self.adaln_table))?;

        let text_bytes = text_tokens * hidden * 4;
        match cached_text {
            Some(text) if text.context == inputs.context => {
                // SAFETY: both buffers hold the text rows.
                unsafe {
                    copy_device(
                        workspace.residual.pointer(),
                        text.states.pointer(),
                        text_bytes,
                    )?
                };
            }
            _ => {
                self.refine_text(&inputs.context, workspace)?;
                let states = DeviceBuffer::new(text_bytes)?;
                // SAFETY: both buffers hold the text rows.
                unsafe { copy_device(states.pointer(), workspace.residual.pointer(), text_bytes)? };
                *cached_text = Some(TextStates {
                    context: inputs.context.clone(),
                    states,
                });
            }
        }

        let audio_rows = DeviceBuffer::from_f32(&pack_audio(&inputs.audio))?;
        let video_rows = DeviceBuffer::from_f32(&video_rows)?;
        self.linear(
            "audio_patch_proj",
            audio_rows.pointer(),
            workspace.residual.pointer_at(audio.start * hidden * 4),
            audio.len(),
            workspace,
        )?;
        self.linear(
            "video_patch_proj",
            video_rows.pointer(),
            workspace.residual.pointer_at(video.start * hidden * 4),
            video.len(),
            workspace,
        )?;

        let sparse_pass = match (sparse_tau, vsa_sparsity) {
            (Some(tau), _) => cached_sparse
                .as_ref()
                .map(|(_, workspace)| SparsePass::Sol {
                    workspace,
                    tau,
                    sinks: SparseSinks::for_layout(&layout),
                }),
            (None, Some(sparsity)) => {
                cached_vsa
                    .as_ref()
                    .map(|(_, plan, workspace)| SparsePass::Vsa {
                        workspace,
                        kept: plan.kept_video_tiles(sparsity),
                    })
            }
            (None, None) => None,
        };
        let mut routed = 0.0;
        // Each block leaves its gated MLP output to the next, so the tables of two blocks are alive
        // at a time.
        let modulations = [
            DeviceBuffer::new(steps * MODALITY_COUNT * BLOCK_CHUNKS * hidden * 4)?,
            DeviceBuffer::new(steps * MODALITY_COUNT * BLOCK_CHUNKS * hidden * 4)?,
        ];
        let mut pending = None;
        let mut blocks = Vec::new();
        for layer in 0..config.layers {
            let prefix = format!("blocks.{layer}");
            let modulation = &modulations[layer % 2];
            self.modulation(
                &format!("{prefix}.adaln_proj.linear"),
                &time_embedding,
                steps,
                modulation,
            )?;
            self.transformer_block(
                &prefix,
                workspace,
                tokens,
                Some(&angles),
                Some((modulation, &block_rows)),
                pending,
                sparse_pass.as_ref(),
            )?;
            pending = Some(modulation);
            if let Some(SparsePass::Sol { workspace, .. }) = &sparse_pass {
                routed += workspace.routed_fraction()?;
            }
            if capture.contains(&layer) {
                self.add_pending(workspace, tokens, modulation, &block_rows)?;
                pending = None;
                let mut residual = workspace.residual.to_f32()?;
                if let Some(plan) = &plan {
                    let restored = plan.restore_video(&residual[video.start * hidden..], hidden);
                    residual[video.start * hidden..].copy_from_slice(&restored);
                }
                blocks.push((layer, residual));
            }
        }
        if let Some(modulation) = pending {
            self.add_pending(workspace, tokens, modulation, &block_rows)?;
        }

        let final_modulation = DeviceBuffer::new(steps * FINAL_CHUNKS * hidden * 4)?;
        self.modulation(
            "final_layer.adaln_proj.linear",
            &time_embedding,
            steps,
            &final_modulation,
        )?;
        self.normalize(
            "final_layer.norm.weight",
            workspace.residual.pointer(),
            workspace.projected.pointer(),
            true,
            tokens,
            Some((&final_modulation, &final_rows, FINAL_CHUNKS, 0, 1)),
        )?;
        let project = |segment: mmh3_core::dit::layout::Segment,
                       head: &str,
                       width: usize|
         -> Result<Vec<f32>, Error> {
            let output = DeviceBuffer::new(segment.len() * width * 4)?;
            self.linear(
                head,
                workspace.projected.pointer_at(segment.start * hidden * 4),
                output.pointer(),
                segment.len(),
                workspace,
            )?;
            Ok(output.to_f32()?)
        };
        let mut video_velocity = project(
            video,
            "final_layer.video_out",
            config.video_patch_features(),
        )?;
        if let Some(plan) = &plan {
            video_velocity = plan.restore_video(&video_velocity, config.video_patch_features());
        }
        let audio_velocity = project(audio, "final_layer.audio_out", config.audio_channels)?;
        Ok(DitOutputs {
            blocks,
            video: unpatchify_video(&video_velocity, video_shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            audio: unpack_audio(&audio_velocity, &inputs.audio.shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            routed_fraction: matches!(sparse_pass, Some(SparsePass::Sol { .. }))
                .then(|| routed / config.layers as f64),
        })
    }
}
