//! One DiT call on the GPU for text-to-video with audio.
//!
//! The residual stream stays in FP32. Linear layers take BF16 inputs and run through the INT8
//! ConvRot GEMM for quantized weights and through cuBLASLt for BF16 and FP32 weights.

use crate::attention::{
    AttentionInputs, AttentionLayout, AttentionPrecision, HEAD_DIM, PreparedAttention,
    QuantizedWorkspace, RouteOverlap, SparseWorkspace, VsaWorkspace, dense_bf16_pointers,
    dense_quantized_pointers, prepare_inputs_pointers, sparse_pointers, vsa_pointers,
};
use crate::gemm::AdapterPointers;
use crate::gemm::interleave_swiglu_rows;
use crate::loader::Uploader;
use crate::model::{
    ADAPTER_RANK_MULTIPLE, DeviceTensors, Down, LowRank, check_quantization, host_tensor,
    i32_buffer,
};
use crate::nvfp4::{
    self, AdapterSource, Columns, Nvfp4Activations, Nvfp4Input, Nvfp4Scale, Nvfp4Weight,
};
use crate::shard::{self, Region, ShardContext};
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
use std::ops::Range;
use std::ptr;
use std::time::Instant;

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
    fn mmh3_add_norm_quantize_nvfp4(
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
        values: *mut c_void,
        scales: *mut c_void,
        tensor_scale: *mut c_void,
        reference: *mut c_void,
        margin: f32,
        observed: *mut c_void,
        exact: c_int,
        tokens: c_int,
        hidden: c_int,
        columns: c_int,
        epsilon: f32,
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
    /// Rows the adapters' down projections cover, which a shared-out block runs over the whole
    /// sequence rather than over this rank's rows.
    adapter_rows: usize,
    /// Whether the blocks have VSA gates.
    gated: bool,
    /// Values per row of the NVFP4 layers' activations, or 0 without NVFP4 layers.
    nvfp4_columns: usize,
    /// The largest rank of the NVFP4 layers' adapters.
    nvfp4_adapter_rank: usize,
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
    /// Inputs of the NVFP4 layers.
    nvfp4: Option<Nvfp4Activations>,
}

impl Workspace {
    fn new(config: &DitConfig, shape: WorkspaceShape) -> Result<Self, CudaError> {
        let (hidden, inner, ffn) = (config.hidden, config.inner(), config.ffn);
        let WorkspaceShape {
            tokens,
            expanded_rows,
            adapter_rank,
            adapter_rows,
            gated,
            nvfp4_columns,
            nvfp4_adapter_rank,
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
            adapter: DeviceBuffer::new(adapter_rows.max(tokens) * adapter_rank.max(1) * 2)?,
            gate: if gated {
                Some(DeviceBuffer::new(tokens * inner * 2)?)
            } else {
                None
            },
            nvfp4: if nvfp4_columns > 0 {
                Some(Nvfp4Activations::new(
                    tokens,
                    nvfp4_columns,
                    nvfp4_adapter_rank,
                )?)
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

/// How a LoRA reaches the INT8 ConvRot layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoraMode {
    /// An adapter next to the INT8 weights with its own work in every call. Its down projection
    /// runs on the quantized input of the layer, and the rest in BF16.
    #[default]
    Adapter,
    /// Added into the weights, which are quantized again, like ComfyUI's merge into INT8
    /// checkpoints. No work per call, but updates smaller than half an INT8 step round away.
    Merge,
}

/// One rank's rows of the velocity, before the video goes back into its own order. Only a shared-out
/// step produces these, and `assemble_velocity` turns them back into a whole one.
#[derive(Clone, Debug)]
pub struct VelocityPart {
    pub rows: Range<usize>,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
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
    /// This rank's rows, when the step was shared out. The velocity above is then empty.
    pub part: Option<VelocityPart>,
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
    /// NVFP4 versions of block linear layers by name, with their adapters, which run the video
    /// rows in place of the INT8 weights.
    nvfp4: HashMap<String, Nvfp4Weight>,
    /// Tensor scale histories of the NVFP4 layers' inputs, by layer. A VSA gate shares the input of
    /// its block's qkv projection.
    nvfp4_scales: HashMap<String, Nvfp4Scale>,
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
            nvfp4: HashMap::new(),
            nvfp4_scales: HashMap::new(),
            adaln_table: host_tensor(file, &format!("{prefix}adaln_t_table"))?,
            inverse_frequencies: host_tensor(file, &format!("{prefix}rope.inv_freq"))?.data,
            buffers: RefCell::default(),
        })
    }

    /// Selects quantized attention for the main DiT blocks. The text refiner stays in BF16.
    pub fn attention_precision(&self) -> AttentionPrecision {
        self.attention_precision
    }

    pub fn set_attention_precision(&mut self, precision: AttentionPrecision) {
        self.attention_precision = precision;
    }

    /// How much neighbouring query tiles shared in the last VSA call, or `None` when no call has
    /// used VSA yet.
    pub fn vsa_route_overlap(&self) -> Result<Option<RouteOverlap>, Error> {
        let buffers = self.buffers.borrow();
        let Some((_, _, workspace)) = buffers.vsa.as_ref() else {
            return Ok(None);
        };
        Ok(Some(workspace.route_overlap()?))
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
    /// `strength · alpha / rank · B · A` on top of its weight, in the way `mode` says for INT8
    /// ConvRot layers. Other layers get an adapter. Adapter ranks that are not a multiple of 64, as
    /// in LoRAs resized to a different rank per layer, are padded with zeros so that every adapter
    /// runs inside the INT8 GEMM. Other tensors of the file replace the checkpoint's tensors of the
    /// same name, or add new ones, as they are, whatever the strength. Returns the number of layers
    /// merged or with an adapter and of tensors replaced or added.
    pub fn add_lora(
        &mut self,
        file: &SafeTensors,
        strength: f32,
        mode: LoraMode,
    ) -> Result<usize, Error> {
        const PREFIX: &str = "diffusion_model.";
        let mut uploader = Uploader::new(file);
        let mut adapters = Vec::new();
        let mut merged = 0;
        for info in file.tensors() {
            let Some(layer) = info
                .name
                .strip_prefix(PREFIX)
                .and_then(|name| name.strip_suffix(".lora_A.weight"))
            else {
                continue;
            };
            let weight = self.tensors.get(&format!("{layer}.weight"))?;
            let (weight_dtype, weight_shape) = (weight.dtype, weight.shape.clone());
            let up = file
                .get(&format!("{PREFIX}{layer}.lora_B.weight"))
                .ok_or_else(|| Error::Model(format!("{layer} has no lora_B")))?;
            let alpha = host_tensor(file, &format!("{PREFIX}{layer}.alpha"))?.data[0];
            let (rank, inputs, outputs) = (info.shape[0], info.shape[1], up.shape[0]);
            if info.dtype != DType::BF16
                || up.dtype != DType::BF16
                || up.shape[1] != rank
                || inputs != weight_shape[1]
                || outputs != weight_shape[0]
            {
                return Err(Error::Model(format!(
                    "the LoRA of {layer} does not fit the layer"
                )));
            }
            if weight_dtype != DType::I8 && weight_dtype != DType::BF16 {
                return Err(Error::Model(format!(
                    "LoRA on {layer} needs BF16 activations"
                )));
            }
            if self.nvfp4.contains_key(layer) {
                return Err(Error::Model(format!(
                    "{layer}: add LoRAs before switching to NVFP4"
                )));
            }
            let scale = strength * alpha / rank as f32;
            if mode == LoraMode::Merge && weight_dtype == DType::I8 {
                let mut down: Vec<f32> = host_tensor(file, &info.name)?.data;
                down.iter_mut().for_each(|value| *value *= scale);
                let mut down = DeviceBuffer::from_f32(&down)?;
                let mut up = DeviceBuffer::from_f32(&host_tensor(file, &up.name)?.data)?;
                if layer.ends_with(".mlp.fc1") {
                    up = interleave_swiglu_rows(&up, outputs, rank * 4)?;
                }
                self.tensors.merge_low_rank(layer, &up, &mut down, rank)?;
                merged += 1;
                continue;
            }
            if self.adapters.contains_key(layer) {
                return Err(Error::Model(format!("{layer} already has a LoRA")));
            }
            let padded_rank = rank.next_multiple_of(ADAPTER_RANK_MULTIPLE);
            let down = if weight_dtype == DType::I8 {
                LowRank::quantize_down(file.data(info), rank, inputs)?
            } else if padded_rank == rank {
                Down::Bf16(uploader.allocate(file, info)?)
            } else {
                let mut down = file.data(info).to_vec();
                down.resize(padded_rank * inputs * 2, 0);
                Down::Bf16(DeviceBuffer::from_bytes(&down)?)
            };
            let up = if padded_rank == rank {
                uploader.allocate(file, up)?
            } else {
                let mut padded_up = vec![0; outputs * padded_rank * 2];
                for (source, destination) in file
                    .data(up)
                    .chunks_exact(rank * 2)
                    .zip(padded_up.chunks_exact_mut(padded_rank * 2))
                {
                    destination[..rank * 2].copy_from_slice(source);
                }
                DeviceBuffer::from_bytes(&padded_up)?
            };
            let adapter = LowRank {
                down,
                up,
                rank: padded_rank,
                inputs,
                outputs,
                scale,
            };
            adapters.push((layer.to_owned(), adapter));
        }
        let mut replaced = Vec::new();
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                continue;
            };
            if [".lora_A.weight", ".lora_B.weight", ".alpha"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                continue;
            }
            if name == "adaln_t_table" {
                self.adaln_table = host_tensor(file, &info.name)?;
            } else {
                if let Some(existing) = self.tensors.optional(name)
                    && (existing.dtype != info.dtype || existing.shape != info.shape)
                {
                    return Err(Error::Model(format!(
                        "{name} in the LoRA does not match the checkpoint's"
                    )));
                }
                self.tensors.insert(name, file, info, &mut uploader)?;
            }
            replaced.push(name.to_owned());
        }
        uploader.run()?;
        for name in &replaced {
            let fc1 = name
                .strip_suffix(".weight")
                .or_else(|| name.strip_suffix(".weight_scale"))
                .filter(|layer| layer.ends_with(".mlp.fc1"));
            if let Some(layer) = fc1
                && self.tensors.is_int8(layer)
            {
                self.tensors.interleave_swiglu(name)?;
            }
        }
        for (layer, adapter) in &mut adapters {
            if layer.ends_with(".mlp.fc1") && self.tensors.is_int8(layer) {
                adapter.up =
                    interleave_swiglu_rows(&adapter.up, adapter.outputs, adapter.rank * 2)?;
            }
        }
        let added = merged + adapters.len() + replaced.len();
        self.adapters.extend(adapters);
        // The adapters may change the refiner.
        self.buffers.get_mut().text = None;
        Ok(added)
    }

    /// Runs the linear layers of the main blocks in NVFP4, requantized from their INT8 ConvRot
    /// weights, and returns how many layers changed. A block's first call quantizes the inputs
    /// from BF16 rows to find their tensor scales, and later calls quantize them in the passes that
    /// produce them. The adapters of these layers go into their NVFP4 weights too, so LoRAs come
    /// first. The INT8 weights and adapters stay for the text and audio rows.
    pub fn use_nvfp4(&mut self) -> Result<usize, Error> {
        for layer in 0..self.config.layers {
            let qkv = format!("blocks.{layer}.attn.qkv_proj");
            for part in [
                "attn.qkv_proj",
                "attn.to_gate_compress",
                "attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
            ] {
                let name = format!("blocks.{layer}.{part}");
                if !self.tensors.is_int8(&name)
                    || self.nvfp4.contains_key(&name)
                    || (part == "attn.to_gate_compress" && !self.nvfp4.contains_key(&qkv))
                {
                    continue;
                }
                let columns = match self.adapters.get(&name) {
                    Some(_) if part == "attn.to_gate_compress" => {
                        return Err(Error::Model(format!(
                            "{name}: NVFP4 VSA gates do not take LoRAs"
                        )));
                    }
                    Some(adapter) => {
                        let Down::Int8 {
                            weights, scales, ..
                        } = &adapter.down
                        else {
                            return Err(Error::Model(format!(
                                "{name}: the adapter has no quantized down projection"
                            )));
                        };
                        Columns::Adapter(AdapterSource {
                            down: weights,
                            down_scales: scales,
                            up: &adapter.up,
                            rank: adapter.rank,
                            scale: adapter.scale,
                        })
                    }
                    // The gate takes the activations of the block's qkv projection.
                    None if part == "attn.to_gate_compress" => {
                        Columns::Zeros(self.nvfp4[&qkv].columns)
                    }
                    None => Columns::Inputs,
                };
                let weight = self.tensors.get(&format!("{name}.weight"))?;
                let scales = self.tensors.get(&format!("{name}.weight_scale"))?;
                let converted = Nvfp4Weight::from_int8_with(
                    &weight.buffer,
                    &scales.buffer,
                    weight.shape[0],
                    weight.shape[1],
                    part == "mlp.fc1",
                    columns,
                )?;
                self.nvfp4.insert(name.clone(), converted);
                if part != "attn.to_gate_compress" {
                    self.nvfp4_scales.insert(name, Nvfp4Scale::new()?);
                }
            }
        }
        Ok(self.nvfp4.len())
    }

    fn nvfp4_activations<'a>(&self, workspace: &'a Workspace) -> &'a Nvfp4Activations {
        workspace
            .nvfp4
            .as_ref()
            .expect("the workspace has NVFP4 buffers when the DiT has NVFP4 layers")
    }

    fn nvfp4_scale(&self, layer: &str) -> Result<&Nvfp4Scale, Error> {
        self.nvfp4_scales
            .get(layer)
            .ok_or_else(|| Error::Model(format!("{layer} has no NVFP4 input scale")))
    }

    /// `add_norm_quantize` for the NVFP4 layer `layer` on `tokens` rows from `first_row`. The first
    /// call finds the exact tensor scale in a pass of its own, and later calls take the scale of
    /// the call before.
    #[allow(clippy::too_many_arguments)]
    fn add_norm_quantize_nvfp4(
        &self,
        weight: &str,
        layer: &str,
        workspace: &Workspace,
        first_row: usize,
        tokens: usize,
        (table, rows): (&DeviceBuffer, &DeviceBuffer),
        gate: Option<(&DeviceBuffer, usize)>,
        shift: usize,
        scale: usize,
    ) -> Result<(), Error> {
        let hidden = self.config.hidden;
        let history = self.nvfp4_scale(layer)?;
        let activations = self.nvfp4_activations(workspace);
        let layer_weight = &self.nvfp4[layer];
        activations.check_fits(tokens, layer_weight.columns);
        let slots = history.next_call();
        // SAFETY: the residual and delta buffers hold the rows up to `first_row + tokens`, the
        // activation buffers fit, checked above, and the modulation rows index the table.
        check(unsafe {
            mmh3_add_norm_quantize_nvfp4(
                workspace.residual.pointer_at(first_row * hidden * 4),
                gate.map_or(ptr::null(), |_| {
                    workspace
                        .delta
                        .pointer_at(first_row * hidden * 2)
                        .cast_const()
                }),
                gate.map_or(ptr::null(), |(gate_table, _)| {
                    gate_table.pointer().cast_const()
                }),
                table.pointer(),
                rows.pointer_at(first_row * 4),
                BLOCK_CHUNKS as c_int,
                gate.map_or(0, |(_, chunk)| chunk) as c_int,
                shift as c_int,
                scale as c_int,
                self.pointer(weight)?,
                activations.values.pointer(),
                activations.scales.pointer(),
                activations.scratch.pointer(),
                slots.reference,
                slots.margin,
                slots.observed,
                slots.exact as c_int,
                tokens as c_int,
                hidden as c_int,
                layer_weight.columns as c_int,
                self.config.norm_eps,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Applies layer `name` to `rows` rows. NVFP4 layers run the first `head` rows through their
    /// INT8 weights.
    fn linear(
        &self,
        name: &str,
        input: *const c_void,
        output: *mut c_void,
        rows: usize,
        head: usize,
        workspace: &Workspace,
    ) -> Result<(), Error> {
        let adapter = self
            .adapters
            .get(name)
            .map(|adapter| (adapter, &workspace.adapter));
        let Some(weight) = self.nvfp4.get(name) else {
            return self.tensors.linear(
                name,
                input,
                output,
                rows,
                &workspace.quantized,
                &workspace.scales,
                adapter,
            );
        };
        if head > 0 {
            self.tensors.linear(
                name,
                input,
                output,
                head,
                &workspace.quantized,
                &workspace.scales,
                adapter,
            )?;
        }
        let activations = self.nvfp4_activations(workspace);
        // SAFETY: the caller passes `rows` rows of input and output for the layer.
        unsafe {
            nvfp4::quantize_pointers(
                Nvfp4Input::Rows(input.byte_add(head * weight.features * 2)),
                rows - head,
                weight,
                self.nvfp4_scale(name)?,
                activations,
            )?;
            nvfp4::gemm_pointers(
                weight,
                activations,
                output.byte_add(head * weight.outputs * 2),
                rows - head,
            )?;
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
    /// `normalized` only when an adapter of the layer needs them.
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
        let normalized = if self
            .adapters
            .get(layer)
            .is_some_and(|adapter| matches!(adapter.down, Down::Bf16(_)))
        {
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

    /// The attention projections over one range of the sequence's rows. A shared-out block runs
    /// them in two goes, its own rows and then the peers', and the arithmetic is the same either
    /// way: a row of the result depends on that row of the input alone.
    #[allow(clippy::too_many_arguments)]
    fn project_inputs(
        &self,
        prefix: &str,
        workspace: &Workspace,
        heads: &Range<usize>,
        rows: Range<usize>,
        normalized: *const c_void,
        scales: *const c_void,
        inputs: *mut c_void,
        gate: *mut c_void,
    ) -> Result<(), Error> {
        let (hidden, whole) = (self.config.hidden, self.config.inner());
        let inner = heads.len() * HEAD_DIM;
        let count = rows.len();
        if count == 0 {
            return Ok(());
        }
        // SAFETY: the regions hold every token's rows, and this range lies inside them.
        let (normalized, scales) = unsafe {
            (
                normalized.byte_add(rows.start * hidden),
                scales.byte_add(rows.start * 4),
            )
        };
        // An adapter's down projection depends on the block's input alone, so the three tensors of
        // a range share one.
        let down = |name: &str| -> Result<Option<AdapterPointers>, Error> {
            let adapter = self
                .adapters
                .get(name)
                .map(|adapter| (adapter, &workspace.adapter));
            // SAFETY: the range holds `count` rows of the input and its scales.
            unsafe {
                self.tensors
                    .adapter_down(adapter, name, count, normalized, scales)
            }
        };
        // The three tensors of one token sit side by side, so each projection writes a column range
        // of a row that is three times as wide as its own result.
        let qkv = format!("{prefix}.attn.qkv_proj");
        let qkv_adapter = down(&qkv)?;
        for tensor in 0..3 {
            let columns =
                (tensor * whole + heads.start * HEAD_DIM)..(tensor * whole + heads.end * HEAD_DIM);
            // SAFETY: the region holds `tokens × 3 × inner` values.
            let into = unsafe { inputs.byte_add((rows.start * 3 * inner + tensor * inner) * 2) };
            self.tensors.linear_quantized_range(
                &qkv,
                into,
                count,
                columns,
                3 * inner,
                normalized,
                scales,
                qkv_adapter,
            )?;
        }
        if !gate.is_null() {
            let name = format!("{prefix}.attn.to_gate_compress");
            let adapter = down(&name)?;
            // SAFETY: the gate holds `tokens × inner` values.
            let into = unsafe { gate.byte_add(rows.start * inner * 2) };
            self.tensors.linear_quantized_range(
                &name,
                into,
                count,
                (heads.start * HEAD_DIM)..(heads.end * HEAD_DIM),
                0,
                normalized,
                scales,
                adapter,
            )?;
        }
        Ok(())
    }

    /// A block when the step is shared out. The exchange is of what the projections consume, not of
    /// what they produce: a rank sends its rows of the quantized block input, an eighth of the
    /// bytes q, k, v and the gate would take, and then runs its own heads' columns over the whole
    /// sequence. The arithmetic is the same and nothing is quantized twice.
    fn sharded_attention(
        &self,
        prefix: &str,
        workspace: &Workspace,
        context: &mut ShardContext<'_>,
        tokens: usize,
        angles: *const c_void,
        sparse: Option<&SparsePass>,
    ) -> Result<(), Error> {
        let config = &self.config;
        let (normalized, scales) = self.publish_inputs(workspace, context, tokens)?;

        let started = Instant::now();
        let shard = context.shard.clone();
        let own = shard.own_heads();
        let inner = own.len() * HEAD_DIM;
        let inputs = context
            .exchange
            .region(Region::Inputs, tokens * 3 * inner * 2)?;
        // Only VSA reads a gate. The region is there on every step of a gated checkpoint, since a
        // schedule may turn VSA on partway through a run, and the dense steps leave it alone.
        let gated = matches!(sparse, Some(SparsePass::Vsa { .. }))
            && self
                .tensors
                .optional(&format!("{prefix}.attn.to_gate_compress.weight"))
                .is_some();
        let gate = if gated {
            context.exchange.region(Region::Gate, tokens * inner * 2)?
        } else {
            ptr::null_mut()
        };
        // This rank's own rows are already here, so they go through the projections while the
        // peers' rows are still crossing the wire: the read below blocks this thread and the
        // device has a third of a second of work queued behind it.
        self.project_inputs(
            prefix,
            workspace,
            &own,
            shard.own_tokens(),
            normalized,
            scales,
            inputs,
            gate,
        )?;
        context.timing.attend += started.elapsed();
        self.send_inputs(context)?;
        self.fetch_inputs(context)?;
        let started = Instant::now();
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            self.project_inputs(
                prefix,
                workspace,
                &own,
                shard.tokens[peer].clone(),
                normalized,
                scales,
                inputs,
                gate,
            )?;
        }

        let query_norm = self.pointer(&format!("{prefix}.attn.q_norm.weight"))?;
        let key_norm = self.pointer(&format!("{prefix}.attn.k_norm.weight"))?;
        let attended = context
            .exchange
            .region(Region::Attended, tokens * inner * 2)?;
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
        // Every workspace here is made for this rank's heads over the whole sequence, which is what
        // the region holds, so the attention runs exactly as it does on one machine.
        let quantized = workspace.attention_quantized.as_ref();
        let prepared = match (sparse, quantized) {
            (Some(SparsePass::Sol { workspace, .. }), _) => {
                Some(PreparedAttention::Sparse(workspace))
            }
            (Some(SparsePass::Vsa { workspace, .. }), _) => Some(PreparedAttention::Vsa(workspace)),
            (None, Some(quantized)) => Some(PreparedAttention::DenseQuantized(quantized)),
            (None, None) => None,
        };
        let ready = if prepared.is_some() {
            AttentionInputs::Prepared
        } else {
            AttentionInputs::Raw
        };
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        // SAFETY: the projections filled `tokens × 3 × own heads × 128` values in the layout the
        // preparation and the attention read, the gate holds one tensor of the same rows, and the
        // angles cover every token because attention sees the whole sequence.
        unsafe {
            let pairs = 3 * config.rope_frequencies;
            match prepared {
                Some(attention) => prepare_inputs_pointers(
                    inputs,
                    query_norm,
                    key_norm,
                    angles,
                    pairs,
                    tokens,
                    own.len(),
                    config.norm_eps,
                    attention,
                )?,
                None => check(mmh3_qk_norm_rope(
                    inputs,
                    query_norm,
                    key_norm,
                    angles,
                    pairs as c_int,
                    tokens as c_int,
                    own.len() as c_int,
                    own.len() as c_int,
                    config.norm_eps,
                    ptr::null_mut(),
                ))?,
            }
            let base = inputs.cast::<u16>();
            match sparse {
                None => match quantized {
                    Some(quantized) => {
                        dense_quantized_pointers(inputs, attended, scale, quantized, ready)?
                    }
                    None => dense_bf16_pointers(inputs, attended, tokens, own.len(), scale)?,
                },
                Some(SparsePass::Sol {
                    workspace: sparse_workspace,
                    tau,
                    sinks,
                }) => sparse_pointers(
                    base.cast(),
                    base.add(inner).cast(),
                    base.add(2 * inner).cast(),
                    attended,
                    tokens,
                    own.len(),
                    &layout,
                    scale,
                    *tau,
                    *sinks,
                    sparse_workspace,
                    ready,
                )?,
                Some(SparsePass::Vsa {
                    workspace: vsa_workspace,
                    kept,
                }) => vsa_pointers(
                    base.cast(),
                    base.add(inner).cast(),
                    base.add(2 * inner).cast(),
                    gate.cast_const(),
                    inner,
                    attended,
                    &layout,
                    scale,
                    *kept,
                    vsa_workspace,
                    ready,
                )?,
            }
        }
        // The attention is on the stream too, and the peers are about to read what it wrote.
        crate::synchronize()?;
        context.timing.attend += started.elapsed();
        self.exchange_outputs(workspace, context, tokens)
    }

    /// Puts this rank's rows of the block input where the peers can read them, and says so. The
    /// rows of the others arrive in `fetch_inputs`, and between the two a rank projects what it
    /// already holds.
    fn publish_inputs(
        &self,
        workspace: &Workspace,
        context: &mut ShardContext<'_>,
        tokens: usize,
    ) -> Result<(*const c_void, *const c_void), Error> {
        let started = Instant::now();
        let hidden = self.config.hidden;
        let rows = context.shard.own_tokens();
        let normalized = context
            .exchange
            .region(Region::Normalized, tokens * hidden)?;
        let scales = context.exchange.region(Region::Scales, tokens * 4)?;
        // SAFETY: the regions hold every token's rows and this rank wrote its own.
        unsafe {
            copy_device(
                normalized.byte_add(rows.start * hidden),
                workspace.quantized.pointer().cast_const(),
                rows.len() * hidden,
            )?;
            copy_device(
                scales.byte_add(rows.start * 4),
                workspace.scales.pointer().cast_const(),
                rows.len() * 4,
            )?;
        }
        // NOTE: the copies are on a stream and the wire is not, so the device has to finish
        // before the rows go out. Only this rank's rows do, which is half of a two-rank region.
        crate::synchronize()?;
        context
            .exchange
            .publish(Region::Normalized, rows.start * hidden, rows.len() * hidden)?;
        context
            .exchange
            .publish(Region::Scales, rows.start * 4, rows.len() * 4)?;
        context.timing.gather += started.elapsed();
        Ok((normalized.cast_const(), scales.cast_const()))
    }

    /// Pushes this rank's rows of the block input to every peer. It runs after the projections of
    /// those same rows are queued, so the device works through them while this thread is on the
    /// wire.
    fn send_inputs(&self, context: &mut ShardContext<'_>) -> Result<(), Error> {
        let hidden = self.config.hidden;
        let started = Instant::now();
        let shard = context.shard.clone();
        let rows = shard.own_tokens();
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            context.exchange.write(
                peer,
                Region::Normalized,
                rows.start * hidden,
                Region::Normalized,
                rows.start * hidden,
                rows.len() * hidden,
            )?;
            context.exchange.write(
                peer,
                Region::Scales,
                rows.start * 4,
                Region::Scales,
                rows.start * 4,
                rows.len() * 4,
            )?;
        }
        context.timing.read += started.elapsed();
        Ok(())
    }

    /// Waits for every rank's rows of the block input to have landed and takes them to the device,
    /// then waits again so that nothing overwrites them before every rank has read them.
    fn fetch_inputs(&self, context: &mut ShardContext<'_>) -> Result<(), Error> {
        let hidden = self.config.hidden;
        let started = Instant::now();
        context.exchange.barrier()?;
        context.timing.barrier += started.elapsed();
        let started = Instant::now();
        let shard = context.shard.clone();
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            let taken = shard.tokens[peer].clone();
            context.exchange.receive(
                Region::Normalized,
                taken.start * hidden,
                taken.len() * hidden,
            )?;
            context
                .exchange
                .receive(Region::Scales, taken.start * 4, taken.len() * 4)?;
        }
        context.timing.read += started.elapsed();
        let started = Instant::now();
        context.exchange.barrier()?;
        context.timing.barrier += started.elapsed();
        Ok(())
    }

    /// Turns "every token, my heads" back into "my tokens, every head", leaving a block's attention
    /// output where the projection that follows expects it.
    fn exchange_outputs(
        &self,
        workspace: &Workspace,
        context: &mut ShardContext<'_>,
        tokens: usize,
    ) -> Result<(), Error> {
        let shard = context.shard.clone();
        let heads = self.config.heads;
        let (rows, own) = (shard.own_tokens(), shard.own_heads());
        let own_inner = own.len() * HEAD_DIM;
        let attended = context
            .exchange
            .region(Region::Attended, tokens * own_inner * 2)?;
        // Each peer takes its own rows of what attention just wrote, so only those leave the
        // device, which on a two-rank run is half of the region.
        let started = Instant::now();
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            let taken = shard.tokens[peer].clone();
            context.exchange.publish(
                Region::Attended,
                taken.start * own_inner * 2,
                taken.len() * own_inner * 2,
            )?;
            context.exchange.write(
                peer,
                Region::Attended,
                taken.start * own_inner * 2,
                Region::Received(shard.rank),
                0,
                taken.len() * own_inner * 2,
            )?;
        }
        context.timing.read += started.elapsed();
        let started = Instant::now();
        context.exchange.barrier()?;
        context.timing.barrier += started.elapsed();
        let started = Instant::now();
        for peer in 0..shard.ranks() {
            if peer == shard.rank {
                continue;
            }
            let inner = shard.heads[peer].len() * HEAD_DIM;
            context
                .exchange
                .receive(Region::Received(peer), 0, rows.len() * inner * 2)?;
        }
        context.timing.read += started.elapsed();
        let started = Instant::now();
        context.exchange.barrier()?;
        context.timing.barrier += started.elapsed();
        let started = Instant::now();
        for peer in 0..shard.ranks() {
            let span = shard.heads[peer].clone();
            let part = if peer == shard.rank {
                // SAFETY: this rank attended its own rows at their offset in the whole sequence.
                unsafe { attended.byte_add(rows.start * own_inner * 2).cast_const() }
            } else {
                context
                    .exchange
                    .region(
                        Region::Received(peer),
                        rows.len() * span.len() * HEAD_DIM * 2,
                    )?
                    .cast_const()
            };
            // SAFETY: the part holds this rank's rows of the peer's heads, and the output holds
            // this rank's rows of every head.
            unsafe {
                shard::unpack(
                    part,
                    workspace.attention.pointer(),
                    rows.len(),
                    heads,
                    HEAD_DIM,
                    span,
                )?
            };
        }
        context.timing.gather += started.elapsed();
        Ok(())
    }

    /// Attention and MLP halves of a block or refiner block. `modulation` carries the block
    /// modulation and `sparse` replaces dense attention with Sol-Attn. NVFP4 layers run the first
    /// `head` rows through their INT8 weights.
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
        head: usize,
        angles: Option<&DeviceBuffer>,
        modulation: Option<(&DeviceBuffer, &DeviceBuffer)>,
        pending: Option<&DeviceBuffer>,
        sparse: Option<&SparsePass>,
        shard: Option<(&mut ShardContext<'_>, usize)>,
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
        // Modulated blocks normalize and quantize the inputs of their INT8 and NVFP4 layers in one
        // pass. NVFP4 layers keep their INT8 weights.
        let nvfp4_qkv = self.nvfp4.get(&qkv).filter(|_| modulation.is_some());
        if let Some(modulation) = modulation.filter(|_| self.tensors.is_int8(&qkv)) {
            let pending_gate = pending.map(|table| (table, MLP_GATE_CHUNK));
            let int8_rows = if nvfp4_qkv.is_some() { head } else { tokens };
            if int8_rows > 0 {
                self.add_norm_quantize(
                    &format!("{prefix}.norm1.weight"),
                    &qkv,
                    workspace,
                    int8_rows,
                    modulation,
                    pending_gate,
                    0,
                    1,
                )?;
                // A shared-out block exchanges what this just quantized and projects afterwards,
                // over the whole sequence and its own heads.
                if shard.is_none() {
                    self.tensors.linear_quantized(
                        &qkv,
                        workspace.qkv.pointer(),
                        int8_rows,
                        &workspace.quantized,
                        &workspace.scales,
                        adapter(&qkv),
                        false,
                    )?;
                }
            }
            if let Some(weight) = nvfp4_qkv {
                self.add_norm_quantize_nvfp4(
                    &format!("{prefix}.norm1.weight"),
                    &qkv,
                    workspace,
                    head,
                    tokens - head,
                    modulation,
                    pending_gate,
                    0,
                    1,
                )?;
                // SAFETY: qkv holds `tokens × 3 × inner` values.
                unsafe {
                    nvfp4::gemm_pointers(
                        weight,
                        self.nvfp4_activations(workspace),
                        workspace.qkv.pointer_at(head * weight.outputs * 2),
                        tokens - head,
                    )?
                };
            }
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
                head,
                workspace,
            )?;
        }
        let vsa_gate = match sparse {
            // A shared-out block's gate comes out of the same exchanged input as its projections.
            Some(SparsePass::Vsa { .. }) if shard.is_none() => {
                let fused = modulation.is_some() && self.tensors.is_int8(&qkv);
                self.vsa_gate(prefix, workspace, tokens, head, fused)?
            }
            _ => None,
        };
        let query_norm = self.pointer(&format!("{prefix}.attn.q_norm.weight"))?;
        let key_norm = self.pointer(&format!("{prefix}.attn.k_norm.weight"))?;
        let angles = angles.map_or(ptr::null(), |angles| angles.pointer().cast_const());
        if let Some((context, whole)) = shard {
            self.sharded_attention(prefix, workspace, context, whole, angles, sparse)?;
        } else {
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
                (Some(SparsePass::Vsa { workspace, .. }), _) => {
                    Some(PreparedAttention::Vsa(workspace))
                }
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
        }
        self.linear(
            &format!("{prefix}.attn.out_proj"),
            workspace.attention.pointer(),
            workspace.delta.pointer(),
            tokens,
            head,
            workspace,
        )?;

        let fc1 = format!("{prefix}.mlp.fc1");
        let fc2 = format!("{prefix}.mlp.fc2");
        let nvfp4_fc1 = self.nvfp4.get(&fc1).filter(|_| modulation.is_some());
        if let Some(modulation) = modulation.filter(|_| self.tensors.is_int8(&fc1)) {
            let int8_rows = if nvfp4_fc1.is_some() { head } else { tokens };
            if int8_rows > 0 {
                self.add_norm_quantize(
                    &format!("{prefix}.norm2.weight"),
                    &fc1,
                    workspace,
                    int8_rows,
                    modulation,
                    Some((modulation.0, 2)),
                    3,
                    4,
                )?;
                self.tensors.linear_quantized(
                    &fc1,
                    workspace.activated.pointer(),
                    int8_rows,
                    &workspace.quantized,
                    &workspace.scales,
                    adapter(&fc1),
                    true,
                )?;
            }
            if let Some(weight) = nvfp4_fc1 {
                return self.nvfp4_mlp(prefix, weight, workspace, tokens, head, modulation);
            }
        } else {
            // NOTE: without modulation, as in the text refiner, an INT8 fc1 runs as a plain GEMM
            // followed by `mmh3_swiglu`, since only the blocks' fc1 rows are interleaved.
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
                head,
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
            &fc2,
            workspace.activated.pointer(),
            workspace.delta.pointer(),
            tokens,
            head,
            workspace,
        )?;
        match modulation {
            Some(_) => Ok(()),
            None => self.add_residual(workspace, tokens, None),
        }
    }

    /// The MLP of block `prefix` with the NVFP4 gate and up projections `weight`, after the first
    /// `head` rows went through the INT8 ones into `activated`.
    fn nvfp4_mlp(
        &self,
        prefix: &str,
        weight: &Nvfp4Weight,
        workspace: &Workspace,
        tokens: usize,
        head: usize,
        modulation: (&DeviceBuffer, &DeviceBuffer),
    ) -> Result<(), Error> {
        let fc1 = format!("{prefix}.mlp.fc1");
        let fc2 = format!("{prefix}.mlp.fc2");
        let tail = tokens - head;
        self.add_norm_quantize_nvfp4(
            &format!("{prefix}.norm2.weight"),
            &fc1,
            workspace,
            head,
            tail,
            modulation,
            Some((modulation.0, 2)),
            3,
            4,
        )?;
        let activations = self.nvfp4_activations(workspace);
        let ffn = self.config.ffn;
        // SAFETY: expanded holds `tokens × 2 × ffn` values, activated `tokens × ffn` and delta
        // `tokens × hidden`.
        unsafe {
            nvfp4::gemm_pointers(weight, activations, workspace.expanded.pointer(), tail)?;
            let Some(down) = self.nvfp4.get(&fc2) else {
                check(mmh3_swiglu(
                    workspace.expanded.pointer(),
                    workspace.activated.pointer_at(head * ffn * 2),
                    tail as c_int,
                    ffn as c_int,
                    ptr::null_mut(),
                ))?;
                return self.linear(
                    &fc2,
                    workspace.activated.pointer(),
                    workspace.delta.pointer(),
                    tokens,
                    0,
                    workspace,
                );
            };
            if head > 0 {
                self.tensors.linear(
                    &fc2,
                    workspace.activated.pointer(),
                    workspace.delta.pointer(),
                    head,
                    &workspace.quantized,
                    &workspace.scales,
                    self.adapters
                        .get(&fc2)
                        .map(|adapter| (adapter, &workspace.adapter)),
                )?;
            }
            nvfp4::quantize_pointers(
                Nvfp4Input::SwiGlu(workspace.expanded.pointer()),
                tail,
                down,
                self.nvfp4_scale(&fc2)?,
                activations,
            )?;
            nvfp4::gemm_pointers(
                down,
                activations,
                workspace.delta.pointer_at(head * down.outputs * 2),
                tail,
            )?;
        }
        Ok(())
    }

    /// Writes the VSA gates of block `prefix` into the workspace and returns them, or None when
    /// the block has no gates. NVFP4 gates take the NVFP4 input of the block's qkv projection, and
    /// its INT8 input for the first `head` rows. With `fused`, the INT8 input of the block's qkv
    /// projection is still in the quantization buffers, and otherwise its BF16 input is in
    /// `normalized`.
    fn vsa_gate(
        &self,
        prefix: &str,
        workspace: &Workspace,
        tokens: usize,
        head: usize,
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
        if let Some(weight) = self.nvfp4.get(&name) {
            if head > 0 {
                self.tensors.linear_quantized(
                    &name,
                    gate.pointer(),
                    head,
                    &workspace.quantized,
                    &workspace.scales,
                    None,
                    false,
                )?;
            }
            // SAFETY: the gate holds `tokens × inner` values, and the NVFP4 activations still hold
            // the input of the block's qkv projection.
            unsafe {
                nvfp4::gemm_pointers(
                    weight,
                    self.nvfp4_activations(workspace),
                    gate.pointer_at(head * weight.outputs * 2),
                    tokens - head,
                )?
            };
        } else if fused && self.tensors.is_int8(&name) {
            self.tensors.linear_quantized(
                &name,
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
                head,
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
            0,
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
                0,
                None,
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
        self.run(inputs, capture, sparse, None)
    }

    /// One rank's share of a step. Ulysses gives this rank a run of the sequence to carry through
    /// the blocks and a run of the heads to attend with, and `context` carries the two exchanges
    /// each block needs. Rank 0 returns the velocity; the others return nothing to unpatchify.
    pub fn forward_shard(
        &self,
        inputs: &DitInputs,
        sparse: Option<&SparseAttention>,
        context: &mut ShardContext<'_>,
    ) -> Result<DitOutputs, Error> {
        self.run(inputs, &[], sparse, Some(context))
    }

    fn run(
        &self,
        inputs: &DitInputs,
        capture: &[usize],
        sparse: Option<&SparseAttention>,
        mut context: Option<&mut ShardContext<'_>>,
    ) -> Result<DitOutputs, Error> {
        let config = &self.config;
        let hidden = config.hidden;
        let video_shape = &inputs.video.shape;
        let text_tokens = inputs.context.shape[0];
        let layout = PackedLayout::for_inputs(inputs);
        let tokens = layout.len();
        // The rows this rank carries through the blocks. Without a shard that is every row, and the
        // block loop below does not know the difference.
        let rows = match &context {
            Some(context) => context.shard.own_tokens(),
            None => 0..tokens,
        };
        let heads = match &context {
            Some(context) => context.shard.own_heads(),
            None => 0..config.heads,
        };
        if let Some(context) = &context {
            if context.shard.tokens.iter().map(Range::len).sum::<usize>() != tokens
                || context.shard.heads.iter().map(Range::len).sum::<usize>() != config.heads
            {
                return Err(Error::Model(
                    "a shard that does not cover the step".to_owned(),
                ));
            }
            if !capture.is_empty() {
                return Err(Error::Model(
                    "a sharded step cannot capture blocks".to_owned(),
                ));
            }
            if !self.nvfp4.is_empty() {
                return Err(Error::Model(
                    "a sharded step cannot run NVFP4 layers".to_owned(),
                ));
            }
            // A rank exchanges the quantized block input, which only the fused normalize and
            // quantize pass of an INT8 qkv projection writes. Without it the peers would read
            // memory no block filled.
            if let Some(layer) = (0..config.layers).find(|layer| {
                !self
                    .tensors
                    .is_int8(&format!("blocks.{layer}.attn.qkv_proj"))
            }) {
                return Err(Error::Model(format!(
                    "a sharded step needs INT8 attention projections, and block {layer} has none"
                )));
            }
            if context.shard.rank > 0 && rows.start < text_tokens {
                return Err(Error::Model(
                    "the text rows must fall to rank 0 alone".to_owned(),
                ));
            }
        }
        let timesteps = StepTimesteps::for_layout(
            &layout,
            inputs.sigma,
            inputs.shift_video,
            inputs.shift_audio,
        );
        let steps = timesteps.values.len();
        let adapter_rank = self
            .adapters
            .values()
            .map(LowRank::scratch_columns)
            .max()
            .unwrap_or(0);
        // The refiner's MLPs still write the gate and up projections for the text tokens.
        let fused = (0..config.layers).all(|layer| {
            let fc1 = format!("blocks.{layer}.mlp.fc1");
            self.tensors.is_int8(&fc1) && !self.nvfp4.contains_key(&fc1)
        });
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
            tokens: rows.len(),
            expanded_rows: if fused {
                text_tokens.saturating_sub(rows.start).min(rows.len())
            } else {
                rows.len()
            },
            adapter_rank,
            adapter_rows: tokens,
            gated: plan.is_some() && self.has_vsa_gates(),
            nvfp4_columns: self
                .nvfp4
                .values()
                .map(|weight| weight.columns)
                .max()
                .unwrap_or(0),
            nvfp4_adapter_rank: self
                .nvfp4
                .values()
                .map(Nvfp4Weight::adapter_rank)
                .max()
                .unwrap_or(0),
        };
        let workspace = match cached_workspace.take() {
            Some(mut workspace) if workspace.shape == shape => {
                workspace.residual.fill_f32(0.0)?;
                workspace
            }
            _ => Workspace::new(config, shape)?,
        };
        let workspace = cached_workspace.insert(workspace);
        let wanted = quantized_dense.then(|| (tokens, heads.len()));
        if workspace
            .attention_quantized
            .as_ref()
            .map(|quantized| (quantized.tokens(), quantized.heads()))
            != wanted
        {
            workspace.attention_quantized = None;
            workspace.attention_quantized = wanted
                .map(|(tokens, heads)| QuantizedWorkspace::new(tokens, heads))
                .transpose()?;
        }
        let workspace = &*workspace;
        match sparse_tau {
            Some(_)
                if cached_sparse.as_ref().is_some_and(|(precision, sparse)| {
                    *precision == self.attention_precision
                        && sparse.tokens() == tokens
                        && sparse.heads() == heads.len()
                }) => {}
            Some(_) => {
                *cached_sparse = None;
                *cached_sparse = Some((
                    self.attention_precision,
                    // A sharded step attends with its own heads over the whole sequence.
                    SparseWorkspace::with_precision(tokens, heads.len(), self.attention_precision)?,
                ));
            }
            None => *cached_sparse = None,
        }
        match &plan {
            Some(plan)
                if cached_vsa
                    .as_ref()
                    .is_some_and(|(precision, cached, workspace)| {
                        *precision == self.attention_precision
                            && cached == plan
                            && workspace.heads() == heads.len()
                    }) => {}
            Some(plan) => {
                *cached_vsa = None;
                *cached_vsa = Some((
                    self.attention_precision,
                    plan.clone(),
                    // A sharded step attends with its own heads over the whole sequence.
                    VsaWorkspace::with_precision(
                        plan,
                        tokens,
                        heads.len(),
                        self.attention_precision,
                    )?,
                ));
            }
            None => *cached_vsa = None,
        }

        // Every table below is one entry per token, so a rank takes the slice its rows cover. The
        // rope angles are the exception: attention sees the whole sequence once the inputs have
        // been exchanged, so they stay whole.
        let block_rows = i32_buffer(&layout.modulation_rows(&timesteps)[rows.clone()])?;
        let final_rows: Vec<usize> = layout
            .segments
            .iter()
            .flat_map(|segment| {
                std::iter::repeat_n(timesteps.index_of(segment.kind), segment.len())
            })
            .collect();
        let final_rows = i32_buffer(&final_rows[rows.clone()])?;
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

        // The text rows sit at the start of the sequence, so rank 0 refines them and no other rank
        // holds any of them.
        if rows.start == 0 {
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
                    unsafe {
                        copy_device(states.pointer(), workspace.residual.pointer(), text_bytes)?
                    };
                    *cached_text = Some(TextStates {
                        context: inputs.context.clone(),
                        states,
                    });
                }
            }
        }

        // Each segment's patches enter the residual stream at its own place, and a rank embeds only
        // the part of it that falls in its rows.
        let embed =
            |projection: &str, values: &[f32], segment: Range<usize>| -> Result<(), Error> {
                let first = segment.start.max(rows.start);
                let last = segment.end.min(rows.end);
                if first >= last {
                    return Ok(());
                }
                let width = values.len() / segment.len();
                let taken = ((first - segment.start) * width)..((last - segment.start) * width);
                let buffer = DeviceBuffer::from_f32(&values[taken])?;
                self.linear(
                    projection,
                    buffer.pointer(),
                    workspace
                        .residual
                        .pointer_at((first - rows.start) * hidden * 4),
                    last - first,
                    0,
                    workspace,
                )
            };
        embed(
            "audio_patch_proj",
            &pack_audio(&inputs.audio),
            audio.start..audio.end,
        )?;
        embed("video_patch_proj", &video_rows, video.start..video.end)?;
        for segment in &layout.segments {
            let (projection, rows) = match segment.kind {
                SegmentKind::KeyframeVideo(index) => (
                    "video_patch_proj",
                    patchify_video(inputs.keyframes[index].video.as_ref().unwrap()),
                ),
                SegmentKind::KeyframeAudio(index) => (
                    "audio_patch_proj",
                    pack_audio(inputs.keyframes[index].audio.as_ref().unwrap()),
                ),
                SegmentKind::ReferenceVideo(index) => (
                    "video_patch_proj",
                    patchify_video(inputs.references[index].video().unwrap()),
                ),
                SegmentKind::ReferenceAudio(index) => (
                    "audio_patch_proj",
                    pack_audio(inputs.references[index].audio().unwrap()),
                ),
                _ => continue,
            };
            embed(projection, &rows, segment.start..segment.end)?;
        }
        // NOTE: NVFP4 layers run the rows before the video (text, conditions and audio), about 1% of
        // the rows at 768p without conditions, through their INT8 weights. At 448×256 against the FP32 reference, this takes the velocity from
        // 3.3e-1 to 2.0e-1 for video and from 2.6e-1 to 9.2e-2 for audio (INT8 6.7e-2 and
        // 3.9e-2). The text rows alone give 2.4e-1 and 2.3e-1.
        let head = if self.nvfp4.is_empty() {
            0
        } else {
            video.start
        };

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
                rows.len(),
                head,
                Some(&angles),
                Some((modulation, &block_rows)),
                pending,
                sparse_pass.as_ref(),
                context.as_deref_mut().map(|context| (context, tokens)),
            )?;
            pending = Some(modulation);
            if let Some(SparsePass::Sol { workspace, .. }) = &sparse_pass {
                routed += workspace.routed_fraction()?;
            }
            if capture.contains(&layer) {
                self.add_pending(workspace, rows.len(), modulation, &block_rows)?;
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
            self.add_pending(workspace, rows.len(), modulation, &block_rows)?;
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
            rows.len(),
            Some((&final_modulation, &final_rows, FINAL_CHUNKS, 0, 1)),
        )?;
        // A rank projects the part of each segment its rows cover, which is the whole of it when
        // nothing is shared out.
        let project = |segment: mmh3_core::dit::layout::Segment,
                       layer: &str,
                       width: usize|
         -> Result<Vec<f32>, Error> {
            let first = segment.start.max(rows.start);
            let last = segment.end.min(rows.end);
            if first >= last {
                return Ok(Vec::new());
            }
            let output = DeviceBuffer::new((last - first) * width * 4)?;
            self.linear(
                layer,
                workspace
                    .projected
                    .pointer_at((first - rows.start) * hidden * 4),
                output.pointer(),
                last - first,
                0,
                workspace,
            )?;
            Ok(output.to_f32()?)
        };
        let mut video_velocity = project(
            video,
            "final_layer.video_out",
            config.video_patch_features(),
        )?;
        let audio_velocity = project(audio, "final_layer.audio_out", config.audio_channels)?;
        if context.is_some() {
            // The rows of one rank cannot be unpatchified on their own, and in VSA's order they are
            // not even the rows the video wants. `assemble_velocity` puts the parts together.
            return Ok(DitOutputs {
                blocks,
                video: Vec::new(),
                audio: Vec::new(),
                routed_fraction: None,
                part: Some(VelocityPart {
                    rows: rows.clone(),
                    video: video_velocity,
                    audio: audio_velocity,
                }),
            });
        }
        if let Some(plan) = &plan {
            video_velocity = plan.restore_video(&video_velocity, config.video_patch_features());
        }
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
            part: None,
        })
    }

    /// Puts the parts of a shared-out step back together, which only the rank that answers to the
    /// caller needs to do. The parts may arrive in any order and must cover the sequence once.
    pub fn assemble_velocity(
        &self,
        inputs: &DitInputs,
        sparse: Option<&SparseAttention>,
        parts: &[VelocityPart],
    ) -> Result<DitOutputs, Error> {
        let config = &self.config;
        let layout = PackedLayout::for_inputs(inputs);
        let (video, audio) = (
            layout.segment(SegmentKind::Video),
            layout.segment(SegmentKind::Audio),
        );
        let plan = sparse
            .filter(|settings| layout.len() >= settings.min_tokens)
            .and_then(|settings| match settings.method {
                SparseMethod::Vsa { .. } => Some(VsaPlan::for_layout(&layout)),
                _ => None,
            });
        let gather = |segment: mmh3_core::dit::layout::Segment,
                      width: usize,
                      take: &dyn Fn(&VelocityPart) -> &Vec<f32>|
         -> Result<Vec<f32>, Error> {
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
                    return Err(Error::Model(format!(
                        "a part of {} values for {} rows",
                        rows.len(),
                        last - first
                    )));
                }
                values[(first - segment.start) * width..(last - segment.start) * width]
                    .copy_from_slice(rows);
                covered += last - first;
            }
            if covered != segment.len() {
                return Err(Error::Model(format!(
                    "the parts cover {covered} of {} rows",
                    segment.len()
                )));
            }
            Ok(values)
        };
        let mut video_velocity = gather(video, config.video_patch_features(), &|part| &part.video)?;
        if let Some(plan) = &plan {
            video_velocity = plan.restore_video(&video_velocity, config.video_patch_features());
        }
        let audio_velocity = gather(audio, config.audio_channels, &|part| &part.audio)?;
        Ok(DitOutputs {
            blocks: Vec::new(),
            video: unpatchify_video(&video_velocity, &inputs.video.shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            audio: unpack_audio(&audio_velocity, &inputs.audio.shape)
                .into_iter()
                .map(|value| -value)
                .collect(),
            routed_fraction: None,
            part: None,
        })
    }
}
