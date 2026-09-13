//! The MiniMax H3 text encoder on the GPU: the first 50 layers of the Qwen3-VL-32B language model. The hidden state
//! after the last layer, without the final norm, conditions the DiT.
//!
//! The residual stream stays in FP32 and the linear layers run through the INT8 ConvRot GEMM like the DiT's. The q, k
//! and v projections, and the gate and up projections, are concatenated on load so each pair or triple takes one GEMM.

use crate::attention::{AttentionLayout, Element, attention_pointers};
use crate::loader::Uploader;
use crate::model::{CONVROT_GROUP, DeviceTensors, Error, check_quantization};
use crate::{DeviceBuffer, check};
use mmh3_core::safetensors::{DType, SafeTensors, TensorInfo};
use mmh3_core::tensor::Tensor;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_embedding_bf16(ids: *const c_void, table: *const c_void, output: *mut c_void, tokens: c_int, hidden: c_int, stream: *mut c_void) -> c_int;
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
}

const HEAD_DIM: usize = 128;
const NORM_EPSILON: f32 = 1e-6;
const ROPE_THETA: f32 = 5_000_000.0;
const PREFIX: &str = "model.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEncoderConfig {
    pub vocabulary: usize,
    pub hidden: usize,
    pub layers: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub ffn: usize,
}

impl TextEncoderConfig {
    fn qkv_width(&self) -> usize {
        (self.query_heads + 2 * self.key_value_heads) * HEAD_DIM
    }
}

/// Buffers sized for one prompt.
struct Workspace {
    residual: DeviceBuffer,
    normalized: DeviceBuffer,
    qkv: DeviceBuffer,
    attention: DeviceBuffer,
    delta: DeviceBuffer,
    expanded: DeviceBuffer,
    activated: DeviceBuffer,
    quantized: DeviceBuffer,
    scales: DeviceBuffer,
}

impl Workspace {
    fn new(config: &TextEncoderConfig, tokens: usize) -> Result<Self, Error> {
        let (hidden, ffn, inner) = (config.hidden, config.ffn, config.query_heads * HEAD_DIM);
        Ok(Workspace {
            residual: DeviceBuffer::new(tokens * hidden * 4)?,
            normalized: DeviceBuffer::new(tokens * hidden * 2)?,
            qkv: DeviceBuffer::new(tokens * config.qkv_width() * 2)?,
            attention: DeviceBuffer::new(tokens * inner * 2)?,
            delta: DeviceBuffer::new(tokens * hidden * 2)?,
            expanded: DeviceBuffer::new(tokens * 2 * ffn * 2)?,
            activated: DeviceBuffer::new(tokens * ffn * 2)?,
            quantized: DeviceBuffer::new(tokens * hidden.max(inner).max(ffn))?,
            scales: DeviceBuffer::new(tokens * 4)?,
        })
    }
}

/// The encoder output and, when requested, the hidden states after some layers.
pub struct TextEncoding {
    /// `[tokens, hidden]`.
    pub context: Tensor,
    pub layers: Vec<(usize, Tensor)>,
}

pub struct CudaTextEncoder {
    config: TextEncoderConfig,
    tensors: DeviceTensors,
}

fn tensor_info<'a>(file: &'a SafeTensors, name: &str) -> Result<&'a TensorInfo, Error> {
    file.get(&format!("{PREFIX}{name}")).ok_or_else(|| Error::Model(format!("missing tensor {PREFIX}{name}")))
}

impl CudaTextEncoder {
    /// Uploads the language model of a ComfyUI text encoder checkpoint with INT8 ConvRot linear layers.
    pub fn load(file: &SafeTensors) -> Result<Self, Error> {
        let embedding = tensor_info(file, "embed_tokens.weight")?;
        let query = tensor_info(file, "layers.0.self_attn.q_proj.weight")?;
        let key = tensor_info(file, "layers.0.self_attn.k_proj.weight")?;
        let config = TextEncoderConfig {
            vocabulary: embedding.shape[0],
            hidden: embedding.shape[1],
            layers: (0..).take_while(|layer| file.get(&format!("{PREFIX}layers.{layer}.input_layernorm.weight")).is_some()).count(),
            query_heads: query.shape[0] / HEAD_DIM,
            key_value_heads: key.shape[0] / HEAD_DIM,
            ffn: tensor_info(file, "layers.0.mlp.gate_proj.weight")?.shape[0],
        };
        if embedding.dtype != DType::BF16 || config.query_heads % config.key_value_heads != 0 || config.hidden % CONVROT_GROUP != 0 {
            return Err(Error::Model(format!("unsupported text encoder {config:?} with {} embeddings", embedding.dtype)));
        }

        let mut tensors = DeviceTensors::default();
        let mut uploader = Uploader::new(file);
        tensors.insert("embed_tokens.weight", file, embedding, &mut uploader)?;
        for layer in 0..config.layers {
            let prefix = format!("layers.{layer}");
            for name in ["input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"] {
                let name = format!("{prefix}.{name}.weight");
                tensors.insert(&name, file, tensor_info(file, &name)?, &mut uploader)?;
            }
            for (name, parts) in [
                ("self_attn.qkv_proj", &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"][..]),
                ("self_attn.o_proj", &["self_attn.o_proj"][..]),
                ("mlp.gate_up_proj", &["mlp.gate_proj", "mlp.up_proj"][..]),
                ("mlp.down_proj", &["mlp.down_proj"][..]),
            ] {
                let parts: Vec<String> = parts.iter().map(|part| format!("{prefix}.{part}")).collect();
                Self::insert_concatenated(&mut tensors, &mut uploader, file, &format!("{prefix}.{name}"), &parts)?;
            }
        }
        uploader.run()?;
        Ok(CudaTextEncoder { config, tensors })
    }

    /// Queues the INT8 ConvRot layers `parts` as one layer `name` whose output rows are theirs in order.
    fn insert_concatenated(tensors: &mut DeviceTensors, uploader: &mut Uploader, file: &SafeTensors, name: &str, parts: &[String]) -> Result<(), Error> {
        let mut weights = Vec::new();
        let mut scales = Vec::new();
        for part in parts {
            let quantization = tensor_info(file, &format!("{part}.comfy_quant"))?;
            check_quantization(part, file.data(quantization))?;
            let weight = tensor_info(file, &format!("{part}.weight"))?;
            let scale = tensor_info(file, &format!("{part}.weight_scale"))?;
            if weight.dtype != DType::I8 || scale.dtype != DType::F32 || scale.element_count() != weight.shape[0] {
                return Err(Error::Model(format!("{part} is not a per-channel INT8 layer")));
            }
            if weights.first().is_some_and(|first: &&TensorInfo| first.shape[1] != weight.shape[1]) {
                return Err(Error::Model(format!("{part} does not share its input width with {name}")));
            }
            weights.push(weight);
            scales.push(scale);
        }
        let outputs: usize = weights.iter().map(|weight| weight.shape[0]).sum();
        let features = weights[0].shape[1];
        let mut concatenate = |infos: &[&TensorInfo]| -> Result<DeviceBuffer, Error> {
            let buffer = DeviceBuffer::new(infos.iter().map(|info| info.byte_count()).sum())?;
            let mut offset = 0;
            for info in infos {
                uploader.queue(file, info, &buffer, offset);
                offset += info.byte_count();
            }
            Ok(buffer)
        };
        let weight = concatenate(&weights)?;
        let scale = concatenate(&scales)?;
        tensors.insert_buffer(&format!("{name}.weight"), weight, DType::I8, vec![outputs, features]);
        tensors.insert_buffer(&format!("{name}.weight_scale"), scale, DType::F32, vec![outputs]);
        Ok(())
    }

    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    fn linear(&self, name: &str, input: &DeviceBuffer, output: &DeviceBuffer, tokens: usize, workspace: &Workspace) -> Result<(), Error> {
        self.tensors.linear(name, input.pointer(), output.pointer(), tokens, &workspace.quantized, &workspace.scales, None)
    }

    fn normalize(&self, weight: &str, workspace: &Workspace, tokens: usize) -> Result<(), Error> {
        // SAFETY: the residual and normalized buffers hold `tokens × hidden` values.
        check(unsafe {
            mmh3_rms_norm_modulate(
                workspace.residual.pointer(),
                self.tensors.pointer(weight)?,
                ptr::null(),
                ptr::null(),
                0,
                0,
                0,
                workspace.normalized.pointer(),
                0,
                tokens as c_int,
                self.config.hidden as c_int,
                NORM_EPSILON,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    fn add_residual(&self, workspace: &Workspace, tokens: usize) -> Result<(), Error> {
        // SAFETY: the residual and delta buffers hold `tokens × hidden` values.
        check(unsafe {
            mmh3_gated_residual_add(
                workspace.residual.pointer(),
                workspace.delta.pointer(),
                ptr::null(),
                ptr::null(),
                0,
                0,
                tokens as c_int,
                self.config.hidden as c_int,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }

    /// Rotary angles `[tokens, 64]` of consecutive text positions.
    fn rope_angles(tokens: usize) -> Vec<f32> {
        let inverse: Vec<f32> = (0..HEAD_DIM / 2).map(|index| 1.0 / ROPE_THETA.powf((2 * index) as f32 / HEAD_DIM as f32)).collect();
        (0..tokens).flat_map(|position| inverse.iter().map(move |&frequency| position as f32 * frequency)).collect()
    }

    /// Encodes token ids and returns the conditioning `[tokens, hidden]`, capturing the hidden states after the
    /// listed layers.
    pub fn encode(&self, ids: &[u32], capture: &[usize]) -> Result<TextEncoding, Error> {
        let config = &self.config;
        let tokens = ids.len();
        if tokens == 0 || ids.iter().any(|&id| id as usize >= config.vocabulary) {
            return Err(Error::Model(format!("token ids must be non-empty and below {}", config.vocabulary)));
        }
        let workspace = Workspace::new(config, tokens)?;
        let id_buffer = DeviceBuffer::from_bytes(&ids.iter().flat_map(|&id| (id as i32).to_le_bytes()).collect::<Vec<_>>())?;
        // SAFETY: the ids are below the vocabulary size, checked above, and the residual holds `tokens × hidden`.
        check(unsafe {
            mmh3_embedding_bf16(
                id_buffer.pointer(),
                self.tensors.pointer("embed_tokens.weight")?,
                workspace.residual.pointer(),
                tokens as c_int,
                config.hidden as c_int,
                ptr::null_mut(),
            )
        })?;
        let angles = DeviceBuffer::from_f32(&Self::rope_angles(tokens))?;

        let qkv_width = config.qkv_width();
        let key_offset = config.query_heads * HEAD_DIM;
        let value_offset = key_offset + config.key_value_heads * HEAD_DIM;
        let layout = AttentionLayout {
            token_stride: [qkv_width as i64, qkv_width as i64, qkv_width as i64, (config.query_heads * HEAD_DIM) as i64],
            head_stride: [HEAD_DIM as i64; 4],
            heads_per_key_value: (config.query_heads / config.key_value_heads) as i32,
            causal: 1,
            ..AttentionLayout::default()
        };
        let mut layers = Vec::new();
        for layer in 0..config.layers {
            let prefix = format!("layers.{layer}");
            self.normalize(&format!("{prefix}.input_layernorm.weight"), &workspace, tokens)?;
            self.linear(&format!("{prefix}.self_attn.qkv_proj"), &workspace.normalized, &workspace.qkv, tokens, &workspace)?;
            let qkv = workspace.qkv.pointer().cast::<u16>();
            // SAFETY: qkv holds `tokens` rows of [q | k | v], the angles cover every token and the attention output
            // holds `tokens × query heads × 128` values.
            unsafe {
                check(mmh3_qk_norm_rope(
                    qkv.cast(),
                    self.tensors.pointer(&format!("{prefix}.self_attn.q_norm.weight"))?,
                    self.tensors.pointer(&format!("{prefix}.self_attn.k_norm.weight"))?,
                    angles.pointer(),
                    (HEAD_DIM / 2) as c_int,
                    tokens as c_int,
                    config.query_heads as c_int,
                    config.key_value_heads as c_int,
                    NORM_EPSILON,
                    ptr::null_mut(),
                ))?;
                attention_pointers(
                    Element::Bf16,
                    HEAD_DIM,
                    qkv.cast(),
                    qkv.add(key_offset).cast(),
                    qkv.add(value_offset).cast(),
                    workspace.attention.pointer(),
                    tokens,
                    config.query_heads,
                    1,
                    &layout,
                    1.0 / (HEAD_DIM as f32).sqrt(),
                )?;
            }
            self.linear(&format!("{prefix}.self_attn.o_proj"), &workspace.attention, &workspace.delta, tokens, &workspace)?;
            self.add_residual(&workspace, tokens)?;

            self.normalize(&format!("{prefix}.post_attention_layernorm.weight"), &workspace, tokens)?;
            self.linear(&format!("{prefix}.mlp.gate_up_proj"), &workspace.normalized, &workspace.expanded, tokens, &workspace)?;
            // SAFETY: expanded holds `tokens × 2 × ffn` values and activated `tokens × ffn`.
            check(unsafe {
                mmh3_swiglu(workspace.expanded.pointer(), workspace.activated.pointer(), tokens as c_int, config.ffn as c_int, ptr::null_mut())
            })?;
            self.linear(&format!("{prefix}.mlp.down_proj"), &workspace.activated, &workspace.delta, tokens, &workspace)?;
            self.add_residual(&workspace, tokens)?;
            if capture.contains(&layer) {
                layers.push((layer, Tensor::new(vec![tokens, config.hidden], workspace.residual.to_f32()?)));
            }
        }
        Ok(TextEncoding { context: Tensor::new(vec![tokens, config.hidden], workspace.residual.to_f32()?), layers })
    }
}
