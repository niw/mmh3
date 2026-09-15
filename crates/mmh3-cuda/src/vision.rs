//! The Qwen3-VL vision tower on the GPU, which embeds the pictures of a prompt for the text
//! encoder.
//!
//! A picture's 16 × 16 patches go through 27 transformer blocks with full attention over the
//! picture and 2D rotary positions. A merger joins each 2 × 2 patches into one embedding of the
//! language model's width, and three more mergers turn the outputs of blocks 8, 16 and 24 into the
//! DeepStack features the language model adds to the picture's embeddings after its first three
//! layers. Like ComfyUI it runs in FP32 with the BF16 weights of the checkpoint.

use crate::model::{Error, LinearKind};
use crate::{DeviceBuffer, check};
use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::vision::{
    PATCH_VALUES, VisionGrid, patchify_picture, position_embeddings, rotary_angles,
};
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_vision_layer_norm(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        epsilon: f32,
        rows: c_int,
        width: c_int,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vision_gelu(
        values: *mut c_void,
        count: usize,
        tanh_approximation: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vision_split_rotate(
        qkv: *const c_void,
        angles: *const c_void,
        tokens: c_int,
        heads: c_int,
        dim: c_int,
        query: *mut c_void,
        key: *mut c_void,
        values: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vision_softmax(
        scores: *mut c_void,
        rows: c_int,
        columns: c_int,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_vision_merge_heads(
        input: *const c_void,
        tokens: c_int,
        heads: c_int,
        dim: c_int,
        output: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    pub(crate) fn mmh3_vision_add(
        values: *mut c_void,
        other: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_cublaslt_matmul(
        kind: c_int,
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        alpha: f32,
        beta: f32,
        stream: *mut c_void,
    ) -> c_int;
}

const PREFIX: &str = "visual.";
const HEAD_DIM: usize = 72;
const NORM_EPSILON: f32 = 1e-6;
/// The blocks whose outputs become DeepStack features, in the order the language model adds them.
const DEEPSTACK_BLOCKS: [usize; 3] = [8, 16, 24];

fn load(file: &SafeTensors, name: &str) -> Result<Tensor, Error> {
    let info = file
        .get(&format!("{PREFIX}{name}"))
        .ok_or_else(|| Error::Model(format!("missing tensor {PREFIX}{name}")))?;
    Tensor::load(file, info).map_err(Error::Model)
}

/// An FP32 linear layer `[outputs, inputs]` with a bias.
struct Linear {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    inputs: usize,
    outputs: usize,
}

impl Linear {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = load(file, &format!("{name}.weight"))?;
        let (outputs, inputs) = (weight.shape[0], weight.data.len() / weight.shape[0]);
        Ok(Linear {
            weight: DeviceBuffer::from_f32(&weight.data)?,
            bias: DeviceBuffer::from_f32(&load(file, &format!("{name}.bias"))?.data)?,
            inputs,
            outputs,
        })
    }

    /// `output = input · weightᵀ + bias` over `rows` rows, plus `beta · output`.
    fn apply(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
        beta: f32,
    ) -> Result<(), Error> {
        assert!(
            input.bytes() >= rows * self.inputs * 4 && output.bytes() >= rows * self.outputs * 4,
            "the buffers are smaller than their rows"
        );
        // SAFETY: the buffers hold their rows, checked above.
        check(unsafe {
            mmh3_cublaslt_matmul(
                LinearKind::F32 as c_int,
                input.pointer(),
                self.weight.pointer(),
                self.bias.pointer(),
                output.pointer(),
                rows as i64,
                self.outputs as i64,
                self.inputs as i64,
                1.0,
                beta,
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}

struct LayerNorm {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
    width: usize,
}

impl LayerNorm {
    fn load(file: &SafeTensors, name: &str) -> Result<Self, Error> {
        let weight = load(file, &format!("{name}.weight"))?;
        Ok(LayerNorm {
            width: weight.data.len(),
            weight: DeviceBuffer::from_f32(&weight.data)?,
            bias: DeviceBuffer::from_f32(&load(file, &format!("{name}.bias"))?.data)?,
        })
    }

    fn apply(&self, input: &DeviceBuffer, output: &DeviceBuffer, rows: usize) -> Result<(), Error> {
        assert!(
            input.bytes() >= rows * self.width * 4 && output.bytes() >= rows * self.width * 4,
            "the buffers are smaller than their rows"
        );
        // SAFETY: the buffers hold their rows, checked above.
        check(unsafe {
            mmh3_vision_layer_norm(
                input.pointer(),
                self.weight.pointer(),
                self.bias.pointer(),
                NORM_EPSILON,
                rows as c_int,
                self.width as c_int,
                output.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    projection: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// A merger: a layer norm, then fc1, GELU and fc2 on each 2 × 2 patches, with the norm over the
/// joined patches (DeepStack) or over each patch before joining them (the main merger).
struct Merger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

/// The embeddings of one picture for the language model, `[tokens, hidden]` each in FP32.
pub struct VisionEmbeddings {
    pub merged: DeviceBuffer,
    /// Added to the picture's embeddings after the language model's first layers, in order.
    pub deepstack: Vec<DeviceBuffer>,
    pub tokens: usize,
}

pub struct CudaVisionEncoder {
    patch_embed: Linear,
    position_table: Tensor,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
    hidden: usize,
    heads: usize,
}

impl CudaVisionEncoder {
    /// Loads the vision tower of a Qwen3-VL text encoder checkpoint, `visual.*`, into FP32.
    pub fn load(file: &SafeTensors) -> Result<Self, Error> {
        let patch_weight = load(file, "patch_embed.proj.weight")?;
        let hidden = patch_weight.shape[0];
        if patch_weight.data.len() != hidden * PATCH_VALUES || !hidden.is_multiple_of(HEAD_DIM) {
            return Err(Error::Model(format!(
                "unsupported vision patch embedding {:?}",
                patch_weight.shape
            )));
        }
        let blocks = (0..)
            .take_while(|block| {
                file.get(&format!("{PREFIX}blocks.{block}.norm1.weight"))
                    .is_some()
            })
            .map(|block| {
                let prefix = format!("blocks.{block}");
                Ok(Block {
                    norm1: LayerNorm::load(file, &format!("{prefix}.norm1"))?,
                    qkv: Linear::load(file, &format!("{prefix}.attn.qkv"))?,
                    projection: Linear::load(file, &format!("{prefix}.attn.proj"))?,
                    norm2: LayerNorm::load(file, &format!("{prefix}.norm2"))?,
                    fc1: Linear::load(file, &format!("{prefix}.mlp.linear_fc1"))?,
                    fc2: Linear::load(file, &format!("{prefix}.mlp.linear_fc2"))?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let merger = |name: &str| -> Result<Merger, Error> {
            Ok(Merger {
                norm: LayerNorm::load(file, &format!("{name}.norm"))?,
                fc1: Linear::load(file, &format!("{name}.linear_fc1"))?,
                fc2: Linear::load(file, &format!("{name}.linear_fc2"))?,
            })
        };
        let deepstack = (0..DEEPSTACK_BLOCKS.len())
            .map(|index| merger(&format!("deepstack_merger_list.{index}")))
            .collect::<Result<Vec<_>, Error>>()?;
        if blocks.len() <= DEEPSTACK_BLOCKS[DEEPSTACK_BLOCKS.len() - 1] {
            return Err(Error::Model(format!(
                "a vision tower of {} blocks has no DeepStack features",
                blocks.len()
            )));
        }
        Ok(CudaVisionEncoder {
            patch_embed: Linear {
                weight: DeviceBuffer::from_f32(&patch_weight.data)?,
                bias: DeviceBuffer::from_f32(&load(file, "patch_embed.proj.bias")?.data)?,
                inputs: PATCH_VALUES,
                outputs: hidden,
            },
            position_table: load(file, "pos_embed.weight")?,
            blocks,
            merger: merger("merger")?,
            deepstack,
            hidden,
            heads: hidden / HEAD_DIM,
        })
    }

    /// The width of the embeddings, the language model's hidden size.
    pub fn output_width(&self) -> usize {
        self.merger.fc2.outputs
    }

    /// Embeds a picture `[height, width, 3]` in [0, 1] with sides that are multiples of 32.
    pub fn encode(&self, picture: &Tensor) -> Result<VisionEmbeddings, Error> {
        let (grid, patches) = patchify_picture(picture);
        let (hidden, rows) = (self.hidden, grid.patches());
        let merged_rows = grid.tokens();
        let buffer = |values: usize| DeviceBuffer::new(values * 4);

        let patches = DeviceBuffer::from_f32(&patches)?;
        let states = buffer(rows * hidden)?;
        self.patch_embed.apply(&patches, &states, rows, 0.0)?;
        let positions = DeviceBuffer::from_f32(&position_embeddings(&self.position_table, grid))?;
        // SAFETY: both buffers hold `rows × hidden` values.
        check(unsafe {
            mmh3_vision_add(
                states.pointer(),
                positions.pointer(),
                rows * hidden,
                ptr::null_mut(),
            )
        })?;
        let angles = DeviceBuffer::from_f32(&rotary_angles(grid))?;

        let normalized = buffer(rows * hidden)?;
        let qkv = buffer(rows * 3 * hidden)?;
        let query = buffer(rows * hidden)?;
        let key = buffer(rows * hidden)?;
        let values = buffer(rows * hidden)?;
        let scores = buffer(rows * rows)?;
        let heads_output = buffer(rows * hidden)?;
        let attention = buffer(rows * hidden)?;
        let ffn = self.blocks[0].fc1.outputs;
        let expanded = buffer(rows * ffn)?;
        let mut deepstack = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            block.norm1.apply(&states, &normalized, rows)?;
            block.qkv.apply(&normalized, &qkv, rows, 0.0)?;
            self.attend(
                &qkv,
                &angles,
                grid,
                [&query, &key, &values, &scores, &heads_output, &attention],
            )?;
            block.projection.apply(&attention, &states, rows, 1.0)?;
            block.norm2.apply(&states, &normalized, rows)?;
            block.fc1.apply(&normalized, &expanded, rows, 0.0)?;
            // SAFETY: expanded holds `rows × ffn` values.
            check(unsafe { mmh3_vision_gelu(expanded.pointer(), rows * ffn, 1, ptr::null_mut()) })?;
            block.fc2.apply(&expanded, &states, rows, 1.0)?;
            if let Some(position) = DEEPSTACK_BLOCKS.iter().position(|&block| block == index) {
                // The DeepStack merger norms the joined patches.
                let merger = &self.deepstack[position];
                merger.norm.apply(&states, &normalized, merged_rows)?;
                deepstack.push(self.merge(merger, &normalized, merged_rows)?);
            }
        }
        // The main merger norms each patch before joining them.
        self.merger.norm.apply(&states, &normalized, rows)?;
        let merged = self.merge(&self.merger, &normalized, merged_rows)?;
        Ok(VisionEmbeddings {
            merged,
            deepstack,
            tokens: merged_rows,
        })
    }

    /// fc2(GELU(fc1(rows))) over rows of 2 × 2 joined patches.
    fn merge(
        &self,
        merger: &Merger,
        joined: &DeviceBuffer,
        rows: usize,
    ) -> Result<DeviceBuffer, Error> {
        let hidden = DeviceBuffer::new(rows * merger.fc1.outputs * 4)?;
        merger.fc1.apply(joined, &hidden, rows, 0.0)?;
        // SAFETY: hidden holds `rows × fc1 outputs` values.
        check(unsafe {
            mmh3_vision_gelu(
                hidden.pointer(),
                rows * merger.fc1.outputs,
                0,
                ptr::null_mut(),
            )
        })?;
        let output = DeviceBuffer::new(rows * merger.fc2.outputs * 4)?;
        merger.fc2.apply(&hidden, &output, rows, 0.0)?;
        Ok(output)
    }

    /// Full attention over the patches of one picture, head by head.
    fn attend(
        &self,
        qkv: &DeviceBuffer,
        angles: &DeviceBuffer,
        grid: VisionGrid,
        [query, key, values, scores, heads_output, attention]: [&DeviceBuffer; 6],
    ) -> Result<(), Error> {
        let (rows, heads) = (grid.patches(), self.heads);
        let head_values = rows * HEAD_DIM;
        // SAFETY: qkv holds `rows × 3 × hidden` values, the angles `rows × 36` and the outputs
        // `rows × hidden` each.
        check(unsafe {
            mmh3_vision_split_rotate(
                qkv.pointer(),
                angles.pointer(),
                rows as c_int,
                heads as c_int,
                HEAD_DIM as c_int,
                query.pointer(),
                key.pointer(),
                values.pointer(),
                ptr::null_mut(),
            )
        })?;
        let at = |buffer: &DeviceBuffer, head: usize| buffer.pointer_at(head * head_values * 4);
        for head in 0..heads {
            // SAFETY: each head holds `rows × 72` values and the scores `rows × rows`.
            unsafe {
                check(mmh3_cublaslt_matmul(
                    LinearKind::F32 as c_int,
                    at(query, head),
                    at(key, head),
                    ptr::null(),
                    scores.pointer(),
                    rows as i64,
                    rows as i64,
                    HEAD_DIM as i64,
                    1.0,
                    0.0,
                    ptr::null_mut(),
                ))?;
                check(mmh3_vision_softmax(
                    scores.pointer(),
                    rows as c_int,
                    rows as c_int,
                    1.0 / (HEAD_DIM as f32).sqrt(),
                    ptr::null_mut(),
                ))?;
                check(mmh3_cublaslt_matmul(
                    LinearKind::F32 as c_int,
                    scores.pointer(),
                    at(values, head),
                    ptr::null(),
                    at(heads_output, head),
                    rows as i64,
                    HEAD_DIM as i64,
                    rows as i64,
                    1.0,
                    0.0,
                    ptr::null_mut(),
                ))?;
            }
        }
        // SAFETY: both buffers hold `rows × hidden` values.
        check(unsafe {
            mmh3_vision_merge_heads(
                heads_output.pointer(),
                rows as c_int,
                heads as c_int,
                HEAD_DIM as c_int,
                attention.pointer(),
                ptr::null_mut(),
            )
        })?;
        Ok(())
    }
}
