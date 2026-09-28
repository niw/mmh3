//! The Qwen3-VL vision tower, which embeds the pictures of a prompt for the text encoder.
//!
//! A picture's 16 × 16 patches go through 27 transformer blocks with full attention over the
//! picture and 2D rotary positions. A merger joins each 2 × 2 patches into one embedding of the
//! language model's width, and three more mergers turn the outputs of blocks 8, 16 and 24 into the
//! DeepStack features the language model adds to the picture's embeddings after its first three
//! layers. Like ComfyUI it runs in FP32 with the BF16 weights of the checkpoint, which the products
//! widen a slab at a time.
use crate::{Error, Result, model::Weights, ops::Array};
use mmh3_core::{
    safetensors::SafeTensors,
    tensor::Tensor,
    vision::{
        MERGE, PATCH_VALUES, VisionGrid, patchify_frames, patchify_picture, position_embeddings,
        rotary_angles,
    },
};

const PREFIX: &str = "visual.";
const HEAD_DIM: usize = 72;
const NORM_EPSILON: f32 = 1e-6;
/// The blocks whose outputs become DeepStack features, in the order the language model adds them.
const DEEPSTACK_BLOCKS: [usize; 3] = [8, 16, 24];

/// The embeddings of one picture for the language model, `[tokens, hidden]` each.
pub struct VisionEmbeddings {
    pub merged: Array,
    /// Added to the picture's embeddings after the language model's first layers, in order.
    pub deepstack: Vec<Array>,
    pub tokens: usize,
}

pub struct MetalVisionEncoder {
    weights: Weights,
    /// The learned 48 × 48 grid of position embeddings, which each picture interpolates on the
    /// host like the reference, in BF16 steps.
    position_table: Tensor,
    blocks: usize,
    hidden: usize,
    heads: usize,
}

impl MetalVisionEncoder {
    /// Loads the vision tower of a Qwen3-VL text encoder checkpoint, `visual.*`, which is about a
    /// gigabyte of BF16 besides the language model.
    pub fn load(file: &SafeTensors) -> Result<Self> {
        let weights = Weights::load(file, PREFIX)?;
        let patch = weights.shape("patch_embed.proj.weight")?;
        let hidden = patch[0];
        if patch[1..].iter().product::<usize>() != PATCH_VALUES || !hidden.is_multiple_of(HEAD_DIM)
        {
            return Err(Error::new(format!(
                "unsupported vision patch embedding {patch:?}"
            )));
        }
        let blocks = (0..)
            .take_while(|block| weights.contains(&format!("blocks.{block}.norm1.weight")))
            .count();
        if blocks <= DEEPSTACK_BLOCKS[DEEPSTACK_BLOCKS.len() - 1]
            || (0..DEEPSTACK_BLOCKS.len()).any(|index| {
                !weights.contains(&format!("deepstack_merger_list.{index}.norm.weight"))
            })
        {
            return Err(Error::new(format!(
                "a vision tower of {blocks} blocks has no DeepStack features"
            )));
        }
        let position_table = weights.host("pos_embed.weight")?;
        Ok(Self {
            weights,
            position_table,
            blocks,
            hidden,
            heads: hidden / HEAD_DIM,
        })
    }

    /// The width of the embeddings, the language model's hidden size.
    pub fn output_width(&self) -> usize {
        self.weights
            .shape("merger.linear_fc2.weight")
            .map_or(0, |shape| shape[0])
    }

    /// Embeds a picture `[height, width, 3]` in [0, 1] with sides that are multiples of 32.
    pub fn encode(&self, picture: &Tensor) -> Result<VisionEmbeddings> {
        let (grid, patches) = patchify_picture(picture);
        self.embed(grid, &patches)
    }

    /// Embeds one block of a clip: two frames of the same size, which take a temporal slot each
    /// instead of a picture repeating itself.
    pub fn encode_frames(&self, first: &Tensor, second: &Tensor) -> Result<VisionEmbeddings> {
        let (grid, patches) = patchify_frames(&[first, second]);
        self.embed(grid, &patches)
    }

    fn embed(&self, grid: VisionGrid, patches: &[f32]) -> Result<VisionEmbeddings> {
        let w = &self.weights;
        let (hidden, rows, tokens) = (self.hidden, grid.patches(), grid.tokens());
        let device = &w.device;
        let patches = Array::from_f32(device, rows, PATCH_VALUES, patches)?;
        let positions = Array::from_f32(
            device,
            rows,
            hidden,
            &position_embeddings(&self.position_table, grid),
        )?;
        let mut states = w.linear(&patches, "patch_embed.proj")?.add(&positions)?;
        let angles = Array::from_f32(device, rows, HEAD_DIM / 2, &rotary_angles(grid))?;

        let mut deepstack = Vec::new();
        for block in 0..self.blocks {
            let p = format!("blocks.{block}");
            let normalized = w.norm(&states, &format!("{p}.norm1"), NORM_EPSILON)?;
            let qkv = w.linear(&normalized, &format!("{p}.attn.qkv"))?;
            // Rows of [3][heads][72]: the queries and keys rotate their first 36 values with
            // their last 36, which is how both the reference and `rope` pair them.
            let part = |index: usize| qkv.slice(0, rows, index * hidden, hidden);
            let query = part(0)?.rope(self.heads, &angles)?;
            let key = part(1)?.rope(self.heads, &angles)?;
            let attention = self.attend(&query, &key, &part(2)?)?;
            states = states.add(&w.linear(&attention, &format!("{p}.attn.proj"))?)?;
            let normalized = w.norm(&states, &format!("{p}.norm2"), NORM_EPSILON)?;
            let expanded = w
                .linear(&normalized, &format!("{p}.mlp.linear_fc1"))?
                .gelu(true)?;
            states = states.add(&w.linear(&expanded, &format!("{p}.mlp.linear_fc2"))?)?;
            if let Some(index) = DEEPSTACK_BLOCKS.iter().position(|&at| at == block) {
                // The DeepStack merger norms the joined patches.
                let name = format!("deepstack_merger_list.{index}");
                let joined = states.reshape(tokens, MERGE * MERGE * hidden)?;
                let normalized = w.norm(&joined, &format!("{name}.norm"), NORM_EPSILON)?;
                deepstack.push(self.merge(&name, &normalized)?);
            }
        }
        // The main merger norms each patch before joining them.
        let normalized = w
            .norm(&states, "merger.norm", NORM_EPSILON)?
            .reshape(tokens, MERGE * MERGE * hidden)?;
        let merged = self.merge("merger", &normalized)?;
        Ok(VisionEmbeddings {
            merged,
            deepstack,
            tokens,
        })
    }

    /// Full attention over the patches of one picture, a head at a time like on CUDA: the scores
    /// and the weighted values are matrix products, which take a fraction of the time `attention`
    /// takes over the thousands of patches of a picture, and the scores of a head are 65 MB at
    /// 1344 × 768.
    fn attend(&self, query: &Array, key: &Array, values: &Array) -> Result<Array> {
        let rows = query.shape()[0];
        let head = |x: &Array, head: usize| x.slice(0, rows, head * HEAD_DIM, HEAD_DIM);
        let heads = (0..self.heads)
            .map(|index| {
                head(query, index)?
                    .linear(&head(key, index)?)?
                    .softmax(1.0 / (HEAD_DIM as f32).sqrt())?
                    .linear(&head(values, index)?.transpose()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Array::concat(&heads, true)
    }

    /// fc2(GELU(fc1(rows))) over rows of 2 × 2 joined patches, with the exact GELU.
    fn merge(&self, name: &str, joined: &Array) -> Result<Array> {
        let w = &self.weights;
        let hidden = w
            .linear(joined, &format!("{name}.linear_fc1"))?
            .gelu(false)?;
        w.linear(&hidden, &format!("{name}.linear_fc2"))
    }
}
