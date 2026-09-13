/// DiT hyperparameters. The released model has hidden 5376, 56 heads of 128, FFN 14336 and 50 blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct DitConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub text_dim: usize,
    pub video_channels: usize,
    pub audio_channels: usize,
    /// Width of the time-embedding coordinates the AdaLN projections consume.
    pub adaln_rank: usize,
    /// Rows of the AdaLN curve table, covering t in [0, 1].
    pub adaln_grid: usize,
    /// RoPE frequencies per axis. Three axes rotate 6 × this many head dimensions.
    pub rope_frequencies: usize,
    pub norm_eps: f32,
}

/// Latent channels are patchified by 1 × 2 × 2 (time, height, width).
pub const PATCH_AREA: usize = 4;

impl DitConfig {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn video_patch_features(&self) -> usize {
        self.video_channels * PATCH_AREA
    }

    pub fn rope_dims(&self) -> usize {
        self.rope_frequencies * 6
    }

    /// Infers the configuration of a pruned checkpoint from tensor shapes, keyed by checkpoint tensor name.
    pub fn from_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<Self, String> {
        let shape = |name: &str| shape_of(name).ok_or_else(|| format!("missing tensor {name}"));
        let count = |prefix: &str| (0..).take_while(|index| shape_of(&format!("{prefix}.{index}.norm1.weight")).is_some()).count();

        let video_patch = shape("video_patch_proj.weight")?;
        let head_dim = shape("blocks.0.attn.q_norm.weight")?[0];
        let qkv = shape("blocks.0.attn.qkv_proj.weight")?;
        let table = shape("adaln_t_table")?;
        let video_features = video_patch[1];
        if video_features % PATCH_AREA != 0 || qkv[0] % (3 * head_dim) != 0 {
            return Err("unexpected projection shapes".to_owned());
        }
        Ok(DitConfig {
            hidden: video_patch[0],
            heads: qkv[0] / 3 / head_dim,
            head_dim,
            ffn: shape("blocks.0.mlp.fc2.weight")?[1],
            layers: count("blocks"),
            refiner_layers: count("token_refiner.blocks"),
            text_dim: shape("condition_proj.weight")?[1],
            video_channels: video_features / PATCH_AREA,
            audio_channels: shape("audio_patch_proj.weight")?[1],
            adaln_rank: table[1],
            adaln_grid: table[0],
            rope_frequencies: shape("rope.inv_freq")?[0],
            norm_eps: 1e-5,
        })
    }
}
