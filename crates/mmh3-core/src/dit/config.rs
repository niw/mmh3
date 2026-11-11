/// DiT hyperparameters. The released models have hidden 5376, 56 heads of 128, FFN 14336 and 50
/// blocks. MiniMax H3 Image has no audio layers and an output head per sampling step.
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
    /// Audio latent channels, 0 for a DiT without audio layers.
    pub audio_channels: usize,
    /// The sigma each video output head was distilled for, in step order, when the checkpoint has
    /// one head per step. Empty for a single head.
    pub step_sigmas: Vec<f32>,
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

    pub fn has_audio(&self) -> bool {
        self.audio_channels > 0
    }

    /// The output head of a step at `sigma`: the one distilled for the nearest sigma, and 0 for a
    /// single head.
    pub fn output_head(&self, sigma: f32) -> usize {
        self.step_sigmas
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left - sigma).abs().total_cmp(&(*right - sigma).abs())
            })
            .map_or(0, |(head, _)| head)
    }

    /// The shape of a DiT read from its checkpoint header alone, with no weights uploaded
    /// anywhere. A leader that hands every step to its ranks and only puts the parts back
    /// together needs this and none of the 21 GB behind it.
    pub fn of(file: &crate::safetensors::SafeTensors, prefix: &str) -> Result<Self, String> {
        let mut config = Self::from_shapes(|name| {
            file.get(&format!("{prefix}{name}"))
                .map(|info| info.shape.clone())
        })?;
        let heads = Self::output_heads(&config, |name| {
            file.get(&format!("{prefix}{name}"))
                .map(|info| info.shape.clone())
        })?;
        if heads > 1 {
            let sigmas = step_sigmas(file.metadata()).ok_or(
                "a DiT with an output head per step needs the steps' sigmas in its metadata",
            )?;
            if sigmas.len() < heads {
                return Err(format!(
                    "{heads} output heads but {} sigmas in the metadata",
                    sigmas.len()
                ));
            }
            config.step_sigmas = sigmas[..heads].to_vec();
        }
        Ok(config)
    }

    /// How many video output heads `final_layer.video_out` stacks.
    fn output_heads(
        config: &DitConfig,
        shape_of: impl Fn(&str) -> Option<Vec<usize>>,
    ) -> Result<usize, String> {
        let rows = shape_of("final_layer.video_out.weight")
            .ok_or("missing tensor final_layer.video_out.weight")?[0];
        let features = config.video_patch_features();
        if rows == 0 || !rows.is_multiple_of(features) {
            return Err(format!(
                "final_layer.video_out has {rows} rows, not a multiple of {features}"
            ));
        }
        Ok(rows / features)
    }

    /// Whether this checkpoint has the VSA gate projections, which decides the regions a shared
    /// step exchanges through. Read from the header beside the shape.
    pub fn gated(file: &crate::safetensors::SafeTensors, prefix: &str) -> bool {
        file.get(&format!("{prefix}blocks.0.attn.to_gate_compress.weight"))
            .is_some()
    }

    /// Infers the configuration of a pruned checkpoint from tensor shapes, keyed by checkpoint
    /// tensor name.
    pub fn from_shapes(shape_of: impl Fn(&str) -> Option<Vec<usize>>) -> Result<Self, String> {
        let shape = |name: &str| shape_of(name).ok_or_else(|| format!("missing tensor {name}"));
        let count = |prefix: &str| {
            (0..)
                .take_while(|index| shape_of(&format!("{prefix}.{index}.norm1.weight")).is_some())
                .count()
        };

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
            audio_channels: shape_of("audio_patch_proj.weight").map_or(0, |shape| shape[1]),
            step_sigmas: Vec::new(),
            adaln_rank: table[1],
            adaln_grid: table[0],
            rope_frequencies: shape("rope.inv_freq")?[0],
            norm_eps: 1e-5,
        })
    }
}

/// The sampling sigmas of a DiT distilled with an output head per step, from the checkpoint's
/// `config` metadata: `{"transformer": {"pdd_sigmas": [...]}}`, ending at 0.
pub fn step_sigmas(metadata: &[(String, String)]) -> Option<Vec<f32>> {
    let (_, text) = metadata.iter().find(|(key, _)| key == "config")?;
    let config = crate::json::parse(text).ok()?;
    config
        .get("transformer")?
        .get("pdd_sigmas")?
        .as_array()?
        .iter()
        .map(|value| value.as_f64().map(|value| value as f32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_step_sigmas() {
        let metadata = vec![(
            "config".to_owned(),
            r#"{"transformer": {"pdd_steps": 3, "pdd_sigmas": [1.0, 0.75, 0.25, 0.0]}}"#.to_owned(),
        )];
        assert_eq!(step_sigmas(&metadata), Some(vec![1.0, 0.75, 0.25, 0.0]));
        assert_eq!(step_sigmas(&[]), None);
    }

    #[test]
    fn picks_the_head_of_the_nearest_sigma() {
        let mut config = DitConfig::from_shapes(|name| {
            Some(match name {
                "video_patch_proj.weight" => vec![8, 96],
                "blocks.0.attn.q_norm.weight" => vec![4],
                "blocks.0.attn.qkv_proj.weight" => vec![24, 8],
                "adaln_t_table" => vec![1025, 8],
                "blocks.0.mlp.fc2.weight" => vec![8, 16],
                "condition_proj.weight" => vec![8, 12],
                "rope.inv_freq" => vec![2],
                "audio_patch_proj.weight" => return None,
                _ if name.ends_with("norm1.weight") => return None,
                _ => vec![1],
            })
        })
        .unwrap();
        assert!(!config.has_audio());
        assert_eq!(config.output_head(0.3), 0);
        config.step_sigmas = vec![1.0, 0.75, 0.25];
        assert_eq!(config.output_head(1.0), 0);
        assert_eq!(config.output_head(0.7), 1);
        assert_eq!(config.output_head(0.2), 2);
    }
}
