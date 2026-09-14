//! Deterministic standard normal noise for the initial latents.

/// xoshiro256** seeded through SplitMix64, turned into normal samples with the Box–Muller
/// transform.
pub struct NormalSampler {
    state: [u64; 4],
    spare: Option<f64>,
}

impl NormalSampler {
    pub fn new(seed: u64) -> Self {
        let mut mixer = seed;
        let mut next_mixed = || {
            mixer = mixer.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = mixer;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        };
        NormalSampler {
            state: [next_mixed(), next_mixed(), next_mixed(), next_mixed()],
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let shifted = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= shifted;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1].
    fn next_uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare as f32;
        }
        let radius = (-2.0 * self.next_uniform().ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.next_uniform();
        self.spare = Some(radius * angle.sin());
        (radius * angle.cos()) as f32
    }

    pub fn samples(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.sample()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_standard_normal_samples() {
        let values = NormalSampler::new(7).samples(200_000);
        let mean = values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|&value| (value as f64 - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(
            mean.abs() < 0.01 && (variance - 1.0).abs() < 0.01,
            "mean {mean}, variance {variance}"
        );
        assert_eq!(NormalSampler::new(7).samples(4), values[..4]);
        assert_ne!(NormalSampler::new(8).samples(4), values[..4]);
    }
}
