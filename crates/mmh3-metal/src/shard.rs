//! One rank's share of a DiT block on Metal.
//!
//! A block needs nothing from its peers for the normalizations, the projections and the MLP, since
//! every rank carries its own run of the sequence. Attention is the exception, because a query has
//! to see every key, so a block moves between "my tokens, every head" and "every token, my heads"
//! and back. These are the two gathers that do it.

use crate::{Error, Result, ops::Array};
use std::ops::Range;

/// Gathers the heads `own_heads` out of `[tokens][tensors][heads][dim]` into
/// `[tokens][tensors][own heads][dim]`, which is the shape the attention already reads. q, k and v
/// go together as three tensors, and VSA's compressed query follows as one.
pub fn pack(
    source: &Array,
    tensors: usize,
    heads: usize,
    own_heads: Range<usize>,
    dim: usize,
) -> Result<Array> {
    let [tokens, columns] = source.shape();
    let span = check(tensors, heads, &own_heads, dim)?;
    if columns != tensors * heads * dim {
        return Err(Error(format!(
            "a source of {columns} columns, not {}",
            tensors * heads * dim
        )));
    }

    let packed = Array::empty(source.device(), tokens, tensors * span * dim)?;
    source.device().run(
        "shard_pack",
        &[&source.buffer, &packed.buffer],
        &[
            packed.len() as u32,
            dim as u32,
            span as u32,
            tensors as u32,
            heads as u32,
            own_heads.start as u32,
        ],
        packed.len(),
        false,
    )?;
    Ok(packed)
}

/// Scatters one rank's share of an attention output, `[tokens][own heads][dim]`, into the rows this
/// rank carries onward, `[tokens][heads][dim]`, at `own_heads`. The heads a peer owns are left as
/// they are, so a rank's own share and the shares that arrive can be written in any order.
pub fn unpack(
    part: &Array,
    output: &Array,
    heads: usize,
    own_heads: Range<usize>,
    dim: usize,
) -> Result<()> {
    let span = check(1, heads, &own_heads, dim)?;
    let ([tokens, columns], [rows, width]) = (part.shape(), output.shape());
    if columns != span * dim {
        return Err(Error(format!(
            "a share of {columns} columns, not {}",
            span * dim
        )));
    }
    if rows != tokens || width != heads * dim {
        return Err(Error(format!(
            "an output of [{rows}, {width}], not [{tokens}, {}]",
            heads * dim
        )));
    }

    output.device().run(
        "shard_unpack",
        &[&part.buffer, &output.buffer],
        &[
            part.len() as u32,
            dim as u32,
            span as u32,
            heads as u32,
            own_heads.start as u32,
        ],
        part.len(),
        false,
    )
}

/// The width of a rank's share of the heads, once the split is known to be one.
fn check(tensors: usize, heads: usize, own_heads: &Range<usize>, dim: usize) -> Result<usize> {
    if tensors == 0 || heads == 0 || dim == 0 {
        return Err(Error("a share of no tensors, heads or dimensions".into()));
    }
    if own_heads.start >= own_heads.end || own_heads.end > heads {
        return Err(Error(format!(
            "heads {}..{} of {heads}",
            own_heads.start, own_heads.end
        )));
    }
    Ok(own_heads.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    const TOKENS: usize = 5;
    const HEADS: usize = 6;
    const DIM: usize = 4;
    const TENSORS: usize = 3;

    fn ramp(count: usize) -> Vec<f32> {
        (0..count).map(|i| i as f32).collect()
    }

    fn source(device: &Device) -> Array {
        let columns = TENSORS * HEADS * DIM;
        Array::from_f32(device, TOKENS, columns, &ramp(TOKENS * columns)).unwrap()
    }

    #[test]
    fn pack_gathers_the_heads_a_rank_owns() {
        let device = Device::new().unwrap();
        let source = source(&device);
        let values = source.to_f32().unwrap();
        let own = 2..5;

        let packed = pack(&source, TENSORS, HEADS, own.clone(), DIM).unwrap();
        assert_eq!(packed.shape(), [TOKENS, TENSORS * own.len() * DIM]);

        let found = packed.to_f32().unwrap();
        for token in 0..TOKENS {
            for tensor in 0..TENSORS {
                for (index, head) in own.clone().enumerate() {
                    for lane in 0..DIM {
                        let from = ((token * TENSORS + tensor) * HEADS + head) * DIM + lane;
                        let into = ((token * TENSORS + tensor) * own.len() + index) * DIM + lane;
                        assert_eq!(found[into], values[from], "token {token} head {head}");
                    }
                }
            }
        }
    }

    /// A rank that owns every head gathers the rows it was already given.
    #[test]
    fn pack_of_every_head_changes_nothing() {
        let device = Device::new().unwrap();
        let source = source(&device);
        let packed = pack(&source, TENSORS, HEADS, 0..HEADS, DIM).unwrap();
        assert_eq!(packed.to_f32().unwrap(), source.to_f32().unwrap());
    }

    #[test]
    fn unpack_scatters_a_share_and_leaves_the_rest_alone() {
        let device = Device::new().unwrap();
        let own = 1..3;
        let part = Array::from_f32(
            &device,
            TOKENS,
            own.len() * DIM,
            &ramp(TOKENS * own.len() * DIM),
        )
        .unwrap();
        let output = Array::zeros(&device, TOKENS, HEADS * DIM).unwrap();

        unpack(&part, &output, HEADS, own.clone(), DIM).unwrap();

        let (share, found) = (part.to_f32().unwrap(), output.to_f32().unwrap());
        for token in 0..TOKENS {
            for head in 0..HEADS {
                for lane in 0..DIM {
                    let at = (token * HEADS + head) * DIM + lane;
                    if own.contains(&head) {
                        let from = (token * own.len() + head - own.start) * DIM + lane;
                        assert_eq!(found[at], share[from], "token {token} head {head}");
                    } else {
                        assert_eq!(found[at], 0.0, "head {head} is not this rank's");
                    }
                }
            }
        }
    }

    /// Every rank's share scattered into one output rebuilds the rows a whole block carries, which
    /// is what the second exchange of a block has to come out as.
    #[test]
    fn the_shares_of_every_rank_rebuild_the_whole_rows() {
        let device = Device::new().unwrap();
        let whole =
            Array::from_f32(&device, TOKENS, HEADS * DIM, &ramp(TOKENS * HEADS * DIM)).unwrap();
        let output = Array::zeros(&device, TOKENS, HEADS * DIM).unwrap();

        for own in [0..1, 1..4, 4..HEADS] {
            let part = pack(&whole, 1, HEADS, own.clone(), DIM).unwrap();
            unpack(&part, &output, HEADS, own, DIM).unwrap();
        }

        assert_eq!(output.to_f32().unwrap(), whole.to_f32().unwrap());
    }

    #[test]
    fn a_share_outside_the_heads_is_refused() {
        let device = Device::new().unwrap();
        let source = source(&device);
        assert!(pack(&source, TENSORS, HEADS, 4..HEADS + 1, DIM).is_err());
        assert!(pack(&source, TENSORS, HEADS, 3..3, DIM).is_err());
        assert!(pack(&source, TENSORS, HEADS + 1, 0..HEADS, DIM).is_err());
    }
}
