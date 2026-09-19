//! One rank's share of a DiT block on Metal.
//!
//! A block needs nothing from its peers for the normalizations, the projections and the MLP, since
//! every rank carries its own run of the sequence. Attention is the exception, because a query has
//! to see every key, so a block moves between "my tokens, every head" and "every token, my heads"
//! and back. These are the two gathers that do it.

use crate::{Buffer, Device, Error, Result, ops::Array};
use mmh3_core::shard::{ExchangeError, Region};
use std::collections::HashMap;
use std::ops::Range;

/// Memory a Metal rank exchanges through. A region is named in bytes and holds INT8 rows, FP32
/// scales or bf16 attention tensors depending on which one it is, so this is bytes rather than an
/// `Array` of one element width.
#[derive(Clone)]
pub struct Memory {
    device: Device,
    buffer: Buffer,
    bytes: usize,
}

impl Memory {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Writes `values` as bf16 at `offset` bytes in, the width the exchange carries a block's
    /// attention tensors in.
    pub fn write_bf16(&self, offset: usize, values: &Array) -> Result<()> {
        let count = values.len();
        self.check(offset, count * 2)?;
        self.device.run(
            "to_bf16",
            &[&values.buffer, &self.buffer],
            &[count as u32, (offset / 2) as u32],
            count,
            false,
        )
    }

    /// The `rows` by `cols` FP32 array of the bf16 values at `offset` bytes in.
    pub fn read_bf16(&self, offset: usize, rows: usize, cols: usize) -> Result<Array> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| Error("a read that overflows".into()))?;
        self.check(offset, count * 2)?;
        let values = Array::empty(&self.device, rows, cols)?;
        self.device.run(
            "from_bf16",
            &[&self.buffer, &values.buffer],
            &[count as u32, (offset / 2) as u32],
            count,
            false,
        )?;
        Ok(values)
    }

    /// An offset into a region is a byte offset, and a bf16 value straddling the end of one would
    /// read a neighbour's memory rather than fail, so the extent is checked before the kernel.
    fn check(&self, offset: usize, bytes: usize) -> Result<()> {
        if offset % 2 != 0 {
            return Err(Error(format!("a bf16 run at an odd offset {offset}")));
        }
        if offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(Error(format!(
                "{bytes} bytes at {offset} of a region of {}",
                self.bytes
            )));
        }
        Ok(())
    }
}

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

/// The exchange of a step that is not shared out after all: one rank, regions in this machine's
/// own memory, and nothing to carry anywhere. A step through it has to come out the same as a step
/// that never went near a shard, which is how a rank's own path is checked without a second
/// machine.
pub struct WholeExchange {
    device: Device,
    regions: HashMap<Region, Memory>,
}

impl WholeExchange {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            regions: HashMap::new(),
        }
    }
}

impl mmh3_core::shard::Exchange for WholeExchange {
    type Memory = Memory;

    fn rank(&self) -> usize {
        0
    }

    fn ranks(&self) -> usize {
        1
    }

    fn region(
        &mut self,
        region: Region,
        bytes: usize,
    ) -> std::result::Result<Memory, ExchangeError> {
        if self.regions.get(&region).map_or(0, Memory::bytes) < bytes {
            let zeros = vec![0u8; bytes];
            let buffer = self
                .device
                .alloc(bytes, Some(&zeros))
                .map_err(|error| ExchangeError(error.to_string()))?;
            self.regions.insert(
                region,
                Memory {
                    device: self.device.clone(),
                    buffer,
                    bytes,
                },
            );
        }
        Ok(self.regions[&region].clone())
    }

    fn write(
        &mut self,
        peer: usize,
        _from: Region,
        _offset: usize,
        _into: Region,
        _peer_offset: usize,
        _bytes: usize,
    ) -> std::result::Result<(), ExchangeError> {
        Err(ExchangeError(format!(
            "one rank cannot write to rank {peer}"
        )))
    }

    fn publish(
        &mut self,
        _region: Region,
        _offset: usize,
        _bytes: usize,
    ) -> std::result::Result<(), ExchangeError> {
        Ok(())
    }

    fn receive(
        &mut self,
        _region: Region,
        _offset: usize,
        _bytes: usize,
    ) -> std::result::Result<(), ExchangeError> {
        Ok(())
    }

    fn barrier(&mut self) -> std::result::Result<(), ExchangeError> {
        Ok(())
    }
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

    /// The exchange carries bf16, so what goes out and comes back is what the host conversion
    /// gives, not merely something close to it.
    #[test]
    fn bf16_matches_the_conversion_on_the_host() {
        use mmh3_core::numeric::{bf16_to_f32, f32_to_bf16};
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        let values: Vec<f32> = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.0 / 3.0,
            -2.718_281_8,
            1e-40,
            3.4e38,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]
        .into_iter()
        .chain((0..117).map(|i| (i as f32 - 58.0) / 7.0))
        .collect();

        let rows = values.len();
        let array = Array::from_f32(&device, rows, 1, &values).unwrap();
        let mut exchange = WholeExchange::new(&device);
        let memory = exchange.region(Region::Attended, rows * 2).unwrap();
        memory.write_bf16(0, &array).unwrap();

        let found = memory.read_bf16(0, rows, 1).unwrap().to_f32().unwrap();
        for (index, &value) in values.iter().enumerate() {
            let expected = bf16_to_f32(f32_to_bf16(value));
            assert_eq!(
                found[index].to_bits(),
                expected.to_bits(),
                "value {index}, {value}"
            );
        }
    }

    /// A NaN stays a NaN rather than turning into an infinity, which a naive truncation does.
    #[test]
    fn bf16_keeps_a_nan_a_nan() {
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        let array = Array::from_f32(&device, 1, 1, &[f32::NAN]).unwrap();
        let mut exchange = WholeExchange::new(&device);
        let memory = exchange.region(Region::Attended, 2).unwrap();
        memory.write_bf16(0, &array).unwrap();
        assert!(memory.read_bf16(0, 1, 1).unwrap().to_f32().unwrap()[0].is_nan());
    }

    /// Regions are addressed by byte offset, so a run that would reach past one is refused rather
    /// than reading whatever is next to it.
    #[test]
    fn a_run_past_the_end_of_a_region_is_refused() {
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        let mut exchange = WholeExchange::new(&device);
        let memory = exchange.region(Region::Attended, 16).unwrap();
        assert!(memory.bytes() >= 16);
        let array = Array::from_f32(&device, 8, 1, &[1.0; 8]).unwrap();
        assert!(memory.write_bf16(2, &array).is_err());
        assert!(memory.read_bf16(2, 8, 1).is_err());
        assert!(memory.read_bf16(1, 4, 1).is_err());
        assert!(memory.write_bf16(0, &array).is_ok());
    }

    /// One rank keeps its regions and has nobody to write to, which is what makes it the exchange a
    /// run uses to check the sharded path against itself.
    #[test]
    fn one_rank_keeps_its_regions_and_writes_to_nobody() {
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        let mut exchange = WholeExchange::new(&device);
        assert_eq!(exchange.rank(), 0);
        assert_eq!(exchange.ranks(), 1);

        let first = exchange.region(Region::Normalized, 64).unwrap();
        let again = exchange.region(Region::Normalized, 32).unwrap();
        assert_eq!(
            first.bytes(),
            again.bytes(),
            "a smaller ask keeps the region"
        );

        let grown = exchange.region(Region::Normalized, 256).unwrap();
        assert_eq!(grown.bytes(), 256);

        assert!(
            exchange
                .write(1, Region::Normalized, 0, Region::Normalized, 0, 8)
                .is_err()
        );
        assert!(exchange.publish(Region::Normalized, 0, 8).is_ok());
        assert!(exchange.receive(Region::Normalized, 0, 8).is_ok());
        assert!(exchange.barrier().is_ok());
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
