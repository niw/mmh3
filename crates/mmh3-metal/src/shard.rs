//! One rank's share of a DiT block on Metal.
//!
//! A block needs nothing from its peers for the normalizations, the projections and the MLP, since
//! every rank carries its own run of the sequence. Attention is the exception, because a query has
//! to see every key, so a block moves between "my tokens, every head" and "every token, my heads"
//! and back. These are the two gathers that do it.

use crate::{Buffer, Device, Error, Result, ops::Array};
pub use mmh3_core::shard::VelocityRows;

/// This backend's regions answer with memory a Metal buffer stands behind, and that is the only
/// word of a shard's context or its timing that is a backend's own.
pub type ShardContext<'a> = mmh3_core::shard::ShardContext<'a, Memory>;
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

    /// Rotates and quantizes `values` into the region at `offset` bytes in, and answers with the
    /// scale of every row beside them. This is how a rank publishes the rows it carries: the
    /// exchange moves a block's input as its INT8 layers consume it, an eighth of the bytes of
    /// what the projections would produce.
    pub fn write_quantized(&self, offset: usize, values: &Array) -> Result<Array> {
        let [rows, cols] = values.shape();
        let count = values.len();
        if offset.checked_add(count).is_none_or(|end| end > self.bytes) {
            return Err(Error(format!(
                "{count} bytes at {offset} of a region of {}",
                self.bytes
            )));
        }

        let rotated = values.rotate()?;
        let scales = Array::empty(&self.device, rows, 1)?;
        let packed = self.device.alloc(count, None)?;
        self.device.run(
            "pack_linear_input",
            &[&rotated.buffer, &packed, &scales.buffer],
            &[cols as u32, 1],
            rows,
            true,
        )?;
        self.device.run(
            "copy_bytes",
            &[&packed, &self.buffer],
            &[count as u32, 0, offset as u32],
            count,
            false,
        )?;
        Ok(scales)
    }

    /// Writes `values` as FP32 at `offset` bytes in, which is how the scale of every row travels
    /// beside the rows themselves.
    pub fn write_f32(&self, offset: usize, values: &Array) -> Result<()> {
        let bytes = values.len() * 4;
        self.check_bytes(offset, bytes)?;
        self.device.run(
            "copy_bytes",
            &[&values.buffer, &self.buffer],
            &[bytes as u32, 0, offset as u32],
            bytes,
            false,
        )
    }

    /// The `rows` by `cols` FP32 array at `offset` bytes in.
    pub fn read_f32(&self, offset: usize, rows: usize, cols: usize) -> Result<Array> {
        let out = Array::empty(&self.device, rows, cols)?;
        let bytes = out.len() * 4;
        self.check_bytes(offset, bytes)?;
        self.device.run(
            "copy_bytes",
            &[&self.buffer, &out.buffer],
            &[bytes as u32, offset as u32, 0],
            bytes,
            false,
        )?;
        Ok(out)
    }

    /// A region of `bytes` zeroed bytes, which is what a transport hands a rank.
    pub fn zeroed(device: &Device, bytes: usize) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            buffer: device.alloc(bytes, Some(&vec![0u8; bytes]))?,
            bytes,
        })
    }

    /// Moves `bytes` from `source`, `offset` bytes in, to `into` bytes into this one, which is
    /// what a write to a peer does once the bytes are on this machine.
    pub fn copy_from(&self, source: &Self, offset: usize, into: usize, bytes: usize) -> Result<()> {
        source.check_bytes(offset, bytes)?;
        self.check_bytes(into, bytes)?;
        self.device.run(
            "copy_bytes",
            &[&source.buffer, &self.buffer],
            &[bytes as u32, offset as u32, into as u32],
            bytes,
            false,
        )
    }

    /// Reads `into.len()` bytes from `offset` bytes in, which is how what a block computed
    /// reaches the host memory a peer reads.
    pub fn read_bytes(&self, offset: usize, into: &mut [u8]) -> Result<()> {
        self.check_bytes(offset, into.len())?;
        let part = Self::zeroed(&self.device, into.len())?;
        part.copy_from(self, offset, 0, into.len())?;
        into.copy_from_slice(&part.buffer.to_bytes()?);
        Ok(())
    }

    /// Writes `from` at `offset` bytes in, which is how what a peer wrote reaches the memory a
    /// block computes in.
    pub fn write_bytes(&self, offset: usize, from: &[u8]) -> Result<()> {
        self.check_bytes(offset, from.len())?;
        let part = Self {
            device: self.device.clone(),
            buffer: self.device.alloc(from.len(), Some(from))?,
            bytes: from.len(),
        };
        self.copy_from(&part, 0, offset, from.len())
    }

    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// An offset into a region is a byte offset, and a bf16 value straddling the end of one would
    /// read a neighbour's memory rather than fail, so the extent is checked before the kernel.
    fn check(&self, offset: usize, bytes: usize) -> Result<()> {
        if offset % 2 != 0 {
            return Err(Error(format!("a bf16 run at an odd offset {offset}")));
        }
        self.check_bytes(offset, bytes)
    }

    fn check_bytes(&self, offset: usize, bytes: usize) -> Result<()> {
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

    /// A rank publishes its rows by rotating and quantizing them into the region its peers read.
    /// Projecting them back out has to give what projecting them here would have given: on the
    /// INT8 road that is the same arithmetic on the same bytes, so it is an equality and not a
    /// tolerance. The exchange may move the rows; it may not change what they mean.
    #[test]
    fn a_published_row_projects_to_what_it_would_have_here() {
        use crate::LinearPrecision;
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            eprintln!("no tensor ops on this device, skipping");
            return;
        }

        // ConvRot wants a multiple of 256 features, which is what a block's width is.
        let (rows, cols, outputs) = (4usize, 256usize, 64usize);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 29) % 197) as f32 / 197.0 - 0.5)
            .collect();
        let x = Array::from_f32(&device, rows, cols, &values).unwrap();
        let weight_bytes: Vec<u8> = (0..outputs * cols)
            .map(|i| (((i * 53) % 255) as i32 - 127) as i8 as u8)
            .collect();
        let weight = device
            .alloc(weight_bytes.len(), Some(&weight_bytes))
            .unwrap();

        let mut exchange = WholeExchange::new(&device);
        let region = exchange.region(Region::Normalized, rows * cols).unwrap();
        let scales = region.write_quantized(0, &x).unwrap();

        let there = Array::linear_quantized(
            &device,
            region.buffer(),
            &scales,
            &weight,
            rows,
            cols,
            outputs,
            LinearPrecision::Int8,
        )
        .unwrap()
        .to_f32()
        .unwrap();
        let here = x
            .rotate()
            .unwrap()
            .linear_int8(&weight, outputs, LinearPrecision::Int8)
            .unwrap()
            .to_f32()
            .unwrap();

        assert_eq!(there.len(), here.len());
        for (index, (&found, &want)) in there.iter().zip(here.iter()).enumerate() {
            assert_eq!(found.to_bits(), want.to_bits(), "value {index}");
        }
    }

    /// Every rank writes its own rows into the same region, so a write must not disturb the rows a
    /// peer already put there.
    #[test]
    fn publishing_rows_leaves_a_peers_rows_alone() {
        use mmh3_core::shard::Exchange;

        let device = Device::new().unwrap();
        let (rows, cols) = (4usize, 256usize);
        let first = Array::from_f32(&device, rows, cols, &vec![0.25; rows * cols]).unwrap();
        let second = Array::from_f32(&device, rows, cols, &vec![-0.5; rows * cols]).unwrap();

        let mut exchange = WholeExchange::new(&device);
        let region = exchange
            .region(Region::Normalized, 2 * rows * cols)
            .unwrap();
        region.write_quantized(0, &first).unwrap();
        let before = region.buffer().to_f32().unwrap();
        region.write_quantized(rows * cols, &second).unwrap();
        let after = region.buffer().to_f32().unwrap();

        let untouched = rows * cols / 4;
        assert_eq!(
            before[..untouched],
            after[..untouched],
            "the first rank's rows moved"
        );
        assert_ne!(
            before[untouched..],
            after[untouched..],
            "the second rank's rows did not land"
        );
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

/// Two ranks in one process, sharing this machine's memory instead of a wire.
///
/// It exists to run the branches a single rank never enters: the writes to a peer, the
/// `Received(peer)` regions and the offsets each rank addresses the others' rows by. Every region
/// is kept per rank, as two machines would keep them, and a write copies between two ranks' own
/// memory rather than reaching into one shared buffer, so an offset that is wrong is still wrong
/// here.
///
/// What it cannot model is time. One thread runs one rank at a time, so a barrier cannot wait for
/// a rank that has not been called yet; the caller sequences the ranks instead, and the barrier is
/// where that sequencing has to hold. A transport that deadlocks on the wire will not deadlock
/// here, which is the one class of fault this does not cover.
#[cfg(test)]
pub(crate) struct PairExchange {
    device: Device,
    rank: usize,
    ranks: usize,
    regions: std::rc::Rc<std::cell::RefCell<HashMap<(usize, Region), Memory>>>,
}

#[cfg(test)]
impl PairExchange {
    pub fn pair(device: &Device, ranks: usize) -> Vec<Self> {
        let regions = std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()));
        (0..ranks)
            .map(|rank| Self {
                device: device.clone(),
                rank,
                ranks,
                regions: regions.clone(),
            })
            .collect()
    }

    fn held(&self, rank: usize, region: Region, bytes: usize) -> Result<Memory> {
        let mut regions = self.regions.borrow_mut();
        let entry = regions.entry((rank, region));
        use std::collections::hash_map::Entry;
        Ok(match entry {
            Entry::Occupied(held) if held.get().bytes() >= bytes => held.get().clone(),
            Entry::Occupied(mut held) => {
                let made = Memory::zeroed(&self.device, bytes)?;
                held.insert(made.clone());
                made
            }
            Entry::Vacant(empty) => empty.insert(Memory::zeroed(&self.device, bytes)?).clone(),
        })
    }
}

#[cfg(test)]
impl mmh3_core::shard::Exchange for PairExchange {
    type Memory = Memory;

    fn rank(&self) -> usize {
        self.rank
    }

    fn ranks(&self) -> usize {
        self.ranks
    }

    fn region(
        &mut self,
        region: Region,
        bytes: usize,
    ) -> std::result::Result<Memory, ExchangeError> {
        self.held(self.rank, region, bytes)
            .map_err(|error| ExchangeError(error.0))
    }

    fn write(
        &mut self,
        peer: usize,
        from: Region,
        offset: usize,
        into: Region,
        peer_offset: usize,
        bytes: usize,
    ) -> std::result::Result<(), ExchangeError> {
        let fail = |error: Error| ExchangeError(error.0);
        let source = self.held(self.rank, from, offset + bytes).map_err(fail)?;
        let destination = self.held(peer, into, peer_offset + bytes).map_err(fail)?;
        destination
            .copy_from(&source, offset, peer_offset, bytes)
            .map_err(fail)
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
