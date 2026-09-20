//! Splitting one DiT step across machines, the way Ulysses does it.
//!
//! Every rank carries a contiguous run of the packed sequence through the blocks, which needs no
//! communication at all for the normalizations, the projections and the MLP. Attention is the
//! exception, since a query has to see every key. So each block exchanges twice: once to turn
//! "my tokens, every head" into "every token, my heads", and once to turn the result back.
//!
//! The exchange moves `tokens × heads × 128` values in bf16 either way. At 768p with two machines
//! that is 28 GB a step, which the link between two Sparks carries in about 3 s while it saves
//! about 7 s of arithmetic.

use crate::{CudaError, DeviceBuffer, check};
/// The split, the regions and the transport are the same words on every backend, so they live in
/// mmh3-core and this module is the CUDA side of them: the gathers, the timing and the exchange a
/// run uses when it shares a step with nobody.
pub use mmh3_core::shard::{
    Exchange, ExchangeError, HEAD_DIM, Region, Shard, VelocityRows, regions,
};
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr;
use std::time::Duration;

unsafe extern "C" {
    fn mmh3_dit_shard_pack(
        source: *const c_void,
        packed: *mut c_void,
        tokens: c_int,
        tensors: c_int,
        heads: c_int,
        dim: c_int,
        first_head: c_int,
        span: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_dit_shard_unpack(
        part: *const c_void,
        output: *mut c_void,
        tokens: c_int,
        heads: c_int,
        dim: c_int,
        first_head: c_int,
        span: c_int,
        stream: *mut c_void,
    ) -> c_int;
}

/// Where a shared-out step spent its time, so that the split can be judged against the arithmetic
/// it saves rather than against a guess. The caller clears it between steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShardTiming {
    /// Gathering a block's inputs and scattering its output, and the wait for the device that
    /// follows each, since a peer may not read memory a kernel has not finished writing.
    pub gather: Duration,
    /// Waiting for the peers to reach the same point.
    pub barrier: Duration,
    /// Reading the peers' memory.
    pub read: Duration,
    /// Attention itself, over the whole sequence for this rank's heads.
    pub attend: Duration,
}

impl ShardTiming {
    pub fn total(&self) -> Duration {
        self.gather + self.barrier + self.read + self.attend
    }
}

/// One rank's share of a step and the transport its blocks exchange over.
pub struct ShardContext<'a> {
    pub shard: Shard,
    pub exchange: &'a mut dyn Exchange<Memory = *mut c_void>,
    pub timing: ShardTiming,
}

/// The exchange of a step that is not shared out after all: one rank, regions in this machine's own
/// memory, and nothing to carry anywhere. A step through it has to come out the same as a step that
/// never went near a shard, which is how the gathers and the exchanged layout are checked without a
/// second machine.
#[derive(Default)]
pub struct WholeExchange {
    regions: HashMap<Region, DeviceBuffer>,
}

impl WholeExchange {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Exchange for WholeExchange {
    type Memory = *mut c_void;

    fn rank(&self) -> usize {
        0
    }

    fn ranks(&self) -> usize {
        1
    }

    fn region(&mut self, region: Region, bytes: usize) -> Result<*mut c_void, ExchangeError> {
        if self.regions.get(&region).map_or(0, DeviceBuffer::bytes) < bytes {
            let held = DeviceBuffer::zeroed(bytes)
                .map_err(|error| ExchangeError(format!("making {region:?}: {error}")))?;
            self.regions.insert(region, held);
        }
        Ok(self.regions[&region].pointer())
    }

    fn write(
        &mut self,
        peer: usize,
        _from: Region,
        _offset: usize,
        _into: Region,
        _peer_offset: usize,
        _bytes: usize,
    ) -> Result<(), ExchangeError> {
        Err(ExchangeError(format!(
            "one rank cannot write to rank {peer}"
        )))
    }

    fn publish(
        &mut self,
        _region: Region,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), ExchangeError> {
        Ok(())
    }

    fn receive(
        &mut self,
        _region: Region,
        _offset: usize,
        _bytes: usize,
    ) -> Result<(), ExchangeError> {
        Ok(())
    }

    fn barrier(&mut self) -> Result<(), ExchangeError> {
        Ok(())
    }
}

/// Gathers heads `span` out of `[tokens][tensors][heads][dim]` into `[tokens][tensors][span][dim]`,
/// which is the shape the attention already reads. q, k and v go together as three tensors, and
/// VSA's compressed query follows as one.
///
/// # Safety
///
/// `source` holds `tokens × tensors × heads × dim` values from its pointer and `packed` holds the
/// gathered ones.
pub unsafe fn pack(
    source: *const c_void,
    packed: *mut c_void,
    tokens: usize,
    tensors: usize,
    heads: usize,
    dim: usize,
    span: Range<usize>,
) -> Result<(), CudaError> {
    check(unsafe {
        mmh3_dit_shard_pack(
            source,
            packed,
            tokens as c_int,
            tensors as c_int,
            heads as c_int,
            dim as c_int,
            span.start as c_int,
            span.len() as c_int,
            ptr::null_mut(),
        )
    })
}

/// Scatters one rank's share of an attention output, `[tokens][span][dim]`, into the rows this rank
/// carries onward, `[tokens][heads][dim]`.
///
/// # Safety
///
/// `part` holds `tokens × span × dim` values and `output` holds `tokens × heads × dim`.
pub unsafe fn unpack(
    part: *const c_void,
    output: *mut c_void,
    tokens: usize,
    heads: usize,
    dim: usize,
    span: Range<usize>,
) -> Result<(), CudaError> {
    check(unsafe {
        mmh3_dit_shard_unpack(
            part,
            output,
            tokens as c_int,
            heads as c_int,
            dim as c_int,
            span.start as c_int,
            span.len() as c_int,
            ptr::null_mut(),
        )
    })
}

/// Tensors one exchange of a block's inputs carries: q, k and v, and VSA's compressed query when
/// the block has one.
pub fn input_tensors(gated: bool) -> usize {
    if gated { 4 } else { 3 }
}

/// Packs a head span and scatters a part back, then reads both results, since a step cannot be
/// trusted to a gather nobody has checked. The kernels only move values, so the check works in
/// bf16 bit patterns and never converts, which keeps a rounding difference from looking like a
/// misplaced value.
pub fn check_layout(tokens: usize, heads: usize, dim: usize) -> Result<(), CudaError> {
    // A step of 40503 walks the whole 16-bit range without repeating, so neighbouring values differ
    // in every position the gathers could confuse.
    let value = |index: usize| (index.wrapping_mul(40503) & 0xffff) as u16;
    let bytes = |count: usize| -> Vec<u8> {
        (0..count)
            .flat_map(|index| value(index).to_le_bytes())
            .collect()
    };
    let read = |buffer: &DeviceBuffer| -> Result<Vec<u16>, CudaError> {
        let mut raw = vec![0u8; buffer.bytes()];
        buffer.copy_to_host(&mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    };
    let mismatch = |what: String, found: u16, wanted: u16| CudaError {
        code: 0,
        message: format!("{what} came back {found:#06x}, not {wanted:#06x}"),
    };

    let span = heads / 2..heads;
    let qkv = DeviceBuffer::from_bytes(&bytes(tokens * 3 * heads * dim))?;
    let packed = DeviceBuffer::new(tokens * 3 * span.len() * dim * 2)?;
    // SAFETY: qkv holds the values its shape describes and packed holds the gathered ones.
    unsafe {
        pack(
            qkv.pointer(),
            packed.pointer(),
            tokens,
            3,
            heads,
            dim,
            span.clone(),
        )?
    };
    let gathered = read(&packed)?;
    for token in 0..tokens {
        for tensor in 0..3 {
            for head in span.clone() {
                for lane in [0, dim / 2, dim - 1] {
                    let wanted = value(((token * 3 + tensor) * heads + head) * dim + lane);
                    let at = ((token * 3 + tensor) * span.len() + head - span.start) * dim + lane;
                    if gathered[at] != wanted {
                        let what = format!("packing token {token} tensor {tensor} head {head}");
                        return Err(mismatch(what, gathered[at], wanted));
                    }
                }
            }
        }
    }

    let part = DeviceBuffer::from_bytes(&bytes(tokens * span.len() * dim))?;
    let output = DeviceBuffer::zeroed(tokens * heads * dim * 2)?;
    // SAFETY: part holds one rank's share of an output and output holds every head's.
    unsafe {
        unpack(
            part.pointer(),
            output.pointer(),
            tokens,
            heads,
            dim,
            span.clone(),
        )?
    };
    let scattered = read(&output)?;
    for token in 0..tokens {
        for head in 0..heads {
            for lane in [0, dim / 2, dim - 1] {
                // Heads outside the span belong to another rank and must stay untouched.
                let wanted = if span.contains(&head) {
                    value((token * span.len() + head - span.start) * dim + lane)
                } else {
                    0
                };
                let at = (token * heads + head) * dim + lane;
                if scattered[at] != wanted {
                    let what = format!("scattering token {token} head {head} lane {lane}");
                    return Err(mismatch(what, scattered[at], wanted));
                }
            }
        }
    }
    Ok(())
}
