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

use crate::attention::HEAD_DIM;
use crate::model::Error;
use crate::{CudaError, DeviceBuffer, check};
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

/// Which tokens and which heads each rank owns. Both cuts are contiguous, so a rank addresses its
/// share with an offset and a length rather than a table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shard {
    pub rank: usize,
    pub tokens: Vec<Range<usize>>,
    pub heads: Vec<Range<usize>>,
}

impl Shard {
    /// Cuts `tokens` and `heads` into `ranks` shares. Tokens divide on a multiple of `alignment`,
    /// which VSA needs so that a rank owns whole tiles, and the remainder of either cut goes to the
    /// last rank.
    pub fn even(rank: usize, ranks: usize, tokens: usize, heads: usize, alignment: usize) -> Self {
        let cut = |total: usize, step: usize| -> Vec<Range<usize>> {
            let each = (total / ranks / step.max(1)) * step.max(1);
            (0..ranks)
                .map(|index| {
                    let start = index * each;
                    let end = if index + 1 == ranks {
                        total
                    } else {
                        start + each
                    };
                    start..end
                })
                .collect()
        };
        Shard {
            rank,
            tokens: cut(tokens, alignment),
            heads: cut(heads, 1),
        }
    }

    pub fn ranks(&self) -> usize {
        self.tokens.len()
    }

    /// The tokens this rank carries through the blocks.
    pub fn own_tokens(&self) -> Range<usize> {
        self.tokens[self.rank].clone()
    }

    /// The heads this rank attends to once a block's inputs have been exchanged.
    pub fn own_heads(&self) -> Range<usize> {
        self.heads[self.rank].clone()
    }

    /// Whether the split is a split at all. One rank runs the sharded path unchanged and never
    /// touches an exchange, which is how the path is checked against a whole run.
    pub fn is_whole(&self) -> bool {
        self.ranks() == 1
    }
}

/// Memory both ranks name the same way. Every rank makes the same regions, so a peer can work out
/// what to read and from where out of the shard alone, and nothing but the addresses crosses the
/// wire when a session opens.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Region {
    /// `[all tokens][hidden]` INT8, the block's input as the projections consume it. Exchanging
    /// this rather than what the projections produce carries an eighth of the bytes and changes
    /// nothing: a rank runs its own heads' columns over the whole sequence instead of every
    /// column over its own rows.
    Normalized,
    /// `[all tokens]` FP32, one scale a row beside `Normalized`.
    Scales,
    /// `[all tokens][q, k, v][own heads][128]`, this rank's attention inputs.
    Inputs,
    /// `[all tokens][own heads][128]`, VSA's compressed query beside them.
    Gate,
    /// `[all tokens][own heads][128]`, what this rank's heads attended, which peers read back.
    Attended,
    /// `[own tokens][q, k, v][peer heads][128]` gathered for one peer.
    SendInputs(usize),
    /// `[own tokens][peer heads][128]` gathered for one peer.
    SendGate(usize),
    /// `[own tokens][peer heads][128]` of a peer's attention output.
    Received(usize),
}

impl Region {
    /// Whether a block's kernels read this region over and over, which is what decides where it
    /// has to live. Attention reads its inputs once per query tile, and where the wire cannot
    /// reach device memory that difference is sevenfold: 24.8 s a step against 3.4 s.
    pub fn is_computed_in(self) -> bool {
        matches!(
            self,
            Region::Normalized | Region::Scales | Region::Inputs | Region::Gate | Region::Attended
        )
    }

    /// A stable name for the wire: both sides list what they made, and a peer looks an entry up
    /// rather than counting on the two lists lining up.
    pub fn code(self) -> (u32, u32) {
        match self {
            Region::Inputs => (0, 0),
            Region::Gate => (1, 0),
            Region::Attended => (2, 0),
            Region::SendInputs(peer) => (3, peer as u32),
            Region::SendGate(peer) => (4, peer as u32),
            Region::Received(peer) => (5, peer as u32),
            Region::Normalized => (6, 0),
            Region::Scales => (7, 0),
        }
    }

    pub fn from_code(kind: u32, peer: u32) -> Option<Self> {
        Some(match kind {
            0 => Region::Inputs,
            1 => Region::Gate,
            2 => Region::Attended,
            3 => Region::SendInputs(peer as usize),
            4 => Region::SendGate(peer as usize),
            5 => Region::Received(peer as usize),
            6 => Region::Normalized,
            7 => Region::Scales,
            _ => return None,
        })
    }
}

/// Every region a shared-out block needs and how long it is, so that a transport can make them all
/// before the first block instead of growing them under the addresses a peer already holds.
pub fn regions(shard: &Shard, tokens: usize, hidden: usize, gated: bool) -> Vec<(Region, usize)> {
    let own = shard.own_heads().len() * HEAD_DIM;
    let rows = shard.own_tokens().len();
    let mut list = vec![
        (Region::Normalized, tokens * hidden),
        (Region::Scales, tokens * 4),
        (Region::Inputs, tokens * 3 * own * 2),
        (Region::Attended, tokens * own * 2),
    ];
    if gated {
        list.push((Region::Gate, tokens * own * 2));
    }
    for peer in 0..shard.ranks() {
        if peer == shard.rank {
            continue;
        }
        let inner = shard.heads[peer].len() * HEAD_DIM;
        list.push((Region::Received(peer), rows * inner * 2));
    }
    list
}

/// Carries the two exchanges a block needs. Regions are named rather than passed because a
/// transport may have to register them, which costs far more than a transfer and so happens once
/// for a whole run.
///
/// A rank never exchanges with itself: its own share is gathered straight into place.
pub trait Exchange {
    fn rank(&self) -> usize;
    fn ranks(&self) -> usize;

    /// This rank's `region`, at least `bytes` long, made on the first call and kept afterwards.
    fn region(&mut self, region: Region, bytes: usize) -> Result<*mut c_void, Error>;

    /// Writes `bytes` of this rank's `from` region, `offset` bytes in, into `peer`'s `into` region
    /// `peer_offset` bytes in. The ranks push rather than pull because the link carries a write at
    /// 13.0 GB/s and a read at 9.6.
    fn write(
        &mut self,
        peer: usize,
        from: Region,
        offset: usize,
        into: Region,
        peer_offset: usize,
        bytes: usize,
    ) -> Result<(), Error>;

    /// Puts `bytes` of `region`, `offset` bytes in, where the wire can reach it. A transport whose
    /// wire cannot reach device memory keeps a copy in memory it can register, and this is where it
    /// refreshes it. Called before the writes that send it.
    fn publish(&mut self, region: Region, offset: usize, bytes: usize) -> Result<(), Error>;

    /// Takes `bytes` of `region`, `offset` bytes in, from where the wire left it to where the
    /// kernels read it. Called after the barrier that says every peer's writes have landed.
    fn receive(&mut self, region: Region, offset: usize, bytes: usize) -> Result<(), Error>;

    /// Every rank has reached this point, so what they wrote may be read and what they read may be
    /// written again.
    fn barrier(&mut self) -> Result<(), Error>;
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
    pub exchange: &'a mut dyn Exchange,
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
    fn rank(&self) -> usize {
        0
    }

    fn ranks(&self) -> usize {
        1
    }

    fn region(&mut self, region: Region, bytes: usize) -> Result<*mut c_void, Error> {
        if self.regions.get(&region).map_or(0, DeviceBuffer::bytes) < bytes {
            self.regions.insert(region, DeviceBuffer::zeroed(bytes)?);
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
    ) -> Result<(), Error> {
        Err(Error::Model(format!(
            "one rank cannot write to rank {peer}"
        )))
    }

    fn publish(&mut self, _region: Region, _offset: usize, _bytes: usize) -> Result<(), Error> {
        Ok(())
    }

    fn receive(&mut self, _region: Region, _offset: usize, _bytes: usize) -> Result<(), Error> {
        Ok(())
    }

    fn barrier(&mut self) -> Result<(), Error> {
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
