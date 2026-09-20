//! How one DiT step is split across machines, the way Ulysses does it, and the memory the ranks
//! exchange through.
//!
//! Every rank carries a contiguous run of the packed sequence through the blocks, which needs no
//! communication at all for the normalizations, the projections and the MLP. Attention is the
//! exception, since a query has to see every key. So each block exchanges twice: once to turn
//! "my tokens, every head" into "every token, my heads", and once to turn the result back.
//!
//! None of this is a backend's business, and both of them have to agree on every word of it: the
//! cuts, the names of the regions and the codes those names cross the wire as. So it lives here,
//! where there is one of each.

use std::fmt;
use std::ops::Range;
use std::time::Duration;

/// Values in one attention head, which every backend lays out the same way.
pub const HEAD_DIM: usize = 128;

/// One rank's rows of the velocity a shared-out step produces, before the video goes back into
/// its own order. Every backend answers with this and the leader puts the rows together, so it is
/// named here rather than twice.
///
/// Not to be confused with `worker::VelocityPart`, which is the handful of numbers describing
/// these rows on the wire.
///
/// NOTE: these are the projections as the final layer leaves them, NOT the velocity. A whole step
/// answers with the velocity, which is their negation, but a rank answers with them as they are
/// and the leader negates once when it assembles the parts. A rank that negates its own rows
/// sends them back the wrong way, and rows driven away from the data look like rows that were
/// never diffused at all, which is a long way from where the sign is.
#[derive(Clone, Debug)]
pub struct VelocityRows {
    pub rows: Range<usize>,
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
}

/// What a transport could not do, in its own words. A backend turns this into whatever its own
/// calls answer with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeError(pub String);

impl fmt::Display for ExchangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ExchangeError {}

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

    /// Cuts `tokens` and `heads` into shares of `weights`, one per rank, so that a machine carries
    /// what it can rather than its turn. Tokens still divide on a multiple of `alignment`, and
    /// what the rounding leaves goes to the rank with the largest weight rather than to the last,
    /// which under a weighted cut may be the slowest. Every rank keeps at least one unit of each,
    /// since a rank with no heads or no rows has nothing to run and nowhere to put it.
    pub fn weighted(
        rank: usize,
        tokens_by: &[f64],
        heads_by: &[f64],
        tokens: usize,
        heads: usize,
        alignment: usize,
    ) -> Self {
        let cut = |weights: &[f64], total: usize, step: usize| -> Vec<Range<usize>> {
            // What the rounding leaves goes to the fastest at this kind of work rather than to
            // the last, which under a weighted cut may be the slowest.
            let fastest = weights
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map_or(0, |(index, _)| index);
            let step = step.max(1);
            let sum: f64 = weights.iter().map(|weight| weight.max(0.0)).sum();
            let mut counts: Vec<usize> = weights
                .iter()
                .map(|weight| {
                    let share = if sum > 0.0 {
                        total as f64 * weight.max(0.0) / sum
                    } else {
                        total as f64 / weights.len() as f64
                    };
                    ((share / step as f64).floor() as usize * step).max(step)
                })
                .collect();
            // The minimum may have overdrawn the total, so the fastest rank gives it back, and
            // what is left over after the rounding goes to it instead.
            let taken: usize = counts.iter().sum();
            if taken <= total {
                counts[fastest] += total - taken;
            } else {
                counts[fastest] = counts[fastest].saturating_sub(taken - total);
            }
            let mut ranges = Vec::with_capacity(counts.len());
            let mut start = 0;
            for count in counts {
                let end = (start + count).min(total);
                ranges.push(start..end);
                start = end;
            }
            if let Some(last) = ranges.last_mut() {
                last.end = total;
            }
            ranges
        };
        Shard {
            rank,
            tokens: cut(tokens_by, tokens, alignment),
            heads: cut(heads_by, heads, 1),
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
    /// What `region` answers with: a pointer where a backend addresses memory by one, and
    /// whatever else where it does not.
    type Memory;

    fn rank(&self) -> usize;
    fn ranks(&self) -> usize;

    /// This rank's `region`, at least `bytes` long, made on the first call and kept afterwards.
    fn region(&mut self, region: Region, bytes: usize) -> Result<Self::Memory, ExchangeError>;

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
    ) -> Result<(), ExchangeError>;

    /// Puts `bytes` of `region`, `offset` bytes in, where the wire can reach it. A transport whose
    /// wire cannot reach device memory keeps a copy in memory it can register, and this is where it
    /// refreshes it. Called before the writes that send it.
    fn publish(&mut self, region: Region, offset: usize, bytes: usize)
    -> Result<(), ExchangeError>;

    /// Takes `bytes` of `region`, `offset` bytes in, from where the wire left it to where the
    /// kernels read it. Called after the barrier that says every peer's writes have landed.
    fn receive(&mut self, region: Region, offset: usize, bytes: usize)
    -> Result<(), ExchangeError>;

    /// Every rank has reached this point, so what they wrote may be read and what they read may be
    /// written again.
    fn barrier(&mut self) -> Result<(), ExchangeError>;
}

/// Where a shared-out step went, which is what decides whether sharing one is worth it at all.
///
/// The four are exclusive and cover a block's share between them, so a backend charges every span
/// to exactly one. `attend` is the only one that would still be there if the step ran alone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// One rank's share of a step and the transport its blocks exchange over. `M` is what the
/// transport's regions answer with, which is a backend's own business and nothing else here.
pub struct ShardContext<'a, M> {
    pub shard: Shard,
    pub exchange: &'a mut dyn Exchange<Memory = M>,
    pub timing: ShardTiming,
}
