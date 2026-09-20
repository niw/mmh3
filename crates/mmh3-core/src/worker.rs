//! The wire protocol one mmh3 run speaks to the machines it borrows, described in `notes/worker.md`.
//!
//! Messages are a fixed 32-byte header, a descriptor of fixed layout for the kind, and payload bytes.
//! Everything is little-endian, as the checkpoints already are, so a CUDA leader and a Metal worker
//! read the same bytes. Nothing here touches a GPU or a model: it is framing and nothing else.

use std::io::{self, Read, Write};

pub const MAGIC: u32 = u32::from_le_bytes(*b"MH3W");
pub const VERSION: u16 = 2;
/// What a worker announces it can serve, and what a leader asks for.
pub const CAPABILITY_ENCODE_TEXT: u32 = 1 << 0;
pub const CAPABILITY_DECODE_VIDEO: u32 = 1 << 1;
pub const CAPABILITY_DECODE_AUDIO: u32 = 1 << 2;
pub const CAPABILITY_DIT_SHARD: u32 = 1 << 3;
pub const TRANSPORT_TCP: u32 = 1 << 0;
pub const TRANSPORT_RDMA: u32 = 1 << 1;

/// What the machines of one run have to agree on: the messages, their fields and the order two
/// ranks send in. It goes up whenever any of those changes, and a rank that meets another number
/// refuses the connection.
///
/// A checkpoint is matched by digest and a protocol is not, which is why this exists: a pair that
/// disagrees about the wire does not fail, it waits, and two ranks each waiting for the other to
/// speak look exactly like a slow machine.
pub const PROTOCOL: u32 = 1;

pub const BACKEND_CUDA: u8 = 1;
pub const BACKEND_METAL: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Kind {
    Hello = 1,
    Welcome = 2,
    Ping = 3,
    Pong = 4,
    Bandwidth = 5,
    BandwidthDone = 6,
    EncodeText = 7,
    TextStates = 8,
    Error = 9,
    DecodeVideo = 10,
    Canvas = 11,
    Release = 12,
    OpenSession = 13,
    SessionReady = 14,
    StepShard = 15,
    VelocityPart = 16,
    Ready = 17,
    CloseSession = 18,
    SessionLinks = 19,
    PrepareCheckpoint = 20,
    DecodeAudio = 21,
    Samples = 22,
    ShardWrite = 23,
    /// One rank announcing itself on a socket it dialed to a peer of the same run.
    JoinShard = 24,
}

impl Kind {
    pub fn from_u16(value: u16) -> Option<Self> {
        Some(match value {
            1 => Kind::Hello,
            2 => Kind::Welcome,
            3 => Kind::Ping,
            4 => Kind::Pong,
            5 => Kind::Bandwidth,
            6 => Kind::BandwidthDone,
            7 => Kind::EncodeText,
            8 => Kind::TextStates,
            9 => Kind::Error,
            10 => Kind::DecodeVideo,
            11 => Kind::Canvas,
            12 => Kind::Release,
            13 => Kind::OpenSession,
            14 => Kind::SessionReady,
            15 => Kind::StepShard,
            16 => Kind::VelocityPart,
            17 => Kind::Ready,
            18 => Kind::CloseSession,
            19 => Kind::SessionLinks,
            20 => Kind::PrepareCheckpoint,
            21 => Kind::DecodeAudio,
            22 => Kind::Samples,
            23 => Kind::ShardWrite,
            24 => Kind::JoinShard,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub kind: Kind,
    /// Ties a reply to its request, zero for a notification.
    pub request: u64,
    pub length: u64,
}

impl Header {
    pub const BYTES: usize = 32;

    pub fn write(&self, writer: &mut impl Write) -> io::Result<()> {
        let mut bytes = [0u8; Self::BYTES];
        bytes[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        bytes[4..6].copy_from_slice(&VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&(self.kind as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&self.request.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.length.to_le_bytes());
        writer.write_all(&bytes)
    }

    pub fn read(reader: &mut impl Read) -> io::Result<Self> {
        let mut bytes = [0u8; Self::BYTES];
        reader.read_exact(&mut bytes)?;
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an mmh3 worker stream",
            ));
        }
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("worker protocol version {version}, not {VERSION}"),
            ));
        }
        let kind = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let kind = Kind::from_u16(kind).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, format!("message kind {kind}"))
        })?;
        Ok(Header {
            kind,
            request: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            length: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        })
    }
}

/// Appends the fixed-layout fields of a descriptor.
#[derive(Default)]
pub struct Encoder(pub Vec<u8>);

impl Encoder {
    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.0.push(value);
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// A length-prefixed UTF-8 string.
    pub fn string(&mut self, value: &str) -> &mut Self {
        self.u32(value.len() as u32);
        self.0.extend_from_slice(value.as_bytes());
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.extend_from_slice(value);
        self
    }

    pub fn finish(self) -> Vec<u8> {
        self.0
    }
}

/// Reads back what `Encoder` wrote, refusing anything that runs off the end.
pub struct Decoder<'a> {
    bytes: &'a [u8],
    /// Bytes read so far, which a descriptor of variable length reports to its caller.
    pub position: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Decoder { bytes, position: 0 }
    }

    pub fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
        let end = self.position.checked_add(count).ok_or_else(short)?;
        if end > self.bytes.len() {
            return Err(short());
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    pub fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn string(&mut self) -> io::Result<String> {
        let length = self.u32()? as usize;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "a string is not UTF-8"))
    }

    pub fn rest(&mut self) -> &'a [u8] {
        let slice = &self.bytes[self.position..];
        self.position = self.bytes.len();
        slice
    }
}

fn short() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "a message ends early")
}

/// What a checkpoint is called between machines: a role for a leader that has never held the file,
/// and the digest of its safetensors header for one that has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub role: String,
    pub digest: u64,
}

impl Checkpoint {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.string(&self.role).u64(self.digest);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(Checkpoint {
            role: decoder.string()?,
            digest: decoder.u64()?,
        })
    }
}

/// A worker's own measurement of itself, omitted when it cannot benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speed {
    pub gemm_tops: f32,
    pub bandwidth_gbytes: f32,
    /// What this machine does at a block's attention, which the sequence cut cannot be made by:
    /// a rank attends its own heads over the WHOLE sequence, so what attention costs it follows
    /// its share of the heads and not its share of the tokens. Zero from a machine that measured
    /// no such thing, and the cut then prices both by the product alone, as it used to.
    pub attention_tops: f32,
}

/// A machine's RoCE address, opaque here: `mmh3_rdma::Address` gives it meaning. A machine offers
/// one per path, since a Spark's NIC answers over two PCIe links and a payload goes over both.
pub const RDMA_ADDRESS_BYTES: usize = 26;
/// The most paths a machine may offer, as `mmh3_rdma` splits a payload over them.
pub const RDMA_PATHS: usize = 4;
/// One address per path, or nothing when the machine has no RoCE.
pub type RdmaAddresses = Option<Vec<[u8; RDMA_ADDRESS_BYTES]>>;

#[derive(Clone, Debug)]
pub struct Hello {
    pub leader: String,
    pub token: String,
    /// Where this side's reliable connection waits, when it has one.
    pub rdma: RdmaAddresses,
    /// What this side speaks, see `PROTOCOL`. Zero from a build made before this was sent.
    pub protocol: u32,
}

#[derive(Clone, Debug)]
pub struct Welcome {
    /// What this side speaks, see `PROTOCOL`. Zero from a build made before this was sent.
    pub protocol: u32,
    pub backend: u8,
    pub device: String,
    pub memory_bytes: u64,
    pub capabilities: u32,
    pub transports: u32,
    pub speed: Option<Speed>,
    pub checkpoints: Vec<Checkpoint>,
    /// Answered when the leader offered one and this side has one too.
    pub rdma: RdmaAddresses,
}

/// Token identifiers of a prompt, to be encoded by whichever machine holds the text encoder.
#[derive(Clone, Debug)]
pub struct EncodeText {
    pub checkpoint: Checkpoint,
    pub ids: Vec<u32>,
}

/// One chunk of a video decode, with the whole latent so the worker denormalizes and tiles it the
/// way the leader would. The tile geometry travels too, since it decides where the seams fall.
#[derive(Clone, Debug)]
pub struct DecodeVideo {
    pub checkpoint: Checkpoint,
    pub chunk: u32,
    pub tile_size: u32,
    pub tile_overlap: u32,
    /// `[channels, frames, height, width]` of the latent that follows as FP32.
    pub shape: [u32; 4],
}

/// A run's audio latent for a worker to decode while the leader decodes the video. It is small
/// enough that it and the waveform go down the socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeAudio {
    pub checkpoint: Checkpoint,
    /// `[channels, 2, frames]` of the latent that follows as FP32.
    pub shape: [u32; 3],
}

impl DecodeAudio {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .string(&self.checkpoint.role)
            .u64(self.checkpoint.digest);
        for value in self.shape {
            encoder.u32(value);
        }
        encoder.finish()
    }

    /// Returns the descriptor and where the payload begins.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let checkpoint = Checkpoint {
            role: decoder.string()?,
            digest: decoder.u64()?,
        };
        let mut shape = [0u32; 3];
        for value in &mut shape {
            *value = decoder.u32()?;
        }
        Ok((DecodeAudio { checkpoint, shape }, decoder.position))
    }
}

/// A decoded waveform, with its shape as the decoder hands it over and the samples as FP32 in the
/// payload.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Samples {
    pub shape: Vec<u32>,
}

impl Samples {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.shape.len() as u32);
        for value in &self.shape {
            encoder.u32(*value);
        }
        encoder.finish()
    }

    /// Returns the descriptor and where the payload begins.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let count = decoder.u32()? as usize;
        let mut shape = Vec::with_capacity(count.min(8));
        for _ in 0..count {
            shape.push(decoder.u32()?);
        }
        Ok((Samples { shape }, decoder.position))
    }
}

/// A chunk's canvas before any blending, `[3, canvas frames, height, width]` FP32, counted in
/// bytes because that is how it travels and how a decoder hands it over. With a remote key the
/// bytes are not in the payload: they wait in the worker's memory for the leader to read, and the
/// worker holds them until a `Release` says it may let go.
#[derive(Clone, Debug)]
pub struct Canvas {
    pub chunk: u32,
    pub bytes: u64,
    pub remote_address: u64,
    pub remote_keys: Vec<u32>,
}

/// Which block-sparse attention a session runs, and on which of its steps. The whole schedule
/// crosses the wire rather than one step's decision, because every rank has to make the same choice
/// on every step and a rank that disagrees attends a different sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseSettings {
    /// 0 dense, 1 Sol-Attn, 2 VSA.
    pub method: u8,
    /// Sol-Attn's routing threshold in standard deviations.
    pub tau: f32,
    /// The fraction of video tiles VSA drops. FP64, because the kept tile count rounds and two
    /// ranks that keep different tiles attend different sequences: 0.9 through FP32 keeps 101 of
    /// 1000 tiles where FP64 keeps 100.
    pub vsa_sparsity: f64,
    /// The fraction of the steps that run dense first.
    pub start_fraction: f32,
    /// Sequences shorter than this stay dense.
    pub min_tokens: u32,
}

impl SparseSettings {
    pub const DENSE: u8 = 0;
    pub const SOL: u8 = 1;
    pub const VSA: u8 = 2;
}

/// A clean condition the run never denoises: a keyframe of first and last frame generation, or a
/// reference of reference to video generation. Its latents follow the context in the payload, in
/// this order, FP32. A shape of zeroes means the condition has no rows of that kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Condition {
    pub kind: u8,
    /// The frame of the target a keyframe anchors at, unused by a reference.
    pub frame_index: u32,
    /// `[channels, frames, height, width]` of the video latent.
    pub video_shape: [u32; 4],
    /// `[channels, 2, frames]` of the audio latent.
    pub audio_shape: [u32; 3],
}

impl Condition {
    pub const KEYFRAME: u8 = 0;
    pub const REFERENCE_PICTURE: u8 = 1;
    pub const REFERENCE_VIDEO: u8 = 2;
    pub const REFERENCE_AUDIO: u8 = 3;

    /// FP32 values the condition carries, video then audio.
    pub fn values(&self) -> usize {
        let count = |shape: &[u32]| -> usize {
            if shape.contains(&0) {
                0
            } else {
                shape.iter().map(|&extent| extent as usize).product()
            }
        };
        count(&self.video_shape) + count(&self.audio_shape)
    }
}

/// A LoRA or a patch the leader added to its DiT, which every rank has to add the same way. The
/// file carries no role, so the digest of its header is what the two machines agree on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Adapter {
    pub digest: u64,
    pub strength: f32,
    /// 0 for an adapter kept beside the weights, 1 for a merge into them.
    pub mode: u8,
}

impl Adapter {
    pub const ADAPTER: u8 = 0;
    pub const MERGE: u8 = 1;
}

/// A cuBLASLt algorithm the leader chose for a GEMM shape, which the other ranks take rather than
/// timing the candidates themselves: a measurement that separates two of near-equal speed
/// separates the arithmetic, and a rank measures while the wire and the others have its device.
///
/// The fields before the words are the key of the choice, and the words are the algorithm as
/// cuBLASLt hands it over. The layout is the one the backend reads, so the table crosses without
/// being rebuilt.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Algorithm {
    pub kind: i64,
    pub bias: i64,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub data: [u64; 8],
}

/// Opening a shared-out step: everything a rank needs to stand up the same layout the leader has,
/// except the latents, which change every step. Both sides work the shard out of `ranks`, so only
/// which rank this worker takes crosses the wire.
#[derive(Clone, Debug)]
pub struct OpenSession {
    pub checkpoint: Checkpoint,
    pub ranks: u32,
    pub rank: u32,
    /// `[channels, frames, height, width]` of the video latent.
    pub video_shape: [u32; 4],
    /// `[channels, 2, frames]` of the audio latent.
    pub audio_shape: [u32; 3],
    /// `[tokens, text dim]` of the context that follows as FP32, with one modality byte per token.
    pub context_shape: [u32; 2],
    pub shift_video: f32,
    pub shift_audio: f32,
    /// How many steps the run takes, which is what decides when sparse attention starts.
    pub steps: u32,
    pub sparse: SparseSettings,
    /// The keyframes and references, in the order their latents follow the context.
    pub conditions: Vec<Condition>,
    /// The LoRAs and patches, in the order they were added.
    pub adapters: Vec<Adapter>,
    /// 0 for bf16 attention, 1 for INT8 QK with FP8 PV.
    pub precision: u8,
    /// What the leader's chosen algorithms depend on, as its backend names it. A rank whose own
    /// name for it differs, a Metal one above all, takes none of them.
    pub algorithm_key: String,
    /// The algorithms the leader chose for the GEMM shapes it has met.
    pub algorithms: Vec<Algorithm>,
    /// What each rank carries, in rank order. Empty leaves every rank an equal share.
    pub shard: Vec<ShardSpan>,
}

/// What one rank of a shared-out step carries: a run of the sequence and a run of the heads. The
/// leader works the cuts out, since it is the only rank that knows what every machine can do, and
/// sends them whole rather than sending what it decided them from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardSpan {
    pub tokens: [u32; 2],
    pub heads: [u32; 2],
}

/// One rank's write into another's region, for a pair of ranks whose path has no reliable
/// connection: the bytes follow the descriptor down the socket instead of crossing on their own.
/// The writes of one exchange go out together, and `last` closes it, so a rank knows when it has
/// everything a peer meant to send, including when that is nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShardWrite {
    /// The rank the bytes are for, which is not always the rank they arrive from: two workers have
    /// no socket to each other, so what one owes the other goes to the leader and the leader
    /// passes it on.
    pub to: u32,
    /// The region the bytes land in, as `Region::code` names it.
    pub kind: u32,
    pub peer: u32,
    pub offset: u64,
    pub bytes: u64,
    pub last: u8,
}

impl ShardWrite {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .u32(self.to)
            .u32(self.kind)
            .u32(self.peer)
            .u64(self.offset)
            .u64(self.bytes)
            .u8(self.last);
        encoder.finish()
    }

    /// Returns the descriptor and where the payload begins.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let write = ShardWrite {
            to: decoder.u32()?,
            kind: decoder.u32()?,
            peer: decoder.u32()?,
            offset: decoder.u64()?,
            bytes: decoder.u64()?,
            last: decoder.u8()?,
        };
        Ok((write, decoder.position))
    }
}

/// Where one rank's memory sits, so the others can read it. One entry per region a step uses, in
/// the order every rank makes them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemoteRegion {
    /// The rank whose memory this is.
    pub owner: u32,
    /// What the region is for, as the backend names it.
    pub kind: u32,
    /// Which peer it belongs to, for the regions that are made one per peer.
    pub peer: u32,
    pub address: u64,
    pub bytes: u64,
    pub keys: Vec<u32>,
}

/// One rank's end of a connection to another, which the leader passes on so that two workers can
/// reach each other without a socket between them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PeerLink {
    /// The rank at the other end.
    pub peer: u32,
    /// The RoCE addresses, one per path, as `mmh3_rdma::Address` lays them out.
    pub addresses: Vec<[u8; RDMA_ADDRESS_BYTES]>,
}

impl PeerLink {
    pub fn encode_all(links: &[PeerLink]) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(links.len() as u32);
        for link in links {
            encoder.u32(link.peer).u8(link.addresses.len() as u8);
            for address in &link.addresses {
                encoder.bytes(address);
            }
        }
        encoder.finish()
    }

    pub fn decode_all(bytes: &[u8]) -> io::Result<Vec<PeerLink>> {
        let mut decoder = Decoder::new(bytes);
        let count = decoder.u32()? as usize;
        let mut links = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            let peer = decoder.u32()?;
            let paths = decoder.u8()? as usize;
            let mut addresses = Vec::with_capacity(paths.min(RDMA_PATHS));
            for _ in 0..paths.min(RDMA_PATHS) {
                let mut address = [0u8; RDMA_ADDRESS_BYTES];
                address.copy_from_slice(decoder.take(RDMA_ADDRESS_BYTES)?);
                addresses.push(address);
            }
            links.push(PeerLink { peer, addresses });
        }
        Ok(links)
    }
}

/// What a rank made of the session it was given: the regions it registered, for the leader to pass
/// round, and how many of the leader's chosen algorithms it took. A rank that took none of them
/// times its own candidates, which two ranks may answer differently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionReady {
    pub algorithms: u32,
    /// The port this rank's session listens on for the peers that reach it over a socket, or
    /// zero where it offers none. The leader pairs it with the host it reached this rank at,
    /// since a rank behind more than one interface cannot say which of them a peer should use.
    pub listen_port: u32,
    pub regions: Vec<RemoteRegion>,
}

/// What a rank says on a socket it has just dialed to a peer, so the peer knows who called and
/// that it belongs to this run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JoinShard {
    pub rank: u32,
    pub token: String,
}

impl JoinShard {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.rank).string(&self.token);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(JoinShard {
            rank: decoder.u32()?,
            token: decoder.string()?,
        })
    }
}

/// Where one rank listens, as the leader passes it on to the others.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PeerListen {
    pub rank: u32,
    pub address: String,
}

/// What the leader answers every rank with once they have all reported: where each of them
/// listens, and the regions of all of them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionTable {
    pub listens: Vec<PeerListen>,
    pub regions: Vec<RemoteRegion>,
}

impl SessionTable {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.listens.len() as u32);
        for listen in &self.listens {
            encoder.u32(listen.rank).string(&listen.address);
        }
        let mut bytes = encoder.finish();
        bytes.extend(RemoteRegion::encode_all(&self.regions));
        bytes
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let count = decoder.u32()? as usize;
        let mut listens = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            listens.push(PeerListen {
                rank: decoder.u32()?,
                address: decoder.string()?,
            });
        }
        Ok(SessionTable {
            listens,
            regions: RemoteRegion::decode_all(decoder.rest())?,
        })
    }
}

impl SessionReady {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.algorithms).u32(self.listen_port);
        let mut bytes = encoder.finish();
        bytes.extend(RemoteRegion::encode_all(&self.regions));
        bytes
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let algorithms = decoder.u32()?;
        let listen_port = decoder.u32()?;
        Ok(SessionReady {
            algorithms,
            listen_port,
            regions: RemoteRegion::decode_all(decoder.rest())?,
        })
    }
}

impl RemoteRegion {
    pub const BYTES: usize = 32;
}

/// One step of a shared-out run: the sigma and the latents, which every rank embeds for itself.
#[derive(Clone, Copy, Debug)]
pub struct StepShard {
    pub step: u32,
    pub sigma: f32,
}

/// The rows of the velocity one rank ended with, before the video goes back into its own order.
#[derive(Clone, Copy, Debug)]
pub struct VelocityPart {
    pub first_row: u32,
    pub last_row: u32,
    pub video_values: u64,
    pub audio_values: u64,
}

impl OpenSession {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .string(&self.checkpoint.role)
            .u64(self.checkpoint.digest)
            .u32(self.ranks)
            .u32(self.rank);
        for value in self.video_shape {
            encoder.u32(value);
        }
        for value in self.audio_shape {
            encoder.u32(value);
        }
        for value in self.context_shape {
            encoder.u32(value);
        }
        encoder
            .f32(self.shift_video)
            .f32(self.shift_audio)
            .u32(self.steps)
            .u8(self.sparse.method)
            .f32(self.sparse.tau)
            .f64(self.sparse.vsa_sparsity)
            .f32(self.sparse.start_fraction)
            .u32(self.sparse.min_tokens)
            .u8(self.precision)
            .u32(self.conditions.len() as u32);
        for condition in &self.conditions {
            encoder.u8(condition.kind).u32(condition.frame_index);
            for value in condition.video_shape {
                encoder.u32(value);
            }
            for value in condition.audio_shape {
                encoder.u32(value);
            }
        }
        encoder.u32(self.adapters.len() as u32);
        for adapter in &self.adapters {
            encoder
                .u64(adapter.digest)
                .f32(adapter.strength)
                .u8(adapter.mode);
        }
        encoder.u32(self.shard.len() as u32);
        for span in &self.shard {
            encoder
                .u32(span.tokens[0])
                .u32(span.tokens[1])
                .u32(span.heads[0])
                .u32(span.heads[1]);
        }
        encoder
            .string(&self.algorithm_key)
            .u32(self.algorithms.len() as u32);
        for algorithm in &self.algorithms {
            encoder
                .u64(algorithm.kind as u64)
                .u64(algorithm.bias as u64)
                .u64(algorithm.m as u64)
                .u64(algorithm.n as u64)
                .u64(algorithm.k as u64);
            for word in algorithm.data {
                encoder.u64(word);
            }
        }
        encoder.finish()
    }

    /// Returns the descriptor and where the payload begins.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let checkpoint = Checkpoint {
            role: decoder.string()?,
            digest: decoder.u64()?,
        };
        let (ranks, rank) = (decoder.u32()?, decoder.u32()?);
        let mut video_shape = [0u32; 4];
        for value in &mut video_shape {
            *value = decoder.u32()?;
        }
        let mut audio_shape = [0u32; 3];
        for value in &mut audio_shape {
            *value = decoder.u32()?;
        }
        let mut context_shape = [0u32; 2];
        for value in &mut context_shape {
            *value = decoder.u32()?;
        }
        let session = OpenSession {
            checkpoint,
            ranks,
            rank,
            video_shape,
            audio_shape,
            context_shape,
            shift_video: decoder.f32()?,
            shift_audio: decoder.f32()?,
            steps: decoder.u32()?,
            sparse: SparseSettings {
                method: decoder.u8()?,
                tau: decoder.f32()?,
                vsa_sparsity: decoder.f64()?,
                start_fraction: decoder.f32()?,
                min_tokens: decoder.u32()?,
            },
            precision: decoder.u8()?,
            conditions: Vec::new(),
            adapters: Vec::new(),
            shard: Vec::new(),
            algorithm_key: String::new(),
            algorithms: Vec::new(),
        };
        let count = decoder.u32()? as usize;
        let mut conditions = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            let mut condition = Condition {
                kind: decoder.u8()?,
                frame_index: decoder.u32()?,
                ..Condition::default()
            };
            for value in &mut condition.video_shape {
                *value = decoder.u32()?;
            }
            for value in &mut condition.audio_shape {
                *value = decoder.u32()?;
            }
            conditions.push(condition);
        }
        let count = decoder.u32()? as usize;
        let mut adapters = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            adapters.push(Adapter {
                digest: decoder.u64()?,
                strength: decoder.f32()?,
                mode: decoder.u8()?,
            });
        }
        let count = decoder.u32()? as usize;
        let mut shard = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            shard.push(ShardSpan {
                tokens: [decoder.u32()?, decoder.u32()?],
                heads: [decoder.u32()?, decoder.u32()?],
            });
        }
        let algorithm_key = decoder.string()?;
        let count = decoder.u32()? as usize;
        let mut algorithms = Vec::with_capacity(count.min(1024));
        for _ in 0..count {
            let mut algorithm = Algorithm {
                kind: decoder.u64()? as i64,
                bias: decoder.u64()? as i64,
                m: decoder.u64()? as i64,
                n: decoder.u64()? as i64,
                k: decoder.u64()? as i64,
                ..Algorithm::default()
            };
            for word in &mut algorithm.data {
                *word = decoder.u64()?;
            }
            algorithms.push(algorithm);
        }
        let session = OpenSession {
            conditions,
            adapters,
            shard,
            algorithm_key,
            algorithms,
            ..session
        };
        Ok((session, decoder.position))
    }
}

impl RemoteRegion {
    pub fn encode_all(regions: &[RemoteRegion]) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(regions.len() as u32);
        for region in regions {
            encoder
                .u32(region.owner)
                .u32(region.kind)
                .u32(region.peer)
                .u64(region.address)
                .u64(region.bytes);
            encode_keys(&mut encoder, &region.keys);
        }
        encoder.finish()
    }

    pub fn decode_all(bytes: &[u8]) -> io::Result<Vec<RemoteRegion>> {
        let mut decoder = Decoder::new(bytes);
        let count = decoder.u32()? as usize;
        (0..count)
            .map(|_| {
                Ok(RemoteRegion {
                    owner: decoder.u32()?,
                    kind: decoder.u32()?,
                    peer: decoder.u32()?,
                    address: decoder.u64()?,
                    bytes: decoder.u64()?,
                    keys: decode_keys(&mut decoder)?,
                })
            })
            .collect()
    }
}

impl StepShard {
    pub const BYTES: usize = 8;

    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u32(self.step).f32(self.sigma);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(StepShard {
            step: decoder.u32()?,
            sigma: decoder.f32()?,
        })
    }
}

impl VelocityPart {
    pub const BYTES: usize = 24;

    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .u32(self.first_row)
            .u32(self.last_row)
            .u64(self.video_values)
            .u64(self.audio_values);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(VelocityPart {
            first_row: decoder.u32()?,
            last_row: decoder.u32()?,
            video_values: decoder.u64()?,
            audio_values: decoder.u64()?,
        })
    }
}

/// `[tokens, hidden]` FP32, as the encoders already return it.
#[derive(Clone, Debug)]
pub struct TextStates {
    pub tokens: usize,
    pub hidden: usize,
}

fn encode_rdma(encoder: &mut Encoder, addresses: &RdmaAddresses) {
    match addresses {
        Some(addresses) => {
            encoder.u8(addresses.len().min(RDMA_PATHS) as u8);
            for address in addresses.iter().take(RDMA_PATHS) {
                encoder.bytes(address);
            }
        }
        None => {
            encoder.u8(0);
        }
    }
}

fn decode_rdma(decoder: &mut Decoder<'_>) -> io::Result<RdmaAddresses> {
    let count = decoder.u8()? as usize;
    if count == 0 {
        return Ok(None);
    }
    let mut addresses = Vec::with_capacity(count.min(RDMA_PATHS));
    for _ in 0..count.min(RDMA_PATHS) {
        let mut address = [0u8; RDMA_ADDRESS_BYTES];
        address.copy_from_slice(decoder.take(RDMA_ADDRESS_BYTES)?);
        addresses.push(address);
    }
    Ok(Some(addresses))
}

/// One key per path, which a peer needs all of to reach a region.
fn encode_keys(encoder: &mut Encoder, keys: &[u32]) {
    encoder.u8(keys.len().min(RDMA_PATHS) as u8);
    for key in keys.iter().take(RDMA_PATHS) {
        encoder.u32(*key);
    }
}

fn decode_keys(decoder: &mut Decoder<'_>) -> io::Result<Vec<u32>> {
    let count = decoder.u8()? as usize;
    let mut keys = Vec::with_capacity(count.min(RDMA_PATHS));
    for _ in 0..count.min(RDMA_PATHS) {
        keys.push(decoder.u32()?);
    }
    Ok(keys)
}

impl Hello {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.string(&self.leader).string(&self.token);
        encode_rdma(&mut encoder, &self.rdma);
        // Last, so a build that does not know about it reads everything before it and ignores
        // this, which is what turns a version difference into a refusal instead of a hang.
        encoder.u32(self.protocol);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let leader = decoder.string()?;
        let token = decoder.string()?;
        let rdma = decode_rdma(&mut decoder)?;
        Ok(Hello {
            leader,
            token,
            rdma,
            protocol: decoder.u32().unwrap_or(0),
        })
    }
}

impl Welcome {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .u8(self.backend)
            .string(&self.device)
            .u64(self.memory_bytes)
            .u32(self.capabilities)
            .u32(self.transports);
        match self.speed {
            Some(speed) => {
                encoder
                    .u8(1)
                    .f32(speed.gemm_tops)
                    .f32(speed.bandwidth_gbytes)
                    .f32(speed.attention_tops);
            }
            None => {
                encoder.u8(0);
            }
        }
        encoder.u32(self.checkpoints.len() as u32);
        for checkpoint in &self.checkpoints {
            encoder.string(&checkpoint.role).u64(checkpoint.digest);
        }
        encode_rdma(&mut encoder, &self.rdma);
        encoder.u32(self.protocol);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let backend = decoder.u8()?;
        let device = decoder.string()?;
        let memory_bytes = decoder.u64()?;
        let capabilities = decoder.u32()?;
        let transports = decoder.u32()?;
        let speed = match decoder.u8()? {
            0 => None,
            _ => Some(Speed {
                gemm_tops: decoder.f32()?,
                bandwidth_gbytes: decoder.f32()?,
                attention_tops: decoder.f32()?,
            }),
        };
        let count = decoder.u32()? as usize;
        let mut checkpoints = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            checkpoints.push(Checkpoint {
                role: decoder.string()?,
                digest: decoder.u64()?,
            });
        }
        let rdma = decode_rdma(&mut decoder)?;
        Ok(Welcome {
            protocol: decoder.u32().unwrap_or(0),
            backend,
            device,
            memory_bytes,
            capabilities,
            transports,
            speed,
            checkpoints,
            rdma,
        })
    }
}

impl EncodeText {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .string(&self.checkpoint.role)
            .u64(self.checkpoint.digest)
            .u32(self.ids.len() as u32);
        for &id in &self.ids {
            encoder.u32(id);
        }
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        let checkpoint = Checkpoint {
            role: decoder.string()?,
            digest: decoder.u64()?,
        };
        let count = decoder.u32()? as usize;
        let mut ids = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            ids.push(decoder.u32()?);
        }
        Ok(EncodeText { checkpoint, ids })
    }
}

impl TextStates {
    /// The descriptor, followed by `tokens × hidden` FP32 values.
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.u64(self.tokens as u64).u64(self.hidden as u64);
        encoder.finish()
    }

    pub const BYTES: usize = 16;

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(TextStates {
            tokens: decoder.u64()? as usize,
            hidden: decoder.u64()? as usize,
        })
    }
}

impl DecodeVideo {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .string(&self.checkpoint.role)
            .u64(self.checkpoint.digest)
            .u32(self.chunk)
            .u32(self.tile_size)
            .u32(self.tile_overlap);
        for extent in self.shape {
            encoder.u32(extent);
        }
        encoder.finish()
    }

    /// The descriptor's length, after which the latent's FP32 values begin.
    pub const BYTES: usize = 4 + 4 + 8 + 4 + 4 + 4 + 16;

    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let checkpoint = Checkpoint {
            role: decoder.string()?,
            digest: decoder.u64()?,
        };
        let request = DecodeVideo {
            checkpoint,
            chunk: decoder.u32()?,
            tile_size: decoder.u32()?,
            tile_overlap: decoder.u32()?,
            shape: [
                decoder.u32()?,
                decoder.u32()?,
                decoder.u32()?,
                decoder.u32()?,
            ],
        };
        Ok((request, decoder.position))
    }
}

impl Canvas {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder
            .u64(self.chunk as u64)
            .u64(self.bytes)
            .u64(self.remote_address);
        encode_keys(&mut encoder, &self.remote_keys);
        encoder.finish()
    }

    /// Returns the descriptor and where the payload begins, since a canvas that comes down the
    /// socket follows it and the keys before it are as many as the paths its sender had.
    pub fn decode(bytes: &[u8]) -> io::Result<(Self, usize)> {
        let mut decoder = Decoder::new(bytes);
        let canvas = Canvas {
            chunk: decoder.u64()? as u32,
            bytes: decoder.u64()?,
            remote_address: decoder.u64()?,
            remote_keys: decode_keys(&mut decoder)?,
        };
        Ok((canvas, decoder.position))
    }
}

/// The digest of a checkpoint: its tensor names, types, shapes and offsets, which pin the layout
/// without reading a byte of the weights. File names differ between machines and stay out of it.
pub fn digest<'a>(
    tensors: impl Iterator<Item = (&'a str, &'a str, &'a [usize], usize, usize)>,
) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (name, dtype, shape, start, end) in tensors {
        mix(name.as_bytes());
        mix(dtype.as_bytes());
        for &extent in shape {
            mix(&(extent as u64).to_le_bytes());
        }
        mix(&(start as u64).to_le_bytes());
        mix(&(end as u64).to_le_bytes());
    }
    hash
}

/// Writes a message and its payload in one place, so every sender frames the same way.
pub fn send(
    writer: &mut impl Write,
    kind: Kind,
    request: u64,
    descriptor: &[u8],
    payload: &[u8],
) -> io::Result<()> {
    Header {
        kind,
        request,
        length: (descriptor.len() + payload.len()) as u64,
    }
    .write(writer)?;
    writer.write_all(descriptor)?;
    writer.write_all(payload)?;
    writer.flush()
}

/// Reads a header and its body. `limit` refuses a body larger than the caller expects.
pub fn receive(reader: &mut impl Read, limit: u64) -> io::Result<(Header, Vec<u8>)> {
    let header = Header::read(reader)?;
    if header.length > limit {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "a {:?} message of {} bytes passes the limit",
                header.kind, header.length
            ),
        ));
    }
    let mut body = vec![0u8; header.length as usize];
    reader.read_exact(&mut body)?;
    Ok((header, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine built before the protocol was named says nothing about it, and both sides have to
    /// read that as zero rather than as a broken message: a refusal with a reason is the point, and
    /// a decode error would say nothing about why the two cannot talk.
    #[test]
    fn a_handshake_without_a_protocol_reads_as_none() {
        let hello = Hello {
            leader: "spark".to_owned(),
            token: "secret".to_owned(),
            rdma: None,
            protocol: PROTOCOL,
        };
        let encoded = hello.encode();
        let older = &encoded[..encoded.len() - 4];
        let decoded = Hello::decode(older).unwrap();
        assert_eq!(decoded.protocol, 0);
        assert_eq!(decoded.leader, hello.leader);
        assert_eq!(decoded.token, hello.token);
        assert_eq!(Hello::decode(&encoded).unwrap().protocol, PROTOCOL);
    }

    #[test]
    fn headers_and_descriptors_survive_a_round_trip() {
        let welcome = Welcome {
            protocol: PROTOCOL,
            backend: BACKEND_CUDA,
            device: "NVIDIA GB10".to_owned(),
            memory_bytes: 128 << 30,
            capabilities: CAPABILITY_ENCODE_TEXT | CAPABILITY_DECODE_VIDEO,
            transports: TRANSPORT_TCP,
            rdma: None,
            speed: Some(Speed {
                gemm_tops: 181.6,
                bandwidth_gbytes: 227.7,
                attention_tops: 42.5,
            }),
            checkpoints: vec![Checkpoint {
                role: "text_encoder.h3.int8_convrot".to_owned(),
                digest: 0x0123_4567_89ab_cdef,
            }],
        };
        let mut stream = Vec::new();
        send(&mut stream, Kind::Welcome, 7, &welcome.encode(), &[]).unwrap();
        let (header, body) = receive(&mut stream.as_slice(), 1 << 20).unwrap();
        assert_eq!(header.kind, Kind::Welcome);
        assert_eq!(header.request, 7);
        let decoded = Welcome::decode(&body).unwrap();
        assert_eq!(decoded.protocol, PROTOCOL);
        assert_eq!(decoded.device, welcome.device);
        assert_eq!(decoded.capabilities, welcome.capabilities);
        assert_eq!(decoded.speed.unwrap().gemm_tops, 181.6);
        assert_eq!(decoded.speed.unwrap().attention_tops, 42.5);
        assert_eq!(decoded.checkpoints, welcome.checkpoints);
    }

    #[test]
    fn a_prompt_and_its_states_survive_a_round_trip() {
        let request = EncodeText {
            checkpoint: Checkpoint {
                role: "text_encoder.h3.int8_convrot".to_owned(),
                digest: 0,
            },
            ids: vec![1, 2, 3, 400_000],
        };
        let mut stream = Vec::new();
        send(&mut stream, Kind::EncodeText, 1, &request.encode(), &[]).unwrap();
        let states = TextStates {
            tokens: 2,
            hidden: 3,
        };
        let values: Vec<f32> = vec![1.0, -2.0, 3.5, 4.0, 5.0, 6.0];
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        send(&mut stream, Kind::TextStates, 1, &states.encode(), &payload).unwrap();

        let mut reader = stream.as_slice();
        let (_, body) = receive(&mut reader, 1 << 20).unwrap();
        assert_eq!(EncodeText::decode(&body).unwrap().ids, request.ids);
        let (_, body) = receive(&mut reader, 1 << 20).unwrap();
        let decoded = TextStates::decode(&body[..TextStates::BYTES]).unwrap();
        assert_eq!((decoded.tokens, decoded.hidden), (2, 3));
        assert_eq!(body.len(), TextStates::BYTES + 24);
    }

    #[test]
    fn a_body_past_the_limit_is_refused() {
        let mut stream = Vec::new();
        send(&mut stream, Kind::Ping, 0, &[0u8; 64], &[]).unwrap();
        assert!(receive(&mut stream.as_slice(), 32).is_err());
    }

    #[test]
    fn a_digest_follows_the_layout_and_not_the_name() {
        let first = digest([("a", "F32", &[2usize, 3][..], 0usize, 24usize)].into_iter());
        let same = digest([("a", "F32", &[2usize, 3][..], 0usize, 24usize)].into_iter());
        let other = digest([("a", "F32", &[3usize, 2][..], 0usize, 24usize)].into_iter());
        assert_eq!(first, same);
        assert_ne!(first, other);
    }
}
