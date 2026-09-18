//! The wire protocol one mmh3 run speaks to the machines it borrows, described in `notes/worker.md`.
//!
//! Messages are a fixed 32-byte header, a descriptor of fixed layout for the kind, and payload bytes.
//! Everything is little-endian, as the checkpoints already are, so a CUDA leader and a Metal worker
//! read the same bytes. Nothing here touches a GPU or a model: it is framing and nothing else.

use std::io::{self, Read, Write};

pub const MAGIC: u32 = u32::from_le_bytes(*b"MH3W");
pub const VERSION: u16 = 1;
/// What a worker announces it can serve, and what a leader asks for.
pub const CAPABILITY_ENCODE_TEXT: u32 = 1 << 0;
pub const CAPABILITY_DECODE_VIDEO: u32 = 1 << 1;
pub const CAPABILITY_DECODE_AUDIO: u32 = 1 << 2;
pub const CAPABILITY_DIT_SHARD: u32 = 1 << 3;
pub const TRANSPORT_TCP: u32 = 1 << 0;
pub const TRANSPORT_RDMA: u32 = 1 << 1;

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
    position: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Decoder { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
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

/// A worker's own measurement of itself, omitted when it cannot benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Speed {
    pub gemm_tops: f32,
    pub bandwidth_gbytes: f32,
}

#[derive(Clone, Debug)]
pub struct Hello {
    pub leader: String,
    pub token: String,
}

#[derive(Clone, Debug)]
pub struct Welcome {
    pub backend: u8,
    pub device: String,
    pub memory_bytes: u64,
    pub capabilities: u32,
    pub transports: u32,
    pub speed: Option<Speed>,
    pub checkpoints: Vec<Checkpoint>,
}

/// Token identifiers of a prompt, to be encoded by whichever machine holds the text encoder.
#[derive(Clone, Debug)]
pub struct EncodeText {
    pub checkpoint: Checkpoint,
    pub ids: Vec<u32>,
}

/// `[tokens, hidden]` FP32, as the encoders already return it.
#[derive(Clone, Debug)]
pub struct TextStates {
    pub tokens: usize,
    pub hidden: usize,
}

impl Hello {
    pub fn encode(&self) -> Vec<u8> {
        let mut encoder = Encoder::default();
        encoder.string(&self.leader).string(&self.token);
        encoder.finish()
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        let mut decoder = Decoder::new(bytes);
        Ok(Hello {
            leader: decoder.string()?,
            token: decoder.string()?,
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
                    .f32(speed.bandwidth_gbytes);
            }
            None => {
                encoder.u8(0);
            }
        }
        encoder.u32(self.checkpoints.len() as u32);
        for checkpoint in &self.checkpoints {
            encoder.string(&checkpoint.role).u64(checkpoint.digest);
        }
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
        Ok(Welcome {
            backend,
            device,
            memory_bytes,
            capabilities,
            transports,
            speed,
            checkpoints,
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

    #[test]
    fn headers_and_descriptors_survive_a_round_trip() {
        let welcome = Welcome {
            backend: BACKEND_CUDA,
            device: "NVIDIA GB10".to_owned(),
            memory_bytes: 128 << 30,
            capabilities: CAPABILITY_ENCODE_TEXT | CAPABILITY_DECODE_VIDEO,
            transports: TRANSPORT_TCP,
            speed: Some(Speed {
                gemm_tops: 181.6,
                bandwidth_gbytes: 227.7,
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
        assert_eq!(decoded.device, welcome.device);
        assert_eq!(decoded.capabilities, welcome.capabilities);
        assert_eq!(decoded.speed.unwrap().gemm_tops, 181.6);
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
