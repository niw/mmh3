//! Serving other mmh3 runs, and borrowing them. The wire format is `mmh3_core::worker`, the design
//! `notes/worker.md`. A worker holds checkpoints the leader may not have, so both sides name them by
//! role and confirm them by digest.

use mmh3_core::safetensors::SafeTensors;
#[cfg(any(feature = "cuda", feature = "metal"))]
use mmh3_core::shard;
#[cfg(any(feature = "cuda", feature = "metal"))]
use mmh3_core::shard::Shard;
use mmh3_core::tensor::Tensor;
#[cfg(feature = "cuda")]
use mmh3_core::worker::OpenSession;
use mmh3_core::worker::{
    self, CAPABILITY_DECODE_AUDIO, CAPABILITY_DECODE_VIDEO, CAPABILITY_DIT_SHARD,
    CAPABILITY_ENCODE_TEXT, Canvas, Checkpoint, DecodeAudio, DecodeVideo, EncodeText, Header,
    Hello, Kind, Samples, TRANSPORT_TCP, TextStates, Welcome,
};
use std::collections::HashMap;
use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

pub const DEFAULT_PORT: u16 = 7833;
/// The video VAE a leader and its workers agree on, as `--video-vae` picks it here.
pub const VIDEO_VAE_ROLE: &str = "video_vae.h3.int8_convrot";
pub const TEXT_ENCODER_ROLE: &str = "text_encoder.h3.int8_convrot";
/// The audio VAE a leader and its workers agree on, as `--audio-vae` picks it here.
pub const AUDIO_VAE_ROLE: &str = "audio_vae.h3.fp32";
/// Bodies larger than this are refused before they are read.
const BODY_LIMIT: u64 = 4 << 30;
/// How long one exchange of a block may take before the run gives up on the peer.
#[cfg(feature = "cuda")]
const EXCHANGE_TIMEOUT: i32 = 60_000;
/// Stream buffers. The default 8 KiB turns a payload of hundreds of megabytes into tens of
/// thousands of syscalls, which a fast link notices.
const STREAM_BUFFER: usize = 4 << 20;

#[cfg(feature = "cuda")]
const BACKEND: u8 = worker::BACKEND_CUDA;
#[cfg(feature = "metal")]
const BACKEND: u8 = worker::BACKEND_METAL;

#[cfg(feature = "cuda")]
type TextEncoder = mmh3_cuda::text_encoder::CudaTextEncoder;
#[cfg(feature = "metal")]
type TextEncoder = mmh3_metal::text_encoder::MetalTextEncoder;
/// The checkpoint a session has loaded, kept for the rest of it.
#[cfg(any(feature = "cuda", feature = "metal"))]
type Resident = Option<(u64, TextEncoder)>;
#[cfg(feature = "cuda")]
type VideoDecoder = mmh3_cuda::vae::CudaVideoDecoder;
#[cfg(feature = "metal")]
type VideoDecoder = mmh3_metal::vae::MetalVideoDecoder;
/// A decoder and the tile geometry it was built for, which a request may change.
#[cfg(any(feature = "cuda", feature = "metal"))]
type ResidentDecoder = Option<(u64, usize, usize, VideoDecoder)>;
#[cfg(feature = "cuda")]
type AudioDecoder = mmh3_cuda::audio_vae::CudaAudioDecoder;
#[cfg(feature = "metal")]
type AudioDecoder = mmh3_metal::audio_vae::MetalAudioDecoder;
#[cfg(any(feature = "cuda", feature = "metal"))]
type ResidentAudio = Option<(u64, AudioDecoder)>;

/// Checkpoints a worker offers, as `(role, path inside the models directory)`. Roles name what a
/// file is for, since the file names differ between machines.
const ROLES: &[(&str, &str)] = &[
    (
        "text_encoder.h3.int8_convrot",
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ),
    (
        "text_encoder.h3.nvfp4_awq",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ),
    (
        "video_vae.h3.int8_convrot",
        "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    ),
    (
        "video_vae.h3.fp16",
        "vae/minimax_h3_video_vae_fp16.safetensors",
    ),
    (
        "audio_vae.h3.fp32",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ),
];

/// `host`, `host:port` or an address, with the worker port when none is given.
pub fn address(text: &str) -> String {
    if text
        .rsplit(':')
        .next()
        .is_some_and(|port| port.parse::<u16>().is_ok())
    {
        text.to_owned()
    } else {
        format!("{text}:{DEFAULT_PORT}")
    }
}

/// Tensors a digest reads bytes of, and how many bytes of each. Enough to tell two fine-tunes of
/// one architecture apart, and few enough to cost a handful of seeks.
const DIGEST_SAMPLES: usize = 32;
const DIGEST_SAMPLE_BYTES: usize = 64;

/// The digest of a checkpoint, which two machines compare instead of file names.
///
/// The header pins every tensor's name, dtype, shape and offset, and a sample of the weights tells
/// two files of the same layout apart: the fl2va and ref2va DiTs have byte-identical headers, so a
/// digest of the layout alone would name either of them.
pub fn digest(file: &SafeTensors) -> u64 {
    use std::io::{Read, Seek, SeekFrom};

    let layout = worker::digest(file.tensors().iter().map(|tensor| {
        (
            tensor.name.as_str(),
            tensor.dtype.name(),
            tensor.shape.as_slice(),
            tensor.data_range.start,
            tensor.data_range.end,
        )
    }));
    let tensors = file.tensors();
    let Ok(mut handle) = std::fs::File::open(file.path()) else {
        return layout;
    };
    let mut hash = layout;
    let mut sample = [0u8; DIGEST_SAMPLE_BYTES];
    for tensor in tensors
        .iter()
        .step_by(tensors.len().div_ceil(DIGEST_SAMPLES).max(1))
    {
        let bytes = tensor.data_range.end - tensor.data_range.start;
        if bytes < DIGEST_SAMPLE_BYTES {
            continue;
        }
        // From the middle of the tensor, where a pruned or padded file is least likely to hold the
        // zeros that two different checkpoints would share.
        let middle = (bytes / 2 - DIGEST_SAMPLE_BYTES / 2) as u64;
        if handle
            .seek(SeekFrom::Start(file.file_offset(tensor) + middle))
            .is_err()
            || handle.read_exact(&mut sample).is_err()
        {
            return layout;
        }
        for byte in sample {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn resolve(models: &Path, role: &str) -> Option<PathBuf> {
    let file = ROLES
        .iter()
        .find(|(name, _)| *name == role)
        .map(|(_, path)| models.join(path))?;
    file.exists().then_some(file)
}

/// One reliable connection to one peer, when both sides have RoCE. Bulk payloads go this way and
/// the socket carries only what asks for them.
#[cfg(feature = "cuda")]
type Rdma = mmh3_rdma::Link;
#[cfg(feature = "cuda")]
type Region = mmh3_rdma::Region;
#[cfg(not(feature = "cuda"))]
type Rdma = ();

/// This machine's RoCE device, or nothing when it has no active port. One device serves every
/// connection of the process, so its memory is registered once however many peers read it.
#[cfg(feature = "cuda")]
fn rdma_device() -> Option<&'static mmh3_rdma::Device> {
    use std::sync::OnceLock;

    static DEVICE: OnceLock<Option<mmh3_rdma::Device>> = OnceLock::new();
    DEVICE
        .get_or_init(|| mmh3_rdma::Device::open("", mmh3_rdma::DEFAULT_GLOBAL_ID).ok())
        .as_ref()
}

/// What this machine does at a GEMM of a DiT's shape and at a copy, which is what the shares of a
/// shared-out step follow. Measured once, since it is a property of the machine and not of a run,
/// and answered with nothing by a backend that cannot measure itself.
pub fn measure_speed() -> Option<worker::Speed> {
    // A machine does not change between runs, so the first one measures and the rest read. The
    // file names the device it was measured on, and another one measures again.
    let cache = crate::models::cache_file("speed.txt");
    let (name, road) = (device_name(), road());
    let kept = cache
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .filter(|text| {
            let mut lines = text.lines();
            lines.next() == Some(SPEED_HEADER) && lines.next() == Some(name.as_str())
        })
        .unwrap_or_default();
    if let Some(speed) = kept.lines().skip(2).find_map(|line| parse_road(line, road)) {
        return Some(speed);
    }

    let speed = measure_now()?;
    if let Some(path) = cache {
        // The roads already measured are kept: a run on one of them should not pay to measure
        // again because a run on another came between.
        let mut text = format!("{SPEED_HEADER}\n{name}\n");
        for line in kept.lines().skip(2) {
            if parse_road(line, road).is_none() && !line.trim().is_empty() {
                text.push_str(line);
                text.push('\n');
            }
        }
        text.push_str(&format!(
            "{road} {} {}\n",
            speed.gemm_tops, speed.bandwidth_gbytes
        ));
        if let Some(directory) = path.parent() {
            let _ = std::fs::create_dir_all(directory);
        }
        if let Err(error) = std::fs::write(&path, text) {
            eprintln!("warning: writing {}: {error}", path.display());
        }
    }
    Some(speed)
}

/// One kept line, when it is the road asked for.
fn parse_road(line: &str, road: &str) -> Option<worker::Speed> {
    let mut fields = line.split_whitespace();
    if fields.next()? != road {
        return None;
    }

    Some(worker::Speed {
        gemm_tops: fields.next()?.parse().ok()?,
        bandwidth_gbytes: fields.next()?.parse().ok()?,
    })
}

/// The precision a block's products run in here, which is what the measurement is of. The same
/// arithmetic on another road is another number, so a kept measurement names the road it took.
fn road() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "bf16"
    }
    #[cfg(feature = "metal")]
    {
        // NOTE: the default of `--linear-precision` on Metal. A run told to take another road is
        // measured against this one, which `Welcome` is sent too early to know about.
        "mps-fp16"
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        "none"
    }
}

/// First line of the file `measure_speed` keeps, followed by the device and then a line for every
/// road measured on it. It goes up when what is measured changes, so an old file is not taken for
/// a new one.
const SPEED_HEADER: &str = "mmh3 machine speed 2";

#[cfg(feature = "cuda")]
fn measure_now() -> Option<worker::Speed> {
    use mmh3_cuda::bench::{GemmKind, gemm, memory_copy};

    // A block's own shape, so the number says something about the work it will be given.
    let (m, n, k) = (4096, 5376, 5376);
    let timing = gemm(GemmKind::Bf16, m, n, k, 10).ok()?;
    let operations = 2.0 * m as f64 * n as f64 * k as f64;
    let copy = memory_copy(1 << 28, 5).ok()?;
    Some(worker::Speed {
        gemm_tops: (operations / (timing.milliseconds as f64 * 1e-3) / 1e12) as f32,
        bandwidth_gbytes: copy.copy_kernel_gigabytes_per_second,
    })
}

#[cfg(feature = "metal")]
fn measure_now() -> Option<worker::Speed> {
    use mmh3_metal::bench::{gemm_operations_per_second, memory_copy_bytes_per_second};

    // The same shape CUDA measures, so the two numbers can be compared and a share cut between
    // them. The road differs and that is the point: each machine times the one it will take.
    let (m, n, k) = (4096, 5376, 5376);
    let device = mmh3_metal::Device::new().ok()?;
    let operations =
        gemm_operations_per_second(&device, mmh3_metal::LinearPrecision::MpsFp16, m, n, k, 10)
            .ok()?;
    let copy = memory_copy_bytes_per_second(&device, 1 << 28, 5).ok()?;
    Some(worker::Speed {
        gemm_tops: (operations / 1e12) as f32,
        bandwidth_gbytes: (copy / 1e9) as f32,
    })
}

/// A backend that cannot measure itself says so, and it is given no share of a step.
#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn measure_now() -> Option<worker::Speed> {
    None
}

/// Opens this machine's side of a connection, with one path per RoCE port, or nothing when it has
/// none.
fn open_rdma() -> Option<(Rdma, Vec<[u8; worker::RDMA_ADDRESS_BYTES]>)> {
    #[cfg(feature = "cuda")]
    {
        let link = rdma_device()?.link().ok()?;
        let addresses = link
            .addresses()
            .ok()?
            .into_iter()
            .map(|address| address.to_bytes())
            .collect();
        Some((link, addresses))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Finishes a connection once the peer's addresses have arrived over a socket.
fn join_rdma(link: Rdma, peer: &[[u8; worker::RDMA_ADDRESS_BYTES]]) -> Option<Rdma> {
    #[cfg(feature = "cuda")]
    {
        let addresses: Option<Vec<mmh3_rdma::Address>> = peer
            .iter()
            .map(|bytes| mmh3_rdma::Address::from_bytes(bytes))
            .collect();
        link.connect(&addresses?).ok()?;
        Some(link)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (link, peer);
        None
    }
}

/// One worker as the leader sees it.
pub struct Worker {
    pub address: String,
    pub welcome: Welcome,
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    request: u64,
    rdma: Option<Rdma>,
    #[cfg(feature = "cuda")]
    staging: Option<Region>,
}

impl Worker {
    /// Connects and exchanges the handshake. The caller decides what to do with a worker that
    /// cannot serve what it wants; nothing here falls back on its own.
    pub fn connect(address: &str, token: &str) -> Result<Self, Box<dyn Error>> {
        let address = self::address(address);
        let target = address
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| format!("{address} resolves to nothing"))?;
        let stream = TcpStream::connect_timeout(&target, Duration::from_secs(10))?;
        stream.set_nodelay(true)?;
        let mut worker = Worker {
            reader: BufReader::with_capacity(STREAM_BUFFER, stream.try_clone()?),
            writer: BufWriter::with_capacity(STREAM_BUFFER, stream),
            welcome: Welcome {
                backend: 0,
                device: String::new(),
                memory_bytes: 0,
                capabilities: 0,
                transports: 0,
                speed: None,
                checkpoints: Vec::new(),
                rdma: None,
            },
            request: 0,
            rdma: None,
            #[cfg(feature = "cuda")]
            staging: None,
            address,
        };
        let offered = open_rdma();
        let hello = Hello {
            leader: hostname(),
            token: token.to_owned(),
            rdma: offered.as_ref().map(|(_, addresses)| addresses.clone()),
        };
        let body = worker.call(Kind::Hello, &hello.encode(), &[])?;
        worker.welcome = Welcome::decode(&body.1)?;
        worker.rdma = match (offered, worker.welcome.rdma.clone()) {
            (Some((connection, _)), Some(peer)) => join_rdma(connection, &peer),
            _ => None,
        };
        Ok(worker)
    }

    /// Whether the bulk of a reply comes over RoCE rather than the socket.
    pub fn reads_remotely(&self) -> bool {
        self.rdma.is_some()
    }

    pub fn serves(&self, capability: u32) -> bool {
        self.welcome.capabilities & capability != 0
    }

    /// What this worker measured of itself, or nothing when its backend cannot measure.
    pub fn speed(&self) -> Option<worker::Speed> {
        self.welcome.speed
    }

    /// The checkpoint this worker holds for `role`, if it holds one.
    pub fn checkpoint(&self, role: &str) -> Option<&Checkpoint> {
        self.welcome
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.role == role)
    }

    pub fn ping(&mut self) -> Result<Duration, Box<dyn Error>> {
        let started = Instant::now();
        self.call(Kind::Ping, &[], &[])?;
        Ok(started.elapsed())
    }

    /// Moves `bytes` and returns the rate in gigabytes a second: over the socket it goes there and
    /// back, over a reliable connection it is read once, which is how the payloads travel.
    pub fn bandwidth(&mut self, bytes: usize) -> Result<(f64, bool), Box<dyn Error>> {
        #[cfg(feature = "cuda")]
        if self.reads_remotely() {
            let request = worker::Canvas {
                chunk: 0,
                bytes: bytes as u64,
                remote_address: 0,
                remote_keys: Vec::new(),
            };
            let started = Instant::now();
            let (header, body) = self.call(Kind::Bandwidth, &request.encode(), &[])?;
            if header.kind != Kind::Canvas {
                return Err(format!(
                    "{} answered a measurement with {:?}",
                    self.address, header.kind
                )
                .into());
            }
            let (offer, _) = worker::Canvas::decode(&body)?;
            self.write_from_staging(&offer)?;
            let seconds = started.elapsed().as_secs_f64();
            return Ok(((bytes as f64 / seconds) / 1e9, true));
        }
        let payload = vec![0u8; bytes];
        let started = Instant::now();
        self.call(Kind::Bandwidth, &[], &payload)?;
        let seconds = started.elapsed().as_secs_f64();
        Ok(((2.0 * bytes as f64 / seconds) / 1e9, false))
    }

    /// The prompt's text states, encoded wherever the text encoder lives.
    pub fn encode_text(&mut self, role: &str, ids: &[u32]) -> Result<Tensor, Box<dyn Error>> {
        let checkpoint = self
            .checkpoint(role)
            .ok_or_else(|| format!("{} has no {role}", self.address))?
            .clone();
        let request = EncodeText {
            checkpoint,
            ids: ids.to_vec(),
        };
        let (header, body) = self.call(Kind::EncodeText, &request.encode(), &[])?;
        if header.kind != Kind::TextStates {
            return Err(
                format!("{} answered a prompt with {:?}", self.address, header.kind).into(),
            );
        }
        let states = TextStates::decode(&body)?;
        let values = &body[TextStates::BYTES..];
        if values.len() != states.tokens * states.hidden * 4 {
            return Err(format!("{} sent {} bytes of states", self.address, values.len()).into());
        }
        Ok(Tensor {
            shape: vec![states.tokens, states.hidden],
            data: values
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
        })
    }

    /// One chunk's canvas, decoded wherever the video VAE lives.
    /// The waveform of a run's audio latent, decoded on this worker. The latent and the samples
    /// are a megabyte or so each, so they go down the socket.
    pub fn decode_audio(&mut self, role: &str, latent: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        let checkpoint = self
            .checkpoint(role)
            .ok_or_else(|| format!("{} has no {role}", self.address))?
            .clone();
        let &[channels, sequences, frames] = latent.shape.as_slice() else {
            return Err(format!("latent shape {:?} is not three dimensions", latent.shape).into());
        };
        let request = DecodeAudio {
            checkpoint,
            shape: [channels as u32, sequences as u32, frames as u32],
        };
        let payload: Vec<u8> = latent
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let (header, body) = self.call(Kind::DecodeAudio, &request.encode(), &payload)?;
        if header.kind != Kind::Samples {
            return Err(
                format!("{} answered the audio with {:?}", self.address, header.kind).into(),
            );
        }
        let (samples, descriptor_bytes) = Samples::decode(&body)?;
        let shape: Vec<usize> = samples
            .shape
            .iter()
            .map(|&extent| extent as usize)
            .collect();
        let values = &body[descriptor_bytes..];
        if values.len() != shape.iter().product::<usize>() * 4 {
            return Err(format!("{} sent {} bytes of audio", self.address, values.len()).into());
        }
        Ok(Tensor::new(
            shape,
            values
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect(),
        ))
    }

    pub fn decode_chunk(
        &mut self,
        role: &str,
        latent: &Tensor,
        chunk: usize,
        tile_size: usize,
        tile_overlap: usize,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        let checkpoint = self
            .checkpoint(role)
            .ok_or_else(|| format!("{} has no {role}", self.address))?
            .clone();
        let &[channels, frames, height, width] = latent.shape.as_slice() else {
            return Err(format!("latent shape {:?} is not four dimensions", latent.shape).into());
        };
        let request = DecodeVideo {
            checkpoint,
            chunk: chunk as u32,
            tile_size: tile_size as u32,
            tile_overlap: tile_overlap as u32,
            shape: [channels as u32, frames as u32, height as u32, width as u32],
        };
        let payload: Vec<u8> = latent
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let (header, body) = self.call(Kind::DecodeVideo, &request.encode(), &payload)?;
        if header.kind != Kind::Canvas {
            return Err(format!("{} answered a chunk with {:?}", self.address, header.kind).into());
        }
        let (canvas, descriptor_bytes) = Canvas::decode(&body)?;
        if !canvas.remote_keys.is_empty() {
            return self.read_canvas(&canvas);
        }
        let payload = &body[descriptor_bytes..];
        if payload.len() as u64 != canvas.bytes {
            return Err(format!("{} sent {} bytes of canvas", self.address, payload.len()).into());
        }
        Ok(payload.to_vec())
    }

    /// Writes this side's memory into the worker's and tells it the memory is free, which is the
    /// way a shared-out step moves a payload.
    #[cfg(feature = "cuda")]
    fn write_from_staging(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        let link = self
            .rdma
            .as_ref()
            .ok_or_else(|| format!("{} pointed at memory with no connection", self.address))?;
        let device = rdma_device().ok_or("this machine has no RoCE port")?;
        let bytes = canvas.bytes as usize;
        if self.staging.as_ref().map(Region::bytes).unwrap_or(0) < bytes {
            self.staging = None;
            self.staging = Some(device.register(vec![0u8; bytes])?);
        }
        let region = self.staging.as_ref().expect("just registered");
        region.write_out(
            link,
            0,
            canvas.remote_address,
            &canvas.remote_keys,
            bytes,
            30_000,
        )?;
        self.request += 1;
        worker::send(&mut self.writer, Kind::Release, self.request, &[], &[])?;
        Ok(())
    }

    /// Reads the worker's memory into this side's, and tells it the memory is free. The bytes stay
    /// in the registered buffer, which the caller copies out of if it needs them.
    #[cfg(feature = "cuda")]
    fn read_into_staging(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        let link = self
            .rdma
            .as_ref()
            .ok_or_else(|| format!("{} pointed at memory with no connection", self.address))?;
        let device = rdma_device().ok_or("this machine has no RoCE port")?;
        let bytes = canvas.bytes as usize;
        // One registered buffer serves every transfer of a run, since registering it costs tens of
        // milliseconds for every hundred megabytes. It only ever grows: a smaller payload reads
        // into the front of it.
        if self.staging.as_ref().map(Region::bytes).unwrap_or(0) < bytes {
            self.staging = None;
            self.staging = Some(device.register(vec![0u8; bytes])?);
        }
        let region = self.staging.as_mut().expect("just registered");
        region.read_from(
            link,
            canvas.remote_address,
            &canvas.remote_keys,
            bytes,
            30_000,
        )?;
        self.request += 1;
        worker::send(&mut self.writer, Kind::Release, self.request, &[], &[])?;
        Ok(())
    }

    /// A canvas read out of the worker's memory.
    #[cfg(feature = "cuda")]
    fn read_canvas(&mut self, canvas: &Canvas) -> Result<Vec<u8>, Box<dyn Error>> {
        self.read_into_staging(canvas)?;
        Ok(self
            .staging
            .as_ref()
            .expect("read into it")
            .as_slice()
            .to_vec())
    }

    #[cfg(not(feature = "cuda"))]
    fn read_canvas(&mut self, _canvas: &Canvas) -> Result<Vec<u8>, Box<dyn Error>> {
        Err("this build reads canvases over the socket only".into())
    }

    /// Tells a worker to read the DiT a run is about to share. It is a notification: the answer
    /// would only be that the read has finished, and the point is not to wait for it.
    pub fn prepare(&mut self, checkpoint: &Checkpoint) -> Result<(), Box<dyn Error>> {
        worker::send(
            &mut self.writer,
            Kind::PrepareCheckpoint,
            0,
            &checkpoint.encode(),
            &[],
        )?;
        Ok(())
    }

    fn call(
        &mut self,
        kind: Kind,
        descriptor: &[u8],
        payload: &[u8],
    ) -> Result<(Header, Vec<u8>), Box<dyn Error>> {
        self.request += 1;
        let request = self.request;
        worker::send(&mut self.writer, kind, request, descriptor, payload)?;
        let (header, body) = worker::receive(&mut self.reader, BODY_LIMIT)?;
        if header.kind == Kind::Error {
            return Err(format!("{}: {}", self.address, String::from_utf8_lossy(&body)).into());
        }
        if header.request != request {
            return Err(format!("{} replied to request {}", self.address, header.request).into());
        }
        Ok((header, body))
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|name| name.trim().to_owned())
        .unwrap_or_else(|_| "leader".to_owned())
}

/// Connects to every address, keeping the ones that answer. A worker that refuses or fails is
/// reported and left out: a generation never depends on one.
pub fn connect_all(addresses: &[&str], token: &str) -> Vec<Worker> {
    let mut workers = Vec::new();
    for address in addresses {
        match Worker::connect(address, token) {
            Ok(worker) => {
                println!(
                    "worker {} on {} with {} GiB",
                    worker.address,
                    worker.welcome.device,
                    worker.welcome.memory_bytes >> 30
                );
                workers.push(worker);
            }
            Err(error) => eprintln!("warning: worker {address}: {error}"),
        }
    }
    workers
}

/// Asks a worker what it is and how fast the link to it runs, the operator's first question.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn probe(address: &str, token: &str) -> Result<(), Box<dyn Error>> {
    let mut worker = Worker::connect(address, token)?;
    let backend = match worker.welcome.backend {
        worker::BACKEND_CUDA => "cuda",
        worker::BACKEND_METAL => "metal",
        _ => "unknown",
    };
    println!(
        "{} is {} on {}, {} GiB, capabilities {:#06b}, transports {:#04b}",
        worker.address,
        backend,
        worker.welcome.device,
        worker.welcome.memory_bytes >> 30,
        worker.welcome.capabilities,
        worker.welcome.transports
    );
    for checkpoint in &worker.welcome.checkpoints {
        println!("  {} {:016x}", checkpoint.role, checkpoint.digest);
    }
    let mut round_trip = Duration::from_secs(1);
    for _ in 0..5 {
        round_trip = round_trip.min(worker.ping()?);
    }
    println!("round trip {:.3} ms", round_trip.as_secs_f64() * 1e3);
    for megabytes in [1usize, 64, 512] {
        // The first call of a size registers the memory, which a run pays once and a measurement
        // should not count at all.
        worker.bandwidth(megabytes << 20)?;
        let (gigabytes, remote) = worker.bandwidth(megabytes << 20)?;
        let how = if remote { "written" } else { "there and back" };
        println!("{megabytes} MiB {how} at {gigabytes:.2} GB/s");
    }
    exchange(&mut worker)?;
    Ok(())
}

/// One DiT step's worth of Ulysses traffic, to see whether the link carries a shard before anything
/// is built on it. Each layer exchanges twice, and the rendezvous happens as often as the payload,
/// which a single large read does not show.
fn exchange(worker: &mut Worker) -> Result<(), Box<dyn Error>> {
    // A 768p step: 31k tokens of 56 heads by 128 in bf16, halved by token and by head. The first
    // leg carries q, k and v, the second carries the attention output.
    const LAYERS: usize = 50;
    const LEGS: [usize; 2] = [336 << 20, 112 << 20];

    for bytes in LEGS {
        worker.bandwidth(bytes)?;
    }
    let started = Instant::now();
    for _ in 0..LAYERS {
        for bytes in LEGS {
            worker.bandwidth(bytes)?;
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let moved = (LAYERS * LEGS.iter().sum::<usize>()) as f64;
    println!(
        "a step's exchange, {} transfers of {:.1} GB, in {seconds:.2} s at {:.2} GB/s",
        LAYERS * LEGS.len(),
        moved / 1e9,
        (moved / seconds) / 1e9
    );
    Ok(())
}

/// Chunks of a decode that other machines are working on. The leader asks for one when its own
/// walk reaches it, and decodes it here if the machine that had it failed.
pub struct RemoteCanvases {
    receiver: Receiver<(usize, Result<Vec<u8>, String>)>,
    delegated: Vec<usize>,
    arrived: HashMap<usize, Vec<u8>>,
}

/// A run's audio decoded on a worker while this machine decodes the video. The workers finish
/// their chunks before the leader finishes its own, so the waveform costs nothing but the asking.
pub struct RemoteAudio {
    receiver: Receiver<Option<Tensor>>,
}

impl RemoteAudio {
    /// Asks the first worker that can decode audio and holds the answer until `take`. The thread
    /// connects as well, so even that is off this machine's path.
    pub fn start(addresses: Vec<String>, token: String, role: &str, latent: &Tensor) -> Self {
        let (sender, receiver) = channel();
        let (role, latent) = (role.to_owned(), latent.clone());
        std::thread::spawn(move || {
            let mut candidates: Vec<Worker> = Vec::new();
            for address in &addresses {
                let Ok(worker) = Worker::connect(address, &token) else {
                    continue;
                };
                if worker.serves(CAPABILITY_DECODE_AUDIO) && worker.checkpoint(&role).is_some() {
                    candidates.push(worker);
                }
            }
            // A machine that takes a share of every step has the busier end of a run, and one that
            // cannot is often another backend with nothing else to do, so the audio goes there
            // first. The sort is stable, so the order given decides within each group.
            candidates.sort_by_key(|worker| worker.serves(CAPABILITY_DIT_SHARD));
            let mut answer = None;
            for worker in &mut candidates {
                match worker.decode_audio(&role, &latent) {
                    Ok(waveform) => {
                        answer = Some(waveform);
                        break;
                    }
                    Err(error) => eprintln!("warning: the audio on {}: {error}", worker.address),
                }
            }
            let _ = sender.send(answer);
        });
        RemoteAudio { receiver }
    }

    /// The waveform a worker decoded, or `None` when this machine should decode it after all.
    pub fn take(self) -> Option<Tensor> {
        self.receiver.recv().ok().flatten()
    }
}

impl RemoteCanvases {
    /// Hands the chunks out over the workers that serve decodes, keeping the first share for the
    /// leader, and starts them at once. `role` is the video VAE both sides must agree on.
    pub fn start(
        workers: Vec<Worker>,
        role: &str,
        latent: &Tensor,
        chunks: usize,
        tile_size: usize,
        tile_overlap: usize,
    ) -> Self {
        let workers: Vec<Worker> = workers
            .into_iter()
            .filter(|worker| {
                worker.serves(CAPABILITY_DECODE_VIDEO) && worker.checkpoint(role).is_some()
            })
            .collect();
        let (sender, receiver) = channel();
        let mut delegated = Vec::new();
        if workers.is_empty() || chunks < 2 {
            return RemoteCanvases {
                receiver,
                delegated,
                arrived: HashMap::new(),
            };
        }
        // One share each, the leader included. The leader keeps the first chunks and the workers
        // take the last ones, so this machine walks its own share while theirs is still coming and
        // never waits at the start. The remainder stays here: a worker's chunk has to cross the
        // wire and be blended in after it lands, and giving it the extra one costs 1.5 s.
        let shares = workers.len() + 1;
        let latent = Arc::new(latent.clone());
        let mut first = chunks.div_ceil(shares);
        for mut worker in workers {
            let share = (chunks - first).div_ceil(shares - 1).min(chunks - first);
            let mine: Vec<usize> = (first..first + share).collect();
            first += share;
            if mine.is_empty() {
                continue;
            }
            delegated.extend(mine.iter().copied());
            let sender = sender.clone();
            let latent = Arc::clone(&latent);
            let role = role.to_owned();
            std::thread::spawn(move || {
                for chunk in mine {
                    let result = worker
                        .decode_chunk(&role, &latent, chunk, tile_size, tile_overlap)
                        .map_err(|error| error.to_string());
                    let failed = result.is_err();
                    if sender.send((chunk, result)).is_err() || failed {
                        // The leader decodes what is left of this worker's share itself.
                        return;
                    }
                }
            });
        }
        RemoteCanvases {
            receiver,
            delegated,
            arrived: HashMap::new(),
        }
    }

    pub fn chunks(&self) -> &[usize] {
        &self.delegated
    }

    /// The canvas of `chunk` when another machine has it, waiting for it if it is still coming, and
    /// `None` when this machine should decode it after all.
    pub fn take(&mut self, chunk: usize) -> Option<Vec<u8>> {
        if !self.delegated.contains(&chunk) {
            return None;
        }
        loop {
            if let Some(values) = self.arrived.remove(&chunk) {
                return Some(values);
            }
            match self.receiver.recv() {
                Ok((arrived, Ok(values))) => {
                    self.arrived.insert(arrived, values);
                }
                Ok((arrived, Err(error))) => {
                    eprintln!("warning: chunk {arrived} comes back here: {error}");
                    self.delegated.retain(|other| *other != arrived);
                    if arrived == chunk {
                        return None;
                    }
                }
                // Every worker is gone, so the rest is this machine's.
                Err(_) => {
                    self.delegated.retain(|other| *other != chunk);
                    return None;
                }
            }
        }
    }
}

/// `mmh3 worker --listen ADDR [--models DIR] [--token FILE]`, which serves until it is stopped.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn serve(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use crate::cli::parse_options;
    use crate::models::models_directory;

    const USAGE: &str = "usage: mmh3 worker [--listen ADDR] [--models DIR] [--token FILE] [--transport auto|socket] | mmh3 worker --probe HOST[:PORT]";
    let options = parse_options(
        arguments,
        &["listen", "models", "token", "probe", "transport"],
        USAGE,
    )?;
    // `--transport socket` keeps this machine's port to itself, so that a leader with one
    // exchanges over the socket as it would with a machine that has none. It is how the socket
    // path is tried between two machines that both have ports.
    let bare = match options.get("transport").copied().unwrap_or("auto") {
        "auto" => false,
        "socket" => true,
        other => return Err(format!("--transport must be auto or socket, not {other}").into()),
    };
    let token = match options.get("token") {
        Some(path) => std::fs::read_to_string(path)?.trim().to_owned(),
        None => String::new(),
    };
    if let Some(address) = options.get("probe") {
        return probe(address, &token);
    }
    let listen = options
        .get("listen")
        .map_or_else(|| format!("0.0.0.0:{DEFAULT_PORT}"), |value| address(value));
    let models = models_directory(&options)
        .ok_or("pass a models directory with --models DIR or MMH3_MODELS")?;

    let checkpoints: Vec<(Checkpoint, PathBuf)> = ROLES
        .iter()
        .filter_map(|(role, _)| {
            let path = resolve(&models, role)?;
            let file = SafeTensors::open(&path).ok()?;
            Some((
                Checkpoint {
                    role: (*role).to_owned(),
                    digest: digest(&file),
                },
                path,
            ))
        })
        .collect();
    #[cfg(feature = "cuda")]
    crate::models::load_algorithm_cache();
    let speed = measure_speed();
    println!("serving {} on {listen}", models.display());
    if let Some(speed) = speed {
        println!(
            "  {:.1} TOPS at a block's GEMM, {:.1} GB/s copying",
            speed.gemm_tops, speed.bandwidth_gbytes
        );
    }
    for (checkpoint, path) in &checkpoints {
        println!(
            "  {} {:016x} {}",
            checkpoint.role,
            checkpoint.digest,
            path.display()
        );
    }

    let listener = TcpListener::bind(&listen)?;
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("warning: accept: {error}");
                continue;
            }
        };
        let checkpoints = checkpoints.clone();
        let token = token.clone();
        let models = models.clone();
        std::thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map_or_else(|_| "?".to_owned(), |address| address.to_string());
            if let Err(error) = session(stream, &models, &checkpoints, &token, bare, speed) {
                eprintln!("worker session with {peer} ended: {error}");
            }
            #[cfg(feature = "cuda")]
            crate::models::save_algorithm_cache();
        });
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn session(
    stream: TcpStream,
    models: &Path,
    checkpoints: &[(Checkpoint, PathBuf)],
    token: &str,
    bare: bool,
    speed: Option<worker::Speed>,
) -> Result<(), Box<dyn Error>> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::with_capacity(STREAM_BUFFER, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER, stream);
    // The text encoder and the video decoder load on their first request and stay for the session.
    let mut encoder: Resident = None;
    let mut decoder: ResidentDecoder = None;
    let mut audio: ResidentAudio = None;
    // A DiT read before it was asked for, which `PrepareCheckpoint` starts while the leader is
    // still encoding its prompt.
    #[cfg(feature = "cuda")]
    let mut prepared: Option<(u64, mmh3_cuda::dit::CudaDit)> = None;
    #[cfg(feature = "cuda")]
    let mut rdma: Option<Rdma> = None;
    #[cfg(feature = "cuda")]
    let mut staging: Option<Region> = None;

    loop {
        let (header, body) = match worker::receive(&mut reader, BODY_LIMIT) {
            Ok(message) => message,
            // A leader that closed the connection is the normal end of a session.
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let reply = |writer: &mut BufWriter<TcpStream>, kind, descriptor: &[u8], payload: &[u8]| {
            worker::send(writer, kind, header.request, descriptor, payload)
        };
        match header.kind {
            Kind::Hello => {
                let hello = Hello::decode(&body)?;
                if hello.token != token {
                    let message = b"the token does not match";
                    reply(&mut writer, Kind::Error, message, &[])?;
                    return Err("a leader offered the wrong token".into());
                }
                // A leader that offers a reliable connection gets this side's, and the bulk of
                // every later reply goes that way instead of down the socket.
                let offered = hello
                    .rdma
                    .as_ref()
                    .filter(|_| !bare)
                    .and_then(|_| open_rdma());
                let welcome = Welcome {
                    backend: BACKEND,
                    device: device_name(),
                    memory_bytes: memory_bytes(),
                    speed,
                    capabilities: CAPABILITY_ENCODE_TEXT
                        | CAPABILITY_DECODE_AUDIO
                        | CAPABILITY_DECODE_VIDEO
                        // A share of a step goes over the connection where the path has one and
                        // down the socket where it has not, so what it asks for is the weights.
                        | if models.join("diffusion_models").is_dir() {
                            CAPABILITY_DIT_SHARD
                        } else {
                            0
                        },
                    transports: TRANSPORT_TCP
                        | if offered.is_some() {
                            worker::TRANSPORT_RDMA
                        } else {
                            0
                        },
                    checkpoints: checkpoints
                        .iter()
                        .map(|(checkpoint, _)| checkpoint.clone())
                        .collect(),
                    rdma: offered.as_ref().map(|(_, addresses)| addresses.clone()),
                };
                reply(&mut writer, Kind::Welcome, &welcome.encode(), &[])?;
                #[cfg(feature = "cuda")]
                {
                    rdma = match (offered, hello.rdma) {
                        (Some((connection, _)), Some(peer)) => join_rdma(connection, &peer),
                        _ => None,
                    };
                }
            }
            Kind::Ping => reply(&mut writer, Kind::Pong, &[], &[])?,
            // A measurement moves the same way a payload would: read out of this side's memory
            // when there is a reliable connection, echoed down the socket when there is not.
            #[cfg(not(feature = "cuda"))]
            Kind::Bandwidth => reply(&mut writer, Kind::BandwidthDone, &[], &body)?,
            #[cfg(feature = "cuda")]
            Kind::Bandwidth => match (rdma.as_ref(), worker::Canvas::decode(&body)) {
                (Some(_), Ok((request, _))) if !body.is_empty() => {
                    let device = rdma_device().ok_or("this machine has no RoCE port")?;
                    let bytes = request.bytes as usize;
                    if staging.as_ref().map(Region::bytes).unwrap_or(0) < bytes {
                        // NOTE: the old region unregisters before the new one registers, so the
                        // two never hold the same memory at once.
                        drop(staging.take());
                        staging = Some(device.register(vec![0u8; bytes])?);
                    }
                    let region = staging.as_ref().expect("just registered");
                    let offer = worker::Canvas {
                        chunk: 0,
                        bytes: request.bytes,
                        remote_address: region.address(),
                        remote_keys: region.remote_keys().to_vec(),
                    };
                    reply(&mut writer, Kind::Canvas, &offer.encode(), &[])?;
                    let (release, _) = worker::receive(&mut reader, BODY_LIMIT)?;
                    if release.kind != Kind::Release {
                        return Err(format!(
                            "a leader sent {:?} while memory waited",
                            release.kind
                        )
                        .into());
                    }
                }
                _ => reply(&mut writer, Kind::BandwidthDone, &[], &body)?,
            },
            Kind::EncodeText => {
                let request = EncodeText::decode(&body)?;
                match encode_text(checkpoints, &request, &mut encoder) {
                    Ok(context) => {
                        let states = TextStates {
                            tokens: context.shape[0],
                            hidden: context.shape[1],
                        };
                        let payload: Vec<u8> = context
                            .data
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect();
                        reply(&mut writer, Kind::TextStates, &states.encode(), &payload)?;
                    }
                    Err(error) => {
                        reply(&mut writer, Kind::Error, error.to_string().as_bytes(), &[])?
                    }
                }
            }
            // A shared-out step takes the connection over until the leader closes it, since
            // both ranks then barrier and read each other rather than trading requests.
            #[cfg(feature = "cuda")]
            Kind::PrepareCheckpoint => {
                // A notification rather than a request: the leader has a prompt to encode and does
                // not wait for the read, which is the whole point of sending it early.
                #[cfg(feature = "cuda")]
                match worker::Checkpoint::decode(&body) {
                    Ok(checkpoint) => prepared = prepare_dit(models, &checkpoint),
                    Err(error) => eprintln!("warning: a prepared checkpoint: {error}"),
                }
            }
            #[cfg(not(feature = "cuda"))]
            Kind::OpenSession => {
                let message = b"this build takes no share of a step";
                reply(&mut writer, Kind::Error, message, &[])?;
            }
            #[cfg(feature = "cuda")]
            Kind::OpenSession => {
                let (open, payload) = OpenSession::decode(&body)?;
                if let Err(error) = serve_shard(
                    &mut reader,
                    &mut writer,
                    rdma.as_ref(),
                    models,
                    &open,
                    &body[payload..],
                    prepared.take(),
                ) {
                    reply(&mut writer, Kind::Error, error.to_string().as_bytes(), &[])?;
                    return Err(error);
                }
            }
            Kind::DecodeAudio => {
                let started = Instant::now();
                match decode_audio(checkpoints, &body, &mut audio) {
                    Ok(waveform) => {
                        let shape = waveform.shape.iter().map(|&extent| extent as u32).collect();
                        let payload: Vec<u8> = waveform
                            .data
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect();
                        println!(
                            "decoded the audio in {:.1} s",
                            started.elapsed().as_secs_f64()
                        );
                        reply(
                            &mut writer,
                            Kind::Samples,
                            &Samples { shape }.encode(),
                            &payload,
                        )?;
                    }
                    Err(error) => {
                        reply(&mut writer, Kind::Error, error.to_string().as_bytes(), &[])?;
                    }
                }
            }
            Kind::DecodeVideo => {
                let started = Instant::now();
                match decode_video(checkpoints, &body, &mut decoder) {
                    Ok((chunk, payload)) => {
                        let megabytes = payload.len() >> 20;
                        // A reliable connection has the leader read the canvas out of this side's
                        // memory. Only the CUDA build opens one.
                        #[cfg(feature = "cuda")]
                        if rdma.is_some() {
                            // One registered canvas serves the session. Registering hundreds of
                            // megabytes costs seconds, so every chunk copies into the same memory,
                            // which only ever grows.
                            let device = match rdma_device() {
                                Some(device) => device,
                                None => {
                                    let message = b"this machine has no RoCE port";
                                    reply(&mut writer, Kind::Error, message, &[])?;
                                    continue;
                                }
                            };
                            if staging.as_ref().map(Region::bytes).unwrap_or(0) < payload.len() {
                                staging = None;
                                match device.register(vec![0u8; payload.len()]) {
                                    Ok(region) => staging = Some(region),
                                    Err(error) => {
                                        let message = format!("registering a canvas: {error}");
                                        reply(&mut writer, Kind::Error, message.as_bytes(), &[])?;
                                        continue;
                                    }
                                }
                            }
                            let region = staging.as_mut().expect("just registered");
                            region.as_mut_slice()[..payload.len()].copy_from_slice(&payload);
                            let canvas = Canvas {
                                chunk,
                                bytes: payload.len() as u64,
                                remote_address: region.address(),
                                remote_keys: region.remote_keys().to_vec(),
                            };
                            println!(
                                "decoded chunk {chunk} in {:.1} s, {megabytes} MiB to read",
                                started.elapsed().as_secs_f64()
                            );
                            reply(&mut writer, Kind::Canvas, &canvas.encode(), &[])?;
                            // The memory holds this canvas until the leader says it has read it.
                            let (release, _) = worker::receive(&mut reader, BODY_LIMIT)?;
                            if release.kind != Kind::Release {
                                return Err(format!(
                                    "a leader sent {:?} while a canvas waited",
                                    release.kind
                                )
                                .into());
                            }
                            continue;
                        }
                        // Without one the canvas goes down the socket.
                        let canvas = Canvas {
                            chunk,
                            bytes: payload.len() as u64,
                            remote_address: 0,
                            remote_keys: Vec::new(),
                        };
                        println!(
                            "decoded chunk {chunk} in {:.1} s, {megabytes} MiB back",
                            started.elapsed().as_secs_f64()
                        );
                        reply(&mut writer, Kind::Canvas, &canvas.encode(), &payload)?;
                    }
                    Err(error) => {
                        reply(&mut writer, Kind::Error, error.to_string().as_bytes(), &[])?
                    }
                }
            }
            other => {
                let message = format!("a worker does not serve {other:?}");
                reply(&mut writer, Kind::Error, message.as_bytes(), &[])?;
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn encode_text(
    checkpoints: &[(Checkpoint, PathBuf)],
    request: &EncodeText,
    resident: &mut Resident,
) -> Result<Tensor, Box<dyn Error>> {
    let (checkpoint, path) = checkpoints
        .iter()
        .find(|(checkpoint, _)| {
            checkpoint.role == request.checkpoint.role
                && (request.checkpoint.digest == 0
                    || request.checkpoint.digest == checkpoint.digest)
        })
        .ok_or_else(|| {
            format!(
                "no checkpoint for {} {:016x}",
                request.checkpoint.role, request.checkpoint.digest
            )
        })?;
    if resident.as_ref().map(|(digest, _)| *digest) != Some(checkpoint.digest) {
        let started = Instant::now();
        let file = SafeTensors::open(path)?;
        *resident = Some((checkpoint.digest, TextEncoder::load(&file)?));
        println!(
            "loaded {} in {:.1} s",
            path.display(),
            started.elapsed().as_secs_f64()
        );
    }
    let encoder = &resident.as_ref().expect("just loaded").1;
    let started = Instant::now();
    let context = encoder.encode(&request.ids, &[])?.context;
    println!(
        "encoded {} tokens in {:.1} s",
        context.shape[0],
        started.elapsed().as_secs_f64()
    );
    Ok(context)
}

/// Decodes one chunk of a video for a leader, loading the video VAE if this session has not.
/// Decodes a run's audio latent, keeping the VAE for the rest of the session. The latent and the
/// waveform are a megabyte or so, so they go down the socket rather than over the wire.
#[cfg(any(feature = "cuda", feature = "metal"))]
fn decode_audio(
    checkpoints: &[(Checkpoint, PathBuf)],
    body: &[u8],
    resident: &mut ResidentAudio,
) -> Result<Tensor, Box<dyn Error>> {
    let (request, descriptor_bytes) = DecodeAudio::decode(body)?;
    let (checkpoint, path) = checkpoints
        .iter()
        .find(|(checkpoint, _)| {
            checkpoint.role == request.checkpoint.role
                && (request.checkpoint.digest == 0
                    || request.checkpoint.digest == checkpoint.digest)
        })
        .ok_or_else(|| format!("no checkpoint for {}", request.checkpoint.role))?;
    if resident.as_ref().map(|(digest, _)| *digest) != Some(checkpoint.digest) {
        let started = Instant::now();
        let file = SafeTensors::open(path)?;
        *resident = Some((checkpoint.digest, AudioDecoder::load(&file, "")?));
        println!(
            "loaded {} in {:.1} s",
            path.display(),
            started.elapsed().as_secs_f64()
        );
    }
    let shape: Vec<usize> = request
        .shape
        .iter()
        .map(|&extent| extent as usize)
        .collect();
    let values = &body[descriptor_bytes..];
    let expected = shape.iter().product::<usize>() * 4;
    if values.len() != expected {
        return Err(format!("a latent of {} bytes, not {expected}", values.len()).into());
    }
    let latent = Tensor::new(
        shape,
        values
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    );
    let decoder = &resident.as_ref().expect("just loaded").1;
    Ok(decoder.decode(&latent)?)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn decode_video(
    checkpoints: &[(Checkpoint, PathBuf)],
    body: &[u8],
    resident: &mut ResidentDecoder,
) -> Result<(u32, Vec<u8>), Box<dyn Error>> {
    let (request, descriptor_bytes) = DecodeVideo::decode(body)?;
    let (checkpoint, path) = checkpoints
        .iter()
        .find(|(checkpoint, _)| {
            checkpoint.role == request.checkpoint.role
                && (request.checkpoint.digest == 0
                    || request.checkpoint.digest == checkpoint.digest)
        })
        .ok_or_else(|| format!("no checkpoint for {}", request.checkpoint.role))?;
    let (tile_size, tile_overlap) = (request.tile_size as usize, request.tile_overlap as usize);
    if resident
        .as_ref()
        .map(|(digest, size, overlap, _)| (*digest, *size, *overlap))
        != Some((checkpoint.digest, tile_size, tile_overlap))
    {
        let started = Instant::now();
        let file = SafeTensors::open(path)?;
        *resident = Some((
            checkpoint.digest,
            tile_size,
            tile_overlap,
            VideoDecoder::load(&file, "", tile_size, tile_overlap)?,
        ));
        println!(
            "loaded {} in {:.1} s",
            path.display(),
            started.elapsed().as_secs_f64()
        );
    }
    let shape: Vec<usize> = request
        .shape
        .iter()
        .map(|&extent| extent as usize)
        .collect();
    let values = &body[descriptor_bytes..];
    let expected = shape.iter().product::<usize>() * 4;
    if values.len() != expected {
        return Err(format!("a latent of {} bytes, not {expected}", values.len()).into());
    }
    let latent = Tensor::new(
        shape,
        values
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    );
    let decoder = &resident.as_ref().expect("just loaded").3;
    Ok((
        request.chunk,
        decoder.decode_chunk(&latent, request.chunk as usize)?,
    ))
}

fn device_name() -> String {
    #[cfg(feature = "cuda")]
    {
        mmh3_cuda::device_info(0).map_or_else(|_| "CUDA".to_owned(), |info| info.name)
    }
    #[cfg(feature = "metal")]
    {
        "Apple GPU".to_owned()
    }
}

fn memory_bytes() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    text.lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(0, |kibibytes| kibibytes * 1024)
}

/// One rank's socket to another: every peer for the leader, the leader alone for a worker, since
/// the barriers go through it and the workers never talk to each other over TCP.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct Socket<'a> {
    reader: &'a mut BufReader<TcpStream>,
    writer: &'a mut BufWriter<TcpStream>,
}

/// What one rank brings to a shared-out run: its socket to the leader or, for the leader, to each
/// worker, and the connection the payloads travel over.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct Peer<'a> {
    pub rank: usize,
    /// Which backend this peer runs, as `Welcome` named it.
    pub backend: u8,
    reader: &'a mut BufReader<TcpStream>,
    writer: &'a mut BufWriter<TcpStream>,
    /// The connection to this peer, where the path between the two machines has one and this
    /// backend can use it.
    #[cfg(feature = "cuda")]
    link: Option<&'a Rdma>,
}

/// What a block addresses a region by: a pointer where this backend reaches its memory by one.
#[cfg(feature = "cuda")]
type BlockMemory = *mut std::ffi::c_void;

/// Where a rank's blocks compute, beside the host memory its peers write into. This is the whole
/// of what an exchange does differently from one backend to the next: the sockets above it, the
/// queueing, the order two ranks send in and the barriers are the same everywhere.
///
/// A backend that cannot compute in the memory a peer writes to keeps its own beside each region
/// and copies between the two. One that can answers with that memory and copies nothing.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct Computed(HashMap<shard::Region, mmh3_cuda::DeviceBuffer>);

#[cfg(feature = "cuda")]
impl Computed {
    /// Memory for a region a block computes in, kept for the run. Regions a block only exchanges
    /// through are not given any: they are read and written where the peer left them.
    fn make(&mut self, region: shard::Region, bytes: usize) -> Result<(), Box<dyn Error>> {
        if region.is_computed_in() {
            self.0
                .insert(region, mmh3_cuda::DeviceBuffer::zeroed(bytes)?);
        }
        Ok(())
    }

    /// Whether this backend holds memory of its own for `region`, and so has something to copy.
    fn holds(&self, region: shard::Region) -> bool {
        self.0.contains_key(&region)
    }

    /// What a block addresses `region` by. `held` is the host memory a peer writes into, which a
    /// backend with nothing of its own beside it answers with.
    fn memory(&self, region: shard::Region, held: &mut [u8]) -> BlockMemory {
        match self.0.get(&region) {
            Some(buffer) => buffer.pointer(),
            None => held.as_mut_ptr().cast(),
        }
    }

    /// Copies what a block computed in `region` into `into`, which is what a peer reads.
    fn download(
        &self,
        region: shard::Region,
        offset: usize,
        into: &mut [u8],
    ) -> Result<(), shard::ExchangeError> {
        let Some(buffer) = self.0.get(&region) else {
            return Ok(());
        };
        // SAFETY: this buffer is as long as the held one, so the same range fits both.
        unsafe { mmh3_cuda::download(into, buffer.pointer().byte_add(offset)) }
            .map_err(|error| exchange_error(format!("taking {region:?} off the device: {error}")))
    }

    /// Copies what a peer wrote into the memory a block computes in.
    fn upload(
        &self,
        region: shard::Region,
        offset: usize,
        from: &[u8],
    ) -> Result<(), shard::ExchangeError> {
        let Some(buffer) = self.0.get(&region) else {
            return Ok(());
        };
        // SAFETY: as `download`.
        unsafe { mmh3_cuda::upload(buffer.pointer().byte_add(offset), from) }
            .map_err(|error| exchange_error(format!("putting {region:?} on the device: {error}")))
    }
}

/// What a block addresses a region by on Metal: the memory itself, since a Metal buffer belongs
/// to the context that made it and a host pointer is not one.
#[cfg(feature = "metal")]
type BlockMemory = mmh3_metal::shard::Memory;

/// Where a rank's blocks compute, beside the host memory its peers write into.
///
/// Metal keeps its own memory for EVERY region rather than only the computed-in ones, and copies
/// for all of them. It has no choice: a block cannot compute in the host buffer a peer writes to,
/// because a buffer belongs to the context that made it and `held` is plain memory. So `memory`
/// has no fallback to give, and the invariant that every registered region was made at session
/// open is what holds it up rather than a convenience.
#[cfg(feature = "metal")]
#[derive(Default)]
struct Computed(HashMap<shard::Region, mmh3_metal::shard::Memory>);

#[cfg(feature = "metal")]
impl Computed {
    fn make(&mut self, region: shard::Region, bytes: usize) -> Result<(), Box<dyn Error>> {
        let device = mmh3_metal::Device::shared()?;
        self.0
            .insert(region, mmh3_metal::shard::Memory::zeroed(&device, bytes)?);
        Ok(())
    }

    fn holds(&self, region: shard::Region) -> bool {
        self.0.contains_key(&region)
    }

    fn memory(&self, region: shard::Region, _held: &mut [u8]) -> BlockMemory {
        self.0
            .get(&region)
            .cloned()
            .expect("every registered region is made when the session opens")
    }

    fn download(
        &self,
        region: shard::Region,
        offset: usize,
        into: &mut [u8],
    ) -> Result<(), shard::ExchangeError> {
        let Some(memory) = self.0.get(&region) else {
            return Ok(());
        };
        memory
            .read_bytes(offset, into)
            .map_err(|error| exchange_error(format!("taking {region:?} off the device: {error}")))
    }

    fn upload(
        &self,
        region: shard::Region,
        offset: usize,
        from: &[u8],
    ) -> Result<(), shard::ExchangeError> {
        let Some(memory) = self.0.get(&region) else {
            return Ok(());
        };
        memory
            .write_bytes(offset, from)
            .map_err(|error| exchange_error(format!("putting {region:?} on the device: {error}")))
    }
}

/// One rank's side of the exchanges a shared-out step makes. Every rank builds the same thing: the
/// sockets carry the barriers, the connections carry the payloads, and every region is registered
/// once before the first block because registering hundreds of megabytes costs far more than
/// moving them.
///
/// The leader is rank 0 and holds a socket to every other rank. A worker holds one to the leader,
/// so a barrier is a message to the leader and one back, and the addresses of the connections
/// between two workers pass through it as well.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub struct Exchanger<'a> {
    rank: usize,
    ranks: usize,
    sockets: HashMap<usize, Socket<'a>>,
    /// One connection per peer whose path has one, over which this rank writes into their memory.
    /// A peer with none is written to over the socket instead, which is every peer on a backend
    /// with no transport of its own.
    #[cfg(feature = "cuda")]
    links: HashMap<usize, LinkOf<'a>>,
    /// The memory a peer's writes land in.
    regions: HashMap<shard::Region, Held>,
    /// The backend each peer runs, which decides whether its choices are comparable with this
    /// rank's at all.
    backends: HashMap<usize, u8>,
    /// What a peer with no connection is owed, until the barrier that sends it.
    pending: HashMap<usize, Vec<Queued>>,
    /// Where this rank's blocks compute. Attention reads its inputs once per query tile, so a
    /// backend that would do that over host memory pays about seven times as much.
    computed: Computed,
    peers: HashMap<(usize, shard::Region), worker::RemoteRegion>,
}

/// Memory a step exchanges through. Where this machine has a device to register it with, a peer
/// writes into it from its own machine and nothing passes through the socket. Where it has not,
/// which is every machine without a RoCE port, it is a plain buffer a peer fills down the socket.
#[cfg(any(feature = "cuda", feature = "metal"))]
enum Held {
    #[cfg(feature = "cuda")]
    Registered(Region),
    Plain(Vec<u8>),
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Held {
    fn bytes(&self) -> usize {
        match self {
            #[cfg(feature = "cuda")]
            Held::Registered(region) => region.bytes(),
            Held::Plain(buffer) => buffer.len(),
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "cuda")]
            Held::Registered(region) => region.as_slice(),
            Held::Plain(buffer) => buffer,
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            #[cfg(feature = "cuda")]
            Held::Registered(region) => region.as_mut_slice(),
            Held::Plain(buffer) => buffer,
        }
    }

    /// Where a peer writes it from its own machine, and nothing when only the socket reaches it.
    fn remote(&self) -> (u64, Vec<u32>) {
        match self {
            #[cfg(feature = "cuda")]
            Held::Registered(region) => (region.address(), region.remote_keys().to_vec()),
            Held::Plain(_) => (0, Vec::new()),
        }
    }
}

/// One write a rank owes a peer it has no connection to, kept until the barrier sends it. The
/// bytes stay where they are until then: the region they come from does not change in between.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct Queued {
    from: shard::Region,
    offset: usize,
    into: shard::Region,
    peer_offset: usize,
    bytes: usize,
}

/// A connection this rank already had, or one it opened for a peer it has no socket to.
#[cfg(feature = "cuda")]
enum LinkOf<'a> {
    Held(&'a Rdma),
    Opened(Rdma),
}

#[cfg(feature = "cuda")]
impl LinkOf<'_> {
    fn link(&self) -> &Rdma {
        match self {
            LinkOf::Held(link) => link,
            LinkOf::Opened(link) => link,
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl<'a> Exchanger<'a> {
    /// Stands a run up: the ranks that cannot reach each other over TCP swap the addresses of their
    /// connections through the leader, then every rank registers its regions and the leader passes
    /// the table of them round. After this nothing is allocated again, so the addresses the peers
    /// hold stay good for the run.
    pub fn open(
        shard: &Shard,
        tokens: usize,
        hidden: usize,
        gated: bool,
        peers: Vec<Peer<'a>>,
        algorithms: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let (rank, ranks) = (shard.rank, shard.ranks());
        let mut sockets = HashMap::new();
        let mut backends = HashMap::new();
        #[cfg(feature = "cuda")]
        let mut links = HashMap::new();
        for peer in peers {
            backends.insert(peer.rank, peer.backend);
            #[cfg(feature = "cuda")]
            if let Some(link) = peer.link {
                links.insert(peer.rank, LinkOf::Held(link));
            }
            sockets.insert(
                peer.rank,
                Socket {
                    reader: peer.reader,
                    writer: peer.writer,
                },
            );
        }
        let mut exchanger = Exchanger {
            rank,
            ranks,
            sockets,
            #[cfg(feature = "cuda")]
            links,
            regions: HashMap::new(),
            backends,
            pending: HashMap::new(),
            computed: Computed::default(),
            peers: HashMap::new(),
        };
        #[cfg(feature = "cuda")]
        exchanger.link_peers()?;
        exchanger.register(shard, tokens, hidden, gated, algorithms)?;
        Ok(exchanger)
    }

    /// The connection this rank writes into `peer`'s memory over, or nothing when the two have
    /// only a socket between them and the bytes wait for a barrier instead.
    #[cfg(feature = "cuda")]
    fn link_to(
        &self,
        peer: usize,
        into: shard::Region,
        remote: &worker::RemoteRegion,
    ) -> Result<Option<&Rdma>, shard::ExchangeError> {
        match self.links.get(&peer) {
            Some(_) if remote.keys.is_empty() => Err(exchange_error(format!(
                "rank {peer} has a connection here but named no address for {into:?}"
            ))),
            Some(link) => Ok(Some(link.link())),
            None => Ok(None),
        }
    }

    /// Opens a connection to every rank this one has no socket to, and connects it to the other
    /// end through the leader. With two ranks there is nothing to do.
    #[cfg(feature = "cuda")]
    fn link_peers(&mut self) -> Result<(), Box<dyn Error>> {
        // A machine with no port opens no connection to anyone: its exchanges go down the socket,
        // and a peer that has a port cannot use one it has nothing to pair with.
        let Some(device) = rdma_device() else {
            return Ok(());
        };
        let mut opened = Vec::new();
        let mut mine = Vec::new();
        for peer in 0..self.ranks {
            // A rank this one has a socket to is reached by the connection the session opened, if
            // it opened one, and by that socket otherwise. Only the ranks with neither need one
            // made here, which is a worker and another worker.
            if peer == self.rank
                || self.links.contains_key(&peer)
                || self.sockets.contains_key(&peer)
            {
                continue;
            }
            let link = device.link()?;
            mine.push(worker::PeerLink {
                peer: peer as u32,
                addresses: link
                    .addresses()?
                    .into_iter()
                    .map(|address| address.to_bytes())
                    .collect(),
            });
            opened.push((peer, link));
        }
        if self.rank > 0 {
            // A worker offers the leader what it made for the other workers and takes back what
            // they made for it.
            self.send(
                0,
                Kind::SessionLinks,
                &worker::PeerLink::encode_all(&mine),
                &[],
            )?;
            let (header, body) = self.receive(0)?;
            self.expect(header, Kind::SessionLinks, &body)?;
            let theirs = worker::PeerLink::decode_all(&body)?;
            for (peer, link) in opened {
                let addresses: Option<Vec<mmh3_rdma::Address>> = theirs
                    .iter()
                    .find(|entry| entry.peer as usize == peer)
                    .and_then(|entry| {
                        entry
                            .addresses
                            .iter()
                            .map(|bytes| mmh3_rdma::Address::from_bytes(bytes))
                            .collect()
                    });
                let addresses = addresses.ok_or_else(|| {
                    format!("rank {peer} offered no connection and has no socket here either")
                })?;
                link.connect(&addresses)?;
                self.links.insert(peer, LinkOf::Opened(link));
            }
            return Ok(());
        }
        // The leader has a socket to every rank, so it opens no connection of its own here and
        // only passes the addresses on: what rank k made for rank l goes to rank l.
        let mut offered: Vec<(usize, Vec<worker::PeerLink>)> = Vec::new();
        for peer in 1..self.ranks {
            let (header, body) = self.receive(peer)?;
            self.expect(header, Kind::SessionLinks, &body)?;
            offered.push((peer, worker::PeerLink::decode_all(&body)?));
        }
        for peer in 1..self.ranks {
            let theirs: Vec<worker::PeerLink> = offered
                .iter()
                .filter(|(owner, _)| *owner != peer)
                .filter_map(|(owner, links)| {
                    links
                        .iter()
                        .find(|entry| entry.peer as usize == peer)
                        .map(|entry| worker::PeerLink {
                            peer: *owner as u32,
                            addresses: entry.addresses.clone(),
                        })
                })
                .collect();
            self.send(
                peer,
                Kind::SessionLinks,
                &worker::PeerLink::encode_all(&theirs),
                &[],
            )?;
        }
        Ok(())
    }

    /// Registers every region a step will use and passes the table of them round. `algorithms` is
    /// how many cuBLASLt choices this rank took from the leader, or sent as the leader.
    fn register(
        &mut self,
        shard: &Shard,
        tokens: usize,
        hidden: usize,
        gated: bool,
        algorithms: u32,
    ) -> Result<(), Box<dyn Error>> {
        // A machine with a port registers its memory, so a peer writes into it from its own
        // machine. One without gets plain buffers and its peers write down the socket: the table
        // carries no address for them, which is how a peer knows which way to send.
        #[cfg(feature = "cuda")]
        let device = rdma_device();
        let mut table = Vec::new();
        for (region, bytes) in shard::regions(shard, tokens, hidden, gated) {
            // A region a block computes in gets a device buffer beside it, and the two are kept in
            // step by `publish` on the way out and by the copy that follows a read on the way in.
            #[cfg(feature = "cuda")]
            let held = match device {
                Some(device) => Held::Registered(device.register(vec![0u8; bytes])?),
                None => Held::Plain(vec![0u8; bytes]),
            };
            #[cfg(not(feature = "cuda"))]
            let held = Held::Plain(vec![0u8; bytes]);
            let (kind, peer) = region.code();
            let (address, keys) = held.remote();
            table.push(worker::RemoteRegion {
                owner: self.rank as u32,
                kind,
                peer,
                address,
                bytes: bytes as u64,
                keys,
            });
            self.regions.insert(region, held);
            self.computed.make(region, bytes)?;
        }
        let table = if self.rank > 0 {
            self.send(
                0,
                Kind::SessionReady,
                &worker::SessionReady {
                    algorithms,
                    regions: table,
                }
                .encode(),
                &[],
            )?;
            let (header, body) = self.receive(0)?;
            self.expect(header, Kind::SessionReady, &body)?;
            worker::RemoteRegion::decode_all(&body)?
        } else {
            // The leader collects every rank's table and sends the whole of it back, so that a
            // rank looks an entry up by its owner rather than counting on the lists lining up.
            let mut whole = table;
            for peer in 1..self.ranks {
                let (header, body) = self.receive(peer)?;
                self.expect(header, Kind::SessionReady, &body)?;
                let ready = worker::SessionReady::decode(&body)?;
                // A CUDA rank that took none of them times its own candidates, so the two may
                // choose differently and the run is then only what these machines together
                // produce. A rank on another backend never had them to take, which is ordinary
                // and says nothing: warning there would make a Mac warn on every run.
                if ready.algorithms == 0
                    && algorithms > 0
                    && self.backends.get(&peer) == Some(&worker::BACKEND_CUDA)
                {
                    println!(
                        "rank {peer} took none of the {algorithms} cuBLASLt algorithms, so it \
                         chooses its own"
                    );
                }
                whole.extend(ready.regions);
            }
            let encoded = worker::RemoteRegion::encode_all(&whole);
            for peer in 1..self.ranks {
                self.send(peer, Kind::SessionReady, &encoded, &[])?;
            }
            whole
        };
        for entry in table {
            if entry.owner as usize == self.rank {
                continue;
            }
            if let Some(region) = shard::Region::from_code(entry.kind, entry.peer) {
                self.peers.insert((entry.owner as usize, region), entry);
            }
        }
        Ok(())
    }

    fn socket(&mut self, peer: usize) -> Result<&mut Socket<'a>, Box<dyn Error>> {
        self.sockets
            .get_mut(&peer)
            .ok_or_else(|| format!("this rank has no socket to rank {peer}").into())
    }

    fn send(
        &mut self,
        peer: usize,
        kind: Kind,
        descriptor: &[u8],
        payload: &[u8],
    ) -> Result<(), Box<dyn Error>> {
        let socket = self.socket(peer)?;
        worker::send(socket.writer, kind, 0, descriptor, payload)?;
        Ok(())
    }

    fn receive(&mut self, peer: usize) -> Result<(worker::Header, Vec<u8>), Box<dyn Error>> {
        let socket = self.socket(peer)?;
        Ok(worker::receive(socket.reader, BODY_LIMIT)?)
    }

    fn expect(
        &self,
        header: worker::Header,
        kind: Kind,
        body: &[u8],
    ) -> Result<(), Box<dyn Error>> {
        if header.kind == Kind::Error {
            return Err(format!(
                "a peer refused a session: {}",
                String::from_utf8_lossy(body)
            )
            .into());
        }
        if header.kind != kind {
            return Err(format!("a peer answered a session with {:?}", header.kind).into());
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn exchange_error(message: String) -> shard::ExchangeError {
    shard::ExchangeError(message)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Exchanger<'_> {
    /// Sends what every peer with no connection is owed and takes in what it owes this rank. The
    /// lower rank of a pair sends first and the higher takes first, since two ranks that both
    /// sent hundreds of megabytes at once would fill each other's socket and stop.
    ///
    /// Both ends of a pair decide this the same way: a rank with no port registers nothing, so it
    /// names no address, so neither side has a connection to the other and both queue.
    fn exchange_pending(&mut self) -> Result<(), shard::ExchangeError> {
        let mut peers: Vec<usize> = self.sockets.keys().copied().collect();
        // A peer this rank can write into the memory of is owed nothing here. There are no such
        // peers on a backend with no transport of its own, where every write is already queued.
        #[cfg(feature = "cuda")]
        peers.retain(|peer| !self.links.contains_key(peer));
        peers.sort_unstable();
        for peer in peers {
            if self.rank < peer {
                self.send_pending(peer)?;
                self.take_pending(peer)?;
            } else {
                self.take_pending(peer)?;
                self.send_pending(peer)?;
            }
        }
        Ok(())
    }

    /// This rank's writes to `peer`, the last of them saying so. A rank with nothing to send says
    /// that too, so the other end never waits for a message that is not coming.
    fn send_pending(&mut self, peer: usize) -> Result<(), shard::ExchangeError> {
        let queued = self.pending.remove(&peer).unwrap_or_default();
        let Exchanger {
            sockets, regions, ..
        } = self;
        let socket = sockets
            .get_mut(&peer)
            .ok_or_else(|| exchange_error(format!("no socket to rank {peer}")))?;
        for (index, write) in queued.iter().enumerate() {
            let held = regions
                .get(&write.from)
                .ok_or_else(|| exchange_error(format!("no {:?} was made", write.from)))?;
            if write.offset + write.bytes > held.bytes() {
                return Err(exchange_error(format!(
                    "{} bytes at {} of a {} byte {:?}",
                    write.bytes,
                    write.offset,
                    held.bytes(),
                    write.from
                )));
            }
            let (kind, into_peer) = write.into.code();
            let descriptor = worker::ShardWrite {
                kind,
                peer: into_peer,
                offset: write.peer_offset as u64,
                bytes: write.bytes as u64,
                last: u8::from(index + 1 == queued.len()),
            };
            worker::send(
                socket.writer,
                Kind::ShardWrite,
                0,
                &descriptor.encode(),
                &held.as_slice()[write.offset..write.offset + write.bytes],
            )
            .map_err(|error| exchange_error(format!("writing to rank {peer}: {error}")))?;
        }
        if queued.is_empty() {
            let descriptor = worker::ShardWrite {
                last: 1,
                ..worker::ShardWrite::default()
            };
            worker::send(
                socket.writer,
                Kind::ShardWrite,
                0,
                &descriptor.encode(),
                &[],
            )
            .map_err(|error| exchange_error(format!("writing to rank {peer}: {error}")))?;
        }
        Ok(())
    }

    /// What `peer` wrote, put where its region says, until the one that says it is the last.
    fn take_pending(&mut self, peer: usize) -> Result<(), shard::ExchangeError> {
        loop {
            let Exchanger {
                sockets, regions, ..
            } = self;
            let socket = sockets
                .get_mut(&peer)
                .ok_or_else(|| exchange_error(format!("no socket to rank {peer}")))?;
            let (header, body) = worker::receive(socket.reader, BODY_LIMIT)
                .map_err(|error| exchange_error(format!("reading from rank {peer}: {error}")))?;
            if header.kind != Kind::ShardWrite {
                return Err(exchange_error(format!(
                    "rank {peer} sent {:?} while its writes were due",
                    header.kind
                )));
            }
            let (write, payload_at) = worker::ShardWrite::decode(&body)
                .map_err(|error| exchange_error(format!("a write of rank {peer}: {error}")))?;
            if write.bytes > 0 {
                let region = shard::Region::from_code(write.kind, write.peer).ok_or_else(|| {
                    exchange_error(format!("rank {peer} named region {}", write.kind))
                })?;
                let held = regions
                    .get_mut(&region)
                    .ok_or_else(|| exchange_error(format!("no {region:?} was made")))?;
                let (offset, bytes) = (write.offset as usize, write.bytes as usize);
                let payload = &body[payload_at..];
                if offset + bytes > held.bytes() || payload.len() != bytes {
                    return Err(exchange_error(format!(
                        "rank {peer} wrote {} bytes at {offset} of a {} byte {region:?}",
                        payload.len(),
                        held.bytes()
                    )));
                }
                held.as_mut_slice()[offset..offset + bytes].copy_from_slice(payload);
            }
            if write.last == 1 {
                return Ok(());
            }
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl shard::Exchange for Exchanger<'_> {
    type Memory = BlockMemory;

    fn rank(&self) -> usize {
        self.rank
    }

    fn ranks(&self) -> usize {
        self.ranks
    }

    fn region(
        &mut self,
        region: shard::Region,
        bytes: usize,
    ) -> Result<BlockMemory, shard::ExchangeError> {
        let held = self
            .regions
            .get_mut(&region)
            .ok_or_else(|| exchange_error(format!("no {region:?} was registered")))?;
        if held.bytes() < bytes {
            return Err(exchange_error(format!(
                "{region:?} holds {} bytes, not the {bytes} a block wants",
                held.bytes()
            )));
        }
        Ok(self.computed.memory(region, held.as_mut_slice()))
    }

    fn publish(
        &mut self,
        region: shard::Region,
        offset: usize,
        bytes: usize,
    ) -> Result<(), shard::ExchangeError> {
        if !self.computed.holds(region) {
            return Ok(());
        }
        let held = self
            .regions
            .get_mut(&region)
            .ok_or_else(|| exchange_error(format!("no {region:?} was registered")))?;
        if offset + bytes > held.bytes() {
            return Err(exchange_error(format!(
                "{bytes} bytes at {offset} of a {} byte {region:?}",
                held.bytes()
            )));
        }
        self.computed.download(
            region,
            offset,
            &mut held.as_mut_slice()[offset..offset + bytes],
        )
    }

    fn receive(
        &mut self,
        region: shard::Region,
        offset: usize,
        bytes: usize,
    ) -> Result<(), shard::ExchangeError> {
        if !self.computed.holds(region) {
            return Ok(());
        }
        let held = self
            .regions
            .get(&region)
            .ok_or_else(|| exchange_error(format!("no {region:?} was registered")))?;
        if offset + bytes > held.bytes() {
            return Err(exchange_error(format!(
                "{bytes} bytes at {offset} of a {} byte {region:?}",
                held.bytes()
            )));
        }
        self.computed
            .upload(region, offset, &held.as_slice()[offset..offset + bytes])
    }

    fn write(
        &mut self,
        peer: usize,
        from: shard::Region,
        offset: usize,
        into: shard::Region,
        peer_offset: usize,
        bytes: usize,
    ) -> Result<(), shard::ExchangeError> {
        if peer == self.rank {
            return Err(exchange_error("a rank cannot write to itself".to_owned()));
        }
        let remote = self
            .peers
            .get(&(peer, into))
            .cloned()
            .ok_or_else(|| exchange_error(format!("rank {peer} offered no {into:?}")))?;
        if peer_offset + bytes > remote.bytes as usize {
            return Err(exchange_error(format!(
                "{bytes} bytes at {peer_offset} of a {} byte {into:?}",
                remote.bytes
            )));
        }
        // A peer that named an address is written to from here. One that named none has no port,
        // so the bytes go down the socket, and they wait for the barrier that sends them: two
        // ranks writing to each other at once would fill both sockets and stop.
        #[cfg(feature = "cuda")]
        if let Some(link) = self.link_to(peer, into, &remote)? {
            let held = self
                .regions
                .get(&from)
                .ok_or_else(|| exchange_error(format!("no {from:?} was registered")))?;
            let Held::Registered(held) = held else {
                return Err(exchange_error(format!(
                    "{from:?} is not memory a connection can send"
                )));
            };
            held.write_out(
                link,
                offset,
                remote.address + peer_offset as u64,
                &remote.keys,
                bytes,
                EXCHANGE_TIMEOUT,
            )
            .map_err(|error| exchange_error(format!("writing {into:?} of rank {peer}: {error}")))?;
            return Ok(());
        }
        if !self.sockets.contains_key(&peer) {
            return Err(exchange_error(format!(
                "rank {peer} has neither a connection here nor a socket"
            )));
        }
        self.pending.entry(peer).or_default().push(Queued {
            from,
            offset,
            into,
            peer_offset,
            bytes,
        });
        Ok(())
    }

    fn barrier(&mut self) -> Result<(), shard::ExchangeError> {
        self.exchange_pending()?;
        let arrive = |exchanger: &mut Self, peer: usize| -> Result<(), shard::ExchangeError> {
            exchanger
                .send(peer, Kind::Ready, &[], &[])
                .map_err(|error| exchange_error(format!("reaching a barrier: {error}")))
        };
        let wait = |exchanger: &mut Self, peer: usize| -> Result<(), shard::ExchangeError> {
            let (header, _) = exchanger
                .receive(peer)
                .map_err(|error| exchange_error(format!("waiting at a barrier: {error}")))?;
            if header.kind != Kind::Ready {
                return Err(exchange_error(format!(
                    "rank {peer} sent {:?} at a barrier",
                    header.kind
                )));
            }
            Ok(())
        };
        if self.rank > 0 {
            arrive(self, 0)?;
            return wait(self, 0);
        }
        for peer in 1..self.ranks {
            wait(self, peer)?;
        }
        for peer in 1..self.ranks {
            arrive(self, peer)?;
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda", feature = "metal"))]
impl Exchanger<'_> {
    /// An ordinary message over the same socket the barriers use, so that a whole run goes through
    /// one exchanger and the regions are registered once rather than once a step.
    pub fn send_to(
        &mut self,
        peer: usize,
        kind: Kind,
        descriptor: &[u8],
        payload: &[u8],
    ) -> Result<(), Box<dyn Error>> {
        self.send(peer, kind, descriptor, payload)
    }

    pub fn receive_from(
        &mut self,
        peer: usize,
    ) -> Result<(worker::Header, Vec<u8>), Box<dyn Error>> {
        self.receive(peer)
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }
}

/// The file a digest names, in the given directories of this machine's models. A leader passes its
/// own by path and a worker will have named it something else, so the digest of the safetensors
/// header is what the two agree on.
#[cfg(feature = "cuda")]
fn find_checkpoint(models: &Path, digest_wanted: u64, directories: &[&str]) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = directories
        .iter()
        .filter_map(|directory| std::fs::read_dir(models.join(directory)).ok())
        .flatten()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|kind| kind == "safetensors"))
        .collect();
    files.sort();
    files
        .into_iter()
        .find(|path| SafeTensors::open(path).is_ok_and(|file| digest(&file) == digest_wanted))
}

/// The keyframes and references of a session, read back from what `session_conditions` named and
/// `session_payload` carried.
#[cfg(feature = "cuda")]
fn session_conditions_of(
    open: &OpenSession,
    mut values: &[u8],
) -> Result<
    (
        Vec<mmh3_core::dit::inputs::Keyframe>,
        Vec<mmh3_core::dit::inputs::Reference>,
    ),
    Box<dyn Error>,
> {
    use mmh3_core::dit::inputs::{Keyframe, Reference};
    use mmh3_core::worker::Condition;

    let mut latent = |shape: &[u32]| -> Result<Option<Tensor>, Box<dyn Error>> {
        if shape.contains(&0) {
            return Ok(None);
        }
        let shape: Vec<usize> = shape.iter().map(|&extent| extent as usize).collect();
        let bytes = shape.iter().product::<usize>() * 4;
        if values.len() < bytes {
            return Err("a session carried too few condition bytes".into());
        }
        let (taken, rest) = values.split_at(bytes);
        values = rest;
        Ok(Some(Tensor::new(
            shape,
            taken
                .chunks_exact(4)
                .map(|four| f32::from_le_bytes(four.try_into().unwrap()))
                .collect(),
        )))
    };
    let (mut keyframes, mut references) = (Vec::new(), Vec::new());
    for condition in &open.conditions {
        let video = latent(&condition.video_shape)?;
        let audio = latent(&condition.audio_shape)?;
        let missing = || -> Box<dyn Error> { "a condition carried no latent".into() };
        match condition.kind {
            Condition::KEYFRAME => keyframes.push(Keyframe {
                frame_index: condition.frame_index as usize,
                video,
                audio,
            }),
            Condition::REFERENCE_PICTURE => {
                references.push(Reference::Picture(video.ok_or_else(missing)?))
            }
            Condition::REFERENCE_VIDEO => references.push(Reference::Video {
                video: video.ok_or_else(missing)?,
                audio,
            }),
            Condition::REFERENCE_AUDIO => {
                references.push(Reference::Audio(audio.ok_or_else(missing)?))
            }
            other => return Err(format!("a session named condition {other}").into()),
        }
    }
    Ok((keyframes, references))
}

/// Reads the DiT a run is about to share, so that it is here when the session opens rather than
/// after it. The leader sends this while it still has its prompt to encode.
#[cfg(feature = "cuda")]
fn prepare_dit(
    models: &Path,
    checkpoint: &worker::Checkpoint,
) -> Option<(u64, mmh3_cuda::dit::CudaDit)> {
    let path = find_checkpoint(models, checkpoint.digest, &["diffusion_models"])?;
    let started = Instant::now();
    let file = SafeTensors::open(&path).ok()?;
    match mmh3_cuda::dit::CudaDit::load(&file, "") {
        Ok(dit) => {
            println!(
                "read {} in {:.1} s before it was asked for",
                path.display(),
                started.elapsed().as_secs_f64()
            );
            Some((checkpoint.digest, dit))
        }
        Err(error) => {
            eprintln!("warning: reading {} early: {error}", path.display());
            None
        }
    }
}

/// A rank that is not the leader, running its share of every step until the leader closes the
/// session. It holds the DiT and the registered regions for the whole run.
#[cfg(feature = "cuda")]
fn serve_shard(
    reader: &mut BufReader<TcpStream>,
    writer: &mut BufWriter<TcpStream>,
    connection: Option<&Rdma>,
    models: &Path,
    open: &OpenSession,
    payload: &[u8],
    prepared: Option<(u64, mmh3_cuda::dit::CudaDit)>,
) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::inputs::DitInputs;
    use mmh3_core::dit::layout::PackedLayout;
    use mmh3_core::dit::timestep::Modality;

    // The DiT `PrepareCheckpoint` read while the leader was encoding, when it is the one this
    // session asks for.
    let mut dit = match prepared {
        Some((digest, dit)) if digest == open.checkpoint.digest => dit,
        _ => {
            let path = find_checkpoint(models, open.checkpoint.digest, &["diffusion_models"])
                .ok_or_else(|| {
                    format!(
                        "no diffusion model here has the digest {:016x}",
                        open.checkpoint.digest
                    )
                })?;
            let started = Instant::now();
            let dit = mmh3_cuda::dit::CudaDit::load(&SafeTensors::open(&path)?, "")?;
            println!(
                "loaded {} in {:.1} s",
                path.display(),
                started.elapsed().as_secs_f64()
            );
            dit
        }
    };
    dit.set_attention_precision(if open.precision == 1 {
        mmh3_cuda::attention::AttentionPrecision::Int8Fp8
    } else {
        mmh3_cuda::attention::AttentionPrecision::Bf16
    });
    // The leader's LoRAs and patches, in the order it added them, since they do not commute.
    for adapter in &open.adapters {
        let path =
            find_checkpoint(models, adapter.digest, &["loras", "patches"]).ok_or_else(|| {
                format!(
                    "no LoRA or patch here has the digest {:016x}",
                    adapter.digest
                )
            })?;
        let started = Instant::now();
        let mode = if adapter.mode == mmh3_core::worker::Adapter::MERGE {
            mmh3_cuda::dit::LoraMode::Merge
        } else {
            mmh3_cuda::dit::LoraMode::Adapter
        };
        let layers = dit.add_lora(&SafeTensors::open(&path)?, adapter.strength, mode)?;
        println!(
            "added {} to {layers} layers at strength {} in {:.1} s",
            path.display(),
            adapter.strength,
            started.elapsed().as_secs_f64()
        );
    }

    let shape = |dimensions: &[u32]| -> Vec<usize> {
        dimensions.iter().map(|value| *value as usize).collect()
    };
    let (context_tokens, context_width) = (
        open.context_shape[0] as usize,
        open.context_shape[1] as usize,
    );
    let context_bytes = context_tokens * context_width * 4;
    if payload.len() < context_bytes + context_tokens {
        return Err("a session carried too few context bytes".into());
    }
    let context = Tensor::new(
        vec![context_tokens, context_width],
        payload[..context_bytes]
            .chunks_exact(4)
            .map(|four| f32::from_le_bytes(four.try_into().unwrap()))
            .collect(),
    );
    let modalities: Vec<Modality> = payload[context_bytes..context_bytes + context_tokens]
        .iter()
        .map(|value| match value {
            1 => Modality::Text,
            2 => Modality::Audio,
            _ => Modality::Video,
        })
        .collect();
    let (keyframes, references) =
        session_conditions_of(open, &payload[context_bytes + context_tokens..])?;
    // Every rank embeds the latents itself, so only their shapes are needed to lay the step out.
    // The values arrive with each step.
    let video_shape = shape(&open.video_shape);
    let audio_shape = shape(&open.audio_shape);
    let mut inputs = DitInputs {
        video: Tensor::new(video_shape.clone(), vec![0.0; video_shape.iter().product()]),
        audio: Tensor::new(audio_shape.clone(), vec![0.0; audio_shape.iter().product()]),
        context,
        context_modalities: modalities,
        keyframes,
        references,
        sigma: 1.0,
        shift_video: open.shift_video,
        shift_audio: open.shift_audio,
    };
    let tokens = PackedLayout::for_inputs(&inputs).len();
    let sparse = sparse_attention(&open.sparse);
    // The cuts the leader worked out, since it is the only rank that knows what every machine can
    // do. An empty table is an even share, which is what a leader that measured nothing sends.
    let shard = if open.shard.is_empty() {
        Shard::even(
            open.rank as usize,
            open.ranks as usize,
            tokens,
            dit.config().heads,
            1,
        )
    } else {
        let span = |pair: [u32; 2]| pair[0] as usize..pair[1] as usize;
        let shard = Shard {
            rank: open.rank as usize,
            tokens: open.shard.iter().map(|part| span(part.tokens)).collect(),
            heads: open.shard.iter().map(|part| span(part.heads)).collect(),
        };
        if shard.ranks() != open.ranks as usize
            || shard.tokens.last().map(|part| part.end) != Some(tokens)
            || shard.heads.last().map(|part| part.end) != Some(dit.config().heads)
        {
            return Err(format!(
                "a leader cut {tokens} tokens and {} heads into {:?} and {:?}",
                dit.config().heads,
                shard.tokens,
                shard.heads
            )
            .into());
        }
        shard
    };
    let gated = dit.has_vsa_gates();
    // The leader's choices for the GEMM shapes, which this rank runs rather than timing the
    // candidates while the wire and the other ranks have its device. A key of its own that the
    // leader's does not match, which is what another GPU or another backend has, leaves them.
    let algorithms = match mmh3_cuda::algorithms::key() {
        Ok(key) if key == open.algorithm_key => {
            mmh3_cuda::algorithms::adopt_matmul(&open.algorithms)
        }
        _ => 0,
    };
    // A worker's only socket is the one the leader opened, so the leader is its only peer here and
    // the connections to the other ranks come out of the rendezvous.
    let peers = vec![Peer {
        rank: 0,
        // A worker never compares its choices with the leader's, so it does not need to be told
        // which backend the leader runs. Only the leader reads this, about its workers.
        backend: 0,
        reader,
        writer,
        link: connection,
    }];
    let mut exchanger = Exchanger::open(
        &shard,
        tokens,
        dit.config().hidden,
        gated,
        peers,
        algorithms as u32,
    )?;
    println!(
        "rank {} of {} takes tokens {:?} and heads {:?}",
        shard.rank,
        shard.ranks(),
        shard.own_tokens(),
        shard.own_heads()
    );

    loop {
        let (header, body) = exchanger.receive_from(0)?;
        match header.kind {
            Kind::CloseSession => return Ok(()),
            Kind::StepShard => {}
            other => return Err(format!("a leader sent {other:?} in a session").into()),
        }
        let step = worker::StepShard::decode(&body)?;
        let values = &body[worker::StepShard::BYTES..];
        let video_values: usize = video_shape.iter().product();
        let audio_values: usize = audio_shape.iter().product();
        if values.len() != (video_values + audio_values) * 4 {
            return Err(format!("a step carried {} bytes of latent", values.len()).into());
        }
        let floats: Vec<f32> = values
            .chunks_exact(4)
            .map(|four| f32::from_le_bytes(four.try_into().unwrap()))
            .collect();
        inputs.video = Tensor::new(video_shape.clone(), floats[..video_values].to_vec());
        inputs.audio = Tensor::new(audio_shape.clone(), floats[video_values..].to_vec());
        inputs.sigma = step.sigma;

        let started = Instant::now();
        let mut context = mmh3_cuda::shard::ShardContext {
            shard: shard.clone(),
            exchange: &mut exchanger,
            timing: Default::default(),
        };
        let step_sparse =
            sparse.filter(|sparse| sparse.applies_to_step(step.step as usize, open.steps as usize));
        let outputs = dit.forward_shard(&inputs, step_sparse.as_ref(), &mut context)?;
        let part = outputs.part.ok_or("a shared step returned no rows")?;
        println!(
            "step {} of rank {} in {:.1} s, {}",
            step.step,
            shard.rank,
            started.elapsed().as_secs_f64(),
            describe_timing(&context.timing, started.elapsed())
        );
        let descriptor = worker::VelocityPart {
            first_row: part.rows.start as u32,
            last_row: part.rows.end as u32,
            video_values: part.video.len() as u64,
            audio_values: part.audio.len() as u64,
        };
        let mut body = Vec::with_capacity((part.video.len() + part.audio.len()) * 4);
        for value in part.video.iter().chain(&part.audio) {
            body.extend_from_slice(&value.to_le_bytes());
        }
        exchanger.send_to(0, Kind::VelocityPart, &descriptor.encode(), &body)?;
    }
}

#[cfg(feature = "cuda")]
impl Worker {
    /// This worker as a rank of a shared-out run, with the session opened on it. Nothing else may
    /// use the worker while the exchanger lives, since from here on the ranks barrier and read
    /// each other rather than trading requests.
    pub fn as_peer<'a>(
        &'a mut self,
        rank: usize,
        open: &OpenSession,
        payload: &[u8],
    ) -> Result<Peer<'a>, Box<dyn Error>> {
        let Worker {
            reader,
            writer,
            rdma,
            ..
        } = self;
        let link = rdma.as_ref();
        worker::send(writer, Kind::OpenSession, 0, &open.encode(), payload)?;
        Ok(Peer {
            rank,
            backend: self.welcome.backend,
            reader,
            writer,
            link,
        })
    }
}

/// Opens a run on every worker and stands the leader's side of it up. The workers take ranks 1
/// upwards in the order they are given, which is the order `OpenSession` named them.
#[cfg(feature = "cuda")]
pub fn open_shard<'a>(
    workers: &'a mut [Worker],
    opens: &[OpenSession],
    payload: &[u8],
    shard: &Shard,
    tokens: usize,
    hidden: usize,
    gated: bool,
    algorithms: u32,
) -> Result<Exchanger<'a>, Box<dyn Error>> {
    let mut peers = Vec::with_capacity(workers.len());
    for (index, worker) in workers.iter_mut().enumerate() {
        let open = opens
            .get(index)
            .ok_or("a worker was given no session to open")?;
        peers.push(worker.as_peer(index + 1, open, payload)?);
    }
    Exchanger::open(shard, tokens, hidden, gated, peers, algorithms)
}

/// The block-sparse attention of a run, as a session carries it. Every rank has to choose the same
/// attention on the same steps, so what crosses the wire is the schedule and not one step of it.
#[cfg(feature = "cuda")]
pub fn sparse_settings(
    sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
) -> worker::SparseSettings {
    use mmh3_core::dit::sparse::SparseMethod;

    let Some(sparse) = sparse else {
        return worker::SparseSettings::default();
    };
    let (method, tau, vsa_sparsity) = match sparse.method {
        SparseMethod::Sol { tau } => (worker::SparseSettings::SOL, tau, 0.0),
        SparseMethod::Vsa { sparsity } => (worker::SparseSettings::VSA, 0.0, sparsity),
    };
    worker::SparseSettings {
        method,
        tau,
        vsa_sparsity,
        start_fraction: sparse.start_fraction,
        min_tokens: sparse.min_tokens as u32,
    }
}

/// What `sparse_settings` carried, as the blocks take it.
#[cfg(feature = "cuda")]
pub fn sparse_attention(
    settings: &worker::SparseSettings,
) -> Option<mmh3_core::dit::sparse::SparseAttention> {
    use mmh3_core::dit::sparse::{SparseAttention, SparseMethod};

    let method = match settings.method {
        worker::SparseSettings::SOL => SparseMethod::Sol { tau: settings.tau },
        worker::SparseSettings::VSA => SparseMethod::Vsa {
            sparsity: settings.vsa_sparsity,
        },
        _ => return None,
    };
    Some(SparseAttention {
        method,
        start_fraction: settings.start_fraction,
        min_tokens: settings.min_tokens as usize,
    })
}

/// A latent's shape as a session names it, or zeroes when there is none.
#[cfg(feature = "cuda")]
fn condition_shape<const N: usize>(latent: Option<&Tensor>) -> [u32; N] {
    let mut shape = [0u32; N];
    if let Some(latent) = latent {
        for (extent, value) in shape.iter_mut().zip(&latent.shape) {
            *extent = *value as u32;
        }
    }
    shape
}

/// The keyframes and references of a run as a session names them, in the order their latents follow
/// the context in the payload.
#[cfg(feature = "cuda")]
pub fn session_conditions(
    keyframes: &[mmh3_core::dit::inputs::Keyframe],
    references: &[mmh3_core::dit::inputs::Reference],
) -> Vec<worker::Condition> {
    use mmh3_core::worker::Condition;

    let mut conditions = Vec::with_capacity(keyframes.len() + references.len());
    for keyframe in keyframes {
        conditions.push(Condition {
            kind: Condition::KEYFRAME,
            frame_index: keyframe.frame_index as u32,
            video_shape: condition_shape(keyframe.video.as_ref()),
            audio_shape: condition_shape(keyframe.audio.as_ref()),
        });
    }
    for reference in references {
        conditions.push(Condition {
            kind: match reference {
                mmh3_core::dit::inputs::Reference::Picture(_) => Condition::REFERENCE_PICTURE,
                mmh3_core::dit::inputs::Reference::Video { .. } => Condition::REFERENCE_VIDEO,
                mmh3_core::dit::inputs::Reference::Audio(_) => Condition::REFERENCE_AUDIO,
            },
            frame_index: 0,
            video_shape: condition_shape(reference.video()),
            audio_shape: condition_shape(reference.audio()),
        });
    }
    conditions
}

/// The context, its modalities and the conditions' latents, as a session carries them. The
/// conditions follow in the order `session_conditions` lists them, video before audio.
#[cfg(feature = "cuda")]
pub fn session_payload(
    context: &Tensor,
    modalities: &[mmh3_core::dit::timestep::Modality],
    keyframes: &[mmh3_core::dit::inputs::Keyframe],
    references: &[mmh3_core::dit::inputs::Reference],
) -> Vec<u8> {
    use mmh3_core::dit::timestep::Modality;

    let mut payload = Vec::with_capacity(context.data.len() * 4 + context.shape[0]);
    for value in &context.data {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for token in 0..context.shape[0] {
        // An empty list means every token is text, which is what the encoders give without
        // pictures in the prompt.
        payload.push(match modalities.get(token) {
            Some(Modality::Text) | None => 1,
            Some(Modality::Audio) => 2,
            Some(Modality::Video) => 0,
        });
    }
    let latents = keyframes
        .iter()
        .flat_map(|keyframe| [keyframe.video.as_ref(), keyframe.audio.as_ref()])
        .chain(
            references
                .iter()
                .flat_map(|reference| [reference.video(), reference.audio()]),
        );
    for latent in latents.flatten() {
        for value in &latent.data {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    payload
}

/// One step of a shared-out run from the leader's side: hand out the latents, take this rank's
/// share, and collect what the others ended with.
#[cfg(feature = "cuda")]
pub fn step_shard(
    exchanger: &mut Exchanger<'_>,
    dit: &mmh3_cuda::dit::CudaDit,
    inputs: &mmh3_core::dit::inputs::DitInputs,
    sparse: Option<&mmh3_core::dit::sparse::SparseAttention>,
    shard: &Shard,
    step: usize,
) -> Result<mmh3_cuda::dit::DitOutputs, Box<dyn Error>> {
    let started = Instant::now();
    let descriptor = worker::StepShard {
        step: step as u32,
        sigma: inputs.sigma,
    };
    let mut latents = Vec::with_capacity((inputs.video.data.len() + inputs.audio.data.len()) * 4);
    for value in inputs.video.data.iter().chain(&inputs.audio.data) {
        latents.extend_from_slice(&value.to_le_bytes());
    }
    for peer in 1..shard.ranks() {
        exchanger.send_to(peer, Kind::StepShard, &descriptor.encode(), &latents)?;
    }

    let mut context = mmh3_cuda::shard::ShardContext {
        shard: shard.clone(),
        exchange: exchanger,
        timing: Default::default(),
    };
    let outputs = dit.forward_shard(inputs, sparse, &mut context)?;
    let timing = context.timing;
    let mut parts = vec![outputs.part.ok_or("a shared step returned no rows")?];
    for peer in 1..shard.ranks() {
        let (header, body) = exchanger.receive_from(peer)?;
        if header.kind != Kind::VelocityPart {
            return Err(format!("rank {peer} answered a step with {:?}", header.kind).into());
        }
        let part = worker::VelocityPart::decode(&body)?;
        let values = &body[worker::VelocityPart::BYTES..];
        let wanted = (part.video_values + part.audio_values) as usize * 4;
        if values.len() != wanted {
            return Err(format!("a rank sent {} bytes of velocity", values.len()).into());
        }
        let floats: Vec<f32> = values
            .chunks_exact(4)
            .map(|four| f32::from_le_bytes(four.try_into().unwrap()))
            .collect();
        let split = part.video_values as usize;
        parts.push(shard::VelocityRows {
            rows: part.first_row as usize..part.last_row as usize,
            video: floats[..split].to_vec(),
            audio: floats[split..].to_vec(),
        });
    }
    println!("  {}", describe_timing(&timing, started.elapsed()));
    Ok(dit.assemble_velocity(inputs, sparse, &parts)?)
}

/// Where a shared-out step went, which is what decides whether sharing one is worth it at all.
#[cfg(feature = "cuda")]
pub fn describe_timing(
    timing: &mmh3_cuda::shard::ShardTiming,
    whole: std::time::Duration,
) -> String {
    let seconds = |span: std::time::Duration| span.as_secs_f64();
    format!(
        "gather {:.1} s, barrier {:.1} s, read {:.1} s, attend {:.1} s, everything else {:.1} s",
        seconds(timing.gather),
        seconds(timing.barrier),
        seconds(timing.read),
        seconds(timing.attend),
        seconds(whole).max(seconds(timing.total())) - seconds(timing.total()),
    )
}
