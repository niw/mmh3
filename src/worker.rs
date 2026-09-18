//! Serving other mmh3 runs, and borrowing them. The wire format is `mmh3_core::worker`, the design
//! `notes/worker.md`. A worker holds checkpoints the leader may not have, so both sides name them by
//! role and confirm them by digest.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::worker::{
    self, CAPABILITY_DECODE_VIDEO, CAPABILITY_ENCODE_TEXT, Canvas, Checkpoint, DecodeVideo,
    EncodeText, Header, Hello, Kind, TRANSPORT_TCP, TextStates, Welcome,
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
/// Bodies larger than this are refused before they are read.
const BODY_LIMIT: u64 = 4 << 30;
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

/// The reliable connection to one peer, when both sides have RoCE. Bulk payloads go this way and
/// the socket carries only what asks for them.
#[cfg(feature = "cuda")]
type Rdma = mmh3_rdma::Connection;
#[cfg(feature = "cuda")]
type Region = mmh3_rdma::Region;
#[cfg(not(feature = "cuda"))]
type Rdma = ();
#[cfg(not(feature = "cuda"))]
type Region = ();

/// Opens this machine's side of a reliable connection, or nothing when it has no RoCE port.
fn open_rdma() -> Option<(Rdma, [u8; worker::RDMA_ADDRESS_BYTES])> {
    #[cfg(feature = "cuda")]
    {
        let connection = mmh3_rdma::Connection::open("", mmh3_rdma::DEFAULT_GLOBAL_ID).ok()?;
        let address = connection.address().ok()?;
        Some((connection, address.to_bytes()))
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

/// Finishes the connection once the peer's address has arrived over the socket.
fn join_rdma(connection: Rdma, peer: &[u8; worker::RDMA_ADDRESS_BYTES]) -> Option<Rdma> {
    #[cfg(feature = "cuda")]
    {
        let address = mmh3_rdma::Address::from_bytes(peer)?;
        connection.connect(&address).ok()?;
        Some(connection)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (connection, peer);
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
            staging: None,
            address,
        };
        let offered = open_rdma();
        let hello = Hello {
            leader: hostname(),
            token: token.to_owned(),
            rdma: offered.as_ref().map(|(_, address)| *address),
        };
        let body = worker.call(Kind::Hello, &hello.encode(), &[])?;
        worker.welcome = Welcome::decode(&body.1)?;
        worker.rdma = match (offered, worker.welcome.rdma) {
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
        if self.reads_remotely() {
            let request = worker::Canvas {
                chunk: 0,
                bytes: bytes as u64,
                remote_address: 0,
                remote_key: 0,
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
            let offer = worker::Canvas::decode(&body)?;
            self.read_into_staging(&offer)?;
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
        let canvas = Canvas::decode(&body)?;
        if canvas.remote_key != 0 {
            return self.read_canvas(&canvas);
        }
        let payload = &body[Canvas::BYTES..];
        if payload.len() as u64 != canvas.bytes {
            return Err(format!("{} sent {} bytes of canvas", self.address, payload.len()).into());
        }
        Ok(payload.to_vec())
    }

    /// Reads the worker's memory into this side's, and tells it the memory is free. The bytes stay
    /// in the registered buffer, which the caller copies out of if it needs them.
    #[cfg(feature = "cuda")]
    fn read_into_staging(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        let connection = self
            .rdma
            .as_ref()
            .ok_or_else(|| format!("{} pointed at memory with no connection", self.address))?;
        let bytes = canvas.bytes as usize;
        // One registered buffer serves every transfer of a run, since registering it costs tens of
        // milliseconds for every hundred megabytes. It only ever grows: a smaller payload reads
        // into the front of it.
        if self.staging.as_ref().map(Region::bytes).unwrap_or(0) < bytes {
            self.staging = None;
            self.staging = Some(connection.register(vec![0u8; bytes])?);
        }
        let region = self.staging.as_mut().expect("just registered");
        region.read_from(
            connection,
            canvas.remote_address,
            canvas.remote_key,
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
        let how = if remote { "read" } else { "there and back" };
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
        // never waits at the start.
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

    const USAGE: &str = "usage: mmh3 worker [--listen ADDR] [--models DIR] [--token FILE] | mmh3 worker --probe HOST[:PORT]";
    let options = parse_options(arguments, &["listen", "models", "token", "probe"], USAGE)?;
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
    println!("serving {} on {listen}", models.display());
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
        std::thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map_or_else(|_| "?".to_owned(), |address| address.to_string());
            if let Err(error) = session(stream, &checkpoints, &token) {
                eprintln!("worker session with {peer} ended: {error}");
            }
        });
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn session(
    stream: TcpStream,
    checkpoints: &[(Checkpoint, PathBuf)],
    token: &str,
) -> Result<(), Box<dyn Error>> {
    stream.set_nodelay(true)?;
    let mut reader = BufReader::with_capacity(STREAM_BUFFER, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(STREAM_BUFFER, stream);
    // The text encoder and the video decoder load on their first request and stay for the session.
    let mut encoder: Resident = None;
    let mut decoder: ResidentDecoder = None;
    let mut rdma: Option<Rdma> = None;
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
                let offered = hello.rdma.and_then(|_| open_rdma());
                let welcome = Welcome {
                    backend: BACKEND,
                    device: device_name(),
                    memory_bytes: memory_bytes(),
                    capabilities: CAPABILITY_ENCODE_TEXT | CAPABILITY_DECODE_VIDEO,
                    transports: TRANSPORT_TCP
                        | if offered.is_some() {
                            worker::TRANSPORT_RDMA
                        } else {
                            0
                        },
                    speed: None,
                    checkpoints: checkpoints
                        .iter()
                        .map(|(checkpoint, _)| checkpoint.clone())
                        .collect(),
                    rdma: offered.as_ref().map(|(_, address)| *address),
                };
                reply(&mut writer, Kind::Welcome, &welcome.encode(), &[])?;
                rdma = match (offered, hello.rdma) {
                    (Some((connection, _)), Some(peer)) => join_rdma(connection, &peer),
                    _ => None,
                };
            }
            Kind::Ping => reply(&mut writer, Kind::Pong, &[], &[])?,
            // A measurement moves the same way a payload would: read out of this side's memory
            // when there is a reliable connection, echoed down the socket when there is not.
            Kind::Bandwidth => match (rdma.as_ref(), worker::Canvas::decode(&body)) {
                (Some(connection), Ok(request)) if !body.is_empty() => {
                    let bytes = request.bytes as usize;
                    if staging.as_ref().map(Region::bytes).unwrap_or(0) < bytes {
                        // NOTE: the old region unregisters before the new one registers, so the
                        // two never hold the same memory at once.
                        drop(staging.take());
                        staging = Some(connection.register(vec![0u8; bytes])?);
                    }
                    let region = staging.as_ref().expect("just registered");
                    let offer = worker::Canvas {
                        chunk: 0,
                        bytes: request.bytes,
                        remote_address: region.address(),
                        remote_key: region.remote_key(),
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
            Kind::DecodeVideo => {
                let started = Instant::now();
                match decode_video(checkpoints, &body, &mut decoder) {
                    Ok((chunk, payload)) => {
                        let megabytes = payload.len() >> 20;
                        let mut canvas = Canvas {
                            chunk,
                            bytes: payload.len() as u64,
                            remote_address: 0,
                            remote_key: 0,
                        };
                        // Without a reliable connection the canvas goes down the socket.
                        let Some(connection) = rdma.as_ref() else {
                            println!(
                                "decoded chunk {chunk} in {:.1} s, {megabytes} MiB back",
                                started.elapsed().as_secs_f64()
                            );
                            reply(&mut writer, Kind::Canvas, &canvas.encode(), &payload)?;
                            continue;
                        };
                        // One registered canvas serves the session. Registering hundreds of
                        // megabytes costs seconds, so every chunk copies into the same memory,
                        // which only ever grows.
                        if staging.as_ref().map(Region::bytes).unwrap_or(0) < payload.len() {
                            staging = None;
                            match connection.register(vec![0u8; payload.len()]) {
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
                        canvas.remote_address = region.address();
                        canvas.remote_key = region.remote_key();
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
