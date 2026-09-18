//! Serving other mmh3 runs, and borrowing them. The wire format is `mmh3_core::worker`, the design
//! `notes/worker.md`. A worker holds checkpoints the leader may not have, so both sides name them by
//! role and confirm them by digest.

use mmh3_core::safetensors::SafeTensors;
use mmh3_core::tensor::Tensor;
use mmh3_core::worker::{
    self, CAPABILITY_ENCODE_TEXT, Checkpoint, EncodeText, Header, Hello, Kind, TRANSPORT_TCP,
    TextStates, Welcome,
};
use std::error::Error;
use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const DEFAULT_PORT: u16 = 7833;
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

/// One worker as the leader sees it.
pub struct Worker {
    pub address: String,
    pub welcome: Welcome,
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    request: u64,
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
            },
            request: 0,
            address,
        };
        let hello = Hello {
            leader: hostname(),
            token: token.to_owned(),
        };
        let body = worker.call(Kind::Hello, &hello.encode(), &[])?;
        worker.welcome = Welcome::decode(&body.1)?;
        Ok(worker)
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

    /// Sends and reads back `bytes`, and returns the round trip in gigabytes a second.
    pub fn bandwidth(&mut self, bytes: usize) -> Result<f64, Box<dyn Error>> {
        let payload = vec![0u8; bytes];
        let started = Instant::now();
        self.call(Kind::Bandwidth, &[], &payload)?;
        let seconds = started.elapsed().as_secs_f64();
        Ok((2.0 * bytes as f64 / seconds) / 1e9)
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
        let gigabytes = worker.bandwidth(megabytes << 20)?;
        println!("{megabytes} MiB there and back at {gigabytes:.2} GB/s");
    }
    Ok(())
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
    // The text encoder loads on the first prompt and stays for the rest of the session.
    let mut encoder: Resident = None;

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
                let welcome = Welcome {
                    backend: BACKEND,
                    device: device_name(),
                    memory_bytes: memory_bytes(),
                    capabilities: CAPABILITY_ENCODE_TEXT,
                    transports: TRANSPORT_TCP,
                    speed: None,
                    checkpoints: checkpoints
                        .iter()
                        .map(|(checkpoint, _)| checkpoint.clone())
                        .collect(),
                };
                reply(&mut writer, Kind::Welcome, &welcome.encode(), &[])?;
            }
            Kind::Ping => reply(&mut writer, Kind::Pong, &[], &[])?,
            Kind::Bandwidth => reply(&mut writer, Kind::BandwidthDone, &[], &body)?,
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
