//! Models kept in memory between the requests that use them.
//!
//! A machine with the memory for every model it is asked for keeps them all, and one without lets
//! go of the model it has used least until the next one fits. What identifies a kept model is not
//! the file alone but everything the load read from it, so a decoder built for one tile geometry
//! is not the one built for another.
//!
//! One table serves the whole process on both backends. A model belongs to the device that read
//! it rather than to the thread that asked for it, so the connection that reads a checkpoint is
//! not the only one that gets to use it.

use mmh3_core::safetensors::SafeTensors;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
pub type TextEncoder = mmh3_cuda::text_encoder::CudaTextEncoder;
#[cfg(feature = "metal")]
pub type TextEncoder = mmh3_metal::text_encoder::MetalTextEncoder;
#[cfg(feature = "cuda")]
pub type VideoDecoder = mmh3_cuda::vae::CudaVideoDecoder;
#[cfg(feature = "metal")]
pub type VideoDecoder = mmh3_metal::vae::MetalVideoDecoder;
#[cfg(feature = "cuda")]
pub type AudioDecoder = mmh3_cuda::audio_vae::CudaAudioDecoder;
#[cfg(feature = "metal")]
pub type AudioDecoder = mmh3_metal::audio_vae::MetalAudioDecoder;
/// The DiT a rank runs its share of a step on.
#[cfg(feature = "cuda")]
pub type Dit = mmh3_cuda::dit::CudaDit;
#[cfg(feature = "metal")]
pub type Dit = mmh3_metal::dit::MetalDit;

/// The checkpoint a kept video decoder was built from, with the tile geometry it was built for.
/// A request may ask for another geometry, which is another decoder.
pub type VideoDecoderKey = (u64, usize, usize);

/// One LoRA or patch a DiT has taken: the file, the strength as the bits of the number so that two
/// of these compare, and whether it was merged into the weights or kept beside them.
pub type AdapterKey = (u64, u32, u8);

/// What a kept DiT is: the checkpoint it was read from, and the adapters applied to it in the
/// order they were applied, since they do not commute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DitKey {
    pub checkpoint: u64,
    pub adapters: Vec<AdapterKey>,
}

/// One model, and when it was last used.
struct Slot<K, T> {
    kept: Option<(K, T)>,
    used: u64,
}

impl<K, T> Slot<K, T> {
    const fn new() -> Self {
        Self {
            kept: None,
            used: 0,
        }
    }

    /// When this slot was last used, or None when it holds nothing to let go of.
    fn releasable(&self) -> Option<u64> {
        self.kept.as_ref().map(|_| self.used)
    }
}

/// Which model to let go of.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    TextEncoder,
    VideoDecoder,
    AudioDecoder,
    Dit,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::TextEncoder => "the text encoder",
            Kind::VideoDecoder => "the video VAE",
            Kind::AudioDecoder => "the audio VAE",
            Kind::Dit => "the DiT",
        }
    }
}

/// The models this machine is holding on to.
pub struct Models {
    text_encoder: Slot<u64, TextEncoder>,
    video_decoder: Slot<VideoDecoderKey, VideoDecoder>,
    audio_decoder: Slot<u64, AudioDecoder>,
    dit: Slot<DitKey, Dit>,
    /// Counts the uses, so that the least recent one is the smallest.
    tick: u64,
    /// When a model was last asked for, for letting go of them all after a quiet while.
    touched: Option<Instant>,
}

impl Default for Models {
    fn default() -> Self {
        Self::new()
    }
}

impl Models {
    pub const fn new() -> Self {
        Self {
            text_encoder: Slot::new(),
            video_decoder: Slot::new(),
            audio_decoder: Slot::new(),
            dit: Slot::new(),
            tick: 0,
            touched: None,
        }
    }

    /// The text encoder of `path`, loaded unless the kept one came from the same checkpoint.
    pub fn text_encoder(&mut self, path: &Path, key: u64) -> Result<&TextEncoder, Box<dyn Error>> {
        if self.text_encoder.kept.as_ref().map(|(kept, _)| *kept) != Some(key) {
            self.text_encoder.kept = None;
            let file = SafeTensors::open(path)?;
            let encoder = self.read(|| load(path, || TextEncoder::load(&file)))?;
            self.text_encoder.kept = Some((key, encoder));
        }
        self.text_encoder.used = self.touch();
        Ok(&self.text_encoder.kept.as_ref().expect("just loaded").1)
    }

    /// The video decoder of `path` for the tile geometry `key` names.
    pub fn video_decoder(
        &mut self,
        path: &Path,
        key: VideoDecoderKey,
    ) -> Result<&VideoDecoder, Box<dyn Error>> {
        if self.video_decoder.kept.as_ref().map(|(kept, _)| *kept) != Some(key) {
            self.video_decoder.kept = None;
            let file = SafeTensors::open(path)?;
            let (_, tile_size, tile_overlap) = key;
            let decoder = self.read(|| {
                load(path, || {
                    VideoDecoder::load(&file, "", tile_size, tile_overlap)
                })
            })?;
            self.video_decoder.kept = Some((key, decoder));
        }
        self.video_decoder.used = self.touch();
        Ok(&self.video_decoder.kept.as_ref().expect("just loaded").1)
    }

    /// The audio decoder of `path`, loaded unless the kept one came from the same checkpoint.
    pub fn audio_decoder(
        &mut self,
        path: &Path,
        key: u64,
    ) -> Result<&AudioDecoder, Box<dyn Error>> {
        if self.audio_decoder.kept.as_ref().map(|(kept, _)| *kept) != Some(key) {
            self.audio_decoder.kept = None;
            let file = SafeTensors::open(path)?;
            let decoder = self.read(|| load(path, || AudioDecoder::load(&file, "")))?;
            self.audio_decoder.kept = Some((key, decoder));
        }
        self.audio_decoder.used = self.touch();
        Ok(&self.audio_decoder.kept.as_ref().expect("just loaded").1)
    }

    /// Reads a model. When the device has no memory left, lets go of the model used least and
    /// tries again, until there is nothing left to let go of and the read fails as it would have
    /// on a machine that was holding nothing.
    ///
    /// Nothing is set aside beforehand. What a read and the work after it take cannot be known
    /// from the size of a checkpoint, and a machine that guessed would either let go of models it
    /// had the room for or keep models it did not.
    pub fn read<T, E: Into<Box<dyn Error>>>(
        &mut self,
        mut read: impl FnMut() -> Result<T, E>,
    ) -> Result<T, Box<dyn Error>> {
        loop {
            match read().map_err(Into::into) {
                Err(error) if out_of_memory(&*error) => match self.release_oldest() {
                    Some(released) => println!(
                        "the device had no memory left, so let go of {} and tried again",
                        released.name()
                    ),
                    None => return Err(error),
                },
                result => return result,
            }
        }
    }

    /// Counts one use, and says which one it is.
    fn touch(&mut self) -> u64 {
        self.tick += 1;
        self.touched = Some(Instant::now());
        self.tick
    }

    /// How long since a model was last asked for, or None while none has been.
    fn idle_for(&self) -> Option<Duration> {
        self.touched.map(|touched| touched.elapsed())
    }

    /// Lets go of every model, and says how many there were.
    pub fn release_all(&mut self) -> usize {
        let mut released = 0;
        while self.release_oldest().is_some() {
            released += 1;
        }
        self.touched = None;
        released
    }

    /// What the kept DiT is, for a caller deciding whether it has anything to read at all.
    pub fn dit_key(&self) -> Option<&DitKey> {
        self.dit.kept.as_ref().map(|(key, _)| key)
    }

    /// The kept DiT, taken out of the table so that the caller can build on it. It is in the table
    /// again once the caller puts it back, and gone when it does not: a DiT that took half of a
    /// session's adapters is not one anybody asked for.
    pub fn take_dit(&mut self) -> Option<(DitKey, Dit)> {
        self.dit.kept.take()
    }

    /// Keeps a DiT for the sessions that follow.
    pub fn keep_dit(&mut self, key: DitKey, dit: Dit) -> &mut Dit {
        self.dit.used = self.touch();
        &mut self.dit.kept.insert((key, dit)).1
    }

    /// Lets go of the model used least recently, and says which it was.
    fn release_oldest(&mut self) -> Option<Kind> {
        let oldest = [
            (self.text_encoder.releasable(), Kind::TextEncoder),
            (self.video_decoder.releasable(), Kind::VideoDecoder),
            (self.audio_decoder.releasable(), Kind::AudioDecoder),
            (self.dit.releasable(), Kind::Dit),
        ]
        .into_iter()
        .filter_map(|(used, kind)| Some((used?, kind)))
        .min_by_key(|(used, _)| *used);
        let (_, kind) = oldest?;
        match kind {
            Kind::TextEncoder => self.text_encoder.kept = None,
            Kind::VideoDecoder => self.video_decoder.kept = None,
            Kind::AudioDecoder => self.audio_decoder.kept = None,
            Kind::Dit => self.dit.kept = None,
        }
        Some(kind)
    }
}

/// `1 model` or `3 models`, for a line that says how many were let go of.
fn counted(models: usize) -> String {
    match models {
        1 => "1 model".to_owned(),
        models => format!("{models} models"),
    }
}

/// Loads a model, saying what it read and how long it took.
fn load<T, E: Into<Box<dyn Error>>>(
    path: &Path,
    read: impl FnOnce() -> Result<T, E>,
) -> Result<T, Box<dyn Error>> {
    let started = Instant::now();
    let model = read().map_err(Into::into)?;
    println!(
        "loaded {} in {:.1} s",
        path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(model)
}

/// Takes `--vram-budget GB`, which holds this process to that much of the device and fails an
/// allocation past it as a device that small would. It is how a machine with memory to spare
/// answers for one without.
pub fn take_budget(options: &HashMap<&str, &str>) -> Result<(), Box<dyn Error>> {
    let gigabytes = crate::cli::option_float(options, "vram-budget", 0.0)?;
    if gigabytes < 0.0 {
        return Err("--vram-budget must not be negative".into());
    }
    let budget = (f64::from(gigabytes) * (1u64 << 30) as f64) as usize;
    #[cfg(feature = "cuda")]
    mmh3_cuda::set_allocation_limit(budget);
    #[cfg(feature = "metal")]
    mmh3_metal::set_allocation_limit(budget);
    Ok(())
}

/// Whether an error is the device saying it has no memory left, which a caller holding models it
/// could let go of can do something about.
#[cfg(feature = "cuda")]
pub fn out_of_memory(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<mmh3_cuda::CudaError>()
        .is_some_and(mmh3_cuda::CudaError::is_out_of_memory)
        || error
            .downcast_ref::<mmh3_cuda::model::Error>()
            .is_some_and(mmh3_cuda::model::Error::is_out_of_memory)
}

#[cfg(feature = "metal")]
pub fn out_of_memory(error: &(dyn Error + 'static)) -> bool {
    error
        .downcast_ref::<mmh3_metal::Error>()
        .is_some_and(mmh3_metal::Error::is_out_of_memory)
}

/// Where one connection takes its models from, which is the table this process shares between all
/// of them.
#[derive(Default)]
pub struct Table;

impl Table {
    pub const fn new() -> Self {
        Self
    }

    /// Lets go of everything when the device ran out of memory, so that a request that failed for
    /// want of it does not leave the next one to fail the same way. A read finds its own room by
    /// letting go and trying again, but the work after a read has no such moment: it is handed a
    /// model and runs, and this is where a machine that has filled itself up gets a clear device
    /// back.
    pub fn let_go_if_out_of_memory<T>(&mut self, result: &Result<T, Box<dyn Error>>) {
        let Err(error) = result else {
            return;
        };
        if !out_of_memory(&**error) {
            return;
        }
        match self.borrow().release_all() {
            0 => {}
            released => println!(
                "the device ran out of memory, so let go of {}",
                counted(released)
            ),
        }
    }

    /// The models, for as long as the returned borrow lives, which is one request at a time across
    /// the whole process. That is also all the device can do at once.
    ///
    /// A session that panicked while holding the models left them loaded, and the next request may
    /// still use them: what that session was in the middle of is its own business and not theirs.
    pub fn borrow(&mut self) -> impl std::ops::DerefMut<Target = Models> + '_ {
        locked()
    }
}

/// The models of this process, made on the first ask.
fn locked() -> std::sync::MutexGuard<'static, Models> {
    use std::sync::{Mutex, OnceLock, PoisonError};

    static MODELS: OnceLock<Mutex<Models>> = OnceLock::new();
    MODELS
        .get_or_init(|| Mutex::new(Models::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Lets go of every model after `idle` without a request, so that a machine nothing is asking
/// anything of is a machine with its memory back.
pub fn release_when_idle(idle: Duration) {
    // Looking oftener than this would wake a machine that is doing nothing, and seldomer would
    // hold the memory well past the quiet that was asked for.
    let interval = (idle / 4).clamp(Duration::from_secs(1), Duration::from_secs(60));
    let watch = move || {
        loop {
            std::thread::sleep(interval);
            let mut models = locked();
            if models.idle_for().is_some_and(|quiet| quiet >= idle) {
                match models.release_all() {
                    0 => {}
                    released => println!(
                        "let go of {} after {} s with nothing to do",
                        counted(released),
                        idle.as_secs()
                    ),
                }
            }
        }
    };
    if let Err(error) = std::thread::Builder::new()
        .name("idle".to_owned())
        .spawn(watch)
    {
        eprintln!("warning: nothing will watch for idle memory: {error}");
    }
}
