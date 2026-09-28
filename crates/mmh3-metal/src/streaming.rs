//! The units of a model's weights that a Mac reads again each time it runs them, when the share of
//! memory the GPU may fill cannot hold them all. `mmh3_core::streaming` says what a unit is and
//! holds what the backends share: the layouts, the reads from the disk and the order units are
//! given up in.
//!
//! The GPU reads the host's memory, so a unit needs no copy at all: a thread of its own reads it
//! from the disk into a region of pages, and its tensors become buffers over that region where
//! they lie. A unit read on the way lands in one of a few regions that the units take turns in,
//! while the ones before it run. A region is read into again only once Metal has let go of every
//! buffer over it, which is after the last command that reads them completes. A unit kept back
//! has a region of its own.
//!
//! Metal does not refuse memory past the GPU's share: it pages. So a model decides from what the
//! device says it has left whether a whole checkpoint fits, and keeps back units only within it.

use crate::model::{Weight, Weights};
use crate::{Error, PAGE, Region, Result, free_bytes};
use mmh3_core::safetensors::{SafeTensors, TensorInfo};
use mmh3_core::streaming::{Arrangement, Files, Unit, UnitReader};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Memory a model leaves free beside what it keeps and what its calls compute in.
const HEADROOM: usize = 1 << 30;
/// The regions units read on the way take turns in: one the GPU reads, one ready for it, and one
/// the reader fills.
const REGIONS: usize = 3;
/// How long a unit waits for Metal to let go of a region after the work over it has completed.
const RELEASE_TIMEOUT: Duration = Duration::from_secs(10);

/// A unit laid out so that each of its tensors starts on a page, where a buffer can be made over
/// it.
pub(crate) fn unit(name: &str) -> Unit {
    Unit::aligned(name, PAGE)
}

/// Whether `bytes` more fit in the GPU's share of memory beside the headroom.
pub(crate) fn fits(bytes: usize) -> Result<bool> {
    Ok(free_bytes()? >= bytes + HEADROOM)
}

/// A failure to read a unit, which says what it could not read.
fn read_error(error: impl std::fmt::Display) -> Error {
    Error::new(format!("reading weights: {error}"))
}

/// The thread that reads units into regions.
struct Loader {
    requests: Option<Sender<(Unit, Arc<Region>)>>,
    answers: Receiver<std::result::Result<Arc<Region>, String>>,
    thread: Option<JoinHandle<()>>,
}

impl Loader {
    fn start(files: &Files) -> Result<Self> {
        let mut reader = UnitReader::new(files).map_err(read_error)?;
        let (requests, received) = channel::<(Unit, Arc<Region>)>();
        let (answer, answers) = channel();
        let thread = std::thread::Builder::new()
            .name("weights".to_owned())
            .spawn(move || {
                for (unit, mut region) in received {
                    let result = match Region::bytes_mut(&mut region) {
                        Some(bytes) => reader.read(&unit, bytes).map_err(|error| error.to_string()),
                        None => Err(format!("{}: its region is still in use", unit.name)),
                    }
                    .map(|()| region);
                    if answer.send(result).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| Error::new(format!("starting the weight reader: {error}")))?;
        Ok(Loader {
            requests: Some(requests),
            answers,
            thread: Some(thread),
        })
    }

    fn send(&self, unit: &Unit, region: Arc<Region>) -> Result<()> {
        self.requests
            .as_ref()
            .expect("the requests close only on drop")
            .send((unit.clone(), region))
            .map_err(|_| Error::new("the weight reader stopped".into()))
    }

    /// Waits for the oldest unit asked for and not answered yet.
    fn answer(&self) -> Result<Arc<Region>> {
        match self.answers.recv() {
            Ok(Ok(region)) => Ok(region),
            Ok(Err(error)) => Err(read_error(error)),
            Err(_) => Err(Error::new("the weight reader stopped".into())),
        }
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The units of one model, which of them stay on the device, and the thread that reads the rest.
pub(crate) struct Stream {
    /// What the units are, for the line that says how many are kept, such as "DiT blocks".
    what: &'static str,
    files: Files,
    units: Vec<Unit>,
    /// The order in which the units are given up.
    give_up: Vec<usize>,
    kept: Vec<bool>,
    /// The memory a call computes in that the kept units were chosen beside, once they were.
    settled: Option<usize>,
    loader: Option<Loader>,
    /// The regions units read on the way take turns in, which nothing holds.
    spare: Vec<Arc<Region>>,
    /// Those whose units have run, until Metal lets go of the buffers over them.
    used: VecDeque<Arc<Region>>,
    /// How many of them there are, wherever they are.
    regions: usize,
    /// Whether units may be kept back at all.
    keeps: bool,
}

impl Stream {
    /// `units` in the order a call runs them, none of them on the device yet. `give_up` is the
    /// order in which they go when there is not the room for all of them.
    pub(crate) fn new(
        what: &'static str,
        files: Files,
        units: Vec<Unit>,
        give_up: Vec<usize>,
    ) -> Self {
        assert_eq!(give_up.len(), units.len(), "every unit has a place to go");
        Stream {
            what,
            files,
            kept: vec![false; units.len()],
            units,
            give_up,
            settled: None,
            loader: None,
            spare: Vec::new(),
            used: VecDeque::new(),
            regions: 0,
            keeps: true,
        }
    }

    /// Adds `file`'s tensor to unit `index` or replaces the unit's own, which a patch does. The
    /// units kept back are let go of, and the reader and its regions are made again, since the
    /// units' files and sizes have changed.
    pub(crate) fn insert(
        &mut self,
        weights: &Weights,
        index: usize,
        name: &str,
        file: &SafeTensors,
        info: &TensorInfo,
    ) {
        for (unit, kept) in self.units.iter().zip(&mut self.kept) {
            if *kept {
                weights.unload_unit(unit.tensors().iter().map(|tensor| tensor.name.as_str()));
                *kept = false;
            }
        }
        self.settled = None;
        self.loader = None;
        self.spare.clear();
        self.used.clear();
        self.regions = 0;
        let piece = self.files.piece(file, info);
        self.units[index].insert(
            name,
            info.dtype,
            info.shape.clone(),
            vec![piece],
            Arrangement::Contiguous,
        );
    }

    /// Keeps no unit back, for a model that runs once: it reads each unit once either way, and a
    /// unit kept back only holds memory that the model after it wants.
    pub(crate) fn keep_none(&mut self) {
        self.keeps = false;
    }

    fn largest(&self) -> usize {
        self.units.iter().map(Unit::bytes).max().unwrap_or(0)
    }

    /// Starts the thread, the first time a unit is read.
    fn start(&mut self) -> Result<()> {
        if self.loader.is_none() {
            self.loader = Some(Loader::start(&self.files)?);
        }
        Ok(())
    }

    /// A region to read a unit on the way into, if one is free: a spare one, one Metal has let go
    /// of, or, while there are fewer than `REGIONS`, a new one.
    fn free_region(&mut self) -> Result<Option<Arc<Region>>> {
        if let Some(position) = self.used.iter().position(Region::unused) {
            let region = self.used.remove(position).expect("found above");
            self.spare.push(region);
        }
        if let Some(region) = self.spare.pop() {
            return Ok(Some(region));
        }
        if self.regions < REGIONS {
            self.regions += 1;
            return Region::new(self.largest()).map(Some);
        }
        Ok(None)
    }

    /// Waits for Metal to let go of a region a unit has run from, once the work that reads it has
    /// completed.
    fn wait_for_region(&mut self, weights: &Weights) -> Result<Arc<Region>> {
        weights.device.synchronize()?;
        let start = Instant::now();
        loop {
            if let Some(region) = self.free_region()? {
                return Ok(region);
            }
            if self.used.is_empty() || start.elapsed() > RELEASE_TIMEOUT {
                return Err(Error::new(format!(
                    "the {} read on the way still hold their regions",
                    self.what
                )));
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// The buffers of `unit`'s tensors, over its bytes in `region`.
    fn buffers(
        weights: &Weights,
        unit: &Unit,
        region: &Arc<Region>,
    ) -> Result<Vec<(String, Weight)>> {
        unit.tensors()
            .iter()
            .map(|tensor| {
                Ok((
                    tensor.name.clone(),
                    Weight {
                        buffer: weights.device.wrap(region, tensor.offset, tensor.bytes())?,
                        shape: tensor.shape.clone(),
                        dtype: tensor.dtype,
                    },
                ))
            })
            .collect()
    }

    /// Keeps back as many units as fit beside `work` bytes of a call's own, which a model calls
    /// before each call. A call that computes in more than the kept units were chosen beside lets
    /// go of them and chooses again. The units given up last are kept first.
    pub(crate) fn settle(&mut self, weights: &Weights, work: usize) -> Result<()> {
        if self.settled.is_some_and(|settled| settled >= work) {
            return Ok(());
        }
        for (unit, kept) in self.units.iter().zip(&mut self.kept) {
            if *kept {
                weights.unload_unit(unit.tensors().iter().map(|tensor| tensor.name.as_str()));
                *kept = false;
            }
        }
        // The units let go of are the device's again once the work that reads them is done.
        weights.device.synchronize()?;
        self.start()?;
        // The regions units read on the way take turns in, and one more for a region Metal has not
        // let go of yet, besides what the call computes in.
        let reserved = work + (REGIONS + 1) * self.largest() + HEADROOM;
        let mut room = match self.keeps {
            true => free_bytes()?.saturating_sub(reserved),
            false => 0,
        };
        let mut chosen = Vec::new();
        for &index in self.give_up.iter().rev() {
            let bytes = self.units[index].bytes();
            if bytes > room {
                break;
            }
            room -= bytes;
            chosen.push(index);
        }
        // Each unit kept back is read into a region of its own, a couple of them ahead.
        let loader = self.loader.as_ref().expect("started above");
        let mut sent = VecDeque::new();
        for &index in &chosen {
            if sent.len() == 2 {
                let region = loader.answer()?;
                let unit = &self.units[sent.pop_front().expect("a unit being read")];
                weights.load_unit(Self::buffers(weights, unit, &region)?);
            }
            let unit = &self.units[index];
            loader.send(unit, Region::new(unit.bytes())?)?;
            sent.push_back(index);
        }
        while let Some(index) = sent.pop_front() {
            let region = loader.answer()?;
            weights.load_unit(Self::buffers(weights, &self.units[index], &region)?);
        }
        println!(
            "keeping {} of {} {} on the device and reading the rest as they run",
            chosen.len(),
            self.units.len(),
            self.what
        );
        for index in chosen {
            self.kept[index] = true;
        }
        self.settled = Some(work);
        Ok(())
    }

    /// A pass over `order`, the units one call runs in the order it runs them.
    pub(crate) fn pass<'a>(
        &'a mut self,
        weights: &'a Weights,
        order: impl IntoIterator<Item = usize>,
    ) -> Result<Pass<'a>> {
        self.start()?;
        let queue = order
            .into_iter()
            .filter(|&index| !self.kept[index])
            .collect();
        let mut pass = Pass {
            stream: self,
            weights,
            queue,
            next: 0,
            loading: VecDeque::new(),
            ready: HashMap::new(),
        };
        pass.request()?;
        Ok(pass)
    }
}

/// The units read on the way during one call, in the order it runs them.
pub(crate) struct Pass<'a> {
    stream: &'a mut Stream,
    weights: &'a Weights,
    queue: Vec<usize>,
    /// The first unit of the queue not asked for yet.
    next: usize,
    /// Units asked for whose answers have not come yet, oldest first.
    loading: VecDeque<usize>,
    /// Units read and not run yet.
    ready: HashMap<usize, Arc<Region>>,
}

impl Pass<'_> {
    /// Asks for the next units while there are regions to read them into.
    fn request(&mut self) -> Result<()> {
        while let Some(&index) = self.queue.get(self.next) {
            let Some(region) = self.stream.free_region()? else {
                break;
            };
            self.send(index, region)?;
        }
        Ok(())
    }

    fn send(&mut self, index: usize, region: Arc<Region>) -> Result<()> {
        let loader = self
            .stream
            .loader
            .as_ref()
            .expect("a pass starts the loader");
        loader.send(&self.stream.units[index], region)?;
        self.loading.push_back(index);
        self.next += 1;
        Ok(())
    }

    /// Brings `index`'s tensors to the device, before the work that reads them is encoded.
    pub(crate) fn enter(&mut self, index: usize) -> Result<()> {
        if self.stream.kept[index] {
            return Ok(());
        }
        while !self.ready.contains_key(&index) {
            // Every region holds a unit that has run, so the next one waits for one of them.
            if self.loading.is_empty() && self.queue.get(self.next) == Some(&index) {
                let region = self.stream.wait_for_region(self.weights)?;
                self.send(index, region)?;
            }
            let Some(answered) = self.loading.pop_front() else {
                return Err(Error::new(format!(
                    "{} runs out of the order its pass was given",
                    self.stream.units[index].name
                )));
            };
            let loader = self
                .stream
                .loader
                .as_ref()
                .expect("a pass starts the loader");
            // A read that failed loses its region, which the next one makes again.
            let region = loader.answer().inspect_err(|_| self.stream.regions -= 1)?;
            self.ready.insert(answered, region);
        }
        let region = self.ready.remove(&index).expect("read above");
        let buffers = Stream::buffers(self.weights, &self.stream.units[index], &region);
        self.stream.used.push_back(region);
        self.weights.load_unit(buffers?);
        self.request()
    }

    /// Lets go of `index`'s tensors. The work already encoded keeps their buffers until it is done.
    pub(crate) fn leave(&mut self, index: usize) {
        if !self.stream.kept[index] {
            let unit = &self.stream.units[index];
            self.weights
                .unload_unit(unit.tensors().iter().map(|tensor| tensor.name.as_str()));
        }
    }
}

impl Drop for Pass<'_> {
    /// Waits for the reads still under way, so that the next pass reads only its own answers, and
    /// takes back their regions. A read that failed leaves one to make again.
    fn drop(&mut self) {
        while self.loading.pop_front().is_some() {
            let loader = self
                .stream
                .loader
                .as_ref()
                .expect("a pass starts the loader");
            match loader.answer() {
                Ok(region) => self.stream.spare.push(region),
                Err(_) => self.stream.regions -= 1,
            }
        }
        self.stream
            .spare
            .extend(self.ready.drain().map(|(_, region)| region));
    }
}
