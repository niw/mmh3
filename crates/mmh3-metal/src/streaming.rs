//! The units of a model's weights that a Mac reads again each time it runs them, when the share of
//! memory the GPU may fill cannot hold them all. `mmh3_core::streaming` says what a unit is and
//! holds what the backends share: the layouts, the reads from the disk and the order units are
//! given up in.
//!
//! The GPU reads the host's memory, so a unit read on the way needs no copy stream. A thread of its
//! own reads the next unit from the disk while the one before runs, and the unit's tensors become
//! buffers of their own when it runs, which the command buffers that read them keep alive until
//! they are done. The runtime waits for its batch before an allocation this large, so no more than
//! a unit or two are alive at a time.
//!
//! Metal does not refuse memory past the GPU's share: it pages. So a model decides from what the
//! device says it has left whether a whole checkpoint fits, and keeps back units only within it.

use crate::model::{Weight, Weights};
use crate::{Error, Result, free_bytes};
use mmh3_core::streaming::{Files, Unit, UnitReader};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

/// Memory a model leaves free beside what it keeps and what its calls compute in.
const HEADROOM: usize = 1 << 30;

/// Whether `bytes` more fit in the GPU's share of memory beside the headroom.
pub(crate) fn fits(bytes: usize) -> Result<bool> {
    Ok(free_bytes()? >= bytes + HEADROOM)
}

/// A failure to read a unit, which says what it could not read.
fn read_error(error: impl std::fmt::Display) -> Error {
    Error::new(format!("reading weights: {error}"))
}

/// The thread that reads units into host memory.
struct Loader {
    requests: Option<Sender<(Unit, Vec<u8>)>>,
    answers: Receiver<std::result::Result<Vec<u8>, String>>,
    thread: Option<JoinHandle<()>>,
}

impl Loader {
    fn start(files: &Files) -> Result<Self> {
        let mut reader = UnitReader::new(files).map_err(read_error)?;
        let (requests, received) = channel::<(Unit, Vec<u8>)>();
        let (answer, answers) = channel();
        let thread = std::thread::Builder::new()
            .name("weights".to_owned())
            .spawn(move || {
                for (unit, mut host) in received {
                    let result = reader
                        .read(&unit, &mut host)
                        .map(|()| host)
                        .map_err(|error| error.to_string());
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

    fn send(&self, unit: &Unit, host: Vec<u8>) -> Result<()> {
        self.requests
            .as_ref()
            .expect("the requests close only on drop")
            .send((unit.clone(), host))
            .map_err(|_| Error::new("the weight reader stopped".into()))
    }

    /// Waits for the oldest unit asked for and not answered yet.
    fn answer(&self) -> Result<Vec<u8>> {
        match self.answers.recv() {
            Ok(Ok(host)) => Ok(host),
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
    /// The host buffers units are read into, two of them while no read is under way.
    spare: Vec<Vec<u8>>,
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
        }
    }

    fn largest(&self) -> usize {
        self.units.iter().map(Unit::bytes).max().unwrap_or(0)
    }

    /// Starts the thread and makes its host buffers, the first time a unit is read.
    fn start(&mut self) -> Result<()> {
        if self.loader.is_none() {
            self.loader = Some(Loader::start(&self.files)?);
        }
        let largest = self.largest();
        while self.spare.len() < 2 {
            self.spare.push(vec![0; largest]);
        }
        Ok(())
    }

    /// The buffers of `unit`'s tensors, from its bytes in `host`.
    fn buffers(weights: &Weights, unit: &Unit, host: &[u8]) -> Result<Vec<(String, Weight)>> {
        unit.tensors()
            .iter()
            .map(|tensor| {
                let bytes = &host[tensor.offset..tensor.offset + tensor.bytes()];
                Ok((
                    tensor.name.clone(),
                    Weight {
                        buffer: weights.device.alloc(bytes.len(), Some(bytes))?,
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
        // Two host buffers, and the units of the call before still in flight when the next one
        // is made, besides what the call computes in.
        let reserved = work + 4 * self.largest() + HEADROOM;
        let mut room = free_bytes()?.saturating_sub(reserved);
        let mut chosen = Vec::new();
        for &index in self.give_up.iter().rev() {
            let bytes = self.units[index].bytes();
            if bytes > room {
                break;
            }
            room -= bytes;
            chosen.push(index);
        }
        let loader = self.loader.as_ref().expect("started above");
        let mut sent = VecDeque::new();
        for &index in &chosen {
            if self.spare.is_empty() {
                let host = loader.answer()?;
                let unit = &self.units[sent.pop_front().expect("a unit being read")];
                weights.load_unit(Self::buffers(weights, unit, &host)?);
                self.spare.push(host);
            }
            loader.send(
                &self.units[index],
                self.spare.pop().expect("a spare buffer"),
            )?;
            sent.push_back(index);
        }
        while let Some(index) = sent.pop_front() {
            let host = loader.answer()?;
            weights.load_unit(Self::buffers(weights, &self.units[index], &host)?);
            self.spare.push(host);
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
    ready: HashMap<usize, Vec<u8>>,
}

impl Pass<'_> {
    /// Asks for the next units while there are host buffers to read them into.
    fn request(&mut self) -> Result<()> {
        while let Some(&index) = self.queue.get(self.next) {
            let Some(host) = self.stream.spare.pop() else {
                break;
            };
            let loader = self
                .stream
                .loader
                .as_ref()
                .expect("a pass starts the loader");
            loader.send(&self.stream.units[index], host)?;
            self.loading.push_back(index);
            self.next += 1;
        }
        Ok(())
    }

    /// Brings `index`'s tensors to the device, before the work that reads them is encoded.
    pub(crate) fn enter(&mut self, index: usize) -> Result<()> {
        if self.stream.kept[index] {
            return Ok(());
        }
        while !self.ready.contains_key(&index) {
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
            let host = loader.answer()?;
            self.ready.insert(answered, host);
        }
        let host = self.ready.remove(&index).expect("read above");
        let buffers = Stream::buffers(self.weights, &self.stream.units[index], &host);
        self.stream.spare.push(host);
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
    /// takes back their host buffers. A read that failed leaves one to make again.
    fn drop(&mut self) {
        while self.loading.pop_front().is_some() {
            let loader = self
                .stream
                .loader
                .as_ref()
                .expect("a pass starts the loader");
            if let Ok(host) = loader.answer() {
                self.stream.spare.push(host);
            }
        }
        self.stream
            .spare
            .extend(self.ready.drain().map(|(_, host)| host));
    }
}
