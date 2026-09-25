//! The units of a model's weights that the device reads again each time it runs them, when it
//! cannot keep them all. `mmh3_core::streaming` says what a unit is and holds what the backends
//! share: the layouts, the reads from the disk and the order units are given up in.
//!
//! The units read on the way take turns in two regions of one device buffer. While the kernels of
//! one unit run from one region, a thread of its own reads the next unit from the disk into pinned
//! memory and copies it into the other region on a stream of its own. A unit larger than one region
//! takes both. The kernels wait for a copy through an event the copy records, and a copy waits for
//! the kernels that last read its region through an event they record, so neither the thread that
//! launches the kernels nor the one that reads waits on the other while there is work to overlap.
//!
//! A model starts out holding every unit as it was loaded. When the device runs out of memory, it
//! lets go of them all and keeps back, before its next call, as many as the memory left beside
//! that call's buffers allows. The rest are read on the way.

use crate::loader::PinnedBuffer;
use crate::model::{DeviceTensors, Error};
use crate::{CudaError, DeviceBuffer, check, free_bytes};
use mmh3_core::safetensors::DType;
use mmh3_core::streaming::{Arrangement, Files, Piece, Unit, UnitReader};
use std::collections::{HashMap, VecDeque};
use std::ffi::{c_int, c_void};
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

unsafe extern "C" {
    fn mmh3_cuda_stream_create(stream: *mut *mut c_void) -> c_int;
    fn mmh3_cuda_stream_destroy(stream: *mut c_void) -> c_int;
    fn mmh3_cuda_stream_synchronize(stream: *mut c_void) -> c_int;
    fn mmh3_cuda_stream_wait_event(stream: *mut c_void, event: *mut c_void) -> c_int;
    fn mmh3_cuda_event_create(event: *mut *mut c_void) -> c_int;
    fn mmh3_cuda_event_destroy(event: *mut c_void) -> c_int;
    fn mmh3_cuda_event_record(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn mmh3_cuda_event_synchronize(event: *mut c_void) -> c_int;
    fn mmh3_cuda_copy_to_device_async(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_cuda_get_device(device: *mut c_int) -> c_int;
    fn mmh3_cuda_set_device(device: c_int) -> c_int;
}

/// Memory a model leaves free beside the units it keeps, for what its call allocates after it has
/// settled which units those are.
const HEADROOM: usize = 512 << 20;

/// Whether `bytes` more fit on the device beside the headroom.
pub(crate) fn fits(bytes: usize) -> Result<bool, Error> {
    Ok(free_bytes()? >= bytes + HEADROOM)
}

/// Where a unit read on the way sits: one of the two regions, or both for a unit too large for one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    One(usize),
    Both,
}

impl Region {
    fn halves(self) -> &'static [usize] {
        match self {
            Region::One(0) => &[0],
            Region::One(_) => &[1],
            Region::Both => &[0, 1],
        }
    }

    fn offset(self, region_bytes: usize) -> usize {
        match self {
            Region::One(half) => half * region_bytes,
            Region::Both => 0,
        }
    }
}

/// Where a unit's weights are.
enum Kept {
    /// In tensors of their own in the model's table, as a whole load leaves them.
    Tensors,
    /// In one buffer laid out as the unit is.
    Buffer(DeviceBuffer),
    /// Nowhere: read into a region whenever the unit runs.
    Read(Region),
}

/// The copy stream and the events both threads use.
struct Handles {
    stream: *mut c_void,
    /// Recorded on the copy stream after the last copy into each region, or out of each half of
    /// the pinned memory.
    loaded: [*mut c_void; 2],
    /// Recorded on the default stream after the last kernel that read each region.
    done: [*mut c_void; 2],
}

// SAFETY: CUDA streams and events belong to the context rather than to a thread, and the runtime
// takes calls on them from any thread.
unsafe impl Send for Handles {}
unsafe impl Sync for Handles {}

impl Handles {
    fn new() -> Result<Self, CudaError> {
        let mut handles = Handles {
            stream: ptr::null_mut(),
            loaded: [ptr::null_mut(); 2],
            done: [ptr::null_mut(); 2],
        };
        // SAFETY: the fields are valid out pointers, and drop releases whatever was created.
        unsafe {
            check(mmh3_cuda_stream_create(&mut handles.stream))?;
            for event in handles.loaded.iter_mut().chain(&mut handles.done) {
                check(mmh3_cuda_event_create(event))?;
            }
        }
        Ok(handles)
    }
}

impl Drop for Handles {
    fn drop(&mut self) {
        // SAFETY: the handles were created here and are destroyed once, after the work on them.
        unsafe {
            if !self.stream.is_null() {
                mmh3_cuda_stream_synchronize(self.stream);
            }
            for &event in self.loaded.iter().chain(&self.done) {
                if !event.is_null() {
                    mmh3_cuda_event_destroy(event);
                }
            }
            if !self.stream.is_null() {
                mmh3_cuda_stream_destroy(self.stream);
            }
        }
    }
}

/// A device address the reading thread copies to.
struct Destination(*mut c_void);

// SAFETY: a device address means the same on every thread of the process.
unsafe impl Send for Destination {}

/// One unit for the reading thread to bring to the device.
struct Request {
    unit: Unit,
    destination: Destination,
    /// The halves of the pinned memory the unit passes through, which are also the regions a unit
    /// read on the way lands in.
    halves: &'static [usize],
    /// Whether the copy waits for the kernels that last read the regions it lands in.
    gated: bool,
}

/// The thread that reads units and copies them to the device.
struct Loader {
    handles: Arc<Handles>,
    requests: Option<Sender<Request>>,
    answers: Receiver<Result<(), String>>,
    thread: Option<JoinHandle<()>>,
}

impl Loader {
    /// Starts a thread that reads through `region_bytes` bytes of pinned memory for each region.
    fn start(files: &Files, region_bytes: usize) -> Result<Self, Error> {
        let mut device = 0;
        // SAFETY: a valid out pointer.
        check(unsafe { mmh3_cuda_get_device(&mut device) })?;
        let handles = Arc::new(Handles::new()?);
        let mut pinned = PinnedBuffer::new(2 * region_bytes)?;
        let mut reader = UnitReader::new(files)
            .map_err(|error| Error::Model(format!("reading weights: {error}")))?;
        let (requests, received) = channel::<Request>();
        let (answer, answers) = channel();
        let thread_handles = Arc::clone(&handles);
        let thread = std::thread::Builder::new()
            .name("weights".to_owned())
            .spawn(move || {
                let handles = thread_handles;
                // SAFETY: plain device selection for this thread.
                let selected = check(unsafe { mmh3_cuda_set_device(device) })
                    .map_err(|error| error.to_string());
                for request in received {
                    let result = selected.clone().and_then(|()| {
                        load(&handles, &mut reader, &mut pinned, region_bytes, &request)
                    });
                    if answer.send(result).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| Error::Model(format!("starting the weight reader: {error}")))?;
        Ok(Loader {
            handles,
            requests: Some(requests),
            answers,
            thread: Some(thread),
        })
    }

    fn send(&self, request: Request) -> Result<(), Error> {
        self.requests
            .as_ref()
            .expect("the requests close only on drop")
            .send(request)
            .map_err(|_| Error::Model("the weight reader stopped".to_owned()))
    }

    /// Waits for the answer to the oldest request not yet answered.
    fn answer(&self) -> Result<(), Error> {
        match self.answers.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(Error::Model(format!("reading weights: {error}"))),
            Err(_) => Err(Error::Model("the weight reader stopped".to_owned())),
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

/// Reads one unit through its halves of the pinned memory and copies it to its destination.
fn load(
    handles: &Handles,
    reader: &mut UnitReader,
    pinned: &mut PinnedBuffer,
    region_bytes: usize,
    request: &Request,
) -> Result<(), String> {
    let unit = &request.unit;
    let start = request.halves[0] * region_bytes;
    // SAFETY: the events belong to the handles, and waiting on one never recorded returns at once.
    for &half in request.halves {
        check(unsafe { mmh3_cuda_event_synchronize(handles.loaded[half]) })
            .map_err(|error| error.to_string())?;
    }
    let staged = &mut pinned.as_mut_slice()[start..start + unit.bytes()];
    reader
        .read(unit, staged)
        .map_err(|error| error.to_string())?;
    // SAFETY: the destination holds the unit, the pinned bytes stay untouched until the event
    // recorded after the copy, and the handles outlive this call.
    unsafe {
        if request.gated {
            for &half in request.halves {
                check(mmh3_cuda_stream_wait_event(
                    handles.stream,
                    handles.done[half],
                ))
                .map_err(|error| error.to_string())?;
            }
        }
        check(mmh3_cuda_copy_to_device_async(
            request.destination.0,
            staged.as_ptr().cast(),
            unit.bytes(),
            handles.stream,
        ))
        .map_err(|error| error.to_string())?;
        for &half in request.halves {
            check(mmh3_cuda_event_record(handles.loaded[half], handles.stream))
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

/// The two regions units read on the way take turns in, and the thread that fills them.
struct Regions {
    /// Bytes of one region.
    bytes: usize,
    /// Goes before the regions, so that no copy into them is left when they are freed.
    loader: Loader,
    device: DeviceBuffer,
    /// Files the loader reads from, which a patch adds to.
    files: usize,
    /// The unit each region holds whole, which a call that runs it again need not read.
    holds: [Option<usize>; 2],
}

/// The units of one model and where each one's weights are.
pub(crate) struct Stream {
    /// What the units are, for the line that says how many are kept, such as "text encoder layers".
    what: &'static str,
    files: Files,
    units: Vec<Unit>,
    kept: Vec<Kept>,
    /// Units by name, which is the prefix of their tensors' names.
    names: HashMap<String, usize>,
    /// The order in which the units are given up.
    give_up: Vec<usize>,
    /// The units a model runs at every call, whose size sets the regions'.
    sized: Range<usize>,
    regions: Option<Regions>,
    /// Whether the units kept back fit beside the buffers of the calls so far.
    settled: bool,
}

impl Stream {
    /// `units` in the order a call runs them, all held as tensors of their own, as a whole load
    /// leaves them. `give_up` is the order in which they go when the device is short of memory.
    pub(crate) fn new(
        what: &'static str,
        files: Files,
        units: Vec<Unit>,
        give_up: Vec<usize>,
        sized: Range<usize>,
    ) -> Self {
        assert_eq!(give_up.len(), units.len(), "every unit has a place to go");
        Stream {
            names: units
                .iter()
                .enumerate()
                .map(|(index, unit)| (unit.name.clone(), index))
                .collect(),
            kept: units.iter().map(|_| Kept::Tensors).collect(),
            what,
            files,
            units,
            give_up,
            sized,
            regions: None,
            settled: false,
        }
    }

    pub(crate) fn files(&mut self) -> &mut Files {
        &mut self.files
    }

    /// Whether some units are read on the way rather than held as a whole load leaves them.
    pub(crate) fn reads(&self) -> bool {
        self.regions.is_some()
    }

    /// The unit a tensor belongs to: the one whose name is the longest prefix of the tensor's,
    /// up to a dot.
    pub(crate) fn unit_of(&self, name: &str) -> Option<usize> {
        name.match_indices('.')
            .rev()
            .find_map(|(end, _)| self.names.get(&name[..end]).copied())
    }

    /// Adds a tensor to a unit or replaces one of it, which a model does when a patch brings the
    /// tensor. A model that holds the unit as tensors of its own changes those itself, and one
    /// that reads units on the way calls `changed` once it has changed them all.
    pub(crate) fn insert(
        &mut self,
        index: usize,
        name: &str,
        dtype: DType,
        shape: Vec<usize>,
        pieces: Vec<Piece>,
        arrangement: Arrangement,
    ) {
        self.units[index].insert(name, dtype, shape, pieces, arrangement);
    }

    /// Lays the regions out again for units that `insert` changed, when units are read on the way.
    pub(crate) fn changed(&mut self, tensors: &mut DeviceTensors) -> Result<(), Error> {
        match self.reads() {
            true => self.let_go(tensors),
            false => Ok(()),
        }
    }

    /// Lets go of every unit held as a whole load left it, and reads them on the way from now on.
    pub(crate) fn start_reading(&mut self, tensors: &mut DeviceTensors) -> Result<(), Error> {
        self.let_go(tensors)
    }

    /// Makes room on a device that ran out of memory, and says whether there was any to make: a
    /// model holding its units as it loaded them starts reading them on the way, and one that
    /// already does lets go of the units it kept back, to keep back fewer before its next call.
    pub(crate) fn make_room(&mut self, tensors: &mut DeviceTensors) -> Result<bool, Error> {
        if !self.reads() {
            self.start_reading(tensors)?;
            return Ok(true);
        }
        if !self.kept.iter().any(|kept| matches!(kept, Kept::Buffer(_))) {
            return Ok(false);
        }
        self.let_go(tensors)?;
        Ok(true)
    }

    /// Lets go of every unit kept back, sizes the regions for the units as they are now and points
    /// the tensors of every unit at them. The next call keeps back what fits again.
    fn let_go(&mut self, tensors: &mut DeviceTensors) -> Result<(), Error> {
        // The views go before what they point into, so that a model that fails to size the
        // regions below is missing the tensors rather than reading memory it no longer holds.
        for (unit, kept) in self.units.iter().zip(&mut self.kept) {
            for tensor in unit.tensors() {
                tensors.remove(&tensor.name);
            }
            *kept = Kept::Read(Region::Both);
        }
        let largest = self.units.iter().map(Unit::bytes).max().unwrap_or(0);
        let bytes = self.units[self.sized.clone()]
            .iter()
            .map(Unit::bytes)
            .max()
            .unwrap_or(0)
            .max(largest.div_ceil(2))
            .next_multiple_of(mmh3_core::streaming::UNIT_ALIGNMENT);
        let files = self.files.paths().len();
        if self
            .regions
            .as_ref()
            .is_none_or(|regions| regions.bytes != bytes || regions.files != files)
        {
            // The old regions and their thread go first, so that their memory is there for these.
            self.regions = None;
            let device = DeviceBuffer::new(2 * bytes)?;
            self.regions = Some(Regions {
                bytes,
                loader: Loader::start(&self.files, bytes)?,
                device,
                files,
                holds: [None; 2],
            });
        }
        self.assign(tensors);
        self.settled = false;
        Ok(())
    }

    /// Gives every unit read on the way its region, alternating so that one is read while the one
    /// before runs, and points every unit's tensors at where its weights are.
    fn assign(&mut self, tensors: &mut DeviceTensors) {
        let regions = self
            .regions
            .as_mut()
            .expect("units read on the way have regions");
        regions.holds = [None; 2];
        let mut next = 0;
        for (unit, kept) in self.units.iter().zip(&mut self.kept) {
            let (owner, offset) = match kept {
                Kept::Tensors => continue,
                Kept::Buffer(buffer) => (&*buffer, 0),
                Kept::Read(region) => {
                    *region = if unit.bytes() > regions.bytes {
                        Region::Both
                    } else {
                        next += 1;
                        Region::One((next - 1) % 2)
                    };
                    (&regions.device, region.offset(regions.bytes))
                }
            };
            for tensor in unit.tensors() {
                // SAFETY: the unit's layout fits its buffer or its region, and the views go before
                // either does: every change to where a unit is comes back here.
                let view =
                    unsafe { DeviceBuffer::view(owner, offset + tensor.offset, tensor.bytes()) };
                tensors.insert_buffer(&tensor.name, view, tensor.dtype, tensor.shape.clone());
            }
        }
    }

    /// Keeps back as many units as fit beside what the device holds now, which a model calls once
    /// the buffers of a call are there. The units given up last are kept first.
    pub(crate) fn settle(&mut self, tensors: &mut DeviceTensors) -> Result<(), Error> {
        if self.settled || !self.reads() {
            return Ok(());
        }
        let mut kept = Vec::new();
        let mut free = free_bytes()?;
        for &index in self.give_up.iter().rev() {
            let bytes = self.units[index].bytes();
            if free < bytes + HEADROOM {
                break;
            }
            match DeviceBuffer::new(bytes) {
                Ok(buffer) => kept.push((index, buffer)),
                Err(error) if error.is_out_of_memory() => break,
                Err(error) => return Err(error.into()),
            }
            free -= bytes;
        }
        let regions = self
            .regions
            .as_mut()
            .expect("units read on the way have regions");
        // The units kept back pass through the pinned memory one half at a time, or through both
        // for a unit too large for one.
        for (sent, (index, buffer)) in kept.iter().enumerate() {
            let unit = &self.units[*index];
            regions.loader.send(Request {
                unit: unit.clone(),
                destination: Destination(buffer.pointer()),
                halves: if unit.bytes() > regions.bytes {
                    Region::Both.halves()
                } else {
                    Region::One(sent % 2).halves()
                },
                gated: false,
            })?;
        }
        for _ in &kept {
            regions.loader.answer()?;
        }
        // SAFETY: the stream belongs to the loader.
        check(unsafe { mmh3_cuda_stream_synchronize(regions.loader.handles.stream) })?;
        println!(
            "keeping {} of {} {} on the device and reading the rest as they run",
            kept.len(),
            self.units.len(),
            self.what
        );
        for (index, buffer) in kept {
            self.kept[index] = Kept::Buffer(buffer);
        }
        self.assign(tensors);
        self.settled = true;
        Ok(())
    }

    /// A pass over `order`, the units one call runs in the order it runs them.
    pub(crate) fn pass(
        &mut self,
        order: impl IntoIterator<Item = usize>,
    ) -> Result<Pass<'_>, Error> {
        let queue = order
            .into_iter()
            .filter(|&index| matches!(self.kept[index], Kept::Read(_)))
            .collect();
        let mut pass = Pass {
            stream: self,
            queue,
            next: 0,
            busy: [None; 2],
            loading: VecDeque::new(),
        };
        pass.request()?;
        Ok(pass)
    }
}

/// The units read on the way during one call, in the order it runs them.
pub(crate) struct Pass<'a> {
    stream: &'a mut Stream,
    queue: Vec<usize>,
    /// The first unit of the queue not asked for yet.
    next: usize,
    /// The unit each region is taken by, from when it is asked for until it has run.
    busy: [Option<usize>; 2],
    /// Units asked for whose answers have not come yet, oldest first.
    loading: VecDeque<usize>,
}

impl Pass<'_> {
    fn region(&self, index: usize) -> Option<Region> {
        match self.stream.kept[index] {
            Kept::Read(region) => Some(region),
            _ => None,
        }
    }

    /// Asks for the units whose regions are free, in order.
    fn request(&mut self) -> Result<(), Error> {
        while let Some(&index) = self.queue.get(self.next) {
            let region = self
                .region(index)
                .expect("the queue holds units read on the way");
            if region
                .halves()
                .iter()
                .any(|&half| self.busy[half].is_some())
            {
                break;
            }
            self.next += 1;
            for &half in region.halves() {
                self.busy[half] = Some(index);
            }
            let regions = self
                .stream
                .regions
                .as_mut()
                .expect("units read on the way have regions");
            if region
                .halves()
                .iter()
                .all(|&half| regions.holds[half] == Some(index))
            {
                continue;
            }
            for &half in region.halves() {
                regions.holds[half] = None;
            }
            regions.loader.send(Request {
                unit: self.stream.units[index].clone(),
                destination: Destination(regions.device.pointer_at(region.offset(regions.bytes))),
                halves: region.halves(),
                gated: true,
            })?;
            self.loading.push_back(index);
        }
        Ok(())
    }

    /// Waits until `index`'s weights are where its tensors point, before the kernels that read
    /// them are launched.
    pub(crate) fn enter(&mut self, index: usize) -> Result<(), Error> {
        let Some(region) = self.region(index) else {
            return Ok(());
        };
        assert!(
            self.busy.contains(&Some(index)),
            "{} runs out of the order its pass was given",
            self.stream.units[index].name
        );
        while self.loading.contains(&index) {
            let answered = self.loading.pop_front().expect("a unit being loaded");
            let regions = self
                .stream
                .regions
                .as_mut()
                .expect("units read on the way have regions");
            regions.loader.answer()?;
            let region = self.region(answered).expect("a unit read on the way");
            let regions = self
                .stream
                .regions
                .as_mut()
                .expect("units read on the way have regions");
            for &half in region.halves() {
                regions.holds[half] = Some(answered);
            }
        }
        let regions = self
            .stream
            .regions
            .as_ref()
            .expect("units read on the way have regions");
        for &half in region.halves() {
            // SAFETY: the event belongs to the loader, and the null stream is the default stream.
            check(unsafe {
                mmh3_cuda_stream_wait_event(ptr::null_mut(), regions.loader.handles.loaded[half])
            })?;
        }
        Ok(())
    }

    /// Frees `index`'s region for the unit after it once the kernels launched so far are done.
    pub(crate) fn leave(&mut self, index: usize) -> Result<(), Error> {
        let Some(region) = self.region(index) else {
            return Ok(());
        };
        let regions = self
            .stream
            .regions
            .as_ref()
            .expect("units read on the way have regions");
        for &half in region.halves() {
            // SAFETY: the event belongs to the loader, and the null stream is the default stream.
            check(unsafe {
                mmh3_cuda_event_record(regions.loader.handles.done[half], ptr::null_mut())
            })?;
            self.busy[half] = None;
        }
        self.request()
    }
}

impl Drop for Pass<'_> {
    /// Waits for the answers still owed, so that the next pass reads only its own. A unit whose
    /// read failed is held by no region.
    fn drop(&mut self) {
        while let Some(index) = self.loading.pop_front() {
            let region = self.region(index).expect("a unit read on the way");
            let regions = self
                .stream
                .regions
                .as_mut()
                .expect("units read on the way have regions");
            let held = regions.loader.answer().is_ok().then_some(index);
            for &half in region.halves() {
                regions.holds[half] = held;
            }
        }
    }
}
