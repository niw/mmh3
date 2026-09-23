//! CUDA backend. Kernels and their C ABI launchers live in `kernels/*.cu`.
#![cfg(target_os = "linux")]

pub mod algorithms;
pub mod attention;
pub mod audio_encoder;
pub mod audio_vae;
pub mod bench;
pub mod dit;
pub mod gemm;
pub mod loader;
pub mod model;
pub mod nvfp4;
pub mod shard;
mod streaming;
pub mod text_encoder;
pub mod vae;
pub mod video_encoder;
pub mod vision;

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

// Device buffers hold f32 and other values in the GPU's little-endian byte order, which host slices
// share.
const _: () = assert!(
    cfg!(target_endian = "little"),
    "mmh3 assumes a little-endian host"
);

#[repr(C)]
struct RawDeviceInfo {
    name: [c_char; 256],
    compute_major: i32,
    compute_minor: i32,
    multiprocessor_count: i32,
    shared_memory_per_block_optin: i32,
    shared_memory_per_multiprocessor: i32,
    l2_cache_bytes: i32,
    integrated: i32,
    total_global_bytes: u64,
}

unsafe extern "C" {
    fn mmh3_cuda_device_count(count: *mut c_int) -> c_int;
    fn mmh3_cuda_device_info(device: c_int, info: *mut RawDeviceInfo) -> c_int;
    fn mmh3_cuda_memory_info(free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
    fn mmh3_cuda_reads_host_memory(reads: *mut c_int) -> c_int;
    fn mmh3_cuda_get_device(device: *mut c_int) -> c_int;
    fn mmh3_cuda_set_device(device: c_int) -> c_int;
    fn mmh3_cuda_can_access_peer(device: c_int, peer: c_int, can: *mut c_int) -> c_int;
    fn mmh3_cuda_enable_peer_access(peer: c_int) -> c_int;
    fn mmh3_cuda_copy_across(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn mmh3_cuda_error_string(code: c_int) -> *const c_char;
    fn mmh3_cuda_malloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn mmh3_cuda_free_on(device: c_int, pointer: *mut c_void) -> c_int;
    fn mmh3_cuda_copy_to_device(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn mmh3_cuda_copy_to_host(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn mmh3_cuda_copy_device(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
    ) -> c_int;
    fn mmh3_cuda_memset(pointer: *mut c_void, value: c_int, bytes: usize) -> c_int;
    fn mmh3_cuda_synchronize() -> c_int;
    fn mmh3_cublaslt_status_string(code: c_int) -> *const c_char;
    fn mmh3_cuda_fill_f32(data: *mut f32, value: f32, count: usize) -> c_int;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaError {
    pub code: i32,
    pub message: String,
}

/// `cudaErrorMemoryAllocation`, which a caller holding memory it could let go of can do something
/// about, unlike every other way a call here fails.
pub const OUT_OF_MEMORY: i32 = 2;

/// `cudaErrorInvalidDevice`, which is also what this build answers for a device it cannot index.
pub const INVALID_DEVICE: i32 = 101;

/// The most devices this process computes on. It matches `MMH3_MAX_DEVICES` in `device.cuh`, which
/// caches the same kind of per-device answer on the other side of the ABI.
pub const MAX_DEVICES: usize = 16;

impl CudaError {
    pub fn is_out_of_memory(&self) -> bool {
        self.code == OUT_OF_MEMORY
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "CUDA error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for CudaError {}

pub(crate) fn check(code: c_int) -> Result<(), CudaError> {
    if code == 0 {
        return Ok(());
    }
    // SAFETY: both functions return static NUL-terminated strings. Codes from 1000 up are cuBLAS
    // statuses.
    let message = unsafe {
        CStr::from_ptr(if code >= 1000 {
            mmh3_cublaslt_status_string(code)
        } else {
            mmh3_cuda_error_string(code)
        })
    }
    .to_string_lossy()
    .into_owned();
    Err(CudaError { code, message })
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub shared_memory_per_block_optin: i32,
    pub shared_memory_per_multiprocessor: i32,
    pub l2_cache_bytes: i32,
    pub integrated: bool,
    pub total_global_bytes: u64,
}

pub fn device_count() -> Result<usize, CudaError> {
    let mut count = 0;
    // SAFETY: count is a valid out pointer.
    check(unsafe { mmh3_cuda_device_count(&mut count) })?;
    Ok(count as usize)
}

pub fn device_info(device: usize) -> Result<DeviceInfo, CudaError> {
    let mut raw = std::mem::MaybeUninit::<RawDeviceInfo>::zeroed();
    // SAFETY: raw is a writable RawDeviceInfo with the same layout as the C struct.
    check(unsafe { mmh3_cuda_device_info(device as c_int, raw.as_mut_ptr()) })?;
    // SAFETY: the call above initialized every field.
    let raw = unsafe { raw.assume_init() };
    // SAFETY: the launcher always NUL-terminates the name.
    let name = unsafe { CStr::from_ptr(raw.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(DeviceInfo {
        name,
        compute_capability: (raw.compute_major, raw.compute_minor),
        multiprocessor_count: raw.multiprocessor_count,
        shared_memory_per_block_optin: raw.shared_memory_per_block_optin,
        shared_memory_per_multiprocessor: raw.shared_memory_per_multiprocessor,
        l2_cache_bytes: raw.l2_cache_bytes,
        integrated: raw.integrated != 0,
        total_global_bytes: raw.total_global_bytes,
    })
}

/// The device this thread computes on.
///
/// NOTE: the current device belongs to the host thread rather than to the process, so a thread
/// that has not chosen one is on device 0 whatever another thread is doing. A device this build
/// cannot index is refused rather than counted against another one.
pub fn current_device() -> Result<usize, CudaError> {
    let mut device = 0;
    // SAFETY: device is a valid out pointer.
    check(unsafe { mmh3_cuda_get_device(&mut device) })?;
    let device = device as usize;
    if device >= MAX_DEVICES {
        return Err(CudaError {
            code: INVALID_DEVICE,
            message: format!(
                "device {device} is past the {MAX_DEVICES} this build computes on at once"
            ),
        });
    }
    Ok(device)
}

/// Computes on `device` from this thread on, so every allocation and launch that follows goes
/// there. A rank of a shared step binds the device of the thread that runs it and lets it be.
pub fn set_device(device: usize) -> Result<(), CudaError> {
    if device >= MAX_DEVICES {
        return Err(CudaError {
            code: INVALID_DEVICE,
            message: format!(
                "device {device} is past the {MAX_DEVICES} this build computes on at once"
            ),
        });
    }
    // SAFETY: no pointers.
    check(unsafe { mmh3_cuda_set_device(device as c_int) })
}

/// Whether `device` can read `peer`'s memory directly, which is what makes an exchange with another
/// card of this process a copy between the two rather than one through the host.
pub fn can_access_peer(device: usize, peer: usize) -> Result<bool, CudaError> {
    let mut can = 0;
    // SAFETY: can is a valid out pointer.
    check(unsafe { mmh3_cuda_can_access_peer(device as c_int, peer as c_int, &mut can) })?;
    Ok(can != 0)
}

/// Lets the device this thread computes on read `peer`'s memory. A pair that has it already is no
/// error, and a pair that cannot have it at all says so.
pub fn enable_peer_access(peer: usize) -> Result<(), CudaError> {
    // SAFETY: no pointers.
    check(unsafe { mmh3_cuda_enable_peer_access(peer as c_int) })
}

/// Copies `bytes` bytes between two addresses this process holds, on one card, between two cards
/// or through the host, in order with the kernels on the default stream of the device this thread
/// computes on. It returns before the copy is done, which `synchronize` waits for.
///
/// # Safety
/// Both ranges must lie inside live allocations of this process.
pub unsafe fn copy_across(
    destination: *mut c_void,
    source: *const c_void,
    bytes: usize,
) -> Result<(), CudaError> {
    // SAFETY: the caller keeps both ranges inside their allocations.
    check(unsafe { mmh3_cuda_copy_across(destination, source, bytes) })
}

/// Free and total device memory in bytes.
///
/// NOTE: this is the whole device's, not this process's. What another program on the same GPU
/// holds is missing from the free figure, and what this process holds is missing from it too.
/// The Metal backend answers a different question under the same name, since a Mac has no figure
/// like this one to give.
pub fn memory_info() -> Result<(usize, usize), CudaError> {
    let (mut free_bytes, mut total_bytes) = (0, 0);
    // SAFETY: both are valid out pointers.
    check(unsafe { mmh3_cuda_memory_info(&mut free_bytes, &mut total_bytes) })?;
    Ok((free_bytes, total_bytes))
}

/// Whether a kernel on this thread's device reads ordinary host memory where it lies, through the
/// host's own page tables, as a GB10 or a Grace GPU does. A discrete GPU does not: without HMM it
/// cannot read such memory at all, and with it every access faults its way across PCIe.
pub fn reads_host_memory() -> Result<bool, CudaError> {
    let mut reads = 0;
    // SAFETY: a valid out pointer.
    check(unsafe { mmh3_cuda_reads_host_memory(&mut reads) })?;
    Ok(reads != 0)
}

/// What `memory_info` answers for `device`, from whichever device this thread computes on, which
/// it is left computing on afterwards.
pub fn memory_info_on(device: usize) -> Result<(usize, usize), CudaError> {
    let here = current_device()?;
    set_device(device)?;
    let answer = memory_info();
    set_device(here)?;
    answer
}

pub fn synchronize() -> Result<(), CudaError> {
    // SAFETY: no arguments.
    check(unsafe { mmh3_cuda_synchronize() })
}

/// Copies `bytes` bytes between device addresses, in order with the kernels on the default stream.
///
/// # Safety
/// Both ranges must lie inside live device allocations.
pub(crate) unsafe fn copy_device(
    destination: *mut c_void,
    source: *const c_void,
    bytes: usize,
) -> Result<(), CudaError> {
    // SAFETY: the caller keeps both ranges inside their allocations.
    check(unsafe { mmh3_cuda_copy_device(destination, source, bytes) })
}

/// Copies host bytes into device memory, in order with the kernels on the default stream.
///
/// # Safety
/// The destination must hold `source.len()` bytes inside a live device allocation.
pub unsafe fn upload(destination: *mut c_void, source: &[u8]) -> Result<(), CudaError> {
    // SAFETY: the caller keeps the destination inside its allocation.
    check(unsafe { mmh3_cuda_copy_to_device(destination, source.as_ptr().cast(), source.len()) })
}

/// Copies device memory into host bytes.
///
/// # Safety
/// The source must hold `destination.len()` bytes inside a live device allocation.
pub unsafe fn download(destination: &mut [u8], source: *const c_void) -> Result<(), CudaError> {
    // SAFETY: the caller keeps the source inside its allocation.
    check(unsafe {
        mmh3_cuda_copy_to_host(destination.as_mut_ptr().cast(), source, destination.len())
    })
}

/// Bytes in live device allocations, per device. Everything a model and its work allocate comes
/// through `DeviceBuffer`, so this is what this process is holding, which the device does not say:
/// what it reports free is what every program on the GPU has left between them.
static ALLOCATED: [AtomicUsize; MAX_DEVICES] = [const { AtomicUsize::new(0) }; MAX_DEVICES];
/// The most this process will hold on one device, or zero for as much as the device will give.
static LIMIT: [AtomicUsize; MAX_DEVICES] = [const { AtomicUsize::new(0) }; MAX_DEVICES];

/// Bytes in live device allocations on the device this thread computes on.
pub fn allocated_bytes() -> usize {
    current_device().map_or(0, |device| ALLOCATED[device].load(Ordering::Relaxed))
}

/// Holds this process to `bytes` on every device, beyond which an allocation fails as it would on a
/// device that small. Zero lifts the limit.
///
/// NOTE: the budget is what one device may give rather than what they may give between them, which
/// is what `--vram-budget` means on a machine that answers for a smaller one.
pub fn set_allocation_limit(bytes: usize) {
    for limit in &LIMIT {
        limit.store(bytes, Ordering::Relaxed);
    }
}

/// Bytes this process may still allocate on the device this thread computes on: what the device
/// has left, held to what the limit leaves.
pub fn free_bytes() -> Result<usize, CudaError> {
    let device = current_device()?;
    let (free, _) = memory_info()?;
    Ok(match LIMIT[device].load(Ordering::Relaxed) {
        0 => free,
        limit => free.min(limit.saturating_sub(ALLOCATED[device].load(Ordering::Relaxed))),
    })
}

/// A device allocation, or a view into one that somebody else owns.
pub struct DeviceBuffer {
    pointer: *mut c_void,
    bytes: usize,
    owned: bool,
    /// The device it was allocated on, which is where it is freed and where a kernel may read it.
    device: usize,
}

// SAFETY: a device address is unique across the process rather than to a thread, and every
// kernel and copy here is submitted to the legacy default stream, which orders the work of all
// threads on one device against each other. So a buffer means the same thing on whichever thread
// reads it, and a model built from buffers can be kept in one place and used from the thread that
// asks for it. Two threads using one model at the same time would still be two runs over one set
// of scratch buffers, which is what the table that hands models out prevents by handing one out at
// a time, per device.
//
// NOTE: the buffer carries the device it was allocated on, since the thread that drops it need not
// be computing on that one, and a kernel reading it has to be on that device or a peer of it.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self, CudaError> {
        let device = current_device()?;
        match LIMIT[device].load(Ordering::Relaxed) {
            0 => {}
            limit if ALLOCATED[device].load(Ordering::Relaxed) + bytes > limit => {
                return Err(CudaError {
                    code: OUT_OF_MEMORY,
                    message: format!(
                        "{bytes} bytes would take this process past its limit on device {device}"
                    ),
                });
            }
            _ => {}
        }
        let mut pointer = ptr::null_mut();
        // SAFETY: pointer is a valid out pointer.
        check(unsafe { mmh3_cuda_malloc(&mut pointer, bytes) })?;
        ALLOCATED[device].fetch_add(bytes, Ordering::Relaxed);
        Ok(Self {
            pointer,
            bytes,
            owned: true,
            device,
        })
    }

    /// `bytes` bytes from `offset` into `owner`, freed with it rather than with the view.
    ///
    /// # Safety
    /// The view must not outlive the allocation it points into.
    pub(crate) unsafe fn view(owner: &DeviceBuffer, offset: usize, bytes: usize) -> Self {
        assert!(
            offset + bytes <= owner.bytes,
            "the view reaches past the end of the buffer"
        );
        Self {
            pointer: owner.pointer_at(offset),
            bytes,
            owned: false,
            device: owner.device,
        }
    }

    /// A buffer filled with zero bytes.
    pub fn zeroed(bytes: usize) -> Result<Self, CudaError> {
        let buffer = Self::new(bytes)?;
        // SAFETY: the allocation holds `bytes` bytes.
        check(unsafe { mmh3_cuda_memset(buffer.pointer, 0, bytes) })?;
        Ok(buffer)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CudaError> {
        let mut buffer = Self::new(bytes.len())?;
        buffer.copy_from_host(bytes)?;
        Ok(buffer)
    }

    pub fn from_f32(values: &[f32]) -> Result<Self, CudaError> {
        // SAFETY: the byte view covers exactly the storage of the slice.
        Self::from_bytes(unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        })
    }

    pub fn to_f32(&self) -> Result<Vec<f32>, CudaError> {
        self.to_f32_range(0, self.bytes / 4)
    }

    /// Copies `count` f32 values starting at value `first`.
    pub fn to_f32_range(&self, first: usize, count: usize) -> Result<Vec<f32>, CudaError> {
        let mut values = vec![0.0f32; count];
        // SAFETY: the byte view covers exactly the storage of the vector, and every bit pattern is
        // a valid f32.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), count * 4) };
        self.copy_range_to_host(first * 4, bytes)?;
        Ok(values)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The device this buffer lives on.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Raw CUDA device address for native API interoperability. This is not a host pointer.
    pub fn pointer(&self) -> *mut c_void {
        self.pointer
    }

    /// Pointer `offset` bytes into the allocation.
    pub(crate) fn pointer_at(&self, offset: usize) -> *mut c_void {
        assert!(offset <= self.bytes, "offset is past the end of the buffer");
        // SAFETY: the offset stays within the allocation, checked above.
        unsafe { self.pointer.cast::<u8>().add(offset).cast() }
    }

    /// Copies `source` to `offset` bytes into the allocation.
    pub fn copy_from_host_at(&mut self, offset: usize, source: &[u8]) -> Result<(), CudaError> {
        assert!(
            offset + source.len() <= self.bytes,
            "the range reaches past the end of the buffer"
        );
        // SAFETY: the range lies inside the allocation, checked above.
        check(unsafe {
            mmh3_cuda_copy_to_device(
                self.pointer_at(offset),
                source.as_ptr().cast(),
                source.len(),
            )
        })
    }

    pub fn copy_from_host(&mut self, source: &[u8]) -> Result<(), CudaError> {
        assert_eq!(source.len(), self.bytes, "host and device sizes differ");
        // SAFETY: both ranges are valid for self.bytes bytes.
        check(unsafe { mmh3_cuda_copy_to_device(self.pointer, source.as_ptr().cast(), self.bytes) })
    }

    /// Copies `destination.len()` bytes starting `offset` bytes into the allocation.
    pub fn copy_range_to_host(
        &self,
        offset: usize,
        destination: &mut [u8],
    ) -> Result<(), CudaError> {
        assert!(
            offset + destination.len() <= self.bytes,
            "the range reaches past the end of the buffer"
        );
        // SAFETY: the range lies inside the allocation, checked above.
        check(unsafe {
            mmh3_cuda_copy_to_host(
                destination.as_mut_ptr().cast(),
                self.pointer_at(offset),
                destination.len(),
            )
        })
    }

    pub fn copy_to_host(&self, destination: &mut [u8]) -> Result<(), CudaError> {
        assert_eq!(
            destination.len(),
            self.bytes,
            "host and device sizes differ"
        );
        // SAFETY: both ranges are valid for self.bytes bytes.
        check(unsafe {
            mmh3_cuda_copy_to_host(destination.as_mut_ptr().cast(), self.pointer, self.bytes)
        })
    }

    pub fn fill_f32(&mut self, value: f32) -> Result<(), CudaError> {
        // SAFETY: the kernel writes only within the allocation.
        check(unsafe { mmh3_cuda_fill_f32(self.pointer.cast(), value, self.bytes / 4) })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        ALLOCATED[self.device].fetch_sub(self.bytes, Ordering::Relaxed);
        // SAFETY: pointer came from cudaMalloc on self.device and is freed once.
        unsafe {
            mmh3_cuda_free_on(self.device as c_int, self.pointer);
        }
    }
}
