//! CUDA backend. Kernels and their C ABI launchers live in `kernels/*.cu`.

pub mod attention;
pub mod audio_vae;
pub mod bench;
pub mod dit;
pub mod gemm;
pub mod loader;
pub mod model;
pub mod nvfp4;
pub mod text_encoder;
pub mod vae;
pub mod video_encoder;

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::ptr;

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
    fn mmh3_cuda_error_string(code: c_int) -> *const c_char;
    fn mmh3_cuda_malloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn mmh3_cuda_free(pointer: *mut c_void) -> c_int;
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

/// Free and total device memory in bytes.
pub fn memory_info() -> Result<(usize, usize), CudaError> {
    let (mut free_bytes, mut total_bytes) = (0, 0);
    // SAFETY: both are valid out pointers.
    check(unsafe { mmh3_cuda_memory_info(&mut free_bytes, &mut total_bytes) })?;
    Ok((free_bytes, total_bytes))
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

/// An owned device allocation.
pub struct DeviceBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self, CudaError> {
        let mut pointer = ptr::null_mut();
        // SAFETY: pointer is a valid out pointer.
        check(unsafe { mmh3_cuda_malloc(&mut pointer, bytes) })?;
        Ok(Self { pointer, bytes })
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
        // SAFETY: pointer came from cudaMalloc and is freed once.
        unsafe {
            mmh3_cuda_free(self.pointer);
        }
    }
}
