//! Apple GPU inference with MPS dense products and MPP packed low-precision products.
//! One device serves the whole process, one thread at a time, and a buffer belongs to it rather
//! than to a thread. Bounded batches complete at host reads or explicit waits.
#![cfg(target_os = "macos")]

pub mod audio_vae;
pub mod bench;
pub mod dit;
mod model;
pub mod ops;
pub mod shard;
mod streaming;
pub mod text_encoder;
pub mod vae;

use std::{
    ffi::{CStr, CString, c_char, c_void},
    fmt,
    ptr::NonNull,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
};

/// Something a call here could not do, and what it says for itself.
#[derive(Debug)]
pub struct Error {
    pub message: String,
    out_of_memory: bool,
}

impl Error {
    /// A failure whose only answer is to report it.
    pub fn new(message: String) -> Self {
        Self {
            message,
            out_of_memory: false,
        }
    }

    /// The device having no room for something, which a caller holding memory it could let go of
    /// can do something about.
    pub fn out_of_memory(message: String) -> Self {
        Self {
            message,
            out_of_memory: true,
        }
    }

    /// Whether this is the device saying it has no memory left, which a caller holding memory it
    /// could let go of can do something about, unlike every other way a call here fails.
    pub fn is_out_of_memory(&self) -> bool {
        self.out_of_memory
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::new(value.to_string())
    }
}

/// Precision of INT8-checkpoint linear layers. Residuals stay FP32, and attention takes a
/// precision of its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LinearPrecision {
    #[default]
    Fp32,

    /// MPP FP16 activations × INT8 weights, with FP32 accumulation and output.
    Fp16,

    /// MPP dynamically quantized INT8 activations × INT8 weights, with INT32 accumulation.
    Int8,

    /// FP16 inputs through MPS, for comparison with the direct MPP path.
    MpsFp16,
}

/// Precision of dense attention over heads of 128, which the DiT and the text encoder use. Other
/// head widths use FP32 either way.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AttentionPrecision {
    #[default]
    Fp32,

    /// FP16 products on the matrix units, with FP32 softmax and accumulation. Needs macOS 26.
    Fp16,
}

pub type Result<T> = std::result::Result<T, Error>;

unsafe extern "C" {
    fn mmh3_metal_error() -> *const c_char;
    fn mmh3_metal_out_of_memory() -> bool;
    fn mmh3_metal_create(source: *const c_char, tensor_source: *const c_char) -> *mut c_void;
    fn mmh3_metal_supports_tensor_ops(context: *mut c_void) -> bool;
    fn mmh3_metal_destroy(context: *mut c_void);
    fn mmh3_metal_name(context: *mut c_void) -> *const c_char;
    fn mmh3_metal_synchronize(context: *mut c_void) -> i32;
    fn mmh3_metal_alloc(context: *mut c_void, bytes: usize, data: *const c_void) -> *mut c_void;
    fn mmh3_metal_free(context: *mut c_void, buffer: *mut c_void);
    fn mmh3_metal_stats(context: *mut c_void, output: *mut u64);
    fn mmh3_metal_memory_info(output: *mut u64) -> i32;
    fn mmh3_metal_read(buffer: *mut c_void, output: *mut c_void, bytes: usize);
    fn mmh3_metal_dispatch(
        context: *mut c_void,
        name: *const c_char,
        buffers: *const *mut c_void,
        count: usize,
        parameters: *const c_void,
        bytes: usize,
        threads: usize,
        group_size: usize,
        groups: bool,
    ) -> i32;
    fn mmh3_metal_matmul(
        context: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        m: usize,
        n: usize,
        k: usize,
        stride: usize,
        offset: usize,
        half_inputs: bool,
    ) -> i32;
}

fn native_error() -> Error {
    // SAFETY: the native runtime returns a thread-local, NUL-terminated error string and the
    // reason beside it, both of the failure this thread has just had.
    Error {
        message: unsafe { CStr::from_ptr(mmh3_metal_error()) }
            .to_string_lossy()
            .into_owned(),
        out_of_memory: unsafe { mmh3_metal_out_of_memory() },
    }
}

/// Bytes in live allocations of this process.
///
/// NOTE: a buffer this process has let go of is not always a buffer the device has back. The
/// runtime keeps a small pool of them to hand out again, and what that pool holds is counted here
/// as free, since a buffer of the same size is served from it without asking the device at all.
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// The most this process will hold, or zero for as much as the device will give.
static LIMIT: AtomicUsize = AtomicUsize::new(0);

/// What a Metal device leaves this process and what it recommends one process fill, in bytes.
///
/// NOTE: the recommendation is not the memory the machine has. It is what one process may hold
/// before the system starts taking memory back from it, which on a Mac is most of the memory but
/// not all of it. What it leaves is that less what this process has already taken, so what other
/// programs on the same Mac hold is not counted against it.
pub fn memory_info() -> Result<(usize, usize)> {
    let mut values = [0; 2];
    // SAFETY: the native function writes exactly two counters and retains no pointer.
    check(unsafe { mmh3_metal_memory_info(values.as_mut_ptr()) })?;
    let [working_set, allocated] = values.map(|value| value as usize);
    Ok((working_set.saturating_sub(allocated), working_set))
}

/// Bytes this process may still fill: what the device leaves it, held to what the limit leaves.
pub fn free_bytes() -> Result<usize> {
    let (free, _) = memory_info()?;
    Ok(match LIMIT.load(Ordering::Relaxed) {
        0 => free,
        limit => free.min(limit.saturating_sub(ALLOCATED.load(Ordering::Relaxed))),
    })
}

/// Bytes in live allocations of this process.
pub fn allocated_bytes() -> usize {
    ALLOCATED.load(Ordering::Relaxed)
}

/// Holds this process to `bytes`, beyond which an allocation fails as it would on a device that
/// small. Zero lifts the limit.
pub fn set_allocation_limit(bytes: usize) {
    LIMIT.store(bytes, Ordering::Relaxed);
}

fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(native_error())
    }
}

struct Context(NonNull<c_void>);

// SAFETY: the native context takes its own lock for the whole of every call that touches it, so
// two threads calling here make two calls one after the other rather than one call twice. Nothing
// it holds — the queue, the compiled pipelines, the batch being encoded — is reached except
// through those calls, and the pointer means the same on whichever thread holds it.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of the context returned by create.
        unsafe { mmh3_metal_destroy(self.0.as_ptr()) }
    }
}

#[derive(Clone)]
pub struct Device(Arc<Context>);
#[derive(Debug, Clone, Copy)]
pub struct DeviceStats {
    pub command_buffers: u64,
    pub buffer_allocations: u64,
    pub buffer_reuses: u64,

    /// Device allocation high-water mark observed when this context allocates buffers.
    pub peak_allocated_bytes: u64,
}

impl Device {
    pub fn supports_tensor_ops(&self) -> bool {
        // SAFETY: this device owns the live native context.
        unsafe { mmh3_metal_supports_tensor_ops(self.0.0.as_ptr()) }
    }

    pub fn stats(&self) -> DeviceStats {
        let mut values = [0; 4];
        // SAFETY: the native function writes exactly four counters and retains no pointer.
        unsafe { mmh3_metal_stats(self.0.0.as_ptr(), values.as_mut_ptr()) };
        DeviceStats {
            command_buffers: values[0],
            buffer_allocations: values[1],
            buffer_reuses: values[2],
            peak_allocated_bytes: values[3],
        }
    }

    /// Complete pending GPU work and report asynchronous execution errors.
    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: this thread owns the live context and all work is submitted to its queue.
        check(unsafe { mmh3_metal_synchronize(self.0.0.as_ptr()) })
    }

    /// The device this process computes on, made on the first ask and kept.
    ///
    /// A buffer belongs to the context that made it and no other context may read it, so memory a
    /// block computes in and memory an exchange hands round have to come from one device. Making
    /// a second one also pays to compile the kernels again, which is not free. One device for the
    /// process is also what lets a model outlive the thread that read it, so a machine asked for
    /// the same checkpoint twice reads it once.
    pub fn shared() -> Result<Self> {
        static SHARED: Mutex<Option<Device>> = Mutex::new(None);

        let mut shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        match shared.as_ref() {
            Some(device) => Ok(device.clone()),
            None => Ok(shared.insert(Self::new()?).clone()),
        }
    }

    pub fn new() -> Result<Self> {
        let source = CString::new(include_str!("../kernels/ops.metal")).unwrap();
        let tensor_source = CString::new(include_str!("../kernels/matmul.metal")).unwrap();
        // SAFETY: source is a valid string and is copied by the native runtime.
        let pointer =
            NonNull::new(unsafe { mmh3_metal_create(source.as_ptr(), tensor_source.as_ptr()) })
                .ok_or_else(native_error)?;
        Ok(Self(Arc::new(Context(pointer))))
    }

    pub fn name(&self) -> String {
        // SAFETY: the device remains alive while the returned string is copied.
        unsafe { CStr::from_ptr(mmh3_metal_name(self.0.0.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    pub(crate) fn alloc(&self, bytes: usize, data: Option<&[u8]>) -> Result<Buffer> {
        if bytes == 0 || data.is_some_and(|data| data.len() != bytes) {
            return Err(Error::new(
                "Metal buffers must have a nonzero, matching size".into(),
            ));
        }

        match LIMIT.load(Ordering::Relaxed) {
            0 => {}
            limit if ALLOCATED.load(Ordering::Relaxed) + bytes > limit => {
                return Err(Error {
                    message: format!("{bytes} bytes would take this process past its limit"),
                    out_of_memory: true,
                });
            }
            _ => {}
        }
        // SAFETY: optional data covers bytes, and the native call copies it before returning.
        let pointer = NonNull::new(unsafe {
            mmh3_metal_alloc(
                self.0.0.as_ptr(),
                bytes,
                data.map_or(std::ptr::null(), |data| data.as_ptr().cast()),
            )
        })
        .ok_or_else(native_error)?;
        ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
        Ok(Buffer(Arc::new(Allocation {
            pointer,
            bytes,
            device: self.clone(),
        })))
    }

    pub(crate) fn run(
        &self,
        name: &str,
        buffers: &[&Buffer],
        params: &[u32],
        threads: usize,
        groups: bool,
    ) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }

        if buffers.iter().any(|b| !Arc::ptr_eq(&self.0, &b.0.device.0)) {
            return Err(Error::new(
                "Metal buffers belong to different devices".into(),
            ));
        }

        // The tensor kernels run on the number of SIMD groups their products share.
        let group_size = match name {
            "mpp_int8" | "mpp_lora" => 128,
            _ => 256,
        };
        let name = CString::new(name).unwrap();
        let pointers: Vec<_> = buffers.iter().map(|b| b.0.pointer.as_ptr()).collect();
        // SAFETY: internal callers validate the kernel's shapes and buffer extents. The call is
        // encoder copies parameters and the command buffer retains all bound buffers.
        check(unsafe {
            mmh3_metal_dispatch(
                self.0.0.as_ptr(),
                name.as_ptr(),
                pointers.as_ptr(),
                pointers.len(),
                params.as_ptr().cast(),
                std::mem::size_of_val(params),
                threads,
                group_size,
                groups,
            )
        })
    }
}

struct Allocation {
    pointer: NonNull<c_void>,
    bytes: usize,
    device: Device,
}

// SAFETY: a Metal buffer belongs to the context that made it rather than to a thread, and every
// call that encodes work over one, or frees it, goes through that context. So a buffer means the
// same on whichever thread reads it, and a model built from buffers can be kept in one place and
// used from the thread that asks for it. Two threads using one model at the same time would still
// be two runs over one set of scratch buffers, which is what the table that hands models out
// prevents by handing one out at a time.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        ALLOCATED.fetch_sub(self.bytes, Ordering::Relaxed);
        // SAFETY: this is the final Rust buffer owner. Pending commands retain the Metal buffer.
        unsafe { mmh3_metal_free(self.device.0.0.as_ptr(), self.pointer.as_ptr()) }
    }
}

#[derive(Clone)]
pub(crate) struct Buffer(Arc<Allocation>);
impl Buffer {
    /// Copies `bytes` of `source`, `from` bytes in, to `into` bytes into this one.
    pub(crate) fn copy_range(
        &self,
        source: &Buffer,
        from: usize,
        into: usize,
        bytes: usize,
    ) -> Result<()> {
        if from + bytes > source.0.bytes || into + bytes > self.0.bytes {
            return Err(Error::new("a copy outside a buffer".into()));
        }
        self.0.device.run(
            "copy_bytes",
            &[source, self],
            &[bytes as u32, from as u32, into as u32],
            bytes,
            false,
        )
    }

    /// The bytes this buffer holds, which is how a region reaches the host memory a peer reads.
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>> {
        self.0.device.synchronize()?;
        let mut data = vec![0u8; self.0.bytes];
        // SAFETY: the destination covers the entire allocation. Commands have completed.
        unsafe {
            mmh3_metal_read(
                self.0.pointer.as_ptr(),
                data.as_mut_ptr().cast(),
                self.0.bytes,
            )
        }
        Ok(data)
    }

    pub(crate) fn to_f32(&self) -> Result<Vec<f32>> {
        self.0.device.synchronize()?;
        let mut data = vec![0.0; self.0.bytes / 4];
        // SAFETY: the destination covers the entire allocation. Commands have completed.
        unsafe {
            mmh3_metal_read(
                self.0.pointer.as_ptr(),
                data.as_mut_ptr().cast(),
                self.0.bytes,
            )
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::Device;

    /// Memory no device has is the one failure a caller can answer by letting go of something.
    #[test]
    fn an_impossible_allocation_says_it_ran_out_of_memory() {
        let device = Device::new().unwrap();
        let Err(error) = device.alloc(1 << 48, None) else {
            panic!("a device found 256 TB");
        };
        assert!(error.is_out_of_memory(), "{error}");
    }
}
