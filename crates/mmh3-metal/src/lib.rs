//! Apple GPU inference with MPS dense products and MPP packed low-precision products.
//! Buffers and queues are thread-local. Bounded batches complete at host reads or explicit waits.
#![cfg(target_os = "macos")]

pub mod audio_vae;
pub mod bench;
pub mod dit;
mod model;
pub mod ops;
pub mod shard;
pub mod text_encoder;
pub mod vae;

use std::{
    ffi::{CStr, CString, c_char, c_void},
    fmt,
    ptr::NonNull,
    rc::Rc,
};

#[derive(Debug)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self(value.to_string())
    }
}

/// Precision of INT8-checkpoint linear layers. Residuals, attention and LoRAs stay FP32.
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

pub type Result<T> = std::result::Result<T, Error>;

unsafe extern "C" {
    fn mmh3_metal_error() -> *const c_char;
    fn mmh3_metal_create(source: *const c_char, tensor_source: *const c_char) -> *mut c_void;
    fn mmh3_metal_supports_tensor_ops(context: *mut c_void) -> bool;
    fn mmh3_metal_destroy(context: *mut c_void);
    fn mmh3_metal_name(context: *mut c_void) -> *const c_char;
    fn mmh3_metal_synchronize(context: *mut c_void) -> i32;
    fn mmh3_metal_alloc(context: *mut c_void, bytes: usize, data: *const c_void) -> *mut c_void;
    fn mmh3_metal_free(context: *mut c_void, buffer: *mut c_void);
    fn mmh3_metal_stats(context: *mut c_void, output: *mut u64);
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
    // SAFETY: the native runtime returns a thread-local, NUL-terminated error string.
    Error(
        unsafe { CStr::from_ptr(mmh3_metal_error()) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(native_error())
    }
}

struct Context(NonNull<c_void>);
impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: this is the last owner of the context returned by create.
        unsafe { mmh3_metal_destroy(self.0.as_ptr()) }
    }
}

#[derive(Clone)]
pub struct Device(Rc<Context>);
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

    pub fn new() -> Result<Self> {
        let source = CString::new(include_str!("../kernels/ops.metal")).unwrap();
        let tensor_source = CString::new(include_str!("../kernels/matmul.metal")).unwrap();
        // SAFETY: source is a valid string and is copied by the native runtime.
        let pointer =
            NonNull::new(unsafe { mmh3_metal_create(source.as_ptr(), tensor_source.as_ptr()) })
                .ok_or_else(native_error)?;
        Ok(Self(Rc::new(Context(pointer))))
    }

    pub fn name(&self) -> String {
        // SAFETY: the device remains alive while the returned string is copied.
        unsafe { CStr::from_ptr(mmh3_metal_name(self.0.0.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    pub(crate) fn alloc(&self, bytes: usize, data: Option<&[u8]>) -> Result<Buffer> {
        if bytes == 0 || data.is_some_and(|data| data.len() != bytes) {
            return Err(Error(
                "Metal buffers must have a nonzero, matching size".into(),
            ));
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
        Ok(Buffer(Rc::new(Allocation {
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

        if buffers.iter().any(|b| !Rc::ptr_eq(&self.0, &b.0.device.0)) {
            return Err(Error("Metal buffers belong to different devices".into()));
        }

        let group_size = if name.starts_with("mpp_") { 128 } else { 256 };
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

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: this is the final Rust buffer owner. Pending commands retain the Metal buffer.
        unsafe { mmh3_metal_free(self.device.0.0.as_ptr(), self.pointer.as_ptr()) }
    }
}

#[derive(Clone)]
pub(crate) struct Buffer(Rc<Allocation>);
impl Buffer {
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
