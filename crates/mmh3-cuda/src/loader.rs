//! Streams checkpoint tensors into device buffers: direct reads of the file into two pinned staging buffers,
//! overlapped with asynchronous copies to the GPU, so loading neither copies from pageable memory nor fills the page
//! cache.

use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::direct_file::{DIRECT_ALIGNMENT, DirectFile};
use mmh3_core::safetensors::{SafeTensors, TensorInfo};
use std::ffi::{c_int, c_void};
use std::io;
use std::path::PathBuf;
use std::ptr;

unsafe extern "C" {
    fn mmh3_cuda_host_alloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    fn mmh3_cuda_host_free(pointer: *mut c_void) -> c_int;
    fn mmh3_cuda_stream_create(stream: *mut *mut c_void) -> c_int;
    fn mmh3_cuda_stream_destroy(stream: *mut c_void) -> c_int;
    fn mmh3_cuda_stream_synchronize(stream: *mut c_void) -> c_int;
    fn mmh3_cuda_event_create(event: *mut *mut c_void) -> c_int;
    fn mmh3_cuda_event_destroy(event: *mut c_void) -> c_int;
    fn mmh3_cuda_event_record(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn mmh3_cuda_event_synchronize(event: *mut c_void) -> c_int;
    fn mmh3_cuda_copy_to_device_async(destination: *mut c_void, source: *const c_void, bytes: usize, stream: *mut c_void) -> c_int;
}

/// Bytes each staging buffer holds, a multiple of the direct-read alignment.
const STAGING_BYTES: usize = 64 << 20;
/// Queued ranges closer than this are read in one pass rather than seeking over the gap.
const MERGE_GAP: u64 = 4 << 20;

#[derive(Debug)]
pub enum LoadError {
    Cuda(CudaError),
    Io(io::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Cuda(error) => write!(formatter, "{error}"),
            LoadError::Io(error) => write!(formatter, "reading weights: {error}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<CudaError> for LoadError {
    fn from(error: CudaError) -> Self {
        LoadError::Cuda(error)
    }
}

impl From<io::Error> for LoadError {
    fn from(error: io::Error) -> Self {
        LoadError::Io(error)
    }
}

struct Upload {
    file_offset: u64,
    length: usize,
    destination: *mut c_void,
}

/// Pinned host memory freed on drop.
struct PinnedBuffer {
    pointer: *mut c_void,
    bytes: usize,
}

impl PinnedBuffer {
    fn new(bytes: usize) -> Result<Self, CudaError> {
        let mut pointer = ptr::null_mut();
        // SAFETY: pointer is a valid out pointer.
        check(unsafe { mmh3_cuda_host_alloc(&mut pointer, bytes) })?;
        Ok(PinnedBuffer { pointer, bytes })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation holds `bytes` bytes and lives as long as self.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.cast(), self.bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer came from cudaHostAlloc and is freed once.
        unsafe {
            mmh3_cuda_host_free(self.pointer);
        }
    }
}

/// A CUDA stream and one event per staging buffer, destroyed on drop.
struct CopyQueue {
    stream: *mut c_void,
    events: [*mut c_void; 2],
}

impl CopyQueue {
    fn new() -> Result<Self, CudaError> {
        let mut queue = CopyQueue { stream: ptr::null_mut(), events: [ptr::null_mut(); 2] };
        // SAFETY: the fields are valid out pointers, and drop releases whatever was created.
        unsafe {
            check(mmh3_cuda_stream_create(&mut queue.stream))?;
            for event in &mut queue.events {
                check(mmh3_cuda_event_create(event))?;
            }
        }
        Ok(queue)
    }
}

impl Drop for CopyQueue {
    fn drop(&mut self) {
        // SAFETY: the handles were created by this queue and are destroyed once.
        unsafe {
            if !self.stream.is_null() {
                mmh3_cuda_stream_synchronize(self.stream);
            }
            for &event in &self.events {
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

/// Collects tensors of one file to upload and streams them in file order.
pub(crate) struct Uploader {
    path: PathBuf,
    uploads: Vec<Upload>,
}

impl Uploader {
    pub(crate) fn new(file: &SafeTensors) -> Self {
        Uploader { path: file.path().to_owned(), uploads: Vec::new() }
    }

    /// Queues the bytes of `info` for `offset` bytes into `buffer`. The buffer must outlive `run`.
    pub(crate) fn queue(&mut self, file: &SafeTensors, info: &TensorInfo, buffer: &DeviceBuffer, offset: usize) {
        assert!(file.path() == self.path, "the tensor comes from another file");
        assert!(offset + info.byte_count() <= buffer.bytes(), "the tensor does not fit in its buffer");
        if info.byte_count() > 0 {
            self.uploads.push(Upload { file_offset: file.file_offset(info), length: info.byte_count(), destination: buffer.pointer_at(offset) });
        }
    }

    /// Allocates a device buffer for `info` and queues its bytes.
    pub(crate) fn allocate(&mut self, file: &SafeTensors, info: &TensorInfo) -> Result<DeviceBuffer, CudaError> {
        let buffer = DeviceBuffer::new(info.byte_count())?;
        self.queue(file, info, &buffer, 0);
        Ok(buffer)
    }

    /// Streams every queued tensor and returns the bytes uploaded.
    pub(crate) fn run(mut self) -> Result<usize, LoadError> {
        self.uploads.sort_by_key(|upload| upload.file_offset);
        let file = DirectFile::open(&self.path)?;
        let mut staging = [PinnedBuffer::new(STAGING_BYTES)?, PinnedBuffer::new(STAGING_BYTES)?];
        let queue = CopyQueue::new()?;

        // Aligned ranges covering the uploads, merged across small gaps, with the end of the data each one needs.
        let alignment = DIRECT_ALIGNMENT as u64;
        let mut ranges: Vec<(u64, u64, u64)> = Vec::new();
        for upload in &self.uploads {
            let data_end = upload.file_offset + upload.length as u64;
            let start = upload.file_offset / alignment * alignment;
            let end = data_end.div_ceil(alignment) * alignment;
            match ranges.last_mut() {
                Some(last) if start <= last.1 + MERGE_GAP => {
                    last.1 = last.1.max(end);
                    last.2 = last.2.max(data_end);
                }
                _ => ranges.push((start, end, data_end)),
            }
        }

        let mut next_upload = 0;
        let mut chunk = 0usize;
        for (range_start, range_end, data_end) in ranges {
            let mut position = range_start;
            while position < range_end {
                let slot = chunk % 2;
                if chunk >= 2 {
                    // SAFETY: the event was recorded after the copies that last read this staging buffer.
                    check(unsafe { mmh3_cuda_event_synchronize(queue.events[slot]) })?;
                }
                let length = ((range_end - position) as usize).min(STAGING_BYTES);
                let read = file.read_at(position, &mut staging[slot].as_mut_slice()[..length])?;
                let chunk_end = position + read as u64;
                // Only the aligned tail past the end of the file may come back short.
                if chunk_end < data_end.min(position + length as u64) {
                    return Err(LoadError::Io(io::Error::new(io::ErrorKind::UnexpectedEof, "the checkpoint is shorter than its header says")));
                }
                while next_upload < self.uploads.len() && self.uploads[next_upload].file_offset + (self.uploads[next_upload].length as u64) <= position {
                    next_upload += 1;
                }
                for upload in &self.uploads[next_upload..] {
                    if upload.file_offset >= chunk_end {
                        break;
                    }
                    let start = upload.file_offset.max(position);
                    let end = (upload.file_offset + upload.length as u64).min(chunk_end);
                    if start >= end {
                        continue;
                    }
                    // SAFETY: the destination range lies inside the buffer queued for this upload, and the source
                    // inside the bytes just read into the staging buffer, which stays untouched until its event.
                    check(unsafe {
                        mmh3_cuda_copy_to_device_async(
                            upload.destination.cast::<u8>().add((start - upload.file_offset) as usize).cast(),
                            staging[slot].pointer.cast::<u8>().add((start - position) as usize).cast(),
                            (end - start) as usize,
                            queue.stream,
                        )
                    })?;
                }
                // SAFETY: both handles belong to the queue.
                check(unsafe { mmh3_cuda_event_record(queue.events[slot], queue.stream) })?;
                position += length as u64;
                chunk += 1;
            }
        }
        // SAFETY: the stream belongs to the queue.
        check(unsafe { mmh3_cuda_stream_synchronize(queue.stream) })?;
        Ok(self.uploads.iter().map(|upload| upload.length).sum())
    }
}
