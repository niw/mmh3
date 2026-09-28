//! H.264 decoding with NVDEC, for the reference clips of reference to video generation.
//!
//! The driver's parser and decoder run behind `native/nvdec.cpp`, which hands every frame back as
//! NV12 in host memory. The conversion to RGB is `mmh3-input`'s, with the colour matrix and range
//! the stream declares, or the guess players make when it declares none.
#![cfg(target_os = "linux")]

use mmh3_core::tensor::Tensor;
pub use mmh3_input::colour::{Colour, nv12_to_rgb};
use mmh3_input::mp4::Mp4File;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io;

unsafe extern "C" {
    fn mmh3_nvdec_create(
        parameter_sets: *const u8,
        length: usize,
        handle: *mut *mut c_void,
    ) -> c_int;
    fn mmh3_nvdec_size(
        handle: *mut c_void,
        width: *mut c_int,
        height: *mut c_int,
        matrix: *mut c_int,
        full_range: *mut c_int,
    ) -> c_int;
    fn mmh3_nvdec_feed(handle: *mut c_void, data: *const u8, length: usize) -> c_int;
    fn mmh3_nvdec_flush(handle: *mut c_void) -> c_int;
    fn mmh3_nvdec_ready(handle: *mut c_void) -> usize;
    fn mmh3_nvdec_take(handle: *mut c_void, nv12: *mut u8, capacity: usize) -> c_int;
    fn mmh3_nvdec_error(handle: *mut c_void) -> *const c_char;
    fn mmh3_nvdec_destroy(handle: *mut c_void);
}

/// An H.264 decoder holding the driver's parser and decoder.
pub struct Decoder {
    handle: *mut c_void,
    /// The size the frames come out at, known once the first access unit has been parsed.
    size: Option<(usize, usize)>,
    colour: Option<Colour>,
}

impl Decoder {
    /// Opens a parser for a track's Annex B parameter sets. The frame size follows the first
    /// access unit, which is when the driver reports the format.
    pub fn open(parameter_sets: &[u8]) -> io::Result<Self> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: the parameter sets are a valid slice and the out pointer is local.
        let status = unsafe {
            mmh3_nvdec_create(parameter_sets.as_ptr(), parameter_sets.len(), &mut handle)
        };
        if status != 0 || handle.is_null() {
            return Err(io::Error::other("NVDEC did not open the video track"));
        }
        Ok(Decoder {
            handle,
            size: None,
            colour: None,
        })
    }

    /// The size of the decoded frames, once the driver has reported it.
    pub fn size(&self) -> Option<(usize, usize)> {
        self.size
    }

    /// The colour the stream declares, once the driver has reported its format.
    pub fn colour(&self) -> Option<Colour> {
        self.colour
    }

    fn refresh_size(&mut self) {
        if self.size.is_some() {
            return;
        }
        let (mut width, mut height, mut matrix, mut full_range) = (0, 0, 0, 0);
        // SAFETY: the handle is live and the out pointers are local.
        let known = unsafe {
            mmh3_nvdec_size(
                self.handle,
                &mut width,
                &mut height,
                &mut matrix,
                &mut full_range,
            )
        };
        if known == 0 && width > 0 && height > 0 {
            self.size = Some((width as usize, height as usize));
            self.colour = Some(Colour::of(matrix, full_range != 0, height as usize));
        }
    }

    /// The driver's last complaint, if it made one.
    fn failure(&self) -> Option<String> {
        // SAFETY: the handle is live and the message, when there is one, is a C string that
        // outlives this call.
        unsafe {
            let message = mmh3_nvdec_error(self.handle);
            message
                .as_ref()
                .map(|_| CStr::from_ptr(message).to_string_lossy().into_owned())
        }
    }

    fn error(&self, what: &str) -> io::Error {
        match self.failure() {
            Some(message) => io::Error::other(format!("{what}: {message}")),
            None => io::Error::other(what.to_owned()),
        }
    }

    /// Parses one access unit, which may complete frames.
    pub fn feed(&mut self, access_unit: &[u8]) -> io::Result<()> {
        // SAFETY: the access unit is a valid slice and the handle is live.
        let status =
            unsafe { mmh3_nvdec_feed(self.handle, access_unit.as_ptr(), access_unit.len()) };
        if status != 0 {
            return Err(self.error("NVDEC failed on an access unit"));
        }
        self.refresh_size();
        Ok(())
    }

    /// Ends the stream so the parser sends the frames it still holds to display.
    pub fn flush(&mut self) -> io::Result<()> {
        // SAFETY: the handle is live.
        if unsafe { mmh3_nvdec_flush(self.handle) } != 0 {
            return Err(self.error("NVDEC failed at the end of the stream"));
        }
        self.refresh_size();
        Ok(())
    }

    /// Takes the oldest decoded frame as NV12, or `None` when none is ready.
    pub fn take(&mut self) -> io::Result<Option<Vec<u8>>> {
        // SAFETY: the handle is live.
        if unsafe { mmh3_nvdec_ready(self.handle) } == 0 {
            return Ok(None);
        }
        let (width, height) = self
            .size
            .ok_or_else(|| io::Error::other("NVDEC has a frame of an unknown size"))?;
        let mut frame = vec![0u8; width * height * 3 / 2];
        // SAFETY: the buffer holds a whole NV12 frame of the decoder's size.
        if unsafe { mmh3_nvdec_take(self.handle, frame.as_mut_ptr(), frame.len()) } != 0 {
            return Err(self.error("NVDEC did not hand over a frame"));
        }
        Ok(Some(frame))
    }

    /// The frame's pixels in [0, 1] as `[height, width, 3]`, in the stream's colour.
    pub fn to_rgb(&self, nv12: &[u8]) -> io::Result<Tensor> {
        let (width, height) = self
            .size
            .ok_or_else(|| io::Error::other("NVDEC has not reported a frame size"))?;
        let colour = self
            .colour
            .ok_or_else(|| io::Error::other("NVDEC has not reported a frame's colour"))?;
        Ok(nv12_to_rgb(nv12, width, height, colour))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the handle was created by the adapter and is dropped once.
        unsafe { mmh3_nvdec_destroy(self.handle) };
    }
}

// SAFETY: the handle is only used behind `&mut self`, and the adapter keeps no thread-local state
// beyond the CUDA context it adopts.
unsafe impl Send for Decoder {}

/// Decodes the first `frames` frames of a file's video track as `[frames, height, width, 3]` in
/// [0, 1]. Fewer frames come back when the track is shorter.
pub fn decode_frames(file: &mut Mp4File, frames: usize) -> io::Result<Tensor> {
    let parameter_sets = file.track().parameter_sets.clone();
    let samples = file.track().samples.len();
    let mut decoder = Decoder::open(&parameter_sets)?;
    let mut pixels: Vec<f32> = Vec::new();
    let mut decoded = 0;
    for index in 0..samples {
        if decoded >= frames {
            break;
        }
        let unit = file.access_unit(index)?;
        decoder.feed(&unit)?;
        decoded += collect(&mut decoder, &mut pixels, frames - decoded)?;
    }
    if decoded < frames {
        decoder.flush()?;
        decoded += collect(&mut decoder, &mut pixels, frames - decoded)?;
    }
    let (width, height) = decoder
        .size()
        .ok_or_else(|| io::Error::other("the video track decoded no frame"))?;
    if decoded == 0 {
        return Err(io::Error::other("the video track decoded no frame"));
    }
    Ok(Tensor::new(vec![decoded, height, width, 3], pixels))
}

/// Moves up to `wanted` ready frames into `pixels`, returning how many it took.
fn collect(decoder: &mut Decoder, pixels: &mut Vec<f32>, wanted: usize) -> io::Result<usize> {
    let mut taken = 0;
    while taken < wanted {
        let Some(nv12) = decoder.take()? else { break };
        pixels.extend(decoder.to_rgb(&nv12)?.data);
        taken += 1;
    }
    Ok(taken)
}
