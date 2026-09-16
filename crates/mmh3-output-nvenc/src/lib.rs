//! NVENC H.264 adapter for MP4 output. GPU-specific ownership and native APIs stay in this crate.
#![cfg(target_os = "linux")]
use mmh3_cuda::{DeviceBuffer, vae::CudaVideoFrames};
use mmh3_output::mp4::{H264Config, Sample, VideoEncoder, annex_b_sample};
use mmh3_output::{MediaSpec, Result};
use std::ffi::{CStr, c_char, c_void};
use std::ptr::NonNull;

#[repr(C)]
struct RawPacket {
    data: *const u8,
    size: u32,
    keyframe: u32,
    pts: u64,
}
unsafe extern "C" {
    fn mmh3_nvenc_open(
        input: *mut c_void,
        width: u32,
        height: u32,
        pitch: u32,
        fps: u32,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut c_void;
    fn mmh3_nvenc_headers(handle: *mut c_void, data: *mut *const u8, size: *mut u32);
    fn mmh3_nvenc_encode(
        handle: *mut c_void,
        pts: u64,
        packet: *mut RawPacket,
        error: *mut c_char,
        error_size: usize,
    ) -> i32;
    fn mmh3_nvenc_release(handle: *mut c_void);
    fn mmh3_nvenc_finish(handle: *mut c_void, error: *mut c_char, error_size: usize) -> i32;
    fn mmh3_nvenc_close(handle: *mut c_void);
}
fn failure(error: &[c_char]) -> Box<dyn std::error::Error> {
    // SAFETY: the buffer is zero-initialized. The C adapter writes at most size-1 bytes and
    // NUL-terminates them.
    format!(
        "NVENC: {}. Use --out FILE.webm or --ffmpeg to select another output.",
        unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy()
    )
    .into()
}

/// Owns a single registered device input. Encoding is synchronous with no B frames
/// or lookahead. The input is reused only after NVENC releases it. Works with discrete
/// VRAM as well as GB10. No host mapping or unified-memory assumption is made.
pub struct NvencEncoder {
    handle: NonNull<c_void>,
    input: DeviceBuffer,
    spec: MediaSpec,
    pitch: usize,
    config: H264Config,
    next_frame: usize,
    finished: bool,
}
impl NvencEncoder {
    pub fn new(spec: MediaSpec) -> Result<Self> {
        spec.validate()?;
        let width = u16::try_from(spec.width)?;
        let height = u16::try_from(spec.height)?;
        let fps = u32::try_from(spec.fps)?;
        fps.checked_mul(2).ok_or("frame rate overflow")?;
        let pitch = spec.width.div_ceil(256) * 256;
        let input = DeviceBuffer::new(pitch * spec.height * 3 / 2)?;
        let mut error = [0; 2048];
        // SAFETY: input owns a CUDA allocation with the required pitch and size. It
        // outlives the native encoder. Drop unregisters it before DeviceBuffer is freed.
        let handle = unsafe {
            mmh3_nvenc_open(
                input.pointer(),
                u32::from(width),
                u32::from(height),
                pitch as u32,
                fps,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(handle).ok_or_else(|| failure(&error))?;
        let mut encoder = Self {
            handle,
            input,
            spec,
            pitch,
            config: H264Config {
                sps: vec![],
                pps: vec![],
            },
            next_frame: 0,
            finished: false,
        };
        let mut data = std::ptr::null();
        let mut size = 0;
        // SAFETY: the native handle is live. The headers remain valid until the encoder is
        // destroyed.
        unsafe { mmh3_nvenc_headers(handle.as_ptr(), &mut data, &mut size) };
        if data.is_null() || size == 0 {
            return Err("NVENC returned no codec configuration".into());
        }
        encoder.config =
            H264Config::from_annex_b(unsafe { std::slice::from_raw_parts(data, size as usize) })?;
        Ok(encoder)
    }
}
impl VideoEncoder for NvencEncoder {
    type Frames = CudaVideoFrames;
    fn config(&self) -> &H264Config {
        &self.config
    }
    fn encode_frame(&mut self, frames: &CudaVideoFrames, index: usize) -> Result<Sample> {
        if self.finished
            || index != self.next_frame
            || (frames.width(), frames.height(), frames.frames())
                != (self.spec.width, self.spec.height, self.spec.frames)
        {
            return Err(
                "NVENC input dimensions or frame order do not match the prepared session".into(),
            );
        }
        frames.write_nv12_frame(index, &mut self.input, self.pitch)?;
        let mut raw = RawPacket {
            data: std::ptr::null(),
            size: 0,
            keyframe: 0,
            pts: 0,
        };
        let mut error = [0; 2048];
        // SAFETY: conversion completed on CUDA. The input remains registered and alive.
        if unsafe {
            mmh3_nvenc_encode(
                self.handle.as_ptr(),
                index as u64,
                &mut raw,
                error.as_mut_ptr(),
                error.len(),
            )
        } != 0
        {
            return Err(failure(&error));
        }
        let result = if raw.data.is_null() || raw.size == 0 {
            Err("NVENC returned an empty frame".into())
        } else {
            // SAFETY: NVENC keeps this bitstream locked until release below. Only compressed bytes
            // are copied to CPU.
            annex_b_sample(unsafe { std::slice::from_raw_parts(raw.data, raw.size as usize) })
        };
        unsafe { mmh3_nvenc_release(self.handle.as_ptr()) };
        let data = result?;
        if raw.pts != index as u64 {
            return Err("unexpected NVENC presentation timestamp".into());
        }
        self.next_frame += 1;
        Ok(Sample {
            data,
            dts: raw.pts,
            pts: i64::try_from(raw.pts)?,
            duration: 1,
            keyframe: raw.keyframe != 0,
        })
    }
    fn finish(&mut self) -> Result<Vec<Sample>> {
        if self.finished || self.next_frame != self.spec.frames {
            return Err("incomplete or already finished NVENC stream".into());
        }
        let mut error = [0; 2048];
        // SAFETY: all frames have been synchronously retrieved, so EOS cannot leave delayed
        // packets.
        if unsafe { mmh3_nvenc_finish(self.handle.as_ptr(), error.as_mut_ptr(), error.len()) } != 0
        {
            return Err(failure(&error));
        }
        self.finished = true;
        Ok(Vec::new())
    }
}
impl Drop for NvencEncoder {
    fn drop(&mut self) {
        // SAFETY: the handle is unique and live. The native destructor unregisters the input before
        // Rust drops its allocation.
        unsafe { mmh3_nvenc_close(self.handle.as_ptr()) };
    }
}
