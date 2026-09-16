//! VideoToolbox H.264 adapter for the shared MP4 writer and CPU AAC encoder.
#![cfg(target_os = "macos")]

use mmh3_core::media::Yuv420;
use mmh3_output::mp4::{H264Config, Sample, VideoEncoder};
use mmh3_output::{MediaSpec, Result};
use std::{
    ffi::{CStr, c_char, c_void},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

unsafe extern "C" {
    #[cfg(feature = "metal")]
    fn mmh3_vt_enable_metal(handle: *mut c_void, source: *const c_char) -> i32;
    #[cfg(feature = "metal")]
    fn mmh3_vt_encode_metal(
        handle: *mut c_void,
        buffer: *mut c_void,
        frames: usize,
        source_index: i64,
        index: i64,
    ) -> i32;
    fn mmh3_vt_error() -> *const c_char;
    fn mmh3_vt_create(width: i32, height: i32, fps: i32) -> *mut c_void;
    fn mmh3_vt_destroy(handle: *mut c_void);
    fn mmh3_vt_encode(handle: *mut c_void, frame: *const u8, index: i64) -> i32;
    fn mmh3_vt_bytes(handle: *mut c_void, kind: i32, count: *mut usize) -> *const u8;
    fn mmh3_vt_keyframe(handle: *mut c_void) -> bool;
}

fn failure() -> Box<dyn std::error::Error> {
    // SAFETY: the bridge returns a thread-local NUL-terminated error string. Copy it before
    // another bridge call can replace it.
    let detail = unsafe {
        let pointer = mmh3_vt_error();
        if pointer.is_null() {
            "unknown encoder error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };

    format!("VideoToolbox: {detail}").into()
}

/// A synchronous H.264 hardware encoder. Input is planar BT.709 limited-range YUV420.
/// The Swift bridge copies each frame into an NV12 pixel buffer and waits for the callback.
/// The handle and its borrowed packet storage stay on the creating thread.
pub struct VideoToolboxEncoder {
    handle: NonNull<c_void>,
    spec: MediaSpec,
    config: H264Config,
    next_frame: usize,
    finished: bool,
    _thread: PhantomData<Rc<()>>,
}

impl VideoToolboxEncoder {
    pub fn new(spec: MediaSpec) -> Result<Self> {
        spec.validate()?;
        let width = i32::from(u16::try_from(spec.width)?);
        let height = i32::from(u16::try_from(spec.height)?);
        let fps = i32::try_from(spec.fps)?;
        i64::try_from(spec.frames)?;
        // SAFETY: dimensions and frame rate are positive and fit the native API. The returned
        // retained object has one owner and is invalidated and released by Drop.
        let handle =
            NonNull::new(unsafe { mmh3_vt_create(width, height, fps) }).ok_or_else(failure)?;
        Ok(Self {
            handle,
            spec,
            config: H264Config {
                sps: Vec::new(),
                pps: Vec::new(),
            },

            next_frame: 0,
            finished: false,
            _thread: PhantomData,
        })
    }

    fn packet(&mut self, index: usize) -> Result<Sample> {
        let config = H264Config {
            sps: self.bytes(1)?,
            pps: self.bytes(2)?,
        };

        if index != 0 && (self.config.sps != config.sps || self.config.pps != config.pps) {
            self.finished = true;
            return Err("H.264 configuration changed during encoding".into());
        }

        self.config = config;
        let packet = Sample {
            data: self.bytes(0)?,
            dts: index as u64,
            pts: index as i64,
            duration: 1,
            // SAFETY: the synchronous encode call populated the live encoder's result.
            keyframe: unsafe { mmh3_vt_keyframe(self.handle.as_ptr()) },
        };

        self.next_frame += 1;
        Ok(packet)
    }

    fn bytes(&self, kind: i32) -> Result<Vec<u8>> {
        let mut count = 0;
        // SAFETY: the live encoder owns immutable NSData storage until the next encode call.
        // This method copies it synchronously. There is no concurrent access to the handle.
        let pointer = unsafe { mmh3_vt_bytes(self.handle.as_ptr(), kind, &mut count) };

        if pointer.is_null() || count == 0 || count > isize::MAX as usize {
            return Err("VideoToolbox returned empty or oversized data".into());
        }

        // SAFETY: the bridge reports the length of the allocation backing pointer.
        Ok(unsafe { std::slice::from_raw_parts(pointer, count) }.to_vec())
    }
}

impl VideoEncoder for VideoToolboxEncoder {
    type Frames = Yuv420;
    fn config(&self) -> &H264Config {
        &self.config
    }

    fn encode_frame(&mut self, frames: &Yuv420, index: usize) -> Result<Sample> {
        if self.finished || index != self.next_frame || index >= self.spec.frames {
            return Err("video frames must be submitted once in order before finishing".into());
        }

        if (frames.width, frames.height, frames.frames)
            != (self.spec.width, self.spec.height, self.spec.frames)
            || frames.data.len()
                != self.spec.frames * Yuv420::frame_bytes(self.spec.height, self.spec.width)
        {
            return Err("video does not match the encoder specification".into());
        }

        let size = Yuv420::frame_bytes(frames.height, frames.width);
        let source = &frames.data[index * size..(index + 1) * size];
        // SAFETY: source contains a complete planar frame and stays alive until the synchronous
        // bridge has copied and encoded it. The callback is complete before the bridge returns.

        if unsafe { mmh3_vt_encode(self.handle.as_ptr(), source.as_ptr(), index as i64) } != 0 {
            self.finished = true;
            return Err(failure());
        }

        self.packet(index)
    }

    fn finish(&mut self) -> Result<Vec<Sample>> {
        if self.finished || self.next_frame != self.spec.frames {
            return Err("incomplete or already finished VideoToolbox encoder".into());
        }

        self.finished = true;
        // Every encode call drains its frame and frame reordering is disabled.
        Ok(Vec::new())
    }
}

impl Drop for VideoToolboxEncoder {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner. Invalidation finishes callbacks before releasing it.
        unsafe { mmh3_vt_destroy(self.handle.as_ptr()) }
    }
}

/// Encodes GPU-resident RGB frames through shared NV12 CVPixelBuffers without CPU pixel copies.
#[cfg(feature = "metal")]
pub struct MetalVideoToolboxEncoder(VideoToolboxEncoder);
#[cfg(feature = "metal")]
impl MetalVideoToolboxEncoder {
    pub fn new(spec: MediaSpec) -> Result<Self> {
        let encoder = VideoToolboxEncoder::new(spec)?;
        let source = std::ffi::CString::new(include_str!("../kernels/nv12.metal")).unwrap();
        // SAFETY: the bridge copies the shader source and retains the conversion resources in
        // the live encoder. This happens before model loading.

        if unsafe { mmh3_vt_enable_metal(encoder.handle.as_ptr(), source.as_ptr()) } != 0 {
            return Err(failure());
        }

        Ok(Self(encoder))
    }
}

#[cfg(feature = "metal")]
impl VideoEncoder for MetalVideoToolboxEncoder {
    type Frames = mmh3_metal::vae::MetalVideoFrames;
    fn config(&self) -> &H264Config {
        self.0.config()
    }

    fn encode_frame(&mut self, frames: &Self::Frames, index: usize) -> Result<Sample> {
        let encoder = &mut self.0;
        if encoder.finished || index != encoder.next_frame || index >= encoder.spec.frames {
            return Err("video frames must be submitted once in order before finishing".into());
        }

        let [_, count, height, width] = frames.shape();
        let range = frames.frame_range();
        if height != encoder.spec.height
            || width != encoder.spec.width
            || !range.contains(&index)
            || range.end > encoder.spec.frames
        {
            return Err("Metal video does not match the encoder specification".into());
        }

        // SAFETY: frames owns a completed RGB Metal allocation with validated dimensions. The
        // bridge only reads it, completes the conversion, then drains the encoder before returning.
        if unsafe {
            mmh3_vt_encode_metal(
                encoder.handle.as_ptr(),
                frames.as_metal_buffer()?,
                count,
                (index - range.start) as i64,
                index as i64,
            )
        } != 0
        {
            encoder.finished = true;
            return Err(failure());
        }

        encoder.packet(index)
    }

    fn finish(&mut self) -> Result<Vec<Sample>> {
        self.0.finish()
    }
}
