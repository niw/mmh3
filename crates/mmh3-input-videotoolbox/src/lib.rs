//! H.264 decoding with VideoToolbox, for the reference clips of reference to video generation on
//! macOS.
//!
//! The decompression session runs behind `native/runtime.swift`, which hands every frame back as
//! NV12 in host memory, as the NVDEC adapter does on Linux. The session releases frames in the
//! order they are decoded, so each carries the display position its sample has in the track, and
//! the frames are put in display order here. The conversion to RGB is `mmh3-input`'s, with the
//! colour matrix and range the stream declares, or the guess players make when it declares none,
//! so a clip comes out of either decoder the same.
#![cfg(target_os = "macos")]

use mmh3_core::tensor::Tensor;
pub use mmh3_input::colour::{Colour, nv12_to_rgb};
use mmh3_input::mp4::Mp4File;
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io;

unsafe extern "C" {
    fn mmh3_vtdec_create(parameter_sets: *const u8, length: usize) -> *mut c_void;
    fn mmh3_vtdec_size(
        handle: *mut c_void,
        width: *mut c_int,
        height: *mut c_int,
        matrix: *mut c_int,
        full_range: *mut c_int,
    );
    fn mmh3_vtdec_feed(
        handle: *mut c_void,
        data: *const u8,
        length: usize,
        position: usize,
    ) -> c_int;
    fn mmh3_vtdec_flush(handle: *mut c_void) -> c_int;
    fn mmh3_vtdec_ready(handle: *mut c_void) -> usize;
    fn mmh3_vtdec_take(
        handle: *mut c_void,
        nv12: *mut u8,
        capacity: usize,
        position: *mut usize,
    ) -> c_int;
    fn mmh3_vtdec_error() -> *const c_char;
    fn mmh3_vtdec_destroy(handle: *mut c_void);
}

/// The bridge's last complaint on this thread, with what failed.
fn failure(what: &str) -> io::Error {
    // SAFETY: the bridge returns a thread-local NUL-terminated error string. Copy it before
    // another bridge call can replace it.
    let detail = unsafe {
        let pointer = mmh3_vtdec_error();
        (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
    };
    match detail {
        Some(detail) => io::Error::other(format!("{what}: {detail}")),
        None => io::Error::other(what.to_owned()),
    }
}

/// NAL units in Annex B form, each after a start code, as VideoToolbox takes them: each after its
/// length in four bytes.
fn length_prefixed(annex_b: &[u8]) -> io::Result<Vec<u8>> {
    let mut starts = Vec::new();
    let mut offset = 0;
    while offset + 3 <= annex_b.len() {
        if annex_b[offset..offset + 3] == [0, 0, 1] {
            offset += 3;
            starts.push(offset);
        } else {
            offset += 1;
        }
    }
    let mut units = Vec::with_capacity(annex_b.len() + 4);
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).map_or(annex_b.len(), |next| next - 3);
        // A unit never ends in a zero byte, so trailing zeros belong to the next start code.
        let unit = &annex_b[start..end];
        let length = unit
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |last| last + 1);
        if length == 0 {
            continue;
        }
        let length_bytes = u32::try_from(length)
            .map_err(|_| io::Error::other("a NAL unit is too long"))?
            .to_be_bytes();
        units.extend_from_slice(&length_bytes);
        units.extend_from_slice(&unit[..length]);
    }
    if units.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the data holds no NAL unit",
        ));
    }
    Ok(units)
}

/// An H.264 decoder holding a VideoToolbox decompression session.
pub struct Decoder {
    handle: *mut c_void,
    /// The size the frames come out at, which the sequence set crops them to.
    size: (usize, usize),
    colour: Colour,
    /// Decoded frames waiting for those shown before them, by display position.
    pending: BTreeMap<usize, Vec<u8>>,
    /// The display position of the next frame to hand out.
    next: usize,
    /// Whether the stream has ended, after which no frame is waited for.
    ended: bool,
}

impl Decoder {
    /// Opens a session for a track's Annex B parameter sets, which carry the frame size and colour.
    pub fn open(parameter_sets: &[u8]) -> io::Result<Self> {
        let parameter_sets = length_prefixed(parameter_sets)?;
        // SAFETY: the parameter sets are a valid slice, which the bridge copies what it keeps of.
        let handle = unsafe { mmh3_vtdec_create(parameter_sets.as_ptr(), parameter_sets.len()) };
        if handle.is_null() {
            return Err(failure("VideoToolbox did not open the video track"));
        }
        let (mut width, mut height, mut matrix, mut full_range) = (0, 0, 0, 0);
        // SAFETY: the handle is live and the out pointers are local.
        unsafe {
            mmh3_vtdec_size(
                handle,
                &mut width,
                &mut height,
                &mut matrix,
                &mut full_range,
            )
        };
        let (width, height) = (width as usize, height as usize);
        Ok(Decoder {
            handle,
            size: (width, height),
            colour: Colour::of(matrix, full_range != 0, height),
            pending: BTreeMap::new(),
            next: 0,
            ended: false,
        })
    }

    /// The size of the decoded frames.
    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    /// The colour the stream declares.
    pub fn colour(&self) -> Colour {
        self.colour
    }

    /// Decodes one Annex B access unit, whose frame is shown at `position` among the track's
    /// frames, counted from zero.
    pub fn feed(&mut self, access_unit: &[u8], position: usize) -> io::Result<()> {
        let unit = length_prefixed(access_unit)?;
        // SAFETY: the unit is a valid slice, which the bridge copies, and the handle is live.
        if unsafe { mmh3_vtdec_feed(self.handle, unit.as_ptr(), unit.len(), position) } != 0 {
            return Err(failure("VideoToolbox failed on an access unit"));
        }
        Ok(())
    }

    /// Ends the stream so the session releases the frames it still holds, and the frames shown
    /// after a missing one come out without it.
    pub fn flush(&mut self) -> io::Result<()> {
        // SAFETY: the handle is live.
        if unsafe { mmh3_vtdec_flush(self.handle) } != 0 {
            return Err(failure("VideoToolbox failed at the end of the stream"));
        }
        self.ended = true;
        Ok(())
    }

    /// Takes the next decoded frame in display order as NV12, or `None` when it is not ready.
    pub fn take(&mut self) -> io::Result<Option<Vec<u8>>> {
        let (width, height) = self.size;
        // SAFETY: the handle is live.
        while unsafe { mmh3_vtdec_ready(self.handle) } > 0 {
            let mut frame = vec![0u8; width * height * 3 / 2];
            let mut position = 0;
            // SAFETY: the buffer holds a whole NV12 frame of the decoder's size and the out
            // pointer is local.
            let status = unsafe {
                mmh3_vtdec_take(self.handle, frame.as_mut_ptr(), frame.len(), &mut position)
            };
            if status != 0 {
                return Err(io::Error::other("VideoToolbox did not hand over a frame"));
            }
            self.pending.insert(position, frame);
        }
        let Some(entry) = self.pending.first_entry() else {
            return Ok(None);
        };
        if *entry.key() != self.next && !self.ended {
            return Ok(None);
        }
        self.next = *entry.key() + 1;
        Ok(Some(entry.remove()))
    }

    /// The frame's pixels in [0, 1] as `[height, width, 3]`, in the stream's colour.
    pub fn to_rgb(&self, nv12: &[u8]) -> Tensor {
        let (width, height) = self.size;
        nv12_to_rgb(nv12, width, height, self.colour)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: this is the sole owner. Invalidating the session finishes its callbacks before
        // the bridge releases it.
        unsafe { mmh3_vtdec_destroy(self.handle) };
    }
}

// SAFETY: the handle is only used behind `&mut self`, and the bridge locks what its callbacks
// share with the calls.
unsafe impl Send for Decoder {}

/// Decodes the first `frames` frames of a file's video track as `[frames, height, width, 3]` in
/// [0, 1]. Fewer frames come back when the track is shorter.
pub fn decode_frames(file: &mut Mp4File, frames: usize) -> io::Result<Tensor> {
    let parameter_sets = file.track().parameter_sets.clone();
    let positions = file.track().display_positions();
    let mut decoder = Decoder::open(&parameter_sets)?;
    let mut pixels: Vec<f32> = Vec::new();
    let mut decoded = 0;
    for (index, &position) in positions.iter().enumerate() {
        if decoded >= frames {
            break;
        }
        let unit = file.access_unit(index)?;
        decoder.feed(&unit, position)?;
        decoded += collect(&mut decoder, &mut pixels, frames - decoded)?;
    }
    if decoded < frames {
        decoder.flush()?;
        decoded += collect(&mut decoder, &mut pixels, frames - decoded)?;
    }
    if decoded == 0 {
        return Err(io::Error::other("the video track decoded no frame"));
    }
    let (width, height) = decoder.size();
    Ok(Tensor::new(vec![decoded, height, width, 3], pixels))
}

/// Moves up to `wanted` ready frames into `pixels`, returning how many it took.
fn collect(decoder: &mut Decoder, pixels: &mut Vec<f32>, wanted: usize) -> io::Result<usize> {
    let mut taken = 0;
    while taken < wanted {
        let Some(nv12) = decoder.take()? else { break };
        pixels.extend(decoder.to_rgb(&nv12).data);
        taken += 1;
    }
    Ok(taken)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_every_nal_unit_with_its_length() {
        // Start codes of three and four bytes, the zero before a four-byte one left out.
        let annex_b = [
            0, 0, 0, 1, 0x67, 0xaa, 0, 0, 1, 0x68, 0xbb, 0xcc, 0, 0, 0, 1, 0x65,
        ];
        assert_eq!(
            length_prefixed(&annex_b).unwrap(),
            vec![
                0, 0, 0, 2, 0x67, 0xaa, 0, 0, 0, 3, 0x68, 0xbb, 0xcc, 0, 0, 0, 1, 0x65
            ]
        );
        assert!(length_prefixed(&[0, 0, 0, 1]).is_err());
    }
}
