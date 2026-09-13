use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

unsafe extern "C" {
    fn mmap(address: *mut c_void, length: usize, protection: i32, flags: i32, descriptor: i32, offset: i64) -> *mut c_void;
    fn munmap(address: *mut c_void, length: usize) -> i32;
}

/// A read-only memory mapping of a whole file.
pub struct MappedFile {
    address: *mut c_void,
    length: usize,
}

// SAFETY: the mapping is read-only and is never mutated after creation.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let length = usize::try_from(file.metadata()?.len()).map_err(|_| io::Error::other("file is too large to map"))?;
        if length == 0 {
            return Ok(Self { address: ptr::null_mut(), length: 0 });
        }
        // SAFETY: the arguments describe a private read-only mapping of an open descriptor.
        let address = unsafe { mmap(ptr::null_mut(), length, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
        if address as usize == usize::MAX {
            return Err(io::Error::last_os_error());
        }
        Ok(Self { address, length })
    }

    pub fn as_bytes(&self) -> &[u8] {
        if self.length == 0 {
            return &[];
        }
        // SAFETY: the mapping stays valid and unmodified for the lifetime of self.
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.length) }
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        if self.length != 0 {
            // SAFETY: address and length come from a successful mmap call.
            unsafe {
                munmap(self.address, self.length);
            }
        }
    }
}
