//! File reads that bypass the page cache, for streaming large checkpoints once.
//!
//! A checkpoint read through the page cache stays in memory next to its copy on the GPU, which on
//! unified-memory systems such as GB10 holds the weights twice in the same RAM. Direct reads leave
//! nothing behind.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Offsets, lengths and buffer addresses of direct reads are multiples of this.
pub const DIRECT_ALIGNMENT: usize = 4096;

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const O_DIRECT: i32 = 0o200000;
#[cfg(all(target_os = "linux", not(target_arch = "aarch64")))]
const O_DIRECT: i32 = 0o40000;
const POSIX_FADV_DONTNEED: i32 = 4;

unsafe extern "C" {
    fn posix_fadvise(descriptor: i32, offset: i64, length: i64, advice: i32) -> i32;
}

pub struct DirectFile {
    file: File,
    direct: bool,
}

impl DirectFile {
    /// Opens `path` for direct reads, or, where the file system refuses them, for buffered reads
    /// that drop what they read from the page cache.
    pub fn open(path: &Path) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::OpenOptionsExt;
            if let Ok(file) = OpenOptions::new()
                .read(true)
                .custom_flags(O_DIRECT)
                .open(path)
            {
                return Ok(DirectFile { file, direct: true });
            }
        }
        Ok(DirectFile {
            file: OpenOptions::new().read(true).open(path)?,
            direct: false,
        })
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// Fills `buffer` from `offset` and returns the bytes read, fewer only at the end of the file.
    /// For direct reads `offset`, the buffer address and its length must be multiples of
    /// [`DIRECT_ALIGNMENT`].
    pub fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        let mut filled = 0;
        while filled < buffer.len() {
            let read = self
                .file
                .read_at(&mut buffer[filled..], offset + filled as u64)?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        if !self.direct {
            // SAFETY: plain advice on an open descriptor.
            unsafe {
                posix_fadvise(
                    self.file.as_raw_fd(),
                    offset as i64,
                    filled as i64,
                    POSIX_FADV_DONTNEED,
                );
            }
        }
        Ok(filled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_aligned_blocks_to_the_end_of_the_file() {
        let path = std::env::temp_dir().join(format!("mmh3-direct-{}", std::process::id()));
        let contents: Vec<u8> = (0..3 * DIRECT_ALIGNMENT + 100)
            .map(|index| (index % 251) as u8)
            .collect();
        std::fs::write(&path, &contents).unwrap();
        let file = DirectFile::open(&path).unwrap();
        // An aligned buffer: over-allocate and start at the first aligned address.
        let mut storage = vec![0u8; 5 * DIRECT_ALIGNMENT];
        let start = storage.as_ptr().align_offset(DIRECT_ALIGNMENT);
        let buffer = &mut storage[start..start + 4 * DIRECT_ALIGNMENT];
        let read = file.read_at(DIRECT_ALIGNMENT as u64, buffer).unwrap();
        assert_eq!(read, 2 * DIRECT_ALIGNMENT + 100);
        assert_eq!(&buffer[..read], &contents[DIRECT_ALIGNMENT..]);
        std::fs::remove_file(&path).unwrap();
    }
}
