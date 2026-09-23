//! A RoCE device with memory registered on it, reliable connections to other machines, and reads
//! of their memory.
//!
//! The bulk of a distributed decode or a sharded step is too large for a socket: over the link
//! between two DGX Sparks, `ib_write_bw` reaches 13.3 GB/s where TCP reaches 1.5. Control stays on
//! the socket, which every machine has, and only the payloads come this way.
//!
//! The registrations belong to the device rather than to one connection, so a rank that shares a
//! step with several peers registers its regions once and offers them all one remote key.
//!
//! The verbs calls live in `src/rdma.c`, since their structures belong to the installed headers.

#![cfg(target_os = "linux")]

use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;

/// What one side sends the other so both can reach the same connection. Fixed layout: it crosses
/// the wire as its bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Address {
    pub queue_pair: u32,
    pub sequence: u32,
    pub local_id: u16,
    pub global_id: [u8; 16],
}

impl Address {
    pub const BYTES: usize = 26;

    pub fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0u8; Self::BYTES];
        bytes[0..4].copy_from_slice(&self.queue_pair.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[8..10].copy_from_slice(&self.local_id.to_le_bytes());
        bytes[10..26].copy_from_slice(&self.global_id);
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::BYTES {
            return None;
        }
        Some(Address {
            queue_pair: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            sequence: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            local_id: u16::from_le_bytes(bytes[8..10].try_into().unwrap()),
            global_id: bytes[10..26].try_into().unwrap(),
        })
    }
}

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for Error {}

/// One connection's share of a payload, as `rdma.c` lays it out.
#[repr(C)]
struct Part {
    link: *mut c_void,
    region: *mut c_void,
    local: *mut u8,
    bytes: usize,
    remote_address: u64,
    remote_key: u32,
}

/// How many paths a payload may be split over, and how long a device name may be, as `rdma.c`
/// defines them.
const PARTS: usize = 4;
const NAME_BYTES: usize = 64;

#[cfg(not(mmh3_rdma_without_verbs))]
unsafe extern "C" {
    fn mmh3_rdma_device_names(names: *mut c_char, capacity: c_int) -> c_int;
    fn mmh3_rdma_open(device: *const c_char, global_id_index: c_int) -> *mut c_void;
    fn mmh3_rdma_close(rdma: *mut c_void);
    fn mmh3_rdma_link_open(rdma: *mut c_void) -> *mut c_void;
    fn mmh3_rdma_link_close(link: *mut c_void);
    fn mmh3_rdma_link_address(link: *mut c_void, address: *mut Address) -> c_int;
    fn mmh3_rdma_link_connect(link: *mut c_void, peer: *const Address) -> c_int;
    fn mmh3_rdma_register(
        rdma: *mut c_void,
        buffer: *mut c_void,
        bytes: usize,
        remote_key: *mut u32,
    ) -> *mut c_void;
    fn mmh3_rdma_unregister(region: *mut c_void);
    fn mmh3_rdma_read_all(parts: *const Part, count: c_int, milliseconds: c_int) -> c_int;
    fn mmh3_rdma_write_all(parts: *const Part, count: c_int, milliseconds: c_int) -> c_int;
}

/// Stand-ins for `rdma.c` when it was built without the libibverbs headers. No device is ever
/// found, so no handle exists for the rest to be called with.
#[cfg(mmh3_rdma_without_verbs)]
mod without_verbs {
    use super::{Address, Part, c_char, c_int, c_void};

    pub unsafe fn mmh3_rdma_device_names(_names: *mut c_char, _capacity: c_int) -> c_int {
        0
    }
    pub unsafe fn mmh3_rdma_open(_device: *const c_char, _global_id_index: c_int) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn mmh3_rdma_close(_rdma: *mut c_void) {}
    pub unsafe fn mmh3_rdma_link_open(_rdma: *mut c_void) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn mmh3_rdma_link_close(_link: *mut c_void) {}
    pub unsafe fn mmh3_rdma_link_address(_link: *mut c_void, _address: *mut Address) -> c_int {
        -1
    }
    pub unsafe fn mmh3_rdma_link_connect(_link: *mut c_void, _peer: *const Address) -> c_int {
        -1
    }
    pub unsafe fn mmh3_rdma_register(
        _rdma: *mut c_void,
        _buffer: *mut c_void,
        _bytes: usize,
        _remote_key: *mut u32,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn mmh3_rdma_unregister(_region: *mut c_void) {}
    pub unsafe fn mmh3_rdma_read_all(_parts: *const Part, _count: c_int, _ms: c_int) -> c_int {
        -1
    }
    pub unsafe fn mmh3_rdma_write_all(_parts: *const Part, _count: c_int, _ms: c_int) -> c_int {
        -1
    }
}
#[cfg(mmh3_rdma_without_verbs)]
use without_verbs::*;

/// The RoCEv2 entry of a port's table, which is what these machines use.
pub const DEFAULT_GLOBAL_ID: i32 = 3;

/// The RoCE paths of this machine and their protection domains, which the memory regions belong
/// to. A Spark's one NIC hangs off two PCIe Gen5 x4 links and answers to both for the one cable,
/// and a payload split over the two carries 22.2 GB/s against the 13.0 of either.
pub struct Device {
    handles: Vec<*mut c_void>,
}

// The handle moves and is shared between threads: one device serves every connection of a process,
// and libibverbs allows concurrent calls on a context, a protection domain and its registrations.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Opens `device`, or every device with an active port when it is empty.
    pub fn open(device: &str, global_id_index: i32) -> Result<Self, Error> {
        let names: Vec<String> = if device.is_empty() {
            let mut bytes = vec![0u8; PARTS * NAME_BYTES];
            // SAFETY: the buffer holds `PARTS` names of `NAME_BYTES` each.
            let found =
                unsafe { mmh3_rdma_device_names(bytes.as_mut_ptr().cast(), PARTS as c_int) };
            (0..found.max(0) as usize)
                .map(|index| {
                    let start = index * NAME_BYTES;
                    let end = bytes[start..start + NAME_BYTES]
                        .iter()
                        .position(|byte| *byte == 0)
                        .unwrap_or(NAME_BYTES);
                    String::from_utf8_lossy(&bytes[start..start + end]).into_owned()
                })
                .collect()
        } else {
            vec![device.to_owned()]
        };
        // Both ends pair their paths by the order of this list, so it has to be one both agree on.
        let mut names = names;
        names.sort();
        let mut handles = Vec::with_capacity(names.len());
        for name in &names {
            let name = CString::new(name.as_str())
                .map_err(|_| Error("a device name has a nul".to_owned()))?;
            // SAFETY: the name outlives the call, which copies what it needs.
            let handle = unsafe { mmh3_rdma_open(name.as_ptr(), global_id_index) };
            if !handle.is_null() {
                handles.push(handle);
            }
        }
        if handles.is_empty() {
            return Err(Error(format!(
                "no RoCE port to open{}",
                if device.is_empty() {
                    String::new()
                } else {
                    format!(" on {device}")
                }
            )));
        }
        Ok(Device { handles })
    }

    /// How many paths a payload is split over.
    pub fn paths(&self) -> usize {
        self.handles.len()
    }

    /// One connection of this device, for one peer: a queue pair on each of its paths.
    pub fn link(&self) -> Result<Link, Error> {
        let mut handles = Vec::with_capacity(self.handles.len());
        for device in &self.handles {
            // SAFETY: the handle is live, and the link keeps the device alive through the borrow.
            let handle = unsafe { mmh3_rdma_link_open(*device) };
            if handle.is_null() {
                return Err(Error("opening a connection".to_owned()));
            }
            handles.push(handle);
        }
        Ok(Link { handles })
    }

    /// Takes `buffer` and registers it on this device, which every connection of it may then
    /// serve. The region owns the memory so that it can outlive one transfer: registering hundreds
    /// of megabytes costs seconds, and every transfer of a run reuses the same region.
    pub fn register(&self, buffer: Vec<u8>) -> Result<Region, Error> {
        let mut buffer = buffer;
        let (address, bytes) = (buffer.as_ptr() as u64, buffer.len());
        let mut handles = Vec::with_capacity(self.handles.len());
        let mut keys = Vec::with_capacity(self.handles.len());
        // The paths have a protection domain each, so the same memory takes a registration and a
        // key on every one of them. The address is the buffer's either way.
        for device in &self.handles {
            let mut remote_key = 0u32;
            // SAFETY: the buffer moves into the region, which keeps it at this address until it
            // unregisters in `Drop`.
            let handle = unsafe {
                mmh3_rdma_register(*device, buffer.as_mut_ptr().cast(), bytes, &mut remote_key)
            };
            if handle.is_null() {
                return Err(Error(format!("registering {bytes} bytes")));
            }
            handles.push(handle);
            keys.push(remote_key);
        }
        Ok(Region {
            handles,
            address,
            keys,
            buffer,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        for handle in &self.handles {
            // SAFETY: the handles came from `open` and are dropped once, after every link of them.
            unsafe { mmh3_rdma_close(*handle) };
        }
    }
}

/// One reliable connection to one peer: a queue pair on each of the device's paths.
pub struct Link {
    handles: Vec<*mut c_void>,
}

// The handle is only ever used from the thread that owns the link, so it moves between threads but
// is never shared.
unsafe impl Send for Link {}

impl Link {
    pub fn paths(&self) -> usize {
        self.handles.len()
    }

    /// This side's addresses, one per path, for the other side to connect to.
    pub fn addresses(&self) -> Result<Vec<Address>, Error> {
        let mut addresses = Vec::with_capacity(self.handles.len());
        for handle in &self.handles {
            let mut address = Address::default();
            // SAFETY: the handle is live and the address is this call's to fill.
            let status = unsafe { mmh3_rdma_link_address(*handle, &mut address) };
            if status != 0 {
                return Err(Error(format!("reading the local address: {status}")));
            }
            addresses.push(address);
        }
        Ok(addresses)
    }

    /// Moves every path to ready. Both sides call this with the other's addresses, which the two
    /// take in the same order, so a path talks to the one the peer opened beside it.
    pub fn connect(&self, peers: &[Address]) -> Result<(), Error> {
        if peers.len() != self.handles.len() {
            return Err(Error(format!(
                "{} addresses for {} paths",
                peers.len(),
                self.handles.len()
            )));
        }
        for (handle, peer) in self.handles.iter().zip(peers) {
            // SAFETY: the handle is live and the peer outlives the call.
            let status = unsafe { mmh3_rdma_link_connect(*handle, peer) };
            if status != 0 {
                return Err(Error(format!("connecting: {status}")));
            }
        }
        Ok(())
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        for handle in &self.handles {
            // SAFETY: the handles came from `Device::link` and are dropped once.
            unsafe { mmh3_rdma_link_close(*handle) };
        }
    }
}

/// Memory the device's connections may read and write, and what the other side needs to reach it.
pub struct Region {
    handles: Vec<*mut c_void>,
    address: u64,
    keys: Vec<u32>,
    buffer: Vec<u8>,
}

// The region owns its memory and the handle that describes it, and one thread uses both.
unsafe impl Send for Region {}

impl Region {
    /// Where this memory sits, for the other side's read.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// One key per path, which the peer needs all of to reach this memory.
    pub fn remote_keys(&self) -> &[u32] {
        &self.keys
    }

    pub fn bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    /// Splits `bytes` of this region, `offset` bytes in, over the connection's paths and writes
    /// them into the peer's memory. Over the link between two Sparks one path runs at 13.0 GB/s
    /// and the two together at 22.2, so a rank pushes its rows to the peers rather than letting
    /// them pull, and splits them.
    pub fn write_out(
        &self,
        link: &Link,
        offset: usize,
        remote_address: u64,
        remote_keys: &[u32],
        bytes: usize,
        milliseconds: i32,
    ) -> Result<(), Error> {
        self.parted(
            link,
            offset,
            remote_address,
            remote_keys,
            bytes,
            true,
            milliseconds,
        )
    }

    /// `write_out` the other way round, which is how a canvas comes back from a worker.
    pub fn read_into(
        &mut self,
        link: &Link,
        offset: usize,
        remote_address: u64,
        remote_keys: &[u32],
        bytes: usize,
        milliseconds: i32,
    ) -> Result<(), Error> {
        self.parted(
            link,
            offset,
            remote_address,
            remote_keys,
            bytes,
            false,
            milliseconds,
        )
    }

    /// `read_into` from the start of the region.
    pub fn read_from(
        &mut self,
        link: &Link,
        remote_address: u64,
        remote_keys: &[u32],
        bytes: usize,
        milliseconds: i32,
    ) -> Result<(), Error> {
        self.read_into(link, 0, remote_address, remote_keys, bytes, milliseconds)
    }

    /// One transfer, cut into a piece per path. The cut is by whole 4 KiB pages so that neither
    /// side straddles one, and the last path takes the remainder.
    #[allow(clippy::too_many_arguments)]
    fn parted(
        &self,
        link: &Link,
        offset: usize,
        remote_address: u64,
        remote_keys: &[u32],
        bytes: usize,
        writing: bool,
        milliseconds: i32,
    ) -> Result<(), Error> {
        if offset + bytes > self.buffer.len() {
            return Err(Error(format!(
                "{bytes} bytes at {offset} of {} of memory",
                self.buffer.len()
            )));
        }
        let paths = self
            .handles
            .len()
            .min(link.handles.len())
            .min(remote_keys.len());
        if paths == 0 {
            return Err(Error("a connection with no path".to_owned()));
        }
        const PAGE: usize = 4096;
        let each = (bytes / paths / PAGE) * PAGE;
        let mut parts = Vec::with_capacity(paths);
        for (path, key) in remote_keys.iter().enumerate().take(paths) {
            let start = path * each;
            let length = if path + 1 == paths {
                bytes - start
            } else {
                each
            };
            parts.push(Part {
                link: link.handles[path],
                region: self.handles[path],
                // SAFETY: the range lies inside the buffer, checked above.
                local: unsafe { self.buffer.as_ptr().add(offset + start).cast_mut() },
                bytes: length,
                remote_address: remote_address + start as u64,
                remote_key: *key,
            });
        }
        // SAFETY: every part points inside this region's buffer and names a live connection.
        let status = unsafe {
            if writing {
                mmh3_rdma_write_all(parts.as_ptr(), parts.len() as c_int, milliseconds)
            } else {
                mmh3_rdma_read_all(parts.as_ptr(), parts.len() as c_int, milliseconds)
            }
        };
        if status != 0 {
            let what = if writing { "writing" } else { "reading" };
            return Err(Error(format!("{what} {bytes} bytes: {status}")));
        }
        Ok(())
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        for handle in &self.handles {
            // SAFETY: the handles came from `register` and are dropped once, before the buffer.
            unsafe { mmh3_rdma_unregister(*handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_survives_the_wire() {
        let address = Address {
            queue_pair: 0x0228,
            sequence: 0xee6711,
            local_id: 0,
            global_id: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 169, 254, 102, 229],
        };
        assert_eq!(Address::from_bytes(&address.to_bytes()), Some(address));
        assert_eq!(Address::from_bytes(&[0u8; 4]), None);
    }
}
