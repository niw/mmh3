//! One reliable connection to another machine over RoCE, and reads of its memory.
//!
//! The bulk of a distributed decode or a sharded step is too large for a socket: over the link
//! between two DGX Sparks, `ib_write_bw` reaches 13.3 GB/s where TCP reaches 1.5. Control stays on
//! the socket, which every machine has, and only the payloads come this way.
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

unsafe extern "C" {
    fn mmh3_rdma_open(device: *const c_char, global_id_index: c_int) -> *mut c_void;
    fn mmh3_rdma_close(rdma: *mut c_void);
    fn mmh3_rdma_address(rdma: *mut c_void, address: *mut Address) -> c_int;
    fn mmh3_rdma_connect(rdma: *mut c_void, peer: *const Address) -> c_int;
    fn mmh3_rdma_register(
        rdma: *mut c_void,
        buffer: *mut c_void,
        bytes: usize,
        remote_key: *mut u32,
    ) -> *mut c_void;
    fn mmh3_rdma_unregister(region: *mut c_void);
    fn mmh3_rdma_read_all(
        rdma: *mut c_void,
        region: *mut c_void,
        local: *mut c_void,
        bytes: usize,
        remote_address: u64,
        remote_key: u32,
        milliseconds: c_int,
    ) -> c_int;
}

/// The RoCEv2 entry of a port's table, which is what these machines use.
pub const DEFAULT_GLOBAL_ID: i32 = 3;

pub struct Connection {
    handle: *mut c_void,
}

// The handle is only ever used from the thread that owns the connection, and one connection serves
// one peer, so it moves between threads but is never shared.
unsafe impl Send for Connection {}

impl Connection {
    /// Opens the first active port of `device`, or of any device when it is empty.
    pub fn open(device: &str, global_id_index: i32) -> Result<Self, Error> {
        let name = CString::new(device).map_err(|_| Error("a device name has a nul".to_owned()))?;
        // SAFETY: the name outlives the call, which copies what it needs.
        let handle = unsafe { mmh3_rdma_open(name.as_ptr(), global_id_index) };
        if handle.is_null() {
            return Err(Error(format!(
                "no RoCE port to open{}",
                if device.is_empty() {
                    String::new()
                } else {
                    format!(" on {device}")
                }
            )));
        }
        Ok(Connection { handle })
    }

    /// This side's address, for the other side to connect to.
    pub fn address(&self) -> Result<Address, Error> {
        let mut address = Address::default();
        // SAFETY: the handle is live and the address is this call's to fill.
        let status = unsafe { mmh3_rdma_address(self.handle, &mut address) };
        if status != 0 {
            return Err(Error(format!("reading the local address: {status}")));
        }
        Ok(address)
    }

    /// Moves the connection to ready. Both sides call this with the other's address.
    pub fn connect(&self, peer: &Address) -> Result<(), Error> {
        // SAFETY: the handle is live and the peer outlives the call.
        let status = unsafe { mmh3_rdma_connect(self.handle, peer) };
        if status != 0 {
            return Err(Error(format!("connecting: {status}")));
        }
        Ok(())
    }

    /// Takes `buffer` and registers it for this connection. The region owns the memory so that it
    /// can outlive one transfer: registering hundreds of megabytes costs seconds, and every
    /// transfer of a run reuses the same region.
    pub fn register(&self, buffer: Vec<u8>) -> Result<Region, Error> {
        let mut buffer = buffer;
        let mut remote_key = 0u32;
        let (address, bytes) = (buffer.as_ptr() as u64, buffer.len());
        // SAFETY: the buffer moves into the region, which keeps it at this address until it
        // unregisters in `Drop`.
        let handle = unsafe {
            mmh3_rdma_register(
                self.handle,
                buffer.as_mut_ptr().cast(),
                bytes,
                &mut remote_key,
            )
        };
        if handle.is_null() {
            return Err(Error(format!("registering {bytes} bytes")));
        }
        Ok(Region {
            handle,
            address,
            remote_key,
            buffer,
        })
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        // SAFETY: the handle came from `open` and is dropped once.
        unsafe { mmh3_rdma_close(self.handle) };
    }
}

/// Memory this connection may read and write, and what the other side needs to reach it.
pub struct Region {
    handle: *mut c_void,
    address: u64,
    remote_key: u32,
    buffer: Vec<u8>,
}

// The region owns its memory and the handle that describes it, and one thread uses both.
unsafe impl Send for Region {}

impl Region {
    /// Where this memory sits, for the other side's read.
    pub fn address(&self) -> u64 {
        self.address
    }

    pub fn remote_key(&self) -> u32 {
        self.remote_key
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

    /// Reads the peer's memory into this one, splitting what one work request cannot carry.
    pub fn read_from(
        &mut self,
        connection: &Connection,
        remote_address: u64,
        remote_key: u32,
        bytes: usize,
        milliseconds: i32,
    ) -> Result<(), Error> {
        self.read_into(
            connection,
            0,
            remote_address,
            remote_key,
            bytes,
            milliseconds,
        )
    }

    /// `read_from` landing `offset` bytes into this region, which is how a rank collects the peers'
    /// shares of one sequence side by side.
    pub fn read_into(
        &mut self,
        connection: &Connection,
        offset: usize,
        remote_address: u64,
        remote_key: u32,
        bytes: usize,
        milliseconds: i32,
    ) -> Result<(), Error> {
        if offset + bytes > self.buffer.len() {
            return Err(Error(format!(
                "a read of {bytes} bytes at {offset} into {} of memory",
                self.buffer.len()
            )));
        }
        // SAFETY: the region and its buffer are live, and the peer's range is its own to check.
        let status = unsafe {
            mmh3_rdma_read_all(
                connection.handle,
                self.handle,
                self.buffer.as_mut_ptr().add(offset).cast(),
                bytes,
                remote_address,
                remote_key,
                milliseconds,
            )
        };
        if status != 0 {
            return Err(Error(format!("reading {bytes} bytes: {status}")));
        }
        Ok(())
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: the handle came from `register` and is dropped once, before the buffer.
        unsafe { mmh3_rdma_unregister(self.handle) };
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
