//! Weights read from their checkpoints while a model runs, for a device that cannot hold them all.
//!
//! A model is cut into units, a layer or a block each, and the device either keeps a unit or reads
//! it again every time it runs it. A unit is laid out as one region, so that its bytes land in
//! place with one copy, whether that place is a buffer of its own or one of the regions the units
//! read on the way take turns in.
//!
//! Which units a device keeps, the order it gives them up in and how their bytes come off the disk
//! are the same on every backend, so they live here. How the bytes reach the device, and how a
//! backend knows that a region is free again, are its own business.

use crate::direct_file::{DIRECT_ALIGNMENT, DirectFile};
use crate::safetensors::{DType, SafeTensors, TensorInfo};
use std::alloc::{Layout, alloc, dealloc};
use std::io;
use std::path::{Path, PathBuf};

/// Tensors of a unit start at multiples of this, which every kernel's vector loads accept.
pub const UNIT_ALIGNMENT: usize = 256;

/// Bytes a reader holds between the file and the unit.
const STAGING_BYTES: usize = 64 << 20;
/// Pieces closer than this are read in one pass rather than seeking over the gap.
const MERGE_GAP: u64 = 4 << 20;

/// The bytes of one tensor, or of one part of a tensor, in a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    /// Index of the checkpoint in the unit's `Files`.
    pub file: usize,
    /// Offset from the start of the file.
    pub offset: u64,
    pub bytes: usize,
}

/// How the pieces of a tensor are arranged in its place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrangement {
    /// End to end, in order.
    Contiguous,
    /// Rows of `row_bytes` whose first half and second half are interleaved `group` rows at a
    /// time: `group` rows of the first half, then the same rows of the second half. It is the order
    /// in which the CUDA INT8 GEMM wants the SwiGLU gate and up projections.
    InterleavedHalves { row_bytes: usize, group: usize },
}

/// One tensor of a unit and where it sits in the unit's region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub pieces: Vec<Piece>,
    pub arrangement: Arrangement,
    /// Offset from the start of the unit's region.
    pub offset: usize,
}

impl UnitTensor {
    pub fn bytes(&self) -> usize {
        self.pieces.iter().map(|piece| piece.bytes).sum()
    }

    /// Where byte `position` of the tensor's pieces, taken end to end, lands in the tensor, and how
    /// many bytes from there on follow it without a break.
    fn place(&self, position: usize) -> (usize, usize) {
        match self.arrangement {
            Arrangement::Contiguous => (position, self.bytes() - position),
            Arrangement::InterleavedHalves { row_bytes, group } => {
                let half = self.bytes() / row_bytes / 2;
                let (row, column) = (position / row_bytes, position % row_bytes);
                let (second, row) = (row >= half, row % half);
                let placed = row / group * 2 * group + usize::from(second) * group + row % group;
                let run = (group - row % group) * row_bytes - column;
                (placed * row_bytes + column, run)
            }
        }
    }
}

/// A group of tensors the device keeps or reads again together, such as one block of a model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Unit {
    pub name: String,
    tensors: Vec<UnitTensor>,
    bytes: usize,
}

impl Unit {
    pub fn new(name: &str) -> Self {
        Unit {
            name: name.to_owned(),
            tensors: Vec::new(),
            bytes: 0,
        }
    }

    pub fn tensors(&self) -> &[UnitTensor] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&UnitTensor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Bytes of the unit's region.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Adds a tensor, or replaces the one of the same name, and lays the unit out again.
    pub fn insert(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<usize>,
        pieces: Vec<Piece>,
        arrangement: Arrangement,
    ) {
        let tensor = UnitTensor {
            name: name.to_owned(),
            dtype,
            shape,
            pieces,
            arrangement,
            offset: 0,
        };
        if let Arrangement::InterleavedHalves { row_bytes, group } = arrangement {
            assert!(
                tensor.bytes().is_multiple_of(2 * group * row_bytes),
                "{name}: the rows do not split into halves of whole groups"
            );
        }
        match self.tensors.iter_mut().find(|kept| kept.name == name) {
            Some(kept) => *kept = tensor,
            None => self.tensors.push(tensor),
        }
        let mut offset = 0;
        for tensor in &mut self.tensors {
            tensor.offset = offset;
            offset = (offset + tensor.bytes()).next_multiple_of(UNIT_ALIGNMENT);
        }
        self.bytes = offset;
    }
}

/// The checkpoints the pieces of a model's units come from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Files {
    paths: Vec<PathBuf>,
}

impl Files {
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// The piece of `file` that holds `tensor`.
    pub fn piece(&mut self, file: &SafeTensors, tensor: &TensorInfo) -> Piece {
        Piece {
            file: self.index(file.path()),
            offset: file.file_offset(tensor),
            bytes: tensor.byte_count(),
        }
    }

    fn index(&mut self, path: &Path) -> usize {
        match self.paths.iter().position(|known| known == path) {
            Some(index) => index,
            None => {
                self.paths.push(path.to_owned());
                self.paths.len() - 1
            }
        }
    }
}

/// Host memory aligned for direct reads.
struct AlignedBuffer {
    pointer: *mut u8,
    bytes: usize,
}

impl AlignedBuffer {
    fn new(bytes: usize) -> Self {
        let layout = Layout::from_size_align(bytes, DIRECT_ALIGNMENT).expect("a valid layout");
        // SAFETY: the layout has a non-zero size.
        let pointer = unsafe { alloc(layout) };
        if pointer.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBuffer { pointer, bytes }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation holds `bytes` bytes and lives as long as self.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.bytes) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.bytes, DIRECT_ALIGNMENT).expect("a valid layout");
        // SAFETY: the pointer came from `alloc` with this layout.
        unsafe { dealloc(self.pointer, layout) };
    }
}

// SAFETY: the buffer is plain memory owned by one reader.
unsafe impl Send for AlignedBuffer {}

/// Reads units from their checkpoints with direct reads, so that a unit read at every step leaves
/// nothing in the page cache.
pub struct UnitReader {
    files: Vec<DirectFile>,
    staging: AlignedBuffer,
}

/// One piece of one tensor of the unit being read.
struct Wanted {
    file: usize,
    offset: u64,
    bytes: usize,
    tensor: usize,
    /// Offset of the piece within the tensor's pieces, taken end to end.
    position: usize,
}

impl UnitReader {
    pub fn new(files: &Files) -> io::Result<Self> {
        Ok(UnitReader {
            files: files
                .paths
                .iter()
                .map(|path| DirectFile::open(path))
                .collect::<io::Result<_>>()?,
            staging: AlignedBuffer::new(STAGING_BYTES),
        })
    }

    /// Reads `unit` into `destination`, laid out as the unit says. The files are those the reader
    /// was made with, so a unit whose pieces name a file added since needs a new reader.
    pub fn read(&mut self, unit: &Unit, destination: &mut [u8]) -> io::Result<()> {
        assert!(
            destination.len() >= unit.bytes(),
            "{}: the destination is smaller than the unit",
            unit.name
        );
        let mut wanted = Vec::new();
        for (index, tensor) in unit.tensors.iter().enumerate() {
            let mut position = 0;
            for piece in &tensor.pieces {
                if piece.file >= self.files.len() {
                    return Err(io::Error::other(format!(
                        "{}: a piece from a file the reader does not have",
                        tensor.name
                    )));
                }
                wanted.push(Wanted {
                    file: piece.file,
                    offset: piece.offset,
                    bytes: piece.bytes,
                    tensor: index,
                    position,
                });
                position += piece.bytes;
            }
        }
        wanted.sort_by_key(|piece| (piece.file, piece.offset));

        // Aligned ranges covering the pieces, merged across small gaps, with the end of the data
        // each one needs and the pieces it covers.
        let alignment = DIRECT_ALIGNMENT as u64;
        let mut ranges: Vec<(usize, u64, u64, u64, std::ops::Range<usize>)> = Vec::new();
        for (index, piece) in wanted.iter().enumerate() {
            let data_end = piece.offset + piece.bytes as u64;
            let start = piece.offset / alignment * alignment;
            let end = data_end.div_ceil(alignment) * alignment;
            match ranges.last_mut() {
                Some(last) if last.0 == piece.file && start <= last.2 + MERGE_GAP => {
                    last.2 = last.2.max(end);
                    last.3 = last.3.max(data_end);
                    last.4.end = index + 1;
                }
                _ => ranges.push((piece.file, start, end, data_end, index..index + 1)),
            }
        }

        for (file, range_start, range_end, data_end, covered) in ranges {
            let mut position = range_start;
            while position < range_end {
                let length = ((range_end - position) as usize).min(STAGING_BYTES);
                let staging = &mut self.staging.as_mut_slice()[..length];
                let read = self.files[file].read_at(position, staging)?;
                let chunk_end = position + read as u64;
                // Only the aligned tail past the end of the file may come back short.
                if chunk_end < data_end.min(position + length as u64) {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "the checkpoint is shorter than its header says",
                    ));
                }
                for piece in &wanted[covered.clone()] {
                    let start = piece.offset.max(position);
                    let end = (piece.offset + piece.bytes as u64).min(chunk_end);
                    if start >= end {
                        continue;
                    }
                    let tensor = &unit.tensors[piece.tensor];
                    let mut from = (start - position) as usize;
                    let mut at = piece.position + (start - piece.offset) as usize;
                    let mut left = (end - start) as usize;
                    while left > 0 {
                        let (placed, run) = tensor.place(at);
                        let length = run.min(left);
                        let target = tensor.offset + placed;
                        destination[target..target + length]
                            .copy_from_slice(&staging[from..from + length]);
                        from += length;
                        at += length;
                        left -= length;
                    }
                }
                position += length as u64;
            }
        }
        Ok(())
    }
}

/// The order in which a device gives up `count` units of the same kind, spread out so that the
/// units it reads on the way have units it keeps between them. A unit read while the ones before it
/// run is a unit the device does not wait for.
pub fn give_up_order(count: usize) -> Vec<usize> {
    let bits = count.next_power_of_two().trailing_zeros();
    (0..count.next_power_of_two())
        .map(|index| match bits {
            0 => 0,
            bits => index.reverse_bits() >> (usize::BITS - bits),
        })
        .filter(|&index| index < count)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_checkpoint(path: &Path, tensors: &[(&str, &[u8])]) {
        let mut header = String::from("{");
        let mut offset = 0;
        for (index, (name, bytes)) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"U8\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
                bytes.len(),
                offset + bytes.len()
            ));
            offset += bytes.len();
        }
        header.push('}');
        let mut contents = (header.len() as u64).to_le_bytes().to_vec();
        contents.extend_from_slice(header.as_bytes());
        for (_, bytes) in tensors {
            contents.extend_from_slice(bytes);
        }
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn reads_a_unit_into_its_layout() {
        let path = std::env::temp_dir().join(format!("mmh3-streaming-{}", std::process::id()));
        // Two tensors of the same layer, a third one far away, and a matrix of eight rows of four
        // bytes whose halves interleave two rows at a time.
        let first: Vec<u8> = (0..1000).map(|index| (index % 251) as u8).collect();
        let second: Vec<u8> = (0..300).map(|index| (index % 7) as u8 + 100).collect();
        let gap = vec![0u8; 9 << 20];
        let rows: Vec<u8> = (0..32).collect();
        write_checkpoint(
            &path,
            &[
                ("first", &first),
                ("second", &second),
                ("gap", &gap),
                ("rows", &rows),
            ],
        );
        let file = SafeTensors::open(&path).unwrap();
        let mut files = Files::default();
        let mut unit = Unit::new("layer");
        let piece = |files: &mut Files, name: &str| files.piece(&file, file.get(name).unwrap());
        let pieces = vec![piece(&mut files, "second"), piece(&mut files, "first")];
        unit.insert(
            "joined",
            DType::U8,
            vec![1300],
            pieces,
            Arrangement::Contiguous,
        );
        let pieces = vec![piece(&mut files, "rows")];
        unit.insert(
            "rows",
            DType::U8,
            vec![8, 4],
            pieces,
            Arrangement::InterleavedHalves {
                row_bytes: 4,
                group: 2,
            },
        );
        assert_eq!(unit.get("rows").unwrap().offset, 1536);
        assert_eq!(unit.bytes(), 1536 + 256);

        let mut destination = vec![0u8; unit.bytes()];
        UnitReader::new(&files)
            .unwrap()
            .read(&unit, &mut destination)
            .unwrap();
        assert_eq!(&destination[..300], &second[..]);
        assert_eq!(&destination[300..1300], &first[..]);
        let row = |index: u8| (index * 4..index * 4 + 4).collect::<Vec<u8>>();
        let expected: Vec<u8> = [0, 1, 4, 5, 2, 3, 6, 7].into_iter().flat_map(row).collect();
        assert_eq!(&destination[1536..1568], &expected[..]);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn gives_up_units_spread_out() {
        for count in [0, 1, 2, 3, 7, 50, 64] {
            let mut order = give_up_order(count);
            assert_eq!(order.len(), count);
            order.sort_unstable();
            assert!(order.iter().copied().eq(0..count));
        }
        let order = give_up_order(50);
        assert_eq!(&order[..4], &[0, 32, 16, 48]);
        // Any first few units given up are at least a few units apart.
        let mut first: Vec<usize> = order[..6].to_vec();
        first.sort_unstable();
        assert!(first.windows(2).all(|pair| pair[1] - pair[0] >= 4));
    }
}
