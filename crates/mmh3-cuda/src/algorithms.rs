//! The cuBLASLt algorithms chosen for the GEMM shapes of a run, the files that keep them between
//! runs, and the consistency that chooses none by measuring.
//!
//! A choice comes from timing the heuristic's candidates on the shape's own operands, which costs
//! about a second on the first step of a run. A file taken over at startup spares it: a run that
//! has met its shapes before times none of them.
//!
//! A choice belongs to the GPU it was timed on, so every device keeps a table of its own and a
//! file keeps a section for every kind of GPU that wrote to it. Two kinds of card in one machine
//! then each find theirs, rather than each saving over the other's and timing everything again.

use crate::{CudaError, MAX_DEVICES, check, device_count};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int};
use std::path::Path;
use std::{fs, io, ptr};

unsafe extern "C" {
    fn mmh3_cublaslt_matmul_algorithms(
        device: c_int,
        algorithms: *mut Algorithm,
        capacity: c_int,
    ) -> c_int;
    fn mmh3_cublaslt_matmul_adopt(device: c_int, algorithms: *const Algorithm, count: c_int);
    fn mmh3_cublaslt_matmul_algorithm_key(
        device: c_int,
        key: *mut c_char,
        capacity: c_int,
    ) -> c_int;
    fn mmh3_cublaslt_nvfp4_algorithms(
        device: c_int,
        algorithms: *mut Nvfp4Algorithm,
        capacity: c_int,
    ) -> c_int;
    fn mmh3_cublaslt_nvfp4_adopt(device: c_int, algorithms: *const Nvfp4Algorithm, count: c_int);
    fn mmh3_cublaslt_nvfp4_algorithm_key(device: c_int, key: *mut c_char, capacity: c_int)
    -> c_int;
    fn mmh3_cublaslt_consistent_by_default(consistent: c_int);
    fn mmh3_cublaslt_consistent_on_this_thread(consistent: c_int);
}

/// Makes the GEMMs of every thread that has not been told otherwise consistent, or not. A
/// consistent GEMM gives every row the same arithmetic whatever the shape of the call it is in, the
/// process it runs in or what a measurement said, so a step split across ranks or machines comes
/// out bit for bit what it does whole. It runs the algorithm cuBLASLt's heuristic names for the
/// layer rather than the fastest one.
pub fn set_consistent(consistent: bool) {
    // SAFETY: it stores a flag.
    unsafe { mmh3_cublaslt_consistent_by_default(c_int::from(consistent)) }
}

/// Makes the GEMMs of this thread consistent or not whatever `set_consistent` said, which is how a
/// worker's connection follows the leader it serves.
pub fn set_consistent_on_this_thread(consistent: bool) {
    // SAFETY: it stores a flag of this thread.
    unsafe { mmh3_cublaslt_consistent_on_this_thread(c_int::from(consistent)) }
}

/// The algorithm chosen for a plain GEMM. The fields before the words are the key of the choice,
/// and the words are the algorithm as cuBLASLt hands it over, in the layout the backend reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Algorithm {
    kind: i64,
    bias: i64,
    m: i64,
    n: i64,
    k: i64,
    data: [u64; 8],
}

/// The algorithm chosen for an NVFP4 GEMM of `m × k` activations and `n` outputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Nvfp4Algorithm {
    m: i64,
    n: i64,
    k: i64,
    data: [u64; 8],
}

/// First line of a file of chosen algorithms, followed by a section for every key: the line the key
/// writes, which starts with [`KEY_PREFIX`], and a line per shape, the fields that name the shape
/// and the eight words of the algorithm in hexadecimal. A file of one section is what a build
/// before the sections wrote.
const MATMUL_HEADER: &str = "mmh3 cuBLASLt matmul algorithms";
const NVFP4_HEADER: &str = "mmh3 cuBLASLt NVFP4 algorithms";

/// How every key line starts, which no line of numbers does.
const KEY_PREFIX: &str = "cuBLASLt ";

fn parse(line: &str, fields: usize) -> Option<(Vec<i64>, [u64; 8])> {
    let columns: Vec<&str> = line.split_whitespace().collect();
    if columns.len() != fields + 8 {
        return None;
    }
    let named = columns[..fields]
        .iter()
        .map(|text| text.parse().ok())
        .collect::<Option<Vec<i64>>>()?;
    let mut data = [0; 8];
    for (word, text) in data.iter_mut().zip(&columns[fields..]) {
        *word = u64::from_str_radix(text, 16).ok()?;
    }
    Some((named, data))
}

fn line(named: &[i64], data: &[u64; 8]) -> String {
    let mut text = named
        .iter()
        .map(i64::to_string)
        .collect::<Vec<String>>()
        .join(" ");
    for word in data {
        text.push_str(&format!(" {word:016x}"));
    }
    text.push('\n');
    text
}

/// What a table of chosen algorithms depends on, as the device names it.
fn key_of(of: impl FnOnce(*mut c_char, c_int) -> c_int) -> Result<String, CudaError> {
    let mut key = [0u8; 512];
    check(of(key.as_mut_ptr().cast(), key.len() as c_int))?;
    Ok(CStr::from_bytes_until_nul(&key)
        .map(|key| key.to_string_lossy().into_owned())
        .unwrap_or_default())
}

/// The sections of `text` by key, and none from a damaged file or from one written under another
/// header.
fn sections(text: &str, header: &str) -> BTreeMap<String, Vec<String>> {
    let mut sections = BTreeMap::new();
    let mut lines = text.lines();
    if lines.next() != Some(header) {
        return sections;
    }
    let mut current: Option<&mut Vec<String>> = None;
    for line in lines {
        if line.starts_with(KEY_PREFIX) {
            current = Some(sections.entry(line.to_owned()).or_default());
        } else if let Some(entries) = current.as_mut()
            && !line.trim().is_empty()
        {
            entries.push(line.to_owned());
        }
    }
    sections
}

/// The sections `path` keeps, and none from a missing file.
fn read(path: &Path, header: &str) -> io::Result<BTreeMap<String, Vec<String>>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(sections(&text, header)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(error) => Err(error),
    }
}

/// `text` with the sections of `saved` in place of its own of the same keys, and every other one
/// kept, so that a GPU this process does not have loses nothing by its saving.
fn merged(text: &str, header: &str, saved: BTreeMap<String, Vec<String>>) -> String {
    let mut all = sections(text, header);
    all.extend(saved);
    let mut written = format!("{header}\n");
    for (key, entries) in all {
        written.push_str(&key);
        written.push('\n');
        for entry in entries {
            written.push_str(&entry);
            written.push('\n');
        }
    }
    written
}

/// Writes `saved` into `path` beside the sections of other keys it holds, and returns whether it
/// wrote. It writes nothing when the file already holds them.
fn write(path: &Path, header: &str, saved: BTreeMap<String, Vec<String>>) -> io::Result<bool> {
    let existing = match fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let text = merged(existing.as_deref().unwrap_or(""), header, saved);
    if existing.is_some_and(|existing| existing == text) {
        return Ok(false);
    }
    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory)?;
    }
    // Another process may be saving the same file, so each writes a temporary file of its own and
    // the rename decides whose lands, rather than two writing one file into a mix.
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    fs::write(&temporary, text)?;
    fs::rename(&temporary, path)?;
    Ok(true)
}

/// The devices this process may have chosen algorithms on.
fn devices() -> std::ops::Range<usize> {
    0..device_count().unwrap_or(0).min(MAX_DEVICES)
}

/// Takes the section of every device's key out of `path` and hands its entries to `adopt`, and
/// returns how many entries it took, counting a section two devices share once.
fn load<T>(
    path: &Path,
    header: &str,
    key: impl Fn(usize) -> Result<String, CudaError>,
    parse: impl Fn(&str) -> Option<T>,
    adopt: impl Fn(usize, &[T]),
) -> io::Result<usize> {
    let sections = read(path, header)?;
    let mut taken = 0;
    let mut counted = Vec::new();
    for device in devices() {
        let key = key(device).map_err(io::Error::other)?;
        let Some(entries) = sections.get(&key) else {
            continue;
        };
        let Some(algorithms) = entries
            .iter()
            .map(|entry| parse(entry))
            .collect::<Option<Vec<T>>>()
        else {
            continue;
        };
        adopt(device, &algorithms);
        if !counted.contains(&key) {
            taken += algorithms.len();
            counted.push(key);
        }
    }
    Ok(taken)
}

/// Writes what every device chose to `path`, a section per key, and returns whether it wrote. The
/// devices of one key share a section, since what one of them timed is as good for the other.
fn save<T>(
    path: &Path,
    header: &str,
    key: impl Fn(usize) -> Result<String, CudaError>,
    chosen: impl Fn(usize) -> Vec<T>,
    line_of: impl Fn(&T) -> String,
) -> io::Result<bool> {
    let mut saved: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for device in devices() {
        let algorithms = chosen(device);
        if algorithms.is_empty() {
            continue;
        }
        let key = key(device).map_err(io::Error::other)?;
        let entries = saved.entry(key).or_default();
        for algorithm in &algorithms {
            let line = line_of(algorithm);
            let line = line.trim_end();
            // Two devices of one kind may both have chosen for a shape, and the first choice is
            // the one kept, as adopting it would.
            let shape = |entry: &str| entry.rsplitn(9, ' ').nth(8).map(str::to_owned);
            if !entries.iter().any(|entry| shape(entry) == shape(line)) {
                entries.push(line.to_owned());
            }
        }
    }
    if saved.is_empty() {
        return Ok(false);
    }
    // In the order of the shapes rather than of the devices that chose them, so that the same
    // choices write the same file whichever device met a shape first.
    for entries in saved.values_mut() {
        entries.sort();
    }
    write(path, header, saved)
}

fn chosen<T: Copy + Default>(of: impl Fn(*mut T, c_int) -> c_int) -> Vec<T> {
    // SAFETY of both calls: a capacity of 0 writes nothing, and the vector holds `count` entries.
    let count = of(ptr::null_mut(), 0) as usize;
    let mut algorithms = vec![T::default(); count];
    let total = of(algorithms.as_mut_ptr(), count as c_int);
    algorithms.truncate(total as usize);
    algorithms
}

/// What the algorithms chosen on `device` depend on: a section is taken over only by a device whose
/// key it names, which one written on another GPU or with another cuBLASLt does not.
fn matmul_key(device: usize) -> Result<String, CudaError> {
    key_of(|key, capacity| unsafe {
        mmh3_cublaslt_matmul_algorithm_key(device as c_int, key, capacity)
    })
}

fn nvfp4_key(device: usize) -> Result<String, CudaError> {
    key_of(|key, capacity| unsafe {
        mmh3_cublaslt_nvfp4_algorithm_key(device as c_int, key, capacity)
    })
}

/// Takes over the algorithms that `save_matmul` left in `path` for ordinary GEMM shapes, each device
/// those of its own key, and returns how many it took. A shape that has one chosen already keeps it.
pub fn load_matmul(path: &Path) -> io::Result<usize> {
    load(
        path,
        MATMUL_HEADER,
        matmul_key,
        |entry| {
            let (named, data) = parse(entry, 5)?;
            Some(Algorithm {
                kind: named[0],
                bias: named[1],
                m: named[2],
                n: named[3],
                k: named[4],
                data,
            })
        },
        // SAFETY: the slice holds `algorithms.len()` algorithms.
        |device, algorithms| unsafe {
            mmh3_cublaslt_matmul_adopt(
                device as c_int,
                algorithms.as_ptr(),
                algorithms.len() as c_int,
            )
        },
    )
}

/// Writes the algorithms this process chose for ordinary GEMM shapes to `path` for `load_matmul`,
/// and returns whether it wrote.
pub fn save_matmul(path: &Path) -> io::Result<bool> {
    save(
        path,
        MATMUL_HEADER,
        matmul_key,
        |device| {
            chosen(|algorithms, capacity| unsafe {
                mmh3_cublaslt_matmul_algorithms(device as c_int, algorithms, capacity)
            })
        },
        |algorithm: &Algorithm| {
            line(
                &[
                    algorithm.kind,
                    algorithm.bias,
                    algorithm.m,
                    algorithm.n,
                    algorithm.k,
                ],
                &algorithm.data,
            )
        },
    )
}

/// Takes over the algorithms that `save_nvfp4` left in `path` for NVFP4 GEMM shapes, each device
/// those of its own key, and returns how many it took.
pub fn load_nvfp4(path: &Path) -> io::Result<usize> {
    load(
        path,
        NVFP4_HEADER,
        nvfp4_key,
        |entry| {
            let (named, data) = parse(entry, 3)?;
            Some(Nvfp4Algorithm {
                m: named[0],
                n: named[1],
                k: named[2],
                data,
            })
        },
        // SAFETY: the slice holds `algorithms.len()` algorithms.
        |device, algorithms| unsafe {
            mmh3_cublaslt_nvfp4_adopt(
                device as c_int,
                algorithms.as_ptr(),
                algorithms.len() as c_int,
            )
        },
    )
}

/// Writes the algorithms this process chose for NVFP4 GEMM shapes to `path` for `load_nvfp4`, and
/// returns whether it wrote.
pub fn save_nvfp4(path: &Path) -> io::Result<bool> {
    save(
        path,
        NVFP4_HEADER,
        nvfp4_key,
        |device| {
            chosen(|algorithms, capacity| unsafe {
                mmh3_cublaslt_nvfp4_algorithms(device as c_int, algorithms, capacity)
            })
        },
        |algorithm: &Nvfp4Algorithm| {
            line(&[algorithm.m, algorithm.n, algorithm.k], &algorithm.data)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{BTreeMap, merged, sections};

    const HEADER: &str = "mmh3 cuBLASLt matmul algorithms";

    fn section(key: &str, entries: &[&str]) -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([(
            key.to_owned(),
            entries.iter().map(|entry| (*entry).to_owned()).collect(),
        )])
    }

    /// Saving the choices of one kind of GPU keeps what another kind saved before, so that a
    /// machine with both finds each on the next run.
    #[test]
    fn a_save_keeps_the_sections_of_other_gpus() {
        let first = merged(
            "",
            HEADER,
            section("cuBLASLt 1, Some GPU", &["0 0 1 2 3 a"]),
        );
        let both = merged(
            &first,
            HEADER,
            section("cuBLASLt 1, Another GPU", &["0 0 4 5 6 b"]),
        );
        let read = sections(&both, HEADER);
        assert_eq!(read["cuBLASLt 1, Some GPU"], ["0 0 1 2 3 a"]);
        assert_eq!(read["cuBLASLt 1, Another GPU"], ["0 0 4 5 6 b"]);

        let again = merged(
            &both,
            HEADER,
            section("cuBLASLt 1, Some GPU", &["0 0 7 8 9 c"]),
        );
        let read = sections(&again, HEADER);
        assert_eq!(read["cuBLASLt 1, Some GPU"], ["0 0 7 8 9 c"]);
        assert_eq!(read["cuBLASLt 1, Another GPU"], ["0 0 4 5 6 b"]);
    }

    /// A file of one section, which is all a file held before there were sections, reads as that
    /// section, and one under another header reads as nothing.
    #[test]
    fn reads_a_file_of_one_section() {
        let text = format!("{HEADER}\ncuBLASLt 1, Some GPU\n0 0 1 2 3 a\n");
        assert_eq!(
            sections(&text, HEADER)["cuBLASLt 1, Some GPU"],
            ["0 0 1 2 3 a"]
        );
        assert!(sections(&text, "mmh3 cuBLASLt NVFP4 algorithms").is_empty());
    }
}
