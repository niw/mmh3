//! The cuBLASLt algorithms chosen for the GEMM shapes of a run, the files that keep them between
//! runs, and the consistency that chooses none by measuring.
//!
//! A choice comes from timing the heuristic's candidates on the shape's own operands, which costs
//! about a second on the first step of a run. A file taken over at startup spares it: a run that
//! has met its shapes before times none of them.

use crate::{CudaError, check};
use std::ffi::{CStr, c_char, c_int};
use std::path::Path;
use std::{fs, io, ptr};

unsafe extern "C" {
    fn mmh3_cublaslt_matmul_algorithms(algorithms: *mut Algorithm, capacity: c_int) -> c_int;
    fn mmh3_cublaslt_matmul_adopt(algorithms: *const Algorithm, count: c_int);
    fn mmh3_cublaslt_matmul_algorithm_key(key: *mut c_char, capacity: c_int) -> c_int;
    fn mmh3_cublaslt_nvfp4_algorithms(algorithms: *mut Nvfp4Algorithm, capacity: c_int) -> c_int;
    fn mmh3_cublaslt_nvfp4_adopt(algorithms: *const Nvfp4Algorithm, count: c_int);
    fn mmh3_cublaslt_nvfp4_algorithm_key(key: *mut c_char, capacity: c_int) -> c_int;
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

/// First line of a file of chosen algorithms, followed by the line its key writes and a line per
/// shape: the fields that name the shape and the eight words of the algorithm in hexadecimal.
const MATMUL_HEADER: &str = "mmh3 cuBLASLt matmul algorithms";
const NVFP4_HEADER: &str = "mmh3 cuBLASLt NVFP4 algorithms";

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

/// The entry lines of `path`, and none from a missing or damaged file or from one written with
/// another cuBLASLt version, GPU, descriptors or rule.
fn read(path: &Path, header: &str, key: &str) -> io::Result<Vec<String>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut lines = text.lines();
    if lines.next() != Some(header) || lines.next() != Some(key) {
        return Ok(Vec::new());
    }
    Ok(lines.map(str::to_owned).collect())
}

/// Writes the entries to `path` and returns whether it wrote. It writes nothing when the file
/// already holds them.
fn write(path: &Path, header: &str, key: &str, entries: &str) -> io::Result<bool> {
    let text = format!("{header}\n{key}\n{entries}");
    if fs::read_to_string(path).is_ok_and(|existing| existing == text) {
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

fn chosen<T: Copy + Default>(of: impl Fn(*mut T, c_int) -> c_int) -> Vec<T> {
    // SAFETY of both calls: a capacity of 0 writes nothing, and the vector holds `count` entries.
    let count = of(ptr::null_mut(), 0) as usize;
    let mut algorithms = vec![T::default(); count];
    let total = of(algorithms.as_mut_ptr(), count as c_int);
    algorithms.truncate(total as usize);
    algorithms
}

/// What the algorithms chosen here depend on: a file is taken over only when it names the same
/// thing, which one written on another GPU or with another cuBLASLt does not.
fn key() -> Result<String, CudaError> {
    key_of(|key, capacity| unsafe { mmh3_cublaslt_matmul_algorithm_key(key, capacity) })
}

/// The algorithms this process has chosen for ordinary GEMM shapes.
fn chosen_matmul() -> Vec<Algorithm> {
    chosen(|algorithms, capacity| unsafe { mmh3_cublaslt_matmul_algorithms(algorithms, capacity) })
}

/// Takes over the algorithms that `save_matmul` left in `path` for ordinary GEMM shapes and
/// returns how many it took. A shape that has one chosen here already keeps it.
pub fn load_matmul(path: &Path) -> io::Result<usize> {
    let key = key().map_err(io::Error::other)?;
    let Some(algorithms) = read(path, MATMUL_HEADER, &key)?
        .iter()
        .map(|entry| {
            let (named, data) = parse(entry, 5)?;
            Some(Algorithm {
                kind: named[0],
                bias: named[1],
                m: named[2],
                n: named[3],
                k: named[4],
                data,
            })
        })
        .collect::<Option<Vec<Algorithm>>>()
    else {
        return Ok(0);
    };
    // SAFETY: the slice holds `algorithms.len()` algorithms.
    unsafe { mmh3_cublaslt_matmul_adopt(algorithms.as_ptr(), algorithms.len() as c_int) };
    Ok(algorithms.len())
}

/// Writes the algorithms this process chose for ordinary GEMM shapes to `path` for `load_matmul`,
/// and returns whether it wrote.
pub fn save_matmul(path: &Path) -> io::Result<bool> {
    let algorithms = chosen_matmul();
    if algorithms.is_empty() {
        return Ok(false);
    }
    let key = key().map_err(io::Error::other)?;
    let mut entries = String::new();
    for algorithm in &algorithms {
        entries.push_str(&line(
            &[
                algorithm.kind,
                algorithm.bias,
                algorithm.m,
                algorithm.n,
                algorithm.k,
            ],
            &algorithm.data,
        ));
    }
    write(path, MATMUL_HEADER, &key, &entries)
}

/// Takes over the algorithms that `save_nvfp4` left in `path` for NVFP4 GEMM shapes and returns
/// how many it took.
pub fn load_nvfp4(path: &Path) -> io::Result<usize> {
    let key = key_of(|key, capacity| unsafe { mmh3_cublaslt_nvfp4_algorithm_key(key, capacity) })
        .map_err(io::Error::other)?;
    let Some(algorithms) = read(path, NVFP4_HEADER, &key)?
        .iter()
        .map(|entry| {
            let (named, data) = parse(entry, 3)?;
            Some(Nvfp4Algorithm {
                m: named[0],
                n: named[1],
                k: named[2],
                data,
            })
        })
        .collect::<Option<Vec<Nvfp4Algorithm>>>()
    else {
        return Ok(0);
    };
    // SAFETY: the slice holds `algorithms.len()` algorithms.
    unsafe { mmh3_cublaslt_nvfp4_adopt(algorithms.as_ptr(), algorithms.len() as c_int) };
    Ok(algorithms.len())
}

/// Writes the algorithms this process chose for NVFP4 GEMM shapes to `path` for `load_nvfp4`, and
/// returns whether it wrote.
pub fn save_nvfp4(path: &Path) -> io::Result<bool> {
    let algorithms = chosen(|algorithms, capacity| unsafe {
        mmh3_cublaslt_nvfp4_algorithms(algorithms, capacity)
    });
    if algorithms.is_empty() {
        return Ok(false);
    }
    let key = key_of(|key, capacity| unsafe { mmh3_cublaslt_nvfp4_algorithm_key(key, capacity) })
        .map_err(io::Error::other)?;
    let mut entries = String::new();
    for algorithm in &algorithms {
        entries.push_str(&line(
            &[algorithm.m, algorithm.n, algorithm.k],
            &algorithm.data,
        ));
    }
    write(path, NVFP4_HEADER, &key, &entries)
}
