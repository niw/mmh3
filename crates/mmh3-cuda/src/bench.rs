//! Micro-benchmarks that establish hardware baselines at H3 shapes.

use crate::CudaError;
use std::ffi::{CStr, c_char, c_int};

unsafe extern "C" {
    fn mmh3_bench_cublaslt_gemm(
        kind: c_int,
        m: i64,
        n: i64,
        k: i64,
        iterations: c_int,
        best_milliseconds: *mut f32,
        algorithm_count: *mut c_int,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn mmh3_bench_int8_gemm(
        m: i64,
        n: i64,
        k: i64,
        iterations: c_int,
        best_milliseconds: *mut f32,
        configs_timed: *mut c_int,
        best_config: *mut c_int,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn mmh3_bench_mma_peak(
        kind: c_int,
        iterations: c_int,
        tera_operations_per_second: *mut f32,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn mmh3_bench_attention(
        tokens: c_int,
        heads: c_int,
        iterations: c_int,
        milliseconds: *mut f32,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
    fn mmh3_bench_memory_copy(
        bytes: usize,
        iterations: c_int,
        kernel_gigabytes_per_second: *mut f32,
        memcpy_gigabytes_per_second: *mut f32,
        message: *mut c_char,
        message_size: usize,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKind {
    Bf16,
    Fp8E4m3,
    Int8,
    Nvfp4,
    /// The mmh3 INT8 kernel with per-row and per-column scales and BF16 output.
    Int8Mmh3,
}

impl GemmKind {
    pub const ALL: [GemmKind; 5] =
        [GemmKind::Bf16, GemmKind::Fp8E4m3, GemmKind::Int8, GemmKind::Nvfp4, GemmKind::Int8Mmh3];

    pub fn name(self) -> &'static str {
        match self {
            GemmKind::Bf16 => "bf16",
            GemmKind::Fp8E4m3 => "fp8",
            GemmKind::Int8 => "int8",
            GemmKind::Nvfp4 => "nvfp4",
            GemmKind::Int8Mmh3 => "int8-mmh3",
        }
    }

    pub fn from_name(name: &str) -> Option<GemmKind> {
        GemmKind::ALL.into_iter().find(|kind| kind.name() == name)
    }

    fn cublaslt_code(self) -> Option<c_int> {
        match self {
            GemmKind::Bf16 => Some(0),
            GemmKind::Fp8E4m3 => Some(1),
            GemmKind::Int8 => Some(2),
            GemmKind::Nvfp4 => Some(3),
            GemmKind::Int8Mmh3 => None,
        }
    }
}

/// A tensor core instruction form, named by operand and accumulator types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaKind {
    Bf16AccumulateF32,
    F16AccumulateF16,
    Int8AccumulateS32,
    Fp8AccumulateF32,
    Fp8AccumulateF16,
}

impl MmaKind {
    pub const ALL: [MmaKind; 5] = [
        MmaKind::Bf16AccumulateF32,
        MmaKind::F16AccumulateF16,
        MmaKind::Int8AccumulateS32,
        MmaKind::Fp8AccumulateF32,
        MmaKind::Fp8AccumulateF16,
    ];

    pub fn instruction(self) -> &'static str {
        match self {
            MmaKind::Bf16AccumulateF32 => "m16n8k16 bf16 → f32",
            MmaKind::F16AccumulateF16 => "m16n8k16 f16 → f16",
            MmaKind::Int8AccumulateS32 => "m16n8k32 s8 → s32",
            MmaKind::Fp8AccumulateF32 => "m16n8k32 e4m3 → f32",
            MmaKind::Fp8AccumulateF16 => "m16n8k32 e4m3 → f16",
        }
    }

    fn code(self) -> c_int {
        self as c_int
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GemmTiming {
    pub milliseconds: f32,
    /// cuBLASLt heuristic algorithms or mmh3 tile configurations that ran.
    pub candidates_timed: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryBandwidth {
    pub copy_kernel_gigabytes_per_second: f32,
    pub memcpy_gigabytes_per_second: f32,
}

fn failure(code: c_int, message: &[c_char]) -> CudaError {
    // SAFETY: the launcher writes a NUL-terminated message into the zeroed buffer.
    let text = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy().into_owned();
    CudaError { code, message: text }
}

/// Times y[m, n] = x[m, k] · w[n, k]ᵀ and reports the fastest candidate for the kind.
pub fn gemm(kind: GemmKind, m: usize, n: usize, k: usize, iterations: usize) -> Result<GemmTiming, CudaError> {
    let mut milliseconds = 0.0;
    let mut candidates_timed = 0;
    let mut message = [0 as c_char; 256];
    // SAFETY: all out pointers are valid and the message buffer length is passed along.
    let code = unsafe {
        match kind.cublaslt_code() {
            Some(code) => mmh3_bench_cublaslt_gemm(
                code,
                m as i64,
                n as i64,
                k as i64,
                iterations as c_int,
                &mut milliseconds,
                &mut candidates_timed,
                message.as_mut_ptr(),
                message.len(),
            ),
            None => {
                let mut best_config = 0;
                mmh3_bench_int8_gemm(
                    m as i64,
                    n as i64,
                    k as i64,
                    iterations as c_int,
                    &mut milliseconds,
                    &mut candidates_timed,
                    &mut best_config,
                    message.as_mut_ptr(),
                    message.len(),
                )
            }
        }
    };
    if code != 0 {
        return Err(failure(code, &message));
    }
    Ok(GemmTiming { milliseconds, candidates_timed })
}

/// Tensor core throughput of one instruction form in tera-operations per second.
pub fn mma_peak(kind: MmaKind, iterations: usize) -> Result<f32, CudaError> {
    let mut throughput = 0.0;
    let mut message = [0 as c_char; 256];
    // SAFETY: the out pointer is valid and the message buffer length is passed along.
    let code = unsafe {
        mmh3_bench_mma_peak(kind.code(), iterations as c_int, &mut throughput, message.as_mut_ptr(), message.len())
    };
    if code != 0 {
        return Err(failure(code, &message));
    }
    Ok(throughput)
}

/// Milliseconds per dense attention call over `tokens` tokens and `heads` heads of dimension 128.
pub fn attention(tokens: usize, heads: usize, iterations: usize) -> Result<f32, CudaError> {
    let mut milliseconds = 0.0;
    let mut message = [0 as c_char; 256];
    // SAFETY: the out pointer is valid and the message buffer length is passed along.
    let code = unsafe {
        mmh3_bench_attention(
            tokens as c_int,
            heads as c_int,
            iterations as c_int,
            &mut milliseconds,
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if code != 0 {
        return Err(failure(code, &message));
    }
    Ok(milliseconds)
}

pub fn memory_copy(bytes: usize, iterations: usize) -> Result<MemoryBandwidth, CudaError> {
    let mut copy_kernel = 0.0;
    let mut memcpy = 0.0;
    let mut message = [0 as c_char; 256];
    // SAFETY: all out pointers are valid and the message buffer length is passed along.
    let code = unsafe {
        mmh3_bench_memory_copy(
            bytes,
            iterations as c_int,
            &mut copy_kernel,
            &mut memcpy,
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if code != 0 {
        return Err(failure(code, &message));
    }
    Ok(MemoryBandwidth { copy_kernel_gigabytes_per_second: copy_kernel, memcpy_gigabytes_per_second: memcpy })
}
