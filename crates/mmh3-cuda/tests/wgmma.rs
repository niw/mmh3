#![cfg(target_os = "linux")]
//! The first rung of checking the Hopper kernels: one warpgroup's wgmma over a tile in the TMA 128B
//! swizzle, against the host, with both operands in shared memory, with A in registers and B
//! MN-major, and in FP8 with B in rows of 64 bytes. It runs on an sm_90 card and checks that every
//! other card refuses it.

use mmh3_core::numeric::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
use mmh3_cuda::DeviceBuffer;
use std::ffi::{c_int, c_void};
use std::ptr;

unsafe extern "C" {
    fn mmh3_wgmma_check(
        bf16: c_int,
        n: c_int,
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_wgmma_check_fp8(
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_wgmma_check_transposed(
        f16: c_int,
        n: c_int,
        a: *const c_void,
        v: *const c_void,
        d: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
}

/// cudaErrorNotSupported.
const NOT_SUPPORTED: c_int = 801;
const A_ROWS: usize = 64;
const ROW_BYTES: usize = 128;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn int8(&mut self) -> i8 {
        ((self.next() % 255) as i32 - 127) as i8
    }

    fn uniform(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 4.0 - 2.0
    }

    fn bf16(&mut self) -> u16 {
        f32_to_bf16(self.uniform())
    }
}

fn is_hopper() -> bool {
    let device = mmh3_cuda::current_device().unwrap();
    mmh3_cuda::device_info(device).unwrap().compute_capability == (9, 0)
}

/// Runs the check for `n` columns and returns D as its 32-bit words.
fn run(bf16: bool, n: usize, a: &[u8], b: &[u8]) -> Result<Vec<u32>, c_int> {
    let (a_buffer, b_buffer) = (
        DeviceBuffer::from_bytes(a).unwrap(),
        DeviceBuffer::from_bytes(b).unwrap(),
    );
    let d_buffer = DeviceBuffer::new(A_ROWS * n * 4).unwrap();
    // SAFETY: A holds 64 rows and B `n` rows of 128 bytes, and D 64 × n words.
    let status = unsafe {
        mmh3_wgmma_check(
            c_int::from(bf16),
            n as c_int,
            a_buffer.pointer(),
            b_buffer.pointer(),
            d_buffer.pointer(),
            ptr::null_mut(),
        )
    };
    if status != 0 {
        return Err(status);
    }
    let mut bytes = vec![0; A_ROWS * n * 4];
    d_buffer.copy_to_host(&mut bytes).unwrap();
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect())
}

#[test]
fn only_hopper_runs_it() {
    if is_hopper() {
        return;
    }
    let operand = vec![0; (A_ROWS + 256) * ROW_BYTES];
    assert_eq!(
        run(false, 256, &operand, &operand),
        Err(NOT_SUPPORTED),
        "a card without wgmma must refuse it"
    );
}

#[test]
fn multiplies_int8_tiles() {
    if !is_hopper() {
        eprintln!("wgmma needs an sm_90 card, skipping");
        return;
    }
    for n in [128, 256] {
        let mut random = Random(n as u64);
        let a: Vec<i8> = (0..A_ROWS * ROW_BYTES).map(|_| random.int8()).collect();
        let b: Vec<i8> = (0..n * ROW_BYTES).map(|_| random.int8()).collect();
        let d = run(
            false,
            n,
            &a.iter().map(|&value| value as u8).collect::<Vec<_>>(),
            &b.iter().map(|&value| value as u8).collect::<Vec<_>>(),
        )
        .unwrap();
        for row in 0..A_ROWS {
            for column in 0..n {
                let expected: i32 = (0..ROW_BYTES)
                    .map(|k| a[row * ROW_BYTES + k] as i32 * b[column * ROW_BYTES + k] as i32)
                    .sum();
                assert_eq!(
                    d[row * n + column] as i32,
                    expected,
                    "n {n}, row {row}, column {column}"
                );
            }
        }
    }
}

#[test]
fn multiplies_bf16_tiles() {
    if !is_hopper() {
        eprintln!("wgmma needs an sm_90 card, skipping");
        return;
    }
    let k = ROW_BYTES / 2;
    for n in [128, 256] {
        let mut random = Random(n as u64 + 7);
        let a: Vec<u16> = (0..A_ROWS * k).map(|_| random.bf16()).collect();
        let b: Vec<u16> = (0..n * k).map(|_| random.bf16()).collect();
        let bytes = |values: &[u16]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let d = run(true, n, &bytes(&a), &bytes(&b)).unwrap();
        for row in 0..A_ROWS {
            for column in 0..n {
                let expected: f64 = (0..k)
                    .map(|index| {
                        bf16_to_f32(a[row * k + index]) as f64
                            * bf16_to_f32(b[column * k + index]) as f64
                    })
                    .sum();
                let actual = f32::from_bits(d[row * n + column]) as f64;
                assert!(
                    (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "n {n}, row {row}, column {column}: expected {expected}, got {actual}"
                );
            }
        }
    }
}

#[test]
fn multiplies_registers_by_transposed_tiles() {
    if !is_hopper() {
        eprintln!("wgmma needs an sm_90 card, skipping");
        return;
    }
    let keys = 64;
    for f16 in [false, true] {
        for n in [64, 128] {
            let mut random = Random(n as u64 + u64::from(f16) * 3);
            let mut values = |count: usize| -> Vec<u16> {
                (0..count)
                    .map(|_| {
                        let value = random.uniform();
                        if f16 {
                            f32_to_f16(value)
                        } else {
                            f32_to_bf16(value)
                        }
                    })
                    .collect()
            };
            let (a, v) = (values(A_ROWS * keys), values(keys * n));
            let to_f32 = |bits: u16| {
                if f16 {
                    f16_to_f32(bits)
                } else {
                    bf16_to_f32(bits)
                }
            };
            let bytes = |values: &[u16]| -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
            let (a_buffer, v_buffer) = (
                DeviceBuffer::from_bytes(&bytes(&a)).unwrap(),
                DeviceBuffer::from_bytes(&bytes(&v)).unwrap(),
            );
            let d_buffer = DeviceBuffer::new(A_ROWS * n * 4).unwrap();
            // SAFETY: A holds 64 × 64 values, V 64 × n and D 64 × n words.
            let status = unsafe {
                mmh3_wgmma_check_transposed(
                    c_int::from(f16),
                    n as c_int,
                    a_buffer.pointer(),
                    v_buffer.pointer(),
                    d_buffer.pointer(),
                    ptr::null_mut(),
                )
            };
            assert_eq!(status, 0);
            let mut d = vec![0; A_ROWS * n * 4];
            d_buffer.copy_to_host(&mut d).unwrap();
            let d: Vec<f32> = d
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&word| f32::from_le_bytes(word))
                .collect();
            for row in 0..A_ROWS {
                for column in 0..n {
                    let expected: f64 = (0..keys)
                        .map(|key| {
                            to_f32(a[row * keys + key]) as f64 * to_f32(v[key * n + column]) as f64
                        })
                        .sum();
                    let actual = d[row * n + column] as f64;
                    assert!(
                        (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                        "f16 {f16}, n {n}, row {row}, column {column}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }
}

/// An FP8 E4M3 value, which has no infinities and only 0x7f and 0xff for NaN.
fn e4m3_to_f64(bits: u8) -> f64 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 3) & 0xf);
    let mantissa = f64::from(bits & 0x7);
    sign * if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    }
}

#[test]
fn multiplies_fp8_tiles() {
    if !is_hopper() {
        eprintln!("wgmma needs an sm_90 card, skipping");
        return;
    }
    let (k, n) = (64, 128);
    let mut random = Random(9);
    // Small magnitudes, clear of NaN, so that FP32 sums them without rounding.
    let mut byte = || (random.next() as u8) & 0xb7;
    let a: Vec<u8> = (0..A_ROWS * k).map(|_| byte()).collect();
    let b: Vec<u8> = (0..n * k).map(|_| byte()).collect();
    let (a_buffer, b_buffer) = (
        DeviceBuffer::from_bytes(&a).unwrap(),
        DeviceBuffer::from_bytes(&b).unwrap(),
    );
    let d_buffer = DeviceBuffer::new(A_ROWS * n * 4).unwrap();
    // SAFETY: A holds 64 × 64 bytes, B 128 × 64 and D 64 × 128 words.
    let status = unsafe {
        mmh3_wgmma_check_fp8(
            a_buffer.pointer(),
            b_buffer.pointer(),
            d_buffer.pointer(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, 0);
    let mut d = vec![0; A_ROWS * n * 4];
    d_buffer.copy_to_host(&mut d).unwrap();
    for row in 0..A_ROWS {
        for column in 0..n {
            let expected: f64 = (0..k)
                .map(|index| e4m3_to_f64(a[row * k + index]) * e4m3_to_f64(b[column * k + index]))
                .sum();
            let offset = (row * n + column) * 4;
            let actual = f32::from_le_bytes(d[offset..offset + 4].try_into().unwrap()) as f64;
            assert!(
                (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                "row {row}, column {column}: expected {expected}, got {actual}"
            );
        }
    }
}
