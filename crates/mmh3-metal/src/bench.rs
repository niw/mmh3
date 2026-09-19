//! What this machine does at the shapes a DiT block works in, which is what a leader cuts a
//! shared-out step by.
//!
//! The number is operations a second at a block's own product, not a rating of the machine: the
//! same arithmetic on a different road is a different number, so a measurement carries the
//! precision it was taken at.

use crate::{Device, Error, LinearPrecision, Result, ops::Array};
use std::time::Instant;

/// Operations a second at an `m × k` by `k × n` product with INT8 weights, the shape and the road
/// a block's own linear layers take. The first trial pays for the pipelines and is dropped, and
/// the median of the rest is taken.
pub fn gemm_operations_per_second(
    device: &Device,
    precision: LinearPrecision,
    m: usize,
    n: usize,
    k: usize,
    trials: usize,
) -> Result<f64> {
    let values: Vec<f32> = (0..m * k).map(|i| (i % 101) as f32 / 100.0 - 0.5).collect();
    let input = Array::from_f32(device, m, k, &values)?;
    let bytes: Vec<u8> = (0..n * k).map(|i| (i % 251) as u8).collect();
    let weight = device.alloc(bytes.len(), Some(&bytes))?;

    let seconds = median(
        trials,
        || {
            let product = input.linear_int8(&weight, n, precision)?;
            std::hint::black_box(&product);
            Ok(())
        },
        device,
    )?;
    Ok(2.0 * m as f64 * n as f64 * k as f64 / seconds)
}

/// Bytes a second moving `bytes` from one buffer to another, counting the read and the write as
/// the CUDA copy does, so the two backends' numbers mean the same thing.
pub fn memory_copy_bytes_per_second(device: &Device, bytes: usize, trials: usize) -> Result<f64> {
    let values = bytes / 4;
    if values == 0 {
        return Err(Error("a copy of no bytes".into()));
    }

    let source = Array::zeros(device, 1, values)?;
    let destination = Array::empty(device, 1, values)?;
    let seconds = median(
        trials,
        || {
            device.run(
                "copy_rows",
                &[&source.buffer, &destination.buffer],
                &[values as u32, values as u32, values as u32, 0, 0],
                values,
                false,
            )
        },
        device,
    )?;
    Ok(2.0 * (values * 4) as f64 / seconds)
}

/// Runs `work` `trials` times and answers with the median of all but the first, which pays for
/// whatever the device sets up on its way to the first call.
fn median(trials: usize, mut work: impl FnMut() -> Result<()>, device: &Device) -> Result<f64> {
    let mut times = Vec::new();
    for trial in 0..trials.max(2) {
        device.synchronize()?;
        let started = Instant::now();
        work()?;
        device.synchronize()?;
        if trial > 0 {
            times.push(started.elapsed().as_secs_f64());
        }
    }

    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    if !(median > 0.0) {
        return Err(Error("a measurement that took no time".into()));
    }
    Ok(median)
}
