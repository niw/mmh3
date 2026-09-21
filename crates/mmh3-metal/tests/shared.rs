#![cfg(target_os = "macos")]
//! The device the whole process computes on, which is its own test binary so that the threads
//! here are the only ones asking for it.
use mmh3_metal::{Device, ops::Array};

const THREADS: usize = 4;
const ROUNDS: usize = 20;

/// Every thread that asks gets the one device, and what each of them computes on it is its own.
#[test]
fn threads_compute_on_one_device() {
    let before = Device::shared().unwrap().stats();
    std::thread::scope(|scope| {
        for thread in 0..THREADS {
            scope.spawn(move || round_trips(thread as f32));
        }
    });

    // Buffers of the process's own device, which is the device every thread took its own from.
    let after = Device::shared().unwrap().stats();
    let buffers = (after.buffer_allocations + after.buffer_reuses)
        - (before.buffer_allocations + before.buffer_reuses);
    assert!(
        buffers >= (THREADS * ROUNDS * 2) as u64,
        "{buffers} buffers came from one device"
    );
}

fn round_trips(salt: f32) {
    let device = Device::shared().unwrap();
    let rows = 32;
    let cols = 64;
    let values: Vec<f32> = (0..rows * cols).map(|i| i as f32 + salt).collect();
    for _ in 0..ROUNDS {
        let held = Array::from_f32(&device, rows, cols, &values).unwrap();
        assert_eq!(held.to_f32().unwrap(), values);
        let zeros = Array::zeros(&device, rows, cols).unwrap();
        assert!(zeros.to_f32().unwrap().iter().all(|&value| value == 0.0));
    }
}
