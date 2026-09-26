#![cfg(target_os = "macos")]
//! A limit on what this process holds, which is its own test binary so that a limit set here is
//! nobody else's.
use mmh3_metal::{Device, ops::Array};
use std::sync::Barrier;

/// An allocation past the limit fails as it would on a device that small, and the memory the
/// process was holding is its own again once it lets go.
#[test]
fn an_allocation_past_the_limit_finds_no_memory() {
    let device = Device::new().unwrap();
    let rows = 64;
    let cols = 64;
    let bytes = rows * cols * 4;

    let held = Array::zeros(&device, rows, cols).unwrap();
    assert_eq!(mmh3_metal::allocated_bytes(), bytes);

    // Room for what is held and nothing more.
    mmh3_metal::set_allocation_limit(bytes);
    let Err(error) = Array::zeros(&device, rows, cols) else {
        panic!("a limit gave away memory it had none of");
    };
    assert!(error.is_out_of_memory(), "{error}");

    drop(held);
    assert_eq!(mmh3_metal::allocated_bytes(), 0);
    Array::zeros(&device, rows, cols).unwrap();

    mmh3_metal::set_allocation_limit(0);

    // Keep this in the same test: the limit and the allocation count belong to the process.
    concurrent_allocations_respect_the_limit(&device);
}

fn concurrent_allocations_respect_the_limit(device: &Device) {
    let budget = 1024 * 1024;
    mmh3_metal::set_allocation_limit(budget);
    let barrier = Barrier::new(16);
    let held = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..16)
            .map(|_| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    Array::zeros(device, 1, budget / 4)
                })
            })
            .collect();
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });

    let successful = held.iter().filter(|result| result.is_ok()).count();
    assert_eq!(successful, 1, "only one allocation fits in the budget");
    assert_eq!(mmh3_metal::allocated_bytes(), budget);
    for error in held.iter().filter_map(|result| result.as_ref().err()) {
        assert!(error.is_out_of_memory(), "{error}");
    }
    drop(held);
    assert_eq!(mmh3_metal::allocated_bytes(), 0);
    // Refused allocations must not leave reservations behind.
    Array::zeros(device, 1, budget / 4).unwrap();
    mmh3_metal::set_allocation_limit(0);
}
