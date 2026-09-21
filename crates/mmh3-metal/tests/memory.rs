#![cfg(target_os = "macos")]
//! What the device says it has left, which is its own test binary so that what this process takes
//! is what this file took.
use mmh3_metal::{Device, ops::Array};

/// The device counts what a context takes against the process, so what it leaves shrinks by the
/// memory a model would hold.
#[test]
fn taking_memory_leaves_less_of_it() {
    let device = Device::new().unwrap();
    let (before, recommended) = mmh3_metal::memory_info().unwrap();
    assert!(
        before > 0 && before <= recommended,
        "{before}/{recommended}"
    );

    let rows = 1024;
    let cols = 1024;
    let held = Array::zeros(&device, rows, cols).unwrap();
    let (after, _) = mmh3_metal::memory_info().unwrap();
    assert!(
        before - after >= rows * cols * 4,
        "an array of {} bytes left {before} and then {after}",
        rows * cols * 4
    );

    drop(held);
}
