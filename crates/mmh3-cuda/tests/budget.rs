#![cfg(target_os = "linux")]
//! A limit on what this process holds of a device, which is its own test binary so that a limit
//! set here is nobody else's.
use mmh3_cuda::DeviceBuffer;

/// An allocation past the limit fails as it would on a device that small, and the memory the
/// process was holding is its own again once it lets go. The counters are per device, so this
/// answers for the one this thread computes on.
#[test]
fn an_allocation_past_the_limit_finds_no_memory() {
    let bytes = 64 * 64 * 4;

    let held = DeviceBuffer::zeroed(bytes).unwrap();
    assert_eq!(held.device(), mmh3_cuda::current_device().unwrap());
    assert_eq!(mmh3_cuda::allocated_bytes(), bytes);

    // Room for what is held and nothing more.
    mmh3_cuda::set_allocation_limit(bytes);
    let Err(error) = DeviceBuffer::zeroed(bytes) else {
        panic!("a limit gave away memory it had none of");
    };
    assert!(error.is_out_of_memory(), "{error}");

    drop(held);
    assert_eq!(mmh3_cuda::allocated_bytes(), 0);
    DeviceBuffer::zeroed(bytes).unwrap();

    mmh3_cuda::set_allocation_limit(0);
}
