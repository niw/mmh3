#![cfg(target_os = "linux")]
use mmh3_cuda::DeviceBuffer;

#[test]
fn fill_kernel_writes_every_element() {
    let count = 100_003;
    let mut buffer = DeviceBuffer::new(count * 4).unwrap();
    buffer.fill_f32(3.5).unwrap();
    let mut bytes = vec![0; count * 4];
    buffer.copy_to_host(&mut bytes).unwrap();
    assert!(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&chunk| f32::from_le_bytes(chunk) == 3.5)
    );
}

#[test]
fn reports_a_device() {
    assert!(mmh3_cuda::device_count().unwrap() > 0);
    let info = mmh3_cuda::device_info(0).unwrap();
    assert!(!info.name.is_empty());
    assert!(info.compute_capability.0 >= 12);
}

/// The device belongs to the host thread, and a buffer remembers the one it was made on so that it
/// is freed there whatever the thread has moved to since.
#[test]
fn a_buffer_says_which_device_it_is_on() {
    let device = mmh3_cuda::current_device().unwrap();
    mmh3_cuda::set_device(device).unwrap();
    assert_eq!(mmh3_cuda::current_device().unwrap(), device);

    let buffer = DeviceBuffer::zeroed(64).unwrap();
    assert_eq!(buffer.device(), device);
}

/// A device this build cannot index is refused rather than counted against another one, since the
/// allocation counters and the kernels' per-device answers are indexed by it.
#[test]
fn refuses_a_device_past_the_ones_it_indexes() {
    let error = mmh3_cuda::set_device(mmh3_cuda::MAX_DEVICES).unwrap_err();
    assert_eq!(error.code, mmh3_cuda::INVALID_DEVICE);
}

/// Every device the driver reports can be computed on, which is what binding one has to mean.
#[test]
fn binds_every_device_the_driver_reports() {
    let count = mmh3_cuda::device_count().unwrap();
    for device in 0..count.min(mmh3_cuda::MAX_DEVICES) {
        mmh3_cuda::set_device(device).unwrap();
        assert_eq!(mmh3_cuda::current_device().unwrap(), device);
        assert_eq!(DeviceBuffer::zeroed(64).unwrap().device(), device);
    }
    mmh3_cuda::set_device(0).unwrap();
}
