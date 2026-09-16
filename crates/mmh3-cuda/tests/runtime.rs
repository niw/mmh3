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
