#![cfg(target_os = "linux")]
use mmh3_core::{media::Yuv420, tensor::Tensor};
use mmh3_cuda::{DeviceBuffer, vae::CudaVideoFrames};

#[test]
fn device_nv12_matches_reference_with_pitched_rows_and_guard_bytes() {
    let (frames, height, width, pitch) = (3, 50, 66, 128);
    let pixels: Vec<f32> = (0..3 * frames * height * width)
        .map(|i| ((i * 37 + i / 11) % 1009) as f32 / 1008.0)
        .collect();
    let reference =
        Yuv420::from_pixels(&Tensor::new(vec![3, frames, height, width], pixels.clone())).unwrap();
    let video = CudaVideoFrames::from_rgb(
        DeviceBuffer::from_f32(&pixels).unwrap(),
        frames,
        height,
        width,
    )
    .unwrap();
    let size = pitch * height * 3 / 2;
    let mut output = DeviceBuffer::from_bytes(&vec![0xa5; size + 64]).unwrap();
    for frame in 0..frames {
        video.write_nv12_frame(frame, &mut output, pitch).unwrap();
        let mut bytes = vec![0; size + 64];
        output.copy_to_host(&mut bytes).unwrap();
        let expected = &reference.data[frame * width * height * 3 / 2..];
        for row in 0..height {
            assert_eq!(
                &bytes[row * pitch..row * pitch + width],
                &expected[row * width..(row + 1) * width]
            );
            assert!(
                bytes[row * pitch + width..(row + 1) * pitch]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                assert_eq!(
                    bytes[(height + row) * pitch + 2 * col],
                    expected[width * height + row * width / 2 + col]
                );
                assert_eq!(
                    bytes[(height + row) * pitch + 2 * col + 1],
                    expected[width * height * 5 / 4 + row * width / 2 + col]
                );
            }
            assert!(
                bytes[(height + row) * pitch + width..(height + row + 1) * pitch]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
        }
        assert!(bytes[size..].iter().all(|&byte| byte == 0xa5));
    }
    assert!(video.write_nv12_frame(frames, &mut output, pitch).is_err());
    assert!(video.write_nv12_frame(0, &mut output, width - 1).is_err());
    let mut small = DeviceBuffer::new(size - 1).unwrap();
    assert!(video.write_nv12_frame(0, &mut small, pitch).is_err());
}
