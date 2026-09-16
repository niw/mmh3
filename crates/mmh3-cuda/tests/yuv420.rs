#![cfg(target_os = "linux")]
use mmh3_core::media::Yuv420;
use mmh3_core::tensor::Tensor;
use mmh3_cuda::DeviceBuffer;
use mmh3_cuda::vae;

#[test]
fn converts_like_the_cpu() {
    let (frames, height, width) = (3, 6, 10);
    let mut state = 5u64;
    let data: Vec<f32> = (0..3 * frames * height * width)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect();
    let pixels = Tensor::new(vec![3, frames, height, width], data);
    let buffer = DeviceBuffer::from_f32(&pixels.data).unwrap();
    assert_eq!(
        vae::yuv420(&buffer, frames, height, width).unwrap(),
        Yuv420::from_pixels(&pixels).unwrap()
    );
}
