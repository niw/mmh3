//! Repeatable, model-free timings. Run with `cargo run -p mmh3-metal --release --example profile`.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use mmh3_metal::{Device, ops::Array};
    use std::time::Instant;

    let device = Device::new()?;
    println!("device: {}", device.name());
    for tokens in [512, 2048] {
        let data: Vec<_> = (0..tokens * 8 * 128)
            .map(|i| ((i % 101) as f32 - 50.0) / 50.0)
            .collect();
        let q = Array::from_f32(&device, tokens, 8 * 128, &data)?;
        for causal in [false, true] {
            q.attention(&q, &q, 8, 8, causal)?.to_f32()?;
            let mut times = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                let values = q.attention(&q, &q, 8, 8, causal)?.to_f32()?;
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                assert!(values.iter().all(|v| v.is_finite()));
            }

            times.sort_by(f64::total_cmp);
            println!(
                "attention tokens={tokens} heads=8 dim=128 causal={causal}: {:.3} ms",
                times[2]
            );
        }
    }

    let input = Array::from_f32(&device, 256, 1024, &vec![1.0; 256 * 1024])?;
    let before = device.stats();
    let start = Instant::now();
    let mut result = input.clone();

    for _ in 0..128 {
        result = result.unary(0, 0.99)?;
    }

    let values = result.to_f32()?;
    assert!((values[0] - 0.99f32.powi(128)).abs() < 1e-5);
    println!(
        "128 elementwise operations: {:.3} ms, {} command buffers",
        start.elapsed().as_secs_f64() * 1000.0,
        device.stats().command_buffers - before.command_buffers
    );
    println!("runtime: {:?}", device.stats());

    use mmh3_core::{dit::inputs::DitInputs, safetensors::SafeTensors, tensor::Tensor};
    use mmh3_metal::dit::MetalDit;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/dit_tiny.safetensors");
    let file = SafeTensors::open(&path)?;
    let tensor = |name: &str| -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut tensor = Tensor::load(&file, file.get(name).unwrap())?;
        tensor.shape.remove(0);
        Ok(tensor)
    };

    let inputs = DitInputs {
        video: tensor("input.video")?,
        audio: tensor("input.audio")?,
        context: tensor("input.context")?,
        context_modalities: Vec::new(),
        keyframes: Vec::new(),
        references: Vec::new(),
        sigma: 0.7,
        shift_video: 3.0,
        shift_audio: 3.0,
    };

    let dit = MetalDit::load(&file, "weight.")?;
    let prepared = dit.prepare_text(&inputs.context)?;
    prepared.forward(&inputs, &[], None)?;
    let before = dit.device().stats();
    let mut times = Vec::new();

    for _ in 0..5 {
        let start = Instant::now();
        let output = prepared.forward(&inputs, &[], None)?;
        std::hint::black_box(output);
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(f64::total_cmp);
    println!(
        "tiny DiT, prepared prompt: {:.3} ms/step, {:.1} command buffers/step",
        times[2],
        (dit.device().stats().command_buffers - before.command_buffers) as f64 / 5.0
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
