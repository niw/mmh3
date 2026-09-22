//! CUDA kernel benchmarks.

use crate::USAGE;
use mmh3::cli::{option_number, parse_options};
use std::error::Error;

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    match arguments.first().map(String::as_str) {
        Some("gemm") => bench_gemm(&arguments[1..]),
        Some("memory") => bench_memory(&arguments[1..]),
        Some("mma") => bench_mma(&arguments[1..]),
        Some("attention") => bench_attention(&arguments[1..]),
        Some("vsa") => bench_vsa(&arguments[1..]),
        Some("encoder") => bench_encoder(&arguments[1..]),
        _ => Err(USAGE.into()),
    }
}

/// Output and input features of the four linear layers in every DiT block.
const DIT_LINEAR_SHAPES: [(&str, usize, usize); 4] = [
    ("qkv", 21504, 5376),
    ("out", 5376, 7168),
    ("fc1", 28672, 5376),
    ("fc2", 5376, 14336),
];

const DIT_BLOCK_COUNT: usize = 50;

fn bench_gemm(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::bench::{GemmKind, gemm};

    let options = parse_options(arguments, &["tokens", "iterations", "kinds"], &[], USAGE)?;
    let tokens = option_number(&options, "tokens", 38_710)?;
    let iterations = option_number(&options, "iterations", 10)?;
    let kinds = match options.get("kinds") {
        Some(list) => list
            .split(',')
            .map(|name| {
                GemmKind::from_name(name).ok_or_else(|| format!("unknown GEMM kind {name}"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => GemmKind::ALL.to_vec(),
    };

    println!(
        "y[{tokens}, N] = x[{tokens}, K] · w[N, K]ᵀ, best candidate of each kind, {iterations} runs each"
    );
    println!(
        "{:<6} {:>6} {:>6}  {:<9} {:>10} {:>9} {:>10}",
        "layer", "N", "K", "kind", "ms", "TFLOPS", "candidates"
    );
    let mut step_seconds: Vec<(GemmKind, Option<f64>)> =
        kinds.iter().map(|&kind| (kind, Some(0.0))).collect();
    for (layer, output_features, input_features) in DIT_LINEAR_SHAPES {
        for (kind, total) in step_seconds.iter_mut() {
            let label = format!(
                "{layer:<6} {output_features:>6} {input_features:>6}  {:<9}",
                kind.name()
            );
            match gemm(*kind, tokens, output_features, input_features, iterations) {
                Ok(timing) => {
                    let operations =
                        2.0 * tokens as f64 * output_features as f64 * input_features as f64;
                    let teraflops = operations / (timing.milliseconds as f64 * 1e-3) / 1e12;
                    println!(
                        "{label} {:>10.3} {:>9.1} {:>10}",
                        timing.milliseconds, teraflops, timing.candidates_timed
                    );
                    if let Some(seconds) = total {
                        *seconds += timing.milliseconds as f64 * DIT_BLOCK_COUNT as f64 / 1e3;
                    }
                }
                Err(error) => {
                    println!("{label} {}", error.message);
                    *total = None;
                }
            }
        }
    }
    println!();
    println!("linear time per DiT forward ({DIT_BLOCK_COUNT} blocks)");
    for (kind, total) in &step_seconds {
        match total {
            Some(seconds) => println!("  {:<9} {seconds:.2} s", kind.name()),
            None => println!("  {:<9} unavailable", kind.name()),
        }
    }
    Ok(())
}

fn bench_memory(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse_options(arguments, &["megabytes", "iterations"], &[], USAGE)?;
    let megabytes = option_number(&options, "megabytes", 2048)?;
    let iterations = option_number(&options, "iterations", 20)?;
    let bandwidth = mmh3_cuda::bench::memory_copy(megabytes << 20, iterations)?;
    println!("device copy of {megabytes} MiB, {iterations} runs (read + write)");
    println!(
        "  copy kernel  {:.1} GB/s",
        bandwidth.copy_kernel_gigabytes_per_second
    );
    println!(
        "  cudaMemcpy   {:.1} GB/s",
        bandwidth.memcpy_gigabytes_per_second
    );
    Ok(())
}

fn bench_mma(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_cuda::bench::{MmaKind, mma_peak};

    let options = parse_options(arguments, &["iterations"], &[], USAGE)?;
    let iterations = option_number(&options, "iterations", 4096)?;
    println!(
        "register-only mma.sync throughput, {iterations} iterations of 8 independent chains per warp"
    );
    for kind in MmaKind::ALL {
        let throughput = mma_peak(kind, iterations)?;
        println!("  {:<22} {throughput:>7.1} T/s", kind.instruction());
    }
    Ok(())
}

fn bench_attention(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse_options(arguments, &["tokens", "heads", "iterations"], &[], USAGE)?;
    let tokens = option_number(&options, "tokens", 38_710)?;
    let heads = option_number(&options, "heads", 56)?;
    let iterations = option_number(&options, "iterations", 3)?;
    let milliseconds = mmh3_cuda::bench::attention(tokens, heads, iterations)?;
    let operations = 4.0 * (tokens as f64).powi(2) * 128.0 * heads as f64;
    let teraflops = operations / (milliseconds as f64 * 1e-3) / 1e12;
    println!("dense attention, {tokens} tokens, {heads} heads of 128, {iterations} runs");
    println!("  {milliseconds:.3} ms per call, {teraflops:.1} TFLOPS");
    println!(
        "  {:.2} s per DiT forward ({DIT_BLOCK_COUNT} blocks)",
        milliseconds as f64 * DIT_BLOCK_COUNT as f64 / 1e3
    );
    Ok(())
}

/// VSA over the packed sequence of a generation shape, with random BF16 inputs and gates: the pass
/// that normalizes q and k and prepares the tiles, then the attention.
fn bench_vsa(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3_core::dit::layout::PackedLayout;
    use mmh3_core::dit::vsa::{FASTH3_SPARSITY, VsaPlan};
    use mmh3_core::generation::GenerationShape;
    use mmh3_core::numeric::f32_to_bf16;
    use mmh3_cuda::DeviceBuffer;
    use mmh3_cuda::attention::{
        self, AttentionInputs, AttentionLayout, AttentionOffsets, AttentionPrecision, HEAD_DIM,
        HeadNorm, PreparedAttention, VsaWorkspace,
    };
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &[
            "width",
            "height",
            "frames",
            "heads",
            "iterations",
            "attention-precision",
        ],
        &[],
        USAGE,
    )?;
    let shape = GenerationShape::new(
        option_number(&options, "width", 1344)?,
        option_number(&options, "height", 768)?,
        option_number(&options, "frames", 124)?,
    )?;
    let heads = option_number(&options, "heads", 56)?;
    let iterations = option_number(&options, "iterations", 10)?;
    let precision = match options
        .get("attention-precision")
        .copied()
        .unwrap_or("bf16")
    {
        "bf16" => AttentionPrecision::Bf16,
        "int8-fp8" => AttentionPrecision::Int8Fp8,
        other => {
            return Err(
                format!("--attention-precision must be bf16 or int8-fp8, not {other}").into(),
            );
        }
    };
    let video = shape.video_latent_shape();
    let layout = PackedLayout::text_to_video(
        36,
        video[1],
        video[2],
        video[3],
        shape.audio_latent_shape()[2],
    );
    let plan = VsaPlan::for_layout(&layout);
    let tokens = layout.len();
    let inner = heads * HEAD_DIM;
    let kept = plan.kept_video_tiles(FASTH3_SPARSITY);
    let pairs = 48;

    let mut state = 1u64;
    let mut random = |count: usize| -> Vec<f32> {
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };
    let bf16 = |values: &[f32]| -> Result<DeviceBuffer, Box<dyn Error>> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&value| f32_to_bf16(value).to_le_bytes())
            .collect();
        Ok(DeviceBuffer::from_bytes(&bytes)?)
    };
    let mut input = bf16(&random(tokens * 3 * inner))?;
    let gate = bf16(&random(tokens * inner))?;
    let ones = bf16(&[1.0; HEAD_DIM])?;
    let angles = DeviceBuffer::from_f32(&random(tokens * pairs))?;
    let norm = HeadNorm {
        query_weight: &ones,
        key_weight: &ones,
        angles: Some(&angles),
        pairs,
        epsilon: 1e-6,
    };
    let mut output = DeviceBuffer::new(tokens * inner * 2)?;
    let layout = AttentionLayout {
        token_stride: [
            (3 * inner) as i64,
            (3 * inner) as i64,
            (3 * inner) as i64,
            inner as i64,
        ],
        head_stride: [HEAD_DIM as i64; 4],
        ..AttentionLayout::default()
    };
    let offsets = AttentionOffsets {
        query: 0,
        key: inner,
        value: 2 * inner,
        output: 0,
    };
    let workspace = VsaWorkspace::with_precision(&plan, tokens, heads, precision)?;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut run = || -> Result<(), Box<dyn Error>> {
        attention::prepare_inputs(
            &mut input,
            &norm,
            tokens,
            heads,
            PreparedAttention::Vsa(&workspace),
        )?;
        attention::vsa(
            &input,
            Some(&gate),
            &mut output,
            offsets,
            &layout,
            scale,
            kept,
            &workspace,
            AttentionInputs::Prepared,
        )?;
        Ok(())
    };
    run()?;
    mmh3_cuda::synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        run()?;
    }
    mmh3_cuda::synchronize()?;
    let milliseconds = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    let fraction = workspace.selected_fraction()?;
    println!(
        "VSA ({precision:?}), {tokens} tokens in {} tiles, {kept} of {} video tiles kept, {heads} heads of 128, {iterations} runs",
        plan.tiles(),
        plan.video_tiles()
    );
    println!(
        "  {milliseconds:.3} ms per call with the input pass, {:.1}% of the tile pairs selected",
        100.0 * fraction
    );
    println!(
        "  {:.2} s per DiT forward ({DIT_BLOCK_COUNT} blocks)",
        milliseconds * DIT_BLOCK_COUNT as f64 / 1e3
    );
    Ok(())
}

fn bench_encoder(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    use mmh3::models::{VIDEO_VAE_FILE, option_path};
    use mmh3_core::safetensors::SafeTensors;
    use mmh3_core::tensor::Tensor;
    use mmh3_cuda::video_encoder::{
        CudaVideoEncoder, DEFAULT_TILE_OVERLAP_MIN, DEFAULT_TILE_SIZE, Temporal,
    };
    use std::path::Path;
    use std::time::Instant;

    let options = parse_options(
        arguments,
        &[
            "models",
            "weights",
            "width",
            "height",
            "frames",
            "tile-size",
            "iterations",
        ],
        &[],
        USAGE,
    )?;
    let weights_path = option_path(&options, "weights", VIDEO_VAE_FILE)?;
    let width = option_number(&options, "width", 1344)?;
    let height = option_number(&options, "height", 768)?;
    let frames = option_number(&options, "frames", 124)?;
    let tile_size = option_number(&options, "tile-size", DEFAULT_TILE_SIZE)?;
    let iterations = option_number(&options, "iterations", 1)?;

    let started = Instant::now();
    let encoder = CudaVideoEncoder::load(
        &SafeTensors::open(Path::new(&weights_path))?,
        Temporal::Clip,
        tile_size,
        DEFAULT_TILE_OVERLAP_MIN,
    )?;
    println!(
        "loaded the encoder of {weights_path} in {:.1} s",
        started.elapsed().as_secs_f64()
    );

    // A smooth gradient with a moving band stands in for a clip: the cost is the shape's, not the
    // content's.
    let values = (0..frames * height * width * 3)
        .map(|index| {
            let pixel = index / 3;
            let (row, column) = (pixel / width % height, pixel % width);
            let frame = pixel / (height * width);
            ((row + column + 4 * frame) % 255) as f32 / 255.0
        })
        .collect();
    let clip = Tensor::new(vec![frames, height, width, 3], values);
    println!(
        "{width}×{height}, {frames} frames in {}×{} tiles, {} latent frames, {iterations} runs",
        tile_size,
        tile_size,
        CudaVideoEncoder::latent_frames(frames)
    );
    let mut best = f64::INFINITY;
    for _ in 0..iterations {
        let started = Instant::now();
        let posterior = encoder.encode_clip(&clip)?;
        let elapsed = started.elapsed().as_secs_f64();
        best = best.min(elapsed);
        println!(
            "  {elapsed:.2} s for the moments {:?}",
            posterior.mean.shape
        );
    }
    println!("{best:.2} s per clip at best");
    Ok(())
}
