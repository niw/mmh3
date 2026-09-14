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

    let options = parse_options(arguments, &["tokens", "iterations", "kinds"], USAGE)?;
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
    let options = parse_options(arguments, &["megabytes", "iterations"], USAGE)?;
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

    let options = parse_options(arguments, &["iterations"], USAGE)?;
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
    let options = parse_options(arguments, &["tokens", "heads", "iterations"], USAGE)?;
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
