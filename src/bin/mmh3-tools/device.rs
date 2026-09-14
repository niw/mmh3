//! CUDA device information.

use crate::format::{format_bytes, format_count};
use std::error::Error;

pub(crate) fn run() -> Result<(), Box<dyn Error>> {
    let count = mmh3_cuda::device_count()?;
    for device in 0..count {
        let info = mmh3_cuda::device_info(device)?;
        println!("device {device}: {}", info.name);
        println!(
            "  compute capability     {}.{}",
            info.compute_capability.0, info.compute_capability.1
        );
        println!("  multiprocessors        {}", info.multiprocessor_count);
        println!(
            "  shared memory / block  {} bytes (opt-in)",
            format_count(info.shared_memory_per_block_optin as usize)
        );
        println!(
            "  shared memory / SM     {} bytes",
            format_count(info.shared_memory_per_multiprocessor as usize)
        );
        println!(
            "  L2 cache               {}",
            format_bytes(info.l2_cache_bytes as usize)
        );
        println!("  integrated             {}", info.integrated);
        println!(
            "  global memory          {}",
            format_bytes(info.total_global_bytes as usize)
        );
    }
    let (free_bytes, total_bytes) = mmh3_cuda::memory_info()?;
    println!(
        "memory    {} free of {}",
        format_bytes(free_bytes),
        format_bytes(total_bytes)
    );
    Ok(())
}
