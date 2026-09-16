//! Application-level selection and validation of the initial Metal backend.
use crate::{
    cli::option_float,
    models::{model_file, option_path},
};
use mmh3_core::safetensors::SafeTensors;
use mmh3_metal::dit::MetalDit;
use std::{collections::HashMap, error::Error, path::Path};

/// Reject unsupported modes before reading images, loading weights or allocating GPU memory.
pub fn validate_options(options: &HashMap<&str, &str>) -> Result<(), Box<dyn Error>> {
    for option in [
        "first-frame",
        "last-frame",
        "reference",
        "reference-audio",
        "reference-video",
        "patch",
        "sparse-tau",
        "sparse-start",
        "vsa-sparsity",
    ] {
        if options.contains_key(option) {
            return Err(format!("--{option} is not supported by the initial Metal backend. It supports text-to-video and adapter LoRAs").into());
        }
    }

    for (option, supported) in [
        ("attention", "dense"),
        ("attention-precision", "fp32"),
        ("lora-mode", "adapter"),
    ] {
        if let Some(value) = options.get(option)
            && *value != supported
        {
            return Err(format!("Metal supports --{option} {supported}, not {value}").into());
        }
    }

    if let Some(value) = options.get("linear-precision")
        && !["fp32", "fp16", "int8", "mps-fp16"].contains(value)
    {
        return Err(format!(
            "Metal supports --linear-precision fp32, fp16, int8 or mps-fp16, not {value}"
        )
        .into());
    }

    Ok(())
}

pub fn load_dit(
    options: &HashMap<&str, &str>,
    name: &str,
    default_file: &str,
) -> Result<MetalDit, Box<dyn Error>> {
    validate_options(options)?;
    let path = option_path(options, name, default_file)?;
    let started = std::time::Instant::now();
    let mut dit = MetalDit::load(&SafeTensors::open(Path::new(&path))?, "")?;
    let precision = match options
        .get("linear-precision")
        .copied()
        .unwrap_or("mps-fp16")
    {
        "fp16" => mmh3_metal::LinearPrecision::Fp16,
        "int8" => mmh3_metal::LinearPrecision::Int8,
        "mps-fp16" => mmh3_metal::LinearPrecision::MpsFp16,
        _ => mmh3_metal::LinearPrecision::Fp32,
    };

    dit.set_linear_precision(precision)?;
    println!("Metal DiT INT8-weight products: {precision:?}");
    if dit.has_vsa_gates() {
        return Err(
            "VSA/FastH3 checkpoints need VSA attention, which is not implemented on Metal yet"
                .into(),
        );
    }

    if let Some(path) = model_file(options, "lora", &["loras"])? {
        let count = dit.add_lora(
            &SafeTensors::open(Path::new(&path))?,
            option_float(options, "lora-strength", 1.0)?,
        )?;
        println!("added {count} LoRA adapters from {path}");
    }

    println!(
        "loaded {path} on {} in {:.1} s",
        dit.device().name(),
        started.elapsed().as_secs_f64()
    );
    Ok(dit)
}
