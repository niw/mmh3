//! Application-level selection and validation of the initial Metal backend.
use crate::{
    cli::option_float,
    models::{model_file, option_path},
};
use mmh3_core::safetensors::SafeTensors;
use mmh3_metal::dit::MetalDit;
use std::{collections::HashMap, error::Error, path::Path};

/// Reject what this machine cannot do however a run is arranged, before reading images or weights.
///
/// Encoding a picture or a clip is this machine's own work whoever runs the steps, so a run that
/// needs one cannot be arranged around this backend at all.
pub fn validate_options(options: &HashMap<&str, &str>) -> Result<(), Box<dyn Error>> {
    for option in [
        "first-frame",
        "last-frame",
        "reference",
        "reference-audio",
        "reference-video",
    ] {
        if options.contains_key(option) {
            return Err(format!("--{option} is not supported by the initial Metal backend. It supports text-to-video and adapter LoRAs").into());
        }
    }

    Ok(())
}

/// Reject what this machine cannot do when it runs the steps itself.
///
/// A run that hands every step to a worker never reaches here, so this backend can coordinate one
/// that asks for what only another backend runs. What a rank of this backend refuses it refuses
/// for itself, when the session opens, and says so there.
pub fn validate_step_options(options: &HashMap<&str, &str>) -> Result<(), Box<dyn Error>> {
    for option in ["patch", "sparse-tau", "sparse-start", "vsa-sparsity"] {
        if options.contains_key(option) {
            return Err(format!(
                "Metal cannot run a step that --{option} asks for. Hand every step to a worker with --worker, or drop it"
            )
            .into());
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
            return Err(format!(
                "Metal runs --{option} {supported}, not {value}. Hand every step to a worker with --worker, or drop it"
            )
            .into());
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
    validate_step_options(options)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn options(pairs: &[(&'static str, &'static str)]) -> HashMap<&'static str, &'static str> {
        pairs.iter().copied().collect()
    }

    /// What a coordinator may ask for and what it may not, which is the whole of the split. A run
    /// this machine hands out every step of never reaches `validate_step_options`, so the first
    /// check has to let through everything that is only a step's business.
    #[test]
    fn a_coordinator_may_ask_for_what_it_cannot_run_itself() {
        for (option, value) in [
            ("attention", "sol"),
            ("attention", "vsa"),
            ("attention-precision", "int8-fp8"),
            ("lora-mode", "merge"),
            ("patch", "some.safetensors"),
            ("vsa-sparsity", "0.9"),
        ] {
            let options = options(&[(option, value)]);
            assert!(
                validate_options(&options).is_ok(),
                "--{option} {value} is a step's business, not this machine's"
            );
            assert!(
                validate_step_options(&options).is_err(),
                "--{option} {value} must be refused by the machine that runs the step"
            );
        }
    }

    /// Encoding a picture is this machine's own work whoever runs the steps, so no arrangement
    /// makes it possible and the first check refuses it.
    #[test]
    fn what_no_worker_can_take_is_refused_whatever_the_run_arranges() {
        for option in [
            "first-frame",
            "last-frame",
            "reference",
            "reference-audio",
            "reference-video",
        ] {
            let options = options(&[(option, "missing.png")]);
            assert!(
                validate_options(&options).is_err(),
                "--{option} cannot be handed to a worker"
            );
        }
    }
}
