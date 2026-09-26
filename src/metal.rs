//! Application-level selection and validation of the initial Metal backend.
use crate::{
    cli::option_float,
    models::{model_file, option_path},
};
use mmh3_core::safetensors::SafeTensors;
use mmh3_metal::{AttentionPrecision, LinearPrecision, dit::MetalDit};
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

    for (option, supported) in [("attention", "dense"), ("lora-mode", "adapter")] {
        if let Some(value) = options.get(option)
            && *value != supported
        {
            return Err(format!(
                "Metal runs --{option} {supported}, not {value}. Hand every step to a worker with --worker, or drop it"
            )
            .into());
        }
    }

    if let Some(value) = options.get("attention-precision")
        && !["fp32", "fp16"].contains(value)
    {
        return Err(format!(
            "Metal runs --attention-precision fp32 or fp16, not {value}. Hand every step to a worker with --worker, or drop it"
        )
        .into());
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

/// Whether this Mac's GPU has the matrix units the defaults run on: the tensor operations of
/// macOS 26, the Neural Accelerators on an M5 or later.
fn has_tensor_ops() -> bool {
    mmh3_metal::Device::shared().is_ok_and(|device| device.supports_tensor_ops())
}

/// The default of `--linear-precision`: INT8 activations on the matrix units, as CUDA runs them,
/// and FP16 through MPS on a Mac without them.
pub fn default_linear_precision() -> LinearPrecision {
    if has_tensor_ops() {
        LinearPrecision::Int8
    } else {
        LinearPrecision::MpsFp16
    }
}

/// The default of `--attention-precision`: FP16 on the matrix units, FP32 on a Mac without them.
pub fn default_attention_precision() -> AttentionPrecision {
    if has_tensor_ops() {
        AttentionPrecision::Fp16
    } else {
        AttentionPrecision::Fp32
    }
}

/// `--attention-precision`, FP16 on the matrix units by default and FP32 on a Mac without them.
pub fn attention_precision(
    options: &HashMap<&str, &str>,
) -> Result<AttentionPrecision, Box<dyn Error>> {
    Ok(match options.get("attention-precision").copied() {
        Some("fp32") => AttentionPrecision::Fp32,
        Some("fp16") => AttentionPrecision::Fp16,
        Some(value) => {
            return Err(
                format!("Metal runs --attention-precision fp32 or fp16, not {value}").into(),
            );
        }
        None => default_attention_precision(),
    })
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
    let mut dit = MetalDit::load_fitting(&SafeTensors::open(Path::new(&path))?, "")?;
    let precision = match options.get("linear-precision").copied() {
        Some("fp16") => LinearPrecision::Fp16,
        Some("int8") => LinearPrecision::Int8,
        Some("mps-fp16") => LinearPrecision::MpsFp16,
        Some(_) => LinearPrecision::Fp32,
        None => default_linear_precision(),
    };

    dit.set_linear_precision(precision)?;
    let attention = attention_precision(options)?;
    dit.set_attention_precision(attention)?;
    println!("Metal DiT INT8-weight products: {precision:?}, attention: {attention:?}");
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
