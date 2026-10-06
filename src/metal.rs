//! Application-level selection and validation of the Metal backend.
use crate::{
    cli::option_float,
    models::{model_file, option_path},
};
use mmh3_core::dit::veda::{VEDA_PREDICTOR_FILE, VedaPredictor};
use mmh3_core::safetensors::SafeTensors;
use mmh3_metal::{AttentionPrecision, LinearPrecision, dit::MetalDit};
use std::{collections::HashMap, error::Error, path::Path};

/// Reject what this machine cannot do when it runs the steps itself.
///
/// A run that hands every step to a worker never reaches here, so this backend can coordinate one
/// that asks for what only another backend runs. What a rank of this backend refuses it refuses
/// for itself, when the session opens, and says so there.
pub fn validate_step_options(options: &HashMap<&str, &str>) -> Result<(), Box<dyn Error>> {
    if let Some(value) = options.get("lora-mode")
        && *value != "adapter"
    {
        return Err(format!(
            "Metal runs --lora-mode adapter, not {value}. Hand every step to a worker with --worker, or drop it"
        )
        .into());
    }

    if let Some(value) = options.get("attention-precision")
        && !["fp32", "fp16", "int8-fp16"].contains(value)
    {
        return Err(format!(
            "Metal runs --attention-precision fp32, fp16 or int8-fp16, not {value}. Hand every step to a worker with --worker, or drop it"
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
        Some("int8-fp16") => AttentionPrecision::Int8,
        Some(value) => {
            return Err(format!(
                "Metal runs --attention-precision fp32, fp16 or int8-fp16, not {value}"
            )
            .into());
        }
        None => default_attention_precision(),
    })
}

pub fn load_dit(
    options: &HashMap<&str, &str>,
    name: &str,
    default_file: &str,
) -> Result<MetalDit, Box<dyn Error>> {
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
    // NOTE: a patch is a LoRA file whose other tensors replace or add checkpoint tensors, and it
    // applies as adapters at full strength.
    if let Some(patch) = model_file(options, "patch", &["patches", "loras"])? {
        let count = dit.add_lora(&SafeTensors::open(Path::new(&patch))?, 1.0)?;
        println!("applied {patch} to {count} layers and tensors");
    }
    if options.get("attention") == Some(&"veda") {
        let path = option_path(options, "veda-predictor", VEDA_PREDICTOR_FILE)?;
        let predictor = VedaPredictor::open(Path::new(&path))?;
        dit.set_veda_predictor(&predictor)?;
        println!(
            "loaded the Veda predictor {path}, trained to keep {} of the video tiles",
            predictor.keep_ratio
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

    /// What only a step's machine runs is refused by the machine that runs the step.
    #[test]
    fn a_step_refuses_what_it_cannot_run() {
        for (option, value) in [("attention-precision", "int8-fp8"), ("lora-mode", "merge")] {
            let options = options(&[(option, value)]);
            assert!(
                validate_step_options(&options).is_err(),
                "--{option} {value} must be refused by the machine that runs the step"
            );
        }
    }

    /// Sol-Attn, VSA and the patches that bring VSA's gates run here.
    #[test]
    fn a_step_may_attend_sparsely_and_take_a_patch() {
        for (option, value) in [
            ("attention", "sol"),
            ("attention", "vsa"),
            ("sparse-tau", "1.3"),
            ("sparse-start", "0"),
            ("vsa-sparsity", "0.9"),
            ("patch", "some.safetensors"),
            ("attention-precision", "int8-fp16"),
        ] {
            let options = options(&[(option, value)]);
            assert!(
                validate_step_options(&options).is_ok(),
                "--{option} {value} runs on Metal"
            );
        }
    }
}
