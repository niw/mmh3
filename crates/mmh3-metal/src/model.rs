use crate::{
    Buffer, Device, Error, LinearPrecision, Result,
    ops::{Array, convert, dtype_code},
};
use mmh3_core::{
    json,
    safetensors::{DType, SafeTensors},
    tensor::Tensor,
};
use std::collections::HashMap;

struct Weight {
    buffer: Buffer,
    shape: Vec<usize>,
    dtype: DType,
}

struct Adapter {
    down: Array,
    up: Array,
    scale: f32,
}
pub(crate) struct Weights {
    pub device: Device,
    pub linear_precision: LinearPrecision,
    tensors: HashMap<String, Weight>,
    adapters: HashMap<String, Adapter>,
}

impl Weights {
    pub fn load(file: &SafeTensors, prefix: &str) -> Result<Self> {
        Self::load_selected(file, prefix, |_| true)
    }

    pub fn load_selected(
        file: &SafeTensors,
        prefix: &str,
        keep: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let device = Device::shared()?;
        let mut tensors = HashMap::new();
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(prefix) else {
                continue;
            };

            if !keep(name) {
                continue;
            }

            if name.ends_with(".comfy_quant") {
                let text =
                    std::str::from_utf8(file.data(info)).map_err(|e| Error::new(e.to_string()))?;
                let value = json::parse(text).map_err(|e| Error::new(e.to_string()))?;
                if value.get("format").and_then(json::Value::as_str) != Some("int8_tensorwise")
                    || value.get("convrot").and_then(json::Value::as_bool) != Some(true)
                    || value.get("convrot_groupsize").and_then(json::Value::as_u64) != Some(256)
                {
                    return Err(Error::new(format!(
                        "{name}: unsupported quantization {text}"
                    )));
                }

                continue;
            }

            dtype_code(info.dtype)?;
            if info.element_count() == 0 {
                return Err(Error::new(format!("empty weight {name}")));
            }

            tensors.insert(
                name.to_owned(),
                Weight {
                    buffer: device.alloc(info.byte_count(), Some(file.data(info)))?,
                    shape: info.shape.clone(),
                    dtype: info.dtype,
                },
            );
        }

        for (name, weight) in &tensors {
            if weight.dtype == DType::I8 {
                let layer = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| Error::new(format!("unexpected INT8 tensor {name}")))?;
                if file.get(&format!("{prefix}{layer}.comfy_quant")).is_none()
                    || weight.shape.len() != 2
                    || !weight.shape[1].is_multiple_of(256)
                    || tensors
                        .get(&format!("{layer}.weight_scale"))
                        .is_none_or(|s| {
                            s.dtype != DType::F32
                                || s.shape.iter().product::<usize>() != weight.shape[0]
                        })
                {
                    return Err(Error::new(format!(
                        "{name}: expected per-output INT8 ConvRot scales and metadata"
                    )));
                }
            }
        }

        Ok(Self {
            device,
            tensors,
            adapters: HashMap::new(),
            linear_precision: LinearPrecision::Fp32,
        })
    }

    /// Whether a layer's weights are INT8, which is what an exchanged block input reaches.
    pub fn is_int8(&self, name: &str) -> bool {
        self.tensors
            .get(&format!("{name}.weight"))
            .is_some_and(|w| w.dtype == DType::I8)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Prepack audio convolution matrices once, directly on the GPU, replacing their originals.
    pub fn prepare_audio_convolutions(&mut self) -> Result<()> {
        for (name, weight) in &mut self.tensors {
            if !name.ends_with(".weight") || weight.shape.len() != 3 {
                continue;
            }

            let transpose = name.starts_with("decoder.ups.");
            let (inputs, outputs, kernel) = if transpose {
                (weight.shape[0], weight.shape[1], weight.shape[2])
            } else {
                (weight.shape[1], weight.shape[0], weight.shape[2])
            };

            let packed = Array::empty(&self.device, outputs * kernel, inputs)?;
            self.device.run(
                "audio_reorder_weight",
                &[&weight.buffer, &packed.buffer],
                &[
                    packed.len() as u32,
                    inputs as u32,
                    outputs as u32,
                    kernel as u32,
                    transpose as u32,
                    dtype_code(weight.dtype)?,
                ],
                packed.len(),
                false,
            )?;
            weight.buffer = packed.buffer;
            weight.dtype = DType::F32;
        }

        Ok(())
    }

    pub fn embedding(&self, name: &str, ids: &[u32]) -> Result<Array> {
        let w = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::new(format!("missing tensor {name}")))?;
        if w.shape.len() != 2 || ids.is_empty() || ids.iter().any(|&i| i as usize >= w.shape[0]) {
            return Err(Error::new("invalid embedding indices".into()));
        }

        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_ne_bytes()).collect();
        let indices = self.device.alloc(bytes.len(), Some(&bytes))?;
        let out = Array::empty(&self.device, ids.len(), w.shape[1])?;
        self.device.run(
            "embedding",
            &[&w.buffer, &indices, &out.buffer],
            &[out.len() as u32, out.cols as u32, dtype_code(w.dtype)?],
            out.len(),
            false,
        )?;
        Ok(out)
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.tensors
            .get(name)
            .map(|w| w.shape.clone())
            .ok_or_else(|| Error::new(format!("missing tensor {name}")))
    }

    pub fn array(&self, name: &str) -> Result<Array> {
        let w = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::new(format!("missing tensor {name}")))?;
        let rows = if w.shape.len() >= 2 { w.shape[0] } else { 1 };
        let count = w.shape.iter().product::<usize>();
        convert(&self.device, &w.buffer, rows, count / rows, w.dtype)
    }

    pub fn vector(&self, name: &str) -> Result<Array> {
        let a = self.array(name)?;
        a.reshape(1, a.len())
    }

    pub fn host(&self, name: &str) -> Result<Tensor> {
        Ok(Tensor::new(self.shape(name)?, self.array(name)?.to_f32()?))
    }

    pub fn linear(&self, x: &Array, name: &str) -> Result<Array> {
        let key = format!("{name}.weight");
        let w = self
            .tensors
            .get(&key)
            .ok_or_else(|| Error::new(format!("missing tensor {key}")))?;
        if w.shape.len() < 2 || w.shape[1..].iter().product::<usize>() != x.cols {
            return Err(Error::new(format!("{name}: linear input shape mismatch")));
        }

        let mut result = if w.dtype == DType::I8 {
            let rotated = x.rotate()?;
            let product = if self.linear_precision == LinearPrecision::Fp32 {
                rotated.linear_packed(&w.buffer, w.shape[0], w.dtype)?
            } else {
                rotated.linear_int8(&w.buffer, w.shape[0], self.linear_precision)?
            };

            product.mul(&self.vector(&format!("{name}.weight_scale"))?)?
        } else if w.dtype != DType::F32 {
            x.linear_packed(&w.buffer, w.shape[0], w.dtype)?
        } else {
            x.linear(&self.array(&key)?)?
        };

        let bias = format!("{name}.bias");
        if self.contains(&bias) {
            result = result.add(&self.vector(&bias)?)?;
        }

        if let Some(adapter) = self.adapters.get(name) {
            result = result.add(
                &x.linear(&adapter.down)?
                    .linear(&adapter.up)?
                    .unary(0, adapter.scale)?,
            )?;
        }

        Ok(result)
    }

    /// `linear` for an input another rank has already rotated and quantized, which is how a
    /// block's input crosses the wire. Neither is done again: the rotation is in the bytes, and
    /// quantizing a second time would not give them back.
    ///
    /// `keep` names ranges of the weight's output rows, laid out in the order given, and an empty
    /// one is the whole weight. A rank that attends a few of a block's heads needs a few of its
    /// projection's outputs, and the rest would be computed and then dropped. That waste does not
    /// shrink as a rank's share shrinks, so at one head out of fifty-six it is nearly the whole
    /// product.
    pub fn linear_quantized(
        &self,
        input: &crate::Buffer,
        scales: &Array,
        rows: usize,
        name: &str,
        keep: &[std::ops::Range<usize>],
    ) -> Result<Array> {
        let key = format!("{name}.weight");
        let w = self
            .tensors
            .get(&key)
            .ok_or_else(|| Error::new(format!("missing tensor {key}")))?;
        if w.dtype != DType::I8 {
            return Err(Error::new(format!(
                "{name}: an exchanged input reaches INT8 layers only"
            )));
        }
        let cols = w.shape[1..].iter().product::<usize>();
        let outputs = w.shape[0];
        let kept: usize = keep.iter().map(|range| range.len()).sum();
        if keep.iter().any(|range| range.end > outputs) {
            return Err(Error::new(format!("{name}: outputs outside the weight")));
        }

        let taken = if keep.is_empty() {
            None
        } else {
            Some(gather_rows(&self.device, &w.buffer, cols, keep)?)
        };
        let (weight, outputs) = match &taken {
            Some(buffer) => (buffer, kept),
            None => (&w.buffer, outputs),
        };
        let mut result = Array::linear_quantized(
            &self.device,
            input,
            scales,
            weight,
            rows,
            cols,
            outputs,
            self.linear_precision,
        )?
        .mul(&self.kept_vector(&format!("{name}.weight_scale"), keep)?)?;

        let bias = format!("{name}.bias");
        if self.contains(&bias) {
            result = result.add(&self.kept_vector(&bias, keep)?)?;
        }

        // A LoRA's down projection runs on the rows this rank was handed, so the rows suffice.
        // Its weights are rotated to meet them: the activations arrive rotated, and the product of
        // two rotated sides is the product of the unrotated pair.
        if let Some(adapter) = self.adapters.get(name) {
            let rotated = Array::from_quantized(&self.device, input, scales, rows, cols)?;
            // The up projection answers one column per output, so it is cut the same way.
            let up = self.kept_up(&adapter.up, keep)?;
            result = result.add(
                &rotated
                    .linear(&adapter.down.rotate()?)?
                    .linear(&up)?
                    .unary(0, adapter.scale)?,
            )?;
        }

        Ok(result)
    }

    /// `vector`, cut to `keep` the way the weight beside it was. An empty `keep` is the whole of
    /// it, which is the answer `vector` gives.
    fn kept_vector(&self, name: &str, keep: &[std::ops::Range<usize>]) -> Result<Array> {
        let whole = self.vector(name)?;
        if keep.is_empty() {
            return Ok(whole);
        }
        let kept: usize = keep.iter().map(|range| range.len()).sum();
        let out = Array::empty(&self.device, 1, kept)?;
        let mut written = 0;
        for range in keep {
            out.buffer
                .copy_range(&whole.buffer, range.start * 4, written * 4, range.len() * 4)?;
            written += range.len();
        }
        Ok(out)
    }

    /// An adapter's up projection cut to `keep`. It answers one row per output, the same way the
    /// weight does, so the same contiguous gather serves.
    fn kept_up(&self, up: &Array, keep: &[std::ops::Range<usize>]) -> Result<Array> {
        if keep.is_empty() {
            return Ok(up.clone());
        }
        let kept: usize = keep.iter().map(|range| range.len()).sum();
        let buffer = gather_rows(&self.device, &up.buffer, up.cols * 4, keep)?;
        Ok(Array {
            buffer,
            rows: kept,
            cols: up.cols,
        })
    }

    pub fn norm(&self, x: &Array, name: &str, epsilon: f32) -> Result<Array> {
        let bias = format!("{name}.bias");
        let mut y = x.norm(
            &self.vector(&format!("{name}.weight"))?,
            epsilon,
            self.contains(&bias),
        )?;
        if self.contains(&bias) {
            y = y.add(&self.vector(&bias)?)?;
        }

        Ok(y)
    }

    pub fn add_lora(&mut self, file: &SafeTensors, strength: f32) -> Result<usize> {
        if !strength.is_finite() {
            return Err(Error::new("LoRA strength must be finite".into()));
        }

        let mut adapters = Vec::new();
        const PREFIX: &str = "diffusion_model.";
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                return Err(Error::new(format!("unsupported LoRA tensor {}", info.name)));
            };

            if name.ends_with(".lora_B.weight") || name.ends_with(".alpha") {
                continue;
            }

            let Some(layer) = name.strip_suffix(".lora_A.weight") else {
                return Err(Error::new("Metal supports adapter LoRAs. Replacement-weight patches are not supported yet".into()));
            };

            let load = |name: &str| -> Result<Tensor> {
                Tensor::load(
                    file,
                    file.get(name)
                        .ok_or_else(|| Error::new(format!("missing LoRA tensor {name}")))?,
                )
                .map_err(Error::new)
            };

            let down = load(&info.name)?;
            let up = load(&format!("{PREFIX}{layer}.lora_B.weight"))?;
            let alpha = load(&format!("{PREFIX}{layer}.alpha"))?;
            let shape = self.shape(&format!("{layer}.weight"))?;
            if down.shape.len() != 2
                || up.shape.len() != 2
                || shape.len() != 2
                || down.shape[0] == 0
                || down.shape[1] != shape[1]
                || up.shape != [shape[0], down.shape[0]]
                || alpha.data.len() != 1
                || !alpha.data[0].is_finite()
            {
                return Err(Error::new(format!("LoRA does not fit {layer}")));
            }

            if self.adapters.contains_key(layer) {
                return Err(Error::new(format!("{layer} already has a LoRA")));
            }

            adapters.push((
                layer.to_owned(),
                Adapter {
                    down: Array::from_f32(&self.device, down.shape[0], down.shape[1], &down.data)?,
                    up: Array::from_f32(&self.device, up.shape[0], up.shape[1], &up.data)?,
                    scale: strength * alpha.data[0] / down.shape[0] as f32,
                },
            ));
        }

        if adapters.is_empty() {
            return Err(Error::new("no supported LoRA layers found".into()));
        }

        let count = adapters.len();
        self.adapters.extend(adapters);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_convrot_matches_explicit_regular_hadamard_product() {
        let device = Device::new().unwrap();
        let rows = 3;
        let inputs = 512;
        let outputs = 5;
        let x: Vec<f32> = (0..rows * inputs)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) / 31.0)
            .collect();
        let weight: Vec<i8> = (0..inputs * outputs)
            .map(|i| ((i * 7 % 255) as i32 - 127) as i8)
            .collect();
        let scales = [0.001, 0.02, 0.0, 0.003, 0.004];
        let mut tensors = HashMap::new();
        tensors.insert(
            "linear.weight".into(),
            Weight {
                buffer: device
                    .alloc(
                        weight.len(),
                        Some(&weight.iter().map(|&v| v as u8).collect::<Vec<_>>()),
                    )
                    .unwrap(),
                shape: vec![outputs, inputs],
                dtype: DType::I8,
            },
        );
        tensors.insert(
            "linear.weight_scale".into(),
            Weight {
                buffer: Array::from_f32(&device, 1, outputs, &scales)
                    .unwrap()
                    .buffer,
                shape: vec![outputs],
                dtype: DType::F32,
            },
        );
        let mut w = Weights {
            device: device.clone(),
            tensors,
            adapters: HashMap::new(),
            linear_precision: LinearPrecision::Fp32,
        };

        let actual = w
            .linear(
                &Array::from_f32(&device, rows, inputs, &x).unwrap(),
                "linear",
            )
            .unwrap()
            .to_f32()
            .unwrap();
        // Independent dense H256 matrix: a radix-4 digit of row XOR column equal to 3
        // contributes a minus sign. Two groups exercise the transform boundary.
        for row in 0..rows {
            let rotated: Vec<f64> = (0..inputs)
                .map(|out| {
                    (0..256)
                        .map(|col| {
                            let bits = (out % 256) ^ col;
                            let negatives = (0..4)
                                .filter(|shift| (bits >> (2 * shift)) & 3 == 3)
                                .count();
                            x[row * inputs + out / 256 * 256 + col] as f64
                                * if negatives % 2 == 0 { 1.0 } else { -1.0 }
                        })
                        .sum::<f64>()
                        / 16.0
                })
                .collect();
            for out in 0..outputs {
                let expected = rotated
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * weight[out * inputs + i] as f64)
                    .sum::<f64>()
                    * scales[out] as f64;
                let got = actual[row * outputs + out] as f64;
                assert!(
                    (got - expected).abs() < 2e-4,
                    "{row}/{out}: {got} != {expected}"
                );
            }
        }

        if device.supports_tensor_ops() {
            // LoRA must still use the original (unrotated, unquantized) activation.
            let down: Vec<f32> = (0..inputs).map(|i| (i % 7) as f32 / 100.0).collect();
            let up = [0.5, -0.2, 0.1, 0.0, 1.0];
            w.adapters.insert(
                "linear".into(),
                Adapter {
                    down: Array::from_f32(&device, 1, inputs, &down).unwrap(),
                    up: Array::from_f32(&device, outputs, 1, &up).unwrap(),
                    scale: 0.7,
                },
            );

            for precision in [LinearPrecision::Fp16, LinearPrecision::Int8] {
                w.linear_precision = precision;
                let got = w
                    .linear(
                        &Array::from_f32(&device, rows, inputs, &x).unwrap(),
                        "linear",
                    )
                    .unwrap()
                    .to_f32()
                    .unwrap();
                let mut error = 0.0;
                let mut norm = 0.0;

                for r in 0..rows {
                    let hidden: f64 = (0..inputs)
                        .map(|k| x[r * inputs + k] as f64 * down[k] as f64)
                        .sum();
                    for n in 0..outputs {
                        let expected = actual[r * outputs + n] as f64 + hidden * up[n] as f64 * 0.7;
                        error += (got[r * outputs + n] as f64 - expected).powi(2);
                        norm += expected.powi(2);
                        assert!(got[r * outputs + n].is_finite());
                    }
                }

                let relative = (error / norm).sqrt();
                assert!(
                    relative
                        < if precision == LinearPrecision::Int8 {
                            0.03
                        } else {
                            0.002
                        },
                    "{precision:?} ConvRot + LoRA relative L2 {relative}"
                );
            }
        }
    }
}

/// `keep`, ranges of `source`'s rows of `row_bytes` each, copied out end to end. The rows of one
/// range are contiguous, so this is one copy per range rather than one per row.
fn gather_rows(
    device: &crate::Device,
    source: &crate::Buffer,
    row_bytes: usize,
    keep: &[std::ops::Range<usize>],
) -> Result<crate::Buffer> {
    let kept: usize = keep.iter().map(|range| range.len()).sum();
    let out = device.alloc(kept * row_bytes, None)?;
    let mut written = 0;
    for range in keep {
        out.copy_range(
            source,
            range.start * row_bytes,
            written * row_bytes,
            range.len() * row_bytes,
        )?;
        written += range.len();
    }
    Ok(out)
}
