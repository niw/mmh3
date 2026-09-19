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
        let device = Device::new()?;
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
                    std::str::from_utf8(file.data(info)).map_err(|e| Error(e.to_string()))?;
                let value = json::parse(text).map_err(|e| Error(e.to_string()))?;
                if value.get("format").and_then(json::Value::as_str) != Some("int8_tensorwise")
                    || value.get("convrot").and_then(json::Value::as_bool) != Some(true)
                    || value.get("convrot_groupsize").and_then(json::Value::as_u64) != Some(256)
                {
                    return Err(Error(format!("{name}: unsupported quantization {text}")));
                }

                continue;
            }

            dtype_code(info.dtype)?;
            if info.element_count() == 0 {
                return Err(Error(format!("empty weight {name}")));
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
                    .ok_or_else(|| Error(format!("unexpected INT8 tensor {name}")))?;
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
                    return Err(Error(format!(
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
            .ok_or_else(|| Error(format!("missing tensor {name}")))?;
        if w.shape.len() != 2 || ids.is_empty() || ids.iter().any(|&i| i as usize >= w.shape[0]) {
            return Err(Error("invalid embedding indices".into()));
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
            .ok_or_else(|| Error(format!("missing tensor {name}")))
    }

    pub fn array(&self, name: &str) -> Result<Array> {
        let w = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("missing tensor {name}")))?;
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
            .ok_or_else(|| Error(format!("missing tensor {key}")))?;
        if w.shape.len() < 2 || w.shape[1..].iter().product::<usize>() != x.cols {
            return Err(Error(format!("{name}: linear input shape mismatch")));
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
    pub fn linear_quantized(
        &self,
        input: &crate::Buffer,
        scales: &Array,
        rows: usize,
        name: &str,
    ) -> Result<Array> {
        let key = format!("{name}.weight");
        let w = self
            .tensors
            .get(&key)
            .ok_or_else(|| Error(format!("missing tensor {key}")))?;
        if w.dtype != DType::I8 {
            return Err(Error(format!(
                "{name}: an exchanged input reaches INT8 layers only"
            )));
        }
        // NOTE: CUDA runs an adapter's down projection on the quantized rows themselves, so
        // those rows are enough for it. Metal has no such path, so a rank holding only them
        // cannot apply one. Refusing beats dropping the adapter silently.
        if self.adapters.contains_key(name) {
            return Err(Error(format!(
                "{name}: an adapter needs the input a rank did not receive"
            )));
        }

        let cols = w.shape[1..].iter().product::<usize>();
        let mut result = Array::linear_quantized(
            &self.device,
            input,
            scales,
            &w.buffer,
            rows,
            cols,
            w.shape[0],
            self.linear_precision,
        )?
        .mul(&self.vector(&format!("{name}.weight_scale"))?)?;

        let bias = format!("{name}.bias");
        if self.contains(&bias) {
            result = result.add(&self.vector(&bias)?)?;
        }

        Ok(result)
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
            return Err(Error("LoRA strength must be finite".into()));
        }

        let mut adapters = Vec::new();
        const PREFIX: &str = "diffusion_model.";
        for info in file.tensors() {
            let Some(name) = info.name.strip_prefix(PREFIX) else {
                return Err(Error(format!("unsupported LoRA tensor {}", info.name)));
            };

            if name.ends_with(".lora_B.weight") || name.ends_with(".alpha") {
                continue;
            }

            let Some(layer) = name.strip_suffix(".lora_A.weight") else {
                return Err(Error("Metal supports adapter LoRAs. Replacement-weight patches are not supported yet".into()));
            };

            let load = |name: &str| -> Result<Tensor> {
                Tensor::load(
                    file,
                    file.get(name)
                        .ok_or_else(|| Error(format!("missing LoRA tensor {name}")))?,
                )
                .map_err(Error)
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
                return Err(Error(format!("LoRA does not fit {layer}")));
            }

            if self.adapters.contains_key(layer) {
                return Err(Error(format!("{layer} already has a LoRA")));
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
            return Err(Error("no supported LoRA layers found".into()));
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
