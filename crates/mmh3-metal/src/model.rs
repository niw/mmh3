use crate::{
    AttentionPrecision, Buffer, Device, Error, LinearPrecision, Result,
    ops::{Array, Finish, Lora, convert, dtype_code},
};
use mmh3_core::{
    json,
    numeric::f32_to_f16,
    safetensors::{DType, SafeTensors},
    tensor::Tensor,
};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone)]
pub(crate) struct Weight {
    pub(crate) buffer: Buffer,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: DType,
}

/// A LoRA's two projections: FP16 on a Mac with matrix units to read them, FP32 on one without.
/// Never both, since the adapters of a real LoRA take gigabytes.
enum Adapter {
    Fp32 {
        down: Array,
        up: Array,
        scale: f32,
    },
    /// `down` is `rank × inputs` halves and `up` is `outputs × rank`, with the scale in it.
    Fp16 {
        down: Buffer,
        up: Buffer,
        rank: usize,
        outputs: usize,
    },
}

impl Adapter {
    /// `down` is `rank × inputs` and `up` is `outputs × rank`, row-major.
    fn new(
        device: &Device,
        down: (&[f32], usize, usize),
        up: (&[f32], usize, usize),
        scale: f32,
        half: bool,
    ) -> Result<Self> {
        if !half {
            return Ok(Self::Fp32 {
                down: Array::from_f32(device, down.1, down.2, down.0)?,
                up: Array::from_f32(device, up.1, up.2, up.0)?,
                scale,
            });
        }

        let halves = |values: &mut dyn Iterator<Item = f32>| -> Vec<u8> {
            values.flat_map(|v| f32_to_f16(v).to_le_bytes()).collect()
        };
        let down_bytes = halves(&mut down.0.iter().copied());
        let up_bytes = halves(&mut up.0.iter().map(|&v| v * scale));
        Ok(Self::Fp16 {
            down: device.alloc(down_bytes.len(), Some(&down_bytes))?,
            up: device.alloc(up_bytes.len(), Some(&up_bytes))?,
            rank: down.1,
            outputs: up.1,
        })
    }

    /// What the adapter adds to the product of `x`. On the matrix units the FP16 weights read the
    /// FP32 rows as they are.
    fn apply(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Fp32 { down, up, scale } => x.linear(down)?.linear(up)?.unary(0, *scale),
            Self::Fp16 {
                down,
                up,
                rank,
                outputs,
            } => x.linear_half(down, *rank)?.linear_half(up, *outputs),
        }
    }
}

pub(crate) struct Weights {
    pub device: Device,
    pub linear_precision: LinearPrecision,
    pub attention_precision: AttentionPrecision,
    tensors: HashMap<String, Weight>,
    /// Tensors of the units read on the way, while their units are on the device.
    loaded: RefCell<HashMap<String, Weight>>,
    /// The type and shape of every tensor left on the disk, which a caller may ask about while its
    /// unit is not on the device.
    absent: HashMap<String, (DType, Vec<usize>)>,
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
        Self::load_leaving(file, prefix, keep, |_| false)
    }

    /// `load_selected`, leaving the tensors `leave` names on the disk. Their units bring them to the
    /// device with `load_unit` when they run.
    pub fn load_leaving(
        file: &SafeTensors,
        prefix: &str,
        keep: impl Fn(&str) -> bool,
        leave: impl Fn(&str) -> bool,
    ) -> Result<Self> {
        let device = Device::shared()?;
        let mut tensors = HashMap::new();
        let mut absent = HashMap::new();
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

            if leave(name) {
                absent.insert(name.to_owned(), (info.dtype, info.shape.clone()));
                continue;
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

        let described: HashMap<&str, (DType, &[usize])> = tensors
            .iter()
            .map(|(name, weight)| (name.as_str(), (weight.dtype, weight.shape.as_slice())))
            .chain(
                absent
                    .iter()
                    .map(|(name, (dtype, shape))| (name.as_str(), (*dtype, shape.as_slice()))),
            )
            .collect();
        for (name, &(dtype, shape)) in &described {
            if dtype == DType::I8 {
                let layer = name
                    .strip_suffix(".weight")
                    .ok_or_else(|| Error::new(format!("unexpected INT8 tensor {name}")))?;
                if file.get(&format!("{prefix}{layer}.comfy_quant")).is_none()
                    || shape.len() != 2
                    || !shape[1].is_multiple_of(256)
                    || described
                        .get(format!("{layer}.weight_scale").as_str())
                        .is_none_or(|&(scale_dtype, scale_shape)| {
                            scale_dtype != DType::F32
                                || scale_shape.iter().product::<usize>() != shape[0]
                        })
                {
                    return Err(Error::new(format!(
                        "{name}: expected per-output INT8 ConvRot scales and metadata"
                    )));
                }
            }
        }

        drop(described);
        Ok(Self {
            device,
            tensors,
            loaded: RefCell::default(),
            absent,
            adapters: HashMap::new(),
            linear_precision: LinearPrecision::Fp32,
            attention_precision: AttentionPrecision::Fp32,
        })
    }

    /// A tensor on the device, whether it stays there or its unit was brought for this call.
    fn weight(&self, name: &str) -> Result<Weight> {
        if let Some(weight) = self.tensors.get(name) {
            return Ok(weight.clone());
        }
        if let Some(weight) = self.loaded.borrow().get(name) {
            return Ok(weight.clone());
        }
        Err(Error::new(match self.absent.contains_key(name) {
            true => format!("{name} is on the disk, and its unit is not on the device"),
            false => format!("missing tensor {name}"),
        }))
    }

    /// The type and shape of a tensor, wherever it is.
    fn describe(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        self.tensors
            .get(name)
            .map(|weight| (weight.dtype, weight.shape.clone()))
            .or_else(|| self.absent.get(name).cloned())
    }

    /// Brings tensors of a unit read on the way to the device, until `unload_unit` takes them.
    pub(crate) fn load_unit(&self, tensors: Vec<(String, Weight)>) {
        self.loaded.borrow_mut().extend(tensors);
    }

    pub(crate) fn unload_unit<'a>(&self, names: impl IntoIterator<Item = &'a str>) {
        let mut loaded = self.loaded.borrow_mut();
        for name in names {
            loaded.remove(name);
        }
    }

    /// Whether a layer's weights are INT8, which is what an exchanged block input reaches.
    pub fn is_int8(&self, name: &str) -> bool {
        self.describe(&format!("{name}.weight"))
            .is_some_and(|(dtype, _)| dtype == DType::I8)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name) || self.absent.contains_key(name)
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
        self.embedding_of(&self.weight(name)?, ids)
    }

    /// The rows `ids` of an embedding table.
    pub(crate) fn embedding_of(&self, w: &Weight, ids: &[u32]) -> Result<Array> {
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
        self.describe(name)
            .map(|(_, shape)| shape)
            .ok_or_else(|| Error::new(format!("missing tensor {name}")))
    }

    pub fn array(&self, name: &str) -> Result<Array> {
        let w = self.weight(name)?;
        let rows = if w.shape.len() >= 2 { w.shape[0] } else { 1 };
        let count = w.shape.iter().product::<usize>();
        convert(&self.device, &w.buffer, rows, count / rows, w.dtype)
    }

    /// FP16 attention runs on the matrix units, so a device without them refuses it here rather
    /// than quietly running FP32.
    pub fn set_attention_precision(&mut self, precision: AttentionPrecision) -> Result<()> {
        if precision == AttentionPrecision::Fp16 && !self.device.supports_tensor_ops() {
            return Err(Error::new(
                "FP16 Metal attention requires macOS 26 and Apple silicon".into(),
            ));
        }
        self.attention_precision = precision;
        Ok(())
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
        let w = self.weight(&key)?;
        if w.shape.len() < 2 || w.shape[1..].iter().product::<usize>() != x.cols {
            return Err(Error::new(format!("{name}: linear input shape mismatch")));
        }

        let bias = format!("{name}.bias");
        let bias = if self.contains(&bias) {
            Some(self.vector(&bias)?)
        } else {
            None
        };
        // An FP16 adapter's up projection is left to the product, which runs it a tile at a time.
        let (addend, lora) = match self.adapters.get(name) {
            None => (None, None),
            Some(Adapter::Fp16 { down, up, rank, .. }) => (
                None,
                Some(Lora {
                    mid: x.linear_half(down, *rank)?,
                    up,
                }),
            ),
            Some(adapter) => (Some(adapter.apply(x)?), None),
        };

        if w.dtype == DType::I8 {
            let rotated = x.rotate()?;
            let scales = self.vector(&format!("{name}.weight_scale"))?;
            let finish = Finish {
                scales: Some(&scales),
                bias: bias.as_ref(),
                addend,
                lora,
            };
            return if self.linear_precision == LinearPrecision::Fp32 {
                finish.apply(rotated.linear_packed(&w.buffer, w.shape[0], w.dtype)?)
            } else {
                rotated.linear_int8(&w.buffer, w.shape[0], self.linear_precision, finish)
            };
        }

        let result = if w.dtype != DType::F32 {
            x.linear_packed(&w.buffer, w.shape[0], w.dtype)?
        } else {
            x.linear(&self.array(&key)?)?
        };
        Finish {
            scales: None,
            bias: bias.as_ref(),
            addend,
            lora,
        }
        .apply(result)
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
        let w = self.weight(&key)?;
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
        // A LoRA's down projection runs on the rows this rank was handed, so the rows suffice. The
        // up projection answers one column per output, so it is cut the way the weight was.
        let cut_up;
        let (addend, lora) = match self.adapters.get(name) {
            None => (None, None),
            // FP32 weights are rotated to meet the rows: the product of two rotated sides is the
            // product of the unrotated pair.
            Some(Adapter::Fp32 { down, up, scale }) => (
                Some(
                    Array::from_quantized(&self.device, input, scales, rows, cols)?
                        .linear(&down.rotate()?)?
                        .linear(&self.kept_up(up, keep)?)?
                        .unary(0, *scale)?,
                ),
                None,
            ),
            // FP16 weights stay as they are and the rows are rotated back instead: ConvRot's
            // Hadamard transform is symmetric and orthogonal, so it undoes itself. The up
            // projection is left to the product, which runs it a tile at a time.
            Some(Adapter::Fp16 { down, up, rank, .. }) => {
                let up = if keep.is_empty() {
                    up
                } else {
                    cut_up = gather_rows(&self.device, up, rank * 2, keep)?;
                    &cut_up
                };
                let mid = Array::from_quantized(&self.device, input, scales, rows, cols)?
                    .rotate()?
                    .linear_half(down, *rank)?;
                (None, Some(Lora { mid, up }))
            }
        };
        let bias = format!("{name}.bias");
        let bias = if self.contains(&bias) {
            Some(self.kept_vector(&bias, keep)?)
        } else {
            None
        };
        let weight_scales = self.kept_vector(&format!("{name}.weight_scale"), keep)?;
        Array::linear_quantized(
            &self.device,
            input,
            scales,
            weight,
            rows,
            cols,
            outputs,
            self.linear_precision,
            Finish {
                scales: Some(&weight_scales),
                bias: bias.as_ref(),
                addend,
                lora,
            },
        )
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
                Adapter::new(
                    &self.device,
                    (&down.data, down.shape[0], down.shape[1]),
                    (&up.data, up.shape[0], up.shape[1]),
                    strength * alpha.data[0] / down.shape[0] as f32,
                    self.device.supports_tensor_ops(),
                )?,
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
            loaded: RefCell::default(),
            absent: HashMap::new(),
            adapters: HashMap::new(),
            linear_precision: LinearPrecision::Fp32,
            attention_precision: AttentionPrecision::Fp32,
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
            // Both roads of the adapter: FP32, and FP16 weights on the matrix units.
            let down: Vec<f32> = (0..inputs).map(|i| (i % 7) as f32 / 100.0).collect();
            let up = [0.5, -0.2, 0.1, 0.0, 1.0];
            let cases = [
                (LinearPrecision::Fp16, true),
                (LinearPrecision::Int8, true),
                (LinearPrecision::MpsFp16, true),
                (LinearPrecision::Int8, false),
            ];
            for (precision, half) in cases {
                w.adapters.insert(
                    "linear".into(),
                    Adapter::new(&device, (&down, 1, inputs), (&up, outputs, 1), 0.7, half)
                        .unwrap(),
                );
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
                    "{precision:?} ConvRot + LoRA (FP16 adapter {half}) relative L2 {relative}"
                );
            }

            // An exchanged input arrives rotated and quantized, and a rank may want a few of the
            // outputs. The FP16 adapter rotates the rows back where the FP32 one rotates its
            // weights, and the two must agree, whole or cut.
            use crate::shard::WholeExchange;
            use mmh3_core::shard::{Exchange, Region};
            let mut exchange = WholeExchange::new(&device);
            let region = exchange.region(Region::Normalized, rows * inputs).unwrap();
            let scales = region
                .write_quantized(0, &Array::from_f32(&device, rows, inputs, &x).unwrap())
                .unwrap();
            w.linear_precision = LinearPrecision::Int8;
            for keep in [vec![], vec![1..2, 3..5]] {
                let mut answers = Vec::new();
                for half in [false, true] {
                    w.adapters.insert(
                        "linear".into(),
                        Adapter::new(&device, (&down, 1, inputs), (&up, outputs, 1), 0.7, half)
                            .unwrap(),
                    );
                    answers.push(
                        w.linear_quantized(region.buffer(), &scales, rows, "linear", &keep)
                            .unwrap()
                            .to_f32()
                            .unwrap(),
                    );
                }
                let largest = answers[0].iter().fold(0.0f32, |m, v| m.max(v.abs()));
                for (a, b) in answers[0].iter().zip(&answers[1]) {
                    assert!(
                        (a - b).abs() <= largest * 2e-3,
                        "exchanged LoRA, keep {keep:?}: FP32 {a} against FP16 {b}"
                    );
                }
            }
        }
    }

    /// What an adapter adds, alone: the layer with it less the layer without it, over several
    /// tiles of the product and a rank that is not a whole tile, on every road.
    #[test]
    fn a_lora_adds_what_its_two_products_make() {
        let device = Device::new().unwrap();
        let (rows, inputs, outputs, rank) = (200, 512, 130, 24);
        let x: Vec<f32> = (0..rows * inputs)
            .map(|i| ((i * 13 % 97) as f32 - 48.0) / 31.0)
            .collect();
        let weight: Vec<u8> = (0..inputs * outputs)
            .map(|i| ((i * 7 % 255) as i32 - 127) as i8 as u8)
            .collect();
        let scales: Vec<f32> = (0..outputs).map(|i| 0.001 + i as f32 * 1e-5).collect();
        let down: Vec<f32> = (0..rank * inputs)
            .map(|i| ((i * 11 % 23) as f32 - 11.0) / 40.0)
            .collect();
        let up: Vec<f32> = (0..outputs * rank)
            .map(|i| ((i * 5 % 17) as f32 - 8.0) / 30.0)
            .collect();
        let mut tensors = HashMap::new();
        tensors.insert(
            "linear.weight".into(),
            Weight {
                buffer: device.alloc(weight.len(), Some(&weight)).unwrap(),
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
            loaded: RefCell::default(),
            absent: HashMap::new(),
            adapters: HashMap::new(),
            linear_precision: LinearPrecision::Fp32,
            attention_precision: AttentionPrecision::Fp32,
        };
        let input = Array::from_f32(&device, rows, inputs, &x).unwrap();
        let expected: Vec<f64> = (0..rows)
            .flat_map(|r| {
                let hidden: Vec<f64> = (0..rank)
                    .map(|k| {
                        (0..inputs)
                            .map(|i| x[r * inputs + i] as f64 * down[k * inputs + i] as f64)
                            .sum()
                    })
                    .collect();
                let up = &up;
                (0..outputs).map(move |n| {
                    (0..rank)
                        .map(|k| hidden[k] * up[n * rank + k] as f64 * 0.5)
                        .sum::<f64>()
                })
            })
            .collect();
        let largest = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));

        let tensor_ops = device.supports_tensor_ops();
        for (precision, half) in [
            (LinearPrecision::Fp32, false),
            (LinearPrecision::MpsFp16, false),
            (LinearPrecision::MpsFp16, tensor_ops),
            (LinearPrecision::Fp16, tensor_ops),
            (LinearPrecision::Int8, tensor_ops),
        ] {
            if matches!(precision, LinearPrecision::Fp16 | LinearPrecision::Int8) && !tensor_ops {
                continue;
            }
            w.linear_precision = precision;
            w.adapters.clear();
            let without = w.linear(&input, "linear").unwrap().to_f32().unwrap();
            w.adapters.insert(
                "linear".into(),
                Adapter::new(
                    &device,
                    (&down, rank, inputs),
                    (&up, outputs, rank),
                    0.5,
                    half,
                )
                .unwrap(),
            );
            let with = w.linear(&input, "linear").unwrap().to_f32().unwrap();
            for (index, want) in expected.iter().enumerate() {
                let got = (with[index] - without[index]) as f64;
                assert!(
                    (got - want).abs() <= largest * 3e-3,
                    "{precision:?}, FP16 adapter {half}: at {index} the LoRA added {got}, not {want}"
                );
            }
        }
    }

    /// ConvRot's Hadamard transform is symmetric as well as orthogonal, so it undoes itself. An
    /// FP16 LoRA leans on that to take back the rows an exchange delivers rotated.
    #[test]
    fn convrot_undoes_itself() {
        let device = Device::new().unwrap();
        let values: Vec<f32> = (0..3 * 512)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let twice = Array::from_f32(&device, 3, 512, &values)
            .unwrap()
            .rotate()
            .unwrap()
            .rotate()
            .unwrap()
            .to_f32()
            .unwrap();
        for (a, b) in values.iter().zip(&twice) {
            assert!((a - b).abs() < 1e-5, "{a} came back as {b}");
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
