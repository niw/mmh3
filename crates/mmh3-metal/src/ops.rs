//! FP32 tensor operations, with packed low-precision inputs used inside matrix products.
use crate::{Buffer, Device, Error, LinearPrecision, Result, check, mmh3_metal_matmul};
use mmh3_core::safetensors::DType;

#[derive(Clone)]
pub struct Array {
    pub(crate) buffer: Buffer,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}
pub(crate) struct RowMap {
    buffer: Buffer,
    len: usize,
    maximum: usize,
}

impl RowMap {
    pub fn new(device: &Device, rows: &[usize]) -> Result<Self> {
        let maximum = *rows
            .iter()
            .max()
            .ok_or_else(|| Error::new("empty row map".into()))?;
        if maximum > u32::MAX as usize {
            return Err(Error::new("row index too large".into()));
        }

        let bytes: Vec<_> = rows
            .iter()
            .flat_map(|&i| (i as u32).to_ne_bytes())
            .collect();
        Ok(Self {
            buffer: device.alloc(bytes.len(), Some(&bytes))?,
            len: rows.len(),
            maximum,
        })
    }
}

fn size(rows: usize, cols: usize) -> Result<usize> {
    let n = rows
        .checked_mul(cols)
        .filter(|&n| n > 0 && n <= u32::MAX as usize)
        .ok_or_else(|| Error::new("Metal array shape is empty or too large".into()))?;
    n.checked_mul(4)
        .ok_or_else(|| Error::new("Metal allocation size overflow".into()))
}

impl Array {
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn from_f32(device: &Device, rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        let bytes = size(rows, cols)?;
        if data.len() != bytes / 4 {
            return Err(Error::new("array data does not match its shape".into()));
        }

        // SAFETY: every f32 consists of four initialized bytes, read only during allocation.
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), bytes) };
        Ok(Self {
            buffer: device.alloc(bytes, Some(data))?,
            rows,
            cols,
        })
    }

    pub fn zeros(device: &Device, rows: usize, cols: usize) -> Result<Self> {
        let out = Self::empty(device, rows, cols)?;
        device.run(
            "fill_zero",
            &[&out.buffer],
            &[out.len() as u32],
            out.len(),
            false,
        )?;
        Ok(out)
    }

    pub(crate) fn empty(device: &Device, rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            buffer: device.alloc(size(rows, cols)?, None)?,
            rows,
            cols,
        })
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.buffer.to_f32()
    }

    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub(crate) fn device(&self) -> &Device {
        &self.buffer.0.device
    }

    pub(crate) fn reshape(&self, rows: usize, cols: usize) -> Result<Self> {
        if size(rows, cols)? / 4 != self.len() {
            return Err(Error::new("reshape changes the number of values".into()));
        }

        Ok(Self {
            buffer: self.buffer.clone(),
            rows,
            cols,
        })
    }

    pub fn linear(&self, weight: &Self) -> Result<Self> {
        if self.cols != weight.cols || !std::sync::Arc::ptr_eq(&self.device().0, &weight.device().0)
        {
            return Err(Error::new("matrix product shapes or devices differ".into()));
        }

        let out = Self::empty(self.device(), self.rows, weight.rows)?;
        // SAFETY: input, weight and output extents match M×K, N×K, M×N. All share a device.
        check(unsafe {
            mmh3_metal_matmul(
                self.device().0.0.as_ptr(),
                self.buffer.0.pointer.as_ptr(),
                weight.buffer.0.pointer.as_ptr(),
                out.buffer.0.pointer.as_ptr(),
                self.rows,
                weight.rows,
                self.cols,
                weight.rows,
                0,
                false,
            )
        })?;
        Ok(out)
    }

    /// Expand packed weights in reusable FP32 slabs of at most 32 MiB (or one weight row).
    /// Consecutive encoders on the same queue order scratch writes after the previous MPS read.
    pub(crate) fn linear_packed(
        &self,
        weight: &Buffer,
        outputs: usize,
        dtype: DType,
    ) -> Result<Self> {
        self.linear_packed_slab(
            weight,
            outputs,
            dtype,
            (32 * 1024 * 1024 / 4 / self.cols).max(1),
        )
    }

    pub(crate) fn linear_packed_slab(
        &self,
        weight: &Buffer,
        outputs: usize,
        dtype: DType,
        slab_rows: usize,
    ) -> Result<Self> {
        if !std::sync::Arc::ptr_eq(&self.device().0, &weight.0.device.0)
            || outputs
                .checked_mul(self.cols)
                .and_then(|n| n.checked_mul(dtype.size_in_bytes()))
                != Some(weight.0.bytes)
            || slab_rows == 0
        {
            return Err(Error::new("packed matrix shape or device mismatch".into()));
        }

        size(outputs, self.cols)?;
        let out = Self::empty(self.device(), self.rows, outputs)?;
        let scratch = Self::empty(self.device(), slab_rows.min(outputs), self.cols)?;
        for first in (0..outputs).step_by(slab_rows) {
            let count = slab_rows.min(outputs - first);
            self.device().run(
                "convert_float",
                &[weight, &scratch.buffer],
                &[
                    (count * self.cols) as u32,
                    dtype_code(dtype)?,
                    (first * self.cols) as u32,
                ],
                count * self.cols,
                false,
            )?;
            // SAFETY: each product writes a disjoint column slab in the full output matrix.
            // The command buffer retains scratch until all encoded reads complete.
            check(unsafe {
                mmh3_metal_matmul(
                    self.device().0.0.as_ptr(),
                    self.buffer.0.pointer.as_ptr(),
                    scratch.buffer.0.pointer.as_ptr(),
                    out.buffer.0.pointer.as_ptr(),
                    self.rows,
                    count,
                    self.cols,
                    outputs,
                    first,
                    false,
                )
            })?;
        }

        Ok(out)
    }

    /// MPP reads checkpoint INT8 bytes directly; only the current activation is packed.
    pub(crate) fn linear_int8(
        &self,
        weight: &Buffer,
        outputs: usize,
        precision: LinearPrecision,
    ) -> Result<Self> {
        if !std::sync::Arc::ptr_eq(&self.device().0, &weight.0.device.0)
            || outputs.checked_mul(self.cols) != Some(weight.0.bytes)
        {
            return Err(Error::new("packed matrix shape or device mismatch".into()));
        }

        size(outputs, self.cols)?;
        // TensorOps uses signed extents/indices. Bound the INT32 worst-case sum too:
        // quantized input is [-127,127], checkpoint weights can contain -128.
        if precision == LinearPrecision::Fp32
            || self.len() > i32::MAX as usize
            || outputs * self.cols > i32::MAX as usize
            || self
                .rows
                .checked_mul(outputs)
                .is_none_or(|n| n > i32::MAX as usize)
            || (precision == LinearPrecision::Int8 && self.cols > i32::MAX as usize / (127 * 128))
        {
            return self.linear_packed(weight, outputs, DType::I8);
        }

        let int8 = precision == LinearPrecision::Int8;
        let packed = self
            .device()
            .alloc(self.len() * if int8 { 1 } else { 2 }, None)?;
        let scales = Self::empty(self.device(), self.rows, 1)?;
        self.device().run(
            "pack_linear_input",
            &[&self.buffer, &packed, &scales.buffer],
            &[self.cols as u32, int8 as u32],
            self.rows,
            true,
        )?;
        Self::product_of_packed(
            self.device(),
            &packed,
            &scales,
            weight,
            self.rows,
            self.cols,
            outputs,
            precision,
        )
    }

    /// The product of an input that another rank already rotated and quantized: INT8 values with
    /// one FP32 scale a row, as the exchange carries them. A rank must not quantize these again.
    /// The rounding is not idempotent, so two ranks projecting the same tokens would disagree
    /// about them, and a block's attention would see two different sequences.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_quantized(
        device: &Device,
        input: &Buffer,
        scales: &Self,
        weight: &Buffer,
        rows: usize,
        cols: usize,
        outputs: usize,
        precision: LinearPrecision,
    ) -> Result<Self> {
        let values = size(rows, cols)? / 4;
        if input.0.bytes < values
            || scales.rows != rows
            || scales.cols != 1
            || outputs.checked_mul(cols) != Some(weight.0.bytes)
        {
            return Err(Error::new("quantized input shape mismatch".into()));
        }

        // The packed roads take the INT8 values as they are: a value in [-127, 127] is exact as a
        // half, and the scale is applied to the rows of the result either way.
        if precision != LinearPrecision::Fp32 {
            let packed = if precision == LinearPrecision::Int8 {
                input.clone()
            } else {
                let halves = device.alloc(values * 2, None)?;
                device.run(
                    "int8_to_half",
                    &[input, &halves],
                    &[values as u32, 0],
                    values,
                    false,
                )?;
                halves
            };
            return Self::product_of_packed(
                device, &packed, scales, weight, rows, cols, outputs, precision,
            );
        }

        Self::from_quantized(device, input, scales, rows, cols)?.linear_packed(
            weight,
            outputs,
            DType::I8,
        )
    }

    /// An input another rank rotated and quantized, back in FP32 and still rotated. Rotated is
    /// what a rotated weight wants: rotating one side alone is a different product, and one that
    /// runs.
    pub(crate) fn from_quantized(
        device: &Device,
        input: &Buffer,
        scales: &Self,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let values = size(rows, cols)? / 4;
        if input.0.bytes < values || scales.rows != rows || scales.cols != 1 {
            return Err(Error::new("quantized input shape mismatch".into()));
        }

        let restored = Self::empty(device, rows, cols)?;
        device.run(
            "dequantize_rows",
            &[input, &scales.buffer, &restored.buffer],
            &[values as u32, cols as u32],
            values,
            false,
        )?;
        Ok(restored)
    }

    /// The product once the input is packed for `precision`: INT8 values where it is INT8 and
    /// halves otherwise, with one scale a row beside them. The scale is applied to the rows of the
    /// result, so either packing answers with the same values.
    #[allow(clippy::too_many_arguments)]
    fn product_of_packed(
        device: &Device,
        packed: &Buffer,
        scales: &Self,
        weight: &Buffer,
        rows: usize,
        cols: usize,
        outputs: usize,
        precision: LinearPrecision,
    ) -> Result<Self> {
        let out = Self::empty(device, rows, outputs)?;
        if precision == LinearPrecision::MpsFp16 {
            let slab_rows = (32 * 1024 * 1024 / 2 / cols).max(1);
            let scratch = device.alloc(slab_rows.min(outputs) * cols * 2, None)?;
            for first in (0..outputs).step_by(slab_rows) {
                let count = slab_rows.min(outputs - first);
                device.run(
                    "int8_to_half",
                    &[weight, &scratch],
                    &[(count * cols) as u32, (first * cols) as u32],
                    count * cols,
                    false,
                )?;
                // SAFETY: FP16 inputs cover M×K and the current N×K slab; output is FP32.
                check(unsafe {
                    mmh3_metal_matmul(
                        device.0.0.as_ptr(),
                        packed.0.pointer.as_ptr(),
                        scratch.0.pointer.as_ptr(),
                        out.buffer.0.pointer.as_ptr(),
                        rows,
                        count,
                        cols,
                        outputs,
                        first,
                        true,
                    )
                })?;
            }

            device.run(
                "scale_linear_rows",
                &[&out.buffer, &scales.buffer],
                &[out.len() as u32, outputs as u32],
                out.len(),
                false,
            )?;
            return Ok(out);
        }

        device.run(
            if precision == LinearPrecision::Int8 {
                "mpp_int8"
            } else {
                "mpp_fp16"
            },
            &[packed, weight, &out.buffer, &scales.buffer],
            &[rows as u32, outputs as u32, cols as u32],
            rows.div_ceil(64) * outputs.div_ceil(64),
            true,
        )?;
        Ok(out)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 0)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 1)
    }

    fn binary(&self, rhs: &Self, operation: u32) -> Result<Self> {
        if rhs.cols != self.cols || !(rhs.rows == 1 || rhs.rows == self.rows) {
            return Err(Error::new("incompatible elementwise shapes".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "binary_op",
            &[&self.buffer, &rhs.buffer, &out.buffer],
            &[self.len() as u32, rhs.len() as u32, operation],
            self.len(),
            false,
        )?;
        Ok(out)
    }

    pub fn unary(&self, operation: u32, scale: f32) -> Result<Self> {
        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "unary_op",
            &[&self.buffer, &out.buffer],
            &[self.len() as u32, operation, scale.to_bits()],
            self.len(),
            false,
        )?;
        Ok(out)
    }

    pub fn slice(&self, row: usize, rows: usize, col: usize, cols: usize) -> Result<Self> {
        if row.checked_add(rows).is_none_or(|end| end > self.rows)
            || col.checked_add(cols).is_none_or(|end| end > self.cols)
        {
            return Err(Error::new("slice is outside the array".into()));
        }

        let out = Self::empty(self.device(), rows, cols)?;
        self.device().run(
            "slice_rows",
            &[&self.buffer, &out.buffer],
            &[
                out.len() as u32,
                self.cols as u32,
                cols as u32,
                row as u32,
                col as u32,
            ],
            out.len(),
            false,
        )?;
        Ok(out)
    }

    pub fn concat(parts: &[Self], columns: bool) -> Result<Self> {
        let first = parts
            .first()
            .ok_or_else(|| Error::new("cannot concatenate no arrays".into()))?;
        let (rows, cols) = if columns {
            if parts.iter().any(|p| p.rows != first.rows) {
                return Err(Error::new("concat row mismatch".into()));
            }

            (first.rows, parts.iter().map(|p| p.cols).sum())
        } else {
            if parts.iter().any(|p| p.cols != first.cols) {
                return Err(Error::new("concat column mismatch".into()));
            }

            (parts.iter().map(|p| p.rows).sum(), first.cols)
        };

        let out = Self::empty(first.device(), rows, cols)?;
        let mut offset = 0;
        for p in parts {
            first.device().run(
                "copy_rows",
                &[&p.buffer, &out.buffer],
                &[
                    p.len() as u32,
                    p.cols as u32,
                    cols as u32,
                    if columns { 0 } else { offset },
                    if columns { offset } else { 0 },
                ],
                p.len(),
                false,
            )?;
            offset += if columns {
                p.cols as u32
            } else {
                p.rows as u32
            };
        }

        Ok(out)
    }

    /// RMS normalization, or LayerNorm when center is true.
    pub fn norm(&self, weight: &Self, epsilon: f32, center: bool) -> Result<Self> {
        if weight.len() != self.cols {
            return Err(Error::new(
                "normalization weight has the wrong width".into(),
            ));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "normalize",
            &[&self.buffer, &weight.buffer, &out.buffer],
            &[self.cols as u32, epsilon.to_bits(), center as u32],
            self.rows,
            true,
        )?;
        Ok(out)
    }

    pub fn rope(&self, heads: usize, angles: &Self) -> Result<Self> {
        if heads == 0
            || !self.cols.is_multiple_of(heads)
            || angles.rows != self.rows
            || angles.cols * 2 > self.cols / heads
        {
            return Err(Error::new("invalid rotary embedding shape".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "rotary",
            &[&self.buffer, &angles.buffer, &out.buffer],
            &[
                self.len() as u32,
                self.cols as u32,
                (self.cols / heads) as u32,
                angles.cols as u32,
            ],
            self.len(),
            false,
        )?;
        Ok(out)
    }

    /// Online softmax: memory is O(tokens × heads × head_dim), never O(tokens²).
    pub fn attention(
        &self,
        key: &Self,
        value: &Self,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Self> {
        if heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || !self.cols.is_multiple_of(heads)
            || self.cols / heads > 256
            || key.rows != value.rows
            || key.cols != value.cols
            || key.cols != kv_heads * (self.cols / heads)
            || (causal && self.rows != key.rows)
        {
            return Err(Error::new("unsupported attention layout".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            &format!(
                "attention_{}",
                (self.cols / heads).next_power_of_two().max(32)
            ),
            &[&self.buffer, &key.buffer, &value.buffer, &out.buffer],
            &[
                key.rows as u32,
                heads as u32,
                kv_heads as u32,
                (self.cols / heads) as u32,
                causal as u32,
                self.rows as u32,
            ],
            self.rows.div_ceil(8) * heads,
            true,
        )?;
        Ok(out)
    }

    pub fn swiglu(&self) -> Result<Self> {
        if !self.cols.is_multiple_of(2) {
            return Err(Error::new("SwiGLU requires two equal halves".into()));
        }

        self.slice(0, self.rows, 0, self.cols / 2)?
            .unary(1, 1.0)?
            .mul(&self.slice(0, self.rows, self.cols / 2, self.cols / 2)?)
    }

    pub(crate) fn modulate(
        &self,
        m: &Self,
        rows: &RowMap,
        shift: usize,
        scale: usize,
    ) -> Result<Self> {
        self.modulation(m, rows, None, shift, scale)
    }

    pub(crate) fn add_gated(
        &self,
        delta: &Self,
        m: &Self,
        rows: &RowMap,
        gate: usize,
    ) -> Result<Self> {
        self.modulation(m, rows, Some(delta), gate, gate)
    }

    fn modulation(
        &self,
        m: &Self,
        rows: &RowMap,
        delta: Option<&Self>,
        a: usize,
        b: usize,
    ) -> Result<Self> {
        if rows.len != self.rows
            || rows.maximum >= m.rows
            || !m.cols.is_multiple_of(self.cols)
            || a >= m.cols / self.cols
            || b >= m.cols / self.cols
            || delta.is_some_and(|d| d.shape() != self.shape())
        {
            return Err(Error::new("invalid modulation shapes".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "modulation",
            &[
                &self.buffer,
                &m.buffer,
                &rows.buffer,
                &delta.unwrap_or(self).buffer,
                &out.buffer,
            ],
            &[
                self.len() as u32,
                self.cols as u32,
                m.cols as u32,
                a as u32,
                b as u32,
                delta.is_some() as u32,
            ],
            self.len(),
            false,
        )?;
        Ok(out)
    }

    pub(crate) fn rotate(&self) -> Result<Self> {
        if !self.cols.is_multiple_of(256) {
            return Err(Error::new(
                "ConvRot requires a multiple of 256 features".into(),
            ));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "hadamard",
            &[&self.buffer, &out.buffer],
            &[0],
            self.len() / 256,
            true,
        )?;
        Ok(out)
    }
}
pub(crate) fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::I8 => Ok(3),
        _ => Err(Error::new(format!(
            "unsupported Metal tensor dtype {dtype}"
        ))),
    }
}
pub(crate) fn convert(
    device: &Device,
    buffer: &Buffer,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Array> {
    let count = size(rows, cols)? / 4;
    if count.checked_mul(dtype.size_in_bytes()) != Some(buffer.0.bytes) {
        return Err(Error::new(
            "weight shape does not match its allocation".into(),
        ));
    }

    if dtype == DType::F32 {
        return Ok(Array {
            buffer: buffer.clone(),
            rows,
            cols,
        });
    }

    let out = Array::empty(device, rows, cols)?;
    device.run(
        "convert_float",
        &[buffer, &out.buffer],
        &[out.len() as u32, dtype_code(dtype)?, 0],
        out.len(),
        false,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmh3_core::numeric::{f32_to_bf16, f32_to_f16};

    #[test]
    #[ignore = "real-model-size product timing; run alone on an idle GPU in release mode"]
    fn profile_packed_products() {
        use std::time::Instant;
        let device = Device::new().unwrap();
        eprintln!(
            "{}; includes input packing, weight expansion and scale restoration",
            device.name()
        );
        for (rows, outputs, cols) in [(1510, 21504, 5376), (1510, 5376, 14336), (1510, 5376, 7168)]
        {
            let input: Vec<f32> = (0..rows * cols)
                .map(|i| (i % 101) as f32 / 100.0 - 0.5)
                .collect();
            let x = Array::from_f32(&device, rows, cols, &input).unwrap();
            let bytes: Vec<u8> = (0..outputs * cols).map(|i| (i % 256) as u8).collect();
            let weights = device.alloc(bytes.len(), Some(&bytes)).unwrap();

            for precision in [
                LinearPrecision::Fp32,
                LinearPrecision::MpsFp16,
                LinearPrecision::Fp16,
                LinearPrecision::Int8,
            ] {
                if matches!(precision, LinearPrecision::Fp16 | LinearPrecision::Int8)
                    && !device.supports_tensor_ops()
                {
                    continue;
                }

                let mut times = Vec::new();
                for trial in 0..6 {
                    device.synchronize().unwrap();
                    let start = Instant::now();
                    let result = x.linear_int8(&weights, outputs, precision).unwrap();
                    device.synchronize().unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    std::hint::black_box(result);

                    if trial > 0 {
                        times.push(elapsed);
                    }
                }

                times.sort_by(f64::total_cmp);
                eprintln!(
                    "{precision:?} [{rows},{outputs},{cols}]: median {:.3} ms",
                    times[2]
                );
            }
        }
    }

    /// ConvRot is orthogonal, so rotating both sides of a product leaves the product alone. That
    /// is what lets a rank hold rotated weights and feed them the rotated activations the exchange
    /// delivers. Rotating one side alone would still run and still produce numbers.
    #[test]
    fn rotating_both_sides_of_a_product_leaves_it_alone() {
        let device = Device::new().unwrap();
        let (rows, cols, outputs) = (4usize, 512usize, 8usize);
        let x = Array::from_f32(
            &device,
            rows,
            cols,
            &(0..rows * cols)
                .map(|i| ((i * 31) % 251) as f32 / 251.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let w = Array::from_f32(
            &device,
            outputs,
            cols,
            &(0..outputs * cols)
                .map(|i| ((i * 17) % 173) as f32 / 173.0 - 0.5)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let plain = x.linear(&w).unwrap().to_f32().unwrap();
        let both = x
            .rotate()
            .unwrap()
            .linear(&w.rotate().unwrap())
            .unwrap()
            .to_f32()
            .unwrap();
        let largest = plain.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (index, (&found, &want)) in both.iter().zip(plain.iter()).enumerate() {
            assert!(
                (found - want).abs() <= largest * 1e-5,
                "value {index}: {found} against {want}"
            );
        }

        // Rotating one side alone is a different product, which is the failure this guards.
        let one = x.rotate().unwrap().linear(&w).unwrap().to_f32().unwrap();
        assert!(
            one.iter()
                .zip(plain.iter())
                .any(|(&found, &want)| (found - want).abs() > largest * 1e-3),
            "rotating one side changed nothing, so this test proves nothing"
        );
    }

    /// A LoRA's down projection runs on the rows the exchange delivers, which arrive rotated, so
    /// its weights are rotated to meet them. Both sides use the same quantized rows here, so the
    /// quantization cancels out of the comparison and what is left is the rotation alone: turning
    /// the rows back and using the plain weights has to give what the rotated weights give.
    #[test]
    fn an_adapter_meets_the_exchanged_rows_in_the_rotated_frame() {
        let device = Device::new().unwrap();
        let (rows, cols, rank, outputs) = (4usize, 512usize, 8usize, 16usize);
        let make = |r: usize, c: usize, step: usize, modulus: usize| {
            Array::from_f32(
                &device,
                r,
                c,
                &(0..r * c)
                    .map(|i| ((i * step) % modulus) as f32 / modulus as f32 - 0.5)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let x = make(rows, cols, 31, 251);
        let down = make(rank, cols, 17, 173);
        let up = make(outputs, rank, 11, 97);

        // NOTE: both sides below go through the same quantization so that it cancels out and only
        // the rotation is under test. Do not turn this into a measurement of what quantizing
        // costs: ConvRot spreads a row towards its largest value, which helps real activations
        // and hurts flat values like these, so on this data it makes INT8 several times worse.
        let rotated = x.rotate().unwrap();
        let packed = device.alloc(rows * cols, None).unwrap();
        let scales = Array::empty(&device, rows, 1).unwrap();
        device
            .run(
                "pack_linear_input",
                &[&rotated.buffer, &packed, &scales.buffer],
                &[cols as u32, 1],
                rows,
                true,
            )
            .unwrap();
        let delivered = Array::from_quantized(&device, &packed, &scales, rows, cols).unwrap();

        let through_rotated_weights = delivered
            .linear(&down.rotate().unwrap())
            .unwrap()
            .linear(&up)
            .unwrap()
            .to_f32()
            .unwrap();
        // The same rows turned back, against the weights as they are. ConvRot is its own inverse.
        let through_plain_weights = delivered
            .rotate()
            .unwrap()
            .linear(&down)
            .unwrap()
            .linear(&up)
            .unwrap()
            .to_f32()
            .unwrap();

        let largest = through_plain_weights
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()));
        for (index, (&found, &want)) in through_rotated_weights
            .iter()
            .zip(through_plain_weights.iter())
            .enumerate()
        {
            assert!(
                (found - want).abs() <= largest * 1e-4,
                "value {index}: {found} against {want}, largest {largest}"
            );
        }

        // What the quantization costs, reported rather than asserted: it is the same INT8 the
        // exchange carries every row in, and CUDA's adapter consumes it too.
        let exact = x
            .linear(&down)
            .unwrap()
            .linear(&up)
            .unwrap()
            .to_f32()
            .unwrap();
        let worst = through_rotated_weights
            .iter()
            .zip(exact.iter())
            .fold(0.0f32, |m, (&found, &want)| m.max((found - want).abs()));
        eprintln!("quantization costs {worst:.6} of {largest:.6}");
    }

    /// The exchange carries a block's input already rotated and quantized, so a rank projecting a
    /// peer's rows works from INT8 values it did not produce. Every road has to agree about what
    /// those values mean, or two ranks would see two different sequences.
    #[test]
    fn a_quantized_input_projects_the_same_on_every_road() {
        let device = Device::new().unwrap();
        let (rows, cols, outputs) = (5usize, 128usize, 64usize);

        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37) % 211) as f32 / 211.0 - 0.5)
            .collect();
        let x = Array::from_f32(&device, rows, cols, &values).unwrap();
        let packed = device.alloc(rows * cols, None).unwrap();
        let scales = Array::empty(&device, rows, 1).unwrap();
        device
            .run(
                "pack_linear_input",
                &[&x.buffer, &packed, &scales.buffer],
                &[cols as u32, 1],
                rows,
                true,
            )
            .unwrap();

        let weight_bytes: Vec<u8> = (0..outputs * cols)
            .map(|i| (((i * 53) % 255) as i32 - 127) as i8 as u8)
            .collect();
        let weight = device
            .alloc(weight_bytes.len(), Some(&weight_bytes))
            .unwrap();

        // What the INT8 values mean: the integer product of a row and a column, times the row's
        // scale. Read the quantized rows back rather than recomputing what the kernel chose.
        let quantized: Vec<f32> = packed.to_f32().unwrap();
        let bytes: Vec<i8> = quantized
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .map(|byte| byte as i8)
            .collect();
        let row_scales = scales.to_f32().unwrap();
        let mut expected = Vec::with_capacity(rows * outputs);
        for row in 0..rows {
            for output in 0..outputs {
                let mut sum = 0i32;
                for column in 0..cols {
                    sum += bytes[row * cols + column] as i32
                        * weight_bytes[output * cols + column] as i8 as i32;
                }
                expected.push(sum as f32 * row_scales[row]);
            }
        }

        for precision in [
            LinearPrecision::Fp32,
            LinearPrecision::MpsFp16,
            LinearPrecision::Fp16,
            LinearPrecision::Int8,
        ] {
            if matches!(precision, LinearPrecision::Fp16 | LinearPrecision::Int8)
                && !device.supports_tensor_ops()
            {
                continue;
            }

            let found = Array::linear_quantized(
                &device, &packed, &scales, &weight, rows, cols, outputs, precision,
            )
            .unwrap()
            .to_f32()
            .unwrap();
            let largest = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (index, &want) in expected.iter().enumerate() {
                assert!(
                    (found[index] - want).abs() <= largest * 2e-3,
                    "{precision:?} at {index}: {} against {want}",
                    found[index]
                );
            }
        }
    }

    #[test]
    fn tensor_ops_match_independent_quantized_products() {
        use mmh3_core::numeric::f16_to_f32;
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }

        for (rows, outputs, cols) in [(3, 5, 37), (65, 73, 257), (3, 7, 5376)] {
            let input: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let r = i / cols;
                    if r == 0 {
                        0.0
                    } else {
                        let range = if r == 1 { 1e8 } else { 1e-8 };
                        (((i * 19 + 7) % 997) as f32 - 498.0) / 498.0 * range
                    }
                })
                .collect();
            let weights: Vec<i8> = (0..outputs * cols)
                .map(|i| ((i * 29 % 256) as i32 - 128) as i8)
                .collect();
            let bytes: Vec<u8> = weights.iter().map(|&v| v as u8).collect();
            let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
            let x = Array::from_f32(&device, rows, cols, &input).unwrap();

            for precision in [
                LinearPrecision::Fp16,
                LinearPrecision::Int8,
                LinearPrecision::MpsFp16,
            ] {
                let actual = x
                    .linear_int8(&weight, outputs, precision)
                    .unwrap()
                    .to_f32()
                    .unwrap();
                let mut error = 0.0;
                let mut norm = 0.0;
                for r in 0..rows {
                    let maximum = input[r * cols..(r + 1) * cols]
                        .iter()
                        .fold(0.0f32, |a, v| a.max(v.abs()));
                    let scale = if maximum == 0.0 { 1.0 } else { maximum };
                    for n in 0..outputs {
                        let mut expected = 0.0f64;
                        let mut reference = 0.0f64;
                        let mut magnitude = 0.0f64;
                        for k in 0..cols {
                            let normalized = input[r * cols + k] / scale;
                            let q = if precision == LinearPrecision::Int8 {
                                (normalized * 127.0).clamp(-127.0, 127.0).round_ties_even()
                            } else {
                                f16_to_f32(f32_to_f16(normalized))
                            };

                            let term = q as f64 * weights[n * cols + k] as f64;
                            expected += term;
                            magnitude += term.abs();
                            reference += input[r * cols + k] as f64 * weights[n * cols + k] as f64;
                        }

                        let multiplier = if precision == LinearPrecision::Int8 {
                            scale / 127.0
                        } else {
                            scale
                        };

                        expected *= multiplier as f64;
                        magnitude *= multiplier as f64;
                        let value = actual[r * outputs + n] as f64;
                        assert!(value.is_finite());
                        assert!(
                            (value - expected).abs() <= magnitude * 3e-6 + 1e-15,
                            "{precision:?} [{rows},{outputs},{cols}] [{r},{n}]: {value} != {expected}"
                        );
                        error += (value - reference).powi(2);
                        norm += reference.powi(2);
                    }
                }

                let relative = (error / norm).sqrt();
                eprintln!("{precision:?} [{rows},{outputs},{cols}]: relative L2 {relative:.6}");
                assert!(
                    relative
                        < if precision == LinearPrecision::Int8 {
                            0.02
                        } else {
                            0.001
                        }
                );
            }
        }
    }

    #[test]
    fn packed_products_match_dense_across_partial_slabs() {
        let device = Device::new().unwrap();
        // More than 64 encoders with one-row slabs crosses a batch boundary.
        let (rows, outputs, cols) = (7, 73, 37);
        let values: Vec<_> = (0..outputs * cols).map(|i| (i % 17) as f32 - 8.0).collect();
        let input: Vec<_> = (0..rows * cols).map(|i| (i % 13) as f32 / 16.0).collect();
        let x = Array::from_f32(&device, rows, cols, &input).unwrap();
        let dense = Array::from_f32(&device, outputs, cols, &values).unwrap();
        let expected = x.linear(&dense).unwrap().to_f32().unwrap();

        for dtype in [DType::F16, DType::BF16, DType::I8] {
            let bytes: Vec<u8> = match dtype {
                DType::F16 => values
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_ne_bytes())
                    .collect(),
                DType::BF16 => values
                    .iter()
                    .flat_map(|&v| f32_to_bf16(v).to_ne_bytes())
                    .collect(),
                _ => values.iter().map(|&v| v as i8 as u8).collect(),
            };

            let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
            for slab in [1, 4, 13, 32] {
                let actual = x
                    .linear_packed_slab(&weight, outputs, dtype, slab)
                    .unwrap()
                    .to_f32()
                    .unwrap();
                assert_eq!(actual, expected, "{dtype:?}, slab={slab}");
            }
        }
    }
}
