//! FP32 tensor operations, with packed low-precision inputs used inside matrix products.
use crate::{
    AttentionPrecision, Buffer, Device, Error, LinearPrecision, Result, check, mmh3_metal_matmul,
};
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

/// Queries one threadgroup of `flash_attention` answers.
const TENSOR_ATTENTION_QUERIES: usize = 64;

/// Output rows and columns one threadgroup of `mpp_int8` or `mpp_fp16` answers.
const PRODUCT_TILE: (usize, usize) = (128, 64);

/// What a product of INT8 weights does to its result besides restoring the rows' scales. Each
/// step is a pass over the whole output when it runs on its own, so the products on the matrix
/// units and the MPS product's one pass after it take them all at once.
#[derive(Default)]
pub(crate) struct Finish<'a> {
    /// One scale an output: the weight's own.
    pub scales: Option<&'a Array>,
    /// One bias an output.
    pub bias: Option<&'a Array>,
    /// A result of the product's shape to add, such as a LoRA's. The products on the matrix units
    /// write theirs over it.
    pub addend: Option<Array>,
    /// A LoRA whose up projection is still to run. The products on the matrix units run it a tile
    /// at a time, so its output is never written whole.
    pub lora: Option<Lora<'a>>,
}

/// The result of a LoRA's down projection, `mid`, and the FP16 weights of its up projection, with
/// the adapter's scale in them.
pub(crate) struct Lora<'a> {
    pub mid: Array,
    pub up: &'a Buffer,
}

impl Finish<'_> {
    fn check(&self, rows: usize, outputs: usize) -> Result<()> {
        let vector = |v: Option<&Array>| v.is_none_or(|v| v.shape() == [1, outputs]);
        if !vector(self.scales)
            || !vector(self.bias)
            || self
                .addend
                .as_ref()
                .is_some_and(|a| a.shape() != [rows, outputs])
            || self
                .lora
                .as_ref()
                .is_some_and(|l| l.mid.rows != rows || l.up.0.bytes != outputs * l.mid.cols * 2)
        {
            return Err(Error::new("product finish shapes mismatch".into()));
        }
        Ok(())
    }

    fn flags(&self) -> u32 {
        self.scales.is_some() as u32
            | (self.bias.is_some() as u32) << 1
            | (self.addend.is_some() as u32) << 2
            | (self.lora.is_some() as u32) << 3
    }

    /// The LoRA's up projection run whole and added to the addend, for a product that cannot run
    /// it a tile at a time.
    fn lora_into_addend(mut self, outputs: usize) -> Result<Self> {
        if let Some(lora) = self.lora.take() {
            let part = lora.mid.linear_half(lora.up, outputs)?;
            self.addend = Some(match self.addend.take() {
                Some(addend) => addend.add(&part)?,
                None => part,
            });
        }
        Ok(self)
    }

    /// The same steps, a pass each, for a product that cannot take them itself.
    pub(crate) fn apply(self, mut result: Array) -> Result<Array> {
        self.check(result.rows, result.cols)?;
        let this = self.lora_into_addend(result.cols)?;
        if let Some(scales) = this.scales {
            result = result.mul(scales)?;
        }
        if let Some(bias) = this.bias {
            result = result.add(bias)?;
        }
        if let Some(addend) = &this.addend {
            result = result.add(addend)?;
        }
        Ok(result)
    }
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
        finish: Finish,
    ) -> Result<Self> {
        self.packed_product(Packing::Plain, weight, outputs, precision, finish)
    }

    /// `linear_int8` of these rows rotated by ConvRot's Hadamard transform, which a ConvRot
    /// layer's weights expect. The rotation happens as the rows are packed.
    pub(crate) fn rotate_linear_int8(
        &self,
        weight: &Buffer,
        outputs: usize,
        precision: LinearPrecision,
        finish: Finish,
    ) -> Result<Self> {
        self.packed_product(Packing::Rotate, weight, outputs, precision, finish)
    }

    /// `rotate_linear_int8` of SwiGLU of these rows, an MLP's gate and up halves side by side.
    /// SwiGLU happens as the rows are packed, so the gated rows are never written out.
    pub(crate) fn swiglu_rotate_linear_int8(
        &self,
        weight: &Buffer,
        outputs: usize,
        precision: LinearPrecision,
        finish: Finish,
    ) -> Result<Self> {
        self.packed_product(Packing::GatedRotate, weight, outputs, precision, finish)
    }

    fn packed_product(
        &self,
        packing: Packing,
        weight: &Buffer,
        outputs: usize,
        precision: LinearPrecision,
        finish: Finish,
    ) -> Result<Self> {
        if packing == Packing::GatedRotate && !self.cols.is_multiple_of(2) {
            return Err(Error::new("SwiGLU requires two equal halves".into()));
        }
        let cols = if packing == Packing::GatedRotate {
            self.cols / 2
        } else {
            self.cols
        };
        if !std::sync::Arc::ptr_eq(&self.device().0, &weight.0.device.0)
            || outputs.checked_mul(cols) != Some(weight.0.bytes)
        {
            return Err(Error::new("packed matrix shape or device mismatch".into()));
        }

        size(outputs, cols)?;
        // TensorOps uses signed extents/indices. Bound the INT32 worst-case sum too:
        // quantized input is [-127,127], checkpoint weights can contain -128.
        if precision == LinearPrecision::Fp32
            || self.len() > i32::MAX as usize
            || outputs * cols > i32::MAX as usize
            || self
                .rows
                .checked_mul(outputs)
                .is_none_or(|n| n > i32::MAX as usize)
            || (precision == LinearPrecision::Int8 && cols > i32::MAX as usize / (127 * 128))
        {
            let input = match packing {
                Packing::Plain => self.clone(),
                Packing::Rotate => self.rotate()?,
                Packing::GatedRotate => self.swiglu()?.rotate()?,
            };
            return finish.apply(input.linear_packed(weight, outputs, DType::I8)?);
        }
        if packing != Packing::Plain && !cols.is_multiple_of(256) {
            return Err(Error::new(
                "ConvRot requires a multiple of 256 features".into(),
            ));
        }

        self.pack(packing, precision)?
            .product(weight, outputs, finish)
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
        finish: Finish,
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
                device, &packed, scales, weight, rows, cols, outputs, precision, finish,
            );
        }

        finish.apply(
            Self::from_quantized(device, input, scales, rows, cols)?.linear_packed(
                weight,
                outputs,
                DType::I8,
            )?,
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
        finish: Finish,
    ) -> Result<Self> {
        finish.check(rows, outputs)?;
        let finish = if precision == LinearPrecision::MpsFp16 {
            finish.lora_into_addend(outputs)?
        } else {
            finish
        };
        let flags = finish.flags();
        // An absent vector binds the row scales in its place, which the flags then never read.
        let output_scales = finish.scales.map_or(&scales.buffer, |v| &v.buffer);
        let bias = finish.bias.map_or(&scales.buffer, |v| &v.buffer);
        if precision == LinearPrecision::MpsFp16 {
            let out = Self::empty(device, rows, outputs)?;
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

            let addend = finish.addend.as_ref().map_or(&out.buffer, |a| &a.buffer);
            device.run(
                "finish_linear",
                &[&out.buffer, &scales.buffer, output_scales, bias, addend],
                &[out.len() as u32, outputs as u32, flags],
                out.len(),
                false,
            )?;
            return Ok(out);
        }

        // The products on the matrix units add what their output holds, so an addend is theirs.
        let (mid, up, rank) = match &finish.lora {
            Some(lora) => (&lora.mid.buffer, lora.up, lora.mid.cols),
            None => (&scales.buffer, &scales.buffer, 0),
        };
        let out = match finish.addend {
            Some(addend) => addend,
            None => Self::empty(device, rows, outputs)?,
        };
        device.run(
            if precision != LinearPrecision::Int8 {
                "mpp_fp16"
            } else if rows >= INT8_GROUP.0
                && outputs >= INT8_GROUP.1
                && cols.is_multiple_of(INT8_PRODUCT_DEPTH)
            {
                "mpp_int8_inside"
            } else {
                "mpp_int8"
            },
            &[
                packed,
                weight,
                &out.buffer,
                &scales.buffer,
                output_scales,
                bias,
                mid,
                up,
            ],
            &[
                rows as u32,
                outputs as u32,
                cols as u32,
                flags,
                rank as u32,
                if precision == LinearPrecision::Int8 {
                    INT8_PRODUCT_BAND
                } else {
                    product_band(rows, outputs, cols)
                },
            ],
            if precision == LinearPrecision::Int8 {
                rows.div_ceil(INT8_PRODUCT_TILE) * outputs.div_ceil(INT8_PRODUCT_TILE)
            } else {
                rows.div_ceil(PRODUCT_TILE.0) * outputs.div_ceil(PRODUCT_TILE.1)
            },
            true,
        )?;
        Ok(out)
    }

    /// `self · weightᵀ` on the matrix units for `outputs` rows of FP16 weights, `self.cols` each,
    /// with the FP32 rows as they are: the two products of a LoRA.
    pub(crate) fn linear_half(&self, weight: &Buffer, outputs: usize) -> Result<Self> {
        if !std::sync::Arc::ptr_eq(&self.device().0, &weight.0.device.0)
            || outputs.checked_mul(self.cols * 2) != Some(weight.0.bytes)
        {
            return Err(Error::new("FP16 weight shape or device mismatch".into()));
        }
        // TensorOps uses signed extents and indices.
        if [self.len(), outputs * self.cols, self.rows * outputs]
            .iter()
            .any(|&n| n > i32::MAX as usize)
        {
            return Err(Error::new("FP16 weight product too large".into()));
        }

        let out = Self::empty(self.device(), self.rows, outputs)?;
        self.device().run(
            "mpp_lora",
            &[&self.buffer, weight, &out.buffer],
            &[self.rows as u32, outputs as u32, self.cols as u32],
            self.rows.div_ceil(64) * outputs.div_ceil(64),
            true,
        )?;
        Ok(out)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, 0)
    }

    /// `self + delta × scale`, `scale` a row of `self.cols` values, in one pass.
    pub(crate) fn add_scaled(&self, delta: &Self, scale: &Self) -> Result<Self> {
        if delta.shape() != self.shape() || scale.len() != self.cols {
            return Err(Error::new("incompatible elementwise shapes".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "add_scaled",
            &[&self.buffer, &delta.buffer, &scale.buffer, &out.buffer],
            &[self.len() as u32, self.cols as u32],
            self.len(),
            false,
        )?;
        Ok(out)
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

    /// `attention` at `precision`. FP16 takes the matrix units for heads of 64 or 128 where the
    /// device has them, and anything else is the FP32 attention.
    pub fn attention_at(
        &self,
        key: &Self,
        value: &Self,
        heads: usize,
        kv_heads: usize,
        causal: bool,
        precision: AttentionPrecision,
    ) -> Result<Self> {
        if precision.on_matrix_units()
            && heads > 0
            && kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && self.cols.is_multiple_of(heads)
            && [64, 128].contains(&(self.cols / heads))
            && key.cols == kv_heads * (self.cols / heads)
            && key.shape() == value.shape()
            && !(causal && self.rows != key.rows)
            && self.device().supports_tensor_ops()
        {
            return self.tensor_attention(key, value, heads, kv_heads, causal);
        }
        self.attention(key, value, heads, kv_heads, causal)
    }

    /// Attention over heads of 64 or 128 on the matrix units: FP16 copies of the inputs, FP32
    /// softmax and accumulation. See `flash_attention`.
    fn tensor_attention(
        &self,
        key: &Self,
        value: &Self,
        heads: usize,
        kv_heads: usize,
        causal: bool,
    ) -> Result<Self> {
        let (q, k, v) = (self.to_half()?, key.to_half()?, value.to_half()?);
        half_attention(
            self.device(),
            [&q, &k, &v],
            [self.rows, key.rows],
            heads,
            kv_heads,
            self.cols / heads,
            causal,
        )
    }

    /// A block's attention inputs from its qkv projection's output, rows of `[3][heads][dim]`, or
    /// of `[heads][3][dim]` with `per_head`, for heads of up to 256:
    /// the queries and keys RMS-normalized per head by `q_norm` and `k_norm` and rotated by
    /// `angles`, and all three in FP16 for the matrix units with `half`. With `int8`, which takes
    /// `half`, the queries and keys are also quantized to INT8 for the scores, one scale a token
    /// and head. One pass reads the projection and writes what the attention reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_inputs(
        &self,
        heads: usize,
        (q_norm, k_norm): (&Self, &Self),
        epsilon: f32,
        angles: Option<&Self>,
        half: bool,
        per_head: bool,
        int8: bool,
    ) -> Result<AttentionInputs> {
        let dim = self.cols / 3 / heads.max(1);
        let inner = heads * dim;
        let pairs = angles.map_or(0, |angles| angles.cols);
        if heads == 0
            || self.cols != 3 * inner
            || dim > 256
            || q_norm.len() != dim
            || k_norm.len() != dim
            || angles.is_some_and(|angles| angles.rows != self.rows || 2 * pairs > dim)
            || (int8 && !half)
        {
            return Err(Error::new("invalid attention input shapes".into()));
        }

        let device = self.device();
        let bytes = self.rows * inner * if half { 2 } else { 4 };
        let (query, key, value) = (
            device.alloc(bytes, None)?,
            device.alloc(bytes, None)?,
            device.alloc(bytes, None)?,
        );
        let quantized = int8
            .then(|| -> Result<QuantizedQueryKeys> {
                Ok(QuantizedQueryKeys {
                    query: device.alloc(self.rows * inner, None)?,
                    key: device.alloc(self.rows * inner, None)?,
                    scales: device.alloc(2 * self.rows * heads * 4, None)?,
                })
            })
            .transpose()?;
        device.run(
            "attention_inputs",
            &[
                &self.buffer,
                &q_norm.buffer,
                &k_norm.buffer,
                &angles.unwrap_or(q_norm).buffer,
                &query,
                &key,
                &value,
                quantized.as_ref().map_or(&value, |q| &q.query),
                quantized.as_ref().map_or(&value, |q| &q.key),
                quantized.as_ref().map_or(&value, |q| &q.scales),
            ],
            &[
                heads as u32,
                pairs as u32,
                half as u32,
                epsilon.to_bits(),
                dim as u32,
                per_head as u32,
                int8 as u32,
                self.rows as u32,
            ],
            self.rows,
            true,
        )?;
        Ok(AttentionInputs {
            query,
            key,
            value,
            rows: self.rows,
            heads,
            dim,
            half,
            quantized,
        })
    }

    /// The FP16 copy of an input that the attention on the matrix units reads.
    pub(crate) fn to_half(&self) -> Result<Buffer> {
        let copy = self.device().alloc(self.len() * 2, None)?;
        self.device().run(
            "float_to_half",
            &[&self.buffer, &copy],
            &[self.len() as u32],
            self.len(),
            false,
        )?;
        Ok(copy)
    }

    pub fn swiglu(&self) -> Result<Self> {
        if !self.cols.is_multiple_of(2) {
            return Err(Error::new("SwiGLU requires two equal halves".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols / 2)?;
        self.device().run(
            "swiglu",
            &[&self.buffer, &out.buffer],
            &[out.len() as u32, out.cols as u32],
            out.len(),
            false,
        )?;
        Ok(out)
    }

    /// A residual stream's next block input in one pass, as `add_gated`, `norm_modulate` and the
    /// packing of `rotate_linear_int8` would make it: `self`, plus `delta` gated by chunk `gate` of
    /// its row of `gates` when there is one, normalized without centering by `weight`, modulated by
    /// chunks `shift` and `scale` of its row of `m`, rotated and packed for `precision`. Returns
    /// the new residual when there was a delta, the rows before rotation with `keep`, which a
    /// LoRA's down projection reads, and the packed rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_norm_pack(
        &self,
        delta: Option<(&Self, &Self, usize)>,
        weight: &Self,
        epsilon: f32,
        (m, rows, shift, scale): (&Self, &RowMap, usize, usize),
        precision: LinearPrecision,
        keep: bool,
    ) -> Result<(Option<Self>, Option<Self>, PackedRows)> {
        let chunks = |table: &Self| table.cols / self.cols;
        if !PackedRows::fit(self.cols, precision)
            || weight.len() != self.cols
            || rows.len != self.rows
            || rows.maximum >= m.rows
            || !m.cols.is_multiple_of(self.cols)
            || shift.max(scale) >= chunks(m)
            || delta.is_some_and(|(delta, gates, gate)| {
                delta.shape() != self.shape()
                    || rows.maximum >= gates.rows
                    || !gates.cols.is_multiple_of(self.cols)
                    || gate >= chunks(gates)
            })
        {
            return Err(Error::new("invalid add_norm_pack shapes".into()));
        }

        let device = self.device();
        let int8 = precision == LinearPrecision::Int8;
        let hidden = delta
            .map(|_| Self::empty(device, self.rows, self.cols))
            .transpose()?;
        let normed = keep
            .then(|| Self::empty(device, self.rows, self.cols))
            .transpose()?;
        let packed = device.alloc(self.len() * if int8 { 1 } else { 2 }, None)?;
        let scales = Self::empty(device, self.rows, 1)?;
        let (delta_buffer, gates, gate) = match delta {
            Some((delta, gates, gate)) => (&delta.buffer, gates, gate),
            None => (&self.buffer, m, 0),
        };
        device.run(
            "add_norm_pack",
            &[
                &self.buffer,
                delta_buffer,
                &gates.buffer,
                &m.buffer,
                &rows.buffer,
                &weight.buffer,
                hidden.as_ref().map_or(&self.buffer, |h| &h.buffer),
                normed.as_ref().map_or(&self.buffer, |n| &n.buffer),
                &packed,
                &scales.buffer,
            ],
            &[
                self.cols as u32,
                epsilon.to_bits(),
                m.cols as u32,
                shift as u32,
                scale as u32,
                delta.is_some() as u32,
                gate as u32,
                gates.cols as u32,
                int8 as u32,
                keep as u32,
                delta.is_some() as u32,
            ],
            self.rows,
            true,
        )?;
        Ok((
            hidden,
            normed,
            PackedRows {
                buffer: packed,
                scales,
                rows: self.rows,
                cols: self.cols,
                precision,
            },
        ))
    }

    /// These rows packed for a product at `precision`, INT8 or FP16 with one scale a row, as
    /// `packing` has them: as they are, rotated, or SwiGLU of their halves rotated.
    pub(crate) fn pack(&self, packing: Packing, precision: LinearPrecision) -> Result<PackedRows> {
        let cols = if packing == Packing::GatedRotate {
            self.cols / 2
        } else {
            self.cols
        };
        if precision == LinearPrecision::Fp32
            || (packing == Packing::GatedRotate && !self.cols.is_multiple_of(2))
            || (packing != Packing::Plain && !cols.is_multiple_of(256))
        {
            return Err(Error::new("rows that cannot be packed".into()));
        }
        let int8 = precision == LinearPrecision::Int8;
        let packed = self
            .device()
            .alloc(self.rows * cols * if int8 { 1 } else { 2 }, None)?;
        let scales = Self::empty(self.device(), self.rows, 1)?;
        self.device().run(
            if packing == Packing::Plain {
                "pack_linear_input"
            } else {
                "rotate_pack_linear_input"
            },
            &[&self.buffer, &packed, &scales.buffer],
            &[
                cols as u32,
                int8 as u32,
                (packing == Packing::GatedRotate) as u32,
            ],
            self.rows,
            true,
        )?;
        Ok(PackedRows {
            buffer: packed,
            scales,
            rows: self.rows,
            cols,
            precision,
        })
    }

    /// `norm` without centering followed by `modulate`, in one pass.
    pub(crate) fn norm_modulate(
        &self,
        weight: &Self,
        epsilon: f32,
        m: &Self,
        rows: &RowMap,
        shift: usize,
        scale: usize,
    ) -> Result<Self> {
        if weight.len() != self.cols
            || rows.len != self.rows
            || rows.maximum >= m.rows
            || !m.cols.is_multiple_of(self.cols)
            || shift >= m.cols / self.cols
            || scale >= m.cols / self.cols
        {
            return Err(Error::new("invalid modulation shapes".into()));
        }

        let out = Self::empty(self.device(), self.rows, self.cols)?;
        self.device().run(
            "normalize_modulate",
            &[
                &self.buffer,
                &weight.buffer,
                &m.buffer,
                &rows.buffer,
                &out.buffer,
            ],
            &[
                self.cols as u32,
                epsilon.to_bits(),
                m.cols as u32,
                shift as u32,
                scale as u32,
            ],
            self.rows,
            true,
        )?;
        Ok(out)
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
/// Outputs a side of one threadgroup of the INT8 product, `PRODUCT_TILE` in matmul.metal.
const INT8_PRODUCT_TILE: usize = 128;
/// The rows and weights one SIMD group of the INT8 product answers, `PRODUCT_GROUP_ROWS` and
/// `PRODUCT_GROUP_COLS`, and the depth its groups take together, `PRODUCT_DEPTH`. A product of at
/// least one group's outputs and whole depths runs `mpp_int8_inside`, which needs no checked loads.
const INT8_GROUP: (usize, usize) = (64, 32);
const INT8_PRODUCT_DEPTH: usize = 256;
/// The row tiles the threadgroups of the INT8 product go down before taking the next column tile.
/// Bands of 4 were the fastest for every product of a MiniMax H3 block.
const INT8_PRODUCT_BAND: u32 = 4;

/// How many row tiles the threadgroups of `packed_product`, which runs the FP16 products, go down
/// before taking the next column tile, which decides what the ones running together share in the
/// cache. Tuned at MiniMax H3's shapes while it also ran the INT8 ones: at a DiT block's rows,
/// bands of 16 suit the wide qkv projection and fc1 and bands of 2 the deep out projection and fc2,
/// where bands of 24 or 4 are slower than taking a row tile's columns in turn; a video VAE tile's
/// few rows take bands of 4.
fn product_band(rows: usize, outputs: usize, inputs: usize) -> u32 {
    if rows <= 4096 {
        4
    } else if outputs >= 2 * inputs {
        16
    } else {
        2
    }
}

/// Rows rotated and packed for a ConvRot layer's product, INT8 or FP16 as its precision takes them,
/// with one scale a row, which `add_norm_pack` makes.
pub(crate) struct PackedRows {
    buffer: Buffer,
    scales: Array,
    rows: usize,
    cols: usize,
    precision: LinearPrecision,
}

impl PackedRows {
    /// Whether rows of `width` values at `precision` can be packed by `add_norm_pack`.
    pub(crate) fn fit(width: usize, precision: LinearPrecision) -> bool {
        precision != LinearPrecision::Fp32
            && width.is_multiple_of(256)
            && width <= ADD_NORM_PACK_WIDTH
    }

    /// The rows as INT8 weights of a product and one scale an output, which is what a LoRA's
    /// down projection packed at INT8 is to the rows of its layer.
    pub(crate) fn as_weight(&self) -> Result<(&Buffer, Array)> {
        if self.precision != LinearPrecision::Int8 {
            return Err(Error::new("only INT8 rows are a product's weights".into()));
        }
        Ok((&self.buffer, self.scales.reshape(1, self.rows)?))
    }

    /// The product of these rows with a ConvRot layer's INT8 `weight` of `outputs` rows.
    pub(crate) fn product(&self, weight: &Buffer, outputs: usize, finish: Finish) -> Result<Array> {
        if outputs.checked_mul(self.cols) != Some(weight.0.bytes)
            || [
                self.rows * self.cols,
                outputs * self.cols,
                self.rows * outputs,
            ]
            .iter()
            .any(|&n| n > i32::MAX as usize)
        {
            return Err(Error::new("packed matrix shape mismatch".into()));
        }
        Array::product_of_packed(
            self.scales.device(),
            &self.buffer,
            &self.scales,
            weight,
            self.rows,
            self.cols,
            outputs,
            self.precision,
            finish,
        )
    }
}

/// The widest rows `add_norm_pack` keeps in registers: three 256-value blocks for each of eight SIMD
/// groups, `PACK_BLOCKS` in ops.metal.
const ADD_NORM_PACK_WIDTH: usize = 3 * 8 * 256;

/// What a packed product does to its rows as it packs them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Packing {
    Plain,
    /// ConvRot's Hadamard rotation.
    Rotate,
    /// SwiGLU of the rows' two halves, then the rotation.
    GatedRotate,
}

/// Whether attention at `precision` over heads of `dim` runs on the matrix units, which read FP16
/// inputs.
pub(crate) fn attends_in_half(device: &Device, precision: AttentionPrecision, dim: usize) -> bool {
    precision.on_matrix_units() && [64, 128].contains(&dim) && device.supports_tensor_ops()
}

/// `flash_attention` over FP16 query, key and value rows of `rows` = [queries, keys].
fn half_attention(
    device: &Device,
    [q, k, v]: [&Buffer; 3],
    [queries, keys]: [usize; 2],
    heads: usize,
    kv_heads: usize,
    dim: usize,
    causal: bool,
) -> Result<Array> {
    let out = Array::empty(device, queries, heads * dim)?;
    device.run(
        &format!("mpp_attention_{dim}"),
        &[q, k, v, &out.buffer],
        &[
            keys as u32,
            heads as u32,
            kv_heads as u32,
            dim as u32,
            causal as u32,
            queries as u32,
        ],
        queries.div_ceil(TENSOR_ATTENTION_QUERIES) * heads,
        true,
    )?;
    Ok(out)
}

/// A block's queries and keys in INT8, `[tokens, heads × dim]`, with one FP32 scale a token and
/// head, the queries' first.
pub(crate) struct QuantizedQueryKeys {
    pub(crate) query: Buffer,
    pub(crate) key: Buffer,
    pub(crate) scales: Buffer,
}

/// A block's query, key and value, `[tokens, heads × dim]` each, in FP16 for the attention on the
/// matrix units or in FP32 for the FP32 attention. The scores take INT8 queries and keys when
/// they are quantized.
pub struct AttentionInputs {
    pub(crate) query: Buffer,
    pub(crate) key: Buffer,
    pub(crate) value: Buffer,
    pub(crate) rows: usize,
    pub(crate) heads: usize,
    pub(crate) dim: usize,
    pub(crate) half: bool,
    pub(crate) quantized: Option<QuantizedQueryKeys>,
}

impl AttentionInputs {
    /// The inputs of FP32 arrays, converted to FP16 with `half`.
    pub fn from_arrays(
        query: &Array,
        key: &Array,
        value: &Array,
        heads: usize,
        half: bool,
    ) -> Result<Self> {
        if heads == 0
            || !query.cols.is_multiple_of(heads)
            || key.shape() != query.shape()
            || value.shape() != query.shape()
        {
            return Err(Error::new("invalid attention input shapes".into()));
        }
        let buffer = |x: &Array| {
            if half {
                x.to_half()
            } else {
                Ok(x.buffer.clone())
            }
        };
        Ok(Self {
            query: buffer(query)?,
            key: buffer(key)?,
            value: buffer(value)?,
            rows: query.rows,
            heads,
            dim: query.cols / heads,
            half,
            quantized: None,
        })
    }

    pub fn width(&self) -> usize {
        self.heads * self.dim
    }

    /// Dense attention of every row over every row.
    pub(crate) fn attend(&self) -> Result<Array> {
        let device = &self.query.0.device;
        if let Some(quantized) = &self.quantized {
            if self.dim != 128 {
                return Err(Error::new("INT8 scores take heads of 128".into()));
            }
            let out = Array::empty(device, self.rows, self.width())?;
            device.run(
                "mpp_attention_int8_128",
                &[
                    &self.query,
                    &self.key,
                    &self.value,
                    &out.buffer,
                    &quantized.query,
                    &quantized.key,
                    &quantized.scales,
                ],
                &[
                    self.rows as u32,
                    self.heads as u32,
                    self.heads as u32,
                    self.dim as u32,
                    0,
                    self.rows as u32,
                ],
                self.rows.div_ceil(TENSOR_ATTENTION_QUERIES) * self.heads,
                true,
            )?;
            return Ok(out);
        }
        if self.half {
            return half_attention(
                device,
                [&self.query, &self.key, &self.value],
                [self.rows, self.rows],
                self.heads,
                self.heads,
                self.dim,
                false,
            );
        }
        let array = |buffer: &Buffer| Array {
            buffer: buffer.clone(),
            rows: self.rows,
            cols: self.width(),
        };
        array(&self.query).attention(
            &array(&self.key),
            &array(&self.value),
            self.heads,
            self.heads,
            false,
        )
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

    /// The fused pass gives what slicing the projection, normalizing, rotating and attending give
    /// one step at a time, in both precisions and with and without angles.
    #[test]
    fn attention_inputs_match_the_steps_they_fuse() {
        let device = Device::new().unwrap();
        let (rows, heads, dim, pairs) = (150, 3, 128, 48);
        let inner = heads * dim;
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * seed + 7) % 89) as f32 / 44.0 - 1.0)
                .collect()
        };
        let qkv = Array::from_f32(&device, rows, 3 * inner, &fill(rows * 3 * inner, 31)).unwrap();
        let q_norm = Array::from_f32(&device, 1, dim, &fill(dim, 17)).unwrap();
        let k_norm = Array::from_f32(&device, 1, dim, &fill(dim, 13)).unwrap();
        let angles = Array::from_f32(&device, rows, pairs, &fill(rows * pairs, 7)).unwrap();
        for angles in [None, Some(&angles)] {
            let step = |offset: usize, weight: &Array| {
                let x = qkv
                    .slice(0, rows, offset, inner)
                    .unwrap()
                    .reshape(rows * heads, dim)
                    .unwrap()
                    .norm(weight, 1e-5, false)
                    .unwrap()
                    .reshape(rows, inner)
                    .unwrap();
                angles.map_or(x.clone(), |angles| x.rope(heads, angles).unwrap())
            };
            let (q, k) = (step(0, &q_norm), step(inner, &k_norm));
            let v = qkv.slice(0, rows, 2 * inner, inner).unwrap();
            let per_head: Vec<f32> = {
                let values = qkv.to_f32().unwrap();
                (0..rows * 3 * inner)
                    .map(|i| {
                        let (row, rest) = (i / (3 * inner), i % (3 * inner));
                        let (head, tensor, d) = (rest / (3 * dim), rest / dim % 3, rest % dim);
                        values[row * 3 * inner + tensor * inner + head * dim + d]
                    })
                    .collect()
            };
            let per_head = Array::from_f32(&device, rows, 3 * inner, &per_head).unwrap();
            for half in [false, true]
                .into_iter()
                .take(1 + device.supports_tensor_ops() as usize)
            {
                let precision = if half {
                    AttentionPrecision::Fp16
                } else {
                    AttentionPrecision::Fp32
                };
                let expected = q
                    .attention_at(&k, &v, heads, heads, false, precision)
                    .unwrap()
                    .to_f32()
                    .unwrap();
                for (layout, input) in [(false, &qkv), (true, &per_head)] {
                    let got = input
                        .attention_inputs(
                            heads,
                            (&q_norm, &k_norm),
                            1e-5,
                            angles,
                            half,
                            layout,
                            false,
                        )
                        .unwrap()
                        .attend()
                        .unwrap()
                        .to_f32()
                        .unwrap();
                    let worst = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        worst < 1e-4,
                        "FP16 {half}, angles {}, per head {layout}: {worst}",
                        angles.is_some()
                    );
                }
                // INT8 scores round the queries and keys to 127 steps of each token's and head's
                // largest value, which moves the output further, but not far.
                if half {
                    let got = qkv
                        .attention_inputs(
                            heads,
                            (&q_norm, &k_norm),
                            1e-5,
                            angles,
                            true,
                            false,
                            true,
                        )
                        .unwrap()
                        .attend()
                        .unwrap()
                        .to_f32()
                        .unwrap();
                    let worst = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(worst < 2e-2, "INT8, angles {}: {worst}", angles.is_some());
                }
            }
        }
    }

    #[test]
    fn tensor_attention_matches_reference() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        // Ragged tiles on both sides, grouped heads, and both masks. With a slope, scores climb
        // key by key far past the margin the kernel lets a row's maximum grow before rescaling.
        for (rows, keys, heads, kv_heads, causal, slope) in [
            (150, 200, 4, 2, false, 0.0),
            (150, 150, 4, 2, true, 0.0),
            (1, 70, 2, 2, false, 0.0),
            (64, 64, 1, 1, true, 0.0),
            (130, 300, 2, 1, false, 0.1),
            (200, 200, 2, 2, true, 0.1),
        ] {
            for dim in [64, 128] {
                let fill = |n: usize, seed: usize| -> Vec<f32> {
                    (0..n)
                        .map(|i| ((i * seed + 7) % 89) as f32 / 44.0 - 1.0)
                        .collect()
                };
                let q = fill(rows * heads * dim, 31);
                let mut k = fill(keys * kv_heads * dim, 17);
                // Every query has a positive first feature, so a key's score grows with its own.
                let q: Vec<f32> = q
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| if i % dim == 0 { 1.0 } else { x })
                    .collect();
                for (i, x) in k.iter_mut().enumerate() {
                    if i % dim == 0 {
                        *x += slope * (i / (kv_heads * dim)) as f32 * (dim as f32).sqrt();
                    }
                }
                let v = fill(keys * kv_heads * dim, 53);
                let got = Array::from_f32(&device, rows, heads * dim, &q)
                    .unwrap()
                    .attention_at(
                        &Array::from_f32(&device, keys, kv_heads * dim, &k).unwrap(),
                        &Array::from_f32(&device, keys, kv_heads * dim, &v).unwrap(),
                        heads,
                        kv_heads,
                        causal,
                        AttentionPrecision::Fp16,
                    )
                    .unwrap()
                    .to_f32()
                    .unwrap();

                for row in 0..rows {
                    for head in 0..heads {
                        let kh = head / (heads / kv_heads);
                        let seen = if causal { row + 1 } else { keys };
                        let scores: Vec<f64> = (0..seen)
                            .map(|t| {
                                (0..dim)
                                    .map(|c| {
                                        q[(row * heads + head) * dim + c] as f64
                                            * k[(t * kv_heads + kh) * dim + c] as f64
                                    })
                                    .sum::<f64>()
                                    / (dim as f64).sqrt()
                            })
                            .collect();
                        let top = scores.iter().cloned().fold(f64::MIN, f64::max);
                        let weights: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
                        let total: f64 = weights.iter().sum();
                        for c in 0..dim {
                            let expected = (0..seen)
                                .map(|t| weights[t] * v[(t * kv_heads + kh) * dim + c] as f64)
                                .sum::<f64>()
                                / total;
                            let actual = got[(row * heads + head) * dim + c] as f64;
                            assert!(
                                (actual - expected).abs() < 1e-2,
                                "{rows}×{keys}×{dim} causal {causal} slope {slope}: row {row} head {head} col {c}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

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
                    let result = x
                        .linear_int8(&weights, outputs, precision, Finish::default())
                        .unwrap();
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

    /// Taking SwiGLU as the rows are packed is the same arithmetic as the swiglu kernel first.
    #[test]
    fn gating_while_packing_matches_gating_first() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let (rows, cols, outputs) = (37usize, 1280usize, 96usize);
        let x = Array::from_f32(
            &device,
            rows,
            2 * cols,
            &(0..rows * 2 * cols)
                .map(|i| ((i * 31) % 251) as f32 / 25.0 - 5.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let bytes: Vec<u8> = (0..outputs * cols).map(|i| (i * 37 % 251) as u8).collect();
        let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
        for precision in [LinearPrecision::Int8, LinearPrecision::Fp16] {
            let first = x
                .swiglu()
                .unwrap()
                .rotate_linear_int8(&weight, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            let fused = x
                .swiglu_rotate_linear_int8(&weight, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            assert_eq!(first, fused, "{precision:?}");
        }
    }

    /// Adding, normalizing and packing in one pass gives what add_gated, norm_modulate and the
    /// packing product give in three. The residual is the same to the bit; the norm sums in
    /// another order, so a packed value can land on the other side of a rounding edge.
    #[test]
    fn adding_normalizing_and_packing_match_the_three_passes() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let (rows, width, outputs, chunks) = (37usize, 1280usize, 96usize, 6usize);
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * seed + 7) % 89) as f32 / 44.0 - 1.0)
                .collect()
        };
        let x = Array::from_f32(&device, rows, width, &fill(rows * width, 31)).unwrap();
        let delta = Array::from_f32(&device, rows, width, &fill(rows * width, 23)).unwrap();
        let weight = Array::from_f32(&device, 1, width, &fill(width, 17)).unwrap();
        let m = Array::from_f32(&device, 3, chunks * width, &fill(3 * chunks * width, 13)).unwrap();
        let gates =
            Array::from_f32(&device, 3, chunks * width, &fill(3 * chunks * width, 11)).unwrap();
        let map = RowMap::new(&device, &(0..rows).map(|row| row % 3).collect::<Vec<_>>()).unwrap();
        let bytes: Vec<u8> = (0..outputs * width).map(|i| (i * 37 % 251) as u8).collect();
        let layer = device.alloc(bytes.len(), Some(&bytes)).unwrap();
        for precision in [LinearPrecision::Int8, LinearPrecision::Fp16] {
            let hidden = x.add_gated(&delta, &gates, &map, 5).unwrap();
            let normed = hidden.norm_modulate(&weight, 1e-5, &m, &map, 0, 1).unwrap();
            let expected = normed
                .rotate_linear_int8(&layer, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            let (fused_hidden, fused_normed, packed) = x
                .add_norm_pack(
                    Some((&delta, &gates, 5)),
                    &weight,
                    1e-5,
                    (&m, &map, 0, 1),
                    precision,
                    true,
                )
                .unwrap();
            assert_eq!(
                fused_hidden.unwrap().to_f32().unwrap(),
                hidden.to_f32().unwrap()
            );
            let close = |a: &[f32], b: &[f32], tolerance: f32| {
                let largest = b.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let worst = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst <= largest * tolerance,
                    "{precision:?}: {worst} of {largest}"
                );
            };
            close(
                &fused_normed.unwrap().to_f32().unwrap(),
                &normed.to_f32().unwrap(),
                1e-5,
            );
            let got = packed
                .product(&layer, outputs, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            close(&got, &expected, 1e-2);
        }
    }

    /// Normalizing and modulating in one pass is the same arithmetic as the two passes.
    #[test]
    fn normalizing_while_modulating_matches_the_two_passes() {
        let device = Device::new().unwrap();
        let (rows, width, chunks) = (9usize, 640usize, 6usize);
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * seed + 7) % 89) as f32 / 44.0 - 1.0)
                .collect()
        };
        let x = Array::from_f32(&device, rows, width, &fill(rows * width, 31)).unwrap();
        let weight = Array::from_f32(&device, 1, width, &fill(width, 17)).unwrap();
        let m = Array::from_f32(&device, 3, chunks * width, &fill(3 * chunks * width, 13)).unwrap();
        let map = RowMap::new(&device, &(0..rows).map(|row| row % 3).collect::<Vec<_>>()).unwrap();
        let two = x
            .norm(&weight, 1e-5, false)
            .unwrap()
            .modulate(&m, &map, 3, 4)
            .unwrap()
            .to_f32()
            .unwrap();
        let one = x
            .norm_modulate(&weight, 1e-5, &m, &map, 3, 4)
            .unwrap()
            .to_f32()
            .unwrap();
        assert_eq!(one, two);
    }

    /// Rotating as the rows are packed is the same arithmetic as rotating first: the Hadamard
    /// stages add the same values in the same order, so the products agree to the bit.
    #[test]
    fn rotating_while_packing_matches_rotating_first() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let (rows, cols, outputs) = (37usize, 1280usize, 96usize);
        let x = Array::from_f32(
            &device,
            rows,
            cols,
            &(0..rows * cols)
                .map(|i| ((i * 31) % 251) as f32 / 25.0 - 5.0)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let bytes: Vec<u8> = (0..outputs * cols).map(|i| (i * 37 % 251) as u8).collect();
        let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
        for precision in [LinearPrecision::Int8, LinearPrecision::Fp16] {
            let first = x
                .rotate()
                .unwrap()
                .linear_int8(&weight, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            let fused = x
                .rotate_linear_int8(&weight, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap();
            assert_eq!(first, fused, "{precision:?}");
        }
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
                &device,
                &packed,
                &scales,
                &weight,
                rows,
                cols,
                outputs,
                precision,
                Finish::default(),
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
                    .linear_int8(&weight, outputs, precision, Finish::default())
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

    /// The INT8 product's finish, which runs on its registers: the outputs' scales, a bias, an
    /// addend and a LoRA of a rank that does not fill a fragment, over tiles its edges cut.
    #[test]
    fn the_int8_product_finishes_as_the_host_would() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let (rows, outputs, cols, rank) = (150usize, 200usize, 272usize, 20usize);
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * seed + 7) % 89) as f32 / 44.0 - 1.0)
                .collect()
        };
        let x = Array::from_f32(&device, rows, cols, &fill(rows * cols, 31)).unwrap();
        let bytes: Vec<u8> = (0..outputs * cols).map(|i| (i * 37 % 251) as u8).collect();
        let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
        let plain = x
            .linear_int8(&weight, outputs, LinearPrecision::Int8, Finish::default())
            .unwrap()
            .to_f32()
            .unwrap();

        let (output_scales, bias) = (fill(outputs, 13), fill(outputs, 11));
        let (addend, mid, up) = (
            fill(rows * outputs, 17),
            fill(rows * rank, 19),
            fill(outputs * rank, 23),
        );
        let upload =
            |values: &[f32], r: usize, c: usize| Array::from_f32(&device, r, c, values).unwrap();
        let up_bytes: Vec<u8> = up
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        let up_buffer = device.alloc(up_bytes.len(), Some(&up_bytes)).unwrap();
        let (scales_array, bias_array) = (
            upload(&output_scales, 1, outputs),
            upload(&bias, 1, outputs),
        );
        let finished = x
            .linear_int8(
                &weight,
                outputs,
                LinearPrecision::Int8,
                Finish {
                    scales: Some(&scales_array),
                    bias: Some(&bias_array),
                    addend: Some(upload(&addend, rows, outputs)),
                    lora: Some(Lora {
                        mid: upload(&mid, rows, rank),
                        up: &up_buffer,
                    }),
                },
            )
            .unwrap()
            .to_f32()
            .unwrap();

        let half = |v: f32| mmh3_core::numeric::f16_to_f32(f32_to_f16(v));
        let mut worst = 0.0f32;
        for r in 0..rows {
            for n in 0..outputs {
                let lora: f32 = (0..rank)
                    .map(|k| half(mid[r * rank + k]) * half(up[n * rank + k]))
                    .sum();
                let expected = plain[r * outputs + n] * output_scales[n]
                    + bias[n]
                    + lora
                    + addend[r * outputs + n];
                worst = worst
                    .max((finished[r * outputs + n] - expected).abs() / expected.abs().max(1.0));
            }
        }
        assert!(worst < 1e-3, "{worst}");
    }

    /// The INT8 product's tiles are 128 rows of eight SIMD groups, so rows past the first 64 of
    /// a tile are another group's. A product that ran on too few groups left them unwritten, and
    /// every smaller test here fits in one group's rows.
    #[test]
    fn the_int8_product_covers_whole_tiles() {
        let device = Device::new().unwrap();
        if !device.supports_tensor_ops() {
            return;
        }
        let (rows, outputs, cols) = (300usize, 384usize, 512usize);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919 % 2001) as f32) / 1000.0 - 1.0)
            .collect();
        let x = Array::from_f32(&device, rows, cols, &values).unwrap();
        let bytes: Vec<u8> = (0..outputs * cols).map(|i| (i * 131 % 251) as u8).collect();
        let weight = device.alloc(bytes.len(), Some(&bytes)).unwrap();
        let product = |precision| {
            x.linear_int8(&weight, outputs, precision, Finish::default())
                .unwrap()
                .to_f32()
                .unwrap()
        };
        let (int8, fp16) = (
            product(LinearPrecision::Int8),
            product(LinearPrecision::Fp16),
        );
        let largest = fp16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = int8
            .iter()
            .zip(&fp16)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= largest * 0.02, "{worst} of {largest}");
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
