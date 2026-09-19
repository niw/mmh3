//! Pieces shared by the model runners: errors, checkpoint tensors on the device and cuBLASLt linear
//! layers.

use crate::gemm::{
    AdapterPointers, Int8Output, int8_pointers, interleave_swiglu_rows, merge_low_rank,
    rotate_quantize_pointers,
};
use crate::loader::{LoadError, Uploader};
use crate::{CudaError, DeviceBuffer, check};
use mmh3_core::json;
use mmh3_core::safetensors::{DType, SafeTensors, TensorInfo};
use mmh3_core::tensor::Tensor;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::fmt;
use std::ptr;

unsafe extern "C" {
    fn mmh3_cublaslt_matmul(
        kind: c_int,
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        alpha: f32,
        beta: f32,
        stream: *mut c_void,
    ) -> c_int;
    fn mmh3_cublaslt_linear(
        kind: c_int,
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut c_void,
    ) -> c_int;
}

/// ConvRot rotates activations in groups of this many features.
pub(crate) const CONVROT_GROUP: usize = 256;

/// The INT8 GEMM adds low-rank adapters whose rank is a multiple of this inside its epilogue.
pub(crate) const ADAPTER_RANK_MULTIPLE: usize = 64;

#[derive(Debug)]
pub enum Error {
    Cuda(CudaError),
    Model(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cuda(error) => write!(formatter, "{error}"),
            Error::Model(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<mmh3_core::shard::ExchangeError> for Error {
    fn from(error: mmh3_core::shard::ExchangeError) -> Self {
        Error::Model(error.0)
    }
}

impl From<CudaError> for Error {
    fn from(error: CudaError) -> Self {
        Error::Cuda(error)
    }
}

impl From<LoadError> for Error {
    fn from(error: LoadError) -> Self {
        match error {
            LoadError::Cuda(error) => Error::Cuda(error),
            LoadError::Io(error) => Error::Model(format!("reading weights: {error}")),
        }
    }
}

pub(crate) struct DeviceTensor {
    pub(crate) buffer: DeviceBuffer,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
}

/// Checkpoint tensors on the device, keyed by their names without the checkpoint prefix.
#[derive(Default)]
pub(crate) struct DeviceTensors(HashMap<String, DeviceTensor>);

impl DeviceTensors {
    /// Allocates the tensor on the device and queues its bytes on `uploader`, which fills it when
    /// it runs.
    pub(crate) fn insert(
        &mut self,
        name: &str,
        file: &SafeTensors,
        info: &TensorInfo,
        uploader: &mut Uploader,
    ) -> Result<(), CudaError> {
        let buffer = uploader.allocate(file, info)?;
        self.0.insert(
            name.to_owned(),
            DeviceTensor {
                buffer,
                dtype: info.dtype,
                shape: info.shape.clone(),
            },
        );
        Ok(())
    }

    pub(crate) fn insert_buffer(
        &mut self,
        name: &str,
        buffer: DeviceBuffer,
        dtype: DType,
        shape: Vec<usize>,
    ) {
        self.0.insert(
            name.to_owned(),
            DeviceTensor {
                buffer,
                dtype,
                shape,
            },
        );
    }

    pub(crate) fn get(&self, name: &str) -> Result<&DeviceTensor, Error> {
        self.0
            .get(name)
            .ok_or_else(|| Error::Model(format!("missing tensor {name}")))
    }

    pub(crate) fn optional(&self, name: &str) -> Option<&DeviceTensor> {
        self.0.get(name)
    }

    pub(crate) fn pointer(&self, name: &str) -> Result<*const c_void, Error> {
        Ok(self.get(name)?.buffer.pointer())
    }

    /// Reorders the rows of `name` for the INT8 GEMM's SwiGLU output, see `interleave_swiglu_rows`.
    pub(crate) fn interleave_swiglu(&mut self, name: &str) -> Result<(), Error> {
        let tensor = self
            .0
            .get_mut(name)
            .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        let rows = tensor.shape[0];
        tensor.buffer = interleave_swiglu_rows(&tensor.buffer, rows, tensor.buffer.bytes() / rows)?;
        Ok(())
    }

    /// Merges `up · down` into the INT8 ConvRot layer `name`, see `gemm::merge_low_rank`.
    pub(crate) fn merge_low_rank(
        &mut self,
        name: &str,
        up: &DeviceBuffer,
        down: &mut DeviceBuffer,
        rank: usize,
    ) -> Result<(), Error> {
        let weight_name = format!("{name}.weight");
        let scale_name = format!("{name}.weight_scale");
        let (Some(mut weight), Some(mut scales)) =
            (self.0.remove(&weight_name), self.0.remove(&scale_name))
        else {
            return Err(Error::Model(format!("{name} is not an INT8 ConvRot layer")));
        };
        let (outputs, features) = (weight.shape[0], weight.shape[1]);
        let result = merge_low_rank(
            &mut weight.buffer,
            &mut scales.buffer,
            up,
            down,
            outputs,
            features,
            rank,
        );
        self.0.insert(weight_name, weight);
        self.0.insert(scale_name, scales);
        Ok(result?)
    }

    /// Whether `{name}.weight` is INT8, so that `linear_swiglu` applies once its rows are
    /// interleaved.
    pub(crate) fn is_int8(&self, name: &str) -> bool {
        self.optional(&format!("{name}.weight"))
            .is_some_and(|weight| weight.dtype == DType::I8)
    }

    /// Applies the INT8 ConvRot layer `name` to rows whose rotated INT8 values and scales already
    /// sit in `quantized` and `activation_scales`. With `swiglu`, the weights, weight scales and
    /// adapter up weights went through `interleave_swiglu` and the output is `silu(gate) · up`,
    /// `rows × outputs / 2` BF16 values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_quantized(
        &self,
        name: &str,
        output: *mut c_void,
        rows: usize,
        quantized: &DeviceBuffer,
        activation_scales: &DeviceBuffer,
        adapter: Option<(&LowRank, &DeviceBuffer)>,
        swiglu: bool,
    ) -> Result<(), Error> {
        let mut output = Int8Output::bf16(output);
        output.swiglu = swiglu;
        self.int8_linear(
            name,
            ptr::null(),
            true,
            output,
            rows,
            quantized,
            activation_scales,
            adapter,
        )
    }

    /// The down projection of `name`'s adapter over `rows` of a quantized input, which the column
    /// ranges of `linear_quantized_range` then share: it depends on the input alone, and a block
    /// runs three of them over the same rows.
    ///
    /// # Safety
    /// `quantized` must hold `rows × inputs` values and `activation_scales` `rows`.
    pub(crate) unsafe fn adapter_down(
        &self,
        adapter: Option<(&LowRank, &DeviceBuffer)>,
        name: &str,
        rows: usize,
        quantized: *const c_void,
        activation_scales: *const c_void,
    ) -> Result<Option<AdapterPointers>, Error> {
        match adapter {
            Some((low_rank, scratch)) if low_rank.rank % ADAPTER_RANK_MULTIPLE == 0 => {
                // SAFETY: the caller guarantees the input extents.
                Ok(Some(unsafe {
                    low_rank.project_down_quantized(quantized, activation_scales, rows, scratch)?
                }))
            }
            Some(_) => Err(Error::Model(format!(
                "{name}: the adapter rank must be a multiple of {ADAPTER_RANK_MULTIPLE}"
            ))),
            None => Ok(None),
        }
    }

    /// `linear_quantized` over a range of the output columns, with the rows of the result `stride`
    /// elements apart. The weight is `[outputs, features]` and its scales `[outputs]`, so a range of
    /// the columns is a range of both, and so is the adapter's up projection, `[outputs, rank]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_quantized_range(
        &self,
        name: &str,
        output: *mut c_void,
        rows: usize,
        columns: std::ops::Range<usize>,
        stride: usize,
        quantized: *const c_void,
        activation_scales: *const c_void,
        adapter: Option<AdapterPointers>,
    ) -> Result<(), Error> {
        let weight = self.get(&format!("{name}.weight"))?;
        let (outputs, features) = (weight.shape[0], weight.shape[1]);
        let weight_scales = self.get(&format!("{name}.weight_scale"))?;
        if weight.dtype != DType::I8
            || features % CONVROT_GROUP != 0
            || self.optional(&format!("{name}.bias")).is_some()
        {
            return Err(Error::Model(format!("{name}: unsupported INT8 layer")));
        }
        if columns.end > outputs {
            return Err(Error::Model(format!(
                "{name}: columns {columns:?} of {outputs}"
            )));
        }
        let mut output = Int8Output::bf16(output);
        output.stride = stride;
        let adapter = adapter.map(|mut adapter| {
            // SAFETY: the up projection holds `outputs × rank` BF16 values and the range lies
            // inside it, checked above.
            adapter.up = unsafe { adapter.up.byte_add(columns.start * adapter.rank * 2) };
            adapter
        });
        // SAFETY: the caller keeps `rows × features` quantized values and `rows` scales, and a
        // range of the columns is the matching range of the weight's rows and of its scales.
        unsafe {
            int8_pointers(
                quantized,
                weight.buffer.pointer_at(columns.start * features),
                activation_scales,
                weight_scales.buffer.pointer_at(columns.start * 4),
                output,
                rows,
                columns.len(),
                features,
                adapter,
            )?;
        }
        Ok(())
    }

    /// The INT8 ConvRot path of `linear`: quantizes the BF16 input unless `input_quantized`, and
    /// runs the INT8 GEMM with the adapter inside it. The adapter's down projection runs on the
    /// quantized input.
    #[allow(clippy::too_many_arguments)]
    fn int8_linear(
        &self,
        name: &str,
        input: *const c_void,
        input_quantized: bool,
        output: Int8Output,
        rows: usize,
        quantized: &DeviceBuffer,
        activation_scales: &DeviceBuffer,
        adapter: Option<(&LowRank, &DeviceBuffer)>,
    ) -> Result<(), Error> {
        let weight = self.get(&format!("{name}.weight"))?;
        let (outputs, features) = (weight.shape[0], weight.shape[1]);
        let weight_scales = self.get(&format!("{name}.weight_scale"))?;
        if weight.dtype != DType::I8
            || features % CONVROT_GROUP != 0
            || self.optional(&format!("{name}.bias")).is_some()
        {
            return Err(Error::Model(format!("{name}: unsupported INT8 layer")));
        }
        assert!(
            quantized.bytes() >= rows * features && activation_scales.bytes() >= rows * 4,
            "{name}: quantization buffers are too small"
        );
        // SAFETY: `quantized` holds `rows × features` values and `activation_scales` `rows`,
        // checked above. The adapter writes `rows × scratch_columns()` values into its scratch
        // buffer.
        unsafe {
            if !input_quantized {
                rotate_quantize_pointers(
                    input,
                    false,
                    quantized.pointer(),
                    activation_scales.pointer(),
                    rows,
                    features,
                )?;
            }
            let adapter = match adapter {
                Some((low_rank, scratch)) if low_rank.rank % ADAPTER_RANK_MULTIPLE == 0 => {
                    Some(low_rank.project_down_quantized(
                        quantized.pointer(),
                        activation_scales.pointer(),
                        rows,
                        scratch,
                    )?)
                }
                Some(_) => {
                    return Err(Error::Model(format!(
                        "{name}: the adapter rank must be a multiple of {ADAPTER_RANK_MULTIPLE}"
                    )));
                }
                None => None,
            };
            int8_pointers(
                quantized.pointer(),
                weight.buffer.pointer(),
                activation_scales.pointer(),
                weight_scales.buffer.pointer(),
                output,
                rows,
                outputs,
                features,
                adapter,
            )?;
        }
        Ok(())
    }
    /// Applies `{name}.weight` (and `{name}.bias` when present) to `rows` rows, plus a low-rank
    /// adapter with a scratch buffer of `rows × rank` BF16 values. The input and output types
    /// follow the weight: BF16 for BF16 and INT8 ConvRot weights, FP32 for FP32 weights. INT8
    /// ConvRot layers quantize the rotated input into `quantized` and `activation_scales` and add
    /// an adapter whose rank is a multiple of 64 inside the INT8 GEMM.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear(
        &self,
        name: &str,
        input: *const c_void,
        output: *mut c_void,
        rows: usize,
        quantized: &DeviceBuffer,
        activation_scales: &DeviceBuffer,
        adapter: Option<(&LowRank, &DeviceBuffer)>,
    ) -> Result<(), Error> {
        let weight = self.get(&format!("{name}.weight"))?;
        let (outputs, features) = (weight.shape[0], weight.shape[1]);
        let bias = self
            .optional(&format!("{name}.bias"))
            .map_or(ptr::null(), |bias| bias.buffer.pointer().cast_const());
        match weight.dtype {
            DType::I8 => self.int8_linear(
                name,
                input,
                false,
                Int8Output::bf16(output),
                rows,
                quantized,
                activation_scales,
                adapter,
            )?,
            DType::BF16 | DType::F32 => {
                let kind = if weight.dtype == DType::BF16 {
                    LinearKind::Bf16
                } else {
                    LinearKind::F32
                };
                // SAFETY: the caller passes buffers of `rows × features` inputs and
                // `rows × outputs` outputs, and the adapter's scratch buffer holds `rows × rank`
                // values.
                unsafe {
                    cublaslt_linear(
                        kind,
                        input,
                        weight.buffer.pointer(),
                        bias,
                        output,
                        rows,
                        outputs,
                        features,
                    )?;
                    if let Some((low_rank, scratch)) = adapter {
                        low_rank.apply(input, output, rows, scratch)?;
                    }
                }
            }
            other => {
                return Err(Error::Model(format!(
                    "{name}: unsupported weight dtype {other}"
                )));
            }
        }
        Ok(())
    }
}

pub(crate) fn host_tensor(file: &SafeTensors, name: &str) -> Result<Tensor, Error> {
    let info = file
        .get(name)
        .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
    Tensor::load(file, info).map_err(Error::Model)
}

pub(crate) fn i32_buffer(values: &[usize]) -> Result<DeviceBuffer, CudaError> {
    DeviceBuffer::from_bytes(
        &values
            .iter()
            .flat_map(|&value| (value as i32).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn check_quantization(name: &str, bytes: &[u8]) -> Result<(), Error> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| Error::Model(format!("{name} is not UTF-8")))?;
    let value = json::parse(text).map_err(|error| Error::Model(format!("{name}: {error}")))?;
    let format = value.get("format").and_then(json::Value::as_str);
    let convrot = value.get("convrot").and_then(json::Value::as_bool);
    let group = value.get("convrot_groupsize").and_then(json::Value::as_u64);
    if format != Some("int8_tensorwise")
        || convrot != Some(true)
        || group != Some(CONVROT_GROUP as u64)
    {
        return Err(Error::Model(format!(
            "{name}: unsupported quantization {text}"
        )));
    }
    Ok(())
}

/// Element type of a cuBLASLt linear layer's input, weight, bias and output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearKind {
    Bf16 = 0,
    F32 = 1,
    F16 = 2,
}

/// `output[rows, outputs] = input[rows, features] · weightᵀ + bias` through cuBLASLt.
///
/// # Safety
/// `input` must hold `rows × features` elements, `weight` `outputs × features`, `bias` `outputs` or
/// be null, and `output` `rows × outputs`, all of `kind`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cublaslt_linear(
    kind: LinearKind,
    input: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    output: *mut c_void,
    rows: usize,
    outputs: usize,
    features: usize,
) -> Result<(), CudaError> {
    // SAFETY: the caller guarantees the extents.
    check(unsafe {
        mmh3_cublaslt_linear(
            kind as c_int,
            input,
            weight,
            bias,
            output,
            rows as i64,
            outputs as i64,
            features as i64,
            ptr::null_mut(),
        )
    })
}

/// The down projection of a low-rank adapter.
pub(crate) enum Down {
    /// BF16 `[rank, inputs]`, applied to BF16 inputs.
    Bf16(DeviceBuffer),
    /// For INT8 ConvRot layers: the rows rotated like the activations and quantized to INT8
    /// `[columns, inputs]` with one scale per row, applied to the layer's quantized input. `columns`
    /// is the rank padded with zero rows to a multiple of 128, the INT8 GEMM's tile width.
    Int8 {
        weights: DeviceBuffer,
        scales: DeviceBuffer,
        columns: usize,
    },
}

/// A low-rank adapter of one linear layer: output += scale · up · (down · input).
pub(crate) struct LowRank {
    pub(crate) down: Down,
    /// BF16 `[outputs, rank]`.
    pub(crate) up: DeviceBuffer,
    pub(crate) rank: usize,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) scale: f32,
}

impl LowRank {
    /// Quantizes a BF16 down projection `[rank, inputs]`, given as its little-endian bytes, for an
    /// INT8 ConvRot layer, see `Down::Int8`.
    pub(crate) fn quantize_down(
        down: &[u8],
        rank: usize,
        inputs: usize,
    ) -> Result<Down, CudaError> {
        let columns = rank.next_multiple_of(128);
        let mut padded = down.to_vec();
        padded.resize(columns * inputs * 2, 0);
        let padded = DeviceBuffer::from_bytes(&padded)?;
        let weights = DeviceBuffer::new(columns * inputs)?;
        let scales = DeviceBuffer::new(columns * 4)?;
        // SAFETY: `padded` holds `columns × inputs` BF16 values, `weights` as many bytes and
        // `scales` `columns` f32 values.
        unsafe {
            rotate_quantize_pointers(
                padded.pointer(),
                false,
                weights.pointer(),
                scales.pointer(),
                columns,
                inputs,
            )?
        };
        Ok(Down::Int8 {
            weights,
            scales,
            columns,
        })
    }

    /// Values per input row the adapter's scratch buffer needs.
    pub(crate) fn scratch_columns(&self) -> usize {
        match self.down {
            Down::Bf16(_) => self.rank,
            Down::Int8 { columns, .. } => columns,
        }
    }

    /// Writes the INT8 down projection of `rows` quantized input rows into `scratch`, which holds
    /// at least `rows × scratch_columns()` BF16 values, and returns the operands for an INT8 GEMM
    /// that adds the rest.
    ///
    /// # Safety
    /// `quantized` and `activation_scales` must hold `rows × inputs` INT8 values and `rows` scales.
    pub(crate) unsafe fn project_down_quantized(
        &self,
        quantized: *const c_void,
        activation_scales: *const c_void,
        rows: usize,
        scratch: &DeviceBuffer,
    ) -> Result<AdapterPointers, Error> {
        let Down::Int8 {
            weights,
            scales,
            columns,
        } = &self.down
        else {
            return Err(Error::Model(
                "the adapter of an INT8 layer needs a quantized down projection".to_owned(),
            ));
        };
        assert!(
            scratch.bytes() >= rows * columns * 2,
            "the adapter scratch buffer is too small"
        );
        // SAFETY: the caller guarantees the input extents, the down projection holds `columns ×
        // inputs` values and scratch `rows × columns`, checked above.
        unsafe {
            int8_pointers(
                quantized,
                weights.pointer(),
                activation_scales,
                scales.pointer(),
                Int8Output::bf16(scratch.pointer()),
                rows,
                *columns,
                self.inputs,
                None,
            )?
        };
        Ok(AdapterPointers {
            down: scratch.pointer(),
            up: self.up.pointer(),
            rank: self.rank,
            down_stride: *columns,
            scale: self.scale,
        })
    }

    /// Adds the adapter's contribution for `rows` BF16 input rows to the BF16 output, through
    /// `scratch`, which holds at least `rows × rank` BF16 values.
    ///
    /// # Safety
    /// `input` must hold `rows × inputs` BF16 values and `output` `rows × outputs`.
    pub(crate) unsafe fn apply(
        &self,
        input: *const c_void,
        output: *mut c_void,
        rows: usize,
        scratch: &DeviceBuffer,
    ) -> Result<(), Error> {
        let Down::Bf16(down) = &self.down else {
            return Err(Error::Model(
                "a quantized down projection needs an INT8 layer".to_owned(),
            ));
        };
        assert!(
            scratch.bytes() >= rows * self.rank * 2,
            "the adapter scratch buffer is too small"
        );
        // SAFETY: the caller guarantees the input and output extents, and scratch holds
        // `rows × rank`, checked above.
        unsafe {
            cublaslt_linear(
                LinearKind::Bf16,
                input,
                down.pointer(),
                ptr::null(),
                scratch.pointer(),
                rows,
                self.rank,
                self.inputs,
            )?;
            check(mmh3_cublaslt_matmul(
                LinearKind::Bf16 as c_int,
                scratch.pointer(),
                self.up.pointer(),
                ptr::null(),
                output,
                rows as i64,
                self.outputs as i64,
                self.rank as i64,
                self.scale,
                1.0,
                ptr::null_mut(),
            ))?;
        }
        Ok(())
    }
}
