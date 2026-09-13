//! Pieces shared by the model runners: errors, checkpoint tensors on the device and cuBLASLt linear layers.

use crate::gemm::{AdapterPointers, Int8Output, int8_pointers, interleave_swiglu_rows, rotate_quantize_pointers};
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
    /// Allocates the tensor on the device and queues its bytes on `uploader`, which fills it when it runs.
    pub(crate) fn insert(&mut self, name: &str, file: &SafeTensors, info: &TensorInfo, uploader: &mut Uploader) -> Result<(), CudaError> {
        let buffer = uploader.allocate(file, info)?;
        self.0.insert(name.to_owned(), DeviceTensor { buffer, dtype: info.dtype, shape: info.shape.clone() });
        Ok(())
    }

    pub(crate) fn insert_buffer(&mut self, name: &str, buffer: DeviceBuffer, dtype: DType, shape: Vec<usize>) {
        self.0.insert(name.to_owned(), DeviceTensor { buffer, dtype, shape });
    }

    pub(crate) fn get(&self, name: &str) -> Result<&DeviceTensor, Error> {
        self.0.get(name).ok_or_else(|| Error::Model(format!("missing tensor {name}")))
    }

    pub(crate) fn optional(&self, name: &str) -> Option<&DeviceTensor> {
        self.0.get(name)
    }

    pub(crate) fn pointer(&self, name: &str) -> Result<*const c_void, Error> {
        Ok(self.get(name)?.buffer.pointer())
    }

    /// Reorders the rows of `name` for the INT8 GEMM's SwiGLU output, see `interleave_swiglu_rows`.
    pub(crate) fn interleave_swiglu(&mut self, name: &str) -> Result<(), Error> {
        let tensor = self.0.get_mut(name).ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        let rows = tensor.shape[0];
        tensor.buffer = interleave_swiglu_rows(&tensor.buffer, rows, tensor.buffer.bytes() / rows)?;
        Ok(())
    }

    /// Whether `{name}.weight` is INT8, so that `linear_swiglu` applies once its rows are interleaved.
    pub(crate) fn is_int8(&self, name: &str) -> bool {
        self.optional(&format!("{name}.weight")).is_some_and(|weight| weight.dtype == DType::I8)
    }

    /// Applies the INT8 ConvRot layer `name` to rows whose rotated INT8 values and scales already sit in `quantized`
    /// and `activation_scales`. `input` holds the same rows in BF16 for the adapter's down projection and is not read
    /// without an adapter. With `swiglu`, the weights, weight scales and adapter up weights went through
    /// `interleave_swiglu` and the output is `silu(gate) · up`, `rows × outputs / 2` BF16 values.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_quantized(
        &self,
        name: &str,
        input: *const c_void,
        output: *mut c_void,
        rows: usize,
        quantized: &DeviceBuffer,
        activation_scales: &DeviceBuffer,
        adapter: Option<(&LowRank, &DeviceBuffer)>,
        swiglu: bool,
    ) -> Result<(), Error> {
        let mut output = Int8Output::bf16(output);
        output.swiglu = swiglu;
        self.int8_linear(name, input, true, output, rows, quantized, activation_scales, adapter)
    }

    /// The INT8 ConvRot path of `linear`: quantizes the BF16 input unless `input_quantized`, and runs the INT8 GEMM with
    /// the adapter inside it when its rank is a multiple of 64.
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
        if weight.dtype != DType::I8 || features % CONVROT_GROUP != 0 || self.optional(&format!("{name}.bias")).is_some() {
            return Err(Error::Model(format!("{name}: unsupported INT8 layer")));
        }
        assert!(quantized.bytes() >= rows * features && activation_scales.bytes() >= rows * 4, "{name}: quantization buffers are too small");
        // SAFETY: `quantized` holds `rows × features` values and `activation_scales` `rows`, checked above. The
        // adapter reads the BF16 input and writes `rows × rank` values into its scratch buffer.
        unsafe {
            let fused = match adapter {
                Some((low_rank, scratch)) if low_rank.rank % 64 == 0 => Some(low_rank.project_down(input, rows, scratch)?),
                Some(_) if output.swiglu => return Err(Error::Model(format!("{name}: the adapter rank must be a multiple of 64"))),
                _ => None,
            };
            if !input_quantized {
                rotate_quantize_pointers(input, false, quantized.pointer(), activation_scales.pointer(), rows, features)?;
            }
            int8_pointers(quantized.pointer(), weight.buffer.pointer(), activation_scales.pointer(), weight_scales.buffer.pointer(), output, rows, outputs, features, fused)?;
            if let (None, Some((low_rank, scratch))) = (fused, adapter) {
                low_rank.apply(input, output.pointer, rows, scratch)?;
            }
        }
        Ok(())
    }
    /// Applies `{name}.weight` (and `{name}.bias` when present) to `rows` rows, plus a low-rank adapter with a
    /// scratch buffer of `rows × rank` BF16 values. The input and output types follow the weight: BF16 for BF16 and
    /// INT8 ConvRot weights, FP32 for FP32 weights. INT8 ConvRot layers quantize the rotated input into `quantized` and
    /// `activation_scales` and add an adapter whose rank is a multiple of 64 inside the INT8 GEMM.
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
        let bias = self.optional(&format!("{name}.bias")).map_or(ptr::null(), |bias| bias.buffer.pointer().cast_const());
        match weight.dtype {
            DType::I8 => self.int8_linear(name, input, false, Int8Output::bf16(output), rows, quantized, activation_scales, adapter)?,
            DType::BF16 | DType::F32 => {
                let kind = if weight.dtype == DType::BF16 { LinearKind::Bf16 } else { LinearKind::F32 };
                // SAFETY: the caller passes buffers of `rows × features` inputs and `rows × outputs` outputs, and the
                // adapter's scratch buffer holds `rows × rank` values.
                unsafe {
                    cublaslt_linear(kind, input, weight.buffer.pointer(), bias, output, rows, outputs, features)?;
                    if let Some((low_rank, scratch)) = adapter {
                        low_rank.apply(input, output, rows, scratch)?;
                    }
                }
            }
            other => return Err(Error::Model(format!("{name}: unsupported weight dtype {other}"))),
        }
        Ok(())
    }
}

pub(crate) fn host_tensor(file: &SafeTensors, name: &str) -> Result<Tensor, Error> {
    let info = file.get(name).ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
    Tensor::load(file, info).map_err(Error::Model)
}

pub(crate) fn i32_buffer(values: &[usize]) -> Result<DeviceBuffer, CudaError> {
    DeviceBuffer::from_bytes(&values.iter().flat_map(|&value| (value as i32).to_le_bytes()).collect::<Vec<_>>())
}

pub(crate) fn check_quantization(name: &str, bytes: &[u8]) -> Result<(), Error> {
    let text = std::str::from_utf8(bytes).map_err(|_| Error::Model(format!("{name} is not UTF-8")))?;
    let value = json::parse(text).map_err(|error| Error::Model(format!("{name}: {error}")))?;
    let format = value.get("format").and_then(json::Value::as_str);
    let convrot = value.get("convrot").and_then(json::Value::as_bool);
    let group = value.get("convrot_groupsize").and_then(json::Value::as_u64);
    if format != Some("int8_tensorwise") || convrot != Some(true) || group != Some(CONVROT_GROUP as u64) {
        return Err(Error::Model(format!("{name}: unsupported quantization {text}")));
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
/// `input` must hold `rows × features` elements, `weight` `outputs × features`, `bias` `outputs` or be null, and
/// `output` `rows × outputs`, all of `kind`.
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
        mmh3_cublaslt_linear(kind as c_int, input, weight, bias, output, rows as i64, outputs as i64, features as i64, ptr::null_mut())
    })
}

/// A low-rank adapter of one linear layer: output += scale · up · (down · input), in BF16.
pub(crate) struct LowRank {
    /// `[rank, inputs]`.
    pub(crate) down: DeviceBuffer,
    /// `[outputs, rank]`.
    pub(crate) up: DeviceBuffer,
    pub(crate) rank: usize,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) scale: f32,
}

impl LowRank {
    /// Writes `down · input` for `rows` BF16 input rows into `scratch`, which holds at least `rows × rank` BF16 values,
    /// and returns the operands for an INT8 GEMM that adds the rest.
    ///
    /// # Safety
    /// `input` must hold `rows × inputs` BF16 values.
    pub(crate) unsafe fn project_down(&self, input: *const c_void, rows: usize, scratch: &DeviceBuffer) -> Result<AdapterPointers, CudaError> {
        assert!(scratch.bytes() >= rows * self.rank * 2, "the adapter scratch buffer is too small");
        // SAFETY: the caller guarantees the input extent, and scratch holds `rows × rank`, checked above.
        unsafe { cublaslt_linear(LinearKind::Bf16, input, self.down.pointer(), ptr::null(), scratch.pointer(), rows, self.rank, self.inputs)? };
        Ok(AdapterPointers { down: scratch.pointer(), up: self.up.pointer(), rank: self.rank, scale: self.scale })
    }

    /// Adds the adapter's contribution for `rows` BF16 input rows to the BF16 output, through `scratch`, which holds
    /// at least `rows × rank` BF16 values.
    ///
    /// # Safety
    /// `input` must hold `rows × inputs` BF16 values and `output` `rows × outputs`.
    pub(crate) unsafe fn apply(&self, input: *const c_void, output: *mut c_void, rows: usize, scratch: &DeviceBuffer) -> Result<(), CudaError> {
        assert!(scratch.bytes() >= rows * self.rank * 2, "the adapter scratch buffer is too small");
        // SAFETY: the caller guarantees the input and output extents, and scratch holds `rows × rank`, checked above.
        unsafe {
            cublaslt_linear(LinearKind::Bf16, input, self.down.pointer(), ptr::null(), scratch.pointer(), rows, self.rank, self.inputs)?;
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
            ))
        }
    }
}
