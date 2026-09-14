use crate::numeric::{bf16_to_f32, f16_to_f32};
use crate::safetensors::{DType, SafeTensors, TensorInfo};

/// A dense row-major FP32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape does not match the data length"
        );
        Tensor { shape, data }
    }

    /// Reads a floating-point tensor from a checkpoint and widens it to FP32.
    pub fn load(file: &SafeTensors, info: &TensorInfo) -> Result<Self, String> {
        let bytes = file.data(info);
        let data = match info.dtype {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&chunk| bf16_to_f32(u16::from_le_bytes(chunk)))
                .collect(),
            DType::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&chunk| f16_to_f32(u16::from_le_bytes(chunk)))
                .collect(),
            other => {
                return Err(format!(
                    "tensor {} has unsupported dtype {other}",
                    info.name
                ));
            }
        };
        Ok(Tensor::new(info.shape.clone(), data))
    }
}
