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
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
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
