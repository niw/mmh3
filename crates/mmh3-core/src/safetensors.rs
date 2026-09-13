//! Read-only access to safetensors files through a memory mapping.

use crate::json::{self, Value};
use crate::mapped_file::MappedFile;
use std::collections::HashMap;
use std::fmt;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    F8E4M3,
    F8E5M2,
}

impl DType {
    pub fn from_name(name: &str) -> Option<DType> {
        Some(match name {
            "BOOL" => DType::Bool,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "U16" => DType::U16,
            "I16" => DType::I16,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "U32" => DType::U32,
            "I32" => DType::I32,
            "F32" => DType::F32,
            "U64" => DType::U64,
            "I64" => DType::I64,
            "F64" => DType::F64,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "BOOL",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::U16 => "U16",
            DType::I16 => "I16",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::U32 => "U32",
            DType::I32 => "I32",
            DType::F32 => "F32",
            DType::U64 => "U64",
            DType::I64 => "I64",
            DType::F64 => "F64",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E5M2 => 1,
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.pad(self.name())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte range relative to the start of the data section.
    pub data_range: Range<usize>,
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_count(&self) -> usize {
        self.data_range.len()
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(error) => write!(formatter, "{error}"),
            Error::Format(message) => write!(formatter, "invalid safetensors file: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::Io(error)
    }
}

fn format_error(message: impl Into<String>) -> Error {
    Error::Format(message.into())
}

pub struct SafeTensors {
    path: PathBuf,
    file: MappedFile,
    data_offset: usize,
    tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    metadata: Vec<(String, String)>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = MappedFile::open(path)?;
        let bytes = file.as_bytes();
        let length_bytes: [u8; 8] = bytes
            .get(..8)
            .and_then(|prefix| prefix.try_into().ok())
            .ok_or_else(|| format_error("file is shorter than the header length field"))?;
        let header_length = usize::try_from(u64::from_le_bytes(length_bytes))
            .map_err(|_| format_error("header length does not fit in memory"))?;
        let data_offset = 8usize
            .checked_add(header_length)
            .filter(|&offset| offset <= bytes.len())
            .ok_or_else(|| format_error("header extends past the end of the file"))?;
        let header_text =
            std::str::from_utf8(&bytes[8..data_offset]).map_err(|_| format_error("header is not valid UTF-8"))?;
        let header = json::parse(header_text).map_err(|error| format_error(format!("header JSON: {error}")))?;
        let members = header.as_object().ok_or_else(|| format_error("header is not a JSON object"))?;

        let data_length = bytes.len() - data_offset;
        let mut tensors = Vec::with_capacity(members.len());
        let mut metadata = Vec::new();
        for (name, entry) in members {
            if name == "__metadata__" {
                metadata = parse_metadata(entry)?;
            } else {
                tensors.push(parse_tensor(name, entry, data_length)?);
            }
        }
        tensors.sort_by_key(|tensor| tensor.data_range.start);
        let index = tensors
            .iter()
            .enumerate()
            .map(|(position, tensor)| (tensor.name.clone(), position))
            .collect();
        Ok(Self { path: path.to_owned(), file, data_offset, tensors, index, metadata })
    }

    /// Tensors in the order their data appears in the file.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&position| &self.tensors[position])
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Offset of the tensor's first byte from the start of the file.
    pub fn file_offset(&self, tensor: &TensorInfo) -> u64 {
        (self.data_offset + tensor.data_range.start) as u64
    }

    pub fn data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = self.data_offset + tensor.data_range.start;
        &self.file.as_bytes()[start..start + tensor.byte_count()]
    }

    pub fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }

    pub fn file_size(&self) -> usize {
        self.file.as_bytes().len()
    }

    pub fn header_size(&self) -> usize {
        self.data_offset - 8
    }
}

fn parse_metadata(entry: &Value) -> Result<Vec<(String, String)>, Error> {
    let members = entry.as_object().ok_or_else(|| format_error("__metadata__ is not an object"))?;
    members
        .iter()
        .map(|(key, value)| {
            let text = value.as_str().ok_or_else(|| format_error(format!("metadata value for {key} is not a string")))?;
            Ok((key.clone(), text.to_owned()))
        })
        .collect()
}

fn parse_tensor(name: &str, entry: &Value, data_length: usize) -> Result<TensorInfo, Error> {
    let invalid = |message: &str| format_error(format!("tensor {name}: {message}"));
    let dtype_name = entry.get("dtype").and_then(Value::as_str).ok_or_else(|| invalid("missing dtype"))?;
    let dtype = DType::from_name(dtype_name).ok_or_else(|| invalid(&format!("unsupported dtype {dtype_name}")))?;
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing shape"))?
        .iter()
        .map(|dimension| dimension.as_u64().and_then(|value| usize::try_from(value).ok()))
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| invalid("shape is not a list of sizes"))?;
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| invalid("missing data_offsets"))?;
    let begin = offsets[0].as_u64().and_then(|value| usize::try_from(value).ok());
    let end = offsets[1].as_u64().and_then(|value| usize::try_from(value).ok());
    let (Some(begin), Some(end)) = (begin, end) else {
        return Err(invalid("data_offsets are not sizes"));
    };
    if begin > end || end > data_length {
        return Err(invalid("data_offsets are outside the data section"));
    }
    let expected_bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |total, &dimension| total.checked_mul(dimension))
        .ok_or_else(|| invalid("shape is too large"))?;
    if expected_bytes != end - begin {
        return Err(invalid(&format!("shape needs {expected_bytes} bytes but the data has {}", end - begin)));
    }
    Ok(TensorInfo { name: name.to_owned(), dtype, shape, data_range: begin..end })
}
