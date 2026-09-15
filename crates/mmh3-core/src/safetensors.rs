//! Safetensors files: read-only access through a memory mapping, and writing FP32 tensors.

use crate::json::{self, Value};
use crate::mapped_file::MappedFile;
use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
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
        let header_text = std::str::from_utf8(&bytes[8..data_offset])
            .map_err(|_| format_error("header is not valid UTF-8"))?;
        let header = json::parse(header_text)
            .map_err(|error| format_error(format!("header JSON: {error}")))?;
        let members = header
            .as_object()
            .ok_or_else(|| format_error("header is not a JSON object"))?;

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
        Ok(Self {
            path: path.to_owned(),
            file,
            data_offset,
            tensors,
            index,
            metadata,
        })
    }

    /// Tensors in the order their data appears in the file.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.index
            .get(name)
            .map(|&position| &self.tensors[position])
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

/// Writes FP32 tensors `(name, shape, values)` and string metadata as a safetensors file.
pub fn write_f32(
    path: &Path,
    tensors: &[(&str, &[usize], &[f32])],
    metadata: &[(&str, &str)],
) -> io::Result<()> {
    let entries: Vec<String> = metadata
        .iter()
        .map(|(key, value)| format!("{}:{}", json::quote(key), json::quote(value)))
        .collect();
    let mut header = format!("{{\"__metadata__\":{{{}}}", entries.join(","));
    let mut offset = 0;
    for (name, shape, values) in tensors {
        assert_eq!(
            shape.iter().product::<usize>(),
            values.len(),
            "{name}: the shape does not match the values"
        );
        let end = offset + values.len() * 4;
        let dimensions: Vec<String> = shape.iter().map(usize::to_string).collect();
        header.push_str(&format!(
            ",{}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{offset},{end}]}}",
            json::quote(name),
            dimensions.join(",")
        ));
        offset = end;
    }
    header.push('}');
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut writer = io::BufWriter::new(std::fs::File::create(path)?);
    writer.write_all(&(header.len() as u64).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for (_, _, values) in tensors {
        for value in values.iter() {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    writer.flush()
}

fn parse_metadata(entry: &Value) -> Result<Vec<(String, String)>, Error> {
    let members = entry
        .as_object()
        .ok_or_else(|| format_error("__metadata__ is not an object"))?;
    members
        .iter()
        .map(|(key, value)| {
            let text = value
                .as_str()
                .ok_or_else(|| format_error(format!("metadata value for {key} is not a string")))?;
            Ok((key.clone(), text.to_owned()))
        })
        .collect()
}

fn parse_tensor(name: &str, entry: &Value, data_length: usize) -> Result<TensorInfo, Error> {
    let invalid = |message: &str| format_error(format!("tensor {name}: {message}"));
    let dtype_name = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("missing dtype"))?;
    let dtype = DType::from_name(dtype_name)
        .ok_or_else(|| invalid(&format!("unsupported dtype {dtype_name}")))?;
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing shape"))?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
        })
        .collect::<Option<Vec<usize>>>()
        .ok_or_else(|| invalid("shape is not a list of sizes"))?;
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| invalid("missing data_offsets"))?;
    let begin = offsets[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok());
    let end = offsets[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok());
    let (Some(begin), Some(end)) = (begin, end) else {
        return Err(invalid("data_offsets are not sizes"));
    };
    if begin > end || end > data_length {
        return Err(invalid("data_offsets are outside the data section"));
    }
    let expected_bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |total, &dimension| {
            total.checked_mul(dimension)
        })
        .ok_or_else(|| invalid("shape is too large"))?;
    if expected_bytes != end - begin {
        return Err(invalid(&format!(
            "shape needs {expected_bytes} bytes but the data has {}",
            end - begin
        )));
    }
    Ok(TensorInfo {
        name: name.to_owned(),
        dtype,
        shape,
        data_range: begin..end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_back_written_tensors_and_metadata() {
        let path =
            std::env::temp_dir().join(format!("mmh3-write-{}.safetensors", std::process::id()));
        let values = [1.0f32, -2.5, 3.25, 0.0, 5.0, 6.0];
        write_f32(
            &path,
            &[("video", &[2, 3], &values), ("audio", &[1], &[7.0])],
            &[("prompt", "a \"quoted\" prompt")],
        )
        .unwrap();
        let file = SafeTensors::open(&path).unwrap();
        assert_eq!(
            file.metadata(),
            &[("prompt".to_owned(), "a \"quoted\" prompt".to_owned())]
        );
        let video = file.get("video").unwrap();
        assert_eq!(
            (video.dtype, video.shape.as_slice()),
            (DType::F32, &[2, 3][..])
        );
        let read: Vec<f32> = file
            .data(video)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(read, values);
        assert_eq!(file.data(file.get("audio").unwrap()), 7.0f32.to_le_bytes());
        std::fs::remove_file(&path).unwrap();
    }
}
