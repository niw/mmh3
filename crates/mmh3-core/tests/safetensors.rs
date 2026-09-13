use mmh3_core::safetensors::{DType, SafeTensors};
use std::path::PathBuf;

fn write_file(name: &str, header: &str, data: &[u8]) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn reads_tensors_metadata_and_data() {
    let header = r#"{"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let path = write_file("valid.safetensors", header, &[0, 0, 128, 63, 1, 2, 3, 4]);
    let file = SafeTensors::open(&path).unwrap();

    let names: Vec<&str> = file.tensors().iter().map(|tensor| tensor.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(file.metadata(), [("format".to_owned(), "pt".to_owned())]);

    let first = file.get("a").unwrap();
    assert_eq!(first.dtype, DType::F32);
    assert_eq!(f32::from_le_bytes(file.data(first).try_into().unwrap()), 1.0);
    assert_eq!(file.data(file.get("b").unwrap()), [1, 2, 3, 4]);
}

#[test]
fn rejects_size_mismatch() {
    let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let path = write_file("mismatch.safetensors", header, &[0; 4]);
    assert!(SafeTensors::open(&path).is_err());
}

#[test]
fn rejects_offsets_past_the_end() {
    let header = r#"{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
    let path = write_file("truncated.safetensors", header, &[0; 4]);
    assert!(SafeTensors::open(&path).is_err());
}
