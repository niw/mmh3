//! Checkpoint inspection and tensor summaries.

use crate::USAGE;
use crate::format::{format_bytes, format_count};
use mmh3_core::json;
use mmh3_core::safetensors::{DType, SafeTensors};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

struct TensorGroup {
    pattern: String,
    count: usize,
    dtype: Option<DType>,
    shape: Vec<usize>,
    uniform_shape: bool,
    elements: usize,
}

pub(crate) fn run(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut path = None;
    let mut list_all = false;
    for argument in arguments {
        match argument.as_str() {
            "--all" => list_all = true,
            _ if path.is_none() => path = Some(argument),
            _ => return Err(USAGE.into()),
        }
    }
    let path = path.ok_or(USAGE)?;
    let file = SafeTensors::open(Path::new(path))?;

    println!("file      {path}");
    println!(
        "size      {} bytes ({})",
        format_count(file.file_size()),
        format_bytes(file.file_size())
    );
    println!("header    {} bytes", format_count(file.header_size()));
    println!("tensors   {}", format_count(file.tensors().len()));
    for (key, value) in file.metadata() {
        println!("metadata  {key}: {}", truncate(value, 160));
    }
    println!();

    if list_all {
        for tensor in file.tensors() {
            println!(
                "{:<8} {:<24} {}",
                tensor.dtype,
                format_shape(&tensor.shape),
                tensor.name
            );
        }
        return Ok(());
    }

    let mut groups: Vec<TensorGroup> = Vec::new();
    let mut group_index: HashMap<String, usize> = HashMap::new();
    let mut quantization_formats: Vec<(String, usize)> = Vec::new();
    let mut bytes_by_dtype: Vec<(DType, usize)> = Vec::new();
    for tensor in file.tensors() {
        let pattern = layer_pattern(&tensor.name);
        let position = *group_index.entry(pattern.clone()).or_insert_with(|| {
            groups.push(TensorGroup {
                pattern,
                count: 0,
                dtype: Some(tensor.dtype),
                shape: tensor.shape.clone(),
                uniform_shape: true,
                elements: 0,
            });
            groups.len() - 1
        });
        let group = &mut groups[position];
        group.count += 1;
        group.elements += tensor.element_count();
        if group.dtype != Some(tensor.dtype) {
            group.dtype = None;
        }
        if group.shape != tensor.shape {
            group.uniform_shape = false;
        }

        match bytes_by_dtype
            .iter_mut()
            .find(|(dtype, _)| *dtype == tensor.dtype)
        {
            Some((_, bytes)) => *bytes += tensor.byte_count(),
            None => bytes_by_dtype.push((tensor.dtype, tensor.byte_count())),
        }

        if tensor.name.ends_with(".comfy_quant") && tensor.dtype == DType::U8 {
            let text = std::str::from_utf8(file.data(tensor))?;
            let description = describe_quantization(text);
            match quantization_formats
                .iter_mut()
                .find(|(known, _)| *known == description)
            {
                Some((_, count)) => *count += 1,
                None => quantization_formats.push((description, 1)),
            }
        }
    }

    println!(
        "{:>5}  {:<8} {:<24} {:>9}  pattern",
        "count", "dtype", "shape", "elements"
    );
    for group in &groups {
        let dtype = group.dtype.map_or("mixed", DType::name);
        let shape = if group.uniform_shape {
            format_shape(&group.shape)
        } else {
            "(varies)".to_owned()
        };
        println!(
            "{:>5}  {:<8} {:<24} {:>9}  {}",
            group.count,
            dtype,
            shape,
            format_elements(group.elements),
            group.pattern
        );
    }

    if !quantization_formats.is_empty() {
        println!();
        println!("quantization (comfy_quant)");
        for (description, count) in &quantization_formats {
            println!("{count:>5}  {description}");
        }
    }

    println!();
    println!("bytes by dtype");
    bytes_by_dtype.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
    for (dtype, bytes) in &bytes_by_dtype {
        println!("{:>8}  {}", dtype.name(), format_bytes(*bytes));
    }
    Ok(())
}

/// Replaces numeric path components with N so that per-layer tensors group together.
fn layer_pattern(name: &str) -> String {
    name.split('.')
        .map(|component| {
            if component.bytes().all(|byte| byte.is_ascii_digit()) {
                "N"
            } else {
                component
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn describe_quantization(text: &str) -> String {
    let Ok(value) = json::parse(text) else {
        return text.to_owned();
    };
    let Some(members) = value.as_object() else {
        return text.to_owned();
    };
    members
        .iter()
        .map(|(key, value)| {
            let rendered = match value {
                json::Value::String(text) => text.clone(),
                json::Value::Bool(flag) => flag.to_string(),
                json::Value::Number(number) => number.to_string(),
                other => format!("{other:?}"),
            };
            format!("{key}={rendered}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_shape(shape: &[usize]) -> String {
    let dimensions: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", dimensions.join(", "))
}

fn format_elements(value: usize) -> String {
    match value {
        1_000_000_000.. => format!("{:.3}B", value as f64 / 1e9),
        1_000_000.. => format!("{:.2}M", value as f64 / 1e6),
        1_000.. => format!("{:.1}K", value as f64 / 1e3),
        _ => value.to_string(),
    }
}

fn truncate(text: &str, limit: usize) -> String {
    match text.char_indices().nth(limit) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}
