//! Formatting shared by checkpoint inspection and device reports.

pub(crate) fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::new();
    for (position, digit) in digits.chars().enumerate() {
        if position > 0 && (digits.len() - position).is_multiple_of(3) {
            output.push(',');
        }
        output.push(digit);
    }
    output
}

pub(crate) fn format_bytes(value: usize) -> String {
    match value {
        1_000_000_000.. => format!("{:.2} GB", value as f64 / 1e9),
        1_000_000.. => format!("{:.2} MB", value as f64 / 1e6),
        1_000.. => format!("{:.1} KB", value as f64 / 1e3),
        _ => format!("{value} B"),
    }
}
