//! A minimal JSON parser for safetensors headers, model configs and tokenizer files.

use std::fmt;

const MAX_DEPTH: usize = 512;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    /// Members in document order.
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.iter().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the number when it is a non-negative integer that f64 represents exactly.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(value) if *value >= 0.0 && value.fract() == 0.0 && *value <= 9_007_199_254_740_992.0 => {
                Some(*value as u64)
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(members) => Some(members),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: &'static str,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for ParseError {}

pub fn parse(text: &str) -> Result<Value, ParseError> {
    let mut parser = Parser { text, bytes: text.as_bytes(), position: 0, depth: 0 };
    parser.skip_whitespace();
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.position != parser.bytes.len() {
        return Err(parser.error("trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    position: usize,
    depth: usize,
}

impl Parser<'_> {
    fn error(&self, message: &'static str) -> ParseError {
        ParseError { offset: self.position, message }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn skip_whitespace(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.position += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b't') => self.parse_literal(b"true", Value::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", Value::Bool(false)),
            Some(b'n') => self.parse_literal(b"null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("unexpected character")),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_literal(&mut self, literal: &[u8], value: Value) -> Result<Value, ParseError> {
        if self.bytes[self.position..].starts_with(literal) {
            self.position += literal.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn enter_container(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error("nesting is too deep"));
        }
        self.position += 1;
        self.skip_whitespace();
        Ok(())
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.enter_container()?;
        let mut members = Vec::new();
        if self.peek() == Some(b'}') {
            self.position += 1;
            self.depth -= 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected an object key"));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("expected ':'"));
            }
            self.position += 1;
            self.skip_whitespace();
            let value = self.parse_value()?;
            members.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
        self.depth -= 1;
        Ok(Value::Object(members))
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.enter_container()?;
        let mut values = Vec::new();
        if self.peek() == Some(b']') {
            self.position += 1;
            self.depth -= 1;
            return Ok(Value::Array(values));
        }
        loop {
            self.skip_whitespace();
            values.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
        self.depth -= 1;
        Ok(Value::Array(values))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.position += 1;
        let mut output = String::new();
        loop {
            let run_start = self.position;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.position += 1;
            }
            // The run ends on an ASCII byte, so both ends are char boundaries.
            output.push_str(&self.text[run_start..self.position]);
            match self.peek() {
                Some(b'"') => {
                    self.position += 1;
                    return Ok(output);
                }
                Some(b'\\') => {
                    self.position += 1;
                    self.parse_escape(&mut output)?;
                }
                Some(_) => return Err(self.error("control character in string")),
                None => return Err(self.error("unterminated string")),
            }
        }
    }

    fn parse_escape(&mut self, output: &mut String) -> Result<(), ParseError> {
        let escaped = self.peek().ok_or_else(|| self.error("unterminated escape"))?;
        self.position += 1;
        match escaped {
            b'"' => output.push('"'),
            b'\\' => output.push('\\'),
            b'/' => output.push('/'),
            b'b' => output.push('\u{8}'),
            b'f' => output.push('\u{c}'),
            b'n' => output.push('\n'),
            b'r' => output.push('\r'),
            b't' => output.push('\t'),
            b'u' => {
                let first = self.parse_hex4()?;
                let code_point = match first {
                    0xD800..=0xDBFF => {
                        if !self.bytes[self.position..].starts_with(b"\\u") {
                            return Err(self.error("unpaired surrogate"));
                        }
                        self.position += 2;
                        let second = self.parse_hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&second) {
                            return Err(self.error("invalid low surrogate"));
                        }
                        0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                    }
                    0xDC00..=0xDFFF => return Err(self.error("unpaired surrogate")),
                    _ => first,
                };
                output.push(char::from_u32(code_point).ok_or_else(|| self.error("invalid code point"))?);
            }
            _ => return Err(self.error("invalid escape")),
        }
        Ok(())
    }

    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        let digits = self
            .bytes
            .get(self.position..self.position + 4)
            .ok_or_else(|| self.error("truncated unicode escape"))?;
        let mut value = 0;
        for &digit in digits {
            let nibble = (digit as char).to_digit(16).ok_or_else(|| self.error("invalid hex digit"))?;
            value = value * 16 + nibble;
        }
        self.position += 4;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek() {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => self.skip_digits(),
            _ => return Err(self.error("invalid number")),
        }
        if self.peek() == Some(b'.') {
            self.position += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid fraction"));
            }
            self.skip_digits();
        }
        if let Some(b'e' | b'E') = self.peek() {
            self.position += 1;
            if let Some(b'+' | b'-') = self.peek() {
                self.position += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("invalid exponent"));
            }
            self.skip_digits();
        }
        self.text[start..self.position]
            .parse()
            .map(Value::Number)
            .map_err(|_| ParseError { offset: start, message: "invalid number" })
    }

    fn skip_digits(&mut self) {
        while let Some(b'0'..=b'9') = self.peek() {
            self.position += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values() {
        let value = parse(r#" {"a": [1, -2.5e2, true, null], "b": {"c": "d"}} "#).unwrap();
        assert_eq!(value.get("a").unwrap().as_array().unwrap()[1].as_f64(), Some(-250.0));
        assert_eq!(value.get("b").unwrap().get("c").unwrap().as_str(), Some("d"));
        assert_eq!(value.get("a").unwrap().as_array().unwrap()[3], Value::Null);
    }

    #[test]
    fn decodes_escapes_and_surrogate_pairs() {
        let value = parse(r#""\"\\\/\n\u00e9\ud83c\udfac""#).unwrap();
        assert_eq!(value.as_str(), Some("\"\\/\né🎬"));
    }

    #[test]
    fn keeps_non_ascii_text() {
        assert_eq!(parse("\"日本語\"").unwrap().as_str(), Some("日本語"));
    }

    #[test]
    fn converts_exact_integers() {
        assert_eq!(parse("20970379616").unwrap().as_u64(), Some(20_970_379_616));
        assert_eq!(parse("1.5").unwrap().as_u64(), None);
        assert_eq!(parse("-1").unwrap().as_u64(), None);
    }

    #[test]
    fn rejects_malformed_input() {
        for text in ["", "{", "[1,]", "{\"a\" 1}", "01", "1.", "\"\\x\"", "\"\\ud800\"", "tru", "1 2"] {
            assert!(parse(text).is_err(), "accepted {text:?}");
        }
    }
}
