//! Compares the tokenizer with Hugging Face tokenizers on tests/fixtures/tokenizer_cases.json.

use mmh3_core::json;
use mmh3_core::tokenizer::Tokenizer;

const CASES: &str = include_str!("../../../tests/fixtures/tokenizer_cases.json");

#[test]
fn matches_hugging_face_tokenizers() {
    let tokenizer = Tokenizer::h3();
    let cases = json::parse(CASES).unwrap();
    let mut failures = Vec::new();
    for case in cases.as_array().unwrap() {
        let text = case.get("text").unwrap().as_str().unwrap();
        let expected: Vec<u32> = case
            .get("ids")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        let actual = tokenizer.encode(text);
        if actual != expected {
            failures.push(format!(
                "{text:?}\n  expected {expected:?}\n  actual   {actual:?}"
            ));
        }
        assert_eq!(
            tokenizer.decode(&actual),
            mmh3_core::tokenizer::nfc(text),
            "decode of {text:?}"
        );
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.as_array().unwrap().len(),
        failures.join("\n")
    );
}
