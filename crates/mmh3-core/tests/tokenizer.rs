//! Compares the tokenizer with Hugging Face tokenizers on tests/fixtures/tokenizer_cases.json.
//!
//! The tokenizer file comes from MMH3_TOKENIZER, or from tokenizer/tokenizer.json in the MMH3_MODELS directory. The
//! test is skipped without either.

use mmh3_core::json;
use mmh3_core::tokenizer::Tokenizer;
use std::path::PathBuf;

const CASES: &str = include_str!("../../../tests/fixtures/tokenizer_cases.json");

fn tokenizer_path() -> Option<PathBuf> {
    std::env::var_os("MMH3_TOKENIZER")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("MMH3_MODELS").map(|models| PathBuf::from(models).join("tokenizer/tokenizer.json")))
}

#[test]
fn matches_hugging_face_tokenizers() {
    let Some(path) = tokenizer_path().filter(|path| path.exists()) else {
        eprintln!("skipped: set MMH3_TOKENIZER or MMH3_MODELS to a MiniMax H3 tokenizer.json");
        return;
    };
    let tokenizer = Tokenizer::load_h3(&path).unwrap();
    let cases = json::parse(CASES).unwrap();
    let mut failures = Vec::new();
    for case in cases.as_array().unwrap() {
        let text = case.get("text").unwrap().as_str().unwrap();
        let expected: Vec<u32> = case.get("ids").unwrap().as_array().unwrap().iter().map(|id| id.as_u64().unwrap() as u32).collect();
        let actual = tokenizer.encode(text);
        if actual != expected {
            failures.push(format!("{text:?}\n  expected {expected:?}\n  actual   {actual:?}"));
        }
        assert_eq!(tokenizer.decode(&actual), mmh3_core::tokenizer::nfc(text), "decode of {text:?}");
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.as_array().unwrap().len(), failures.join("\n"));
}
