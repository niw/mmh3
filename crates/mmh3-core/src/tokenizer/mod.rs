//! The byte-level BPE tokenizer of Qwen2 and Qwen3, read from a Hugging Face `tokenizer.json`.
//!
//! Encoding splits the text on added tokens, applies NFC to the rest, splits it with the Qwen2 pre-tokenizer, maps
//! every byte to its printable stand-in and merges the pieces by rank. No special tokens are added around the text.

mod pretokenize;
mod unicode;
mod unicode_tables;

pub use pretokenize::pretokenize;
pub use unicode::{is_letter, nfc};

use crate::json::{self, Value};
use std::collections::HashMap;
use std::path::Path;

/// Tokens MiniMax H3 adds to the Qwen3 vocabulary, with the ids of the released tokenizer.
pub const H3_EXTRA_TOKENS: [(&str, u32); 7] = [
    ("<d>", 151_669),
    ("</d>", 151_670),
    ("<|cutoff|>", 151_671),
    ("<|lyrics_start|>", 151_672),
    ("<|lyrics_end|>", 151_673),
    ("<|caption_start|>", 151_674),
    ("<|caption_end|>", 151_675),
];

/// GPT-2's printable stand-ins for the 256 byte values.
fn byte_characters() -> [char; 256] {
    let mut characters = ['\0'; 256];
    let mut next = 256;
    for byte in 0..256u32 {
        let printable = (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte);
        characters[byte as usize] = if printable {
            char::from_u32(byte).unwrap()
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
    }
    characters
}

pub struct Tokenizer {
    vocabulary: HashMap<String, u32>,
    /// Token text by id, byte-level for the BPE vocabulary and verbatim for added tokens.
    tokens: HashMap<u32, String>,
    /// Rank and merged id of each mergeable pair of ids.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens, matched verbatim in the input before normalization.
    added: Vec<(String, u32)>,
    byte_tokens: [u32; 256],
    character_bytes: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
        Self::from_json(&text)
    }

    /// The MiniMax H3 tokenizer: `tokenizer.json` of Qwen3 plus [`H3_EXTRA_TOKENS`].
    pub fn load_h3(path: &Path) -> Result<Self, String> {
        let mut tokenizer = Self::load(path)?;
        for (content, id) in H3_EXTRA_TOKENS {
            tokenizer.add_token(content, id);
        }
        Ok(tokenizer)
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let root = json::parse(text).map_err(|error| format!("tokenizer.json: {error}"))?;
        let model = root.get("model").ok_or("tokenizer.json has no model")?;
        if model.get("type").and_then(Value::as_str) != Some("BPE") {
            return Err("only BPE tokenizers are supported".into());
        }
        let mut vocabulary = HashMap::new();
        for (token, id) in model.get("vocab").and_then(Value::as_object).ok_or("the model has no vocab")? {
            let id = id.as_u64().ok_or("vocab ids must be integers")? as u32;
            vocabulary.insert(token.clone(), id);
        }
        let tokens: HashMap<u32, String> = vocabulary.iter().map(|(token, &id)| (id, token.clone())).collect();

        let mut merges = HashMap::new();
        for (rank, merge) in model.get("merges").and_then(Value::as_array).ok_or("the model has no merges")?.iter().enumerate() {
            let (left, right) = match merge {
                Value::String(pair) => pair.split_once(' ').ok_or("a merge is not a pair")?,
                Value::Array(pair) if pair.len() == 2 => {
                    (pair[0].as_str().ok_or("a merge is not a pair")?, pair[1].as_str().ok_or("a merge is not a pair")?)
                }
                _ => return Err("a merge is not a pair".into()),
            };
            let id = |token: &str| vocabulary.get(token).copied().ok_or_else(|| format!("merge part {token:?} is not in the vocab"));
            let merged = id(&format!("{left}{right}"))?;
            merges.entry((id(left)?, id(right)?)).or_insert((rank as u32, merged));
        }

        let characters = byte_characters();
        let mut byte_tokens = [0; 256];
        for (byte, character) in characters.iter().enumerate() {
            byte_tokens[byte] = *vocabulary.get(&character.to_string()).ok_or("the vocab lacks a byte token")?;
        }
        let character_bytes = characters.iter().enumerate().map(|(byte, &character)| (character, byte as u8)).collect();

        let mut tokenizer = Tokenizer { vocabulary, tokens, merges, added: Vec::new(), byte_tokens, character_bytes };
        for added in root.get("added_tokens").and_then(Value::as_array).unwrap_or(&[]) {
            let content = added.get("content").and_then(Value::as_str).ok_or("an added token has no content")?;
            let id = added.get("id").and_then(Value::as_u64).ok_or("an added token has no id")? as u32;
            tokenizer.add_token(content, id);
        }
        Ok(tokenizer)
    }

    /// Adds a token that is matched verbatim in the input, like the special tokens of `tokenizer.json`.
    pub fn add_token(&mut self, content: &str, id: u32) {
        self.added.retain(|(existing, _)| existing != content);
        self.added.push((content.to_owned(), id));
        self.tokens.insert(id, content.to_owned());
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.added.iter().find(|(content, _)| content == token).map(|&(_, id)| id).or_else(|| self.vocabulary.get(token).copied())
    }

    /// The leftmost, then longest, added token in `text`, as its byte offset, length and id.
    fn next_added(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (content, id) in &self.added {
            if let Some(offset) = text.find(content.as_str()) {
                let better = match best {
                    None => true,
                    Some((best_offset, best_length, _)) => offset < best_offset || (offset == best_offset && content.len() > best_length),
                };
                if better {
                    best = Some((offset, content.len(), *id));
                }
            }
        }
        best
    }

    fn encode_piece(&self, piece: &str, output: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = piece.bytes().map(|byte| self.byte_tokens[byte as usize]).collect();
        loop {
            let best = symbols
                .windows(2)
                .filter_map(|pair| self.merges.get(&(pair[0], pair[1])).map(|&(rank, merged)| (rank, (pair[0], pair[1]), merged)))
                .min_by_key(|&(rank, _, _)| rank);
            let Some((_, (left, right), merged)) = best else {
                break;
            };
            let mut merged_symbols = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len() && symbols[index] == left && symbols[index + 1] == right {
                    merged_symbols.push(merged);
                    index += 2;
                } else {
                    merged_symbols.push(symbols[index]);
                    index += 1;
                }
            }
            symbols = merged_symbols;
        }
        output.extend(symbols);
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let (segment, added) = match self.next_added(rest) {
                Some((offset, length, id)) => {
                    let segment = &rest[..offset];
                    rest = &rest[offset + length..];
                    (segment, Some(id))
                }
                None => (std::mem::take(&mut rest), None),
            };
            for piece in pretokenize(&nfc(segment)) {
                self.encode_piece(&piece, &mut ids);
            }
            ids.extend(added);
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            let Some(token) = self.tokens.get(id) else {
                continue;
            };
            if self.added.iter().any(|&(_, added)| added == *id) {
                bytes.extend_from_slice(token.as_bytes());
            } else {
                bytes.extend(token.chars().filter_map(|character| self.character_bytes.get(&character)));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary of the 256 byte tokens plus "ab", "abc" and " a", in the order of their merges.
    fn tiny_tokenizer() -> Tokenizer {
        let characters = byte_characters();
        let mut entries: Vec<(String, u32)> = characters.iter().enumerate().map(|(byte, character)| (character.to_string(), byte as u32)).collect();
        let space = characters[b' ' as usize];
        entries.push(("ab".into(), 256));
        entries.push(("abc".into(), 257));
        entries.push((format!("{space}a"), 258));
        let vocab_json: Vec<String> = entries.iter().map(|(token, id)| format!("{}: {id}", json_string(token))).collect();
        let merges = [json_string("a b"), json_string("ab c"), json_string(&format!("{space} a"))];
        let text = format!(
            r#"{{"model": {{"type": "BPE", "vocab": {{{}}}, "merges": [{}]}}, "added_tokens": [{{"id": 300, "content": "<|x|>"}}]}}"#,
            vocab_json.join(", "),
            merges.join(", ")
        );
        Tokenizer::from_json(&text).unwrap()
    }

    fn json_string(text: &str) -> String {
        let mut quoted = String::from("\"");
        for character in text.chars() {
            match character {
                '"' => quoted.push_str("\\\""),
                '\\' => quoted.push_str("\\\\"),
                character if (character as u32) < 0x20 => quoted.push_str(&format!("\\u{:04x}", character as u32)),
                character => quoted.push(character),
            }
        }
        quoted.push('"');
        quoted
    }

    #[test]
    fn merges_by_rank_and_splits_added_tokens() {
        let tokenizer = tiny_tokenizer();
        assert_eq!(tokenizer.encode("abc"), [257]);
        assert_eq!(tokenizer.encode("abab"), [256, 256]);
        assert_eq!(tokenizer.encode("x ac"), [b'x' as u32, 258, b'c' as u32]);
        assert_eq!(tokenizer.encode("x abc"), [b'x' as u32, b' ' as u32, 257]);
        assert_eq!(tokenizer.encode("ab<|x|>c"), [256, 300, b'c' as u32]);
        assert_eq!(tokenizer.decode(&tokenizer.encode("x abc<|x|>é")), "x abc<|x|>é");
    }
}
