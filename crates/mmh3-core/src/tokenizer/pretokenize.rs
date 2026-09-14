//! The Qwen2 pre-tokenizer: splits normalized text into the pieces that BPE encodes separately.
//!
//! It matches this pattern leftmost-first, with the alternatives tried in order at every position:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```

use super::unicode::is_letter;

fn is_number(character: char) -> bool {
    character.is_numeric()
}

fn is_line_break(character: char) -> bool {
    character == '\r' || character == '\n'
}

fn is_other(character: char) -> bool {
    !character.is_whitespace() && !is_letter(character) && !is_number(character)
}

/// End of the run of characters from `start` that satisfy `predicate`.
fn run_end(characters: &[char], start: usize, predicate: impl Fn(char) -> bool) -> usize {
    let mut end = start;
    while end < characters.len() && predicate(characters[end]) {
        end += 1;
    }
    end
}

fn contraction(characters: &[char], start: usize) -> Option<usize> {
    if characters[start] != '\'' {
        return None;
    }
    let lower = |offset: usize| {
        characters
            .get(start + offset)
            .map(|character| character.to_ascii_lowercase())
    };
    match (lower(1), lower(2)) {
        (Some('s' | 't' | 'm' | 'd'), _) => Some(start + 2),
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(start + 3),
        _ => None,
    }
}

fn letters(characters: &[char], start: usize) -> Option<usize> {
    let first = characters[start];
    if !is_line_break(first) && !is_letter(first) && !is_number(first) {
        if characters
            .get(start + 1)
            .is_some_and(|&next| is_letter(next))
        {
            return Some(run_end(characters, start + 1, is_letter));
        }
    }
    is_letter(first).then(|| run_end(characters, start, is_letter))
}

fn symbols(characters: &[char], start: usize) -> Option<usize> {
    let begin = if characters[start] == ' '
        && characters
            .get(start + 1)
            .is_some_and(|&next| is_other(next))
    {
        start + 1
    } else {
        start
    };
    if !is_other(characters[begin]) {
        return None;
    }
    let end = run_end(characters, begin, is_other);
    Some(run_end(characters, end, is_line_break))
}

fn whitespace(characters: &[char], start: usize) -> Option<usize> {
    let end = run_end(characters, start, char::is_whitespace);
    if end == start {
        return None;
    }
    // \s*[\r\n]+ ends right after the last line break of the run.
    if let Some(last_break) = (start..end)
        .rev()
        .find(|&index| is_line_break(characters[index]))
    {
        return Some(last_break + 1);
    }
    // \s+(?!\S) leaves the last whitespace character for the next piece when a non-space follows.
    if end == characters.len() {
        return Some(end);
    }
    if end - start > 1 {
        return Some(end - 1);
    }
    Some(end)
}

/// Splits `text` into pre-tokens. Every character belongs to exactly one piece.
pub fn pretokenize(text: &str) -> Vec<String> {
    let characters: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < characters.len() {
        let end = contraction(&characters, start)
            .or_else(|| letters(&characters, start))
            .or_else(|| is_number(characters[start]).then_some(start + 1))
            .or_else(|| symbols(&characters, start))
            .or_else(|| whitespace(&characters, start))
            .expect("every character matches an alternative");
        pieces.push(characters[start..end].iter().collect());
        start = end;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<String> {
        pretokenize(text)
    }

    #[test]
    fn splits_words_numbers_and_symbols() {
        assert_eq!(split("Hello world"), ["Hello", " world"]);
        assert_eq!(
            split("I'm here, it'S 2026!"),
            [
                "I", "'m", " here", ",", " it", "'S", " ", "2", "0", "2", "6", "!"
            ]
        );
        assert_eq!(split("we'LL they're"), ["we", "'LL", " they", "'re"]);
        assert_eq!(split("a.b"), ["a", ".b"]);
        assert_eq!(split("(x) {y}"), ["(x", ")", " {", "y", "}"]);
        assert_eq!(split(" !!\n\nz"), [" !!\n\n", "z"]);
    }

    #[test]
    fn splits_whitespace_like_the_pattern() {
        assert_eq!(split("a  b"), ["a", " ", " b"]);
        assert_eq!(split("a \n b"), ["a", " \n", " b"]);
        assert_eq!(split("a\n\n  b"), ["a", "\n\n", " ", " b"]);
        assert_eq!(split("end   "), ["end", "   "]);
        assert_eq!(split("\t\tx"), ["\t", "\tx"]);
        assert_eq!(split(" "), [" "]);
    }
}
