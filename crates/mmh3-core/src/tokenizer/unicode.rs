//! Unicode letter classification and NFC normalization over the generated tables.

use super::unicode_tables::{
    COMBINING_CLASSES, COMPOSITIONS, DECOMPOSITIONS, LETTERS, NFC_STABLE_BELOW,
};

const HANGUL_SYLLABLE_BASE: u32 = 0xAC00;
const HANGUL_LEADING_BASE: u32 = 0x1100;
const HANGUL_VOWEL_BASE: u32 = 0x1161;
const HANGUL_TRAILING_BASE: u32 = 0x11A7;
const HANGUL_LEADING_COUNT: u32 = 19;
const HANGUL_VOWEL_COUNT: u32 = 21;
const HANGUL_TRAILING_COUNT: u32 = 28;
const HANGUL_BLOCK: u32 = HANGUL_VOWEL_COUNT * HANGUL_TRAILING_COUNT;
const HANGUL_SYLLABLE_COUNT: u32 = HANGUL_LEADING_COUNT * HANGUL_BLOCK;

/// Whether `character` has general category L, as `\p{L}` in the pre-tokenizer pattern.
pub fn is_letter(character: char) -> bool {
    let code_point = character as u32;
    if code_point < 0x80 {
        return character.is_ascii_alphabetic();
    }
    LETTERS
        .binary_search_by(|&(start, end)| {
            if end < code_point {
                std::cmp::Ordering::Less
            } else if start > code_point {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn combining_class(code_point: u32) -> u8 {
    if code_point < NFC_STABLE_BELOW {
        return 0;
    }
    match COMBINING_CLASSES.binary_search_by(|&(start, end, _)| {
        if end < code_point {
            std::cmp::Ordering::Less
        } else if start > code_point {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(index) => COMBINING_CLASSES[index].2,
        Err(_) => 0,
    }
}

fn decompose(code_point: u32, output: &mut Vec<u32>) {
    let syllable = code_point.wrapping_sub(HANGUL_SYLLABLE_BASE);
    if syllable < HANGUL_SYLLABLE_COUNT {
        output.push(HANGUL_LEADING_BASE + syllable / HANGUL_BLOCK);
        output.push(HANGUL_VOWEL_BASE + syllable % HANGUL_BLOCK / HANGUL_TRAILING_COUNT);
        if !syllable.is_multiple_of(HANGUL_TRAILING_COUNT) {
            output.push(HANGUL_TRAILING_BASE + syllable % HANGUL_TRAILING_COUNT);
        }
        return;
    }
    match DECOMPOSITIONS.binary_search_by_key(&code_point, |&(source, _, _)| source) {
        Ok(index) => {
            let (_, first, second) = DECOMPOSITIONS[index];
            decompose(first, output);
            if second != 0 {
                decompose(second, output);
            }
        }
        Err(_) => output.push(code_point),
    }
}

fn compose(first: u32, second: u32) -> Option<u32> {
    let leading = first.wrapping_sub(HANGUL_LEADING_BASE);
    let vowel = second.wrapping_sub(HANGUL_VOWEL_BASE);
    if leading < HANGUL_LEADING_COUNT && vowel < HANGUL_VOWEL_COUNT {
        return Some(HANGUL_SYLLABLE_BASE + leading * HANGUL_BLOCK + vowel * HANGUL_TRAILING_COUNT);
    }
    let syllable = first.wrapping_sub(HANGUL_SYLLABLE_BASE);
    let trailing = second.wrapping_sub(HANGUL_TRAILING_BASE);
    if syllable < HANGUL_SYLLABLE_COUNT
        && syllable.is_multiple_of(HANGUL_TRAILING_COUNT)
        && (1..HANGUL_TRAILING_COUNT).contains(&trailing)
    {
        return Some(first + trailing);
    }
    COMPOSITIONS
        .binary_search_by_key(&(first, second), |&(start, next, _)| (start, next))
        .ok()
        .map(|index| COMPOSITIONS[index].2)
}

/// Canonical composition after canonical decomposition (Unicode normalization form C).
pub fn nfc(text: &str) -> String {
    if text
        .chars()
        .all(|character| (character as u32) < NFC_STABLE_BELOW)
    {
        return text.to_owned();
    }
    let mut decomposed = Vec::with_capacity(text.len());
    for character in text.chars() {
        decompose(character as u32, &mut decomposed);
    }
    // Canonical ordering: a stable sort of each run of non-starters by combining class.
    let mut start = 0;
    while start < decomposed.len() {
        if combining_class(decomposed[start]) == 0 {
            start += 1;
            continue;
        }
        let mut end = start;
        while end < decomposed.len() && combining_class(decomposed[end]) != 0 {
            end += 1;
        }
        decomposed[start..end].sort_by_key(|&code_point| combining_class(code_point));
        start = end;
    }

    let mut composed: Vec<u32> = Vec::with_capacity(decomposed.len());
    let mut starter = None;
    let mut last_class = 0;
    for code_point in decomposed {
        let class = combining_class(code_point);
        if let Some(starter_index) = starter {
            let adjacent = composed.len() - 1 == starter_index;
            let blocked = !adjacent && (last_class == 0 || last_class >= class);
            if !blocked && let Some(composite) = compose(composed[starter_index], code_point) {
                composed[starter_index] = composite;
                continue;
            }
        }
        if class == 0 {
            starter = Some(composed.len());
        }
        last_class = class;
        composed.push(code_point);
    }
    composed.into_iter().filter_map(char::from_u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_letters() {
        for letter in ['a', 'Z', 'é', 'ß', 'Ж', 'あ', 'ア', '漢', '한', 'ـ', 'ª'] {
            assert!(is_letter(letter), "{letter:?}");
        }
        // Devanagari vowel signs and circled letters are alphabetic but not general category L.
        for other in [
            '1', ' ', '_', '!', '\u{94D}', '\u{93E}', 'Ⓐ', '²', '🐼', '\u{301}',
        ] {
            assert!(!is_letter(other), "{other:?}");
        }
    }

    #[test]
    fn composes_canonically() {
        assert_eq!(nfc("plain ASCII"), "plain ASCII");
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("\u{212B}"), "Å");
        assert_eq!(nfc("A\u{30A}"), "Å");
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "각");
        assert_eq!(nfc("각"), "각");
        assert_eq!(nfc("か\u{3099}"), "が");
        // Canonical ordering puts the dot below (220) before the dot above (230), and both compose.
        assert_eq!(nfc("q\u{307}\u{323}"), "q\u{323}\u{307}");
        assert_eq!(nfc("s\u{307}\u{323}"), "\u{1E69}");
        // A starter in between blocks the composition.
        assert_eq!(nfc("e\u{200B}\u{301}"), "e\u{200B}\u{301}");
        assert_eq!(nfc("\u{344}"), "\u{308}\u{301}");
    }
}
