//! Unicode String Processing
//!
//! UTF-8 validation and character counting (with an AVX2 all-ASCII fast
//! path), a bidirectional UTF-8 to UTF-32 iterator, lowercasing, and string
//! analysis. Two functions are approximations, and each documents its
//! limitation: [`UnicodeProcessor::with_normalization`] composes a small
//! subset of NFC, and [`utils::display_width`] uses a fixed list of Unicode
//! blocks.

use crate::error::{Result, ZiporaError};
use crate::succinct::rank_select::simd::SimdCapabilities;
use std::char;
use std::str;

/// Hardware-accelerated UTF-8 byte counting
///
/// Uses lookup table for single bytes and SIMD for batch operations
/// when available, based on research implementations.
#[inline(always)]
pub fn utf8_byte_count(byte: u8) -> usize {
    // Optimized lookup table for valid UTF-8 leading byte analysis (RFC 3629)
    static UTF8_BYTE_COUNT_LUT: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let byte = i as u8;
            table[i] = if byte < 0x80 {
                1 // ASCII (0x00..=0x7F)
            } else if byte < 0xC2 {
                0 // Continuation byte (0x80..=0xBF) or overlong 2-byte lead (0xC0..=0xC1)
            } else if byte < 0xE0 {
                2 // 2-byte sequence (0xC2..=0xDF)
            } else if byte < 0xF0 {
                3 // 3-byte sequence (0xE0..=0xEF)
            } else if byte < 0xF5 {
                4 // 4-byte sequence (0xF0..=0xF4)
            } else {
                0 // Invalid (> U+10FFFF: 0xF5..=0xFF)
            };
            i += 1;
        }
        table
    };

    UTF8_BYTE_COUNT_LUT[byte as usize] as usize
}

/// SIMD-accelerated UTF-8 validation and character counting
///
/// Processes multiple bytes in parallel when SIMD is available,
/// falling back to scalar implementation otherwise.
pub fn validate_utf8_and_count_chars(bytes: &[u8]) -> Result<usize> {
    let caps = SimdCapabilities::get();

    if caps.cpu_features.has_avx2 && bytes.len() >= 32 {
        validate_utf8_and_count_chars_simd(bytes)
    } else {
        validate_utf8_and_count_chars_scalar(bytes)
    }
}

/// Scalar UTF-8 validation and character counting
fn validate_utf8_and_count_chars_scalar(bytes: &[u8]) -> Result<usize> {
    match str::from_utf8(bytes) {
        Ok(s) => Ok(s.chars().count()),
        Err(e) => Err(ZiporaError::invalid_data(format!("Invalid UTF-8: {}", e))),
    }
}

/// SIMD UTF-8 validation (when available)
#[cfg(target_arch = "x86_64")]
fn validate_utf8_and_count_chars_simd(bytes: &[u8]) -> Result<usize> {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 verified via is_x86_feature_detected!("avx2")
        if unsafe { is_ascii_simd(bytes) } {
            // All ASCII - character count equals byte count
            return Ok(bytes.len());
        }
    }

    // Fall back to scalar for multi-byte UTF-8 sequences
    validate_utf8_and_count_chars_scalar(bytes)
}

#[cfg(not(target_arch = "x86_64"))]
fn validate_utf8_and_count_chars_simd(bytes: &[u8]) -> Result<usize> {
    validate_utf8_and_count_chars_scalar(bytes)
}

/// Check if bytes are all ASCII using AVX2
///
/// # Safety
///
/// The caller must ensure the CPU supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_ascii_simd(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;

    // SAFETY: AVX2 is guaranteed by #[target_feature], bytes.as_ptr().add(i) is valid when i + 32 <= len
    unsafe {
        let mut i = 0;
        let len = bytes.len();

        // Process 32 bytes at a time with AVX2
        while i + 32 <= len {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
            let has_high_bit = _mm256_movemask_epi8(chunk);

            if has_high_bit != 0 {
                return false; // Found non-ASCII
            }

            i += 32;
        }

        // Process remaining bytes
        for &byte in &bytes[i..] {
            if byte >= 0x80 {
                return false;
            }
        }

        true
    }
}

/// Unicode string processor with configurable strategies
pub struct UnicodeProcessor {
    normalize: bool,
    case_fold: bool,
}

impl UnicodeProcessor {
    /// Create a new Unicode processor with default settings
    pub fn new() -> Self {
        Self {
            normalize: false,
            case_fold: false,
        }
    }

    /// Enable partial canonical composition in [`process`](Self::process)
    /// (off by default).
    ///
    /// Limitation: this is a small subset of NFC. Only these pairs are
    /// composed, and only when the second character directly follows the
    /// first:
    /// - Hangul jamo: L (U+1100..=U+1112) + V (U+1161..=U+1175) into an LV
    ///   syllable, and an LV syllable + T (U+11A8..=U+11C2) into an LVT
    ///   syllable.
    /// - 54 Latin pairs: A, E, I, O, U and a, e, i, o, u with grave (U+0300)
    ///   or circumflex (U+0302); the same plus Y and y with acute (U+0301)
    ///   or diaeresis (U+0308); A, N, O and a, n, o with tilde (U+0303); A
    ///   and a with ring above (U+030A); C and c with cedilla (U+0327).
    ///
    /// Nothing is decomposed first, combining marks are not put in canonical
    /// order, a Latin base takes at most one mark, and singletons are left
    /// alone. So `"z\u{030C}"`, `"a\u{0328}"`, `"a\u{0323}\u{0302}"` and
    /// `"\u{212B}"` come out unchanged (NFC: `"\u{017E}"`, `"\u{0105}"`,
    /// `"\u{1EAD}"`, `"\u{00C5}"`), and the output can differ from NFC
    /// even where a pair is composed: `"a\u{0301}\u{0323}"` becomes
    /// `"\u{00E1}\u{0323}"`, while NFC gives `"\u{1EA1}\u{0301}"`.
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.normalize = enable;
        self
    }

    /// Enable lowercasing (with `str::to_lowercase`) before normalization,
    /// for case-insensitive processing. This is not full Unicode case
    /// folding: for example `"ß"` stays `"ß"` rather than becoming `"ss"`.
    pub fn with_case_folding(mut self, enable: bool) -> Self {
        self.case_fold = enable;
        self
    }

    /// Process a string with the configured options
    pub fn process(&mut self, input: &str) -> Result<String> {
        let mut result = if self.case_fold {
            input.to_lowercase()
        } else {
            input.to_string()
        };

        if self.normalize {
            result = self.normalize_basic(&result)?;
        }

        Ok(result)
    }

    /// Partial canonical composition, not NFC: composes each character with
    /// the one before it when `compose_pair` knows the pair (see the
    /// limitation on `with_normalization`)
    fn normalize_basic(&self, input: &str) -> Result<String> {
        let mut out: Vec<char> = Vec::with_capacity(input.len());

        for ch in input.chars() {
            if let Some(last) = out.last_mut()
                && let Some(composed) = Self::compose_pair(*last, ch)
            {
                *last = composed;
                continue;
            }
            out.push(ch);
        }

        Ok(out.into_iter().collect())
    }

    /// Canonical pair composition `(base, combining_or_jamo) -> Option<char>`
    fn compose_pair(base: char, mark: char) -> Option<char> {
        let b = base as u32;
        let m = mark as u32;

        // 1. Hangul L + V -> LV syllable (Unicode Standard §3.12)
        if (0x1100..=0x1112).contains(&b) && (0x1161..=0x1175).contains(&m) {
            let l_index = b - 0x1100;
            let v_index = m - 0x1161;
            let syllable = 0xAC00 + (l_index * 21 + v_index) * 28;
            return char::from_u32(syllable);
        }
        // 2. Hangul LV + T -> LVT syllable
        if (0xAC00..=0xD7A3).contains(&b)
            && (b - 0xAC00).is_multiple_of(28)
            && (0x11A8..=0x11C2).contains(&m)
        {
            let t_index = m - 0x11A7;
            return char::from_u32(b + t_index);
        }

        // 3. Common Latin base + Combining Diacritical Marks (U+0300..=U+0327)
        match (base, mark) {
            // Grave (U+0300)
            ('A', '\u{0300}') => Some('À'),
            ('E', '\u{0300}') => Some('È'),
            ('I', '\u{0300}') => Some('Ì'),
            ('O', '\u{0300}') => Some('Ò'),
            ('U', '\u{0300}') => Some('Ù'),
            ('a', '\u{0300}') => Some('à'),
            ('e', '\u{0300}') => Some('è'),
            ('i', '\u{0300}') => Some('ì'),
            ('o', '\u{0300}') => Some('ò'),
            ('u', '\u{0300}') => Some('ù'),
            // Acute (U+0301)
            ('A', '\u{0301}') => Some('Á'),
            ('E', '\u{0301}') => Some('É'),
            ('I', '\u{0301}') => Some('Í'),
            ('O', '\u{0301}') => Some('Ó'),
            ('U', '\u{0301}') => Some('Ú'),
            ('Y', '\u{0301}') => Some('Ý'),
            ('a', '\u{0301}') => Some('á'),
            ('e', '\u{0301}') => Some('é'),
            ('i', '\u{0301}') => Some('í'),
            ('o', '\u{0301}') => Some('ó'),
            ('u', '\u{0301}') => Some('ú'),
            ('y', '\u{0301}') => Some('ý'),
            // Circumflex (U+0302)
            ('A', '\u{0302}') => Some('Â'),
            ('E', '\u{0302}') => Some('Ê'),
            ('I', '\u{0302}') => Some('Î'),
            ('O', '\u{0302}') => Some('Ô'),
            ('U', '\u{0302}') => Some('Û'),
            ('a', '\u{0302}') => Some('â'),
            ('e', '\u{0302}') => Some('ê'),
            ('i', '\u{0302}') => Some('î'),
            ('o', '\u{0302}') => Some('ô'),
            ('u', '\u{0302}') => Some('û'),
            // Tilde (U+0303)
            ('A', '\u{0303}') => Some('Ã'),
            ('N', '\u{0303}') => Some('Ñ'),
            ('O', '\u{0303}') => Some('Õ'),
            ('a', '\u{0303}') => Some('ã'),
            ('n', '\u{0303}') => Some('ñ'),
            ('o', '\u{0303}') => Some('õ'),
            // Diaeresis (U+0308)
            ('A', '\u{0308}') => Some('Ä'),
            ('E', '\u{0308}') => Some('Ë'),
            ('I', '\u{0308}') => Some('Ï'),
            ('O', '\u{0308}') => Some('Ö'),
            ('U', '\u{0308}') => Some('Ü'),
            ('Y', '\u{0308}') => Some('Ÿ'),
            ('a', '\u{0308}') => Some('ä'),
            ('e', '\u{0308}') => Some('ë'),
            ('i', '\u{0308}') => Some('ï'),
            ('o', '\u{0308}') => Some('ö'),
            ('u', '\u{0308}') => Some('ü'),
            ('y', '\u{0308}') => Some('ÿ'),
            // Ring above (U+030A)
            ('A', '\u{030A}') => Some('Å'),
            ('a', '\u{030A}') => Some('å'),
            // Cedilla (U+0327)
            ('C', '\u{0327}') => Some('Ç'),
            ('c', '\u{0327}') => Some('ç'),
            _ => None,
        }
    }

    /// Analyze Unicode properties of a string
    pub fn analyze(&self, input: &str) -> UnicodeAnalysis {
        let mut analysis = UnicodeAnalysis::default();

        for ch in input.chars() {
            analysis.char_count += 1;

            if ch.is_ascii() {
                analysis.ascii_count += 1;
            }

            if ch.is_alphabetic() {
                analysis.alphabetic_count += 1;
            }

            if ch.is_numeric() {
                analysis.numeric_count += 1;
            }

            if ch.is_whitespace() {
                analysis.whitespace_count += 1;
            }

            if ch.is_control() {
                analysis.control_count += 1;
            }

            // Track Unicode blocks
            let codepoint = ch as u32;
            if codepoint <= 0x7F {
                analysis.basic_latin += 1;
            } else if codepoint <= 0xFF {
                analysis.latin_supplement += 1;
            } else if codepoint <= 0x024F {
                analysis.extended_latin += 1;
            } else {
                analysis.other_unicode += 1;
            }
        }

        analysis.byte_count = input.len();
        analysis
    }
}

impl Default for UnicodeProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Analysis results for Unicode strings
#[derive(Debug, Clone, Default)]
pub struct UnicodeAnalysis {
    pub byte_count: usize,
    pub char_count: usize,
    pub ascii_count: usize,
    pub alphabetic_count: usize,
    pub numeric_count: usize,
    pub whitespace_count: usize,
    pub control_count: usize,

    // Unicode block counts
    pub basic_latin: usize,
    pub latin_supplement: usize,
    pub extended_latin: usize,
    pub other_unicode: usize,
}

impl UnicodeAnalysis {
    /// Check if the string is pure ASCII
    pub fn is_ascii(&self) -> bool {
        self.ascii_count == self.char_count
    }

    /// Calculate the average bytes per character
    pub fn avg_bytes_per_char(&self) -> f64 {
        if self.char_count == 0 {
            0.0
        } else {
            self.byte_count as f64 / self.char_count as f64
        }
    }

    /// Get the Unicode complexity score (0.0 = ASCII, 1.0 = complex Unicode)
    pub fn complexity_score(&self) -> f64 {
        if self.char_count == 0 {
            return 0.0;
        }

        let non_ascii_ratio = (self.char_count - self.ascii_count) as f64 / self.char_count as f64;
        let avg_bytes = self.avg_bytes_per_char();

        // Combine non-ASCII ratio with average byte length
        (non_ascii_ratio * 0.7) + ((avg_bytes - 1.0) / 3.0 * 0.3).min(0.3)
    }
}

/// Bidirectional UTF-8 to UTF-32 iterator
///
/// Efficiently converts between UTF-8 bytes and Unicode codepoints
/// with support for both forward and backward iteration.
pub struct Utf8ToUtf32Iterator<'a> {
    bytes: &'a [u8],
    position: usize,
    current_char: Option<char>,
    current_span: Option<(usize, usize)>,
}

impl<'a> Utf8ToUtf32Iterator<'a> {
    /// Create a new iterator from UTF-8 bytes
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        // Validate UTF-8 first
        str::from_utf8(bytes)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid UTF-8: {}", e)))?;

        Ok(Self {
            bytes,
            position: 0,
            current_char: None,
            current_span: None,
        })
    }

    /// Get the current character
    pub fn current(&self) -> Option<char> {
        self.current_char
    }

    /// Move to the next character
    pub fn next_char(&mut self) -> Option<char> {
        let start = self.current_span.map_or(self.position, |(_, end)| end);
        if start >= self.bytes.len() {
            self.position = self.bytes.len();
            self.current_char = None;
            self.current_span = None;
            return None;
        }

        let byte = self.bytes[start];
        let char_len = utf8_byte_count(byte);

        if char_len == 0 || start + char_len > self.bytes.len() {
            self.current_char = None;
            self.current_span = None;
            return None;
        }

        let char_bytes = &self.bytes[start..start + char_len];
        match str::from_utf8(char_bytes) {
            Ok(s) => {
                self.current_char = s.chars().next();
                self.current_span = Some((start, start + char_len));
                self.position = start + char_len;
                self.current_char
            }
            Err(_) => {
                self.current_char = None;
                self.current_span = None;
                None
            }
        }
    }

    /// Move to the previous character (if supported)
    pub fn prev_char(&mut self) -> Option<char> {
        let end = self.current_span.map_or(self.position, |(start, _)| start);
        if end == 0 {
            self.position = 0;
            self.current_char = None;
            self.current_span = None;
            return None;
        }

        // Walk backward to find the start of the previous character
        let mut pos = end - 1;

        // Skip continuation bytes
        while pos > 0 && (self.bytes[pos] & 0xC0) == 0x80 {
            pos -= 1;
        }

        // Now at the start of a character
        let byte = self.bytes[pos];
        let char_len = utf8_byte_count(byte);

        if char_len == 0 || pos + char_len > self.bytes.len() {
            self.current_char = None;
            self.current_span = None;
            return None;
        }

        let char_bytes = &self.bytes[pos..pos + char_len];
        match str::from_utf8(char_bytes) {
            Ok(s) => {
                self.current_char = s.chars().next();
                self.current_span = Some((pos, pos + char_len));
                self.position = pos;
                self.current_char
            }
            Err(_) => {
                self.current_char = None;
                self.current_span = None;
                None
            }
        }
    }

    /// Get the current byte position
    pub fn byte_position(&self) -> usize {
        self.position
    }

    /// Reset to the beginning
    pub fn reset(&mut self) {
        self.position = 0;
        self.current_char = None;
        self.current_span = None;
    }
}

/// Utility functions for Unicode operations
pub mod utils {
    use super::*;

    /// Convert UTF-8 string to lowercase with proper Unicode handling
    pub fn to_lowercase_unicode(input: &str) -> String {
        input.to_lowercase()
    }

    /// Convert UTF-8 string to uppercase with proper Unicode handling
    pub fn to_uppercase_unicode(input: &str) -> String {
        input.to_uppercase()
    }

    /// Approximate terminal display width of `input`: the sum, over its
    /// characters, of
    /// - 0 for control characters (general category Cc) and for characters
    ///   in the combining-mark blocks U+0300..=U+036F, U+1AB0..=U+1AFF,
    ///   U+1DC0..=U+1DFF, U+20D0..=U+20FF and U+FE20..=U+FE2F;
    /// - 2 for characters in U+1100..=U+115F, U+2E80..=U+2FDF,
    ///   U+3000..=U+4DBF, U+4E00..=U+A4CF, U+AC00..=U+D7AF,
    ///   U+F900..=U+FAFF, U+FE10..=U+FE1F, U+FE30..=U+FE6F,
    ///   U+FF00..=U+FFEF, U+20000..=U+2A6DF, U+2A700..=U+2B81F and
    ///   U+2F800..=U+2FA1F;
    /// - 1 for everything else.
    ///
    /// Limitation: these are fixed ranges, not the Unicode East Asian Width
    /// and general category data, so some widths are wrong, for example:
    /// - nonspacing marks and format characters outside those blocks count
    ///   as 1 (U+05B0, U+0E31, U+FE0F, U+200B, U+200D);
    /// - wide characters outside the ranges count as 1 (the emoji U+1F600,
    ///   and the CJK ideographs from Extension E on, such as U+2B820 and
    ///   U+30000);
    /// - narrow characters inside them count as 2 (the halfwidth forms in
    ///   U+FF61..=U+FFDC and U+FFE8..=U+FFEE, and U+303F), and so do the
    ///   combining marks U+302A..=U+302D, U+3099 and U+309A.
    pub fn display_width(input: &str) -> usize {
        input
            .chars()
            .map(|ch| {
                if ch.is_control() || is_combining_mark(ch) {
                    0
                } else if ch as u32 >= 0x1100 && is_wide_char(ch) {
                    2 // East Asian Wide / Fullwidth characters
                } else {
                    1 // Standard single-column width
                }
            })
            .sum()
    }

    /// Whether `ch` is in one of the five combining-mark blocks listed on
    /// `display_width`. Nonspacing marks in other blocks, such as U+05B0 or
    /// U+0E31, are not included.
    fn is_combining_mark(ch: char) -> bool {
        let cp = ch as u32;
        (0x0300..=0x036F).contains(&cp)
            || (0x1AB0..=0x1AFF).contains(&cp)
            || (0x1DC0..=0x1DFF).contains(&cp)
            || (0x20D0..=0x20FF).contains(&cp)
            || (0xFE20..=0xFE2F).contains(&cp)
    }

    /// Whether `ch` is in the block list below, which approximates East
    /// Asian Width W and F (UAX #11); see the limitation on `display_width`
    fn is_wide_char(ch: char) -> bool {
        let codepoint = ch as u32;

        // East Asian Wide and Fullwidth ranges
        (0x1100..=0x115F).contains(&codepoint) ||  // Hangul Jamo
        (0x2E80..=0x2EFF).contains(&codepoint) ||  // CJK Radicals Supplement
        (0x2F00..=0x2FDF).contains(&codepoint) ||  // Kangxi Radicals
        (0x3000..=0x303F).contains(&codepoint) ||  // CJK Symbols and Punctuation
        (0x3040..=0x309F).contains(&codepoint) ||  // Hiragana
        (0x30A0..=0x30FF).contains(&codepoint) ||  // Katakana
        (0x3100..=0x312F).contains(&codepoint) ||  // Bopomofo
        (0x3130..=0x318F).contains(&codepoint) ||  // Hangul Compatibility Jamo
        (0x3190..=0x319F).contains(&codepoint) ||  // Kanbun
        (0x31A0..=0x31BF).contains(&codepoint) ||  // Bopomofo Extended
        (0x31C0..=0x31EF).contains(&codepoint) ||  // CJK Strokes
        (0x31F0..=0x31FF).contains(&codepoint) ||  // Katakana Phonetic Extensions
        (0x3200..=0x32FF).contains(&codepoint) ||  // Enclosed CJK Letters and Months
        (0x3300..=0x33FF).contains(&codepoint) ||  // CJK Compatibility
        (0x3400..=0x4DBF).contains(&codepoint) ||  // CJK Extension A
        (0x4E00..=0x9FFF).contains(&codepoint) ||  // CJK Unified Ideographs
        (0xA000..=0xA48F).contains(&codepoint) ||  // Yi Syllables
        (0xA490..=0xA4CF).contains(&codepoint) ||  // Yi Radicals
        (0xAC00..=0xD7AF).contains(&codepoint) ||  // Hangul Syllables
        (0xF900..=0xFAFF).contains(&codepoint) ||  // CJK Compatibility Ideographs
        (0xFE10..=0xFE1F).contains(&codepoint) ||  // Vertical Forms
        (0xFE30..=0xFE4F).contains(&codepoint) ||  // CJK Compatibility Forms
        (0xFE50..=0xFE6F).contains(&codepoint) ||  // Small Form Variants
        (0xFF00..=0xFFEF).contains(&codepoint) ||  // Halfwidth and Fullwidth Forms
        (0x20000..=0x2A6DF).contains(&codepoint) || // CJK Extension B
        (0x2A700..=0x2B73F).contains(&codepoint) || // CJK Extension C
        (0x2B740..=0x2B81F).contains(&codepoint) || // CJK Extension D
        (0x2F800..=0x2FA1F).contains(&codepoint) // CJK Compatibility Ideographs Supplement
    }

    /// Extract all Unicode codepoints from a string
    pub fn extract_codepoints(input: &str) -> Vec<u32> {
        input.chars().map(|ch| ch as u32).collect()
    }

    /// Check if string contains only printable Unicode characters
    pub fn is_printable(input: &str) -> bool {
        input
            .chars()
            .all(|ch| !ch.is_control() || ch == '\t' || ch == '\n' || ch == '\r')
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_byte_count() {
        assert_eq!(utf8_byte_count(0x41), 1); // 'A'
        assert_eq!(utf8_byte_count(0xC2), 2); // Start of 2-byte sequence
        assert_eq!(utf8_byte_count(0xE2), 3); // Start of 3-byte sequence  
        assert_eq!(utf8_byte_count(0xF0), 4); // Start of 4-byte sequence
        assert_eq!(utf8_byte_count(0x80), 0); // Continuation byte
    }

    #[test]
    fn test_unicode_analysis() {
        let processor = UnicodeProcessor::new();

        // ASCII string
        let analysis = processor.analyze("Hello World!");
        assert!(analysis.is_ascii());
        assert_eq!(analysis.char_count, 12);
        assert_eq!(analysis.byte_count, 12);
        assert_eq!(analysis.ascii_count, 12);
        assert_eq!(analysis.avg_bytes_per_char(), 1.0);

        // Mixed Unicode string
        let analysis = processor.analyze("Hello 世界!");
        assert!(!analysis.is_ascii());
        assert_eq!(analysis.char_count, 9);
        assert!(analysis.byte_count > 9); // Multi-byte characters
        assert_eq!(analysis.ascii_count, 7);
        assert!(analysis.avg_bytes_per_char() > 1.0);
    }

    #[test]
    fn test_utf8_iterator() {
        let text = "Hello 世界!";
        let mut iter = Utf8ToUtf32Iterator::new(text.as_bytes()).unwrap();

        let chars: Vec<char> = text.chars().collect();
        let mut collected = Vec::new();

        while let Some(ch) = iter.next_char() {
            collected.push(ch);
        }

        assert_eq!(chars, collected);
    }

    #[test]
    fn test_backward_iteration() {
        let text = "Héllo!";
        let mut iter = Utf8ToUtf32Iterator::new(text.as_bytes()).unwrap();

        // Move to end
        while iter.next_char().is_some() {}

        // Iterate backward
        let mut backward_chars = Vec::new();
        while let Some(ch) = iter.prev_char() {
            backward_chars.push(ch);
        }

        let forward_chars: Vec<char> = text.chars().collect();
        backward_chars.reverse();

        assert_eq!(forward_chars, backward_chars);
    }

    #[test]
    fn test_unicode_validation() {
        // Valid UTF-8
        let valid = "Hello 世界!";
        assert!(validate_utf8_and_count_chars(valid.as_bytes()).is_ok());

        // Invalid UTF-8
        let invalid = &[0xFF, 0xFE, 0xFD];
        assert!(validate_utf8_and_count_chars(invalid).is_err());
    }

    #[test]
    fn test_display_width() {
        assert_eq!(utils::display_width("Hello"), 5);
        assert_eq!(utils::display_width("世界"), 4); // Wide characters
        assert_eq!(utils::display_width("Hello世界"), 9); // Mixed
    }

    #[test]
    fn test_case_conversion() {
        assert_eq!(utils::to_lowercase_unicode("HELLO"), "hello");
        assert_eq!(utils::to_uppercase_unicode("hello"), "HELLO");

        // Unicode case conversion
        assert_eq!(utils::to_lowercase_unicode("WORLD"), "world");
        assert_eq!(utils::to_uppercase_unicode("world"), "WORLD");
    }

    #[test]
    fn test_processor_configuration() {
        let mut processor = UnicodeProcessor::new()
            .with_case_folding(true)
            .with_normalization(true);

        let result = processor.process("HELLO World").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_codepoint_extraction() {
        let text = "A世";
        let codepoints = utils::extract_codepoints(text);
        assert_eq!(codepoints, vec![0x41, 0x4E16]); // 'A' and '世'
    }

    #[test]
    fn test_c8_6_unicode_invalid_lead_bytes_bidirectional_iter_nfc_and_width() {
        // 1. Overlong 0xC0..=0xC1 and out-of-range 0xF5..=0xFF must return 0.
        for b in [0xC0u8, 0xC1, 0xF5, 0xF6, 0xF7, 0xF8, 0xFF] {
            assert_eq!(utf8_byte_count(b), 0, "invalid lead byte {b:#04x}");
        }

        // 2. Bidirectional Utf8ToUtf32Iterator alternating next_char and prev_char.
        let mut iter = Utf8ToUtf32Iterator::new("aé世".as_bytes()).unwrap();
        assert_eq!(iter.next_char(), Some('a'));
        assert_eq!(iter.next_char(), Some('é'));
        assert_eq!(iter.prev_char(), Some('a'));
        assert_eq!(iter.next_char(), Some('é'));
        assert_eq!(iter.next_char(), Some('世'));
        assert_eq!(iter.prev_char(), Some('é'));

        // 3. UnicodeProcessor NFC composition and extended_latin block range (<= 0x024F).
        let mut proc = UnicodeProcessor::new().with_normalization(true);
        assert_eq!(proc.process("e\u{0301}cole").unwrap(), "école");
        assert_eq!(proc.process("\u{1100}\u{1161}\u{11A8}").unwrap(), "각");

        let analysis = proc.analyze("ĀΩЖ");
        // 'Ā' (U+0100) is Extended Latin; 'Ω' (U+03A9, Greek) and 'Ж' (U+0416, Cyrillic) are other_unicode.
        assert_eq!(analysis.extended_latin, 1);
        assert_eq!(analysis.other_unicode, 2);

        // 4. display_width includes U+1100 as width 2 and combining marks as width 0.
        assert_eq!(utils::display_width("\u{1100}"), 2);
        assert_eq!(utils::display_width("e\u{0301}"), 1);
    }

    #[test]
    fn test_s20_r7_documented_limitations_hold() {
        // Every example given in the limitations on `with_normalization`,
        // `with_case_folding` and `utils::display_width`. If a table is
        // extended, change those docs and this test together.
        let mut p = UnicodeProcessor::new().with_normalization(true);
        // Composed: a Latin pair from the table and Hangul L + V + T.
        assert_eq!(p.process("e\u{0301}").unwrap(), "\u{00E9}");
        assert_eq!(p.process("\u{1100}\u{1161}\u{11A8}").unwrap(), "\u{AC01}");
        // Left unchanged, though NFC composes them.
        for s in ["z\u{030C}", "a\u{0328}", "a\u{0323}\u{0302}", "\u{212B}"] {
            assert_eq!(p.process(s).unwrap(), s, "{s:?}");
        }
        // Composed differently from NFC (U+1EA1 U+0301), as marks are not
        // reordered.
        assert_eq!(p.process("a\u{0301}\u{0323}").unwrap(), "\u{00E1}\u{0323}");

        let mut lower = UnicodeProcessor::new().with_case_folding(true);
        assert_eq!(lower.process("\u{00DF}").unwrap(), "\u{00DF}");

        let width = |c: char| utils::display_width(c.encode_utf8(&mut [0; 4]));
        for c in ['\u{05B0}', '\u{0E31}', '\u{FE0F}', '\u{200B}', '\u{200D}'] {
            assert_eq!(width(c), 1, "U+{:04X}", c as u32);
        }
        for c in ['\u{1F600}', '\u{2B820}', '\u{30000}'] {
            assert_eq!(width(c), 1, "U+{:04X}", c as u32);
        }
        for c in [
            '\u{FF61}', '\u{FFDC}', '\u{FFE8}', '\u{FFEE}', '\u{303F}', '\u{302A}', '\u{302D}',
            '\u{3099}', '\u{309A}',
        ] {
            assert_eq!(width(c), 2, "U+{:04X}", c as u32);
        }
    }
}
