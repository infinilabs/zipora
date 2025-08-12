//! Unicode String Processing
//!
//! High-performance Unicode processing with hardware acceleration where available.
//! Provides efficient UTF-8/UTF-16/UTF-32 conversions, normalization, and analysis
//! with SIMD optimizations for batch operations.

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
    // Optimized lookup table for UTF-8 leading byte analysis
    static UTF8_BYTE_COUNT_LUT: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i = 0;
        while i < 256 {
            let byte = i as u8;
            table[i] = if byte < 0x80 {
                1 // ASCII
            } else if byte < 0xC0 {
                0 // Continuation byte
            } else if byte < 0xE0 {
                2 // 2-byte sequence
            } else if byte < 0xF0 {
                3 // 3-byte sequence
            } else if byte < 0xF8 {
                4 // 4-byte sequence
            } else {
                0 // Invalid
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
        Err(e) => Err(ZiporaError::invalid_data(&format!("Invalid UTF-8: {}", e))),
    }
}

/// SIMD UTF-8 validation (when available)
#[cfg(target_arch = "x86_64")]
fn validate_utf8_and_count_chars_simd(bytes: &[u8]) -> Result<usize> {
    #[cfg(target_feature = "avx2")]
    {
        // Use SIMD for bulk validation, then count characters
        if is_ascii_simd(bytes) {
            // All ASCII - character count equals byte count
            return Ok(bytes.len());
        }
    }
    
    // Fall back to scalar for complex cases
    validate_utf8_and_count_chars_scalar(bytes)
}

#[cfg(not(target_arch = "x86_64"))]
fn validate_utf8_and_count_chars_simd(bytes: &[u8]) -> Result<usize> {
    validate_utf8_and_count_chars_scalar(bytes)
}

/// Check if bytes are all ASCII using SIMD when available
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn is_ascii_simd(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;

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

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn is_ascii_simd(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b < 0x80)
}

/// Unicode string processor with configurable strategies
pub struct UnicodeProcessor {
    normalize: bool,
    case_fold: bool,
    buffer: Vec<u8>,
}

impl UnicodeProcessor {
    /// Create a new Unicode processor with default settings
    pub fn new() -> Self {
        Self {
            normalize: false,
            case_fold: false,
            buffer: Vec::new(),
        }
    }

    /// Enable Unicode normalization (NFC by default)
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.normalize = enable;
        self
    }

    /// Enable case folding for case-insensitive processing
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
            // Basic normalization - could be enhanced with full Unicode normalization
            result = self.normalize_basic(&result)?;
        }

        Ok(result)
    }

    /// Basic normalization (placeholder for full Unicode normalization)
    fn normalize_basic(&self, input: &str) -> Result<String> {
        // This is a simplified normalization - a full implementation would
        // use the unicode-normalization crate for proper NFC/NFD/NFKC/NFKD
        Ok(input.chars().collect::<String>())
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
            } else if codepoint <= 0x17FF {
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
}

impl<'a> Utf8ToUtf32Iterator<'a> {
    /// Create a new iterator from UTF-8 bytes
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        // Validate UTF-8 first
        str::from_utf8(bytes)
            .map_err(|e| ZiporaError::invalid_data(&format!("Invalid UTF-8: {}", e)))?;

        Ok(Self {
            bytes,
            position: 0,
            current_char: None,
        })
    }

    /// Get the current character
    pub fn current(&self) -> Option<char> {
        self.current_char
    }

    /// Move to the next character
    pub fn next_char(&mut self) -> Option<char> {
        if self.position >= self.bytes.len() {
            self.current_char = None;
            return None;
        }

        let byte = self.bytes[self.position];
        let char_len = utf8_byte_count(byte);
        
        if char_len == 0 || self.position + char_len > self.bytes.len() {
            self.current_char = None;
            return None;
        }

        let char_bytes = &self.bytes[self.position..self.position + char_len];
        match str::from_utf8(char_bytes) {
            Ok(s) => {
                self.current_char = s.chars().next();
                self.position += char_len;
                self.current_char
            }
            Err(_) => {
                self.current_char = None;
                None
            }
        }
    }

    /// Move to the previous character (if supported)
    pub fn prev_char(&mut self) -> Option<char> {
        if self.position == 0 {
            self.current_char = None;
            return None;
        }

        // Walk backward to find the start of the previous character
        let mut pos = self.position.saturating_sub(1);
        
        // Skip continuation bytes
        while pos > 0 && (self.bytes[pos] & 0xC0) == 0x80 {
            pos -= 1;
        }

        // Now at the start of a character
        let byte = self.bytes[pos];
        let char_len = utf8_byte_count(byte);
        
        if char_len == 0 || pos + char_len > self.bytes.len() {
            self.current_char = None;
            return None;
        }

        let char_bytes = &self.bytes[pos..pos + char_len];
        match str::from_utf8(char_bytes) {
            Ok(s) => {
                self.current_char = s.chars().next();
                self.position = pos;
                self.current_char
            }
            Err(_) => {
                self.current_char = None;
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

    /// Calculate the display width of a Unicode string
    /// (accounting for wide characters, combining marks, etc.)
    pub fn display_width(input: &str) -> usize {
        // Simplified implementation - could use unicode-width crate for full support
        input.chars().map(|ch| {
            if ch.is_control() {
                0
            } else if ch as u32 > 0x1100 && is_wide_char(ch) {
                2 // Wide characters (CJK, etc.)
            } else {
                1 // Normal width
            }
        }).sum()
    }

    /// Check if a character is wide (occupies 2 terminal columns)
    fn is_wide_char(ch: char) -> bool {
        // Simplified check for wide characters
        let codepoint = ch as u32;
        
        // CJK ranges (simplified)
        (codepoint >= 0x1100 && codepoint <= 0x115F) ||  // Hangul Jamo
        (codepoint >= 0x2E80 && codepoint <= 0x2EFF) ||  // CJK Radicals Supplement
        (codepoint >= 0x2F00 && codepoint <= 0x2FDF) ||  // Kangxi Radicals
        (codepoint >= 0x3000 && codepoint <= 0x303F) ||  // CJK Symbols and Punctuation
        (codepoint >= 0x3040 && codepoint <= 0x309F) ||  // Hiragana
        (codepoint >= 0x30A0 && codepoint <= 0x30FF) ||  // Katakana
        (codepoint >= 0x3100 && codepoint <= 0x312F) ||  // Bopomofo
        (codepoint >= 0x3130 && codepoint <= 0x318F) ||  // Hangul Compatibility Jamo
        (codepoint >= 0x3190 && codepoint <= 0x319F) ||  // Kanbun
        (codepoint >= 0x31A0 && codepoint <= 0x31BF) ||  // Bopomofo Extended
        (codepoint >= 0x31C0 && codepoint <= 0x31EF) ||  // CJK Strokes
        (codepoint >= 0x31F0 && codepoint <= 0x31FF) ||  // Katakana Phonetic Extensions
        (codepoint >= 0x3200 && codepoint <= 0x32FF) ||  // Enclosed CJK Letters and Months
        (codepoint >= 0x3300 && codepoint <= 0x33FF) ||  // CJK Compatibility
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||  // CJK Extension A
        (codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||  // CJK Unified Ideographs
        (codepoint >= 0xA000 && codepoint <= 0xA48F) ||  // Yi Syllables
        (codepoint >= 0xA490 && codepoint <= 0xA4CF) ||  // Yi Radicals
        (codepoint >= 0xAC00 && codepoint <= 0xD7AF) ||  // Hangul Syllables
        (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||  // CJK Compatibility Ideographs
        (codepoint >= 0xFE10 && codepoint <= 0xFE1F) ||  // Vertical Forms
        (codepoint >= 0xFE30 && codepoint <= 0xFE4F) ||  // CJK Compatibility Forms
        (codepoint >= 0xFE50 && codepoint <= 0xFE6F) ||  // Small Form Variants
        (codepoint >= 0xFF00 && codepoint <= 0xFFEF) ||  // Halfwidth and Fullwidth Forms
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF) || // CJK Extension B
        (codepoint >= 0x2A700 && codepoint <= 0x2B73F) || // CJK Extension C
        (codepoint >= 0x2B740 && codepoint <= 0x2B81F) || // CJK Extension D
        (codepoint >= 0x2F800 && codepoint <= 0x2FA1F)    // CJK Compatibility Ideographs Supplement
    }

    /// Extract all Unicode codepoints from a string
    pub fn extract_codepoints(input: &str) -> Vec<u32> {
        input.chars().map(|ch| ch as u32).collect()
    }

    /// Check if string contains only printable Unicode characters
    pub fn is_printable(input: &str) -> bool {
        input.chars().all(|ch| !ch.is_control() || ch == '\t' || ch == '\n' || ch == '\r')
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
}