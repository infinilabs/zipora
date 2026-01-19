//! Zero-copy string operations with SIMD optimization
//!
//! This module provides high-performance string types optimized for minimal copying
//! and maximum throughput using SIMD instructions where available.
//!
//! ## Features
//!
//! - **FastStr**: Zero-copy string operations with SIMD hashing
//! - **SIMD Search**: SSE4.2 PCMPESTRI-based string search operations with hybrid strategies
//! - **BMI2 String Operations**: Hardware-accelerated string processing with BMI2 instructions
//! - **Lexicographic Iterators**: Efficient iteration over sorted string collections
//! - **Unicode Processing**: Full Unicode support with proper handling
//! - **Line Processing**: Utilities for processing large text files line by line
//! - **Numeric Comparison**: `decimal_strcmp` and `realnum_strcmp` for numeric string comparison
//! - **String Joining**: Efficient `join`, `join_str`, `JoinBuilder` utilities
//! - **Word Boundary**: `is_word_boundary`, `words`, `word_count` for text tokenization
//! - **Hex Encoding**: `hex_decode`, `hex_encode` for hex string conversion

mod fast_str;
mod lexicographic_iterator;
mod unicode;
mod line_processor;
mod bmi2_string_ops;
mod simd_search;
mod numeric_compare;
mod join;
mod word_boundary;
mod hex;

pub use fast_str::FastStr;

// Numeric string comparison exports
pub use numeric_compare::{
    decimal_strcmp, decimal_strcmp_with_sign,
    realnum_strcmp, realnum_strcmp_with_sign,
};

// Lexicographic iterator exports
pub use lexicographic_iterator::{
    LexicographicIterator, SortedVecLexIterator, StreamingLexIterator,
    LexIteratorBuilder,
};

// Unicode processing exports
pub use unicode::{
    UnicodeProcessor, UnicodeAnalysis, Utf8ToUtf32Iterator,
    utf8_byte_count, validate_utf8_and_count_chars,
};

// Line processing exports
pub use line_processor::{
    LineProcessor, LineProcessorConfig, LineProcessorStats,
    LineSplitter,
};

// SIMD search operations exports
pub use simd_search::{
    SimdStringSearch, SearchTier, MultiSearchResult,
    get_global_simd_search, sse42_strchr, sse42_strstr, 
    sse42_multi_search, sse42_strcmp,
};

// BMI2 string operations exports
pub use bmi2_string_ops::{
    Bmi2StringProcessor, CharClass, CharFilter, CharRun, CompressionAnalysis,
    StringDictionary, DictionaryEntry, DictionaryMatch,
    get_global_bmi2_processor, validate_utf8_bmi2, count_utf8_chars_bmi2,
    search_string_bmi2, wildcard_match_bmi2, to_lowercase_ascii_bmi2,
    to_uppercase_ascii_bmi2, hash_string_bmi2, detect_runs_bmi2,
};

// String join utilities exports
pub use join::{
    join, join_str, join_fast_str, join_iter, join_bytes_iter,
    JoinBuilder,
};

// Word boundary detection exports
pub use word_boundary::{
    is_word_boundary, is_word_char, is_whitespace, is_punctuation,
    find_word_boundaries, words, word_count, word_at_position,
    WordIterator,
};

// Hex encoding/decoding exports
pub use hex::{
    hex_decode, hex_decode_bytes, hex_decode_to_slice,
    hex_encode, hex_encode_upper, hex_encode_to_bytes, hex_encode_to_slice,
    hex_char_to_nibble, nibble_to_hex_lower, nibble_to_hex_upper,
    is_valid_hex, parse_hex_byte,
};

/// Utility modules for common string operations
pub mod utils {
    pub use super::lexicographic_iterator::utils as lex_utils;
    pub use super::unicode::utils as unicode_utils;
    pub use super::line_processor::utils as line_utils;
}

/// SIMD string search utilities for hardware-accelerated string operations
pub mod simd {
    pub use super::simd_search::*;
}

/// BMI2 string processing utilities for advanced string operations
pub mod bmi2 {
    pub use super::bmi2_string_ops::*;
}
