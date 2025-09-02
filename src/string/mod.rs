//! Zero-copy string operations with SIMD optimization
//!
//! This module provides high-performance string types optimized for minimal copying
//! and maximum throughput using SIMD instructions where available.
//!
//! ## Features
//!
//! - **FastStr**: Zero-copy string operations with SIMD hashing
//! - **BMI2 String Operations**: Hardware-accelerated string processing with BMI2 instructions
//! - **Lexicographic Iterators**: Efficient iteration over sorted string collections
//! - **Unicode Processing**: Full Unicode support with proper handling
//! - **Line Processing**: Utilities for processing large text files line by line

mod fast_str;
mod lexicographic_iterator;
mod unicode;
mod line_processor;
mod bmi2_string_ops;

pub use fast_str::FastStr;

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

// BMI2 string operations exports
pub use bmi2_string_ops::{
    Bmi2StringProcessor, CharClass, CharFilter, CharRun, CompressionAnalysis,
    StringDictionary, DictionaryEntry, DictionaryMatch,
    get_global_bmi2_processor, validate_utf8_bmi2, count_utf8_chars_bmi2,
    search_string_bmi2, wildcard_match_bmi2, to_lowercase_ascii_bmi2,
    to_uppercase_ascii_bmi2, hash_string_bmi2, detect_runs_bmi2,
};

/// Utility modules for common string operations
pub mod utils {
    pub use super::lexicographic_iterator::utils as lex_utils;
    pub use super::unicode::utils as unicode_utils;
    pub use super::line_processor::utils as line_utils;
}

/// BMI2 string processing utilities for advanced string operations
pub mod bmi2 {
    pub use super::bmi2_string_ops::*;
}
