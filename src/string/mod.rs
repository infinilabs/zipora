//! Zero-copy string operations with SIMD optimization
//!
//! This module provides high-performance string types optimized for minimal copying
//! and maximum throughput using SIMD instructions where available.
//!
//! ## Features
//!
//! - **FastStr**: Zero-copy string operations with SIMD hashing
//! - **Lexicographic Iterators**: Efficient iteration over sorted string collections
//! - **Unicode Processing**: Full Unicode support with proper handling
//! - **Line Processing**: Utilities for processing large text files line by line

mod fast_str;
mod lexicographic_iterator;
mod unicode;
mod line_processor;

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

/// Utility modules for common string operations
pub mod utils {
    pub use super::lexicographic_iterator::utils as lex_utils;
    pub use super::unicode::utils as unicode_utils;
    pub use super::line_processor::utils as line_utils;
}
