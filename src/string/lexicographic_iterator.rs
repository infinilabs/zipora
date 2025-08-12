//! Lexicographic String Iterators
//!
//! Efficient iteration over sorted string collections with support for various backends
//! including tries, sorted vectors, and streaming sources. Designed for high-performance
//! string processing with zero-copy operations where possible.

use crate::error::ZiporaError;
use std::cmp::Ordering;

/// Core trait for lexicographic iteration over string collections
///
/// Provides efficient iteration patterns inspired by research implementations,
/// with support for bidirectional movement, seeking, and binary search operations.
pub trait LexicographicIterator {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Get the current string without advancing the iterator
    fn current(&self) -> Option<&str>;

    /// Move to the next string in lexicographic order
    /// Returns true if successful, false if at end
    fn next(&mut self) -> std::result::Result<bool, Self::Error>;

    /// Move to the previous string in lexicographic order  
    /// Returns true if successful, false if at beginning
    fn prev(&mut self) -> std::result::Result<bool, Self::Error>;

    /// Move to the first string in the collection
    fn seek_start(&mut self) -> std::result::Result<bool, Self::Error>;

    /// Move to the last string in the collection
    fn seek_end(&mut self) -> std::result::Result<bool, Self::Error>;

    /// Binary search for the first string >= target
    /// Returns true if exact match found, false if positioned at next larger string
    fn seek_lower_bound(&mut self, target: &str) -> std::result::Result<bool, Self::Error>;

    /// Binary search for the first string > target
    fn seek_upper_bound(&mut self, target: &str) -> std::result::Result<bool, Self::Error> {
        let exact_match = self.seek_lower_bound(target)?;
        if exact_match {
            // Move to next string after exact match
            self.next()?;
        }
        Ok(false) // Never an exact match by definition
    }

    /// Get an estimate of the total number of strings (if available)
    fn size_hint(&self) -> Option<usize> {
        None
    }

    /// Check if the iterator is at the beginning
    fn is_at_start(&self) -> bool {
        false // Default implementation - subclasses should override
    }

    /// Check if the iterator is at the end
    fn is_at_end(&self) -> bool {
        self.current().is_none()
    }
}

/// High-performance lexicographic iterator for sorted string vectors
///
/// Optimized for collections that fit in memory with O(log n) seeking
/// and O(1) sequential access patterns.
pub struct SortedVecLexIterator<'a> {
    strings: &'a [String],
    position: Option<usize>,
}

impl<'a> SortedVecLexIterator<'a> {
    /// Create a new iterator from a sorted string slice
    ///
    /// # Panics
    /// Panics in debug mode if the input is not sorted
    pub fn new(strings: &'a [String]) -> Self {
        debug_assert!(
            strings.windows(2).all(|w| w[0] <= w[1]),
            "Input strings must be sorted lexicographically"
        );

        let position = if strings.is_empty() { None } else { Some(0) };

        Self { strings, position }
    }

    /// Binary search implementation optimized for string comparison
    fn binary_search_by<F>(&self, mut compare: F) -> std::result::Result<usize, usize>
    where
        F: FnMut(&str) -> Ordering,
    {
        let mut left = 0;
        let mut right = self.strings.len();

        while left < right {
            let mid = left + (right - left) / 2;
            match compare(&self.strings[mid]) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }
}

impl<'a> LexicographicIterator for SortedVecLexIterator<'a> {
    type Error = ZiporaError;

    fn current(&self) -> Option<&str> {
        self.position.map(|pos| self.strings[pos].as_str())
    }

    fn next(&mut self) -> std::result::Result<bool, Self::Error> {
        match self.position {
            Some(pos) if pos + 1 < self.strings.len() => {
                self.position = Some(pos + 1);
                Ok(true)
            }
            _ => {
                self.position = None;
                Ok(false)
            }
        }
    }

    fn prev(&mut self) -> std::result::Result<bool, Self::Error> {
        match self.position {
            Some(pos) if pos > 0 => {
                self.position = Some(pos - 1);
                Ok(true)
            }
            _ => {
                self.position = if self.strings.is_empty() { None } else { Some(0) };
                Ok(false)
            }
        }
    }

    fn seek_start(&mut self) -> std::result::Result<bool, Self::Error> {
        self.position = if self.strings.is_empty() { None } else { Some(0) };
        Ok(self.position.is_some())
    }

    fn seek_end(&mut self) -> std::result::Result<bool, Self::Error> {
        self.position = if self.strings.is_empty() {
            None
        } else {
            Some(self.strings.len() - 1)
        };
        Ok(self.position.is_some())
    }

    fn seek_lower_bound(&mut self, target: &str) -> std::result::Result<bool, Self::Error> {
        match self.binary_search_by(|s| s.cmp(target)) {
            Ok(pos) => {
                self.position = Some(pos);
                Ok(true) // Exact match
            }
            Err(pos) => {
                self.position = if pos < self.strings.len() { Some(pos) } else { None };
                Ok(false) // No exact match
            }
        }
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.strings.len())
    }

    fn is_at_start(&self) -> bool {
        matches!(self.position, Some(0))
    }

    fn is_at_end(&self) -> bool {
        self.position.is_none()
    }
}

/// Streaming lexicographic iterator for large datasets
///
/// Designed for datasets that don't fit in memory, using buffered reading
/// and incremental processing patterns.
pub struct StreamingLexIterator<R: std::io::Read> {
    reader: std::io::BufReader<R>,
    current_line: String,
    buffer: Vec<u8>,
    finished: bool,
    line_number: usize,
}

impl<R: std::io::Read> StreamingLexIterator<R> {
    /// Create a new streaming iterator
    ///
    /// Note: This assumes the input stream contains sorted strings, one per line
    pub fn new(reader: R) -> Self {
        Self {
            reader: std::io::BufReader::new(reader),
            current_line: String::new(),
            buffer: Vec::with_capacity(8192), // 8KB initial buffer
            finished: false,
            line_number: 0,
        }
    }

    /// Read the next line from the stream
    fn read_next_line(&mut self) -> std::result::Result<bool, std::io::Error> {
        use std::io::BufRead;

        self.current_line.clear();
        match self.reader.read_line(&mut self.current_line) {
            Ok(0) => {
                self.finished = true;
                Ok(false)
            }
            Ok(_) => {
                // Remove trailing newline
                if self.current_line.ends_with('\n') {
                    self.current_line.pop();
                    if self.current_line.ends_with('\r') {
                        self.current_line.pop();
                    }
                }
                self.line_number += 1;
                Ok(true)
            }
            Err(e) => Err(e),
        }
    }
}

impl<R: std::io::Read> LexicographicIterator for StreamingLexIterator<R> {
    type Error = ZiporaError;

    fn current(&self) -> Option<&str> {
        if self.finished || self.current_line.is_empty() {
            None
        } else {
            Some(&self.current_line)
        }
    }

    fn next(&mut self) -> std::result::Result<bool, Self::Error> {
        self.read_next_line()
            .map_err(|e| ZiporaError::io_error(&format!("Failed to read line {}: {}", self.line_number + 1, e)))
    }

    fn prev(&mut self) -> std::result::Result<bool, Self::Error> {
        // Streaming iterators don't support backward movement
        Err(ZiporaError::not_supported("Streaming iterator does not support backward movement"))
    }

    fn seek_start(&mut self) -> std::result::Result<bool, Self::Error> {
        // Would require re-opening the stream - not supported for general readers
        Err(ZiporaError::not_supported("Streaming iterator does not support seeking"))
    }

    fn seek_end(&mut self) -> std::result::Result<bool, Self::Error> {
        Err(ZiporaError::not_supported("Streaming iterator does not support seeking"))
    }

    fn seek_lower_bound(&mut self, _target: &str) -> std::result::Result<bool, Self::Error> {
        Err(ZiporaError::not_supported("Streaming iterator does not support seeking"))
    }

    fn is_at_end(&self) -> bool {
        self.finished
    }
}

/// Builder for creating lexicographic iterators with different backends
pub struct LexIteratorBuilder {
    optimize_for_memory: bool,
    buffer_size: usize,
}

impl Default for LexIteratorBuilder {
    fn default() -> Self {
        Self {
            optimize_for_memory: false,
            buffer_size: 8192,
        }
    }
}

impl LexIteratorBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Optimize for memory usage over speed
    pub fn optimize_for_memory(mut self, enable: bool) -> Self {
        self.optimize_for_memory = enable;
        self
    }

    /// Set the buffer size for streaming operations
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Build a sorted vector iterator
    pub fn build_sorted_vec<'a>(self, strings: &'a [String]) -> SortedVecLexIterator<'a> {
        SortedVecLexIterator::new(strings)
    }

    /// Build a streaming iterator
    pub fn build_streaming<R: std::io::Read>(self, reader: R) -> StreamingLexIterator<R> {
        StreamingLexIterator::new(reader)
    }
}

/// Utility functions for common lexicographic operations
pub mod utils {
    use super::*;

    /// Collect all strings from an iterator into a vector
    pub fn collect_all<I>(mut iterator: I) -> std::result::Result<Vec<String>, I::Error>
    where
        I: LexicographicIterator,
    {
        let mut result = Vec::new();
        
        iterator.seek_start()?;
        while let Some(s) = iterator.current() {
            result.push(s.to_string());
            if !iterator.next()? {
                break;
            }
        }
        
        Ok(result)
    }

    /// Find the common prefix of all strings in an iterator
    pub fn find_common_prefix<I>(mut iterator: I) -> std::result::Result<String, I::Error>
    where
        I: LexicographicIterator,
    {
        iterator.seek_start()?;
        
        let first = match iterator.current() {
            Some(s) => s.to_string(),
            None => return Ok(String::new()),
        };

        let mut common = first;
        
        while iterator.next()? {
            if let Some(current) = iterator.current() {
                // Find common prefix between current common prefix and this string
                let new_len = common
                    .chars()
                    .zip(current.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                
                if new_len == 0 {
                    return Ok(String::new());
                }
                
                common.truncate(common.char_indices().nth(new_len).map_or(common.len(), |(i, _)| i));
            }
        }
        
        Ok(common)
    }

    /// Count strings with a given prefix
    pub fn count_with_prefix<I>(mut iterator: I, prefix: &str) -> std::result::Result<usize, I::Error>
    where
        I: LexicographicIterator,
    {
        let mut count = 0;
        
        // Seek to first string with prefix
        iterator.seek_lower_bound(prefix)?;
        
        while let Some(current) = iterator.current() {
            if current.starts_with(prefix) {
                count += 1;
                if !iterator.next()? {
                    break;
                }
            } else {
                break; // No more strings with this prefix
            }
        }
        
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_vec_iterator_basic() {
        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
        ];

        let mut iter = SortedVecLexIterator::new(&strings);

        // Test initial position
        assert_eq!(iter.current(), Some("apple"));
        assert!(iter.is_at_start());

        // Test forward iteration
        assert!(iter.next().unwrap());
        assert_eq!(iter.current(), Some("banana"));

        assert!(iter.next().unwrap());
        assert_eq!(iter.current(), Some("cherry"));

        assert!(iter.next().unwrap());
        assert_eq!(iter.current(), Some("date"));

        // Test end condition
        assert!(!iter.next().unwrap());
        assert!(iter.is_at_end());

        // Test seeking
        assert!(iter.seek_start().unwrap());
        assert_eq!(iter.current(), Some("apple"));

        assert!(iter.seek_end().unwrap());
        assert_eq!(iter.current(), Some("date"));
    }

    #[test]
    fn test_sorted_vec_iterator_seeking() {
        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "grape".to_string(),
            "orange".to_string(),
        ];

        let mut iter = SortedVecLexIterator::new(&strings);

        // Test exact match
        assert!(iter.seek_lower_bound("cherry").unwrap());
        assert_eq!(iter.current(), Some("cherry"));

        // Test no exact match - should position at next larger
        assert!(!iter.seek_lower_bound("coconut").unwrap());
        assert_eq!(iter.current(), Some("grape"));

        // Test seeking beyond end
        assert!(!iter.seek_lower_bound("zebra").unwrap());
        assert!(iter.is_at_end());

        // Test upper bound
        iter.seek_start().unwrap();
        assert!(!iter.seek_upper_bound("cherry").unwrap());
        assert_eq!(iter.current(), Some("grape"));
    }

    #[test]
    fn test_backward_iteration() {
        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        let mut iter = SortedVecLexIterator::new(&strings);

        // Move to end and iterate backward
        iter.seek_end().unwrap();
        assert_eq!(iter.current(), Some("cherry"));

        assert!(iter.prev().unwrap());
        assert_eq!(iter.current(), Some("banana"));

        assert!(iter.prev().unwrap());
        assert_eq!(iter.current(), Some("apple"));

        // Should stay at beginning
        assert!(!iter.prev().unwrap());
        assert_eq!(iter.current(), Some("apple"));
    }

    #[test]
    fn test_empty_collection() {
        let strings: Vec<String> = vec![];
        let mut iter = SortedVecLexIterator::new(&strings);

        assert!(iter.is_at_end());
        assert_eq!(iter.current(), None);
        assert!(!iter.next().unwrap());
        assert!(!iter.prev().unwrap());
        assert!(!iter.seek_start().unwrap());
        assert!(!iter.seek_end().unwrap());
    }

    #[test]
    fn test_utility_functions() {
        let strings = vec![
            "other".to_string(),
            "test1".to_string(),
            "test2".to_string(),
            "test3".to_string(),
        ];

        let iter = SortedVecLexIterator::new(&strings);

        // Test common prefix
        let common = utils::find_common_prefix(iter).unwrap();
        assert_eq!(common, ""); // No common prefix for all strings

        // Test prefix counting
        let iter = SortedVecLexIterator::new(&strings);
        let count = utils::count_with_prefix(iter, "test").unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_builder_pattern() {
        let strings = vec!["a".to_string(), "b".to_string()];

        let iter = LexIteratorBuilder::new()
            .optimize_for_memory(true)
            .buffer_size(4096)
            .build_sorted_vec(&strings);

        assert_eq!(iter.current(), Some("a"));
    }
}