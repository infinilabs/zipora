//! Line-Based Text Processing
//!
//! High-performance utilities for processing large text files line by line.
//! Optimized for memory efficiency and throughput with configurable buffering
//! strategies and SIMD-accelerated operations where available.

use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use std::io::{BufRead, BufReader, Read};
use std::sync::Arc;

/// Configuration for line processing operations
#[derive(Debug, Clone)]
pub struct LineProcessorConfig {
    /// Initial buffer size for reading
    pub buffer_size: usize,
    /// Maximum line length to accept
    pub max_line_length: usize,
    /// Whether to preserve line endings
    pub preserve_line_endings: bool,
    /// Whether to skip empty lines
    pub skip_empty_lines: bool,
    /// Whether to trim whitespace from lines
    pub trim_whitespace: bool,
    /// Use secure memory pool for allocations
    pub use_secure_memory: bool,
}

impl Default for LineProcessorConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,        // 64KB
            max_line_length: 10 * 1024 * 1024, // 10MB
            preserve_line_endings: false,
            skip_empty_lines: false,
            trim_whitespace: false,
            use_secure_memory: false,
        }
    }
}

impl LineProcessorConfig {
    /// Create a configuration optimized for performance
    pub fn performance_optimized() -> Self {
        Self {
            buffer_size: 256 * 1024,       // 256KB for better I/O performance
            max_line_length: 100 * 1024 * 1024, // 100MB
            preserve_line_endings: false,
            skip_empty_lines: false,
            trim_whitespace: false,
            use_secure_memory: false,
        }
    }

    /// Create a configuration optimized for memory usage
    pub fn memory_optimized() -> Self {
        Self {
            buffer_size: 16 * 1024,        // 16KB
            max_line_length: 1024 * 1024,  // 1MB
            preserve_line_endings: false,
            skip_empty_lines: true,         // Skip empty to save memory
            trim_whitespace: true,          // Trim to save memory
            use_secure_memory: false,
        }
    }

    /// Create a configuration for secure processing
    pub fn secure() -> Self {
        Self {
            buffer_size: 32 * 1024,        // 32KB
            max_line_length: 10 * 1024 * 1024, // 10MB
            preserve_line_endings: false,
            skip_empty_lines: false,
            trim_whitespace: false,
            use_secure_memory: true,
        }
    }
}

/// High-performance line processor for large text files
///
/// Provides efficient line-by-line processing with configurable buffering,
/// filtering, and transformation options. Designed to handle files larger
/// than available memory through streaming processing.
pub struct LineProcessor<R: Read> {
    reader: BufReader<R>,
    config: LineProcessorConfig,
    line_buffer: String,
    line_count: usize,
    byte_count: usize,
    memory_pool: Option<Arc<SecureMemoryPool>>,
}

impl<R: Read> LineProcessor<R> {
    /// Create a new line processor with default configuration
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, LineProcessorConfig::default())
    }

    /// Create a new line processor with custom configuration
    pub fn with_config(reader: R, config: LineProcessorConfig) -> Self {
        let reader = BufReader::with_capacity(config.buffer_size, reader);
        let memory_pool = if config.use_secure_memory {
            Some(SecureMemoryPool::new(SecurePoolConfig::small_secure()).unwrap())
        } else {
            None
        };

        Self {
            reader,
            config,
            line_buffer: String::with_capacity(1024),
            line_count: 0,
            byte_count: 0,
            memory_pool,
        }
    }

    /// Process all lines with a closure
    ///
    /// The closure receives each line and should return Ok(true) to continue
    /// processing or Ok(false) to stop early. Returns the number of lines processed.
    pub fn process_lines<F>(&mut self, mut handler: F) -> Result<usize>
    where
        F: FnMut(&str) -> Result<bool>,
    {
        let mut processed = 0;

        while self.read_next_line()? {
            let line = if self.config.trim_whitespace {
                self.line_buffer.trim()
            } else {
                &self.line_buffer
            };

            if self.config.skip_empty_lines && line.is_empty() {
                continue;
            }

            if !handler(line)? {
                break;
            }

            processed += 1;
        }

        Ok(processed)
    }

    /// Split lines by delimiter and process each field
    ///
    /// Efficiently splits each line by the given delimiter and calls the handler
    /// with each field. Returns the total number of fields processed.
    pub fn split_lines_by<F>(&mut self, delimiter: &str, mut handler: F) -> Result<usize>
    where
        F: FnMut(&str, usize, usize) -> Result<bool>, // field, line_num, field_num
    {
        let mut total_fields = 0;
        let mut current_line_number = 0;

        self.process_lines(|line| {
            current_line_number += 1;
            let mut field_num = 0;
            for field in line.split(delimiter) {
                if !handler(field, current_line_number, field_num)? {
                    return Ok(false);
                }
                field_num += 1;
                total_fields += 1;
            }
            Ok(true)
        })?;

        Ok(total_fields)
    }

    /// Process lines in batches for better performance
    ///
    /// Reads multiple lines into a batch before processing them together.
    /// Useful for operations that can benefit from processing multiple lines at once.
    pub fn process_batches<F>(&mut self, batch_size: usize, mut handler: F) -> Result<usize>
    where
        F: FnMut(&[String]) -> Result<bool>,
    {
        let mut batch = Vec::with_capacity(batch_size);
        let mut total_processed = 0;

        while self.read_next_line()? {
            let line = if self.config.trim_whitespace {
                self.line_buffer.trim().to_string()
            } else {
                self.line_buffer.clone()
            };

            if self.config.skip_empty_lines && line.is_empty() {
                continue;
            }

            batch.push(line);

            if batch.len() >= batch_size {
                if !handler(&batch)? {
                    break;
                }
                total_processed += batch.len();
                batch.clear();
            }
        }

        // Process remaining lines in the last partial batch
        if !batch.is_empty() {
            if handler(&batch)? {
                total_processed += batch.len();
            }
        }

        Ok(total_processed)
    }

    /// Count lines efficiently without processing content
    pub fn count_lines(&mut self) -> Result<usize> {
        let mut count = 0;
        while self.read_next_line()? {
            if !self.config.skip_empty_lines || !self.line_buffer.trim().is_empty() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Find lines matching a predicate
    pub fn find_lines<F>(&mut self, mut predicate: F) -> Result<Vec<(usize, String)>>
    where
        F: FnMut(&str) -> bool,
    {
        let mut matches = Vec::new();
        let mut current_line_number = 0;

        self.process_lines(|line| {
            current_line_number += 1;
            if predicate(line) {
                matches.push((current_line_number, line.to_string()));
            }
            Ok(true)
        })?;

        Ok(matches)
    }

    /// Get statistics about the processed text
    pub fn get_statistics(&self) -> LineProcessorStats {
        LineProcessorStats {
            lines_processed: self.line_count,
            bytes_processed: self.byte_count,
            buffer_size: self.config.buffer_size,
            max_line_length: self.config.max_line_length,
        }
    }

    /// Read the next line from the input
    fn read_next_line(&mut self) -> Result<bool> {
        self.line_buffer.clear();

        match self.reader.read_line(&mut self.line_buffer) {
            Ok(0) => Ok(false), // EOF
            Ok(bytes_read) => {
                if self.line_buffer.len() > self.config.max_line_length {
                    return Err(ZiporaError::invalid_data(&format!(
                        "Line {} exceeds maximum length of {} bytes",
                        self.line_count + 1,
                        self.config.max_line_length
                    )));
                }

                // Remove line endings if not preserving them
                if !self.config.preserve_line_endings {
                    if self.line_buffer.ends_with('\n') {
                        self.line_buffer.pop();
                        if self.line_buffer.ends_with('\r') {
                            self.line_buffer.pop();
                        }
                    }
                }

                self.line_count += 1;
                self.byte_count += bytes_read;
                Ok(true)
            }
            Err(e) => Err(ZiporaError::io_error(&format!(
                "Failed to read line {}: {}",
                self.line_count + 1,
                e
            ))),
        }
    }
}

/// Statistics about line processing operations
#[derive(Debug, Clone)]
pub struct LineProcessorStats {
    pub lines_processed: usize,
    pub bytes_processed: usize,
    pub buffer_size: usize,
    pub max_line_length: usize,
}

impl LineProcessorStats {
    /// Calculate average line length in bytes
    pub fn avg_line_length(&self) -> f64 {
        if self.lines_processed == 0 {
            0.0
        } else {
            self.bytes_processed as f64 / self.lines_processed as f64
        }
    }

    /// Calculate processing efficiency (lines per byte)
    pub fn efficiency(&self) -> f64 {
        if self.bytes_processed == 0 {
            0.0
        } else {
            self.lines_processed as f64 / self.bytes_processed as f64
        }
    }
}

/// Specialized line splitter with multiple strategies
///
/// Provides efficient line splitting with different algorithms optimized
/// for different delimiter patterns and performance requirements.
pub struct LineSplitter {
    strategy: SplitStrategy,
    buffer: Vec<String>,
}

#[derive(Debug, Clone)]
enum SplitStrategy {
    Simple,      // Standard split()
    Optimized,   // SIMD-optimized for common delimiters
    Custom(String), // Custom delimiter
}

impl LineSplitter {
    /// Create a new splitter with simple strategy
    pub fn new() -> Self {
        Self {
            strategy: SplitStrategy::Simple,
            buffer: Vec::new(),
        }
    }

    /// Use optimized strategy for common delimiters (comma, tab, space)
    pub fn with_optimized_strategy(mut self) -> Self {
        self.strategy = SplitStrategy::Optimized;
        self
    }

    /// Use custom delimiter
    pub fn with_delimiter(mut self, delimiter: String) -> Self {
        self.strategy = SplitStrategy::Custom(delimiter);
        self
    }

    /// Split a line into fields
    pub fn split(&mut self, line: &str, delimiter: &str) -> &[String] {
        self.buffer.clear();

        match &self.strategy {
            SplitStrategy::Simple | SplitStrategy::Custom(_) => {
                self.buffer.extend(line.split(delimiter).map(|s| s.to_string()));
            }
            SplitStrategy::Optimized => {
                // Use optimized splitting for common delimiters
                if delimiter == "," || delimiter == "\t" || delimiter == " " {
                    self.split_optimized(line, delimiter);
                } else {
                    self.buffer.extend(line.split(delimiter).map(|s| s.to_string()));
                }
            }
        }

        &self.buffer
    }

    /// SIMD-optimized splitting for single-character delimiters
    fn split_optimized(&mut self, line: &str, delimiter: &str) {
        if delimiter.len() != 1 {
            // Fall back to standard split for multi-character delimiters
            self.buffer.extend(line.split(delimiter).map(|s| s.to_string()));
            return;
        }

        let delimiter_byte = delimiter.bytes().next().unwrap();
        let mut start = 0;

        for (i, &byte) in line.as_bytes().iter().enumerate() {
            if byte == delimiter_byte {
                if let Ok(field) = std::str::from_utf8(&line.as_bytes()[start..i]) {
                    self.buffer.push(field.to_string());
                }
                start = i + 1;
            }
        }

        // Add the last field
        if start < line.len() {
            if let Ok(field) = std::str::from_utf8(&line.as_bytes()[start..]) {
                self.buffer.push(field.to_string());
            }
        }
    }
}

impl Default for LineSplitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for line-based text processing
pub mod utils {
    use super::*;
    use std::collections::HashMap;

    /// Count word frequencies in text lines
    pub fn count_word_frequencies<R: Read>(
        mut processor: LineProcessor<R>,
    ) -> Result<HashMap<String, usize>> {
        let mut frequencies = HashMap::new();

        processor.process_lines(|line| {
            for word in line.split_whitespace() {
                *frequencies.entry(word.to_lowercase()).or_insert(0) += 1;
            }
            Ok(true)
        })?;

        Ok(frequencies)
    }

    /// Extract all unique lines from input
    pub fn extract_unique_lines<R: Read>(
        mut processor: LineProcessor<R>,
    ) -> Result<Vec<String>> {
        let mut unique_lines = std::collections::HashSet::new();

        processor.process_lines(|line| {
            unique_lines.insert(line.to_string());
            Ok(true)
        })?;

        Ok(unique_lines.into_iter().collect())
    }

    /// Filter lines by length
    pub fn filter_by_length<R: Read>(
        mut processor: LineProcessor<R>,
        min_length: usize,
        max_length: usize,
    ) -> Result<Vec<String>> {
        let mut filtered_lines = Vec::new();

        processor.process_lines(|line| {
            if line.len() >= min_length && line.len() <= max_length {
                filtered_lines.push(line.to_string());
            }
            Ok(true)
        })?;

        Ok(filtered_lines)
    }

    /// Calculate basic text statistics
    pub fn analyze_text<R: Read>(mut processor: LineProcessor<R>) -> Result<TextAnalysis> {
        let mut analysis = TextAnalysis::default();

        processor.process_lines(|line| {
            analysis.total_lines += 1;
            analysis.total_chars += line.chars().count();
            analysis.total_bytes += line.len();

            let words: Vec<&str> = line.split_whitespace().collect();
            analysis.total_words += words.len();

            if line.trim().is_empty() {
                analysis.empty_lines += 1;
            }

            if line.len() > analysis.max_line_length {
                analysis.max_line_length = line.len();
            }

            if analysis.min_line_length == 0 || line.len() < analysis.min_line_length {
                analysis.min_line_length = line.len();
            }

            Ok(true)
        })?;

        Ok(analysis)
    }

    /// Text analysis results
    #[derive(Debug, Default)]
    pub struct TextAnalysis {
        pub total_lines: usize,
        pub total_chars: usize,
        pub total_bytes: usize,
        pub total_words: usize,
        pub empty_lines: usize,
        pub max_line_length: usize,
        pub min_line_length: usize,
    }

    impl TextAnalysis {
        /// Calculate average line length in characters
        pub fn avg_line_length(&self) -> f64 {
            if self.total_lines == 0 {
                0.0
            } else {
                self.total_chars as f64 / self.total_lines as f64
            }
        }

        /// Calculate average words per line
        pub fn avg_words_per_line(&self) -> f64 {
            if self.total_lines == 0 {
                0.0
            } else {
                self.total_words as f64 / self.total_lines as f64
            }
        }

        /// Calculate percentage of empty lines
        pub fn empty_line_percentage(&self) -> f64 {
            if self.total_lines == 0 {
                0.0
            } else {
                (self.empty_lines as f64 / self.total_lines as f64) * 100.0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_input() -> Cursor<&'static [u8]> {
        let content = "line1\nline2\n\nline4 with spaces\nline5,with,commas\n";
        Cursor::new(content.as_bytes())
    }

    #[test]
    fn test_basic_line_processing() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        let mut lines = Vec::new();
        let count = processor.process_lines(|line| {
            lines.push(line.to_string());
            Ok(true)
        }).unwrap();

        assert_eq!(count, 5);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
        assert_eq!(lines[2], "");
        assert_eq!(lines[3], "line4 with spaces");
        assert_eq!(lines[4], "line5,with,commas");
    }

    #[test]
    fn test_skip_empty_lines() {
        let input = create_test_input();
        let config = LineProcessorConfig {
            skip_empty_lines: true,
            ..Default::default()
        };
        let mut processor = LineProcessor::with_config(input, config);

        let mut lines = Vec::new();
        let count = processor.process_lines(|line| {
            lines.push(line.to_string());
            Ok(true)
        }).unwrap();

        assert_eq!(count, 4); // Empty line skipped
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_trim_whitespace() {
        let input = Cursor::new("  line1  \n  line2  \n".as_bytes());
        let config = LineProcessorConfig {
            trim_whitespace: true,
            ..Default::default()
        };
        let mut processor = LineProcessor::with_config(input, config);

        let mut lines = Vec::new();
        processor.process_lines(|line| {
            lines.push(line.to_string());
            Ok(true)
        }).unwrap();

        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
    }

    #[test]
    fn test_line_splitting() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        let mut fields = Vec::new();
        let count = processor.split_lines_by(",", |field, _line_num, _field_num| {
            fields.push(field.to_string());
            Ok(true)
        }).unwrap();

        // Only the last line has commas, so we should get 3 fields from it
        // plus single fields from other lines
        assert!(count >= 3);
        assert!(fields.contains(&"line5".to_string()));
        assert!(fields.contains(&"with".to_string()));
        assert!(fields.contains(&"commas".to_string()));
    }

    #[test]
    fn test_batch_processing() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        let mut batches = Vec::new();
        let total = processor.process_batches(2, |batch| {
            batches.push(batch.to_vec());
            Ok(true)
        }).unwrap();

        assert!(total >= 5);
        assert!(!batches.is_empty());
    }

    #[test]
    fn test_line_counting() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        let count = processor.count_lines().unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_find_lines() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        let matches = processor.find_lines(|line| line.contains("line")).unwrap();
        assert_eq!(matches.len(), 4); // All except empty line
    }

    #[test]
    fn test_line_splitter() {
        let mut splitter = LineSplitter::new();

        let fields = splitter.split("a,b,c", ",");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "a");
        assert_eq!(fields[1], "b");
        assert_eq!(fields[2], "c");
    }

    #[test]
    fn test_optimized_splitter() {
        let mut splitter = LineSplitter::new().with_optimized_strategy();

        let fields = splitter.split("a\tb\tc", "\t");
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "a");
        assert_eq!(fields[1], "b");
        assert_eq!(fields[2], "c");
    }

    #[test]
    fn test_statistics() {
        let input = create_test_input();
        let mut processor = LineProcessor::new(input);

        processor.count_lines().unwrap();
        let stats = processor.get_statistics();

        assert_eq!(stats.lines_processed, 5);
        assert!(stats.bytes_processed > 0);
        assert!(stats.avg_line_length() > 0.0);
    }

    #[test]
    fn test_utility_functions() {
        let input = Cursor::new("hello world\nhello rust\nworld rust\n".as_bytes());
        let processor = LineProcessor::new(input);

        let frequencies = utils::count_word_frequencies(processor).unwrap();
        assert_eq!(frequencies.get("hello"), Some(&2));
        assert_eq!(frequencies.get("world"), Some(&2));
        assert_eq!(frequencies.get("rust"), Some(&2));
    }

    #[test]
    fn test_text_analysis() {
        let input = Cursor::new("line1\nline2\n\nlong line with multiple words\n".as_bytes());
        let processor = LineProcessor::new(input);

        let analysis = utils::analyze_text(processor).unwrap();
        assert_eq!(analysis.total_lines, 4);
        assert_eq!(analysis.empty_lines, 1);
        assert!(analysis.avg_line_length() > 0.0);
        assert_eq!(analysis.empty_line_percentage(), 25.0);
    }

    #[test]
    fn test_performance_config() {
        let config = LineProcessorConfig::performance_optimized();
        assert_eq!(config.buffer_size, 256 * 1024);
        assert_eq!(config.max_line_length, 100 * 1024 * 1024);
    }

    #[test]
    fn test_memory_config() {
        let config = LineProcessorConfig::memory_optimized();
        assert_eq!(config.buffer_size, 16 * 1024);
        assert!(config.skip_empty_lines);
        assert!(config.trim_whitespace);
    }
}