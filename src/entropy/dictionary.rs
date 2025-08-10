//! Dictionary-based compression implementation
//!
//! This module provides LZ-style dictionary compression algorithms that find
//! and encode repeated substrings for efficient compression.

use crate::algorithms::SuffixArray;
use crate::error::{Result, ZiporaError};
use std::collections::HashMap;

/// Rolling hash implementation for fast string matching
#[derive(Debug, Clone)]
struct RollingHash {
    hash: u64,
    base: u64,
    modulus: u64,
    window_size: usize,
    power: u64,
}

impl RollingHash {
    const BASE: u64 = 257;
    const MODULUS: u64 = (1u64 << 61) - 1; // Large prime

    fn new(window_size: usize) -> Self {
        let mut power: u64 = 1;
        for _ in 0..window_size.saturating_sub(1) {
            power = power.wrapping_mul(Self::BASE) % Self::MODULUS;
        }

        Self {
            hash: 0,
            base: Self::BASE,
            modulus: Self::MODULUS,
            window_size,
            power,
        }
    }

    fn hash_slice(&mut self, data: &[u8]) -> u64 {
        self.hash = 0;
        for &byte in data.iter().take(self.window_size) {
            self.hash = (self.hash.wrapping_mul(self.base) + byte as u64) % self.modulus;
        }
        self.hash
    }

    fn roll(&mut self, old_byte: u8, new_byte: u8) -> u64 {
        // Remove old byte and add new byte
        let old_contrib = (old_byte as u64).wrapping_mul(self.power) % self.modulus;
        self.hash = (self.hash + self.modulus - old_contrib) % self.modulus;
        self.hash = (self.hash.wrapping_mul(self.base) + new_byte as u64) % self.modulus;
        self.hash
    }
}

/// Simple bloom filter for quick pattern rejection
#[derive(Debug)]
struct BloomFilter {
    bits: Vec<u64>,
    size: usize,
    hash_functions: usize,
}

impl BloomFilter {
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let size = (-((expected_items as f64) * false_positive_rate.ln()) / (2.0_f64.ln().powi(2)))
            .ceil() as usize;
        let hash_functions = ((size as f64 / expected_items as f64) * 2.0_f64.ln()).ceil() as usize;
        let num_u64s = (size + 63) / 64;

        Self {
            bits: vec![0; num_u64s],
            size,
            hash_functions: hash_functions.max(1).min(8), // Limit to reasonable range
        }
    }

    fn insert(&mut self, item: &[u8]) {
        for i in 0..self.hash_functions {
            let hash = self.hash_item(item, i);
            let index = hash % self.size;
            let word_index = index / 64;
            let bit_index = index % 64;
            self.bits[word_index] |= 1u64 << bit_index;
        }
    }

    fn contains(&self, item: &[u8]) -> bool {
        for i in 0..self.hash_functions {
            let hash = self.hash_item(item, i);
            let index = hash % self.size;
            let word_index = index / 64;
            let bit_index = index % 64;
            if (self.bits[word_index] & (1u64 << bit_index)) == 0 {
                return false;
            }
        }
        true
    }

    fn hash_item(&self, item: &[u8], seed: usize) -> usize {
        let mut hash = seed as u64;
        for &byte in item {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash as usize
    }
}

/// Dictionary entry for compression
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DictionaryEntry {
    /// Offset to the previous occurrence
    pub offset: u32,
    /// Length of the match
    pub length: u32,
}

impl DictionaryEntry {
    /// Create new dictionary entry
    pub fn new(offset: u32, length: u32) -> Self {
        Self { offset, length }
    }
}

/// Dictionary builder for creating compression dictionaries
#[derive(Debug)]
pub struct DictionaryBuilder {
    max_entries: usize,
    min_match_length: usize,
    max_match_length: usize,
    window_size: usize,
}

impl DictionaryBuilder {
    /// Create new dictionary builder
    pub fn new() -> Self {
        Self {
            max_entries: 4096,
            min_match_length: 3,
            max_match_length: 258,
            window_size: 32768,
        }
    }

    /// Set maximum number of dictionary entries
    pub fn max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set minimum match length
    pub fn min_match_length(mut self, min_length: usize) -> Self {
        self.min_match_length = min_length;
        self
    }

    /// Set maximum match length
    pub fn max_match_length(mut self, max_length: usize) -> Self {
        self.max_match_length = max_length;
        self
    }

    /// Set sliding window size
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Build dictionary from training data
    pub fn build(&self, data: &[u8]) -> Dictionary {
        let mut entries = HashMap::new();
        let mut hash_table: HashMap<u32, Vec<usize>> = HashMap::new();

        for i in 0..data.len().saturating_sub(self.min_match_length - 1) {
            // Create hash for current position
            let hash = self.hash_bytes(&data[i..i + self.min_match_length]);

            // Look for matches in the hash table
            if let Some(positions) = hash_table.get(&hash) {
                for &pos in positions.iter().rev() {
                    if i - pos > self.window_size {
                        break;
                    }

                    let match_len = self.find_match_length(&data, pos, i);
                    if match_len >= self.min_match_length {
                        let offset = (i - pos) as u32;
                        let entry = DictionaryEntry::new(offset, match_len as u32);

                        // Use the substring as key
                        let key = data[i..i + match_len].to_vec();
                        entries.insert(key, entry);

                        if entries.len() >= self.max_entries {
                            break;
                        }
                    }
                }
            }

            // Add current position to hash table
            hash_table.entry(hash).or_insert_with(Vec::new).push(i);

            if entries.len() >= self.max_entries {
                break;
            }
        }

        Dictionary { entries }
    }

    /// Hash a sequence of bytes
    fn hash_bytes(&self, bytes: &[u8]) -> u32 {
        let mut hash = 0u32;
        for &byte in bytes {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    /// Find the length of a match between two positions
    fn find_match_length(&self, data: &[u8], pos1: usize, pos2: usize) -> usize {
        let max_len = (data.len() - pos2).min(self.max_match_length);
        let mut len = 0;

        while len < max_len && data[pos1 + len] == data[pos2 + len] {
            len += 1;
        }

        len
    }
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Dictionary for compression/decompression
#[derive(Debug, Clone)]
pub struct Dictionary {
    entries: HashMap<Vec<u8>, DictionaryEntry>,
}

impl Dictionary {
    /// Create empty dictionary
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Get dictionary entry for a sequence
    pub fn get(&self, sequence: &[u8]) -> Option<&DictionaryEntry> {
        self.entries.get(sequence)
    }

    /// Add entry to dictionary
    pub fn insert(&mut self, sequence: Vec<u8>, entry: DictionaryEntry) {
        self.entries.insert(sequence, entry);
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize dictionary
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Write number of entries
        result.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());

        for (sequence, entry) in &self.entries {
            // Write sequence length and data
            result.extend_from_slice(&(sequence.len() as u16).to_le_bytes());
            result.extend_from_slice(sequence);

            // Write entry data
            result.extend_from_slice(&entry.offset.to_le_bytes());
            result.extend_from_slice(&entry.length.to_le_bytes());
        }

        result
    }

    /// Deserialize dictionary
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(ZiporaError::invalid_data("Dictionary data too short"));
        }

        let num_entries = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut entries = HashMap::new();
        let mut offset = 4;

        for _ in 0..num_entries {
            if offset + 2 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated dictionary data"));
            }

            // Read sequence length
            let seq_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + seq_len + 8 > data.len() {
                return Err(ZiporaError::invalid_data("Truncated dictionary sequence"));
            }

            // Read sequence
            let sequence = data[offset..offset + seq_len].to_vec();
            offset += seq_len;

            // Read entry data
            let entry_offset = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let entry_length = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            entries.insert(sequence, DictionaryEntry::new(entry_offset, entry_length));
        }

        Ok(Self { entries })
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Dictionary compressor
#[derive(Debug)]
pub struct DictionaryCompressor {
    dictionary: Dictionary,
    min_match_length: usize,
    max_match_length: usize,
}

impl DictionaryCompressor {
    /// Create compressor with dictionary
    pub fn new(dictionary: Dictionary) -> Self {
        Self {
            dictionary,
            min_match_length: 3,
            max_match_length: 258,
        }
    }

    /// Set minimum match length
    pub fn min_match_length(mut self, min_length: usize) -> Self {
        self.min_match_length = min_length;
        self
    }

    /// Set maximum match length
    pub fn max_match_length(mut self, max_length: usize) -> Self {
        self.max_match_length = max_length;
        self
    }

    /// Compress data using LZ77-style sliding window compression
    ///
    /// This implementation uses proper LZ77 back-reference semantics where:
    /// - offset = distance back from current output position
    /// - length = number of bytes to copy
    ///
    /// The encoding format is:
    /// - Literal: flag(0) + byte
    /// - Match: flag(1) + offset(4 bytes) + length(4 bytes)
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut pos = 0;
        let window_size = 32768; // Standard LZ77 window size

        while pos < data.len() {
            let mut best_match_offset = 0;
            let mut best_match_length = 0;

            // Search backwards in the sliding window for matches
            let search_start = pos.saturating_sub(window_size);

            for search_pos in search_start..pos {
                let max_match_len = (data.len() - pos).min(self.max_match_length);
                let mut match_len = 0;

                // Find match length
                while match_len < max_match_len
                    && data[search_pos + match_len] == data[pos + match_len]
                {
                    match_len += 1;
                }

                // Update best match if this is longer and meets minimum length
                // Increase minimum match length to compensate for encoding overhead
                if match_len >= self.min_match_length.max(10) && match_len > best_match_length {
                    best_match_offset = pos - search_pos; // Distance back from current position
                    best_match_length = match_len;
                }
            }

            if best_match_length > 0 {
                // Encode as match: flag(1) + offset + length
                result.push(1); // Match flag
                result.extend_from_slice(&(best_match_offset as u32).to_le_bytes());
                result.extend_from_slice(&(best_match_length as u32).to_le_bytes());
                pos += best_match_length;
            } else {
                // Encode as literal: flag(0) + byte
                result.push(0); // Literal flag
                result.push(data[pos]);
                pos += 1;
            }
        }

        Ok(result)
    }

    /// Decompress data using dictionary
    pub fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut pos = 0;

        while pos < compressed_data.len() {
            if pos >= compressed_data.len() {
                break;
            }

            let flag = compressed_data[pos];
            pos += 1;

            if flag == 0 {
                // Literal byte
                if pos >= compressed_data.len() {
                    return Err(ZiporaError::invalid_data(
                        "Unexpected end of compressed data",
                    ));
                }
                result.push(compressed_data[pos]);
                pos += 1;
            } else if flag == 1 {
                // Dictionary match
                if pos + 8 > compressed_data.len() {
                    return Err(ZiporaError::invalid_data("Truncated match data"));
                }

                let offset = u32::from_le_bytes([
                    compressed_data[pos],
                    compressed_data[pos + 1],
                    compressed_data[pos + 2],
                    compressed_data[pos + 3],
                ]);
                pos += 4;

                let length = u32::from_le_bytes([
                    compressed_data[pos],
                    compressed_data[pos + 1],
                    compressed_data[pos + 2],
                    compressed_data[pos + 3],
                ]) as usize;
                pos += 4;

                // This is LZ77-style back-reference compression
                // offset = distance back from current position
                // length = number of bytes to copy
                if offset == 0 || result.len() < offset as usize {
                    return Err(ZiporaError::invalid_data("Invalid back-reference offset"));
                }

                let start_pos = result.len() - offset as usize;

                // Handle potential overlapping copies by copying byte by byte
                // This is necessary when the copy length > offset (pattern repeats)
                for i in 0..length {
                    let copy_pos = start_pos + i;
                    if copy_pos < result.len() {
                        let byte = result[copy_pos];
                        result.push(byte);
                    } else {
                        // This case happens when length > offset (repeating pattern)
                        // Copy from the already copied portion
                        let wrapped_pos = start_pos + (i % offset as usize);
                        if wrapped_pos < result.len() {
                            let byte = result[wrapped_pos];
                            result.push(byte);
                        } else {
                            return Err(ZiporaError::invalid_data(
                                "Back-reference calculation error",
                            ));
                        }
                    }
                }
            } else {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid compression flag: {}",
                    flag
                )));
            }
        }

        Ok(result)
    }

    /// Get the dictionary
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    /// Estimate compression ratio
    pub fn estimate_compression_ratio(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let compressed = self.compress(data).unwrap_or_else(|_| data.to_vec());
        compressed.len() as f64 / data.len() as f64
    }
}

/// Optimized dictionary compressor using suffix arrays and advanced algorithms
///
/// This compressor addresses the performance bottleneck in the original implementation
/// by replacing O(n²) linear search with O(log n) suffix array search, adding rolling
/// hash for efficient pattern matching, and bloom filters for quick rejection.
#[derive(Debug)]
pub struct OptimizedDictionaryCompressor {
    suffix_array: SuffixArray,
    text: Vec<u8>,
    bloom_filter: BloomFilter,
    hash_table: HashMap<u64, Vec<usize>>,
    min_match_length: usize,
    max_match_length: usize,
    window_size: usize,
}

impl OptimizedDictionaryCompressor {
    /// Create optimized compressor from training data
    pub fn new(data: &[u8]) -> Result<Self> {
        Self::with_config(data, 3, 258, 32768)
    }

    /// Create optimized compressor with custom configuration
    pub fn with_config(
        data: &[u8],
        min_match_length: usize,
        max_match_length: usize,
        window_size: usize,
    ) -> Result<Self> {
        // Build suffix array for fast pattern search
        let suffix_array = SuffixArray::new(data)?;

        // Create bloom filter for quick rejection
        let expected_patterns = data.len() / min_match_length;
        let mut bloom_filter = BloomFilter::new(expected_patterns, 0.01); // 1% false positive rate

        // Build rolling hash table for O(1) pattern lookup
        let mut hash_table: HashMap<u64, Vec<usize>> = HashMap::new();
        if data.len() >= min_match_length {
            let mut rolling_hash = RollingHash::new(min_match_length);

            // Initialize hash for first window
            let first_hash = rolling_hash.hash_slice(&data[0..min_match_length]);
            hash_table
                .entry(first_hash)
                .or_insert_with(Vec::new)
                .push(0);

            // Roll through remaining positions
            for i in 1..=data.len().saturating_sub(min_match_length) {
                let hash = rolling_hash.roll(data[i - 1], data[i + min_match_length - 1]);
                hash_table.entry(hash).or_insert_with(Vec::new).push(i);
            }
        }

        // Populate bloom filter with potential patterns
        for i in 0..data.len().saturating_sub(min_match_length - 1) {
            let pattern = &data[i..i + min_match_length];
            bloom_filter.insert(pattern);
        }

        Ok(Self {
            suffix_array,
            text: data.to_vec(),
            bloom_filter,
            hash_table,
            min_match_length,
            max_match_length,
            window_size,
        })
    }

    /// Compress data using optimized LZ77-style algorithm with rolling hash
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut pos = 0;

        // Create rolling hash for input data if we have enough data
        let mut rolling_hash = if data.len() >= self.min_match_length {
            Some(RollingHash::new(self.min_match_length))
        } else {
            None
        };

        while pos < data.len() {
            let mut best_match_offset = 0;
            let mut best_match_length = 0;

            // Only search if we have enough data for minimum match
            if pos + self.min_match_length <= data.len() {
                let pattern = &data[pos..pos + self.min_match_length];

                // Quick rejection using bloom filter
                if self.bloom_filter.contains(pattern) {
                    // Use rolling hash for fast candidate lookup
                    let hash = if let Some(ref mut rh) = rolling_hash {
                        if pos == 0 {
                            rh.hash_slice(pattern)
                        } else {
                            rh.roll(data[pos - 1], data[pos + self.min_match_length - 1])
                        }
                    } else {
                        0 // Fallback for very small data
                    };

                    // Try rolling hash optimization first
                    if let Some(candidate_positions) = self.hash_table.get(&hash) {
                        // Check each candidate position for the best match within the sliding window
                        for &suffix_pos in candidate_positions {
                            // Only consider positions that came before current position (look backwards)
                            if suffix_pos >= pos {
                                continue; // Can't reference future positions
                            }

                            let distance = pos - suffix_pos;
                            if distance > self.window_size || distance == 0 {
                                continue;
                            }

                            // Verify the pattern matches (hash collision check)
                            if suffix_pos + self.min_match_length <= self.text.len() {
                                let training_pattern =
                                    &self.text[suffix_pos..suffix_pos + self.min_match_length];
                                if training_pattern != pattern {
                                    continue; // Hash collision, skip
                                }
                            } else {
                                continue;
                            }

                            // Extend the match as far as possible
                            let max_possible = (data.len() - pos).min(self.max_match_length);
                            let mut match_length = self.min_match_length;

                            while match_length < max_possible
                                && suffix_pos + match_length < self.text.len()
                                && self.text[suffix_pos + match_length] == data[pos + match_length]
                            {
                                match_length += 1;
                            }

                            // Update best match if this is better and meets minimum length requirement
                            // Apply the same minimum length threshold as original algorithm
                            if match_length >= self.min_match_length.max(10)
                                && match_length > best_match_length
                            {
                                best_match_offset = distance;
                                best_match_length = match_length;
                            }
                        }
                    }

                    // Fallback to suffix array search if rolling hash didn't find good matches
                    // This ensures we don't miss any matches due to hash table limitations
                    if best_match_length == 0 {
                        // Use suffix array to find all occurrences of the pattern
                        let (start, count) = self.suffix_array.search(&self.text, pattern);

                        // Check each occurrence for the best match within the sliding window
                        for i in start..start + count {
                            if let Some(suffix_pos) = self.suffix_array.suffix_at_rank(i) {
                                // Only consider positions that came before current position (look backwards)
                                if suffix_pos >= pos {
                                    continue; // Can't reference future positions
                                }

                                let distance = pos - suffix_pos;
                                if distance > self.window_size || distance == 0 {
                                    continue;
                                }

                                // Extend the match as far as possible
                                let max_possible = (data.len() - pos).min(self.max_match_length);
                                let mut match_length = self.min_match_length;

                                while match_length < max_possible
                                    && suffix_pos + match_length < self.text.len()
                                    && self.text[suffix_pos + match_length]
                                        == data[pos + match_length]
                                {
                                    match_length += 1;
                                }

                                // Update best match if this is better and meets minimum length requirement
                                if match_length >= self.min_match_length.max(10)
                                    && match_length > best_match_length
                                {
                                    best_match_offset = distance;
                                    best_match_length = match_length;
                                }
                            }
                        }
                    }
                }
            }

            // Encode the best match or literal
            if best_match_length >= self.min_match_length.max(10) {
                // Encode as match: flag(1) + offset + length
                result.push(1);
                result.extend_from_slice(&(best_match_offset as u32).to_le_bytes());
                result.extend_from_slice(&(best_match_length as u32).to_le_bytes());
                pos += best_match_length;
            } else {
                // Encode as literal: flag(0) + byte
                result.push(0);
                result.push(data[pos]);
                pos += 1;
            }
        }

        Ok(result)
    }

    /// Decompress data (same algorithm as original)
    pub fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut pos = 0;

        while pos < compressed_data.len() {
            if pos >= compressed_data.len() {
                break;
            }

            let flag = compressed_data[pos];
            pos += 1;

            if flag == 0 {
                // Literal byte
                if pos >= compressed_data.len() {
                    return Err(ZiporaError::invalid_data(
                        "Unexpected end of compressed data",
                    ));
                }
                result.push(compressed_data[pos]);
                pos += 1;
            } else if flag == 1 {
                // Dictionary match
                if pos + 8 > compressed_data.len() {
                    return Err(ZiporaError::invalid_data("Truncated match data"));
                }

                let offset = u32::from_le_bytes([
                    compressed_data[pos],
                    compressed_data[pos + 1],
                    compressed_data[pos + 2],
                    compressed_data[pos + 3],
                ]);
                pos += 4;

                let length = u32::from_le_bytes([
                    compressed_data[pos],
                    compressed_data[pos + 1],
                    compressed_data[pos + 2],
                    compressed_data[pos + 3],
                ]) as usize;
                pos += 4;

                if offset == 0 || result.len() < offset as usize {
                    return Err(ZiporaError::invalid_data("Invalid back-reference offset"));
                }

                let start_pos = result.len() - offset as usize;

                // Handle overlapping copies
                for i in 0..length {
                    let copy_pos = start_pos + i;
                    if copy_pos < result.len() {
                        let byte = result[copy_pos];
                        result.push(byte);
                    } else {
                        let wrapped_pos = start_pos + (i % offset as usize);
                        if wrapped_pos < result.len() {
                            let byte = result[wrapped_pos];
                            result.push(byte);
                        } else {
                            return Err(ZiporaError::invalid_data(
                                "Back-reference calculation error",
                            ));
                        }
                    }
                }
            } else {
                return Err(ZiporaError::invalid_data(format!(
                    "Invalid compression flag: {}",
                    flag
                )));
            }
        }

        Ok(result)
    }

    /// Estimate compression ratio
    pub fn estimate_compression_ratio(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let compressed = self.compress(data).unwrap_or_else(|_| data.to_vec());
        compressed.len() as f64 / data.len() as f64
    }

    /// Get compression statistics
    pub fn stats(&self) -> String {
        format!(
            "OptimizedDictionaryCompressor: text_len={}, min_match={}, max_match={}, window={}",
            self.text.len(),
            self.min_match_length,
            self.max_match_length,
            self.window_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_entry() {
        let entry = DictionaryEntry::new(10, 5);
        assert_eq!(entry.offset, 10);
        assert_eq!(entry.length, 5);
    }

    #[test]
    fn test_dictionary_basic_operations() {
        let mut dict = Dictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);

        let sequence = b"hello".to_vec();
        let entry = DictionaryEntry::new(5, 3);
        dict.insert(sequence.clone(), entry.clone());

        assert!(!dict.is_empty());
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.get(&sequence), Some(&entry));
    }

    #[test]
    fn test_dictionary_builder() {
        let data = b"hello world hello world hello";
        let builder = DictionaryBuilder::new();
        let dict = builder.build(data);

        assert!(!dict.is_empty());
        // Should find repeated "hello" and "world"
    }

    #[test]
    fn test_dictionary_serialization() {
        let mut dict = Dictionary::new();
        dict.insert(b"hello".to_vec(), DictionaryEntry::new(10, 5));
        dict.insert(b"world".to_vec(), DictionaryEntry::new(15, 5));

        let serialized = dict.serialize();
        let deserialized = Dictionary::deserialize(&serialized).unwrap();

        assert_eq!(dict.len(), deserialized.len());
        assert_eq!(dict.get(b"hello"), deserialized.get(b"hello"));
        assert_eq!(dict.get(b"world"), deserialized.get(b"world"));
    }

    #[test]
    fn test_dictionary_compression() {
        let data = b"hello world hello world hello world";

        // Build dictionary from the data
        let builder = DictionaryBuilder::new();
        let dict = builder.build(data);

        let compressor = DictionaryCompressor::new(dict);
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_dictionary_compression_small_data() {
        let data = b"hi";

        let builder = DictionaryBuilder::new();
        let dict = builder.build(data);

        let compressor = DictionaryCompressor::new(dict);
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_dictionary_builder_configuration() {
        let builder = DictionaryBuilder::new()
            .max_entries(100)
            .min_match_length(4)
            .max_match_length(100)
            .window_size(1024);

        assert_eq!(builder.max_entries, 100);
        assert_eq!(builder.min_match_length, 4);
        assert_eq!(builder.max_match_length, 100);
        assert_eq!(builder.window_size, 1024);
    }

    #[test]
    fn test_dictionary_compression_ratio() {
        // Use data with longer repeated patterns that will benefit from compression
        let pattern =
            b"this is a long repeated pattern that should compress well because it repeats often";
        let mut data = Vec::new();
        for _ in 0..5 {
            data.extend_from_slice(pattern);
        }

        let builder = DictionaryBuilder::new();
        let dict = builder.build(&data);

        let compressor = DictionaryCompressor::new(dict);
        let ratio = compressor.estimate_compression_ratio(&data);

        // Should achieve good compression due to long repeated patterns
        assert!(
            ratio < 1.0,
            "Compression ratio was {:.3}, expected < 1.0",
            ratio
        );
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_empty_dictionary() {
        let dict = Dictionary::new();
        let compressor = DictionaryCompressor::new(dict);

        let data = b"hello world";
        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        // Should still work with all literals
        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_dictionary_error_handling() {
        // Test deserializing invalid data
        let invalid_data = vec![1, 2, 3]; // Too short
        let result = Dictionary::deserialize(&invalid_data);
        assert!(result.is_err());

        // Test decompressing invalid data
        let dict = Dictionary::new();
        let compressor = DictionaryCompressor::new(dict);
        let invalid_compressed = vec![1]; // Incomplete match data
        let result = compressor.decompress(&invalid_compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_long_repeated_pattern() {
        let pattern = b"abcdefghijk";
        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(pattern);
        }

        let builder = DictionaryBuilder::new();
        let dict = builder.build(&data);

        let compressor = DictionaryCompressor::new(dict);
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);

        // Should achieve good compression ratio
        let ratio = compressed.len() as f64 / data.len() as f64;
        assert!(ratio < 0.8); // Should compress to less than 80%
    }

    // Tests for optimized dictionary compression
    #[test]
    fn test_optimized_compressor_basic() {
        let data = b"hello world hello world hello world";
        let compressor = OptimizedDictionaryCompressor::new(data).unwrap();

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_optimized_compressor_small_data() {
        let data = b"hi";
        let compressor = OptimizedDictionaryCompressor::new(data).unwrap();

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_optimized_compressor_configuration() {
        let data = b"test data test data test data";
        let compressor = OptimizedDictionaryCompressor::with_config(data, 4, 100, 1024).unwrap();

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
        assert!(compressor.stats().contains("min_match=4"));
    }

    #[test]
    fn test_optimized_compressor_long_patterns() {
        let pattern = b"this is a very long pattern that should compress very well when repeated multiple times";
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(pattern);
        }

        let compressor = OptimizedDictionaryCompressor::new(&data).unwrap();
        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);

        // Should achieve excellent compression due to long repeated patterns
        let ratio = compressed.len() as f64 / data.len() as f64;
        assert!(
            ratio < 0.5,
            "Compression ratio was {:.3}, expected < 0.5",
            ratio
        );
    }

    #[test]
    fn test_optimized_compressor_vs_original() {
        let pattern = b"abcdefgh";
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(pattern);
        }

        // Test original compressor
        let builder = DictionaryBuilder::new();
        let dict = builder.build(&data);
        let original_compressor = DictionaryCompressor::new(dict);
        let original_compressed = original_compressor.compress(&data).unwrap();
        let original_decompressed = original_compressor
            .decompress(&original_compressed)
            .unwrap();

        // Test optimized compressor
        let optimized_compressor = OptimizedDictionaryCompressor::new(&data).unwrap();
        let optimized_compressed = optimized_compressor.compress(&data).unwrap();
        let optimized_decompressed = optimized_compressor
            .decompress(&optimized_compressed)
            .unwrap();

        // Both should decompress correctly
        assert_eq!(data, original_decompressed);
        assert_eq!(data, optimized_decompressed);

        // Optimized version should achieve similar or better compression
        let original_ratio = original_compressed.len() as f64 / data.len() as f64;
        let optimized_ratio = optimized_compressed.len() as f64 / data.len() as f64;

        println!("Original compression ratio: {:.3}", original_ratio);
        println!("Optimized compression ratio: {:.3}", optimized_ratio);

        // The optimized version should not be significantly worse
        assert!(optimized_ratio <= original_ratio + 0.1);
    }

    #[test]
    fn test_rolling_hash() {
        let mut hasher = RollingHash::new(3);

        // Test basic hashing
        let hash1 = hasher.hash_slice(b"abc");
        let hash2 = hasher.hash_slice(b"abc");
        assert_eq!(hash1, hash2);

        // Test rolling
        let _hash3 = hasher.hash_slice(b"abc");
        let hash4 = hasher.roll(b'a', b'd'); // abc -> bcd
        let expected_hash = hasher.hash_slice(b"bcd");
        assert_eq!(hash4, expected_hash);
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1000, 0.01);

        // Test basic insertion and lookup
        filter.insert(b"hello");
        assert!(filter.contains(b"hello"));

        // False negatives should not occur
        filter.insert(b"world");
        assert!(filter.contains(b"world"));

        // Test with multiple items
        let items = [b"foo", b"bar", b"baz", b"qux"];
        for item in &items {
            filter.insert(*item);
        }

        for item in &items {
            assert!(filter.contains(*item));
        }
    }

    #[test]
    fn test_optimized_compression_ratio() {
        // Test with highly compressible data
        let pattern = b"AAABBBCCCDDDEEEFFFGGGHHHIIIJJJKKKLLLMMMNNNOOOPPP";
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(pattern);
        }

        let compressor = OptimizedDictionaryCompressor::new(&data).unwrap();
        let ratio = compressor.estimate_compression_ratio(&data);

        // Should achieve good compression
        assert!(
            ratio < 0.8,
            "Compression ratio was {:.3}, expected < 0.8",
            ratio
        );
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_empty_data_handling() {
        let empty_data = b"";
        let compressor = OptimizedDictionaryCompressor::new(empty_data).unwrap();

        let compressed = compressor.compress(empty_data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(empty_data.to_vec(), decompressed);
    }

    #[test]
    fn test_single_byte_data() {
        let data = b"a";
        let compressor = OptimizedDictionaryCompressor::new(data).unwrap();

        let compressed = compressor.compress(data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data.to_vec(), decompressed);
    }

    #[test]
    fn test_no_repeated_patterns() {
        // Generate data with no repeated patterns
        let data: Vec<u8> = (0..255).collect();
        let compressor = OptimizedDictionaryCompressor::new(&data).unwrap();

        let compressed = compressor.compress(&data).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(data, decompressed);

        // Should not compress well (ratio should be close to 1.0 or higher due to overhead)
        let ratio = compressed.len() as f64 / data.len() as f64;
        assert!(ratio >= 0.9); // Little to no compression expected
    }
}
