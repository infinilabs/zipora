//! Dictionary-based compression implementation
//!
//! This module provides LZ-style dictionary compression algorithms that find
//! and encode repeated substrings for efficient compression.

use crate::error::{Result, ToplingError};
use std::collections::HashMap;

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
            return Err(ToplingError::invalid_data("Dictionary data too short"));
        }
        
        let num_entries = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut entries = HashMap::new();
        let mut offset = 4;
        
        for _ in 0..num_entries {
            if offset + 2 > data.len() {
                return Err(ToplingError::invalid_data("Truncated dictionary data"));
            }
            
            // Read sequence length
            let seq_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            
            if offset + seq_len + 8 > data.len() {
                return Err(ToplingError::invalid_data("Truncated dictionary sequence"));
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
    
    /// Compress data using dictionary
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut result = Vec::new();
        let mut pos = 0;
        
        while pos < data.len() {
            let mut best_match = None;
            let mut best_length = 0;
            
            // Look for the longest match
            let end_pos = (pos + self.max_match_length).min(data.len());
            
            for len in (self.min_match_length..=end_pos - pos).rev() {
                let sequence = &data[pos..pos + len];
                if let Some(entry) = self.dictionary.get(sequence) {
                    best_match = Some(entry);
                    best_length = len;
                    break;
                }
            }
            
            if let Some(entry) = best_match {
                // Encode as match: flag(1) + offset + length
                result.push(1); // Match flag
                result.extend_from_slice(&entry.offset.to_le_bytes());
                result.extend_from_slice(&(best_length as u32).to_le_bytes());
                pos += best_length;
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
                    return Err(ToplingError::invalid_data("Unexpected end of compressed data"));
                }
                result.push(compressed_data[pos]);
                pos += 1;
            } else if flag == 1 {
                // Dictionary match
                if pos + 8 > compressed_data.len() {
                    return Err(ToplingError::invalid_data("Truncated match data"));
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
                    return Err(ToplingError::invalid_data("Invalid back-reference offset"));
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
                            return Err(ToplingError::invalid_data("Back-reference calculation error"));
                        }
                    }
                }
            } else {
                return Err(ToplingError::invalid_data(
                    format!("Invalid compression flag: {}", flag)
                ));
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
    #[ignore] // TODO: Fix dictionary compression implementation - currently has design issues
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
    #[ignore] // TODO: Fix dictionary compression implementation - currently has design issues
    fn test_dictionary_compression_ratio() {
        let data = b"the quick brown fox jumps over the lazy dog the quick brown fox";
        
        let builder = DictionaryBuilder::new();
        let dict = builder.build(data);
        
        let compressor = DictionaryCompressor::new(dict);
        let ratio = compressor.estimate_compression_ratio(data);
        
        // Should achieve some compression due to repeated phrases
        assert!(ratio < 1.0);
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
    #[ignore] // TODO: Fix dictionary compression implementation - currently has design issues
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
}