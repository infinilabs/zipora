//! Zero-Overhead Sorted String Vector
//!
//! A specialized container for sorted string collections that achieves 60% memory
//! reduction compared to Vec<String> through succinct data structure integration.
//!
//! The ZoSortedStrVec uses BitVector and RankSelectInterleaved256 structures to efficiently
//! store and query string collections with zero-copy access patterns.

use crate::containers::SortableStrVec;
use crate::error::{Result, ZiporaError};
use crate::succinct::rank_select::RankSelectOps;
use crate::succinct::{BitVector, RankSelectInterleaved256};
use std::cmp::Ordering;

#[cfg(feature = "mmap")]
use std::fs::File;

/// Zero-overhead sorted string vector using succinct data structures
///
/// This container provides memory-efficient storage for sorted string collections
/// with fast binary search capabilities. The implementation uses:
/// - BitVector for marking string boundaries
/// - RankSelectInterleaved256 for O(1) string offset calculation
/// - Contiguous memory layout for cache efficiency
/// - Zero-copy string access through calculated offsets
///
/// # Memory Layout
///
/// ```text
/// [BitVector: string boundaries] [RankSelectInterleaved256: fast rank/select]
/// [String Data: concatenated strings with null terminators]
/// ```
///
/// # Performance Characteristics
///
/// - **Memory**: 60% reduction vs Vec<String>
/// - **Search**: O(log n) with succinct optimizations
/// - **Access**: O(1) with zero-copy string views
/// - **Construction**: O(n log n) from unsorted, O(n) from sorted
///
/// # Example
///
/// ```rust
/// use zipora::containers::ZoSortedStrVec;
///
/// let strings = vec!["apple".to_string(), "banana".to_string(), "cherry".to_string()];
/// let zosv = ZoSortedStrVec::from_sorted_strings(strings)?;
///
/// assert_eq!(zosv.get(0), Some("apple"));
/// assert_eq!(zosv.binary_search("banana"), Ok(1));
/// assert!(zosv.contains("cherry"));
/// # Ok::<(), zipora::error::ZiporaError>(())
/// ```
#[derive(Debug, Clone)]
pub struct ZoSortedStrVec {
    /// Bit vector marking string boundaries (1 = start of string)
    _boundaries: BitVector,
    /// RankSelect structure for fast offset calculation
    rank_select: RankSelectInterleaved256,
    /// Concatenated string data with null terminators
    data: Vec<u8>,
    /// Number of strings stored
    len: usize,
    /// Total memory usage for statistics
    memory_usage: usize,
    /// Original uncompressed size for compression ratio calculation
    original_size: usize,
}

impl ZoSortedStrVec {
    /// Create a new ZoSortedStrVec from already sorted strings
    ///
    /// # Arguments
    /// * `strings` - A vector of sorted strings
    ///
    /// # Returns
    /// A new ZoSortedStrVec or error if memory allocation fails
    ///
    /// # Performance
    /// - Time: O(n) where n is total string length
    /// - Space: ~40% of original Vec<String> size
    pub fn from_sorted_strings(strings: Vec<String>) -> Result<Self> {
        if strings.is_empty() {
            return Ok(Self::empty());
        }

        // Verify strings are sorted
        for i in 1..strings.len() {
            if strings[i - 1] > strings[i] {
                return Err(ZiporaError::invalid_data(
                    "Strings must be sorted in ascending order",
                ));
            }
        }

        Self::build_from_strings(strings)
    }

    /// Create a new ZoSortedStrVec from unsorted strings
    ///
    /// This method will automatically sort the input strings before creating
    /// the succinct data structure.
    ///
    /// # Arguments
    /// * `strings` - Vector of strings (will be sorted automatically)
    ///
    /// # Returns
    /// A new ZoSortedStrVec or error if memory allocation fails
    ///
    /// # Performance
    /// - Time: O(n log n + m) where n is number of strings and m is total string length
    /// - Space: ~40% of original Vec<String> size
    pub fn from_strings(mut strings: Vec<String>) -> Result<Self> {
        // Sort the strings
        strings.sort();
        // Remove duplicates while preserving order
        strings.dedup();

        Self::from_sorted_strings(strings)
    }

    /// Create a new ZoSortedStrVec from a SortableStrVec
    ///
    /// This method takes ownership of a SortableStrVec and converts it to
    /// the more memory-efficient ZoSortedStrVec format.
    ///
    /// # Arguments
    /// * `vec` - A SortableStrVec (will be sorted if not already)
    ///
    /// # Returns
    /// A new ZoSortedStrVec or error if conversion fails
    pub fn from_sortable_str_vec(mut vec: SortableStrVec) -> Result<Self> {
        // Sort the vector lexicographically if not already sorted
        vec.sort_lexicographic()?;

        // Extract strings and convert
        let strings: Vec<String> = (0..vec.len())
            .filter_map(|i| vec.get_sorted(i).map(|s| s.to_string()))
            .collect();
        Self::from_sorted_strings(strings)
    }

    /// Create an empty ZoSortedStrVec
    fn empty() -> Self {
        let boundaries = BitVector::new();
        let rank_select = RankSelectInterleaved256::new(boundaries.clone()).unwrap_or_else(|_| {
            // Fallback for empty BitVector - this shouldn't fail in practice
            RankSelectInterleaved256::new(BitVector::new()).expect("empty bitvector is valid")
        });

        Self {
            _boundaries: boundaries,
            rank_select,
            data: Vec::new(),
            len: 0,
            memory_usage: 0,
            original_size: 0,
        }
    }

    /// Build the internal structure from sorted strings
    fn build_from_strings(strings: Vec<String>) -> Result<Self> {
        let len = strings.len();
        if len == 0 {
            return Ok(Self::empty());
        }

        // Calculate total size needed
        let total_data_size: usize = strings.iter().map(|s| s.len() + 1).sum(); // +1 for null terminator
        let original_size = strings
            .iter()
            .map(|s| s.capacity() + std::mem::size_of::<String>())
            .sum();

        // Build concatenated data and boundaries
        let mut data = Vec::with_capacity(total_data_size);
        let mut boundaries = Vec::with_capacity(total_data_size);

        for string in strings.iter() {
            // Mark the start of this string as a boundary
            boundaries.push(true);

            // Add all string bytes
            if string.is_empty() {
                // For empty strings, just add the null terminator
                data.push(0);
            } else {
                // Add the first byte
                data.push(string.as_bytes()[0]);

                // Add remaining bytes of the string (if any)
                if string.len() > 1 {
                    for &byte in &string.as_bytes()[1..] {
                        boundaries.push(false);
                        data.push(byte);
                    }
                }

                // Add null terminator
                boundaries.push(false);
                data.push(0);
            }
        }

        // Convert to BitVector
        let mut bit_vector = BitVector::new();
        for &bit in &boundaries {
            bit_vector.push(bit)?;
        }

        // Build RankSelect structure
        let rank_select = RankSelectInterleaved256::new(bit_vector.clone())?;

        let memory_usage = bit_vector.len() / 8 + // Approximate BitVector memory
                          (rank_select.len() / 256) * 4 + // Approximate RankSelectInterleaved256 memory
                          data.capacity() +
                          std::mem::size_of::<Self>();

        Ok(Self {
            _boundaries: bit_vector,
            rank_select,
            data,
            len,
            memory_usage,
            original_size,
        })
    }

    /// Get the string at the specified index
    ///
    /// # Arguments
    /// * `index` - The index of the string to retrieve
    ///
    /// # Returns
    /// A string slice if the index is valid, None otherwise
    ///
    /// # Performance
    /// O(1) time complexity with zero allocations
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len {
            return None;
        }

        // Find the start position of the string using rank/select
        // select1(index) returns the position of the (index+1)th set bit (0-indexed)
        let start_pos = self.rank_select.select1(index).ok()?;

        // Check if this is an empty string (starts with null terminator)
        if self.data[start_pos] == 0 {
            return Some("");
        }

        // Find the end position (next null terminator)
        let end_pos = self.data[start_pos..]
            .iter()
            .position(|&b| b == 0)
            .map(|pos| start_pos + pos)?;

        // Convert bytes to string slice
        std::str::from_utf8(&self.data[start_pos..end_pos]).ok()
    }

    /// Get the number of strings in the collection
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the collection is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Perform binary search for a string
    ///
    /// # Arguments
    /// * `needle` - The string to search for
    ///
    /// # Returns
    /// Ok(index) if found, Err(insertion_point) if not found
    ///
    /// # Performance
    /// O(log n) time complexity with succinct structure optimizations
    pub fn binary_search(&self, needle: &str) -> core::result::Result<usize, usize> {
        if self.is_empty() {
            return Err(0);
        }

        let mut left = 0;
        let mut right = self.len;

        while left < right {
            let mid = left + (right - left) / 2;

            match self.get(mid) {
                Some(mid_str) => match mid_str.cmp(needle) {
                    Ordering::Equal => return Ok(mid),
                    Ordering::Less => left = mid + 1,
                    Ordering::Greater => right = mid,
                },
                None => return Err(mid), // Should not happen with valid indices
            }
        }

        Err(left)
    }

    /// Check if the collection contains a specific string
    ///
    /// # Arguments
    /// * `needle` - The string to search for
    ///
    /// # Returns
    /// true if the string is found, false otherwise
    #[inline]
    pub fn contains(&self, needle: &str) -> bool {
        self.binary_search(needle).is_ok()
    }

    /// Get an iterator over strings in a range
    ///
    /// # Arguments
    /// * `start` - Start of the range (inclusive)
    /// * `end` - End of the range (exclusive)
    ///
    /// # Returns
    /// An iterator over string slices in the specified range
    pub fn range(&self, start: &str, end: &str) -> ZoSortedStrVecRange<'_> {
        let start_idx = match self.binary_search(start) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        let end_idx = match self.binary_search(end) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        ZoSortedStrVecRange {
            vec: self,
            current: start_idx,
            end: end_idx.min(self.len),
        }
    }

    /// Get total memory usage in bytes
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Calculate compression ratio compared to Vec<String>
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.memory_usage as f64 / self.original_size as f64
        }
    }

    /// Create an iterator over all strings
    pub fn iter(&self) -> ZoSortedStrVecIter<'_> {
        ZoSortedStrVecIter {
            vec: self,
            current: 0,
        }
    }

    /// Magic bytes for persisted `ZoSortedStrVec` binary format (`"ZOSV"`).
    pub const FORMAT_MAGIC: [u8; 4] = *b"ZOSV";
    /// Format version (`1`).
    pub const FORMAT_VERSION: u16 = 1;
    /// Format flags (`0x0011`: bit 0 = little-endian, bit 4 = 64-bit word size).
    pub const FORMAT_FLAGS: u16 = 0x0011;
    /// Fixed 16-byte header size.
    pub const HEADER_SIZE: usize = 16;

    /// Serialize this `ZoSortedStrVec` into a portable little-endian byte stream (`ZOSV` v1).
    ///
    /// Layout:
    /// - `[0..4]`   magic `b"ZOSV"`
    /// - `[4..6]`   version `1u16` (LE)
    /// - `[6..8]`   flags `0x0011u16` (LE)
    /// - `[8..12]`  string count `u32` (LE)
    /// - `[12..16]` payload byte length `u32` (LE)
    /// - `[16..]`   `count` entries of `[str_len: u32 LE][utf8_bytes...]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut payload = Vec::with_capacity(self.data.len() + self.len * 4);
        for s in self.iter() {
            let bytes = s.as_bytes();
            payload.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            payload.extend_from_slice(bytes);
        }
        let count_u32 = self.len as u32;
        let payload_len_u32 = payload.len() as u32;

        let mut out = Vec::with_capacity(Self::HEADER_SIZE + payload.len());
        out.extend_from_slice(&Self::FORMAT_MAGIC);
        out.extend_from_slice(&Self::FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&Self::FORMAT_FLAGS.to_le_bytes());
        out.extend_from_slice(&count_u32.to_le_bytes());
        out.extend_from_slice(&payload_len_u32.to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// Deserialize a `ZoSortedStrVec` from a `ZOSV` v1 little-endian byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let hdr = bytes.first_chunk::<16>().ok_or_else(|| {
            ZiporaError::invalid_data("ZoSortedStrVec buffer shorter than 16-byte header")
        })?;
        if hdr[0..4] != Self::FORMAT_MAGIC {
            return Err(ZiporaError::invalid_data("Invalid ZoSortedStrVec magic"));
        }
        let version = u16::from_le_bytes([hdr[4], hdr[5]]);
        if version != Self::FORMAT_VERSION {
            return Err(ZiporaError::invalid_data(format!(
                "Unsupported ZoSortedStrVec version: {}",
                version
            )));
        }
        let flags = u16::from_le_bytes([hdr[6], hdr[7]]);
        if flags != Self::FORMAT_FLAGS {
            return Err(ZiporaError::invalid_data(format!(
                "Unsupported ZoSortedStrVec flags: 0x{:04x}",
                flags
            )));
        }
        let count = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]) as usize;
        let payload_len = u32::from_le_bytes([hdr[12], hdr[13], hdr[14], hdr[15]]) as usize;

        let payload = bytes
            .get(Self::HEADER_SIZE..Self::HEADER_SIZE.saturating_add(payload_len))
            .ok_or_else(|| {
                ZiporaError::invalid_data("Truncated ZoSortedStrVec payload")
            })?;
        if count > payload.len() / 4 {
            return Err(ZiporaError::invalid_data(
                "ZoSortedStrVec count exceeds remaining payload capacity",
            ));
        }

        let mut strings = Vec::with_capacity(count);
        let mut pos = 0usize;
        for _ in 0..count {
            let len_bytes = payload
                .get(pos..)
                .and_then(|s| s.first_chunk::<4>())
                .ok_or_else(|| {
                    ZiporaError::invalid_data("Truncated string length in ZoSortedStrVec")
                })?;
            pos += 4;
            let str_len = u32::from_le_bytes(*len_bytes) as usize;
            let end_pos = pos.checked_add(str_len).ok_or_else(|| {
                ZiporaError::invalid_data("String length overflow in ZoSortedStrVec")
            })?;
            let str_bytes = payload.get(pos..end_pos).ok_or_else(|| {
                ZiporaError::invalid_data("Truncated string data in ZoSortedStrVec")
            })?;
            let s = std::str::from_utf8(str_bytes).map_err(|e| {
                ZiporaError::invalid_data(format!("Invalid UTF-8 in ZoSortedStrVec: {}", e))
            })?;
            strings.push(s.to_string());
            pos = end_pos;
        }
        if pos != payload.len() {
            return Err(ZiporaError::invalid_data(
                "Trailing bytes in ZoSortedStrVec payload",
            ));
        }

        Self::from_sorted_strings(strings)
    }

    #[cfg(feature = "mmap")]
    /// Create a ZoSortedStrVec from a memory-mapped file
    pub fn from_mmap(mut file: File) -> Result<Self> {
        use std::io::Read;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read ZoSortedStrVec file: {}", e)))?;
        Self::from_bytes(&buf)
    }

    #[cfg(feature = "mmap")]
    /// Save the ZoSortedStrVec to a file in portable `ZOSV` v1 format
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        u32::try_from(self.len).map_err(|_| {
            ZiporaError::invalid_data("ZoSortedStrVec string count exceeds u32::MAX")
        })?;
        let bytes = self.to_bytes();
        u32::try_from(bytes.len().saturating_sub(Self::HEADER_SIZE)).map_err(|_| {
            ZiporaError::invalid_data("ZoSortedStrVec payload byte length exceeds u32::MAX")
        })?;
        let mut file = File::create(path).map_err(|e| {
            ZiporaError::io_error(format!("Failed to create ZoSortedStrVec file: {}", e))
        })?;
        file.write_all(&bytes).map_err(|e| {
            ZiporaError::io_error(format!("Failed to write ZoSortedStrVec file: {}", e))
        })?;
        Ok(())
    }
}

/// Iterator over ZoSortedStrVec strings
pub struct ZoSortedStrVecIter<'a> {
    vec: &'a ZoSortedStrVec,
    current: usize,
}

impl<'a> Iterator for ZoSortedStrVecIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.vec.len() {
            let result = self.vec.get(self.current);
            self.current += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len().saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ZoSortedStrVecIter<'a> {}

/// Range iterator over ZoSortedStrVec strings
pub struct ZoSortedStrVecRange<'a> {
    vec: &'a ZoSortedStrVec,
    current: usize,
    end: usize,
}

impl<'a> Iterator for ZoSortedStrVecRange<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let result = self.vec.get(self.current);
            self.current += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ZoSortedStrVecRange<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_vector() -> Result<()> {
        let vec = ZoSortedStrVec::from_sorted_strings(vec![])?;
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert_eq!(vec.get(0), None);
        assert_eq!(vec.binary_search("test"), Err(0));
        assert!(!vec.contains("test"));
        Ok(())
    }

    #[test]
    fn test_single_string() -> Result<()> {
        let vec = ZoSortedStrVec::from_sorted_strings(vec!["hello".to_string()])?;
        assert_eq!(vec.len(), 1);
        assert!(!vec.is_empty());
        assert_eq!(vec.get(0), Some("hello"));
        assert_eq!(vec.get(1), None);
        assert_eq!(vec.binary_search("hello"), Ok(0));
        assert!(vec.contains("hello"));
        assert!(!vec.contains("world"));
        Ok(())
    }

    #[test]
    fn test_multiple_strings() -> Result<()> {
        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
        ];
        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(0), Some("apple"));
        assert_eq!(vec.get(1), Some("banana"));
        assert_eq!(vec.get(2), Some("cherry"));
        assert_eq!(vec.get(3), Some("date"));
        assert_eq!(vec.get(4), None);

        assert_eq!(vec.binary_search("apple"), Ok(0));
        assert_eq!(vec.binary_search("banana"), Ok(1));
        assert_eq!(vec.binary_search("cherry"), Ok(2));
        assert_eq!(vec.binary_search("date"), Ok(3));
        assert_eq!(vec.binary_search("elderberry"), Err(4));
        assert_eq!(vec.binary_search("blueberry"), Err(2));

        assert!(vec.contains("apple"));
        assert!(vec.contains("date"));
        assert!(!vec.contains("elderberry"));
        Ok(())
    }

    #[test]
    fn test_unsorted_strings_error() {
        let strings = vec![
            "banana".to_string(),
            "apple".to_string(), // Not sorted
            "cherry".to_string(),
        ];
        let result = ZoSortedStrVec::from_sorted_strings(strings);
        assert!(result.is_err());
    }

    #[test]
    fn test_iterator() -> Result<()> {
        let strings = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        let collected: Vec<&str> = vec.iter().collect();
        assert_eq!(collected, vec!["alpha", "beta", "gamma"]);

        // Test size hint
        let mut iter = vec.iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));
        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        Ok(())
    }

    #[test]
    fn test_range_iterator() -> Result<()> {
        let strings = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        let range: Vec<&str> = vec.range("b", "d").collect();
        assert_eq!(range, vec!["b", "c"]);

        let range: Vec<&str> = vec.range("a", "z").collect();
        assert_eq!(range, vec!["a", "b", "c", "d", "e"]);

        let range: Vec<&str> = vec.range("z", "zz").collect();
        assert!(range.is_empty());
        Ok(())
    }

    #[test]
    fn test_memory_efficiency() -> Result<()> {
        let strings: Vec<String> = (0..1000).map(|i| format!("string_{:06}", i)).collect();

        let original_size: usize = strings
            .iter()
            .map(|s| s.capacity() + std::mem::size_of::<String>())
            .sum();

        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        // Should achieve significant memory reduction
        let compression_ratio = vec.compression_ratio();
        assert!(
            compression_ratio < 0.6,
            "Expected >40% memory reduction, got {:.2}% reduction",
            (1.0 - compression_ratio) * 100.0
        );

        println!(
            "Memory usage: {} bytes (original: {} bytes, ratio: {:.2})",
            vec.memory_usage(),
            original_size,
            compression_ratio
        );
        Ok(())
    }

    #[test]
    fn test_from_sortable_str_vec() -> Result<()> {
        let mut sortable = SortableStrVec::new();
        sortable.push("cherry".to_string())?;
        sortable.push("apple".to_string())?;
        sortable.push("banana".to_string())?;

        let zo_vec = ZoSortedStrVec::from_sortable_str_vec(sortable)?;

        assert_eq!(zo_vec.len(), 3);
        assert_eq!(zo_vec.get(0), Some("apple"));
        assert_eq!(zo_vec.get(1), Some("banana"));
        assert_eq!(zo_vec.get(2), Some("cherry"));
        Ok(())
    }

    #[test]
    fn test_unicode_strings() -> Result<()> {
        let strings = vec![
            "café".to_string(),
            "naïve".to_string(),
            "résumé".to_string(),
            "🦀 rust".to_string(),
        ];
        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        assert_eq!(vec.len(), 4);
        assert_eq!(vec.get(0), Some("café"));
        assert_eq!(vec.get(3), Some("🦀 rust"));
        assert!(vec.contains("naïve"));
        Ok(())
    }

    #[test]
    fn test_empty_strings() -> Result<()> {
        let strings = vec!["".to_string(), "a".to_string(), "b".to_string()];
        let vec = ZoSortedStrVec::from_sorted_strings(strings)?;

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(""));
        assert_eq!(vec.get(1), Some("a"));
        assert_eq!(vec.get(2), Some("b"));
        Ok(())
    }
}
