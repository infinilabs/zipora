//! ZReorderMap - Space-efficient blob store reordering utility
//!
//! This module provides a Run-Length Encoded (RLE) data structure for storing
//! reordering mappings, typically used to optimize blob store access patterns.
//!
//! # File Format
//!
//! The format consists of a 16-byte header followed by RLE-encoded entries:
//!
//! **Header (16 bytes):**
//! - `[0-7]`: size (u64, little-endian) - total number of elements
//! - `[8-15]`: sign (i64, little-endian) - either 1 or -1
//!
//! **Body (RLE entries):**
//! - 5 bytes: encoded_value = (base_value << 1) | has_single_flag
//!   - If LSB is 1: single value, sequence length = 1
//!   - If LSB is 0: var_uint sequence length follows
//! - 0-N bytes: optional var_uint encoding of sequence length (only if LSB == 0)
//!
//! # Examples
//!
//! ```rust
//! use zipora::blob_store::reorder_map::{ZReorderMap, ZReorderMapBuilder};
//! use std::fs;
//! use tempfile::NamedTempFile;
//!
//! # fn main() -> zipora::error::Result<()> {
//! let temp_file = NamedTempFile::new()?;
//! let path = temp_file.path();
//!
//! // Build a reorder map with ascending sequences
//! {
//!     let mut builder = ZReorderMapBuilder::new(path, 10, 1)?;
//!     for i in 0..10 {
//!         builder.push(i)?;
//!     }
//!     builder.finish()?;
//! }
//!
//! // Read back the mapping
//! let map = ZReorderMap::open(path)?;
//! let values: Vec<usize> = map.collect();
//! assert_eq!(values, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, ZiporaError};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A space-efficient iterator over reordering mappings using RLE compression.
///
/// This structure memory-maps a file containing RLE-encoded reordering data,
/// allowing efficient sequential access without loading the entire dataset into RAM.
pub struct ZReorderMap {
    /// Memory-mapped file
    _file: File,
    /// Memory-mapped data
    mmap: Mmap,
    /// File path
    _path: PathBuf,
    /// Current read position in data
    pos: usize,
    /// Current mapped value
    current_value: usize,
    /// Remaining elements in current sequence
    seq_length: usize,
    /// Total number of elements in the map
    size: usize,
    /// Current iteration index
    index: usize,
    /// Sign indicating sequence direction (1 or -1)
    sign: i64,
}

impl ZReorderMap {
    /// Opens a ZReorderMap from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the reorder map file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is too small (< 16 bytes for header)
    /// - The header contains invalid data
    /// - The first entry cannot be read
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use zipora::blob_store::reorder_map::ZReorderMap;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// let map = ZReorderMap::open("reorder.map")?;
    /// println!("Total elements: {}", map.size());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();

        // Open file in read-only mode
        let file = OpenOptions::new()
            .read(true)
            .open(&path_buf)?;

        // Memory-map the file
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .map(&file)
                .map_err(|e| ZiporaError::Io(e))?
        };

        // Validate minimum file size
        if mmap.len() < 16 {
            return Err(ZiporaError::invalid_data(format!(
                "ZReorderMap file too small: {} bytes, expected at least 16",
                mmap.len()
            )));
        }

        let mut map = Self {
            _file: file,
            mmap,
            _path: path_buf,
            pos: 0,
            current_value: 0,
            seq_length: 0,
            size: 0,
            index: 0,
            sign: 0,
        };

        map.rewind()?;
        Ok(map)
    }

    /// Checks if the iterator has reached the end.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use zipora::blob_store::reorder_map::ZReorderMap;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// let mut map = ZReorderMap::open("reorder.map")?;
    /// while !map.eof() {
    ///     if let Some(value) = map.next() {
    ///         println!("Value: {}", value);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn eof(&self) -> bool {
        self.index >= self.size
    }

    /// Returns the total number of elements in the map.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the current iteration index.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if called when at EOF.
    #[inline]
    pub fn index(&self) -> usize {
        debug_assert!(self.index < self.size, "index() called at EOF");
        self.index
    }

    /// Returns the current mapped value without advancing.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if called when at EOF.
    #[inline]
    pub fn current(&self) -> usize {
        debug_assert!(self.index < self.size, "current() called at EOF");
        self.current_value
    }

    /// Resets the iterator to the beginning.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file is too small (< 16 bytes)
    /// - The first entry cannot be read
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use zipora::blob_store::reorder_map::ZReorderMap;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// let mut map = ZReorderMap::open("reorder.map")?;
    /// // Iterate once
    /// for _ in &mut map {}
    /// // Reset and iterate again
    /// map.rewind()?;
    /// for _ in &mut map {}
    /// # Ok(())
    /// # }
    /// ```
    pub fn rewind(&mut self) -> Result<()> {
        if self.mmap.len() < 16 {
            return Err(ZiporaError::invalid_data(
                "ZReorderMap file too small for header"
            ));
        }

        // Read header: [size: u64][sign: i64]
        let mut header = [0u8; 8];

        // Read size
        header.copy_from_slice(&self.mmap[0..8]);
        self.size = u64::from_le_bytes(header) as usize;

        // SECURITY FIX (v2.1.1): Validate size to prevent DoS attacks
        // Note: ZReorderMap uses RLE compression, so actual file size can be much smaller
        // than the number of elements. We validate against an upper bound instead.
        const MAX_REASONABLE_SIZE: usize = usize::MAX / 100;
        if self.size > MAX_REASONABLE_SIZE {
            return Err(ZiporaError::invalid_data(format!(
                "ZReorderMap size {} exceeds reasonable limit {}",
                self.size, MAX_REASONABLE_SIZE
            )));
        }

        // Read sign
        header.copy_from_slice(&self.mmap[8..16]);
        self.sign = i64::from_le_bytes(header);

        // Validate sign
        if self.sign != 1 && self.sign != -1 {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid sign value: {}, expected 1 or -1",
                self.sign
            )));
        }

        // Reset position
        self.pos = 16;
        self.index = 0;
        self.seq_length = 0;

        // Read first entry if not empty
        if self.size > 0 {
            self.read_entry()?;
        }

        Ok(())
    }

    /// Reads the next RLE entry from the file.
    ///
    /// Updates `current_value` and `seq_length` with the next sequence.
    fn read_entry(&mut self) -> Result<()> {
        // Read 5 bytes for encoded value
        if self.pos + 5 > self.mmap.len() {
            return Err(ZiporaError::invalid_data(
                "ZReorderMap: read value out of range"
            ));
        }

        // Read 5 bytes into a u64 (little-endian)
        let mut encoded = [0u8; 8];
        encoded[..5].copy_from_slice(&self.mmap[self.pos..self.pos + 5]);
        let encoded_value = u64::from_le_bytes(encoded) as usize;
        self.pos += 5;

        // Check LSB for single vs sequence
        if encoded_value & 1 != 0 {
            // Single value: LSB is 1
            self.seq_length = 1;
            // Extract actual value (shift right to remove encoding bit)
            self.current_value = encoded_value >> 1;
            return Ok(());
        }

        // Extract actual value (shift right to remove encoding bit)
        self.current_value = encoded_value >> 1;

        // Sequence: read var_uint for length
        self.seq_length = self.read_var_uint()?;

        // Validate position after var_uint read
        if self.pos > self.mmap.len() {
            return Err(ZiporaError::invalid_data(
                "ZReorderMap: read seq out of range"
            ));
        }

        Ok(())
    }

    /// Reads a variable-length unsigned integer (LEB128 encoding).
    ///
    /// Updates `self.pos` to point after the var_uint.
    fn read_var_uint(&mut self) -> Result<usize> {
        let mut result: usize = 0;
        let mut shift = 0;

        loop {
            if self.pos >= self.mmap.len() {
                return Err(ZiporaError::invalid_data(
                    "ZReorderMap: var_uint extends beyond file"
                ));
            }

            let byte = self.mmap[self.pos];
            self.pos += 1;

            // Check for overflow before shifting
            if shift >= 64 {
                return Err(ZiporaError::invalid_data(
                    "ZReorderMap: var_uint overflow"
                ));
            }

            // Add the lower 7 bits
            result |= ((byte & 0x7F) as usize) << shift;

            // If high bit is not set, we're done
            if byte & 0x80 == 0 {
                break;
            }

            shift += 7;
        }

        Ok(result)
    }
}

impl Iterator for ZReorderMap {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof() {
            return None;
        }

        debug_assert!(self.seq_length > 0, "seq_length must be > 0");

        // Get current value
        let value = self.current_value;

        // Advance to next value
        self.index += 1;

        // Update current value based on sign
        if self.sign > 0 {
            self.current_value = self.current_value.wrapping_add(self.sign as usize);
        } else {
            // For negative sign, we need signed arithmetic
            self.current_value = (self.current_value as i64 + self.sign) as usize;
        }

        // Decrement sequence counter
        self.seq_length -= 1;

        // Read next entry if sequence exhausted and more elements remain
        if self.seq_length == 0 && !self.eof() {
            if let Err(_) = self.read_entry() {
                // Error reading next entry - mark as EOF
                self.index = self.size;
                return Some(value);
            }
        }

        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.size.saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ZReorderMap {
    fn len(&self) -> usize {
        self.size.saturating_sub(self.index)
    }
}

/// Builder for creating ZReorderMap files.
///
/// This builder accepts values in sequence and automatically detects and compresses
/// consecutive sequences using RLE encoding.
///
/// # Examples
///
/// ```rust
/// use zipora::blob_store::reorder_map::ZReorderMapBuilder;
/// use tempfile::NamedTempFile;
///
/// # fn main() -> zipora::error::Result<()> {
/// let temp_file = NamedTempFile::new()?;
/// let path = temp_file.path();
///
/// let mut builder = ZReorderMapBuilder::new(path, 6, 1)?;
/// builder.push(100)?;
/// builder.push(101)?;
/// builder.push(102)?;
/// builder.push(200)?;
/// builder.push(300)?;
/// builder.push(301)?;
/// builder.finish()?;
/// # Ok(())
/// # }
/// ```
pub struct ZReorderMapBuilder {
    /// Output file
    file: File,
    /// Base value of current sequence
    base_value: usize,
    /// Length of current sequence
    seq_length: usize,
    /// Sign indicating sequence direction (1 or -1)
    sign: i64,
    /// Remaining elements to be pushed
    remaining_size: usize,
    /// Write buffer for batching writes
    buffer: Vec<u8>,
}

impl ZReorderMapBuilder {
    /// Creates a new builder for a reorder map file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the file will be created
    /// * `size` - Total number of elements that will be pushed
    /// * `sign` - Direction of sequences: 1 for ascending, -1 for descending
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created
    /// - `sign` is not 1 or -1
    /// - Writing the header fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use zipora::blob_store::reorder_map::ZReorderMapBuilder;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// // Create builder for 100 elements with ascending sequences
    /// let mut builder = ZReorderMapBuilder::new("reorder.map", 100, 1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new<P: AsRef<Path>>(path: P, size: usize, sign: i64) -> Result<Self> {
        // Validate sign
        if sign != 1 && sign != -1 {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid sign value: {}, expected 1 or -1",
                sign
            )));
        }

        // Create file
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Write header: [size: u64][sign: i64]
        file.write_all(&size.to_le_bytes())?;
        file.write_all(&sign.to_le_bytes())?;

        Ok(Self {
            file,
            base_value: usize::MAX,  // Invalid initial value
            seq_length: 0,
            sign,
            remaining_size: size,
            buffer: Vec::with_capacity(4096),
        })
    }

    /// Pushes the next value in the reorder mapping.
    ///
    /// Values should be pushed in the order they appear in the mapping.
    /// The builder automatically detects and compresses consecutive sequences.
    ///
    /// # Arguments
    ///
    /// * `value` - The next mapping value
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `value` exceeds 40-bit limit (0x7FFFFFFFFF)
    /// - All expected elements have already been pushed
    /// - Writing to file fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::blob_store::reorder_map::ZReorderMapBuilder;
    /// use tempfile::NamedTempFile;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// let temp_file = NamedTempFile::new()?;
    /// let mut builder = ZReorderMapBuilder::new(temp_file.path(), 5, 1)?;
    /// builder.push(10)?;
    /// builder.push(11)?;
    /// builder.push(12)?;
    /// builder.push(20)?;
    /// builder.push(21)?;
    /// builder.finish()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn push(&mut self, value: usize) -> Result<()> {
        // Validate remaining size
        if self.remaining_size == 0 {
            return Err(ZiporaError::invalid_data(
                "Cannot push more values: size limit reached"
            ));
        }

        // Validate value fits in 40 bits
        if value > 0x7FFFFFFFFF {
            return Err(ZiporaError::invalid_data(format!(
                "Value {} exceeds 40-bit limit (0x7FFFFFFFFF)",
                value
            )));
        }

        // Calculate expected next value in sequence
        let next_expected = if self.seq_length == 0 {
            // First value - always starts new sequence
            usize::MAX  // Will never match
        } else {
            // Calculate using signed arithmetic
            (self.base_value as i64 + self.seq_length as i64 * self.sign) as usize
        };

        if value != next_expected {
            // Sequence broken - write current sequence if any
            self.write_sequence()?;

            // Start new sequence
            self.base_value = value;
            self.seq_length = 1;
        } else {
            // Continue current sequence
            self.seq_length += 1;
        }

        self.remaining_size -= 1;
        Ok(())
    }

    /// Finishes building the reorder map and flushes all data to disk.
    ///
    /// This method MUST be called after all values have been pushed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not all expected values have been pushed
    /// - Writing or flushing fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::blob_store::reorder_map::ZReorderMapBuilder;
    /// use tempfile::NamedTempFile;
    ///
    /// # fn main() -> zipora::error::Result<()> {
    /// let temp_file = NamedTempFile::new()?;
    /// let mut builder = ZReorderMapBuilder::new(temp_file.path(), 3, 1)?;
    /// builder.push(0)?;
    /// builder.push(1)?;
    /// builder.push(2)?;
    /// builder.finish()?;  // Must call finish!
    /// # Ok(())
    /// # }
    /// ```
    pub fn finish(mut self) -> Result<()> {
        // Validate all elements pushed
        if self.remaining_size != 0 {
            return Err(ZiporaError::invalid_data(format!(
                "finish() called with {} elements remaining",
                self.remaining_size
            )));
        }

        // Write final sequence
        self.write_sequence()?;

        // Flush buffer to file
        self.file.write_all(&self.buffer)?;
        self.buffer.clear();

        // Sync to disk
        self.file.flush()?;
        self.file.sync_all()?;

        Ok(())
    }

    /// Writes the current sequence to the buffer.
    ///
    /// Automatically flushes buffer if it grows too large.
    fn write_sequence(&mut self) -> Result<()> {
        if self.seq_length == 0 {
            return Ok(());
        }

        if self.seq_length == 1 {
            // Single value: encode with LSB = 1
            let encoded = (self.base_value << 1) | 1;
            let bytes = encoded.to_le_bytes();
            self.buffer.extend_from_slice(&bytes[..5]);
        } else {
            // Sequence: encode with LSB = 0, then var_uint length
            let encoded = self.base_value << 1;
            let bytes = encoded.to_le_bytes();
            self.buffer.extend_from_slice(&bytes[..5]);

            // Write var_uint
            self.write_var_uint(self.seq_length)?;
        }

        // Flush buffer if it's getting large
        if self.buffer.len() >= 4096 {
            self.file.write_all(&self.buffer)?;
            self.buffer.clear();
        }

        Ok(())
    }

    /// Writes a variable-length unsigned integer (LEB128 encoding).
    fn write_var_uint(&mut self, mut value: usize) -> Result<()> {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;

            if value != 0 {
                byte |= 0x80;  // Set continuation bit
            }

            self.buffer.push(byte);

            if value == 0 {
                break;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ascending_sequence() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Build map with single ascending sequence
        {
            let mut builder = ZReorderMapBuilder::new(path, 100, 1)?;
            for i in 0..100 {
                builder.push(i)?;
            }
            builder.finish()?;
        }

        // Read back
        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), 100);

        let values: Vec<usize> = map.collect();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(values, expected);

        Ok(())
    }

    #[test]
    fn test_descending_sequence() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Build map with descending sequence
        {
            let mut builder = ZReorderMapBuilder::new(path, 100, -1)?;
            for i in (0..100).rev() {
                builder.push(i)?;
            }
            builder.finish()?;
        }

        // Read back
        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), 100);

        let values: Vec<usize> = map.collect();
        let expected: Vec<usize> = (0..100).rev().collect();
        assert_eq!(values, expected);

        Ok(())
    }

    #[test]
    fn test_mixed_sequences() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Build map with multiple sequences
        let input = vec![100, 101, 102, 200, 300, 301];
        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        // Read back
        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), input.len());

        let values: Vec<usize> = map.collect();
        assert_eq!(values, input);

        Ok(())
    }

    #[test]
    fn test_single_values() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Build map with no sequences (all single values)
        let input = vec![5, 10, 15, 20, 25];
        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        // Read back
        let map = ZReorderMap::open(path)?;
        let values: Vec<usize> = map.collect();
        assert_eq!(values, input);

        Ok(())
    }

    #[test]
    fn test_large_dataset() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let size = 1_000_000;

        // Build map with large dataset (good compression)
        {
            let mut builder = ZReorderMapBuilder::new(path, size, 1)?;
            for i in 0..size {
                builder.push(i)?;
            }
            builder.finish()?;
        }

        // Read back
        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), size);

        // Verify first and last values
        let values: Vec<usize> = map.take(5).collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_rewind() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let input = vec![10, 11, 12, 20, 21];
        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        // Read multiple times with rewind
        let mut map = ZReorderMap::open(path)?;

        for _ in 0..3 {
            let values: Vec<usize> = map.by_ref().collect();
            assert_eq!(values, input);

            assert!(map.eof());
            map.rewind()?;
            assert!(!map.eof());
        }

        Ok(())
    }

    #[test]
    fn test_bounds_checking() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Test: value exceeds 40-bit limit
        {
            let mut builder = ZReorderMapBuilder::new(path, 1, 1)?;
            let result = builder.push(0x8000000000);  // > 40 bits
            assert!(result.is_err());
        }

        // Test: push more than size
        {
            let mut builder = ZReorderMapBuilder::new(path, 2, 1)?;
            builder.push(1)?;
            builder.push(2)?;
            let result = builder.push(3);
            assert!(result.is_err());
        }

        // Test: finish without all elements
        {
            let builder = ZReorderMapBuilder::new(path, 5, 1)?;
            let result = builder.finish();
            assert!(result.is_err());
        }

        Ok(())
    }

    #[test]
    fn test_invalid_sign() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let result = ZReorderMapBuilder::new(path, 10, 0);
        assert!(result.is_err());

        let result = ZReorderMapBuilder::new(path, 10, 2);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_empty_map() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        {
            let builder = ZReorderMapBuilder::new(path, 0, 1)?;
            builder.finish()?;
        }

        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), 0);
        assert!(map.eof());

        let values: Vec<usize> = map.collect();
        assert!(values.is_empty());

        Ok(())
    }

    #[test]
    fn test_iterator_traits() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let input = vec![0, 1, 2, 3, 4];
        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        let map = ZReorderMap::open(path)?;

        // Test size_hint
        let (lower, upper) = map.size_hint();
        assert_eq!(lower, input.len());
        assert_eq!(upper, Some(input.len()));

        // Test ExactSizeIterator
        assert_eq!(map.len(), input.len());

        Ok(())
    }

    #[test]
    fn test_current_and_index() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        let input = vec![100, 101, 102];
        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        let mut map = ZReorderMap::open(path)?;

        assert_eq!(map.current(), 100);
        assert_eq!(map.index(), 0);

        map.next();
        assert_eq!(map.current(), 101);
        assert_eq!(map.index(), 1);

        map.next();
        assert_eq!(map.current(), 102);
        assert_eq!(map.index(), 2);

        Ok(())
    }

    #[test]
    fn test_var_uint_encoding() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Create a sequence that will require var_uint with multiple bytes
        let mut input = vec![0];
        // Add large gap that creates long sequence
        for i in 1..1000 {
            input.push(i);
        }

        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        let map = ZReorderMap::open(path)?;
        let values: Vec<usize> = map.collect();
        assert_eq!(values, input);

        Ok(())
    }

    #[test]
    fn test_compression_efficiency() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Single long sequence should compress very well
        let size = 10000;
        {
            let mut builder = ZReorderMapBuilder::new(path, size, 1)?;
            for i in 0..size {
                builder.push(i)?;
            }
            builder.finish()?;
        }

        // Check file size
        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len();

        // Header (16) + 5 bytes value + var_uint for 10000 (~2 bytes) = ~23 bytes
        // Should be much smaller than 10000 * 8 bytes uncompressed
        assert!(file_size < 100, "Expected < 100 bytes, got {}", file_size);

        // Verify correctness
        let map = ZReorderMap::open(path)?;
        assert_eq!(map.size(), size);

        Ok(())
    }

    #[test]
    fn test_alternating_sequences() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Alternating short sequences
        let input = vec![
            0, 1, 2,       // Sequence 1
            10, 11,        // Sequence 2
            20, 21, 22, 23, // Sequence 3
            100,           // Single value
            200, 201,      // Sequence 4
        ];

        {
            let mut builder = ZReorderMapBuilder::new(path, input.len(), 1)?;
            for &val in &input {
                builder.push(val)?;
            }
            builder.finish()?;
        }

        let map = ZReorderMap::open(path)?;
        let values: Vec<usize> = map.collect();
        assert_eq!(values, input);

        Ok(())
    }

    #[test]
    fn test_maximum_value() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path();

        // Test maximum valid value (40-bit max)
        let max_val = 0x7FFFFFFFFF;
        {
            let mut builder = ZReorderMapBuilder::new(path, 1, 1)?;
            builder.push(max_val)?;
            builder.finish()?;
        }

        let map = ZReorderMap::open(path)?;
        let values: Vec<usize> = map.collect();
        assert_eq!(values, vec![max_val]);

        Ok(())
    }
}
