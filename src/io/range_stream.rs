//! Range-based stream operations for partial file access
//!
//! This module provides streaming utilities for accessing specific byte ranges
//! within larger streams or files. Useful for partial downloads, parallel processing,
//! and memory-efficient handling of large files.

use std::cmp;
use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::error::{Result, ZiporaError};
use crate::io::DataInput;

/// A reader that limits access to a specific byte range within an underlying stream
pub struct RangeReader<R> {
    inner: R,
    start_pos: u64,
    current_pos: u64,
    end_pos: u64,
    total_size: Option<u64>,
}

impl<R> RangeReader<R> {
    /// Create a new range reader for the specified byte range
    ///
    /// # Arguments
    /// * `inner` - The underlying reader
    /// * `start` - Starting byte position (inclusive)
    /// * `length` - Number of bytes to read from start position
    ///
    /// # Example
    /// ```rust
    /// use std::io::Cursor;
    /// use zipora::io::RangeReader;
    ///
    /// let data = b"Hello, World!";
    /// let cursor = Cursor::new(data);
    /// let mut range_reader = RangeReader::new(cursor, 7, 5); // "World"
    /// ```
    pub fn new(inner: R, start: u64, length: u64) -> Self {
        Self {
            inner,
            start_pos: start,
            current_pos: start,
            end_pos: start.saturating_add(length),
            total_size: None,
        }
    }

    /// Create a new range reader with an end position
    ///
    /// # Arguments
    /// * `inner` - The underlying reader
    /// * `start` - Starting byte position (inclusive)
    /// * `end` - Ending byte position (exclusive)
    pub fn with_range(inner: R, start: u64, end: u64) -> Self {
        Self {
            inner,
            start_pos: start,
            current_pos: start,
            end_pos: end,
            total_size: None,
        }
    }

    /// Get the underlying reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the underlying reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consume this range reader and return the underlying reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get the starting position of this range
    pub fn start_position(&self) -> u64 {
        self.start_pos
    }

    /// Get the ending position of this range
    pub fn end_position(&self) -> u64 {
        self.end_pos
    }

    /// Get the current position within the range
    pub fn current_position(&self) -> u64 {
        self.current_pos
    }

    /// Get the total length of this range
    pub fn range_length(&self) -> u64 {
        self.end_pos.saturating_sub(self.start_pos)
    }

    /// Get the number of bytes remaining to read
    pub fn remaining(&self) -> u64 {
        self.end_pos.saturating_sub(self.current_pos)
    }

    /// Check if we've reached the end of the range
    pub fn is_at_end(&self) -> bool {
        self.current_pos >= self.end_pos
    }

    /// Set the total size of the underlying stream (for validation)
    pub fn set_total_size(&mut self, size: u64) {
        self.total_size = Some(size);
        // Adjust end position if it exceeds total size
        if self.end_pos > size {
            self.end_pos = size;
        }
    }

    /// Get progress through the range as a percentage (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        let range_len = self.range_length();
        if range_len == 0 {
            1.0
        } else {
            let read_bytes = self.current_pos.saturating_sub(self.start_pos);
            read_bytes as f64 / range_len as f64
        }
    }
}

impl<R: Read + Seek> RangeReader<R> {
    /// Create a new range reader and seek to the start position
    pub fn new_and_seek(mut inner: R, start: u64, length: u64) -> Result<Self> {
        inner.seek(SeekFrom::Start(start))
            .map_err(|e| ZiporaError::io_error(format!("Failed to seek to start position {}: {}", start, e)))?;
        
        Ok(Self::new(inner, start, length))
    }

    /// Reset to the beginning of the range
    pub fn reset(&mut self) -> Result<()> {
        self.inner.seek(SeekFrom::Start(self.start_pos))
            .map_err(|e| ZiporaError::io_error(format!("Failed to reset to start position: {}", e)))?;
        self.current_pos = self.start_pos;
        Ok(())
    }

    /// Seek within the range (relative to range start)
    pub fn seek_in_range(&mut self, pos: u64) -> Result<u64> {
        let absolute_pos = self.start_pos.saturating_add(pos);
        if absolute_pos >= self.end_pos {
            return Err(ZiporaError::invalid_data(
                format!("Seek position {} is beyond range end {}", absolute_pos, self.end_pos)
            ));
        }

        self.inner.seek(SeekFrom::Start(absolute_pos))
            .map_err(|e| ZiporaError::io_error(format!("Failed to seek within range: {}", e)))?;
        self.current_pos = absolute_pos;
        Ok(pos)
    }
}

impl<R: Read> Read for RangeReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.is_at_end() {
            return Ok(0);
        }

        let remaining = self.remaining() as usize;
        let to_read = cmp::min(buf.len(), remaining);
        
        let bytes_read = self.inner.read(&mut buf[..to_read])?;
        self.current_pos += bytes_read as u64;
        
        Ok(bytes_read)
    }
}

impl<R: Read + Seek> Seek for RangeReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target_pos = match pos {
            SeekFrom::Start(pos) => self.start_pos.saturating_add(pos),
            SeekFrom::End(offset) => {
                if offset >= 0 {
                    self.end_pos.saturating_add(offset as u64)
                } else {
                    self.end_pos.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.current_pos.saturating_add(offset as u64)
                } else {
                    self.current_pos.saturating_sub((-offset) as u64)
                }
            }
        };

        // Clamp to range bounds
        let clamped_pos = cmp::max(self.start_pos, cmp::min(target_pos, self.end_pos));
        
        self.inner.seek(SeekFrom::Start(clamped_pos))?;
        self.current_pos = clamped_pos;
        
        Ok(clamped_pos - self.start_pos)
    }
}

/// A writer that limits writes to a specific byte range
pub struct RangeWriter<W> {
    inner: W,
    start_pos: u64,
    current_pos: u64,
    end_pos: u64,
    bytes_written: u64,
}

impl<W> RangeWriter<W> {
    /// Create a new range writer for the specified byte range
    pub fn new(inner: W, start: u64, length: u64) -> Self {
        Self {
            inner,
            start_pos: start,
            current_pos: start,
            end_pos: start.saturating_add(length),
            bytes_written: 0,
        }
    }

    /// Create a new range writer with an end position
    pub fn with_range(inner: W, start: u64, end: u64) -> Self {
        Self {
            inner,
            start_pos: start,
            current_pos: start,
            end_pos: end,
            bytes_written: 0,
        }
    }

    /// Get the underlying writer
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the underlying writer
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Consume this range writer and return the underlying writer
    pub fn into_inner(self) -> W {
        self.inner
    }

    /// Get the starting position of this range
    pub fn start_position(&self) -> u64 {
        self.start_pos
    }

    /// Get the ending position of this range
    pub fn end_position(&self) -> u64 {
        self.end_pos
    }

    /// Get the current position within the range
    pub fn current_position(&self) -> u64 {
        self.current_pos
    }

    /// Get the total length of this range
    pub fn range_length(&self) -> u64 {
        self.end_pos.saturating_sub(self.start_pos)
    }

    /// Get the number of bytes remaining to write
    pub fn remaining(&self) -> u64 {
        self.end_pos.saturating_sub(self.current_pos)
    }

    /// Get the number of bytes written so far
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Check if we've reached the end of the range
    pub fn is_at_end(&self) -> bool {
        self.current_pos >= self.end_pos
    }
}

impl<W: Write + Seek> RangeWriter<W> {
    /// Create a new range writer and seek to the start position
    pub fn new_and_seek(mut inner: W, start: u64, length: u64) -> Result<Self> {
        inner.seek(SeekFrom::Start(start))
            .map_err(|e| ZiporaError::io_error(format!("Failed to seek to start position {}: {}", start, e)))?;
        
        Ok(Self::new(inner, start, length))
    }
}

impl<W: Write> Write for RangeWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.is_at_end() {
            return Ok(0);
        }

        let remaining = self.remaining() as usize;
        let to_write = cmp::min(buf.len(), remaining);
        
        let bytes_written = self.inner.write(&buf[..to_write])?;
        self.current_pos += bytes_written as u64;
        self.bytes_written += bytes_written as u64;
        
        Ok(bytes_written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: Write + Seek> Seek for RangeWriter<W> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let target_pos = match pos {
            SeekFrom::Start(pos) => self.start_pos.saturating_add(pos),
            SeekFrom::End(offset) => {
                if offset >= 0 {
                    self.end_pos.saturating_add(offset as u64)
                } else {
                    self.end_pos.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::Current(offset) => {
                if offset >= 0 {
                    self.current_pos.saturating_add(offset as u64)
                } else {
                    self.current_pos.saturating_sub((-offset) as u64)
                }
            }
        };

        // Clamp to range bounds
        let clamped_pos = cmp::max(self.start_pos, cmp::min(target_pos, self.end_pos));
        
        self.inner.seek(SeekFrom::Start(clamped_pos))?;
        self.current_pos = clamped_pos;
        
        Ok(clamped_pos - self.start_pos)
    }
}

/// Implementation of DataInput for RangeReader
impl<R: Read> DataInput for RangeReader<R> {
    fn read_u8(&mut self) -> Result<u8> {
        if self.remaining() < 1 {
            return Err(ZiporaError::io_error("Range exhausted"));
        }
        
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u8: {}", e)))?;
        Ok(buf[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        if self.remaining() < 2 {
            return Err(ZiporaError::io_error("Range exhausted"));
        }
        
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u16: {}", e)))?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.remaining() < 4 {
            return Err(ZiporaError::io_error("Range exhausted"));
        }
        
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u32: {}", e)))?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.remaining() < 8 {
            return Err(ZiporaError::io_error("Range exhausted"));
        }
        
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u64: {}", e)))?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_var_int(&mut self) -> Result<u64> {
        crate::io::var_int::VarInt::read_from(self)
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.remaining() < buf.len() as u64 {
            return Err(ZiporaError::io_error("Range exhausted"));
        }
        
        self.read_exact(buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read bytes: {}", e)))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if self.remaining() < n as u64 {
            return Err(ZiporaError::io_error("Cannot skip past range end"));
        }
        
        let mut buf = vec![0u8; n.min(8192)];
        let mut remaining = n;
        
        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            self.read_exact(&mut buf[..to_read])
                .map_err(|e| ZiporaError::io_error(format!("Failed to skip bytes: {}", e)))?;
            remaining -= to_read;
        }
        
        Ok(())
    }

    fn position(&self) -> Option<u64> {
        Some(self.current_pos.saturating_sub(self.start_pos))
    }

    fn has_remaining(&self) -> Option<bool> {
        Some(!self.is_at_end())
    }
}

/// Multi-range reader for handling multiple discontinuous ranges
pub struct MultiRangeReader<R> {
    inner: R,
    ranges: Vec<(u64, u64)>, // (start, end) pairs
    current_range: usize,
    current_pos: u64,
}

impl<R> MultiRangeReader<R> {
    /// Create a new multi-range reader
    pub fn new(inner: R, ranges: Vec<(u64, u64)>) -> Self {
        Self {
            inner,
            ranges,
            current_range: 0,
            current_pos: 0,
        }
    }

    /// Add a new range to read
    pub fn add_range(&mut self, start: u64, end: u64) {
        self.ranges.push((start, end));
    }

    /// Get the total number of bytes across all ranges
    pub fn total_length(&self) -> u64 {
        self.ranges.iter()
            .map(|(start, end)| end.saturating_sub(*start))
            .sum()
    }

    /// Get the current range being read
    pub fn current_range(&self) -> Option<(u64, u64)> {
        self.ranges.get(self.current_range).copied()
    }

    /// Move to the next range
    pub fn next_range(&mut self) -> bool {
        if self.current_range + 1 < self.ranges.len() {
            self.current_range += 1;
            self.current_pos = 0;
            true
        } else {
            false
        }
    }
}

impl<R: Read + Seek> Read for MultiRangeReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.current_range >= self.ranges.len() {
            return Ok(0); // No more ranges
        }

        let (start, end) = self.ranges[self.current_range];
        let absolute_pos = start + self.current_pos;
        
        if absolute_pos >= end {
            // Current range exhausted, move to next
            if !self.next_range() {
                return Ok(0);
            }
            return self.read(buf);
        }

        // Seek to current position in range
        self.inner.seek(SeekFrom::Start(absolute_pos))?;
        
        // Read limited by range end
        let remaining_in_range = (end - absolute_pos) as usize;
        let to_read = cmp::min(buf.len(), remaining_in_range);
        
        let bytes_read = self.inner.read(&mut buf[..to_read])?;
        self.current_pos += bytes_read as u64;
        
        Ok(bytes_read)
    }
}

/// Convenience functions for creating range readers and writers
pub mod range {
    use super::*;
    
    /// Create a range reader from any Read + Seek type
    pub fn reader<R: Read + Seek>(inner: R, start: u64, length: u64) -> Result<RangeReader<R>> {
        RangeReader::new_and_seek(inner, start, length)
    }
    
    /// Create a range writer from any Write + Seek type
    pub fn writer<W: Write + Seek>(inner: W, start: u64, length: u64) -> Result<RangeWriter<W>> {
        RangeWriter::new_and_seek(inner, start, length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_range_reader_basic() {
        let data = b"Hello, World! This is a test.";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new_and_seek(cursor, 7, 5).unwrap(); // "World"

        let mut buf = String::new();
        reader.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "World");
    }

    #[test]
    fn test_range_reader_with_seek() {
        let data = b"Hello, World! This is a test.";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new_and_seek(cursor, 7, 5).unwrap(); // "World"

        let mut buf = String::new();
        reader.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "World");
    }

    #[test]
    fn test_range_reader_position_tracking() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new(cursor, 0, 13);

        assert_eq!(reader.current_position(), 0);
        assert_eq!(reader.remaining(), 13);
        assert_eq!(reader.range_length(), 13);
        assert!(!reader.is_at_end());

        let mut buf = [0u8; 5];
        reader.read(&mut buf).unwrap();
        assert_eq!(reader.current_position(), 5);
        assert_eq!(reader.remaining(), 8);
    }

    #[test]
    fn test_range_reader_progress() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new(cursor, 0, 10);

        assert_eq!(reader.progress(), 0.0);

        let mut buf = [0u8; 5];
        reader.read(&mut buf).unwrap();
        assert_eq!(reader.progress(), 0.5);

        reader.read(&mut buf).unwrap();
        assert_eq!(reader.progress(), 1.0);
    }

    #[test]
    fn test_range_reader_bounds() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new_and_seek(cursor, 7, 5).unwrap(); // "World"

        // Read exactly the range
        let mut buf = String::new();
        reader.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "World");

        // Further reads should return 0
        let mut extra_buf = [0u8; 10];
        let bytes_read = reader.read(&mut extra_buf).unwrap();
        assert_eq!(bytes_read, 0);
    }

    #[test]
    fn test_range_reader_seek_in_range() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new_and_seek(cursor, 0, 13).unwrap();

        // Seek to position 7 (start of "World")
        reader.seek_in_range(7).unwrap();
        
        let mut buf = [0u8; 5];
        reader.read(&mut buf).unwrap();
        assert_eq!(&buf, b"World");
    }

    #[test]
    fn test_range_reader_data_input() {
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let cursor = Cursor::new(&data[..]);
        let mut reader = RangeReader::new_and_seek(cursor, 2, 4).unwrap(); // bytes 3,4,5,6 (2,3,4,5 in data)

        assert_eq!(reader.read_u8().unwrap(), 3);
        assert_eq!(reader.read_u8().unwrap(), 4);
        assert_eq!(reader.read_u16().unwrap(), u16::from_le_bytes([5, 6]));
        assert!(reader.read_u8().is_err()); // Should fail - range exhausted
    }

    #[test]
    fn test_range_writer_basic() {
        let mut buffer = vec![0u8; 20];
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = RangeWriter::new_and_seek(cursor, 5, 5).unwrap();

            writer.write_all(b"Hello").unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(&buffer[5..10], b"Hello");
        assert_eq!(&buffer[..5], &[0u8; 5]); // Unchanged
        assert_eq!(&buffer[10..], &[0u8; 10]); // Unchanged
    }

    #[test]
    fn test_range_writer_bounds() {
        let mut buffer = vec![0u8; 10];
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = RangeWriter::new_and_seek(cursor, 5, 3).unwrap();

            // Try to write more than the range allows
            let bytes_written = writer.write(b"Hello").unwrap();
            assert_eq!(bytes_written, 3); // Only 3 bytes should be written
            writer.flush().unwrap();
        }

        assert_eq!(&buffer[5..8], b"Hel");
        assert_eq!(&buffer[8..], &[0u8; 2]); // Unchanged
    }

    #[test]
    fn test_multi_range_reader() {
        let data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let cursor = Cursor::new(data);
        
        // Read ranges: A-C (0-3), G-I (6-9), M-O (12-15)
        let ranges = vec![(0, 3), (6, 9), (12, 15)];
        let mut reader = MultiRangeReader::new(cursor, ranges);

        let mut result = String::new();
        reader.read_to_string(&mut result).unwrap();
        assert_eq!(result, "ABCGHIMNO");
    }

    #[test]
    fn test_range_convenience_functions() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = range::reader(cursor, 7, 5).unwrap();

        let mut buf = String::new();
        reader.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "World");
    }

    #[test]
    fn test_range_reader_reset() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = RangeReader::new_and_seek(cursor, 7, 5).unwrap();

        // Read some data
        let mut buf = [0u8; 2];
        reader.read(&mut buf).unwrap();
        assert_eq!(&buf, b"Wo");

        // Reset and read again
        reader.reset().unwrap();
        reader.read(&mut buf).unwrap();
        assert_eq!(&buf, b"Wo");
    }
}