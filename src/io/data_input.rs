//! Data input abstractions and implementations
//!
//! This module provides traits and implementations for reading various data types
//! from different sources including byte slices, files, and memory-mapped regions.

use std::io::{self, Read, Cursor};
use std::path::Path;
use std::fs::File;

use crate::error::{Result, ToplingError};
use crate::io::var_int::VarInt;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

/// Trait for reading structured data from various sources
pub trait DataInput {
    /// Read a single byte
    fn read_u8(&mut self) -> Result<u8>;
    
    /// Read a 16-bit unsigned integer in little-endian format
    fn read_u16(&mut self) -> Result<u16>;
    
    /// Read a 32-bit unsigned integer in little-endian format
    fn read_u32(&mut self) -> Result<u32>;
    
    /// Read a 64-bit unsigned integer in little-endian format
    fn read_u64(&mut self) -> Result<u64>;
    
    /// Read a variable-length encoded integer
    fn read_var_int(&mut self) -> Result<u64>;
    
    /// Read exact number of bytes into the provided buffer
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()>;
    
    /// Read a vector of bytes with the specified length
    fn read_vec(&mut self, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.read_bytes(&mut buf)?;
        Ok(buf)
    }
    
    /// Read a length-prefixed byte vector (length as varint)
    fn read_length_prefixed_bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.read_var_int()? as usize;
        self.read_vec(len)
    }
    
    /// Read a string with the specified length (UTF-8 encoded)
    fn read_string(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_vec(len)?;
        String::from_utf8(bytes).map_err(|e| {
            ToplingError::invalid_data(format!("Invalid UTF-8 string: {}", e))
        })
    }
    
    /// Read a length-prefixed string (length as varint, UTF-8 encoded)
    fn read_length_prefixed_string(&mut self) -> Result<String> {
        let bytes = self.read_length_prefixed_bytes()?;
        String::from_utf8(bytes).map_err(|e| {
            ToplingError::invalid_data(format!("Invalid UTF-8 string: {}", e))
        })
    }
    
    /// Skip the specified number of bytes
    fn skip(&mut self, n: usize) -> Result<()>;
    
    /// Get the current position (if supported)
    fn position(&self) -> Option<u64> {
        None
    }
    
    /// Check if there are more bytes to read (if supported)
    fn has_remaining(&self) -> Option<bool> {
        None
    }
}

/// DataInput implementation for byte slices
pub struct SliceDataInput<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SliceDataInput<'a> {
    /// Create a new SliceDataInput from a byte slice
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
        }
    }
    
    /// Get the current position
    pub fn pos(&self) -> usize {
        self.position
    }
    
    /// Get the remaining bytes
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }
    
    /// Check if there are more bytes to read
    pub fn has_more(&self) -> bool {
        self.position < self.data.len()
    }
    
    /// Get a slice of the remaining data
    pub fn remaining_slice(&self) -> &'a [u8] {
        &self.data[self.position..]
    }
}

impl<'a> DataInput for SliceDataInput<'a> {
    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.data.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }
    
    fn read_u16(&mut self) -> Result<u16> {
        if self.position + 2 > self.data.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 2];
        self.position += 2;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    
    fn read_u32(&mut self) -> Result<u32> {
        if self.position + 4 > self.data.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 4];
        self.position += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    
    fn read_u64(&mut self) -> Result<u64> {
        if self.position + 8 > self.data.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 8];
        self.position += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
    
    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }
    
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.position + buf.len() > self.data.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        buf.copy_from_slice(&self.data[self.position..self.position + buf.len()]);
        self.position += buf.len();
        Ok(())
    }
    
    fn skip(&mut self, n: usize) -> Result<()> {
        if self.position + n > self.data.len() {
            return Err(ToplingError::io_error("Cannot skip past end of data"));
        }
        self.position += n;
        Ok(())
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.position as u64)
    }
    
    fn has_remaining(&self) -> Option<bool> {
        Some(self.position < self.data.len())
    }
}

/// DataInput implementation for std::io::Read types
pub struct ReaderDataInput<R> {
    reader: R,
    position: u64,
}

impl<R: Read> ReaderDataInput<R> {
    /// Create a new ReaderDataInput from a Read type
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            position: 0,
        }
    }
    
    /// Get the current position
    pub fn pos(&self) -> u64 {
        self.position
    }
    
    /// Convert back to the underlying reader
    pub fn into_inner(self) -> R {
        self.reader
    }
}

impl<R: Read> DataInput for ReaderDataInput<R> {
    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read_exact(&mut buf).map_err(|e| {
            ToplingError::io_error(format!("Failed to read u8: {}", e))
        })?;
        self.position += 1;
        Ok(buf[0])
    }
    
    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.reader.read_exact(&mut buf).map_err(|e| {
            ToplingError::io_error(format!("Failed to read u16: {}", e))
        })?;
        self.position += 2;
        Ok(u16::from_le_bytes(buf))
    }
    
    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader.read_exact(&mut buf).map_err(|e| {
            ToplingError::io_error(format!("Failed to read u32: {}", e))
        })?;
        self.position += 4;
        Ok(u32::from_le_bytes(buf))
    }
    
    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader.read_exact(&mut buf).map_err(|e| {
            ToplingError::io_error(format!("Failed to read u64: {}", e))
        })?;
        self.position += 8;
        Ok(u64::from_le_bytes(buf))
    }
    
    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }
    
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        self.reader.read_exact(buf).map_err(|e| {
            ToplingError::io_error(format!("Failed to read bytes: {}", e))
        })?;
        self.position += buf.len() as u64;
        Ok(())
    }
    
    fn skip(&mut self, n: usize) -> Result<()> {
        let mut buf = vec![0u8; n.min(8192)]; // Read in chunks for large skips
        let mut remaining = n;
        
        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            self.reader.read_exact(&mut buf[..to_read]).map_err(|e| {
                ToplingError::io_error(format!("Failed to skip bytes: {}", e))
            })?;
            remaining -= to_read;
        }
        
        self.position += n as u64;
        Ok(())
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.position)
    }
}

/// DataInput implementation for memory-mapped files
#[cfg(feature = "mmap")]
pub struct MmapDataInput {
    mmap: Mmap,
    position: usize,
}

#[cfg(feature = "mmap")]
impl MmapDataInput {
    /// Create a new MmapDataInput from a file path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            ToplingError::io_error(format!("Failed to open file: {}", e))
        })?;
        
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                ToplingError::io_error(format!("Failed to memory map file: {}", e))
            })?
        };
        
        Ok(Self {
            mmap,
            position: 0,
        })
    }
    
    /// Create a new MmapDataInput from an existing memory map
    pub fn from_mmap(mmap: Mmap) -> Self {
        Self {
            mmap,
            position: 0,
        }
    }
    
    /// Get the current position
    pub fn pos(&self) -> usize {
        self.position
    }
    
    /// Get the total size of the memory-mapped region
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
    
    /// Check if the memory-mapped region is empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
    
    /// Get the remaining bytes
    pub fn remaining(&self) -> usize {
        self.len().saturating_sub(self.position)
    }
    
    /// Get a slice of the remaining data
    pub fn remaining_slice(&self) -> &[u8] {
        &self.mmap[self.position..]
    }
    
    /// Get the entire memory-mapped slice
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}

#[cfg(feature = "mmap")]
impl DataInput for MmapDataInput {
    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.mmap.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let value = self.mmap[self.position];
        self.position += 1;
        Ok(value)
    }
    
    fn read_u16(&mut self) -> Result<u16> {
        if self.position + 2 > self.mmap.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 2];
        self.position += 2;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }
    
    fn read_u32(&mut self) -> Result<u32> {
        if self.position + 4 > self.mmap.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 4];
        self.position += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    
    fn read_u64(&mut self) -> Result<u64> {
        if self.position + 8 > self.mmap.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 8];
        self.position += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
    
    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }
    
    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.position + buf.len() > self.mmap.len() {
            return Err(ToplingError::io_error("Unexpected end of data"));
        }
        buf.copy_from_slice(&self.mmap[self.position..self.position + buf.len()]);
        self.position += buf.len();
        Ok(())
    }
    
    fn skip(&mut self, n: usize) -> Result<()> {
        if self.position + n > self.mmap.len() {
            return Err(ToplingError::io_error("Cannot skip past end of data"));
        }
        self.position += n;
        Ok(())
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.position as u64)
    }
    
    fn has_remaining(&self) -> Option<bool> {
        Some(self.position < self.mmap.len())
    }
}

/// Convenience function to create a DataInput from a byte slice
pub fn from_slice(data: &[u8]) -> SliceDataInput<'_> {
    SliceDataInput::new(data)
}

/// Convenience function to create a DataInput from a Read type
pub fn from_reader<R: Read>(reader: R) -> ReaderDataInput<R> {
    ReaderDataInput::new(reader)
}

/// Convenience function to create a DataInput from a file path
#[cfg(feature = "mmap")]
pub fn from_file<P: AsRef<Path>>(path: P) -> Result<MmapDataInput> {
    MmapDataInput::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_slice_data_input_basic() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut input = SliceDataInput::new(&data);
        
        assert_eq!(input.position(), Some(0));
        assert_eq!(input.has_remaining(), Some(true));
        
        assert_eq!(input.read_u8().unwrap(), 1);
        assert_eq!(input.pos(), 1);
        
        assert_eq!(input.read_u16().unwrap(), u16::from_le_bytes([2, 3]));
        assert_eq!(input.pos(), 3);
        
        assert_eq!(input.read_u32().unwrap(), u32::from_le_bytes([4, 5, 6, 7]));
        assert_eq!(input.pos(), 7);
        
        let mut buf = [0u8; 2];
        input.read_bytes(&mut buf).unwrap();
        assert_eq!(buf, [8, 9]);
        
        assert_eq!(input.remaining(), 1);
        assert_eq!(input.read_u8().unwrap(), 10);
        
        assert_eq!(input.remaining(), 0);
        assert_eq!(input.has_remaining(), Some(false));
        
        // Should fail on reading past end
        assert!(input.read_u8().is_err());
    }

    #[test]
    fn test_slice_data_input_skip() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut input = SliceDataInput::new(&data);
        
        input.skip(3).unwrap();
        assert_eq!(input.pos(), 3);
        assert_eq!(input.read_u8().unwrap(), 4);
        
        input.skip(2).unwrap();
        assert_eq!(input.read_u8().unwrap(), 7);
        
        // Should fail when skipping past end
        assert!(input.skip(10).is_err());
    }

    #[test]
    fn test_reader_data_input() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let cursor = Cursor::new(data);
        let mut input = ReaderDataInput::new(cursor);
        
        assert_eq!(input.position(), Some(0));
        
        assert_eq!(input.read_u8().unwrap(), 1);
        assert_eq!(input.pos(), 1);
        
        assert_eq!(input.read_u16().unwrap(), u16::from_le_bytes([2, 3]));
        assert_eq!(input.pos(), 3);
        
        let mut buf = [0u8; 3];
        input.read_bytes(&mut buf).unwrap();
        assert_eq!(buf, [4, 5, 6]);
        assert_eq!(input.pos(), 6);
        
        input.skip(2).unwrap();
        assert_eq!(input.pos(), 8);
        assert_eq!(input.read_u8().unwrap(), 9);
    }

    #[test]
    fn test_var_int_encoding() {
        let test_values = [0, 127, 128, 16383, 16384, 2097151, 2097152, u64::MAX];
        
        for &value in &test_values {
            let mut encoded = Vec::new();
            VarInt::write_to(&mut encoded, value).unwrap();
            
            let mut input = SliceDataInput::new(&encoded);
            let decoded = input.read_var_int().unwrap();
            
            assert_eq!(value, decoded, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_string_operations() {
        let test_string = "Hello, 世界! 🦀";
        let bytes = test_string.as_bytes();
        
        // Test fixed-length string
        let mut input = SliceDataInput::new(bytes);
        let decoded = input.read_string(bytes.len()).unwrap();
        assert_eq!(test_string, decoded);
        
        // Test length-prefixed string
        let mut encoded = Vec::new();
        VarInt::write_to(&mut encoded, bytes.len() as u64).unwrap();
        encoded.extend_from_slice(bytes);
        
        let mut input = SliceDataInput::new(&encoded);
        let decoded = input.read_length_prefixed_string().unwrap();
        assert_eq!(test_string, decoded);
    }

    #[test]
    fn test_length_prefixed_bytes() {
        let data = b"test data";
        
        let mut encoded = Vec::new();
        VarInt::write_to(&mut encoded, data.len() as u64).unwrap();
        encoded.extend_from_slice(data);
        
        let mut input = SliceDataInput::new(&encoded);
        let decoded = input.read_length_prefixed_bytes().unwrap();
        assert_eq!(data, &decoded[..]);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_data_input() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let data = b"Hello, memory mapped world!";
        temp_file.write_all(data).unwrap();
        temp_file.flush().unwrap();
        
        let mut input = MmapDataInput::open(temp_file.path()).unwrap();
        
        assert_eq!(input.len(), data.len());
        assert!(!input.is_empty());
        assert_eq!(input.remaining(), data.len());
        
        let first_word = input.read_vec(5).unwrap();
        assert_eq!(&first_word, b"Hello");
        
        input.skip(2).unwrap(); // Skip ", "
        let second_word = input.read_vec(6).unwrap();
        assert_eq!(&second_word, b"memory");
        
        let remaining_data = input.remaining_slice();
        assert_eq!(remaining_data, b" mapped world!");
    }
    
    #[test]
    fn test_convenience_functions() {
        let data = [1, 2, 3, 4];
        let mut input = from_slice(&data);
        assert_eq!(input.read_u32().unwrap(), u32::from_le_bytes([1, 2, 3, 4]));
        
        let cursor = Cursor::new(vec![5, 6, 7, 8]);
        let mut input = from_reader(cursor);
        assert_eq!(input.read_u32().unwrap(), u32::from_le_bytes([5, 6, 7, 8]));
    }
}