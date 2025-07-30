//! Data output abstractions and implementations
//!
//! This module provides traits and implementations for writing various data types
//! to different destinations including byte vectors, files, and memory-mapped regions.

use std::io::{self, Write};
use std::path::Path;
use std::fs::{File, OpenOptions};

use crate::error::{Result, ToplingError};
use crate::io::var_int::VarInt;

/// Trait for writing structured data to various destinations
pub trait DataOutput {
    /// Write a single byte
    fn write_u8(&mut self, value: u8) -> Result<()>;
    
    /// Write a 16-bit unsigned integer in little-endian format
    fn write_u16(&mut self, value: u16) -> Result<()>;
    
    /// Write a 32-bit unsigned integer in little-endian format
    fn write_u32(&mut self, value: u32) -> Result<()>;
    
    /// Write a 64-bit unsigned integer in little-endian format
    fn write_u64(&mut self, value: u64) -> Result<()>;
    
    /// Write a variable-length encoded integer
    fn write_var_int(&mut self, value: u64) -> Result<()>;
    
    /// Write bytes from the provided buffer
    fn write_bytes(&mut self, data: &[u8]) -> Result<()>;
    
    /// Write a length-prefixed byte slice (length as varint)
    fn write_length_prefixed_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.write_var_int(data.len() as u64)?;
        self.write_bytes(data)
    }
    
    /// Write a string with UTF-8 encoding
    fn write_string(&mut self, s: &str) -> Result<()> {
        self.write_bytes(s.as_bytes())
    }
    
    /// Write a length-prefixed string (length as varint, UTF-8 encoded)
    fn write_length_prefixed_string(&mut self, s: &str) -> Result<()> {
        self.write_length_prefixed_bytes(s.as_bytes())
    }
    
    /// Flush any buffered data to the underlying destination
    fn flush(&mut self) -> Result<()>;
    
    /// Get the current position (if supported)
    fn position(&self) -> Option<u64> {
        None
    }
    
    /// Get the total number of bytes written (if supported)
    fn bytes_written(&self) -> Option<u64> {
        None
    }
}

/// DataOutput implementation for Vec<u8>
pub struct VecDataOutput {
    data: Vec<u8>,
}

impl VecDataOutput {
    /// Create a new VecDataOutput
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    /// Create a new VecDataOutput with the specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    
    /// Get the number of bytes written
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if no bytes have been written
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get a reference to the underlying data
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    /// Convert into the underlying Vec<u8>
    pub fn into_vec(self) -> Vec<u8> {
        self.data
    }
    
    /// Clear all written data
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Reserve additional capacity
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }
}

impl Default for VecDataOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl DataOutput for VecDataOutput {
    fn write_u8(&mut self, value: u8) -> Result<()> {
        self.data.push(value);
        Ok(())
    }
    
    fn write_u16(&mut self, value: u16) -> Result<()> {
        self.data.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }
    
    fn write_u32(&mut self, value: u32) -> Result<()> {
        self.data.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }
    
    fn write_u64(&mut self, value: u64) -> Result<()> {
        self.data.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }
    
    fn write_var_int(&mut self, value: u64) -> Result<()> {
        VarInt::write_to(&mut self.data, value)?;
        Ok(())
    }
    
    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.data.extend_from_slice(data);
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        // No-op for Vec, data is always "flushed"
        Ok(())
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
    
    fn bytes_written(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

/// DataOutput implementation for std::io::Write types
pub struct WriterDataOutput<W> {
    writer: W,
    bytes_written: u64,
}

impl<W: Write> WriterDataOutput<W> {
    /// Create a new WriterDataOutput from a Write type
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            bytes_written: 0,
        }
    }
    
    /// Get the number of bytes written
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    /// Convert back to the underlying writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> DataOutput for WriterDataOutput<W> {
    fn write_u8(&mut self, value: u8) -> Result<()> {
        self.writer.write_all(&[value]).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u8: {}", e))
        })?;
        self.bytes_written += 1;
        Ok(())
    }
    
    fn write_u16(&mut self, value: u16) -> Result<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u16: {}", e))
        })?;
        self.bytes_written += 2;
        Ok(())
    }
    
    fn write_u32(&mut self, value: u32) -> Result<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u32: {}", e))
        })?;
        self.bytes_written += 4;
        Ok(())
    }
    
    fn write_u64(&mut self, value: u64) -> Result<()> {
        self.writer.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u64: {}", e))
        })?;
        self.bytes_written += 8;
        Ok(())
    }
    
    fn write_var_int(&mut self, value: u64) -> Result<()> {
        let start_bytes = self.bytes_written;
        VarInt::write_to(self, value)?;
        Ok(())
    }
    
    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data).map_err(|e| {
            ToplingError::io_error(format!("Failed to write bytes: {}", e))
        })?;
        self.bytes_written += data.len() as u64;
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        self.writer.flush().map_err(|e| {
            ToplingError::io_error(format!("Failed to flush: {}", e))
        })
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.bytes_written)
    }
    
    fn bytes_written(&self) -> Option<u64> {
        Some(self.bytes_written)
    }
}

// Implement Write for WriterDataOutput to support VarInt writing
impl<W: Write> Write for WriterDataOutput<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.writer.write(buf)?;
        self.bytes_written += written as u64;
        Ok(written)
    }
    
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// DataOutput implementation for files
pub struct FileDataOutput {
    file: File,
    bytes_written: u64,
}

impl FileDataOutput {
    /// Create a new file for writing, truncating if it exists
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path).map_err(|e| {
            ToplingError::io_error(format!("Failed to create file: {}", e))
        })?;
        
        Ok(Self {
            file,
            bytes_written: 0,
        })
    }
    
    /// Open an existing file for writing (append mode)
    pub fn append<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                ToplingError::io_error(format!("Failed to open file for append: {}", e))
            })?;
        
        let bytes_written = file.metadata()
            .map(|m| m.len())
            .unwrap_or(0);
        
        Ok(Self {
            file,
            bytes_written,
        })
    }
    
    /// Get the number of bytes written
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
    
    /// Sync all data to disk
    pub fn sync_all(&mut self) -> Result<()> {
        self.file.sync_all().map_err(|e| {
            ToplingError::io_error(format!("Failed to sync file: {}", e))
        })
    }
    
    /// Sync data (but not metadata) to disk
    pub fn sync_data(&mut self) -> Result<()> {
        self.file.sync_data().map_err(|e| {
            ToplingError::io_error(format!("Failed to sync file data: {}", e))
        })
    }
}

impl DataOutput for FileDataOutput {
    fn write_u8(&mut self, value: u8) -> Result<()> {
        self.file.write_all(&[value]).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u8: {}", e))
        })?;
        self.bytes_written += 1;
        Ok(())
    }
    
    fn write_u16(&mut self, value: u16) -> Result<()> {
        self.file.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u16: {}", e))
        })?;
        self.bytes_written += 2;
        Ok(())
    }
    
    fn write_u32(&mut self, value: u32) -> Result<()> {
        self.file.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u32: {}", e))
        })?;
        self.bytes_written += 4;
        Ok(())
    }
    
    fn write_u64(&mut self, value: u64) -> Result<()> {
        self.file.write_all(&value.to_le_bytes()).map_err(|e| {
            ToplingError::io_error(format!("Failed to write u64: {}", e))
        })?;
        self.bytes_written += 8;
        Ok(())
    }
    
    fn write_var_int(&mut self, value: u64) -> Result<()> {
        VarInt::write_to(self, value)?;
        Ok(())
    }
    
    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.file.write_all(data).map_err(|e| {
            ToplingError::io_error(format!("Failed to write bytes: {}", e))
        })?;
        self.bytes_written += data.len() as u64;
        Ok(())
    }
    
    fn flush(&mut self) -> Result<()> {
        self.file.flush().map_err(|e| {
            ToplingError::io_error(format!("Failed to flush: {}", e))
        })
    }
    
    fn position(&self) -> Option<u64> {
        Some(self.bytes_written)
    }
    
    fn bytes_written(&self) -> Option<u64> {
        Some(self.bytes_written)
    }
}

// Implement Write for FileDataOutput to support VarInt writing
impl Write for FileDataOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.file.write(buf)?;
        self.bytes_written += written as u64;
        Ok(written)
    }
    
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Convenience function to create a DataOutput from a Vec<u8>
pub fn to_vec() -> VecDataOutput {
    VecDataOutput::new()
}

/// Convenience function to create a DataOutput with capacity
pub fn to_vec_with_capacity(capacity: usize) -> VecDataOutput {
    VecDataOutput::with_capacity(capacity)
}

/// Convenience function to create a DataOutput from a Write type
pub fn to_writer<W: Write>(writer: W) -> WriterDataOutput<W> {
    WriterDataOutput::new(writer)
}

/// Convenience function to create a DataOutput for a new file
pub fn to_file<P: AsRef<Path>>(path: P) -> Result<FileDataOutput> {
    FileDataOutput::create(path)
}

/// Convenience function to create a DataOutput for appending to a file
pub fn to_file_append<P: AsRef<Path>>(path: P) -> Result<FileDataOutput> {
    FileDataOutput::append(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::data_input::{DataInput, SliceDataInput};
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_vec_data_output_basic() {
        let mut output = VecDataOutput::new();
        
        assert_eq!(output.len(), 0);
        assert!(output.is_empty());
        assert_eq!(output.position(), Some(0));
        assert_eq!(output.bytes_written(), Some(0));
        
        output.write_u8(42).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output.position(), Some(1));
        
        output.write_u16(0x1234).unwrap();
        output.write_u32(0x12345678).unwrap();
        output.write_u64(0x123456789ABCDEF0).unwrap();
        
        let expected = [
            42,
            0x34, 0x12,  // u16 little-endian
            0x78, 0x56, 0x34, 0x12,  // u32 little-endian
            0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,  // u64 little-endian
        ];
        
        assert_eq!(output.as_slice(), &expected);
        
        let vec = output.into_vec();
        assert_eq!(vec, expected);
    }

    #[test]
    fn test_vec_data_output_bytes_and_strings() {
        let mut output = VecDataOutput::new();
        
        let data = b"hello world";
        output.write_bytes(data).unwrap();
        
        let text = "test string";
        output.write_string(text).unwrap();
        
        let mut expected = Vec::new();
        expected.extend_from_slice(data);
        expected.extend_from_slice(text.as_bytes());
        
        assert_eq!(output.as_slice(), &expected);
    }

    #[test]
    fn test_vec_data_output_length_prefixed() {
        let mut output = VecDataOutput::new();
        
        let data = b"test data";
        output.write_length_prefixed_bytes(data).unwrap();
        
        let text = "test string";
        output.write_length_prefixed_string(text).unwrap();
        
        // Verify we can read it back
        let mut input = SliceDataInput::new(output.as_slice());
        
        let read_data = input.read_length_prefixed_bytes().unwrap();
        assert_eq!(&read_data, data);
        
        let read_text = input.read_length_prefixed_string().unwrap();
        assert_eq!(read_text, text);
    }

    #[test]
    fn test_writer_data_output() {
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut output = WriterDataOutput::new(cursor);
            
            output.write_u8(42).unwrap();
            output.write_u16(0x1234).unwrap();
            output.write_bytes(b"hello").unwrap();
            DataOutput::flush(&mut output).unwrap();
            
            assert_eq!(output.bytes_written(), 8);
        }
        
        let expected = [
            42,
            0x34, 0x12,  // u16 little-endian
            b'h', b'e', b'l', b'l', b'o',
        ];
        
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_file_data_output() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        // Write data to file
        {
            let mut output = FileDataOutput::create(path).unwrap();
            
            output.write_u8(42).unwrap();
            output.write_u16(0x1234).unwrap();
            output.write_string("hello").unwrap();
            DataOutput::flush(&mut output).unwrap();
            output.sync_all().unwrap();
            
            assert_eq!(output.bytes_written(), 8);
        }
        
        // Read data back
        let data = std::fs::read(path).unwrap();
        let expected = [
            42,
            0x34, 0x12,  // u16 little-endian
            b'h', b'e', b'l', b'l', b'o',
        ];
        
        assert_eq!(data, expected);
    }

    #[test]
    fn test_file_data_output_append() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();
        
        // Write initial data
        std::fs::write(path, b"initial").unwrap();
        
        // Append more data
        {
            let mut output = FileDataOutput::append(path).unwrap();
            output.write_string(" appended").unwrap();
            DataOutput::flush(&mut output).unwrap();
        }
        
        // Read all data back
        let data = std::fs::read(path).unwrap();
        assert_eq!(data, b"initial appended");
    }

    #[test]
    fn test_var_int_round_trip() {
        let test_values = [0, 127, 128, 16383, 16384, 2097151, 2097152, u64::MAX];
        
        for &value in &test_values {
            let mut output = VecDataOutput::new();
            output.write_var_int(value).unwrap();
            
            let mut input = crate::io::data_input::SliceDataInput::new(output.as_slice());
            let decoded = input.read_var_int().unwrap();
            
            assert_eq!(value, decoded, "Failed for value {}", value);
        }
    }

    #[test]
    fn test_capacity_management() {
        let mut output = VecDataOutput::with_capacity(100);
        // Note: capacity() is not available on slices, check the underlying Vec
        
        output.reserve(200);
        
        output.write_bytes(b"test").unwrap();
        output.clear();
        assert!(output.is_empty());
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn test_convenience_functions() {
        let mut output = to_vec();
        output.write_u32(0x12345678).unwrap();
        
        let mut output = to_vec_with_capacity(1024);
        output.write_string("test").unwrap();
        
        let buffer = Vec::new();
        let cursor = Cursor::new(buffer);
        let mut output = to_writer(cursor);
        output.write_u8(42).unwrap();
    }

    #[test]
    fn test_round_trip_all_types() {
        let mut output = VecDataOutput::new();
        
        // Write various data types
        output.write_u8(42).unwrap();
        output.write_u16(0x1234).unwrap();
        output.write_u32(0x12345678).unwrap();
        output.write_u64(0x123456789ABCDEF0).unwrap();
        output.write_var_int(12345).unwrap();
        output.write_bytes(b"hello").unwrap();
        output.write_string(" world").unwrap();
        output.write_length_prefixed_string("length prefixed").unwrap();
        
        // Read everything back
        let mut input = SliceDataInput::new(output.as_slice());
        
        assert_eq!(input.read_u8().unwrap(), 42);
        assert_eq!(input.read_u16().unwrap(), 0x1234);
        assert_eq!(input.read_u32().unwrap(), 0x12345678);
        assert_eq!(input.read_u64().unwrap(), 0x123456789ABCDEF0);
        assert_eq!(input.read_var_int().unwrap(), 12345);
        
        let mut buf = [0u8; 5];
        input.read_bytes(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        
        let text = input.read_string(6).unwrap();
        assert_eq!(text, " world");
        
        let prefixed_text = input.read_length_prefixed_string().unwrap();
        assert_eq!(prefixed_text, "length prefixed");
        
        assert!(!input.has_more());
    }
}