//! Memory-mapped I/O implementations
//!
//! This module provides memory-mapped implementations of DataInput and DataOutput
//! traits for high-performance, zero-copy file operations.

use crate::error::{Result, ToplingError};
use crate::io::{DataInput, DataOutput, VarInt};
use std::fs::{File, OpenOptions};
use std::path::Path;

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapMut, MmapOptions};

/// Memory-mapped file input for zero-copy reading operations
///
/// Provides efficient reading from memory-mapped files without copying data.
/// Ideal for large files and random access patterns.
///
/// # Examples
///
/// ```rust
/// use infini_zip::io::MemoryMappedInput;
/// use infini_zip::DataInput;
/// use std::fs::File;
/// use std::io::Write;
///
/// # use tempfile::NamedTempFile;
/// # let mut temp_file = NamedTempFile::new().unwrap();
/// # temp_file.write_all(&[0x01, 0x02, 0x03, 0x04]).unwrap();
/// # let temp_path = temp_file.path();
/// let file = File::open(temp_path).unwrap();
/// let mut input = MemoryMappedInput::new(file).unwrap();
///
/// let value = input.read_u32().unwrap();
/// assert_eq!(value, 0x04030201); // Little-endian
/// ```
#[cfg(feature = "mmap")]
pub struct MemoryMappedInput {
    mmap: Mmap,
    position: usize,
}

#[cfg(feature = "mmap")]
impl MemoryMappedInput {
    /// Creates a new memory-mapped input from a file
    pub fn new(file: File) -> Result<Self> {
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ToplingError::io_error(format!("Failed to memory-map file: {}", e)))?
        };

        Ok(MemoryMappedInput { mmap, position: 0 })
    }

    /// Creates a new memory-mapped input from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ToplingError::io_error(format!("Failed to open file: {}", e)))?;
        Self::new(file)
    }

    /// Returns the total size of the mapped region
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns true if the mapped region is empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Returns the current position in the mapped region
    pub fn position(&self) -> usize {
        self.position
    }

    /// Seeks to a specific position in the mapped region
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.mmap.len() {
            return Err(ToplingError::out_of_bounds(pos, self.mmap.len()));
        }
        self.position = pos;
        Ok(())
    }

    /// Returns the remaining bytes available for reading
    pub fn remaining(&self) -> usize {
        self.mmap.len().saturating_sub(self.position)
    }

    /// Reads a slice of bytes without copying (zero-copy)
    pub fn read_slice(&mut self, len: usize) -> Result<&[u8]> {
        let end_pos = self.position + len;
        if end_pos > self.mmap.len() {
            return Err(ToplingError::out_of_bounds(end_pos, self.mmap.len()));
        }

        let slice = &self.mmap[self.position..end_pos];
        self.position = end_pos;
        Ok(slice)
    }

    /// Peeks at bytes without advancing the position
    pub fn peek_slice(&self, len: usize) -> Result<&[u8]> {
        let end_pos = self.position + len;
        if end_pos > self.mmap.len() {
            return Err(ToplingError::out_of_bounds(end_pos, self.mmap.len()));
        }

        Ok(&self.mmap[self.position..end_pos])
    }
}

#[cfg(feature = "mmap")]
impl DataInput for MemoryMappedInput {
    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.mmap.len() {
            return Err(ToplingError::out_of_bounds(self.position, self.mmap.len()));
        }

        let value = self.mmap[self.position];
        self.position += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_slice(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_slice(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_slice(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_var_int(&mut self) -> Result<u64> {
        let mut result = 0u64;
        let mut shift = 0;

        for _ in 0..VarInt::MAX_ENCODED_LEN {
            let byte = self.read_u8()?;
            result |= ((byte & 0x7F) as u64) << shift;

            if byte & 0x80 == 0 {
                return Ok(result);
            }

            shift += 7;
            if shift >= 64 {
                return Err(ToplingError::invalid_data("Variable integer overflow"));
            }
        }

        Err(ToplingError::invalid_data("Variable integer too long"))
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        let data = self.read_slice(buf.len())?;
        buf.copy_from_slice(data);
        Ok(())
    }

    fn read_length_prefixed_string(&mut self) -> Result<String> {
        let len = self.read_var_int()? as usize;
        let bytes = self.read_slice(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| ToplingError::invalid_data(format!("Invalid UTF-8 string: {}", e)))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let new_pos = self.position + n;
        if new_pos > self.mmap.len() {
            return Err(ToplingError::out_of_bounds(new_pos, self.mmap.len()));
        }
        self.position = new_pos;
        Ok(())
    }
}

/// Memory-mapped file output for efficient writing operations
///
/// Provides efficient writing to memory-mapped files with automatic growth.
/// Useful for sequential writing patterns and large file generation.
///
/// # Examples
///
/// ```rust
/// use infini_zip::io::MemoryMappedOutput;
/// use infini_zip::DataOutput;
/// use tempfile::NamedTempFile;
///
/// let temp_file = NamedTempFile::new().unwrap();
/// let mut output = MemoryMappedOutput::create(temp_file.path(), 1024).unwrap();
///
/// output.write_u32(0x12345678).unwrap();
/// output.write_length_prefixed_string("hello").unwrap();
/// output.flush().unwrap();
/// ```
#[cfg(feature = "mmap")]
pub struct MemoryMappedOutput {
    file: File,
    mmap: MmapMut,
    position: usize,
    capacity: usize,
}

#[cfg(feature = "mmap")]
impl MemoryMappedOutput {
    /// Creates a new memory-mapped output file with specified initial size
    pub fn create<P: AsRef<Path>>(path: P, initial_size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| ToplingError::io_error(format!("Failed to create file: {}", e)))?;

        // Set the file size
        file.set_len(initial_size as u64)
            .map_err(|e| ToplingError::io_error(format!("Failed to set file size: {}", e)))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| ToplingError::io_error(format!("Failed to memory-map file: {}", e)))?
        };

        Ok(MemoryMappedOutput {
            file,
            mmap,
            position: 0,
            capacity: initial_size,
        })
    }

    /// Opens an existing file for memory-mapped writing
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| ToplingError::io_error(format!("Failed to open file: {}", e)))?;

        let file_size = file
            .metadata()
            .map_err(|e| ToplingError::io_error(format!("Failed to get file metadata: {}", e)))?
            .len() as usize;

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| ToplingError::io_error(format!("Failed to memory-map file: {}", e)))?
        };

        Ok(MemoryMappedOutput {
            file,
            mmap,
            position: 0,
            capacity: file_size,
        })
    }

    /// Returns the current position in the mapped region
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the total capacity of the mapped region
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the remaining space available for writing
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.position)
    }

    /// Seeks to a specific position in the mapped region
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.capacity {
            return Err(ToplingError::out_of_bounds(pos, self.capacity));
        }
        self.position = pos;
        Ok(())
    }

    /// Ensures the mapped region has at least the specified capacity
    fn ensure_capacity(&mut self, required: usize) -> Result<()> {
        if required <= self.capacity {
            return Ok(());
        }

        // Grow by 50% or to required size, whichever is larger
        let new_size = std::cmp::max(required, self.capacity + (self.capacity / 2));

        // Flush current mmap and grow the file
        drop(std::mem::replace(
            &mut self.mmap,
            MmapMut::map_anon(0).unwrap(),
        ));

        self.file
            .set_len(new_size as u64)
            .map_err(|e| ToplingError::io_error(format!("Failed to grow file: {}", e)))?;

        self.mmap = unsafe {
            MmapOptions::new()
                .map_mut(&self.file)
                .map_err(|e| ToplingError::io_error(format!("Failed to remap file: {}", e)))?
        };

        self.capacity = new_size;
        Ok(())
    }

    /// Writes a slice of bytes to the mapped region
    pub fn write_slice(&mut self, data: &[u8]) -> Result<()> {
        let required_capacity = self.position + data.len();
        self.ensure_capacity(required_capacity)?;

        self.mmap[self.position..self.position + data.len()].copy_from_slice(data);
        self.position += data.len();
        Ok(())
    }

    /// Truncates the file to the current position
    pub fn truncate(&mut self) -> Result<()> {
        // Flush to ensure all data is written
        self.mmap
            .flush()
            .map_err(|e| ToplingError::io_error(format!("Failed to flush mmap: {}", e)))?;

        // Unmap before truncating
        drop(std::mem::replace(
            &mut self.mmap,
            MmapMut::map_anon(0).unwrap(),
        ));

        // Truncate the file
        self.file
            .set_len(self.position as u64)
            .map_err(|e| ToplingError::io_error(format!("Failed to truncate file: {}", e)))?;

        // Remap with new size
        self.mmap = unsafe {
            MmapOptions::new().map_mut(&self.file).map_err(|e| {
                ToplingError::io_error(format!("Failed to remap truncated file: {}", e))
            })?
        };

        self.capacity = self.position;
        Ok(())
    }
}

#[cfg(feature = "mmap")]
impl DataOutput for MemoryMappedOutput {
    fn write_u8(&mut self, value: u8) -> Result<()> {
        self.write_slice(&[value])
    }

    fn write_u16(&mut self, value: u16) -> Result<()> {
        self.write_slice(&value.to_le_bytes())
    }

    fn write_u32(&mut self, value: u32) -> Result<()> {
        self.write_slice(&value.to_le_bytes())
    }

    fn write_u64(&mut self, value: u64) -> Result<()> {
        self.write_slice(&value.to_le_bytes())
    }

    fn write_var_int(&mut self, value: u64) -> Result<()> {
        let encoded = VarInt::encode(value);
        self.write_slice(&encoded)
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.write_slice(data)
    }

    fn write_length_prefixed_string(&mut self, s: &str) -> Result<()> {
        self.write_var_int(s.len() as u64)?;
        self.write_slice(s.as_bytes())
    }

    fn flush(&mut self) -> Result<()> {
        self.mmap.flush().map_err(|e| {
            ToplingError::io_error(format!("Failed to flush memory-mapped file: {}", e))
        })
    }
}

// Provide stub implementations when mmap feature is disabled
#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedInput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedInput {
    pub fn new(_file: File) -> Result<Self> {
        Err(ToplingError::invalid_operation(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }

    pub fn from_path<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(ToplingError::invalid_operation(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }
}

#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedOutput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedOutput {
    pub fn create<P: AsRef<Path>>(_path: P, _initial_size: usize) -> Result<Self> {
        Err(ToplingError::invalid_operation(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedOutput.",
        ))
    }

    pub fn open<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(ToplingError::invalid_operation(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedOutput.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_input_basic() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0x01, 0x02, 0x03, 0x04]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        assert_eq!(input.len(), 4);
        assert!(!input.is_empty());
        assert_eq!(input.position(), 0);
        assert_eq!(input.remaining(), 4);

        assert_eq!(input.read_u8().unwrap(), 0x01);
        assert_eq!(input.position(), 1);
        assert_eq!(input.remaining(), 3);

        assert_eq!(input.read_u8().unwrap(), 0x02);
        assert_eq!(input.read_u16().unwrap(), 0x0403); // Little-endian
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_input_from_path() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Hello, World!").unwrap();
        temp_file.flush().unwrap();

        let mut input = MemoryMappedInput::from_path(temp_file.path()).unwrap();
        assert_eq!(input.len(), 13);

        let slice = input.read_slice(5).unwrap();
        assert_eq!(slice, b"Hello");

        input.seek(7).unwrap();
        let slice = input.read_slice(6).unwrap();
        assert_eq!(slice, b"World!");
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_input_operations() {
        let mut temp_file = NamedTempFile::new().unwrap();

        // Write test data
        temp_file.write_all(&0x12345678u32.to_le_bytes()).unwrap();
        temp_file
            .write_all(&0x9ABCDEF012345678u64.to_le_bytes())
            .unwrap();

        // Write variable integer
        let var_bytes = VarInt::encode(300);
        temp_file.write_all(&var_bytes).unwrap();

        // Write length-prefixed string
        let test_str = "Hello, Memory Mapping!";
        let str_len_bytes = VarInt::encode(test_str.len() as u64);
        temp_file.write_all(&str_len_bytes).unwrap();
        temp_file.write_all(test_str.as_bytes()).unwrap();

        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Test reading
        assert_eq!(input.read_u32().unwrap(), 0x12345678);
        assert_eq!(input.read_u64().unwrap(), 0x9ABCDEF012345678);
        assert_eq!(input.read_var_int().unwrap(), 300);
        assert_eq!(input.read_length_prefixed_string().unwrap(), test_str);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_input_peek_and_skip() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"0123456789").unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Test peek
        let peeked = input.peek_slice(3).unwrap();
        assert_eq!(peeked, b"012");
        assert_eq!(input.position(), 0); // Position shouldn't change

        // Test skip
        input.skip(2).unwrap();
        assert_eq!(input.position(), 2);

        let slice = input.read_slice(3).unwrap();
        assert_eq!(slice, b"234");
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_output_basic() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_output.dat");

        let mut output = MemoryMappedOutput::create(&file_path, 1024).unwrap();
        assert_eq!(output.capacity(), 1024);
        assert_eq!(output.position(), 0);
        assert_eq!(output.remaining(), 1024);

        output.write_u32(0x12345678).unwrap();
        assert_eq!(output.position(), 4);

        output.write_slice(b"Hello").unwrap();
        assert_eq!(output.position(), 9);

        output.flush().unwrap();

        // Verify by reading back
        let file = File::open(&file_path).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();
        assert_eq!(input.read_u32().unwrap(), 0x12345678);

        let slice = input.read_slice(5).unwrap();
        assert_eq!(slice, b"Hello");
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_output_growth() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_growth.dat");

        let mut output = MemoryMappedOutput::create(&file_path, 10).unwrap();
        assert_eq!(output.capacity(), 10);

        // Write more data than initial capacity
        let large_data = vec![0xAB; 20];
        output.write_slice(&large_data).unwrap();

        // Capacity should have grown
        assert!(output.capacity() >= 20);
        assert_eq!(output.position(), 20);

        output.flush().unwrap();

        // Verify data
        let file = File::open(&file_path).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();
        let read_data = input.read_slice(20).unwrap();
        assert_eq!(read_data, &large_data);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_output_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_ops.dat");

        let mut output = MemoryMappedOutput::create(&file_path, 1024).unwrap();

        // Test various write operations
        output.write_u8(0xFF).unwrap();
        output.write_u16(0x1234).unwrap();
        output.write_u32(0x56789ABC).unwrap();
        output.write_u64(0xDEF0123456789ABC).unwrap();
        output.write_var_int(12345).unwrap();
        output.write_length_prefixed_string("Test String").unwrap();

        output.flush().unwrap();

        // Verify by reading back
        let file = File::open(&file_path).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        assert_eq!(input.read_u8().unwrap(), 0xFF);
        assert_eq!(input.read_u16().unwrap(), 0x1234);
        assert_eq!(input.read_u32().unwrap(), 0x56789ABC);
        assert_eq!(input.read_u64().unwrap(), 0xDEF0123456789ABC);
        assert_eq!(input.read_var_int().unwrap(), 12345);
        assert_eq!(input.read_length_prefixed_string().unwrap(), "Test String");
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_output_truncate() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_truncate.dat");

        let mut output = MemoryMappedOutput::create(&file_path, 1024).unwrap();
        output.write_slice(b"Hello, World!").unwrap();

        let written_len = output.position();
        output.truncate().unwrap();

        assert_eq!(output.capacity(), written_len);

        // Verify file size
        let metadata = std::fs::metadata(&file_path).unwrap();
        assert_eq!(metadata.len() as usize, written_len);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_memory_mapped_input_bounds_checking() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[1, 2, 3]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Reading beyond bounds should fail
        assert!(input.read_slice(5).is_err());
        assert!(input.peek_slice(5).is_err());

        input.seek(2).unwrap();
        assert!(input.read_u16().is_err()); // Would read beyond end

        // Seeking beyond bounds should fail
        assert!(input.seek(10).is_err());
    }

    #[cfg(not(feature = "mmap"))]
    #[test]
    fn test_mmap_disabled_error() {
        use std::fs::File;

        // When mmap feature is disabled, should return appropriate errors
        let temp_file = NamedTempFile::new().unwrap();
        let file = File::open(temp_file.path()).unwrap();

        let result = MemoryMappedInput::new(file);
        assert!(result.is_err());

        let result = MemoryMappedOutput::create(temp_file.path(), 1024);
        assert!(result.is_err());
    }
}
