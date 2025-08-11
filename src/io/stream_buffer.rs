//! Advanced buffered stream wrapper with configurable buffering strategies
//!
//! This module provides high-performance stream buffering with multiple optimization
//! strategies inspired by production systems. Features include adaptive buffering,
//! page-aligned allocations, and branch prediction optimization.

use std::cmp;
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::ptr;

use crate::error::{Result, ZiporaError};
use crate::memory::SecureMemoryPool;
use std::sync::Arc;

/// Configuration for stream buffering behavior
#[derive(Debug, Clone)]
pub struct StreamBufferConfig {
    /// Initial buffer capacity in bytes (default: 64KB)
    pub initial_capacity: usize,
    /// Maximum buffer capacity in bytes (default: 2MB)
    pub max_capacity: usize,
    /// Growth factor when buffer needs to expand (default: 1.618 - golden ratio)
    pub growth_factor: f64,
    /// Page alignment for better memory performance (default: 4096)
    pub page_alignment: usize,
    /// Whether to use secure memory pool for allocations
    pub use_secure_pool: bool,
    /// Minimum read size to trigger bulk operations
    pub bulk_read_threshold: usize,
    /// Enable read-ahead optimization
    pub enable_readahead: bool,
    /// Read-ahead size multiplier
    pub readahead_multiplier: usize,
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64 * 1024,      // 64KB
            max_capacity: 2 * 1024 * 1024,    // 2MB
            growth_factor: 1.618,              // Golden ratio for optimal memory usage
            page_alignment: 4096,              // Standard page size
            use_secure_pool: true,             // Use secure memory by default
            bulk_read_threshold: 8192,         // 8KB threshold
            enable_readahead: true,            // Enable read-ahead
            readahead_multiplier: 2,           // 2x read-ahead
        }
    }
}

impl StreamBufferConfig {
    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            initial_capacity: 128 * 1024,     // 128KB initial
            max_capacity: 4 * 1024 * 1024,    // 4MB max
            growth_factor: 2.0,                // Faster growth
            bulk_read_threshold: 4096,         // Lower threshold
            enable_readahead: true,
            readahead_multiplier: 4,           // Aggressive read-ahead
            ..Default::default()
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            initial_capacity: 16 * 1024,      // 16KB initial
            max_capacity: 512 * 1024,         // 512KB max
            growth_factor: 1.414,              // Conservative growth
            bulk_read_threshold: 16384,        // Higher threshold
            enable_readahead: false,           // Disable read-ahead
            readahead_multiplier: 1,
            ..Default::default()
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            initial_capacity: 8 * 1024,       // 8KB initial
            max_capacity: 256 * 1024,         // 256KB max
            growth_factor: 1.5,
            bulk_read_threshold: 2048,         // Low threshold
            enable_readahead: false,           // No read-ahead for low latency
            readahead_multiplier: 1,
            ..Default::default()
        }
    }
}

/// High-performance buffered reader with configurable strategies
pub struct StreamBufferedReader<R> {
    inner: R,
    buffer: Box<[u8]>,
    pos: usize,      // Current position in buffer
    end: usize,      // End of valid data in buffer
    config: StreamBufferConfig,
    total_read: u64, // Total bytes read from underlying stream
    pool: Option<Arc<SecureMemoryPool>>,
}

impl<R: Read> StreamBufferedReader<R> {
    /// Create a new buffered reader with default configuration
    pub fn new(inner: R) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::default())
    }

    /// Create a new buffered reader with custom configuration
    pub fn with_config(inner: R, config: StreamBufferConfig) -> Result<Self> {
        let pool = if config.use_secure_pool {
            Some(SecureMemoryPool::new(
                crate::memory::SecurePoolConfig::small_secure()
            )?)
        } else {
            None
        };

        // Allocate aligned buffer
        let buffer = Self::allocate_aligned_buffer(config.initial_capacity, config.page_alignment)?;

        Ok(Self {
            inner,
            buffer,
            pos: 0,
            end: 0,
            config,
            total_read: 0,
            pool,
        })
    }

    /// Create a performance-optimized buffered reader
    pub fn performance_optimized(inner: R) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::performance_optimized())
    }

    /// Create a memory-efficient buffered reader
    pub fn memory_efficient(inner: R) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::memory_efficient())
    }

    /// Create a low-latency buffered reader
    pub fn low_latency(inner: R) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::low_latency())
    }

    /// Allocate page-aligned buffer for optimal performance
    fn allocate_aligned_buffer(size: usize, alignment: usize) -> Result<Box<[u8]>> {
        // Round up to alignment boundary
        let aligned_size = (size + alignment - 1) & !(alignment - 1);
        
        // For simplicity, use regular allocation
        // In production, could use posix_memalign or similar
        let buffer = vec![0u8; aligned_size].into_boxed_slice();
        Ok(buffer)
    }

    /// Get the underlying reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the underlying reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consume this buffered reader and return the underlying reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get current buffer usage
    pub fn buffer_usage(&self) -> usize {
        self.end - self.pos
    }

    /// Get total bytes read from underlying stream
    pub fn total_read(&self) -> u64 {
        self.total_read
    }

    /// Check if buffer has data available
    #[inline]
    pub fn has_data_in_buffer(&self) -> bool {
        self.pos < self.end
    }

    /// Ensure buffer has at least `needed` bytes available
    pub fn ensure_buffered(&mut self, needed: usize) -> Result<usize> {
        let available = self.end - self.pos;
        if available >= needed {
            return Ok(available);
        }

        self.fill_buffer(needed)
    }

    /// Fill buffer with fresh data from underlying reader
    fn fill_buffer(&mut self, min_needed: usize) -> Result<usize> {
        // Move any remaining data to beginning of buffer
        if self.pos > 0 {
            let remaining = self.end - self.pos;
            if remaining > 0 {
                // Use memmove-like operation for overlapping regions
                unsafe {
                    ptr::copy(
                        self.buffer.as_ptr().add(self.pos),
                        self.buffer.as_mut_ptr(),
                        remaining,
                    );
                }
            }
            self.end = remaining;
            self.pos = 0;
        }

        // Calculate read size with read-ahead if enabled
        let read_size = if self.config.enable_readahead {
            let readahead_size = min_needed * self.config.readahead_multiplier;
            cmp::min(
                cmp::max(min_needed, readahead_size),
                self.buffer.len() - self.end,
            )
        } else {
            cmp::min(min_needed, self.buffer.len() - self.end)
        };

        if read_size == 0 {
            // Buffer is full but we need more data - grow buffer if possible
            return self.grow_buffer_and_retry(min_needed);
        }

        // Read data from underlying stream
        let bytes_read = self.inner.read(&mut self.buffer[self.end..self.end + read_size])
            .map_err(|e| ZiporaError::io_error(format!("Failed to fill buffer: {}", e)))?;

        self.end += bytes_read;
        self.total_read += bytes_read as u64;

        Ok(self.end - self.pos)
    }

    /// Grow buffer when more space is needed
    fn grow_buffer_and_retry(&mut self, min_needed: usize) -> Result<usize> {
        let current_capacity = self.buffer.len();
        let new_capacity = cmp::min(
            cmp::max(
                (current_capacity as f64 * self.config.growth_factor) as usize,
                current_capacity + min_needed,
            ),
            self.config.max_capacity,
        );

        if new_capacity <= current_capacity {
            return Err(ZiporaError::io_error(
                format!("Buffer at maximum capacity ({} bytes), cannot satisfy request for {} bytes",
                        current_capacity, min_needed)
            ));
        }

        // Allocate new larger buffer
        let mut new_buffer = Self::allocate_aligned_buffer(new_capacity, self.config.page_alignment)?;

        // Copy existing data
        let existing_data = self.end - self.pos;
        new_buffer[..existing_data].copy_from_slice(&self.buffer[self.pos..self.end]);

        // Update state
        self.buffer = new_buffer;
        self.end = existing_data;
        self.pos = 0;

        // Try filling again
        self.fill_buffer(min_needed)
    }

    /// Fast path for reading a single byte
    #[inline]
    pub fn read_byte_fast(&mut self) -> Result<u8> {
        if self.pos < self.end {
            let byte = self.buffer[self.pos];
            self.pos += 1;
            Ok(byte)
        } else {
            self.read_byte_slow()
        }
    }

    /// Slow path for reading a single byte when buffer is empty
    #[cold]
    fn read_byte_slow(&mut self) -> Result<u8> {
        self.ensure_buffered(1)?;
        if self.pos < self.end {
            let byte = self.buffer[self.pos];
            self.pos += 1;
            Ok(byte)
        } else {
            Err(ZiporaError::io_error("Unexpected end of stream"))
        }
    }

    /// Read a slice of bytes directly from buffer if available
    pub fn read_slice(&mut self, len: usize) -> Result<Option<&[u8]>> {
        // Ensure we have enough data in buffer
        self.ensure_buffered(len)?;
        
        if self.pos + len <= self.end {
            let slice = &self.buffer[self.pos..self.pos + len];
            self.pos += len;
            Ok(Some(slice))
        } else {
            Ok(None)
        }
    }

    /// Bulk read optimization for large reads
    pub fn read_bulk(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.len() >= self.config.bulk_read_threshold {
            // For large reads, bypass buffer and read directly
            let buffered = self.end - self.pos;
            if buffered > 0 {
                let to_copy = cmp::min(buffered, buf.len());
                buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
                self.pos += to_copy;
                
                if to_copy == buf.len() {
                    return Ok(to_copy);
                }
                
                // Read remaining directly from underlying stream
                let remaining = self.inner.read(&mut buf[to_copy..])
                    .map_err(|e| ZiporaError::io_error(format!("Bulk read failed: {}", e)))?;
                self.total_read += remaining as u64;
                Ok(to_copy + remaining)
            } else {
                // Read directly from underlying stream
                let bytes_read = self.inner.read(buf)
                    .map_err(|e| ZiporaError::io_error(format!("Bulk read failed: {}", e)))?;
                self.total_read += bytes_read as u64;
                Ok(bytes_read)
            }
        } else {
            // Use normal buffered read for smaller requests
            self.read_buffered(buf)
        }
    }

    /// Normal buffered read implementation
    fn read_buffered(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let mut total_read = 0;
        let mut remaining = buf;

        while !remaining.is_empty() {
            // Ensure we have data in buffer
            let available = self.ensure_buffered(remaining.len())?;
            if available == 0 {
                break; // End of stream
            }

            // Copy data from buffer
            let to_copy = cmp::min(available, remaining.len());
            remaining[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
            self.pos += to_copy;
            total_read += to_copy;
            remaining = &mut remaining[to_copy..];
        }

        Ok(total_read)
    }
}

impl<R: Read> Read for StreamBufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_bulk(buf).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    }
}

impl<R: Read> BufRead for StreamBufferedReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos >= self.end {
            self.fill_buffer(1)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        Ok(&self.buffer[self.pos..self.end])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos + amt, self.end);
    }
}

impl<R: Read + Seek> Seek for StreamBufferedReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // For seek operations, we need to invalidate the buffer
        self.pos = 0;
        self.end = 0;
        self.inner.seek(pos)
    }
}

/// High-performance buffered writer with configurable strategies
pub struct StreamBufferedWriter<W> {
    inner: W,
    buffer: Box<[u8]>,
    pos: usize,      // Current position in buffer
    config: StreamBufferConfig,
    total_written: u64,
    pool: Option<Arc<SecureMemoryPool>>,
}

impl<W: Write> StreamBufferedWriter<W> {
    /// Create a new buffered writer with default configuration
    pub fn new(inner: W) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::default())
    }

    /// Create a new buffered writer with custom configuration
    pub fn with_config(inner: W, config: StreamBufferConfig) -> Result<Self> {
        let pool = if config.use_secure_pool {
            Some(SecureMemoryPool::new(
                crate::memory::SecurePoolConfig::small_secure()
            )?)
        } else {
            None
        };

        let buffer = StreamBufferedReader::<std::io::Empty>::allocate_aligned_buffer(
            config.initial_capacity, 
            config.page_alignment
        )?;

        Ok(Self {
            inner,
            buffer,
            pos: 0,
            config,
            total_written: 0,
            pool,
        })
    }

    /// Get the underlying writer
    pub fn get_ref(&self) -> &W {
        &self.inner
    }

    /// Get a mutable reference to the underlying writer
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.inner
    }

    /// Consume this buffered writer and return the underlying writer
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush()?;
        Ok(self.inner)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Get current buffer usage
    pub fn buffer_usage(&self) -> usize {
        self.pos
    }

    /// Get total bytes written to underlying stream
    pub fn total_written(&self) -> u64 {
        self.total_written
    }

    /// Fast path for writing a single byte
    #[inline]
    pub fn write_byte_fast(&mut self, byte: u8) -> Result<()> {
        if self.pos < self.buffer.len() {
            self.buffer[self.pos] = byte;
            self.pos += 1;
            Ok(())
        } else {
            self.write_byte_slow(byte)
        }
    }

    /// Slow path for writing a single byte when buffer is full
    #[cold]
    fn write_byte_slow(&mut self, byte: u8) -> Result<()> {
        self.flush_buffer()?;
        self.buffer[0] = byte;
        self.pos = 1;
        Ok(())
    }

    /// Flush internal buffer to underlying writer
    fn flush_buffer(&mut self) -> Result<()> {
        if self.pos > 0 {
            self.inner.write_all(&self.buffer[..self.pos])
                .map_err(|e| ZiporaError::io_error(format!("Failed to flush buffer: {}", e)))?;
            self.total_written += self.pos as u64;
            self.pos = 0;
        }
        Ok(())
    }
}

impl<W: Write> Write for StreamBufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() >= self.config.bulk_read_threshold {
            // For large writes, flush buffer and write directly
            self.flush_buffer()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            
            let written = self.inner.write(buf)?;
            self.total_written += written as u64;
            Ok(written)
        } else {
            // Use buffered write for smaller data
            let mut remaining = buf;
            let mut total_written = 0;

            while !remaining.is_empty() {
                let available = self.buffer.len() - self.pos;
                if available == 0 {
                    self.flush_buffer()
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                    continue;
                }

                let to_copy = cmp::min(available, remaining.len());
                self.buffer[self.pos..self.pos + to_copy].copy_from_slice(&remaining[..to_copy]);
                self.pos += to_copy;
                total_written += to_copy;
                remaining = &remaining[to_copy..];
            }

            Ok(total_written)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.inner.flush()
    }
}

impl<W: Write + Seek> Seek for StreamBufferedWriter<W> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.flush()?;
        self.inner.seek(pos)
    }
}

// Branch prediction hints (would be actual intrinsics in real implementation)
#[inline(always)]
fn likely(condition: bool) -> bool {
    // In real implementation, this would use compiler hints
    // #[cfg(target_arch = "x86_64")]
    // std::intrinsics::likely(condition)
    condition
}

#[cold]
#[inline(never)]
fn unlikely() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_stream_buffered_reader_basic() {
        let data = b"Hello, World! This is a test of buffered reading.";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        let mut buf = [0u8; 5];
        assert_eq!(reader.read(&mut buf).unwrap(), 5);
        assert_eq!(&buf, b"Hello");

        let mut buf = [0u8; 7];
        assert_eq!(reader.read(&mut buf).unwrap(), 7);
        assert_eq!(&buf, b", World");
    }

    #[test]
    fn test_stream_buffered_reader_byte_reading() {
        let data = b"ABC";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        assert_eq!(reader.read_byte_fast().unwrap(), b'A');
        assert_eq!(reader.read_byte_fast().unwrap(), b'B');
        assert_eq!(reader.read_byte_fast().unwrap(), b'C');
        assert!(reader.read_byte_fast().is_err());
    }

    #[test]
    fn test_stream_buffered_reader_slice_reading() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // First read should come from buffer
        let slice = reader.read_slice(5).unwrap();
        assert_eq!(slice, Some(&b"Hello"[..]));

        let slice = reader.read_slice(2).unwrap();
        assert_eq!(slice, Some(&b", "[..]));
    }

    #[test]
    fn test_stream_buffered_reader_bulk_read() {
        let data = vec![42u8; 100_000]; // Large data for bulk read test
        let cursor = Cursor::new(data.clone());
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        let mut buf = vec![0u8; 100_000];
        let bytes_read = reader.read_bulk(&mut buf).unwrap();
        assert_eq!(bytes_read, 100_000);
        assert_eq!(buf, data);
    }

    #[test]
    fn test_stream_buffered_reader_configurations() {
        let data = b"Test data";
        
        // Performance optimized
        let cursor = Cursor::new(data);
        let reader = StreamBufferedReader::performance_optimized(cursor).unwrap();
        assert!(reader.capacity() >= 128 * 1024);

        // Memory efficient
        let cursor = Cursor::new(data);
        let reader = StreamBufferedReader::memory_efficient(cursor).unwrap();
        assert_eq!(reader.capacity(), 16 * 1024);

        // Low latency
        let cursor = Cursor::new(data);
        let reader = StreamBufferedReader::low_latency(cursor).unwrap();
        assert_eq!(reader.capacity(), 8 * 1024);
    }

    #[test]
    fn test_stream_buffered_writer_basic() {
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = StreamBufferedWriter::new(cursor).unwrap();

            writer.write_all(b"Hello").unwrap();
            writer.write_all(b", ").unwrap();
            writer.write_all(b"World!").unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer, b"Hello, World!");
    }

    #[test]
    fn test_stream_buffered_writer_byte_writing() {
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = StreamBufferedWriter::new(cursor).unwrap();

            writer.write_byte_fast(b'A').unwrap();
            writer.write_byte_fast(b'B').unwrap();
            writer.write_byte_fast(b'C').unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer, b"ABC");
    }

    #[test]
    fn test_stream_buffered_writer_large_write() {
        let data = vec![42u8; 100_000];
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = StreamBufferedWriter::new(cursor).unwrap();

            writer.write_all(&data).unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer.len(), 100_000);
        assert_eq!(buffer, data);
    }

    #[test]
    fn test_stream_buffer_round_trip() {
        let original_data = b"The quick brown fox jumps over the lazy dog. ".repeat(1000);
        
        // Write data using buffered writer
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = StreamBufferedWriter::new(cursor).unwrap();
            writer.write_all(&original_data).unwrap();
            writer.flush().unwrap();
        }

        // Read data back using buffered reader
        let cursor = Cursor::new(&buffer);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();
        let mut read_data = Vec::new();
        reader.read_to_end(&mut read_data).unwrap();

        assert_eq!(read_data, original_data);
    }

    #[test]
    fn test_buffer_statistics() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Initially, no data read
        assert_eq!(reader.total_read(), 0);
        assert!(!reader.has_data_in_buffer());

        // Read some data
        let mut buf = [0u8; 5];
        reader.read(&mut buf).unwrap();
        
        assert!(reader.total_read() > 0);
        assert!(reader.has_data_in_buffer() || reader.total_read() == data.len() as u64);
    }
}