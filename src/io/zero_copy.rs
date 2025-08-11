//! Zero-copy stream optimizations for high-performance I/O
//!
//! This module provides advanced zero-copy operations that minimize data movement
//! between different layers of the I/O stack. Features include direct buffer access,
//! memory-mapped operations, and vectored I/O for maximum throughput.

use std::io::{self, Read, Write, IoSlice, IoSliceMut};
use std::ptr::NonNull;
use std::marker::PhantomData;
use std::slice;

use crate::error::{Result, ZiporaError};
use crate::memory::SecureMemoryPool;
use std::sync::Arc;

/// Trait for zero-copy input operations
pub trait ZeroCopyRead {
    /// Get a direct reference to buffered data without copying
    ///
    /// Returns `None` if the requested amount of data is not available
    /// in the buffer, or if zero-copy access is not possible.
    fn zc_read(&mut self, len: usize) -> Result<Option<&[u8]>>;

    /// Advance the read position by the specified number of bytes
    ///
    /// This should be called after processing data obtained from `zc_read`
    fn zc_advance(&mut self, len: usize) -> Result<()>;

    /// Get the total amount of data available for zero-copy reading
    fn zc_available(&self) -> usize;

    /// Try to ensure at least `len` bytes are available for zero-copy reading
    fn zc_ensure(&mut self, len: usize) -> Result<usize>;
}

/// Trait for zero-copy output operations
pub trait ZeroCopyWrite {
    /// Get a direct mutable reference to write buffer
    ///
    /// Returns `None` if the requested buffer size is not available
    /// or if zero-copy access is not possible.
    fn zc_write(&mut self, len: usize) -> Result<Option<&mut [u8]>>;

    /// Commit the specified number of bytes that were written to the buffer
    ///
    /// This should be called after writing data to buffer obtained from `zc_write`
    fn zc_commit(&mut self, len: usize) -> Result<()>;

    /// Get the total amount of buffer space available for zero-copy writing
    fn zc_write_available(&self) -> usize;

    /// Try to ensure at least `len` bytes are available for zero-copy writing
    fn zc_ensure_write(&mut self, len: usize) -> Result<usize>;
}

/// Zero-copy buffer for high-performance I/O operations
pub struct ZeroCopyBuffer {
    data: NonNull<u8>,
    capacity: usize,
    read_pos: usize,
    write_pos: usize,
    pool: Option<Arc<SecureMemoryPool>>,
    _phantom: PhantomData<u8>,
}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer with the specified capacity
    pub fn new(capacity: usize) -> Result<Self> {
        Self::with_pool(capacity, None)
    }

    /// Create a new zero-copy buffer using a secure memory pool
    pub fn with_secure_pool(capacity: usize) -> Result<Self> {
        let pool = SecureMemoryPool::new(
            crate::memory::SecurePoolConfig::small_secure()
        )?;
        Self::with_pool(capacity, Some(pool))
    }

    /// Create a buffer with an optional memory pool
    fn with_pool(capacity: usize, pool: Option<Arc<SecureMemoryPool>>) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(capacity, 64) // 64-byte alignment for SIMD
            .map_err(|e| ZiporaError::io_error(format!("Invalid buffer layout: {}", e)))?;

        let data = unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                return Err(ZiporaError::io_error("Failed to allocate zero-copy buffer"));
            }
            NonNull::new_unchecked(ptr)
        };

        Ok(Self {
            data,
            capacity,
            read_pos: 0,
            write_pos: 0,
            pool,
            _phantom: PhantomData,
        })
    }

    /// Get the buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of bytes available for reading
    pub fn available(&self) -> usize {
        self.write_pos - self.read_pos
    }

    /// Get the number of bytes available for writing
    pub fn write_available(&self) -> usize {
        self.capacity - self.write_pos
    }

    /// Get the current read position
    pub fn read_position(&self) -> usize {
        self.read_pos
    }

    /// Get the current write position
    pub fn write_position(&self) -> usize {
        self.write_pos
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.write_pos == self.capacity
    }

    /// Reset the buffer to empty state
    pub fn reset(&mut self) {
        self.read_pos = 0;
        self.write_pos = 0;
    }

    /// Compact the buffer by moving unread data to the beginning
    pub fn compact(&mut self) {
        if self.read_pos > 0 {
            let available = self.available();
            if available > 0 {
                unsafe {
                    std::ptr::copy(
                        self.data.as_ptr().add(self.read_pos),
                        self.data.as_ptr(),
                        available,
                    );
                }
            }
            self.read_pos = 0;
            self.write_pos = available;
        }
    }

    /// Get a slice of the readable data
    pub fn readable_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self.data.as_ptr().add(self.read_pos),
                self.available(),
            )
        }
    }

    /// Get a mutable slice of the writable space
    pub fn writable_slice(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(
                self.data.as_ptr().add(self.write_pos),
                self.write_available(),
            )
        }
    }

    /// Fill the buffer from a reader, returning the number of bytes read
    pub fn fill_from<R: Read>(&mut self, reader: &mut R) -> Result<usize> {
        if self.is_full() {
            self.compact();
        }

        let writable = self.writable_slice();
        if writable.is_empty() {
            return Ok(0);
        }

        let bytes_read = reader.read(writable)
            .map_err(|e| ZiporaError::io_error(format!("Failed to fill buffer: {}", e)))?;
        
        self.write_pos += bytes_read;
        Ok(bytes_read)
    }

    /// Drain the buffer to a writer, returning the number of bytes written
    pub fn drain_to<W: Write>(&mut self, writer: &mut W) -> Result<usize> {
        let readable = self.readable_slice();
        if readable.is_empty() {
            return Ok(0);
        }

        let bytes_written = writer.write(readable)
            .map_err(|e| ZiporaError::io_error(format!("Failed to drain buffer: {}", e)))?;
        
        self.read_pos += bytes_written;
        Ok(bytes_written)
    }
}

impl ZeroCopyRead for ZeroCopyBuffer {
    fn zc_read(&mut self, len: usize) -> Result<Option<&[u8]>> {
        if self.available() >= len {
            let slice = unsafe {
                slice::from_raw_parts(
                    self.data.as_ptr().add(self.read_pos),
                    len,
                )
            };
            Ok(Some(slice))
        } else {
            Ok(None)
        }
    }

    fn zc_advance(&mut self, len: usize) -> Result<()> {
        if self.read_pos + len > self.write_pos {
            return Err(ZiporaError::invalid_data("Cannot advance past available data"));
        }
        self.read_pos += len;
        Ok(())
    }

    fn zc_available(&self) -> usize {
        self.available()
    }

    fn zc_ensure(&mut self, len: usize) -> Result<usize> {
        Ok(self.available().min(len))
    }
}

impl ZeroCopyWrite for ZeroCopyBuffer {
    fn zc_write(&mut self, len: usize) -> Result<Option<&mut [u8]>> {
        if self.write_available() >= len {
            let slice = unsafe {
                slice::from_raw_parts_mut(
                    self.data.as_ptr().add(self.write_pos),
                    len,
                )
            };
            Ok(Some(slice))
        } else {
            Ok(None)
        }
    }

    fn zc_commit(&mut self, len: usize) -> Result<()> {
        if self.write_pos + len > self.capacity {
            return Err(ZiporaError::invalid_data("Cannot commit past buffer capacity"));
        }
        self.write_pos += len;
        Ok(())
    }

    fn zc_write_available(&self) -> usize {
        self.write_available()
    }

    fn zc_ensure_write(&mut self, len: usize) -> Result<usize> {
        if self.write_available() < len {
            self.compact();
        }
        Ok(self.write_available().min(len))
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(self.capacity, 64);
            std::alloc::dealloc(self.data.as_ptr(), layout);
        }
    }
}

// SAFETY: ZeroCopyBuffer manages its own memory correctly
unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

/// Zero-copy reader wrapper that provides direct buffer access
pub struct ZeroCopyReader<R> {
    inner: R,
    buffer: ZeroCopyBuffer,
    eof: bool,
}

impl<R: Read> ZeroCopyReader<R> {
    /// Create a new zero-copy reader with default buffer size (64KB)
    pub fn new(inner: R) -> Result<Self> {
        Self::with_capacity(inner, 64 * 1024)
    }

    /// Create a new zero-copy reader with specified buffer capacity
    pub fn with_capacity(inner: R, capacity: usize) -> Result<Self> {
        Ok(Self {
            inner,
            buffer: ZeroCopyBuffer::new(capacity)?,
            eof: false,
        })
    }

    /// Create a new zero-copy reader with secure memory pool
    pub fn with_secure_buffer(inner: R, capacity: usize) -> Result<Self> {
        Ok(Self {
            inner,
            buffer: ZeroCopyBuffer::with_secure_pool(capacity)?,
            eof: false,
        })
    }

    /// Get the underlying reader
    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    /// Get a mutable reference to the underlying reader
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// Consume this reader and return the underlying reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Ensure the buffer has at least `len` bytes available
    fn ensure_buffered(&mut self, len: usize) -> Result<()> {
        while self.buffer.available() < len && !self.eof {
            let bytes_read = self.buffer.fill_from(&mut self.inner)
                .map_err(|e| ZiporaError::io_error(format!("Fill buffer failed: {}", e)))?;
            if bytes_read == 0 {
                self.eof = true;
                break;
            }
        }
        Ok(())
    }

    /// Read data with zero-copy optimization when possible
    pub fn read_optimized(&mut self, buf: &mut [u8]) -> Result<usize> {
        // Try zero-copy first
        if let Some(zc_data) = self.zc_read(buf.len())? {
            let len = zc_data.len();
            buf[..len].copy_from_slice(zc_data);
            self.zc_advance(len)?;
            return Ok(len);
        }

        // Fall back to normal read
        self.read(buf).map_err(|e| ZiporaError::io_error(format!("Read failed: {}", e)))
    }

    /// Peek at data without consuming it
    pub fn peek(&mut self, len: usize) -> Result<&[u8]> {
        self.ensure_buffered(len)?;
        let available = self.buffer.available().min(len);
        Ok(&self.buffer.readable_slice()[..available])
    }

    /// Skip the specified number of bytes efficiently
    pub fn skip_bytes(&mut self, mut len: usize) -> Result<()> {
        // First, skip from buffer if available
        let buffered = self.buffer.available().min(len);
        if buffered > 0 {
            self.buffer.zc_advance(buffered)?;
            len -= buffered;
        }

        // Skip remaining bytes by reading into a temporary buffer
        let mut temp_buf = vec![0u8; 8192];
        while len > 0 {
            let to_skip = len.min(temp_buf.len());
            let bytes_read = self.inner.read(&mut temp_buf[..to_skip])
                .map_err(|e| ZiporaError::io_error(format!("Failed to skip bytes: {}", e)))?;
            
            if bytes_read == 0 {
                return Err(ZiporaError::io_error("Unexpected end of stream while skipping"));
            }
            
            len -= bytes_read;
        }

        Ok(())
    }
}

impl<R: Read> Read for ZeroCopyReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // If buffer has data, use it first
        let buffered = self.buffer.available();
        if buffered > 0 {
            let to_copy = buffered.min(buf.len());
            buf[..to_copy].copy_from_slice(&self.buffer.readable_slice()[..to_copy]);
            self.buffer.read_pos += to_copy;
            return Ok(to_copy);
        }

        // For large reads, bypass buffer
        if buf.len() >= self.buffer.capacity() / 2 {
            return self.inner.read(buf);
        }

        // Fill buffer and read from it
        if !self.eof {
            let bytes_read = self.buffer.fill_from(&mut self.inner)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            if bytes_read == 0 {
                self.eof = true;
            }
        }

        let available = self.buffer.available();
        if available == 0 {
            return Ok(0);
        }

        let to_copy = available.min(buf.len());
        buf[..to_copy].copy_from_slice(&self.buffer.readable_slice()[..to_copy]);
        self.buffer.read_pos += to_copy;
        Ok(to_copy)
    }
}

impl<R: Read> ZeroCopyRead for ZeroCopyReader<R> {
    fn zc_read(&mut self, len: usize) -> Result<Option<&[u8]>> {
        self.ensure_buffered(len)?;
        self.buffer.zc_read(len)
    }

    fn zc_advance(&mut self, len: usize) -> Result<()> {
        self.buffer.zc_advance(len)
    }

    fn zc_available(&self) -> usize {
        self.buffer.zc_available()
    }

    fn zc_ensure(&mut self, len: usize) -> Result<usize> {
        self.ensure_buffered(len)?;
        Ok(self.buffer.available().min(len))
    }
}

/// Zero-copy writer wrapper that provides direct buffer access
pub struct ZeroCopyWriter<W> {
    inner: W,
    buffer: ZeroCopyBuffer,
}

impl<W: Write> ZeroCopyWriter<W> {
    /// Create a new zero-copy writer with default buffer size (64KB)
    pub fn new(inner: W) -> Result<Self> {
        Self::with_capacity(inner, 64 * 1024)
    }

    /// Create a new zero-copy writer with specified buffer capacity
    pub fn with_capacity(inner: W, capacity: usize) -> Result<Self> {
        Ok(Self {
            inner,
            buffer: ZeroCopyBuffer::new(capacity)?,
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

    /// Consume this writer and return the underlying writer
    pub fn into_inner(mut self) -> io::Result<W> {
        self.flush()?;
        Ok(self.inner)
    }

    /// Flush the internal buffer to the underlying writer
    fn flush_buffer(&mut self) -> io::Result<()> {
        while !self.buffer.is_empty() {
            let bytes_written = self.buffer.drain_to(&mut self.inner)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            if bytes_written == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "Failed to drain buffer completely"));
            }
        }
        self.buffer.reset();
        Ok(())
    }
}

impl<W: Write> Write for ZeroCopyWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // For large writes, flush buffer and write directly
        if buf.len() >= self.buffer.capacity() / 2 {
            self.flush_buffer()?;
            return self.inner.write(buf);
        }

        // Try to fit in buffer
        if self.buffer.write_available() < buf.len() {
            self.flush_buffer()?;
        }

        // If still doesn't fit, write directly
        if self.buffer.write_available() < buf.len() {
            return self.inner.write(buf);
        }

        // Write to buffer
        let writable = self.buffer.writable_slice();
        let to_copy = buf.len().min(writable.len());
        writable[..to_copy].copy_from_slice(&buf[..to_copy]);
        self.buffer.write_pos += to_copy;
        Ok(to_copy)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()?;
        self.inner.flush()
    }
}

impl<W: Write> ZeroCopyWrite for ZeroCopyWriter<W> {
    fn zc_write(&mut self, len: usize) -> Result<Option<&mut [u8]>> {
        if self.buffer.write_available() < len {
            self.flush_buffer()
                .map_err(|e| ZiporaError::io_error(format!("Flush failed: {}", e)))?;
        }
        self.buffer.zc_write(len)
    }

    fn zc_commit(&mut self, len: usize) -> Result<()> {
        self.buffer.zc_commit(len)
    }

    fn zc_write_available(&self) -> usize {
        self.buffer.zc_write_available()
    }

    fn zc_ensure_write(&mut self, len: usize) -> Result<usize> {
        if self.buffer.write_available() < len {
            self.flush_buffer()
                .map_err(|e| ZiporaError::io_error(format!("Flush failed: {}", e)))?;
        }
        Ok(self.buffer.write_available().min(len))
    }
}

/// Vectored I/O operations for efficient bulk transfers
pub struct VectoredIO;

impl VectoredIO {
    /// Read data into multiple buffers in a single system call
    pub fn read_vectored<R: Read>(reader: &mut R, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // Fallback implementation for readers that don't support vectored I/O
        let mut total = 0;
        for buf in bufs {
            if buf.is_empty() {
                continue;
            }
            match reader.read(buf) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
            }
        }
        Ok(total)
    }

    /// Write data from multiple buffers in a single system call
    pub fn write_vectored<W: Write>(writer: &mut W, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        // Fallback implementation for writers that don't support vectored I/O
        let mut total = 0;
        for buf in bufs {
            if buf.is_empty() {
                continue;
            }
            match writer.write(buf) {
                Ok(n) => total += n,
                Err(e) => return if total > 0 { Ok(total) } else { Err(e) },
            }
        }
        Ok(total)
    }
}

/// Memory-mapped zero-copy operations
#[cfg(feature = "mmap")]
pub mod mmap {
    use super::*;
    use memmap2::Mmap;
    use std::fs::File;

    /// Zero-copy reader for memory-mapped files
    pub struct MmapZeroCopyReader {
        mmap: Mmap,
        pos: usize,
    }

    impl MmapZeroCopyReader {
        /// Create a new memory-mapped zero-copy reader
        pub fn new(file: File) -> Result<Self> {
            let mmap = unsafe {
                Mmap::map(&file)
                    .map_err(|e| ZiporaError::io_error(format!("Failed to memory map file: {}", e)))?
            };

            Ok(Self { mmap, pos: 0 })
        }

        /// Get the total size of the mapped region
        pub fn len(&self) -> usize {
            self.mmap.len()
        }

        /// Check if the mapped region is empty
        pub fn is_empty(&self) -> bool {
            self.mmap.is_empty()
        }

        /// Get the current position
        pub fn position(&self) -> usize {
            self.pos
        }

        /// Set the position within the mapped region
        pub fn set_position(&mut self, pos: usize) -> Result<()> {
            if pos > self.mmap.len() {
                return Err(ZiporaError::invalid_data("Position beyond mapped region"));
            }
            self.pos = pos;
            Ok(())
        }

        /// Get the entire mapped slice
        pub fn as_slice(&self) -> &[u8] {
            &self.mmap
        }

        /// Get a slice from the current position
        pub fn remaining_slice(&self) -> &[u8] {
            &self.mmap[self.pos..]
        }
    }

    impl ZeroCopyRead for MmapZeroCopyReader {
        fn zc_read(&mut self, len: usize) -> Result<Option<&[u8]>> {
            if self.pos + len <= self.mmap.len() {
                let slice = &self.mmap[self.pos..self.pos + len];
                Ok(Some(slice))
            } else {
                Ok(None)
            }
        }

        fn zc_advance(&mut self, len: usize) -> Result<()> {
            if self.pos + len > self.mmap.len() {
                return Err(ZiporaError::invalid_data("Cannot advance past mapped region"));
            }
            self.pos += len;
            Ok(())
        }

        fn zc_available(&self) -> usize {
            self.mmap.len() - self.pos
        }

        fn zc_ensure(&mut self, len: usize) -> Result<usize> {
            Ok(self.zc_available().min(len))
        }
    }

    impl Read for MmapZeroCopyReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let available = self.zc_available();
            if available == 0 {
                return Ok(0);
            }

            let to_copy = available.min(buf.len());
            buf[..to_copy].copy_from_slice(&self.mmap[self.pos..self.pos + to_copy]);
            self.pos += to_copy;
            Ok(to_copy)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_zero_copy_buffer_basic() {
        let mut buffer = ZeroCopyBuffer::new(1024).unwrap();
        
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.available(), 0);
        assert_eq!(buffer.write_available(), 1024);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_zero_copy_buffer_read_write() {
        let mut buffer = ZeroCopyBuffer::new(1024).unwrap();
        
        // Write some data
        if let Some(write_buf) = buffer.zc_write(5).unwrap() {
            write_buf.copy_from_slice(b"hello");
            buffer.zc_commit(5).unwrap();
        }

        assert_eq!(buffer.available(), 5);
        assert_eq!(buffer.write_available(), 1019);

        // Read the data
        if let Some(read_buf) = buffer.zc_read(5).unwrap() {
            assert_eq!(read_buf, b"hello");
            buffer.zc_advance(5).unwrap();
        }

        assert_eq!(buffer.available(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_zero_copy_buffer_compact() {
        let mut buffer = ZeroCopyBuffer::new(10).unwrap();
        
        // Fill buffer
        if let Some(write_buf) = buffer.zc_write(10).unwrap() {
            write_buf.copy_from_slice(b"0123456789");
            buffer.zc_commit(10).unwrap();
        }

        // Read half
        buffer.zc_advance(5).unwrap();
        assert_eq!(buffer.available(), 5);
        assert_eq!(buffer.write_available(), 0);

        // Compact
        buffer.compact();
        assert_eq!(buffer.available(), 5);
        assert_eq!(buffer.write_available(), 5);
        assert_eq!(buffer.readable_slice(), b"56789");
    }

    #[test]
    fn test_zero_copy_reader() {
        let data = b"Hello, World! This is a test of zero-copy reading.";
        let cursor = Cursor::new(data);
        let mut reader = ZeroCopyReader::new(cursor).unwrap();

        // Test zero-copy read
        if let Some(zc_data) = reader.zc_read(5).unwrap() {
            assert_eq!(zc_data, b"Hello");
            reader.zc_advance(5).unwrap();
        }

        // Test peek
        let peeked = reader.peek(7).unwrap();
        assert_eq!(peeked, b", World");

        // Test regular read
        let mut buf = [0u8; 7];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b", World");
    }

    #[test]
    fn test_zero_copy_writer() {
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = ZeroCopyWriter::new(cursor).unwrap();

            // Test zero-copy write
            if let Some(zc_buf) = writer.zc_write(5).unwrap() {
                zc_buf.copy_from_slice(b"Hello");
                writer.zc_commit(5).unwrap();
            }

            // Test regular write
            writer.write_all(b", World!").unwrap();
            writer.flush().unwrap();
        }

        assert_eq!(buffer, b"Hello, World!");
    }

    #[test]
    fn test_zero_copy_reader_skip() {
        let data = b"Hello, World! This is a test.";
        let cursor = Cursor::new(data);
        let mut reader = ZeroCopyReader::new(cursor).unwrap();

        reader.skip_bytes(7).unwrap(); // Skip "Hello, "
        
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"World");
    }

    #[test]
    fn test_vectored_io() {
        let data = b"Hello, World!";
        let mut cursor = Cursor::new(data);

        let mut buf1 = [0u8; 5];
        let mut buf2 = [0u8; 2];
        let mut buf3 = [0u8; 6];
        
        let mut bufs = [
            IoSliceMut::new(&mut buf1),
            IoSliceMut::new(&mut buf2),
            IoSliceMut::new(&mut buf3),
        ];

        let bytes_read = VectoredIO::read_vectored(&mut cursor, &mut bufs).unwrap();
        assert_eq!(bytes_read, 13);
        assert_eq!(&buf1, b"Hello");
        assert_eq!(&buf2, b", ");
        assert_eq!(&buf3, b"World!");
    }

    #[test]
    fn test_zero_copy_round_trip() {
        let original_data = b"The quick brown fox jumps over the lazy dog.";
        
        // Write using zero-copy writer
        let mut buffer = Vec::new();
        {
            let cursor = Cursor::new(&mut buffer);
            let mut writer = ZeroCopyWriter::new(cursor).unwrap();
            writer.write_all(original_data).unwrap();
            writer.flush().unwrap();
        }

        // Read using zero-copy reader
        let cursor = Cursor::new(&buffer);
        let mut reader = ZeroCopyReader::new(cursor).unwrap();
        let mut read_data = Vec::new();
        reader.read_to_end(&mut read_data).unwrap();

        assert_eq!(read_data, original_data);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_mmap_zero_copy_reader() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut temp_file = NamedTempFile::new().unwrap();
        let data = b"Hello, memory mapped zero-copy world!";
        temp_file.write_all(data).unwrap();
        temp_file.flush().unwrap();

        let file = temp_file.reopen().unwrap();
        let mut reader = mmap::MmapZeroCopyReader::new(file).unwrap();

        assert_eq!(reader.len(), data.len());
        assert_eq!(reader.as_slice(), data);

        // Test zero-copy read
        if let Some(zc_data) = reader.zc_read(5).unwrap() {
            assert_eq!(zc_data, b"Hello");
            reader.zc_advance(5).unwrap();
        }

        assert_eq!(reader.position(), 5);
        assert_eq!(reader.remaining_slice(), &data[5..]);
    }
}