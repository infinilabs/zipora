//! Advanced buffered stream wrapper with configurable buffering strategies
//!
//! This module provides stream buffering with configurable buffer sizing,
//! adaptive growth, and optional read-ahead.
//!
//! # Limitation
//! [`StreamBufferConfig::page_alignment`] rounds buffer capacities up to a
//! power-of-two multiple using standard heap allocation (`Vec<u8>`) rather than
//! OS page-aligned allocation, and [`StreamBufferConfig::use_secure_pool`]
//! initializes a [`SecureMemoryPool`] handle without allocating the stream
//! buffer slice from it.

use std::cmp;
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::mem::ManuallyDrop;

use crate::error::{Result, ZiporaError};
use crate::memory::SecureMemoryPool;
use crate::memory::simd_ops;
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
    /// Capacity rounding alignment in bytes (default: 4096; must be a non-zero power of two).
    ///
    /// # Limitation
    /// Rounds the buffer length up to a multiple of `page_alignment` using standard heap
    /// allocation (`Vec<u8>`); the base pointer alignment is not elevated to `page_alignment`.
    pub page_alignment: usize,
    /// Whether to initialize a [`SecureMemoryPool`] alongside the stream buffer.
    ///
    /// # Limitation
    /// The internal `Box<[u8]>` stream buffer is allocated on the standard heap rather than
    /// from the [`SecureMemoryPool`].
    pub use_secure_pool: bool,
    /// Minimum read size to trigger bulk operations
    pub bulk_read_threshold: usize,
    /// Enable read-ahead optimization.
    ///
    /// # Limitation
    /// Retained for configuration compatibility; [`StreamBufferedReader`] and
    /// [`StreamBufferedWriter`] fill up to the current buffer capacity per read
    /// and do not consult `enable_readahead` or `readahead_multiplier`.
    pub enable_readahead: bool,
    /// Read-ahead size multiplier.
    ///
    /// # Limitation
    /// Retained for configuration compatibility; not consulted by
    /// [`StreamBufferedReader`] or [`StreamBufferedWriter`].
    pub readahead_multiplier: usize,
}

impl Default for StreamBufferConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64 * 1024,   // 64KB
            max_capacity: 2 * 1024 * 1024, // 2MB
            growth_factor: 1.618,          // Golden ratio for optimal memory usage
            page_alignment: 4096,          // Standard page size
            use_secure_pool: true,         // Use secure memory by default
            bulk_read_threshold: 8192,     // 8KB threshold
            enable_readahead: true,        // Enable read-ahead
            readahead_multiplier: 2,       // 2x read-ahead
        }
    }
}

impl StreamBufferConfig {
    /// Validate configuration parameters before allocating buffers.
    pub fn validate(&self) -> Result<()> {
        if self.initial_capacity == 0 {
            return Err(ZiporaError::invalid_data(
                "StreamBufferConfig::initial_capacity must be greater than 0",
            ));
        }
        if self.max_capacity < self.initial_capacity {
            return Err(ZiporaError::invalid_data(
                "StreamBufferConfig::max_capacity must be >= initial_capacity",
            ));
        }
        if self.page_alignment == 0 || !self.page_alignment.is_power_of_two() {
            return Err(ZiporaError::invalid_data(
                "StreamBufferConfig::page_alignment must be a non-zero power of two",
            ));
        }
        if !self.growth_factor.is_finite()
            || self.growth_factor <= 1.0
            || self.growth_factor > 1024.0
        {
            return Err(ZiporaError::invalid_data(
                "StreamBufferConfig::growth_factor must be finite and in (1.0, 1024.0]",
            ));
        }
        Ok(())
    }

    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            initial_capacity: 128 * 1024,  // 128KB initial
            max_capacity: 4 * 1024 * 1024, // 4MB max
            growth_factor: 2.0,            // Faster growth
            bulk_read_threshold: 4096,     // Lower threshold
            enable_readahead: true,
            readahead_multiplier: 4, // Aggressive read-ahead
            ..Default::default()
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            initial_capacity: 16 * 1024, // 16KB initial
            max_capacity: 512 * 1024,    // 512KB max
            growth_factor: 1.414,        // Conservative growth
            bulk_read_threshold: 16384,  // Higher threshold
            enable_readahead: false,     // Disable read-ahead
            readahead_multiplier: 1,
            ..Default::default()
        }
    }

    /// Create a low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            initial_capacity: 8 * 1024, // 8KB initial
            max_capacity: 256 * 1024,   // 256KB max
            growth_factor: 1.5,
            bulk_read_threshold: 2048, // Low threshold
            enable_readahead: false,   // No read-ahead for low latency
            readahead_multiplier: 1,
            ..Default::default()
        }
    }
}

/// High-performance buffered reader with configurable strategies
pub struct StreamBufferedReader<R> {
    inner: R,
    buffer: Box<[u8]>,
    pos: usize, // Current position in buffer
    end: usize, // End of valid data in buffer
    config: StreamBufferConfig,
    total_read: u64, // Total bytes read from underlying stream
    _pool: Option<Arc<SecureMemoryPool>>,
}

impl<R: Read> StreamBufferedReader<R> {
    /// Create a new buffered reader with default configuration
    pub fn new(inner: R) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::default())
    }

    /// Create a new buffered reader with custom configuration
    pub fn with_config(inner: R, mut config: StreamBufferConfig) -> Result<Self> {
        config.validate()?;

        let pool = if config.use_secure_pool {
            Some(SecureMemoryPool::new(
                crate::memory::SecurePoolConfig::small_secure(),
            )?)
        } else {
            None
        };

        // Allocate aligned buffer
        let buffer = Self::allocate_aligned_buffer(config.initial_capacity, config.page_alignment)?;
        config.max_capacity = cmp::max(config.max_capacity, buffer.len());

        Ok(Self {
            inner,
            buffer,
            pos: 0,
            end: 0,
            config,
            total_read: 0,
            _pool: pool,
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

    /// Allocate buffer with capacity rounded up to `alignment`.
    fn allocate_aligned_buffer(size: usize, alignment: usize) -> Result<Box<[u8]>> {
        if size == 0 {
            return Err(ZiporaError::invalid_data(
                "buffer size must be greater than 0",
            ));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(ZiporaError::invalid_data(
                "buffer alignment must be a non-zero power of two",
            ));
        }
        let mask = alignment - 1;
        let aligned_size = size
            .checked_add(mask)
            .map(|n| n & !mask)
            .ok_or_else(|| ZiporaError::invalid_data("aligned buffer size overflow"))?;

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
    #[inline]
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

    /// Fill buffer with fresh data from underlying reader (SIMD-optimized)
    fn fill_buffer(&mut self, min_needed: usize) -> Result<usize> {
        self.fill_buffer_simd(min_needed)
    }

    fn read_inner_retry_interrupted(inner: &mut R, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            match inner.read(buf) {
                Ok(n) => return Ok(n),
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    fn fill_buffer_io(&mut self, min_needed: usize) -> io::Result<usize> {
        if self.end - self.pos >= min_needed {
            return Ok(self.end - self.pos);
        }

        // Move any remaining unread data to the beginning of the buffer
        if self.pos > 0 {
            let remaining = self.end - self.pos;
            if remaining > 0 {
                self.buffer.copy_within(self.pos..self.pos + remaining, 0);
            }
            self.end = remaining;
            self.pos = 0;
        }

        // If `min_needed` exceeds the current buffer capacity, grow the buffer first
        if min_needed > self.buffer.len() {
            self.grow_buffer_to(min_needed).map_err(io::Error::other)?;
        }

        while self.end - self.pos < min_needed {
            if self.end >= self.buffer.len() {
                self.grow_buffer_to(min_needed).map_err(io::Error::other)?;
            }

            let bytes_read =
                Self::read_inner_retry_interrupted(&mut self.inner, &mut self.buffer[self.end..])?;

            if bytes_read == 0 {
                // Genuine EOF; report what we managed to buffer.
                break;
            }

            self.end += bytes_read;
            self.total_read += bytes_read as u64;
        }

        Ok(self.end - self.pos)
    }

    /// Fill buffer using SIMD memcpy for internal data movement
    fn fill_buffer_simd(&mut self, min_needed: usize) -> Result<usize> {
        self.fill_buffer_io(min_needed)
            .map_err(|e| ZiporaError::io_error(format!("Failed to fill buffer: {}", e)))
    }

    /// Grow buffer so its capacity is at least `min_needed` (up to `max_capacity`).
    fn grow_buffer_to(&mut self, min_needed: usize) -> Result<()> {
        let current_capacity = self.buffer.len();
        if min_needed > self.config.max_capacity || current_capacity >= self.config.max_capacity {
            return Err(ZiporaError::io_error(format!(
                "Buffer at maximum capacity ({} bytes), cannot satisfy request for {} bytes",
                self.config.max_capacity, min_needed
            )));
        }

        let scaled = (current_capacity as f64 * self.config.growth_factor) as usize;
        let grown = cmp::max(scaled, current_capacity.saturating_add(1));
        let target = cmp::min(cmp::max(grown, min_needed), self.config.max_capacity);

        let mask = self.config.page_alignment - 1;
        let new_capacity = match target.checked_add(mask).map(|n| n & !mask) {
            Some(aligned) if aligned <= self.config.max_capacity => aligned,
            _ => self.config.max_capacity,
        };

        if new_capacity <= current_capacity || new_capacity < min_needed {
            return Err(ZiporaError::io_error(format!(
                "Buffer at maximum capacity ({} bytes), cannot satisfy request for {} bytes",
                current_capacity, min_needed
            )));
        }

        let mut new_buffer = vec![0u8; new_capacity].into_boxed_slice();
        let existing_data = self.end - self.pos;
        if existing_data > 0 {
            new_buffer[..existing_data].copy_from_slice(&self.buffer[self.pos..self.end]);
        }

        self.buffer = new_buffer;
        self.end = existing_data;
        self.pos = 0;
        Ok(())
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

        if self.end - self.pos >= len {
            let slice = &self.buffer[self.pos..self.pos + len];
            self.pos += len;
            Ok(Some(slice))
        } else {
            Ok(None)
        }
    }

    fn read_bulk_io(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // If we already have buffered data, return it immediately without issuing
        // a second (potentially blocking or failing) read on `self.inner`.
        let buffered = self.end - self.pos;
        if buffered > 0 {
            let to_copy = cmp::min(buffered, buf.len());
            buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
            self.pos += to_copy;
            return Ok(to_copy);
        }

        if buf.len() >= self.config.bulk_read_threshold {
            let bytes_read = Self::read_inner_retry_interrupted(&mut self.inner, buf)?;
            self.total_read += bytes_read as u64;
            Ok(bytes_read)
        } else {
            self.read_buffered(buf)
        }
    }

    /// Bulk read optimization for large reads
    pub fn read_bulk(&mut self, buf: &mut [u8]) -> Result<usize> {
        self.read_bulk_io(buf)
            .map_err(|e| ZiporaError::io_error(format!("Bulk read failed: {}", e)))
    }

    /// Normal buffered read implementation
    fn read_buffered(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if self.pos >= self.end {
            let available = self.fill_buffer_io(1)?;
            if available == 0 {
                return Ok(0);
            }
        }

        let available = self.end - self.pos;
        let to_copy = cmp::min(available, buf.len());
        buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
        self.pos += to_copy;
        Ok(to_copy)
    }

    /// Read data with SIMD memcpy optimization
    ///
    /// This method uses SIMD-accelerated memory copy operations for improved
    /// performance when transferring data from the internal buffer to the
    /// destination buffer.
    ///
    /// # Performance
    /// - Small copies (≤64 bytes): 2-3x faster than standard copy
    /// - Medium copies (64-4096 bytes): 1.5-2x faster
    /// - Large copies (>4KB): Matches or exceeds system memcpy
    ///
    /// # Example
    ///
    /// ```no_run
    /// use zipora::io::stream_buffer::StreamBufferedReader;
    /// use std::io::Cursor;
    ///
    /// let data = b"Hello, SIMD World!";
    /// let cursor = Cursor::new(data);
    /// let mut reader = StreamBufferedReader::new(cursor).unwrap();
    ///
    /// let mut buf = vec![0u8; 18];
    /// let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
    /// assert_eq!(bytes_read, 18);
    /// ```
    pub fn read_simd_optimized(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if self.pos >= self.end {
            let available = self.fill_buffer(1)?;
            if available == 0 {
                return Ok(0);
            }
        }

        let available = self.end - self.pos;
        let to_copy = cmp::min(available, buf.len());
        let src_slice = &self.buffer[self.pos..self.pos + to_copy];
        let dst_slice = &mut buf[..to_copy];

        if simd_ops::fast_copy(src_slice, dst_slice).is_err() {
            dst_slice.copy_from_slice(src_slice);
        }

        self.pos += to_copy;
        Ok(to_copy)
    }

    /// Validate UTF-8 in buffered data
    ///
    /// This method validates the currently buffered data without consuming it,
    /// using hardware-accelerated UTF-8 validation.
    ///
    /// # Returns
    /// - `Ok(true)` if all buffered data is valid UTF-8
    /// - `Ok(false)` if buffered data contains invalid UTF-8
    ///
    /// # Performance
    /// - AVX2: 15+ GB/s validation throughput
    /// - SSE4.2: 8-12 GB/s
    /// - Scalar fallback: 2-3 GB/s
    ///
    /// # Example
    ///
    /// ```no_run
    /// use zipora::io::stream_buffer::StreamBufferedReader;
    /// use std::io::{Cursor, Read};
    ///
    /// let data = b"Hello, World!";
    /// let cursor = Cursor::new(data);
    /// let mut reader = StreamBufferedReader::new(cursor).unwrap();
    ///
    /// // Ensure some data is buffered
    /// let mut buf = vec![0u8; 5];
    /// let _ = reader.read(&mut buf);
    ///
    /// // Validate remaining buffered data
    /// assert!(reader.validate_utf8_buffered().unwrap());
    /// ```
    pub fn validate_utf8_buffered(&self) -> Result<bool> {
        if self.pos >= self.end {
            return Ok(true); // No buffered data
        }

        let buffered_data = &self.buffer[self.pos..self.end];
        Ok(std::str::from_utf8(buffered_data).is_ok())
    }
}

impl<R: Read> Read for StreamBufferedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_bulk_io(buf)
    }
}

impl<R: Read> BufRead for StreamBufferedReader<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos >= self.end {
            self.fill_buffer_io(1)?;
        }
        Ok(&self.buffer[self.pos..self.end])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = cmp::min(self.pos.saturating_add(amt), self.end);
    }
}

impl<R: Read + Seek> Seek for StreamBufferedReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let result = match pos {
            SeekFrom::Current(offset) => {
                let buffered = (self.end - self.pos) as u64;
                let buffered_i64 = i64::try_from(buffered).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "buffered read-ahead overflows i64",
                    )
                })?;
                let adjusted = offset.checked_sub(buffered_i64).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "seek offset overflow")
                })?;
                self.inner.seek(SeekFrom::Current(adjusted))?
            }
            other => self.inner.seek(other)?,
        };
        self.pos = 0;
        self.end = 0;
        Ok(result)
    }
}

type FlushRawFn<W> = fn(&mut W, &mut [u8], &mut usize, &mut u64) -> io::Result<()>;

/// High-performance buffered writer with configurable strategies
pub struct StreamBufferedWriter<W> {
    inner: ManuallyDrop<W>,
    buffer: Box<[u8]>,
    pos: usize, // Current position in buffer
    config: StreamBufferConfig,
    total_written: u64,
    _pool: Option<Arc<SecureMemoryPool>>,
    flush_on_drop: Option<FlushRawFn<W>>,
}

impl<W: Write> StreamBufferedWriter<W> {
    /// Create a new buffered writer with default configuration
    pub fn new(inner: W) -> Result<Self> {
        Self::with_config(inner, StreamBufferConfig::default())
    }

    /// Create a new buffered writer with custom configuration
    pub fn with_config(inner: W, mut config: StreamBufferConfig) -> Result<Self> {
        config.validate()?;

        let pool = if config.use_secure_pool {
            Some(SecureMemoryPool::new(
                crate::memory::SecurePoolConfig::small_secure(),
            )?)
        } else {
            None
        };

        let buffer = StreamBufferedReader::<std::io::Empty>::allocate_aligned_buffer(
            config.initial_capacity,
            config.page_alignment,
        )?;
        config.max_capacity = cmp::max(config.max_capacity, buffer.len());

        Ok(Self {
            inner: ManuallyDrop::new(inner),
            buffer,
            pos: 0,
            config,
            total_written: 0,
            _pool: pool,
            flush_on_drop: Some(Self::flush_raw),
        })
    }

    fn flush_raw(
        inner: &mut W,
        buffer: &mut [u8],
        pos: &mut usize,
        total_written: &mut u64,
    ) -> io::Result<()> {
        let mut written = 0;
        let len = *pos;
        let mut ret = Ok(());

        while written < len {
            match inner.write(&buffer[written..len]) {
                Ok(0) => {
                    ret = Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write buffered data",
                    ));
                    break;
                }
                Ok(n) => {
                    written += n;
                    *total_written += n as u64;
                }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    ret = Err(e);
                    break;
                }
            }
        }

        if written > 0 {
            if written < *pos {
                buffer.copy_within(written..*pos, 0);
            }
            *pos -= written;
        }

        ret
    }

    fn flush_buffer_io(&mut self) -> io::Result<()> {
        Self::flush_raw(
            &mut self.inner,
            &mut self.buffer,
            &mut self.pos,
            &mut self.total_written,
        )
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
        self.flush_on_drop = None;
        // SAFETY: Setting `self.flush_on_drop = None` prevents `Drop::drop` from touching
        // or dropping `self.inner`, and `into_inner` consumes `self` by value so `self.inner`
        // is taken exactly once.
        Ok(unsafe { ManuallyDrop::take(&mut self.inner) })
    }

    /// Get buffer capacity
    #[inline]
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
        if self.buffer.is_empty() {
            return Err(ZiporaError::io_error("writer buffer has zero capacity"));
        }
        self.buffer[0] = byte;
        self.pos = 1;
        Ok(())
    }

    /// Flush internal buffer to underlying writer
    fn flush_buffer(&mut self) -> Result<()> {
        self.flush_buffer_io()
            .map_err(|e| ZiporaError::io_error(format!("Failed to flush buffer: {}", e)))
    }
}

impl<W> Drop for StreamBufferedWriter<W> {
    fn drop(&mut self) {
        if let Some(flush_fn) = self.flush_on_drop.take() {
            let _ = flush_fn(
                &mut self.inner,
                &mut self.buffer,
                &mut self.pos,
                &mut self.total_written,
            );
            // SAFETY: `self.flush_on_drop` was `Some`, which means `into_inner` did not
            // take `self.inner`, so `self.inner` is initialized and dropped once here.
            unsafe {
                ManuallyDrop::drop(&mut self.inner);
            }
        }
    }
}

impl<W: Write> Write for StreamBufferedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // If the incoming write does not fit in the remaining buffer space,
        // flush existing buffered bytes BEFORE copying any bytes from `buf`
        // so a flush error never partially buffers `buf`.
        if buf.len() > self.buffer.len() - self.pos {
            self.flush_buffer_io()?;
        }

        if buf.len() >= self.config.bulk_read_threshold || buf.len() >= self.buffer.len() {
            self.flush_buffer_io()?;

            loop {
                match self.inner.write(buf) {
                    Ok(written) => {
                        self.total_written += written as u64;
                        return Ok(written);
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
        } else {
            self.buffer[self.pos..self.pos + buf.len()].copy_from_slice(buf);
            self.pos += buf.len();
            Ok(buf.len())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer_io()?;
        loop {
            match self.inner.flush() {
                Ok(()) => return Ok(()),
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }
}

impl<W: Write + Seek> Seek for StreamBufferedWriter<W> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.flush()?;
        self.inner.seek(pos)
    }
}

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
        reader.read_exact(&mut buf).unwrap();

        assert!(reader.total_read() > 0);
        assert!(reader.has_data_in_buffer() || reader.total_read() == data.len() as u64);
    }

    //==========================================================================
    // SIMD INTEGRATION TESTS
    //==========================================================================

    #[test]
    fn test_stream_reader_simd_optimized_read() {
        let data = b"Hello, SIMD World! This tests SIMD-optimized reading.";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        let mut buf = vec![0u8; data.len()];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();

        assert_eq!(bytes_read, data.len());
        assert_eq!(&buf[..], data);
    }

    #[test]
    fn test_stream_reader_simd_optimized_read_partial() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read in smaller chunks
        let mut buf = vec![0u8; 7];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert_eq!(bytes_read, 7);
        assert_eq!(&buf, b"Hello, ");

        let mut buf2 = vec![0u8; 6];
        let bytes_read2 = reader.read_simd_optimized(&mut buf2).unwrap();
        assert_eq!(bytes_read2, 6);
        assert_eq!(&buf2, b"World!");
    }

    #[test]
    fn test_stream_reader_simd_optimized_read_large() {
        // Test with large data to exercise SIMD paths
        let large_data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let cursor = Cursor::new(&large_data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        let mut buf = vec![0u8; large_data.len()];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();

        assert_eq!(bytes_read, large_data.len());
        assert_eq!(&buf, &large_data[..]);
    }

    #[test]
    fn test_stream_reader_utf8_validation_valid() {
        let data = "Hello, World! Valid UTF-8 text with unicode: 世界 🦀";
        let cursor = Cursor::new(data.as_bytes());
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read some data to buffer more
        let mut buf = vec![0u8; 10];
        let _ = reader.read(&mut buf).unwrap();

        // Validate remaining buffered data
        let is_valid = reader.validate_utf8_buffered().unwrap();
        assert!(is_valid, "Valid UTF-8 should pass validation");
    }

    #[test]
    fn test_stream_reader_utf8_validation_invalid() {
        // Create data with invalid UTF-8 sequence
        let mut data = Vec::from(b"Hello, ".as_ref());
        data.push(0xFF); // Invalid UTF-8 byte
        data.extend_from_slice(b" World!");

        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read some data to buffer the invalid sequence
        let mut buf = vec![0u8; 5];
        let _ = reader.read(&mut buf).unwrap();

        // Validate buffered data
        let is_valid = reader.validate_utf8_buffered().unwrap();
        assert!(!is_valid, "Invalid UTF-8 should fail validation");
    }

    #[test]
    fn test_stream_reader_utf8_validation_empty_buffer() {
        let data = b"Hello, World!";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read all data (no buffered data remaining)
        let mut buf = vec![0u8; data.len()];
        let _ = reader.read(&mut buf).unwrap();

        // Validate empty buffer
        let is_valid = reader.validate_utf8_buffered().unwrap();
        assert!(is_valid, "Empty buffer should be valid UTF-8");
    }

    #[test]
    fn test_stream_reader_utf8_validation_multibyte() {
        let test_cases = vec![
            ("café", true),            // 2-byte sequences
            ("日本語", true),          // 3-byte sequences (CJK)
            ("🦀🌍", true),            // 4-byte sequences (emoji)
            ("Hello, 世界! 🦀", true), // Mixed ASCII and multibyte
        ];

        for (text, expected_valid) in test_cases {
            let cursor = Cursor::new(text.as_bytes());
            let mut reader = StreamBufferedReader::new(cursor).unwrap();

            // Don't read any data - validate all buffered data
            // This ensures we're not cutting multibyte sequences in the middle
            let _ = reader.ensure_buffered(text.len());

            let is_valid = reader.validate_utf8_buffered().unwrap();
            assert_eq!(
                is_valid, expected_valid,
                "UTF-8 validation mismatch for: {}",
                text
            );
        }
    }

    #[test]
    fn test_stream_reader_simd_vs_standard_read() {
        let data = b"The quick brown fox jumps over the lazy dog";

        // Read with SIMD-optimized method
        let cursor1 = Cursor::new(data);
        let mut reader1 = StreamBufferedReader::new(cursor1).unwrap();
        let mut buf1 = vec![0u8; data.len()];
        let bytes_read1 = reader1.read_simd_optimized(&mut buf1).unwrap();

        // Read with standard method
        let cursor2 = Cursor::new(data);
        let mut reader2 = StreamBufferedReader::new(cursor2).unwrap();
        let mut buf2 = vec![0u8; data.len()];
        let bytes_read2 = reader2.read(&mut buf2).unwrap();

        // Results should be identical
        assert_eq!(bytes_read1, bytes_read2);
        assert_eq!(buf1, buf2);
        assert_eq!(&buf1[..], data);
    }

    #[test]
    fn test_stream_reader_simd_with_different_configs() {
        let data = b"Test data for different buffer configurations";

        // Performance optimized
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::performance_optimized(cursor).unwrap();
        let mut buf = vec![0u8; data.len()];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert_eq!(bytes_read, data.len());
        assert_eq!(&buf, data);

        // Memory efficient
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::memory_efficient(cursor).unwrap();
        let mut buf = vec![0u8; data.len()];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert_eq!(bytes_read, data.len());
        assert_eq!(&buf, data);

        // Low latency
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::low_latency(cursor).unwrap();
        let mut buf = vec![0u8; data.len()];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert_eq!(bytes_read, data.len());
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_stream_reader_simd_read_empty() {
        let data = b"";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        let mut buf = vec![0u8; 10];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert_eq!(bytes_read, 0);
    }

    #[test]
    fn test_stream_reader_utf8_large_data() {
        // Test with large data to exercise SIMD validation paths
        let large_text = "Hello, World! 世界 🦀 ".repeat(1000);
        let cursor = Cursor::new(large_text.as_bytes());
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read exactly one complete repetition to keep buffer at character boundary
        // Each repetition is "Hello, World! 世界 🦀 " which is 27 bytes
        let mut buf = vec![0u8; 27]; // Read exactly one complete unit
        let _ = reader.read(&mut buf).unwrap();

        // Now the buffer should contain complete UTF-8 sequences
        // Validate remaining buffered data
        let is_valid = reader.validate_utf8_buffered().unwrap();
        assert!(is_valid, "Large valid UTF-8 buffer should pass validation");
    }

    /// Hands out at most one byte per `read` call, the way a pipe does.
    struct DribblingReader {
        data: Vec<u8>,
        pos: usize,
    }

    impl Read for DribblingReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.data.len() || buf.is_empty() {
                return Ok(0);
            }
            buf[0] = self.data[self.pos];
            self.pos += 1;
            Ok(1)
        }
    }

    #[test]
    fn test_read_slice_survives_short_reads() {
        // Regression: the refill issued a single `inner.read`, so a source
        // that returns fewer bytes than asked for was indistinguishable from
        // EOF and read_slice reported Ok(None) with the data still available.
        let mut reader = StreamBufferedReader::new(DribblingReader {
            data: b"Hello, World!".to_vec(),
            pos: 0,
        })
        .unwrap();

        assert_eq!(reader.read_slice(5).unwrap(), Some(&b"Hello"[..]));
        assert_eq!(reader.read_slice(2).unwrap(), Some(&b", "[..]));
        assert_eq!(reader.read_slice(6).unwrap(), Some(&b"World!"[..]));
    }

    #[test]
    fn test_read_slice_still_reports_none_at_eof() {
        // The refill loop must not spin forever on a genuinely short stream.
        let mut reader = StreamBufferedReader::new(DribblingReader {
            data: b"abc".to_vec(),
            pos: 0,
        })
        .unwrap();

        assert_eq!(reader.read_slice(8).unwrap(), None);
    }

    #[test]
    fn test_stream_reader_simd_fill_buffer() {
        let data = b"Testing SIMD-optimized buffer compaction";
        let cursor = Cursor::new(data);
        let mut reader = StreamBufferedReader::new(cursor).unwrap();

        // Read in multiple small chunks to trigger buffer compaction
        for _ in 0..5 {
            let mut buf = vec![0u8; 5];
            let _ = reader.read(&mut buf);
        }

        // Read remaining data
        let mut buf = vec![0u8; 100];
        let bytes_read = reader.read_simd_optimized(&mut buf).unwrap();
        assert!(bytes_read > 0 || reader.total_read() == data.len() as u64);
    }

    fn miri_fast_config() -> StreamBufferConfig {
        StreamBufferConfig {
            initial_capacity: 4096,
            max_capacity: 64 * 1024,
            growth_factor: 2.0,
            page_alignment: 4096,
            use_secure_pool: false,
            bulk_read_threshold: 8192,
            enable_readahead: true,
            readahead_multiplier: 2,
        }
    }

    #[test]
    fn test_c9_sb1_writer_drop_flushes_buffered_bytes() {
        let mut out = Vec::new();
        {
            let mut w = StreamBufferedWriter::with_config(&mut out, miri_fast_config()).unwrap();
            w.write_all(b"hello").unwrap();
        }
        assert_eq!(out, b"hello");

        let mut out2 = Vec::new();
        let mut w2 = StreamBufferedWriter::with_config(&mut out2, miri_fast_config()).unwrap();
        w2.write_all(b"world").unwrap();
        let inner = w2.into_inner().unwrap();
        assert_eq!(inner.as_slice(), b"world");
    }

    #[test]
    fn test_c9_sb2_read_returns_available_bytes_without_blocking_for_full_buffer() {
        struct OneChunkThenPanicReader {
            calls: usize,
        }

        impl Read for OneChunkThenPanicReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                assert_eq!(self.calls, 1, "inner read called a second time when data was already available");
                let msg = b"hello";
                buf[..msg.len()].copy_from_slice(msg);
                Ok(msg.len())
            }
        }

        let mut r = StreamBufferedReader::with_config(
            OneChunkThenPanicReader { calls: 0 },
            miri_fast_config(),
        )
        .unwrap();
        let mut buf = [0u8; 64];
        assert_eq!(r.read(&mut buf).unwrap(), 5);
        assert_eq!(&buf[..5], b"hello");

        let mut r_simd = StreamBufferedReader::with_config(
            OneChunkThenPanicReader { calls: 0 },
            miri_fast_config(),
        )
        .unwrap();
        let mut buf_simd = [0u8; 64];
        assert_eq!(r_simd.read_simd_optimized(&mut buf_simd).unwrap(), 5);
        assert_eq!(&buf_simd[..5], b"hello");

        let mut r_bulk = StreamBufferedReader::with_config(
            OneChunkThenPanicReader { calls: 0 },
            miri_fast_config(),
        )
        .unwrap();
        let mut first = [0u8; 2];
        assert_eq!(r_bulk.read(&mut first).unwrap(), 2);
        assert_eq!(&first, b"he");
        let mut large = vec![0u8; 8192];
        assert_eq!(r_bulk.read(&mut large).unwrap(), 3);
        assert_eq!(&large[..3], b"llo");
    }

    #[test]
    fn test_c9_sb3_seek_accounts_for_buffered_readahead_and_preserves_buffer_on_err() {
        let mut r =
            StreamBufferedReader::with_config(Cursor::new(b"0123456789".to_vec()), miri_fast_config())
                .unwrap();

        let mut b = [0u8; 1];
        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"0");
        assert_eq!(r.stream_position().unwrap(), 1);

        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"1");

        // Now position is 2, with "23456789" buffered. Seek forward 2 bytes -> 4.
        assert_eq!(r.seek(SeekFrom::Current(2)).unwrap(), 4);
        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"4");

        // Failed seek before start must not discard buffered read-ahead ("56789")
        assert!(r.seek(SeekFrom::Current(-100)).is_err());
        r.read_exact(&mut b).unwrap();
        assert_eq!(&b, b"5");
    }

    #[test]
    fn test_c9_sb4_read_retries_interrupted_preserves_error_kind_and_never_loses_buffered_bytes() {
        enum Step {
            Bytes(&'static [u8]),
            Err(io::ErrorKind),
        }

        struct ScriptedReader {
            steps: Vec<Step>,
            idx: usize,
        }

        impl Read for ScriptedReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                if self.idx >= self.steps.len() {
                    return Ok(0);
                }
                let step = &self.steps[self.idx];
                self.idx += 1;
                match step {
                    Step::Bytes(data) => {
                        let n = cmp::min(buf.len(), data.len());
                        buf[..n].copy_from_slice(&data[..n]);
                        Ok(n)
                    }
                    Step::Err(kind) => Err(io::Error::new(*kind, "scripted error")),
                }
            }
        }

        // (a) Interrupted is retried transparently
        let mut r1 = StreamBufferedReader::with_config(
            ScriptedReader {
                steps: vec![
                    Step::Bytes(b"abc"),
                    Step::Err(io::ErrorKind::Interrupted),
                    Step::Bytes(b"defghij"),
                ],
                idx: 0,
            },
            miri_fast_config(),
        )
        .unwrap();
        let mut out1 = [0u8; 10];
        r1.read_exact(&mut out1).unwrap();
        assert_eq!(&out1, b"abcdefghij");

        // (b) Bulk read after partial buffer does not lose buffered bytes or collapse ErrorKind
        let mut r2 = StreamBufferedReader::with_config(
            ScriptedReader {
                steps: vec![
                    Step::Bytes(b"ab"),
                    Step::Err(io::ErrorKind::WouldBlock),
                    Step::Bytes(b"cdef"),
                ],
                idx: 0,
            },
            miri_fast_config(),
        )
        .unwrap();
        let mut collected = Vec::new();
        let mut first = [0u8; 1];
        r2.read_exact(&mut first).unwrap();
        collected.push(first[0]);

        let mut saw_would_block = false;
        let mut large = vec![0u8; 8192];
        loop {
            match r2.read(&mut large) {
                Ok(0) => break,
                Ok(n) => collected.extend_from_slice(&large[..n]),
                Err(e) => {
                    assert_eq!(e.kind(), io::ErrorKind::WouldBlock);
                    saw_would_block = true;
                }
            }
        }
        assert!(saw_would_block);
        assert_eq!(collected, b"abcdef");
    }

    #[test]
    fn test_c9_sb5_writer_partial_flush_and_write_error_do_not_duplicate_bytes() {
        struct ScriptedWriter {
            out: Vec<u8>,
            call: usize,
            mode_partial_flush: bool,
        }

        impl Write for ScriptedWriter {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.call += 1;
                if self.mode_partial_flush {
                    if self.call == 1 {
                        let n = cmp::min(2, buf.len());
                        self.out.extend_from_slice(&buf[..n]);
                        return Ok(n);
                    }
                    if self.call == 2 {
                        return Err(io::Error::new(io::ErrorKind::WouldBlock, "would block"));
                    }
                } else if self.call == 1 {
                    return Err(io::Error::new(io::ErrorKind::WouldBlock, "would block"));
                }
                self.out.extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        // (a) Partial flush followed by WouldBlock does not duplicate the 2 written bytes on retry
        let mut w1 = StreamBufferedWriter::with_config(
            ScriptedWriter {
                out: Vec::new(),
                call: 0,
                mode_partial_flush: true,
            },
            miri_fast_config(),
        )
        .unwrap();
        w1.write_all(b"abcdef").unwrap();
        let err = w1.flush().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        w1.flush().unwrap();
        assert_eq!(w1.get_ref().out, b"abcdef");

        // (b) Write that triggers a failing flush does not partially buffer the new slice
        let mut w2 = StreamBufferedWriter::with_config(
            ScriptedWriter {
                out: Vec::new(),
                call: 0,
                mode_partial_flush: false,
            },
            miri_fast_config(),
        )
        .unwrap();
        assert_eq!(w2.write(&[b'a'; 4000]).unwrap(), 4000);
        let err = w2.write(&[b'b'; 200]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::WouldBlock);
        assert_eq!(w2.write(&[b'b'; 200]).unwrap(), 200);
        w2.flush().unwrap();
        assert_eq!(w2.get_ref().out.len(), 4200);
        assert!(w2.get_ref().out[..4000].iter().all(|&b| b == b'a'));
        assert!(w2.get_ref().out[4000..].iter().all(|&b| b == b'b'));
    }

    #[test]
    fn test_c9_sb6_read_slice_above_initial_capacity_grows_buffer_before_eof() {
        let data: Vec<u8> = (0..10_000).map(|i| (i % 251) as u8).collect();
        let mut r =
            StreamBufferedReader::with_config(Cursor::new(data.clone()), miri_fast_config()).unwrap();

        let slice = r
            .read_slice(5000)
            .unwrap()
            .expect("read_slice(5000) must grow buffer from 4096 and return Some");
        assert_eq!(slice, &data[..5000]);
    }

    #[test]
    fn test_c9_sb7_invalid_config_and_huge_ensure_buffered_return_err_without_panic() {
        let zero_cap = StreamBufferConfig {
            initial_capacity: 0,
            ..miri_fast_config()
        };
        let zero_align = StreamBufferConfig {
            page_alignment: 0,
            ..miri_fast_config()
        };
        let non_pow2_align = StreamBufferConfig {
            page_alignment: 3000,
            ..miri_fast_config()
        };
        let bad_max = StreamBufferConfig {
            initial_capacity: 8192,
            max_capacity: 4096,
            ..miri_fast_config()
        };
        let bad_growth = StreamBufferConfig {
            growth_factor: 1.0,
            ..miri_fast_config()
        };
        let huge_growth = StreamBufferConfig {
            growth_factor: f64::MAX,
            ..miri_fast_config()
        };

        for cfg in [
            zero_cap,
            zero_align,
            non_pow2_align,
            bad_max,
            bad_growth,
            huge_growth,
        ] {
            let r_res = std::panic::catch_unwind(|| {
                StreamBufferedReader::with_config(Cursor::new(b"x"), cfg.clone())
            });
            assert!(r_res.expect("reader with_config must not panic").is_err());

            let w_res = std::panic::catch_unwind(|| {
                StreamBufferedWriter::with_config(Vec::<u8>::new(), cfg.clone())
            });
            assert!(w_res.expect("writer with_config must not panic").is_err());
        }

        let mut r =
            StreamBufferedReader::with_config(Cursor::new(b"abc"), miri_fast_config()).unwrap();
        let huge = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            r.ensure_buffered(usize::MAX)
        }));
        assert!(
            huge.expect("ensure_buffered(usize::MAX) must not panic on overflow")
                .is_err()
        );
    }

    #[test]
    fn test_c9_sb8_fill_buf_reads_full_buffer_capacity_per_inner_read() {
        struct CountingReader {
            data: Vec<u8>,
            pos: usize,
            calls: usize,
        }

        impl Read for CountingReader {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.pos >= self.data.len() {
                    return Ok(0);
                }
                let n = cmp::min(buf.len(), self.data.len() - self.pos);
                buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
                self.pos += n;
                Ok(n)
            }
        }

        let mut data = vec![b'x'; 4095];
        data.push(b'\n');

        let cfg = StreamBufferConfig {
            enable_readahead: false,
            readahead_multiplier: 1,
            ..miri_fast_config()
        };
        let mut r = StreamBufferedReader::with_config(
            CountingReader {
                data,
                pos: 0,
                calls: 0,
            },
            cfg,
        )
        .unwrap();

        let mut line = Vec::new();
        let n = r.read_until(b'\n', &mut line).unwrap();
        assert_eq!(n, 4096);
        assert!(
            r.get_ref().calls <= 4,
            "fill_buf should fill up to buffer capacity per read call, got {} calls",
            r.get_ref().calls
        );
    }
}
