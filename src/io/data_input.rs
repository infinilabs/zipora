//! Data input abstractions and implementations
//!
//! This module provides traits and implementations for reading various data types
//! from different sources including byte slices, files, and memory-mapped regions.

#[cfg(feature = "mmap")]
use std::fs::File;
use std::io::Read;
#[cfg(feature = "mmap")]
use std::path::Path;

use crate::error::{Result, ZiporaError};
use crate::io::var_int::VarInt;

#[cfg(feature = "mmap")]
use memmap2::Mmap;

/// Most bytes a streaming reader (`remaining_bytes() == None`) allocates ahead
/// of the input that has to back them.
///
/// Lengths and element counts read from the input are untrusted. Code that
/// sizes a buffer from one without knowing the remaining input length allocates
/// at most this much before the input has delivered the corresponding bytes.
pub(crate) const MAX_READ_AHEAD_ALLOC: usize = 64 * 1024;

/// Maximum upfront byte reservation (`64 MiB`) when `remaining_bytes()` is
/// `Some(r)` (such as [`SliceDataInput`] or `MmapDataInput`).
pub(crate) const MAX_KNOWN_INPUT_PREALLOC: usize = 64 * 1024 * 1024;

/// Error for a read of `len` bytes from a source with only `remaining` left.
fn short_input(len: usize, remaining: u64) -> ZiporaError {
    ZiporaError::io_error(format!(
        "Unexpected end of data: {} bytes requested, {} remaining",
        len, remaining
    ))
}

/// Fail before allocating when a source that knows its remaining length
/// cannot supply `len` more bytes.
fn check_len_available(len: usize, remaining: Option<u64>) -> Result<()> {
    match remaining {
        Some(remaining) if u64::try_from(len).unwrap_or(u64::MAX) > remaining => {
            Err(short_input(len, remaining))
        }
        _ => Ok(()),
    }
}

/// Capacity to reserve before decoding `count` elements of `elem_size`
/// bytes each, where `count` was read from the input.
///
/// - When `remaining` is `None` (streaming input), reserves at most
///   `MAX_READ_AHEAD_ALLOC / elem_size.max(1)` elements (`64 KiB`).
/// - When `remaining` is `Some(r)` (backed by an in-memory slice or file
///   mapping with `r` bytes left), reserves up to `min(r, MAX_KNOWN_INPUT_PREALLOC) / elem_size.max(1)`
///   elements (and at most `r` elements), so any valid count backed by `r`
///   wire bytes up to `64 MiB` (such as 1M `u64`s = `8 MiB`) is reserved in a
///   single allocation without intermediate reallocations.
pub(crate) fn prealloc_capacity(count: usize, elem_size: usize, remaining: Option<u64>) -> usize {
    let elem_size = elem_size.max(1);
    match remaining {
        Some(r) => {
            let r_usize = usize::try_from(r).unwrap_or(usize::MAX);
            let byte_budget = r_usize.min(MAX_KNOWN_INPUT_PREALLOC);
            count.min(byte_budget / elem_size).min(r_usize)
        }
        None => count.min(MAX_READ_AHEAD_ALLOC / elem_size),
    }
}

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

    /// Upper bound on the number of bytes that can still be read, if the
    /// source knows one.
    ///
    /// [`read_vec`](Self::read_vec) consults it before sizing a buffer from a
    /// length read from the input, so a corrupt prefix fails with an error
    /// instead of allocating memory the input cannot back, and the container
    /// deserializers in `complex_types` and `smart_ptr` use it to cap the
    /// capacity they reserve for an element count read from the input.
    /// `None` (the default) means the source does not know; readers then
    /// grow their buffers in bounded steps as the bytes arrive.
    /// Implementations must never report fewer bytes than can actually be
    /// read.
    fn remaining_bytes(&self) -> Option<u64> {
        None
    }

    /// Read a vector of bytes with the specified length
    ///
    /// `len` is treated as untrusted. If [`remaining_bytes`](Self::remaining_bytes)
    /// reports fewer than `len` bytes, the call fails before allocating.
    /// When `remaining_bytes` is `Some` and `len <= MAX_KNOWN_INPUT_PREALLOC`,
    /// the verified length is read in one allocation; otherwise the buffer
    /// grows as the bytes arrive in steps of at most `max(64 KiB, bytes already read)`.
    fn read_vec(&mut self, len: usize) -> Result<Vec<u8>> {
        let remaining = self.remaining_bytes();
        check_len_available(len, remaining)?;
        if remaining.is_some() && len <= MAX_KNOWN_INPUT_PREALLOC {
            let mut buf = Vec::new();
            buf.try_reserve_exact(len)
                .map_err(|_| ZiporaError::out_of_memory(len))?;
            buf.resize(len, 0);
            self.read_bytes(&mut buf)?;
            return Ok(buf);
        }
        let mut buf = Vec::new();
        while buf.len() < len {
            let filled = buf.len();
            let step = (len - filled).min(MAX_READ_AHEAD_ALLOC.max(filled));
            buf.try_reserve_exact(step)
                .map_err(|_| ZiporaError::out_of_memory(filled.saturating_add(step)))?;
            buf.resize(filled + step, 0);
            self.read_bytes(&mut buf[filled..])?;
        }
        Ok(buf)
    }

    /// Read a length-prefixed byte vector (length as varint)
    fn read_length_prefixed_bytes(&mut self) -> Result<Vec<u8>> {
        let len = self.read_var_int()?;
        let len = usize::try_from(len).map_err(|_| {
            ZiporaError::invalid_data(format!("Length prefix {} does not fit in usize", len))
        })?;
        self.read_vec(len)
    }

    /// Read a string with the specified length (UTF-8 encoded)
    fn read_string(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_vec(len)?;
        String::from_utf8(bytes)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid UTF-8 string: {}", e)))
    }

    /// Read a length-prefixed string (length as varint, UTF-8 encoded)
    fn read_length_prefixed_string(&mut self) -> Result<String> {
        let bytes = self.read_length_prefixed_bytes()?;
        String::from_utf8(bytes)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid UTF-8 string: {}", e)))
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
        Self { data, position: 0 }
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
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let value = self.data[self.position];
        self.position += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16> {
        if self.position + 2 > self.data.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 2];
        self.position += 2;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.position + 4 > self.data.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 4];
        self.position += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.position + 8 > self.data.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.data[self.position..self.position + 8];
        self.position += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.position + buf.len() > self.data.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        buf.copy_from_slice(&self.data[self.position..self.position + buf.len()]);
        self.position += buf.len();
        Ok(())
    }

    fn remaining_bytes(&self) -> Option<u64> {
        Some(self.remaining() as u64)
    }

    fn read_vec(&mut self, len: usize) -> Result<Vec<u8>> {
        let remaining = self.remaining();
        if len > remaining {
            return Err(short_input(len, remaining as u64));
        }
        // `len <= remaining` keeps `end` within `data`.
        let end = self.position + len;
        let vec = self.data[self.position..end].to_vec();
        self.position = end;
        Ok(vec)
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.remaining() {
            return Err(ZiporaError::io_error("Cannot skip past end of data"));
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
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u8: {}", e)))?;
        self.position += 1;
        Ok(buf[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u16: {}", e)))?;
        self.position += 2;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u32: {}", e)))?;
        self.position += 4;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.reader
            .read_exact(&mut buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read u64: {}", e)))?;
        self.position += 8;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        self.reader
            .read_exact(buf)
            .map_err(|e| ZiporaError::io_error(format!("Failed to read bytes: {}", e)))?;
        self.position += buf.len() as u64;
        Ok(())
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let mut buf = vec![0u8; n.min(8192)]; // Read in chunks for large skips
        let mut remaining = n;

        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            self.reader
                .read_exact(&mut buf[..to_read])
                .map_err(|e| ZiporaError::io_error(format!("Failed to skip bytes: {}", e)))?;
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
    ///
    /// Limitation: the file is mapped, not copied, and nothing here stops
    /// another handle or process from changing it. memmap2 documents any
    /// such change while the map is alive as undefined behaviour. In
    /// practice, if the file is truncated, a read that touches a page past
    /// the new end raises SIGBUS and kills the process; if it is modified,
    /// reads observe the new bytes, possibly in the middle of a value. Only
    /// open files that nothing writes to for the lifetime of the returned
    /// value.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to open file: {}", e)))?;

        // SAFETY: `Mmap::map` requires that the mapped file is not modified
        // or truncated, by this or any other process, while the map is alive
        // (memmap2 0.9.9 `Mmap` docs, src/lib.rs:681-687). This function
        // cannot enforce that, so it is passed on to the caller as the
        // `Limitation:` on `open` above. The map is read-only and `file` is
        // a handle that `File::open` just opened for reading.
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|e| ZiporaError::io_error(format!("Failed to memory map file: {}", e)))?
        };

        Ok(Self { mmap, position: 0 })
    }

    /// Create a new MmapDataInput from an existing memory map
    pub fn from_mmap(mmap: Mmap) -> Self {
        Self { mmap, position: 0 }
    }

    /// Get the current position
    pub fn pos(&self) -> usize {
        self.position
    }

    /// Get the total size of the memory-mapped region
    #[inline]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if the memory-mapped region is empty
    #[inline]
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
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}

#[cfg(feature = "mmap")]
impl DataInput for MmapDataInput {
    fn read_u8(&mut self) -> Result<u8> {
        if self.position >= self.mmap.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let value = self.mmap[self.position];
        self.position += 1;
        Ok(value)
    }

    fn read_u16(&mut self) -> Result<u16> {
        if self.position + 2 > self.mmap.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 2];
        self.position += 2;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.position + 4 > self.mmap.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 4];
        self.position += 4;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.position + 8 > self.mmap.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        let bytes = &self.mmap[self.position..self.position + 8];
        self.position += 8;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_var_int(&mut self) -> Result<u64> {
        VarInt::read_from(self)
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        if self.position + buf.len() > self.mmap.len() {
            return Err(ZiporaError::io_error("Unexpected end of data"));
        }
        buf.copy_from_slice(&self.mmap[self.position..self.position + buf.len()]);
        self.position += buf.len();
        Ok(())
    }

    fn remaining_bytes(&self) -> Option<u64> {
        Some(self.remaining() as u64)
    }

    fn read_vec(&mut self, len: usize) -> Result<Vec<u8>> {
        let remaining = self.remaining();
        if len > remaining {
            return Err(short_input(len, remaining as u64));
        }
        // `len <= remaining` keeps `end` within the map.
        let end = self.position + len;
        let vec = self.mmap[self.position..end].to_vec();
        self.position = end;
        Ok(vec)
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        if n > self.remaining() {
            return Err(ZiporaError::io_error("Cannot skip past end of data"));
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
///
/// Limitation: the file is memory-mapped; see [`MmapDataInput::open`] for
/// what happens if it is truncated or modified while it is being read.
#[cfg(feature = "mmap")]
pub fn from_file<P: AsRef<Path>>(path: P) -> Result<MmapDataInput> {
    MmapDataInput::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::io::Write;
    use tempfile::NamedTempFile;

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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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

    /// Reader that returns at most one byte per call, fails every other call
    /// with `Interrupted`, and records the largest buffer it was asked to
    /// fill.
    struct TrickleReader {
        data: Vec<u8>,
        pos: usize,
        interrupt_next: bool,
        max_request: usize,
    }

    impl TrickleReader {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                pos: 0,
                interrupt_next: true,
                max_request: 0,
            }
        }
    }

    impl Read for TrickleReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.max_request = self.max_request.max(buf.len());
            let interrupt = self.interrupt_next;
            self.interrupt_next = !interrupt;
            if interrupt {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            if buf.is_empty() || self.pos >= self.data.len() {
                return Ok(0);
            }
            let step = if cfg!(miri) { 4096 } else { 1 };
            let n = buf.len().min(step).min(self.data.len() - self.pos);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    /// Length prefixes with no payload behind them. The u64::MAX prefix must
    /// stay first: before the fix it panicked ("capacity overflow"), which
    /// fails the test before the 2^45 prefix, whose allocation request
    /// aborted the process, is attempted.
    const OVERSIZED_PREFIXES: [&[u8]; 2] = [
        &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01],
        &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x08],
    ];

    #[test]
    fn test_c9_di1_length_prefix_beyond_slice_is_err() {
        for prefix in OVERSIZED_PREFIXES {
            let r = std::panic::catch_unwind(|| {
                SliceDataInput::new(prefix).read_length_prefixed_bytes()
            });
            assert!(matches!(r, Ok(Err(_))), "prefix {:02x?}: {:?}", prefix, r);
        }

        // One byte short of the declared length.
        let mut input = SliceDataInput::new(&[3, b'a', b'b']);
        assert!(input.read_length_prefixed_bytes().is_err());

        // An oversized explicit length fails without consuming input.
        let mut input = SliceDataInput::new(&[1, 2, 3]);
        assert!(input.read_vec(4).is_err());
        assert!(input.read_string(usize::MAX).is_err());
        assert_eq!(input.pos(), 0);
        assert_eq!(input.read_vec(3).unwrap(), vec![1, 2, 3]);
        assert_eq!(input.read_vec(0).unwrap(), Vec::<u8>::new());
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore)] // Miri cannot map files.
    fn test_c9_di1_length_prefix_beyond_mmap_is_err() {
        for prefix in OVERSIZED_PREFIXES {
            let mut temp_file = NamedTempFile::new().unwrap();
            temp_file.write_all(prefix).unwrap();
            temp_file.flush().unwrap();
            let path = temp_file.path().to_path_buf();
            let r = std::panic::catch_unwind(move || {
                MmapDataInput::open(&path)
                    .unwrap()
                    .read_length_prefixed_bytes()
            });
            assert!(matches!(r, Ok(Err(_))), "prefix {:02x?}: {:?}", prefix, r);
        }
    }

    #[test]
    fn test_c9_di1_stream_length_prefix_allocates_with_the_data() {
        // A 2^30 length prefix followed by three payload bytes.
        let mut data = Vec::new();
        VarInt::write_to(&mut data, 1 << 30).unwrap();
        data.extend_from_slice(b"abc");
        let mut input = ReaderDataInput::new(TrickleReader::new(data));
        assert!(input.read_length_prefixed_bytes().is_err());
        // `read_vec` hands the reader the buffer it allocated, so the largest
        // request is the largest buffer allocated ahead of the data.
        let max_request = input.into_inner().max_request;
        assert!(
            max_request <= 64 * 1024,
            "allocated {} bytes ahead of the data",
            max_request
        );
    }

    #[test]
    fn test_c9_di1_stream_read_vec_crosses_growth_steps() {
        // Lengths around the 64 KiB growth step, delivered one byte per call
        // with an Interrupted error before every byte.
        let lens: &[usize] = if cfg!(miri) {
            &[0, 1, 65_537]
        } else {
            &[0, 1, 65_535, 65_536, 65_537, 3 * 65_536 + 5]
        };
        for &len in lens {
            let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let mut data = Vec::new();
            VarInt::write_to(&mut data, len as u64).unwrap();
            data.extend_from_slice(&payload);
            data.push(0xAA);
            let mut input = ReaderDataInput::new(TrickleReader::new(data));
            assert_eq!(
                input.read_length_prefixed_bytes().unwrap(),
                payload,
                "len {}",
                len
            );
            assert_eq!(input.read_u8().unwrap(), 0xAA, "len {}", len);
        }
    }

    #[test]
    fn test_c9_di1_remaining_bytes_reports_what_is_left() {
        let mut input = SliceDataInput::new(&[1, 2, 3]);
        assert_eq!(input.remaining_bytes(), Some(3));
        input.read_u8().unwrap();
        assert_eq!(input.remaining_bytes(), Some(2));
        input.read_vec(2).unwrap();
        assert_eq!(input.remaining_bytes(), Some(0));
        let input = ReaderDataInput::new(Cursor::new(vec![1, 2, 3]));
        assert_eq!(input.remaining_bytes(), None);
    }

    #[test]
    fn test_c9_di2_skip_overflow_is_err() {
        let data = [1u8, 2, 3];
        let r = std::panic::catch_unwind(|| {
            let mut input = SliceDataInput::new(&data);
            input.read_u8().unwrap();
            let skipped = input.skip(usize::MAX);
            (skipped.is_err(), input.pos())
        });
        assert_eq!(r.ok(), Some((true, 1)));

        let mut input = SliceDataInput::new(&data);
        input.read_u8().unwrap();
        assert!(input.skip(3).is_err());
        input.skip(2).unwrap();
        assert_eq!(input.pos(), 3);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore)] // Miri cannot map files.
    fn test_c9_di2_mmap_skip_overflow_is_err() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[1, 2, 3]).unwrap();
        temp_file.flush().unwrap();
        let path = temp_file.path().to_path_buf();
        let r = std::panic::catch_unwind(move || {
            let mut input = MmapDataInput::open(&path).unwrap();
            input.read_u8().unwrap();
            let skipped = input.skip(usize::MAX);
            (skipped.is_err(), input.pos())
        });
        assert_eq!(r.ok(), Some((true, 1)));
    }

    #[test]
    fn test_c9_ct1_prealloc_capacity_is_bounded() {
        // Without a known remaining length, the read-ahead budget caps a huge count: 64 KiB / 8 B = 8192.
        assert_eq!(prealloc_capacity(usize::MAX, 8, None), 8192);
        // With a known remaining length, pre-allocation is capped at MAX_KNOWN_INPUT_PREALLOC (64 MiB).
        assert_eq!(
            prealloc_capacity(usize::MAX, 1, Some(u64::MAX)),
            MAX_KNOWN_INPUT_PREALLOC
        );
        // 1M u64 elements (8 MiB) in a known-length slice reserve the full 1M capacity at once.
        assert_eq!(
            prealloc_capacity(1_000_000, 8, Some(8_000_000)),
            1_000_000
        );
        // A count within the budget is kept.
        assert_eq!(prealloc_capacity(10, 8, None), 10);
        // A source that knows its length caps the count at its bytes left.
        assert_eq!(prealloc_capacity(1000, 1, Some(5)), 5);
        assert_eq!(prealloc_capacity(1000, 1, Some(0)), 0);
        // Zero-sized elements do not divide by zero.
        assert_eq!(prealloc_capacity(7, 0, None), 7);
        // Elements larger than the budget get no pre-allocation when remaining is unknown.
        assert_eq!(prealloc_capacity(3, MAX_READ_AHEAD_ALLOC + 1, None), 0);
    }

    #[test]
    fn test_s22_r4_vec_u64_known_remaining_preallocates_full_count() {
        use crate::io::data_output::VecDataOutput;
        use crate::io::smart_ptr::SerializableType;

        let n = if cfg!(miri) { 1024 } else { 100_000 };
        let expected: Vec<u64> = (0..n as u64).collect();
        let mut out = VecDataOutput::with_capacity(4 + n * 8);
        expected.serialize(&mut out).unwrap();

        let mut input = SliceDataInput::new(out.as_slice());
        assert_eq!(input.remaining_bytes(), Some((4 + n * 8) as u64));
        let decoded = Vec::<u64>::deserialize(&mut input).unwrap();
        assert_eq!(decoded, expected);
    }
}

