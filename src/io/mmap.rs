//! Memory-mapped I/O implementations
//!
//! This module provides memory-mapped implementations of DataInput and DataOutput
//! traits for high-performance, zero-copy file operations.

use crate::error::{Result, ZiporaError};
#[cfg(feature = "mmap")]
use crate::io::{DataInput, DataOutput, VarInt};
#[cfg(feature = "mmap")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "mmap")]
use std::io::{BufReader, Read as StdRead, Seek, SeekFrom};
#[cfg(feature = "mmap")]
use std::path::Path;

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapMut, MmapOptions};

#[cfg(target_os = "linux")]
#[cfg(feature = "mmap")]
use crate::memory::hugepage::{HUGEPAGE_SIZE_1GB, HUGEPAGE_SIZE_2MB, HugePage};

/// Thresholds for adaptive memory mapping strategy
#[cfg(feature = "mmap")]
const SMALL_FILE_THRESHOLD: u64 = 4 * 1024; // 4KB - use buffered I/O to avoid mmap overhead
#[cfg(feature = "mmap")]
const HUGEPAGE_2MB_THRESHOLD: u64 = 1024 * 1024; // 1MB - use 2MB hugepages
#[cfg(feature = "mmap")]
const HUGEPAGE_1GB_THRESHOLD: u64 = 100 * 1024 * 1024; // 100MB - use 1GB hugepages

/// Access pattern hints for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessPattern {
    /// Sequential access pattern - optimizes for readahead
    Sequential,
    /// Random access pattern - disables readahead, optimizes for TLB efficiency  
    Random,
    /// Mixed access pattern - balanced optimization
    Mixed,
    /// Unknown pattern - uses conservative defaults
    Unknown,
}

/// Strategy used by MemoryMappedInput based on file size and system capabilities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputStrategy {
    /// Use buffered I/O for small files to avoid mmap overhead
    BufferedIO,
    /// Standard memory mapping with 4KB pages
    StandardMmap,
    /// Memory mapping with hugepages (2MB or 1GB)
    /// Note: This currently incurs an upfront copy from the file into the hugepage.
    HugepageMmap,
}

/// Memory-mapped file input for fast reading operations
///
/// Provides efficient reading from memory-mapped files. The standard mmap
/// strategy avoids copying data, while the hugepage strategy incurs an upfront copy.
/// Ideal for large files and random access patterns.
///
/// # Examples
///
/// ```rust
/// use zipora::io::MemoryMappedInput;
/// use zipora::DataInput;
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
    strategy: InputStrategy,
    file_size: u64,
    position: usize,
    access_pattern: AccessPattern,

    // Strategy-specific storage
    mmap: Option<Mmap>,
    buffered_reader: Option<BufReader<File>>,
    #[cfg(target_os = "linux")]
    hugepage: Option<HugePage>,
}

#[cfg(feature = "mmap")]
impl MemoryMappedInput {
    /// Creates a new memory-mapped input from a file with adaptive strategy.
    ///
    /// Limitation: when the file is memory-mapped (`StandardMmap`), `memmap2`
    /// (`MmapOptions::map`, `memmap2` 0.9.9 `src/lib.rs:681-687`) requires that
    /// no other handle or process modifies or truncates the underlying file
    /// while the mapping is alive. Concurrent modification is undefined
    /// behaviour because the mapped pages are exposed as `&[u8]`, and at the OS
    /// level accessing pages past the new end of a truncated file raises
    /// `SIGBUS`, which safe Rust cannot catch.
    pub fn new(file: File) -> Result<Self> {
        Self::new_with_pattern(file, AccessPattern::Unknown)
    }

    /// Creates a new memory-mapped input with access pattern hint.
    ///
    /// Limitation: see [`MemoryMappedInput::new`] — the underlying file must
    /// not be concurrently modified or truncated while mapped (`memmap2` 0.9.9
    /// `src/lib.rs:681-687`; truncation of mapped pages raises `SIGBUS`).
    pub fn new_with_pattern(file: File, access_pattern: AccessPattern) -> Result<Self> {
        let file_size = file.metadata().map_err(ZiporaError::Io)?.len();

        let strategy = Self::select_strategy(file_size);

        let mut input = MemoryMappedInput {
            strategy,
            file_size,
            position: 0,
            access_pattern,
            mmap: None,
            buffered_reader: None,
            #[cfg(target_os = "linux")]
            hugepage: None,
        };

        input.initialize_strategy(file)?;
        Ok(input)
    }

    /// Selects the optimal strategy based on file size and system capabilities
    fn select_strategy(file_size: u64) -> InputStrategy {
        if file_size <= SMALL_FILE_THRESHOLD {
            // For small files, buffered I/O avoids mmap overhead
            InputStrategy::BufferedIO
        } else if file_size >= HUGEPAGE_2MB_THRESHOLD {
            // For large files, try hugepages for better TLB efficiency
            #[cfg(target_os = "linux")]
            {
                if crate::memory::hugepage::hugepages_available() {
                    InputStrategy::HugepageMmap
                } else {
                    InputStrategy::StandardMmap
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                InputStrategy::StandardMmap
            }
        } else {
            // Medium files use standard memory mapping
            InputStrategy::StandardMmap
        }
    }

    /// Initializes the chosen strategy
    fn initialize_strategy(&mut self, mut file: File) -> Result<()> {
        match self.strategy {
            InputStrategy::BufferedIO => {
                // For small files, use buffered I/O to avoid mmap overhead
                let buf_size = std::cmp::min(self.file_size as usize, 8192); // Max 8KB buffer
                file.seek(SeekFrom::Start(0)).map_err(ZiporaError::Io)?;
                self.buffered_reader = Some(BufReader::with_capacity(buf_size, file));
            }

            InputStrategy::StandardMmap => {
                // SAFETY: `MmapOptions::map` requires that the underlying file
                // is not concurrently modified or truncated while the `Mmap` is
                // alive (`memmap2` 0.9.9 `src/lib.rs:681-687`); the caller
                // upholds that external precondition as documented under
                // `Limitation:` on `MemoryMappedInput::new`.
                let mmap = unsafe { MmapOptions::new().map(&file).map_err(ZiporaError::Io)? };

                // Synchronize `file_size` with the actual mapped length in case
                // the file size changed between `metadata()` and `map(&file)`.
                self.file_size = mmap.len() as u64;

                // Apply advanced madvise hints
                self.apply_madvise_hints(&mmap)?;
                self.mmap = Some(mmap);
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                // Try to use hugepages for large files
                let hugepage_size = if self.file_size >= HUGEPAGE_1GB_THRESHOLD {
                    HUGEPAGE_SIZE_1GB
                } else {
                    HUGEPAGE_SIZE_2MB
                };

                match HugePage::new(self.file_size as usize, hugepage_size) {
                    Ok(mut hugepage) => {
                        // Note: "HugepageMmap" is not true zero-copy. It allocates a hugepage and copies the file into it.
                        self.copy_file_to_hugepage(&mut file, &mut hugepage)?;
                        self.hugepage = Some(hugepage);
                    }
                    Err(_) => {
                        // Fallback to standard mmap if hugepage allocation fails
                        self.strategy = InputStrategy::StandardMmap;
                        self.initialize_strategy(file)?;
                    }
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                // Fallback to standard mmap on non-Linux systems
                self.strategy = InputStrategy::StandardMmap;
                self.initialize_strategy(file)?;
            }
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn copy_file_to_hugepage(&mut self, file: &mut File, hugepage: &mut HugePage) -> Result<()> {
        file.seek(SeekFrom::Start(0)).map_err(ZiporaError::Io)?;

        let hugepage_slice = hugepage.as_mut_slice();
        let target_len = hugepage_slice.len();
        let mut total_read = 0;

        while total_read < target_len {
            let chunk_end = (total_read + 64 * 1024).min(target_len);
            match file.read(&mut hugepage_slice[total_read..chunk_end]) {
                Ok(0) => {
                    return Err(ZiporaError::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "file shrank before hugepage copy completed",
                    )));
                }
                Ok(bytes_read) => {
                    total_read += bytes_read;
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(ZiporaError::Io(e)),
            }
        }

        self.file_size = target_len as u64;
        Ok(())
    }

    /// Apply advanced madvise hints for optimal performance.
    ///
    /// Return values from `libc::madvise` and `libc::mlock` are intentionally
    /// ignored because both calls are non-semantic advisory hints: `MADV_HUGEPAGE`
    /// returns `EINVAL` on filesystems or kernels without file-backed THP
    /// support, and `mlock` returns `ENOMEM`/`EPERM` when `RLIMIT_MEMLOCK` is
    /// exceeded on unprivileged processes.
    fn apply_madvise_hints(&self, mmap: &Mmap) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let len = mmap.len();
            if len == 0 {
                return Ok(());
            }
            let ptr = mmap.as_ptr() as *mut libc::c_void;

            // SAFETY: `ptr` and `len` come from a live non-empty `Mmap`, so
            // `[ptr, ptr + len)` is a valid mapped region in this process's
            // address space. `madvise` and `mlock` are advisory kernel hints
            // that do not alter memory contents or Rust aliasing invariants.
            unsafe {
                // Set access pattern hint
                match self.access_pattern {
                    AccessPattern::Sequential => {
                        // Optimize for sequential access with aggressive readahead
                        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
                        libc::madvise(ptr, len, libc::MADV_WILLNEED);

                        // For large sequential files, prefetch the entire file
                        if len > 64 * 1024 {
                            self.prefetch_sequential(ptr, len)?;
                        }
                    }
                    AccessPattern::Random => {
                        // Disable readahead for random access
                        libc::madvise(ptr, len, libc::MADV_RANDOM);

                        // Enable transparent hugepages for better TLB efficiency
                        if len > 2 * 1024 * 1024 {
                            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
                        }
                    }
                    AccessPattern::Mixed => {
                        // Use normal access pattern with some readahead
                        libc::madvise(ptr, len, libc::MADV_NORMAL);

                        // Conservative prefetch for mixed workloads
                        if len > 2 * 1024 * 1024 {
                            libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
                        }
                    }
                    AccessPattern::Unknown => {
                        // Conservative defaults
                        libc::madvise(ptr, len, libc::MADV_NORMAL);
                    }
                }

                // Enable memory locking for small, frequently accessed files
                if len <= 16 * 1024 * 1024 && self.access_pattern == AccessPattern::Sequential {
                    // Try to lock pages in memory (ignore failures)
                    let _ = libc::mlock(ptr, len);
                }
            }
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn prefetch_sequential(&self, ptr: *mut libc::c_void, len: usize) -> Result<()> {
        // Implement intelligent prefetching for sequential access
        const PREFETCH_WINDOW: usize = 2 * 1024 * 1024; // 2MB prefetch window
        const CACHE_LINE_SIZE: usize = 64;

        // SAFETY: `ptr` points to the start of a live `Mmap` of `len` bytes.
        // In each loop iteration `offset < len` and `prefetch_size <= len - offset`,
        // so `prefetch_ptr` and every `cache_line` address lie within
        // `[ptr, ptr + len)`. `madvise` and `_mm_prefetch` are non-faulting
        // advisory hints.
        unsafe {
            let mut offset = 0;
            while offset < len {
                let prefetch_size = std::cmp::min(PREFETCH_WINDOW, len - offset);
                let prefetch_ptr = (ptr as *const u8).add(offset) as *const libc::c_void;

                // Use POSIX_FADV_WILLNEED for async prefetch
                // This is more efficient than synchronous readahead
                libc::madvise(
                    prefetch_ptr as *mut libc::c_void,
                    prefetch_size,
                    libc::MADV_WILLNEED,
                );

                // Hardware prefetch hints for L1/L2 cache
                for i in (0..prefetch_size).step_by(CACHE_LINE_SIZE) {
                    let cache_line = (prefetch_ptr as *const u8).add(i);
                    #[cfg(target_arch = "x86_64")]
                    {
                        // Use PREFETCHNTA for streaming data
                        std::arch::x86_64::_mm_prefetch(
                            cache_line as *const i8,
                            std::arch::x86_64::_MM_HINT_NTA,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    let _ = cache_line;
                }

                offset += prefetch_size;
            }
        }

        Ok(())
    }

    /// Creates a new memory-mapped input from a file path.
    ///
    /// Limitation: see [`MemoryMappedInput::new`] — the underlying file must
    /// not be concurrently modified or truncated while mapped (`memmap2` 0.9.9
    /// `src/lib.rs:681-687`; truncation of mapped pages raises `SIGBUS`).
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(ZiporaError::Io)?;
        Self::new(file)
    }

    /// Creates a new memory-mapped input from a file path with access pattern hint.
    ///
    /// Limitation: see [`MemoryMappedInput::new`] — the underlying file must
    /// not be concurrently modified or truncated while mapped (`memmap2` 0.9.9
    /// `src/lib.rs:681-687`; truncation of mapped pages raises `SIGBUS`).
    pub fn from_path_with_pattern<P: AsRef<Path>>(
        path: P,
        access_pattern: AccessPattern,
    ) -> Result<Self> {
        let file = File::open(path).map_err(ZiporaError::Io)?;
        Self::new_with_pattern(file, access_pattern)
    }

    /// Returns the total size of the file
    #[inline]
    pub fn len(&self) -> usize {
        self.file_size as usize
    }

    /// Returns true if the file is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.file_size == 0
    }

    /// Returns the current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the strategy being used
    pub fn strategy(&self) -> InputStrategy {
        self.strategy
    }

    /// Seeks to a specific position
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.file_size as usize {
            return Err(ZiporaError::out_of_bounds(pos, self.file_size as usize));
        }

        match self.strategy {
            InputStrategy::BufferedIO => {
                if let Some(ref mut reader) = self.buffered_reader {
                    reader
                        .seek(SeekFrom::Start(pos as u64))
                        .map_err(ZiporaError::Io)?;
                }
            }
            _ => {
                // For memory-mapped strategies, just update position
            }
        }

        self.position = pos;
        Ok(())
    }

    /// Returns the remaining bytes available for reading
    pub fn remaining(&self) -> usize {
        (self.file_size as usize).saturating_sub(self.position)
    }

    /// Reads a slice of bytes (zero-copy when possible)
    pub fn read_slice(&mut self, len: usize) -> Result<Vec<u8>> {
        let file_len = self.len();
        let end_pos = self
            .position
            .checked_add(len)
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, file_len))?;
        if end_pos > file_len {
            return Err(ZiporaError::out_of_bounds(end_pos, file_len));
        }

        let data = match self.strategy {
            InputStrategy::BufferedIO => {
                if let Some(ref mut reader) = self.buffered_reader {
                    let mut buffer = vec![0u8; len];
                    if let Err(e) = reader.read_exact(&mut buffer) {
                        // Restore the reader position to `self.position` so a
                        // partial read before EOF/error does not desynchronize
                        // `BufReader` from `self.position` (C9 I4).
                        let _ = reader.seek(SeekFrom::Start(self.position as u64));
                        return Err(ZiporaError::Io(e));
                    }
                    buffer
                } else {
                    return Err(ZiporaError::invalid_data("Buffered reader not initialized"));
                }
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    if end_pos > mmap.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, mmap.len()));
                    }
                    mmap[self.position..end_pos].to_vec()
                } else {
                    return Err(ZiporaError::invalid_data("Memory map not initialized"));
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    let slice = hugepage.as_slice();
                    if end_pos > slice.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, slice.len()));
                    }
                    slice[self.position..end_pos].to_vec()
                } else {
                    return Err(ZiporaError::invalid_data("Hugepage not initialized"));
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                return Err(ZiporaError::not_supported(
                    "Hugepages not available on this platform",
                ));
            }
        };

        self.position = end_pos;
        Ok(data)
    }

    /// Reads a slice of bytes without copying (zero-copy, memory-mapped only)
    pub fn read_slice_zero_copy(&mut self, len: usize) -> Result<&[u8]> {
        let file_len = self.len();
        let end_pos = self
            .position
            .checked_add(len)
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, file_len))?;
        if end_pos > file_len {
            return Err(ZiporaError::out_of_bounds(end_pos, file_len));
        }

        let slice = match self.strategy {
            InputStrategy::BufferedIO => {
                return Err(ZiporaError::not_supported(
                    "Zero-copy not available for buffered I/O strategy",
                ));
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    if end_pos > mmap.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, mmap.len()));
                    }
                    &mmap[self.position..end_pos]
                } else {
                    return Err(ZiporaError::invalid_data("Memory map not initialized"));
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    let hp_slice = hugepage.as_slice();
                    if end_pos > hp_slice.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, hp_slice.len()));
                    }
                    &hp_slice[self.position..end_pos]
                } else {
                    return Err(ZiporaError::invalid_data("Hugepage not initialized"));
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                return Err(ZiporaError::not_supported(
                    "Hugepages not available on this platform",
                ));
            }
        };

        self.position = end_pos;
        Ok(slice)
    }

    /// Peeks at bytes without advancing the position (zero-copy when possible)
    pub fn peek_slice(&self, len: usize) -> Result<Vec<u8>> {
        let file_len = self.len();
        let end_pos = self
            .position
            .checked_add(len)
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, file_len))?;
        if end_pos > file_len {
            return Err(ZiporaError::out_of_bounds(end_pos, file_len));
        }

        match self.strategy {
            InputStrategy::BufferedIO => {
                // For buffered I/O, peek is not efficiently supported
                Err(ZiporaError::not_supported(
                    "Peek not efficiently supported for buffered I/O strategy",
                ))
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    if end_pos > mmap.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, mmap.len()));
                    }
                    Ok(mmap[self.position..end_pos].to_vec())
                } else {
                    Err(ZiporaError::invalid_data("Memory map not initialized"))
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    let slice = hugepage.as_slice();
                    if end_pos > slice.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, slice.len()));
                    }
                    Ok(slice[self.position..end_pos].to_vec())
                } else {
                    Err(ZiporaError::invalid_data("Hugepage not initialized"))
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => Err(ZiporaError::not_supported(
                "Hugepages not available on this platform",
            )),
        }
    }

    /// Peeks at bytes without advancing the position (zero-copy, memory-mapped only)
    pub fn peek_slice_zero_copy(&self, len: usize) -> Result<&[u8]> {
        let file_len = self.len();
        let end_pos = self
            .position
            .checked_add(len)
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, file_len))?;
        if end_pos > file_len {
            return Err(ZiporaError::out_of_bounds(end_pos, file_len));
        }

        match self.strategy {
            InputStrategy::BufferedIO => Err(ZiporaError::not_supported(
                "Zero-copy peek not available for buffered I/O strategy",
            )),

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    if end_pos > mmap.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, mmap.len()));
                    }
                    Ok(&mmap[self.position..end_pos])
                } else {
                    Err(ZiporaError::invalid_data("Memory map not initialized"))
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    let slice = hugepage.as_slice();
                    if end_pos > slice.len() {
                        return Err(ZiporaError::out_of_bounds(end_pos, slice.len()));
                    }
                    Ok(&slice[self.position..end_pos])
                } else {
                    Err(ZiporaError::invalid_data("Hugepage not initialized"))
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => Err(ZiporaError::not_supported(
                "Hugepages not available on this platform",
            )),
        }
    }
}

#[cfg(feature = "mmap")]
impl DataInput for MemoryMappedInput {
    fn read_u8(&mut self) -> Result<u8> {
        let data = self.read_slice(1)?;
        Ok(data[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let data = self.read_slice(2)?;
        Ok(u16::from_le_bytes([data[0], data[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let data = self.read_slice(4)?;
        Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let data = self.read_slice(8)?;
        Ok(u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
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
                return Err(ZiporaError::invalid_data("Variable integer overflow"));
            }
        }

        Err(ZiporaError::invalid_data("Variable integer too long"))
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<()> {
        let data = self.read_slice(buf.len())?;
        buf.copy_from_slice(&data);
        Ok(())
    }

    fn read_length_prefixed_string(&mut self) -> Result<String> {
        let raw_len = self.read_var_int()?;
        let len = usize::try_from(raw_len)
            .map_err(|_| ZiporaError::out_of_bounds(usize::MAX, self.len()))?;
        let data = self.read_slice(len)?;
        String::from_utf8(data)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid UTF-8 string: {}", e)))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let new_pos = self
            .position
            .checked_add(n)
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, self.len()))?;
        self.seek(new_pos)
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
/// use zipora::io::MemoryMappedOutput;
/// use zipora::DataOutput;
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
    mmap: Option<MmapMut>,
    position: usize,
}

#[cfg(feature = "mmap")]
impl MemoryMappedOutput {
    /// Creates a new memory-mapped output file with specified initial size.
    ///
    /// Limitation: `MmapOptions::map_mut` cannot prevent another handle or
    /// process from concurrently modifying or truncating the underlying file
    /// while this output mapping is alive (`memmap2` 0.9.9 `src/lib.rs:725-731`).
    /// Concurrent access is undefined behaviour (`&mut [u8]` aliasing) and OS
    /// file truncation beneath a live mapping raises `SIGBUS`, which safe Rust
    /// cannot catch.
    pub fn create<P: AsRef<Path>>(path: P, initial_size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(ZiporaError::Io)?;

        // Set the file size
        file.set_len(initial_size as u64).map_err(ZiporaError::Io)?;

        // SAFETY: `MmapOptions::map_mut` requires that the underlying file is
        // not concurrently modified or truncated out-of-band while the mapping
        // is alive (`memmap2` 0.9.9 `src/lib.rs:725-731`); the caller upholds
        // that external precondition as documented under `Limitation:` above.
        let mmap = unsafe { MmapOptions::new().map_mut(&file).map_err(ZiporaError::Io)? };

        Ok(MemoryMappedOutput {
            file,
            mmap: Some(mmap),
            position: 0,
        })
    }

    /// Opens an existing file for memory-mapped writing.
    ///
    /// Limitation: see [`MemoryMappedOutput::create`] — the underlying file
    /// must not be concurrently modified or truncated by another handle or
    /// process while mapped (`memmap2` 0.9.9 `src/lib.rs:725-731`; accessing
    /// truncated pages raises `SIGBUS`).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(ZiporaError::Io)?;

        // SAFETY: `MmapOptions::map_mut` requires that the underlying file is
        // not concurrently modified or truncated out-of-band while the mapping
        // is alive (`memmap2` 0.9.9 `src/lib.rs:725-731`); the caller upholds
        // that external precondition as documented under `Limitation:` above.
        let mmap = unsafe { MmapOptions::new().map_mut(&file).map_err(ZiporaError::Io)? };

        Ok(MemoryMappedOutput {
            file,
            mmap: Some(mmap),
            position: 0,
        })
    }

    /// Returns the current position in the mapped region
    pub fn position(&self) -> usize {
        self.position
    }

    /// Returns the total capacity of the mapped region
    #[inline]
    pub fn capacity(&self) -> usize {
        self.mmap.as_ref().map_or(0, |m| m.len())
    }

    /// Returns the remaining space available for writing
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.position)
    }

    /// Seeks to a specific position in the mapped region
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        let cap = self.capacity();
        if pos > cap {
            return Err(ZiporaError::out_of_bounds(pos, cap));
        }
        self.position = pos;
        Ok(())
    }

    /// Ensures the mapped region has at least the specified capacity
    fn ensure_capacity(&mut self, required: usize) -> Result<()> {
        let current_capacity = self.capacity();
        if self.mmap.is_some() && required <= current_capacity {
            return Ok(());
        }

        // Grow by 50% or to required size, whichever is larger
        let grown = current_capacity.saturating_add(current_capacity / 2);
        let new_size = std::cmp::max(required, grown);

        // Drop the current mapping before resizing the file
        self.mmap = None;

        if let Err(e) = self.file.set_len(new_size as u64) {
            // SAFETY: `self.file` is the open file handle for this output. If
            // remapping at the existing file size succeeds, we restore the
            // mapping; otherwise `self.mmap` stays `None` (`capacity() == 0`).
            // The caller upholds `memmap2`'s no-external-mutation contract.
            self.mmap = unsafe { MmapOptions::new().map_mut(&self.file) }.ok();
            return Err(ZiporaError::Io(e));
        }

        // SAFETY: `self.file` was resized to `new_size` above, and the caller
        // upholds `memmap2`'s requirement (`memmap2` 0.9.9 `src/lib.rs:725-731`)
        // that the file is not concurrently modified or truncated out-of-band.
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&self.file)
                .map_err(ZiporaError::Io)?
        };

        self.mmap = Some(mmap);
        Ok(())
    }

    /// Writes a slice of bytes to the mapped region
    pub fn write_slice(&mut self, data: &[u8]) -> Result<()> {
        let required_capacity = self
            .position
            .checked_add(data.len())
            .ok_or_else(|| ZiporaError::out_of_bounds(usize::MAX, self.capacity()))?;
        self.ensure_capacity(required_capacity)?;

        let mmap = self
            .mmap
            .as_mut()
            .ok_or_else(|| ZiporaError::invalid_data("Memory map not initialized"))?;
        mmap[self.position..required_capacity].copy_from_slice(data);
        self.position = required_capacity;
        Ok(())
    }

    /// Truncates the file to the current position
    pub fn truncate(&mut self) -> Result<()> {
        if let Some(ref mmap) = self.mmap {
            mmap.flush().map_err(ZiporaError::Io)?;
        }

        // Unmap before truncating the underlying file
        self.mmap = None;

        if let Err(e) = self.file.set_len(self.position as u64) {
            // SAFETY: `self.file` is the open file handle for this output. If
            // remapping at the existing file size succeeds, we restore the
            // mapping; otherwise `self.mmap` stays `None` (`capacity() == 0`).
            // The caller upholds `memmap2`'s no-external-mutation contract.
            self.mmap = unsafe { MmapOptions::new().map_mut(&self.file) }.ok();
            return Err(ZiporaError::Io(e));
        }

        // SAFETY: `self.file` was truncated to `self.position` above, and the
        // caller upholds `memmap2`'s requirement (`memmap2` 0.9.9
        // `src/lib.rs:725-731`) that the file is not concurrently modified or
        // truncated out-of-band while mapped.
        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&self.file)
                .map_err(ZiporaError::Io)?
        };

        self.mmap = Some(mmap);
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
        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| ZiporaError::invalid_data("Memory map not initialized"))?;
        mmap.flush().map_err(ZiporaError::Io)
    }
}

// Fallback definitions when the mmap feature is disabled
#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedInput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedInput {
    pub fn new(_file: std::fs::File) -> Result<Self> {
        Err(ZiporaError::not_supported(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }

    pub fn from_path<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(ZiporaError::not_supported(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }
}

#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedOutput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedOutput {
    pub fn create<P: AsRef<std::path::Path>>(_path: P, _initial_size: usize) -> Result<Self> {
        Err(ZiporaError::not_supported(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedOutput.",
        ))
    }

    pub fn open<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        Err(ZiporaError::not_supported(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedOutput.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "mmap")]
    use std::io::Write;
    use tempfile::NamedTempFile;
    #[cfg(feature = "mmap")]
    use tempfile::TempDir;

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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_memory_mapped_input_peek_and_skip() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Create a larger file to force standard mmap strategy
        let mut test_data = b"0123456789".to_vec();
        test_data.resize(8192, b'X'); // 8KB file - above 4KB threshold
        temp_file.write_all(&test_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Should use standard mmap for larger file
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
        assert_eq!(read_data, large_data);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
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

    // Tests for adaptive memory mapping behavior
    #[cfg(feature = "mmap")]
    #[test]
    fn test_small_file_uses_buffered_io() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let small_data = vec![0x42u8; 2048]; // 2KB file - below 4KB threshold
        temp_file.write_all(&small_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new(file).unwrap();

        // Should use buffered I/O for small files
        assert_eq!(input.strategy(), InputStrategy::BufferedIO);
        assert_eq!(input.len(), 2048);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_medium_file_uses_standard_mmap() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let medium_data = vec![0x55u8; 64 * 1024]; // 64KB file - between 4KB and 1MB
        temp_file.write_all(&medium_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new(file).unwrap();

        // Should use standard memory mapping for medium files
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);
        assert_eq!(input.len(), 64 * 1024);
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_large_file_attempts_hugepages() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let large_data = vec![0x77u8; 2 * 1024 * 1024]; // 2MB file - above hugepage threshold
        temp_file.write_all(&large_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new(file).unwrap();

        // Should attempt hugepages (may fallback to standard mmap if hugepages unavailable)
        let strategy = input.strategy();
        assert!(
            strategy == InputStrategy::HugepageMmap || strategy == InputStrategy::StandardMmap,
            "Large file should use hugepages or fallback to standard mmap"
        );
        assert_eq!(input.len(), 2 * 1024 * 1024);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_access_pattern_hints() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = vec![0x88u8; 16 * 1024]; // 16KB file
        temp_file.write_all(&test_data).unwrap();
        temp_file.flush().unwrap();

        // Test sequential access pattern
        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new_with_pattern(file, AccessPattern::Sequential).unwrap();
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

        // Test random access pattern
        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new_with_pattern(file, AccessPattern::Random).unwrap();
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

        // Test mixed access pattern
        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new_with_pattern(file, AccessPattern::Mixed).unwrap();
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_adaptive_read_operations() {
        // Test small file with buffered I/O
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&0x12345678u32.to_le_bytes()).unwrap();
        temp_file
            .write_all(&0x9ABCDEF012345678u64.to_le_bytes())
            .unwrap();
        let var_bytes = VarInt::encode(42);
        temp_file.write_all(&var_bytes).unwrap();
        let test_str = "Adaptive test";
        let str_len_bytes = VarInt::encode(test_str.len() as u64);
        temp_file.write_all(&str_len_bytes).unwrap();
        temp_file.write_all(test_str.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Should use buffered I/O for small file
        assert_eq!(input.strategy(), InputStrategy::BufferedIO);

        // Test that all read operations work correctly
        assert_eq!(input.read_u32().unwrap(), 0x12345678);
        assert_eq!(input.read_u64().unwrap(), 0x9ABCDEF012345678);
        assert_eq!(input.read_var_int().unwrap(), 42);
        assert_eq!(input.read_length_prefixed_string().unwrap(), test_str);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_zero_copy_operations() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_data = vec![0x99u8; 32 * 1024]; // 32KB file - uses standard mmap
        temp_file.write_all(&test_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

        // Test zero-copy read
        let slice = input.read_slice_zero_copy(1024).unwrap();
        assert_eq!(slice.len(), 1024);
        assert!(slice.iter().all(|&b| b == 0x99));

        // Test zero-copy peek
        let peeked = input.peek_slice_zero_copy(512).unwrap();
        assert_eq!(peeked.len(), 512);
        assert!(peeked.iter().all(|&b| b == 0x99));

        // Position should not have changed after peek
        assert_eq!(input.position(), 1024);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_buffered_io_limitations() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let small_data = vec![0xAAu8; 1024]; // Small file - uses buffered I/O
        temp_file.write_all(&small_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        assert_eq!(input.strategy(), InputStrategy::BufferedIO);

        // Zero-copy operations should not be available for buffered I/O
        assert!(input.read_slice_zero_copy(100).is_err());
        assert!(input.peek_slice_zero_copy(100).is_err());

        // Regular peek should also not be efficiently supported
        assert!(input.peek_slice(100).is_err());

        // But regular read operations should work
        let data = input.read_slice(100).unwrap();
        assert_eq!(data.len(), 100);
        assert!(data.iter().all(|&b| b == 0xAA));
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_seek_and_position_adaptive() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // Create a larger file to force standard mmap strategy for consistent behavior
        let original_data = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        temp_file.write_all(original_data).unwrap();
        // Pad with zeros to make it larger than 4KB threshold
        let padding = vec![0u8; 8192 - original_data.len()];
        temp_file.write_all(&padding).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Should use standard mmap for larger file
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

        assert_eq!(input.position(), 0);
        assert_eq!(input.remaining(), 8192);

        // Test seeking
        input.seek(10).unwrap();
        assert_eq!(input.position(), 10);
        assert_eq!(input.remaining(), 8192 - 10);

        // Read and verify position updates
        let data = input.read_slice(5).unwrap();
        assert_eq!(data, b"ABCDE");
        assert_eq!(input.position(), 15);

        // Test skip
        input.skip(5).unwrap();
        assert_eq!(input.position(), 20);

        let data = input.read_slice(3).unwrap();
        // Position 20-22 in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" is "KLM"
        assert_eq!(data, b"KLM");
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_bounds_checking_adaptive() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"Small").unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();

        // Should use buffered I/O
        assert_eq!(input.strategy(), InputStrategy::BufferedIO);

        // Reading beyond bounds should fail
        assert!(input.read_slice(10).is_err());

        // Seeking beyond bounds should fail
        assert!(input.seek(20).is_err());

        // Reading exactly at bounds should work
        let data = input.read_slice(5).unwrap();
        assert_eq!(data, b"Small");

        // Reading after consuming all data should fail
        assert!(input.read_slice(1).is_err());
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_file_size_thresholds() {
        // Test exactly at threshold boundaries
        let threshold_tests = vec![
            (SMALL_FILE_THRESHOLD as usize - 1, InputStrategy::BufferedIO), // Just below 4KB
            (SMALL_FILE_THRESHOLD as usize, InputStrategy::BufferedIO),     // Exactly 4KB
            (
                SMALL_FILE_THRESHOLD as usize + 1,
                InputStrategy::StandardMmap,
            ), // Just above 4KB
            (
                HUGEPAGE_2MB_THRESHOLD as usize - 1,
                InputStrategy::StandardMmap,
            ), // Just below 1MB
        ];

        for (size, expected_strategy) in threshold_tests {
            let mut temp_file = NamedTempFile::new().unwrap();
            let test_data = vec![0xBBu8; size];
            temp_file.write_all(&test_data).unwrap();
            temp_file.flush().unwrap();

            let file = File::open(temp_file.path()).unwrap();
            let input = MemoryMappedInput::new(file).unwrap();

            assert_eq!(
                input.strategy(),
                expected_strategy,
                "File size {} should use strategy {:?}, got {:?}",
                size,
                expected_strategy,
                input.strategy()
            );
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_hugepage_fallback() {
        // Create a large file that would normally use hugepages
        let mut temp_file = NamedTempFile::new().unwrap();
        let large_data = vec![0xCCu8; 5 * 1024 * 1024]; // 5MB file
        temp_file.write_all(&large_data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let input = MemoryMappedInput::new(file).unwrap();

        // Should attempt hugepages but may fallback to standard mmap
        let strategy = input.strategy();
        assert!(
            strategy == InputStrategy::HugepageMmap || strategy == InputStrategy::StandardMmap,
            "Large file should attempt hugepages with graceful fallback"
        );

        // File should still be readable regardless of strategy
        assert_eq!(input.len(), 5 * 1024 * 1024);
    }

    // -----------------------------------------------------------------------
    // C9 regression and guard tests (I1-I4, zero-length mmap, MAP_SHARED, madvise)
    // -----------------------------------------------------------------------

    #[cfg(feature = "mmap")]
    #[test]
    fn test_c9_i1_buffered_read_and_skip_reject_position_plus_len_overflow() {
        let mut temp_file = NamedTempFile::new().unwrap();
        // 1 leading byte followed by a 10-byte varint encoding u64::MAX
        let mut payload = vec![0xAAu8];
        payload.extend_from_slice(&VarInt::encode(u64::MAX));
        temp_file.write_all(&payload).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();
        assert_eq!(input.strategy(), InputStrategy::BufferedIO);

        assert_eq!(input.read_u8().unwrap(), 0xAA);
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.read_slice(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.read_slice_zero_copy(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.peek_slice(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.peek_slice_zero_copy(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.skip(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        // Reading a length-prefixed string whose varint is u64::MAX must fail
        // with OutOfBounds instead of overflowing `position + len` or panicking.
        assert!(matches!(
            input.read_length_prefixed_string(),
            Err(ZiporaError::OutOfBounds { .. })
        ));
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_i1_mmap_read_and_output_write_reject_position_plus_len_overflow() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&vec![0x5Au8; 8192]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();
        assert_eq!(input.strategy(), InputStrategy::StandardMmap);

        input.seek(1).unwrap();
        assert!(matches!(
            input.read_slice(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.read_slice_zero_copy(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.peek_slice(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.peek_slice_zero_copy(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        assert!(matches!(
            input.skip(usize::MAX),
            Err(ZiporaError::OutOfBounds { .. })
        ));
        assert_eq!(input.position(), 1);

        let temp_dir = TempDir::new().unwrap();
        let out_path = temp_dir.path().join("i1_out.dat");
        let mut output = MemoryMappedOutput::create(&out_path, 16).unwrap();
        output.position = usize::MAX;
        assert!(matches!(
            output.write_slice(&[1]),
            Err(ZiporaError::OutOfBounds { .. })
        ));
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_i2_failed_grow_and_truncate_do_not_leave_stale_capacity() {
        let temp_dir = TempDir::new().unwrap();

        // 1. Failed grow in ensure_capacity must not leave capacity > mmap.len()
        //    (which caused the next write within old capacity to panic by
        //    indexing a 0-byte map_anon placeholder).
        let grow_path = temp_dir.path().join("i2_grow.dat");
        let mut output = MemoryMappedOutput::create(&grow_path, 64).unwrap();
        output.write_slice(b"hello").unwrap();
        assert_eq!(output.position(), 5);
        assert_eq!(output.capacity(), 64);

        // Replace `self.file` with a read-only handle so `set_len` fails.
        output.file = File::open(&grow_path).unwrap();
        assert!(output.write_slice(&[0xAB; 128]).is_err());
        assert_eq!(output.position(), 5);
        let _ = output.seek(0);
        // Must either succeed (if the previous mapping was preserved) or return
        // Err (if unmapped), never panic on a 0-length placeholder slice.
        let _ = output.write_u8(b'H');

        // 2. Failed truncate must likewise never leave stale capacity over a
        //    0-length placeholder mapping.
        let trunc_path = temp_dir.path().join("i2_trunc.dat");
        let mut output2 = MemoryMappedOutput::create(&trunc_path, 64).unwrap();
        output2.write_slice(b"hello").unwrap();
        output2.file = File::open(&trunc_path).unwrap();
        assert!(output2.truncate().is_err());
        let _ = output2.seek(0);
        let _ = output2.write_u8(b'H');
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_i3_mmap_tracks_actual_mapping_len_on_shrink() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&vec![0x33u8; 8192]).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        // Simulate the file shrinking between metadata() and initialize_strategy().
        temp_file.as_file().set_len(4096).unwrap();

        let mut input = MemoryMappedInput {
            strategy: InputStrategy::StandardMmap,
            file_size: 8192,
            position: 0,
            access_pattern: AccessPattern::Unknown,
            mmap: None,
            buffered_reader: None,
            #[cfg(target_os = "linux")]
            hugepage: None,
        };
        input.initialize_strategy(file).unwrap();

        // Both `read_slice` and `len()` must respect the actual mapped size
        // (4096) rather than indexing `mmap[0..5000]` and panicking.
        assert!(input.read_slice(5000).is_err());
        assert!(input.read_slice_zero_copy(5000).is_err());
        assert!(input.peek_slice(5000).is_err());
        assert!(input.peek_slice_zero_copy(5000).is_err());
        assert_eq!(input.len(), 4096);
        assert_eq!(input.read_slice(4096).unwrap().len(), 4096);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn test_c9_i4_buffered_io_preserves_error_kind_and_restores_reader_position() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let data: Vec<u8> = (0..100u8).collect();
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();

        let file = File::open(temp_file.path()).unwrap();
        let mut input = MemoryMappedInput::new(file).unwrap();
        assert_eq!(input.strategy(), InputStrategy::BufferedIO);

        // Truncate the underlying file to 10 bytes so a 20-byte read hits EOF
        // after advancing the underlying BufReader 10 bytes.
        temp_file.as_file().set_len(10).unwrap();

        let err = input.read_slice(20).unwrap_err();
        match err {
            ZiporaError::Io(ref io_err) => {
                assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected ZiporaError::Io(UnexpectedEof), got {other:?}"),
        }
        assert_eq!(input.position(), 0);

        // Because `position` is still 0, reading the 5 bytes that remain at
        // offset 0..5 must succeed and return [0, 1, 2, 3, 4].
        let prefix = input.read_slice(5).unwrap();
        assert_eq!(prefix, vec![0, 1, 2, 3, 4]);
        assert_eq!(input.position(), 5);
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_empty_file_input_and_output() {
        let temp_dir = TempDir::new().unwrap();
        let empty_path = temp_dir.path().join("empty.dat");

        // Create a 0-byte file via MemoryMappedOutput::create(..., 0)
        {
            let mut out = MemoryMappedOutput::create(&empty_path, 0).unwrap();
            assert_eq!(out.capacity(), 0);
            assert_eq!(out.position(), 0);
            assert_eq!(out.remaining(), 0);
            out.write_slice(&[]).unwrap();
            out.flush().unwrap();
            out.truncate().unwrap();
            assert_eq!(out.capacity(), 0);
        }

        // Read the 0-byte file via MemoryMappedInput::from_path / new / new_with_pattern
        {
            let mut input = MemoryMappedInput::from_path(&empty_path).unwrap();
            assert_eq!(input.len(), 0);
            assert!(input.is_empty());
            assert_eq!(input.remaining(), 0);
            assert!(input.read_slice(0).unwrap().is_empty());
            assert!(input.read_u8().is_err());

            let file = File::open(&empty_path).unwrap();
            let mut input2 =
                MemoryMappedInput::new_with_pattern(file, AccessPattern::Sequential).unwrap();
            assert_eq!(input2.len(), 0);
            assert!(input2.read_slice(0).unwrap().is_empty());
        }

        // Open the existing 0-byte file via MemoryMappedOutput::open, then grow it
        {
            let mut out = MemoryMappedOutput::open(&empty_path).unwrap();
            assert_eq!(out.capacity(), 0);
            out.write_slice(b"grown from zero").unwrap();
            out.truncate().unwrap();
        }

        let mut input = MemoryMappedInput::from_path(&empty_path).unwrap();
        assert_eq!(input.read_slice(15).unwrap(), b"grown from zero");
    }

    #[cfg(feature = "mmap")]
    #[test]
    #[cfg_attr(miri, ignore = "real file mapping: Miri cannot execute mmap")]
    fn test_c9_map_shared_persistence_without_explicit_flush_and_madvise_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let shared_path = temp_dir.path().join("map_shared.dat");

        // Write 96 KiB (> 64 KiB so Sequential exercises prefetch_sequential)
        // and drop MemoryMappedOutput WITHOUT calling flush(). Because
        // MmapOptions::map_mut uses MAP_SHARED, the writes must persist across
        // drop + reopen.
        let expected: Vec<u8> = (0..(96 * 1024)).map(|i| (i % 251) as u8).collect();
        {
            let mut out = MemoryMappedOutput::create(&shared_path, expected.len()).unwrap();
            out.write_slice(&expected).unwrap();
            // Intentionally drop `out` without `out.flush()`.
        }

        for pattern in [
            AccessPattern::Sequential,
            AccessPattern::Random,
            AccessPattern::Mixed,
            AccessPattern::Unknown,
        ] {
            let mut input =
                MemoryMappedInput::from_path_with_pattern(&shared_path, pattern).unwrap();
            assert_eq!(input.strategy(), InputStrategy::StandardMmap);
            assert_eq!(input.len(), expected.len());
            let zc = input.read_slice_zero_copy(expected.len()).unwrap();
            assert_eq!(zc, expected.as_slice());
        }
    }
}
