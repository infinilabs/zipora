//! Memory-mapped I/O implementations
//!
//! This module provides memory-mapped implementations of DataInput and DataOutput
//! traits for high-performance, zero-copy file operations.

use crate::error::{Result, ZiporaError};
use crate::io::{DataInput, DataOutput, VarInt};
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::io::{BufReader, Read as StdRead, Seek, SeekFrom};

#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapMut, MmapOptions};

#[cfg(target_os = "linux")]
use crate::memory::hugepage::{HugePage, HUGEPAGE_SIZE_2MB, HUGEPAGE_SIZE_1GB};

/// Thresholds for adaptive memory mapping strategy
const SMALL_FILE_THRESHOLD: u64 = 4 * 1024; // 4KB - use buffered I/O to avoid mmap overhead
const HUGEPAGE_2MB_THRESHOLD: u64 = 1024 * 1024; // 1MB - use 2MB hugepages
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
    HugepageMmap,
}

/// Memory-mapped file input for zero-copy reading operations
///
/// Provides efficient reading from memory-mapped files without copying data.
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
    /// Creates a new memory-mapped input from a file with adaptive strategy
    pub fn new(file: File) -> Result<Self> {
        Self::new_with_pattern(file, AccessPattern::Unknown)
    }

    /// Creates a new memory-mapped input with access pattern hint
    pub fn new_with_pattern(file: File, access_pattern: AccessPattern) -> Result<Self> {
        let file_size = file.metadata()
            .map_err(|e| ZiporaError::io_error(format!("Failed to get file metadata: {}", e)))?
            .len();

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
                file.seek(SeekFrom::Start(0))
                    .map_err(|e| ZiporaError::io_error(format!("Failed to seek file: {}", e)))?;
                self.buffered_reader = Some(BufReader::with_capacity(buf_size, file));
            }

            InputStrategy::StandardMmap => {
                let mmap = unsafe {
                    MmapOptions::new()
                        .map(&file)
                        .map_err(|e| ZiporaError::io_error(format!("Failed to memory-map file: {}", e)))?
                };

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
                    Ok(hugepage) => {
                        // Copy file data to hugepage memory for zero-copy access
                        self.copy_file_to_hugepage(&mut file, &hugepage)?;
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
    fn copy_file_to_hugepage(&mut self, file: &mut File, hugepage: &HugePage) -> Result<()> {
        file.seek(SeekFrom::Start(0))
            .map_err(|e| ZiporaError::io_error(format!("Failed to seek file: {}", e)))?;
        
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB read buffer
        let mut total_read = 0;
        let hugepage_slice = unsafe {
            std::slice::from_raw_parts_mut(
                hugepage.as_slice().as_ptr() as *mut u8,
                hugepage.size()
            )
        };

        while total_read < self.file_size as usize {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| ZiporaError::io_error(format!("Failed to read file: {}", e)))?;
            
            if bytes_read == 0 {
                break;
            }

            hugepage_slice[total_read..total_read + bytes_read]
                .copy_from_slice(&buffer[..bytes_read]);
            total_read += bytes_read;
        }

        Ok(())
    }

    /// Apply advanced madvise hints for optimal performance
    fn apply_madvise_hints(&self, mmap: &Mmap) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            let len = mmap.len();

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

        unsafe {
            let mut offset = 0;
            while offset < len {
                let prefetch_size = std::cmp::min(PREFETCH_WINDOW, len - offset);
                let prefetch_ptr = (ptr as *const u8).add(offset) as *const libc::c_void;
                
                // Use POSIX_FADV_WILLNEED for async prefetch
                // This is more efficient than synchronous readahead
                libc::madvise(prefetch_ptr as *mut libc::c_void, prefetch_size, libc::MADV_WILLNEED);
                
                // Hardware prefetch hints for L1/L2 cache
                for i in (0..prefetch_size).step_by(CACHE_LINE_SIZE) {
                    let cache_line = (prefetch_ptr as *const u8).add(i);
                    #[cfg(target_arch = "x86_64")]
                    {
                        // Use PREFETCHNTA for streaming data
                        std::arch::x86_64::_mm_prefetch(
                            cache_line as *const i8, 
                            std::arch::x86_64::_MM_HINT_NTA
                        );
                    }
                }
                
                offset += prefetch_size;
            }
        }

        Ok(())
    }

    /// Creates a new memory-mapped input from a file path
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to open file: {}", e)))?;
        Self::new(file)
    }

    /// Creates a new memory-mapped input from a file path with access pattern hint
    pub fn from_path_with_pattern<P: AsRef<Path>>(path: P, access_pattern: AccessPattern) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| ZiporaError::io_error(format!("Failed to open file: {}", e)))?;
        Self::new_with_pattern(file, access_pattern)
    }

    /// Returns the total size of the file
    pub fn len(&self) -> usize {
        self.file_size as usize
    }

    /// Returns true if the file is empty
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
                    reader.seek(SeekFrom::Start(pos as u64))
                        .map_err(|e| ZiporaError::io_error(format!("Failed to seek: {}", e)))?;
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
        let end_pos = self.position + len;
        if end_pos > self.file_size as usize {
            return Err(ZiporaError::out_of_bounds(end_pos, self.file_size as usize));
        }

        let data = match self.strategy {
            InputStrategy::BufferedIO => {
                if let Some(ref mut reader) = self.buffered_reader {
                    let mut buffer = vec![0u8; len];
                    reader.read_exact(&mut buffer)
                        .map_err(|e| ZiporaError::io_error(format!("Failed to read: {}", e)))?;
                    buffer
                } else {
                    return Err(ZiporaError::invalid_data("Buffered reader not initialized"));
                }
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    mmap[self.position..end_pos].to_vec()
                } else {
                    return Err(ZiporaError::invalid_data("Memory map not initialized"));
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    hugepage.as_slice()[self.position..end_pos].to_vec()
                } else {
                    return Err(ZiporaError::invalid_data("Hugepage not initialized"));
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                return Err(ZiporaError::not_supported("Hugepages not available on this platform"));
            }
        };

        self.position = end_pos;
        Ok(data)
    }

    /// Reads a slice of bytes without copying (zero-copy, memory-mapped only)
    pub fn read_slice_zero_copy(&mut self, len: usize) -> Result<&[u8]> {
        let end_pos = self.position + len;
        if end_pos > self.file_size as usize {
            return Err(ZiporaError::out_of_bounds(end_pos, self.file_size as usize));
        }

        let slice = match self.strategy {
            InputStrategy::BufferedIO => {
                return Err(ZiporaError::not_supported(
                    "Zero-copy not available for buffered I/O strategy"
                ));
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    &mmap[self.position..end_pos]
                } else {
                    return Err(ZiporaError::invalid_data("Memory map not initialized"));
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    &hugepage.as_slice()[self.position..end_pos]
                } else {
                    return Err(ZiporaError::invalid_data("Hugepage not initialized"));
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                return Err(ZiporaError::not_supported("Hugepages not available on this platform"));
            }
        };

        self.position = end_pos;
        Ok(slice)
    }

    /// Peeks at bytes without advancing the position (zero-copy when possible)
    pub fn peek_slice(&self, len: usize) -> Result<Vec<u8>> {
        let end_pos = self.position + len;
        if end_pos > self.file_size as usize {
            return Err(ZiporaError::out_of_bounds(end_pos, self.file_size as usize));
        }

        match self.strategy {
            InputStrategy::BufferedIO => {
                // For buffered I/O, peek is not efficiently supported
                Err(ZiporaError::not_supported(
                    "Peek not efficiently supported for buffered I/O strategy"
                ))
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    Ok(mmap[self.position..end_pos].to_vec())
                } else {
                    Err(ZiporaError::invalid_data("Memory map not initialized"))
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    Ok(hugepage.as_slice()[self.position..end_pos].to_vec())
                } else {
                    Err(ZiporaError::invalid_data("Hugepage not initialized"))
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                Err(ZiporaError::not_supported("Hugepages not available on this platform"))
            }
        }
    }

    /// Peeks at bytes without advancing the position (zero-copy, memory-mapped only)
    pub fn peek_slice_zero_copy(&self, len: usize) -> Result<&[u8]> {
        let end_pos = self.position + len;
        if end_pos > self.file_size as usize {
            return Err(ZiporaError::out_of_bounds(end_pos, self.file_size as usize));
        }

        match self.strategy {
            InputStrategy::BufferedIO => {
                Err(ZiporaError::not_supported(
                    "Zero-copy peek not available for buffered I/O strategy"
                ))
            }

            InputStrategy::StandardMmap => {
                if let Some(ref mmap) = self.mmap {
                    Ok(&mmap[self.position..end_pos])
                } else {
                    Err(ZiporaError::invalid_data("Memory map not initialized"))
                }
            }

            #[cfg(target_os = "linux")]
            InputStrategy::HugepageMmap => {
                if let Some(ref hugepage) = self.hugepage {
                    Ok(&hugepage.as_slice()[self.position..end_pos])
                } else {
                    Err(ZiporaError::invalid_data("Hugepage not initialized"))
                }
            }

            #[cfg(not(target_os = "linux"))]
            InputStrategy::HugepageMmap => {
                Err(ZiporaError::not_supported("Hugepages not available on this platform"))
            }
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
        let len = self.read_var_int()? as usize;
        let data = self.read_slice(len)?;
        String::from_utf8(data)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid UTF-8 string: {}", e)))
    }

    fn skip(&mut self, n: usize) -> Result<()> {
        let new_pos = self.position + n;
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
            .map_err(|e| ZiporaError::io_error(format!("Failed to create file: {}", e)))?;

        // Set the file size
        file.set_len(initial_size as u64)
            .map_err(|e| ZiporaError::io_error(format!("Failed to set file size: {}", e)))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| ZiporaError::io_error(format!("Failed to memory-map file: {}", e)))?
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
            .map_err(|e| ZiporaError::io_error(format!("Failed to open file: {}", e)))?;

        let file_size = file
            .metadata()
            .map_err(|e| ZiporaError::io_error(format!("Failed to get file metadata: {}", e)))?
            .len() as usize;

        let mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| ZiporaError::io_error(format!("Failed to memory-map file: {}", e)))?
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
            return Err(ZiporaError::out_of_bounds(pos, self.capacity));
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
            .map_err(|e| ZiporaError::io_error(format!("Failed to grow file: {}", e)))?;

        self.mmap = unsafe {
            MmapOptions::new()
                .map_mut(&self.file)
                .map_err(|e| ZiporaError::io_error(format!("Failed to remap file: {}", e)))?
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
            .map_err(|e| ZiporaError::io_error(format!("Failed to flush mmap: {}", e)))?;

        // Unmap before truncating
        drop(std::mem::replace(
            &mut self.mmap,
            MmapMut::map_anon(0).unwrap(),
        ));

        // Truncate the file
        self.file
            .set_len(self.position as u64)
            .map_err(|e| ZiporaError::io_error(format!("Failed to truncate file: {}", e)))?;

        // Remap with new size
        self.mmap = unsafe {
            MmapOptions::new().map_mut(&self.file).map_err(|e| {
                ZiporaError::io_error(format!("Failed to remap truncated file: {}", e))
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
            ZiporaError::io_error(format!("Failed to flush memory-mapped file: {}", e))
        })
    }
}

// Provide stub implementations when mmap feature is disabled
#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedInput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedInput {
    pub fn new(_file: File) -> Result<Self> {
        Err(ZiporaError::invalid_data(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }

    pub fn from_path<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(ZiporaError::invalid_data(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedInput.",
        ))
    }
}

#[cfg(not(feature = "mmap"))]
pub struct MemoryMappedOutput;

#[cfg(not(feature = "mmap"))]
impl MemoryMappedOutput {
    pub fn create<P: AsRef<Path>>(_path: P, _initial_size: usize) -> Result<Self> {
        Err(ZiporaError::invalid_data(
            "Memory mapping is not available. Enable the 'mmap' feature to use MemoryMappedOutput.",
        ))
    }

    pub fn open<P: AsRef<Path>>(_path: P) -> Result<Self> {
        Err(ZiporaError::invalid_data(
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
        assert_eq!(read_data, large_data);
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
        temp_file.write_all(&0x9ABCDEF012345678u64.to_le_bytes()).unwrap();
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
    fn test_file_size_thresholds() {
        // Test exactly at threshold boundaries
        let threshold_tests = vec![
            (SMALL_FILE_THRESHOLD as usize - 1, InputStrategy::BufferedIO),      // Just below 4KB
            (SMALL_FILE_THRESHOLD as usize, InputStrategy::BufferedIO),          // Exactly 4KB  
            (SMALL_FILE_THRESHOLD as usize + 1, InputStrategy::StandardMmap),    // Just above 4KB
            (HUGEPAGE_2MB_THRESHOLD as usize - 1, InputStrategy::StandardMmap),  // Just below 1MB
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
                size, expected_strategy, input.strategy()
            );
        }
    }

    #[cfg(all(feature = "mmap", target_os = "linux"))]
    #[test]
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
}
