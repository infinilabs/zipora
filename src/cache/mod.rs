//! # LRU Page Cache
//!
//! High-performance caching layer for blob operations with sophisticated eviction policies,
//! page-aligned memory management, and lock-free optimizations.
//!
//! ## Features
//!
//! - **LRU Page Cache**: Sophisticated caching layer for blob operations
//! - **Page-Aligned Memory**: 4KB/2MB page-aligned allocations for optimal performance
//! - **Multi-Shard Architecture**: Configurable sharding for reduced contention
//! - **Lock-Free Operations**: Reference counting and atomic operations for high throughput
//! - **Hardware Prefetching**: SIMD prefetch hints for cache optimization
//! - **Huge Page Support**: 2MB huge pages for large cache allocations
//! - **Thread-Safe**: Configurable locking granularity with futex support
//! - **Performance Monitoring**: Comprehensive statistics and conflict tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    LRU Page Cache                           │
//! ├─────────────────────────────────────────────────────────────┤
//! │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
//! │ │   Shard 0   │ │   Shard 1   │ │   Shard N   │  ...      │
//! │ │             │ │             │ │             │           │
//! │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │           │
//! │ │ │Hash Tbl │ │ │ │Hash Tbl │ │ │ │Hash Tbl │ │           │
//! │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │           │
//! │ │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │           │
//! │ │ │LRU List │ │ │ │LRU List │ │ │ │LRU List │ │           │
//! │ │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │           │
//! │ └─────────────┘ └─────────────┘ └─────────────┘           │
//! ├─────────────────────────────────────────────────────────────┤
//! │                   Page Memory Pool                         │
//! │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐  │
//! │  │ 4KB Page  │ │ 4KB Page  │ │ 4KB Page  │ │    ...    │  │
//! │  └───────────┘ └───────────┘ └───────────┘ └───────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```rust
//! use zipora::cache::{LruPageCache, PageCacheConfig, CacheBuffer};
//! # use zipora::error::Result;
//! # fn example() -> Result<()> {
//!
//! // Create high-performance page cache
//! let config = PageCacheConfig::performance_optimized()
//!     .with_capacity(256 * 1024 * 1024) // 256MB cache
//!     .with_shards(8)                   // 8 shards for reduced contention
//!     .with_huge_pages(true);           // Use 2MB huge pages
//!
//! let cache = LruPageCache::new(config)?;
//!
//! // Register files for caching
//! let file_id = cache.register_file(1)?;
//!
//! // Cache-aware blob reading
//! let data = cache.read(file_id, 0, 1024)?;
//!
//! // Batch operations for improved performance
//! let requests = vec![(file_id, 0, 1024), (file_id, 1024, 1024)];
//! let results = cache.read_batch(requests)?;
//! # Ok(())
//! # }
//! ```

pub mod basic_cache;
pub mod buffer;
pub mod config;
pub mod stats;

#[cfg(test)]
mod simple_tests;

pub use basic_cache::{LruPageCache, SingleLruPageCache};
pub use buffer::*;
pub use config::*;
pub use stats::*;

use crate::error::{Result, ZiporaError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Standard page size for cache operations (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Page bits for efficient bit shifting
pub const PAGE_BITS: usize = 12;

/// Huge page size for large allocations (2MB)
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Maximum number of shards supported
pub const MAX_SHARDS: usize = 64;

/// Cache line size for alignment optimizations
pub const CACHE_CACHE_LINE_SIZE: usize = 64;

/// Page cache error types
#[derive(Debug, Clone, PartialEq)]
pub enum CacheError {
    /// Cache is full and cannot accommodate new entries
    CacheFull,
    /// Invalid page size or alignment
    InvalidPageSize,
    /// File not found in cache
    FileNotFound,
    /// Invalid shard configuration
    InvalidShardConfig,
    /// Memory allocation failed
    AllocationFailed,
    /// Hardware feature not available
    HardwareUnsupported,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::CacheFull => write!(f, "Cache is full"),
            CacheError::InvalidPageSize => write!(f, "Invalid page size or alignment"),
            CacheError::FileNotFound => write!(f, "File not found in cache"),
            CacheError::InvalidShardConfig => write!(f, "Invalid shard configuration"),
            CacheError::AllocationFailed => write!(f, "Memory allocation failed"),
            CacheError::HardwareUnsupported => write!(f, "Required hardware feature not available"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<CacheError> for ZiporaError {
    fn from(err: CacheError) -> Self {
        ZiporaError::invalid_data(err.to_string())
    }
}

/// Cache hit type for performance monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheHitType {
    /// Direct cache hit
    Hit = 0,
    /// Evicted other pages
    EvictedOthers = 1,
    /// Used initial free page
    InitialFree = 2,
    /// Reused dropped page
    DroppedFree = 3,
    /// Hit while another thread was loading
    HitOthersLoad = 4,
    /// Mixed operation (multi-page)
    Mix = 5,
    /// Cache miss
    Miss = 6,
}

impl CacheHitType {
    /// Convert to index for statistics array
    pub fn as_index(self) -> usize {
        self as usize
    }

    /// Get human-readable description
    pub fn description(self) -> &'static str {
        match self {
            CacheHitType::Hit => "Cache Hit",
            CacheHitType::EvictedOthers => "Evicted Others",
            CacheHitType::InitialFree => "Initial Free",
            CacheHitType::DroppedFree => "Dropped Free",
            CacheHitType::HitOthersLoad => "Hit Others Load",
            CacheHitType::Mix => "Mixed Operation",
            CacheHitType::Miss => "Cache Miss",
        }
    }
}

/// File ID type for cache operations
pub type FileId = u32;

/// Page ID type for efficient packing
pub type PageId = u32;

/// Cache node index type
pub type NodeIndex = u32;

/// Invalid node index constant
pub const INVALID_NODE: NodeIndex = u32::MAX;

/// Hash function for file-page combinations
#[inline]
pub fn hash_file_page(file_id: FileId, page_id: PageId) -> u64 {
    let fi_page_id = ((file_id as u64) << 32) | (page_id as u64);
    // Bit rotation for better distribution
    let hash1 = fi_page_id.rotate_left(3);
    // Endian-aware mixing
    hash1.swap_bytes()
}

/// Get shard ID for given file-page combination
#[inline]
pub fn get_shard_id(file_id: FileId, page_id: PageId, num_shards: u32) -> u32 {
    if num_shards <= 1 {
        return 0;
    }
    let hash = hash_file_page(file_id, page_id);
    (hash % (num_shards as u64)) as u32
}

/// Hardware prefetch hint for cache optimization
#[inline]
pub fn prefetch_hint(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: _mm_prefetch is a CPU hint instruction that never fails or dereferences ptr,
        // simply signals the CPU to load the cache line containing ptr into L1 cache
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        // SAFETY: `prfm pldl1keep` is a non-faulting L1 cache prefetch hint on aarch64.
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", all(target_arch = "aarch64", not(miri)))))]
    {
        let _ = ptr;
    }
}

/// Backing storage for a managed cache file
#[derive(Debug)]
enum FileBacking {
    /// OS file handle
    Real(File),
    /// Virtual in-memory backing buffer (registered with `fd == -1`)
    Virtual(Vec<u8>),
}

/// File handle manager for efficient file operations
#[derive(Debug)]
pub struct FileManager {
    /// Map of file IDs to file handles and metadata
    files: RwLock<HashMap<FileId, FileEntry>>,
    /// Next available file ID
    next_file_id: AtomicU32,
}

/// File entry containing file handle and metadata
#[derive(Debug)]
struct FileEntry {
    /// Backing storage for I/O operations
    backing: FileBacking,
    /// File path for error reporting
    path: PathBuf,
    /// File size in bytes
    size: u64,
    /// Whether the file handle was opened with write permission
    writable: bool,
}

impl Default for FileManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FileManager {
    /// Create a new file manager
    pub fn new() -> Self {
        Self {
            files: RwLock::new(HashMap::new()),
            next_file_id: AtomicU32::new(1), // Start from 1, reserve 0 for invalid
        }
    }

    /// Open a file and register it with the manager
    pub fn open_file<P: AsRef<Path>>(&self, path: P) -> Result<FileId> {
        let path = path.as_ref().to_path_buf();
        let (file, writable) = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        {
            Ok(file) => (file, true),
            Err(_) => {
                let file = File::open(&path).map_err(|e| {
                    ZiporaError::invalid_data(format!("Failed to open file {:?}: {}", path, e))
                })?;
                (file, false)
            }
        };

        let size = file
            .metadata()
            .map_err(|e| {
                ZiporaError::invalid_data(format!("Failed to get file metadata {:?}: {}", path, e))
            })?
            .len();

        // ORDERING: Relaxed is sufficient for unique monotonic FileId allocation.
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);

        let entry = FileEntry {
            backing: FileBacking::Real(file),
            path,
            size,
            writable,
        };

        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        files.insert(file_id, entry);

        Ok(file_id)
    }

    /// Register an in-memory virtual file (for `fd == -1`)
    pub fn register_virtual_file(&self) -> Result<FileId> {
        // ORDERING: Relaxed is sufficient for unique monotonic FileId allocation.
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        let entry = FileEntry {
            backing: FileBacking::Virtual(Vec::new()),
            path: PathBuf::from(format!("<virtual:{}>", file_id)),
            size: 0,
            writable: true,
        };
        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        files.insert(file_id, entry);
        Ok(file_id)
    }

    /// Ensure that a registered file is open with write permission
    pub fn ensure_writable(&self, file_id: FileId) -> Result<()> {
        let files = self
            .files
            .read()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        let entry = files
            .get(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        if !entry.writable {
            return Err(ZiporaError::invalid_data(format!(
                "File {:?} (ID {}) is opened read-only",
                entry.path, file_id
            )));
        }

        Ok(())
    }

    /// Get file size
    pub fn file_size(&self, file_id: FileId) -> Result<u64> {
        let files = self
            .files
            .read()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        files
            .get(&file_id)
            .map(|entry| entry.size)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))
    }

    /// Calculate page ID from offset
    pub fn offset_to_page_id(offset: u64) -> PageId {
        (offset / PAGE_SIZE as u64) as PageId
    }

    /// Calculate offset within page
    pub fn offset_within_page(offset: u64) -> usize {
        (offset % PAGE_SIZE as u64) as usize
    }

    /// Calculate page-aligned offset
    pub fn page_aligned_offset(page_id: PageId) -> u64 {
        (page_id as u64) * (PAGE_SIZE as u64)
    }

    /// Read a page from file with proper error handling
    pub fn read_page(&self, file_id: FileId, page_id: PageId, buffer: &mut [u8]) -> Result<usize> {
        if buffer.len() != PAGE_SIZE {
            return Err(ZiporaError::invalid_data(format!(
                "Buffer size {} != PAGE_SIZE {}",
                buffer.len(),
                PAGE_SIZE
            )));
        }

        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        let entry = files
            .get_mut(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        let offset = Self::page_aligned_offset(page_id);

        // Check if offset is within file bounds
        if offset >= entry.size {
            buffer.fill(0);
            return Ok(0); // End of file
        }

        let bytes_to_read = std::cmp::min(PAGE_SIZE, (entry.size - offset) as usize);
        let bytes_read = match &mut entry.backing {
            FileBacking::Real(file) => {
                file.seek(SeekFrom::Start(offset)).map_err(|e| {
                    ZiporaError::invalid_data(format!(
                        "Failed to seek to offset {} in file {:?}: {}",
                        offset, entry.path, e
                    ))
                })?;

                let mut read_so_far = 0;
                while read_so_far < bytes_to_read {
                    match file.read(&mut buffer[read_so_far..bytes_to_read]) {
                        Ok(0) => break,
                        Ok(n) => read_so_far += n,
                        Err(e) => {
                            return Err(ZiporaError::invalid_data(format!(
                                "Failed to read from file {:?} at offset {}: {}",
                                entry.path, offset, e
                            )));
                        }
                    }
                }
                read_so_far
            }
            FileBacking::Virtual(data) => {
                let start = offset as usize;
                let end = (start + bytes_to_read).min(data.len());
                let count = end.saturating_sub(start);
                if count > 0 {
                    buffer[..count].copy_from_slice(&data[start..end]);
                }
                count
            }
        };

        // Zero-fill remaining buffer if we read less than a full page
        if bytes_read < PAGE_SIZE {
            buffer[bytes_read..].fill(0);
        }

        Ok(bytes_read)
    }

    /// Write a page back to the underlying file or virtual buffer
    pub fn write_page(&self, file_id: FileId, page_id: PageId, data: &[u8]) -> Result<()> {
        use std::io::Write;

        if data.len() > PAGE_SIZE {
            return Err(ZiporaError::invalid_data(format!(
                "Page data length {} exceeds PAGE_SIZE {}",
                data.len(),
                PAGE_SIZE
            )));
        }

        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        let entry = files
            .get_mut(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        if !entry.writable {
            return Err(ZiporaError::invalid_data(format!(
                "File {:?} (ID {}) is opened read-only",
                entry.path, file_id
            )));
        }

        let offset = Self::page_aligned_offset(page_id);
        let end_offset = offset + data.len() as u64;

        match &mut entry.backing {
            FileBacking::Real(file) => {
                file.seek(SeekFrom::Start(offset)).map_err(|e| {
                    ZiporaError::invalid_data(format!(
                        "Failed to seek to offset {} in file {:?}: {}",
                        offset, entry.path, e
                    ))
                })?;
                file.write_all(data).map_err(|e| {
                    ZiporaError::invalid_data(format!(
                        "Failed to write to file {:?} at offset {}: {}",
                        entry.path, offset, e
                    ))
                })?;
                file.flush().map_err(|e| {
                    ZiporaError::invalid_data(format!(
                        "Failed to flush file {:?}: {}",
                        entry.path, e
                    ))
                })?;
            }
            FileBacking::Virtual(buf) => {
                let end = end_offset as usize;
                if buf.len() < end {
                    buf.resize(end, 0);
                }
                let start = offset as usize;
                buf[start..end].copy_from_slice(data);
            }
        }

        if end_offset > entry.size {
            entry.size = end_offset;
        }

        Ok(())
    }

    /// Read partial data that may span multiple pages
    pub fn read_data(
        &self,
        file_id: FileId,
        offset: u64,
        length: usize,
        buffer: &mut [u8],
    ) -> Result<usize> {
        if buffer.len() < length {
            return Err(ZiporaError::invalid_data(format!(
                "Buffer size {} < requested length {}",
                buffer.len(),
                length
            )));
        }

        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        let entry = files
            .get_mut(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        // Check bounds
        if offset >= entry.size {
            return Ok(0);
        }

        let bytes_to_read = std::cmp::min(length, (entry.size - offset) as usize);

        match &mut entry.backing {
            FileBacking::Real(file) => {
                file.seek(SeekFrom::Start(offset)).map_err(|e| {
                    ZiporaError::invalid_data(format!(
                        "Failed to seek to offset {} in file {:?}: {}",
                        offset, entry.path, e
                    ))
                })?;

                let mut bytes_read = 0;
                while bytes_read < bytes_to_read {
                    match file.read(&mut buffer[bytes_read..bytes_to_read]) {
                        Ok(0) => break,
                        Ok(n) => bytes_read += n,
                        Err(e) => {
                            return Err(ZiporaError::invalid_data(format!(
                                "Failed to read from file {:?} at offset {}: {}",
                                entry.path, offset, e
                            )));
                        }
                    }
                }

                Ok(bytes_read)
            }
            FileBacking::Virtual(data) => {
                let start = offset as usize;
                let end = (start + bytes_to_read).min(data.len());
                let count = end.saturating_sub(start);
                if count > 0 {
                    buffer[..count].copy_from_slice(&data[start..end]);
                }
                Ok(count)
            }
        }
    }

    /// Close a file and remove it from management
    pub fn close_file(&self, file_id: FileId) -> Result<()> {
        let mut files = self
            .files
            .write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;

        files
            .remove(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        Ok(())
    }
}
