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
//!
//! // Create high-performance page cache
//! let config = PageCacheConfig::performance_optimized()
//!     .with_capacity(256 * 1024 * 1024) // 256MB cache
//!     .with_shards(8)                   // 8 shards for reduced contention
//!     .with_huge_pages(true);           // Use 2MB huge pages
//!
//! let cache = LruPageCache::new(config)?;
//!
//! // Cache-aware blob reading
//! let mut buffer = CacheBuffer::new();
//! let data = cache.read(file_id, offset, length, &mut buffer)?;
//!
//! // Batch operations for improved performance
//! let requests = vec![(file_id1, offset1, len1), (file_id2, offset2, len2)];
//! let results = cache.read_batch(requests)?;
//! ```

pub mod config;
pub mod stats;
pub mod buffer;
pub mod basic_cache;

#[cfg(test)]
mod simple_tests;

pub use config::*;
pub use stats::*;
pub use buffer::*;
pub use basic_cache::{LruPageCache, SingleLruPageCache};

use crate::error::{Result, ZiporaError};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

/// Standard page size for cache operations (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Page bits for efficient bit shifting
pub const PAGE_BITS: usize = 12;

/// Huge page size for large allocations (2MB)
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Maximum number of shards supported
pub const MAX_SHARDS: usize = 64;

/// Cache line size for alignment optimizations
pub const CACHE_LINE_SIZE: usize = 64;

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
    let hash1 = (fi_page_id << 3) | (fi_page_id >> 61);
    // Endian-aware mixing
    hash1.swap_bytes()
}

/// Get shard ID for given file-page combination
#[inline]
pub fn get_shard_id(file_id: FileId, page_id: PageId, num_shards: u32) -> u32 {
    let hash = hash_file_page(file_id, page_id);
    (hash % (num_shards as u64)) as u32
}

/// Hardware prefetch hint for cache optimization
#[inline]
pub fn prefetch_hint(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            std::arch::aarch64::__pldl1keep(ptr);
        }
    }
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
    /// File handle for I/O operations
    file: File,
    /// File path for error reporting
    path: PathBuf,
    /// File size in bytes
    size: u64,
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
        let file = File::open(&path)
            .map_err(|e| ZiporaError::invalid_data(format!("Failed to open file {:?}: {}", path, e)))?;
        
        let size = file.metadata()
            .map_err(|e| ZiporaError::invalid_data(format!("Failed to get file metadata {:?}: {}", path, e)))?
            .len();

        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        
        let entry = FileEntry { file, path, size };
        
        let mut files = self.files.write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        files.insert(file_id, entry);
        
        Ok(file_id)
    }

    /// Get file size
    pub fn file_size(&self, file_id: FileId) -> Result<u64> {
        let files = self.files.read()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        
        files.get(&file_id)
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
                "Buffer size {} != PAGE_SIZE {}", buffer.len(), PAGE_SIZE
            )));
        }

        let mut files = self.files.write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        
        let entry = files.get_mut(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        let offset = Self::page_aligned_offset(page_id);
        
        // Check if offset is within file bounds
        if offset >= entry.size {
            return Ok(0); // End of file
        }

        // Seek to the desired position
        entry.file.seek(SeekFrom::Start(offset))
            .map_err(|e| ZiporaError::invalid_data(format!(
                "Failed to seek to offset {} in file {:?}: {}", offset, entry.path, e
            )))?;

        // Read the page data
        let bytes_to_read = std::cmp::min(PAGE_SIZE, (entry.size - offset) as usize);
        let mut bytes_read = 0;
        
        while bytes_read < bytes_to_read {
            match entry.file.read(&mut buffer[bytes_read..bytes_to_read]) {
                Ok(0) => break, // EOF reached
                Ok(n) => bytes_read += n,
                Err(e) => return Err(ZiporaError::invalid_data(format!(
                    "Failed to read from file {:?} at offset {}: {}", entry.path, offset, e
                ))),
            }
        }

        // Zero-fill remaining buffer if we read less than a full page
        if bytes_read < PAGE_SIZE {
            buffer[bytes_read..].fill(0);
        }

        Ok(bytes_read)
    }

    /// Read partial data that may span multiple pages
    pub fn read_data(&self, file_id: FileId, offset: u64, length: usize, buffer: &mut [u8]) -> Result<usize> {
        if buffer.len() < length {
            return Err(ZiporaError::invalid_data(format!(
                "Buffer size {} < requested length {}", buffer.len(), length
            )));
        }

        let files = self.files.read()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        
        let entry = files.get(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        // Check bounds
        if offset >= entry.size {
            return Ok(0);
        }

        let bytes_to_read = std::cmp::min(length, (entry.size - offset) as usize);
        
        drop(files); // Release read lock before potentially long I/O operation
        
        // Re-acquire write lock for file operations
        let mut files = self.files.write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        
        let entry = files.get_mut(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;

        // Seek and read
        entry.file.seek(SeekFrom::Start(offset))
            .map_err(|e| ZiporaError::invalid_data(format!(
                "Failed to seek to offset {} in file {:?}: {}", offset, entry.path, e
            )))?;

        let mut bytes_read = 0;
        while bytes_read < bytes_to_read {
            match entry.file.read(&mut buffer[bytes_read..bytes_to_read]) {
                Ok(0) => break,
                Ok(n) => bytes_read += n,
                Err(e) => return Err(ZiporaError::invalid_data(format!(
                    "Failed to read from file {:?} at offset {}: {}", entry.path, offset, e
                ))),
            }
        }

        Ok(bytes_read)
    }

    /// Close a file and remove it from management
    pub fn close_file(&self, file_id: FileId) -> Result<()> {
        let mut files = self.files.write()
            .map_err(|_| ZiporaError::invalid_data("FileManager lock poisoned".to_string()))?;
        
        files.remove(&file_id)
            .ok_or_else(|| ZiporaError::invalid_data(format!("File ID {} not found", file_id)))?;
        
        Ok(())
    }
}