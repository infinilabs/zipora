//! C-compatible type definitions
//!
//! This module defines opaque types and structures that can be safely
//! passed across the C FFI boundary.

use std::os::raw::{c_char, c_int};

/// Opaque handle for FastVec
#[repr(C)]
pub struct CFastVec {
    _private: [u8; 0],
}

/// Opaque handle for MemoryPool
#[repr(C)]
pub struct CMemoryPool {
    _private: [u8; 0],
}

/// Opaque handle for BlobStore
#[repr(C)]
pub struct CBlobStore {
    _private: [u8; 0],
}

/// Opaque handle for SuffixArray
#[repr(C)]
pub struct CSuffixArray {
    _private: [u8; 0],
}

/// Opaque handle for RadixSort
#[repr(C)]
pub struct CRadixSort {
    _private: [u8; 0],
}

/// Opaque handle for MultiWayMerge
#[repr(C)]
pub struct CMultiWayMerge {
    _private: [u8; 0],
}

/// Configuration structure for memory pools
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CMemoryPoolConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in pool
    pub max_chunks: usize,
    /// Alignment requirement for allocations
    pub alignment: usize,
}

impl Default for CMemoryPoolConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024,
            max_chunks: 100,
            alignment: 8,
        }
    }
}

impl From<CMemoryPoolConfig> for crate::memory::PoolConfig {
    fn from(config: CMemoryPoolConfig) -> Self {
        crate::memory::PoolConfig::new(config.chunk_size, config.max_chunks, config.alignment)
    }
}

/// Statistics structure for memory pools
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CMemoryPoolStats {
    /// Total bytes allocated
    pub allocated: u64,
    /// Total bytes available in pool
    pub available: u64,
    /// Number of chunks in pool
    pub chunks: usize,
    /// Number of allocations served
    pub alloc_count: u64,
    /// Number of deallocations
    pub dealloc_count: u64,
    /// Number of pool hits (reused memory)
    pub pool_hits: u64,
    /// Number of pool misses (new allocations)
    pub pool_misses: u64,
}

impl From<crate::memory::pool::PoolStats> for CMemoryPoolStats {
    fn from(stats: crate::memory::pool::PoolStats) -> Self {
        Self {
            allocated: stats.allocated,
            available: stats.available,
            chunks: stats.chunks,
            alloc_count: stats.alloc_count,
            dealloc_count: stats.dealloc_count,
            pool_hits: stats.pool_hits,
            pool_misses: stats.pool_misses,
        }
    }
}

/// Configuration structure for radix sort
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CRadixSortConfig {
    /// Use parallel processing for large datasets
    pub use_parallel: c_int,
    /// Threshold for switching to parallel processing
    pub parallel_threshold: usize,
    /// Radix size (typically 8 or 16 bits)
    pub radix_bits: usize,
    /// Use counting sort for small datasets
    pub use_counting_sort_threshold: usize,
    /// Enable SIMD optimizations when available
    pub use_simd: c_int,
}

impl Default for CRadixSortConfig {
    fn default() -> Self {
        let rust_config = crate::algorithms::RadixSortConfig::default();
        Self {
            use_parallel: if rust_config.use_parallel { 1 } else { 0 },
            parallel_threshold: rust_config.parallel_threshold,
            radix_bits: rust_config.radix_bits,
            use_counting_sort_threshold: rust_config.use_counting_sort_threshold,
            use_simd: if rust_config.use_simd { 1 } else { 0 },
        }
    }
}

impl From<CRadixSortConfig> for crate::algorithms::RadixSortConfig {
    fn from(config: CRadixSortConfig) -> Self {
        Self {
            use_parallel: config.use_parallel != 0,
            parallel_threshold: config.parallel_threshold,
            radix_bits: config.radix_bits,
            use_counting_sort_threshold: config.use_counting_sort_threshold,
            use_simd: config.use_simd != 0,
        }
    }
}

/// Statistics structure for algorithms
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CAlgorithmStats {
    /// Total items processed
    pub items_processed: usize,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Memory used in bytes
    pub memory_used: usize,
    /// Whether parallel processing was used
    pub used_parallel: c_int,
    /// Whether SIMD optimizations were used
    pub used_simd: c_int,
}

impl From<crate::algorithms::AlgorithmStats> for CAlgorithmStats {
    fn from(stats: crate::algorithms::AlgorithmStats) -> Self {
        Self {
            items_processed: stats.items_processed,
            processing_time_us: stats.processing_time_us,
            memory_used: stats.memory_used,
            used_parallel: if stats.used_parallel { 1 } else { 0 },
            used_simd: if stats.used_simd { 1 } else { 0 },
        }
    }
}

/// Blob store statistics
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CBlobStoreStats {
    /// Number of records stored
    pub record_count: usize,
    /// Total bytes stored
    pub total_bytes: u64,
    /// Number of get operations
    pub get_count: u64,
    /// Number of put operations
    pub put_count: u64,
}

impl From<crate::blob_store::BlobStoreStats> for CBlobStoreStats {
    fn from(stats: crate::blob_store::BlobStoreStats) -> Self {
        Self {
            record_count: stats.record_count,
            total_bytes: stats.total_bytes,
            get_count: stats.get_count,
            put_count: stats.put_count,
        }
    }
}

/// Memory configuration for the library
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CMemoryConfig {
    /// Enable memory pools for frequent allocations
    pub use_pools: c_int,
    /// Enable hugepage allocation when available
    pub use_hugepages: c_int,
    /// Default pool chunk size in bytes
    pub pool_chunk_size: usize,
    /// Maximum memory to keep in pools
    pub max_pool_memory: usize,
}

impl Default for CMemoryConfig {
    fn default() -> Self {
        let rust_config = crate::memory::MemoryConfig::default();
        Self {
            use_pools: if rust_config.use_pools { 1 } else { 0 },
            use_hugepages: if rust_config.use_hugepages { 1 } else { 0 },
            pool_chunk_size: rust_config.pool_chunk_size,
            max_pool_memory: rust_config.max_pool_memory,
        }
    }
}

impl From<CMemoryConfig> for crate::memory::MemoryConfig {
    fn from(config: CMemoryConfig) -> Self {
        Self {
            use_pools: config.use_pools != 0,
            use_hugepages: config.use_hugepages != 0,
            pool_chunk_size: config.pool_chunk_size,
            max_pool_memory: config.max_pool_memory,
        }
    }
}

/// Global memory statistics
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CMemoryStats {
    /// Total bytes allocated through pools
    pub pool_allocated: u64,
    /// Total bytes available in pools
    pub pool_available: u64,
    /// Number of active pool chunks
    pub pool_chunks: usize,
    /// Number of hugepages allocated
    pub hugepages_allocated: usize,
}

impl From<crate::memory::MemoryStats> for CMemoryStats {
    fn from(stats: crate::memory::MemoryStats) -> Self {
        Self {
            pool_allocated: stats.pool_allocated,
            pool_available: stats.pool_available,
            pool_chunks: stats.pool_chunks,
            hugepages_allocated: stats.hugepages_allocated,
        }
    }
}

/// Buffer structure for passing data to/from C
#[repr(C)]
#[derive(Debug)]
pub struct CBuffer {
    /// Pointer to data
    pub data: *mut u8,
    /// Size of data in bytes
    pub size: usize,
    /// Capacity of allocated buffer
    pub capacity: usize,
}

impl CBuffer {
    /// Create a new empty buffer
    pub fn new() -> Self {
        Self {
            data: std::ptr::null_mut(),
            size: 0,
            capacity: 0,
        }
    }

    /// Create a buffer from a Vec<u8>
    pub fn from_vec(mut vec: Vec<u8>) -> Self {
        let data = vec.as_mut_ptr();
        let size = vec.len();
        let capacity = vec.capacity();
        std::mem::forget(vec); // Don't drop the Vec

        Self {
            data,
            size,
            capacity,
        }
    }

    /// Convert back to a Vec<u8>
    ///
    /// # Safety
    ///
    /// The buffer must have been created from a Vec<u8> and not modified
    /// in an incompatible way.
    pub unsafe fn into_vec(self) -> Vec<u8> {
        if self.data.is_null() {
            return Vec::new();
        }

        unsafe { Vec::from_raw_parts(self.data, self.size, self.capacity) }
    }

    /// Get the data as a slice
    ///
    /// # Safety
    ///
    /// The buffer must contain valid data of the specified size.
    pub unsafe fn as_slice(&self) -> &[u8] {
        if self.data.is_null() || self.size == 0 {
            return &[];
        }

        unsafe { std::slice::from_raw_parts(self.data, self.size) }
    }

    /// Get the data as a mutable slice
    ///
    /// # Safety
    ///
    /// The buffer must contain valid data of the specified size.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.data.is_null() || self.size == 0 {
            return &mut [];
        }

        unsafe { std::slice::from_raw_parts_mut(self.data, self.size) }
    }
}

impl Default for CBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// String structure for C FFI
#[repr(C)]
#[derive(Debug)]
pub struct CString {
    /// Pointer to null-terminated string
    pub data: *mut c_char,
    /// Length of string (not including null terminator)
    pub len: usize,
}

impl CString {
    /// Create a new empty string
    pub fn new() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
        }
    }

    /// Create from a Rust string
    pub fn from_string(s: String) -> Result<Self, std::ffi::NulError> {
        let cstring = std::ffi::CString::new(s)?;
        let len = cstring.as_bytes().len();
        let data = cstring.into_raw();

        Ok(Self { data, len })
    }

    /// Convert back to a Rust string
    ///
    /// # Safety
    ///
    /// The data pointer must point to a valid null-terminated string.
    pub unsafe fn into_string(self) -> Result<String, std::str::Utf8Error> {
        if self.data.is_null() {
            return Ok(String::new());
        }

        let cstring = unsafe { std::ffi::CString::from_raw(self.data) };
        cstring.to_str().map(|s| s.to_owned())
    }

    /// Get as a string slice
    ///
    /// # Safety
    ///
    /// The data pointer must point to a valid null-terminated string.
    pub unsafe fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        if self.data.is_null() {
            return Ok("");
        }

        let cstr = unsafe { std::ffi::CStr::from_ptr(self.data) };
        cstr.to_str()
    }
}

impl Default for CString {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CString {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                let _ = std::ffi::CString::from_raw(self.data);
                // Automatic cleanup when CString is dropped
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_config_conversion() {
        let c_config = CMemoryPoolConfig {
            chunk_size: 1024,
            max_chunks: 100,
            alignment: 16,
        };

        let rust_config: crate::memory::PoolConfig = c_config.into();
        assert_eq!(rust_config.chunk_size, 1024);
        assert_eq!(rust_config.max_chunks, 100);
        assert_eq!(rust_config.alignment, 16);
    }

    #[test]
    fn test_radix_sort_config_conversion() {
        let c_config = CRadixSortConfig {
            use_parallel: 1,
            parallel_threshold: 10000,
            radix_bits: 8,
            use_counting_sort_threshold: 256,
            use_simd: 0,
        };

        let rust_config: crate::algorithms::RadixSortConfig = c_config.into();
        assert!(rust_config.use_parallel);
        assert_eq!(rust_config.parallel_threshold, 10000);
        assert_eq!(rust_config.radix_bits, 8);
        assert_eq!(rust_config.use_counting_sort_threshold, 256);
        assert!(!rust_config.use_simd);
    }

    #[test]
    fn test_buffer_operations() {
        let vec = vec![1u8, 2, 3, 4, 5];
        let buffer = CBuffer::from_vec(vec);

        assert!(!buffer.data.is_null());
        assert_eq!(buffer.size, 5);
        assert!(buffer.capacity >= 5);

        unsafe {
            let slice = buffer.as_slice();
            assert_eq!(slice, &[1, 2, 3, 4, 5]);

            let recovered_vec = buffer.into_vec();
            assert_eq!(recovered_vec, vec![1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn test_cstring_operations() {
        let test_string = "Hello, world!".to_string();
        let c_string = CString::from_string(test_string.clone()).unwrap();

        assert!(!c_string.data.is_null());
        assert_eq!(c_string.len, 13);

        unsafe {
            let str_ref = c_string.as_str().unwrap();
            assert_eq!(str_ref, "Hello, world!");

            let recovered_string = c_string.into_string().unwrap();
            assert_eq!(recovered_string, test_string);
        }
    }

    #[test]
    fn test_default_configs() {
        let pool_config = CMemoryPoolConfig::default();
        assert!(pool_config.chunk_size > 0);
        assert!(pool_config.max_chunks > 0);
        assert!(pool_config.alignment > 0);

        let sort_config = CRadixSortConfig::default();
        assert!(sort_config.radix_bits > 0);
        assert!(sort_config.use_counting_sort_threshold > 0);

        let memory_config = CMemoryConfig::default();
        assert!(memory_config.pool_chunk_size > 0);
        assert!(memory_config.max_pool_memory > 0);
    }
}
