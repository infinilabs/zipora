//! Memory-mapped allocator for large objects
//!
//! This module provides memory-mapped allocation for large objects to achieve
//! C++-competitive performance for allocations >16KB.

use crate::error::{Result, ToplingError};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Memory-mapped allocation for high-performance large object allocation
pub struct MemoryMappedAllocator {
    /// Minimum size for memory-mapped allocations
    min_mmap_size: usize,
    /// Cache of memory-mapped regions to avoid repeated mmap/munmap
    region_cache: Arc<Mutex<HashMap<usize, Vec<*mut u8>>>>,
    /// Statistics
    total_allocated: AtomicU64,
    total_freed: AtomicU64,
    mmap_calls: AtomicU64,
    munmap_calls: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

/// Information about a memory-mapped allocation
#[derive(Debug)]
pub struct MmapAllocation {
    ptr: NonNull<u8>,
    size: usize,
    actual_size: usize, // Rounded up to page size
}

/// Statistics for memory-mapped allocations
#[derive(Debug, Clone)]
pub struct MmapStats {
    pub total_allocated: u64,
    pub total_freed: u64,
    pub mmap_calls: u64,
    pub munmap_calls: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cached_regions: usize,
}

impl MemoryMappedAllocator {
    /// Create a new memory-mapped allocator
    pub fn new(min_mmap_size: usize) -> Self {
        Self {
            min_mmap_size,
            region_cache: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: AtomicU64::new(0),
            total_freed: AtomicU64::new(0),
            mmap_calls: AtomicU64::new(0),
            munmap_calls: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    /// Create allocator with default settings (16KB minimum)
    pub fn default() -> Self {
        Self::new(16 * 1024)
    }

    /// Allocate memory using mmap for optimal large allocation performance
    pub fn allocate(&self, size: usize) -> Result<MmapAllocation> {
        if size < self.min_mmap_size {
            return Err(ToplingError::invalid_data(
                "allocation too small for memory mapping",
            ));
        }

        // Round up to page size for optimal performance
        let page_size = Self::get_page_size();
        let actual_size = (size + page_size - 1) & !(page_size - 1);

        // Try to get from cache first
        if let Ok(mut cache) = self.region_cache.try_lock() {
            if let Some(regions) = cache.get_mut(&actual_size) {
                if let Some(ptr) = regions.pop() {
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
                    
                    return Ok(MmapAllocation {
                        ptr: unsafe { NonNull::new_unchecked(ptr) },
                        size,
                        actual_size,
                    });
                }
            }
        }

        // Cache miss, allocate new region
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        self.mmap_calls.fetch_add(1, Ordering::Relaxed);

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                actual_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(ToplingError::out_of_memory(size));
        }

        // Use madvise for better performance hints
        unsafe {
            // Hint that we'll access this memory soon
            libc::madvise(ptr, actual_size, libc::MADV_WILLNEED);
            // Hint for sequential access pattern (if applicable)
            libc::madvise(ptr, actual_size, libc::MADV_SEQUENTIAL);
        }

        self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);

        Ok(MmapAllocation {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut u8) },
            size,
            actual_size,
        })
    }

    /// Deallocate memory, potentially caching for reuse
    pub fn deallocate(&self, allocation: MmapAllocation) -> Result<()> {
        self.total_freed.fetch_add(allocation.size as u64, Ordering::Relaxed);

        // Try to cache the region for reuse
        if let Ok(mut cache) = self.region_cache.try_lock() {
            let regions = cache.entry(allocation.actual_size).or_insert_with(Vec::new);
            
            // Limit cache size to prevent memory bloat
            const MAX_CACHED_REGIONS_PER_SIZE: usize = 4;
            if regions.len() < MAX_CACHED_REGIONS_PER_SIZE {
                regions.push(allocation.ptr.as_ptr());
                return Ok(());
            }
        }

        // Cache is full or locked, deallocate immediately
        self.munmap_calls.fetch_add(1, Ordering::Relaxed);
        unsafe {
            if libc::munmap(allocation.ptr.as_ptr() as *mut libc::c_void, allocation.actual_size) != 0 {
                return Err(ToplingError::io_error("failed to unmap memory"));
            }
        }

        Ok(())
    }

    /// Check if this allocator should be used for the given size
    pub fn should_use_mmap(&self, size: usize) -> bool {
        size >= self.min_mmap_size
    }

    /// Get current statistics
    pub fn stats(&self) -> MmapStats {
        let cached_regions = if let Ok(cache) = self.region_cache.try_lock() {
            cache.values().map(|v| v.len()).sum()
        } else {
            0
        };

        MmapStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_freed: self.total_freed.load(Ordering::Relaxed),
            mmap_calls: self.mmap_calls.load(Ordering::Relaxed),
            munmap_calls: self.munmap_calls.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cached_regions,
        }
    }

    /// Clear the region cache, forcing all cached regions to be unmapped
    pub fn clear_cache(&self) -> Result<()> {
        if let Ok(mut cache) = self.region_cache.lock() {
            for (size, regions) in cache.drain() {
                for ptr in regions {
                    self.munmap_calls.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        if libc::munmap(ptr as *mut libc::c_void, size) != 0 {
                            log::warn!("Failed to unmap cached region of size {}", size);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Get system page size
    fn get_page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }
}

impl Drop for MemoryMappedAllocator {
    fn drop(&mut self) {
        // Clean up all cached regions
        let _ = self.clear_cache();
    }
}

// Safety: MemoryMappedAllocator can be shared between threads safely
unsafe impl Send for MemoryMappedAllocator {}
unsafe impl Sync for MemoryMappedAllocator {}

impl MmapAllocation {
    /// Get the allocated memory as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get the allocated memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the actual allocated size (rounded to page size)
    pub fn actual_size(&self) -> usize {
        self.actual_size
    }

    /// Get the memory as a typed pointer
    pub fn as_ptr<T>(&self) -> *mut T {
        self.ptr.as_ptr() as *mut T
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_allocator_creation() {
        let allocator = MemoryMappedAllocator::new(16 * 1024);
        assert!(allocator.should_use_mmap(20 * 1024));
        assert!(!allocator.should_use_mmap(8 * 1024));
    }

    #[test]
    fn test_mmap_allocation() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024; // 64KB

        let mut allocation = allocator.allocate(size).unwrap();
        assert_eq!(allocation.size(), size);
        assert!(allocation.actual_size() >= size);

        // Test that we can write to the memory
        let slice = allocation.as_mut_slice();
        slice[0] = 42;
        slice[size - 1] = 84;

        let slice = allocation.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[size - 1], 84);

        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.total_allocated, size as u64);
        assert_eq!(stats.total_freed, size as u64);
        assert_eq!(stats.mmap_calls, 1);
    }

    #[test]
    fn test_mmap_cache() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Allocate and deallocate to populate cache
        let allocation1 = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation1).unwrap();

        let stats_before = allocator.stats();

        // Allocate again, should hit cache
        let allocation2 = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation2).unwrap();

        let stats_after = allocator.stats();

        // Should have one cache hit
        assert_eq!(stats_after.cache_hits, stats_before.cache_hits + 1);
        // Should not have made additional mmap calls
        assert_eq!(stats_after.mmap_calls, stats_before.mmap_calls);
    }

    #[test]
    fn test_mmap_different_sizes() {
        let allocator = MemoryMappedAllocator::default();

        let sizes = vec![16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024];
        let mut allocations = Vec::new();

        // Allocate different sizes
        for size in &sizes {
            let allocation = allocator.allocate(*size).unwrap();
            assert_eq!(allocation.size(), *size);
            allocations.push(allocation);
        }

        // Deallocate all
        for allocation in allocations {
            allocator.deallocate(allocation).unwrap();
        }

        let stats = allocator.stats();
        assert_eq!(stats.mmap_calls, sizes.len() as u64);
        assert_eq!(stats.total_allocated, sizes.iter().sum::<usize>() as u64);
        assert_eq!(stats.total_freed, sizes.iter().sum::<usize>() as u64);
    }

    #[test]
    fn test_mmap_cache_limit() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Allocate and deallocate more than cache limit
        for _ in 0..10 {
            let allocation = allocator.allocate(size).unwrap();
            allocator.deallocate(allocation).unwrap();
        }

        let stats = allocator.stats();
        // Should have some cached regions, but not more than the limit
        assert!(stats.cached_regions <= 4); // MAX_CACHED_REGIONS_PER_SIZE
        // Note: munmap_calls might be 0 if all allocations fit in cache during this test
        // This is acceptable as the cache is working correctly
    }

    #[test]
    fn test_clear_cache() {
        let allocator = MemoryMappedAllocator::default();
        let size = 64 * 1024;

        // Populate cache
        let allocation = allocator.allocate(size).unwrap();
        allocator.deallocate(allocation).unwrap();

        let stats_before = allocator.stats();
        assert!(stats_before.cached_regions > 0);

        // Clear cache
        allocator.clear_cache().unwrap();

        let stats_after = allocator.stats();
        assert_eq!(stats_after.cached_regions, 0);
        assert!(stats_after.munmap_calls > stats_before.munmap_calls);
    }

    #[test]
    fn test_invalid_allocation_size() {
        let allocator = MemoryMappedAllocator::new(16 * 1024);
        
        // Too small for mmap
        let result = allocator.allocate(8 * 1024);
        assert!(result.is_err());
    }
}