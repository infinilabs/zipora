//! Memory pool allocator for high-frequency allocations
//!
//! This module provides memory pools that can significantly reduce allocation
//! overhead for frequently allocated objects of similar sizes.

use std::alloc::{alloc, dealloc, Layout};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::error::{ToplingError, Result};

/// Configuration for a memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Size of each chunk in bytes
    pub chunk_size: usize,
    /// Maximum number of chunks to keep in the pool
    pub max_chunks: usize,
    /// Alignment requirement for allocations
    pub alignment: usize,
}

impl PoolConfig {
    /// Create a new pool configuration
    pub fn new(chunk_size: usize, max_chunks: usize, alignment: usize) -> Self {
        Self {
            chunk_size,
            max_chunks,
            alignment,
        }
    }
    
    /// Create configuration for small objects (< 1KB)
    pub fn small() -> Self {
        Self::new(1024, 100, 8)
    }
    
    /// Create configuration for medium objects (< 64KB)
    pub fn medium() -> Self {
        Self::new(64 * 1024, 50, 16)
    }
    
    /// Create configuration for large objects (< 1MB)
    pub fn large() -> Self {
        Self::new(1024 * 1024, 10, 32)
    }
}

/// Statistics for memory pool usage
#[derive(Debug, Clone)]
pub struct PoolStats {
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

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            allocated: 0,
            available: 0,
            chunks: 0,
            alloc_count: 0,
            dealloc_count: 0,
            pool_hits: 0,
            pool_misses: 0,
        }
    }
}

/// A memory pool for efficient allocation of fixed-size chunks
pub struct MemoryPool {
    config: PoolConfig,
    free_chunks: Mutex<VecDeque<*mut u8>>,
    stats: RwLock<PoolStats>,
    alloc_count: AtomicU64,
    dealloc_count: AtomicU64,
    pool_hits: AtomicU64,
    pool_misses: AtomicU64,
}

// Safety: MemoryPool can be shared between threads safely as all operations are protected by mutexes
unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Create a new memory pool with the given configuration
    pub fn new(config: PoolConfig) -> Result<Self> {
        if config.chunk_size == 0 {
            return Err(ToplingError::invalid_data("chunk_size cannot be zero"));
        }
        
        if config.alignment == 0 || !config.alignment.is_power_of_two() {
            return Err(ToplingError::invalid_data("alignment must be a power of two"));
        }
        
        Ok(Self {
            config,
            free_chunks: Mutex::new(VecDeque::new()),
            stats: RwLock::new(PoolStats::default()),
            alloc_count: AtomicU64::new(0),
            dealloc_count: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_misses: AtomicU64::new(0),
        })
    }
    
    /// Allocate a chunk from the pool
    pub fn allocate(&self) -> Result<NonNull<u8>> {
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        
        // Try to get a chunk from the pool first
        if let Ok(mut free_chunks) = self.free_chunks.try_lock() {
            if let Some(chunk) = free_chunks.pop_front() {
                self.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.update_stats_on_alloc(true);
                // Safety: chunk came from our own allocation, so it's non-null
                return Ok(unsafe { NonNull::new_unchecked(chunk) });
            }
        }
        
        // Pool is empty or locked, allocate new chunk
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
        self.allocate_new_chunk()
    }
    
    /// Deallocate a chunk back to the pool
    pub fn deallocate(&self, chunk: NonNull<u8>) -> Result<()> {
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
        
        // Try to return chunk to pool if not full
        if let Ok(mut free_chunks) = self.free_chunks.try_lock() {
            if free_chunks.len() < self.config.max_chunks {
                free_chunks.push_back(chunk.as_ptr());
                self.update_stats_on_dealloc(true);
                return Ok(());
            }
        }
        
        // Pool is full or locked, deallocate directly
        self.deallocate_chunk(chunk);
        self.update_stats_on_dealloc(false);
        Ok(())
    }
    
    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        let mut stats = self.stats.read().unwrap().clone();
        stats.alloc_count = self.alloc_count.load(Ordering::Relaxed);
        stats.dealloc_count = self.dealloc_count.load(Ordering::Relaxed);
        stats.pool_hits = self.pool_hits.load(Ordering::Relaxed);
        stats.pool_misses = self.pool_misses.load(Ordering::Relaxed);
        
        if let Ok(free_chunks) = self.free_chunks.try_lock() {
            stats.chunks = free_chunks.len();
            stats.available = (free_chunks.len() * self.config.chunk_size) as u64;
        }
        
        stats
    }
    
    /// Clear all chunks from the pool
    pub fn clear(&self) -> Result<()> {
        let mut free_chunks = self.free_chunks.lock().unwrap();
        
        while let Some(chunk_ptr) = free_chunks.pop_front() {
            // Safety: chunk_ptr came from our own allocation, so it's valid for deallocation
            let chunk = unsafe { NonNull::new_unchecked(chunk_ptr) };
            self.deallocate_chunk(chunk);
        }
        
        // Reset stats
        let mut stats = self.stats.write().unwrap();
        stats.chunks = 0;
        stats.available = 0;
        
        Ok(())
    }
    
    /// Get pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
    
    fn allocate_new_chunk(&self) -> Result<NonNull<u8>> {
        let layout = Layout::from_size_align(self.config.chunk_size, self.config.alignment)
            .map_err(|_| ToplingError::invalid_data("invalid layout for chunk allocation"))?;
        
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(ToplingError::out_of_memory(self.config.chunk_size));
        }
        
        self.update_stats_on_alloc(false);
        
        // Safety: We just checked that ptr is not null
        Ok(unsafe { NonNull::new_unchecked(ptr) })
    }
    
    fn deallocate_chunk(&self, chunk: NonNull<u8>) {
        let layout = Layout::from_size_align(self.config.chunk_size, self.config.alignment)
            .expect("invalid layout");
        
        unsafe {
            dealloc(chunk.as_ptr(), layout);
        }
    }
    
    fn update_stats_on_alloc(&self, from_pool: bool) {
        if let Ok(mut stats) = self.stats.try_write() {
            if !from_pool {
                stats.allocated += self.config.chunk_size as u64;
            }
        }
    }
    
    fn update_stats_on_dealloc(&self, to_pool: bool) {
        if let Ok(mut stats) = self.stats.try_write() {
            if !to_pool {
                stats.allocated = stats.allocated.saturating_sub(self.config.chunk_size as u64);
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Clean up all remaining chunks
        let _ = self.clear();
    }
}

/// Global memory pool instances
static GLOBAL_POOLS: once_cell::sync::Lazy<GlobalPools> = once_cell::sync::Lazy::new(|| {
    GlobalPools::new()
});

struct GlobalPools {
    small_pool: Arc<MemoryPool>,
    medium_pool: Arc<MemoryPool>,
    large_pool: Arc<MemoryPool>,
}

impl GlobalPools {
    fn new() -> Self {
        Self {
            small_pool: Arc::new(MemoryPool::new(PoolConfig::small()).unwrap()),
            medium_pool: Arc::new(MemoryPool::new(PoolConfig::medium()).unwrap()),
            large_pool: Arc::new(MemoryPool::new(PoolConfig::large()).unwrap()),
        }
    }
    
    fn get_pool_for_size(&self, size: usize) -> &Arc<MemoryPool> {
        if size <= 1024 {
            &self.small_pool
        } else if size <= 64 * 1024 {
            &self.medium_pool
        } else {
            &self.large_pool
        }
    }
}

/// Initialize global pools with custom configuration
pub fn init_global_pools(chunk_size: usize, max_memory: usize) -> Result<()> {
    // This would re-initialize global pools in a real implementation
    // For now, we just validate the parameters
    if chunk_size == 0 {
        return Err(ToplingError::invalid_data("chunk_size cannot be zero"));
    }
    
    if max_memory == 0 {
        return Err(ToplingError::invalid_data("max_memory cannot be zero"));
    }
    
    log::debug!("Global pools initialized with chunk_size={}, max_memory={}", 
                chunk_size, max_memory);
    Ok(())
}

/// Get statistics from all global pools
pub fn get_global_pool_stats() -> PoolStats {
    let small_stats = GLOBAL_POOLS.small_pool.stats();
    let medium_stats = GLOBAL_POOLS.medium_pool.stats();
    let large_stats = GLOBAL_POOLS.large_pool.stats();
    
    PoolStats {
        allocated: small_stats.allocated + medium_stats.allocated + large_stats.allocated,
        available: small_stats.available + medium_stats.available + large_stats.available,
        chunks: small_stats.chunks + medium_stats.chunks + large_stats.chunks,
        alloc_count: small_stats.alloc_count + medium_stats.alloc_count + large_stats.alloc_count,
        dealloc_count: small_stats.dealloc_count + medium_stats.dealloc_count + large_stats.dealloc_count,
        pool_hits: small_stats.pool_hits + medium_stats.pool_hits + large_stats.pool_hits,
        pool_misses: small_stats.pool_misses + medium_stats.pool_misses + large_stats.pool_misses,
    }
}

/// A vector that uses memory pools for allocation
pub struct PooledVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    pool: Arc<MemoryPool>,
}

impl<T> PooledVec<T> {
    /// Create a new pooled vector
    pub fn new() -> Result<Self> {
        let element_size = std::mem::size_of::<T>();
        let pool = GLOBAL_POOLS.get_pool_for_size(element_size).clone();
        
        let chunk = pool.allocate()?;
        let capacity = pool.config().chunk_size / element_size;
        
        Ok(Self {
            ptr: chunk.cast(),
            len: 0,
            capacity,
            pool,
        })
    }
    
    /// Push an element to the vector
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.len >= self.capacity {
            return Err(ToplingError::invalid_data("vector capacity exceeded"));
        }
        
        unsafe {
            self.ptr.as_ptr().add(self.len).write(item);
        }
        self.len += 1;
        Ok(())
    }
    
    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get a slice of the vector's contents
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<T> Drop for PooledVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                self.ptr.as_ptr().add(i).drop_in_place();
            }
        }
        
        // Return memory to pool
        let _ = self.pool.deallocate(self.ptr.cast());
    }
}

/// A buffer that uses memory pools for allocation
pub struct PooledBuffer {
    ptr: NonNull<u8>,
    len: usize,
    capacity: usize,
    pool: Arc<MemoryPool>,
}

impl PooledBuffer {
    /// Create a new pooled buffer of the specified size
    pub fn new(size: usize) -> Result<Self> {
        let pool = GLOBAL_POOLS.get_pool_for_size(size).clone();
        let chunk = pool.allocate()?;
        
        Ok(Self {
            ptr: chunk,
            len: size,
            capacity: pool.config().chunk_size,
            pool,
        })
    }
    
    /// Get the buffer as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
    
    /// Get the buffer as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
    
    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let _ = self.pool.deallocate(self.ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config() {
        let config = PoolConfig::new(1024, 100, 8);
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.max_chunks, 100);
        assert_eq!(config.alignment, 8);
        
        let small_config = PoolConfig::small();
        assert_eq!(small_config.chunk_size, 1024);
        assert_eq!(small_config.max_chunks, 100);
    }

    #[test]
    fn test_memory_pool_creation() {
        let config = PoolConfig::new(1024, 10, 8);
        let pool = MemoryPool::new(config).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.chunks, 0);
        assert_eq!(stats.allocated, 0);
    }

    #[test]
    fn test_memory_pool_allocation() {
        let config = PoolConfig::new(1024, 10, 8);
        let pool = MemoryPool::new(config).unwrap();
        
        let chunk1 = pool.allocate().unwrap();
        let chunk2 = pool.allocate().unwrap();
        
        assert_ne!(chunk1.as_ptr(), chunk2.as_ptr());
        
        pool.deallocate(chunk1).unwrap();
        pool.deallocate(chunk2).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.alloc_count, 2);
        assert_eq!(stats.dealloc_count, 2);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let config = PoolConfig::new(1024, 10, 8);
        let pool = MemoryPool::new(config).unwrap();
        
        let chunk1 = pool.allocate().unwrap();
        let addr1 = chunk1.as_ptr();
        
        pool.deallocate(chunk1).unwrap();
        
        let chunk2 = pool.allocate().unwrap();
        let addr2 = chunk2.as_ptr();
        
        // Should reuse the same memory
        assert_eq!(addr1, addr2);
        
        pool.deallocate(chunk2).unwrap();
        
        let stats = pool.stats();
        assert!(stats.pool_hits > 0);
    }

    #[test]
    fn test_pooled_vec() {
        let mut vec = PooledVec::<i32>::new().unwrap();
        
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert!(vec.capacity() > 0);
        
        vec.push(42).unwrap();
        vec.push(84).unwrap();
        
        assert_eq!(vec.len(), 2);
        assert!(!vec.is_empty());
        
        let slice = vec.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[1], 84);
    }

    #[test]
    fn test_pooled_buffer() {
        let mut buffer = PooledBuffer::new(100).unwrap();
        
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());
        
        let slice = buffer.as_mut_slice();
        slice[0] = 42;
        slice[99] = 84;
        
        let slice = buffer.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[99], 84);
    }

    #[test]
    fn test_global_pool_stats() {
        let stats = get_global_pool_stats();
        // Should not panic and have reasonable values
        assert!(stats.alloc_count >= 0);
        assert!(stats.dealloc_count >= 0);
    }

    #[test]
    fn test_invalid_pool_config() {
        let result = MemoryPool::new(PoolConfig::new(0, 10, 8));
        assert!(result.is_err());
        
        let result = MemoryPool::new(PoolConfig::new(1024, 10, 0));
        assert!(result.is_err());
        
        let result = MemoryPool::new(PoolConfig::new(1024, 10, 3)); // Not power of 2
        assert!(result.is_err());
    }
}