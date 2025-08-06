//! Memory management utilities and allocators
//!
//! This module provides high-performance memory management features including
//! memory pools, bump allocators, and hugepage support for optimal performance.

pub mod bump;
pub mod cache;
pub mod hugepage;
pub mod mmap;
pub mod pool;
pub mod secure_pool;
pub mod tiered;

// Re-export main types
pub use bump::{BumpAllocator, BumpArena};
pub use cache::{
    CacheAlignedVec, NumaStats, NumaPoolStats, get_numa_stats, set_current_numa_node, 
    numa_alloc_aligned, numa_dealloc, get_optimal_numa_node, init_numa_pools, 
    clear_numa_pools, CACHE_LINE_SIZE
};
pub use mmap::{MemoryMappedAllocator, MmapAllocation};
pub use pool::{MemoryPool, PoolConfig, PooledBuffer, PooledVec};
pub use secure_pool::{
    SecureMemoryPool, SecurePoolConfig, SecurePoolStats, SecurePooledPtr,
    get_global_pool_for_size, get_global_secure_pool_stats, size_to_class
};
pub use tiered::{TieredMemoryAllocator, TieredAllocation, TieredConfig, get_tiered_stats, tiered_allocate, tiered_deallocate};

#[cfg(target_os = "linux")]
pub use hugepage::{HugePage, HugePageAllocator};

/// Configuration for memory management behavior
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable memory pools for frequent allocations
    pub use_pools: bool,
    /// Enable hugepage allocation when available
    pub use_hugepages: bool,
    /// Default pool chunk size in bytes
    pub pool_chunk_size: usize,
    /// Maximum memory to keep in pools
    pub max_pool_memory: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_pools: true,
            use_hugepages: cfg!(target_os = "linux"),
            pool_chunk_size: 64 * 1024,        // 64KB chunks
            max_pool_memory: 16 * 1024 * 1024, // 16MB max
        }
    }
}

/// Initialize memory management with the given configuration
pub fn init_memory_management(config: MemoryConfig) -> crate::Result<()> {
    log::debug!("Initializing memory management with config: {:?}", config);

    if config.use_pools {
        pool::init_global_pools(config.pool_chunk_size, config.max_pool_memory)?;
    }

    #[cfg(target_os = "linux")]
    if config.use_hugepages {
        hugepage::init_hugepage_support()?;
    }

    Ok(())
}

/// Get current memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total bytes allocated through pools
    pub pool_allocated: u64,
    /// Total bytes available in pools
    pub pool_available: u64,
    /// Number of active pool chunks
    pub pool_chunks: usize,
    /// Number of hugepages allocated
    pub hugepages_allocated: usize,
}

/// Get global memory statistics
pub fn get_memory_stats() -> MemoryStats {
    let pool_stats = pool::get_global_pool_stats();

    #[cfg(target_os = "linux")]
    let hugepages_allocated = hugepage::get_hugepage_count();
    #[cfg(not(target_os = "linux"))]
    let hugepages_allocated = 0;

    MemoryStats {
        pool_allocated: pool_stats.allocated,
        pool_available: pool_stats.available,
        pool_chunks: pool_stats.chunks,
        hugepages_allocated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_config_default() {
        let config = MemoryConfig::default();
        assert!(config.use_pools);
        assert_eq!(config.pool_chunk_size, 64 * 1024);
        assert_eq!(config.max_pool_memory, 16 * 1024 * 1024);

        #[cfg(target_os = "linux")]
        assert!(config.use_hugepages);

        #[cfg(not(target_os = "linux"))]
        assert!(!config.use_hugepages);
    }

    #[test]
    fn test_memory_stats() {
        let stats = get_memory_stats();
        // Should not panic and should have reasonable values
        assert!(stats.pool_chunks <= 1000); // Sanity check
    }

    #[test]
    fn test_init_memory_management() {
        let config = MemoryConfig {
            use_pools: true,
            use_hugepages: false, // Disable for test
            pool_chunk_size: 4096,
            max_pool_memory: 1024 * 1024,
        };

        let result = init_memory_management(config);
        assert!(result.is_ok());
    }
}
