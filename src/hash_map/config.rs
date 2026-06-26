//! Configuration types for [`ZiporaHashMap`](super::ZiporaHashMap).

use crate::memory::SecureMemoryPool;
use std::sync::Arc;

/// Hash strategy configuration for unified hash map
#[derive(Debug, Clone)]
pub enum HashStrategy {
    /// Robin Hood hashing with probe distance optimization
    RobinHood {
        max_probe_distance: u16,
        variance_reduction: bool,
        backward_shift: bool,
    },
    /// Chaining with hash caching for collision resolution
    Chaining {
        load_factor: f64,
        hash_cache: bool,
        compact_links: bool,
    },
    /// Hopscotch hashing with neighborhood management
    Hopscotch {
        neighborhood_size: u8,
        displacement_threshold: u16,
    },
    /// Linear probing with cache-friendly access patterns
    LinearProbing {
        max_probe_distance: u16,
        cache_aligned: bool,
    },
    /// Cuckoo hashing with multiple hash functions
    Cuckoo {
        num_hash_functions: u8,
        max_evictions: u16,
    },
}

/// Storage strategy for memory layout and allocation
#[derive(Debug, Clone)]
pub enum HashStorageStrategy {
    /// Standard heap allocation with FastVec
    Standard {
        initial_capacity: usize,
        growth_factor: f64,
    },
    /// Inline storage for small collections (N ≤ threshold)
    SmallInline {
        inline_capacity: usize,
        fallback_threshold: usize,
    },
    /// Cache-optimized allocation with alignment
    CacheOptimized {
        cache_line_size: usize,
        numa_aware: bool,
        huge_pages: bool,
    },
    /// String-specialized storage with interning
    StringOptimized {
        arena_size: usize,
        prefix_cache: bool,
        interning: bool,
    },
    /// Memory pool allocation with SecureMemoryPool
    PoolAllocated {
        pool: Arc<SecureMemoryPool>,
        chunk_size: usize,
    },
}

/// Optimization strategy for SIMD, cache, and performance features
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Standard optimization level
    Standard,
    /// SIMD-accelerated operations
    SimdAccelerated {
        string_ops: bool,
        bulk_ops: bool,
        hash_computation: bool,
    },
    /// Cache-aware optimizations
    CacheAware {
        prefetch_distance: usize,
        hot_cold_separation: bool,
        access_pattern_tracking: bool,
    },
    /// High-performance combination of all optimizations
    HighPerformance {
        simd_enabled: bool,
        cache_optimized: bool,
        prefetch_enabled: bool,
        numa_aware: bool,
    },
}

/// Configuration for unified hash map
#[derive(Debug, Clone)]
pub struct ZiporaHashMapConfig {
    pub hash_strategy: HashStrategy,
    pub storage_strategy: HashStorageStrategy,
    pub optimization_strategy: OptimizationStrategy,
    pub initial_capacity: usize,
    pub load_factor: f64,
}

impl Default for ZiporaHashMapConfig {
    fn default() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 64,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: HashStorageStrategy::Standard {
                initial_capacity: 16,
                growth_factor: 2.0,
            },
            optimization_strategy: OptimizationStrategy::HighPerformance {
                simd_enabled: true,
                cache_optimized: true,
                prefetch_enabled: true,
                numa_aware: true,
            },
            initial_capacity: 16,
            load_factor: 0.75,
        }
    }
}

impl ZiporaHashMapConfig {
    /// Create configuration for cache-optimized hash map
    pub fn cache_optimized() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 32,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: HashStorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: true,
                huge_pages: false,
            },
            optimization_strategy: OptimizationStrategy::CacheAware {
                prefetch_distance: 2,
                hot_cold_separation: true,
                access_pattern_tracking: true,
            },
            initial_capacity: 64,
            load_factor: 0.6,
        }
    }

    /// Create configuration for string-optimized hash map
    pub fn string_optimized() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 48,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: HashStorageStrategy::StringOptimized {
                arena_size: 4096,
                prefix_cache: true,
                interning: true,
            },
            optimization_strategy: OptimizationStrategy::SimdAccelerated {
                string_ops: true,
                bulk_ops: true,
                hash_computation: true,
            },
            initial_capacity: 32,
            load_factor: 0.7,
        }
    }

    /// Create configuration for small hash map with inline storage
    pub fn small_inline(inline_capacity: usize) -> Self {
        Self {
            hash_strategy: HashStrategy::LinearProbing {
                max_probe_distance: inline_capacity as u16,
                cache_aligned: true,
            },
            storage_strategy: HashStorageStrategy::SmallInline {
                inline_capacity,
                fallback_threshold: inline_capacity * 2,
            },
            optimization_strategy: OptimizationStrategy::Standard,
            initial_capacity: inline_capacity,
            load_factor: 1.0, // Use all inline capacity
        }
    }

    /// Create configuration for concurrent access with pool allocation
    pub fn concurrent_pool(pool: Arc<SecureMemoryPool>) -> Self {
        Self {
            hash_strategy: HashStrategy::Hopscotch {
                neighborhood_size: 32,
                displacement_threshold: 128,
            },
            storage_strategy: HashStorageStrategy::PoolAllocated {
                pool,
                chunk_size: 1024,
            },
            optimization_strategy: OptimizationStrategy::HighPerformance {
                simd_enabled: true,
                cache_optimized: true,
                prefetch_enabled: true,
                numa_aware: true,
            },
            initial_capacity: 64,
            load_factor: 0.65,
        }
    }
}
