use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use std::sync::Arc;

/// Default memory pool for serde deserialization
fn default_memory_pool() -> Arc<SecureMemoryPool> {
    // SAFETY: This function is only called during serde deserialization where we need
    // a default pool. We try small_secure first, then default. If both fail, we create
    // a minimal emergency pool with extremely conservative settings that cannot fail.
    SecureMemoryPool::new(SecurePoolConfig::small_secure())
        .or_else(|_| SecureMemoryPool::new(SecurePoolConfig::default()))
        .unwrap_or_else(|_| {
            // Emergency fallback: Create minimal pool with settings that cannot fail
            let mut emergency_config = SecurePoolConfig::default();
            emergency_config.chunk_size = 64; // Minimal chunk size
            emergency_config.max_chunks = 2; // Very limited pool
            emergency_config.use_guard_pages = false; // Disable to avoid allocation failures
            // SAFETY: Minimal config (64B chunks, 2 max, no guards) designed to never fail except in catastrophic OOM
            SecureMemoryPool::new(emergency_config)
                .expect("CRITICAL: Emergency pool creation failed - this should never happen")
        })
}

/// Trie algorithm strategy
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrieStrategy {
    /// Patricia trie with path compression
    Patricia {
        max_path_length: usize,
        compression_threshold: usize,
        adaptive_compression: bool,
    },
    /// Critical-bit trie with binary decisions
    CriticalBit {
        cache_critical_bytes: bool,
        optimize_for_strings: bool,
        bit_level_optimization: bool,
    },
    /// Double array trie with constant-time transitions
    DoubleArray {
        initial_capacity: usize,
        growth_factor: f64,
        free_list_management: bool,
        auto_shrink: bool,
    },
    /// LOUDS trie with succinct representations
    Louds {
        nesting_levels: usize,
        fragment_compression: bool,
        adaptive_backends: bool,
        cache_aligned: bool,
    },
    /// Compressed sparse trie for space efficiency
    CompressedSparse {
        sparse_threshold: f64,
        compression_level: u8,
        adaptive_sparse: bool,
    },
}

/// Storage strategy for memory layout and data structures
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrieStorageStrategy {
    /// Standard vector-based storage
    Standard {
        initial_capacity: usize,
        growth_factor: f64,
    },
    /// Succinct data structures with rank/select
    Succinct {
        bit_vector_type: BitVectorType,
        rank_select_type: RankSelectType,
        interleaved_layout: bool,
    },
    /// Cache-optimized storage with alignment
    CacheOptimized {
        cache_line_size: usize,
        numa_aware: bool,
        prefetch_enabled: bool,
    },
    /// Memory pool allocation
    PoolAllocated {
        #[cfg_attr(feature = "serde", serde(skip, default = "default_memory_pool"))]
        pool: Arc<SecureMemoryPool>,
        size_class: usize,
        chunk_size: usize,
    },
    /// Hybrid storage combining multiple strategies
    Hybrid {
        primary: Box<TrieStorageStrategy>,
        secondary: Box<TrieStorageStrategy>,
        switch_threshold: usize,
    },
}

/// Compression strategy for space optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrieCompressionStrategy {
    /// No compression - full node storage
    None,
    /// Path compression for single-child chains
    PathCompression {
        min_path_length: usize,
        max_path_length: usize,
        adaptive_threshold: bool,
    },
    /// Fragment-based compression for common substrings
    FragmentCompression {
        fragment_size: usize,
        frequency_threshold: f64,
        dictionary_size: usize,
    },
    /// Hierarchical compression with multiple levels
    Hierarchical {
        levels: usize,
        compression_ratio: f64,
        adaptive_levels: bool,
    },
    /// Adaptive compression choosing best strategy
    Adaptive {
        strategies: Vec<TrieCompressionStrategy>,
        decision_threshold: usize,
    },
}

/// Rank/select implementation choice
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RankSelectType {
    /// Interleaved 256-bit blocks
    Interleaved256,
    /// Mixed implementation with dual-dimension interleaving
    MixedIL256,
    /// Extended mixed with multi-dimensional support
    MixedXL256,
    /// Bit-packed hierarchical caching
    MixedXLBitPacked,
    /// Simple implementation for small data
    Simple,
    /// Adaptive selection based on data characteristics
    Adaptive,
}

/// Bit vector implementation choice
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BitVectorType {
    /// Standard bit vector
    Standard,
    /// Rank/select optimized
    RankSelectOptimized,
    /// Cache-aligned bit vector
    CacheAligned,
    /// Compressed bit vector
    Compressed,
}

/// Configuration for unified trie
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZiporaTrieConfig {
    pub trie_strategy: TrieStrategy,
    pub storage_strategy: TrieStorageStrategy,
    pub compression_strategy: TrieCompressionStrategy,
    pub rank_select_type: RankSelectType,
    pub enable_simd: bool,
    pub enable_concurrency: bool,
    pub cache_optimization: bool,
}

impl Default for ZiporaTrieConfig {
    fn default() -> Self {
        Self {
            trie_strategy: TrieStrategy::DoubleArray {
                initial_capacity: 256,
                growth_factor: 1.5,
                free_list_management: false,
                auto_shrink: false,
            },
            storage_strategy: TrieStorageStrategy::Standard {
                initial_capacity: 64,
                growth_factor: 2.0,
            },
            compression_strategy: TrieCompressionStrategy::None,
            rank_select_type: RankSelectType::Adaptive,
            enable_simd: true,
            enable_concurrency: false,
            cache_optimization: true,
        }
    }
}

impl ZiporaTrieConfig {
    /// Get the maximum levels equivalent based on strategy
    pub fn max_levels(&self) -> usize {
        match &self.trie_strategy {
            TrieStrategy::Louds { nesting_levels, .. } => *nesting_levels,
            TrieStrategy::Patricia { .. } => 4, // Default for Patricia
            TrieStrategy::CriticalBit { .. } => 6, // Default for CritBit
            TrieStrategy::DoubleArray { .. } => 5, // Default for DoubleArray
            TrieStrategy::CompressedSparse { .. } => 6, // Default for Sparse
        }
    }

    /// Create configuration for cache-optimized trie
    pub fn cache_optimized() -> Self {
        Self {
            trie_strategy: TrieStrategy::DoubleArray {
                initial_capacity: 512,
                growth_factor: 1.5,
                free_list_management: false,
                auto_shrink: false,
            },
            storage_strategy: TrieStorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: true,
                prefetch_enabled: true,
            },
            compression_strategy: TrieCompressionStrategy::None,
            rank_select_type: RankSelectType::MixedIL256,
            enable_simd: true,
            enable_concurrency: false,
            cache_optimization: true,
        }
    }

    /// Create configuration for space-optimized trie
    pub fn space_optimized() -> Self {
        Self {
            trie_strategy: TrieStrategy::Louds {
                nesting_levels: 4,
                fragment_compression: true,
                adaptive_backends: true,
                cache_aligned: false,
            },
            storage_strategy: TrieStorageStrategy::Succinct {
                bit_vector_type: BitVectorType::Compressed,
                rank_select_type: RankSelectType::MixedXLBitPacked,
                interleaved_layout: true,
            },
            compression_strategy: TrieCompressionStrategy::Hierarchical {
                levels: 3,
                compression_ratio: 0.7,
                adaptive_levels: true,
            },
            rank_select_type: RankSelectType::MixedXLBitPacked,
            enable_simd: true,
            enable_concurrency: false,
            cache_optimization: false,
        }
    }

    /// Create configuration for high-performance concurrent trie
    pub fn concurrent_high_performance(pool: Arc<SecureMemoryPool>) -> Self {
        Self {
            trie_strategy: TrieStrategy::DoubleArray {
                initial_capacity: 1, // Referenced project: minimal start (line 70: states.resize(1))
                growth_factor: 1.5,
                free_list_management: true,
                auto_shrink: false,
            },
            storage_strategy: TrieStorageStrategy::PoolAllocated {
                pool,
                size_class: 1024,
                chunk_size: 4096,
            },
            compression_strategy: TrieCompressionStrategy::None,
            rank_select_type: RankSelectType::Adaptive,
            enable_simd: true,
            enable_concurrency: true,
            cache_optimization: true,
        }
    }

    /// Create configuration for sparse data optimization
    pub fn sparse_optimized() -> Self {
        Self {
            trie_strategy: TrieStrategy::CompressedSparse {
                sparse_threshold: 0.3,
                compression_level: 6,
                adaptive_sparse: true,
            },
            storage_strategy: TrieStorageStrategy::Succinct {
                bit_vector_type: BitVectorType::Compressed,
                rank_select_type: RankSelectType::Simple,
                interleaved_layout: false,
            },
            compression_strategy: TrieCompressionStrategy::FragmentCompression {
                fragment_size: 8,
                frequency_threshold: 0.1,
                dictionary_size: 4096,
            },
            rank_select_type: RankSelectType::Simple,
            enable_simd: true,
            enable_concurrency: false,
            cache_optimization: false,
        }
    }

    /// Create configuration for string-specialized trie
    pub fn string_specialized() -> Self {
        Self {
            trie_strategy: TrieStrategy::CriticalBit {
                cache_critical_bytes: true,
                optimize_for_strings: true,
                bit_level_optimization: true,
            },
            storage_strategy: TrieStorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: false,
                prefetch_enabled: true,
            },
            compression_strategy: TrieCompressionStrategy::PathCompression {
                min_path_length: 1,
                max_path_length: 64,
                adaptive_threshold: true,
            },
            rank_select_type: RankSelectType::Interleaved256,
            enable_simd: true,
            enable_concurrency: false,
            cache_optimization: true,
        }
    }
}
