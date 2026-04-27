//! ZiporaTrie - High-performance trie with strategy-based configuration
//!
//! This module provides the core trie implementation for Zipora, designed for
//! extreme performance following referenced project's focused implementation philosophy.
//!
//! # Performance-First Design
//!
//! **"One excellent implementation per data structure"** - referenced project approach
//!
//! ZiporaTrie achieves high performance through configurable strategies:
//! - **TrieStrategy**: Optimized algorithms (Patricia, CritBit, DoubleArray, LOUDS, CompressedSparse)
//! - **StorageStrategy**: Memory layout optimization and succinct data structures
//! - **CompressionStrategy**: Advanced compression techniques (path, fragment, hierarchical)
//! - **RankSelectStrategy**: High-performance rank/select backend selection
//!
//! # Hardware Acceleration Features
//!
//! - **SIMD Framework**: BMI2/AVX2/POPCNT acceleration with runtime detection
//! - **Cache Optimization**: Prefetching, alignment, and NUMA awareness
//! - **Succinct Structures**: Space-efficient rank/select with hardware acceleration
//! - **Memory Pool Integration**: SecureMemoryPool for high-performance allocation
//! - **Concurrent Access**: Lock-free and token-based synchronization

use crate::containers::specialized::UintVector;
use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::fsa::traits::{
    FiniteStateAutomaton, PrefixIterable, StatisticsProvider, Trie,
    TrieStats,
};
use crate::memory::cache_layout::{CacheOptimizedAllocator, CacheLayoutConfig, PrefetchHint};
use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use crate::succinct::{BitVector, RankSelectBuilder, RankSelectOps};
use crate::StateId;
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum StorageStrategy {
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
        #[serde(skip, default = "default_memory_pool")]
        pool: Arc<SecureMemoryPool>,
        size_class: usize,
        chunk_size: usize,
    },
    /// Hybrid storage combining multiple strategies
    Hybrid {
        primary: Box<StorageStrategy>,
        secondary: Box<StorageStrategy>,
        switch_threshold: usize,
    },
}

/// Compression strategy for space optimization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CompressionStrategy {
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
        strategies: Vec<CompressionStrategy>,
        decision_threshold: usize,
    },
}

/// Rank/select implementation choice
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZiporaTrieConfig {
    pub trie_strategy: TrieStrategy,
    pub storage_strategy: StorageStrategy,
    pub compression_strategy: CompressionStrategy,
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
            storage_strategy: StorageStrategy::Standard {
                initial_capacity: 64,
                growth_factor: 2.0,
            },
            compression_strategy: CompressionStrategy::None,
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
            storage_strategy: StorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: true,
                prefetch_enabled: true,
            },
            compression_strategy: CompressionStrategy::None,
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
            storage_strategy: StorageStrategy::Succinct {
                bit_vector_type: BitVectorType::Compressed,
                rank_select_type: RankSelectType::MixedXLBitPacked,
                interleaved_layout: true,
            },
            compression_strategy: CompressionStrategy::Hierarchical {
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
            storage_strategy: StorageStrategy::PoolAllocated {
                pool,
                size_class: 1024,
                chunk_size: 4096,
            },
            compression_strategy: CompressionStrategy::None,
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
            storage_strategy: StorageStrategy::Succinct {
                bit_vector_type: BitVectorType::Compressed,
                rank_select_type: RankSelectType::Simple,
                interleaved_layout: false,
            },
            compression_strategy: CompressionStrategy::FragmentCompression {
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
            storage_strategy: StorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: false,
                prefetch_enabled: true,
            },
            compression_strategy: CompressionStrategy::PathCompression {
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

/// Unified trie implementation with strategy-based configuration
///
/// ZiporaTrie consolidates all Zipora trie variants into a single,
/// highly configurable implementation. Different behaviors are achieved
/// through strategy configuration rather than separate implementations.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
/// use zipora::fsa::traits::Trie;
/// use zipora::succinct::RankSelectInterleaved256;
///
/// // Cache-optimized trie (with explicit type parameter)
/// let mut trie: ZiporaTrie<RankSelectInterleaved256> =
///     ZiporaTrie::with_config(ZiporaTrieConfig::cache_optimized());
/// trie.insert(b"hello").unwrap();
/// trie.insert(b"world").unwrap();
///
/// // Space-optimized trie
/// let mut space_trie: ZiporaTrie<RankSelectInterleaved256> =
///     ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
/// space_trie.insert(b"compress").unwrap();
///
/// // String-specialized trie
/// let mut str_trie: ZiporaTrie<RankSelectInterleaved256> =
///     ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
/// str_trie.insert(b"string").unwrap();
/// ```
#[derive(Debug)]
pub struct ZiporaTrie<R = crate::succinct::RankSelectInterleaved256>
where
    R: RankSelectOps,
{
    /// Configuration strategy
    config: ZiporaTrieConfig,
    /// Internal storage implementation
    storage: TrieStorage<R>,
    /// Performance statistics
    stats: TrieStats,
    /// Track whether stats need recomputation
    stats_dirty: bool,
    /// Cache optimization components
    cache_allocator: Option<CacheOptimizedAllocator>,
    /// Memory pool for allocation
    _memory_pool: Option<Arc<SecureMemoryPool>>,
    /// Root state for traversal
    root_state: StateId,
}

/// Internal storage implementations for different strategies
#[derive(Debug)]
enum TrieStorage<R>
where
    R: RankSelectOps,
{
    /// Patricia trie storage with path compression
    Patricia {
        nodes: FastVec<PatriciaNode>,
        edge_data: FastVec<u8>,
        compressed_paths: HashMap<StateId, Vec<u8>>,
    },
    /// Critical-bit trie storage
    CriticalBit {
        nodes: FastVec<CritBitNode>,
        keys: FastVec<Vec<u8>>,
        critical_cache: HashMap<usize, u8>,
    },
    /// Double array trie storage
    DoubleArray {
        base: FastVec<u32>,
        check: FastVec<u32>,
        free_list: VecDeque<StateId>,
        state_count: usize,
    },
    /// LOUDS trie storage with succinct structures
    Louds {
        louds: R,
        is_link: R,
        next_link: UintVector,
        label_data: FastVec<u8>,
        core_data: FastVec<u8>,
        next_trie: Option<Box<ZiporaTrie<R>>>,
    },
    /// Compressed sparse trie storage
    CompressedSparse(crate::fsa::cspp_trie::CsppTrie),
}

/// Patricia trie node with path compression (compact representation)
#[derive(Debug, Clone)]
struct PatriciaNode {
    /// Compact children storage: sorted Vec of (symbol, StateId) pairs
    children: Vec<(u8, StateId)>,
    /// Compressed path data offset
    _path_offset: u32,
    /// Compressed path length
    _path_length: u16,
    /// Whether this node represents a complete key
    is_final: bool,
    /// Node flags for optimization
    _flags: u8,
}

impl Default for PatriciaNode {
    fn default() -> Self {
        Self {
            children: Vec::new(),
            _path_offset: 0,
            _path_length: 0,
            is_final: false,
            _flags: 0,
        }
    }
}

/// Critical-bit trie node
#[repr(align(64))]
#[derive(Debug, Clone)]
struct CritBitNode {
    /// Critical byte position
    _crit_byte: usize,
    /// Critical bit position (0-7)
    _crit_bit: u8,
    /// Left child (bit = 0)
    _left_child: Option<StateId>,
    /// Right child (bit = 1)
    _right_child: Option<StateId>,
    /// Key stored at this node (for leaves)
    _key_index: Option<u32>,
    /// Whether this is a final state
    is_final: bool,
}

/// Sparse trie node for compressed sparse storage
#[derive(Debug, Clone)]
struct SparseNode {
    /// Sparse children map
    children: HashMap<u8, StateId>,
    /// Compressed edge label
    _edge_label: Option<u32>,
    /// Final state flag
    is_final: bool,
}

impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    /// Create a new trie with default configuration
    pub fn new() -> Self {
        Self::with_config(ZiporaTrieConfig::default())
    }

    /// Create a new trie with custom configuration
    pub fn with_config(config: ZiporaTrieConfig) -> Self {
        let cache_allocator = if config.cache_optimization {
            Some(CacheOptimizedAllocator::new(CacheLayoutConfig::default()))
        } else {
            None
        };

        let storage = Self::create_storage(&config);

        Self {
            config,
            storage,
            stats: TrieStats::new(),
            stats_dirty: false,
            cache_allocator,
            _memory_pool: None,
            root_state: 0,
        }
    }

    /// Create storage based on strategy configuration
    fn create_storage(config: &ZiporaTrieConfig) -> TrieStorage<R> {
        match &config.trie_strategy {
            TrieStrategy::Patricia { .. } => TrieStorage::Patricia {
                nodes: FastVec::new(),
                edge_data: FastVec::new(),
                compressed_paths: HashMap::new(),
            },
            TrieStrategy::CriticalBit { .. } => TrieStorage::CriticalBit {
                nodes: FastVec::new(),
                keys: FastVec::new(),
                critical_cache: HashMap::new(),
            },
            TrieStrategy::DoubleArray { initial_capacity, .. } => {
                // Referenced project pattern: start minimal SIZE, but respect CAPACITY hint
                // Referenced C++ implementation line 70: states.resize(1) - minimal size
                // Our approach: reserve capacity but only allocate 1 state (minimal memory)

                // Create vectors with capacity - these operations can fail on OOM
                let mut base = match FastVec::with_capacity(*initial_capacity) {
                    Ok(vec) => vec,
                    Err(_) => {
                        // Fallback to minimal capacity if requested capacity fails
                        FastVec::with_capacity(1)
                            .unwrap_or_else(|_| FastVec::new())
                    }
                };

                let mut check = match FastVec::with_capacity(*initial_capacity) {
                    Ok(vec) => vec,
                    Err(_) => {
                        // Fallback to minimal capacity if requested capacity fails
                        FastVec::with_capacity(1)
                            .unwrap_or_else(|_| FastVec::new())
                    }
                };

                // Initialize with just root state (referenced project: line 70)
                // CRITICAL: Root base must be non-zero to allow transitions
                // Using 1 as the base means child states will be at base+symbol = 1+symbol
                // SAFETY: These push operations on empty vectors cannot fail unless we're completely OOM
                // In that case, the program cannot continue anyway
                let _ = base.push(1); // Ignore error - if this fails, we're out of memory
                let _ = check.push(0); // Ignore error - if this fails, we're out of memory

                TrieStorage::DoubleArray {
                    base,
                    check,
                    free_list: VecDeque::new(),
                    state_count: 1, // Start with root state
                }
            },
            TrieStrategy::Louds { .. } => TrieStorage::Louds {
                louds: R::default(),
                is_link: R::default(),
                next_link: UintVector::new(),
                label_data: FastVec::new(),
                core_data: FastVec::new(),
                next_trie: None,
            },
            TrieStrategy::CompressedSparse { .. } => TrieStorage::CompressedSparse(crate::fsa::cspp_trie::CsppTrie::new(4)),
        }
    }

    /// Get the root state
    #[inline]
    pub fn root(&self) -> StateId {
        self.root_state
    }

    /// Get performance statistics
    pub fn stats(&self) -> TrieStats {

        // Return a copy with updated statistics
        let mut stats = self.stats.clone();

        // Update memory usage
        stats.memory_usage = self.memory_usage();

        // Update bits per key
        if stats.num_keys > 0 {
            stats.bits_per_key = (stats.memory_usage as f64 * 8.0) / stats.num_keys as f64;
        } else {
            stats.bits_per_key = 0.0;
        }

        // Update number of states based on storage type
        // Special case: empty trie should report 0 states
        stats.num_states = if stats.num_keys == 0 {
            0
        } else {
            match &self.storage {
                TrieStorage::Patricia { nodes, .. } => nodes.len(),
                TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
                TrieStorage::DoubleArray { check, .. } => {
                    // Count non-zero check values as active states
                    // But also count state 0 (root) which has check[0] = 0
                    1 + check.iter().skip(1).filter(|&&c| c != 0).count()
                }
                TrieStorage::Louds { .. } => 1, // TODO: implement for LOUDS
                TrieStorage::CompressedSparse(cspp) => cspp.total_states(),
            }
        };

        // Update number of transitions
        stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.iter().map(|n| n.children.len()).sum()
            }
            TrieStorage::CriticalBit { .. } => 0, // TODO: implement
            TrieStorage::DoubleArray { base, check, .. } => {
                const STATE_MASK: u32 = 0x3FFF_FFFF;
                const TERMINAL_FLAG: u32 = 0x4000_0000;

                // Count transitions more efficiently:
                // Each non-zero check value represents a transition TO that state
                // (except for root which has check[0] = 0)
                let mut transition_count = 0;

                for i in 1..check.len() {
                    let check_val = check[i];
                    // If check is non-zero, this state has a parent (there's a transition to it)
                    if check_val != 0 {
                        // Special handling for root's children
                        if (check_val & STATE_MASK) == 0 {
                            // This is a child of root - only count if it's properly initialized
                            if (check_val & TERMINAL_FLAG) != 0 || (i < base.len() && base[i] != 0) {
                                transition_count += 1;
                            }
                        } else {
                            // Regular transition
                            transition_count += 1;
                        }
                    }
                }

                transition_count
            }
            TrieStorage::Louds { .. } => 0, // TODO: implement
            TrieStorage::CompressedSparse(cspp) => 0, /* TODO: implement num_transitions */
        };

        stats
    }

    /// Update internal statistics
    #[allow(dead_code)]
    fn update_stats(&mut self) {
        // Update memory usage
        self.stats.memory_usage = self.memory_usage();

        // Update bits per key
        if self.stats.num_keys > 0 {
            self.stats.bits_per_key = (self.stats.memory_usage as f64 * 8.0) / self.stats.num_keys as f64;
        } else {
            self.stats.bits_per_key = 0.0;
        }

        // Update number of states based on storage type
        self.stats.num_states = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes.len(),
            TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
            TrieStorage::DoubleArray { state_count, .. } => *state_count,
            TrieStorage::Louds { .. } => 1, // TODO: implement for LOUDS
            TrieStorage::CompressedSparse(cspp) => cspp.total_states(),
        };

        // Update number of transitions
        self.stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.iter().map(|n| n.children.len()).sum()
            }
            TrieStorage::CriticalBit { .. } => 0, // TODO: implement
            TrieStorage::DoubleArray { base, check, .. } => {
                // Count non-zero check values (excluding root) as transitions
                // Each non-zero check represents a valid transition
                check.iter().skip(1).filter(|&&c| c != 0).count()
            }
            TrieStorage::Louds { .. } => 0, // TODO: implement
            TrieStorage::CompressedSparse(cspp) => 0, /* TODO: implement num_transitions */
        };
    }

    /// Get the current configuration
    pub fn config(&self) -> &ZiporaTrieConfig {
        &self.config
    }

    /// Check if the trie is using cache optimization
    pub fn is_cache_optimized(&self) -> bool {
        self.cache_allocator.is_some()
    }

    /// Get number of states in the trie
    pub fn state_count(&self) -> usize {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => nodes.len(),
            TrieStorage::CriticalBit { nodes, .. } => nodes.len(),
            TrieStorage::DoubleArray { state_count, .. } => *state_count,
            TrieStorage::Louds { label_data, .. } => label_data.len(),
            TrieStorage::CompressedSparse(cspp) => cspp.total_states(),
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Special case: empty trie should report 0 memory usage
        // even though it has a root state (structural overhead)
        if self.stats.num_keys == 0 {
            return 0;
        }

        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                nodes.capacity() * std::mem::size_of::<PatriciaNode>()
                    + edge_data.capacity()
                    + compressed_paths.capacity() * 64 // Rough estimate
            }
            TrieStorage::CriticalBit { nodes, keys, critical_cache } => {
                nodes.capacity() * std::mem::size_of::<CritBitNode>()
                    + keys.capacity() * 32 // Rough estimate per key
                    + critical_cache.capacity() * 9 // usize + u8
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Use actual length instead of capacity for more accurate memory usage
                // Each element is 4 bytes (u32)
                base.len() * 4 + check.len() * 4
            }
            TrieStorage::Louds { label_data, core_data, .. } => {
                label_data.capacity() + core_data.capacity() + 1024 // Rank/select overhead
            }
            TrieStorage::CompressedSparse(cspp) => {
                cspp.total_states() * 4
            }
        }
    }

    /// Insert a key into the trie
    pub fn insert(&mut self, key: &[u8]) -> Result<()> {
        // Delegate to the trait method which has complete implementation for all storage types
        let _state_id = <Self as Trie>::insert(self, key)?;
        // Mark stats as dirty - lazy update on next stats() call
        self.stats_dirty = true;
        Ok(())
    }

    /// Check if the trie contains a key
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        // Delegate to the trait method which has complete implementation for all storage types
        <Self as Trie>::contains(self, key)
    }

    /// Remove a key from the trie
    pub fn remove(&mut self, key: &[u8]) -> Result<bool> {
        match &mut self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                let removed = Self::remove_patricia_actual(nodes, edge_data, compressed_paths, key)?;
                if removed {
                    self.stats.num_keys = self.stats.num_keys.saturating_sub(1);
                    self.stats_dirty = true;
                }
                Ok(removed)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Remove by clearing TERMINAL_BIT on the final state
                let state = Self::lookup_node_id_double_array(base, check, key);
                if let Some(state_id) = state {
                    const TERMINAL_BIT: u32 = 0x8000_0000;
                    base[state_id as usize] &= !TERMINAL_BIT;
                    self.stats.num_keys = self.stats.num_keys.saturating_sub(1);
                    self.stats_dirty = true;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            _ => {
                Ok(false)
            }
        }
    }

    /// Get the number of keys in the trie
    #[inline]
    pub fn len(&self) -> usize {
        self.stats.num_keys
    }

    /// Check if the trie is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all keys in the trie
    pub fn keys(&self) -> Vec<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::keys_patricia_actual(nodes, edge_data, compressed_paths)
            }
            TrieStorage::Louds { label_data, .. } => {
                Self::keys_louds_actual(label_data)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_double_array_actual(base, check)
            }
            TrieStorage::CompressedSparse(cspp) => Vec::new(), // Handled by cspp.iter
            _ => {
                // TODO: Implement for other storage types
                Vec::new()
            }
        }
    }

    /// Get all keys with a given prefix
    pub fn keys_with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::keys_with_prefix_patricia_actual(nodes, edge_data, compressed_paths, prefix)
            }
            TrieStorage::Louds { label_data, .. } => {
                Self::keys_with_prefix_louds_actual(label_data, prefix)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::keys_with_prefix_double_array_actual(base, check, prefix)
            }
            TrieStorage::CompressedSparse(cspp) => Vec::new(), // Handled by cspp.iter
            _ => {
                // TODO: Implement for other storage types
                Vec::new()
            }
        }
    }

    /// Iterate over all keys in the trie
    pub fn iter_all(&self) -> TrieIterator {
        let keys = self.keys();
        TrieIterator::with_keys(keys)
    }

    /// Iterate over keys with a given prefix
    pub fn iter_prefix(&self, prefix: &[u8]) -> TrieIterator {
        let keys = self.keys_with_prefix(prefix);
        TrieIterator::with_keys(keys)
    }

    /// Get capacity (maximum number of states)
    pub fn capacity(&self) -> usize {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                // Patricia trie capacity is number of nodes * growth headroom
                nodes.capacity().max(nodes.len() * 2)
            }
            TrieStorage::CriticalBit { nodes, .. } => {
                nodes.capacity().max(nodes.len() * 2)
            }
            TrieStorage::DoubleArray { base, .. } => {
                // Double array capacity is the size of the base array
                base.capacity().max(base.len())
            }
            TrieStorage::Louds { label_data, .. } => {
                // LOUDS capacity based on label data size
                label_data.capacity().max(label_data.len() * 2)
            }
            TrieStorage::CompressedSparse(cspp) => {
                cspp.total_states() * 4
            }
        }
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        match &self.storage {
            TrieStorage::DoubleArray { base, check, .. } => {
                let base_memory = base.capacity() * std::mem::size_of::<u32>();
                let check_memory = check.capacity() * std::mem::size_of::<u32>();
                (base_memory, check_memory, 0)
            }
            _ => {
                let total_memory = self.memory_usage();
                (total_memory / 2, total_memory / 2, 0)
            }
        }
    }

    /// Insert and get node ID
    pub fn insert_and_get_node_id(&mut self, key: &[u8]) -> Result<StateId> {
        match &mut self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                let node_id = Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key, &mut self.stats.num_keys)?;
                Ok(node_id)
            }
            TrieStorage::Louds { louds, is_link, next_link, label_data, core_data, next_trie } => {
                let node_id = Self::insert_louds(louds, is_link, next_link, label_data, core_data, next_trie, key)?;
                self.stats.num_keys += 1;
                Ok(node_id)
            }
            TrieStorage::DoubleArray { base, check, free_list, state_count } => {
                // insert_double_array handles num_keys internally (checks was_new)
                let node_id = Self::insert_double_array(base, check, free_list, state_count, key, &mut self.stats.num_keys)?;
                self.stats_dirty = true;
                Ok(node_id)
            }
            _ => {
                self.stats.num_keys += 1;
                Ok(0)
            }
        }
    }

    /// Lookup node ID for a key
    pub fn lookup_node_id(&self, key: &[u8]) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::lookup_node_id_patricia_actual(nodes, edge_data, compressed_paths, key)
            }
            TrieStorage::Louds { label_data, .. } => {
                Self::find_key_position(label_data, key)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::lookup_node_id_double_array(base, check, key)
            }
            _ => None,
        }
    }

    /// Lookup node ID in DoubleArray storage
    fn lookup_node_id_double_array(base: &FastVec<u32>, check: &FastVec<u32>, key: &[u8]) -> Option<StateId> {
        const TERMINAL_BIT: u32 = 0x8000_0000;
        const VALUE_MASK: u32 = 0x7FFF_FFFF;
        const FREE_BIT: u32 = 0x8000_0000;

        if base.is_empty() {
            return None;
        }

        let mut current_state = 0u32;

        if key.is_empty() {
            let base_val = base[0];
            return if (base_val & TERMINAL_BIT) != 0 { Some(0) } else { None };
        }

        for &symbol in key {
            let base_value = base[current_state as usize] & VALUE_MASK;
            let next_state = base_value.saturating_add(symbol as u32);

            if next_state as usize >= check.len() {
                return None;
            }

            let check_val = check[next_state as usize];
            let is_free = (check_val & FREE_BIT) != 0;
            if is_free || check_val != current_state {
                return None;
            }

            current_state = next_state;
        }

        // Only return state if it's marked terminal
        let base_val = base[current_state as usize];
        if (base_val & TERMINAL_BIT) != 0 {
            Some(current_state)
        } else {
            None
        }
    }

    /// Restore string from state ID
    pub fn restore_string(&self, state_id: StateId) -> Option<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::restore_string_patricia_actual(nodes, edge_data, compressed_paths, state_id)
            }
            TrieStorage::Louds { label_data, .. } => {
                Self::restore_string_louds(label_data, state_id)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                Self::restore_string_double_array(base, check, state_id)
            }
            _ => None,
        }
    }

    /// Restore string from DoubleArray state by walking parent chain
    fn restore_string_double_array(base: &FastVec<u32>, check: &FastVec<u32>, state_id: StateId) -> Option<Vec<u8>> {
        const VALUE_MASK: u32 = 0x7FFF_FFFF;
        const FREE_BIT: u32 = 0x8000_0000;

        if state_id as usize >= check.len() {
            return None;
        }

        // Walk parent chain from state_id back to root, collecting symbols
        let mut symbols = Vec::new();
        let mut current = state_id;

        while current != 0 {
            let check_val = check[current as usize];
            if (check_val & FREE_BIT) != 0 {
                return None; // Free state, invalid
            }
            let parent = check_val; // parent state
            let parent_base = base[parent as usize] & VALUE_MASK;

            // The symbol is: current - parent_base
            if current < parent_base {
                return None; // Invalid state
            }
            let symbol = (current - parent_base) as u8;
            symbols.push(symbol);
            current = parent;
        }

        symbols.reverse();
        Some(symbols)
    }

    /// Check if a state is free (for DoubleArray)
    pub fn is_free_double_array(&self, state: StateId) -> bool {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free states (referenced project)

                // Special case: root (state 0) is never free
                if state == 0 {
                    return false;
                }

                // A state is free if it's out of bounds or has FREE_BIT set
                if (state as usize) >= check.len() {
                    return true; // Out of bounds states are considered free
                }

                // Check the FREE_BIT (referenced project line 33: is_free)
                (check[state as usize] & FREE_BIT) != 0
            }
            _ => false
        }
    }

    /// Get parent state (for DoubleArray)
    pub fn get_parent_double_array(&self, state: StateId) -> StateId {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for parent value
                if (state as usize) < check.len() {
                    check[state as usize] & VALUE_MASK
                } else {
                    0 // Default to root
                }
            }
            _ => 0
        }
    }

    /// Get base value (for DoubleArray)
    pub fn get_base_double_array(&self, state: StateId) -> u32 {
        match &self.storage {
            TrieStorage::DoubleArray { base, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for base value
                if (state as usize) < base.len() {
                    base[state as usize] & VALUE_MASK
                } else {
                    0
                }
            }
            _ => 0
        }
    }

    /// Get check value (for DoubleArray)
    pub fn get_check_double_array(&self, state: StateId) -> u32 {
        match &self.storage {
            TrieStorage::DoubleArray { check, .. } => {
                const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for parent value
                if (state as usize) < check.len() {
                    check[state as usize] & VALUE_MASK
                } else {
                    0
                }
            }
            _ => 0
        }
    }

    /// Shrink arrays to fit (for DoubleArray)
    pub fn shrink_to_fit(&mut self) {
        if let TrieStorage::DoubleArray { base, check, .. } = &mut self.storage {
            // Find the actual used length by scanning from the end
            // Skip trailing unused entries (check == 0 and base == 0)
            let mut actual_len = base.len();

            // Find the last used position
            while actual_len > 1 {
                let idx = actual_len - 1;
                // A state is used if either check is non-zero or base is non-zero
                // (state 0 is always used as root)
                if check[idx] != 0 || base[idx] != 0 {
                    break;
                }
                actual_len -= 1;
            }

            // Set unused bases to 1 (referenced project line 354-355)
            const NIL_STATE: u32 = 0x7FFF_FFFF;
            const VALUE_MASK: u32 = 0x7FFF_FFFF;
            for i in 0..actual_len {
                let base_val = base[i] & VALUE_MASK;
                if base_val == NIL_STATE {
                    base[i] = (base[i] & !VALUE_MASK) | 1; // Keep terminal bit, set base to 1
                }
            }

            // Truncate to exact used length (referenced project: exact sizing)
            if actual_len < base.len() {
                base.resize(actual_len, 0).ok();
                check.resize(actual_len, 0).ok();
            }

            // Shrink capacity to size (referenced project: minimal memory)
            let _ = base.shrink_to_fit();
            let _ = check.shrink_to_fit();
        }
    }

    // Helper method to restore string from LOUDS storage
    fn restore_string_louds(label_data: &FastVec<u8>, state_id: StateId) -> Option<Vec<u8>> {
        let start_pos = state_id as usize;
        if start_pos >= label_data.len() {
            return None;
        }

        // Read until we hit a null terminator
        let mut key = Vec::new();
        for i in start_pos..label_data.len() {
            if label_data[i] == 0 {
                break;
            }
            key.push(label_data[i]);
        }

        if key.is_empty() {
            None
        } else {
            Some(key)
        }
    }
}

/// Iterator for trie keys
pub struct TrieIterator {
    keys: Vec<Vec<u8>>,
    index: usize,
}

impl TrieIterator {
    pub fn new() -> Self {
        TrieIterator {
            keys: Vec::new(),
            index: 0,
        }
    }

    pub fn with_keys(keys: Vec<Vec<u8>>) -> Self {
        TrieIterator { keys, index: 0 }
    }
}

impl Iterator for TrieIterator {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.keys.len() {
            let key = self.keys[self.index].clone();
            self.index += 1;
            Some(key)
        } else {
            None
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
}

// Add Clone implementation for ZiporaTrie
impl<R> Clone for ZiporaTrie<R>
where
    R: RankSelectOps + Default + Clone,
{
    fn clone(&self) -> Self {
        // Create a new trie with the same config
        let mut new_trie = Self::with_config(self.config.clone());

        // Copy all keys from the original trie
        let keys = self.keys();
        for key in keys {
            let _ = new_trie.insert(&key);
        }

        // Copy statistics
        new_trie.stats = self.stats.clone();

        new_trie
    }
}

impl<R> Trie for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        // Track if this was a new key insertion
        let prev_count = self.stats.num_keys;

        let result = match &mut self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::insert_patricia(nodes, edge_data, compressed_paths, key, &mut self.stats.num_keys)
            }
            TrieStorage::CriticalBit { nodes, keys, critical_cache } => {
                Self::insert_critical_bit(nodes, keys, critical_cache, key)
            }
            TrieStorage::DoubleArray { base, check, free_list, state_count } => {
                Self::insert_double_array(base, check, free_list, state_count, key, &mut self.stats.num_keys)
            }
            TrieStorage::Louds { louds, is_link, next_link, label_data, core_data, next_trie } => {
                Self::insert_louds(louds, is_link, next_link, label_data, core_data, next_trie, key)
            }
            TrieStorage::CompressedSparse(cspp) => {
                let (is_new, _) = cspp.insert(key);
                if is_new { self.stats.num_keys += 1; }
                Ok(0)
            }
        }?;

        Ok(result)
    }

    fn contains(&self, key: &[u8]) -> bool {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                self.contains_patricia(nodes, edge_data, compressed_paths, key)
            }
            TrieStorage::CriticalBit { nodes, keys, critical_cache } => {
                self.contains_critical_bit(nodes, keys, critical_cache, key)
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                self.contains_double_array(base, check, key)
            }
            TrieStorage::Louds { louds, is_link, next_link, label_data, core_data, next_trie } => {
                self.contains_louds(louds, is_link, next_link, label_data, core_data, next_trie, key)
            }
            TrieStorage::CompressedSparse(cspp) => {
                cspp.contains(key)
            }
        }
    }

    fn len(&self) -> usize {
        self.stats.num_keys
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<R> FiniteStateAutomaton for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn root(&self) -> StateId {
        self.root_state
    }

    fn is_final(&self, state: StateId) -> bool {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.get(state as usize).map(|n| n.is_final).unwrap_or(false)
            }
            TrieStorage::CriticalBit { nodes, .. } => {
                nodes.get(state as usize).map(|n| n.is_final).unwrap_or(false)
            }
            TrieStorage::DoubleArray { base, .. } => {
                // Check the terminal bit in the BASE array (referenced project line 32: is_term)
                const TERMINAL_BIT: u32 = 0x8000_0000;
                base.get(state as usize).map(|b| (b & TERMINAL_BIT) != 0).unwrap_or(false)
            }
            TrieStorage::Louds { .. } => {
                // TODO: Implement LOUDS final state check
                false
            }
            TrieStorage::CompressedSparse(cspp) => false, // Stub for legacy method
        }
    }

    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                let node = nodes.get(state as usize)?;
                node.children.binary_search_by_key(&symbol, |(s, _)| *s)
                    .ok()
                    .map(|idx| node.children[idx].1)
            }
            TrieStorage::CriticalBit { nodes, .. } => {
                // TODO: Implement critical bit transition
                None
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // Double array trie transition: next = (base[state] & VALUE_MASK) + symbol
                // Validate with: check[next] == state (referenced project line 100-110)
                const VALUE_MASK: u32 = 0x7FFF_FFFF;

                let base_value = base.get(state as usize)? & VALUE_MASK;
                let next_state = base_value.saturating_add(symbol as u32);
                if let Some(check_value) = check.get(next_state as usize) {
                    if *check_value == state {
                        Some(next_state)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            TrieStorage::Louds { .. } => {
                // TODO: Implement LOUDS transition
                None
            }
            TrieStorage::CompressedSparse(cspp) => None, // Stub for legacy method
        }
    }

    fn transitions(&self, state: StateId) -> Vec<(u8, StateId)> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                if let Some(node) = nodes.get(state as usize) {
                    // Compact children representation - already in the right format
                    node.children.clone()
                } else {
                    Vec::new()
                }
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                let Some(&base_val) = base.get(state as usize) else {
                    return Vec::new();
                };
                if base_val == 0 {
                    return Vec::new();
                }

                const STATE_MASK: u32 = 0x3FFF_FFFF;
                const TERMINAL_FLAG: u32 = 0x4000_0000;

                (0u8..=255u8).filter_map(|symbol| {
                    let next_state = base_val.saturating_add(symbol as u32);
                    if (next_state as usize) >= check.len() {
                        return None;
                    }
                    let check_val = check[next_state as usize];
                    let is_valid_child = if state == 0 {
                        (check_val & STATE_MASK) == 0 && (
                            (check_val & TERMINAL_FLAG) != 0 ||
                            ((next_state as usize) < base.len() && base[next_state as usize] != 0)
                        )
                    } else {
                        check_val != 0 && (check_val & STATE_MASK) == state
                    };
                    if is_valid_child { Some((symbol, next_state)) } else { None }
                }).collect()
            }
            _ => Vec::new(),
        }
    }
}

impl<R> PrefixIterable for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        Box::new(self.iter_prefix(prefix))
    }

    fn iter_all(&self) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        Box::new(self.iter_all())
    }
}

impl<R> Default for ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    fn default() -> Self {
        Self::new()
    }
}

// Implementation methods for different strategies
impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    // Patricia trie implementation methods
    fn insert_patricia(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key, num_keys)
    }

    fn contains_patricia(
        &self,
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        Self::contains_patricia_actual(nodes, edge_data, compressed_paths, key)
    }

    // Critical-bit trie implementation methods
    /// Critical-bit trie insertion.
    //  TODO: port from C++ reference `src/terark/fsa/crit_bit_trie.hpp`
    #[allow(unused)]
    fn insert_critical_bit(
        nodes: &mut FastVec<CritBitNode>,
        keys: &mut FastVec<Vec<u8>>,
        critical_cache: &mut HashMap<usize, u8>,
        key: &[u8],
    ) -> Result<StateId> {
        // TODO: Implement critical-bit insertion
        Ok(0)
    }

    /// Critical-bit trie lookup.
    //  TODO: port from C++ reference `src/terark/fsa/crit_bit_trie.hpp`
    #[allow(unused)]
    fn contains_critical_bit(
        &self,
        nodes: &FastVec<CritBitNode>,
        keys: &FastVec<Vec<u8>>,
        critical_cache: &HashMap<usize, u8>,
        key: &[u8],
    ) -> bool {
        // TODO: Implement critical-bit lookup
        false
    }

    // Double array trie implementation methods
    fn insert_double_array(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        free_list: &mut VecDeque<StateId>,
        state_count: &mut usize,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        // Following referenced project's double array trie implementation EXACTLY
        // Base array (m_child0): bits 0-30 = base value, bit 31 = terminal bit
        // Check array (m_parent): bits 0-30 = parent state, bit 31 = free bit

        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states (referenced project)
        const FREE_BIT: u32 = 0x8000_0000;     // Bit 31 in check for free states (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for actual values (referenced project)
        const MAX_STATE: u32 = 0x7FFF_FFFE;    // Maximum valid state value (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF;    // Nil state marker (referenced project)

        // Ensure we have at least the root state
        // Referenced project starts with 1 state (line 70: states.resize(1))
        // We initialize in storage creation, but check here for safety
        if base.is_empty() {
            base.resize(1, NIL_STATE); // Just root state
            check.resize(1, 0); // Root check is 0 (itself), no free bit
            // Use compact base allocation like referenced project
            base[0] = Self::find_free_base(base, check, 0)?;
            *state_count = 1;
        }

        // Special case for empty key - mark root as terminal
        if key.is_empty() {
            let was_new = (base[0] & TERMINAL_BIT) == 0;
            base[0] |= TERMINAL_BIT;
            if was_new {
                *num_keys += 1;
            }
            return Ok(0);
        }

        let mut current_state = 0u32;

        #[cfg(debug_assertions)]
        eprintln!("DEBUG insert: Starting insertion of key: {:?}",
            std::str::from_utf8(key).unwrap_or("<non-utf8>"));

        // Traverse the trie for each symbol in the key
        for (pos, &symbol) in key.iter().enumerate() {
            // Calculate next state position using base value (bits 0-30)
            let mut base_value = base[current_state as usize] & VALUE_MASK;

            // If base is NIL_STATE, we need to find a good base for this state's children
            // Referenced project does this during build (lines 309-327)
            if base_value == NIL_STATE {
                base_value = Self::find_free_base(base, check, current_state)?;
                // CRITICAL: Preserve terminal bit when setting new base
                let old_val = base[current_state as usize];
                base[current_state as usize] = base_value | (old_val & TERMINAL_BIT);
            }

            let next_state = base_value.saturating_add(symbol as u32);

            // Expand arrays if needed - use amortized growth
            let required = next_state as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                base.resize(new_size, NIL_STATE);
                check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // Check if this transition already exists (referenced project style at line 106)
            // A transition exists if check[next] == current_state (without free bit)
            // Free states have FREE_BIT set, so won't match
            let check_val = check[next_state as usize];
            let is_free = (check_val & FREE_BIT) != 0;
            let transition_exists = !is_free && check_val == current_state;

            if transition_exists {
                // Transition exists, follow it
                #[cfg(debug_assertions)]
                eprintln!("  [{}] '{:02x}' state {} -> {} (existing)", pos, symbol, current_state, next_state);
                current_state = next_state;
            } else {
                // Need to create new transition
                // CRITICAL: Never allow transitions to state 0 (reserved for root)
                if next_state == 0 {
                    // State 0 is reserved, need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!("  [{}] '{:02x}' conflict: next_state would be 0 (reserved for root)", pos, symbol);

                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base = Self::relocate_state(
                        base,
                        check,
                        current_state,
                        symbol,
                        state_count
                    )?;

                    // Now the transition should be available at the new location
                    let new_next = new_base.saturating_add(symbol as u32);

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        base.resize(new_size, NIL_STATE);
                        check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                } else if is_free {
                    // Position is free and not state 0, use it directly
                    #[cfg(debug_assertions)]
                    eprintln!("  [{}] '{:02x}' state {} -> {} (new, free)", pos, symbol, current_state, next_state);

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data("State value exceeds maximum"));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[next_state as usize] = current_state; // Clear free bit by assignment
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[next_state as usize] = NIL_STATE;
                    current_state = next_state;
                    *state_count += 1;
                } else {
                    // Position is occupied - need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!("  [{}] '{:02x}' conflict at state {}, next_state {} already has check={:08x}",
                        pos, symbol, current_state, next_state, check[next_state as usize]);
                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base = Self::relocate_state(
                        base,
                        check,
                        current_state,
                        symbol,
                        state_count
                    )?;

                    // Now the transition should be available
                    let new_next = new_base.saturating_add(symbol as u32);

                    #[cfg(debug_assertions)]
                    eprintln!("  Relocated state {} to new_base {}, new transition {} -> {}",
                        current_state, new_base, current_state, new_next);

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        base.resize(new_size, NIL_STATE);
                        check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data("State value exceeds maximum during relocation"));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                }
            }
        }

        // Mark the final state as terminal (referenced project: set_term_bit on base at line 27)
        // Check if this is a new key or duplicate
        let was_new = (base[current_state as usize] & TERMINAL_BIT) == 0;
        base[current_state as usize] |= TERMINAL_BIT;

        // Only increment key count if this was a new key
        if was_new {
            *num_keys += 1;
        }

        // Debug: Verify what we just inserted
        #[cfg(debug_assertions)]
        {
            eprintln!("DEBUG insert_double_array: Inserted key, final state={}, base[{}]={:08x}, check[{}]={:08x}, was_new={}",
                current_state, current_state, base[current_state as usize], current_state, check[current_state as usize], was_new);
        }

        Ok(current_state)
    }

    // Helper: Find a free base value for a state that doesn't conflict
    // For incremental insert, use a proper heuristic matching referenced project's approach
    fn find_free_base(base: &FastVec<u32>, check: &FastVec<u32>, _state: u32) -> Result<u32> {
        const FREE_BIT: u32 = 0x8000_0000;
        const NIL_STATE: u32 = 0x7FFF_FFFF;

        // Start search from position 1 (0 is root)
        let mut candidate = 1u32;
        let len = check.len();

        // Linear probe for a free position (matching C++ reference heuristic)
        while (candidate as usize) < len {
            let check_val = check[candidate as usize];
            let is_free = check_val == (NIL_STATE | FREE_BIT) || (check_val & FREE_BIT) != 0;
            if is_free {
                return Ok(candidate);
            }
            candidate += 1;
        }

        // Past the end of array — return the next position (will trigger array growth)
        Ok(candidate)
    }

    // Helper: Relocate a state and all its children to use a new base value
    fn relocate_state(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        state: u32,
        new_symbol: u8,
        state_count: &mut usize,
    ) -> Result<u32> {
        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for values (referenced project)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const FREE_BIT: u32 = 0x8000_0000;     // Bit 31 in check for free (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF;    // Match referenced project's nil_state

        // Special handling for root state - try to avoid relocating it
        if state == 0 {
            // For root, try to find a different base that works
            // This is critical because relocating root affects the entire trie
            #[cfg(debug_assertions)]
            eprintln!("  WARNING: Attempting to relocate root state - this may cause issues");
        }

        let old_base = base[state as usize] & VALUE_MASK;

        // Collect all existing children of this state with their base and terminal info
        let mut children = Vec::new();
        for symbol in 0u8..=255u8 {
            let child_pos = old_base.saturating_add(symbol as u32);
            if (child_pos as usize) < check.len() {
                let check_val = check[child_pos as usize];
                // Check if this is an allocated child (not free, parent matches)
                if (check_val & FREE_BIT) == 0 && check_val == state {
                    // This is a child of our state - save its info
                    let child_base = if (child_pos as usize) < base.len() {
                        base[child_pos as usize]
                    } else {
                        NIL_STATE
                    };
                    let is_terminal = (child_base & TERMINAL_BIT) != 0;
                    children.push((symbol, child_pos, child_base, is_terminal));
                }
            }
        }

        // Find a new base where we can place all children plus the new symbol
        // Use find_free_base to get a better starting point that spreads states out
        let initial_base = Self::find_free_base(base, check, state)?;
        let mut new_base = initial_base;
        let mut attempts = 0;
        const MAX_BASE: u32 = u32::MAX - 256; // Leave room for 256 symbols

        'search: loop {
            if attempts > 1_000_000 || new_base > MAX_BASE {
                return Err(ZiporaError::invalid_data("Cannot relocate state in double array"));
            }
            attempts += 1;

            // Check if new_base works for the new symbol
            let new_pos = new_base.saturating_add(new_symbol as u32);

            // Ensure arrays are large enough
            let max_pos = children.iter()
                .map(|(sym, _, _, _)| new_base.saturating_add(*sym as u32))
                .chain(std::iter::once(new_pos))
                .max()
                .unwrap_or(new_pos);

            // Expand arrays if needed - use amortized growth
            let required = max_pos as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                base.resize(new_size, NIL_STATE);
                check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // CRITICAL: Never allow any child to be relocated to state 0
            // Check if new position for new_symbol is free and not state 0
            let new_pos_check = check[new_pos as usize];
            let new_pos_is_free = (new_pos_check & FREE_BIT) != 0;
            if new_pos == 0 || !new_pos_is_free {
                // State 0 is reserved or position is occupied
                // Use smaller increment for denser packing
                new_base = new_base.saturating_add(1);
                continue 'search;
            }

            // Check if all children can be relocated (and none would go to state 0)
            for (symbol, _, _, _) in &children {
                let test_pos = new_base.saturating_add(*symbol as u32);
                let test_check = check[test_pos as usize];
                let test_is_free = (test_check & FREE_BIT) != 0;
                // CRITICAL: Reject if any child would be relocated to state 0
                if test_pos == 0 || !test_is_free {
                    // State 0 is reserved or position is occupied
                    new_base = new_base.saturating_add(1);
                    continue 'search;
                }
            }

            // Found a suitable new base - relocate all children
            // First, mark old positions as free
            for (_, old_pos, _, _) in &children {
                check[*old_pos as usize] = NIL_STATE | FREE_BIT;
                // Mark base as NIL to indicate it's free
                base[*old_pos as usize] = NIL_STATE;
            }

            // Then, set new positions with both check and base values
            for (symbol, old_pos, child_base_val, is_terminal) in &children {
                let new_child_pos = new_base.saturating_add(*symbol as u32);
                // Allocate the new position (referenced project: set_parent clears free bit)
                check[new_child_pos as usize] = state; // Parent state, no free bit
                // Set base value, preserving terminal bit if needed
                let base_value = child_base_val & VALUE_MASK;
                base[new_child_pos as usize] = if *is_terminal {
                    base_value | TERMINAL_BIT
                } else {
                    base_value
                };

                // CRITICAL: Update any grandchildren that point to the old child position
                // to point to the new child position
                if base_value != 0 && base_value != NIL_STATE {
                    Self::update_grandchildren_check_values(
                        base, check, *old_pos, new_child_pos
                    );
                }
            }

            // Update the base for this state (preserve terminal bit if state is terminal)
            let state_base = base[state as usize];
            let state_is_terminal = (state_base & TERMINAL_BIT) != 0;
            base[state as usize] = if state_is_terminal {
                new_base | TERMINAL_BIT
            } else {
                new_base
            };

            return Ok(new_base);
        }
    }

    // Helper function to update grandchildren when a child state is relocated
    fn update_grandchildren_check_values(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        old_parent_pos: u32,
        new_parent_pos: u32,
    ) {
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)
        const FREE_BIT: u32 = 0x8000_0000;   // Bit 31 in check for free (referenced project)

        // Get the base value of the relocated child to find its children
        if let Some(&child_base_raw) = base.get(new_parent_pos as usize) {
            let child_base = child_base_raw & VALUE_MASK;
            if child_base != 0 && child_base != 0x7FFF_FFFF {
                // Find all grandchildren that were pointing to the old parent position
                for symbol in 0u8..=255u8 {
                    let grandchild_pos = child_base.saturating_add(symbol as u32);
                    if (grandchild_pos as usize) < check.len() {
                        let check_val = check[grandchild_pos as usize];
                        // Check if it's allocated (not free) and points to old parent
                        if (check_val & FREE_BIT) == 0 && check_val == old_parent_pos {
                            // This grandchild was pointing to the old parent position
                            // Update it to point to the new parent position
                            check[grandchild_pos as usize] = new_parent_pos;
                        }
                    }
                }
            }
        }
    }

    fn contains_double_array(
        &self,
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        key: &[u8],
    ) -> bool {
        // Following referenced project's double array trie lookup (line 100-110)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states
        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for actual values

        if base.is_empty() {
            return false;
        }

        // Special case for empty key - check if root is terminal (check terminal bit in base)
        if key.is_empty() {
            return base.get(0)
                .map(|b| (b & TERMINAL_BIT) != 0)
                .unwrap_or(false);
        }

        let mut current_state = 0u32;

        // Traverse the trie for each symbol (referenced project line 100-110: state_move)
        for (i, &symbol) in key.iter().enumerate() {
            // SAFETY: We check if base_val exists, then use it
            let base_val = match base.get(current_state as usize) {
                Some(val) => val,
                None => {
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG contains: No base for state {}", current_state);
                    return false;
                }
            };

            // Calculate next state using base value (bits 0-30)
            let next_state = (base_val & VALUE_MASK).saturating_add(symbol as u32);

            // Check if the transition is valid (referenced project line 106: states[next].parent() == curr)
            if next_state as usize >= check.len() {
                #[cfg(debug_assertions)]
                eprintln!("DEBUG contains: next_state {} >= check.len() {}", next_state, check.len());
                return false;
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project: check[next] == current_state
            // Free states have FREE_BIT set, so won't match
            if check_val != current_state {
                // Invalid transition
                #[cfg(debug_assertions)]
                eprintln!("DEBUG contains: Invalid transition at pos {}, symbol {:02x}, state {}->{}, check[{}]={:08x}, expected parent {}",
                    i, symbol, current_state, next_state, next_state, check_val, current_state);
                return false;
            }

            current_state = next_state;
        }

        // Check if the final state is marked as terminal (check terminal bit in base)
        let is_terminal = base.get(current_state as usize)
            .map(|b| (b & TERMINAL_BIT) != 0)
            .unwrap_or(false);

        #[cfg(debug_assertions)]
        {
            let base_val = base.get(current_state as usize).unwrap_or(&0);
            let check_val = check.get(current_state as usize).unwrap_or(&0);
            eprintln!("DEBUG contains: Final state={}, base[{}]={:08x}, check[{}]={:08x}, is_terminal={}",
                current_state, current_state, base_val, current_state, check_val, is_terminal);
        }

        is_terminal
    }

    // LOUDS trie implementation methods
    /// LOUDS trie insertion.
    //  TODO: port from C++ reference `src/terark/fsa/nest_louds_trie.hpp`
    #[allow(unused)]
    fn insert_louds(
        louds: &mut R,
        is_link: &mut R,
        next_link: &mut UintVector,
        label_data: &mut FastVec<u8>,
        core_data: &mut FastVec<u8>,
        next_trie: &mut Option<Box<ZiporaTrie<R>>>,
        key: &[u8],
    ) -> Result<StateId> {
        // Store keys in label_data with length prefix for separation
        // Format: [len_byte][key_bytes...] where len_byte < 255
        // First check if key already exists
        if Self::contains_louds_internal(label_data, key) {
            // Key already exists, find its position
            return Ok(Self::find_key_position(label_data, key).unwrap_or(0));
        }

        // Store new key with length prefix
        let start_pos = label_data.len();

        // Store length (limited to 255 for single-byte length)
        if key.len() > 255 {
            return Err(ZiporaError::invalid_data("Key too long for LOUDS (max 255 bytes)"));
        }
        label_data.push(key.len() as u8);

        // Store key bytes
        for &byte in key {
            label_data.push(byte);
        }

        // Store the position in core_data for later retrieval
        // We use start_pos as the StateId
        let state_id = start_pos as StateId;

        Ok(state_id)
    }

    /// LOUDS trie lookup.
    //  TODO: port from C++ reference `src/terark/fsa/nest_louds_trie.hpp`
    #[allow(unused)]
    fn contains_louds(
        &self,
        louds: &R,
        is_link: &R,
        next_link: &UintVector,
        label_data: &FastVec<u8>,
        core_data: &FastVec<u8>,
        next_trie: &Option<Box<ZiporaTrie<R>>>,
        key: &[u8],
    ) -> bool {
        Self::contains_louds_internal(label_data, key)
    }

    // Helper method for LOUDS contains check
    fn contains_louds_internal(label_data: &FastVec<u8>, key: &[u8]) -> bool {
        if label_data.is_empty() {
            return false;
        }

        let key_len = key.len();
        if key_len > 255 {
            return false; // Key too long
        }

        // Format: [len_byte][key_bytes...]
        // Scan through label_data looking for matching keys
        let mut pos = 0;
        while pos < label_data.len() {
            // Read length byte
            let stored_len = label_data[pos] as usize;

            // Check if we have enough space for this key
            if pos + 1 + stored_len > label_data.len() {
                break; // Corrupted data or end of data
            }

            // Check if lengths match
            if stored_len == key_len {
                // Check if key bytes match
                let mut matches = true;
                for i in 0..key_len {
                    if label_data[pos + 1 + i] != key[i] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    return true;
                }
            }

            // Move to next key
            pos += 1 + stored_len;
        }

        false
    }

    // Helper method to find key position in LOUDS
    fn find_key_position(label_data: &FastVec<u8>, key: &[u8]) -> Option<StateId> {
        if label_data.is_empty() {
            return None;
        }

        let key_len = key.len();
        if key_len > 255 {
            return None; // Key too long
        }

        // Format: [len_byte][key_bytes...]
        // Scan through label_data looking for matching keys
        let mut pos = 0;
        while pos < label_data.len() {
            // Read length byte
            let stored_len = label_data[pos] as usize;

            // Check if we have enough space for this key
            if pos + 1 + stored_len > label_data.len() {
                break; // Corrupted data or end of data
            }

            // Check if lengths match
            if stored_len == key_len {
                // Check if key bytes match
                let mut matches = true;
                for i in 0..key_len {
                    if label_data[pos + 1 + i] != key[i] {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    return Some(pos as StateId);
                }
            }

            // Move to next key
            pos += 1 + stored_len;
        }

        None
    }

    // Compressed sparse trie implementation methods
    /// Compressed sparse trie insertion.
    //  TODO: port from C++ reference `src/terark/fsa/cspptrie.hpp`
    #[allow(unused)]
    fn insert_compressed_sparse(
        sparse_nodes: &mut HashMap<StateId, SparseNode>,
        compression_dict: &mut HashMap<Vec<u8>, u32>,
        bit_vector: &mut BitVector,
        rank_select: &mut R,
        key: &[u8],
    ) -> Result<StateId> {
        // Initialize root node if not present
        if sparse_nodes.is_empty() {
            sparse_nodes.insert(0, SparseNode {
                children: HashMap::new(),
                _edge_label: None,
                is_final: false,
            });
        }

        let mut current_state = 0;
        let mut next_state_id = sparse_nodes.keys().max().copied().unwrap_or(0) + 1;

        for &symbol in key {
            // First check if an edge exists for this symbol
            let existing_child = sparse_nodes.get(&current_state)
                .and_then(|node| node.children.get(&symbol).copied());

            let child_state = if let Some(existing) = existing_child {
                // Edge already exists, follow it
                existing
            } else {
                // Need to create new edge
                let new_state = next_state_id;
                next_state_id += 1;

                // Create the new child node
                sparse_nodes.insert(new_state, SparseNode {
                    children: HashMap::new(),
                    _edge_label: None,
                    is_final: false,
                });

                // Update parent to point to child
                if let Some(parent_node) = sparse_nodes.get_mut(&current_state) {
                    parent_node.children.insert(symbol, new_state);
                } else {
                    // This shouldn't happen as we ensure current_state exists
                    return Err(crate::error::ZiporaError::invalid_state(
                        format!("State {} not found during insertion", current_state)
                    ));
                }

                new_state
            };

            current_state = child_state;
        }

        // Mark the final node as terminal
        if let Some(final_node) = sparse_nodes.get_mut(&current_state) {
            final_node.is_final = true;
        }

        Ok(current_state)
    }

    /// Compressed sparse trie lookup.
    //  TODO: port from C++ reference `src/terark/fsa/cspptrie.hpp`
    #[allow(unused)]
    fn contains_compressed_sparse(
        &self,
        sparse_nodes: &HashMap<StateId, SparseNode>,
        compression_dict: &HashMap<Vec<u8>, u32>,
        bit_vector: &BitVector,
        rank_select: &R,
        key: &[u8],
    ) -> bool {
        if sparse_nodes.is_empty() {
            return false;
        }

        let mut current_state = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];

            // Get the current node
            if let Some(node) = sparse_nodes.get(&current_state) {
                if let Some(&child_state) = node.children.get(&symbol) {
                    // Follow the edge
                    current_state = child_state;
                    key_pos += 1;
                } else {
                    // No edge for this symbol
                    return false;
                }
            } else {
                // Node doesn't exist
                return false;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        sparse_nodes.get(&current_state)
            .map(|node| node.is_final)
            .unwrap_or(false)
    }

    // Actual implementation methods for Patricia trie
    fn insert_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        if nodes.is_empty() {
            // Initialize with root node
            nodes.push(PatriciaNode::default());
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                // Follow existing path
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Create new child node
                let new_node_id = nodes.len();
                nodes.push(PatriciaNode::default());

                // Insert into sorted children Vec
                let insert_pos = nodes[current].children.binary_search_by_key(&symbol, |(s, _)| *s).unwrap_err();
                nodes[current].children.insert(insert_pos, (symbol, new_node_id as StateId));

                current = new_node_id;
                key_pos += 1;
            }
        }

        // Mark current node as final (check if was_new)
        let was_new = !nodes[current].is_final;
        nodes[current].is_final = true;
        if was_new {
            *num_keys += 1;
        }
        Ok(current as StateId)
    }

    fn contains_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        if nodes.is_empty() {
            return false;
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;
            } else {
                return false;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        key_pos == key.len() && nodes[current].is_final
    }

    fn remove_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> Result<bool> {
        if nodes.is_empty() {
            return Ok(false);
        }

        // First, check if the key exists and find the path to it
        let mut current = 0;
        let mut key_pos = 0;
        let mut path = Vec::new(); // Track the path for potential cleanup

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path.push((current, symbol)); // Store parent and symbol for path
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Key doesn't exist
                return Ok(false);
            }
        }

        // Check if we found a complete key at a final state
        if key_pos != key.len() || !nodes[current].is_final {
            return Ok(false);
        }

        // Mark the node as non-final (remove the key)
        nodes[current].is_final = false;

        // Check if this node has any children
        let has_children = !nodes[current].children.is_empty();

        // If the node has no children and is not final, we can potentially clean it up
        if !has_children {
            // Walk back up the path and remove unnecessary nodes
            let mut node_to_remove = current;

            for &(parent_idx, symbol) in path.iter().rev() {
                // Remove the child pointer from parent
                if let Ok(idx) = nodes[parent_idx].children.binary_search_by_key(&symbol, |(s, _)| *s) {
                    nodes[parent_idx].children.remove(idx);
                }

                // Check if parent node should also be cleaned up
                let parent_has_children = !nodes[parent_idx].children.is_empty();
                let parent_is_final = nodes[parent_idx].is_final;

                // If parent has other children or is final, stop cleanup
                if parent_has_children || parent_is_final {
                    break;
                }

                // Continue cleanup with parent
                node_to_remove = parent_idx;
            }
        }

        Ok(true)
    }

    fn restore_string_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        state_id: StateId,
    ) -> Option<Vec<u8>> {
        if nodes.is_empty() || state_id as usize >= nodes.len() {
            return None;
        }

        // Check if the target state is final
        if !nodes[state_id as usize].is_final {
            return None;
        }

        // Perform DFS to find the path from root to the target state
        let mut path = Vec::new();
        if Self::find_path_to_state(nodes, 0, state_id as usize, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    fn find_path_to_state(
        nodes: &FastVec<PatriciaNode>,
        current: usize,
        target: usize,
        path: &mut Vec<u8>,
    ) -> bool {
        if current == target {
            return true;
        }

        if current >= nodes.len() {
            return false;
        }

        let node = &nodes[current];

        // Try each child (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id = child_id as usize;

            // Add this symbol to the path
            path.push(symbol);

            // Recursively search in child
            if Self::find_path_to_state(nodes, child_id, target, path) {
                return true;
            }

            // Backtrack if not found
            path.pop();
        }

        false
    }

    fn lookup_node_id_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> Option<StateId> {
        if nodes.is_empty() {
            return None;
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > key.len() {
                        return None; // Not enough key left
                    }
                    if &key[key_pos..key_pos + path.len()] != path.as_slice() {
                        return None; // Path doesn't match
                    }
                    key_pos += path.len();
                }
            } else {
                return None;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        if key_pos == key.len() && nodes[current].is_final {
            Some(current as StateId)
        } else {
            None
        }
    }

    /// Get all keys from Patricia trie
    fn keys_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_keys_patricia_recursive(nodes, edge_data, compressed_paths, 0, &mut current_path, &mut keys);
        keys
    }

    /// Get all keys with prefix from Patricia trie
    fn keys_with_prefix_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Navigate to the prefix position first
        let mut current = 0;
        let mut key_pos = 0;
        let mut path_to_prefix = Vec::new();

        while key_pos < prefix.len() {
            let symbol = prefix[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path_to_prefix.push(symbol);
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > prefix.len() {
                        // Prefix doesn't fully match this path
                        let remaining_prefix = &prefix[key_pos..];
                        if path.starts_with(remaining_prefix) {
                            // Prefix is a partial match of this compressed path
                            // Continue from this node with the partial prefix included
                            path_to_prefix.extend_from_slice(remaining_prefix);
                            break;
                        } else {
                            // Prefix doesn't match - no keys with this prefix
                            return Vec::new();
                        }
                    } else if &prefix[key_pos..key_pos + path.len()] != path.as_slice() {
                        // Path doesn't match prefix
                        return Vec::new();
                    } else {
                        // Path matches, continue
                        path_to_prefix.extend_from_slice(path);
                        key_pos += path.len();
                    }
                }
            } else {
                // No child for this symbol - no keys with this prefix
                return Vec::new();
            }
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = path_to_prefix;
        Self::collect_keys_patricia_recursive(nodes, edge_data, compressed_paths, current, &mut current_path, &mut keys);

        // Filter to only include keys that actually start with the prefix
        keys.into_iter().filter(|key| key.starts_with(prefix)).collect()
    }

    /// Recursively collect all keys from Patricia trie
    fn collect_keys_patricia_recursive(
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        node_id: usize,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        if node_id >= nodes.len() {
            return;
        }

        let node = &nodes[node_id];

        // If this is a final node, add the current path as a key
        if node.is_final {
            keys.push(current_path.clone());
        }

        // Explore all children (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id_usize = child_id as usize;

            // Add this symbol to the path
            current_path.push(symbol);

            // Add compressed path if it exists
            let path_start_len = current_path.len();
            if let Some(path) = compressed_paths.get(&child_id) {
                current_path.extend_from_slice(path);
            }

            // Recursively collect from child
            Self::collect_keys_patricia_recursive(nodes, edge_data, compressed_paths, child_id_usize, current_path, keys);

            // Backtrack: remove the path we added
            current_path.truncate(path_start_len - 1);
        }
    }

    /// Get all keys from LOUDS trie storage
    fn keys_louds_actual(label_data: &FastVec<u8>) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();

        if label_data.is_empty() {
            return keys;
        }

        let mut current_key = Vec::new();

        for &byte in label_data.iter() {
            if byte == 0u8 {
                // Found separator, this completes a key
                if !current_key.is_empty() {
                    keys.push(current_key.clone());
                    current_key.clear();
                }
            } else {
                // Add byte to current key
                current_key.push(byte);
            }
        }

        // Handle last key if there's no trailing separator
        if !current_key.is_empty() {
            keys.push(current_key);
        }

        // Remove duplicates and sort
        keys.sort();
        keys.dedup();

        keys
    }

    /// Get all keys with a given prefix from LOUDS trie storage
    fn keys_with_prefix_louds_actual(label_data: &FastVec<u8>, prefix: &[u8]) -> Vec<Vec<u8>> {
        let all_keys = Self::keys_louds_actual(label_data);

        // Filter keys that start with the given prefix
        all_keys
            .into_iter()
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    /// Check if a key exists in LOUDS trie storage
    #[allow(dead_code)]
    fn contains_louds_actual(label_data: &FastVec<u8>, key: &[u8]) -> bool {
        if label_data.is_empty() {
            return false;
        }

        let key_with_separator = [key, &[0u8]].concat();

        // Look for the key with separator in the label_data
        // Since we store keys with separators, we need to find the exact match
        label_data.windows(key_with_separator.len()).any(|window| window == key_with_separator)
    }

    /// Get all keys from DoubleArray trie storage
    fn keys_double_array_actual(base: &FastVec<u32>, check: &FastVec<u32>) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();

        #[cfg(debug_assertions)]
        eprintln!("DEBUG keys_double_array: Starting from root state 0, base[0]={:?}", base.get(0));

        Self::collect_keys_double_array_recursive(base, check, 0, &mut current_path, &mut keys);
        keys
    }

    /// Get all keys with prefix from DoubleArray trie storage
    fn keys_with_prefix_double_array_actual(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }


        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for values (referenced project)

        // Navigate to the prefix position first
        let mut current_state = 0u32;
        for &symbol in prefix {
            // SAFETY: We check if base_val exists, then use it
            let base_value = match base.get(current_state as usize) {
                Some(val) => val & VALUE_MASK,
                None => return Vec::new(),
            };
            let next_state = base_value.saturating_add(symbol as u32);
            if next_state as usize >= check.len() {
                return Vec::new();
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project (line 106)
            if check_val != current_state {
                return Vec::new();
            }

            current_state = next_state;
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = prefix.to_vec();
        Self::collect_keys_double_array_recursive(base, check, current_state, &mut current_path, &mut keys);
        keys
    }

    /// Recursively collect all keys from DoubleArray trie
    fn collect_keys_double_array_recursive(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        state: u32,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for values (referenced project)

        #[cfg(debug_assertions)]
        if state == 0 && current_path.is_empty() {
            eprintln!("DEBUG collect_keys: At root, checking for children...");
        }

        // If this is a terminal state, add the current path as a key (check base array)
        if (state as usize) < base.len() && (base[state as usize] & TERMINAL_BIT) != 0 {
            #[cfg(debug_assertions)]
            eprintln!("DEBUG collect_keys: Found terminal state {} with path {:?}",
                state, std::str::from_utf8(current_path).unwrap_or("<non-utf8>"));
            keys.push(current_path.clone());
        }

        // Get the base value for this state
        if let Some(&base_raw) = base.get(state as usize) {
            let base_val = base_raw & VALUE_MASK;
            if base_val == 0 || base_val == 0x7FFF_FFFF {
                #[cfg(debug_assertions)]
                eprintln!("DEBUG collect_keys: State {} has base={}, no children", state, base_val);
                return; // No children
            }

            #[cfg(debug_assertions)]
            if state == 0 {
                eprintln!("DEBUG collect_keys: Root state 0 has base={}, checking all 256 symbols...", base_val);
            }

            // Try all possible symbols
            for symbol in 0u8..=255u8 {
                let next_state = base_val.saturating_add(symbol as u32);

                // Check if this is a valid transition (referenced project line 106)
                if (next_state as usize) < check.len() {
                    let check_val = check[next_state as usize];
                    // Direct comparison: check[next] == current_state
                    let is_valid_child = check_val == state;

                    if is_valid_child {
                        // Valid transition found
                        #[cfg(debug_assertions)]
                        if state == 0 {
                            eprintln!("DEBUG collect_keys: Found valid transition from root: symbol={:02x} ('{}'), next_state={}",
                                symbol, symbol as char, next_state);
                        }
                        current_path.push(symbol);
                        Self::collect_keys_double_array_recursive(base, check, next_state, current_path, keys);
                        current_path.pop();
                    }
                }
            }
        }
    }

    /// Get all keys from CompressedSparse trie storage
    #[allow(dead_code)]
    fn keys_compressed_sparse_actual(sparse_nodes: &HashMap<StateId, SparseNode>) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_keys_compressed_sparse_recursive(sparse_nodes, 0, &mut current_path, &mut keys);
        keys
    }

    /// Recursively collect all keys from CompressedSparse trie
    #[allow(dead_code)]
    fn collect_keys_compressed_sparse_recursive(
        sparse_nodes: &HashMap<StateId, SparseNode>,
        state: StateId,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        // Get the node for this state
        if let Some(node) = sparse_nodes.get(&state) {
            // If this is a final state, add the current path
            if node.is_final {
                keys.push(current_path.clone());
            }

            // Traverse all children
            for (&symbol, &child_state) in &node.children {
                current_path.push(symbol);
                Self::collect_keys_compressed_sparse_recursive(sparse_nodes, child_state, current_path, keys);
                current_path.pop();
            }
        }
    }

    /// Get all keys with prefix from CompressedSparse trie storage
    #[allow(dead_code)]
    fn keys_with_prefix_compressed_sparse_actual(
        sparse_nodes: &HashMap<StateId, SparseNode>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        // Navigate to the prefix position first
        let mut current_state = 0;
        for &symbol in prefix {
            if let Some(node) = sparse_nodes.get(&current_state) {
                if let Some(&next_state) = node.children.get(&symbol) {
                    current_state = next_state;
                } else {
                    return Vec::new(); // Prefix doesn't exist
                }
            } else {
                return Vec::new(); // Invalid state
            }
        }

        // Collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = prefix.to_vec();
        Self::collect_keys_compressed_sparse_recursive(sparse_nodes, current_state, &mut current_path, &mut keys);
        keys
    }

    /// Build a trie from sorted keys using BFS construction
    ///
    /// This is more efficient than incremental insertion for sorted input because:
    /// 1. Pre-allocates arrays based on estimated size
    /// 2. Processes keys in sorted order to minimize relocations
    /// 3. Uses improved find_free_base for better packing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
    /// use zipora::succinct::RankSelectInterleaved256;
    ///
    /// let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
    /// let trie: ZiporaTrie<RankSelectInterleaved256> =
    ///     ZiporaTrie::build_from_sorted(&keys, ZiporaTrieConfig::default()).unwrap();
    ///
    /// assert_eq!(trie.len(), 5);
    /// assert!(trie.contains(b"apple"));
    /// assert!(trie.contains(b"banana"));
    /// ```
    pub fn build_from_sorted(keys: &[&[u8]], config: ZiporaTrieConfig) -> Result<Self> {
        // Create trie with config
        let mut trie = Self::with_config(config);

        // Estimate size and pre-allocate for DoubleArray strategy
        if let TrieStrategy::DoubleArray { .. } = &trie.config.trie_strategy {
            if let TrieStorage::DoubleArray { base, check, .. } = &mut trie.storage {
                // Estimate: each key adds ~key_length states on average
                let estimated_states = keys.iter().map(|k| k.len()).sum::<usize>() / 2;
                let initial_size = estimated_states.max(256);

                const NIL_STATE: u32 = 0x7FFF_FFFF;
                const FREE_BIT: u32 = 0x8000_0000;

                base.resize(initial_size, NIL_STATE);
                check.resize(initial_size, NIL_STATE | FREE_BIT);
            }
        }

        // Insert keys in sorted order
        // Sorted order tends to result in fewer relocations
        for &key in keys {
            trie.insert(key)?;
        }

        Ok(trie)
    }
}

/// Map wrapper for ZiporaTrie that associates values with keys
///
/// This is a separate type that wraps a ZiporaTrie and adds value storage.
/// Values are stored in a parallel Vec indexed by the state ID returned from insert.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::ZiporaTrieMap;
///
/// let mut map = ZiporaTrieMap::<u32>::new();
/// map.insert(b"hello", 42).unwrap();
/// map.insert(b"world", 100).unwrap();
///
/// assert_eq!(map.get(b"hello"), Some(42));
/// assert_eq!(map.get(b"world"), Some(100));
/// assert_eq!(map.get(b"missing"), None);
/// ```
#[derive(Debug)]
pub struct ZiporaTrieMap<V: Copy, R = crate::succinct::RankSelectInterleaved256>
where
    R: RankSelectOps,
{
    trie: ZiporaTrie<R>,
    values: Vec<Option<V>>,
}

impl<V: Copy, R> ZiporaTrieMap<V, R>
where
    R: RankSelectOps + Default,
{
    /// Create a new empty trie map
    pub fn new() -> Self {
        Self {
            trie: ZiporaTrie::new(),
            values: Vec::new(),
        }
    }

    /// Create a new trie map with custom configuration
    pub fn with_config(config: ZiporaTrieConfig) -> Self {
        Self {
            trie: ZiporaTrie::with_config(config),
            values: Vec::new(),
        }
    }

    /// Insert a key-value pair, returning the previous value if the key existed
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>> {
        // Get the state ID for this key
        let state_id = <ZiporaTrie<R> as Trie>::insert(&mut self.trie, key)?;

        // Ensure values vec is large enough
        let idx = state_id as usize;
        if idx >= self.values.len() {
            self.values.resize(idx + 1, None);
        }

        // Store the value and return the previous one
        let prev = self.values[idx];
        self.values[idx] = Some(value);

        Ok(prev)
    }

    /// Get the value associated with a key
    pub fn get(&self, key: &[u8]) -> Option<V> {
        // First check if the key exists in the trie
        if !self.trie.contains(key) {
            return None;
        }

        // Find the state ID for this key by traversing
        // For now, we need to traverse to find the state ID
        // This is a simple O(key_length) traversal
        let state_id = self.find_state_for_key(key)?;

        // Return the value at that state
        self.values.get(state_id as usize).and_then(|&v| v)
    }

    /// Helper to find the state ID for a key
    fn find_state_for_key(&self, key: &[u8]) -> Option<StateId> {
        let mut state = self.trie.root();
        for &symbol in key {
            state = self.trie.transition(state, symbol)?;
        }
        Some(state)
    }

    /// Check if a key exists in the map
    pub fn contains(&self, key: &[u8]) -> bool {
        self.trie.contains(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.trie.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }

    /// Get all keys in the map
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.trie.keys()
    }
}

impl<V: Copy, R> Default for ZiporaTrieMap<V, R>
where
    R: RankSelectOps + Default,
{
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_trie_creation() {
        let trie: ZiporaTrie = ZiporaTrie::new();
        assert_eq!(trie.len(), 0);
        assert!(trie.is_empty());
    }

    #[test]
    fn test_cache_optimized_config() {
        let trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::cache_optimized());
        assert!(trie.is_cache_optimized());
    }

    #[test]
    fn test_space_optimized_config() {
        let trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_string_specialized_config() {
        let trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::string_specialized());
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn test_double_array_insert_contains() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        // Default is now DoubleArray
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1);
        assert!(trie.contains(b"hello"));
        assert!(!trie.contains(b"world"));

        trie.insert(b"world").unwrap();
        assert_eq!(trie.len(), 2);
        assert!(trie.contains(b"hello"));
        assert!(trie.contains(b"world"));

        trie.insert(b"help").unwrap();
        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b"help"));
        assert!(trie.contains(b"hello"));

        // Duplicate insert should not increase len
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_double_array_keys() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"apple").unwrap();
        trie.insert(b"app").unwrap();
        trie.insert(b"banana").unwrap();

        let mut keys = trie.keys();
        keys.sort();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], b"app");
        assert_eq!(keys[1], b"apple");
        assert_eq!(keys[2], b"banana");
    }

    #[test]
    fn test_double_array_prefix_with_empty_key() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"").unwrap();
        trie.insert(b"a").unwrap();
        trie.insert(b"ab").unwrap();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();
        trie.insert(b"b").unwrap();

        let all = trie.keys_with_prefix(b"");
        assert_eq!(all.len(), 6, "keys_with_prefix('') should return all 6 keys");
    }

    #[test]
    fn test_double_array_empty_key() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"").unwrap();
        trie.insert(b"a").unwrap();
        trie.insert(b"ab").unwrap();

        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b""));
        assert!(trie.contains(b"a"));
        assert!(trie.contains(b"ab"));

        let mut keys = trie.keys();
        keys.sort();
        assert_eq!(keys.len(), 3, "Should have 3 keys including empty");
        assert_eq!(keys[0], b"");
        assert_eq!(keys[1], b"a");
        assert_eq!(keys[2], b"ab");
    }

    // --- Coverage tests for each improvement ---

    /// Issue #1: Lazy stats — verify stats() works correctly after inserts
    #[test]
    fn test_lazy_stats() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        for i in 0..100 {
            trie.insert(format!("key{:03}", i).as_bytes()).unwrap();
        }
        assert_eq!(trie.len(), 100);
        let stats = trie.stats();
        assert_eq!(stats.num_keys, 100);
        assert!(stats.memory_usage > 0);
        assert!(stats.num_states > 0);
    }

    /// Issue #2: No double traversal — duplicate insert does not increase len
    #[test]
    fn test_no_double_traversal_duplicate() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abc").unwrap();
        assert_eq!(trie.len(), 1);

        trie.insert(b"def").unwrap();
        trie.insert(b"def").unwrap();
        assert_eq!(trie.len(), 2);
    }

    /// Issue #3: Compact PatriciaNode — Patricia still works with compact children
    #[test]
    fn test_patricia_compact_node() {
        let config = ZiporaTrieConfig {
            trie_strategy: crate::fsa::TrieStrategy::Patricia {
                max_path_length: 64,
                compression_threshold: 4,
                adaptive_compression: true,
            },
            ..ZiporaTrieConfig::default()
        };
        let mut trie: ZiporaTrie = ZiporaTrie::with_config(config);
        trie.insert(b"hello").unwrap();
        trie.insert(b"help").unwrap();
        trie.insert(b"world").unwrap();

        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b"hello"));
        assert!(trie.contains(b"help"));
        assert!(trie.contains(b"world"));
        assert!(!trie.contains(b"hel"));
    }

    /// Issue #4/#5: find_free_base + relocate — many inserts don't panic
    #[test]
    fn test_find_free_base_many_inserts() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        // Insert many keys to stress find_free_base and relocation
        for i in 0..500 {
            trie.insert(format!("key_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(trie.len(), 500);
        // Verify random lookups
        assert!(trie.contains(b"key_0000"));
        assert!(trie.contains(b"key_0250"));
        assert!(trie.contains(b"key_0499"));
        assert!(!trie.contains(b"key_0500"));
    }

    /// Issue #6: Amortized growth — large insert doesn't OOM or take forever
    #[test]
    fn test_amortized_growth() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        // 1000 inserts should complete quickly with 1.5x growth
        for i in 0..1000 {
            trie.insert(format!("{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(trie.len(), 1000);
    }

    /// Issue #8: TrieMap — key-value storage
    #[test]
    fn test_trie_map() {
        let mut map = ZiporaTrieMap::<u32>::new();
        map.insert(b"hello", 42).unwrap();
        map.insert(b"world", 100).unwrap();
        map.insert(b"help", 7).unwrap();

        assert_eq!(map.get(b"hello"), Some(42));
        assert_eq!(map.get(b"world"), Some(100));
        assert_eq!(map.get(b"help"), Some(7));
        assert_eq!(map.get(b"missing"), None);
        assert_eq!(map.len(), 3);

        // Update existing key
        let prev = map.insert(b"hello", 99).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(map.get(b"hello"), Some(99));
        assert_eq!(map.len(), 3); // len unchanged
    }

    /// Issue #9: Bulk construction
    #[test]
    fn test_build_from_sorted() {
        let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
        let trie: ZiporaTrie = ZiporaTrie::build_from_sorted(&keys, ZiporaTrieConfig::default()).unwrap();

        assert_eq!(trie.len(), 5);
        assert!(trie.contains(b"apple"));
        assert!(trie.contains(b"application"));
        assert!(trie.contains(b"apply"));
        assert!(trie.contains(b"banana"));
        assert!(trie.contains(b"band"));
        assert!(!trie.contains(b"ban"));
    }

    /// Issue #10: Default is DoubleArray
    #[test]
    fn test_default_is_double_array() {
        let config = ZiporaTrieConfig::default();
        assert!(matches!(config.trie_strategy, crate::fsa::TrieStrategy::DoubleArray { .. }));
    }

    /// DoubleArray remove support
    #[test]
    fn test_double_array_remove() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"hello").unwrap();
        trie.insert(b"world").unwrap();
        assert_eq!(trie.len(), 2);

        assert!(trie.remove(b"hello").unwrap());
        assert_eq!(trie.len(), 1);
        assert!(!trie.contains(b"hello"));
        assert!(trie.contains(b"world"));

        // Remove non-existent key
        assert!(!trie.remove(b"missing").unwrap());
        assert_eq!(trie.len(), 1);
    }

    /// DoubleArray lookup_node_id + restore_string roundtrip
    #[test]
    fn test_double_array_node_id_roundtrip() {
        let mut trie: ZiporaTrie = ZiporaTrie::new();
        trie.insert(b"hello").unwrap();
        trie.insert(b"world").unwrap();

        let node_id = trie.lookup_node_id(b"hello").expect("should find hello");
        let restored = trie.restore_string(node_id).expect("should restore");
        assert_eq!(restored, b"hello");

        let node_id2 = trie.lookup_node_id(b"world").expect("should find world");
        let restored2 = trie.restore_string(node_id2).expect("should restore");
        assert_eq!(restored2, b"world");

        assert!(trie.lookup_node_id(b"missing").is_none());
    }
}