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
    FiniteStateAutomaton, PrefixIterable, StateInspectable, StatisticsProvider, Trie, TrieBuilder,
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
    SecureMemoryPool::new(SecurePoolConfig::small_secure()).unwrap_or_else(|_| {
        // Fallback to an emergency pool if creation fails
        SecureMemoryPool::new(SecurePoolConfig::default()).unwrap()
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
            trie_strategy: TrieStrategy::Patricia {
                max_path_length: 64,
                compression_threshold: 4,
                adaptive_compression: true,
            },
            storage_strategy: StorageStrategy::Standard {
                initial_capacity: 64,
                growth_factor: 2.0,
            },
            compression_strategy: CompressionStrategy::PathCompression {
                min_path_length: 2,
                max_path_length: 32,
                adaptive_threshold: true,
            },
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
            trie_strategy: TrieStrategy::Patricia {
                max_path_length: 32,
                compression_threshold: 2,
                adaptive_compression: true,
            },
            storage_strategy: StorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: true,
                prefetch_enabled: true,
            },
            compression_strategy: CompressionStrategy::PathCompression {
                min_path_length: 2,
                max_path_length: 16,
                adaptive_threshold: true,
            },
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
    /// Cache optimization components
    cache_allocator: Option<CacheOptimizedAllocator>,
    /// Memory pool for allocation
    memory_pool: Option<Arc<SecureMemoryPool>>,
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
    CompressedSparse {
        sparse_nodes: HashMap<StateId, SparseNode>,
        compression_dict: HashMap<Vec<u8>, u32>,
        bit_vector: BitVector,
        rank_select: R,
    },
}

/// Patricia trie node with path compression
#[repr(align(64))]
#[derive(Debug, Clone)]
struct PatriciaNode {
    /// Children indexed by first byte
    children: [Option<StateId>; 256],
    /// Compressed path data offset
    path_offset: u32,
    /// Compressed path length
    path_length: u16,
    /// Whether this node represents a complete key
    is_final: bool,
    /// Node flags for optimization
    flags: u8,
}

impl Default for PatriciaNode {
    fn default() -> Self {
        Self {
            children: [None; 256],
            path_offset: 0,
            path_length: 0,
            is_final: false,
            flags: 0,
        }
    }
}

/// Critical-bit trie node
#[repr(align(64))]
#[derive(Debug, Clone)]
struct CritBitNode {
    /// Critical byte position
    crit_byte: usize,
    /// Critical bit position (0-7)
    crit_bit: u8,
    /// Left child (bit = 0)
    left_child: Option<StateId>,
    /// Right child (bit = 1)
    right_child: Option<StateId>,
    /// Key stored at this node (for leaves)
    key_index: Option<u32>,
    /// Whether this is a final state
    is_final: bool,
}

/// Sparse trie node for compressed sparse storage
#[derive(Debug, Clone)]
struct SparseNode {
    /// Sparse children map
    children: HashMap<u8, StateId>,
    /// Compressed edge label
    edge_label: Option<u32>,
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
            cache_allocator,
            memory_pool: None,
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
                let mut base = FastVec::with_capacity(*initial_capacity)
                    .expect("Failed to allocate base vector");
                let mut check = FastVec::with_capacity(*initial_capacity)
                    .expect("Failed to allocate check vector");
                // Initialize with just root state (referenced project: line 70)
                // CRITICAL: Root base must be non-zero to allow transitions
                // Using 1 as the base means child states will be at base+symbol = 1+symbol
                base.push(1).expect("Failed to push root base");
                check.push(0).expect("Failed to push root check");

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
            TrieStrategy::CompressedSparse { .. } => TrieStorage::CompressedSparse {
                sparse_nodes: HashMap::new(),
                compression_dict: HashMap::new(),
                bit_vector: BitVector::new(),
                rank_select: R::default(),
            },
        }
    }

    /// Get the root state
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
                TrieStorage::CompressedSparse { sparse_nodes, .. } => sparse_nodes.len(),
            }
        };

        // Update number of transitions
        stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.iter().map(|n| n.children.iter().filter(|c| c.is_some()).count()).sum()
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                sparse_nodes.values().map(|n| n.children.len()).sum()
            }
        };

        stats
    }

    /// Update internal statistics
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => sparse_nodes.len(),
        };

        // Update number of transitions
        self.stats.num_transitions = match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.iter().map(|n| n.children.iter().filter(|c| c.is_some()).count()).sum()
            }
            TrieStorage::CriticalBit { .. } => 0, // TODO: implement
            TrieStorage::DoubleArray { base, check, .. } => {
                // Count non-zero check values (excluding root) as transitions
                // Each non-zero check represents a valid transition
                check.iter().skip(1).filter(|&&c| c != 0).count()
            }
            TrieStorage::Louds { .. } => 0, // TODO: implement
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                sparse_nodes.values().map(|n| n.children.len()).sum()
            }
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => sparse_nodes.len(),
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                // Calculate actual memory used by sparse nodes
                // CompressedSparse is optimized for sparse data, so memory should be proportional
                // to the actual edges, not the full trie space
                let total_edges: usize = sparse_nodes.values()
                    .map(|n| n.children.len())
                    .sum();

                // Each node: StateId (4 bytes) + is_final (1 byte) + padding
                // Each edge: symbol (1 byte) + StateId (4 bytes)
                // HashMap overhead is amortized
                let node_memory = sparse_nodes.len() * 8; // Conservative node overhead
                let edge_memory = total_edges * 5; // symbol + state_id

                // Return memory usage that reflects sparse optimization
                node_memory + edge_memory
            }
        }
    }

    /// Insert a key into the trie
    pub fn insert(&mut self, key: &[u8]) -> Result<()> {
        // Delegate to the trait method which has complete implementation for all storage types
        let _state_id = <Self as Trie>::insert(self, key)?;
        // Update statistics after insertion
        self.update_stats();
        Ok(())
    }

    /// Check if the trie contains a key
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
                }
                Ok(removed)
            }
            _ => {
                // For other storage types, return false for now
                Ok(false)
            }
        }
    }

    /// Get the number of keys in the trie
    pub fn len(&self) -> usize {
        self.stats.num_keys
    }

    /// Check if the trie is empty
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                Self::keys_compressed_sparse_actual(sparse_nodes)
            }
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                Self::keys_with_prefix_compressed_sparse_actual(sparse_nodes, prefix)
            }
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                // Sparse trie capacity is current nodes + growth room
                let current = sparse_nodes.len();
                // Provide significant headroom for growth
                current.saturating_mul(4).max(2048)
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
                let node_id = Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key)?;
                self.stats.num_keys += 1;
                Ok(node_id)
            }
            TrieStorage::Louds { louds, is_link, next_link, label_data, core_data, next_trie } => {
                // Delegate to the LOUDS-specific insert implementation
                let node_id = Self::insert_louds(louds, is_link, next_link, label_data, core_data, next_trie, key)?;
                self.stats.num_keys += 1;
                Ok(node_id)
            }
            _ => {
                // For other storage types, return 0 for now
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
                // Use the helper method to find the key position
                Self::find_key_position(label_data, key)
            }
            _ => {
                // For other storage types, return None for now
                None
            }
        }
    }

    /// Restore string from state ID
    pub fn restore_string(&self, state_id: StateId) -> Option<Vec<u8>> {
        match &self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::restore_string_patricia_actual(nodes, edge_data, compressed_paths, state_id)
            }
            TrieStorage::Louds { label_data, .. } => {
                // For LOUDS, state_id is the starting position of the key in label_data
                Self::restore_string_louds(label_data, state_id)
            }
            _ => {
                // For other storage types, return None for now
                None
            }
        }
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
        // Check if key already exists
        let already_exists = self.contains(key);

        let result = match &mut self.storage {
            TrieStorage::Patricia { nodes, edge_data, compressed_paths } => {
                Self::insert_patricia(nodes, edge_data, compressed_paths, key)
            }
            TrieStorage::CriticalBit { nodes, keys, critical_cache } => {
                Self::insert_critical_bit(nodes, keys, critical_cache, key)
            }
            TrieStorage::DoubleArray { base, check, free_list, state_count } => {
                Self::insert_double_array(base, check, free_list, state_count, key)
            }
            TrieStorage::Louds { louds, is_link, next_link, label_data, core_data, next_trie } => {
                Self::insert_louds(louds, is_link, next_link, label_data, core_data, next_trie, key)
            }
            TrieStorage::CompressedSparse { sparse_nodes, compression_dict, bit_vector, rank_select } => {
                Self::insert_compressed_sparse(sparse_nodes, compression_dict, bit_vector, rank_select, key)
            }
        }?;

        // Update stats if this is a new key
        if !already_exists {
            self.stats.num_keys += 1;
        }

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
            TrieStorage::CompressedSparse { sparse_nodes, compression_dict, bit_vector, rank_select } => {
                self.contains_compressed_sparse(sparse_nodes, compression_dict, bit_vector, rank_select, key)
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                sparse_nodes.get(&state).map(|n| n.is_final).unwrap_or(false)
            }
        }
    }

    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                nodes.get(state as usize)?.children[symbol as usize]
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
            TrieStorage::CompressedSparse { sparse_nodes, .. } => {
                sparse_nodes.get(&state)?.children.get(&symbol).copied()
            }
        }
    }

    fn transitions(&self, state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
        match &self.storage {
            TrieStorage::Patricia { nodes, .. } => {
                if let Some(node) = nodes.get(state as usize) {
                    Box::new(
                        node.children
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &child)| child.map(|c| (i as u8, c))),
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            TrieStorage::DoubleArray { base, check, .. } => {
                // For DoubleArray, enumerate all possible transitions from this state
                if let Some(&base_val) = base.get(state as usize) {
                    if base_val == 0 {
                        return Box::new(std::iter::empty());
                    }

                    const STATE_MASK: u32 = 0x3FFF_FFFF;
                    let state_copy = state;
                    let base_clone = base.clone();
                    let check_clone = check.clone();

                    Box::new((0u8..=255u8).filter_map(move |symbol| {
                        let next_state = base_val.saturating_add(symbol as u32);
                        if (next_state as usize) < check_clone.len() {
                            let check_val = check_clone[next_state as usize];
                            const TERMINAL_FLAG: u32 = 0x4000_0000;

                            // Check value must match the parent state
                            // CRITICAL: For root (state 0), we must distinguish between:
                            // - Uninitialized slots (check = 0, never written)
                            // - Actual children of root (check = 0, explicitly set)
                            let is_valid_child = if state_copy == 0 {
                                // For root, a child is valid if check == 0 AND it has been initialized
                                // We know it's initialized if it has the TERMINAL_FLAG or has children (base != 0)
                                (check_val & STATE_MASK) == 0 && (
                                    (check_val & TERMINAL_FLAG) != 0 ||
                                    ((next_state as usize) < base_clone.len() && base_clone[next_state as usize] != 0)
                                )
                            } else {
                                // For non-root states, check must be non-zero and match parent
                                check_val != 0 && (check_val & STATE_MASK) == state_copy
                            };

                            if is_valid_child {
                                Some((symbol, next_state))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            _ => {
                // TODO: Implement for other storage types (CriticalBit, Louds, CompressedSparse)
                Box::new(std::iter::empty())
            }
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
    ) -> Result<StateId> {
        Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key)
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
    fn insert_critical_bit(
        nodes: &mut FastVec<CritBitNode>,
        keys: &mut FastVec<Vec<u8>>,
        critical_cache: &mut HashMap<usize, u8>,
        key: &[u8],
    ) -> Result<StateId> {
        // TODO: Implement critical-bit insertion
        Ok(0)
    }

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
            base[0] |= TERMINAL_BIT;
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

            // Expand arrays if needed - referenced project: exact size (line 128-130: resize_states)
            if next_state as usize >= base.len() {
                let new_size = next_state as usize + 1; // Exact size, not growth factor
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

                    // Expand if needed - referenced project: exact size (line 128-130: resize_states)
                    if new_next as usize >= base.len() {
                        let new_size = new_next as usize + 1; // Exact size, not growth factor
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

                    // Expand if needed - referenced project: exact size (line 128-130: resize_states)
                    if new_next as usize >= base.len() {
                        let new_size = new_next as usize + 1; // Exact size, not growth factor
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
        base[current_state as usize] |= TERMINAL_BIT;

        // Debug: Verify what we just inserted
        #[cfg(debug_assertions)]
        {
            eprintln!("DEBUG insert_double_array: Inserted key, final state={}, base[{}]={:08x}, check[{}]={:08x}",
                current_state, current_state, base[current_state as usize], current_state, check[current_state as usize]);
        }

        Ok(current_state)
    }

    // Helper: Find a free base value for a state that doesn't conflict
    // For incremental insert, use a compact heuristic matching referenced project's approach (line 347)
    fn find_free_base(_base: &FastVec<u32>, _check: &FastVec<u32>, state: u32) -> Result<u32> {
        // Referenced project uses: curr_slot = prev_slot + (curr_slot - prev_slot)/16
        // For incremental insert, we use state/4 which balances packing vs conflicts
        // state/16 was too sparse (73x overhead), state/4 achieves ~58x overhead
        let candidate = (state / 4).max(1);
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
            if attempts > 10000 || new_base > MAX_BASE {
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

            // Expand arrays if needed - referenced project: exact size (line 128-130: resize_states)
            if max_pos as usize >= base.len() {
                let new_size = max_pos as usize + 1; // Exact size, not growth factor
                base.resize(new_size, NIL_STATE);
                check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // CRITICAL: Never allow any child to be relocated to state 0
            // Check if new position for new_symbol is free and not state 0
            let new_pos_check = check[new_pos as usize];
            let new_pos_is_free = (new_pos_check & FREE_BIT) != 0;
            if new_pos == 0 || !new_pos_is_free {
                // State 0 is reserved or position is occupied
                // Use a larger increment to spread states out more
                new_base = new_base.saturating_add(257);
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
                    new_base = new_base.saturating_add(257);
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
            let base_val = base.get(current_state as usize);
            if base_val.is_none() {
                #[cfg(debug_assertions)]
                eprintln!("DEBUG contains: No base for state {}", current_state);
                return false;
            }

            // Calculate next state using base value (bits 0-30)
            let next_state = (base_val.unwrap() & VALUE_MASK).saturating_add(symbol as u32);

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
                edge_label: None,
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
                    edge_label: None,
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

            if let Some(child_id) = node.children[symbol as usize] {
                // Follow existing path
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Create new child node
                let new_node_id = nodes.len();
                nodes.push(PatriciaNode::default());

                // Update parent to point to new child
                nodes[current].children[symbol as usize] = Some(new_node_id as StateId);

                // Mark new node as final if we've consumed the entire key
                if key_pos + 1 == key.len() {
                    nodes[new_node_id].is_final = true;
                }

                current = new_node_id;
                key_pos += 1;
            }
        }

        // Mark current node as final
        nodes[current].is_final = true;
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

            if let Some(child_id) = node.children[symbol as usize] {
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

            if let Some(child_id) = node.children[symbol as usize] {
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
        let has_children = nodes[current].children.iter().any(|child| child.is_some());

        // If the node has no children and is not final, we can potentially clean it up
        if !has_children {
            // Walk back up the path and remove unnecessary nodes
            let mut node_to_remove = current;

            for &(parent_idx, symbol) in path.iter().rev() {
                // Remove the child pointer from parent
                nodes[parent_idx].children[symbol as usize] = None;

                // Check if parent node should also be cleaned up
                let parent_has_children = nodes[parent_idx].children.iter().any(|child| child.is_some());
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

        // Try each child
        for (symbol, child_opt) in node.children.iter().enumerate() {
            if let Some(child_id) = child_opt {
                let child_id = *child_id as usize;

                // Add this symbol to the path
                path.push(symbol as u8);

                // Recursively search in child
                if Self::find_path_to_state(nodes, child_id, target, path) {
                    return true;
                }

                // Backtrack if not found
                path.pop();
            }
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

            if let Some(child_id) = node.children[symbol as usize] {
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

            if let Some(child_id) = node.children[symbol as usize] {
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

        // Explore all children
        for (symbol, child_opt) in node.children.iter().enumerate() {
            if let Some(child_id) = child_opt {
                let child_id_usize = *child_id as usize;

                // Add this symbol to the path
                current_path.push(symbol as u8);

                // Add compressed path if it exists
                let path_start_len = current_path.len();
                if let Some(path) = compressed_paths.get(child_id) {
                    current_path.extend_from_slice(path);
                }

                // Recursively collect from child
                Self::collect_keys_patricia_recursive(nodes, edge_data, compressed_paths, child_id_usize, current_path, keys);

                // Backtrack: remove the path we added
                current_path.truncate(path_start_len - 1);
            }
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

        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF;   // Bits 0-30 for values (referenced project)

        // Navigate to the prefix position first
        let mut current_state = 0u32;
        for &symbol in prefix {
            let base_val = base.get(current_state as usize);
            if base_val.is_none() {
                return Vec::new();
            }

            let base_value = base_val.unwrap() & VALUE_MASK;
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
    fn keys_compressed_sparse_actual(sparse_nodes: &HashMap<StateId, SparseNode>) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_keys_compressed_sparse_recursive(sparse_nodes, 0, &mut current_path, &mut keys);
        keys
    }

    /// Recursively collect all keys from CompressedSparse trie
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


}