//! Strategy Traits for Unified Trie Implementation
//!
//! This module defines the strategy traits that enable the unified ZiporaTrie
//! to support all existing trie variants through pluggable algorithms.
//!
//! # Strategy Architecture
//!
//! The strategy pattern allows different algorithms to be combined:
//! - **TrieAlgorithmStrategy**: Core trie algorithm (Patricia, CritBit, DoubleArray, LOUDS)
//! - **CompressionStrategy**: Path and fragment compression techniques
//! - **SuccinctStorageStrategy**: Rank/select and bit vector implementations
//! - **ConcurrencyStrategy**: Token-based synchronization and concurrent access
//!
//! This enables a single unified implementation to support all use cases that
//! previously required separate implementations.

use crate::containers::specialized::UintVector;
use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::fsa::traits::{TrieStats, StatisticsProvider};
use crate::memory::cache_layout::{CacheOptimizedAllocator, PrefetchHint};
use crate::succinct::{BitVector, RankSelectOps};
use crate::StateId;
use std::collections::{HashMap, VecDeque};

/// Core trie algorithm strategy
pub trait TrieAlgorithmStrategy {
    /// Configuration for this algorithm
    type Config: Clone;

    /// Context/state maintained by this strategy
    type Context: Default;

    /// Node type used by this strategy
    type Node: Clone;

    /// Initialize the algorithm with given configuration
    fn initialize(config: &Self::Config) -> Self::Context;

    /// Insert a key and return the final state ID
    fn insert(
        &self,
        context: &mut Self::Context,
        nodes: &mut FastVec<Self::Node>,
        key: &[u8],
        config: &Self::Config,
    ) -> Result<StateId>;

    /// Look up a key and return whether it exists
    fn lookup(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        key: &[u8],
        config: &Self::Config,
    ) -> bool;

    /// Perform state transition with given symbol
    fn transition(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        symbol: u8,
        config: &Self::Config,
    ) -> Option<StateId>;

    /// Check if state is final
    fn is_final(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        config: &Self::Config,
    ) -> bool;

    /// Get all transitions from a state
    fn transitions(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        config: &Self::Config,
    ) -> Vec<(u8, StateId)>;

    /// Optimize the trie structure (e.g., minimize, compress)
    fn optimize(
        &self,
        context: &mut Self::Context,
        nodes: &mut FastVec<Self::Node>,
        config: &Self::Config,
    ) -> Result<()>;

    /// Get algorithm-specific statistics
    fn statistics(&self, context: &Self::Context, nodes: &FastVec<Self::Node>) -> AlgorithmStats;

    /// Estimate memory usage
    fn memory_usage(&self, context: &Self::Context, nodes: &FastVec<Self::Node>) -> usize;
}

/// Compression strategy for space optimization
pub trait CompressionStrategy {
    /// Configuration for compression
    type Config: Clone;

    /// Compression context/state
    type Context: Default;

    /// Compressed data representation
    type CompressedData: Clone;

    /// Initialize compression with configuration
    fn initialize(config: &Self::Config) -> Self::Context;

    /// Compress a path or fragment
    fn compress(
        &self,
        context: &mut Self::Context,
        data: &[u8],
        config: &Self::Config,
    ) -> Result<Self::CompressedData>;

    /// Decompress data back to original form
    fn decompress(
        &self,
        context: &Self::Context,
        compressed: &Self::CompressedData,
        config: &Self::Config,
    ) -> Result<Vec<u8>>;

    /// Check if compression is beneficial for given data
    fn should_compress(
        &self,
        context: &Self::Context,
        data: &[u8],
        config: &Self::Config,
    ) -> bool;

    /// Get compression ratio achieved
    fn compression_ratio(&self, context: &Self::Context) -> f64;

    /// Update compression dictionary/statistics
    fn update_dictionary(
        &self,
        context: &mut Self::Context,
        data: &[u8],
        frequency: u32,
        config: &Self::Config,
    );

    /// Get compression statistics
    fn compression_stats(&self, context: &Self::Context) -> CompressionStats;
}

/// Succinct storage strategy for rank/select operations
pub trait SuccinctStorageStrategy {
    /// Rank/select implementation type
    type RankSelect: RankSelectOps + Default;

    /// Configuration for succinct structures
    type Config: Clone;

    /// Storage context
    type Context: Default;

    /// Initialize succinct storage
    fn initialize(config: &Self::Config) -> Self::Context;

    /// Create a bit vector with given capacity
    fn create_bit_vector(
        &self,
        context: &mut Self::Context,
        capacity: usize,
        config: &Self::Config,
    ) -> BitVector;

    /// Build rank/select structure from bit vector
    fn build_rank_select(
        &self,
        context: &mut Self::Context,
        bit_vector: &BitVector,
        config: &Self::Config,
    ) -> Result<Self::RankSelect>;

    /// Optimize storage layout for cache efficiency
    fn optimize_layout(
        &self,
        context: &mut Self::Context,
        rank_select: &mut Self::RankSelect,
        config: &Self::Config,
    ) -> Result<()>;

    /// Get storage efficiency metrics
    fn storage_efficiency(&self, context: &Self::Context) -> StorageEfficiency;

    /// Estimate memory usage of succinct structures
    fn succinct_memory_usage(&self, context: &Self::Context) -> usize;
}

/// Concurrency strategy for thread-safe operations
pub trait ConcurrencyStrategy {
    /// Configuration for concurrency
    type Config: Clone;

    /// Concurrency context (locks, tokens, etc.)
    type Context: Default + Send + Sync;

    /// Reader token type
    type ReaderToken;

    /// Writer token type
    type WriterToken;

    /// Initialize concurrency control
    fn initialize(config: &Self::Config) -> Self::Context;

    /// Acquire a reader token for read operations
    fn acquire_read_token(&self, context: &Self::Context) -> Result<Self::ReaderToken>;

    /// Acquire a writer token for write operations
    fn acquire_write_token(&self, context: &Self::Context) -> Result<Self::WriterToken>;

    /// Release a reader token
    fn release_read_token(&self, context: &Self::Context, token: Self::ReaderToken);

    /// Release a writer token
    fn release_write_token(&self, context: &Self::Context, token: Self::WriterToken);

    /// Check if concurrent read access is allowed
    fn allow_concurrent_reads(&self, context: &Self::Context) -> bool;

    /// Check if concurrent write access is allowed
    fn allow_concurrent_writes(&self, context: &Self::Context) -> bool;

    /// Get concurrency statistics
    fn concurrency_stats(&self, context: &Self::Context) -> ConcurrencyStats;
}

/// Statistics for algorithm performance
#[derive(Debug, Default, Clone)]
pub struct AlgorithmStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub max_depth: usize,
    pub avg_branching_factor: f64,
    pub path_compression_ratio: f64,
    pub cache_efficiency: f64,
}

/// Compression performance statistics
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub dictionary_size: usize,
    pub fragments_compressed: usize,
    pub compression_time_ns: u64,
}

/// Storage efficiency metrics
#[derive(Debug, Default, Clone)]
pub struct StorageEfficiency {
    pub bits_per_node: f64,
    pub rank_select_overhead: f64,
    pub cache_hit_ratio: f64,
    pub space_utilization: f64,
}

/// Concurrency performance statistics
#[derive(Debug, Default, Clone)]
pub struct ConcurrencyStats {
    pub active_readers: usize,
    pub active_writers: usize,
    pub reader_wait_time_ns: u64,
    pub writer_wait_time_ns: u64,
    pub lock_contention_ratio: f64,
    pub token_cache_hits: u64,
}

// Concrete strategy implementations

/// Patricia trie algorithm strategy
pub struct PatriciaAlgorithmStrategy;

#[derive(Debug, Clone)]
pub struct PatriciaConfig {
    pub max_path_length: usize,
    pub compression_threshold: usize,
    pub adaptive_compression: bool,
}

#[derive(Debug, Default)]
pub struct PatriciaContext {
    pub compressed_paths: HashMap<StateId, Vec<u8>>,
    pub path_stats: PathCompressionStats,
}

#[derive(Debug, Default)]
pub struct PathCompressionStats {
    pub paths_compressed: usize,
    pub total_path_length: usize,
    pub compressed_path_length: usize,
}

/// Patricia trie node
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct PatriciaNode {
    /// Children indexed by first byte
    pub children: [Option<StateId>; 256],
    /// Compressed path data offset
    pub path_offset: u32,
    /// Compressed path length
    pub path_length: u16,
    /// Whether this node represents a complete key
    pub is_final: bool,
    /// Node flags for optimization
    pub flags: u8,
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

impl TrieAlgorithmStrategy for PatriciaAlgorithmStrategy {
    type Config = PatriciaConfig;
    type Context = PatriciaContext;
    type Node = PatriciaNode;

    fn initialize(config: &Self::Config) -> Self::Context {
        PatriciaContext::default()
    }

    fn insert(
        &self,
        context: &mut Self::Context,
        nodes: &mut FastVec<Self::Node>,
        key: &[u8],
        config: &Self::Config,
    ) -> Result<StateId> {
        if nodes.is_empty() {
            nodes.push(PatriciaNode::default());
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            if let Some(child_id) = node.children[symbol as usize] {
                // Follow existing edge
                current = child_id as usize;
                key_pos += 1;

                // Check for compressed path
                if let Some(path) = context.compressed_paths.get(&(child_id)) {
                    let path_clone = path.clone(); // Clone to avoid borrow conflict
                    let match_len = self.match_path(key, key_pos, &path_clone);
                    if match_len == path_clone.len() {
                        // Full path match
                        key_pos += match_len;
                    } else if match_len < path_clone.len() {
                        // Partial path match - need to split
                        return self.split_compressed_path(
                            context, nodes, current, key, key_pos, &path_clone, match_len, config,
                        );
                    }
                }
            } else {
                // Create new child
                let new_node_id = nodes.len();
                nodes.push(PatriciaNode::default());

                // Update parent to point to new child
                nodes[current].children[symbol as usize] = Some(new_node_id as StateId);

                // Check if we should compress the remaining path
                let remaining_key = &key[key_pos + 1..];
                if remaining_key.len() >= config.compression_threshold {
                    context.compressed_paths.insert(new_node_id as StateId, remaining_key.to_vec());
                    context.path_stats.paths_compressed += 1;
                    context.path_stats.total_path_length += remaining_key.len();
                    context.path_stats.compressed_path_length += 1; // Compressed to single node
                }

                nodes[new_node_id].is_final = true;
                return Ok(new_node_id as StateId);
            }
        }

        nodes[current].is_final = true;
        Ok(current as StateId)
    }

    fn lookup(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        key: &[u8],
        config: &Self::Config,
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

                // Check compressed path
                if let Some(path) = context.compressed_paths.get(&child_id) {
                    if !self.match_compressed_path(key, key_pos, path) {
                        return false;
                    }
                    key_pos += path.len();
                }
            } else {
                return false;
            }
        }

        key_pos == key.len() && nodes[current].is_final
    }

    fn transition(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        symbol: u8,
        config: &Self::Config,
    ) -> Option<StateId> {
        if state as usize >= nodes.len() {
            return None;
        }

        nodes[state as usize].children[symbol as usize]
    }

    fn is_final(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        config: &Self::Config,
    ) -> bool {
        if state as usize >= nodes.len() {
            return false;
        }

        nodes[state as usize].is_final
    }

    fn transitions(
        &self,
        context: &Self::Context,
        nodes: &FastVec<Self::Node>,
        state: StateId,
        config: &Self::Config,
    ) -> Vec<(u8, StateId)> {
        if state as usize >= nodes.len() {
            return Vec::new();
        }

        let node = &nodes[state as usize];
        node.children
            .iter()
            .enumerate()
            .filter_map(|(i, &child)| child.map(|c| (i as u8, c)))
            .collect()
    }

    fn optimize(
        &self,
        context: &mut Self::Context,
        nodes: &mut FastVec<Self::Node>,
        config: &Self::Config,
    ) -> Result<()> {
        // TODO: Implement optimization (path compression, node merging, etc.)
        Ok(())
    }

    fn statistics(&self, context: &Self::Context, nodes: &FastVec<Self::Node>) -> AlgorithmStats {
        let edge_count = nodes.iter().map(|n| n.children.iter().filter(|c| c.is_some()).count()).sum();
        let compression_ratio = if context.path_stats.total_path_length > 0 {
            context.path_stats.compressed_path_length as f64 / context.path_stats.total_path_length as f64
        } else {
            1.0
        };

        AlgorithmStats {
            node_count: nodes.len(),
            edge_count,
            max_depth: 0, // TODO: Calculate max depth
            avg_branching_factor: if nodes.is_empty() { 0.0 } else { edge_count as f64 / nodes.len() as f64 },
            path_compression_ratio: compression_ratio,
            cache_efficiency: 0.0, // TODO: Calculate cache efficiency
        }
    }

    fn memory_usage(&self, context: &Self::Context, nodes: &FastVec<Self::Node>) -> usize {
        let node_memory = nodes.capacity() * std::mem::size_of::<PatriciaNode>();
        let path_memory = context.compressed_paths.iter()
            .map(|(_, path)| path.len())
            .sum::<usize>();
        node_memory + path_memory
    }
}

impl PatriciaAlgorithmStrategy {
    fn match_path(&self, key: &[u8], start_pos: usize, path: &[u8]) -> usize {
        let mut i = 0;
        while i < path.len() && start_pos + i < key.len() && key[start_pos + i] == path[i] {
            i += 1;
        }
        i
    }

    fn match_compressed_path(&self, key: &[u8], start_pos: usize, path: &[u8]) -> bool {
        if start_pos + path.len() > key.len() {
            return false;
        }

        key[start_pos..start_pos + path.len()] == *path
    }

    fn split_compressed_path(
        &self,
        context: &mut PatriciaContext,
        nodes: &mut FastVec<PatriciaNode>,
        current: usize,
        key: &[u8],
        key_pos: usize,
        path: &[u8],
        match_len: usize,
        config: &PatriciaConfig,
    ) -> Result<StateId> {
        // TODO: Implement path splitting for partial matches
        Ok(current as StateId)
    }
}

/// Path compression strategy
pub struct PathCompressionStrategy;

#[derive(Debug, Clone)]
pub struct PathCompressionConfig {
    pub min_path_length: usize,
    pub max_path_length: usize,
    pub adaptive_threshold: bool,
}

#[derive(Debug, Default)]
pub struct PathCompressionContext {
    pub compressed_paths: HashMap<u32, Vec<u8>>,
    pub compression_stats: CompressionStats,
}

impl CompressionStrategy for PathCompressionStrategy {
    type Config = PathCompressionConfig;
    type Context = PathCompressionContext;
    type CompressedData = u32; // Index into compressed_paths

    fn initialize(config: &Self::Config) -> Self::Context {
        PathCompressionContext::default()
    }

    fn compress(
        &self,
        context: &mut Self::Context,
        data: &[u8],
        config: &Self::Config,
    ) -> Result<Self::CompressedData> {
        if data.len() < config.min_path_length {
            return Err(ZiporaError::invalid_data("Path too short for compression"));
        }

        let index = context.compressed_paths.len() as u32;
        context.compressed_paths.insert(index, data.to_vec());

        // Update stats
        context.compression_stats.original_size += data.len();
        context.compression_stats.compressed_size += 4; // Just the index
        context.compression_stats.fragments_compressed += 1;

        Ok(index)
    }

    fn decompress(
        &self,
        context: &Self::Context,
        compressed: &Self::CompressedData,
        config: &Self::Config,
    ) -> Result<Vec<u8>> {
        context.compressed_paths.get(compressed)
            .map(|path| path.clone())
            .ok_or_else(|| ZiporaError::invalid_data("Compressed path not found"))
    }

    fn should_compress(
        &self,
        context: &Self::Context,
        data: &[u8],
        config: &Self::Config,
    ) -> bool {
        data.len() >= config.min_path_length && data.len() <= config.max_path_length
    }

    fn compression_ratio(&self, context: &Self::Context) -> f64 {
        if context.compression_stats.original_size > 0 {
            context.compression_stats.compressed_size as f64 / context.compression_stats.original_size as f64
        } else {
            1.0
        }
    }

    fn update_dictionary(
        &self,
        context: &mut Self::Context,
        data: &[u8],
        frequency: u32,
        config: &Self::Config,
    ) {
        // For path compression, we don't maintain a frequency dictionary
        // but we could track usage statistics here
    }

    fn compression_stats(&self, context: &Self::Context) -> CompressionStats {
        context.compression_stats.clone()
    }
}

/// No-op concurrency strategy for single-threaded access
pub struct SingleThreadedConcurrencyStrategy;

#[derive(Debug, Clone)]
pub struct SingleThreadedConfig;

#[derive(Debug, Default)]
pub struct SingleThreadedContext;

pub struct NoOpToken;

impl ConcurrencyStrategy for SingleThreadedConcurrencyStrategy {
    type Config = SingleThreadedConfig;
    type Context = SingleThreadedContext;
    type ReaderToken = NoOpToken;
    type WriterToken = NoOpToken;

    fn initialize(config: &Self::Config) -> Self::Context {
        SingleThreadedContext
    }

    fn acquire_read_token(&self, context: &Self::Context) -> Result<Self::ReaderToken> {
        Ok(NoOpToken)
    }

    fn acquire_write_token(&self, context: &Self::Context) -> Result<Self::WriterToken> {
        Ok(NoOpToken)
    }

    fn release_read_token(&self, context: &Self::Context, token: Self::ReaderToken) {
        // No-op
    }

    fn release_write_token(&self, context: &Self::Context, token: Self::WriterToken) {
        // No-op
    }

    fn allow_concurrent_reads(&self, context: &Self::Context) -> bool {
        false // Single-threaded
    }

    fn allow_concurrent_writes(&self, context: &Self::Context) -> bool {
        false // Single-threaded
    }

    fn concurrency_stats(&self, context: &Self::Context) -> ConcurrencyStats {
        ConcurrencyStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patricia_algorithm_strategy() {
        let strategy = PatriciaAlgorithmStrategy;
        let config = PatriciaConfig {
            max_path_length: 64,
            compression_threshold: 4,
            adaptive_compression: true,
        };
        let mut context = PatriciaAlgorithmStrategy::initialize(&config);
        let mut nodes = FastVec::new();

        // Test insertion
        let result = strategy.insert(&mut context, &mut nodes, b"hello", &config);
        assert!(result.is_ok());

        // Test lookup
        assert!(strategy.lookup(&context, &nodes, b"hello", &config));
        assert!(!strategy.lookup(&context, &nodes, b"world", &config));
    }

    #[test]
    fn test_path_compression_strategy() {
        let strategy = PathCompressionStrategy;
        let config = PathCompressionConfig {
            min_path_length: 2,
            max_path_length: 32,
            adaptive_threshold: true,
        };
        let mut context = PathCompressionStrategy::initialize(&config);

        // Test compression
        let data = b"hello_world";
        let compressed = strategy.compress(&mut context, data, &config);
        assert!(compressed.is_ok());

        // Test decompression
        let decompressed = strategy.decompress(&context, &compressed.unwrap(), &config);
        assert!(decompressed.is_ok());
        assert_eq!(decompressed.unwrap(), data);
    }

    #[test]
    fn test_single_threaded_concurrency() {
        let strategy = SingleThreadedConcurrencyStrategy;
        let config = SingleThreadedConfig;
        let context = SingleThreadedConcurrencyStrategy::initialize(&config);

        let read_token = strategy.acquire_read_token(&context);
        assert!(read_token.is_ok());

        let write_token = strategy.acquire_write_token(&context);
        assert!(write_token.is_ok());

        assert!(!strategy.allow_concurrent_reads(&context));
        assert!(!strategy.allow_concurrent_writes(&context));
    }
}