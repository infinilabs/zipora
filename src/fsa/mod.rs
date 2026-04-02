//! High-performance Finite State Automata and Trie implementation
//!
//! **ZiporaTrie**: Single, highly optimized trie implementation with strategy-based
//! configuration. One Patricia class with concurrency level as a config enum,
//! not separate wrapper types.
//!
//! # Examples
//!
//! ```rust
//! use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
//! use zipora::fsa::traits::Trie;
//! use zipora::succinct::RankSelectInterleaved256;
//!
//! let mut trie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::new();
//! trie.insert(b"hello").unwrap();
//! assert!(trie.contains(b"hello"));
//! ```

// Core implementation modules
pub mod double_array;
pub mod zipora_trie;
pub mod strategy_traits;

// Core infrastructure modules
pub mod cache;
pub mod fast_search;
pub mod graph_walker;
pub mod simple_implementations;
pub mod token;
pub mod traits;
pub mod version_sync;
pub mod dawg;

// Core ZiporaTrie implementation
pub use zipora_trie::{
    ZiporaTrie, ZiporaTrieConfig, ZiporaTrieMap,
    TrieStrategy, StorageStrategy, CompressionStrategy, RankSelectType, BitVectorType,
};

// Strategy traits for advanced configuration
pub use strategy_traits::{
    TrieAlgorithmStrategy, CompressionStrategy as CompressionStrategyTrait,
    ConcurrencyStrategy,
    AlgorithmStats, CompressionStats, StorageEfficiency, ConcurrencyStats,
    PatriciaAlgorithmStrategy, PatriciaConfig, PatriciaContext, PatriciaNode,
    PathCompressionStrategy, PathCompressionConfig, PathCompressionContext,
    SingleThreadedConcurrencyStrategy, SingleThreadedConfig, SingleThreadedContext, NoOpToken,
};

// Core infrastructure
pub use cache::{
    CacheStrategy, CachedState, FsaCache, FsaCacheConfig, FsaCacheStats, ZeroPathData,
    FastStateCache,
};
pub use dawg::{
    DawgConfig, DawgState, DawgStats, NestedTrieDawg, TerminalStrategy, TransitionTable,
};
pub use fast_search::{
    FastSearchConfig, FastSearchEngine, HardwareCapabilities, SearchStrategy,
    fast_search_byte, fast_search_byte_max_16, binary_search_byte,
};
pub use graph_walker::{
    BfsGraphWalker, CfsGraphWalker, DfsGraphWalker, GraphVisitor, GraphWalker, GraphWalkerFactory,
    MultiPassWalker, SimpleVertex, Vertex, VertexColor, WalkMethod, WalkStats, WalkerConfig,
    FastBfsWalker, FastDfsWalker, FastCfsWalker,
};
pub use simple_implementations::{
    SimpleDawg, SimpleFastSearch, SimpleFsaCache, SimpleGraphWalker,
};
pub use token::{
    GlobalTokenStats, TokenCache, TokenCacheStats, TokenManager,
    with_reader_token, with_writer_token,
};
pub use traits::{
    FiniteStateAutomaton, FsaError, PrefixIterable, StatisticsProvider, Trie,
    TrieStats,
};
pub use version_sync::{
    LazyFreeItem, LazyFreeList, LazyFreeStats, ReaderToken as VersionReaderToken,
    VersionManager, VersionManagerStats, WriterToken as VersionWriterToken,
};

// ============================================================================
// LEGACY COMPATIBILITY - Minimal wrappers for old API names
// ============================================================================
//
// Previously 1,092 LOC of forwarding wrappers. Now thin modules with
// just the types needed for backward compat. All delegate to ZiporaTrie.

// DoubleArrayTrie is the standalone optimized implementation in double_array.rs.
// No wrapper — use it directly.
pub use double_array::{DoubleArrayTrie, DoubleArrayTrieMap, DoubleArrayTrieCursor, MapValue, RangeIter};

/// NestedLoudsTrie compatibility wrapper
pub mod nested_louds_trie {
    use super::*;
    use crate::error::Result;
    use std::marker::PhantomData;

    #[derive(Debug, Clone)]
    pub struct NestedTrieStats {
        pub key_count: usize, pub total_memory: usize, pub num_states: usize,
        pub num_keys: usize, pub num_transitions: usize, pub max_depth: usize,
        pub avg_depth: f64, pub memory_usage: usize, pub bits_per_key: f64,
    }

    #[derive(Debug, Clone)]
    pub struct FragmentStats { pub compression_ratio: f64, pub fragment_count: usize }

    pub type NestingLevel = u8;

    #[derive(Debug, Clone)]
    pub struct NestingConfig {
        pub max_levels: usize, pub fragment_compression_ratio: f64,
        pub min_fragment_size: usize, pub max_fragment_size: usize,
        pub cache_optimization: bool, pub cache_block_size: usize,
        pub density_switch_threshold: f64, pub adaptive_backend_selection: bool,
        pub memory_pool_size: usize,
    }
    impl Default for NestingConfig {
        fn default() -> Self {
            Self { max_levels: 3, fragment_compression_ratio: 0.5, min_fragment_size: 64,
                   max_fragment_size: 65536, cache_optimization: true, cache_block_size: 64,
                   density_switch_threshold: 0.5, adaptive_backend_selection: true,
                   memory_pool_size: 0 }
        }
    }
    // NestingConfig has all public fields + Default — use struct literals directly:
    // NestingConfig { max_levels: 5, ..Default::default() }

    pub struct NestedLoudsTrie<T> { trie: ZiporaTrie, _marker: PhantomData<T> }

    impl<T> NestedLoudsTrie<T> {
        pub fn new() -> Result<Self> {
            Ok(Self { trie: ZiporaTrie::with_config(ZiporaTrieConfig::space_optimized()), _marker: PhantomData })
        }
        pub fn with_config(config: NestingConfig) -> Result<Self> {
            let mut tc = ZiporaTrieConfig::space_optimized();
            tc.trie_strategy = TrieStrategy::Louds {
                nesting_levels: config.max_levels, fragment_compression: true,
                adaptive_backends: true, cache_aligned: config.cache_optimization,
            };
            Ok(Self { trie: ZiporaTrie::with_config(tc), _marker: PhantomData })
        }
        pub fn insert(&mut self, key: &[u8]) -> Result<()> { self.trie.insert(key) }
        #[inline]
        pub fn contains(&self, key: &[u8]) -> bool { self.trie.contains(key) }
        pub fn lookup(&self, key: &[u8]) -> Option<()> { if self.trie.contains(key) { Some(()) } else { None } }
        #[inline]
        pub fn len(&self) -> usize { self.trie.len() }
        pub fn is_empty(&self) -> bool { self.trie.is_empty() }
        pub fn stats(&self) -> TrieStats { self.trie.stats().clone() }
        #[inline]
        pub fn memory_usage(&self) -> usize { self.trie.memory_usage() }
        pub fn config(&self) -> NestingConfig { NestingConfig::default() }
        pub fn active_levels(&self) -> usize { 1 }
        pub fn performance_stats(&self) -> NestedTrieStats {
            let s = self.trie.stats();
            NestedTrieStats {
                key_count: s.num_keys, total_memory: s.memory_usage, num_states: s.num_states,
                num_keys: s.num_keys, num_transitions: s.num_transitions, max_depth: s.max_depth,
                avg_depth: s.avg_depth, memory_usage: s.memory_usage,
                bits_per_key: if s.num_keys > 0 { (s.memory_usage * 8) as f64 / s.num_keys as f64 } else { 0.0 },
            }
        }
        pub fn fragment_stats(&self) -> FragmentStats { FragmentStats { compression_ratio: 0.5, fragment_count: 0 } }
        pub fn builder() -> NestedLoudsTrieBuilder<T> { NestedLoudsTrieBuilder::new() }
        pub fn bits_per_key(&self) -> f64 {
            let s = self.trie.stats();
            if s.num_keys > 0 { (s.memory_usage * 8) as f64 / s.num_keys as f64 } else { 0.0 }
        }
    }

    impl<T> Trie for NestedLoudsTrie<T> {
        fn insert(&mut self, key: &[u8]) -> Result<crate::StateId> { self.trie.insert(key)?; Ok(0) }
        fn contains(&self, key: &[u8]) -> bool { self.trie.contains(key) }
        fn lookup(&self, key: &[u8]) -> Option<crate::StateId> { if self.contains(key) { Some(0) } else { None } }
        fn len(&self) -> usize { self.trie.len() }
        fn is_empty(&self) -> bool { self.trie.is_empty() }
    }

    impl<T> FiniteStateAutomaton for NestedLoudsTrie<T> {
        fn root(&self) -> u32 { self.trie.root() }
        fn accepts(&self, key: &[u8]) -> bool { self.trie.accepts(key) }
        fn longest_prefix(&self, input: &[u8]) -> Option<usize> { self.trie.longest_prefix(input) }
        fn transition(&self, state: u32, symbol: u8) -> Option<u32> { self.trie.transition(state, symbol) }
        fn transitions(&self, state: u32) -> Vec<(u8, u32)> { self.trie.transitions(state) }
        fn is_final(&self, state: u32) -> bool { self.trie.is_final(state) }
    }

    pub struct NestedLoudsTrieBuilder<T> { config: NestingConfig, _marker: PhantomData<T> }
    impl<T> NestedLoudsTrieBuilder<T> {
        pub fn new() -> Self { Self { config: NestingConfig::default(), _marker: PhantomData } }
        pub fn with_config(config: NestingConfig) -> Self { Self { config, _marker: PhantomData } }
        pub fn build_from_iter<I>(self, keys: I) -> Result<NestedLoudsTrie<T>>
        where I: IntoIterator<Item = Vec<u8>> {
            let mut t = NestedLoudsTrie::with_config(self.config)?;
            for k in keys { t.insert(&k)?; }
            Ok(t)
        }
    }

}

pub use nested_louds_trie::{
    NestedLoudsTrie, NestingConfig, NestingLevel, NestedTrieStats, FragmentStats,
};

/// CompressedSparseTrie compatibility wrapper
pub mod compressed_sparse_trie {
    use super::*;
    use crate::error::Result;
    use std::sync::Arc;
    use crate::memory::SecureMemoryPool;

    pub type ConcurrencyLevel = crate::fsa::version_sync::ConcurrencyLevel;
    pub type ReaderToken = super::VersionReaderToken;
    pub type WriterToken = super::VersionWriterToken;

    pub struct CompressedSparseTrie { trie: ZiporaTrie }

    impl CompressedSparseTrie {
        pub fn new(_level: ConcurrencyLevel) -> Result<Self> {
            Ok(Self { trie: ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized()) })
        }
        pub fn with_memory_pool(_level: ConcurrencyLevel, _pool: Arc<SecureMemoryPool>) -> Result<Self> {
            Self::new(_level)
        }
        pub fn insert(&mut self, key: &[u8]) -> Result<()> { self.trie.insert(key) }
        #[inline]
        pub fn contains(&self, key: &[u8]) -> bool { self.trie.contains(key) }
        pub fn lookup(&self, key: &[u8]) -> Option<()> { if self.trie.contains(key) { Some(()) } else { None } }
        pub fn insert_with_token(&mut self, key: &[u8], _token: &WriterToken) -> Result<()> { self.trie.insert(key) }
        pub fn contains_with_token(&self, key: &[u8], _token: &ReaderToken) -> bool { self.trie.contains(key) }
        pub fn lookup_with_token(&self, key: &[u8], _token: &ReaderToken) -> Option<()> { self.lookup(key) }
        #[inline]
        pub fn len(&self) -> usize { self.trie.len() }
        pub fn is_empty(&self) -> bool { self.trie.is_empty() }
        pub fn stats(&self) -> TrieStats { self.trie.stats().clone() }
    }

    impl Trie for CompressedSparseTrie {
        fn insert(&mut self, key: &[u8]) -> Result<crate::StateId> { self.trie.insert(key)?; Ok(0) }
        fn contains(&self, key: &[u8]) -> bool { self.trie.contains(key) }
        fn lookup(&self, key: &[u8]) -> Option<crate::StateId> { if self.contains(key) { Some(0) } else { None } }
        fn len(&self) -> usize { self.trie.len() }
        fn is_empty(&self) -> bool { self.trie.is_empty() }
    }

    impl FiniteStateAutomaton for CompressedSparseTrie {
        fn root(&self) -> u32 { self.trie.root() }
        fn accepts(&self, key: &[u8]) -> bool { self.trie.accepts(key) }
        fn longest_prefix(&self, input: &[u8]) -> Option<usize> { self.trie.longest_prefix(input) }
        fn transition(&self, state: u32, symbol: u8) -> Option<u32> { self.trie.transition(state, symbol) }
        fn transitions(&self, state: u32) -> Vec<(u8, u32)> { self.trie.transitions(state) }
        fn is_final(&self, state: u32) -> bool { self.trie.is_final(state) }
    }
}

pub use compressed_sparse_trie::{CompressedSparseTrie, ConcurrencyLevel, ReaderToken, WriterToken};

/// Type aliases for algorithm-specific trie variants
pub type PatriciaTrie = ZiporaTrie;
pub type CritBitTrie = ZiporaTrie;
