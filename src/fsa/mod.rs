//! Finite State Automata and Trie structures
//!
//! This module provides various trie implementations including LOUDS tries,
//! critical-bit tries, Patricia tries, compressed sparse Patricia tries,
//! and basic FSA interfaces, along with advanced infrastructure components
//! for high-performance FSA operations.

pub mod cache;
pub mod compressed_sparse_trie;
pub mod crit_bit_trie;
pub mod dawg;
pub mod double_array_trie;
pub mod fast_search;
pub mod graph_walker;
pub mod louds_trie;
pub mod nested_louds_trie;
pub mod patricia_trie;
pub mod simple_implementations;
pub mod traits;

// Re-export core types
pub use cache::{
    CacheStrategy, CachedState, FsaCache, FsaCacheConfig, FsaCacheStats, ZeroPathData,
};
pub use compressed_sparse_trie::{
    CompressedSparseTrie, ConcurrencyLevel, ReaderToken, WriterToken,
};
pub use crit_bit_trie::{CritBitTrie, CritBitTrieBuilder, CritBitTriePrefixIterator};
pub use dawg::{
    DawgConfig, DawgState, DawgStats, NestedTrieDawg, TerminalStrategy, TransitionTable,
};
pub use double_array_trie::{
    DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig, DoubleArrayTriePrefixIterator,
};
pub use fast_search::{
    FastSearchConfig, FastSearchEngine, HardwareCapabilities, SearchStrategy,
};
pub use graph_walker::{
    BfsGraphWalker, CfsGraphWalker, DfsGraphWalker, GraphVisitor, GraphWalker, GraphWalkerFactory,
    MultiPassWalker, SimpleVertex, Vertex, VertexColor, WalkMethod, WalkStats, WalkerConfig,
};
pub use louds_trie::{LoudsTrie, LoudsTrieBuilder, LoudsTriePrefixIterator};
pub use nested_louds_trie::{
    FragmentStats, NestedLoudsTrie, NestedLoudsTrieBuilder, NestedLoudsTriePrefixIterator,
    NestedTrieStats, NestingConfig, NestingConfigBuilder,
};
pub use patricia_trie::{PatriciaTrie, PatriciaTrieBuilder, PatriciaTriePrefixIterator};
pub use simple_implementations::{
    SimpleDawg, SimpleFastSearch, SimpleFsaCache, SimpleGraphWalker,
};
pub use traits::{
    FiniteStateAutomaton, FsaError, PrefixIterable, StateInspectable, StatisticsProvider, Trie,
    TrieBuilder, TrieStats,
};
