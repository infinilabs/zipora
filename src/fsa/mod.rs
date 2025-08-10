//! Finite State Automata and Trie structures
//!
//! This module provides various trie implementations including LOUDS tries,
//! critical-bit tries, Patricia tries, compressed sparse Patricia tries,
//! and basic FSA interfaces.

pub mod compressed_sparse_trie;
pub mod crit_bit_trie;
pub mod double_array_trie;
pub mod louds_trie;
pub mod nested_louds_trie;
pub mod patricia_trie;
pub mod traits;

// Re-export core types
pub use compressed_sparse_trie::{
    CompressedSparseTrie, ConcurrencyLevel, ReaderToken, WriterToken,
};
pub use crit_bit_trie::{CritBitTrie, CritBitTrieBuilder, CritBitTriePrefixIterator};
pub use double_array_trie::{
    DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig, DoubleArrayTriePrefixIterator,
};
pub use louds_trie::{LoudsTrie, LoudsTrieBuilder, LoudsTriePrefixIterator};
pub use nested_louds_trie::{
    FragmentStats, NestedLoudsTrie, NestedLoudsTrieBuilder, NestedLoudsTriePrefixIterator,
    NestedTrieStats, NestingConfig, NestingConfigBuilder,
};
pub use patricia_trie::{PatriciaTrie, PatriciaTrieBuilder, PatriciaTriePrefixIterator};
pub use traits::{
    FiniteStateAutomaton, FsaError, PrefixIterable, StateInspectable, StatisticsProvider, Trie,
    TrieBuilder, TrieStats,
};
