//! Finite State Automata and Trie structures
//!
//! This module provides various trie implementations including LOUDS tries,
//! critical-bit tries, Patricia tries, and basic FSA interfaces.

pub mod traits;
pub mod louds_trie;
pub mod crit_bit_trie;
pub mod patricia_trie;

// Re-export core types
pub use traits::{
    FiniteStateAutomaton, Trie, TrieBuilder, StateInspectable,
    StatisticsProvider, TrieStats, PrefixIterable, FsaError
};
pub use louds_trie::{LoudsTrie, LoudsTrieBuilder, LoudsTriePrefixIterator};
pub use crit_bit_trie::{CritBitTrie, CritBitTrieBuilder, CritBitTriePrefixIterator};
pub use patricia_trie::{PatriciaTrie, PatriciaTrieBuilder, PatriciaTriePrefixIterator};