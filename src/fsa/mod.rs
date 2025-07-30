//! Finite State Automata and Trie structures
//!
//! This module provides various trie implementations including LOUDS tries,
//! compressed sparse Patricia tries, and basic FSA interfaces.

pub mod traits;
pub mod louds_trie;

// Re-export core types
pub use traits::{
    FiniteStateAutomaton, Trie, TrieBuilder, StateInspectable,
    StatisticsProvider, TrieStats, PrefixIterable, FsaError
};
pub use louds_trie::{LoudsTrie, LoudsTrieBuilder, LoudsTriePrefixIterator};