//! Core FSA (Finite State Automaton) traits and abstractions
//!
//! This module defines the fundamental traits for various automaton types
//! including tries, DAWGs, and other finite state structures.

use crate::StateId;
use crate::error::Result;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Core trait for finite state automaton operations
pub trait FiniteStateAutomaton {
    /// Get the initial/root state
    fn root(&self) -> StateId;

    /// Check if a state is final (accepting)
    fn is_final(&self, state: StateId) -> bool;

    /// Transition from a state given an input symbol
    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId>;

    /// Get all possible transitions from a state
    fn transitions(&self, state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_>;

    /// Check if the automaton accepts a given input sequence
    fn accepts(&self, input: &[u8]) -> bool {
        let mut state = self.root();
        for &symbol in input {
            match self.transition(state, symbol) {
                Some(next_state) => state = next_state,
                None => return false,
            }
        }
        self.is_final(state)
    }

    /// Find the longest prefix of input that leads to a final state
    fn longest_prefix(&self, input: &[u8]) -> Option<usize> {
        let mut state = self.root();
        let mut last_final = None;

        for (i, &symbol) in input.iter().enumerate() {
            if self.is_final(state) {
                last_final = Some(i);
            }

            match self.transition(state, symbol) {
                Some(next_state) => state = next_state,
                None => return last_final,
            }
        }

        // Check final state after consuming all input
        if self.is_final(state) {
            Some(input.len())
        } else {
            last_final
        }
    }
}

/// Trait for automata that support prefix-based iteration
pub trait PrefixIterable: FiniteStateAutomaton {
    /// Get an iterator over all strings with the given prefix
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_>;

    /// Get an iterator over all accepted strings (empty prefix)
    fn iter_all(&self) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        self.iter_prefix(&[])
    }
}

/// Trait for trie data structures
pub trait Trie: FiniteStateAutomaton {
    /// Insert a key into the trie and return its state ID
    ///
    /// # Arguments
    /// * `key` - The key to insert
    ///
    /// # Returns
    /// * The state ID representing the inserted key
    fn insert(&mut self, key: &[u8]) -> Result<StateId>;

    /// Look up a key in the trie
    ///
    /// # Arguments
    /// * `key` - The key to look up
    ///
    /// # Returns
    /// * `Some(StateId)` if the key exists, `None` otherwise
    fn lookup(&self, key: &[u8]) -> Option<StateId> {
        let mut state = self.root();
        for &symbol in key {
            state = self.transition(state, symbol)?;
        }
        if self.is_final(state) {
            Some(state)
        } else {
            None
        }
    }

    /// Check if a key exists in the trie
    fn contains(&self, key: &[u8]) -> bool {
        self.lookup(key).is_some()
    }

    /// Get the number of keys in the trie
    fn len(&self) -> usize;

    /// Check if the trie is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for immutable trie construction
pub trait TrieBuilder<T: Trie> {
    /// Build a trie from a sorted iterator of keys
    ///
    /// # Arguments
    /// * `keys` - Iterator over sorted keys
    ///
    /// # Returns
    /// * The constructed trie
    fn build_from_sorted<I>(keys: I) -> Result<T>
    where
        I: IntoIterator<Item = Vec<u8>>;

    /// Build a trie from an unsorted iterator of keys
    ///
    /// # Arguments
    /// * `keys` - Iterator over keys (will be sorted internally)
    ///
    /// # Returns
    /// * The constructed trie
    fn build_from_unsorted<I>(keys: I) -> Result<T>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut sorted_keys: Vec<Vec<u8>> = keys.into_iter().collect();
        sorted_keys.sort();
        sorted_keys.dedup();
        Self::build_from_sorted(sorted_keys)
    }
}

/// Trait for automata that support state inspection
pub trait StateInspectable: FiniteStateAutomaton {
    /// Get the outgoing degree (number of transitions) from a state
    fn out_degree(&self, state: StateId) -> usize;

    /// Get all outgoing symbols from a state
    fn out_symbols(&self, state: StateId) -> Vec<u8>;

    /// Check if a state has any outgoing transitions
    fn is_leaf(&self, state: StateId) -> bool {
        self.out_degree(state) == 0
    }
}

/// Statistics about trie structure and performance
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TrieStats {
    /// Number of states in the trie
    pub num_states: usize,
    /// Number of keys stored
    pub num_keys: usize,
    /// Total number of transitions
    pub num_transitions: usize,
    /// Maximum depth of any key
    pub max_depth: usize,
    /// Average depth of keys
    pub avg_depth: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Space efficiency (bits per key)
    pub bits_per_key: f64,
}

impl TrieStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate bits per key
    pub fn calculate_bits_per_key(&mut self) {
        if self.num_keys > 0 {
            self.bits_per_key = (self.memory_usage * 8) as f64 / self.num_keys as f64;
        }
    }

    /// Calculate average depth
    pub fn calculate_avg_depth(&mut self, total_depth: usize) {
        if self.num_keys > 0 {
            self.avg_depth = total_depth as f64 / self.num_keys as f64;
        }
    }
}

/// Trait for automata that provide performance statistics
pub trait StatisticsProvider {
    /// Get detailed statistics about the trie
    fn stats(&self) -> TrieStats;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize {
        self.stats().memory_usage
    }

    /// Get space efficiency in bits per key
    fn bits_per_key(&self) -> f64 {
        self.stats().bits_per_key
    }
}

/// Error types specific to FSA operations
#[derive(Debug, Clone)]
pub enum FsaError {
    /// Invalid state ID
    InvalidState(StateId),
    /// Invalid symbol for transition
    InvalidSymbol(u8),
    /// Trie construction failed
    ConstructionFailed(String),
    /// Operation not supported
    NotSupported(String),
}

impl std::fmt::Display for FsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FsaError::InvalidState(state) => write!(f, "Invalid state ID: {}", state),
            FsaError::InvalidSymbol(symbol) => write!(f, "Invalid symbol: {}", symbol),
            FsaError::ConstructionFailed(msg) => write!(f, "Trie construction failed: {}", msg),
            FsaError::NotSupported(op) => write!(f, "Operation not supported: {}", op),
        }
    }
}

impl std::error::Error for FsaError {}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing
    struct MockTrie {
        keys: std::collections::HashSet<Vec<u8>>,
    }

    impl MockTrie {
        fn new() -> Self {
            Self {
                keys: std::collections::HashSet::new(),
            }
        }
    }

    impl FiniteStateAutomaton for MockTrie {
        fn root(&self) -> StateId {
            0
        }

        fn is_final(&self, _state: StateId) -> bool {
            true // Simplified for testing
        }

        fn transition(&self, _state: StateId, _symbol: u8) -> Option<StateId> {
            Some(1) // Simplified for testing
        }

        fn transitions(&self, _state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
            Box::new(std::iter::empty())
        }
    }

    impl Trie for MockTrie {
        fn insert(&mut self, key: &[u8]) -> Result<StateId> {
            self.keys.insert(key.to_vec());
            Ok(1)
        }

        fn lookup(&self, key: &[u8]) -> Option<StateId> {
            if self.keys.contains(key) {
                Some(1)
            } else {
                None
            }
        }

        fn len(&self) -> usize {
            self.keys.len()
        }
    }

    #[test]
    fn test_trie_basic_operations() {
        let mut trie = MockTrie::new();

        assert!(trie.is_empty());

        trie.insert(b"hello").unwrap();
        trie.insert(b"world").unwrap();

        assert_eq!(trie.len(), 2);
        assert!(!trie.is_empty());

        assert!(trie.contains(b"hello"));
        assert!(trie.contains(b"world"));
        assert!(!trie.contains(b"foo"));
    }

    #[test]
    fn test_fsa_accepts() {
        let trie = MockTrie::new();

        // With our simplified mock, everything is accepted
        assert!(trie.accepts(b"anything"));
    }

    #[test]
    fn test_trie_stats() {
        let mut stats = TrieStats::new();
        stats.num_keys = 100;
        stats.memory_usage = 1024;

        stats.calculate_bits_per_key();
        assert!((stats.bits_per_key - 81.92).abs() < 0.01);

        stats.calculate_avg_depth(500);
        assert!((stats.avg_depth - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_fsa_error_display() {
        let error = FsaError::InvalidState(42);
        assert_eq!(error.to_string(), "Invalid state ID: 42");

        let error = FsaError::InvalidSymbol(65);
        assert_eq!(error.to_string(), "Invalid symbol: 65");

        let error = FsaError::ConstructionFailed("test".to_string());
        assert_eq!(error.to_string(), "Trie construction failed: test");

        let error = FsaError::NotSupported("test op".to_string());
        assert_eq!(error.to_string(), "Operation not supported: test op");
    }
}
