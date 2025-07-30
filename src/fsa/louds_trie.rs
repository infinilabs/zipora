//! LOUDS (Level-Order Unary Degree Sequence) Trie implementation
//!
//! This module provides a space-efficient trie implementation using the LOUDS
//! representation for the tree structure. LOUDS tries offer excellent space
//! efficiency and fast traversal while supporting all standard trie operations.

use std::collections::{VecDeque, HashMap};

use crate::error::{Result, ToplingError};
use crate::fsa::traits::{
    FiniteStateAutomaton, Trie, TrieBuilder, StateInspectable, 
    StatisticsProvider, TrieStats, PrefixIterable
};
use crate::succinct::{BitVector, RankSelect256};
use crate::{StateId, FastVec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// LOUDS Trie implementation using succinct data structures
///
/// The LOUDS (Level-Order Unary Degree Sequence) trie represents the tree structure
/// using a succinct bit vector encoding. This provides excellent space efficiency
/// while maintaining fast access times for trie operations.
///
/// # Structure
/// - `louds_bits`: LOUDS bit sequence representing tree structure
/// - `labels`: Edge labels in level order
/// - `is_final`: Bit vector marking final states
/// - `rank_select`: Rank/select structure for navigation
///
/// # Examples
///
/// ```rust
/// use infini_zip::fsa::{LoudsTrie, Trie, TrieBuilder};
///
/// let keys = vec![b"cat".to_vec(), b"car".to_vec(), b"card".to_vec()];
/// let trie = LoudsTrie::build_from_sorted(keys).unwrap();
///
/// assert!(trie.contains(b"car"));
/// assert!(trie.contains(b"card"));
/// assert!(!trie.contains(b"care"));
/// ```
#[derive(Debug, Clone)]
pub struct LoudsTrie {
    /// LOUDS bit sequence representing the tree structure
    louds_bits: BitVector,
    /// Rank-select structure for efficient navigation
    rank_select: RankSelect256,
    /// Edge labels stored in level order
    labels: FastVec<u8>,
    /// Bit vector marking final (accepting) states
    is_final: BitVector,
    /// Number of keys stored in the trie
    num_keys: usize,
}

impl LoudsTrie {
    /// Create a new empty LOUDS trie
    pub fn new() -> Self {
        let mut louds_bits = BitVector::new();
        let mut is_final = BitVector::new();
        
        // Initialize with root state
        // Root has LOUDS representation: 10 (one child, end of children)
        louds_bits.push(true).unwrap();
        louds_bits.push(false).unwrap();
        is_final.push(false).unwrap(); // Root is not final by default
        
        let rank_select = RankSelect256::new(louds_bits.clone()).unwrap();
        
        Self {
            louds_bits,
            rank_select,
            labels: FastVec::new(),
            is_final,
            num_keys: 0,
        }
    }
    
    /// Get the position in LOUDS sequence for a state
    fn state_to_louds_pos(&self, state: StateId) -> usize {
        if state == 0 {
            0 // Root is at position 0
        } else {
            // Find the position of the state's bit in LOUDS sequence
            self.rank_select.select1(state as usize).unwrap_or(0)
        }
    }
    
    /// Get the parent state of a given state
    fn parent(&self, state: StateId) -> Option<StateId> {
        if state == 0 {
            return None; // Root has no parent
        }
        
        let pos = self.state_to_louds_pos(state);
        if pos == 0 {
            return None;
        }
        
        // Find the parent by counting 1s before this position
        let parent_rank = self.rank_select.rank1(pos.saturating_sub(1));
        if parent_rank > 0 {
            Some(parent_rank as StateId - 1)
        } else {
            Some(0)
        }
    }
    
    /// Get the first child position in the labels array
    fn first_child_label_pos(&self, state: StateId) -> usize {
        if state == 0 {
            0
        } else {
            // Count the number of edges before this state
            let pos = self.state_to_louds_pos(state);
            self.rank_select.rank0(pos)
        }
    }
    
    /// Get the number of children for a state
    fn child_count(&self, state: StateId) -> usize {
        let start_pos = self.state_to_louds_pos(state);
        let mut count = 0;
        
        // Count consecutive 1s starting from start_pos
        for i in start_pos..self.louds_bits.len() {
            if let Some(bit) = self.louds_bits.get(i) {
                if bit {
                    count += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        count
    }
    
    /// Rebuild the rank-select structure after modifications
    fn rebuild_rank_select(&mut self) -> Result<()> {
        self.rank_select = RankSelect256::new(self.louds_bits.clone())?;
        Ok(())
    }
    
    /// Navigate to a child state with the given label
    fn goto_child(&self, state: StateId, label: u8) -> Option<StateId> {
        let child_count = self.child_count(state);
        if child_count == 0 {
            return None;
        }
        
        let first_label_pos = self.first_child_label_pos(state);
        
        // Linear search through children (could be optimized with binary search)
        for i in 0..child_count {
            if let Some(&child_label) = self.labels.get(first_label_pos + i) {
                if child_label == label {
                    // Calculate child state ID
                    let louds_pos = self.state_to_louds_pos(state) + i + 1;
                    return Some(self.rank_select.rank1(louds_pos) as StateId);
                }
            }
        }
        
        None
    }
    
    /// Add a new state as a child of the given parent with the specified label
    fn add_child(&mut self, parent: StateId, label: u8, is_final: bool) -> Result<StateId> {
        // Find insertion point in LOUDS sequence
        let parent_pos = self.state_to_louds_pos(parent);
        let child_count = self.child_count(parent);
        
        // Insert new 1 bit for the new child
        let insert_pos = parent_pos + child_count;
        self.louds_bits.insert(insert_pos, true)?;
        
        // If this is the first child, we need to add a terminating 0
        if child_count == 0 {
            self.louds_bits.insert(insert_pos + 1, false)?;
        }
        
        // Insert the label
        let label_pos = self.first_child_label_pos(parent) + child_count;
        self.labels.insert(label_pos, label)?;
        
        // Add final state marker
        let new_state_id = self.is_final.len() as StateId;
        self.is_final.push(is_final)?;
        
        // Rebuild rank-select structure
        self.rebuild_rank_select()?;
        
        if is_final {
            self.num_keys += 1;
        }
        
        Ok(new_state_id)
    }
}

impl Default for LoudsTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl FiniteStateAutomaton for LoudsTrie {
    fn root(&self) -> StateId {
        0
    }
    
    fn is_final(&self, state: StateId) -> bool {
        self.is_final.get(state as usize).unwrap_or(false)
    }
    
    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        self.goto_child(state, symbol)
    }
    
    fn transitions(&self, state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
        let child_count = self.child_count(state);
        let first_label_pos = self.first_child_label_pos(state);
        
        let iter = (0..child_count).filter_map(move |i| {
            if let Some(&label) = self.labels.get(first_label_pos + i) {
                let child_state = self.goto_child(state, label)?;
                Some((label, child_state))
            } else {
                None
            }
        });
        
        Box::new(iter)
    }
}

impl Trie for LoudsTrie {
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        let mut state = self.root();
        
        // Traverse as far as possible
        let mut i = 0;
        while i < key.len() {
            if let Some(next_state) = self.transition(state, key[i]) {
                state = next_state;
                i += 1;
            } else {
                break;
            }
        }
        
        // Add remaining suffix
        while i < key.len() {
            let is_final = i == key.len() - 1;
            state = self.add_child(state, key[i], is_final)?;
            i += 1;
        }
        
        // Mark final state if we traversed the entire key
        if i == key.len() && !self.is_final(state) {
            // Update the final state marker
            if let Some(mut final_bit) = self.is_final.get_mut(state as usize) {
                if !final_bit.get() {
                    final_bit.set(true)?;
                    self.num_keys += 1;
                }
            }
        }
        
        Ok(state)
    }
    
    fn len(&self) -> usize {
        self.num_keys
    }
}

impl StateInspectable for LoudsTrie {
    fn out_degree(&self, state: StateId) -> usize {
        self.child_count(state)
    }
    
    fn out_symbols(&self, state: StateId) -> Vec<u8> {
        let child_count = self.child_count(state);
        let first_label_pos = self.first_child_label_pos(state);
        
        (0..child_count)
            .filter_map(|i| self.labels.get(first_label_pos + i).copied())
            .collect()
    }
}

impl StatisticsProvider for LoudsTrie {
    fn stats(&self) -> TrieStats {
        let louds_memory = self.louds_bits.len() / 8 + 1; // Approximate
        let labels_memory = self.labels.len();
        let final_memory = self.is_final.len() / 8 + 1; // Approximate
        let rank_select_memory = 256; // Approximate overhead
        
        let memory_usage = louds_memory + labels_memory + final_memory + rank_select_memory;
        
        let mut stats = TrieStats {
            num_states: self.is_final.len(),
            num_keys: self.num_keys,
            num_transitions: self.labels.len(),
            max_depth: 0, // Would require traversal to calculate
            avg_depth: 0.0, // Would require traversal to calculate
            memory_usage,
            bits_per_key: 0.0,
        };
        
        stats.calculate_bits_per_key();
        stats
    }
}

/// Builder for constructing LOUDS tries from sorted key sequences
pub struct LoudsTrieBuilder;

impl TrieBuilder<LoudsTrie> for LoudsTrieBuilder {
    fn build_from_sorted<I>(keys: I) -> Result<LoudsTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut trie = LoudsTrie::new();
        
        for key in keys {
            trie.insert(&key)?;
        }
        
        Ok(trie)
    }
}

/// Iterator for prefix enumeration in LOUDS tries
pub struct LoudsTriePrefixIterator<'a> {
    trie: &'a LoudsTrie,
    stack: VecDeque<(StateId, Vec<u8>)>,
}

impl<'a> LoudsTriePrefixIterator<'a> {
    fn new(trie: &'a LoudsTrie, prefix: &[u8]) -> Option<Self> {
        let mut state = trie.root();
        
        // Navigate to prefix state
        for &symbol in prefix {
            state = trie.transition(state, symbol)?;
        }
        
        let mut stack = VecDeque::new();
        stack.push_back((state, prefix.to_vec()));
        
        Some(Self { trie, stack })
    }
}

impl<'a> Iterator for LoudsTriePrefixIterator<'a> {
    type Item = Vec<u8>;
    
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((state, path)) = self.stack.pop_front() {
            // If this is a final state, yield the path
            let is_final = self.trie.is_final(state);
            
            // Add children to stack for future exploration
            for (symbol, child_state) in self.trie.transitions(state) {
                let mut child_path = path.clone();
                child_path.push(symbol);
                self.stack.push_back((child_state, child_path));
            }
            
            if is_final {
                return Some(path);
            }
        }
        
        None
    }
}

impl PrefixIterable for LoudsTrie {
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        match LoudsTriePrefixIterator::new(self, prefix) {
            Some(iter) => Box::new(iter),
            None => Box::new(std::iter::empty()),
        }
    }
}

// Implement builder as associated function
impl LoudsTrie {
    /// Build a LOUDS trie from a sorted iterator of keys
    pub fn build_from_sorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        LoudsTrieBuilder::build_from_sorted(keys)
    }
    
    /// Build a LOUDS trie from an unsorted iterator of keys
    pub fn build_from_unsorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        LoudsTrieBuilder::build_from_unsorted(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsa::traits::{Trie, FiniteStateAutomaton, StateInspectable};

    #[test]
    fn test_louds_trie_basic_operations() {
        let mut trie = LoudsTrie::new();
        
        assert!(trie.is_empty());
        assert_eq!(trie.root(), 0);
        
        // Insert some keys
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        
        assert_eq!(trie.len(), 3);
        assert!(!trie.is_empty());
        
        // Test lookups
        assert!(trie.contains(b"cat"));
        assert!(trie.contains(b"car"));
        assert!(trie.contains(b"card"));
        assert!(!trie.contains(b"ca"));
        assert!(!trie.contains(b"care"));
        assert!(!trie.contains(b"dog"));
    }

    #[test]
    fn test_louds_trie_transitions() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();
        
        let root = trie.root();
        let a_state = trie.transition(root, b'a').unwrap();
        let b_state = trie.transition(a_state, b'b').unwrap();
        
        // Should have two children: 'c' and 'd'
        assert_eq!(trie.out_degree(b_state), 2);
        let symbols = trie.out_symbols(b_state);
        assert!(symbols.contains(&b'c'));
        assert!(symbols.contains(&b'd'));
        
        let c_state = trie.transition(b_state, b'c').unwrap();
        let d_state = trie.transition(b_state, b'd').unwrap();
        
        assert!(trie.is_final(c_state));
        assert!(trie.is_final(d_state));
    }

    #[test]
    fn test_louds_trie_accepts() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"hello").unwrap();
        trie.insert(b"world").unwrap();
        
        assert!(trie.accepts(b"hello"));
        assert!(trie.accepts(b"world"));
        assert!(!trie.accepts(b"hell"));
        assert!(!trie.accepts(b"worlds"));
        assert!(!trie.accepts(b"foo"));
    }

    #[test]
    fn test_louds_trie_longest_prefix() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        trie.insert(b"care").unwrap();
        
        assert_eq!(trie.longest_prefix(b"car"), Some(3));
        assert_eq!(trie.longest_prefix(b"card"), Some(4));
        assert_eq!(trie.longest_prefix(b"cards"), Some(4));
        assert_eq!(trie.longest_prefix(b"caring"), Some(4));
        assert_eq!(trie.longest_prefix(b"cat"), None);
    }

    #[test]
    fn test_louds_trie_builder() {
        let keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
        ];
        
        let trie = LoudsTrie::build_from_sorted(keys.clone()).unwrap();
        assert_eq!(trie.len(), 4);
        
        for key in &keys {
            assert!(trie.contains(key));
        }
        
        // Test with unsorted keys
        let mut unsorted_keys = keys.clone();
        unsorted_keys.reverse();
        
        let trie2 = LoudsTrie::build_from_unsorted(unsorted_keys).unwrap();
        assert_eq!(trie2.len(), 4);
        
        for key in &keys {
            assert!(trie2.contains(key));
        }
    }

    #[test]
    fn test_louds_trie_prefix_iteration() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        trie.insert(b"care").unwrap();
        trie.insert(b"cat").unwrap();
        
        // Test prefix "car"
        let mut car_results: Vec<Vec<u8>> = trie.iter_prefix(b"car").collect();
        car_results.sort();
        
        let expected = vec![b"car".to_vec(), b"card".to_vec(), b"care".to_vec()];
        assert_eq!(car_results, expected);
        
        // Test prefix "ca"
        let mut ca_results: Vec<Vec<u8>> = trie.iter_prefix(b"ca").collect();
        ca_results.sort();
        
        let expected = vec![
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
            b"cat".to_vec(),
        ];
        assert_eq!(ca_results, expected);
        
        // Test non-existent prefix (should return empty iterator)
        let dog_results: Vec<Vec<u8>> = trie.iter_prefix(b"dog").collect();
        assert!(dog_results.is_empty());
    }

    #[test]
    fn test_louds_trie_empty_key() {
        let mut trie = LoudsTrie::new();
        
        // Insert empty key
        trie.insert(b"").unwrap();
        assert_eq!(trie.len(), 1);
        assert!(trie.contains(b""));
        
        // Root should now be final
        assert!(trie.is_final(trie.root()));
    }

    #[test]
    fn test_louds_trie_duplicate_keys() {
        let mut trie = LoudsTrie::new();
        
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1);
        
        // Insert the same key again
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1); // Should not increase
        
        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_louds_trie_statistics() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();
        
        let stats = trie.stats();
        assert_eq!(stats.num_keys, 3);
        assert!(stats.memory_usage > 0);
        assert!(stats.bits_per_key > 0.0);
    }

    #[test]
    fn test_louds_trie_large_keys() {
        let mut trie = LoudsTrie::new();
        
        // Test with longer keys
        let long_key = b"this_is_a_very_long_key_for_testing_purposes";
        trie.insert(long_key).unwrap();
        
        assert!(trie.contains(long_key));
        assert_eq!(trie.len(), 1);
        
        // Test prefix of long key
        let prefix = b"this_is_a_very";
        assert_eq!(trie.longest_prefix(long_key), Some(long_key.len()));
        assert_eq!(trie.longest_prefix(prefix), None);
    }

    #[test]
    fn test_louds_trie_transitions_iterator() {
        let mut trie = LoudsTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();
        trie.insert(b"ac").unwrap();
        
        let root = trie.root();
        let a_state = trie.transition(root, b'a').unwrap();
        
        // Should have transitions to 'b' and 'c'
        let transitions: Vec<(u8, StateId)> = trie.transitions(a_state).collect();
        assert_eq!(transitions.len(), 2);
        
        let symbols: Vec<u8> = transitions.iter().map(|(s, _)| *s).collect();
        assert!(symbols.contains(&b'b'));
        assert!(symbols.contains(&b'c'));
    }
}