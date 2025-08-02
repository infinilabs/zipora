//! Critical-Bit Trie implementation
//!
//! This module provides a space-efficient trie implementation using the critical-bit
//! (radix) tree approach. Critical-bit tries are particularly efficient for string
//! keys by compressing common prefixes and using binary decisions at each node.
//!
//! Critical-bit tries offer excellent performance for:
//! - Longest prefix matching
//! - Ordered iteration
//! - Space efficiency for sparse key sets
//! - Fast insertion and lookup operations

use crate::error::Result;
use crate::fsa::traits::{
    FiniteStateAutomaton, PrefixIterable, StateInspectable, StatisticsProvider, Trie, TrieBuilder,
    TrieStats,
};
use crate::{FastVec, StateId};

/// Node in a critical-bit trie
#[derive(Debug, Clone)]
struct CritBitNode {
    /// The byte position that differentiates the two subtrees
    crit_byte: usize,
    /// The bit position within the critical byte (0-7)
    crit_bit: u8,
    /// Left child node index (bit = 0)
    left_child: Option<usize>,
    /// Right child node index (bit = 1)  
    right_child: Option<usize>,
    /// The key stored at this node (for leaves)
    key: Option<FastVec<u8>>,
    /// Whether this node represents a complete key
    is_final: bool,
}

impl CritBitNode {
    /// Create a new internal node
    fn new_internal(crit_byte: usize, crit_bit: u8) -> Self {
        Self {
            crit_byte,
            crit_bit,
            left_child: None,
            right_child: None,
            key: None,
            is_final: false,
        }
    }

    /// Create a new leaf node
    fn new_leaf(key: FastVec<u8>, is_final: bool) -> Self {
        Self {
            crit_byte: 0,
            crit_bit: 0,
            left_child: None,
            right_child: None,
            key: Some(key),
            is_final,
        }
    }

    /// Check if this is a leaf node
    fn is_leaf(&self) -> bool {
        self.left_child.is_none() && self.right_child.is_none()
    }

    /// Get the child based on bit value
    fn get_child(&self, bit: bool) -> Option<usize> {
        if bit {
            self.right_child
        } else {
            self.left_child
        }
    }

    /// Set the child based on bit value
    fn set_child(&mut self, bit: bool, child: usize) {
        if bit {
            self.right_child = Some(child);
        } else {
            self.left_child = Some(child);
        }
    }
}

/// Critical-Bit Trie implementation
///
/// A critical-bit trie is a compressed trie that stores only the critical
/// bits needed to distinguish between keys. This provides excellent space
/// efficiency and fast lookup times.
///
/// # Structure
/// - Nodes are stored in a vector for cache efficiency
/// - Each internal node stores a critical bit position
/// - Leaves store the complete keys
/// - The tree maintains lexicographic order
///
/// # Examples
///
/// ```rust
/// use infini_zip::fsa::{CritBitTrie, Trie};
///
/// let mut trie = CritBitTrie::new();
/// trie.insert(b"hello").unwrap();
/// trie.insert(b"help").unwrap();
/// trie.insert(b"world").unwrap();
///
/// assert!(trie.contains(b"hello"));
/// assert!(trie.contains(b"help"));
/// assert!(!trie.contains(b"he"));
/// ```
#[derive(Debug, Clone)]
pub struct CritBitTrie {
    /// Vector of nodes for cache-efficient storage
    nodes: Vec<CritBitNode>,
    /// Index of the root node
    root: Option<usize>,
    /// Number of keys stored in the trie
    num_keys: usize,
}

impl CritBitTrie {
    /// Create a new empty critical-bit trie
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
            num_keys: 0,
        }
    }

    /// Find the critical bit position between two keys
    fn find_critical_bit(key1: &[u8], key2: &[u8]) -> (usize, u8) {
        let mut byte_pos = 0;
        let min_len = key1.len().min(key2.len());

        // Find first differing byte
        while byte_pos < min_len && key1[byte_pos] == key2[byte_pos] {
            byte_pos += 1;
        }

        // If one key is a prefix of the other
        if byte_pos == min_len {
            if key1.len() != key2.len() {
                // Different lengths - use the position where they differ in length
                // The critical bit will be at the first byte position where the shorter string ends
                return (min_len, 8); // Use bit 8 to indicate end-of-string vs continuation
            } else {
                // Same keys - shouldn't happen in practice
                return (byte_pos, 0);
            }
        }

        // Find the critical bit within the differing byte
        let byte1 = key1[byte_pos];
        let byte2 = key2[byte_pos];
        let diff = byte1 ^ byte2;

        // Find the most significant differing bit
        let bit_pos = 7 - diff.leading_zeros() as u8;

        (byte_pos, bit_pos)
    }

    /// Test a bit in a key at the given position
    fn test_bit(key: &[u8], byte_pos: usize, bit_pos: u8) -> bool {
        if bit_pos == 8 {
            // Special case: bit position 8 represents "end-of-string"
            // Return true if we're at or beyond the end of the key (shorter key ends here)
            // Return false if the key continues beyond this position (longer key)
            byte_pos >= key.len()
        } else if byte_pos >= key.len() {
            false // Treat missing bytes as 0
        } else {
            let byte_val = key[byte_pos];
            (byte_val >> bit_pos) & 1 == 1
        }
    }

    /// Add a new node and return its index
    fn add_node(&mut self, node: CritBitNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    /// Insert a key into the trie
    fn insert_recursive(&mut self, node_idx: Option<usize>, key: &[u8]) -> Result<usize> {
        let key_vec = {
            let mut vec = FastVec::new();
            for &byte in key {
                vec.push(byte)?;
            }
            vec
        };

        // If no node exists, create a leaf
        let Some(node_idx) = node_idx else {
            let leaf = CritBitNode::new_leaf(key_vec, true);
            let idx = self.add_node(leaf);
            self.num_keys += 1;
            return Ok(idx);
        };

        // Check if this is a leaf and handle accordingly
        let (is_leaf, existing_key_opt, is_final) = {
            let node = &self.nodes[node_idx];
            (node.is_leaf(), node.key.clone(), node.is_final)
        };

        // If this is a leaf, we need to split
        if is_leaf {
            let existing_key = existing_key_opt.unwrap();
            let existing_key_slice: &[u8] = existing_key.as_slice();

            // If keys are identical, just mark as final
            if existing_key_slice == key {
                self.nodes[node_idx].is_final = true;
                if !is_final {
                    self.num_keys += 1;
                }
                return Ok(node_idx);
            }

            // Find critical bit
            let (crit_byte, crit_bit) = Self::find_critical_bit(existing_key_slice, key);

            // Create new internal node
            let mut internal = CritBitNode::new_internal(crit_byte, crit_bit);

            // Create new leaf for the new key
            let new_leaf = CritBitNode::new_leaf(key_vec, true);
            let new_leaf_idx = self.add_node(new_leaf);

            // Determine which side each key goes on
            let existing_bit = Self::test_bit(existing_key_slice, crit_byte, crit_bit);
            let new_bit = Self::test_bit(key, crit_byte, crit_bit);

            internal.set_child(existing_bit, node_idx);
            internal.set_child(new_bit, new_leaf_idx);

            let internal_idx = self.add_node(internal);
            self.num_keys += 1;

            Ok(internal_idx)
        } else {
            // Navigate down the tree
            let (crit_byte, crit_bit, child_idx) = {
                let node = &self.nodes[node_idx];
                let bit = Self::test_bit(key, node.crit_byte, node.crit_bit);
                let child_idx = node.get_child(bit);
                (node.crit_byte, node.crit_bit, child_idx)
            };

            let bit = Self::test_bit(key, crit_byte, crit_bit);
            let new_child_idx = self.insert_recursive(child_idx, key)?;

            // Update the child pointer if it changed
            if child_idx != Some(new_child_idx) {
                self.nodes[node_idx].set_child(bit, new_child_idx);
            }

            Ok(node_idx)
        }
    }

    /// Find a key in the trie
    fn find_node(&self, key: &[u8]) -> Option<usize> {
        let mut current = self.root?;

        loop {
            let node = &self.nodes[current];

            if node.is_leaf() {
                let stored_key = node.key.as_ref()?;
                if stored_key.as_slice() == key && node.is_final {
                    return Some(current);
                } else {
                    return None;
                }
            }

            // Navigate to child based on critical bit
            let bit = Self::test_bit(key, node.crit_byte, node.crit_bit);
            current = node.get_child(bit)?;
        }
    }

    /// Get all keys with a given prefix
    fn collect_keys_with_prefix(&self, node_idx: usize, prefix: &[u8], results: &mut Vec<Vec<u8>>) {
        let node = &self.nodes[node_idx];

        if node.is_leaf() {
            if let Some(key) = &node.key {
                let key_slice = key.as_slice();
                if key_slice.starts_with(prefix) && node.is_final {
                    results.push(key_slice.to_vec());
                }
            }
            return;
        }

        // Check both children
        if let Some(left_idx) = node.left_child {
            self.collect_keys_with_prefix(left_idx, prefix, results);
        }
        if let Some(right_idx) = node.right_child {
            self.collect_keys_with_prefix(right_idx, prefix, results);
        }
    }

    /// Calculate the maximum depth of the trie
    fn calculate_max_depth(&self, node_idx: usize, current_depth: usize) -> usize {
        let node = &self.nodes[node_idx];

        if node.is_leaf() {
            return current_depth;
        }

        let mut max_depth = current_depth;

        if let Some(left_idx) = node.left_child {
            max_depth = max_depth.max(self.calculate_max_depth(left_idx, current_depth + 1));
        }
        if let Some(right_idx) = node.right_child {
            max_depth = max_depth.max(self.calculate_max_depth(right_idx, current_depth + 1));
        }

        max_depth
    }
}

impl Default for CritBitTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl FiniteStateAutomaton for CritBitTrie {
    fn root(&self) -> StateId {
        self.root.unwrap_or(0) as StateId
    }

    fn is_final(&self, state: StateId) -> bool {
        if let Some(node) = self.nodes.get(state as usize) {
            node.is_final
        } else {
            false
        }
    }

    fn transition(&self, state: StateId, _symbol: u8) -> Option<StateId> {
        let node = self.nodes.get(state as usize)?;

        if node.is_leaf() {
            return None; // Leaves have no transitions
        }

        // For internal nodes, we can't directly transition by symbol
        // This is a limitation of the critical-bit representation
        // We'd need to modify the interface or use a different approach
        None
    }

    fn transitions(&self, _state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
        // Critical-bit tries don't have direct symbol transitions
        // This is a fundamental mismatch with the FSA interface
        Box::new(std::iter::empty())
    }
}

impl Trie for CritBitTrie {
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        if self.root.is_none() {
            let mut key_vec = FastVec::new();
            for &byte in key {
                key_vec.push(byte)?;
            }
            let leaf = CritBitNode::new_leaf(key_vec, true);
            let idx = self.add_node(leaf);
            self.root = Some(idx);
            self.num_keys += 1;
            Ok(idx as StateId)
        } else {
            let new_root = self.insert_recursive(self.root, key)?;
            self.root = Some(new_root);
            Ok(new_root as StateId)
        }
    }

    fn lookup(&self, key: &[u8]) -> Option<StateId> {
        self.find_node(key).map(|idx| idx as StateId)
    }

    fn len(&self) -> usize {
        self.num_keys
    }
}

impl StateInspectable for CritBitTrie {
    fn out_degree(&self, state: StateId) -> usize {
        if let Some(node) = self.nodes.get(state as usize) {
            if node.is_leaf() {
                0
            } else {
                let mut degree = 0;
                if node.left_child.is_some() {
                    degree += 1;
                }
                if node.right_child.is_some() {
                    degree += 1;
                }
                degree
            }
        } else {
            0
        }
    }

    fn out_symbols(&self, _state: StateId) -> Vec<u8> {
        // Critical-bit tries don't have direct symbol transitions
        Vec::new()
    }
}

impl StatisticsProvider for CritBitTrie {
    fn stats(&self) -> TrieStats {
        let memory_usage = self.nodes.len() * std::mem::size_of::<CritBitNode>();
        let max_depth = if let Some(root) = self.root {
            self.calculate_max_depth(root, 0)
        } else {
            0
        };

        let mut stats = TrieStats {
            num_states: self.nodes.len(),
            num_keys: self.num_keys,
            num_transitions: 0, // Not applicable for critical-bit tries
            max_depth,
            avg_depth: 0.0, // Would require full traversal to calculate
            memory_usage,
            bits_per_key: 0.0,
        };

        stats.calculate_bits_per_key();
        stats
    }
}

/// Builder for constructing critical-bit tries from sorted key sequences
pub struct CritBitTrieBuilder;

impl TrieBuilder<CritBitTrie> for CritBitTrieBuilder {
    fn build_from_sorted<I>(keys: I) -> Result<CritBitTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut trie = CritBitTrie::new();

        for key in keys {
            trie.insert(&key)?;
        }

        Ok(trie)
    }

    fn build_from_unsorted<I>(keys: I) -> Result<CritBitTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut sorted_keys: Vec<Vec<u8>> = keys.into_iter().collect();
        sorted_keys.sort();
        sorted_keys.dedup();
        Self::build_from_sorted(sorted_keys)
    }
}

/// Iterator for prefix enumeration in critical-bit tries
pub struct CritBitTriePrefixIterator {
    results: Vec<Vec<u8>>,
    index: usize,
}

impl Iterator for CritBitTriePrefixIterator {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.results.len() {
            let result = self.results[self.index].clone();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

impl PrefixIterable for CritBitTrie {
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        let mut results = Vec::new();

        if let Some(root) = self.root {
            self.collect_keys_with_prefix(root, prefix, &mut results);
        }

        results.sort(); // Maintain lexicographic order

        Box::new(CritBitTriePrefixIterator { results, index: 0 })
    }
}

// Implement builder as associated function
impl CritBitTrie {
    /// Build a critical-bit trie from a sorted iterator of keys
    pub fn build_from_sorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        CritBitTrieBuilder::build_from_sorted(keys)
    }

    /// Build a critical-bit trie from an unsorted iterator of keys
    pub fn build_from_unsorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        CritBitTrieBuilder::build_from_unsorted(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsa::traits::{PrefixIterable, Trie};

    #[test]
    fn test_crit_bit_trie_basic_operations() {
        let mut trie = CritBitTrie::new();

        assert!(trie.is_empty());

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
    fn test_crit_bit_trie_critical_bit_calculation() {
        // Test critical bit finding
        let (byte_pos, _bit_pos) = CritBitTrie::find_critical_bit(b"cat", b"car");
        assert_eq!(byte_pos, 2); // Third byte differs

        let (byte_pos, _bit_pos) = CritBitTrie::find_critical_bit(b"hello", b"help");
        assert_eq!(byte_pos, 3); // Fourth byte differs

        let (byte_pos, _bit_pos) = CritBitTrie::find_critical_bit(b"a", b"ab");
        assert_eq!(byte_pos, 1); // Length difference
    }

    #[test]
    fn test_crit_bit_trie_prefix_iteration() {
        let mut trie = CritBitTrie::new();
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
    }

    #[test]
    fn test_crit_bit_trie_duplicate_keys() {
        let mut trie = CritBitTrie::new();

        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1);

        // Insert the same key again
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1); // Should not increase

        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_crit_bit_trie_empty_key() {
        let mut trie = CritBitTrie::new();

        // Insert empty key
        trie.insert(b"").unwrap();
        assert_eq!(trie.len(), 1);
        assert!(trie.contains(b""));
    }

    #[test]
    fn test_crit_bit_trie_builder() {
        let keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
        ];

        let trie = CritBitTrie::build_from_sorted(keys.clone()).unwrap();
        assert_eq!(trie.len(), 4);

        for key in &keys {
            assert!(trie.contains(key));
        }

        // Test with unsorted keys
        let mut unsorted_keys = keys.clone();
        unsorted_keys.reverse();

        let trie2 = CritBitTrie::build_from_unsorted(unsorted_keys).unwrap();
        assert_eq!(trie2.len(), 4);

        for key in &keys {
            assert!(trie2.contains(key));
        }
    }

    #[test]
    fn test_crit_bit_trie_statistics() {
        let mut trie = CritBitTrie::new();
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();

        let stats = trie.stats();
        assert_eq!(stats.num_keys, 3);
        assert!(stats.memory_usage > 0);
        assert!(stats.max_depth > 0);
    }

    #[test]
    fn test_crit_bit_trie_large_keys() {
        let mut trie = CritBitTrie::new();

        // Test with longer keys
        let long_key = b"this_is_a_very_long_key_for_testing_purposes";
        trie.insert(long_key).unwrap();

        assert!(trie.contains(long_key));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn test_crit_bit_trie_bit_testing() {
        // Test bit testing function
        let key = b"hello";

        // Test various bit positions
        assert!(!CritBitTrie::test_bit(key, 0, 7)); // 'h' = 0x68, bit 7 = 0
        assert!(CritBitTrie::test_bit(key, 0, 6)); // 'h' = 0x68, bit 6 = 1

        // Test out of bounds
        assert!(!CritBitTrie::test_bit(key, 10, 0)); // Beyond key length
    }

    #[test]
    fn test_crit_bit_trie_prefix_bug_fix() {
        // Test the specific bug case: inserting "cat", "car", "card"
        let mut trie = CritBitTrie::new();

        // Insert in the problematic order
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();

        // All three should be found
        assert!(trie.contains(b"cat"), "cat should be found");
        assert!(
            trie.contains(b"car"),
            "car should be found after inserting card"
        );
        assert!(trie.contains(b"card"), "card should be found");

        // Count should be correct
        assert_eq!(trie.len(), 3);

        // Non-existent keys should not be found
        assert!(!trie.contains(b"ca"));
        assert!(!trie.contains(b"cars"));
        assert!(!trie.contains(b"care"));
    }

    #[test]
    fn test_crit_bit_trie_various_prefix_relationships() {
        let _trie = CritBitTrie::new();

        // Test various prefix relationships
        let test_cases = [
            (b"a".as_slice(), b"ab".as_slice()),
            (b"hello".as_slice(), b"help".as_slice()),
            (b"car".as_slice(), b"card".as_slice()),
            (b"test".as_slice(), b"testing".as_slice()),
            (b"".as_slice(), b"x".as_slice()), // empty string vs single char
        ];

        for (shorter, longer) in &test_cases {
            let mut local_trie = CritBitTrie::new();

            // Test insertion in both orders
            local_trie.insert(shorter).unwrap();
            local_trie.insert(longer).unwrap();
            assert!(
                local_trie.contains(shorter),
                "Failed to find shorter key: {:?}",
                std::str::from_utf8(shorter)
            );
            assert!(
                local_trie.contains(longer),
                "Failed to find longer key: {:?}",
                std::str::from_utf8(longer)
            );

            // Test reverse order
            let mut reverse_trie = CritBitTrie::new();
            reverse_trie.insert(longer).unwrap();
            reverse_trie.insert(shorter).unwrap();
            assert!(
                reverse_trie.contains(shorter),
                "Failed to find shorter key in reverse order: {:?}",
                std::str::from_utf8(shorter)
            );
            assert!(
                reverse_trie.contains(longer),
                "Failed to find longer key in reverse order: {:?}",
                std::str::from_utf8(longer)
            );
        }
    }

    #[test]
    fn test_crit_bit_trie_critical_bit_fix() {
        // Test the specific critical bit calculations that were problematic

        // car vs card: should use virtual end-of-string bit
        let (byte_pos, bit_pos) = CritBitTrie::find_critical_bit(b"car", b"card");
        assert_eq!(byte_pos, 3);
        assert_eq!(bit_pos, 8); // Virtual end-of-string bit

        // Verify the bit values are different
        let car_bit = CritBitTrie::test_bit(b"car", byte_pos, bit_pos);
        let card_bit = CritBitTrie::test_bit(b"card", byte_pos, bit_pos);
        assert_ne!(
            car_bit, card_bit,
            "Bits should be different to distinguish keys"
        );
        assert!(car_bit, "car should have bit=1 (end-of-string)");
        assert!(!card_bit, "card should have bit=0 (string continues)");

        // a vs ab: should use virtual end-of-string bit
        let (byte_pos, bit_pos) = CritBitTrie::find_critical_bit(b"a", b"ab");
        assert_eq!(byte_pos, 1);
        assert_eq!(bit_pos, 8); // Virtual end-of-string bit

        // Verify the bit values are different
        let a_bit = CritBitTrie::test_bit(b"a", byte_pos, bit_pos);
        let ab_bit = CritBitTrie::test_bit(b"ab", byte_pos, bit_pos);
        assert_ne!(
            a_bit, ab_bit,
            "Bits should be different to distinguish keys"
        );
        assert!(a_bit, "a should have bit=1 (end-of-string)");
        assert!(!ab_bit, "ab should have bit=0 (string continues)");
    }

    #[test]
    fn test_crit_bit_trie_zero_byte_prefix() {
        // Test case where the extra byte is 0x00
        let mut trie = CritBitTrie::new();

        let key1 = b"test";
        let key2 = b"test\x00"; // key1 + null byte

        trie.insert(key1).unwrap();
        trie.insert(key2).unwrap();

        assert!(trie.contains(key1), "key1 should be found");
        assert!(trie.contains(key2), "key2 should be found");
        assert_eq!(trie.len(), 2);

        // Test critical bit calculation for zero byte
        let (byte_pos, bit_pos) = CritBitTrie::find_critical_bit(key1, key2);
        assert_eq!(byte_pos, 4); // After "test"
        assert_eq!(bit_pos, 8); // Should use virtual end-of-string bit
    }
}
