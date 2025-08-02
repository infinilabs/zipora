//! Patricia Trie implementation
//!
//! This module provides a Patricia trie (Practical Algorithm to Retrieve Information
//! Coded in Alphanumeric) implementation. Patricia tries are compressed prefix trees
//! that eliminate single-child nodes by storing edge labels as strings rather than
//! individual characters.
//!
//! Patricia tries offer excellent performance for:
//! - String key storage with common prefixes
//! - Longest prefix matching
//! - Memory efficiency through path compression
//! - Fast insertion, deletion, and lookup operations

use crate::error::Result;
use crate::fsa::traits::{
    FiniteStateAutomaton, PrefixIterable, StateInspectable, StatisticsProvider, Trie, TrieBuilder,
    TrieStats,
};
use crate::{FastVec, StateId};
use std::collections::HashMap;

/// Node in a Patricia trie
#[derive(Debug, Clone)]
struct PatriciaNode {
    /// The edge label leading to this node (compressed path)
    edge_label: FastVec<u8>,
    /// Children mapped by first byte of their edge label
    children: HashMap<u8, usize>,
    /// Whether this node represents a complete key
    is_final: bool,
    /// The complete key stored at this node (for final nodes)
    key: Option<FastVec<u8>>,
}

impl PatriciaNode {
    /// Create a new Patricia node
    fn new(edge_label: FastVec<u8>, is_final: bool) -> Self {
        Self {
            edge_label,
            children: HashMap::new(),
            is_final,
            key: None,
        }
    }

    /// Create a new root node
    fn new_root() -> Self {
        Self {
            edge_label: FastVec::new(),
            children: HashMap::new(),
            is_final: false,
            key: None,
        }
    }

    /// Set the complete key for this node
    fn set_key(&mut self, key: FastVec<u8>) -> Result<()> {
        self.key = Some(key);
        Ok(())
    }

    /// Get the edge label as a slice
    fn edge_label_slice(&self) -> &[u8] {
        self.edge_label.as_slice()
    }

    /// Check if this node has any children
    #[allow(dead_code)]
    fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get child node index by the first byte of the remaining key
    fn get_child(&self, first_byte: u8) -> Option<usize> {
        self.children.get(&first_byte).copied()
    }

    /// Add a child node
    fn add_child(&mut self, first_byte: u8, child_idx: usize) {
        self.children.insert(first_byte, child_idx);
    }

    /// Remove a child node
    #[allow(dead_code)]
    fn remove_child(&mut self, first_byte: u8) -> Option<usize> {
        self.children.remove(&first_byte)
    }
}

/// Patricia Trie implementation
///
/// A Patricia trie is a compressed trie where nodes with a single child are merged
/// with their parents. Edge labels store the entire compressed path, making the
/// trie more space-efficient for sparse key sets.
///
/// # Structure
/// - Nodes store compressed edge labels
/// - Children are indexed by the first byte of their edge label
/// - The tree maintains lexicographic order
/// - Path compression eliminates redundant nodes
///
/// # Examples
///
/// ```rust
/// use infini_zip::fsa::{PatriciaTrie, Trie};
///
/// let mut trie = PatriciaTrie::new();
/// trie.insert(b"hello").unwrap();
/// trie.insert(b"help").unwrap();
/// trie.insert(b"world").unwrap();
///
/// assert!(trie.contains(b"hello"));
/// assert!(trie.contains(b"help"));
/// assert!(!trie.contains(b"he"));
/// ```
#[derive(Debug, Clone)]
pub struct PatriciaTrie {
    /// Vector of nodes for cache-efficient storage
    nodes: Vec<PatriciaNode>,
    /// Index of the root node
    root: usize,
    /// Number of keys stored in the trie
    num_keys: usize,
}

impl PatriciaTrie {
    /// Create a new empty Patricia trie
    pub fn new() -> Self {
        let mut nodes = Vec::new();
        let root_node = PatriciaNode::new_root();
        nodes.push(root_node);

        Self {
            nodes,
            root: 0,
            num_keys: 0,
        }
    }

    /// Add a new node and return its index
    fn add_node(&mut self, node: PatriciaNode) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    /// Find the longest common prefix between two byte slices
    fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
        let mut i = 0;
        while i < a.len() && i < b.len() && a[i] == b[i] {
            i += 1;
        }
        i
    }

    /// Insert a key into the trie
    fn insert_recursive(&mut self, node_idx: usize, key: &[u8], full_key: &[u8]) -> Result<()> {
        let node_edge_label = self.nodes[node_idx].edge_label.clone();
        let edge_slice = node_edge_label.as_slice();

        // Find longest common prefix between key and edge label
        let common_len = Self::longest_common_prefix(key, edge_slice);

        if common_len == edge_slice.len() {
            // The edge label is a prefix of the key
            if common_len == key.len() {
                // Exact match - mark as final
                if !self.nodes[node_idx].is_final {
                    self.nodes[node_idx].is_final = true;
                    let mut key_vec = FastVec::new();
                    for &byte in full_key {
                        key_vec.push(byte)?;
                    }
                    self.nodes[node_idx].set_key(key_vec)?;
                    self.num_keys += 1;
                }
                return Ok(());
            } else {
                // Continue with remaining key
                let remaining_key = &key[common_len..];
                let first_byte = remaining_key[0];

                if let Some(child_idx) = self.nodes[node_idx].get_child(first_byte) {
                    // Child exists, recurse
                    self.insert_recursive(child_idx, remaining_key, full_key)?;
                } else {
                    // Create new child
                    let mut edge_label = FastVec::new();
                    for &byte in remaining_key {
                        edge_label.push(byte)?;
                    }

                    let mut new_node = PatriciaNode::new(edge_label, true);
                    let mut key_vec = FastVec::new();
                    for &byte in full_key {
                        key_vec.push(byte)?;
                    }
                    new_node.set_key(key_vec)?;

                    let new_idx = self.add_node(new_node);
                    self.nodes[node_idx].add_child(first_byte, new_idx);
                    self.num_keys += 1;
                }
                return Ok(());
            }
        }

        // Need to split the current node
        let old_edge_slice = edge_slice.to_vec();
        let old_children = self.nodes[node_idx].children.clone();
        let old_is_final = self.nodes[node_idx].is_final;
        let old_key = self.nodes[node_idx].key.clone();

        // Update current node to represent common prefix
        let mut common_prefix = FastVec::new();
        for &byte in &edge_slice[..common_len] {
            common_prefix.push(byte)?;
        }
        self.nodes[node_idx].edge_label = common_prefix;
        self.nodes[node_idx].children.clear();
        self.nodes[node_idx].is_final = false;
        self.nodes[node_idx].key = None;

        // Create node for the original edge's suffix
        if common_len < old_edge_slice.len() {
            let mut old_suffix = FastVec::new();
            for &byte in &old_edge_slice[common_len..] {
                old_suffix.push(byte)?;
            }

            let mut old_node = PatriciaNode::new(old_suffix, old_is_final);
            old_node.children = old_children;
            old_node.key = old_key;

            let old_idx = self.add_node(old_node);
            let old_first_byte = old_edge_slice[common_len];
            self.nodes[node_idx].add_child(old_first_byte, old_idx);
        }

        // Handle the new key
        if common_len == key.len() {
            // New key matches common prefix exactly
            self.nodes[node_idx].is_final = true;
            let mut key_vec = FastVec::new();
            for &byte in full_key {
                key_vec.push(byte)?;
            }
            self.nodes[node_idx].set_key(key_vec)?;
            self.num_keys += 1;
        } else {
            // Create node for new key's suffix
            let remaining_key = &key[common_len..];
            let mut new_suffix = FastVec::new();
            for &byte in remaining_key {
                new_suffix.push(byte)?;
            }

            let mut new_node = PatriciaNode::new(new_suffix, true);
            let mut key_vec = FastVec::new();
            for &byte in full_key {
                key_vec.push(byte)?;
            }
            new_node.set_key(key_vec)?;

            let new_idx = self.add_node(new_node);
            let new_first_byte = remaining_key[0];
            self.nodes[node_idx].add_child(new_first_byte, new_idx);
            self.num_keys += 1;
        }

        Ok(())
    }

    /// Find a key in the trie and return its node index
    fn find_node(&self, key: &[u8]) -> Option<usize> {
        let mut current_idx = self.root;
        let mut remaining_key = key;

        loop {
            let node = &self.nodes[current_idx];
            let edge_slice = node.edge_label_slice();

            // Check if remaining key starts with this node's edge label
            if remaining_key.len() < edge_slice.len() {
                return None; // Key is shorter than edge label
            }

            if !remaining_key.starts_with(edge_slice) {
                return None; // Key doesn't match edge label
            }

            remaining_key = &remaining_key[edge_slice.len()..];

            if remaining_key.is_empty() {
                // We've consumed the entire key
                return if node.is_final {
                    Some(current_idx)
                } else {
                    None
                };
            }

            // Continue to child
            let first_byte = remaining_key[0];
            if let Some(child_idx) = node.get_child(first_byte) {
                current_idx = child_idx;
            } else {
                return None; // No matching child
            }
        }
    }

    /// Collect all keys with a given prefix
    fn collect_keys_with_prefix(
        &self,
        node_idx: usize,
        current_path: &[u8],
        prefix: &[u8],
        results: &mut Vec<Vec<u8>>,
    ) {
        let node = &self.nodes[node_idx];
        let edge_slice = node.edge_label_slice();

        // Build full path to this node
        let mut full_path = current_path.to_vec();
        full_path.extend_from_slice(edge_slice);

        // Check if this node represents a key with the desired prefix
        if node.is_final && full_path.starts_with(prefix) {
            results.push(full_path.clone());
        }

        // Recurse to children
        for &child_idx in node.children.values() {
            self.collect_keys_with_prefix(child_idx, &full_path, prefix, results);
        }
    }

    /// Calculate the maximum depth of the trie
    #[allow(dead_code)]
    fn calculate_max_depth(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];

        if node.children.is_empty() {
            return 1;
        }

        let mut max_child_depth = 0;
        for &child_idx in node.children.values() {
            max_child_depth = max_child_depth.max(self.calculate_max_depth(child_idx));
        }

        1 + max_child_depth
    }

    /// Get statistics about nodes and transitions
    fn collect_stats(
        &self,
        node_idx: usize,
        depth: usize,
        stats: &mut (usize, usize, usize, usize),
    ) {
        let node = &self.nodes[node_idx];

        // Count this node
        stats.0 += 1;

        // Count transitions
        stats.1 += node.children.len();

        // Update max depth
        stats.2 = stats.2.max(depth);

        // Count depth for average calculation
        if node.is_final {
            stats.3 += depth;
        }

        // Recurse to children
        for &child_idx in node.children.values() {
            self.collect_stats(child_idx, depth + 1, stats);
        }
    }
}

impl Default for PatriciaTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl FiniteStateAutomaton for PatriciaTrie {
    fn root(&self) -> StateId {
        self.root as StateId
    }

    fn is_final(&self, state: StateId) -> bool {
        if let Some(node) = self.nodes.get(state as usize) {
            node.is_final
        } else {
            false
        }
    }

    fn transition(&self, state: StateId, symbol: u8) -> Option<StateId> {
        if let Some(node) = self.nodes.get(state as usize) {
            node.get_child(symbol).map(|idx| idx as StateId)
        } else {
            None
        }
    }

    fn transitions(&self, state: StateId) -> Box<dyn Iterator<Item = (u8, StateId)> + '_> {
        if let Some(node) = self.nodes.get(state as usize) {
            let iter = node
                .children
                .iter()
                .map(|(&byte, &idx)| (byte, idx as StateId));
            Box::new(iter.collect::<Vec<_>>().into_iter())
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl Trie for PatriciaTrie {
    fn insert(&mut self, key: &[u8]) -> Result<StateId> {
        self.insert_recursive(self.root, key, key)?;
        Ok(self.root as StateId)
    }

    fn lookup(&self, key: &[u8]) -> Option<StateId> {
        self.find_node(key).map(|idx| idx as StateId)
    }

    fn len(&self) -> usize {
        self.num_keys
    }
}

impl StateInspectable for PatriciaTrie {
    fn out_degree(&self, state: StateId) -> usize {
        if let Some(node) = self.nodes.get(state as usize) {
            node.children.len()
        } else {
            0
        }
    }

    fn out_symbols(&self, state: StateId) -> Vec<u8> {
        if let Some(node) = self.nodes.get(state as usize) {
            node.children.keys().copied().collect()
        } else {
            Vec::new()
        }
    }
}

impl StatisticsProvider for PatriciaTrie {
    fn stats(&self) -> TrieStats {
        let node_memory = self.nodes.len() * std::mem::size_of::<PatriciaNode>();
        let edge_memory: usize = self.nodes.iter().map(|node| node.edge_label.len()).sum();
        let children_memory: usize = self
            .nodes
            .iter()
            .map(|node| node.children.len() * std::mem::size_of::<(u8, usize)>())
            .sum();

        let memory_usage = node_memory + edge_memory + children_memory;

        let mut collect_stats = (0, 0, 0, 0); // (nodes, transitions, max_depth, total_depth)
        self.collect_stats(self.root, 0, &mut collect_stats);

        let avg_depth = if self.num_keys > 0 {
            collect_stats.3 as f64 / self.num_keys as f64
        } else {
            0.0
        };

        let mut stats = TrieStats {
            num_states: self.nodes.len(),
            num_keys: self.num_keys,
            num_transitions: collect_stats.1,
            max_depth: collect_stats.2,
            avg_depth,
            memory_usage,
            bits_per_key: 0.0,
        };

        stats.calculate_bits_per_key();
        stats
    }
}

/// Builder for constructing Patricia tries from sorted key sequences
pub struct PatriciaTrieBuilder;

impl TrieBuilder<PatriciaTrie> for PatriciaTrieBuilder {
    fn build_from_sorted<I>(keys: I) -> Result<PatriciaTrie>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut trie = PatriciaTrie::new();

        for key in keys {
            trie.insert(&key)?;
        }

        Ok(trie)
    }
}

/// Iterator for prefix enumeration in Patricia tries
pub struct PatriciaTriePrefixIterator {
    results: Vec<Vec<u8>>,
    index: usize,
}

impl Iterator for PatriciaTriePrefixIterator {
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

impl PrefixIterable for PatriciaTrie {
    fn iter_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Vec<u8>> + '_> {
        let mut results = Vec::new();
        self.collect_keys_with_prefix(self.root, &[], prefix, &mut results);
        results.sort(); // Maintain lexicographic order

        Box::new(PatriciaTriePrefixIterator { results, index: 0 })
    }
}

// Implement builder as associated function
impl PatriciaTrie {
    /// Build a Patricia trie from a sorted iterator of keys
    pub fn build_from_sorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        PatriciaTrieBuilder::build_from_sorted(keys)
    }

    /// Build a Patricia trie from an unsorted iterator of keys
    pub fn build_from_unsorted<I>(keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        PatriciaTrieBuilder::build_from_unsorted(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsa::traits::{FiniteStateAutomaton, PrefixIterable, StateInspectable, Trie};

    #[test]
    fn test_patricia_trie_basic_operations() {
        let mut trie = PatriciaTrie::new();

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
    fn test_patricia_trie_path_compression() {
        let mut trie = PatriciaTrie::new();

        // Insert keys that should result in path compression
        trie.insert(b"hello").unwrap();
        trie.insert(b"help").unwrap();

        assert_eq!(trie.len(), 2);
        assert!(trie.contains(b"hello"));
        assert!(trie.contains(b"help"));
        assert!(!trie.contains(b"hel"));
        assert!(!trie.contains(b"he"));

        // The trie should have compressed the common prefix "hel"
        // We can test this indirectly through correct behavior

        // Add another key that shares less prefix
        trie.insert(b"world").unwrap();
        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b"world"));
    }

    #[test]
    fn test_patricia_trie_prefix_splitting() {
        let mut trie = PatriciaTrie::new();

        // Insert a longer key first
        trie.insert(b"testing").unwrap();

        // Then insert a prefix of that key
        trie.insert(b"test").unwrap();

        assert_eq!(trie.len(), 2);
        assert!(trie.contains(b"test"));
        assert!(trie.contains(b"testing"));
        assert!(!trie.contains(b"te"));

        // Insert another key that forces splitting
        trie.insert(b"tea").unwrap();

        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b"tea"));
        assert!(trie.contains(b"test"));
        assert!(trie.contains(b"testing"));
    }

    #[test]
    fn test_patricia_trie_prefix_iteration() {
        let mut trie = PatriciaTrie::new();
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

        // Test non-existent prefix
        let dog_results: Vec<Vec<u8>> = trie.iter_prefix(b"dog").collect();
        assert!(dog_results.is_empty());
    }

    #[test]
    fn test_patricia_trie_duplicate_keys() {
        let mut trie = PatriciaTrie::new();

        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1);

        // Insert the same key again
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 1); // Should not increase

        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_patricia_trie_empty_key() {
        let mut trie = PatriciaTrie::new();

        // Insert empty key
        trie.insert(b"").unwrap();
        assert_eq!(trie.len(), 1);
        assert!(trie.contains(b""));

        // Insert another key
        trie.insert(b"hello").unwrap();
        assert_eq!(trie.len(), 2);
        assert!(trie.contains(b""));
        assert!(trie.contains(b"hello"));
    }

    #[test]
    fn test_patricia_trie_builder() {
        let keys = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
            b"care".to_vec(),
        ];

        let trie = PatriciaTrie::build_from_sorted(keys.clone()).unwrap();
        assert_eq!(trie.len(), 4);

        for key in &keys {
            assert!(trie.contains(key));
        }

        // Test with unsorted keys
        let mut unsorted_keys = keys.clone();
        unsorted_keys.reverse();

        let trie2 = PatriciaTrie::build_from_unsorted(unsorted_keys).unwrap();
        assert_eq!(trie2.len(), 4);

        for key in &keys {
            assert!(trie2.contains(key));
        }
    }

    #[test]
    fn test_patricia_trie_statistics() {
        let mut trie = PatriciaTrie::new();
        trie.insert(b"cat").unwrap();
        trie.insert(b"car").unwrap();
        trie.insert(b"card").unwrap();

        let stats = trie.stats();
        assert_eq!(stats.num_keys, 3);
        assert!(stats.memory_usage > 0);
        assert!(stats.max_depth > 0);
    }

    #[test]
    fn test_patricia_trie_large_keys() {
        let mut trie = PatriciaTrie::new();

        // Test with longer keys
        let long_key = b"this_is_a_very_long_key_for_testing_purposes";
        trie.insert(long_key).unwrap();

        assert!(trie.contains(long_key));
        assert_eq!(trie.len(), 1);

        // Test with a similar long key
        let similar_key = b"this_is_a_very_long_key_for_testing_patricia_tries";
        trie.insert(similar_key).unwrap();

        assert!(trie.contains(long_key));
        assert!(trie.contains(similar_key));
        assert_eq!(trie.len(), 2);
    }

    #[test]
    fn test_patricia_trie_transitions() {
        let mut trie = PatriciaTrie::new();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abd").unwrap();
        trie.insert(b"ac").unwrap();

        let root = trie.root();

        // Check transitions from root
        let transitions: Vec<(u8, StateId)> = trie.transitions(root).collect();
        assert!(!transitions.is_empty());

        // The exact structure depends on path compression,
        // but we should be able to navigate to children
        let symbols = trie.out_symbols(root);
        assert!(!symbols.is_empty());
    }

    #[test]
    fn test_patricia_trie_common_prefix_edge_cases() {
        let mut trie = PatriciaTrie::new();

        // Test various prefix relationships
        trie.insert(b"a").unwrap();
        trie.insert(b"ab").unwrap();
        trie.insert(b"abc").unwrap();
        trie.insert(b"abcd").unwrap();

        assert_eq!(trie.len(), 4);
        assert!(trie.contains(b"a"));
        assert!(trie.contains(b"ab"));
        assert!(trie.contains(b"abc"));
        assert!(trie.contains(b"abcd"));

        // Test prefix iteration
        let all_results: Vec<Vec<u8>> = trie.iter_prefix(b"").collect();
        assert_eq!(all_results.len(), 4);

        let ab_results: Vec<Vec<u8>> = trie.iter_prefix(b"ab").collect();
        assert_eq!(ab_results.len(), 3); // "ab", "abc", "abcd"
    }
}
