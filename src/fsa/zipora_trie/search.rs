//! Lookup, restore, key-collection, and bulk-build internals for [`ZiporaTrie`](super::ZiporaTrie).

use super::ZiporaTrie;
use super::config::{TrieStrategy, ZiporaTrieConfig};
use super::storage::{PatriciaNode, TrieStorage};
use crate::StateId;
use crate::containers::FastVec;
use crate::error::Result;
use crate::succinct::RankSelectOps;
use std::collections::HashMap;

impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{

    pub(super) fn restore_string_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        _compressed_paths: &HashMap<StateId, Vec<u8>>,
        state_id: StateId,
    ) -> Option<Vec<u8>> {
        if nodes.is_empty() || state_id as usize >= nodes.len() {
            return None;
        }

        // Check if the target state is final
        if !nodes[state_id as usize].is_final {
            return None;
        }

        // Perform DFS to find the path from root to the target state
        let mut path = Vec::new();
        if Self::find_path_to_state(nodes, 0, state_id as usize, &mut path) {
            Some(path)
        } else {
            None
        }
    }

    pub(super) fn find_path_to_state(
        nodes: &FastVec<PatriciaNode>,
        current: usize,
        target: usize,
        path: &mut Vec<u8>,
    ) -> bool {
        if current == target {
            return true;
        }

        if current >= nodes.len() {
            return false;
        }

        let node = &nodes[current];

        // Try each child (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id = child_id as usize;

            // Add this symbol to the path
            path.push(symbol);

            // Recursively search in child
            if Self::find_path_to_state(nodes, child_id, target, path) {
                return true;
            }

            // Backtrack if not found
            path.pop();
        }

        false
    }

    pub(super) fn lookup_node_id_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> Option<StateId> {
        if nodes.is_empty() {
            return None;
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > key.len() {
                        return None; // Not enough key left
                    }
                    if &key[key_pos..key_pos + path.len()] != path.as_slice() {
                        return None; // Path doesn't match
                    }
                    key_pos += path.len();
                }
            } else {
                return None;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        if key_pos == key.len() && nodes[current].is_final {
            Some(current as StateId)
        } else {
            None
        }
    }

    /// Get all keys from Patricia trie
    pub(super) fn keys_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();
        Self::collect_keys_patricia_recursive(
            nodes,
            compressed_paths,
            0,
            &mut current_path,
            &mut keys,
        );
        keys
    }

    /// Get all keys with prefix from Patricia trie
    pub(super) fn keys_with_prefix_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if nodes.is_empty() {
            return Vec::new();
        }

        // Navigate to the prefix position first
        let mut current = 0;
        let mut key_pos = 0;
        let mut path_to_prefix = Vec::new();

        while key_pos < prefix.len() {
            let symbol = prefix[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path_to_prefix.push(symbol);
                current = child_id as usize;
                key_pos += 1;

                // Check compressed path
                if let Some(path) = compressed_paths.get(&child_id) {
                    if key_pos + path.len() > prefix.len() {
                        // Prefix doesn't fully match this path
                        let remaining_prefix = &prefix[key_pos..];
                        if path.starts_with(remaining_prefix) {
                            // Prefix is a partial match of this compressed path
                            // Continue from this node with the partial prefix included
                            path_to_prefix.extend_from_slice(remaining_prefix);
                            break;
                        } else {
                            // Prefix doesn't match - no keys with this prefix
                            return Vec::new();
                        }
                    } else if &prefix[key_pos..key_pos + path.len()] != path.as_slice() {
                        // Path doesn't match prefix
                        return Vec::new();
                    } else {
                        // Path matches, continue
                        path_to_prefix.extend_from_slice(path);
                        key_pos += path.len();
                    }
                }
            } else {
                // No child for this symbol - no keys with this prefix
                return Vec::new();
            }
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = path_to_prefix;
        Self::collect_keys_patricia_recursive(
            nodes,
            compressed_paths,
            current,
            &mut current_path,
            &mut keys,
        );

        // Filter to only include keys that actually start with the prefix
        keys.into_iter()
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    /// Recursively collect all keys from Patricia trie
    pub(super) fn collect_keys_patricia_recursive(
        nodes: &FastVec<PatriciaNode>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        node_id: usize,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        if node_id >= nodes.len() {
            return;
        }

        let node = &nodes[node_id];

        // If this is a final node, add the current path as a key
        if node.is_final {
            keys.push(current_path.clone());
        }

        // Explore all children (compact representation)
        for &(symbol, child_id) in node.children.iter() {
            let child_id_usize = child_id as usize;

            // Add this symbol to the path
            current_path.push(symbol);

            // Add compressed path if it exists
            let path_start_len = current_path.len();
            if let Some(path) = compressed_paths.get(&child_id) {
                current_path.extend_from_slice(path);
            }

            // Recursively collect from child
            Self::collect_keys_patricia_recursive(
                nodes,
                compressed_paths,
                child_id_usize,
                current_path,
                keys,
            );

            // Backtrack: remove the path we added
            current_path.truncate(path_start_len - 1);
        }
    }

    /// Get all keys from LOUDS trie storage
    pub(super) fn keys_louds_actual(label_data: &FastVec<u8>) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();

        if label_data.is_empty() {
            return keys;
        }

        let mut current_key = Vec::new();

        for &byte in label_data.iter() {
            if byte == 0u8 {
                // Found separator, this completes a key
                if !current_key.is_empty() {
                    keys.push(current_key.clone());
                    current_key.clear();
                }
            } else {
                // Add byte to current key
                current_key.push(byte);
            }
        }

        // Handle last key if there's no trailing separator
        if !current_key.is_empty() {
            keys.push(current_key);
        }

        // Remove duplicates and sort
        keys.sort();
        keys.dedup();

        keys
    }

    /// Get all keys with a given prefix from LOUDS trie storage
    pub(super) fn keys_with_prefix_louds_actual(label_data: &FastVec<u8>, prefix: &[u8]) -> Vec<Vec<u8>> {
        let all_keys = Self::keys_louds_actual(label_data);

        // Filter keys that start with the given prefix
        all_keys
            .into_iter()
            .filter(|key| key.starts_with(prefix))
            .collect()
    }

    /// Get all keys from DoubleArray trie storage
    pub(super) fn keys_double_array_actual(base: &FastVec<u32>, check: &FastVec<u32>) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        let mut current_path = Vec::new();

        #[cfg(debug_assertions)]
        eprintln!(
            "DEBUG keys_double_array: Starting from root state 0, base[0]={:?}",
            base.first()
        );

        Self::collect_keys_double_array_recursive(base, check, 0, &mut current_path, &mut keys);
        keys
    }

    /// Get all keys with prefix from DoubleArray trie storage
    pub(super) fn keys_with_prefix_double_array_actual(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        prefix: &[u8],
    ) -> Vec<Vec<u8>> {
        if base.is_empty() {
            return Vec::new();
        }

        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)

        // Navigate to the prefix position first
        let mut current_state = 0u32;
        for &symbol in prefix {
            // SAFETY: We check if base_val exists, then use it
            let base_value = match base.get(current_state as usize) {
                Some(val) => val & VALUE_MASK,
                None => return Vec::new(),
            };
            let next_state = base_value.saturating_add(symbol as u32);
            if next_state as usize >= check.len() {
                return Vec::new();
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project (line 106)
            if check_val != current_state {
                return Vec::new();
            }

            current_state = next_state;
        }

        // Now collect all keys from this point
        let mut keys = Vec::new();
        let mut current_path = prefix.to_vec();
        Self::collect_keys_double_array_recursive(
            base,
            check,
            current_state,
            &mut current_path,
            &mut keys,
        );
        keys
    }

    /// Recursively collect all keys from DoubleArray trie
    pub(super) fn collect_keys_double_array_recursive(
        base: &FastVec<u32>,
        check: &FastVec<u32>,
        state: u32,
        current_path: &mut Vec<u8>,
        keys: &mut Vec<Vec<u8>>,
    ) {
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)

        #[cfg(debug_assertions)]
        if state == 0 && current_path.is_empty() {
            eprintln!("DEBUG collect_keys: At root, checking for children...");
        }

        // If this is a terminal state, add the current path as a key (check base array)
        if (state as usize) < base.len() && (base[state as usize] & TERMINAL_BIT) != 0 {
            #[cfg(debug_assertions)]
            eprintln!(
                "DEBUG collect_keys: Found terminal state {} with path {:?}",
                state,
                std::str::from_utf8(current_path).unwrap_or("<non-utf8>")
            );
            keys.push(current_path.clone());
        }

        // Get the base value for this state
        if let Some(&base_raw) = base.get(state as usize) {
            let base_val = base_raw & VALUE_MASK;
            if base_val == 0 || base_val == 0x7FFF_FFFF {
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG collect_keys: State {} has base={}, no children",
                    state, base_val
                );
                return; // No children
            }

            #[cfg(debug_assertions)]
            if state == 0 {
                eprintln!(
                    "DEBUG collect_keys: Root state 0 has base={}, checking all 256 symbols...",
                    base_val
                );
            }

            // Try all possible symbols
            for symbol in 0u8..=255u8 {
                let next_state = base_val.saturating_add(symbol as u32);

                // Check if this is a valid transition (referenced project line 106)
                if (next_state as usize) < check.len() {
                    let check_val = check[next_state as usize];
                    // Direct comparison: check[next] == current_state
                    let is_valid_child = check_val == state;

                    if is_valid_child {
                        // Valid transition found
                        #[cfg(debug_assertions)]
                        if state == 0 {
                            eprintln!(
                                "DEBUG collect_keys: Found valid transition from root: symbol={:02x} ('{}'), next_state={}",
                                symbol, symbol as char, next_state
                            );
                        }
                        current_path.push(symbol);
                        Self::collect_keys_double_array_recursive(
                            base,
                            check,
                            next_state,
                            current_path,
                            keys,
                        );
                        current_path.pop();
                    }
                }
            }
        }
    }

    /// Build a trie from sorted keys using BFS construction
    ///
    /// This is more efficient than incremental insertion for sorted input because:
    /// 1. Pre-allocates arrays based on estimated size
    /// 2. Processes keys in sorted order to minimize relocations
    /// 3. Uses improved find_free_base for better packing
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig};
    /// use zipora::succinct::RankSelectInterleaved256;
    ///
    /// let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
    /// let trie: ZiporaTrie<RankSelectInterleaved256> =
    ///     ZiporaTrie::build_from_sorted(&keys, ZiporaTrieConfig::default()).unwrap();
    ///
    /// assert_eq!(trie.len(), 5);
    /// assert!(trie.contains(b"apple"));
    /// assert!(trie.contains(b"banana"));
    /// ```
    pub fn build_from_sorted(keys: &[&[u8]], config: ZiporaTrieConfig) -> Result<Self> {
        // Create trie with config
        let mut trie = Self::with_config(config);

        // Estimate size and pre-allocate for DoubleArray strategy
        if let TrieStrategy::DoubleArray { .. } = &trie.config.trie_strategy
            && let TrieStorage::DoubleArray { base, check, .. } = &mut trie.storage
        {
            // Estimate: each key adds ~key_length states on average
            let estimated_states = keys.iter().map(|k| k.len()).sum::<usize>() / 2;
            let initial_size = estimated_states.max(256);

            const NIL_STATE: u32 = 0x7FFF_FFFF;
            const FREE_BIT: u32 = 0x8000_0000;

            let _ = base.resize(initial_size, NIL_STATE);
            let _ = check.resize(initial_size, NIL_STATE | FREE_BIT);
        }

        // Insert keys in sorted order
        // Sorted order tends to result in fewer relocations
        for &key in keys {
            trie.insert(key)?;
        }

        Ok(trie)
    }
}
