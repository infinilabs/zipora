//! Insertion, construction, and double-array mutation internals for [`ZiporaTrie`](super::ZiporaTrie).

use super::ZiporaTrie;
use super::storage::{
    CritBitNode, DA_FREE_BIT, DA_MAX_STATE, DA_NIL_STATE, DA_TERMINAL_BIT, DA_VALUE_MASK,
    DaFreeList, PatriciaNode,
};
use crate::StateId;
use crate::containers::FastVec;
use crate::containers::specialized::UintVector;
use crate::error::{Result, ZiporaError};
use crate::succinct::RankSelectOps;
use std::collections::HashMap;

impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    // Patricia trie implementation methods
    pub(super) fn insert_patricia(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        free_list: &mut Vec<StateId>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        Self::insert_patricia_actual(nodes, edge_data, compressed_paths, free_list, key, num_keys)
    }

    pub(super) fn contains_patricia(
        &self,
        nodes: &FastVec<PatriciaNode>,
        edge_data: &FastVec<u8>,
        compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        Self::contains_patricia_actual(nodes, edge_data, compressed_paths, key)
    }

    // Critical-bit trie strategy is not implemented; insert returns NotSupported.
    pub(super) fn insert_critical_bit(
        _nodes: &mut FastVec<CritBitNode>,
        _keys: &mut FastVec<Vec<u8>>,
        _critical_cache: &mut HashMap<usize, u8>,
        _key: &[u8],
    ) -> Result<StateId> {
        Err(ZiporaError::not_supported(
            "CriticalBit trie strategy is not yet implemented",
        ))
    }

    pub(super) fn contains_critical_bit(
        &self,
        _nodes: &FastVec<CritBitNode>,
        _keys: &FastVec<Vec<u8>>,
        _critical_cache: &HashMap<usize, u8>,
        _key: &[u8],
    ) -> bool {
        false
    }

    // Double array trie implementation methods
    //
    // The arguments are the destructured fields of `TrieStorage::DoubleArray`
    // plus the two counters the caller owns; grouping them behind a struct
    // would only move the same borrow split somewhere else.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn insert_double_array(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        free_list: &mut DaFreeList,
        state_count: &mut usize,
        key: &[u8],
        num_keys: &mut usize,
        relocations: &mut Vec<(u32, u32)>,
    ) -> Result<StateId> {
        // Storage construction seeds the root; keep the arrays usable if it did not.
        if base.is_empty() {
            base.push(DA_NIL_STATE)?;
            check.push(0)?;
            *state_count = 1;
        }

        // Empty key: the root itself is the terminal state.
        if key.is_empty() {
            let was_new = (base[0] & DA_TERMINAL_BIT) == 0;
            base[0] |= DA_TERMINAL_BIT;
            if was_new {
                *num_keys += 1;
            }
            return Ok(0);
        }

        let mut current_state = 0u32;
        for &symbol in key {
            let mut base_value = base[current_state as usize] & DA_VALUE_MASK;
            if base_value == DA_NIL_STATE {
                // First child of this state: pick a base whose slot for `symbol` is free.
                base_value = Self::find_free_base(check, free_list, &[symbol]);
                base[current_state as usize] =
                    base_value | (base[current_state as usize] & DA_TERMINAL_BIT);
            }

            let next_state = base_value + symbol as u32;
            Self::ensure_da_capacity(base, check, free_list, next_state as usize + 1)?;

            // A transition exists iff check[next] == parent; free cells carry
            // DA_FREE_BIT and can never compare equal to a state id.
            let check_val = check[next_state as usize];
            if check_val == current_state {
                current_state = next_state;
            } else if (check_val & DA_FREE_BIT) != 0 {
                free_list.claim(base, check, next_state, current_state);
                *state_count += 1;
                current_state = next_state;
            } else {
                // Slot belongs to another state's child: move all of
                // current_state's children to a base with room for `symbol`.
                let new_base = Self::relocate_state(
                    base,
                    check,
                    free_list,
                    current_state,
                    symbol,
                    relocations,
                )?;
                let new_next = new_base + symbol as u32;
                free_list.claim(base, check, new_next, current_state);
                *state_count += 1;
                current_state = new_next;
            }
        }

        // Mark the final state as terminal (referenced project: set_term_bit on base).
        let was_new = (base[current_state as usize] & DA_TERMINAL_BIT) == 0;
        base[current_state as usize] |= DA_TERMINAL_BIT;
        if was_new {
            *num_keys += 1;
        }

        Ok(current_state)
    }

    /// Find a base such that the slot `base + s` is free for every `s` in
    /// `symbols`.
    ///
    /// Walks the free-cell list only — never the occupied prefix — trying each
    /// free cell as the home of the lowest symbol (`base = cell - min`). Bases
    /// are `>= 1`, so no child can land on the root slot. For a single symbol
    /// the first eligible cell always fits. After `MAX_TRIALS` failed
    /// multi-symbol candidates the block is placed at the high-water mark,
    /// above which every slot is free by construction (growing the array if
    /// it does not reach that far). Without that cap a wide node in a
    /// fragmented array would walk every hole, which is O(n) again.
    pub(super) fn find_free_base(check: &FastVec<u32>, free_list: &DaFreeList, symbols: &[u8]) -> u32 {
        const MAX_TRIALS: u32 = 256;

        let min_symbol = symbols.iter().copied().min().map_or(0, u32::from);
        let len = check.len() as u32;
        let mut trials = 0;
        let mut cell = free_list.head;
        while cell != DA_NIL_STATE {
            if cell > min_symbol {
                let candidate = cell - min_symbol;
                let fits = symbols.iter().all(|&s| {
                    let pos = candidate + s as u32;
                    pos >= len || (check[pos as usize] & DA_FREE_BIT) != 0
                });
                if fits {
                    return candidate;
                }
                trials += 1;
                if trials == MAX_TRIALS {
                    break;
                }
            }
            cell = check[cell as usize] & DA_VALUE_MASK;
        }
        free_list.high_water.max(min_symbol + 1) - min_symbol
    }

    /// Grow `base`/`check` so that index `required - 1` exists, threading every
    /// new cell onto the free list in ascending order. All growth must go
    /// through here: a free cell that is not linked is invisible to
    /// `find_free_base`.
    pub(super) fn ensure_da_capacity(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        free_list: &mut DaFreeList,
        required: usize,
    ) -> Result<()> {
        let old_len = base.len();
        if required <= old_len {
            return Ok(());
        }
        let new_len = required.max(old_len * 3 / 2).max(256);
        if new_len > DA_MAX_STATE as usize + 1 {
            return Err(ZiporaError::invalid_data(
                "double array exceeds the maximum state count",
            ));
        }
        base.resize(new_len, DA_NIL_STATE)?;
        check.resize(new_len, DA_NIL_STATE | DA_FREE_BIT)?;
        for cell in old_len..new_len {
            free_list.push_back(base, check, cell as u32);
        }
        Ok(())
    }

    // Helper: Relocate a state and all its children to a new base that also
    // has room for `new_symbol`. Every moved child is recorded in
    // `relocations` as (old_state, new_state) so side tables indexed by state
    // ID (e.g. ZiporaTrieMap values) can follow.
    pub(super) fn relocate_state(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        free_list: &mut DaFreeList,
        state: u32,
        new_symbol: u8,
        relocations: &mut Vec<(u32, u32)>,
    ) -> Result<u32> {
        let old_base = base[state as usize] & DA_VALUE_MASK;

        // (symbol, old position, raw base including the terminal bit)
        let mut children: Vec<(u8, u32, u32)> = Vec::new();
        for symbol in 0u8..=255u8 {
            let pos = old_base + symbol as u32;
            if (pos as usize) < check.len() && check[pos as usize] == state {
                children.push((symbol, pos, base[pos as usize]));
            }
        }

        let mut symbols: Vec<u8> = children.iter().map(|c| c.0).collect();
        symbols.push(new_symbol);
        let new_base = Self::find_free_base(check, free_list, &symbols);
        let max_symbol = symbols.iter().copied().max().unwrap_or(new_symbol);
        Self::ensure_da_capacity(base, check, free_list, (new_base + max_symbol as u32) as usize + 1)?;

        // The new slots were all free when chosen and the old slots are still
        // occupied, so the two sets are disjoint: claim first, release after.
        for &(symbol, old_pos, raw_base) in &children {
            let new_pos = new_base + symbol as u32;
            free_list.claim(base, check, new_pos, state);
            base[new_pos as usize] = raw_base;
            relocations.push((old_pos, new_pos));

            // Grandchildren still name the old position as their parent.
            let child_base = raw_base & DA_VALUE_MASK;
            if child_base != 0 && child_base != DA_NIL_STATE {
                Self::update_grandchildren_check_values(base, check, old_pos, new_pos);
            }
        }
        for &(_, old_pos, _) in &children {
            free_list.push_front(base, check, old_pos);
        }

        base[state as usize] = new_base | (base[state as usize] & DA_TERMINAL_BIT);
        Ok(new_base)
    }

    // Helper function to update grandchildren when a child state is relocated
    pub(super) fn update_grandchildren_check_values(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        old_parent_pos: u32,
        new_parent_pos: u32,
    ) {
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free (referenced project)

        // Get the base value of the relocated child to find its children
        if let Some(&child_base_raw) = base.get(new_parent_pos as usize) {
            let child_base = child_base_raw & VALUE_MASK;
            if child_base != 0 && child_base != 0x7FFF_FFFF {
                // Find all grandchildren that were pointing to the old parent position
                for symbol in 0u8..=255u8 {
                    let grandchild_pos = child_base.saturating_add(symbol as u32);
                    if (grandchild_pos as usize) < check.len() {
                        let check_val = check[grandchild_pos as usize];
                        // Check if it's allocated (not free) and points to old parent
                        if (check_val & FREE_BIT) == 0 && check_val == old_parent_pos {
                            // This grandchild was pointing to the old parent position
                            // Update it to point to the new parent position
                            check[grandchild_pos as usize] = new_parent_pos;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn contains_double_array(&self, base: &FastVec<u32>, check: &FastVec<u32>, key: &[u8]) -> bool {
        // Following referenced project's double array trie lookup (line 100-110)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for actual values

        if base.is_empty() {
            return false;
        }

        // Special case for empty key - check if root is terminal (check terminal bit in base)
        if key.is_empty() {
            return base
                .first()
                .map(|b| (b & TERMINAL_BIT) != 0)
                .unwrap_or(false);
        }

        let mut current_state = 0u32;

        // Traverse the trie for each symbol (referenced project line 100-110: state_move)
        for &symbol in key.iter() {
            // SAFETY: We check if base_val exists, then use it
            let base_val = match base.get(current_state as usize) {
                Some(val) => val,
                None => {
                    return false;
                }
            };

            // Calculate next state using base value (bits 0-30)
            let next_state = (base_val & VALUE_MASK).saturating_add(symbol as u32);

            // Check if the transition is valid (referenced project line 106: states[next].parent() == curr)
            if next_state as usize >= check.len() {
                return false;
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project: check[next] == current_state
            // Free states have FREE_BIT set, so won't match
            if check_val != current_state {
                // Invalid transition
                return false;
            }

            current_state = next_state;

        }

        // Check if the final state is marked as terminal (check terminal bit in base)
        base.get(current_state as usize)
            .map(|b| (b & TERMINAL_BIT) != 0)
            .unwrap_or(false)
    }

    // LOUDS trie strategy is not implemented; insert returns NotSupported.
    pub(super) fn insert_louds(
        _louds: &mut R,
        _is_link: &mut R,
        _next_link: &mut UintVector,
        _label_data: &mut FastVec<u8>,
        _core_data: &mut FastVec<u8>,
        _next_trie: &mut Option<Box<ZiporaTrie<R>>>,
        _key: &[u8],
    ) -> Result<StateId> {
        Err(ZiporaError::not_supported(
            "LOUDS trie strategy is not yet implemented",
        ))
    }

    #[allow(clippy::too_many_arguments)] // internal helper; arg bundle would add indirection
    pub(super) fn contains_louds(
        &self,
        _louds: &R,
        _is_link: &R,
        _next_link: &UintVector,
        _label_data: &FastVec<u8>,
        _core_data: &FastVec<u8>,
        _next_trie: &Option<Box<ZiporaTrie<R>>>,
        _key: &[u8],
    ) -> bool {
        false
    }

    // Actual implementation methods for Patricia trie
    pub(super) fn insert_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        _edge_data: &mut FastVec<u8>,
        _compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        free_list: &mut Vec<StateId>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        if nodes.is_empty() {
            // Initialize with root node
            let _ = nodes.push(PatriciaNode::default());
        }

        let mut current = 0;
        let mut key_pos = 0;

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                // Follow existing path
                let child_id = node.children[idx].1;
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Create new child node, reusing a recycled id when available.
                // Recycled nodes were reset on removal, so no stale state leaks.
                let new_node_id = match free_list.pop() {
                    Some(id) => id as usize,
                    None => {
                        let _ = nodes.push(PatriciaNode::default());
                        nodes.len() - 1
                    }
                };

                // Insert into sorted children Vec
                let insert_pos = nodes[current]
                    .children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                    .unwrap_err();
                nodes[current]
                    .children
                    .insert(insert_pos, (symbol, new_node_id as StateId));

                current = new_node_id;
                key_pos += 1;
            }
        }

        // Mark current node as final (check if was_new)
        let was_new = !nodes[current].is_final;
        nodes[current].is_final = true;
        if was_new {
            *num_keys += 1;
        }
        Ok(current as StateId)
    }

    pub(super) fn contains_patricia_actual(
        nodes: &FastVec<PatriciaNode>,
        _edge_data: &FastVec<u8>,
        _compressed_paths: &HashMap<StateId, Vec<u8>>,
        key: &[u8],
    ) -> bool {
        if nodes.is_empty() {
            return false;
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
            } else {
                return false;
            }
        }

        // Check if we've consumed the entire key and reached a final state
        key_pos == key.len() && nodes[current].is_final
    }

    pub(super) fn remove_patricia_actual(
        nodes: &mut FastVec<PatriciaNode>,
        _edge_data: &mut FastVec<u8>,
        _compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        free_list: &mut Vec<StateId>,
        key: &[u8],
    ) -> Result<bool> {
        if nodes.is_empty() {
            return Ok(false);
        }

        // First, check if the key exists and find the path to it
        let mut current = 0;
        let mut key_pos = 0;
        let mut path = Vec::new(); // Track the path for potential cleanup

        while key_pos < key.len() {
            let symbol = key[key_pos];
            let node = &nodes[current];

            // Binary search in compact children
            if let Ok(idx) = node.children.binary_search_by_key(&symbol, |(s, _)| *s) {
                let child_id = node.children[idx].1;
                path.push((current, symbol)); // Store parent and symbol for path
                current = child_id as usize;
                key_pos += 1;
            } else {
                // Key doesn't exist
                return Ok(false);
            }
        }

        // Check if we found a complete key at a final state
        if key_pos != key.len() || !nodes[current].is_final {
            return Ok(false);
        }

        // Mark the node as non-final (remove the key)
        nodes[current].is_final = false;

        // Check if this node has any children
        let has_children = !nodes[current].children.is_empty();

        // If the node has no children and is not final, we can potentially clean it up
        if !has_children {
            // Walk back up the path and remove unnecessary nodes
            for &(parent_idx, symbol) in path.iter().rev() {
                // Remove the child pointer from parent and recycle the node.
                // The unlinked node is always a leaf here (childless and
                // non-final), so no subtree is orphaned.
                if let Ok(idx) = nodes[parent_idx]
                    .children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                {
                    let (_, child_id) = nodes[parent_idx].children.remove(idx);
                    nodes[child_id as usize] = PatriciaNode::default();
                    free_list.push(child_id);
                }

                // Check if parent node should also be cleaned up
                let parent_has_children = !nodes[parent_idx].children.is_empty();
                let parent_is_final = nodes[parent_idx].is_final;

                // If parent has other children or is final, stop cleanup
                if parent_has_children || parent_is_final {
                    break;
                }
            }
        }

        Ok(true)
    }
}
