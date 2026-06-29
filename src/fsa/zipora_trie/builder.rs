//! Insertion, construction, and double-array mutation internals for [`ZiporaTrie`](super::ZiporaTrie).

use super::ZiporaTrie;
use super::storage::{CritBitNode, PatriciaNode};
use crate::StateId;
use crate::containers::FastVec;
use crate::containers::specialized::UintVector;
use crate::error::{Result, ZiporaError};
use crate::succinct::RankSelectOps;
use std::collections::{HashMap, VecDeque};

impl<R> ZiporaTrie<R>
where
    R: RankSelectOps + Default,
{
    // Patricia trie implementation methods
    pub(super) fn insert_patricia(
        nodes: &mut FastVec<PatriciaNode>,
        edge_data: &mut FastVec<u8>,
        compressed_paths: &mut HashMap<StateId, Vec<u8>>,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        Self::insert_patricia_actual(nodes, edge_data, compressed_paths, key, num_keys)
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

    // Critical-bit trie implementation methods
    //  TODO: port from C++ reference `src/terark/fsa/crit_bit_trie.hpp`
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
    pub(super) fn insert_double_array(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        _free_list: &mut VecDeque<StateId>,
        state_count: &mut usize,
        key: &[u8],
        num_keys: &mut usize,
    ) -> Result<StateId> {
        // Following referenced project's double array trie implementation EXACTLY
        // Base array (m_child0): bits 0-30 = base value, bit 31 = terminal bit
        // Check array (m_parent): bits 0-30 = parent state, bit 31 = free bit

        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal states (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free states (referenced project)
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for actual values (referenced project)
        const MAX_STATE: u32 = 0x7FFF_FFFE; // Maximum valid state value (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF; // Nil state marker (referenced project)

        // Ensure we have at least the root state
        // Referenced project starts with 1 state (line 70: states.resize(1))
        // We initialize in storage creation, but check here for safety
        if base.is_empty() {
            let _ = base.resize(1, NIL_STATE); // Just root state
            let _ = check.resize(1, 0); // Root check is 0 (itself), no free bit
            // Use compact base allocation like referenced project
            base[0] = Self::find_free_base(base, check, 0)?;
            *state_count = 1;
        }

        // Special case for empty key - mark root as terminal
        if key.is_empty() {
            let was_new = (base[0] & TERMINAL_BIT) == 0;
            base[0] |= TERMINAL_BIT;
            if was_new {
                *num_keys += 1;
            }
            return Ok(0);
        }

        let mut current_state = 0u32;

        #[cfg(debug_assertions)]
        eprintln!(
            "DEBUG insert: Starting insertion of key: {:?}",
            std::str::from_utf8(key).unwrap_or("<non-utf8>")
        );

        // Traverse the trie for each symbol in the key
        for (_pos, &symbol) in key.iter().enumerate() {
            #[cfg(debug_assertions)]
            let pos = _pos;
            // Calculate next state position using base value (bits 0-30)
            let mut base_value = base[current_state as usize] & VALUE_MASK;

            // If base is NIL_STATE, we need to find a good base for this state's children
            // Referenced project does this during build (lines 309-327)
            if base_value == NIL_STATE {
                base_value = Self::find_free_base(base, check, current_state)?;
                // CRITICAL: Preserve terminal bit when setting new base
                let old_val = base[current_state as usize];
                base[current_state as usize] = base_value | (old_val & TERMINAL_BIT);
            }

            let next_state = base_value.saturating_add(symbol as u32);

            // Expand arrays if needed - use amortized growth
            let required = next_state as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                let _ = base.resize(new_size, NIL_STATE);
                let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // Check if this transition already exists (referenced project style at line 106)
            // A transition exists if check[next] == current_state (without free bit)
            // Free states have FREE_BIT set, so won't match
            let check_val = check[next_state as usize];
            let is_free = (check_val & FREE_BIT) != 0;
            let transition_exists = !is_free && check_val == current_state;

            if transition_exists {
                // Transition exists, follow it
                #[cfg(debug_assertions)]
                eprintln!(
                    "  [{}] '{:02x}' state {} -> {} (existing)",
                    pos, symbol, current_state, next_state
                );
                current_state = next_state;
            } else {
                // Need to create new transition
                // CRITICAL: Never allow transitions to state 0 (reserved for root)
                if next_state == 0 {
                    // State 0 is reserved, need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' conflict: next_state would be 0 (reserved for root)",
                        pos, symbol
                    );

                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base =
                        Self::relocate_state(base, check, current_state, symbol, state_count)?;

                    // Now the transition should be available at the new location
                    let new_next = new_base.saturating_add(symbol as u32);

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        let _ = base.resize(new_size, NIL_STATE);
                        let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                } else if is_free {
                    // Position is free and not state 0, use it directly
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' state {} -> {} (new, free)",
                        pos, symbol, current_state, next_state
                    );

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data("State value exceeds maximum"));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[next_state as usize] = current_state; // Clear free bit by assignment
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[next_state as usize] = NIL_STATE;
                    current_state = next_state;
                    *state_count += 1;
                } else {
                    // Position is occupied - need to relocate
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  [{}] '{:02x}' conflict at state {}, next_state {} already has check={:08x}",
                        pos, symbol, current_state, next_state, check[next_state as usize]
                    );
                    // We must relocate ALL children of current_state to maintain consistency
                    let new_base =
                        Self::relocate_state(base, check, current_state, symbol, state_count)?;

                    // Now the transition should be available
                    let new_next = new_base.saturating_add(symbol as u32);

                    #[cfg(debug_assertions)]
                    eprintln!(
                        "  Relocated state {} to new_base {}, new transition {} -> {}",
                        current_state, new_base, current_state, new_next
                    );

                    // Expand if needed - use amortized growth
                    let required = new_next as usize + 1;
                    if required > base.len() {
                        let new_size = required.max(base.len() * 3 / 2).max(256);
                        let _ = base.resize(new_size, NIL_STATE);
                        let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
                    }

                    // Ensure the parent state fits within VALUE_MASK
                    if current_state > MAX_STATE {
                        return Err(ZiporaError::invalid_data(
                            "State value exceeds maximum during relocation",
                        ));
                    }
                    // Allocate the state (referenced project: set_parent clears free bit)
                    check[new_next as usize] = current_state; // No free bit
                    // Initialize base to NIL_STATE - will be set when children are added
                    // (referenced project line 354-355: set to 1 for unused states)
                    base[new_next as usize] = NIL_STATE;
                    current_state = new_next;
                    *state_count += 1;
                }
            }

        }

        // Mark the final state as terminal (referenced project: set_term_bit on base at line 27)
        // Check if this is a new key or duplicate
        let was_new = (base[current_state as usize] & TERMINAL_BIT) == 0;
        base[current_state as usize] |= TERMINAL_BIT;

        // Only increment key count if this was a new key
        if was_new {
            *num_keys += 1;
        }

        // Debug: Verify what we just inserted
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "DEBUG insert_double_array: Inserted key, final state={}, base[{}]={:08x}, check[{}]={:08x}, was_new={}",
                current_state,
                current_state,
                base[current_state as usize],
                current_state,
                check[current_state as usize],
                was_new
            );
        }

        Ok(current_state)
    }

    // Helper: Find a free base value for a state that doesn't conflict
    // For incremental insert, use a proper heuristic matching referenced project's approach
    pub(super) fn find_free_base(_base: &FastVec<u32>, check: &FastVec<u32>, _state: u32) -> Result<u32> {
        const FREE_BIT: u32 = 0x8000_0000;
        const NIL_STATE: u32 = 0x7FFF_FFFF;

        // Start search from position 1 (0 is root)
        let mut candidate = 1u32;
        let len = check.len();

        // Linear probe for a free position (matching C++ reference heuristic)
        while (candidate as usize) < len {
            let check_val = check[candidate as usize];
            let is_free = check_val == (NIL_STATE | FREE_BIT) || (check_val & FREE_BIT) != 0;
            if is_free {
                return Ok(candidate);
            }
            candidate += 1;
        }

        // Past the end of array — return the next position (will trigger array growth)
        Ok(candidate)
    }

    // Helper: Relocate a state and all its children to use a new base value
    pub(super) fn relocate_state(
        base: &mut FastVec<u32>,
        check: &mut FastVec<u32>,
        state: u32,
        new_symbol: u8,
        _state_count: &mut usize,
    ) -> Result<u32> {
        const VALUE_MASK: u32 = 0x7FFF_FFFF; // Bits 0-30 for values (referenced project)
        const TERMINAL_BIT: u32 = 0x8000_0000; // Bit 31 in base for terminal (referenced project)
        const FREE_BIT: u32 = 0x8000_0000; // Bit 31 in check for free (referenced project)
        const NIL_STATE: u32 = 0x7FFF_FFFF; // Match referenced project's nil_state

        // Special handling for root state - try to avoid relocating it
        if state == 0 {
            // For root, try to find a different base that works
            // This is critical because relocating root affects the entire trie
            #[cfg(debug_assertions)]
            eprintln!("  WARNING: Attempting to relocate root state - this may cause issues");
        }

        let old_base = base[state as usize] & VALUE_MASK;

        // Collect all existing children of this state with their base and terminal info
        let mut children = Vec::new();
        for symbol in 0u8..=255u8 {
            let child_pos = old_base.saturating_add(symbol as u32);
            if (child_pos as usize) < check.len() {
                let check_val = check[child_pos as usize];
                // Check if this is an allocated child (not free, parent matches)
                if (check_val & FREE_BIT) == 0 && check_val == state {
                    // This is a child of our state - save its info
                    let child_base = if (child_pos as usize) < base.len() {
                        base[child_pos as usize]
                    } else {
                        NIL_STATE
                    };
                    let is_terminal = (child_base & TERMINAL_BIT) != 0;
                    children.push((symbol, child_pos, child_base, is_terminal));
                }
            }
        }

        // Find a new base where we can place all children plus the new symbol
        // Use find_free_base to get a better starting point that spreads states out
        let initial_base = Self::find_free_base(base, check, state)?;
        let mut new_base = initial_base;
        let mut attempts = 0;
        const MAX_BASE: u32 = u32::MAX - 256; // Leave room for 256 symbols

        'search: loop {
            if attempts > 1_000_000 || new_base > MAX_BASE {
                return Err(ZiporaError::invalid_data(
                    "Cannot relocate state in double array",
                ));
            }
            attempts += 1;

            // Check if new_base works for the new symbol
            let new_pos = new_base.saturating_add(new_symbol as u32);

            // Ensure arrays are large enough
            let max_pos = children
                .iter()
                .map(|(sym, _, _, _)| new_base.saturating_add(*sym as u32))
                .chain(std::iter::once(new_pos))
                .max()
                .unwrap_or(new_pos);

            // Expand arrays if needed - use amortized growth
            let required = max_pos as usize + 1;
            if required > base.len() {
                let new_size = required.max(base.len() * 3 / 2).max(256);
                let _ = base.resize(new_size, NIL_STATE);
                let _ = check.resize(new_size, NIL_STATE | FREE_BIT);
            }

            // CRITICAL: Never allow any child to be relocated to state 0
            // Check if new position for new_symbol is free and not state 0
            let new_pos_check = check[new_pos as usize];
            let new_pos_is_free = (new_pos_check & FREE_BIT) != 0;
            if new_pos == 0 || !new_pos_is_free {
                // State 0 is reserved or position is occupied
                // Use smaller increment for denser packing
                new_base = new_base.saturating_add(1);
                continue 'search;
            }

            // Check if all children can be relocated (and none would go to state 0)
            for (symbol, _, _, _) in &children {
                let test_pos = new_base.saturating_add(*symbol as u32);
                let test_check = check[test_pos as usize];
                let test_is_free = (test_check & FREE_BIT) != 0;
                // CRITICAL: Reject if any child would be relocated to state 0
                if test_pos == 0 || !test_is_free {
                    // State 0 is reserved or position is occupied
                    new_base = new_base.saturating_add(1);
                    continue 'search;
                }
            }

            // Found a suitable new base - relocate all children
            // First, mark old positions as free
            for (_, old_pos, _, _) in &children {
                check[*old_pos as usize] = NIL_STATE | FREE_BIT;
                // Mark base as NIL to indicate it's free
                base[*old_pos as usize] = NIL_STATE;
            }

            // Then, set new positions with both check and base values
            for (symbol, old_pos, child_base_val, is_terminal) in &children {
                let new_child_pos = new_base.saturating_add(*symbol as u32);
                // Allocate the new position (referenced project: set_parent clears free bit)
                check[new_child_pos as usize] = state; // Parent state, no free bit
                // Set base value, preserving terminal bit if needed
                let base_value = child_base_val & VALUE_MASK;
                base[new_child_pos as usize] = if *is_terminal {
                    base_value | TERMINAL_BIT
                } else {
                    base_value
                };

                // CRITICAL: Update any grandchildren that point to the old child position
                // to point to the new child position
                if base_value != 0 && base_value != NIL_STATE {
                    Self::update_grandchildren_check_values(base, check, *old_pos, new_child_pos);
                }
            }

            // Update the base for this state (preserve terminal bit if state is terminal)
            let state_base = base[state as usize];
            let state_is_terminal = (state_base & TERMINAL_BIT) != 0;
            base[state as usize] = if state_is_terminal {
                new_base | TERMINAL_BIT
            } else {
                new_base
            };

            return Ok(new_base);
        }
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
        for (_i, &symbol) in key.iter().enumerate() {
            #[cfg(debug_assertions)]
            let i = _i;
            // SAFETY: We check if base_val exists, then use it
            let base_val = match base.get(current_state as usize) {
                Some(val) => val,
                None => {
                    #[cfg(debug_assertions)]
                    eprintln!("DEBUG contains: No base for state {}", current_state);
                    return false;
                }
            };

            // Calculate next state using base value (bits 0-30)
            let next_state = (base_val & VALUE_MASK).saturating_add(symbol as u32);

            // Check if the transition is valid (referenced project line 106: states[next].parent() == curr)
            if next_state as usize >= check.len() {
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG contains: next_state {} >= check.len() {}",
                    next_state,
                    check.len()
                );
                return false;
            }

            let check_val = check[next_state as usize];
            // Direct comparison like referenced project: check[next] == current_state
            // Free states have FREE_BIT set, so won't match
            if check_val != current_state {
                // Invalid transition
                #[cfg(debug_assertions)]
                eprintln!(
                    "DEBUG contains: Invalid transition at pos {}, symbol {:02x}, state {}->{}, check[{}]={:08x}, expected parent {}",
                    i, symbol, current_state, next_state, next_state, check_val, current_state
                );
                return false;
            }

            current_state = next_state;

        }

        // Check if the final state is marked as terminal (check terminal bit in base)
        let is_terminal = base
            .get(current_state as usize)
            .map(|b| (b & TERMINAL_BIT) != 0)
            .unwrap_or(false);

        #[cfg(debug_assertions)]
        {
            let base_val = base.get(current_state as usize).unwrap_or(&0);
            let check_val = check.get(current_state as usize).unwrap_or(&0);
            eprintln!(
                "DEBUG contains: Final state={}, base[{}]={:08x}, check[{}]={:08x}, is_terminal={}",
                current_state, current_state, base_val, current_state, check_val, is_terminal
            );
        }

        is_terminal
    }

    // LOUDS trie implementation methods
    //  TODO: port from C++ reference `src/terark/fsa/nest_louds_trie.hpp`
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
                // Create new child node
                let new_node_id = nodes.len();
                let _ = nodes.push(PatriciaNode::default());

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
                // Remove the child pointer from parent
                if let Ok(idx) = nodes[parent_idx]
                    .children
                    .binary_search_by_key(&symbol, |(s, _)| *s)
                {
                    nodes[parent_idx].children.remove(idx);
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
