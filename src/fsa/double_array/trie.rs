use crate::error::{Result, ZiporaError};

use super::state::*;

///
/// ```rust
/// use zipora::fsa::double_array::DoubleArrayTrie;
///
/// let mut trie = DoubleArrayTrie::new();
/// trie.insert(b"hello").unwrap();
/// trie.insert(b"help").unwrap();
/// trie.insert(b"world").unwrap();
///
/// assert!(trie.contains(b"hello"));
/// assert!(trie.contains(b"help"));
/// assert!(!trie.contains(b"hel"));
/// assert_eq!(trie.len(), 3);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleArrayTrie {
    pub(crate) states: Vec<DaState>,
    pub(crate) ninfos: Vec<NInfo>,
    num_keys: usize,
    /// Heuristic search position
    search_head: usize,
}

impl DoubleArrayTrie {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create with pre-allocated capacity.
    ///
    /// Minimum capacity is 256 to guarantee that `base ^ ch` (where ch is
    /// any byte 0-255) is always in-bounds for states with `child0 = 0`.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(256);
        let mut states = Vec::with_capacity(cap);
        states.push(DaState::new_root());
        states.resize(cap, DaState::new_free());

        let ninfos = vec![NInfo::default(); cap];

        Self {
            states,
            ninfos,
            num_keys: 0,
            search_head: 1,
        }
    }

    /// Number of keys in the trie.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Check if the trie is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.num_keys == 0
    }

    /// Total number of allocated states.
    #[inline]
    pub fn total_states(&self) -> usize {
        self.states.len()
    }

    /// Memory usage in bytes.
    #[inline]
    pub fn mem_size(&self) -> usize {
        self.states.len() * std::mem::size_of::<DaState>()
            + self.ninfos.len() * std::mem::size_of::<NInfo>()
    }

    /// Check if a state is terminal.
    #[inline(always)]
    pub fn is_term(&self, state: u32) -> bool {
        (state as usize) < self.ninfos.len() && self.ninfos[state as usize].is_term()
    }

    /// Check if a state is free.
    #[inline(always)]
    pub fn is_free(&self, state: u32) -> bool {
        (state as usize) >= self.states.len() || self.states[state as usize].is_free()
    }

    /// Single state transition: `next = base[curr] ^ ch`, valid if `check[next] == curr`.
    /// Returns NIL_STATE if transition doesn't exist.
    #[inline(always)]
    pub fn state_move(&self, curr: u32, ch: u8) -> u32 {
        // SAFETY: `curr` is typically from a previous state_move or 0 (root).
        let base = unsafe { self.states.get_unchecked(curr as usize) }.child0();
        let next = (base ^ ch as u32) as usize;
        // SAFETY: same invariant as contains()
        debug_assert!(next < self.states.len());
        let next_state = unsafe { self.states.get_unchecked(next) };

        if next_state.parent == curr {
            next as u32
        } else {
            NIL_STATE
        }
    }

    /// Insert a key. Returns true if the key was new.
    #[inline]
    pub fn insert(&mut self, key: &[u8]) -> Result<bool> {
        self.insert_with_relocate_cb(key, |_, _| {})
    }

    /// Insert a key, calling `on_relocate(old_pos, new_pos)` whenever a child
    /// state is moved during collision resolution. This lets external value
    /// arrays stay in sync with state IDs.
    ///
    /// Uses consult (relocate-smaller-side) to minimize total relocations.
    pub fn insert_with_relocate_cb(
        &mut self,
        key: &[u8],
        mut on_relocate: impl FnMut(u32, u32),
    ) -> Result<bool> {
        if key.is_empty() {
            let was_new = !self.ninfos[0].is_term();
            self.ninfos[0].set_term();
            if was_new {
                self.num_keys += 1;
            }
            return Ok(was_new);
        }

        let mut curr = 0u32;

        for &ch in key {
            // child0 == 0 means "no children" (find_free_base never returns 0).
            let base = self.states[curr as usize].child0;
            if base == 0 {
                let new_base = self.find_free_base(&[ch])?;
                self.set_base_padded(curr as usize, new_base);
                let next = new_base ^ ch as u32;
                self.ensure_capacity(next as usize + 1);

                self.states[next as usize].child0 = 0;
                self.states[next as usize].set_parent(curr);
                self.add_child_link(curr as usize, ch);
                curr = next;
            } else {
                let next = base ^ ch as u32;
                self.ensure_capacity(next as usize + 1);

                if !self.states[next as usize].is_free()
                    && self.states[next as usize].parent() == curr
                {
                    curr = next;
                } else if self.states[next as usize].is_free() {
                    self.states[next as usize].child0 = 0;
                    self.states[next as usize].set_parent(curr);
                    self.add_child_link(curr as usize, ch);
                    curr = next;
                } else {
                    // Conflict — position occupied by another parent's child.
                    // Consult: relocate the side with fewer children.
                    let conflict_parent = self.states[next as usize].parent();

                    // Guard: conflict_parent must be valid and not an ancestor
                    // of curr (relocating an ancestor would corrupt traversal).
                    let can_consult = (conflict_parent as usize) < self.ninfos.len()
                        && conflict_parent != curr
                        && !self.is_ancestor(conflict_parent, curr);

                    if can_consult {
                        let curr_n = self.count_children(curr) + 1;
                        let conf_n = self.count_children(conflict_parent);
                        if curr_n > conf_n {
                            // Relocate conflict parent (fewer children)
                            let old_base_cf = self.states[conflict_parent as usize].child0();
                            self.relocate_existing(conflict_parent)?;
                            let new_base_cf = self.states[conflict_parent as usize].child0();

                            // Notify callback about conflict parent's moved children
                            Self::notify_relocated(
                                &self.ninfos,
                                conflict_parent as usize,
                                old_base_cf,
                                new_base_cf,
                                &mut on_relocate,
                            );

                            self.states[next as usize].child0 = 0;
                            self.states[next as usize].set_parent(curr);
                            self.add_child_link(curr as usize, ch);
                            curr = next;
                            continue;
                        }
                    }

                    // Default: relocate curr's children
                    let old_base = self.states[curr as usize].child0();
                    let new_base = self.relocate(curr, ch)?;

                    Self::notify_relocated_excluding(
                        &self.ninfos,
                        curr as usize,
                        old_base,
                        new_base,
                        ch,
                        &mut on_relocate,
                    );

                    let next = new_base ^ ch as u32;
                    self.ensure_capacity(next as usize + 1);
                    self.states[next as usize].child0 = 0;
                    self.states[next as usize].set_parent(curr);
                    self.add_child_link(curr as usize, ch);
                    curr = next;
                }
            }
        }

        let was_new = !self.ninfos[curr as usize].is_term();
        self.ninfos[curr as usize].set_term();
        if was_new {
            self.num_keys += 1;
        }
        Ok(was_new)
    }

    /// Notify callback about all moved children of a parent after relocation.
    fn notify_relocated(
        ninfos: &[NInfo],
        parent_pos: usize,
        old_base: u32,
        new_base: u32,
        on_relocate: &mut impl FnMut(u32, u32),
    ) {
        let mut c = ninfos[parent_pos].first_child();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            on_relocate(old_base ^ label as u32, new_base ^ label as u32);
            let child_pos = (new_base ^ label as u32) as usize;
            c = if child_pos < ninfos.len() {
                ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    /// Notify callback about moved children, excluding `exclude_ch` (new child).
    fn notify_relocated_excluding(
        ninfos: &[NInfo],
        parent_pos: usize,
        old_base: u32,
        new_base: u32,
        exclude_ch: u8,
        on_relocate: &mut impl FnMut(u32, u32),
    ) {
        let mut c = ninfos[parent_pos].first_child();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            if label != exclude_ch {
                on_relocate(old_base ^ label as u32, new_base ^ label as u32);
            }
            let child_pos = (new_base ^ label as u32) as usize;
            c = if child_pos < ninfos.len() {
                ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    /// Check if a key exists — tight loop, 1 branch per transition.
    ///
    /// # Safety invariant
    /// All allocated states have child0 set by `set_base_padded` (valid base)
    /// or initialized to 0 (leaf). Both guarantee `child0 ^ ch < states.len()`
    /// for any ch in 0..=255. Root parent is NIL_STATE (never matches curr).
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        let states = self.states.as_slice();

        if key.is_empty() {
            return self.ninfos[0].is_term();
        }

        let ninfos = self.ninfos.as_slice();
        let mut curr = 0usize;
        for &ch in key {
            // SAFETY: curr < states.len() from parent match validation in previous iteration or curr=0 (root)
            let base = unsafe { states.get_unchecked(curr) }.child0;
            let next = (base ^ ch as u32) as usize;
            // SAFETY: set_base_padded guarantees (base | 0xFF) < len for valid bases.
            // Leaf states have child0=0, so next=ch ∈ [0,255] < 256 ≤ len.
            // Free states have parent with FREE_BIT set, never matching curr.
            debug_assert!(
                next < states.len(),
                "OOB: next={next}, len={}",
                states.len()
            );
            // SAFETY: next = base ^ ch < states.len() verified by set_base_padded invariant and debug_assert above
            let next_state = unsafe { states.get_unchecked(next) };
            if next_state.parent != curr as u32 {
                return false;
            }
            curr = next;
        }

        // SAFETY: curr has been verified to be < states.len() (and therefore ninfos.len() which is the same size)
        unsafe { ninfos.get_unchecked(curr) }.is_term()
    }

    /// Lookup key and return its terminal state, or None.
    #[inline]
    pub fn lookup_state(&self, key: &[u8]) -> Option<u32> {
        let states = self.states.as_slice();

        if key.is_empty() {
            return if self.ninfos[0].is_term() {
                Some(0)
            } else {
                None
            };
        }

        let ninfos = self.ninfos.as_slice();
        let mut curr = 0usize;
        for &ch in key {
            let base = states[curr].child0;
            let next = (base ^ ch as u32) as usize;
            // SAFETY: same invariant as contains()
            debug_assert!(next < states.len());
            let next_state = unsafe { states.get_unchecked(next) };
            if next_state.parent != curr as u32 {
                return None;
            }
            curr = next;
        }

        if ninfos[curr].is_term() {
            Some(curr as u32)
        } else {
            None
        }
    }

    /// Remove a key. Returns true if the key existed.
    ///
    /// After clearing the terminal flag, prunes dead-end nodes that are
    /// no longer part of any valid key path (leaf states with no children
    /// and no terminal flag).
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if let Some(state) = self.lookup_state(key)
            && self.ninfos[state as usize].is_term()
        {
            self.ninfos[state as usize].clear_term();
            self.num_keys -= 1;
            self.prune_dead_branch(state);
            return true;
        }
        false
    }

    /// Restore the key string from a state by walking the parent chain.
    pub fn restore_key(&self, state: u32) -> Option<Vec<u8>> {
        if state as usize >= self.states.len() {
            return None;
        }
        if self.states[state as usize].is_free() {
            return None;
        }

        let mut symbols = Vec::new();
        let mut curr = state;

        while curr != 0 {
            let parent = self.states[curr as usize].parent();
            let parent_base = self.states[parent as usize].child0();
            let symbol = (curr ^ parent_base) as u8;
            symbols.push(symbol);
            curr = parent;
        }

        symbols.reverse();
        Some(symbols)
    }

    /// Prune dead-end nodes after a key removal.
    /// Walks up the parent chain freeing non-terminal leaf nodes and
    /// removing them from their parent's NInfo child chain.
    fn prune_dead_branch(&mut self, state: u32) {
        let mut curr = state;

        while curr != 0 {
            // Stop if this state is terminal or has children
            if self.ninfos[curr as usize].is_term() {
                break;
            }
            if self.ninfos[curr as usize].first_child() != NINFO_NONE {
                break;
            }

            // This state is a dead leaf — remove it
            let parent = self.states[curr as usize].parent();
            if parent as usize >= self.states.len() {
                break;
            }

            let parent_base = self.states[parent as usize].child0();
            let label = (curr ^ parent_base) as u8;

            // Remove from parent's NInfo child chain
            self.remove_child_link(parent as usize, label);

            // Free the state
            self.ninfos[curr as usize] = NInfo::default();
            self.states[curr as usize].set_free();

            // Continue up to parent
            curr = parent;
        }
    }

    /// Remove a label from the sorted sibling chain of parent_pos.
    fn remove_child_link(&mut self, parent_pos: usize, label: u8) {
        let label_enc = label_to_ninfo(label);
        let base = self.states[parent_pos].child0();

        let first = self.ninfos[parent_pos].first_child();
        if first == NINFO_NONE {
            return;
        }

        if first == label_enc {
            // Removing the first child — update parent's child pointer
            let child_pos = (base ^ label as u32) as usize;
            let next = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
            self.ninfos[parent_pos].set_first_child(next);
            return;
        }

        // Walk chain to find and unlink the label
        let mut prev_enc = first;
        loop {
            let prev_label = (prev_enc - 1) as u8;
            let prev_pos = (base ^ prev_label as u32) as usize;
            if prev_pos >= self.ninfos.len() {
                break;
            }
            let next_enc = self.ninfos[prev_pos].sibling;
            if next_enc == label_enc {
                // Found it — unlink by pointing prev to label's next
                let label_pos = (base ^ label as u32) as usize;
                let after = if label_pos < self.ninfos.len() {
                    self.ninfos[label_pos].sibling
                } else {
                    NINFO_NONE
                };
                self.ninfos[prev_pos].sibling = after;
                return;
            }
            if next_enc == NINFO_NONE {
                break;
            }
            prev_enc = next_enc;
        }
    }

    /// Get all keys in the trie.
    pub fn keys(&self) -> Vec<Vec<u8>> {
        let mut result = Vec::with_capacity(self.num_keys);
        let mut path = Vec::new();
        self.collect_keys(0, &mut path, &mut result);
        result
    }

    /// Get all keys with a given prefix.
    pub fn keys_with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        // Navigate to prefix state
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.state_move(curr, ch);
            if next == NIL_STATE {
                return Vec::new();
            }
            curr = next;
        }

        let mut result = Vec::new();
        let mut path = prefix.to_vec();
        self.collect_keys(curr, &mut path, &mut result);
        result
    }

    /// Iterate all children of a state, calling `f(symbol, child_state)`.
    #[inline]
    pub fn for_each_child(&self, state: u32, mut f: impl FnMut(u8, u32)) {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return;
        }
        let base = self.states[state as usize].child0();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                f(label, child_pos as u32);
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    /// Find the child with the given symbol, or the next higher child.
    /// Returns (index, exact_match) where index is into get_children() result.
    #[inline]
    fn lower_bound_child(&self, state: u32, symbol: u8) -> Option<(u8, u32)> {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return None;
        }
        let base = self.states[state as usize].child0();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            if label < symbol {
                let child_pos = (base ^ label as u32) as usize;
                c = if child_pos < self.ninfos.len() {
                    self.ninfos[child_pos].sibling
                } else {
                    NINFO_NONE
                };
                continue;
            }
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                return Some((label, child_pos as u32));
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
        None
    }

    /// Find the highest child with symbol < given symbol.
    #[inline]
    fn prev_child(&self, state: u32, symbol: u32) -> Option<(u8, u32)> {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return None;
        }
        let base = self.states[state as usize].child0();
        let mut result = None;
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if (label as u32) < symbol
                && child_pos < self.states.len()
                && !self.states[child_pos].is_free()
            {
                result = Some((label, child_pos as u32));
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
        result
    }

    /// Get the first (lowest symbol) child of a state.
    #[inline]
    fn first_child(&self, state: u32) -> Option<(u8, u32)> {
        let c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return None;
        }
        let base = self.states[state as usize].child0();

        let label = (c - 1) as u8;
        let child_pos = (base ^ label as u32) as usize;
        if child_pos < self.states.len() && !self.states[child_pos].is_free() {
            Some((label, child_pos as u32))
        } else {
            None
        }
    }

    /// Get the last (highest symbol) child of a state.
    #[inline]
    fn last_child(&self, state: u32) -> Option<(u8, u32)> {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return None;
        }
        let base = self.states[state as usize].child0();
        let mut result = None;
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                result = Some((label, child_pos as u32));
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
        result
    }

    /// Call `f` for each key with the given prefix — zero allocation.
    ///
    /// The callback receives a `&[u8]` reference to the key bytes.
    /// Unlike `keys_with_prefix()`, this does not allocate `Vec<Vec<u8>>`.
    pub fn for_each_key_with_prefix(&self, prefix: &[u8], mut f: impl FnMut(&[u8])) {
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.state_move(curr, ch);
            if next == NIL_STATE {
                return;
            }
            curr = next;
        }
        let mut path = prefix.to_vec();
        self.walk_keys(curr, &mut path, &mut f);
    }

    /// Internal DFS for callback-based key iteration.
    fn walk_keys(&self, state: u32, path: &mut Vec<u8>, f: &mut impl FnMut(&[u8])) {
        if state as usize >= self.states.len() {
            return;
        }

        if self.ninfos[state as usize].is_term() {
            f(path);
        }

        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return;
        }
        let base = self.states[state as usize].child0();

        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                path.push(label);
                self.walk_keys(child_pos as u32, path, f);
                path.pop();
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    /// Build from sorted keys (more efficient than incremental insert).
    pub fn build_from_sorted(keys: &[&[u8]]) -> Result<Self> {
        if keys.is_empty() {
            return Ok(Self::new());
        }

        // Estimate total states needed
        let total_bytes: usize = keys.iter().map(|k| k.len()).sum();
        let estimated_states = (total_bytes / 2).max(256);
        let mut trie = Self::with_capacity(estimated_states * 3 / 2);

        for &key in keys {
            trie.insert(key)?;
        }

        trie.shrink_to_fit();
        Ok(trie)
    }

    // =========================================================================
    // Consult — relocate the smaller side on conflict
    // =========================================================================

    /// Count children of a state using sibling chain (O(k)).
    #[inline]
    fn count_children(&self, state: u32) -> usize {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return 0;
        }
        let base = self.states[state as usize].child0();
        let mut count = 0;
        while c != NINFO_NONE {
            count += 1;
            let label = (c - 1) as u8;
            let pos = (base ^ label as u32) as usize;
            c = if pos < self.ninfos.len() {
                self.ninfos[pos].sibling
            } else {
                NINFO_NONE
            };
        }
        count
    }

    /// Shrink internal arrays to fit actual usage.
    /// Preserves XOR padding invariant: (base | 0xFF) + 1 must be in-bounds.
    pub fn shrink_to_fit(&mut self) {
        // Find highest reachable position to preserve padding invariant
        let mut max_reachable = 0usize;
        for (i, s) in self.states.iter().enumerate() {
            if !s.is_free() {
                // Check if this state actually has children (via NInfo)
                if i < self.ninfos.len() && self.ninfos[i].first_child() != NINFO_NONE {
                    let base = s.child0();
                    // With XOR, max child pos = base | 0xFF
                    max_reachable = max_reachable.max((base as usize | 0xFF) + 1);
                }
            }
        }

        let last_used = self.states.iter().rposition(|s| !s.is_free()).unwrap_or(0);
        // Never shrink below 256: leaf states have child0=0, so contains() does
        // get_unchecked(0 ^ ch) with ch up to 255 (same invariant as with_capacity).
        let new_len = (last_used + 1)
            .max(max_reachable)
            .max(256)
            .min(self.states.len());
        self.states.truncate(new_len);
        self.states.shrink_to_fit();
        self.ninfos.truncate(new_len);
        self.ninfos.shrink_to_fit();

        self.search_head = 1;
    }

    // --- Internal methods ---

    /// Insert label into the sorted sibling chain of parent_pos.
    fn add_child_link(&mut self, parent_pos: usize, label: u8) {
        let label_enc = label_to_ninfo(label);
        let base = self.states[parent_pos].child0();

        let first = self.ninfos[parent_pos].first_child();
        if first == NINFO_NONE || label_enc < first {
            // New first child
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling = first;
            }
            self.ninfos[parent_pos].set_first_child(label_enc);
        } else if first == label_enc {
            // Already first child
        } else {
            // Walk chain to find insertion point
            let mut prev_enc = first;
            loop {
                let prev_label = (prev_enc - 1) as u8;
                let prev_pos = (base ^ prev_label as u32) as usize;
                if prev_pos >= self.ninfos.len() {
                    break;
                }
                let next_enc = self.ninfos[prev_pos].sibling;
                if next_enc == NINFO_NONE || label_enc < next_enc {
                    let child_pos = (base ^ label as u32) as usize;
                    if child_pos < self.ninfos.len() {
                        self.ninfos[child_pos].sibling = next_enc;
                        self.ninfos[prev_pos].sibling = label_enc;
                    }
                    break;
                }
                if next_enc == label_enc {
                    break;
                } // Already present
                prev_enc = next_enc;
            }
        }
    }

    /// Ensure states array is large enough, using 1.5x amortized growth.
    /// New states are NOT added to the free list — they're detected as free
    /// by `is_free()`. Only explicitly freed states go on the free list.
    #[inline]
    fn ensure_capacity(&mut self, required: usize) {
        if required <= self.states.len() {
            return;
        }
        let new_len = required.max(self.states.len() * 3 / 2).max(256);
        self.states.resize(new_len, DaState::new_free());
        self.ninfos.resize(new_len, NInfo::default());
    }

    /// Set base and ensure all XOR-reachable positions are in-bounds.
    /// With XOR transitions, child positions span [base & !0xFF, base | 0xFF].
    #[inline]
    fn set_base_padded(&mut self, state: usize, base: u32) {
        self.states[state].set_child0(base);
        self.ensure_capacity((base as usize | 0xFF) + 1);
    }

    /// Find a free base value where all given children symbols can be placed.
    ///
    /// Uses search_head to skip past occupied regions at the start, then
    /// probes candidates by incrementing base. For single-child inserts,
    /// jumps directly to the next free position when the candidate is occupied.
    fn find_free_base(&mut self, children: &[u8]) -> Result<u32> {
        debug_assert!(!children.is_empty());

        let ch0 = children[0] as u32;
        let single = children.len() == 1;

        // Advance search_head past any occupied region
        while self.search_head < self.states.len() && !self.states[self.search_head].is_free() {
            self.search_head += 1;
        }

        let mut base = (self.search_head as u32) ^ ch0;
        if base == 0 {
            base = 1;
        }

        let mut attempts = 0u32;

        loop {
            if attempts > 1_000_000 || base > MAX_STATE {
                return Err(ZiporaError::invalid_data(
                    "Double array: cannot find free base",
                ));
            }
            attempts += 1;

            self.ensure_capacity((base as usize | 0xFF) + 1);

            let first_pos = (base ^ ch0) as usize;
            if first_pos == 0 || !self.states[first_pos].is_free() {
                base += 1;
                continue;
            }

            if single {
                self.search_head = first_pos;
                return Ok(base);
            }

            // Multi-child: check remaining positions
            let all_free = children[1..].iter().all(|&ch| {
                let pos = (base ^ ch as u32) as usize;
                pos > 0 && self.states[pos].is_free()
            });

            if all_free {
                self.search_head = first_pos;
                return Ok(base);
            }
            base += 1;
        }
    }

    /// Relocate all children of `state` to a new base that also accommodates `new_ch`.
    fn relocate(&mut self, state: u32, new_ch: u8) -> Result<u32> {
        // Collect existing children via NInfo chain
        let old_base = self.states[state as usize].child0();
        let mut children_symbols = Vec::new();

        {
            let mut c = self.ninfos[state as usize].first_child();
            while c != NINFO_NONE {
                let label = (c - 1) as u8;
                let child_pos = (old_base ^ label as u32) as usize;
                if child_pos < self.states.len()
                    && !self.states[child_pos].is_free()
                    && !children_symbols.contains(&label)
                {
                    children_symbols.push(label);
                }
                c = if child_pos < self.ninfos.len() {
                    self.ninfos[child_pos].sibling
                } else {
                    NINFO_NONE
                };
            }
        }

        if !children_symbols.contains(&new_ch) {
            children_symbols.push(new_ch);
        }
        children_symbols.sort_unstable();

        let new_base = self.find_free_base(&children_symbols)?;

        self.ninfos[state as usize].set_first_child(NINFO_NONE);

        // Move existing children to new positions
        {
            for &ch in &children_symbols {
                if ch == new_ch {
                    continue;
                }
                let old_pos = old_base ^ ch as u32;
                let new_pos = new_base ^ ch as u32;

                if old_pos as usize >= self.states.len() {
                    continue;
                }
                if self.states[old_pos as usize].is_free() {
                    continue;
                }
                if self.states[old_pos as usize].parent() != state {
                    continue;
                }

                self.ensure_capacity(new_pos as usize + 1);

                let old_state = self.states[old_pos as usize];
                self.states[new_pos as usize].child0 = old_state.child0;
                self.states[new_pos as usize].set_parent(state);

                let old_ninfo = self.ninfos[old_pos as usize];
                self.ninfos[new_pos as usize].child = old_ninfo.child;
                self.ninfos[new_pos as usize].sibling = NINFO_NONE;

                // Update grandchildren to point to new parent position
                {
                    let mut gc = old_ninfo.first_child();
                    if gc != NINFO_NONE {
                        let child_base = old_state.child0();
                        while gc != NINFO_NONE {
                            let glabel = (gc - 1) as u8;
                            let gpos = (child_base ^ glabel as u32) as usize;
                            if gpos < self.states.len()
                                && !self.states[gpos].is_free()
                                && self.states[gpos].parent() == old_pos
                            {
                                self.states[gpos].set_parent(new_pos);
                            }
                            gc = if gpos < self.ninfos.len() {
                                self.ninfos[gpos].sibling
                            } else {
                                NINFO_NONE
                            };
                        }
                    }
                }

                self.ninfos[old_pos as usize] = NInfo::default();
                self.states[old_pos as usize].set_free();
            }
        }

        // Update state's base to new location
        self.set_base_padded(state as usize, new_base);

        // Rebuild parent's NInfo chain from scratch (excluding new_ch, caller will add it)
        for &ch in &children_symbols {
            if ch != new_ch {
                self.add_child_link(state as usize, ch);
            }
        }

        Ok(new_base)
    }

    /// Relocate ALL existing children of `state` without adding a new child.
    /// Used by consult when relocating the conflict parent.
    /// Returns the new base value.
    fn relocate_existing(&mut self, state: u32) -> Result<u32> {
        let old_base = self.states[state as usize].child0();

        let mut children_symbols: Vec<u8> = Vec::new();
        let mut c = self.ninfos[state as usize].first_child();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (old_base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                children_symbols.push(label);
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }

        if children_symbols.is_empty() {
            return Ok(old_base);
        }
        children_symbols.sort_unstable();

        let new_base = self.find_free_base(&children_symbols)?;

        self.ninfos[state as usize].set_first_child(NINFO_NONE);

        for &ch in &children_symbols {
            let old_pos = old_base ^ ch as u32;
            let new_pos = new_base ^ ch as u32;

            if old_pos as usize >= self.states.len() {
                continue;
            }
            if self.states[old_pos as usize].is_free() {
                continue;
            }
            if self.states[old_pos as usize].parent() != state {
                continue;
            }

            self.ensure_capacity(new_pos as usize + 1);

            let old_state = self.states[old_pos as usize];
            self.states[new_pos as usize].child0 = old_state.child0;
            self.states[new_pos as usize].set_parent(state);

            let old_ninfo = self.ninfos[old_pos as usize];
            self.ninfos[new_pos as usize].child = old_ninfo.child;
            self.ninfos[new_pos as usize].sibling = NINFO_NONE;

            {
                let mut gc = old_ninfo.first_child();
                if gc != NINFO_NONE {
                    let child_base = old_state.child0();
                    while gc != NINFO_NONE {
                        let glabel = (gc - 1) as u8;
                        let gpos = (child_base ^ glabel as u32) as usize;
                        if gpos < self.states.len()
                            && !self.states[gpos].is_free()
                            && self.states[gpos].parent() == old_pos
                        {
                            self.states[gpos].set_parent(new_pos);
                        }
                        gc = if gpos < self.ninfos.len() {
                            self.ninfos[gpos].sibling
                        } else {
                            NINFO_NONE
                        };
                    }
                }
            }

            self.ninfos[old_pos as usize] = NInfo::default();
            self.states[old_pos as usize].set_free();
        }

        self.set_base_padded(state as usize, new_base);

        for &ch in &children_symbols {
            self.add_child_link(state as usize, ch);
        }

        Ok(new_base)
    }

    /// Check if `ancestor` is an ancestor of `descendant` in the trie.
    ///
    /// Required for correctness in consult (relocate-smaller-side): relocating
    /// an ancestor of `curr` would invalidate our traversal state. The depth-256
    /// bound prevents infinite loops on malformed parent chains.
    fn is_ancestor(&self, ancestor: u32, descendant: u32) -> bool {
        let mut curr = descendant;
        let mut depth = 0;
        while curr != 0 && depth < 256 {
            let parent = self.states[curr as usize].parent();
            if parent == ancestor {
                return true;
            }
            curr = parent;
            depth += 1;
        }
        false
    }

    /// Recursively collect keys via DFS.
    fn collect_keys(&self, state: u32, path: &mut Vec<u8>, keys: &mut Vec<Vec<u8>>) {
        if state as usize >= self.states.len() {
            return;
        }

        if self.ninfos[state as usize].is_term() {
            keys.push(path.clone());
        }

        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return;
        }
        let base = self.states[state as usize].child0();

        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                path.push(label);
                self.collect_keys(child_pos as u32, path, keys);
                path.pop();
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }
}

impl Default for DoubleArrayTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DoubleArrayTrie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoubleArrayTrie")
            .field("num_keys", &self.num_keys)
            .field("total_states", &self.states.len())
            .field("mem_size", &self.mem_size())
            .finish()
    }
}

/// Bidirectional lexicographic cursor over a `DoubleArrayTrie`.
///
/// Supports `seek_lower_bound`, `next`, `prev`, `seek_begin`, `seek_end`.
/// Matching the C++ reference's `ADFA_LexIterator` interface.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::double_array::DoubleArrayTrie;
///
/// let mut trie = DoubleArrayTrie::new();
/// for word in &["apple", "banana", "cherry", "date", "elderberry"] {
///     trie.insert(word.as_bytes()).unwrap();
/// }
///
/// let mut cursor = trie.cursor();
///
/// // Seek to first key >= "c"
/// assert!(cursor.seek_lower_bound(b"c"));
/// assert_eq!(cursor.key(), b"cherry");
///
/// // Advance
/// assert!(cursor.next());
/// assert_eq!(cursor.key(), b"date");
///
/// // Go back
/// assert!(cursor.prev());
/// assert_eq!(cursor.key(), b"cherry");
/// ```
pub struct DoubleArrayTrieCursor<'a> {
    pub(crate) trie: &'a DoubleArrayTrie,
    /// Stack of (state, next_symbol_to_try) for DFS position tracking
    pub(crate) stack: Vec<(u32, u16)>,
    /// Current key bytes
    current_key: Vec<u8>,
    /// Whether the cursor is positioned on a valid key
    valid: bool,
}

impl<'a> DoubleArrayTrieCursor<'a> {
    /// Create a new cursor (not positioned).
    pub(crate) fn new(trie: &'a DoubleArrayTrie) -> Self {
        Self {
            trie,
            stack: Vec::with_capacity(64),
            current_key: Vec::with_capacity(64),
            valid: false,
        }
    }

    /// Current key bytes. Only valid when `is_valid()` is true.
    #[inline]
    pub fn key(&self) -> &[u8] {
        &self.current_key
    }

    /// Whether the cursor is positioned on a valid key.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Seek to the first key in the trie.
    pub fn seek_begin(&mut self) -> bool {
        self.stack.clear();
        self.current_key.clear();
        self.valid = false;

        // Start at root
        if self.trie.states.is_empty() {
            return false;
        }

        // If root is terminal (empty key), we're done
        if self.trie.ninfos[0].is_term() {
            self.stack.push((0, 0));
            self.valid = true;
            return true;
        }

        // Push root and descend to first terminal
        self.stack.push((0, 0));
        self.descend_to_next_terminal()
    }

    /// Seek to the last key in the trie.
    pub fn seek_end(&mut self) -> bool {
        self.stack.clear();
        self.current_key.clear();
        self.valid = false;

        if self.trie.states.is_empty() {
            return false;
        }

        // Start at root, descend to rightmost (highest symbol) terminal
        self.stack.push((0, 256));
        self.descend_to_rightmost_terminal(0)
    }

    /// Seek to the first key >= `target`.
    /// Matching C++ `seek_lower_bound` algorithm: binary search per level.
    pub fn seek_lower_bound(&mut self, target: &[u8]) -> bool {
        self.stack.clear();
        self.current_key.clear();
        self.valid = false;

        if self.trie.states.is_empty() {
            return false;
        }

        let mut curr = 0u32;

        for &ch in target {
            // Find child >= ch (matching C++ lower_bound_ex on children)
            match self.trie.lower_bound_child(curr, ch) {
                Some((found_ch, next_state)) if found_ch == ch => {
                    // Exact match — follow this transition
                    self.stack.push((curr, ch as u16 + 1));
                    self.current_key.push(ch);
                    curr = next_state;
                }
                Some((found_ch, _)) => {
                    // No exact match, but found higher child
                    // Position at that child and descend to first terminal
                    self.stack.push((curr, found_ch as u16));
                    return self.advance_from_stack();
                }
                None => {
                    // No child >= ch — backtrack to find next key
                    self.stack.push((curr, 256)); // Exhausted all children
                    return self.advance_from_stack();
                }
            }
        }

        // Key fully consumed. If current state is terminal, exact match.
        if self.trie.ninfos[curr as usize].is_term() {
            self.stack.push((curr, 0));
            self.valid = true;
            return true;
        }

        // Not terminal — descend to first terminal in subtree
        self.stack.push((curr, 0));
        self.descend_to_next_terminal()
    }

    /// Advance to the next key in lexicographic order.
    #[allow(clippy::should_implement_trait)] // inherent next() mutates cursor in-place and returns success bool; not std::iter::Iterator
    pub fn next(&mut self) -> bool {
        if !self.valid {
            return false;
        }

        // Pop current terminal state, advance from its children or backtrack
        if let Some(&(state, _)) = self.stack.last() {
            // Try to descend into children of current state
            if self.trie.ninfos[state as usize].first_child() != NINFO_NONE {
                let len = self.stack.len();
                self.stack[len - 1] = (state, 0);
                return self.descend_to_next_terminal();
            }
        }

        // No children — backtrack
        self.advance_from_stack()
    }

    /// Move to the previous key in lexicographic order.
    ///
    /// Uses parent-chain walk: O(key_depth) per call.
    /// After prev(), the stack is rebuilt for next() compatibility.
    pub fn prev(&mut self) -> bool {
        if !self.valid {
            return false;
        }
        self.valid = false;

        // Get current state from the actual trie (not stack — avoids stale state)
        let saved_key = self.current_key.clone();
        let mut state = match self.trie.lookup_state(&saved_key) {
            Some(s) => s,
            None => return false,
        };

        // Walk up parent chain looking for a lower path
        let mut depth = saved_key.len();

        loop {
            if state == 0 {
                // At root
                if depth == 0 {
                    // Empty key is current — nothing before it
                    self.current_key.clear();
                    self.stack.clear();
                    return false;
                }
                // Shouldn't reach here
                return false;
            }

            let parent = self.trie.states[state as usize].parent();
            let parent_base = self.trie.states[parent as usize].child0();
            let my_symbol = state ^ parent_base;
            depth -= 1;

            // Find highest sibling with symbol < my_symbol
            if let Some((ch, sibling)) = self.trie.prev_child(parent, my_symbol) {
                // Descend to rightmost terminal in sibling's subtree
                self.current_key.truncate(depth);
                self.current_key.push(ch);
                self.stack.clear();
                self.stack.push((sibling, 256));
                if self.descend_to_rightmost_terminal(sibling) {
                    self.rebuild_stack_from_key();
                    return true;
                }
                self.current_key.truncate(depth);
            }

            // No lower sibling — is parent terminal?
            if self.trie.ninfos[parent as usize].is_term() {
                self.current_key.truncate(depth);
                self.rebuild_stack_from_key();
                self.valid = true;
                return true;
            }

            state = parent;
        }
    }

    /// Rebuild stack from root following current_key (ensures next() works after prev()).
    fn rebuild_stack_from_key(&mut self) {
        let key = self.current_key.clone();
        self.stack.clear();

        let mut curr = 0u32;
        self.stack.push((0, 0));

        for &ch in key.iter() {
            let next = self.trie.state_move(curr, ch);
            if next == NIL_STATE {
                break;
            }
            let len = self.stack.len();
            self.stack[len - 1].1 = ch as u16 + 1;
            self.stack.push((next, 0));
            curr = next;
        }
    }

    /// Descend to the rightmost (lexicographically last) terminal in the subtree.
    /// Matching C++ decr's descent: always go to last child.
    fn descend_to_rightmost_terminal(&mut self, state: u32) -> bool {
        let mut curr = state;
        loop {
            match self.trie.last_child(curr) {
                Some((ch, next_state)) => {
                    let len = self.stack.len();
                    self.stack[len - 1].1 = ch as u16;
                    self.current_key.push(ch);
                    self.stack.push((next_state, 256));
                    curr = next_state;
                }
                None => {
                    // No children — is this state terminal?
                    if self.trie.ninfos[curr as usize].is_term() {
                        self.valid = true;
                        return true;
                    }
                    return false;
                }
            }
        }
    }

    // --- Internal helpers ---

    /// From current stack position, find the next child >= the symbol at stack top,
    /// then descend to the first terminal in that subtree.
    fn advance_from_stack(&mut self) -> bool {
        self.valid = false;

        while let Some(&mut (state, ref mut next_sym)) = self.stack.last_mut() {
            if *next_sym > 255 {
                self.stack.pop();
                self.current_key.pop();
                continue;
            }

            match self.trie.lower_bound_child(state, *next_sym as u8) {
                Some((ch, next_state)) => {
                    *next_sym = ch as u16 + 1;
                    self.current_key.push(ch);

                    if self.trie.ninfos[next_state as usize].is_term() {
                        self.stack.push((next_state, 0));
                        self.valid = true;
                        return true;
                    }

                    self.stack.push((next_state, 0));
                    // Continue descending in next iteration
                }
                None => {
                    self.stack.pop();
                    self.current_key.pop();
                }
            }
        }

        false
    }

    /// Descend from current stack top to the first (leftmost) terminal in the subtree.
    fn descend_to_next_terminal(&mut self) -> bool {
        loop {
            let &(state, _) = match self.stack.last() {
                Some(s) => s,
                None => return false,
            };

            match self.trie.first_child(state) {
                Some((ch, next_state)) => {
                    let len = self.stack.len();
                    self.stack[len - 1].1 = ch as u16 + 1;
                    self.current_key.push(ch);

                    if self.trie.ninfos[next_state as usize].is_term() {
                        self.stack.push((next_state, 0));
                        self.valid = true;
                        return true;
                    }

                    self.stack.push((next_state, 0));
                }
                None => {
                    return self.advance_from_stack();
                }
            }
        }
    }
}

impl DoubleArrayTrie {
    /// Create a bidirectional lexicographic cursor over this trie.
    #[inline]
    pub fn cursor(&self) -> DoubleArrayTrieCursor<'_> {
        DoubleArrayTrieCursor::new(self)
    }

    /// Iterate all keys in the lexicographic range `[from, to)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::fsa::double_array::DoubleArrayTrie;
    ///
    /// let mut trie = DoubleArrayTrie::new();
    /// for w in &["apple", "banana", "cherry", "date"] {
    ///     trie.insert(w.as_bytes()).unwrap();
    /// }
    ///
    /// let range: Vec<Vec<u8>> = trie.range(b"b", b"d").collect();
    /// assert_eq!(range.len(), 2); // banana, cherry
    /// ```
    pub fn range<'a>(&'a self, from: &[u8], to: &[u8]) -> RangeIter<'a> {
        let mut cursor = self.cursor();
        let valid = cursor.seek_lower_bound(from);
        RangeIter {
            cursor,
            upper_bound: to.to_vec(),
            started: valid,
        }
    }
}

/// Iterator over a lexicographic range `[from, to)` in a `DoubleArrayTrie`.
pub struct RangeIter<'a> {
    cursor: DoubleArrayTrieCursor<'a>,
    upper_bound: Vec<u8>,
    started: bool,
}

impl<'a> Iterator for RangeIter<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            return None;
        }

        if !self.cursor.is_valid() {
            return None;
        }

        let key = self.cursor.key();
        // Check upper bound (exclusive)
        if key >= self.upper_bound.as_slice() {
            return None;
        }

        let result = key.to_vec();
        self.started = self.cursor.next();
        Some(result)
    }
}
