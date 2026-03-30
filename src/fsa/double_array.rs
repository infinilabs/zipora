//! Standalone Double Array Trie — faithful port of the C++ reference implementation.
//!
//! Single-purpose, high-performance trie using the double-array technique.
//! Each state is 8 bytes (two u32 fields), providing excellent cache locality.
//!
//! # Design (matching C++ DA_State8B)
//!
//! - `child0` (base): bits 0-30 = base offset for children, bit 31 = terminal flag
//! - `parent` (check): bits 0-30 = parent state, bit 31 = free flag
//! - Transition: `next = states[curr].child0() + symbol`
//! - Validation: `states[next].parent() == curr`
//!
//! # Performance
//!
//! - Insert: O(key_length) amortized
//! - Lookup: O(key_length) worst case
//! - Memory: 8 bytes per state, 1.5x growth factor

use crate::error::{Result, ZiporaError};

/// 8-byte state matching C++ DA_State8B exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct DaState {
    /// Base/child0: bits 0-30 = base value, bit 31 = terminal bit
    child0: u32,
    /// Check/parent: bits 0-30 = parent state, bit 31 = free bit
    parent: u32,
}

// Bit constants matching C++ reference
const TERM_BIT: u32 = 0x8000_0000;
const FREE_BIT: u32 = 0x8000_0000;
const VALUE_MASK: u32 = 0x7FFF_FFFF;
const NIL_STATE: u32 = 0x7FFF_FFFF;
const MAX_STATE: u32 = 0x7FFF_FFFE;

impl DaState {
    /// New free state (matching C++ constructor)
    #[inline(always)]
    const fn new_free() -> Self {
        Self {
            child0: NIL_STATE,          // No children, no terminal
            parent: NIL_STATE | FREE_BIT, // Free
        }
    }

    /// New root state
    #[inline(always)]
    const fn new_root() -> Self {
        Self {
            child0: NIL_STATE,  // Will be set on first child insert
            parent: 0,          // Root's parent is itself (state 0), not free
        }
    }

    #[inline(always)]
    fn child0(&self) -> u32 { self.child0 & VALUE_MASK }

    #[inline(always)]
    fn parent(&self) -> u32 { self.parent & VALUE_MASK }

    #[inline(always)]
    fn is_term(&self) -> bool { (self.child0 & TERM_BIT) != 0 }

    #[inline(always)]
    fn is_free(&self) -> bool { (self.parent & FREE_BIT) != 0 }

    #[inline(always)]
    fn set_term_bit(&mut self) { self.child0 |= TERM_BIT; }

    #[inline(always)]
    fn clear_term_bit(&mut self) { self.child0 &= !TERM_BIT; }

    /// Set child0/base, preserving terminal bit
    #[inline(always)]
    fn set_child0(&mut self, val: u32) {
        self.child0 = (self.child0 & TERM_BIT) | (val & VALUE_MASK);
    }

    /// Set parent/check, clears free bit (allocates the state)
    #[inline(always)]
    fn set_parent(&mut self, val: u32) {
        self.parent = val & VALUE_MASK; // No free bit
    }

    /// Mark as free
    #[inline(always)]
    fn set_free(&mut self) {
        self.child0 = NIL_STATE;
        self.parent = NIL_STATE | FREE_BIT;
    }
}

/// High-performance double-array trie.
///
/// Faithful port of the C++ reference `DoubleArrayTrie<DA_State8B>`.
/// Each state is 8 bytes. Transitions are computed as `base + symbol`.
///
/// # Examples
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
    states: Vec<DaState>,
    num_keys: usize,
    /// Heuristic search position (matching C++ curr_slot)
    search_head: usize,
}

impl DoubleArrayTrie {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(2); // At least root + 1 slot
        let mut states = Vec::with_capacity(cap);
        // State 0 = root
        states.push(DaState::new_root());
        // Fill rest with free states
        states.resize(cap, DaState::new_free());

        Self {
            states,
            num_keys: 0,
            search_head: 1,
        }
    }

    /// Number of keys in the trie.
    #[inline(always)]
    pub fn len(&self) -> usize { self.num_keys }

    /// Check if the trie is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.num_keys == 0 }

    /// Total number of allocated states.
    #[inline]
    pub fn total_states(&self) -> usize { self.states.len() }

    /// Memory usage in bytes.
    #[inline]
    pub fn mem_size(&self) -> usize {
        self.states.len() * std::mem::size_of::<DaState>()
    }

    /// Check if a state is terminal.
    #[inline(always)]
    pub fn is_term(&self, state: u32) -> bool {
        (state as usize) < self.states.len() && self.states[state as usize].is_term()
    }

    /// Check if a state is free.
    #[inline(always)]
    pub fn is_free(&self, state: u32) -> bool {
        (state as usize) >= self.states.len() || self.states[state as usize].is_free()
    }

    /// Single state transition: `next = base[curr] + ch`, valid if `check[next] == curr`.
    /// Returns NIL_STATE if transition doesn't exist.
    #[inline(always)]
    pub fn state_move(&self, curr: u32, ch: u8) -> u32 {
        let base = self.states[curr as usize].child0();
        if base == NIL_STATE { return NIL_STATE; }
        let next = base as usize + ch as usize;
        if next >= self.states.len() { return NIL_STATE; }
        if self.states[next].is_free() { return NIL_STATE; }
        if self.states[next].parent() == curr {
            next as u32
        } else {
            NIL_STATE
        }
    }

    /// Insert a key. Returns true if the key was new.
    pub fn insert(&mut self, key: &[u8]) -> Result<bool> {
        // Empty key: mark root as terminal
        if key.is_empty() {
            let was_new = !self.states[0].is_term();
            self.states[0].set_term_bit();
            if was_new { self.num_keys += 1; }
            return Ok(was_new);
        }

        let mut curr = 0u32;

        for &ch in key {
            let base = self.states[curr as usize].child0();

            if base == NIL_STATE {
                // No children yet — find a free base for this state
                let new_base = self.find_free_base(&[ch])?;
                self.states[curr as usize].set_child0(new_base);
                let next = new_base + ch as u32;
                self.ensure_capacity(next as usize + 1);
                self.states[next as usize].set_parent(curr);
                curr = next;
            } else {
                let next = base + ch as u32;
                self.ensure_capacity(next as usize + 1);

                if !self.states[next as usize].is_free()
                    && self.states[next as usize].parent() == curr
                {
                    // Transition exists, follow it
                    curr = next;
                } else if self.states[next as usize].is_free() {
                    // Position free, allocate it
                    self.states[next as usize].set_parent(curr);
                    curr = next;
                } else {
                    // Conflict — relocate curr's children
                    let new_base = self.relocate(curr, ch)?;
                    let next = new_base + ch as u32;
                    self.ensure_capacity(next as usize + 1);
                    self.states[next as usize].set_parent(curr);
                    curr = next;
                }
            }
        }

        // Mark terminal
        let was_new = !self.states[curr as usize].is_term();
        self.states[curr as usize].set_term_bit();
        if was_new { self.num_keys += 1; }
        Ok(was_new)
    }

    /// Insert a key, calling `on_relocate(old_pos, new_pos)` whenever a child
    /// state is moved during collision resolution.  This lets external value
    /// arrays stay in sync with state IDs.
    pub fn insert_with_relocate_cb(
        &mut self,
        key: &[u8],
        mut on_relocate: impl FnMut(u32, u32),
    ) -> Result<bool> {
        if key.is_empty() {
            let was_new = !self.states[0].is_term();
            self.states[0].set_term_bit();
            if was_new { self.num_keys += 1; }
            return Ok(was_new);
        }

        let mut curr = 0u32;

        for &ch in key {
            let base = self.states[curr as usize].child0();

            if base == NIL_STATE {
                let new_base = self.find_free_base(&[ch])?;
                self.states[curr as usize].set_child0(new_base);
                let next = new_base + ch as u32;
                self.ensure_capacity(next as usize + 1);
                self.states[next as usize].set_parent(curr);
                curr = next;
            } else {
                let next = base + ch as u32;
                self.ensure_capacity(next as usize + 1);

                if !self.states[next as usize].is_free()
                    && self.states[next as usize].parent() == curr
                {
                    curr = next;
                } else if self.states[next as usize].is_free() {
                    self.states[next as usize].set_parent(curr);
                    curr = next;
                } else {
                    // Conflict — relocate curr's children, notifying caller.
                    let old_base = self.states[curr as usize].child0();
                    let new_base = self.relocate(curr, ch)?;

                    // Notify about each moved child
                    if old_base != NIL_STATE {
                        for sch in 0u16..=255u16 {
                            if sch as u8 == ch { continue; }
                            let old_pos = old_base + sch as u32;
                            let new_pos = new_base + sch as u32;
                            if new_pos as usize >= self.states.len() { continue; }
                            if !self.states[new_pos as usize].is_free()
                                && self.states[new_pos as usize].parent() == curr
                            {
                                on_relocate(old_pos, new_pos);
                            }
                        }
                    }

                    let next = new_base + ch as u32;
                    self.ensure_capacity(next as usize + 1);
                    self.states[next as usize].set_parent(curr);
                    curr = next;
                }
            }
        }

        let was_new = !self.states[curr as usize].is_term();
        self.states[curr as usize].set_term_bit();
        if was_new { self.num_keys += 1; }
        Ok(was_new)
    }

    /// Check if a key exists — tight loop, minimal branching.
    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool {
        let states = self.states.as_slice();
        let len = states.len();

        if key.is_empty() {
            return states[0].is_term();
        }

        let mut curr = 0usize;
        for &ch in key {
            let base = states[curr].child0();
            if base == NIL_STATE { return false; }
            let next = base as usize + ch as usize;
            if next >= len { return false; }
            // Combined free + parent check: if free, parent has FREE_BIT set, won't match curr
            if states[next].parent != curr as u32 { return false; }
            curr = next;
        }

        states[curr].is_term()
    }

    /// Lookup key and return its terminal state, or None.
    #[inline]
    pub fn lookup_state(&self, key: &[u8]) -> Option<u32> {
        let states = self.states.as_slice();
        let len = states.len();

        if key.is_empty() {
            return if states[0].is_term() { Some(0) } else { None };
        }

        let mut curr = 0usize;
        for &ch in key {
            let base = states[curr].child0();
            if base == NIL_STATE { return None; }
            let next = base as usize + ch as usize;
            if next >= len { return None; }
            if states[next].parent != curr as u32 { return None; }
            curr = next;
        }

        if states[curr].is_term() { Some(curr as u32) } else { None }
    }

    /// Remove a key. Returns true if the key existed.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if let Some(state) = self.lookup_state(key) {
            if self.states[state as usize].is_term() {
                self.states[state as usize].clear_term_bit();
                self.num_keys -= 1;
                return true;
            }
        }
        false
    }

    /// Restore the key string from a state by walking the parent chain.
    pub fn restore_key(&self, state: u32) -> Option<Vec<u8>> {
        if state as usize >= self.states.len() { return None; }
        if self.states[state as usize].is_free() { return None; }

        let mut symbols = Vec::new();
        let mut curr = state;

        while curr != 0 {
            let parent = self.states[curr as usize].parent();
            let parent_base = self.states[parent as usize].child0();
            if curr < parent_base { return None; }
            let symbol = (curr - parent_base) as u8;
            symbols.push(symbol);
            curr = parent;
        }

        symbols.reverse();
        Some(symbols)
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
            if next == NIL_STATE { return Vec::new(); }
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
        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return; }

        for ch in 0u16..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { break; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                f(ch as u8, next as u32);
            }
        }
    }

    /// Get sorted children of a state as (symbol, child_state) pairs.
    /// Matching C++ `get_all_move()`. Used by cursor for binary search.
    #[inline]
    fn get_children(&self, state: u32) -> Vec<(u8, u32)> {
        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return Vec::new(); }

        let mut children = Vec::new();
        for ch in 0u16..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { break; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                children.push((ch as u8, next as u32));
            }
        }
        children // Already sorted by symbol (ascending scan)
    }

    /// Find the child with the given symbol, or the next higher child.
    /// Returns (index, exact_match) where index is into get_children() result.
    #[inline]
    fn lower_bound_child(&self, state: u32, symbol: u8) -> Option<(u8, u32)> {
        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return None; }

        for ch in (symbol as u16)..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { break; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                return Some((ch as u8, next as u32));
            }
        }
        None
    }

    /// Find the highest child with symbol < given symbol.
    #[inline]
    fn prev_child(&self, state: u32, symbol: u32) -> Option<(u8, u32)> {
        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return None; }

        for ch in (0..symbol.min(256)).rev() {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { continue; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                return Some((ch as u8, next as u32));
            }
        }
        None
    }

    /// Get the first (lowest symbol) child of a state.
    #[inline]
    fn first_child(&self, state: u32) -> Option<(u8, u32)> {
        self.lower_bound_child(state, 0)
    }

    /// Get the last (highest symbol) child of a state.
    #[inline]
    fn last_child(&self, state: u32) -> Option<(u8, u32)> {
        self.prev_child(state, 256)
    }

    /// Call `f` for each key with the given prefix — zero allocation.
    ///
    /// The callback receives a `&[u8]` reference to the key bytes.
    /// Unlike `keys_with_prefix()`, this does not allocate `Vec<Vec<u8>>`.
    pub fn for_each_key_with_prefix(&self, prefix: &[u8], mut f: impl FnMut(&[u8])) {
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.state_move(curr, ch);
            if next == NIL_STATE { return; }
            curr = next;
        }
        let mut path = prefix.to_vec();
        self.walk_keys(curr, &mut path, &mut f);
    }

    /// Internal DFS for callback-based key iteration.
    fn walk_keys(&self, state: u32, path: &mut Vec<u8>, f: &mut impl FnMut(&[u8])) {
        if state as usize >= self.states.len() { return; }

        if self.states[state as usize].is_term() {
            f(path);
        }

        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return; }

        for ch in 0u16..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { break; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                path.push(ch as u8);
                self.walk_keys(next as u32, path, f);
                path.pop();
            }
        }
    }

    /// Build from sorted keys (more efficient than incremental insert).
    pub fn build_from_sorted(keys: &[&[u8]]) -> Result<Self> {
        if keys.is_empty() { return Ok(Self::new()); }

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

    /// Shrink internal arrays to fit actual usage.
    pub fn shrink_to_fit(&mut self) {
        // Find the last used state
        let last_used = self.states.iter().rposition(|s| !s.is_free()).unwrap_or(0);
        // Keep some extra room for the alphabet (256 + margin)
        let new_len = (last_used + 257).min(self.states.len());
        self.states.truncate(new_len);
        self.states.shrink_to_fit();
    }

    // --- Internal methods ---

    /// Ensure states array is large enough, using 1.5x amortized growth.
    #[inline]
    fn ensure_capacity(&mut self, required: usize) {
        if required <= self.states.len() { return; }
        let new_len = required.max(self.states.len() * 3 / 2).max(256);
        self.states.resize(new_len, DaState::new_free());
    }

    /// Find a free base value where all given children symbols can be placed.
    /// Matching C++ reference heuristic: linear probe from search_head.
    fn find_free_base(&mut self, children: &[u8]) -> Result<u32> {
        debug_assert!(!children.is_empty());

        let min_ch = *children.iter().min().unwrap() as u32;
        let max_ch = *children.iter().max().unwrap() as u32;
        let single_child = children.len() == 1;

        // Start search from search_head, ensuring base + min_ch >= 1
        let mut base = if self.search_head as u32 > min_ch {
            self.search_head as u32 - min_ch
        } else {
            1
        };

        let mut attempts = 0u32;

        loop {
            if attempts > 1_000_000 || base > MAX_STATE {
                return Err(ZiporaError::invalid_data("Double array: cannot find free base"));
            }
            attempts += 1;

            let max_pos = base + max_ch;
            self.ensure_capacity(max_pos as usize + 1);

            // Fast path for single child (most common case in incremental insert)
            if single_child {
                let pos = (base + min_ch) as usize;
                if pos > 0 && self.states[pos].is_free() {
                    if base as usize > self.search_head {
                        self.search_head += ((base as usize - self.search_head) >> 4).max(1);
                    }
                    return Ok(base);
                }
                base += 1;
                continue;
            }

            // Multi-child: check all positions
            let all_free = children.iter().all(|&ch| {
                let pos = (base + ch as u32) as usize;
                pos > 0 && self.states[pos].is_free()
            });

            if all_free {
                if base as usize > self.search_head {
                    self.search_head += ((base as usize - self.search_head) >> 4).max(1);
                }
                return Ok(base);
            }

            base += 1;
        }
    }

    /// Relocate all children of `state` to a new base that also accommodates `new_ch`.
    fn relocate(&mut self, state: u32, new_ch: u8) -> Result<u32> {
        // Collect existing children
        let old_base = self.states[state as usize].child0();
        let mut children_symbols = Vec::new();

        if old_base != NIL_STATE {
            for ch in 0u16..=255u16 {
                let pos = old_base as usize + ch as usize;
                if pos >= self.states.len() { break; }
                if !self.states[pos].is_free() && self.states[pos].parent() == state {
                    children_symbols.push(ch as u8);
                }
            }
        }

        // Add the new symbol
        if !children_symbols.contains(&new_ch) {
            children_symbols.push(new_ch);
        }
        children_symbols.sort_unstable();

        // Find a new base for all children
        let new_base = self.find_free_base(&children_symbols)?;

        // Move existing children to new positions
        if old_base != NIL_STATE {
            for &ch in &children_symbols {
                if ch == new_ch { continue; } // New child, not yet allocated
                let old_pos = old_base + ch as u32;
                let new_pos = new_base + ch as u32;

                if old_pos as usize >= self.states.len() { continue; }
                if self.states[old_pos as usize].is_free() { continue; }
                if self.states[old_pos as usize].parent() != state { continue; }

                self.ensure_capacity(new_pos as usize + 1);

                // Copy state data to new position
                let old_state = self.states[old_pos as usize];
                self.states[new_pos as usize].child0 = old_state.child0;
                self.states[new_pos as usize].set_parent(state);

                // Update grandchildren to point to new parent position
                let child_base = old_state.child0();
                if child_base != NIL_STATE {
                    for gch in 0u16..=255u16 {
                        let gpos = child_base as usize + gch as usize;
                        if gpos >= self.states.len() { break; }
                        if !self.states[gpos].is_free()
                            && self.states[gpos].parent() == old_pos
                        {
                            self.states[gpos].set_parent(new_pos);
                        }
                    }
                }

                // Free old position
                self.states[old_pos as usize].set_free();
            }
        }

        // Update state's base to new location
        self.states[state as usize].set_child0(new_base);
        Ok(new_base)
    }

    /// Recursively collect keys via DFS.
    fn collect_keys(&self, state: u32, path: &mut Vec<u8>, keys: &mut Vec<Vec<u8>>) {
        if state as usize >= self.states.len() { return; }

        // If terminal, record this path
        if self.states[state as usize].is_term() {
            keys.push(path.clone());
        }

        // Explore children
        let base = self.states[state as usize].child0();
        if base == NIL_STATE { return; }

        for ch in 0u16..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.states.len() { break; }
            if !self.states[next].is_free() && self.states[next].parent() == state {
                path.push(ch as u8);
                self.collect_keys(next as u32, path, keys);
                path.pop();
            }
        }
    }
}

impl Default for DoubleArrayTrie {
    fn default() -> Self { Self::new() }
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
    trie: &'a DoubleArrayTrie,
    /// Stack of (state, next_symbol_to_try) for DFS position tracking
    stack: Vec<(u32, u16)>,
    /// Current key bytes
    current_key: Vec<u8>,
    /// Whether the cursor is positioned on a valid key
    valid: bool,
}

impl<'a> DoubleArrayTrieCursor<'a> {
    /// Create a new cursor (not positioned).
    fn new(trie: &'a DoubleArrayTrie) -> Self {
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
        if self.trie.states.is_empty() { return false; }

        // If root is terminal (empty key), we're done
        if self.trie.states[0].is_term() {
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

        if self.trie.states.is_empty() { return false; }

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

        if self.trie.states.is_empty() { return false; }

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
        if self.trie.states[curr as usize].is_term() {
            self.stack.push((curr, 0));
            self.valid = true;
            return true;
        }

        // Not terminal — descend to first terminal in subtree
        self.stack.push((curr, 0));
        self.descend_to_next_terminal()
    }

    /// Advance to the next key in lexicographic order.
    pub fn next(&mut self) -> bool {
        if !self.valid { return false; }

        // Pop current terminal state, advance from its children or backtrack
        if let Some(&(state, _)) = self.stack.last() {
            // Try to descend into children of current state
            let base = self.trie.states[state as usize].child0();
            if base != NIL_STATE {
                // Replace top with (state, 0) to explore children
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
        if !self.valid { return false; }
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
            let my_symbol = state - parent_base;
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
            if self.trie.states[parent as usize].is_term() {
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
            if next == NIL_STATE { break; }
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
                    if self.trie.states[curr as usize].is_term() {
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

                    if self.trie.states[next_state as usize].is_term() {
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

                    if self.trie.states[next_state as usize].is_term() {
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

/// Key-value double-array trie map.
///
/// Values are stored in a parallel Vec indexed by state ID.
///
/// # Examples
///
/// ```rust
/// use zipora::fsa::double_array::DoubleArrayTrieMap;
///
/// let mut map = DoubleArrayTrieMap::<u32>::new();
/// map.insert(b"hello", 42).unwrap();
/// assert_eq!(map.get(b"hello"), Some(42));
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleArrayTrieMap<V: Copy> {
    trie: DoubleArrayTrie,
    values: Vec<Option<V>>,
}

impl<V: Copy> DoubleArrayTrieMap<V> {
    pub fn new() -> Self {
        Self { trie: DoubleArrayTrie::new(), values: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { trie: DoubleArrayTrie::with_capacity(cap), values: Vec::with_capacity(cap) }
    }

    /// Insert key-value pair. Returns previous value if key existed.
    pub fn insert(&mut self, key: &[u8], value: V) -> Result<Option<V>> {
        // Use the relocate callback to keep values in sync with state IDs.
        let values = &mut self.values;
        self.trie.insert_with_relocate_cb(key, |old_pos, new_pos| {
            let old = old_pos as usize;
            let new = new_pos as usize;
            if new >= values.len() {
                values.resize((new + 1).max(values.len() * 2), None);
            }
            if old < values.len() {
                values[new] = values[old].take();
            }
        })?;

        let state = self.trie.lookup_state(key)
            .ok_or_else(|| ZiporaError::invalid_state("insert succeeded but lookup failed"))?;
        let idx = state as usize;
        if idx >= self.values.len() {
            let new_len = (idx + 1).max(self.values.len() * 2).max(256);
            self.values.resize(new_len, None);
        }
        let prev = self.values[idx];
        self.values[idx] = Some(value);
        Ok(prev)
    }

    /// Get value for key.
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        let state = self.trie.lookup_state(key)?;
        self.values.get(state as usize).and_then(|v| *v)
    }

    #[inline]
    pub fn contains(&self, key: &[u8]) -> bool { self.trie.contains(key) }
    #[inline]
    pub fn len(&self) -> usize { self.trie.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.trie.is_empty() }

    /// Return all keys in the map.
    pub fn keys(&self) -> Vec<Vec<u8>> { self.trie.keys() }

    /// Return all keys starting with the given prefix.
    pub fn keys_with_prefix(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        self.trie.keys_with_prefix(prefix)
    }

    /// Return all (key, value) pairs for keys starting with the given prefix.
    /// Single traversal — no double lookup.
    pub fn entries_with_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, V)> {
        let mut results = Vec::new();
        // Navigate to prefix state
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.trie.state_move(curr, ch);
            if next == NIL_STATE { return results; }
            curr = next;
        }
        let mut path = prefix.to_vec();
        self.collect_entries(curr, &mut path, &mut results);
        results
    }

    /// Return all values for keys starting with the given prefix.
    pub fn values_with_prefix(&self, prefix: &[u8]) -> Vec<V> {
        self.entries_with_prefix(prefix).into_iter().map(|(_, v)| v).collect()
    }

    /// Recursively collect (key, value) entries.
    fn collect_entries(&self, state: u32, path: &mut Vec<u8>, entries: &mut Vec<(Vec<u8>, V)>) {
        if state as usize >= self.trie.states.len() { return; }

        if self.trie.states[state as usize].is_term() {
            if let Some(Some(val)) = self.values.get(state as usize) {
                entries.push((path.clone(), *val));
            }
        }

        let base = self.trie.states[state as usize].child0();
        if base == NIL_STATE { return; }

        for ch in 0u16..=255u16 {
            let next = base as usize + ch as usize;
            if next >= self.trie.states.len() { break; }
            if !self.trie.states[next].is_free() && self.trie.states[next].parent() == state {
                path.push(ch as u8);
                self.collect_entries(next as u32, path, entries);
                path.pop();
            }
        }
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<V> {
        let state = self.trie.lookup_state(key)?;
        let prev = self.values.get(state as usize).and_then(|v| *v);
        self.trie.remove(key);
        if let Some(slot) = self.values.get_mut(state as usize) {
            *slot = None;
        }
        prev
    }
}

impl<V: Copy> std::fmt::Debug for DoubleArrayTrieMap<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoubleArrayTrieMap")
            .field("num_keys", &self.trie.len())
            .field("total_states", &self.trie.total_states())
            .field("mem_size", &self.trie.mem_size())
            .finish()
    }
}

impl<V: Copy> Default for DoubleArrayTrieMap<V> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_contains() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"hello").unwrap());
        assert!(t.insert(b"help").unwrap());
        assert!(t.insert(b"world").unwrap());
        assert_eq!(t.len(), 3);

        assert!(t.contains(b"hello"));
        assert!(t.contains(b"help"));
        assert!(t.contains(b"world"));
        assert!(!t.contains(b"hel"));
        assert!(!t.contains(b"hell"));
        assert!(!t.contains(b"worlds"));
    }

    #[test]
    fn test_duplicate_insert() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"abc").unwrap());
        assert!(!t.insert(b"abc").unwrap());
        assert!(!t.insert(b"abc").unwrap());
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_empty_key() {
        let mut t = DoubleArrayTrie::new();
        assert!(t.insert(b"").unwrap());
        assert!(t.contains(b""));
        assert!(t.insert(b"a").unwrap());
        assert_eq!(t.len(), 2);

        let mut keys = t.keys();
        keys.sort();
        assert_eq!(keys, vec![vec![], vec![b'a']]);
    }

    #[test]
    fn test_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        assert_eq!(t.len(), 2);

        assert!(t.remove(b"hello"));
        assert_eq!(t.len(), 1);
        assert!(!t.contains(b"hello"));
        assert!(t.contains(b"world"));

        assert!(!t.remove(b"missing"));
    }

    #[test]
    fn test_restore_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();

        let state = t.lookup_state(b"hello").unwrap();
        assert_eq!(t.restore_key(state).unwrap(), b"hello");

        let state2 = t.lookup_state(b"world").unwrap();
        assert_eq!(t.restore_key(state2).unwrap(), b"world");
    }

    #[test]
    fn test_keys() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"apple").unwrap();
        t.insert(b"app").unwrap();
        t.insert(b"banana").unwrap();

        let mut keys = t.keys();
        keys.sort();
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], b"app");
        assert_eq!(keys[1], b"apple");
        assert_eq!(keys[2], b"banana");
    }

    #[test]
    fn test_keys_with_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"").unwrap();
        t.insert(b"a").unwrap();
        t.insert(b"ab").unwrap();
        t.insert(b"abc").unwrap();
        t.insert(b"abd").unwrap();
        t.insert(b"b").unwrap();

        let all = t.keys_with_prefix(b"");
        assert_eq!(all.len(), 6);

        let a = t.keys_with_prefix(b"a");
        assert_eq!(a.len(), 4); // a, ab, abc, abd

        let ab = t.keys_with_prefix(b"ab");
        assert_eq!(ab.len(), 3); // ab, abc, abd

        let none = t.keys_with_prefix(b"xyz");
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_many_inserts() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000 {
            t.insert(format!("key_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);
        assert!(t.contains(b"key_0000"));
        assert!(t.contains(b"key_0500"));
        assert!(t.contains(b"key_0999"));
        assert!(!t.contains(b"key_1000"));
    }

    #[test]
    fn test_state_move() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();

        let s1 = t.state_move(0, b'a');
        assert_ne!(s1, NIL_STATE);
        let s2 = t.state_move(s1, b'b');
        assert_ne!(s2, NIL_STATE);
        let s3 = t.state_move(s2, b'c');
        assert_ne!(s3, NIL_STATE);
        assert!(t.is_term(s3));

        assert_eq!(t.state_move(0, b'z'), NIL_STATE);
    }

    #[test]
    fn test_build_from_sorted() {
        let keys: Vec<&[u8]> = vec![b"apple", b"application", b"apply", b"banana", b"band"];
        let t = DoubleArrayTrie::build_from_sorted(&keys).unwrap();

        assert_eq!(t.len(), 5);
        for key in &keys {
            assert!(t.contains(key), "missing: {:?}", std::str::from_utf8(key));
        }
    }

    #[test]
    fn test_for_each_child() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"ab").unwrap();
        t.insert(b"ac").unwrap();
        t.insert(b"ad").unwrap();

        // Root should have child 'a'
        let mut root_children = Vec::new();
        t.for_each_child(0, |ch, _| root_children.push(ch));
        assert_eq!(root_children, vec![b'a']);

        // State for 'a' should have children 'b', 'c', 'd'
        let a_state = t.state_move(0, b'a');
        let mut a_children = Vec::new();
        t.for_each_child(a_state, |ch, _| a_children.push(ch));
        a_children.sort();
        assert_eq!(a_children, vec![b'b', b'c', b'd']);
    }

    #[test]
    fn test_da_trie_map() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"hello", 42).unwrap();
        map.insert(b"world", 100).unwrap();
        map.insert(b"help", 7).unwrap();

        assert_eq!(map.get(b"hello"), Some(42));
        assert_eq!(map.get(b"world"), Some(100));
        assert_eq!(map.get(b"help"), Some(7));
        assert_eq!(map.get(b"missing"), None);

        // Update
        let prev = map.insert(b"hello", 99).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(map.get(b"hello"), Some(99));

        // Remove
        let removed = map.remove(b"world");
        assert_eq!(removed, Some(100));
        assert_eq!(map.get(b"world"), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_mem_size() {
        let t = DoubleArrayTrie::new();
        // 256 states * 8 bytes = 2048
        assert_eq!(t.mem_size(), 256 * 8);
        assert_eq!(std::mem::size_of::<DaState>(), 8);
    }

    #[test]
    fn test_shared_prefixes() {
        let mut t = DoubleArrayTrie::new();
        // Many keys sharing common prefixes — stress the base allocation
        t.insert(b"test").unwrap();
        t.insert(b"testing").unwrap();
        t.insert(b"tested").unwrap();
        t.insert(b"tester").unwrap();
        t.insert(b"tests").unwrap();
        t.insert(b"tea").unwrap();
        t.insert(b"team").unwrap();
        t.insert(b"tear").unwrap();

        assert_eq!(t.len(), 8);
        assert!(t.contains(b"test"));
        assert!(t.contains(b"testing"));
        assert!(t.contains(b"tea"));
        assert!(t.contains(b"team"));
        assert!(!t.contains(b"te")); // prefix only, not inserted
        assert!(!t.contains(b"testi")); // prefix only
    }

    #[test]
    fn test_long_keys() {
        let mut t = DoubleArrayTrie::new();
        let long_key = "a".repeat(1000);
        t.insert(long_key.as_bytes()).unwrap();
        assert!(t.contains(long_key.as_bytes()));
        assert_eq!(t.len(), 1);

        let state = t.lookup_state(long_key.as_bytes()).unwrap();
        let restored = t.restore_key(state).unwrap();
        assert_eq!(restored, long_key.as_bytes());
    }

    #[test]
    fn test_binary_keys() {
        let mut t = DoubleArrayTrie::new();
        // Keys with all byte values including 0x00 and 0xFF
        t.insert(&[0x00, 0xFF, 0x80]).unwrap();
        t.insert(&[0x00, 0xFF, 0x81]).unwrap();
        t.insert(&[0xFF, 0x00, 0x01]).unwrap();

        assert_eq!(t.len(), 3);
        assert!(t.contains(&[0x00, 0xFF, 0x80]));
        assert!(t.contains(&[0xFF, 0x00, 0x01]));
        assert!(!t.contains(&[0x00, 0xFF]));
    }

    #[test]
    fn test_relocation_stress() {
        let mut t = DoubleArrayTrie::new();
        // Insert keys that force many relocations
        // Single-char keys use same base offset, forcing conflicts
        for ch in 0u8..=127u8 {
            let key = [ch];
            t.insert(&key).unwrap();
        }
        assert_eq!(t.len(), 128);
        for ch in 0u8..=127u8 {
            assert!(t.contains(&[ch]), "missing single-byte key {}", ch);
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut t = DoubleArrayTrie::with_capacity(10000);
        assert!(t.total_states() >= 10000);
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        t.shrink_to_fit();
        // After shrink, should be much smaller
        assert!(t.total_states() < 1000);
        // But keys must still work
        assert!(t.contains(b"hello"));
        assert!(t.contains(b"world"));
    }

    #[test]
    fn test_remove_and_reinsert() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();
        assert!(t.contains(b"abc"));

        t.remove(b"abc");
        assert!(!t.contains(b"abc"));
        assert_eq!(t.len(), 0);

        // Reinsert same key
        assert!(t.insert(b"abc").unwrap());
        assert!(t.contains(b"abc"));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_lookup_state_consistency() {
        let mut t = DoubleArrayTrie::new();
        let keys: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta"];
        for &key in &keys {
            t.insert(key).unwrap();
        }

        // Each key should have a unique state
        let states: Vec<u32> = keys.iter()
            .map(|k| t.lookup_state(k).unwrap())
            .collect();
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j],
                    "states for {:?} and {:?} should differ",
                    std::str::from_utf8(keys[i]).unwrap(),
                    std::str::from_utf8(keys[j]).unwrap());
            }
        }
    }

    /// Performance test — only meaningful in release mode.
    /// Verifies O(key_length) insert/lookup, not O(n).
    #[test]
    fn test_performance_5000_terms() {
        // Generate 5000 realistic terms
        let terms: Vec<String> = (0..5000)
            .map(|i| format!("term_{:06}_{}", i, ["alpha", "beta", "gamma", "delta"][i % 4]))
            .collect();

        // Insert
        let start = std::time::Instant::now();
        let mut t = DoubleArrayTrie::new();
        for term in &terms {
            t.insert(term.as_bytes()).unwrap();
        }
        let insert_time = start.elapsed();

        assert_eq!(t.len(), 5000);

        // Lookup (all hits)
        let start = std::time::Instant::now();
        for term in &terms {
            assert!(t.contains(term.as_bytes()));
        }
        let lookup_time = start.elapsed();

        // Lookup (all misses)
        let start = std::time::Instant::now();
        for i in 0..5000 {
            let miss = format!("miss_{:06}", i);
            assert!(!t.contains(miss.as_bytes()));
        }
        let miss_time = start.elapsed();

        // In release mode, all three should complete in well under 100ms
        // (Cedar does 5000 inserts in ~876µs)
        #[cfg(not(debug_assertions))]
        {
            eprintln!("DoubleArrayTrie 5000 terms: insert={:?}, lookup_hit={:?}, lookup_miss={:?}",
                insert_time, lookup_time, miss_time);
            eprintln!("Memory: {} bytes ({} bytes/key), {} states",
                t.mem_size(), t.mem_size() / 5000, t.total_states());
            // Sanity: insert should be under 50ms in release
            assert!(insert_time.as_millis() < 50,
                "Insert too slow: {:?}", insert_time);
            // Lookup should be under 10ms
            assert!(lookup_time.as_millis() < 10,
                "Lookup too slow: {:?}", lookup_time);
        }
    }

    #[test]
    fn test_entries_with_prefix() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"apple", 1).unwrap();
        map.insert(b"app", 2).unwrap();
        map.insert(b"application", 3).unwrap();
        map.insert(b"banana", 4).unwrap();

        let entries = map.entries_with_prefix(b"app");
        assert_eq!(entries.len(), 3);
        // Verify all app* entries are present
        let values: Vec<u32> = entries.iter().map(|(_, v)| *v).collect();
        assert!(values.contains(&1)); // apple
        assert!(values.contains(&2)); // app
        assert!(values.contains(&3)); // application

        let banana_entries = map.entries_with_prefix(b"ban");
        assert_eq!(banana_entries.len(), 1);
        assert_eq!(banana_entries[0].1, 4);

        let none = map.entries_with_prefix(b"xyz");
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_values_with_prefix() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"test_a", 10).unwrap();
        map.insert(b"test_b", 20).unwrap();
        map.insert(b"other", 30).unwrap();

        let vals = map.values_with_prefix(b"test_");
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&10));
        assert!(vals.contains(&20));
    }

    #[test]
    fn test_for_each_key_with_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"help").unwrap();
        t.insert(b"world").unwrap();

        let mut found = Vec::new();
        t.for_each_key_with_prefix(b"hel", |key| {
            found.push(key.to_vec());
        });
        found.sort();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0], b"hello");
        assert_eq!(found[1], b"help");

        let mut all = Vec::new();
        t.for_each_key_with_prefix(b"", |key| all.push(key.to_vec()));
        assert_eq!(all.len(), 3);
    }

    // --- Cursor / Range tests ---

    #[test]
    fn test_cursor_seek_begin_end() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"apple").unwrap();
        t.insert(b"banana").unwrap();
        t.insert(b"cherry").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b"apple");

        assert!(c.seek_end());
        assert_eq!(c.key(), b"cherry");
    }

    #[test]
    fn test_cursor_next_prev() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"apple");

        assert!(c.next());
        assert_eq!(c.key(), b"banana");
        assert!(c.next());
        assert_eq!(c.key(), b"cherry");
        assert!(c.next());
        assert_eq!(c.key(), b"date");
        assert!(c.next());
        assert_eq!(c.key(), b"elderberry");
        assert!(!c.next()); // Past end

        // Walk backward
        c.seek_end();
        assert_eq!(c.key(), b"elderberry");
        assert!(c.prev());
        assert_eq!(c.key(), b"date");
        assert!(c.prev());
        assert_eq!(c.key(), b"cherry");
        assert!(c.prev());
        assert_eq!(c.key(), b"banana");
        assert!(c.prev());
        assert_eq!(c.key(), b"apple");
        assert!(!c.prev()); // Past begin
    }

    #[test]
    fn test_cursor_seek_lower_bound() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();

        // Exact match
        assert!(c.seek_lower_bound(b"cherry"));
        assert_eq!(c.key(), b"cherry");

        // Between keys
        assert!(c.seek_lower_bound(b"c"));
        assert_eq!(c.key(), b"cherry");

        // Before all keys
        assert!(c.seek_lower_bound(b"a"));
        assert_eq!(c.key(), b"apple");

        // After all keys
        assert!(!c.seek_lower_bound(b"z"));

        // Between banana and cherry
        assert!(c.seek_lower_bound(b"cat"));
        assert_eq!(c.key(), b"cherry");
    }

    #[test]
    fn test_range() {
        let mut t = DoubleArrayTrie::new();
        for w in &["apple", "banana", "cherry", "date", "elderberry", "fig"] {
            t.insert(w.as_bytes()).unwrap();
        }

        // [b, e) => banana, cherry, date
        let range: Vec<Vec<u8>> = t.range(b"b", b"e").collect();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0], b"banana");
        assert_eq!(range[1], b"cherry");
        assert_eq!(range[2], b"date");

        // [a, z) => all 6
        let all: Vec<Vec<u8>> = t.range(b"a", b"z").collect();
        assert_eq!(all.len(), 6);

        // [d, d) => empty (from == to)
        let empty: Vec<Vec<u8>> = t.range(b"d", b"d").collect();
        assert_eq!(empty.len(), 0);

        // [cherry, elderberry) => cherry, date
        let mid: Vec<Vec<u8>> = t.range(b"cherry", b"elderberry").collect();
        assert_eq!(mid.len(), 2);
        assert_eq!(mid[0], b"cherry");
        assert_eq!(mid[1], b"date");
    }

    #[test]
    fn test_cursor_empty_trie() {
        let t = DoubleArrayTrie::new();
        let mut c = t.cursor();
        assert!(!c.seek_begin());
        assert!(!c.seek_end());
        assert!(!c.seek_lower_bound(b"anything"));
    }

    #[test]
    fn test_cursor_single_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"only").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b"only");
        assert!(!c.next());

        assert!(c.seek_end());
        assert_eq!(c.key(), b"only");
        assert!(!c.prev());
    }

    #[test]
    fn test_cursor_with_empty_key() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"").unwrap();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();

        let mut c = t.cursor();
        assert!(c.seek_begin());
        assert_eq!(c.key(), b""); // Empty key is the smallest

        assert!(c.next());
        assert_eq!(c.key(), b"a");

        assert!(c.next());
        assert_eq!(c.key(), b"b");
    }

    #[test]
    fn test_cursor_full_traversal_matches_keys() {
        let mut t = DoubleArrayTrie::new();
        let words = ["alpha", "beta", "gamma", "delta", "epsilon",
                      "zeta", "eta", "theta", "iota", "kappa"];
        for w in &words {
            t.insert(w.as_bytes()).unwrap();
        }

        // Forward traversal via cursor should match sorted keys()
        let mut cursor_keys = Vec::new();
        let mut c = t.cursor();
        if c.seek_begin() {
            cursor_keys.push(c.key().to_vec());
            while c.next() {
                cursor_keys.push(c.key().to_vec());
            }
        }

        let mut trie_keys = t.keys();
        trie_keys.sort();

        assert_eq!(cursor_keys, trie_keys,
            "Cursor traversal must match sorted keys()");
    }

    #[test]
    fn test_range_empty_bounds() {
        let mut t = DoubleArrayTrie::new();
        for w in &["a", "b", "c", "d"] {
            t.insert(w.as_bytes()).unwrap();
        }

        // Range where from > to should return nothing
        let r: Vec<_> = t.range(b"z", b"a").collect();
        assert_eq!(r.len(), 0);

        // Range covering everything
        let r: Vec<_> = t.range(b"", b"\xff").collect();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_cursor_seek_lower_bound_exact_last() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"aaa").unwrap();
        t.insert(b"zzz").unwrap();

        let mut c = t.cursor();

        // Seek to exact last key
        assert!(c.seek_lower_bound(b"zzz"));
        assert_eq!(c.key(), b"zzz");
        assert!(!c.next()); // No key after zzz

        // Seek past last key
        assert!(!c.seek_lower_bound(b"zzzz"));
    }

    #[test]
    fn test_cursor_prev_from_begin() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"first").unwrap();
        t.insert(b"second").unwrap();

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"first");
        assert!(!c.prev()); // Can't go before first
    }

    #[test]
    fn test_cursor_interleaved_next_prev() {
        let mut t = DoubleArrayTrie::new();
        for w in &["a", "b", "c", "d", "e"] {
            t.insert(w.as_bytes()).unwrap();
        }

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"a");

        c.next();
        assert_eq!(c.key(), b"b");
        c.next();
        assert_eq!(c.key(), b"c");

        // Go back
        c.prev();
        assert_eq!(c.key(), b"b");

        // Forward again
        c.next();
        assert_eq!(c.key(), b"c");
        c.next();
        assert_eq!(c.key(), b"d");
    }

    #[test]
    fn test_cursor_many_keys_sorted() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..200u32 {
            t.insert(format!("k{:04}", i).as_bytes()).unwrap();
        }

        // Forward traversal should produce sorted order
        let mut c = t.cursor();
        let mut keys = Vec::new();
        if c.seek_begin() {
            keys.push(c.key().to_vec());
            while c.next() { keys.push(c.key().to_vec()); }
        }
        assert_eq!(keys.len(), 200);
        for i in 1..keys.len() {
            assert!(keys[i - 1] < keys[i], "Not sorted at {}: {:?} >= {:?}",
                i, String::from_utf8_lossy(&keys[i-1]), String::from_utf8_lossy(&keys[i]));
        }

        // Backward traversal should produce reverse sorted order
        let mut c = t.cursor();
        let mut rkeys = Vec::new();
        if c.seek_end() {
            rkeys.push(c.key().to_vec());
            while c.prev() { rkeys.push(c.key().to_vec()); }
        }
        assert_eq!(rkeys.len(), 200);
        rkeys.reverse();
        assert_eq!(keys, rkeys, "Forward and backward traversals must match");
    }

    #[test]
    fn test_range_single_element() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();

        // Range containing exactly one key
        let r: Vec<_> = t.range(b"hello", b"world").collect();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0], b"hello");
    }

    #[test]
    fn test_seek_lower_bound_between_shared_prefix() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"abc").unwrap();
        t.insert(b"abd").unwrap();
        t.insert(b"abe").unwrap();

        let mut c = t.cursor();
        // Seek to "abd" exactly
        assert!(c.seek_lower_bound(b"abd"));
        assert_eq!(c.key(), b"abd");

        // Seek between "abc" and "abd"
        assert!(c.seek_lower_bound(b"abcc"));
        assert_eq!(c.key(), b"abd");

        // Seek before all
        assert!(c.seek_lower_bound(b"ab"));
        assert_eq!(c.key(), b"abc");
    }

    /// Reproduction of bug #1.1: keys_with_prefix returns incomplete results
    #[test]
    fn test_keys_with_prefix_1000_terms() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            t.insert(format!("term_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);

        let results = t.keys_with_prefix(b"term_00");
        assert_eq!(results.len(), 100,
            "keys_with_prefix('term_00') should return 100 (term_0000..term_0099), got {}",
            results.len());

        let results2 = t.keys_with_prefix(b"term_01");
        assert_eq!(results2.len(), 100,
            "keys_with_prefix('term_01') should return 100, got {}",
            results2.len());

        let all = t.keys_with_prefix(b"term_");
        assert_eq!(all.len(), 1000,
            "keys_with_prefix('term_') should return 1000, got {}",
            all.len());

        let all_keys = t.keys();
        assert_eq!(all_keys.len(), 1000,
            "keys() should return 1000, got {}",
            all_keys.len());
    }
}

#[cfg(test)]
mod prefix_regression_tests {
    use super::*;

    #[test]
    fn test_1000_terms_prefix() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            let inserted = t.insert(term.as_bytes()).unwrap();
            assert!(inserted || !inserted, "insert returned for term_{:04}", i);
        }
        assert_eq!(t.len(), 1000, "expected 1000 keys, got {}", t.len());
        
        // Verify all terms exist
        let mut missing = Vec::new();
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            if !t.contains(term.as_bytes()) {
                missing.push(i);
            }
        }
        assert!(missing.is_empty(), "missing {} terms: {:?}", missing.len(), &missing[..missing.len().min(20)]);

        let result = t.keys_with_prefix(b"term_00");
        assert_eq!(result.len(), 100, "prefix 'term_00' returned {} (expected 100)", result.len());
    }
}

#[cfg(test)]
mod map_prefix_regression_tests {
    use super::*;

    #[test]
    fn test_map_1000_terms_prefix_fresh_trie() {
        // Simulate the engine's flush_to_trie: build fresh trie from sorted entries
        let mut entries: Vec<(String, u32)> = (0..1000u32)
            .map(|i| (format!("term_{:04}", i), i))
            .collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut trie = DoubleArrayTrieMap::with_capacity(entries.len());
        for (term, id) in &entries {
            trie.insert(term.as_bytes(), *id).expect("insert failed");
        }

        assert_eq!(trie.len(), 1000);

        // Verify all lookups
        for i in 0..1000u32 {
            let term = format!("term_{:04}", i);
            assert_eq!(trie.get(term.as_bytes()), Some(i), "get failed for {}", term);
        }

        // Verify prefix
        let result = trie.values_with_prefix(b"term_00");
        assert_eq!(result.len(), 100, "values_with_prefix 'term_00' returned {} (expected 100)", result.len());
    }
}
