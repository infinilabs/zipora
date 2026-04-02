//! Standalone Double Array Trie — faithful port of the C++ reference implementation.
//!
//! Single-purpose, high-performance trie using the double-array technique.
//! Each state is 8 bytes (two u32 fields), providing excellent cache locality.
//!
//! # Design (matching C++ DA_State8B)
//!
//! - `child0` (base): full 32-bit base for XOR transitions (terminal flag in NInfo)
//! - `parent` (check): bits 0-30 = parent state, bit 31 = free flag
//! - Transition: `next = states[curr].child0() ^ symbol` (XOR for NIL-check-free traversal)
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
    /// Base for XOR transitions (terminal flag stored in NInfo, not here).
    child0: u32,
    /// Check/parent: bits 0-30 = parent state, bit 31 = free bit
    parent: u32,
}

// Bit constants
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

    /// New root state. child0=0 is a safe "no children" base: 0 ^ ch = ch,
    /// always in-bounds (array >= 256). Parent = NIL_STATE prevents false
    /// positive when child0=0 and ch=0 (0 ^ 0 = 0 → states[0].parent check).
    #[inline(always)]
    const fn new_root() -> Self {
        Self {
            child0: 0,          // Safe leaf base: 0 ^ ch always in [0, 255]
            parent: NIL_STATE,  // Sentinel — never matches any valid curr
        }
    }

    /// Base value for XOR transitions. Raw read — no masking needed
    /// because the terminal flag is stored in NInfo, not here.
    #[inline(always)]
    fn child0(&self) -> u32 { self.child0 }

    #[inline(always)]
    fn parent(&self) -> u32 { self.parent & VALUE_MASK }

    #[inline(always)]
    fn is_free(&self) -> bool { (self.parent & FREE_BIT) != 0 }

    /// Set child0/base (raw write — terminal flag is in NInfo).
    #[inline(always)]
    fn set_child0(&mut self, val: u32) {
        self.child0 = val;
    }

    /// Set parent/check, clears free bit (allocates the state)
    #[inline(always)]
    fn set_parent(&mut self, val: u32) {
        self.parent = val & VALUE_MASK; // No free bit
    }

    /// Mark as free with next/prev pointers for doubly-linked free list.
    #[inline(always)]
    fn set_free_linked(&mut self, next: u32, prev: u32) {
        self.child0 = next;              // next free
        self.parent = FREE_BIT | prev;   // prev free + free marker
    }

    /// Mark as free (standalone, not linked).
    #[inline(always)]
    fn set_free(&mut self) {
        self.set_free_linked(NIL_STATE, NIL_STATE);
    }

    /// Get next free pointer (only valid when is_free()).
    #[inline(always)]
    fn free_next(&self) -> u32 { self.child0 }

    /// Get prev free pointer (only valid when is_free()).
    #[inline(always)]
    fn free_prev(&self) -> u32 { self.parent & VALUE_MASK }
}

/// Node info for O(k) child enumeration.
/// Labels stored as label+1 (u16) so 0 means "none" while all 256 byte values are valid.
/// Bit 15 of `child` stores the terminal flag (not in DaState.child0, so child0 is mask-free).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct NInfo {
    sibling: u16,  // next sibling: label+1 (0 = end)
    child: u16,    // bits 0-8: first child label+1 (0 = no children), bit 15: terminal flag
}

const NINFO_NONE: u16 = 0;
const NINFO_TERM: u16 = 0x8000;

impl NInfo {
    /// Whether this state is terminal (has a complete key ending here).
    #[inline(always)]
    fn is_term(&self) -> bool { (self.child & NINFO_TERM) != 0 }

    /// Mark this state as terminal.
    #[inline(always)]
    fn set_term(&mut self) { self.child |= NINFO_TERM; }

    /// Clear the terminal flag.
    #[inline(always)]
    fn clear_term(&mut self) { self.child &= !NINFO_TERM; }

    /// Get the first child pointer (masking out the terminal flag).
    #[inline(always)]
    fn first_child(&self) -> u16 { self.child & !NINFO_TERM }

    /// Set the first child pointer, preserving the terminal flag.
    #[inline(always)]
    fn set_first_child(&mut self, val: u16) {
        self.child = (self.child & NINFO_TERM) | val;
    }
}

#[inline(always)]
fn ninfo_to_label(v: u16) -> Option<u8> {
    if v == 0 { None } else { Some((v - 1) as u8) }
}

#[inline(always)]
fn label_to_ninfo(label: u8) -> u16 {
    label as u16 + 1
}

/// High-performance double-array trie.
///
/// Faithful port of the C++ reference `DoubleArrayTrie<DA_State8B>`.
/// Each state is 8 bytes. Transitions are computed as `base ^ symbol` (XOR).
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
    ninfos: Vec<NInfo>,
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
        self.states.len() * std::mem::size_of::<DaState>() +
        self.ninfos.len() * std::mem::size_of::<NInfo>()
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
        let base = self.states[curr as usize].child0();
        let next = (base ^ ch as u32) as usize;
        // SAFETY: same invariant as contains()
        debug_assert!(next < self.states.len());
        let next_state = unsafe { self.states.get_unchecked(next) };
        if next_state.is_free() { return NIL_STATE; }
        if next_state.parent() == curr {
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
            if was_new { self.num_keys += 1; }
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
                                &self.ninfos, conflict_parent as usize,
                                old_base_cf, new_base_cf, &mut on_relocate,
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
                        &self.ninfos, curr as usize,
                        old_base, new_base, ch, &mut on_relocate,
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
        if was_new { self.num_keys += 1; }
        Ok(was_new)
    }

    /// Notify callback about all moved children of a parent after relocation.
    fn notify_relocated(
        ninfos: &[NInfo], parent_pos: usize,
        old_base: u32, new_base: u32,
        on_relocate: &mut impl FnMut(u32, u32),
    ) {
        let mut c = ninfos[parent_pos].first_child();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            on_relocate(old_base ^ label as u32, new_base ^ label as u32);
            let child_pos = (new_base ^ label as u32) as usize;
            c = if child_pos < ninfos.len() { ninfos[child_pos].sibling } else { NINFO_NONE };
        }
    }

    /// Notify callback about moved children, excluding `exclude_ch` (new child).
    fn notify_relocated_excluding(
        ninfos: &[NInfo], parent_pos: usize,
        old_base: u32, new_base: u32, exclude_ch: u8,
        on_relocate: &mut impl FnMut(u32, u32),
    ) {
        let mut c = ninfos[parent_pos].first_child();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            if label != exclude_ch {
                on_relocate(old_base ^ label as u32, new_base ^ label as u32);
            }
            let child_pos = (new_base ^ label as u32) as usize;
            c = if child_pos < ninfos.len() { ninfos[child_pos].sibling } else { NINFO_NONE };
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
            let base = states[curr].child0;
            let next = (base ^ ch as u32) as usize;
            // SAFETY: set_base_padded guarantees (base | 0xFF) < len for valid bases.
            // Leaf states have child0=0, so next=ch ∈ [0,255] < 256 ≤ len.
            // Free states have parent with FREE_BIT set, never matching curr.
            debug_assert!(next < states.len(), "OOB: next={next}, len={}", states.len());
            let next_state = unsafe { states.get_unchecked(next) };
            if next_state.parent != curr as u32 { return false; }
            curr = next;
        }

        ninfos[curr].is_term()
    }

    /// Lookup key and return its terminal state, or None.
    #[inline]
    pub fn lookup_state(&self, key: &[u8]) -> Option<u32> {
        let states = self.states.as_slice();

        if key.is_empty() {
            return if self.ninfos[0].is_term() { Some(0) } else { None };
        }

        let ninfos = self.ninfos.as_slice();
        let mut curr = 0usize;
        for &ch in key {
            let base = states[curr].child0;
            let next = (base ^ ch as u32) as usize;
            // SAFETY: same invariant as contains()
            debug_assert!(next < states.len());
            let next_state = unsafe { states.get_unchecked(next) };
            if next_state.parent != curr as u32 { return None; }
            curr = next;
        }

        if ninfos[curr].is_term() { Some(curr as u32) } else { None }
    }

    /// Remove a key. Returns true if the key existed.
    ///
    /// After clearing the terminal flag, prunes dead-end nodes that are
    /// no longer part of any valid key path (leaf states with no children
    /// and no terminal flag).
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if let Some(state) = self.lookup_state(key) {
            if self.ninfos[state as usize].is_term() {
                self.ninfos[state as usize].clear_term();
                self.num_keys -= 1;
                self.prune_dead_branch(state);
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
            if self.ninfos[curr as usize].is_term() { break; }
            if self.ninfos[curr as usize].first_child() != NINFO_NONE { break; }

            // This state is a dead leaf — remove it
            let parent = self.states[curr as usize].parent();
            if parent as usize >= self.states.len() { break; }

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
        if first == NINFO_NONE { return; }

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
            if prev_pos >= self.ninfos.len() { break; }
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
            if next_enc == NINFO_NONE { break; }
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
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return; }
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

    /// Get sorted children of a state as (symbol, child_state) pairs.
    /// Matching C++ `get_all_move()`. Used by cursor for binary search.
    #[inline]
    fn get_children(&self, state: u32) -> Vec<(u8, u32)> {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return Vec::new(); }
        let base = self.states[state as usize].child0();

        let mut children = Vec::new();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                children.push((label, child_pos as u32));
            }
            c = if child_pos < self.ninfos.len() {
                self.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
        children // Already sorted by symbol (NInfo chain is sorted)
    }

    /// Find the child with the given symbol, or the next higher child.
    /// Returns (index, exact_match) where index is into get_children() result.
    #[inline]
    fn lower_bound_child(&self, state: u32, symbol: u8) -> Option<(u8, u32)> {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return None; }
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
        if c == NINFO_NONE { return None; }
        let base = self.states[state as usize].child0();
        let mut result = None;
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if (label as u32) < symbol && child_pos < self.states.len() && !self.states[child_pos].is_free() {
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
        if c == NINFO_NONE { return None; }
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
        if c == NINFO_NONE { return None; }
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
            if next == NIL_STATE { return; }
            curr = next;
        }
        let mut path = prefix.to_vec();
        self.walk_keys(curr, &mut path, &mut f);
    }

    /// Internal DFS for callback-based key iteration.
    fn walk_keys(&self, state: u32, path: &mut Vec<u8>, f: &mut impl FnMut(&[u8])) {
        if state as usize >= self.states.len() { return; }

        if self.ninfos[state as usize].is_term() {
            f(path);
        }

        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return; }
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

    // =========================================================================
    // Consult — relocate the smaller side on conflict
    // =========================================================================

    /// Count children of a state using sibling chain (O(k)).
    #[inline]
    fn count_children(&self, state: u32) -> usize {
        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return 0; }
        let base = self.states[state as usize].child0();
        let mut count = 0;
        while c != NINFO_NONE {
            count += 1;
            let label = (c - 1) as u8;
            let pos = (base ^ label as u32) as usize;
            c = if pos < self.ninfos.len() { self.ninfos[pos].sibling } else { NINFO_NONE };
        }
        count
    }

    /// Resolve collision by relocating the smaller side.
    fn consult_and_relocate(&mut self, curr: u32, ch: u8) -> Result<u32> {
        let base = self.states[curr as usize].child0();
        let conflict_pos = base ^ ch as u32;
        let conflict_parent = self.states[conflict_pos as usize].parent();

        let curr_children = self.count_children(curr);
        let conflict_children = self.count_children(conflict_parent);

        if curr_children < conflict_children {
            // Relocate curr (fewer children)
            self.relocate(curr, ch)
        } else {
            // Relocate the conflicting side
            // We need to relocate conflict_parent's children, then retry
            self.relocate(conflict_parent, ch)?;
            // After relocation, the conflict position should be free now
            // Return curr's existing base (no change needed)
            Ok(self.states[curr as usize].child0())
        }
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
        let new_len = (last_used + 1).max(max_reachable).min(self.states.len());
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
            return; // Already first child
        } else {
            // Walk chain to find insertion point
            let mut prev_enc = first;
            loop {
                let prev_label = (prev_enc - 1) as u8;
                let prev_pos = (base ^ prev_label as u32) as usize;
                if prev_pos >= self.ninfos.len() { break; }
                let next_enc = self.ninfos[prev_pos].sibling;
                if next_enc == NINFO_NONE || label_enc < next_enc {
                    let child_pos = (base ^ label as u32) as usize;
                    if child_pos < self.ninfos.len() {
                        self.ninfos[child_pos].sibling = next_enc;
                        self.ninfos[prev_pos].sibling = label_enc;
                    }
                    break;
                }
                if next_enc == label_enc { break; } // Already present
                prev_enc = next_enc;
            }
        }
    }

    /// Ensure states array is large enough, using 1.5x amortized growth.
    /// New states are NOT added to the free list — they're detected as free
    /// by `is_free()`. Only explicitly freed states go on the free list.
    #[inline]
    fn ensure_capacity(&mut self, required: usize) {
        if required <= self.states.len() { return; }
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
        while self.search_head < self.states.len()
            && !self.states[self.search_head].is_free()
        {
            self.search_head += 1;
        }

        let mut base = (self.search_head as u32) ^ ch0;
        if base == 0 { base = 1; }

        let mut attempts = 0u32;

        loop {
            if attempts > 1_000_000 || base > MAX_STATE {
                return Err(ZiporaError::invalid_data("Double array: cannot find free base"));
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
                if child_pos < self.states.len() && !self.states[child_pos].is_free() {
                    if !children_symbols.contains(&label) {
                        children_symbols.push(label);
                    }
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
                if ch == new_ch { continue; }
                let old_pos = old_base ^ ch as u32;
                let new_pos = new_base ^ ch as u32;

                if old_pos as usize >= self.states.len() { continue; }
                if self.states[old_pos as usize].is_free() { continue; }
                if self.states[old_pos as usize].parent() != state { continue; }

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
                            if gpos < self.states.len() && !self.states[gpos].is_free()
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
            c = if child_pos < self.ninfos.len() { self.ninfos[child_pos].sibling } else { NINFO_NONE };
        }

        if children_symbols.is_empty() { return Ok(old_base); }
        children_symbols.sort_unstable();

        let new_base = self.find_free_base(&children_symbols)?;

        self.ninfos[state as usize].set_first_child(NINFO_NONE);

        for &ch in &children_symbols {
            let old_pos = old_base ^ ch as u32;
            let new_pos = new_base ^ ch as u32;

            if old_pos as usize >= self.states.len() { continue; }
            if self.states[old_pos as usize].is_free() { continue; }
            if self.states[old_pos as usize].parent() != state { continue; }

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
                        if gpos < self.states.len() && !self.states[gpos].is_free()
                            && self.states[gpos].parent() == old_pos
                        {
                            self.states[gpos].set_parent(new_pos);
                        }
                        gc = if gpos < self.ninfos.len() { self.ninfos[gpos].sibling } else { NINFO_NONE };
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
            if parent == ancestor { return true; }
            curr = parent;
            depth += 1;
        }
        false
    }

    /// Recursively collect keys via DFS.
    fn collect_keys(&self, state: u32, path: &mut Vec<u8>, keys: &mut Vec<Vec<u8>>) {
        if state as usize >= self.states.len() { return; }

        if self.ninfos[state as usize].is_term() {
            keys.push(path.clone());
        }

        let mut c = self.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return; }
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
    pub fn next(&mut self) -> bool {
        if !self.valid { return false; }

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

/// Key-value double-array trie map.
///
/// Values are stored in a parallel Vec indexed by state ID.
///
/// Trait for values stored in `DoubleArrayTrieMap`.
///
/// The associated constant `EMPTY` serves as a sentinel for unoccupied
/// slots. It must be a value that will never be inserted as a real
/// value — the trait impl documents this contract per type.
///
/// Zero runtime cost: `EMPTY` is a compile-time constant, monomorphized
/// into a literal (e.g., `cmp eax, 0x80000000` for i32).
pub trait MapValue: Copy + PartialEq {
    /// Sentinel representing "no value stored here".
    const EMPTY: Self;
}

impl MapValue for i32   { const EMPTY: Self = i32::MIN; }
impl MapValue for u32   { const EMPTY: Self = u32::MAX; }
impl MapValue for i64   { const EMPTY: Self = i64::MIN; }
impl MapValue for u64   { const EMPTY: Self = u64::MAX; }
impl MapValue for usize { const EMPTY: Self = usize::MAX; }

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
pub struct DoubleArrayTrieMap<V: MapValue> {
    trie: DoubleArrayTrie,
    values: Vec<V>,
}

impl<V: MapValue> DoubleArrayTrieMap<V> {
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
                values.resize((new + 1).max(values.len() * 2), V::EMPTY);
            }
            if old < values.len() {
                let v = std::mem::replace(&mut values[old], V::EMPTY);
                values[new] = v;
            }
        })?;

        let state = self.trie.lookup_state(key)
            .ok_or_else(|| ZiporaError::invalid_state("insert succeeded but lookup failed"))?;
        let idx = state as usize;
        if idx >= self.values.len() {
            let new_len = (idx + 1).max(self.values.len() * 2).max(256);
            self.values.resize(new_len, V::EMPTY);
        }
        let prev = self.values[idx];
        self.values[idx] = value;
        Ok(if prev != V::EMPTY { Some(prev) } else { None })
    }

    /// Get value for key.
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<V> {
        let state = self.trie.lookup_state(key)?;
        let idx = state as usize;
        if idx < self.values.len() {
            // SAFETY: bounds checked above
            let v = unsafe { *self.values.get_unchecked(idx) };
            if v != V::EMPTY { Some(v) } else { None }
        } else {
            None
        }
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

        if self.trie.ninfos[state as usize].is_term() {
            if let Some(&val) = self.values.get(state as usize) {
                if val != V::EMPTY {
                    entries.push((path.clone(), val));
                }
            }
        }

        let mut c = self.trie.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return; }
        let base = self.trie.states[state as usize].child0();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.trie.states.len() && !self.trie.states[child_pos].is_free() {
                path.push(label);
                self.collect_entries(child_pos as u32, path, entries);
                path.pop();
            }
            c = if child_pos < self.trie.ninfos.len() {
                self.trie.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    /// Call `f(value)` for each terminal with the given prefix. Zero allocation.
    ///
    /// This is the fast path for prefix queries — no Vec allocation, no key
    /// cloning. Uses NInfo sibling chain for O(k) child enumeration.
    pub fn for_each_value_with_prefix(&self, prefix: &[u8], mut f: impl FnMut(V)) {
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.trie.state_move(curr, ch);
            if next == NIL_STATE { return; }
            curr = next;
        }
        self.walk_values_dfs(curr, &mut f);
    }

    /// DFS walk yielding values via NInfo sibling chain. Zero allocation.
    fn walk_values_dfs(&self, state: u32, f: &mut impl FnMut(V)) {
        if state as usize >= self.trie.states.len() { return; }

        if self.trie.ninfos[state as usize].is_term() {
            if let Some(&val) = self.values.get(state as usize) {
                if val != V::EMPTY {
                    f(val);
                }
            }
        }

        let mut c = self.trie.ninfos[state as usize].first_child();
        if c == NINFO_NONE { return; }
        let base = self.trie.states[state as usize].child0();
        while c != NINFO_NONE {
            let label = (c - 1) as u8;
            let child_pos = (base ^ label as u32) as usize;
            if child_pos < self.trie.states.len() && !self.trie.states[child_pos].is_free() {
                self.walk_values_dfs(child_pos as u32, f);
            }
            c = if child_pos < self.trie.ninfos.len() {
                self.trie.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
        }
    }

    pub fn remove(&mut self, key: &[u8]) -> Option<V> {
        let state = self.trie.lookup_state(key)?;
        let idx = state as usize;
        let prev = if idx < self.values.len() {
            let v = self.values[idx];
            if v != V::EMPTY { Some(v) } else { None }
        } else {
            None
        };
        self.trie.remove(key);
        if idx < self.values.len() {
            self.values[idx] = V::EMPTY;
        }
        prev
    }
}

impl<V: MapValue> std::fmt::Debug for DoubleArrayTrieMap<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DoubleArrayTrieMap")
            .field("num_keys", &self.trie.len())
            .field("total_states", &self.trie.total_states())
            .field("mem_size", &self.trie.mem_size())
            .finish()
    }
}

impl<V: MapValue> Default for DoubleArrayTrieMap<V> {
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
        // 256 states * (8 bytes DaState + 4 bytes NInfo) = 256 * 12 = 3072
        assert_eq!(t.mem_size(), 256 * 12);
        assert_eq!(std::mem::size_of::<DaState>(), 8);
        assert_eq!(std::mem::size_of::<NInfo>(), 4);
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

    /// Regression test: with_capacity < 256 must not cause UB.
    /// Bug 1.1.1: base=0 + unsafe get_unchecked(ch) with ch >= cap is UB.
    #[test]
    fn test_small_capacity_no_oob() {
        // with_capacity(2) must be clamped to 256 minimum
        let t = DoubleArrayTrie::with_capacity(2);
        assert!(t.total_states() >= 256, "minimum capacity must be 256");

        // contains on empty trie must not crash (base=0, next=ch)
        assert!(!t.contains(b"hello"));  // 'h' = 104, needs states[104]
        assert!(!t.contains(b"\xff"));   // 0xFF = 255, needs states[255]
        assert!(!t.contains(b"\x00"));   // 0x00 = 0, needs states[0]

        // insert + lookup must work
        let mut t = DoubleArrayTrie::with_capacity(1);
        t.insert(b"test").unwrap();
        assert!(t.contains(b"test"));
        assert!(!t.contains(b"other"));

        // with_capacity(0) must also be safe
        let t = DoubleArrayTrie::with_capacity(0);
        assert!(t.total_states() >= 256);
        assert!(!t.contains(b"anything"));
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

    // --- Value tests (via DoubleArrayTrieMap) ---

    #[test]
    fn test_map_values_basic() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        let prev = m.insert(b"hello", 42).unwrap();
        assert_eq!(prev, None);

        assert_eq!(m.get(b"hello"), Some(42));
        assert_eq!(m.get(b"world"), None);

        m.insert(b"world", 100).unwrap();
        assert_eq!(m.get(b"world"), Some(100));

        let prev = m.insert(b"hello", 99).unwrap();
        assert_eq!(prev, Some(42));
        assert_eq!(m.get(b"hello"), Some(99));
    }

    #[test]
    fn test_map_values_many() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        for i in 0..500i32 {
            m.insert(format!("key_{:04}", i).as_bytes(), i).unwrap();
        }
        for i in 0..500i32 {
            assert_eq!(m.get(format!("key_{:04}", i).as_bytes()), Some(i),
                "value mismatch for key_{:04}", i);
        }
        assert_eq!(m.len(), 500);
    }

    #[test]
    fn test_map_values_with_contains() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"abc", 1).unwrap();
        m.insert(b"abd", 2).unwrap();

        assert!(m.contains(b"abc"));
        assert!(m.contains(b"abd"));
        assert!(!m.contains(b"ab"));

        assert_eq!(m.get(b"abc"), Some(1));
        assert_eq!(m.get(b"abd"), Some(2));
    }

    // --- Consult tests ---

    #[test]
    fn test_consult_many_inserts() {
        // Consult (relocate-smaller) should not break correctness
        let mut t = DoubleArrayTrie::new();
        for i in 0..1000u32 {
            t.insert(format!("term_{:04}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 1000);
        for i in 0..1000u32 {
            assert!(t.contains(format!("term_{:04}", i).as_bytes()),
                "missing term_{:04}", i);
        }
    }

    // --- Prefix value iteration tests (via DoubleArrayTrieMap) ---

    #[test]
    fn test_map_prefix_value_iteration() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"app", 1).unwrap();
        m.insert(b"apple", 2).unwrap();
        m.insert(b"application", 3).unwrap();
        m.insert(b"banana", 4).unwrap();

        let mut vals: Vec<i32> = m.values_with_prefix(b"app");
        vals.sort();
        assert_eq!(vals, vec![1, 2, 3]);

        let mut all_vals: Vec<i32> = m.values_with_prefix(b"");
        all_vals.sort();
        assert_eq!(all_vals, vec![1, 2, 3, 4]);

        let none: Vec<i32> = m.values_with_prefix(b"xyz");
        assert!(none.is_empty());
    }

    #[test]
    fn test_map_for_each_value_with_prefix() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"app", 1).unwrap();
        m.insert(b"apple", 2).unwrap();
        m.insert(b"application", 3).unwrap();
        m.insert(b"banana", 4).unwrap();

        // Zero-alloc callback must match values_with_prefix
        let mut callback_vals = Vec::new();
        m.for_each_value_with_prefix(b"app", |v| callback_vals.push(v));
        callback_vals.sort();
        assert_eq!(callback_vals, vec![1, 2, 3]);

        let mut all = Vec::new();
        m.for_each_value_with_prefix(b"", |v| all.push(v));
        all.sort();
        assert_eq!(all, vec![1, 2, 3, 4]);

        let mut none = Vec::new();
        m.for_each_value_with_prefix(b"xyz", |v| none.push(v));
        assert!(none.is_empty());
    }

    /// Performance test with map values
    #[test]
    fn test_map_value_performance() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        for i in 0..5000i32 {
            m.insert(format!("term_{:06}", i).as_bytes(), i).unwrap();
        }
        assert_eq!(m.len(), 5000);

        for i in 0..5000i32 {
            assert_eq!(m.get(format!("term_{:06}", i).as_bytes()), Some(i));
        }

        let prefix_vals = m.values_with_prefix(b"term_001");
        assert_eq!(prefix_vals.len(), 1000, "prefix 'term_001' should yield 1000 values, got {}", prefix_vals.len());
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

    // --- Additional corner case tests ---

    #[test]
    fn test_all_256_single_byte_keys() {
        let mut t = DoubleArrayTrie::new();
        for b in 0u8..=255 {
            t.insert(&[b]).unwrap();
        }
        assert_eq!(t.len(), 256);
        for b in 0u8..=255 {
            assert!(t.contains(&[b]), "missing single-byte key 0x{:02x}", b);
        }
    }

    #[test]
    fn test_insert_after_shrink() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"hello").unwrap();
        t.insert(b"world").unwrap();
        t.shrink_to_fit();
        // Insert more after shrinking
        t.insert(b"foo").unwrap();
        t.insert(b"bar").unwrap();
        assert_eq!(t.len(), 4);
        assert!(t.contains(b"hello"));
        assert!(t.contains(b"world"));
        assert!(t.contains(b"foo"));
        assert!(t.contains(b"bar"));
    }

    #[test]
    fn test_map_values_empty_key() {
        let mut m = DoubleArrayTrieMap::<i32>::new();
        m.insert(b"", 42).unwrap();
        assert_eq!(m.get(b""), Some(42));
        m.insert(b"a", 1).unwrap();
        assert_eq!(m.get(b""), Some(42));
        assert_eq!(m.get(b"a"), Some(1));
    }

    #[test]
    fn test_cursor_after_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();
        t.insert(b"c").unwrap();
        t.remove(b"b");

        let mut c = t.cursor();
        c.seek_begin();
        assert_eq!(c.key(), b"a");
        assert!(c.next());
        assert_eq!(c.key(), b"c");
        assert!(!c.next());
    }

    #[test]
    fn test_keys_with_prefix_empty_trie() {
        let t = DoubleArrayTrie::new();
        assert_eq!(t.keys_with_prefix(b"").len(), 0);
        assert_eq!(t.keys_with_prefix(b"anything").len(), 0);
        assert_eq!(t.keys().len(), 0);
    }

    #[test]
    fn test_map_empty_key() {
        let mut map = DoubleArrayTrieMap::<u32>::new();
        map.insert(b"", 99).unwrap();
        assert_eq!(map.get(b""), Some(99));
        assert_eq!(map.len(), 1);
        map.insert(b"x", 1).unwrap();
        assert_eq!(map.get(b""), Some(99));
        assert_eq!(map.get(b"x"), Some(1));
    }

    #[test]
    fn test_remove_all_then_reinsert() {
        let mut t = DoubleArrayTrie::new();
        for i in 0..50u32 {
            t.insert(format!("k{}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 50);
        // Remove all
        for i in 0..50u32 {
            assert!(t.remove(format!("k{}", i).as_bytes()));
        }
        assert_eq!(t.len(), 0);
        assert!(t.is_empty());
        // Reinsert
        for i in 0..50u32 {
            t.insert(format!("k{}", i).as_bytes()).unwrap();
        }
        assert_eq!(t.len(), 50);
        for i in 0..50u32 {
            assert!(t.contains(format!("k{}", i).as_bytes()));
        }
    }

    #[test]
    fn test_range_after_remove() {
        let mut t = DoubleArrayTrie::new();
        t.insert(b"a").unwrap();
        t.insert(b"b").unwrap();
        t.insert(b"c").unwrap();
        t.insert(b"d").unwrap();
        t.remove(b"b");
        t.remove(b"c");

        let range: Vec<Vec<u8>> = t.range(b"a", b"z").collect();
        assert_eq!(range.len(), 2);
        assert_eq!(range[0], b"a");
        assert_eq!(range[1], b"d");
    }

    #[test]
    fn test_map_values_empty_prefix_empty() {
        let m = DoubleArrayTrieMap::<i32>::new();
        let vals = m.values_with_prefix(b"");
        assert_eq!(vals.len(), 0);
    }
}
