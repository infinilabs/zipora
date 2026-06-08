use crate::error::{Result, ZiporaError};
use std::cmp::Ordering;

use super::iterators::*;
use super::state::*;
use super::trie::*;

pub trait MapValue: Copy + PartialEq {
    /// Sentinel representing "no value stored here".
    const EMPTY: Self;
}

impl MapValue for i32 {
    const EMPTY: Self = i32::MIN;
}
impl MapValue for u32 {
    const EMPTY: Self = u32::MAX;
}
impl MapValue for i64 {
    const EMPTY: Self = i64::MIN;
}
impl MapValue for u64 {
    const EMPTY: Self = u64::MAX;
}
impl MapValue for usize {
    const EMPTY: Self = usize::MAX;
}

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
    pub(crate) trie: DoubleArrayTrie,
    pub(crate) values: Vec<V>,
}

impl<V: MapValue> DoubleArrayTrieMap<V> {
    pub fn new() -> Self {
        Self {
            trie: DoubleArrayTrie::new(),
            values: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            trie: DoubleArrayTrie::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
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

        let state = self
            .trie
            .lookup_state(key)
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
    pub fn contains(&self, key: &[u8]) -> bool {
        self.trie.contains(key)
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.trie.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }

    /// Return all keys in the map.
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.trie.keys()
    }

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
            if next == NIL_STATE {
                return results;
            }
            curr = next;
        }
        let mut path = prefix.to_vec();
        self.collect_entries(curr, &mut path, &mut results);
        results
    }

    /// Return all values for keys starting with the given prefix.
    pub fn values_with_prefix(&self, prefix: &[u8]) -> Vec<V> {
        self.entries_with_prefix(prefix)
            .into_iter()
            .map(|(_, v)| v)
            .collect()
    }

    /// Recursively collect (key, value) entries.
    fn collect_entries(&self, state: u32, path: &mut Vec<u8>, entries: &mut Vec<(Vec<u8>, V)>) {
        if state as usize >= self.trie.states.len() {
            return;
        }

        if self.trie.ninfos[state as usize].is_term()
            && let Some(&val) = self.values.get(state as usize)
            && val != V::EMPTY
        {
            entries.push((path.clone(), val));
        }

        let mut c = self.trie.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return;
        }
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
            if next == NIL_STATE {
                return;
            }
            curr = next;
        }
        self.walk_values_dfs(curr, &mut f);
    }

    /// DFS walk yielding values via NInfo sibling chain. Zero allocation.
    fn walk_values_dfs(&self, state: u32, f: &mut impl FnMut(V)) {
        if state as usize >= self.trie.states.len() {
            return;
        }

        if self.trie.ninfos[state as usize].is_term()
            && let Some(&val) = self.values.get(state as usize)
            && val != V::EMPTY
        {
            f(val);
        }

        let mut c = self.trie.ninfos[state as usize].first_child();
        if c == NINFO_NONE {
            return;
        }
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

    /// Returns a lazy iterator over all (key, value) pairs with keys starting with `prefix`.
    ///
    /// Unlike `entries_with_prefix()`, this does not allocate a Vec of all results.
    /// Results are yielded one at a time via DFS traversal of the trie's sibling chains.
    pub fn iter_prefix(&self, prefix: &[u8]) -> PrefixIterator<'_, V> {
        // Walk prefix to reach the subtree root
        let mut curr = 0u32;
        for &ch in prefix {
            let next = self.trie.state_move(curr, ch);
            if next == NIL_STATE {
                return PrefixIterator {
                    trie: self,
                    stack: Vec::new(),
                    path: Vec::new(),
                };
            }
            curr = next;
        }

        let state = curr as usize;
        let first_child = if state < self.trie.ninfos.len() {
            self.trie.ninfos[state].first_child()
        } else {
            NINFO_NONE
        };

        let path = prefix.to_vec();
        let frame = PrefixFrame {
            state: curr,
            next_sibling: first_child,
            checked_terminal: false,
            depth: prefix.len(),
        };
        PrefixIterator {
            trie: self,
            stack: vec![frame],
            path,
        }
    }

    /// Returns a lazy iterator over all (key, value) pairs within edit distance
    /// `max_dist` of `query`.
    ///
    /// Uses incremental Levenshtein distance with trie-based pruning.
    /// Subtrees where the minimum possible edit distance exceeds `max_dist` are skipped.
    pub fn iter_fuzzy(&self, query: &[u8], max_dist: usize) -> FuzzyIterator<'_, V> {
        // Row 0: edit distance of empty string vs query[0..j]
        let row0: Vec<usize> = (0..=query.len()).collect();

        let first_child = if !self.trie.states.is_empty() {
            self.trie.ninfos[0].first_child()
        } else {
            NINFO_NONE
        };

        let frame = FuzzyFrame {
            state: 0, // root
            next_sibling: first_child,
            checked_terminal: false,
            depth: 0,
        };

        FuzzyIterator {
            trie: self,
            stack: vec![frame],
            path: Vec::new(),
            query: query.to_vec(),
            max_dist,
            dp_columns: vec![row0],
            spare_rows: Vec::new(),
        }
    }

    /// Create a cursor for manual traversal of the map.
    pub fn cursor(&self) -> DoubleArrayTrieMapCursor<'_, V> {
        DoubleArrayTrieMapCursor::new(self)
    }

    /// Iterator over a lexicographic range `[from, to)` in the map.
    pub fn range<'a>(&'a self, from: &[u8], to: &[u8]) -> MapRangeIter<'a, V> {
        let mut cursor = self.cursor();
        let valid = cursor.seek_lower_bound(from);
        MapRangeIter {
            cursor,
            upper_bound: to.to_vec(),
            started: valid,
        }
    }
}

/// Cursor for traversing a `DoubleArrayTrieMap`.
pub struct DoubleArrayTrieMapCursor<'a, V: MapValue> {
    pub(crate) map: &'a DoubleArrayTrieMap<V>,
    pub(crate) inner: DoubleArrayTrieCursor<'a>,
}

impl<'a, V: MapValue> DoubleArrayTrieMapCursor<'a, V> {
    fn new(map: &'a DoubleArrayTrieMap<V>) -> Self {
        Self {
            map,
            inner: DoubleArrayTrieCursor::new(&map.trie),
        }
    }

    /// Current key bytes.
    #[inline]
    pub fn key(&self) -> &[u8] {
        self.inner.key()
    }

    /// Current value.
    #[inline]
    pub fn value(&self) -> Option<V> {
        if self.inner.is_valid() {
            // The last state in the stack is the current terminal state
            let (state, _) = self.inner.stack.last()?;
            self.map
                .values
                .get(*state as usize)
                .cloned()
                .filter(|&v| v != V::EMPTY)
        } else {
            None
        }
    }

    /// Whether the cursor is positioned on a valid key.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Seek to the first key in the map.
    pub fn seek_begin(&mut self) -> bool {
        self.inner.seek_begin()
    }

    /// Seek to the last key in the map.
    pub fn seek_end(&mut self) -> bool {
        self.inner.seek_end()
    }

    /// Seek to the first key >= `target`.
    pub fn seek_lower_bound(&mut self, target: &[u8]) -> bool {
        self.inner.seek_lower_bound(target)
    }

    /// Advance to the next key.
    #[allow(clippy::should_implement_trait)] // inherent next() mutates cursor in-place and returns success bool; not std::iter::Iterator
    pub fn next(&mut self) -> bool {
        self.inner.next()
    }
}

/// Iterator over a lexicographic range in a `DoubleArrayTrieMap`.
pub struct MapRangeIter<'a, V: MapValue> {
    cursor: DoubleArrayTrieMapCursor<'a, V>,
    upper_bound: Vec<u8>,
    started: bool,
}

impl<'a, V: MapValue> Iterator for MapRangeIter<'a, V> {
    type Item = (Vec<u8>, V);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started || !self.cursor.is_valid() {
            return None;
        }

        let key = self.cursor.key();
        if key >= self.upper_bound.as_slice() {
            return None;
        }

        let result_key = key.to_vec();
        let result_val = self.cursor.value().expect("Valid cursor must have a value");
        self.started = self.cursor.next();
        Some((result_key, result_val))
    }
}

/// Stack frame for prefix iteration DFS.
impl<V: MapValue> Default for DoubleArrayTrieMap<V> {
    fn default() -> Self {
        Self::new()
    }
}
