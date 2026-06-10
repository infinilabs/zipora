
use super::map::{DoubleArrayTrieMap, MapValue};
use super::state::NINFO_NONE;

pub(crate) struct PrefixFrame {
    pub(crate) state: u32,
    /// Next sibling to visit (NInfo encoding: label+1, 0=done).
    pub(crate) next_sibling: u16,
    /// Whether we've checked/yielded this state's terminal value.
    pub(crate) checked_terminal: bool,
    /// Depth of the path when this frame was pushed (path[0..depth] is the key).
    pub(crate) depth: usize,
}

/// Lazy iterator over all (key, value) pairs with keys starting with a given prefix.
///
/// Traverses NInfo sibling chains using an explicit DFS stack, yielding entries
/// one at a time without collecting all results into a Vec. This prevents memory
/// spikes on broad queries (e.g., prefix="a").
pub struct PrefixIterator<'a, V: MapValue> {
    pub(crate) trie: &'a DoubleArrayTrieMap<V>,
    pub(crate) stack: Vec<PrefixFrame>,
    pub(crate) path: Vec<u8>,
}

impl<'a, V: MapValue> Iterator for PrefixIterator<'a, V> {
    type Item = (Vec<u8>, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.stack.last_mut()?;
            let state = frame.state;

            // First, check if this state is terminal and should yield
            if !frame.checked_terminal {
                frame.checked_terminal = true;
                let state_idx = state as usize;
                if state_idx < self.trie.trie.ninfos.len()
                    && self.trie.trie.ninfos[state_idx].is_term()
                    && let Some(&val) = self.trie.values.get(state_idx)
                    && val != V::EMPTY
                {
                    return Some((self.path[..frame.depth].to_vec(), val));
                }
            }

            // Then try to visit the next child
            if frame.next_sibling == NINFO_NONE {
                // No more children — pop
                self.stack.pop();
                continue;
            }

            let label = (frame.next_sibling - 1) as u8;
            let base = self.trie.trie.states[state as usize].child0();
            let child_pos = (base ^ label as u32) as usize;

            // Advance sibling cursor BEFORE pushing child
            frame.next_sibling = if child_pos < self.trie.trie.ninfos.len() {
                self.trie.trie.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
            let parent_depth = frame.depth;

            // Validate child
            if child_pos >= self.trie.trie.states.len()
                || self.trie.trie.states[child_pos].is_free()
            {
                continue;
            }

            // Set path for child
            let child_depth = parent_depth + 1;
            self.path.truncate(parent_depth);
            self.path.push(label);

            // Push child frame
            let first_child = if child_pos < self.trie.trie.ninfos.len() {
                self.trie.trie.ninfos[child_pos].first_child()
            } else {
                NINFO_NONE
            };
            self.stack.push(PrefixFrame {
                state: child_pos as u32,
                next_sibling: first_child,
                checked_terminal: false,
                depth: child_depth,
            });
        }
    }
}

/// Stack frame for fuzzy iteration DFS.
pub(crate) struct FuzzyFrame {
    pub(crate) state: u32,
    pub(crate) next_sibling: u16,
    pub(crate) checked_terminal: bool,
    pub(crate) depth: usize,
}

/// Lazy iterator over all (key, value) pairs within edit distance `max_dist` of a query.
///
/// Uses incremental Levenshtein distance computation with DP rows maintained
/// per depth level. Prunes subtrees where `min(dp_row) > max_dist`.
pub struct FuzzyIterator<'a, V: MapValue> {
    pub(crate) trie: &'a DoubleArrayTrieMap<V>,
    pub(crate) stack: Vec<FuzzyFrame>,
    pub(crate) path: Vec<u8>,
    pub(crate) query: Vec<u8>,
    pub(crate) max_dist: usize,
    /// dp_columns[d] = edit distance row for path[0..d] vs query[0..j], j in 0..=query.len()
    pub(crate) dp_columns: Vec<Vec<usize>>,
    /// Recycled DP row buffers to avoid heap allocation per DFS transition.
    pub(crate) spare_rows: Vec<Vec<usize>>,
}

impl<'a, V: MapValue> FuzzyIterator<'a, V> {
    /// Compute DP row in-place into `row` and return the row minimum.
    pub(crate) fn compute_row_inplace(
        prev_row: &[usize],
        query: &[u8],
        c: u8,
        row: &mut Vec<usize>,
    ) -> usize {
        let len = query.len() + 1;
        row.resize(len, 0);
        row[0] = prev_row[0] + 1;
        let mut min_val = row[0];
        for j in 1..len {
            let cost = if query[j - 1] == c { 0 } else { 1 };
            let val = (prev_row[j] + 1)
                .min(row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
            row[j] = val;
            if val < min_val {
                min_val = val;
            }
        }
        min_val
    }
}

impl<'a, V: MapValue> Iterator for FuzzyIterator<'a, V> {
    type Item = (Vec<u8>, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let frame = self.stack.last_mut()?;
            let state = frame.state;
            let depth = frame.depth;

            // Check terminal
            if !frame.checked_terminal {
                frame.checked_terminal = true;
                let state_idx = state as usize;
                if depth < self.dp_columns.len()
                    && self.dp_columns[depth][self.query.len()] <= self.max_dist
                    && state_idx < self.trie.trie.ninfos.len()
                    && self.trie.trie.ninfos[state_idx].is_term()
                    && let Some(&val) = self.trie.values.get(state_idx)
                    && val != V::EMPTY
                {
                    return Some((self.path[..depth].to_vec(), val));
                }
            }

            // Try next child
            if frame.next_sibling == NINFO_NONE {
                self.stack.pop();
                let target_len = if let Some(parent) = self.stack.last() {
                    parent.depth + 1
                } else {
                    1
                };
                while self.dp_columns.len() > target_len {
                    if let Some(row) = self.dp_columns.pop() {
                        self.spare_rows.push(row);
                    }
                }
                continue;
            }

            let label = (frame.next_sibling - 1) as u8;
            let base = self.trie.trie.states[state as usize].child0();
            let child_pos = (base ^ label as u32) as usize;

            // Advance sibling cursor
            frame.next_sibling = if child_pos < self.trie.trie.ninfos.len() {
                self.trie.trie.ninfos[child_pos].sibling
            } else {
                NINFO_NONE
            };
            let parent_depth = frame.depth;

            // Validate child
            if child_pos >= self.trie.trie.states.len()
                || self.trie.trie.states[child_pos].is_free()
            {
                continue;
            }

            // Compute DP row in-place using a recycled buffer
            let mut row = self.spare_rows.pop().unwrap_or_default();
            let min_val = Self::compute_row_inplace(
                &self.dp_columns[parent_depth],
                &self.query,
                label,
                &mut row,
            );

            if min_val > self.max_dist {
                self.spare_rows.push(row);
                continue;
            }

            // Set path
            let child_depth = parent_depth + 1;
            self.path.truncate(parent_depth);
            self.path.push(label);

            // Set DP column for child depth
            self.dp_columns.truncate(child_depth);
            self.dp_columns.push(row);

            // Push child frame
            let first_child = if child_pos < self.trie.trie.ninfos.len() {
                self.trie.trie.ninfos[child_pos].first_child()
            } else {
                NINFO_NONE
            };
            self.stack.push(FuzzyFrame {
                state: child_pos as u32,
                next_sibling: first_child,
                checked_terminal: false,
                depth: child_depth,
            });
        }
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
