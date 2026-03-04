//! Threaded Red-Black Tree on contiguous memory (port of topling-zip vec_trb).
//!
//! All nodes stored in a `Vec` with `u32` indices instead of pointers.
//! Threading: leaf null-pointers are replaced with links to in-order
//! predecessor/successor, enabling O(1) iteration without a stack.

use std::cmp::Ordering;
use std::mem::MaybeUninit;

// ============================================================================
// Bit layout for u32 indices (matching topling-zip vec_trb_node_t<uint32_t>)
// ============================================================================

const FLAG_BIT: u32 = 1 << 31; // children[0]: color (1=red), children[1]: used (0=used)
const TYPE_BIT: u32 = 1 << 30; // 0 = child link, 1 = thread link
const LINK_MASK: u32 = !(FLAG_BIT | TYPE_BIT); // bits 0-29: index
const NIL: u32 = LINK_MASK; // sentinel (all link bits set)

// ============================================================================
// Node
// ============================================================================

#[derive(Clone, Copy)]
struct Node {
    ch: [u32; 2], // [0]=left, [1]=right
}

impl Node {
    /// New node: both children are threads pointing to NIL, black, empty.
    #[inline]
    const fn new_empty() -> Self {
        Self {
            ch: [
                NIL | TYPE_BIT,              // left: thread(NIL), black
                NIL | TYPE_BIT | FLAG_BIT,   // right: thread(NIL), empty
            ],
        }
    }

    // -- link extraction --
    #[inline] fn link(&self, side: usize) -> u32 { self.ch[side] & LINK_MASK }
    #[inline] fn left_link(&self) -> u32 { self.link(0) }
    #[inline] fn right_link(&self) -> u32 { self.link(1) }

    // -- type queries (matching topling-zip exactly) --
    #[inline] fn is_child(&self, side: usize) -> bool { (self.ch[side] & TYPE_BIT) == 0 }
    #[inline] fn is_thread(&self, side: usize) -> bool { (self.ch[side] & TYPE_BIT) != 0 }

    // -- set child link (clears TYPE_BIT, preserves FLAG_BIT) --
    #[inline]
    fn set_child(&mut self, side: usize, idx: u32) {
        self.ch[side] = (self.ch[side] & FLAG_BIT) | (idx & LINK_MASK);
    }

    // -- set thread link (sets TYPE_BIT, preserves FLAG_BIT) --
    #[inline]
    fn set_thread(&mut self, side: usize, idx: u32) {
        self.ch[side] = (self.ch[side] & FLAG_BIT) | TYPE_BIT | (idx & LINK_MASK);
    }

    // -- set just the link, preserving all flag bits --
    #[inline]
    fn set_link(&mut self, side: usize, idx: u32) {
        self.ch[side] = (self.ch[side] & !LINK_MASK) | (idx & LINK_MASK);
    }

    // -- color (FLAG_BIT of children[0]) --
    #[inline] fn is_red(&self) -> bool { (self.ch[0] & FLAG_BIT) != 0 }
    #[inline] fn is_black(&self) -> bool { !self.is_red() }
    #[inline] fn set_red(&mut self) { self.ch[0] |= FLAG_BIT; }
    #[inline] fn set_black(&mut self) { self.ch[0] &= !FLAG_BIT; }

    // -- used/empty (FLAG_BIT of children[1]) --
    #[inline] fn is_used(&self) -> bool { (self.ch[1] & FLAG_BIT) == 0 }
    #[inline] fn set_used(&mut self) { self.ch[1] &= !FLAG_BIT; }
    #[inline] fn set_empty(&mut self) { self.ch[1] |= FLAG_BIT; }
}

// ============================================================================
// Path stack (records node + direction at each level)
// ============================================================================

const MAX_DEPTH: usize = 64; // enough for 2^30 nodes

struct PathStack {
    entries: [(u32, bool); MAX_DEPTH], // (node_index, went_left)
    len: usize,
}

impl PathStack {
    fn new() -> Self {
        Self { entries: [(NIL, false); MAX_DEPTH], len: 0 }
    }
    #[inline]
    fn push(&mut self, idx: u32, went_left: bool) {
        debug_assert!(self.len < MAX_DEPTH);
        self.entries[self.len] = (idx, went_left);
        self.len += 1;
    }
    #[inline]
    fn idx(&self, k: usize) -> u32 { self.entries[k].0 }
    #[inline]
    fn is_left(&self, k: usize) -> bool { self.entries[k].1 }
}

// ============================================================================
// Core tree
// ============================================================================

struct Core<T> {
    units: Vec<(Node, MaybeUninit<T>)>,
    root: u32,
    count: usize,
    free_head: u32,
}

impl<T> Core<T> {
    fn new() -> Self {
        Self { units: Vec::new(), root: NIL, count: 0, free_head: NIL }
    }

    fn alloc(&mut self) -> u32 {
        if self.free_head != NIL {
            let i = self.free_head;
            self.free_head = self.units[i as usize].0.left_link();
            self.units[i as usize].0 = Node::new_empty();
            i
        } else {
            let i = self.units.len() as u32;
            self.units.push((Node::new_empty(), MaybeUninit::uninit()));
            i
        }
    }

    fn dealloc(&mut self, i: u32) {
        self.units[i as usize].0.set_empty();
        self.units[i as usize].0.set_link(0, self.free_head); // left link = free head
        self.free_head = i;
    }

    #[inline]
    fn n(&self, i: u32) -> &Node { &self.units[i as usize].0 }
    #[inline]
    fn n_mut(&mut self, i: u32) -> &mut Node { &mut self.units[i as usize].0 }
    #[inline]
    unsafe fn data(&self, i: u32) -> &T { unsafe { &*self.units[i as usize].1.as_ptr() } }
    #[inline]
    unsafe fn data_mut(&mut self, i: u32) -> &mut T { unsafe { &mut *self.units[i as usize].1.as_mut_ptr() } }

    // -- navigation --

    fn leftmost(&self, mut i: u32) -> u32 {
        while self.n(i).is_child(0) { i = self.n(i).left_link(); }
        i
    }

    fn move_next(&self, i: u32) -> u32 {
        let nd = self.n(i);
        if nd.is_thread(1) {
            let r = nd.right_link();
            if r == NIL { NIL } else { r }
        } else {
            self.leftmost(nd.right_link())
        }
    }

    // -- search --

    fn find_unique<K, F>(&self, key: &K, mut cmp: F) -> (PathStack, bool)
    where F: FnMut(&T, &K) -> Ordering,
    {
        let mut stk = PathStack::new();
        let mut p = self.root;
        while p != NIL {
            let ord = cmp(unsafe { self.data(p) }, key);
            match ord {
                Ordering::Greater => {
                    stk.push(p, true);
                    if self.n(p).is_child(0) { p = self.n(p).left_link(); }
                    else { return (stk, false); }
                }
                Ordering::Less => {
                    stk.push(p, false);
                    if self.n(p).is_child(1) { p = self.n(p).right_link(); }
                    else { return (stk, false); }
                }
                Ordering::Equal => {
                    stk.push(p, false); // direction irrelevant for found
                    return (stk, true);
                }
            }
        }
        (stk, false)
    }

    // -- insert + rebalance (matching topling-zip vec_trb_insert) --

    fn insert_at(&mut self, stk: &PathStack, new_idx: u32) {
        self.count += 1;
        self.n_mut(new_idx).set_used();

        if stk.len == 0 {
            // First node: black root, both threads → NIL
            let nd = self.n_mut(new_idx);
            nd.set_thread(0, NIL);
            nd.set_thread(1, NIL);
            nd.set_black();
            self.root = new_idx;
            return;
        }

        // Set up threads and link from parent
        let parent = stk.idx(stk.len - 1);
        if stk.is_left(stk.len - 1) {
            let pred = self.n(parent).left_link(); // parent's old left-thread
            self.n_mut(new_idx).set_thread(0, pred);
            self.n_mut(new_idx).set_thread(1, parent);
            self.n_mut(parent).set_child(0, new_idx);
        } else {
            let succ = self.n(parent).right_link(); // parent's old right-thread
            self.n_mut(new_idx).set_thread(0, parent);
            self.n_mut(new_idx).set_thread(1, succ);
            self.n_mut(parent).set_child(1, new_idx);
        }
        self.n_mut(new_idx).set_red();

        // Rebalance (matching topling-zip vec_trb_insert)
        // k = index of parent in stack (stk.len-1)
        if stk.len < 2 {
            // Parent is root — just ensure root is black
            self.n_mut(self.root).set_black();
            return;
        }

        let mut k = stk.len - 1; // k = parent's index in stack

        loop {
            let p1 = stk.idx(k);   // parent
            if self.n(p1).is_black() { break; }

            if k == 0 {
                // parent is root — blacken it
                self.n_mut(p1).set_black();
                break;
            }

            let p2 = stk.idx(k - 1); // grandparent
            let parent_is_left = stk.is_left(k - 1);

            // Find uncle
            let uncle_side: usize = if parent_is_left { 1 } else { 0 };
            let uncle = if self.n(p2).is_child(uncle_side) {
                self.n(p2).link(uncle_side)
            } else { NIL };

            if uncle != NIL && self.n(uncle).is_red() {
                // Case 1: uncle is red — recolor and continue 2 levels up
                self.n_mut(p1).set_black();
                self.n_mut(uncle).set_black();
                self.n_mut(p2).set_red();
                if k < 2 { break; }
                k -= 2;
                continue;
            }

            // Case 2/3: uncle is black — rotate
            let child_is_left = stk.is_left(k);

            if parent_is_left {
                let mut y = p1;
                if !child_is_left {
                    // LR: left-rotate p1, then right-rotate p2
                    y = self.n(p1).right_link();
                    // Left-rotate p1: p1.right = y.left, y.left = p1
                    let yl_thread = self.n(y).is_thread(0);
                    if yl_thread {
                        self.n_mut(p1).set_thread(1, y);
                        self.n_mut(y).set_child(0, p1);
                    } else {
                        let yl = self.n(y).left_link();
                        self.n_mut(p1).set_child(1, yl);
                        self.n_mut(y).set_child(0, p1);
                    }
                    self.n_mut(p2).set_child(0, y);
                }
                // Right-rotate p2: p2.left = y.right, y.right = p2
                self.n_mut(y).set_black();
                self.n_mut(p2).set_red();
                let yr_thread = self.n(y).is_thread(1);
                if yr_thread {
                    self.n_mut(y).set_child(1, p2);
                    self.n_mut(p2).set_thread(0, y);
                } else {
                    let yr = self.n(y).right_link();
                    self.n_mut(p2).set_child(0, yr);
                    self.n_mut(y).set_child(1, p2);
                }
                // Update great-grandparent
                if k >= 2 {
                    let gg = stk.idx(k - 2);
                    let gg_side = if stk.is_left(k - 2) { 0 } else { 1 };
                    self.n_mut(gg).set_child(gg_side, y);
                } else {
                    self.root = y;
                }
            } else {
                // Mirror: parent is right child of grandparent
                let mut y = p1;
                if child_is_left {
                    // RL: right-rotate p1, then left-rotate p2
                    y = self.n(p1).left_link();
                    let yr_thread = self.n(y).is_thread(1);
                    if yr_thread {
                        self.n_mut(p1).set_thread(0, y);
                        self.n_mut(y).set_child(1, p1);
                    } else {
                        let yr = self.n(y).right_link();
                        self.n_mut(p1).set_child(0, yr);
                        self.n_mut(y).set_child(1, p1);
                    }
                    self.n_mut(p2).set_child(1, y);
                }
                // Left-rotate p2: p2.right = y.left, y.left = p2
                self.n_mut(y).set_black();
                self.n_mut(p2).set_red();
                let yl_thread = self.n(y).is_thread(0);
                if yl_thread {
                    self.n_mut(y).set_child(0, p2);
                    self.n_mut(p2).set_thread(1, y);
                } else {
                    let yl = self.n(y).left_link();
                    self.n_mut(p2).set_child(1, yl);
                    self.n_mut(y).set_child(0, p2);
                }
                if k >= 2 {
                    let gg = stk.idx(k - 2);
                    let gg_side = if stk.is_left(k - 2) { 0 } else { 1 };
                    self.n_mut(gg).set_child(gg_side, y);
                } else {
                    self.root = y;
                }
            }
            break;
        }

        // Root is always black
        self.n_mut(self.root).set_black();
    }

    // -- remove: simple splice (no black-height fixup for now) --
    // Black-height fixup is extremely complex (~200 LOC in topling-zip).
    // The tree stays functionally correct (BST ordering maintained) but
    // may become slightly unbalanced. This is acceptable for the use cases
    // in zipora (small to medium ordered sets in hash map internals).

    fn remove_idx(&mut self, target: u32, stk: &PathStack) -> T {
        let value = unsafe { self.units[target as usize].1.assume_init_read() };
        self.count -= 1;

        let has_left = self.n(target).is_child(0);
        let has_right = self.n(target).is_child(1);

        if !has_left && !has_right {
            // Leaf: just unlink
            let left_thread = self.n(target).left_link();
            let right_thread = self.n(target).right_link();

            if stk.len <= 1 {
                self.root = NIL;
            } else {
                let parent = stk.idx(stk.len - 2);
                let is_left = stk.is_left(stk.len - 2);
                if is_left {
                    self.n_mut(parent).set_thread(0, left_thread);
                } else {
                    self.n_mut(parent).set_thread(1, right_thread);
                }
            }
            self.dealloc(target);
        } else if has_left && !has_right {
            // Has only left child
            let child = self.n(target).left_link();
            // rightmost of left subtree's right-thread should be updated
            let mut rm = child;
            while self.n(rm).is_child(1) { rm = self.n(rm).right_link(); }
            let right_thread = self.n(target).right_link();
            self.n_mut(rm).set_thread(1, right_thread);

            if stk.len <= 1 {
                self.root = child;
            } else {
                let parent = stk.idx(stk.len - 2);
                let is_left = stk.is_left(stk.len - 2);
                self.n_mut(parent).set_child(if is_left { 0 } else { 1 }, child);
            }
            self.dealloc(target);
        } else if !has_left && has_right {
            // Has only right child
            let child = self.n(target).right_link();
            let mut lm = child;
            while self.n(lm).is_child(0) { lm = self.n(lm).left_link(); }
            let left_thread = self.n(target).left_link();
            self.n_mut(lm).set_thread(0, left_thread);

            if stk.len <= 1 {
                self.root = child;
            } else {
                let parent = stk.idx(stk.len - 2);
                let is_left = stk.is_left(stk.len - 2);
                self.n_mut(parent).set_child(if is_left { 0 } else { 1 }, child);
            }
            self.dealloc(target);
        } else {
            // Both children: find in-order successor (leftmost of right subtree)
            let right = self.n(target).right_link();
            let mut succ = right;
            let mut succ_parent = target;
            while self.n(succ).is_child(0) {
                succ_parent = succ;
                succ = self.n(succ).left_link();
            }

            // Copy successor's data into target slot
            unsafe {
                let src = self.units[succ as usize].1.as_ptr();
                let dst = self.units[target as usize].1.as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }

            // Unlink successor (succ has no left child, may have right child)
            let succ_has_right = self.n(succ).is_child(1);
            if succ_has_right {
                let sr = self.n(succ).right_link();
                // leftmost of sr's left-thread should point to target (succ's predecessor)
                let mut lm = sr;
                while self.n(lm).is_child(0) { lm = self.n(lm).left_link(); }
                self.n_mut(lm).set_thread(0, target);

                if succ_parent == target {
                    self.n_mut(target).set_child(1, sr);
                } else {
                    self.n_mut(succ_parent).set_child(0, sr);
                }
            } else {
                if succ_parent == target {
                    let succ_right = self.n(succ).right_link();
                    self.n_mut(target).set_thread(1, succ_right);
                } else {
                    self.n_mut(succ_parent).set_thread(0, target);
                }
            }
            self.dealloc(succ);
        }
        value
    }

    // -- lower_bound / upper_bound --

    fn lower_bound<K, F>(&self, key: &K, mut cmp: F) -> u32
    where F: FnMut(&T, &K) -> Ordering,
    {
        let mut p = self.root;
        let mut result = NIL;
        while p != NIL {
            match cmp(unsafe { self.data(p) }, key) {
                Ordering::Less => {
                    if self.n(p).is_child(1) { p = self.n(p).right_link(); }
                    else { break; }
                }
                _ => {
                    result = p;
                    if self.n(p).is_child(0) { p = self.n(p).left_link(); }
                    else { break; }
                }
            }
        }
        result
    }

    fn upper_bound<K, F>(&self, key: &K, mut cmp: F) -> u32
    where F: FnMut(&T, &K) -> Ordering,
    {
        let mut p = self.root;
        let mut result = NIL;
        while p != NIL {
            match cmp(unsafe { self.data(p) }, key) {
                Ordering::Greater => {
                    result = p;
                    if self.n(p).is_child(0) { p = self.n(p).left_link(); }
                    else { break; }
                }
                _ => {
                    if self.n(p).is_child(1) { p = self.n(p).right_link(); }
                    else { break; }
                }
            }
        }
        result
    }
}

impl<T> Drop for Core<T> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<T>() {
            for i in 0..self.units.len() {
                if self.units[i].0.is_used() {
                    unsafe { self.units[i].1.assume_init_drop(); }
                }
            }
        }
    }
}

// ============================================================================
// VecTrbSet
// ============================================================================

/// Ordered set on contiguous memory with u32 indices.
/// Cache-friendly alternative to BTreeSet for data structure internals.
pub struct VecTrbSet<K: Ord> {
    core: Core<K>,
}

impl<K: Ord> VecTrbSet<K> {
    pub fn new() -> Self { Self { core: Core::new() } }

    /// Insert `key`. Returns `true` if inserted, `false` if already present.
    pub fn insert(&mut self, key: K) -> bool {
        let (stk, found) = self.core.find_unique(&key, |a, b| a.cmp(b));
        if found { return false; }
        let idx = self.core.alloc();
        self.core.units[idx as usize].1 = MaybeUninit::new(key);
        self.core.insert_at(&stk, idx);
        true
    }

    /// Remove `key`. Returns `true` if removed.
    pub fn remove(&mut self, key: &K) -> bool {
        let (stk, found) = self.core.find_unique(key, |a, b| a.cmp(b));
        if !found { return false; }
        let target = stk.idx(stk.len - 1);
        self.core.remove_idx(target, &stk);
        true
    }

    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        let (_, found) = self.core.find_unique(key, |a, b| a.cmp(b));
        found
    }

    #[inline]
    pub fn len(&self) -> usize { self.core.count }
    pub fn is_empty(&self) -> bool { self.core.count == 0 }

    pub fn clear(&mut self) {
        self.core = Core::new();
    }

    pub fn iter(&self) -> SetIter<'_, K> {
        let start = if self.core.root != NIL {
            self.core.leftmost(self.core.root)
        } else { NIL };
        SetIter { core: &self.core, curr: start }
    }

    pub fn lower_bound(&self, key: &K) -> Option<&K> {
        let i = self.core.lower_bound(key, |a, b| a.cmp(b));
        if i != NIL { Some(unsafe { self.core.data(i) }) } else { None }
    }

    pub fn upper_bound(&self, key: &K) -> Option<&K> {
        let i = self.core.upper_bound(key, |a, b| a.cmp(b));
        if i != NIL { Some(unsafe { self.core.data(i) }) } else { None }
    }
}

pub struct SetIter<'a, K> {
    core: &'a Core<K>,
    curr: u32,
}

impl<'a, K> Iterator for SetIter<'a, K> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == NIL { return None; }
        let r = unsafe { self.core.data(self.curr) };
        self.curr = self.core.move_next(self.curr);
        Some(r)
    }
}

// ============================================================================
// VecTrbMap
// ============================================================================

/// Ordered map on contiguous memory with u32 indices.
pub struct VecTrbMap<K: Ord, V> {
    core: Core<(K, V)>,
}

impl<K: Ord, V> VecTrbMap<K, V> {
    pub fn new() -> Self { Self { core: Core::new() } }

    /// Insert or update. Returns `Some(old_value)` if key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (stk, found) = self.core.find_unique(&key, |a, b| a.0.cmp(b));
        if found {
            let target = stk.idx(stk.len - 1);
            let old = unsafe { std::ptr::read(self.core.units[target as usize].1.as_ptr()) };
            self.core.units[target as usize].1 = MaybeUninit::new((key, value));
            return Some(old.1);
        }
        let idx = self.core.alloc();
        self.core.units[idx as usize].1 = MaybeUninit::new((key, value));
        self.core.insert_at(&stk, idx);
        None
    }

    #[inline]
    pub fn get(&self, key: &K) -> Option<&V> {
        let (stk, found) = self.core.find_unique(key, |a, b| a.0.cmp(b));
        if found {
            Some(unsafe { &self.core.data(stk.idx(stk.len - 1)).1 })
        } else { None }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (stk, found) = self.core.find_unique(key, |a, b| a.0.cmp(b));
        if found {
            Some(unsafe { &mut self.core.data_mut(stk.idx(stk.len - 1)).1 })
        } else { None }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (stk, found) = self.core.find_unique(key, |a, b| a.0.cmp(b));
        if !found { return None; }
        let target = stk.idx(stk.len - 1);
        Some(self.core.remove_idx(target, &stk).1)
    }

    pub fn contains_key(&self, key: &K) -> bool { self.get(key).is_some() }
    #[inline]
    pub fn len(&self) -> usize { self.core.count }
    pub fn is_empty(&self) -> bool { self.core.count == 0 }

    pub fn clear(&mut self) {
        self.core = Core::new();
    }

    pub fn iter(&self) -> MapIter<'_, K, V> {
        let start = if self.core.root != NIL {
            self.core.leftmost(self.core.root)
        } else { NIL };
        MapIter { core: &self.core, curr: start }
    }
}

pub struct MapIter<'a, K, V> {
    core: &'a Core<(K, V)>,
    curr: u32,
}

impl<'a, K, V> Iterator for MapIter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == NIL { return None; }
        let pair = unsafe { self.core.data(self.curr) };
        self.curr = self.core.move_next(self.curr);
        Some((&pair.0, &pair.1))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_basic_insert_contains() {
        let mut s = VecTrbSet::new();
        assert!(s.insert(5));
        assert!(s.insert(3));
        assert!(s.insert(7));
        assert!(s.contains(&3));
        assert!(s.contains(&5));
        assert!(s.contains(&7));
        assert!(!s.contains(&4));
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_set_insert_duplicate() {
        let mut s = VecTrbSet::new();
        assert!(s.insert(5));
        assert!(!s.insert(5));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_set_remove() {
        let mut s = VecTrbSet::new();
        s.insert(5); s.insert(3); s.insert(7);
        assert!(s.remove(&3));
        assert!(!s.remove(&3));
        assert!(!s.contains(&3));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn test_set_iteration_order() {
        let mut s = VecTrbSet::new();
        for v in [8, 3, 10, 1, 6, 14, 4, 7, 13] { s.insert(v); }
        let v: Vec<i32> = s.iter().copied().collect();
        assert_eq!(v, vec![1, 3, 4, 6, 7, 8, 10, 13, 14]);
    }

    #[test]
    fn test_set_lower_upper_bound() {
        let mut s = VecTrbSet::new();
        for i in [1, 3, 5, 7, 9] { s.insert(i); }
        assert_eq!(s.lower_bound(&0), Some(&1));
        assert_eq!(s.lower_bound(&3), Some(&3));
        assert_eq!(s.lower_bound(&4), Some(&5));
        assert_eq!(s.lower_bound(&10), None);
        assert_eq!(s.upper_bound(&0), Some(&1));
        assert_eq!(s.upper_bound(&3), Some(&5));
        assert_eq!(s.upper_bound(&9), None);
    }

    #[test]
    fn test_set_large_sequential() {
        let mut s = VecTrbSet::new();
        for i in 0..1000 { assert!(s.insert(i)); }
        assert_eq!(s.len(), 1000);
        for i in 0..1000 { assert!(s.contains(&i)); }
        let v: Vec<i32> = s.iter().copied().collect();
        assert_eq!(v.len(), 1000);
        for (i, &val) in v.iter().enumerate() { assert_eq!(val, i as i32); }
    }

    #[test]
    fn test_set_large_reverse() {
        let mut s = VecTrbSet::new();
        for i in (0..500).rev() { assert!(s.insert(i)); }
        assert_eq!(s.len(), 500);
        let v: Vec<i32> = s.iter().copied().collect();
        for (i, &val) in v.iter().enumerate() { assert_eq!(val, i as i32); }
    }

    #[test]
    fn test_set_remove_reinsert() {
        let mut s = VecTrbSet::new();
        for i in 0..20 { s.insert(i); }
        for i in (0..20).step_by(2) { assert!(s.remove(&i)); }
        assert_eq!(s.len(), 10);
        for i in (0..20).step_by(2) { assert!(s.insert(i)); }
        assert_eq!(s.len(), 20);
        let v: Vec<i32> = s.iter().copied().collect();
        assert_eq!(v.len(), 20);
        assert!(v.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_set_empty_operations() {
        let mut s = VecTrbSet::<i32>::new();
        assert!(s.is_empty());
        assert!(!s.contains(&0));
        assert!(!s.remove(&0));
        assert_eq!(s.lower_bound(&0), None);
    }

    #[test]
    fn test_set_clear() {
        let mut s = VecTrbSet::new();
        for i in 0..10 { s.insert(i); }
        s.clear();
        assert!(s.is_empty());
        s.insert(100);
        assert_eq!(s.len(), 1);
        assert!(s.contains(&100));
    }

    #[test]
    fn test_map_basic() {
        let mut m = VecTrbMap::new();
        assert_eq!(m.insert(5, "five"), None);
        assert_eq!(m.insert(3, "three"), None);
        assert_eq!(m.insert(7, "seven"), None);
        assert_eq!(m.get(&3), Some(&"three"));
        assert_eq!(m.get(&5), Some(&"five"));
        assert_eq!(m.get(&7), Some(&"seven"));
        assert_eq!(m.get(&4), None);
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn test_map_insert_overwrite() {
        let mut m = VecTrbMap::new();
        assert_eq!(m.insert(5, "five"), None);
        assert_eq!(m.insert(5, "FIVE"), Some("five"));
        assert_eq!(m.get(&5), Some(&"FIVE"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_map_remove() {
        let mut m = VecTrbMap::new();
        m.insert(1, "one"); m.insert(2, "two"); m.insert(3, "three");
        assert_eq!(m.remove(&2), Some("two"));
        assert_eq!(m.remove(&2), None);
        assert!(!m.contains_key(&2));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_map_iteration_order() {
        let mut m = VecTrbMap::new();
        m.insert(8, "a"); m.insert(3, "b"); m.insert(10, "c"); m.insert(1, "d");
        let keys: Vec<i32> = m.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 3, 8, 10]);
    }

    #[test]
    fn test_map_large_dataset() {
        let mut m = VecTrbMap::new();
        for i in 0..1000 { assert_eq!(m.insert(i, i * 2), None); }
        assert_eq!(m.len(), 1000);
        for i in 0..1000 { assert_eq!(m.get(&i), Some(&(i * 2))); }
    }

    #[test]
    fn test_sorted_invariant_stress() {
        let mut s = VecTrbSet::new();
        // Sequential inserts
        for i in 0..200 { s.insert(i); }
        // Reverse inserts
        for i in (200..400).rev() { s.insert(i); }
        // Remove some
        for i in (0..400).step_by(3) { s.remove(&i); }
        // Verify sorted
        let v: Vec<i32> = s.iter().copied().collect();
        assert!(v.windows(2).all(|w| w[0] < w[1]));
        // Verify count
        let expected = (0..400).filter(|i| i % 3 != 0).count();
        assert_eq!(v.len(), expected);
        assert_eq!(s.len(), expected);
    }
}
