//! Compressed Sparse Parallel Patricia (CSPP) Trie
//!
//! A high-performance, path-compressed radix trie designed for memory efficiency
//! and concurrent read/write access. Ported from the C++ `topling-zip` reference.

use crate::error::{Result, ZiporaError};

pub const ALIGN_SIZE: usize = 4;
pub const NIL_STATE: u32 = u32::MAX;
pub const MAX_ZPATH: usize = 254;
pub const INITIAL_STATE: u32 = 0;

pub const SKIP_SLOTS: [u32; 16] = [
    1, 1, 1,           // 0, 1, 2
    2, 2, 2, 2,        // 3, 4, 5, 6
    5,                 // 7
    10,                // 8
    u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, // 9-14
    2,                 // 15
];

#[repr(C, align(4))]
#[derive(Clone, Copy)]
pub union PatriciaNode {
    pub meta: MetaInfo,
    pub big: BigCount,
    pub child: u32,
    pub bytes: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MetaInfo {
    pub flags: u8,        // n_cnt_type:4 | b_is_final:1 | b_lazy_free:1 | b_set_final:1 | b_lock:1
    pub n_zpath_len: u8,
    pub c_label: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BigCount {
    pub _unused: u16,
    pub n_children: u16,
}

impl PatriciaNode {
    #[inline(always)]
    pub fn empty() -> Self {
        PatriciaNode { child: NIL_STATE }
    }
}

pub struct NodeView<'a> {
    nodes: &'a [PatriciaNode],
    curr: u32,
}

impl<'a> NodeView<'a> {
    #[inline(always)]
    pub fn new(nodes: &'a [PatriciaNode], curr: u32) -> Self {
        debug_assert!((curr as usize) < nodes.len());
        Self { nodes, curr }
    }

    #[inline(always)]
    pub fn meta(&self) -> MetaInfo {
        unsafe { self.nodes.get_unchecked(self.curr as usize).meta }
    }

    #[inline(always)]
    pub fn big(&self) -> BigCount {
        unsafe { self.nodes.get_unchecked(self.curr as usize).big }
    }

    #[inline(always)]
    pub fn child(&self, offset: usize) -> u32 {
        unsafe { self.nodes.get_unchecked(self.curr as usize + offset).child }
    }

    #[inline(always)]
    pub fn bytes(&self, offset: usize) -> [u8; 4] {
        unsafe { self.nodes.get_unchecked(self.curr as usize + offset).bytes }
    }

    #[inline(always)]
    pub fn cnt_type(&self) -> u8 {
        self.meta().flags & 0x0F
    }

    #[inline(always)]
    pub fn is_final(&self) -> bool {
        (self.meta().flags & 0x10) != 0
    }

    #[inline(always)]
    pub fn zpath_len(&self) -> usize {
        self.meta().n_zpath_len as usize
    }

    #[inline(always)]
    pub fn n_children(&self) -> usize {
        let t = self.cnt_type();
        if t <= 6 {
            t as usize
        } else {
            self.big().n_children as usize
        }
    }

    #[inline(always)]
    pub fn skip_slots(&self) -> usize {
        SKIP_SLOTS[self.cnt_type() as usize] as usize
    }

    #[inline(always)]
    fn get_label(&self, idx: usize) -> u8 {
        if idx < 2 {
            self.meta().c_label[idx]
        } else {
            self.bytes(1)[idx - 2]
        }
    }

    #[inline(always)]
    pub fn state_move(&self, ch: u8) -> u32 {
        let cnt_type = self.cnt_type();
        match cnt_type {
            0 => NIL_STATE,
            1 => {
                if ch == self.meta().c_label[0] {
                    self.child(1)
                } else {
                    NIL_STATE
                }
            }
            2 => {
                let meta = self.meta();
                if ch == meta.c_label[1] {
                    self.child(2)
                } else if ch == meta.c_label[0] {
                    self.child(1)
                } else {
                    NIL_STATE
                }
            }
            3 => {
                if ch == self.get_label(2) { return self.child(4); }
                if ch == self.get_label(1) { return self.child(3); }
                if ch == self.get_label(0) { return self.child(2); }
                NIL_STATE
            }
            4 => {
                if ch == self.get_label(3) { return self.child(5); }
                if ch == self.get_label(2) { return self.child(4); }
                if ch == self.get_label(1) { return self.child(3); }
                if ch == self.get_label(0) { return self.child(2); }
                NIL_STATE
            }
            5 => {
                if ch == self.get_label(4) { return self.child(6); }
                if ch == self.get_label(3) { return self.child(5); }
                if ch == self.get_label(2) { return self.child(4); }
                if ch == self.get_label(1) { return self.child(3); }
                if ch == self.get_label(0) { return self.child(2); }
                NIL_STATE
            }
            6 => {
                if ch == self.get_label(5) { return self.child(7); }
                if ch == self.get_label(4) { return self.child(6); }
                if ch == self.get_label(3) { return self.child(5); }
                if ch == self.get_label(2) { return self.child(4); }
                if ch == self.get_label(1) { return self.child(3); }
                if ch == self.get_label(0) { return self.child(2); }
                NIL_STATE
            }
            7 => {
                let n_children = self.n_children();
                let label_slice = unsafe {
                    let ptr = self.nodes.as_ptr().add(self.curr as usize + 1) as *const u8;
                    std::slice::from_raw_parts(ptr, 16)
                };
                let idx = crate::fsa::fast_search::fast_search_byte_max_16(&label_slice[0..n_children], ch);
                if idx < n_children {
                    self.child(5 + idx)
                } else {
                    NIL_STATE
                }
            }
            8 => {
                let bitmap_slice = unsafe {
                    let ptr = self.nodes.as_ptr().add(self.curr as usize + 2) as *const u8;
                    std::slice::from_raw_parts(ptr, 32)
                };
                let byte_idx = (ch / 8) as usize;
                let bit_idx = ch % 8;
                if (bitmap_slice[byte_idx] & (1 << bit_idx)) != 0 {
                    let data_ptr = unsafe { self.nodes.as_ptr().add(self.curr as usize + 1) as *const u8 };
                    let i = (ch / 64) as usize;
                    let w = unsafe {
                        std::ptr::read_unaligned(data_ptr.add(4 + i * 8) as *const u64)
                    };
                    let b = unsafe { *data_ptr.add(i) } as usize;
                    let mask = (1u64 << (ch % 64)) - 1;
                    let idx = b + (w & mask).count_ones() as usize;
                    self.child(10 + idx)
                } else {
                    NIL_STATE
                }
            }
            15 => {
                self.child(2 + ch as usize)
            }
            _ => NIL_STATE,
        }
    }

    pub fn zpath_slice(&self) -> &'a [u8] {
        let zlen = self.zpath_len();
        if zlen == 0 {
            return &[];
        }
        let skip = self.skip_slots();
        let n_children = self.n_children();
        let offset = skip + n_children;
        unsafe {
            let ptr = self.nodes.as_ptr().add(self.curr as usize + offset) as *const u8;
            std::slice::from_raw_parts(ptr, zlen)
        }
    }

    pub fn valpos(&self) -> usize {
        let skip = self.skip_slots();
        let n_children = self.n_children();
        let zlen = self.zpath_len();
        let offset = skip + n_children;
        let zpath_padded = (zlen + 3) & !3; // align_up to 4
        (self.curr as usize + offset) * 4 + zpath_padded
    }

    #[inline(always)]
    pub fn for_each_child<F>(&self, mut f: F)
    where
        F: FnMut(u8, u32),
    {
        let cnt_type = self.cnt_type();
        match cnt_type {
            0 => {}
            1 => {
                f(self.meta().c_label[0], self.child(1));
            }
            2 => {
                f(self.meta().c_label[0], self.child(1));
                f(self.meta().c_label[1], self.child(2));
            }
            3 => {
                f(self.get_label(0), self.child(2));
                f(self.get_label(1), self.child(3));
                f(self.get_label(2), self.child(4));
            }
            4 => {
                f(self.get_label(0), self.child(2));
                f(self.get_label(1), self.child(3));
                f(self.get_label(2), self.child(4));
                f(self.get_label(3), self.child(5));
            }
            5 => {
                f(self.get_label(0), self.child(2));
                f(self.get_label(1), self.child(3));
                f(self.get_label(2), self.child(4));
                f(self.get_label(3), self.child(5));
                f(self.get_label(4), self.child(6));
            }
            6 => {
                f(self.get_label(0), self.child(2));
                f(self.get_label(1), self.child(3));
                f(self.get_label(2), self.child(4));
                f(self.get_label(3), self.child(5));
                f(self.get_label(4), self.child(6));
                f(self.get_label(5), self.child(7));
            }
            7 => {
                let n_children = self.n_children();
                let label_slice = unsafe {
                    let ptr = self.nodes.as_ptr().add(self.curr as usize + 1) as *const u8;
                    std::slice::from_raw_parts(ptr, 16)
                };
                for i in 0..n_children {
                    f(label_slice[i], self.child(5 + i));
                }
            }
            8 => {
                let bitmap_slice = unsafe {
                    let ptr = self.nodes.as_ptr().add(self.curr as usize + 2) as *const u8;
                    std::slice::from_raw_parts(ptr, 32)
                };
                let mut child_idx = 0;
                for byte_idx in 0..32 {
                    let mut b = bitmap_slice[byte_idx];
                    let mut bit_offset = 0;
                    while b != 0 {
                        let tz = b.trailing_zeros();
                        let ch = (byte_idx * 8) as u8 + tz as u8;
                        f(ch, self.child(10 + child_idx));
                        child_idx += 1;
                        b &= b - 1;
                        bit_offset += tz + 1;
                    }
                }
            }
            15 => {
                for ch in 0..=255 {
                    let child = self.child(2 + ch as usize);
                    if child != NIL_STATE {
                        f(ch as u8, child);
                    }
                }
            }
            _ => {}
        }
    }

}

pub struct CsppTrie {
    pub mempool: Vec<PatriciaNode>,
    pub n_words: usize,
    pub n_nodes: usize,
    pub valsize: usize,
    pub max_word_len: usize,
}

impl CsppTrie {
    pub fn new(valsize: usize) -> Self {
        let mut trie = Self {
            mempool: Vec::new(),
            n_words: 0,
            n_nodes: 1, // root
            valsize,
            max_word_len: 0,
        };
        trie.init_root();
        trie
    }

    fn init_root(&mut self) {
        // Fast node (15) takes 258 slots (meta + real_cnt + 256 children)
        // plus value size
        let val_slots = (self.valsize + 3) / 4;
        let root_slots = 2 + 256 + val_slots;
        self.mempool.resize(root_slots, PatriciaNode::empty());
        
        // Setup root meta
        self.mempool[0].meta = MetaInfo {
            flags: 15, // cnt_type = 15
            n_zpath_len: 0,
            c_label: [0, 0],
        };
        // Setup big.n_children
        self.mempool[0].big = BigCount {
            _unused: 15,
            n_children: 256,
        };
        // Setup real_cnt (0 children initially)
        self.mempool[1].big = BigCount {
            _unused: 0,
            n_children: 0,
        };
        // All children are already NIL_STATE because we initialized with PatriciaNode::empty()
    }

    #[inline]
    pub fn node_view(&self, pos: u32) -> NodeView {
        NodeView::new(&self.mempool, pos)
    }

    #[inline]
    pub fn total_states(&self) -> usize {
        self.mempool.len()
    }

    #[inline]
    pub fn num_words(&self) -> usize {
        self.n_words
    }

    #[inline]
    pub fn get_value<T: Copy>(&self, valpos: usize) -> T {
        debug_assert!(valpos + std::mem::size_of::<T>() <= self.mempool.len() * 4);
        unsafe {
            let ptr = self.mempool.as_ptr() as *const u8;
            std::ptr::read_unaligned(ptr.add(valpos) as *const T)
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Option<usize> {

        let mut curr = INITIAL_STATE;
        let mut pos = 0;

        loop {
            let view = self.node_view(curr);
            let zlen = view.zpath_len();

            if zlen > 0 {
                let zpath = view.zpath_slice();
                let match_len = std::cmp::min(zlen, key.len() - pos);
                if &key[pos..pos + match_len] != &zpath[..match_len] {
                    return None;
                }
                pos += match_len;
                if key.len() - pos < zlen - match_len { // key ended before zpath
                    return None;
                }
                if key.len() == pos {
                    if view.is_final() {
                        return Some(view.valpos());
                    }
                    return None;
                }
            } else {
                if key.len() == pos {
                    if view.is_final() {
                        return Some(view.valpos());
                    }
                    return None;
                }
            }

            let next = view.state_move(key[pos]);
            if next == NIL_STATE {
                return None;
            }
            curr = next;
            pos += 1;
        }
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        self.lookup(key).is_some()
    }
}


pub struct IterEntry {
    pub state: u32,
    pub child_idx: usize,
    pub n_children: usize,
    pub zpath_consumed: bool,
}

pub struct CsppTrieIterator<'a, T> {
    trie: &'a CsppTrie,
    stack: Vec<IterEntry>,
    word: Vec<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: Copy> CsppTrieIterator<'a, T> {
    pub fn new(trie: &'a CsppTrie) -> Self {
        Self {
            trie,
            stack: Vec::with_capacity(32),
            word: Vec::with_capacity(32),
            _marker: std::marker::PhantomData,
        }
    }







    pub fn seek_begin(&mut self) -> bool {
        self.stack.clear();
        self.word.clear();
        self.stack.push(IterEntry {
            state: INITIAL_STATE,
            child_idx: 0,
            n_children: self.trie.node_view(INITIAL_STATE).n_children(),
            zpath_consumed: false,
        });
        self.descend_leftmost()
    }

    fn descend_leftmost(&mut self) -> bool {
        while let Some(mut top) = self.stack.pop() {
            let view = self.trie.node_view(top.state);
            
            if !top.zpath_consumed {
                let zlen = view.zpath_len();
                if zlen > 0 {
                    self.word.extend_from_slice(view.zpath_slice());
                }
                top.zpath_consumed = true;
                
                self.stack.push(top);
                if view.is_final() {
                    return true;
                }
                top = self.stack.pop().unwrap();
            }

            if top.child_idx < view.n_children() {
                let mut first_child = None;
                let mut current_idx = 0;
                view.for_each_child(|ch, child_state| {
                    if current_idx == top.child_idx {
                        first_child = Some((ch, child_state));
                    }
                    current_idx += 1;
                });

                top.child_idx += 1;
                self.stack.push(top);

                if let Some((ch, child_state)) = first_child {
                    self.word.push(ch);
                    self.stack.push(IterEntry {
                        state: child_state,
                        child_idx: 0,
                        n_children: self.trie.node_view(child_state).n_children(),
                        zpath_consumed: false,
                    });
                }
            } else {
                self.stack.push(top);
                return self.incr();
            }
        }
        false
    }

    pub fn incr(&mut self) -> bool {
        while let Some(mut top) = self.stack.pop() {
            let view = self.trie.node_view(top.state);
            
            if top.child_idx < view.n_children() {
                let mut next_child = None;
                let mut current_idx = 0;
                view.for_each_child(|ch, child_state| {
                    if current_idx == top.child_idx {
                        next_child = Some((ch, child_state));
                    }
                    current_idx += 1;
                });
                
                top.child_idx += 1;
                self.stack.push(top);

                if let Some((ch, child_state)) = next_child {
                    self.word.push(ch);
                    self.stack.push(IterEntry {
                        state: child_state,
                        child_idx: 0,
                        n_children: self.trie.node_view(child_state).n_children(),
                        zpath_consumed: false,
                    });
                    if self.descend_leftmost() {
                        return true;
                    }
                }
            } else {
                if let Some(_) = self.stack.last() {
                    let backtrack_len = 1 + view.zpath_len();
                    self.word.truncate(self.word.len().saturating_sub(backtrack_len));
                } else {
                    self.word.clear();
                    return false;
                }
            }
        }
        false
    }

    pub fn word(&self) -> &[u8] {
        &self.word
    }

    pub fn value(&self) -> T {
        let top = self.stack.last().unwrap();
        let view = self.trie.node_view(top.state);
        self.trie.get_value(view.valpos())
    }
}
