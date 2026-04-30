//! Compressed Sparse Parallel Patricia (CSPP) Trie
//!
//! A high-performance, path-compressed radix trie designed for memory efficiency
//! and concurrent read/write access. Ported from the C++ `topling-zip` reference.

use crate::error::{Result, ZiporaError};
use bytemuck::{Pod, Zeroable};

pub const ALIGN_SIZE: usize = 4;
pub const NIL_STATE: u32 = u32::MAX;
pub const MAX_ZPATH: usize = 254;
pub const INITIAL_STATE: u32 = 0;

// Free list: max block size handled by fast bins (in slots).
// Blocks larger than this go to a simple large-block list.
const FREE_LIST_MAX_SLOTS: usize = 128;
// Sentinel for empty free list bucket
const FREE_LIST_NIL: u32 = u32::MAX;

pub const SKIP_SLOTS: [u32; 16] = [
    1,
    1,
    1, // 0, 1, 2
    2,
    2,
    2,
    2,  // 3, 4, 5, 6
    5,  // 7
    10, // 8
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX,
    u32::MAX, // 9-14
    2,        // 15
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
    pub flags: u8, // n_cnt_type:4 | b_is_final:1 | b_lazy_free:1 | b_set_final:1 | b_lock:1
    pub n_zpath_len: u8,
    pub c_label: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BigCount {
    pub _unused: u16,
    pub n_children: u16,
}

unsafe impl Zeroable for MetaInfo {}
unsafe impl Pod for MetaInfo {}

unsafe impl Zeroable for BigCount {}
unsafe impl Pod for BigCount {}

impl PatriciaNode {
    #[inline(always)]
    pub fn empty() -> Self {
        PatriciaNode { child: NIL_STATE }
    }
}

/// Memory pool statistics, matching C++ Patricia::MemStat.
#[derive(Debug, Clone, Default)]
pub struct MemStat {
    pub fastbin: Vec<usize>,
    pub used_size: usize,
    pub capacity: usize,
    pub frag_size: usize,
    pub large_size: usize,
    pub large_cnt: usize,
    pub lazy_free_sum: usize,
    pub lazy_free_cnt: usize,
}

/// Item deferred for lazy reclamation (Phase C.2).
#[derive(Clone, Copy)]
struct LazyFreeItem {
    slot: u32,
    slots: u32,
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
            // SAFETY: cnt_type 7/8/15 store n_children in bytes 2-3 of slot 0
            // via BigCount union (overlaps c_label in MetaInfo).
            // C++ reference: p->big.n_children reads from slot 0.
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
                if ch == self.get_label(2) {
                    return self.child(4);
                }
                if ch == self.get_label(1) {
                    return self.child(3);
                }
                if ch == self.get_label(0) {
                    return self.child(2);
                }
                NIL_STATE
            }
            4 => {
                if ch == self.get_label(3) {
                    return self.child(5);
                }
                if ch == self.get_label(2) {
                    return self.child(4);
                }
                if ch == self.get_label(1) {
                    return self.child(3);
                }
                if ch == self.get_label(0) {
                    return self.child(2);
                }
                NIL_STATE
            }
            5 => {
                if ch == self.get_label(4) {
                    return self.child(6);
                }
                if ch == self.get_label(3) {
                    return self.child(5);
                }
                if ch == self.get_label(2) {
                    return self.child(4);
                }
                if ch == self.get_label(1) {
                    return self.child(3);
                }
                if ch == self.get_label(0) {
                    return self.child(2);
                }
                NIL_STATE
            }
            6 => {
                if ch == self.get_label(5) {
                    return self.child(7);
                }
                if ch == self.get_label(4) {
                    return self.child(6);
                }
                if ch == self.get_label(3) {
                    return self.child(5);
                }
                if ch == self.get_label(2) {
                    return self.child(4);
                }
                if ch == self.get_label(1) {
                    return self.child(3);
                }
                if ch == self.get_label(0) {
                    return self.child(2);
                }
                NIL_STATE
            }
            7 => {
                let n_children = self.n_children();
                // SAFETY: Labels for cnt_type 7 start at slot 1 (byte 4 from node start),
                // spanning 16 bytes across slots 1-4. C++ reference: p[1].bytes
                let label_slice = unsafe {
                    let ptr = self.nodes.as_ptr().add(self.curr as usize + 1) as *const u8;
                    std::slice::from_raw_parts(ptr, 16)
                };
                let idx = crate::fsa::fast_search::fast_search_byte_max_16(
                    &label_slice[0..n_children],
                    ch,
                );
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
                    let data_ptr =
                        unsafe { self.nodes.as_ptr().add(self.curr as usize + 1) as *const u8 };
                    let i = (ch / 64) as usize;
                    let w =
                        unsafe { std::ptr::read_unaligned(data_ptr.add(4 + i * 8) as *const u64) };
                    let b = unsafe { *data_ptr.add(i) } as usize;
                    let mask = (1u64 << (ch % 64)) - 1;
                    let idx = b + (w & mask).count_ones() as usize;
                    self.child(10 + idx)
                } else {
                    NIL_STATE
                }
            }
            15 => self.child(2 + ch as usize),
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
                // SAFETY: Labels at slots 1-4 (16 bytes). C++ reference: p[1].bytes
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
                    while b != 0 {
                        let tz = b.trailing_zeros();
                        let ch = (byte_idx * 8) as u8 + tz as u8;
                        f(ch, self.child(10 + child_idx));
                        child_idx += 1;
                        b &= b - 1;
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

impl std::fmt::Debug for CsppTrie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsppTrie")
            .field("n_words", &self.n_words)
            .field("n_nodes", &self.n_nodes)
            .finish()
    }
}

pub struct CsppTrie {
    pub mempool: Vec<PatriciaNode>,
    pub n_words: usize,
    pub n_nodes: usize,
    pub valsize: usize,
    pub max_word_len: usize,
    // Phase C: size-bucketed free list (intrusive linked list per slot-count bucket)
    fast_bins: Vec<u32>, // fast_bins[slots-1] = head of free list for that slot count
    large_list: Vec<(u32, u32)>, // (slot, n_slots) for blocks > FREE_LIST_MAX_SLOTS
    frag_size: usize,    // total bytes in all free lists
    // Phase C.2: lazy free list for reader safety
    lazy_free_list: Vec<LazyFreeItem>,
}

impl CsppTrie {
    pub fn new(valsize: usize) -> Self {
        let mut trie = Self {
            mempool: Vec::new(),
            n_words: 0,
            n_nodes: 1, // root
            valsize,
            max_word_len: 0,
            fast_bins: vec![FREE_LIST_NIL; FREE_LIST_MAX_SLOTS],
            large_list: Vec::new(),
            frag_size: 0,
            lazy_free_list: Vec::new(),
        };
        trie.init_root();
        trie
    }

    fn init_root(&mut self) {
        // Fast node (15) takes 258 slots (meta + real_cnt + 256 children)
        // plus value size
        let val_slots = self.valsize.div_ceil(4);
        let root_slots = 2 + 256 + val_slots;
        self.mempool.resize(root_slots, PatriciaNode::empty());

        // Setup root meta
        self.mempool[0].meta = MetaInfo {
            flags: 15, // cnt_type = 15
            n_zpath_len: 0,
            c_label: [0, 0],
        };
        // Setup big.n_children
        unsafe {
            let meta_ptr = &mut self.mempool[0].meta as *mut MetaInfo as *mut u8;
            // n_children occupies bytes 2 and 3 of the 4-byte node.
            // We must write it as u16 without overwriting the MetaInfo bytes 0 and 1.
            std::ptr::write_unaligned(meta_ptr.add(2) as *mut u16, 256);
        }
        // Setup real_cnt (0 children initially)
        self.mempool[1].big = BigCount {
            _unused: 0,
            n_children: 0,
        };
        // All children are already NIL_STATE because we initialized with PatriciaNode::empty()
    }

    #[inline]
    pub fn node_view(&self, pos: u32) -> NodeView<'_> {
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

    #[inline]
    pub fn set_value<T: Copy>(&mut self, valpos: usize, val: T) {
        debug_assert!(valpos + std::mem::size_of::<T>() <= self.mempool.len() * 4);
        unsafe {
            let ptr = self.mempool.as_mut_ptr() as *mut u8;
            std::ptr::write_unaligned(ptr.add(valpos) as *mut T, val);
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
                if key[pos..pos + match_len] != zpath[..match_len] {
                    return None;
                }
                pos += match_len;
                if key.len() - pos < zlen - match_len {
                    // key ended before zpath
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

    // ========== Phase C: Memory Pool ==========

    /// Allocate `byte_size` bytes from the mempool.
    /// Checks size-bucketed free list first, then bump-allocates.
    fn alloc_node(&mut self, byte_size: usize) -> u32 {
        let slots = byte_size.div_ceil(4);

        // Fast path: check free list for this slot count
        if slots > 0 && slots <= FREE_LIST_MAX_SLOTS {
            let bin_idx = slots - 1;
            let head = self.fast_bins[bin_idx];
            if head != FREE_LIST_NIL {
                // Pop from intrusive linked list
                let next = unsafe { self.mempool[head as usize].child };
                self.fast_bins[bin_idx] = next;
                self.frag_size -= slots * ALIGN_SIZE;
                return head;
            }
        } else if slots > FREE_LIST_MAX_SLOTS {
            // Search large block list for first-fit
            if let Some(idx) = self
                .large_list
                .iter()
                .position(|&(_, s)| s as usize >= slots)
            {
                let (pos, block_slots) = self.large_list.swap_remove(idx);
                self.frag_size -= block_slots as usize * ALIGN_SIZE;
                // Split remainder back to free list if leftover is significant
                let leftover = block_slots as usize - slots;
                if leftover > 0 {
                    self.free_node(pos + slots as u32, leftover * ALIGN_SIZE);
                }
                return pos;
            }
        }

        // Slow path: bump allocation
        let pos = self.mempool.len() as u32;
        self.mempool
            .resize(self.mempool.len() + slots, PatriciaNode::empty());
        pos
    }

    /// Free `byte_size` bytes starting at `slot` back to the free list.
    fn free_node(&mut self, slot: u32, byte_size: usize) {
        let slots = byte_size.div_ceil(4);
        if slots == 0 {
            return;
        }

        // Shrink-from-end optimization
        if slot as usize + slots == self.mempool.len() {
            self.mempool.truncate(slot as usize);
            return;
        }

        if slots <= FREE_LIST_MAX_SLOTS {
            // Push to size-bucketed free list (intrusive: store next pointer in first slot)
            let bin_idx = slots - 1;
            unsafe {
                (*self.mempool.as_mut_ptr().add(slot as usize)).child = self.fast_bins[bin_idx];
            }
            self.fast_bins[bin_idx] = slot;
        } else {
            // Large block list
            self.large_list.push((slot, slots as u32));
        }
        self.frag_size += slots * ALIGN_SIZE;
    }

    /// Defer freeing a node until all readers have finished (EBR).
    /// For `SingleThreadStrict` mode, call `free_node` directly instead.
    pub fn free_node_deferred_pub(&mut self, slot: u32, byte_size: usize) {
        self.free_node_deferred(slot, byte_size);
    }

    fn free_node_deferred(&mut self, slot: u32, byte_size: usize) {
        let slots = byte_size.div_ceil(4) as u32;
        self.lazy_free_list.push(LazyFreeItem { slot, slots });
    }

    /// Reclaim all deferred free nodes. Call when no readers are active.
    pub fn reclaim_lazy_frees(&mut self) {
        let items: Vec<_> = self.lazy_free_list.drain(..).collect();
        for item in items {
            self.free_node(item.slot, item.slots as usize * ALIGN_SIZE);
        }
    }

    fn realloc_node(&mut self, old_slot: u32, old_size: usize, new_size: usize) -> u32 {
        let old_slots = old_size.div_ceil(4);
        let new_slots = new_size.div_ceil(4);
        if old_slots == new_slots {
            return old_slot;
        }
        // If at end of mempool, extend in place
        if old_slot as usize + old_slots == self.mempool.len() {
            self.mempool
                .resize(old_slot as usize + new_slots, PatriciaNode::empty());
            return old_slot;
        }
        // Allocate new, copy old data
        let new_slot = self.alloc_node(new_size);
        let copy_slots = old_slots.min(new_slots);
        unsafe {
            let src = self.mempool.as_ptr().add(old_slot as usize);
            let dst = self.mempool.as_mut_ptr().add(new_slot as usize);
            std::ptr::copy_nonoverlapping(src, dst, copy_slots);
        }
        self.free_node(old_slot, old_size);
        new_slot
    }

    /// Return memory statistics matching C++ Patricia::MemStat.
    pub fn mem_get_stat(&self) -> MemStat {
        let mut fastbin = Vec::with_capacity(FREE_LIST_MAX_SLOTS);
        for bin_idx in 0..FREE_LIST_MAX_SLOTS {
            let mut count = 0;
            let mut head = self.fast_bins[bin_idx];
            while head != FREE_LIST_NIL {
                count += 1;
                head = unsafe { self.mempool[head as usize].child };
            }
            fastbin.push(count);
        }

        let large_size: usize = self
            .large_list
            .iter()
            .map(|&(_, s)| s as usize * ALIGN_SIZE)
            .sum();
        let lazy_sum: usize = self
            .lazy_free_list
            .iter()
            .map(|i| i.slots as usize * ALIGN_SIZE)
            .sum();

        MemStat {
            fastbin,
            used_size: self.mempool.len() * ALIGN_SIZE,
            capacity: self.mempool.capacity() * ALIGN_SIZE,
            frag_size: self.frag_size,
            large_size,
            large_cnt: self.large_list.len(),
            lazy_free_sum: lazy_sum,
            lazy_free_cnt: self.lazy_free_list.len(),
        }
    }

    /// Total fragmented (reclaimable) bytes across all free lists.
    pub fn mem_frag_size(&self) -> usize {
        self.frag_size
    }

    /// Create a chain of nodes for the remaining key suffix.
    /// Returns (head_slot, valpos_byte_offset).
    fn new_suffix_chain(&mut self, suffix: &[u8]) -> (u32, usize) {
        let mut remaining = suffix;
        let mut head = NIL_STATE;
        let mut prev_child_slot: u32 = NIL_STATE;

        // Link nodes for suffix segments > MAX_ZPATH
        while remaining.len() > MAX_ZPATH {
            let link_size = ALIGN_SIZE * 2 + MAX_ZPATH; // meta(4) + child(4) + zpath(254)
            let node = self.alloc_node(link_size);
            unsafe {
                let p = self.mempool.as_mut_ptr().add(node as usize);
                (*p).meta = MetaInfo {
                    flags: 1, // cnt_type=1
                    n_zpath_len: MAX_ZPATH as u8,
                    c_label: [remaining[MAX_ZPATH], 0],
                };
                (*p.add(1)).child = NIL_STATE; // placeholder, filled by next iteration
                // SAFETY: zpath starts at slot 2 (skip=1, n_children=1 → offset = 2 slots)
                let zpath_dst = p.add(2) as *mut u8;
                std::ptr::copy_nonoverlapping(remaining.as_ptr(), zpath_dst, MAX_ZPATH);
                // Pad 254 → 256 (2 bytes)
                *zpath_dst.add(254) = 0;
                *zpath_dst.add(255) = 0;
            }
            if head == NIL_STATE {
                head = node;
            }
            if prev_child_slot != NIL_STATE {
                unsafe {
                    (*self.mempool.as_mut_ptr().add(prev_child_slot as usize)).child = node;
                }
            }
            prev_child_slot = node + 1; // child pointer is at slot node+1
            remaining = &remaining[MAX_ZPATH + 1..];
        }

        // Final leaf node: cnt_type=0, is_final=true
        let zpath_padded = (remaining.len() + 3) & !3;
        let leaf_size = ALIGN_SIZE + zpath_padded + self.valsize;
        let node = self.alloc_node(leaf_size);
        let valpos;
        unsafe {
            let p = self.mempool.as_mut_ptr().add(node as usize);
            (*p).meta = MetaInfo {
                flags: 0x10, // cnt_type=0, is_final=true
                n_zpath_len: remaining.len() as u8,
                c_label: [0, 0],
            };
            // SAFETY: zpath at byte offset 4 (skip=1, n_children=0)
            let zpath_dst = (p as *mut u8).add(ALIGN_SIZE);
            std::ptr::copy_nonoverlapping(remaining.as_ptr(), zpath_dst, remaining.len());
            for i in remaining.len()..zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            valpos = (node as usize + 1) * ALIGN_SIZE + zpath_padded;
        }
        if head == NIL_STATE {
            head = node;
        }
        if prev_child_slot != NIL_STATE {
            unsafe {
                (*self.mempool.as_mut_ptr().add(prev_child_slot as usize)).child = node;
            }
        }
        (head, valpos)
    }

    /// Build a cnt_type 8 (bitmap) node from sorted labels and children.
    fn build_bitmap_node(
        &mut self,
        labels: &[u8],
        children: &[u32],
        n_children: usize,
        flags: u8,
        zpath_len: usize,
        trailing: &[u8],
        trailing_len: usize,
    ) -> u32 {
        let node_size = (10 + n_children) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(node_size);
        unsafe {
            let p = self.mempool.as_mut_ptr().add(node as usize);
            // Meta: cnt_type=8
            let new_flags = (flags & !0x0F) | 8;
            (*p).meta = MetaInfo {
                flags: new_flags,
                n_zpath_len: zpath_len as u8,
                c_label: [0, 0],
            };
            // n_children in slot 0 bytes 2-3
            std::ptr::write_unaligned((p as *mut u8).add(2) as *mut u16, n_children as u16);
            // Build bitmap at slots 2-9 (32 bytes)
            let bmp = p.add(2) as *mut u8;
            std::ptr::write_bytes(bmp, 0, 32);
            for i in 0..n_children {
                let label = labels[i];
                *bmp.add(label as usize / 8) |= 1 << (label % 8);
            }
            // Compute rank prefix at slot 1 bytes 0-3
            let rank = p.add(1) as *mut u8;
            let mut cumulative = 0u32;
            for q in 0..4 {
                *rank.add(q) = cumulative as u8;
                let w = std::ptr::read_unaligned(bmp.add(q * 8) as *const u64);
                cumulative += w.count_ones();
            }
            // Children at slots 10+
            for i in 0..n_children {
                (*p.add(10 + i)).child = children[i];
            }
            // Trailing data
            if trailing_len > 0 {
                let dst = (p as *mut u8).add((10 + n_children) * ALIGN_SIZE);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    /// Add a child to an existing cnt_type 8 (bitmap) node.
    fn add_state_move_bitmap(&mut self, curr: u32, ch: u8, suffix_node: u32) -> u32 {
        // Phase 1: Extract all data from old node
        let meta = unsafe { self.mempool[curr as usize].meta };
        let zpath_len = meta.n_zpath_len as usize;
        let is_final = meta.flags & 0x10 != 0;
        let old_n = unsafe { self.mempool[curr as usize].big }.n_children as usize;

        let mut bitmap = [0u8; 32];
        let mut rank_prefix = [0u8; 4];
        unsafe {
            let bmp_src = self.mempool.as_ptr().add(curr as usize + 2) as *const u8;
            std::ptr::copy_nonoverlapping(bmp_src, bitmap.as_mut_ptr(), 32);
            let rank_src = self.mempool.as_ptr().add(curr as usize + 1) as *const u8;
            std::ptr::copy_nonoverlapping(rank_src, rank_prefix.as_mut_ptr(), 4);
        }
        let mut old_children = [0u32; 257];
        for i in 0..old_n {
            old_children[i] = unsafe { self.mempool[curr as usize + 10 + i].child };
        }
        let zpath_padded = (zpath_len + 3) & !3;
        let trailing_len = zpath_padded + if is_final { self.valsize } else { 0 };
        let mut trailing = [0u8; 512];
        if trailing_len > 0 {
            let off = (10 + old_n) * ALIGN_SIZE;
            unsafe {
                let src = (self.mempool.as_ptr().add(curr as usize) as *const u8).add(off);
                std::ptr::copy_nonoverlapping(src, trailing.as_mut_ptr(), trailing_len);
            }
        }

        // Phase 2: Find ch's insertion rank, update bitmap
        let ch_rank = {
            let q = (ch / 64) as usize;
            let w = unsafe { std::ptr::read_unaligned(bitmap.as_ptr().add(q * 8) as *const u64) };
            let mask = (1u64 << (ch % 64)) - 1;
            rank_prefix[q] as usize + (w & mask).count_ones() as usize
        };
        bitmap[(ch / 8) as usize] |= 1 << (ch % 8);
        // Recompute rank prefix
        let mut cumulative = 0u32;
        for q in 0..4 {
            rank_prefix[q] = cumulative as u8;
            let w = unsafe { std::ptr::read_unaligned(bitmap.as_ptr().add(q * 8) as *const u64) };
            cumulative += w.count_ones();
        }
        // Insert child at ch_rank
        for i in (ch_rank..old_n).rev() {
            old_children[i + 1] = old_children[i];
        }
        old_children[ch_rank] = suffix_node;
        let new_n = old_n + 1;

        // Phase 3: Build new node
        let node_size = (10 + new_n) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(node_size);
        unsafe {
            let p = self.mempool.as_mut_ptr().add(node as usize);
            (*p).meta = MetaInfo {
                flags: meta.flags, // cnt_type stays 8
                n_zpath_len: zpath_len as u8,
                c_label: [0, 0],
            };
            std::ptr::write_unaligned((p as *mut u8).add(2) as *mut u16, new_n as u16);
            let rank_dst = p.add(1) as *mut u8;
            std::ptr::copy_nonoverlapping(rank_prefix.as_ptr(), rank_dst, 4);
            let bmp_dst = p.add(2) as *mut u8;
            std::ptr::copy_nonoverlapping(bitmap.as_ptr(), bmp_dst, 32);
            for i in 0..new_n {
                (*p.add(10 + i)).child = old_children[i];
            }
            if trailing_len > 0 {
                let dst = (p as *mut u8).add((10 + new_n) * ALIGN_SIZE);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    /// Add a child transition to an existing node, handling all cnt_type transitions.
    /// Returns the slot of the new node (which replaces curr).
    fn add_state_move(&mut self, curr: u32, ch: u8, suffix_node: u32) -> u32 {
        // Phase 1: Extract ALL data from old node into locals
        let meta = unsafe { self.mempool[curr as usize].meta };
        let cnt_type = meta.flags & 0x0F;

        if cnt_type == 8 {
            return self.add_state_move_bitmap(curr, ch, suffix_node);
        }

        let zpath_len = meta.n_zpath_len as usize;
        let is_final = meta.flags & 0x10 != 0;
        let old_skip = SKIP_SLOTS[cnt_type as usize] as usize;
        let old_n: usize = if cnt_type <= 6 {
            cnt_type as usize
        } else {
            unsafe { self.mempool[curr as usize].big }.n_children as usize
        };

        // Extract labels
        let mut labels = [0u8; 17];
        match cnt_type {
            0 => {}
            1 | 2 => {
                labels[0] = meta.c_label[0];
                if cnt_type >= 2 {
                    labels[1] = meta.c_label[1];
                }
            }
            3..=6 => {
                labels[0] = meta.c_label[0];
                labels[1] = meta.c_label[1];
                let pad = unsafe { self.mempool[curr as usize + 1].bytes };
                for i in 2..old_n {
                    labels[i] = pad[i - 2];
                }
            }
            7 => unsafe {
                let src = self.mempool.as_ptr().add(curr as usize + 1) as *const u8;
                for i in 0..old_n {
                    labels[i] = *src.add(i);
                }
            },
            _ => unreachable!(),
        }

        // Extract children
        let mut children = [0u32; 17];
        for i in 0..old_n {
            children[i] = unsafe { self.mempool[curr as usize + old_skip + i].child };
        }

        // Extract trailing data (zpath + optional value)
        let zpath_padded = (zpath_len + 3) & !3;
        let trailing_len = zpath_padded + if is_final { self.valsize } else { 0 };
        let mut trailing = [0u8; 512];
        if trailing_len > 0 {
            let trailing_start = (old_skip + old_n) * ALIGN_SIZE;
            unsafe {
                let src =
                    (self.mempool.as_ptr().add(curr as usize) as *const u8).add(trailing_start);
                std::ptr::copy_nonoverlapping(src, trailing.as_mut_ptr(), trailing_len);
            }
        }

        // Phase 2: Insert ch into sorted labels
        let idx = labels[..old_n].partition_point(|&l| l < ch);
        for i in (idx..old_n).rev() {
            labels[i + 1] = labels[i];
            children[i + 1] = children[i];
        }
        labels[idx] = ch;
        children[idx] = suffix_node;
        let new_n = old_n + 1;

        // Phase 3: Determine new cnt_type and build new node
        let new_cnt_type: u8 = match cnt_type {
            0..=5 => cnt_type + 1,
            6 => 7,
            7 if old_n < 16 => 7,
            7 => 8, // 16 → 17
            _ => unreachable!(),
        };

        if new_cnt_type == 8 {
            return self.build_bitmap_node(
                &labels,
                &children,
                new_n,
                meta.flags,
                zpath_len,
                &trailing,
                trailing_len,
            );
        }

        let new_skip = SKIP_SLOTS[new_cnt_type as usize] as usize;
        let new_size = (new_skip + new_n) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(new_size);

        unsafe {
            let p = self.mempool.as_mut_ptr().add(node as usize);
            let new_flags = (meta.flags & !0x0F) | new_cnt_type;

            match new_cnt_type {
                1 | 2 => {
                    (*p).meta = MetaInfo {
                        flags: new_flags,
                        n_zpath_len: zpath_len as u8,
                        c_label: [labels[0], if new_cnt_type >= 2 { labels[1] } else { 0 }],
                    };
                }
                3..=6 => {
                    (*p).meta = MetaInfo {
                        flags: new_flags,
                        n_zpath_len: zpath_len as u8,
                        c_label: [labels[0], labels[1]],
                    };
                    // Extra labels in slot 1 bytes
                    let pad_ptr = p.add(1) as *mut u8;
                    for i in 2..new_n {
                        *pad_ptr.add(i - 2) = labels[i];
                    }
                    for i in (new_n - 2)..4 {
                        *pad_ptr.add(i) = 0;
                    }
                }
                7 => {
                    (*p).meta = MetaInfo {
                        flags: new_flags,
                        n_zpath_len: zpath_len as u8,
                        c_label: [0, 0],
                    };
                    // n_children in slot 0 bytes 2-3
                    std::ptr::write_unaligned((p as *mut u8).add(2) as *mut u16, new_n as u16);
                    // Labels in slots 1-4
                    let lbl_ptr = p.add(1) as *mut u8;
                    for i in 0..new_n {
                        *lbl_ptr.add(i) = labels[i];
                    }
                    for i in new_n..16 {
                        *lbl_ptr.add(i) = 0;
                    }
                }
                _ => unreachable!(),
            }

            // Write children
            for i in 0..new_n {
                (*p.add(new_skip + i)).child = children[i];
            }

            // Write trailing data
            if trailing_len > 0 {
                let dst = (p as *mut u8).add((new_skip + new_n) * ALIGN_SIZE);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    /// Split a node at a zpath mismatch position.
    /// Creates a new parent (cnt_type=2) with two children: old suffix and new suffix.
    /// Returns (new_parent_slot, old_suffix_slot).
    fn fork(
        &mut self,
        curr: u32,
        zidx: usize,
        old_skip: usize,
        old_n_children: usize,
        zpath_len: usize,
        node_size: usize,
        zpath_buf: &[u8],
        new_char: u8,
        new_suffix_node: u32,
    ) -> (u32, u32) {
        let old_char = zpath_buf[zidx];
        let suffix_zlen = zpath_len - zidx - 1;
        let suffix_zpath_padded = (suffix_zlen + 3) & !3;
        let val_size =
            node_size - ((old_skip + old_n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3));
        let suffix_size = (old_skip + old_n_children) * ALIGN_SIZE + suffix_zpath_padded + val_size;

        // Allocate suffix node (copy of old node with shortened zpath)
        let suffix_node = self.alloc_node(suffix_size);
        unsafe {
            let base = self.mempool.as_mut_ptr();
            let src = base.add(curr as usize) as *const u8;
            let dst = base.add(suffix_node as usize) as *mut u8;
            // Copy structural part (skip + children area)
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            std::ptr::copy_nonoverlapping(src, dst, struct_size);
            // Set new zpath_len
            (*base.add(suffix_node as usize)).meta.n_zpath_len = suffix_zlen as u8;
            // Copy suffix zpath (after the split point)
            let zpath_dst = dst.add(struct_size);
            for i in 0..suffix_zlen {
                *zpath_dst.add(i) = zpath_buf[zidx + 1 + i];
            }
            for i in suffix_zlen..suffix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            // Copy value data (if any)
            if val_size > 0 {
                let old_val_off = struct_size + ((zpath_len + 3) & !3);
                std::ptr::copy_nonoverlapping(
                    src.add(old_val_off),
                    zpath_dst.add(suffix_zpath_padded),
                    val_size,
                );
            }
        }

        // Allocate new parent (cnt_type=2, 2 children, zpath prefix)
        let prefix_zpath_padded = (zidx + 3) & !3;
        let parent_size = 3 * ALIGN_SIZE + prefix_zpath_padded; // meta + child0 + child1 + zpath
        let parent = self.alloc_node(parent_size);
        unsafe {
            let base = self.mempool.as_mut_ptr();
            let p = base.add(parent as usize);
            let (label0, child0, label1, child1) = if old_char < new_char {
                (old_char, suffix_node, new_char, new_suffix_node)
            } else {
                (new_char, new_suffix_node, old_char, suffix_node)
            };
            (*p).meta = MetaInfo {
                flags: 2, // cnt_type=2
                n_zpath_len: zidx as u8,
                c_label: [label0, label1],
            };
            (*p.add(1)).child = child0;
            (*p.add(2)).child = child1;
            // Copy zpath prefix
            let zpath_dst = (p as *mut u8).add(3 * ALIGN_SIZE);
            for i in 0..zidx {
                *zpath_dst.add(i) = zpath_buf[i];
            }
            for i in zidx..prefix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
        }
        (parent, suffix_node)
    }

    /// Split at a position where the key is a prefix of an existing node's zpath.
    /// Creates a prefix node (cnt_type=1, is_final) and a suffix node.
    /// Returns (prefix_node_slot, valpos).
    fn split_zpath(
        &mut self,
        curr: u32,
        split_pos: usize,
        old_skip: usize,
        old_n_children: usize,
        zpath_len: usize,
        node_size: usize,
        zpath_buf: &[u8],
    ) -> (u32, usize) {
        let split_char = zpath_buf[split_pos];
        let suffix_zlen = zpath_len - split_pos - 1;
        let suffix_zpath_padded = (suffix_zlen + 3) & !3;
        let val_size =
            node_size - ((old_skip + old_n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3));
        let suffix_size = (old_skip + old_n_children) * ALIGN_SIZE + suffix_zpath_padded + val_size;

        // Allocate suffix (same structure as old node, shortened zpath)
        let suffix_node = self.alloc_node(suffix_size);
        unsafe {
            let base = self.mempool.as_mut_ptr();
            let src = base.add(curr as usize) as *const u8;
            let dst = base.add(suffix_node as usize) as *mut u8;
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            std::ptr::copy_nonoverlapping(src, dst, struct_size);
            (*base.add(suffix_node as usize)).meta.n_zpath_len = suffix_zlen as u8;
            let zpath_dst = dst.add(struct_size);
            for i in 0..suffix_zlen {
                *zpath_dst.add(i) = zpath_buf[split_pos + 1 + i];
            }
            for i in suffix_zlen..suffix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            if val_size > 0 {
                let old_val_off = struct_size + ((zpath_len + 3) & !3);
                std::ptr::copy_nonoverlapping(
                    src.add(old_val_off),
                    zpath_dst.add(suffix_zpath_padded),
                    val_size,
                );
            }
        }

        // Allocate prefix node (cnt_type=1, is_final=true)
        let prefix_zpath_padded = (split_pos + 3) & !3;
        let prefix_size = 2 * ALIGN_SIZE + prefix_zpath_padded + self.valsize;
        let prefix_node = self.alloc_node(prefix_size);
        let valpos;
        unsafe {
            let base = self.mempool.as_mut_ptr();
            let p = base.add(prefix_node as usize);
            (*p).meta = MetaInfo {
                flags: 1 | 0x10, // cnt_type=1, is_final=true
                n_zpath_len: split_pos as u8,
                c_label: [split_char, 0],
            };
            (*p.add(1)).child = suffix_node;
            let zpath_dst = (p as *mut u8).add(2 * ALIGN_SIZE);
            for i in 0..split_pos {
                *zpath_dst.add(i) = zpath_buf[i];
            }
            for i in split_pos..prefix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            valpos = (prefix_node as usize + 2) * ALIGN_SIZE + prefix_zpath_padded;
        }
        (prefix_node, valpos)
    }

    /// Find the mempool slot containing the child pointer for label `ch`.
    /// Returns NIL_STATE if `ch` is not a child of this node.
    fn find_child_slot(&self, curr: u32, ch: u8) -> u32 {
        let view = self.node_view(curr);
        let cnt_type = view.cnt_type();
        match cnt_type {
            0 => NIL_STATE,
            1 => {
                if ch == view.meta().c_label[0] {
                    curr + 1
                } else {
                    NIL_STATE
                }
            }
            2 => {
                let meta = view.meta();
                if ch == meta.c_label[0] {
                    curr + 1
                } else if ch == meta.c_label[1] {
                    curr + 2
                } else {
                    NIL_STATE
                }
            }
            3..=6 => {
                for i in 0..cnt_type as usize {
                    if ch == view.get_label(i) {
                        return curr + 2 + i as u32;
                    }
                }
                NIL_STATE
            }
            7 => {
                let n = view.n_children();
                let label_slice = unsafe {
                    let ptr = self.mempool.as_ptr().add(curr as usize + 1) as *const u8;
                    std::slice::from_raw_parts(ptr, 16)
                };
                let idx = crate::fsa::fast_search::fast_search_byte_max_16(&label_slice[..n], ch);
                if idx < n {
                    curr + 5 + idx as u32
                } else {
                    NIL_STATE
                }
            }
            8 => {
                let bitmap_slice = unsafe {
                    let ptr = self.mempool.as_ptr().add(curr as usize + 2) as *const u8;
                    std::slice::from_raw_parts(ptr, 32)
                };
                let byte_idx = (ch / 8) as usize;
                let bit_idx = ch % 8;
                if (bitmap_slice[byte_idx] & (1 << bit_idx)) != 0 {
                    let data_ptr =
                        unsafe { self.mempool.as_ptr().add(curr as usize + 1) as *const u8 };
                    let i = (ch / 64) as usize;
                    let w =
                        unsafe { std::ptr::read_unaligned(data_ptr.add(4 + i * 8) as *const u64) };
                    let b = unsafe { *data_ptr.add(i) } as usize;
                    let mask = (1u64 << (ch % 64)) - 1;
                    let idx = b + (w & mask).count_ones() as usize;
                    curr + 10 + idx as u32
                } else {
                    NIL_STATE
                }
            }
            15 => curr + 2 + ch as u32,
            _ => NIL_STATE,
        }
    }

    /// Insert a key into the trie.
    /// Returns (is_new_insertion, valpos_byte_offset).
    pub fn insert(&mut self, key: &[u8]) -> (bool, usize) {
        let mut curr_slot: u32 = NIL_STATE; // slot containing parent's child pointer to curr
        let mut curr: u32 = INITIAL_STATE;
        let mut pos: usize = 0;

        loop {
            // Extract node properties (drop borrow before any mutation)
            let (cnt_type, zpath_len, is_final, skip, n_children) = {
                let view = self.node_view(curr);
                (
                    view.cnt_type(),
                    view.zpath_len(),
                    view.is_final(),
                    view.skip_slots(),
                    view.n_children(),
                )
            };

            let node_size = (skip + n_children) * ALIGN_SIZE
                + ((zpath_len + 3) & !3)
                + if is_final { self.valsize } else { 0 };

            if zpath_len > 0 {
                // Copy zpath to stack buffer before any mutation
                let mut zpath_buf = [0u8; 256];
                let zpath_off = (skip + n_children) * ALIGN_SIZE;
                unsafe {
                    let src =
                        (self.mempool.as_ptr().add(curr as usize) as *const u8).add(zpath_off);
                    std::ptr::copy_nonoverlapping(src, zpath_buf.as_mut_ptr(), zpath_len);
                }

                // Compare key against zpath
                let remaining_key = key.len() - pos;
                let match_len = std::cmp::min(zpath_len, remaining_key);
                let mut mismatch_at: Option<usize> = None;
                for i in 0..match_len {
                    if key[pos + i] != zpath_buf[i] {
                        mismatch_at = Some(i);
                        break;
                    }
                }

                if let Some(zidx) = mismatch_at {
                    // ForkBranch: divergence within zpath
                    let (new_suffix, valpos) = self.new_suffix_chain(&key[pos + zidx + 1..]);
                    let (new_parent, _old_suffix) = self.fork(
                        curr,
                        zidx,
                        skip,
                        n_children,
                        zpath_len,
                        node_size,
                        &zpath_buf[..zpath_len],
                        key[pos + zidx],
                        new_suffix,
                    );
                    if curr_slot != NIL_STATE {
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr_slot as usize)).child = new_parent;
                        }
                    }
                    self.free_node(curr, node_size);
                    self.n_words += 1;
                    if key.len() > self.max_word_len {
                        self.max_word_len = key.len();
                    }
                    return (true, valpos);
                }

                pos += match_len;

                if remaining_key < zpath_len {
                    // SplitZpath: key exhausted within zpath
                    let (prefix_node, valpos) = self.split_zpath(
                        curr,
                        match_len,
                        skip,
                        n_children,
                        zpath_len,
                        node_size,
                        &zpath_buf[..zpath_len],
                    );
                    if curr_slot != NIL_STATE {
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr_slot as usize)).child =
                                prefix_node;
                        }
                    }
                    self.free_node(curr, node_size);
                    self.n_words += 1;
                    if key.len() > self.max_word_len {
                        self.max_word_len = key.len();
                    }
                    return (true, valpos);
                }

                if pos == key.len() {
                    // Key exhausted at zpath end
                    if is_final {
                        let vp = (curr as usize + skip + n_children) * ALIGN_SIZE
                            + ((zpath_len + 3) & !3);
                        return (false, vp);
                    }
                    // MarkFinalState
                    let old_size = node_size;
                    let new_size = old_size + self.valsize;
                    let new_curr = self.realloc_node(curr, old_size, new_size);
                    unsafe {
                        (*self.mempool.as_mut_ptr().add(new_curr as usize))
                            .meta
                            .flags |= 0x10;
                    }
                    if curr_slot != NIL_STATE && new_curr != curr {
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr_slot as usize)).child = new_curr;
                        }
                    }
                    let vp = (new_curr as usize + skip + n_children) * ALIGN_SIZE
                        + ((zpath_len + 3) & !3);
                    self.n_words += 1;
                    if key.len() > self.max_word_len {
                        self.max_word_len = key.len();
                    }
                    return (true, vp);
                }
            } else {
                // No zpath
                if pos == key.len() {
                    if is_final {
                        let vp = (curr as usize + skip + n_children) * ALIGN_SIZE;
                        return (false, vp);
                    }
                    if cnt_type == 15 {
                        // MarkFinalStateOnFastNode: value space already allocated
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr as usize)).meta.flags |= 0x10;
                        }
                        let vp = (curr as usize + 2 + 256) * ALIGN_SIZE;
                        self.n_words += 1;
                        if key.len() > self.max_word_len {
                            self.max_word_len = key.len();
                        }
                        return (true, vp);
                    }
                    // MarkFinalState for non-fast node
                    let old_size = node_size;
                    let new_size = old_size + self.valsize;
                    let new_curr = self.realloc_node(curr, old_size, new_size);
                    unsafe {
                        (*self.mempool.as_mut_ptr().add(new_curr as usize))
                            .meta
                            .flags |= 0x10;
                    }
                    if curr_slot != NIL_STATE && new_curr != curr {
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr_slot as usize)).child = new_curr;
                        }
                    }
                    let vp = (new_curr as usize + skip + n_children) * ALIGN_SIZE;
                    self.n_words += 1;
                    if key.len() > self.max_word_len {
                        self.max_word_len = key.len();
                    }
                    return (true, vp);
                }
            }

            // Transition on key[pos]
            let ch = key[pos];
            let next = self.node_view(curr).state_move(ch);

            if next == NIL_STATE {
                // MatchFail: no child for this byte
                let (suffix_node, valpos) = self.new_suffix_chain(&key[pos + 1..]);

                if cnt_type != 15 {
                    let new_curr = self.add_state_move(curr, ch, suffix_node);
                    if curr_slot != NIL_STATE {
                        unsafe {
                            (*self.mempool.as_mut_ptr().add(curr_slot as usize)).child = new_curr;
                        }
                    }
                    self.free_node(curr, node_size);
                } else {
                    // Fast node: direct child write
                    unsafe {
                        (*self
                            .mempool
                            .as_mut_ptr()
                            .add(curr as usize + 2 + ch as usize))
                        .child = suffix_node;
                        // Increment real count at slot 1
                        let real_cnt = &mut (*self.mempool.as_mut_ptr().add(curr as usize + 1)).big;
                        real_cnt.n_children += 1;
                    }
                }
                self.n_words += 1;
                if key.len() > self.max_word_len {
                    self.max_word_len = key.len();
                }
                return (true, valpos);
            }

            // Advance to next node
            curr_slot = self.find_child_slot(curr, ch);
            curr = next;
            pos += 1;
        }
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
                top = self.stack.pop().expect("stack empty");
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
                if self.stack.last().is_some() {
                    let backtrack_len = 1 + view.zpath_len();
                    self.word
                        .truncate(self.word.len().saturating_sub(backtrack_len));
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
        let top = self.stack.last().expect("stack empty");
        let view = self.trie.node_view(top.state);
        self.trie.get_value(view.valpos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsa::traits::Trie;

    #[test]
    fn test_cspp_trie_basic_insertion_and_lookup() {
        let mut trie = CsppTrie::new(16);
        assert_eq!(trie.num_words(), 0);

        let key1 = b"apple";
        let key2 = b"application";
        let key3 = b"banana";

        trie.insert(key1);
        trie.insert(key2);
        trie.insert(key3);

        assert_eq!(trie.num_words(), 3);

        assert!(trie.contains(key1));
        assert!(trie.contains(key2));
        assert!(trie.contains(key3));
        assert!(!trie.contains(b"app"));
        assert!(!trie.contains(b"bananas"));
    }

    #[test]
    fn test_cspp_trie_large_dataset() {
        let mut trie = CsppTrie::new(16);
        for i in 0..1000 {
            let key = format!("key{:05}", i);
            trie.insert(key.as_bytes());
        }

        assert_eq!(trie.num_words(), 1000);
        for i in 0..1000 {
            let key = format!("key{:05}", i);
            assert!(trie.contains(key.as_bytes()));
        }
        assert!(!trie.contains(b"key10000"));
    }

    // ========== Core Operations Tests ==========

    #[test]
    fn test_empty_trie_state() {
        let trie = CsppTrie::new(0);
        assert_eq!(trie.num_words(), 0);
        assert!(trie.total_states() >= 1, "root node should exist");
        assert!(!trie.contains(b"anything"));
    }

    #[test]
    fn test_single_key_insert_lookup() {
        let mut trie = CsppTrie::new(0);
        let key = b"hello";
        let (is_new, _valpos) = trie.insert(key);
        assert!(is_new);
        assert_eq!(trie.num_words(), 1);
        assert!(trie.lookup(key).is_some());
        assert!(trie.contains(key));
    }

    #[test]
    fn test_duplicate_insert_returns_same_valpos() {
        let mut trie = CsppTrie::new(4);
        let key = b"duplicate";
        let (is_new1, valpos1) = trie.insert(key);
        assert!(is_new1);
        let (is_new2, valpos2) = trie.insert(key);
        assert!(!is_new2, "second insert should not be new");
        assert_eq!(valpos1, valpos2, "valpos should be identical");
        assert_eq!(trie.num_words(), 1, "word count should not increase");
    }

    #[test]
    fn test_empty_key_insertion() {
        let mut trie = CsppTrie::new(0);
        let (is_new, _valpos) = trie.insert(b"");
        assert!(is_new);
        assert_eq!(trie.num_words(), 1);
        assert!(trie.contains(b""));
        assert!(!trie.contains(b"x"));
    }

    #[test]
    fn test_prefix_keys() {
        let mut trie = CsppTrie::new(0);
        trie.insert(b"a");
        trie.insert(b"ab");
        trie.insert(b"abc");

        assert_eq!(trie.num_words(), 3);
        assert!(trie.contains(b"a"));
        assert!(trie.contains(b"ab"));
        assert!(trie.contains(b"abc"));
        assert!(!trie.contains(b"abcd"));
    }

    #[test]
    fn test_shared_prefix_keys() {
        let mut trie = CsppTrie::new(0);
        trie.insert(b"apple");
        trie.insert(b"application");
        trie.insert(b"app");

        assert_eq!(trie.num_words(), 3);
        assert!(trie.contains(b"apple"));
        assert!(trie.contains(b"application"));
        assert!(trie.contains(b"app"));
        assert!(!trie.contains(b"appl"));
    }

    #[test]
    fn test_diverging_keys() {
        let mut trie = CsppTrie::new(0);
        trie.insert(b"cat");
        trie.insert(b"car");
        trie.insert(b"cab");

        assert_eq!(trie.num_words(), 3);
        assert!(trie.contains(b"cat"));
        assert!(trie.contains(b"car"));
        assert!(trie.contains(b"cab"));
        assert!(!trie.contains(b"ca"));
        assert!(!trie.contains(b"can"));
    }

    #[test]
    fn test_contains_method() {
        let mut trie = CsppTrie::new(0);
        let keys = [b"foo", b"bar", b"baz"];
        for &key in &keys {
            trie.insert(key);
        }

        for &key in &keys {
            assert!(trie.contains(key), "should contain inserted key");
        }
        assert!(!trie.contains(b"missing"), "should not contain missing key");
        assert!(!trie.contains(b"fo"), "should not contain prefix");
        assert!(!trie.contains(b"foobar"), "should not contain extension");
    }

    // ========== Value Storage Tests ==========

    #[test]
    fn test_value_u32() {
        let mut trie = CsppTrie::new(4);
        let key = b"test";
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);

        let value = 0x12345678u32;
        trie.set_value(valpos, value);
        let retrieved: u32 = trie.get_value(valpos);
        assert_eq!(retrieved, value);
    }

    #[test]
    fn test_value_u64() {
        let mut trie = CsppTrie::new(8);
        let key = b"test64";
        let (is_new, valpos) = trie.insert(key);
        assert!(is_new);

        let value = 0x123456789ABCDEF0u64;
        trie.set_value(valpos, value);
        let retrieved: u64 = trie.get_value(valpos);
        assert_eq!(retrieved, value);
    }

    #[test]
    fn test_valsize_zero() {
        let mut trie = CsppTrie::new(0);
        trie.insert(b"key1");
        trie.insert(b"key2");
        trie.insert(b"key3");

        assert_eq!(trie.num_words(), 3);
        assert!(trie.lookup(b"key1").is_some());
        assert!(trie.lookup(b"key2").is_some());
        assert!(trie.lookup(b"key3").is_some());
    }

    // ========== Node Type Transition Tests ==========

    #[test]
    fn test_cnt_type_growth() {
        let mut trie = CsppTrie::new(0);
        // Insert keys forcing cnt_type transitions: 0 -> 1 -> 2 -> ... -> 8
        for i in 0u8..17 {
            let key = [b'x', i];
            trie.insert(&key);
        }
        assert_eq!(trie.num_words(), 17);
        for i in 0u8..17 {
            let key = [b'x', i];
            assert!(trie.contains(&key));
        }
    }

    #[test]
    fn test_bitmap_node_17_children() {
        let mut trie = CsppTrie::new(0);
        // 17 children forces cnt_type 8 (bitmap)
        for i in 0u8..17 {
            let key = [i];
            trie.insert(&key);
        }
        assert_eq!(trie.num_words(), 17);
        for i in 0u8..17 {
            assert!(trie.contains(&[i]));
        }
    }

    #[test]
    fn test_many_children_single_parent() {
        let mut trie = CsppTrie::new(0);
        // Insert all 256 single-byte keys (forces fast node at root)
        for i in 0u8..=255 {
            trie.insert(&[i]);
        }
        assert_eq!(trie.num_words(), 256);
        for i in 0u8..=255 {
            assert!(trie.contains(&[i]));
        }
    }

    #[test]
    fn test_mixed_depth_keys() {
        let mut trie = CsppTrie::new(0);
        trie.insert(b"a");
        trie.insert(b"ab");
        trie.insert(b"abc");
        trie.insert(b"abcd");
        trie.insert(b"abcde");
        trie.insert(b"b");
        trie.insert(b"bc");

        assert_eq!(trie.num_words(), 7);
        assert!(trie.contains(b"a"));
        assert!(trie.contains(b"ab"));
        assert!(trie.contains(b"abc"));
        assert!(trie.contains(b"abcd"));
        assert!(trie.contains(b"abcde"));
        assert!(trie.contains(b"b"));
        assert!(trie.contains(b"bc"));
    }

    // ========== Zpath Operations Tests ==========

    #[test]
    fn test_long_common_prefix() {
        let mut trie = CsppTrie::new(0);
        let mut key1 = vec![b'a'; 200];
        key1.push(b'x');
        let mut key2 = vec![b'a'; 200];
        key2.push(b'y');

        trie.insert(&key1);
        trie.insert(&key2);

        assert_eq!(trie.num_words(), 2);
        assert!(trie.contains(&key1));
        assert!(trie.contains(&key2));
    }

    #[test]
    fn test_max_zpath_boundary() {
        let mut trie = CsppTrie::new(0);
        let key = vec![b'z'; 254];
        trie.insert(&key);

        assert_eq!(trie.num_words(), 1);
        assert!(trie.contains(&key));
    }

    #[test]
    fn test_very_long_key() {
        let mut trie = CsppTrie::new(0);
        let key = vec![b'm'; 1000];
        trie.insert(&key);

        assert_eq!(trie.num_words(), 1);
        assert!(trie.contains(&key));
        assert!(!trie.contains(&key[..999]));
    }

    // ========== Memory Management Tests ==========

    #[test]
    fn test_mem_stat_after_inserts() {
        let mut trie = CsppTrie::new(4);
        for i in 0..100 {
            let key = format!("key{:03}", i);
            trie.insert(key.as_bytes());
        }

        let stat = trie.mem_get_stat();
        assert_eq!(trie.num_words(), 100);
        assert!(
            stat.used_size > 1000,
            "used_size should reflect 100 insertions (usually > 1000 bytes, got {})",
            stat.used_size
        );
        assert!(
            stat.capacity >= stat.used_size,
            "capacity {} should be >= used_size {}",
            stat.capacity,
            stat.used_size
        );
        // The node pool usually expands to at least 4096 capacity during this test
        assert!(
            stat.capacity >= 4096,
            "capacity should have grown significantly, got {}",
            stat.capacity
        );
    }

    #[test]
    fn test_mem_frag_size() {
        let mut trie = CsppTrie::new(0);
        let frag_initial = trie.mem_frag_size();
        assert_eq!(frag_initial, 0, "new trie should have zero fragmentation");

        trie.insert(b"test1");
        trie.insert(b"test2");
        trie.insert(b"test3");

        // Let's cause structural mutations, which will leave old nodes in memory.
        trie.insert(b"test1xyz");
        trie.insert(b"test2abc");

        trie.reclaim_lazy_frees(); // Reclaim memory to turn it into proper fragmentation
        let frag_after = trie.mem_frag_size();
        assert!(
            frag_after > 0,
            "Trie should have memory fragmentation after mutations"
        );
    }

    #[test]
    fn test_lazy_free_and_reclaim() {
        let mut trie = CsppTrie::new(4);
        trie.insert(b"test");

        // Defer freeing a small node (1 slot = 4 bytes)
        trie.free_node_deferred_pub(10, 4);

        let stat_before = trie.mem_get_stat();
        assert_eq!(stat_before.lazy_free_cnt, 1);
        assert_eq!(stat_before.lazy_free_sum, 4);

        // Reclaim lazy frees
        trie.reclaim_lazy_frees();

        let stat_after = trie.mem_get_stat();
        assert_eq!(stat_after.lazy_free_cnt, 0);
        assert_eq!(stat_after.lazy_free_sum, 0);
    }

    // ========== Iterator Tests ==========

    #[test]
    fn test_iterator_empty() {
        let trie = CsppTrie::new(0);
        let mut iter = CsppTrieIterator::<u32>::new(&trie);
        assert!(
            !iter.seek_begin(),
            "seek_begin on empty trie should return false"
        );
    }

    #[test]
    fn test_iterator_sorted_order() {
        let mut trie = CsppTrie::new(0);
        let keys = [b"dog".as_slice(), b"cat", b"bird", b"ant", b"elephant"];
        for &key in &keys {
            trie.insert(key);
        }

        let mut iter = CsppTrieIterator::<u32>::new(&trie);
        let mut words = Vec::new();
        if iter.seek_begin() {
            words.push(iter.word().to_vec());
            while iter.incr() {
                words.push(iter.word().to_vec());
            }
        }

        // Should be in lexicographic order
        let mut expected: Vec<Vec<u8>> = keys.iter().map(|k| k.to_vec()).collect();
        expected.sort();
        assert_eq!(words, expected);
    }

    #[test]
    fn test_iterator_all_words() {
        let mut trie = CsppTrie::new(0);
        let n = 20;
        for i in 0..n {
            let key = format!("key{:02}", i);
            trie.insert(key.as_bytes());
        }

        let mut iter = CsppTrieIterator::<u32>::new(&trie);
        let mut count = 0;
        if iter.seek_begin() {
            count = 1;
            while iter.incr() {
                count += 1;
            }
        }

        assert_eq!(count, n, "iterator should visit all words");
    }

    #[test]
    fn test_iterator_value_retrieval() {
        let mut trie = CsppTrie::new(8); // Allocate space for values

        let keys = [
            b"apple".as_slice(),
            b"banana".as_slice(),
            b"cherry".as_slice(),
        ];
        for (i, &key) in keys.iter().enumerate() {
            let (_, valpos) = trie.insert(key);
            trie.set_value::<u64>(valpos, (i * 100) as u64);
        }

        let mut iter = CsppTrieIterator::<u64>::new(&trie);
        let mut found_values = Vec::new();
        if iter.seek_begin() {
            found_values.push(iter.value());
            while iter.incr() {
                found_values.push(iter.value());
            }
        }

        // Lexicographical order: "apple" (0), "banana" (100), "cherry" (200)
        assert_eq!(found_values, vec![0, 100, 200]);
    }
}
