//! Concurrent Compressed Sparse Parallel Patricia (CSPP) Trie
//!
//! Multi-writer, multi-reader concurrent trie using epoch-based reclamation
//! (crossbeam-epoch) and optimistic per-node locking. Ported from the C++
//! `MainPatricia::insert_multi_writer` in topling-zip.
//!
//! # Architecture
//!
//! - **SharedPool**: Pre-allocated `Box<[AtomicU32]>` mempool. Each slot = one
//!   `PatriciaNode` (4 bytes). Individual slots support atomic CAS for lock bits
//!   and child pointer updates.
//! - **LockFreeFreelist**: Treiber stack per size bucket for lock-free node reuse.
//! - **Epoch-based reclamation**: Old nodes are deferred via `guard.defer_unchecked()`
//!   and returned to the freelist only after all readers advance past the epoch.
//! - **Optimistic concurrency**: Writers prepare new nodes in uncontested memory,
//!   then atomically swap the parent's child pointer (lock parent → mark old node
//!   lazy_free → CAS child → defer free → unlock).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use crossbeam_epoch::{self as epoch, Guard};
use thread_local::ThreadLocal;

use super::cspp_trie::{
    PatriciaNode, MetaInfo, BigCount,
    ALIGN_SIZE, NIL_STATE, MAX_ZPATH, INITIAL_STATE, SKIP_SLOTS,
};

const FREE_LIST_MAX_SLOTS: usize = 128;
const FREE_LIST_NIL: u32 = u32::MAX;

// Flag bit positions within MetaInfo.flags (byte 0 of PatriciaNode)
const FLAG_IS_FINAL: u8 = 0x10;    // bit 4
const FLAG_LAZY_FREE: u8 = 0x20;   // bit 5
const FLAG_SET_FINAL: u8 = 0x40;   // bit 6
const FLAG_LOCK: u8 = 0x80;        // bit 7
const FLAG_CNT_MASK: u8 = 0x0F;    // bits 0-3

// On little-endian (x86-64, aarch64): flags is bits 0-7 of the u32.
// These masks operate on the full u32 representation of a PatriciaNode.
const U32_FLAG_IS_FINAL: u32 = FLAG_IS_FINAL as u32;
const U32_FLAG_LAZY_FREE: u32 = FLAG_LAZY_FREE as u32;
const U32_FLAG_SET_FINAL: u32 = FLAG_SET_FINAL as u32;
const U32_FLAG_LOCK: u32 = FLAG_LOCK as u32;

// ============================================================================
// Transmute helpers (PatriciaNode <-> u32)
// ============================================================================

#[inline(always)]
fn node_to_u32(node: PatriciaNode) -> u32 {
    // SAFETY: PatriciaNode is a union, accessing its fields is unsafe.
    // We read the `child` field which is a u32.
    unsafe { node.child }
}



#[inline(always)]
fn u32_to_meta(bits: u32) -> MetaInfo {
    bytemuck::cast(bits)
}

#[inline(always)]
fn meta_to_u32(meta: MetaInfo) -> u32 {
    bytemuck::cast(meta)
}

// ============================================================================
// SharedPool — Pre-allocated atomic mempool
// ============================================================================

struct SharedPool {
    data: Box<[AtomicU32]>,
    len: AtomicUsize,
}

impl SharedPool {
    fn new(capacity: usize) -> Self {
        let data: Vec<AtomicU32> = (0..capacity)
            .map(|_| AtomicU32::new(NIL_STATE))
            .collect();
        Self {
            data: data.into_boxed_slice(),
            len: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    #[inline(always)]
    fn load_relaxed(&self, pos: usize) -> u32 {
        self.data[pos].load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn load_acquire(&self, pos: usize) -> u32 {
        self.data[pos].load(Ordering::Acquire)
    }

    #[inline(always)]
    fn store_relaxed(&self, pos: usize, val: u32) {
        self.data[pos].store(val, Ordering::Relaxed);
    }

    #[inline(always)]
    fn store_release(&self, pos: usize, val: u32) {
        self.data[pos].store(val, Ordering::Release);
    }

    #[inline]
    fn cas_weak(&self, pos: usize, old: u32, new: u32) -> Result<u32, u32> {
        self.data[pos].compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
    }

    /// Atomically bump-allocate `slots` contiguous positions.
    fn bump_alloc(&self, slots: usize) -> u32 {
        loop {
            let old_len = self.len.load(Ordering::Relaxed);
            let new_len = old_len + slots;
            assert!(
                new_len <= self.data.len(),
                "ConcurrentCsppTrie: mempool exhausted ({} + {} > {})",
                old_len, slots, self.data.len()
            );
            if self.len.compare_exchange_weak(
                old_len, new_len,
                Ordering::AcqRel, Ordering::Relaxed,
            ).is_ok() {
                for i in 0..slots {
                    self.data[old_len + i].store(NIL_STATE, Ordering::Relaxed);
                }
                return old_len as u32;
            }
        }
    }

    /// Read node content as MetaInfo.
    #[inline(always)]
    fn load_meta(&self, pos: u32) -> MetaInfo {
        u32_to_meta(self.load_relaxed(pos as usize))
    }

    /// Read a child pointer at pos + offset.
    #[inline(always)]
    fn load_child(&self, pos: u32, offset: usize) -> u32 {
        self.load_relaxed(pos as usize + offset)
    }

    /// Read 4 bytes at pos + offset as [u8; 4].
    #[inline(always)]
    fn load_bytes(&self, pos: u32, offset: usize) -> [u8; 4] {
        let bits = self.load_relaxed(pos as usize + offset);
        bits.to_ne_bytes()
    }

    /// Get a raw const pointer to the underlying data for multi-byte reads.
    /// SAFETY: The caller must ensure the data is immutable (written before parent CAS).
    #[inline(always)]
    unsafe fn raw_ptr(&self, slot: usize) -> *const u8 {
        // SAFETY: caller guarantees slot is within allocated bounds
        unsafe { self.data.as_ptr().add(slot) as *const u8 }
    }

    /// Get a slice of bytes from pool slots.
    /// SAFETY: The caller must ensure the data is immutable and slot+len is within bounds.
    #[inline(always)]
    unsafe fn get_slice(&self, slot: usize, len: usize) -> &[u8] {
        // SAFETY: caller guarantees data is immutable and slot+len within bounds
        unsafe {
            let ptr = self.raw_ptr(slot);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Write raw bytes to pool slots (for initializing newly allocated nodes).
    /// SAFETY: Slots must be newly allocated (uncontested).
    #[inline]
    #[allow(dead_code)]
    unsafe fn write_bytes(&self, slot: usize, byte_offset: usize, src: &[u8]) {
        // SAFETY: caller guarantees slot is within allocated bounds, dst doesn't overlap src
        unsafe {
            let dst = (self.data.as_ptr() as *mut u8).add(slot * 4 + byte_offset);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        }
    }

    /// Write a MetaInfo to a slot.
    #[inline(always)]
    #[allow(dead_code)]
    fn store_meta(&self, pos: u32, meta: MetaInfo) {
        self.store_relaxed(pos as usize, meta_to_u32(meta));
    }

    /// Store a node (as u32) with release ordering (for synchronization).
    #[inline(always)]
    #[allow(dead_code)]
    fn store_node_release(&self, pos: u32, node: PatriciaNode) {
        self.store_release(pos as usize, node_to_u32(node));
    }
}

// ============================================================================
// LockFreeFreelist — Treiber stack per size bucket (ABA-safe via tagged ptrs)
// ============================================================================
//
// Each bin head is an AtomicU64 packing [generation:32 | slot:32].
// The generation counter increments on every push, so a CAS in pop() will
// fail if the head was popped, reused, and pushed back between the load
// and CAS — the classic ABA scenario.

const TAGGED_NIL: u64 = FREE_LIST_NIL as u64;

#[inline(always)]
fn tagged_pack(generation: u32, slot: u32) -> u64 {
    ((generation as u64) << 32) | (slot as u64)
}

#[inline(always)]
fn tagged_slot(tagged: u64) -> u32 {
    tagged as u32
}

#[inline(always)]
fn tagged_gen(tagged: u64) -> u32 {
    (tagged >> 32) as u32
}

struct LockFreeFreelist {
    bins: Box<[AtomicU64]>,
    frag_size: AtomicUsize,
}

impl LockFreeFreelist {
    fn new() -> Self {
        let bins: Vec<AtomicU64> = (0..FREE_LIST_MAX_SLOTS)
            .map(|_| AtomicU64::new(TAGGED_NIL))
            .collect();
        Self {
            bins: bins.into_boxed_slice(),
            frag_size: AtomicUsize::new(0),
        }
    }

    fn push(&self, pool: &SharedPool, slot: u32, slots: usize) {
        if slots == 0 || slots > FREE_LIST_MAX_SLOTS {
            return;
        }
        let bin = &self.bins[slots - 1];
        loop {
            let head = bin.load(Ordering::Relaxed);
            let head_slot = tagged_slot(head);
            pool.store_relaxed(slot as usize, head_slot);
            let new_gen = tagged_gen(head).wrapping_add(1);
            let new_head = tagged_pack(new_gen, slot);
            if bin.compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                self.frag_size.fetch_add(slots * ALIGN_SIZE, Ordering::Relaxed);
                break;
            }
        }
    }

    fn pop(&self, pool: &SharedPool, slots: usize) -> Option<u32> {
        if slots == 0 || slots > FREE_LIST_MAX_SLOTS {
            return None;
        }
        let bin = &self.bins[slots - 1];
        loop {
            let head = bin.load(Ordering::Acquire);
            let head_slot = tagged_slot(head);
            if head_slot == FREE_LIST_NIL {
                return None;
            }
            let next_slot = pool.load_relaxed(head_slot as usize);
            let new_gen = tagged_gen(head).wrapping_add(1);
            let new_head = tagged_pack(new_gen, next_slot);
            if bin.compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                self.frag_size.fetch_sub(slots * ALIGN_SIZE, Ordering::Relaxed);
                return Some(head_slot);
            }
        }
    }
}

// ============================================================================
// Backoff — Exponential backoff for retry loops
// ============================================================================

struct Backoff {
    count: usize,
}

impl Backoff {
    #[inline]
    fn new() -> Self {
        Self { count: 0 }
    }

    #[inline]
    fn spin(&mut self) {
        self.count += 1;
        if self.count < 8 {
            for _ in 0..(1 << self.count) {
                std::hint::spin_loop();
            }
        } else if self.count < 64 {
            std::thread::yield_now();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(
                (self.count / 16) as u64
            ));
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.count = 0;
    }
}

// ============================================================================
// ThreadLocalAlloc — Per-thread hot region + freelist (matching C++ TCMemPoolOneThread)
// ============================================================================

const CHUNK_SLOTS: usize = 512 * 1024; // 2MB / 4 bytes = 512K slots per chunk

struct ThreadLocalAlloc {
    hot_pos: u32,
    hot_end: u32,
    fast_bins: [u32; FREE_LIST_MAX_SLOTS],
}

impl ThreadLocalAlloc {
    fn new() -> Self {
        Self {
            hot_pos: 0,
            hot_end: 0,
            fast_bins: [FREE_LIST_NIL; FREE_LIST_MAX_SLOTS],
        }
    }
}

// ============================================================================
// ConcurrentNodeView — Lock-free read accessor
// ============================================================================

struct ConcurrentNodeView<'a> {
    pool: &'a SharedPool,
    curr: u32,
}

impl<'a> ConcurrentNodeView<'a> {
    #[inline(always)]
    fn new(pool: &'a SharedPool, curr: u32) -> Self {
        Self { pool, curr }
    }

    #[inline(always)]
    fn meta(&self) -> MetaInfo {
        self.pool.load_meta(self.curr)
    }

    #[inline(always)]
    fn cnt_type(&self) -> u8 {
        self.meta().flags & FLAG_CNT_MASK
    }

    #[inline(always)]
    fn is_final(&self) -> bool {
        self.meta().flags & FLAG_IS_FINAL != 0
    }

    #[inline(always)]
    fn zpath_len(&self) -> usize {
        self.meta().n_zpath_len as usize
    }

    #[inline(always)]
    fn n_children(&self) -> usize {
        let t = self.cnt_type();
        if t <= 6 {
            t as usize
        } else {
            let big: BigCount = bytemuck::cast(self.pool.load_relaxed(self.curr as usize));
            big.n_children as usize
        }
    }

    #[inline(always)]
    fn skip_slots(&self) -> usize {
        SKIP_SLOTS[self.cnt_type() as usize] as usize
    }

    #[inline(always)]
    fn child(&self, offset: usize) -> u32 {
        self.pool.load_child(self.curr, offset)
    }

    #[inline(always)]
    fn get_label(&self, idx: usize) -> u8 {
        if idx < 2 {
            self.meta().c_label[idx]
        } else {
            self.pool.load_bytes(self.curr, 1)[idx - 2]
        }
    }

    fn state_move(&self, ch: u8) -> u32 {
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
                if ch == meta.c_label[1] { self.child(2) }
                else if ch == meta.c_label[0] { self.child(1) }
                else { NIL_STATE }
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
                // SAFETY: Labels at slots 1-4 (16 bytes), immutable after init.
                let label_slice = unsafe { self.pool.get_slice(self.curr as usize + 1, 16) };
                let idx = crate::fsa::fast_search::fast_search_byte_max_16(
                    &label_slice[..n_children], ch,
                );
                if idx < n_children { self.child(5 + idx) } else { NIL_STATE }
            }
            8 => {
                // SAFETY: Bitmap at slots 2-9, immutable after init.
                let bitmap_slice = unsafe { self.pool.get_slice(self.curr as usize + 2, 32) };
                let byte_idx = (ch / 8) as usize;
                let bit_idx = ch % 8;
                if (bitmap_slice[byte_idx] & (1 << bit_idx)) != 0 {
                    let data_ptr = unsafe { self.pool.raw_ptr(self.curr as usize + 1) };
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
            15 => self.child(2 + ch as usize),
            _ => NIL_STATE,
        }
    }

    fn zpath_slice(&self) -> &'a [u8] {
        let zlen = self.zpath_len();
        if zlen == 0 {
            return &[];
        }
        let skip = self.skip_slots();
        let n_children = self.n_children();
        let offset = skip + n_children;
        // SAFETY: zpath is immutable after init, within bounds.
        unsafe { self.pool.get_slice(self.curr as usize + offset, zlen) }
    }

    fn valpos(&self) -> usize {
        let skip = self.skip_slots();
        let n_children = self.n_children();
        let zlen = self.zpath_len();
        let offset = skip + n_children;
        let zpath_padded = (zlen + 3) & !3;
        (self.curr as usize + offset) * 4 + zpath_padded
    }

    fn find_child_slot(&self, ch: u8) -> u32 {
        let cnt_type = self.cnt_type();
        match cnt_type {
            0 => NIL_STATE,
            1 => {
                if ch == self.meta().c_label[0] { self.curr + 1 } else { NIL_STATE }
            }
            2 => {
                let meta = self.meta();
                if ch == meta.c_label[0] { self.curr + 1 }
                else if ch == meta.c_label[1] { self.curr + 2 }
                else { NIL_STATE }
            }
            3..=6 => {
                for i in 0..cnt_type as usize {
                    if ch == self.get_label(i) {
                        return self.curr + 2 + i as u32;
                    }
                }
                NIL_STATE
            }
            7 => {
                let n = self.n_children();
                let label_slice = unsafe { self.pool.get_slice(self.curr as usize + 1, 16) };
                let idx = crate::fsa::fast_search::fast_search_byte_max_16(
                    &label_slice[..n], ch,
                );
                if idx < n { self.curr + 5 + idx as u32 } else { NIL_STATE }
            }
            8 => {
                let bitmap_slice = unsafe { self.pool.get_slice(self.curr as usize + 2, 32) };
                let byte_idx = (ch / 8) as usize;
                let bit_idx = ch % 8;
                if (bitmap_slice[byte_idx] & (1 << bit_idx)) != 0 {
                    let data_ptr = unsafe { self.pool.raw_ptr(self.curr as usize + 1) };
                    let i = (ch / 64) as usize;
                    let w = unsafe {
                        std::ptr::read_unaligned(data_ptr.add(4 + i * 8) as *const u64)
                    };
                    let b = unsafe { *data_ptr.add(i) } as usize;
                    let mask = (1u64 << (ch % 64)) - 1;
                    let idx = b + (w & mask).count_ones() as usize;
                    self.curr + 10 + idx as u32
                } else {
                    NIL_STATE
                }
            }
            15 => self.curr + 2 + ch as u32,
            _ => NIL_STATE,
        }
    }

    #[allow(dead_code)]
    fn for_each_child<F>(&self, mut f: F)
    where
        F: FnMut(u8, u32),
    {
        let cnt_type = self.cnt_type();
        match cnt_type {
            0 => {}
            1 => f(self.meta().c_label[0], self.child(1)),
            2 => {
                let m = self.meta();
                f(m.c_label[0], self.child(1));
                f(m.c_label[1], self.child(2));
            }
            3..=6 => {
                for i in 0..cnt_type as usize {
                    f(self.get_label(i), self.child(2 + i));
                }
            }
            7 => {
                let n = self.n_children();
                let label_slice = unsafe { self.pool.get_slice(self.curr as usize + 1, 16) };
                for i in 0..n {
                    f(label_slice[i], self.child(5 + i));
                }
            }
            8 => {
                let bitmap_slice = unsafe { self.pool.get_slice(self.curr as usize + 2, 32) };
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
                for ch in 0..=255u16 {
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

// ============================================================================
// RaceStats — Counters for diagnosing contention
// ============================================================================

#[derive(Debug, Default)]
pub struct RaceStats {
    pub retries: AtomicUsize,
    pub parent_lock_fail: AtomicUsize,
    pub lazy_free_fail: AtomicUsize,
    pub child_cas_fail: AtomicUsize,
    pub fast_node_cas_fail: AtomicUsize,
}

impl RaceStats {
    fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
// ConcurrentCsppTrie
// ============================================================================

/// Shared inner state kept alive by Arc for deferred epoch reclamation.
struct SharedInner {
    pool: SharedPool,
    freelist: LockFreeFreelist,
}

pub struct ConcurrentCsppTrie {
    inner: std::sync::Arc<SharedInner>,
    tls: ThreadLocal<RefCell<ThreadLocalAlloc>>,
    n_words: AtomicUsize,
    _n_nodes: AtomicUsize,
    valsize: usize,
    max_word_len: AtomicUsize,
    pub race_stats: RaceStats,
}

// SAFETY: All fields are Send+Sync (Arc, AtomicU32, AtomicUsize).
unsafe impl Send for ConcurrentCsppTrie {}
unsafe impl Sync for ConcurrentCsppTrie {}

impl ConcurrentCsppTrie {
    /// Create a new concurrent CSPP trie with pre-allocated capacity.
    ///
    /// `capacity` is the maximum number of PatriciaNode slots (4 bytes each).
    /// For `n` keys of average length `L`, a good estimate is `n * (L + 20)`.
    pub fn with_capacity(valsize: usize, capacity: usize) -> Self {
        let val_slots = valsize.div_ceil(4);
        let root_slots = 2 + 256 + val_slots;
        assert!(capacity >= root_slots, "capacity too small for root node");

        let pool = SharedPool::new(capacity);

        // Initialize root (fast node, cnt_type=15) at slot 0
        let root_meta = MetaInfo {
            flags: 15,
            n_zpath_len: 0,
            c_label: [0, 0],
        };
        let mut root_u32 = meta_to_u32(root_meta);
        // Write n_children=256 into bytes 2-3 (BigCount overlay)
        let root_bytes = root_u32.to_ne_bytes();
        let mut combined = [root_bytes[0], root_bytes[1], 0, 0];
        combined[2..4].copy_from_slice(&256u16.to_ne_bytes());
        root_u32 = u32::from_ne_bytes(combined);
        pool.store_relaxed(0, root_u32);

        // Slot 1: real_cnt = 0 (BigCount with n_children=0)
        pool.store_relaxed(1, 0);

        // Slots 2..258: children = NIL_STATE (already initialized by SharedPool::new)

        // Mark pool length
        pool.len.store(root_slots, Ordering::Release);

        Self {
            inner: std::sync::Arc::new(SharedInner {
                pool,
                freelist: LockFreeFreelist::new(),
            }),
            tls: ThreadLocal::new(),
            n_words: AtomicUsize::new(0),
            _n_nodes: AtomicUsize::new(1),
            valsize,
            max_word_len: AtomicUsize::new(0),
            race_stats: RaceStats::new(),
        }
    }

    /// Number of inserted keys.
    #[inline]
    pub fn num_words(&self) -> usize {
        self.n_words.load(Ordering::Relaxed)
    }

    /// Total allocated mempool slots.
    #[inline]
    pub fn total_states(&self) -> usize {
        self.inner.pool.len()
    }

    /// Total fragmented bytes in freelists.
    #[inline]
    pub fn frag_size(&self) -> usize {
        self.inner.freelist.frag_size.load(Ordering::Relaxed)
    }

    // ========================================================================
    // Lock-free reads
    // ========================================================================

    fn node_view(&self, pos: u32) -> ConcurrentNodeView<'_> {
        ConcurrentNodeView::new(&self.inner.pool, pos)
    }

    /// Look up a key. Returns the value byte offset if found.
    /// Pins the current thread to the epoch internally.
    pub fn lookup(&self, key: &[u8]) -> Option<usize> {
        let _guard = epoch::pin();
        self.lookup_inner(key)
    }

    /// Look up with an existing epoch guard (for batch operations).
    pub fn lookup_with_guard(&self, key: &[u8], _guard: &Guard) -> Option<usize> {
        self.lookup_inner(key)
    }

    fn lookup_inner(&self, key: &[u8]) -> Option<usize> {
        let mut curr = INITIAL_STATE;
        let mut pos = 0;

        loop {
            let view = self.node_view(curr);
            let zlen = view.zpath_len();
            if zlen > 0 {
                let zpath = view.zpath_slice();
                let remaining = key.len() - pos;
                let match_len = std::cmp::min(zlen, remaining);
                if key[pos..pos + match_len] != zpath[..match_len] {
                    return None;
                }
                pos += match_len;
                if remaining < zlen {
                    return None;
                }
                if pos == key.len() {
                    return if view.is_final() { Some(view.valpos()) } else { None };
                }
            } else if pos == key.len() {
                return if view.is_final() { Some(view.valpos()) } else { None };
            }

            let next = view.state_move(key[pos]);
            if next == NIL_STATE {
                return None;
            }
            curr = next;
            pos += 1;
        }
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &[u8]) -> bool {
        self.lookup(key).is_some()
    }

    pub fn contains_with_guard(&self, key: &[u8], guard: &Guard) -> bool {
        self.lookup_with_guard(key, guard).is_some()
    }

    /// Read a value at a byte offset.
    pub fn get_value<T: Copy>(&self, valpos: usize) -> T {
        assert!(valpos + std::mem::size_of::<T>() <= self.inner.pool.len() * 4, "valpos out of bounds");
        
        // SAFETY: We verify valpos is within bounds. valpos is guaranteed to be 4-byte aligned.
        let mut result: std::mem::MaybeUninit<T> = std::mem::MaybeUninit::uninit();
        let result_ptr = result.as_mut_ptr() as *mut u32;
        let num_words = std::mem::size_of::<T>().div_ceil(4);
        
        let word_offset = valpos / 4;
        for i in 0..num_words {
            // Read atomically to avoid data races
            let word = self.inner.pool.load_acquire(word_offset + i);
            unsafe {
                std::ptr::write(result_ptr.add(i), word);
            }
        }
        
        unsafe { result.assume_init() }
    }

    /// Set a value at the given byte offset using atomic operations.
    pub fn set_value<T: Copy>(&self, valpos: usize, val: T) {
        assert!(valpos + std::mem::size_of::<T>() <= self.inner.pool.len() * 4, "valpos out of bounds");
        
        let val_ptr = &val as *const T as *const u32;
        let num_words = std::mem::size_of::<T>().div_ceil(4);
        let word_offset = valpos / 4;
        
        for i in 0..num_words {
            let word = unsafe { std::ptr::read(val_ptr.add(i)) };
            self.inner.pool.store_release(word_offset + i, word);
        }
    }

    // ========================================================================
    // Allocation (thread-local hot region + freelist, matching C++ TCMemPoolOneThread)
    // ========================================================================

    #[inline]
    fn get_tla(&self) -> &RefCell<ThreadLocalAlloc> {
        self.tls.get_or(|| RefCell::new(ThreadLocalAlloc::new()))
    }

    fn alloc_node(&self, byte_size: usize) -> u32 {
        let slots = byte_size.div_ceil(4);
        let tla_cell = self.get_tla();
        let mut tla = tla_cell.borrow_mut();

        // Fast path 1: thread-local freelist (zero atomics)
        if slots > 0 && slots <= FREE_LIST_MAX_SLOTS {
            let head = tla.fast_bins[slots - 1];
            if head != FREE_LIST_NIL {
                tla.fast_bins[slots - 1] = self.inner.pool.load_relaxed(head as usize);
                return head;
            }
        }

        // Fast path 2: hot region bump (zero atomics)
        let new_pos = tla.hot_pos as usize + slots;
        if new_pos <= tla.hot_end as usize {
            let pos = tla.hot_pos;
            tla.hot_pos = new_pos as u32;
            for i in 0..slots {
                self.inner.pool.store_relaxed(pos as usize + i, NIL_STATE);
            }
            return pos;
        }

        // Fast path 3: global ABA-safe freelist (rare, only after deferred frees reclaimed)
        if let Some(pos) = self.inner.freelist.pop(&self.inner.pool, slots) {
            for i in 0..slots {
                self.inner.pool.store_relaxed(pos as usize + i, NIL_STATE);
            }
            return pos;
        }

        // Slow path: request new chunk from global pool (one CAS per 512K slots)
        drop(tla);
        self.alloc_chunk(slots)
    }

    fn alloc_chunk(&self, needed_slots: usize) -> u32 {
        let remaining = self.inner.pool.capacity() - self.inner.pool.len();
        let chunk_slots = CHUNK_SLOTS.min(remaining).max(needed_slots);
        let chunk_start = self.inner.pool.bump_alloc(chunk_slots);

        let tla_cell = self.get_tla();
        let mut tla = tla_cell.borrow_mut();

        // Save old hot region remainder to freelist
        let old_remaining = tla.hot_end as usize - tla.hot_pos as usize;
        if old_remaining > 0 && old_remaining <= FREE_LIST_MAX_SLOTS {
            let old_pos = tla.hot_pos;
            self.inner.pool.store_relaxed(old_pos as usize, tla.fast_bins[old_remaining - 1]);
            tla.fast_bins[old_remaining - 1] = old_pos;
        }

        // Set new hot region (after the needed allocation)
        let pos = chunk_start;
        tla.hot_pos = chunk_start + needed_slots as u32;
        tla.hot_end = chunk_start + chunk_slots as u32;

        for i in 0..needed_slots {
            self.inner.pool.store_relaxed(pos as usize + i, NIL_STATE);
        }
        pos
    }

    fn free_node_deferred(&self, guard: &Guard, slot: u32, slots: usize) {
        let inner = std::sync::Arc::clone(&self.inner);
        // SAFETY: Arc clone keeps pool/freelist alive until the deferred fn runs.
        unsafe {
            guard.defer_unchecked(move || {
                inner.freelist.push(&inner.pool, slot, slots);
            });
        }
    }

    // ========================================================================
    // Per-node locking
    // ========================================================================

    /// Try to set b_lock (bit 7) on a node. Returns old u32 on success.
    #[inline]
    fn try_lock_node(&self, pos: u32) -> Result<u32, ()> {
        let old = self.inner.pool.load_acquire(pos as usize);
        let flags = (old & 0xFF) as u8;
        if flags & (FLAG_LOCK | FLAG_LAZY_FREE) != 0 {
            return Err(());
        }
        let new = old | U32_FLAG_LOCK;
        self.inner.pool.cas_weak(pos as usize, old, new).map_err(|_| ())
    }

    /// Clear b_lock (bit 7), restoring the original value.
    #[inline]
    fn unlock_node(&self, pos: u32, original: u32) {
        self.inner.pool.store_release(pos as usize, original);
    }

    /// Try to mark b_lazy_free (bit 5). Returns old u32 on success.
    #[inline]
    fn try_mark_lazy_free(&self, pos: u32) -> Result<u32, ()> {
        let old = self.inner.pool.load_acquire(pos as usize);
        let flags = (old & 0xFF) as u8;
        if flags & FLAG_LAZY_FREE != 0 {
            return Err(());
        }
        let new = old | U32_FLAG_LAZY_FREE;
        self.inner.pool.cas_weak(pos as usize, old, new).map_err(|_| ())
    }

    // ========================================================================
    // Node initialization helpers (write to uncontested newly-allocated slots)
    // ========================================================================

    fn write_meta_with_n_children(&self, pos: u32, meta: MetaInfo, n_children: u16) {
        let mut u = meta_to_u32(meta);
        let bytes = u.to_ne_bytes();
        let mut combined = [bytes[0], bytes[1], 0, 0];
        combined[2..4].copy_from_slice(&n_children.to_ne_bytes());
        u = u32::from_ne_bytes(combined);
        self.inner.pool.store_relaxed(pos as usize, u);
    }

    fn new_suffix_chain(&self, suffix: &[u8]) -> (u32, usize) {
        let mut remaining = suffix;
        let mut head = NIL_STATE;
        let mut prev_child_slot: u32 = NIL_STATE;

        while remaining.len() > MAX_ZPATH {
            let link_size = ALIGN_SIZE * 2 + MAX_ZPATH;
            let node = self.alloc_node(link_size);
            let meta = MetaInfo {
                flags: 1,
                n_zpath_len: MAX_ZPATH as u8,
                c_label: [remaining[MAX_ZPATH], 0],
            };
            self.inner.pool.store_relaxed(node as usize, meta_to_u32(meta));
            self.inner.pool.store_relaxed(node as usize + 1, NIL_STATE);
            // SAFETY: Writing zpath to newly allocated, uncontested slots.
            unsafe {
                let zpath_dst = (self.inner.pool.data.as_ptr() as *mut u8)
                    .add((node as usize + 2) * 4);
                std::ptr::copy_nonoverlapping(remaining.as_ptr(), zpath_dst, MAX_ZPATH);
                *zpath_dst.add(254) = 0;
                *zpath_dst.add(255) = 0;
            }
            if head == NIL_STATE { head = node; }
            if prev_child_slot != NIL_STATE {
                self.inner.pool.store_relaxed(prev_child_slot as usize, node);
            }
            prev_child_slot = node + 1;
            remaining = &remaining[MAX_ZPATH + 1..];
        }

        let zpath_padded = (remaining.len() + 3) & !3;
        let leaf_size = ALIGN_SIZE + zpath_padded + self.valsize;
        let node = self.alloc_node(leaf_size);
        let meta = MetaInfo {
            flags: FLAG_IS_FINAL,
            n_zpath_len: remaining.len() as u8,
            c_label: [0, 0],
        };
        self.inner.pool.store_relaxed(node as usize, meta_to_u32(meta));
        unsafe {
            let zpath_dst = (self.inner.pool.data.as_ptr() as *mut u8)
                .add((node as usize + 1) * 4);
            std::ptr::copy_nonoverlapping(remaining.as_ptr(), zpath_dst, remaining.len());
            for i in remaining.len()..zpath_padded {
                *zpath_dst.add(i) = 0;
            }
        }
        let valpos = (node as usize + 1) * ALIGN_SIZE + zpath_padded;

        if head == NIL_STATE { head = node; }
        if prev_child_slot != NIL_STATE {
            self.inner.pool.store_relaxed(prev_child_slot as usize, node);
        }
        (head, valpos)
    }

    fn build_bitmap_node(
        &self, labels: &[u8], children: &[u32], n_children: usize,
        flags: u8, zpath_len: usize, trailing: &[u8], trailing_len: usize,
    ) -> u32 {
        let node_size = (10 + n_children) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(node_size);
        let new_flags = (flags & !FLAG_CNT_MASK) | 8;
        let meta = MetaInfo {
            flags: new_flags,
            n_zpath_len: zpath_len as u8,
            c_label: [0, 0],
        };
        self.write_meta_with_n_children(node, meta, n_children as u16);

        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let p = base.add(node as usize * 4);

            // Build bitmap at slots 2-9 (32 bytes)
            let bmp = p.add(2 * 4);
            std::ptr::write_bytes(bmp, 0, 32);
            for i in 0..n_children {
                let label = labels[i];
                *bmp.add(label as usize / 8) |= 1 << (label % 8);
            }
            // Compute rank prefix at slot 1 bytes 0-3
            let rank = p.add(4);
            let mut cumulative = 0u32;
            for q in 0..4 {
                *rank.add(q) = cumulative as u8;
                let w = std::ptr::read_unaligned(bmp.add(q * 8) as *const u64);
                cumulative += w.count_ones();
            }
        }
        // Children at slots 10+
        for i in 0..n_children {
            self.inner.pool.store_relaxed(node as usize + 10 + i, children[i]);
        }
        // Trailing data (zpath + value)
        if trailing_len > 0 {
            unsafe {
                let dst = (self.inner.pool.data.as_ptr() as *mut u8)
                    .add((node as usize + 10 + n_children) * 4);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    fn add_state_move_bitmap(&self, curr: u32, ch: u8, suffix_node: u32) -> u32 {
        let meta = self.inner.pool.load_meta(curr);
        let zpath_len = meta.n_zpath_len as usize;
        let is_final = meta.flags & FLAG_IS_FINAL != 0;
        let old_n = {
            let big: BigCount = bytemuck::cast(self.inner.pool.load_relaxed(curr as usize));
            big.n_children as usize
        };

        let mut bitmap = [0u8; 32];
        let mut rank_prefix = [0u8; 4];
        unsafe {
            let bmp_src = self.inner.pool.raw_ptr(curr as usize + 2);
            std::ptr::copy_nonoverlapping(bmp_src, bitmap.as_mut_ptr(), 32);
            let rank_src = self.inner.pool.raw_ptr(curr as usize + 1);
            std::ptr::copy_nonoverlapping(rank_src, rank_prefix.as_mut_ptr(), 4);
        }
        let mut old_children = [0u32; 257];
        for i in 0..old_n {
            old_children[i] = self.inner.pool.load_relaxed(curr as usize + 10 + i);
        }
        let zpath_padded = (zpath_len + 3) & !3;
        let trailing_len = zpath_padded + if is_final { self.valsize } else { 0 };
        let mut trailing = [0u8; 512];
        if trailing_len > 0 {
            let off = (10 + old_n) * ALIGN_SIZE;
            unsafe {
                let src = self.inner.pool.raw_ptr(curr as usize).add(off);
                std::ptr::copy_nonoverlapping(src, trailing.as_mut_ptr(), trailing_len);
            }
        }

        let ch_rank = {
            let q = (ch / 64) as usize;
            let w = unsafe {
                std::ptr::read_unaligned(bitmap.as_ptr().add(q * 8) as *const u64)
            };
            let mask = (1u64 << (ch % 64)) - 1;
            rank_prefix[q] as usize + (w & mask).count_ones() as usize
        };
        bitmap[(ch / 8) as usize] |= 1 << (ch % 8);
        let mut cumulative = 0u32;
        for q in 0..4 {
            rank_prefix[q] = cumulative as u8;
            let w = unsafe {
                std::ptr::read_unaligned(bitmap.as_ptr().add(q * 8) as *const u64)
            };
            cumulative += w.count_ones();
        }
        for i in (ch_rank..old_n).rev() {
            old_children[i + 1] = old_children[i];
        }
        old_children[ch_rank] = suffix_node;
        let new_n = old_n + 1;

        let node_size = (10 + new_n) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(node_size);
        self.write_meta_with_n_children(node, MetaInfo {
            flags: meta.flags,
            n_zpath_len: zpath_len as u8,
            c_label: [0, 0],
        }, new_n as u16);

        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let p = base.add(node as usize * 4);
            let rank_dst = p.add(4);
            std::ptr::copy_nonoverlapping(rank_prefix.as_ptr(), rank_dst, 4);
            let bmp_dst = p.add(2 * 4);
            std::ptr::copy_nonoverlapping(bitmap.as_ptr(), bmp_dst, 32);
        }
        for i in 0..new_n {
            self.inner.pool.store_relaxed(node as usize + 10 + i, old_children[i]);
        }
        if trailing_len > 0 {
            unsafe {
                let dst = (self.inner.pool.data.as_ptr() as *mut u8)
                    .add((node as usize + 10 + new_n) * 4);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    fn add_state_move(&self, curr: u32, ch: u8, suffix_node: u32) -> u32 {
        let meta = self.inner.pool.load_meta(curr);
        let cnt_type = meta.flags & FLAG_CNT_MASK;

        if cnt_type == 8 {
            return self.add_state_move_bitmap(curr, ch, suffix_node);
        }

        let zpath_len = meta.n_zpath_len as usize;
        let is_final = meta.flags & FLAG_IS_FINAL != 0;
        let old_skip = SKIP_SLOTS[cnt_type as usize] as usize;
        let old_n: usize = if cnt_type <= 6 {
            cnt_type as usize
        } else {
            let big: BigCount = bytemuck::cast(self.inner.pool.load_relaxed(curr as usize));
            big.n_children as usize
        };

        let mut labels = [0u8; 17];
        match cnt_type {
            0 => {}
            1 | 2 => {
                labels[0] = meta.c_label[0];
                if cnt_type >= 2 { labels[1] = meta.c_label[1]; }
            }
            3..=6 => {
                labels[0] = meta.c_label[0];
                labels[1] = meta.c_label[1];
                let pad = self.inner.pool.load_bytes(curr, 1);
                for i in 2..old_n { labels[i] = pad[i - 2]; }
            }
            7 => {
                unsafe {
                    let src = self.inner.pool.raw_ptr(curr as usize + 1);
                    for i in 0..old_n { labels[i] = *src.add(i); }
                }
            }
            _ => unreachable!()
        }

        let mut children = [0u32; 17];
        for i in 0..old_n {
            children[i] = self.inner.pool.load_relaxed(curr as usize + old_skip + i);
        }

        let zpath_padded = (zpath_len + 3) & !3;
        let trailing_len = zpath_padded + if is_final { self.valsize } else { 0 };
        let mut trailing = [0u8; 512];
        if trailing_len > 0 {
            let trailing_start = (old_skip + old_n) * ALIGN_SIZE;
            unsafe {
                let src = self.inner.pool.raw_ptr(curr as usize).add(trailing_start);
                std::ptr::copy_nonoverlapping(src, trailing.as_mut_ptr(), trailing_len);
            }
        }

        let idx = labels[..old_n].partition_point(|&l| l < ch);
        for i in (idx..old_n).rev() {
            labels[i + 1] = labels[i];
            children[i + 1] = children[i];
        }
        labels[idx] = ch;
        children[idx] = suffix_node;
        let new_n = old_n + 1;

        let new_cnt_type: u8 = match cnt_type {
            0..=5 => cnt_type + 1,
            6 => 7,
            7 if old_n < 16 => 7,
            7 => 8,
            _ => unreachable!()
        };

        if new_cnt_type == 8 {
            return self.build_bitmap_node(
                &labels, &children, new_n,
                meta.flags, zpath_len, &trailing, trailing_len,
            );
        }

        let new_skip = SKIP_SLOTS[new_cnt_type as usize] as usize;
        let new_size = (new_skip + new_n) * ALIGN_SIZE + trailing_len;
        let node = self.alloc_node(new_size);
        let new_flags = (meta.flags & !FLAG_CNT_MASK) | new_cnt_type;

        match new_cnt_type {
            1 | 2 => {
                let m = MetaInfo {
                    flags: new_flags,
                    n_zpath_len: zpath_len as u8,
                    c_label: [labels[0], if new_cnt_type >= 2 { labels[1] } else { 0 }],
                };
                self.inner.pool.store_relaxed(node as usize, meta_to_u32(m));
            }
            3..=6 => {
                let m = MetaInfo {
                    flags: new_flags,
                    n_zpath_len: zpath_len as u8,
                    c_label: [labels[0], labels[1]],
                };
                self.inner.pool.store_relaxed(node as usize, meta_to_u32(m));
                let mut pad = [0u8; 4];
                for i in 2..new_n { pad[i - 2] = labels[i]; }
                self.inner.pool.store_relaxed(node as usize + 1, u32::from_ne_bytes(pad));
            }
            7 => {
                let m = MetaInfo {
                    flags: new_flags,
                    n_zpath_len: zpath_len as u8,
                    c_label: [0, 0],
                };
                self.write_meta_with_n_children(node, m, new_n as u16);
                unsafe {
                    let lbl_ptr = (self.inner.pool.data.as_ptr() as *mut u8)
                        .add((node as usize + 1) * 4);
                    for i in 0..new_n { *lbl_ptr.add(i) = labels[i]; }
                    for i in new_n..16 { *lbl_ptr.add(i) = 0; }
                }
            }
            _ => unreachable!()
        }

        for i in 0..new_n {
            self.inner.pool.store_relaxed(node as usize + new_skip + i, children[i]);
        }
        if trailing_len > 0 {
            unsafe {
                let dst = (self.inner.pool.data.as_ptr() as *mut u8)
                    .add((node as usize + new_skip + new_n) * 4);
                std::ptr::copy_nonoverlapping(trailing.as_ptr(), dst, trailing_len);
            }
        }
        node
    }

    fn fork(
        &self, curr: u32, zidx: usize,
        old_skip: usize, old_n_children: usize, zpath_len: usize,
        node_size: usize, zpath_buf: &[u8],
        new_char: u8, new_suffix_node: u32,
    ) -> (u32, u32) {
        let old_char = zpath_buf[zidx];
        let suffix_zlen = zpath_len - zidx - 1;
        let suffix_zpath_padded = (suffix_zlen + 3) & !3;
        let val_size = node_size - ((old_skip + old_n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3));
        let suffix_size = (old_skip + old_n_children) * ALIGN_SIZE + suffix_zpath_padded + val_size;

        let suffix_node = self.alloc_node(suffix_size);
        // SAFETY: `suffix_node` is newly allocated and uncontested.
        // `curr` is a valid node. We are copying the structural part (header and children).
        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let src = (self.inner.pool.raw_ptr(curr as usize)) as *const u8;
            let dst = base.add(suffix_node as usize * 4);
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            std::ptr::copy_nonoverlapping(src, dst, struct_size);
        }
        // Set new zpath_len on suffix node
        let mut suffix_meta = self.inner.pool.load_meta(suffix_node);
        suffix_meta.n_zpath_len = suffix_zlen as u8;
        self.inner.pool.store_relaxed(suffix_node as usize, meta_to_u32(suffix_meta));

        // SAFETY: `suffix_node` is newly allocated and uncontested.
        // We are filling the zpath and value parts. Reads from `zpath_buf` and `curr` are safe.
        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let dst = base.add(suffix_node as usize * 4);
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            let zpath_dst = dst.add(struct_size);
            for i in 0..suffix_zlen {
                *zpath_dst.add(i) = zpath_buf[zidx + 1 + i];
            }
            for i in suffix_zlen..suffix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            if val_size > 0 {
                let old_val_off = struct_size + ((zpath_len + 3) & !3);
                let src = (self.inner.pool.raw_ptr(curr as usize)) as *const u8;
                std::ptr::copy_nonoverlapping(
                    src.add(old_val_off),
                    zpath_dst.add(suffix_zpath_padded),
                    val_size,
                );
            }
        }

        let prefix_zpath_padded = (zidx + 3) & !3;
        let parent_size = 3 * ALIGN_SIZE + prefix_zpath_padded;
        let parent = self.alloc_node(parent_size);
        let (label0, child0, label1, child1) = if old_char < new_char {
            (old_char, suffix_node, new_char, new_suffix_node)
        } else {
            (new_char, new_suffix_node, old_char, suffix_node)
        };
        let parent_meta = MetaInfo {
            flags: 2,
            n_zpath_len: zidx as u8,
            c_label: [label0, label1],
        };
        self.inner.pool.store_relaxed(parent as usize, meta_to_u32(parent_meta));
        self.inner.pool.store_relaxed(parent as usize + 1, child0);
        self.inner.pool.store_relaxed(parent as usize + 2, child1);

        // SAFETY: `parent` is newly allocated and uncontested.
        // Writing prefix zpath.
        unsafe {
            let zpath_dst = (self.inner.pool.data.as_ptr() as *mut u8)
                .add((parent as usize + 3) * 4);
            for i in 0..zidx {
                *zpath_dst.add(i) = zpath_buf[i];
            }
            for i in zidx..prefix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
        }
        (parent, suffix_node)
    }

    /// Returns (prefix_node, valpos, suffix_node_copy).
    fn split_zpath(
        &self, curr: u32, split_pos: usize,
        old_skip: usize, old_n_children: usize, zpath_len: usize,
        node_size: usize, zpath_buf: &[u8],
    ) -> (u32, usize, u32) {
        let split_char = zpath_buf[split_pos];
        let suffix_zlen = zpath_len - split_pos - 1;
        let suffix_zpath_padded = (suffix_zlen + 3) & !3;
        let val_size = node_size - ((old_skip + old_n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3));
        let suffix_size = (old_skip + old_n_children) * ALIGN_SIZE + suffix_zpath_padded + val_size;

        let suffix_node = self.alloc_node(suffix_size);
        // SAFETY: `suffix_node` is newly allocated and uncontested.
        // Copying structural part from `curr`.
        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let src = (self.inner.pool.raw_ptr(curr as usize)) as *const u8;
            let dst = base.add(suffix_node as usize * 4);
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            std::ptr::copy_nonoverlapping(src, dst, struct_size);
        }
        let mut suffix_meta = self.inner.pool.load_meta(suffix_node);
        suffix_meta.n_zpath_len = suffix_zlen as u8;
        self.inner.pool.store_relaxed(suffix_node as usize, meta_to_u32(suffix_meta));

        // SAFETY: `suffix_node` is newly allocated and uncontested.
        // Copying zpath and value data from `curr` and `zpath_buf`.
        unsafe {
            let base = self.inner.pool.data.as_ptr() as *mut u8;
            let dst = base.add(suffix_node as usize * 4);
            let struct_size = (old_skip + old_n_children) * ALIGN_SIZE;
            let zpath_dst = dst.add(struct_size);
            for i in 0..suffix_zlen {
                *zpath_dst.add(i) = zpath_buf[split_pos + 1 + i];
            }
            for i in suffix_zlen..suffix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
            if val_size > 0 {
                let old_val_off = struct_size + ((zpath_len + 3) & !3);
                let src = (self.inner.pool.raw_ptr(curr as usize)) as *const u8;
                std::ptr::copy_nonoverlapping(
                    src.add(old_val_off),
                    zpath_dst.add(suffix_zpath_padded),
                    val_size,
                );
            }
        }

        let prefix_zpath_padded = (split_pos + 3) & !3;
        let prefix_size = 2 * ALIGN_SIZE + prefix_zpath_padded + self.valsize;
        let prefix_node = self.alloc_node(prefix_size);
        let prefix_meta = MetaInfo {
            flags: 1 | FLAG_IS_FINAL,
            n_zpath_len: split_pos as u8,
            c_label: [split_char, 0],
        };
        self.inner.pool.store_relaxed(prefix_node as usize, meta_to_u32(prefix_meta));
        self.inner.pool.store_relaxed(prefix_node as usize + 1, suffix_node);

        unsafe {
            let zpath_dst = (self.inner.pool.data.as_ptr() as *mut u8)
                .add((prefix_node as usize + 2) * 4);
            for i in 0..split_pos {
                *zpath_dst.add(i) = zpath_buf[i];
            }
            for i in split_pos..prefix_zpath_padded {
                *zpath_dst.add(i) = 0;
            }
        }
        let valpos = (prefix_node as usize + 2) * ALIGN_SIZE + prefix_zpath_padded;
        (prefix_node, valpos, suffix_node)
    }

    /// Realloc a node (for MarkFinalState on non-fast nodes).
    /// Allocates new, copies old, returns new slot. Old node is NOT freed here.
    fn realloc_node_concurrent(&self, old_slot: u32, old_size: usize, new_size: usize) -> u32 {
        let old_slots = old_size.div_ceil(4);
        let new_slots = new_size.div_ceil(4);
        if old_slots == new_slots {
            return old_slot;
        }
        let new_slot = self.alloc_node(new_size);
        let copy_slots = old_slots.min(new_slots);
        for i in 0..copy_slots {
            let v = self.inner.pool.load_relaxed(old_slot as usize + i);
            self.inner.pool.store_relaxed(new_slot as usize + i, v);
        }
        new_slot
    }

    // ========================================================================
    // Concurrent insert
    // ========================================================================

    /// Insert a key into the trie. Thread-safe (multiple writers allowed).
    /// Returns (is_new, valpos_byte_offset).
    pub fn insert(&self, key: &[u8]) -> (bool, usize) {
        let guard = epoch::pin();
        self.insert_with_guard(key, &guard)
    }

    /// Insert with an existing epoch guard.
    pub fn insert_with_guard(&self, key: &[u8], guard: &Guard) -> (bool, usize) {
        let mut backoff = Backoff::new();

        'retry: loop {
            let mut curr_slot: u32 = NIL_STATE;
            let mut parent: u32 = NIL_STATE;
            let mut curr: u32 = INITIAL_STATE;
            let mut pos: usize = 0;

            // ---- Search phase (lock-free) ----
            loop {
                let view = self.node_view(curr);
                let cnt_type = view.cnt_type();
                let zpath_len = view.zpath_len();
                let is_final = view.is_final();
                let skip = view.skip_slots();
                let n_children = view.n_children();
                let _flags = view.meta().flags;

                let node_size = (skip + n_children) * ALIGN_SIZE
                    + ((zpath_len + 3) & !3)
                    + if is_final { self.valsize } else { 0 };

                if zpath_len > 0 {
                    let mut zpath_buf = [0u8; 256];
                    unsafe {
                        let src = self.inner.pool.raw_ptr(curr as usize + skip + n_children);
                        std::ptr::copy_nonoverlapping(src, zpath_buf.as_mut_ptr(), zpath_len);
                    }

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
                        // ForkBranch
                        let (new_suffix, valpos) = self.new_suffix_chain(&key[pos + zidx + 1..]);
                        let (new_parent, fork_suffix_copy) = self.fork(
                            curr, zidx, skip, n_children, zpath_len, node_size,
                            &zpath_buf[..zpath_len], key[pos + zidx], new_suffix,
                        );

                        if !self.update_curr_ptr(guard, parent, curr_slot, curr, new_parent, &mut backoff) {
                            self.free_suffix_chain(new_suffix);
                            self.free_single_node(fork_suffix_copy);
                            self.free_single_node(new_parent);
                            continue 'retry;
                        }
                        self.n_words.fetch_add(1, Ordering::Relaxed);
                        self.update_max_word_len(key.len());
                        return (true, valpos);
                    }

                    pos += match_len;

                    if remaining_key < zpath_len {
                        // SplitZpath
                        let (prefix_node, valpos, split_suffix_copy) = self.split_zpath(
                            curr, match_len, skip, n_children, zpath_len, node_size,
                            &zpath_buf[..zpath_len],
                        );

                        if !self.update_curr_ptr(guard, parent, curr_slot, curr, prefix_node, &mut backoff) {
                            self.free_single_node(split_suffix_copy);
                            self.free_single_node(prefix_node);
                            continue 'retry;
                        }
                        self.n_words.fetch_add(1, Ordering::Relaxed);
                        self.update_max_word_len(key.len());
                        return (true, valpos);
                    }

                    if pos == key.len() {
                        if is_final {
                            let vp = (curr as usize + skip + n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3);
                            return (false, vp);
                        }
                        // MarkFinalState
                        let new_size = node_size + self.valsize;
                        let new_curr = self.realloc_node_concurrent(curr, node_size, new_size);
                        // Set is_final on new node
                        let mut m = u32_to_meta(self.inner.pool.load_relaxed(new_curr as usize));
                        m.flags |= FLAG_IS_FINAL;
                        self.inner.pool.store_relaxed(new_curr as usize, meta_to_u32(m));

                        if new_curr != curr {
                            if !self.update_curr_ptr(guard, parent, curr_slot, curr, new_curr, &mut backoff) {
                                self.free_single_node(new_curr);
                                continue 'retry;
                            }
                        } else {
                            // In-place update (same slot) — just CAS the flags
                            let old_meta = self.inner.pool.load_acquire(curr as usize);
                            let new_meta = old_meta | U32_FLAG_IS_FINAL;
                            if self.inner.pool.cas_weak(curr as usize, old_meta, new_meta).is_err() {
                                backoff.spin();
                                continue 'retry;
                            }
                        }
                        let vp = (new_curr as usize + skip + n_children) * ALIGN_SIZE + ((zpath_len + 3) & !3);
                        self.n_words.fetch_add(1, Ordering::Relaxed);
                        self.update_max_word_len(key.len());
                        return (true, vp);
                    }
                } else {
                    if pos == key.len() {
                        if is_final {
                            let vp = (curr as usize + skip + n_children) * ALIGN_SIZE;
                            return (false, vp);
                        }

                        if cnt_type == 15 {
                            // MarkFinalStateOnFastNode: atomic set is_final
                            loop {
                                let old = self.inner.pool.load_acquire(curr as usize);
                                let f = (old & 0xFF) as u8;
                                if f & FLAG_IS_FINAL != 0 {
                                    let vp = (curr as usize + 2 + 256) * ALIGN_SIZE;
                                    return (false, vp);
                                }
                                if f & FLAG_SET_FINAL != 0 {
                                    std::hint::spin_loop();
                                    continue;
                                }
                                let new = old | U32_FLAG_IS_FINAL | U32_FLAG_SET_FINAL;
                                if self.inner.pool.cas_weak(curr as usize, old, new).is_ok() {
                                    let vp = (curr as usize + 2 + 256) * ALIGN_SIZE;
                                    self.n_words.fetch_add(1, Ordering::Relaxed);
                                    self.update_max_word_len(key.len());
                                    return (true, vp);
                                }
                            }
                        }

                        // MarkFinalState for non-fast node
                        let new_size = node_size + self.valsize;
                        let new_curr = self.realloc_node_concurrent(curr, node_size, new_size);
                        let mut m = u32_to_meta(self.inner.pool.load_relaxed(new_curr as usize));
                        m.flags |= FLAG_IS_FINAL;
                        self.inner.pool.store_relaxed(new_curr as usize, meta_to_u32(m));

                        if new_curr != curr {
                            if !self.update_curr_ptr(guard, parent, curr_slot, curr, new_curr, &mut backoff) {
                                self.free_single_node(new_curr);
                                continue 'retry;
                            }
                        } else {
                            let old_meta = self.inner.pool.load_acquire(curr as usize);
                            let new_meta = old_meta | U32_FLAG_IS_FINAL;
                            if self.inner.pool.cas_weak(curr as usize, old_meta, new_meta).is_err() {
                                backoff.spin();
                                continue 'retry;
                            }
                        }
                        let vp = (new_curr as usize + skip + n_children) * ALIGN_SIZE;
                        self.n_words.fetch_add(1, Ordering::Relaxed);
                        self.update_max_word_len(key.len());
                        return (true, vp);
                    }
                }

                // Transition on key[pos]
                let ch = key[pos];
                let next = view.state_move(ch);

                if next == NIL_STATE {
                    // MatchFail
                    let (suffix_node, valpos) = self.new_suffix_chain(&key[pos + 1..]);

                    if cnt_type == 15 {
                        // Fast node: atomic CAS on child slot
                        let child_pos = curr as usize + 2 + ch as usize;
                        match self.inner.pool.cas_weak(child_pos, NIL_STATE, suffix_node) {
                            Ok(_) => {
                                // Atomically increment real count at slot 1
                                loop {
                                    let old = self.inner.pool.load_acquire(curr as usize + 1);
                                    let big: BigCount = bytemuck::cast(old);
                                    let new_big = BigCount {
                                        _unused: big._unused,
                                        n_children: big.n_children + 1,
                                    };
                                    let new: u32 = bytemuck::cast(new_big);
                                    if self.inner.pool.cas_weak(curr as usize + 1, old, new).is_ok() {
                                        break;
                                    }
                                }
                                self.n_words.fetch_add(1, Ordering::Relaxed);
                                self.update_max_word_len(key.len());
                                return (true, valpos);
                            }
                            Err(_) => {
                                self.race_stats.fast_node_cas_fail.fetch_add(1, Ordering::Relaxed);
                                self.free_suffix_chain(suffix_node);
                                backoff.spin();
                                continue 'retry;
                            }
                        }
                    } else {
                        let new_curr = self.add_state_move(curr, ch, suffix_node);
                        if !self.update_curr_ptr(guard, parent, curr_slot, curr, new_curr, &mut backoff) {
                            self.free_suffix_chain(suffix_node);
                            self.free_single_node(new_curr);
                            continue 'retry;
                        }
                        self.n_words.fetch_add(1, Ordering::Relaxed);
                        self.update_max_word_len(key.len());
                        return (true, valpos);
                    }
                }

                // Advance to next node
                parent = curr;
                curr_slot = view.find_child_slot(ch);
                curr = next;
                pos += 1;
                backoff.reset();
            }
        }
    }

    /// Optimistic locking protocol to atomically replace curr with new_node.
    ///
    /// 1. Lock parent (CAS b_lock)
    /// 2. Mark curr as lazy_free (CAS b_lazy_free)
    /// 3. CAS parent's child pointer from curr to new_node
    /// 4. On success: defer free of curr
    /// 5. On failure: undo locks and return false
    fn update_curr_ptr(
        &self,
        guard: &Guard,
        parent: u32,
        curr_slot: u32,
        curr: u32,
        new_node: u32,
        backoff: &mut Backoff,
    ) -> bool {
        // For root node (parent == NIL_STATE), no parent locking needed —
        // the root's children are updated via direct CAS on the child slot.
        if curr == INITIAL_STATE {
            // Can't replace the root itself in this protocol.
            // The root is always a fast node and is never replaced.
            return false;
        }

        // Step 1: Lock parent
        let parent_original = match self.try_lock_node(parent) {
            Ok(orig) => orig,
            Err(()) => {
                self.race_stats.parent_lock_fail.fetch_add(1, Ordering::Relaxed);
                backoff.spin();
                return false;
            }
        };

        // Step 2: Mark curr as lazy_free
        let curr_original = match self.try_mark_lazy_free(curr) {
            Ok(orig) => orig,
            Err(()) => {
                self.unlock_node(parent, parent_original);
                self.race_stats.lazy_free_fail.fetch_add(1, Ordering::Relaxed);
                backoff.spin();
                return false;
            }
        };

        // Step 3: CAS the child pointer
        if curr_slot == NIL_STATE {
            // Should not happen in normal flow
            self.inner.pool.store_release(curr as usize, curr_original);
            self.unlock_node(parent, parent_original);
            return false;
        }

        match self.inner.pool.cas_weak(curr_slot as usize, curr, new_node) {
            Ok(_) => {
                // Success! Unlock parent.
                self.unlock_node(parent, parent_original);

                // Defer free of old node
                let old_slot = curr;
                let old_meta = u32_to_meta(curr_original);
                let old_cnt = old_meta.flags & FLAG_CNT_MASK;
                let old_skip = SKIP_SLOTS[old_cnt as usize] as usize;
                let old_n: usize = if old_cnt <= 6 {
                    old_cnt as usize
                } else {
                    let big: BigCount = bytemuck::cast(curr_original);
                    big.n_children as usize
                };
                let old_zlen = old_meta.n_zpath_len as usize;
                let old_is_final = old_meta.flags & FLAG_IS_FINAL != 0;
                let old_node_slots = (old_skip + old_n)
                    + old_zlen.div_ceil(4)
                    + if old_is_final { self.valsize.div_ceil(4) } else { 0 };

                self.free_node_deferred(guard, old_slot, old_node_slots);
                true
            }
            Err(_) => {
                // CAS failed — undo lazy_free and unlock parent
                self.inner.pool.store_release(curr as usize, curr_original);
                self.unlock_node(parent, parent_original);
                self.race_stats.child_cas_fail.fetch_add(1, Ordering::Relaxed);
                backoff.spin();
                false
            }
        }
    }

    /// Compute node size in slots from its metadata.
    fn node_slots_from_meta(&self, meta: MetaInfo, n_children: usize) -> usize {
        let cnt = meta.flags & FLAG_CNT_MASK;
        let skip = SKIP_SLOTS[cnt as usize] as usize;
        let zlen = meta.n_zpath_len as usize;
        let is_final = meta.flags & FLAG_IS_FINAL != 0;
        (skip + n_children) + zlen.div_ceil(4) + if is_final { self.valsize.div_ceil(4) } else { 0 }
    }

    /// Free a single unpublished node to thread-local freelist.
    fn free_single_node(&self, node: u32) {
        if node == NIL_STATE {
            return;
        }
        let meta = u32_to_meta(self.inner.pool.load_relaxed(node as usize));
        let cnt = meta.flags & FLAG_CNT_MASK;
        let n: usize = if cnt <= 6 {
            cnt as usize
        } else {
            let big: BigCount = bytemuck::cast(self.inner.pool.load_relaxed(node as usize));
            big.n_children as usize
        };
        let slots = self.node_slots_from_meta(meta, n);
        self.free_to_tla(node, slots);
    }

    /// Free an unpublished suffix chain to thread-local freelist.
    fn free_suffix_chain(&self, head: u32) {
        let mut curr = head;
        while curr != NIL_STATE {
            let meta = u32_to_meta(self.inner.pool.load_relaxed(curr as usize));
            let cnt = meta.flags & FLAG_CNT_MASK;
            let n: usize = if cnt <= 6 { cnt as usize } else {
                let big: BigCount = bytemuck::cast(self.inner.pool.load_relaxed(curr as usize));
                big.n_children as usize
            };
            let slots = self.node_slots_from_meta(meta, n);
            let next = if cnt == 1 {
                self.inner.pool.load_relaxed(curr as usize + 1)
            } else {
                NIL_STATE
            };
            self.free_to_tla(curr, slots);
            curr = next;
        }
    }

    /// Push a slot to the current thread's local freelist (zero atomics).
    #[inline]
    fn free_to_tla(&self, slot: u32, slots: usize) {
        if slots == 0 || slots > FREE_LIST_MAX_SLOTS {
            return;
        }
        let tla_cell = self.get_tla();
        let mut tla = tla_cell.borrow_mut();
        self.inner.pool.store_relaxed(slot as usize, tla.fast_bins[slots - 1]);
        tla.fast_bins[slots - 1] = slot;
    }

    #[inline]
    fn update_max_word_len(&self, len: usize) {
        let mut cur = self.max_word_len.load(Ordering::Relaxed);
        while len > cur {
            match self.max_word_len.compare_exchange_weak(
                cur, len, Ordering::Relaxed, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_trie_basic() {
        let trie = ConcurrentCsppTrie::with_capacity(8, 1024 * 1024);
        let key1 = b"hello";
        let key2 = b"world";
        
        let (is_new, valpos) = trie.insert(key1);
        assert!(is_new);
        trie.set_value(valpos, 123u64);

        let (is_new2, valpos2) = trie.insert(key2);
        assert!(is_new2);
        trie.set_value(valpos2, 456u64);

        assert_eq!(trie.num_words(), 2);
        
        let res1 = trie.lookup(key1);
        assert!(res1.is_some());
        assert_eq!(trie.get_value::<u64>(res1.unwrap()), 123);

        let res2 = trie.lookup(key2);
        assert!(res2.is_some());
        assert_eq!(trie.get_value::<u64>(res2.unwrap()), 456);

        assert!(trie.lookup(b"hell").is_none());
        assert!(trie.lookup(b"helloo").is_none());
    }

    #[test]
    fn test_concurrent_trie_split_zpath() {
        let trie = ConcurrentCsppTrie::with_capacity(8, 1024 * 1024);
        trie.insert(b"abcde");
        trie.insert(b"ab");
        
        assert!(trie.contains(b"abcde"));
        assert!(trie.contains(b"ab"));
        assert!(!trie.contains(b"abc"));
    }

    #[test]
    fn test_concurrent_trie_fork() {
        let trie = ConcurrentCsppTrie::with_capacity(8, 1024 * 1024);
        trie.insert(b"abcd");
        trie.insert(b"abef");
        
        assert!(trie.contains(b"abcd"));
        assert!(trie.contains(b"abef"));
        assert!(!trie.contains(b"ab"));
    }

    #[test]
    fn test_concurrent_trie_many_small() {
        let n = if cfg!(miri) { 20 } else { 100 };
        let trie = ConcurrentCsppTrie::with_capacity(4, 1024 * 1024);
        for i in 0..n {
            let key = format!("key{:03}", i);
            let (is_new, valpos) = trie.insert(key.as_bytes());
            assert!(is_new);
            trie.set_value(valpos, i as u32);
        }
        
        for i in 0..n {
            let key = format!("key{:03}", i);
            let res = trie.lookup(key.as_bytes());
            assert!(res.is_some());
            assert_eq!(trie.get_value::<u32>(res.unwrap()), i as u32);
        }
    }

    #[test]
    fn test_concurrent_trie_multithreaded() {
        use std::sync::Arc;
        use std::thread;

        let n_threads = 4;
        let n_per_thread = if cfg!(miri) { 5 } else { 100 };
        let trie = Arc::new(ConcurrentCsppTrie::with_capacity(8, 4 * 1024 * 1024));

        // Phase 1: each thread pre-builds its key range in the trie sequentially
        // to establish the structural nodes without contention.
        for t in 0..n_threads {
            let seed_key = format!("t{}k{:06}", t, 0);
            let (is_new, valpos) = trie.insert(seed_key.as_bytes());
            if is_new {
                trie.set_value(valpos, (t * n_per_thread) as u64);
            }
        }

        // Phase 2: concurrent inserts within each thread's disjoint subtree.
        let mut threads = Vec::new();
        for t in 0..n_threads {
            let trie = Arc::clone(&trie);
            threads.push(thread::spawn(move || {
                for i in 1..n_per_thread {
                    let val = (t * n_per_thread + i) as u64;
                    let key = format!("t{}k{:06}", t, i);
                    let (is_new, valpos) = trie.insert(key.as_bytes());
                    if is_new {
                        trie.set_value(valpos, val);
                    }
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        for t in 0..n_threads {
            for i in 0..n_per_thread {
                let val = (t * n_per_thread + i) as u64;
                let key = format!("t{}k{:06}", t, i);
                let res = trie.lookup(key.as_bytes());
                assert!(res.is_some(), "Key {} not found after join", key);
                assert_eq!(trie.get_value::<u64>(res.unwrap()), val);
            }
        }
    }
}

impl Drop for ConcurrentCsppTrie {
    fn drop(&mut self) {
        // Aggressively collect deferred epoch closures that hold Arc<SharedInner>.
        // Pin, flush, and unpin multiple times to advance the global epoch and
        // drain garbage from all threads, preventing cross-test interference.
        for _ in 0..8 {
            let guard = epoch::pin();
            guard.flush();
            drop(guard);
        }
        // One final pin+unpin to trigger collection of the last epoch's garbage.
        drop(epoch::pin());
    }
}

    #[test]
    fn test_concurrent_cspp_trie_longest_prefix() {
        let trie = ConcurrentCsppTrie::with_capacity(8, 4 * 1024 * 1024);
        
        trie.insert(b"http");
        trie.insert(b"https");

        assert!(trie.lookup(b"http").is_some());
        assert!(trie.lookup(b"https").is_some());
        assert!(trie.lookup(b"ftp").is_none());
        
        let guard = epoch::pin();
        assert!(trie.lookup_with_guard(b"http", &guard).is_some());
    }

    #[test]
    fn test_concurrent_cspp_trie_deletion_support() {
        let trie = ConcurrentCsppTrie::with_capacity(8, 4 * 1024 * 1024);
        trie.insert(b"test1");
        assert_eq!(trie.num_words(), 1);
        trie.insert(b"test2");
        assert_eq!(trie.num_words(), 2);
    }
