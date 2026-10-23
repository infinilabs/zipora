//! Lock-free memory pool implementation for high-performance concurrent allocation
//!
//! This module provides a lock-free memory pool that uses atomic operations and
//! CAS (Compare-And-Swap) loops for thread-safe allocation without locks.
//!
//! # Architecture
//!
//! - **Fast Bins**: Small/medium allocations use atomic lock-free stacks
//! - **Skip List**: Large allocations use probabilistic skip list with minimal locking  
//! - **Offset Addressing**: Uses 32-bit offsets instead of 64-bit pointers for efficiency
//! - **False Sharing Prevention**: Cache-line alignment to prevent performance degradation
//!
//! # Performance Characteristics
//!
//! - **Lock-free hot path**: Most allocations avoid locks entirely
//! - **CAS retry loops**: Atomic operations with exponential backoff
//! - **Cache efficiency**: Offset-based addressing improves cache utilization
//! - **Concurrent throughput**: Scales with number of CPU cores

use super::CachePadded;
use crate::error::{Result, ZiporaError};
use crate::memory::cache_layout::{CacheLayoutConfig, CacheOptimizedAllocator, PrefetchHint};
use crate::memory::simd_ops::{fast_fill, fast_prefetch};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread;
use std::time::Duration;

/// Alignment for lock-free operations (4 or 8 bytes)
const ALIGN_SIZE: usize = 8;

// H13: the free-list next-pointers are read/written by type-punning block
// memory to `AtomicU32` (see `offset_to_ptr` call sites). That pun is only
// sound while every block offset is aligned to at least 4 bytes.
const _: () = assert!(
    ALIGN_SIZE >= 4 && ALIGN_SIZE.is_power_of_two(),
    "AtomicU32 free-list type-pun requires ALIGN_SIZE >= 4 (power of two)"
);

/// Number of fast bins for small/medium allocations
const FAST_BIN_COUNT: usize = 64;
/// Threshold for fast bin vs skip list (above this uses skip list)
const FAST_BIN_THRESHOLD: usize = 8192;
/// Sentinel value for empty lists
const LIST_TAIL: u32 = 0;
/// Bytes reserved in front of every block's user memory. The fast-bin free-list
/// link lives here, never in bytes a caller may own: a Treiber pop must read the
/// head block's link while a competing thread may already have handed that block
/// to user code, and a link stored inline would make that read race the user's
/// writes (a data race in the memory model, and a ThreadSanitizer report from
/// `test_concurrent_alloc_write_free_same_class`). The header is only ever
/// accessed as an `AtomicU32`, so the remaining reader/pusher overlap is
/// atomic-vs-atomic and well defined. Costs 8 bytes per block.
const BLOCK_HEADER: usize = ALIGN_SIZE;

/// Backoff schedule cap for the unbudgeted fast-bin push loop. The loop is
/// unbounded (see `deallocate_to_fast_bin`), so the retry counter it feeds to
/// `backoff` has to stop growing somewhere: a linear schedule would otherwise
/// sleep for longer and longer with every failure.
const MAX_BACKOFF_RETRY: u32 = 16;

/// Size classes for fast bins (similar to jemalloc)
const FAST_BIN_SIZES: &[usize] = &[
    8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208,
    224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024,
    1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
    4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192,
];

/// Compile-time lookup table mapping `size.div_ceil(ALIGN_SIZE)` (`0..=1024`) to its fast bin index.
const SIZE_CLASS_LUT_LEN: usize = (FAST_BIN_THRESHOLD / ALIGN_SIZE) + 1;

const SIZE_CLASS_LUT: [u8; SIZE_CLASS_LUT_LEN] = {
    let mut lut = [0u8; SIZE_CLASS_LUT_LEN];
    let mut idx = 0usize;
    while idx < SIZE_CLASS_LUT_LEN {
        let size = idx * ALIGN_SIZE;
        let mut bin = 0usize;
        while bin < FAST_BIN_COUNT {
            if size <= FAST_BIN_SIZES[bin] {
                lut[idx] = bin as u8;
                break;
            }
            bin += 1;
        }
        idx += 1;
    }
    lut
};

/// Per-thread magazine capacity per fast bin (matching tcache small-bin batching).
const MAGAZINE_CAP: usize = 16;
/// Magic prefix stored in `[user - 4 .. user]` while a fast-bin block is live.
const FAST_BIN_LIVE_TAG: u32 = 0xA110_C800;
/// Magic prefix stored in `[user - 4 .. user]` while a fast-bin block is free.
const FAST_BIN_FREE_TAG: u32 = 0xF2EE_0800;

static NEXT_LOCKFREE_POOL_ID: AtomicU64 = AtomicU64::new(1);

#[repr(align(64))]
struct ThreadMagazine {
    pool_id: u64,
    is_dirty: AtomicBool,
    non_empty_bins: AtomicU64,
    slots: [[AtomicU32; MAGAZINE_CAP]; FAST_BIN_COUNT],
}

impl ThreadMagazine {
    fn new(pool_id: u64) -> Self {
        Self {
            pool_id,
            is_dirty: AtomicBool::new(false),
            non_empty_bins: AtomicU64::new(0),
            slots: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU32::new(0))),
        }
    }

    fn drain_into_arena(&self, arena: &PoolArena) -> usize {
        if self.pool_id != arena.pool_id {
            return 0;
        }
        // ORDERING: AcqRel atomically claims the dirty state of this magazine so
        // `arena.dirty_magazines` is decremented at most once per dirty transition.
        if self.is_dirty.swap(false, Ordering::AcqRel) {
            // ORDERING: Release ensures slot drains are ordered with the dirty-count update.
            arena.dirty_magazines.fetch_sub(1, Ordering::Release);
        }
        // ORDERING: AcqRel atomically takes the bitmask of bins that have had items pushed.
        let mut bins_mask = self.non_empty_bins.swap(0, Ordering::AcqRel);
        let mut drained = 0usize;
        while bins_mask != 0 {
            let bin = bins_mask.trailing_zeros() as usize;
            bins_mask &= bins_mask - 1;
            for slot in &self.slots[bin] {
                // ORDERING: Relaxed pre-check avoids unnecessary RMW on empty slots;
                // AcqRel on swap synchronizes-with the owning thread's AcqRel swap in `push`
                // and prevents double-claim against concurrent owner `pop` (`S49-R2`).
                if slot.load(Ordering::Relaxed) != 0 {
                    let off = slot.swap(0, Ordering::AcqRel);
                    if off != 0 && arena.push_free_fast_bin_offset(bin, off).is_ok() {
                        drained += 1;
                    }
                }
            }
        }
        drained
    }

    fn clear_all(&self) {
        // ORDERING: Relaxed is sufficient when clearing dead pool offsets after arena drop.
        self.is_dirty.store(false, Ordering::Relaxed);
        // ORDERING: AcqRel takes the bitmask of non-empty bins to clear.
        let mut bins_mask = self.non_empty_bins.swap(0, Ordering::AcqRel);
        while bins_mask != 0 {
            let bin = bins_mask.trailing_zeros() as usize;
            bins_mask &= bins_mask - 1;
            for slot in &self.slots[bin] {
                // ORDERING: Relaxed is sufficient when clearing dead pool offsets.
                if slot.load(Ordering::Relaxed) != 0 {
                    slot.store(0, Ordering::Relaxed);
                }
            }
        }
    }
}

struct MagazineState {
    pool_id: u64,
    allocated_mask: u64,
    counts: [u8; FAST_BIN_COUNT],
    shared: *const ThreadMagazine,
}

impl MagazineState {
    const fn new() -> Self {
        Self {
            pool_id: 0,
            allocated_mask: 0,
            counts: [0; FAST_BIN_COUNT],
            shared: std::ptr::null(),
        }
    }

    #[cold]
    #[inline(never)]
    fn flush_to_current_owner(&mut self) {
        self.pool_id = 0;
        self.allocated_mask = 0;
        self.counts = [0; FAST_BIN_COUNT];
        self.shared = std::ptr::null();
        let _ = MAGAZINE_OWNER_GUARD.try_with(|guard| {
            let mut g = guard.borrow_mut();
            let prev_owner = g.owner.take();
            if let Some(shared) = g.shared.take() {
                if let Some(weak) = prev_owner
                    && let Some(arena) = weak.upgrade()
                {
                    shared.drain_into_arena(&arena);
                    arena.unregister_magazine(&shared);
                } else {
                    shared.clear_all();
                }
            }
        });
    }

    #[cold]
    #[inline(never)]
    fn switch_pool(&mut self, pool_id: u64, arena: &Arc<PoolArena>, bin_index: usize) {
        if self.pool_id != 0 {
            self.flush_to_current_owner();
        }
        if let Ok(shared_ptr) = MAGAZINE_OWNER_GUARD.try_with(|guard| {
            let mut g = guard.borrow_mut();
            if let Some(old_shared) = g.shared.take() {
                if let Some(old_weak) = g.owner.take()
                    && let Some(old_arena) = old_weak.upgrade()
                {
                    old_shared.drain_into_arena(&old_arena);
                    old_arena.unregister_magazine(&old_shared);
                } else {
                    old_shared.clear_all();
                }
            }
            let shared = Arc::new(ThreadMagazine::new(pool_id));
            arena.register_magazine(&shared);
            let ptr = Arc::as_ptr(&shared);
            g.owner = Some(Arc::downgrade(arena));
            g.shared = Some(shared);
            ptr
        }) {
            self.shared = shared_ptr;
            self.pool_id = pool_id;
            self.allocated_mask = 1u64 << (bin_index & (FAST_BIN_COUNT - 1));
        }
    }
}

struct FastBinMagazine {
    state: std::cell::UnsafeCell<MagazineState>,
}

const _: () = assert!(
    !std::mem::needs_drop::<FastBinMagazine>(),
    "FastBinMagazine must not have Drop glue so thread_local .with() never panics during teardown"
);

impl FastBinMagazine {
    const fn new() -> Self {
        Self {
            state: std::cell::UnsafeCell::new(MagazineState::new()),
        }
    }

    #[inline(always)]
    fn pop(&self, pool_id: u64, bin_index: usize) -> Option<u32> {
        // SAFETY: `FastBinMagazine` is stored in `thread_local!` (`!Sync`), and
        // `pop` is a leaf function that never re-enters `FAST_BIN_MAGAZINE`.
        let state = unsafe { &mut *self.state.get() };
        if state.pool_id != pool_id {
            return None;
        }
        let bin = bin_index & (FAST_BIN_COUNT - 1);
        let mut c = state.counts[bin] as usize;
        if c == 0 {
            return None;
        }
        // SAFETY: `state.pool_id == pool_id != 0`, which holds only while `MAGAZINE_OWNER_GUARD`
        // is alive on this thread and owns the `Arc<ThreadMagazine>` pointed to by `state.shared`.
        let shared = unsafe { &*state.shared };
        while c > 0 {
            c -= 1;
            // ORDERING: AcqRel atomically claims the slot and zeroes it in a single RMW
            // operation so a concurrent `drain_all_magazines` stealer can never claim the
            // same offset (`S49-R2`).
            let off = shared.slots[bin][c & (MAGAZINE_CAP - 1)].swap(0, Ordering::AcqRel);
            if off != 0 {
                state.counts[bin] = c as u8;
                return Some(off);
            }
        }
        state.counts[bin] = 0;
        None
    }

    #[inline(always)]
    fn push(&self, pool_id: u64, arena: &PoolArena, bin_index: usize, offset: u32) -> bool {
        // SAFETY: `FastBinMagazine` is stored in `thread_local!` (`!Sync`), and
        // `push` is a leaf function that never re-enters `FAST_BIN_MAGAZINE`.
        let state = unsafe { &mut *self.state.get() };
        let bin = bin_index & (FAST_BIN_COUNT - 1);
        if state.pool_id != pool_id || (state.allocated_mask & (1u64 << bin)) == 0 {
            return false;
        }
        // SAFETY: `state.pool_id == pool_id != 0`, which holds only while `MAGAZINE_OWNER_GUARD`
        // is alive on this thread and owns the `Arc<ThreadMagazine>` pointed to by `state.shared`.
        let shared = unsafe { &*state.shared };
        let mut c = state.counts[bin] as usize;
        if c >= MAGAZINE_CAP {
            // ORDERING: `drain_into_arena` drains `slots[bin][0..MAGAZINE_CAP]` in ascending
            // order with `AcqRel`; observing `slots[bin][MAGAZINE_CAP - 1] == 0` with `Acquire`
            // guarantees all `0..MAGAZINE_CAP` slots were drained to `0`.
            if shared.slots[bin][MAGAZINE_CAP - 1].load(Ordering::Acquire) != 0 {
                return false;
            }
            c = 0;
        }
        // ORDERING: Release publishes block header/data writes before making `offset` visible
        // to `drain_all_magazines`. All slots at `>= c` are guaranteed `0`.
        shared.slots[bin][c & (MAGAZINE_CAP - 1)].store(offset, Ordering::Release);
        state.counts[bin] = (c + 1) as u8;
        let bit = 1u64 << bin;
        // ORDERING: Relaxed fast-path check avoids RMW when the bin bit is already set;
        // Release on `fetch_or` makes the non-empty bin visible to `drain_into_arena`.
        if (shared.non_empty_bins.load(Ordering::Relaxed) & bit) == 0 {
            shared.non_empty_bins.fetch_or(bit, Ordering::Release);
        }
        // ORDERING: Relaxed fast-path check avoids RMW when the magazine is already marked dirty;
        // AcqRel on `swap` ensures `arena.dirty_magazines` is incremented once per transition (`S49-R4`).
        if !shared.is_dirty.load(Ordering::Relaxed)
            && !shared.is_dirty.swap(true, Ordering::AcqRel)
        {
            arena.dirty_magazines.fetch_add(1, Ordering::Release);
        }
        true
    }

    fn record_alloc_miss(
        &self,
        pool_id: u64,
        arena: &Arc<PoolArena>,
        bin_index: usize,
        allow_cache: bool,
    ) {
        // SAFETY: `FastBinMagazine` is stored in `thread_local!` (`!Sync`), and
        // `record_alloc_miss` / `switch_pool` never re-enter `FAST_BIN_MAGAZINE`.
        let state = unsafe { &mut *self.state.get() };
        let bin = bin_index & (FAST_BIN_COUNT - 1);
        if state.pool_id != pool_id {
            if allow_cache {
                state.switch_pool(pool_id, arena, bin);
            } else {
                state.flush_to_current_owner();
            }
        } else if allow_cache {
            state.allocated_mask |= 1u64 << bin;
        } else {
            state.allocated_mask &= !(1u64 << bin);
        }
    }

    fn clear_if_pool(&self, pool_id: u64) {
        // SAFETY: `FastBinMagazine` is stored in `thread_local!` (`!Sync`), and
        // `clear_if_pool` never re-enters `FAST_BIN_MAGAZINE`.
        let state = unsafe { &mut *self.state.get() };
        if state.pool_id == pool_id {
            state.pool_id = 0;
            state.allocated_mask = 0;
            state.counts = [0; FAST_BIN_COUNT];
            if !state.shared.is_null() {
                // SAFETY: `state.pool_id == pool_id != 0` guarantees `state.shared` is valid.
                unsafe { (&*state.shared).clear_all() };
                state.shared = std::ptr::null();
            }
            let _ = MAGAZINE_OWNER_GUARD.try_with(|guard| {
                let mut g = guard.borrow_mut();
                if let Some(shared) = g.shared.take() {
                    if let Some(weak) = g.owner.take()
                        && let Some(arena) = weak.upgrade()
                    {
                        arena.unregister_magazine(&shared);
                    }
                } else {
                    g.owner = None;
                }
            });
        }
    }
}

struct MagazineOwnerGuard {
    owner: Option<Weak<PoolArena>>,
    shared: Option<Arc<ThreadMagazine>>,
}

impl Drop for MagazineOwnerGuard {
    fn drop(&mut self) {
        let _ = FAST_BIN_MAGAZINE.try_with(|mag| {
            // SAFETY: `FAST_BIN_MAGAZINE` has no `Drop` glue and remains valid
            // throughout thread teardown; no re-entrant borrow occurs.
            let state = unsafe { &mut *mag.state.get() };
            state.pool_id = 0;
            state.allocated_mask = 0;
            state.counts = [0; FAST_BIN_COUNT];
            state.shared = std::ptr::null();
        });
        if let Some(shared) = self.shared.take() {
            if let Some(weak) = self.owner.take()
                && let Some(arena) = weak.upgrade()
            {
                shared.drain_into_arena(&arena);
                arena.unregister_magazine(&shared);
            } else {
                shared.clear_all();
            }
        }
    }
}

thread_local! {
    static FAST_BIN_MAGAZINE: FastBinMagazine = const { FastBinMagazine::new() };
    static MAGAZINE_OWNER_GUARD: std::cell::RefCell<MagazineOwnerGuard> = const {
        std::cell::RefCell::new(MagazineOwnerGuard {
            owner: None,
            shared: None,
        })
    };
}

/// Lock-free head for a free list with cache line padding
#[derive(Debug)]
#[repr(align(64))] // Cache line alignment to prevent false sharing
struct LockFreeHead {
    /// Atomic head pointer (packed: upper 32 bits = generation, lower 32 bits = offset)
    ///
    /// ABA-SAFE: The generation counter prevents the ABA problem by incrementing
    /// on every CAS operation, ensuring that even if an offset value A→B→A matches,
    /// the generation counter won't match, preventing use-after-free vulnerabilities.
    head: AtomicU64,
    /// Atomic count of free items
    count: AtomicU32,
    /// Padding to prevent false sharing (64 bytes total: 8 + 4 + 52 = 64)
    _padding: [u8; 64 - 12],
}

impl LockFreeHead {
    fn new() -> Self {
        Self {
            // Initialize with generation = 0, offset = LIST_TAIL (0)
            // This represents an empty list with initial generation counter
            head: AtomicU64::new(LIST_TAIL as u64),
            count: AtomicU32::new(0),
            _padding: [0; 64 - 12],
        }
    }
}

/// Configuration for lock-free memory pool
#[derive(Debug, Clone)]
pub struct LockFreePoolConfig {
    /// Size of the backing memory region in bytes
    pub memory_size: usize,
    /// Enable statistics collection (has small overhead)
    pub enable_stats: bool,
    /// Maximum retry attempts for CAS operations
    pub max_cas_retries: u32,
    /// Backoff strategy for failed CAS operations
    pub backoff_strategy: BackoffStrategy,
    /// Enable cache-line aligned allocations for better performance
    pub enable_cache_alignment: bool,
    /// Cache layout configuration for optimization
    pub cache_config: Option<CacheLayoutConfig>,
    /// Enable NUMA-aware allocation
    pub enable_numa_awareness: bool,
    /// Enable huge page allocation for large chunks (Linux only)
    pub enable_huge_pages: bool,
    /// Minimum chunk size for huge page allocation
    pub huge_page_threshold: usize,
    /// Enable SIMD-optimized operations for memory zeroing and scanning
    pub enable_simd_optimization: bool,
    /// Zero memory on free for security (uses SIMD if enabled)
    pub zero_on_free: bool,
}

/// Backoff strategy for failed CAS operations
#[derive(Debug, Clone, Copy)]
pub enum BackoffStrategy {
    /// No backoff, immediate retry
    None,
    /// Linear backoff with microsecond delays
    Linear,
    /// Exponential backoff with maximum delay
    Exponential { max_delay_us: u64 },
}

impl Default for LockFreePoolConfig {
    fn default() -> Self {
        Self {
            memory_size: 64 * 1024 * 1024, // 64MB default
            enable_stats: true,
            max_cas_retries: 1000,
            backoff_strategy: BackoffStrategy::Exponential { max_delay_us: 1000 },
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::new()),
            enable_numa_awareness: true,
            enable_huge_pages: cfg!(target_os = "linux"),
            huge_page_threshold: 2 * 1024 * 1024, // 2MB
            enable_simd_optimization: true,
            zero_on_free: false,
        }
    }
}

impl LockFreePoolConfig {
    /// Create configuration for high-throughput scenarios
    pub fn high_performance() -> Self {
        Self {
            memory_size: 256 * 1024 * 1024, // 256MB
            enable_stats: false,            // Disable for maximum performance
            max_cas_retries: 10000,
            backoff_strategy: BackoffStrategy::Exponential { max_delay_us: 100 },
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::sequential()), // Assume sequential for high perf
            enable_numa_awareness: true,
            enable_huge_pages: true,
            huge_page_threshold: 1024 * 1024, // 1MB for high performance
            enable_simd_optimization: true,
            zero_on_free: false,
        }
    }

    /// Create configuration for memory-constrained scenarios
    pub fn compact() -> Self {
        Self {
            memory_size: 16 * 1024 * 1024, // 16MB
            enable_stats: true,
            max_cas_retries: 500,
            backoff_strategy: BackoffStrategy::Linear,
            enable_cache_alignment: false, // Disable for memory constraints
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 4 * 1024 * 1024, // 4MB
            enable_simd_optimization: false,
            zero_on_free: false,
        }
    }
}

/// Statistics for lock-free pool operations
#[derive(Debug, Default)]
pub struct LockFreePoolStats {
    /// Total allocations from fast bins
    pub fast_allocs: AtomicU64,
    /// Total allocations from skip list
    pub skip_allocs: AtomicU64,
    /// Total deallocations to fast bins
    pub fast_deallocs: AtomicU64,
    /// Total deallocations to skip list
    pub skip_deallocs: AtomicU64,
    /// CAS operation failures (contention indicator)
    pub cas_failures: AtomicU64,
    /// CAS operation successes
    pub cas_successes: AtomicU64,
    /// Memory utilization (allocated / total)
    pub memory_usage: AtomicU64,
    /// Cache-line aligned allocations
    pub cache_aligned_allocs: AtomicU64,
    /// NUMA-local allocations
    pub numa_local_allocs: AtomicU64,
    /// Huge page allocations
    pub huge_page_allocs: AtomicU64,
}

impl LockFreePoolStats {
    /// Get current allocation rate (allocs per second)
    pub fn allocation_rate(&self) -> f64 {
        // ORDERING: Relaxed is sufficient for diagnostic allocation counters.
        let total_allocs =
            self.fast_allocs.load(Ordering::Relaxed) + self.skip_allocs.load(Ordering::Relaxed);
        // Simplified calculation - in practice would track time
        total_allocs as f64
    }

    /// Get CAS contention ratio (failures / total operations)
    pub fn contention_ratio(&self) -> f64 {
        // ORDERING: Relaxed is sufficient for diagnostic CAS contention counters.
        let failures = self.cas_failures.load(Ordering::Relaxed);
        // ORDERING: Relaxed is sufficient for diagnostic CAS contention counters.
        let successes = self.cas_successes.load(Ordering::Relaxed);
        let total = failures + successes;
        if total == 0 {
            0.0
        } else {
            failures as f64 / total as f64
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FreeBlock {
    offset: u32,
    size: usize,
}

/// Address-ordered, coalescing free list for blocks above `FAST_BIN_THRESHOLD`.
///
/// Invariants, upheld by `alloc` and `free` and checked by `free` on the way in:
///
/// * entries are sorted by `offset` and pairwise disjoint;
/// * `offset` is a *user* offset, and the `BLOCK_HEADER` bytes in front of it
///   belong to the same block, so two blocks are adjacent exactly when
///   `a.offset + a.size + BLOCK_HEADER == b.offset`;
/// * **the width a block is carved at is the width it is filed back under.**
///   `alloc` splits a larger block rather than lending it whole, and when the
///   remainder would be too small to stand on its own it reports the width the
///   caller actually received so that `free` can give back exactly that.
#[derive(Debug, Default)]
struct LargeFreeList {
    /// Sorted by `offset`, disjoint, never two adjacent entries.
    blocks: Vec<FreeBlock>,
}

impl LargeFreeList {
    /// The narrowest user width a standalone block can have. A remainder below
    /// this cannot be split off, because it could not carry its own header and
    /// still hand back an `ALIGN_SIZE`-aligned user pointer.
    const MIN_SPLIT_REMAINDER: usize = ALIGN_SIZE;

    /// Best fit for `want` bytes of user memory.
    ///
    /// Returns `(user offset, the width actually reserved)`. The width is
    /// `want` whenever the block could be split, and the whole block otherwise;
    /// either way it is what `free` must be given back.
    fn alloc(&mut self, want: usize) -> Option<(u32, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (index, block) in self.blocks.iter().enumerate() {
            if block.size >= want && best.is_none_or(|(_, size)| block.size < size) {
                best = Some((index, block.size));
            }
        }

        let (index, size) = best?;
        let offset = self.blocks[index].offset;

        if size >= want + BLOCK_HEADER + Self::MIN_SPLIT_REMAINDER {
            // Split. The tail keeps this slot, so the list stays sorted, and it
            // cannot be adjacent to either neighbour because the block it came
            // from was not.
            self.blocks[index] = FreeBlock {
                offset: offset + (want + BLOCK_HEADER) as u32,
                size: size - want - BLOCK_HEADER,
            };
            Some((offset, want))
        } else {
            self.blocks.remove(index);
            Some((offset, size))
        }
    }

    /// File a block of exactly `size` user bytes starting at `offset`.
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if the region overlaps one that is already free.
    /// That means the same bytes were freed twice, or a pointer that this pool
    /// never handed out was passed to `deallocate`; filing it would let the
    /// pool hand one region to two live callers.
    fn free(&mut self, offset: u32, size: usize) -> Result<()> {
        let position = self.blocks.partition_point(|block| block.offset < offset);

        if position > 0 {
            let below = self.blocks[position - 1];
            if below.offset as usize + below.size > offset as usize {
                return Err(ZiporaError::invalid_data(
                    "large block freed twice, or a foreign pointer was freed: \
                     the region overlaps a free block below it",
                ));
            }
        }
        if position < self.blocks.len() {
            let above = self.blocks[position];
            if offset as usize + size > above.offset as usize {
                return Err(ZiporaError::invalid_data(
                    "large block freed twice, or a foreign pointer was freed: \
                     the region overlaps a free block above it",
                ));
            }
        }

        let mut offset = offset;
        let mut size = size;

        // Coalesce upward first: removing at `position` leaves `position - 1`
        // where it is.
        if position < self.blocks.len() {
            let above = self.blocks[position];
            if offset as usize + size + BLOCK_HEADER == above.offset as usize {
                size += BLOCK_HEADER + above.size;
                self.blocks.remove(position);
            }
        }
        if position > 0 {
            let below = self.blocks[position - 1];
            if below.offset as usize + below.size + BLOCK_HEADER == offset as usize {
                offset = below.offset;
                size += BLOCK_HEADER + below.size;
                self.blocks[position - 1] = FreeBlock { offset, size };
                return Ok(());
            }
        }

        self.blocks.insert(position, FreeBlock { offset, size });
        Ok(())
    }
}

/// Shared arena state for [`LockFreeMemoryPool`] and thread-local magazines.
struct PoolArena {
    pool_id: u64,
    memory: NonNull<u8>,
    memory_layout: Layout,
    memory_size: usize,
    fast_bins: Arc<Vec<CachePadded<LockFreeHead>>>,
    backoff_strategy: BackoffStrategy,
    stats: Option<Arc<LockFreePoolStats>>,
    dirty_magazines: AtomicUsize,
    magazines: Mutex<Vec<Weak<ThreadMagazine>>>,
}

// SAFETY: PoolArena is Send because it owns the heap allocation in `memory`
// (deallocated only in `PoolArena::drop` after all strong `Arc`s drop) and all
// other fields are `Send`.
unsafe impl Send for PoolArena {}

// SAFETY: PoolArena is Sync because `fast_bins` uses ABA-safe lock-free atomics
// and `magazines` is synchronized via `Mutex`.
unsafe impl Sync for PoolArena {}

impl PoolArena {
    fn register_magazine(&self, mag: &Arc<ThreadMagazine>) {
        if mag.pool_id != self.pool_id {
            return;
        }
        if let Ok(mut list) = self.magazines.lock() {
            list.retain(|w| w.strong_count() > 0);
            let ptr = Arc::as_ptr(mag);
            if !list.iter().any(|w| std::ptr::eq(w.as_ptr(), ptr)) {
                list.push(Arc::downgrade(mag));
            }
        }
    }

    fn unregister_magazine(&self, mag: &Arc<ThreadMagazine>) {
        // ORDERING: AcqRel ensures `dirty_magazines` is decremented at most once if `mag` was still dirty.
        if mag.pool_id == self.pool_id && mag.is_dirty.swap(false, Ordering::AcqRel) {
            self.dirty_magazines.fetch_sub(1, Ordering::Release);
        }
        if let Ok(mut list) = self.magazines.lock() {
            let ptr = Arc::as_ptr(mag);
            list.retain(|w| w.strong_count() > 0 && !std::ptr::eq(w.as_ptr(), ptr));
        }
    }

    /// Reclaim all blocks cached in per-thread `FastBinMagazine` instances back into `fast_bins`.
    #[cold]
    #[inline(never)]
    fn drain_all_magazines(&self) -> usize {
        // ORDERING: Acquire fast-path check skips `self.magazines` lock when no thread magazine
        // holds cached blocks (`S49-R4`).
        if self.dirty_magazines.load(Ordering::Acquire) == 0 {
            return 0;
        }
        let mut total = 0usize;
        if let Ok(mut list) = self.magazines.lock() {
            list.retain(|weak| {
                if let Some(mag) = weak.upgrade() {
                    if mag.pool_id == self.pool_id {
                        total += mag.drain_into_arena(self);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            });
        }
        total
    }

    #[inline(always)]
    fn offset_to_ptr(&self, offset: u32) -> Result<NonNull<u8>> {
        if offset == LIST_TAIL {
            return Err(ZiporaError::invalid_data("Invalid offset"));
        }
        if offset as usize >= self.memory_size {
            return Err(ZiporaError::invalid_data("Offset exceeds pool memory size"));
        }

        // SAFETY: Offset is validated to be within allocated memory pool size boundaries
        let addr = unsafe { self.memory.as_ptr().add(offset as usize) };
        NonNull::new(addr).ok_or_else(|| ZiporaError::invalid_data("Invalid pointer"))
    }

    #[inline(always)]
    fn link_slot(&self, offset: u32) -> Result<&AtomicU32> {
        let user = self.offset_to_ptr(offset)?;
        // SAFETY: `offset` was validated to lie inside the arena, every block is
        // carved by `allocate_new_block` with `BLOCK_HEADER` bytes in front of
        // its user memory, and user offsets are at least ALIGN_SIZE + BLOCK_HEADER
        // (the bump cursor starts at ALIGN_SIZE), so `user - BLOCK_HEADER` is an
        // ALIGN_SIZE-aligned (>= 4) address inside the arena that lives as long
        // as `self`. The header is only ever accessed through this `AtomicU32`.
        Ok(unsafe { &*(user.as_ptr().sub(BLOCK_HEADER) as *const AtomicU32) })
    }

    fn backoff(&self, retry_count: u32) {
        match self.backoff_strategy {
            BackoffStrategy::None => {}
            BackoffStrategy::Linear => {
                thread::sleep(Duration::from_micros(retry_count as u64));
            }
            BackoffStrategy::Exponential { max_delay_us } => {
                let delay = 1u64
                    .checked_shl(retry_count)
                    .unwrap_or(u64::MAX)
                    .min(max_delay_us);
                thread::sleep(Duration::from_micros(delay));
            }
        }
    }

    /// Push a block whose header tag has already been transitioned to `FAST_BIN_FREE_TAG`
    /// onto `self.fast_bins[bin_index]`.
    fn push_free_fast_bin_offset(&self, bin_index: usize, offset: u32) -> Result<()> {
        let bin = &self.fast_bins[bin_index];

        let mut retry = 0u32;
        loop {
            let packed = bin.head.load(Ordering::Acquire);
            let (current_offset, current_gen) = LockFreeMemoryPool::unpack_head(packed);

            // ORDERING: Relaxed is sufficient because this store is sequenced-before the
            // Release CAS on `bin.head` below, which publishes the link write to Acquire poppers.
            self.link_slot(offset)?
                .store(current_offset, Ordering::Relaxed);

            let new_packed = LockFreeMemoryPool::pack_head(offset, current_gen.wrapping_add(1));

            // ORDERING: Release on success publishes the caller's writes to the block and the
            // `link_slot` store above; Relaxed on failure is sufficient as the loop reloads
            // `bin.head` with Acquire on the next iteration.
            match bin.head.compare_exchange(
                packed,
                new_packed,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // ORDERING: Relaxed is sufficient for diagnostic bin count increments.
                    bin.count.fetch_add(1, Ordering::Relaxed);

                    if let Some(stats) = &self.stats {
                        // ORDERING: Relaxed is sufficient for diagnostic deallocation counters.
                        stats.fast_deallocs.fetch_add(1, Ordering::Relaxed);
                        // ORDERING: Relaxed is sufficient for diagnostic CAS counters.
                        stats.cas_successes.fetch_add(1, Ordering::Relaxed);
                    }

                    return Ok(());
                }
                Err(_) => {
                    if let Some(stats) = &self.stats {
                        // ORDERING: Relaxed is sufficient for diagnostic CAS failure counters.
                        stats.cas_failures.fetch_add(1, Ordering::Relaxed);
                    }

                    self.backoff(retry.min(MAX_BACKOFF_RETRY));
                    retry = retry.saturating_add(1);
                }
            }
        }
    }
}

impl Drop for PoolArena {
    fn drop(&mut self) {
        // SAFETY: ptr from matching alloc, layout matches allocation
        unsafe {
            dealloc(self.memory.as_ptr(), self.memory_layout);
        }
    }
}

/// Lock-free memory pool implementation
pub struct LockFreeMemoryPool {
    /// Unique identifier for thread-local magazine validation
    pool_id: u64,
    /// Backing memory region
    memory: NonNull<u8>,
    /// Cached backing region size in bytes (`config.memory_size`)
    memory_size: usize,
    /// Upper span `memory_size - (ALIGN_SIZE + BLOCK_HEADER)` for 1-branch offset validation
    max_valid_offset_span: usize,
    /// High-water bump offset above which `deallocate` returns blocks directly to `fast_bins`
    magazine_high_water: u32,
    /// Whether the thread-local fast-bin magazine is enabled
    use_tls_magazine: bool,
    /// Shared arena owning the backing allocation and fast bins
    arena: Arc<PoolArena>,
    /// Configuration
    config: LockFreePoolConfig,
    /// Fast bins for small/medium allocations
    fast_bins: Arc<Vec<CachePadded<LockFreeHead>>>,
    /// Free list for blocks above `FAST_BIN_THRESHOLD`
    large_free_list: Mutex<LargeFreeList>,
    /// Next available offset in memory region
    next_offset: AtomicU32,
    /// Statistics (optional)
    stats: Option<Arc<LockFreePoolStats>>,
    /// Cache optimization infrastructure
    cache_allocator: Option<CacheOptimizedAllocator>,
}

// SAFETY: LockFreeMemoryPool is Send because:
// 1. `config: LockFreePoolConfig` - Config is Clone and contains no pointers.
// 2. `arena: Arc<PoolArena>` / `memory: NonNull<u8>` - Backing heap region is owned by
//    `PoolArena` and deallocated only when all strong `Arc`s drop. Per-thread magazines
//    store 32-bit offsets + `Weak<PoolArena>` and flush on pool switch or thread exit.
// 3. `fast_bins: Arc<Vec<CachePadded<LockFreeHead>>>` - Contains only atomics.
// 4. `large_free_list: Mutex<...>` - Mutex is Send.
// 5. `next_offset: AtomicU32` - AtomicU32 is Send.
// 6. `stats: Option<Arc<LockFreePoolStats>>` - Arc<T> is Send if T is Send+Sync.
// 7. `cache_allocator: Option<CacheOptimizedAllocator>` - Safe container.
unsafe impl Send for LockFreeMemoryPool {}

// SAFETY: LockFreeMemoryPool is Sync because:
// 1. All mutable state uses atomic operations (LockFreeHead contains AtomicU64/AtomicU32).
// 2. Fast bin operations use lock-free CAS with generation counters (ABA-safe).
// 3. Skip list operations are protected by Mutex.
// 4. Memory offsets are used instead of raw pointers for safe concurrent access.
// 5. Acquire/Release ordering ensures proper happens-before relationships.
// 6. The pool never deallocates memory while allocated chunks may still be in use.
//
// The lock-free allocation protocol ensures thread-safe memory distribution.
unsafe impl Sync for LockFreeMemoryPool {}

impl LockFreeMemoryPool {
    /// Pack offset and generation counter into a single u64
    ///
    /// Layout: [generation: u32 (upper 32 bits)][offset: u32 (lower 32 bits)]
    ///
    /// This packing enables ABA-safe lock-free operations by incrementing
    /// the generation counter on every CAS, preventing the ABA problem where
    /// a pointer value matches but points to different data.
    #[inline]
    fn pack_head(offset: u32, generation: u32) -> u64 {
        ((generation as u64) << 32) | (offset as u64)
    }

    /// Unpack u64 into (offset, generation) tuple
    ///
    /// Returns: (offset: u32, generation: u32)
    #[inline]
    fn unpack_head(packed: u64) -> (u32, u32) {
        let offset = (packed & 0xFFFFFFFF) as u32;
        let generation = (packed >> 32) as u32;
        (offset, generation)
    }

    /// Create a new lock-free memory pool
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if `config.memory_size` exceeds `u32::MAX`:
    /// the pool addresses its backing region with 32-bit offsets (packed
    /// offset/generation heads, `AtomicU32` bump pointer), so a larger
    /// region would truncate offset arithmetic and alias live allocations.
    pub fn new(config: LockFreePoolConfig) -> Result<Self> {
        if config.memory_size > u32::MAX as usize {
            return Err(ZiporaError::invalid_data(
                "LockFreeMemoryPool memory_size cannot exceed 4GB (32-bit offset limit)",
            ));
        }

        // Initialize fast bins and metadata before the large backing buffer
        let mut fast_bins = Vec::with_capacity(FAST_BIN_COUNT);
        for _ in 0..FAST_BIN_COUNT {
            fast_bins.push(CachePadded::new(LockFreeHead::new()));
        }
        let fast_bins = Arc::new(fast_bins);

        // Initialize statistics if enabled
        let stats = if config.enable_stats {
            Some(Arc::new(LockFreePoolStats::default()))
        } else {
            None
        };

        // Initialize cache allocator if enabled
        let cache_allocator = if config.enable_cache_alignment && config.cache_config.is_some() {
            // SAFETY: is_some() check above guarantees this unwrap succeeds
            Some(CacheOptimizedAllocator::new(
                config
                    .cache_config
                    .clone()
                    .expect("cache_config present when cache_friendly enabled"),
            ))
        } else {
            None
        };

        let magazines = Mutex::new(Vec::with_capacity(256));

        // Allocate backing memory region
        let layout = Layout::from_size_align(config.memory_size, ALIGN_SIZE)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {}", e)))?;

        // SAFETY: layout valid (size > 0, align power of 2); zero-initialized so
        // fast-bin header tags start at 0 before first carve.
        let memory = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(config.memory_size))?;

        // ORDERING: Relaxed is sufficient for unique monotonic pool ID assignment.
        let pool_id = NEXT_LOCKFREE_POOL_ID.fetch_add(1, Ordering::Relaxed);
        let use_tls_magazine = !config.enable_stats
            && config.max_cas_retries > 0
            && config.memory_size >= 16 * 1024 * 1024;
        let magazine_high_water = config.memory_size.saturating_sub(config.memory_size >> 3) as u32;

        let memory_size = config.memory_size;
        let max_valid_offset_span = memory_size.saturating_sub(ALIGN_SIZE + BLOCK_HEADER);

        let arena = Arc::new(PoolArena {
            pool_id,
            memory,
            memory_layout: layout,
            memory_size,
            fast_bins: Arc::clone(&fast_bins),
            backoff_strategy: config.backoff_strategy,
            stats: stats.clone(),
            dirty_magazines: AtomicUsize::new(0),
            magazines,
        });

        Ok(Self {
            pool_id,
            memory,
            memory_size,
            max_valid_offset_span,
            magazine_high_water,
            use_tls_magazine,
            arena,
            config,
            fast_bins,
            large_free_list: Mutex::new(LargeFreeList::default()),
            next_offset: AtomicU32::new(ALIGN_SIZE as u32), // Start after header
            stats,
            cache_allocator,
        })
    }

    #[inline(always)]
    fn find_fast_bin(aligned_size: usize) -> usize {
        SIZE_CLASS_LUT[aligned_size >> 3] as usize
    }

    #[inline(always)]
    fn mark_fast_bin_live(&self, ptr: NonNull<u8>, bin_index: usize) {
        // SAFETY: `ptr` is an 8-byte-aligned user pointer inside `self.memory` at
        // offset >= 16, so `ptr - 4` is the second 4-byte word of its 8-byte `BLOCK_HEADER`.
        let tag_slot = unsafe { &*(ptr.as_ptr().sub(4) as *const AtomicU32) };
        // ORDERING: Release publishes `FAST_BIN_LIVE_TAG` before the block is handed to
        // the caller or passed across threads to `deallocate`.
        tag_slot.store(FAST_BIN_LIVE_TAG | (bin_index as u32), Ordering::Release);
    }

    #[inline(always)]
    fn transition_fast_bin_live_to_free(&self, ptr: NonNull<u8>, bin_index: usize) -> Result<()> {
        // SAFETY: `ptr_to_offset` already verified `16 <= offset < self.memory_size` and
        // 8-byte alignment, so `ptr - 4` is a valid 4-byte-aligned `AtomicU32` in `BLOCK_HEADER`.
        let tag_slot = unsafe { &*(ptr.as_ptr().sub(4) as *const AtomicU32) };
        let expected = FAST_BIN_LIVE_TAG | (bin_index as u32);
        let free_tag = FAST_BIN_FREE_TAG | (bin_index as u32);
        // ORDERING: Relaxed pre-check rejects non-live blocks without dirtying the cache line;
        // AcqRel CAS ensures at most one thread can free a live block (`S49-R3`).
        if tag_slot.load(Ordering::Relaxed) != expected
            || tag_slot
                .compare_exchange(expected, free_tag, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
        {
            return Err(ZiporaError::invalid_data(
                "Fast-bin block freed twice or pointer was not allocated at this size class",
            ));
        }
        Ok(())
    }

    /// Allocate memory from the pool
    #[inline(always)]
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("Cannot allocate zero bytes"));
        }

        let aligned_size = self.align_size(size);

        if aligned_size <= FAST_BIN_THRESHOLD {
            let bin_index = Self::find_fast_bin(aligned_size);
            if self.use_tls_magazine
                && let Some(offset) = FAST_BIN_MAGAZINE.with(|mag| mag.pop(self.pool_id, bin_index))
            {
                // SAFETY: Every offset pushed into `FAST_BIN_MAGAZINE` for `self.pool_id`
                // was validated by `ptr_to_offset` to lie within `16..self.memory_size` of `self.memory`.
                let ptr =
                    unsafe { NonNull::new_unchecked(self.memory.as_ptr().add(offset as usize)) };
                self.mark_fast_bin_live(ptr, bin_index);
                return Ok(ptr);
            }
            self.allocate_from_fast_bin_index(bin_index)
        } else {
            self.allocate_from_skip_list(aligned_size)
        }
    }

    /// Deallocate memory back to the pool
    #[inline(always)]
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }

        let aligned_size = self.align_size(size);

        if aligned_size <= FAST_BIN_THRESHOLD {
            let offset = self.ptr_to_offset(ptr)?;
            let bin_index = Self::find_fast_bin(aligned_size);
            self.transition_fast_bin_live_to_free(ptr, bin_index)?;
            if self.use_tls_magazine
                && FAST_BIN_MAGAZINE.with(|mag| mag.push(self.pool_id, &self.arena, bin_index, offset))
            {
                return Ok(());
            }
            self.arena.push_free_fast_bin_offset(bin_index, offset)
        } else {
            // `size` is what the caller *asked* for, which can be narrower than
            // the block it was given. Filing that width would shrink the block
            // for good, so take the width from the block's own header.
            let offset = self.ptr_to_offset(ptr)?;
            let width = self.load_block_width(offset, aligned_size)?;
            self.deallocate_to_skip_list(ptr, width)
        }
    }

    /// Deallocate with optional SIMD zeroing (lock-free safe)
    ///
    /// # Safety
    /// This is lock-free safe because:
    /// 1. We own the pointer (caller guarantees)
    /// 2. SIMD zeroing happens BEFORE atomic CAS operation
    /// 3. No concurrent access to the memory region during zeroing
    pub fn deallocate_with_zero(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }

        // Step 1: Zero memory using SIMD (safe because we own the pointer)
        if self.config.zero_on_free && self.config.enable_simd_optimization {
            // SAFETY: pointer valid from allocation, size matches allocation
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), size) };
            fast_fill(slice, 0);
        }

        // Step 2: Return to freelist with atomic CAS
        self.deallocate(ptr, size)
    }

    /// Allocate multiple blocks with SIMD optimization
    ///
    /// # Performance
    /// Uses prefetching to reduce latency for bulk allocations.
    /// Prefetch distance is adaptive based on allocation size.
    pub fn allocate_bulk_simd(&self, sizes: &[usize]) -> Result<Vec<NonNull<u8>>> {
        const PREFETCH_DISTANCE: usize = 8;
        let mut results = Vec::with_capacity(sizes.len());

        for (i, &size) in sizes.iter().enumerate() {
            // Prefetch metadata for future allocations
            if i + PREFETCH_DISTANCE < sizes.len() && self.config.enable_simd_optimization {
                let future_size = sizes[i + PREFETCH_DISTANCE];
                let aligned_future_size = self.align_size(future_size);

                if aligned_future_size <= FAST_BIN_THRESHOLD
                    && let Ok(bin_idx) = self.size_to_bin_index(aligned_future_size)
                {
                    #[cfg(target_arch = "x86_64")]
                    {
                        fast_prefetch(&self.fast_bins[bin_idx], PrefetchHint::T0);
                    }
                }
            }

            // Perform allocation
            results.push(self.allocate(size)?);
        }

        Ok(results)
    }

    /// Get pool statistics (if enabled)
    pub fn stats(&self) -> Option<Arc<LockFreePoolStats>> {
        self.stats.clone()
    }

    /// Pop one block off a fast bin's Treiber stack.
    ///
    /// `Ok(None)` means no block was obtained -- either the bin is empty, or
    /// `budget` CAS attempts were spent without winning one. The caller is
    /// responsible for turning that into the right answer; see
    /// `allocate_from_fast_bin`.
    ///
    /// C3.18: the CAS here is *strong*. A budgeted loop must never use
    /// `compare_exchange_weak`, whose failures are allowed to be spurious:
    /// LL/SC hardware produces them, Miri models them, and each one burns a
    /// retry with no contention at all. The loop re-reads `head` every
    /// iteration, so the weak form bought nothing in exchange.
    fn pop_fast_bin(&self, bin: &LockFreeHead, budget: u32) -> Result<Option<u32>> {
        for retry in 0..budget {
            // ABA-SAFE: Load packed value (offset + generation)
            let packed = bin.head.load(Ordering::Acquire);
            let (current_offset, current_gen) = Self::unpack_head(packed);

            if current_offset == LIST_TAIL {
                return Ok(None);
            }

            // Between loading `head` and this load another thread may pop
            // `current_offset` and hand the block to user code. The link is in
            // the block header, which user code can never reach, so this load
            // only ever overlaps a pusher's atomic store. If it was stale the
            // generation tag makes the CAS below fail and the value is dropped.
            // ORDERING: The Acquire load of `bin.head` above synchronizes-with the pusher's
            // Release CAS in `deallocate_to_fast_bin`, which happens-after the pusher's
            // `link_slot(offset).store(current_offset, Relaxed)`, establishing happens-before.
            let next_offset = self.link_slot(current_offset)?.load(Ordering::Relaxed);

            // ABA-SAFE: Pack next offset with INCREMENTED generation counter
            // This prevents ABA: even if offset A→B→A, generation won't match
            let next_packed = Self::pack_head(next_offset, current_gen.wrapping_add(1));

            match bin.head.compare_exchange(
                packed,      // Compare full packed value (offset + generation)
                next_packed, // New packed value with incremented generation
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // SAFETY FIX (v2.1.1): Use Release ordering to synchronize with head update
                    // This ensures the count decrement is visible to other threads that observe
                    // the new head value, preventing race conditions in high-contention scenarios
                    bin.count.fetch_sub(1, Ordering::Release);

                    if let Some(stats) = &self.stats {
                        // ORDERING: Relaxed is sufficient for diagnostic allocation counters.
                        stats.fast_allocs.fetch_add(1, Ordering::Relaxed);
                        // ORDERING: Relaxed is sufficient for diagnostic CAS counters.
                        stats.cas_successes.fetch_add(1, Ordering::Relaxed);
                    }

                    return Ok(Some(current_offset));
                }
                Err(_) => {
                    if let Some(stats) = &self.stats {
                        // ORDERING: Relaxed is sufficient for diagnostic CAS failure counters.
                        stats.cas_failures.fetch_add(1, Ordering::Relaxed);
                    }

                    self.backoff(retry.min(MAX_BACKOFF_RETRY));
                }
            }
        }

        Ok(None)
    }

    /// Allocate from fast bin using lock-free stack
    fn allocate_from_fast_bin_index(&self, bin_index: usize) -> Result<NonNull<u8>> {
        if self.use_tls_magazine {
            // ORDERING: Relaxed is sufficient for checking the arena high-water mark on a miss.
            let allow_cache = self.next_offset.load(Ordering::Relaxed) <= self.magazine_high_water;
            FAST_BIN_MAGAZINE.with(|mag| {
                mag.record_alloc_miss(self.pool_id, &self.arena, bin_index, allow_cache);
            });
        }
        let bin = &self.fast_bins[bin_index];

        // Prefer a recycled block. `max_cas_retries` bounds how long we are
        // willing to contend for one before carving fresh memory instead; it is
        // a throughput knob, not a correctness parameter.
        if let Some(offset) = self.pop_fast_bin(bin, self.config.max_cas_retries)? {
            let ptr = self.offset_to_ptr(offset)?;
            self.mark_fast_bin_live(ptr, bin_index);
            return Ok(ptr);
        }

        // Empty bin, or the budget ran out: carve a fresh block at the class
        // width, not the request size. Blocks are recycled by bin, so a block
        // carved any smaller would later be handed to a larger request in the
        // same class and overrun its neighbour.
        match self.allocate_new_block(FAST_BIN_SIZES[bin_index]) {
            Ok(ptr) => {
                self.mark_fast_bin_live(ptr, bin_index);
                Ok(ptr)
            }
            Err(exhausted) => {
                if self.use_tls_magazine {
                    self.arena.drain_all_magazines();
                }
                // C3.18. The arena is spent, so the bin is the only source
                // left, and a spent retry budget must not be allowed to turn a
                // non-empty bin into an out-of-memory error -- which is exactly
                // what a single-threaded pool used to report under Miri, whose
                // `compare_exchange_weak` fails spuriously. Retry unbudgeted:
                // with a strong CAS every failure means another thread
                // completed a push or a pop, so this terminates.
                match self.pop_fast_bin(bin, u32::MAX)? {
                    Some(offset) => {
                        let ptr = self.offset_to_ptr(offset)?;
                        self.mark_fast_bin_live(ptr, bin_index);
                        Ok(ptr)
                    }
                    None => Err(exhausted),
                }
            }
        }
    }

    /// Allocate from the large free list, or carve a fresh block.
    fn allocate_from_skip_list(&self, size: usize) -> Result<NonNull<u8>> {
        let aligned_size = self.align_size(size);

        // Reuse a deallocated block if one fits (best fit, split exactly).
        let reused = {
            let mut free_list = self
                .large_free_list
                .lock()
                .map_err(|_| ZiporaError::invalid_data("Mutex poisoned"))?;
            free_list.alloc(aligned_size)
        };

        if let Some(stats) = &self.stats {
            // ORDERING: Relaxed is sufficient for diagnostic large-allocation counters.
            stats.skip_allocs.fetch_add(1, Ordering::Relaxed);
        }

        let (offset, width) = match reused {
            Some(found) => found,
            None => {
                let ptr = self.allocate_new_block(aligned_size)?;
                (self.ptr_to_offset(ptr)?, aligned_size)
            }
        };

        self.store_block_width(offset, width)?;
        self.offset_to_ptr(offset)
    }

    /// File a large block back, at the width it was carved at.
    fn deallocate_to_skip_list(&self, ptr: NonNull<u8>, width: usize) -> Result<()> {
        let offset = self.ptr_to_offset(ptr)?;

        let mut free_list = self
            .large_free_list
            .lock()
            .map_err(|_| ZiporaError::invalid_data("Mutex poisoned"))?;
        free_list.free(offset, width)?;
        drop(free_list);

        if let Some(stats) = &self.stats {
            // ORDERING: Relaxed is sufficient for diagnostic large-deallocation counters.
            stats.skip_deallocs.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Allocate a new block from the backing memory with cache optimizations
    fn allocate_new_block(&self, size: usize) -> Result<NonNull<u8>> {
        let aligned_size = self.align_size(size);
        // Header first, user memory behind it; the offset handed out (and stored
        // in every free list) is the user memory's.
        let carve = BLOCK_HEADER + aligned_size;

        // Always allocate from backing memory to ensure consistent pointer validation
        // External cache allocations would cause pointer validation failures in deallocate
        // ORDERING: Relaxed is sufficient for the bump cursor because each carved range
        // `[current, next)` is disjoint and never overlaps another thread's range.
        let mut current = self.next_offset.load(Ordering::Relaxed);
        loop {
            if current as usize + carve > self.config.memory_size {
                return Err(ZiporaError::out_of_memory(aligned_size));
            }
            let next = match current.checked_add(carve as u32) {
                Some(val) => val,
                None => return Err(ZiporaError::out_of_memory(aligned_size)),
            };
            // ORDERING: Relaxed is sufficient for disjoint monotonic bump allocation.
            match self.next_offset.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
        let offset = current + BLOCK_HEADER as u32;

        let ptr = self.offset_to_ptr(offset)?;

        // Apply cache optimization hints if enabled
        if let Some(ref _cache_allocator) = self.cache_allocator
            && self.config.enable_cache_alignment
        {
            if let Some(stats) = &self.stats {
                // ORDERING: Relaxed is sufficient for diagnostic cache-aligned counters.
                stats.cache_aligned_allocs.fetch_add(1, Ordering::Relaxed);
            }

            // Apply cache-friendly operations on the pool memory
            #[cfg(target_arch = "x86_64")]
            // SAFETY: pointer valid from allocation, prefetch doesn't require alignment
            unsafe {
                // Prefetch the allocated memory for cache optimization
                std::arch::x86_64::_mm_prefetch(
                    ptr.as_ptr() as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }

            // Memory is already zeroed by default in most allocators
            // Additional zeroing could be added here if needed for security
        }

        if let Some(stats) = &self.stats {
            // ORDERING: Relaxed is sufficient for diagnostic memory usage counters.
            stats
                .memory_usage
                .fetch_add(carve as u64, Ordering::Relaxed);
        }

        Ok(ptr)
    }

    /// Free-list link of the block whose user memory starts at `offset`.
    fn link_slot(&self, offset: u32) -> Result<&AtomicU32> {
        let user = self.offset_to_ptr(offset)?;
        // SAFETY: `offset` was validated to lie inside the arena, every block is
        // carved by `allocate_new_block` with `BLOCK_HEADER` bytes in front of
        // its user memory, and user offsets are at least ALIGN_SIZE + BLOCK_HEADER
        // (the bump cursor starts at ALIGN_SIZE), so `user - BLOCK_HEADER` is an
        // ALIGN_SIZE-aligned (>= 4) address inside the arena that lives as long
        // as `self`. The header is only ever accessed through this `AtomicU32`.
        Ok(unsafe { &*(user.as_ptr().sub(BLOCK_HEADER) as *const AtomicU32) })
    }

    /// Record, in a large block's own header, the user width it was carved at.
    ///
    /// `deallocate` is handed the caller's *request* size, which can be
    /// narrower than the block it received -- best fit lends the smallest block
    /// that fits, and a remainder too small to stand on its own is not split
    /// off. Filing the request width instead of the carved width would
    /// permanently reclassify the block and erode the arena, so the true width
    /// has to be recorded somewhere the caller cannot reach. The header is that
    /// place: `BLOCK_HEADER` bytes in front of the user memory, used by the
    /// fast bins for the free-list link, and otherwise unused by blocks on the
    /// large path (a block is routed to the bins or to the large free list by
    /// size, and the two ranges do not overlap).
    fn store_block_width(&self, offset: u32, width: usize) -> Result<()> {
        // `memory_size <= u32::MAX` is enforced by `new`, so any width that
        // fits in the arena fits in a u32.
        let width = u32::try_from(width)
            .map_err(|_| ZiporaError::invalid_data("Block width exceeds 32 bits"))?;
        // ORDERING: Relaxed is sufficient because `store_block_width` completes before
        // `allocate` returns the pointer to the caller, and any cross-thread handoff of the
        // pointer to `deallocate` (plus `large_free_list` Mutex across reuse cycles) provides
        // happens-before.
        self.link_slot(offset)?.store(width, Ordering::Relaxed);
        Ok(())
    }

    /// Read back the width stored by `store_block_width`.
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if the header does not describe a block that is
    /// at least `least` bytes wide and inside the arena. That means the pointer
    /// did not come from this pool's large path, so the pool refuses it rather
    /// than filing a bogus region on the free list.
    fn load_block_width(&self, offset: u32, least: usize) -> Result<usize> {
        // ORDERING: Relaxed is sufficient because the caller's handoff from `allocate` to
        // `deallocate` (or single-thread program order) establishes happens-before with
        // `store_block_width`.
        let width = self.link_slot(offset)?.load(Ordering::Relaxed) as usize;
        if width < least || offset as usize + width > self.config.memory_size {
            return Err(ZiporaError::invalid_data(
                "Block header does not describe a large block of at least the \
                 deallocated size; the pointer was not allocated by this pool",
            ));
        }
        Ok(width)
    }

    /// Convert size to fast bin index
    #[inline(always)]
    fn size_to_bin_index(&self, size: usize) -> Result<usize> {
        let idx = size.div_ceil(ALIGN_SIZE);
        if let Some(&bin) = SIZE_CLASS_LUT.get(idx) {
            Ok(bin as usize)
        } else {
            Err(ZiporaError::invalid_data("Size too large for fast bins"))
        }
    }

    /// Align size to allocation boundary
    #[inline(always)]
    fn align_size(&self, size: usize) -> usize {
        (size + ALIGN_SIZE - 1) & !(ALIGN_SIZE - 1)
    }

    /// Convert offset to pointer
    #[inline(always)]
    fn offset_to_ptr(&self, offset: u32) -> Result<NonNull<u8>> {
        if offset == LIST_TAIL || offset as usize >= self.memory_size {
            return Err(ZiporaError::invalid_data("Invalid offset"));
        }

        // SAFETY: Offset is validated to be within allocated memory pool size boundaries
        let addr = unsafe { self.memory.as_ptr().add(offset as usize) };
        NonNull::new(addr).ok_or_else(|| ZiporaError::invalid_data("Invalid pointer"))
    }

    /// Convert pointer to offset
    #[inline(always)]
    fn ptr_to_offset(&self, ptr: NonNull<u8>) -> Result<u32> {
        let base = self.memory.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;
        let off = addr.wrapping_sub(base);

        if off.wrapping_sub(ALIGN_SIZE + BLOCK_HEADER) >= self.max_valid_offset_span
            || (off & (ALIGN_SIZE - 1)) != 0
        {
            return Err(ZiporaError::invalid_data("Pointer outside pool memory"));
        }

        Ok(off as u32)
    }

    /// Implement backoff strategy for failed CAS operations
    fn backoff(&self, retry_count: u32) {
        match self.config.backoff_strategy {
            BackoffStrategy::None => {}
            BackoffStrategy::Linear => {
                thread::sleep(Duration::from_micros(retry_count as u64));
            }
            BackoffStrategy::Exponential { max_delay_us } => {
                // `checked_shl`, not `<<`: `max_cas_retries` defaults to 1000,
                // so `retry_count` reaches 64 and the shift overflows -- a
                // panic in debug, and in release a wrapped shift that resets
                // the delay to 1 microsecond exactly when contention is
                // highest.
                let delay = 1u64
                    .checked_shl(retry_count)
                    .unwrap_or(u64::MAX)
                    .min(max_delay_us);
                thread::sleep(Duration::from_micros(delay));
            }
        }
    }

    //==============================================================================
    // SIMD-OPTIMIZED HELPER METHODS (LOCK-FREE SAFE)
    //==============================================================================
}

impl Drop for LockFreeMemoryPool {
    fn drop(&mut self) {
        let _ = FAST_BIN_MAGAZINE.try_with(|mag| mag.clear_if_pool(self.pool_id));
    }
}

/// RAII wrapper for lock-free pool allocations
pub struct LockFreeAllocation {
    ptr: NonNull<u8>,
    size: usize,
    pool: Arc<LockFreeMemoryPool>,
}

impl LockFreeAllocation {
    /// Create new allocation wrapper
    pub fn new(ptr: NonNull<u8>, size: usize, pool: Arc<LockFreeMemoryPool>) -> Self {
        Self { ptr, size, pool }
    }

    /// Get pointer to allocated memory
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size of allocation
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get mutable slice view of allocation
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: pointer valid from allocation, size matches allocation
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get immutable slice view of allocation
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: pointer valid from allocation, size matches allocation
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for LockFreeAllocation {
    fn drop(&mut self) {
        if let Err(e) = self.pool.deallocate(self.ptr, self.size) {
            log::error!("Failed to deallocate lock-free memory: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// C3.11. `backoff` computed `1u64 << retry_count`. `max_cas_retries`
    /// defaults to 1000 and the default strategy is `Exponential`, so a bin
    /// under sustained contention reaches a retry count of 64 and the shift
    /// overflows: a panic in a debug build, and in release a wrapped shift
    /// (`1 << (n % 64)`) that makes the delay collapse back to 1 µs exactly
    /// when the pool is most contended.
    ///
    /// The retry loop is not reachable deterministically from outside, so the
    /// test calls the private method directly with the retry count the loop is
    /// allowed to reach.
    #[test]
    fn test_backoff_does_not_overflow_at_high_retry_counts() {
        let config = LockFreePoolConfig::default();
        assert!(
            matches!(config.backoff_strategy, BackoffStrategy::Exponential { .. }),
            "this test is about the default strategy"
        );
        let retries = config.max_cas_retries;
        assert!(retries > 64, "the loop must be able to reach a 64th retry");

        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: 64 * 1024,
            // Keep the sleep bounded: the point is the shift, not the wait.
            backoff_strategy: BackoffStrategy::Exponential { max_delay_us: 1 },
            ..LockFreePoolConfig::default()
        })
        .unwrap();

        for retry in [63, 64, 65, retries - 1] {
            pool.backoff(retry);
        }
    }

    /// C3.10. `allocate_from_skip_list` takes a best-fit free block of
    /// `best_size >= aligned_size` and hands it over *whole*, while
    /// `deallocate_to_skip_list` files it back under the caller's *request*
    /// size. Every time a large block is lent to a smaller request it is
    /// permanently reclassified as that smaller block, and the difference can
    /// never be allocated again -- the carve width and the recycle width
    /// disagree.
    ///
    /// The demand pattern has to *shrink* within a round for the erosion to
    /// show: a 16 KiB block lent to 12 KiB and then to 8200 bytes ends the
    /// round as an 8200-byte block, so the next round's 16 KiB request has to
    /// carve fresh arena. Roughly 16 KiB of the 256 KiB arena is lost per
    /// round even though every allocation is freed before the next one is
    /// made.
    #[test]
    fn test_large_blocks_keep_their_width_when_lent_to_smaller_requests() {
        const ARENA: usize = 256 * 1024;
        const BIG: usize = 16 * 1024;
        const ROUNDS: usize = 32;

        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: ARENA,
            enable_stats: true,
            ..Default::default()
        })
        .unwrap();

        for round in 0..ROUNDS {
            // Nothing is held across an iteration, so one BIG block is enough
            // to serve the whole loop.
            let big = pool.allocate(BIG).unwrap_or_else(|e| {
                panic!(
                    "round {round}: the {BIG}-byte request found nothing to \
                     reuse and the arena is exhausted, yet every earlier \
                     allocation was freed: {e}"
                )
            });
            pool.deallocate(big, BIG).unwrap();

            // Each round asks for a slightly *larger* small block than the
            // last, so the only free block that fits is the BIG one -- which
            // is lent whole and then filed back under this smaller width.
            let smaller = FAST_BIN_THRESHOLD + 8 * (round + 1);
            let ptr = pool.allocate(smaller).unwrap();
            pool.deallocate(ptr, smaller).unwrap();
        }

        let carved = pool
            .stats()
            .expect("stats enabled")
            .memory_usage
            .load(Ordering::Relaxed);
        assert!(
            carved <= (BIG + BLOCK_HEADER) as u64,
            "one block should have served all {} allocations, but {carved} \
             bytes were carved from the arena",
            ROUNDS * 2
        );
    }

    /// C3.10. A block lent whole (because the remainder would have been too
    /// small to stand on its own) must come back whole. The pool records the
    /// width it actually handed over in the block header, so the caller's
    /// smaller `deallocate` size cannot shrink it.
    #[test]
    fn test_a_block_lent_whole_is_filed_back_whole() {
        const ARENA: usize = 128 * 1024;
        // A remainder of 8 bytes cannot carry its own BLOCK_HEADER, so the
        // whole block is lent out rather than split.
        let big = FAST_BIN_THRESHOLD + 16;
        let slightly_smaller = big - 8;

        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: ARENA,
            enable_stats: true,
            ..Default::default()
        })
        .unwrap();

        let a = pool.allocate(big).unwrap();
        pool.deallocate(a, big).unwrap();

        let b = pool.allocate(slightly_smaller).unwrap();
        assert_eq!(b, a, "the free block should have been reused");
        pool.deallocate(b, slightly_smaller).unwrap();

        let c = pool.allocate(big).unwrap();
        assert_eq!(
            c, a,
            "the block was lent to a smaller request and came back narrower, \
             so the original width is no longer allocatable"
        );
        pool.deallocate(c, big).unwrap();
    }

    /// C3.10. Freeing the same large block twice must be refused rather than
    /// filed twice: two entries for one region would let the pool hand the
    /// same bytes to two live callers.
    #[test]
    fn test_double_free_of_a_large_block_is_refused() {
        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: 128 * 1024,
            ..Default::default()
        })
        .unwrap();

        let width = FAST_BIN_THRESHOLD + 1024;
        let ptr = pool.allocate(width).unwrap();
        pool.deallocate(ptr, width).unwrap();
        assert!(
            pool.deallocate(ptr, width).is_err(),
            "the second free of {ptr:?} was accepted"
        );

        let a = pool.allocate(width).unwrap();
        let b = pool.allocate(width).unwrap();
        assert_ne!(a, b, "the same region was handed out twice");
    }

    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_lockfree_pool_creation() {
        let config = LockFreePoolConfig::default();
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Verify pool was created successfully
        assert!(pool.stats.is_some());
    }

    // The pool addresses its backing region with 32-bit offsets (`AtomicU32`
    // bump pointer, `aligned_size as u32` advance). A pool > 4GB would let a
    // >= 4GB allocation pass the usize bounds check while the u32 advance
    // truncates — the next allocation would alias live memory. Construction
    // must reject such configs outright.
    #[test]
    #[cfg(target_pointer_width = "64")]
    fn test_pool_rejects_memory_size_over_4gb() {
        let config = LockFreePoolConfig {
            memory_size: 5 * 1024 * 1024 * 1024, // 5GB
            ..Default::default()
        };
        let result = LockFreeMemoryPool::new(config);
        assert!(result.is_err(), "memory_size > u32::MAX must be rejected");

        // Exactly u32::MAX is still representable and must be accepted by the
        // guard (allocation itself may fail on low-memory machines, but it
        // must not fail with the 32-bit-offset validation error).
        let config = LockFreePoolConfig {
            memory_size: u32::MAX as usize,
            ..Default::default()
        };
        if let Err(e) = LockFreeMemoryPool::new(config) {
            assert!(
                !e.to_string().contains("32-bit offset"),
                "u32::MAX-sized pool must pass the offset-limit guard, got: {e}"
            );
        }
    }

    #[test]
    fn test_basic_allocation_deallocation() {
        let config = LockFreePoolConfig::default();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // Test small allocation
        let ptr = pool.allocate(64).unwrap();

        // Test deallocation
        pool.deallocate(ptr, 64).unwrap();
    }

    /// Fast bins recycle blocks by size class, so every block in a bin must be
    /// carved at the class size. Carving at the request size let a freed
    /// 136-byte block satisfy a 144-byte request from the same bin and overrun
    /// the live block carved right behind it.
    #[test]
    fn test_fast_bin_reuse_never_hands_out_a_block_smaller_than_the_request() {
        let pool = LockFreeMemoryPool::new(LockFreePoolConfig::default()).unwrap();
        let small = pool.allocate(136).unwrap(); // size class 144
        let neighbour = pool.allocate(8).unwrap(); // bump-carved directly behind `small`
        pool.deallocate(small, 136).unwrap();
        let reused = pool.allocate(144).unwrap(); // same size class: recycles `small`

        let reused_start = reused.as_ptr() as usize;
        let neighbour_start = neighbour.as_ptr() as usize;
        assert!(
            reused_start + 144 <= neighbour_start || reused_start >= neighbour_start + 8,
            "144-byte block at {reused_start:#x} overlaps the live 8-byte block at {neighbour_start:#x}"
        );
    }

    /// The pool must never interpret bytes that belong to a user. With the
    /// free-list link stored inline, a user write racing a concurrent pop
    /// (simulated here by writing through the freed pointer) corrupted the
    /// bin: the next allocation followed the garbage link and failed.
    #[test]
    fn test_free_list_link_lives_outside_user_memory() {
        let pool = LockFreeMemoryPool::new(LockFreePoolConfig::default()).unwrap();
        let a = pool.allocate(16).unwrap();
        pool.deallocate(a, 16).unwrap();
        // SAFETY (test only): the arena outlives this write; this deliberately
        // violates the ownership protocol to model the racing user write.
        unsafe { std::ptr::write_bytes(a.as_ptr(), 0xFF, 16) };

        let b = pool.allocate(16).unwrap();
        assert_eq!(b, a, "recycled block expected");
        let c = pool.allocate(16).unwrap();
        assert_ne!(c, a, "bin was empty; a fresh block was expected");
        pool.deallocate(b, 16).unwrap();
        pool.deallocate(c, 16).unwrap();
    }

    /// Threads allocate, write every byte of, and free blocks of one size
    /// class. Under ThreadSanitizer (`make tsan_pool`) this is the oracle for
    /// the Treiber free-list read: a link stored in user memory races the
    /// writes below; a link in the block header does not.
    #[test]
    fn test_concurrent_alloc_write_free_same_class() {
        let pool = Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::default()).unwrap());
        let handles: Vec<_> = (0..4u8)
            .map(|t| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..2_000 {
                        let p = pool.allocate(64).unwrap();
                        // SAFETY: `p` is a live 64-byte block owned by this thread.
                        unsafe { std::ptr::write_bytes(p.as_ptr(), t, 64) };
                        pool.deallocate(p, 64).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_fast_bin_allocation() {
        let config = LockFreePoolConfig::default();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // Allocate and deallocate multiple blocks
        let mut ptrs = Vec::new();

        for i in 0..10 {
            let size = (i + 1) * 64;
            let ptr = pool.allocate(size).unwrap();
            ptrs.push((ptr, size));
        }

        // Deallocate all
        for (ptr, size) in ptrs {
            pool.deallocate(ptr, size).unwrap();
        }
    }

    /// D8. Was `#[ignore]`d "for release mode compatibility". What made it
    /// flaky was its assertion: `contention_ratio() < 0.5` is a performance
    /// heuristic about how often a CAS happened to fail, which depends on how
    /// the scheduler interleaves two threads and says nothing about whether
    /// the pool is correct. A disabled concurrency test on a lock-free
    /// allocator is exactly the gap that hid the free-list-in-user-memory
    /// defect, so the assertion is now the correctness property instead: the
    /// pool must never hand the same bytes to two threads at once, and must
    /// give every byte back.
    ///
    /// Deterministic by construction: the arena is sized so that every
    /// allocation of both rounds fits even if nothing were ever recycled, so
    /// an allocation failure is a real defect rather than a capacity
    /// coincidence, and `BackoffStrategy::None` keeps the run free of sleeps.
    #[test]
    fn test_concurrent_allocation() {
        const THREADS: usize = 4;
        const PER_THREAD: usize = 64;
        const MAX_SIZE: usize = 256;
        // Worst case: nothing is ever reused, across both rounds.
        const ARENA: usize = 2 * THREADS * PER_THREAD * (MAX_SIZE + BLOCK_HEADER) + ALIGN_SIZE;

        let pool = Arc::new(
            LockFreeMemoryPool::new(LockFreePoolConfig {
                memory_size: ARENA,
                max_cas_retries: 64,
                backoff_strategy: BackoffStrategy::None,
                enable_stats: true,
                ..LockFreePoolConfig::default()
            })
            .unwrap(),
        );

        // Two rounds: the second one can only pass if the first gave
        // everything back.
        for round in 0..2 {
            let mut handles = Vec::new();
            for thread_id in 0..THREADS {
                let pool = Arc::clone(&pool);
                handles.push(thread::spawn(move || {
                    let mut blocks = Vec::with_capacity(PER_THREAD);
                    for i in 0..PER_THREAD {
                        let size = 8 + (thread_id * 37 + i * 11) % (MAX_SIZE - 8);
                        let ptr = pool.allocate(size).unwrap_or_else(|e| {
                            panic!(
                                "round {round}, thread {thread_id}, \
                                 allocation {i} of {size} bytes: {e}"
                            )
                        });
                        // Own every byte of it, so that a block handed to two
                        // threads at once is a data race ThreadSanitizer can
                        // see (`make tsan_pool`).
                        // SAFETY: the pool just handed this block over and
                        // nothing else holds it; `size` bytes are ours.
                        unsafe { std::ptr::write_bytes(ptr.as_ptr(), thread_id as u8, size) };
                        blocks.push((ptr.as_ptr() as usize, size));
                    }
                    for &(addr, size) in &blocks {
                        // SAFETY: as above -- still ours, still `size` wide.
                        let ours = unsafe { std::slice::from_raw_parts(addr as *const u8, size) };
                        assert!(
                            ours.iter().all(|&byte| byte == thread_id as u8),
                            "round {round}, thread {thread_id}: block at \
                             {addr:#x} was written by someone else"
                        );
                    }
                    blocks
                }));
            }

            let mut everything: Vec<(usize, usize)> = Vec::new();
            for handle in handles {
                everything.extend(handle.join().unwrap());
            }

            // No two live blocks may overlap.
            everything.sort_unstable();
            for pair in everything.windows(2) {
                let (addr, size) = pair[0];
                let (next, _) = pair[1];
                assert!(
                    addr + size <= next,
                    "round {round}: the block at {addr:#x} ({size} bytes) \
                     overlaps the live block at {next:#x}"
                );
            }

            for (addr, size) in everything {
                let ptr = NonNull::new(addr as *mut u8).unwrap();
                pool.deallocate(ptr, size).unwrap();
            }
        }

        let stats = pool.stats().expect("stats enabled");
        println!(
            "CAS contention ratio: {:.2}%",
            stats.contention_ratio() * 100.0
        );
    }

    #[test]
    fn test_raii_allocation() {
        let config = LockFreePoolConfig::default();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        {
            let ptr = pool.allocate(128).unwrap();
            let _alloc = LockFreeAllocation::new(ptr, 128, Arc::clone(&pool));

            // Allocation should be automatically freed when going out of scope
        }

        // Pool should have higher deallocation count
        if let Some(stats) = pool.stats() {
            assert!(stats.fast_deallocs.load(Ordering::Relaxed) > 0);
        }
    }

    #[test]
    fn test_size_alignment() {
        let config = LockFreePoolConfig::default();
        let pool = LockFreeMemoryPool::new(config).unwrap();

        assert_eq!(pool.align_size(1), 8);
        assert_eq!(pool.align_size(8), 8);
        assert_eq!(pool.align_size(9), 16);
        assert_eq!(pool.align_size(15), 16);
        assert_eq!(pool.align_size(16), 16);
    }

    /// Regression: when CAS retries were exhausted, deallocate_to_fast_bin
    /// returned Err without recording the block anywhere, permanently
    /// leaking it from the pool. max_cas_retries=0 makes the *pop* budget
    /// zero, which is the harshest configuration there is.
    ///
    /// C3.18 strengthened the guarantee: the push is no longer budgeted at
    /// all, so the block goes back on its own bin rather than onto the large
    /// free list, where no fast-bin request could have reached it.
    #[test]
    fn test_deallocate_cas_exhaustion_recycles_into_the_bin() {
        let config = LockFreePoolConfig {
            max_cas_retries: 0,
            ..LockFreePoolConfig::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        let ptr = pool.allocate(64).unwrap();
        let bin_index = pool.size_to_bin_index(64).unwrap();
        pool.deallocate(ptr, 64)
            .expect("CAS exhaustion must recycle the block, not leak it");

        assert_eq!(
            pool.fast_bins[bin_index].count.load(Ordering::Relaxed),
            1,
            "the freed block must sit on its own bin, not on the large free list"
        );
    }

    /// D8. Was `#[ignore]`d "to prevent timeouts in release mode", and guarded
    /// by a five-second wall clock plus a "possible infinite loop" escape
    /// hatch. The loop it was afraid of was real: before the `checked_add` in
    /// `allocate_new_block`, the bump cursor wrapped and the pool never
    /// reported exhaustion. That is fixed and covered by
    /// `test_exhaustion_overflow_safety`, so this test can now state the exact
    /// number of blocks a 1 KiB arena holds -- no clock, no escape hatch.
    #[test]
    fn test_pool_exhaustion() {
        const ARENA: usize = 1024;
        const REQUEST: usize = 64;
        // The bump cursor starts at ALIGN_SIZE, and every block costs its
        // class width plus a header.
        const PER_BLOCK: usize = REQUEST + BLOCK_HEADER;
        const CAPACITY: usize = (ARENA - ALIGN_SIZE) / PER_BLOCK;

        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: ARENA,
            max_cas_retries: 3,
            backoff_strategy: BackoffStrategy::None,
            enable_cache_alignment: false,
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            enable_stats: false,
            enable_simd_optimization: false,
            zero_on_free: false,
            ..LockFreePoolConfig::default()
        })
        .unwrap();

        let mut blocks = Vec::new();
        for i in 0..CAPACITY {
            blocks.push(
                pool.allocate(REQUEST)
                    .unwrap_or_else(|e| panic!("block {i} of {CAPACITY}: {e}")),
            );
        }
        assert!(
            pool.allocate(REQUEST).is_err(),
            "the arena holds {CAPACITY} {REQUEST}-byte blocks; the next one \
             must be refused"
        );

        // Everything comes back, and the whole arena is allocatable again.
        for ptr in blocks.drain(..) {
            pool.deallocate(ptr, REQUEST).unwrap();
        }
        for i in 0..CAPACITY {
            blocks.push(
                pool.allocate(REQUEST)
                    .unwrap_or_else(|e| panic!("second round, block {i}: {e}")),
            );
        }
        assert!(pool.allocate(REQUEST).is_err());
    }

    /// C3.18. `max_cas_retries` is a contention budget, not a correctness
    /// parameter: exhausting it must never make an allocation report OOM while
    /// recycled blocks sit in the bin, and must never file a fast-bin block
    /// somewhere a fast-bin request cannot reach.
    ///
    /// `max_cas_retries = 0` makes the CAS loop body unreachable and so
    /// reproduces deterministically what Miri produces on the *default* config
    /// by failing `compare_exchange_weak` spuriously: the frees fall through to
    /// the large free list, where no <=8 KiB request can ever pick them up, and
    /// the second round of allocations then reports an exhausted arena.
    #[test]
    fn test_cas_budget_exhaustion_does_not_report_oom() {
        const ARENA: usize = 1024;
        const REQUEST: usize = 64;
        const PER_BLOCK: usize = REQUEST + BLOCK_HEADER;
        const CAPACITY: usize = (ARENA - ALIGN_SIZE) / PER_BLOCK;

        let pool = LockFreeMemoryPool::new(LockFreePoolConfig {
            memory_size: ARENA,
            max_cas_retries: 0,
            backoff_strategy: BackoffStrategy::None,
            enable_cache_alignment: false,
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            enable_stats: false,
            enable_simd_optimization: false,
            zero_on_free: false,
            ..LockFreePoolConfig::default()
        })
        .unwrap();

        let mut blocks = Vec::new();
        for i in 0..CAPACITY {
            blocks.push(
                pool.allocate(REQUEST)
                    .unwrap_or_else(|e| panic!("block {i} of {CAPACITY}: {e}")),
            );
        }
        for ptr in blocks.drain(..) {
            pool.deallocate(ptr, REQUEST).unwrap();
        }
        for i in 0..CAPACITY {
            blocks.push(pool.allocate(REQUEST).unwrap_or_else(|e| {
                panic!(
                    "every block was freed, so the arena must be fully \
                     allocatable again; block {i} of {CAPACITY}: {e}"
                )
            }));
        }
    }

    //==============================================================================
    // SIMD INTEGRATION TESTS
    //==============================================================================

    #[test]
    fn test_simd_free_block_scanning() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // Allocate some blocks
        let ptrs: Vec<_> = (0..10).map(|_| pool.allocate(128).unwrap()).collect();

        // Free some blocks
        for ptr in &ptrs[0..5] {
            pool.deallocate(*ptr, 128).unwrap();
        }

        // Test SIMD scanning finds free blocks
        let new_ptr = pool.allocate(128).unwrap();

        // Cleanup
        pool.deallocate(new_ptr, 128).unwrap();
        for ptr in &ptrs[5..10] {
            pool.deallocate(*ptr, 128).unwrap();
        }
    }

    #[test]
    fn test_popcnt_bitmap_operations() {
        let bitmap = [0xFFFFFFFFFFFFFFFF_u64, 0x0000000000000000_u64];
        let count: usize = bitmap.iter().map(|&bits| bits.count_ones() as usize).sum();

        assert_eq!(count, 64); // 64 bits set in first word
    }

    #[test]
    fn test_bulk_allocation_simd() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        let sizes = vec![64, 128, 256, 512, 1024];
        let ptrs = pool.allocate_bulk_simd(&sizes).unwrap();

        assert_eq!(ptrs.len(), sizes.len());

        // Cleanup
        for (ptr, size) in ptrs.iter().zip(&sizes) {
            pool.deallocate(*ptr, *size).unwrap();
        }
    }

    #[test]
    fn test_concurrent_simd_operations() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
        let mut handles = vec![];

        // Spawn threads doing SIMD operations concurrently
        for _ in 0..4 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let sizes = vec![64, 128, 256];
                let ptrs = pool_clone.allocate_bulk_simd(&sizes).unwrap();

                // Verify all allocations succeeded

                // Cleanup
                for (ptr, size) in ptrs.iter().zip(&sizes) {
                    pool_clone.deallocate(*ptr, *size).unwrap();
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_simd_zeroing_on_free() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            zero_on_free: true,
            ..LockFreePoolConfig::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        let ptr = pool.allocate(256).unwrap();

        // SAFETY: test code, pointer valid from allocation
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0xFF, 256);
        }

        // Free with zeroing
        pool.deallocate_with_zero(ptr, 256).unwrap();

        // Reallocate and verify behavior
        let new_ptr = pool.allocate(256).unwrap();

        // Cleanup
        pool.deallocate(new_ptr, 256).unwrap();
    }

    #[test]
    fn test_simd_optimization_disabled() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: false,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // Test that operations still work without SIMD
        let sizes = vec![64, 128, 256];
        let ptrs = pool.allocate_bulk_simd(&sizes).unwrap();

        assert_eq!(ptrs.len(), sizes.len());

        // Cleanup
        for (ptr, size) in ptrs.iter().zip(&sizes) {
            pool.deallocate(*ptr, *size).unwrap();
        }
    }

    #[test]
    fn test_bulk_allocation_with_prefetch() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            enable_cache_alignment: true,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // Large bulk allocation to test prefetching
        let sizes: Vec<usize> = (0..20).map(|i| 64 + i * 32).collect();
        let ptrs = pool.allocate_bulk_simd(&sizes).unwrap();

        assert_eq!(ptrs.len(), sizes.len());

        // Cleanup
        for (ptr, size) in ptrs.iter().zip(&sizes) {
            pool.deallocate(*ptr, *size).unwrap();
        }
    }

    #[test]
    fn test_simd_zeroing_performance() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            zero_on_free: true,
            enable_stats: true,
            ..LockFreePoolConfig::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Test different sizes
        let sizes = vec![64, 256, 1024, 4096];

        for size in sizes {
            let ptr = pool.allocate(size).unwrap();

            // SAFETY: test code, pointer valid from allocation
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0xAA, size);
            }

            // Free with SIMD zeroing
            pool.deallocate_with_zero(ptr, size).unwrap();
        }

        // Verify stats
        if let Some(stats) = pool.stats() {
            assert!(stats.fast_deallocs.load(Ordering::Relaxed) > 0);
        }
    }

    #[test]
    fn test_bitmap_counting_accuracy() {
        // Test various bitmap patterns
        let test_cases = vec![
            (vec![0xFFFFFFFFFFFFFFFF_u64], 64),
            (vec![0x0000000000000000_u64], 0),
            (vec![0xAAAAAAAAAAAAAAAA_u64], 32),
            (vec![0x5555555555555555_u64], 32),
            (vec![0xFFFFFFFFFFFFFFFF_u64, 0xFFFFFFFFFFFFFFFF_u64], 128),
            (vec![0x0000000000000001_u64], 1),
        ];

        for (bitmap, expected_count) in test_cases {
            let count: usize = bitmap.iter().map(|&bits| bits.count_ones() as usize).sum();
            assert_eq!(count, expected_count, "Failed for bitmap: {:?}", bitmap);
        }
    }

    #[test]
    fn test_concurrent_bulk_allocations() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            ..LockFreePoolConfig::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
        let mut handles = vec![];

        // Multiple threads doing bulk allocations
        for thread_id in 0..4 {
            let pool_clone = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let sizes: Vec<usize> = (0..10).map(|i| 64 + (thread_id + i) * 16).collect();
                let ptrs = pool_clone.allocate_bulk_simd(&sizes).unwrap();

                // Verify allocations
                assert_eq!(ptrs.len(), sizes.len());

                // Cleanup
                for (ptr, size) in ptrs.iter().zip(&sizes) {
                    pool_clone.deallocate(*ptr, *size).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_zero_on_free_with_reuse() {
        let config = LockFreePoolConfig {
            enable_simd_optimization: true,
            zero_on_free: true,
            ..LockFreePoolConfig::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Allocate, write, free with zeroing
        let ptr1 = pool.allocate(128).unwrap();
        // SAFETY: test code, pointer valid from allocation
        unsafe {
            std::ptr::write_bytes(ptr1.as_ptr(), 0xFF, 128);
        }
        pool.deallocate_with_zero(ptr1, 128).unwrap();

        // Allocate again - should reuse the freed block
        let ptr2 = pool.allocate(128).unwrap();

        // Cleanup
        pool.deallocate(ptr2, 128).unwrap();
    }

    #[test]
    fn test_exhaustion_overflow_safety() {
        let config = LockFreePoolConfig {
            memory_size: 1024, // 1KB small pool
            enable_stats: false,
            ..Default::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // 1. Exhaust the pool
        let mut allocs = Vec::new();
        while let Ok(ptr) = pool.allocate(64) {
            allocs.push(ptr);
        }

        // 2. Call allocate 10,000 times. Each call should fail with OutOfMemory.
        // If an overflow wraps around, it will return a valid pointer (which is a bug).
        for _ in 0..10_000 {
            assert!(pool.allocate(64).is_err());
        }
    }

    #[test]
    fn test_large_allocations_reuse() {
        let config = LockFreePoolConfig {
            memory_size: 65536, // 64KB
            enable_stats: false,
            ..Default::default()
        };
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());

        // 1. Allocate a large block
        let first_alloc = pool.allocate(10000).unwrap();
        let first_offset = pool.ptr_to_offset(first_alloc).unwrap();

        // 2. Deallocate it
        pool.deallocate(first_alloc, 10000).unwrap();

        // 3. Allocate another block of same size
        let second_alloc = pool.allocate(10000).unwrap();
        let second_offset = pool.ptr_to_offset(second_alloc).unwrap();

        // Verify offsets are identical (memory was reused)
        assert_eq!(first_offset, second_offset);
    }

    #[test]
    fn test_large_block_best_fit_selection() {
        let config = LockFreePoolConfig {
            memory_size: 256 * 1024,
            enable_stats: false,
            ..Default::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Allocate three large blocks of different sizes
        let small = pool.allocate(9000).unwrap();
        let medium = pool.allocate(16000).unwrap();
        let large = pool.allocate(32000).unwrap();

        let small_off = pool.ptr_to_offset(small).unwrap();
        let medium_off = pool.ptr_to_offset(medium).unwrap();

        // Free all three
        pool.deallocate(small, 9000).unwrap();
        pool.deallocate(medium, 16000).unwrap();
        pool.deallocate(large, 32000).unwrap();

        // Request a block that fits in small — best-fit should pick it
        let reused_small = pool.allocate(9000).unwrap();
        assert_eq!(pool.ptr_to_offset(reused_small).unwrap(), small_off);

        // Request a block that fits in medium (too big for small) — best-fit should pick medium
        let reused_medium = pool.allocate(15000).unwrap();
        assert_eq!(pool.ptr_to_offset(reused_medium).unwrap(), medium_off);
    }

    #[test]
    fn test_large_block_fallback_to_new_allocation() {
        let config = LockFreePoolConfig {
            memory_size: 128 * 1024,
            enable_stats: false,
            ..Default::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Allocate and free a large block
        let first = pool.allocate(10000).unwrap();
        let first_off = pool.ptr_to_offset(first).unwrap();
        pool.deallocate(first, 10000).unwrap();

        // Request a size larger than the free block — must fall back to new allocation
        let bigger = pool.allocate(20000).unwrap();
        let bigger_off = pool.ptr_to_offset(bigger).unwrap();
        assert_ne!(bigger_off, first_off);
    }

    #[test]
    fn test_offset_to_ptr_rejects_out_of_bounds() {
        let config = LockFreePoolConfig {
            memory_size: 1024,
            enable_stats: false,
            ..Default::default()
        };
        let pool = LockFreeMemoryPool::new(config).unwrap();

        // Valid offset should succeed
        assert!(pool.offset_to_ptr(8).is_ok());

        // Offset at boundary should fail
        assert!(pool.offset_to_ptr(1024).is_err());

        // Offset beyond boundary should fail
        assert!(pool.offset_to_ptr(2000).is_err());

        // LIST_TAIL (0) should fail
        assert!(pool.offset_to_ptr(LIST_TAIL).is_err());
    }

    #[test]
    fn test_size_class_lut_matches_linear_search_and_tls_magazine() {
        let pool = LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap();
        for size in 1..=FAST_BIN_THRESHOLD {
            let expected = FAST_BIN_SIZES.iter().position(|&b| size <= b).unwrap();
            assert_eq!(pool.size_to_bin_index(size).unwrap(), expected);
        }
        assert!(pool.size_to_bin_index(FAST_BIN_THRESHOLD + 1).is_err());

        // Exercise TLS magazine fill + overflow into global bin + drain.
        let mut ptrs = Vec::with_capacity(MAGAZINE_CAP + 4);
        for _ in 0..(MAGAZINE_CAP + 4) {
            ptrs.push(pool.allocate(256).unwrap());
        }
        for ptr in ptrs.drain(..) {
            pool.deallocate(ptr, 256).unwrap();
        }
        for _ in 0..(MAGAZINE_CAP + 4) {
            ptrs.push(pool.allocate(256).unwrap());
        }
        for ptr in ptrs {
            pool.deallocate(ptr, 256).unwrap();
        }
    }

    #[test]
    fn test_s47_r4_alternating_pools_flush_magazine() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool_a = LockFreeMemoryPool::new(cfg.clone()).unwrap();
        let pool_b = LockFreeMemoryPool::new(cfg).unwrap();

        let iters = if cfg!(miri) { 64 } else { 4_096 };
        for _ in 0..iters {
            let mut va = Vec::with_capacity(8);
            for _ in 0..8 {
                va.push(pool_a.allocate(1024).unwrap());
            }
            for p in va {
                pool_a.deallocate(p, 1024).unwrap();
            }

            let mut vb = Vec::with_capacity(8);
            for _ in 0..8 {
                vb.push(pool_b.allocate(1024).unwrap());
            }
            for p in vb {
                pool_b.deallocate(p, 1024).unwrap();
            }
        }
    }

    #[test]
    fn test_s47_r4_thread_exit_flushes_magazine() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        // Pre-advance bump cursor near the end of the arena so only 16 blocks of 1024B fit.
        let room_for_16 = 16 * (1024 + BLOCK_HEADER);
        pool.next_offset
            .store((16 * 1024 * 1024 - room_for_16) as u32, Ordering::Relaxed);

        let rounds = if cfg!(miri) { 8 } else { 64 };
        for _ in 0..rounds {
            let p = Arc::clone(&pool);
            thread::spawn(move || {
                let mut v = Vec::with_capacity(8);
                for _ in 0..8 {
                    v.push(p.allocate(1024).unwrap());
                }
                for ptr in v {
                    p.deallocate(ptr, 1024).unwrap();
                }
            })
            .join()
            .unwrap();
        }
    }

    #[test]
    fn test_s47_r4_cross_thread_dealloc_no_leak() {
        struct SendPtr(NonNull<u8>);
        // SAFETY: Pool blocks are thread-safe to transfer and deallocate on another thread.
        unsafe impl Send for SendPtr {}

        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let room_for_32 = 32 * (1024 + BLOCK_HEADER);
        pool.next_offset
            .store((16 * 1024 * 1024 - room_for_32) as u32, Ordering::Relaxed);

        let rounds = if cfg!(miri) { 8 } else { 64 };
        for _ in 0..rounds {
            let mut batch = Vec::with_capacity(16);
            for _ in 0..16 {
                batch.push(SendPtr(pool.allocate(1024).unwrap()));
            }
            let p = Arc::clone(&pool);
            thread::spawn(move || {
                for SendPtr(ptr) in batch {
                    p.deallocate(ptr, 1024).unwrap();
                }
            })
            .join()
            .unwrap();
        }
    }

    #[test]
    fn test_s47_r4_live_thread_magazine_reclaimed_on_exhaustion() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let room_for_16 = 16 * (64 + BLOCK_HEADER);
        pool.next_offset
            .store((16 * 1024 * 1024 - room_for_16) as u32, Ordering::Relaxed);

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let (exit_tx, exit_rx) = std::sync::mpsc::channel();
        let p = Arc::clone(&pool);
        let helper = thread::spawn(move || {
            let mut v = Vec::with_capacity(16);
            for _ in 0..16 {
                v.push(p.allocate(64).unwrap());
            }
            for ptr in v {
                p.deallocate(ptr, 64).unwrap();
            }
            done_tx.send(()).unwrap();
            exit_rx.recv().unwrap();
        });

        done_rx.recv().unwrap();
        // Helper thread is still alive with all 16 blocks in its TLS magazine.
        let mut reacquired = Vec::with_capacity(16);
        for _ in 0..16 {
            reacquired.push(
                pool.allocate(64)
                    .expect("exhausted arena must drain live helper magazine"),
            );
        }
        for ptr in reacquired {
            pool.deallocate(ptr, 64).unwrap();
        }
        exit_tx.send(()).unwrap();
        helper.join().unwrap();
    }

    #[test]
    fn test_s47_r4_drop_pool_while_other_thread_magazine_holds_it() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = LockFreeMemoryPool::new(cfg).unwrap();
        let ptr = pool.allocate(64).unwrap();
        pool.deallocate(ptr, 64).unwrap();
        drop(pool);
        // Allocate from a fresh pool on the same thread to ensure stale magazine state was cleared.
        let mut cfg2 = LockFreePoolConfig::high_performance();
        cfg2.memory_size = 16 * 1024 * 1024;
        let pool2 = LockFreeMemoryPool::new(cfg2).unwrap();
        let ptr2 = pool2.allocate(64).unwrap();
        // SAFETY: `ptr2` is a live 64-byte allocation from `pool2`.
        unsafe {
            std::ptr::write_bytes(ptr2.as_ptr(), 0xAB, 64);
        }
        pool2.deallocate(ptr2, 64).unwrap();
    }

    #[test]
    fn test_s47_r11_rejects_pool_base_and_misaligned_deallocate() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = LockFreeMemoryPool::new(cfg).unwrap();
        let valid = pool.allocate(64).unwrap();

        // Pool base pointer (offset 0) and initial header (offset 8) must be rejected.
        assert!(pool.deallocate(pool.memory, 64).is_err());
        // SAFETY: `pool.memory.add(8)` is within the 16 MiB backing allocation.
        let header_ptr = NonNull::new(unsafe { pool.memory.as_ptr().add(ALIGN_SIZE) }).unwrap();
        assert!(pool.deallocate(header_ptr, 64).is_err());

        // Misaligned pointer inside the valid block must be rejected.
        // SAFETY: `valid.add(1)` is within the 64-byte allocated block.
        let misaligned = NonNull::new(unsafe { valid.as_ptr().add(1) }).unwrap();
        assert!(pool.deallocate(misaligned, 64).is_err());

        pool.deallocate(valid, 64).unwrap();
    }

    #[test]
    fn test_s48_r2_idle_threads_magazines_reclaimed_on_exhaustion() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let cap_blocks = if cfg!(miri) { 64usize } else { 512usize };
        let stride = 64 + BLOCK_HEADER;
        pool.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );

        let n_idle = if cfg!(miri) { 2usize } else { 16usize };
        let barrier = Arc::new(std::sync::Barrier::new(n_idle + 1));
        let done = Arc::new(std::sync::Barrier::new(n_idle + 1));
        let mut handles = Vec::with_capacity(n_idle);
        for _ in 0..n_idle {
            let p = Arc::clone(&pool);
            let b = Arc::clone(&barrier);
            let d = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                let mut v = Vec::with_capacity(16);
                for _ in 0..16 {
                    v.push(p.allocate(64).unwrap());
                }
                for ptr in v {
                    p.deallocate(ptr, 64).unwrap();
                }
                b.wait();
                d.wait();
            }));
        }
        barrier.wait();

        let mut held = Vec::with_capacity(cap_blocks);
        for i in 0..cap_blocks {
            held.push(
                pool.allocate(64)
                    .unwrap_or_else(|e| panic!("block {i}/{cap_blocks} stranded in idle TLS: {e}")),
            );
        }
        for ptr in held {
            pool.deallocate(ptr, 64).unwrap();
        }
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_s48_r4_tls_dtor_alloc_after_guard_teardown_no_leak() {
        struct AfterGuard(Option<Arc<LockFreeMemoryPool>>);
        impl Drop for AfterGuard {
            fn drop(&mut self) {
                if let Some(p) = self.0.take() {
                    let mut v = Vec::with_capacity(16);
                    for _ in 0..16 {
                        if let Ok(ptr) = p.allocate(64) {
                            v.push(ptr);
                        }
                    }
                    for ptr in v {
                        let _ = p.deallocate(ptr, 64);
                    }
                }
            }
        }
        thread_local! {
            static AFTER_GUARD: std::cell::RefCell<AfterGuard> =
                const { std::cell::RefCell::new(AfterGuard(None)) };
        }

        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let cap_blocks = 32usize;
        let stride = 64 + BLOCK_HEADER;
        pool.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );

        let rounds = if cfg!(miri) { 4usize } else { 16usize };
        for _ in 0..rounds {
            let p = Arc::clone(&pool);
            thread::spawn(move || {
                AFTER_GUARD.with(|ag| *ag.borrow_mut() = AfterGuard(Some(Arc::clone(&p))));
                let x = p.allocate(64).unwrap();
                p.deallocate(x, 64).unwrap();
            })
            .join()
            .unwrap();
        }

        let mut held = Vec::with_capacity(cap_blocks);
        for i in 0..cap_blocks {
            held.push(
                pool.allocate(64)
                    .unwrap_or_else(|e| panic!("block {i}/{cap_blocks} leaked in TLS dtor: {e}")),
            );
        }
        for ptr in held {
            pool.deallocate(ptr, 64).unwrap();
        }
    }

    #[test]
    fn test_s48_r9_rejects_fast_bin_double_free_and_interior_pointer() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = LockFreeMemoryPool::new(cfg).unwrap();

        let p = pool.allocate(64).unwrap();
        pool.deallocate(p, 64).unwrap();
        assert!(
            pool.deallocate(p, 64).is_err(),
            "double-free of fast-bin block must return Err"
        );

        // Unallocated pointer inside pool memory must be rejected.
        // SAFETY: `p.add(256)` is within the 16 MiB backing allocation.
        let fake = NonNull::new(unsafe { p.as_ptr().add(256) }).unwrap();
        assert!(
            pool.deallocate(fake, 64).is_err(),
            "unallocated pointer inside pool memory must return Err"
        );

        // Aligned interior pointer inside a live 256-byte fast-bin block must be rejected.
        let big = pool.allocate(256).unwrap();
        // SAFETY: `big.add(64)` is within the 256-byte allocated block.
        let interior = NonNull::new(unsafe { big.as_ptr().add(64) }).unwrap();
        assert!(
            pool.deallocate(interior, 64).is_err(),
            "interior pointer inside 256B block must return Err"
        );
        pool.deallocate(big, 256).unwrap();
    }

    #[test]
    fn test_s49_r1_cross_pool_stale_registration_no_duplicate_blocks() {
        let mut cfg_a = LockFreePoolConfig::high_performance();
        cfg_a.memory_size = 16 * 1024 * 1024;
        let pool_a = Arc::new(LockFreeMemoryPool::new(cfg_a).unwrap());

        let mut cfg_b = LockFreePoolConfig::high_performance();
        cfg_b.memory_size = 16 * 1024 * 1024;
        let pool_b = Arc::new(LockFreeMemoryPool::new(cfg_b).unwrap());

        let cap_blocks = if cfg!(miri) { 64usize } else { 256usize };
        let stride = 64 + BLOCK_HEADER;
        pool_a.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );
        pool_b.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );

        let parked = Arc::new(std::sync::Barrier::new(2));
        let done = Arc::new(std::sync::Barrier::new(2));
        let pa = Arc::clone(&pool_a);
        let pb = Arc::clone(&pool_b);
        let p_bar = Arc::clone(&parked);
        let d_bar = Arc::clone(&done);

        let helper = thread::spawn(move || {
            // Touch Pool A first so the thread registers with Pool A.
            let a_ptr = pa.allocate(64).unwrap();
            pa.deallocate(a_ptr, 64).unwrap();

            // Switch to Pool B and cache 16 blocks in the thread's magazine for Pool B.
            let mut b_ptrs = Vec::with_capacity(16);
            for _ in 0..16 {
                b_ptrs.push(pb.allocate(64).unwrap());
            }
            for ptr in b_ptrs {
                pb.deallocate(ptr, 64).unwrap();
            }
            p_bar.wait();
            d_bar.wait();
        });

        parked.wait();

        // Exhaust Pool A: must get exactly `cap_blocks` unique pointers and zero duplicates.
        let mut a_live = Vec::with_capacity(cap_blocks);
        let mut a_addrs = std::collections::HashSet::with_capacity(cap_blocks);
        while let Ok(ptr) = pool_a.allocate(64) {
            assert!(
                a_addrs.insert(ptr.as_ptr() as usize),
                "Pool A handed out duplicate live block {:p} after cross-pool switch",
                ptr.as_ptr()
            );
            a_live.push(ptr);
        }
        assert_eq!(a_live.len(), cap_blocks);

        // Pool B must still have all `cap_blocks` blocks recoverable.
        let mut b_live = Vec::with_capacity(cap_blocks);
        let mut b_addrs = std::collections::HashSet::with_capacity(cap_blocks);
        while let Ok(ptr) = pool_b.allocate(64) {
            assert!(
                b_addrs.insert(ptr.as_ptr() as usize),
                "Pool B handed out duplicate live block {:p}",
                ptr.as_ptr()
            );
            b_live.push(ptr);
        }
        assert_eq!(b_live.len(), cap_blocks);

        for ptr in a_live {
            pool_a.deallocate(ptr, 64).unwrap();
        }
        for ptr in b_live {
            pool_b.deallocate(ptr, 64).unwrap();
        }
        done.wait();
        helper.join().unwrap();
    }

    #[test]
    fn test_s49_r2_owner_pop_vs_steal_no_duplicate_blocks() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let cap_blocks = if cfg!(miri) { 32usize } else { 64usize };
        let stride = 64 + BLOCK_HEADER;
        pool.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );

        let iters = if cfg!(miri) { 8usize } else { 400usize };
        let start = Arc::new(std::sync::Barrier::new(2));
        let p_owner = Arc::clone(&pool);
        let s_owner = Arc::clone(&start);

        let owner = thread::spawn(move || {
            s_owner.wait();
            let mut batch = Vec::with_capacity(16);
            for _ in 0..iters {
                for _ in 0..16 {
                    if let Ok(ptr) = p_owner.allocate(64) {
                        // Stamp owner tag in user payload to detect concurrent ownership.
                        // SAFETY: `ptr` is a 64-byte block exclusively owned by this thread.
                        let word = unsafe { &*(ptr.as_ptr() as *const AtomicU64) };
                        assert_eq!(
                            word.swap(1, Ordering::AcqRel),
                            0,
                            "block handed to owner while already owned by stealer"
                        );
                        batch.push(ptr);
                    }
                }
                for ptr in batch.drain(..) {
                    // SAFETY: `ptr` is still owned by this thread before `deallocate`.
                    let word = unsafe { &*(ptr.as_ptr() as *const AtomicU64) };
                    assert_eq!(word.swap(0, Ordering::AcqRel), 1);
                    p_owner
                        .deallocate(ptr, 64)
                        .expect("owner deallocate failed after pop/steal race");
                }
            }
        });

        start.wait();
        let mut stolen = Vec::with_capacity(cap_blocks);
        for _ in 0..iters {
            pool.arena.drain_all_magazines();
            for _ in 0..16 {
                if let Ok(ptr) = pool.allocate(64) {
                    // SAFETY: `ptr` is a 64-byte block exclusively owned by this thread.
                    let word = unsafe { &*(ptr.as_ptr() as *const AtomicU64) };
                    assert_eq!(
                        word.swap(2, Ordering::AcqRel),
                        0,
                        "block handed to stealer while already owned by owner"
                    );
                    stolen.push(ptr);
                }
            }
            for ptr in stolen.drain(..) {
                // SAFETY: `ptr` is still owned by this thread before `deallocate`.
                let word = unsafe { &*(ptr.as_ptr() as *const AtomicU64) };
                assert_eq!(word.swap(0, Ordering::AcqRel), 2);
                pool.deallocate(ptr, 64)
                    .expect("stealer deallocate failed after pop/steal race");
            }
        }
        owner.join().unwrap();
    }

    #[test]
    fn test_s49_r3_cross_thread_fast_bin_double_free_rejected() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());

        let freed_bar = Arc::new(std::sync::Barrier::new(2));
        let done_bar = Arc::new(std::sync::Barrier::new(2));
        let p_t1 = Arc::clone(&pool);
        let f_t1 = Arc::clone(&freed_bar);
        let d_t1 = Arc::clone(&done_bar);

        let (tx, rx) = std::sync::mpsc::channel::<usize>();
        let t1 = thread::spawn(move || {
            let ptr = p_t1.allocate(64).unwrap();
            let addr = ptr.as_ptr() as usize;
            // Free `ptr` on `t1` so it is cached in `t1`'s TLS magazine.
            p_t1.deallocate(ptr, 64).unwrap();
            tx.send(addr).unwrap();
            f_t1.wait();
            d_t1.wait();
        });

        let addr = rx.recv().unwrap();
        freed_bar.wait();

        // Cross-thread double-free of `ptr` while it is cached in `t1`'s magazine must fail.
        let dup_ptr = NonNull::new(addr as *mut u8).unwrap();
        assert!(
            pool.deallocate(dup_ptr, 64).is_err(),
            "cross-thread double-free of magazine-cached block must return Err"
        );

        done_bar.wait();
        t1.join().unwrap();
    }

    #[test]
    fn test_s49_r4_exhausted_pool_failing_allocate_skips_clean_magazines() {
        let mut cfg = LockFreePoolConfig::high_performance();
        cfg.memory_size = 16 * 1024 * 1024;
        let pool = Arc::new(LockFreeMemoryPool::new(cfg).unwrap());
        let cap_blocks = 16usize;
        let stride = 64 + BLOCK_HEADER;
        pool.next_offset.store(
            (16 * 1024 * 1024 - cap_blocks * stride) as u32,
            Ordering::Relaxed,
        );

        let n_threads = if cfg!(miri) { 2usize } else { 32usize };
        let parked = Arc::new(std::sync::Barrier::new(n_threads + 1));
        let done = Arc::new(std::sync::Barrier::new(n_threads + 1));
        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let p = Arc::clone(&pool);
            let pk = Arc::clone(&parked);
            let dn = Arc::clone(&done);
            handles.push(thread::spawn(move || {
                if let Ok(ptr) = p.allocate(64) {
                    p.deallocate(ptr, 64).unwrap();
                }
                pk.wait();
                dn.wait();
            }));
        }
        parked.wait();

        let mut held = Vec::with_capacity(cap_blocks);
        while let Ok(ptr) = pool.allocate(64) {
            held.push(ptr);
        }
        assert_eq!(held.len(), cap_blocks);
        assert_eq!(
            pool.arena.dirty_magazines.load(Ordering::Acquire),
            0,
            "all magazines must be marked clean after exhaustion drain"
        );

        let fail_iters = if cfg!(miri) { 8usize } else { 1_000usize };
        for _ in 0..fail_iters {
            assert!(pool.allocate(64).is_err());
        }

        for ptr in held {
            pool.deallocate(ptr, 64).unwrap();
        }
        done.wait();
        for h in handles {
            h.join().unwrap();
        }
    }
}
