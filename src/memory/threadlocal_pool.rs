//! Thread-local memory pool caching for reduced contention
//!
//! This module provides per-instance thread-local slab caching that reduces contention
//! on global allocators by maintaining per-thread allocation caches.
//!
//! # Architecture
//!
//! - **Per-Pool TLS Ownership**: Each [`ThreadLocalMemoryPool`] owns its own
//!   per-thread cache table so independent pools never share slabs or statistics
//! - **Reference-Counted Slabs**: Replaced, cleared, or slot-recycled slabs stay
//!   alive via atomic reference counting until every live allocation drops
//! - **Size-Class Carving**: Bump and free-list paths both carve full size-class
//!   quanta (`16..=4096` bytes, 16-byte aligned)
//! - **Standalone Oversize Path**: Requests above `arena_size / 4` or above the
//!   maximum size class allocate directly without borrowing the thread cache
//!
//! # Performance Benefits
//!
//! - **Zero-contention hot paths**: Slab allocations avoid cross-thread contention
//! - **Reduced cache misses**: Thread-local slabs and free lists stay hot in CPU cache
//! - **Batch overhead amortization**: Statistics synchronization happens in batches

use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, RefCell};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use thread_local::ThreadLocal;

/// Default arena size for thread-local allocation (2MB)
const DEFAULT_ARENA_SIZE: usize = 2 * 1024 * 1024;
/// Threshold for lazy synchronization (256KB)
const SYNC_THRESHOLD: isize = 256 * 1024;
/// Maximum number of cached chunks per size class
const MAX_CACHED_CHUNKS: usize = 64;
/// Alignment for slab arenas and standalone oversize allocations
const SLAB_ALIGN: usize = 16;
/// Size classes for thread-local caching (all multiples of `SLAB_ALIGN`)
const TLS_SIZE_CLASSES: [usize; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];

/// Sentinel stored in `OS_TID` once `OS_TID_GUARD::drop` runs during thread teardown.
const DEAD_OS_TID: u64 = u64::MAX;

/// Monotonic OS-thread identifier generator (`1 .. DEAD_OS_TID`).
static NEXT_OS_TID: AtomicU64 = AtomicU64::new(1);

struct OsTidGuard;

impl Drop for OsTidGuard {
    fn drop(&mut self) {
        let _ = OS_TID.try_with(|cell| cell.set(DEAD_OS_TID));
    }
}

thread_local! {
    static OS_TID: Cell<u64> = const { Cell::new(0) };
    static OS_TID_GUARD: OsTidGuard = const { OsTidGuard };
}

const _: () = assert!(!std::mem::needs_drop::<Cell<u64>>());

enum TidState {
    Live(u64),
    Uninit,
    Dead,
}

#[inline]
fn probe_os_tid() -> TidState {
    OS_TID
        .try_with(|cell| match cell.get() {
            0 => TidState::Uninit,
            DEAD_OS_TID => TidState::Dead,
            id => TidState::Live(id),
        })
        .unwrap_or(TidState::Dead)
}

/// Initializes `OS_TID` and registers `OS_TID_GUARD` on the calling thread.
///
/// Must only be called after `self.caches.get_or(...)` so `OS_TID_GUARD` is
/// registered strictly after `thread_local`'s `ThreadGuard` and runs before
/// `ThreadGuard::drop` during LIFO thread-local teardown.
#[cold]
fn init_os_tid() -> u64 {
    OS_TID
        .try_with(|cell| {
            if OS_TID_GUARD.try_with(|_| ()).is_err() {
                cell.set(DEAD_OS_TID);
                return 0;
            }
            let next = NEXT_OS_TID.fetch_add(1, Ordering::Relaxed);
            cell.set(next);
            next
        })
        .unwrap_or(0)
}

/// Reads the calling thread's ID without registering any TLS destructor; returns `0`
/// if uninitialized or once `OS_TID_GUARD::drop` has marked teardown (`DEAD_OS_TID`).
#[inline]
fn live_os_tid() -> u64 {
    match probe_os_tid() {
        TidState::Live(id) => id,
        TidState::Uninit | TidState::Dead => 0,
    }
}

/// Returns the smallest `TLS_SIZE_CLASSES` index whose block size is `>= size`.
#[inline]
fn size_to_class_index(size: usize) -> Option<usize> {
    TLS_SIZE_CLASSES
        .iter()
        .position(|&class_size| size <= class_size)
}

/// Configuration for thread-local memory pool
#[derive(Debug, Clone)]
pub struct ThreadLocalPoolConfig {
    /// Size of arena allocated per thread (`1..=u32::MAX`)
    pub arena_size: usize,
    /// Retained for configuration compatibility; [`ThreadLocalMemoryPool`] grows
    /// its per-thread cache table dynamically as threads access the pool.
    pub max_threads: usize,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Synchronization threshold for batch updates
    pub sync_threshold: isize,
    /// Maximum cached chunks per size class
    pub max_cached_chunks: usize,
    /// Retained for configuration compatibility; currently inert on the allocation
    /// path because [`ThreadLocalMemoryPool`] serves slab and standalone oversize
    /// allocations from the system allocator (when `true`, [`ThreadLocalMemoryPool::new`]
    /// still validates and initializes the [`SecureMemoryPool`] instance).
    pub use_secure_memory: bool,
}

impl Default for ThreadLocalPoolConfig {
    fn default() -> Self {
        Self {
            arena_size: DEFAULT_ARENA_SIZE,
            max_threads: 256,
            enable_stats: true,
            sync_threshold: SYNC_THRESHOLD,
            max_cached_chunks: MAX_CACHED_CHUNKS,
            use_secure_memory: true,
        }
    }
}

impl ThreadLocalPoolConfig {
    /// Create configuration for high-performance scenarios
    pub fn high_performance() -> Self {
        Self {
            arena_size: 8 * 1024 * 1024, // 8MB per thread
            max_threads: 1024,
            enable_stats: false,
            sync_threshold: 1024 * 1024, // 1MB threshold
            max_cached_chunks: 128,
            use_secure_memory: false, // Skip security for max performance
        }
    }

    /// Create configuration for memory-constrained scenarios
    pub fn compact() -> Self {
        Self {
            arena_size: 512 * 1024, // 512KB per thread
            max_threads: 64,
            enable_stats: true,
            sync_threshold: 64 * 1024, // 64KB threshold
            max_cached_chunks: 32,
            use_secure_memory: true,
        }
    }
}

/// Statistics for thread-local pool operations
#[derive(Debug, Default)]
pub struct ThreadLocalPoolStats {
    /// Thread-local cache hits
    pub cache_hits: AtomicU64,
    /// Thread-local cache misses (standalone oversize allocations)
    pub cache_misses: AtomicU64,
    /// Arena allocations
    pub arena_allocations: AtomicU64,
    /// Chunks allocated from hot area
    pub hot_allocations: AtomicU64,
    /// Batch synchronizations with global pool
    pub batch_syncs: AtomicU64,
    /// Total memory in thread-local caches
    pub cached_memory: AtomicU64,
}

impl ThreadLocalPoolStats {
    /// Get cache hit ratio (0.0 to 1.0)
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get average allocation locality (higher is better)
    pub fn locality_score(&self) -> f64 {
        let hot = self.hot_allocations.load(Ordering::Relaxed);
        let arena = self.arena_allocations.load(Ordering::Relaxed);
        let total = hot + arena;
        if total == 0 {
            0.0
        } else {
            hot as f64 / total as f64
        }
    }
}

/// Contiguous arena slab owned via atomic reference counting over a raw pointer
/// (`Box::into_raw` / `Box::from_raw`) so container moves never retag `Slab`
/// under Stacked or Tree Borrows and cross-thread TLS teardowns are race-free.
struct Slab {
    base: NonNull<u8>,
    layout: Layout,
    owner_tid: u64,
    /// `+1` while installed in `ThreadCache::current`, `+1` per live `ThreadLocalAllocation`.
    refs: AtomicUsize,
}

impl Slab {
    /// Allocates a new zero-initialized slab with `refs == 2` (`1` for
    /// `ThreadCache::current` and `1` for the first carved `ThreadLocalAllocation`).
    fn new_for_first_alloc(arena_size: usize, owner_tid: u64) -> Result<NonNull<Self>> {
        if arena_size == 0 || arena_size > u32::MAX as usize {
            return Err(ZiporaError::invalid_data(
                "arena_size must be in 1..=u32::MAX",
            ));
        }
        let layout = Layout::from_size_align(arena_size, SLAB_ALIGN)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {e}")))?;

        // SAFETY: `arena_size > 0` checked above guarantees `layout.size() > 0`.
        let base = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(arena_size))?;

        let slab = Self {
            base,
            layout,
            owner_tid,
            refs: AtomicUsize::new(2),
        };

        let raw = Box::into_raw(Box::new(slab));
        // SAFETY: `Box::into_raw` always returns a non-null pointer.
        Ok(unsafe { NonNull::new_unchecked(raw) })
    }

    #[inline]
    fn inc_ref(slab: NonNull<Self>) {
        // SAFETY: Caller holds a live reference (`refs >= 1`) via `ThreadCache::current`.
        unsafe { slab.as_ref() }
            .refs
            .fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn dec_ref(slab: NonNull<Self>) {
        let prev = {
            // SAFETY: Caller owns one counted reference (`refs >= 1`) to `slab`.
            unsafe { slab.as_ref() }.refs.fetch_sub(1, Ordering::AcqRel)
        };
        if prev == 1 {
            // SAFETY: `prev == 1` with `AcqRel` ordering synchronizes with all prior
            // releases across threads and guarantees unique ownership; no `&Slab` is in scope.
            drop(unsafe { Box::from_raw(slab.as_ptr()) });
        }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        // SAFETY: `base` was allocated with `layout` in `Slab::new_for_first_alloc` and
        // `Slab::dec_ref` only drops `Slab` once `refs` reaches 0.
        unsafe {
            dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

struct ThreadCache {
    owner_tid: u64,
    // Raw ownership: every pointer comes from `Slab::new_for_first_alloc` (`Box::into_raw`)
    // and holds `+1` in `Slab::refs` while installed in `current`, released via
    // `Slab::dec_ref` in `evict_current` or `ThreadCache::drop`.
    current: Option<NonNull<Slab>>,
    pos: usize,
    free: [Vec<u32>; TLS_SIZE_CLASSES.len()],
    frag_inc: isize,
}

// SAFETY: `ThreadCache` lives inside `ThreadLocal<RefCell<ThreadCache>>`, which only grants
// `&RefCell<ThreadCache>` to one thread at a time (slot recycling occurs only after
// `ThreadGuard::drop` synchronizes via `THREAD_ID_MANAGER`'s mutex, and any late TLS
// destructor on the exiting thread only touches atomic `Slab::refs`). Cross-thread `Drop`
// of `ThreadCache` occurs only when the last `Arc<ThreadLocalMemoryPool>` drops, releasing
// `current` via atomic `Slab::dec_ref`.
unsafe impl Send for ThreadCache {}

impl ThreadCache {
    fn new(owner_tid: u64) -> Self {
        Self {
            owner_tid,
            current: None,
            pos: 0,
            free: std::array::from_fn(|_| Vec::new()),
            frag_inc: 0,
        }
    }

    /// Clears all per-class free lists and returns the byte count removed.
    fn drain_free_bytes(&mut self) -> u64 {
        let mut removed = 0u64;
        for (i, list) in self.free.iter_mut().enumerate() {
            if !list.is_empty() {
                removed += (list.len() as u64) * (TLS_SIZE_CLASSES[i] as u64);
                list.clear();
            }
        }
        removed
    }

    /// Evicts `self.current`: clears `self.free`, resets `self.pos = 0`, and
    /// releases `current`'s `+1` reference via `Slab::dec_ref`.
    fn evict_current(&mut self, stats: Option<&Arc<ThreadLocalPoolStats>>) {
        let drained = self.drain_free_bytes();
        self.pos = 0;
        if drained > 0
            && let Some(stats) = stats
        {
            stats.cached_memory.fetch_sub(drained, Ordering::Relaxed);
        }
        if let Some(old) = self.current.take() {
            Slab::dec_ref(old);
        }
    }

    /// Ensures this cache slot is owned by `tid`; if recycled from an exited thread,
    /// evicts the previous owner's slab and resets per-thread counters.
    #[inline]
    fn ensure_owner(&mut self, tid: u64, stats: Option<&Arc<ThreadLocalPoolStats>>) {
        if self.owner_tid != tid {
            self.evict_current(stats);
            self.frag_inc = 0;
            self.owner_tid = tid;
        }
    }

    fn allocate_in_class(
        &mut self,
        tid: u64,
        class: usize,
        arena_size: usize,
        stats: Option<&Arc<ThreadLocalPoolStats>>,
    ) -> Result<(NonNull<u8>, NonNull<Slab>)> {
        self.ensure_owner(tid, stats);

        let class_size = TLS_SIZE_CLASSES[class];

        if let Some(slab) = self.current {
            if let Some(offset) = self.free[class].pop() {
                Slab::inc_ref(slab);
                let base = {
                    // SAFETY: `slab` is valid with `refs >= 2` while installed in `self.current`.
                    unsafe { slab.as_ref() }.base.as_ptr()
                };
                // SAFETY: `offset` was carved inside `[0, arena_size - class_size]` from `base`,
                // which is non-null and valid for `arena_size` bytes.
                let ptr = unsafe { NonNull::new_unchecked(base.add(offset as usize)) };
                if let Some(stats) = stats {
                    stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                    stats
                        .cached_memory
                        .fetch_sub(class_size as u64, Ordering::Relaxed);
                }
                return Ok((ptr, slab));
            }

            if arena_size - self.pos >= class_size {
                let offset = self.pos;
                self.pos = offset + class_size;
                Slab::inc_ref(slab);
                let base = {
                    // SAFETY: `slab` is valid with `refs >= 2` while installed in `self.current`.
                    unsafe { slab.as_ref() }.base.as_ptr()
                };
                // SAFETY: `offset + class_size <= arena_size`, so `base + offset` is in-bounds and non-null.
                let ptr = unsafe { NonNull::new_unchecked(base.add(offset)) };
                if let Some(stats) = stats {
                    stats.hot_allocations.fetch_add(1, Ordering::Relaxed);
                }
                return Ok((ptr, slab));
            }

            self.evict_current(stats);
        }

        let slab = Slab::new_for_first_alloc(arena_size, tid)?;
        let ptr = {
            // SAFETY: `slab` was just allocated by `Slab::new_for_first_alloc`.
            unsafe { slab.as_ref() }.base
        };
        self.current = Some(slab);
        self.pos = class_size;

        if let Some(stats) = stats {
            stats.arena_allocations.fetch_add(1, Ordering::Relaxed);
            stats.hot_allocations.fetch_add(1, Ordering::Relaxed);
        }

        Ok((ptr, slab))
    }
}

impl Drop for ThreadCache {
    fn drop(&mut self) {
        if let Some(slab) = self.current.take() {
            Slab::dec_ref(slab);
        }
    }
}

/// Thread-local memory pool with per-instance slab caching.
///
/// Slab and standalone oversize allocations are served from the system allocator;
/// [`ThreadLocalPoolConfig::use_secure_memory`] is currently inert on the allocation path.
pub struct ThreadLocalMemoryPool {
    /// Configuration
    config: ThreadLocalPoolConfig,
    /// Global secure memory pool retained when `config.use_secure_memory` is enabled
    _global_pool: Option<Arc<SecureMemoryPool>>,
    /// Statistics (optional)
    stats: Option<Arc<ThreadLocalPoolStats>>,
    /// Per-instance thread-local slab caches
    caches: ThreadLocal<RefCell<ThreadCache>>,
}

impl ThreadLocalMemoryPool {
    /// Create new thread-local memory pool
    pub fn new(config: ThreadLocalPoolConfig) -> Result<Arc<Self>> {
        if config.arena_size == 0 || config.arena_size > u32::MAX as usize {
            return Err(ZiporaError::invalid_data(
                "arena_size must be in 1..=u32::MAX",
            ));
        }
        Layout::from_size_align(config.arena_size, SLAB_ALIGN)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid arena layout: {e}")))?;

        let global_pool = if config.use_secure_memory {
            let secure_config = SecurePoolConfig::medium_secure();
            Some(SecureMemoryPool::new(secure_config)?)
        } else {
            None
        };

        let stats = if config.enable_stats {
            Some(Arc::new(ThreadLocalPoolStats::default()))
        } else {
            None
        };

        Ok(Arc::new(Self {
            config,
            _global_pool: global_pool,
            stats,
            caches: ThreadLocal::new(),
        }))
    }

    /// Allocate memory using thread-local cache
    pub fn allocate(self: &Arc<Self>, size: usize) -> Result<ThreadLocalAllocation> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("Cannot allocate zero bytes"));
        }

        let arena_size = self.config.arena_size;
        let tid_state = probe_os_tid();
        let class_opt = if !matches!(tid_state, TidState::Dead) && size <= arena_size / 4 {
            size_to_class_index(size).filter(|&c| TLS_SIZE_CLASSES[c] <= arena_size)
        } else {
            None
        };

        let Some(class) = class_opt else {
            return self.allocate_standalone(size);
        };

        let cache_cell = self.caches.get_or(|| RefCell::new(ThreadCache::new(0)));
        let tid = match tid_state {
            TidState::Live(id) => id,
            TidState::Uninit => {
                let id = init_os_tid();
                if id == 0 {
                    return self.allocate_standalone(size);
                }
                id
            }
            TidState::Dead => return self.allocate_standalone(size),
        };
        let (ptr, slab) = cache_cell.borrow_mut().allocate_in_class(
            tid,
            class,
            arena_size,
            self.stats.as_ref(),
        )?;

        Ok(ThreadLocalAllocation {
            ptr,
            size,
            backing: Backing::Slab {
                slab,
                class: class as u8,
            },
            pool: Arc::clone(self),
        })
    }

    fn allocate_standalone(self: &Arc<Self>, size: usize) -> Result<ThreadLocalAllocation> {
        let layout = Layout::from_size_align(size, SLAB_ALIGN)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {e}")))?;

        // SAFETY: `size > 0` checked by caller guarantees `layout.size() > 0`.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(size))?;

        if let Some(stats) = &self.stats {
            stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        Ok(ThreadLocalAllocation {
            ptr,
            size,
            backing: Backing::Standalone(layout),
            pool: Arc::clone(self),
        })
    }

    /// Get pool statistics
    pub fn stats(&self) -> Option<Arc<ThreadLocalPoolStats>> {
        self.stats.clone()
    }

    /// Get memory usage statistics
    #[inline]
    pub fn memory_usage(&self) -> usize {
        if let Some(stats) = &self.stats {
            stats.cached_memory.load(Ordering::Relaxed) as usize
        } else {
            0
        }
    }

    /// Clear the calling thread's slab cache, releasing the cache's slab reference
    /// while any live allocations keep their backing slab alive until dropped.
    pub fn clear_caches(&self) {
        let tid = live_os_tid();
        if tid == 0 {
            return;
        }
        if let Some(cache_cell) = self.caches.get()
            && let Ok(mut cache) = cache_cell.try_borrow_mut()
        {
            cache.evict_current(self.stats.as_ref());
            cache.frag_inc = 0;
            cache.owner_tid = tid;
        }
    }
}

enum Backing {
    Slab { slab: NonNull<Slab>, class: u8 },
    Standalone(Layout),
}

/// RAII wrapper for thread-local pool allocations
pub struct ThreadLocalAllocation {
    ptr: NonNull<u8>,
    size: usize,
    backing: Backing,
    pool: Arc<ThreadLocalMemoryPool>,
}

impl ThreadLocalAllocation {
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
        // SAFETY: `self.ptr` points to a uniquely owned, initialized region (initially zeroed)
        // of at least `self.size` bytes (`size <= class_size` for slab blocks or `size` for standalone).
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get immutable slice view of allocation
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to a uniquely owned, initialized region (initially zeroed)
        // of at least `self.size` bytes (`size <= class_size` for slab blocks or `size` for standalone).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for ThreadLocalAllocation {
    fn drop(&mut self) {
        match self.backing {
            Backing::Standalone(layout) => {
                // SAFETY: `self.ptr` was allocated with `alloc_zeroed(layout)` in
                // `ThreadLocalMemoryPool::allocate` and is uniquely owned by `self`.
                unsafe {
                    dealloc(self.ptr.as_ptr(), layout);
                }
            }
            Backing::Slab { slab, class } => {
                let class_idx = usize::from(class);
                let class_size = TLS_SIZE_CLASSES[class_idx];
                let (base_addr, owner_tid) = {
                    // SAFETY: `slab` was created by `Slab::new_for_first_alloc` and this
                    // `ThreadLocalAllocation` holds a `+1` reference in `slab.refs`.
                    let s_ref = unsafe { slab.as_ref() };
                    (s_ref.base.as_ptr() as usize, s_ref.owner_tid)
                };

                let tid = live_os_tid();
                if tid != 0
                    && owner_tid == tid
                    && let Some(cache_cell) = self.pool.caches.get()
                    && let Ok(mut cache) = cache_cell.try_borrow_mut()
                    && cache.owner_tid == tid
                    && cache.current == Some(slab)
                {
                    let list = &mut cache.free[class_idx];
                    if list.len() < self.pool.config.max_cached_chunks {
                        let offset = (self.ptr.as_ptr() as usize - base_addr) as u32;
                        list.push(offset);

                        if let Some(stats) = &self.pool.stats {
                            stats
                                .cached_memory
                                .fetch_add(class_size as u64, Ordering::Relaxed);
                            cache.frag_inc = cache.frag_inc.saturating_sub(class_size as isize);
                            let neg_threshold =
                                self.pool.config.sync_threshold.max(0).saturating_neg();
                            if cache.frag_inc < neg_threshold {
                                stats.batch_syncs.fetch_add(1, Ordering::Relaxed);
                                cache.frag_inc = 0;
                            }
                        }
                    }
                }

                Slab::dec_ref(slab);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn test_threadlocal_pool_creation() {
        let config = ThreadLocalPoolConfig::default();
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        // Verify pool was created successfully
        assert!(pool.stats.is_some());
    }

    #[test]
    fn test_invalid_arena_size_rejected() {
        let zero_cfg = ThreadLocalPoolConfig {
            arena_size: 0,
            use_secure_memory: false,
            ..ThreadLocalPoolConfig::default()
        };
        assert!(ThreadLocalMemoryPool::new(zero_cfg).is_err());

        #[cfg(target_pointer_width = "64")]
        {
            let huge_cfg = ThreadLocalPoolConfig {
                arena_size: (u32::MAX as usize) + 1,
                use_secure_memory: false,
                ..ThreadLocalPoolConfig::default()
            };
            assert!(ThreadLocalMemoryPool::new(huge_cfg).is_err());
        }
    }

    #[test]
    fn test_basic_allocation_deallocation() {
        let config = ThreadLocalPoolConfig::default();
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        // Test allocation
        let alloc = pool.allocate(64).unwrap();
        assert_eq!(alloc.size(), 64);
        assert!(!alloc.as_ptr().is_null());

        // Allocation automatically freed on drop
    }

    #[test]
    fn test_thread_local_caching() {
        let config = ThreadLocalPoolConfig::default();
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        // Allocate, drop to populate free list, then reallocate to hit cache
        {
            let alloc1 = pool.allocate(64).unwrap();
            let _alloc2 = pool.allocate(128).unwrap();
            drop(alloc1);
            let _alloc3 = pool.allocate(64).unwrap(); // Should hit cache
        }

        // Check statistics
        if let Some(stats) = pool.stats() {
            let hits = stats.cache_hits.load(Ordering::Relaxed);
            assert!(hits >= 1);
        }
    }

    #[test]
    fn test_hot_area_allocation() {
        let config = ThreadLocalPoolConfig {
            arena_size: 4096, // Small arena for testing
            ..ThreadLocalPoolConfig::default()
        };
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        // Allocate many small blocks (should use hot area)
        let mut allocations = Vec::new();
        for i in 0..10 {
            let alloc = pool.allocate(32 + i).unwrap();
            allocations.push(alloc);
        }

        // Check statistics
        if let Some(stats) = pool.stats() {
            let hot_allocs = stats.hot_allocations.load(Ordering::Relaxed);
            let arena_allocs = stats.arena_allocations.load(Ordering::Relaxed);
            assert!(hot_allocs > 0);
            assert!(arena_allocs > 0);
        }
    }

    #[test]
    fn test_concurrent_thread_local_allocation() {
        let config = ThreadLocalPoolConfig::high_performance();
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        let mut handles = Vec::new();
        for t in 0..4u8 {
            let pool_clone = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                let mut allocations = Vec::new();
                for i in 0..10u8 {
                    let mut alloc = pool_clone.allocate(64 + usize::from(i)).unwrap();
                    alloc.as_mut_slice().fill(t ^ i);
                    allocations.push(alloc);
                }
                for (i, alloc) in allocations.iter().enumerate() {
                    assert!(alloc.as_slice().iter().all(|&b| b == (t ^ (i as u8))));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Reproduces LIFO TLS teardown + `thread_local::ThreadLocal` slot recycling:
    /// T1 registers `LATE_TLS` before its first `pool.allocate(64)` so `thread_local`'s
    /// `ThreadGuard::drop` runs BEFORE `LateTlsHolder::drop` on thread exit, returning
    /// T1's slot ID to `THREAD_ID_MANAGER` while T1 still holds a live allocation.
    /// T2 then claims that recycled slot and runs `allocate` + `clear_caches` + `allocate`
    /// concurrently with T1 reading, mutating, and dropping its surviving allocation.
    #[test]
    fn test_tls_destructor_slot_recycling_is_race_free() {
        struct LateTlsHolder {
            alloc: Option<ThreadLocalAllocation>,
            dtor_entered: Option<Arc<Barrier>>,
            overlap_barrier: Option<Arc<Barrier>>,
        }

        impl Drop for LateTlsHolder {
            fn drop(&mut self) {
                if let Some(entered) = self.dtor_entered.take() {
                    entered.wait();
                }
                if let Some(overlap) = self.overlap_barrier.take() {
                    overlap.wait();
                }
                if let Some(mut alloc) = self.alloc.take() {
                    assert!(alloc.as_slice().iter().all(|&b| b == 0xCC));
                    alloc.as_mut_slice().fill(0x33);
                    assert!(alloc.as_slice().iter().all(|&b| b == 0x33));
                    drop(alloc);
                }
            }
        }

        thread_local! {
            static LATE_TLS: RefCell<LateTlsHolder> = const {
                RefCell::new(LateTlsHolder {
                    alloc: None,
                    dtor_entered: None,
                    overlap_barrier: None,
                })
            };
        }

        let pool = ThreadLocalMemoryPool::new(ThreadLocalPoolConfig {
            arena_size: 4096,
            use_secure_memory: false,
            ..ThreadLocalPoolConfig::default()
        })
        .unwrap();

        let dtor_entered = Arc::new(Barrier::new(2));
        let overlap_barrier = Arc::new(Barrier::new(2));

        let pool_t1 = Arc::clone(&pool);
        let dtor_entered_t1 = Arc::clone(&dtor_entered);
        let overlap_t1 = Arc::clone(&overlap_barrier);
        let t1 = std::thread::spawn(move || {
            // Register `LATE_TLS` 1st on T1 before `thread_local` registers `THREAD_GUARD` 2nd.
            LATE_TLS.with(|cell| {
                let mut holder = cell.borrow_mut();
                holder.dtor_entered = Some(dtor_entered_t1);
                holder.overlap_barrier = Some(overlap_t1);
            });

            // Populate T1's free list and keep one live allocation in `LATE_TLS`.
            drop(pool_t1.allocate(64).unwrap());
            let mut survivor = pool_t1.allocate(64).unwrap();
            survivor.as_mut_slice().fill(0xCC);
            LATE_TLS.with(|cell| {
                cell.borrow_mut().alloc = Some(survivor);
            });
        });

        // Wait until T1 has run `ThreadGuard::drop` and entered `LateTlsHolder::drop`.
        dtor_entered.wait();

        let pool_t2 = Arc::clone(&pool);
        let overlap_t2 = Arc::clone(&overlap_barrier);
        let t2 = std::thread::spawn(move || {
            // Claim T1's recycled `ThreadLocal` slot before releasing T1 at `overlap_barrier`.
            let mut b1 = pool_t2.allocate(64).unwrap();
            b1.as_mut_slice().fill(0xAA);

            overlap_t2.wait();

            pool_t2.clear_caches();
            let mut b2 = pool_t2.allocate(64).unwrap();
            b2.as_mut_slice().fill(0xBB);
            assert!(b1.as_slice().iter().all(|&b| b == 0xAA));
            assert!(b2.as_slice().iter().all(|&b| b == 0xBB));
        });

        t2.join().unwrap();
        t1.join().unwrap();
    }

    #[test]
    fn test_size_class_mapping() {
        // Test size class mapping
        assert_eq!(size_to_class_index(8), Some(0)); // -> 16
        assert_eq!(size_to_class_index(16), Some(0)); // -> 16
        assert_eq!(size_to_class_index(17), Some(1)); // -> 32
        assert_eq!(size_to_class_index(64), Some(3)); // -> 64
        assert_eq!(size_to_class_index(5000), None); // Too large
    }

    #[test]
    fn test_cache_overflow() {
        let config = ThreadLocalPoolConfig {
            max_cached_chunks: 2, // Very small cache
            ..ThreadLocalPoolConfig::default()
        };
        let pool = ThreadLocalMemoryPool::new(config).unwrap();

        // Allocate and deallocate more than cache capacity
        for _ in 0..5 {
            let alloc = pool.allocate(64).unwrap();
            drop(alloc); // Force deallocation
        }

        // Should not crash and should fall back gracefully
    }

    #[test]
    fn test_different_configurations() {
        // Test high performance config
        let hp_config = ThreadLocalPoolConfig::high_performance();
        let hp_pool = ThreadLocalMemoryPool::new(hp_config).unwrap();
        assert!(hp_pool.stats.is_none()); // Stats disabled for performance

        // Test compact config
        let compact_config = ThreadLocalPoolConfig::compact();
        let compact_pool = ThreadLocalMemoryPool::new(compact_config).unwrap();
        assert!(compact_pool.stats.is_some()); // Stats enabled
        assert_eq!(compact_pool.config.arena_size, 512 * 1024);
    }
}
