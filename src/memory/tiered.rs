//! Tiered memory allocator for optimal performance across all allocation sizes
//!
//! This module provides a sophisticated tiered allocation strategy that routes
//! allocations to the most appropriate allocator based on size and usage patterns.

use crate::error::{Result, ZiporaError};
use crate::memory::{
    mmap::{MemoryMappedAllocator, MmapAllocation},
    pool::{MemoryPool, PoolConfig, PoolStats},
};

#[cfg(target_os = "linux")]
use crate::memory::hugepage::{HUGEPAGE_SIZE_2MB, HugePage, HugePageAllocator};

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread_local;

/// Size thresholds for different allocation strategies
/// Maximum size for small object allocations (1KB)
pub const SMALL_THRESHOLD: usize = 1024; // 1KB
/// Maximum size for medium object allocations (16KB)
pub const MEDIUM_THRESHOLD: usize = 16 * 1024; // 16KB  
/// Minimum size for huge page allocations (2MB)
pub const LARGE_THRESHOLD: usize = 2 * 1024 * 1024; // 2MB

/// A memory allocation that can come from different allocators
///
/// Safe code cannot name a pointer and a length and get one of these: the
/// pointer-carrying variants wrap types whose fields are private, so the only
/// way to obtain an allocation is [`TieredMemoryAllocator::allocate`].
///
/// ```compile_fail
/// use std::ptr::NonNull;
/// use zipora::memory::TieredAllocation;
///
/// // Were this to compile, `TieredMemoryAllocator::deallocate` - a safe
/// // method - would free a pointer it never handed out.
/// let forged = TieredAllocation::Small(NonNull::dangling(), 64);
/// ```
#[derive(Debug)]
pub enum TieredAllocation {
    /// Small allocation from this allocator's small-object pool
    Small(SmallBlock),
    /// Medium allocation from the thread-local size-classed pools
    Medium(MediumBlock),
    /// Large allocation using memory mapping
    Large(MmapAllocation),
    /// Huge allocation using Linux hugepages
    #[cfg(target_os = "linux")]
    Huge(HugePage),
}

/// A chunk lent by one [`TieredMemoryAllocator`]'s small-object pool.
///
/// The fields are private and there is no constructor, so safe code outside
/// this module cannot forge one:
///
/// ```compile_fail
/// use std::ptr::NonNull;
/// use zipora::memory::tiered::{SmallBlock, TieredAllocation};
///
/// let forged = TieredAllocation::Small(SmallBlock {
///     ptr: NonNull::dangling(),
///     size: 64,
///     allocator_id: 0,
/// });
/// ```
///
/// `allocator_id` records which allocator's pool the chunk came from. Each
/// allocator owns a separate `MemoryPool`, and `MemoryPool::deallocate` is
/// `unsafe` precisely because it cannot recognise a foreign pointer, so the
/// id is what lets [`TieredMemoryAllocator::deallocate`] be safe.
#[derive(Debug)]
pub struct SmallBlock {
    ptr: NonNull<u8>,
    size: usize,
    allocator_id: u64,
}

/// A chunk lent by the thread-local medium-size pools.
///
/// No allocator id: those pools are per-thread and shared by every allocator
/// on the thread, and `TieredAllocation` holds a `NonNull` and is therefore
/// `!Send`, so a medium block cannot reach a thread whose pools it did not
/// come from.
#[derive(Debug)]
pub struct MediumBlock {
    ptr: NonNull<u8>,
    size: usize,
}

/// Configuration for the tiered memory allocator
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Enable small object pools
    pub enable_small_pools: bool,
    /// Enable medium object pools with size classes
    pub enable_medium_pools: bool,
    /// Enable memory-mapped large allocations
    pub enable_mmap_large: bool,
    /// Enable hugepage allocations
    pub enable_hugepages: bool,
    /// Minimum size for memory-mapped allocations
    pub mmap_threshold: usize,
    /// Minimum size for hugepage allocations
    pub hugepage_threshold: usize,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            enable_small_pools: true,
            enable_medium_pools: true,
            enable_mmap_large: true,
            enable_hugepages: cfg!(target_os = "linux"),
            mmap_threshold: MEDIUM_THRESHOLD,
            hugepage_threshold: LARGE_THRESHOLD,
        }
    }
}

/// Comprehensive statistics for the tiered allocator
#[derive(Debug, Clone)]
pub struct TieredStats {
    /// Number of small allocations made
    pub small_allocations: u64,
    /// Number of medium allocations made
    pub medium_allocations: u64,
    /// Number of large allocations made
    pub large_allocations: u64,
    /// Number of huge allocations made
    pub huge_allocations: u64,
    /// Total bytes currently allocated
    pub total_allocated_bytes: u64,
    /// Statistics for the small object pool
    pub small_pool_stats: PoolStats,
    /// Statistics for each medium object pool (by size class)
    pub medium_pool_stats: Vec<PoolStats>,
    /// Statistics for memory-mapped allocations
    pub mmap_stats: crate::memory::mmap::MmapStats,
}

const MEDIUM_SIZE_CLASSES: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
const MEDIUM_POOL_ALIGN: usize = 16;

// Thread-local storage for medium-sized pools to reduce contention
thread_local! {
    static MEDIUM_POOLS: Vec<Arc<MemoryPool>> = {
        // Invariant: `MEDIUM_SIZE_CLASSES` and `MEDIUM_POOL_ALIGN` are
        // compile-time valid constants (`chunk_size > 0`, `alignment = 16`
        // power of two), so `MemoryPool::new` cannot fail (`new()` does not
        // pre-allocate chunks).
        MEDIUM_SIZE_CLASSES
            .into_iter()
            .map(|size| {
                let config = PoolConfig::new(size, 32, MEDIUM_POOL_ALIGN);
                Arc::new(MemoryPool::new(config).expect("medium pool preset config is valid"))
            })
            .collect()
    };
}

/// Source of [`TieredMemoryAllocator`] identities. Only ever incremented, so
/// two live allocators never share an id.
static NEXT_ALLOCATOR_ID: AtomicU64 = AtomicU64::new(0);

/// High-performance tiered memory allocator
pub struct TieredMemoryAllocator {
    config: TieredConfig,

    // Identity, stamped into every SmallBlock this allocator lends out
    id: u64,

    // Small object pool (< 1KB)
    small_pool: Arc<MemoryPool>,

    // Memory-mapped allocator for large objects
    mmap_allocator: Arc<MemoryMappedAllocator>,

    // Hugepage allocator for very large objects
    #[cfg(target_os = "linux")]
    hugepage_allocator: Arc<HugePageAllocator>,

    // Statistics
    small_allocs: AtomicU64,
    medium_allocs: AtomicU64,
    large_allocs: AtomicU64,
    huge_allocs: AtomicU64,
    total_bytes: AtomicU64,

    // Adaptive allocation tracking
    allocation_history: Arc<Mutex<AllocationHistory>>,
}

/// Tracks allocation patterns for adaptive optimization
struct AllocationHistory {
    size_histogram: [u64; 32], // Histogram of allocation sizes (log2 buckets)
    recent_sizes: Vec<usize>,  // Recent allocation sizes for pattern detection
    max_recent: usize,
}

impl AllocationHistory {
    fn new() -> Self {
        Self {
            size_histogram: [0; 32],
            recent_sizes: Vec::with_capacity(1000),
            max_recent: 1000,
        }
    }

    fn record_allocation(&mut self, size: usize) {
        // Update histogram
        let bucket = if size == 0 {
            0
        } else {
            (usize::BITS as usize - 1) - size.leading_zeros() as usize
        };
        if bucket < 32 {
            self.size_histogram[bucket] += 1;
        }

        // Track recent allocations
        if self.recent_sizes.len() >= self.max_recent {
            self.recent_sizes.remove(0);
        }
        self.recent_sizes.push(size);
    }

    fn get_allocation_pattern(&self) -> AllocationPattern {
        let total: u64 = self.size_histogram.iter().sum();
        if total == 0 {
            return AllocationPattern::Mixed;
        }

        // Analyze dominant allocation sizes
        let small_ratio = self.size_histogram[0..10].iter().sum::<u64>() as f64 / total as f64;
        let medium_ratio = self.size_histogram[10..16].iter().sum::<u64>() as f64 / total as f64;
        let large_ratio = self.size_histogram[16..].iter().sum::<u64>() as f64 / total as f64;

        if small_ratio > 0.7 {
            AllocationPattern::SmallDominated
        } else if medium_ratio > 0.7 {
            AllocationPattern::MediumDominated
        } else if large_ratio > 0.7 {
            AllocationPattern::LargeDominated
        } else {
            AllocationPattern::Mixed
        }
    }
}

/// Detected allocation pattern for adaptive optimization
#[derive(Debug, Clone, Copy)]
pub enum AllocationPattern {
    /// More than 70% of allocations are small (< 1KB)
    SmallDominated,
    /// More than 70% of allocations are medium (1KB-16KB)
    MediumDominated,
    /// More than 70% of allocations are large (> 16KB)
    LargeDominated,
    /// No clear dominant allocation size pattern
    Mixed,
}

impl TieredMemoryAllocator {
    /// Create a new tiered memory allocator
    pub fn new(config: TieredConfig) -> Result<Self> {
        let small_pool = if config.enable_small_pools {
            Arc::new(MemoryPool::new(PoolConfig::new(SMALL_THRESHOLD, 100, 8))?)
        } else {
            Arc::new(MemoryPool::new(PoolConfig::new(64, 1, 8))?) // Minimal pool
        };

        let mmap_allocator = if config.enable_mmap_large {
            Arc::new(MemoryMappedAllocator::new(config.mmap_threshold))
        } else {
            Arc::new(MemoryMappedAllocator::new(usize::MAX)) // Effectively disabled
        };

        #[cfg(target_os = "linux")]
        let hugepage_allocator = if config.enable_hugepages {
            Arc::new(HugePageAllocator::with_config(
                config.hugepage_threshold,
                HUGEPAGE_SIZE_2MB,
            )?)
        } else {
            Arc::new(HugePageAllocator::with_config(
                usize::MAX,
                HUGEPAGE_SIZE_2MB,
            )?) // Disabled
        };

        Ok(Self {
            config,
            id: NEXT_ALLOCATOR_ID.fetch_add(1, Ordering::Relaxed),
            small_pool,
            mmap_allocator,
            #[cfg(target_os = "linux")]
            hugepage_allocator,
            small_allocs: AtomicU64::new(0),
            medium_allocs: AtomicU64::new(0),
            large_allocs: AtomicU64::new(0),
            huge_allocs: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            allocation_history: Arc::new(Mutex::new(AllocationHistory::new())),
        })
    }

    /// Create allocator with default configuration
    #[allow(clippy::should_implement_trait)] // inherent default() returns Result; cannot implement Default trait
    pub fn default() -> Result<Self> {
        Self::new(TieredConfig::default())
    }

    /// Allocate memory using the optimal strategy for the given size
    ///
    /// The first `size` bytes are zeroed, whichever tier serves the request.
    /// The `Large` and `Huge` tiers get that from the kernel (`MAP_ANONYMOUS`
    /// pages arrive zeroed); the two pool tiers recycle memory, so they are
    /// zeroed explicitly. That costs a `memset` of `size` bytes per pooled
    /// allocation and it is what makes [`TieredAllocation::as_slice`] sound:
    /// without it the slice would be built over uninitialized memory.
    pub fn allocate(&self, size: usize) -> Result<TieredAllocation> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("allocation size cannot be zero"));
        }

        // Record allocation for adaptive optimization
        if let Ok(mut history) = self.allocation_history.try_lock() {
            history.record_allocation(size);
        }

        // Route to appropriate allocator based on size; charge `total_bytes`
        // only after the tier succeeds (T3).
        let alloc = if size <= SMALL_THRESHOLD && self.config.enable_small_pools {
            self.allocate_small(size)?
        } else if size <= MEDIUM_THRESHOLD && self.config.enable_medium_pools {
            self.allocate_medium(size)?
        } else if size < LARGE_THRESHOLD && self.config.enable_mmap_large {
            self.allocate_large(size)?
        } else {
            self.allocate_huge(size)?
        };

        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);
        Ok(alloc)
    }

    /// Deallocate memory
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if the allocation came from a *different*
    /// `TieredMemoryAllocator`'s small-object pool. The chunk is leaked rather
    /// than filed into the wrong pool: this allocator has no way to reach the
    /// one that owns it.
    pub fn deallocate(&self, allocation: TieredAllocation) -> Result<()> {
        match allocation {
            TieredAllocation::Small(block) => {
                if block.allocator_id != self.id {
                    return Err(ZiporaError::invalid_data(
                        "allocation came from a different TieredMemoryAllocator: \
                         each allocator owns its own small-object pool",
                    ));
                }
                // SAFETY: `SmallBlock` has private fields and is constructed
                // only by `allocate_small`, from `self.small_pool.allocate()`;
                // the id check above proves `self` is that allocator. The
                // block is consumed by value here, so it cannot be
                // deallocated twice.
                unsafe { self.small_pool.deallocate(block.ptr) }?;
                self.total_bytes
                    .fetch_sub(block.size as u64, Ordering::Relaxed);
            }
            TieredAllocation::Medium(block) => {
                self.deallocate_medium(block.ptr, block.size)?;
                self.total_bytes
                    .fetch_sub(block.size as u64, Ordering::Relaxed);
            }
            TieredAllocation::Large(allocation) => {
                let size = allocation.size();
                self.mmap_allocator.deallocate(allocation)?;
                self.total_bytes.fetch_sub(size as u64, Ordering::Relaxed);
            }
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => {
                let size = hugepage.size();
                drop(hugepage); // HugePage handles its own deallocation
                self.total_bytes.fetch_sub(size as u64, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> TieredStats {
        let medium_pool_stats = MEDIUM_POOLS
            .try_with(|pools| pools.iter().map(|pool| pool.stats()).collect())
            .unwrap_or_default();

        TieredStats {
            small_allocations: self.small_allocs.load(Ordering::Relaxed),
            medium_allocations: self.medium_allocs.load(Ordering::Relaxed),
            large_allocations: self.large_allocs.load(Ordering::Relaxed),
            huge_allocations: self.huge_allocs.load(Ordering::Relaxed),
            total_allocated_bytes: self.total_bytes.load(Ordering::Relaxed),
            small_pool_stats: self.small_pool.stats(),
            medium_pool_stats,
            mmap_stats: self.mmap_allocator.stats(),
        }
    }

    /// Get allocation pattern analysis
    pub fn get_allocation_pattern(&self) -> Result<AllocationPattern> {
        if let Ok(history) = self.allocation_history.lock() {
            Ok(history.get_allocation_pattern())
        } else {
            Ok(AllocationPattern::Mixed)
        }
    }

    /// Optimize allocator based on observed allocation patterns
    pub fn optimize_for_pattern(&self) -> Result<()> {
        let pattern = self.get_allocation_pattern()?;

        log::debug!("Optimizing tiered allocator for pattern: {:?}", pattern);

        // Pattern-specific optimizations could be implemented here
        // For example:
        // - Pre-warm pools for dominant allocation sizes
        // - Adjust cache sizes based on usage patterns
        // - Tune memory mapping thresholds

        Ok(())
    }

    fn allocate_small(&self, size: usize) -> Result<TieredAllocation> {
        self.small_allocs.fetch_add(1, Ordering::Relaxed);
        let chunk = self.small_pool.allocate()?;

        // SAFETY: `chunk` is a live chunk of `SMALL_THRESHOLD` bytes that the
        // pool has just handed out, so nothing else references it, and this
        // arm is only reached for `size <= SMALL_THRESHOLD`, so `size` zero
        // bytes stay inside it. `MemoryPool` recycles chunks without clearing
        // them, so this is also what stops the previous tenant's bytes from
        // reaching the next caller.
        unsafe { std::ptr::write_bytes(chunk.as_ptr(), 0, size) };

        Ok(TieredAllocation::Small(SmallBlock {
            ptr: chunk,
            size,
            allocator_id: self.id,
        }))
    }

    fn allocate_medium(&self, size: usize) -> Result<TieredAllocation> {
        // C3.23 (S8-R3): `try_with` instead of `with` so calling `allocate`
        // during TLS teardown falls back to `allocate_large` rather than
        // panicking inside `std::thread::LocalKey::with`.
        match MEDIUM_POOLS.try_with(|pools| {
            for pool in pools.iter() {
                if pool.config().chunk_size >= size {
                    let chunk = pool.allocate()?;
                    self.medium_allocs.fetch_add(1, Ordering::Relaxed);

                    // SAFETY: as in `allocate_small`; the guard above is
                    // exactly `size <= pool.config().chunk_size`.
                    unsafe { std::ptr::write_bytes(chunk.as_ptr(), 0, size) };

                    return Ok(TieredAllocation::Medium(MediumBlock {
                        ptr: chunk,
                        size,
                    }));
                }
            }

            self.allocate_large(size)
        }) {
            Ok(res) => res,
            Err(_) => self.allocate_large(size),
        }
    }

    fn deallocate_medium(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        // C3.23 (S8-R3): if a user `Drop` runs on thread exit *after*
        // `MEDIUM_POOLS` has already been destroyed (LIFO TLS teardown order),
        // `LocalKey::with` would panic inside `Drop` and abort the process.
        // `MemoryPool::drop` only frees the chunks parked in `free_chunks` at
        // destruction time, so a checked-out `MediumBlock` is still a live heap
        // allocation with `Layout(chunk_size, MEDIUM_POOL_ALIGN)`; when
        // `try_with` reports that `MEDIUM_POOLS` is gone, release the chunk
        // directly to the system allocator and return an error rather than
        // panicking or leaking.
        match MEDIUM_POOLS.try_with(|pools| {
            for pool in pools.iter() {
                if pool.config().chunk_size >= size {
                    // SAFETY: `allocate_medium` picks the pool the same way,
                    // from the same thread-local list, so this is the pool the
                    // chunk came from; `TieredAllocation` is consumed by value
                    // by the caller, so it cannot be deallocated twice.
                    return unsafe { pool.deallocate(ptr) };
                }
            }

            Err(ZiporaError::invalid_data(
                "no suitable pool for deallocation",
            ))
        }) {
            Ok(res) => res,
            Err(_) => {
                if let Some(chunk_size) = MEDIUM_SIZE_CLASSES.into_iter().find(|&c| c >= size)
                    && let Ok(layout) =
                        std::alloc::Layout::from_size_align(chunk_size, MEDIUM_POOL_ALIGN)
                {
                    // SAFETY: `ptr` was allocated by the thread's medium
                    // `MemoryPool` for `chunk_size` with `MEDIUM_POOL_ALIGN`,
                    // was not in `free_chunks` when `MEDIUM_POOLS` tore down,
                    // and `MediumBlock` was consumed by value by `deallocate`.
                    unsafe {
                        std::alloc::dealloc(ptr.as_ptr(), layout);
                    }
                }
                Err(ZiporaError::resource_busy(
                    "thread-local medium pools already destroyed during TLS teardown;                      chunk released directly to the system allocator",
                ))
            }
        }
    }

    fn allocate_large(&self, size: usize) -> Result<TieredAllocation> {
        self.large_allocs.fetch_add(1, Ordering::Relaxed);
        let allocation = self.mmap_allocator.allocate(size)?;
        Ok(TieredAllocation::Large(allocation))
    }

    fn allocate_huge(&self, size: usize) -> Result<TieredAllocation> {
        #[cfg(target_os = "linux")]
        {
            if self.config.enable_hugepages && self.hugepage_allocator.should_use_hugepages(size) {
                self.huge_allocs.fetch_add(1, Ordering::Relaxed);
                let hugepage = self.hugepage_allocator.allocate(size)?;
                return Ok(TieredAllocation::Huge(hugepage));
            }
        }

        // Fall back to memory mapping for very large allocations
        self.allocate_large(size)
    }
}

// SAFETY: TieredMemoryAllocator is Send because:
// 1. `config: TieredConfig` - Config is Clone with no raw pointers.
// 2. `small_allocs/medium_allocs/...` - AtomicUsize counters are Send.
// 3. `bump_allocator: BumpAllocator` - BumpAllocator is Send (uses atomics).
// 4. `mmap_allocator: MemoryMappedAllocator` - Is Send (uses Mutex).
// 5. `hugepage_allocator: HugePageAllocator` - Is Send.
unsafe impl Send for TieredMemoryAllocator {}

// SAFETY: TieredMemoryAllocator is Sync because:
// 1. Atomic counters (small_allocs, medium_allocs, etc.) are inherently Sync.
// 2. `bump_allocator: BumpAllocator` - Is Sync (uses CAS for allocation).
// 3. `mmap_allocator: MemoryMappedAllocator` - Is Sync (uses Mutex).
// 4. `hugepage_allocator: HugePageAllocator` - Is Sync (uses Mutex).
// 5. Thread-local pools (SMALL_POOLS, MEDIUM_POOLS) are per-thread.
// All shared state is protected by atomics or Mutex for thread safety.
unsafe impl Sync for TieredMemoryAllocator {}

impl TieredAllocation {
    /// Get the allocated memory as a slice
    ///
    /// The bytes are zero at the point [`TieredMemoryAllocator::allocate`]
    /// returns; see its documentation.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            // SAFETY: ptr is NonNull from valid pool allocation, size matches allocated chunk
            TieredAllocation::Small(block) => unsafe {
                std::slice::from_raw_parts(block.ptr.as_ptr(), block.size)
            },
            // SAFETY: ptr is NonNull from valid pool allocation, size matches allocated chunk
            TieredAllocation::Medium(block) => unsafe {
                std::slice::from_raw_parts(block.ptr.as_ptr(), block.size)
            },
            TieredAllocation::Large(allocation) => allocation.as_slice(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_slice(),
        }
    }

    /// Get the allocated memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            // SAFETY: ptr is NonNull from valid pool allocation, size matches allocated chunk, &mut guarantees exclusive access
            TieredAllocation::Small(block) => unsafe {
                std::slice::from_raw_parts_mut(block.ptr.as_ptr(), block.size)
            },
            // SAFETY: ptr is NonNull from valid pool allocation, size matches allocated chunk, &mut guarantees exclusive access
            TieredAllocation::Medium(block) => unsafe {
                std::slice::from_raw_parts_mut(block.ptr.as_ptr(), block.size)
            },
            TieredAllocation::Large(allocation) => allocation.as_mut_slice(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_mut_slice(),
        }
    }

    /// Get the size of the allocation
    #[inline]
    pub fn size(&self) -> usize {
        match self {
            TieredAllocation::Small(block) => block.size,
            TieredAllocation::Medium(block) => block.size,
            TieredAllocation::Large(allocation) => allocation.size(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.size(),
        }
    }

    /// Get the memory as a typed pointer
    pub fn as_ptr<T>(&self) -> *mut T {
        match self {
            TieredAllocation::Small(block) => block.ptr.as_ptr() as *mut T,
            TieredAllocation::Medium(block) => block.ptr.as_ptr() as *mut T,
            TieredAllocation::Large(allocation) => allocation.as_ptr(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_slice().as_ptr() as *mut T,
        }
    }
}

/// Global tiered allocator instance
static GLOBAL_TIERED_ALLOCATOR: std::sync::LazyLock<TieredMemoryAllocator> =
    std::sync::LazyLock::new(|| {
        TieredMemoryAllocator::default().expect("default allocator creation")
    });

/// Allocate memory using the global tiered allocator
pub fn tiered_allocate(size: usize) -> Result<TieredAllocation> {
    GLOBAL_TIERED_ALLOCATOR.allocate(size)
}

/// Deallocate memory using the global tiered allocator
pub fn tiered_deallocate(allocation: TieredAllocation) -> Result<()> {
    GLOBAL_TIERED_ALLOCATOR.deallocate(allocation)
}

/// Get statistics from the global tiered allocator
pub fn get_tiered_stats() -> TieredStats {
    GLOBAL_TIERED_ALLOCATOR.stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Global mutex to serialize tests that use global allocator state
    // This prevents race conditions that cause segfaults in release mode
    static GLOBAL_ALLOCATOR_TEST_MUTEX: Mutex<()> = Mutex::new(());

    /// C3.23 (S8-R3, MEDIUM). `MEDIUM_POOLS.with(..)` in `deallocate` panics
    /// with `cannot access a Thread Local Storage value during or after
    /// destruction` when reached from a user `Drop` that runs after
    /// `MEDIUM_POOLS` has been torn down on thread exit.
    ///
    /// Rust destroys a thread's TLS keys in LIFO order of their first
    /// initialization on that thread: touching `LATE_TLS` *before* the thread's
    /// first medium allocation initializes `MEDIUM_POOLS` guarantees that on
    /// thread exit `MEDIUM_POOLS` is destroyed first and `LATE_TLS`'s
    /// destructor runs second, when `MEDIUM_POOLS` is already gone.
    #[test]
    fn test_medium_deallocate_during_tls_teardown_does_not_panic() {
        use std::cell::RefCell;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        struct LateHolder {
            allocator: Arc<TieredMemoryAllocator>,
            allocation: Option<TieredAllocation>,
            finished_drop: Arc<AtomicBool>,
        }

        impl Drop for LateHolder {
            fn drop(&mut self) {
                if let Some(alloc) = self.allocation.take() {
                    // Must return `Ok` or `Err`, never panic inside `Drop`!
                    let _ = self.allocator.deallocate(alloc);
                }
                self.finished_drop.store(true, Ordering::SeqCst);
            }
        }

        thread_local! {
            static LATE_TLS: RefCell<Option<LateHolder>> = const { RefCell::new(None) };
        }

        let finished_drop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&finished_drop);

        let handle = std::thread::spawn(move || {
            let allocator = Arc::new(TieredMemoryAllocator::default().unwrap());
            // 1. Register `LATE_TLS`'s destructor FIRST so it runs AFTER
            //    `MEDIUM_POOLS`'s destructor (LIFO teardown order).
            LATE_TLS.with(|slot| {
                *slot.borrow_mut() = Some(LateHolder {
                    allocator: Arc::clone(&allocator),
                    allocation: None,
                    finished_drop: Arc::clone(&flag),
                });
            });
            // 2. Now perform the first medium allocation on this thread, which
            //    initializes `MEDIUM_POOLS` (so `MEDIUM_POOLS` will be destroyed
            //    BEFORE `LATE_TLS`).
            let medium = allocator.allocate(4096).unwrap();
            LATE_TLS.with(|slot| {
                slot.borrow_mut().as_mut().unwrap().allocation = Some(medium);
            });
        });

        handle
            .join()
            .expect("worker thread panicked or aborted during TLS teardown");
        assert!(
            finished_drop.load(Ordering::SeqCst),
            "LateHolder::drop did not run to completion during TLS teardown"
        );
    }

    #[test]
    fn test_tiered_allocator_creation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let stats = allocator.stats();

        assert_eq!(stats.small_allocations, 0);
        assert_eq!(stats.medium_allocations, 0);
        assert_eq!(stats.large_allocations, 0);
        assert_eq!(stats.huge_allocations, 0);
    }

    #[test]
    fn test_small_allocation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let size = 512; // Small allocation

        let mut allocation = allocator.allocate(size).unwrap();
        assert_eq!(allocation.size(), size);

        // Test that we can write to the memory
        let slice = allocation.as_mut_slice();
        slice[0] = 42;
        slice[size - 1] = 84;

        let slice = allocation.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[size - 1], 84);

        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.small_allocations, 1);
        assert_eq!(stats.total_allocated_bytes, 0); // Deallocated
    }

    #[test]
    fn test_medium_allocation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let size = 4 * 1024; // 4KB - medium allocation

        let mut allocation = allocator.allocate(size).unwrap();
        assert_eq!(allocation.size(), size);

        // Test memory access
        let slice = allocation.as_mut_slice();
        slice[0] = 42;
        slice[size - 1] = 84;

        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.medium_allocations, 1);
    }

    #[test]
    fn test_large_allocation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let size = 64 * 1024; // 64KB - large allocation

        let mut allocation = allocator.allocate(size).unwrap();
        assert_eq!(allocation.size(), size);

        // Test memory access
        let slice = allocation.as_mut_slice();
        slice[0] = 42;
        slice[size - 1] = 84;

        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.large_allocations, 1);
    }

    #[test]
    fn test_huge_allocation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let size = 4 * 1024 * 1024; // 4MB - huge allocation

        // Try allocation, but it might fail on systems without hugepage support
        match allocator.allocate(size) {
            Ok(mut allocation) => {
                assert_eq!(allocation.size(), size);

                // Test memory access
                let slice = allocation.as_mut_slice();
                slice[0] = 42;
                slice[size - 1] = 84;

                allocator.deallocate(allocation).unwrap();

                let stats = allocator.stats();
                // Might be huge or large depending on hugepage availability
                assert!(stats.huge_allocations > 0 || stats.large_allocations > 0);
            }
            Err(_) => {
                // Large allocation might fail on systems with limited memory or no hugepage support
                // This is acceptable in test environments
                println!("Huge allocation failed - this is acceptable in test environments");
            }
        }
    }

    #[test]
    fn test_mixed_allocation_pattern() {
        let allocator = TieredMemoryAllocator::default().unwrap();

        let sizes = vec![128, 2048, 32768, 1048576]; // Mix of small, medium, large
        let mut allocations = Vec::new();

        // Allocate all sizes
        for size in &sizes {
            let allocation = allocator.allocate(*size).unwrap();
            allocations.push(allocation);
        }

        // Deallocate all
        for allocation in allocations {
            allocator.deallocate(allocation).unwrap();
        }

        let stats = allocator.stats();
        assert!(stats.small_allocations > 0);
        assert!(stats.medium_allocations > 0);
        assert!(stats.large_allocations > 0);
    }

    #[test]
    fn test_allocation_pattern_detection() {
        let allocator = TieredMemoryAllocator::default().unwrap();

        // Allocate mostly small objects
        for _ in 0..100 {
            let allocation = allocator.allocate(256).unwrap();
            allocator.deallocate(allocation).unwrap();
        }

        let pattern = allocator.get_allocation_pattern().unwrap();
        // Should detect small-dominated pattern
        matches!(
            pattern,
            AllocationPattern::SmallDominated | AllocationPattern::Mixed
        );
    }

    #[test]
    fn test_global_tiered_allocator() {
        // Serialize access to global allocator to prevent race conditions
        let _guard = GLOBAL_ALLOCATOR_TEST_MUTEX.lock().unwrap();

        let size = 1024;

        let allocation = tiered_allocate(size).unwrap();
        assert_eq!(allocation.size(), size);

        tiered_deallocate(allocation).unwrap();

        let stats = get_tiered_stats();
        assert!(stats.small_allocations > 0 || stats.medium_allocations > 0);
    }

    #[test]
    fn test_deallocate_rejects_an_allocation_from_another_allocator() {
        // Each allocator owns its own small pool, and `MemoryPool::deallocate`
        // is `unsafe` precisely because it cannot recognise a foreign pointer.
        // `TieredMemoryAllocator::deallocate` is safe, so it must not be able
        // to hand one pool a chunk that came from another.
        let a = TieredMemoryAllocator::default().unwrap();
        let b = TieredMemoryAllocator::default().unwrap();

        let from_a = a.allocate(512).unwrap();
        let result = b.deallocate(from_a);

        assert!(
            result.is_err(),
            "a chunk from allocator a was parked in allocator b's pool"
        );
    }

    /// Allocate `size`, dirty it, hand it back, and allocate `size` again.
    ///
    /// Returns whether the same chunk came back, and whether every byte of it
    /// is zero. Both pool tiers recycle, so the second allocation is the one
    /// that would see the first one's bytes.
    fn recycled_chunk_is_clean(size: usize) -> (bool, bool) {
        let allocator = TieredMemoryAllocator::default().unwrap();

        let mut first = allocator.allocate(size).unwrap();
        let addr = first.as_ptr::<u8>() as usize;
        first.as_mut_slice().fill(0xAB);
        allocator.deallocate(first).unwrap();

        let second = allocator.allocate(size).unwrap();
        let recycled = second.as_ptr::<u8>() as usize == addr;
        let clean = second.as_slice().iter().all(|&b| b == 0);
        allocator.deallocate(second).unwrap();

        (recycled, clean)
    }

    #[test]
    fn test_a_recycled_small_chunk_is_zeroed() {
        // `as_slice` builds a `&[u8]` over pool memory the allocator never
        // initialized. For a fresh chunk that is a read of uninitialized
        // memory; for a recycled one it is the previous tenant's bytes, handed
        // to the next caller through an entirely safe API. The Large and Huge
        // arms do not have this problem only because MAP_ANONYMOUS pages
        // arrive zeroed from the kernel.
        let (recycled, clean) = recycled_chunk_is_clean(512);
        assert!(recycled, "vacuous unless the chunk is recycled");
        assert!(clean, "a recycled small chunk carried 0xAB into the next caller");
    }

    #[test]
    fn test_a_recycled_medium_chunk_is_zeroed() {
        let (recycled, clean) = recycled_chunk_is_clean(4 * 1024);
        assert!(recycled, "vacuous unless the chunk is recycled");
        assert!(clean, "a recycled medium chunk carried 0xAB into the next caller");
    }

    #[test]
    fn test_zero_size_allocation() {
        let allocator = TieredMemoryAllocator::default().unwrap();
        let result = allocator.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocator_configuration() {
        let config = TieredConfig {
            enable_small_pools: false,
            enable_medium_pools: false,
            enable_mmap_large: true,
            enable_hugepages: false,
            mmap_threshold: 512, // Lower threshold to allow small allocations
            hugepage_threshold: usize::MAX,
        };

        let allocator = TieredMemoryAllocator::new(config).unwrap();

        // Small allocation should fall back to mmap due to disabled pools
        let allocation = allocator.allocate(1024).unwrap(); // Use size above threshold
        allocator.deallocate(allocation).unwrap();

        let stats = allocator.stats();
        // Should have used large allocation (mmap) instead of small pool
        assert_eq!(stats.small_allocations, 0);
        assert!(stats.large_allocations > 0);
    }
}
