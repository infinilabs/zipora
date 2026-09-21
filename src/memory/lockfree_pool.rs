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
use crate::memory::cache_layout::{
    CacheLayoutConfig, CacheOptimizedAllocator, PrefetchHint,
};
use crate::memory::simd_ops::{fast_fill, fast_prefetch};
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
        let total_allocs =
            self.fast_allocs.load(Ordering::Relaxed) + self.skip_allocs.load(Ordering::Relaxed);
        // Simplified calculation - in practice would track time
        total_allocs as f64
    }

    /// Get CAS contention ratio (failures / total operations)
    pub fn contention_ratio(&self) -> f64 {
        let failures = self.cas_failures.load(Ordering::Relaxed);
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

/// Lock-free memory pool implementation
pub struct LockFreeMemoryPool {
    /// Configuration
    config: LockFreePoolConfig,
    /// Backing memory region
    memory: NonNull<u8>,
    /// Memory layout for deallocation
    memory_layout: Layout,
    /// Fast bins for small/medium allocations
    fast_bins: Vec<CachePadded<LockFreeHead>>,
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
// 2. `memory: NonNull<u8>` - Raw pointer to heap-allocated memory owned by this struct.
//    Memory is allocated in `new()` and deallocated in `Drop`. No thread-local state.
// 3. `memory_layout: Layout` - Trivially Send.
// 4. `fast_bins: Vec<CachePadded<LockFreeHead>>` - Contains only atomics.
// 5. `skip_list_head: Mutex<...>` - Mutex is Send.
// 6. `next_offset: AtomicU32` - AtomicU32 is Send.
// 7. `stats: Option<Arc<LockFreePoolStats>>` - Arc<T> is Send if T is Send+Sync.
// 8. `cache_allocator: Option<CacheOptimizedAllocator>` - Safe container.
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
        // Allocate backing memory region
        let layout = Layout::from_size_align(config.memory_size, ALIGN_SIZE)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {}", e)))?;

        // SAFETY: layout valid (size > 0, align power of 2)
        let memory = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(config.memory_size))?;

        // Initialize fast bins
        let mut fast_bins = Vec::with_capacity(FAST_BIN_COUNT);
        for _ in 0..FAST_BIN_COUNT {
            fast_bins.push(CachePadded::new(LockFreeHead::new()));
        }

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

        Ok(Self {
            config,
            memory,
            memory_layout: layout,
            fast_bins,
            large_free_list: Mutex::new(LargeFreeList::default()),

            next_offset: AtomicU32::new(ALIGN_SIZE as u32), // Start after header
            stats,
            cache_allocator,
        })
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("Cannot allocate zero bytes"));
        }

        let aligned_size = self.align_size(size);

        if aligned_size <= FAST_BIN_THRESHOLD {
            self.allocate_from_fast_bin(aligned_size)
        } else {
            self.allocate_from_skip_list(aligned_size)
        }
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        if size == 0 {
            return Ok(());
        }

        let aligned_size = self.align_size(size);

        if aligned_size <= FAST_BIN_THRESHOLD {
            self.deallocate_to_fast_bin(ptr, aligned_size)
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
                        stats.fast_allocs.fetch_add(1, Ordering::Relaxed);
                        stats.cas_successes.fetch_add(1, Ordering::Relaxed);
                    }

                    return Ok(Some(current_offset));
                }
                Err(_) => {
                    if let Some(stats) = &self.stats {
                        stats.cas_failures.fetch_add(1, Ordering::Relaxed);
                    }

                    self.backoff(retry.min(MAX_BACKOFF_RETRY));
                }
            }
        }

        Ok(None)
    }

    /// Allocate from fast bin using lock-free stack
    fn allocate_from_fast_bin(&self, size: usize) -> Result<NonNull<u8>> {
        let bin_index = self.size_to_bin_index(size)?;
        let bin = &self.fast_bins[bin_index];

        // Prefer a recycled block. `max_cas_retries` bounds how long we are
        // willing to contend for one before carving fresh memory instead; it is
        // a throughput knob, not a correctness parameter.
        if let Some(offset) = self.pop_fast_bin(bin, self.config.max_cas_retries)? {
            return self.offset_to_ptr(offset);
        }

        // Empty bin, or the budget ran out: carve a fresh block at the class
        // width, not the request size. Blocks are recycled by bin, so a block
        // carved any smaller would later be handed to a larger request in the
        // same class and overrun its neighbour.
        match self.allocate_new_block(FAST_BIN_SIZES[bin_index]) {
            Ok(ptr) => Ok(ptr),
            Err(exhausted) => {
                // C3.18. The arena is spent, so the bin is the only source
                // left, and a spent retry budget must not be allowed to turn a
                // non-empty bin into an out-of-memory error -- which is exactly
                // what a single-threaded pool used to report under Miri, whose
                // `compare_exchange_weak` fails spuriously. Retry unbudgeted:
                // with a strong CAS every failure means another thread
                // completed a push or a pop, so this terminates.
                match self.pop_fast_bin(bin, u32::MAX)? {
                    Some(offset) => self.offset_to_ptr(offset),
                    None => Err(exhausted),
                }
            }
        }
    }

    /// Deallocate to fast bin using lock-free stack
    fn deallocate_to_fast_bin(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        let bin_index = self.size_to_bin_index(size)?;
        let bin = &self.fast_bins[bin_index];
        let offset = self.ptr_to_offset(ptr)?;

        // C3.18. Unbudgeted by design, unlike the pop side: a push onto a
        // Treiber stack always has somewhere to go, so a retry budget here can
        // only ever produce a worse answer than waiting. It used to file the
        // block on the large free list once the budget ran out, where no
        // <=8 KiB request can ever pick it up again; with `compare_exchange_weak`
        // that path was reachable with no contention at all. With a strong CAS
        // every failure means another thread completed a push or a pop, so this
        // loop terminates.
        let mut retry = 0u32;
        loop {
            // ABA-SAFE: Load packed value (offset + generation)
            let packed = bin.head.load(Ordering::Acquire);
            let (current_offset, current_gen) = Self::unpack_head(packed);

            // Link to the current head through the block header (offset only;
            // the generation lives in the bin head).
            self.link_slot(offset)?.store(current_offset, Ordering::Relaxed);

            // ABA-SAFE: Pack new offset with INCREMENTED generation counter
            let new_packed = Self::pack_head(offset, current_gen.wrapping_add(1));

            match bin.head.compare_exchange(
                packed,     // Compare full packed value (offset + generation)
                new_packed, // New packed value with incremented generation
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    bin.count.fetch_add(1, Ordering::Relaxed);

                    if let Some(stats) = &self.stats {
                        stats.fast_deallocs.fetch_add(1, Ordering::Relaxed);
                        stats.cas_successes.fetch_add(1, Ordering::Relaxed);
                    }

                    return Ok(());
                }
                Err(_) => {
                    if let Some(stats) = &self.stats {
                        stats.cas_failures.fetch_add(1, Ordering::Relaxed);
                    }

                    self.backoff(retry.min(MAX_BACKOFF_RETRY));
                    retry = retry.saturating_add(1);
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
        let mut current = self.next_offset.load(Ordering::Relaxed);
        loop {
            if current as usize + carve > self.config.memory_size {
                return Err(ZiporaError::out_of_memory(aligned_size));
            }
            let next = match current.checked_add(carve as u32) {
                Some(val) => val,
                None => return Err(ZiporaError::out_of_memory(aligned_size)),
            };
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
            stats.memory_usage.fetch_add(carve as u64, Ordering::Relaxed);
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
    fn size_to_bin_index(&self, size: usize) -> Result<usize> {
        for (index, &bin_size) in FAST_BIN_SIZES.iter().enumerate() {
            if size <= bin_size {
                return Ok(index);
            }
        }
        Err(ZiporaError::invalid_data("Size too large for fast bins"))
    }

    /// Align size to allocation boundary
    fn align_size(&self, size: usize) -> usize {
        (size + ALIGN_SIZE - 1) & !(ALIGN_SIZE - 1)
    }

    /// Convert offset to pointer
    fn offset_to_ptr(&self, offset: u32) -> Result<NonNull<u8>> {
        if offset == LIST_TAIL {
            return Err(ZiporaError::invalid_data("Invalid offset"));
        }
        if offset as usize >= self.config.memory_size {
            return Err(ZiporaError::invalid_data("Offset exceeds pool memory size"));
        }

        // SAFETY: Offset is validated to be within allocated memory pool size boundaries
        let addr = unsafe { self.memory.as_ptr().add(offset as usize) };
        NonNull::new(addr).ok_or_else(|| ZiporaError::invalid_data("Invalid pointer"))
    }

    /// Convert pointer to offset
    fn ptr_to_offset(&self, ptr: NonNull<u8>) -> Result<u32> {
        let base = self.memory.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;

        if addr < base || addr >= base + self.config.memory_size {
            return Err(ZiporaError::invalid_data("Pointer outside pool memory"));
        }

        Ok((addr - base) as u32)
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
        // SAFETY: ptr from matching alloc, layout matches allocation
        unsafe {
            dealloc(self.memory.as_ptr(), self.memory_layout);
        }
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
        const ARENA: usize =
            2 * THREADS * PER_THREAD * (MAX_SIZE + BLOCK_HEADER) + ALIGN_SIZE;

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
                        unsafe {
                            std::ptr::write_bytes(ptr.as_ptr(), thread_id as u8, size)
                        };
                        blocks.push((ptr.as_ptr() as usize, size));
                    }
                    for &(addr, size) in &blocks {
                        // SAFETY: as above -- still ours, still `size` wide.
                        let ours = unsafe {
                            std::slice::from_raw_parts(addr as *const u8, size)
                        };
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
}
