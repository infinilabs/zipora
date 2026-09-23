//! Five-Level Concurrency Management System
//!
//! This module implements a sophisticated 5-level concurrency management system
//! inspired by advanced memory pool architectures, providing graduated concurrency
//! control options for different performance and threading requirements.
//!
//! ## The 5 Levels of Concurrency Control
//!
//! 1. **Level 1: No Locking** - Pure single-threaded operation with zero synchronization overhead
//! 2. **Level 2: Mutex-based Locking** - Fine-grained locking with separate mutexes per size class
//! 3. **Level 3: Lock-free Programming** - Atomic compare-and-swap operations for small allocations
//! 4. **Level 4: Thread-local Caching** - Per-thread local memory pools to minimize cross-thread contention
//! 5. **Level 5: Fixed Capacity Variant** - Bounded memory allocation with no expansion
//!
//! ## Design Principles
//!
//! - **API Compatibility**: All levels share consistent interfaces
//! - **Graduated Complexity**: Each level builds sophistication while maintaining simpler fallbacks
//! - **Hardware Awareness**: Cache alignment, atomic operations, prefetching
//! - **Adaptive Selection**: Choose appropriate level based on thread count, allocation patterns, and performance requirements
//! - **Composability**: Different components can use different concurrency levels

use crate::error::{Result, ZiporaError};
use crate::memory::cache_layout::CacheLayoutConfig;
// Memory pool integration (currently unused in this implementation)
// use crate::memory::SecureMemoryPool;
use std::sync::{Arc, Mutex};
// Additional sync primitives (currently unused)
// use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
// Additional utilities (currently unused)
// use std::collections::HashMap;
// use std::marker::PhantomData;
use std::ptr::NonNull;
// use std::mem::MaybeUninit;
use std::alloc::{Layout, alloc, dealloc};

/// Memory offset type for 32-bit addressing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MemOffset(u32);

impl MemOffset {
    const NULL: Self = MemOffset(u32::MAX);

    /// Narrow a byte offset into the 32-bit offset space.
    ///
    /// The `debug_assert!` is a redundant backstop, not the guard: every arena
    /// in this module is sized by `FiveLevelPoolConfig`, and
    /// [`FiveLevelPoolConfig::validate`] rejects any `initial_capacity`,
    /// `arena_size`, or `fixed_capacity` above
    /// [`FiveLevelPoolConfig::MAX_ARENA_SIZE`] before a chunk is allocated.
    /// Offsets are therefore strictly below `u32::MAX` by construction, which
    /// is what working agreement 4 requires: the narrowing cannot be reached
    /// from safe public API with an out-of-range value.
    fn new(offset: usize) -> Self {
        debug_assert!(offset < u32::MAX as usize);
        MemOffset(offset as u32)
    }

    fn to_usize(self) -> usize {
        if self.0 == u32::MAX {
            usize::MAX
        } else {
            self.0 as usize
        }
    }

    fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

/// Configuration for the 5-level memory pool system
#[derive(Debug, Clone)]
pub struct FiveLevelPoolConfig {
    /// Maximum block size for fast bins (typically 8KB-64KB)
    pub max_fast_block_size: usize,
    /// Alignment requirement (must be power of 2, >= 4)
    pub alignment: usize,
    /// Initial capacity for the memory pool
    pub initial_capacity: usize,
    /// Thread-local arena size for Level 4
    pub arena_size: usize,
    /// Fixed capacity for Level 5 (0 = use dynamic)
    pub fixed_capacity: Option<usize>,
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
}

impl Default for FiveLevelPoolConfig {
    fn default() -> Self {
        Self {
            max_fast_block_size: 32 * 1024, // 32KB
            alignment: 8,
            initial_capacity: 1024 * 1024, // 1MB
            arena_size: 2 * 1024 * 1024, // 2MB
            fixed_capacity: None,
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::default()),
            enable_numa_awareness: true,
            enable_huge_pages: false,
            huge_page_threshold: 2 * 1024 * 1024, // 2MB
        }
    }
}

impl FiveLevelPoolConfig {
    pub fn performance_optimized() -> Self {
        Self {
            max_fast_block_size: 64 * 1024, // 64KB
            alignment: 16,
            initial_capacity: 8 * 1024 * 1024, // 8MB
            arena_size: 4 * 1024 * 1024,       // 4MB
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::default()),
            enable_numa_awareness: true,
            enable_huge_pages: true,
            huge_page_threshold: 1024 * 1024, // 1MB
            ..Default::default()
        }
    }

    pub fn memory_optimized() -> Self {
        Self {
            max_fast_block_size: 16 * 1024, // 16KB
            alignment: 8,
            initial_capacity: 512 * 1024, // 512KB
            arena_size: 1024 * 1024,      // 1MB
            enable_cache_alignment: false,
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 4 * 1024 * 1024, // 4MB
            ..Default::default()
        }
    }

    /// Largest arena this module can address.
    ///
    /// Offsets are [`MemOffset`], a `u32`, and `u32::MAX` is reserved as the
    /// null sentinel, so no arena may reach it.
    pub const MAX_ARENA_SIZE: usize = u32::MAX as usize;

    /// Validate every field the pools read without checking it again.
    ///
    /// All fields of this struct are `pub`, so every value tested here is
    /// reachable from safe code. Before this existed the only field that was
    /// validated at all was `alignment`, and only incidentally, by
    /// `Layout::from_size_align` inside `MemoryChunk::new` — which reports
    /// "Invalid memory layout" and names nothing.
    ///
    /// # Errors
    ///
    /// Returns [`ZiporaError::invalid_data`] if:
    /// - `alignment` is zero, is not a power of two (the `(size + a - 1) & !(a - 1)`
    ///   mask in `align_up` is meaningless otherwise), or is smaller than the
    ///   4-byte free-list link stored in every fast-bin block;
    /// - `max_fast_block_size` is zero, or is not a whole multiple of
    ///   `alignment` — the bin index is `size / alignment - 1`, so a partial
    ///   unit sizes a bin that no request can ever select;
    /// - `initial_capacity`, `arena_size`, or `fixed_capacity` is zero. A
    ///   zero-sized `Layout` is valid to construct but undefined behaviour to
    ///   pass to [`std::alloc::alloc`];
    /// - any of those three exceeds [`Self::MAX_ARENA_SIZE`], which would
    ///   truncate offsets in [`MemOffset`] and hand the same offset out for
    ///   two different live blocks.
    pub fn validate(&self) -> Result<()> {
        /// Width of the `u32` free-list link written into a freed fast-bin block.
        const LINK_WIDTH: usize = std::mem::size_of::<u32>();

        if self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(ZiporaError::invalid_data(format!(
                "pool alignment must be a non-zero power of two, got {}",
                self.alignment
            )));
        }
        if self.alignment < LINK_WIDTH {
            return Err(ZiporaError::invalid_data(format!(
                "pool alignment must be at least {LINK_WIDTH} bytes to hold the \
                 free-list link, got {}",
                self.alignment
            )));
        }
        if self.max_fast_block_size == 0 {
            return Err(ZiporaError::invalid_data(
                "max_fast_block_size must be non-zero",
            ));
        }
        if !self.max_fast_block_size.is_multiple_of(self.alignment) {
            return Err(ZiporaError::invalid_data(format!(
                "max_fast_block_size ({}) must be a multiple of alignment ({})",
                self.max_fast_block_size, self.alignment
            )));
        }

        for (name, value) in [
            ("initial_capacity", Some(self.initial_capacity)),
            ("arena_size", Some(self.arena_size)),
            ("fixed_capacity", self.fixed_capacity),
        ] {
            let Some(value) = value else { continue };
            if value == 0 {
                return Err(ZiporaError::invalid_data(format!(
                    "{name} must be non-zero: a zero-sized layout is undefined \
                     behaviour for the global allocator"
                )));
            }
            if value > Self::MAX_ARENA_SIZE {
                return Err(ZiporaError::invalid_data(format!(
                    "{name} ({value}) exceeds the 32-bit offset space ({}); \
                     offsets would truncate and two live blocks would alias",
                    Self::MAX_ARENA_SIZE
                )));
            }
        }

        Ok(())
    }

    pub fn realtime() -> Self {
        Self {
            max_fast_block_size: 8 * 1024, // 8KB
            alignment: 8,
            initial_capacity: 256 * 1024,           // 256KB
            arena_size: 512 * 1024,                 // 512KB
            fixed_capacity: Some(16 * 1024 * 1024), // 16MB fixed
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::sequential()),
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 8 * 1024 * 1024, // 8MB
        }
    }
}

/// Address-ordered, coalescing, best-fit free list for blocks above
/// `max_fast_block_size`.
///
/// D3. Levels 1-3 each shipped a stub here: `NoLockingPool::free_to_skip_list`,
/// `MutexBasedPool::free_to_skip_list` and `LockFreePool::free_to_huge_mutex`
/// all bound the offset as `_offset` and dropped it behind a
/// `// TODO: Implement skip list insertion` marker, while still charging
/// `used_memory -= size` and `fragment_size += size`, and the matching
/// `alloc_from_skip_list` always bumped from the end of the arena. Every free
/// above `max_fast_block_size` lost its block permanently: a 1 MiB arena
/// cycling one 64 KiB block ran out of memory on the sixteenth iteration.
///
/// This is deliberately *not* the skip list the marker promised.
/// `lockfree_pool.rs` already solves the identical problem with a plain
/// best-fit `Vec` behind a mutex, and hand-rolling a concurrent skip list would
/// add new `unsafe` surface during a pass whose purpose is to remove it. The
/// pool keeps its name: "five level" counts concurrency levels, not skip-list
/// levels.
///
/// Two invariants hold at all times:
/// - `regions` is sorted by offset and no two entries are adjacent or
///   overlapping, because [`Self::free`] merges with both neighbours;
/// - a carve returns exactly the requested width, because [`Self::alloc`]
///   returns any remainder to the list instead of handing it to the caller.
///   That is what keeps the carve width equal to the recycle width: a later
///   `free(offset, size)` files back precisely what was taken, so a region
///   cannot shrink across a reuse cycle.
#[derive(Debug, Default)]
struct HugeFreeList {
    /// Free regions as `(offset, size)`.
    regions: Vec<(usize, usize)>,
    /// Sum of `size` over `regions`, maintained incrementally.
    total_bytes: usize,
}

impl HugeFreeList {
    /// Carve exactly `size` bytes out of the smallest region that fits.
    ///
    /// Returns `None` if no region is large enough, in which case the caller
    /// falls back to bumping the end of the arena.
    fn alloc(&mut self, size: usize) -> Option<usize> {
        let index = self
            .regions
            .iter()
            .enumerate()
            .filter(|(_, (_, region_size))| *region_size >= size)
            .min_by_key(|(_, (_, region_size))| *region_size)
            .map(|(index, _)| index)?;

        let (offset, region_size) = self.regions[index];
        if region_size == size {
            self.regions.remove(index);
        } else {
            // Keep the tail, hand back the head.
            self.regions[index] = (offset + size, region_size - size);
        }
        self.total_bytes -= size;
        Some(offset)
    }

    /// Return `[offset, offset + size)` to the list, merging it with either
    /// neighbour it touches.
    fn free(&mut self, offset: usize, size: usize) {
        let index = self.regions.partition_point(|(start, _)| *start < offset);
        debug_assert!(
            index == self.regions.len() || offset + size <= self.regions[index].0,
            "huge free list: [{offset}, {}) overlaps the region above it",
            offset + size
        );
        debug_assert!(
            index == 0 || self.regions[index - 1].0 + self.regions[index - 1].1 <= offset,
            "huge free list: [{offset}, {}) overlaps the region below it",
            offset + size
        );

        self.regions.insert(index, (offset, size));
        self.total_bytes += size;

        // Merge upwards first so the index of the inserted entry stays valid.
        if index + 1 < self.regions.len() {
            let (next_start, next_size) = self.regions[index + 1];
            if offset + size == next_start {
                self.regions[index].1 += next_size;
                self.regions.remove(index + 1);
            }
        }
        if index > 0 {
            let (prev_start, prev_size) = self.regions[index - 1];
            if prev_start + prev_size == offset {
                self.regions[index - 1].1 += self.regions[index].1;
                self.regions.remove(index);
            }
        }
    }

    /// Total bytes currently held in the list.
    fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of distinct (non-adjacent) free regions.
    fn node_count(&self) -> usize {
        self.regions.len()
    }
}

/// Reject a zero-byte request before it reaches the bin arithmetic.
///
/// Both `alloc_from_fast_bin` and `free_to_fast_bin` compute
/// `bin_index = size / alignment - 1`. `align_up(0)` is 0, so that subtraction
/// underflows: a debug build panics with "attempt to subtract with overflow"
/// and a release build wraps to `usize::MAX`, skips the bin entirely, and
/// carves a zero-width block off the end of the arena. `alloc` and `free` are
/// safe public API on all five levels, so the request is refused up front.
fn reject_zero_size(size: usize, operation: &str) -> Result<()> {
    if size == 0 {
        return Err(ZiporaError::invalid_data(format!(
            "zero-sized {operation} is not supported: block sizes must be at \
             least one alignment unit"
        )));
    }
    Ok(())
}

/// Free list head for fast bins
#[derive(Debug, Clone)]
struct FreeListHead {
    head: MemOffset,
    count: u32,
}

impl Default for FreeListHead {
    fn default() -> Self {
        Self {
            head: MemOffset::NULL,
            count: 0,
        }
    }
}

/// Cache-line aligned free list head for lock-free operations
#[derive(Debug)]
#[repr(align(64))]
struct LockFreeFreeListHead {
    head: AtomicU32,
    count: AtomicU32,
    _padding: [u8; 64 - 8], // Ensure 64-byte alignment
}

impl Default for LockFreeFreeListHead {
    fn default() -> Self {
        Self {
            head: AtomicU32::new(u32::MAX),
            count: AtomicU32::new(0),
            _padding: [0; 64 - 8],
        }
    }
}

/// Memory chunk representation
#[derive(Debug)]
struct MemoryChunk {
    data: NonNull<u8>,
    size: usize,
    capacity: usize,
    /// The exact `Layout` `data` was allocated with. `GlobalAlloc::dealloc`
    /// requires the same layout that was passed to `alloc`, and the chunk's
    /// alignment comes from `FiveLevelPoolConfig::alignment` (8 by default,
    /// 16 for `performance_optimized`), not from `align_of::<u8>()`. Keeping
    /// the layout is the only way to honour that contract.
    layout: Layout,
}

// SAFETY: MemoryChunk is Send because:
// 1. `data: NonNull<u8>` - Raw pointer to heap-allocated memory owned by this struct.
//    Memory is allocated in `new()` and deallocated in `Drop`. No thread-local state.
// 2. `size: usize` - Mutable state but only accessed through &mut self.
// 3. `capacity: usize` - Immutable after construction, trivially Send.
// 4. `layout: Layout` - Plain data, immutable after construction.
unsafe impl Send for MemoryChunk {}

// SAFETY: MemoryChunk is Sync because:
// 1. `data: NonNull<u8>` - Read-only access through &self via `offset_ptr()`.
// 2. `size`/`capacity`/`layout` - Read-only through &self.
// 3. Mutable operations require &mut self (exclusive access).
// 4. The chunk provides raw memory that callers must synchronize.
//
// Note: MemoryChunk itself is Sync, but the memory it points to requires
// external synchronization when accessed from multiple threads.
unsafe impl Sync for MemoryChunk {}

impl MemoryChunk {
    fn new(capacity: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(capacity, alignment)
            .map_err(|_| ZiporaError::invalid_data("Invalid memory layout"))?;

        // SAFETY: layout is valid from Layout::from_size_align
        let data = unsafe { alloc(layout) };
        if data.is_null() {
            return Err(ZiporaError::resource_exhausted("Failed to allocate memory"));
        }

        // SAFETY: We checked data.is_null() above, so this is guaranteed to succeed
        let non_null_data = unsafe { NonNull::new_unchecked(data) };

        Ok(Self {
            data: non_null_data,
            size: 0,
            capacity,
            layout,
        })
    }

    unsafe fn offset_ptr(&self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.capacity);
        // SAFETY: caller ensures offset <= capacity per function contract
        unsafe { self.data.as_ptr().add(offset) }
    }

    fn can_allocate(&self, size: usize) -> bool {
        self.size + size <= self.capacity
    }
}

impl Drop for MemoryChunk {
    fn drop(&mut self) {
        // SAFETY: `self.layout` is the exact layout `self.data` was allocated
        // with in `MemoryChunk::new`, and `data` is only deallocated here, once.
        unsafe {
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

/// Level 1: No Locking - Single-threaded memory pool
pub struct NoLockingPool {
    config: FiveLevelPoolConfig,
    memory: MemoryChunk,
    free_lists: Vec<FreeListHead>,

    fragment_size: usize,
    /// Free blocks above `max_fast_block_size`. See [`HugeFreeList`].
    huge_free_list: HugeFreeList,
    used_memory: usize, // Track actual used memory (high-water mark minus freed blocks)
}

impl NoLockingPool {
    /// # Errors
    ///
    /// Returns an error if `config` fails [`FiveLevelPoolConfig::validate`], or
    /// if the backing arena cannot be allocated.
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        config.validate()?;
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = vec![FreeListHead::default(); num_bins];

        Ok(Self {
            config,
            memory,
            free_lists,

            fragment_size: 0,
            huge_free_list: HugeFreeList::default(),
            used_memory: 0,
        })
    }

    /// Allocate memory block of given size
    ///
    /// # Errors
    ///
    /// Returns an error for a zero-byte request (see [`reject_zero_size`]) or
    /// when the arena is exhausted.
    pub fn alloc(&mut self, size: usize) -> Result<MemOffset> {
        reject_zero_size(size, "allocation")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin(aligned_size)
        } else {
            self.alloc_huge(aligned_size)
        }
    }

    /// Free previously allocated memory block
    ///
    /// # Errors
    ///
    /// Returns an error for a zero-byte block (see [`reject_zero_size`]).
    pub fn free(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        reject_zero_size(size, "free")?;
        let aligned_size = self.align_up(size);

        // Check if this is at the end of used memory
        if offset.to_usize() + aligned_size == self.memory.size {
            self.memory.size = offset.to_usize();
            self.used_memory -= aligned_size;
            return Ok(());
        }

        if aligned_size <= self.config.max_fast_block_size {
            self.free_to_fast_bin(offset, aligned_size)
        } else {
            self.free_huge(offset, aligned_size)
        }
    }

    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }

    fn alloc_from_fast_bin(&mut self, size: usize) -> Result<MemOffset> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let head = &mut self.free_lists[bin_index];
            if !head.head.is_null() {
                // Pop from free list
                let offset = head.head;
                // SAFETY: offset from free list points to valid u32 storing next offset
                unsafe {
                    let ptr = self.memory.offset_ptr(offset.to_usize()) as *mut u32;
                    head.head = MemOffset(*ptr);
                }
                head.count -= 1;
                self.fragment_size -= size;
                self.used_memory += size; // Track reused memory as used
                return Ok(offset);
            }
        }

        // Allocate from end of memory
        self.alloc_from_end(size)
    }

    fn alloc_from_end(&mut self, size: usize) -> Result<MemOffset> {
        if !self.memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(self.memory.size);
        self.memory.size += size;
        self.used_memory += size;
        Ok(offset)
    }

    /// Serve a block above `max_fast_block_size` from the huge free list,
    /// falling back to the end of the arena.
    fn alloc_huge(&mut self, size: usize) -> Result<MemOffset> {
        if let Some(offset) = self.huge_free_list.alloc(size) {
            self.fragment_size -= size;
            self.used_memory += size;
            return Ok(MemOffset::new(offset));
        }
        self.alloc_from_end(size)
    }

    fn free_to_fast_bin(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let head = &mut self.free_lists[bin_index];

            // Push to free list
            // SAFETY: offset points to freed block of size >= 4, can store u32
            unsafe {
                let ptr = self.memory.offset_ptr(offset.to_usize()) as *mut u32;
                *ptr = head.head.0;
            }
            head.head = offset;
            head.count += 1;
            self.fragment_size += size;
            self.used_memory -= size; // Decrease used memory when freed
        }

        Ok(())
    }

    /// Return a block above `max_fast_block_size` to the huge free list.
    fn free_huge(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        self.huge_free_list.free(offset.to_usize(), size);
        self.fragment_size += size;
        self.used_memory -= size; // Decrease used memory when freed
        Ok(())
    }

    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_capacity: self.memory.capacity,
            used_memory: self.used_memory,
            fragment_size: self.fragment_size,
            huge_size_sum: self.huge_free_list.total_bytes(),
            huge_node_count: self.huge_free_list.node_count(),
            free_list_count: self.free_lists.len(),
        }
    }
}

/// Statistics for memory pool performance monitoring
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_capacity: usize,
    pub used_memory: usize,
    pub fragment_size: usize,
    pub huge_size_sum: usize,
    pub huge_node_count: usize,
    pub free_list_count: usize,
}

impl PoolStats {
    pub fn utilization(&self) -> f64 {
        if self.total_capacity == 0 {
            0.0
        } else {
            self.used_memory as f64 / self.total_capacity as f64
        }
    }

    pub fn fragmentation_ratio(&self) -> f64 {
        if self.used_memory == 0 {
            0.0
        } else {
            self.fragment_size as f64 / self.used_memory as f64
        }
    }
}

/// Level 2: Mutex-based Locking - Fine-grained locking memory pool
pub struct MutexBasedPool {
    config: FiveLevelPoolConfig,
    memory: Arc<Mutex<MemoryChunk>>,
    free_lists: Vec<Mutex<FreeListHead>>,

    fragment_size: AtomicUsize,
    /// Free blocks above `max_fast_block_size`. See [`HugeFreeList`].
    huge_free_list: Mutex<HugeFreeList>,
}

impl MutexBasedPool {
    /// # Errors
    ///
    /// Returns an error if `config` fails [`FiveLevelPoolConfig::validate`], or
    /// if the backing arena cannot be allocated.
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        config.validate()?;
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = (0..num_bins)
            .map(|_| Mutex::new(FreeListHead::default()))
            .collect();

        Ok(Self {
            config,
            memory: Arc::new(Mutex::new(memory)),
            free_lists,

            fragment_size: AtomicUsize::new(0),
            huge_free_list: Mutex::new(HugeFreeList::default()),
        })
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte request (see [`reject_zero_size`]) or
    /// when the arena is exhausted.
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        reject_zero_size(size, "allocation")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin(aligned_size)
        } else {
            self.alloc_huge(aligned_size)
        }
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte block (see [`reject_zero_size`]).
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        reject_zero_size(size, "free")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            self.free_to_fast_bin(offset, aligned_size)
        } else {
            self.free_huge(offset, aligned_size)
        }
    }

    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }

    fn alloc_from_fast_bin(&self, size: usize) -> Result<MemOffset> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let mut head = self.free_lists[bin_index].lock().map_err(|e| {
                ZiporaError::resource_busy(format!("Free list mutex poisoned: {}", e))
            })?;
            if !head.head.is_null() {
                let offset = head.head;
                // SAFETY: offset from free list points to valid u32 storing next offset
                unsafe {
                    let memory = self.memory.lock().map_err(|e| {
                        ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e))
                    })?;
                    let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                    head.head = MemOffset(*ptr);
                }
                head.count -= 1;
                self.fragment_size.fetch_sub(size, Ordering::Relaxed);
                return Ok(offset);
            }
        }

        // Allocate from end
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }

    /// Serve a block above `max_fast_block_size` from the huge free list,
    /// falling back to the end of the arena.
    ///
    /// The huge-list lock is released before the memory lock is taken, so the
    /// two are never held at once and cannot deadlock against each other.
    fn alloc_huge(&self, size: usize) -> Result<MemOffset> {
        {
            let mut huge = self.huge_free_list.lock().map_err(|e| {
                ZiporaError::resource_busy(format!("Huge free list mutex poisoned: {}", e))
            })?;
            if let Some(offset) = huge.alloc(size) {
                self.fragment_size.fetch_sub(size, Ordering::Relaxed);
                return Ok(MemOffset::new(offset));
            }
        }

        let mut memory = self
            .memory
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }

    fn free_to_fast_bin(&self, offset: MemOffset, size: usize) -> Result<()> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let mut head = self.free_lists[bin_index].lock().map_err(|e| {
                ZiporaError::resource_busy(format!("Free list mutex poisoned: {}", e))
            })?;

            // SAFETY: offset points to freed block of size >= 4, can store u32
            unsafe {
                let memory = self.memory.lock().map_err(|e| {
                    ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e))
                })?;
                let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                *ptr = head.head.0;
            }
            head.head = offset;
            head.count += 1;
            self.fragment_size.fetch_add(size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Return a block above `max_fast_block_size` to the huge free list.
    fn free_huge(&self, offset: MemOffset, size: usize) -> Result<()> {
        self.fragment_size.fetch_add(size, Ordering::Relaxed);
        let mut huge = self.huge_free_list.lock().map_err(|e| {
            ZiporaError::resource_busy(format!("Huge free list mutex poisoned: {}", e))
        })?;
        huge.free(offset.to_usize(), size);
        Ok(())
    }

    pub fn stats(&self) -> PoolStats {
        let memory = self.memory.lock().unwrap_or_else(|e| e.into_inner());
        let huge = self.huge_free_list.lock().unwrap_or_else(|e| e.into_inner());

        PoolStats {
            total_capacity: memory.capacity,
            used_memory: memory.size,
            fragment_size: self.fragment_size.load(Ordering::Relaxed),
            huge_size_sum: huge.total_bytes(),
            huge_node_count: huge.node_count(),
            free_list_count: self.free_lists.len(),
        }
    }
}

impl MutexBasedPool {
    /// Pop one recycled block from `self.free_lists[bin_index]` if the bin is
    /// non-empty, without advancing the arena bump cursor.
    #[inline]
    fn try_pop_fast_bin(&self, size: usize) -> Option<MemOffset> {
        // Fast check: if fewer than `size` fragmented bytes exist across all
        // bins, this bin cannot hold a free block of `size` bytes.
        if self.fragment_size.load(Ordering::Relaxed) < size {
            return None;
        }
        let bin_index = (size / self.config.alignment) - 1;
        let mut head = self.free_lists.get(bin_index)?.lock().ok()?;
        if head.head.is_null() {
            return None;
        }
        let offset = head.head;
        // SAFETY: offset from free list points to valid u32 storing next offset
        unsafe {
            let memory = self.memory.lock().ok()?;
            let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
            head.head = MemOffset(*ptr);
        }
        head.count -= 1;
        self.fragment_size.fetch_sub(size, Ordering::Relaxed);
        Some(offset)
    }

    /// Carve up to `max_bytes` (and at least `min_bytes`) from the bump cursor
    /// of `self.memory`, returning `Some((start, end))` in `self.memory`'s
    /// single address space. Used by `ThreadLocalPool` so thread-local bump
    /// allocations stay in the global pool's address space without locking the
    /// global pool on every allocation.
    fn carve_slab(&self, max_bytes: usize, min_bytes: usize) -> Option<(usize, usize)> {
        let mut memory = self.memory.lock().ok()?;
        let remaining = memory.capacity.saturating_sub(memory.size);
        if remaining < min_bytes {
            return None;
        }
        let take = max_bytes.min(remaining);
        let start = memory.size;
        memory.size += take;
        Some((start, start + take))
    }
}

// SAFETY: MutexBasedPool is Send because:
// 1. `config: FiveLevelPoolConfig` - Config is Clone, no pointers.
// 2. `memory: Arc<Mutex<MemoryChunk>>` - Arc<Mutex<T>> is Send if T is Send.
// 3. `free_lists: Vec<Mutex<FreeListHead>>` - Mutex<T> is Send if T is Send.
// 4. `fragment_size: AtomicUsize` - AtomicUsize is Send.
// 5. `huge_free_list: Mutex<HugeFreeList>` - Mutex<T> is Send if T is Send.
unsafe impl Send for MutexBasedPool {}

// SAFETY: MutexBasedPool is Sync because:
// 1. All mutable state is protected by Mutex (provides interior mutability).
// 2. `memory` access is serialized by Arc<Mutex<...>>.
// 3. `free_lists` - Each size class has its own Mutex for fine-grained locking.
// 4. `fragment_size` - AtomicUsize provides atomic updates.
// 5. Mutex provides mutual exclusion for all shared mutable state.
unsafe impl Sync for MutexBasedPool {}

/// Level 3: Lock-free Programming - Compare-and-swap memory pool
pub struct LockFreePool {
    config: FiveLevelPoolConfig,
    memory: Arc<Mutex<MemoryChunk>>, // Still need mutex for memory expansion
    free_lists: Vec<LockFreeFreeListHead>,
    fragment_size: AtomicUsize,
    /// Free blocks above `max_fast_block_size`. See [`HugeFreeList`]. Huge
    /// blocks are rare and large, so they stay behind a mutex.
    huge_free_list: Mutex<HugeFreeList>,
}

impl LockFreePool {
    /// # Errors
    ///
    /// Returns an error if `config` fails [`FiveLevelPoolConfig::validate`], or
    /// if the backing arena cannot be allocated.
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        config.validate()?;
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = (0..num_bins)
            .map(|_| LockFreeFreeListHead::default())
            .collect();

        Ok(Self {
            config,
            memory: Arc::new(Mutex::new(memory)),
            free_lists,
            fragment_size: AtomicUsize::new(0),
            huge_free_list: Mutex::new(HugeFreeList::default()),
        })
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte request (see [`reject_zero_size`]) or
    /// when the arena is exhausted.
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        reject_zero_size(size, "allocation")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin_lockfree(aligned_size)
        } else {
            self.alloc_huge(aligned_size)
        }
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte block (see [`reject_zero_size`]).
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        reject_zero_size(size, "free")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            self.free_to_fast_bin_lockfree(offset, aligned_size)
        } else {
            self.free_huge(offset, aligned_size)
        }
    }

    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }

    fn alloc_from_fast_bin_lockfree(&self, size: usize) -> Result<MemOffset> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let head = &self.free_lists[bin_index];

            // Lock-free compare-exchange loop
            loop {
                let current_head = head.head.load(Ordering::Acquire);
                if current_head == u32::MAX {
                    break; // No free blocks
                }

                // Get next pointer from the free block
                // SAFETY: current_head from free list points to valid u32 storing next offset
                let next_head = unsafe {
                    let memory = self.memory.lock().map_err(|e| {
                        ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e))
                    })?;
                    let ptr = memory.offset_ptr(current_head as usize) as *const u32;
                    *ptr
                };

                // Try to update head atomically
                match head.head.compare_exchange_weak(
                    current_head,
                    next_head,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        head.count.fetch_sub(1, Ordering::Relaxed);
                        self.fragment_size.fetch_sub(size, Ordering::Relaxed);
                        return Ok(MemOffset::new(current_head as usize));
                    }
                    Err(_) => {
                        // Retry loop
                        std::hint::spin_loop();
                    }
                }
            }
        }

        // Fall back to mutex allocation
        let mut memory = self
            .memory
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }

    fn free_to_fast_bin_lockfree(&self, offset: MemOffset, size: usize) -> Result<()> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let head = &self.free_lists[bin_index];

            // Lock-free insertion
            loop {
                let current_head = head.head.load(Ordering::Acquire);

                // Write next pointer into freed block
                // SAFETY: offset points to freed block of size >= 4, can store u32
                unsafe {
                    let memory = self.memory.lock().map_err(|e| {
                        ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e))
                    })?;
                    let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                    *ptr = current_head;
                }

                // Try to update head atomically
                match head.head.compare_exchange_weak(
                    current_head,
                    offset.0,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        head.count.fetch_add(1, Ordering::Relaxed);
                        self.fragment_size.fetch_add(size, Ordering::Relaxed);
                        return Ok(());
                    }
                    Err(_) => {
                        // Retry loop
                        std::hint::spin_loop();
                    }
                }
            }
        }

        Ok(())
    }

    /// Serve a block above `max_fast_block_size` from the huge free list,
    /// falling back to the end of the arena.
    ///
    /// The huge-list lock is released before the memory lock is taken, so the
    /// two are never held at once and cannot deadlock against each other.
    fn alloc_huge(&self, size: usize) -> Result<MemOffset> {
        {
            let mut huge = self.huge_free_list.lock().map_err(|e| {
                ZiporaError::resource_busy(format!("Huge free list mutex poisoned: {}", e))
            })?;
            if let Some(offset) = huge.alloc(size) {
                self.fragment_size.fetch_sub(size, Ordering::Relaxed);
                return Ok(MemOffset::new(offset));
            }
        }

        let mut memory = self
            .memory
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }

    /// Return a block above `max_fast_block_size` to the huge free list.
    fn free_huge(&self, offset: MemOffset, size: usize) -> Result<()> {
        self.fragment_size.fetch_add(size, Ordering::Relaxed);
        let mut huge = self.huge_free_list.lock().map_err(|e| {
            ZiporaError::resource_busy(format!("Huge free list mutex poisoned: {}", e))
        })?;
        huge.free(offset.to_usize(), size);
        Ok(())
    }

    pub fn stats(&self) -> PoolStats {
        let memory = self.memory.lock().unwrap_or_else(|e| e.into_inner());
        let huge = self.huge_free_list.lock().unwrap_or_else(|e| e.into_inner());

        PoolStats {
            total_capacity: memory.capacity,
            used_memory: memory.size,
            fragment_size: self.fragment_size.load(Ordering::Relaxed),
            huge_size_sum: huge.total_bytes(),
            huge_node_count: huge.node_count(),
            free_list_count: self.free_lists.len(),
        }
    }
}

// SAFETY: LockFreePool is Send because:
// 1. `config: FiveLevelPoolConfig` - Config is Clone, no pointers.
// 2. `memory: Arc<Mutex<MemoryChunk>>` - Arc<Mutex<T>> is Send if T is Send.
// 3. `free_lists: Vec<LockFreeFreeListHead>` - Contains only atomics.
// 4. `fragment_size: AtomicUsize` - AtomicUsize is Send.
// 5. `huge_free_list: Mutex<HugeFreeList>` - Mutex<T> is Send if T is Send.
unsafe impl Send for LockFreePool {}

// SAFETY: LockFreePool is Sync because:
// 1. Fast bin operations use lock-free atomic CAS (AtomicU32 head/count).
// 2. Memory expansion is protected by Arc<Mutex<...>>.
// 3. Huge allocations are protected by Mutex.
// 4. AtomicUsize fragment tracking is inherently thread-safe.
// 5. Lock-free operations use proper Acquire/Release ordering.
unsafe impl Sync for LockFreePool {}

static NEXT_THREAD_LOCAL_POOL_ID: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// Level 4: Thread-local Caching - Per-thread fast-bin cache + bump slab
/// backed by a single `MutexBasedPool` address space.
///
/// C3.24 (F4) + C3.26 (S9-R1, S9-R2):
/// - Every `MemOffset` handed out by `ThreadLocalPool` comes from
///   `self.global_pool`'s single address space (either popped from a lazy
///   per-bin `Vec<MemOffset>`, bumped from a thread-local `[hot_pos..hot_end)`
///   slab carved out of `self.global_pool`, or allocated directly from
///   `self.global_pool`), so two live allocations never share a `MemOffset`.
/// - `THREAD_CACHE` is a compact `Vec<ThreadLocalCache>` whose MRU tail is
///   checked first (`last.pool_id == self.id`), avoiding a `HashMap` SipHash on
///   every `alloc`/`free`.
/// - `local_free_lists` grows lazily to `bin_index + 1` instead of
///   preallocating `max_fast_block_size / alignment` (4,096) empty `Vec`s, and
///   `push` is `O(1)` without an `O(n)` linear scan.
/// - `Drop for ThreadLocalPool` removes `self.id` from the dropping thread's
///   `THREAD_CACHE` immediately, and `Weak<MutexBasedPool>` lazily evicts dead
///   entries on any other thread that ever touched the pool.
pub struct ThreadLocalPool {
    id: u64,
    config: FiveLevelPoolConfig,
    global_pool: Arc<MutexBasedPool>,
}

thread_local! {
    static THREAD_CACHE: std::cell::RefCell<Vec<ThreadLocalCache>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

struct ThreadLocalCache {
    pool_id: u64,
    pool_weak: std::sync::Weak<MutexBasedPool>,
    hot_pos: usize,
    hot_end: usize,
    cached_bytes: usize,
    max_cached_bytes: usize,
    local_free_lists: Vec<Vec<MemOffset>>,
}

impl ThreadLocalCache {
    fn new(
        pool_id: u64,
        pool_weak: std::sync::Weak<MutexBasedPool>,
        max_cached_bytes: usize,
    ) -> Self {
        Self {
            pool_id,
            pool_weak,
            hot_pos: 0,
            hot_end: 0,
            cached_bytes: 0,
            max_cached_bytes,
            local_free_lists: Vec::new(),
        }
    }

    #[inline]
    fn pop(&mut self, bin_index: usize, block_size: usize) -> Option<MemOffset> {
        let offset = self.local_free_lists.get_mut(bin_index)?.pop()?;
        self.cached_bytes = self.cached_bytes.saturating_sub(block_size);
        Some(offset)
    }

    #[inline]
    fn alloc_from_slab(&mut self, block_size: usize) -> Option<MemOffset> {
        if self.hot_pos + block_size <= self.hot_end {
            let offset = self.hot_pos;
            self.hot_pos += block_size;
            Some(MemOffset::new(offset))
        } else {
            None
        }
    }

    #[inline]
    fn try_push(&mut self, offset: MemOffset, bin_index: usize, block_size: usize) -> bool {
        if self.cached_bytes + block_size > self.max_cached_bytes {
            return false;
        }
        if bin_index >= self.local_free_lists.len() {
            self.local_free_lists.resize_with(bin_index + 1, Vec::new);
        }
        self.local_free_lists[bin_index].push(offset);
        self.cached_bytes += block_size;
        true
    }
}

impl ThreadLocalPool {
    /// # Errors
    ///
    /// Returns an error if `config` fails [`FiveLevelPoolConfig::validate`], or
    /// if the shared global pool cannot be created.
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        config.validate()?;
        let mut global_cfg = config.clone();
        global_cfg.initial_capacity = global_cfg.initial_capacity.max(config.arena_size);
        let global_pool = Arc::new(MutexBasedPool::new(global_cfg)?);

        Ok(Self {
            id: NEXT_THREAD_LOCAL_POOL_ID.fetch_add(1, Ordering::Relaxed),
            config,
            global_pool,
        })
    }

    #[inline]
    fn with_thread_cache<R>(
        &self,
        f: impl FnOnce(&mut ThreadLocalCache) -> R,
    ) -> Option<R> {
        THREAD_CACHE
            .try_with(|caches| {
                let mut caches = caches.borrow_mut();
                if let Some(last) = caches.last_mut()
                    && last.pool_id == self.id
                {
                    return f(last);
                }
                if let Some(idx) = caches.iter().position(|c| c.pool_id == self.id) {
                    let last = caches.len() - 1;
                    caches.swap(idx, last);
                    return f(&mut caches[last]);
                }
                // Lazily prune entries whose owning ThreadLocalPool was dropped
                // on another thread before inserting a new entry.
                caches.retain(|c| c.pool_weak.strong_count() > 0);
                caches.push(ThreadLocalCache::new(
                    self.id,
                    Arc::downgrade(&self.global_pool),
                    self.config.arena_size,
                ));
                let last = caches.len() - 1;
                f(&mut caches[last])
            })
            .ok()
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte request (see [`reject_zero_size`]) or
    /// when the arena is exhausted.
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        reject_zero_size(size, "allocation")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            let bin_index = (aligned_size / self.config.alignment).saturating_sub(1);

            if let Some(Some(offset)) = self.with_thread_cache(|cache| {
                if let Some(off) = cache.pop(bin_index, aligned_size) {
                    return Some(off);
                }
                if let Some(off) = cache.alloc_from_slab(aligned_size) {
                    return Some(off);
                }
                // S10-R2: reuse a block that spilled into `global_pool`'s fast
                // bin before carving fresh bump memory.
                if let Some(off) = self.global_pool.try_pop_fast_bin(aligned_size) {
                    return Some(off);
                }
                // S10-R1: cap the per-thread slab at `arena_size / 64` so a
                // 32 KiB fast block under the default 2 MiB arena takes one
                // block (32 KiB) instead of reserving 1 MiB, while 64 B blocks
                // still get a full 64-block (4 KiB) slab.
                let max_slab = (aligned_size * 64)
                    .min(self.config.arena_size / 64)
                    .max(aligned_size);
                if let Some((start, end)) = self.global_pool.carve_slab(max_slab, aligned_size) {
                    cache.hot_pos = start + aligned_size;
                    cache.hot_end = end;
                    return Some(MemOffset::new(start));
                }
                None
            }) {
                return Ok(offset);
            }
        }

        self.global_pool.alloc(aligned_size)
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte block (see [`reject_zero_size`]).
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        reject_zero_size(size, "free")?;
        let aligned_size = self.align_up(size);

        if aligned_size <= self.config.max_fast_block_size {
            let bin_index = (aligned_size / self.config.alignment).saturating_sub(1);

            if self
                .with_thread_cache(|cache| cache.try_push(offset, bin_index, aligned_size))
                .unwrap_or(false)
            {
                return Ok(());
            }
        }

        self.global_pool.free(offset, aligned_size)
    }
    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }

    pub fn stats(&self) -> PoolStats {
        self.global_pool.stats()
    }
}

// SAFETY: ThreadLocalPool is Send because:
// 1. `id: u64` and `config: FiveLevelPoolConfig` contain no raw pointers.
// 2. `global_pool: Arc<MutexBasedPool>` - Arc<T> is Send because MutexBasedPool is Send + Sync.
// 3. Thread-local caches (THREAD_CACHE) stay on the thread that created them.
unsafe impl Send for ThreadLocalPool {}

// SAFETY: ThreadLocalPool is Sync because:
// 1. Thread-local caches (THREAD_CACHE) are accessed only by the current thread via `RefCell`.
// 2. Shared operations on `global_pool: Arc<MutexBasedPool>` are synchronized by internal `Mutex`es.
// 3. `id` and `config` are immutable after construction.
unsafe impl Sync for ThreadLocalPool {}

impl Drop for ThreadLocalPool {
    fn drop(&mut self) {
        let id = self.id;
        let _ = THREAD_CACHE.try_with(|caches| {
            caches
                .borrow_mut()
                .retain(|c| c.pool_id != id && c.pool_weak.strong_count() > 0);
        });
    }
}

/// Level 5: Fixed Capacity - Bounded memory pool for real-time systems
pub struct FixedCapacityPool {
    config: FiveLevelPoolConfig,
    inner: NoLockingPool,
    max_capacity: usize,
}

impl FixedCapacityPool {
    pub fn new(mut config: FiveLevelPoolConfig) -> Result<Self> {
        let max_capacity = config.fixed_capacity.unwrap_or(config.initial_capacity);
        config.initial_capacity = max_capacity;

        let inner = NoLockingPool::new(config.clone())?;

        Ok(Self {
            config,
            inner,
            max_capacity,
        })
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte request (see [`reject_zero_size`]) or
    /// when the fixed capacity would be exceeded.
    pub fn alloc(&mut self, size: usize) -> Result<MemOffset> {
        reject_zero_size(size, "allocation")?;
        let aligned_size = self.align_up(size);

        // Check capacity before allocation
        let stats = self.inner.stats();
        if stats.used_memory + aligned_size > self.max_capacity {
            return Err(ZiporaError::resource_exhausted("Fixed capacity exceeded"));
        }

        self.inner.alloc(aligned_size)
    }

    /// # Errors
    ///
    /// Returns an error for a zero-byte block (see [`reject_zero_size`]).
    pub fn free(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        self.inner.free(offset, size)
    }

    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }

    pub fn remaining_capacity(&self) -> usize {
        let stats = self.inner.stats();
        self.max_capacity.saturating_sub(stats.used_memory)
    }

    pub fn is_at_capacity(&self) -> bool {
        self.remaining_capacity() == 0
    }

    pub fn stats(&self) -> PoolStats {
        let mut stats = self.inner.stats();
        stats.total_capacity = self.max_capacity;
        stats
    }
}

/// Concurrency level selection for adaptive pool management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyLevel {
    /// Single-threaded operation (Level 1)
    SingleThread,
    /// Multi-threaded with mutex-based locking (Level 2)
    MultiThreadMutex,
    /// Multi-threaded with lock-free operations (Level 3)
    MultiThreadLockFree,
    /// Thread-local caching (Level 4)
    ThreadLocal,
    /// Fixed capacity for real-time systems (Level 5)
    FixedCapacity,
}

/// Pool variant enum for type-erased storage
enum PoolVariant {
    Level1(Box<NoLockingPool>),
    Level2(Arc<MutexBasedPool>),
    Level3(Arc<LockFreePool>),
    Level4(Arc<ThreadLocalPool>),
    Level5(Box<FixedCapacityPool>),
}

/// Adaptive pool manager that selects appropriate concurrency level
pub struct AdaptiveFiveLevelPool {
    level: ConcurrencyLevel,
    pool: PoolVariant,
}

impl AdaptiveFiveLevelPool {
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        let level = Self::select_optimal_level(&config);

        let pool = match level {
            ConcurrencyLevel::SingleThread => {
                PoolVariant::Level1(Box::new(NoLockingPool::new(config)?))
            }
            ConcurrencyLevel::MultiThreadMutex => {
                PoolVariant::Level2(Arc::new(MutexBasedPool::new(config)?))
            }
            ConcurrencyLevel::MultiThreadLockFree => {
                PoolVariant::Level3(Arc::new(LockFreePool::new(config)?))
            }
            ConcurrencyLevel::ThreadLocal => {
                PoolVariant::Level4(Arc::new(ThreadLocalPool::new(config)?))
            }
            ConcurrencyLevel::FixedCapacity => {
                PoolVariant::Level5(Box::new(FixedCapacityPool::new(config)?))
            }
        };

        Ok(Self { level, pool })
    }

    /// Create pool with explicit level selection
    pub fn with_level(config: FiveLevelPoolConfig, level: ConcurrencyLevel) -> Result<Self> {
        let pool = match level {
            ConcurrencyLevel::SingleThread => {
                PoolVariant::Level1(Box::new(NoLockingPool::new(config)?))
            }
            ConcurrencyLevel::MultiThreadMutex => {
                PoolVariant::Level2(Arc::new(MutexBasedPool::new(config)?))
            }
            ConcurrencyLevel::MultiThreadLockFree => {
                PoolVariant::Level3(Arc::new(LockFreePool::new(config)?))
            }
            ConcurrencyLevel::ThreadLocal => {
                PoolVariant::Level4(Arc::new(ThreadLocalPool::new(config)?))
            }
            ConcurrencyLevel::FixedCapacity => {
                PoolVariant::Level5(Box::new(FixedCapacityPool::new(config)?))
            }
        };

        Ok(Self { level, pool })
    }

    fn select_optimal_level(config: &FiveLevelPoolConfig) -> ConcurrencyLevel {
        // Sophisticated heuristic for level selection
        if config.fixed_capacity.is_some() {
            return ConcurrencyLevel::FixedCapacity;
        }

        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        match cpu_count {
            1 => ConcurrencyLevel::SingleThread,
            2..=4 => {
                // For small numbers of cores, use mutex-based approach
                if config.max_fast_block_size > 16 * 1024 {
                    ConcurrencyLevel::MultiThreadMutex
                } else {
                    ConcurrencyLevel::MultiThreadLockFree
                }
            }
            5..=16 => {
                // Medium core count benefits from lock-free or thread-local
                if config.arena_size > 1024 * 1024 {
                    ConcurrencyLevel::ThreadLocal
                } else {
                    ConcurrencyLevel::MultiThreadLockFree
                }
            }
            _ => {
                // High core count definitely benefits from thread-local caching
                ConcurrencyLevel::ThreadLocal
            }
        }
    }

    pub fn alloc(&mut self, size: usize) -> Result<MemOffset> {
        match &mut self.pool {
            PoolVariant::Level1(pool) => pool.alloc(size),
            PoolVariant::Level2(pool) => pool.alloc(size),
            PoolVariant::Level3(pool) => pool.alloc(size),
            PoolVariant::Level4(pool) => pool.alloc(size),
            PoolVariant::Level5(pool) => pool.alloc(size),
        }
    }

    pub fn free(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        match &mut self.pool {
            PoolVariant::Level1(pool) => pool.free(offset, size),
            PoolVariant::Level2(pool) => pool.free(offset, size),
            PoolVariant::Level3(pool) => pool.free(offset, size),
            PoolVariant::Level4(pool) => pool.free(offset, size),
            PoolVariant::Level5(pool) => pool.free(offset, size),
        }
    }

    pub fn current_level(&self) -> ConcurrencyLevel {
        self.level
    }

    pub fn stats(&self) -> PoolStats {
        match &self.pool {
            PoolVariant::Level1(pool) => pool.stats(),
            PoolVariant::Level2(pool) => pool.stats(),
            PoolVariant::Level3(pool) => pool.stats(),
            PoolVariant::Level4(pool) => pool.stats(),
            PoolVariant::Level5(pool) => pool.stats(),
        }
    }

    /// Get a cloneable handle for multi-threaded pools
    pub fn get_handle(&self) -> Result<FiveLevelPoolHandle> {
        match &self.pool {
            PoolVariant::Level2(pool) => Ok(FiveLevelPoolHandle::Level2(Arc::clone(pool))),
            PoolVariant::Level3(pool) => Ok(FiveLevelPoolHandle::Level3(Arc::clone(pool))),
            PoolVariant::Level4(pool) => Ok(FiveLevelPoolHandle::Level4(Arc::clone(pool))),
            _ => Err(ZiporaError::invalid_data(
                "Pool level doesn't support handles",
            )),
        }
    }
}

/// Handle for multi-threaded pool access
#[derive(Clone)]
pub enum FiveLevelPoolHandle {
    Level2(Arc<MutexBasedPool>),
    Level3(Arc<LockFreePool>),
    Level4(Arc<ThreadLocalPool>),
}

impl FiveLevelPoolHandle {
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        match self {
            FiveLevelPoolHandle::Level2(pool) => pool.alloc(size),
            FiveLevelPoolHandle::Level3(pool) => pool.alloc(size),
            FiveLevelPoolHandle::Level4(pool) => pool.alloc(size),
        }
    }

    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        match self {
            FiveLevelPoolHandle::Level2(pool) => pool.free(offset, size),
            FiveLevelPoolHandle::Level3(pool) => pool.free(offset, size),
            FiveLevelPoolHandle::Level4(pool) => pool.free(offset, size),
        }
    }

    pub fn stats(&self) -> PoolStats {
        match self {
            FiveLevelPoolHandle::Level2(pool) => pool.stats(),
            FiveLevelPoolHandle::Level3(pool) => pool.stats(),
            FiveLevelPoolHandle::Level4(pool) => pool.stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert that `config` is rejected by every pool constructor that takes
    /// one. `FiveLevelPoolConfig` fields are all `pub`, so any of these values
    /// is reachable from safe code.
    fn expect_config_rejected(config: FiveLevelPoolConfig, why: &str) {
        let err = NoLockingPool::new(config.clone())
            .err()
            .unwrap_or_else(|| panic!("NoLockingPool::new accepted a config that {why}"));
        let message = err.to_string();
        assert!(
            MutexBasedPool::new(config.clone()).is_err(),
            "MutexBasedPool::new accepted a config that {why}"
        );
        assert!(
            LockFreePool::new(config.clone()).is_err(),
            "LockFreePool::new accepted a config that {why}"
        );
        assert!(
            ThreadLocalPool::new(config).is_err(),
            "ThreadLocalPool::new accepted a config that {why}"
        );
        assert!(
            message.to_lowercase().contains("align")
                || message.to_lowercase().contains("capacity")
                || message.to_lowercase().contains("size"),
            "error should name the offending field, got: {message}"
        );
    }

    /// C3.5. `alloc_from_fast_bin` computes `bin_index = size / alignment - 1`.
    /// For a zero-byte request `align_up(0)` is 0, so the subtraction underflows:
    /// a debug build panics with "attempt to subtract with overflow" and a
    /// release build wraps to `usize::MAX`, silently falling through to a
    /// zero-width carve from the end of the arena. `alloc(0)` is reachable from
    /// safe public API on every level.
    #[test]
    fn test_zero_sized_allocation_is_rejected_not_wrapped() {
        let mut level1 = NoLockingPool::new(FiveLevelPoolConfig::default()).unwrap();
        assert!(
            level1.alloc(0).is_err(),
            "NoLockingPool::alloc(0) must return an error, not underflow the bin index"
        );

        let level2 = MutexBasedPool::new(FiveLevelPoolConfig::default()).unwrap();
        assert!(
            level2.alloc(0).is_err(),
            "MutexBasedPool::alloc(0) must return an error"
        );

        let level3 = LockFreePool::new(FiveLevelPoolConfig::default()).unwrap();
        assert!(
            level3.alloc(0).is_err(),
            "LockFreePool::alloc(0) must return an error"
        );
    }

    /// C3.5. `FiveLevelPoolConfig::alignment` is a `pub` field with no
    /// validation anywhere. `MemoryChunk::new` is the only thing that ever
    /// looks at it, and it only rejects values `Layout` rejects. Zero reaches
    /// `num_bins = max_fast_block_size / alignment` first and divides by zero.
    #[test]
    fn test_zero_alignment_is_rejected() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                alignment: 0,
                ..FiveLevelPoolConfig::default()
            },
            "has a zero alignment",
        );
    }

    /// C3.5. A non-power-of-two alignment makes `align_up`'s
    /// `(size + a - 1) & !(a - 1)` mask meaningless, so the bin index and the
    /// carve width stop agreeing.
    #[test]
    fn test_non_power_of_two_alignment_is_rejected() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                alignment: 24,
                ..FiveLevelPoolConfig::default()
            },
            "has a non-power-of-two alignment",
        );
    }

    /// C3.5. The free-list link is a `u32`, so an alignment below 4 cannot
    /// hold it.
    #[test]
    fn test_alignment_below_the_link_width_is_rejected() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                alignment: 2,
                ..FiveLevelPoolConfig::default()
            },
            "has an alignment smaller than the free-list link",
        );
    }

    /// C3.5. `initial_capacity: 0` builds a valid zero-sized `Layout` and
    /// hands it to `std::alloc::alloc`, which is undefined behaviour: the
    /// `GlobalAlloc` contract requires `layout.size() != 0`. Oracle:
    /// `make miri_pool`.
    #[test]
    fn test_zero_capacity_is_rejected_before_allocating() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                initial_capacity: 0,
                ..FiveLevelPoolConfig::default()
            },
            "has a zero initial capacity",
        );
    }

    /// C3.5. Offsets are `u32`. `MemOffset::new` guards the narrowing with a
    /// `debug_assert!`, which is compiled out in release, so a pool larger than
    /// 4 GiB silently truncates offsets and hands the same `MemOffset` out for
    /// two different live blocks. Working agreement 4 forbids `debug_assert!`
    /// as the only guard on a value reachable from safe public API, so the
    /// capacity has to be capped at construction instead.
    #[test]
    fn test_capacity_beyond_the_offset_space_is_rejected() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                initial_capacity: (u32::MAX as usize) + 1,
                ..FiveLevelPoolConfig::default()
            },
            "exceeds the 32-bit offset space",
        );
    }

    /// C3.5. `max_fast_block_size` drives `num_bins`, and a value that is not a
    /// whole number of `alignment` units leaves the top bin unreachable while
    /// still sizing the vector for it.
    #[test]
    fn test_fast_block_size_must_be_a_multiple_of_alignment() {
        expect_config_rejected(
            FiveLevelPoolConfig {
                max_fast_block_size: 100,
                ..FiveLevelPoolConfig::default()
            },
            "has a fast block size that is not a multiple of the alignment",
        );
    }

    /// Deterministic xorshift64* so the randomized model test below is
    /// reproducible without a dependency.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Larger than `FiveLevelPoolConfig::default().max_fast_block_size`
    /// (32 KiB), so every request of this size takes the huge path.
    const HUGE: usize = 64 * 1024;

    /// C3.6 (D3). `NoLockingPool::free_to_skip_list`,
    /// `MutexBasedPool::free_to_skip_list` and `LockFreePool::free_to_huge_mutex`
    /// all took the offset by `_offset` and dropped it, behind a
    /// `// TODO: Implement skip list insertion` marker, while still charging
    /// `used_memory -= size` and `fragment_size += size`. The matching
    /// `alloc_from_skip_list` always bumped from the end of the arena. Every
    /// free above `max_fast_block_size` therefore lost its block permanently,
    /// and a pool that reports plenty of free capacity runs out of memory.
    ///
    /// Each iteration parks a small guard block above the huge one so that
    /// `free` cannot take the tail-rollback fast path and has to go through
    /// the huge free list.
    #[test]
    fn test_large_blocks_are_reused_after_free_level1() {
        let mut pool = NoLockingPool::new(FiveLevelPoolConfig {
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        let mut guards = Vec::new();
        for iteration in 0..100 {
            let huge = pool.alloc(HUGE).unwrap_or_else(|e| {
                panic!(
                    "iteration {iteration}: a 64 KiB allocation failed in a 1 MiB \
                     arena after {iteration} complete alloc/free cycles: {e}"
                )
            });
            guards.push(pool.alloc(64).unwrap());
            pool.free(huge, HUGE).unwrap();
        }
        drop(guards);
    }

    /// C3.6 (D3). Same leak, Level 2.
    #[test]
    fn test_large_blocks_are_reused_after_free_level2() {
        let pool = MutexBasedPool::new(FiveLevelPoolConfig {
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        for iteration in 0..100 {
            let huge = pool.alloc(HUGE).unwrap_or_else(|e| {
                panic!("iteration {iteration}: 64 KiB allocation failed: {e}")
            });
            pool.free(huge, HUGE).unwrap();
        }
    }

    /// C3.6 (D3). Same leak, Level 3.
    #[test]
    fn test_large_blocks_are_reused_after_free_level3() {
        let pool = LockFreePool::new(FiveLevelPoolConfig {
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        for iteration in 0..100 {
            let huge = pool.alloc(HUGE).unwrap_or_else(|e| {
                panic!("iteration {iteration}: 64 KiB allocation failed: {e}")
            });
            pool.free(huge, HUGE).unwrap();
        }
    }

    /// C3.6 (D3). Adjacent freed regions must coalesce, otherwise an arena that
    /// has been cycled through small huge-path blocks can never satisfy a
    /// larger one again even though the bytes are contiguous and free.
    #[test]
    fn test_adjacent_large_frees_coalesce() {
        let mut pool = NoLockingPool::new(FiveLevelPoolConfig {
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        // Three adjacent huge blocks, plus a guard so no free hits the
        // tail-rollback path.
        let first = pool.alloc(HUGE).unwrap();
        let second = pool.alloc(HUGE).unwrap();
        let third = pool.alloc(HUGE).unwrap();
        let _guard = pool.alloc(64).unwrap();
        assert_eq!(second.to_usize(), first.to_usize() + HUGE);
        assert_eq!(third.to_usize(), second.to_usize() + HUGE);

        pool.free(second, HUGE).unwrap();
        pool.free(first, HUGE).unwrap();
        pool.free(third, HUGE).unwrap();

        let merged = pool.alloc(3 * HUGE).unwrap();
        assert_eq!(
            merged.to_usize(),
            first.to_usize(),
            "three adjacent 64 KiB frees must coalesce into one 192 KiB region"
        );
    }

    /// C3.6 (D3). The huge path must carve exactly what was asked for. If a
    /// best-fit region larger than the request were handed over whole, the
    /// caller's `free(offset, size)` would file back only `size` and the
    /// region would shrink on every reuse cycle -- the carve/recycle width
    /// mismatch that this stage is auditing for.
    #[test]
    fn test_huge_carve_width_equals_recycle_width() {
        let mut pool = NoLockingPool::new(FiveLevelPoolConfig {
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        let big = pool.alloc(256 * 1024).unwrap();
        let _guard = pool.alloc(64).unwrap();
        pool.free(big, 256 * 1024).unwrap();

        // Carve a smaller block out of the freed region, hand it straight
        // back, and require the region to be whole again.
        for cycle in 0..8 {
            let small = pool.alloc(HUGE).unwrap();
            assert_eq!(
                small.to_usize(),
                big.to_usize(),
                "cycle {cycle}: best fit should reuse the head of the freed region"
            );
            pool.free(small, HUGE).unwrap();
            let whole = pool.alloc(256 * 1024).unwrap_or_else(|e| {
                panic!("cycle {cycle}: the 256 KiB region shrank across reuse: {e}")
            });
            assert_eq!(whole.to_usize(), big.to_usize());
            pool.free(whole, 256 * 1024).unwrap();
        }
    }

    /// C3.6 (D3). Randomized model check: no two live blocks may ever overlap,
    /// whichever path served them. Interleaves fast-bin and huge-path traffic
    /// so the tail-rollback fast path, the fast bins and the huge free list all
    /// participate.
    #[test]
    fn test_no_two_live_blocks_ever_overlap() {
        let mut pool = NoLockingPool::new(FiveLevelPoolConfig {
            initial_capacity: 4 * 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        })
        .unwrap();

        let mut live: Vec<(usize, usize)> = Vec::new();
        let mut state = 0x2545_F491_4F6C_DD1D_u64;

        for step in 0..4000 {
            let roll = xorshift(&mut state);
            let should_free = !live.is_empty() && (roll & 1) == 1;

            if should_free {
                let index = (xorshift(&mut state) as usize) % live.len();
                let (offset, size) = live.swap_remove(index);
                pool.free(MemOffset::new(offset), size).unwrap();
                continue;
            }

            // Half fast-bin sizes, half huge sizes, always a multiple of the
            // 8-byte alignment so the request width is the carve width.
            let size = if (roll >> 1) & 1 == 0 {
                8 + 8 * ((xorshift(&mut state) as usize) % 512)
            } else {
                40 * 1024 + 8 * ((xorshift(&mut state) as usize) % 4096)
            };

            let Ok(offset) = pool.alloc(size) else {
                continue; // genuine exhaustion is not a defect
            };
            let offset = offset.to_usize();

            for &(other, other_size) in &live {
                assert!(
                    offset + size <= other || other + other_size <= offset,
                    "step {step}: new block [{offset}, {}) overlaps live block \
                     [{other}, {})",
                    offset + size,
                    other + other_size
                );
            }
            live.push((offset, size));
        }
    }

    /// C3.2. `MemoryChunk::new` allocates with `config.alignment` (8 by
    /// default, 16 for `performance_optimized`), but `Drop` deallocated with
    /// `align_of::<u8>()` = 1. `GlobalAlloc::dealloc` requires the *same*
    /// layout that was used to allocate, so every pool construction/drop was
    /// undefined behaviour. Oracle: `make miri_pool`.
    #[test]
    fn test_pool_drop_uses_the_allocation_layout() {
        for config in [
            FiveLevelPoolConfig::default(),
            FiveLevelPoolConfig::performance_optimized(),
            FiveLevelPoolConfig::memory_optimized(),
            FiveLevelPoolConfig {
                alignment: 64,
                initial_capacity: 64 * 1024,
                ..FiveLevelPoolConfig::default()
            },
        ] {
            let alignment = config.alignment;
            let pool = NoLockingPool::new(config).unwrap();
            assert_eq!(
                pool.memory.data.as_ptr() as usize % alignment,
                0,
                "chunk base must honour the configured alignment {alignment}"
            );
            drop(pool);
        }
    }

    #[test]
    fn test_no_locking_pool_basic() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let mut pool = NoLockingPool::new(config)?;

        // Test allocation
        let offset1 = pool.alloc(64)?;
        let offset2 = pool.alloc(128)?;

        assert_ne!(offset1.to_usize(), offset2.to_usize());

        // Test deallocation
        pool.free(offset1, 64)?;
        pool.free(offset2, 128)?;

        // Test reallocation - should reuse the 64-byte block from bin 7
        let offset3 = pool.alloc(64)?;
        // Should reuse the freed 64-byte block (same size class)
        assert_eq!(offset3.to_usize(), offset1.to_usize());

        // Test reallocation of 128-byte block
        let offset4 = pool.alloc(128)?;
        // Should reuse the freed 128-byte block (same size class)
        assert_eq!(offset4.to_usize(), offset2.to_usize());

        Ok(())
    }

    #[test]
    fn test_mutex_based_pool_concurrent() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = Arc::new(MutexBasedPool::new(config)?);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || -> Result<()> {
                    for _ in 0..100 {
                        let offset = pool.alloc(64)?;
                        pool.free(offset, 64)?;
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    }

    #[test]
    fn test_lock_free_pool_concurrent() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = Arc::new(LockFreePool::new(config)?);

        #[cfg(not(miri))]
        const THREADS: usize = 8;
        #[cfg(miri)]
        const THREADS: usize = 4;
        #[cfg(not(miri))]
        const ITERS: usize = 1000;
        #[cfg(miri)]
        const ITERS: usize = 32;

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || -> Result<()> {
                    for _ in 0..ITERS {
                        let offset = pool.alloc(64)?;
                        pool.free(offset, 64)?;
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    }

    #[test]
    fn test_thread_local_pool() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = Arc::new(ThreadLocalPool::new(config)?);

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || -> Result<()> {
                    // Each thread should use its own cache
                    for _ in 0..100 {
                        let offset = pool.alloc(128)?;
                        pool.free(offset, 128)?;
                    }
                    Ok(())
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    }

    #[test]
    fn test_fixed_capacity_pool() -> Result<()> {
        let config = FiveLevelPoolConfig {
            fixed_capacity: Some(8192), // 8KB fixed capacity
            ..Default::default()
        };

        let mut pool = FixedCapacityPool::new(config)?;

        // Should be able to allocate within capacity
        let offset1 = pool.alloc(1024)?;
        let _offset2 = pool.alloc(2048)?;

        assert_eq!(pool.remaining_capacity(), 8192 - 1024 - 2048);

        // Should fail when exceeding capacity
        let result = pool.alloc(6000); // Would exceed remaining capacity
        assert!(result.is_err());

        // Free some memory
        pool.free(offset1, 1024)?;
        assert_eq!(pool.remaining_capacity(), 8192 - 2048);

        // Should be able to allocate again
        let _offset3 = pool.alloc(1024)?;

        Ok(())
    }

    #[test]
    fn test_all_levels_explicit_selection() -> Result<()> {
        let config = FiveLevelPoolConfig::default();

        let levels = [
            ConcurrencyLevel::SingleThread,
            ConcurrencyLevel::MultiThreadMutex,
            ConcurrencyLevel::MultiThreadLockFree,
            ConcurrencyLevel::ThreadLocal,
        ];

        for level in levels {
            let mut pool = AdaptiveFiveLevelPool::with_level(config.clone(), level)?;

            let offset = pool.alloc(256)?;
            pool.free(offset, 256)?;

            assert_eq!(pool.current_level(), level);
        }

        // Test fixed capacity separately
        let mut fixed_config = config.clone();
        fixed_config.fixed_capacity = Some(4096);
        let mut fixed_pool =
            AdaptiveFiveLevelPool::with_level(fixed_config, ConcurrencyLevel::FixedCapacity)?;

        let offset = fixed_pool.alloc(1024)?;
        fixed_pool.free(offset, 1024)?;
        assert_eq!(fixed_pool.current_level(), ConcurrencyLevel::FixedCapacity);

        Ok(())
    }

    #[test]
    fn test_adaptive_pool_selection() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let mut pool = AdaptiveFiveLevelPool::new(config)?;

        // Test basic allocation
        let offset = pool.alloc(256)?;
        pool.free(offset, 256)?;

        // Selection depends on CPU core count and config
        let level = pool.current_level();
        assert!(matches!(
            level,
            ConcurrencyLevel::SingleThread
                | ConcurrencyLevel::MultiThreadMutex
                | ConcurrencyLevel::MultiThreadLockFree
                | ConcurrencyLevel::ThreadLocal
        ));

        Ok(())
    }

    #[test]
    fn test_configuration_presets() -> Result<()> {
        let configs = [
            FiveLevelPoolConfig::performance_optimized(),
            FiveLevelPoolConfig::memory_optimized(),
            FiveLevelPoolConfig::realtime(),
        ];

        for config in configs {
            let mut pool = AdaptiveFiveLevelPool::new(config)?;
            let offset = pool.alloc(1024)?;
            pool.free(offset, 1024)?;
        }

        Ok(())
    }

    #[test]
    fn test_pool_handles() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = AdaptiveFiveLevelPool::with_level(config, ConcurrencyLevel::MultiThreadMutex)?;

        let handle = pool.get_handle()?;

        // Test concurrent access through handles
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let handle = handle.clone();
                std::thread::spawn(move || -> Result<()> {
                    for _ in 0..50 {
                        let offset = handle.alloc(64)?;
                        handle.free(offset, 64)?;
                    }
                    Ok(())
                })
            })
            .collect();

        for thread_handle in handles {
            thread_handle.join().unwrap()?;
        }

        Ok(())
    }

    /// C3.24 (F4). `ThreadLocalPool` handed out offsets from two independent
    /// zero-based address spaces -- `ThreadLocalCache::hot_pos` (`0..arena_size/2`)
    /// and `global_pool` (`0..initial_capacity`) -- wrapped in the same
    /// `MemOffset` type, and `free` routed by `offset.to_usize() < cache.arena.len()`.
    ///
    /// As soon as the local hot area (`arena_size / 2` bytes) fills, the very
    /// next small allocation falls back to `global_pool.alloc()`, which starts
    /// its own bump cursor at `0` and returns `MemOffset(0)` while the first
    /// local allocation at `MemOffset(0)` is still live.
    /// C3.26 (S9-R1, MEDIUM regression in C3.24). `THREAD_CACHE` was keyed by
    /// `pool_id` with no `Drop` on `ThreadLocalPool` and no eviction of dead
    /// pools, and each entry preallocated `max_fast_block_size / alignment`
    /// (4,096) bin `Vec`s (~96 KiB). Running 1,000 create/alloc/free/drop
    /// cycles on one thread grew `THREAD_CACHE` by 1,000 entries (~96 MB).
    /// C3.27 (S10-R1, MEDIUM regression in C3.26). `max_slab` was capped at
    /// `arena_size / 2` (1 MiB under the default 2 MiB config), so each thread
    /// allocating a single 32 KiB fast block carved a 1 MiB slab and two
    /// threads exhausted the entire 2 MiB shared pool at 64 KiB (3%) live.
    #[test]
    fn test_thread_local_pool_large_fast_blocks_do_not_exhaust_shared_arena() {
        let pool = Arc::new(ThreadLocalPool::new(FiveLevelPoolConfig::default()).unwrap());
        let barrier = Arc::new(std::sync::Barrier::new(4));

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let res = pool.alloc(32 * 1024);
                    barrier.wait();
                    let off = res.unwrap_or_else(|e| {
                        panic!("thread {t}: 32 KiB alloc in 2 MiB pool failed: {e}")
                    });
                    pool.free(off, 32 * 1024).unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    /// C3.27 (S10-R2, LOW regression in C3.26). `ThreadLocalPool::alloc`
    /// carved fresh slabs from `global_pool` before checking `global_pool`'s
    /// fast-bin free list, so blocks that spilled to `global_pool` on `free`
    /// were not reused until the bump cursor hit capacity.
    #[test]
    fn test_thread_local_pool_reuses_global_fast_bin_before_carving_new_slabs() {
        let config = FiveLevelPoolConfig {
            arena_size: 64 * 1024,
            initial_capacity: 1024 * 1024,
            ..FiveLevelPoolConfig::default()
        };
        let pool = ThreadLocalPool::new(config).unwrap();

        for round in 0..3 {
            let mut offsets = Vec::with_capacity(8000);
            for _ in 0..8000 {
                offsets.push(pool.alloc(64).unwrap());
            }
            let stats_live = pool.stats();
            assert_eq!(
                stats_live.used_memory,
                8000 * 64,
                "round {round}: used_memory climbed to {} instead of reusing spilled global fast-bin blocks",
                stats_live.used_memory
            );
            assert_eq!(
                stats_live.fragment_size, 0,
                "round {round}: fragment_size remained {} while all 8000 blocks were live",
                stats_live.fragment_size
            );
            for off in offsets {
                pool.free(off, 64).unwrap();
            }
        }
    }

    #[test]
    fn test_thread_local_pool_drop_evicts_thread_cache_entries() {
        let before = THREAD_CACHE.with(|c| c.borrow().len());

        for _ in 0..200 {
            let pool = ThreadLocalPool::new(FiveLevelPoolConfig::default()).unwrap();
            let off = pool.alloc(64).unwrap();
            pool.free(off, 64).unwrap();
            drop(pool);
        }

        let after = THREAD_CACHE.with(|c| c.borrow().len());
        assert_eq!(
            after, before,
            "dropping 200 ThreadLocalPools leaked {} entries in THREAD_CACHE",
            after.saturating_sub(before)
        );
    }

    #[test]
    fn test_thread_local_pool_never_hands_out_duplicate_live_offsets() {
        let config = FiveLevelPoolConfig {
            max_fast_block_size: 64,
            alignment: 8,
            initial_capacity: 1024,
            arena_size: 128, // hot_end = 64 -> holds 1 x 64-byte block before spilling
            fixed_capacity: None,
            enable_cache_alignment: false,
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 2 * 1024 * 1024,
        };
        let pool = ThreadLocalPool::new(config).unwrap();

        let first = pool.alloc(64).unwrap();
        let second = pool.alloc(64).unwrap();
        assert_ne!(
            first, second,
            "two simultaneously live allocations from ThreadLocalPool collided at {first:?}"
        );

        pool.free(first, 64).unwrap();
        pool.free(second, 64).unwrap();
    }

    #[test]
    fn test_memory_alignment() -> Result<()> {
        let config = FiveLevelPoolConfig {
            alignment: 16,
            ..Default::default()
        };

        let mut pool = NoLockingPool::new(config)?;

        let offset = pool.alloc(17)?; // Request 17 bytes

        // Should be aligned to 16 bytes
        assert_eq!(offset.to_usize() % 16, 0);

        pool.free(offset, 17)?;

        Ok(())
    }

    #[test]
    fn test_large_allocations() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let mut pool = NoLockingPool::new(config)?;

        // Test allocation larger than fast bin limit
        let large_size = 128 * 1024; // 128KB
        let offset = pool.alloc(large_size)?;

        let stats = pool.stats();
        assert!(stats.used_memory >= large_size);

        pool.free(offset, large_size)?;

        Ok(())
    }

    #[test]
    fn test_fragmentation_tracking() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let mut pool = NoLockingPool::new(config)?;

        // Allocate and free to create fragmentation
        let offset1 = pool.alloc(64)?;
        let offset2 = pool.alloc(128)?;
        let offset3 = pool.alloc(64)?;

        pool.free(offset2, 128)?; // Free middle block

        let stats = pool.stats();
        assert!(stats.fragment_size > 0);
        assert!(stats.fragmentation_ratio() > 0.0);

        pool.free(offset1, 64)?;
        pool.free(offset3, 64)?;

        Ok(())
    }

    #[test]
    fn test_pool_stats() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let mut pool = NoLockingPool::new(config)?;

        let stats_before = pool.stats();
        assert_eq!(stats_before.used_memory, 0);
        assert_eq!(stats_before.utilization(), 0.0);

        let _offset = pool.alloc(1024)?;
        let stats_after = pool.stats();
        assert!(stats_after.used_memory >= 1024);
        assert!(stats_after.utilization() > 0.0);

        Ok(())
    }

    #[test]
    fn test_stress_test_single_thread() -> Result<()> {
        let config = FiveLevelPoolConfig {
            // Increase initial capacity for stress test
            initial_capacity: 8 * 1024 * 1024, // 8MB
            ..Default::default()
        };
        let mut pool = NoLockingPool::new(config)?;

        const NUM_ALLOCS: usize = 1000; // Reduced for more manageable test
        let mut offsets = Vec::with_capacity(NUM_ALLOCS);

        // Allocate many blocks
        for i in 0..NUM_ALLOCS {
            let size = 64 + (i % 256); // Variable sizes 64-319 bytes
            let offset = pool.alloc(size)?;
            offsets.push((offset, size));
        }

        // Free them all
        for (offset, size) in offsets {
            pool.free(offset, size)?;
        }

        let stats = pool.stats();
        assert!(stats.fragment_size > 0); // Should have some fragmentation

        Ok(())
    }
}
