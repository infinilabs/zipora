//! Fixed capacity memory pool for predictable allocation
//!
//! This module provides memory pools with fixed capacity that guarantee
//! predictable allocation behavior and bounded memory usage.
//!
//! # Use Cases
//!
//! - **Real-time systems**: Predictable allocation with no dynamic growth
//! - **Embedded systems**: Bounded memory usage with compile-time guarantees
//! - **Resource quotas**: Enforce strict memory limits per component
//! - **Testing environments**: Reproducible allocation patterns
//!
//! # Architecture
//!
//! - **Fixed-size blocks**: All allocations are from pre-allocated chunks
//! - **Free list management**: Efficient O(1) allocation/deallocation
//! - **Capacity enforcement**: Hard limits prevent memory growth
//! - **Fragmentation control**: Size classes minimize fragmentation

use crate::error::{Result, ZiporaError};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering, compiler_fence};
use std::sync::{Arc, Mutex, OnceLock};

/// Alignment for fixed capacity allocations
const ALIGN_SIZE: usize = 8;
/// Minimum alignment enforced by configuration validation. `secure_clear`
/// zeroes blocks with `u32` writes when a block is not 8-byte aligned, so this
/// must stay at least `align_of::<u32>()`.
const MIN_BLOCK_ALIGN: usize = 4;
/// Minimum block size enforced by configuration validation
const MIN_BLOCK_SIZE: usize = 16;
/// Sentinel value for free list termination (`u32::MAX` is never a valid block index)
const LIST_TAIL: u32 = u32::MAX;
/// Per-block state: block is currently on a free list (or in transition while unlinked)
const FREE: u8 = 0;
/// Per-block state: block is currently checked out to a live allocation
const LIVE: u8 = 1;
/// Configuration for fixed capacity memory pool
#[derive(Debug, Clone)]
pub struct FixedCapacityPoolConfig {
    /// Maximum block size supported by this pool
    pub max_block_size: usize,
    /// Total number of blocks to pre-allocate
    pub total_blocks: usize,
    /// Alignment requirement for allocations
    pub alignment: usize,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Pre-allocate all memory on creation
    pub eager_allocation: bool,
    /// Use secure memory clearing on deallocation
    pub secure_clear: bool,
}

impl Default for FixedCapacityPoolConfig {
    fn default() -> Self {
        Self {
            max_block_size: 4096,
            total_blocks: 1000,
            alignment: ALIGN_SIZE,
            enable_stats: true,
            eager_allocation: true,
            secure_clear: false,
        }
    }
}

impl FixedCapacityPoolConfig {
    /// Create configuration for small objects (≤ 1KB)
    pub fn small_objects() -> Self {
        Self {
            max_block_size: 1024,
            total_blocks: 10000,
            alignment: 8,
            enable_stats: true,
            eager_allocation: true,
            secure_clear: false,
        }
    }

    /// Create configuration for medium objects (≤ 64KB)
    pub fn medium_objects() -> Self {
        Self {
            max_block_size: 64 * 1024,
            total_blocks: 1000,
            alignment: 16,
            enable_stats: true,
            eager_allocation: true,
            secure_clear: false,
        }
    }

    /// Create configuration for real-time systems
    pub fn realtime() -> Self {
        Self {
            max_block_size: 8192,
            total_blocks: 5000,
            alignment: 64,       // Cache line aligned
            enable_stats: false, // Minimize overhead
            eager_allocation: true,
            secure_clear: false,
        }
    }

    /// Create configuration for secure systems
    pub fn secure() -> Self {
        Self {
            max_block_size: 4096,
            total_blocks: 2000,
            alignment: 8,
            enable_stats: true,
            eager_allocation: true,
            secure_clear: true, // Clear memory on deallocation
        }
    }
}

/// Statistics for fixed capacity pool
#[derive(Debug, Default)]
pub struct FixedCapacityPoolStats {
    /// Total allocations served
    pub allocations: AtomicU64,
    /// Total deallocations processed
    pub deallocations: AtomicU64,
    /// Current number of allocated blocks
    pub active_blocks: AtomicUsize,
    /// Peak number of allocated blocks
    pub peak_blocks: AtomicUsize,
    /// Allocation failures due to capacity
    pub allocation_failures: AtomicU64,
    /// Memory utilization (allocated / total capacity)
    pub utilization: AtomicU32, // As percentage * 100
}

impl FixedCapacityPoolStats {
    /// Get current utilization as percentage (0.0 to 100.0)
    pub fn utilization_percent(&self) -> f64 {
        self.utilization.load(Ordering::Relaxed) as f64 / 100.0
    }

    /// Get allocation success rate
    pub fn success_rate(&self) -> f64 {
        let successes = self.allocations.load(Ordering::Relaxed);
        let failures = self.allocation_failures.load(Ordering::Relaxed);
        let total = successes + failures;
        if total == 0 {
            1.0
        } else {
            successes as f64 / total as f64
        }
    }

    /// Check if pool is at capacity
    pub fn is_at_capacity(&self, total_blocks: usize) -> bool {
        self.active_blocks.load(Ordering::Relaxed) >= total_blocks
    }
}

/// Heap arena and out-of-band per-block metadata owned by [`Inner`].
struct Backing {
    base: NonNull<u8>,
    layout: Layout,
    /// Out-of-band next-free-block index (`0..total_blocks` or `LIST_TAIL`).
    next: Box<[AtomicU32]>,
    /// Per-block liveness state (`FREE` or `LIVE`).
    state: Box<[AtomicU8]>,
}

// SAFETY: `base` owns a heap allocation of `layout` whose disjoint per-block
// slices are handed out exclusively through atomic `heads`, `next`, and `state`.
unsafe impl Send for Backing {}

// SAFETY: Shared references to `Backing` never alias mutable user slices; block
// ownership transitions are synchronized via AcqRel operations on `heads` and `state`.
unsafe impl Sync for Backing {}

impl Drop for Backing {
    fn drop(&mut self) {
        // SAFETY: `base` was allocated with `alloc_zeroed(self.layout)` in
        // `Inner::build_backing`, `self.layout.size() > 0`, and `Backing` drops
        // only when the last `Arc<Inner>` (the pool and all live allocations) drops.
        unsafe {
            dealloc(self.base.as_ptr(), self.layout);
        }
    }
}

struct Inner {
    config: FixedCapacityPoolConfig,
    size_classes: Vec<usize>,
    stats: Option<Arc<FixedCapacityPoolStats>>,
    /// Per-class Treiber stack head: `(tag << 32) | u64::from(head_idx)`.
    heads: Box<[AtomicU64]>,
    counts: Box<[AtomicU32]>,
    backing: OnceLock<Backing>,
    init: Mutex<()>,
}

impl Inner {
    fn build_backing(&self) -> Result<Backing> {
        let total_size = self.config.total_blocks * self.config.max_block_size;
        let layout = Layout::from_size_align(total_size, self.config.alignment)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {}", e)))?;

        // SAFETY: `validate_config` verified `total_size > 0`, `alignment` is a power
        // of two, and `Layout::from_size_align(total_size, alignment)` succeeded.
        let raw = unsafe { alloc_zeroed(layout) };
        let base = NonNull::new(raw).ok_or_else(|| ZiporaError::out_of_memory(total_size))?;

        let total_blocks = self.config.total_blocks;
        let next: Box<[AtomicU32]> = (0..total_blocks)
            .map(|i| {
                let link = if i + 1 < total_blocks {
                    (i + 1) as u32
                } else {
                    LIST_TAIL
                };
                AtomicU32::new(link)
            })
            .collect();
        let state: Box<[AtomicU8]> = (0..total_blocks).map(|_| AtomicU8::new(FREE)).collect();

        Ok(Backing {
            base,
            layout,
            next,
            state,
        })
    }

    /// Return the initialized [`Backing`], initializing it and populating the
    /// largest size-class free list atomically on first access.
    fn backing(&self) -> Result<&Backing> {
        if let Some(backing) = self.backing.get() {
            return Ok(backing);
        }

        let _guard = self
            .init
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Init mutex poisoned: {}", e)))?;
        if let Some(backing) = self.backing.get() {
            return Ok(backing);
        }

        let backing = self.build_backing()?;
        let largest_class = self.size_classes.len() - 1;
        // Publish free-list count and head BEFORE `OnceLock::set` so any thread
        // whose `OnceLock::get` Acquire-observes `Some(&Backing)` also observes
        // the fully populated free list.
        self.counts[largest_class].store(self.config.total_blocks as u32, Ordering::Release);
        self.heads[largest_class].store(0, Ordering::Release);
        let _ = self.backing.set(backing);

        self.backing
            .get()
            .ok_or_else(|| ZiporaError::invalid_data("Backing initialization failed"))
    }

    fn try_pop_class(&self, backing: &Backing, class: usize) -> Option<NonNull<u8>> {
        let head = &self.heads[class];
        let mut cur = head.load(Ordering::Acquire);
        loop {
            let idx = cur as u32;
            if idx == LIST_TAIL {
                return None;
            }

            let next = backing.next[idx as usize].load(Ordering::Relaxed);
            let next_tag = (cur >> 32).wrapping_add(1);
            let new_head = (next_tag << 32) | u64::from(next);

            match head.compare_exchange_weak(cur, new_head, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    self.counts[class].fetch_sub(1, Ordering::Relaxed);
                    let prev_state = backing.state[idx as usize].swap(LIVE, Ordering::AcqRel);
                    debug_assert_eq!(prev_state, FREE);
                    let byte_offset = (idx as usize) * self.config.max_block_size;
                    // SAFETY: `idx < config.total_blocks`, so `byte_offset < backing.layout.size()`,
                    // and `backing.base` is a live non-null allocation of `backing.layout`.
                    let ptr =
                        unsafe { NonNull::new_unchecked(backing.base.as_ptr().add(byte_offset)) };
                    return Some(ptr);
                }
                Err(actual) => {
                    cur = actual;
                }
            }
        }
    }

    /// Pop a block from `size_class_index`, or from the smallest larger class
    /// that has one.
    ///
    /// Returns the block together with the class it actually came from so
    /// `deallocate` returns it to the same class at the same width (C3.7).
    fn allocate_from_free_list(
        &self,
        backing: &Backing,
        size_class_index: usize,
    ) -> Result<(NonNull<u8>, usize)> {
        for class in size_class_index..self.size_classes.len() {
            if let Some(ptr) = self.try_pop_class(backing, class) {
                return Ok((ptr, class));
            }
        }

        if let Some(stats) = &self.stats {
            stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
        }

        Err(ZiporaError::out_of_memory(
            self.size_classes[size_class_index],
        ))
    }

    fn find_size_class(&self, size: usize) -> Result<usize> {
        for (index, &class_size) in self.size_classes.iter().enumerate() {
            if size <= class_size {
                return Ok(index);
            }
        }
        Err(ZiporaError::invalid_data("Size too large"))
    }

    fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        if size_class_index >= self.size_classes.len() {
            return Err(ZiporaError::invalid_data("Invalid size class index"));
        }

        let backing = self
            .backing
            .get()
            .ok_or_else(|| ZiporaError::invalid_data("Memory not allocated"))?;

        let base = backing.base.as_ptr().addr();
        let addr = ptr.as_ptr().addr();
        let Some(byte_offset) = addr.checked_sub(base) else {
            return Err(ZiporaError::invalid_data("Pointer outside pool"));
        };
        if byte_offset >= backing.layout.size() {
            return Err(ZiporaError::invalid_data("Pointer outside pool"));
        }
        if !byte_offset.is_multiple_of(self.config.max_block_size) {
            return Err(ZiporaError::invalid_data(
                "Pointer misaligned to block boundary",
            ));
        }
        let idx = byte_offset / self.config.max_block_size;

        // Atomically claim the transition from LIVE to FREE BEFORE touching user
        // memory, free lists, or statistics so double frees fail closed immediately.
        if backing.state[idx]
            .compare_exchange(LIVE, FREE, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(ZiporaError::invalid_data(
                "Double free or invalid block state",
            ));
        }

        if self.config.secure_clear {
            let clear_len = self.size_classes[size_class_index];
            // SAFETY: `idx < config.total_blocks` and `clear_len <= config.max_block_size`,
            // so `[byte_offset .. byte_offset + clear_len)` lies within the `Backing` arena,
            // and winning `LIVE -> FREE` above grants exclusive access to block `idx` until
            // the CAS push below publishes `idx` onto `heads[size_class_index]`.
            // Alignment: the arena is aligned to `config.alignment` and blocks are
            // `max_block_size` apart, a multiple of it, so every block starts at a
            // multiple of `config.alignment >= MIN_BLOCK_ALIGN`. The `u64` branch
            // checks 8-byte alignment at run time; the `u32` branch relies on
            // `MIN_BLOCK_ALIGN >= align_of::<u32>()`.
            unsafe {
                let base_ptr = backing.base.as_ptr().add(byte_offset);
                if (base_ptr as usize).is_multiple_of(std::mem::align_of::<u64>()) {
                    let word_size = std::mem::size_of::<u64>();
                    let word_count = clear_len / word_size;
                    let word_ptr = base_ptr as *mut u64;
                    for w in 0..word_count {
                        word_ptr.add(w).write_volatile(0);
                    }
                    for offset in (word_count * word_size)..clear_len {
                        base_ptr.add(offset).write_volatile(0);
                    }
                } else {
                    debug_assert!((base_ptr as usize).is_multiple_of(std::mem::align_of::<u32>()));
                    let word_size = std::mem::size_of::<u32>();
                    let word_count = clear_len / word_size;
                    let word_ptr = base_ptr as *mut u32;
                    for w in 0..word_count {
                        word_ptr.add(w).write_volatile(0);
                    }
                    for offset in (word_count * word_size)..clear_len {
                        base_ptr.add(offset).write_volatile(0);
                    }
                }
            }
            compiler_fence(Ordering::SeqCst);
        }

        let head = &self.heads[size_class_index];
        let mut cur = head.load(Ordering::Acquire);
        loop {
            backing.next[idx].store(cur as u32, Ordering::Relaxed);
            let next_tag = (cur >> 32).wrapping_add(1);
            let new_head = (next_tag << 32) | (idx as u64);
            match head.compare_exchange_weak(cur, new_head, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    self.counts[size_class_index].fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Err(actual) => {
                    cur = actual;
                }
            }
        }

        if let Some(stats) = &self.stats {
            stats.deallocations.fetch_add(1, Ordering::Relaxed);
            let prev = stats.active_blocks.fetch_sub(1, Ordering::Relaxed);
            let active = prev.saturating_sub(1);
            let utilization = (active.saturating_mul(10000) / self.config.total_blocks) as u32;
            stats.utilization.store(utilization, Ordering::Relaxed);
        }

        Ok(())
    }
}

/// Fixed capacity memory pool implementation
pub struct FixedCapacityMemoryPool {
    inner: Arc<Inner>,
}

impl FixedCapacityMemoryPool {
    /// Reject a configuration the block layout or `u32` free-list links cannot represent.
    fn validate_config(config: &FixedCapacityPoolConfig) -> Result<()> {
        if config.total_blocks == 0 {
            return Err(ZiporaError::invalid_data(
                "FixedCapacityPoolConfig::total_blocks must be non-zero",
            ));
        }
        if config.total_blocks >= LIST_TAIL as usize {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::total_blocks ({}) must be less than u32::MAX ({})",
                config.total_blocks, LIST_TAIL
            )));
        }
        if !config.alignment.is_power_of_two() {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::alignment ({}) must be a power of two",
                config.alignment
            )));
        }
        if config.alignment < MIN_BLOCK_ALIGN {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::alignment ({}) must be at least {} (minimum block alignment)",
                config.alignment, MIN_BLOCK_ALIGN
            )));
        }
        if config.max_block_size < MIN_BLOCK_SIZE {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::max_block_size ({}) must be at least {} (minimum block size)",
                config.max_block_size, MIN_BLOCK_SIZE
            )));
        }
        if !config.max_block_size.is_multiple_of(config.alignment) {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::max_block_size ({}) must be a multiple of alignment ({}); \
                 blocks are carved at multiples of max_block_size and would not meet the requested alignment",
                config.max_block_size, config.alignment
            )));
        }
        let total_size = config
            .total_blocks
            .checked_mul(config.max_block_size)
            .ok_or_else(|| {
                ZiporaError::invalid_data(format!(
                    "FixedCapacityPoolConfig capacity overflows usize: {} blocks x {} bytes",
                    config.total_blocks, config.max_block_size
                ))
            })?;
        Layout::from_size_align(total_size, config.alignment).map_err(|e| {
            ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig invalid layout ({} bytes, align {}): {}",
                total_size, config.alignment, e
            ))
        })?;
        Ok(())
    }

    /// Create a new fixed capacity memory pool
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if the configuration cannot be represented by the
    /// block layout: zero blocks, `total_blocks >= u32::MAX`, a non-power-of-two
    /// alignment, an alignment below 4 bytes, a `max_block_size` smaller than
    /// 16 bytes or not a multiple of `alignment`, or a total capacity that
    /// overflows `Layout`.
    pub fn new(config: FixedCapacityPoolConfig) -> Result<Self> {
        Self::validate_config(&config)?;

        let size_classes = Self::generate_size_classes(config.max_block_size, config.alignment);
        let num_classes = size_classes.len();

        let heads: Box<[AtomicU64]> = (0..num_classes)
            .map(|_| AtomicU64::new(u64::from(LIST_TAIL)))
            .collect();
        let counts: Box<[AtomicU32]> = (0..num_classes).map(|_| AtomicU32::new(0)).collect();

        let stats = if config.enable_stats {
            Some(Arc::new(FixedCapacityPoolStats::default()))
        } else {
            None
        };

        let eager = config.eager_allocation;
        let inner = Arc::new(Inner {
            config,
            size_classes,
            stats,
            heads,
            counts,
            backing: OnceLock::new(),
            init: Mutex::new(()),
        });

        if eager {
            inner.backing()?;
        }

        Ok(Self { inner })
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Result<FixedCapacityAllocation> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("Cannot allocate zero bytes"));
        }

        if size > self.inner.config.max_block_size {
            return Err(ZiporaError::invalid_data(format!(
                "Allocation size {} exceeds maximum {}",
                size, self.inner.config.max_block_size
            )));
        }

        let backing = self.inner.backing()?;

        // Find the smallest class that could serve the request, then take
        // whatever class actually had a block. `actual_class_index` is the
        // class the block must be filed back under; using the *requested*
        // class instead is what permanently demoted blocks (C3.7).
        let size_class_index = self.inner.find_size_class(size)?;
        let (ptr, actual_class_index) = self
            .inner
            .allocate_from_free_list(backing, size_class_index)?;
        let actual_size = self.inner.size_classes[actual_class_index];

        if let Some(stats) = &self.inner.stats {
            stats.allocations.fetch_add(1, Ordering::Relaxed);
            let active = stats.active_blocks.fetch_add(1, Ordering::Relaxed) + 1;

            loop {
                let current_peak = stats.peak_blocks.load(Ordering::Relaxed);
                if active <= current_peak
                    || stats
                        .peak_blocks
                        .compare_exchange_weak(
                            current_peak,
                            active,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                {
                    break;
                }
            }

            let utilization =
                (active.saturating_mul(10000) / self.inner.config.total_blocks) as u32;
            stats.utilization.store(utilization, Ordering::Relaxed);
        }

        Ok(FixedCapacityAllocation::new(
            ptr,
            actual_size,
            actual_class_index,
            Arc::clone(&self.inner),
        ))
    }

    /// Deallocate memory back to the pool (raw test entry point paired with `mem::forget(alloc)`).
    #[cfg(test)]
    fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        self.inner.deallocate(ptr, size_class_index)?;
        assert!(
            Arc::strong_count(&self.inner) >= 2,
            "raw deallocate requires a forgotten FixedCapacityAllocation lease"
        );
        // SAFETY: `self.inner.deallocate` succeeded on the unique `LIVE -> FREE`
        // transition for `ptr`, and `&self` holds its own `Arc<Inner>` (`strong_count >= 2`),
        // so decrementing one strong count reclaims the forgotten `FixedCapacityAllocation`'s
        // `Arc<Inner>` reference without dropping `Inner`.
        unsafe {
            Arc::decrement_strong_count(Arc::as_ptr(&self.inner));
        }
        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> Option<Arc<FixedCapacityPoolStats>> {
        self.inner.stats.clone()
    }

    /// Get total capacity in bytes
    pub fn total_capacity(&self) -> usize {
        self.inner.config.total_blocks * self.inner.config.max_block_size
    }

    /// Get available capacity in bytes
    pub fn available_capacity(&self) -> usize {
        if let Some(stats) = &self.inner.stats {
            let used_blocks = stats.active_blocks.load(Ordering::Relaxed);
            let available_blocks = self.inner.config.total_blocks.saturating_sub(used_blocks);
            available_blocks * self.inner.config.max_block_size
        } else {
            0 // Can't determine without stats
        }
    }

    /// Check if pool has capacity for allocation
    pub fn has_capacity(&self, size: usize) -> bool {
        if size > self.inner.config.max_block_size {
            return false;
        }

        if let Some(stats) = &self.inner.stats {
            !stats.is_at_capacity(self.inner.config.total_blocks)
        } else {
            true // Assume capacity without stats
        }
    }

    /// Generate size classes based on maximum size and alignment
    fn generate_size_classes(max_size: usize, alignment: usize) -> Vec<usize> {
        let mut classes = Vec::new();
        let mut current_size = alignment;

        while current_size <= max_size {
            classes.push(current_size);

            // Use fibonacci-like growth for size classes
            let grown = if current_size < 128 {
                current_size.checked_add(alignment)
            } else if current_size < 1024 {
                current_size.checked_mul(3).map(|n| n / 2)
            } else {
                current_size.checked_mul(2)
            };

            let Some(next) = grown.and_then(|n| n.checked_add(alignment - 1)) else {
                break;
            };
            let aligned = next & !(alignment - 1);
            if aligned <= current_size {
                break;
            }
            current_size = aligned;
        }

        // Ensure max size is included
        if classes.is_empty() || classes[classes.len() - 1] != max_size {
            classes.push(max_size);
        }

        classes
    }
}

/// RAII wrapper for fixed capacity allocations
pub struct FixedCapacityAllocation {
    ptr: NonNull<u8>,
    size: usize,
    size_class_index: usize,
    pool: Arc<Inner>,
}

impl FixedCapacityAllocation {
    /// Create new allocation wrapper
    fn new(ptr: NonNull<u8>, size: usize, size_class_index: usize, pool: Arc<Inner>) -> Self {
        Self {
            ptr,
            size,
            size_class_index,
            pool,
        }
    }

    /// Get pointer to allocated memory.
    ///
    /// Bytes written through this pointer within `0..self.size()` must be
    /// initialized: [`Self::as_slice`] and [`Self::as_mut_slice`] expose them as
    /// `u8`, on this allocation and, unless `secure_clear` is set, on later
    /// allocations that reuse the block.
    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size of allocation
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get mutable slice view
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `self.ptr` points to a block exclusively owned by `self`, `self.pool`
        // keeps the `Backing` arena alive for the lifetime of `self`, and the arena was
        // initially zeroed by `alloc_zeroed` and only mutated via initialized writes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get immutable slice view
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `self.ptr` points to a block owned by `self`, `self.pool` keeps the
        // `Backing` arena alive for the lifetime of `self`, and the arena was initially
        // zeroed by `alloc_zeroed` and only mutated via initialized writes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for FixedCapacityAllocation {
    fn drop(&mut self) {
        if let Err(e) = self.pool.deallocate(self.ptr, self.size_class_index) {
            log::error!("Failed to deallocate fixed capacity memory: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `FixedCapacityMemoryPool` is not `Debug`, so `expect_err` is unavailable.
    fn expect_config_rejected(config: FixedCapacityPoolConfig, why: &str) -> String {
        match FixedCapacityMemoryPool::new(config) {
            Ok(_) => panic!("{why}"),
            Err(e) => e.to_string(),
        }
    }

    /// C3.1 (CRITICAL, Miri-confirmed). `initialize_free_lists` writes a
    /// 16-byte `BlockHeader` at the start of every block. With blocks narrower
    /// than the header the last write runs past the end of the arena — inside
    /// `new()`, from entirely safe code. Miri at the parent commit:
    /// "Undefined Behavior: constructing invalid value of type
    /// &mut BlockHeader: encountered a dangling reference (going beyond the
    /// bounds of its allocation)" at `fixed_capacity_pool.rs:475`, reached from
    /// `FixedCapacityMemoryPool::new`.
    #[test]
    fn test_config_rejects_block_narrower_than_block_header() {
        for max_block_size in [4usize, 8, 12] {
            let config = FixedCapacityPoolConfig {
                max_block_size,
                total_blocks: 2,
                alignment: 4,
                enable_stats: false,
                eager_allocation: true,
                secure_clear: false,
            };
            let err = expect_config_rejected(
                config,
                "max_block_size below the minimum block size must be rejected",
            );
            assert!(
                err.contains("minimum block size"),
                "unexpected error for max_block_size {max_block_size}: {err}"
            );
        }

        // Exactly the header size is representable and must still be accepted.
        let config = FixedCapacityPoolConfig {
            max_block_size: MIN_BLOCK_SIZE,
            total_blocks: 2,
            alignment: MIN_BLOCK_ALIGN,
            enable_stats: false,
            eager_allocation: true,
            secure_clear: false,
        };
        assert!(FixedCapacityMemoryPool::new(config).is_ok());
    }

    /// C3.1. `total_blocks == 0` makes the arena zero-sized; `std::alloc::alloc`
    /// with a zero-sized layout is undefined behaviour, and the statistics path
    /// divides by `total_blocks`.
    #[test]
    fn test_config_rejects_zero_total_blocks() {
        let config = FixedCapacityPoolConfig {
            total_blocks: 0,
            ..FixedCapacityPoolConfig::default()
        };
        let err = expect_config_rejected(config, "total_blocks == 0 must be rejected");
        assert!(err.contains("total_blocks"), "got: {err}");
    }

    /// C3.1. Blocks are carved at `i * max_block_size`, so the arena's
    /// alignment only reaches every block when `max_block_size` is a multiple
    /// of `alignment`. Otherwise the pool hands out pointers that violate the
    /// alignment it was configured with.
    #[test]
    fn test_config_rejects_block_size_not_multiple_of_alignment() {
        let config = FixedCapacityPoolConfig {
            max_block_size: 100,
            total_blocks: 4,
            alignment: 64,
            enable_stats: false,
            eager_allocation: true,
            secure_clear: false,
        };
        let err = expect_config_rejected(
            config,
            "max_block_size 100 with alignment 64 must be rejected",
        );
        assert!(err.contains("multiple of alignment"), "got: {err}");
    }

    /// C3.1. A non-power-of-two alignment is not a valid `Layout` alignment and
    /// makes the `& !(alignment - 1)` rounding in `generate_size_classes`
    /// nonsense; an alignment below the header's would misalign the header.
    #[test]
    fn test_config_rejects_bad_alignment() {
        let err = expect_config_rejected(
            FixedCapacityPoolConfig {
                alignment: 24,
                ..FixedCapacityPoolConfig::default()
            },
            "non-power-of-two alignment must be rejected",
        );
        assert!(err.contains("power of two"), "got: {err}");

        let err = expect_config_rejected(
            FixedCapacityPoolConfig {
                alignment: 2,
                ..FixedCapacityPoolConfig::default()
            },
            "alignment below the minimum block alignment must be rejected",
        );
        assert!(err.contains("minimum block alignment"), "got: {err}");
    }

    /// C3.1. Every preset must survive its own validation.
    #[test]
    fn test_all_presets_pass_validation() {
        for (name, config) in [
            ("default", FixedCapacityPoolConfig::default()),
            ("small_objects", FixedCapacityPoolConfig::small_objects()),
            ("medium_objects", FixedCapacityPoolConfig::medium_objects()),
            ("realtime", FixedCapacityPoolConfig::realtime()),
            ("secure", FixedCapacityPoolConfig::secure()),
        ] {
            FixedCapacityMemoryPool::validate_config(&config)
                .unwrap_or_else(|e| panic!("preset {name} failed validation: {e}"));
        }
    }

    /// C3.7. `initialize_free_lists` files every block on the *largest* size
    /// class and nothing ever moves one down. A small request therefore finds
    /// `LIST_TAIL` in its own class, falls into `allocate_by_splitting`, which
    /// does not split -- "For simplicity, just return the larger block" -- and
    /// hands over a full `max_block_size` block. The allocation then records
    /// the *requested* class, so `deallocate_to_free_list` files that block
    /// under the small class. The block is still physically `max_block_size`
    /// wide and still `max_block_size` away from its neighbours, but the pool
    /// has permanently reclassified it. The carve width and the recycle width
    /// disagree, so a later full-width request can never find it again.
    #[test]
    fn test_small_allocations_do_not_permanently_demote_blocks() {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 4096,
            total_blocks: 4,
            ..FixedCapacityPoolConfig::default()
        })
        .unwrap();

        // Hold all four at once so each one is carved out of the largest class.
        let small: Vec<_> = (0..4)
            .map(|i| {
                pool.allocate(8)
                    .unwrap_or_else(|e| panic!("small allocation {i} failed: {e}"))
            })
            .collect();
        drop(small);

        assert_eq!(
            pool.available_capacity(),
            4 * 4096,
            "every block was returned"
        );
        let big = pool.allocate(4096).unwrap_or_else(|e| {
            panic!(
                "all four 4096-byte blocks were returned and available_capacity() \
                 reports {} bytes free, but a full-width request failed: {e}",
                pool.available_capacity()
            )
        });
        drop(big);
    }

    /// C3.7. The width the caller is told about has to be the width that was
    /// taken out of circulation. Blocks sit at a `max_block_size` stride, so an
    /// eight-byte request still consumes a whole block; reporting the size
    /// class of the *request* understates the reservation and is precisely what
    /// let `deallocate` refile the block under the wrong class.
    #[test]
    fn test_allocation_reports_the_width_it_actually_reserved() {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 4096,
            total_blocks: 2,
            ..FixedCapacityPoolConfig::default()
        })
        .unwrap();

        let alloc = pool.allocate(8).unwrap();
        assert_eq!(
            alloc.size(),
            4096,
            "an 8-byte request reserves a whole 4096-byte block"
        );
    }

    /// C3.7. A pool that is drained and refilled with a *changing* width must
    /// not erode. Alternating full-width and minimum-width rounds demotes every
    /// block to the smallest class on the odd round, after which the even round
    /// cannot find a single full-width block. A steady demand pattern hides
    /// this, because the demoted classes happen to match the next round's
    /// requests.
    #[test]
    fn test_alternating_width_cycles_do_not_erode_capacity() {
        const BLOCKS: usize = 8;

        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 4096,
            total_blocks: BLOCKS,
            ..FixedCapacityPoolConfig::default()
        })
        .unwrap();

        for round in 0..8 {
            let width = if round % 2 == 0 { 4096 } else { 8 };
            let held: Vec<_> = (0..BLOCKS)
                .map(|i| {
                    pool.allocate(width).unwrap_or_else(|e| {
                        panic!(
                            "round {round} ({width} bytes), allocation {i}: the pool \
                             was fully drained and refilled on every previous round, \
                             so all {BLOCKS} blocks are free: {e}"
                        )
                    })
                })
                .collect();
            drop(held);
        }
    }

    #[test]
    fn test_pool_creation() {
        let config = FixedCapacityPoolConfig::default();
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        assert!(pool.stats().is_some());
        assert_eq!(pool.total_capacity(), 4096 * 1000);
    }

    #[test]
    fn test_basic_allocation() {
        let config = FixedCapacityPoolConfig::small_objects();
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        let alloc = pool.allocate(64).unwrap();
        // C3.7: `small_objects()` has `max_block_size: 1024` and every block
        // starts on the largest class, so a 64-byte request reserves a whole
        // 1024-byte block. This used to report 64 -- the size class of the
        // *request* -- which is what let `deallocate` refile the block under
        // the 64-byte class and lose it for good.
        assert_eq!(alloc.size(), 1024);
        assert!(!alloc.as_ptr().is_null());
    }

    #[test]
    fn test_capacity_limits() {
        let config = FixedCapacityPoolConfig {
            max_block_size: 128,
            total_blocks: 10,
            ..FixedCapacityPoolConfig::default()
        };
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        // Allocate until capacity is reached
        let mut allocations = Vec::new();

        for i in 0..15 {
            // Try more than capacity
            match pool.allocate(64) {
                Ok(alloc) => allocations.push(alloc),
                Err(_) => {
                    assert!(i >= 10, "Should reach capacity around block 10");
                    break;
                }
            }
        }

        // Should have allocated exactly the capacity
        assert!(allocations.len() <= 10);

        // Check statistics
        if let Some(stats) = pool.stats() {
            assert!(stats.allocation_failures.load(Ordering::Relaxed) > 0);
        }
    }

    #[test]
    fn test_size_classes() {
        let classes = FixedCapacityMemoryPool::generate_size_classes(1024, 8);

        // Should include multiple size classes
        assert!(classes.len() > 1);

        // Should be sorted
        for i in 1..classes.len() {
            assert!(classes[i] > classes[i - 1]);
        }

        // Should end with max size
        assert_eq!(classes[classes.len() - 1], 1024);
    }

    #[test]
    fn test_different_configurations() {
        // Test realtime config
        let rt_config = FixedCapacityPoolConfig::realtime();
        let rt_pool = FixedCapacityMemoryPool::new(rt_config).unwrap();
        assert!(rt_pool.stats().is_none()); // Stats disabled for performance

        // Test secure config
        let secure_config = FixedCapacityPoolConfig::secure();
        let secure_pool = FixedCapacityMemoryPool::new(secure_config).unwrap();
        assert!(secure_pool.inner.config.secure_clear);
    }

    #[test]
    fn test_lazy_allocation() {
        let config = FixedCapacityPoolConfig {
            eager_allocation: false,
            ..FixedCapacityPoolConfig::default()
        };
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        // Memory should not be allocated yet
        assert!(pool.inner.backing.get().is_none());

        // First allocation should trigger memory allocation
        let _alloc = pool.allocate(64).unwrap();
        assert!(pool.inner.backing.get().is_some());
    }

    #[test]
    fn test_allocation_statistics() {
        let config = FixedCapacityPoolConfig::small_objects();
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        // Allocate some blocks
        let mut allocations = Vec::new();
        for _ in 0..5 {
            allocations.push(pool.allocate(32).unwrap());
        }

        // Check statistics
        if let Some(stats) = pool.stats() {
            assert_eq!(stats.allocations.load(Ordering::Relaxed), 5);
            assert_eq!(stats.active_blocks.load(Ordering::Relaxed), 5);
            assert!(stats.utilization_percent() > 0.0);
        }

        // Drop allocations to test deallocation stats
        allocations.clear();

        if let Some(stats) = pool.stats() {
            assert_eq!(stats.deallocations.load(Ordering::Relaxed), 5);
            assert_eq!(stats.active_blocks.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn test_secure_clearing() {
        let config = FixedCapacityPoolConfig::secure();
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        // Allocate and write data
        {
            let mut alloc = pool.allocate(64).unwrap();
            let slice = alloc.as_mut_slice();
            slice.fill(0xAA); // Write pattern
        } // Memory should be cleared on drop

        // Allocate again and check if cleared
        let alloc = pool.allocate(64).unwrap();
        let slice = alloc.as_slice();

        // In secure mode, memory should be zeroed
        // Note: This test assumes the same block is reused
        for &byte in slice.iter().take(64) {
            if byte == 0xAA {
                // If we find the pattern, secure clearing might not be working
                // But this could also be a different block, so we can't assert
                break;
            }
        }
    }

    #[test]
    fn test_deallocate_validation_and_secure_zeroing() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FixedCapacityMemoryPool>();

        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 64,
            total_blocks: 2,
            alignment: 8,
            enable_stats: true,
            eager_allocation: true,
            secure_clear: true,
        })
        .unwrap();

        let mut victim = pool.allocate(64).unwrap();
        victim.as_mut_slice().fill(0xCC);
        let (ptr, class) = (victim.ptr, victim.size_class_index);
        std::mem::forget(victim);

        // Out-of-range size class, outside-pool pointer, and misaligned interior pointer all fail.
        assert!(pool.deallocate(ptr, usize::MAX).is_err());
        let bogus = NonNull::<u8>::dangling();
        assert!(pool.deallocate(bogus, class).is_err());
        let interior = unsafe { NonNull::new_unchecked(ptr.as_ptr().add(8)) };
        assert!(pool.deallocate(interior, class).is_err());

        // First valid free succeeds and zeroes the block; second free fails without underflowing stats.
        pool.deallocate(ptr, class).unwrap();
        assert!(pool.deallocate(ptr, class).is_err());
        let stats = pool.stats().unwrap();
        assert_eq!(stats.active_blocks.load(Ordering::Relaxed), 0);
        assert_eq!(stats.deallocations.load(Ordering::Relaxed), 1);

        // Re-popping the LIFO head block returns zeroed memory.
        let reused = pool.allocate(64).unwrap();
        assert_eq!(reused.as_ptr(), ptr.as_ptr());
        assert!(reused.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_concurrent_lazy_alloc_free_stress() {
        let (blocks, threads, rounds) = if cfg!(miri) {
            (12, 3, 4)
        } else {
            (512, 8, 500)
        };
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 64,
            total_blocks: blocks,
            alignment: 8,
            enable_stats: true,
            eager_allocation: false,
            secure_clear: false,
        })
        .unwrap();
        let barrier = std::sync::Barrier::new(threads);
        std::thread::scope(|s| {
            for t in 0..threads {
                let (pool, barrier) = (&pool, &barrier);
                s.spawn(move || {
                    barrier.wait();
                    for r in 0..rounds {
                        let mut alloc = pool.allocate(64).unwrap();
                        let tag = (t * 17 + r) as u8;
                        alloc.as_mut_slice().fill(tag);
                        assert!(alloc.as_slice().iter().all(|&b| b == tag));
                    }
                });
            }
        });
        let stats = pool.stats().unwrap();
        assert_eq!(stats.active_blocks.load(Ordering::Relaxed), 0);
        assert_eq!(
            stats.allocations.load(Ordering::Relaxed),
            (threads * rounds) as u64
        );
        assert_eq!(
            stats.deallocations.load(Ordering::Relaxed),
            (threads * rounds) as u64
        );
    }
}

/// Reliability-first plan (v2), group G2: tests that need the private
/// `deallocate`. The public-API G2 tests are in `tests/plan_v2_g2.rs`.
#[cfg(test)]
mod plan_v2_fcp {
    use super::*;

    /// S3(b). `deallocate_to_free_list` (699-724) re-stamps the magic and
    /// pushes without checking that the block is live, and
    /// `allocate_from_free_list` (612-661) never clears the magic on pop. A
    /// second free of the same block links it to itself (`next == offset`), so
    /// the pool then hands the same block out twice. Stats are disabled
    /// because at v5.0.0 the second free would otherwise panic first on
    /// `active_blocks.fetch_sub(1) - 1` underflow (412).
    #[test]
    fn test_v2_s3_double_free_is_rejected_and_never_aliases() {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 64,
            total_blocks: 4,
            enable_stats: false,
            ..FixedCapacityPoolConfig::default()
        })
        .unwrap();
        let victim = pool.allocate(64).unwrap();
        let (ptr, class) = (victim.ptr, victim.size_class_index);
        std::mem::forget(victim);

        pool.deallocate(ptr, class)
            .expect("the first free of a live block succeeds");
        let second = pool.deallocate(ptr, class);

        let a = pool.allocate(64).unwrap();
        let b = pool.allocate(64).unwrap();
        assert_ne!(
            a.as_ptr(),
            b.as_ptr(),
            "after a double free the pool handed out the same block twice"
        );
        assert!(
            second.is_err(),
            "the second free of the same block must be rejected"
        );
    }

    #[test]
    fn test_v2_s3_treiber_head_tag_prevents_aba_on_recycled_block() {
        let pool = FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
            max_block_size: 64,
            total_blocks: 3,
            alignment: 8,
            enable_stats: false,
            eager_allocation: true,
            secure_clear: true,
        })
        .unwrap();
        let top_class = pool.inner.size_classes.len() - 1;
        let head = &pool.inner.heads[top_class];

        // Initially head is block 0 with tag 0 (`0 -> 1 -> 2 -> LIST_TAIL`).
        let initial_head = head.load(Ordering::Acquire);
        assert_eq!(initial_head as u32, 0);
        assert_eq!(initial_head >> 32, 0);

        // Pop block 0 (`a`) and block 1 (`b`), then return block 0 (`a`) so head
        // points to block 0 again, but now with `next[0] == 2` instead of `1`.
        let mut a = pool.allocate(64).unwrap();
        a.as_mut_slice().fill(0xAA);
        let b = pool.allocate(64).unwrap();
        drop(a);

        let recycled_head = head.load(Ordering::Acquire);
        assert_eq!(
            recycled_head as u32, 0,
            "block 0 should be back at the Treiber head"
        );
        assert_ne!(
            recycled_head >> 32,
            initial_head >> 32,
            "ABA generation tag must advance across pop + push of block 0"
        );

        // A stale popper that observed `initial_head` (block 0, tag 0) and cached
        // `next = 1` (which is currently live in `b`!) must fail its CAS.
        let stale_new_head = (1u64 << 32) | 1u64;
        assert!(
            head.compare_exchange(
                initial_head,
                stale_new_head,
                Ordering::AcqRel,
                Ordering::Acquire
            )
            .is_err(),
            "stale ABA CAS must fail when block 0 has been popped and re-pushed"
        );

        // And `secure_clear` must have wiped block 0 when `a` was dropped.
        let a_reused = pool.allocate(64).unwrap();
        assert!(a_reused.as_slice().iter().all(|&byte| byte == 0));
        drop(b);
    }
}
