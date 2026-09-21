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
use std::alloc::{Layout, alloc, dealloc};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Alignment for fixed capacity allocations
const ALIGN_SIZE: usize = 8;
/// Magic value for free list termination (uses max value to avoid collision with valid offsets)
const LIST_TAIL: u32 = u32::MAX;
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

/// Free list head for a size class
#[derive(Debug)]
struct FreeListHead {
    /// Head of free list (as offset)
    head: AtomicU32,
    /// Count of free blocks in this size class
    count: AtomicU32,
}

impl FreeListHead {
    fn new() -> Self {
        Self {
            head: AtomicU32::new(LIST_TAIL),
            count: AtomicU32::new(0),
        }
    }
}

/// Block header for tracking allocation info
#[repr(C)]
#[derive(Debug)]
struct BlockHeader {
    /// Size class index
    size_class: u32,
    /// Magic number for corruption detection
    magic: u32,
    /// Next block in free list (when free)
    next: u32,
    /// Padding to maintain alignment
    _padding: u32,
}

const BLOCK_HEADER_MAGIC: u32 = 0xDEADBEEF;

/// Fixed capacity memory pool implementation
pub struct FixedCapacityMemoryPool {
    /// Configuration
    config: FixedCapacityPoolConfig,
    /// Pre-allocated memory region (uses UnsafeCell for lazy initialization)
    memory: UnsafeCell<Option<NonNull<u8>>>,
    /// Layout for memory deallocation (uses UnsafeCell for lazy initialization)
    memory_layout: UnsafeCell<Option<Layout>>,
    /// Free lists for each size class (uses UnsafeCell for lazy initialization)
    free_lists: UnsafeCell<Vec<FreeListHead>>,
    /// Size classes (in bytes)
    size_classes: Vec<usize>,
    /// Statistics (optional)
    stats: Option<Arc<FixedCapacityPoolStats>>,
    /// Mutex for thread safety during initialization
    init_mutex: Mutex<bool>,
}

// SAFETY: FixedCapacityMemoryPool is Send because:
// 1. `config: FixedCapacityPoolConfig` - Config is Clone with no raw pointers.
// 2. `memory: UnsafeCell<Option<NonNull<u8>>>` - Protected by init_mutex.
// 3. `memory_layout: UnsafeCell<Option<Layout>>` - Protected by init_mutex.
// 4. `free_lists: UnsafeCell<Vec<FreeListHead>>` - Contains only atomics.
// 5. `size_classes: Vec<usize>` - Immutable after construction.
// 6. `stats: Option<Arc<...>>` - Arc is Send.
// 7. `init_mutex: Mutex<bool>` - Mutex is Send.
unsafe impl Send for FixedCapacityMemoryPool {}

// SAFETY: FixedCapacityMemoryPool is Sync because:
// 1. Initialization of UnsafeCell fields is protected by `init_mutex`.
// 2. After initialization, `free_lists` contains atomic head/count fields.
// 3. `stats` uses Arc for thread-safe reference counting.
// 4. `size_classes` is immutable after construction.
// 5. The init_mutex ensures only one thread performs initialization.
//
// IMPORTANT: The UnsafeCell fields must only be accessed:
// - During initialization (under init_mutex lock)
// - After initialization for atomic operations on free_lists
unsafe impl Sync for FixedCapacityMemoryPool {}

impl FixedCapacityMemoryPool {
    /// Reject a configuration the block layout cannot represent.
    ///
    /// Every block carries a [`BlockHeader`] written *at the block's own
    /// address* while the block sits on a free list, so a block narrower than
    /// the header would write past its own extent — and, for the last block,
    /// past the end of the arena. `initialize_free_lists` does this for every
    /// block during construction, so the overrun happens inside `new()` before
    /// the caller ever sees a pointer.
    ///
    /// Blocks are carved at `i * max_block_size`, so the arena alignment only
    /// propagates to every block when `max_block_size` is a multiple of
    /// `alignment`; otherwise the pool silently breaks the alignment it
    /// promises.
    fn validate_config(config: &FixedCapacityPoolConfig) -> Result<()> {
        if config.total_blocks == 0 {
            return Err(ZiporaError::invalid_data(
                "FixedCapacityPoolConfig::total_blocks must be non-zero",
            ));
        }
        if !config.alignment.is_power_of_two() {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::alignment ({}) must be a power of two",
                config.alignment
            )));
        }
        if config.alignment < align_of::<BlockHeader>() {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::alignment ({}) must be at least {} (block header alignment)",
                config.alignment,
                align_of::<BlockHeader>()
            )));
        }
        if config.max_block_size < size_of::<BlockHeader>() {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::max_block_size ({}) must be at least {} (block header size)",
                config.max_block_size,
                size_of::<BlockHeader>()
            )));
        }
        if !config.max_block_size.is_multiple_of(config.alignment) {
            return Err(ZiporaError::invalid_data(format!(
                "FixedCapacityPoolConfig::max_block_size ({}) must be a multiple of alignment ({}); \
                 blocks are carved at multiples of max_block_size and would not meet the requested alignment",
                config.max_block_size, config.alignment
            )));
        }
        config
            .total_blocks
            .checked_mul(config.max_block_size)
            .ok_or_else(|| {
                ZiporaError::invalid_data(format!(
                    "FixedCapacityPoolConfig capacity overflows usize: {} blocks x {} bytes",
                    config.total_blocks, config.max_block_size
                ))
            })?;
        Ok(())
    }

    /// Create a new fixed capacity memory pool
    ///
    /// # Errors
    ///
    /// Returns `invalid_data` if the configuration cannot be represented by the
    /// block layout: zero blocks, a non-power-of-two alignment, an alignment
    /// below the block header's, a `max_block_size` smaller than the block
    /// header or not a multiple of `alignment`, or a total capacity that
    /// overflows `usize`. See [`Self::validate_config`].
    pub fn new(config: FixedCapacityPoolConfig) -> Result<Self> {
        Self::validate_config(&config)?;

        // Generate size classes
        let size_classes = Self::generate_size_classes(config.max_block_size, config.alignment);
        let num_classes = size_classes.len();

        // Initialize free lists
        let mut free_lists = Vec::with_capacity(num_classes);
        for _ in 0..num_classes {
            free_lists.push(FreeListHead::new());
        }

        // Initialize statistics
        let stats = if config.enable_stats {
            Some(Arc::new(FixedCapacityPoolStats::default()))
        } else {
            None
        };

        let mut pool = Self {
            config,
            memory: UnsafeCell::new(None),
            memory_layout: UnsafeCell::new(None),
            free_lists: UnsafeCell::new(free_lists),
            size_classes,
            stats,
            init_mutex: Mutex::new(false),
        };

        // Pre-allocate memory if requested
        if pool.config.eager_allocation {
            pool.allocate_backing_memory()?;
        }

        Ok(pool)
    }

    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Result<FixedCapacityAllocation> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("Cannot allocate zero bytes"));
        }

        if size > self.config.max_block_size {
            return Err(ZiporaError::invalid_data(format!(
                "Allocation size {} exceeds maximum {}",
                size, self.config.max_block_size
            )));
        }

        // Ensure memory is allocated
        self.ensure_memory_allocated()?;

        // Find the smallest class that could serve the request, then take
        // whatever class actually had a block. `actual_class_index` is the
        // class the block must be filed back under; using the *requested*
        // class instead is what permanently demoted blocks (C3.7).
        let size_class_index = self.find_size_class(size)?;
        let (ptr, actual_class_index) = self.allocate_from_free_list(size_class_index)?;
        let actual_size = self.size_classes[actual_class_index];

        // Update statistics
        if let Some(stats) = &self.stats {
            stats.allocations.fetch_add(1, Ordering::Relaxed);
            let active = stats.active_blocks.fetch_add(1, Ordering::Relaxed) + 1;

            // Update peak
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

            // Update utilization
            let utilization = (active * 10000 / self.config.total_blocks) as u32;
            stats.utilization.store(utilization, Ordering::Relaxed);
        }

        Ok(FixedCapacityAllocation::new(
            ptr,
            actual_size,
            actual_class_index,
            self,
        ))
    }

    /// Deallocate memory back to the pool
    fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        // Verify the pointer is valid
        self.verify_pointer(ptr)?;

        // Clear memory if secure mode is enabled
        if self.config.secure_clear {
            let size = self.size_classes[size_class_index];
            // SAFETY: pointer valid and size matches allocation, verified by verify_pointer above
            unsafe {
                std::ptr::write_bytes(ptr.as_ptr(), 0, size);
            }
        }

        // Return to free list
        self.deallocate_to_free_list(ptr, size_class_index)?;

        // Update statistics
        if let Some(stats) = &self.stats {
            stats.deallocations.fetch_add(1, Ordering::Relaxed);
            let active = stats.active_blocks.fetch_sub(1, Ordering::Relaxed) - 1;

            // Update utilization
            let utilization = (active * 10000 / self.config.total_blocks) as u32;
            stats.utilization.store(utilization, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> Option<Arc<FixedCapacityPoolStats>> {
        self.stats.clone()
    }

    /// Get total capacity in bytes
    pub fn total_capacity(&self) -> usize {
        self.config.total_blocks * self.config.max_block_size
    }

    /// Get available capacity in bytes
    pub fn available_capacity(&self) -> usize {
        if let Some(stats) = &self.stats {
            let used_blocks = stats.active_blocks.load(Ordering::Relaxed);
            let available_blocks = self.config.total_blocks.saturating_sub(used_blocks);
            available_blocks * self.config.max_block_size
        } else {
            0 // Can't determine without stats
        }
    }

    /// Check if pool has capacity for allocation
    pub fn has_capacity(&self, size: usize) -> bool {
        if size > self.config.max_block_size {
            return false;
        }

        if let Some(stats) = &self.stats {
            !stats.is_at_capacity(self.config.total_blocks)
        } else {
            true // Assume capacity without stats
        }
    }

    /// Allocate backing memory region (for mutable access)
    fn allocate_backing_memory(&mut self) -> Result<()> {
        let total_size = self.config.total_blocks * self.config.max_block_size;
        let layout = Layout::from_size_align(total_size, self.config.alignment)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {}", e)))?;

        // SAFETY: layout valid (size > 0, align power of 2), ptr null-checked below
        let memory = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(total_size))?;

        // SAFETY: UnsafeCell access protected by &mut self exclusive borrow
        unsafe {
            *self.memory.get() = Some(memory);
            *self.memory_layout.get() = Some(layout);
        }

        // Initialize free lists with all blocks
        self.initialize_free_lists()?;

        Ok(())
    }

    /// Allocate backing memory region (for shared/const access via UnsafeCell)
    fn allocate_backing_memory_internal(&self) -> Result<()> {
        let total_size = self.config.total_blocks * self.config.max_block_size;
        let layout = Layout::from_size_align(total_size, self.config.alignment)
            .map_err(|e| ZiporaError::invalid_data(format!("Invalid layout: {}", e)))?;

        // SAFETY: layout valid (size > 0, align power of 2), ptr null-checked below
        let memory = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(total_size))?;

        // SAFETY: UnsafeCell access protected by init_mutex in ensure_memory_allocated
        unsafe {
            *self.memory.get() = Some(memory);
            *self.memory_layout.get() = Some(layout);
        }

        // Initialize free lists with all blocks
        self.initialize_free_lists_internal()?;

        Ok(())
    }

    /// Ensure memory is allocated (lazy allocation)
    fn ensure_memory_allocated(&self) -> Result<()> {
        // SAFETY: reading Option discriminant is safe even with concurrent writes due to atomic init_mutex
        unsafe {
            if (*self.memory.get()).is_some() {
                return Ok(());
            }
        }

        // Use mutex to prevent race conditions during initialization
        let mut initialized = self
            .init_mutex
            .lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Init mutex poisoned: {}", e)))?;
        if !*initialized {
            // Initialize memory using UnsafeCell for interior mutability
            self.allocate_backing_memory_internal()?;
            *initialized = true;
        }

        Ok(())
    }

    /// Initialize free lists with all available blocks (for mutable access)
    fn initialize_free_lists(&mut self) -> Result<()> {
        // SAFETY: UnsafeCell access protected by &mut self exclusive borrow
        let memory = unsafe {
            (*self.memory.get()).ok_or_else(|| ZiporaError::invalid_data("Memory not allocated"))?
        };

        let block_size = self.config.max_block_size;

        // Initialize all blocks as free in the largest size class
        let largest_class = self.size_classes.len() - 1;
        // SAFETY: UnsafeCell access protected by &mut self exclusive borrow
        let free_lists = unsafe { &mut *self.free_lists.get() };
        let free_list = &free_lists[largest_class];

        for i in 0..self.config.total_blocks {
            let offset = i * block_size;
            // SAFETY: pointer valid from alloc, offset < total_capacity checked by loop bounds
            let block_ptr = unsafe { memory.as_ptr().add(offset) };

            // SAFETY: block_ptr valid, aligned, and within allocated memory region
            let header = unsafe { &mut *(block_ptr as *mut BlockHeader) };
            header.size_class = largest_class as u32;
            header.magic = BLOCK_HEADER_MAGIC;

            if i < self.config.total_blocks - 1 {
                header.next = ((i + 1) * block_size) as u32;
            } else {
                header.next = LIST_TAIL;
            }
        }

        // Set up free list head
        free_list.head.store(0, Ordering::Relaxed);
        free_list
            .count
            .store(self.config.total_blocks as u32, Ordering::Relaxed);

        Ok(())
    }

    /// Initialize free lists with all available blocks (for shared access via UnsafeCell)
    fn initialize_free_lists_internal(&self) -> Result<()> {
        // SAFETY: UnsafeCell access protected by init_mutex in allocate_backing_memory_internal
        let memory = unsafe {
            (*self.memory.get()).ok_or_else(|| ZiporaError::invalid_data("Memory not allocated"))?
        };

        let block_size = self.config.max_block_size;

        // Initialize all blocks as free in the largest size class
        let largest_class = self.size_classes.len() - 1;
        // SAFETY: UnsafeCell access protected by init_mutex in allocate_backing_memory_internal
        let free_lists = unsafe { &mut *self.free_lists.get() };
        let free_list = &free_lists[largest_class];

        for i in 0..self.config.total_blocks {
            let offset = i * block_size;
            // SAFETY: pointer valid from alloc, offset < total_capacity checked by loop bounds
            let block_ptr = unsafe { memory.as_ptr().add(offset) };

            // SAFETY: block_ptr valid, aligned, and within allocated memory region
            let header = unsafe { &mut *(block_ptr as *mut BlockHeader) };
            header.size_class = largest_class as u32;
            header.magic = BLOCK_HEADER_MAGIC;

            if i < self.config.total_blocks - 1 {
                header.next = ((i + 1) * block_size) as u32;
            } else {
                header.next = LIST_TAIL;
            }
        }

        // Set up free list head
        free_list.head.store(0, Ordering::Relaxed);
        free_list
            .count
            .store(self.config.total_blocks as u32, Ordering::Relaxed);

        Ok(())
    }

    /// Pop a block from `size_class_index`, or from the smallest larger class
    /// that has one.
    ///
    /// Returns the block together with the class it actually came from. The
    /// caller must file it back under *that* class: the blocks of this pool sit
    /// at a fixed `max_block_size` stride and are never physically split, so a
    /// block taken from a larger class is still a larger-class block.
    fn allocate_from_free_list(&self, size_class_index: usize) -> Result<(NonNull<u8>, usize)> {
        // SAFETY: UnsafeCell immutable after initialization, only atomic fields accessed
        let free_lists = unsafe { &*self.free_lists.get() };
        let free_list = &free_lists[size_class_index];

        // Try to pop from free list
        loop {
            let current_head = free_list.head.load(Ordering::Acquire);

            if current_head == LIST_TAIL {
                return self.allocate_from_larger_class(size_class_index);
            }

            // SAFETY: memory initialized by ensure_memory_allocated before allocate_from_free_list
            let memory = unsafe {
                (*self.memory.get())
                    .ok_or_else(|| ZiporaError::invalid_data("Memory not allocated"))?
            };
            // SAFETY: pointer valid from alloc, current_head offset validated by free list
            let block_ptr = unsafe { memory.as_ptr().add(current_head as usize) };
            // SAFETY: block_ptr valid, aligned to BlockHeader, within allocated region
            let header = unsafe { &*(block_ptr as *const BlockHeader) };

            // Verify header integrity
            if header.magic != BLOCK_HEADER_MAGIC {
                return Err(ZiporaError::invalid_data("Block header corrupted"));
            }

            let next_offset = header.next;

            // Try to update head atomically
            if free_list
                .head
                .compare_exchange_weak(
                    current_head,
                    next_offset,
                    Ordering::Release,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                free_list.count.fetch_sub(1, Ordering::Relaxed);
                let ptr = NonNull::new(block_ptr)
                    .ok_or_else(|| ZiporaError::invalid_data("Null block pointer"))?;
                return Ok((ptr, size_class_index));
            }

            // CAS failed, retry
        }
    }

    /// Serve a request from the smallest larger class that has a free block.
    ///
    /// Despite the name it replaced (`allocate_by_splitting`) nothing is split
    /// here, and nothing can be: `initialize_free_lists` lays every block out
    /// at a `max_block_size` stride, so the size-class vector describes a
    /// partition that does not exist in memory. The block is handed over whole
    /// and the class it came from travels with it, so `deallocate` returns it
    /// to the same class at the same width.
    fn allocate_from_larger_class(
        &self,
        size_class_index: usize,
    ) -> Result<(NonNull<u8>, usize)> {
        for larger_class in (size_class_index + 1)..self.size_classes.len() {
            // SAFETY: UnsafeCell immutable after initialization, only atomic fields accessed
            let free_lists = unsafe { &*self.free_lists.get() };
            let free_list = &free_lists[larger_class];
            let head = free_list.head.load(Ordering::Acquire);

            // The `head` check is not redundant: an empty class would send
            // `allocate_from_free_list` straight back into this function for
            // the classes above `larger_class`, which this loop already walks.
            if head != LIST_TAIL
                && let Ok(found) = self.allocate_from_free_list(larger_class)
            {
                return Ok(found);
            }
        }

        // No available blocks
        if let Some(stats) = &self.stats {
            stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
        }

        Err(ZiporaError::out_of_memory(
            self.size_classes[size_class_index],
        ))
    }

    /// Deallocate to free list
    fn deallocate_to_free_list(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        // SAFETY: UnsafeCell immutable after initialization, only atomic fields accessed
        let free_lists = unsafe { &*self.free_lists.get() };
        let free_list = &free_lists[size_class_index];
        let offset = self.ptr_to_offset(ptr)?;

        // SAFETY: ptr validated by ptr_to_offset above, aligned to BlockHeader
        let header = unsafe { &mut *(ptr.as_ptr() as *mut BlockHeader) };
        header.size_class = size_class_index as u32;
        header.magic = BLOCK_HEADER_MAGIC;

        // Add to free list
        loop {
            let current_head = free_list.head.load(Ordering::Acquire);
            header.next = current_head;

            if free_list
                .head
                .compare_exchange_weak(current_head, offset, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                free_list.count.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
        }
    }

    /// Find appropriate size class for allocation
    fn find_size_class(&self, size: usize) -> Result<usize> {
        for (index, &class_size) in self.size_classes.iter().enumerate() {
            if size <= class_size {
                return Ok(index);
            }
        }
        Err(ZiporaError::invalid_data("Size too large"))
    }

    /// Generate size classes based on maximum size and alignment
    fn generate_size_classes(max_size: usize, alignment: usize) -> Vec<usize> {
        let mut classes = Vec::new();
        let mut current_size = alignment;

        while current_size <= max_size {
            classes.push(current_size);

            // Use fibonacci-like growth for size classes
            if current_size < 128 {
                current_size += alignment;
            } else if current_size < 1024 {
                current_size = (current_size * 3) / 2;
            } else {
                current_size *= 2;
            }

            // Align to boundary
            current_size = (current_size + alignment - 1) & !(alignment - 1);
        }

        // Ensure max size is included
        if classes.is_empty() || classes[classes.len() - 1] != max_size {
            classes.push(max_size);
        }

        classes
    }

    /// Convert pointer to offset
    fn ptr_to_offset(&self, ptr: NonNull<u8>) -> Result<u32> {
        // SAFETY: memory initialized before ptr_to_offset called in allocation/deallocation
        let memory = unsafe {
            (*self.memory.get()).ok_or_else(|| ZiporaError::invalid_data("Memory not allocated"))?
        };

        let base = memory.as_ptr() as usize;
        let addr = ptr.as_ptr() as usize;

        if addr < base || addr >= base + self.total_capacity() {
            return Err(ZiporaError::invalid_data("Pointer outside pool"));
        }

        Ok((addr - base) as u32)
    }

    /// Verify pointer is within pool bounds
    fn verify_pointer(&self, ptr: NonNull<u8>) -> Result<()> {
        self.ptr_to_offset(ptr)?;
        Ok(())
    }
}

impl Drop for FixedCapacityMemoryPool {
    fn drop(&mut self) {
        // SAFETY: &mut self exclusive access, layout matches alloc, ptr from matching alloc
        unsafe {
            if let (Some(memory), Some(layout)) = (*self.memory.get(), *self.memory_layout.get()) {
                dealloc(memory.as_ptr(), layout);
            }
        }
    }
}

/// RAII wrapper for fixed capacity allocations
pub struct FixedCapacityAllocation {
    ptr: NonNull<u8>,
    size: usize,
    size_class_index: usize,
    pool: *const FixedCapacityMemoryPool,
}

impl FixedCapacityAllocation {
    /// Create new allocation wrapper
    fn new(
        ptr: NonNull<u8>,
        size: usize,
        size_class_index: usize,
        pool: &FixedCapacityMemoryPool,
    ) -> Self {
        Self {
            ptr,
            size,
            size_class_index,
            pool: pool as *const _,
        }
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

    /// Get mutable slice view
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: pointer valid from allocation, size matches allocation
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get immutable slice view
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: pointer valid from allocation, size matches allocation
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for FixedCapacityAllocation {
    fn drop(&mut self) {
        // SAFETY: pool pointer valid during allocation lifetime, ptr/size_class from allocation
        unsafe {
            if let Err(e) = (*self.pool).deallocate(self.ptr, self.size_class_index) {
                log::error!("Failed to deallocate fixed capacity memory: {}", e);
            }
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
                "max_block_size below size_of::<BlockHeader>() must be rejected",
            );
            assert!(
                err.contains("block header size"),
                "unexpected error for max_block_size {max_block_size}: {err}"
            );
        }

        // Exactly the header size is representable and must still be accepted.
        let config = FixedCapacityPoolConfig {
            max_block_size: size_of::<BlockHeader>(),
            total_blocks: 2,
            alignment: align_of::<BlockHeader>(),
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
        let err =
            expect_config_rejected(config, "max_block_size 100 with alignment 64 must be rejected");
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
            "alignment below the block header's must be rejected",
        );
        assert!(err.contains("block header alignment"), "got: {err}");
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

        assert!(pool.stats.is_some());
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
        assert!(rt_pool.stats.is_none()); // Stats disabled for performance

        // Test secure config
        let secure_config = FixedCapacityPoolConfig::secure();
        let secure_pool = FixedCapacityMemoryPool::new(secure_config).unwrap();
        assert!(secure_pool.config.secure_clear);
    }

    #[test]
    fn test_lazy_allocation() {
        let config = FixedCapacityPoolConfig {
            eager_allocation: false,
            ..FixedCapacityPoolConfig::default()
        };
        let pool = FixedCapacityMemoryPool::new(config).unwrap();

        // Memory should not be allocated yet
        // SAFETY: test code, single-threaded access to check initialization state
        unsafe {
            assert!((*pool.memory.get()).is_none());
        }

        // First allocation should trigger memory allocation
        let _alloc = pool.allocate(64).unwrap();
        // SAFETY: test code, single-threaded access to check initialization state
        unsafe {
            assert!((*pool.memory.get()).is_some());
        }
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
}
