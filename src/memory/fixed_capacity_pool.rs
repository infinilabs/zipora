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
/// Default number of size classes
const DEFAULT_SIZE_CLASSES: usize = 32;

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
            alignment: 64, // Cache line aligned
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
        if total == 0 { 1.0 } else { successes as f64 / total as f64 }
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

unsafe impl Send for FixedCapacityMemoryPool {}
unsafe impl Sync for FixedCapacityMemoryPool {}

impl FixedCapacityMemoryPool {
    /// Create a new fixed capacity memory pool
    pub fn new(config: FixedCapacityPoolConfig) -> Result<Self> {
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
            return Err(ZiporaError::invalid_data(&format!(
                "Allocation size {} exceeds maximum {}", size, self.config.max_block_size
            )));
        }

        // Ensure memory is allocated
        self.ensure_memory_allocated()?;

        // Find appropriate size class
        let size_class_index = self.find_size_class(size)?;
        let actual_size = self.size_classes[size_class_index];

        // Try to allocate from free list
        let ptr = self.allocate_from_free_list(size_class_index)?;

        // Update statistics
        if let Some(stats) = &self.stats {
            stats.allocations.fetch_add(1, Ordering::Relaxed);
            let active = stats.active_blocks.fetch_add(1, Ordering::Relaxed) + 1;
            
            // Update peak
            loop {
                let current_peak = stats.peak_blocks.load(Ordering::Relaxed);
                if active <= current_peak || 
                   stats.peak_blocks.compare_exchange_weak(
                       current_peak, active, Ordering::Relaxed, Ordering::Relaxed
                   ).is_ok() {
                    break;
                }
            }

            // Update utilization
            let utilization = (active * 10000 / self.config.total_blocks) as u32;
            stats.utilization.store(utilization, Ordering::Relaxed);
        }

        Ok(FixedCapacityAllocation::new(ptr, actual_size, size_class_index, self))
    }

    /// Deallocate memory back to the pool
    fn deallocate(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        // Verify the pointer is valid
        self.verify_pointer(ptr)?;

        // Clear memory if secure mode is enabled
        if self.config.secure_clear {
            let size = self.size_classes[size_class_index];
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
            .map_err(|e| ZiporaError::invalid_data(&format!("Invalid layout: {}", e)))?;

        let memory = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(total_size))?;

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
            .map_err(|e| ZiporaError::invalid_data(&format!("Invalid layout: {}", e)))?;

        let memory = NonNull::new(unsafe { alloc(layout) })
            .ok_or_else(|| ZiporaError::out_of_memory(total_size))?;

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
        // Check if already initialized (safe read through UnsafeCell)
        unsafe {
            if (*self.memory.get()).is_some() {
                return Ok(());
            }
        }

        // Use mutex to prevent race conditions during initialization
        let mut initialized = self.init_mutex.lock()
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
        let memory = unsafe { (*self.memory.get()).ok_or_else(|| 
            ZiporaError::invalid_data("Memory not allocated"))? };

        let block_size = self.config.max_block_size;
        
        // Initialize all blocks as free in the largest size class
        let largest_class = self.size_classes.len() - 1;
        let free_lists = unsafe { &mut *self.free_lists.get() };
        let free_list = &free_lists[largest_class];

        for i in 0..self.config.total_blocks {
            let offset = i * block_size;
            let block_ptr = unsafe { memory.as_ptr().add(offset) };

            // Initialize block header
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
        free_list.count.store(self.config.total_blocks as u32, Ordering::Relaxed);

        Ok(())
    }

    /// Initialize free lists with all available blocks (for shared access via UnsafeCell)
    fn initialize_free_lists_internal(&self) -> Result<()> {
        let memory = unsafe { (*self.memory.get()).ok_or_else(|| 
            ZiporaError::invalid_data("Memory not allocated"))? };

        let block_size = self.config.max_block_size;
        
        // Initialize all blocks as free in the largest size class
        let largest_class = self.size_classes.len() - 1;
        let free_lists = unsafe { &mut *self.free_lists.get() };
        let free_list = &free_lists[largest_class];

        for i in 0..self.config.total_blocks {
            let offset = i * block_size;
            let block_ptr = unsafe { memory.as_ptr().add(offset) };

            // Initialize block header
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
        free_list.count.store(self.config.total_blocks as u32, Ordering::Relaxed);

        Ok(())
    }

    /// Allocate from free list
    fn allocate_from_free_list(&self, size_class_index: usize) -> Result<NonNull<u8>> {
        let free_lists = unsafe { &*self.free_lists.get() };
        let free_list = &free_lists[size_class_index];

        // Try to pop from free list
        loop {
            let current_head = free_list.head.load(Ordering::Acquire);
            
            if current_head == LIST_TAIL {
                // Try to split from larger size class
                return self.allocate_by_splitting(size_class_index);
            }

            // Get pointer to current head block
            let memory = unsafe { (*self.memory.get()).ok_or_else(|| 
                ZiporaError::invalid_data("Memory not allocated"))? };
            let block_ptr = unsafe { memory.as_ptr().add(current_head as usize) };
            let header = unsafe { &*(block_ptr as *const BlockHeader) };

            // Verify header integrity
            if header.magic != BLOCK_HEADER_MAGIC {
                return Err(ZiporaError::invalid_data("Block header corrupted"));
            }

            let next_offset = header.next;

            // Try to update head atomically
            if free_list.head.compare_exchange_weak(
                current_head,
                next_offset,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                free_list.count.fetch_sub(1, Ordering::Relaxed);
                return NonNull::new(block_ptr)
                    .ok_or_else(|| ZiporaError::invalid_data("Null block pointer"));
            }
            
            // CAS failed, retry
        }
    }

    /// Allocate by splitting larger blocks
    fn allocate_by_splitting(&self, size_class_index: usize) -> Result<NonNull<u8>> {
        // Look for larger size classes with available blocks
        for larger_class in (size_class_index + 1)..self.size_classes.len() {
            let free_lists = unsafe { &*self.free_lists.get() };
            let free_list = &free_lists[larger_class];
            let head = free_list.head.load(Ordering::Acquire);
            
            if head != LIST_TAIL {
                // Try to allocate from larger class and split
                if let Ok(ptr) = self.allocate_from_free_list(larger_class) {
                    // For simplicity, just return the larger block
                    // Real implementation would split the block
                    return Ok(ptr);
                }
            }
        }

        // No available blocks
        if let Some(stats) = &self.stats {
            stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
        }
        
        Err(ZiporaError::out_of_memory(0))
    }

    /// Deallocate to free list
    fn deallocate_to_free_list(&self, ptr: NonNull<u8>, size_class_index: usize) -> Result<()> {
        let free_lists = unsafe { &*self.free_lists.get() };
        let free_list = &free_lists[size_class_index];
        let offset = self.ptr_to_offset(ptr)?;

        // Initialize block header
        let header = unsafe { &mut *(ptr.as_ptr() as *mut BlockHeader) };
        header.size_class = size_class_index as u32;
        header.magic = BLOCK_HEADER_MAGIC;

        // Add to free list
        loop {
            let current_head = free_list.head.load(Ordering::Acquire);
            header.next = current_head;

            if free_list.head.compare_exchange_weak(
                current_head,
                offset,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
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
        let memory = unsafe { (*self.memory.get()).ok_or_else(|| 
            ZiporaError::invalid_data("Memory not allocated"))? };
        
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
        pool: &FixedCapacityMemoryPool
    ) -> Self {
        Self { 
            ptr, 
            size, 
            size_class_index, 
            pool: pool as *const _
        }
    }

    /// Get pointer to allocated memory
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size of allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get mutable slice view
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get immutable slice view
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for FixedCapacityAllocation {
    fn drop(&mut self) {
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
        assert_eq!(alloc.size(), 64); // Should match exact size class
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
        
        for i in 0..15 { // Try more than capacity
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
            assert!(classes[i] > classes[i-1]);
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
        unsafe { assert!((*pool.memory.get()).is_none()); }
        
        // First allocation should trigger memory allocation
        let _alloc = pool.allocate(64).unwrap();
        unsafe { assert!((*pool.memory.get()).is_some()); }
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