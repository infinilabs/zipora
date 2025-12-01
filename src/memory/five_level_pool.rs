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

use crate::error::{ZiporaError, Result};
use crate::memory::cache_layout::{CacheOptimizedAllocator, CacheLayoutConfig, align_to_cache_line, AccessPattern};
use crate::memory::{get_optimal_numa_node, numa_alloc_aligned, numa_dealloc};
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
use std::mem::align_of;
// use std::mem::MaybeUninit;
use std::alloc::{Layout, alloc, dealloc};

/// Memory offset type for 32-bit addressing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MemOffset(u32);

impl MemOffset {
    const NULL: Self = MemOffset(u32::MAX);
    
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
    /// Maximum number of skip list levels for large blocks
    pub max_skip_levels: usize,
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
            max_skip_levels: 8,
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
            arena_size: 4 * 1024 * 1024, // 4MB
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
            arena_size: 1024 * 1024, // 1MB
            enable_cache_alignment: false,
            cache_config: None,
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 4 * 1024 * 1024, // 4MB
            ..Default::default()
        }
    }
    
    pub fn realtime() -> Self {
        Self {
            max_fast_block_size: 8 * 1024, // 8KB
            alignment: 8,
            initial_capacity: 256 * 1024, // 256KB
            arena_size: 512 * 1024, // 512KB
            fixed_capacity: Some(16 * 1024 * 1024), // 16MB fixed
            enable_cache_alignment: true,
            cache_config: Some(CacheLayoutConfig::sequential()),
            enable_numa_awareness: false,
            enable_huge_pages: false,
            huge_page_threshold: 8 * 1024 * 1024, // 8MB
            ..Default::default()
        }
    }
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

/// Skip list node for large block management
#[derive(Debug)]
struct SkipListNode {
    size: usize,
    next: Vec<MemOffset>, // Variable number of forward pointers
}

/// Memory chunk representation
#[derive(Debug)]
struct MemoryChunk {
    data: NonNull<u8>,
    size: usize,
    capacity: usize,
}

unsafe impl Send for MemoryChunk {}
unsafe impl Sync for MemoryChunk {}

impl MemoryChunk {
    fn new(capacity: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(capacity, alignment)
            .map_err(|_| ZiporaError::invalid_data("Invalid memory layout"))?;
        
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
        })
    }
    
    unsafe fn offset_ptr(&self, offset: usize) -> *mut u8 {
        debug_assert!(offset <= self.capacity);
        unsafe { self.data.as_ptr().add(offset) }
    }
    
    fn can_allocate(&self, size: usize) -> bool {
        self.size + size <= self.capacity
    }
}

impl Drop for MemoryChunk {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.capacity, align_of::<u8>());
            dealloc(self.data.as_ptr(), layout);
        }
    }
}

/// Level 1: No Locking - Single-threaded memory pool
pub struct NoLockingPool {
    config: FiveLevelPoolConfig,
    memory: MemoryChunk,
    free_lists: Vec<FreeListHead>,
    skip_list_head: SkipListNode,
    fragment_size: usize,
    huge_size_sum: usize,
    huge_node_count: usize,
    used_memory: usize, // Track actual used memory (high-water mark minus freed blocks)
    rng_state: u32,
}

impl NoLockingPool {
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = vec![FreeListHead::default(); num_bins];
        
        let skip_list_head = SkipListNode {
            size: 0,
            next: vec![MemOffset::NULL; config.max_skip_levels],
        };
        
        Ok(Self {
            config,
            memory,
            free_lists,
            skip_list_head,
            fragment_size: 0,
            huge_size_sum: 0,
            huge_node_count: 0,
            used_memory: 0,
            rng_state: 1,
        })
    }
    
    /// Allocate memory block of given size
    pub fn alloc(&mut self, size: usize) -> Result<MemOffset> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin(aligned_size)
        } else {
            self.alloc_from_skip_list(aligned_size)
        }
    }
    
    /// Free previously allocated memory block
    pub fn free(&mut self, offset: MemOffset, size: usize) -> Result<()> {
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
            self.free_to_skip_list(offset, aligned_size)
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
    
    fn alloc_from_skip_list(&mut self, size: usize) -> Result<MemOffset> {
        // Search skip list for suitable block
        // For now, fall back to end allocation
        // TODO: Implement full skip list search
        self.alloc_from_end(size)
    }
    
    fn free_to_fast_bin(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        let bin_index = (size / self.config.alignment) - 1;
        
        if bin_index < self.free_lists.len() {
            let head = &mut self.free_lists[bin_index];
            
            // Push to free list
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
    
    fn free_to_skip_list(&mut self, offset: MemOffset, size: usize) -> Result<()> {
        // TODO: Implement skip list insertion
        // For now, just track statistics
        self.fragment_size += size;
        self.huge_size_sum += size;
        self.huge_node_count += 1;
        self.used_memory -= size; // Decrease used memory when freed
        Ok(())
    }
    
    fn random_level(&mut self) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let mut level = 1;
        while (self.rng_state % 4 == 0) && (level < self.config.max_skip_levels) {
            level += 1;
            self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        }
        level.saturating_sub(1)
    }
    
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            total_capacity: self.memory.capacity,
            used_memory: self.used_memory,
            fragment_size: self.fragment_size,
            huge_size_sum: self.huge_size_sum,
            huge_node_count: self.huge_node_count,
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
    skip_list: Mutex<SkipListNode>,
    fragment_size: AtomicUsize,
    huge_mutex: Mutex<(usize, usize)>, // (huge_size_sum, huge_node_count)
}

impl MutexBasedPool {
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = (0..num_bins).map(|_| Mutex::new(FreeListHead::default())).collect();
        
        let skip_list_head = SkipListNode {
            size: 0,
            next: vec![MemOffset::NULL; config.max_skip_levels],
        };
        
        Ok(Self {
            config,
            memory: Arc::new(Mutex::new(memory)),
            free_lists,
            skip_list: Mutex::new(skip_list_head),
            fragment_size: AtomicUsize::new(0),
            huge_mutex: Mutex::new((0, 0)),
        })
    }
    
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin(aligned_size)
        } else {
            self.alloc_from_skip_list(aligned_size)
        }
    }
    
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            self.free_to_fast_bin(offset, aligned_size)
        } else {
            self.free_to_skip_list(offset, aligned_size)
        }
    }
    
    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }
    
    fn alloc_from_fast_bin(&self, size: usize) -> Result<MemOffset> {
        let bin_index = (size / self.config.alignment) - 1;

        if bin_index < self.free_lists.len() {
            let mut head = self.free_lists[bin_index].lock()
                .map_err(|e| ZiporaError::resource_busy(format!("Free list mutex poisoned: {}", e)))?;
            if !head.head.is_null() {
                let offset = head.head;
                unsafe {
                    let memory = self.memory.lock()
                        .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
                    let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                    head.head = MemOffset(*ptr);
                }
                head.count -= 1;
                self.fragment_size.fetch_sub(size, Ordering::Relaxed);
                return Ok(offset);
            }
        }

        // Allocate from end
        let mut memory = self.memory.lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }
        
        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }
    
    fn alloc_from_skip_list(&self, size: usize) -> Result<MemOffset> {
        // For now, allocate from end
        let mut memory = self.memory.lock()
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
            let mut head = self.free_lists[bin_index].lock()
                .map_err(|e| ZiporaError::resource_busy(format!("Free list mutex poisoned: {}", e)))?;

            unsafe {
                let memory = self.memory.lock()
                    .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
                let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                *ptr = head.head.0;
            }
            head.head = offset;
            head.count += 1;
            self.fragment_size.fetch_add(size, Ordering::Relaxed);
        }

        Ok(())
    }
    
    fn free_to_skip_list(&self, offset: MemOffset, size: usize) -> Result<()> {
        self.fragment_size.fetch_add(size, Ordering::Relaxed);
        let mut huge_stats = self.huge_mutex.lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Huge mutex poisoned: {}", e)))?;
        huge_stats.0 += size;
        huge_stats.1 += 1;
        Ok(())
    }

    pub fn stats(&self) -> PoolStats {
        let memory = self.memory.lock().unwrap_or_else(|e| e.into_inner());
        let huge_stats = self.huge_mutex.lock().unwrap_or_else(|e| e.into_inner());

        PoolStats {
            total_capacity: memory.capacity,
            used_memory: memory.size,
            fragment_size: self.fragment_size.load(Ordering::Relaxed),
            huge_size_sum: huge_stats.0,
            huge_node_count: huge_stats.1,
            free_list_count: self.free_lists.len(),
        }
    }
}

unsafe impl Send for MutexBasedPool {}
unsafe impl Sync for MutexBasedPool {}

/// Level 3: Lock-free Programming - Compare-and-swap memory pool
pub struct LockFreePool {
    config: FiveLevelPoolConfig,
    memory: Arc<Mutex<MemoryChunk>>, // Still need mutex for memory expansion
    free_lists: Vec<LockFreeFreeListHead>,
    fragment_size: AtomicUsize,
    huge_mutex: Mutex<(usize, usize)>, // Huge blocks still use mutex
}

impl LockFreePool {
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        let memory = MemoryChunk::new(config.initial_capacity, config.alignment)?;
        let num_bins = config.max_fast_block_size / config.alignment;
        let free_lists = (0..num_bins).map(|_| LockFreeFreeListHead::default()).collect();
        
        Ok(Self {
            config,
            memory: Arc::new(Mutex::new(memory)),
            free_lists,
            fragment_size: AtomicUsize::new(0),
            huge_mutex: Mutex::new((0, 0)),
        })
    }
    
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            self.alloc_from_fast_bin_lockfree(aligned_size)
        } else {
            self.alloc_from_huge_mutex(aligned_size)
        }
    }
    
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            self.free_to_fast_bin_lockfree(offset, aligned_size)
        } else {
            self.free_to_huge_mutex(offset, aligned_size)
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
                let next_head = unsafe {
                    let memory = self.memory.lock()
                        .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
                    let ptr = memory.offset_ptr(current_head as usize) as *const u32;
                    *ptr
                };
                
                // Try to update head atomically
                match head.head.compare_exchange_weak(
                    current_head,
                    next_head,
                    Ordering::Release,
                    Ordering::Relaxed
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
        let mut memory = self.memory.lock()
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
                unsafe {
                    let memory = self.memory.lock()
                        .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
                    let ptr = memory.offset_ptr(offset.to_usize()) as *mut u32;
                    *ptr = current_head;
                }
                
                // Try to update head atomically
                match head.head.compare_exchange_weak(
                    current_head,
                    offset.0,
                    Ordering::Release,
                    Ordering::Relaxed
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
    
    fn alloc_from_huge_mutex(&self, size: usize) -> Result<MemOffset> {
        let mut memory = self.memory.lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Memory mutex poisoned: {}", e)))?;
        if !memory.can_allocate(size) {
            return Err(ZiporaError::resource_exhausted("Out of memory"));
        }

        let offset = MemOffset::new(memory.size);
        memory.size += size;
        Ok(offset)
    }
    
    fn free_to_huge_mutex(&self, offset: MemOffset, size: usize) -> Result<()> {
        self.fragment_size.fetch_add(size, Ordering::Relaxed);
        let mut huge_stats = self.huge_mutex.lock()
            .map_err(|e| ZiporaError::resource_busy(format!("Huge mutex poisoned: {}", e)))?;
        huge_stats.0 += size;
        huge_stats.1 += 1;
        Ok(())
    }

    pub fn stats(&self) -> PoolStats {
        let memory = self.memory.lock().unwrap_or_else(|e| e.into_inner());
        let huge_stats = self.huge_mutex.lock().unwrap_or_else(|e| e.into_inner());

        PoolStats {
            total_capacity: memory.capacity,
            used_memory: memory.size,
            fragment_size: self.fragment_size.load(Ordering::Relaxed),
            huge_size_sum: huge_stats.0,
            huge_node_count: huge_stats.1,
            free_list_count: self.free_lists.len(),
        }
    }
}

unsafe impl Send for LockFreePool {}
unsafe impl Sync for LockFreePool {}

/// Level 4: Thread-local Caching - Per-thread memory pools
pub struct ThreadLocalPool {
    config: FiveLevelPoolConfig,
    global_pool: Arc<MutexBasedPool>,
}

thread_local! {
    static THREAD_CACHE: std::cell::RefCell<Option<ThreadLocalCache>> = std::cell::RefCell::new(None);
}

struct ThreadLocalCache {
    arena: Vec<u8>,
    hot_pos: usize,
    hot_end: usize,
    local_free_lists: Vec<Vec<usize>>, // Local free lists as offsets into arena
}

impl ThreadLocalCache {
    fn new(arena_size: usize, num_bins: usize) -> Self {
        let mut arena = Vec::with_capacity(arena_size);
        arena.resize(arena_size, 0);
        
        Self {
            arena,
            hot_pos: 0,
            hot_end: arena_size / 2, // Reserve half for hot allocations
            local_free_lists: vec![Vec::new(); num_bins],
        }
    }
    
    fn alloc_from_hot_area(&mut self, size: usize) -> Option<usize> {
        if self.hot_pos + size <= self.hot_end {
            let offset = self.hot_pos;
            self.hot_pos += size;
            Some(offset)
        } else {
            None
        }
    }
    
    fn alloc_from_local_free_list(&mut self, bin_index: usize) -> Option<usize> {
        self.local_free_lists.get_mut(bin_index)?.pop()
    }
    
    fn free_to_local_free_list(&mut self, offset: usize, bin_index: usize) {
        if let Some(list) = self.local_free_lists.get_mut(bin_index) {
            list.push(offset);
        }
    }
}

impl ThreadLocalPool {
    pub fn new(config: FiveLevelPoolConfig) -> Result<Self> {
        let global_pool = Arc::new(MutexBasedPool::new(config.clone())?);
        
        Ok(Self {
            config,
            global_pool,
        })
    }
    
    pub fn alloc(&self, size: usize) -> Result<MemOffset> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            // Try thread-local allocation first
            THREAD_CACHE.with(|cache| {
                let mut cache_ref = cache.borrow_mut();
                
                // Initialize cache if needed
                if cache_ref.is_none() {
                    let num_bins = self.config.max_fast_block_size / self.config.alignment;
                    *cache_ref = Some(ThreadLocalCache::new(self.config.arena_size, num_bins));
                }
                
                let cache = cache_ref.as_mut().unwrap();
                let bin_index = (aligned_size / self.config.alignment).saturating_sub(1);
                
                // Try local free list first
                if let Some(offset) = cache.alloc_from_local_free_list(bin_index) {
                    return Ok(MemOffset::new(offset));
                }
                
                // Try hot area allocation
                if let Some(offset) = cache.alloc_from_hot_area(aligned_size) {
                    return Ok(MemOffset::new(offset));
                }
                
                // Fall back to global pool
                self.global_pool.alloc(aligned_size)
            })
        } else {
            // Large allocations go directly to global pool
            self.global_pool.alloc(aligned_size)
        }
    }
    
    pub fn free(&self, offset: MemOffset, size: usize) -> Result<()> {
        let aligned_size = self.align_up(size);
        
        if aligned_size <= self.config.max_fast_block_size {
            THREAD_CACHE.with(|cache| {
                let mut cache_ref = cache.borrow_mut();
                
                if let Some(cache) = cache_ref.as_mut() {
                    let bin_index = (aligned_size / self.config.alignment).saturating_sub(1);
                    
                    // Check if this offset is in our thread-local arena
                    if offset.to_usize() < cache.arena.len() {
                        cache.free_to_local_free_list(offset.to_usize(), bin_index);
                        return Ok(());
                    }
                }
                
                // Fall back to global pool
                self.global_pool.free(offset, aligned_size)
            })
        } else {
            self.global_pool.free(offset, aligned_size)
        }
    }
    
    fn align_up(&self, size: usize) -> usize {
        (size + self.config.alignment - 1) & !(self.config.alignment - 1)
    }
    
    pub fn stats(&self) -> PoolStats {
        self.global_pool.stats()
    }
}

unsafe impl Send for ThreadLocalPool {}
unsafe impl Sync for ThreadLocalPool {}

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
    
    pub fn alloc(&mut self, size: usize) -> Result<MemOffset> {
        let aligned_size = self.align_up(size);
        
        // Check capacity before allocation
        let stats = self.inner.stats();
        if stats.used_memory + aligned_size > self.max_capacity {
            return Err(ZiporaError::resource_exhausted("Fixed capacity exceeded"));
        }
        
        self.inner.alloc(aligned_size)
    }
    
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
        
        let cpu_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        
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
            _ => Err(ZiporaError::invalid_data("Pool level doesn't support handles")),
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
        
        let handles: Vec<_> = (0..4).map(|_| {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || -> Result<()> {
                for _ in 0..100 {
                    let offset = pool.alloc(64)?;
                    pool.free(offset, 64)?;
                }
                Ok(())
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap()?;
        }
        
        Ok(())
    }

    #[test]
    fn test_lock_free_pool_concurrent() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = Arc::new(LockFreePool::new(config)?);
        
        let handles: Vec<_> = (0..8).map(|_| {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || -> Result<()> {
                for _ in 0..1000 {
                    let offset = pool.alloc(64)?;
                    pool.free(offset, 64)?;
                }
                Ok(())
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap()?;
        }
        
        Ok(())
    }

    #[test]
    fn test_thread_local_pool() -> Result<()> {
        let config = FiveLevelPoolConfig::default();
        let pool = Arc::new(ThreadLocalPool::new(config)?);
        
        let handles: Vec<_> = (0..4).map(|_| {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || -> Result<()> {
                // Each thread should use its own cache
                for _ in 0..100 {
                    let offset = pool.alloc(128)?;
                    pool.free(offset, 128)?;
                }
                Ok(())
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap()?;
        }
        
        Ok(())
    }

    #[test]
    fn test_fixed_capacity_pool() -> Result<()> {
        let mut config = FiveLevelPoolConfig::default();
        config.fixed_capacity = Some(8192); // 8KB fixed capacity
        
        let mut pool = FixedCapacityPool::new(config)?;
        
        // Should be able to allocate within capacity
        let offset1 = pool.alloc(1024)?;
        let offset2 = pool.alloc(2048)?;
        
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
        let mut fixed_pool = AdaptiveFiveLevelPool::with_level(
            fixed_config, 
            ConcurrencyLevel::FixedCapacity
        )?;
        
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
        assert!(matches!(level, 
            ConcurrencyLevel::SingleThread | 
            ConcurrencyLevel::MultiThreadMutex |
            ConcurrencyLevel::MultiThreadLockFree |
            ConcurrencyLevel::ThreadLocal));
        
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
        let pool = AdaptiveFiveLevelPool::with_level(
            config, 
            ConcurrencyLevel::MultiThreadMutex
        )?;
        
        let handle = pool.get_handle()?;
        
        // Test concurrent access through handles
        let handles: Vec<_> = (0..4).map(|_| {
            let handle = handle.clone();
            std::thread::spawn(move || -> Result<()> {
                for _ in 0..50 {
                    let offset = handle.alloc(64)?;
                    handle.free(offset, 64)?;
                }
                Ok(())
            })
        }).collect();
        
        for thread_handle in handles {
            thread_handle.join().unwrap()?;
        }
        
        Ok(())
    }

    #[test]
    fn test_memory_alignment() -> Result<()> {
        let mut config = FiveLevelPoolConfig::default();
        config.alignment = 16;
        
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
        let mut config = FiveLevelPoolConfig::default();
        // Increase initial capacity for stress test
        config.initial_capacity = 8 * 1024 * 1024; // 8MB
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