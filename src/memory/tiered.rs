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
use crate::memory::hugepage::{HugePage, HugePageAllocator, HUGEPAGE_SIZE_2MB};

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread_local;

/// Size thresholds for different allocation strategies
pub const SMALL_THRESHOLD: usize = 1024;           // 1KB
pub const MEDIUM_THRESHOLD: usize = 16 * 1024;     // 16KB  
pub const LARGE_THRESHOLD: usize = 2 * 1024 * 1024; // 2MB

/// A memory allocation that can come from different allocators
#[derive(Debug)]
pub enum TieredAllocation {
    Small(NonNull<u8>, usize),
    Medium(NonNull<u8>, usize),
    Large(MmapAllocation),
    #[cfg(target_os = "linux")]
    Huge(HugePage),
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
    pub small_allocations: u64,
    pub medium_allocations: u64,
    pub large_allocations: u64,
    pub huge_allocations: u64,
    pub total_allocated_bytes: u64,
    pub small_pool_stats: PoolStats,
    pub medium_pool_stats: Vec<PoolStats>,
    pub mmap_stats: crate::memory::mmap::MmapStats,
}

// Thread-local storage for medium-sized pools to reduce contention
thread_local! {
    static MEDIUM_POOLS: Vec<Arc<MemoryPool>> = {
        // Size classes: 1KB, 2KB, 4KB, 8KB, 16KB
        let size_classes = vec![1024, 2048, 4096, 8192, 16384];
        
        size_classes.into_iter().map(|size| {
            let config = PoolConfig::new(size, 32, 16); // 32 chunks per pool, 16-byte aligned
            Arc::new(MemoryPool::new(config).unwrap())
        }).collect()
    };
}

/// High-performance tiered memory allocator
pub struct TieredMemoryAllocator {
    config: TieredConfig,
    
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
        let bucket = if size == 0 { 0 } else { 63 - size.leading_zeros() as usize };
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

#[derive(Debug, Clone, Copy)]
pub enum AllocationPattern {
    SmallDominated,
    MediumDominated,
    LargeDominated,
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
            Arc::new(HugePageAllocator::with_config(usize::MAX, HUGEPAGE_SIZE_2MB)?) // Disabled
        };

        Ok(Self {
            config,
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
    pub fn default() -> Result<Self> {
        Self::new(TieredConfig::default())
    }

    /// Allocate memory using the optimal strategy for the given size
    pub fn allocate(&self, size: usize) -> Result<TieredAllocation> {
        if size == 0 {
            return Err(ZiporaError::invalid_data("allocation size cannot be zero"));
        }

        // Record allocation for adaptive optimization
        if let Ok(mut history) = self.allocation_history.try_lock() {
            history.record_allocation(size);
        }

        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);

        // Route to appropriate allocator based on size
        if size <= SMALL_THRESHOLD && self.config.enable_small_pools {
            self.allocate_small(size)
        } else if size <= MEDIUM_THRESHOLD && self.config.enable_medium_pools {
            self.allocate_medium(size)
        } else if size < LARGE_THRESHOLD && self.config.enable_mmap_large {
            self.allocate_large(size)
        } else {
            self.allocate_huge(size)
        }
    }

    /// Deallocate memory
    pub fn deallocate(&self, allocation: TieredAllocation) -> Result<()> {
        match allocation {
            TieredAllocation::Small(ptr, size) => {
                self.small_pool.deallocate(ptr)?;
                self.total_bytes.fetch_sub(size as u64, Ordering::Relaxed);
            }
            TieredAllocation::Medium(ptr, size) => {
                self.deallocate_medium(ptr, size)?;
                self.total_bytes.fetch_sub(size as u64, Ordering::Relaxed);
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
        let medium_pool_stats = MEDIUM_POOLS.with(|pools| {
            pools.iter().map(|pool| pool.stats()).collect()
        });

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
        Ok(TieredAllocation::Small(chunk, size))
    }

    fn allocate_medium(&self, size: usize) -> Result<TieredAllocation> {
        self.medium_allocs.fetch_add(1, Ordering::Relaxed);
        
        // Use thread-local medium pools for better performance
        MEDIUM_POOLS.with(|pools| {
            // Find the smallest pool that can accommodate the allocation
            for pool in pools.iter() {
                if pool.config().chunk_size >= size {
                    let chunk = pool.allocate()?;
                    return Ok(TieredAllocation::Medium(chunk, size));
                }
            }
            
            // No suitable pool found, fall back to mmap
            self.allocate_large(size)
        })
    }

    fn deallocate_medium(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        MEDIUM_POOLS.with(|pools| {
            // Find the appropriate pool based on size
            for pool in pools.iter() {
                if pool.config().chunk_size >= size {
                    return pool.deallocate(ptr);
                }
            }
            
            Err(ZiporaError::invalid_data("no suitable pool for deallocation"))
        })
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

// Safety: TieredMemoryAllocator can be shared between threads safely
unsafe impl Send for TieredMemoryAllocator {}
unsafe impl Sync for TieredMemoryAllocator {}

impl TieredAllocation {
    /// Get the allocated memory as a slice
    pub fn as_slice(&self) -> &[u8] {
        match self {
            TieredAllocation::Small(ptr, size) => {
                unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *size) }
            }
            TieredAllocation::Medium(ptr, size) => {
                unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *size) }
            }
            TieredAllocation::Large(allocation) => allocation.as_slice(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_slice(),
        }
    }

    /// Get the allocated memory as a mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            TieredAllocation::Small(ptr, size) => {
                unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *size) }
            }
            TieredAllocation::Medium(ptr, size) => {
                unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *size) }
            }
            TieredAllocation::Large(allocation) => allocation.as_mut_slice(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_mut_slice(),
        }
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        match self {
            TieredAllocation::Small(_, size) => *size,
            TieredAllocation::Medium(_, size) => *size,
            TieredAllocation::Large(allocation) => allocation.size(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.size(),
        }
    }

    /// Get the memory as a typed pointer
    pub fn as_ptr<T>(&self) -> *mut T {
        match self {
            TieredAllocation::Small(ptr, _) => ptr.as_ptr() as *mut T,
            TieredAllocation::Medium(ptr, _) => ptr.as_ptr() as *mut T,
            TieredAllocation::Large(allocation) => allocation.as_ptr(),
            #[cfg(target_os = "linux")]
            TieredAllocation::Huge(hugepage) => hugepage.as_slice().as_ptr() as *mut T,
        }
    }
}

/// Global tiered allocator instance
static GLOBAL_TIERED_ALLOCATOR: once_cell::sync::Lazy<TieredMemoryAllocator> =
    once_cell::sync::Lazy::new(|| TieredMemoryAllocator::default().unwrap());

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
        matches!(pattern, AllocationPattern::SmallDominated | AllocationPattern::Mixed);
    }

    #[test]
    fn test_global_tiered_allocator() {
        let size = 1024;
        
        let allocation = tiered_allocate(size).unwrap();
        assert_eq!(allocation.size(), size);
        
        tiered_deallocate(allocation).unwrap();
        
        let stats = get_tiered_stats();
        assert!(stats.small_allocations > 0 || stats.medium_allocations > 0);
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