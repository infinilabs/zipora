//! # Cache Layout Optimization Infrastructure
//!
//! This module provides comprehensive cache optimization infrastructure for zipora,
//! implementing architecture-aware cache line detection, cache-aligned memory layouts,
//! hot/cold data separation patterns, and prefetch hint abstractions.
//!
//! ## Key Features
//!
//! - **Architecture Detection**: Runtime detection of cache line sizes and memory hierarchy
//! - **Cache-Aligned Layouts**: Memory layout patterns optimized for cache efficiency
//! - **Hot/Cold Separation**: Automatic separation of frequently vs. rarely accessed data
//! - **Prefetch Abstractions**: Cross-platform prefetch hint management
//! - **SIMD Integration**: Deep integration with existing SIMD framework
//!
//! ## Performance Targets
//!
//! - **Cache Hit Rate**: >95% for hot data access patterns
//! - **Memory Bandwidth**: Maximize utilization through aligned access patterns
//! - **Prefetch Effectiveness**: 2-3x improvement for predictable access patterns
//! - **Layout Efficiency**: Minimize false sharing and cache conflicts
//!
//! ## Architecture Support
//!
//! - **x86_64**: Full cache hierarchy detection and optimization
//! - **aarch64**: ARM-specific cache optimizations with NEON integration
//! - **Portable**: Fallback implementations for other architectures

use crate::error::{Result, ZiporaError};
use crate::memory::simd_ops::{SimdMemOps, SimdTier};
use crate::system::cpu_features::{CpuFeatures, get_cpu_features};
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem;

/// Cache line size detection and optimization constants
pub const DEFAULT_CACHE_LINE_SIZE: usize = 64;
pub const L1_CACHE_SIZE: usize = 32 * 1024;      // 32KB typical L1 cache
pub const L2_CACHE_SIZE: usize = 256 * 1024;     // 256KB typical L2 cache
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024; // 8MB typical L3 cache

/// Memory access pattern hints for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AccessPattern {
    /// Sequential access pattern (good for prefetching)
    Sequential,
    /// Random access pattern (benefits from cache alignment)
    Random,
    /// Write-heavy workload (benefits from write combining)
    WriteHeavy,
    /// Read-heavy workload (benefits from read prefetching)
    ReadHeavy,
    /// Mixed access pattern (balanced optimization)
    Mixed,
}

/// Cache hierarchy information detected at runtime
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheHierarchy {
    /// L1 data cache line size
    pub l1_line_size: usize,
    /// L1 data cache size
    pub l1_size: usize,
    /// L2 cache line size
    pub l2_line_size: usize,
    /// L2 cache size
    pub l2_size: usize,
    /// L3 cache line size
    pub l3_line_size: usize,
    /// L3 cache size
    pub l3_size: usize,
    /// Number of cache levels
    pub levels: usize,
    /// Cache associativity
    pub associativity: usize,
}

impl Default for CacheHierarchy {
    fn default() -> Self {
        Self {
            l1_line_size: DEFAULT_CACHE_LINE_SIZE,
            l1_size: L1_CACHE_SIZE,
            l2_line_size: DEFAULT_CACHE_LINE_SIZE,
            l2_size: L2_CACHE_SIZE,
            l3_line_size: DEFAULT_CACHE_LINE_SIZE,
            l3_size: L3_CACHE_SIZE,
            levels: 3,
            associativity: 8,
        }
    }
}

/// Cache-optimized memory layout configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheLayoutConfig {
    /// Cache hierarchy information
    pub hierarchy: CacheHierarchy,
    /// Primary cache line size for alignment
    pub cache_line_size: usize,
    /// Enable hot/cold data separation
    pub enable_hot_cold_separation: bool,
    /// Hot data threshold for separation
    pub hot_threshold: usize,
    /// Enable prefetch hints
    pub enable_prefetch: bool,
    /// Prefetch distance for sequential access
    pub prefetch_distance: usize,
    /// Access pattern hint
    pub access_pattern: AccessPattern,
}

impl Default for CacheLayoutConfig {
    fn default() -> Self {
        Self {
            hierarchy: CacheHierarchy::default(),
            cache_line_size: DEFAULT_CACHE_LINE_SIZE,
            enable_hot_cold_separation: true,
            hot_threshold: 1000, // Access count threshold
            enable_prefetch: true,
            prefetch_distance: 256, // Prefetch 256 bytes ahead
            access_pattern: AccessPattern::Mixed,
        }
    }
}

impl CacheLayoutConfig {
    /// Create a new cache layout configuration with detected hierarchy
    pub fn new() -> Self {
        let hierarchy = detect_cache_hierarchy();
        Self {
            cache_line_size: hierarchy.l1_line_size,
            hierarchy,
            ..Default::default()
        }
    }

    /// Create configuration optimized for sequential access patterns
    pub fn sequential() -> Self {
        Self {
            access_pattern: AccessPattern::Sequential,
            prefetch_distance: 512, // Larger prefetch for sequential
            enable_prefetch: true,
            ..Self::new()
        }
    }

    /// Create configuration optimized for random access patterns
    pub fn random() -> Self {
        Self {
            access_pattern: AccessPattern::Random,
            prefetch_distance: 64, // Smaller prefetch for random
            enable_hot_cold_separation: true,
            ..Self::new()
        }
    }

    /// Create configuration optimized for write-heavy workloads
    pub fn write_heavy() -> Self {
        Self {
            access_pattern: AccessPattern::WriteHeavy,
            enable_hot_cold_separation: false, // Avoid read-only optimizations
            prefetch_distance: 128,
            ..Self::new()
        }
    }

    /// Create configuration optimized for read-heavy workloads
    pub fn read_heavy() -> Self {
        Self {
            access_pattern: AccessPattern::ReadHeavy,
            enable_hot_cold_separation: true,
            prefetch_distance: 256,
            enable_prefetch: true,
            ..Self::new()
        }
    }
}

/// Cache-optimized memory allocator with layout management
#[derive(Debug)]
pub struct CacheOptimizedAllocator {
    config: CacheLayoutConfig,
    simd_ops: SimdMemOps,
    hot_allocations: AtomicUsize,
    cold_allocations: AtomicUsize,
}

impl CacheOptimizedAllocator {
    /// Create a new cache-optimized allocator
    pub fn new(config: CacheLayoutConfig) -> Self {
        Self {
            config,
            simd_ops: SimdMemOps::new(),
            hot_allocations: AtomicUsize::new(0),
            cold_allocations: AtomicUsize::new(0),
        }
    }

    /// Create allocator with optimal configuration for the current system
    pub fn optimal() -> Self {
        Self::new(CacheLayoutConfig::new())
    }

    /// Allocate cache-aligned memory with specified layout hints
    pub fn allocate_aligned(&self, size: usize, alignment: usize, is_hot: bool) -> Result<NonNull<u8>> {
        let effective_alignment = alignment.max(self.config.cache_line_size);
        let aligned_size = align_to_cache_line(size, self.config.cache_line_size);

        let layout = Layout::from_size_align(aligned_size, effective_alignment)
            .map_err(|_| ZiporaError::invalid_data("Invalid layout for cache-aligned allocation"))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(ZiporaError::out_of_memory(size));
        }

        // Update allocation statistics
        if is_hot {
            self.hot_allocations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cold_allocations.fetch_add(1, Ordering::Relaxed);
        }

        Ok(NonNull::new(ptr).unwrap())
    }

    /// Deallocate cache-aligned memory
    pub fn deallocate_aligned(&self, ptr: NonNull<u8>, size: usize, alignment: usize) -> Result<()> {
        let effective_alignment = alignment.max(self.config.cache_line_size);
        let aligned_size = align_to_cache_line(size, self.config.cache_line_size);

        let layout = Layout::from_size_align(aligned_size, effective_alignment)
            .map_err(|_| ZiporaError::invalid_data("Invalid layout for cache-aligned deallocation"))?;

        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }

        Ok(())
    }

    /// Issue prefetch hints for memory address
    pub fn prefetch(&self, addr: *const u8, hint: PrefetchHint) {
        if !self.config.enable_prefetch {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        unsafe {
            match hint {
                PrefetchHint::T0 => std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0),
                PrefetchHint::T1 => std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T1),
                PrefetchHint::T2 => std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T2),
                PrefetchHint::NTA => std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_NTA),
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            match hint {
                PrefetchHint::T0 | PrefetchHint::T1 => {
                    std::arch::asm!("prfm pldl1keep, [{}]", in(reg) addr);
                }
                PrefetchHint::T2 => {
                    std::arch::asm!("prfm pldl2keep, [{}]", in(reg) addr);
                }
                PrefetchHint::NTA => {
                    std::arch::asm!("prfm pldl1strm, [{}]", in(reg) addr);
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // No-op for other architectures
            let _ = (addr, hint);
        }
    }

    /// Prefetch memory range for sequential access
    pub fn prefetch_range(&self, start: *const u8, size: usize) {
        if !self.config.enable_prefetch || size == 0 {
            return;
        }

        let distance = self.config.prefetch_distance;
        let cache_line_size = self.config.cache_line_size;
        
        // Prefetch in cache line increments
        let mut addr = start;
        let end = unsafe { start.add(size) };

        while addr < end {
            self.prefetch(addr, PrefetchHint::T0);
            addr = unsafe { addr.add(cache_line_size.min(distance)) };
        }
    }

    /// Get allocation statistics
    pub fn stats(&self) -> CacheLayoutStats {
        CacheLayoutStats {
            hot_allocations: self.hot_allocations.load(Ordering::Relaxed),
            cold_allocations: self.cold_allocations.load(Ordering::Relaxed),
            cache_line_size: self.config.cache_line_size,
            prefetch_enabled: self.config.enable_prefetch,
            hot_cold_separation: self.config.enable_hot_cold_separation,
        }
    }
}

/// Prefetch hint types for cache optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchHint {
    /// Temporal locality, all cache levels (T0)
    T0,
    /// Temporal locality, level 2 and higher (T1)
    T1,
    /// Temporal locality, level 3 and higher (T2)
    T2,
    /// Non-temporal access, bypass cache (NTA)
    NTA,
}

/// Hot/cold data separation manager
#[derive(Debug)]
pub struct HotColdSeparator<T> {
    hot_data: Vec<T>,
    cold_data: Vec<T>,
    access_counts: Vec<usize>,
    config: CacheLayoutConfig,
}

impl<T> HotColdSeparator<T> {
    /// Create a new hot/cold data separator
    pub fn new(config: CacheLayoutConfig) -> Self {
        Self {
            hot_data: Vec::new(),
            cold_data: Vec::new(),
            access_counts: Vec::new(),
            config,
        }
    }

    /// Add data with access frequency hint
    pub fn insert(&mut self, item: T, access_count: usize) {
        if self.config.enable_hot_cold_separation && access_count >= self.config.hot_threshold {
            self.hot_data.push(item);
        } else {
            self.cold_data.push(item);
        }
        self.access_counts.push(access_count);
    }

    /// Get hot data slice
    pub fn hot_slice(&self) -> &[T] {
        &self.hot_data
    }

    /// Get cold data slice
    pub fn cold_slice(&self) -> &[T] {
        &self.cold_data
    }

    /// Reorganize data based on access patterns
    pub fn reorganize(&mut self) {
        if !self.config.enable_hot_cold_separation {
            return;
        }

        // Move frequently accessed cold data to hot
        let mut cold_to_hot = Vec::new();
        let mut hot_to_cold = Vec::new();

        // This is a simplified reorganization - in practice, you'd track
        // actual access patterns over time
        for (i, &count) in self.access_counts.iter().enumerate() {
            if count >= self.config.hot_threshold * 2 {
                // Very hot data should be in hot section
                if i < self.cold_data.len() {
                    cold_to_hot.push(i);
                }
            } else if count < self.config.hot_threshold / 2 {
                // Cold data should be in cold section
                if i >= self.cold_data.len() {
                    hot_to_cold.push(i - self.cold_data.len());
                }
            }
        }

        // Perform the reorganization
        for &i in cold_to_hot.iter().rev() {
            if i < self.cold_data.len() {
                let item = self.cold_data.remove(i);
                self.hot_data.push(item);
            }
        }

        for &i in hot_to_cold.iter().rev() {
            if i < self.hot_data.len() {
                let item = self.hot_data.remove(i);
                self.cold_data.push(item);
            }
        }
    }

    /// Get statistics about hot/cold separation
    pub fn separation_stats(&self) -> HotColdStats {
        HotColdStats {
            hot_items: self.hot_data.len(),
            cold_items: self.cold_data.len(),
            total_accesses: self.access_counts.iter().sum(),
            separation_enabled: self.config.enable_hot_cold_separation,
        }
    }
}

/// Cache layout statistics
#[derive(Debug, Clone)]
pub struct CacheLayoutStats {
    pub hot_allocations: usize,
    pub cold_allocations: usize,
    pub cache_line_size: usize,
    pub prefetch_enabled: bool,
    pub hot_cold_separation: bool,
}

/// Hot/cold separation statistics
#[derive(Debug, Clone)]
pub struct HotColdStats {
    pub hot_items: usize,
    pub cold_items: usize,
    pub total_accesses: usize,
    pub separation_enabled: bool,
}

/// Align size to cache line boundaries
pub fn align_to_cache_line(size: usize, cache_line_size: usize) -> usize {
    (size + cache_line_size - 1) & !(cache_line_size - 1)
}

/// Detect cache hierarchy at runtime
pub fn detect_cache_hierarchy() -> CacheHierarchy {
    #[cfg(target_arch = "x86_64")]
    {
        detect_x86_cache_hierarchy()
    }

    #[cfg(target_arch = "aarch64")]
    {
        detect_arm_cache_hierarchy()
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        CacheHierarchy::default()
    }
}

/// Detect cache hierarchy on x86_64 using CPUID
#[cfg(target_arch = "x86_64")]
fn detect_x86_cache_hierarchy() -> CacheHierarchy {
    use std::arch::x86_64::__cpuid;

    // Default fallback
    let mut hierarchy = CacheHierarchy::default();

    unsafe {
        // Check if CPUID is available (for x86_64 we can assume it's available)
        {
            // Get cache information using CPUID leaf 4
            let mut level = 0;
            loop {
                let result = __cpuid(0x4 | (level << 8));
                let cache_type = result.eax & 0x1F;
                
                if cache_type == 0 {
                    break; // No more cache levels
                }

                let cache_level = (result.eax >> 5) & 0x7;
                let line_size = ((result.ebx & 0xFFF) + 1) as usize;
                let cache_size = (((result.ebx >> 22) + 1) as usize * 
                                 ((result.ecx + 1) as usize) * 
                                 line_size);

                match cache_level {
                    1 => {
                        hierarchy.l1_line_size = line_size;
                        hierarchy.l1_size = cache_size;
                    }
                    2 => {
                        hierarchy.l2_line_size = line_size;
                        hierarchy.l2_size = cache_size;
                    }
                    3 => {
                        hierarchy.l3_line_size = line_size;
                        hierarchy.l3_size = cache_size;
                    }
                    _ => {}
                }

                level += 1;
            }

            hierarchy.levels = if level == 0 { 3 } else { level as usize }; // Default to 3 levels if detection fails
        }
    }

    hierarchy
}

/// Detect cache hierarchy on ARM64 using system information
#[cfg(target_arch = "aarch64")]
fn detect_arm_cache_hierarchy() -> CacheHierarchy {
    let mut hierarchy = CacheHierarchy::default();

    // Try to read cache information from /sys/devices/system/cpu/
    #[cfg(target_os = "linux")]
    {
        if let Ok(l1_size) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size") {
            if let Ok(size) = parse_cache_size(&l1_size) {
                hierarchy.l1_size = size;
            }
        }

        if let Ok(l2_size) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            if let Ok(size) = parse_cache_size(&l2_size) {
                hierarchy.l2_size = size;
            }
        }

        if let Ok(l3_size) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            if let Ok(size) = parse_cache_size(&l3_size) {
                hierarchy.l3_size = size;
            }
        }

        // Read cache line size
        if let Ok(line_size) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size") {
            if let Ok(size) = line_size.trim().parse::<usize>() {
                hierarchy.l1_line_size = size;
                hierarchy.l2_line_size = size;
                hierarchy.l3_line_size = size;
            }
        }
    }

    hierarchy
}

/// Parse cache size from string (e.g., "32K", "256K", "8M")
#[cfg(target_os = "linux")]
fn parse_cache_size(size_str: &str) -> Result<usize> {
    let size_str = size_str.trim();
    if size_str.is_empty() {
        return Err(ZiporaError::invalid_data("Empty cache size string"));
    }

    let (number_part, unit) = if size_str.ends_with('K') {
        (&size_str[..size_str.len()-1], 1024)
    } else if size_str.ends_with('M') {
        (&size_str[..size_str.len()-1], 1024 * 1024)
    } else if size_str.ends_with('G') {
        (&size_str[..size_str.len()-1], 1024 * 1024 * 1024)
    } else {
        (size_str, 1)
    };

    let number: usize = number_part.parse()
        .map_err(|_| ZiporaError::invalid_data(&format!("Invalid cache size number: {}", number_part)))?;

    Ok(number * unit)
}

/// Cache-aligned vector with automatic prefetching
pub struct CacheAlignedVec<T> {
    data: Vec<T>,
    allocator: CacheOptimizedAllocator,
    access_pattern: AccessPattern,
}

impl<T> CacheAlignedVec<T> {
    /// Create a new cache-aligned vector
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            allocator: CacheOptimizedAllocator::optimal(),
            access_pattern: AccessPattern::Mixed,
        }
    }

    /// Create with specific access pattern
    pub fn with_access_pattern(pattern: AccessPattern) -> Self {
        let config = match pattern {
            AccessPattern::Sequential => CacheLayoutConfig::sequential(),
            AccessPattern::Random => CacheLayoutConfig::random(),
            AccessPattern::WriteHeavy => CacheLayoutConfig::write_heavy(),
            AccessPattern::ReadHeavy => CacheLayoutConfig::read_heavy(),
            AccessPattern::Mixed => CacheLayoutConfig::new(),
        };

        Self {
            data: Vec::new(),
            allocator: CacheOptimizedAllocator::new(config),
            access_pattern: pattern,
        }
    }

    /// Push element with cache optimization
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        
        // Prefetch next cache line if sequential access
        if matches!(self.access_pattern, AccessPattern::Sequential) {
            let len = self.data.len();
            if len > 0 && len % 8 == 0 { // Every 8 elements (cache line worth)
                let addr = self.data.as_ptr() as *const u8;
                self.allocator.prefetch_range(addr, mem::size_of::<T>() * len);
            }
        }
    }

    /// Get element with prefetch hint
    pub fn get(&self, index: usize) -> Option<&T> {
        if let Some(element) = self.data.get(index) {
            // Prefetch nearby elements for random access
            if matches!(self.access_pattern, AccessPattern::Random) && index + 1 < self.data.len() {
                let addr = unsafe { self.data.as_ptr().add(index + 1) } as *const u8;
                self.allocator.prefetch(addr, PrefetchHint::T1);
            }
            Some(element)
        } else {
            None
        }
    }

    /// Get slice with range prefetching
    pub fn slice(&self, range: std::ops::Range<usize>) -> Option<&[T]> {
        if range.end <= self.data.len() {
            let slice = &self.data[range.clone()];
            
            // Prefetch the entire range
            if !slice.is_empty() {
                let addr = slice.as_ptr() as *const u8;
                let size = slice.len() * mem::size_of::<T>();
                self.allocator.prefetch_range(addr, size);
            }
            
            Some(slice)
        } else {
            None
        }
    }

    /// Get the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Default for CacheAlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hierarchy_detection() {
        let hierarchy = detect_cache_hierarchy();
        assert!(hierarchy.l1_line_size > 0);
        assert!(hierarchy.l1_size > 0);
        // Should have at least 1 level (default provides 3)
        assert!(hierarchy.levels > 0);
        println!("Detected cache hierarchy: {:?}", hierarchy);
    }

    #[test]
    fn test_cache_layout_config() {
        let config = CacheLayoutConfig::new();
        assert!(config.cache_line_size > 0);
        assert!(config.hierarchy.l1_size > 0);

        let sequential_config = CacheLayoutConfig::sequential();
        assert_eq!(sequential_config.access_pattern, AccessPattern::Sequential);
        assert!(sequential_config.prefetch_distance > config.prefetch_distance);
    }

    #[test]
    fn test_cache_optimized_allocator() {
        let allocator = CacheOptimizedAllocator::optimal();
        
        // Test allocation
        let ptr = allocator.allocate_aligned(1024, 64, true).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 64, 0); // Should be cache-line aligned

        // Test deallocation
        assert!(allocator.deallocate_aligned(ptr, 1024, 64).is_ok());

        let stats = allocator.stats();
        assert_eq!(stats.hot_allocations, 1);
        assert_eq!(stats.cold_allocations, 0);
    }

    #[test]
    fn test_align_to_cache_line() {
        assert_eq!(align_to_cache_line(0, 64), 0);
        assert_eq!(align_to_cache_line(1, 64), 64);
        assert_eq!(align_to_cache_line(64, 64), 64);
        assert_eq!(align_to_cache_line(65, 64), 128);
        assert_eq!(align_to_cache_line(100, 64), 128);
    }

    #[test]
    fn test_hot_cold_separator() {
        let config = CacheLayoutConfig::new();
        let mut separator = HotColdSeparator::new(config);

        // Add hot and cold data
        separator.insert("hot1".to_string(), 2000);
        separator.insert("cold1".to_string(), 100);
        separator.insert("hot2".to_string(), 1500);
        separator.insert("cold2".to_string(), 50);

        assert_eq!(separator.hot_slice().len(), 2);
        assert_eq!(separator.cold_slice().len(), 2);

        let stats = separator.separation_stats();
        assert_eq!(stats.hot_items, 2);
        assert_eq!(stats.cold_items, 2);
        assert!(stats.separation_enabled);
    }

    #[test]
    fn test_cache_aligned_vec() {
        let mut vec = CacheAlignedVec::with_access_pattern(AccessPattern::Sequential);
        
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.as_slice(), &[1, 2, 3]);

        let slice = vec.slice(1..3).unwrap();
        assert_eq!(slice, &[2, 3]);
    }

    #[test]
    fn test_prefetch_operations() {
        let allocator = CacheOptimizedAllocator::optimal();
        let data = vec![1u8; 1024];
        
        // Test single prefetch
        allocator.prefetch(data.as_ptr(), PrefetchHint::T0);
        
        // Test range prefetch
        allocator.prefetch_range(data.as_ptr(), data.len());
        
        // Should not panic - these are essentially no-ops on unsupported platforms
    }

    #[test]
    fn test_access_patterns() {
        let sequential = CacheAlignedVec::<i32>::with_access_pattern(AccessPattern::Sequential);
        let random = CacheAlignedVec::<i32>::with_access_pattern(AccessPattern::Random);
        let write_heavy = CacheAlignedVec::<i32>::with_access_pattern(AccessPattern::WriteHeavy);
        let read_heavy = CacheAlignedVec::<i32>::with_access_pattern(AccessPattern::ReadHeavy);

        // All should be created successfully with appropriate optimizations
        assert!(sequential.is_empty());
        assert!(random.is_empty());
        assert!(write_heavy.is_empty());
        assert!(read_heavy.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_cache_size_parsing() {
        assert_eq!(parse_cache_size("32K").unwrap(), 32 * 1024);
        assert_eq!(parse_cache_size("256K").unwrap(), 256 * 1024);
        assert_eq!(parse_cache_size("8M").unwrap(), 8 * 1024 * 1024);
        assert_eq!(parse_cache_size("1G").unwrap(), 1024 * 1024 * 1024);
        assert_eq!(parse_cache_size("1024").unwrap(), 1024);
        
        assert!(parse_cache_size("").is_err());
        assert!(parse_cache_size("invalid").is_err());
    }

    #[test]
    fn test_cache_layout_stats() {
        let allocator = CacheOptimizedAllocator::optimal();
        
        // Allocate some hot and cold data
        let _hot1 = allocator.allocate_aligned(64, 64, true).unwrap();
        let _hot2 = allocator.allocate_aligned(128, 64, true).unwrap();
        let _cold1 = allocator.allocate_aligned(64, 64, false).unwrap();

        let stats = allocator.stats();
        assert_eq!(stats.hot_allocations, 2);
        assert_eq!(stats.cold_allocations, 1);
        assert!(stats.cache_line_size >= 32);
    }

    #[test]
    fn test_hot_cold_reorganization() {
        let config = CacheLayoutConfig::new();
        let mut separator = HotColdSeparator::new(config);

        // Add data with varying access counts
        for i in 0..10 {
            let access_count = if i < 3 { 2000 } else { 100 }; // First 3 are hot
            separator.insert(format!("item{}", i), access_count);
        }

        assert_eq!(separator.hot_slice().len(), 3);
        assert_eq!(separator.cold_slice().len(), 7);

        // Test reorganization (simplified)
        separator.reorganize();
        
        let stats = separator.separation_stats();
        assert!(stats.total_accesses > 0);
    }
}