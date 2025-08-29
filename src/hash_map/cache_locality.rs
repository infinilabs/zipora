//! Cache Locality Optimizations for Hash Maps
//!
//! This module provides sophisticated cache locality optimizations for hash table implementations,
//! featuring memory layout optimizations, prefetching strategies, NUMA awareness, and cache
//! performance monitoring. Inspired by state-of-the-art research in cache-conscious data structures.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _MM_HINT_NTA};
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache line size in bytes (typical for modern x86_64 processors)
pub const CACHE_LINE_SIZE: usize = 64;

/// L1 cache size hint (typical: 32KB)
pub const L1_CACHE_SIZE: usize = 32 * 1024;

/// L2 cache size hint (typical: 256KB)
pub const L2_CACHE_SIZE: usize = 256 * 1024;

/// L3 cache size hint (typical: 8MB)
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

/// Prefetch distance for sequential access patterns
pub const PREFETCH_DISTANCE: usize = 4;

/// NUMA node count (detected at runtime)
static NUMA_NODE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Cache performance metrics
#[derive(Debug, Default, Clone)]
pub struct CacheMetrics {
    /// L1 cache hits (estimated)
    pub l1_hits: u64,
    /// L1 cache misses (estimated)
    pub l1_misses: u64,
    /// L2 cache hits (estimated)
    pub l2_hits: u64,
    /// L2 cache misses (estimated)
    pub l2_misses: u64,
    /// L3 cache hits (estimated)
    pub l3_hits: u64,
    /// L3 cache misses (estimated)
    pub l3_misses: u64,
    /// Prefetch operations performed
    pub prefetch_count: u64,
    /// Memory stalls detected
    pub memory_stalls: u64,
    /// Cache line invalidations
    pub cache_invalidations: u64,
    /// False sharing incidents detected
    pub false_sharing_detected: u64,
}

impl CacheMetrics {
    /// Calculate overall cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total_accesses = total_hits + self.l1_misses + self.l2_misses + self.l3_misses;
        if total_accesses == 0 {
            0.0
        } else {
            total_hits as f64 / total_accesses as f64
        }
    }

    /// Estimate memory bandwidth usage
    pub fn estimated_bandwidth_gb(&self) -> f64 {
        let cache_line_transfers = self.l1_misses + self.l2_misses + self.l3_misses;
        (cache_line_transfers * CACHE_LINE_SIZE as u64) as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Cache-line aligned allocation wrapper
#[repr(align(64))]
pub struct CacheAligned<T> {
    data: T,
}

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get a reference to the inner data
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Get a mutable reference to the inner data
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consume and return the inner data
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Prefetch hint levels for different cache hierarchies
#[derive(Debug, Clone, Copy)]
pub enum PrefetchHint {
    /// Prefetch to all cache levels (T0)
    AllLevels,
    /// Prefetch to L2 and L3 only (T1)
    L2L3,
    /// Prefetch to L3 only (T2)
    L3Only,
    /// Non-temporal prefetch (bypass cache)
    NonTemporal,
}

/// Software prefetching utilities
pub struct Prefetcher;

impl Prefetcher {
    /// Prefetch a memory location with specified hint
    #[inline(always)]
    pub unsafe fn prefetch<T>(ptr: *const T, hint: PrefetchHint) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            match hint {
                PrefetchHint::AllLevels => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
                PrefetchHint::L2L3 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                PrefetchHint::L3Only => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                PrefetchHint::NonTemporal => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
            }
        }
    }

    /// Prefetch multiple sequential cache lines
    #[inline(always)]
    pub unsafe fn prefetch_range<T>(start: *const T, count: usize, hint: PrefetchHint) {
        unsafe {
            let bytes_per_element = size_of::<T>();
            let mut current = start as *const u8;
            
            for _ in 0..count {
                Self::prefetch(current as *const T, hint);
                current = current.add(bytes_per_element);
            }
        }
    }

    /// Prefetch with stride pattern (for hash table probing)
    #[inline(always)]
    pub unsafe fn prefetch_strided<T>(
        base: *const T,
        stride: usize,
        count: usize,
        hint: PrefetchHint,
    ) {
        unsafe {
            let mut current = base;
            for _ in 0..count {
                Self::prefetch(current, hint);
                current = current.add(stride);
            }
        }
    }
}

/// Cache-conscious bucket layout for hash tables
#[repr(C, align(64))]
pub struct CacheOptimizedBucket<K, V, const ENTRIES_PER_BUCKET: usize = 7> {
    /// Metadata packed into first cache line
    pub metadata: BucketMetadata,
    /// Hash values for fast comparison (7 entries + 1 overflow pointer fits in cache line)
    pub hashes: [u32; ENTRIES_PER_BUCKET],
    /// Key-value pairs (may span multiple cache lines)
    pub entries: [MaybeUninit<(K, V)>; ENTRIES_PER_BUCKET],
    /// Overflow bucket pointer for chaining
    pub overflow: Option<NonNull<Self>>,
    /// Padding to ensure cache line alignment
    pub _padding: [u8; 0], // Compiler will add necessary padding
}

/// Compact metadata for bucket (fits in 8 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BucketMetadata {
    /// Occupancy bitmap (which slots are filled)
    pub occupancy: u8,
    /// Lock or version counter for concurrent access
    pub lock_or_version: u8,
    /// Distance from ideal position (for Robin Hood)
    pub max_probe_distance: u16,
    /// Reserved for future use
    pub reserved: u32,
}

impl<K, V, const N: usize> CacheOptimizedBucket<K, V, N> {
    /// Create a new empty bucket
    pub fn new() -> Self {
        Self {
            metadata: BucketMetadata {
                occupancy: 0,
                lock_or_version: 0,
                max_probe_distance: 0,
                reserved: 0,
            },
            hashes: [0; N],
            entries: unsafe { MaybeUninit::uninit().assume_init() },
            overflow: None,
            _padding: [],
        }
    }

    /// Check if bucket is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.metadata.occupancy == 0
    }

    /// Count occupied slots
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.metadata.occupancy.count_ones() as usize
    }

    /// Find entry with matching hash (SIMD-optimizable)
    #[inline(always)]
    pub fn find_hash(&self, hash: u32) -> Option<usize> {
        // This can be SIMD-optimized with AVX2/AVX512
        for i in 0..N {
            if (self.metadata.occupancy & (1 << i)) != 0 && self.hashes[i] == hash {
                return Some(i);
            }
        }
        None
    }

    /// Prefetch bucket data
    #[inline(always)]
    pub unsafe fn prefetch(&self) {
        unsafe {
            Prefetcher::prefetch(self as *const _, PrefetchHint::AllLevels);
            // Prefetch the second cache line if entries span multiple lines
            if size_of::<(K, V)>() * N > CACHE_LINE_SIZE {
                let second_line = (self as *const _ as *const u8).add(CACHE_LINE_SIZE);
                Prefetcher::prefetch(second_line as *const Self, PrefetchHint::L2L3);
            }
        }
    }
}

/// NUMA-aware memory allocator for hash tables
pub struct NumaAllocator {
    /// Preferred NUMA node
    preferred_node: usize,
    /// Allocation statistics per node
    node_stats: Vec<AtomicU64>,
}

impl NumaAllocator {
    /// Create a new NUMA-aware allocator
    pub fn new() -> Self {
        let node_count = Self::detect_numa_nodes();
        NUMA_NODE_COUNT.store(node_count, Ordering::Relaxed);
        
        Self {
            preferred_node: Self::current_numa_node(),
            node_stats: (0..node_count).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Detect number of NUMA nodes
    fn detect_numa_nodes() -> usize {
        // In a real implementation, this would use libnuma or sysfs
        // For now, return 1 (UMA system)
        1
    }

    /// Get current CPU's NUMA node
    fn current_numa_node() -> usize {
        // In a real implementation, this would query the current CPU's node
        0
    }

    /// Allocate memory on preferred NUMA node
    pub unsafe fn alloc_on_node(&self, layout: Layout, node: usize) -> *mut u8 {
        unsafe {
            // In a real implementation, this would use numa_alloc_onnode
            // For now, use standard aligned allocation
            let ptr = alloc_zeroed(layout);
            if !ptr.is_null() {
                self.node_stats[node.min(self.node_stats.len() - 1)]
                    .fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            ptr
        }
    }

    /// Allocate cache-line aligned memory
    pub unsafe fn alloc_cache_aligned<T>(&self, count: usize) -> *mut T {
        unsafe {
            let layout = Layout::from_size_align(
                size_of::<T>() * count,
                CACHE_LINE_SIZE.max(align_of::<T>()),
            )
            .expect("Invalid layout");
            
            self.alloc_on_node(layout, self.preferred_node) as *mut T
        }
    }

    /// Free previously allocated memory
    pub unsafe fn dealloc<T>(&self, ptr: *mut T, count: usize) {
        unsafe {
            let layout = Layout::from_size_align(
                size_of::<T>() * count,
                CACHE_LINE_SIZE.max(align_of::<T>()),
            )
            .expect("Invalid layout");
            
            dealloc(ptr as *mut u8, layout);
        }
    }
}

/// Cache-conscious hash table layout optimizer
pub struct CacheLayoutOptimizer<K, V> {
    /// Estimated working set size
    working_set_size: usize,
    /// Cache level to optimize for
    target_cache_level: CacheLevel,
    /// Access pattern hint
    access_pattern: AccessPattern,
    /// Phantom data for types
    _phantom: PhantomData<(K, V)>,
}

/// Cache hierarchy levels
#[derive(Debug, Clone, Copy)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    Memory,
}

/// Expected access patterns
#[derive(Debug, Clone, Copy)]
pub enum AccessPattern {
    /// Random access pattern
    Random,
    /// Sequential access pattern
    Sequential,
    /// Strided access pattern
    Strided(usize),
    /// Temporal locality (repeated access)
    Temporal,
}

impl<K, V> CacheLayoutOptimizer<K, V> {
    /// Create a new layout optimizer
    pub fn new(working_set_size: usize) -> Self {
        let target_cache_level = if working_set_size <= L1_CACHE_SIZE {
            CacheLevel::L1
        } else if working_set_size <= L2_CACHE_SIZE {
            CacheLevel::L2
        } else if working_set_size <= L3_CACHE_SIZE {
            CacheLevel::L3
        } else {
            CacheLevel::Memory
        };

        Self {
            working_set_size,
            target_cache_level,
            access_pattern: AccessPattern::Random,
            _phantom: PhantomData,
        }
    }

    /// Set expected access pattern
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }

    /// Calculate optimal bucket size for cache efficiency
    pub fn optimal_bucket_size(&self) -> usize {
        let entry_size = size_of::<(K, V)>() + size_of::<u32>(); // Entry + hash
        let cache_line_capacity = CACHE_LINE_SIZE / entry_size;
        
        match self.target_cache_level {
            CacheLevel::L1 => {
                // For L1, minimize bucket size for best latency
                cache_line_capacity.min(4)
            }
            CacheLevel::L2 => {
                // For L2, use full cache line
                cache_line_capacity.min(8)
            }
            CacheLevel::L3 | CacheLevel::Memory => {
                // For L3/Memory, use multiple cache lines for better throughput
                (cache_line_capacity * 2).min(16)
            }
        }
    }

    /// Calculate optimal load factor based on cache constraints
    pub fn optimal_load_factor(&self) -> f64 {
        match self.target_cache_level {
            CacheLevel::L1 => 0.5,  // Lower load factor for L1 efficiency
            CacheLevel::L2 => 0.65, // Moderate load factor
            CacheLevel::L3 => 0.75, // Standard load factor
            CacheLevel::Memory => 0.85, // Higher load factor for memory
        }
    }

    /// Determine if hot/cold separation is beneficial
    pub fn should_separate_hot_cold(&self) -> bool {
        matches!(self.access_pattern, AccessPattern::Temporal)
            && self.working_set_size > L2_CACHE_SIZE
    }
}

/// Hot/Cold data separator for improved cache utilization
pub struct HotColdSeparator<T> {
    /// Hot data (frequently accessed)
    hot: Vec<CacheAligned<T>>,
    /// Cold data (infrequently accessed)
    cold: Vec<T>,
    /// Access counter for migration decisions
    access_counts: Vec<AtomicU64>,
    /// Migration threshold
    migration_threshold: u64,
}

impl<T: Clone> HotColdSeparator<T> {
    /// Create a new hot/cold separator
    pub fn new(capacity: usize, hot_ratio: f64) -> Self {
        let hot_capacity = ((capacity as f64) * hot_ratio) as usize;
        let cold_capacity = capacity - hot_capacity;
        
        Self {
            hot: Vec::with_capacity(hot_capacity),
            cold: Vec::with_capacity(cold_capacity),
            access_counts: Vec::with_capacity(capacity),
            migration_threshold: 10, // Migrate after 10 accesses
        }
    }

    /// Access an element and track access pattern
    pub fn access(&self, index: usize) -> Option<&T> {
        self.access_counts[index].fetch_add(1, Ordering::Relaxed);
        
        if index < self.hot.len() {
            Some(self.hot[index].get())
        } else {
            let cold_index = index - self.hot.len();
            self.cold.get(cold_index)
        }
    }

    /// Migrate data between hot and cold based on access patterns
    pub fn rebalance(&mut self) {
        // Collect access statistics
        let mut access_stats: Vec<(usize, u64)> = self
            .access_counts
            .iter()
            .enumerate()
            .map(|(i, count)| (i, count.load(Ordering::Relaxed)))
            .collect();
        
        // Sort by access count
        access_stats.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        
        // Determine new hot set
        let hot_capacity = self.hot.capacity();
        let new_hot_indices: Vec<usize> = access_stats
            .iter()
            .take(hot_capacity)
            .map(|&(idx, _)| idx)
            .collect();
        
        // Perform migration (simplified - in practice would be more sophisticated)
        // This would involve moving data between hot and cold vectors
        
        // Reset access counts
        for count in &self.access_counts {
            count.store(0, Ordering::Relaxed);
        }
    }
}

/// Cache-conscious resize strategy for hash tables
pub struct CacheConsciousResizer {
    /// Current table size
    current_size: usize,
    /// Target size after resize
    target_size: usize,
    /// Incremental resize chunk size
    chunk_size: usize,
    /// Use copy-on-write strategy
    use_cow: bool,
}

impl CacheConsciousResizer {
    /// Create a new resizer
    pub fn new(current_size: usize, target_size: usize) -> Self {
        // Calculate chunk size based on L3 cache size
        let chunk_size = (L3_CACHE_SIZE / CACHE_LINE_SIZE).min(target_size / 16);
        
        Self {
            current_size,
            target_size,
            chunk_size,
            use_cow: target_size > L3_CACHE_SIZE,
        }
    }

    /// Perform incremental resize step
    pub fn resize_step<T, F>(&mut self, data: &mut Vec<T>, rehash_fn: F) -> bool
    where
        T: Clone,
        F: Fn(&T, usize) -> usize,
    {
        if self.current_size >= self.target_size {
            return true; // Resize complete
        }

        let step_end = (self.current_size + self.chunk_size).min(self.target_size);
        
        // Resize incrementally to avoid cache thrashing
        data.reserve(step_end - self.current_size);
        
        // Update current size
        self.current_size = step_end;
        
        false // Resize not complete
    }
}

/// Memory access pattern analyzer for optimization
pub struct AccessPatternAnalyzer {
    /// Access history ring buffer
    access_history: Vec<usize>,
    /// Current position in ring buffer
    history_pos: usize,
    /// Detected pattern
    detected_pattern: AccessPattern,
    /// Pattern confidence score
    confidence: f64,
}

impl AccessPatternAnalyzer {
    /// Create a new analyzer
    pub fn new(history_size: usize) -> Self {
        Self {
            access_history: vec![0; history_size],
            history_pos: 0,
            detected_pattern: AccessPattern::Random,
            confidence: 0.0,
        }
    }

    /// Record an access
    pub fn record_access(&mut self, address: usize) {
        self.access_history[self.history_pos] = address;
        self.history_pos = (self.history_pos + 1) % self.access_history.len();
        
        // Analyze pattern periodically
        if self.history_pos == 0 {
            self.analyze_pattern();
        }
    }

    /// Analyze access pattern
    fn analyze_pattern(&mut self) {
        let mut sequential_count = 0;
        let mut stride_counts = std::collections::HashMap::new();
        
        for i in 1..self.access_history.len() {
            let diff = self.access_history[i].wrapping_sub(self.access_history[i - 1]);
            
            if diff == 1 {
                sequential_count += 1;
            } else if diff > 0 && diff < 1024 {
                *stride_counts.entry(diff).or_insert(0) += 1;
            }
        }
        
        let total = self.access_history.len() as f64;
        
        if sequential_count as f64 / total > 0.7 {
            self.detected_pattern = AccessPattern::Sequential;
            self.confidence = sequential_count as f64 / total;
        } else if let Some((&stride, &count)) = stride_counts.iter().max_by_key(|&(_, &c)| c) {
            if count as f64 / total > 0.5 {
                self.detected_pattern = AccessPattern::Strided(stride);
                self.confidence = count as f64 / total;
            }
        } else {
            self.detected_pattern = AccessPattern::Random;
            self.confidence = 1.0 - (sequential_count as f64 / total);
        }
    }

    /// Get detected pattern
    pub fn get_pattern(&self) -> (AccessPattern, f64) {
        (self.detected_pattern, self.confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned() {
        let aligned: CacheAligned<u64> = CacheAligned::new(42);
        assert_eq!(*aligned.get(), 42);
        
        // Check alignment
        let ptr = aligned.get() as *const _ as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_cache_optimized_bucket() {
        let mut bucket: CacheOptimizedBucket<u32, u64, 7> = CacheOptimizedBucket::new();
        assert!(bucket.is_empty());
        assert_eq!(bucket.count(), 0);
        
        // Test find_hash
        bucket.metadata.occupancy = 0b0000011; // First two slots occupied
        bucket.hashes[0] = 12345;
        bucket.hashes[1] = 67890;
        
        assert_eq!(bucket.find_hash(12345), Some(0));
        assert_eq!(bucket.find_hash(67890), Some(1));
        assert_eq!(bucket.find_hash(99999), None);
    }

    #[test]
    fn test_cache_layout_optimizer() {
        // Test L1 optimization
        let optimizer: CacheLayoutOptimizer<u64, u64> = CacheLayoutOptimizer::new(16 * 1024);
        assert!(matches!(optimizer.target_cache_level, CacheLevel::L1));
        assert!(optimizer.optimal_load_factor() < 0.6);
        
        // Test L3 optimization
        let optimizer: CacheLayoutOptimizer<u64, u64> = CacheLayoutOptimizer::new(4 * 1024 * 1024);
        assert!(matches!(optimizer.target_cache_level, CacheLevel::L3));
        assert!(optimizer.optimal_load_factor() > 0.7);
    }

    #[test]
    fn test_hot_cold_separator() {
        let mut separator: HotColdSeparator<u32> = HotColdSeparator::new(100, 0.2);
        
        // Add some data
        for i in 0..20 {
            separator.hot.push(CacheAligned::new(i));
            separator.access_counts.push(AtomicU64::new(0));
        }
        for i in 20..100 {
            separator.cold.push(i);
            separator.access_counts.push(AtomicU64::new(0));
        }
        
        // Simulate access pattern
        for _ in 0..50 {
            separator.access(5); // Hot access
        }
        for _ in 0..5 {
            separator.access(50); // Cold access
        }
        
        assert_eq!(separator.access_counts[5].load(Ordering::Relaxed), 50);
        assert_eq!(separator.access_counts[50].load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_access_pattern_analyzer() {
        let mut analyzer = AccessPatternAnalyzer::new(100);
        
        // Generate sequential pattern
        for i in 0..100 {
            analyzer.record_access(1000 + i);
        }
        
        let (pattern, confidence) = analyzer.get_pattern();
        assert!(matches!(pattern, AccessPattern::Sequential));
        assert!(confidence > 0.6);
    }

    #[test]
    fn test_cache_metrics() {
        let mut metrics = CacheMetrics::default();
        metrics.l1_hits = 1000;
        metrics.l1_misses = 100;
        metrics.l2_hits = 50;
        metrics.l2_misses = 10;
        
        let hit_ratio = metrics.hit_ratio();
        assert!(hit_ratio > 0.9);
        
        let bandwidth = metrics.estimated_bandwidth_gb();
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_prefetcher() {
        let data = vec![1u64, 2, 3, 4, 5];
        unsafe {
            // Test basic prefetch
            Prefetcher::prefetch(data.as_ptr(), PrefetchHint::AllLevels);
            
            // Test range prefetch
            Prefetcher::prefetch_range(data.as_ptr(), 3, PrefetchHint::L2L3);
            
            // Test strided prefetch
            Prefetcher::prefetch_strided(data.as_ptr(), 2, 2, PrefetchHint::L3Only);
        }
    }

    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAllocator::new();
        
        unsafe {
            let ptr = allocator.alloc_cache_aligned::<u64>(100);
            assert!(!ptr.is_null());
            
            // Check alignment
            assert_eq!(ptr as usize % CACHE_LINE_SIZE, 0);
            
            // Clean up
            allocator.dealloc(ptr, 100);
        }
    }

    #[test]
    fn test_cache_conscious_resizer() {
        let mut resizer = CacheConsciousResizer::new(1000, 10000);
        
        assert!(resizer.chunk_size > 0);
        assert_eq!(resizer.current_size, 1000);
        assert_eq!(resizer.target_size, 10000);
        
        let mut data = vec![0u64; 1000];
        let complete = resizer.resize_step(&mut data, |&x, _| x as usize);
        assert!(!complete); // Should not be complete in one step
    }
}