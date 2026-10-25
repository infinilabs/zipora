//! Cache Locality Optimizations for Hash Maps
//!
//! This module provides sophisticated cache locality optimizations for hash table implementations,
//! featuring memory layout optimizations, prefetching strategies, NUMA awareness, and cache
//! performance monitoring. Inspired by state-of-the-art research in cache-conscious data structures.

use std::alloc::{Layout, alloc_zeroed, dealloc};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_MM_HINT_NTA, _MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _mm_prefetch};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache line size in bytes (typical for modern x86_64 processors)
pub const CACHE_LINE_SIZE: usize = 64;

/// L1 cache size hint (typical: 32KB)
pub const L1_CACHE_SIZE: usize = 32 * 1024;

/// L2 cache size hint (typical: 256KB)
pub const L2_CACHE_SIZE: usize = 256 * 1024;

/// L3 cache size hint (typical: 8MB)
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024;

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
    /// Create new cache metrics
    pub fn new() -> Self {
        Self::default()
    }

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
#[derive(Clone, Debug)]
pub struct CacheAligned<T> {
    data: T,
}

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get a reference to the inner data
    #[inline]
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
#[derive(Default)]
pub struct Prefetcher;

impl Prefetcher {
    /// Create new prefetcher
    pub fn new() -> Self {
        Prefetcher
    }

    /// Prefetch a memory location with specified hint
    ///
    /// # Safety
    /// `ptr` is used only as a prefetch hint and is never dereferenced, so any
    /// address (even invalid) is sound; the CPU ignores faulting hints.
    #[inline(always)]
    pub unsafe fn prefetch<T>(ptr: *const T, hint: PrefetchHint) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: _mm_prefetch is safe to call with any pointer, even if invalid
        // it only hints the CPU and does not dereference memory
        unsafe {
            match hint {
                PrefetchHint::AllLevels => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
                PrefetchHint::L2L3 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                PrefetchHint::L3Only => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                PrefetchHint::NonTemporal => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (ptr, hint);
        }
    }

    /// Prefetch multiple sequential cache lines
    ///
    /// # Safety
    /// The caller must ensure `start` is valid and the range
    /// `start .. start + count * size_of::<T>()` stays within one allocation.
    #[inline(always)]
    pub unsafe fn prefetch_range<T>(start: *const T, count: usize, hint: PrefetchHint) {
        // SAFETY: Caller must ensure start is valid and start..start+count*sizeof(T) is in bounds
        // Pointer arithmetic is safe because we iterate count times adding bytes_per_element
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
    ///
    /// # Safety
    /// The caller must ensure `base` and every `base + i * stride` for
    /// `i in 0..count` are valid addresses.
    #[inline(always)]
    pub unsafe fn prefetch_strided<T>(
        base: *const T,
        stride: usize,
        count: usize,
        hint: PrefetchHint,
    ) {
        // SAFETY: Caller must ensure base and all base+i*stride for i in 0..count are valid pointers
        unsafe {
            let mut current = base;
            for _ in 0..count {
                Self::prefetch(current, hint);
                current = current.add(stride);
            }
        }
    }
}

/// NUMA-aware memory allocator for hash tables
///
/// Allocations are not bound to a NUMA node: `alloc_on_node` uses the global
/// allocator (`alloc_zeroed`) whatever `node` is, and `node` only selects the
/// per-node statistics counter. Node detection reports a single node.
pub struct NumaAllocator {
    /// Preferred NUMA node
    preferred_node: usize,
    /// Allocation statistics per node
    node_stats: Vec<AtomicU64>,
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
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
    ///
    /// A zero-size `layout` allocates nothing and returns a dangling pointer
    /// aligned to `layout.align()`.
    ///
    /// # Safety
    /// `layout` must have valid alignment; if `layout.size() > 0`, the returned
    /// pointer must later be freed with the same layout.
    pub unsafe fn alloc_on_node(&self, layout: Layout, node: usize) -> *mut u8 {
        if layout.size() == 0 {
            return std::ptr::without_provenance_mut(layout.align());
        }
        // SAFETY: `layout.size() > 0` is verified above and `Layout` guarantees valid alignment.
        unsafe {
            // In a real implementation, this would use numa_alloc_onnode
            // For now, use standard aligned allocation
            let ptr = alloc_zeroed(layout);
            if !ptr.is_null() && !self.node_stats.is_empty() {
                self.node_stats[node.min(self.node_stats.len() - 1)]
                    .fetch_add(layout.size() as u64, Ordering::Relaxed);
            }
            ptr
        }
    }

    /// Allocate cache-line aligned memory
    ///
    /// Returns a null pointer if the allocation fails or if
    /// `size_of::<T>() * count` overflows or is too large for a [`Layout`].
    /// Both used to be caller preconditions: the multiplication panicked with
    /// overflow checks and wrapped to a short allocation without them, and an
    /// oversized `Layout` hit an `expect`. A zero-byte request (`count == 0`
    /// or a zero-sized `T`) allocates nothing and returns a dangling pointer
    /// aligned to `max(64, align_of::<T>())`.
    ///
    /// # Safety
    /// The caller must free a non-null result via [`Self::dealloc`] with the
    /// same `count`.
    pub unsafe fn alloc_cache_aligned<T>(&self, count: usize) -> *mut T {
        let align = CACHE_LINE_SIZE.max(align_of::<T>());
        let Some(total_bytes) = size_of::<T>().checked_mul(count) else {
            return std::ptr::null_mut();
        };
        if total_bytes == 0 {
            return std::ptr::without_provenance_mut(align);
        }
        let Ok(layout) = Layout::from_size_align(total_bytes, align) else {
            return std::ptr::null_mut();
        };
        // SAFETY: `layout` has `total_bytes > 0` and valid power-of-two alignment.
        unsafe { self.alloc_on_node(layout, self.preferred_node) as *mut T }
    }

    /// Free previously allocated memory
    ///
    /// # Safety
    /// `ptr` must originate from [`Self::alloc_cache_aligned`] with the same
    /// `count`, and must not be used after this call.
    pub unsafe fn dealloc<T>(&self, ptr: *mut T, count: usize) {
        if ptr.is_null() {
            return;
        }
        let Some(total_bytes) = size_of::<T>().checked_mul(count) else {
            return;
        };
        if total_bytes == 0 {
            return;
        }
        let align = CACHE_LINE_SIZE.max(align_of::<T>());
        let Ok(layout) = Layout::from_size_align(total_bytes, align) else {
            return;
        };
        // SAFETY: Caller guarantees `ptr` was allocated with `alloc_cache_aligned::<T>(count)`
        // and `total_bytes > 0` matches the allocation layout.
        unsafe {
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
        // An entry larger than a cache line used to give a capacity of 0, so
        // every level returned a bucket size of 0.
        let cache_line_capacity = (CACHE_LINE_SIZE / entry_size).max(1);

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
            CacheLevel::L1 => 0.5,      // Lower load factor for L1 efficiency
            CacheLevel::L2 => 0.65,     // Moderate load factor
            CacheLevel::L3 => 0.75,     // Standard load factor
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
///
/// Elements are addressed by a global index: hot elements first, then cold
/// ones. No public method inserts elements, so a separator built with
/// [`HotColdSeparator::new`] stays empty: `access` returns `None` and
/// `rebalance` has nothing to move.
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
    ///
    /// `hot_ratio` is clamped to `[0.0, 1.0]` (NaN gives no hot capacity). A
    /// ratio above 1.0 used to underflow `capacity - hot_capacity`.
    pub fn new(capacity: usize, hot_ratio: f64) -> Self {
        let hot_capacity = ((capacity as f64) * hot_ratio.clamp(0.0, 1.0)) as usize;
        // `capacity as f64` can round up, so the product can exceed `capacity`
        let cold_capacity = capacity.saturating_sub(hot_capacity);

        Self {
            hot: Vec::with_capacity(hot_capacity),
            cold: Vec::with_capacity(cold_capacity),
            access_counts: Vec::with_capacity(capacity),
            migration_threshold: 10, // Migrate after 10 accesses
        }
    }

    /// Access an element and track access pattern
    ///
    /// Returns `None` if `index` is out of range.
    pub fn access(&self, index: usize) -> Option<&T> {
        let elem = if index < self.hot.len() {
            Some(self.hot[index].get())
        } else {
            let cold_index = index - self.hot.len();
            self.cold.get(cold_index)
        }?;

        if let Some(counter) = self.access_counts.get(index) {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        Some(elem)
    }

    /// Migrate data between hot and cold based on access patterns
    ///
    /// Each cold element accessed at least `migration_threshold` (10) times
    /// since the last rebalance swaps places with the least-accessed hot
    /// element, if that element has strictly fewer accesses. Swapped elements
    /// change their global index. All access counts are then reset.
    pub fn rebalance(&mut self) {
        let hot_len = self.hot.len();
        let cold_len = self.cold.len();
        if hot_len > 0 && cold_len > 0 && self.access_counts.len() >= hot_len + cold_len {
            for cold_idx in 0..cold_len {
                let global_cold_idx = hot_len + cold_idx;
                let cold_accesses = self.access_counts[global_cold_idx].load(Ordering::Relaxed);
                if cold_accesses >= self.migration_threshold {
                    // Find coldest slot in hot tier with strictly fewer accesses
                    if let Some((coldest_hot_idx, min_hot_accesses)) = (0..hot_len)
                        .map(|i| (i, self.access_counts[i].load(Ordering::Relaxed)))
                        .min_by_key(|&(_, count)| count)
                        && min_hot_accesses < cold_accesses
                    {
                        std::mem::swap(
                            self.hot[coldest_hot_idx].get_mut(),
                            &mut self.cold[cold_idx],
                        );
                        self.access_counts[coldest_hot_idx].store(cold_accesses, Ordering::Relaxed);
                        self.access_counts[global_cold_idx]
                            .store(min_hot_accesses, Ordering::Relaxed);
                    }
                }
            }
        }

        // Reset access counts for the next epoch
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
    _use_cow: bool,
}

impl CacheConsciousResizer {
    /// Create a new resizer
    pub fn new(current_size: usize, target_size: usize) -> Self {
        // Calculate chunk size based on L3 cache size (at least 1 to guarantee forward progress)
        let chunk_size = (L3_CACHE_SIZE / CACHE_LINE_SIZE).min((target_size / 16).max(1));

        Self {
            current_size,
            target_size,
            chunk_size,
            _use_cow: target_size > L3_CACHE_SIZE,
        }
    }

    /// Perform incremental resize step
    ///
    /// Reserves room in `data` for the next chunk (at most `chunk_size` more
    /// elements beyond `data.len()`) and advances the internal size toward
    /// `target_size`. `_rehash_fn` is not called: moving entries is left to
    /// the caller. Returns `true` when called after the resize has completed,
    /// and `false` after a step that advanced it.
    pub fn resize_step<T, F>(&mut self, data: &mut Vec<T>, _rehash_fn: F) -> bool
    where
        T: Clone,
        F: Fn(&T, usize) -> usize,
    {
        if self.current_size >= self.target_size {
            return true; // Resize complete
        }

        let step_end = self
            .current_size
            .saturating_add(self.chunk_size)
            .min(self.target_size);

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
    ///
    /// A `history_size` of 0 is treated as 1 (an empty history made
    /// `record_access` index an empty buffer).
    pub fn new(history_size: usize) -> Self {
        let effective_size = history_size.max(1);
        Self {
            access_history: vec![0; effective_size],
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
        let metrics = CacheMetrics {
            l1_hits: 1000,
            l1_misses: 100,
            l2_hits: 50,
            l2_misses: 10,
            ..Default::default()
        };

        let hit_ratio = metrics.hit_ratio();
        assert!(hit_ratio > 0.9);

        let bandwidth = metrics.estimated_bandwidth_gb();
        assert!(bandwidth > 0.0);
    }

    #[test]
    fn test_prefetcher() {
        let data = [1u64, 2, 3, 4, 5];
        // SAFETY: data.as_ptr() is valid, ranges are within allocated Vec bounds
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

        // SAFETY: Valid allocation count, proper deallocation with same count
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

    /// A separator holding hot element 10 (index 0) and cold element 20 (index 1).
    fn separator_with_one_hot_and_one_cold() -> HotColdSeparator<u32> {
        let mut separator: HotColdSeparator<u32> = HotColdSeparator::new(4, 0.5);
        separator.hot.push(CacheAligned::new(10));
        separator.access_counts.push(AtomicU64::new(0));
        separator.cold.push(20);
        separator.access_counts.push(AtomicU64::new(0));
        separator
    }

    #[test]
    fn test_c7_5_hot_cold_access_out_of_range_is_none() {
        // `access` indexed `access_counts` before any bounds check.
        let empty: HotColdSeparator<u32> = HotColdSeparator::new(4, 0.5);
        assert_eq!(empty.access(0), None);
        assert_eq!(empty.access(999), None);

        let separator = separator_with_one_hot_and_one_cold();
        assert_eq!(separator.access(2), None);
        assert_eq!(separator.access(usize::MAX), None);
    }

    #[test]
    fn test_c7_5_hot_cold_rebalance_promotes_a_frequently_accessed_cold_element() {
        // `rebalance` only reset the access counts; nothing ever migrated.
        let mut separator = separator_with_one_hot_and_one_cold();
        for _ in 0..15 {
            assert_eq!(separator.access(1), Some(&20));
        }
        separator.rebalance();
        assert_eq!(
            separator.access(0),
            Some(&20),
            "rebalance must promote frequently accessed cold item into hot tier"
        );
        assert_eq!(separator.access(1), Some(&10));

        // Below the threshold (10 accesses) nothing moves.
        let mut separator = separator_with_one_hot_and_one_cold();
        for _ in 0..9 {
            separator.access(1);
        }
        separator.rebalance();
        assert_eq!(separator.access(0), Some(&10));
    }

    #[test]
    fn test_c7_5_hot_cold_ratio_outside_unit_interval_is_clamped() {
        // hot_ratio > 1.0 underflowed `capacity - hot_capacity`.
        for ratio in [1.5, f64::INFINITY, -0.5, f64::NEG_INFINITY, f64::NAN] {
            let separator: HotColdSeparator<u32> = HotColdSeparator::new(4, ratio);
            assert_eq!(separator.access(0), None, "ratio {ratio}");
        }
    }

    #[test]
    fn test_c7_5_optimal_bucket_size_is_at_least_one_for_entries_over_a_cache_line() {
        let big_opt: CacheLayoutOptimizer<[u8; 128], [u8; 128]> = CacheLayoutOptimizer::new(1024);
        assert_eq!(big_opt.optimal_bucket_size(), 1);
    }

    #[test]
    fn test_c7_5_resizer_with_target_below_16_makes_progress() {
        // `target_size / 16 == 0` made the chunk size 0, so no step advanced.
        let mut small_resizer = CacheConsciousResizer::new(4, 8);
        let mut small_data = vec![0u32; 4];
        let mut steps = 0;
        while !small_resizer.resize_step(&mut small_data, |&x, _| x as usize) {
            steps += 1;
            assert!(steps <= 8, "CacheConsciousResizer failed to make progress");
        }
    }

    #[test]
    fn test_c7_5_access_pattern_analyzer_with_zero_history_records_accesses() {
        // A zero-length history made record_access index an empty buffer.
        let mut zero_analyzer = AccessPatternAnalyzer::new(0);
        zero_analyzer.record_access(0x1000);
        zero_analyzer.record_access(0x1001);
    }

    #[test]
    fn test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator() {
        // A zero-size Layout reached `alloc_zeroed`, which is UB (Miri reports
        // it; natively the system allocator happens to return a pointer).
        let allocator = NumaAllocator::new();
        // SAFETY: every pointer is freed with the count it was allocated with.
        unsafe {
            let zero_ptr = allocator.alloc_cache_aligned::<u64>(0);
            assert!(!zero_ptr.is_null());
            assert_eq!(zero_ptr as usize % CACHE_LINE_SIZE, 0);
            allocator.dealloc(zero_ptr, 0);

            let zst_ptr = allocator.alloc_cache_aligned::<()>(32);
            assert!(!zst_ptr.is_null());
            assert_eq!(zst_ptr as usize % CACHE_LINE_SIZE, 0);
            allocator.dealloc(zst_ptr, 32);
        }
    }

    #[test]
    fn test_c7_5_numa_allocator_overflowing_request_returns_null() {
        // `size_of::<T>() * count` panicked with overflow checks (and wrapped
        // without them); an oversized Layout hit an `expect`.
        let allocator = NumaAllocator::new();
        // SAFETY: both requests fail and return null; there is nothing to free.
        unsafe {
            assert!(allocator.alloc_cache_aligned::<u64>(usize::MAX).is_null());
            assert!(allocator.alloc_cache_aligned::<u8>(usize::MAX).is_null());
        }
    }
}
