//! Cache-Oblivious Sorting
//!
//! [`CacheObliviousSort`] picks a strategy from the input length and the
//! detected cache sizes, then sorts in place, sequentially.
//!
//! ## Key Features
//!
//! - **Recursive multiway merge sort** (the `CacheOblivious` strategy):
//!   inputs of at most `small_threshold` elements (1024 by default) are
//!   sorted with `slice::sort`; larger inputs are split into `k` parts that
//!   are sorted recursively and then merged through a binary heap. At the
//!   top level `k` is √(L2 size / L2 line size) clamped to 2..=64, one level
//!   down it is ⌊√k⌋ (at least 2), and below that it is 2. This is not
//!   funnelsort.
//! - **Adaptive Algorithm Selection**: Choose between cache-aware and cache-oblivious
//! - **Van Emde Boas Layout**: Limitation: [`VanEmdeBoas`] does not reorder
//!   its data (see its docs).
//!
//! ## Performance Characteristics
//!
//! - **Cache complexity**: the O(1 + N/B * log_{M/B}(N/B)) bound of
//!   funnelsort is not claimed: below the top two levels every merge is
//!   2-way.
//! - **Parallelism**: none; `use_parallel` has no effect.
//! - **SIMD**: none. The only intrinsics are `_mm_prefetch` hints, issued on
//!   x86-64 hosts with AVX2.
//!
//! ## Algorithm Selection Strategy
//!
//! The strategy is chosen from `len * 8` bytes (8 bytes per element,
//! whatever `T` is); the cache-aware paths then use the real size of the
//! data:
//!
//! - **Up to L1**: cache-aware sort. For elements of at most 8 bytes that is
//!   insertion sort; larger elements can take the L2 path (quicksort) or the
//!   L3 path (merge sort).
//! - **L1 to L3**: the recursive multiway merge sort.
//! - **Above L3**: the cache-aware sort if the real size fits in L2,
//!   otherwise the recursive multiway merge sort. Nothing is sorted on disk.
//! - There are no type-specific paths: strings and numbers go through the
//!   same generic code.

use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::Result;
use crate::memory::SecureMemoryPool;
use crate::memory::cache_layout::{CacheHierarchy, detect_cache_hierarchy};
use crate::system::cpu_features::{CpuFeatures, get_cpu_features};
use std::cmp;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

/// Configuration for cache-oblivious algorithms
#[derive(Debug, Clone)]
pub struct CacheObliviousConfig {
    /// Cache hierarchy information for adaptive selection
    pub cache_hierarchy: CacheHierarchy,
    /// For inputs that fit in L1 (with AVX2 and at least 16 elements, or
    /// SSE4.2 and at least 8), run the insertion sort variant that adds
    /// prefetch hints on AVX2 hosts. No SIMD instructions are used either way.
    pub use_simd: bool,
    /// Limitation: has no effect; the sort is sequential.
    pub use_parallel: bool,
    /// Threshold for switching between algorithms
    pub small_threshold: usize,
    /// Limitation: has no effect; buffers come from the global allocator.
    pub memory_pool: Option<Arc<SecureMemoryPool>>,
    /// CPU features for optimization
    pub cpu_features: &'static CpuFeatures,
}

impl Default for CacheObliviousConfig {
    fn default() -> Self {
        Self {
            cache_hierarchy: detect_cache_hierarchy(),
            use_simd: cfg!(feature = "simd"),
            use_parallel: true,
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: get_cpu_features(),
        }
    }
}

/// Cache-oblivious sorting algorithm with adaptive strategy selection
pub struct CacheObliviousSort {
    config: CacheObliviousConfig,
    stats: AlgorithmStats,
}

impl CacheObliviousSort {
    /// Create a new cache-oblivious sort instance
    pub fn new() -> Self {
        Self::with_config(CacheObliviousConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CacheObliviousConfig) -> Self {
        Self {
            config,
            stats: AlgorithmStats {
                items_processed: 0,
                processing_time_us: 0,
                memory_used: 0,
                used_parallel: false,
                used_simd: false,
            },
        }
    }

    /// Sort `data` in place with the strategy that [`AdaptiveAlgorithmSelector`]
    /// picks (see the module docs)
    pub fn sort<T: Clone + Ord>(&mut self, data: &mut [T]) -> Result<()> {
        let start_time = Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        // Adaptive algorithm selection based on data size and cache hierarchy
        let selector = AdaptiveAlgorithmSelector::new(&self.config);
        let strategy = selector.select_strategy(data.len(), &self.config.cache_hierarchy);

        match strategy {
            CacheObliviousSortingStrategy::CacheOblivious => {
                self.cache_oblivious_sort(data)?;
            }
            CacheObliviousSortingStrategy::CacheAware => {
                self.cache_aware_sort(data)?;
            }
            CacheObliviousSortingStrategy::Hybrid => {
                self.hybrid_sort(data)?;
            }
        }

        // Update statistics
        self.stats.items_processed = data.len();
        self.stats.processing_time_us = start_time.elapsed().as_micros() as u64;
        // The sort is sequential and uses no SIMD instructions (only
        // prefetch hints), whatever `use_parallel` and `use_simd` say.
        self.stats.used_simd = false;
        self.stats.used_parallel = false;

        Ok(())
    }

    /// Recursive multiway merge sort (the `CacheOblivious` strategy; see the
    /// module docs). This is not funnelsort.
    pub fn cache_oblivious_sort<T: Clone + Ord>(&mut self, data: &mut [T]) -> Result<()> {
        if data.len() <= self.config.small_threshold {
            Self::leaf_sort(data);
            return Ok(());
        }

        // Cache-oblivious recursive subdivision
        let k = self.calculate_funnel_width(data.len());
        self.funnel_sort_recursive(data, k)?;

        Ok(())
    }

    /// Inputs of at most `small_threshold` elements go to `leaf_sort`.
    /// Larger ones are split into `k` parts (`k` clamped to 2..=n), each
    /// sorted recursively with ⌊√k⌋ parts (at least 2), and the parts are
    /// merged with `cache_oblivious_merge`.
    fn funnel_sort_recursive<T: Clone + Ord>(&mut self, data: &mut [T], k: usize) -> Result<()> {
        let n = data.len();

        if n <= 1 || n <= self.config.small_threshold {
            Self::leaf_sort(data);
            return Ok(());
        }

        // Clamp k to [2, n] and sqrt_k to >= 2 so recursive subproblems strictly shrink
        let k = k.max(2).min(n);
        let sqrt_k = ((k as f64).sqrt() as usize).max(2);

        // Recursively sort k sublists using balanced (i * n) / k .. ((i + 1) * n) / k boundaries
        for i in 0..k {
            let start = (i * n) / k;
            let end = ((i + 1) * n) / k;

            if start < end {
                self.funnel_sort_recursive(&mut data[start..end], sqrt_k)?;
            }
        }

        // Cache-oblivious k-way merge using funnel
        self.cache_oblivious_merge(data, k)?;

        Ok(())
    }

    /// Merges the `k` sorted parts of `data` through a binary heap into a
    /// temporary buffer, then copies it back. On AVX2 hosts with more than 64
    /// elements it first issues a prefetch hint for the start of each part.
    fn cache_oblivious_merge<T: Clone + Ord>(&mut self, data: &mut [T], k: usize) -> Result<()> {
        let n = data.len();
        if k <= 1 || n <= 1 {
            return Ok(());
        }

        let mut segments = Vec::with_capacity(k);

        // Prefetch initial data for all segments
        if self.config.cpu_features.has_avx2 && n > 64 {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            {
                for i in 0..k {
                    let start = (i * n) / k;
                    if start < n {
                        // SAFETY: start < n ensures valid pointer within slice bounds
                        unsafe {
                            let ptr = data.as_ptr().add(start) as *const i8;
                            std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
                        }
                    }
                }
            }
        }

        // Set up merge segments: (current_pos, end) matching funnel_sort_recursive boundaries
        for i in 0..k {
            let start = (i * n) / k;
            let end = ((i + 1) * n) / k;

            if start < end {
                segments.push((start, end));
            }
        }

        // K-way merge via a min-heap keyed on (head value, segment index):
        // O(N log k) total, instead of scanning all k segments (and shifting
        // the segment vector on exhaustion) for every emitted element.
        // Ties break on the lower segment index, preserving input order for
        // equal elements.
        let mut temp = Vec::with_capacity(n);
        let mut heap: BinaryHeap<Reverse<(T, usize)>> = BinaryHeap::with_capacity(segments.len());
        for (i, &(pos, _end)) in segments.iter().enumerate() {
            heap.push(Reverse((data[pos].clone(), i)));
        }

        while let Some(Reverse((value, seg))) = heap.pop() {
            temp.push(value);
            let (pos, end) = &mut segments[seg];
            *pos += 1;
            if *pos < *end {
                heap.push(Reverse((data[*pos].clone(), seg)));
            }
        }

        // Copy back to original array with cache-optimized transfer
        self.cache_optimized_copy(&temp, data);

        Ok(())
    }

    /// Copies `src` into `dst` with `clone_from_slice`. On AVX2 hosts with
    /// more than 256 elements it copies 64 elements at a time and issues
    /// prefetch hints ahead of the copy.
    fn cache_optimized_copy<T: Clone>(&self, src: &[T], dst: &mut [T]) {
        if src.len() != dst.len() {
            dst.clone_from_slice(src);
            return;
        }

        // For large arrays, use cache-aware copying with prefetch
        if src.len() > 256 && self.config.cpu_features.has_avx2 {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            {
                let chunk_size = 64; // Process in cache-line sized chunks
                for i in (0..src.len()).step_by(chunk_size) {
                    let end = std::cmp::min(i + chunk_size, src.len());

                    // Prefetch next chunk
                    if end + chunk_size < src.len() {
                        // SAFETY: end + chunk_size < src.len() ensures valid pointers within slice bounds
                        unsafe {
                            let src_ptr = src.as_ptr().add(end + chunk_size) as *const i8;
                            let dst_ptr = dst.as_mut_ptr().add(end + chunk_size) as *const i8;
                            std::arch::x86_64::_mm_prefetch(
                                src_ptr,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                            std::arch::x86_64::_mm_prefetch(
                                dst_ptr,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }
                    }

                    // Copy current chunk
                    dst[i..end].clone_from_slice(&src[i..end]);
                }
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            {
                dst.clone_from_slice(src);
            }
        } else {
            // Regular copy for small arrays or when SIMD unavailable
            dst.clone_from_slice(src);
        }
    }

    /// Cache-aware sorting for small data that fits in cache
    fn cache_aware_sort<T: Clone + Ord>(&mut self, data: &mut [T]) -> Result<()> {
        // Use optimized algorithms for data that fits in specific cache levels
        if std::mem::size_of_val(data) <= self.config.cache_hierarchy.l1_size {
            self.l1_optimized_sort(data);
        } else if std::mem::size_of_val(data) <= self.config.cache_hierarchy.l2_size {
            self.l2_optimized_sort(data);
        } else {
            self.l3_optimized_sort(data);
        }
        Ok(())
    }

    /// Hybrid approach combining cache-aware and cache-oblivious
    fn hybrid_sort<T: Clone + Ord>(&mut self, data: &mut [T]) -> Result<()> {
        let data_size = std::mem::size_of_val(data);

        if data_size <= self.config.cache_hierarchy.l2_size {
            // Use cache-aware for data that fits in L2
            self.cache_aware_sort(data)
        } else {
            // Use cache-oblivious for larger data
            self.cache_oblivious_sort(data)
        }
    }

    /// Calculate optimal funnel width based on data size
    fn calculate_funnel_width(&self, n: usize) -> usize {
        // Optimal funnel width for cache-oblivious sorting
        // k = Θ(√(M/B)) where M is cache size and B is block size
        let cache_size = self.config.cache_hierarchy.l2_size;
        let block_size = self.config.cache_hierarchy.l2_line_size.max(1);

        let k = ((cache_size / block_size) as f64).sqrt() as usize;
        cmp::max(k, 2).min(cmp::min(n.max(2), 64)) // Bounded between 2 and 64
    }

    /// Base case of the recursive merge sort: inputs of at most `small_threshold`
    /// elements (1024 by default).
    ///
    /// Until S20-R4 this was the O(n^2) `insertion_sort`, and with the default
    /// threshold its inner loop dominated: replacing it halves the time of a
    /// 50k-element sort, and a shift in that loop's code alignment alone (no
    /// instruction changed) had made the whole sort about 50% slower. The std
    /// sort is stable like `insertion_sort`, so the output is unchanged, and it
    /// is O(n log n).
    fn leaf_sort<T: Ord>(data: &mut [T]) {
        data.sort();
    }

    /// Optimized insertion sort for small arrays
    fn insertion_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        for i in 1..data.len() {
            let key = data[i].clone();
            let mut j = i;

            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1].clone();
                j -= 1;
            }

            data[j] = key;
        }
    }

    /// Insertion sort for inputs that fit in L1. With `use_simd`, and AVX2 with
    /// at least 16 elements or SSE4.2 with at least 8, it runs
    /// `simd_insertion_sort`, which adds prefetch hints on AVX2 hosts. No SIMD
    /// instructions either way.
    fn l1_optimized_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        // Use insertion sort with prefetching for L1 cache
        if self.config.use_simd
            && ((self.config.cpu_features.has_avx2 && data.len() >= 16)
                || (self.config.cpu_features.has_sse42 && data.len() >= 8))
        {
            self.simd_insertion_sort(data);
        } else {
            self.insertion_sort(data);
        }
    }

    /// L2 cache-optimized sorting  
    fn l2_optimized_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        // Use quicksort with cache-aware partitioning
        if data.len() > 16 {
            self.cache_aware_quicksort(data, 0, data.len() - 1);
        } else {
            self.insertion_sort(data);
        }
    }

    /// L3 cache-optimized sorting
    fn l3_optimized_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        // Use merge sort with cache-aware block sizes
        if data.len() > 32 {
            self.cache_aware_mergesort(data);
        } else {
            self.insertion_sort(data);
        }
    }

    /// Insertion sort with prefetch hints. Despite the name it uses no SIMD
    /// instructions: on AVX2 hosts with more than 64 elements it issues a
    /// prefetch hint for every 8th element, then runs
    /// `cache_aware_insertion_sort`.
    fn simd_insertion_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        // Prefetch data for cache optimization
        if data.len() > 64 {
            // Prefetch cache lines ahead for better memory access patterns
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            {
                if self.config.cpu_features.has_avx2 {
                    // SAFETY: i < data.len() by step_by(8) ensures valid pointers within slice bounds
                    unsafe {
                        for i in (0..data.len()).step_by(8) {
                            let ptr = data.as_ptr().add(i) as *const i8;
                            std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
                        }
                    }
                }
            }
        }

        // Use cache-aware insertion sort with optimized memory access patterns
        self.cache_aware_insertion_sort(data);
    }

    /// Cache-aware insertion sort with prefetch optimization
    fn cache_aware_insertion_sort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        for i in 1..data.len() {
            let key = data[i].clone();
            let mut j = i;

            // Prefetch next cache line for better memory access
            if i + 8 < data.len() && self.config.cpu_features.has_avx2 {
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                // SAFETY: i + 8 < data.len() ensures valid pointer within slice bounds
                unsafe {
                    let ptr = data.as_ptr().add(i + 8) as *const i8;
                    std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T1);
                }
            }

            while j > 0 && data[j - 1] > key {
                data[j] = data[j - 1].clone();
                j -= 1;
            }

            data[j] = key;
        }
    }

    /// Cache-aware quicksort with optimal partitioning
    fn cache_aware_quicksort<T: Clone + Ord>(&mut self, data: &mut [T], low: usize, high: usize) {
        if low < high {
            let pivot = self.cache_aware_partition(data, low, high);

            if pivot > 0 {
                self.cache_aware_quicksort(data, low, pivot - 1);
            }
            if pivot < high {
                self.cache_aware_quicksort(data, pivot + 1, high);
            }
        }
    }

    /// Cache-aware partitioning for quicksort
    fn cache_aware_partition<T: Clone + Ord>(
        &mut self,
        data: &mut [T],
        low: usize,
        high: usize,
    ) -> usize {
        let pivot = data[high].clone();
        let mut i = low;

        for j in low..high {
            if data[j] <= pivot {
                data.swap(i, j);
                i += 1;
            }
        }

        data.swap(i, high);
        i
    }

    /// Cache-aware merge sort
    fn cache_aware_mergesort<T: Clone + Ord>(&mut self, data: &mut [T]) {
        let len = data.len();
        if len <= 1 {
            return;
        }

        let mid = len / 2;
        let mut left = data[..mid].to_vec();
        let mut right = data[mid..].to_vec();

        self.cache_aware_mergesort(&mut left);
        self.cache_aware_mergesort(&mut right);

        self.cache_aware_merge(data, &left, &right);
    }

    /// Cache-aware merge operation
    fn cache_aware_merge<T: Clone + Ord>(&mut self, data: &mut [T], left: &[T], right: &[T]) {
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                data[k] = left[i].clone();
                i += 1;
            } else {
                data[k] = right[j].clone();
                j += 1;
            }
            k += 1;
        }

        while i < left.len() {
            data[k] = left[i].clone();
            i += 1;
            k += 1;
        }

        while j < right.len() {
            data[k] = right[j].clone();
            j += 1;
            k += 1;
        }
    }
}

/// Adaptive algorithm selector for choosing optimal sorting strategy
pub struct AdaptiveAlgorithmSelector {
    cache_hierarchy: CacheHierarchy,
}

impl AdaptiveAlgorithmSelector {
    /// Create a new adaptive algorithm selector
    pub fn new(config: &CacheObliviousConfig) -> Self {
        Self {
            cache_hierarchy: config.cache_hierarchy.clone(),
        }
    }

    /// Select optimal sorting strategy based on data characteristics
    pub fn select_strategy(
        &self,
        data_size: usize,
        cache_hierarchy: &CacheHierarchy,
    ) -> CacheObliviousSortingStrategy {
        let element_size = std::mem::size_of::<u64>(); // Assume worst-case element size
        let data_bytes = data_size * element_size;

        if data_bytes <= cache_hierarchy.l1_size {
            // Small data: use cache-aware optimizations
            CacheObliviousSortingStrategy::CacheAware
        } else if data_bytes <= cache_hierarchy.l3_size {
            // Medium data: cache-oblivious is optimal
            CacheObliviousSortingStrategy::CacheOblivious
        } else {
            // Large data: hybrid approach
            CacheObliviousSortingStrategy::Hybrid
        }
    }

    /// Analyze data characteristics for algorithm selection
    pub fn analyze_data<T>(&self, data: &[T]) -> DataCharacteristics {
        DataCharacteristics {
            size: data.len(),
            memory_footprint: std::mem::size_of_val(data),
            fits_in_l1: std::mem::size_of_val(data) <= self.cache_hierarchy.l1_size,
            fits_in_l2: std::mem::size_of_val(data) <= self.cache_hierarchy.l2_size,
            fits_in_l3: std::mem::size_of_val(data) <= self.cache_hierarchy.l3_size,
        }
    }
}

/// Cache-oblivious sorting strategy enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheObliviousSortingStrategy {
    /// Use cache-aware algorithms with explicit cache knowledge
    CacheAware,
    /// Use the recursive multiway merge sort (see the module docs)
    CacheOblivious,
    /// Use hybrid approach combining both strategies
    Hybrid,
}

/// Data characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Number of elements
    pub size: usize,
    /// Memory footprint in bytes
    pub memory_footprint: usize,
    /// Whether data fits in L1 cache
    pub fits_in_l1: bool,
    /// Whether data fits in L2 cache
    pub fits_in_l2: bool,
    /// Whether data fits in L3 cache
    pub fits_in_l3: bool,
}

/// Van Emde Boas layout optimization for cache-optimal data structures
/// Enhanced with Zipora's cache optimization infrastructure
///
/// Limitation: no Van Emde Boas reordering is done. `data` is stored in its
/// original order and `cache_optimal_index` is the identity, so `get` is a
/// bounds-checked index plus, on x86-64 with AVX2, more than 64 elements and
/// a non-ZST `T`, a prefetch hint one cache line ahead.
pub struct VanEmdeBoas<T> {
    data: Vec<T>,
    cpu_features: &'static CpuFeatures,
    cache_line_size: usize,
}

impl<T: Clone> VanEmdeBoas<T> {
    /// Create a new Van Emde Boas layout with enhanced cache optimization
    pub fn new(data: Vec<T>, cache_hierarchy: CacheHierarchy) -> Self {
        let cpu_features = get_cpu_features();
        let cache_line_size = cache_hierarchy.l1_line_size;
        Self {
            data,
            cpu_features,
            cache_line_size,
        }
    }

    /// Create with custom CPU features for testing
    pub fn with_cpu_features(
        data: Vec<T>,
        cache_hierarchy: CacheHierarchy,
        cpu_features: &'static CpuFeatures,
    ) -> Self {
        let cache_line_size = cache_hierarchy.l1_line_size;
        Self {
            data,
            cpu_features,
            cache_line_size,
        }
    }

    /// Get element `index` (see the limitation on [`VanEmdeBoas`]: plain
    /// indexing plus a prefetch hint)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.data.len() {
            let physical_index = self.cache_optimal_index(index);

            // Prefetch nearby cache lines for better performance
            if self.cpu_features.has_avx2 && self.data.len() > 64 && std::mem::size_of::<T>() > 0 {
                let cache_line_elements =
                    self.cache_line_size / std::mem::size_of::<T>().max(1);
                let _ = cache_line_elements;
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                {
                    // Prefetch next cache line
                    if physical_index + cache_line_elements < self.data.len() {
                        // SAFETY: physical_index + cache_line_elements < data.len() ensures valid pointer within slice bounds
                        unsafe {
                            let ptr = self.data.as_ptr().add(physical_index + cache_line_elements)
                                as *const i8;
                            std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T1);
                        }
                    }
                }
            }

            Some(&self.data[physical_index])
        } else {
            None
        }
    }

    /// Convert logical index to physical index. This is the identity (clamped
    /// to the last element): see the limitation on [`VanEmdeBoas`].
    fn cache_optimal_index(&self, logical_index: usize) -> usize {
        if logical_index < self.data.len() {
            logical_index
        } else {
            self.data.len() - 1
        }
    }
}

impl Algorithm for CacheObliviousSort {
    type Config = CacheObliviousConfig;
    type Input = Vec<i32>; // Example type
    type Output = Vec<i32>;

    fn execute(&self, config: &Self::Config, mut input: Self::Input) -> Result<Self::Output> {
        let mut sorter = Self::with_config(config.clone());
        sorter.sort(&mut input)?;
        Ok(input)
    }

    fn stats(&self) -> AlgorithmStats {
        self.stats.clone()
    }

    fn estimate_memory(&self, input_size: usize) -> usize {
        // Estimate memory usage for cache-oblivious sorting
        input_size * std::mem::size_of::<i32>() * 2 // Temporary buffer
    }

    fn supports_parallel(&self) -> bool {
        // Sequential only; `use_parallel` has no effect.
        false
    }

    fn supports_simd(&self) -> bool {
        // No SIMD instructions; `use_simd` only selects prefetch hints.
        false
    }
}

impl Default for CacheObliviousSort {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_oblivious_sort_basic() {
        let mut sorter = CacheObliviousSort::new();
        let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];

        assert!(sorter.sort(&mut data).is_ok());
        assert_eq!(data, vec![1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]);
    }

    #[test]
    fn test_cache_oblivious_sort_empty() {
        let mut sorter = CacheObliviousSort::new();
        let mut data: Vec<i32> = vec![];

        assert!(sorter.sort(&mut data).is_ok());
        assert!(data.is_empty());
    }

    #[test]
    fn test_cache_oblivious_sort_single_element() {
        let mut sorter = CacheObliviousSort::new();
        let mut data = vec![42];

        assert!(sorter.sort(&mut data).is_ok());
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn test_cache_oblivious_sort_large() {
        let mut sorter = CacheObliviousSort::new();
        let mut data: Vec<i32> = (0..10000).rev().collect();
        let expected: Vec<i32> = (0..10000).collect();

        assert!(sorter.sort(&mut data).is_ok());
        assert_eq!(data, expected);
    }

    #[test]
    fn test_funnel_merge_many_chunks_matches_std_sort() {
        // Calls cache_oblivious_sort directly (bypassing the adaptive
        // selector) so funnel_sort_recursive + cache_oblivious_merge run
        // over many chunks, with duplicates and an uneven final chunk.
        let mut sorter = CacheObliviousSort::new();
        let mut data: Vec<u32> = (0..20_001u32)
            .map(|i| i.wrapping_mul(2_654_435_761) % 1_000)
            .collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        assert!(sorter.cache_oblivious_sort(&mut data).is_ok());
        assert_eq!(data, expected);
    }

    #[test]
    fn test_adaptive_algorithm_selector() {
        let config = CacheObliviousConfig::default();
        let selector = AdaptiveAlgorithmSelector::new(&config);

        // Small data should use cache-aware
        let strategy = selector.select_strategy(100, &config.cache_hierarchy);
        assert_eq!(strategy, CacheObliviousSortingStrategy::CacheAware);

        // Large data should use hybrid (larger than L3 cache)
        let large_size = config.cache_hierarchy.l3_size / 8 + 1000; // Exceed L3 cache
        let strategy = selector.select_strategy(large_size, &config.cache_hierarchy);
        assert_eq!(strategy, CacheObliviousSortingStrategy::Hybrid);
    }

    #[test]
    fn test_data_characteristics_analysis() {
        let config = CacheObliviousConfig::default();
        let selector = AdaptiveAlgorithmSelector::new(&config);
        let data = vec![1, 2, 3, 4, 5];

        let characteristics = selector.analyze_data(&data);
        assert_eq!(characteristics.size, 5);
        assert_eq!(
            characteristics.memory_footprint,
            5 * std::mem::size_of::<i32>()
        );
        assert!(characteristics.fits_in_l1);
    }

    #[test]
    fn test_van_emde_boas_layout() {
        let cache_hierarchy = CacheHierarchy::default();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let veb = VanEmdeBoas::new(data, cache_hierarchy);

        // Test basic access
        assert_eq!(veb.get(0), Some(&1));
        assert_eq!(veb.get(7), Some(&8));
        assert_eq!(veb.get(8), None);
    }

    #[test]
    fn test_funnel_width_calculation() {
        let sorter = CacheObliviousSort::new();
        let width = sorter.calculate_funnel_width(1000);
        assert!((2..=64).contains(&width));
    }

    #[test]
    fn test_insertion_sort() {
        let mut sorter = CacheObliviousSort::new();
        let mut data = vec![5, 2, 8, 1, 9];

        sorter.insertion_sort(&mut data);
        assert_eq!(data, vec![1, 2, 5, 8, 9]);
    }

    #[test]
    fn test_algorithm_trait_implementation() {
        let sorter = CacheObliviousSort::new();
        let config = CacheObliviousConfig::default();
        let input = vec![3, 1, 4, 1, 5];

        let result = sorter.execute(&config, input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 1, 3, 4, 5]);

        // Sequential, no SIMD instructions (S20-R6).
        assert!(!sorter.supports_parallel());
        assert!(!sorter.supports_simd());
        assert!(sorter.estimate_memory(1000) > 0);
    }

    #[test]
    fn test_s20_r6_cache_oblivious_reports_sequential_scalar_sort() {
        // CacheObliviousSort has no parallel code and no SIMD instructions
        // (only prefetch hints), yet `sort` reported
        // `used_parallel = use_parallel && len > 10000` and
        // `used_simd = use_simd && has_avx2`, and `supports_parallel()` and
        // `supports_simd()` returned true.
        let mut sorter = CacheObliviousSort::with_config(CacheObliviousConfig {
            use_parallel: true,
            use_simd: true,
            ..Default::default()
        });
        let mut data: Vec<u32> = (0..10_001u32)
            .map(|i| i.wrapping_mul(2_654_435_761))
            .collect();
        let mut want = data.clone();
        want.sort_unstable();
        sorter.sort(&mut data).unwrap();
        assert_eq!(data, want);
        let stats = Algorithm::stats(&sorter);
        assert!(
            !stats.used_parallel,
            "used_parallel after sorting 10,001 elements"
        );
        assert!(!stats.used_simd, "used_simd with use_simd = true");
        assert!(!sorter.supports_parallel());
        assert!(!sorter.supports_simd());
    }

    #[test]
    fn test_cache_oblivious_config_default() {
        let config = CacheObliviousConfig::default();
        assert!(config.cache_hierarchy.l1_size > 0);
        assert!(config.cache_hierarchy.l2_size > 0);
        assert!(config.cache_hierarchy.l3_size > 0);
        assert_eq!(config.small_threshold, 1024);
    }

    #[test]
    fn test_sorting_strategies() {
        let strategies = [
            CacheObliviousSortingStrategy::CacheAware,
            CacheObliviousSortingStrategy::CacheOblivious,
            CacheObliviousSortingStrategy::Hybrid,
        ];

        for strategy in &strategies {
            assert_ne!(format!("{:?}", strategy), "");
        }
    }

    #[test]
    fn test_c8_8_funnel_sort_deep_recursion_and_veb_zst() {
        // 1. VanEmdeBoas::get on >64 ZST elements previously panicked with division by zero.
        let veb_zst = VanEmdeBoas::new(vec![(); 100], CacheHierarchy::default());
        assert_eq!(veb_zst.get(0), Some(&()));
        assert_eq!(veb_zst.get(99), Some(&()));

        // 2. Deep funnel_sort_recursive with small_threshold = 2 and l2_line_size = 0 previously
        // panicked on division by zero or recursed infinitely when sqrt_k reached 1 or n < k.
        let mut config = CacheObliviousConfig {
            small_threshold: 2,
            ..Default::default()
        };
        config.cache_hierarchy.l2_line_size = 0;
        let mut sorter = CacheObliviousSort::with_config(config);
        let mut data: Vec<i32> = (0..120).map(|i| (i * 37 + 13) % 53).collect();
        let mut expected = data.clone();
        expected.sort_unstable();

        sorter.cache_oblivious_sort(&mut data).unwrap();
        assert_eq!(data, expected);
    }

    #[test]
    fn test_s20_r4_funnel_leaf_sort_is_stable_and_matches_std() {
        // Guard for the leaf-sort swap (insertion sort -> std stable sort):
        // ordered by `key` only, so the `tag`s of equal keys show whether the
        // whole funnel sort (leaves + heap merge) is still stable.
        #[derive(Debug, Clone)]
        struct Rec {
            key: u16,
            tag: u32,
        }
        impl PartialEq for Rec {
            fn eq(&self, other: &Self) -> bool {
                self.key == other.key
            }
        }
        impl Eq for Rec {}
        impl PartialOrd for Rec {
            fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Rec {
            fn cmp(&self, other: &Self) -> cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }

        let n: u32 = if cfg!(miri) { 3_000 } else { 50_000 };
        let input: Vec<Rec> = (0..n)
            .map(|i| Rec {
                key: ((i.wrapping_mul(2_654_435_761) >> 16) % 500) as u16,
                tag: i,
            })
            .collect();
        let mut want = input.clone();
        want.sort_by_key(|r| r.key);
        let want: Vec<(u16, u32)> = want.iter().map(|r| (r.key, r.tag)).collect();

        // The default leaf size (1024) and a small one for deeper recursion.
        for small_threshold in [1024, 16] {
            let mut sorter = CacheObliviousSort::with_config(CacheObliviousConfig {
                small_threshold,
                ..Default::default()
            });
            let mut v = input.clone();
            sorter.cache_oblivious_sort(&mut v).unwrap();
            let got: Vec<(u16, u32)> = v.iter().map(|r| (r.key, r.tag)).collect();
            assert!(got == want, "small_threshold={small_threshold}");
        }
    }
}
