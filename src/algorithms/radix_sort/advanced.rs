use super::config::{
    AdvancedAlgorithmStats, AdvancedRadixSortConfig, CpuFeatures, DataCharacteristics, PhaseTimes,
    SortingStrategy,
};
use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, SecurePoolConfig};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

// AVX2/BMI2 intrinsics for advanced SIMD acceleration
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_srlv_epi32,
    _mm256_storeu_si256,
};

/// Trait for sortable data types with radix sort optimizations
pub trait RadixSortable: Clone + Copy + Send + Sync + Ord {
    /// Extract the key for radix sorting (used for LSD radix sort)
    fn extract_key(&self) -> u64;

    /// Get byte at the specified position for MSD radix sort
    fn get_byte(&self, position: usize) -> Option<u8>;

    /// Get the maximum number of bytes for this data type
    fn max_bytes(&self) -> usize;

    /// Check if this type is suitable for parallel processing
    fn supports_parallel() -> bool {
        true
    }
}

impl RadixSortable for u32 {
    fn extract_key(&self) -> u64 {
        *self as u64
    }

    fn get_byte(&self, position: usize) -> Option<u8> {
        if position < 4 {
            Some(((*self >> (8 * (3 - position))) & 0xFF) as u8)
        } else {
            None
        }
    }

    fn max_bytes(&self) -> usize {
        4
    }
}

impl RadixSortable for u64 {
    fn extract_key(&self) -> u64 {
        *self
    }

    fn get_byte(&self, position: usize) -> Option<u8> {
        if position < 8 {
            Some(((*self >> (8 * (7 - position))) & 0xFF) as u8)
        } else {
            None
        }
    }

    fn max_bytes(&self) -> usize {
        8
    }
}

// Note: Vec<u8> cannot implement RadixSortable because it's not Copy
// Instead, we'll provide a specialized string radix sort for &[u8] or create a wrapper type

/// A copyable string wrapper for radix sorting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RadixString<'a> {
    data: &'a [u8],
}

impl<'a> RadixString<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }
}

impl<'a> RadixSortable for RadixString<'a> {
    fn extract_key(&self) -> u64 {
        // For strings, we extract the first 8 bytes as a key
        let mut key = 0u64;
        for (i, &byte) in self.data.iter().take(8).enumerate() {
            key |= (byte as u64) << (8 * (7 - i));
        }
        key
    }

    fn get_byte(&self, position: usize) -> Option<u8> {
        self.data.get(position).copied()
    }

    fn max_bytes(&self) -> usize {
        self.data.len()
    }

    fn supports_parallel() -> bool {
        true // String sorting can be parallelized with careful design
    }
}

/// Advanced radix sort implementation with multiple algorithm variants
pub struct AdvancedRadixSort<T: RadixSortable> {
    config: AdvancedRadixSortConfig,
    stats: AdvancedAlgorithmStats,
    memory_pool: Option<Arc<SecureMemoryPool>>,
    cpu_features: CpuFeatures,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: RadixSortable> AdvancedRadixSort<T> {
    /// Create a new advanced radix sort instance
    pub fn new() -> Result<Self> {
        Self::with_config(AdvancedRadixSortConfig::default())
    }

    /// Create an advanced radix sort instance with custom configuration
    pub fn with_config(config: AdvancedRadixSortConfig) -> Result<Self> {
        let cpu_features = CpuFeatures::detect();
        let memory_pool = if config.use_secure_memory {
            // Create a secure pool config for the memory budget
            let pool_config = if config.memory_budget <= 64 * 1024 {
                SecurePoolConfig::small_secure()
            } else if config.memory_budget <= 1024 * 1024 {
                SecurePoolConfig::medium_secure()
            } else {
                SecurePoolConfig::large_secure()
            };

            Some(SecureMemoryPool::new(pool_config)?)
        } else {
            None
        };

        Ok(Self {
            config,
            stats: AdvancedAlgorithmStats {
                basic_stats: AlgorithmStats {
                    items_processed: 0,
                    processing_time_us: 0,
                    memory_used: 0,
                    used_parallel: false,
                    used_simd: false,
                },
                strategy_used: SortingStrategy::Adaptive,
                cpu_features_used: cpu_features.clone(),
                estimated_cache_misses: 0,
                peak_memory_bytes: 0,
                threads_used: 0,
                phase_times: PhaseTimes {
                    analysis_time_us: 0,
                    allocation_time_us: 0,
                    sorting_time_us: 0,
                    merging_time_us: 0,
                    cleanup_time_us: 0,
                },
            },
            memory_pool,
            cpu_features,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create an advanced radix sort instance with shared memory pool
    pub fn with_memory_pool(
        config: AdvancedRadixSortConfig,
        memory_pool: Arc<SecureMemoryPool>,
    ) -> Self {
        let cpu_features = CpuFeatures::detect();

        Self {
            config,
            stats: AdvancedAlgorithmStats {
                basic_stats: AlgorithmStats {
                    items_processed: 0,
                    processing_time_us: 0,
                    memory_used: 0,
                    used_parallel: false,
                    used_simd: false,
                },
                strategy_used: SortingStrategy::Adaptive,
                cpu_features_used: cpu_features.clone(),
                estimated_cache_misses: 0,
                peak_memory_bytes: 0,
                threads_used: 0,
                phase_times: PhaseTimes {
                    analysis_time_us: 0,
                    allocation_time_us: 0,
                    sorting_time_us: 0,
                    merging_time_us: 0,
                    cleanup_time_us: 0,
                },
            },
            memory_pool: Some(memory_pool),
            cpu_features,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sort data using adaptive strategy selection
    pub fn sort(&mut self, data: &mut [T]) -> Result<()> {
        let total_start = Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        // Phase 1: Data analysis for adaptive strategy selection
        let analysis_start = Instant::now();
        let strategy = self.select_strategy(data)?;
        self.stats.phase_times.analysis_time_us = analysis_start.elapsed().as_micros() as u64;

        // Phase 2: Execute the selected strategy
        let sorting_start = Instant::now();
        match strategy {
            SortingStrategy::Insertion => self.insertion_sort(data)?,
            SortingStrategy::TimSort => self.tim_sort(data)?,
            SortingStrategy::LsdRadix => self.lsd_radix_sort(data)?,
            SortingStrategy::MsdRadix => self.msd_radix_sort(data, 0)?,
            SortingStrategy::Adaptive => {
                // This shouldn't happen as select_strategy should return a concrete strategy
                return Err(ZiporaError::invalid_data(
                    "Invalid adaptive strategy selection",
                ));
            }
        }
        self.stats.phase_times.sorting_time_us = sorting_start.elapsed().as_micros() as u64;

        // Update final statistics
        let total_elapsed = total_start.elapsed();
        self.stats.basic_stats.items_processed = data.len();
        self.stats.basic_stats.processing_time_us = total_elapsed.as_micros() as u64;
        self.stats.strategy_used = strategy;

        Ok(())
    }

    /// Select the optimal sorting strategy based on data characteristics
    fn select_strategy(&self, data: &[T]) -> Result<SortingStrategy> {
        // If a specific strategy is forced, use it
        if let Some(strategy) = self.config.force_strategy {
            return Ok(strategy);
        }

        // If adaptive strategy is disabled, default to LSD radix sort
        if !self.config.adaptive_strategy {
            return Ok(SortingStrategy::LsdRadix);
        }

        let size = data.len();

        // For very small datasets, use insertion sort
        if size <= self.config.insertion_sort_threshold {
            return Ok(SortingStrategy::Insertion);
        }

        // Quick sortedness check for Tim sort
        if self.is_nearly_sorted(data) {
            return Ok(SortingStrategy::TimSort);
        }

        // For most cases, use LSD radix sort with SIMD optimizations
        Ok(SortingStrategy::LsdRadix)
    }

    /// Check if data is nearly sorted (good candidate for Tim sort)
    fn is_nearly_sorted(&self, data: &[T]) -> bool {
        if data.len() < 2 {
            return true;
        }

        let mut inversions = 0usize;
        let sample_size = std::cmp::min(1000, data.len()); // Sample for large datasets

        for i in 1..sample_size {
            if data[i].extract_key() < data[i - 1].extract_key() {
                inversions += 1;
            }
        }

        // Consider nearly sorted if inversions are less than 10% of sampled pairs
        inversions < sample_size / 10
    }

    /// Get performance statistics from the last execution
    pub fn stats(&self) -> &AdvancedAlgorithmStats {
        &self.stats
    }

    /// Estimate memory requirements for the given input size
    pub fn estimate_memory(&self, input_size: usize) -> usize {
        let radix_size = 1usize << self.config.radix_bits;

        // Base memory for buffers and counting arrays
        let base_memory = input_size * std::mem::size_of::<T>() + // Buffer
                         radix_size * std::mem::size_of::<usize>(); // Counts

        // Additional memory for parallel processing
        let parallel_memory =
            if self.config.use_parallel && input_size >= self.config.parallel_threshold {
                let num_threads = if self.config.num_threads > 0 {
                    self.config.num_threads
                } else {
                    rayon::current_num_threads()
                };
                base_memory * num_threads / 4 // Rough estimate for parallel buffers
            } else {
                0
            };

        base_memory + parallel_memory
    }

    /// Insertion sort for small datasets
    fn insertion_sort(&mut self, data: &mut [T]) -> Result<()> {
        for i in 1..data.len() {
            let key = data[i];
            let key_value = key.extract_key();
            let mut j = i;

            while j > 0 && data[j - 1].extract_key() > key_value {
                data[j] = data[j - 1];
                j -= 1;
            }
            data[j] = key;
        }

        self.stats.basic_stats.used_parallel = false;
        self.stats.basic_stats.used_simd = false;
        Ok(())
    }

    /// Tim sort for nearly sorted data (simplified implementation)
    fn tim_sort(&mut self, data: &mut [T]) -> Result<()> {
        // This is a simplified version - a full Tim sort implementation would be much more complex
        // For now, we use the standard library's unstable_sort which is based on pattern-defeating quicksort
        data.sort_unstable_by_key(|item| item.extract_key());

        self.stats.basic_stats.used_parallel = false;
        self.stats.basic_stats.used_simd = false;
        Ok(())
    }

    /// LSD radix sort with enhanced SIMD optimizations
    fn lsd_radix_sort(&mut self, data: &mut [T]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let should_use_parallel = data.len() >= self.config.parallel_threshold
            && self.config.use_parallel
            && T::supports_parallel();

        if should_use_parallel {
            self.lsd_radix_sort_parallel(data)
        } else {
            self.lsd_radix_sort_sequential(data)
        }
    }

    /// Sequential LSD radix sort with SIMD optimizations
    fn lsd_radix_sort_sequential(&mut self, data: &mut [T]) -> Result<()> {
        let radix = 1usize << self.config.radix_bits;
        let mask = (radix - 1) as u64;

        // Allocate buffers - use memory pool if available
        let buffer = if let Some(ref _pool) = self.memory_pool {
            // For now, fall back to regular allocation
            // TODO: Implement proper memory pool integration for generic types
            vec![data[0]; data.len()]
        } else {
            vec![data[0]; data.len()]
        };

        let mut buffer = buffer;
        let mut counts = vec![0usize; radix];

        // Determine maximum number of passes needed
        let max_key = data
            .iter()
            .map(|item| item.extract_key())
            .max()
            .unwrap_or(0);
        let max_passes = if max_key == 0 {
            1
        } else {
            (64 - max_key.leading_zeros() as usize).div_ceil(self.config.radix_bits)
        };

        for pass in 0..max_passes {
            let shift = pass * self.config.radix_bits;

            // Count occurrences with SIMD optimization when available
            counts.fill(0);

            if self.config.use_simd && self.cpu_features.has_advanced_simd() && data.len() >= 16 {
                self.count_digits_simd(data, shift, mask, &mut counts)?;
            } else {
                // Sequential counting
                for item in data.iter() {
                    let key = item.extract_key();
                    let digit = ((key >> shift) & mask) as usize;
                    counts[digit] += 1;
                }
            }

            // Convert counts to positions (cumulative sum)
            let mut pos = 0;
            for count in counts.iter_mut() {
                let old_count = *count;
                *count = pos;
                pos += old_count;
            }

            // Distribute elements to buffer
            for item in data.iter() {
                let key = item.extract_key();
                let digit = ((key >> shift) & mask) as usize;
                buffer[counts[digit]] = *item;
                counts[digit] += 1;
            }

            // Copy back to original array
            data.copy_from_slice(&buffer);
        }

        self.stats.basic_stats.used_parallel = false;
        self.stats.basic_stats.used_simd =
            self.config.use_simd && self.cpu_features.has_advanced_simd();
        Ok(())
    }

    /// Parallel LSD radix sort with work-stealing
    fn lsd_radix_sort_parallel(&mut self, data: &mut [T]) -> Result<()> {
        let num_threads = if self.config.num_threads > 0 {
            self.config.num_threads
        } else {
            rayon::current_num_threads()
        };

        // For very large datasets, use parallel radix sort
        if data.len() < 2 * self.config.parallel_threshold {
            return self.lsd_radix_sort_sequential(data);
        }

        // Split into chunks and sort in parallel
        let chunk_size = data.len().div_ceil(num_threads);

        data.par_chunks_mut(chunk_size).for_each(|chunk| {
            if let Ok(mut temp_sorter) = AdvancedRadixSort::with_config(AdvancedRadixSortConfig {
                use_parallel: false,
                ..self.config.clone()
            }) {
                let _ = temp_sorter.lsd_radix_sort_sequential(chunk);
            }
        });

        // Use multi-way merge to combine sorted chunks
        self.multiway_merge_chunks(data, chunk_size)?;

        self.stats.basic_stats.used_parallel = true;
        self.stats.basic_stats.used_simd =
            self.config.use_simd && self.cpu_features.has_advanced_simd();
        self.stats.threads_used = num_threads;
        Ok(())
    }

    /// MSD radix sort for string-like data
    fn msd_radix_sort(&mut self, data: &mut [T], depth: usize) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        // Switch to insertion sort for small datasets or deep recursion
        if data.len() <= self.config.insertion_sort_threshold || depth > 64 {
            return self.insertion_sort(data);
        }

        // Create buckets for each possible byte value (0-255) plus end-of-string bucket
        let mut buckets: Vec<Vec<T>> = vec![Vec::new(); 257];

        // Distribute items into buckets based on byte at current depth
        for &item in data.iter() {
            let bucket_index = if let Some(byte) = item.get_byte(depth) {
                byte as usize + 1 // +1 to reserve bucket 0 for end-of-string
            } else {
                0 // End of string
            };
            buckets[bucket_index].push(item);
        }

        // Recursively sort each bucket and collect results back to original data
        let mut offset = 0;
        for (i, mut bucket) in buckets.into_iter().enumerate() {
            if bucket.len() > 1 && i > 0 {
                // Skip bucket 0 (end-of-string) as it's already sorted
                self.msd_radix_sort(&mut bucket, depth + 1)?;
            }

            // Copy bucket back to data
            for item in bucket {
                data[offset] = item;
                offset += 1;
            }
        }

        self.stats.basic_stats.used_parallel = false; // MSD is inherently sequential for strings
        self.stats.basic_stats.used_simd = false;
        Ok(())
    }

    /// SIMD-optimized digit counting for LSD radix sort
    fn count_digits_simd(
        &self,
        data: &[T],
        shift: usize,
        mask: u64,
        counts: &mut [usize],
    ) -> Result<()> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("bmi2")
            {
                // SAFETY: avx2 and bmi2 features detected at runtime
                unsafe {
                    self.count_digits_avx2_bmi2(data, shift, mask, counts)?;
                }
                return Ok(());
            }
        }

        // Fallback to sequential counting
        for item in data.iter() {
            let key = item.extract_key();
            let digit = ((key >> shift) & mask) as usize;
            counts[digit] += 1;
        }

        Ok(())
    }

    /// AVX2 + BMI2 accelerated digit counting
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi2")]
    unsafe fn count_digits_avx2_bmi2(
        &self,
        data: &[T],
        shift: usize,
        mask: u64,
        counts: &mut [usize],
    ) -> Result<()> {
        // This is a simplified version - full SIMD implementation would be more complex
        // For generic types, we need to extract keys first
        let mut keys: Vec<u64> = data.iter().map(|item| item.extract_key()).collect();

        let mut i = 0;
        let shift_vec = _mm256_set1_epi32(shift as i32);
        let mask_vec = _mm256_set1_epi32(mask as i32);

        // Process 8 u64 values at a time using AVX2 (requires casting to u32)
        while i + 8 <= keys.len() {
            // Load 8 u64 values as u32 (lower 32 bits)
            // Use stack-allocated array instead of Vec to avoid heap allocation in hot loop
            let keys_u32: [u32; 8] = [
                keys[i] as u32,
                keys[i + 1] as u32,
                keys[i + 2] as u32,
                keys[i + 3] as u32,
                keys[i + 4] as u32,
                keys[i + 5] as u32,
                keys[i + 6] as u32,
                keys[i + 7] as u32,
            ];
            // SAFETY: #[target_feature] ensures avx2, keys_u32 is 8 u32s = 32 bytes
            let values = unsafe { _mm256_loadu_si256(keys_u32.as_ptr() as *const __m256i) };

            // Shift and mask to extract digits
            let shifted = if shift > 0 {
                // SAFETY: #[target_feature] ensures avx2/bmi2, values is valid __m256i
                _mm256_srlv_epi32(values, shift_vec)
            } else {
                values
            };
            // SAFETY: #[target_feature] ensures avx2, shifted is valid __m256i
            let digits = _mm256_and_si256(shifted, mask_vec);

            // Extract digits and count them
            let mut digit_array = [0u32; 8];
            // SAFETY: digit_array is 8 u32s = 32 bytes, matches __m256i size
            unsafe { _mm256_storeu_si256(digit_array.as_mut_ptr() as *mut __m256i, digits) };

            for &digit in &digit_array {
                if (digit as usize) < counts.len() {
                    counts[digit as usize] += 1;
                }
            }

            i += 8;
        }

        // Handle remaining elements
        for &key in &keys[i..] {
            let digit = ((key >> shift) & mask) as usize;
            if digit < counts.len() {
                counts[digit] += 1;
            }
        }

        Ok(())
    }

    /// Multi-way merge for combining parallel sorted chunks
    fn multiway_merge_chunks(&self, data: &mut [T], _chunk_size: usize) -> Result<()> {
        // For now, use a simple approach: collect all elements and sort
        // This is less efficient than true multi-way merge but ensures correctness
        // TODO: Implement proper multi-way merge when the generic bounds are resolved

        // Since we know all chunks are already sorted, we can use unstable sort
        // which will be very fast for mostly-sorted data
        data.sort_unstable();

        Ok(())
    }
}

impl<T: RadixSortable> Default for AdvancedRadixSort<T> {
    fn default() -> Self {
        // SAFETY: AdvancedRadixSort::new() only fails on memory allocation errors.
        // Use unwrap_or_else with panic as this type has complex dependencies.
        Self::new().unwrap_or_else(|e| {
            panic!(
                "AdvancedRadixSort creation failed in Default: {}. \
                   This indicates severe memory pressure.",
                e
            )
        })
    }
}

impl<T: RadixSortable> Algorithm for AdvancedRadixSort<T> {
    type Config = AdvancedRadixSortConfig;
    type Input = Vec<T>;
    type Output = Vec<T>;

    fn execute(&self, config: &Self::Config, mut input: Self::Input) -> Result<Self::Output> {
        let mut sorter = Self::with_config(config.clone())?;
        sorter.sort(&mut input)?;
        Ok(input)
    }

    fn stats(&self) -> AlgorithmStats {
        self.stats.basic_stats.clone()
    }

    fn estimate_memory(&self, input_size: usize) -> usize {
        self.estimate_memory(input_size)
    }

    fn supports_parallel(&self) -> bool {
        T::supports_parallel()
    }

    fn supports_simd(&self) -> bool {
        self.config.use_simd
    }
}

/// Convenience type alias for advanced u32 radix sort
pub type AdvancedU32RadixSort = AdvancedRadixSort<u32>;

/// Convenience type alias for advanced u64 radix sort
pub type AdvancedU64RadixSort = AdvancedRadixSort<u64>;

/// Convenience type alias for advanced string radix sort
pub type AdvancedStringRadixSort<'a> = AdvancedRadixSort<RadixString<'a>>;
