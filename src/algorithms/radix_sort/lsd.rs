use super::advanced::RadixSortable;
use super::config::RadixSortConfig;
use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::Result;
use rayon::prelude::*;

// AVX2/BMI2 intrinsics for advanced SIMD acceleration

// AVX-512 intrinsics (avx512 feature)
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use std::arch::x86_64::{
    __m512i, _mm512_and_si512, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_srlv_epi32,
    _mm512_storeu_si512,
};

/// Maximum value range for which counting sort is selected.
///
/// `counting_sort_u32` allocates `(max_val + 1)` buckets, so its memory cost is
/// O(value range), not O(len). Selecting it on element count alone allows a
/// handful of large-valued elements (e.g. a single `u32::MAX`) to request 32 GiB.
/// Above this range LSD radix sort is used instead (bounded O(len) memory).
const COUNTING_SORT_MAX_RANGE: usize = 1 << 16; // 65_536 buckets (512 KiB)

/// High-performance radix sort implementation
pub struct RadixSort {
    config: RadixSortConfig,
    stats: AlgorithmStats,
}

impl RadixSort {
    /// Create a new radix sort instance
    pub fn new() -> Self {
        Self::with_config(RadixSortConfig::default())
    }

    /// Create a radix sort instance with custom configuration
    pub fn with_config(config: RadixSortConfig) -> Self {
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

    /// Sort a slice of unsigned 32-bit integers
    pub fn sort_u32(&mut self, data: &mut [u32]) -> Result<()> {
        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        let used_parallel =
            data.len() >= self.config.parallel_threshold && self.config.use_parallel;

        if used_parallel {
            self.sort_u32_parallel(data)?;
        } else {
            self.sort_u32_sequential(data)?;
        }

        let elapsed = start_time.elapsed();
        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: self.estimate_memory_u32(data.len()),
            used_parallel,
            used_simd: self.config.use_simd,
        };

        Ok(())
    }

    /// Sort a slice of unsigned 64-bit integers
    pub fn sort_u64(&mut self, data: &mut [u64]) -> Result<()> {
        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        let used_parallel =
            data.len() >= self.config.parallel_threshold && self.config.use_parallel;

        if used_parallel {
            self.sort_u64_parallel(data)?;
        } else {
            self.sort_u64_sequential(data)?;
        }

        let elapsed = start_time.elapsed();
        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: self.estimate_memory_u64(data.len()),
            used_parallel,
            used_simd: self.config.use_simd,
        };

        Ok(())
    }

    /// Get algorithm statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
    }

    /// Sort a slice of byte arrays by their content
    pub fn sort_bytes(&mut self, data: &mut Vec<Vec<u8>>) -> Result<()> {
        let start_time = std::time::Instant::now();

        if data.is_empty() {
            return Ok(());
        }

        self.sort_bytes_msd(data.as_mut_slice(), 0)?;

        let elapsed = start_time.elapsed();
        let total_bytes: usize = data.iter().map(|v| v.len()).sum();

        self.stats = AlgorithmStats {
            items_processed: data.len(),
            processing_time_us: elapsed.as_micros() as u64,
            memory_used: total_bytes + data.len() * std::mem::size_of::<Vec<u8>>(),
            used_parallel: false, // MSD radix sort is inherently sequential for strings
            used_simd: false,
        };

        Ok(())
    }

    fn sort_u32_sequential(&self, data: &mut [u32]) -> Result<()> {
        // Counting sort is chosen only when the dataset is small AND its value
        // range is bounded — its bucket array is sized by the max value, so a
        // large range would allocate excessive memory regardless of len.
        if data.len() <= self.config.use_counting_sort_threshold {
            let max_val = data.iter().copied().max().unwrap_or(0) as usize;
            if max_val <= COUNTING_SORT_MAX_RANGE {
                self.counting_sort_u32(data);
                return Ok(());
            }
        }

        let radix = 1usize << self.config.radix_bits;
        let mask = (radix - 1) as u32;
        let mut buffer = vec![0u32; data.len()];
        let mut counts = vec![0usize; radix];

        let max_passes = 32_usize.div_ceil(self.config.radix_bits);

        for pass in 0..max_passes {
            let shift = pass * self.config.radix_bits;

            // Count occurrences with AVX-512 acceleration when available
            counts.fill(0);

            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            {
                if self.config.use_simd && data.len() >= 16 && shift < 24 {
                    // Use AVX-512 for faster counting when available
                    if std::arch::is_x86_feature_detected!("avx512f")
                        && std::arch::is_x86_feature_detected!("avx512bw")
                    {
                        // SAFETY: avx512f and avx512bw features detected at runtime
                        unsafe {
                            self.count_digits_avx512(data, shift, mask, &mut counts);
                        }
                    } else {
                        // Fallback to sequential counting
                        for &value in data.iter() {
                            let digit = ((value >> shift) & mask) as usize;
                            counts[digit] += 1;
                        }
                    }
                } else {
                    // Sequential counting for small data or high shifts
                    for &value in data.iter() {
                        let digit = ((value >> shift) & mask) as usize;
                        counts[digit] += 1;
                    }
                }
            }

            #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
            {
                // Standard sequential counting
                for &value in data.iter() {
                    let digit = ((value >> shift) & mask) as usize;
                    counts[digit] += 1;
                }
            }

            // Convert counts to positions
            let mut pos = 0;
            for count in counts.iter_mut() {
                let old_count = *count;
                *count = pos;
                pos += old_count;
            }

            // Distribute elements
            for &value in data.iter() {
                let digit = ((value >> shift) & mask) as usize;
                buffer[counts[digit]] = value;
                counts[digit] += 1;
            }

            // Copy back
            data.copy_from_slice(&buffer);
        }

        Ok(())
    }

    fn sort_u32_parallel(&self, data: &mut [u32]) -> Result<()> {
        // For very large datasets, use parallel radix sort
        // This is a simplified version - full parallel radix sort is quite complex

        if data.len() < 2 * self.config.parallel_threshold {
            return self.sort_u32_sequential(data);
        }

        // Split into chunks and sort in parallel
        let num_threads = rayon::current_num_threads();
        let chunk_size = data.len().div_ceil(num_threads); // Round up

        data.par_chunks_mut(chunk_size).for_each(|chunk| {
            let temp_sorter = RadixSort::with_config(RadixSortConfig {
                use_parallel: false,
                ..self.config.clone()
            });
            let _ = temp_sorter.sort_u32_sequential(chunk);
        });

        // Use proper multi-way merge instead of sort_unstable()
        self.multiway_merge_u32_chunks(data, chunk_size)?;

        Ok(())
    }

    fn sort_u64_sequential(&self, data: &mut [u64]) -> Result<()> {
        let radix = 1usize << self.config.radix_bits;
        let mask = (radix - 1) as u64;
        let mut buffer = vec![0u64; data.len()];
        let mut counts = vec![0usize; radix];

        let max_passes = 64_usize.div_ceil(self.config.radix_bits);

        for pass in 0..max_passes {
            let shift = pass * self.config.radix_bits;

            // Count occurrences
            counts.fill(0);
            for &value in data.iter() {
                let digit = ((value >> shift) & mask) as usize;
                counts[digit] += 1;
            }

            // Convert counts to positions
            let mut pos = 0;
            for count in counts.iter_mut() {
                let old_count = *count;
                *count = pos;
                pos += old_count;
            }

            // Distribute elements
            for &value in data.iter() {
                let digit = ((value >> shift) & mask) as usize;
                buffer[counts[digit]] = value;
                counts[digit] += 1;
            }

            // Copy back
            data.copy_from_slice(&buffer);
        }

        Ok(())
    }

    fn sort_u64_parallel(&self, data: &mut [u64]) -> Result<()> {
        // Similar to u32 parallel sort
        if data.len() < 2 * self.config.parallel_threshold {
            return self.sort_u64_sequential(data);
        }

        let num_threads = rayon::current_num_threads();
        let chunk_size = data.len().div_ceil(num_threads); // Round up

        data.par_chunks_mut(chunk_size).for_each(|chunk| {
            let temp_sorter = RadixSort::with_config(RadixSortConfig {
                use_parallel: false,
                ..self.config.clone()
            });
            let _ = temp_sorter.sort_u64_sequential(chunk);
        });

        // Use proper multi-way merge instead of sort_unstable()
        self.multiway_merge_u64_chunks(data, chunk_size)?;

        Ok(())
    }

    fn counting_sort_u32(&self, data: &mut [u32]) {
        if data.is_empty() {
            return;
        }

        // Find maximum value to determine count buffer size
        // SAFETY: is_empty() check above guarantees iterator has at least one element
        let max_val = *data.iter().max().expect("non-empty input") as usize;
        let mut counts = vec![0usize; max_val + 1];

        // Count occurrences
        for &value in data.iter() {
            counts[value as usize] += 1;
        }

        // Reconstruct sorted array
        let mut index = 0;
        for (value, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                data[index] = value as u32;
                index += 1;
            }
        }
    }

    fn sort_bytes_msd(&self, data: &mut [Vec<u8>], depth: usize) -> Result<()> {
        if data.len() <= 1 {
            return Ok(());
        }

        let mut counts = [0usize; 257];

        for item in data.iter() {
            let b = if depth < item.len() {
                item[depth] as usize + 1
            } else {
                0
            };
            counts[b] += 1;
        }

        let mut offsets = [0usize; 257];
        let mut current_pos = 0;
        for i in 0..257 {
            offsets[i] = current_pos;
            current_pos += counts[i];
        }

        let mut next_free = offsets;

        for b in 0..257 {
            let end = if b == 256 { data.len() } else { offsets[b + 1] };
            while next_free[b] < end {
                let pos = next_free[b];
                let item_b = if depth < data[pos].len() {
                    data[pos][depth] as usize + 1
                } else {
                    0
                };

                if item_b == b {
                    next_free[b] += 1;
                } else {
                    data.swap(pos, next_free[item_b]);
                    next_free[item_b] += 1;
                }
            }
        }

        // Recursively sort each bucket (skip bucket 0 as it contains strings that have ended)
        for b in 1..257 {
            let start = offsets[b];
            let end = if b == 256 { data.len() } else { offsets[b + 1] };
            if end - start > 1 {
                self.sort_bytes_msd(&mut data[start..end], depth + 1)?;
            }
        }

        Ok(())
    }

    fn estimate_memory_u32(&self, len: usize) -> usize {
        let radix = 1usize << self.config.radix_bits;
        len * std::mem::size_of::<u32>() + // buffer
        radix * std::mem::size_of::<usize>() // counts
    }

    fn estimate_memory_u64(&self, len: usize) -> usize {
        let radix = 1usize << self.config.radix_bits;
        len * std::mem::size_of::<u64>() + // buffer
        radix * std::mem::size_of::<usize>() // counts
    }

    /// Multi-way merge for u32 chunks
    fn multiway_merge_u32_chunks(&self, data: &mut [u32], chunk_size: usize) -> Result<()> {
        use crate::algorithms::multiway_merge::{MultiWayMerge, SliceSource};

        // If no merging needed
        if data.is_empty() || chunk_size >= data.len() {
            return Ok(());
        }

        let mut sources = Vec::new();
        for chunk in data.chunks(chunk_size) {
            sources.push(SliceSource::new(chunk));
        }

        let mut merger = MultiWayMerge::new();
        let merged = merger.merge(sources)?;

        data.copy_from_slice(&merged);

        Ok(())
    }

    /// Multi-way merge for u64 chunks
    fn multiway_merge_u64_chunks(&self, data: &mut [u64], chunk_size: usize) -> Result<()> {
        use crate::algorithms::multiway_merge::{MultiWayMerge, SliceSource};

        if data.is_empty() || chunk_size >= data.len() {
            return Ok(());
        }

        let mut sources = Vec::new();
        for chunk in data.chunks(chunk_size) {
            sources.push(SliceSource::new(chunk));
        }

        let mut merger = MultiWayMerge::new();
        let merged = merger.merge(sources)?;

        data.copy_from_slice(&merged);

        Ok(())
    }

    /// AVX-512 accelerated digit counting for radix sort
    ///
    /// Processes multiple integers simultaneously to count digit occurrences,
    /// providing significant speedup for the counting phase of radix sort.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn count_digits_avx512(
        &self,
        data: &[u32],
        shift: usize,
        mask: u32,
        counts: &mut [usize],
    ) {
        let mut i = 0;
        let shift_vec = _mm512_set1_epi32(shift as i32);

        // Process 16 u32 values at a time using AVX-512
        while i + 16 <= data.len() {
            // Load 16 x u32 values (512 bits)
            // SAFETY: #[target_feature] ensures avx512f, i+16 <= data.len() ensures 64 bytes available
            let values = unsafe { _mm512_loadu_si512(data[i..].as_ptr() as *const __m512i) };

            // Shift all values to extract the desired digit
            let shifted = if shift > 0 {
                _mm512_srlv_epi32(values, shift_vec) // Variable shift per element
            } else {
                values
            };

            // Apply mask to get only the desired bits
            let mask_vec = _mm512_set1_epi32(mask as i32);
            let digits = _mm512_and_si512(shifted, mask_vec);

            // Extract digits and count them
            // Note: This could be further optimized with gather operations
            // For now, extract to array and count sequentially
            let mut digit_array = [0u32; 16];
            // SAFETY: digit_array is 16 u32s = 64 bytes, matches __m512i size
            unsafe { _mm512_storeu_si512(digit_array.as_mut_ptr() as *mut __m512i, digits) };

            for digit in digit_array.iter() {
                counts[*digit as usize] += 1;
            }

            i += 16;
        }

        // Handle remaining elements sequentially
        for &value in &data[i..] {
            let digit = ((value >> shift) & mask) as usize;
            counts[digit] += 1;
        }
    }
}

impl Default for RadixSort {
    fn default() -> Self {
        Self::new()
    }
}

impl Algorithm for RadixSort {
    type Config = RadixSortConfig;
    type Input = Vec<u32>;
    type Output = Vec<u32>;

    fn execute(&self, config: &Self::Config, mut input: Self::Input) -> Result<Self::Output> {
        let mut sorter = Self::with_config(config.clone());
        sorter.sort_u32(&mut input)?;
        Ok(input)
    }

    fn stats(&self) -> AlgorithmStats {
        self.stats.clone()
    }

    fn estimate_memory(&self, input_size: usize) -> usize {
        self.estimate_memory_u32(input_size)
    }

    fn supports_parallel(&self) -> bool {
        true
    }

    fn supports_simd(&self) -> bool {
        cfg!(feature = "simd")
    }
}

/// Specialized radix sort for key-value pairs
pub struct KeyValueRadixSort<K, V> {
    config: RadixSortConfig,
    _phantom: std::marker::PhantomData<(K, V)>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct KVPair {
    key: u64,
    index: usize,
}

impl RadixSortable for KVPair {
    fn extract_key(&self) -> u64 {
        self.key
    }

    fn get_byte(&self, position: usize) -> Option<u8> {
        if position < 8 {
            Some((self.key >> ((7 - position) * 8)) as u8)
        } else {
            None
        }
    }

    fn max_bytes(&self) -> usize {
        8
    }
}

impl<K, V> KeyValueRadixSort<K, V>
where
    K: Copy + Into<u64>,
    V: Clone,
{
    /// Create a new key-value radix sort
    pub fn new() -> Self {
        Self {
            config: RadixSortConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Sort key-value pairs by key
    pub fn sort_by_key(&self, data: &mut [(K, V)]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let mut indices: Vec<KVPair> = data
            .iter()
            .enumerate()
            .map(|(i, (k, _))| KVPair {
                key: (*k).into(),
                index: i,
            })
            .collect();

        let mut config = super::config::AdvancedRadixSortConfig::default();
        config.use_parallel = self.config.use_parallel;
        config.parallel_threshold = self.config.parallel_threshold;
        config.radix_bits = self.config.radix_bits;

        let mut sorter = super::advanced::AdvancedRadixSort::<KVPair>::with_config(config)
            .or_else(|_| super::advanced::AdvancedRadixSort::new())?;

        sorter.sort(&mut indices)?;

        let mut targets = vec![0; data.len()];
        for (i, ki) in indices.iter().enumerate() {
            targets[ki.index] = i;
        }

        for i in 0..data.len() {
            while targets[i] != i {
                let alt = targets[i];
                data.swap(i, alt);
                targets.swap(i, alt);
            }
        }

        Ok(())
    }
}

impl<K, V> Default for KeyValueRadixSort<K, V>
where
    K: Copy + Into<u64>,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}
