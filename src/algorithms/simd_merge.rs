//! SIMD-Optimized Operations for Multi-way Merge Algorithms
//!
//! This module provides hardware-accelerated implementations of comparison and merge
//! operations using SIMD (Single Instruction, Multiple Data) instructions:
//!
//! - **AVX2/BMI2 acceleration**: Optimized for x86_64 with modern instruction sets
//! - **Vectorized comparisons**: Process multiple elements simultaneously
//! - **Cache-friendly algorithms**: Optimized memory access patterns
//! - **Fallback implementations**: Safe fallbacks for unsupported hardware

use crate::error::{Result, ZiporaError};
use std::cmp::Ordering;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable AVX-512 instructions when available
    pub use_avx512: bool,
    /// Enable AVX2 instructions when available
    pub use_avx2: bool,
    /// Enable BMI2 instructions when available
    pub use_bmi2: bool,
    /// Minimum vector size for SIMD operations
    pub min_vector_size: usize,
    /// Cache line prefetch distance
    pub prefetch_distance: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            use_avx512: is_x86_feature_detected!("avx512f"),
            #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
            use_avx512: false,
            use_avx2: is_x86_feature_detected!("avx2"),
            use_bmi2: is_x86_feature_detected!("bmi2"),
            min_vector_size: 8,
            prefetch_distance: 2,
        }
    }
}

/// SIMD-optimized comparison operations
pub struct SimdComparator {
    config: SimdConfig,
}

impl SimdComparator {
    /// Create a new SIMD comparator
    pub fn new() -> Self {
        Self::with_config(SimdConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Compare two slices of i32 values using SIMD when possible
    pub fn compare_i32_slices(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        if left.len() != right.len() {
            return Err(ZiporaError::invalid_parameter(
                "Input slices must have equal length",
            ));
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if self.config.use_avx512
            && !cfg!(miri)
            && is_x86_feature_detected!("avx512f")
            && left.len() >= 16
        {
            // SAFETY: AVX-512F availability is verified at runtime via `is_x86_feature_detected!("avx512f")`.
            return unsafe { self.compare_i32_simd_avx512(left, right) };
        }

        if self.config.use_avx2 && left.len() >= self.config.min_vector_size {
            self.compare_i32_simd(left, right)
        } else {
            Ok(left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a.cmp(b))
                .collect())
        }
    }

    /// AVX-512 optimized comparison for i32 arrays
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn compare_i32_simd_avx512(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        let mut results = Vec::with_capacity(left.len());
        let chunk_size = 16;
        let mut i = 0;

        while i + chunk_size <= left.len() {
            // SAFETY: AVX-512F guaranteed by #[target_feature(enable = "avx512f")].
            // Pointer arithmetic bounded by while condition: i + chunk_size <= left.len().
            unsafe {
                let left_ptr = left.as_ptr().add(i);
                let right_ptr = right.as_ptr().add(i);

                #[cfg(not(miri))]
                if let Some(prefetch_offset) = chunk_size
                    .checked_mul(self.config.prefetch_distance)
                    .filter(|&d| d > 0)
                    && i.checked_add(prefetch_offset)
                        .is_some_and(|idx| idx < left.len())
                {
                    _mm_prefetch(left_ptr.add(prefetch_offset) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(right_ptr.add(prefetch_offset) as *const i8, _MM_HINT_T0);
                }

                let left_vec = _mm512_loadu_si512(left_ptr as *const __m512i);
                let right_vec = _mm512_loadu_si512(right_ptr as *const __m512i);

                let eq_mask = _mm512_cmpeq_epi32_mask(left_vec, right_vec);
                let gt_mask = _mm512_cmpgt_epi32_mask(left_vec, right_vec);

                for j in 0..chunk_size {
                    let ordering = if (eq_mask >> j) & 1 != 0 {
                        Ordering::Equal
                    } else if (gt_mask >> j) & 1 != 0 {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };
                    results.push(ordering);
                }

                i += chunk_size;
            }
        }

        while i < left.len() {
            results.push(left[i].cmp(&right[i]));
            i += 1;
        }

        Ok(results)
    }

    /// AVX2-optimized comparison for i32 arrays
    #[cfg(target_arch = "x86_64")]
    fn compare_i32_simd(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        if !is_x86_feature_detected!("avx2") {
            return Ok(left
                .iter()
                .zip(right.iter())
                .map(|(a, b)| a.cmp(b))
                .collect());
        }

        // SAFETY: `is_x86_feature_detected!("avx2")` was verified above.
        unsafe { self.compare_i32_simd_avx2(left, right) }
    }

    /// AVX2 inner implementation for i32 slice comparison.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx2`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn compare_i32_simd_avx2(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        let mut results = Vec::with_capacity(left.len());
        let chunk_size = 8; // AVX2 can process 8 i32s at once

        // SAFETY: AVX2 feature is guaranteed by caller and `#[target_feature(enable = "avx2")]`,
        // and all pointer arithmetic is bounds-checked below.
        unsafe {
            let mut i = 0;

            // Process chunks of 8 elements with AVX2
            while i + chunk_size <= left.len() {
                let left_ptr = left.as_ptr().add(i);
                let right_ptr = right.as_ptr().add(i);

                // Prefetch next chunk if configured
                #[cfg(not(miri))]
                if let Some(prefetch_offset) = chunk_size
                    .checked_mul(self.config.prefetch_distance)
                    .filter(|&d| d > 0)
                    && i.checked_add(prefetch_offset)
                        .is_some_and(|idx| idx < left.len())
                {
                    let prefetch_left = left_ptr.add(prefetch_offset);
                    let prefetch_right = right_ptr.add(prefetch_offset);
                    _mm_prefetch(prefetch_left as *const i8, _MM_HINT_T0);
                    _mm_prefetch(prefetch_right as *const i8, _MM_HINT_T0);
                }

                // Load 8 i32 values into AVX2 registers
                let left_vec = _mm256_loadu_si256(left_ptr as *const __m256i);
                let right_vec = _mm256_loadu_si256(right_ptr as *const __m256i);

                // Compare vectors
                let eq_mask = _mm256_cmpeq_epi32(left_vec, right_vec);
                let gt_mask = _mm256_cmpgt_epi32(left_vec, right_vec);

                // Extract comparison results
                let eq_bits = _mm256_movemask_epi8(eq_mask) as u32;
                let gt_bits = _mm256_movemask_epi8(gt_mask) as u32;

                // Convert bit masks to ordering results
                for j in 0..chunk_size {
                    let bit_offset = j * 4; // Each i32 uses 4 bytes
                    let eq_bit = (eq_bits >> bit_offset) & 0xF;
                    let gt_bit = (gt_bits >> bit_offset) & 0xF;

                    let ordering = if eq_bit != 0 {
                        Ordering::Equal
                    } else if gt_bit != 0 {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    };

                    results.push(ordering);
                }

                i += chunk_size;
            }

            // Handle remaining elements with scalar comparison
            while i < left.len() {
                results.push(left[i].cmp(&right[i]));
                i += 1;
            }
        }

        Ok(results)
    }

    /// Fallback implementation for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    fn compare_i32_simd(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        Ok(left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a.cmp(b))
            .collect())
    }

    /// Find minimum value in an array using SIMD
    pub fn find_min_i32(&self, values: &[i32]) -> Option<(usize, i32)> {
        if values.is_empty() {
            return None;
        }

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if self.config.use_avx512
            && !cfg!(miri)
            && is_x86_feature_detected!("avx512f")
            && values.len() >= 16
        {
            // SAFETY: AVX-512F availability is verified at runtime via `is_x86_feature_detected!("avx512f")`.
            return unsafe { self.find_min_i32_simd_avx512(values) };
        }

        if self.config.use_avx2 && values.len() >= self.config.min_vector_size {
            self.find_min_i32_simd(values)
        } else {
            values
                .iter()
                .enumerate()
                .min_by_key(|(_, val)| *val)
                .map(|(idx, val)| (idx, *val))
        }
    }

    /// AVX-512 optimized minimum finding for i32 arrays.
    /// Uses _mm512_min_epi32 tree reduction + cmpeq mask to find both value and index.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn find_min_i32_simd_avx512(&self, values: &[i32]) -> Option<(usize, i32)> {
        if values.is_empty() {
            return None;
        }

        let chunk_size = 16;
        let mut global_min = i32::MAX;
        let mut global_min_idx = 0;
        let mut i = 0;

        // SAFETY: AVX-512F guaranteed by #[target_feature(enable = "avx512f")].
        // Pointer arithmetic bounded by while condition: i + chunk_size <= values.len().
        let mut min_vec = _mm512_set1_epi32(i32::MAX);

        while i + chunk_size <= values.len() {
            // SAFETY: AVX-512F guaranteed by #[target_feature(enable = "avx512f")], i + 16 <= values.len() by while condition
            unsafe {
                let vec = _mm512_loadu_si512(values.as_ptr().add(i) as *const __m512i);
                let new_min = _mm512_min_epi32(min_vec, vec);

                if _mm512_cmpeq_epi32_mask(new_min, min_vec) != 0xFFFF {
                    // min changed — find which lane has the new minimum
                    let chunk_min = _mm512_reduce_min_epi32(vec);
                    if chunk_min < global_min {
                        global_min = chunk_min;
                        let min_broadcast = _mm512_set1_epi32(chunk_min);
                        let mask = _mm512_cmpeq_epi32_mask(vec, min_broadcast);
                        global_min_idx = i + mask.trailing_zeros() as usize;
                    }
                    min_vec = new_min;
                }

                i += chunk_size;
            }
        }

        while i < values.len() {
            if values[i] < global_min {
                global_min = values[i];
                global_min_idx = i;
            }
            i += 1;
        }

        Some((global_min_idx, global_min))
    }

    /// AVX2-optimized minimum finding for i32 arrays
    #[cfg(target_arch = "x86_64")]
    fn find_min_i32_simd(&self, values: &[i32]) -> Option<(usize, i32)> {
        if !is_x86_feature_detected!("avx2") || values.is_empty() {
            return values
                .iter()
                .enumerate()
                .min_by_key(|(_, val)| *val)
                .map(|(idx, val)| (idx, *val));
        }

        // SAFETY: `is_x86_feature_detected!("avx2")` was verified above.
        unsafe { self.find_min_i32_simd_avx2(values) }
    }

    /// AVX2 inner implementation for minimum finding.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx2` and `values` is non-empty.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_min_i32_simd_avx2(&self, values: &[i32]) -> Option<(usize, i32)> {
        // SAFETY: AVX2 feature is guaranteed by caller and `#[target_feature(enable = "avx2")]`,
        // and pointer arithmetic is bounds-checked before AVX2 intrinsics.
        unsafe {
            let chunk_size = 8;
            let mut global_min = i32::MAX;
            let mut global_min_idx = 0;

            let mut i = 0;

            // Process chunks with AVX2
            while i + chunk_size <= values.len() {
                // SAFETY: loop condition guarantees i + chunk_size <= values.len()
                let ptr = values.as_ptr().add(i);
                let vec = _mm256_loadu_si256(ptr as *const __m256i);

                // Find minimum in this chunk
                let mut chunk_values = [0i32; 8];
                _mm256_storeu_si256(chunk_values.as_mut_ptr() as *mut __m256i, vec);

                for (j, &val) in chunk_values.iter().enumerate() {
                    if val < global_min {
                        global_min = val;
                        global_min_idx = i + j;
                    }
                }

                i += chunk_size;
            }

            // Handle remaining elements
            while i < values.len() {
                if values[i] < global_min {
                    global_min = values[i];
                    global_min_idx = i;
                }
                i += 1;
            }

            Some((global_min_idx, global_min))
        }
    }

    /// Fallback implementation for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    fn find_min_i32_simd(&self, values: &[i32]) -> Option<(usize, i32)> {
        values
            .iter()
            .enumerate()
            .min_by_key(|(_, val)| *val)
            .map(|(idx, val)| (idx, *val))
    }

    /// Merge two sorted i32 arrays using SIMD optimizations
    pub fn merge_sorted_i32(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if self.config.use_avx512
            && !cfg!(miri)
            && is_x86_feature_detected!("avx512f")
            && (left.len() + right.len()) >= 32
        {
            // SAFETY: AVX-512F availability is verified at runtime via `is_x86_feature_detected!("avx512f")`.
            return unsafe { self.merge_sorted_i32_simd_avx512(left, right) };
        }

        if self.config.use_avx2
            && (left.len() + right.len()) >= self.config.min_vector_size.saturating_mul(2)
        {
            self.merge_sorted_i32_simd(left, right)
        } else {
            self.merge_sorted_i32_scalar(left, right)
        }
    }

    /// AVX-512 merge for sorted i32 arrays with vector block copy on non-overlapping runs.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn merge_sorted_i32_simd_avx512(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        let mut result = Vec::with_capacity(left.len() + right.len());
        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left.len() && right_idx < right.len() {
            if left_idx + 16 <= left.len() && left[left_idx + 15] <= right[right_idx] {
                // SAFETY: AVX-512F is enabled and `left_idx + 16 <= left.len()`.
                unsafe {
                    self.simd_copy_i32_avx512(&left[left_idx..left_idx + 16], &mut result);
                }
                left_idx += 16;
            } else if right_idx + 16 <= right.len() && right[right_idx + 15] < left[left_idx] {
                // SAFETY: AVX-512F is enabled and `right_idx + 16 <= right.len()`.
                unsafe {
                    self.simd_copy_i32_avx512(&right[right_idx..right_idx + 16], &mut result);
                }
                right_idx += 16;
            } else if left[left_idx] <= right[right_idx] {
                result.push(left[left_idx]);
                left_idx += 1;
            } else {
                result.push(right[right_idx]);
                right_idx += 1;
            }
        }

        // SAFETY: AVX-512F guaranteed by #[target_feature(enable = "avx512f")]
        unsafe {
            if left_idx < left.len() {
                self.simd_copy_i32_avx512(&left[left_idx..], &mut result);
            }
            if right_idx < right.len() {
                self.simd_copy_i32_avx512(&right[right_idx..], &mut result);
            }
        }

        result
    }

    /// SIMD-optimized merge for sorted i32 arrays
    #[cfg(target_arch = "x86_64")]
    fn merge_sorted_i32_simd(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        if !is_x86_feature_detected!("avx2") {
            return self.merge_sorted_i32_scalar(left, right);
        }

        let mut result = Vec::with_capacity(left.len() + right.len());
        let mut left_idx = 0;
        let mut right_idx = 0;

        // Use AVX2 8-element block copy when an entire 8-element run precedes the other stream's head
        while left_idx < left.len() && right_idx < right.len() {
            if left_idx + 8 <= left.len() && left[left_idx + 7] <= right[right_idx] {
                // SAFETY: AVX2 is verified above and slice is in bounds.
                unsafe {
                    self.simd_copy_i32(&left[left_idx..left_idx + 8], &mut result);
                }
                left_idx += 8;
            } else if right_idx + 8 <= right.len() && right[right_idx + 7] < left[left_idx] {
                // SAFETY: AVX2 is verified above and slice is in bounds.
                unsafe {
                    self.simd_copy_i32(&right[right_idx..right_idx + 8], &mut result);
                }
                right_idx += 8;
            } else if left[left_idx] <= right[right_idx] {
                result.push(left[left_idx]);
                left_idx += 1;
            } else {
                result.push(right[right_idx]);
                right_idx += 1;
            }
        }

        // Use SIMD for copying remaining elements
        // SAFETY: AVX2 is verified above and `simd_copy_i32` performs bounds-checked operations on slices.
        unsafe {
            if left_idx < left.len() {
                let remaining = &left[left_idx..];
                self.simd_copy_i32(remaining, &mut result);
            }

            if right_idx < right.len() {
                let remaining = &right[right_idx..];
                self.simd_copy_i32(remaining, &mut result);
            }
        }

        result
    }

    /// AVX-512 optimized array copying.
    ///
    /// Invariant: `reserve(src.len())` must be called before `as_mut_ptr()`, and
    /// no Vec reallocation may occur between `as_mut_ptr()` and `set_len()`.
    /// The subsequent `push` calls for remainder elements are safe because
    /// `reserve(src.len())` guarantees capacity for all elements.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn simd_copy_i32_avx512(&self, src: &[i32], dest: &mut Vec<i32>) {
        let chunk_size = 16;
        let mut i = 0;

        dest.reserve(src.len());

        // SAFETY: reserve() ensures capacity for all src elements.
        // dest_ptr is obtained AFTER reserve, so no reallocation occurs during SIMD stores.
        unsafe {
            let dest_ptr = dest.as_mut_ptr().add(dest.len());

            while i + chunk_size <= src.len() {
                let vec = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
                _mm512_storeu_si512(dest_ptr.add(i) as *mut __m512i, vec);
                i += chunk_size;
            }

            dest.set_len(dest.len() + i);
        }

        while i < src.len() {
            dest.push(src[i]);
            i += 1;
        }
    }

    /// SIMD-optimized array copying.
    ///
    /// # Safety
    /// Caller must ensure the CPU supports `avx2`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn simd_copy_i32(&self, src: &[i32], dest: &mut Vec<i32>) {
        let chunk_size = 8;
        let mut i = 0;

        // Reserve space
        dest.reserve(src.len());
        // SAFETY: dest.len() is within allocated capacity after reserve() call above
        let dest_ptr = unsafe { dest.as_mut_ptr().add(dest.len()) };

        // Copy chunks with AVX2
        while i + chunk_size <= src.len() {
            // SAFETY: i + chunk_size <= src.len() guarantees src.as_ptr().add(i) is in bounds
            let src_ptr = unsafe { src.as_ptr().add(i) };
            let vec = unsafe { _mm256_loadu_si256(src_ptr as *const __m256i) };
            // SAFETY: dest has reserved space, dest_ptr.add(i) is within allocated capacity
            unsafe { _mm256_storeu_si256(dest_ptr.add(i) as *mut __m256i, vec) };
            i += chunk_size;
        }

        // Update vector length
        // SAFETY: We've written i elements to dest starting at dest.len(), so dest.len() + i is valid
        unsafe { dest.set_len(dest.len() + i) };

        // Copy remaining elements
        while i < src.len() {
            dest.push(src[i]);
            i += 1;
        }
    }

    /// Fallback SIMD merge for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    fn merge_sorted_i32_simd(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        self.merge_sorted_i32_scalar(left, right)
    }

    /// Scalar merge implementation
    fn merge_sorted_i32_scalar(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        let mut result = Vec::with_capacity(left.len() + right.len());
        let mut left_iter = left.iter();
        let mut right_iter = right.iter();

        let mut left_current = left_iter.next();
        let mut right_current = right_iter.next();

        loop {
            match (left_current, right_current) {
                (Some(l), Some(r)) => {
                    if l <= r {
                        result.push(*l);
                        left_current = left_iter.next();
                    } else {
                        result.push(*r);
                        right_current = right_iter.next();
                    }
                }
                (Some(l), None) => {
                    result.push(*l);
                    result.extend(left_iter.copied());
                    break;
                }
                (None, Some(r)) => {
                    result.push(*r);
                    result.extend(right_iter.copied());
                    break;
                }
                (None, None) => break,
            }
        }

        result
    }

    /// Check if SIMD optimizations are available
    pub fn simd_available(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.config.use_avx2 && is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &SimdConfig {
        &self.config
    }
}

impl Default for SimdComparator {
    fn default() -> Self {
        Self::new()
    }
}

/// Specialized SIMD operations for different data types
pub struct SimdOperations;

impl SimdOperations {
    /// Parallel comparison of multiple value pairs
    pub fn parallel_compare_i32(pairs: &[(i32, i32)]) -> Vec<Ordering> {
        let comparator = SimdComparator::new();

        if pairs.is_empty() {
            return Vec::new();
        }

        let (left, right): (Vec<i32>, Vec<i32>) = pairs.iter().copied().unzip();

        comparator
            .compare_i32_slices(&left, &right)
            .unwrap_or_else(|_| pairs.iter().map(|(a, b)| a.cmp(b)).collect())
    }

    /// Find indices of minimum values in multiple arrays
    pub fn find_multiple_mins(arrays: &[&[i32]]) -> Vec<Option<(usize, i32)>> {
        let comparator = SimdComparator::new();

        arrays
            .iter()
            .map(|arr| comparator.find_min_i32(arr))
            .collect()
    }

    /// Merge multiple sorted arrays using SIMD optimizations
    pub fn merge_multiple_sorted(arrays: Vec<Vec<i32>>) -> Vec<i32> {
        if arrays.len() <= 1 {
            return arrays.into_iter().next().unwrap_or_default();
        }

        let comparator = SimdComparator::new();

        // Binary merge tree approach
        let mut current_arrays = arrays;

        while current_arrays.len() > 1 {
            let mut next_arrays = Vec::with_capacity(current_arrays.len().div_ceil(2));
            let mut iter = current_arrays.into_iter();

            while let Some(first) = iter.next() {
                if let Some(second) = iter.next() {
                    next_arrays.push(comparator.merge_sorted_i32(&first, &second));
                } else {
                    next_arrays.push(first);
                }
            }

            current_arrays = next_arrays;
        }

        current_arrays.into_iter().next().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_comparator_creation() {
        let comparator = SimdComparator::new();
        assert!(comparator.config().min_vector_size > 0);
    }

    #[test]
    fn test_compare_i32_slices() {
        let comparator = SimdComparator::new();

        let left = vec![1, 5, 3, 8, 2];
        let right = vec![2, 4, 3, 6, 1];

        let result = comparator.compare_i32_slices(&left, &right).unwrap();

        assert_eq!(
            result,
            vec![
                Ordering::Less,    // 1 < 2
                Ordering::Greater, // 5 > 4
                Ordering::Equal,   // 3 == 3
                Ordering::Greater, // 8 > 6
                Ordering::Greater, // 2 > 1
            ]
        );
    }

    #[test]
    fn test_find_min_i32() {
        let comparator = SimdComparator::new();

        let values = vec![5, 2, 8, 1, 9, 3];
        let result = comparator.find_min_i32(&values).unwrap();

        assert_eq!(result, (3, 1));
    }

    #[test]
    fn test_find_min_empty() {
        let comparator = SimdComparator::new();
        let result = comparator.find_min_i32(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_merge_sorted_i32() {
        let comparator = SimdComparator::new();

        let left = vec![1, 3, 5, 7];
        let right = vec![2, 4, 6, 8];

        let result = comparator.merge_sorted_i32(&left, &right);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_merge_uneven_arrays() {
        let comparator = SimdComparator::new();

        let left = vec![1, 5, 9];
        let right = vec![2, 3, 4, 6, 7, 8];

        let result = comparator.merge_sorted_i32(&left, &right);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_parallel_compare() {
        let pairs = vec![(1, 2), (5, 3), (4, 4), (9, 7)];
        let result = SimdOperations::parallel_compare_i32(&pairs);

        assert_eq!(
            result,
            vec![
                Ordering::Less,
                Ordering::Greater,
                Ordering::Equal,
                Ordering::Greater,
            ]
        );
    }

    #[test]
    fn test_find_multiple_mins() {
        let arr1 = [5, 2, 8, 1];
        let arr2 = [9, 3, 7, 4];
        let arr3 = [6];

        let arrays = [&arr1[..], &arr2[..], &arr3[..]];
        let result = SimdOperations::find_multiple_mins(&arrays);

        assert_eq!(result, vec![Some((3, 1)), Some((1, 3)), Some((0, 6)),]);
    }

    #[test]
    fn test_merge_multiple_sorted() {
        let arrays = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];

        let result = SimdOperations::merge_multiple_sorted(arrays);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_simd_available() {
        let comparator = SimdComparator::new();
        // Should not panic
        let _available = comparator.simd_available();
    }

    #[test]
    fn test_mismatched_slice_lengths() {
        let comparator = SimdComparator::new();

        let left = vec![1, 2, 3];
        let right = vec![1, 2];

        let result = comparator.compare_i32_slices(&left, &right);
        assert!(result.is_err());
    }

    #[test]
    fn test_large_array_simd_path() {
        let config = SimdConfig {
            min_vector_size: 4, // Lower threshold for testing
            ..Default::default()
        };

        let comparator = SimdComparator::with_config(config);

        let left: Vec<i32> = (0..16).collect();
        let right: Vec<i32> = (1..17).collect();

        let result = comparator.compare_i32_slices(&left, &right).unwrap();

        // All should be Less since left[i] < right[i] for all i
        assert!(result.iter().all(|&ord| ord == Ordering::Less));
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_c8_7_simd_comparator_forced_config_and_overflow_prefetch() {
        let forced = SimdComparator::with_config(SimdConfig {
            use_avx512: true,
            use_avx2: true,
            use_bmi2: true,
            min_vector_size: 1,
            prefetch_distance: usize::MAX / 4,
        });
        let scalar = SimdComparator::with_config(SimdConfig {
            use_avx512: false,
            use_avx2: false,
            use_bmi2: false,
            min_vector_size: usize::MAX,
            prefetch_distance: 0,
        });

        let left: Vec<i32> = (0..40).map(|i| i * 3 - 20).collect();
        let right: Vec<i32> = (0..40).map(|i| i * 2 + 5).collect();

        // Previously overflowed `chunk_size * prefetch_distance` and dispatched to
        // `#[target_feature(enable = "avx512f")]` without checking runtime feature / Miri.
        assert_eq!(
            forced.compare_i32_slices(&left, &right).unwrap(),
            scalar.compare_i32_slices(&left, &right).unwrap()
        );
        assert_eq!(forced.find_min_i32(&left), scalar.find_min_i32(&left));
        assert_eq!(
            forced.merge_sorted_i32(&left, &right),
            scalar.merge_sorted_i32(&left, &right)
        );
    }
}
