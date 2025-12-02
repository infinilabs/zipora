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
                "Input slices must have equal length"
            ));
        }

        if self.config.use_avx2 && left.len() >= self.config.min_vector_size {
            self.compare_i32_simd(left, right)
        } else {
            Ok(left.iter().zip(right.iter()).map(|(a, b)| a.cmp(b)).collect())
        }
    }

    /// AVX2-optimized comparison for i32 arrays
    #[cfg(target_arch = "x86_64")]
    fn compare_i32_simd(&self, left: &[i32], right: &[i32]) -> Result<Vec<Ordering>> {
        if !is_x86_feature_detected!("avx2") {
            return Ok(left.iter().zip(right.iter()).map(|(a, b)| a.cmp(b)).collect());
        }

        let mut results = Vec::with_capacity(left.len());
        let chunk_size = 8; // AVX2 can process 8 i32s at once

        unsafe {
            let mut i = 0;
            
            // Process chunks of 8 elements with AVX2
            while i + chunk_size <= left.len() {
                let left_ptr = left.as_ptr().add(i);
                let right_ptr = right.as_ptr().add(i);

                // Prefetch next chunk if configured
                if self.config.prefetch_distance > 0 && i + chunk_size * self.config.prefetch_distance < left.len() {
                    let prefetch_left = left_ptr.add(chunk_size * self.config.prefetch_distance);
                    let prefetch_right = right_ptr.add(chunk_size * self.config.prefetch_distance);
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
        Ok(left.iter().zip(right.iter()).map(|(a, b)| a.cmp(b)).collect())
    }

    /// Find minimum value in an array using SIMD
    pub fn find_min_i32(&self, values: &[i32]) -> Option<(usize, i32)> {
        if values.is_empty() {
            return None;
        }

        if self.config.use_avx2 && values.len() >= self.config.min_vector_size {
            self.find_min_i32_simd(values)
        } else {
            values.iter().enumerate().min_by_key(|(_, val)| *val).map(|(idx, val)| (idx, *val))
        }
    }

    /// AVX2-optimized minimum finding for i32 arrays
    #[cfg(target_arch = "x86_64")]
    fn find_min_i32_simd(&self, values: &[i32]) -> Option<(usize, i32)> {
        if !is_x86_feature_detected!("avx2") || values.is_empty() {
            return values.iter().enumerate().min_by_key(|(_, val)| *val).map(|(idx, val)| (idx, *val));
        }

        unsafe {
            let chunk_size = 8;
            let mut global_min = i32::MAX;
            let mut global_min_idx = 0;

            let mut i = 0;
            
            // Process chunks with AVX2
            while i + chunk_size <= values.len() {
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
        values.iter().enumerate().min_by_key(|(_, val)| *val).map(|(idx, val)| (idx, *val))
    }

    /// Merge two sorted i32 arrays using SIMD optimizations
    pub fn merge_sorted_i32(&self, left: &[i32], right: &[i32]) -> Vec<i32> {
        if self.config.use_avx2 && (left.len() + right.len()) >= self.config.min_vector_size * 2 {
            self.merge_sorted_i32_simd(left, right)
        } else {
            self.merge_sorted_i32_scalar(left, right)
        }
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

        // Use SIMD for bulk copying when one array is exhausted
        while left_idx < left.len() && right_idx < right.len() {
            // For now, use scalar comparison - full SIMD merge is complex
            if left[left_idx] <= right[right_idx] {
                result.push(left[left_idx]);
                left_idx += 1;
            } else {
                result.push(right[right_idx]);
                right_idx += 1;
            }
        }

        // Use SIMD for copying remaining elements
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

    /// SIMD-optimized array copying
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_copy_i32(&self, src: &[i32], dest: &mut Vec<i32>) {
        let chunk_size = 8;
        let mut i = 0;

        // Reserve space
        dest.reserve(src.len());
        let dest_ptr = unsafe { dest.as_mut_ptr().add(dest.len()) };

        // Copy chunks with AVX2
        while i + chunk_size <= src.len() {
            let src_ptr = unsafe { src.as_ptr().add(i) };
            let vec = unsafe { _mm256_loadu_si256(src_ptr as *const __m256i) };
            unsafe { _mm256_storeu_si256(dest_ptr.add(i) as *mut __m256i, vec) };
            i += chunk_size;
        }

        // Update vector length
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
        
        comparator.compare_i32_slices(&left, &right).unwrap_or_else(|_| {
            pairs.iter().map(|(a, b)| a.cmp(b)).collect()
        })
    }

    /// Find indices of minimum values in multiple arrays
    pub fn find_multiple_mins(arrays: &[&[i32]]) -> Vec<Option<(usize, i32)>> {
        let comparator = SimdComparator::new();
        
        arrays.iter().map(|arr| comparator.find_min_i32(arr)).collect()
    }

    /// Merge multiple sorted arrays using SIMD optimizations
    pub fn merge_multiple_sorted(arrays: Vec<Vec<i32>>) -> Vec<i32> {
        if arrays.is_empty() {
            return Vec::new();
        }

        if arrays.len() == 1 {
            // SAFETY: len() == 1 check above guarantees exactly one element
            return arrays.into_iter().next().unwrap();
        }

        let comparator = SimdComparator::new();
        
        // Binary merge tree approach
        let mut current_arrays = arrays;
        
        while current_arrays.len() > 1 {
            let mut next_arrays = Vec::new();
            
            let mut i = 0;
            while i + 1 < current_arrays.len() {
                let merged = comparator.merge_sorted_i32(&current_arrays[i], &current_arrays[i + 1]);
                next_arrays.push(merged);
                i += 2;
            }
            
            // Handle odd number of arrays
            if i < current_arrays.len() {
                // SAFETY: i < current_arrays.len() check above guarantees nth(i) succeeds
                next_arrays.push(current_arrays.into_iter().nth(i).unwrap());
            }

            current_arrays = next_arrays;
        }

        // SAFETY: After while loop, len() <= 1. Empty case returned early at line 382-384, so len() == 1
        current_arrays.into_iter().next().unwrap()
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
        
        assert_eq!(result, vec![
            Ordering::Less,    // 1 < 2
            Ordering::Greater, // 5 > 4
            Ordering::Equal,   // 3 == 3
            Ordering::Greater, // 8 > 6
            Ordering::Greater, // 2 > 1
        ]);
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
        
        assert_eq!(result, vec![
            Ordering::Less,
            Ordering::Greater,
            Ordering::Equal,
            Ordering::Greater,
        ]);
    }

    #[test]
    fn test_find_multiple_mins() {
        let arr1 = vec![5, 2, 8, 1];
        let arr2 = vec![9, 3, 7, 4];
        let arr3 = vec![6];
        
        let arrays = vec![&arr1[..], &arr2[..], &arr3[..]];
        let result = SimdOperations::find_multiple_mins(&arrays);
        
        assert_eq!(result, vec![
            Some((3, 1)),
            Some((1, 3)),
            Some((0, 6)),
        ]);
    }

    #[test]
    fn test_merge_multiple_sorted() {
        let arrays = vec![
            vec![1, 4, 7],
            vec![2, 5, 8],
            vec![3, 6, 9],
        ];
        
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
        let mut config = SimdConfig::default();
        config.min_vector_size = 4; // Lower threshold for testing
        
        let comparator = SimdComparator::with_config(config);
        
        let left: Vec<i32> = (0..16).collect();
        let right: Vec<i32> = (1..17).collect();
        
        let result = comparator.compare_i32_slices(&left, &right).unwrap();
        
        // All should be Less since left[i] < right[i] for all i
        assert!(result.iter().all(|&ord| ord == Ordering::Less));
        assert_eq!(result.len(), 16);
    }
}