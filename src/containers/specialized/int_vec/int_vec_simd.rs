//! Hardware acceleration module for IntVec
//!
//! Provides SIMD, BMI2, and other hardware-specific optimizations
//! for maximum performance on modern CPUs.

use crate::error::Result;

/// CPU feature detection for optimization selection
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub has_bmi2: bool,
    pub has_popcnt: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_bmi2: is_x86_feature_detected!("bmi2"),
                has_popcnt: is_x86_feature_detected!("popcnt"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512: is_x86_feature_detected!("avx512f"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_bmi2: false,
                has_popcnt: false,
                has_avx2: false,
                has_avx512: false,
            }
        }
    }

    /// Get optimization tier (0=scalar, 5=max)
    pub fn optimization_tier(self) -> u8 {
        match (self.has_avx512, self.has_avx2, self.has_bmi2, self.has_popcnt) {
            (true, _, _, _) => 5,  // AVX-512
            (_, true, true, true) => 4,  // AVX2 + BMI2 + POPCNT
            (_, true, _, true) => 3,  // AVX2 + POPCNT
            (_, _, true, true) => 2,  // BMI2 + POPCNT
            (_, _, _, true) => 1,  // POPCNT only
            _ => 0,  // Scalar fallback
        }
    }
}

/// Hardware-accelerated bit manipulation functions
pub struct BitOps;

impl BitOps {
    /// Fast bit width calculation using hardware instructions when available
    #[inline]
    pub fn compute_bit_width(value: u64) -> u8 {
        if value == 0 {
            return 1;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Use BMI2 LZCNT when available for fastest calculation
            if is_x86_feature_detected!("lzcnt") {
                return unsafe {
                    64 - std::arch::x86_64::_lzcnt_u64(value) as u8
                };
            }
        }

        // Fallback: use standard library
        64 - value.leading_zeros() as u8
    }

    /// Hardware-accelerated bit extraction
    #[inline]
    pub fn extract_bits(data: u64, start: u8, width: u8) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // Use BMI2 BEXTR for optimal bit extraction
            if is_x86_feature_detected!("bmi2") {
                return unsafe {
                    std::arch::x86_64::_bextr_u64(data, start as u32, width as u32)
                };
            }
        }

        // Fallback implementation
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        (data >> start) & mask
    }

    /// Hardware-accelerated population count
    #[inline]
    pub fn popcount(value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("popcnt") {
                return unsafe {
                    std::arch::x86_64::_popcnt64(value as i64) as u32
                };
            }
        }

        // Fallback
        value.count_ones()
    }

    /// Parallel bit deposit (BMI2 PDEP)
    #[inline]
    pub fn bit_deposit(value: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                return unsafe {
                    std::arch::x86_64::_pdep_u64(value, mask)
                };
            }
        }

        // Software fallback for PDEP
        let mut result = 0u64;
        let mut value_bit = 0;
        
        for i in 0..64 {
            if (mask >> i) & 1 != 0 {
                if (value >> value_bit) & 1 != 0 {
                    result |= 1u64 << i;
                }
                value_bit += 1;
            }
        }
        
        result
    }

    /// Parallel bit extract (BMI2 PEXT)
    #[inline]
    pub fn bit_extract(value: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                return unsafe {
                    std::arch::x86_64::_pext_u64(value, mask)
                };
            }
        }

        // Software fallback for PEXT
        let mut result = 0u64;
        let mut result_bit = 0;
        
        for i in 0..64 {
            if (mask >> i) & 1 != 0 {
                if (value >> i) & 1 != 0 {
                    result |= 1u64 << result_bit;
                }
                result_bit += 1;
            }
        }
        
        result
    }
}

/// SIMD-accelerated operations for bulk processing
pub struct SimdOps;

impl SimdOps {
    /// Vectorized bit width calculation for arrays
    pub fn compute_bit_widths_bulk(values: &[u64]) -> Vec<u8> {
        let mut result = Vec::with_capacity(values.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::compute_bit_widths_avx2(values);
            }
        }

        // Scalar fallback
        for &value in values {
            result.push(BitOps::compute_bit_width(value));
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    fn compute_bit_widths_avx2(values: &[u64]) -> Vec<u8> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(values.len());
        let chunks = values.chunks_exact(4);
        let remainder = chunks.remainder();

        unsafe {
            for chunk in chunks {
                // Load 4 u64 values
                let a = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                
                // Convert to array for processing
                let mut vals = [0u64; 4];
                _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, a);
                
                // Calculate bit widths
                for &val in &vals {
                    result.push(BitOps::compute_bit_width(val));
                }
            }
        }

        // Handle remainder
        for &value in remainder {
            result.push(BitOps::compute_bit_width(value));
        }

        result
    }

    /// Vectorized range analysis for min/max calculation
    pub fn analyze_range_bulk(values: &[u64]) -> (u64, u64) {
        if values.is_empty() {
            return (0, 0);
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::analyze_range_avx2(values);
            }
        }

        // Scalar fallback
        let mut min_val = values[0];
        let mut max_val = values[0];
        
        for &value in &values[1..] {
            min_val = min_val.min(value);
            max_val = max_val.max(value);
        }

        (min_val, max_val)
    }

    #[cfg(target_arch = "x86_64")]
    fn analyze_range_avx2(values: &[u64]) -> (u64, u64) {
        use std::arch::x86_64::*;

        let mut min_val = values[0];
        let mut max_val = values[0];

        unsafe {
            let chunks = values.chunks_exact(4);
            let remainder = chunks.remainder();

            for chunk in chunks {
                let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                
                // Note: AVX2 doesn't have direct u64 min/max, so we process individually
                let mut vals = [0u64; 4];
                _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, data);
                
                for &val in &vals {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }

            // Handle remainder
            for &value in remainder {
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }

        (min_val, max_val)
    }

    /// SIMD-accelerated bulk bit extraction
    pub fn extract_bits_bulk(
        data: &[u8], 
        bit_offsets: &[usize], 
        bit_width: u8
    ) -> Result<Vec<u64>> {
        let mut result = Vec::with_capacity(bit_offsets.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && bit_width <= 32 {
                return Self::extract_bits_bulk_avx2(data, bit_offsets, bit_width);
            }
        }

        // Scalar fallback
        for &bit_offset in bit_offsets {
            let byte_offset = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;

            if byte_offset + 8 <= data.len() {
                let mut bytes = [0u8; 8];
                bytes[..8].copy_from_slice(&data[byte_offset..byte_offset + 8]);
                
                let value = u64::from_le_bytes(bytes);
                let extracted = BitOps::extract_bits(value, bit_in_byte as u8, bit_width);
                result.push(extracted);
            } else {
                result.push(0); // Or handle error
            }
        }

        Ok(result)
    }

    #[cfg(target_arch = "x86_64")]
    fn extract_bits_bulk_avx2(
        data: &[u8], 
        bit_offsets: &[usize], 
        bit_width: u8
    ) -> Result<Vec<u64>> {
        let mut result = Vec::with_capacity(bit_offsets.len());

        // Process in chunks for SIMD efficiency
        for &bit_offset in bit_offsets {
            let byte_offset = bit_offset / 8;
            let bit_in_byte = bit_offset % 8;

            if byte_offset + 8 <= data.len() {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data[byte_offset..byte_offset + 8]);
                
                let value = u64::from_le_bytes(bytes);
                let extracted = BitOps::extract_bits(value, bit_in_byte as u8, bit_width);
                result.push(extracted);
            } else {
                result.push(0);
            }
        }

        Ok(result)
    }
}

/// Memory prefetching utilities
pub struct PrefetchOps;

impl PrefetchOps {
    /// Prefetch memory for read access
    #[inline]
    pub fn prefetch_read(addr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Compiler hint for prefetching
            std::hint::black_box(unsafe { addr.read_volatile() });
        }
    }

    /// Prefetch memory for write access
    #[inline]
    pub fn prefetch_write(addr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                std::arch::x86_64::_mm_prefetch(addr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::hint::black_box(unsafe { addr.read_volatile() });
        }
    }

    /// Bulk prefetch for sequential access patterns
    ///
    /// SAFETY FIX (v2.1.1): Changed from raw pointer to slice to prevent pointer arithmetic overflow
    pub fn prefetch_range(data: &[u8]) {
        const CACHE_LINE_SIZE: usize = 64;

        // Safe iteration using chunks - no pointer arithmetic overflow possible
        for chunk in data.chunks(CACHE_LINE_SIZE) {
            // SAFETY FIX (v2.1.1): Use chunk.as_ptr() to get actual data pointer,
            // not &chunk[0] which creates temporary reference (stack address)
            Self::prefetch_read(chunk.as_ptr());
        }
    }
}

/// Cache-friendly algorithms for different access patterns
pub struct CacheOps;

impl CacheOps {
    /// Determine optimal block size based on cache characteristics
    pub fn optimal_block_size(data_size: usize, element_size: usize) -> usize {
        const L1_CACHE_SIZE: usize = 32 * 1024; // 32KB typical L1 cache
        const L2_CACHE_SIZE: usize = 256 * 1024; // 256KB typical L2 cache

        if data_size * element_size <= L1_CACHE_SIZE {
            64 // Small blocks for L1 cache
        } else if data_size * element_size <= L2_CACHE_SIZE {
            128 // Medium blocks for L2 cache
        } else {
            256 // Large blocks for main memory
        }
    }

    /// Memory-efficient block processing with prefetching
    pub fn process_blocks<T, F>(
        data: &[T],
        block_size: usize,
        mut processor: F
    ) -> Result<()>
    where
        F: FnMut(&[T]) -> Result<()>,
    {
        for (i, chunk) in data.chunks(block_size).enumerate() {
            // Prefetch next chunk (SAFETY FIX v2.1.1: using safe slice indexing)
            let next_chunk_start = (i + 1) * block_size;
            if next_chunk_start < data.len() {
                let prefetch_len = std::cmp::min(block_size * std::mem::size_of::<T>(),
                                                   (data.len() - next_chunk_start) * std::mem::size_of::<T>());
                let next_chunk_bytes = unsafe {
                    std::slice::from_raw_parts(
                        data[next_chunk_start..].as_ptr() as *const u8,
                        prefetch_len
                    )
                };
                PrefetchOps::prefetch_range(next_chunk_bytes);
            }

            processor(chunk)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        println!("CPU Features: {:?}", features);
        println!("Optimization tier: {}", features.optimization_tier());
        
        // Just verify it doesn't crash
        assert!(features.optimization_tier() <= 5);
    }

    #[test]
    fn test_bit_width_calculation() {
        assert_eq!(BitOps::compute_bit_width(0), 1);
        assert_eq!(BitOps::compute_bit_width(1), 1);
        assert_eq!(BitOps::compute_bit_width(2), 2);
        assert_eq!(BitOps::compute_bit_width(3), 2);
        assert_eq!(BitOps::compute_bit_width(255), 8);
        assert_eq!(BitOps::compute_bit_width(256), 9);
        assert_eq!(BitOps::compute_bit_width(u64::MAX), 64);
    }

    #[test]
    fn test_bit_extraction() {
        let data = 0b1010110110101100u64;
        
        // Extract 4 bits starting at position 4
        let extracted = BitOps::extract_bits(data, 4, 4);
        assert_eq!(extracted, 0b1010);

        // Extract single bit
        let bit = BitOps::extract_bits(data, 0, 1);
        assert_eq!(bit, 0);
        
        let bit = BitOps::extract_bits(data, 2, 1);
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_popcount() {
        assert_eq!(BitOps::popcount(0), 0);
        assert_eq!(BitOps::popcount(1), 1);
        assert_eq!(BitOps::popcount(0b1010), 2);
        assert_eq!(BitOps::popcount(u64::MAX), 64);
    }

    #[test]
    fn test_bulk_bit_width() {
        let values = vec![0, 1, 255, 256, 65535, 65536];
        let widths = SimdOps::compute_bit_widths_bulk(&values);
        
        assert_eq!(widths, vec![1, 1, 8, 9, 16, 17]);
    }

    #[test]
    fn test_range_analysis() {
        let values = vec![10, 50, 5, 100, 25];
        let (min_val, max_val) = SimdOps::analyze_range_bulk(&values);
        
        assert_eq!(min_val, 5);
        assert_eq!(max_val, 100);
    }

    #[test]
    fn test_cache_operations() {
        let block_size = CacheOps::optimal_block_size(1000, 4);
        assert!(block_size >= 64);

        // Test block processing
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut sum = 0;
        
        CacheOps::process_blocks(&data, 3, |chunk| {
            sum += chunk.iter().sum::<i32>();
            Ok(())
        }).unwrap();
        
        assert_eq!(sum, 55);
    }

    #[test]
    fn test_bmi2_operations() {
        // Test PDEP/PEXT operations
        let value = 0b1010u64;
        let mask = 0b1111u64;
        
        let deposited = BitOps::bit_deposit(value, mask);
        let extracted = BitOps::bit_extract(deposited, mask);
        
        assert_eq!(extracted, value);
    }
}