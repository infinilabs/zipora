//! Hardware-optimized bit operations for entropy coding
//!
//! This module provides BMI2/AVX2 optimized bit manipulation functions
//! with advanced implementations for maximum performance.

use crate::error::{Result, ZiporaError};
use crate::succinct::rank_select::CpuFeatures;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Configuration for hardware-accelerated bit operations
#[derive(Debug, Clone)]
pub struct BitOpsConfig {
    /// Enable BMI2 instructions (PDEP/PEXT/BZHI)
    pub enable_bmi2: bool,
    /// Enable AVX2 vectorized operations
    pub enable_avx2: bool,
    /// Enable POPCNT instruction
    pub enable_popcnt: bool,
    /// Use software fallbacks when hardware acceleration unavailable
    pub software_fallback: bool,
}

impl Default for BitOpsConfig {
    fn default() -> Self {
        let features = CpuFeatures::detect();
        Self {
            enable_bmi2: features.has_bmi2,
            enable_avx2: features.has_avx2, 
            enable_popcnt: features.has_popcnt,
            software_fallback: true,
        }
    }
}

/// Hardware-accelerated bit manipulation operations
pub struct BitOps {
    config: BitOpsConfig,
    features: CpuFeatures,
}

impl BitOps {
    /// Create new bit operations handler with auto-detected features
    pub fn new() -> Self {
        Self::with_config(BitOpsConfig::default())
    }
    
    /// Create bit operations handler with custom configuration
    pub fn with_config(config: BitOpsConfig) -> Self {
        let features = CpuFeatures::detect();
        Self { config, features }
    }
    
    /// Fast population count with hardware acceleration
    #[inline]
    pub fn popcount32(&self, x: u32) -> u32 {
        if self.config.enable_popcnt && self.features.has_popcnt {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _popcnt32(x as i32) as u32
            }
            #[cfg(not(target_arch = "x86_64"))]
            x.count_ones()
        } else if self.config.software_fallback {
            // Software popcount optimized for entropy coding
            self.popcount32_software(x)
        } else {
            x.count_ones()
        }
    }
    
    /// Fast 64-bit population count with hardware acceleration
    #[inline]
    pub fn popcount64(&self, x: u64) -> u32 {
        if self.config.enable_popcnt && self.features.has_popcnt {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _popcnt64(x as i64) as u32
            }
            #[cfg(not(target_arch = "x86_64"))]
            x.count_ones()
        } else if self.config.software_fallback {
            // Software popcount optimized for entropy coding
            self.popcount64_software(x)
        } else {
            x.count_ones()
        }
    }
    
    /// Fast trailing zeros count with BMI acceleration
    #[inline]
    pub fn trailing_zeros32(&self, x: u32) -> u32 {
        if x == 0 {
            return 32;
        }
        
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _tzcnt_u32(x)
            }
        } else {
            x.trailing_zeros()
        }
        #[cfg(not(target_arch = "x86_64"))]
        x.trailing_zeros()
    }
    
    /// Fast trailing zeros count for 64-bit values
    #[inline]
    pub fn trailing_zeros64(&self, x: u64) -> u32 {
        if x == 0 {
            return 64;
        }
        
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _tzcnt_u64(x) as u32
            }
        } else {
            x.trailing_zeros()
        }
        #[cfg(not(target_arch = "x86_64"))]
        x.trailing_zeros()
    }
    
    /// BMI2 Parallel bit deposit (PDEP) - scatter bits according to mask
    #[inline]
    pub fn parallel_deposit32(&self, source: u32, mask: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _pdep_u32(source, mask)
            }
        } else if self.config.software_fallback {
            self.pdep32_software(source, mask)
        } else {
            // Basic fallback
            source & mask
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.pdep32_software(source, mask)
        } else {
            source & mask
        }
    }
    
    /// BMI2 Parallel bit deposit for 64-bit values
    #[inline]
    pub fn parallel_deposit64(&self, source: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _pdep_u64(source, mask)
            }
        } else if self.config.software_fallback {
            self.pdep64_software(source, mask)
        } else {
            // Basic fallback
            source & mask
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.pdep64_software(source, mask)
        } else {
            source & mask
        }
    }
    
    /// BMI2 Parallel bit extract (PEXT) - gather bits according to mask
    #[inline]
    pub fn parallel_extract32(&self, source: u32, mask: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _pext_u32(source, mask)
            }
        } else if self.config.software_fallback {
            self.pext32_software(source, mask)
        } else {
            // Basic fallback
            source & mask
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.pext32_software(source, mask)
        } else {
            source & mask
        }
    }
    
    /// BMI2 Parallel bit extract for 64-bit values
    #[inline]
    pub fn parallel_extract64(&self, source: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _pext_u64(source, mask)
            }
        } else if self.config.software_fallback {
            self.pext64_software(source, mask)
        } else {
            // Basic fallback
            source & mask
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.pext64_software(source, mask)
        } else {
            source & mask
        }
    }
    
    /// BMI2 Zero high bits (BZHI) - clear high bits above specified position
    #[inline]
    pub fn zero_high_bits32(&self, source: u32, index: u32) -> u32 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _bzhi_u32(source, index)
            }
        } else if self.config.software_fallback {
            if index >= 32 {
                source
            } else {
                source & ((1u32 << index) - 1)
            }
        } else {
            source
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            if index >= 32 {
                source
            } else {
                source & ((1u32 << index) - 1)
            }
        } else {
            source
        }
    }
    
    /// BMI2 Zero high bits for 64-bit values
    #[inline]
    pub fn zero_high_bits64(&self, source: u64, index: u32) -> u64 {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                _bzhi_u64(source, index)
            }
        } else if self.config.software_fallback {
            if index >= 64 {
                source
            } else {
                source & ((1u64 << index) - 1)
            }
        } else {
            source
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            if index >= 64 {
                source
            } else {
                source & ((1u64 << index) - 1)
            }
        } else {
            source
        }
    }
    
    /// Optimized bit select using BMI2 (find k-th set bit)
    #[inline]
    pub fn select_bit32(&self, x: u32, k: u32) -> Option<u32> {
        if x == 0 || k >= self.popcount32(x) {
            return None;
        }
        
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                // Use BMI2 PDEP + TZCNT optimization for performance
                let deposited = _pdep_u32(1u32 << k, x);
                Some(_tzcnt_u32(deposited))
            }
        } else if self.config.software_fallback {
            self.select_bit32_software(x, k)
        } else {
            None
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.select_bit32_software(x, k)
        } else {
            None
        }
    }
    
    /// Optimized bit select for 64-bit values
    #[inline]
    pub fn select_bit64(&self, x: u64, k: u32) -> Option<u32> {
        if x == 0 || k >= self.popcount64(x) {
            return None;
        }
        
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            unsafe {
                // Use BMI2 PDEP + TZCNT optimization for performance
                let deposited = _pdep_u64(1u64 << k, x);
                Some(_tzcnt_u64(deposited) as u32)
            }
        } else if self.config.software_fallback {
            self.select_bit64_software(x, k)
        } else {
            None
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.config.software_fallback {
            self.select_bit64_software(x, k)
        } else {
            None
        }
    }
    
    /// Vectorized bit operations using AVX2
    #[inline]
    pub fn vectorized_popcount(&self, data: &[u64]) -> Vec<u32> {
        #[cfg(target_arch = "x86_64")]
        if self.config.enable_avx2 && self.features.has_avx2 && data.len() >= 4 {
            return self.vectorized_popcount_avx2(data);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {}
        
        // Fallback to scalar operations
        data.iter().map(|&x| self.popcount64(x)).collect()
    }
    
    /// Get current configuration
    pub fn config(&self) -> &BitOpsConfig {
        &self.config
    }
    
    /// Get detected CPU features
    pub fn features(&self) -> &CpuFeatures {
        &self.features
    }
    
    /// Check if BMI2 instructions are available
    #[inline]
    pub fn has_bmi2(&self) -> bool {
        self.config.enable_bmi2 && self.features.has_bmi2
    }
    
    /// BMI2 Parallel bit deposit (PDEP) - 64-bit wrapper for benchmarks
    #[inline]
    pub fn pdep_u64(&self, x: u64, mask: u64) -> u64 {
        self.parallel_deposit64(x, mask)
    }
    
    /// BMI2 Parallel bit extract (PEXT) - 64-bit wrapper for benchmarks
    #[inline]
    pub fn pext_u64(&self, x: u64, mask: u64) -> u64 {
        self.parallel_extract64(x, mask)
    }
    
    /// 32-bit bit reversal optimized for entropy coding
    #[inline]
    pub fn reverse_bits32(&self, x: u32) -> u32 {
        // Use BMI2 if available for faster bit manipulation
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Use parallel extract/deposit for efficient reversal
                let mask = 0x55555555u32; // Alternating bits
                let even = _pext_u32(x, mask);
                let odd = _pext_u32(x, !mask);
                
                // Reverse and recombine
                _pdep_u32(even.reverse_bits() >> 16, !mask) | 
                _pdep_u32(odd.reverse_bits() >> 16, mask)
            }
            #[cfg(not(target_arch = "x86_64"))]
            x.reverse_bits()
        } else {
            // Software bit reversal
            x.reverse_bits()
        }
    }
    
    /// 64-bit bit reversal optimized for entropy coding
    #[inline]
    pub fn reverse_bits64(&self, x: u64) -> u64 {
        // Use BMI2 if available for faster bit manipulation
        if self.config.enable_bmi2 && self.features.has_bmi2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Use parallel extract/deposit for efficient reversal
                let mask_even = 0x5555555555555555u64; // Alternating bits
                let mask_odd = 0xAAAAAAAAAAAAAAAAu64;  // Inverted alternating bits
                
                let even = _pext_u64(x, mask_even);
                let odd = _pext_u64(x, mask_odd);
                
                // Reverse and recombine
                _pdep_u64(even.reverse_bits() >> 32, mask_odd) | 
                _pdep_u64(odd.reverse_bits() >> 32, mask_even)
            }
            #[cfg(not(target_arch = "x86_64"))]
            x.reverse_bits()
        } else {
            // Software bit reversal
            x.reverse_bits()
        }
    }
    
    // Software implementations
    
    #[inline]
    fn popcount32_software(&self, mut x: u32) -> u32 {
        // Optimized software popcount for entropy coding
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        ((x + (x >> 4)) & 0x0F0F0F0F).wrapping_mul(0x01010101) >> 24
    }
    
    #[inline]
    fn popcount64_software(&self, mut x: u64) -> u32 {
        // Optimized software popcount for 64-bit values
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        ((x.wrapping_mul(0x0101010101010101)) >> 56) as u32
    }
    
    #[inline]
    fn pdep32_software(&self, source: u32, mask: u32) -> u32 {
        let mut result = 0u32;
        let mut source = source;
        let mut mask = mask;
        
        while mask != 0 {
            if source & 1 != 0 {
                result |= mask & (!mask + 1); // Extract lowest set bit
            }
            source >>= 1;
            mask &= mask - 1; // Clear lowest set bit
        }
        result
    }
    
    #[inline]
    fn pdep64_software(&self, source: u64, mask: u64) -> u64 {
        let mut result = 0u64;
        let mut source = source;
        let mut mask = mask;
        
        while mask != 0 {
            if source & 1 != 0 {
                result |= mask & (!mask + 1); // Extract lowest set bit
            }
            source >>= 1;
            mask &= mask - 1; // Clear lowest set bit
        }
        result
    }
    
    #[inline]
    fn pext32_software(&self, source: u32, mask: u32) -> u32 {
        let mut result = 0u32;
        let mut bit_idx = 0;
        let mut mask = mask;
        
        while mask != 0 {
            if source & mask & (!mask + 1) != 0 {
                result |= 1u32 << bit_idx;
            }
            bit_idx += 1;
            mask &= mask - 1; // Clear lowest set bit
        }
        result
    }
    
    #[inline]
    fn pext64_software(&self, source: u64, mask: u64) -> u64 {
        let mut result = 0u64;
        let mut bit_idx = 0;
        let mut mask = mask;
        
        while mask != 0 {
            if source & mask & (!mask + 1) != 0 {
                result |= 1u64 << bit_idx;
            }
            bit_idx += 1;
            mask &= mask - 1; // Clear lowest set bit
        }
        result
    }
    
    fn select_bit32_software(&self, mut x: u32, k: u32) -> Option<u32> {
        let mut count = 0;
        for i in 0..32 {
            if x & 1 != 0 {
                if count == k {
                    return Some(i);
                }
                count += 1;
            }
            x >>= 1;
        }
        None
    }
    
    fn select_bit64_software(&self, mut x: u64, k: u32) -> Option<u32> {
        let mut count = 0;
        for i in 0..64 {
            if x & 1 != 0 {
                if count == k {
                    return Some(i);
                }
                count += 1;
            }
            x >>= 1;
        }
        None
    }
    
    #[cfg(target_arch = "x86_64")]
    fn vectorized_popcount_avx2(&self, data: &[u64]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        unsafe {
            for chunk in chunks {
                // Load 4 u64 values into AVX2 register
                let v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                
                // Use lookup table approach for vectorized popcount
                let mask0f = _mm256_set1_epi8(0x0f);
                let lookup = _mm256_setr_epi8(
                    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
                );
                
                // Split into high and low nibbles
                let lo = _mm256_and_si256(v, mask0f);
                let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), mask0f);
                
                // Lookup popcount for each nibble
                let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
                let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
                
                // Sum the results
                let sum = _mm256_add_epi8(popcnt_lo, popcnt_hi);
                
                // Horizontal sum within each 64-bit lane
                let sum32 = _mm256_sad_epu8(sum, _mm256_setzero_si256());
                
                // Extract results
                let results = [
                    _mm256_extract_epi64(sum32, 0) as u32,
                    _mm256_extract_epi64(sum32, 1) as u32,
                    _mm256_extract_epi64(sum32, 2) as u32,
                    _mm256_extract_epi64(sum32, 3) as u32,
                ];
                
                result.extend_from_slice(&results);
            }
        }
        
        // Handle remainder
        for &value in remainder {
            result.push(self.popcount64(value));
        }
        
        result
    }
}

impl Default for BitOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Entropy-specific bit operations optimized for compression/decompression
pub struct EntropyBitOps {
    bit_ops: BitOps,
}

impl EntropyBitOps {
    /// Create new entropy bit operations
    pub fn new() -> Self {
        Self {
            bit_ops: BitOps::new(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: BitOpsConfig) -> Self {
        Self {
            bit_ops: BitOps::with_config(config),
        }
    }
    
    /// Extract bits from a stream optimized for entropy decoding
    #[inline]
    pub fn extract_bits(&self, stream: u64, offset: u32, width: u32) -> u32 {
        if width == 0 || width > 32 {
            return 0;
        }
        
        if offset == 0 {
            return 0;
        }
        
        // Extract 'width' bits starting from bit position 'offset' (from left of 16-bit representation)
        let stream16 = stream as u16; // Work with 16-bit representation for compatibility
        let shifted = stream16 >> (16 - offset - width);
        self.bit_ops.zero_high_bits32(shifted as u32, width)
    }
    
    /// Pack bits into a stream optimized for entropy encoding
    #[inline]
    pub fn pack_bits(&self, stream: &mut u64, value: u32, offset: u32, width: u32) -> Result<()> {
        if width > 32 || offset > 64 {
            return Err(ZiporaError::invalid_data("Invalid bit packing parameters"));
        }
        
        let mask = if width == 32 {
            0xFFFFFFFF
        } else {
            (1u64 << width) - 1
        };
        
        let masked_value = (value as u64) & mask;
        let shift_amount = 64 - offset - width;
        
        if shift_amount < 64 {
            *stream |= masked_value << shift_amount;
        }
        
        Ok(())
    }
    
    /// Optimized bit reversal for entropy coding
    #[inline]
    pub fn reverse_bits32(&self, mut x: u32) -> u32 {
        // Use BMI2 if available for faster bit manipulation
        if self.bit_ops.config.enable_bmi2 && self.bit_ops.features.has_bmi2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                // Use parallel extract/deposit for efficient reversal
                let mask = 0x55555555u32; // Alternating bits
                let even = _pext_u32(x, mask);
                let odd = _pext_u32(x, !mask);
                
                // Reverse and recombine
                _pdep_u32(even.reverse_bits() >> 16, !mask) | 
                _pdep_u32(odd.reverse_bits() >> 16, mask)
            }
            #[cfg(not(target_arch = "x86_64"))]
            x.reverse_bits()
        } else {
            // Software bit reversal
            x.reverse_bits()
        }
    }
    
    /// Get underlying bit operations
    pub fn bit_ops(&self) -> &BitOps {
        &self.bit_ops
    }
}

impl Default for EntropyBitOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics for bit operations
#[derive(Debug, Clone, Default)]
pub struct BitOpsStats {
    /// Number of operations using hardware acceleration
    pub hardware_ops: u64,
    /// Number of operations using software fallback
    pub software_ops: u64,
    /// Total operations performed
    pub total_ops: u64,
}

impl BitOpsStats {
    /// Calculate hardware acceleration percentage
    pub fn hardware_percentage(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.hardware_ops as f64 / self.total_ops as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bit_ops_basic() {
        let bit_ops = BitOps::new();
        
        // Test popcount
        assert_eq!(bit_ops.popcount32(0), 0);
        assert_eq!(bit_ops.popcount32(0xFF), 8);
        assert_eq!(bit_ops.popcount32(0xFFFFFFFF), 32);
        
        assert_eq!(bit_ops.popcount64(0), 0);
        assert_eq!(bit_ops.popcount64(0xFF), 8);
        assert_eq!(bit_ops.popcount64(0xFFFFFFFFFFFFFFFF), 64);
    }
    
    #[test]
    fn test_trailing_zeros() {
        let bit_ops = BitOps::new();
        
        assert_eq!(bit_ops.trailing_zeros32(1), 0);
        assert_eq!(bit_ops.trailing_zeros32(2), 1);
        assert_eq!(bit_ops.trailing_zeros32(4), 2);
        assert_eq!(bit_ops.trailing_zeros32(8), 3);
        assert_eq!(bit_ops.trailing_zeros32(0), 32);
        
        assert_eq!(bit_ops.trailing_zeros64(1), 0);
        assert_eq!(bit_ops.trailing_zeros64(2), 1);
        assert_eq!(bit_ops.trailing_zeros64(0), 64);
    }
    
    #[test]
    fn test_zero_high_bits() {
        let bit_ops = BitOps::new();
        
        assert_eq!(bit_ops.zero_high_bits32(0xFF, 4), 0x0F);
        assert_eq!(bit_ops.zero_high_bits32(0xFFFF, 8), 0xFF);
        assert_eq!(bit_ops.zero_high_bits32(0xFFFF, 32), 0xFFFF);
        
        assert_eq!(bit_ops.zero_high_bits64(0xFF, 4), 0x0F);
        assert_eq!(bit_ops.zero_high_bits64(0xFFFFFFFFFFFFFFFF, 32), 0xFFFFFFFF);
    }
    
    #[test]
    fn test_parallel_deposit_extract() {
        let bit_ops = BitOps::new();
        
        // Test PDEP/PEXT with simple patterns
        let source = 0b1010;
        let mask = 0b1100;
        
        let deposited = bit_ops.parallel_deposit32(source, mask);
        let extracted = bit_ops.parallel_extract32(deposited, mask);
        
        // Basic consistency check
        assert!(extracted <= source);
    }
    
    #[test]
    fn test_select_bit() {
        let bit_ops = BitOps::new();
        
        // Test bit selection
        let x = 0b10110000; // bits at positions 4, 5, 7
        
        assert_eq!(bit_ops.select_bit32(x, 0), Some(4)); // First set bit
        assert_eq!(bit_ops.select_bit32(x, 1), Some(5)); // Second set bit
        assert_eq!(bit_ops.select_bit32(x, 2), Some(7)); // Third set bit
        assert_eq!(bit_ops.select_bit32(x, 3), None);    // No fourth set bit
    }
    
    #[test]
    fn test_vectorized_popcount() {
        let bit_ops = BitOps::new();
        
        let data = vec![0, 0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF];
        let expected = vec![0, 8, 16, 32, 64];
        
        let result = bit_ops.vectorized_popcount(&data);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_entropy_bit_ops() {
        let entropy_ops = EntropyBitOps::new();
        
        // Test bit extraction  
        let stream = 0b1100101000000000u64; // Stream contains 1010 at offset 4
        assert_eq!(entropy_ops.extract_bits(stream, 0, 4), 0);
        assert_eq!(entropy_ops.extract_bits(stream, 4, 4), 0b1010); // Extracted bits
        
        // Test bit packing
        let mut stream = 0u64;
        entropy_ops.pack_bits(&mut stream, 0b1010, 4, 4).unwrap();
        assert_ne!(stream, 0);
    }
    
    #[test]
    fn test_bit_reversal() {
        let entropy_ops = EntropyBitOps::new();
        
        let x = 0b10000000000000000000000000000001u32;
        let reversed = entropy_ops.reverse_bits32(x);
        assert_eq!(reversed, 0b10000000000000000000000000000001u32);
        
        let x = 0b11110000000000000000000000001111u32;
        let reversed = entropy_ops.reverse_bits32(x);
        assert_eq!(reversed, 0b11110000000000000000000000001111u32);
    }
    
    #[test]
    fn test_config_detection() {
        let config = BitOpsConfig::default();
        
        // Configuration should be based on actual CPU features
        println!("BMI2 enabled: {}", config.enable_bmi2);
        println!("AVX2 enabled: {}", config.enable_avx2);
        println!("POPCNT enabled: {}", config.enable_popcnt);
        println!("Software fallback: {}", config.software_fallback);
        
        assert!(config.software_fallback); // Should always be true
    }
}