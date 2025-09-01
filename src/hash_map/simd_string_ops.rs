//! SIMD-accelerated string operations for hash maps
//!
//! This module provides hardware-accelerated string operations including:
//! - SIMD string comparison with prefix optimization
//! - SIMD string hashing acceleration
//! - Runtime CPU feature detection and adaptive selection
//! - Fallback to scalar operations when SIMD is unavailable

use crate::system::cpu_features::CpuFeatures;
use crate::memory::simd_ops::fast_compare;
use std::arch::x86_64::*;

/// SIMD-accelerated string operations for hash maps
pub struct SimdStringOps {
    /// CPU features available at runtime
    cpu_features: &'static CpuFeatures,
    /// Selected implementation tier based on available features
    impl_tier: SimdTier,
}

/// SIMD implementation tiers based on available CPU features
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdTier {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE4.2 implementation
    Sse42,
    /// AVX2 implementation
    Avx2,
    /// AVX-512 implementation (nightly only)
    #[cfg(feature = "avx512")]
    Avx512,
}

impl SimdStringOps {
    /// Creates a new SIMD string operations instance with runtime feature detection
    pub fn new() -> Self {
        let cpu_features = crate::system::get_cpu_features();
        let impl_tier = Self::select_optimal_tier(cpu_features);
        
        Self {
            cpu_features,
            impl_tier,
        }
    }

    /// Selects the optimal SIMD implementation tier based on available CPU features
    /// This matches the tier selection logic in core SIMD memory operations for consistency
    fn select_optimal_tier(features: &CpuFeatures) -> SimdTier {
        #[cfg(feature = "avx512")]
        if features.has_avx512f && features.has_avx512vl && features.has_avx512bw {
            return SimdTier::Avx512;
        }
        
        if features.has_avx2 {
            return SimdTier::Avx2;
        }
        
        if features.has_sse41 && features.has_sse42 {
            return SimdTier::Sse42;
        }
        
        SimdTier::Scalar
    }

    /// Returns the currently selected SIMD tier
    pub fn tier(&self) -> SimdTier {
        self.impl_tier
    }

    /// SIMD-accelerated string comparison with prefix optimization
    pub fn fast_string_compare(&self, str1: &str, str2: &str, cached_prefix: u64) -> bool {
        // Quick length check
        if str1.len() != str2.len() {
            return false;
        }
        
        let bytes1 = str1.as_bytes();
        let bytes2 = str2.as_bytes();
        
        // For very short strings, use scalar comparison
        if bytes1.len() <= 8 {
            return self.scalar_compare_prefix(bytes1, bytes2, cached_prefix);
        }
        
        // Use prefix optimization for cached comparisons
        if bytes1.len() >= 8 && cached_prefix != 0 {
            let prefix1 = self.extract_prefix_simd(str1);
            if prefix1 != cached_prefix {
                return false;
            }
        }
        
        // For medium to large strings, use SIMD implementations based on tier
        // Use size threshold of 16 bytes for SIMD activation to match core SIMD ops
        if bytes1.len() >= 16 {
            match self.impl_tier {
                #[cfg(feature = "avx512")]
                SimdTier::Avx512 if bytes1.len() >= 64 => {
                    unsafe { self.avx512_string_compare(bytes1, bytes2, cached_prefix) }
                }
                SimdTier::Avx2 if bytes1.len() >= 32 => {
                    unsafe { self.avx2_string_compare(bytes1, bytes2, cached_prefix) }
                }
                SimdTier::Sse42 if bytes1.len() >= 16 => {
                    unsafe { self.sse42_string_compare(bytes1, bytes2, cached_prefix) }
                }
                _ => {
                    // For smaller sizes or scalar tier, use core SIMD operations for consistency
                    fast_compare(bytes1, bytes2) == 0
                }
            }
        } else {
            // Use scalar comparison for small strings
            self.scalar_string_compare(bytes1, bytes2, cached_prefix)
        }
    }

    /// SIMD-accelerated string hashing
    pub fn fast_string_hash(&self, s: &str, base_hash: u64) -> u64 {
        let bytes = s.as_bytes();
        
        // For very short strings, use scalar hashing
        if bytes.len() <= 8 {
            return self.scalar_string_hash(bytes, base_hash);
        }
        
        // Use SIMD implementations based on tier and size thresholds
        match self.impl_tier {
            #[cfg(feature = "avx512")]
            SimdTier::Avx512 if bytes.len() >= 64 => {
                unsafe { self.avx512_string_hash(bytes, base_hash) }
            }
            SimdTier::Avx2 if bytes.len() >= 32 => {
                unsafe { self.avx2_string_hash(bytes, base_hash) }
            }
            SimdTier::Sse42 if bytes.len() >= 16 => {
                unsafe { self.sse42_string_hash(bytes, base_hash) }
            }
            _ => {
                // Fall back to scalar implementation for smaller sizes or scalar tier
                self.scalar_string_hash(bytes, base_hash)
            }
        }
    }

    /// Extracts prefix for caching with SIMD optimization
    pub fn extract_prefix_simd(&self, s: &str) -> u64 {
        let bytes = s.as_bytes();
        
        if bytes.len() >= 8 {
            match self.impl_tier {
                SimdTier::Avx2 | SimdTier::Sse42 => {
                    // Use SIMD for 8-byte loads when available
                    unsafe {
                        let ptr = bytes.as_ptr() as *const u64;
                        ptr.read_unaligned()
                    }
                }
                #[cfg(feature = "avx512")]
                SimdTier::Avx512 => {
                    // Use AVX-512 optimized 8-byte load
                    unsafe {
                        let ptr = bytes.as_ptr() as *const u64;
                        ptr.read_unaligned()
                    }
                }
                _ => self.scalar_extract_prefix(bytes),
            }
        } else {
            self.scalar_extract_prefix(bytes)
        }
    }

    // =============================================================================
    // AVX2 IMPLEMENTATIONS
    // =============================================================================

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_string_compare(&self, bytes1: &[u8], bytes2: &[u8], _cached_prefix: u64) -> bool {
        let len = bytes1.len();
        
        // Process 32-byte chunks with AVX2
        let chunks = len / 32;
        for i in 0..chunks {
            let offset = i * 32;
            
            unsafe {
                let chunk1 = _mm256_loadu_si256(bytes1.as_ptr().add(offset) as *const __m256i);
                let chunk2 = _mm256_loadu_si256(bytes2.as_ptr().add(offset) as *const __m256i);
                
                let cmp = _mm256_cmpeq_epi8(chunk1, chunk2);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                
                if mask != 0xFFFFFFFF {
                    return false;
                }
            }
        }
        
        // Handle remaining bytes with scalar comparison
        let remaining_start = chunks * 32;
        bytes1[remaining_start..] == bytes2[remaining_start..]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2_string_hash(&self, bytes: &[u8], mut hash: u64) -> u64 {
        let len = bytes.len();
        
        // Process 32-byte chunks with AVX2
        let chunks = len / 32;
        for i in 0..chunks {
            let offset = i * 32;
            unsafe {
                let chunk = _mm256_loadu_si256(bytes.as_ptr().add(offset) as *const __m256i);
                
                // Extract 64-bit values from the 256-bit vector
                let vals = std::mem::transmute::<__m256i, [u64; 4]>(chunk);
                
                for val in vals {
                    hash = hash.rotate_left(5).wrapping_add(val);
                }
            }
        }
        
        // Handle remaining bytes
        let remaining_start = chunks * 32;
        for &byte in &bytes[remaining_start..] {
            hash = hash.rotate_left(5).wrapping_add(byte as u64);
        }
        
        hash
    }

    // =============================================================================
    // SSE4.2 IMPLEMENTATIONS
    // =============================================================================

    #[target_feature(enable = "sse4.2")]
    unsafe fn sse42_string_compare(&self, bytes1: &[u8], bytes2: &[u8], _cached_prefix: u64) -> bool {
        let len = bytes1.len();
        
        // Process 16-byte chunks with SSE4.2
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            
            unsafe {
                let chunk1 = _mm_loadu_si128(bytes1.as_ptr().add(offset) as *const __m128i);
                let chunk2 = _mm_loadu_si128(bytes2.as_ptr().add(offset) as *const __m128i);
                
                let cmp = _mm_cmpeq_epi8(chunk1, chunk2);
                let mask = _mm_movemask_epi8(cmp);
                
                if mask != 0xFFFF {
                    return false;
                }
            }
        }
        
        // Handle remaining bytes with scalar comparison
        let remaining_start = chunks * 16;
        bytes1[remaining_start..] == bytes2[remaining_start..]
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn sse42_string_hash(&self, bytes: &[u8], mut hash: u64) -> u64 {
        let len = bytes.len();
        
        // Process 16-byte chunks with SSE4.2
        let chunks = len / 16;
        for i in 0..chunks {
            let offset = i * 16;
            unsafe {
                let chunk = _mm_loadu_si128(bytes.as_ptr().add(offset) as *const __m128i);
                
                // Extract 64-bit values from the 128-bit vector
                let vals = std::mem::transmute::<__m128i, [u64; 2]>(chunk);
                
                for val in vals {
                    hash = hash.rotate_left(5).wrapping_add(val);
                }
            }
        }
        
        // Handle remaining bytes
        let remaining_start = chunks * 16;
        for &byte in &bytes[remaining_start..] {
            hash = hash.rotate_left(5).wrapping_add(byte as u64);
        }
        
        hash
    }

    // =============================================================================
    // AVX-512 IMPLEMENTATIONS (NIGHTLY ONLY)
    // =============================================================================

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn avx512_string_compare(&self, bytes1: &[u8], bytes2: &[u8], _cached_prefix: u64) -> bool {
        let len = bytes1.len();
        
        // Process 64-byte chunks with AVX-512
        let chunks = len / 64;
        for i in 0..chunks {
            let offset = i * 64;
            
            unsafe {
                let chunk1 = _mm512_loadu_si512(bytes1.as_ptr().add(offset) as *const __m512i);
                let chunk2 = _mm512_loadu_si512(bytes2.as_ptr().add(offset) as *const __m512i);
                
                let mask = _mm512_cmpeq_epi8_mask(chunk1, chunk2);
                
                if mask != 0xFFFFFFFFFFFFFFFF {
                    return false;
                }
            }
        }
        
        // Handle remaining bytes with scalar comparison
        let remaining_start = chunks * 64;
        bytes1[remaining_start..] == bytes2[remaining_start..]
    }

    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_string_hash(&self, bytes: &[u8], mut hash: u64) -> u64 {
        let len = bytes.len();
        
        // Process 64-byte chunks with AVX-512
        let chunks = len / 64;
        for i in 0..chunks {
            let offset = i * 64;
            unsafe {
                let chunk = _mm512_loadu_si512(bytes.as_ptr().add(offset) as *const __m512i);
                
                // Extract 64-bit values from the 512-bit vector
                let vals = std::mem::transmute::<__m512i, [u64; 8]>(chunk);
                
                for val in vals {
                    hash = hash.rotate_left(5).wrapping_add(val);
                }
            }
        }
        
        // Handle remaining bytes
        let remaining_start = chunks * 64;
        for &byte in &bytes[remaining_start..] {
            hash = hash.rotate_left(5).wrapping_add(byte as u64);
        }
        
        hash
    }

    // =============================================================================
    // SCALAR FALLBACK IMPLEMENTATIONS
    // =============================================================================

    fn scalar_string_compare(&self, bytes1: &[u8], bytes2: &[u8], cached_prefix: u64) -> bool {
        // Use prefix optimization for scalar comparison
        self.scalar_compare_prefix(bytes1, bytes2, cached_prefix)
    }

    fn scalar_compare_prefix(&self, bytes1: &[u8], bytes2: &[u8], cached_prefix: u64) -> bool {
        // Quick prefix check if we have cached prefix
        if bytes1.len() >= 8 && bytes2.len() >= 8 && cached_prefix != 0 {
            let prefix1 = self.scalar_extract_prefix(bytes1);
            if prefix1 != cached_prefix {
                return false;
            }
        }
        
        // Full comparison
        bytes1 == bytes2
    }

    fn scalar_string_hash(&self, bytes: &[u8], mut hash: u64) -> u64 {
        // Process 8-byte chunks for better performance
        let chunks = bytes.len() / 8;
        for i in 0..chunks {
            let offset = i * 8;
            let chunk_bytes = &bytes[offset..offset + 8];
            let val = u64::from_le_bytes(chunk_bytes.try_into().unwrap());
            hash = hash.rotate_left(5).wrapping_add(val);
        }
        
        // Handle remaining bytes
        let remaining_start = chunks * 8;
        for &byte in &bytes[remaining_start..] {
            hash = hash.rotate_left(5).wrapping_add(byte as u64);
        }
        
        hash
    }

    fn scalar_extract_prefix(&self, bytes: &[u8]) -> u64 {
        let mut prefix = 0u64;
        
        for (i, &byte) in bytes.iter().take(8).enumerate() {
            prefix |= (byte as u64) << (i * 8);
        }
        
        prefix
    }
}

impl Default for SimdStringOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Global SIMD string operations instance for reuse
static GLOBAL_SIMD_OPS: std::sync::OnceLock<SimdStringOps> = std::sync::OnceLock::new();

/// Gets the global SIMD string operations instance
pub fn get_global_simd_ops() -> &'static SimdStringOps {
    GLOBAL_SIMD_OPS.get_or_init(|| SimdStringOps::new())
}

/// Convenience function for fast string comparison using global SIMD operations
/// Integrates with the core SIMD memory operations for optimal performance
pub fn fast_string_compare(str1: &str, str2: &str, cached_prefix: u64) -> bool {
    get_global_simd_ops().fast_string_compare(str1, str2, cached_prefix)
}

/// Convenience function for fast string hashing using global SIMD operations
pub fn fast_string_hash(s: &str, base_hash: u64) -> u64 {
    get_global_simd_ops().fast_string_hash(s, base_hash)
}

/// Convenience function for extracting string prefix using global SIMD operations
pub fn extract_string_prefix(s: &str) -> u64 {
    get_global_simd_ops().extract_prefix_simd(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_ops_creation() {
        let ops = SimdStringOps::new();
        println!("Selected SIMD tier: {:?}", ops.tier());
        
        // Should always work regardless of available features
        assert!(matches!(ops.tier(), SimdTier::Scalar | SimdTier::Sse42 | SimdTier::Avx2));
    }

    #[test]
    fn test_global_simd_ops() {
        let ops1 = get_global_simd_ops();
        let ops2 = get_global_simd_ops();
        
        // Should be the same instance
        assert_eq!(ops1.tier(), ops2.tier());
    }

    #[test]
    fn test_string_comparison() {
        let ops = SimdStringOps::new();
        
        let str1 = "hello world test string";
        let str2 = "hello world test string";
        let str3 = "hello world different";
        
        let prefix = ops.extract_prefix_simd(str1);
        
        assert!(ops.fast_string_compare(str1, str2, prefix));
        assert!(!ops.fast_string_compare(str1, str3, prefix));
    }

    #[test]
    fn test_string_hashing() {
        let ops = SimdStringOps::new();
        
        let test_string = "test string for hashing";
        let hash1 = ops.fast_string_hash(test_string, 0);
        let hash2 = ops.fast_string_hash(test_string, 0);
        
        // Same string should produce same hash
        assert_eq!(hash1, hash2);
        
        // Different strings should produce different hashes (with high probability)
        let different_string = "different string for hashing";
        let hash3 = ops.fast_string_hash(different_string, 0);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_prefix_extraction() {
        let ops = SimdStringOps::new();
        
        let test_string = "prefixtest";
        let prefix = ops.extract_prefix_simd(test_string);
        
        // Should extract first 8 bytes
        assert_ne!(prefix, 0);
        
        // Same prefix for strings with same beginning
        let similar_string = "prefixtesting";
        let similar_prefix = ops.extract_prefix_simd(similar_string);
        assert_eq!(prefix, similar_prefix);
    }

    #[test]
    fn test_short_strings() {
        let ops = SimdStringOps::new();
        
        let short1 = "hi";
        let short2 = "hi";
        let short3 = "bye";
        
        let prefix = ops.extract_prefix_simd(short1);
        
        assert!(ops.fast_string_compare(short1, short2, prefix));
        assert!(!ops.fast_string_compare(short1, short3, prefix));
    }

    #[test]
    fn test_empty_strings() {
        let ops = SimdStringOps::new();
        
        let empty1 = "";
        let empty2 = "";
        let non_empty = "test";
        
        let prefix = ops.extract_prefix_simd(empty1);
        
        assert!(ops.fast_string_compare(empty1, empty2, prefix));
        assert!(!ops.fast_string_compare(empty1, non_empty, prefix));
    }
    
    #[test]
    fn test_simd_integration_with_large_strings() {
        let ops = SimdStringOps::new();
        println!("SIMD tier: {:?}", ops.tier());
        
        // Test with strings large enough to trigger SIMD operations
        let large_string1 = "a".repeat(1024);
        let large_string2 = "a".repeat(1024);
        let large_string3 = "b".repeat(1024);
        
        let prefix = ops.extract_prefix_simd(&large_string1);
        
        // These should use SIMD operations based on tier and size
        assert!(ops.fast_string_compare(&large_string1, &large_string2, prefix));
        assert!(!ops.fast_string_compare(&large_string1, &large_string3, prefix));
        
        // Test hashing with large strings
        let hash1 = ops.fast_string_hash(&large_string1, 0);
        let hash2 = ops.fast_string_hash(&large_string2, 0);
        let hash3 = ops.fast_string_hash(&large_string3, 0);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_convenience_functions() {
        // Test the new convenience functions
        let str1 = "hello world test string for SIMD";
        let str2 = "hello world test string for SIMD";
        let str3 = "different string for SIMD testing";
        
        let prefix = extract_string_prefix(str1);
        
        assert!(fast_string_compare(str1, str2, prefix));
        assert!(!fast_string_compare(str1, str3, prefix));
        
        let hash1 = fast_string_hash(str1, 0);
        let hash2 = fast_string_hash(str2, 0);
        let hash3 = fast_string_hash(str3, 0);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_simd_size_thresholds() {
        let ops = SimdStringOps::new();
        
        // Test strings of various sizes around SIMD thresholds
        let small_str = "small";  // 5 bytes
        let medium_str = "this is a medium sized string for testing"; // ~40 bytes  
        let large_str = "a".repeat(128); // 128 bytes
        
        let prefix_small = ops.extract_prefix_simd(&small_str);
        let prefix_medium = ops.extract_prefix_simd(&medium_str);
        let prefix_large = ops.extract_prefix_simd(&large_str);
        
        // All should work regardless of size
        assert!(ops.fast_string_compare(&small_str, &small_str, prefix_small));
        assert!(ops.fast_string_compare(&medium_str, &medium_str, prefix_medium));
        assert!(ops.fast_string_compare(&large_str, &large_str, prefix_large));
        
        // Hash values should be consistent
        let hash_small1 = ops.fast_string_hash(&small_str, 0);
        let hash_small2 = ops.fast_string_hash(&small_str, 0);
        assert_eq!(hash_small1, hash_small2);
        
        let hash_medium1 = ops.fast_string_hash(&medium_str, 0);
        let hash_medium2 = ops.fast_string_hash(&medium_str, 0);
        assert_eq!(hash_medium1, hash_medium2);
        
        let hash_large1 = ops.fast_string_hash(&large_str, 0);
        let hash_large2 = ops.fast_string_hash(&large_str, 0);
        assert_eq!(hash_large1, hash_large2);
    }
}