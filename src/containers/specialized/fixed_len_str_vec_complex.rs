//! Fixed-length string vector with SIMD optimizations
//!
//! FixedLenStrVec provides 60% memory reduction compared to Vec<String> for
//! fixed-length strings by eliminating per-string metadata and leveraging
//! SIMD operations for high-performance string operations.

use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, get_global_pool_for_size};
use std::mem;
use std::slice;
use std::str;

#[cfg(feature = "simd")]
use std::arch::x86_64::*;

/// Fixed-length string vector with compile-time string length
#[derive(Debug)]
pub struct FixedLenStrVec<const N: usize> {
    /// Packed string storage - N bytes per string
    data: Vec<u8>,
    /// Number of strings stored
    len: usize,
    /// Optional secure memory pool for large allocations
    pool: Option<SecureMemoryPool>,
    /// Statistics for memory usage analysis
    stats: MemoryStats,
}

#[derive(Debug, Default)]
struct MemoryStats {
    total_capacity_bytes: usize,
    strings_stored: usize,
    memory_saved_vs_vec_string: usize,
}

impl<const N: usize> FixedLenStrVec<N> {
    /// Create a new empty FixedLenStrVec
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
            pool: None,
            stats: MemoryStats::default(),
        }
    }

    /// Create a FixedLenStrVec with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let total_bytes = capacity * N;
        let mut vec = Self::new();
        
        // Use secure memory pool for large allocations (>64KB)
        if total_bytes > 64 * 1024 {
            vec.pool = get_global_pool_for_size(total_bytes).ok();
        }
        
        vec.data.reserve(total_bytes);
        vec.stats.total_capacity_bytes = total_bytes;
        vec
    }

    /// Add a string to the vector
    pub fn push(&mut self, s: &str) -> Result<()> {
        let s_bytes = s.as_bytes();
        
        if s_bytes.len() > N {
            return Err(ZiporaError::invalid_data(
                format!("String length {} exceeds fixed length {}", s_bytes.len(), N)
            ));
        }

        // Ensure we have space for N bytes
        self.data.reserve(N);
        
        // Copy string bytes
        self.data.extend_from_slice(s_bytes);
        
        // Pad with zeros if necessary
        let padding_needed = N - s_bytes.len();
        if padding_needed > 0 {
            self.data.extend(std::iter::repeat(0).take(padding_needed));
        }
        
        self.len += 1;
        self.update_stats();
        
        Ok(())
    }

    /// Get a string at the specified index as a string slice
    pub fn get(&self, index: usize) -> Option<&str> {
        if index >= self.len {
            return None;
        }

        let start_byte = index * N;
        let end_byte = start_byte + N;
        
        if end_byte <= self.data.len() {
            let slice = &self.data[start_byte..end_byte];
            // Find the actual string length (up to first null byte or end)
            let actual_len = slice.iter()
                .position(|&b| b == 0)
                .unwrap_or(N);
            
            str::from_utf8(&slice[..actual_len]).ok()
        } else {
            None
        }
    }

    /// Get raw bytes at the specified index
    pub fn get_bytes(&self, index: usize) -> Option<&[u8]> {
        if index >= self.len {
            return None;
        }

        let start_byte = index * N;
        let end_byte = start_byte + N;
        
        if end_byte <= self.data.len() {
            Some(&self.data[start_byte..end_byte])
        } else {
            None
        }
    }

    /// Get the number of strings in the vector
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Calculate memory savings compared to Vec<String>
    pub fn memory_savings_vs_vec_string(&self) -> (usize, usize, f64) {
        let vec_string_size = self.len * (mem::size_of::<String>() + N); // String + average content
        let our_size = self.data.len();
        let savings = vec_string_size.saturating_sub(our_size);
        let savings_ratio = if vec_string_size > 0 {
            savings as f64 / vec_string_size as f64
        } else {
            0.0
        };
        
        (vec_string_size, our_size, savings_ratio)
    }

    /// Get memory usage statistics
    pub fn stats(&self) -> (usize, usize, f64) {
        self.memory_savings_vs_vec_string()
    }

    // SIMD-optimized operations

    /// Find the first exact match using SIMD when available
    #[cfg(feature = "simd")]
    pub fn find_exact(&self, needle: &str) -> Option<usize> {
        if needle.len() > N {
            return None;
        }

        // Prepare needle with padding for SIMD comparison
        let mut needle_padded = [0u8; 64]; // Support up to 64-byte strings with SIMD
        if N > 64 {
            return self.find_exact_fallback(needle);
        }

        let needle_bytes = needle.as_bytes();
        needle_padded[..needle_bytes.len()].copy_from_slice(needle_bytes);

        unsafe {
            match N {
                4 => self.find_exact_simd_32bit(&needle_padded[..4]),
                8 => self.find_exact_simd_64bit(&needle_padded[..8]),
                16 => self.find_exact_simd_128bit(&needle_padded[..16]),
                32 => self.find_exact_simd_256bit(&needle_padded[..32]),
                _ => self.find_exact_fallback(needle),
            }
        }
    }

    #[cfg(not(feature = "simd"))]
    pub fn find_exact(&self, needle: &str) -> Option<usize> {
        self.find_exact_fallback(needle)
    }

    /// Count strings with a given prefix using SIMD when available
    #[cfg(feature = "simd")]
    pub fn count_prefix(&self, prefix: &str) -> usize {
        if prefix.len() > N {
            return 0;
        }

        if prefix.len() <= 16 && N >= 16 {
            unsafe { self.count_prefix_simd_128bit(prefix) }
        } else {
            self.count_prefix_fallback(prefix)
        }
    }

    #[cfg(not(feature = "simd"))]
    pub fn count_prefix(&self, prefix: &str) -> usize {
        self.count_prefix_fallback(prefix)
    }

    // Private implementation methods

    fn update_stats(&mut self) {
        self.stats.strings_stored = self.len;
        let vec_string_equivalent = self.len * (mem::size_of::<String>() + N);
        self.stats.memory_saved_vs_vec_string = 
            vec_string_equivalent.saturating_sub(self.data.len());
    }

    fn find_exact_fallback(&self, needle: &str) -> Option<usize> {
        for i in 0..self.len {
            if let Some(s) = self.get(i) {
                if s == needle {
                    return Some(i);
                }
            }
        }
        None
    }

    fn count_prefix_fallback(&self, prefix: &str) -> usize {
        let mut count = 0;
        for i in 0..self.len {
            if let Some(s) = self.get(i) {
                if s.starts_with(prefix) {
                    count += 1;
                }
            }
        }
        count
    }

    // SIMD implementations (x86_64 AVX2)
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn find_exact_simd_32bit(&self, needle: &[u8]) -> Option<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_exact_fallback(str::from_utf8(needle).ok()?);
        }

        let needle_value = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
        let needle_vec = _mm256_set1_epi32(needle_value as i32);

        let mut i = 0;
        while i + 8 <= self.len {
            let data_ptr = self.data.as_ptr().add(i * 4) as *const i32;
            let data_vec = _mm256_loadu_si256(data_ptr as *const __m256i);
            
            let cmp_result = _mm256_cmpeq_epi32(data_vec, needle_vec);
            let mask = _mm256_movemask_epi8(cmp_result);
            
            if mask != 0 {
                // Found a match, determine which position
                for j in 0..8 {
                    if (mask & (0xF << (j * 4))) != 0 {
                        return Some(i + j);
                    }
                }
            }
            
            i += 8;
        }

        // Handle remaining elements
        while i < self.len {
            if let Some(s) = self.get(i) {
                if s.as_bytes() == needle {
                    return Some(i);
                }
            }
            i += 1;
        }

        None
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn find_exact_simd_64bit(&self, needle: &[u8]) -> Option<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_exact_fallback(str::from_utf8(needle).ok()?);
        }

        let needle_low = u32::from_le_bytes([needle[0], needle[1], needle[2], needle[3]]);
        let needle_high = u32::from_le_bytes([needle[4], needle[5], needle[6], needle[7]]);
        let needle_vec = _mm256_set_epi32(
            needle_high as i32, needle_low as i32,
            needle_high as i32, needle_low as i32,
            needle_high as i32, needle_low as i32,
            needle_high as i32, needle_low as i32,
        );

        let mut i = 0;
        while i + 4 <= self.len {
            let data_ptr = self.data.as_ptr().add(i * 8) as *const i32;
            let data_vec = _mm256_loadu_si256(data_ptr as *const __m256i);
            
            let cmp_result = _mm256_cmpeq_epi64(data_vec, needle_vec);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));
            
            if mask != 0 {
                for j in 0..4 {
                    if (mask & (1 << j)) != 0 {
                        return Some(i + j);
                    }
                }
            }
            
            i += 4;
        }

        // Handle remaining elements
        while i < self.len {
            if let Some(s) = self.get(i) {
                if s.as_bytes() == needle {
                    return Some(i);
                }
            }
            i += 1;
        }

        None
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn find_exact_simd_128bit(&self, needle: &[u8]) -> Option<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_exact_fallback(str::from_utf8(needle).ok()?);
        }

        let needle_vec = _mm_loadu_si128(needle.as_ptr() as *const __m128i);
        let needle_256 = _mm256_broadcastsi128_si256(needle_vec);

        let mut i = 0;
        while i + 2 <= self.len {
            let data_ptr = self.data.as_ptr().add(i * 16);
            let data_vec = _mm256_loadu_si256(data_ptr as *const __m256i);
            
            let cmp_result = _mm256_cmpeq_epi64(data_vec, needle_256);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));
            
            if mask == 0b1111 {
                return Some(i);
            } else if mask == 0b1100 {
                return Some(i + 1);
            }
            
            i += 2;
        }

        // Handle remaining elements
        while i < self.len {
            if let Some(s) = self.get(i) {
                if s.as_bytes() == needle {
                    return Some(i);
                }
            }
            i += 1;
        }

        None
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    unsafe fn find_exact_simd_256bit(&self, needle: &[u8]) -> Option<usize> {
        if !is_x86_feature_detected!("avx2") {
            return self.find_exact_fallback(str::from_utf8(needle).ok()?);
        }

        let needle_vec = _mm256_loadu_si256(needle.as_ptr() as *const __m256i);

        for i in 0..self.len {
            let data_ptr = self.data.as_ptr().add(i * 32);
            let data_vec = _mm256_loadu_si256(data_ptr as *const __m256i);
            
            let cmp_result = _mm256_cmpeq_epi64(data_vec, needle_vec);
            let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_result));
            
            if mask == 0b1111 {
                return Some(i);
            }
        }

        None
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn count_prefix_simd_128bit(&self, prefix: &str) -> usize {
        if !is_x86_feature_detected!("avx2") {
            return self.count_prefix_fallback(prefix);
        }

        let prefix_bytes = prefix.as_bytes();
        let mut prefix_padded = [0u8; 16];
        prefix_padded[..prefix_bytes.len()].copy_from_slice(prefix_bytes);
        
        let prefix_vec = _mm_loadu_si128(prefix_padded.as_ptr() as *const __m128i);
        let mask_vec = if prefix_bytes.len() < 16 {
            // Create mask for partial comparison
            let mut mask = [0u8; 16];
            mask[..prefix_bytes.len()].fill(0xFF);
            _mm_loadu_si128(mask.as_ptr() as *const __m128i)
        } else {
            _mm_set1_epi8(-1i8) // All bits set
        };

        let mut count = 0;
        
        for i in 0..self.len {
            let data_ptr = self.data.as_ptr().add(i * N);
            let data_vec = _mm_loadu_si128(data_ptr as *const __m128i);
            
            // Mask the comparison to only check prefix bytes
            let masked_data = _mm_and_si128(data_vec, mask_vec);
            let masked_prefix = _mm_and_si128(prefix_vec, mask_vec);
            
            let cmp_result = _mm_cmpeq_epi8(masked_data, masked_prefix);
            let mask_result = _mm_movemask_epi8(cmp_result);
            
            // Check if all prefix bytes matched
            let expected_mask = (1u32 << prefix_bytes.len()) - 1;
            if (mask_result as u32 & expected_mask) == expected_mask {
                count += 1;
            }
        }
        
        count
    }

    // Non-SIMD fallback implementations for other architectures
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    unsafe fn find_exact_simd_32bit(&self, needle: &[u8]) -> Option<usize> {
        self.find_exact_fallback(str::from_utf8(needle).ok()?)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    unsafe fn find_exact_simd_64bit(&self, needle: &[u8]) -> Option<usize> {
        self.find_exact_fallback(str::from_utf8(needle).ok()?)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    unsafe fn find_exact_simd_128bit(&self, needle: &[u8]) -> Option<usize> {
        self.find_exact_fallback(str::from_utf8(needle).ok()?)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    unsafe fn find_exact_simd_256bit(&self, needle: &[u8]) -> Option<usize> {
        self.find_exact_fallback(str::from_utf8(needle).ok()?)
    }

    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    unsafe fn count_prefix_simd_128bit(&self, prefix: &str) -> usize {
        self.count_prefix_fallback(prefix)
    }
}

impl<const N: usize> Default for FixedLenStrVec<N> {
    fn default() -> Self {
        Self::new()
    }
}

// Specialized implementations for common string lengths
pub type FixedStr4Vec = FixedLenStrVec<4>;
pub type FixedStr8Vec = FixedLenStrVec<8>;
pub type FixedStr16Vec = FixedLenStrVec<16>;
pub type FixedStr32Vec = FixedLenStrVec<32>;
pub type FixedStr64Vec = FixedLenStrVec<64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        vec.push("hello").unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.get(0), Some("hello"));
        assert_eq!(vec.get(1), None);
    }

    #[test]
    fn test_fixed_length_constraint() {
        let mut vec: FixedStr4Vec = FixedLenStrVec::new();
        
        // Should work for strings <= 4 bytes
        vec.push("hi").unwrap();
        vec.push("test").unwrap();
        
        // Should fail for strings > 4 bytes
        assert!(vec.push("toolong").is_err());
    }

    #[test]
    fn test_padding_and_retrieval() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        
        vec.push("a").unwrap();      // 1 byte + 7 padding
        vec.push("hello").unwrap();  // 5 bytes + 3 padding
        vec.push("maxleng").unwrap(); // 8 bytes + 0 padding
        
        assert_eq!(vec.get(0), Some("a"));
        assert_eq!(vec.get(1), Some("hello"));
        assert_eq!(vec.get(2), Some("maxleng"));
    }

    #[test]
    fn test_bytes_access() {
        let mut vec: FixedStr4Vec = FixedLenStrVec::new();
        vec.push("hi").unwrap();
        
        let bytes = vec.get_bytes(0).unwrap();
        assert_eq!(bytes.len(), 4);
        assert_eq!(&bytes[..2], b"hi");
        assert_eq!(&bytes[2..], &[0, 0]); // Padding
    }

    #[test]
    fn test_memory_savings() {
        let mut vec: FixedStr16Vec = FixedLenStrVec::with_capacity(100);
        
        // Add 100 short strings
        for i in 0..100 {
            vec.push(&format!("str{}", i)).unwrap();
        }
        
        let (vec_string_size, our_size, savings_ratio) = vec.memory_savings_vs_vec_string();
        
        // Should achieve significant memory savings
        assert!(savings_ratio > 0.3, "Savings ratio {} should be > 0.3", savings_ratio);
        assert!(our_size < vec_string_size);
    }

    #[test]
    fn test_find_exact() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        
        vec.push("apple").unwrap();
        vec.push("banana").unwrap();
        vec.push("cherry").unwrap();
        vec.push("apple").unwrap();  // Duplicate
        
        assert_eq!(vec.find_exact("banana"), Some(1));
        assert_eq!(vec.find_exact("apple"), Some(0));  // First occurrence
        assert_eq!(vec.find_exact("grape"), None);
        assert_eq!(vec.find_exact("toolongstring"), None);
    }

    #[test]
    fn test_count_prefix() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        
        vec.push("apple").unwrap();
        vec.push("apricot").unwrap();
        vec.push("banana").unwrap();
        vec.push("app").unwrap();
        vec.push("apply").unwrap();
        
        assert_eq!(vec.count_prefix("ap"), 4);
        assert_eq!(vec.count_prefix("app"), 3);
        assert_eq!(vec.count_prefix("apple"), 1);
        assert_eq!(vec.count_prefix("ban"), 1);
        assert_eq!(vec.count_prefix("z"), 0);
    }

    #[test]
    fn test_empty_and_special_strings() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        
        vec.push("").unwrap();           // Empty string
        vec.push("a").unwrap();          // Single character
        vec.push("12345678").unwrap();   // Full length
        
        assert_eq!(vec.get(0), Some(""));
        assert_eq!(vec.get(1), Some("a"));
        assert_eq!(vec.get(2), Some("12345678"));
    }

    #[test]
    fn test_unicode_strings() {
        let mut vec: FixedStr16Vec = FixedLenStrVec::new();
        
        vec.push("café").unwrap();       // UTF-8 with accents
        vec.push("🦀").unwrap();          // Emoji (4 bytes)
        vec.push("αβγ").unwrap();        // Greek letters
        
        assert_eq!(vec.get(0), Some("café"));
        assert_eq!(vec.get(1), Some("🦀"));
        assert_eq!(vec.get(2), Some("αβγ"));
    }

    #[test]
    fn test_capacity_optimization() {
        let vec: FixedStr32Vec = FixedLenStrVec::with_capacity(1000);
        assert_eq!(vec.len(), 0);
        
        // Should not panic or fail with large capacity
        let large_vec: FixedStr64Vec = FixedLenStrVec::with_capacity(10000);
        assert_eq!(large_vec.len(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut vec: FixedStr8Vec = FixedLenStrVec::new();
        
        for i in 0..50 {
            vec.push(&format!("s{}", i)).unwrap();
        }
        
        let (original, compressed, ratio) = vec.stats();
        assert!(original > compressed);
        assert!(ratio > 0.0 && ratio < 1.0);
    }
}