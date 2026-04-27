//! Fast byte search in sorted arrays for FSA child-label lookup.
//!
//! Core algorithm for trie node child lookup. The input is a **sorted** byte array
//! of child labels and we need to find the position of a single key byte.
//!
//! # Strategy Selection
//!
//! | Array length | Algorithm                              |
//! |-------------|----------------------------------------|
//! | 0-16        | SSE4.2 `_mm_cmpestri` (single call)    |
//! | 17-35       | SSE4.2 (2-3 calls, `fast_search_byte_max_35`) |
//! | ≥36         | Binary search                          |
//! | (no SSE4.2) | Binary search (all sizes)              |

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::error::Result;

// ============================================================================
// Core sorted-array search functions for sorted byte arrays
// ============================================================================

/// Binary search for `key` in sorted `data[0..len]`.
/// Returns index of `key` if found, or `len` if not found.
#[inline]
pub fn binary_search_byte(data: &[u8], key: u8) -> usize {
    let len = data.len();
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        let mid = (lo + hi) / 2;
        // SAFETY: mid < hi <= len, so mid is in bounds
        if unsafe { *data.get_unchecked(mid) } < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    // SAFETY: lo < len checked by condition, so lo is in bounds
    if lo < len && unsafe { *data.get_unchecked(lo) } == key {
        lo
    } else {
        len
    }
}

/// SSE4.2 search for `key` in sorted `data[0..len]` where `len <= 16`.
/// Returns index of `key` if found, or a value >= `len` if not found.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn sse4_2_search_byte(data: *const u8, len: i32, key: u8) -> usize {
    debug_assert!(len <= 16);
    // SAFETY: SSE4.2 guaranteed by target_feature, buf prevents out-of-bounds reads, len <= 16 enforced by debug_assert
    unsafe {
        let key128 = _mm_set1_epi8(key as i8);
        // Copy to 16-byte stack buffer to avoid reading past allocation boundary.
        // _mm_loadu_si128 always reads 16 bytes, but `data` may point to fewer
        // valid bytes (e.g., a 6-byte trie child-label array on the stack).
        let mut buf = [0u8; 16];
        core::ptr::copy_nonoverlapping(data, buf.as_mut_ptr(), len as usize);
        let data128 = _mm_loadu_si128(buf.as_ptr() as *const __m128i);
        // pcmpestri: find first position of key in data[0..len]
        let idx = _mm_cmpestri(
            key128, 1,           // needle: single byte
            data128, len,        // haystack: data[0..len]
            _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ORDERED | _SIDD_LEAST_SIGNIFICANT,
        );
        idx as usize
    }
}

/// Fast search for `key` in sorted `data[0..len]` where `len <= 35`.
/// Uses up to 3 SSE4.2 calls for optimal performance.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.2")]
unsafe fn fast_search_byte_max_35(data: *const u8, len: usize, key: u8) -> usize {
    debug_assert!(len <= 35);
    if len <= 16 {
        // SAFETY: SSE4.2 guaranteed by #[target_feature], data valid from caller for len bytes
        return unsafe { sse4_2_search_byte(data, len as i32, key) };
    }
    // First 16 bytes
    // SAFETY: SSE4.2 guaranteed by #[target_feature], len > 16 checked above
    let pos = unsafe { sse4_2_search_byte(data, 16, key) };
    if pos < 16 {
        return pos;
    }
    if len <= 32 {
        // SAFETY: data+16 valid since len > 16 and caller ensures len bytes accessible
        let pos2 = unsafe { sse4_2_search_byte(data.add(16), (len - 16) as i32, key) };
        return if pos2 < len - 16 { 16 + pos2 } else { len };
    }
    // 16..32
    // SAFETY: data+16 valid since len > 32 checked above
    let pos2 = unsafe { sse4_2_search_byte(data.add(16), 16, key) };
    if pos2 < 16 {
        return 16 + pos2;
    }
    // 32..len
    // SAFETY: data+32 valid since len > 32 (else would have returned above)
    let pos3 = unsafe { sse4_2_search_byte(data.add(32), (len - 32) as i32, key) };
    if pos3 < len - 32 { 32 + pos3 } else { len }
}

/// Primary entry point: search for `key` in sorted byte array `data`.
/// Returns the index of `key` if found, or `data.len()` if not found.
///
/// Strategy:
/// - ≤16 bytes: SSE4.2 `_mm_cmpestri` (single instruction)
/// - 17-35 bytes: SSE4.2 (2-3 calls)
/// - ≥36 bytes: binary search
///
/// This is the critical hot-path function for trie child-label lookup.
#[inline]
pub fn fast_search_byte(data: &[u8], key: u8) -> usize {
    let len = data.len();
    if len == 0 {
        return 0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: SSE4.2 feature detected at runtime, data.as_ptr() valid from slice
            unsafe {
                if len <= 16 {
                    let idx = sse4_2_search_byte(data.as_ptr(), len as i32, key);
                    return if idx < len { idx } else { len };
                }
                if len <= 35 {
                    return fast_search_byte_max_35(data.as_ptr(), len, key);
                }
            }
        }
    }
    binary_search_byte(data, key)
}

/// Search for `key` in sorted `data`, max 16 bytes.
/// Uses SSE4.2 intrinsics when available for optimal performance.
#[inline]
pub fn fast_search_byte_max_16(data: &[u8], key: u8) -> usize {
    debug_assert!(data.len() <= 16);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: SSE4.2 feature detected at runtime, data.as_ptr() valid from slice
            unsafe {
                let idx = sse4_2_search_byte(data.as_ptr(), data.len() as i32, key);
                return if idx < data.len() { idx } else { data.len() };
            }
        }
    }
    binary_search_byte(data, key)
}

// ============================================================================
// Configuration and engine types (kept for backward compatibility)
// ============================================================================

/// Search algorithm strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Binary search (always correct, O(log n))
    Linear,
    /// SSE-based SIMD search
    Simd,
    /// SSE4.2 string search instructions
    Sse42,
    /// Rank-select accelerated search
    RankSelect,
    /// Adaptive selection based on data size
    Adaptive,
}

/// Configuration for fast search algorithms.
#[derive(Debug, Clone)]
pub struct FastSearchConfig {
    pub strategy: SearchStrategy,
    pub rank_select_threshold: usize,
    pub auto_detect_features: bool,
    pub enable_parallel: bool,
    pub parallel_chunk_size: usize,
}

impl Default for FastSearchConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::Adaptive,
            rank_select_threshold: 36,
            auto_detect_features: true,
            enable_parallel: false,
            parallel_chunk_size: 4096,
        }
    }
}

impl FastSearchConfig {
    pub fn for_small_arrays() -> Self {
        Self { strategy: SearchStrategy::Simd, rank_select_threshold: 1024, enable_parallel: false, ..Default::default() }
    }
    pub fn for_large_arrays() -> Self {
        Self { strategy: SearchStrategy::RankSelect, rank_select_threshold: 16, enable_parallel: true, parallel_chunk_size: 8192, ..Default::default() }
    }
    pub fn performance_optimized() -> Self {
        Self { strategy: SearchStrategy::Adaptive, rank_select_threshold: 64, auto_detect_features: true, enable_parallel: true, parallel_chunk_size: 16384 }
    }
}

/// Hardware capabilities for optimization.
#[derive(Debug, Clone, Copy)]
pub struct HardwareCapabilities {
    pub has_sse42: bool,
    pub has_avx2: bool,
    pub has_bmi2: bool,
    pub has_popcnt: bool,
}

impl HardwareCapabilities {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_sse42: is_x86_feature_detected!("sse4.2"),
                has_avx2: is_x86_feature_detected!("avx2"),
                has_bmi2: is_x86_feature_detected!("bmi2"),
                has_popcnt: is_x86_feature_detected!("popcnt"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self { has_sse42: false, has_avx2: false, has_bmi2: false, has_popcnt: false }
        }
    }

    pub fn best_strategy(&self, data_size: usize, rank_select_threshold: usize) -> SearchStrategy {
        if data_size >= rank_select_threshold { SearchStrategy::RankSelect }
        else if data_size <= 35 && self.has_sse42 { SearchStrategy::Sse42 }
        else if data_size <= 128 { SearchStrategy::Simd }
        else { SearchStrategy::Linear }
    }
}

/// Fast byte search engine — wraps `fast_search_byte` with config.
///
/// For hot-path trie lookups, prefer calling `fast_search_byte()` directly.
pub struct FastSearchEngine {
    _config: FastSearchConfig,
    capabilities: HardwareCapabilities,
    // Removed rank_select_cache — not needed for sorted-array position lookup
}

impl FastSearchEngine {
    pub fn new() -> Self {
        Self::with_config(FastSearchConfig::default())
    }

    pub fn with_config(config: FastSearchConfig) -> Self {
        let capabilities = if config.auto_detect_features {
            HardwareCapabilities::detect()
        } else {
            HardwareCapabilities { has_sse42: false, has_avx2: false, has_bmi2: false, has_popcnt: false }
        };
        Self { _config: config, capabilities }
    }

    /// Search for all occurrences of `target` in `data` (general search).
    pub fn search_byte(&mut self, data: &[u8], target: u8) -> Result<Vec<usize>> {
        Ok(data.iter().enumerate()
            .filter_map(|(i, &b)| if b == target { Some(i) } else { None })
            .collect())
    }

    /// Search for multiple byte values simultaneously.
    pub fn search_multiple(&mut self, data: &[u8], targets: &[u8]) -> Result<Vec<Vec<usize>>> {
        targets.iter().map(|&t| self.search_byte(data, t)).collect()
    }

    /// Find first occurrence of a byte value.
    pub fn find_first(&self, data: &[u8], target: u8) -> Option<usize> {
        data.iter().position(|&b| b == target)
    }

    /// Find last occurrence of a byte value.
    pub fn find_last(&self, data: &[u8], target: u8) -> Option<usize> {
        data.iter().rposition(|&b| b == target)
    }

    /// Count occurrences of a byte value.
    pub fn count_byte(&mut self, data: &[u8], target: u8) -> Result<usize> {
        Ok(data.iter().filter(|&&b| b == target).count())
    }

    pub fn capabilities(&self) -> HardwareCapabilities {
        self.capabilities
    }

    pub fn clear_cache(&mut self) {
        // No-op — rank_select_cache removed
    }
}

impl Default for FastSearchEngine {
    fn default() -> Self { Self::new() }
}

/// Utility functions for fast byte search operations.
pub mod utils {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Search for first occurrence of any target byte.
    pub fn search_any_of(data: &[u8], targets: &[u8]) -> Option<usize> {
        for (i, &byte) in data.iter().enumerate() {
            if targets.contains(&byte) {
                return Some(i);
            }
        }
        None
    }

    /// Search for pattern occurrences in data.
    pub fn search_pattern(data: &[u8], pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() || pattern.len() > data.len() {
            return Vec::new();
        }
        (0..=(data.len() - pattern.len()))
            .filter(|&i| data[i..i + pattern.len()] == *pattern)
            .collect()
    }

    /// Fast popcount using hardware acceleration if available.
    pub fn popcount(data: &[u8]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("popcnt") {
                let mut count = 0usize;
                // SAFETY: popcnt feature detected at runtime
                unsafe {
                    let chunks = data.chunks_exact(8);
                    let remainder = chunks.remainder();
                    for chunk in chunks {
                        let value = u64::from_le_bytes(chunk.try_into().expect("chunk is 8 bytes"));
                        count += _popcnt64(value as i64) as usize;
                    }
                    for &byte in remainder {
                        count += byte.count_ones() as usize;
                    }
                }
                return count;
            }
        }
        data.iter().map(|&b| b.count_ones() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Core sorted-array search tests =====

    #[test]
    fn test_binary_search_byte_basic() {
        let data = [2, 5, 8, 12, 15, 20];
        assert_eq!(binary_search_byte(&data, 5), 1);
        assert_eq!(binary_search_byte(&data, 2), 0);
        assert_eq!(binary_search_byte(&data, 20), 5);
        assert_eq!(binary_search_byte(&data, 12), 3);
    }

    #[test]
    fn test_binary_search_byte_not_found() {
        let data = [2, 5, 8, 12, 15, 20];
        assert_eq!(binary_search_byte(&data, 1), 6);   // before first
        assert_eq!(binary_search_byte(&data, 3), 6);   // between
        assert_eq!(binary_search_byte(&data, 21), 6);  // after last
        assert_eq!(binary_search_byte(&data, 10), 6);  // between
    }

    #[test]
    fn test_binary_search_byte_empty() {
        let data: [u8; 0] = [];
        assert_eq!(binary_search_byte(&data, 42), 0);
    }

    #[test]
    fn test_binary_search_byte_single() {
        assert_eq!(binary_search_byte(&[42], 42), 0);
        assert_eq!(binary_search_byte(&[42], 43), 1);
    }

    #[test]
    fn test_fast_search_byte_small() {
        // Typical trie child labels: sorted, ≤16 bytes
        let labels = [b'a', b'c', b'e', b'g', b'z'];
        assert_eq!(fast_search_byte(&labels, b'a'), 0);
        assert_eq!(fast_search_byte(&labels, b'c'), 1);
        assert_eq!(fast_search_byte(&labels, b'e'), 2);
        assert_eq!(fast_search_byte(&labels, b'z'), 4);
        assert_eq!(fast_search_byte(&labels, b'b'), 5); // not found
        assert_eq!(fast_search_byte(&labels, b'd'), 5); // not found
    }

    #[test]
    fn test_fast_search_byte_16() {
        // Exactly 16 bytes — SSE4.2 boundary
        let data: Vec<u8> = (0..16).map(|i| i * 10).collect();
        for i in 0..16 {
            assert_eq!(fast_search_byte(&data, i * 10), i as usize);
        }
        assert_eq!(fast_search_byte(&data, 5), 16); // not found
    }

    #[test]
    fn test_fast_search_byte_17_to_35() {
        // 17-35 bytes — multi-SSE4.2 path
        let data: Vec<u8> = (0..25).map(|i| i * 5).collect();
        for i in 0..25 {
            assert_eq!(fast_search_byte(&data, i * 5), i as usize);
        }
        assert_eq!(fast_search_byte(&data, 1), 25); // not found
    }

    #[test]
    fn test_fast_search_byte_large() {
        // ≥36 bytes — binary search path
        let data: Vec<u8> = (0..50).map(|i| i * 3).collect();
        for i in 0..50 {
            assert_eq!(fast_search_byte(&data, i * 3), i as usize);
        }
        assert_eq!(fast_search_byte(&data, 1), 50); // not found
    }

    #[test]
    fn test_fast_search_byte_max_16() {
        let data = [1, 3, 5, 7, 9, 11, 13, 15];
        assert_eq!(fast_search_byte_max_16(&data, 7), 3);
        assert_eq!(fast_search_byte_max_16(&data, 6), 8); // not found
    }

    #[test]
    fn test_fast_search_byte_all_256() {
        // Full 256-byte sorted array
        let data: Vec<u8> = (0..=255).collect();
        for i in 0u16..256 {
            assert_eq!(fast_search_byte(&data, i as u8), i as usize);
        }
    }

    #[test]
    fn test_fast_search_byte_duplicates() {
        // With duplicates (returns first occurrence via binary search lower_bound)
        let data = [1, 3, 3, 3, 5, 7];
        let pos = fast_search_byte(&data, 3);
        assert!(pos < data.len());
        assert_eq!(data[pos], 3);
    }

    // ===== Legacy FastSearchEngine tests (backward compat) =====

    #[test]
    fn test_hardware_capabilities() {
        let caps = HardwareCapabilities::detect();
        let _ = caps.has_sse42;
        let _ = caps.has_avx2;
    }

    #[test]
    fn test_fast_search_linear() {
        let mut engine = FastSearchEngine::with_config(FastSearchConfig {
            strategy: SearchStrategy::Linear,
            ..Default::default()
        });
        let data = b"hello world hello";
        let positions = engine.search_byte(data, b'l').unwrap();
        assert_eq!(positions, vec![2, 3, 9, 14, 15]);
    }

    #[test]
    fn test_fast_search_simd() {
        let mut engine = FastSearchEngine::with_config(FastSearchConfig {
            strategy: SearchStrategy::Simd,
            ..Default::default()
        });
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let positions = engine.search_byte(data, b'a').unwrap();
        assert_eq!(positions, vec![0]);
    }

    #[test]
    fn test_find_first_last() {
        let engine = FastSearchEngine::new();
        let data = b"hello world hello";
        assert_eq!(engine.find_first(data, b'l'), Some(2));
        assert_eq!(engine.find_last(data, b'l'), Some(15));
        assert_eq!(engine.find_first(data, b'z'), None);
        assert_eq!(engine.find_last(data, b'z'), None);
    }

    #[test]
    fn test_count_byte() {
        let mut engine = FastSearchEngine::new();
        let data = b"hello world hello";
        assert_eq!(engine.count_byte(data, b'l').unwrap(), 5);
        assert_eq!(engine.count_byte(data, b'o').unwrap(), 3);
        assert_eq!(engine.count_byte(data, b'z').unwrap(), 0);
    }

    #[test]
    fn test_search_multiple() {
        let mut engine = FastSearchEngine::new();
        let data = b"hello world";
        let targets = [b'l', b'o'];
        let results = engine.search_multiple(data, &targets).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![2, 3, 9]);
        assert_eq!(results[1], vec![4, 7]);
    }

    #[test]
    fn test_adaptive_strategy() {
        let mut engine = FastSearchEngine::with_config(FastSearchConfig {
            strategy: SearchStrategy::Adaptive,
            rank_select_threshold: 10,
            ..Default::default()
        });
        let small_data = b"hello";
        let positions = engine.search_byte(small_data, b'l').unwrap();
        assert_eq!(positions, vec![2, 3]);

        let large_data = vec![b'a'; 100];
        let positions = engine.search_byte(&large_data, b'a').unwrap();
        assert_eq!(positions.len(), 100);
    }

    #[test]
    fn test_rank_select_cache() {
        let mut engine = FastSearchEngine::with_config(FastSearchConfig {
            strategy: SearchStrategy::RankSelect,
            ..Default::default()
        });
        let data = b"hello world hello universe";
        let positions1 = engine.search_byte(data, b'l').unwrap();
        let positions2 = engine.search_byte(data, b'l').unwrap();
        assert_eq!(positions1, positions2);
        engine.clear_cache();
        let positions3 = engine.search_byte(data, b'l').unwrap();
        assert_eq!(positions1, positions3);
    }

    #[test]
    fn test_config_variants() {
        let small_config = FastSearchConfig::for_small_arrays();
        assert_eq!(small_config.strategy, SearchStrategy::Simd);
        assert!(!small_config.enable_parallel);
        let large_config = FastSearchConfig::for_large_arrays();
        assert_eq!(large_config.strategy, SearchStrategy::RankSelect);
        assert!(large_config.enable_parallel);
        let perf_config = FastSearchConfig::performance_optimized();
        assert_eq!(perf_config.strategy, SearchStrategy::Adaptive);
        assert!(perf_config.auto_detect_features);
    }

    #[test]
    fn test_utils_search_any_of() {
        let data = b"hello world";
        assert_eq!(utils::search_any_of(data, b"lw"), Some(2));
        assert_eq!(utils::search_any_of(data, b"xz"), None);
    }

    #[test]
    fn test_utils_search_pattern() {
        let data = b"hello world hello universe";
        assert_eq!(utils::search_pattern(data, b"hello"), vec![0, 12]);
        assert_eq!(utils::search_pattern(data, b"xyz"), Vec::<usize>::new());
    }

    #[test]
    fn test_utils_popcount() {
        let data = [0xFF, 0x00, 0x0F, 0xF0];
        assert_eq!(utils::popcount(&data), 16);
    }

    #[test]
    fn test_empty_data() {
        let mut engine = FastSearchEngine::new();
        let empty_data = b"";
        assert_eq!(engine.search_byte(empty_data, b'a').unwrap(), Vec::<usize>::new());
        assert_eq!(engine.find_first(empty_data, b'a'), None);
        assert_eq!(engine.find_last(empty_data, b'a'), None);
        assert_eq!(engine.count_byte(empty_data, b'a').unwrap(), 0);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_large_data_performance() {
        let mut engine = FastSearchEngine::new();
        let large_data = vec![b'a'; 10000];
        let start = std::time::Instant::now();
        let count = engine.count_byte(&large_data, b'a').unwrap();
        let duration = start.elapsed();
        assert_eq!(count, 10000);
        assert!(duration.as_millis() < 100);
    }
}
