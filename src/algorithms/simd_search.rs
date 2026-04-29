//! SIMD-Accelerated Search Primitives for Block-Max WAND
//!
//! Provides cursor-based search primitives optimized with SIMD instructions:
//! - `simd_gallop_to`: Exponential search + SIMD linear scan within sorted u32 slices
//! - `simd_block_filter`: SIMD threshold filtering on f32 score blocks
//!
//! These primitives are designed for the engine's BMW (Block-Max WAND) algorithm,
//! where they replace scalar galloping and block scoring loops.

// ============================================================================
// Public API
// ============================================================================

/// Check if AVX2 is available, cached for performance.
#[inline]
pub fn has_avx2() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Check if SSE2 is available, cached for performance.
#[inline]
pub fn has_sse2() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("sse2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// SIMD-accelerated galloping search within a sorted u32 slice.
/// Advances `*cursor` to the first position where `arr[*cursor] >= target`.
/// Returns true if such a position exists, false if target > last element.
///
/// # Algorithm
/// 1. Quick checks: empty arr, cursor past end, current already >= target
/// 2. Exponential (galloping) phase: step 1,2,4,8,... until arr[pos] >= target
/// 3. SIMD scan phase (AVX2/SSE2): vectorized comparison within narrowed range
/// 4. Scalar tail: remaining elements < vector width
///
/// # Examples
/// ```
/// use zipora::algorithms::simd_search::simd_gallop_to;
///
/// let arr = vec![1, 5, 10, 15, 20, 25, 30];
/// let mut cursor = 0;
///
/// assert!(simd_gallop_to(&arr, &mut cursor, 15));
/// assert_eq!(cursor, 3); // arr[3] == 15
///
/// assert!(simd_gallop_to(&arr, &mut cursor, 22));
/// assert_eq!(cursor, 5); // arr[5] == 25, first >= 22
///
/// assert!(!simd_gallop_to(&arr, &mut cursor, 100));
/// assert_eq!(cursor, arr.len()); // no element >= 100
/// ```
#[inline]
pub fn simd_gallop_to(arr: &[u32], cursor: &mut usize, target: u32) -> bool {
    // Quick checks
    if arr.is_empty() || *cursor >= arr.len() {
        *cursor = arr.len();
        return false;
    }

    // SAFETY: cursor < arr.len() checked above
    let current = unsafe { *arr.get_unchecked(*cursor) };
    if current >= target {
        return true;
    }

    // Exponential search (galloping)
    let mut lo = *cursor;
    let mut step = 1usize;
    let mut hi = lo + step;

    while hi < arr.len() {
        // SAFETY: hi < arr.len() checked above
        let val = unsafe { *arr.get_unchecked(hi) };
        if val >= target {
            break;
        }
        lo = hi;
        step = step.saturating_mul(2);
        hi = lo.saturating_add(step);
    }
    // Adjust hi to be exclusive upper bound, capped at arr.len()
    // If we broke because arr[hi] >= target, we need to include hi in scan
    hi = (hi + 1).min(arr.len());

    // SIMD scan phase within [lo, hi)
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 support verified by runtime check
            return unsafe { gallop_scan_avx2(arr, lo, hi, target, cursor) };
        }
        if has_sse2() {
            // SAFETY: SSE2 support verified by runtime check
            return unsafe { gallop_scan_sse2(arr, lo, hi, target, cursor) };
        }
    }

    gallop_scan_scalar(arr, lo, hi, target, cursor)
}

/// SIMD-accelerated block filter: compare f32 scores against a threshold.
/// Returns (bitmask, count) where bit i is set if scores[i] > theta.
/// Processes up to 64 elements (bitmask is u64).
/// Panics if scores.len() > 64 or doc_ids.len() != scores.len().
///
/// # Examples
/// ```
/// use zipora::algorithms::simd_search::simd_block_filter;
///
/// let doc_ids = vec![1, 2, 3, 4];
/// let scores = vec![10.0, 5.0, 15.0, 3.0];
/// let theta = 7.0;
///
/// let (mask, count) = simd_block_filter(&doc_ids, &scores, theta);
/// assert_eq!(count, 2); // scores[0]=10 and scores[2]=15 are > 7
/// assert_eq!(mask & 1, 1); // bit 0 set (10 > 7)
/// assert_eq!(mask & 2, 0); // bit 1 not set (5 <= 7)
/// assert_eq!(mask & 4, 4); // bit 2 set (15 > 7)
/// assert_eq!(mask & 8, 0); // bit 3 not set (3 <= 7)
/// ```
#[inline]
pub fn simd_block_filter(doc_ids: &[u32], scores: &[f32], theta: f32) -> (u64, usize) {
    assert!(scores.len() <= 64, "Block size must be <= 64 elements");
    assert_eq!(
        doc_ids.len(),
        scores.len(),
        "doc_ids and scores must have same length"
    );

    if scores.is_empty() {
        return (0, 0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: AVX2 support verified by runtime check
            return unsafe { block_filter_avx2(scores, theta) };
        }
    }

    block_filter_scalar(scores, theta)
}

// ============================================================================
// AVX2 implementation (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gallop_scan_avx2(
    arr: &[u32],
    lo: usize,
    hi: usize,
    target: u32,
    cursor: &mut usize,
) -> bool {
    use std::arch::x86_64::*;

    const BIAS: u32 = 0x80000000;
    let mut pos = lo;

    // SAFETY: AVX2 guaranteed by #[target_feature(enable = "avx2")]
    unsafe {
        let target_biased = _mm256_set1_epi32((target ^ BIAS) as i32);

        // Process 8 u32s at a time
        while pos + 8 <= hi {
            // SAFETY: pos + 8 <= hi <= arr.len()
            let arr_vec = _mm256_loadu_si256(arr.as_ptr().add(pos) as *const __m256i);
            let arr_biased = _mm256_xor_si256(arr_vec, _mm256_set1_epi32(BIAS as i32));

            // arr >= target <==> NOT(target > arr)
            let gt_mask = _mm256_cmpgt_epi32(target_biased, arr_biased);
            let movemask = _mm256_movemask_epi8(gt_mask) as u32;

            if movemask != 0xFFFFFFFF {
                // Found at least one element >= target
                // movemask has 32 bits (4 bits per i32), find first zero bit group
                let trailing_ones = movemask.trailing_ones() as usize;
                let elem_idx = trailing_ones / 4;
                *cursor = pos + elem_idx;
                return true;
            }

            pos += 8;
        }
    }

    // Scalar tail
    gallop_scan_scalar(arr, pos, hi, target, cursor)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn gallop_scan_sse2(
    arr: &[u32],
    lo: usize,
    hi: usize,
    target: u32,
    cursor: &mut usize,
) -> bool {
    use std::arch::x86_64::*;

    const BIAS: u32 = 0x80000000;
    let mut pos = lo;

    // SAFETY: SSE2 guaranteed by #[target_feature(enable = "sse2")]
    unsafe {
        let target_biased = _mm_set1_epi32((target ^ BIAS) as i32);

        // Process 4 u32s at a time
        while pos + 4 <= hi {
            // SAFETY: pos + 4 <= hi <= arr.len()
            let arr_vec = _mm_loadu_si128(arr.as_ptr().add(pos) as *const __m128i);
            let arr_biased = _mm_xor_si128(arr_vec, _mm_set1_epi32(BIAS as i32));

            // arr >= target <==> NOT(target > arr)
            let gt_mask = _mm_cmpgt_epi32(target_biased, arr_biased);
            let movemask = _mm_movemask_epi8(gt_mask) as u32;

            if movemask != 0xFFFF {
                // Found at least one element >= target
                let trailing_ones = movemask.trailing_ones() as usize;
                let elem_idx = trailing_ones / 4;
                *cursor = pos + elem_idx;
                return true;
            }

            pos += 4;
        }
    }

    // Scalar tail
    gallop_scan_scalar(arr, pos, hi, target, cursor)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_filter_avx2(scores: &[f32], theta: f32) -> (u64, usize) {
    use std::arch::x86_64::*;

    let mut mask = 0u64;
    let mut pos = 0usize;

    // SAFETY: AVX2 guaranteed by #[target_feature(enable = "avx2")]
    unsafe {
        let theta_vec = _mm256_set1_ps(theta);

        // Process 8 f32s at a time
        while pos + 8 <= scores.len() {
            // SAFETY: pos + 8 <= scores.len()
            let scores_vec = _mm256_loadu_ps(scores.as_ptr().add(pos));

            // _CMP_GT_OQ = 14 (greater than, ordered, quiet)
            let cmp = _mm256_cmp_ps::<14>(scores_vec, theta_vec);
            let movemask = _mm256_movemask_ps(cmp) as u64;

            // Shift and OR into u64 mask
            mask |= movemask << pos;
            pos += 8;
        }
    }

    // Scalar tail
    while pos < scores.len() {
        // SAFETY: pos < scores.len()
        if unsafe { *scores.get_unchecked(pos) } > theta {
            mask |= 1u64 << pos;
        }
        pos += 1;
    }

    let count = mask.count_ones() as usize;
    (mask, count)
}

// ============================================================================
// Scalar fallback
// ============================================================================

fn gallop_scan_scalar(arr: &[u32], lo: usize, hi: usize, target: u32, cursor: &mut usize) -> bool {
    let mut pos = lo;

    while pos < hi {
        // SAFETY: pos < hi <= arr.len()
        let val = unsafe { *arr.get_unchecked(pos) };
        if val >= target {
            *cursor = pos;
            return true;
        }
        pos += 1;
    }

    *cursor = arr.len();
    false
}

fn block_filter_scalar(scores: &[f32], theta: f32) -> (u64, usize) {
    let mut mask = 0u64;

    for (i, &score) in scores.iter().enumerate() {
        if score > theta {
            mask |= 1u64 << i;
        }
    }

    let count = mask.count_ones() as usize;
    (mask, count)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // simd_gallop_to tests
    // ========================================================================

    #[test]
    fn test_gallop_empty_array() {
        let arr: Vec<u32> = vec![];
        let mut cursor = 0;
        assert!(!simd_gallop_to(&arr, &mut cursor, 10));
        assert_eq!(cursor, 0);
    }

    #[test]
    fn test_gallop_cursor_past_end() {
        let arr = vec![1, 2, 3, 4, 5];
        let mut cursor = 10;
        assert!(!simd_gallop_to(&arr, &mut cursor, 3));
        assert_eq!(cursor, arr.len());
    }

    #[test]
    fn test_gallop_already_satisfied() {
        let arr = vec![1, 5, 10, 15, 20];
        let mut cursor = 2;
        assert!(simd_gallop_to(&arr, &mut cursor, 10));
        assert_eq!(cursor, 2); // arr[2] == 10
    }

    #[test]
    fn test_gallop_target_past_end() {
        let arr = vec![1, 5, 10, 15, 20];
        let mut cursor = 0;
        assert!(!simd_gallop_to(&arr, &mut cursor, 100));
        assert_eq!(cursor, arr.len());
    }

    #[test]
    fn test_gallop_exact_match() {
        let arr = vec![1, 5, 10, 15, 20, 25, 30];
        let mut cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 15));
        assert_eq!(cursor, 3); // arr[3] == 15
    }

    #[test]
    fn test_gallop_between_elements() {
        let arr = vec![1, 5, 10, 15, 20, 25, 30];
        let mut cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 12));
        assert_eq!(cursor, 3); // arr[3] == 15, first >= 12
    }

    #[test]
    fn test_gallop_single_element() {
        let arr = vec![42];
        let mut cursor = 0;

        // Target below
        assert!(simd_gallop_to(&arr, &mut cursor, 40));
        assert_eq!(cursor, 0);

        // Target equal
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 42));
        assert_eq!(cursor, 0);

        // Target above
        cursor = 0;
        assert!(!simd_gallop_to(&arr, &mut cursor, 50));
        assert_eq!(cursor, 1);
    }

    #[test]
    fn test_gallop_small_array() {
        // Array smaller than SIMD width (8 for AVX2, 4 for SSE2)
        let arr = vec![1, 3, 5];
        let mut cursor = 0;

        assert!(simd_gallop_to(&arr, &mut cursor, 4));
        assert_eq!(cursor, 2); // arr[2] == 5
    }

    #[test]
    fn test_gallop_large_array() {
        // 10K+ elements
        let arr: Vec<u32> = (0..10000).map(|i| i * 2).collect();
        let mut cursor = 0;

        // Target at beginning
        assert!(simd_gallop_to(&arr, &mut cursor, 100));
        assert_eq!(cursor, 50); // arr[50] == 100

        // Target in middle
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 10000));
        assert_eq!(cursor, 5000); // arr[5000] == 10000

        // Target near end
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 19500));
        assert_eq!(cursor, 9750); // arr[9750] == 19500

        // Target between elements
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 1001));
        assert_eq!(cursor, 501); // arr[501] == 1002
    }

    #[test]
    fn test_gallop_sequential() {
        let arr: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let mut cursor = 0;

        // Multiple sequential gallops
        assert!(simd_gallop_to(&arr, &mut cursor, 500));
        assert_eq!(cursor, 50);

        assert!(simd_gallop_to(&arr, &mut cursor, 750));
        assert_eq!(cursor, 75);

        assert!(simd_gallop_to(&arr, &mut cursor, 1000));
        assert_eq!(cursor, 100);

        assert!(simd_gallop_to(&arr, &mut cursor, 9000));
        assert_eq!(cursor, 900);
    }

    #[test]
    fn test_gallop_u32_max_boundary() {
        let arr = vec![
            u32::MAX - 1000,
            u32::MAX - 500,
            u32::MAX - 100,
            u32::MAX - 10,
            u32::MAX,
        ];
        let mut cursor = 0;

        // Target below MAX
        assert!(simd_gallop_to(&arr, &mut cursor, u32::MAX - 200));
        assert_eq!(cursor, 2); // arr[2] == MAX - 100

        // Target equal to MAX
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, u32::MAX));
        assert_eq!(cursor, 4);

        // Verify already at MAX
        cursor = 4;
        assert!(simd_gallop_to(&arr, &mut cursor, u32::MAX - 1));
        assert_eq!(cursor, 4);
    }

    // ========================================================================
    // simd_block_filter tests
    // ========================================================================

    #[test]
    fn test_filter_empty() {
        let doc_ids: Vec<u32> = vec![];
        let scores: Vec<f32> = vec![];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);
        assert_eq!(mask, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_filter_all_above() {
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![10.0, 20.0, 30.0, 40.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);
        assert_eq!(mask, 0b1111);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_filter_all_below() {
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 10.0);
        assert_eq!(mask, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_filter_mixed() {
        let doc_ids = vec![1, 2, 3, 4, 5, 6];
        let scores = vec![10.0, 5.0, 15.0, 3.0, 20.0, 7.5];
        let theta = 7.0;
        let (mask, count) = simd_block_filter(&doc_ids, &scores, theta);

        // scores[0]=10 > 7, scores[2]=15 > 7, scores[4]=20 > 7, scores[5]=7.5 > 7
        assert_eq!(count, 4);
        assert_eq!(mask & (1 << 0), 1 << 0); // bit 0 set
        assert_eq!(mask & (1 << 1), 0); // bit 1 not set
        assert_eq!(mask & (1 << 2), 1 << 2); // bit 2 set
        assert_eq!(mask & (1 << 3), 0); // bit 3 not set
        assert_eq!(mask & (1 << 4), 1 << 4); // bit 4 set
        assert_eq!(mask & (1 << 5), 1 << 5); // bit 5 set
    }

    #[test]
    fn test_filter_exact_boundary() {
        // score == theta should NOT qualify (strict >)
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![10.0, 5.0, 5.0, 15.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);

        // Only scores[0]=10 and scores[3]=15 are > 5.0
        assert_eq!(count, 2);
        assert_eq!(mask & (1 << 0), 1 << 0);
        assert_eq!(mask & (1 << 1), 0);
        assert_eq!(mask & (1 << 2), 0);
        assert_eq!(mask & (1 << 3), 1 << 3);
    }

    #[test]
    fn test_filter_nan_scores() {
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![10.0, f32::NAN, 15.0, 5.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 7.0);

        // NaN comparisons always return false, so NaN should not qualify
        // scores[0]=10 > 7, scores[2]=15 > 7
        assert_eq!(count, 2);
        assert_eq!(mask & (1 << 0), 1 << 0);
        assert_eq!(mask & (1 << 1), 0); // NaN not > 7
        assert_eq!(mask & (1 << 2), 1 << 2);
        assert_eq!(mask & (1 << 3), 0);
    }

    #[test]
    fn test_filter_single_element() {
        let doc_ids = vec![1];
        let scores = vec![10.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);
        assert_eq!(mask, 1);
        assert_eq!(count, 1);

        let (mask2, count2) = simd_block_filter(&doc_ids, &scores, 15.0);
        assert_eq!(mask2, 0);
        assert_eq!(count2, 0);
    }

    #[test]
    fn test_filter_full_64() {
        let doc_ids: Vec<u32> = (0..64).collect();
        let scores: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 31.5);

        // Scores 32..64 are > 31.5, so 32 elements qualify
        assert_eq!(count, 32);

        // Check that bits 32..64 are set
        for i in 0..32 {
            assert_eq!(mask & (1u64 << i), 0, "bit {} should not be set", i);
        }
        for i in 32..64 {
            assert_eq!(mask & (1u64 << i), 1u64 << i, "bit {} should be set", i);
        }
    }

    #[test]
    fn test_filter_8_elements() {
        // Exactly one AVX2 iteration
        let doc_ids: Vec<u32> = (0..8).collect();
        let scores = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);

        // scores[1]=10, scores[3]=20, scores[5]=30, scores[7]=40 are > 5.0
        assert_eq!(count, 4);
        assert_eq!(mask, 0b10101010);
    }

    #[test]
    #[should_panic(expected = "Block size must be <= 64 elements")]
    fn test_filter_too_large() {
        let doc_ids: Vec<u32> = (0..65).collect();
        let scores: Vec<f32> = vec![1.0; 65];
        let _ = simd_block_filter(&doc_ids, &scores, 0.5);
    }

    #[test]
    #[should_panic(expected = "doc_ids and scores must have same length")]
    fn test_filter_mismatched_lengths() {
        let doc_ids = vec![1, 2, 3];
        let scores = vec![1.0, 2.0];
        let _ = simd_block_filter(&doc_ids, &scores, 0.5);
    }

    // ========================================================================
    // Performance tests (release-only)
    // ========================================================================

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_gallop_performance() {
        use std::time::Instant;

        // 100K array, 10K queries
        let arr: Vec<u32> = (0..100_000).map(|i| i * 2).collect();
        let queries: Vec<u32> = (0..10_000).map(|i| i * 20 + 7).collect();

        let start = Instant::now();
        let mut cursor = 0;
        let mut found_count = 0;

        for &target in &queries {
            if simd_gallop_to(&arr, &mut cursor, target) {
                found_count += 1;
            }
            cursor = 0; // Reset for next query
        }

        let elapsed = start.elapsed();

        assert_eq!(found_count, 10_000); // All queries should find something

        // Should complete in reasonable time (< 10ms on modern hardware)
        assert!(
            elapsed.as_millis() < 100,
            "Performance test too slow: {:?}",
            elapsed
        );

        println!(
            "Gallop performance: {} queries in {:?} ({:.2} ns/query)",
            queries.len(),
            elapsed,
            elapsed.as_nanos() as f64 / queries.len() as f64
        );
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_filter_performance() {
        use std::time::Instant;

        // 10K calls with 64-element blocks
        let doc_ids: Vec<u32> = (0..64).collect();
        let scores: Vec<f32> = (0..64).map(|i| i as f32 * 1.5).collect();
        let theta = 50.0;

        let start = Instant::now();
        let mut total_count = 0;

        for _ in 0..10_000 {
            let (_, count) = simd_block_filter(&doc_ids, &scores, theta);
            total_count += count;
        }

        let elapsed = start.elapsed();

        assert!(total_count > 0);

        // Should complete in reasonable time (< 50ms on modern hardware)
        assert!(
            elapsed.as_millis() < 100,
            "Performance test too slow: {:?}",
            elapsed
        );

        println!(
            "Filter performance: 10K calls in {:?} ({:.2} ns/call)",
            elapsed,
            elapsed.as_nanos() as f64 / 10_000.0
        );
    }

    #[test]
    fn test_gallop_target_zero() {
        // Target 0 exercises the lower u32 boundary with the XOR bias trick
        let arr = vec![0, 10, 20, 30];
        let mut cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 0));
        assert_eq!(cursor, 0);

        // Array without 0 — target 0 is less than all elements
        let arr2 = vec![5, 10, 15];
        let mut cursor2 = 0;
        assert!(simd_gallop_to(&arr2, &mut cursor2, 0));
        assert_eq!(cursor2, 0); // arr2[0]=5 >= 0
    }

    #[test]
    fn test_gallop_duplicates() {
        let arr = vec![1, 5, 5, 5, 10, 10, 20];
        let mut cursor = 0;

        // Should land on first occurrence of 5
        assert!(simd_gallop_to(&arr, &mut cursor, 5));
        assert_eq!(cursor, 1);

        // Search for value between duplicates
        cursor = 0;
        assert!(simd_gallop_to(&arr, &mut cursor, 7));
        assert_eq!(cursor, 4); // First value >= 7 is arr[4]=10
    }

    #[test]
    fn test_gallop_simd_boundary_sizes() {
        // Length 4: exactly one SSE2 vector
        let arr4 = vec![10, 20, 30, 40];
        let mut cursor = 0;
        assert!(simd_gallop_to(&arr4, &mut cursor, 25));
        assert_eq!(cursor, 2); // arr[2]=30

        // Length 7: just under AVX2 width
        let arr7 = vec![1, 2, 3, 4, 5, 6, 7];
        cursor = 0;
        assert!(simd_gallop_to(&arr7, &mut cursor, 5));
        assert_eq!(cursor, 4);

        // Length 9: one AVX2 vector + 1 scalar
        let arr9 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        cursor = 0;
        assert!(simd_gallop_to(&arr9, &mut cursor, 9));
        assert_eq!(cursor, 8);

        // Length 8: exactly one AVX2 vector
        let arr8 = vec![10, 20, 30, 40, 50, 60, 70, 80];
        cursor = 0;
        assert!(simd_gallop_to(&arr8, &mut cursor, 45));
        assert_eq!(cursor, 4); // arr[4]=50

        // Length 16: exactly two AVX2 vectors
        let arr16: Vec<u32> = (1..=16).collect();
        cursor = 0;
        assert!(simd_gallop_to(&arr16, &mut cursor, 15));
        assert_eq!(cursor, 14);
    }

    #[test]
    fn test_gallop_mid_array_advance() {
        let arr: Vec<u32> = (0..100).map(|i| i * 10).collect();
        let mut cursor = 50; // Start at arr[50]=500

        // Search for value requiring advance from mid-position
        assert!(simd_gallop_to(&arr, &mut cursor, 750));
        assert_eq!(cursor, 75);

        // Continue advancing from new position
        assert!(simd_gallop_to(&arr, &mut cursor, 900));
        assert_eq!(cursor, 90);

        // Target past end from mid-position
        assert!(!simd_gallop_to(&arr, &mut cursor, 1000));
        assert_eq!(cursor, arr.len());
    }

    #[test]
    fn test_filter_infinity() {
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![f32::INFINITY, 10.0, f32::NEG_INFINITY, 5.0];

        // +INF > any finite theta
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 100.0);
        assert_eq!(count, 1);
        assert_ne!(mask & (1 << 0), 0); // +INF qualifies

        // theta = +INF: nothing qualifies (not even +INF since not > +INF)
        let (mask2, count2) = simd_block_filter(&doc_ids, &scores, f32::INFINITY);
        assert_eq!(count2, 0);
        assert_eq!(mask2, 0);

        // theta = -INF: everything except -INF qualifies (> not >=)
        let (mask3, count3) = simd_block_filter(&doc_ids, &scores, f32::NEG_INFINITY);
        assert_eq!(count3, 3); // +INF, 10.0, 5.0 are all > -INF
        assert_ne!(mask3 & (1 << 0), 0); // +INF
        assert_ne!(mask3 & (1 << 1), 0); // 10.0
        assert_eq!(mask3 & (1 << 2), 0); // -INF not > -INF
        assert_ne!(mask3 & (1 << 3), 0); // 5.0
    }

    #[test]
    fn test_filter_negative_scores() {
        let doc_ids = vec![1, 2, 3, 4];
        let scores = vec![-5.0, -2.0, 3.0, -10.0];
        let theta = -3.0;

        // -2.0 > -3.0 and 3.0 > -3.0
        let (mask, count) = simd_block_filter(&doc_ids, &scores, theta);
        assert_eq!(count, 2);
        assert_ne!(mask & (1 << 1), 0); // -2.0
        assert_ne!(mask & (1 << 2), 0); // 3.0
        assert_eq!(mask & (1 << 0), 0); // -5.0 not > -3.0
        assert_eq!(mask & (1 << 3), 0); // -10.0 not > -3.0
    }

    #[test]
    fn test_filter_63_elements() {
        let doc_ids: Vec<u32> = (0..63).collect();
        let scores: Vec<f32> = (0..63).map(|i| i as f32).collect();
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 31.0);

        // Scores 32..63 are > 31.0 (31 elements)
        assert_eq!(count, 31);
        for i in 0..=31 {
            assert_eq!(mask & (1u64 << i), 0, "score {} should not qualify", i);
        }
        for i in 32..63 {
            assert_ne!(mask & (1u64 << i), 0, "score {} should qualify", i);
        }
    }

    #[test]
    fn test_filter_7_elements() {
        // Just under AVX2 width
        let doc_ids: Vec<u32> = (0..7).collect();
        let scores = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 4.0);
        // 5.0, 7.0, 6.0 are > 4.0 → indices 1, 3, 5
        assert_eq!(count, 3);
        assert_ne!(mask & (1 << 1), 0);
        assert_ne!(mask & (1 << 3), 0);
        assert_ne!(mask & (1 << 5), 0);
    }

    #[test]
    fn test_filter_9_elements() {
        // Just over AVX2 width — exercises tail handling
        let doc_ids: Vec<u32> = (0..9).collect();
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (mask, count) = simd_block_filter(&doc_ids, &scores, 5.0);
        // 6,7,8,9 are > 5.0 → indices 5,6,7,8
        assert_eq!(count, 4);
        for i in 0..=4 {
            assert_eq!(mask & (1u64 << i), 0);
        }
        for i in 5..9 {
            assert_ne!(mask & (1u64 << i), 0);
        }
    }

    #[test]
    fn test_gallop_consistency_simd_vs_scalar() {
        // Verify SIMD and scalar produce identical results
        let arr: Vec<u32> = (0..1000).map(|i| i * 7 + 3).collect();
        let targets: Vec<u32> = (0..200).map(|i| i * 35).collect();

        for &target in &targets {
            let mut cursor_test = 0;
            let result = simd_gallop_to(&arr, &mut cursor_test, target);

            // Verify against linear scan
            let expected = arr.iter().position(|&v| v >= target);
            match expected {
                Some(pos) => {
                    assert!(result, "should find target {}", target);
                    assert_eq!(cursor_test, pos, "wrong position for target {}", target);
                }
                None => {
                    assert!(!result, "should not find target {}", target);
                    assert_eq!(cursor_test, arr.len());
                }
            }
        }
    }

    #[test]
    fn test_filter_consistency_simd_vs_scalar() {
        // Verify SIMD and scalar produce identical results
        let doc_ids: Vec<u32> = (0..64).collect();
        let scores: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 10.0).collect();

        for theta_i in -20i32..30 {
            let theta = theta_i as f32 * 0.5;
            let (mask, count) = simd_block_filter(&doc_ids, &scores, theta);

            // Compare against manual scalar
            let mut expected_mask = 0u64;
            let mut expected_count = 0;
            for (i, &s) in scores.iter().enumerate() {
                if s > theta {
                    expected_mask |= 1u64 << i;
                    expected_count += 1;
                }
            }
            assert_eq!(mask, expected_mask, "mask mismatch for theta={}", theta);
            assert_eq!(count, expected_count, "count mismatch for theta={}", theta);
        }
    }
}
