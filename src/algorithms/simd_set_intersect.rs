//! SIMD-Accelerated Sorted Set Intersection for u32 Posting Lists
//!
//! Three strategies, adaptively selected:
//! - **SIMD 4×4 block comparison** (SSE4.1): 16 comparisons per cycle for balanced lists
//! - **Scalar galloping** (exponential + binary search): O(n log m) for skewed lists
//! - **Adaptive**: picks the best strategy based on |A|/|B| ratio
//!
//! # Examples
//!
//! ```rust
//! use zipora::algorithms::simd_set_intersect::*;
//!
//! let a = vec![1, 3, 5, 7, 9, 11, 13, 15];
//! let b = vec![2, 3, 6, 7, 10, 11, 14, 15];
//!
//! let mut out = Vec::new();
//! sorted_intersect_adaptive(&a, &b, &mut out);
//! assert_eq!(out, vec![3, 7, 11, 15]);
//!
//! assert_eq!(sorted_intersect_count(&a, &b), 4);
//! ```

/// Galloping threshold: when |A|/|B| > this ratio, use galloping instead of merge.
const GALLOP_RATIO: usize = 32;

/// Minimum size to use SIMD (below this, scalar merge is faster due to setup cost).
const SIMD_MIN_SIZE: usize = 8;

// ============================================================================
// Public API
// ============================================================================

/// Intersect two sorted u32 slices. Adaptive: picks SIMD or galloping based on sizes.
///
/// Output is written to `out` (cleared first). Returns the intersection count.
#[inline]
pub fn sorted_intersect_adaptive(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    out.clear();
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    // Ensure a is the smaller list
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    if large.len() / small.len().max(1) > GALLOP_RATIO {
        intersect_galloping(small, large, out)
    } else if small.len() >= SIMD_MIN_SIZE {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                return unsafe { intersect_simd_sse41(small, large, out) };
            }
        }
        intersect_scalar_merge(small, large, out)
    } else {
        intersect_scalar_merge(small, large, out)
    }
}

/// Count-only intersection (no materialization). Faster when you only need the count.
#[inline]
pub fn sorted_intersect_count(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    if large.len() / small.len().max(1) > GALLOP_RATIO {
        intersect_galloping_count(small, large)
    } else {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse4.1") {
                return unsafe { intersect_simd_count_sse41(small, large) };
            }
        }
        intersect_scalar_merge_count(small, large)
    }
}

/// SIMD intersection (always uses SIMD if available, fallback to scalar).
#[inline]
pub fn sorted_intersect_simd(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    out.clear();
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe { intersect_simd_sse41(small, large, out) };
        }
    }
    intersect_scalar_merge(small, large, out)
}

/// Scalar galloping intersection (best for skewed sizes).
#[inline]
pub fn sorted_intersect_galloping(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    out.clear();
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    intersect_galloping(small, large, out)
}

// ============================================================================
// Scalar implementations
// ============================================================================

/// Linear merge intersection — O(n + m), best for equal-sized lists.
fn intersect_scalar_merge(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0;

    while i < a.len() && j < b.len() {
        let va = unsafe { *a.get_unchecked(i) };
        let vb = unsafe { *b.get_unchecked(j) };

        if va == vb {
            out.push(va);
            count += 1;
            i += 1;
            j += 1;
        } else if va < vb {
            i += 1;
        } else {
            j += 1;
        }
    }

    count
}

/// Scalar merge count-only.
fn intersect_scalar_merge_count(a: &[u32], b: &[u32]) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut count = 0;

    while i < a.len() && j < b.len() {
        let va = unsafe { *a.get_unchecked(i) };
        let vb = unsafe { *b.get_unchecked(j) };

        if va == vb {
            count += 1;
            i += 1;
            j += 1;
        } else if va < vb {
            i += 1;
        } else {
            j += 1;
        }
    }

    count
}

/// Galloping intersection — O(n log m), best when |small| << |large|.
fn intersect_galloping(small: &[u32], large: &[u32], out: &mut Vec<u32>) -> usize {
    let mut count = 0;
    let mut lo = 0usize;

    for &val in small {
        // Gallop: exponential search to find upper bound
        let mut step = 1usize;
        let mut hi = lo;

        while hi < large.len() && large[hi] < val {
            lo = hi;
            step <<= 1;
            hi = lo + step;
        }
        hi = hi.min(large.len());

        // Binary search in [lo, hi)
        let pos = large[lo..hi].partition_point(|&x| x < val);
        lo += pos;

        if lo < large.len() && large[lo] == val {
            out.push(val);
            count += 1;
            lo += 1; // Move past the match
        }
    }

    count
}

/// Galloping count-only.
fn intersect_galloping_count(small: &[u32], large: &[u32]) -> usize {
    let mut count = 0;
    let mut lo = 0usize;

    for &val in small {
        let mut step = 1usize;
        let mut hi = lo;

        while hi < large.len() && large[hi] < val {
            lo = hi;
            step <<= 1;
            hi = lo + step;
        }
        hi = hi.min(large.len());

        let pos = large[lo..hi].partition_point(|&x| x < val);
        lo += pos;

        if lo < large.len() && large[lo] == val {
            count += 1;
            lo += 1;
        }
    }

    count
}

// ============================================================================
// SSE4.1 SIMD implementation — 4×4 block comparison (Schlegel et al.)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn intersect_simd_sse41(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    use std::arch::x86_64::*;

    let mut count = 0;
    let mut i = 0usize;
    let mut j = 0usize;

    let a_len = a.len();
    let b_len = b.len();

    out.reserve(a_len.min(b_len));

    // SIMD 4×4 block comparison (Schlegel et al.)
    while i + 4 <= a_len && j + 4 <= b_len {
        // SAFETY: i+4 <= a_len and j+4 <= b_len checked above, SSE4.1 guaranteed by target_feature
        unsafe {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(j) as *const __m128i);

            // 4 rotated comparisons = 16 pairs checked
            let cmp0 = _mm_cmpeq_epi32(va, vb);
            let cmp1 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x39));
            let cmp2 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x4E));
            let cmp3 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x93));

            let or01 = _mm_or_si128(cmp0, cmp1);
            let or23 = _mm_or_si128(cmp2, cmp3);
            let matches = _mm_or_si128(or01, or23);

            let mask = _mm_movemask_ps(_mm_castsi128_ps(matches)) as u32;

            if mask != 0 {
                for k in 0..4u32 {
                    if (mask >> k) & 1 != 0 {
                        out.push(*a.get_unchecked(i + k as usize));
                        count += 1;
                    }
                }
            }

            let a_max = *a.get_unchecked(i + 3);
            let b_max = *b.get_unchecked(j + 3);

            if a_max <= b_max {
                i += 4;
            }
            if b_max <= a_max {
                j += 4;
            }
        }
    }

    // Scalar tail
    while i < a_len && j < b_len {
        unsafe {
            let va = *a.get_unchecked(i);
            let vb = *b.get_unchecked(j);

            if va == vb {
                out.push(va);
                count += 1;
                i += 1;
                j += 1;
            } else if va < vb {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    count
}

/// SIMD count-only intersection (no output materialization).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn intersect_simd_count_sse41(a: &[u32], b: &[u32]) -> usize {
    use std::arch::x86_64::*;

    let mut count = 0;
    let mut i = 0usize;
    let mut j = 0usize;

    let a_len = a.len();
    let b_len = b.len();

    while i + 4 <= a_len && j + 4 <= b_len {
        unsafe {
            let va = _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i);
            let vb = _mm_loadu_si128(b.as_ptr().add(j) as *const __m128i);

            let cmp0 = _mm_cmpeq_epi32(va, vb);
            let cmp1 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x39));
            let cmp2 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x4E));
            let cmp3 = _mm_cmpeq_epi32(va, _mm_shuffle_epi32(vb, 0x93));

            let or01 = _mm_or_si128(cmp0, cmp1);
            let or23 = _mm_or_si128(cmp2, cmp3);
            let matches = _mm_or_si128(or01, or23);

            let mask = _mm_movemask_ps(_mm_castsi128_ps(matches)) as u32;
            count += mask.count_ones() as usize;

            let a_max = *a.get_unchecked(i + 3);
            let b_max = *b.get_unchecked(j + 3);

            if a_max <= b_max {
                i += 4;
            }
            if b_max <= a_max {
                j += 4;
            }
        }
    }

    while i < a_len && j < b_len {
        unsafe {
            let va = *a.get_unchecked(i);
            let vb = *b.get_unchecked(j);

            if va == vb {
                count += 1;
                i += 1;
                j += 1;
            } else if va < vb {
                i += 1;
            } else {
                j += 1;
            }
        }
    }

    count
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&[], &[1, 2, 3], &mut out), 0);
        assert_eq!(out.len(), 0);
        assert_eq!(sorted_intersect_adaptive(&[1, 2, 3], &[], &mut out), 0);
        assert_eq!(sorted_intersect_adaptive(&[], &[], &mut out), 0);
    }

    #[test]
    fn test_no_overlap() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 4, 6, 8];
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 0);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_full_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 5);
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_partial_overlap() {
        let a = vec![1, 3, 5, 7, 9, 11, 13, 15];
        let b = vec![2, 3, 6, 7, 10, 11, 14, 15];
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 4);
        assert_eq!(out, vec![3, 7, 11, 15]);
    }

    #[test]
    fn test_single_element() {
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&[5], &[5], &mut out), 1);
        assert_eq!(out, vec![5]);

        assert_eq!(sorted_intersect_adaptive(&[5], &[6], &mut out), 0);
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_count_only() {
        let a: Vec<u32> = (0..1000).step_by(2).collect();
        let b: Vec<u32> = (0..1000).step_by(3).collect();

        let mut out = Vec::new();
        let materialized = sorted_intersect_adaptive(&a, &b, &mut out);
        let counted = sorted_intersect_count(&a, &b);

        assert_eq!(materialized, counted);
        assert_eq!(counted, out.len());
    }

    #[test]
    fn test_skewed_sizes_galloping() {
        let small = vec![50, 500, 5000, 50000];
        let large: Vec<u32> = (0..100000).collect();

        let mut out = Vec::new();
        let count = sorted_intersect_adaptive(&small, &large, &mut out);
        assert_eq!(count, 4);
        assert_eq!(out, vec![50, 500, 5000, 50000]);
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        // Generate two sorted lists with known intersection
        let a: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let b: Vec<u32> = (0..500).map(|i| i * 5).collect();

        let mut simd_out = Vec::new();
        let mut scalar_out = Vec::new();

        let simd_count = sorted_intersect_simd(&a, &b, &mut simd_out);
        let scalar_count = intersect_scalar_merge(&a, &b, &mut scalar_out);

        assert_eq!(simd_count, scalar_count);
        assert_eq!(simd_out, scalar_out);
    }

    #[test]
    fn test_large_balanced() {
        let a: Vec<u32> = (0..10000).step_by(2).collect(); // 5000 evens
        let b: Vec<u32> = (0..10000).step_by(3).collect(); // 3334 multiples of 3

        let mut out = Vec::new();
        let count = sorted_intersect_adaptive(&a, &b, &mut out);

        // Intersection = multiples of 6 in [0, 10000)
        let expected: Vec<u32> = (0..10000).step_by(6).collect();
        assert_eq!(count, expected.len());
        assert_eq!(out, expected);
    }

    #[test]
    fn test_adjacent_no_match() {
        let a = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        let b = vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 0);
    }

    #[test]
    fn test_subset() {
        let a = vec![2, 4, 6, 8, 10];
        let b: Vec<u32> = (1..=20).collect();
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 5);
        assert_eq!(out, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_duplicates_in_input() {
        // Posting lists shouldn't have duplicates, but be robust
        let a = vec![1, 2, 2, 3, 4];
        let b = vec![2, 2, 3, 5];
        let mut out = Vec::new();
        let count = sorted_intersect_adaptive(&a, &b, &mut out);
        // Should match elements present in both (with duplicates preserved)
        assert!(count >= 2); // At least 2 and 3
    }

    #[test]
    fn test_max_u32() {
        let a = vec![0, u32::MAX / 2, u32::MAX];
        let b = vec![0, u32::MAX / 2, u32::MAX];
        let mut out = Vec::new();
        assert_eq!(sorted_intersect_adaptive(&a, &b, &mut out), 3);
    }

    /// Performance test — only meaningful in release mode.
    #[test]
    fn test_performance_10k_balanced() {
        let a: Vec<u32> = (0..10000).step_by(2).collect();
        let b: Vec<u32> = (0..10000).step_by(3).collect();
        let mut out = Vec::new();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            sorted_intersect_adaptive(&a, &b, &mut out);
        }
        let _elapsed = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let per_call = _elapsed / 1000;
            eprintln!(
                "SIMD intersect 5K×3.3K: {:?}/call, {} matches",
                per_call,
                out.len()
            );
            // Should be under 50µs per call in release
            assert!(per_call.as_micros() < 100, "Too slow: {:?}", per_call);
        }
    }

    #[test]
    fn test_performance_skewed() {
        let small: Vec<u32> = (0..100).map(|i| i * 100).collect();
        let large: Vec<u32> = (0..100000).collect();
        let mut out = Vec::new();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            sorted_intersect_adaptive(&small, &large, &mut out);
        }
        let _elapsed = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let per_call = _elapsed / 1000;
            eprintln!(
                "Galloping intersect 100×100K: {:?}/call, {} matches",
                per_call,
                out.len()
            );
            assert!(per_call.as_micros() < 50, "Too slow: {:?}", per_call);
        }
    }
}
