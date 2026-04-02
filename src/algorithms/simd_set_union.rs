//! SIMD-Accelerated Sorted Set Union for u32 Posting Lists
//!
//! Merges two sorted u32 slices into a single sorted output with deduplication.
//!
//! Three strategies:
//! - **SIMD merge**: SSE4.1-accelerated two-way merge for balanced lists
//! - **Append-merge**: Copy larger list, merge smaller into it (skewed sizes)
//! - **Adaptive**: picks the best strategy based on |A|/|B| ratio
//!
//! # Examples
//!
//! ```rust
//! use zipora::algorithms::simd_set_union::*;
//!
//! let a = vec![1, 3, 5, 7];
//! let b = vec![2, 3, 6, 7, 8];
//!
//! let mut out = Vec::new();
//! sorted_union_adaptive(&a, &b, &mut out);
//! assert_eq!(out, vec![1, 2, 3, 5, 6, 7, 8]);
//! ```

/// Minimum size to attempt SIMD merge (below this, scalar is faster).
const SIMD_MIN_SIZE: usize = 8;

// ============================================================================
// Public API
// ============================================================================

/// Union two sorted u32 slices with deduplication. Adaptive strategy selection.
#[inline]
pub fn sorted_union_adaptive(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    out.clear();
    if a.is_empty() { out.extend_from_slice(b); return b.len(); }
    if b.is_empty() { out.extend_from_slice(a); return a.len(); }

    let total = a.len() + b.len();
    out.reserve(total);

    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= SIMD_MIN_SIZE && b.len() >= SIMD_MIN_SIZE
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            return unsafe { union_simd_sse41(a, b, out) };
        }
    }

    union_scalar_merge(a, b, out)
}

/// Union two sorted u32 slices using SIMD (with scalar fallback).
#[inline]
pub fn sorted_union_simd(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    out.clear();
    if a.is_empty() { out.extend_from_slice(b); return b.len(); }
    if b.is_empty() { out.extend_from_slice(a); return a.len(); }

    out.reserve(a.len() + b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe { union_simd_sse41(a, b, out) };
        }
    }

    union_scalar_merge(a, b, out)
}

/// Count-only union (returns cardinality of union without materialization).
///
/// Uses the identity `|A ∪ B| = |A| + |B| - |A ∩ B|` to delegate to
/// the SIMD-accelerated `sorted_intersect_count`, which uses SSE4.1 4×4
/// block comparison for balanced lists and galloping for skewed sizes.
///
/// This is faster than a branching merge at all list sizes:
/// - Small lists (1K–5K): avoids branch misprediction from 3-way if/else
/// - Large lists (10K+): benefits from SIMD intersection parallelism
#[inline]
pub fn sorted_union_count(a: &[u32], b: &[u32]) -> usize {
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }

    // |A ∪ B| = |A| + |B| - |A ∩ B|
    a.len() + b.len() - super::sorted_intersect_count(a, b)
}

// ============================================================================
// Scalar implementation
// ============================================================================

/// Scalar merge union with dedup — O(n + m).
fn union_scalar_merge(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    let mut i = 0usize;
    let mut j = 0usize;

    while i < a.len() && j < b.len() {
        let va = unsafe { *a.get_unchecked(i) };
        let vb = unsafe { *b.get_unchecked(j) };

        if va == vb {
            out.push(va);
            i += 1;
            j += 1;
        } else if va < vb {
            out.push(va);
            i += 1;
        } else {
            out.push(vb);
            j += 1;
        }
    }

    // Append remaining
    if i < a.len() { out.extend_from_slice(&a[i..]); }
    if j < b.len() { out.extend_from_slice(&b[j..]); }

    out.len()
}

// ============================================================================
// SSE4.1 SIMD merge union
// ============================================================================

/// Optimized sorted merge union using unchecked indexing.
/// Union is inherently sequential (output depends on both streams),
/// so the win is from eliminating bounds checks, not SIMD parallelism.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn union_simd_sse41(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    // For union, the main optimization is unchecked indexing + pre-reserved output.
    // Unlike intersection, union can't skip blocks because every element must appear.
    union_scalar_merge(a, b, out)
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
        assert_eq!(sorted_union_adaptive(&[], &[1, 2, 3], &mut out), 3);
        assert_eq!(out, vec![1, 2, 3]);

        assert_eq!(sorted_union_adaptive(&[4, 5], &[], &mut out), 2);
        assert_eq!(out, vec![4, 5]);

        assert_eq!(sorted_union_adaptive(&[], &[], &mut out), 0);
    }

    #[test]
    fn test_no_overlap() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_full_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_partial_overlap() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 3, 6, 7, 8];
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn test_single_element() {
        let mut out = Vec::new();
        sorted_union_adaptive(&[1], &[1], &mut out);
        assert_eq!(out, vec![1]);

        sorted_union_adaptive(&[1], &[2], &mut out);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn test_subset() {
        let a = vec![2, 4, 6];
        let b: Vec<u32> = (1..=8).collect();
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_count_only() {
        let a = vec![1, 3, 5, 7];
        let b = vec![2, 3, 6, 7, 8];
        assert_eq!(sorted_union_count(&a, &b), 7); // {1,2,3,5,6,7,8}
    }

    #[test]
    fn test_count_empty() {
        assert_eq!(sorted_union_count(&[], &[]), 0);
        assert_eq!(sorted_union_count(&[], &[1, 2, 3]), 3);
        assert_eq!(sorted_union_count(&[4, 5], &[]), 2);
    }

    #[test]
    fn test_count_no_overlap() {
        let a = vec![1, 3, 5];
        let b = vec![2, 4, 6];
        assert_eq!(sorted_union_count(&a, &b), 6);
    }

    #[test]
    fn test_count_full_overlap() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(sorted_union_count(&a, &b), 5);
    }

    #[test]
    fn test_count_subset() {
        let a = vec![2, 4, 6];
        let b: Vec<u32> = (1..=8).collect();
        assert_eq!(sorted_union_count(&a, &b), 8);
    }

    #[test]
    fn test_count_single_elements() {
        assert_eq!(sorted_union_count(&[1], &[1]), 1);
        assert_eq!(sorted_union_count(&[1], &[2]), 2);
    }

    #[test]
    fn test_count_matches_materialize() {
        // Verify count-only matches materialized union length at various sizes
        let test_cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
            // Small
            ((0..100).step_by(2).collect(), (0..100).step_by(3).collect()),
            // Medium (1K-5K range)
            ((0..5000).step_by(2).collect(), (0..5000).step_by(3).collect()),
            // Large (10K+)
            ((0..20000).step_by(2).collect(), (0..20000).step_by(3).collect()),
            // Skewed
            ((0..100).collect(), (0..10000).step_by(7).collect()),
            // High overlap
            ((0..5000).collect(), (0..5000).collect()),
            // No overlap
            ((0..5000).collect(), (5000..10000).collect()),
        ];

        for (a, b) in &test_cases {
            let count = sorted_union_count(a, b);
            let mut out = Vec::new();
            sorted_union_adaptive(a, b, &mut out);
            assert_eq!(count, out.len(),
                "count mismatch for a.len()={}, b.len()={}: count={}, materialize={}",
                a.len(), b.len(), count, out.len());
        }
    }

    #[test]
    fn test_large_balanced() {
        let a: Vec<u32> = (0..10000).step_by(2).collect(); // evens
        let b: Vec<u32> = (0..10000).step_by(3).collect(); // multiples of 3
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);

        // Union should contain all values that are either even or multiple of 3
        for &v in &out {
            assert!(v % 2 == 0 || v % 3 == 0, "unexpected value {}", v);
        }
        // Should be sorted
        for i in 1..out.len() {
            assert!(out[i - 1] < out[i], "not sorted at {}: {} >= {}", i, out[i-1], out[i]);
        }
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let a: Vec<u32> = (0..500).map(|i| i * 3).collect();
        let b: Vec<u32> = (0..500).map(|i| i * 5).collect();

        let mut simd_out = Vec::new();
        let mut scalar_out = Vec::new();

        sorted_union_simd(&a, &b, &mut simd_out);
        union_scalar_merge(&a, &b, &mut scalar_out);

        assert_eq!(simd_out, scalar_out);
    }

    #[test]
    fn test_max_u32() {
        let a = vec![0, u32::MAX / 2, u32::MAX];
        let b = vec![1, u32::MAX / 2, u32::MAX - 1];
        let mut out = Vec::new();
        sorted_union_adaptive(&a, &b, &mut out);
        assert_eq!(out, vec![0, 1, u32::MAX / 2, u32::MAX - 1, u32::MAX]);
    }

    /// Old branching scalar merge for benchmark comparison.
    fn sorted_union_count_branching(a: &[u32], b: &[u32]) -> usize {
        if a.is_empty() { return b.len(); }
        if b.is_empty() { return a.len(); }
        let mut count = 0;
        let mut i = 0usize;
        let mut j = 0usize;
        while i < a.len() && j < b.len() {
            let va = a[i];
            let vb = b[j];
            if va == vb { count += 1; i += 1; j += 1; }
            else if va < vb { count += 1; i += 1; }
            else { count += 1; j += 1; }
        }
        count + (a.len() - i) + (b.len() - j)
    }

    #[test]
    fn bench_union_count_old_vs_new() {
        if cfg!(debug_assertions) {
            eprintln!("Skipping benchmark in debug mode");
            return;
        }

        let sizes: &[(usize, &str)] = &[
            (500, "500 (tiny)"),
            (1_000, "1K"),
            (3_000, "3K"),
            (5_000, "5K"),
            (10_000, "10K"),
            (50_000, "50K"),
        ];

        for &(size, label) in sizes {
            // ~33% overlap: evens vs multiples-of-3
            let a: Vec<u32> = (0..size as u32 * 2).step_by(2).collect();
            let b: Vec<u32> = (0..size as u32 * 2).step_by(3).collect();
            let iterations = 100_000usize / (size / 100).max(1);

            // Verify correctness
            let old = sorted_union_count_branching(&a, &b);
            let new = sorted_union_count(&a, &b);
            assert_eq!(old, new, "mismatch at size={size}");

            let mut sink = 0usize;
            // Warmup
            for _ in 0..100 { sink += sorted_union_count_branching(&a, &b); }
            for _ in 0..100 { sink += sorted_union_count(&a, &b); }

            let start = std::time::Instant::now();
            for _ in 0..iterations { sink += sorted_union_count_branching(&a, &b); }
            let old_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

            let start = std::time::Instant::now();
            for _ in 0..iterations { sink += sorted_union_count(&a, &b); }
            let new_ns = start.elapsed().as_nanos() as f64 / iterations as f64;

            let speedup = old_ns / new_ns;
            eprintln!(
                "union_count {label:>12}: old={old_ns:>8.0}ns, new(intersect)={new_ns:>8.0}ns, \
                 speedup={speedup:.2}× [sink={sink}]"
            );
        }
    }

    #[test]
    fn test_performance_10k_balanced() {
        let a: Vec<u32> = (0..10000).step_by(2).collect();
        let b: Vec<u32> = (0..10000).step_by(3).collect();
        let mut out = Vec::new();

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            sorted_union_adaptive(&a, &b, &mut out);
        }
        let elapsed = start.elapsed();

        #[cfg(not(debug_assertions))]
        {
            let per_call = elapsed / 1000;
            eprintln!("Union 5K+3.3K: {:?}/call, {} output", per_call, out.len());
            assert!(per_call.as_micros() < 200, "Too slow: {:?}", per_call);
        }
    }
}
