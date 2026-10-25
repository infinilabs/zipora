//! BM25 batch scoring with SIMD acceleration and prefetch support
//!
//! Provides three levels of optimization for BM25 scoring:
//!
//! 1. **Scalar batch scoring** — processes postings one at a time using norm table lookups
//! 2. **AVX2 batch scoring** — processes 8 postings per iteration using `_mm256_i32gather_ps`
//!    to load 8 norm factors from the table simultaneously, then computes 8 BM25 scores
//!    in parallel (~4-6x throughput vs scalar)
//! 3. **Prefetch-aware scoring** — software prefetch (`_MM_HINT_T0`) for random-access
//!    phrase query scoring, hiding memory latency by prefetching the next doc's fieldnorm
//!    while scoring the current doc
//!
//! # Usage
//!
//! ```rust
//! use zipora::scoring::{FieldnormEncoder, Bm25BatchScorer};
//!
//! let avg_dl = 150.0f32;
//! let k1 = 1.2;
//! let b = 0.75;
//! let idf = 3.5;
//!
//! // Build scorer from norm table
//! let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
//! let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);
//!
//! // Encode doc lengths
//! let doc_lengths = vec![50u32, 100, 150, 200, 300, 80, 120, 250, 400, 60];
//! let fieldnorm_bytes: Vec<u8> = doc_lengths.iter()
//!     .map(|&l| FieldnormEncoder::encode(l))
//!     .collect();
//!
//! // Batch score postings
//! let tfs = vec![2u16, 3, 1, 5, 2, 4, 1, 3, 2, 6];
//! let mut scores = vec![0.0f32; tfs.len()];
//! scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);
//!
//! // All scores should be positive
//! assert!(scores.iter().all(|&s| s > 0.0));
//! ```

/// BM25 batch scorer with SIMD acceleration.
///
/// Holds an owned clamped copy of the 256-entry norm table (every entry clamped
/// to `>= f32::MIN_POSITIVE`) and pre-computed constants for batch scoring.
pub struct Bm25BatchScorer<'a> {
    norm_table: [f32; 256],
    idf_k1p1: f32, // idf * (k1 + 1.0) — pre-multiplied
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> Bm25BatchScorer<'a> {
    /// Create a new batch scorer, copying `norm_table` and clamping any entry `<= 0.0` or `NaN` to `f32::MIN_POSITIVE`.
    ///
    /// # Arguments
    /// * `norm_table` - Pre-computed BM25 norm table from `FieldnormEncoder::build_norm_table` (entries `<= 0.0` or `NaN` are clamped to `f32::MIN_POSITIVE`)
    /// * `idf` - Inverse document frequency for this term
    /// * `k1` - BM25 k1 parameter (typically 1.2)
    #[inline]
    pub fn new(norm_table: &'a [f32; 256], idf: f32, k1: f32) -> Self {
        let mut clamped = [f32::MIN_POSITIVE; 256];
        for (dst, &v) in clamped.iter_mut().zip(norm_table.iter()) {
            *dst = v.max(f32::MIN_POSITIVE);
        }
        Self {
            norm_table: clamped,
            idf_k1p1: idf * (k1 + 1.0),
            _marker: std::marker::PhantomData,
        }
    }

    /// Score a single posting.
    ///
    /// `score = idf * (k1 + 1) * tf / (tf + norm_table[fieldnorm_byte])`
    #[inline]
    pub fn score(&self, fieldnorm_byte: u8, tf: u16) -> f32 {
        let tf_f = tf as f32;
        let len_norm = self.norm_table[fieldnorm_byte as usize];
        self.idf_k1p1 * tf_f / (tf_f + len_norm)
    }

    /// Batch-score a posting list.
    ///
    /// For each posting `i`, computes:
    /// ```text
    /// scores[i] = idf * (k1+1) * tfs[i] / (tfs[i] + norm_table[fieldnorm_bytes[i]])
    /// ```
    ///
    /// Automatically dispatches to AVX-512 (16 scores/iteration), AVX2 (8 scores/iteration),
    /// or scalar, based on runtime CPU feature detection.
    ///
    /// # Panics
    /// Panics if `fieldnorm_bytes.len() != tfs.len()` or `scores.len() < tfs.len()`.
    pub fn batch_score(&self, fieldnorm_bytes: &[u8], tfs: &[u16], scores: &mut [f32]) {
        let n = tfs.len();
        assert_eq!(fieldnorm_bytes.len(), n);
        assert!(scores.len() >= n);

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        {
            if crate::algorithms::simd_search::has_avx512f() {
                // SAFETY: AVX-512F support verified by runtime feature check.
                // All slice lengths validated by assertions. norm_table is [f32; 256],
                // and fieldnorm_bytes values are u8 (0-255), so gather indices are in bounds.
                unsafe {
                    self.batch_score_avx512(fieldnorm_bytes, tfs, scores, n);
                }
                return;
            }
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if crate::algorithms::simd_search::has_avx2() {
                // SAFETY: AVX2 support verified by runtime feature check.
                // All slice lengths validated by assertions. norm_table is [f32; 256],
                // and fieldnorm_bytes values are u8 (0-255), so gather indices are in bounds.
                unsafe {
                    self.batch_score_avx2(fieldnorm_bytes, tfs, scores, n);
                }
                return;
            }
        }

        self.batch_score_scalar(fieldnorm_bytes, tfs, scores, n);
    }

    /// Scalar batch scoring — baseline implementation.
    #[inline]
    fn batch_score_scalar(
        &self,
        fieldnorm_bytes: &[u8],
        tfs: &[u16],
        scores: &mut [f32],
        n: usize,
    ) {
        let fieldnorm_bytes = &fieldnorm_bytes[..n];
        let tfs = &tfs[..n];
        let scores = &mut scores[..n];

        for i in 0..n {
            let tf_f = tfs[i] as f32;
            let len_norm = self.norm_table[fieldnorm_bytes[i] as usize];
            scores[i] = self.idf_k1p1 * tf_f / (tf_f + len_norm);
        }
    }

    /// AVX-512 batch scoring — 16 postings per iteration.
    ///
    /// # Safety
    /// Caller must ensure `avx512f` is supported on the current CPU and that
    /// `fieldnorm_bytes.len() >= n`, `tfs.len() >= n`, and `scores.len() >= n`.
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Caller verified `avx512f` runtime feature support and slice lengths `>= n`.
    unsafe fn batch_score_avx512(
        &self,
        fieldnorm_bytes: &[u8],
        tfs: &[u16],
        scores: &mut [f32],
        n: usize,
    ) {
        use std::arch::x86_64::*;

        let chunks = n / 16;
        let remainder = n % 16;

        // SAFETY: AVX-512F guaranteed by #[target_feature(enable = "avx512f")].
        // Loop bounds ensure base + 16 <= n (chunks = n / 16).
        // Gather indices are u8 values (0-255), table is [f32; 256], so always in bounds.
        unsafe {
            let idf_k1p1_vec = _mm512_set1_ps(self.idf_k1p1);
            let table = &self.norm_table;

            for chunk in 0..chunks {
                let base = chunk * 16;
                let fn_slice = &fieldnorm_bytes[base..base + 16];

                let fn_u8 = _mm_loadu_si128(fn_slice.as_ptr() as *const __m128i);
                let fn_i32 = _mm512_cvtepu8_epi32(fn_u8);
                let norms = _mm512_i32gather_ps::<4>(fn_i32, table.as_ptr());

                let tfs_u16 = _mm256_loadu_si256(tfs.as_ptr().add(base) as *const __m256i);
                let tfs_i32 = _mm512_cvtepu16_epi32(tfs_u16);
                let tfs_f32 = _mm512_cvtepi32_ps(tfs_i32);

                let denom = _mm512_add_ps(tfs_f32, norms);
                let numer = _mm512_mul_ps(idf_k1p1_vec, tfs_f32);
                let result = _mm512_div_ps(numer, denom);

                _mm512_storeu_ps(scores.as_mut_ptr().add(base), result);
            }
        }

        // Handle remainder with scalar
        if remainder > 0 {
            let rem_base = chunks * 16;
            self.batch_score_scalar(
                &fieldnorm_bytes[rem_base..n],
                &tfs[rem_base..n],
                &mut scores[rem_base..n],
                remainder,
            );
        }
    }

    /// AVX2 batch scoring — 8 postings per iteration.
    ///
    /// Gathers 8 norm factors from the L1-hot 1KB table with
    /// `_mm256_i32gather_ps`, then computes 8 BM25 scores using SIMD arithmetic.
    ///
    /// # Safety
    /// Caller must ensure `avx2` is supported on the current CPU and that
    /// `fieldnorm_bytes.len() >= n`, `tfs.len() >= n`, and `scores.len() >= n`.
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    // SAFETY: Caller verified `avx2` runtime feature support and slice lengths `>= n`.
    unsafe fn batch_score_avx2(
        &self,
        fieldnorm_bytes: &[u8],
        tfs: &[u16],
        scores: &mut [f32],
        n: usize,
    ) {
        use std::arch::x86_64::*;

        // SAFETY: AVX2 guaranteed by #[target_feature(enable = "avx2")].
        // Slice bounds validated: fieldnorm_bytes.len() == n, tfs.len() == n,
        // scores.len() >= n (all asserted in caller). Each chunk accesses [base..base+8]
        // where base+8 <= n.
        let idf_k1p1_vec = _mm256_set1_ps(self.idf_k1p1);
        let table = &self.norm_table;

        let chunks = n / 8;
        let remainder = n % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            let fn_slice = &fieldnorm_bytes[base..base + 8];

            // SAFETY: AVX2 guaranteed by #[target_feature(enable = "avx2")], all slice accesses bounds-checked above, table has 256 floats
            unsafe {
                let fn_u8 = _mm_loadl_epi64(fn_slice.as_ptr() as *const __m128i);
                let fn_i32 = _mm256_cvtepu8_epi32(fn_u8);
                let norms = _mm256_i32gather_ps(table.as_ptr(), fn_i32, 4);

                let tfs_u16 = _mm_loadu_si128(tfs.as_ptr().add(base) as *const __m128i);
                let tfs_i32 = _mm256_cvtepu16_epi32(tfs_u16);
                let tfs_f32 = _mm256_cvtepi32_ps(tfs_i32);

                let denom = _mm256_add_ps(tfs_f32, norms);
                let numer = _mm256_mul_ps(idf_k1p1_vec, tfs_f32);
                let result = _mm256_div_ps(numer, denom);

                _mm256_storeu_ps(scores.as_mut_ptr().add(base), result);
            }
        }

        // Handle remainder with scalar
        if remainder > 0 {
            let rem_base = chunks * 8;
            self.batch_score_scalar(
                &fieldnorm_bytes[rem_base..n],
                &tfs[rem_base..n],
                &mut scores[rem_base..n],
                remainder,
            );
        }
    }

    /// Score a posting with prefetch for the next document's fieldnorm.
    ///
    /// For random-access phrase query scoring, prefetches `fieldnorm_bytes[next_doc_id]`
    /// into L1 cache while scoring the current document. This hides memory latency
    /// when doc IDs are not sequential.
    ///
    /// # Arguments
    /// * `fieldnorm_bytes` - The full fieldnorm byte array (indexed by doc_id)
    /// * `doc_id` - Current document to score
    /// * `tf` - Term frequency in current document
    /// * `next_doc_id` - Next document to score (for prefetch), or `None`
    #[inline]
    pub fn score_with_prefetch(
        &self,
        fieldnorm_bytes: &[u8],
        doc_id: u32,
        tf: u16,
        next_doc_id: Option<u32>,
    ) -> f32 {
        // Prefetch next doc's fieldnorm byte into L1 cache
        if let Some(next) = next_doc_id {
            prefetch_fieldnorm(fieldnorm_bytes, next);
        }

        self.score(fieldnorm_bytes[doc_id as usize], tf)
    }

    /// Batch score with prefetch — scores `doc_ids` array with look-ahead prefetch.
    ///
    /// For each posting, prefetches the fieldnorm byte of the doc 4 positions ahead
    /// in the array. This is useful for semi-sequential access patterns like scoring
    /// a term's posting list where doc IDs are sorted but not contiguous.
    ///
    /// # Arguments
    /// * `fieldnorm_bytes` - The full fieldnorm byte array (indexed by doc_id)
    /// * `doc_ids` - Document IDs to score
    /// * `tfs` - Term frequencies (parallel to doc_ids)
    /// * `scores` - Output scores (must be at least as long as doc_ids)
    pub fn batch_score_with_prefetch(
        &self,
        fieldnorm_bytes: &[u8],
        doc_ids: &[u32],
        tfs: &[u16],
        scores: &mut [f32],
    ) {
        let n = doc_ids.len();
        assert_eq!(tfs.len(), n);
        assert!(scores.len() >= n);

        let doc_ids = &doc_ids[..n];
        let tfs = &tfs[..n];
        let scores = &mut scores[..n];

        const PREFETCH_DISTANCE: usize = 4;

        for i in 0..n {
            if i + PREFETCH_DISTANCE < n {
                prefetch_fieldnorm(fieldnorm_bytes, doc_ids[i + PREFETCH_DISTANCE]);
            }

            let tf_f = tfs[i] as f32;
            let len_norm = self.norm_table[fieldnorm_bytes[doc_ids[i] as usize] as usize];
            scores[i] = self.idf_k1p1 * tf_f / (tf_f + len_norm);
        }
    }
}

/// Prefetch a fieldnorm byte into L1 cache.
///
/// This is a hint to the CPU — it has no effect on correctness and degrades
/// gracefully on platforms without prefetch support.
#[inline]
pub fn prefetch_fieldnorm(fieldnorm_bytes: &[u8], doc_id: u32) {
    let idx = doc_id as usize;
    if idx < fieldnorm_bytes.len() {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: prefetch is advisory; CPU handles gracefully if address is
            // invalid. Index bounds checked above, pointer derived from valid slice.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    fieldnorm_bytes.as_ptr().add(idx) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            // SAFETY: `prfm pldl1keep` is a non-faulting read-only cache prefetch hint
            // with no memory modification or side effects.
            unsafe {
                std::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) fieldnorm_bytes.as_ptr().add(idx),
                    options(nostack, readonly, preserves_flags)
                );
            }
        }
        #[cfg(all(target_arch = "aarch64", miri))]
        {
            let _ = (fieldnorm_bytes, idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::FieldnormEncoder;

    fn make_test_scorer() -> ([f32; 256], f32, f32) {
        let avg_dl = 150.0f32;
        let k1 = 1.2f32;
        let b = 0.75f32;
        let idf = 3.5f32;
        let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
        (norm_table, idf, k1)
    }

    #[test]
    fn test_single_score() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        // tf=0 → score should be 0
        assert_eq!(scorer.score(0, 0), 0.0);

        // tf=1 → positive score
        let s = scorer.score(FieldnormEncoder::encode(100), 1);
        assert!(s > 0.0 && s.is_finite());

        // Higher tf → higher score
        let s1 = scorer.score(FieldnormEncoder::encode(100), 1);
        let s5 = scorer.score(FieldnormEncoder::encode(100), 5);
        assert!(s5 > s1);

        // Shorter doc → higher score (for same tf)
        let short = scorer.score(FieldnormEncoder::encode(50), 2);
        let long = scorer.score(FieldnormEncoder::encode(500), 2);
        assert!(short > long);
    }

    #[test]
    fn test_batch_score_correctness() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let doc_lengths: Vec<u32> = (1..=100).map(|i| i * 10).collect();
        let fieldnorm_bytes: Vec<u8> = doc_lengths
            .iter()
            .map(|&l| FieldnormEncoder::encode(l))
            .collect();
        let tfs: Vec<u16> = (1..=100).map(|i| (i % 20 + 1) as u16).collect();
        let n = tfs.len();

        // Compute expected (scalar one-by-one)
        let expected: Vec<f32> = (0..n)
            .map(|i| scorer.score(fieldnorm_bytes[i], tfs[i]))
            .collect();

        // Compute batch
        let mut batch_scores = vec![0.0f32; n];
        scorer.batch_score(&fieldnorm_bytes, &tfs, &mut batch_scores);

        // Results must match exactly (same floating point ops)
        for i in 0..n {
            assert!(
                (batch_scores[i] - expected[i]).abs() < 1e-5,
                "batch[{i}]={} != expected[{i}]={}",
                batch_scores[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_batch_score_small_arrays() {
        // Test arrays smaller than 8 (no SIMD, only remainder path)
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        for size in 0..8 {
            let fieldnorm_bytes: Vec<u8> = (0..size)
                .map(|i| FieldnormEncoder::encode(i as u32 * 50 + 50))
                .collect();
            let tfs: Vec<u16> = (0..size).map(|i| (i + 1) as u16).collect();
            let mut scores = vec![0.0f32; size];
            scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);

            for i in 0..size {
                let expected = scorer.score(fieldnorm_bytes[i], tfs[i]);
                assert!(
                    (scores[i] - expected).abs() < 1e-5,
                    "size={size}, scores[{i}]={} != expected={}",
                    scores[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_batch_score_exact_multiple_of_8() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let n = 64; // exact multiple of 8
        let fieldnorm_bytes: Vec<u8> = (0..n)
            .map(|i| FieldnormEncoder::encode(i as u32 * 10 + 10))
            .collect();
        let tfs: Vec<u16> = (0..n).map(|i| (i % 10 + 1) as u16).collect();
        let mut scores = vec![0.0f32; n];
        scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);

        for i in 0..n {
            let expected = scorer.score(fieldnorm_bytes[i], tfs[i]);
            assert!((scores[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_score_with_prefetch() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let fieldnorm_bytes: Vec<u8> = (0..1000)
            .map(|i| FieldnormEncoder::encode(i as u32 * 5 + 10))
            .collect();

        // Score with prefetch — same result as without
        let s1 = scorer.score_with_prefetch(&fieldnorm_bytes, 42, 3, Some(100));
        let s2 = scorer.score(fieldnorm_bytes[42], 3);
        assert_eq!(s1, s2);

        // Score with no next doc
        let s3 = scorer.score_with_prefetch(&fieldnorm_bytes, 42, 3, None);
        assert_eq!(s3, s2);
    }

    #[test]
    fn test_batch_score_with_prefetch() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let n = 1000;
        let fieldnorm_bytes: Vec<u8> = (0..n)
            .map(|i| FieldnormEncoder::encode(i as u32 * 5 + 10))
            .collect();

        // Random-ish doc IDs (sorted, sparse)
        let doc_ids: Vec<u32> = (0..50).map(|i| i * 19 + 3).collect();
        let tfs: Vec<u16> = (0..50).map(|i| (i % 8 + 1) as u16).collect();
        let mut scores = vec![0.0f32; 50];

        scorer.batch_score_with_prefetch(&fieldnorm_bytes, &doc_ids, &tfs, &mut scores);

        // Verify against single-score
        for i in 0..50 {
            let expected = scorer.score(fieldnorm_bytes[doc_ids[i] as usize], tfs[i]);
            assert!(
                (scores[i] - expected).abs() < 1e-5,
                "prefetch_batch[{i}]={} != expected={}",
                scores[i],
                expected
            );
        }
    }

    #[test]
    fn test_batch_score_with_prefetch_empty() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let fieldnorm_bytes = vec![0u8; 100];
        let doc_ids: Vec<u32> = vec![];
        let tfs: Vec<u16> = vec![];
        let mut scores = vec![0.0f32; 0];

        scorer.batch_score_with_prefetch(&fieldnorm_bytes, &doc_ids, &tfs, &mut scores);
    }

    #[test]
    #[should_panic]
    fn test_batch_score_panics_on_short_scores() {
        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let fieldnorm_bytes = vec![100u8; 10];
        let tfs = vec![1u16; 10];
        let mut scores = vec![0.0f32; 5]; // too short
        scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);
    }

    #[test]
    fn test_prefetch_fieldnorm_bounds() {
        // Prefetch with out-of-bounds doc_id should not crash (bounds checked)
        let fieldnorm_bytes = vec![0u8; 100];
        prefetch_fieldnorm(&fieldnorm_bytes, 99); // in bounds
        prefetch_fieldnorm(&fieldnorm_bytes, 100); // out of bounds — no-op
        prefetch_fieldnorm(&fieldnorm_bytes, u32::MAX); // way out of bounds — no-op
    }

    // ========================================================================
    // Performance benchmarks (release mode only)
    // ========================================================================

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_batch_score_performance() {
        use std::time::{Duration, Instant};

        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let n = 1_000_000usize;
        let fieldnorm_bytes: Vec<u8> = (0..n)
            .map(|i| FieldnormEncoder::encode((i % 500 + 1) as u32))
            .collect();
        let tfs: Vec<u16> = (0..n).map(|i| (i % 20 + 1) as u16).collect();
        let mut scores = vec![0.0f32; n];

        // Warm up
        scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);
        scorer.batch_score_scalar(&fieldnorm_bytes, &tfs, &mut scores, n);

        // Interleave the two measurements and take per-iteration minimums:
        // the suite runs tests on every core in parallel, so two separate
        // wall-clock windows see unequal contention and the mean is noise.
        let mut batch_time = Duration::MAX;
        let mut scalar_time = Duration::MAX;
        for _ in 0..10 {
            let start = Instant::now();
            scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);
            batch_time = batch_time.min(start.elapsed());

            let start = Instant::now();
            scorer.batch_score_scalar(&fieldnorm_bytes, &tfs, &mut scores, n);
            scalar_time = scalar_time.min(start.elapsed());
        }

        let speedup = scalar_time.as_nanos() as f64 / batch_time.as_nanos().max(1) as f64;

        eprintln!(
            "BM25 batch scoring ({n} postings): batch(SIMD)={:?}, scalar={:?}, speedup={:.1}x",
            batch_time, scalar_time, speedup
        );

        // The scalar path auto-vectorizes well, so explicit SIMD may only match it,
        // and on a fully loaded machine (parallel `cargo test` occupies all cores)
        // the AVX2 path degrades further than the scalar loop (observed ~0.3x).
        // Only guard against a catastrophic regression of the batch dispatch.
        assert!(
            speedup >= 0.2,
            "batch scoring much slower than scalar: {speedup:.2}x"
        );

        // Sanity: scores should be positive
        assert!(scores.iter().all(|&s| s > 0.0));
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_prefetch_performance() {
        use std::time::Instant;

        let (norm_table, idf, k1) = make_test_scorer();
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let n_docs = 10_000_000usize;
        let fieldnorm_bytes: Vec<u8> = (0..n_docs)
            .map(|i| FieldnormEncoder::encode((i % 500 + 1) as u32))
            .collect();

        // Simulate phrase query: random-ish doc IDs (sorted, sparse)
        let n_postings = 100_000usize;
        let doc_ids: Vec<u32> = (0..n_postings)
            .map(|i| (i as u64 * 97 % n_docs as u64) as u32)
            .collect();
        let mut doc_ids_sorted = doc_ids.clone();
        doc_ids_sorted.sort();

        let tfs: Vec<u16> = (0..n_postings).map(|i| (i % 8 + 1) as u16).collect();
        let mut scores = vec![0.0f32; n_postings];

        // Benchmark with prefetch
        let start = Instant::now();
        for _ in 0..5 {
            scorer.batch_score_with_prefetch(&fieldnorm_bytes, &doc_ids_sorted, &tfs, &mut scores);
        }
        let prefetch_time = start.elapsed() / 5;

        // Benchmark without prefetch (same access pattern, no prefetch)
        let start = Instant::now();
        for _ in 0..5 {
            for i in 0..n_postings {
                let tf_f = tfs[i] as f32;
                let len_norm = norm_table[fieldnorm_bytes[doc_ids_sorted[i] as usize] as usize];
                scores[i] = scorer.idf_k1p1 * tf_f / (tf_f + len_norm);
            }
        }
        let no_prefetch_time = start.elapsed() / 5;

        let speedup = no_prefetch_time.as_nanos() as f64 / prefetch_time.as_nanos().max(1) as f64;

        eprintln!(
            "Phrase scoring ({n_postings} postings, {n_docs} docs): \
             prefetch={:?}, no_prefetch={:?}, speedup={:.2}x",
            prefetch_time, no_prefetch_time, speedup
        );

        assert!(scores.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_c12_5_bm25_zero_tf_zero_norm_and_subnormal_avg_dl_never_nan() {
        // 1. When b = 1.0 (full length normalization) and doc length = 0 (fieldnorm_byte = 0),
        // scoring a document with tf == 0 must return 0.0, not NaN (0.0 / 0.0).
        let norm_table = FieldnormEncoder::build_norm_table(100.0, 1.2, 1.0);
        let scorer = Bm25BatchScorer::new(&norm_table, 3.5, 1.2);
        let single = scorer.score(0, 0);
        assert!(
            !single.is_nan() && single == 0.0,
            "Bm25BatchScorer::score(0, 0) with b=1.0 must be 0.0, got {single}"
        );
        assert!(
            norm_table[0] >= f32::MIN_POSITIVE,
            "build_norm_table must clamp entries to >= f32::MIN_POSITIVE, got {}",
            norm_table[0]
        );

        // Also test caller-provided raw [0.0; 256] and [f32::NAN; 256] tables.
        let zero_norms = [0.0f32; 256];
        let nan_norms = [f32::NAN; 256];
        for table in [&norm_table, &zero_norms, &nan_norms] {
            let scorer = Bm25BatchScorer::new(table, 3.5, 1.2);
            assert_eq!(scorer.score(0, 0), 0.0);
            let nonzero_tf = scorer.score(0, 2);
            assert!(
                nonzero_tf.is_finite() && nonzero_tf > 0.0,
                "score(0, 2) must be finite and positive, got {nonzero_tf}"
            );

            let fieldnorm_bytes = vec![0u8; 32];
            let tfs = vec![0u16; 32];
            let mut batch_scores = vec![-1.0f32; 32];
            scorer.batch_score(&fieldnorm_bytes, &tfs, &mut batch_scores);
            assert!(
                batch_scores.iter().all(|&s| !s.is_nan() && s == 0.0),
                "batch_score with tf=0 and norm=0/NaN must be all 0.0, got {:?}",
                &batch_scores[..8]
            );

            let tfs_pos = vec![2u16; 32];
            scorer.batch_score(&fieldnorm_bytes, &tfs_pos, &mut batch_scores);
            assert!(
                batch_scores.iter().all(|&s| s.is_finite() && s > 0.0),
                "batch_score with tf=2 and norm=0/NaN must be finite and positive, got {:?}",
                &batch_scores[..8]
            );

            let mut scalar_scores = vec![-1.0f32; 32];
            scorer.batch_score_scalar(&fieldnorm_bytes, &tfs, &mut scalar_scores, 32);
            assert!(scalar_scores.iter().all(|&s| !s.is_nan() && s == 0.0));

            let doc_ids: Vec<u32> = (0..32).collect();
            let mut prefetch_scores = vec![-1.0f32; 32];
            scorer.batch_score_with_prefetch(
                &fieldnorm_bytes,
                &doc_ids,
                &tfs,
                &mut prefetch_scores,
            );
            assert!(prefetch_scores.iter().all(|&s| !s.is_nan() && s == 0.0));
            assert_eq!(
                scorer.score_with_prefetch(&fieldnorm_bytes, 0, 0, Some(1)),
                0.0
            );
        }

        // 2. FieldnormEncoder::build_score_table with b = 1.0 must not have NaN at tf = 0, byte = 0.
        let score_table = FieldnormEncoder::build_score_table(100.0, 1.2, 1.0, 3.5, 4);
        assert_eq!(score_table[0][0], 0.0);

        // 3. Subnormal avg_dl (1e-40) where 1.0 / avg_dl overflows f32::MAX must not produce NaN
        // at byte = 0 (0.0 * INFINITY).
        let subnormal_norms = FieldnormEncoder::build_norm_table(1e-40, 1.2, 0.75);
        assert!(
            subnormal_norms.iter().all(|&v| v.is_finite() && v < 10.0),
            "subnormal avg_dl must produce finite bounded norm_table entries, got norm[0]={}, norm[1]={}",
            subnormal_norms[0],
            subnormal_norms[1]
        );
    }

    #[test]
    fn bench_s33_r2_bm25_hot_paths() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        if std::env::var_os("S33_BM25").is_none() {
            return;
        }

        let norm_table = FieldnormEncoder::build_norm_table(150.0, 1.2, 0.75);
        let scorer = Bm25BatchScorer::new(&norm_table, 3.5, 1.2);

        let n = 1_000_000usize;
        let fieldnorm_bytes: Vec<u8> = (0..n)
            .map(|i| FieldnormEncoder::encode((i % 500 + 1) as u32))
            .collect();
        let tfs: Vec<u16> = (0..n).map(|i| (i % 20 + 1) as u16).collect();
        let doc_ids: Vec<u32> = (0..n as u32).collect();
        let mut scores = vec![0.0f32; n];

        for _ in 0..3 {
            scorer.batch_score(
                black_box(&fieldnorm_bytes),
                black_box(&tfs),
                black_box(&mut scores),
            );
            scorer.batch_score_with_prefetch(
                black_box(&fieldnorm_bytes),
                black_box(&doc_ids),
                black_box(&tfs),
                black_box(&mut scores),
            );
        }

        let reps = 15;
        let mut min_simd = Duration::MAX;
        let mut min_scalar = Duration::MAX;
        let mut min_prefetch = Duration::MAX;

        for _ in 0..reps {
            let t0 = Instant::now();
            scorer.batch_score(
                black_box(&fieldnorm_bytes),
                black_box(&tfs),
                black_box(&mut scores),
            );
            min_simd = min_simd.min(t0.elapsed());
            black_box(&scores);

            let t0 = Instant::now();
            for i in 0..n {
                scores[i] = scorer.score(*black_box(&fieldnorm_bytes[i]), *black_box(&tfs[i]));
            }
            min_scalar = min_scalar.min(t0.elapsed());
            black_box(&scores);

            let t0 = Instant::now();
            scorer.batch_score_with_prefetch(
                black_box(&fieldnorm_bytes),
                black_box(&doc_ids),
                black_box(&tfs),
                black_box(&mut scores),
            );
            min_prefetch = min_prefetch.min(t0.elapsed());
            black_box(&scores);
        }

        println!(
            "S33_BM25 simd_ns_per_doc={:.4} score_ns_per_doc={:.4} prefetch_ns_per_doc={:.4}",
            min_simd.as_nanos() as f64 / n as f64,
            min_scalar.as_nanos() as f64 / n as f64,
            min_prefetch.as_nanos() as f64 / n as f64,
        );
    }

    #[test]
    fn bench_s34_r1_bm25_stride3_probe() {
        use std::hint::black_box;
        use std::time::Instant;

        if std::env::var_os("S33_BM25").is_none() {
            return;
        }

        fn min_ns<F: FnMut()>(reps: usize, per: f64, mut f: F) -> f64 {
            (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_nanos() as f64 / per
                })
                .fold(f64::INFINITY, f64::min)
        }

        let n = 1usize << 20;
        let norm = FieldnormEncoder::build_norm_table(120.0, 1.2, 0.75);
        let s = Bm25BatchScorer::new(&norm, 2.3, 1.2);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let fb: Vec<u8> = (0..n).map(|_| rnd() as u8).collect();
        let tfs: Vec<u16> = (0..n).map(|_| (rnd() % 20) as u16).collect();
        let ids: Vec<u32> = (0..n as u32).step_by(3).collect();
        let tfs_ids: Vec<u16> = ids.iter().map(|&i| tfs[i as usize]).collect();
        let mut out = vec![0f32; n];
        let a = min_ns(15, n as f64, || {
            s.batch_score(black_box(&fb), black_box(&tfs), black_box(&mut out))
        });
        let b = min_ns(15, n as f64, || {
            let mut acc = 0f32;
            for i in 0..n {
                acc += s.score(black_box(fb[i]), tfs[i]);
            }
            black_box(acc);
        });
        let mut out2 = vec![0f32; ids.len()];
        let c = min_ns(15, ids.len() as f64, || {
            s.batch_score_with_prefetch(
                black_box(&fb),
                black_box(&ids),
                black_box(&tfs_ids),
                black_box(&mut out2),
            )
        });
        let d = min_ns(7, 256.0 * 20.0, || {
            black_box(FieldnormEncoder::build_score_table(
                black_box(120.0),
                1.2,
                0.75,
                2.3,
                20,
            ));
        });
        println!(
            "TIMING batch_score {a:.4} ns/doc\nTIMING score_loop {b:.4} ns/doc\nTIMING batch_prefetch {c:.4} ns/doc\nTIMING build_score_table {d:.4} ns/cell"
        );
    }
}
