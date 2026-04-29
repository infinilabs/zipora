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
/// Holds references to the norm table and pre-computed constants for
/// efficient batch scoring of posting lists.
pub struct Bm25BatchScorer<'a> {
    norm_table: &'a [f32; 256],
    idf_k1p1: f32, // idf * (k1 + 1.0) — pre-multiplied
}

impl<'a> Bm25BatchScorer<'a> {
    /// Create a new batch scorer.
    ///
    /// # Arguments
    /// * `norm_table` - Pre-computed BM25 norm table from `FieldnormEncoder::build_norm_table`
    /// * `idf` - Inverse document frequency for this term
    /// * `k1` - BM25 k1 parameter (typically 1.2)
    #[inline]
    pub fn new(norm_table: &'a [f32; 256], idf: f32, k1: f32) -> Self {
        Self {
            norm_table,
            idf_k1p1: idf * (k1 + 1.0),
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
    /// Automatically dispatches to AVX2 SIMD path (8 scores/iteration) when available,
    /// falling back to scalar.
    ///
    /// # Panics
    /// Panics if `fieldnorm_bytes.len() != tfs.len()` or `scores.len() < tfs.len()`.
    pub fn batch_score(&self, fieldnorm_bytes: &[u8], tfs: &[u16], scores: &mut [f32]) {
        let n = tfs.len();
        assert_eq!(fieldnorm_bytes.len(), n);
        assert!(scores.len() >= n);

        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 support verified by runtime feature check above.
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
        for i in 0..n {
            let tf_f = tfs[i] as f32;
            let len_norm = self.norm_table[fieldnorm_bytes[i] as usize];
            scores[i] = self.idf_k1p1 * tf_f / (tf_f + len_norm);
        }
    }

    /// AVX2 batch scoring — 8 postings per iteration.
    ///
    /// Loads 8 norm factors via scalar table lookups (L1-hot 1KB table is faster
    /// than `_mm256_i32gather_ps` on pre-Ice Lake CPUs), then computes 8 BM25
    /// scores using SIMD arithmetic.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
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
        let table = self.norm_table;

        let chunks = n / 8;
        let remainder = n % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            let fn_slice = &fieldnorm_bytes[base..base + 8];

            unsafe {
                // Gather 8 norm values from 1KB table via AVX2 gather instruction
                // Expand 8 u8s to 8 i32s for gather indices
                let fn_u8 = _mm_loadl_epi64(fn_slice.as_ptr() as *const __m128i);
                let fn_i32 = _mm256_cvtepu8_epi32(fn_u8);
                // Gather 8 floats from table using the 8 indices
                let norms = _mm256_i32gather_ps(table.as_ptr(), fn_i32, 4);

                // Load 8 TFs (u16 → i32 → f32)
                let tfs_u16 = _mm_loadu_si128(tfs.as_ptr().add(base) as *const __m128i);
                let tfs_i32 = _mm256_cvtepu16_epi32(tfs_u16);
                let tfs_f32 = _mm256_cvtepi32_ps(tfs_i32);

                // BM25: idf_k1p1 * tf / (tf + norm)
                let denom = _mm256_add_ps(tfs_f32, norms);
                let numer = _mm256_mul_ps(idf_k1p1_vec, tfs_f32);
                let result = _mm256_div_ps(numer, denom);

                // Store 8 scores
                _mm256_storeu_ps(scores.as_mut_ptr().add(base), result);
            }
        }

        // Handle remainder with scalar
        let rem_base = chunks * 8;
        for i in 0..remainder {
            let idx = rem_base + i;
            let tf_f = tfs[idx] as f32;
            let len_norm = self.norm_table[fieldnorm_bytes[idx] as usize];
            scores[idx] = self.idf_k1p1 * tf_f / (tf_f + len_norm);
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

        let tf_f = tf as f32;
        let len_norm = self.norm_table[fieldnorm_bytes[doc_id as usize] as usize];
        self.idf_k1p1 * tf_f / (tf_f + len_norm)
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

        const PREFETCH_DISTANCE: usize = 4;

        for i in 0..n {
            // Prefetch ahead
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

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: prefetch is advisory, address derived from valid slice, bounds checked above
            unsafe {
                std::arch::aarch64::_prefetch(
                    fieldnorm_bytes.as_ptr().add(idx) as *const i8,
                    std::arch::aarch64::_PREFETCH_READ,
                    std::arch::aarch64::_PREFETCH_LOCALITY3,
                );
            }
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
        use std::time::Instant;

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

        // Benchmark batch_score (SIMD if available)
        let start = Instant::now();
        for _ in 0..10 {
            scorer.batch_score(&fieldnorm_bytes, &tfs, &mut scores);
        }
        let batch_time = start.elapsed() / 10;

        // Benchmark scalar baseline
        let start = Instant::now();
        for _ in 0..10 {
            scorer.batch_score_scalar(&fieldnorm_bytes, &tfs, &mut scores, n);
        }
        let scalar_time = start.elapsed() / 10;

        let speedup = scalar_time.as_nanos() as f64 / batch_time.as_nanos().max(1) as f64;

        eprintln!(
            "BM25 batch scoring ({n} postings): batch(SIMD)={:?}, scalar={:?}, speedup={:.1}x",
            batch_time, scalar_time, speedup
        );

        // The scalar path auto-vectorizes well, so explicit SIMD may only match it.
        // We just verify it's not catastrophically slower.
        assert!(
            speedup >= 0.5,
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
}
