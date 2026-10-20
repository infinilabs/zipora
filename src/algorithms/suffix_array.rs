//! Suffix array construction and LCP array computation
//!
//! This module implements the SA-IS (Suffix Array - Induced Sorting) algorithm
//! for linear-time suffix array construction, along with LCP array computation.

use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::Result;
use std::cmp::Ordering;
use std::time::Instant;

/// Suffix array construction algorithms
///
/// Limitation: only SA-IS is implemented. `DivSufSort`, `DC3` and
/// `LarssonSadakane` are kept for API compatibility and run SA-IS, so every
/// variant builds the same suffix array with the same algorithm; `Adaptive`
/// only adds the pass that picks a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SuffixArrayAlgorithm {
    /// SA-IS (Suffix Array by Induced Sorting) - linear time
    SAIS,
    /// Runs SA-IS; there is no DivSufSort implementation
    DivSufSort,
    /// Runs SA-IS; there is no DC3 implementation
    DC3,
    /// Runs SA-IS; there is no Larsson-Sadakane implementation
    LarssonSadakane,
    /// Picks one of the variants above from data characteristics; all of
    /// them run SA-IS, so the choice only costs the analysis pass
    #[default]
    Adaptive,
}

impl SuffixArrayAlgorithm {
    /// Get a human-readable description of the algorithm
    pub fn description(&self) -> &'static str {
        match self {
            Self::SAIS => "SA-IS: Linear-time induced sorting algorithm",
            Self::DivSufSort => "DivSufSort: runs SA-IS (no DivSufSort implementation)",
            Self::DC3 => "DC3: runs SA-IS (no DC3 implementation)",
            Self::LarssonSadakane => {
                "Larsson-Sadakane: runs SA-IS (no Larsson-Sadakane implementation)"
            }
            Self::Adaptive => "Adaptive: picks a variant by data characteristics; all run SA-IS",
        }
    }
}

/// Data characteristics for adaptive algorithm selection
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Size of the input text
    pub text_length: usize,
    /// Effective alphabet size (number of unique characters)
    pub alphabet_size: usize,
    /// Repetition ratio (0.0 = no repetition, 1.0 = highly repetitive)
    pub repetition_ratio: f64,
    /// Average run length of identical characters
    pub average_run_length: f64,
    /// Entropy measure (lower = more repetitive)
    pub entropy: f64,
}

/// Configuration for suffix array construction
#[derive(Debug, Clone)]
pub struct SuffixArrayConfig {
    /// Algorithm to use for suffix array construction
    pub algorithm: SuffixArrayAlgorithm,
    /// Limitation: has no effect. Construction is always sequential (there is
    /// no parallel SA-IS), and `stats().used_parallel` is always false.
    pub use_parallel: bool,
    /// Limitation: has no effect, see `use_parallel`.
    pub parallel_threshold: usize,
    /// Limitation: has no effect. Construction never computes an LCP array;
    /// use `LcpArray::new` or `EnhancedSuffixArray::with_lcp`.
    pub compute_lcp: bool,
    /// Limitation: has no effect; every alphabet size runs the same SA-IS.
    pub optimize_small_alphabet: bool,
    /// Threshold for adaptive algorithm selection
    pub adaptive_threshold: usize,
}

impl Default for SuffixArrayConfig {
    fn default() -> Self {
        Self {
            algorithm: SuffixArrayAlgorithm::default(),
            use_parallel: true,
            parallel_threshold: 100_000,
            compute_lcp: false,
            optimize_small_alphabet: true,
            adaptive_threshold: 10_000,
        }
    }
}

type Bucket2Table = Option<Box<[(u32, u32)]>>;

/// A suffix array data structure
#[derive(Debug)]
pub struct SuffixArray {
    /// The suffix array itself (indices into the original string)
    sa: Vec<usize>,
    /// Precomputed `[lo, hi)` rank intervals in `sa` for each 2-byte prefix `(b0 << 8) | b1`
    /// when `text_len >= 4096` (`E3.9`).
    bucket2: Bucket2Table,
    /// The original string length
    text_len: usize,
    /// Performance statistics
    stats: AlgorithmStats,
}

impl SuffixArray {
    /// Create a new suffix array from the given text
    pub fn new(text: &[u8]) -> Result<Self> {
        Self::with_config(text, &SuffixArrayConfig::default())
    }

    /// Create a suffix array with custom configuration
    pub fn with_config(text: &[u8], config: &SuffixArrayConfig) -> Result<Self> {
        let builder = SuffixArrayBuilder::new(config.clone());
        builder.build(text)
    }

    /// Get the suffix array
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.sa
    }

    /// Get the length of the original text
    pub fn text_len(&self) -> usize {
        self.text_len
    }

    /// Get the suffix at the given rank
    pub fn suffix_at_rank(&self, rank: usize) -> Option<usize> {
        self.sa.get(rank).copied()
    }

    /// Binary search for a pattern in the suffix array
    pub fn search(&self, text: &[u8], pattern: &[u8]) -> (usize, usize) {
        let (left, right) = self.search_range(text, pattern);
        (left, right.saturating_sub(left))
    }

    /// Find the range of suffixes that start with the given pattern (`E3.9`).
    ///
    /// Uses precomputed 2-gram bucket intervals (when available) + LCP-accelerated
    /// `lower_bound` followed by early return on misses (avoiding a second full
    /// binary search when `pattern` does not occur in `text`) and exponential
    /// galloping from `left + 1` on hits so the upper bound is resolved in
    /// $O(\log(\text{matches}))$ cache-local probes.
    pub fn search_range(&self, text: &[u8], pattern: &[u8]) -> (usize, usize) {
        if pattern.is_empty() {
            return (0, self.sa.len());
        }
        let n = self.sa.len();
        if n == 0 {
            return (0, 0);
        }

        let (lo_bound, hi_bound, init_lcp) = if pattern.len() >= 2
            && let Some(ref b2) = self.bucket2
        {
            let key = ((pattern[0] as usize) << 8) | (pattern[1] as usize);
            let (lo_u32, hi_u32) = b2[key];
            let lo_b = lo_u32 as usize;
            let hi_b = hi_u32 as usize;
            if lo_b == hi_b || pattern.len() == 2 {
                return (lo_b, hi_b);
            }
            (lo_b, hi_b, 2usize)
        } else {
            (0usize, n, 0usize)
        };

        let left = self.lower_bound_in(text, pattern, lo_bound, hi_bound, init_lcp);
        if left >= hi_bound || !text[self.sa[left]..].starts_with(pattern) {
            return (left, left);
        }

        // Pattern occurs at `left`. Gallop forward from `left + 1` to bracket the
        // upper bound in O(log(match_count)) probes instead of O(log n) DRAM misses.
        let mut lo = left + 1;
        if lo >= hi_bound || !text[self.sa[lo]..].starts_with(pattern) {
            return (left, lo);
        }

        let mut step = 2usize;
        let mut hi = (lo + step).min(hi_bound);
        while hi < hi_bound && text[self.sa[hi]..].starts_with(pattern) {
            lo = hi;
            step <<= 1;
            hi = (lo + step).min(hi_bound);
        }

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if text[self.sa[mid]..].starts_with(pattern) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        (left, lo)
    }

    fn lower_bound_in(
        &self,
        text: &[u8],
        pattern: &[u8],
        mut left: usize,
        mut right: usize,
        init_lcp: usize,
    ) -> usize {
        let mut lcp_left = init_lcp;
        let mut lcp_right = init_lcp;
        let pat_len = pattern.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let suffix = &text[self.sa[mid]..];
            let skip = lcp_left.min(lcp_right);
            let min_len = suffix.len().min(pat_len);
            let mut k = skip.min(min_len);

            while k < min_len && suffix[k] == pattern[k] {
                k += 1;
            }

            let ord = if k == pat_len {
                Ordering::Equal
            } else if k == suffix.len() {
                Ordering::Less
            } else {
                suffix[k].cmp(&pattern[k])
            };

            if ord == Ordering::Less {
                left = mid + 1;
                lcp_left = k;
            } else {
                right = mid;
                lcp_right = k;
            }
        }

        left
    }

    /// Get performance statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
    }

    /// Analyze text characteristics for adaptive algorithm selection
    pub fn analyze_text_characteristics(text: &[u8]) -> DataCharacteristics {
        if text.is_empty() {
            return DataCharacteristics {
                text_length: 0,
                alphabet_size: 0,
                repetition_ratio: 0.0,
                average_run_length: 0.0,
                entropy: 0.0,
            };
        }

        let text_length = text.len();

        // Count character frequencies for alphabet size and entropy
        let mut freq = [0u32; 256];
        for &byte in text {
            freq[byte as usize] += 1;
        }

        let alphabet_size = freq.iter().filter(|&&count| count > 0).count();

        // Calculate entropy
        let entropy = Self::calculate_entropy(&freq, text_length);

        // Calculate repetition ratio and average run length
        let (repetition_ratio, average_run_length) = Self::calculate_repetition_metrics(text);

        DataCharacteristics {
            text_length,
            alphabet_size,
            repetition_ratio,
            average_run_length,
            entropy,
        }
    }

    /// Calculate Shannon entropy of the text
    fn calculate_entropy(freq: &[u32; 256], text_length: usize) -> f64 {
        let mut entropy = 0.0;
        let len_f64 = text_length as f64;

        for &count in freq.iter() {
            if count > 0 {
                let p = count as f64 / len_f64;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Calculate repetition metrics (repetition ratio and average run length)
    fn calculate_repetition_metrics(text: &[u8]) -> (f64, f64) {
        if text.len() <= 1 {
            return (0.0, 1.0);
        }

        let mut total_run_length = 0;
        let mut num_runs = 0;
        let mut current_run_length = 1;
        let mut repeated_chars = 0;

        for i in 1..text.len() {
            if text[i] == text[i - 1] {
                current_run_length += 1;
                repeated_chars += 1;
            } else {
                total_run_length += current_run_length;
                num_runs += 1;
                current_run_length = 1;
            }
        }

        // Add the last run
        total_run_length += current_run_length;
        num_runs += 1;

        let repetition_ratio = repeated_chars as f64 / text.len() as f64;
        let average_run_length = if num_runs > 0 {
            total_run_length as f64 / num_runs as f64
        } else {
            1.0
        };

        (repetition_ratio, average_run_length)
    }
}

/// Builder for constructing suffix arrays
pub struct SuffixArrayBuilder {
    config: SuffixArrayConfig,
}

impl SuffixArrayBuilder {
    /// Create a new suffix array builder
    pub fn new(config: SuffixArrayConfig) -> Self {
        Self { config }
    }

    /// Select an algorithm variant based on data characteristics
    ///
    /// Limitation: every variant runs SA-IS, so the choice does not change
    /// how the suffix array is built.
    pub fn select_algorithm(&self, text: &[u8]) -> SuffixArrayAlgorithm {
        if self.config.algorithm != SuffixArrayAlgorithm::Adaptive {
            return self.config.algorithm;
        }

        if text.len() < self.config.adaptive_threshold {
            // For small inputs, use simple algorithms
            return SuffixArrayAlgorithm::DC3;
        }

        let characteristics = SuffixArray::analyze_text_characteristics(text);

        // Decision logic based on data characteristics. The comments give the
        // use each variant was meant for; all of them run SA-IS.
        if characteristics.alphabet_size <= 4 {
            // Very small alphabet - SA-IS handles this well
            SuffixArrayAlgorithm::SAIS
        } else if characteristics.repetition_ratio > 0.7 {
            // Highly repetitive data - Larsson-Sadakane is optimized for this
            SuffixArrayAlgorithm::LarssonSadakane
        } else if characteristics.entropy < 2.0 && characteristics.text_length < 100_000 {
            // Low entropy, small to medium size - DC3 is good for moderate repetition
            SuffixArrayAlgorithm::DC3
        } else if characteristics.text_length > 1_000_000 {
            // Large input with high entropy - DivSufSort is optimized for practical performance
            SuffixArrayAlgorithm::DivSufSort
        } else if characteristics.text_length > 50_000 {
            // Medium-large input - DivSufSort handles this efficiently
            SuffixArrayAlgorithm::DivSufSort
        } else {
            // Small input with moderate entropy - SA-IS is most reliable
            SuffixArrayAlgorithm::SAIS
        }
    }

    /// Build a suffix array from the given text
    pub fn build(&self, text: &[u8]) -> Result<SuffixArray> {
        let start_time = Instant::now();

        if text.is_empty() {
            return Ok(SuffixArray {
                sa: Vec::new(),
                bucket2: None,
                text_len: 0,
                stats: AlgorithmStats {
                    items_processed: 0,
                    processing_time_us: 0,
                    memory_used: 0,
                    used_parallel: false,
                    used_simd: false,
                },
            });
        }

        // There is no parallel construction: `use_parallel` has no effect.
        let (sa, bucket2) = self.build_sequential(text)?;

        let elapsed = start_time.elapsed();
        let memory_used = sa.len() * std::mem::size_of::<usize>();

        Ok(SuffixArray {
            text_len: text.len(),
            stats: AlgorithmStats {
                items_processed: text.len(),
                processing_time_us: elapsed.as_micros() as u64,
                memory_used,
                used_parallel: false,
                used_simd: false,
            },
            bucket2,
            sa,
        })
    }

    fn build_sequential(&self, text: &[u8]) -> Result<(Vec<usize>, Bucket2Table)> {
        if text.is_empty() {
            return Ok((Vec::new(), None));
        }

        if text.len() == 1 {
            return Ok((vec![0], None));
        }

        // Every SuffixArrayAlgorithm variant delegates to SA-IS (`sais_construct`);
        // `Adaptive` dispatches directly to `sais_construct` to avoid the 2-pass
        // `analyze_text_characteristics` scan over `text` (`E3.9`).
        match self.config.algorithm {
            SuffixArrayAlgorithm::SAIS | SuffixArrayAlgorithm::Adaptive => {
                self.sais_construct(text)
            }
            SuffixArrayAlgorithm::DC3 => self.dc3_construct(text),
            SuffixArrayAlgorithm::DivSufSort => self.divsufsort_construct(text),
            SuffixArrayAlgorithm::LarssonSadakane => self.larsson_sadakane_construct(text),
        }
    }

    /// SA-IS (Suffix Array by Induced Sorting) — linear-time O(n).
    ///
    /// Operates directly on the byte slice `text` at the top level using a compact
    /// `WordBitVec` (`1 bit/pos` = `1.25 MiB` at `10 MiB` instead of `40 MiB` `Vec<u32>` plus
    /// `10 MiB` `Vec<bool>`), keeping `text` and `t` inside the 32 MiB L3 cache during
    /// induced-sorting random lookups (`E3.9`).
    fn sais_construct(&self, text: &[u8]) -> Result<(Vec<usize>, Bucket2Table)> {
        let n = text.len();
        if n == 0 {
            return Ok((Vec::new(), None));
        }
        if n == 1 {
            return Ok((vec![0], None));
        }
        if n >= u32::MAX as usize {
            return Err(crate::error::ZiporaError::invalid_data(
                "text too large for SA-IS (must be < 4 GiB)",
            ));
        }

        let (sa, bucket2) = sais_core_bytes(text);

        // The sentinel suffix (position n) is the global minimum, so it always
        // occupies sa[0]; the suffix array of `text` is the remainder.
        debug_assert_eq!(sa[0] as usize, n, "sentinel suffix must sort first");
        Ok((sa[1..].iter().map(|&x| x as usize).collect(), bucket2))
    }

    /// DC3 entry point; construction is delegated to SA-IS for every length.
    ///
    /// The former `n == 2` shortcut returned `[0, 1]` when `text[0] == text[1]`,
    /// but the shorter suffix sorts first, so the answer is `[1, 0]` (S20-R2).
    fn dc3_construct(&self, text: &[u8]) -> Result<(Vec<usize>, Bucket2Table)> {
        if text.is_empty() {
            return Ok((Vec::new(), None));
        }

        if text.len() == 1 {
            return Ok((vec![0], None));
        }

        self.sais_construct(text)
    }

    /// DivSufSort entry point; construction is delegated to SA-IS (there is
    /// no DivSufSort implementation).
    fn divsufsort_construct(&self, text: &[u8]) -> Result<(Vec<usize>, Bucket2Table)> {
        let n = text.len();
        if n == 0 {
            return Ok((Vec::new(), None));
        }
        if n == 1 {
            return Ok((vec![0], None));
        }

        self.sais_construct(text)
    }

    /// Larsson-Sadakane entry point; construction is delegated to SA-IS
    /// (there is no Larsson-Sadakane implementation).
    fn larsson_sadakane_construct(&self, text: &[u8]) -> Result<(Vec<usize>, Bucket2Table)> {
        let n = text.len();
        if n == 0 {
            return Ok((Vec::new(), None));
        }
        if n == 1 {
            return Ok((vec![0], None));
        }

        self.sais_construct(text)
    }
}

/// Marks an unfilled slot in the working suffix array during SA-IS.
const SAIS_EMPTY: u32 = u32::MAX;

/// Compact 1-bit-per-suffix type vector (`true` = S-type, `false` = L-type).
///
/// Uses 8x less memory than `Vec<bool>` (`1.25 MiB` vs `10 MiB` at `N = 10^7`),
/// keeping the type vector cache-resident during SA-IS induced sorting.
struct WordBitVec {
    words: Vec<u64>,
}

impl WordBitVec {
    #[inline]
    fn new_false(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
        }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> bool {
        // SAFETY: callers only query indices `< len`.
        unsafe { ((*self.words.get_unchecked(idx >> 6) >> (idx & 63)) & 1) != 0 }
    }

    #[inline(always)]
    fn set_true(&mut self, idx: usize) {
        // SAFETY: callers only set indices `< len`.
        unsafe {
            *self.words.get_unchecked_mut(idx >> 6) |= 1u64 << (idx & 63);
        }
    }

    #[inline(always)]
    fn is_lms(&self, idx: usize) -> bool {
        idx > 0 && self.get(idx) && !self.get(idx - 1)
    }
}

/// Bucket boundaries for an integer alphabet. With `heads = true` returns the
/// start index of each character's bucket; otherwise the (exclusive) end index.
fn sais_buckets(counts: &[usize], heads: bool) -> Vec<usize> {
    let mut res = vec![0usize; counts.len()];
    let mut sum = 0usize;
    for (i, &c) in counts.iter().enumerate() {
        if heads {
            res[i] = sum;
            sum += c;
        } else {
            sum += c;
            res[i] = sum;
        }
    }
    res
}

/// Top-level induced sorting directly over `text: &[u8]` (with virtual sentinel `0` at index `text.len()`).
fn sais_induce_bytes(text: &[u8], sa: &mut [u32], t: &WordBitVec, counts: &[usize; 257]) {
    let n = sa.len();
    let text_len = text.len();

    // L-type pass at bucket heads.
    let mut heads = [0usize; 257];
    let mut sum = 0usize;
    for c in 0..257 {
        heads[c] = sum;
        sum += counts[c];
    }
    for i in 0..n {
        let p = sa[i];
        if p != SAIS_EMPTY && p > 0 {
            let j = (p - 1) as usize;
            if !t.get(j) {
                let c = if j < text_len {
                    // SAFETY: `j < text_len`.
                    unsafe { *text.get_unchecked(j) as usize + 1 }
                } else {
                    0
                };
                // SAFETY: `c < 257` and `heads[c] < n` from exact bucket prefix sums.
                unsafe {
                    let h = *heads.get_unchecked(c);
                    *sa.get_unchecked_mut(h) = j as u32;
                    *heads.get_unchecked_mut(c) = h + 1;
                }
            }
        }
    }

    // S-type pass at bucket tails.
    let mut tails = [0usize; 257];
    let mut sum = 0usize;
    for c in 0..257 {
        sum += counts[c];
        tails[c] = sum;
    }
    for i in (0..n).rev() {
        let p = sa[i];
        if p != SAIS_EMPTY && p > 0 {
            let j = (p - 1) as usize;
            if t.get(j) {
                let c = if j < text_len {
                    // SAFETY: `j < text_len`.
                    unsafe { *text.get_unchecked(j) as usize + 1 }
                } else {
                    0
                };
                // SAFETY: `c < 257` and `tails[c] - 1 < n` from exact bucket prefix sums.
                unsafe {
                    let tl = *tails.get_unchecked(c) - 1;
                    *tails.get_unchecked_mut(c) = tl;
                    *sa.get_unchecked_mut(tl) = j as u32;
                }
            }
        }
    }
}

#[inline]
fn sais_lms_substr_eq_bytes(text: &[u8], t: &WordBitVec, a: usize, b: usize, n: usize) -> bool {
    if a == b {
        return true;
    }
    let text_len = text.len();
    let mut d = 0usize;
    loop {
        let (ia, ib) = (a + d, b + d);
        if ia >= n || ib >= n {
            return false;
        }
        let ca = if ia < text_len {
            text[ia] as u16 + 1
        } else {
            0
        };
        let cb = if ib < text_len {
            text[ib] as u16 + 1
        } else {
            0
        };
        if ca != cb || t.get(ia) != t.get(ib) {
            return false;
        }
        if d > 0 {
            let (la, lb) = (t.is_lms(ia), t.is_lms(ib));
            if la && lb {
                return true;
            }
            if la || lb {
                return false;
            }
        }
        d += 1;
    }
}

/// Top-level SA-IS operating directly on `text: &[u8]` with a virtual sentinel at index `text.len()`.
fn sais_core_bytes(text: &[u8]) -> (Vec<u32>, Bucket2Table) {
    let text_len = text.len();
    let n = text_len + 1;
    let mut sa = vec![SAIS_EMPTY; n];
    if text_len == 0 {
        sa[0] = 0;
        return (sa, None);
    }

    let build_b2 = text_len >= 4096 && !cfg!(miri);
    let mut gram2 = if build_b2 {
        vec![0u32; 65_536]
    } else {
        Vec::new()
    };

    let mut t = WordBitVec::new_false(n);
    t.set_true(n - 1); // sentinel is S-type; index `n - 2` (`text_len - 1`) is L-type (`false`).
    for i in (0..text_len.saturating_sub(1)).rev() {
        let b0 = text[i];
        let b1 = text[i + 1];
        if b0 < b1 || (b0 == b1 && t.get(i + 1)) {
            t.set_true(i);
        }
        if build_b2 {
            // SAFETY: `((b0 as usize) << 8) | (b1 as usize) < 65_536 == gram2.len()`.
            unsafe {
                *gram2.get_unchecked_mut(((b0 as usize) << 8) | (b1 as usize)) += 1;
            }
        }
    }

    let mut counts = [0usize; 257];
    counts[0] = 1;
    let bucket2 = if build_b2 {
        let last_b = text[text_len - 1] as usize;
        let mut ranges = vec![(0u32, 0u32); 65_536].into_boxed_slice();
        let mut sum = 0u32;
        for b0 in 0..256usize {
            let mut c_b0 = 0usize;
            if b0 == last_b {
                sum += 1;
                c_b0 += 1;
            }
            let base = b0 << 8;
            for b1 in 0..256usize {
                let idx = base | b1;
                let c = gram2[idx];
                ranges[idx] = (sum, sum + c);
                sum += c;
                c_b0 += c as usize;
            }
            counts[b0 + 1] = c_b0;
        }
        debug_assert_eq!(sum as usize, text_len);
        Some(ranges)
    } else {
        for &b in text {
            counts[b as usize + 1] += 1;
        }
        None
    };

    let mut tails = [0usize; 257];
    let mut sum = 0usize;
    for c in 0..257 {
        sum += counts[c];
        tails[c] = sum;
    }
    // Seed LMS suffixes (indices `1..text_len` plus sentinel `text_len`).
    sa[0] = text_len as u32;
    for (i, &b) in text.iter().enumerate().skip(1) {
        if t.is_lms(i) {
            let c = b as usize + 1;
            tails[c] -= 1;
            sa[tails[c]] = i as u32;
        }
    }
    sais_induce_bytes(text, &mut sa, &t, &counts);

    let mut n1 = 0usize;
    for i in 0..n {
        let p = sa[i];
        if p != SAIS_EMPTY && t.is_lms(p as usize) {
            sa[n1] = p;
            n1 += 1;
        }
    }
    sa[n1..n].fill(SAIS_EMPTY);

    let mut name = 0u32;
    let mut prev: Option<usize> = None;
    for i in 0..n1 {
        let pos = sa[i] as usize;
        let diff = match prev {
            None => true,
            Some(prev_pos) => !sais_lms_substr_eq_bytes(text, &t, prev_pos, pos, n),
        };
        if diff {
            name += 1;
            prev = Some(pos);
        }
        sa[n1 + pos / 2] = name - 1;
    }

    let mut j = n;
    for i in (n1..n).rev() {
        if sa[i] != SAIS_EMPTY {
            j -= 1;
            sa[j] = sa[i];
        }
    }
    let s1: Vec<u32> = sa[n - n1..n].to_vec();

    let sa1 = if (name as usize) < n1 {
        sais_core(&s1, name as usize)
    } else {
        let mut tmp = vec![SAIS_EMPTY; n1];
        for (i, &name_i) in s1.iter().enumerate() {
            tmp[name_i as usize] = i as u32;
        }
        tmp
    };

    let mut lms_positions = Vec::with_capacity(n1);
    for i in 1..n {
        if t.is_lms(i) {
            lms_positions.push(i as u32);
        }
    }
    sa.fill(SAIS_EMPTY);
    let mut sum = 0usize;
    for c in 0..257 {
        sum += counts[c];
        tails[c] = sum;
    }
    for i in (0..n1).rev() {
        let pos = lms_positions[sa1[i] as usize] as usize;
        let c = if pos < text_len {
            text[pos] as usize + 1
        } else {
            0
        };
        tails[c] -= 1;
        sa[tails[c]] = pos as u32;
    }
    sais_induce_bytes(text, &mut sa, &t, &counts);

    (sa, bucket2)
}

/// Induced sorting: with the LMS suffixes already seeded at their bucket tails
/// in `sa`, induce the order of all L-type suffixes (left→right) then all
/// S-type suffixes (right→left). Linear time.
fn sais_induce(s: &[u32], sa: &mut [u32], t: &WordBitVec, counts: &[usize]) {
    let n = s.len();

    // L-type pass at bucket heads.
    let mut heads = sais_buckets(counts, true);
    for i in 0..n {
        let p = sa[i];
        if p != SAIS_EMPTY && p > 0 {
            let j = (p - 1) as usize;
            if !t.get(j) {
                // SAFETY: `j < n` and `s[j] < counts.len()`.
                unsafe {
                    let c = *s.get_unchecked(j) as usize;
                    let h = *heads.get_unchecked(c);
                    *sa.get_unchecked_mut(h) = j as u32;
                    *heads.get_unchecked_mut(c) = h + 1;
                }
            }
        }
    }

    // S-type pass at bucket tails.
    let mut tails = sais_buckets(counts, false);
    for i in (0..n).rev() {
        let p = sa[i];
        if p != SAIS_EMPTY && p > 0 {
            let j = (p - 1) as usize;
            if t.get(j) {
                // SAFETY: `j < n` and `s[j] < counts.len()`.
                unsafe {
                    let c = *s.get_unchecked(j) as usize;
                    let tl = *tails.get_unchecked(c) - 1;
                    *tails.get_unchecked_mut(c) = tl;
                    *sa.get_unchecked_mut(tl) = j as u32;
                }
            }
        }
    }
}

/// Whether the two LMS substrings starting at `a` and `b` are identical. Walks
/// until both substrings end (next LMS position) or a difference is found. The
/// summed length of all LMS substrings is O(n), so total naming cost is O(n).
fn sais_lms_substr_eq(s: &[u32], t: &WordBitVec, a: usize, b: usize, n: usize) -> bool {
    if a == b {
        return true;
    }
    let mut d = 0usize;
    loop {
        let (ia, ib) = (a + d, b + d);
        if ia >= n || ib >= n {
            return false;
        }
        if s[ia] != s[ib] || t.get(ia) != t.get(ib) {
            return false;
        }
        if d > 0 {
            let (la, lb) = (t.is_lms(ia), t.is_lms(ib));
            if la && lb {
                return true; // both substrings end here, equal so far
            }
            if la || lb {
                return false; // one ended before the other
            }
        }
        d += 1;
    }
}

/// Core SA-IS over an integer string `s` whose last element is a unique global
/// minimum, with alphabet size `k`. Returns the suffix array of `s`. The
/// Nong–Zhang–Chan induced-sorting construction, linear time and space.
fn sais_core(s: &[u32], k: usize) -> Vec<u32> {
    let n = s.len();
    let mut sa = vec![SAIS_EMPTY; n];
    if n == 0 {
        return sa;
    }
    if n == 1 {
        sa[0] = 0;
        return sa;
    }

    // Suffix types: true = S-type, false = L-type. The last suffix is S-type.
    let mut t = WordBitVec::new_false(n);
    t.set_true(n - 1);
    for i in (0..n - 1).rev() {
        if s[i] < s[i + 1] || (s[i] == s[i + 1] && t.get(i + 1)) {
            t.set_true(i);
        }
    }

    // Bucket sizes.
    let mut counts = vec![0usize; k];
    for &c in s {
        counts[c as usize] += 1;
    }

    // Stage 1: seed LMS suffixes at bucket tails, then induce to sort the LMS
    // substrings into `sa`.
    let mut tails = sais_buckets(&counts, false);
    for (i, &ch) in s.iter().enumerate() {
        if t.is_lms(i) {
            let c = ch as usize;
            tails[c] -= 1;
            sa[tails[c]] = i as u32;
        }
    }
    sais_induce(s, &mut sa, &t, &counts);

    // Stage 2: compact the sorted LMS suffixes to the front and name their
    // substrings. Names are written at `pos / 2` (LMS positions are never
    // adjacent, so these slots are distinct).
    let mut n1 = 0usize;
    for i in 0..n {
        let p = sa[i];
        if p != SAIS_EMPTY && t.is_lms(p as usize) {
            sa[n1] = p;
            n1 += 1;
        }
    }
    sa[n1..n].fill(SAIS_EMPTY);

    let mut name = 0u32;
    let mut prev: Option<usize> = None;
    for i in 0..n1 {
        let pos = sa[i] as usize;
        let diff = match prev {
            None => true,
            Some(prev_pos) => !sais_lms_substr_eq(s, &t, prev_pos, pos, n),
        };
        if diff {
            name += 1;
            prev = Some(pos);
        }
        sa[n1 + pos / 2] = name - 1;
    }

    // Compact names into the tail to form the reduced string `s1`.
    let mut j = n;
    for i in (n1..n).rev() {
        if sa[i] != SAIS_EMPTY {
            j -= 1;
            sa[j] = sa[i];
        }
    }
    let s1: Vec<u32> = sa[n - n1..n].to_vec();

    // Stage 3: if any name repeats, recurse; otherwise ranks are already the SA.
    let sa1 = if (name as usize) < n1 {
        sais_core(&s1, name as usize)
    } else {
        let mut tmp = vec![SAIS_EMPTY; n1];
        for (i, &name_i) in s1.iter().enumerate() {
            tmp[name_i as usize] = i as u32;
        }
        tmp
    };

    // Stage 4: map sorted reduced suffixes back to original LMS positions, then
    // re-induce the full suffix array.
    let mut lms_positions = Vec::with_capacity(n1);
    for i in 0..n {
        if t.is_lms(i) {
            lms_positions.push(i as u32);
        }
    }
    sa.fill(SAIS_EMPTY);
    let mut tails = sais_buckets(&counts, false);
    for i in (0..n1).rev() {
        let pos = lms_positions[sa1[i] as usize] as usize;
        let c = s[pos] as usize;
        tails[c] -= 1;
        sa[tails[c]] = pos as u32;
    }
    sais_induce(s, &mut sa, &t, &counts);

    sa
}

impl Algorithm for SuffixArrayBuilder {
    type Config = SuffixArrayConfig;
    type Input = Vec<u8>;
    type Output = SuffixArray;

    fn execute(&self, config: &Self::Config, input: Self::Input) -> Result<Self::Output> {
        let builder = Self::new(config.clone());
        builder.build(&input)
    }

    fn stats(&self) -> AlgorithmStats {
        // Return default stats - actual stats come from the built suffix array
        AlgorithmStats {
            items_processed: 0,
            processing_time_us: 0,
            memory_used: 0,
            used_parallel: false,
            used_simd: false,
        }
    }

    fn estimate_memory(&self, input_size: usize) -> usize {
        // Suffix array requires one usize per character
        input_size * std::mem::size_of::<usize>()
    }

    fn supports_parallel(&self) -> bool {
        // Construction is sequential; `use_parallel` has no effect.
        false
    }
}

/// LCP (Longest Common Prefix) array
pub struct LcpArray {
    /// The LCP array values
    lcp: Vec<usize>,
    /// Performance statistics
    stats: AlgorithmStats,
}

impl LcpArray {
    /// Compute LCP array from suffix array and original text
    pub fn new(text: &[u8], suffix_array: &SuffixArray) -> Result<Self> {
        let start_time = Instant::now();

        let lcp = Self::compute_lcp_kasai(text, suffix_array.as_slice())?;

        let elapsed = start_time.elapsed();
        let memory_used = lcp.len() * std::mem::size_of::<usize>();

        Ok(Self {
            stats: AlgorithmStats {
                items_processed: text.len(),
                processing_time_us: elapsed.as_micros() as u64,
                memory_used,
                used_parallel: false,
                used_simd: false,
            },
            lcp,
        })
    }

    /// Get the LCP array
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.lcp
    }

    /// Get the LCP value at the given index
    pub fn lcp_at(&self, index: usize) -> Option<usize> {
        self.lcp.get(index).copied()
    }

    /// Get performance statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
    }

    // Kasai's algorithm for computing LCP array in linear time
    fn compute_lcp_kasai(text: &[u8], sa: &[usize]) -> Result<Vec<usize>> {
        if text.len() != sa.len() {
            return Err(crate::error::ZiporaError::invalid_data(
                "Text length must match suffix array length",
            ));
        }
        let n = text.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Compute inverse suffix array with bounds checking
        let mut rank = vec![0; n];
        for i in 0..n {
            if sa[i] < n {
                rank[sa[i]] = i;
            }
        }

        let mut lcp = vec![0; n];
        let mut h = 0;

        for i in 0..n {
            if rank[i] > 0 {
                let j = sa[rank[i] - 1];

                while i + h < n && j + h < n && text[i + h] == text[j + h] {
                    h += 1;
                }

                lcp[rank[i]] = h;

                h = h.saturating_sub(1);
            }
        }

        Ok(lcp)
    }
}

/// Enhanced suffix array with additional functionality
pub struct EnhancedSuffixArray {
    /// Base suffix array
    sa: SuffixArray,
    /// LCP array
    lcp: Option<LcpArray>,
    /// BWT (Burrows-Wheeler Transform) - optional
    bwt: Option<Vec<u8>>,
}

impl EnhancedSuffixArray {
    /// Create an enhanced suffix array with LCP
    pub fn with_lcp(text: &[u8]) -> Result<Self> {
        let config = SuffixArrayConfig {
            compute_lcp: true,
            ..Default::default()
        };

        let sa = SuffixArray::with_config(text, &config)?;
        let lcp = Some(LcpArray::new(text, &sa)?);

        Ok(Self { sa, lcp, bwt: None })
    }

    /// Create an enhanced suffix array with BWT
    pub fn with_bwt(text: &[u8]) -> Result<Self> {
        let sa = SuffixArray::new(text)?;
        let bwt = Some(Self::compute_bwt(text, sa.as_slice()));

        Ok(Self { sa, lcp: None, bwt })
    }

    /// Get the suffix array
    pub fn suffix_array(&self) -> &SuffixArray {
        &self.sa
    }

    /// Get the LCP array if available
    pub fn lcp_array(&self) -> Option<&LcpArray> {
        self.lcp.as_ref()
    }

    /// Get the BWT if available
    pub fn bwt(&self) -> Option<&[u8]> {
        self.bwt.as_deref()
    }

    fn compute_bwt(text: &[u8], sa: &[usize]) -> Vec<u8> {
        let mut bwt = Vec::with_capacity(text.len());

        for &suffix_start in sa {
            if suffix_start == 0 {
                bwt.push(text[text.len() - 1]);
            } else {
                bwt.push(text[suffix_start - 1]);
            }
        }

        bwt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_array_empty() {
        let sa = SuffixArray::new(b"").unwrap();
        assert_eq!(sa.as_slice().len(), 0);
        assert_eq!(sa.text_len(), 0);
    }

    #[test]
    fn test_suffix_array_simple() {
        let text = b"banana";
        let sa = SuffixArray::new(text).unwrap();

        assert_eq!(sa.as_slice().len(), 6);
        assert_eq!(sa.text_len(), 6);

        // Check that it's properly sorted
        let suffixes = sa.as_slice();
        for i in 1..suffixes.len() {
            let suffix1 = &text[suffixes[i - 1]..];
            let suffix2 = &text[suffixes[i]..];
            assert!(suffix1 <= suffix2);
        }
    }

    #[test]
    fn test_suffix_array_search() {
        let text = b"banana";
        let sa = SuffixArray::new(text).unwrap();

        let (start, count) = sa.search(text, b"an");
        assert!(count > 0);

        // Verify all found suffixes start with "an"
        for i in start..start + count {
            let suffix_idx = sa.suffix_at_rank(i).unwrap();
            let suffix = &text[suffix_idx..];
            assert!(suffix.starts_with(b"an"));
        }
    }

    #[test]
    fn test_suffix_array_search_not_found() {
        let text = b"banana";
        let sa = SuffixArray::new(text).unwrap();

        let (_, count) = sa.search(text, b"xyz");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_lcp_array() {
        let text = b"banana";
        let sa = SuffixArray::new(text).unwrap();
        let lcp = LcpArray::new(text, &sa).unwrap();

        assert_eq!(lcp.as_slice().len(), 6);

        // First element should be 0
        assert_eq!(lcp.lcp_at(0), Some(0));
    }

    #[test]
    fn test_enhanced_suffix_array() {
        let text = b"banana";
        let esa = EnhancedSuffixArray::with_lcp(text).unwrap();

        assert_eq!(esa.suffix_array().text_len(), 6);
        assert!(esa.lcp_array().is_some());
        assert!(esa.bwt().is_none());
    }

    #[test]
    fn test_enhanced_suffix_array_with_bwt() {
        let text = b"banana";
        let esa = EnhancedSuffixArray::with_bwt(text).unwrap();

        assert_eq!(esa.suffix_array().text_len(), 6);
        assert!(esa.lcp_array().is_none());
        assert!(esa.bwt().is_some());
        assert_eq!(esa.bwt().unwrap().len(), 6);
    }

    #[test]
    fn test_suffix_array_config() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::SAIS,
            use_parallel: false,
            parallel_threshold: 1000,
            compute_lcp: true,
            optimize_small_alphabet: false,
            adaptive_threshold: 10_000,
        };

        let text = b"test";
        let sa = SuffixArray::with_config(text, &config).unwrap();
        assert_eq!(sa.text_len(), 4);
        assert!(!sa.stats().used_parallel);
    }

    #[test]
    fn test_algorithm_trait() {
        let builder = SuffixArrayBuilder::new(SuffixArrayConfig::default());

        // Construction is sequential (S20-R6).
        assert!(!builder.supports_parallel());
        assert!(!builder.supports_simd());

        let memory_estimate = builder.estimate_memory(1000);
        assert_eq!(memory_estimate, 1000 * std::mem::size_of::<usize>());
    }

    #[test]
    fn test_suffix_array_algorithm_enum() {
        assert_eq!(
            SuffixArrayAlgorithm::default(),
            SuffixArrayAlgorithm::Adaptive
        );

        // Test descriptions
        assert!(!SuffixArrayAlgorithm::SAIS.description().is_empty());
        assert!(!SuffixArrayAlgorithm::DC3.description().is_empty());
        assert!(!SuffixArrayAlgorithm::DivSufSort.description().is_empty());
        assert!(
            !SuffixArrayAlgorithm::LarssonSadakane
                .description()
                .is_empty()
        );
        assert!(!SuffixArrayAlgorithm::Adaptive.description().is_empty());
    }

    #[test]
    fn test_data_characteristics_analysis() {
        // Test empty string
        let chars = SuffixArray::analyze_text_characteristics(b"");
        assert_eq!(chars.text_length, 0);
        assert_eq!(chars.alphabet_size, 0);

        // Test simple string
        let chars = SuffixArray::analyze_text_characteristics(b"abcd");
        assert_eq!(chars.text_length, 4);
        assert_eq!(chars.alphabet_size, 4);
        assert!(chars.entropy > 0.0);

        // Test repetitive string
        let chars = SuffixArray::analyze_text_characteristics(b"aaaa");
        assert_eq!(chars.text_length, 4);
        assert_eq!(chars.alphabet_size, 1);
        assert!(chars.repetition_ratio > 0.5);
        assert_eq!(chars.entropy, 0.0); // Single character has zero entropy

        // Test mixed string
        let chars = SuffixArray::analyze_text_characteristics(b"banana");
        assert_eq!(chars.text_length, 6);
        assert_eq!(chars.alphabet_size, 3); // 'a', 'b', 'n'
        assert!(chars.entropy > 0.0);
        assert!(chars.repetition_ratio < 1.0);
    }

    #[test]
    fn test_adaptive_algorithm_selection() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::Adaptive,
            adaptive_threshold: 100,
            ..Default::default()
        };
        let builder = SuffixArrayBuilder::new(config);

        // Small input should select DC3
        let algorithm = builder.select_algorithm(b"small");
        assert_eq!(algorithm, SuffixArrayAlgorithm::DC3);

        // Small alphabet should select SA-IS
        let algorithm = builder.select_algorithm(&vec![b'a'; 1000]);
        assert_eq!(algorithm, SuffixArrayAlgorithm::SAIS);

        // Non-adaptive config should return the specified algorithm
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::SAIS,
            ..Default::default()
        };
        let builder = SuffixArrayBuilder::new(config);
        let algorithm = builder.select_algorithm(b"any text");
        assert_eq!(algorithm, SuffixArrayAlgorithm::SAIS);
    }

    #[test]
    fn test_dc3_algorithm() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::DC3,
            ..Default::default()
        };

        // Test simple string with DC3 (currently falls back to SA-IS)
        let text = b"banana";
        let sa = SuffixArray::with_config(text, &config).unwrap();

        assert_eq!(sa.as_slice().len(), 6);
        assert_eq!(sa.text_len(), 6);

        // Verify the result is properly sorted (should work since it falls back to SA-IS)
        let suffixes = sa.as_slice();
        for i in 1..suffixes.len() {
            let suffix1 = &text[suffixes[i - 1]..];
            let suffix2 = &text[suffixes[i]..];
            assert!(
                suffix1 <= suffix2,
                "Suffix at {} ({:?}) should be <= suffix at {} ({:?})",
                i - 1,
                suffix1,
                i,
                suffix2
            );
        }
    }

    #[test]
    fn test_algorithm_consistency() {
        // Test that our new algorithms produce valid suffix arrays
        let test_cases = [
            b"banana".as_slice(),
            b"abcdef".as_slice(),
            b"mississippi".as_slice(),
            b"aaaaaa".as_slice(),
            b"abab".as_slice(),
        ];

        for &text in &test_cases {
            // Build with DivSufSort (our working implementation)
            let config_div = SuffixArrayConfig {
                algorithm: SuffixArrayAlgorithm::DivSufSort,
                ..Default::default()
            };
            let sa_div = SuffixArray::with_config(text, &config_div).unwrap();

            // Build with Larsson-Sadakane (our working implementation)
            let config_ls = SuffixArrayConfig {
                algorithm: SuffixArrayAlgorithm::LarssonSadakane,
                ..Default::default()
            };
            let sa_ls = SuffixArray::with_config(text, &config_ls).unwrap();

            // Verify both results are properly sorted suffix arrays
            verify_suffix_array_is_sorted(text, sa_div.as_slice());
            verify_suffix_array_is_sorted(text, sa_ls.as_slice());
        }
    }

    fn verify_suffix_array_is_sorted(text: &[u8], sa: &[usize]) {
        for i in 1..sa.len() {
            let suffix1 = &text[sa[i - 1]..];
            let suffix2 = &text[sa[i]..];
            assert!(
                suffix1 <= suffix2,
                "Suffix array not properly sorted at position {}",
                i
            );
        }
    }

    /// Reference suffix array via direct suffix comparison sort (O(n^2 log n),
    /// trivially correct). Used as the oracle for SA-IS correctness.
    fn reference_suffix_array(text: &[u8]) -> Vec<usize> {
        let mut sa: Vec<usize> = (0..text.len()).collect();
        sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
        sa
    }

    /// SA-IS must produce the exact suffix array for arbitrary input, including
    /// varied data with >256 distinct LMS substrings, repetitive data, and text.
    #[test]
    fn test_sais_correctness_vs_reference() {
        let cfg = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::SAIS,
            ..Default::default()
        };

        let mut cases: Vec<Vec<u8>> = vec![
            b"a".to_vec(),
            b"ab".to_vec(),
            b"ba".to_vec(),
            b"aa".to_vec(),
            b"banana".to_vec(),
            b"mississippi".to_vec(),
            b"aaaaaa".to_vec(),
            b"abababab".to_vec(),
            b"the quick brown fox jumps over the lazy dog".to_vec(),
        ];

        // Pseudo-random varied data (LCG) across several sizes — exercises the
        // recursion and >256 distinct LMS substrings. Miri stops at 1,000
        // bytes (343 LMS substrings, all distinct) and uses shorter
        // repetitive and sweep inputs.
        let (lcg_sizes, ab_reps, sweep_len): (&[usize], usize, u32) = if cfg!(miri) {
            (&[50, 200, 1000], 200, 512)
        } else {
            (&[50, 200, 1000, 5000], 2000, 4000)
        };
        let mut x: u32 = 0x1234_5678;
        for &n in lcg_sizes {
            let v: Vec<u8> = (0..n)
                .map(|_| {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                    (x >> 24) as u8
                })
                .collect();
            cases.push(v);
        }
        // Highly repetitive, small alphabet.
        cases.push((0..ab_reps).flat_map(|_| *b"ab").collect());
        // Full-alphabet sweep (256 distinct bytes repeated).
        cases.push((0..sweep_len).map(|i| (i % 256) as u8).collect());

        for text in &cases {
            let sa = SuffixArray::with_config(text, &cfg).unwrap();
            assert_eq!(
                sa.as_slice().len(),
                text.len(),
                "SA length mismatch for input len {}",
                text.len()
            );
            assert_eq!(
                sa.as_slice(),
                reference_suffix_array(text).as_slice(),
                "SA-IS produced an incorrect suffix array for input len {}",
                text.len()
            );
        }
    }

    /// Randomized SA-IS stress test over small alphabets and many lengths.
    /// Small alphabets produce many equal LMS substrings, forcing deep recursion
    /// — the most fragile part of SA-IS — so this catches edge cases the fixed
    /// inputs above may miss. Deterministic (seeded LCG), no wall-clock/RNG.
    #[test]
    fn test_sais_randomized_small_alphabet() {
        let cfg = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::SAIS,
            ..Default::default()
        };
        let mut x: u32 = 0x9E37_79B9;
        let mut next = |modulo: u32| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 16) % modulo
        };

        // Many trials across alphabet sizes {2,3,4} and lengths up to 300
        // (Miri: 10 trials per alphabet instead of 200).
        let trials = if cfg!(miri) { 10 } else { 200 };
        for alphabet in [2u32, 3, 4] {
            for _ in 0..trials {
                let len = (next(300) + 1) as usize;
                let data: Vec<u8> = (0..len).map(|_| next(alphabet) as u8).collect();
                let sa = SuffixArray::with_config(&data, &cfg).unwrap();
                assert_eq!(
                    sa.as_slice(),
                    reference_suffix_array(&data).as_slice(),
                    "SA-IS mismatch: alphabet={alphabet}, data={data:?}"
                );
            }
        }
    }

    #[test]
    fn test_dc3_edge_cases() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::DC3,
            ..Default::default()
        };

        // Empty string (DC3 falls back to SA-IS)
        let sa = SuffixArray::with_config(b"", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 0);

        // Single character (DC3 falls back to SA-IS)
        let sa = SuffixArray::with_config(b"a", &config).unwrap();
        assert_eq!(sa.as_slice(), &[0]);

        // Two characters (DC3 falls back to SA-IS)
        // Note: There may be edge cases in the SA-IS implementation for very short strings
        let text = b"ab";
        let sa = SuffixArray::with_config(text, &config).unwrap();
        assert_eq!(sa.as_slice().len(), 2);

        let text = b"ba";
        let sa = SuffixArray::with_config(text, &config).unwrap();
        assert_eq!(sa.as_slice().len(), 2);
    }

    #[test]
    fn test_adaptive_with_different_data_types() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::Adaptive,
            adaptive_threshold: 10,
            ..Default::default()
        };

        // Test various data patterns to ensure adaptive selection works
        let test_cases = [
            (b"xyz".as_slice(), "small input"), // Changed to avoid length issue
            (b"aaaaaaaaaaaaaaaaaaaa".as_slice(), "repetitive input"),
            (b"abcdefghijklmnopqrstuvwxyz".as_slice(), "diverse alphabet"),
            (&vec![b'a'; 2000], "large repetitive"),
        ];

        for (text, description) in &test_cases {
            let sa = SuffixArray::with_config(text, &config).unwrap();

            // Just verify basic properties - adaptive selection should produce valid suffix arrays
            assert_eq!(
                sa.as_slice().len(),
                text.len(),
                "Suffix array length mismatch for {}",
                description
            );
            assert_eq!(
                sa.text_len(),
                text.len(),
                "Text length mismatch for {}",
                description
            );

            // Verify all indices are valid
            for &idx in sa.as_slice() {
                assert!(
                    idx < text.len(),
                    "Invalid suffix index {} for text length {} in {}",
                    idx,
                    text.len(),
                    description
                );
            }
        }
    }

    #[test]
    fn test_divsufsort_algorithm() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::DivSufSort,
            ..Default::default()
        };

        let test_cases = [
            b"banana".as_slice(),
            b"abcdef".as_slice(),
            b"mississippi".as_slice(),
            b"abab".as_slice(),
            b"aaaa".as_slice(),
            b"abcdefghijklmnopqrstuvwxyz".as_slice(),
        ];

        for &text in &test_cases {
            let sa = SuffixArray::with_config(text, &config).unwrap();

            // Verify basic properties
            assert_eq!(
                sa.as_slice().len(),
                text.len(),
                "DivSufSort length mismatch for text: {:?}",
                std::str::from_utf8(text)
            );
            assert_eq!(sa.text_len(), text.len());

            // Verify all indices are valid and unique
            let mut indices = sa.as_slice().to_vec();
            indices.sort_unstable();
            for (i, &idx) in indices.iter().enumerate() {
                assert_eq!(idx, i, "Missing or duplicate index in DivSufSort result");
            }

            // Verify suffix array is properly sorted
            let suffixes = sa.as_slice();
            for i in 1..suffixes.len() {
                let suffix1 = &text[suffixes[i - 1]..];
                let suffix2 = &text[suffixes[i]..];
                assert!(
                    suffix1 <= suffix2,
                    "DivSufSort: Suffix at {} ({:?}) should be <= suffix at {} ({:?})",
                    i - 1,
                    suffix1,
                    i,
                    suffix2
                );
            }
        }
    }

    #[test]
    fn test_larsson_sadakane_algorithm() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::LarssonSadakane,
            ..Default::default()
        };

        let test_cases = [
            b"banana".as_slice(),
            b"abcdef".as_slice(),
            b"mississippi".as_slice(),
            b"abab".as_slice(),
            b"aaaa".as_slice(),
            b"ababababab".as_slice(),   // Repetitive pattern
            b"aaaabbbbcccc".as_slice(), // Another repetitive pattern
        ];

        for &text in &test_cases {
            let sa = SuffixArray::with_config(text, &config).unwrap();

            // Verify basic properties
            assert_eq!(
                sa.as_slice().len(),
                text.len(),
                "Larsson-Sadakane length mismatch for text: {:?}",
                std::str::from_utf8(text)
            );
            assert_eq!(sa.text_len(), text.len());

            // Verify all indices are valid and unique
            let mut indices = sa.as_slice().to_vec();
            indices.sort_unstable();
            for (i, &idx) in indices.iter().enumerate() {
                assert_eq!(
                    idx, i,
                    "Missing or duplicate index in Larsson-Sadakane result"
                );
            }

            // Verify suffix array is properly sorted
            let suffixes = sa.as_slice();
            for i in 1..suffixes.len() {
                let suffix1 = &text[suffixes[i - 1]..];
                let suffix2 = &text[suffixes[i]..];
                assert!(
                    suffix1 <= suffix2,
                    "Larsson-Sadakane: Suffix at {} ({:?}) should be <= suffix at {} ({:?})",
                    i - 1,
                    suffix1,
                    i,
                    suffix2
                );
            }
        }
    }

    #[test]
    fn test_algorithm_consistency_all() {
        // Test that our new algorithms produce valid suffix arrays
        let test_cases = [
            b"banana".as_slice(),
            b"abcdef".as_slice(),
            b"mississippi".as_slice(),
            b"aaaa".as_slice(),
            b"abab".as_slice(),
        ];

        for &text in &test_cases {
            // Build with our working algorithms
            let config_dc3 = SuffixArrayConfig {
                algorithm: SuffixArrayAlgorithm::DC3,
                ..Default::default()
            };
            let sa_dc3 = SuffixArray::with_config(text, &config_dc3).unwrap();

            let config_divsufsort = SuffixArrayConfig {
                algorithm: SuffixArrayAlgorithm::DivSufSort,
                ..Default::default()
            };
            let sa_divsufsort = SuffixArray::with_config(text, &config_divsufsort).unwrap();

            let config_ls = SuffixArrayConfig {
                algorithm: SuffixArrayAlgorithm::LarssonSadakane,
                ..Default::default()
            };
            let sa_ls = SuffixArray::with_config(text, &config_ls).unwrap();

            // Verify all algorithms produce properly sorted suffix arrays
            verify_suffix_array_is_sorted(text, sa_dc3.as_slice());
            verify_suffix_array_is_sorted(text, sa_divsufsort.as_slice());
            verify_suffix_array_is_sorted(text, sa_ls.as_slice());
        }
    }

    #[test]
    fn test_divsufsort_edge_cases() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::DivSufSort,
            ..Default::default()
        };

        // Empty string
        let sa = SuffixArray::with_config(b"", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 0);

        // Single character
        let sa = SuffixArray::with_config(b"a", &config).unwrap();
        assert_eq!(sa.as_slice(), &[0]);

        // Two characters
        let sa = SuffixArray::with_config(b"ab", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 2);
        assert!(
            sa.as_slice()[0] < sa.as_slice()[1]
                || b"ab"[sa.as_slice()[0]..] <= b"ab"[sa.as_slice()[1]..]
        );

        // All same characters
        let sa = SuffixArray::with_config(b"aaaa", &config).unwrap();
        assert_eq!(sa.as_slice(), &[3, 2, 1, 0]); // Longest suffix first when all equal
    }

    #[test]
    fn test_larsson_sadakane_edge_cases() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::LarssonSadakane,
            ..Default::default()
        };

        // Empty string
        let sa = SuffixArray::with_config(b"", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 0);

        // Single character
        let sa = SuffixArray::with_config(b"a", &config).unwrap();
        assert_eq!(sa.as_slice(), &[0]);

        // Two characters
        let sa = SuffixArray::with_config(b"ab", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 2);
        assert!(
            sa.as_slice()[0] < sa.as_slice()[1]
                || b"ab"[sa.as_slice()[0]..] <= b"ab"[sa.as_slice()[1]..]
        );

        // All same characters (this should be efficient for Larsson-Sadakane)
        let sa = SuffixArray::with_config(b"aaaa", &config).unwrap();
        assert_eq!(sa.as_slice(), &[3, 2, 1, 0]); // Longest suffix first when all equal

        // Highly repetitive pattern
        let sa = SuffixArray::with_config(b"abababab", &config).unwrap();
        assert_eq!(sa.as_slice().len(), 8);

        // Verify it's sorted
        let text = b"abababab";
        let suffixes = sa.as_slice();
        for i in 1..suffixes.len() {
            let suffix1 = &text[suffixes[i - 1]..];
            let suffix2 = &text[suffixes[i]..];
            assert!(suffix1 <= suffix2);
        }
    }

    #[test]
    fn test_adaptive_algorithm_selection_updated() {
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::Adaptive,
            adaptive_threshold: 100,
            ..Default::default()
        };
        let builder = SuffixArrayBuilder::new(config);

        // Small input should select DC3 (less than adaptive_threshold)
        let algorithm = builder.select_algorithm(b"small");
        assert_eq!(algorithm, SuffixArrayAlgorithm::DC3);

        // Highly repetitive should select Larsson-Sadakane
        let algorithm = builder.select_algorithm(&vec![b'a'; 1000]);
        // The adaptive logic may select SAIS for very small alphabets (size 1), so allow either
        assert!(
            algorithm == SuffixArrayAlgorithm::LarssonSadakane
                || algorithm == SuffixArrayAlgorithm::SAIS
        );

        // Large and medium inputs select DivSufSort (`text_length > 50_000`).
        // Miri interprets every byte of the characteristics scan, so it
        // checks both at 50,001, the smallest length that selects it.
        let (large, medium) = if cfg!(miri) {
            (50_001, 50_001)
        } else {
            (100_000, 60_000)
        };

        // Large input should select DivSufSort
        let large_diverse: Vec<u8> = (0..large).map(|i| (i % 256) as u8).collect();
        let algorithm = builder.select_algorithm(&large_diverse);
        assert_eq!(algorithm, SuffixArrayAlgorithm::DivSufSort);

        // Medium size should select DivSufSort
        let medium_diverse: Vec<u8> = (0..medium).map(|i| (i % 256) as u8).collect();
        let algorithm = builder.select_algorithm(&medium_diverse);
        assert_eq!(algorithm, SuffixArrayAlgorithm::DivSufSort);
    }

    #[test]
    fn test_repetitive_data_performance() {
        // Test that adaptive selection works for repetitive data and produces correct results
        // Use DivSufSort which handles repetitive data correctly without the SA-IS edge case
        let config = SuffixArrayConfig {
            algorithm: SuffixArrayAlgorithm::DivSufSort,
            adaptive_threshold: 10,
            ..Default::default()
        };

        // Create highly repetitive data (exactly 225 characters to match test expectation)
        let repetitive_text = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";

        let builder = SuffixArrayBuilder::new(config);
        let _algorithm = builder.select_algorithm(repetitive_text);

        // Algorithm selection may vary based on data characteristics
        // Just verify that the resulting suffix array is correct

        let sa = SuffixArray::with_config(repetitive_text, &builder.config).unwrap();

        // Verify the result is correct
        assert_eq!(sa.as_slice().len(), repetitive_text.len());

        // Verify it's properly sorted
        let suffixes = sa.as_slice();
        for i in 1..suffixes.len() {
            let suffix1 = &repetitive_text[suffixes[i - 1]..];
            let suffix2 = &repetitive_text[suffixes[i]..];
            assert!(
                suffix1 <= suffix2,
                "Repetitive data suffix array not properly sorted at position {}",
                i
            );
        }
    }

    #[test]
    fn test_c8_13_suffix_array_kasai_validation_and_differential_vs_naive() {
        // 1. LcpArray::new must reject mismatched text and suffix array lengths instead of panicking.
        let sa_short = SuffixArray::new(b"abc").unwrap();
        assert!(LcpArray::new(b"abcdef", &sa_short).is_err());
        assert!(LcpArray::new(b"a", &sa_short).is_err());

        // 2. Differential test across SAIS, DC3, DivSufSort, LarssonSadakane, Adaptive,
        // LCP, BWT, and search vs naive suffix sort on small alphabets, all-equal bytes,
        // and embedded NUL bytes up to n = 2000 (32 under Miri).
        let max_n = if cfg!(miri) { 32 } else { 2000 };
        let corpora: Vec<Vec<u8>> = vec![
            vec![],
            vec![0],
            vec![b'a'; max_n / 4],
            (0..max_n).map(|i| ((i * 7 + (i >> 3)) % 2) as u8).collect(), // {0, 1} with embedded 0x00
            (0..max_n)
                .map(|i| 1 + ((i * 13 + (i >> 2)) % 3) as u8)
                .collect(), // {1, 2, 3}
            b"\0a\0\0b\0a\0\0a\0b\0".to_vec(),
        ];

        for text in &corpora {
            let mut naive_sa: Vec<usize> = (0..text.len()).collect();
            naive_sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));

            let mut naive_lcp = vec![0usize; text.len()];
            for i in 1..text.len() {
                let s1 = &text[naive_sa[i - 1]..];
                let s2 = &text[naive_sa[i]..];
                naive_lcp[i] = s1.iter().zip(s2.iter()).take_while(|(a, b)| a == b).count();
            }

            for alg in [
                SuffixArrayAlgorithm::SAIS,
                SuffixArrayAlgorithm::DC3,
                SuffixArrayAlgorithm::DivSufSort,
                SuffixArrayAlgorithm::LarssonSadakane,
                SuffixArrayAlgorithm::Adaptive,
            ] {
                let cfg = SuffixArrayConfig {
                    algorithm: alg,
                    compute_lcp: true,
                    use_parallel: false,
                    ..Default::default()
                };
                let sa = SuffixArray::with_config(text, &cfg).unwrap();
                assert_eq!(
                    sa.as_slice(),
                    naive_sa.as_slice(),
                    "SA mismatch for {:?}",
                    alg
                );

                let lcp = LcpArray::new(text, &sa).unwrap();
                assert_eq!(
                    lcp.as_slice(),
                    naive_lcp.as_slice(),
                    "LCP mismatch for {:?}",
                    alg
                );
            }

            if !text.is_empty() {
                let pat_len = (text.len() / 3).clamp(1, 8);
                let pattern = &text[..pat_len];
                let sa = SuffixArray::new(text).unwrap();
                let (lo, hi) = sa.search_range(text, pattern);
                let (search_lo, count) = sa.search(text, pattern);
                assert_eq!(search_lo, lo);
                assert_eq!(count, hi - lo);
                let mut matches: Vec<usize> = sa.as_slice()[lo..hi].to_vec();
                matches.sort_unstable();
                let expected_matches: Vec<usize> = (0..=text.len() - pat_len)
                    .filter(|&i| &text[i..i + pat_len] == pattern)
                    .collect();
                assert_eq!(matches, expected_matches);
            }
        }
    }

    /// Every byte string of length `0..=max_len` over `alphabet`.
    fn all_byte_strings_up_to(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        let mut out = vec![Vec::new()];
        let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::with_capacity(frontier.len() * alphabet.len());
            for s in &frontier {
                for &c in alphabet {
                    let mut t = s.clone();
                    t.push(c);
                    next.push(t);
                }
            }
            out.extend(next.iter().cloned());
            frontier = next;
        }
        out
    }

    #[test]
    fn test_s20_r2_suffix_array_exhaustive_short_texts_all_paths() {
        // Every text of length 0..=10 over {a,b}, 0..=7 over {a,b,c}, and
        // 0..=6 over {0x00,0x01,0xFF}, so every length-2 input with equal
        // bytes is covered. Each text of length 2 or more costs 33 SA-IS
        // constructions, about 0.65 s each under Miri, so Miri checks
        // lengths 0..=2 over {a,b} ("aa" and "bb" included) and 0..=1 over
        // the other two alphabets. Under Miri, 0x00 and 0xFF reach SA-IS in
        // the byte sweep of test_sais_correctness_vs_reference (0x00 also
        // in test_c8_13).
        let (ab, abc, bin) = if cfg!(miri) { (2, 1, 1) } else { (10, 7, 6) };
        let mut corpora: Vec<(&[u8], Vec<Vec<u8>>)> = vec![
            (b"ab", all_byte_strings_up_to(b"ab", ab)),
            (b"abc", all_byte_strings_up_to(b"abc", abc)),
            (
                &[0x00, 0x01, 0xFF],
                all_byte_strings_up_to(&[0x00, 0x01, 0xFF], bin),
            ),
        ];

        let mut wrong: std::collections::BTreeMap<String, usize> = Default::default();
        let mut first: Vec<String> = Vec::new();
        let mut fail = |path: String, detail: String| {
            if first.len() < 8 {
                first.push(format!("{path}: {detail}"));
            }
            *wrong.entry(path).or_default() += 1;
        };
        let mut texts_checked = 0usize;

        for (alphabet, texts) in corpora.iter_mut() {
            let patterns = all_byte_strings_up_to(alphabet, 2);
            for text in texts.iter() {
                texts_checked += 1;
                let text = text.as_slice();
                let mut naive_sa: Vec<usize> = (0..text.len()).collect();
                naive_sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
                let naive_lcp: Vec<usize> = (0..text.len())
                    .map(|i| {
                        if i == 0 {
                            return 0;
                        }
                        let (s1, s2) = (&text[naive_sa[i - 1]..], &text[naive_sa[i]..]);
                        s1.iter().zip(s2).take_while(|(x, y)| x == y).count()
                    })
                    .collect();
                let naive_bwt: Vec<u8> = naive_sa
                    .iter()
                    .map(|&p| text[if p == 0 { text.len() - 1 } else { p - 1 }])
                    .collect();

                for alg in [
                    SuffixArrayAlgorithm::SAIS,
                    SuffixArrayAlgorithm::DC3,
                    SuffixArrayAlgorithm::DivSufSort,
                    SuffixArrayAlgorithm::LarssonSadakane,
                    SuffixArrayAlgorithm::Adaptive,
                ] {
                    for (label, cfg) in [
                        (
                            "sequential",
                            SuffixArrayConfig {
                                algorithm: alg,
                                use_parallel: false,
                                ..Default::default()
                            },
                        ),
                        (
                            "parallel-entry",
                            SuffixArrayConfig {
                                algorithm: alg,
                                use_parallel: true,
                                parallel_threshold: 1,
                                ..Default::default()
                            },
                        ),
                        (
                            "adaptive_threshold=0",
                            SuffixArrayConfig {
                                algorithm: alg,
                                use_parallel: false,
                                adaptive_threshold: 0,
                                ..Default::default()
                            },
                        ),
                    ] {
                        let sa = SuffixArray::with_config(text, &cfg).unwrap();
                        if sa.as_slice() != naive_sa.as_slice() {
                            fail(
                                format!("{alg:?}/{label}/with_config"),
                                format!("text={text:?} got {:?} want {naive_sa:?}", sa.as_slice()),
                            );
                        }
                        let built = SuffixArrayBuilder::new(cfg.clone()).build(text).unwrap();
                        if built.as_slice() != naive_sa.as_slice() {
                            fail(
                                format!("{alg:?}/{label}/builder"),
                                format!(
                                    "text={text:?} got {:?} want {naive_sa:?}",
                                    built.as_slice()
                                ),
                            );
                        }
                        let lcp = LcpArray::new(text, &sa).unwrap();
                        if lcp.as_slice() != naive_lcp.as_slice() {
                            fail(
                                format!("{alg:?}/{label}/kasai"),
                                format!(
                                    "text={text:?} got {:?} want {naive_lcp:?}",
                                    lcp.as_slice()
                                ),
                            );
                        }
                    }
                }

                // Default entry points (Adaptive, n < 10,000 -> DC3 before the fix).
                let sa = SuffixArray::new(text).unwrap();
                if sa.as_slice() != naive_sa.as_slice() {
                    fail(
                        "SuffixArray::new".into(),
                        format!("text={text:?} got {:?} want {naive_sa:?}", sa.as_slice()),
                    );
                }
                let esa = EnhancedSuffixArray::with_lcp(text).unwrap();
                if esa.suffix_array().as_slice() != naive_sa.as_slice()
                    || esa.lcp_array().map(|l| l.as_slice()) != Some(naive_lcp.as_slice())
                {
                    fail(
                        "EnhancedSuffixArray::with_lcp".into(),
                        format!("text={text:?}"),
                    );
                }
                let esa = EnhancedSuffixArray::with_bwt(text).unwrap();
                if esa.bwt() != Some(naive_bwt.as_slice()) {
                    fail(
                        "EnhancedSuffixArray::with_bwt".into(),
                        format!("text={text:?} got {:?} want {naive_bwt:?}", esa.bwt()),
                    );
                }

                // search / search_range against a naive scan, patterns of length 1..=2.
                for pattern in patterns.iter().filter(|p| !p.is_empty()) {
                    let (lo, hi) = sa.search_range(text, pattern);
                    let mut found: Vec<usize> = sa.as_slice()[lo..hi].to_vec();
                    found.sort_unstable();
                    let expected: Vec<usize> = (0..text.len())
                        .filter(|&i| text[i..].starts_with(pattern))
                        .collect();
                    if found != expected || sa.search(text, pattern) != (lo, hi - lo) {
                        fail(
                            "SuffixArray::search_range".into(),
                            format!(
                                "text={text:?} pattern={pattern:?} got {found:?} want {expected:?}"
                            ),
                        );
                    }
                }
            }
        }

        assert!(
            wrong.is_empty(),
            "S20-R2: {} wrong results over {texts_checked} texts, per path: {wrong:#?}; first: {first:#?}",
            wrong.values().sum::<usize>()
        );
    }

    #[test]
    fn test_s20_r6_suffix_array_reports_sequential_construction() {
        // There is no parallel construction (the former `build_parallel` only
        // called `build_sequential`), yet `stats().used_parallel` echoed
        // `use_parallel && len >= parallel_threshold` and `supports_parallel()`
        // returned true.
        let text = b"mississippi$banana";
        let config = SuffixArrayConfig {
            use_parallel: true,
            parallel_threshold: 1,
            ..Default::default()
        };
        let sa = SuffixArray::with_config(text, &config).unwrap();
        let sequential = SuffixArray::with_config(
            text,
            &SuffixArrayConfig {
                use_parallel: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sa.as_slice(), sequential.as_slice());
        assert!(
            !sa.stats().used_parallel,
            "used_parallel with use_parallel = true, parallel_threshold = 1"
        );
        assert!(!SuffixArrayBuilder::new(config).supports_parallel());
    }
}
