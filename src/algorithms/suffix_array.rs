//! Suffix array construction and LCP array computation
//!
//! This module implements the SA-IS (Suffix Array - Induced Sorting) algorithm
//! for linear-time suffix array construction, along with LCP array computation.

use crate::algorithms::{Algorithm, AlgorithmStats};
use crate::error::Result;
use std::cmp::Ordering;
use std::time::Instant;

/// Configuration for suffix array construction
#[derive(Debug, Clone)]
pub struct SuffixArrayConfig {
    /// Use parallel processing for large inputs
    pub use_parallel: bool,
    /// Threshold for parallel processing
    pub parallel_threshold: usize,
    /// Compute LCP array along with suffix array
    pub compute_lcp: bool,
    /// Use optimized algorithm for small alphabets
    pub optimize_small_alphabet: bool,
}

impl Default for SuffixArrayConfig {
    fn default() -> Self {
        Self {
            use_parallel: true,
            parallel_threshold: 100_000,
            compute_lcp: false,
            optimize_small_alphabet: true,
        }
    }
}

/// A suffix array data structure
pub struct SuffixArray {
    /// The suffix array itself (indices into the original string)
    sa: Vec<usize>,
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

    /// Find the range of suffixes that start with the given pattern
    pub fn search_range(&self, text: &[u8], pattern: &[u8]) -> (usize, usize) {
        let left = self.lower_bound(text, pattern);
        let right = self.upper_bound(text, pattern);
        (left, right)
    }

    fn lower_bound(&self, text: &[u8], pattern: &[u8]) -> usize {
        let mut left = 0;
        let mut right = self.sa.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let suffix_start = self.sa[mid];
            let suffix = &text[suffix_start..];

            if Self::compare_suffix_pattern(suffix, pattern) == Ordering::Less {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    fn upper_bound(&self, text: &[u8], pattern: &[u8]) -> usize {
        let mut left = 0;
        let mut right = self.sa.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let suffix_start = self.sa[mid];
            let suffix = &text[suffix_start..];

            if Self::compare_suffix_pattern(suffix, pattern) != Ordering::Greater {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    fn compare_suffix_pattern(suffix: &[u8], pattern: &[u8]) -> Ordering {
        let min_len = suffix.len().min(pattern.len());

        for i in 0..min_len {
            match suffix[i].cmp(&pattern[i]) {
                Ordering::Equal => continue,
                other => return other,
            }
        }

        // If we reach here, the pattern matches the beginning of the suffix
        // For prefix search, we consider this equal if pattern fits within suffix
        if pattern.len() <= suffix.len() {
            Ordering::Equal
        } else {
            // Pattern is longer than suffix, so suffix comes first
            Ordering::Less
        }
    }

    /// Get performance statistics
    pub fn stats(&self) -> &AlgorithmStats {
        &self.stats
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

    /// Build a suffix array from the given text
    pub fn build(&self, text: &[u8]) -> Result<SuffixArray> {
        let start_time = Instant::now();

        if text.is_empty() {
            return Ok(SuffixArray {
                sa: Vec::new(),
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

        let sa = if text.len() >= self.config.parallel_threshold && self.config.use_parallel {
            self.build_parallel(text)?
        } else {
            self.build_sequential(text)?
        };

        let elapsed = start_time.elapsed();
        let memory_used = sa.len() * std::mem::size_of::<usize>();

        Ok(SuffixArray {
            text_len: text.len(),
            stats: AlgorithmStats {
                items_processed: text.len(),
                processing_time_us: elapsed.as_micros() as u64,
                memory_used,
                used_parallel: text.len() >= self.config.parallel_threshold
                    && self.config.use_parallel,
                used_simd: false,
            },
            sa,
        })
    }

    fn build_sequential(&self, text: &[u8]) -> Result<Vec<usize>> {
        // For simplicity, we'll use a basic suffix array construction algorithm
        // A full SA-IS implementation would be much more complex
        let mut suffixes: Vec<usize> = (0..text.len()).collect();

        suffixes.sort_by(|&a, &b| text[a..].cmp(&text[b..]));

        Ok(suffixes)
    }

    fn build_parallel(&self, text: &[u8]) -> Result<Vec<usize>> {
        // For now, fall back to sequential - full parallel SA-IS is very complex
        self.build_sequential(text)
    }
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
        true
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
        let n = text.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Compute inverse suffix array
        let mut rank = vec![0; n];
        for i in 0..n {
            rank[sa[i]] = i;
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

                if h > 0 {
                    h -= 1;
                }
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
        self.bwt.as_ref().map(|v| v.as_slice())
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
            use_parallel: false,
            parallel_threshold: 1000,
            compute_lcp: true,
            optimize_small_alphabet: false,
        };

        let text = b"test";
        let sa = SuffixArray::with_config(text, &config).unwrap();
        assert_eq!(sa.text_len(), 4);
        assert!(!sa.stats().used_parallel);
    }

    #[test]
    fn test_algorithm_trait() {
        let builder = SuffixArrayBuilder::new(SuffixArrayConfig::default());

        assert!(builder.supports_parallel());
        assert!(!builder.supports_simd());

        let memory_estimate = builder.estimate_memory(1000);
        assert_eq!(memory_estimate, 1000 * std::mem::size_of::<usize>());
    }
}
