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
#[derive(Debug)]
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
        if text.is_empty() {
            return Ok(Vec::new());
        }

        if text.len() == 1 {
            return Ok(vec![0]);
        }

        // Use SA-IS algorithm for efficient suffix array construction
        Ok(self.sais_construct(text)?)
    }

    /// SA-IS (Suffix Array by Induced Sorting) algorithm implementation
    fn sais_construct(&self, text: &[u8]) -> Result<Vec<usize>> {
        let n = text.len();
        
        // Find alphabet size
        let alphabet_size = if self.config.optimize_small_alphabet {
            256 // Full byte alphabet
        } else {
            text.iter().max().unwrap_or(&0).wrapping_add(1) as usize
        };

        // Step 1: Classify suffixes as L-type or S-type
        let (suffix_types, is_lms) = self.classify_suffixes(text)?;

        // Step 2: Find LMS suffixes
        let lms_suffixes = self.find_lms_suffixes(&is_lms);

        if lms_suffixes.is_empty() {
            // All suffixes are L-type (monotonically decreasing string)
            return Ok((0..n).rev().collect());
        }

        // Step 3: Sort LMS suffixes
        let mut sa = vec![0; n];
        let mut bucket = vec![0; alphabet_size];
        let mut bucket_heads = vec![0; alphabet_size];
        let mut bucket_tails = vec![0; alphabet_size];

        // Count character frequencies
        for &ch in text {
            bucket[ch as usize] += 1;
        }

        // Compute bucket boundaries
        self.compute_bucket_boundaries(&bucket, &mut bucket_heads, &mut bucket_tails);

        // Initialize SA with sentinel values
        for i in 0..n {
            sa[i] = n; // Use n as sentinel (invalid index)
        }

        // Place LMS suffixes at the end of their buckets
        for &lms_idx in lms_suffixes.iter().rev() {
            let ch = text[lms_idx] as usize;
            if bucket_tails[ch] > 0 {
                bucket_tails[ch] -= 1;
                sa[bucket_tails[ch]] = lms_idx;
            }
        }

        // Induce L-type suffixes
        self.induce_l_type(&mut sa, text, &suffix_types, &bucket_heads)?;

        // Induce S-type suffixes
        self.induce_s_type(&mut sa, text, &suffix_types, &bucket_tails)?;

        // Step 4: Compact LMS suffixes and check if they're unique
        let lms_sa = self.compact_lms_suffixes(&sa, &is_lms);
        let lms_names = self.name_lms_substrings(text, &lms_sa, &lms_suffixes)?;

        // Check if all LMS substrings are unique
        let max_name = lms_names.iter().max().copied().unwrap_or(0);
        
        if (max_name as usize) < lms_suffixes.len() {
            // Not all LMS substrings are unique, recursively sort them
            let reduced_sa = self.sais_construct(&lms_names)?;
            
            // Map back to original indices
            let mut sorted_lms = Vec::new();
            for &rank in &reduced_sa {
                sorted_lms.push(lms_suffixes[rank]);
            }

            // Rebuild SA with sorted LMS suffixes
            self.rebuild_sa_with_sorted_lms(text, &sorted_lms, &suffix_types, alphabet_size)
        } else {
            // All LMS substrings are unique, SA is complete
            Ok(sa.into_iter().filter(|&x| x < n).collect())
        }
    }

    /// Classify each suffix as L-type or S-type
    fn classify_suffixes(&self, text: &[u8]) -> Result<(Vec<bool>, Vec<bool>)> {
        let n = text.len();
        let mut suffix_types = vec![false; n]; // false = L-type, true = S-type
        let mut is_lms = vec![false; n];

        if n == 0 {
            return Ok((suffix_types, is_lms));
        }

        // Last suffix is S-type by definition
        suffix_types[n - 1] = true;

        // Classify suffixes from right to left
        for i in (0..n - 1).rev() {
            if text[i] < text[i + 1] {
                suffix_types[i] = true; // S-type
            } else if text[i] > text[i + 1] {
                suffix_types[i] = false; // L-type
            } else {
                // Same character, inherit from next position
                suffix_types[i] = suffix_types[i + 1];
            }
        }

        // Find LMS positions (Left-Most S-type)
        for i in 1..n {
            if suffix_types[i] && !suffix_types[i - 1] {
                is_lms[i] = true;
            }
        }

        Ok((suffix_types, is_lms))
    }

    /// Find all LMS suffix positions
    fn find_lms_suffixes(&self, is_lms: &[bool]) -> Vec<usize> {
        is_lms.iter()
            .enumerate()
            .filter_map(|(i, &is_lms_pos)| if is_lms_pos { Some(i) } else { None })
            .collect()
    }

    /// Compute bucket head and tail positions
    fn compute_bucket_boundaries(
        &self,
        bucket: &[usize],
        bucket_heads: &mut [usize],
        bucket_tails: &mut [usize],
    ) {
        let mut sum = 0;
        for i in 0..bucket.len() {
            bucket_heads[i] = sum;
            sum += bucket[i];
            bucket_tails[i] = sum;
        }
    }

    /// Induce L-type suffixes from left to right
    fn induce_l_type(
        &self,
        sa: &mut [usize],
        text: &[u8],
        suffix_types: &[bool],
        bucket_heads: &[usize],
    ) -> Result<()> {
        let n = text.len();
        let mut heads = bucket_heads.to_vec();

        for i in 0..n {
            if sa[i] == n {
                continue; // Skip sentinel values
            }

            let j = sa[i];
            if j > 0 && !suffix_types[j - 1] {
                // Predecessor is L-type
                let ch = text[j - 1] as usize;
                if heads[ch] < n {
                    sa[heads[ch]] = j - 1;
                    heads[ch] += 1;
                }
            }
        }

        Ok(())
    }

    /// Induce S-type suffixes from right to left
    fn induce_s_type(
        &self,
        sa: &mut [usize],
        text: &[u8],
        suffix_types: &[bool],
        bucket_tails: &[usize],
    ) -> Result<()> {
        let n = text.len();
        let mut tails = bucket_tails.to_vec();

        for i in (0..n).rev() {
            if sa[i] == n {
                continue; // Skip sentinel values
            }

            let j = sa[i];
            if j > 0 && suffix_types[j - 1] {
                // Predecessor is S-type
                let ch = text[j - 1] as usize;
                if tails[ch] > 0 {
                    tails[ch] -= 1;
                    sa[tails[ch]] = j - 1;
                }
            }
        }

        Ok(())
    }

    /// Compact LMS suffixes from the suffix array
    fn compact_lms_suffixes(&self, sa: &[usize], is_lms: &[bool]) -> Vec<usize> {
        sa.iter()
            .filter_map(|&pos| {
                if pos < is_lms.len() && is_lms[pos] {
                    Some(pos)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Assign names to LMS substrings based on their lexicographic order
    fn name_lms_substrings(
        &self,
        text: &[u8],
        lms_sa: &[usize],
        lms_suffixes: &[usize],
    ) -> Result<Vec<u8>> {
        let mut names = vec![0u8; lms_suffixes.len()];
        let mut current_name = 0u8;

        if !lms_sa.is_empty() {
            names[0] = current_name;

            for i in 1..lms_sa.len() {
                if !self.are_lms_substrings_equal(text, lms_sa[i - 1], lms_sa[i], lms_suffixes)? {
                    current_name = current_name.wrapping_add(1);
                }
                
                // Find position of lms_sa[i] in lms_suffixes
                let pos = lms_suffixes.iter().position(|&x| x == lms_sa[i])
                    .ok_or_else(|| crate::error::ZiporaError::invalid_data("LMS suffix not found"))?;
                names[pos] = current_name;
            }
        }

        Ok(names)
    }

    /// Check if two LMS substrings are equal
    fn are_lms_substrings_equal(
        &self,
        text: &[u8],
        pos1: usize,
        pos2: usize,
        lms_suffixes: &[usize],
    ) -> Result<bool> {
        if pos1 >= text.len() || pos2 >= text.len() {
            return Ok(false);
        }

        // Find the end of each LMS substring
        let end1 = self.find_lms_substring_end(pos1, lms_suffixes, text.len());
        let end2 = self.find_lms_substring_end(pos2, lms_suffixes, text.len());

        let len1 = end1 - pos1;
        let len2 = end2 - pos2;

        if len1 != len2 {
            return Ok(false);
        }

        // Compare character by character
        for i in 0..len1 {
            if text[pos1 + i] != text[pos2 + i] {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Find the end position of an LMS substring
    fn find_lms_substring_end(&self, start: usize, lms_suffixes: &[usize], text_len: usize) -> usize {
        // Find next LMS position after start
        lms_suffixes.iter()
            .find(|&&pos| pos > start)
            .copied()
            .unwrap_or(text_len)
    }

    /// Rebuild the suffix array with sorted LMS suffixes
    fn rebuild_sa_with_sorted_lms(
        &self,
        text: &[u8],
        sorted_lms: &[usize],
        suffix_types: &[bool],
        alphabet_size: usize,
    ) -> Result<Vec<usize>> {
        let n = text.len();
        let mut sa = vec![n; n]; // Initialize with sentinel values
        let mut bucket = vec![0; alphabet_size];
        let mut bucket_heads = vec![0; alphabet_size];
        let mut bucket_tails = vec![0; alphabet_size];

        // Count character frequencies
        for &ch in text {
            bucket[ch as usize] += 1;
        }

        // Compute bucket boundaries
        self.compute_bucket_boundaries(&bucket, &mut bucket_heads, &mut bucket_tails);

        // Place sorted LMS suffixes
        for &lms_pos in sorted_lms.iter().rev() {
            let ch = text[lms_pos] as usize;
            if bucket_tails[ch] > 0 {
                bucket_tails[ch] -= 1;
                sa[bucket_tails[ch]] = lms_pos;
            }
        }

        // Induce L-type and S-type suffixes
        self.induce_l_type(&mut sa, text, suffix_types, &bucket_heads)?;
        self.induce_s_type(&mut sa, text, suffix_types, &bucket_tails)?;

        Ok(sa.into_iter().filter(|&x| x < n).collect())
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
