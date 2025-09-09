//! Pattern Matching Engine for PA-Zip Dictionary
//!
//! This module provides the pattern matching engine that combines DFA cache
//! lookups with suffix array binary search to find optimal matches for compression.
//! It implements the core matching logic used by the PA-Zip compression algorithm.
//!
//! # Algorithm Overview
//!
//! The pattern matching follows a two-tier approach:
//! 1. **Fast path**: Use DFA cache for O(1) prefix matching
//! 2. **Slow path**: Use suffix array binary search for complete pattern matching
//! 3. **Extension**: Extend cache matches using suffix array for maximum length
//!
//! # Performance Optimizations
//!
//! - **SIMD acceleration**: For bulk character comparisons
//! - **Cache-friendly access**: Sequential memory access patterns
//! - **Early termination**: Stop search when no better match possible
//! - **Adaptive thresholds**: Adjust search strategy based on match quality

use crate::algorithms::suffix_array::SuffixArray;
use crate::compression::dict_zip::dfa_cache::CacheMatch;
use crate::error::Result;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "simd")]
use std::arch::x86_64::*;

use std::cmp::Ordering;
use std::sync::Arc;

/// Match result containing position and length information
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Match {
    /// Length of the matched pattern
    pub length: usize,
    /// Position in the dictionary where the pattern was found
    pub dict_position: usize,
    /// Position in the input where the match starts
    pub input_position: usize,
    /// Match quality score (0.0 to 1.0, higher is better)
    pub quality: f64,
    /// Whether this match came from DFA cache (true) or suffix array (false)
    pub from_cache: bool,
}

impl Match {
    /// Create a new match
    pub fn new(
        length: usize,
        dict_position: usize,
        input_position: usize,
        from_cache: bool,
    ) -> Self {
        let quality = Self::calculate_quality(length);
        Self {
            length,
            dict_position,
            input_position,
            quality,
            from_cache,
        }
    }

    /// Calculate match quality based on length
    fn calculate_quality(length: usize) -> f64 {
        // Quality increases with length, but with diminishing returns
        let normalized_length = length as f64 / 256.0; // Normalize to typical max pattern length
        1.0 - (-normalized_length * 2.0).exp() // Exponential approach to 1.0
    }

    /// Check if this match is better than another
    pub fn is_better_than(&self, other: &Match) -> bool {
        // Primarily compare by length, then by quality
        match self.length.cmp(&other.length) {
            Ordering::Greater => true,
            Ordering::Equal => self.quality > other.quality,
            Ordering::Less => false,
        }
    }
}

/// Configuration for pattern matching
#[derive(Debug, Clone)]
pub struct MatcherConfig {
    /// Minimum match length to consider
    pub min_match_length: usize,
    /// Maximum match length to search for
    pub max_match_length: usize,
    /// Use SIMD optimizations for comparisons
    pub enable_simd: bool,
    /// Maximum number of suffix array comparisons per search
    pub max_sa_comparisons: usize,
    /// Early termination threshold (stop if match quality exceeds this)
    pub early_termination_quality: f64,
    /// Cache extension maximum additional length
    pub max_cache_extension: usize,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            min_match_length: 4,
            max_match_length: 256,
            enable_simd: cfg!(feature = "simd"),
            max_sa_comparisons: 100,
            early_termination_quality: 0.95,
            max_cache_extension: 64,
        }
    }
}

/// Statistics for pattern matching performance
#[derive(Debug, Clone, Default)]
pub struct MatcherStats {
    /// Total number of match attempts
    pub total_matches: u64,
    /// Number of successful matches
    pub successful_matches: u64,
    /// Number of cache extensions performed
    pub cache_extensions: u64,
    /// Number of full suffix array searches
    pub full_sa_searches: u64,
    /// Total comparison operations performed
    pub total_comparisons: u64,
    /// Average match length for successful matches
    pub avg_match_length: f64,
    /// Total matching time in microseconds
    pub total_match_time_us: u64,
}

impl MatcherStats {
    /// Calculate success ratio
    pub fn success_ratio(&self) -> f64 {
        if self.total_matches == 0 {
            0.0
        } else {
            self.successful_matches as f64 / self.total_matches as f64
        }
    }

    /// Calculate average matching time
    pub fn avg_match_time_us(&self) -> f64 {
        if self.total_matches == 0 {
            0.0
        } else {
            self.total_match_time_us as f64 / self.total_matches as f64
        }
    }
}

/// High-performance pattern matcher for PA-Zip dictionary
#[derive(Debug, Clone)]
pub struct PatternMatcher {
    /// Reference to suffix array
    suffix_array: Arc<SuffixArray>,
    /// Reference to dictionary text
    dictionary_text: Arc<Vec<u8>>,
    /// Matcher configuration
    config: MatcherConfig,
    /// Performance statistics
    stats: MatcherStats,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    ///
    /// # Arguments
    /// * `suffix_array` - Suffix array for binary search
    /// * `dictionary_text` - Original dictionary text
    /// * `min_length` - Minimum pattern length
    /// * `max_length` - Maximum pattern length
    pub fn new(
        suffix_array: Arc<SuffixArray>,
        dictionary_text: Arc<Vec<u8>>,
        min_length: usize,
        max_length: usize,
    ) -> Self {
        let config = MatcherConfig {
            min_match_length: min_length,
            max_match_length: max_length,
            ..Default::default()
        };

        Self {
            suffix_array,
            dictionary_text,
            config,
            stats: MatcherStats::default(),
        }
    }

    /// Create matcher with custom configuration
    pub fn with_config(
        suffix_array: Arc<SuffixArray>,
        dictionary_text: Arc<Vec<u8>>,
        config: MatcherConfig,
    ) -> Self {
        Self {
            suffix_array,
            dictionary_text,
            config,
            stats: MatcherStats::default(),
        }
    }

    /// Extend a cache match using suffix array to find the longest possible match
    ///
    /// # Arguments
    /// * `input` - Input data being compressed
    /// * `input_pos` - Position in input
    /// * `cache_match` - Initial match from DFA cache
    /// * `max_length` - Maximum total length to search
    ///
    /// # Returns
    /// Extended match or None if cache match cannot be extended
    pub fn extend_match_from_cache(
        &mut self,
        input: &[u8],
        input_pos: usize,
        cache_match: CacheMatch,
        max_length: usize,
    ) -> Result<Option<Match>> {
        let start_time = std::time::Instant::now();
        self.stats.total_matches += 1;

        // Start with the cache match
        let mut best_match = Match::new(
            cache_match.length,
            cache_match.dict_position,
            input_pos,
            true,
        );

        // Try to extend the match beyond what the cache provided
        let extension_start = cache_match.length;
        let max_extension = (max_length - cache_match.length)
            .min(self.config.max_cache_extension)
            .min(input.len() - input_pos - cache_match.length);

        if max_extension > 0 {
            self.stats.cache_extensions += 1;
            
            let extended_length = self.extend_match_at_position(
                input,
                input_pos + extension_start,
                cache_match.dict_position + extension_start,
                max_extension,
            )?;

            if extended_length > 0 {
                best_match.length += extended_length;
                best_match.quality = Match::calculate_quality(best_match.length);
            }
        }

        self.stats.successful_matches += 1;
        self.stats.total_match_time_us += start_time.elapsed().as_micros() as u64;

        // Update rolling average
        let total_length = self.stats.avg_match_length * (self.stats.successful_matches - 1) as f64
                          + best_match.length as f64;
        self.stats.avg_match_length = total_length / self.stats.successful_matches as f64;

        Ok(Some(best_match))
    }

    /// Find longest match using suffix array binary search
    ///
    /// # Arguments
    /// * `input` - Input data being compressed
    /// * `input_pos` - Position in input
    /// * `max_length` - Maximum length to search
    ///
    /// # Returns
    /// Best match found or None
    pub fn find_longest_match_suffix_array(
        &mut self,
        input: &[u8],
        input_pos: usize,
        max_length: usize,
    ) -> Result<Option<Match>> {
        let start_time = std::time::Instant::now();
        self.stats.total_matches += 1;
        self.stats.full_sa_searches += 1;

        if input_pos >= input.len() {
            self.stats.total_match_time_us += start_time.elapsed().as_micros() as u64;
            return Ok(None);
        }

        let search_slice = &input[input_pos..];
        let max_search_len = max_length
            .min(search_slice.len())
            .min(self.config.max_match_length);

        if max_search_len < self.config.min_match_length {
            self.stats.total_match_time_us += start_time.elapsed().as_micros() as u64;
            return Ok(None);
        }

        let mut best_match: Option<Match> = None;
        let mut comparisons = 0;

        // Try patterns of decreasing length for best compression ratio
        for pattern_len in (self.config.min_match_length..=max_search_len).rev() {
            if comparisons >= self.config.max_sa_comparisons {
                break;
            }

            let pattern = &search_slice[..pattern_len];
            let (start_idx, count) = self.suffix_array.search(&self.dictionary_text, pattern);

            if count > 0 {
                // Found matches - take the first one (arbitrary choice among equals)
                if let Some(dict_pos) = self.suffix_array.suffix_at_rank(start_idx) {
                    let match_candidate = Match::new(pattern_len, dict_pos, input_pos, false);
                    
                    if best_match.as_ref().map_or(true, |m| match_candidate.is_better_than(m)) {
                        best_match = Some(match_candidate);
                        
                        // Early termination if match quality is very good
                        if best_match.as_ref().unwrap().quality >= self.config.early_termination_quality {
                            break;
                        }
                    }
                }
            }

            comparisons += 1;
            self.stats.total_comparisons += 1;
        }

        let elapsed_time = start_time.elapsed().as_micros() as u64;
        self.stats.total_match_time_us += elapsed_time;

        if best_match.is_some() {
            self.stats.successful_matches += 1;
            
            // Update rolling average
            let match_len = best_match.as_ref().unwrap().length as f64;
            let total_length = self.stats.avg_match_length * (self.stats.successful_matches - 1) as f64
                              + match_len;
            self.stats.avg_match_length = total_length / self.stats.successful_matches as f64;
        }

        Ok(best_match)
    }

    /// Find all matches for a given pattern
    ///
    /// # Arguments
    /// * `pattern` - Pattern to search for
    /// * `max_matches` - Maximum number of matches to return
    ///
    /// # Returns
    /// Vector of all matches found
    pub fn find_all_matches(&self, pattern: &[u8], max_matches: usize) -> Result<Vec<Match>> {
        if pattern.len() < self.config.min_match_length ||
           pattern.len() > self.config.max_match_length {
            return Ok(Vec::new());
        }

        let (start_idx, count) = self.suffix_array.search(&self.dictionary_text, pattern);
        let mut matches = Vec::new();

        let actual_count = count.min(max_matches);
        for i in 0..actual_count {
            if let Some(dict_pos) = self.suffix_array.suffix_at_rank(start_idx + i) {
                matches.push(Match::new(pattern.len(), dict_pos, 0, false));
            }
        }

        Ok(matches)
    }

    /// Get pattern matching statistics
    pub fn stats(&self) -> &MatcherStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MatcherStats::default();
    }

    /// Get matcher configuration
    pub fn config(&self) -> &MatcherConfig {
        &self.config
    }

    /// Extend a match at a specific position
    fn extend_match_at_position(
        &self,
        input: &[u8],
        input_pos: usize,
        dict_pos: usize,
        max_extension: usize,
    ) -> Result<usize> {
        if input_pos >= input.len() || dict_pos >= self.dictionary_text.len() {
            return Ok(0);
        }

        let input_remaining = &input[input_pos..];
        let dict_remaining = &self.dictionary_text[dict_pos..];
        
        let max_compare = max_extension
            .min(input_remaining.len())
            .min(dict_remaining.len());

        #[cfg(feature = "simd")]
        {
            if max_compare >= 16 && self.config.enable_simd {
                return Ok(self.simd_compare_and_extend(input_remaining, dict_remaining, max_compare));
            }
        }

        // Fallback to scalar comparison
        let mut extension = 0;
        for i in 0..max_compare {
            if input_remaining[i] == dict_remaining[i] {
                extension += 1;
            } else {
                break;
            }
        }

        Ok(extension)
    }

    /// SIMD-accelerated string comparison and extension
    #[cfg(feature = "simd")]
    fn simd_compare_and_extend(&self, input: &[u8], dict: &[u8], max_len: usize) -> usize {
        unsafe {
            let mut pos = 0;
            let simd_chunks = max_len / 16;

            // Process 16-byte chunks with SIMD
            for _ in 0..simd_chunks {
                if pos + 16 > input.len() || pos + 16 > dict.len() {
                    break;
                }

                let input_chunk = _mm_loadu_si128(input.as_ptr().add(pos) as *const __m128i);
                let dict_chunk = _mm_loadu_si128(dict.as_ptr().add(pos) as *const __m128i);
                
                let comparison = _mm_cmpeq_epi8(input_chunk, dict_chunk);
                let mask = _mm_movemask_epi8(comparison) as u16;

                if mask != 0xFFFF {
                    // Found mismatch within this chunk
                    let mismatch_pos = mask.trailing_ones() as usize;
                    return pos + mismatch_pos;
                }

                pos += 16;
            }

            // Handle remaining bytes with scalar comparison
            while pos < max_len && pos < input.len() && pos < dict.len() {
                if input[pos] != dict[pos] {
                    break;
                }
                pos += 1;
            }

            pos
        }
    }
}

/// Builder for creating pattern matchers with custom configurations
pub struct PatternMatcherBuilder {
    config: MatcherConfig,
}

impl PatternMatcherBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MatcherConfig::default(),
        }
    }

    /// Set minimum match length
    pub fn min_match_length(mut self, length: usize) -> Self {
        self.config.min_match_length = length;
        self
    }

    /// Set maximum match length
    pub fn max_match_length(mut self, length: usize) -> Self {
        self.config.max_match_length = length;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn enable_simd(mut self, enable: bool) -> Self {
        self.config.enable_simd = enable;
        self
    }

    /// Set maximum suffix array comparisons
    pub fn max_sa_comparisons(mut self, max: usize) -> Self {
        self.config.max_sa_comparisons = max;
        self
    }

    /// Set early termination quality threshold
    pub fn early_termination_quality(mut self, quality: f64) -> Self {
        self.config.early_termination_quality = quality.clamp(0.0, 1.0);
        self
    }

    /// Build the pattern matcher
    pub fn build(
        self,
        suffix_array: Arc<SuffixArray>,
        dictionary_text: Arc<Vec<u8>>,
    ) -> PatternMatcher {
        PatternMatcher::with_config(suffix_array, dictionary_text, self.config)
    }
}

impl Default for PatternMatcherBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::suffix_array::{SuffixArray, SuffixArrayConfig};
    use crate::compression::dict_zip::dfa_cache::CacheMatch;

    fn create_test_setup() -> (Arc<SuffixArray>, Arc<Vec<u8>>) {
        let text = b"the quick brown fox jumps over the lazy dog".to_vec();
        let sa = SuffixArray::with_config(&text, &SuffixArrayConfig::default()).unwrap();
        (Arc::new(sa), Arc::new(text))
    }

    #[test]
    fn test_matcher_creation() {
        let (sa, text) = create_test_setup();
        let matcher = PatternMatcher::new(sa, text, 3, 20);
        
        assert_eq!(matcher.config().min_match_length, 3);
        assert_eq!(matcher.config().max_match_length, 20);
    }

    #[test]
    fn test_suffix_array_matching() {
        let (sa, text) = create_test_setup();
        let mut matcher = PatternMatcher::new(sa, text, 3, 10);
        
        let input = b"the quick brown";
        let result = matcher.find_longest_match_suffix_array(input, 0, 10).unwrap();
        
        assert!(result.is_some());
        let match_result = result.unwrap();
        assert!(match_result.length >= 3);
        assert!(!match_result.from_cache);
    }

    #[test]
    fn test_cache_match_extension() {
        let (sa, text) = create_test_setup();
        let mut matcher = PatternMatcher::new(sa, text, 3, 20);
        
        // Create a mock cache match
        let cache_match = CacheMatch {
            length: 3,
            dict_position: 0, // "the" at position 0
            frequency: 2,
            state_id: 1,
        };
        
        let input = b"the quick";
        let result = matcher.extend_match_from_cache(input, 0, cache_match, 10).unwrap();
        
        assert!(result.is_some());
        let extended_match = result.unwrap();
        assert!(extended_match.from_cache);
        assert!(extended_match.length >= 3);
    }

    #[test]
    fn test_find_all_matches() {
        let (sa, text) = create_test_setup();
        let matcher = PatternMatcher::new(sa, text, 2, 10);
        
        let pattern = b"the";
        let matches = matcher.find_all_matches(pattern, 10).unwrap();
        
        // "the" appears twice in "the quick brown fox jumps over the lazy dog"
        assert_eq!(matches.len(), 2);
        
        for match_result in &matches {
            assert_eq!(match_result.length, 3);
            assert!(!match_result.from_cache);
        }
    }

    #[test]
    fn test_match_quality() {
        let match1 = Match::new(5, 0, 0, false);
        let match2 = Match::new(3, 10, 0, false);
        
        assert!(match1.is_better_than(&match2));
        assert!(!match2.is_better_than(&match1));
        
        // Test quality calculation
        assert!(match1.quality > match2.quality);
    }

    #[test]
    fn test_matcher_statistics() {
        let (sa, text) = create_test_setup();
        let mut matcher = PatternMatcher::new(sa, text, 3, 10);
        
        // Perform some matches
        let input = b"the";
        matcher.find_longest_match_suffix_array(input, 0, 5).unwrap();
        matcher.find_longest_match_suffix_array(input, 0, 5).unwrap();
        
        let stats = matcher.stats();
        assert_eq!(stats.total_matches, 2);
        assert!(stats.success_ratio() >= 0.0 && stats.success_ratio() <= 1.0);
    }

    #[test]
    fn test_pattern_matcher_builder() {
        let (sa, text) = create_test_setup();
        
        let matcher = PatternMatcherBuilder::new()
            .min_match_length(4)
            .max_match_length(50)
            .enable_simd(false)
            .max_sa_comparisons(200)
            .early_termination_quality(0.9)
            .build(sa, text);
        
        let config = matcher.config();
        assert_eq!(config.min_match_length, 4);
        assert_eq!(config.max_match_length, 50);
        assert!(!config.enable_simd);
        assert_eq!(config.max_sa_comparisons, 200);
        assert_eq!(config.early_termination_quality, 0.9);
    }

    #[test]
    fn test_empty_input_handling() {
        let (sa, text) = create_test_setup();
        let mut matcher = PatternMatcher::new(sa, text, 3, 10);
        
        let empty_input = b"";
        let result = matcher.find_longest_match_suffix_array(empty_input, 0, 10).unwrap();
        assert!(result.is_none());
        
        // Test with position beyond input length
        let input = b"test";
        let result = matcher.find_longest_match_suffix_array(input, 10, 10).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_short_pattern_rejection() {
        let (sa, text) = create_test_setup();
        let matcher = PatternMatcher::new(sa, text, 5, 10); // Min length 5
        
        let short_pattern = b"the"; // Length 3, below minimum
        let matches = matcher.find_all_matches(short_pattern, 10).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_early_termination() {
        let (sa, text) = create_test_setup();
        let mut matcher = PatternMatcher::with_config(
            sa,
            text,
            MatcherConfig {
                early_termination_quality: 0.5, // Very low threshold for testing
                ..Default::default()
            },
        );
        
        let input = b"the quick brown fox";
        let result = matcher.find_longest_match_suffix_array(input, 0, 20).unwrap();
        
        // Should find a match and potentially terminate early
        assert!(result.is_some());
    }
}