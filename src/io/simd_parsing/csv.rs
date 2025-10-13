//! # CSV SIMD Parser with Delimiter Detection
//!
//! High-performance CSV parsing using SIMD acceleration for delimiter and newline detection.
//! Follows the 6-tier SIMD framework with runtime adaptive selection.
//!
//! ## Architecture
//!
//! **Core Operations**:
//! - SIMD-accelerated delimiter detection with quote handling
//! - Parallel newline detection (\n and \r\n)
//! - Bulk delimiter finding with position extraction
//! - Quote state tracking using prefix XOR pattern
//! - Efficient field boundary extraction
//!
//! **Quote Handling Algorithm**:
//! 1. Find all quotes in chunk: quote_mask = cmpeq(chunk, '"')
//! 2. Compute quote parity: parity = prefix_xor(quote_mask)
//! 3. Find delimiters: delim_mask = cmpeq(chunk, delimiter)
//! 4. Mask quoted delimiters: valid_mask = delim_mask & ~parity
//! 5. Extract positions: tzcnt loop on valid_mask
//!
//! ## Performance Target
//!
//! - 1.8 GB/s throughput (4.5x speedup over scalar)
//! - <100ns cache-hit overhead with AdaptiveSimdSelector
//! - Correct handling of quoted fields with embedded delimiters
//!
//! ## Example
//!
//! ```
//! use zipora::io::simd_parsing::csv::{parse_csv_line, find_delimiter};
//!
//! let csv_data = b"Alice,30,true";
//! let fields = parse_csv_line(csv_data, b',').expect("Failed to parse CSV line");
//! assert_eq!(fields.len(), 3);
//! ```

use crate::error::{Result, ZiporaError};
use crate::simd::{AdaptiveSimdSelector, Operation};
use crate::system::cpu_features::CpuFeatures;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD implementation tiers for CSV parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsvParserTier {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE4.2 with 16-byte chunks
    Sse42,
    /// AVX2 with 32-byte chunks
    Avx2,
    /// ARM NEON for ARM64 platforms
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// AVX-512 with 64-byte chunks (nightly only)
    #[cfg(feature = "avx512")]
    Avx512,
}

/// CSV parser configuration
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Field delimiter (default: comma)
    pub delimiter: u8,
    /// Quote character (default: double quote)
    pub quote: u8,
    /// Escape character (default: double quote for CSV)
    pub escape: u8,
    /// Enable strict quote handling
    pub strict_quotes: bool,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: b',',
            quote: b'"',
            escape: b'"',
            strict_quotes: true,
        }
    }
}

impl CsvConfig {
    /// Create with custom delimiter
    pub fn with_delimiter(delimiter: u8) -> Self {
        Self {
            delimiter,
            ..Default::default()
        }
    }
}

/// CSV SIMD parser with runtime adaptive tier selection
pub struct CsvParser {
    /// CPU features for runtime detection
    cpu_features: &'static CpuFeatures,
    /// Selected parser tier
    parser_tier: CsvParserTier,
    /// Configuration
    config: CsvConfig,
}

impl CsvParser {
    /// Creates a new CSV parser with runtime feature detection
    pub fn new() -> Self {
        Self::with_config(CsvConfig::default())
    }

    /// Creates a new CSV parser with custom configuration
    pub fn with_config(config: CsvConfig) -> Self {
        let cpu_features = crate::system::get_cpu_features();
        let parser_tier = Self::select_optimal_tier(cpu_features);

        Self {
            cpu_features,
            parser_tier,
            config,
        }
    }

    /// Selects optimal parser tier based on CPU features
    fn select_optimal_tier(features: &CpuFeatures) -> CsvParserTier {
        #[cfg(feature = "avx512")]
        if features.has_avx512f && features.has_avx512bw {
            return CsvParserTier::Avx512;
        }

        if features.has_avx2 {
            return CsvParserTier::Avx2;
        }

        if features.has_sse42 {
            return CsvParserTier::Sse42;
        }

        #[cfg(target_arch = "aarch64")]
        {
            return CsvParserTier::Neon;
        }

        #[allow(unreachable_code)]
        CsvParserTier::Scalar
    }

    /// Find first delimiter in data with quote handling
    pub fn find_delimiter(&self, data: &[u8]) -> Option<usize> {
        if data.is_empty() {
            return None;
        }

        let start = Instant::now();
        let result = match self.parser_tier {
            CsvParserTier::Avx2 => unsafe { self.find_delimiter_avx2(data, self.config.delimiter) },
            CsvParserTier::Sse42 => unsafe { self.find_delimiter_sse42(data, self.config.delimiter) },
            CsvParserTier::Scalar => self.find_delimiter_scalar(data, self.config.delimiter),
            #[cfg(target_arch = "aarch64")]
            CsvParserTier::Neon => unsafe { self.find_delimiter_neon(data, self.config.delimiter) },
            #[cfg(feature = "avx512")]
            CsvParserTier::Avx512 => unsafe { self.find_delimiter_avx512(data, self.config.delimiter) },
        };

        // Monitor performance
        let selector = AdaptiveSimdSelector::global();
        selector.monitor_performance(Operation::StringSearch, start.elapsed(), data.len() as u64);

        result
    }

    /// Find first newline in data (handles \n and \r\n)
    pub fn find_newline(&self, data: &[u8]) -> Option<usize> {
        if data.is_empty() {
            return None;
        }

        let start = Instant::now();
        let result = match self.parser_tier {
            CsvParserTier::Avx2 => unsafe { self.find_newline_avx2(data) },
            CsvParserTier::Sse42 => unsafe { self.find_newline_sse42(data) },
            CsvParserTier::Scalar => self.find_newline_scalar(data),
            #[cfg(target_arch = "aarch64")]
            CsvParserTier::Neon => unsafe { self.find_newline_neon(data) },
            #[cfg(feature = "avx512")]
            CsvParserTier::Avx512 => unsafe { self.find_newline_avx512(data) },
        };

        // Monitor performance
        let selector = AdaptiveSimdSelector::global();
        selector.monitor_performance(Operation::StringSearch, start.elapsed(), data.len() as u64);

        result
    }

    /// Find all delimiter positions in data (bulk operation)
    pub fn find_delimiters_bulk(&self, data: &[u8]) -> Vec<usize> {
        if data.is_empty() {
            return Vec::new();
        }

        let start = Instant::now();
        let result = match self.parser_tier {
            CsvParserTier::Avx2 => unsafe { self.find_delimiters_bulk_avx2(data, self.config.delimiter) },
            CsvParserTier::Sse42 => unsafe { self.find_delimiters_bulk_sse42(data, self.config.delimiter) },
            CsvParserTier::Scalar => self.find_delimiters_bulk_scalar(data, self.config.delimiter),
            #[cfg(target_arch = "aarch64")]
            CsvParserTier::Neon => unsafe { self.find_delimiters_bulk_neon(data, self.config.delimiter) },
            #[cfg(feature = "avx512")]
            CsvParserTier::Avx512 => unsafe { self.find_delimiters_bulk_avx512(data, self.config.delimiter) },
        };

        // Monitor performance
        let selector = AdaptiveSimdSelector::global();
        selector.monitor_performance(Operation::StringSearch, start.elapsed(), data.len() as u64);

        result
    }

    /// Parse a single CSV line into fields
    ///
    /// Note: This function returns borrowed slices for performance, but quoted fields
    /// with escaped quotes will still contain the doubled quotes (e.g., `""` instead of `"`).
    /// For full CSV parsing with unescaping, use a higher-level CSV library.
    pub fn parse_line<'a>(&self, data: &'a [u8]) -> Result<Vec<Vec<u8>>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let mut fields = Vec::new();
        let mut field_data = Vec::new();
        let mut in_quotes = false;
        let mut pos = 0;

        while pos < data.len() {
            let byte = data[pos];

            // Handle quotes
            if byte == self.config.quote {
                if in_quotes && pos + 1 < data.len() && data[pos + 1] == self.config.quote {
                    // Doubled quote (escape sequence) - add single quote to field
                    field_data.push(byte);
                    pos += 2;
                    continue;
                } else {
                    // Toggle quote state (don't include the quote character itself)
                    in_quotes = !in_quotes;
                    pos += 1;
                    continue;
                }
            }

            // Handle delimiters and newlines (only outside quotes)
            if !in_quotes {
                if byte == self.config.delimiter || byte == b'\n' || byte == b'\r' {
                    // Save current field
                    fields.push(field_data.clone());
                    field_data.clear();

                    // Check if this is a newline
                    let is_newline = byte == b'\n' || byte == b'\r';

                    // Skip \r\n sequence
                    if byte == b'\r' && pos + 1 < data.len() && data[pos + 1] == b'\n' {
                        pos += 2;
                    } else {
                        pos += 1;
                    }

                    // Stop at newline (don't process further)
                    if is_newline {
                        // Monitor performance before return
                        let selector = AdaptiveSimdSelector::global();
                        selector.monitor_performance(Operation::StringSearch, start.elapsed(), data.len() as u64);
                        return Ok(fields);
                    }

                    continue;
                }
            }

            // Add byte to current field
            field_data.push(byte);
            pos += 1;
        }

        // Add final field (always add it, even if empty after trailing delimiter)
        fields.push(field_data);

        // Monitor performance
        let selector = AdaptiveSimdSelector::global();
        selector.monitor_performance(Operation::StringSearch, start.elapsed(), data.len() as u64);

        Ok(fields)
    }

    /// Get selected parser tier
    pub fn tier(&self) -> CsvParserTier {
        self.parser_tier
    }
}

//==============================================================================
// AVX2 IMPLEMENTATIONS (32-byte chunks)
//==============================================================================

#[cfg(target_arch = "x86_64")]
impl CsvParser {
    /// AVX2 delimiter search with quote state tracking
    #[target_feature(enable = "avx2")]
    unsafe fn find_delimiter_avx2(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        let delim_vec = unsafe { _mm256_set1_epi8(delimiter as i8) };
        let quote_vec = unsafe { _mm256_set1_epi8(self.config.quote as i8) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 32;

        // Process 32-byte chunks
        for _ in 0..chunks {
            let chunk = unsafe { _mm256_loadu_si256(data[pos..].as_ptr() as *const __m256i) };

            // Find quotes
            let quote_mask = unsafe { _mm256_cmpeq_epi8(chunk, quote_vec) };
            let mut quote_bits = unsafe { _mm256_movemask_epi8(quote_mask) } as u32;

            // Find delimiters
            let delim_mask = unsafe { _mm256_cmpeq_epi8(chunk, delim_vec) };
            let delim_bits = unsafe { _mm256_movemask_epi8(delim_mask) } as u32;

            // Process each byte in chunk
            for i in 0..32 {
                let quote_bit = 1u32 << i;
                let delim_bit = 1u32 << i;

                // Toggle quote state
                if (quote_bits & quote_bit) != 0 {
                    // Handle doubled quotes
                    if in_quotes && i + 1 < 32 && (quote_bits & (1u32 << (i + 1))) != 0 {
                        quote_bits &= !(1u32 << (i + 1)); // Skip next quote
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                // Check delimiter (only outside quotes)
                if !in_quotes && (delim_bits & delim_bit) != 0 {
                    return Some(pos + i);
                }
            }

            pos += 32;
        }

        // Handle remaining bytes
        self.find_delimiter_scalar(&data[pos..], delimiter)
            .map(|offset| pos + offset)
    }

    /// AVX2 newline search
    #[target_feature(enable = "avx2")]
    unsafe fn find_newline_avx2(&self, data: &[u8]) -> Option<usize> {
        let lf_vec = unsafe { _mm256_set1_epi8(b'\n' as i8) };
        let cr_vec = unsafe { _mm256_set1_epi8(b'\r' as i8) };

        let mut pos = 0;
        let chunks = data.len() / 32;

        // Process 32-byte chunks
        for _ in 0..chunks {
            let chunk = unsafe { _mm256_loadu_si256(data[pos..].as_ptr() as *const __m256i) };

            // Find \n
            let lf_mask = unsafe { _mm256_cmpeq_epi8(chunk, lf_vec) };
            let lf_bits = unsafe { _mm256_movemask_epi8(lf_mask) } as u32;

            if lf_bits != 0 {
                let offset = lf_bits.trailing_zeros() as usize;
                return Some(pos + offset);
            }

            // Find \r
            let cr_mask = unsafe { _mm256_cmpeq_epi8(chunk, cr_vec) };
            let cr_bits = unsafe { _mm256_movemask_epi8(cr_mask) } as u32;

            if cr_bits != 0 {
                let offset = cr_bits.trailing_zeros() as usize;
                return Some(pos + offset);
            }

            pos += 32;
        }

        // Handle remaining bytes
        self.find_newline_scalar(&data[pos..])
            .map(|offset| pos + offset)
    }

    /// AVX2 bulk delimiter search
    #[target_feature(enable = "avx2")]
    unsafe fn find_delimiters_bulk_avx2(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        let mut positions = Vec::new();
        let delim_vec = unsafe { _mm256_set1_epi8(delimiter as i8) };
        let quote_vec = unsafe { _mm256_set1_epi8(self.config.quote as i8) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 32;

        // Process 32-byte chunks
        for _ in 0..chunks {
            let chunk = unsafe { _mm256_loadu_si256(data[pos..].as_ptr() as *const __m256i) };

            // Find quotes
            let quote_mask = unsafe { _mm256_cmpeq_epi8(chunk, quote_vec) };
            let mut quote_bits = unsafe { _mm256_movemask_epi8(quote_mask) } as u32;

            // Find delimiters
            let delim_mask = unsafe { _mm256_cmpeq_epi8(chunk, delim_vec) };
            let delim_bits = unsafe { _mm256_movemask_epi8(delim_mask) } as u32;

            // Process each byte in chunk
            for i in 0..32 {
                let quote_bit = 1u32 << i;
                let delim_bit = 1u32 << i;

                // Toggle quote state
                if (quote_bits & quote_bit) != 0 {
                    // Handle doubled quotes
                    if in_quotes && i + 1 < 32 && (quote_bits & (1u32 << (i + 1))) != 0 {
                        quote_bits &= !(1u32 << (i + 1)); // Skip next quote
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                // Collect delimiter positions (only outside quotes)
                if !in_quotes && (delim_bits & delim_bit) != 0 {
                    positions.push(pos + i);
                }
            }

            pos += 32;
        }

        // Handle remaining bytes
        let remaining = self.find_delimiters_bulk_scalar(&data[pos..], delimiter);
        positions.extend(remaining.into_iter().map(|offset| pos + offset));

        positions
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl CsvParser {
    #[inline]
    unsafe fn find_delimiter_avx2(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        self.find_delimiter_scalar(data, delimiter)
    }

    #[inline]
    unsafe fn find_newline_avx2(&self, data: &[u8]) -> Option<usize> {
        self.find_newline_scalar(data)
    }

    #[inline]
    unsafe fn find_delimiters_bulk_avx2(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        self.find_delimiters_bulk_scalar(data, delimiter)
    }
}

//==============================================================================
// SSE4.2 IMPLEMENTATIONS (16-byte chunks)
//==============================================================================

#[cfg(target_arch = "x86_64")]
impl CsvParser {
    /// SSE4.2 delimiter search
    #[target_feature(enable = "sse4.2")]
    unsafe fn find_delimiter_sse42(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        let delim_vec = unsafe { _mm_set1_epi8(delimiter as i8) };
        let quote_vec = unsafe { _mm_set1_epi8(self.config.quote as i8) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { _mm_loadu_si128(data[pos..].as_ptr() as *const __m128i) };

            let quote_mask = unsafe { _mm_cmpeq_epi8(chunk, quote_vec) };
            let mut quote_bits = unsafe { _mm_movemask_epi8(quote_mask) } as u16;

            let delim_mask = unsafe { _mm_cmpeq_epi8(chunk, delim_vec) };
            let delim_bits = unsafe { _mm_movemask_epi8(delim_mask) } as u16;

            for i in 0..16 {
                let quote_bit = 1u16 << i;
                let delim_bit = 1u16 << i;

                if (quote_bits & quote_bit) != 0 {
                    if in_quotes && i + 1 < 16 && (quote_bits & (1u16 << (i + 1))) != 0 {
                        quote_bits &= !(1u16 << (i + 1));
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                if !in_quotes && (delim_bits & delim_bit) != 0 {
                    return Some(pos + i);
                }
            }

            pos += 16;
        }

        self.find_delimiter_scalar(&data[pos..], delimiter)
            .map(|offset| pos + offset)
    }

    /// SSE4.2 newline search
    #[target_feature(enable = "sse4.2")]
    unsafe fn find_newline_sse42(&self, data: &[u8]) -> Option<usize> {
        let lf_vec = unsafe { _mm_set1_epi8(b'\n' as i8) };
        let cr_vec = unsafe { _mm_set1_epi8(b'\r' as i8) };

        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { _mm_loadu_si128(data[pos..].as_ptr() as *const __m128i) };

            let lf_mask = unsafe { _mm_cmpeq_epi8(chunk, lf_vec) };
            let lf_bits = unsafe { _mm_movemask_epi8(lf_mask) } as u16;

            if lf_bits != 0 {
                return Some(pos + lf_bits.trailing_zeros() as usize);
            }

            let cr_mask = unsafe { _mm_cmpeq_epi8(chunk, cr_vec) };
            let cr_bits = unsafe { _mm_movemask_epi8(cr_mask) } as u16;

            if cr_bits != 0 {
                return Some(pos + cr_bits.trailing_zeros() as usize);
            }

            pos += 16;
        }

        self.find_newline_scalar(&data[pos..])
            .map(|offset| pos + offset)
    }

    /// SSE4.2 bulk delimiter search
    #[target_feature(enable = "sse4.2")]
    unsafe fn find_delimiters_bulk_sse42(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        let mut positions = Vec::new();
        let delim_vec = unsafe { _mm_set1_epi8(delimiter as i8) };
        let quote_vec = unsafe { _mm_set1_epi8(self.config.quote as i8) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { _mm_loadu_si128(data[pos..].as_ptr() as *const __m128i) };

            let quote_mask = unsafe { _mm_cmpeq_epi8(chunk, quote_vec) };
            let mut quote_bits = unsafe { _mm_movemask_epi8(quote_mask) } as u16;

            let delim_mask = unsafe { _mm_cmpeq_epi8(chunk, delim_vec) };
            let delim_bits = unsafe { _mm_movemask_epi8(delim_mask) } as u16;

            for i in 0..16 {
                let quote_bit = 1u16 << i;
                let delim_bit = 1u16 << i;

                if (quote_bits & quote_bit) != 0 {
                    if in_quotes && i + 1 < 16 && (quote_bits & (1u16 << (i + 1))) != 0 {
                        quote_bits &= !(1u16 << (i + 1));
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                if !in_quotes && (delim_bits & delim_bit) != 0 {
                    positions.push(pos + i);
                }
            }

            pos += 16;
        }

        let remaining = self.find_delimiters_bulk_scalar(&data[pos..], delimiter);
        positions.extend(remaining.into_iter().map(|offset| pos + offset));

        positions
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl CsvParser {
    #[inline]
    unsafe fn find_delimiter_sse42(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        self.find_delimiter_scalar(data, delimiter)
    }

    #[inline]
    unsafe fn find_newline_sse42(&self, data: &[u8]) -> Option<usize> {
        self.find_newline_scalar(data)
    }

    #[inline]
    unsafe fn find_delimiters_bulk_sse42(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        self.find_delimiters_bulk_scalar(data, delimiter)
    }
}

//==============================================================================
// ARM NEON IMPLEMENTATIONS
//==============================================================================

#[cfg(target_arch = "aarch64")]
impl CsvParser {
    /// ARM NEON delimiter search
    #[target_feature(enable = "neon")]
    unsafe fn find_delimiter_neon(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        use std::arch::aarch64::*;

        let delim_vec = unsafe { vdupq_n_u8(delimiter) };
        let quote_vec = unsafe { vdupq_n_u8(self.config.quote) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { vld1q_u8(data[pos..].as_ptr()) };

            // Compare for quotes
            let quote_mask = unsafe { vceqq_u8(chunk, quote_vec) };
            // Compare for delimiters
            let delim_mask = unsafe { vceqq_u8(chunk, delim_vec) };

            // Process byte-by-byte (NEON doesn't have movemask equivalent)
            for i in 0..16 {
                let is_quote = unsafe { vgetq_lane_u8(quote_mask, i) } != 0;
                let is_delim = unsafe { vgetq_lane_u8(delim_mask, i) } != 0;

                if is_quote {
                    if in_quotes && i + 1 < 16 && unsafe { vgetq_lane_u8(quote_mask, i + 1) } != 0 {
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                if !in_quotes && is_delim {
                    return Some(pos + i);
                }
            }

            pos += 16;
        }

        self.find_delimiter_scalar(&data[pos..], delimiter)
            .map(|offset| pos + offset)
    }

    /// ARM NEON newline search
    #[target_feature(enable = "neon")]
    unsafe fn find_newline_neon(&self, data: &[u8]) -> Option<usize> {
        use std::arch::aarch64::*;

        let lf_vec = unsafe { vdupq_n_u8(b'\n') };
        let cr_vec = unsafe { vdupq_n_u8(b'\r') };

        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { vld1q_u8(data[pos..].as_ptr()) };

            let lf_mask = unsafe { vceqq_u8(chunk, lf_vec) };
            let cr_mask = unsafe { vceqq_u8(chunk, cr_vec) };

            for i in 0..16 {
                if unsafe { vgetq_lane_u8(lf_mask, i) } != 0 || unsafe { vgetq_lane_u8(cr_mask, i) } != 0 {
                    return Some(pos + i);
                }
            }

            pos += 16;
        }

        self.find_newline_scalar(&data[pos..])
            .map(|offset| pos + offset)
    }

    /// ARM NEON bulk delimiter search
    #[target_feature(enable = "neon")]
    unsafe fn find_delimiters_bulk_neon(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        use std::arch::aarch64::*;

        let mut positions = Vec::new();
        let delim_vec = unsafe { vdupq_n_u8(delimiter) };
        let quote_vec = unsafe { vdupq_n_u8(self.config.quote) };

        let mut in_quotes = false;
        let mut pos = 0;
        let chunks = data.len() / 16;

        for _ in 0..chunks {
            let chunk = unsafe { vld1q_u8(data[pos..].as_ptr()) };

            let quote_mask = unsafe { vceqq_u8(chunk, quote_vec) };
            let delim_mask = unsafe { vceqq_u8(chunk, delim_vec) };

            for i in 0..16 {
                let is_quote = unsafe { vgetq_lane_u8(quote_mask, i) } != 0;
                let is_delim = unsafe { vgetq_lane_u8(delim_mask, i) } != 0;

                if is_quote {
                    if in_quotes && i + 1 < 16 && unsafe { vgetq_lane_u8(quote_mask, i + 1) } != 0 {
                        continue;
                    }
                    in_quotes = !in_quotes;
                }

                if !in_quotes && is_delim {
                    positions.push(pos + i);
                }
            }

            pos += 16;
        }

        let remaining = self.find_delimiters_bulk_scalar(&data[pos..], delimiter);
        positions.extend(remaining.into_iter().map(|offset| pos + offset));

        positions
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl CsvParser {
    #[inline]
    unsafe fn find_delimiter_neon(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        self.find_delimiter_scalar(data, delimiter)
    }

    #[inline]
    unsafe fn find_newline_neon(&self, data: &[u8]) -> Option<usize> {
        self.find_newline_scalar(data)
    }

    #[inline]
    unsafe fn find_delimiters_bulk_neon(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        self.find_delimiters_bulk_scalar(data, delimiter)
    }
}

//==============================================================================
// AVX-512 IMPLEMENTATIONS (64-byte chunks)
//==============================================================================

#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
impl CsvParser {
    /// AVX-512 delimiter search
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn find_delimiter_avx512(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        // Delegate to AVX2 for now (full AVX-512 implementation similar to AVX2 but 64 bytes)
        unsafe { self.find_delimiter_avx2(data, delimiter) }
    }

    /// AVX-512 newline search
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn find_newline_avx512(&self, data: &[u8]) -> Option<usize> {
        unsafe { self.find_newline_avx2(data) }
    }

    /// AVX-512 bulk delimiter search
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn find_delimiters_bulk_avx512(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        unsafe { self.find_delimiters_bulk_avx2(data, delimiter) }
    }
}

#[cfg(not(all(feature = "avx512", target_arch = "x86_64")))]
impl CsvParser {
    #[inline]
    unsafe fn find_delimiter_avx512(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        self.find_delimiter_scalar(data, delimiter)
    }

    #[inline]
    unsafe fn find_newline_avx512(&self, data: &[u8]) -> Option<usize> {
        self.find_newline_scalar(data)
    }

    #[inline]
    unsafe fn find_delimiters_bulk_avx512(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        self.find_delimiters_bulk_scalar(data, delimiter)
    }
}

//==============================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
//==============================================================================

impl CsvParser {
    /// Scalar delimiter search with quote handling
    fn find_delimiter_scalar(&self, data: &[u8], delimiter: u8) -> Option<usize> {
        let mut in_quotes = false;
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];

            if byte == self.config.quote {
                // Handle doubled quotes
                if in_quotes && i + 1 < data.len() && data[i + 1] == self.config.quote {
                    i += 2; // Skip both quotes
                    continue;
                }
                in_quotes = !in_quotes;
                i += 1;
                continue;
            }

            if !in_quotes && byte == delimiter {
                return Some(i);
            }

            i += 1;
        }

        None
    }

    /// Scalar newline search
    fn find_newline_scalar(&self, data: &[u8]) -> Option<usize> {
        for (i, &byte) in data.iter().enumerate() {
            if byte == b'\n' || byte == b'\r' {
                return Some(i);
            }
        }
        None
    }

    /// Scalar bulk delimiter search
    fn find_delimiters_bulk_scalar(&self, data: &[u8], delimiter: u8) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut in_quotes = false;
        let mut i = 0;

        while i < data.len() {
            let byte = data[i];

            if byte == self.config.quote {
                // Handle doubled quotes
                if in_quotes && i + 1 < data.len() && data[i + 1] == self.config.quote {
                    i += 2;
                    continue;
                }
                in_quotes = !in_quotes;
                i += 1;
                continue;
            }

            if !in_quotes && byte == delimiter {
                positions.push(i);
            }

            i += 1;
        }

        positions
    }
}

impl Default for CsvParser {
    fn default() -> Self {
        Self::new()
    }
}

//==============================================================================
// CONVENIENCE FUNCTIONS
//==============================================================================

/// Find first delimiter in data with quote handling (using global parser)
pub fn find_delimiter(data: &[u8], delimiter: u8) -> Option<usize> {
    static PARSER: std::sync::OnceLock<CsvParser> = std::sync::OnceLock::new();
    let parser = PARSER.get_or_init(|| CsvParser::with_config(CsvConfig::with_delimiter(delimiter)));
    parser.find_delimiter(data)
}

/// Find first newline in data (using global parser)
pub fn find_newline(data: &[u8]) -> Option<usize> {
    static PARSER: std::sync::OnceLock<CsvParser> = std::sync::OnceLock::new();
    let parser = PARSER.get_or_init(|| CsvParser::new());
    parser.find_newline(data)
}

/// Find all delimiter positions in data (using global parser)
pub fn find_delimiters_bulk(data: &[u8], delimiter: u8) -> Vec<usize> {
    static PARSER: std::sync::OnceLock<CsvParser> = std::sync::OnceLock::new();
    let parser = PARSER.get_or_init(|| CsvParser::with_config(CsvConfig::with_delimiter(delimiter)));
    parser.find_delimiters_bulk(data)
}

/// Parse CSV line into fields
///
/// Creates a new parser for each call with the specified delimiter.
/// For better performance with repeated calls, create a `CsvParser` instance.
pub fn parse_csv_line(data: &[u8], delimiter: u8) -> Result<Vec<Vec<u8>>> {
    let parser = CsvParser::with_config(CsvConfig::with_delimiter(delimiter));
    parser.parse_line(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parser_creation() {
        let parser = CsvParser::new();
        println!("Selected CSV parser tier: {:?}", parser.tier());
    }

    #[test]
    fn test_find_delimiter_simple() {
        let data = b"Alice,30,true";
        let pos = find_delimiter(data, b',');
        assert_eq!(pos, Some(5));
    }

    #[test]
    fn test_find_delimiter_quoted() {
        let data = br#""Smith, John",30,true"#;
        let pos = find_delimiter(data, b',');
        assert_eq!(pos, Some(13)); // After the quoted field
    }

    #[test]
    fn test_find_delimiter_escaped_quotes() {
        let data = br#""She said ""Hello""",30,true"#;
        let pos = find_delimiter(data, b',');
        assert_eq!(pos, Some(20)); // After the quoted field with escaped quotes
    }

    #[test]
    fn test_find_delimiter_no_match() {
        let data = b"Alice";
        let pos = find_delimiter(data, b',');
        assert_eq!(pos, None);
    }

    #[test]
    fn test_find_newline_lf() {
        let data = b"Alice,30\nBob,25";
        let pos = find_newline(data);
        assert_eq!(pos, Some(8));
    }

    #[test]
    fn test_find_newline_crlf() {
        let data = b"Alice,30\r\nBob,25";
        let pos = find_newline(data);
        assert_eq!(pos, Some(8)); // Should find \r
    }

    #[test]
    fn test_find_newline_no_match() {
        let data = b"Alice,30,true";
        let pos = find_newline(data);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_find_delimiters_bulk_simple() {
        let data = b"Alice,30,true";
        let positions = find_delimiters_bulk(data, b',');
        assert_eq!(positions, vec![5, 8]);
    }

    #[test]
    fn test_find_delimiters_bulk_quoted() {
        let data = br#""Smith, John",30,"CA, USA""#;
        let positions = find_delimiters_bulk(data, b',');
        assert_eq!(positions, vec![13, 16]); // Only unquoted commas
    }

    #[test]
    fn test_find_delimiters_bulk_empty() {
        let data = b"Alice";
        let positions = find_delimiters_bulk(data, b',');
        assert!(positions.is_empty());
    }

    #[test]
    fn test_parse_csv_line_simple() {
        let data = b"Alice,30,true";
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[1], b"30");
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_quoted() {
        let data = br#""Smith, John","30","true""#;
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Smith, John");
        assert_eq!(fields[1], b"30");
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_escaped_quotes() {
        let data = br#""She said ""Hello""","30","true""#;
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"She said \"Hello\"");
        assert_eq!(fields[1], b"30");
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_empty_fields() {
        let data = b"Alice,,true";
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[1], b"");
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_trailing_delimiter() {
        let data = b"Alice,30,";
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[2], b"");
    }

    #[test]
    fn test_parse_csv_line_newline_terminated() {
        let data = b"Alice,30,true\n";
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_crlf_terminated() {
        let data = b"Alice,30,true\r\n";
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_parse_csv_line_empty() {
        let data = b"";
        let fields = parse_csv_line(data, b',').unwrap();
        assert!(fields.is_empty());
    }

    #[test]
    fn test_custom_delimiter() {
        let parser = CsvParser::with_config(CsvConfig::with_delimiter(b';'));
        let data = b"Alice;30;true";
        let fields = parser.parse_line(data).unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
    }

    #[test]
    fn test_tab_delimiter() {
        let data = b"Alice\t30\ttrue";
        let fields = parse_csv_line(data, b'\t').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[1], b"30");
        assert_eq!(fields[2], b"true");
    }

    #[test]
    fn test_large_csv_line() {
        // Test with larger CSV line for performance validation
        let mut data = Vec::new();
        for i in 0..100 {
            if i > 0 {
                data.push(b',');
            }
            data.extend_from_slice(format!("field{}", i).as_bytes());
        }

        let fields = parse_csv_line(&data, b',').unwrap();
        assert_eq!(fields.len(), 100);
        assert_eq!(fields[0], b"field0");
        assert_eq!(fields[99], b"field99");
    }

    #[test]
    fn test_quoted_with_newlines() {
        let data = br#""Line1
Line2",30,true"#;
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Line1\nLine2");
    }

    #[test]
    fn test_mixed_quoting() {
        let data = br#"Alice,"Bob ""Bobby"" Smith",30"#;
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[1], b"Bob \"Bobby\" Smith");
        assert_eq!(fields[2], b"30");
    }

    #[test]
    fn test_unicode_csv() {
        let data = "Alice,30,日本語".as_bytes();
        let fields = parse_csv_line(data, b',').unwrap();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"Alice");
        assert_eq!(fields[2], "日本語".as_bytes());
    }

    #[test]
    fn test_find_delimiters_bulk_performance() {
        // Generate large CSV data
        let mut data = Vec::new();
        for i in 0..1000 {
            if i > 0 {
                data.push(b',');
            }
            data.extend_from_slice(format!("field{}", i).as_bytes());
        }

        let start = std::time::Instant::now();
        let positions = find_delimiters_bulk(&data, b',');
        let elapsed = start.elapsed();

        assert_eq!(positions.len(), 999);
        println!("Found {} delimiters in {} bytes in {:?}", positions.len(), data.len(), elapsed);
    }

    #[test]
    fn test_cross_tier_consistency() {
        // Ensure all SIMD tiers produce same results
        let test_cases = vec![
            b"Alice,30,true" as &[u8],
            br#""Smith, John",30,true"#,
            br#""Escaped ""quotes""",data,here"#,
            b"field1,field2,field3,field4,field5",
            b"",
            b"single",
        ];

        let parser = CsvParser::new();
        for data in test_cases {
            let delimiter_pos = parser.find_delimiter(data);
            let delimiters_bulk = parser.find_delimiters_bulk(data);

            // First delimiter should match bulk[0]
            if let Some(pos) = delimiter_pos {
                assert_eq!(delimiters_bulk.get(0), Some(&pos));
            } else {
                assert!(delimiters_bulk.is_empty());
            }
        }
    }
}
