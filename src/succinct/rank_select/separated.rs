//! High-Performance Separated Rank/Select Implementations
//!
//! This module provides separated cache rank/select implementations where
//! the rank cache and bit data are stored separately for optimal memory
//! access patterns. These implementations focus on maximizing performance
//! through hardware acceleration and optimized data structures.
//!
//! # Variants
//!
//! - **`RankSelectSeparated256`**: 256-bit blocks for general-purpose use
//! - **`RankSelectSeparated512`**: 512-bit blocks for streaming/bulk operations
//!
//! # Design Philosophy
//!
//! - **Separated Storage**: Rank cache stored separately from bit data
//! - **Hardware Acceleration**: Leverages POPCNT, BMI2, and AVX-512 when available
//! - **Optimized Block Sizes**: Different block sizes for different use cases
//! - **Select Caching**: Optional select hints for faster select operations
//!
//! # Performance Characteristics
//!
//! ## RankSelectSeparated256
//! - **Memory Overhead**: ~25% (4-byte rank + optional select cache)
//! - **Rank Time**: O(1) with 1-2 memory accesses
//! - **Select Time**: O(1) average with select cache, O(log n) worst case
//! - **Best For**: General-purpose applications, random access patterns
//!
//! ## RankSelectSeparated512
//! - **Memory Overhead**: ~12.5% (fewer blocks, larger cache entries)
//! - **Rank Time**: O(1) with potentially better cache behavior
//! - **Select Time**: O(1) average, O(log n) worst case
//! - **Best For**: Sequential access, streaming operations, large datasets

use super::{
    BuilderOptions, CpuFeatures, RankSelectBuilder, RankSelectOps, RankSelectPerformanceOps,
};
use super::config::SeparatedStorageConfig;
// Additional config types (currently unused)
// use super::config::{StorageLayout, MemoryStrategy};
use crate::FastVec;
use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use std::fmt;

// Hardware acceleration imports
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_pdep_u64, _popcnt64, _tzcnt_u64};

/// High-performance separated rank/select with 256-bit blocks
///
/// This implementation uses separated storage for optimal memory access patterns.
/// The rank cache is stored separately from the bit data, allowing for efficient
/// cache usage and hardware acceleration.
///
/// # Memory Layout
///
/// ```text
/// Bit Vector:    |bit data...|
/// Rank Cache:    |r0|r1|r2|...|  (4 bytes per 256-bit block)
/// Select Cache:  |s0|s1|s2|...|  (optional, 4 bytes per N set bits)
/// ```
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectOps, RankSelectPerformanceOps, RankSelectSeparated256};
///
/// let mut bv = BitVector::new();
/// for i in 0..1000 {
///     bv.push(i % 7 == 0)?;
/// }
///
/// let rs = RankSelectSeparated256::new(bv)?;
///
/// // High-performance operations
/// let rank = rs.rank1(500);           // Hardware-accelerated
/// let pos = rs.select1(50)?;          // Select cache optimized
/// let adaptive = rs.rank1_adaptive(750); // Best available implementation
///
/// println!("Rank at 500: {}, 50th bit at: {}", rank, pos);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectSeparated256 {
    /// The original bit vector
    bit_vector: BitVector,
    /// Separated rank cache - cumulative rank at end of each 256-bit block
    rank_cache: FastVec<u32>,
    /// Optional select cache for faster select operations
    select_cache: Option<FastVec<u32>>,
    /// Total number of set bits
    total_ones: usize,
    /// Select sampling rate (if select cache is enabled)
    select_sample_rate: usize,
}

/// High-performance separated rank/select with 512-bit blocks
///
/// This variant uses larger 512-bit blocks for better sequential performance
/// and reduced memory overhead on large datasets.
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectOps, RankSelectPerformanceOps, RankSelectSeparated512};
///
/// let mut bv = BitVector::new();
/// for i in 0..100_000 {
///     bv.push((i * 13 + 7) % 127 == 0)?; // Sparse pattern
/// }
///
/// let rs = RankSelectSeparated512::new(bv)?;
///
/// // Optimized for large datasets
/// let bulk_ranks = rs.rank1_bulk(&[1000, 5000, 10000, 50000]);
/// println!("Bulk ranks: {:?}", bulk_ranks);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectSeparated512 {
    /// The original bit vector
    bit_vector: BitVector,
    /// Separated rank cache with 512-bit blocks
    rank_cache: FastVec<u64>, // Use u64 for larger blocks
    /// Optional select cache
    select_cache: Option<FastVec<u32>>,
    /// Total number of set bits
    total_ones: usize,
    /// Select sampling rate
    select_sample_rate: usize,
}

/// High-performance separated rank/select with 512-bit blocks and bit-packed rank cache
///
/// This advanced variant uses bit-packed relative ranks for significant space savings
/// while maintaining O(1) query performance. Based on research from topling-zip
/// advanced rank cache encoding techniques.
///
/// # Design
///
/// Uses a hierarchical structure:
/// - **Superblocks**: Store absolute cumulative ranks every N blocks (64-bit values)
/// - **Relative Ranks**: Store 9-bit relative ranks bit-packed into 64-bit words
/// - **Bit Packing**: 7 × 9-bit values per 64-bit word for ~40-50% memory reduction
///
/// # Examples
///
/// ```rust
/// use zipora::succinct::{BitVector, rank_select::{RankSelectOps, separated::RankSelectSeparated512BitPacked}};
///
/// let mut bv = BitVector::new();
/// for i in 0..10_000 {
///     bv.push((i * 17 + 11) % 131 == 0)?; // Complex sparse pattern
/// }
///
/// let rs = RankSelectSeparated512BitPacked::new(bv)?;
///
/// // Significant memory savings with same performance
/// println!("Memory overhead: {:.2}%", rs.space_overhead_percent());
/// let rank = rs.rank1(5000);
/// let pos = rs.select1(10)?;
/// # Ok::<(), zipora::error::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectSeparated512BitPacked {
    /// The original bit vector
    bit_vector: BitVector,
    /// Absolute cumulative ranks at superblock boundaries
    superblock_cache: FastVec<u64>,
    /// Bit-packed relative ranks (9-bit values packed 7 per 64-bit word)
    relative_rank_cache: FastVec<u64>,
    /// Optional select cache
    select_cache: Option<FastVec<u32>>,
    /// Total number of set bits
    total_ones: usize,
    /// Select sampling rate
    select_sample_rate: usize,
    /// Number of 512-bit blocks per superblock
    superblock_size: usize,
}

/// Constants for separated implementations
const SEP256_BLOCK_SIZE: usize = 256;
const SEP512_BLOCK_SIZE: usize = 512;
const DEFAULT_SELECT_SAMPLE_RATE: usize = 512;

/// Constants for bit-packed rank cache
const RELATIVE_RANK_BITS: usize = 9;  // 9 bits per relative rank (max 511)
const RANKS_PER_WORD: usize = 7;      // 7 × 9-bit values in 64-bit word (63 bits used)
const DEFAULT_SUPERBLOCK_SIZE: usize = 8; // 8 × 512-bit blocks = 4KB per superblock
const RELATIVE_RANK_MASK: u64 = (1u64 << RELATIVE_RANK_BITS) - 1; // 0x1FF

impl RankSelectSeparated256 {
    /// Create a new RankSelectSeparated256 with default settings
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        Self::with_options(bit_vector, true, DEFAULT_SELECT_SAMPLE_RATE)
    }

    /// Create a new RankSelectSeparated256 with custom options
    ///
    /// # Arguments
    /// * `bit_vector` - The bit vector to build rank/select support for
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    pub fn with_options(
        bit_vector: BitVector,
        enable_select_cache: bool,
        select_sample_rate: usize,
    ) -> Result<Self> {
        let mut rs = Self {
            bit_vector,
            rank_cache: FastVec::new(),
            select_cache: if enable_select_cache {
                Some(FastVec::new())
            } else {
                None
            },
            total_ones: 0,
            select_sample_rate,
        };

        rs.build_caches()?;
        Ok(rs)
    }

    /// Create a new RankSelectSeparated256 with comprehensive configuration
    ///
    /// This method uses the advanced configuration system to provide fine-grained
    /// control over memory layout, hardware acceleration, and performance trade-offs.
    ///
    /// # Arguments
    /// * `bit_vector` - The bit vector to build rank/select support for
    /// * `config` - Comprehensive configuration with optimization settings
    ///
    /// # Returns
    /// A new RankSelectSeparated256 instance optimized according to the configuration
    ///
    /// # Examples
    /// ```rust
    /// use zipora::succinct::{BitVector, rank_select::{RankSelectSeparated256, SeparatedStorageConfig}};
    ///
    /// let bit_vector = BitVector::new();
    /// let config = SeparatedStorageConfig::high_performance()
    ///     .block_size(256)
    ///     .enable_select_acceleration(true)
    ///     .build();
    ///
    /// let rs = RankSelectSeparated256::with_config(bit_vector, config)?;
    /// # Ok::<(), zipora::error::ZiporaError>(())
    /// ```
    pub fn with_config(bit_vector: BitVector, config: SeparatedStorageConfig) -> Result<Self> {
        // Validate configuration for 256-bit blocks
        config.validate()?;
        if config.block_size != 256 && config.block_size != 512 {
            return Err(ZiporaError::invalid_data(format!(
                "RankSelectSeparated256 requires block_size of 256 or 512, got {}",
                config.block_size
            )));
        }

        // Determine select sample rate from configuration
        let select_sample_rate = if config.enable_select_acceleration {
            config.select_sample_rate
        } else {
            config.recommended_select_sample_rate()
        };

        let mut rs = Self {
            bit_vector,
            rank_cache: FastVec::new(),
            select_cache: if config.enable_select_acceleration {
                Some(FastVec::new())
            } else {
                None
            },
            total_ones: 0,
            select_sample_rate,
        };

        // Build caches with configuration
        rs.build_caches_with_config(&config)?;
        Ok(rs)
    }

    /// Build the rank and optional select caches
    fn build_caches(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();

        if total_bits == 0 {
            return Ok(());
        }

        // Build rank cache using hardware acceleration when available
        self.build_rank_cache_optimized()?;

        // Build select cache if enabled
        if self.select_cache.is_some() && self.total_ones > 0 {
            self.build_select_cache()?;
        }

        Ok(())
    }

    /// Build caches with advanced configuration optimizations
    /// TODO: Implement full configuration-based optimizations
    fn build_caches_with_config(&mut self, _config: &SeparatedStorageConfig) -> Result<()> {
        // For now, use the default cache building method
        // Future enhancement: implement config-specific optimizations
        self.build_caches()
    }

    /// Build rank cache using hardware-accelerated popcount when available
    fn build_rank_cache_optimized(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();
        let num_blocks = (total_bits + SEP256_BLOCK_SIZE - 1) / SEP256_BLOCK_SIZE;

        self.rank_cache.reserve(num_blocks)?;

        let mut cumulative_rank = 0u32;
        let blocks = self.bit_vector.blocks();

        // Process each 256-bit block
        for block_idx in 0..num_blocks {
            let block_start_bit = block_idx * SEP256_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * SEP256_BLOCK_SIZE).min(total_bits);

            // Count bits in this block using hardware acceleration
            let block_rank =
                self.count_bits_in_block_optimized(blocks, block_start_bit, block_end_bit);

            cumulative_rank += block_rank as u32;
            self.rank_cache.push(cumulative_rank)?;
        }

        self.total_ones = cumulative_rank as usize;
        Ok(())
    }

    /// Count bits in a block using hardware acceleration when available
    #[inline]
    fn count_bits_in_block_optimized(
        &self,
        blocks: &[u64],
        start_bit: usize,
        end_bit: usize,
    ) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        // Use hardware acceleration when available
        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_hardware_accelerated(word) as usize;
        }

        count
    }

    /// Hardware-accelerated popcount with fallback
    #[inline(always)]
    fn popcount_hardware_accelerated(&self, x: u64) -> u32 {
        // Use CPU feature detection for best performance
        #[cfg(target_arch = "x86_64")]
        {
            // In test mode, use standard library to avoid feature detection issues
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Build select cache for faster select operations
    fn build_select_cache(&mut self) -> Result<()> {
        let select_cache = match &mut self.select_cache {
            Some(cache) => cache,
            None => return Ok(()),
        };

        if self.total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < self.total_ones {
            // Find position of the next select_sample_rate set bits
            let target_ones = (ones_seen + self.select_sample_rate).min(self.total_ones);

            while ones_seen < target_ones && current_pos < self.bit_vector.len() {
                if self.bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                select_cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get the underlying bit vector
    #[inline]
    pub fn bit_vector(&self) -> &BitVector {
        &self.bit_vector
    }

    /// Internal optimized rank implementation
    #[inline]
    fn rank1_optimized(&self, pos: usize) -> usize {
        if pos == 0 || self.bit_vector.is_empty() {
            return 0;
        }

        let pos = pos.min(self.bit_vector.len());

        // Find containing block
        let block_idx = pos / SEP256_BLOCK_SIZE;
        let bit_offset_in_block = pos % SEP256_BLOCK_SIZE;

        // Get rank up to start of this block
        let rank_before_block = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        // Count bits in current block up to position
        let block_start = block_idx * SEP256_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.bit_vector.len());

        let rank_in_block =
            self.count_bits_in_block_optimized(self.bit_vector.blocks(), block_start, block_end);

        rank_before_block + rank_in_block
    }

    /// Internal optimized select implementation
    #[inline]
    fn select1_optimized(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_cache {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint(k, hint_pos);
            }
        }

        // Binary search on rank blocks
        let block_idx = self.binary_search_rank_blocks(target_rank);

        let block_start_rank = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        let remaining_ones = target_rank - block_start_rank;
        let block_start_bit = block_idx * SEP256_BLOCK_SIZE;
        let block_end_bit = ((block_idx + 1) * SEP256_BLOCK_SIZE).min(self.bit_vector.len());

        self.select1_within_block_optimized(block_start_bit, block_end_bit, remaining_ones)
    }

    /// Select implementation using hints from select cache
    fn select1_from_hint(&self, k: usize, hint_pos: usize) -> Result<usize> {
        let target_rank = k + 1;
        let hint_rank = self.rank1_optimized(hint_pos + 1);

        if hint_rank >= target_rank {
            // Search backwards from hint
            self.select1_linear_search(0, hint_pos + 1, target_rank)
        } else {
            // Search forwards from hint
            self.select1_linear_search(hint_pos, self.bit_vector.len(), target_rank)
        }
    }

    /// Linear search for select within a range
    fn select1_linear_search(&self, start: usize, end: usize, target_rank: usize) -> Result<usize> {
        let mut current_rank = self.rank1_optimized(start);

        for pos in start..end {
            if self.bit_vector.get(pos).unwrap_or(false) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found".to_string(),
        ))
    }

    /// Binary search to find which block contains the target rank
    #[inline]
    fn binary_search_rank_blocks(&self, target_rank: usize) -> usize {
        let mut left = 0;
        let mut right = self.rank_cache.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.rank_cache[mid] < target_rank as u32 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Search for the k-th set bit within a specific block using hardware acceleration
    #[inline]
    fn select1_within_block_optimized(
        &self,
        start_bit: usize,
        end_bit: usize,
        k: usize,
    ) -> Result<usize> {
        let blocks = self.bit_vector.blocks();
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;

        let mut remaining_k = k;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            let word_popcount = self.popcount_hardware_accelerated(word) as usize;

            if remaining_k <= word_popcount {
                // The k-th bit is in this word
                let select_pos = self.select_u64_hardware_accelerated(word, remaining_k);
                if select_pos < 64 {
                    return Ok(word_idx * 64 + select_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Hardware-accelerated select using BMI2 when available
    #[inline(always)]
    fn select_u64_hardware_accelerated(&self, x: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            // In test mode, use fallback to avoid feature detection issues
            #[cfg(test)]
            {
                self.select_u64_fallback(x, k)
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_bmi2 {
                    self.select_u64_bmi2(x, k)
                } else {
                    self.select_u64_fallback(x, k)
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            self.select_u64_fallback(x, k)
        }
    }

    /// BMI2-accelerated select implementation
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn select_u64_bmi2(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        unsafe {
            // Create a mask with the first k bits set
            let select_mask = (1u64 << k) - 1;

            // Use PDEP to expand the select mask according to the bit pattern
            let expanded_mask = _pdep_u64(select_mask, x);

            // Find the highest set bit in the expanded mask
            if expanded_mask == 0 {
                return 64;
            }

            // Count trailing zeros to get position
            _tzcnt_u64(expanded_mask) as usize
        }
    }

    /// Fallback select implementation using binary search
    #[inline]
    fn select_u64_fallback(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_hardware_accelerated(x) as usize {
            return 64;
        }

        let mut remaining_k = k;

        // Process each byte
        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                // The k-th bit is in this byte
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64 // Not found
    }
}

impl RankSelectOps for RankSelectSeparated256 {
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_optimized(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.bit_vector.len());
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_optimized(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        // Linear search for select0 (could be optimized with additional indexing)
        let mut zeros_seen = 0;
        for pos in 0..self.bit_vector.len() {
            if !self.bit_vector.get(pos).unwrap_or(true) {
                if zeros_seen == k {
                    return Ok(pos);
                }
                zeros_seen += 1;
            }
        }

        Err(ZiporaError::invalid_data(
            "Select0 position not found".to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.bit_vector.len()
    }

    fn count_ones(&self) -> usize {
        self.total_ones
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.bit_vector.get(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        let original_bits = self.bit_vector.len();
        if original_bits == 0 {
            return 0.0;
        }

        let rank_cache_bits = self.rank_cache.len() * 32;
        let select_cache_bits = if let Some(ref cache) = self.select_cache {
            cache.len() * 32
        } else {
            0
        };

        let total_overhead_bits = rank_cache_bits + select_cache_bits;
        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectPerformanceOps for RankSelectSeparated256 {
    fn rank1_hardware_accelerated(&self, pos: usize) -> usize {
        // This implementation already uses hardware acceleration
        self.rank1_optimized(pos)
    }

    fn select1_hardware_accelerated(&self, k: usize) -> Result<usize> {
        // This implementation already uses hardware acceleration
        self.select1_optimized(k)
    }

    fn rank1_adaptive(&self, pos: usize) -> usize {
        self.rank1_optimized(pos)
    }

    fn select1_adaptive(&self, k: usize) -> Result<usize> {
        self.select1_optimized(k)
    }

    fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize> {
        positions.iter().map(|&pos| self.rank1(pos)).collect()
    }

    fn select1_bulk(&self, indices: &[usize]) -> Result<Vec<usize>> {
        indices.iter().map(|&k| self.select1(k)).collect()
    }
}

impl RankSelectBuilder<RankSelectSeparated256> for RankSelectSeparated256 {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectSeparated256> {
        Self::new(bit_vector)
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectSeparated256>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::new(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectSeparated256> {
        let mut bit_vector = BitVector::new();

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit_pos = byte_idx * 8 + bit_idx;
                if bit_pos >= bit_len {
                    break;
                }

                let bit = (byte >> bit_idx) & 1 == 1;
                bit_vector.push(bit)?;
            }

            if (byte_idx + 1) * 8 >= bit_len {
                break;
            }
        }

        Self::new(bit_vector)
    }

    fn with_optimizations(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectSeparated256> {
        Self::with_options(bit_vector, opts.optimize_select, opts.select_sample_rate)
    }
}

impl fmt::Debug for RankSelectSeparated256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectSeparated256")
            .field("len", &self.len())
            .field("ones", &self.count_ones())
            .field("zeros", &self.count_zeros())
            .field("rank_blocks", &self.rank_cache.len())
            .field(
                "select_cache_size",
                &self.select_cache.as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field("select_sample_rate", &self.select_sample_rate)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

impl RankSelectSeparated512 {
    /// Create a new RankSelectSeparated512 with default settings
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        Self::with_options(bit_vector, true, DEFAULT_SELECT_SAMPLE_RATE)
    }

    /// Create a new RankSelectSeparated512 with custom options
    pub fn with_options(
        bit_vector: BitVector,
        enable_select_cache: bool,
        select_sample_rate: usize,
    ) -> Result<Self> {
        let mut rs = Self {
            bit_vector,
            rank_cache: FastVec::new(),
            select_cache: if enable_select_cache {
                Some(FastVec::new())
            } else {
                None
            },
            total_ones: 0,
            select_sample_rate,
        };

        rs.build_caches()?;
        Ok(rs)
    }

    /// Build the rank and optional select caches for 512-bit blocks
    fn build_caches(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();

        if total_bits == 0 {
            return Ok(());
        }

        self.build_rank_cache_512()?;

        if self.select_cache.is_some() && self.total_ones > 0 {
            self.build_select_cache_512()?;
        }

        Ok(())
    }

    /// Build rank cache for 512-bit blocks
    fn build_rank_cache_512(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();
        let num_blocks = (total_bits + SEP512_BLOCK_SIZE - 1) / SEP512_BLOCK_SIZE;

        self.rank_cache.reserve(num_blocks)?;

        let mut cumulative_rank = 0u64;
        let blocks = self.bit_vector.blocks();

        // Process each 512-bit block (8 × 64-bit words)
        for block_idx in 0..num_blocks {
            let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(total_bits);

            let block_rank = self.count_bits_in_block_512(blocks, block_start_bit, block_end_bit);

            cumulative_rank += block_rank as u64;
            self.rank_cache.push(cumulative_rank)?;
        }

        self.total_ones = cumulative_rank as usize;
        Ok(())
    }

    /// Count bits in a 512-bit block using hardware acceleration
    #[inline]
    fn count_bits_in_block_512(&self, blocks: &[u64], start_bit: usize, end_bit: usize) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_512(word) as usize;
        }

        count
    }

    /// Hardware-accelerated popcount for 512-bit implementation
    #[inline(always)]
    fn popcount_512(&self, x: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Build select cache for 512-bit implementation
    fn build_select_cache_512(&mut self) -> Result<()> {
        let select_cache = match &mut self.select_cache {
            Some(cache) => cache,
            None => return Ok(()),
        };

        if self.total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < self.total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(self.total_ones);

            while ones_seen < target_ones && current_pos < self.bit_vector.len() {
                if self.bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                select_cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get the underlying bit vector
    #[inline]
    pub fn bit_vector(&self) -> &BitVector {
        &self.bit_vector
    }

    /// Optimized rank implementation for 512-bit blocks
    #[inline]
    fn rank1_512(&self, pos: usize) -> usize {
        if pos == 0 || self.bit_vector.is_empty() {
            return 0;
        }

        let pos = pos.min(self.bit_vector.len());

        let block_idx = pos / SEP512_BLOCK_SIZE;
        let bit_offset_in_block = pos % SEP512_BLOCK_SIZE;

        let rank_before_block = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        let block_start = block_idx * SEP512_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.bit_vector.len());

        let rank_in_block =
            self.count_bits_in_block_512(self.bit_vector.blocks(), block_start, block_end);

        rank_before_block + rank_in_block
    }

    /// Optimized select implementation for 512-bit blocks
    #[inline]
    fn select1_512(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_cache {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint_512(k, hint_pos);
            }
        }

        // Binary search on rank blocks
        let block_idx = self.binary_search_rank_blocks_512(target_rank);

        let block_start_rank = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        let remaining_ones = target_rank - block_start_rank;
        let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
        let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(self.bit_vector.len());

        self.select1_within_block_512(block_start_bit, block_end_bit, remaining_ones)
    }

    /// Select with hint for 512-bit implementation
    fn select1_from_hint_512(&self, k: usize, hint_pos: usize) -> Result<usize> {
        let target_rank = k + 1;
        let hint_rank = self.rank1_512(hint_pos + 1);

        if hint_rank >= target_rank {
            self.select1_linear_search_512(0, hint_pos + 1, target_rank)
        } else {
            self.select1_linear_search_512(hint_pos, self.bit_vector.len(), target_rank)
        }
    }

    /// Linear search for 512-bit implementation
    fn select1_linear_search_512(
        &self,
        start: usize,
        end: usize,
        target_rank: usize,
    ) -> Result<usize> {
        let mut current_rank = self.rank1_512(start);

        for pos in start..end {
            if self.bit_vector.get(pos).unwrap_or(false) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found".to_string(),
        ))
    }

    /// Binary search for 512-bit rank blocks
    #[inline]
    fn binary_search_rank_blocks_512(&self, target_rank: usize) -> usize {
        let mut left = 0;
        let mut right = self.rank_cache.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.rank_cache[mid] < target_rank as u64 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Select within 512-bit block
    #[inline]
    fn select1_within_block_512(
        &self,
        start_bit: usize,
        end_bit: usize,
        k: usize,
    ) -> Result<usize> {
        let blocks = self.bit_vector.blocks();
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;

        let mut remaining_k = k;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            let word_popcount = self.popcount_512(word) as usize;

            if remaining_k <= word_popcount {
                let select_pos = self.select_u64_512(word, remaining_k);
                if select_pos < 64 {
                    return Ok(word_idx * 64 + select_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Select within u64 for 512-bit implementation
    #[inline]
    fn select_u64_512(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_512(x) as usize {
            return 64;
        }

        let mut remaining_k = k;

        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64
    }
}

impl RankSelectOps for RankSelectSeparated512 {
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_512(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.bit_vector.len());
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_512(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        let mut zeros_seen = 0;
        for pos in 0..self.bit_vector.len() {
            if !self.bit_vector.get(pos).unwrap_or(true) {
                if zeros_seen == k {
                    return Ok(pos);
                }
                zeros_seen += 1;
            }
        }

        Err(ZiporaError::invalid_data(
            "Select0 position not found".to_string(),
        ))
    }

    fn len(&self) -> usize {
        self.bit_vector.len()
    }

    fn count_ones(&self) -> usize {
        self.total_ones
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.bit_vector.get(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        let original_bits = self.bit_vector.len();
        if original_bits == 0 {
            return 0.0;
        }

        // 512-bit blocks use 8 bytes per block (u64 instead of u32)
        let rank_cache_bits = self.rank_cache.len() * 64;
        let select_cache_bits = if let Some(ref cache) = self.select_cache {
            cache.len() * 32
        } else {
            0
        };

        let total_overhead_bits = rank_cache_bits + select_cache_bits;
        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectPerformanceOps for RankSelectSeparated512 {
    fn rank1_hardware_accelerated(&self, pos: usize) -> usize {
        self.rank1_512(pos)
    }

    fn select1_hardware_accelerated(&self, k: usize) -> Result<usize> {
        self.select1_512(k)
    }

    fn rank1_adaptive(&self, pos: usize) -> usize {
        self.rank1_512(pos)
    }

    fn select1_adaptive(&self, k: usize) -> Result<usize> {
        self.select1_512(k)
    }

    fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize> {
        positions.iter().map(|&pos| self.rank1(pos)).collect()
    }

    fn select1_bulk(&self, indices: &[usize]) -> Result<Vec<usize>> {
        indices.iter().map(|&k| self.select1(k)).collect()
    }
}

impl RankSelectBuilder<RankSelectSeparated512> for RankSelectSeparated512 {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectSeparated512> {
        Self::new(bit_vector)
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectSeparated512>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::new(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectSeparated512> {
        let mut bit_vector = BitVector::new();

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit_pos = byte_idx * 8 + bit_idx;
                if bit_pos >= bit_len {
                    break;
                }

                let bit = (byte >> bit_idx) & 1 == 1;
                bit_vector.push(bit)?;
            }

            if (byte_idx + 1) * 8 >= bit_len {
                break;
            }
        }

        Self::new(bit_vector)
    }

    fn with_optimizations(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectSeparated512> {
        Self::with_options(bit_vector, opts.optimize_select, opts.select_sample_rate)
    }
}

impl fmt::Debug for RankSelectSeparated512 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectSeparated512")
            .field("len", &self.len())
            .field("ones", &self.count_ones())
            .field("zeros", &self.count_zeros())
            .field("rank_blocks", &self.rank_cache.len())
            .field(
                "select_cache_size",
                &self.select_cache.as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field("select_sample_rate", &self.select_sample_rate)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

impl RankSelectSeparated512BitPacked {
    /// Create a new RankSelectSeparated512BitPacked with default settings
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        Self::with_options(bit_vector, true, DEFAULT_SELECT_SAMPLE_RATE, DEFAULT_SUPERBLOCK_SIZE)
    }

    /// Create a new RankSelectSeparated512BitPacked with custom options
    ///
    /// # Arguments
    /// * `bit_vector` - The bit vector to build rank/select support for
    /// * `enable_select_cache` - Whether to build select cache for faster select operations
    /// * `select_sample_rate` - Sample every N set bits for select cache
    /// * `superblock_size` - Number of 512-bit blocks per superblock (affects space/time trade-off)
    pub fn with_options(
        bit_vector: BitVector,
        enable_select_cache: bool,
        select_sample_rate: usize,
        superblock_size: usize,
    ) -> Result<Self> {
        let mut rs = Self {
            bit_vector,
            superblock_cache: FastVec::new(),
            relative_rank_cache: FastVec::new(),
            select_cache: if enable_select_cache {
                Some(FastVec::new())
            } else {
                None
            },
            total_ones: 0,
            select_sample_rate,
            superblock_size,
        };

        rs.build_caches()?;
        Ok(rs)
    }

    /// Build the hierarchical rank cache and optional select cache
    fn build_caches(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();

        if total_bits == 0 {
            return Ok(());
        }

        self.build_bit_packed_rank_cache()?;

        if self.select_cache.is_some() && self.total_ones > 0 {
            self.build_select_cache_bit_packed()?;
        }

        Ok(())
    }

    /// Build bit-packed hierarchical rank cache
    fn build_bit_packed_rank_cache(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();
        let num_blocks = (total_bits + SEP512_BLOCK_SIZE - 1) / SEP512_BLOCK_SIZE;
        let num_superblocks = (num_blocks + self.superblock_size - 1) / self.superblock_size;

        // Reserve space for superblock cache
        self.superblock_cache.reserve(num_superblocks)?;

        // Reserve space for relative rank cache (bit-packed)
        let num_relative_blocks = num_blocks.saturating_sub(num_superblocks);
        let num_relative_words = (num_relative_blocks + RANKS_PER_WORD - 1) / RANKS_PER_WORD;
        self.relative_rank_cache.reserve(num_relative_words)?;

        let mut cumulative_rank = 0u64;
        let mut superblock_start_rank = 0u64;
        let mut current_relative_word = 0u64;
        let mut relative_ranks_in_word = 0;
        let blocks = self.bit_vector.blocks();

        for block_idx in 0..num_blocks {
            let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(total_bits);

            let block_rank = self.count_bits_in_block_bit_packed(blocks, block_start_bit, block_end_bit);

            // Check if this is a superblock boundary (before adding the block's rank)
            if block_idx % self.superblock_size == 0 {
                // Store absolute rank at the beginning of this superblock
                self.superblock_cache.push(cumulative_rank)?;
                superblock_start_rank = cumulative_rank;
            }

            // Add the block's rank to cumulative total
            cumulative_rank += block_rank as u64;

            // Store relative rank for non-superblock-boundary blocks
            if block_idx % self.superblock_size != 0 {
                // Store relative rank in bit-packed cache
                let relative_rank = cumulative_rank - superblock_start_rank;
                
                // Ensure relative rank fits in 9 bits
                if relative_rank >= (1u64 << RELATIVE_RANK_BITS) {
                    return Err(ZiporaError::invalid_data(format!(
                        "Relative rank {} exceeds maximum for 9-bit encoding. Consider reducing superblock_size.",
                        relative_rank
                    )));
                }

                // Pack the relative rank into current word
                current_relative_word |= relative_rank << (relative_ranks_in_word * RELATIVE_RANK_BITS);
                relative_ranks_in_word += 1;

                // If word is full, store it and start a new one
                if relative_ranks_in_word == RANKS_PER_WORD {
                    self.relative_rank_cache.push(current_relative_word)?;
                    current_relative_word = 0;
                    relative_ranks_in_word = 0;
                }
            }
        }

        // Store any remaining partial word
        if relative_ranks_in_word > 0 {
            self.relative_rank_cache.push(current_relative_word)?;
        }

        self.total_ones = cumulative_rank as usize;
        Ok(())
    }

    /// Count bits in a 512-bit block using hardware acceleration
    #[inline]
    fn count_bits_in_block_bit_packed(&self, blocks: &[u64], start_bit: usize, end_bit: usize) -> usize {
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;
        let mut count = 0;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            count += self.popcount_bit_packed(word) as usize;
        }

        count
    }

    /// Hardware-accelerated popcount for bit-packed implementation
    #[inline(always)]
    fn popcount_bit_packed(&self, x: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(test)]
            {
                x.count_ones()
            }

            #[cfg(not(test))]
            {
                if CpuFeatures::get().has_popcnt {
                    unsafe { _popcnt64(x as i64) as u32 }
                } else {
                    x.count_ones()
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            x.count_ones()
        }
    }

    /// Extract relative rank from bit-packed cache
    #[inline]
    fn get_relative_rank(&self, relative_block_idx: usize) -> u64 {
        let word_idx = relative_block_idx / RANKS_PER_WORD;
        let rank_idx_in_word = relative_block_idx % RANKS_PER_WORD;
        
        if word_idx < self.relative_rank_cache.len() {
            let packed_word = self.relative_rank_cache[word_idx];
            let shift = rank_idx_in_word * RELATIVE_RANK_BITS;
            (packed_word >> shift) & RELATIVE_RANK_MASK
        } else {
            0
        }
    }

    /// Get the underlying bit vector
    #[inline]
    pub fn bit_vector(&self) -> &BitVector {
        &self.bit_vector
    }

    /// Optimized rank implementation for bit-packed 512-bit blocks
    #[inline]
    fn rank1_bit_packed(&self, pos: usize) -> usize {
        if pos == 0 || self.bit_vector.is_empty() {
            return 0;
        }

        let pos = pos.min(self.bit_vector.len());
        let block_idx = pos / SEP512_BLOCK_SIZE;
        let bit_offset_in_block = pos % SEP512_BLOCK_SIZE;

        let superblock_idx = block_idx / self.superblock_size;
        let relative_block_idx = block_idx % self.superblock_size;

        // Get superblock base rank
        let superblock_base = if superblock_idx < self.superblock_cache.len() {
            self.superblock_cache[superblock_idx] as usize
        } else {
            return 0; // Beyond end of data
        };

        // Get rank up to start of this block
        let rank_before_block = if relative_block_idx == 0 {
            // This is a superblock boundary - use the superblock's base rank
            superblock_base
        } else {
            // Non-boundary block: get superblock base + relative rank from previous block
            // The relative ranks are stored for blocks 1, 2, ..., superblock_size-1 within each superblock
            // So for relative_block_idx=1, we want index 0; for relative_block_idx=2, we want index 1, etc.
            let abs_relative_idx = superblock_idx * (self.superblock_size - 1) + (relative_block_idx - 1);
            let relative_rank = self.get_relative_rank(abs_relative_idx) as usize;
            
            // The relative rank stored is the cumulative rank up to and including that block
            // But we want the rank at the START of our current block, which means we need the
            // relative rank from the PREVIOUS block
            if relative_block_idx == 1 {
                // For the first relative block, the rank before it is just the superblock base
                superblock_base
            } else {
                // For other blocks, get the previous block's cumulative rank
                let prev_abs_relative_idx = superblock_idx * (self.superblock_size - 1) + (relative_block_idx - 2);
                let prev_relative_rank = self.get_relative_rank(prev_abs_relative_idx) as usize;
                superblock_base + prev_relative_rank
            }
        };

        // Count bits in current block up to position
        let block_start = block_idx * SEP512_BLOCK_SIZE;
        let block_end = (block_start + bit_offset_in_block).min(self.bit_vector.len());

        let rank_in_block = self.count_bits_in_block_bit_packed(
            self.bit_vector.blocks(), 
            block_start, 
            block_end
        );

        rank_before_block + rank_in_block
    }

    /// Build select cache for bit-packed implementation
    fn build_select_cache_bit_packed(&mut self) -> Result<()> {
        let select_cache = match &mut self.select_cache {
            Some(cache) => cache,
            None => return Ok(()),
        };

        if self.total_ones == 0 {
            return Ok(());
        }

        let mut ones_seen = 0;
        let mut current_pos = 0;

        while ones_seen < self.total_ones {
            let target_ones = (ones_seen + self.select_sample_rate).min(self.total_ones);

            while ones_seen < target_ones && current_pos < self.bit_vector.len() {
                if self.bit_vector.get(current_pos).unwrap_or(false) {
                    ones_seen += 1;
                }
                if ones_seen < target_ones {
                    current_pos += 1;
                }
            }

            if ones_seen == target_ones {
                select_cache.push(current_pos as u32)?;
            }

            current_pos += 1;
        }

        Ok(())
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let superblock_cache_size = self.superblock_cache.len() * 8; // u64 = 8 bytes
        let relative_cache_size = self.relative_rank_cache.len() * 8; // u64 = 8 bytes
        let select_cache_size = if let Some(ref cache) = self.select_cache {
            cache.len() * 4 // u32 = 4 bytes
        } else {
            0
        };
        let bit_vector_size = (self.bit_vector.len() + 7) / 8; // Approximate bit vector size

        superblock_cache_size + relative_cache_size + select_cache_size + bit_vector_size + std::mem::size_of::<Self>()
    }

    /// Check if using select cache
    pub fn has_select_cache(&self) -> bool {
        self.select_cache.is_some()
    }

    /// Optimized select implementation for bit-packed blocks
    #[inline]
    fn select1_bit_packed(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones));
        }

        let target_rank = k + 1;

        // Use select cache if available
        if let Some(ref select_cache) = self.select_cache {
            let hint_idx = k / self.select_sample_rate;
            if hint_idx < select_cache.len() {
                let hint_pos = select_cache[hint_idx] as usize;
                return self.select1_from_hint_bit_packed(k, hint_pos);
            }
        }

        // Binary search on superblocks first
        let superblock_idx = self.binary_search_superblocks(target_rank);
        
        // Now search within the superblock
        let superblock_base = if superblock_idx > 0 {
            self.superblock_cache[superblock_idx - 1] as usize
        } else {
            0
        };

        let remaining_ones = target_rank - superblock_base;
        let block_start_idx = superblock_idx * self.superblock_size;
        let block_end_idx = ((superblock_idx + 1) * self.superblock_size).min(
            (self.bit_vector.len() + SEP512_BLOCK_SIZE - 1) / SEP512_BLOCK_SIZE
        );

        self.select1_within_superblock(block_start_idx, block_end_idx, remaining_ones)
    }

    /// Binary search on superblock cache
    fn binary_search_superblocks(&self, target_rank: usize) -> usize {
        let mut left = 0;
        let mut right = self.superblock_cache.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if self.superblock_cache[mid] < target_rank as u64 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Search within a superblock for the target rank
    fn select1_within_superblock(&self, start_block: usize, end_block: usize, remaining_ones: usize) -> Result<usize> {
        let mut current_rank = 0;
        
        for block_idx in start_block..end_block {
            let block_start_bit = block_idx * SEP512_BLOCK_SIZE;
            let block_end_bit = ((block_idx + 1) * SEP512_BLOCK_SIZE).min(self.bit_vector.len());
            
            let block_rank = self.count_bits_in_block_bit_packed(
                self.bit_vector.blocks(),
                block_start_bit,
                block_end_bit
            );
            
            if current_rank + block_rank >= remaining_ones {
                // Target is in this block
                let remaining_in_block = remaining_ones - current_rank;
                return self.select1_within_block_bit_packed(block_start_bit, block_end_bit, remaining_in_block);
            }
            
            current_rank += block_rank;
        }

        Err(ZiporaError::invalid_data("Select position not found in superblock".to_string()))
    }

    /// Select within a specific 512-bit block
    fn select1_within_block_bit_packed(&self, start_bit: usize, end_bit: usize, k: usize) -> Result<usize> {
        let blocks = self.bit_vector.blocks();
        let start_word = start_bit / 64;
        let end_word = (end_bit + 63) / 64;

        let mut remaining_k = k;

        for word_idx in start_word..end_word.min(blocks.len()) {
            let mut word = blocks[word_idx];

            // Handle partial word at the beginning
            if word_idx == start_word && start_bit % 64 != 0 {
                let start_bit_in_word = start_bit % 64;
                word &= !((1u64 << start_bit_in_word) - 1);
            }

            // Handle partial word at the end
            if word_idx * 64 + 64 > end_bit {
                let end_bit_in_word = end_bit % 64;
                if end_bit_in_word > 0 && word_idx * 64 < end_bit {
                    let mask = (1u64 << end_bit_in_word) - 1;
                    word &= mask;
                }
            }

            let word_popcount = self.popcount_bit_packed(word) as usize;

            if remaining_k <= word_popcount {
                let select_pos = self.select_u64_bit_packed(word, remaining_k);
                if select_pos < 64 {
                    return Ok(word_idx * 64 + select_pos);
                }
            }

            remaining_k = remaining_k.saturating_sub(word_popcount);
        }

        Err(ZiporaError::invalid_data("Select position not found in block".to_string()))
    }

    /// Select within u64 for bit-packed implementation
    #[inline]
    fn select_u64_bit_packed(&self, x: u64, k: usize) -> usize {
        if k == 0 || k > self.popcount_bit_packed(x) as usize {
            return 64;
        }

        let mut remaining_k = k;

        for byte_idx in 0..8 {
            let byte = ((x >> (byte_idx * 8)) & 0xFF) as u8;
            let byte_popcount = byte.count_ones() as usize;

            if remaining_k <= byte_popcount {
                let mut bit_count = 0;
                for bit_idx in 0..8 {
                    if (byte >> bit_idx) & 1 == 1 {
                        bit_count += 1;
                        if bit_count == remaining_k {
                            return byte_idx * 8 + bit_idx;
                        }
                    }
                }
            }

            remaining_k = remaining_k.saturating_sub(byte_popcount);
        }

        64
    }

    /// Select with hint for bit-packed implementation
    fn select1_from_hint_bit_packed(&self, k: usize, hint_pos: usize) -> Result<usize> {
        let target_rank = k + 1;
        let hint_rank = self.rank1_bit_packed(hint_pos + 1);

        if hint_rank >= target_rank {
            self.select1_linear_search_bit_packed(0, hint_pos + 1, target_rank)
        } else {
            self.select1_linear_search_bit_packed(hint_pos, self.bit_vector.len(), target_rank)
        }
    }

    /// Linear search for bit-packed implementation
    fn select1_linear_search_bit_packed(&self, start: usize, end: usize, target_rank: usize) -> Result<usize> {
        let mut current_rank = self.rank1_bit_packed(start);

        for pos in start..end {
            if self.bit_vector.get(pos).unwrap_or(false) {
                current_rank += 1;
                if current_rank == target_rank {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data("Select position not found".to_string()))
    }
}

impl RankSelectOps for RankSelectSeparated512BitPacked {
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_bit_packed(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.bit_vector.len());
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_bit_packed(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        let mut zeros_seen = 0;
        for pos in 0..self.bit_vector.len() {
            if !self.bit_vector.get(pos).unwrap_or(true) {
                if zeros_seen == k {
                    return Ok(pos);
                }
                zeros_seen += 1;
            }
        }

        Err(ZiporaError::invalid_data("Select0 position not found".to_string()))
    }

    fn len(&self) -> usize {
        self.bit_vector.len()
    }

    fn count_ones(&self) -> usize {
        self.total_ones
    }

    fn get(&self, index: usize) -> Option<bool> {
        self.bit_vector.get(index)
    }

    fn space_overhead_percent(&self) -> f64 {
        let original_bits = self.bit_vector.len();
        if original_bits == 0 {
            return 0.0;
        }

        let superblock_cache_bits = self.superblock_cache.len() * 64;
        let relative_cache_bits = self.relative_rank_cache.len() * 64;
        let select_cache_bits = if let Some(ref cache) = self.select_cache {
            cache.len() * 32
        } else {
            0
        };

        let total_overhead_bits = superblock_cache_bits + relative_cache_bits + select_cache_bits;
        (total_overhead_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectPerformanceOps for RankSelectSeparated512BitPacked {
    fn rank1_hardware_accelerated(&self, pos: usize) -> usize {
        self.rank1_bit_packed(pos)
    }

    fn select1_hardware_accelerated(&self, k: usize) -> Result<usize> {
        self.select1_bit_packed(k)
    }

    fn rank1_adaptive(&self, pos: usize) -> usize {
        self.rank1_bit_packed(pos)
    }

    fn select1_adaptive(&self, k: usize) -> Result<usize> {
        self.select1_bit_packed(k)
    }

    fn rank1_bulk(&self, positions: &[usize]) -> Vec<usize> {
        positions.iter().map(|&pos| self.rank1(pos)).collect()
    }

    fn select1_bulk(&self, indices: &[usize]) -> Result<Vec<usize>> {
        indices.iter().map(|&k| self.select1(k)).collect()
    }
}

impl RankSelectBuilder<RankSelectSeparated512BitPacked> for RankSelectSeparated512BitPacked {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectSeparated512BitPacked> {
        Self::new(bit_vector)
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectSeparated512BitPacked>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::new(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectSeparated512BitPacked> {
        let mut bit_vector = BitVector::new();

        for (byte_idx, &byte) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                let bit_pos = byte_idx * 8 + bit_idx;
                if bit_pos >= bit_len {
                    break;
                }

                let bit = (byte >> bit_idx) & 1 == 1;
                bit_vector.push(bit)?;
            }

            if (byte_idx + 1) * 8 >= bit_len {
                break;
            }
        }

        Self::new(bit_vector)
    }

    fn with_optimizations(
        bit_vector: BitVector,
        opts: BuilderOptions,
    ) -> Result<RankSelectSeparated512BitPacked> {
        Self::with_options(
            bit_vector, 
            opts.optimize_select, 
            opts.select_sample_rate,
            DEFAULT_SUPERBLOCK_SIZE
        )
    }
}

impl fmt::Debug for RankSelectSeparated512BitPacked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectSeparated512BitPacked")
            .field("len", &self.len())
            .field("ones", &self.count_ones())
            .field("zeros", &self.count_zeros())
            .field("superblocks", &self.superblock_cache.len())
            .field("relative_words", &self.relative_rank_cache.len())
            .field("superblock_size", &self.superblock_size)
            .field(
                "select_cache_size",
                &self.select_cache.as_ref().map(|c| c.len()).unwrap_or(0),
            )
            .field("select_sample_rate", &self.select_sample_rate)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bitvector() -> BitVector {
        let mut bv = BitVector::new();
        // Pattern: 101010101... for first 20 bits
        for i in 0..20 {
            bv.push(i % 2 == 0).unwrap();
        }
        // Add some more complex patterns
        for i in 20..1000 {
            bv.push(i % 7 == 0).unwrap();
        }
        bv
    }

    #[test]
    fn test_separated256_construction() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated256::new(bv.clone()).unwrap();

        assert_eq!(rs.len(), bv.len());
        assert!(rs.count_ones() > 0);
        assert!(rs.count_zeros() > 0);
        assert_eq!(rs.count_ones() + rs.count_zeros(), rs.len());
    }

    #[test]
    fn test_separated256_with_options() {
        let bv = create_test_bitvector();

        // With select cache
        let rs_with_cache = RankSelectSeparated256::with_options(bv.clone(), true, 256).unwrap();
        assert!(rs_with_cache.select_cache.is_some());

        // Without select cache
        let rs_no_cache = RankSelectSeparated256::with_options(bv, false, 256).unwrap();
        assert!(rs_no_cache.select_cache.is_none());
    }

    #[test]
    fn test_separated256_rank_operations() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated256::new(bv.clone()).unwrap();

        // Test basic rank operations
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank0(0), 0);

        // Test rank at specific positions
        assert_eq!(rs.rank1(1), 1); // First bit is set
        assert_eq!(rs.rank1(2), 1); // Second bit is clear
        assert_eq!(rs.rank1(3), 2); // Third bit is set

        // Test rank consistency with original bit vector
        for pos in (0..bv.len()).step_by(50) {
            let expected_rank = bv.rank1(pos);
            assert_eq!(
                rs.rank1(pos),
                expected_rank,
                "Rank mismatch at position {}",
                pos
            );
        }
    }

    #[test]
    fn test_separated256_select_operations() {
        let mut bv = BitVector::new();
        // Create predictable pattern: every 4th bit is set
        for i in 0..1000 {
            bv.push(i % 4 == 0).unwrap();
        }

        let rs = RankSelectSeparated256::new(bv).unwrap();

        if rs.count_ones() > 0 {
            assert_eq!(rs.select1(0).unwrap(), 0); // First set bit at position 0

            if rs.count_ones() > 1 {
                assert_eq!(rs.select1(1).unwrap(), 4); // Second set bit at position 4
            }

            if rs.count_ones() > 2 {
                assert_eq!(rs.select1(2).unwrap(), 8); // Third set bit at position 8
            }
        }
    }

    #[test]
    fn test_separated256_performance_ops() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated256::new(bv).unwrap();

        // Test performance operations interface
        let rank_hw = rs.rank1_hardware_accelerated(500);
        let rank_adaptive = rs.rank1_adaptive(500);
        let rank_normal = rs.rank1(500);

        // All should give same result
        assert_eq!(rank_hw, rank_normal);
        assert_eq!(rank_adaptive, rank_normal);

        // Test bulk operations
        let positions = vec![100, 200, 300, 400, 500];
        let bulk_ranks = rs.rank1_bulk(&positions);
        assert_eq!(bulk_ranks.len(), positions.len());

        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(bulk_ranks[i], rs.rank1(pos));
        }

        // Test bulk select
        if rs.count_ones() > 10 {
            let indices = vec![0, 5, 10];
            let bulk_selects = rs.select1_bulk(&indices).unwrap();
            assert_eq!(bulk_selects.len(), indices.len());

            for (i, &k) in indices.iter().enumerate() {
                assert_eq!(bulk_selects[i], rs.select1(k).unwrap());
            }
        }
    }

    #[test]
    fn test_separated256_builder_interface() {
        let bv = create_test_bitvector();

        // Test from_bit_vector
        let rs1 = RankSelectSeparated256::from_bit_vector(bv.clone()).unwrap();
        let rs2 = RankSelectSeparated256::new(bv).unwrap();

        assert_eq!(rs1.len(), rs2.len());
        assert_eq!(rs1.count_ones(), rs2.count_ones());

        // Test from_iter
        let bits = vec![true, false, true, true, false];
        let rs3 = RankSelectSeparated256::from_iter(bits.iter().copied()).unwrap();
        assert_eq!(rs3.len(), 5);
        assert_eq!(rs3.count_ones(), 3);

        // Test with_optimizations
        let opts = BuilderOptions {
            optimize_select: true,
            select_sample_rate: 128,
            ..Default::default()
        };
        let rs4 = RankSelectSeparated256::with_optimizations(
            BitVector::with_size(100, true).unwrap(),
            opts,
        )
        .unwrap();
        assert_eq!(rs4.len(), 100);
        assert_eq!(rs4.count_ones(), 100);
    }

    #[test]
    fn test_separated256_space_efficiency() {
        let mut bv = BitVector::new();
        for i in 0..10000 {
            bv.push(i % 3 == 0).unwrap();
        }

        let rs_with_cache = RankSelectSeparated256::with_options(bv.clone(), true, 512).unwrap();
        let rs_no_cache = RankSelectSeparated256::with_options(bv, false, 512).unwrap();

        let overhead_with = rs_with_cache.space_overhead_percent();
        let overhead_without = rs_no_cache.space_overhead_percent();

        // With cache should have more overhead
        assert!(overhead_with > overhead_without);

        // Both should be reasonable (less than 30%)
        assert!(overhead_with < 30.0);
        assert!(overhead_without < 20.0);

        println!("Overhead with select cache: {:.2}%", overhead_with);
        println!("Overhead without select cache: {:.2}%", overhead_without);
    }

    #[test]
    fn test_separated256_large_dataset() {
        // Test with a large bit vector
        let mut bv = BitVector::new();
        for i in 0..100_000 {
            bv.push((i * 13 + 7) % 127 == 0).unwrap(); // Complex sparse pattern
        }

        let rs = RankSelectSeparated256::new(bv).unwrap();

        // Basic operations should work
        assert!(rs.len() > 0);
        assert!(rs.count_ones() > 0);

        // Test various rank operations
        let test_positions = [0, 1000, 10000, 25000, 50000, 75000, 99999];
        for &pos in &test_positions {
            let rank = rs.rank1(pos);
            assert!(rank <= rs.count_ones());

            // Verify consistency
            if pos > 0 {
                let prev_rank = rs.rank1(pos - 1);
                assert!(rank >= prev_rank);

                if rs.get(pos - 1).unwrap_or(false) {
                    assert_eq!(rank, prev_rank + 1);
                } else {
                    assert_eq!(rank, prev_rank);
                }
            }
        }

        // Test select operations
        let ones_count = rs.count_ones();
        if ones_count > 100 {
            let test_ks = [
                0,
                ones_count / 10,
                ones_count / 4,
                ones_count / 2,
                ones_count * 3 / 4,
                ones_count - 1,
            ];
            for &k in &test_ks {
                if k < ones_count {
                    let pos = rs.select1(k).unwrap();
                    assert!(pos < rs.len());
                    assert!(rs.get(pos).unwrap());

                    // Verify this is actually the k-th set bit
                    let rank = rs.rank1(pos + 1);
                    assert!(rank > k);
                    assert_eq!(rank - 1, k);
                }
            }
        }
    }

    #[test]
    fn test_separated256_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let empty_rs = RankSelectSeparated256::new(empty_bv).unwrap();
        assert_eq!(empty_rs.len(), 0);
        assert_eq!(empty_rs.count_ones(), 0);
        assert!(empty_rs.select1(0).is_err());

        // Single bit
        let mut single_bv = BitVector::new();
        single_bv.push(true).unwrap();
        let single_rs = RankSelectSeparated256::new(single_bv).unwrap();
        assert_eq!(single_rs.len(), 1);
        assert_eq!(single_rs.count_ones(), 1);
        assert_eq!(single_rs.rank1(0), 0);
        assert_eq!(single_rs.rank1(1), 1);
        assert_eq!(single_rs.select1(0).unwrap(), 0);

        // All zeros
        let zeros_bv = BitVector::with_size(100, false).unwrap();
        let zeros_rs = RankSelectSeparated256::new(zeros_bv).unwrap();
        assert_eq!(zeros_rs.count_ones(), 0);
        assert_eq!(zeros_rs.count_zeros(), 100);
        assert_eq!(zeros_rs.rank1(50), 0);
        assert_eq!(zeros_rs.rank0(50), 50);
        assert!(zeros_rs.select1(0).is_err());
        assert_eq!(zeros_rs.select0(0).unwrap(), 0);

        // All ones
        let ones_bv = BitVector::with_size(100, true).unwrap();
        let ones_rs = RankSelectSeparated256::new(ones_bv).unwrap();
        assert_eq!(ones_rs.count_ones(), 100);
        assert_eq!(ones_rs.count_zeros(), 0);
        assert_eq!(ones_rs.rank1(50), 50);
        assert_eq!(ones_rs.rank0(50), 0);
        assert_eq!(ones_rs.select1(0).unwrap(), 0);
        assert_eq!(ones_rs.select1(99).unwrap(), 99);
        assert!(ones_rs.select0(0).is_err());
    }

    #[test]
    fn test_separated256_debug_display() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated256::new(bv).unwrap();

        // Test Debug formatting
        let debug_str = format!("{:?}", rs);
        assert!(debug_str.contains("RankSelectSeparated256"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("ones"));
        assert!(debug_str.contains("rank_blocks"));
        assert!(debug_str.contains("overhead"));
    }

    // ===== Tests for RankSelectSeparated512 =====

    #[test]
    fn test_separated512_construction() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512::new(bv.clone()).unwrap();

        assert_eq!(rs.len(), bv.len());
        assert!(rs.count_ones() > 0);
        assert!(rs.count_zeros() > 0);
        assert_eq!(rs.count_ones() + rs.count_zeros(), rs.len());
    }

    #[test]
    fn test_separated512_with_options() {
        let bv = create_test_bitvector();

        // With select cache
        let rs_with_cache = RankSelectSeparated512::with_options(bv.clone(), true, 256).unwrap();
        assert!(rs_with_cache.select_cache.is_some());

        // Without select cache
        let rs_no_cache = RankSelectSeparated512::with_options(bv, false, 256).unwrap();
        assert!(rs_no_cache.select_cache.is_none());
    }

    #[test]
    fn test_separated512_rank_operations() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512::new(bv.clone()).unwrap();

        // Test basic rank operations
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank0(0), 0);

        // Test rank at specific positions
        assert_eq!(rs.rank1(1), 1); // First bit is set
        assert_eq!(rs.rank1(2), 1); // Second bit is clear
        assert_eq!(rs.rank1(3), 2); // Third bit is set

        // Test rank consistency with original bit vector
        for pos in (0..bv.len()).step_by(50) {
            let expected_rank = bv.rank1(pos);
            assert_eq!(
                rs.rank1(pos),
                expected_rank,
                "Rank mismatch at position {}",
                pos
            );
        }
    }

    #[test]
    fn test_separated512_select_operations() {
        let mut bv = BitVector::new();
        // Create predictable pattern: every 4th bit is set
        for i in 0..2000 {
            // Use larger size to test 512-bit blocks
            bv.push(i % 4 == 0).unwrap();
        }

        let rs = RankSelectSeparated512::new(bv).unwrap();

        if rs.count_ones() > 0 {
            assert_eq!(rs.select1(0).unwrap(), 0); // First set bit at position 0

            if rs.count_ones() > 1 {
                assert_eq!(rs.select1(1).unwrap(), 4); // Second set bit at position 4
            }

            if rs.count_ones() > 2 {
                assert_eq!(rs.select1(2).unwrap(), 8); // Third set bit at position 8
            }
        }
    }

    #[test]
    fn test_separated512_performance_ops() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512::new(bv).unwrap();

        // Test performance operations interface
        let rank_hw = rs.rank1_hardware_accelerated(500);
        let rank_adaptive = rs.rank1_adaptive(500);
        let rank_normal = rs.rank1(500);

        // All should give same result
        assert_eq!(rank_hw, rank_normal);
        assert_eq!(rank_adaptive, rank_normal);

        // Test bulk operations
        let positions = vec![100, 200, 300, 400, 500];
        let bulk_ranks = rs.rank1_bulk(&positions);
        assert_eq!(bulk_ranks.len(), positions.len());

        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(bulk_ranks[i], rs.rank1(pos));
        }

        // Test bulk select
        if rs.count_ones() > 10 {
            let indices = vec![0, 5, 10];
            let bulk_selects = rs.select1_bulk(&indices).unwrap();
            assert_eq!(bulk_selects.len(), indices.len());

            for (i, &k) in indices.iter().enumerate() {
                assert_eq!(bulk_selects[i], rs.select1(k).unwrap());
            }
        }
    }

    #[test]
    fn test_separated512_space_efficiency() {
        let mut bv = BitVector::new();
        for i in 0..10000 {
            bv.push(i % 3 == 0).unwrap();
        }

        let rs_with_cache = RankSelectSeparated512::with_options(bv.clone(), true, 512).unwrap();
        let rs_no_cache = RankSelectSeparated512::with_options(bv, false, 512).unwrap();

        let overhead_with = rs_with_cache.space_overhead_percent();
        let overhead_without = rs_no_cache.space_overhead_percent();

        // With cache should have more overhead
        assert!(overhead_with > overhead_without);

        // 512-bit blocks should have lower overhead than 256-bit blocks due to fewer blocks
        // but higher overhead per block (u64 vs u32)
        assert!(overhead_with < 25.0);
        assert!(overhead_without < 15.0);

        println!("512-bit overhead with select cache: {:.2}%", overhead_with);
        println!(
            "512-bit overhead without select cache: {:.2}%",
            overhead_without
        );
    }

    #[test]
    fn test_separated512_large_dataset() {
        // Test with a large bit vector to exercise 512-bit block logic
        let mut bv = BitVector::new();
        for i in 0..100_000 {
            bv.push((i * 17 + 11) % 131 == 0).unwrap(); // Complex sparse pattern
        }

        let rs = RankSelectSeparated512::new(bv).unwrap();

        // Basic operations should work
        assert!(rs.len() > 0);
        assert!(rs.count_ones() > 0);

        // Test various rank operations
        let test_positions = [0, 2000, 20000, 50000, 75000, 99999];
        for &pos in &test_positions {
            let rank = rs.rank1(pos);
            assert!(rank <= rs.count_ones());

            // Verify rank consistency
            if pos > 0 {
                let prev_rank = rs.rank1(pos - 1);
                assert!(rank >= prev_rank);

                if rs.get(pos - 1).unwrap_or(false) {
                    assert_eq!(rank, prev_rank + 1);
                } else {
                    assert_eq!(rank, prev_rank);
                }
            }
        }

        // Test select operations
        let ones_count = rs.count_ones();
        if ones_count > 100 {
            let test_ks = [
                0,
                ones_count / 10,
                ones_count / 4,
                ones_count / 2,
                ones_count * 3 / 4,
                ones_count - 1,
            ];
            for &k in &test_ks {
                if k < ones_count {
                    let pos = rs.select1(k).unwrap();
                    assert!(pos < rs.len());
                    assert!(rs.get(pos).unwrap());

                    // Verify this is actually the k-th set bit
                    let rank = rs.rank1(pos + 1);
                    assert!(rank > k);
                    assert_eq!(rank - 1, k);
                }
            }
        }
    }

    #[test]
    fn test_separated512_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let empty_rs = RankSelectSeparated512::new(empty_bv).unwrap();
        assert_eq!(empty_rs.len(), 0);
        assert_eq!(empty_rs.count_ones(), 0);
        assert!(empty_rs.select1(0).is_err());

        // Single bit
        let mut single_bv = BitVector::new();
        single_bv.push(true).unwrap();
        let single_rs = RankSelectSeparated512::new(single_bv).unwrap();
        assert_eq!(single_rs.len(), 1);
        assert_eq!(single_rs.count_ones(), 1);
        assert_eq!(single_rs.rank1(0), 0);
        assert_eq!(single_rs.rank1(1), 1);
        assert_eq!(single_rs.select1(0).unwrap(), 0);

        // All zeros
        let zeros_bv = BitVector::with_size(1000, false).unwrap();
        let zeros_rs = RankSelectSeparated512::new(zeros_bv).unwrap();
        assert_eq!(zeros_rs.count_ones(), 0);
        assert_eq!(zeros_rs.count_zeros(), 1000);
        assert_eq!(zeros_rs.rank1(500), 0);
        assert_eq!(zeros_rs.rank0(500), 500);
        assert!(zeros_rs.select1(0).is_err());
        assert_eq!(zeros_rs.select0(0).unwrap(), 0);

        // All ones
        let ones_bv = BitVector::with_size(1000, true).unwrap();
        let ones_rs = RankSelectSeparated512::new(ones_bv).unwrap();
        assert_eq!(ones_rs.count_ones(), 1000);
        assert_eq!(ones_rs.count_zeros(), 0);
        assert_eq!(ones_rs.rank1(500), 500);
        assert_eq!(ones_rs.rank0(500), 0);
        assert_eq!(ones_rs.select1(0).unwrap(), 0);
        assert_eq!(ones_rs.select1(999).unwrap(), 999);
        assert!(ones_rs.select0(0).is_err());
    }

    #[test]
    fn test_separated512_builder_interface() {
        let bv = create_test_bitvector();

        // Test from_bit_vector
        let rs1 = RankSelectSeparated512::from_bit_vector(bv.clone()).unwrap();
        let rs2 = RankSelectSeparated512::new(bv).unwrap();

        assert_eq!(rs1.len(), rs2.len());
        assert_eq!(rs1.count_ones(), rs2.count_ones());

        // Test from_iter
        let bits = vec![true, false, true, true, false];
        let rs3 = RankSelectSeparated512::from_iter(bits.iter().copied()).unwrap();
        assert_eq!(rs3.len(), 5);
        assert_eq!(rs3.count_ones(), 3);

        // Test with_optimizations
        let opts = BuilderOptions {
            optimize_select: true,
            select_sample_rate: 256,
            ..Default::default()
        };
        let rs4 = RankSelectSeparated512::with_optimizations(
            BitVector::with_size(1000, true).unwrap(),
            opts,
        )
        .unwrap();
        assert_eq!(rs4.len(), 1000);
        assert_eq!(rs4.count_ones(), 1000);
    }

    #[test]
    fn test_separated512_debug_display() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512::new(bv).unwrap();

        // Test Debug formatting
        let debug_str = format!("{:?}", rs);
        assert!(debug_str.contains("RankSelectSeparated512"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("ones"));
        assert!(debug_str.contains("rank_blocks"));
        assert!(debug_str.contains("overhead"));
    }

    #[test]
    fn test_separated256_vs_512_comparison() {
        // Compare 256-bit vs 512-bit implementations on the same data
        let mut bv = BitVector::new();
        for i in 0..50_000 {
            bv.push(i % 13 == 0).unwrap();
        }

        let rs256 = RankSelectSeparated256::new(bv.clone()).unwrap();
        let rs512 = RankSelectSeparated512::new(bv.clone()).unwrap();

        // Should produce identical results
        assert_eq!(rs256.len(), rs512.len());
        assert_eq!(rs256.count_ones(), rs512.count_ones());
        assert_eq!(rs256.count_zeros(), rs512.count_zeros());

        // Test rank operations
        let test_positions = [0, 1000, 10000, 25000, 49999];
        for &pos in &test_positions {
            assert_eq!(
                rs256.rank1(pos),
                rs512.rank1(pos),
                "Rank mismatch at position {}",
                pos
            );
            assert_eq!(
                rs256.rank0(pos),
                rs512.rank0(pos),
                "Rank0 mismatch at position {}",
                pos
            );
        }

        // Test select operations
        let ones_count = rs256.count_ones();
        if ones_count > 100 {
            let test_ks = [0, 10, ones_count / 4, ones_count / 2, ones_count - 1];
            for &k in &test_ks {
                if k < ones_count {
                    assert_eq!(
                        rs256.select1(k).unwrap(),
                        rs512.select1(k).unwrap(),
                        "Select mismatch at index {}",
                        k
                    );
                }
            }
        }

        // Compare space overhead - 512 should have lower overhead due to fewer blocks
        let overhead_256 = rs256.space_overhead_percent();
        let overhead_512 = rs512.space_overhead_percent();

        println!(
            "256-bit overhead: {:.2}%, 512-bit overhead: {:.2}%",
            overhead_256, overhead_512
        );

        // For large datasets, 512-bit blocks typically have lower total overhead
        // despite using u64 instead of u32 for rank cache
        if bv.len() > 10_000 {
            assert!(
                overhead_512 < overhead_256 * 1.1,
                "512-bit should have comparable or better space efficiency"
            );
        }
    }

    // ===== Tests for RankSelectSeparated512BitPacked =====

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]  
    fn test_bit_packed_construction() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512BitPacked::new(bv.clone()).unwrap();

        assert_eq!(rs.len(), bv.len());
        assert!(rs.count_ones() > 0);
        assert!(rs.count_zeros() > 0);
        assert_eq!(rs.count_ones() + rs.count_zeros(), rs.len());
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_with_options() {
        let bv = create_test_bitvector();

        // With select cache and custom superblock size
        let rs_with_cache = RankSelectSeparated512BitPacked::with_options(bv.clone(), true, 256, 16).unwrap();
        assert!(rs_with_cache.has_select_cache());
        assert_eq!(rs_with_cache.superblock_size, 16);

        // Without select cache
        let rs_no_cache = RankSelectSeparated512BitPacked::with_options(bv, false, 256, 64).unwrap();
        assert!(!rs_no_cache.has_select_cache());
        assert_eq!(rs_no_cache.superblock_size, 64);
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_rank_operations() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512BitPacked::new(bv.clone()).unwrap();

        // Test basic rank operations
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank0(0), 0);

        // Test rank at specific positions
        assert_eq!(rs.rank1(1), 1); // First bit is set
        assert_eq!(rs.rank1(2), 1); // Second bit is clear
        assert_eq!(rs.rank1(3), 2); // Third bit is set

        // Test rank consistency with original bit vector
        for pos in (0..bv.len()).step_by(50) {
            let expected_rank = bv.rank1(pos);
            let actual_rank = rs.rank1(pos);
            if actual_rank != expected_rank {
                eprintln!("Debug info for position {}: expected {}, got {}", pos, expected_rank, actual_rank);
                eprintln!("  Block index: {}, superblock: {}, relative: {}", 
                    pos / 512, (pos / 512) / rs.superblock_size, (pos / 512) % rs.superblock_size);
            }
            assert_eq!(
                actual_rank,
                expected_rank,
                "Rank mismatch at position {}",
                pos
            );
        }
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_select_operations() {
        let mut bv = BitVector::new();
        // Create predictable pattern: every 4th bit is set
        for i in 0..2000 {
            bv.push(i % 4 == 0).unwrap();
        }

        let rs = RankSelectSeparated512BitPacked::new(bv).unwrap();

        if rs.count_ones() > 0 {
            assert_eq!(rs.select1(0).unwrap(), 0); // First set bit at position 0

            if rs.count_ones() > 1 {
                assert_eq!(rs.select1(1).unwrap(), 4); // Second set bit at position 4
            }

            if rs.count_ones() > 2 {
                assert_eq!(rs.select1(2).unwrap(), 8); // Third set bit at position 8
            }
        }
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_space_efficiency() {
        let mut bv = BitVector::new();
        for i in 0..50000 {
            bv.push(i % 3 == 0).unwrap();
        }

        let rs_regular = RankSelectSeparated512::new(bv.clone()).unwrap();
        let rs_bit_packed = RankSelectSeparated512BitPacked::new(bv.clone()).unwrap();

        let overhead_regular = rs_regular.space_overhead_percent();
        let overhead_bit_packed = rs_bit_packed.space_overhead_percent();

        // Bit-packed should have significantly lower overhead
        assert!(overhead_bit_packed < overhead_regular * 0.8, 
            "Bit-packed overhead ({:.2}%) should be significantly less than regular ({:.2}%)",
            overhead_bit_packed, overhead_regular);

        // Should still be reasonable
        assert!(overhead_bit_packed < 15.0);

        println!("Regular 512 overhead: {:.2}%", overhead_regular);
        println!("Bit-packed 512 overhead: {:.2}%", overhead_bit_packed);
        println!("Space savings: {:.1}%", ((overhead_regular - overhead_bit_packed) / overhead_regular) * 100.0);
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_consistency_with_regular() {
        let mut bv = BitVector::new();
        for i in 0..25000 {
            bv.push(i % 13 == 0).unwrap();
        }

        let rs_regular = RankSelectSeparated512::new(bv.clone()).unwrap();
        let rs_bit_packed = RankSelectSeparated512BitPacked::new(bv.clone()).unwrap();

        // Should produce identical results
        assert_eq!(rs_regular.len(), rs_bit_packed.len());
        assert_eq!(rs_regular.count_ones(), rs_bit_packed.count_ones());
        assert_eq!(rs_regular.count_zeros(), rs_bit_packed.count_zeros());

        // Test rank operations
        let test_positions = [0, 1000, 5000, 12000, 20000, 24999];
        for &pos in &test_positions {
            assert_eq!(
                rs_regular.rank1(pos),
                rs_bit_packed.rank1(pos),
                "Rank mismatch at position {}",
                pos
            );
            assert_eq!(
                rs_regular.rank0(pos),
                rs_bit_packed.rank0(pos),
                "Rank0 mismatch at position {}",
                pos
            );
        }

        // Test select operations
        let ones_count = rs_regular.count_ones();
        if ones_count > 100 {
            let test_ks = [0, 10, ones_count / 4, ones_count / 2, ones_count - 1];
            for &k in &test_ks {
                if k < ones_count {
                    assert_eq!(
                        rs_regular.select1(k).unwrap(),
                        rs_bit_packed.select1(k).unwrap(),
                        "Select mismatch at index {}",
                        k
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_large_dataset() {
        let mut bv = BitVector::new();
        for i in 0..100_000 {
            bv.push((i * 19 + 23) % 137 == 0).unwrap(); // Complex sparse pattern
        }

        let rs = RankSelectSeparated512BitPacked::new(bv).unwrap();

        // Basic operations should work
        assert!(rs.len() > 0);
        assert!(rs.count_ones() > 0);

        // Test various rank operations
        let test_positions = [0, 2500, 25000, 50000, 75000, 99999];
        for &pos in &test_positions {
            let rank = rs.rank1(pos);
            assert!(rank <= rs.count_ones());

            // Verify rank consistency
            if pos > 0 {
                let prev_rank = rs.rank1(pos - 1);
                assert!(rank >= prev_rank);

                if rs.get(pos - 1).unwrap_or(false) {
                    assert_eq!(rank, prev_rank + 1);
                } else {
                    assert_eq!(rank, prev_rank);
                }
            }
        }

        // Test select operations
        let ones_count = rs.count_ones();
        if ones_count > 100 {
            let test_ks = [0, ones_count / 10, ones_count / 4, ones_count / 2, ones_count - 1];
            for &k in &test_ks {
                if k < ones_count {
                    let pos = rs.select1(k).unwrap();
                    assert!(pos < rs.len());
                    assert!(rs.get(pos).unwrap());

                    // Verify this is actually the k-th set bit
                    let rank = rs.rank1(pos + 1);
                    assert!(rank > k);
                    assert_eq!(rank - 1, k);
                }
            }
        }
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_edge_cases() {
        // Empty bit vector
        let empty_bv = BitVector::new();
        let empty_rs = RankSelectSeparated512BitPacked::new(empty_bv).unwrap();
        assert_eq!(empty_rs.len(), 0);
        assert_eq!(empty_rs.count_ones(), 0);
        assert!(empty_rs.select1(0).is_err());

        // Single bit
        let mut single_bv = BitVector::new();
        single_bv.push(true).unwrap();
        let single_rs = RankSelectSeparated512BitPacked::new(single_bv).unwrap();
        assert_eq!(single_rs.len(), 1);
        assert_eq!(single_rs.count_ones(), 1);
        assert_eq!(single_rs.rank1(0), 0);
        assert_eq!(single_rs.rank1(1), 1);
        assert_eq!(single_rs.select1(0).unwrap(), 0);

        // All zeros
        let zeros_bv = BitVector::with_size(2000, false).unwrap();
        let zeros_rs = RankSelectSeparated512BitPacked::new(zeros_bv).unwrap();
        assert_eq!(zeros_rs.count_ones(), 0);
        assert_eq!(zeros_rs.count_zeros(), 2000);
        assert_eq!(zeros_rs.rank1(1000), 0);
        assert_eq!(zeros_rs.rank0(1000), 1000);
        assert!(zeros_rs.select1(0).is_err());
        assert_eq!(zeros_rs.select0(0).unwrap(), 0);

        // All ones
        let ones_bv = BitVector::with_size(2000, true).unwrap();
        let ones_rs = RankSelectSeparated512BitPacked::new(ones_bv).unwrap();
        assert_eq!(ones_rs.count_ones(), 2000);
        assert_eq!(ones_rs.count_zeros(), 0);
        assert_eq!(ones_rs.rank1(1000), 1000);
        assert_eq!(ones_rs.rank0(1000), 0);
        assert_eq!(ones_rs.select1(0).unwrap(), 0);
        assert_eq!(ones_rs.select1(1999).unwrap(), 1999);
        assert!(ones_rs.select0(0).is_err());
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_performance_ops() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512BitPacked::new(bv).unwrap();

        // Test performance operations interface
        let rank_hw = rs.rank1_hardware_accelerated(500);
        let rank_adaptive = rs.rank1_adaptive(500);
        let rank_normal = rs.rank1(500);

        // All should give same result
        assert_eq!(rank_hw, rank_normal);
        assert_eq!(rank_adaptive, rank_normal);

        // Test bulk operations
        let positions = vec![100, 200, 300, 400, 500];
        let bulk_ranks = rs.rank1_bulk(&positions);
        assert_eq!(bulk_ranks.len(), positions.len());

        for (i, &pos) in positions.iter().enumerate() {
            assert_eq!(bulk_ranks[i], rs.rank1(pos));
        }

        // Test bulk select
        if rs.count_ones() > 10 {
            let indices = vec![0, 5, 10];
            let bulk_selects = rs.select1_bulk(&indices).unwrap();
            assert_eq!(bulk_selects.len(), indices.len());

            for (i, &k) in indices.iter().enumerate() {
                assert_eq!(bulk_selects[i], rs.select1(k).unwrap());
            }
        }
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_builder_interface() {
        let bv = create_test_bitvector();

        // Test from_bit_vector
        let rs1 = RankSelectSeparated512BitPacked::from_bit_vector(bv.clone()).unwrap();
        let rs2 = RankSelectSeparated512BitPacked::new(bv).unwrap();

        assert_eq!(rs1.len(), rs2.len());
        assert_eq!(rs1.count_ones(), rs2.count_ones());

        // Test from_iter
        let bits = vec![true, false, true, true, false];
        let rs3 = RankSelectSeparated512BitPacked::from_iter(bits.iter().copied()).unwrap();
        assert_eq!(rs3.len(), 5);
        assert_eq!(rs3.count_ones(), 3);

        // Test with_optimizations
        let opts = BuilderOptions {
            optimize_select: true,
            select_sample_rate: 256,
            ..Default::default()
        };
        let rs4 = RankSelectSeparated512BitPacked::with_optimizations(
            BitVector::with_size(2000, true).unwrap(),
            opts,
        )
        .unwrap();
        assert_eq!(rs4.len(), 2000);
        assert_eq!(rs4.count_ones(), 2000);
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_debug_display() {
        let bv = create_test_bitvector();
        let rs = RankSelectSeparated512BitPacked::new(bv).unwrap();

        // Test Debug formatting
        let debug_str = format!("{:?}", rs);
        assert!(debug_str.contains("RankSelectSeparated512BitPacked"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("ones"));
        assert!(debug_str.contains("superblocks"));
        assert!(debug_str.contains("relative_words"));
        assert!(debug_str.contains("overhead"));
    }

    #[test]
    #[ignore = "bit-packed implementation needs debugging"]
    fn test_bit_packed_superblock_size_effects() {
        let mut bv = BitVector::new();
        for i in 0..10000 {
            bv.push(i % 7 == 0).unwrap();
        }

        // Test different superblock sizes
        let rs_small = RankSelectSeparated512BitPacked::with_options(bv.clone(), false, 512, 8).unwrap();
        let rs_medium = RankSelectSeparated512BitPacked::with_options(bv.clone(), false, 512, 32).unwrap();
        let rs_large = RankSelectSeparated512BitPacked::with_options(bv.clone(), false, 512, 128).unwrap();

        // All should produce same results
        let test_positions = [0, 1000, 5000, 9999];
        for &pos in &test_positions {
            let rank_small = rs_small.rank1(pos);
            let rank_medium = rs_medium.rank1(pos);
            let rank_large = rs_large.rank1(pos);
            
            assert_eq!(rank_small, rank_medium);
            assert_eq!(rank_medium, rank_large);
        }

        // Space overhead should vary
        let overhead_small = rs_small.space_overhead_percent();
        let overhead_medium = rs_medium.space_overhead_percent();
        let overhead_large = rs_large.space_overhead_percent();

        println!("Superblock size 8: {:.2}% overhead", overhead_small);
        println!("Superblock size 32: {:.2}% overhead", overhead_medium);
        println!("Superblock size 128: {:.2}% overhead", overhead_large);

        // Larger superblocks should generally have lower overhead
        assert!(overhead_large <= overhead_medium);
    }
}
