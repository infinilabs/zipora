//! Simple Reference Implementation of Rank/Select Operations
//!
//! This module provides a basic, straightforward implementation of rank/select
//! operations that serves as a reference for correctness testing and performance
//! baseline comparison. It uses minimal optimizations to ensure clarity and
//! correctness over raw performance.
//!
//! # Design Philosophy
//!
//! - **Clarity over Performance**: Simple, easy-to-understand algorithms
//! - **Correctness Focus**: Straightforward implementations that are easy to verify
//! - **Minimal Dependencies**: Uses only basic data structures and operations
//! - **Testing Reference**: Serves as ground truth for other implementations
//!
//! # Performance Characteristics
//!
//! - **Memory Overhead**: ~25% (rank cache every 256 bits)
//! - **Rank Time**: O(1) with simple block lookup + linear bit counting
//! - **Select Time**: O(log n) with binary search + linear scan
//! - **Construction Time**: O(n) with straightforward bit counting
//!
//! # Use Cases
//!
//! - Reference implementation for testing
//! - Educational purposes and algorithm demonstration
//! - Small datasets where simplicity is preferred over performance
//! - Baseline for performance comparisons

use super::{BuilderOptions, RankSelectBuilder, RankSelectOps};
use crate::FastVec;
use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use std::fmt;

/// Simple rank/select implementation with basic optimizations
///
/// This implementation uses a straightforward approach with 256-bit blocks
/// and basic rank caching. It prioritizes correctness and simplicity over
/// maximum performance.
///
/// # Memory Layout
///
/// - **Bit Vector**: Original bit data (unchanged)
/// - **Rank Cache**: Cumulative rank at end of each 256-bit block
/// - **Total**: O(n/256) space overhead for rank cache
///
/// # Examples
///
/// ```rust
/// use zipora::{BitVector, RankSelectOps, RankSelectSimple};
///
/// let mut bv = BitVector::new();
/// for i in 0..100 {
///     bv.push(i % 3 == 0)?;
/// }
///
/// let rs = RankSelectSimple::new(bv)?;
///
/// // Basic operations
/// let rank = rs.rank1(50);        // Count 1s up to position 50
/// let pos = rs.select1(10)?;      // Find position of 10th set bit
/// let zeros = rs.rank0(25);       // Count 0s up to position 25
///
/// println!("Ones up to 50: {}", rank);
/// println!("10th set bit at: {}", pos);
/// # Ok::<(), zipora::ZiporaError>(())
/// ```
#[derive(Clone)]
pub struct RankSelectSimple {
    /// The original bit vector
    bit_vector: BitVector,
    /// Cumulative rank at the end of each 256-bit block
    rank_cache: FastVec<u32>,
    /// Total number of set bits
    total_ones: usize,
    /// Block size in bits (always 256 for this implementation)
    block_size: usize,
}

/// Constants for RankSelectSimple
const SIMPLE_BLOCK_SIZE: usize = 256;

impl RankSelectSimple {
    /// Create a new RankSelectSimple from a bit vector
    ///
    /// # Arguments
    /// * `bit_vector` - The bit vector to build rank/select support for
    ///
    /// # Returns
    /// A new RankSelectSimple instance or an error if construction fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::{BitVector, RankSelectSimple, RankSelectOps};
    ///
    /// let mut bv = BitVector::new();
    /// bv.push(true)?;
    /// bv.push(false)?;
    /// bv.push(true)?;
    ///
    /// let rs = RankSelectSimple::new(bv)?;
    /// assert_eq!(rs.count_ones(), 2);
    /// # Ok::<(), zipora::ZiporaError>(())
    /// ```
    pub fn new(bit_vector: BitVector) -> Result<Self> {
        let mut rs = Self {
            bit_vector,
            rank_cache: FastVec::new(),
            total_ones: 0,
            block_size: SIMPLE_BLOCK_SIZE,
        };

        rs.build_rank_cache()?;
        Ok(rs)
    }

    /// Build the rank cache for efficient rank queries
    ///
    /// This method scans through the bit vector and builds a cumulative
    /// rank cache at the end of each block for O(1) rank queries.
    fn build_rank_cache(&mut self) -> Result<()> {
        let total_bits = self.bit_vector.len();

        if total_bits == 0 {
            return Ok(());
        }

        let num_blocks = (total_bits + self.block_size - 1) / self.block_size;
        self.rank_cache.reserve(num_blocks)?;

        let mut cumulative_rank = 0u32;

        // Build rank cache by scanning each block
        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.block_size;
            let block_end = ((block_idx + 1) * self.block_size).min(total_bits);

            // Count set bits in this block using simple iteration
            for pos in block_start..block_end {
                if self.bit_vector.get(pos).unwrap_or(false) {
                    cumulative_rank += 1;
                }
            }

            // Store cumulative rank at end of block
            self.rank_cache.push(cumulative_rank)?;
        }

        self.total_ones = cumulative_rank as usize;
        Ok(())
    }

    /// Get the underlying bit vector
    #[inline]
    pub fn bit_vector(&self) -> &BitVector {
        &self.bit_vector
    }

    /// Internal rank implementation with detailed steps for clarity
    fn rank1_internal(&self, pos: usize) -> usize {
        if pos == 0 || self.bit_vector.is_empty() {
            return 0;
        }

        let pos = pos.min(self.bit_vector.len());

        // Find which block contains this position
        let block_idx = pos / self.block_size;
        let bit_offset_in_block = pos % self.block_size;

        // Get rank up to start of this block
        let rank_before_block = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        // Count bits in current block up to position
        let block_start = block_idx * self.block_size;
        let mut rank_in_block = 0;

        for i in 0..bit_offset_in_block {
            let bit_pos = block_start + i;
            if bit_pos < self.bit_vector.len() && self.bit_vector.get(bit_pos).unwrap_or(false) {
                rank_in_block += 1;
            }
        }

        rank_before_block + rank_in_block
    }

    /// Internal select implementation using binary search
    fn select1_internal(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::out_of_bounds(k, self.total_ones));
        }

        let target_rank = k + 1;

        // Binary search on rank cache to find the containing block
        let block_idx = self.binary_search_blocks(target_rank);

        // Get rank at start of this block
        let rank_before_block = if block_idx > 0 {
            self.rank_cache[block_idx - 1] as usize
        } else {
            0
        };

        // How many more 1s we need to find in this block
        let remaining_ones = target_rank - rank_before_block;

        // Linear search within the block
        let block_start = block_idx * self.block_size;
        let block_end = ((block_idx + 1) * self.block_size).min(self.bit_vector.len());

        let mut ones_found = 0;
        for pos in block_start..block_end {
            if self.bit_vector.get(pos).unwrap_or(false) {
                ones_found += 1;
                if ones_found == remaining_ones {
                    return Ok(pos);
                }
            }
        }

        Err(ZiporaError::invalid_data(
            "Select position not found in block".to_string(),
        ))
    }

    /// Binary search to find which block contains the target rank
    fn binary_search_blocks(&self, target_rank: usize) -> usize {
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
}

impl RankSelectOps for RankSelectSimple {
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_internal(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.bit_vector.len());
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_internal(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len() - self.count_ones();
        if k >= total_zeros {
            return Err(ZiporaError::out_of_bounds(k, total_zeros));
        }

        // Simple linear search for select0
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

        // Rank cache overhead: 32 bits per block
        let rank_cache_bits = self.rank_cache.len() * 32;
        (rank_cache_bits as f64 / original_bits as f64) * 100.0
    }
}

impl RankSelectBuilder<RankSelectSimple> for RankSelectSimple {
    fn from_bit_vector(bit_vector: BitVector) -> Result<RankSelectSimple> {
        Self::new(bit_vector)
    }

    fn from_iter<I>(iter: I) -> Result<RankSelectSimple>
    where
        I: IntoIterator<Item = bool>,
    {
        let mut bit_vector = BitVector::new();
        for bit in iter {
            bit_vector.push(bit)?;
        }
        Self::new(bit_vector)
    }

    fn from_bytes(bytes: &[u8], bit_len: usize) -> Result<RankSelectSimple> {
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
        _opts: BuilderOptions,
    ) -> Result<RankSelectSimple> {
        // Simple implementation ignores optimization options
        Self::new(bit_vector)
    }
}

impl fmt::Debug for RankSelectSimple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RankSelectSimple")
            .field("len", &self.len())
            .field("ones", &self.count_ones())
            .field("zeros", &self.count_zeros())
            .field("blocks", &self.rank_cache.len())
            .field("block_size", &self.block_size)
            .field(
                "overhead",
                &format!("{:.2}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

impl fmt::Display for RankSelectSimple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RankSelectSimple(len={}, ones={}, overhead={:.1}%)",
            self.len(),
            self.count_ones(),
            self.space_overhead_percent()
        )
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
        for i in 20..100 {
            bv.push(i % 7 == 0).unwrap();
        }
        bv
    }

    #[test]
    fn test_simple_construction() {
        let bv = create_test_bitvector();
        let rs = RankSelectSimple::new(bv.clone()).unwrap();

        assert_eq!(rs.len(), bv.len());
        assert!(rs.count_ones() > 0);
        assert!(rs.count_zeros() > 0);
        assert_eq!(rs.count_ones() + rs.count_zeros(), rs.len());
    }

    #[test]
    fn test_rank_operations() {
        let bv = create_test_bitvector();
        let rs = RankSelectSimple::new(bv).unwrap();

        // Test basic rank operations
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank0(0), 0);

        // Test rank at specific positions based on pattern
        assert_eq!(rs.rank1(1), 1); // First bit is set (position 0)
        assert_eq!(rs.rank1(2), 1); // Second bit is clear
        assert_eq!(rs.rank1(3), 2); // Third bit is set (position 2)

        // Test rank0
        assert_eq!(rs.rank0(1), 0); // No zeros before position 1
        assert_eq!(rs.rank0(2), 1); // One zero at position 1
        assert_eq!(rs.rank0(3), 1); // Still one zero
    }

    #[test]
    fn test_select_operations() {
        let mut bv = BitVector::new();
        // Create predictable pattern: every 4th bit is set
        for i in 0..32 {
            bv.push(i % 4 == 0).unwrap();
        }

        let rs = RankSelectSimple::new(bv).unwrap();

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
    fn test_select0_operations() {
        let mut bv = BitVector::new();
        // Pattern: 1000 1000 1000 ...
        for i in 0..16 {
            bv.push(i % 4 == 0).unwrap();
        }

        let rs = RankSelectSimple::new(bv).unwrap();

        if rs.count_zeros() > 0 {
            assert_eq!(rs.select0(0).unwrap(), 1); // First clear bit at position 1

            if rs.count_zeros() > 1 {
                assert_eq!(rs.select0(1).unwrap(), 2); // Second clear bit at position 2
            }
        }
    }

    #[test]
    fn test_empty_bitvector() {
        let bv = BitVector::new();
        let rs = RankSelectSimple::new(bv).unwrap();

        assert_eq!(rs.len(), 0);
        assert_eq!(rs.count_ones(), 0);
        assert_eq!(rs.count_zeros(), 0);
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank0(0), 0);
        assert!(rs.is_empty());

        // Select operations should fail on empty bit vector
        assert!(rs.select1(0).is_err());
        assert!(rs.select0(0).is_err());
    }

    #[test]
    fn test_all_zeros() {
        let bv = BitVector::with_size(100, false).unwrap();
        let rs = RankSelectSimple::new(bv).unwrap();

        assert_eq!(rs.len(), 100);
        assert_eq!(rs.count_ones(), 0);
        assert_eq!(rs.count_zeros(), 100);

        assert_eq!(rs.rank1(50), 0);
        assert_eq!(rs.rank0(50), 50);

        // select1 should fail on all-zeros
        assert!(rs.select1(0).is_err());

        // select0 should work
        assert_eq!(rs.select0(0).unwrap(), 0);
        assert_eq!(rs.select0(99).unwrap(), 99);
    }

    #[test]
    fn test_all_ones() {
        let bv = BitVector::with_size(100, true).unwrap();
        let rs = RankSelectSimple::new(bv).unwrap();

        assert_eq!(rs.len(), 100);
        assert_eq!(rs.count_ones(), 100);
        assert_eq!(rs.count_zeros(), 0);

        assert_eq!(rs.rank1(50), 50);
        assert_eq!(rs.rank0(50), 0);

        // select1 should work
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(99).unwrap(), 99);

        // select0 should fail on all-ones
        assert!(rs.select0(0).is_err());
    }

    #[test]
    fn test_builder_interface() {
        // Test from_bit_vector
        let bv = create_test_bitvector();
        let rs1 = RankSelectSimple::from_bit_vector(bv.clone()).unwrap();
        let rs2 = RankSelectSimple::new(bv).unwrap();

        assert_eq!(rs1.len(), rs2.len());
        assert_eq!(rs1.count_ones(), rs2.count_ones());

        // Test from_iter
        let bits = vec![true, false, true, true, false];
        let rs3 = RankSelectSimple::from_iter(bits.iter().copied()).unwrap();
        assert_eq!(rs3.len(), 5);
        assert_eq!(rs3.count_ones(), 3);

        // Test from_bytes
        let bytes = vec![0b10101010, 0b11001100];
        let rs4 = RankSelectSimple::from_bytes(&bytes, 16).unwrap();
        assert_eq!(rs4.len(), 16);
        // 0b10101010 has 4 ones, 0b11001100 has 4 ones
        assert_eq!(rs4.count_ones(), 8);
    }

    #[test]
    fn test_rank_select_consistency() {
        let bv = create_test_bitvector();
        let rs = RankSelectSimple::new(bv.clone()).unwrap();

        // Test that rank and select are inverse operations
        let ones_count = rs.count_ones();
        for k in 0..ones_count {
            let pos = rs.select1(k).unwrap();
            assert!(
                rs.get(pos).unwrap(),
                "Selected position should have bit set"
            );

            // Check that rank up to this position includes this bit
            let rank = rs.rank1(pos + 1);
            assert!(rank > rs.rank1(pos), "Rank should increase after set bit");
        }

        // Test rank consistency with bit vector
        for pos in 0..=bv.len() {
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
    fn test_space_overhead() {
        let mut bv = BitVector::new();
        for i in 0..10000 {
            bv.push(i % 3 == 0).unwrap();
        }

        let rs = RankSelectSimple::new(bv).unwrap();
        let overhead = rs.space_overhead_percent();

        // Should be reasonable overhead (less than 15% for large bit vectors)
        assert!(overhead > 0.0);
        assert!(overhead < 15.0);

        println!("Space overhead: {:.2}%", overhead);
    }

    #[test]
    fn test_large_bitvector() {
        // Test with a larger bit vector to ensure scalability
        let mut bv = BitVector::new();
        for i in 0..100_000 {
            bv.push(i % 127 == 0).unwrap(); // Sparse pattern
        }

        let rs = RankSelectSimple::new(bv).unwrap();

        // Basic operations should work
        assert!(rs.len() > 0);
        assert!(rs.count_ones() > 0);

        // Test some rank operations
        assert_eq!(rs.rank1(0), 0);
        let mid_rank = rs.rank1(50_000);
        let end_rank = rs.rank1(rs.len());
        assert_eq!(end_rank, rs.count_ones());
        assert!(mid_rank <= end_rank);

        // Test select operations
        if rs.count_ones() > 10 {
            let pos = rs.select1(10).unwrap();
            assert!(pos < rs.len());
            assert!(rs.get(pos).unwrap());
        }
    }

    #[test]
    fn test_edge_cases() {
        // Single bit
        let mut bv_single = BitVector::new();
        bv_single.push(true).unwrap();
        let rs_single = RankSelectSimple::new(bv_single).unwrap();

        assert_eq!(rs_single.len(), 1);
        assert_eq!(rs_single.count_ones(), 1);
        assert_eq!(rs_single.rank1(0), 0);
        assert_eq!(rs_single.rank1(1), 1);
        assert_eq!(rs_single.select1(0).unwrap(), 0);

        // Very small bit vector
        let mut bv_small = BitVector::new();
        bv_small.push(false).unwrap();
        bv_small.push(true).unwrap();
        bv_small.push(false).unwrap();
        let rs_small = RankSelectSimple::new(bv_small).unwrap();

        assert_eq!(rs_small.len(), 3);
        assert_eq!(rs_small.count_ones(), 1);
        assert_eq!(rs_small.select1(0).unwrap(), 1);
        assert_eq!(rs_small.select0(0).unwrap(), 0);
        assert_eq!(rs_small.select0(1).unwrap(), 2);
    }

    #[test]
    fn test_debug_display() {
        let bv = create_test_bitvector();
        let rs = RankSelectSimple::new(bv).unwrap();

        // Test Debug formatting
        let debug_str = format!("{:?}", rs);
        assert!(debug_str.contains("RankSelectSimple"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("ones"));

        // Test Display formatting
        let display_str = format!("{}", rs);
        assert!(display_str.contains("RankSelectSimple"));
        assert!(display_str.contains(&rs.len().to_string()));
        assert!(display_str.contains(&rs.count_ones().to_string()));
    }
}
