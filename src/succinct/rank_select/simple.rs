//! RankSelectSimple: Minimal rank/select with 256-bit blocks.
//!
//! Port of `rank_select_simple` from topling-zip. Simplest possible implementation:
//! one u32 per 256-bit block storing cumulative rank1. No select acceleration tables.
//! Select uses binary search over the rank cache.
//!
//! Trade-offs vs other variants:
//! - Lowest memory overhead (4 bytes per 256-bit block)
//! - Slower rank1 (O(4) popcounts per query vs O(1) for SE/IL variants)
//! - Slowest select (binary search, no acceleration table)
//! - Good baseline for testing and small bitvectors

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 256;
const WORDS_PER_LINE: usize = LINE_BITS / 64;

/// Minimal rank/select — one u32 per 256-bit block.
pub struct RankSelectSimple {
    bits: Vec<u64>,
    rank_cache: Vec<u32>,
    size: usize,
    max_rank0: usize,
    max_rank1: usize,
}

impl RankSelectSimple {
    /// Build from a BitVector.
    pub fn new(bv: BitVector) -> Result<Self> {
        let size = bv.len();
        let words: Vec<u64> = bv.blocks().to_vec();
        let nlines = (size + LINE_BITS - 1) / LINE_BITS;

        // Build rank cache: one u32 per block = cumulative rank1 at block start
        let mut rank_cache = Vec::with_capacity(nlines + 1);
        let mut cumulative_rank1 = 0u64;
        for i in 0..nlines {
            rank_cache.push(cumulative_rank1 as u32);
            for j in 0..WORDS_PER_LINE {
                let word_idx = i * WORDS_PER_LINE + j;
                if word_idx < words.len() {
                    cumulative_rank1 += words[word_idx].count_ones() as u64;
                }
            }
        }
        rank_cache.push(cumulative_rank1 as u32); // sentinel

        let max_rank1 = cumulative_rank1 as usize;
        let max_rank0 = size - max_rank1;

        Ok(Self { bits: words, rank_cache, size, max_rank0, max_rank1 })
    }

    /// Build from raw bit words.
    pub fn from_words(words: Vec<u64>, size: usize) -> Result<Self> {
        let mut bv = BitVector::new();
        for i in 0..size {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let bit = if word_idx < words.len() { (words[word_idx] >> bit_idx) & 1 == 1 } else { false };
            bv.push(bit)?;
        }
        Self::new(bv)
    }

    #[inline(always)]
    fn popcount_trail(word: u64, bit_count: usize) -> usize {
        if bit_count == 0 { return 0; }
        (word & ((1u64 << bit_count) - 1)).count_ones() as usize
    }

    pub fn max_rank0(&self) -> usize { self.max_rank0 }
    pub fn max_rank1(&self) -> usize { self.max_rank1 }
    pub fn mem_size(&self) -> usize {
        self.bits.len() * 8 + self.rank_cache.len() * 4
    }
}

impl RankSelectOps for RankSelectSimple {
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        assert!(pos <= self.size);
        if pos == 0 { return 0; }
        let block = pos / LINE_BITS;
        let mut rank = self.rank_cache[block] as usize;

        // Sum popcounts of whole words within the block
        let block_word_start = block * WORDS_PER_LINE;
        let target_word = pos / 64;
        for i in block_word_start..target_word {
            if i < self.bits.len() {
                rank += self.bits[i].count_ones() as usize;
            }
        }
        // Add partial word
        let bit_in_word = pos % 64;
        if bit_in_word > 0 && target_word < self.bits.len() {
            rank += Self::popcount_trail(self.bits[target_word], bit_in_word);
        }
        rank
    }

    #[inline]
    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        if k >= self.max_rank1 {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        // Binary search over rank_cache to find the block
        let nblocks = self.rank_cache.len() - 1;
        let mut lo = 0usize;
        let mut hi = nblocks;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.rank_cache[mid] as usize) <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the first block where rank_cache[lo] > k
        let block = lo - 1;
        let mut remaining = k - self.rank_cache[block] as usize;
        let base_bitpos = block * LINE_BITS;

        // Linear scan within the block
        for j in 0..WORDS_PER_LINE {
            let word_idx = block * WORDS_PER_LINE + j;
            if word_idx >= self.bits.len() { break; }
            let word = self.bits[word_idx];
            let ones = word.count_ones() as usize;
            if remaining < ones {
                // Target is in this word — use BMI2 pdep/tzcnt if available
                return Ok(base_bitpos + j * 64 + Self::select_in_word(word, remaining));
            }
            remaining -= ones;
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    fn select0(&self, k: usize) -> Result<usize> {
        if k >= self.max_rank0 {
            return Err(ZiporaError::invalid_data("select0 out of range"));
        }
        // Binary search: find block where cumulative rank0 > k
        let nblocks = self.rank_cache.len() - 1;
        let mut lo = 0usize;
        let mut hi = nblocks;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let rank0_at_mid = mid * LINE_BITS - self.rank_cache[mid] as usize;
            if rank0_at_mid <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let block = lo - 1;
        let rank0_at_block = block * LINE_BITS - self.rank_cache[block] as usize;
        let mut remaining = k - rank0_at_block;
        let base_bitpos = block * LINE_BITS;

        for j in 0..WORDS_PER_LINE {
            let word_idx = block * WORDS_PER_LINE + j;
            if word_idx >= self.bits.len() { break; }
            let word = self.bits[word_idx];
            let zeros = (!word).count_ones() as usize;
            // Clamp zeros for partial last word
            let max_bits = if base_bitpos + (j + 1) * 64 > self.size {
                self.size - (base_bitpos + j * 64)
            } else { 64 };
            let zeros_in_range = if max_bits < 64 {
                ((!word) & ((1u64 << max_bits) - 1)).count_ones() as usize
            } else { zeros };

            if remaining < zeros_in_range {
                return Ok(base_bitpos + j * 64 + Self::select_in_word(!word, remaining));
            }
            remaining -= zeros_in_range;
        }
        Err(ZiporaError::invalid_data("select0 internal error"))
    }

    fn len(&self) -> usize { self.size }
    fn count_ones(&self) -> usize { self.max_rank1 }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.size { return None; }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if word_idx < self.bits.len() {
            Some((self.bits[word_idx] >> bit_idx) & 1 == 1)
        } else {
            Some(false)
        }
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.size == 0 { return 0.0; }
        let bit_bytes = (self.size + 7) / 8;
        let cache_bytes = self.rank_cache.len() * 4;
        (cache_bytes as f64 / bit_bytes as f64) * 100.0
    }
}

impl RankSelectSimple {
    /// Select the k-th set bit within a word (0-indexed).
    #[inline]
    fn select_in_word(word: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                // pdep(1 << k, word) gives a mask with only the k-th set bit
                // tzcnt gives its position
                let mask = 1u64 << k;
                let deposited = unsafe { core::arch::x86_64::_pdep_u64(mask, word) };
                return deposited.trailing_zeros() as usize;
            }
        }
        // Scalar fallback
        let mut w = word;
        for _ in 0..k {
            w &= w - 1; // clear lowest set bit
        }
        w.trailing_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rs(pattern: &[bool]) -> RankSelectSimple {
        let mut bv = BitVector::new();
        for &b in pattern { bv.push(b).unwrap(); }
        RankSelectSimple::new(bv).unwrap()
    }

    #[test]
    fn test_basic() {
        let rs = make_rs(&[true, false, true, false, true]);
        assert_eq!(rs.len(), 5);
        assert_eq!(rs.count_ones(), 3);
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank1(1), 1);
        assert_eq!(rs.rank1(3), 2);
        assert_eq!(rs.rank1(5), 3);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 2);
        assert_eq!(rs.select1(2).unwrap(), 4);
    }

    #[test]
    fn test_invariant() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 7 == 0).collect();
        let rs = make_rs(&pattern);
        for i in 0..=rs.len() {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i, "invariant failed at {}", i);
        }
    }

    #[test]
    fn test_roundtrip() {
        let pattern: Vec<bool> = (0..500).map(|i| i % 5 == 0).collect();
        let rs = make_rs(&pattern);
        for k in 0..rs.count_ones() {
            let pos = rs.select1(k).unwrap();
            assert_eq!(rs.get(pos), Some(true), "select1({}) = {} should be set", k, pos);
            // rank1(pos+1) should be k+1 (rank counts bits BEFORE pos)
            assert_eq!(rs.rank1(pos + 1), k + 1);
        }
    }

    #[test]
    fn test_empty() {
        let rs = make_rs(&[]);
        assert_eq!(rs.len(), 0);
        assert_eq!(rs.rank1(0), 0);
        assert!(rs.select1(0).is_err());
    }

    #[test]
    fn test_all_zeros() {
        let rs = make_rs(&vec![false; 100]);
        assert_eq!(rs.count_ones(), 0);
        assert!(rs.select1(0).is_err());
        assert_eq!(rs.select0(50).unwrap(), 50);
    }

    #[test]
    fn test_all_ones() {
        let rs = make_rs(&vec![true; 100]);
        assert_eq!(rs.count_ones(), 100);
        assert_eq!(rs.select1(50).unwrap(), 50);
        assert!(rs.select0(0).is_err());
    }

    #[test]
    fn test_large() {
        let pattern: Vec<bool> = (0..10000).map(|i| i % 13 == 0).collect();
        let rs = make_rs(&pattern);
        let expected_ones = (0..10000).filter(|i| i % 13 == 0).count();
        assert_eq!(rs.count_ones(), expected_ones);
        // Spot check
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 13);
        assert_eq!(rs.rank1(13), 1);  // bit 0 set before pos 13
        assert_eq!(rs.rank1(14), 2);  // bits 0,13 set before pos 14
        assert_eq!(rs.rank1(26), 2);  // bits 0,13 set before pos 26
    }

    #[test]
    fn test_select0() {
        let pattern: Vec<bool> = (0..100).map(|i| i % 3 == 0).collect();
        let rs = make_rs(&pattern);
        // First zero is at position 1
        assert_eq!(rs.select0(0).unwrap(), 1);
        // Second zero is at position 2
        assert_eq!(rs.select0(1).unwrap(), 2);
    }
}
