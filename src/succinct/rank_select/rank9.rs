//! Rank9: Fast rank/select with 512-bit blocks and 25% space overhead.
//!
//! Based on Vigna's "Broadword Implementation of Rank/Select Queries" (2008).
//!
//! Layout per 512-bit block (16 bytes):
//! - `base`: u64 — cumulative popcount up to this block
//! - `sub`: u64 — 7 packed 9-bit sub-block counts within the block
//!   sub_k = popcount of words 0..k within the block (k = 1..7)
//!
//! Rank is O(1): one lookup in the block index + one popcount of a partial word.
//! Select uses binary search on blocks + select_in_word within a block.

use super::RankSelectOps;
use crate::algorithms::bit_ops::select_in_word;
use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;

/// Words per block. 8 words × 64 bits = 512 bits per block.
const WORDS_PER_BLOCK: usize = 8;
/// Bits per block.
const BITS_PER_BLOCK: usize = WORDS_PER_BLOCK * 64;

/// Rank9 index entry: base rank + packed sub-block counts.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Rank9Entry {
    /// Cumulative popcount of all bits before this block.
    base: u64,
    /// 7 packed 9-bit sub-block counts.
    /// Bits [0..9) = popcount of word 0
    /// Bits [9..18) = popcount of words 0..1
    /// ...
    /// Bits [54..63) = popcount of words 0..6
    /// (word 7 is not stored — it equals block total = next_base - base)
    sub: u64,
}

/// Fast rank/select structure with 512-bit blocks and O(1) rank.
///
/// Space overhead: 25% of the original bitvector (16 bytes per 64 bytes of data).
/// Rank: O(1) — one index lookup + one `popcnt` instruction.
/// Select: O(log n) binary search on blocks + O(1) within block.
///
/// # Examples
///
/// ```rust
/// use zipora::succinct::BitVector;
/// use zipora::succinct::rank_select::{Rank9, RankSelectOps};
///
/// let mut bv = BitVector::new();
/// for i in 0..1000 {
///     bv.push(i % 3 == 0).unwrap();
/// }
/// let r9 = Rank9::new(bv).unwrap();
///
/// // O(1) rank
/// let rank = r9.rank1(500);
/// assert_eq!(rank, 167); // floor(500/3) + 1 = 167
///
/// // Select via binary search
/// let pos = r9.select1(100).unwrap();
/// assert_eq!(pos, 300);
/// ```
pub struct Rank9 {
    /// Raw bitvector words.
    bv: BitVector,
    /// Rank9 index: one entry per 512-bit block, plus sentinel.
    index: Vec<Rank9Entry>,
    /// Total number of valid bits.
    len_bits: usize,
    /// Total number of set bits.
    total_ones: usize,
}

impl Rank9 {
    /// Build from a BitVector.
    pub fn new(bv: BitVector) -> Result<Self> {
        let len_bits = bv.len();
        let num_blocks = bv.blocks().len().div_ceil(WORDS_PER_BLOCK);

        let mut index = Vec::with_capacity(num_blocks + 1);
        let mut cumul: u64 = 0;

        for block_idx in 0..num_blocks {
            let start_word = block_idx * WORDS_PER_BLOCK;
            let mut sub: u64 = 0;
            let mut block_cumul: u64 = 0;

            for k in 0..WORDS_PER_BLOCK {
                let word_idx = start_word + k;
                let word = if word_idx < bv.blocks().len() {
                    bv.blocks()[word_idx]
                } else {
                    0
                };
                block_cumul += word.count_ones() as u64;
                // Store cumulative count after word k in sub-block position k
                // (positions 0..6 are stored; position 7 is implied)
                if k < 7 {
                    sub |= block_cumul << (k * 9);
                }
            }

            index.push(Rank9Entry { base: cumul, sub });
            cumul += block_cumul;
        }

        // Sentinel entry
        index.push(Rank9Entry {
            base: cumul,
            sub: 0,
        });

        Ok(Self {
            bv,
            index,
            len_bits,
            total_ones: cumul as usize,
        })
    }

    /// O(1) rank1: count set bits in [0, pos).
    #[inline]
    pub fn rank1_fast(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.len_bits);

        let block = pos / BITS_PER_BLOCK;
        let entry = &self.index[block];

        let word_in_block = (pos % BITS_PER_BLOCK) / 64;
        let bit_in_word = pos % 64;

        // Base rank from block
        let mut rank = entry.base as usize;

        // Add sub-block count (cumulative within block up to word_in_block)
        if word_in_block > 0 {
            rank += ((entry.sub >> ((word_in_block - 1) * 9)) & 0x1FF) as usize;
        }

        // Add popcount of partial word
        let global_word = block * WORDS_PER_BLOCK + word_in_block;
        if global_word < self.bv.blocks().len() && bit_in_word > 0 {
            let mask = (1u64 << bit_in_word) - 1;
            rank += (self.bv.blocks()[global_word] & mask).count_ones() as usize;
        }

        rank
    }

    /// Select1 via binary search on blocks.
    pub fn select1_fast(&self, k: usize) -> Result<usize> {
        if k >= self.total_ones {
            return Err(ZiporaError::invalid_data(format!(
                "select1({}) out of range (total_ones={})",
                k, self.total_ones
            )));
        }

        let target = k as u64;

        // Binary search for the block containing the k-th set bit
        let num_blocks = self.index.len() - 1;
        let mut lo = 0usize;
        let mut hi = num_blocks;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.index[mid + 1].base <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let block = lo;
        let entry = &self.index[block];
        let mut remaining = target - entry.base;
        let start_word = block * WORDS_PER_BLOCK;

        // Scan sub-blocks to find the word within the block
        let mut word_offset = 0usize;
        for j in 0..7 {
            let sub_cumul = (entry.sub >> (j * 9)) & 0x1FF;
            if sub_cumul > remaining {
                break;
            }
            word_offset = j + 1;
        }

        // Adjust remaining by sub-block count before this word
        if word_offset > 0 {
            let sub_before = (entry.sub >> ((word_offset - 1) * 9)) & 0x1FF;
            remaining -= sub_before;
        }

        let global_word = start_word + word_offset;
        if global_word < self.bv.blocks().len() {
            let word = self.bv.blocks()[global_word];
            let bit_pos = select_in_word(word, remaining as usize);
            Ok(global_word * 64 + bit_pos)
        } else {
            Err(ZiporaError::invalid_data(
                "select1 internal error: word out of range",
            ))
        }
    }

    /// Select0 via binary search on blocks.
    pub fn select0_fast(&self, k: usize) -> Result<usize> {
        let total_zeros = self.len_bits - self.total_ones;
        if k >= total_zeros {
            return Err(ZiporaError::invalid_data(format!(
                "select0({}) out of range (total_zeros={})",
                k, total_zeros
            )));
        }

        let target = k as u64;

        let num_blocks = self.index.len() - 1;
        let mut lo = 0usize;
        let mut hi = num_blocks;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let zeros_before = ((mid + 1) * BITS_PER_BLOCK) as u64 - self.index[mid + 1].base;
            if zeros_before <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        let block = lo;
        let entry = &self.index[block];
        let block_start_bit = (block * BITS_PER_BLOCK) as u64;
        let zeros_before_block = block_start_bit - entry.base;
        let mut remaining = target - zeros_before_block;
        let start_word = block * WORDS_PER_BLOCK;

        // Scan sub-blocks to find the word within the block
        let mut word_offset = 0usize;
        for j in 0..7 {
            let sub_ones = (entry.sub >> (j * 9)) & 0x1FF;
            let sub_bits = ((j + 1) * 64) as u64;
            let sub_zeros = sub_bits - sub_ones;

            if sub_zeros > remaining {
                break;
            }
            word_offset = j + 1;
        }

        // Adjust remaining by sub-block zeros before this word
        if word_offset > 0 {
            let sub_ones_before = (entry.sub >> ((word_offset - 1) * 9)) & 0x1FF;
            let sub_bits_before = (word_offset * 64) as u64;
            let sub_zeros_before = sub_bits_before - sub_ones_before;
            remaining -= sub_zeros_before;
        }

        let global_word = start_word + word_offset;
        if global_word < self.bv.blocks().len() {
            let word = !self.bv.blocks()[global_word]; // Invert word for select0
            let bit_pos = select_in_word(word, remaining as usize);
            Ok(global_word * 64 + bit_pos)
        } else {
            Err(ZiporaError::invalid_data(
                "select0 internal error: word out of range",
            ))
        }
    }

    /// Memory usage in bytes.
    pub fn mem_size(&self) -> usize {
        self.bv.blocks().len() * 8 + self.index.len() * std::mem::size_of::<Rank9Entry>()
    }
}

impl RankSelectOps for Rank9 {
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        self.rank1_fast(pos)
    }

    fn rank0(&self, pos: usize) -> usize {
        let pos = pos.min(self.len_bits);
        pos - self.rank1(pos)
    }

    fn select1(&self, k: usize) -> Result<usize> {
        self.select1_fast(k)
    }

    fn select0(&self, k: usize) -> Result<usize> {
        self.select0_fast(k)
    }

    fn len(&self) -> usize {
        self.len_bits
    }
    fn count_ones(&self) -> usize {
        self.total_ones
    }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len_bits {
            return None;
        }
        let word_idx = index / 64;
        let bit_idx = index % 64;
        if word_idx < self.bv.blocks().len() {
            Some((self.bv.blocks()[word_idx] >> bit_idx) & 1 == 1)
        } else {
            Some(false)
        }
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.len_bits == 0 {
            return 0.0;
        }
        let data_bytes = self.bv.blocks().len() * 8;
        let index_bytes = self.index.len() * std::mem::size_of::<Rank9Entry>();
        (index_bytes as f64 / data_bytes as f64) * 100.0
    }
}

impl std::fmt::Debug for Rank9 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rank9")
            .field("len_bits", &self.len_bits)
            .field("total_ones", &self.total_ones)
            .field("blocks", &(self.index.len().saturating_sub(1)))
            .field(
                "overhead",
                &format!("{:.1}%", self.space_overhead_percent()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bv(pattern: &[bool]) -> BitVector {
        let mut bv = BitVector::new();
        for &b in pattern {
            bv.push(b).unwrap();
        }
        bv
    }

    #[test]
    fn test_empty() {
        let r9 = Rank9::new(BitVector::new()).unwrap();
        assert_eq!(r9.len(), 0);
        assert_eq!(r9.count_ones(), 0);
        assert_eq!(r9.rank1(0), 0);
    }

    #[test]
    fn test_single_bit() {
        let r9 = Rank9::new(make_bv(&[true])).unwrap();
        assert_eq!(r9.len(), 1);
        assert_eq!(r9.count_ones(), 1);
        assert_eq!(r9.rank1(0), 0);
        assert_eq!(r9.rank1(1), 1);
        assert_eq!(r9.select1(0).unwrap(), 0);
    }

    #[test]
    fn test_alternating() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 2 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        assert_eq!(r9.len(), 1000);
        assert_eq!(r9.count_ones(), 500);

        // Rank invariant
        for pos in (0..=1000).step_by(50) {
            assert_eq!(
                r9.rank0(pos) + r9.rank1(pos),
                pos,
                "rank invariant at pos {}",
                pos
            );
        }

        // Select roundtrip
        for k in 0..500 {
            let pos = r9.select1(k).unwrap();
            assert_eq!(r9.rank1(pos + 1), k + 1, "select1({}) roundtrip failed", k);
        }
    }

    #[test]
    fn test_sparse() {
        let pattern: Vec<bool> = (0..10000).map(|i| i % 100 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        assert_eq!(r9.count_ones(), 100);
        assert_eq!(r9.select1(0).unwrap(), 0);
        assert_eq!(r9.select1(1).unwrap(), 100);
        assert_eq!(r9.select1(99).unwrap(), 9900);
        assert_eq!(r9.rank1(100), 1);
        assert_eq!(r9.rank1(200), 2);
    }

    #[test]
    fn test_dense() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 4 != 3).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        assert_eq!(r9.count_ones(), 750);

        for pos in (0..=1000).step_by(37) {
            assert_eq!(r9.rank0(pos) + r9.rank1(pos), pos);
        }
    }

    #[test]
    fn test_crossing_blocks() {
        // Test data crossing 512-bit block boundaries
        let pattern: Vec<bool> = (0..2000).map(|i| i % 13 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        // Verify rank at block boundaries
        for pos in (0..=2000).step_by(64) {
            let expected: usize = (0..pos).filter(|&i| i % 13 == 0).count();
            assert_eq!(r9.rank1(pos), expected, "rank1({}) failed", pos);
        }

        // Verify select roundtrip
        let total = r9.count_ones();
        for k in (0..total).step_by(10) {
            let pos = r9.select1(k).unwrap();
            assert_eq!(pos % 13, 0, "select1({}) = {} not multiple of 13", k, pos);
        }
    }

    #[test]
    fn test_all_ones() {
        let bv = BitVector::with_size(1000, true).unwrap();
        let r9 = Rank9::new(bv).unwrap();
        assert_eq!(r9.count_ones(), 1000);
        assert_eq!(r9.rank1(500), 500);
        assert_eq!(r9.select1(499).unwrap(), 499);
    }

    #[test]
    fn test_all_zeros() {
        let bv = BitVector::with_size(1000, false).unwrap();
        let r9 = Rank9::new(bv).unwrap();
        assert_eq!(r9.count_ones(), 0);
        assert_eq!(r9.rank1(500), 0);
        assert!(r9.select1(0).is_err());
    }

    #[test]
    fn test_space_overhead() {
        let pattern: Vec<bool> = (0..10000).map(|i| i % 3 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        let overhead = r9.space_overhead_percent();
        // Rank9 should have ~25% overhead
        assert!(
            overhead > 20.0 && overhead < 30.0,
            "Expected ~25% overhead, got {:.1}%",
            overhead
        );
    }

    #[test]
    fn test_get() {
        let pattern: Vec<bool> = (0..100).map(|i| i % 5 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        for i in 0..100 {
            assert_eq!(r9.get(i), Some(i % 5 == 0), "get({}) failed", i);
        }
        assert_eq!(r9.get(100), None);
    }

    #[test]
    fn test_large_scale() {
        let pattern: Vec<bool> = (0..100000).map(|i| (i * 13 + 7) % 71 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        // Verify rank at various positions
        for pos in [0, 1000, 10000, 50000, 99999, 100000] {
            let expected: usize = (0..pos).filter(|&i| (i * 13 + 7) % 71 == 0).count();
            assert_eq!(r9.rank1(pos), expected, "rank1({}) failed", pos);
        }

        // Verify select roundtrip for a sample
        let total = r9.count_ones();
        for k in (0..total).step_by(total / 20 + 1) {
            let pos = r9.select1(k).unwrap();
            assert_eq!(
                r9.rank1(pos),
                k,
                "select1({}) roundtrip: rank1({}) != {}",
                k,
                pos,
                k
            );
            assert_eq!(r9.get(pos), Some(true));
        }
    }

    #[test]
    fn test_select0() {
        let pattern: Vec<bool> = (0..100).map(|i| i % 3 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        // First zero is at position 1
        let pos = r9.select0(0).unwrap();
        assert_eq!(r9.get(pos), Some(false));

        // Verify a few select0 values
        let total_zeros = r9.len() - r9.count_ones();
        for k in (0..total_zeros).step_by(total_zeros / 5 + 1) {
            let pos = r9.select0(k).unwrap();
            assert_eq!(
                r9.get(pos),
                Some(false),
                "select0({}) = {} is not a zero bit",
                k,
                pos
            );
        }
    }

    #[test]
    fn test_select0_exhaustive_block_crossing() {
        let pattern: Vec<bool> = (0..2000).map(|i| i % 5 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();
        let total_zeros = r9.len() - r9.count_ones();

        for k in 0..total_zeros {
            let pos = r9.select0(k).unwrap();
            assert_eq!(
                r9.get(pos),
                Some(false),
                "select0({}) = {} is not a zero bit",
                k,
                pos
            );
            assert_eq!(r9.rank0(pos), k, "rank0(select0({})) != {}", k, k);
        }
        assert!(r9.select0(total_zeros).is_err());
    }

    #[test]
    fn test_select0_select1_roundtrip_large() {
        let pattern: Vec<bool> = (0..10000).map(|i| (i * 7 + 3) % 11 < 4).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        let total_ones = r9.count_ones();
        let total_zeros = r9.len() - total_ones;

        for k in (0..total_ones).step_by(total_ones / 50 + 1) {
            let pos = r9.select1(k).unwrap();
            assert_eq!(r9.rank1(pos), k);
            assert_eq!(r9.get(pos), Some(true));
        }

        for k in (0..total_zeros).step_by(total_zeros / 50 + 1) {
            let pos = r9.select0(k).unwrap();
            assert_eq!(r9.rank0(pos), k);
            assert_eq!(r9.get(pos), Some(false));
        }
    }

    #[test]
    fn test_select0_all_zeros() {
        let bv = BitVector::with_size(1024, false).unwrap();
        let r9 = Rank9::new(bv).unwrap();
        for k in 0..1024 {
            assert_eq!(r9.select0(k).unwrap(), k);
        }
    }

    #[test]
    fn test_select0_sparse_ones() {
        let pattern: Vec<bool> = (0..5000).map(|i| i % 500 == 0).collect();
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();
        let total_zeros = r9.len() - r9.count_ones();

        for k in (0..total_zeros).step_by(total_zeros / 20 + 1) {
            let pos = r9.select0(k).unwrap();
            assert_eq!(r9.get(pos), Some(false));
            assert_eq!(r9.rank0(pos), k);
        }
    }

    /// Performance comparison with RankSelectInterleaved256 — release only.
    #[test]
    fn test_rank9_performance() {
        let pattern: Vec<bool> = (0..100000).map(|i| (i * 13 + 7) % 71 == 0).collect();
        #[allow(unused_variables)]
        let r9 = Rank9::new(make_bv(&pattern)).unwrap();

        #[cfg(not(debug_assertions))]
        {
            let positions: Vec<usize> = (0..10000).map(|i| i * 10).collect();
            let iterations = 100;

            let start = std::time::Instant::now();
            let mut sink = 0usize;
            for _ in 0..iterations {
                for &pos in &positions {
                    sink += r9.rank1(pos);
                }
            }
            let elapsed = start.elapsed();
            let per_call = elapsed.as_nanos() as f64 / (iterations as f64 * positions.len() as f64);

            eprintln!(
                "Rank9 rank1: {per_call:.1}ns/call, overhead={:.1}%, [sink={sink}]",
                r9.space_overhead_percent()
            );
        }
    }
}
