//! RankSelectSE512: Side-entry rank/select with 512-bit blocks.
//!
//! Port of `rank_select_se_512_tpl<T>` from topling-zip.
//! Uses 512-bit blocks (8 × 64-bit words) with packed 9-bit sub-block ranks.
//!
//! Sub-block ranks: 7 × 9 bits packed into a u64 (`rela` field).
//! `rela` stores cumulative popcount within the block for words 1-7
//! (word 0 is implicitly 0). Each 9-bit field holds 0-512.
//!
//! Parameterized on index type:
//! - `u32`: max ~4 billion bits (RankSelectSE512_32)
//! - `u64`: unlimited (RankSelectSE512_64)

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 512;
const WORDS_PER_LINE: usize = LINE_BITS / 64; // 8

/// Rank cache for SE512: base + packed 9-bit relative ranks.
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
struct RankCacheSE512 {
    base: u32,  // cumulative rank1 at block start
    rela: u64,  // packed 9-bit sub-block ranks for words 1-7
}

/// Extract the k-th 9-bit sub-rank from the packed `rela` field.
/// k=0 returns 0 (word 0 offset is always 0, not stored).
/// k=1..7 extracts bits [(k-1)*9 .. (k-1)*9+9) from rela.
#[inline(always)]
fn get_rela(rela: u64, k: usize) -> usize {
    if k == 0 { return 0; }
    ((rela >> ((k - 1) * 9)) & 0x1FF) as usize
}

/// Side-entry rank/select with 512-bit blocks and u32 index.
pub struct RankSelectSE512 {
    bits: Vec<u64>,
    rank_cache: Vec<RankCacheSE512>,
    sel0_cache: Option<Vec<u32>>,
    sel1_cache: Option<Vec<u32>>,
    size: usize,
    max_rank0: usize,
    max_rank1: usize,
}

/// Type aliases matching topling-zip naming.
pub type RankSelectSE512_32 = RankSelectSE512;

impl RankSelectSE512 {
    /// Build from a BitVector with select acceleration.
    pub fn new(bv: BitVector) -> Result<Self> {
        Self::with_options(bv, true, true)
    }

    pub fn with_options(bv: BitVector, speed_select0: bool, speed_select1: bool) -> Result<Self> {
        let size = bv.len();
        let bits: Vec<u64> = bv.blocks().to_vec();
        let nlines = (size + LINE_BITS - 1) / LINE_BITS;

        // Build rank cache
        let mut rank_cache = Vec::with_capacity(nlines + 1);
        let mut cumulative = 0u32;
        for i in 0..nlines {
            let mut rela = 0u64;
            let mut r = 0u64;
            for j in 0..WORDS_PER_LINE {
                let word_idx = i * WORDS_PER_LINE + j;
                let pc = if word_idx < bits.len() { bits[word_idx].count_ones() as u64 } else { 0 };
                r += pc;
                // Pack cumulative rank into rela (9 bits per sub-block)
                // rela stores ranks for words 0..j (not including j's popcount yet for word j+1)
                rela |= r << (j * 9);
            }
            // Clear the unused highest bit (bit 63) — topling-zip does `rela &= u64::MAX >> 1`
            rela &= u64::MAX >> 1;
            rank_cache.push(RankCacheSE512 { base: cumulative, rela });
            cumulative += r as u32;
        }
        rank_cache.push(RankCacheSE512 { base: cumulative, rela: 0 }); // sentinel

        let max_rank1 = cumulative as usize;
        let max_rank0 = size - max_rank1;

        let sel0_cache = if speed_select0 && max_rank0 > 0 {
            Some(Self::build_select_cache(&rank_cache, max_rank0, nlines, false))
        } else { None };

        let sel1_cache = if speed_select1 && max_rank1 > 0 {
            Some(Self::build_select_cache(&rank_cache, max_rank1, nlines, true))
        } else { None };

        Ok(Self { bits, rank_cache, sel0_cache, sel1_cache, size, max_rank0, max_rank1 })
    }

    fn build_select_cache(rank_cache: &[RankCacheSE512], max_rank: usize, nlines: usize, is_rank1: bool) -> Vec<u32> {
        let slots = (max_rank + LINE_BITS - 1) / LINE_BITS;
        let mut cache = vec![0u32; slots + 1];
        cache[0] = 0;
        for j in 1..slots {
            let mut k = cache[j - 1] as usize;
            while k < nlines {
                let rank_at_k = if is_rank1 {
                    rank_cache[k].base as usize
                } else {
                    k * LINE_BITS - rank_cache[k].base as usize
                };
                if (is_rank1 && rank_at_k >= LINE_BITS * j) ||
                   (!is_rank1 && rank_at_k > LINE_BITS * j) {
                    break;
                }
                k += 1;
            }
            cache[j] = k as u32;
        }
        cache[slots] = nlines as u32;
        cache
    }

    #[inline(always)]
    fn popcount_trail(word: u64, bit_count: usize) -> usize {
        if bit_count == 0 { return 0; }
        if bit_count >= 64 { return word.count_ones() as usize; }
        (word & ((1u64 << bit_count) - 1)).count_ones() as usize
    }

    #[inline]
    fn select_in_word(word: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                let deposited = unsafe { core::arch::x86_64::_pdep_u64(1u64 << k, word) };
                return deposited.trailing_zeros() as usize;
            }
        }
        let mut w = word;
        for _ in 0..k { w &= w - 1; }
        w.trailing_zeros() as usize
    }

    fn upper_bound(&self, rank: usize, is_rank1: bool) -> usize {
        let cache = if is_rank1 { &self.sel1_cache } else { &self.sel0_cache };
        let (mut lo, mut hi) = if let Some(c) = cache {
            let slot = rank / LINE_BITS;
            (c[slot] as usize, c[slot + 1] as usize)
        } else {
            (0, self.rank_cache.len() - 1)
        };
        while lo < hi {
            let mid = (lo + hi) / 2;
            let val = if is_rank1 { self.rank_cache[mid].base as usize }
                      else { mid * LINE_BITS - self.rank_cache[mid].base as usize };
            if val <= rank { lo = mid + 1; } else { hi = mid; }
        }
        lo
    }

    pub fn max_rank0(&self) -> usize { self.max_rank0 }
    pub fn max_rank1(&self) -> usize { self.max_rank1 }

    pub fn mem_size(&self) -> usize {
        self.bits.len() * 8
        + self.rank_cache.len() * std::mem::size_of::<RankCacheSE512>()
        + self.sel0_cache.as_ref().map_or(0, |c| c.len() * 4)
        + self.sel1_cache.as_ref().map_or(0, |c| c.len() * 4)
    }
}

impl RankSelectOps for RankSelectSE512 {
    /// O(1) rank1 using packed 9-bit sub-block ranks.
    #[inline(always)]
    fn rank1(&self, bitpos: usize) -> usize {
        assert!(bitpos <= self.size);
        if bitpos == 0 { return 0; }
        let block = bitpos / LINE_BITS;
        let rc = self.rank_cache[block];
        let k = (bitpos % LINE_BITS) / 64; // word index within block (0-7)
        let word_idx = bitpos / 64;
        let bit_in_word = bitpos % 64;

        rc.base as usize
            + get_rela(rc.rela, k)
            + Self::popcount_trail(
                if word_idx < self.bits.len() { self.bits[word_idx] } else { 0 },
                bit_in_word,
            )
    }

    #[inline(always)]
    fn rank0(&self, pos: usize) -> usize { pos - self.rank1(pos) }

    fn select1(&self, k: usize) -> Result<usize> {
        if k >= self.max_rank1 {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        let lo = self.upper_bound(k, true);
        assert!(lo > 0);
        let block = lo - 1;
        let rc = self.rank_cache[block];
        let hit = rc.base as usize;
        let base_bitpos = block * LINE_BITS;

        // Binary tree search through 8 words using packed rela
        let target = k - hit;
        for j in (0..WORDS_PER_LINE).rev() {
            let rank_before_j = get_rela(rc.rela, j);
            if target >= rank_before_j {
                let remaining = target - rank_before_j;
                let word_idx = block * WORDS_PER_LINE + j;
                if word_idx < self.bits.len() {
                    return Ok(base_bitpos + j * 64 + Self::select_in_word(self.bits[word_idx], remaining));
                }
            }
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    fn select0(&self, k: usize) -> Result<usize> {
        if k >= self.max_rank0 {
            return Err(ZiporaError::invalid_data("select0 out of range"));
        }
        let lo = self.upper_bound(k, false);
        assert!(lo > 0);
        let block = lo - 1;
        let rc = self.rank_cache[block];
        let hit = block * LINE_BITS - rc.base as usize; // rank0 at block start
        let base_bitpos = block * LINE_BITS;

        let target = k - hit;
        for j in (0..WORDS_PER_LINE).rev() {
            let rank1_before_j = get_rela(rc.rela, j);
            let zeros_before_j = j * 64 - rank1_before_j;
            if target >= zeros_before_j {
                let remaining = target - zeros_before_j;
                let word_idx = block * WORDS_PER_LINE + j;
                let word = if word_idx < self.bits.len() { self.bits[word_idx] } else { 0 };
                return Ok(base_bitpos + j * 64 + Self::select_in_word(!word, remaining));
            }
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
        let overhead = self.mem_size() - bit_bytes;
        (overhead as f64 / bit_bytes as f64) * 100.0
    }
}

impl std::fmt::Debug for RankSelectSE512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectSE512")
            .field("size", &self.size)
            .field("max_rank1", &self.max_rank1)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rs(pattern: &[bool]) -> RankSelectSE512 {
        let mut bv = BitVector::new();
        for &b in pattern { bv.push(b).unwrap(); }
        RankSelectSE512::new(bv).unwrap()
    }

    #[test]
    fn test_basic() {
        let rs = make_rs(&[true, false, true, false, true]);
        assert_eq!(rs.len(), 5);
        assert_eq!(rs.count_ones(), 3);
        assert_eq!(rs.rank1(1), 1);
        assert_eq!(rs.rank1(5), 3);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(2).unwrap(), 4);
    }

    #[test]
    fn test_invariant() {
        let pattern: Vec<bool> = (0..3000).map(|i| i % 7 == 0).collect();
        let rs = make_rs(&pattern);
        for i in 0..=rs.len() {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i, "invariant at {}", i);
        }
    }

    #[test]
    fn test_roundtrip() {
        let pattern: Vec<bool> = (0..2000).map(|i| i % 5 == 0).collect();
        let rs = make_rs(&pattern);
        for k in 0..rs.count_ones() {
            let pos = rs.select1(k).unwrap();
            assert_eq!(rs.get(pos), Some(true));
        }
    }

    #[test]
    fn test_crossing_512_boundary() {
        let mut pattern = vec![false; 600];
        pattern[0] = true;
        pattern[511] = true;  // last bit of first 512-bit block
        pattern[512] = true;  // first bit of second block
        pattern[599] = true;
        let rs = make_rs(&pattern);
        assert_eq!(rs.count_ones(), 4);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 511);
        assert_eq!(rs.select1(2).unwrap(), 512);
        assert_eq!(rs.select1(3).unwrap(), 599);
    }

    #[test]
    fn test_large() {
        let pattern: Vec<bool> = (0..10000).map(|i| i % 13 == 0).collect();
        let rs = make_rs(&pattern);
        let expected = (0..10000).filter(|i| i % 13 == 0).count();
        assert_eq!(rs.count_ones(), expected);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 13);
    }

    #[test]
    fn test_get_rela() {
        // Pack known values
        let mut rela = 0u64;
        rela |= 5 << (0 * 9);  // word 1: 5
        rela |= 12 << (1 * 9); // word 2: 12
        rela |= 20 << (2 * 9); // word 3: 20
        assert_eq!(get_rela(rela, 0), 0);
        assert_eq!(get_rela(rela, 1), 5);
        assert_eq!(get_rela(rela, 2), 12);
        assert_eq!(get_rela(rela, 3), 20);
    }

    #[test]
    fn test_empty() {
        let rs = make_rs(&[]);
        assert_eq!(rs.len(), 0);
        assert_eq!(rs.rank1(0), 0);
    }

    #[test]
    fn test_select0() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let rs = make_rs(&pattern);
        assert_eq!(rs.select0(0).unwrap(), 1);
        assert_eq!(rs.select0(1).unwrap(), 2);
    }
}
