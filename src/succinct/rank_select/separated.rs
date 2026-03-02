//! RankSelectSE256: Side-entry rank/select with 256-bit blocks.
//!
//! Port of `rank_select_se` (rank_select_se_256) from topling-zip.
//! "_se" = "separated": bit data is stored separately from rank index.
//!
//! Key advantage over interleaved: rank cache doesn't pollute bit-data cache lines,
//! better for large bitvectors where you scan bits and do occasional rank queries.
//!
//! Layout:
//! - `bits: Vec<u64>` — raw bit words
//! - `rank_cache: Vec<RankCacheSE>` — one 8-byte entry per 256-bit block
//! - Optional `sel0_cache` / `sel1_cache` for accelerated select

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 256;
const WORDS_PER_LINE: usize = LINE_BITS / 64; // 4

/// Rank cache entry for SE256: 8 bytes per 256-bit block.
/// Matches topling-zip's `RankCache { uint32_t lev1; uint8_t lev2[4]; }`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct RankCacheSE {
    /// Cumulative rank1 at the start of this block.
    lev1: u32,
    /// Sub-block cumulative popcount within this block.
    /// lev2[j] = popcount of words 0..j within this block.
    /// lev2[0] is always 0.
    lev2: [u8; 4],
}

impl RankCacheSE {
    fn new(lev1: u32) -> Self {
        Self { lev1, lev2: [0; 4] }
    }
}

/// Side-entry rank/select with 256-bit blocks.
pub struct RankSelectSE256 {
    bits: Vec<u64>,
    rank_cache: Vec<RankCacheSE>,
    sel0_cache: Option<Vec<u32>>,
    sel1_cache: Option<Vec<u32>>,
    size: usize,
    max_rank0: usize,
    max_rank1: usize,
}

impl RankSelectSE256 {
    /// Build from a BitVector with optional select acceleration.
    pub fn new(bv: BitVector) -> Result<Self> {
        Self::with_options(bv, true, true)
    }

    /// Build with control over select acceleration tables.
    pub fn with_options(bv: BitVector, speed_select0: bool, speed_select1: bool) -> Result<Self> {
        let size = bv.len();
        let bits: Vec<u64> = bv.blocks().to_vec();
        let nlines = (size + LINE_BITS - 1) / LINE_BITS;

        // Build rank cache
        let mut rank_cache = Vec::with_capacity(nlines + 1);
        let mut cumulative = 0u32;
        for i in 0..nlines {
            let mut rc = RankCacheSE::new(cumulative);
            let mut r = 0u32;
            for j in 0..WORDS_PER_LINE {
                rc.lev2[j] = r as u8;
                let word_idx = i * WORDS_PER_LINE + j;
                if word_idx < bits.len() {
                    r += bits[word_idx].count_ones();
                }
            }
            rank_cache.push(rc);
            cumulative += r;
        }
        rank_cache.push(RankCacheSE::new(cumulative)); // sentinel

        let max_rank1 = cumulative as usize;
        let max_rank0 = size - max_rank1;

        // Build select acceleration tables
        let sel0_cache = if speed_select0 && max_rank0 > 0 {
            Some(Self::build_select0_cache(&rank_cache, max_rank0, nlines))
        } else { None };

        let sel1_cache = if speed_select1 && max_rank1 > 0 {
            Some(Self::build_select1_cache(&rank_cache, max_rank1, nlines))
        } else { None };

        Ok(Self { bits, rank_cache, sel0_cache, sel1_cache, size, max_rank0, max_rank1 })
    }

    /// Build select0 acceleration: sel0_cache[r/256] = first block where cumulative rank0 >= r
    fn build_select0_cache(rank_cache: &[RankCacheSE], max_rank0: usize, nlines: usize) -> Vec<u32> {
        let slots = (max_rank0 + LINE_BITS - 1) / LINE_BITS;
        let mut cache = vec![0u32; slots + 1];
        cache[0] = 0;
        for j in 1..slots {
            let mut k = cache[j - 1] as usize;
            while k < nlines && k * LINE_BITS - rank_cache[k].lev1 as usize <= LINE_BITS * j {
                k += 1;
            }
            cache[j] = k as u32;
        }
        cache[slots] = nlines as u32;
        cache
    }

    /// Build select1 acceleration: sel1_cache[r/256] = first block where cumulative rank1 >= r
    fn build_select1_cache(rank_cache: &[RankCacheSE], max_rank1: usize, nlines: usize) -> Vec<u32> {
        let slots = (max_rank1 + LINE_BITS - 1) / LINE_BITS;
        let mut cache = vec![0u32; slots + 1];
        cache[0] = 0;
        for j in 1..slots {
            let mut k = cache[j - 1] as usize;
            while k < nlines && (rank_cache[k].lev1 as usize) < LINE_BITS * j {
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

    /// Select k-th set bit within a u64 word.
    #[inline]
    fn select_in_word(word: u64, k: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("bmi2") {
                let mask = 1u64 << k;
                let deposited = unsafe { core::arch::x86_64::_pdep_u64(mask, word) };
                return deposited.trailing_zeros() as usize;
            }
        }
        let mut w = word;
        for _ in 0..k { w &= w - 1; }
        w.trailing_zeros() as usize
    }

    pub fn max_rank0(&self) -> usize { self.max_rank0 }
    pub fn max_rank1(&self) -> usize { self.max_rank1 }

    pub fn mem_size(&self) -> usize {
        self.bits.len() * 8
        + self.rank_cache.len() * std::mem::size_of::<RankCacheSE>()
        + self.sel0_cache.as_ref().map_or(0, |c| c.len() * 4)
        + self.sel1_cache.as_ref().map_or(0, |c| c.len() * 4)
    }

    /// Prefetch the rank cache entry for a given bit position.
    #[inline]
    pub fn prefetch_rank1(&self, bitpos: usize) {
        let idx = bitpos / LINE_BITS;
        if idx < self.rank_cache.len() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    &self.rank_cache[idx] as *const RankCacheSE as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }

    /// Find the block containing the k-th zero bit using select0 cache + binary search.
    fn select0_upper_bound(&self, rank0: usize) -> usize {
        let (mut lo, mut hi) = if let Some(cache) = &self.sel0_cache {
            let slot = rank0 / LINE_BITS;
            (cache[slot] as usize, cache[slot + 1] as usize)
        } else {
            (0, self.rank_cache.len() - 1)
        };
        while lo < hi {
            let mid = (lo + hi) / 2;
            let rank0_at_mid = mid * LINE_BITS - self.rank_cache[mid].lev1 as usize;
            if rank0_at_mid <= rank0 { lo = mid + 1; } else { hi = mid; }
        }
        lo
    }

    /// Find the block containing the k-th one bit using select1 cache + binary search.
    fn select1_upper_bound(&self, rank1: usize) -> usize {
        let (mut lo, mut hi) = if let Some(cache) = &self.sel1_cache {
            let slot = rank1 / LINE_BITS;
            (cache[slot] as usize, cache[slot + 1] as usize)
        } else {
            (0, self.rank_cache.len() - 1)
        };
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.rank_cache[mid].lev1 as usize) <= rank1 { lo = mid + 1; } else { hi = mid; }
        }
        lo
    }
}

impl RankSelectOps for RankSelectSE256 {
    /// O(1) rank1: two memory accesses (rank cache + bit word).
    #[inline(always)]
    fn rank1(&self, bitpos: usize) -> usize {
        assert!(bitpos <= self.size);
        if bitpos == 0 { return 0; }
        let block = bitpos / LINE_BITS;
        let rc = &self.rank_cache[block];
        let word_in_block = (bitpos / 64) % WORDS_PER_LINE;
        let bit_in_word = bitpos % 64;
        let word_idx = bitpos / 64;

        rc.lev1 as usize
            + rc.lev2[word_in_block] as usize
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
        let lo = self.select1_upper_bound(k);
        assert!(lo > 0);
        let block = lo - 1;
        let rc = &self.rank_cache[block];
        let base_rank1 = rc.lev1 as usize;
        let base_bitpos = block * LINE_BITS;

        // Find the word within the block using lev2
        let remaining_in_block = k - base_rank1;
        for j in (0..WORDS_PER_LINE).rev() {
            if remaining_in_block >= rc.lev2[j] as usize {
                let remaining_in_word = remaining_in_block - rc.lev2[j] as usize;
                let word_idx = block * WORDS_PER_LINE + j;
                if word_idx < self.bits.len() {
                    let word = self.bits[word_idx];
                    return Ok(base_bitpos + j * 64 + Self::select_in_word(word, remaining_in_word));
                }
            }
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    fn select0(&self, k: usize) -> Result<usize> {
        if k >= self.max_rank0 {
            return Err(ZiporaError::invalid_data("select0 out of range"));
        }
        let lo = self.select0_upper_bound(k);
        assert!(lo > 0);
        let block = lo - 1;
        let rc = &self.rank_cache[block];
        let base_rank0 = block * LINE_BITS - rc.lev1 as usize;
        let base_bitpos = block * LINE_BITS;

        let remaining = k - base_rank0;
        // Find word using inverted lev2: zeros_in_block_up_to_j = j*64 - lev2[j]
        for j in (0..WORDS_PER_LINE).rev() {
            let zeros_before_j = j * 64 - rc.lev2[j] as usize;
            if remaining >= zeros_before_j {
                let remaining_in_word = remaining - zeros_before_j;
                let word_idx = block * WORDS_PER_LINE + j;
                let word = if word_idx < self.bits.len() { self.bits[word_idx] } else { 0 };
                return Ok(base_bitpos + j * 64 + Self::select_in_word(!word, remaining_in_word));
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

impl std::fmt::Debug for RankSelectSE256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectSE256")
            .field("size", &self.size)
            .field("max_rank1", &self.max_rank1)
            .field("has_sel0", &self.sel0_cache.is_some())
            .field("has_sel1", &self.sel1_cache.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rs(pattern: &[bool]) -> RankSelectSE256 {
        let mut bv = BitVector::new();
        for &b in pattern { bv.push(b).unwrap(); }
        RankSelectSE256::new(bv).unwrap()
    }

    #[test]
    fn test_basic() {
        let rs = make_rs(&[true, false, true, false, true]);
        assert_eq!(rs.len(), 5);
        assert_eq!(rs.count_ones(), 3);
        assert_eq!(rs.rank1(1), 1);
        assert_eq!(rs.rank1(3), 2);
        assert_eq!(rs.rank1(5), 3);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 2);
        assert_eq!(rs.select1(2).unwrap(), 4);
    }

    #[test]
    fn test_invariant() {
        let pattern: Vec<bool> = (0..2000).map(|i| i % 7 == 0).collect();
        let rs = make_rs(&pattern);
        for i in 0..=rs.len() {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i, "invariant failed at {}", i);
        }
    }

    #[test]
    fn test_roundtrip() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 5 == 0).collect();
        let rs = make_rs(&pattern);
        for k in 0..rs.count_ones() {
            let pos = rs.select1(k).unwrap();
            assert_eq!(rs.get(pos), Some(true));
            assert_eq!(rs.rank1(pos + 1), k + 1);
        }
    }

    #[test]
    fn test_select0_roundtrip() {
        let pattern: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let rs = make_rs(&pattern);
        let zeros = rs.len() - rs.count_ones();
        for k in 0..zeros {
            let pos = rs.select0(k).unwrap();
            assert_eq!(rs.get(pos), Some(false), "select0({}) = {} should be zero", k, pos);
        }
    }

    #[test]
    fn test_large_dataset() {
        let pattern: Vec<bool> = (0..10000).map(|i| i % 17 == 0).collect();
        let rs = make_rs(&pattern);
        let expected_ones = (0..10000).filter(|i| i % 17 == 0).count();
        assert_eq!(rs.count_ones(), expected_ones);
        // Spot-check
        for k in [0, 10, 50, 100, expected_ones - 1] {
            let pos = rs.select1(k).unwrap();
            assert_eq!(pos, k * 17);
        }
    }

    #[test]
    fn test_empty() {
        let rs = make_rs(&[]);
        assert_eq!(rs.len(), 0);
        assert_eq!(rs.rank1(0), 0);
    }

    #[test]
    fn test_crossing_block_boundaries() {
        // Create pattern that exercises block boundaries (256 bits)
        let mut pattern = vec![false; 300];
        pattern[0] = true;
        pattern[255] = true; // last bit of first block
        pattern[256] = true; // first bit of second block
        pattern[299] = true;
        let rs = make_rs(&pattern);
        assert_eq!(rs.count_ones(), 4);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 255);
        assert_eq!(rs.select1(2).unwrap(), 256);
        assert_eq!(rs.select1(3).unwrap(), 299);
    }

    #[test]
    fn test_dense() {
        // Every bit set — worst case for select0
        let rs = make_rs(&vec![true; 1000]);
        assert_eq!(rs.count_ones(), 1000);
        assert_eq!(rs.select1(500).unwrap(), 500);
        assert!(rs.select0(0).is_err());
    }

    #[test]
    fn test_sparse() {
        // Only first and last bits set
        let mut pattern = vec![false; 10000];
        pattern[0] = true;
        pattern[9999] = true;
        let rs = make_rs(&pattern);
        assert_eq!(rs.count_ones(), 2);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 9999);
    }
}
