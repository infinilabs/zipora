//! RankSelectMixedSE512: Two-dimension side-entry rank/select with 512-bit blocks.
//!
//! Port of `rank_select_mixed_se_512` from topling-zip.
//! Stores two independent bitvectors with words interleaved at the word level:
//! `words[i*2 + dim]`. Rank cache is separated (side-entry) with packed 9-bit
//! sub-block ranks per dimension.
//!
//! Better space efficiency than MixedIL256 (0.375 bits/bit/dim vs 2.5).

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 512;
const WORDS_PER_LINE: usize = 8;

/// Rank cache for mixed SE512: base + packed rela for each dimension.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct RankCacheMixed {
    base: [u32; 2],
    rela: [u64; 2],
}

#[inline(always)]
fn get_rela(rela: u64, k: usize) -> usize {
    if k == 0 { return 0; }
    ((rela >> ((k - 1) * 9)) & 0x1FF) as usize
}

/// Two-dimension side-entry rank/select with 512-bit blocks.
pub struct RankSelectMixedSE512 {
    /// Interleaved words: words[i*2 + dim]
    words: Vec<u64>,
    rank_cache: Vec<RankCacheMixed>,
    size: [usize; 2],
    max_rank1: [usize; 2],
}

/// Read-only view of one dimension.
pub struct MixedSE512DimView<'a> {
    parent: &'a RankSelectMixedSE512,
    dim: usize,
}

impl RankSelectMixedSE512 {
    /// Build from two BitVectors.
    pub fn new(bv0: BitVector, bv1: BitVector) -> Result<Self> {
        let size0 = bv0.len();
        let size1 = bv1.len();
        let max_size = size0.max(size1);
        let nlines = (max_size + LINE_BITS - 1) / LINE_BITS;
        let total_words = nlines * WORDS_PER_LINE * 2; // *2 for interleaving

        let b0 = bv0.blocks();
        let b1 = bv1.blocks();

        // Interleave words: [d0_w0, d1_w0, d0_w1, d1_w1, ...]
        let mut words = vec![0u64; total_words];
        for i in 0..nlines * WORDS_PER_LINE {
            words[i * 2] = if i < b0.len() { b0[i] } else { 0 };
            words[i * 2 + 1] = if i < b1.len() { b1[i] } else { 0 };
        }

        // Mask off bits beyond each dimension's size
        for d in 0..2 {
            let sz = if d == 0 { size0 } else { size1 };
            let last_word = sz / 64;
            let last_bit = sz % 64;
            if last_bit > 0 && last_word * 2 + d < words.len() {
                words[last_word * 2 + d] &= (1u64 << last_bit) - 1;
            }
            // Zero out words beyond size
            for w in (last_word + 1)..nlines * WORDS_PER_LINE {
                if w * 2 + d < words.len() {
                    words[w * 2 + d] = 0;
                }
            }
        }

        // Build rank cache
        let mut rank_cache = Vec::with_capacity(nlines + 1);
        let mut cum = [0u32; 2];
        for i in 0..nlines {
            let mut rc = RankCacheMixed { base: cum, rela: [0; 2] };
            for d in 0..2 {
                let mut r = 0u64;
                let mut rela = 0u64;
                for j in 0..WORDS_PER_LINE {
                    let word_idx = (i * WORDS_PER_LINE + j) * 2 + d;
                    let pc = words[word_idx].count_ones() as u64;
                    r += pc;
                    rela |= r << (j * 9);
                }
                rela &= u64::MAX >> 1;
                rc.rela[d] = rela;
                cum[d] += r as u32;
            }
            rank_cache.push(rc);
        }
        rank_cache.push(RankCacheMixed { base: cum, rela: [0; 2] });

        Ok(Self {
            words,
            rank_cache,
            size: [size0, size1],
            max_rank1: [cum[0] as usize, cum[1] as usize],
        })
    }

    pub fn dim0(&self) -> MixedSE512DimView<'_> { MixedSE512DimView { parent: self, dim: 0 } }
    pub fn dim1(&self) -> MixedSE512DimView<'_> { MixedSE512DimView { parent: self, dim: 1 } }

    #[inline]
    pub fn rank1_dim(&self, dim: usize, pos: usize) -> usize {
        assert!(pos <= self.size[dim]);
        if pos == 0 { return 0; }
        let block = pos / LINE_BITS;
        let rc = &self.rank_cache[block];
        let k = (pos % LINE_BITS) / 64;
        let word_idx = (pos / 64) * 2 + dim;
        let bit_in_word = pos % 64;
        let word = if word_idx < self.words.len() { self.words[word_idx] } else { 0 };
        let tail = if bit_in_word > 0 {
            (word & ((1u64 << bit_in_word) - 1)).count_ones() as usize
        } else { 0 };
        rc.base[dim] as usize + get_rela(rc.rela[dim], k) + tail
    }

    #[inline]
    pub fn rank0_dim(&self, dim: usize, pos: usize) -> usize { pos - self.rank1_dim(dim, pos) }

    pub fn select1_dim(&self, dim: usize, k: usize) -> Result<usize> {
        if k >= self.max_rank1[dim] {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        let nlines = self.rank_cache.len() - 1;
        let mut lo = 0usize;
        let mut hi = nlines;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.rank_cache[mid].base[dim] as usize) <= k { lo = mid + 1; } else { hi = mid; }
        }
        let block = lo - 1;
        let rc = &self.rank_cache[block];
        let hit = rc.base[dim] as usize;
        let target = k - hit;
        let base_bitpos = block * LINE_BITS;

        for j in (0..WORDS_PER_LINE).rev() {
            let rank_before_j = get_rela(rc.rela[dim], j);
            if target >= rank_before_j {
                let remaining = target - rank_before_j;
                let word_idx = (block * WORDS_PER_LINE + j) * 2 + dim;
                if word_idx < self.words.len() {
                    return Ok(base_bitpos + j * 64 + select_in_word(self.words[word_idx], remaining));
                }
            }
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    pub fn get_dim(&self, dim: usize, index: usize) -> Option<bool> {
        if index >= self.size[dim] { return None; }
        let word_idx = (index / 64) * 2 + dim;
        let bit_idx = index % 64;
        if word_idx < self.words.len() {
            Some((self.words[word_idx] >> bit_idx) & 1 == 1)
        } else { Some(false) }
    }

    pub fn mem_size(&self) -> usize {
        self.words.len() * 8 + self.rank_cache.len() * std::mem::size_of::<RankCacheMixed>()
    }
}

impl RankSelectOps for MixedSE512DimView<'_> {
    fn rank1(&self, pos: usize) -> usize { self.parent.rank1_dim(self.dim, pos) }
    fn rank0(&self, pos: usize) -> usize { self.parent.rank0_dim(self.dim, pos) }
    fn select1(&self, k: usize) -> Result<usize> { self.parent.select1_dim(self.dim, k) }
    fn select0(&self, _k: usize) -> Result<usize> {
        Err(ZiporaError::invalid_data("select0 not implemented for mixed_se_512"))
    }
    fn len(&self) -> usize { self.parent.size[self.dim] }
    fn count_ones(&self) -> usize { self.parent.max_rank1[self.dim] }
    fn get(&self, index: usize) -> Option<bool> { self.parent.get_dim(self.dim, index) }
    fn space_overhead_percent(&self) -> f64 { 0.0 }
}

#[inline]
fn select_in_word(word: u64, k: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            let d = unsafe { core::arch::x86_64::_pdep_u64(1u64 << k, word) };
            return d.trailing_zeros() as usize;
        }
    }
    let mut w = word;
    for _ in 0..k { w &= w - 1; }
    w.trailing_zeros() as usize
}

impl std::fmt::Debug for RankSelectMixedSE512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectMixedSE512")
            .field("size", &self.size)
            .field("max_rank1", &self.max_rank1)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bv(pattern: &[bool]) -> BitVector {
        let mut bv = BitVector::new();
        for &b in pattern { bv.push(b).unwrap(); }
        bv
    }

    #[test]
    fn test_basic() {
        let bv0 = make_bv(&[true, false, true, false, true]);
        let bv1 = make_bv(&[false, true, false, true, false]);
        let rs = RankSelectMixedSE512::new(bv0, bv1).unwrap();

        assert_eq!(rs.dim0().count_ones(), 3);
        assert_eq!(rs.dim1().count_ones(), 2);
        assert_eq!(rs.dim0().rank1(5), 3);
        assert_eq!(rs.dim1().rank1(4), 2);
        assert_eq!(rs.dim0().select1(0).unwrap(), 0);
        assert_eq!(rs.dim1().select1(0).unwrap(), 1);
    }

    #[test]
    fn test_invariant() {
        let p0: Vec<bool> = (0..2000).map(|i| i % 5 == 0).collect();
        let p1: Vec<bool> = (0..2000).map(|i| i % 7 == 0).collect();
        let rs = RankSelectMixedSE512::new(make_bv(&p0), make_bv(&p1)).unwrap();
        for i in 0..=2000 {
            assert_eq!(rs.dim0().rank0(i) + rs.dim0().rank1(i), i, "dim0 at {}", i);
            assert_eq!(rs.dim1().rank0(i) + rs.dim1().rank1(i), i, "dim1 at {}", i);
        }
    }

    #[test]
    fn test_roundtrip() {
        let p0: Vec<bool> = (0..1000).map(|i| i % 3 == 0).collect();
        let p1: Vec<bool> = (0..1000).map(|i| i % 11 == 0).collect();
        let rs = RankSelectMixedSE512::new(make_bv(&p0), make_bv(&p1)).unwrap();
        for k in 0..rs.dim0().count_ones() {
            let pos = rs.dim0().select1(k).unwrap();
            assert_eq!(rs.get_dim(0, pos), Some(true));
        }
        for k in 0..rs.dim1().count_ones() {
            let pos = rs.dim1().select1(k).unwrap();
            assert_eq!(rs.get_dim(1, pos), Some(true));
        }
    }

    #[test]
    fn test_large_crossing_blocks() {
        let p0: Vec<bool> = (0..5000).map(|i| i % 13 == 0).collect();
        let p1: Vec<bool> = (0..5000).map(|i| i % 17 == 0).collect();
        let rs = RankSelectMixedSE512::new(make_bv(&p0), make_bv(&p1)).unwrap();
        assert_eq!(rs.dim0().select1(0).unwrap(), 0);
        assert_eq!(rs.dim0().select1(1).unwrap(), 13);
        assert_eq!(rs.dim1().select1(0).unwrap(), 0);
        assert_eq!(rs.dim1().select1(1).unwrap(), 17);
    }

    #[test]
    fn test_empty() {
        let rs = RankSelectMixedSE512::new(make_bv(&[]), make_bv(&[])).unwrap();
        assert_eq!(rs.dim0().len(), 0);
        assert_eq!(rs.dim1().len(), 0);
        assert_eq!(rs.dim0().rank1(0), 0);
    }

    #[test]
    fn test_get() {
        let bv0 = make_bv(&[true, false, true]);
        let bv1 = make_bv(&[false, true, false]);
        let rs = RankSelectMixedSE512::new(bv0, bv1).unwrap();
        assert_eq!(rs.get_dim(0, 0), Some(true));
        assert_eq!(rs.get_dim(0, 1), Some(false));
        assert_eq!(rs.get_dim(1, 0), Some(false));
        assert_eq!(rs.get_dim(1, 1), Some(true));
        assert_eq!(rs.get_dim(0, 3), None);
    }

    #[test]
    fn test_different_sizes() {
        let rs = RankSelectMixedSE512::new(make_bv(&vec![true; 100]), make_bv(&vec![false; 50])).unwrap();
        assert_eq!(rs.dim0().len(), 100);
        assert_eq!(rs.dim1().len(), 50);
        assert_eq!(rs.dim0().count_ones(), 100);
        assert_eq!(rs.dim1().count_ones(), 0);
    }
}
