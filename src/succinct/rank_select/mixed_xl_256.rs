//! RankSelectMixedXL256: Multi-dimension interleaved rank/select with 256-bit blocks.
//!
//! Supports 2, 3, or 4 independent bitvectors interleaved in the same structure.
//!
//! Each line contains `Arity × 4` words (bits interleaved at word level) plus
//! rank caches for all dimensions. Words are laid out as:
//! `[d0_w0, d1_w0, ..., dA_w0, d0_w1, d1_w1, ..., dA_w1, ...]`
//!
//! Line structure:
//! - `bit64[Arity * 4]` — interleaved bit words
//! - `mixed[Arity]` — rank cache per dimension: `{base: u32, rlev: [u8; 4]}`

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 256;
const WORDS_PER_LINE: usize = 4;

/// Per-dimension rank cache within a line.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct DimRank {
    base: u32,
    rlev: [u8; 4],
}

/// A line in the XL256 structure: interleaved bits + per-dim rank caches.
/// For arity=2: 8 words (64B bits) + 2 × 8B rank = 80 bytes.
/// For arity=3: 12 words (96B bits) + 3 × 8B rank = 120 bytes.
#[derive(Debug, Clone)]
struct XlLine {
    /// Interleaved bits: bit64[word_in_line * arity + dim]
    bit64: Vec<u64>,
    /// Rank cache per dimension
    mixed: Vec<DimRank>,
}

/// Multi-dimension interleaved rank/select with 256-bit blocks.
///
/// Supports 2-4 independent bitvectors. Access via `dim(d)` views.
pub struct RankSelectMixedXL256 {
    lines: Vec<XlLine>,
    arity: usize,
    size: Vec<usize>,
    max_rank1: Vec<usize>,
}

/// Read-only view of one dimension.
pub struct MixedXL256DimView<'a> {
    parent: &'a RankSelectMixedXL256,
    dim: usize,
}

impl RankSelectMixedXL256 {
    /// Build from multiple BitVectors (2-4 dimensions).
    pub fn new(bitvectors: Vec<BitVector>) -> Result<Self> {
        let arity = bitvectors.len();
        if arity < 2 || arity > 4 {
            return Err(ZiporaError::invalid_data("arity must be 2, 3, or 4"));
        }

        let sizes: Vec<usize> = bitvectors.iter().map(|bv| bv.len()).collect();
        let max_size = *sizes.iter().max().unwrap_or(&0);
        let nlines = (max_size + LINE_BITS - 1) / LINE_BITS;

        let blocks: Vec<&[u64]> = bitvectors.iter().map(|bv| bv.blocks()).collect();

        let mut lines = Vec::with_capacity(nlines);
        let mut cum_rank: Vec<u32> = vec![0; arity];

        for i in 0..nlines {
            let words_per_xl_line = WORDS_PER_LINE * arity;
            let mut bit64 = vec![0u64; words_per_xl_line];
            let mut mixed = Vec::with_capacity(arity);

            for d in 0..arity {
                let mut dr = DimRank { base: cum_rank[d], rlev: [0; 4] };
                let mut r = 0u32;
                for j in 0..WORDS_PER_LINE {
                    dr.rlev[j] = r as u8;
                    let src_idx = i * WORDS_PER_LINE + j;
                    let mut word = if src_idx < blocks[d].len() { blocks[d][src_idx] } else { 0 };

                    // Mask bits beyond dimension size
                    let global_bit = i * LINE_BITS + j * 64;
                    if global_bit + 64 > sizes[d] && global_bit < sizes[d] {
                        let valid = sizes[d] - global_bit;
                        word &= (1u64 << valid) - 1;
                    } else if global_bit >= sizes[d] {
                        word = 0;
                    }

                    bit64[j * arity + d] = word;
                    r += word.count_ones();
                }
                cum_rank[d] += r;
                mixed.push(dr);
            }
            lines.push(XlLine { bit64, mixed });
        }

        let max_rank1: Vec<usize> = cum_rank.iter().map(|&r| r as usize).collect();

        Ok(Self { lines, arity, size: sizes, max_rank1 })
    }

    /// Build from exactly 2 BitVectors.
    pub fn new2(bv0: BitVector, bv1: BitVector) -> Result<Self> {
        Self::new(vec![bv0, bv1])
    }

    /// Build from exactly 3 BitVectors.
    pub fn new3(bv0: BitVector, bv1: BitVector, bv2: BitVector) -> Result<Self> {
        Self::new(vec![bv0, bv1, bv2])
    }

    /// Get a read-only view of dimension `d`.
    pub fn dim(&self, d: usize) -> MixedXL256DimView<'_> {
        assert!(d < self.arity);
        MixedXL256DimView { parent: self, dim: d }
    }

    pub fn arity(&self) -> usize { self.arity }

    #[inline]
    pub fn rank1_dim(&self, dim: usize, pos: usize) -> usize {
        assert!(dim < self.arity && pos <= self.size[dim]);
        if pos == 0 { return 0; }
        let line_idx = pos / LINE_BITS;
        let bit_in_line = pos % LINE_BITS;
        let word_in_line = bit_in_line / 64;
        let bit_in_word = bit_in_line % 64;

        let line = &self.lines[line_idx];
        let dr = &line.mixed[dim];
        let mut rank = dr.base as usize + dr.rlev[word_in_line] as usize;

        if bit_in_word > 0 {
            let word = line.bit64[word_in_line * self.arity + dim];
            rank += (word & ((1u64 << bit_in_word) - 1)).count_ones() as usize;
        }
        rank
    }

    #[inline]
    pub fn rank0_dim(&self, dim: usize, pos: usize) -> usize { pos - self.rank1_dim(dim, pos) }

    pub fn select1_dim(&self, dim: usize, k: usize) -> Result<usize> {
        assert!(dim < self.arity);
        if k >= self.max_rank1[dim] {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        let nlines = self.lines.len();
        let mut lo = 0usize;
        let mut hi = nlines;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.lines[mid].mixed[dim].base as usize) <= k { lo = mid + 1; } else { hi = mid; }
        }
        let block = lo - 1;
        let line = &self.lines[block];
        let dr = &line.mixed[dim];
        let remaining = k - dr.base as usize;
        let base_bitpos = block * LINE_BITS;

        for j in (0..WORDS_PER_LINE).rev() {
            if remaining >= dr.rlev[j] as usize {
                let in_word = remaining - dr.rlev[j] as usize;
                let word = line.bit64[j * self.arity + dim];
                return Ok(base_bitpos + j * 64 + select_in_word(word, in_word));
            }
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    #[inline]
    pub fn get_dim(&self, dim: usize, index: usize) -> Option<bool> {
        if index >= self.size[dim] { return None; }
        let line_idx = index / LINE_BITS;
        let word_in_line = (index % LINE_BITS) / 64;
        let bit_idx = index % 64;
        let word = self.lines[line_idx].bit64[word_in_line * self.arity + dim];
        Some((word >> bit_idx) & 1 == 1)
    }

    #[inline]
    pub fn mem_size(&self) -> usize {
        if self.lines.is_empty() { return 0; }
        self.lines.len() * (self.arity * WORDS_PER_LINE * 8 + self.arity * 8)
    }
}

impl RankSelectOps for MixedXL256DimView<'_> {
    #[inline]
    fn rank1(&self, pos: usize) -> usize { self.parent.rank1_dim(self.dim, pos) }
    fn rank0(&self, pos: usize) -> usize { self.parent.rank0_dim(self.dim, pos) }
    #[inline]
    fn select1(&self, k: usize) -> Result<usize> { self.parent.select1_dim(self.dim, k) }
    fn select0(&self, _k: usize) -> Result<usize> {
        Err(ZiporaError::invalid_data("select0 not implemented for mixed_xl_256"))
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
            // SAFETY: BMI2 feature detected at runtime, _pdep_u64 is pure arithmetic with no memory access
            let d = unsafe { core::arch::x86_64::_pdep_u64(1u64 << k, word) };
            return d.trailing_zeros() as usize;
        }
    }
    let mut w = word;
    for _ in 0..k { w &= w - 1; }
    w.trailing_zeros() as usize
}

impl std::fmt::Debug for RankSelectMixedXL256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectMixedXL256")
            .field("arity", &self.arity)
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
    fn test_arity2_basic() {
        let bv0 = make_bv(&[true, false, true, false, true]);
        let bv1 = make_bv(&[false, true, false, true, false]);
        let rs = RankSelectMixedXL256::new2(bv0, bv1).unwrap();

        assert_eq!(rs.arity(), 2);
        assert_eq!(rs.dim(0).count_ones(), 3);
        assert_eq!(rs.dim(1).count_ones(), 2);
        assert_eq!(rs.dim(0).rank1(5), 3);
        assert_eq!(rs.dim(1).rank1(4), 2);
        assert_eq!(rs.dim(0).select1(0).unwrap(), 0);
        assert_eq!(rs.dim(1).select1(0).unwrap(), 1);
    }

    #[test]
    fn test_arity3() {
        let bv0 = make_bv(&[true, false, false, true]);
        let bv1 = make_bv(&[false, true, false, false]);
        let bv2 = make_bv(&[false, false, true, false]);
        let rs = RankSelectMixedXL256::new3(bv0, bv1, bv2).unwrap();

        assert_eq!(rs.arity(), 3);
        assert_eq!(rs.dim(0).count_ones(), 2);
        assert_eq!(rs.dim(1).count_ones(), 1);
        assert_eq!(rs.dim(2).count_ones(), 1);
        assert_eq!(rs.dim(0).select1(0).unwrap(), 0);
        assert_eq!(rs.dim(0).select1(1).unwrap(), 3);
        assert_eq!(rs.dim(1).select1(0).unwrap(), 1);
        assert_eq!(rs.dim(2).select1(0).unwrap(), 2);
    }

    #[test]
    fn test_arity2_invariant() {
        let p0: Vec<bool> = (0..1000).map(|i| i % 5 == 0).collect();
        let p1: Vec<bool> = (0..1000).map(|i| i % 7 == 0).collect();
        let rs = RankSelectMixedXL256::new2(make_bv(&p0), make_bv(&p1)).unwrap();
        for i in 0..=1000 {
            assert_eq!(rs.dim(0).rank0(i) + rs.dim(0).rank1(i), i, "dim0 at {}", i);
            assert_eq!(rs.dim(1).rank0(i) + rs.dim(1).rank1(i), i, "dim1 at {}", i);
        }
    }

    #[test]
    fn test_arity3_invariant() {
        let p0: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let p1: Vec<bool> = (0..500).map(|i| i % 5 == 0).collect();
        let p2: Vec<bool> = (0..500).map(|i| i % 11 == 0).collect();
        let rs = RankSelectMixedXL256::new3(make_bv(&p0), make_bv(&p1), make_bv(&p2)).unwrap();
        for i in 0..=500 {
            for d in 0..3 {
                assert_eq!(rs.dim(d).rank0(i) + rs.dim(d).rank1(i), i, "dim{} at {}", d, i);
            }
        }
    }

    #[test]
    fn test_roundtrip() {
        let p0: Vec<bool> = (0..800).map(|i| i % 3 == 0).collect();
        let p1: Vec<bool> = (0..800).map(|i| i % 11 == 0).collect();
        let rs = RankSelectMixedXL256::new2(make_bv(&p0), make_bv(&p1)).unwrap();
        for d in 0..2 {
            for k in 0..rs.dim(d).count_ones() {
                let pos = rs.dim(d).select1(k).unwrap();
                assert_eq!(rs.get_dim(d, pos), Some(true), "dim{} select1({})={}", d, k, pos);
            }
        }
    }

    #[test]
    fn test_get() {
        let bv0 = make_bv(&[true, false, true]);
        let bv1 = make_bv(&[false, true, false]);
        let rs = RankSelectMixedXL256::new2(bv0, bv1).unwrap();
        assert_eq!(rs.get_dim(0, 0), Some(true));
        assert_eq!(rs.get_dim(0, 1), Some(false));
        assert_eq!(rs.get_dim(1, 0), Some(false));
        assert_eq!(rs.get_dim(1, 1), Some(true));
        assert_eq!(rs.get_dim(0, 3), None);
    }

    #[test]
    fn test_different_sizes() {
        let bv0 = make_bv(&vec![true; 100]);
        let bv1 = make_bv(&vec![false; 50]);
        let rs = RankSelectMixedXL256::new2(bv0, bv1).unwrap();
        assert_eq!(rs.dim(0).len(), 100);
        assert_eq!(rs.dim(1).len(), 50);
    }

    #[test]
    fn test_invalid_arity() {
        let bv = make_bv(&[true]);
        assert!(RankSelectMixedXL256::new(vec![bv]).is_err());
    }

    #[test]
    fn test_empty() {
        let rs = RankSelectMixedXL256::new2(make_bv(&[]), make_bv(&[])).unwrap();
        assert_eq!(rs.dim(0).len(), 0);
        assert_eq!(rs.dim(1).len(), 0);
        assert_eq!(rs.dim(0).rank1(0), 0);
    }

    #[test]
    fn test_large_arity2() {
        let p0: Vec<bool> = (0..5000).map(|i| i % 13 == 0).collect();
        let p1: Vec<bool> = (0..5000).map(|i| i % 17 == 0).collect();
        let rs = RankSelectMixedXL256::new2(make_bv(&p0), make_bv(&p1)).unwrap();
        let e0 = (0..5000).filter(|i| i % 13 == 0).count();
        let e1 = (0..5000).filter(|i| i % 17 == 0).count();
        assert_eq!(rs.dim(0).count_ones(), e0);
        assert_eq!(rs.dim(1).count_ones(), e1);
        assert_eq!(rs.dim(0).select1(1).unwrap(), 13);
        assert_eq!(rs.dim(1).select1(1).unwrap(), 17);
    }

    #[test]
    fn test_arity4() {
        let bvs: Vec<BitVector> = (0..4).map(|d| {
            make_bv(&(0..100).map(|i| i % (d + 2) == 0).collect::<Vec<_>>())
        }).collect();
        let rs = RankSelectMixedXL256::new(bvs).unwrap();
        assert_eq!(rs.arity(), 4);
        for d in 0..4 {
            let expected = (0..100).filter(|i| i % (d + 2) == 0).count();
            assert_eq!(rs.dim(d).count_ones(), expected, "dim{} ones", d);
            for i in 0..=100 {
                assert_eq!(rs.dim(d).rank0(i) + rs.dim(d).rank1(i), i, "dim{} invariant at {}", d, i);
            }
        }
    }
}
