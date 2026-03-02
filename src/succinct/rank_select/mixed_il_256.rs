//! RankSelectMixedIL256: Two-dimension interleaved rank/select with 256-bit blocks.
//!
//! Port of `rank_select_mixed_il_256` from topling-zip.
//! Stores TWO independent bitvectors interleaved in the same structure.
//! Each 256-bit "line" contains rank caches + bits for BOTH dimensions.
//!
//! Use case: NestLoudsTrie stores LOUDS structure bits and label/link bits
//! together, enabling cache-friendly access to both in a single cache line fetch.
//!
//! Layout per line (80 bytes):
//! ```text
//! [dim0: rlev1(4) + rlev2(4) + bit64(32)] = 40 bytes
//! [dim1: rlev1(4) + rlev2(4) + bit64(32)] = 40 bytes
//! ```

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;
use crate::succinct::BitVector;

const LINE_BITS: usize = 256;
const WORDS_PER_LINE: usize = 4;

/// Single dimension within a mixed line.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct DimLine {
    rlev1: u32,
    rlev2: [u8; 4],
    bit64: [u64; 4],
}

impl DimLine {
    fn new() -> Self {
        Self { rlev1: 0, rlev2: [0; 4], bit64: [0; 4] }
    }

    #[inline(always)]
    fn rank1_within(&self, bit_offset: usize) -> usize {
        let word_idx = bit_offset / 64;
        let bit_in_word = bit_offset % 64;
        let mut rank = self.rlev2[word_idx] as usize;
        if bit_in_word > 0 {
            let mask = (1u64 << bit_in_word) - 1;
            rank += (self.bit64[word_idx] & mask).count_ones() as usize;
        }
        rank
    }
}

/// Mixed interleaved line: two dimensions per block.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct MixedLine {
    dim: [DimLine; 2],
}

impl MixedLine {
    fn new() -> Self {
        Self { dim: [DimLine::new(), DimLine::new()] }
    }
}

/// Two-dimension interleaved rank/select with 256-bit blocks.
///
/// Access individual dimensions via `dim0()` and `dim1()` views,
/// or use `rank1_dim`/`select1_dim` directly.
pub struct RankSelectMixedIL256 {
    lines: Vec<MixedLine>,
    size: [usize; 2],        // bit count per dimension
    max_rank1: [usize; 2],
}

/// Read-only view of one dimension of a MixedIL256.
pub struct MixedDimView<'a> {
    parent: &'a RankSelectMixedIL256,
    dim: usize,
}

impl RankSelectMixedIL256 {
    /// Build from two BitVectors (one per dimension).
    pub fn new(bv0: BitVector, bv1: BitVector) -> Result<Self> {
        let size0 = bv0.len();
        let size1 = bv1.len();
        let max_size = size0.max(size1);
        let nlines = (max_size + LINE_BITS - 1) / LINE_BITS;

        let blocks0 = bv0.blocks();
        let blocks1 = bv1.blocks();

        let mut lines = Vec::with_capacity(nlines);
        let mut cum_rank = [0u32; 2];

        for i in 0..nlines {
            let mut ml = MixedLine::new();
            for d in 0..2 {
                ml.dim[d].rlev1 = cum_rank[d];
                let blocks = if d == 0 { blocks0 } else { blocks1 };
                let dim_size = if d == 0 { size0 } else { size1 };
                let mut r = 0u32;
                for j in 0..WORDS_PER_LINE {
                    ml.dim[d].rlev2[j] = r as u8;
                    let word_idx = i * WORDS_PER_LINE + j;
                    let word = if word_idx < blocks.len() { blocks[word_idx] } else { 0 };
                    // Mask off bits beyond dimension size
                    let global_bit = i * LINE_BITS + j * 64;
                    let effective_word = if global_bit + 64 > dim_size && global_bit < dim_size {
                        let valid_bits = dim_size - global_bit;
                        word & ((1u64 << valid_bits) - 1)
                    } else if global_bit >= dim_size {
                        0
                    } else {
                        word
                    };
                    ml.dim[d].bit64[j] = effective_word;
                    r += effective_word.count_ones();
                }
                cum_rank[d] += r;
            }
            lines.push(ml);
        }

        Ok(Self {
            lines,
            size: [size0, size1],
            max_rank1: [cum_rank[0] as usize, cum_rank[1] as usize],
        })
    }

    /// Get a read-only view of dimension 0.
    pub fn dim0(&self) -> MixedDimView<'_> { MixedDimView { parent: self, dim: 0 } }
    /// Get a read-only view of dimension 1.
    pub fn dim1(&self) -> MixedDimView<'_> { MixedDimView { parent: self, dim: 1 } }

    /// rank1 for a specific dimension.
    #[inline]
    pub fn rank1_dim(&self, dim: usize, pos: usize) -> usize {
        assert!(dim < 2);
        assert!(pos <= self.size[dim]);
        if pos == 0 { return 0; }
        let line_idx = pos / LINE_BITS;
        let bit_in_line = pos % LINE_BITS;
        let dl = &self.lines[line_idx].dim[dim];
        dl.rlev1 as usize + dl.rank1_within(bit_in_line)
    }

    /// rank0 for a specific dimension.
    #[inline]
    pub fn rank0_dim(&self, dim: usize, pos: usize) -> usize {
        pos - self.rank1_dim(dim, pos)
    }

    /// select1 for a specific dimension.
    pub fn select1_dim(&self, dim: usize, k: usize) -> Result<usize> {
        assert!(dim < 2);
        if k >= self.max_rank1[dim] {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        // Binary search over lines
        let nlines = self.lines.len();
        let mut lo = 0usize;
        let mut hi = nlines;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if (self.lines[mid].dim[dim].rlev1 as usize) <= k { lo = mid + 1; } else { hi = mid; }
        }
        let block = lo - 1;
        let dl = &self.lines[block].dim[dim];
        let base = dl.rlev1 as usize;
        let remaining = k - base;
        let base_bitpos = block * LINE_BITS;

        for j in (0..WORDS_PER_LINE).rev() {
            if remaining >= dl.rlev2[j] as usize {
                let in_word = remaining - dl.rlev2[j] as usize;
                let word = dl.bit64[j];
                return Ok(base_bitpos + j * 64 + select_in_word(word, in_word));
            }
        }
        Err(ZiporaError::invalid_data("select1 internal error"))
    }

    /// Get bit value for a specific dimension.
    pub fn get_dim(&self, dim: usize, index: usize) -> Option<bool> {
        if index >= self.size[dim] { return None; }
        let line_idx = index / LINE_BITS;
        let word_idx = (index % LINE_BITS) / 64;
        let bit_idx = index % 64;
        Some((self.lines[line_idx].dim[dim].bit64[word_idx] >> bit_idx) & 1 == 1)
    }

    pub fn size_dim(&self, dim: usize) -> usize { self.size[dim] }
    pub fn max_rank1_dim(&self, dim: usize) -> usize { self.max_rank1[dim] }

    pub fn mem_size(&self) -> usize {
        self.lines.len() * std::mem::size_of::<MixedLine>()
    }
}

/// View of one dimension — implements RankSelectOps.
impl RankSelectOps for MixedDimView<'_> {
    #[inline]
    fn rank1(&self, pos: usize) -> usize { self.parent.rank1_dim(self.dim, pos) }
    #[inline]
    fn rank0(&self, pos: usize) -> usize { self.parent.rank0_dim(self.dim, pos) }
    fn select1(&self, k: usize) -> Result<usize> { self.parent.select1_dim(self.dim, k) }
    fn select0(&self, _k: usize) -> Result<usize> {
        Err(ZiporaError::invalid_data("select0 not yet implemented for mixed"))
    }
    fn len(&self) -> usize { self.parent.size[self.dim] }
    fn count_ones(&self) -> usize { self.parent.max_rank1[self.dim] }
    fn get(&self, index: usize) -> Option<bool> { self.parent.get_dim(self.dim, index) }
    fn space_overhead_percent(&self) -> f64 {
        if self.parent.size[self.dim] == 0 { return 0.0; }
        let bit_bytes = (self.parent.size[self.dim] + 7) / 8;
        let cache_bytes = self.parent.lines.len() * 8; // rank cache portion
        (cache_bytes as f64 / bit_bytes as f64) * 100.0
    }
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

impl std::fmt::Debug for RankSelectMixedIL256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectMixedIL256")
            .field("size", &self.size)
            .field("max_rank1", &self.max_rank1)
            .field("lines", &self.lines.len())
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
    fn test_basic_two_dims() {
        let bv0 = make_bv(&[true, false, true, false, true]); // 3 ones
        let bv1 = make_bv(&[false, true, false, true, false]); // 2 ones
        let rs = RankSelectMixedIL256::new(bv0, bv1).unwrap();

        // Dimension 0
        assert_eq!(rs.dim0().len(), 5);
        assert_eq!(rs.dim0().count_ones(), 3);
        assert_eq!(rs.dim0().rank1(1), 1);
        assert_eq!(rs.dim0().rank1(5), 3);
        assert_eq!(rs.dim0().select1(0).unwrap(), 0);
        assert_eq!(rs.dim0().select1(2).unwrap(), 4);

        // Dimension 1
        assert_eq!(rs.dim1().len(), 5);
        assert_eq!(rs.dim1().count_ones(), 2);
        assert_eq!(rs.dim1().rank1(2), 1);
        assert_eq!(rs.dim1().select1(0).unwrap(), 1);
        assert_eq!(rs.dim1().select1(1).unwrap(), 3);
    }

    #[test]
    fn test_different_sizes() {
        let bv0 = make_bv(&[true; 100]);
        let bv1 = make_bv(&[false; 50]);
        let rs = RankSelectMixedIL256::new(bv0, bv1).unwrap();

        assert_eq!(rs.dim0().len(), 100);
        assert_eq!(rs.dim1().len(), 50);
        assert_eq!(rs.dim0().count_ones(), 100);
        assert_eq!(rs.dim1().count_ones(), 0);
    }

    #[test]
    fn test_invariant() {
        let p0: Vec<bool> = (0..1000).map(|i| i % 5 == 0).collect();
        let p1: Vec<bool> = (0..1000).map(|i| i % 7 == 0).collect();
        let rs = RankSelectMixedIL256::new(make_bv(&p0), make_bv(&p1)).unwrap();

        for i in 0..=1000 {
            assert_eq!(rs.dim0().rank0(i) + rs.dim0().rank1(i), i, "dim0 invariant at {}", i);
            assert_eq!(rs.dim1().rank0(i) + rs.dim1().rank1(i), i, "dim1 invariant at {}", i);
        }
    }

    #[test]
    fn test_roundtrip() {
        let p0: Vec<bool> = (0..500).map(|i| i % 3 == 0).collect();
        let p1: Vec<bool> = (0..500).map(|i| i % 11 == 0).collect();
        let rs = RankSelectMixedIL256::new(make_bv(&p0), make_bv(&p1)).unwrap();

        for k in 0..rs.dim0().count_ones() {
            let pos = rs.dim0().select1(k).unwrap();
            assert_eq!(rs.dim0().get(pos), Some(true));
        }
        for k in 0..rs.dim1().count_ones() {
            let pos = rs.dim1().select1(k).unwrap();
            assert_eq!(rs.dim1().get(pos), Some(true));
        }
    }

    #[test]
    fn test_get() {
        let bv0 = make_bv(&[true, false, true]);
        let bv1 = make_bv(&[false, true, false]);
        let rs = RankSelectMixedIL256::new(bv0, bv1).unwrap();

        assert_eq!(rs.get_dim(0, 0), Some(true));
        assert_eq!(rs.get_dim(0, 1), Some(false));
        assert_eq!(rs.get_dim(1, 0), Some(false));
        assert_eq!(rs.get_dim(1, 1), Some(true));
        assert_eq!(rs.get_dim(0, 3), None);
    }

    #[test]
    fn test_large() {
        let p0: Vec<bool> = (0..5000).map(|i| i % 13 == 0).collect();
        let p1: Vec<bool> = (0..5000).map(|i| i % 17 == 0).collect();
        let rs = RankSelectMixedIL256::new(make_bv(&p0), make_bv(&p1)).unwrap();

        let e0 = (0..5000).filter(|i| i % 13 == 0).count();
        let e1 = (0..5000).filter(|i| i % 17 == 0).count();
        assert_eq!(rs.dim0().count_ones(), e0);
        assert_eq!(rs.dim1().count_ones(), e1);

        assert_eq!(rs.dim0().select1(0).unwrap(), 0);
        assert_eq!(rs.dim0().select1(1).unwrap(), 13);
        assert_eq!(rs.dim1().select1(0).unwrap(), 0);
        assert_eq!(rs.dim1().select1(1).unwrap(), 17);
    }
}
