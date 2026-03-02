//! Trivial rank/select implementations for all-zero and all-one bitvectors.
//!
//! Port of `rank_select_allzero` and `rank_select_allone` from topling-zip.
//! No bit storage — just `size`. All operations are O(1).

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;

/// Rank/select for a bitvector that is entirely zeros.
///
/// No actual bit storage. `rank0(pos) = pos`, `rank1(pos) = 0`.
/// Used when all values are known to be zero (e.g., empty LOUDS labels).
#[derive(Debug, Clone)]
pub struct RankSelectAllZero {
    size: usize,
}

impl RankSelectAllZero {
    pub fn new(size: usize) -> Self { Self { size } }
    #[inline(always)] pub fn is0(&self, i: usize) -> bool { assert!(i < self.size); true }
    #[inline(always)] pub fn is1(&self, i: usize) -> bool { assert!(i < self.size); false }
    pub fn max_rank0(&self) -> usize { self.size }
    pub fn max_rank1(&self) -> usize { 0 }
    pub fn mem_size(&self) -> usize { std::mem::size_of::<Self>() }
    pub fn zero_seq_len(&self, bitpos: usize) -> usize { assert!(bitpos < self.size); self.size - bitpos }
    pub fn one_seq_len(&self, _bitpos: usize) -> usize { 0 }
}

impl RankSelectOps for RankSelectAllZero {
    #[inline(always)]
    fn rank1(&self, pos: usize) -> usize { assert!(pos <= self.size); 0 }
    #[inline(always)]
    fn rank0(&self, pos: usize) -> usize { assert!(pos <= self.size); pos }
    fn select1(&self, _k: usize) -> Result<usize> {
        Err(ZiporaError::invalid_data("select1 on all-zero bitvector"))
    }
    #[inline(always)]
    fn select0(&self, k: usize) -> Result<usize> {
        if k < self.size { Ok(k) } else {
            Err(ZiporaError::invalid_data("select0 out of range"))
        }
    }
    fn len(&self) -> usize { self.size }
    fn count_ones(&self) -> usize { 0 }
    fn get(&self, index: usize) -> Option<bool> {
        if index < self.size { Some(false) } else { None }
    }
    fn space_overhead_percent(&self) -> f64 { 0.0 }
}

/// Rank/select for a bitvector that is entirely ones.
///
/// No actual bit storage. `rank1(pos) = pos`, `rank0(pos) = 0`.
/// Used when all values are known to be one.
#[derive(Debug, Clone)]
pub struct RankSelectAllOne {
    size: usize,
}

impl RankSelectAllOne {
    pub fn new(size: usize) -> Self { Self { size } }
    #[inline(always)] pub fn is0(&self, i: usize) -> bool { assert!(i < self.size); false }
    #[inline(always)] pub fn is1(&self, i: usize) -> bool { assert!(i < self.size); true }
    pub fn max_rank0(&self) -> usize { 0 }
    pub fn max_rank1(&self) -> usize { self.size }
    pub fn mem_size(&self) -> usize { std::mem::size_of::<Self>() }
    pub fn zero_seq_len(&self, _bitpos: usize) -> usize { 0 }
    pub fn one_seq_len(&self, bitpos: usize) -> usize { assert!(bitpos < self.size); self.size - bitpos }
}

impl RankSelectOps for RankSelectAllOne {
    #[inline(always)]
    fn rank1(&self, pos: usize) -> usize { assert!(pos <= self.size); pos }
    #[inline(always)]
    fn rank0(&self, pos: usize) -> usize { assert!(pos <= self.size); 0 }
    #[inline(always)]
    fn select1(&self, k: usize) -> Result<usize> {
        if k < self.size { Ok(k) } else {
            Err(ZiporaError::invalid_data("select1 out of range"))
        }
    }
    fn select0(&self, _k: usize) -> Result<usize> {
        Err(ZiporaError::invalid_data("select0 on all-one bitvector"))
    }
    fn len(&self) -> usize { self.size }
    fn count_ones(&self) -> usize { self.size }
    fn get(&self, index: usize) -> Option<bool> {
        if index < self.size { Some(true) } else { None }
    }
    fn space_overhead_percent(&self) -> f64 { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allzero() {
        let rs = RankSelectAllZero::new(100);
        assert_eq!(rs.len(), 100);
        assert_eq!(rs.count_ones(), 0);
        assert_eq!(rs.rank0(50), 50);
        assert_eq!(rs.rank1(50), 0);
        assert_eq!(rs.select0(42).unwrap(), 42);
        assert!(rs.select1(0).is_err());
        assert_eq!(rs.get(0), Some(false));
        assert_eq!(rs.get(99), Some(false));
        assert_eq!(rs.get(100), None);
        for i in 0..=100 {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i);
        }
    }

    #[test]
    fn test_allone() {
        let rs = RankSelectAllOne::new(100);
        assert_eq!(rs.len(), 100);
        assert_eq!(rs.count_ones(), 100);
        assert_eq!(rs.rank1(50), 50);
        assert_eq!(rs.rank0(50), 0);
        assert_eq!(rs.select1(42).unwrap(), 42);
        assert!(rs.select0(0).is_err());
        assert_eq!(rs.get(0), Some(true));
        assert_eq!(rs.get(99), Some(true));
        assert_eq!(rs.get(100), None);
        for i in 0..=100 {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i);
        }
    }

    #[test]
    fn test_empty() {
        let rs0 = RankSelectAllZero::new(0);
        assert_eq!(rs0.len(), 0);
        assert_eq!(rs0.rank0(0), 0);
        assert_eq!(rs0.rank1(0), 0);

        let rs1 = RankSelectAllOne::new(0);
        assert_eq!(rs1.len(), 0);
        assert_eq!(rs1.rank0(0), 0);
        assert_eq!(rs1.rank1(0), 0);
    }
}
