//! RankSelectFew: Sparse bitvector rank/select.
//!
//! Port of `rank_select_few<P, W>` from topling-zip.
//! Optimized for bitvectors where one value (0 or 1) is much rarer than the other.
//!
//! Instead of storing the full bitvector, only the positions of the rare ("pivot")
//! elements are stored in a sorted array. This gives:
//! - O(log n) rank via binary search on pivot positions
//! - O(1) select for pivots (direct array lookup)
//! - Minimal memory: only stores pivot positions
//!
//! Example: For a 10,000-bit vector with 100 ones, stores 100 positions (400 bytes)
//! instead of 1,250 bytes for the full bitvector + rank cache.

use crate::error::{Result, ZiporaError};
use super::RankSelectOps;

/// Sparse bitvector for data where 1-bits are rare.
///
/// Stores sorted positions of set bits. `rank1` uses binary search.
/// `select1` is O(1) direct array lookup.
pub struct RankSelectFewOne {
    /// Sorted positions of all 1-bits
    positions: Vec<u32>,
    /// Total number of bits (0s + 1s)
    size: usize,
}

impl RankSelectFewOne {
    /// Build from a sorted list of 1-bit positions and total size.
    pub fn new(positions: Vec<u32>, size: usize) -> Result<Self> {
        // Validate: positions must be sorted and within bounds
        for (i, &pos) in positions.iter().enumerate() {
            if pos as usize >= size {
                return Err(ZiporaError::invalid_data("position out of bounds"));
            }
            if i > 0 && pos <= positions[i - 1] {
                return Err(ZiporaError::invalid_data("positions must be strictly sorted"));
            }
        }
        Ok(Self { positions, size })
    }

    /// Build from a BitVector by extracting set bit positions.
    pub fn from_bitvector(bv: &crate::succinct::BitVector) -> Result<Self> {
        let mut positions = Vec::new();
        for i in 0..bv.len() {
            if bv.get(i) == Some(true) {
                positions.push(i as u32);
            }
        }
        Self::new(positions, bv.len())
    }

    /// Number of 1-bits (pivots)
    pub fn num_ones(&self) -> usize { self.positions.len() }
    /// Number of 0-bits
    pub fn num_zeros(&self) -> usize { self.size - self.positions.len() }

    pub fn mem_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.positions.len() * 4
    }

    /// Binary search: number of positions < val
    #[inline]
    fn lower_bound(&self, val: usize) -> usize {
        self.positions.partition_point(|&p| (p as usize) < val)
    }
}

impl RankSelectOps for RankSelectFewOne {
    /// O(log n) rank1 via binary search on pivot positions.
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        assert!(pos <= self.size);
        self.lower_bound(pos)
    }

    #[inline]
    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }

    /// O(1) select1 — direct array lookup.
    #[inline]
    fn select1(&self, k: usize) -> Result<usize> {
        if k >= self.positions.len() {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        Ok(self.positions[k] as usize)
    }

    /// O(log n) select0 — binary search for k-th zero.
    fn select0(&self, k: usize) -> Result<usize> {
        let num_zeros = self.num_zeros();
        if k >= num_zeros {
            return Err(ZiporaError::invalid_data("select0 out of range"));
        }
        // Binary search: find smallest pos where pos - rank1(pos) > k
        // i.e., find the k-th position that is NOT in the positions array
        let mut lo = 0usize;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let zeros_before_mid = mid - self.lower_bound(mid);
            if zeros_before_mid <= k {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is the position where zeros_before = k+1, so k-th zero is at lo-1
        if lo > 0 { Ok(lo - 1) } else { Err(ZiporaError::invalid_data("select0 error")) }
    }

    fn len(&self) -> usize { self.size }
    fn count_ones(&self) -> usize { self.positions.len() }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.size { return None; }
        Some(self.positions.binary_search(&(index as u32)).is_ok())
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.size == 0 { return 0.0; }
        let minimal_bytes = (self.size + 7) / 8;
        let actual_bytes = self.mem_size();
        if actual_bytes < minimal_bytes {
            // Negative overhead = space savings
            -((minimal_bytes - actual_bytes) as f64 / minimal_bytes as f64) * 100.0
        } else {
            ((actual_bytes - minimal_bytes) as f64 / minimal_bytes as f64) * 100.0
        }
    }
}

/// Sparse bitvector for data where 0-bits are rare.
///
/// Mirror of `RankSelectFewOne` but stores positions of 0-bits.
/// `rank0` uses binary search, `select0` is O(1).
pub struct RankSelectFewZero {
    /// Sorted positions of all 0-bits
    positions: Vec<u32>,
    /// Total number of bits
    size: usize,
}

impl RankSelectFewZero {
    pub fn new(positions: Vec<u32>, size: usize) -> Result<Self> {
        for (i, &pos) in positions.iter().enumerate() {
            if pos as usize >= size {
                return Err(ZiporaError::invalid_data("position out of bounds"));
            }
            if i > 0 && pos <= positions[i - 1] {
                return Err(ZiporaError::invalid_data("positions must be strictly sorted"));
            }
        }
        Ok(Self { positions, size })
    }

    pub fn from_bitvector(bv: &crate::succinct::BitVector) -> Result<Self> {
        let mut positions = Vec::new();
        for i in 0..bv.len() {
            if bv.get(i) == Some(false) {
                positions.push(i as u32);
            }
        }
        Self::new(positions, bv.len())
    }

    pub fn num_zeros(&self) -> usize { self.positions.len() }
    pub fn num_ones(&self) -> usize { self.size - self.positions.len() }

    pub fn mem_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.positions.len() * 4
    }

    #[inline]
    fn lower_bound(&self, val: usize) -> usize {
        self.positions.partition_point(|&p| (p as usize) < val)
    }
}

impl RankSelectOps for RankSelectFewZero {
    #[inline]
    fn rank1(&self, pos: usize) -> usize {
        pos - self.rank0(pos)
    }

    /// O(log n) rank0 via binary search.
    #[inline]
    fn rank0(&self, pos: usize) -> usize {
        assert!(pos <= self.size);
        self.lower_bound(pos)
    }

    /// O(log n) select1 — find k-th one.
    fn select1(&self, k: usize) -> Result<usize> {
        let num_ones = self.num_ones();
        if k >= num_ones {
            return Err(ZiporaError::invalid_data("select1 out of range"));
        }
        let mut lo = 0usize;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let ones_before_mid = mid - self.lower_bound(mid);
            if ones_before_mid <= k { lo = mid + 1; } else { hi = mid; }
        }
        if lo > 0 { Ok(lo - 1) } else { Err(ZiporaError::invalid_data("select1 error")) }
    }

    /// O(1) select0 — direct lookup.
    #[inline]
    fn select0(&self, k: usize) -> Result<usize> {
        if k >= self.positions.len() {
            return Err(ZiporaError::invalid_data("select0 out of range"));
        }
        Ok(self.positions[k] as usize)
    }

    fn len(&self) -> usize { self.size }
    fn count_ones(&self) -> usize { self.num_ones() }

    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.size { return None; }
        // If position is in zeros list, it's a 0; otherwise 1
        Some(self.positions.binary_search(&(index as u32)).is_err())
    }

    fn space_overhead_percent(&self) -> f64 {
        if self.size == 0 { return 0.0; }
        let minimal_bytes = (self.size + 7) / 8;
        let actual_bytes = self.mem_size();
        if actual_bytes < minimal_bytes {
            -((minimal_bytes - actual_bytes) as f64 / minimal_bytes as f64) * 100.0
        } else {
            ((actual_bytes - minimal_bytes) as f64 / minimal_bytes as f64) * 100.0
        }
    }
}

impl std::fmt::Debug for RankSelectFewOne {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectFewOne")
            .field("size", &self.size)
            .field("ones", &self.positions.len())
            .finish()
    }
}

impl std::fmt::Debug for RankSelectFewZero {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RankSelectFewZero")
            .field("size", &self.size)
            .field("zeros", &self.positions.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_few_one_basic() {
        // Bits: 0001 0000 0001 (positions 3 and 10 are set)
        let rs = RankSelectFewOne::new(vec![3, 10], 12).unwrap();
        assert_eq!(rs.len(), 12);
        assert_eq!(rs.count_ones(), 2);
        assert_eq!(rs.rank1(0), 0);
        assert_eq!(rs.rank1(3), 0);
        assert_eq!(rs.rank1(4), 1);
        assert_eq!(rs.rank1(10), 1);
        assert_eq!(rs.rank1(11), 2);
        assert_eq!(rs.rank1(12), 2);
        assert_eq!(rs.select1(0).unwrap(), 3);
        assert_eq!(rs.select1(1).unwrap(), 10);
        assert!(rs.select1(2).is_err());
    }

    #[test]
    fn test_few_one_get() {
        let rs = RankSelectFewOne::new(vec![0, 5, 99], 100).unwrap();
        assert_eq!(rs.get(0), Some(true));
        assert_eq!(rs.get(1), Some(false));
        assert_eq!(rs.get(5), Some(true));
        assert_eq!(rs.get(99), Some(true));
        assert_eq!(rs.get(100), None);
    }

    #[test]
    fn test_few_one_invariant() {
        let rs = RankSelectFewOne::new(vec![10, 20, 30, 40, 50], 100).unwrap();
        for i in 0..=100 {
            assert_eq!(rs.rank0(i) + rs.rank1(i), i, "invariant at {}", i);
        }
    }

    #[test]
    fn test_few_one_select0() {
        // Size 10, ones at positions 2 and 7
        // Zeros at: 0, 1, 3, 4, 5, 6, 8, 9
        let rs = RankSelectFewOne::new(vec![2, 7], 10).unwrap();
        assert_eq!(rs.select0(0).unwrap(), 0);
        assert_eq!(rs.select0(1).unwrap(), 1);
        assert_eq!(rs.select0(2).unwrap(), 3);
        assert_eq!(rs.select0(7).unwrap(), 9);
    }

    #[test]
    fn test_few_one_empty_ones() {
        let rs = RankSelectFewOne::new(vec![], 50).unwrap();
        assert_eq!(rs.count_ones(), 0);
        assert!(rs.select1(0).is_err());
        assert_eq!(rs.select0(0).unwrap(), 0);
        assert_eq!(rs.rank1(25), 0);
        assert_eq!(rs.rank0(25), 25);
    }

    #[test]
    fn test_few_zero_basic() {
        // Size 10, zeros at positions 1, 5 — everything else is 1
        let rs = RankSelectFewZero::new(vec![1, 5], 10).unwrap();
        assert_eq!(rs.len(), 10);
        assert_eq!(rs.count_ones(), 8);
        assert_eq!(rs.rank0(0), 0);
        assert_eq!(rs.rank0(2), 1);
        assert_eq!(rs.rank0(6), 2);
        assert_eq!(rs.rank0(10), 2);
        assert_eq!(rs.select0(0).unwrap(), 1);
        assert_eq!(rs.select0(1).unwrap(), 5);
        assert!(rs.select0(2).is_err());
    }

    #[test]
    fn test_few_zero_get() {
        let rs = RankSelectFewZero::new(vec![3, 7], 10).unwrap();
        assert_eq!(rs.get(0), Some(true));
        assert_eq!(rs.get(3), Some(false));
        assert_eq!(rs.get(7), Some(false));
        assert_eq!(rs.get(9), Some(true));
    }

    #[test]
    fn test_few_zero_select1() {
        // Size 8, zeros at 2 and 5. Ones at: 0,1,3,4,6,7
        let rs = RankSelectFewZero::new(vec![2, 5], 8).unwrap();
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 1);
        assert_eq!(rs.select1(2).unwrap(), 3);
        assert_eq!(rs.select1(3).unwrap(), 4);
        assert_eq!(rs.select1(4).unwrap(), 6);
        assert_eq!(rs.select1(5).unwrap(), 7);
    }

    #[test]
    fn test_few_space_savings() {
        // 100,000 bits with only 10 ones — massive space savings
        let positions: Vec<u32> = (0..10).map(|i| i * 10000).collect();
        let rs = RankSelectFewOne::new(positions, 100_000).unwrap();
        // Full bitvector would be 12,500 bytes. We use ~40 bytes.
        assert!(rs.mem_size() < 200);
        // Negative overhead = savings
        assert!(rs.space_overhead_percent() < 0.0);
    }

    #[test]
    fn test_few_from_bitvector() {
        let mut bv = crate::succinct::BitVector::new();
        for i in 0..1000 {
            bv.push(i % 100 == 0).unwrap();
        }
        let rs = RankSelectFewOne::from_bitvector(&bv).unwrap();
        assert_eq!(rs.count_ones(), 10);
        assert_eq!(rs.select1(0).unwrap(), 0);
        assert_eq!(rs.select1(1).unwrap(), 100);
        assert_eq!(rs.select1(9).unwrap(), 900);
    }
}
