const DENSE_THRESHOLD: usize = 64;
const PARTITION_THRESHOLD: usize = 256;
const OPTIMAL_THRESHOLD: usize = 4096;
use crate::algorithms::bit_ops::select_in_word;
use crate::error::{Result, ZiporaError};
use crate::succinct::BitVector;
use std::cmp::Ordering;

use super::basic::{EliasFano, EliasFanoCursor};
use super::optimal::{OptimalPartitionedEliasFano, OptimalPefCursor};
use super::partitioned::{PartitionedEliasFano, PartitionedEliasFanoCursor};

/// Adaptive posting list that selects the best encoding based on list statistics.
///
/// - **Dense** (< 64 elements): Raw `Vec<u32>` — zero decode cost, best for short lists.
/// - **EliasFano** (64-256 elements): Plain EF — good compression, no chunk overhead.
/// - **PartitionedEF** (256-4096 elements): Uniform 128-chunk PEF — cache-local `next_geq`.
/// - **OptimalPEF** (> 4096 elements): DP-optimal variable chunks — best compression.
///
/// All variants support the same query interface: `get`, `next_geq`, `iter`, `len`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HybridPostingList {
    /// Raw sorted array for very short lists.
    Dense(Vec<u32>),
    /// Plain Elias-Fano for medium lists.
    EliasFano(EliasFano),
    /// Uniform Partitioned EF for large lists.
    Partitioned(PartitionedEliasFano),
    /// DP-Optimal Partitioned EF for very large lists.
    Optimal(OptimalPartitionedEliasFano),
}

impl HybridPostingList {
    /// Build from a sorted slice, automatically selecting the best encoding.
    pub fn from_sorted(values: &[u32]) -> Self {
        let n = values.len();
        if n <= DENSE_THRESHOLD {
            Self::Dense(values.to_vec())
        } else if n <= PARTITION_THRESHOLD {
            Self::EliasFano(EliasFano::from_sorted(values))
        } else if n <= OPTIMAL_THRESHOLD {
            Self::Partitioned(PartitionedEliasFano::from_sorted(values))
        } else {
            Self::Optimal(OptimalPartitionedEliasFano::from_sorted(values))
        }
    }

    /// Build from sorted u64 values.
    pub fn from_sorted_u64(values: &[u64]) -> Self {
        let n = values.len();
        if n <= DENSE_THRESHOLD {
            Self::Dense(values.iter().map(|&v| v as u32).collect())
        } else if n <= PARTITION_THRESHOLD {
            Self::EliasFano(EliasFano::from_sorted_u64(values))
        } else if n <= OPTIMAL_THRESHOLD {
            Self::Partitioned(PartitionedEliasFano::from_sorted_u64(values))
        } else {
            Self::Optimal(OptimalPartitionedEliasFano::from_sorted_u64(values))
        }
    }

    /// Force a specific encoding regardless of list size.
    pub fn with_encoding(values: &[u32], encoding: PostingEncoding) -> Self {
        match encoding {
            PostingEncoding::Dense => Self::Dense(values.to_vec()),
            PostingEncoding::EliasFano => Self::EliasFano(EliasFano::from_sorted(values)),
            PostingEncoding::Partitioned => {
                Self::Partitioned(PartitionedEliasFano::from_sorted(values))
            }
            PostingEncoding::Optimal => {
                Self::Optimal(OptimalPartitionedEliasFano::from_sorted(values))
            }
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::EliasFano(ef) => ef.len(),
            Self::Partitioned(pef) => pef.len(),
            Self::Optimal(opef) => opef.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Dense(v) => v.len() * 4 + 8, // discriminant + vec overhead
            Self::EliasFano(ef) => ef.size_bytes() + 8,
            Self::Partitioned(pef) => pef.size_bytes() + 8,
            Self::Optimal(opef) => opef.size_bytes() + 8,
        }
    }

    #[inline]
    pub fn bits_per_element(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        (self.size_bytes() * 8) as f64 / self.len() as f64
    }

    /// Get the i-th element.
    pub fn get(&self, index: usize) -> Option<u64> {
        match self {
            Self::Dense(v) => v.get(index).map(|&x| x as u64),
            Self::EliasFano(ef) => ef.get(index),
            Self::Partitioned(pef) => pef.get(index),
            Self::Optimal(opef) => opef.get(index),
        }
    }

    /// Find the first element >= target.
    #[inline]
    pub fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        match self {
            Self::Dense(v) => {
                // Binary search on sorted array
                match v.binary_search(&(target as u32)) {
                    Ok(i) => Some((i, v[i] as u64)),
                    Err(i) => {
                        if i < v.len() {
                            Some((i, v[i] as u64))
                        } else {
                            None
                        }
                    }
                }
            }
            Self::EliasFano(ef) => ef.next_geq(target),
            Self::Partitioned(pef) => pef.next_geq(target),
            Self::Optimal(opef) => opef.next_geq(target),
        }
    }

    /// Which encoding was selected.
    pub fn encoding(&self) -> PostingEncoding {
        match self {
            Self::Dense(_) => PostingEncoding::Dense,
            Self::EliasFano(_) => PostingEncoding::EliasFano,
            Self::Partitioned(_) => PostingEncoding::Partitioned,
            Self::Optimal(_) => PostingEncoding::Optimal,
        }
    }
}

impl super::PostingList for HybridPostingList {
    fn len(&self) -> usize {
        self.len()
    }
    fn get(&self, index: usize) -> Option<u64> {
        self.get(index)
    }
    fn next_geq(&self, target: u64) -> Option<(usize, u64)> {
        self.next_geq(target)
    }
    fn size_bytes(&self) -> usize {
        self.size_bytes()
    }
}

/// Encoding strategy for posting lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PostingEncoding {
    Dense,
    EliasFano,
    Partitioned,
    Optimal,
}
