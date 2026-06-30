const DENSE_THRESHOLD: usize = 64;
const PARTITION_THRESHOLD: usize = 256;
const OPTIMAL_THRESHOLD: usize = 4096;

use super::basic::EliasFano;
use super::clustered::ClusteredEliasFano;
use super::optimal::OptimalPartitionedEliasFano;
use super::partitioned::PartitionedEliasFano;

/// A list is "run-heavy" (dense/near-contiguous) when its value span is at most
/// twice its length. In this regime `ClusteredEliasFano` wins on space, `next_geq`,
/// and especially block intersection; outside it, OPEF compresses clustered/sparse
/// data better, so the size-based ladder is used instead.
const RUN_HEAVY_SPAN_FACTOR: u64 = 2;

/// Whether a sorted list of `n` elements spanning `[first, last]` is run-heavy.
#[inline]
fn is_run_heavy(first: u64, last: u64, n: usize) -> bool {
    let span = last - first + 1;
    span <= RUN_HEAVY_SPAN_FACTOR * n as u64
}

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
    /// Clustered EF for dense / run-heavy lists (best space, `next_geq`, and
    /// O(1) block intersection via [`intersect_count`](HybridPostingList::intersect_count)).
    Clustered(ClusteredEliasFano),
}

impl HybridPostingList {
    /// Build from a sorted slice, automatically selecting the best encoding.
    pub fn from_sorted(values: &[u32]) -> Self {
        let n = values.len();
        if n <= DENSE_THRESHOLD {
            Self::Dense(values.to_vec())
        } else if is_run_heavy(values[0] as u64, values[n - 1] as u64, n) {
            Self::Clustered(ClusteredEliasFano::from_sorted(values))
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
        } else if is_run_heavy(values[0], values[n - 1], n) {
            Self::Clustered(ClusteredEliasFano::from_sorted_u64(values))
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
            PostingEncoding::Clustered => {
                Self::Clustered(ClusteredEliasFano::from_sorted(values))
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
            Self::Clustered(cef) => cef.len(),
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
            Self::Clustered(cef) => cef.size_bytes() + 8,
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
            Self::Clustered(cef) => cef.get(index),
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
            Self::Clustered(cef) => cef.next_geq(target),
        }
    }

    /// Count elements common to both lists.
    ///
    /// When both lists are `Clustered`, this uses the O(1)-per-run-pair block
    /// intersection (orders of magnitude faster than leapfrog on dense data).
    /// Otherwise it falls back to a `next_geq` leapfrog, identical in result.
    pub fn intersect_count(&self, other: &Self) -> usize {
        if let (Self::Clustered(a), Self::Clustered(b)) = (self, other) {
            return a.intersect_count(b);
        }
        let mut count = 0usize;
        let mut probe = 0u64;
        while let Some((_, va)) = self.next_geq(probe) {
            match other.next_geq(va) {
                Some((_, vb)) if vb == va => {
                    count += 1;
                    probe = va + 1;
                }
                Some((_, vb)) => probe = vb,
                None => break,
            }
        }
        count
    }

    /// Which encoding was selected.
    pub fn encoding(&self) -> PostingEncoding {
        match self {
            Self::Dense(_) => PostingEncoding::Dense,
            Self::EliasFano(_) => PostingEncoding::EliasFano,
            Self::Partitioned(_) => PostingEncoding::Partitioned,
            Self::Optimal(_) => PostingEncoding::Optimal,
            Self::Clustered(_) => PostingEncoding::Clustered,
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
    /// Clustered EF — selected automatically for dense / run-heavy lists.
    Clustered,
}
