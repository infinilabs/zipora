//! Elias-Fano Encoding — quasi-succinct monotone integer sequence.

pub mod basic;
pub(crate) mod chunk;
pub mod hybrid;
pub mod optimal;
pub mod partitioned;

pub use basic::{EliasFano, EliasFanoBatchCursor, EliasFanoCursor, EliasFanoIter};
pub use hybrid::{HybridPostingList, PostingEncoding};
pub use optimal::{
    OptimalPartitionedEliasFano, OptimalPefBatchCursor, OptimalPefCursor, OptimalPefIter,
};
pub use partitioned::{
    PartitionedEliasFano, PartitionedEliasFanoBatchCursor, PartitionedEliasFanoCursor,
    PartitionedEliasFanoIter,
};

/// Common interface for all monotone integer sequence (posting list) encodings.
pub trait PostingList {
    /// Number of elements in the list.
    fn len(&self) -> usize;
    
    /// Whether the list is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the i-th element.
    fn get(&self, index: usize) -> Option<u64>;
    
    /// Find the first element >= target. Returns (index, value).
    fn next_geq(&self, target: u64) -> Option<(usize, u64)>;
    
    /// Estimated memory usage in bytes.
    fn size_bytes(&self) -> usize;
}

#[cfg(test)]
mod tests;
