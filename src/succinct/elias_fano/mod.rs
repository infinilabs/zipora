//! Elias-Fano Encoding — quasi-succinct monotone integer sequence.

pub mod basic;
pub(crate) mod chunk;
pub mod partitioned;
pub mod optimal;
pub mod hybrid;

pub use basic::{EliasFano, EliasFanoBatchCursor, EliasFanoCursor, EliasFanoIter};
pub use hybrid::{HybridPostingList, PostingEncoding};
pub use optimal::{
    OptimalPartitionedEliasFano, OptimalPefBatchCursor, OptimalPefCursor, OptimalPefIter,
};
pub use partitioned::{
    PartitionedEliasFano, PartitionedEliasFanoBatchCursor, PartitionedEliasFanoCursor,
    PartitionedEliasFanoIter,
};

#[cfg(test)]
mod tests;
