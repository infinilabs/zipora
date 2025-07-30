//! Blob storage systems
//!
//! This module provides abstract blob storage with various implementations
//! including memory-based, file-based, and compressed storage.

pub mod traits;
pub mod memory;
pub mod plain;
pub mod compressed;

// Re-export core types
pub use traits::{
    BlobStore, BlobStoreStats, IterableBlobStore, BatchBlobStore, 
    CompressedBlobStore, CompressionStats
};
pub use memory::MemoryBlobStore;
pub use plain::PlainBlobStore;
pub use compressed::CompressionAlgorithm;

#[cfg(feature = "zstd")]
pub use compressed::ZstdBlobStore;

#[cfg(feature = "lz4")]
pub use compressed::Lz4BlobStore;