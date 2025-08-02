//! Blob storage systems
//!
//! This module provides abstract blob storage with various implementations
//! including memory-based, file-based, and compressed storage.

pub mod compressed;
pub mod entropy;
pub mod memory;
pub mod plain;
pub mod traits;

// Re-export core types
pub use compressed::CompressionAlgorithm;
pub use entropy::{
    DictionaryBlobStore, EntropyAlgorithm, EntropyCompressionStats, HuffmanBlobStore, RansBlobStore,
};
pub use memory::MemoryBlobStore;
pub use plain::PlainBlobStore;
pub use traits::{
    BatchBlobStore, BlobStore, BlobStoreStats, CompressedBlobStore, CompressionStats,
    IterableBlobStore,
};

#[cfg(feature = "zstd")]
pub use compressed::ZstdBlobStore;

#[cfg(feature = "lz4")]
pub use compressed::Lz4BlobStore;
