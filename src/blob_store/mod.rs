//! Blob storage systems
//!
//! This module provides abstract blob storage with various implementations
//! including memory-based, file-based, and compressed storage.

pub mod cached_store;
pub mod compressed;
pub mod entropy;
pub mod memory;
pub mod nest_louds_trie_blob_store;
pub mod plain;
pub mod sorted_uint_vec;
pub mod traits;
pub mod zip_offset;
pub mod zip_offset_builder;

// Re-export core types
pub use cached_store::{CachedBlobStore, CacheAwareBlobStore, BlobCacheStats};
pub use compressed::CompressionAlgorithm;
pub use entropy::{
    DictionaryBlobStore, EntropyAlgorithm, EntropyCompressionStats, HuffmanBlobStore, RansBlobStore,
};
pub use memory::MemoryBlobStore;
pub use nest_louds_trie_blob_store::{
    NestLoudsTrieBlobStore, NestLoudsTrieBlobStoreBuilder, TrieBlobStoreConfig, 
    TrieBlobStoreConfigBuilder, TrieBlobStoreStats,
};
pub use plain::PlainBlobStore;
pub use sorted_uint_vec::{SortedUintVec, SortedUintVecBuilder, SortedUintVecConfig};
pub use traits::{
    BatchBlobStore, BlobStore, BlobStoreStats, CompressedBlobStore, CompressionStats,
    IterableBlobStore,
};
pub use zip_offset::{ZipOffsetBlobStore, ZipOffsetBlobStoreConfig};
pub use zip_offset_builder::{ZipOffsetBlobStoreBuilder, BatchZipOffsetBlobStoreBuilder, BuilderStats};

#[cfg(feature = "zstd")]
pub use compressed::ZstdBlobStore;

#[cfg(feature = "lz4")]
pub use compressed::Lz4BlobStore;
