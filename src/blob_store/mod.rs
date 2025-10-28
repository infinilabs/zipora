//! Blob storage systems
//!
//! This module provides abstract blob storage with various implementations
//! including memory-based, file-based, and compressed storage.

pub mod cached_store;
pub mod compressed;
pub mod entropy;
pub mod memory;
pub mod mixed_len;
pub mod nest_louds_trie_blob_store;
pub mod plain;
pub mod reorder_map;
pub mod simple_zip;
pub mod sorted_uint_vec;
pub mod traits;
pub mod zero_length;
pub mod zip_offset;
pub mod zip_offset_builder;

// Re-export core types
pub use cached_store::{CachedBlobStore, CacheAwareBlobStore, BlobCacheStats};
pub use compressed::CompressionAlgorithm;
pub use entropy::{
    DictionaryBlobStore, EntropyAlgorithm, EntropyCompressionStats, HuffmanBlobStore, RansBlobStore,
};
pub use memory::MemoryBlobStore;
pub use mixed_len::MixedLenBlobStore;
pub use nest_louds_trie_blob_store::{
    NestLoudsTrieBlobStore, NestLoudsTrieBlobStoreBuilder, TrieBlobStoreConfig,
    TrieBlobStoreConfigBuilder, TrieBlobStoreStats,
};
pub use plain::PlainBlobStore;
pub use reorder_map::{ZReorderMap, ZReorderMapBuilder};
pub use simple_zip::{SimpleZipBlobStore, SimpleZipConfig, SimpleZipConfigBuilder};
pub use sorted_uint_vec::{SortedUintVec, SortedUintVecBuilder, SortedUintVecConfig};
pub use traits::{
    BatchBlobStore, BlobStore, BlobStoreStats, CompressedBlobStore, CompressionStats,
    IterableBlobStore,
};
pub use zero_length::ZeroLengthBlobStore;
pub use zip_offset::{ZipOffsetBlobStore, ZipOffsetBlobStoreConfig};
pub use zip_offset_builder::{ZipOffsetBlobStoreBuilder, BatchZipOffsetBlobStoreBuilder, BuilderStats};

// Re-export PA-Zip dictionary compression blob store
pub use crate::compression::dict_zip::{
    DictZipBlobStore, DictZipBlobStoreBuilder, DictZipBlobStoreStats, DictZipConfig
};

#[cfg(feature = "zstd")]
pub use compressed::ZstdBlobStore;

#[cfg(feature = "lz4")]
pub use compressed::Lz4BlobStore;
