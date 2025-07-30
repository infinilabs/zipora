//! Compressed blob store implementations
//!
//! This module provides blob store implementations with various compression
//! algorithms including ZSTD, LZ4, and others for space-efficient storage.

use std::io::{self, Read, Write};

use crate::blob_store::traits::{
    BlobStore, BlobStoreStats, IterableBlobStore, BatchBlobStore,
    CompressedBlobStore, CompressionStats
};
use crate::error::{Result, ToplingError};
use crate::RecordId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// ZSTD compressed blob store wrapper
///
/// This implementation wraps another blob store and provides transparent
/// ZSTD compression/decompression of blob data.
///
/// # Examples
///
/// ```rust
/// use infini_zip::blob_store::{BlobStore, MemoryBlobStore, ZstdBlobStore};
///
/// let inner_store = MemoryBlobStore::new();
/// let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
///
/// let data = b"This is some data that will be compressed";
/// let id = compressed_store.put(data).unwrap();
/// let retrieved = compressed_store.get(id).unwrap();
/// assert_eq!(data, &retrieved[..]);
/// ```
#[cfg(feature = "zstd")]
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ZstdBlobStore<S> {
    inner: S,
    compression_level: i32,
    stats: CompressionStats,
}

#[cfg(feature = "zstd")]
impl<S: BlobStore> ZstdBlobStore<S> {
    /// Create a new ZSTD compressed blob store
    ///
    /// # Arguments
    /// * `inner` - The underlying blob store
    /// * `compression_level` - ZSTD compression level (1-22, higher = better compression)
    pub fn new(inner: S, compression_level: i32) -> Self {
        let level = compression_level.clamp(1, 22);
        Self {
            inner,
            compression_level: level,
            stats: CompressionStats::default(),
        }
    }
    
    /// Create with default compression level (3)
    pub fn with_default_compression(inner: S) -> Self {
        Self::new(inner, 3)
    }
    
    /// Get the compression level
    pub fn compression_level(&self) -> i32 {
        self.compression_level
    }
    
    /// Get a reference to the inner store
    pub fn inner(&self) -> &S {
        &self.inner
    }
    
    /// Get a mutable reference to the inner store
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
    
    /// Convert back to the inner store
    pub fn into_inner(self) -> S {
        self.inner
    }
    
    /// Compress data using ZSTD
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, self.compression_level).map_err(|e| {
            ToplingError::io_error(format!("ZSTD compression failed: {}", e))
        })
    }
    
    /// Decompress data using ZSTD
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(compressed_data).map_err(|e| {
            ToplingError::io_error(format!("ZSTD decompression failed: {}", e))
        })
    }
    
    /// Update compression statistics
    fn update_compression_stats(&mut self, original_size: usize, compressed_size: usize) {
        self.stats.uncompressed_size += original_size;
        self.stats.compressed_size += compressed_size;
        self.stats.compressed_count += 1;
        self.stats.compression_ratio = self.stats.ratio();
    }
}

#[cfg(feature = "zstd")]
impl<S: BlobStore> BlobStore for ZstdBlobStore<S> {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let compressed_data = self.inner.get(id)?;
        self.decompress(&compressed_data)
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        let compressed_data = self.compress(data)?;
        let id = self.inner.put(&compressed_data)?;
        
        self.update_compression_stats(data.len(), compressed_data.len());
        
        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        // Update stats before removal
        if let Ok(compressed_data) = self.inner.get(id) {
            if let Ok(original_data) = self.decompress(&compressed_data) {
                self.stats.uncompressed_size = self.stats.uncompressed_size.saturating_sub(original_data.len());
                self.stats.compressed_size = self.stats.compressed_size.saturating_sub(compressed_data.len());
                self.stats.compressed_count = self.stats.compressed_count.saturating_sub(1);
                self.stats.compression_ratio = self.stats.ratio();
            }
        }
        
        self.inner.remove(id)
    }

    fn contains(&self, id: RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        // Return uncompressed size
        match self.inner.get(id) {
            Ok(compressed_data) => {
                let decompressed = self.decompress(&compressed_data)?;
                Ok(Some(decompressed.len()))
            }
            Err(_) => Ok(None),
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> BlobStoreStats {
        self.inner.stats()
    }
}

#[cfg(feature = "zstd")]
impl<S: IterableBlobStore> IterableBlobStore for ZstdBlobStore<S> {
    type IdIter = S::IdIter;

    fn iter_ids(&self) -> Self::IdIter {
        self.inner.iter_ids()
    }
}

#[cfg(feature = "zstd")]
impl<S: BatchBlobStore> BatchBlobStore for ZstdBlobStore<S> {
    fn put_batch<I>(&mut self, blobs: I) -> Result<Vec<RecordId>>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let compressed_blobs: Result<Vec<Vec<u8>>> = blobs
            .into_iter()
            .map(|blob| {
                let compressed = self.compress(&blob)?;
                self.update_compression_stats(blob.len(), compressed.len());
                Ok(compressed)
            })
            .collect();
        
        self.inner.put_batch(compressed_blobs?)
    }

    fn get_batch<I>(&self, ids: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let compressed_results = self.inner.get_batch(ids)?;
        
        compressed_results
            .into_iter()
            .map(|opt_data| {
                match opt_data {
                    Some(compressed) => {
                        let decompressed = self.decompress(&compressed)?;
                        Ok(Some(decompressed))
                    }
                    None => Ok(None),
                }
            })
            .collect()
    }

    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let ids_vec: Vec<RecordId> = ids.into_iter().collect();
        
        // Update stats before removal
        for &id in &ids_vec {
            if let Ok(compressed_data) = self.inner.get(id) {
                if let Ok(original_data) = self.decompress(&compressed_data) {
                    self.stats.uncompressed_size = self.stats.uncompressed_size.saturating_sub(original_data.len());
                    self.stats.compressed_size = self.stats.compressed_size.saturating_sub(compressed_data.len());
                    self.stats.compressed_count = self.stats.compressed_count.saturating_sub(1);
                }
            }
        }
        
        let removed = self.inner.remove_batch(ids_vec)?;
        self.stats.compression_ratio = self.stats.ratio();
        Ok(removed)
    }
}

#[cfg(feature = "zstd")]
impl<S: BlobStore> CompressedBlobStore for ZstdBlobStore<S> {
    fn compression_ratio(&self, id: RecordId) -> Result<Option<f32>> {
        match (self.inner.get(id), self.inner.size(id)) {
            (Ok(compressed_data), Ok(Some(_))) => {
                let decompressed = self.decompress(&compressed_data)?;
                let ratio = compressed_data.len() as f32 / decompressed.len() as f32;
                Ok(Some(ratio))
            }
            _ => Ok(None),
        }
    }

    fn compressed_size(&self, id: RecordId) -> Result<Option<usize>> {
        match self.inner.get(id) {
            Ok(compressed_data) => Ok(Some(compressed_data.len())),
            Err(_) => Ok(None),
        }
    }

    fn compression_stats(&self) -> CompressionStats {
        self.stats.clone()
    }
}

/// LZ4 compressed blob store wrapper (when lz4 feature is enabled)
#[cfg(feature = "lz4")]
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Lz4BlobStore<S> {
    inner: S,
    stats: CompressionStats,
}

#[cfg(feature = "lz4")]
impl<S: BlobStore> Lz4BlobStore<S> {
    /// Create a new LZ4 compressed blob store
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            stats: CompressionStats::default(),
        }
    }
    
    /// Get a reference to the inner store
    pub fn inner(&self) -> &S {
        &self.inner
    }
    
    /// Convert back to the inner store
    pub fn into_inner(self) -> S {
        self.inner
    }
    
    /// Compress data using LZ4
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }
    
    /// Decompress data using LZ4
    fn decompress(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(compressed_data).map_err(|e| {
            ToplingError::io_error(format!("LZ4 decompression failed: {}", e))
        })
    }
    
    /// Update compression statistics
    fn update_compression_stats(&mut self, original_size: usize, compressed_size: usize) {
        self.stats.uncompressed_size += original_size;
        self.stats.compressed_size += compressed_size;
        self.stats.compressed_count += 1;
        self.stats.compression_ratio = self.stats.ratio();
    }
}

#[cfg(feature = "lz4")]
impl<S: BlobStore> BlobStore for Lz4BlobStore<S> {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let compressed_data = self.inner.get(id)?;
        self.decompress(&compressed_data)
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        let compressed_data = self.compress(data)?;
        let id = self.inner.put(&compressed_data)?;
        
        self.update_compression_stats(data.len(), compressed_data.len());
        
        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        self.inner.remove(id)
    }

    fn contains(&self, id: RecordId) -> bool {
        self.inner.contains(id)
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        match self.inner.get(id) {
            Ok(compressed_data) => {
                let decompressed = self.decompress(&compressed_data)?;
                Ok(Some(decompressed.len()))
            }
            Err(_) => Ok(None),
        }
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    fn stats(&self) -> BlobStoreStats {
        self.inner.stats()
    }
}

#[cfg(feature = "lz4")]
impl<S: BlobStore> CompressedBlobStore for Lz4BlobStore<S> {
    fn compression_ratio(&self, id: RecordId) -> Result<Option<f32>> {
        match (self.inner.get(id), self.inner.size(id)) {
            (Ok(compressed_data), Ok(Some(_))) => {
                let decompressed = self.decompress(&compressed_data)?;
                let ratio = compressed_data.len() as f32 / decompressed.len() as f32;
                Ok(Some(ratio))
            }
            _ => Ok(None),
        }
    }

    fn compressed_size(&self, id: RecordId) -> Result<Option<usize>> {
        match self.inner.get(id) {
            Ok(compressed_data) => Ok(Some(compressed_data.len())),
            Err(_) => Ok(None),
        }
    }

    fn compression_stats(&self) -> CompressionStats {
        self.stats.clone()
    }
}

/// Generic compression wrapper that can use different algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompressionAlgorithm {
    #[cfg(feature = "zstd")]
    Zstd { level: i32 },
    #[cfg(feature = "lz4")]
    Lz4,
    None,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        #[cfg(feature = "zstd")]
        return Self::Zstd { level: 3 };
        
        #[cfg(all(feature = "lz4", not(feature = "zstd")))]
        return Self::Lz4;
        
        #[cfg(not(any(feature = "zstd", feature = "lz4")))]
        return Self::None;
    }
}

impl CompressionAlgorithm {
    /// Get the name of the compression algorithm
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "zstd")]
            Self::Zstd { .. } => "zstd",
            #[cfg(feature = "lz4")]
            Self::Lz4 => "lz4",
            Self::None => "none",
        }
    }
    
    /// Check if this algorithm provides compression
    pub fn is_compressed(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::MemoryBlobStore;

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_basic() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        let data = b"Hello, compressed world! This is some test data that should compress well.";
        let id = compressed_store.put(data).unwrap();
        
        assert_eq!(compressed_store.len(), 1);
        assert!(compressed_store.contains(id));
        
        let retrieved = compressed_store.get(id).unwrap();
        assert_eq!(data, &retrieved[..]);
        
        // Check uncompressed size
        let size = compressed_store.size(id).unwrap();
        assert_eq!(size, Some(data.len()));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_compression_stats() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 9); // High compression
        
        // Use repetitive data that should compress well
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let id = compressed_store.put(data).unwrap();
        
        let stats = compressed_store.compression_stats();
        assert_eq!(stats.uncompressed_size, data.len());
        assert!(stats.compressed_size < data.len()); // Should be compressed
        assert_eq!(stats.compressed_count, 1);
        assert!(stats.ratio() < 1.0); // Good compression
        
        // Check individual blob stats
        let ratio = compressed_store.compression_ratio(id).unwrap();
        assert!(ratio.is_some());
        assert!(ratio.unwrap() < 1.0);
        
        let compressed_size = compressed_store.compressed_size(id).unwrap();
        assert!(compressed_size.is_some());
        assert!(compressed_size.unwrap() < data.len());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_batch_operations() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        let blobs = vec![
            b"First blob data".to_vec(),
            b"Second blob data".to_vec(),
            b"Third blob data".to_vec(),
        ];
        
        let ids = compressed_store.put_batch(blobs.clone()).unwrap();
        assert_eq!(ids.len(), 3);
        
        let retrieved = compressed_store.get_batch(ids.clone()).unwrap();
        assert_eq!(retrieved.len(), 3);
        
        for (i, blob_opt) in retrieved.iter().enumerate() {
            assert!(blob_opt.is_some());
            assert_eq!(blob_opt.as_ref().unwrap(), &blobs[i]);
        }
        
        let removed_count = compressed_store.remove_batch(ids).unwrap();
        assert_eq!(removed_count, 3);
        assert_eq!(compressed_store.len(), 0);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_remove_updates_stats() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        let data = b"Test data for removal";
        let id = compressed_store.put(data).unwrap();
        
        let stats_before = compressed_store.compression_stats();
        assert_eq!(stats_before.compressed_count, 1);
        
        compressed_store.remove(id).unwrap();
        
        let stats_after = compressed_store.compression_stats();
        assert_eq!(stats_after.compressed_count, 0);
        assert_eq!(stats_after.uncompressed_size, 0);
        assert_eq!(stats_after.compressed_size, 0);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_different_compression_levels() {
        let data = b"This is test data that will be compressed at different levels.";
        
        // Test with different compression levels
        for level in [1, 3, 9, 19] {
            let inner_store = MemoryBlobStore::new();
            let mut compressed_store = ZstdBlobStore::new(inner_store, level);
            
            let id = compressed_store.put(data).unwrap();
            let retrieved = compressed_store.get(id).unwrap();
            
            assert_eq!(data, &retrieved[..]);
            assert_eq!(compressed_store.compression_level(), level);
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_with_default_compression() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::with_default_compression(inner_store);
        
        assert_eq!(compressed_store.compression_level(), 3);
        
        let data = b"Test data";
        let id = compressed_store.put(data).unwrap();
        let retrieved = compressed_store.get(id).unwrap();
        
        assert_eq!(data, &retrieved[..]);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_inner_access() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        // Test inner access
        let _inner_ref = compressed_store.inner();
        let _inner_mut_ref = compressed_store.inner_mut();
        
        let inner_store = compressed_store.into_inner();
        assert_eq!(inner_store.len(), 0);
    }

    #[cfg(feature = "lz4")]
    #[test]
    fn test_lz4_blob_store_basic() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = Lz4BlobStore::new(inner_store);
        
        let data = b"Hello, LZ4 compressed world! This is test data.";
        let id = compressed_store.put(data).unwrap();
        
        assert_eq!(compressed_store.len(), 1);
        assert!(compressed_store.contains(id));
        
        let retrieved = compressed_store.get(id).unwrap();
        assert_eq!(data, &retrieved[..]);
    }

    #[test]
    fn test_compression_algorithm() {
        let default_algo = CompressionAlgorithm::default();
        assert!(default_algo.is_compressed() || matches!(default_algo, CompressionAlgorithm::None));
        
        #[cfg(feature = "zstd")]
        {
            let zstd_algo = CompressionAlgorithm::Zstd { level: 5 };
            assert_eq!(zstd_algo.name(), "zstd");
            assert!(zstd_algo.is_compressed());
        }
        
        #[cfg(feature = "lz4")]
        {
            let lz4_algo = CompressionAlgorithm::Lz4;
            assert_eq!(lz4_algo.name(), "lz4");
            assert!(lz4_algo.is_compressed());
        }
        
        let none_algo = CompressionAlgorithm::None;
        assert_eq!(none_algo.name(), "none");
        assert!(!none_algo.is_compressed());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_compression_error_handling() {
        let inner_store = MemoryBlobStore::new();
        let compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        // Test decompression of invalid data
        let invalid_compressed = vec![0xFF, 0xFF, 0xFF, 0xFF];
        let result = compressed_store.decompress(&invalid_compressed);
        assert!(result.is_err());
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn test_zstd_blob_store_iteration() {
        let inner_store = MemoryBlobStore::new();
        let mut compressed_store = ZstdBlobStore::new(inner_store, 3);
        
        let data1 = b"First blob";
        let data2 = b"Second blob";
        
        let id1 = compressed_store.put(data1).unwrap();
        let id2 = compressed_store.put(data2).unwrap();
        
        let ids: Vec<RecordId> = compressed_store.iter_ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }
}