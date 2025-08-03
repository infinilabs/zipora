//! Core blob store traits and abstractions
//!
//! This module defines the fundamental abstractions for blob storage systems,
//! providing a unified interface for different storage backends.

use crate::error::Result;
use crate::RecordId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Core trait for blob storage operations
///
/// BlobStore provides the fundamental operations for storing and retrieving
/// arbitrary binary data with unique identifiers.
pub trait BlobStore {
    /// Retrieve a blob by its record ID
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the blob
    ///
    /// # Returns
    /// * `Ok(Vec<u8>)` - The blob data if found
    /// * `Err(ZiporaError)` - If the blob doesn't exist or other error
    fn get(&self, id: RecordId) -> Result<Vec<u8>>;

    /// Store a blob and return its unique ID
    ///
    /// # Arguments
    /// * `data` - The binary data to store
    ///
    /// # Returns
    /// * `Ok(RecordId)` - The unique identifier for the stored blob
    /// * `Err(ZiporaError)` - If storage fails
    fn put(&mut self, data: &[u8]) -> Result<RecordId>;

    /// Remove a blob by its record ID
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the blob to remove
    ///
    /// # Returns
    /// * `Ok(())` - If removal was successful
    /// * `Err(ZiporaError)` - If the blob doesn't exist or removal fails
    fn remove(&mut self, id: RecordId) -> Result<()>;

    /// Check if a blob exists
    ///
    /// # Arguments
    /// * `id` - The unique identifier to check
    ///
    /// # Returns
    /// * `true` if the blob exists, `false` otherwise
    fn contains(&self, id: RecordId) -> bool;

    /// Get the size of a blob without retrieving its data
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the blob
    ///
    /// # Returns
    /// * `Ok(Some(size))` - The size of the blob if it exists
    /// * `Ok(None)` - If the blob doesn't exist
    /// * `Err(ZiporaError)` - If size query fails
    fn size(&self, id: RecordId) -> Result<Option<usize>>;

    /// Get the total number of blobs stored
    fn len(&self) -> usize;

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Flush any pending operations to storage
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get storage statistics
    fn stats(&self) -> BlobStoreStats {
        BlobStoreStats::default()
    }
}

/// Statistics about blob store usage
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BlobStoreStats {
    /// Total number of blobs
    pub blob_count: usize,
    /// Total size of all blob data in bytes
    pub total_size: usize,
    /// Average blob size in bytes
    pub average_size: f64,
    /// Number of get operations
    pub get_count: u64,
    /// Number of put operations
    pub put_count: u64,
    /// Number of remove operations
    pub remove_count: u64,
    /// Cache hit ratio (0.0 to 1.0)
    pub cache_hit_ratio: f64,
}

impl BlobStoreStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics after a get operation
    pub fn record_get(&mut self, hit: bool) {
        self.get_count += 1;
        if hit {
            self.cache_hit_ratio =
                (self.cache_hit_ratio * (self.get_count - 1) as f64 + 1.0) / self.get_count as f64;
        } else {
            self.cache_hit_ratio =
                (self.cache_hit_ratio * (self.get_count - 1) as f64) / self.get_count as f64;
        }
    }

    /// Update statistics after a put operation
    pub fn record_put(&mut self, size: usize) {
        self.put_count += 1;
        self.blob_count += 1;
        self.total_size += size;
        self.average_size = self.total_size as f64 / self.blob_count as f64;
    }

    /// Update statistics after a remove operation
    pub fn record_remove(&mut self, size: usize) {
        self.remove_count += 1;
        if self.blob_count > 0 {
            self.blob_count -= 1;
            self.total_size = self.total_size.saturating_sub(size);
            self.average_size = if self.blob_count > 0 {
                self.total_size as f64 / self.blob_count as f64
            } else {
                0.0
            };
        }
    }
}

/// Trait for blob stores that support iteration
pub trait IterableBlobStore: BlobStore {
    /// Iterator over all record IDs
    type IdIter: Iterator<Item = RecordId>;

    /// Get an iterator over all record IDs
    fn iter_ids(&self) -> Self::IdIter;

    /// Get an iterator over all blobs as (id, data) pairs
    fn iter_blobs(&self) -> BlobIterator<'_, Self> {
        BlobIterator::new(self)
    }
}

/// Iterator over blob data
pub struct BlobIterator<'a, S: BlobStore + ?Sized> {
    store: &'a S,
    ids: Box<dyn Iterator<Item = RecordId> + 'a>,
}

impl<'a, S: IterableBlobStore + ?Sized> BlobIterator<'a, S> {
    fn new(store: &'a S) -> Self {
        Self {
            store,
            ids: Box::new(store.iter_ids()),
        }
    }
}

impl<'a, S: BlobStore + ?Sized> Iterator for BlobIterator<'a, S> {
    type Item = Result<(RecordId, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.ids
            .next()
            .map(|id| self.store.get(id).map(|data| (id, data)))
    }
}

/// Trait for blob stores that support batched operations
pub trait BatchBlobStore: BlobStore {
    /// Put multiple blobs in a single operation
    ///
    /// # Arguments
    /// * `blobs` - Iterator over blob data
    ///
    /// # Returns
    /// * `Ok(Vec<RecordId>)` - The record IDs for all stored blobs
    /// * `Err(ZiporaError)` - If batch operation fails
    fn put_batch<I>(&mut self, blobs: I) -> Result<Vec<RecordId>>
    where
        I: IntoIterator<Item = Vec<u8>>;

    /// Get multiple blobs in a single operation
    ///
    /// # Arguments
    /// * `ids` - Iterator over record IDs
    ///
    /// # Returns
    /// * `Ok(Vec<Option<Vec<u8>>>)` - The blob data for each ID (None if not found)
    /// * `Err(ZiporaError)` - If batch operation fails
    fn get_batch<I>(&self, ids: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = RecordId>;

    /// Remove multiple blobs in a single operation
    ///
    /// # Arguments
    /// * `ids` - Iterator over record IDs to remove
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of blobs actually removed
    /// * `Err(ZiporaError)` - If batch operation fails
    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>;
}

/// Trait for blob stores that support compression
pub trait CompressedBlobStore: BlobStore {
    /// Get the compression ratio for a specific blob
    fn compression_ratio(&self, id: RecordId) -> Result<Option<f32>>;

    /// Get the compressed size of a blob
    fn compressed_size(&self, id: RecordId) -> Result<Option<usize>>;

    /// Get overall compression statistics
    fn compression_stats(&self) -> CompressionStats;
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompressionStats {
    /// Total uncompressed size in bytes
    pub uncompressed_size: usize,
    /// Total compressed size in bytes
    pub compressed_size: usize,
    /// Overall compression ratio
    pub compression_ratio: f32,
    /// Number of compressed blobs
    pub compressed_count: usize,
}

impl CompressionStats {
    /// Calculate compression ratio
    pub fn ratio(&self) -> f32 {
        if self.uncompressed_size > 0 {
            self.compressed_size as f32 / self.uncompressed_size as f32
        } else {
            1.0
        }
    }

    /// Calculate space saved as percentage
    pub fn space_saved_percent(&self) -> f32 {
        (1.0 - self.ratio()) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blob_store_stats() {
        let mut stats = BlobStoreStats::new();

        // Test put recording
        stats.record_put(100);
        assert_eq!(stats.blob_count, 1);
        assert_eq!(stats.total_size, 100);
        assert_eq!(stats.average_size, 100.0);
        assert_eq!(stats.put_count, 1);

        stats.record_put(200);
        assert_eq!(stats.blob_count, 2);
        assert_eq!(stats.total_size, 300);
        assert_eq!(stats.average_size, 150.0);

        // Test remove recording
        stats.record_remove(100);
        assert_eq!(stats.blob_count, 1);
        assert_eq!(stats.total_size, 200);
        assert_eq!(stats.average_size, 200.0);
        assert_eq!(stats.remove_count, 1);

        // Test get recording
        stats.record_get(true); // hit
        stats.record_get(false); // miss
        stats.record_get(true); // hit
        assert_eq!(stats.get_count, 3);
        assert!((stats.cache_hit_ratio - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_compression_stats() {
        let stats = CompressionStats {
            uncompressed_size: 1000,
            compressed_size: 300,
            compression_ratio: 0.3,
            compressed_count: 10,
        };

        assert_eq!(stats.ratio(), 0.3);
        assert_eq!(stats.space_saved_percent(), 70.0);

        let empty_stats = CompressionStats::default();
        assert_eq!(empty_stats.ratio(), 1.0);
        assert_eq!(empty_stats.space_saved_percent(), 0.0);
    }
}
