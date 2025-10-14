//! Zero-length blob store implementation
//!
//! This module provides an optimized blob store for storing only empty blobs
//! (zero-length data). This is a degenerate case that saves maximum memory by
//! storing only the record count.
//!
//! # Use Cases
//!
//! - Placeholder records where only existence matters
//! - Sparse indexes where only presence/absence is tracked
//! - Test data generation with minimal overhead
//! - Record counting without data storage
//!
//! # Memory Efficiency
//!
//! Unlike other blob stores, ZeroLengthBlobStore has:
//! - Zero memory footprint for blob data
//! - Only stores the record count
//! - O(1) operations with minimal overhead
//!
//! # Examples
//!
//! ```rust
//! use zipora::blob_store::{BlobStore, ZeroLengthBlobStore};
//!
//! let mut store = ZeroLengthBlobStore::new();
//!
//! // Can only store empty blobs
//! let id1 = store.put(&[]).unwrap();
//! let id2 = store.put(&[]).unwrap();
//!
//! // Attempting to store non-empty data fails
//! assert!(store.put(b"data").is_err());
//!
//! // Retrieval always returns empty data
//! let data = store.get(id1).unwrap();
//! assert_eq!(data.len(), 0);
//!
//! // Memory usage is minimal
//! assert_eq!(store.mem_size(), 0);
//! ```

use crate::RecordId;
use crate::blob_store::traits::{BatchBlobStore, BlobStore, BlobStoreStats, IterableBlobStore};
use crate::error::{Result, ZiporaError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Optimized blob store for zero-length blobs only
///
/// This implementation stores only the count of empty records with zero
/// memory overhead for data storage. All blobs have zero length.
///
/// # Design
///
/// - Stores only `num_records` (single usize)
/// - Zero bytes for blob data storage
/// - All `get()` operations return empty Vec
/// - `put()` only accepts empty slices
///
/// # Performance Characteristics
///
/// - Memory: O(1) - constant overhead regardless of record count
/// - Get: O(1) - bounds check only
/// - Put: O(1) - increment counter
/// - Size: O(1) - always returns 0
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ZeroLengthBlobStore {
    /// Number of zero-length records stored
    num_records: usize,
    /// Usage statistics
    stats: BlobStoreStats,
}

impl ZeroLengthBlobStore {
    /// Create a new empty zero-length blob store
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::blob_store::ZeroLengthBlobStore;
    ///
    /// let store = ZeroLengthBlobStore::new();
    /// assert_eq!(store.len(), 0);
    /// assert!(store.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            num_records: 0,
            stats: BlobStoreStats::new(),
        }
    }

    /// Create a blob store with a pre-defined number of empty records
    ///
    /// This is useful when loading from serialized data or when you know
    /// the record count in advance.
    ///
    /// # Arguments
    ///
    /// * `records` - Number of empty records to initialize
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::blob_store::ZeroLengthBlobStore;
    ///
    /// let store = ZeroLengthBlobStore::finish(1000);
    /// assert_eq!(store.len(), 1000);
    /// ```
    pub fn finish(records: usize) -> Self {
        let mut stats = BlobStoreStats::new();
        // Record all puts for statistics
        for _ in 0..records {
            stats.record_put(0);
        }

        Self {
            num_records: records,
            stats,
        }
    }

    /// Get the memory size of this blob store in bytes
    ///
    /// Always returns 0 since no blob data is stored.
    ///
    /// # Returns
    ///
    /// Always 0 - this store has zero memory footprint for data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use zipora::blob_store::ZeroLengthBlobStore;
    ///
    /// let store = ZeroLengthBlobStore::finish(1_000_000);
    /// assert_eq!(store.mem_size(), 0); // No data memory used
    /// ```
    pub fn mem_size(&self) -> usize {
        0
    }
}

impl BlobStore for ZeroLengthBlobStore {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        if (id as usize) < self.num_records {
            // Return empty vector for valid IDs
            Ok(Vec::new())
        } else {
            Err(ZiporaError::not_found(format!(
                "Record ID {} not found (total records: {})",
                id, self.num_records
            )))
        }
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        // Only accept empty blobs
        if !data.is_empty() {
            return Err(ZiporaError::invalid_parameter(format!(
                "ZeroLengthBlobStore only accepts empty blobs, got {} bytes",
                data.len()
            )));
        }

        let id = self.num_records as RecordId;
        self.num_records += 1;
        self.stats.record_put(0);
        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        // Cannot remove from a zero-length store as it would create gaps
        // in the record ID sequence
        if (id as usize) < self.num_records {
            Err(ZiporaError::not_supported(
                "ZeroLengthBlobStore does not support removal (would create ID gaps)"
            ))
        } else {
            Err(ZiporaError::not_found(format!(
                "Record ID {} not found (total records: {})",
                id, self.num_records
            )))
        }
    }

    fn contains(&self, id: RecordId) -> bool {
        (id as usize) < self.num_records
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        if (id as usize) < self.num_records {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    }

    fn len(&self) -> usize {
        self.num_records
    }

    fn stats(&self) -> BlobStoreStats {
        self.stats.clone()
    }
}

impl IterableBlobStore for ZeroLengthBlobStore {
    type IdIter = std::ops::Range<RecordId>;

    fn iter_ids(&self) -> Self::IdIter {
        0..(self.num_records as RecordId)
    }
}

impl BatchBlobStore for ZeroLengthBlobStore {
    fn put_batch<I>(&mut self, blobs: I) -> Result<Vec<RecordId>>
    where
        I: IntoIterator<Item = Vec<u8>>,
    {
        let mut ids = Vec::new();
        for blob in blobs {
            let id = self.put(&blob)?;
            ids.push(id);
        }
        Ok(ids)
    }

    fn get_batch<I>(&self, ids: I) -> Result<Vec<Option<Vec<u8>>>>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let mut results = Vec::new();
        for id in ids {
            if (id as usize) < self.num_records {
                results.push(Some(Vec::new()));
            } else {
                results.push(None);
            }
        }
        Ok(results)
    }

    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        // Count how many valid IDs were provided, but don't actually remove
        // (removal not supported)
        let mut count = 0;
        for id in ids {
            if (id as usize) < self.num_records {
                count += 1;
            }
        }

        if count > 0 {
            Err(ZiporaError::not_supported(
                "ZeroLengthBlobStore does not support removal"
            ))
        } else {
            Ok(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_length_basic() {
        let mut store = ZeroLengthBlobStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.mem_size(), 0);
    }

    #[test]
    fn test_zero_length_put_get() {
        let mut store = ZeroLengthBlobStore::new();

        // Can only store empty blobs
        let id1 = store.put(&[]).unwrap();
        assert_eq!(id1, 0);
        assert_eq!(store.len(), 1);

        let id2 = store.put(&[]).unwrap();
        assert_eq!(id2, 1);
        assert_eq!(store.len(), 2);

        // Get returns empty vectors
        let data1 = store.get(id1).unwrap();
        assert_eq!(data1.len(), 0);

        let data2 = store.get(id2).unwrap();
        assert_eq!(data2.len(), 0);
    }

    #[test]
    fn test_zero_length_rejects_non_empty() {
        let mut store = ZeroLengthBlobStore::new();

        // Reject non-empty data
        let result = store.put(b"data");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only accepts empty blobs"));

        // Store should remain empty
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_zero_length_bounds_checking() {
        let mut store = ZeroLengthBlobStore::new();
        store.put(&[]).unwrap();
        store.put(&[]).unwrap();

        // Valid IDs
        assert!(store.contains(0));
        assert!(store.contains(1));
        assert!(store.get(0).is_ok());
        assert!(store.get(1).is_ok());

        // Invalid IDs
        assert!(!store.contains(2));
        assert!(!store.contains(999));
        assert!(store.get(2).is_err());
        assert!(store.get(999).is_err());
    }

    #[test]
    fn test_zero_length_size_query() {
        let mut store = ZeroLengthBlobStore::new();
        let id = store.put(&[]).unwrap();

        // Size is always 0 for valid IDs
        assert_eq!(store.size(id).unwrap(), Some(0));

        // None for invalid IDs
        assert_eq!(store.size(999).unwrap(), None);
    }

    #[test]
    fn test_zero_length_finish() {
        let store = ZeroLengthBlobStore::finish(1000);
        assert_eq!(store.len(), 1000);
        assert_eq!(store.mem_size(), 0);

        // All IDs should be valid
        for i in 0..1000 {
            assert!(store.contains(i));
            assert_eq!(store.get(i).unwrap().len(), 0);
        }

        // Out of bounds
        assert!(!store.contains(1000));
        assert!(store.get(1000).is_err());
    }

    #[test]
    fn test_zero_length_removal_not_supported() {
        let mut store = ZeroLengthBlobStore::new();
        let id = store.put(&[]).unwrap();

        // Removal should fail
        let result = store.remove(id);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not support"));

        // Record should still exist
        assert!(store.contains(id));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_zero_length_iteration() {
        let mut store = ZeroLengthBlobStore::new();
        for _ in 0..10 {
            store.put(&[]).unwrap();
        }

        // Test ID iteration
        let ids: Vec<RecordId> = store.iter_ids().collect();
        assert_eq!(ids.len(), 10);
        assert_eq!(ids, (0..10).collect::<Vec<RecordId>>());

        // Test blob iteration
        let blobs: Result<Vec<(RecordId, Vec<u8>)>> = store.iter_blobs().collect();
        let blobs = blobs.unwrap();
        assert_eq!(blobs.len(), 10);

        for (id, data) in blobs {
            assert!(id < 10);
            assert_eq!(data.len(), 0);
        }
    }

    #[test]
    fn test_zero_length_batch_operations() {
        let mut store = ZeroLengthBlobStore::new();

        // Batch put with empty blobs
        let empty_blobs = vec![Vec::new(), Vec::new(), Vec::new()];
        let ids = store.put_batch(empty_blobs).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids, vec![0, 1, 2]);

        // Batch put with non-empty blob should fail
        let mixed_blobs = vec![Vec::new(), b"data".to_vec()];
        let result = store.put_batch(mixed_blobs);
        assert!(result.is_err());

        // Batch get
        let retrieved = store.get_batch(vec![0, 1, 2, 999]).unwrap();
        assert_eq!(retrieved.len(), 4);
        assert_eq!(retrieved[0], Some(Vec::new()));
        assert_eq!(retrieved[1], Some(Vec::new()));
        assert_eq!(retrieved[2], Some(Vec::new()));
        assert_eq!(retrieved[3], None);

        // Batch remove not supported
        let result = store.remove_batch(vec![0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_length_stats() {
        let mut store = ZeroLengthBlobStore::new();

        for _ in 0..5 {
            store.put(&[]).unwrap();
        }

        let stats = store.stats();
        assert_eq!(stats.blob_count, 5);
        assert_eq!(stats.total_size, 0); // All blobs are zero-length
        assert_eq!(stats.average_size, 0.0);
        assert_eq!(stats.put_count, 5);
    }

    #[test]
    fn test_zero_length_large_count() {
        // Test with large number of records
        let count = 1_000_000;
        let store = ZeroLengthBlobStore::finish(count);

        assert_eq!(store.len(), count);
        assert_eq!(store.mem_size(), 0);

        // Spot check some IDs
        assert!(store.contains(0));
        assert!(store.contains((count / 2) as u32));
        assert!(store.contains((count - 1) as u32));
        assert!(!store.contains(count as u32));
    }

    #[test]
    fn test_zero_length_clone() {
        let mut store = ZeroLengthBlobStore::new();
        store.put(&[]).unwrap();
        store.put(&[]).unwrap();

        let cloned = store.clone();
        assert_eq!(cloned.len(), store.len());
        assert_eq!(cloned.mem_size(), store.mem_size());

        // Both should have same data
        assert!(cloned.contains(0));
        assert!(cloned.contains(1));
    }

    #[test]
    fn test_zero_length_default() {
        let store: ZeroLengthBlobStore = Default::default();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_zero_length_edge_cases() {
        let store = ZeroLengthBlobStore::new();

        // Empty store queries
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert!(!store.contains(0));
        assert!(store.get(0).is_err());
        assert_eq!(store.size(0).unwrap(), None);

        // Iterator on empty store
        let ids: Vec<RecordId> = store.iter_ids().collect();
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_zero_length_memory_efficiency() {
        // Demonstrate memory efficiency compared to storing actual empty vecs
        let store = ZeroLengthBlobStore::finish(10_000);

        // Memory for blob data is zero
        assert_eq!(store.mem_size(), 0);

        // Struct overhead is minimal (just usize + stats)
        let struct_size = std::mem::size_of::<ZeroLengthBlobStore>();
        assert!(struct_size < 200); // Should be very small

        println!(
            "ZeroLengthBlobStore with 10k records: {} bytes struct overhead, 0 bytes data",
            struct_size
        );
    }
}
