//! In-memory blob store implementation
//!
//! This module provides a simple in-memory blob store primarily for testing
//! and development purposes. Data is stored in memory and will be lost when
//! the program exits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::blob_store::traits::{BlobStore, BlobStoreStats, IterableBlobStore, BatchBlobStore};
use crate::error::{Result, ToplingError};
use crate::RecordId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// In-memory blob store implementation
///
/// This implementation stores all blobs in memory using a HashMap.
/// It's primarily intended for testing and development use cases.
///
/// # Examples
///
/// ```rust
/// use infini_zip::blob_store::{BlobStore, MemoryBlobStore};
///
/// let mut store = MemoryBlobStore::new();
/// let data = b"hello world";
/// let id = store.put(data).unwrap();
/// let retrieved = store.get(id).unwrap();
/// assert_eq!(data, &retrieved[..]);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryBlobStore {
    /// Storage for blob data
    data: HashMap<RecordId, Vec<u8>>,
    /// Next available record ID
    next_id: AtomicU32,
    /// Usage statistics
    stats: BlobStoreStats,
}

impl MemoryBlobStore {
    /// Create a new empty memory blob store
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            next_id: AtomicU32::new(1), // Start from 1, 0 is reserved for "null"
            stats: BlobStoreStats::new(),
        }
    }

    /// Create a new memory blob store with the specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::with_capacity(capacity),
            next_id: AtomicU32::new(1),
            stats: BlobStoreStats::new(),
        }
    }

    /// Create a memory blob store from existing data
    pub fn from_data(data: HashMap<RecordId, Vec<u8>>) -> Self {
        let next_id = data.keys().max().map(|&id| id + 1).unwrap_or(1);
        let mut stats = BlobStoreStats::new();
        
        // Initialize stats from existing data
        for blob in data.values() {
            stats.record_put(blob.len());
        }
        
        Self {
            data,
            next_id: AtomicU32::new(next_id),
            stats,
        }
    }

    /// Get the capacity of the underlying HashMap
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Reserve space for additional blobs
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Shrink the capacity to fit the current number of blobs
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Clear all blobs from the store
    pub fn clear(&mut self) {
        self.data.clear();
        self.next_id.store(1, Ordering::Relaxed);
        self.stats = BlobStoreStats::new();
    }

    /// Get a reference to the internal data (for testing)
    #[cfg(test)]
    pub(crate) fn internal_data(&self) -> &HashMap<RecordId, Vec<u8>> {
        &self.data
    }

    /// Generate the next record ID
    fn next_record_id(&self) -> RecordId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

impl Clone for MemoryBlobStore {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            next_id: AtomicU32::new(self.next_id.load(Ordering::Relaxed)),
            stats: self.stats.clone(),
        }
    }
}

impl Default for MemoryBlobStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BlobStore for MemoryBlobStore {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        self.data.get(&id)
            .cloned()
            .ok_or_else(|| ToplingError::not_found(format!("Blob with ID {} not found", id)))
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        let id = self.next_record_id();
        let blob_data = data.to_vec();
        self.data.insert(id, blob_data);
        self.stats.record_put(data.len());
        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        match self.data.remove(&id) {
            Some(data) => {
                self.stats.record_remove(data.len());
                Ok(())
            }
            None => Err(ToplingError::not_found(format!("Blob with ID {} not found", id)))
        }
    }

    fn contains(&self, id: RecordId) -> bool {
        self.data.contains_key(&id)
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        Ok(self.data.get(&id).map(|data| data.len()))
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn stats(&self) -> BlobStoreStats {
        self.stats.clone()
    }
}

impl IterableBlobStore for MemoryBlobStore {
    type IdIter = std::vec::IntoIter<RecordId>;

    fn iter_ids(&self) -> Self::IdIter {
        let mut ids: Vec<RecordId> = self.data.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
    }
}

impl BatchBlobStore for MemoryBlobStore {
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
            results.push(self.data.get(&id).cloned());
        }
        Ok(results)
    }

    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let mut removed_count = 0;
        for id in ids {
            if let Some(data) = self.data.remove(&id) {
                self.stats.record_remove(data.len());
                removed_count += 1;
            }
        }
        Ok(removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::traits::IterableBlobStore;

    #[test]
    fn test_memory_blob_store_basic_operations() {
        let mut store = MemoryBlobStore::new();
        
        // Test empty store
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        
        // Test put operation
        let data1 = b"hello world";
        let id1 = store.put(data1).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert!(store.contains(id1));
        
        // Test get operation
        let retrieved = store.get(id1).unwrap();
        assert_eq!(data1, &retrieved[..]);
        
        // Test size operation
        let size = store.size(id1).unwrap();
        assert_eq!(size, Some(data1.len()));
        
        // Test put another blob
        let data2 = b"goodbye world";
        let id2 = store.put(data2).unwrap();
        assert_eq!(store.len(), 2);
        assert_ne!(id1, id2);
        
        // Test remove operation
        store.remove(id1).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.contains(id1));
        assert!(store.contains(id2));
        
        // Test get after remove
        assert!(store.get(id1).is_err());
        let retrieved2 = store.get(id2).unwrap();
        assert_eq!(data2, &retrieved2[..]);
    }

    #[test]
    fn test_memory_blob_store_errors() {
        let mut store = MemoryBlobStore::new();
        
        // Test get non-existent blob
        let result = store.get(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
        
        // Test remove non-existent blob
        let result = store.remove(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
        
        // Test size of non-existent blob
        let size = store.size(999).unwrap();
        assert_eq!(size, None);
    }

    #[test]
    fn test_memory_blob_store_iteration() {
        let mut store = MemoryBlobStore::new();
        
        // Add some blobs
        let data1 = b"blob1";
        let data2 = b"blob2";
        let data3 = b"blob3";
        
        let id1 = store.put(data1).unwrap();
        let id2 = store.put(data2).unwrap();
        let id3 = store.put(data3).unwrap();
        
        // Test ID iteration
        let ids: Vec<RecordId> = store.iter_ids().collect();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
        assert!(ids.contains(&id3));
        
        // Test blob iteration
        let blobs: Result<Vec<(RecordId, Vec<u8>)>> = store.iter_blobs().collect();
        let blobs = blobs.unwrap();
        assert_eq!(blobs.len(), 3);
        
        // Verify blob contents
        for (id, data) in blobs {
            match id {
                _ if id == id1 => assert_eq!(&data, data1),
                _ if id == id2 => assert_eq!(&data, data2),
                _ if id == id3 => assert_eq!(&data, data3),
                _ => panic!("Unexpected blob ID: {}", id),
            }
        }
    }

    #[test]
    fn test_memory_blob_store_batch_operations() {
        let mut store = MemoryBlobStore::new();
        
        // Test batch put
        let blobs = vec![
            b"blob1".to_vec(),
            b"blob2".to_vec(),
            b"blob3".to_vec(),
        ];
        let ids = store.put_batch(blobs.clone()).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.len(), 3);
        
        // Test batch get
        let retrieved = store.get_batch(ids.clone()).unwrap();
        assert_eq!(retrieved.len(), 3);
        for (i, blob_opt) in retrieved.iter().enumerate() {
            assert!(blob_opt.is_some());
            assert_eq!(blob_opt.as_ref().unwrap(), &blobs[i]);
        }
        
        // Test batch get with some missing IDs
        let mut test_ids = ids.clone();
        test_ids.push(999); // Non-existent ID
        let retrieved = store.get_batch(test_ids).unwrap();
        assert_eq!(retrieved.len(), 4);
        assert!(retrieved[3].is_none());
        
        // Test batch remove
        let removed_count = store.remove_batch(ids).unwrap();
        assert_eq!(removed_count, 3);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_memory_blob_store_capacity_management() {
        let mut store = MemoryBlobStore::with_capacity(10);
        assert!(store.capacity() >= 10);
        
        // Test reserve
        store.reserve(100);
        assert!(store.capacity() >= 100);
        
        // Add some data and test shrink
        for i in 0..5 {
            store.put(format!("blob{}", i).as_bytes()).unwrap();
        }
        
        store.shrink_to_fit();
        assert!(store.capacity() >= store.len());
        
        // Test clear
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_memory_blob_store_from_data() {
        let mut data = HashMap::new();
        data.insert(5, b"blob1".to_vec());
        data.insert(10, b"blob2".to_vec());
        data.insert(15, b"blob3".to_vec());
        
        let store = MemoryBlobStore::from_data(data.clone());
        assert_eq!(store.len(), 3);
        
        // Test that next ID is generated correctly
        let next_id = store.next_id.load(Ordering::Relaxed);
        assert_eq!(next_id, 16);
        
        // Test that all data is accessible
        for (id, expected_data) in data {
            let retrieved = store.get(id).unwrap();
            assert_eq!(retrieved, expected_data);
        }
    }

    #[test]
    fn test_memory_blob_store_stats() {
        let mut store = MemoryBlobStore::new();
        let initial_stats = store.stats();
        assert_eq!(initial_stats.blob_count, 0);
        assert_eq!(initial_stats.total_size, 0);
        
        // Add some blobs
        store.put(b"blob1").unwrap();
        store.put(b"blob22").unwrap();
        
        let stats = store.stats();
        assert_eq!(stats.blob_count, 2);
        assert_eq!(stats.total_size, 11); // 5 + 6 bytes
        assert_eq!(stats.put_count, 2);
    }

    #[test]
    fn test_record_id_generation() {
        let store = MemoryBlobStore::new();
        
        // Test sequential ID generation
        let id1 = store.next_record_id();
        let id2 = store.next_record_id();
        let id3 = store.next_record_id();
        
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
        
        // Test that IDs are unique even in concurrent access
        use std::sync::Arc;
        use std::thread;
        
        let store = Arc::new(MemoryBlobStore::new());
        let mut handles = vec![];
        
        for _ in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                (0..100).map(|_| store_clone.next_record_id()).collect::<Vec<_>>()
            });
            handles.push(handle);
        }
        
        let mut all_ids = Vec::new();
        for handle in handles {
            all_ids.extend(handle.join().unwrap());
        }
        
        // Check that all IDs are unique
        all_ids.sort_unstable();
        for window in all_ids.windows(2) {
            assert_ne!(window[0], window[1], "Found duplicate ID: {}", window[0]);
        }
    }
}