//! Plain file-based blob store implementation
//!
//! This module provides a simple file-based blob store that stores each blob
//! as a separate file on disk. It's suitable for scenarios where persistence
//! is required but advanced features like compression are not needed.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use crate::RecordId;
use crate::blob_store::traits::{BatchBlobStore, BlobStore, BlobStoreStats, IterableBlobStore};
use crate::error::{Result, ZiporaError};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Plain file-based blob store implementation
///
/// This implementation stores each blob as a separate file in a directory.
/// The file names are the record IDs, and the file contents are the blob data.
///
/// # Examples
///
/// ```rust
/// use zipora::blob_store::{BlobStore, PlainBlobStore};
/// use tempfile::tempdir;
///
/// let temp_dir = tempdir().unwrap();
/// let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();
/// let data = b"hello world";
/// let id = store.put(data).unwrap();
/// let retrieved = store.get(id).unwrap();
/// assert_eq!(data, &retrieved[..]);
/// ```
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlainBlobStore {
    /// Base directory for storing blob files
    base_dir: PathBuf,
    /// Next available record ID
    next_id: AtomicU32,
    /// Usage statistics
    stats: BlobStoreStats,
}

impl PlainBlobStore {
    /// Create a new plain blob store in the specified directory
    ///
    /// # Arguments
    /// * `base_dir` - Directory where blob files will be stored
    ///
    /// # Returns
    /// * `Ok(PlainBlobStore)` - Successfully created store
    /// * `Err(ZiporaError)` - If directory creation or initialization fails
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir).map_err(|e| {
                ZiporaError::io_error(format!("Failed to create directory {:?}: {}", base_dir, e))
            })?;
        }

        // Scan existing files to determine next ID and stats
        let (next_id, stats) = Self::scan_directory(&base_dir)?;

        Ok(Self {
            base_dir,
            next_id: AtomicU32::new(next_id),
            stats,
        })
    }

    /// Create a new plain blob store and ensure the directory is empty
    pub fn create_new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Remove directory if it exists and recreate it
        if base_dir.exists() {
            fs::remove_dir_all(&base_dir).map_err(|e| {
                ZiporaError::io_error(format!(
                    "Failed to remove existing directory {:?}: {}",
                    base_dir, e
                ))
            })?;
        }

        fs::create_dir_all(&base_dir).map_err(|e| {
            ZiporaError::io_error(format!("Failed to create directory {:?}: {}", base_dir, e))
        })?;

        Ok(Self {
            base_dir,
            next_id: AtomicU32::new(1),
            stats: BlobStoreStats::new(),
        })
    }

    /// Get the base directory path
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    /// Scan directory to find existing blobs and calculate next ID
    fn scan_directory(base_dir: &Path) -> Result<(u32, BlobStoreStats)> {
        let mut max_id = 0u32;
        let mut stats = BlobStoreStats::new();

        if !base_dir.exists() {
            return Ok((1, stats));
        }

        let entries = fs::read_dir(base_dir).map_err(|e| {
            ZiporaError::io_error(format!("Failed to read directory {:?}: {}", base_dir, e))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                ZiporaError::io_error(format!("Failed to read directory entry: {}", e))
            })?;

            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            // Try to parse filename as record ID
            if let Ok(id) = filename_str.parse::<u32>() {
                max_id = max_id.max(id);

                // Get file size for stats
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        stats.record_put(metadata.len() as usize);
                    }
                }
            }
        }

        Ok((max_id + 1, stats))
    }

    /// Generate the next record ID
    fn next_record_id(&self) -> RecordId {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the file path for a record ID
    fn file_path(&self, id: RecordId) -> PathBuf {
        self.base_dir.join(format!("{}", id))
    }

    /// Check if a blob file exists and is readable
    fn file_exists(&self, id: RecordId) -> bool {
        let path = self.file_path(id);
        path.exists() && path.is_file()
    }

    /// Get all existing blob IDs by scanning the directory
    fn scan_blob_ids(&self) -> Result<Vec<RecordId>> {
        let mut ids = Vec::new();

        let entries = fs::read_dir(&self.base_dir).map_err(|e| {
            ZiporaError::io_error(format!(
                "Failed to read directory {:?}: {}",
                self.base_dir, e
            ))
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                ZiporaError::io_error(format!("Failed to read directory entry: {}", e))
            })?;

            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            // Try to parse filename as record ID
            if let Ok(id) = filename_str.parse::<u32>() {
                if entry.path().is_file() {
                    ids.push(id);
                }
            }
        }

        ids.sort_unstable();
        Ok(ids)
    }
}

impl BlobStore for PlainBlobStore {
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let path = self.file_path(id);

        let mut file = File::open(&path).map_err(|e| {
            ZiporaError::not_found(format!("Blob file {:?} not found: {}", path, e))
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data).map_err(|e| {
            ZiporaError::io_error(format!("Failed to read blob file {:?}: {}", path, e))
        })?;

        Ok(data)
    }

    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        let id = self.next_record_id();
        let path = self.file_path(id);

        let mut file = File::create(&path).map_err(|e| {
            ZiporaError::io_error(format!("Failed to create blob file {:?}: {}", path, e))
        })?;

        file.write_all(data).map_err(|e| {
            ZiporaError::io_error(format!("Failed to write blob file {:?}: {}", path, e))
        })?;

        file.sync_all().map_err(|e| {
            ZiporaError::io_error(format!("Failed to sync blob file {:?}: {}", path, e))
        })?;

        self.stats.record_put(data.len());
        Ok(id)
    }

    fn remove(&mut self, id: RecordId) -> Result<()> {
        let path = self.file_path(id);

        // Get file size before removal for stats
        let size = fs::metadata(&path).map(|m| m.len() as usize).unwrap_or(0);

        fs::remove_file(&path).map_err(|e| {
            ZiporaError::not_found(format!("Failed to remove blob file {:?}: {}", path, e))
        })?;

        self.stats.record_remove(size);
        Ok(())
    }

    fn contains(&self, id: RecordId) -> bool {
        self.file_exists(id)
    }

    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        let path = self.file_path(id);

        match fs::metadata(&path) {
            Ok(metadata) => Ok(Some(metadata.len() as usize)),
            Err(_) => Ok(None),
        }
    }

    fn len(&self) -> usize {
        self.scan_blob_ids().map(|ids| ids.len()).unwrap_or(0)
    }

    fn flush(&mut self) -> Result<()> {
        // For file-based storage, data is already flushed to disk
        // during put operations, so this is a no-op
        Ok(())
    }

    fn stats(&self) -> BlobStoreStats {
        self.stats.clone()
    }
}

impl IterableBlobStore for PlainBlobStore {
    type IdIter = std::vec::IntoIter<RecordId>;

    fn iter_ids(&self) -> Self::IdIter {
        self.scan_blob_ids().unwrap_or_default().into_iter()
    }
}

impl BatchBlobStore for PlainBlobStore {
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
            match self.get(id) {
                Ok(data) => results.push(Some(data)),
                Err(_) => results.push(None),
            }
        }
        Ok(results)
    }

    fn remove_batch<I>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = RecordId>,
    {
        let mut removed_count = 0;
        for id in ids {
            if self.remove(id).is_ok() {
                removed_count += 1;
            }
        }
        Ok(removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_plain_blob_store_basic_operations() {
        let temp_dir = tempdir().unwrap();
        let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();

        // Test empty store
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        // Test put operation
        let data1 = b"hello world";
        let id1 = store.put(data1).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert!(store.contains(id1));

        // Verify file was created
        let file_path = store.file_path(id1);
        assert!(file_path.exists());

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

        // Verify file was removed
        let file_path = store.file_path(id1);
        assert!(!file_path.exists());

        // Test get after remove
        assert!(store.get(id1).is_err());
        let retrieved2 = store.get(id2).unwrap();
        assert_eq!(data2, &retrieved2[..]);
    }

    #[test]
    fn test_plain_blob_store_persistence() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        let data1 = b"persistent data";
        let id1;

        // Create store and add data
        {
            let mut store = PlainBlobStore::new(&dir_path).unwrap();
            id1 = store.put(data1).unwrap();
            assert_eq!(store.len(), 1);
        }

        // Create new store instance and verify data persists
        {
            let store = PlainBlobStore::new(&dir_path).unwrap();
            assert_eq!(store.len(), 1);
            assert!(store.contains(id1));

            let retrieved = store.get(id1).unwrap();
            assert_eq!(data1, &retrieved[..]);
        }
    }

    #[test]
    fn test_plain_blob_store_create_new() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create initial store with data
        {
            let mut store = PlainBlobStore::new(&dir_path).unwrap();
            store.put(b"old data").unwrap();
            assert_eq!(store.len(), 1);
        }

        // Create new store that should clear existing data
        {
            let store = PlainBlobStore::create_new(&dir_path).unwrap();
            assert_eq!(store.len(), 0);
            assert!(store.is_empty());
        }
    }

    #[test]
    fn test_plain_blob_store_errors() {
        let temp_dir = tempdir().unwrap();
        let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();

        // Test get non-existent blob
        let result = store.get(999);
        assert!(result.is_err());

        // Test remove non-existent blob
        let result = store.remove(999);
        assert!(result.is_err());

        // Test size of non-existent blob
        let size = store.size(999).unwrap();
        assert_eq!(size, None);
    }

    #[test]
    fn test_plain_blob_store_iteration() {
        let temp_dir = tempdir().unwrap();
        let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();

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
    }

    #[test]
    fn test_plain_blob_store_batch_operations() {
        let temp_dir = tempdir().unwrap();
        let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();

        // Test batch put
        let blobs = vec![b"blob1".to_vec(), b"blob2".to_vec(), b"blob3".to_vec()];
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

        // Test batch remove
        let removed_count = store.remove_batch(ids).unwrap();
        assert_eq!(removed_count, 3);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_plain_blob_store_large_data() {
        let temp_dir = tempdir().unwrap();
        let mut store = PlainBlobStore::new(temp_dir.path()).unwrap();

        // Test with large blob (1MB)
        let large_data: Vec<u8> = (0..1024 * 1024).map(|i| (i % 256) as u8).collect();
        let id = store.put(&large_data).unwrap();

        let retrieved = store.get(id).unwrap();
        assert_eq!(large_data, retrieved);

        let size = store.size(id).unwrap();
        assert_eq!(size, Some(large_data.len()));
    }

    #[test]
    fn test_plain_blob_store_concurrent_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;

        let temp_dir = tempdir().unwrap();
        let store = Arc::new(Mutex::new(PlainBlobStore::new(temp_dir.path()).unwrap()));
        let mut handles = vec![];

        // Spawn multiple threads to add blobs concurrently
        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            let handle = thread::spawn(move || {
                let data = format!("blob{}", i);
                let mut store = store_clone.lock().unwrap();
                store.put(data.as_bytes()).unwrap()
            });
            handles.push(handle);
        }

        // Collect all IDs
        let mut ids = Vec::new();
        for handle in handles {
            ids.push(handle.join().unwrap());
        }

        // Verify all blobs can be retrieved
        let store = store.lock().unwrap();
        assert_eq!(store.len(), 10);

        for (i, id) in ids.iter().enumerate() {
            let data = store.get(*id).unwrap();
            let expected = format!("blob{}", i);
            assert_eq!(String::from_utf8(data).unwrap(), expected);
        }
    }

    #[test]
    fn test_scan_directory_with_existing_files() {
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path();

        // Create some files manually
        std::fs::write(dir_path.join("5"), b"data1").unwrap();
        std::fs::write(dir_path.join("10"), b"data2").unwrap();
        std::fs::write(dir_path.join("invalid_name"), b"data3").unwrap(); // Should be ignored

        let store = PlainBlobStore::new(dir_path).unwrap();

        // Next ID should be 11 (max existing ID + 1)
        let next_id = store.next_id.load(Ordering::Relaxed);
        assert_eq!(next_id, 11);

        // Should find 2 valid blobs
        assert_eq!(store.len(), 2);
        assert!(store.contains(5));
        assert!(store.contains(10));
        assert!(!store.contains(999));
    }
}
