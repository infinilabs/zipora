//! Asynchronous blob storage implementations

use crate::blob_store::{BlobStore, BlobStoreStats};
use crate::RecordId;
use crate::error::{ToplingError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom};
use tokio::sync::{Mutex, RwLock};

/// Asynchronous blob store trait
#[async_trait::async_trait]
pub trait AsyncBlobStore: Send + Sync {
    /// Store a blob and return its record ID
    async fn put(&self, data: &[u8]) -> Result<RecordId>;
    
    /// Retrieve a blob by its record ID
    async fn get(&self, id: RecordId) -> Result<Vec<u8>>;
    
    /// Remove a blob by its record ID
    async fn remove(&self, id: RecordId) -> Result<()>;
    
    /// Check if a blob exists
    async fn contains(&self, id: RecordId) -> bool;
    
    /// Get the size of a blob without loading it
    async fn size(&self, id: RecordId) -> Result<Option<usize>>;
    
    /// Get the number of blobs in the store
    async fn len(&self) -> usize;
    
    /// Check if the store is empty
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
    
    /// Flush any pending writes
    async fn flush(&self) -> Result<()>;
    
    /// Get store statistics
    async fn stats(&self) -> BlobStoreStats;
    
    /// Batch operations for better performance
    async fn put_batch(&self, data: Vec<&[u8]>) -> Result<Vec<RecordId>> {
        let mut ids = Vec::with_capacity(data.len());
        for item in data {
            ids.push(self.put(item).await?);
        }
        Ok(ids)
    }
    
    async fn get_batch(&self, ids: Vec<RecordId>) -> Result<Vec<Vec<u8>>> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.get(id).await?);
        }
        Ok(results)
    }
}

/// Asynchronous in-memory blob store
pub struct AsyncMemoryBlobStore {
    data: Arc<RwLock<HashMap<RecordId, Vec<u8>>>>,
    next_id: AtomicU64,
    stats: Arc<RwLock<BlobStoreStats>>,
}

impl AsyncMemoryBlobStore {
    /// Create a new async memory blob store
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            stats: Arc::new(RwLock::new(BlobStoreStats::default())),
        }
    }
    
    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
            next_id: AtomicU64::new(1),
            stats: Arc::new(RwLock::new(BlobStoreStats::default())),
        }
    }
}

impl Default for AsyncMemoryBlobStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AsyncBlobStore for AsyncMemoryBlobStore {
    async fn put(&self, data: &[u8]) -> Result<RecordId> {
        let start_time = Instant::now();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as RecordId;
        
        {
            let mut store = self.data.write().await;
            store.insert(id, data.to_vec());
        }
        
        let mut stats = self.stats.write().await;
        stats.put_count += 1;
        
        
        
        Ok(id)
    }
    
    async fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        
        let result = {
            let store = self.data.read().await;
            store.get(&id).cloned()
        };
        
        let mut stats = self.stats.write().await;
        stats.get_count += 1;
        
        
        match result {
            Some(data) => {
                
                Ok(data)
            }
            None => {
                
                Err(ToplingError::invalid_data(&format!("record not found: {}", id)))
            }
        }
    }
    
    async fn remove(&self, id: RecordId) -> Result<()> {
        let start_time = Instant::now();
        
        let removed = {
            let mut store = self.data.write().await;
            store.remove(&id)
        };
        
        let mut stats = self.stats.write().await;
        
        
        
        if removed.is_some() {
            Ok(())
        } else {
            Err(ToplingError::invalid_data(&format!("record not found: {}", id)))
        }
    }
    
    async fn contains(&self, id: RecordId) -> bool {
        let store = self.data.read().await;
        store.contains_key(&id)
    }
    
    async fn size(&self, id: RecordId) -> Result<Option<usize>> {
        let store = self.data.read().await;
        Ok(store.get(&id).map(|data| data.len()))
    }
    
    async fn len(&self) -> usize {
        let store = self.data.read().await;
        store.len()
    }
    
    async fn flush(&self) -> Result<()> {
        // Memory store doesn't need flushing
        Ok(())
    }
    
    async fn stats(&self) -> BlobStoreStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    async fn put_batch(&self, data: Vec<&[u8]>) -> Result<Vec<RecordId>> {
        let start_time = Instant::now();
        let mut ids = Vec::with_capacity(data.len());
        let mut total_bytes = 0;
        
        {
            let mut store = self.data.write().await;
            for item in data {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed) as RecordId;
                store.insert(id, item.to_vec());
                ids.push(id);
                total_bytes += item.len();
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.put_count += ids.len() as u64;
        
        
        
        Ok(ids)
    }
    
    async fn get_batch(&self, ids: Vec<RecordId>) -> Result<Vec<Vec<u8>>> {
        let start_time = Instant::now();
        let mut results = Vec::with_capacity(ids.len());
        let mut total_bytes = 0;
        let mut misses = 0;
        
        {
            let store = self.data.read().await;
            for id in ids {
                match store.get(&id) {
                    Some(data) => {
                        total_bytes += data.len();
                        results.push(data.clone());
                    }
                    None => {
                        misses += 1;
                        return Err(ToplingError::invalid_data(&format!("record not found: {}", id)));
                    }
                }
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.get_count += results.len() as u64;
        
        
        
        
        Ok(results)
    }
}

/// Asynchronous file-based blob store
pub struct AsyncFileStore {
    base_path: PathBuf,
    file_handles: Arc<Mutex<HashMap<RecordId, File>>>,
    metadata: Arc<RwLock<HashMap<RecordId, FileMetadata>>>,
    next_id: AtomicU64,
    stats: Arc<RwLock<BlobStoreStats>>,
}

#[derive(Clone)]
struct FileMetadata {
    size: usize,
    file_path: PathBuf,
    created_at: Instant,
}

impl AsyncFileStore {
    /// Create a new async file store
    pub async fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        
        // Create directory if it doesn't exist
        tokio::fs::create_dir_all(&base_path).await
            .map_err(|e| ToplingError::io_error(&format!("failed to create directory: {}", e)))?;
        
        Ok(Self {
            base_path,
            file_handles: Arc::new(Mutex::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            stats: Arc::new(RwLock::new(BlobStoreStats::default())),
        })
    }
    
    /// Get the file path for a record ID
    fn get_file_path(&self, id: RecordId) -> PathBuf {
        self.base_path.join(format!("blob_{:016x}.dat", id))
    }
}

#[async_trait::async_trait]
impl AsyncBlobStore for AsyncFileStore {
    async fn put(&self, data: &[u8]) -> Result<RecordId> {
        let start_time = Instant::now();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as RecordId;
        let file_path = self.get_file_path(id);
        
        // Write data to file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&file_path)
            .await
            .map_err(|e| ToplingError::io_error(&format!("failed to create file: {}", e)))?;
        
        file.write_all(data).await
            .map_err(|e| ToplingError::io_error(&format!("failed to write data: {}", e)))?;
        
        file.flush().await
            .map_err(|e| ToplingError::io_error(&format!("failed to flush file: {}", e)))?;
        
        // Update metadata
        {
            let mut metadata = self.metadata.write().await;
            metadata.insert(id, FileMetadata {
                size: data.len(),
                file_path: file_path.clone(),
                created_at: Instant::now(),
            });
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.put_count += 1;
            
            
        }
        
        Ok(id)
    }
    
    async fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let start_time = Instant::now();
        let file_path = self.get_file_path(id);
        
        // Check if file exists in metadata
        let metadata = {
            let metadata = self.metadata.read().await;
            metadata.get(&id).cloned()
        };
        
        let size = match metadata {
            Some(meta) => meta.size,
            None => {
                let mut stats = self.stats.write().await;
                stats.get_count += 1;
                
                
                return Err(ToplingError::invalid_data(&format!("record not found: {}", id)));
            }
        };
        
        // Read data from file
        let mut file = File::open(&file_path).await
            .map_err(|e| ToplingError::io_error(&format!("failed to open file: {}", e)))?;
        
        let mut data = vec![0u8; size];
        file.read_exact(&mut data).await
            .map_err(|e| ToplingError::io_error(&format!("failed to read data: {}", e)))?;
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.get_count += 1;
            
            
        }
        
        Ok(data)
    }
    
    async fn remove(&self, id: RecordId) -> Result<()> {
        let start_time = Instant::now();
        let file_path = self.get_file_path(id);
        
        // Remove from metadata first
        let existed = {
            let mut metadata = self.metadata.write().await;
            metadata.remove(&id).is_some()
        };
        
        if !existed {
            return Err(ToplingError::invalid_data(&format!("record not found: {}", id)));
        }
        
        // Remove file
        tokio::fs::remove_file(&file_path).await
            .map_err(|e| ToplingError::io_error(&format!("failed to remove file: {}", e)))?;
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            
            
        }
        
        Ok(())
    }
    
    async fn contains(&self, id: RecordId) -> bool {
        let metadata = self.metadata.read().await;
        metadata.contains_key(&id)
    }
    
    async fn size(&self, id: RecordId) -> Result<Option<usize>> {
        let metadata = self.metadata.read().await;
        Ok(metadata.get(&id).map(|meta| meta.size))
    }
    
    async fn len(&self) -> usize {
        let metadata = self.metadata.read().await;
        metadata.len()
    }
    
    async fn flush(&self) -> Result<()> {
        // Flush all open file handles
        let mut handles = self.file_handles.lock().await;
        for file in handles.values_mut() {
            file.flush().await
                .map_err(|e| ToplingError::io_error(&format!("failed to flush file: {}", e)))?;
        }
        Ok(())
    }
    
    async fn stats(&self) -> BlobStoreStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
}

/// Wrapper that adds compression to any async blob store
pub struct AsyncCompressedBlobStore<S: AsyncBlobStore> {
    inner: S,
    compression_level: i32,
}

impl<S: AsyncBlobStore> AsyncCompressedBlobStore<S> {
    /// Create a new compressed blob store wrapper
    pub fn new(inner: S, compression_level: i32) -> Self {
        Self {
            inner,
            compression_level,
        }
    }
    
    /// Compress data using zstd
    async fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let level = self.compression_level;
        let data = data.to_vec();
        
        tokio::task::spawn_blocking(move || {
            zstd::bulk::compress(&data, level)
                .map_err(|e| ToplingError::configuration(&format!("compression failed: {}", e)))
        }).await
        .map_err(|e| ToplingError::configuration(&format!("compression task failed: {}", e)))?
    }
    
    /// Decompress data using zstd
    async fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let data = data.to_vec();
        
        tokio::task::spawn_blocking(move || {
            zstd::bulk::decompress(&data, 10 * 1024 * 1024) // 10MB limit
                .map_err(|e| ToplingError::configuration(&format!("decompression failed: {}", e)))
        }).await
        .map_err(|e| ToplingError::configuration(&format!("decompression task failed: {}", e)))?
    }
}

#[async_trait::async_trait]
impl<S: AsyncBlobStore> AsyncBlobStore for AsyncCompressedBlobStore<S> {
    async fn put(&self, data: &[u8]) -> Result<RecordId> {
        let compressed = self.compress(data).await?;
        self.inner.put(&compressed).await
    }
    
    async fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        let compressed = self.inner.get(id).await?;
        self.decompress(&compressed).await
    }
    
    async fn remove(&self, id: RecordId) -> Result<()> {
        self.inner.remove(id).await
    }
    
    async fn contains(&self, id: RecordId) -> bool {
        self.inner.contains(id).await
    }
    
    async fn size(&self, id: RecordId) -> Result<Option<usize>> {
        // This returns the compressed size, not the original size
        self.inner.size(id).await
    }
    
    async fn len(&self) -> usize {
        self.inner.len().await
    }
    
    async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }
    
    async fn stats(&self) -> BlobStoreStats {
        self.inner.stats().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio;
    
    #[tokio::test]
    async fn test_async_memory_blob_store() {
        let store = AsyncMemoryBlobStore::new();
        
        // Test put and get
        let data = b"hello world";
        let id = store.put(data).await.unwrap();
        let retrieved = store.get(id).await.unwrap();
        
        assert_eq!(data, retrieved.as_slice());
        assert_eq!(store.len().await, 1);
        assert!(store.contains(id).await);
        
        // Test size
        let size = store.size(id).await.unwrap();
        assert_eq!(size, Some(data.len()));
        
        // Test remove
        store.remove(id).await.unwrap();
        assert_eq!(store.len().await, 0);
        assert!(!store.contains(id).await);
    }
    
    #[tokio::test]
    async fn test_async_file_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = AsyncFileStore::new(temp_dir.path()).await.unwrap();
        
        // Test put and get
        let data = b"hello file world";
        let id = store.put(data).await.unwrap();
        let retrieved = store.get(id).await.unwrap();
        
        assert_eq!(data, retrieved.as_slice());
        assert_eq!(store.len().await, 1);
        assert!(store.contains(id).await);
        
        // Test remove
        store.remove(id).await.unwrap();
        assert_eq!(store.len().await, 0);
        assert!(!store.contains(id).await);
    }
    
    #[tokio::test]
    async fn test_batch_operations() {
        let store = AsyncMemoryBlobStore::new();
        
        // Test batch put
        let data = vec![b"one".as_slice(), b"two".as_slice(), b"three".as_slice()];
        let ids = store.put_batch(data).await.unwrap();
        
        assert_eq!(ids.len(), 3);
        assert_eq!(store.len().await, 3);
        
        // Test batch get
        let retrieved = store.get_batch(ids).await.unwrap();
        assert_eq!(retrieved.len(), 3);
        assert_eq!(retrieved[0], b"one");
        assert_eq!(retrieved[1], b"two");
        assert_eq!(retrieved[2], b"three");
    }
    
    #[tokio::test]
    async fn test_compressed_blob_store() {
        let inner = AsyncMemoryBlobStore::new();
        let store = AsyncCompressedBlobStore::new(inner, 3);
        
        // Test with compressible data
        let data = b"hello world hello world hello world hello world";
        let id = store.put(data).await.unwrap();
        let retrieved = store.get(id).await.unwrap();
        
        assert_eq!(data, retrieved.as_slice());
        assert!(store.contains(id).await);
    }
    
    #[tokio::test]
    async fn test_store_statistics() {
        let store = AsyncMemoryBlobStore::new();
        
        // Perform some operations
        let id1 = store.put(b"data1").await.unwrap();
        let id2 = store.put(b"data2").await.unwrap();
        let _data1 = store.get(id1).await.unwrap();
        let _data2 = store.get(id2).await.unwrap();
        
        let stats = store.stats().await;
        assert_eq!(stats.put_count, 2);
        assert_eq!(stats.get_count, 2);
    }
    
    #[tokio::test]
    async fn test_error_handling() {
        let store = AsyncMemoryBlobStore::new();
        
        // Test get non-existent record
        let result = store.get(999 as RecordId).await;
        assert!(result.is_err());
        
        // Test remove non-existent record
        let result = store.remove(999 as RecordId).await;
        assert!(result.is_err());
        
        // Test size of non-existent record
        let size = store.size(999 as RecordId).await.unwrap();
        assert_eq!(size, None);
    }
}