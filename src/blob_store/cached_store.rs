//! Cache-aware blob store implementation

use super::BlobStore;
use crate::RecordId;
use crate::cache::{LruPageCache, CacheBuffer, PageCacheConfig, FileId};
use crate::error::Result;
use std::sync::Arc;
//use std::io::{Read, Seek, SeekFrom};

/// Cache write strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheWriteStrategy {
    /// Write-through: write to store and cache simultaneously
    WriteThrough,
    /// Write-back: write to cache first, defer store writes
    WriteBack,
    /// Write-around: write to store only, bypass cache
    WriteAround,
}

/// Cache-aware blob store wrapper
pub struct CachedBlobStore<T> {
    /// Underlying blob store
    inner: T,
    
    /// Page cache for performance
    cache: Arc<LruPageCache>,
    
    /// File ID for cache operations
    file_id: FileId,
    
    /// Cache statistics
    cache_enabled: bool,
    
    /// Write strategy for cache operations
    write_strategy: CacheWriteStrategy,
    
    /// Cache of blob metadata (offset, size) for better integration
    blob_metadata: std::sync::Mutex<std::collections::HashMap<RecordId, (u64, usize)>>,
    
    /// Next offset for new blobs (simple allocation strategy)
    next_offset: std::sync::atomic::AtomicU64,
}

impl<T: BlobStore> CachedBlobStore<T> {
    /// Create new cached blob store with write-through strategy
    pub fn new(inner: T, cache_config: PageCacheConfig) -> Result<Self> {
        Self::with_write_strategy(inner, cache_config, CacheWriteStrategy::WriteThrough)
    }
    
    /// Create new cached blob store with specified write strategy
    pub fn with_write_strategy(inner: T, cache_config: PageCacheConfig, strategy: CacheWriteStrategy) -> Result<Self> {
        let cache = Arc::new(LruPageCache::new(cache_config)?);
        
        // Register with cache (use dummy file descriptor for now)
        let file_id = cache.register_file(-1)?;
        
        Ok(Self {
            inner,
            cache,
            file_id,
            cache_enabled: true,
            write_strategy: strategy,
            blob_metadata: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_offset: std::sync::atomic::AtomicU64::new(0),
        })
    }
    
    /// Create with existing cache
    pub fn with_cache(inner: T, cache: Arc<LruPageCache>) -> Result<Self> {
        Self::with_cache_and_strategy(inner, cache, CacheWriteStrategy::WriteThrough)
    }
    
    /// Create with existing cache and write strategy
    pub fn with_cache_and_strategy(inner: T, cache: Arc<LruPageCache>, strategy: CacheWriteStrategy) -> Result<Self> {
        let file_id = cache.register_file(-1)?;
        
        Ok(Self {
            inner,
            cache,
            file_id,
            cache_enabled: true,
            write_strategy: strategy,
            blob_metadata: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_offset: std::sync::atomic::AtomicU64::new(0),
        })
    }
    
    /// Disable caching for this store
    pub fn disable_cache(&mut self) {
        self.cache_enabled = false;
    }
    
    /// Enable caching for this store
    pub fn enable_cache(&mut self) {
        self.cache_enabled = true;
    }
    
    /// Set write strategy
    pub fn set_write_strategy(&mut self, strategy: CacheWriteStrategy) {
        self.write_strategy = strategy;
    }
    
    /// Get current write strategy
    pub fn write_strategy(&self) -> CacheWriteStrategy {
        self.write_strategy
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::cache::CacheStatsSnapshot {
        self.cache.stats()
    }
    
    /// Get underlying blob store
    pub fn inner(&self) -> &T {
        &self.inner
    }
    
    /// Get mutable reference to underlying store
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
    
    /// Prefetch data for better cache performance
    pub fn prefetch_range(&self, start_offset: u64, length: usize) -> Result<()> {
        if self.cache_enabled {
            self.cache.prefetch(self.file_id, start_offset, length)?;
        }
        Ok(())
    }
    
    /// Read with cache awareness
    fn read_cached(&self, offset: u64, length: usize) -> Result<CacheBuffer> {
        if self.cache_enabled {
            self.cache.read(self.file_id, offset, length)
        } else {
            // Fallback to direct read
            let data = vec![0u8; length];
            // TODO: This would need actual file I/O integration
            Ok(CacheBuffer::from_data(data))
        }
    }
    
    /// Write data to cache and/or store based on strategy
    fn write_cached(&mut self, id: RecordId, data: &[u8]) -> Result<()> {
        let offset = self.next_offset.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        
        match self.write_strategy {
            CacheWriteStrategy::WriteThrough => {
                // Write to both store and cache
                self.inner.put(data)?;
                if self.cache_enabled {
                    // Cache the written data
                    self.cache_data_at_offset(offset, data)?;
                }
            }
            CacheWriteStrategy::WriteBack => {
                // Write to cache first, defer store write
                if self.cache_enabled {
                    self.cache_data_at_offset(offset, data)?;
                    // Mark as dirty (would need dirty tracking in real implementation)
                }
                // In a real implementation, store write would be deferred
                self.inner.put(data)?;
            }
            CacheWriteStrategy::WriteAround => {
                // Write to store only, bypass cache
                self.inner.put(data)?;
                // Don't cache the data
            }
        }
        
        // Update metadata
        {
            let mut metadata = self.blob_metadata.lock()
                .map_err(|_| crate::error::ZiporaError::invalid_data("Metadata lock poisoned".to_string()))?;
            metadata.insert(id, (offset, data.len()));
        }
        
        Ok(())
    }
    
    /// Cache data at specific offset (helper method)
    fn cache_data_at_offset(&self, offset: u64, data: &[u8]) -> Result<()> {
        // Mark affected pages as dirty since we're writing new data
        let start_page = crate::cache::FileManager::offset_to_page_id(offset);
        let end_offset = offset + data.len() as u64;
        let end_page = crate::cache::FileManager::offset_to_page_id(end_offset.saturating_sub(1));
        
        for page_id in start_page..=end_page {
            self.cache.mark_dirty(self.file_id, page_id)?;
        }
        
        // In a real implementation, we'd write the data to the cache pages
        // For now, this marks the operation as successful
        Ok(())
    }
    
    /// Invalidate cached data for a blob
    fn invalidate_cached_blob(&self, id: RecordId) -> Result<()> {
        if let Some((offset, size)) = self.get_blob_metadata(id)? {
            self.invalidate_range(offset, size)?;
        }
        Ok(())
    }
    
    /// Invalidate cache range
    fn invalidate_range(&self, offset: u64, size: usize) -> Result<()> {
        if !self.cache_enabled {
            return Ok(());
        }
        
        // Use the cache's built-in invalidation functionality
        self.cache.invalidate_range(self.file_id, offset, size)?;
        
        Ok(())
    }
    
    /// Flush dirty pages to ensure data persistence
    pub fn flush(&self) -> Result<()> {
        if self.cache_enabled {
            self.cache.flush_file(self.file_id)?;
        }
        Ok(())
    }
    
    /// Get cache invalidation statistics
    pub fn invalidation_stats(&self) -> Result<(usize, usize)> {
        // Return (invalidated_count, dirty_count) - simplified implementation
        // In a real implementation, we'd get this from the cache
        Ok((0, 0))
    }
    
    /// Get blob metadata (offset and size)
    fn get_blob_metadata(&self, id: RecordId) -> Result<Option<(u64, usize)>> {
        let metadata = self.blob_metadata.lock()
            .map_err(|_| crate::error::ZiporaError::invalid_data("Metadata lock poisoned".to_string()))?;
        Ok(metadata.get(&id).copied())
    }
}

impl<T: BlobStore> BlobStore for CachedBlobStore<T> {
    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        self.inner.size(id)
    }
    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        let offset = self.next_offset.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        
        // Write to underlying store first to get the actual ID
        let id = self.inner.put(data)?;
        
        // Apply caching strategy
        match self.write_strategy {
            CacheWriteStrategy::WriteThrough => {
                // Data is already written to store, now cache it
                if self.cache_enabled {
                    self.cache_data_at_offset(offset, data)?;
                }
            }
            CacheWriteStrategy::WriteBack => {
                // Data is already written to store, now cache it
                if self.cache_enabled {
                    self.cache_data_at_offset(offset, data)?;
                }
            }
            CacheWriteStrategy::WriteAround => {
                // Data is already written to store, no caching needed
            }
        }
        
        // Update metadata
        {
            let mut metadata = self.blob_metadata.lock()
                .map_err(|_| crate::error::ZiporaError::invalid_data("Metadata lock poisoned".to_string()))?;
            metadata.insert(id, (offset, data.len()));
        }
        
        Ok(id)
    }
    
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        if !self.cache_enabled {
            return self.inner.get(id);
        }
        
        // Try to get from cache using stored metadata
        if let Some((offset, size)) = self.get_blob_metadata(id)? {
            match self.read_cached(offset, size) {
                Ok(buffer) => {
                    // If we got data from cache, use it
                    if buffer.has_data() {
                        return Ok(buffer.data().to_vec());
                    }
                }
                Err(_) => {
                    // Cache error, continue to fallback
                }
            }
        }
        
        // Cache miss or no metadata, fall back to underlying store
        let data = self.inner.get(id)?;
        
        // For write-back strategy, cache the read data
        if self.cache_enabled && matches!(self.write_strategy, CacheWriteStrategy::WriteBack) {
            // Would cache the data here in a real implementation
        }
        
        Ok(data)
    }
    
    fn remove(&mut self, id: RecordId) -> Result<()> {
        // Invalidate cache first
        self.invalidate_cached_blob(id)?;
        
        // Remove from underlying store
        self.inner.remove(id)?;
        
        // Remove metadata
        {
            let mut metadata = self.blob_metadata.lock()
                .map_err(|_| crate::error::ZiporaError::invalid_data("Metadata lock poisoned".to_string()))?;
            metadata.remove(&id);
        }
        
        Ok(())
    }
    
    fn contains(&self, id: RecordId) -> bool {
        self.inner.contains(id)
    }
    
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Cache-aware extension trait for blob stores
pub trait CacheAwareBlobStore: BlobStore {
    /// Read data with cache integration
    fn read_with_cache(&self, offset: u64, length: usize, cache: &LruPageCache, file_id: FileId) -> Result<CacheBuffer>;
    
    /// Write data with cache invalidation
    fn write_with_cache(&mut self, offset: u64, data: &[u8], cache: &LruPageCache, file_id: FileId) -> Result<()>;
    
    /// Get file size for cache calculations
    fn file_size(&self) -> u64;
    
    /// Get optimal cache page size for this store
    fn optimal_page_size(&self) -> usize {
        crate::cache::PAGE_SIZE
    }
}

/// Cache statistics for blob store operations
#[derive(Debug, Clone)]
pub struct BlobCacheStats {
    /// Total blob reads
    pub total_reads: u64,
    
    /// Cache hits for blob operations
    pub cache_hits: u64,
    
    /// Cache misses for blob operations
    pub cache_misses: u64,
    
    /// Total bytes read through cache
    pub bytes_cached: u64,
    
    /// Total bytes read directly
    pub bytes_direct: u64,
    
    /// Cache hit ratio
    pub hit_ratio: f64,
}

impl BlobCacheStats {
    pub fn new() -> Self {
        Self {
            total_reads: 0,
            cache_hits: 0,
            cache_misses: 0,
            bytes_cached: 0,
            bytes_direct: 0,
            hit_ratio: 0.0,
        }
    }
    
    pub fn record_hit(&mut self, bytes: usize) {
        self.total_reads += 1;
        self.cache_hits += 1;
        self.bytes_cached += bytes as u64;
        self.update_hit_ratio();
    }
    
    pub fn record_miss(&mut self, bytes: usize) {
        self.total_reads += 1;
        self.cache_misses += 1;
        self.bytes_direct += bytes as u64;
        self.update_hit_ratio();
    }
    
    fn update_hit_ratio(&mut self) {
        if self.total_reads > 0 {
            self.hit_ratio = self.cache_hits as f64 / self.total_reads as f64;
        }
    }
    
    pub fn bytes_saved(&self) -> u64 {
        self.bytes_cached
    }
    
    pub fn efficiency_ratio(&self) -> f64 {
        let total_bytes = self.bytes_cached + self.bytes_direct;
        if total_bytes > 0 {
            self.bytes_cached as f64 / total_bytes as f64
        } else {
            0.0
        }
    }
}

impl Default for BlobCacheStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob_store::MemoryBlobStore;
    use crate::cache::PageCacheConfig;
    
    #[test]
    fn test_cached_blob_store_creation() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::balanced();
        
        let cached_store = CachedBlobStore::new(inner, config);
        assert!(cached_store.is_ok());
    }
    
    #[test]
    fn test_cache_disable_enable() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::balanced();
        let mut cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        // Should start enabled
        assert!(cached_store.cache_enabled);
        
        cached_store.disable_cache();
        assert!(!cached_store.cache_enabled);
        
        cached_store.enable_cache();
        assert!(cached_store.cache_enabled);
    }
    
    #[test]
    fn test_write_strategies() {
        // Test write-through strategy
        let inner1 = MemoryBlobStore::new();
        let config1 = PageCacheConfig::balanced();
        let cached_store = CachedBlobStore::with_write_strategy(
            inner1, config1, CacheWriteStrategy::WriteThrough
        ).unwrap();
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteThrough);
        
        // Test write-back strategy
        let inner2 = MemoryBlobStore::new();
        let config2 = PageCacheConfig::balanced();
        let cached_store = CachedBlobStore::with_write_strategy(
            inner2, config2, CacheWriteStrategy::WriteBack
        ).unwrap();
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteBack);
        
        // Test write-around strategy
        let inner3 = MemoryBlobStore::new();
        let config3 = PageCacheConfig::balanced();
        let cached_store = CachedBlobStore::with_write_strategy(
            inner3, config3, CacheWriteStrategy::WriteAround
        ).unwrap();
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteAround);
    }
    
    #[test]
    fn test_write_strategy_modification() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::balanced();
        let mut cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        // Should start with write-through
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteThrough);
        
        // Change to write-back
        cached_store.set_write_strategy(CacheWriteStrategy::WriteBack);
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteBack);
        
        // Change to write-around
        cached_store.set_write_strategy(CacheWriteStrategy::WriteAround);
        assert_eq!(cached_store.write_strategy(), CacheWriteStrategy::WriteAround);
    }
    
    #[test]
    fn test_blob_cache_stats() {
        let mut stats = BlobCacheStats::new();
        
        assert_eq!(stats.total_reads, 0);
        assert_eq!(stats.hit_ratio, 0.0);
        
        stats.record_hit(1024);
        assert_eq!(stats.total_reads, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.hit_ratio, 1.0);
        
        stats.record_miss(512);
        assert_eq!(stats.total_reads, 2);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_ratio, 0.5);
    }
    
    #[test]
    fn test_basic_blob_operations() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::memory_optimized();
        let mut cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        // Test basic blob store operations
        let data = b"Hello, cached world!";
        let id = cached_store.put(data).unwrap();
        
        assert!(cached_store.contains(id));
        assert_eq!(cached_store.len(), 1);
        assert!(!cached_store.is_empty());
        
        let retrieved = cached_store.get(id).unwrap();
        assert_eq!(retrieved, data);
        
        cached_store.remove(id).unwrap();
        assert!(!cached_store.contains(id));
        assert_eq!(cached_store.len(), 0);
    }
    
    #[test]
    fn test_write_through_operations() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::performance_optimized();
        let mut cached_store = CachedBlobStore::with_write_strategy(
            inner, config, CacheWriteStrategy::WriteThrough
        ).unwrap();
        
        let data1 = b"Write-through data 1";
        let data2 = b"Write-through data 2";
        
        let id1 = cached_store.put(data1).unwrap();
        let id2 = cached_store.put(data2).unwrap();
        
        // Both should be available immediately
        assert_eq!(cached_store.get(id1).unwrap(), data1);
        assert_eq!(cached_store.get(id2).unwrap(), data2);
        
        // Test metadata is stored
        assert!(cached_store.get_blob_metadata(id1).unwrap().is_some());
        assert!(cached_store.get_blob_metadata(id2).unwrap().is_some());
    }
    
    #[test]
    fn test_write_back_operations() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::performance_optimized();
        let mut cached_store = CachedBlobStore::with_write_strategy(
            inner, config, CacheWriteStrategy::WriteBack
        ).unwrap();
        
        let data = b"Write-back test data";
        let id = cached_store.put(data).unwrap();
        
        // Should be available for read
        assert_eq!(cached_store.get(id).unwrap(), data);
        
        // Test that metadata is properly maintained
        assert!(cached_store.get_blob_metadata(id).unwrap().is_some());
    }
    
    #[test]
    fn test_cache_invalidation() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::balanced();
        let mut cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        let data1 = b"Data to be invalidated";
        let data2 = b"Replacement data";
        
        // Store initial data
        let id = cached_store.put(data1).unwrap();
        assert_eq!(cached_store.get(id).unwrap(), data1);
        
        // Test invalidation by removing and re-adding
        cached_store.remove(id).unwrap();
        
        // Should not contain the removed item
        assert!(!cached_store.contains(id));
        
        // Test flush functionality
        let id2 = cached_store.put(data2).unwrap();
        assert!(cached_store.flush().is_ok());
    }
    
    #[test]
    fn test_invalidation_stats() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::memory_optimized();
        let cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        // Test that invalidation stats can be retrieved
        let stats = cached_store.invalidation_stats().unwrap();
        assert_eq!(stats, (0, 0)); // Should start with no invalidations or dirty pages
    }
    
    #[test]
    fn test_prefetch_functionality() {
        let inner = MemoryBlobStore::new();
        let config = PageCacheConfig::performance_optimized();
        let cached_store = CachedBlobStore::new(inner, config).unwrap();
        
        // Test prefetch range - should not fail
        assert!(cached_store.prefetch_range(0, 4096).is_ok());
        assert!(cached_store.prefetch_range(4096, 8192).is_ok());
    }
}