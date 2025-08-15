//! Cache-aware blob store implementation

use super::BlobStore;
use crate::RecordId;
use crate::cache::{LruPageCache, CacheBuffer, PageCacheConfig, FileId};
use crate::error::Result;
use std::sync::Arc;
use std::io::{Read, Seek, SeekFrom};

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
}

impl<T: BlobStore> CachedBlobStore<T> {
    /// Create new cached blob store
    pub fn new(inner: T, cache_config: PageCacheConfig) -> Result<Self> {
        let cache = Arc::new(LruPageCache::new(cache_config)?);
        
        // Register with cache (use dummy file descriptor for now)
        let file_id = cache.register_file(-1)?;
        
        Ok(Self {
            inner,
            cache,
            file_id,
            cache_enabled: true,
        })
    }
    
    /// Create with existing cache
    pub fn with_cache(inner: T, cache: Arc<LruPageCache>) -> Result<Self> {
        let file_id = cache.register_file(-1)?;
        
        Ok(Self {
            inner,
            cache,
            file_id,
            cache_enabled: true,
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
            let mut data = vec![0u8; length];
            // TODO: This would need actual file I/O integration
            Ok(CacheBuffer::from_data(data))
        }
    }
}

impl<T: BlobStore> BlobStore for CachedBlobStore<T> {
    fn size(&self, id: RecordId) -> Result<Option<usize>> {
        self.inner.size(id)
    }
    fn put(&mut self, data: &[u8]) -> Result<RecordId> {
        // Write-through: write to underlying store
        let id = self.inner.put(data)?;
        
        // TODO: Potentially cache the written data
        // For now, just invalidate any conflicting cache entries
        
        Ok(id)
    }
    
    fn get(&self, id: RecordId) -> Result<Vec<u8>> {
        if !self.cache_enabled {
            return self.inner.get(id);
        }
        
        // Try to calculate offset for cache lookup
        // This would need integration with the specific blob store format
        let estimated_offset = id as u64 * 1024; // Placeholder
        let estimated_size = 1024; // Placeholder
        
        match self.read_cached(estimated_offset, estimated_size) {
            Ok(buffer) => {
                // If we got data from cache, use it
                if buffer.has_data() {
                    Ok(buffer.data().to_vec())
                } else {
                    // Cache miss, fall back to underlying store
                    self.inner.get(id)
                }
            }
            Err(_) => {
                // Cache error, fall back to underlying store
                self.inner.get(id)
            }
        }
    }
    
    fn remove(&mut self, id: RecordId) -> Result<()> {
        // Remove from underlying store
        self.inner.remove(id)?;
        
        // TODO: Invalidate cache entries for this blob
        
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
}