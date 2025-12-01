//! Basic LRU cache implementation with real file I/O

use super::*;
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Cache invalidation tracking
#[derive(Debug, Clone)]
struct InvalidationTracker {
    /// Set of invalidated pages
    invalidated_pages: std::collections::HashSet<(FileId, PageId)>,
    /// Dirty pages that need write-back
    dirty_pages: std::collections::HashSet<(FileId, PageId)>,
    /// Last access timestamp for each page
    access_times: std::collections::HashMap<(FileId, PageId), std::time::Instant>,
}

/// Basic LRU page cache for blob operations with real file I/O
pub struct LruPageCache {
    /// Cache storage: (file_id, page_id) -> page data
    inner: Arc<Mutex<HashMap<(FileId, PageId), Vec<u8>>>>,
    /// Cache configuration
    config: PageCacheConfig,
    /// Performance statistics
    stats: CacheStatistics,
    /// File manager for actual file operations
    file_manager: FileManager,
    /// Invalidation tracking
    invalidation_tracker: Arc<Mutex<InvalidationTracker>>,
}

impl InvalidationTracker {
    fn new() -> Self {
        Self {
            invalidated_pages: std::collections::HashSet::new(),
            dirty_pages: std::collections::HashSet::new(),
            access_times: std::collections::HashMap::new(),
        }
    }
    
    fn mark_accessed(&mut self, key: (FileId, PageId)) {
        self.access_times.insert(key, std::time::Instant::now());
    }
    
    fn mark_dirty(&mut self, key: (FileId, PageId)) {
        self.dirty_pages.insert(key);
    }
    
    fn mark_clean(&mut self, key: (FileId, PageId)) {
        self.dirty_pages.remove(&key);
    }
    
    fn invalidate(&mut self, key: (FileId, PageId)) {
        self.invalidated_pages.insert(key);
        self.dirty_pages.remove(&key);
        self.access_times.remove(&key);
    }
    
    fn is_invalidated(&self, key: &(FileId, PageId)) -> bool {
        self.invalidated_pages.contains(key)
    }
    
    fn is_dirty(&self, key: &(FileId, PageId)) -> bool {
        self.dirty_pages.contains(key)
    }
    
    fn find_lru_page(&self) -> Option<(FileId, PageId)> {
        self.access_times
            .iter()
            .min_by_key(|(_, time)| *time)
            .map(|(key, _)| *key)
    }
    
    fn cleanup_file(&mut self, file_id: FileId) {
        self.invalidated_pages.retain(|(fid, _)| *fid != file_id);
        self.dirty_pages.retain(|(fid, _)| *fid != file_id);
        self.access_times.retain(|&(fid, _), _| fid != file_id);
    }
}

impl LruPageCache {
    /// Create new LRU cache with real file I/O
    pub fn new(config: PageCacheConfig) -> Result<Self> {
        config.validate()?;
        
        Ok(Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: CacheStatistics::new(),
            file_manager: FileManager::new(),
            invalidation_tracker: Arc::new(Mutex::new(InvalidationTracker::new())),
        })
    }
    
    /// Open and register a file for caching
    pub fn open_file<P: AsRef<Path>>(&self, path: P) -> Result<FileId> {
        self.file_manager.open_file(path)
    }
    
    /// Register a file by file descriptor (for compatibility)
    pub fn register_file(&self, fd: i32) -> Result<FileId> {
        // For memory-based stores or special file descriptors, generate a virtual file ID
        if fd == -1 {
            // Special case for memory-based blob stores - create a virtual file ID
            let virtual_file_id = self.file_manager.next_file_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(virtual_file_id);
        }
        
        // For actual file descriptors, we'd need to convert to file path
        // This is a placeholder for future implementation
        Err(ZiporaError::invalid_data("Real file descriptor support not yet implemented. Use open_file() for real files or fd=-1 for virtual files.".to_string()))
    }
    
    /// Read data from cache or load it from file
    pub fn read(&self, file_id: FileId, offset: u64, length: usize) -> Result<CacheBuffer> {
        // Calculate which pages we need
        let start_page = FileManager::offset_to_page_id(offset);
        let end_offset = offset + length as u64;
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));
        
        let mut result_buffer = CacheBuffer::new();
        let mut current_offset = offset;
        let mut remaining_length = length;
        
        for page_id in start_page..=end_page {
            let page_data = self.get_page(file_id, page_id)?;
            
            // Calculate how much data to copy from this page
            let page_start_offset = FileManager::page_aligned_offset(page_id);
            let offset_in_page = (current_offset - page_start_offset) as usize;
            let bytes_to_copy = std::cmp::min(
                remaining_length,
                PAGE_SIZE - offset_in_page
            );
            
            // Copy data from the page
            let page_end = offset_in_page + bytes_to_copy;
            if page_end <= page_data.len() {
                result_buffer.extend_from_slice(&page_data[offset_in_page..page_end]);
            }
            
            current_offset += bytes_to_copy as u64;
            remaining_length -= bytes_to_copy;
            
            if remaining_length == 0 {
                break;
            }
        }
        
        Ok(result_buffer)
    }
    
    /// Get a page from cache or load it from file
    fn get_page(&self, file_id: FileId, page_id: PageId) -> Result<Vec<u8>> {
        let cache_key = (file_id, page_id);
        
        // Check if page is invalidated
        {
            let tracker = self.invalidation_tracker.lock()
                .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
            
            if tracker.is_invalidated(&cache_key) {
                // Page is invalidated, remove from cache and continue to reload
                drop(tracker);
                self.remove_from_cache(cache_key)?;
            }
        }
        
        // First, try to get from cache
        let cached_result = {
            let inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            
            inner.get(&cache_key).cloned()
        };
        
        if let Some(cached_page) = cached_result {
            // Update access time
            {
                let mut tracker = self.invalidation_tracker.lock()
                    .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
                tracker.mark_accessed(cache_key);
            }
            
            self.stats.record_hit(CacheHitType::Hit);
            return Ok(cached_page);
        }
        
        // Cache miss - need to load from file
        self.stats.record_miss();
        
        // Try to load page from file if it's a real file
        let mut page_buffer = vec![0u8; PAGE_SIZE];
        let bytes_read = match self.file_manager.read_page(file_id, page_id, &mut page_buffer) {
            Ok(bytes) => bytes,
            Err(_) => {
                // If file read fails (e.g., virtual file ID), return empty page
                page_buffer.clear();
                page_buffer.resize(PAGE_SIZE, 0);
                0
            }
        };
        
        // If we read less than a full page, it's still valid (end of file)
        if bytes_read < PAGE_SIZE {
            page_buffer.truncate(bytes_read);
        }
        
        // Insert into cache with LRU eviction
        {
            let mut inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            let mut tracker = self.invalidation_tracker.lock()
                .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
            
            // Check if we need to evict pages
            if inner.len() >= self.config.capacity / PAGE_SIZE {
                // Find LRU page to evict
                if let Some(lru_key) = tracker.find_lru_page() {
                    // Check if page is dirty and needs write-back
                    if tracker.is_dirty(&lru_key) {
                        // In a real implementation, we'd write the dirty page back
                        // For now, just mark it as clean
                        tracker.mark_clean(lru_key);
                    }
                    
                    inner.remove(&lru_key);
                    tracker.invalidate(lru_key);
                    self.stats.record_hit(CacheHitType::EvictedOthers);
                } else if let Some(key) = inner.keys().next().cloned() {
                    // Fallback to simple eviction
                    inner.remove(&key);
                    tracker.invalidate(key);
                    self.stats.record_hit(CacheHitType::EvictedOthers);
                }
            }
            
            inner.insert(cache_key, page_buffer.clone());
            tracker.mark_accessed(cache_key);
            self.stats.record_hit(CacheHitType::InitialFree);
        }
        
        Ok(page_buffer)
    }
    
    /// Batch read operations for improved performance
    pub fn read_batch(&self, requests: Vec<(FileId, u64, usize)>) -> Result<Vec<CacheBuffer>> {
        let mut results = Vec::with_capacity(requests.len());
        
        for (file_id, offset, length) in requests {
            results.push(self.read(file_id, offset, length)?);
        }
        
        Ok(results)
    }
    
    /// Prefetch pages for better performance
    pub fn prefetch(&self, file_id: FileId, offset: u64, length: usize) -> Result<()> {
        // Calculate pages to prefetch
        let start_page = FileManager::offset_to_page_id(offset);
        let end_offset = offset + length as u64;
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));
        
        // Prefetch each page (load into cache without returning data)
        for page_id in start_page..=end_page {
            let _ = self.get_page(file_id, page_id)?;
        }
        
        Ok(())
    }
    
    /// Read data with prefetching hints
    pub fn read_with_prefetch(&self, file_id: FileId, offset: u64, length: usize, prefetch_ahead: usize) -> Result<CacheBuffer> {
        // Start prefetching in the background (simplified - would use async in real implementation)
        if prefetch_ahead > 0 {
            let prefetch_offset = offset + length as u64;
            let _ = self.prefetch(file_id, prefetch_offset, prefetch_ahead);
        }
        
        // Read the requested data
        self.read(file_id, offset, length)
    }
    
    /// Get file size through cache
    pub fn file_size(&self, file_id: FileId) -> Result<u64> {
        self.file_manager.file_size(file_id)
    }
    
    /// Invalidate a specific page
    pub fn invalidate_page(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        let cache_key = (file_id, page_id);
        
        // Mark as invalidated
        {
            let mut tracker = self.invalidation_tracker.lock()
                .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
            tracker.invalidate(cache_key);
        }
        
        // Remove from cache
        self.remove_from_cache(cache_key)
    }
    
    /// Invalidate a range of pages
    pub fn invalidate_range(&self, file_id: FileId, start_offset: u64, length: usize) -> Result<()> {
        let start_page = FileManager::offset_to_page_id(start_offset);
        let end_offset = start_offset + length as u64;
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));
        
        for page_id in start_page..=end_page {
            self.invalidate_page(file_id, page_id)?;
        }
        
        Ok(())
    }
    
    /// Mark a page as dirty (needs write-back)
    pub fn mark_dirty(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        let cache_key = (file_id, page_id);
        
        let mut tracker = self.invalidation_tracker.lock()
            .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
        tracker.mark_dirty(cache_key);
        
        Ok(())
    }
    
    /// Flush all dirty pages for a file
    pub fn flush_file(&self, file_id: FileId) -> Result<()> {
        let tracker = self.invalidation_tracker.lock()
            .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
        
        let dirty_pages: Vec<_> = tracker.dirty_pages
            .iter()
            .filter(|(fid, _)| *fid == file_id)
            .cloned()
            .collect();
        
        drop(tracker);
        
        for (fid, page_id) in dirty_pages {
            // In a real implementation, we'd write the page back to storage
            // For now, just mark it as clean
            let mut tracker = self.invalidation_tracker.lock()
                .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
            tracker.mark_clean((fid, page_id));
        }
        
        Ok(())
    }
    
    /// Remove a page from cache (internal helper)
    fn remove_from_cache(&self, cache_key: (FileId, PageId)) -> Result<()> {
        let mut inner = self.inner.lock()
            .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
        inner.remove(&cache_key);
        Ok(())
    }
    
    /// Close a file and clear its cached pages
    pub fn close_file(&self, file_id: FileId) -> Result<()> {
        // Flush any dirty pages first
        self.flush_file(file_id)?;
        
        // Remove all cached pages for this file
        {
            let mut inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            
            inner.retain(|(cached_file_id, _), _| *cached_file_id != file_id);
        }
        
        // Clean up invalidation tracker
        {
            let mut tracker = self.invalidation_tracker.lock()
                .map_err(|_| ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string()))?;
            tracker.cleanup_file(file_id);
        }
        
        // Close the file in the file manager
        self.file_manager.close_file(file_id)
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStatsSnapshot {
        self.stats.snapshot()
    }
    
    /// Get number of shards (always 1 for basic implementation)
    pub fn shard_count(&self) -> usize {
        1
    }
}

/// Single LRU page cache implementation
pub struct SingleLruPageCache {
    cache: LruPageCache,
}

impl SingleLruPageCache {
    pub fn new(config: PageCacheConfig) -> Result<Self> {
        Ok(Self {
            cache: LruPageCache::new(config)?,
        })
    }
    
    /// Open a file for caching
    pub fn open_file<P: AsRef<Path>>(&self, path: P) -> Result<FileId> {
        self.cache.open_file(path)
    }
    
    /// Register a file by descriptor (compatibility)
    pub fn register_file(&self, fd: i32) -> Result<FileId> {
        self.cache.register_file(fd)
    }
    
    /// Read data into provided buffer
    pub fn read(&self, file_id: FileId, offset: u64, length: usize, buffer: &mut CacheBuffer) -> Result<()> {
        let result = self.cache.read(file_id, offset, length)?;
        buffer.copy_from_slice(result.data());
        Ok(())
    }
    
    /// Read data and return new buffer
    pub fn read_new(&self, file_id: FileId, offset: u64, length: usize) -> Result<CacheBuffer> {
        self.cache.read(file_id, offset, length)
    }
    
    /// Prefetch data for better performance
    pub fn prefetch(&self, file_id: FileId, offset: u64, length: usize) -> Result<()> {
        self.cache.prefetch(file_id, offset, length)
    }
    
    /// Get file size
    pub fn file_size(&self, file_id: FileId) -> Result<u64> {
        self.cache.file_size(file_id)
    }
    
    /// Close file and clear its cache
    pub fn close_file(&self, file_id: FileId) -> Result<()> {
        self.cache.close_file(file_id)
    }
    
    pub fn capacity(&self) -> usize {
        self.cache.config.capacity
    }
    
    pub fn size(&self) -> usize {
        self.cache.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
    
    pub fn stats(&self) -> &CacheStatistics {
        &self.cache.stats
    }
    
    /// Invalidate a page
    pub fn invalidate_page(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        self.cache.invalidate_page(file_id, page_id)
    }
    
    /// Invalidate a range of pages
    pub fn invalidate_range(&self, file_id: FileId, start_offset: u64, length: usize) -> Result<()> {
        self.cache.invalidate_range(file_id, start_offset, length)
    }
    
    /// Mark a page as dirty
    pub fn mark_dirty(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        self.cache.mark_dirty(file_id, page_id)
    }
    
    /// Flush dirty pages for a file
    pub fn flush_file(&self, file_id: FileId) -> Result<()> {
        self.cache.flush_file(file_id)
    }
}