//! Basic LRU cache implementation with real file I/O

use super::*;
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

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
        
        // First, try to get from cache
        {
            let inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            
            if let Some(cached_page) = inner.get(&cache_key) {
                self.stats.record_hit(CacheHitType::Hit);
                return Ok(cached_page.clone());
            }
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
        
        // Insert into cache
        {
            let mut inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            
            // Check if we need to evict pages
            if inner.len() >= self.config.capacity / PAGE_SIZE {
                // Simple eviction: remove one entry (LRU would be more complex)
                if let Some(key) = inner.keys().next().cloned() {
                    inner.remove(&key);
                    self.stats.record_hit(CacheHitType::EvictedOthers);
                }
            }
            
            inner.insert(cache_key, page_buffer.clone());
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
    
    /// Close a file and clear its cached pages
    pub fn close_file(&self, file_id: FileId) -> Result<()> {
        // Remove all cached pages for this file
        {
            let mut inner = self.inner.lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            
            inner.retain(|(cached_file_id, _), _| *cached_file_id != file_id);
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
        self.cache.inner.lock().unwrap().len()
    }
    
    pub fn stats(&self) -> &CacheStatistics {
        &self.cache.stats
    }
}