//! Basic LRU cache implementation with real file I/O

use super::*;
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Cache invalidation and LRU access tracking
#[derive(Debug, Clone)]
struct InvalidationTracker {
    /// Set of invalidated pages awaiting reload
    invalidated_pages: std::collections::HashSet<(FileId, PageId)>,
    /// Dirty pages that need write-back
    dirty_pages: std::collections::HashSet<(FileId, PageId)>,
    /// Monotonic access sequence number for each cached page
    access_seq: std::collections::HashMap<(FileId, PageId), u64>,
    /// Next sequence number to assign on access
    next_seq: u64,
}

type PageCacheMap = HashMap<(FileId, PageId), Vec<u8>>;

/// Basic LRU page cache for blob operations with real file I/O
pub struct LruPageCache {
    /// Cache storage: (file_id, page_id) -> page data
    inner: Arc<Mutex<PageCacheMap>>,
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
            access_seq: std::collections::HashMap::new(),
            next_seq: 1,
        }
    }

    fn mark_accessed(&mut self, key: (FileId, PageId)) {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        self.access_seq.insert(key, seq);
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
        self.access_seq.remove(&key);
    }

    fn take_invalidated(&mut self, key: &(FileId, PageId)) -> bool {
        self.invalidated_pages.remove(key)
    }

    fn evict(&mut self, key: (FileId, PageId)) {
        self.dirty_pages.remove(&key);
        self.access_seq.remove(&key);
    }

    fn is_dirty(&self, key: &(FileId, PageId)) -> bool {
        self.dirty_pages.contains(key)
    }

    fn find_lru_page(&self) -> Option<(FileId, PageId)> {
        self.access_seq
            .iter()
            .min_by_key(|(_, seq)| *seq)
            .map(|(key, _)| *key)
    }

    fn cleanup_file(&mut self, file_id: FileId) {
        self.invalidated_pages.retain(|(fid, _)| *fid != file_id);
        self.dirty_pages.retain(|(fid, _)| *fid != file_id);
        self.access_seq.retain(|&(fid, _), _| fid != file_id);
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

    /// Register a file by file descriptor (or `-1` for an in-memory virtual file)
    pub fn register_file(&self, fd: i32) -> Result<FileId> {
        if fd == -1 {
            return self.file_manager.register_virtual_file();
        }
        if fd < 0 {
            return Err(ZiporaError::invalid_data(format!(
                "Invalid file descriptor {} (expected >= 0 or -1 for virtual file)",
                fd
            )));
        }

        #[cfg(unix)]
        {
            let fd_path = format!("/dev/fd/{}", fd);
            self.file_manager.open_file(fd_path)
        }
        #[cfg(not(unix))]
        {
            Err(ZiporaError::invalid_data(
                "File descriptor registration is only supported on Unix platforms; use open_file() or fd=-1".to_string(),
            ))
        }
    }

    /// Read data from cache or load it from file
    pub fn read(&self, file_id: FileId, offset: u64, length: usize) -> Result<CacheBuffer> {
        if length == 0 {
            let _ = self.file_manager.file_size(file_id)?;
            return Ok(CacheBuffer::new());
        }

        // Calculate which pages we need
        let start_page = FileManager::offset_to_page_id(offset);
        let end_offset = offset.saturating_add(length as u64);
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));

        let mut result_buffer = CacheBuffer::new();
        let mut current_offset = offset;
        let mut remaining_length = length;

        for page_id in start_page..=end_page {
            let page_data = self.get_page(file_id, page_id)?;

            // Calculate how much data to copy from this page
            let page_start_offset = FileManager::page_aligned_offset(page_id);
            let offset_in_page = (current_offset - page_start_offset) as usize;
            if offset_in_page >= page_data.len() {
                break;
            }

            let bytes_to_copy = std::cmp::min(remaining_length, PAGE_SIZE - offset_in_page);
            let available = std::cmp::min(bytes_to_copy, page_data.len() - offset_in_page);
            result_buffer.extend_from_slice(&page_data[offset_in_page..offset_in_page + available]);

            current_offset += available as u64;
            remaining_length -= available;

            if remaining_length == 0 || available < bytes_to_copy {
                break;
            }
        }

        self.stats.record_bytes_read(result_buffer.len() as u64);
        Ok(result_buffer)
    }

    /// Write data into cache and mark affected pages as dirty for write-back
    pub fn write(&self, file_id: FileId, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            let _ = self.file_manager.file_size(file_id)?;
            return Ok(());
        }

        let start_page = FileManager::offset_to_page_id(offset);
        let end_offset = offset.saturating_add(data.len() as u64);
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));

        let mut current_offset = offset;
        let mut data_pos = 0usize;

        for page_id in start_page..=end_page {
            let mut page_data = self.get_page(file_id, page_id)?;
            let page_start_offset = FileManager::page_aligned_offset(page_id);
            let offset_in_page = (current_offset - page_start_offset) as usize;
            let bytes_to_write = std::cmp::min(data.len() - data_pos, PAGE_SIZE - offset_in_page);
            let required_len = offset_in_page + bytes_to_write;
            if page_data.len() < required_len {
                page_data.resize(required_len, 0);
            }
            page_data[offset_in_page..required_len]
                .copy_from_slice(&data[data_pos..data_pos + bytes_to_write]);

            let cache_key = (file_id, page_id);
            {
                let mut inner = self
                    .inner
                    .lock()
                    .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
                let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                    ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
                })?;
                inner.insert(cache_key, page_data);
                tracker.mark_accessed(cache_key);
                tracker.mark_dirty(cache_key);
            }

            current_offset += bytes_to_write as u64;
            data_pos += bytes_to_write;
        }

        Ok(())
    }

    /// Get a page from cache or load it from file
    fn get_page(&self, file_id: FileId, page_id: PageId) -> Result<Vec<u8>> {
        let cache_key = (file_id, page_id);

        // Consume pending invalidation if present and evict stale entry before reload
        let was_invalidated = {
            let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;
            tracker.take_invalidated(&cache_key)
        };
        if was_invalidated {
            self.remove_from_cache(cache_key)?;
        }

        // First, try to get from cache
        let cached_result = {
            let inner = self
                .inner
                .lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;

            inner.get(&cache_key).cloned()
        };

        if let Some(cached_page) = cached_result {
            // Update access sequence
            {
                let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                    ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
                })?;
                tracker.mark_accessed(cache_key);
            }

            self.stats.record_hit(CacheHitType::Hit);
            return Ok(cached_page);
        }

        // Cache miss - load from file manager
        self.stats.record_miss();

        let mut page_buffer = vec![0u8; PAGE_SIZE];
        let bytes_read = match self
            .file_manager
            .read_page(file_id, page_id, &mut page_buffer)
        {
            Ok(bytes) => bytes,
            Err(err) => {
                self.stats.record_load_failure();
                return Err(err);
            }
        };

        // If we read less than a full page, truncate to valid file bytes
        if bytes_read < PAGE_SIZE {
            page_buffer.truncate(bytes_read);
        }

        // Insert into cache with LRU eviction
        let mut dirty_writeback: Option<((FileId, PageId), Vec<u8>)> = None;
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
            let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;

            let max_pages = (self.config.capacity / self.config.page_size).max(1);
            if inner.len() >= max_pages && !inner.contains_key(&cache_key) {
                let victim_key = tracker
                    .find_lru_page()
                    .or_else(|| inner.keys().next().copied());
                if let Some(lru_key) = victim_key {
                    let evicted_data = inner.remove(&lru_key);
                    if tracker.is_dirty(&lru_key)
                        && let Some(data) = evicted_data
                    {
                        dirty_writeback = Some((lru_key, data));
                    }
                    tracker.evict(lru_key);
                    self.stats.record_eviction();
                    self.stats.record_hit(CacheHitType::EvictedOthers);
                }
            }

            inner.insert(cache_key, page_buffer.clone());
            tracker.mark_accessed(cache_key);
            self.stats.record_bytes_cached(page_buffer.len() as u64);
            self.stats.record_hit(CacheHitType::InitialFree);
        }

        if let Some(((dirty_fid, dirty_pid), dirty_data)) = dirty_writeback {
            self.file_manager
                .write_page(dirty_fid, dirty_pid, &dirty_data)?;
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
        if length == 0 {
            return Ok(());
        }
        // Calculate pages to prefetch
        let start_page = FileManager::offset_to_page_id(offset);
        let end_offset = offset.saturating_add(length as u64);
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));

        // Prefetch each page (load into cache without returning data)
        for page_id in start_page..=end_page {
            let _ = self.get_page(file_id, page_id)?;
        }

        Ok(())
    }

    /// Read data with synchronous lookahead prefetching
    pub fn read_with_prefetch(
        &self,
        file_id: FileId,
        offset: u64,
        length: usize,
        prefetch_ahead: usize,
    ) -> Result<CacheBuffer> {
        if prefetch_ahead > 0 {
            let prefetch_offset = offset.saturating_add(length as u64);
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

        // Mark as invalidated and remove from active cache
        {
            let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;
            tracker.invalidate(cache_key);
        }

        self.remove_from_cache(cache_key)
    }

    /// Invalidate a range of pages
    pub fn invalidate_range(
        &self,
        file_id: FileId,
        start_offset: u64,
        length: usize,
    ) -> Result<()> {
        if length == 0 {
            return Ok(());
        }
        let start_page = FileManager::offset_to_page_id(start_offset);
        let end_offset = start_offset.saturating_add(length as u64);
        let end_page = FileManager::offset_to_page_id(end_offset.saturating_sub(1));

        for page_id in start_page..=end_page {
            self.invalidate_page(file_id, page_id)?;
        }

        Ok(())
    }

    /// Mark a page as dirty (needs write-back)
    pub fn mark_dirty(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        let cache_key = (file_id, page_id);

        let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
            ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
        })?;
        tracker.mark_dirty(cache_key);

        Ok(())
    }

    /// Flush all dirty pages for a file to underlying storage
    pub fn flush_file(&self, file_id: FileId) -> Result<()> {
        let dirty_pages: Vec<(FileId, PageId)> = {
            let tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;

            let mut pages: Vec<_> = tracker
                .dirty_pages
                .iter()
                .filter(|(fid, _)| *fid == file_id)
                .copied()
                .collect();
            pages.sort_unstable_by_key(|&(_, pid)| pid);
            pages
        };

        for (fid, page_id) in dirty_pages {
            let page_data = {
                let inner = self
                    .inner
                    .lock()
                    .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;
                inner.get(&(fid, page_id)).cloned()
            };

            if let Some(data) = page_data {
                self.file_manager.write_page(fid, page_id, &data)?;
            }

            let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;
            tracker.mark_clean((fid, page_id));
        }

        Ok(())
    }

    /// Remove a page from cache (internal helper)
    fn remove_from_cache(&self, cache_key: (FileId, PageId)) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
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
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| ZiporaError::invalid_data("Cache lock poisoned".to_string()))?;

            inner.retain(|(cached_file_id, _), _| *cached_file_id != file_id);
        }

        // Clean up invalidation tracker
        {
            let mut tracker = self.invalidation_tracker.lock().map_err(|_| {
                ZiporaError::invalid_data("Invalidation tracker lock poisoned".to_string())
            })?;
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
    pub fn read(
        &self,
        file_id: FileId,
        offset: u64,
        length: usize,
        buffer: &mut CacheBuffer,
    ) -> Result<()> {
        let result = self.cache.read(file_id, offset, length)?;
        buffer.copy_from_slice(result.data());
        Ok(())
    }

    /// Read data and return new buffer
    pub fn read_new(&self, file_id: FileId, offset: u64, length: usize) -> Result<CacheBuffer> {
        self.cache.read(file_id, offset, length)
    }

    /// Write data into cache and mark affected pages dirty
    pub fn write(&self, file_id: FileId, offset: u64, data: &[u8]) -> Result<()> {
        self.cache.write(file_id, offset, data)
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

    #[inline]
    pub fn capacity(&self) -> usize {
        self.cache.config.capacity
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.cache
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    pub fn stats(&self) -> &CacheStatistics {
        &self.cache.stats
    }

    /// Invalidate a page
    pub fn invalidate_page(&self, file_id: FileId, page_id: PageId) -> Result<()> {
        self.cache.invalidate_page(file_id, page_id)
    }

    /// Invalidate a range of pages
    pub fn invalidate_range(
        &self,
        file_id: FileId,
        start_offset: u64,
        length: usize,
    ) -> Result<()> {
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
