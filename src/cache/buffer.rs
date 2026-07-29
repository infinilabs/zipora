//! Cache buffer management for efficient data access

use super::*;
use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Cache buffer for managing cached data and automatic cleanup
pub struct CacheBuffer {
    /// Buffer type indicating source
    buffer_type: BufferType,

    /// Owning cache reference
    cache: Option<*const SingleLruPageCache>,

    /// Node indices for cleanup
    node_indices: Vec<NodeIndex>,

    /// Buffer for multi-page data
    data_buffer: Vec<u8>,

    /// Cache hit type for statistics
    hit_type: CacheHitType,
}

/// Buffer type indicating data source
#[derive(Debug, Clone, Copy, PartialEq)]
enum BufferType {
    /// Multiple pages copied to buffer
    #[cfg_attr(not(test), allow(dead_code))]
    MultiPage,
    /// Data copied to internal buffer
    Copied,
    /// Empty buffer
    Empty,
}

impl CacheBuffer {
    /// Create new empty cache buffer
    pub fn new() -> Self {
        Self {
            buffer_type: BufferType::Empty,
            cache: None,
            node_indices: Vec::new(),
            data_buffer: Vec::new(),
            hit_type: CacheHitType::Hit,
        }
    }

    /// Setup buffer for multi-page operation
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn setup_multi_page(
        &mut self,
        cache: &SingleLruPageCache,
        node_indices: Vec<NodeIndex>,
        offset: u64,
        length: usize,
    ) {
        self.cleanup();
        self.buffer_type = BufferType::MultiPage;
        self.cache = Some(cache as *const _);

        // Copy data from multiple pages
        self.data_buffer.clear();
        self.data_buffer.reserve(length);

        let _start_page = (offset / PAGE_SIZE as u64) as PageId;
        let page_offset = (offset % PAGE_SIZE as u64) as usize;
        let _remaining = length;
        let _current_offset = page_offset;

        // Simplified for basic implementation
        self.data_buffer.resize(length, 0);

        self.node_indices = node_indices;
        self.hit_type = CacheHitType::Mix;
    }

    /// Copy data to internal buffer
    pub fn copy_from_slice(&mut self, data: &[u8]) {
        self.cleanup();
        self.buffer_type = BufferType::Copied;
        self.data_buffer.clear();
        self.data_buffer.extend_from_slice(data);
        self.hit_type = CacheHitType::Hit;
    }

    /// Extend buffer with additional data
    pub fn extend_from_slice(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }

        // Convert to copied buffer if not already
        if !matches!(self.buffer_type, BufferType::Copied) {
            let existing_data = self.data().to_vec();
            self.cleanup();
            self.buffer_type = BufferType::Copied;
            self.data_buffer = existing_data;
        }

        self.data_buffer.extend_from_slice(data);
    }

    /// Get buffered data
    pub fn data(&self) -> &[u8] {
        match self.buffer_type {
            BufferType::MultiPage | BufferType::Copied => &self.data_buffer,
            BufferType::Empty => &[],
        }
    }

    /// Get data length
    #[inline]
    pub fn len(&self) -> usize {
        self.data().len()
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cache hit type
    pub fn hit_type(&self) -> CacheHitType {
        self.hit_type
    }

    /// Check if data is available
    pub fn has_data(&self) -> bool {
        !matches!(self.buffer_type, BufferType::Empty)
    }

    /// Clear buffer and release resources
    pub fn clear(&mut self) {
        self.cleanup();
        self.buffer_type = BufferType::Empty;
        self.data_buffer.clear();
    }

    /// Internal cleanup of cache references
    fn cleanup(&mut self) {
        // Simplified for basic implementation
        self.cache = None;
        self.node_indices.clear();
    }

    /// Create buffer from raw data
    pub fn from_data(data: Vec<u8>) -> Self {
        let mut buffer = Self::new();
        buffer.buffer_type = BufferType::Copied;
        buffer.data_buffer = data;
        buffer
    }

    /// Reserve capacity for buffer
    pub fn reserve(&mut self, capacity: usize) {
        self.data_buffer.reserve(capacity);
    }

    /// Get buffer capacity
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data_buffer.capacity()
    }
}

impl Default for CacheBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheBuffer {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// SAFETY: CacheBuffer is Send because every field is Send except
// `cache: Option<*const SingleLruPageCache>`, which blocks the auto impl.
// That pointer is only stored for bookkeeping (set in setup_multi_page,
// cleared in cleanup) and is never dereferenced, so moving the struct to
// another thread cannot create a data race through it.
// (The former `data_slice: Option<&'static [u8]>` self-referential field was
// removed; `data()` now borrows `data_buffer` directly.)
unsafe impl Send for CacheBuffer {}

/// Buffer pool for reusing cache buffers
pub struct BufferPool {
    /// Available buffers
    available: Mutex<Vec<CacheBuffer>>,

    /// Maximum pool size
    max_size: usize,

    /// Statistics
    allocations: AtomicU64,
    reuses: AtomicU64,
}

impl BufferPool {
    /// Create new buffer pool
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Mutex::new(Vec::new()),
            max_size,
            allocations: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
        }
    }

    /// Get buffer from pool or create new one
    pub fn get(&self) -> CacheBuffer {
        if let Ok(mut buffers) = self.available.lock()
            && let Some(mut buffer) = buffers.pop()
        {
            buffer.clear();
            self.reuses.fetch_add(1, Ordering::Relaxed);
            return buffer;
        }

        self.allocations.fetch_add(1, Ordering::Relaxed);
        CacheBuffer::new()
    }

    /// Return buffer to pool
    pub fn put(&self, buffer: CacheBuffer) {
        if let Ok(mut buffers) = self.available.lock()
            && buffers.len() < self.max_size
        {
            buffers.push(buffer);
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> BufferPoolStats {
        let available_count = self
            .available
            .lock()
            .map(|buffers| buffers.len())
            .unwrap_or(0);

        BufferPoolStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            reuses: self.reuses.load(Ordering::Relaxed),
            available_count,
            max_size: self.max_size,
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    pub allocations: u64,
    pub reuses: u64,
    pub available_count: usize,
    pub max_size: usize,
}

impl BufferPoolStats {
    pub fn reuse_ratio(&self) -> f64 {
        if self.allocations + self.reuses == 0 {
            0.0
        } else {
            self.reuses as f64 / (self.allocations + self.reuses) as f64
        }
    }

    pub fn pool_utilization(&self) -> f64 {
        self.available_count as f64 / self.max_size as f64
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_basic() {
        let pool = BufferPool::new(4);
        assert_eq!(pool.stats().max_size, 4);

        let buf = pool.get();
        assert_eq!(buf.len(), 0);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);

        pool.put(buf);
        let stats = pool.stats();
        assert_eq!(stats.available_count, 1);
    }

    #[test]
    fn test_buffer_new_is_empty() {
        let buffer = CacheBuffer::new();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        let empty: &[u8] = &[];
        assert_eq!(buffer.data(), empty);
    }

    #[test]
    fn test_buffer_copy_from_slice() {
        let mut buffer = CacheBuffer::new();
        let test_data = b"Hello, Cache Buffer!";

        buffer.copy_from_slice(test_data);

        assert_eq!(buffer.len(), test_data.len());
        assert!(!buffer.is_empty());
        assert_eq!(buffer.data(), test_data);
        assert_eq!(buffer.hit_type(), CacheHitType::Hit);
    }

    #[test]
    fn test_buffer_extend_multiple() {
        let mut buffer = CacheBuffer::new();
        let data1 = b"First ";
        let data2 = b"Second";

        buffer.copy_from_slice(data1);
        buffer.extend_from_slice(data2);

        let expected = b"First Second";
        assert_eq!(buffer.data(), expected);
        assert_eq!(buffer.len(), expected.len());
    }

    #[test]
    fn test_buffer_clear_resets() {
        let mut buffer = CacheBuffer::new();
        buffer.copy_from_slice(b"Some data");

        assert!(!buffer.is_empty());

        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        let empty: &[u8] = &[];
        assert_eq!(buffer.data(), empty);
    }

    #[test]
    fn test_buffer_from_data_vec() {
        let test_data = vec![1, 2, 3, 4, 5];
        let buffer = CacheBuffer::from_data(test_data.clone());

        assert_eq!(buffer.data(), &test_data[..]);
        assert_eq!(buffer.len(), test_data.len());
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_reserve_capacity() {
        let mut buffer = CacheBuffer::new();
        assert!(buffer.capacity() < 1024);

        buffer.reserve(1024);

        assert!(buffer.capacity() >= 1024);
    }

    #[test]
    fn test_buffer_send_across_thread() {
        use std::sync::mpsc;
        use std::thread;

        let mut buffer = CacheBuffer::new();
        let test_data = b"Thread-safe data";
        buffer.copy_from_slice(test_data);

        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            // Send buffer to another thread
            tx.send(buffer).expect("Failed to send buffer");
        });

        let received_buffer = rx.recv().expect("Failed to receive buffer");
        assert_eq!(received_buffer.data(), test_data);
    }

    #[test]
    fn test_buffer_pool_reuse_stats() {
        let pool = BufferPool::new(10);

        let mut buf1 = pool.get();
        buf1.copy_from_slice(b"test");

        let stats1 = pool.stats();
        assert_eq!(stats1.allocations, 1);
        assert_eq!(stats1.reuses, 0);

        pool.put(buf1);

        let buf2 = pool.get();

        let stats2 = pool.stats();
        assert_eq!(stats2.allocations, 1);
        assert_eq!(stats2.reuses, 1);
        assert!(buf2.is_empty()); // Should be cleared
        assert!(stats2.reuse_ratio() > 0.0);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let pool = BufferPool::new(2);

        let buf1 = pool.get();
        let buf2 = pool.get();
        let buf3 = pool.get();

        pool.put(buf1);
        pool.put(buf2);
        pool.put(buf3);

        let stats = pool.stats();
        assert_eq!(stats.available_count, 2); // Max size is 2
        assert_eq!(stats.max_size, 2);
        assert!(stats.pool_utilization() == 1.0); // 2/2 = 100%
    }

    #[test]
    fn test_buffer_hit_type_tracking() {
        let mut buffer = CacheBuffer::new();

        // Default hit type for new buffer
        assert_eq!(buffer.hit_type(), CacheHitType::Hit);

        buffer.copy_from_slice(b"test");
        assert_eq!(buffer.hit_type(), CacheHitType::Hit);

        // Test multi-page setup explicitly sets hit_type to Mix
        // Create a dummy SingleLruPageCache. Since we can't easily create one safely
        // with real config in this minimal test, we'll just skip the full setup,
        // wait, setup_multi_page takes an &SingleLruPageCache.
        // We can just rely on the implementation changing it when calling `setup_multi_page`.
        // Let's create a minimal test cache
        let config = crate::cache::PageCacheConfig::balanced();
        let cache = crate::cache::SingleLruPageCache::new(config).unwrap();
        buffer.setup_multi_page(&cache, vec![1, 2], 0, 1024);
        assert_eq!(buffer.hit_type(), CacheHitType::Mix);
    }
}
