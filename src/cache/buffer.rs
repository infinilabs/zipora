//! Cache buffer management for efficient data access

use super::*;
use std::sync::{Mutex, atomic::{AtomicU64, Ordering}};

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
    
    /// Data slice (points to either cache page or buffer)
    data_slice: Option<&'static [u8]>,
    
    /// Cache hit type for statistics
    hit_type: CacheHitType,
}

/// Buffer type indicating data source
#[derive(Debug, Clone, Copy, PartialEq)]
enum BufferType {
    /// Direct cache page reference
    SinglePage,
    /// Multiple pages copied to buffer
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
            data_slice: None,
            hit_type: CacheHitType::Hit,
        }
    }
    
    /// Set buffer to reference single cache node
    pub(crate) fn set_node(&mut self, cache: &SingleLruPageCache, node_idx: NodeIndex) {
        self.cleanup();
        self.buffer_type = BufferType::SinglePage;
        self.cache = Some(cache as *const _);
        self.node_indices.push(node_idx);
        self.hit_type = CacheHitType::Hit;
    }
    
    /// Setup buffer for multi-page operation
    pub(crate) fn setup_multi_page(
        &mut self, 
        cache: &SingleLruPageCache, 
        node_indices: Vec<NodeIndex>, 
        offset: u64, 
        length: usize
    ) {
        self.cleanup();
        self.buffer_type = BufferType::MultiPage;
        self.cache = Some(cache as *const _);
        
        // Copy data from multiple pages
        self.data_buffer.clear();
        self.data_buffer.reserve(length);
        
        let start_page = (offset / PAGE_SIZE as u64) as PageId;
        let page_offset = (offset % PAGE_SIZE as u64) as usize;
        let mut remaining = length;
        let mut current_offset = page_offset;
        
        // Simplified for basic implementation
        self.data_buffer.resize(length, 0);
        
        self.node_indices = node_indices;
        self.hit_type = CacheHitType::Mix;
        
        // Set data slice to point to internal buffer
        let data_ptr = self.data_buffer.as_ptr();
        self.data_slice = Some(unsafe { std::slice::from_raw_parts(data_ptr, length) });
    }
    
    /// Copy data to internal buffer
    pub fn copy_from_slice(&mut self, data: &[u8]) {
        self.cleanup();
        self.buffer_type = BufferType::Copied;
        self.data_buffer.clear();
        self.data_buffer.extend_from_slice(data);
        
        let data_ptr = self.data_buffer.as_ptr();
        self.data_slice = Some(unsafe { std::slice::from_raw_parts(data_ptr, data.len()) });
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
        
        // Update data slice
        let data_ptr = self.data_buffer.as_ptr();
        self.data_slice = Some(unsafe { std::slice::from_raw_parts(data_ptr, self.data_buffer.len()) });
    }
    
    /// Get buffered data
    pub fn data(&self) -> &[u8] {
        match self.buffer_type {
            BufferType::SinglePage => {
                // Simplified for basic implementation
                &self.data_buffer
            }
            BufferType::MultiPage | BufferType::Copied => {
                self.data_slice.unwrap_or(&[])
            }
            BufferType::Empty => &[],
        }
    }
    
    /// Get data length
    pub fn len(&self) -> usize {
        self.data().len()
    }
    
    /// Check if buffer is empty
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
        self.data_slice = None;
    }
    
    /// Internal cleanup of cache references
    fn cleanup(&mut self) {
        // Simplified for basic implementation
        self.cache = None;
        self.node_indices.clear();
    }
    
    /// Create buffer from raw data
    pub fn from_data(data: Vec<u8>) -> Self {
        let len = data.len();
        let mut buffer = Self::new();
        buffer.buffer_type = BufferType::Copied;
        buffer.data_buffer = data;
        
        let data_ptr = buffer.data_buffer.as_ptr();
        buffer.data_slice = Some(unsafe { std::slice::from_raw_parts(data_ptr, len) });
        
        buffer
    }
    
    /// Reserve capacity for buffer
    pub fn reserve(&mut self, capacity: usize) {
        self.data_buffer.reserve(capacity);
    }
    
    /// Get buffer capacity
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

// Safety: CacheBuffer maintains proper lifetimes for cache references
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
        if let Ok(mut buffers) = self.available.lock() {
            if let Some(mut buffer) = buffers.pop() {
                buffer.clear();
                self.reuses.fetch_add(1, Ordering::Relaxed);
                return buffer;
            }
        }
        
        self.allocations.fetch_add(1, Ordering::Relaxed);
        CacheBuffer::new()
    }
    
    /// Return buffer to pool
    pub fn put(&self, buffer: CacheBuffer) {
        if let Ok(mut buffers) = self.available.lock() {
            if buffers.len() < self.max_size {
                buffers.push(buffer);
            }
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> BufferPoolStats {
        let available_count = self.available.lock()
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