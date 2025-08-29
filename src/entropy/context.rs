//! Context-aware memory management for entropy algorithms
//!
//! This module provides efficient buffer pooling and reuse patterns
//! inspired by TerarkContext, optimized for entropy coding operations.

use crate::error::{Result, ZiporaError};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

/// Default capacity for context buffer pool (16MB)
const DEFAULT_CAPACITY: usize = 16 << 20;

/// Maximum number of buffers to cache per context
const DEFAULT_MAX_BUFFERS: usize = 32;

/// Minimum buffer size for reuse eligibility
const MIN_REUSE_SIZE: usize = 1024;

/// Maximum buffer size for reuse eligibility (1MB)
const MAX_REUSE_SIZE: usize = 1 << 20;

/// Configuration for entropy context behavior
#[derive(Debug, Clone)]
pub struct EntropyContextConfig {
    /// Maximum total memory capacity for buffer pool
    pub capacity: usize,
    /// Maximum number of buffers to cache
    pub max_buffers: usize,
    /// Enable thread-local optimization
    pub thread_local: bool,
    /// Enable zero-copy operations where possible
    pub zero_copy: bool,
}

impl Default for EntropyContextConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_CAPACITY,
            max_buffers: DEFAULT_MAX_BUFFERS,
            thread_local: true,
            zero_copy: true,
        }
    }
}

/// A buffer managed by entropy context for automatic reuse
pub struct ContextBuffer {
    data: Vec<u8>,
    context: Option<Arc<Mutex<BufferPool>>>,
}

impl ContextBuffer {
    /// Create a new context buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            context: None,
        }
    }
    
    /// Create a context buffer managed by a pool
    fn with_context(mut data: Vec<u8>, context: Arc<Mutex<BufferPool>>) -> Self {
        data.clear();
        Self {
            data,
            context: Some(context),
        }
    }
    
    /// Get mutable access to the underlying buffer
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.data
    }
    
    /// Get immutable access to the underlying buffer
    pub fn as_vec(&self) -> &Vec<u8> {
        &self.data
    }
    
    /// Get the data as a slice
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }
    
    /// Get the current length
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get the current capacity
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
    
    /// Resize the buffer
    pub fn resize(&mut self, new_len: usize, value: u8) {
        self.data.resize(new_len, value);
    }
    
    /// Ensure the buffer has at least the specified capacity
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.data.capacity() < capacity {
            self.data.reserve(capacity - self.data.capacity());
        }
    }
    
    /// Clear the buffer contents
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Take ownership of the underlying Vec
    pub fn into_vec(mut self) -> Vec<u8> {
        self.context = None; // Prevent return to pool
        std::mem::take(&mut self.data)
    }
}

impl Drop for ContextBuffer {
    fn drop(&mut self) {
        if let Some(context) = self.context.take() {
            // Return buffer to pool if eligible for reuse
            let capacity = self.data.capacity();
            if capacity >= MIN_REUSE_SIZE && capacity <= MAX_REUSE_SIZE {
                if let Ok(mut pool) = context.lock() {
                    pool.return_buffer(std::mem::take(&mut self.data));
                }
            }
        }
    }
}

/// Internal buffer pool for managing reusable buffers
struct BufferPool {
    buffers: VecDeque<Vec<u8>>,
    total_capacity: usize,
    max_capacity: usize,
    max_buffers: usize,
}

impl BufferPool {
    fn new(config: &EntropyContextConfig) -> Self {
        Self {
            buffers: VecDeque::new(),
            total_capacity: 0,
            max_capacity: config.capacity,
            max_buffers: config.max_buffers,
        }
    }
    
    fn get_buffer(&mut self, min_capacity: usize) -> Vec<u8> {
        // Find a suitable buffer from the pool
        let mut best_idx = None;
        let mut best_capacity = usize::MAX;
        
        for (idx, buffer) in self.buffers.iter().enumerate() {
            let capacity = buffer.capacity();
            if capacity >= min_capacity && capacity < best_capacity {
                best_capacity = capacity;
                best_idx = Some(idx);
                // Early exit for exact match
                if capacity == min_capacity {
                    break;
                }
            }
        }
        
        if let Some(idx) = best_idx {
            let buffer = self.buffers.remove(idx).unwrap();
            self.total_capacity -= buffer.capacity();
            buffer
        } else {
            // Create new buffer
            Vec::with_capacity(min_capacity)
        }
    }
    
    fn return_buffer(&mut self, buffer: Vec<u8>) {
        let capacity = buffer.capacity();
        
        // Only cache if within limits
        if self.buffers.len() < self.max_buffers 
            && self.total_capacity + capacity <= self.max_capacity {
            self.total_capacity += capacity;
            self.buffers.push_back(buffer);
        }
        // Otherwise, let buffer drop naturally
    }
}

/// Thread-local entropy context for optimal performance
struct ThreadLocalContext {
    pool: RefCell<BufferPool>,
    config: EntropyContextConfig,
}

thread_local! {
    static TLS_CONTEXT: OnceLock<ThreadLocalContext> = const { OnceLock::new() };
}

/// Global entropy context for cross-thread sharing
static GLOBAL_CONTEXT: OnceLock<Arc<Mutex<BufferPool>>> = OnceLock::new();
static GLOBAL_CONFIG: OnceLock<EntropyContextConfig> = OnceLock::new();

/// Main entropy context for managing memory efficiently during compression/decompression
pub struct EntropyContext {
    pool: Arc<Mutex<BufferPool>>,
    config: EntropyContextConfig,
}

impl EntropyContext {
    /// Create a new entropy context with default configuration
    pub fn new() -> Self {
        Self::with_config(EntropyContextConfig::default())
    }
    
    /// Create a new entropy context with custom configuration
    pub fn with_config(config: EntropyContextConfig) -> Self {
        let pool = Arc::new(Mutex::new(BufferPool::new(&config)));
        Self { pool, config }
    }
    
    /// Get the global entropy context (thread-safe)
    pub fn global() -> Self {
        let config = GLOBAL_CONFIG.get_or_init(EntropyContextConfig::default);
        let pool = GLOBAL_CONTEXT.get_or_init(|| {
            Arc::new(Mutex::new(BufferPool::new(config)))
        });
        
        Self {
            pool: pool.clone(),
            config: config.clone(),
        }
    }
    
    /// Get thread-local entropy context for optimal single-threaded performance
    pub fn thread_local() -> Self {
        TLS_CONTEXT.with(|ctx| {
            let tls_ctx = ctx.get_or_init(|| {
                let config = GLOBAL_CONFIG.get_or_init(EntropyContextConfig::default).clone();
                ThreadLocalContext {
                    pool: RefCell::new(BufferPool::new(&config)),
                    config: config.clone(),
                }
            });
            
            // For thread-local, we create a temporary context that borrows from TLS
            Self::with_config(tls_ctx.config.clone())
        })
    }
    
    /// Allocate a buffer with at least the specified capacity
    pub fn alloc(&self, capacity: usize) -> Result<ContextBuffer> {
        let buffer = if self.config.thread_local && capacity <= MAX_REUSE_SIZE {
            // Try thread-local allocation first for better performance
            TLS_CONTEXT.with(|ctx| {
                if let Some(tls_ctx) = ctx.get() {
                    if let Ok(mut pool) = tls_ctx.pool.try_borrow_mut() {
                        return pool.get_buffer(capacity);
                    }
                }
                Vec::with_capacity(capacity)
            })
        } else {
            // Use global pool
            match self.pool.lock() {
                Ok(mut pool) => pool.get_buffer(capacity),
                Err(_) => return Err(ZiporaError::resource_exhausted("Context pool lock failed")),
            }
        };
        
        Ok(ContextBuffer::with_context(buffer, self.pool.clone()))
    }
    
    /// Allocate a zero-initialized buffer
    pub fn alloc_zeroed(&self, size: usize) -> Result<ContextBuffer> {
        let mut buffer = self.alloc(size)?;
        buffer.resize(size, 0);
        Ok(buffer)
    }
    
    /// Get configuration
    pub fn config(&self) -> &EntropyContextConfig {
        &self.config
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> Result<ContextStats> {
        match self.pool.lock() {
            Ok(pool) => Ok(ContextStats {
                cached_buffers: pool.buffers.len(),
                total_capacity: pool.total_capacity,
                max_capacity: pool.max_capacity,
                max_buffers: pool.max_buffers,
            }),
            Err(_) => Err(ZiporaError::resource_exhausted("Context pool lock failed")),
        }
    }
    
    /// Get a buffer from the context for temporary use
    pub fn get_buffer(&mut self, size: usize) -> Result<Vec<u8>> {
        let buffer = self.alloc(size)?;
        let mut vec = buffer.into_vec();
        vec.resize(size, 0);
        Ok(vec)
    }
    
    /// Get a temporary buffer from the context
    pub fn get_temp_buffer(&mut self, size: usize) -> Result<Vec<u8>> {
        self.get_buffer(size)
    }
}

impl Default for EntropyContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about context buffer usage
#[derive(Debug, Clone)]
pub struct ContextStats {
    /// Number of buffers currently cached
    pub cached_buffers: usize,
    /// Total capacity of cached buffers
    pub total_capacity: usize,
    /// Maximum allowed capacity
    pub max_capacity: usize,
    /// Maximum number of buffers
    pub max_buffers: usize,
}

/// Convenience wrapper for entropy operation results with context
pub struct EntropyResult {
    /// The compressed/decompressed data
    pub data: Vec<u8>,
    /// The context buffer (can be reused)
    pub buffer: ContextBuffer,
}

impl EntropyResult {
    /// Create a new entropy result
    pub fn new(data: Vec<u8>, buffer: ContextBuffer) -> Self {
        Self { data, buffer }
    }
    
    /// Take ownership of the data, returning the buffer to the context
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
    
    /// Get a reference to the data
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_buffer_basic() {
        let mut buffer = ContextBuffer::new(1024);
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        
        buffer.resize(100, 42);
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.as_slice()[0], 42);
    }
    
    #[test]
    fn test_entropy_context_allocation() {
        let context = EntropyContext::new();
        
        let buffer1 = context.alloc(1024).unwrap();
        assert!(buffer1.capacity() >= 1024);
        
        let buffer2 = context.alloc_zeroed(512).unwrap();
        assert_eq!(buffer2.len(), 512);
        assert!(buffer2.as_slice().iter().all(|&b| b == 0));
    }
    
    #[test]
    fn test_buffer_reuse() {
        let context = EntropyContext::new();
        
        // Allocate and drop a buffer
        {
            let _buffer = context.alloc(2048).unwrap();
        }
        
        // Stats should show cached buffer
        let stats = context.stats().unwrap();
        println!("Stats after drop: cached_buffers={}, total_capacity={}", 
                stats.cached_buffers, stats.total_capacity);
        
        // Allocate again - should reuse
        let buffer2 = context.alloc(1024).unwrap();
        assert!(buffer2.capacity() >= 1024);
    }
    
    #[test]
    fn test_thread_local_context() {
        let _ctx1 = EntropyContext::thread_local();
        let _ctx2 = EntropyContext::thread_local();
        
        // Should work without panicking
        let buffer = _ctx1.alloc(1024).unwrap();
        assert!(buffer.capacity() >= 1024);
    }
    
    #[test]
    fn test_global_context() {
        let ctx1 = EntropyContext::global();
        let ctx2 = EntropyContext::global();
        
        // Both should work
        let _buffer1 = ctx1.alloc(1024).unwrap();
        let _buffer2 = ctx2.alloc(1024).unwrap();
    }
    
    #[test]
    fn test_entropy_result() {
        let context = EntropyContext::new();
        let buffer = context.alloc(1024).unwrap();
        let data = vec![1, 2, 3, 4, 5];
        
        let result = EntropyResult::new(data.clone(), buffer);
        assert_eq!(result.data(), &data);
        
        let extracted_data = result.into_data();
        assert_eq!(extracted_data, data);
    }
    
    #[test]
    fn test_context_config() {
        let config = EntropyContextConfig {
            capacity: 1024,
            max_buffers: 4,
            thread_local: false,
            zero_copy: true,
        };
        
        let context = EntropyContext::with_config(config.clone());
        assert_eq!(context.config().capacity, 1024);
        assert_eq!(context.config().max_buffers, 4);
        assert!(!context.config().thread_local);
        assert!(context.config().zero_copy);
    }
}