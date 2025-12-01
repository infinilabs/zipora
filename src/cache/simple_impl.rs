//! Simplified cache implementation for testing

use super::*;
use crate::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Simple single-page cache implementation for testing
pub struct SingleLruPageCache {
    config: PageCacheConfig,
    stats: CacheStatistics,
    file_counter: std::sync::atomic::AtomicU32,
    // Simplified storage for testing
    data: Arc<Mutex<HashMap<(FileId, u64), Vec<u8>>>>,
    pub nodes: Vec<CacheNode>,
}

impl SingleLruPageCache {
    pub fn new(config: PageCacheConfig) -> Result<Self> {
        Ok(Self {
            config,
            stats: CacheStatistics::new(),
            file_counter: std::sync::atomic::AtomicU32::new(1),
            data: Arc::new(Mutex::new(HashMap::new())),
            nodes: Vec::new(),
        })
    }
    
    pub fn register_file(&self, _fd: i32) -> Result<FileId> {
        Ok(self.file_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
    
    pub fn read(&self, file_id: FileId, offset: u64, length: usize, buffer: &mut CacheBuffer) -> Result<()> {
        self.stats.record_bytes_read(length as u64);
        
        let key = (file_id, offset);
        let mut data = self.data.lock().unwrap_or_else(|e| e.into_inner());
        
        if let Some(cached_data) = data.get(&key) {
            self.stats.record_hit(CacheHitType::Hit);
            buffer.copy_from_slice(&cached_data[..std::cmp::min(length, cached_data.len())]);
        } else {
            self.stats.record_miss();
            // Simulate loading data
            let new_data = vec![0u8; length];
            data.insert(key, new_data.clone());
            buffer.copy_from_slice(&new_data);
            self.stats.record_hit(CacheHitType::InitialFree);
        }
        
        Ok(())
    }
    
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }
    
    pub fn size(&self) -> usize {
        self.data.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
    
    pub fn stats(&self) -> &CacheStatistics {
        &self.stats
    }
}

/// Simple cache node for testing
#[repr(align(64))]
pub struct CacheNode {
    file_id: std::sync::atomic::AtomicU32,
    page_id: std::sync::atomic::AtomicU32,
    ref_count: std::sync::atomic::AtomicU32,
    data: [u8; PAGE_SIZE],
}

impl CacheNode {
    pub fn new() -> Self {
        Self {
            file_id: std::sync::atomic::AtomicU32::new(0),
            page_id: std::sync::atomic::AtomicU32::new(0),
            ref_count: std::sync::atomic::AtomicU32::new(0),
            data: [0; PAGE_SIZE],
        }
    }
    
    pub fn page_data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
    
    pub fn dec_ref(&self) -> u32 {
        self.ref_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed)
    }
}

/// Simple LRU list for testing
pub struct LruList;

impl LruList {
    pub fn new() -> Self {
        Self
    }
    
    pub fn insert_head(&self, _nodes: &[CacheNode], _node_idx: NodeIndex) {
        // Simplified - no actual LRU management
    }
}