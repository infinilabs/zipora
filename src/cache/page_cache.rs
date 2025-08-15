//! Main page cache implementation with multi-shard architecture

use super::*;
use crate::error::Result;
use crate::memory::{SecureMemoryPool, SecurePooledPtr};
use std::sync::{Arc, RwLock, Mutex};
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicU16, AtomicU8, Ordering};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::fs::File;

/// Single-shard LRU page cache implementation
pub struct SingleLruPageCache {
    /// Configuration
    config: PageCacheConfig,
    
    /// Page memory pool
    page_memory: Arc<SecureMemoryPool>,
    
    /// Raw page memory buffer
    page_buffer: SecurePooledPtr,
    
    /// Cache nodes array
    nodes: Vec<CacheNode>,
    
    /// Hash table for fast lookups
    hash_table: HashTable,
    
    /// LRU list for eviction policy
    lru_list: LruList,
    
    /// File information mapping
    file_info: RwLock<HashMap<FileId, Arc<FileInfo>>>,
    
    /// Free node list
    free_list: AtomicU32,
    
    /// Cache statistics
    stats: CacheStatistics,
    
    /// Global mutex for structural changes
    global_mutex: Mutex<()>,
    
    /// Next available file ID
    next_file_id: AtomicU32,
}

impl SingleLruPageCache {
    /// Create new single-shard cache
    pub fn new(config: PageCacheConfig) -> Result<Self> {
        config.validate()?;
        
        let page_count = config.calculate_page_count();
        let hash_size = config.calculate_hash_table_size();
        
        // Create memory pool for page storage
        let pool_config = if config.memory.use_secure_pools {
            crate::memory::SecurePoolConfig::large_secure()
        } else {
            crate::memory::SecurePoolConfig::large_performance()
        };
        
        let page_memory = Arc::new(SecureMemoryPool::new(pool_config)?);
        
        // Allocate page buffer with proper alignment
        let total_memory = page_count * config.page_size;
        let page_buffer = page_memory.allocate_aligned(total_memory, config.memory.alignment)?;
        
        // Apply kernel memory advice
        if config.memory.kernel_advice.huge_pages && config.use_huge_pages {
            #[cfg(target_os = "linux")]
            unsafe {
                libc::madvise(
                    page_buffer.as_ptr() as *mut libc::c_void,
                    total_memory,
                    libc::MADV_HUGEPAGE,
                );
            }
        }
        
        if config.memory.kernel_advice.will_need {
            #[cfg(target_os = "linux")]
            unsafe {
                libc::madvise(
                    page_buffer.as_ptr() as *mut libc::c_void,
                    total_memory,
                    libc::MADV_WILLNEED,
                );
            }
        }
        
        // Initialize cache nodes
        let mut nodes = Vec::with_capacity(page_count + 1); // +1 for sentinel
        nodes.resize_with(page_count + 1, CacheNode::default);
        
        // Initialize free list (link all nodes)
        for i in 1..page_count {
            nodes[i].hash_link.store((i + 1) as u32, Ordering::Relaxed);
        }
        nodes[page_count].hash_link.store(INVALID_NODE, Ordering::Relaxed);
        
        // Create hash table
        let hash_table = HashTable::new(hash_size);
        
        Ok(Self {
            config,
            page_memory,
            page_buffer,
            nodes,
            hash_table,
            lru_list: LruList::new(),
            file_info: RwLock::new(HashMap::new()),
            free_list: AtomicU32::new(1), // Start from node 1 (0 is sentinel)
            stats: CacheStatistics::new(),
            global_mutex: Mutex::new(()),
            next_file_id: AtomicU32::new(1),
        })
    }
    
    /// Register a file for caching
    pub fn register_file(&self, fd: i32) -> Result<FileId> {
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        let file_info = Arc::new(FileInfo::new(fd));
        
        let mut files = self.file_info.write().map_err(|_| CacheError::AllocationFailed)?;
        files.insert(file_id, file_info);
        
        Ok(file_id)
    }
    
    /// Read data from cache or load from file
    pub fn read(&self, file_id: FileId, offset: u64, length: usize, buffer: &mut CacheBuffer) -> Result<&[u8]> {
        let page_offset = offset % PAGE_SIZE as u64;
        let start_page = offset / PAGE_SIZE as u64;
        let end_page = (offset + length as u64 - 1) / PAGE_SIZE as u64;
        
        if start_page == end_page {
            // Single page operation (hot path)
            self.read_single_page(file_id, start_page as PageId, page_offset as usize, length, buffer)
        } else {
            // Multi-page operation
            self.read_multi_page(file_id, offset, length, buffer)
        }
    }
    
    /// Read from single page (optimized hot path)
    fn read_single_page(&self, file_id: FileId, page_id: PageId, offset: usize, length: usize, buffer: &mut CacheBuffer) -> Result<&[u8]> {
        // Fast path: check if page is already in cache
        if let Some(node_idx) = self.hash_table.find(&self.nodes, file_id, page_id) {
            let node = &self.nodes[node_idx as usize];
            
            // Increment reference count
            let old_ref = node.inc_ref();
            
            // Wait for page to be loaded if necessary
            self.wait_for_page_loaded(node);
            
            // Update access statistics
            node.update_last_access();
            self.stats.record_hit(CacheHitType::Hit);
            
            // Move to head of LRU list if not referenced
            if old_ref == 0 {
                self.lru_list.remove(&self.nodes, node_idx);
            }
            
            // Get page data
            let page_ptr = node.page_data_ptr();
            let page_data = unsafe { std::slice::from_raw_parts(page_ptr, PAGE_SIZE) };
            
            // Set up buffer for cleanup
            buffer.set_node(self, node_idx);
            
            return Ok(&page_data[offset..offset + length]);
        }
        
        // Cache miss: allocate new page
        self.load_page(file_id, page_id, offset, length, buffer)
    }
    
    /// Read from multiple pages
    fn read_multi_page(&self, file_id: FileId, offset: u64, length: usize, buffer: &mut CacheBuffer) -> Result<&[u8]> {
        let start_page = (offset / PAGE_SIZE as u64) as PageId;
        let end_page = ((offset + length as u64 - 1) / PAGE_SIZE as u64) as PageId;
        let num_pages = (end_page - start_page + 1) as usize;
        
        // Prefetch multiple pages for better performance
        let mut page_nodes = Vec::with_capacity(num_pages);
        
        for page_id in start_page..=end_page {
            // Prefetch next page information
            if page_id < end_page {
                let bucket_idx = self.hash_table.hash_index(file_id, page_id + 1);
                if bucket_idx < self.hash_table.size() {
                    prefetch_hint(&self.hash_table.buckets[bucket_idx] as *const _ as *const u8);
                }
            }
            
            let node_idx = if let Some(idx) = self.hash_table.find(&self.nodes, file_id, page_id) {
                let node = &self.nodes[idx as usize];
                node.inc_ref();
                self.wait_for_page_loaded(node);
                node.update_last_access();
                idx
            } else {
                self.allocate_and_load_page(file_id, page_id)?
            };
            
            page_nodes.push(node_idx);
        }
        
        // Copy data from multiple pages into buffer
        buffer.setup_multi_page(self, page_nodes, offset, length);
        self.stats.record_hit(CacheHitType::Mix);
        
        Ok(buffer.data())
    }
    
    /// Allocate a cache page node
    fn allocate_page(&self) -> Result<NodeIndex> {
        // Try to get from free list first
        let current_free = self.free_list.load(Ordering::Relaxed);
        if current_free != INVALID_NODE {
            let next_free = self.nodes[current_free as usize].hash_link.load(Ordering::Relaxed);
            if self.free_list.compare_exchange_weak(
                current_free, next_free, Ordering::Relaxed, Ordering::Relaxed
            ).is_ok() {
                return Ok(current_free);
            }
        }
        
        // No free nodes, need to evict LRU
        self.evict_lru_page()
    }
    
    /// Evict least recently used page
    fn evict_lru_page(&self) -> Result<NodeIndex> {
        let _lock = self.global_mutex.lock().map_err(|_| CacheError::AllocationFailed)?;
        
        let lru_node = self.lru_list.get_lru_node();
        if lru_node == INVALID_NODE {
            return Err(CacheError::CacheFull.into());
        }
        
        let node = &self.nodes[lru_node as usize];
        
        // Ensure node is not referenced
        if node.ref_count() > 0 {
            return Err(CacheError::CacheFull.into());
        }
        
        // Remove from hash table and LRU list
        let file_id = node.file_id();
        let page_id = node.page_id();
        
        if file_id != u32::MAX && page_id != u32::MAX {
            self.hash_table.remove(&self.nodes, lru_node, file_id, page_id);
        }
        
        self.lru_list.remove(&self.nodes, lru_node);
        
        // Reset node
        node.reset();
        
        self.stats.record_hit(CacheHitType::EvictedOthers);
        Ok(lru_node)
    }
    
    /// Load page from file
    fn load_page(&self, file_id: FileId, page_id: PageId, offset: usize, length: usize, buffer: &mut CacheBuffer) -> Result<&[u8]> {
        let node_idx = self.allocate_page()?;
        self.allocate_and_load_page_with_node(file_id, page_id, node_idx)?;
        
        let node = &self.nodes[node_idx as usize];
        let page_ptr = node.page_data_ptr();
        let page_data = unsafe { std::slice::from_raw_parts(page_ptr, PAGE_SIZE) };
        
        buffer.set_node(self, node_idx);
        Ok(&page_data[offset..offset + length])
    }
    
    /// Allocate and load page
    fn allocate_and_load_page(&self, file_id: FileId, page_id: PageId) -> Result<NodeIndex> {
        let node_idx = self.allocate_page()?;
        self.allocate_and_load_page_with_node(file_id, page_id, node_idx)?;
        Ok(node_idx)
    }
    
    /// Load page data with specific node
    fn allocate_and_load_page_with_node(&self, file_id: FileId, page_id: PageId, node_idx: NodeIndex) -> Result<()> {
        let node = &self.nodes[node_idx as usize];
        
        // Calculate page memory offset
        let page_offset = (node_idx as usize - 1) * self.config.page_size;
        let page_ptr = unsafe { self.page_buffer.as_ptr().add(page_offset) };
        
        // Initialize node
        node.initialize(file_id, page_id, page_ptr as *mut u8);
        node.inc_ref(); // Reference for this operation
        
        // Insert into hash table
        self.hash_table.insert(&self.nodes, node_idx, file_id, page_id);
        
        // Load page data from file
        self.load_page_data(file_id, page_id, page_ptr as *mut u8)?;
        
        // Mark as loaded
        node.mark_loaded();
        
        Ok(())
    }
    
    /// Load page data from file system
    fn load_page_data(&self, file_id: FileId, page_id: PageId, page_ptr: *mut u8) -> Result<()> {
        let files = self.file_info.read().map_err(|_| CacheError::FileNotFound)?;
        let file_info = files.get(&file_id).ok_or(CacheError::FileNotFound)?;
        
        if file_info.is_closed() {
            return Err(CacheError::FileNotFound.into());
        }
        
        // Calculate file offset
        let file_offset = (page_id as u64) * (PAGE_SIZE as u64);
        
        // Read data from file (this would need actual file I/O implementation)
        // For now, we'll zero the page
        unsafe {
            std::ptr::write_bytes(page_ptr, 0, PAGE_SIZE);
        }
        
        // TODO: Implement actual file reading with async I/O
        // This would integrate with the existing blob store I/O system
        
        Ok(())
    }
    
    /// Wait for page to finish loading
    fn wait_for_page_loaded(&self, node: &CacheNode) {
        while !node.is_page_loaded() {
            std::hint::spin_loop();
            // TODO: Could use fiber yielding here for better async integration
        }
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> &CacheStatistics {
        &self.stats
    }
    
    /// Get configuration
    pub fn config(&self) -> &PageCacheConfig {
        &self.config
    }
}