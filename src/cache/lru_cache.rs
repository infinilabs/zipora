//! Core LRU cache data structures with thread-safe access

use super::*;
use crate::error::Result;
use crate::memory::SecureMemoryPool;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicU16, AtomicU8, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::ptr::NonNull;

/// Cache node representing a cached page with dual linked-list participation
#[repr(align(64))] // Cache line aligned for performance
#[derive(Debug)]
pub struct CacheNode {
    /// Combined file ID (32-bit) + page ID (32-bit) for efficient hashing
    pub fi_offset: AtomicU64,
    
    /// Previous node in file-specific list
    pub fi_prev: AtomicU32,
    
    /// Next node in file-specific list
    pub fi_next: AtomicU32,
    
    /// Previous node in LRU list
    pub lru_prev: AtomicU32,
    
    /// Next node in LRU list
    pub lru_next: AtomicU32,
    
    /// Reference count for thread safety
    pub ref_count: AtomicU16,
    
    /// Loading status flag (0 = not loaded, 1 = loaded)
    pub is_loaded: AtomicU8,
    
    /// Hash collision chain link
    pub hash_link: AtomicU32,
    
    /// Page data pointer (points into page memory pool)
    pub page_data: AtomicU64, // Store as u64 for atomic access
    
    /// Last access timestamp for LRU ordering
    pub last_access: AtomicU64,
    
    /// Access frequency counter for LFU hints
    pub access_count: AtomicU32,
    
    /// Padding to ensure cache line alignment
    _padding: [u8; 8],
}

impl CacheNode {
    /// Create a new uninitialized cache node
    pub fn new() -> Self {
        Self {
            fi_offset: AtomicU64::new(u64::MAX),
            fi_prev: AtomicU32::new(INVALID_NODE),
            fi_next: AtomicU32::new(INVALID_NODE),
            lru_prev: AtomicU32::new(INVALID_NODE),
            lru_next: AtomicU32::new(INVALID_NODE),
            ref_count: AtomicU16::new(0),
            is_loaded: AtomicU8::new(0),
            hash_link: AtomicU32::new(INVALID_NODE),
            page_data: AtomicU64::new(0),
            last_access: AtomicU64::new(0),
            access_count: AtomicU32::new(0),
            _padding: [0; 8],
        }
    }
    
    /// Initialize node with file ID and page ID
    pub fn initialize(&self, file_id: FileId, page_id: PageId, page_ptr: *mut u8) {
        let fi_offset = ((file_id as u64) << 32) | (page_id as u64);
        self.fi_offset.store(fi_offset, Ordering::Relaxed);
        self.page_data.store(page_ptr as u64, Ordering::Relaxed);
        self.is_loaded.store(0, Ordering::Relaxed);
        self.ref_count.store(0, Ordering::Relaxed);
        self.access_count.store(0, Ordering::Relaxed);
        self.update_last_access();
    }
    
    /// Reset node to uninitialized state
    pub fn reset(&self) {
        self.fi_offset.store(u64::MAX, Ordering::Relaxed);
        self.fi_prev.store(INVALID_NODE, Ordering::Relaxed);
        self.fi_next.store(INVALID_NODE, Ordering::Relaxed);
        self.lru_prev.store(INVALID_NODE, Ordering::Relaxed);
        self.lru_next.store(INVALID_NODE, Ordering::Relaxed);
        self.ref_count.store(0, Ordering::Relaxed);
        self.is_loaded.store(0, Ordering::Relaxed);
        self.hash_link.store(INVALID_NODE, Ordering::Relaxed);
        self.page_data.store(0, Ordering::Relaxed);
        self.last_access.store(0, Ordering::Relaxed);
        self.access_count.store(0, Ordering::Relaxed);
    }
    
    /// Get file ID from packed fi_offset
    pub fn file_id(&self) -> FileId {
        let fi_offset = self.fi_offset.load(Ordering::Relaxed);
        (fi_offset >> 32) as FileId
    }
    
    /// Get page ID from packed fi_offset
    pub fn page_id(&self) -> PageId {
        let fi_offset = self.fi_offset.load(Ordering::Relaxed);
        fi_offset as PageId
    }
    
    /// Get combined file-page key
    pub fn fi_offset_key(&self) -> u64 {
        self.fi_offset.load(Ordering::Relaxed)
    }
    
    /// Check if node matches file and page
    pub fn matches(&self, file_id: FileId, page_id: PageId) -> bool {
        let expected = ((file_id as u64) << 32) | (page_id as u64);
        self.fi_offset.load(Ordering::Relaxed) == expected
    }
    
    /// Get page data pointer
    pub fn page_data_ptr(&self) -> *mut u8 {
        self.page_data.load(Ordering::Relaxed) as *mut u8
    }
    
    /// Check if page is loaded
    pub fn is_page_loaded(&self) -> bool {
        self.is_loaded.load(Ordering::Relaxed) != 0
    }
    
    /// Mark page as loaded
    pub fn mark_loaded(&self) {
        self.is_loaded.store(1, Ordering::Release);
    }
    
    /// Increment reference count
    pub fn inc_ref(&self) -> u16 {
        self.ref_count.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Decrement reference count
    pub fn dec_ref(&self) -> u16 {
        self.ref_count.fetch_sub(1, Ordering::Relaxed) - 1
    }
    
    /// Get current reference count
    pub fn ref_count(&self) -> u16 {
        self.ref_count.load(Ordering::Relaxed)
    }
    
    /// Update last access timestamp
    pub fn update_last_access(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_access.store(now, Ordering::Relaxed);
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get last access timestamp
    pub fn last_access_time(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
    
    /// Get access count
    pub fn access_frequency(&self) -> u32 {
        self.access_count.load(Ordering::Relaxed)
    }
}

impl Default for CacheNode {
    fn default() -> Self {
        Self::new()
    }
}

/// LRU linked list operations for cache nodes
pub struct LruList {
    /// Head node index (most recently used)
    head: AtomicU32,
    
    /// Tail node index (least recently used)
    tail: AtomicU32,
    
    /// Number of nodes in list
    count: AtomicU32,
}

impl LruList {
    /// Create new empty LRU list
    pub fn new() -> Self {
        Self {
            head: AtomicU32::new(INVALID_NODE),
            tail: AtomicU32::new(INVALID_NODE),
            count: AtomicU32::new(0),
        }
    }
    
    /// Insert node at head (most recently used)
    pub fn insert_head(&self, nodes: &[CacheNode], node_idx: NodeIndex) {
        let old_head = self.head.load(Ordering::Relaxed);
        
        nodes[node_idx as usize].lru_prev.store(INVALID_NODE, Ordering::Relaxed);
        nodes[node_idx as usize].lru_next.store(old_head, Ordering::Relaxed);
        
        if old_head != INVALID_NODE {
            nodes[old_head as usize].lru_prev.store(node_idx, Ordering::Relaxed);
        } else {
            // First node, also set as tail
            self.tail.store(node_idx, Ordering::Relaxed);
        }
        
        self.head.store(node_idx, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Remove node from list
    pub fn remove(&self, nodes: &[CacheNode], node_idx: NodeIndex) {
        let prev = nodes[node_idx as usize].lru_prev.load(Ordering::Relaxed);
        let next = nodes[node_idx as usize].lru_next.load(Ordering::Relaxed);
        
        if prev != INVALID_NODE {
            nodes[prev as usize].lru_next.store(next, Ordering::Relaxed);
        } else {
            // Removing head
            self.head.store(next, Ordering::Relaxed);
        }
        
        if next != INVALID_NODE {
            nodes[next as usize].lru_prev.store(prev, Ordering::Relaxed);
        } else {
            // Removing tail
            self.tail.store(prev, Ordering::Relaxed);
        }
        
        nodes[node_idx as usize].lru_prev.store(INVALID_NODE, Ordering::Relaxed);
        nodes[node_idx as usize].lru_next.store(INVALID_NODE, Ordering::Relaxed);
        
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
    
    /// Move node to head (mark as most recently used)
    pub fn move_to_head(&self, nodes: &[CacheNode], node_idx: NodeIndex) {
        // Remove from current position
        self.remove(nodes, node_idx);
        // Insert at head
        self.insert_head(nodes, node_idx);
    }
    
    /// Get least recently used node (tail)
    pub fn get_lru_node(&self) -> NodeIndex {
        self.tail.load(Ordering::Relaxed)
    }
    
    /// Get most recently used node (head)
    pub fn get_mru_node(&self) -> NodeIndex {
        self.head.load(Ordering::Relaxed)
    }
    
    /// Get number of nodes in list
    pub fn count(&self) -> u32 {
        self.count.load(Ordering::Relaxed)
    }
    
    /// Check if list is empty
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

impl Default for LruList {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash table for fast page lookups
pub struct HashTable {
    /// Hash buckets (array of node indices)
    buckets: Vec<AtomicU32>,
    
    /// Hash table size (power of 2)
    size: usize,
    
    /// Mask for efficient modulo operation
    mask: u64,
    
    /// Collision statistics
    collisions: AtomicU64,
    
    /// Maximum probe distance encountered
    max_probe_distance: AtomicU32,
}

impl HashTable {
    /// Create new hash table with given size
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Hash table size must be power of 2");
        
        let mut buckets = Vec::with_capacity(size);
        buckets.resize_with(size, || AtomicU32::new(INVALID_NODE));
        
        Self {
            buckets,
            size,
            mask: (size - 1) as u64,
            collisions: AtomicU64::new(0),
            max_probe_distance: AtomicU32::new(0),
        }
    }
    
    /// Get hash bucket index for file-page combination
    #[inline]
    pub fn hash_index(&self, file_id: FileId, page_id: PageId) -> usize {
        let hash = hash_file_page(file_id, page_id);
        (hash & self.mask) as usize
    }
    
    /// Find node in hash table
    pub fn find(&self, nodes: &[CacheNode], file_id: FileId, page_id: PageId) -> Option<NodeIndex> {
        let bucket_idx = self.hash_index(file_id, page_id);
        let mut current = self.buckets[bucket_idx].load(Ordering::Relaxed);
        let mut probe_distance = 0;
        
        while current != INVALID_NODE {
            // Prefetch next node for better cache performance
            if current < nodes.len() as u32 {
                prefetch_hint(&nodes[current as usize] as *const _ as *const u8);
            }
            
            if nodes[current as usize].matches(file_id, page_id) {
                return Some(current);
            }
            
            current = nodes[current as usize].hash_link.load(Ordering::Relaxed);
            probe_distance += 1;
            
            // Update max probe distance statistics
            let current_max = self.max_probe_distance.load(Ordering::Relaxed);
            if probe_distance > current_max {
                let _ = self.max_probe_distance.compare_exchange_weak(
                    current_max, probe_distance, Ordering::Relaxed, Ordering::Relaxed
                );
            }
        }
        
        None
    }
    
    /// Insert node into hash table
    pub fn insert(&self, nodes: &[CacheNode], node_idx: NodeIndex, file_id: FileId, page_id: PageId) {
        let bucket_idx = self.hash_index(file_id, page_id);
        let old_head = self.buckets[bucket_idx].load(Ordering::Relaxed);
        
        // Link new node to old chain
        nodes[node_idx as usize].hash_link.store(old_head, Ordering::Relaxed);
        
        // Update bucket to point to new node
        self.buckets[bucket_idx].store(node_idx, Ordering::Relaxed);
        
        // Update collision statistics
        if old_head != INVALID_NODE {
            self.collisions.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    /// Remove node from hash table
    pub fn remove(&self, nodes: &[CacheNode], node_idx: NodeIndex, file_id: FileId, page_id: PageId) {
        let bucket_idx = self.hash_index(file_id, page_id);
        let current = self.buckets[bucket_idx].load(Ordering::Relaxed);
        
        if current == node_idx {
            // Removing head of chain
            let next = nodes[node_idx as usize].hash_link.load(Ordering::Relaxed);
            self.buckets[bucket_idx].store(next, Ordering::Relaxed);
        } else {
            // Find and remove from middle of chain
            let mut prev = current;
            while prev != INVALID_NODE {
                let next = nodes[prev as usize].hash_link.load(Ordering::Relaxed);
                if next == node_idx {
                    let next_next = nodes[node_idx as usize].hash_link.load(Ordering::Relaxed);
                    nodes[prev as usize].hash_link.store(next_next, Ordering::Relaxed);
                    break;
                }
                prev = next;
            }
        }
        
        // Clear hash link
        nodes[node_idx as usize].hash_link.store(INVALID_NODE, Ordering::Relaxed);
    }
    
    /// Get collision count
    pub fn collision_count(&self) -> u64 {
        self.collisions.load(Ordering::Relaxed)
    }
    
    /// Get maximum probe distance
    pub fn max_probe_distance(&self) -> u32 {
        self.max_probe_distance.load(Ordering::Relaxed)
    }
    
    /// Get load factor
    pub fn load_factor(&self, total_nodes: usize) -> f64 {
        total_nodes as f64 / self.size as f64
    }
    
    /// Get hash table size
    pub fn size(&self) -> usize {
        self.size
    }
}

/// File management for cache operations
#[derive(Debug)]
pub struct FileInfo {
    /// File descriptor or handle
    pub fd: i32,
    
    /// Head of file-specific page list
    pub head_page: AtomicU32,
    
    /// File size in bytes
    pub file_size: AtomicU64,
    
    /// File status flags
    pub status: AtomicU32,
    
    /// Number of cached pages for this file
    pub cached_pages: AtomicU32,
}

impl FileInfo {
    pub fn new(fd: i32) -> Self {
        Self {
            fd,
            head_page: AtomicU32::new(INVALID_NODE),
            file_size: AtomicU64::new(0),
            status: AtomicU32::new(0),
            cached_pages: AtomicU32::new(0),
        }
    }
    
    pub fn is_open(&self) -> bool {
        self.fd >= 0
    }
    
    pub fn mark_closed(&self) {
        self.status.store(1, Ordering::Relaxed); // 1 = closed
    }
    
    pub fn is_closed(&self) -> bool {
        self.status.load(Ordering::Relaxed) != 0
    }
}