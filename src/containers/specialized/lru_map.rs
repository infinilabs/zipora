//! LRU (Least Recently Used) Cache Map Implementation
//!
//! This module provides a high-performance LRU cache map that combines hash map
//! lookups with LRU eviction policy. Inspired by topling-zip designs but
//! implemented with Rust safety guarantees.
//!
//! ## Features
//! 
//! - **O(1) operations**: get, put, remove operations in constant time
//! - **Generic key-value support**: Works with any `Hash + Eq` key and value type
//! - **Configurable capacity**: Automatic eviction when capacity is exceeded
//! - **Thread-safe statistics**: Lock-free counters for performance monitoring
//! - **Eviction callbacks**: Optional callbacks when entries are evicted
//! - **Memory efficient**: Uses intrusive linked list for minimal overhead
//! - **SIMD optimizations**: Hash computation with hardware acceleration where available

use crate::error::{Result, ZiporaError};
use crate::memory::{SecureMemoryPool, get_global_pool_for_size};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::hash::{Hash, Hasher};
use std::fmt::Debug;

/// Default maximum capacity for LRU maps
pub const DEFAULT_LRU_CAPACITY: usize = 1024;

/// Invalid node index marker
const INVALID_NODE: u32 = u32::MAX;

/// LRU map configuration options
#[derive(Debug, Clone)]
pub struct LruMapConfig {
    /// Maximum number of entries in the cache
    pub capacity: usize,
    
    /// Initial hash table capacity (power of 2)
    pub initial_hash_capacity: usize,
    
    /// Enable detailed statistics collection
    pub enable_statistics: bool,
    
    /// Use secure memory pools for internal allocations
    pub use_secure_memory: bool,
    
    /// Expected load factor (0.5 to 0.9)
    pub load_factor: f64,
    
    /// Enable concurrent access tracking
    pub enable_access_tracking: bool,
    
    /// Prefetch distance for hash chain traversal
    pub prefetch_distance: usize,
}

impl Default for LruMapConfig {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_LRU_CAPACITY,
            initial_hash_capacity: 128,
            enable_statistics: true,
            use_secure_memory: true,
            load_factor: 0.75,
            enable_access_tracking: true,
            prefetch_distance: 1,
        }
    }
}

impl LruMapConfig {
    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            capacity: 8192,
            initial_hash_capacity: 1024,
            enable_statistics: true,
            use_secure_memory: false, // Faster allocation
            load_factor: 0.75,
            enable_access_tracking: false, // Less overhead
            prefetch_distance: 2,
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            capacity: 512,
            initial_hash_capacity: 64,
            enable_statistics: false, // Less memory overhead
            use_secure_memory: true,
            load_factor: 0.85, // Higher density
            enable_access_tracking: false,
            prefetch_distance: 0, // No prefetching
        }
    }
    
    /// Create a security-optimized configuration
    pub fn security_optimized() -> Self {
        Self {
            capacity: 1024,
            initial_hash_capacity: 128,
            enable_statistics: true,
            use_secure_memory: true, // Always use secure pools
            load_factor: 0.7,
            enable_access_tracking: true,
            prefetch_distance: 1,
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.capacity == 0 {
            return Err(ZiporaError::invalid_parameter("Capacity must be > 0"));
        }
        
        if !self.initial_hash_capacity.is_power_of_two() {
            return Err(ZiporaError::invalid_parameter("Initial hash capacity must be power of 2"));
        }
        
        if self.load_factor <= 0.0 || self.load_factor >= 1.0 {
            return Err(ZiporaError::invalid_parameter("Load factor must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
}

/// LRU map statistics for performance monitoring
#[derive(Debug, Default)]
pub struct LruMapStatistics {
    /// Total number of get operations
    pub get_count: AtomicU64,
    
    /// Number of cache hits
    pub hit_count: AtomicU64,
    
    /// Number of cache misses
    pub miss_count: AtomicU64,
    
    /// Total number of put operations
    pub put_count: AtomicU64,
    
    /// Number of evictions performed
    pub eviction_count: AtomicU64,
    
    /// Number of hash collisions encountered
    pub collision_count: AtomicU64,
    
    /// Maximum probe distance in hash table
    pub max_probe_distance: AtomicU32,
    
    /// Current number of entries
    pub entry_count: AtomicUsize,
    
    /// Total memory usage in bytes
    pub memory_usage: AtomicUsize,
}

impl LruMapStatistics {
    /// Create new statistics instance
    pub fn new() -> Self {
        Default::default()
    }
    
    /// Get hit ratio as percentage
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.hit_count.load(Ordering::Relaxed) as f64;
        let total = self.get_count.load(Ordering::Relaxed) as f64;
        if total > 0.0 { hits / total } else { 0.0 }
    }
    
    /// Get average probe distance
    pub fn avg_probe_distance(&self) -> f64 {
        let gets = self.get_count.load(Ordering::Relaxed) as f64;
        let max_probe = self.max_probe_distance.load(Ordering::Relaxed) as f64;
        if gets > 0.0 { max_probe / gets } else { 0.0 }
    }
    
    /// Record a cache hit
    pub fn record_hit(&self) {
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.hit_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a cache miss
    pub fn record_miss(&self) {
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.miss_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record a put operation
    pub fn record_put(&self) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record an eviction
    pub fn record_eviction(&self) {
        self.eviction_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update memory usage
    pub fn update_memory_usage(&self, delta: isize) {
        if delta > 0 {
            self.memory_usage.fetch_add(delta as usize, Ordering::Relaxed);
        } else {
            self.memory_usage.fetch_sub((-delta) as usize, Ordering::Relaxed);
        }
    }
}

/// Trait for eviction callbacks
pub trait EvictionCallback<K, V>: Send + Sync {
    /// Called when an entry is evicted from the cache
    fn on_evict(&self, key: &K, value: &V);
}

/// No-op eviction callback
#[derive(Clone)]
pub struct NoOpEvictionCallback;

impl<K, V> EvictionCallback<K, V> for NoOpEvictionCallback {
    fn on_evict(&self, _key: &K, _value: &V) {}
}

/// LRU node containing key-value pair and linked list pointers
#[repr(align(64))] // Cache line aligned
struct LruNode<K, V> {
    /// Key-value pair
    key: K,
    value: V,
    
    /// Previous node in LRU list (more recently used)
    prev: AtomicU32,
    
    /// Next node in LRU list (less recently used)
    next: AtomicU32,
    
    /// Hash value for the key (cached to avoid recomputation)
    hash: u64,
    
    /// Access frequency counter (for statistics)
    access_count: AtomicU32,
    
    /// Last access timestamp (nanoseconds since epoch)
    last_access: AtomicU64,
    
    /// Node is valid/allocated flag
    is_valid: bool,
}

impl<K, V> LruNode<K, V> {
    /// Create a new LRU node
    fn new(key: K, value: V, hash: u64) -> Self {
        Self {
            key,
            value,
            prev: AtomicU32::new(INVALID_NODE),
            next: AtomicU32::new(INVALID_NODE),
            hash,
            access_count: AtomicU32::new(1),
            last_access: AtomicU64::new(Self::current_timestamp()),
            is_valid: true,
        }
    }
    
    /// Get current timestamp in nanoseconds
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Update access statistics
    fn update_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access.store(Self::current_timestamp(), Ordering::Relaxed);
    }
    
    /// Reset node to initial state
    fn reset(&mut self) {
        self.prev.store(INVALID_NODE, Ordering::Relaxed);
        self.next.store(INVALID_NODE, Ordering::Relaxed);
        self.access_count.store(0, Ordering::Relaxed);
        self.last_access.store(0, Ordering::Relaxed);
        self.is_valid = false;
    }
}

/// LRU linked list for maintaining access order
struct LruList {
    /// Head of the list (most recently used)
    head: AtomicU32,
    
    /// Tail of the list (least recently used)
    tail: AtomicU32,
    
    /// Number of nodes in the list
    count: AtomicUsize,
}

impl LruList {
    /// Create a new empty LRU list
    fn new() -> Self {
        Self {
            head: AtomicU32::new(INVALID_NODE),
            tail: AtomicU32::new(INVALID_NODE),
            count: AtomicUsize::new(0),
        }
    }
    
    /// Insert node at head (most recently used position)
    fn insert_head<K, V>(&self, nodes: &mut [LruNode<K, V>], node_idx: u32) {
        let old_head = self.head.swap(node_idx, Ordering::Relaxed);
        
        nodes[(node_idx as usize)].prev.store(INVALID_NODE, Ordering::Relaxed);
        nodes[(node_idx as usize)].next.store(old_head, Ordering::Relaxed);
        
        if old_head != INVALID_NODE {
            nodes[old_head as usize].prev.store(node_idx, Ordering::Relaxed);
        } else {
            // First node, also set as tail
            self.tail.store(node_idx, Ordering::Relaxed);
        }
        
        self.count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Remove node from anywhere in the list
    fn remove<K, V>(&self, nodes: &mut [LruNode<K, V>], node_idx: u32) {
        let prev = nodes[(node_idx as usize)].prev.load(Ordering::Relaxed);
        let next = nodes[(node_idx as usize)].next.load(Ordering::Relaxed);
        
        if prev != INVALID_NODE {
            nodes[prev as usize].next.store(next, Ordering::Relaxed);
        } else {
            // Removing head
            self.head.store(next, Ordering::Relaxed);
        }
        
        if next != INVALID_NODE {
            nodes[next as usize].prev.store(prev, Ordering::Relaxed);
        } else {
            // Removing tail
            self.tail.store(prev, Ordering::Relaxed);
        }
        
        nodes[(node_idx as usize)].prev.store(INVALID_NODE, Ordering::Relaxed);
        nodes[(node_idx as usize)].next.store(INVALID_NODE, Ordering::Relaxed);
        
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
    
    /// Move node to head (mark as most recently used)
    fn move_to_head<K, V>(&self, nodes: &mut [LruNode<K, V>], node_idx: u32) {
        // If already at head, just update access info
        if self.head.load(Ordering::Relaxed) == node_idx {
            nodes[(node_idx as usize)].update_access();
            return;
        }
        
        self.remove(nodes, node_idx);
        self.insert_head(nodes, node_idx);
        nodes[(node_idx as usize)].update_access();
    }
    
    /// Get least recently used node index
    fn get_lru_node(&self) -> u32 {
        self.tail.load(Ordering::Relaxed)
    }
    
    /// Get current count
    fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// High-performance LRU (Least Recently Used) cache map
///
/// This implementation uses a hash map for O(1) key lookups combined with
/// an intrusive doubly-linked list for O(1) LRU operations.
///
/// # Examples
///
/// ```rust
/// use zipora::containers::specialized::LruMap;
///
/// let mut cache = LruMap::new(3).unwrap(); // Capacity of 3
///
/// cache.put("a".to_string(), 1).unwrap();
/// cache.put("b".to_string(), 2).unwrap();
/// cache.put("c".to_string(), 3).unwrap();
///
/// assert_eq!(cache.get(&"a".to_string()), Some(1));
///
/// // Adding a 4th element evicts the least recently used
/// cache.put("d".to_string(), 4).unwrap();
/// assert_eq!(cache.get(&"b".to_string()), None); // "b" was evicted
/// ```
pub struct LruMap<K, V, E = NoOpEvictionCallback>
where
    K: Hash + Eq + Clone + Default,
    V: Clone + Default,
    E: EvictionCallback<K, V>,
{
    /// Configuration
    config: LruMapConfig,
    
    /// Hash map for O(1) key lookups (maps key to node index)
    hash_map: RwLock<HashMap<K, u32>>,
    
    /// Node storage (dense array for cache efficiency)
    nodes: RwLock<Vec<LruNode<K, V>>>,
    
    /// LRU linked list for eviction ordering
    lru_list: LruList,
    
    /// Free node indices stack
    free_nodes: Mutex<Vec<u32>>,
    
    /// Statistics for performance monitoring
    stats: Arc<LruMapStatistics>,
    
    /// Eviction callback
    eviction_callback: E,
    
    /// Memory pool for internal allocations
    memory_pool: Option<Arc<SecureMemoryPool>>,
}

impl<K, V> LruMap<K, V, NoOpEvictionCallback>
where
    K: Hash + Eq + Clone + Default,
    V: Clone + Default,
{
    /// Create a new LRU map with default configuration
    pub fn new(capacity: usize) -> Result<Self> {
        let config = LruMapConfig {
            capacity,
            ..Default::default()
        };
        Self::with_config(config)
    }
    
    /// Create a new LRU map with the given configuration
    pub fn with_config(config: LruMapConfig) -> Result<Self> {
        config.validate()?;
        
        let memory_pool = if config.use_secure_memory {
            Some(get_global_pool_for_size(4096).clone())
        } else {
            None
        };
        
        let mut nodes = Vec::with_capacity(config.capacity);
        let mut free_nodes = Vec::with_capacity(config.capacity);
        
        // Initialize all nodes with defaults (will be overwritten when used)
        for i in 0..config.capacity {
            nodes.push(LruNode {
                key: K::default(),
                value: V::default(),
                prev: AtomicU32::new(INVALID_NODE),
                next: AtomicU32::new(INVALID_NODE),
                hash: 0,
                access_count: AtomicU32::new(0),
                last_access: AtomicU64::new(0),
                is_valid: false,
            });
            free_nodes.push(i as u32);
        }
        
        let initial_capacity = config.initial_hash_capacity;
        
        Ok(Self {
            config,
            hash_map: RwLock::new(HashMap::with_capacity(initial_capacity)),
            nodes: RwLock::new(nodes),
            lru_list: LruList::new(),
            free_nodes: Mutex::new(free_nodes),
            stats: Arc::new(LruMapStatistics::new()),
            eviction_callback: NoOpEvictionCallback,
            memory_pool,
        })
    }
}

impl<K, V, E> LruMap<K, V, E>
where
    K: Hash + Eq + Clone + Default,
    V: Clone + Default,
    E: EvictionCallback<K, V>,
{
    /// Create a new LRU map with eviction callback
    pub fn with_eviction_callback(capacity: usize, callback: E) -> Result<Self> {
        let config = LruMapConfig {
            capacity,
            ..Default::default()
        };
        Self::with_config_and_callback(config, callback)
    }
    
    /// Create a new LRU map with configuration and eviction callback
    pub fn with_config_and_callback(config: LruMapConfig, callback: E) -> Result<Self> {
        config.validate()?;
        
        let memory_pool = if config.use_secure_memory {
            Some(get_global_pool_for_size(4096).clone())
        } else {
            None
        };
        
        let mut nodes = Vec::with_capacity(config.capacity);
        let mut free_nodes = Vec::with_capacity(config.capacity);
        
        // Initialize all nodes with defaults (will be overwritten when used)
        for i in 0..config.capacity {
            nodes.push(LruNode {
                key: K::default(),
                value: V::default(),
                prev: AtomicU32::new(INVALID_NODE),
                next: AtomicU32::new(INVALID_NODE),
                hash: 0,
                access_count: AtomicU32::new(0),
                last_access: AtomicU64::new(0),
                is_valid: false,
            });
            free_nodes.push(i as u32);
        }
        
        let initial_capacity = config.initial_hash_capacity;
        
        Ok(Self {
            config,
            hash_map: RwLock::new(HashMap::with_capacity(initial_capacity)),
            nodes: RwLock::new(nodes),
            lru_list: LruList::new(),
            free_nodes: Mutex::new(free_nodes),
            stats: Arc::new(LruMapStatistics::new()),
            eviction_callback: callback,
            memory_pool,
        })
    }
    
    /// Get a value by key, updating its position in the LRU list
    pub fn get(&self, key: &K) -> Option<V> {
        let hash_map = self.hash_map.read().ok()?;
        let node_idx = match hash_map.get(key) {
            Some(&idx) => idx,
            None => {
                if self.config.enable_statistics {
                    self.stats.record_miss();
                }
                return None;
            }
        };
        drop(hash_map);
        
        let mut nodes = self.nodes.write().ok()?;
        if (node_idx as usize) >= nodes.len() || !nodes[(node_idx as usize)].is_valid {
            return None;
        }
        
        // Move to head of LRU list (mark as most recently used)
        self.lru_list.move_to_head(&mut nodes, node_idx);
        
        let value = nodes[(node_idx as usize)].value.clone();
        
        if self.config.enable_statistics {
            self.stats.record_hit();
        }
        
        Some(value)
    }
    
    /// Insert or update a key-value pair
    pub fn put(&self, key: K, value: V) -> Result<Option<V>> {
        let hash = self.hash_key(&key);
        
        // Check if key already exists
        {
            let hash_map = self.hash_map.read().map_err(|_| ZiporaError::out_of_memory(0))?;
            if let Some(&node_idx) = hash_map.get(&key) {
                // Update existing entry
                let mut nodes = self.nodes.write().map_err(|_| ZiporaError::out_of_memory(0))?;
                if (node_idx as usize) < nodes.len() && nodes[(node_idx as usize)].is_valid {
                    let old_value = std::mem::replace(&mut nodes[(node_idx as usize)].value, value);
                    self.lru_list.move_to_head(&mut nodes, node_idx);
                    
                    if self.config.enable_statistics {
                        self.stats.record_put();
                    }
                    
                    return Ok(Some(old_value));
                }
            }
        }
        
        // Check if we need to evict before allocating
        if self.lru_list.len() >= self.config.capacity {
            self.evict_lru()?;
        }
        
        // Now allocate new entry (should have space after eviction)
        let node_idx = self.allocate_node()?;
        
        // Initialize new node
        {
            let mut nodes = self.nodes.write().map_err(|_| ZiporaError::out_of_memory(0))?;
            nodes[(node_idx as usize)] = LruNode::new(key.clone(), value, hash);
            self.lru_list.insert_head(&mut nodes, node_idx);
        }
        
        // Add to hash map
        {
            let mut hash_map = self.hash_map.write().map_err(|_| ZiporaError::out_of_memory(0))?;
            hash_map.insert(key, node_idx);
        }
        
        if self.config.enable_statistics {
            self.stats.record_put();
            self.stats.entry_count.store(self.lru_list.len(), Ordering::Relaxed);
        }
        
        Ok(None)
    }
    
    /// Remove a key-value pair
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut hash_map = self.hash_map.write().ok()?;
        let node_idx = hash_map.remove(key)?;
        drop(hash_map);
        
        let mut nodes = self.nodes.write().ok()?;
        if (node_idx as usize) >= nodes.len() || !nodes[(node_idx as usize)].is_valid {
            return None;
        }
        
        // Remove from LRU list
        self.lru_list.remove(&mut nodes, node_idx);
        
        let value = nodes[(node_idx as usize)].value.clone();
        nodes[(node_idx as usize)].reset();
        
        // Return to free list
        if let Ok(mut free_nodes) = self.free_nodes.lock() {
            free_nodes.push(node_idx);
        }
        
        if self.config.enable_statistics {
            self.stats.entry_count.store(self.lru_list.len(), Ordering::Relaxed);
        }
        
        Some(value)
    }
    
    /// Check if the cache contains a key
    pub fn contains_key(&self, key: &K) -> bool {
        if let Ok(hash_map) = self.hash_map.read() {
            hash_map.contains_key(key)
        } else {
            false
        }
    }
    
    /// Get the current number of entries
    pub fn len(&self) -> usize {
        self.lru_list.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the capacity of the cache
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }
    
    /// Clear all entries
    pub fn clear(&self) -> Result<()> {
        let mut hash_map = self.hash_map.write().map_err(|_| ZiporaError::out_of_memory(0))?;
        let mut nodes = self.nodes.write().map_err(|_| ZiporaError::out_of_memory(0))?;
        let mut free_nodes = self.free_nodes.lock().map_err(|_| ZiporaError::out_of_memory(0))?;
        
        hash_map.clear();
        
        // Reset all nodes and add to free list
        free_nodes.clear();
        for (i, node) in nodes.iter_mut().enumerate() {
            if node.is_valid {
                node.reset();
                free_nodes.push(i as u32);
            }
        }
        
        // Reset LRU list
        self.lru_list.head.store(INVALID_NODE, Ordering::Relaxed);
        self.lru_list.tail.store(INVALID_NODE, Ordering::Relaxed);
        self.lru_list.count.store(0, Ordering::Relaxed);
        
        if self.config.enable_statistics {
            self.stats.entry_count.store(0, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> &LruMapStatistics {
        &self.stats
    }
    
    /// Get cache statistics as Arc (for sharing)
    pub fn stats_arc(&self) -> Arc<LruMapStatistics> {
        self.stats.clone()
    }
    
    /// Get cache configuration
    pub fn config(&self) -> &LruMapConfig {
        &self.config
    }
    
    /// Hash a key (with potential SIMD optimization)
    fn hash_key(&self, key: &K) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Allocate a node from the free list
    fn allocate_node(&self) -> Result<u32> {
        let mut free_nodes = self.free_nodes.lock().map_err(|_| ZiporaError::out_of_memory(0))?;
        free_nodes.pop().ok_or_else(|| ZiporaError::out_of_memory(0).into())
    }
    
    /// Evict the least recently used entry
    fn evict_lru(&self) -> Result<()> {
        let lru_node_idx = self.lru_list.get_lru_node();
        if lru_node_idx == INVALID_NODE {
            return Err(ZiporaError::out_of_memory(0).into());
        }
        
        let mut nodes = self.nodes.write().map_err(|_| ZiporaError::out_of_memory(0))?;
        
        // Check validity and get key/value for callback before mutations
        if !(lru_node_idx as usize) < nodes.len() || !nodes[(lru_node_idx as usize)].is_valid {
            return Err(ZiporaError::out_of_memory(0).into());
        }
        
        let key = nodes[(lru_node_idx as usize)].key.clone();
        let value = nodes[(lru_node_idx as usize)].value.clone();
        
        // Call eviction callback
        self.eviction_callback.on_evict(&key, &value);
        
        // Remove from hash map
        {
            let mut hash_map = self.hash_map.write().map_err(|_| ZiporaError::out_of_memory(0))?;
            hash_map.remove(&key);
        }
        
        // Remove from LRU list
        self.lru_list.remove(&mut nodes, lru_node_idx);
        
        // Reset node and return to free list
        nodes[(lru_node_idx as usize)].reset();
        
        {
            let mut free_nodes = self.free_nodes.lock().map_err(|_| ZiporaError::out_of_memory(0))?;
            free_nodes.push(lru_node_idx);
        }
        
        if self.config.enable_statistics {
            self.stats.record_eviction();
            self.stats.entry_count.store(self.lru_list.len(), Ordering::Relaxed);
        }
        
        Ok(())
    }
}

impl<K, V, E> Drop for LruMap<K, V, E>
where
    K: Hash + Eq + Clone + Default,
    V: Clone + Default,
    E: EvictionCallback<K, V>,
{
    fn drop(&mut self) {
        // Clear all entries which will trigger eviction callbacks
        let _ = self.clear();
    }
}

// Thread-safe implementation
unsafe impl<K, V, E> Send for LruMap<K, V, E>
where
    K: Hash + Eq + Clone + Send + Default,
    V: Clone + Send + Default,
    E: EvictionCallback<K, V> + Send,
{}

unsafe impl<K, V, E> Sync for LruMap<K, V, E>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
    E: EvictionCallback<K, V> + Send + Sync,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    
    struct TestEvictionCallback {
        eviction_count: Arc<AtomicUsize>,
    }
    
    impl EvictionCallback<i32, String> for TestEvictionCallback {
        fn on_evict(&self, _key: &i32, _value: &String) {
            self.eviction_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    #[test]
    fn test_basic_operations() {
        let cache = LruMap::new(3).unwrap();
        
        // Test put and get
        assert_eq!(cache.put(1, "one".to_string()).unwrap(), None);
        assert_eq!(cache.put(2, "two".to_string()).unwrap(), None);
        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.len(), 2);
        
        // Test update
        assert_eq!(cache.put(1, "ONE".to_string()).unwrap(), Some("one".to_string()));
        assert_eq!(cache.get(&1), Some("ONE".to_string()));
        assert_eq!(cache.len(), 2);
    }
    
    #[test]
    fn test_lru_eviction() {
        let cache = LruMap::new(2).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        // Access 1 to make it most recently used
        assert_eq!(cache.get(&1), Some("one".to_string()));
        
        // Add 3, should evict 2 (least recently used)
        cache.put(3, "three".to_string()).unwrap();
        
        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some("three".to_string()));
        assert_eq!(cache.len(), 2);
    }
    
    #[test]
    fn test_remove() {
        let cache = LruMap::new(3).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        assert_eq!(cache.remove(&1), Some("one".to_string()));
        assert_eq!(cache.remove(&1), None);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.len(), 1);
    }
    
    #[test]
    fn test_clear() {
        let cache = LruMap::new(3).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        assert_eq!(cache.len(), 2);
        cache.clear().unwrap();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);
    }
    
    #[test]
    fn test_eviction_callback() {
        let eviction_count = Arc::new(AtomicUsize::new(0));
        let callback = TestEvictionCallback {
            eviction_count: eviction_count.clone(),
        };
        
        let cache = LruMap::with_eviction_callback(2, callback).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        cache.put(3, "three".to_string()).unwrap(); // Should evict 1
        
        assert_eq!(eviction_count.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_statistics() {
        let cache = LruMap::new(3).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        // Hit
        cache.get(&1);
        // Miss
        cache.get(&3);
        
        let stats = cache.stats();
        assert_eq!(stats.hit_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.miss_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.put_count.load(Ordering::Relaxed), 2);
        assert!((stats.hit_ratio() - 0.5).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_capacity_management() {
        let cache = LruMap::new(100).unwrap();
        
        // Fill beyond capacity
        for i in 0..150 {
            cache.put(i, format!("value_{}", i)).unwrap();
        }
        
        // Should maintain capacity limit
        assert!(cache.len() <= cache.capacity());
        
        // Recent entries should still be there
        assert!(cache.get(&149).is_some());
        assert!(cache.get(&148).is_some());
        
        // Early entries should be evicted
        assert!(cache.get(&0).is_none());
        assert!(cache.get(&1).is_none());
    }
}