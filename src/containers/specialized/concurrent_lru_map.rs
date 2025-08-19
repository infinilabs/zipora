//! Concurrent LRU Cache Map with Sharding
//!
//! This module provides a high-performance, thread-safe LRU cache map that uses
//! sharding to reduce contention in multi-threaded environments.

use super::lru_map::{LruMap, LruMapConfig, LruMapStatistics, EvictionCallback, NoOpEvictionCallback};
use crate::error::{Result, ZiporaError};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::hash_map::DefaultHasher;

/// Configuration for concurrent LRU map
#[derive(Debug, Clone)]
pub struct ConcurrentLruMapConfig {
    /// Base configuration for individual shards
    pub base_config: LruMapConfig,
    
    /// Number of shards (should be power of 2)
    pub shard_count: usize,
    
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for ConcurrentLruMapConfig {
    fn default() -> Self {
        Self {
            base_config: LruMapConfig::default(),
            shard_count: 16, // Good default for most workloads
            load_balancing: LoadBalancingStrategy::Hash,
        }
    }
}

impl ConcurrentLruMapConfig {
    /// Create a performance-optimized configuration
    pub fn performance_optimized() -> Self {
        Self {
            base_config: LruMapConfig::performance_optimized(),
            shard_count: num_cpus::get() * 2, // 2 shards per CPU core
            load_balancing: LoadBalancingStrategy::Hash,
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            base_config: LruMapConfig::memory_optimized(),
            shard_count: 4, // Fewer shards to save memory
            load_balancing: LoadBalancingStrategy::Hash,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.base_config.validate()?;
        
        if self.shard_count == 0 {
            return Err(ZiporaError::invalid_parameter("Shard count must be > 0"));
        }
        
        if !self.shard_count.is_power_of_two() {
            return Err(ZiporaError::invalid_parameter("Shard count must be power of 2"));
        }
        
        Ok(())
    }
}

/// Load balancing strategies for shard selection
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Use hash-based sharding (default)
    Hash,
    /// Round-robin assignment (good for sequential keys)
    RoundRobin,
    /// Use thread ID for affinity-based sharding
    ThreadAffinity,
}

/// Aggregated statistics from all shards
#[derive(Debug, Default)]
pub struct ConcurrentLruMapStatistics {
    /// Statistics from individual shards
    pub shard_stats: Vec<Arc<LruMapStatistics>>,
    
    /// Global operation counter for round-robin
    pub global_counter: AtomicU64,
}

impl ConcurrentLruMapStatistics {
    /// Create new concurrent statistics
    pub fn new(shard_stats: Vec<Arc<LruMapStatistics>>) -> Self {
        Self {
            shard_stats,
            global_counter: AtomicU64::new(0),
        }
    }
    
    /// Get aggregated hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let (total_hits, total_gets) = self.shard_stats.iter().fold((0u64, 0u64), |(hits, gets), stats| {
            (
                hits + stats.hit_count.load(Ordering::Relaxed),
                gets + stats.get_count.load(Ordering::Relaxed),
            )
        });
        
        if total_gets > 0 {
            total_hits as f64 / total_gets as f64
        } else {
            0.0
        }
    }
    
    /// Get total entry count across all shards
    pub fn total_entries(&self) -> usize {
        self.shard_stats.iter().map(|stats| stats.entry_count.load(Ordering::Relaxed)).sum()
    }
    
    /// Get total memory usage across all shards
    pub fn total_memory_usage(&self) -> usize {
        self.shard_stats.iter().map(|stats| stats.memory_usage.load(Ordering::Relaxed)).sum()
    }
    
    /// Get shard with minimum load
    pub fn min_load_shard(&self) -> usize {
        self.shard_stats
            .iter()
            .enumerate()
            .min_by_key(|(_, stats)| stats.entry_count.load(Ordering::Relaxed))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    /// Get shard with maximum load
    pub fn max_load_shard(&self) -> usize {
        self.shard_stats
            .iter()
            .enumerate()
            .max_by_key(|(_, stats)| stats.entry_count.load(Ordering::Relaxed))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    /// Get load balance ratio (perfect balance = 1.0, higher = more imbalanced)
    pub fn load_balance_ratio(&self) -> f64 {
        if self.shard_stats.is_empty() {
            return 1.0;
        }
        
        let loads: Vec<usize> = self.shard_stats
            .iter()
            .map(|stats| stats.entry_count.load(Ordering::Relaxed))
            .collect();
        
        let max_load = *loads.iter().max().unwrap_or(&0);
        let min_load = *loads.iter().min().unwrap_or(&0);
        
        if min_load == 0 {
            if max_load == 0 { 1.0 } else { f64::INFINITY }
        } else {
            max_load as f64 / min_load as f64
        }
    }
}

/// High-performance concurrent LRU cache map with sharding
///
/// This implementation distributes entries across multiple LRU map shards
/// to reduce lock contention and improve scalability in multi-threaded environments.
///
/// # Examples
///
/// ```rust
/// use zipora::containers::specialized::ConcurrentLruMap;
/// 
/// let cache = ConcurrentLruMap::new(1024, 8).unwrap(); // 1024 capacity, 8 shards
/// 
/// cache.put("key1".to_string(), "value1".to_string()).unwrap();
/// cache.put("key2".to_string(), "value2".to_string()).unwrap();
/// 
/// assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
/// assert_eq!(cache.len(), 2);
/// ```
pub struct ConcurrentLruMap<K, V, E = NoOpEvictionCallback>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
    E: EvictionCallback<K, V> + Send + Sync + Clone,
{
    /// Individual LRU map shards
    shards: Vec<Arc<LruMap<K, V, E>>>,
    
    /// Configuration
    config: ConcurrentLruMapConfig,
    
    /// Shard selection mask for efficient modulo
    shard_mask: usize,
    
    /// Aggregated statistics
    stats: ConcurrentLruMapStatistics,
}

impl<K, V> ConcurrentLruMap<K, V, NoOpEvictionCallback>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
{
    /// Create a new concurrent LRU map
    pub fn new(total_capacity: usize, shard_count: usize) -> Result<Self> {
        let config = ConcurrentLruMapConfig {
            base_config: LruMapConfig {
                capacity: total_capacity / shard_count,
                ..Default::default()
            },
            shard_count,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Create with configuration
    pub fn with_config(config: ConcurrentLruMapConfig) -> Result<Self> {
        config.validate()?;
        
        let mut shards = Vec::with_capacity(config.shard_count);
        let mut shard_stats = Vec::with_capacity(config.shard_count);
        
        for _ in 0..config.shard_count {
            let shard = Arc::new(LruMap::with_config(config.base_config.clone())?);
            shard_stats.push(shard.stats_arc());
            shards.push(shard);
        }
        
        Ok(Self {
            shards,
            shard_mask: config.shard_count - 1,
            config,
            stats: ConcurrentLruMapStatistics::new(shard_stats),
        })
    }
}

impl<K, V, E> ConcurrentLruMap<K, V, E>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
    E: EvictionCallback<K, V> + Send + Sync + Clone,
{
    /// Create with eviction callback
    pub fn with_eviction_callback(total_capacity: usize, shard_count: usize, callback: E) -> Result<Self> {
        let config = ConcurrentLruMapConfig {
            base_config: LruMapConfig {
                capacity: total_capacity / shard_count,
                ..Default::default()
            },
            shard_count,
            ..Default::default()
        };
        
        Self::with_config_and_callback(config, callback)
    }
    
    /// Create with configuration and callback
    pub fn with_config_and_callback(config: ConcurrentLruMapConfig, callback: E) -> Result<Self> {
        config.validate()?;
        
        let mut shards = Vec::with_capacity(config.shard_count);
        let mut shard_stats = Vec::with_capacity(config.shard_count);
        
        for _ in 0..config.shard_count {
            let shard = Arc::new(LruMap::with_config_and_callback(
                config.base_config.clone(),
                callback.clone(),
            )?);
            shard_stats.push(shard.stats_arc());
            shards.push(shard);
        }
        
        Ok(Self {
            shards,
            shard_mask: config.shard_count - 1,
            config,
            stats: ConcurrentLruMapStatistics::new(shard_stats),
        })
    }
    
    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let shard_idx = self.select_shard(key);
        self.shards[shard_idx].get(key)
    }
    
    /// Insert or update a key-value pair
    pub fn put(&self, key: K, value: V) -> Result<Option<V>> {
        let shard_idx = self.select_shard(&key);
        self.shards[shard_idx].put(key, value)
    }
    
    /// Remove a key-value pair
    pub fn remove(&self, key: &K) -> Option<V> {
        let shard_idx = self.select_shard(key);
        self.shards[shard_idx].remove(key)
    }
    
    /// Check if the cache contains a key
    pub fn contains_key(&self, key: &K) -> bool {
        let shard_idx = self.select_shard(key);
        self.shards[shard_idx].contains_key(key)
    }
    
    /// Get the current number of entries across all shards
    pub fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.len()).sum()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get total capacity across all shards
    pub fn capacity(&self) -> usize {
        self.shards.iter().map(|shard| shard.capacity()).sum()
    }
    
    /// Clear all entries in all shards
    pub fn clear(&self) -> Result<()> {
        for shard in &self.shards {
            shard.clear()?;
        }
        Ok(())
    }
    
    /// Get aggregated statistics
    pub fn stats(&self) -> &ConcurrentLruMapStatistics {
        &self.stats
    }
    
    /// Get configuration
    pub fn config(&self) -> &ConcurrentLruMapConfig {
        &self.config
    }
    
    /// Get number of shards
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
    
    /// Get statistics for a specific shard
    pub fn shard_stats(&self, shard_idx: usize) -> Option<&LruMapStatistics> {
        self.shards.get(shard_idx).map(|shard| shard.stats())
    }
    
    /// Select shard for a given key
    fn select_shard(&self, key: &K) -> usize {
        match self.config.load_balancing {
            LoadBalancingStrategy::Hash => {
                self.hash_key(key) & self.shard_mask
            }
            LoadBalancingStrategy::RoundRobin => {
                let counter = self.stats.global_counter.fetch_add(1, Ordering::Relaxed);
                (counter as usize) & self.shard_mask
            }
            LoadBalancingStrategy::ThreadAffinity => {
                // Use thread ID for affinity
                let thread_id = std::thread::current().id();
                let hash = self.hash_thread_id(thread_id);
                hash & self.shard_mask
            }
        }
    }
    
    /// Hash a key for shard selection
    fn hash_key(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Use upper bits for better distribution
        ((hash >> 32) ^ hash) as usize
    }
    
    /// Hash thread ID for affinity-based sharding
    fn hash_thread_id(&self, thread_id: std::thread::ThreadId) -> usize {
        let mut hasher = DefaultHasher::new();
        thread_id.hash(&mut hasher);
        hasher.finish() as usize
    }
    
    /// Execute a function on all shards in parallel
    pub fn for_each_shard<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(&LruMap<K, V, E>) -> Result<()> + Send + Clone + 'static,
        K: 'static,
        V: 'static,
        E: 'static,
    {
        use std::thread;
        
        let handles: Vec<_> = self.shards
            .iter()
            .enumerate()
            .map(|(idx, shard)| {
                let shard = shard.clone();
                let mut f = f.clone();
                thread::spawn(move || f(&shard))
            })
            .collect();
        
        for handle in handles {
            handle.join().map_err(|_| ZiporaError::out_of_memory(0))??;
        }
        
        Ok(())
    }
    
    /// Get keys from all shards (expensive operation)
    pub fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        // This is an expensive operation that requires coordination
        // In practice, you might want to implement this differently
        // or provide warnings about its cost
        
        let mut all_keys = Vec::new();
        
        // Note: This is not atomic across shards, so the snapshot
        // might not be perfectly consistent
        for shard in &self.shards {
            // We'd need to add a keys() method to LruMap for this to work
            // For now, this is a placeholder showing the interface
        }
        
        all_keys
    }
    
    /// Get an approximate count of entries in each shard
    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.len()).collect()
    }
    
    /// Rebalance load across shards (advanced operation)
    pub fn rebalance(&self) -> Result<()> {
        // This would be a complex operation that moves entries between
        // shards to achieve better load balance. Implementation would
        // depend on specific requirements and constraints.
        
        // For now, this is a placeholder
        Ok(())
    }
}

// Thread-safe implementation
unsafe impl<K, V, E> Send for ConcurrentLruMap<K, V, E>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
    E: EvictionCallback<K, V> + Send + Sync + Clone,
{}

unsafe impl<K, V, E> Sync for ConcurrentLruMap<K, V, E>
where
    K: Hash + Eq + Clone + Send + Sync + Default,
    V: Clone + Send + Sync + Default,
    E: EvictionCallback<K, V> + Send + Sync + Clone,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_concurrent_basic_operations() {
        let cache = ConcurrentLruMap::new(64, 4).unwrap();
        
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.len(), 2);
    }
    
    #[test]
    fn test_concurrent_access() {
        let cache = Arc::new(ConcurrentLruMap::new(1000, 8).unwrap());
        let num_threads = 4;
        let operations_per_thread = 100;
        
        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let cache = cache.clone();
                thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        let key = thread_id * operations_per_thread + i;
                        let value = format!("value_{}", key);
                        
                        cache.put(key, value.clone()).unwrap();
                        assert_eq!(cache.get(&key), Some(value));
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(cache.len(), num_threads * operations_per_thread);
    }
    
    #[test]
    fn test_shard_distribution() {
        let cache = ConcurrentLruMap::new(64, 4).unwrap();
        
        // Insert many items
        for i in 0..64 {
            cache.put(i, format!("value_{}", i)).unwrap();
        }
        
        let shard_sizes = cache.shard_sizes();
        
        // Check that items are distributed across shards
        assert_eq!(shard_sizes.len(), 4);
        assert!(shard_sizes.iter().all(|&size| size > 0));
        
        // Total should equal cache size
        let total: usize = shard_sizes.iter().sum();
        assert_eq!(total, cache.len());
    }
    
    #[test]
    fn test_statistics() {
        let cache = ConcurrentLruMap::new(32, 2).unwrap();
        
        // Perform operations
        cache.put(1, "one".to_string()).unwrap();
        cache.put(2, "two".to_string()).unwrap();
        
        cache.get(&1); // Hit
        cache.get(&3); // Miss
        
        let stats = cache.stats();
        assert!(stats.hit_ratio() > 0.0);
        assert_eq!(stats.total_entries(), 2);
    }
    
    #[test]
    fn test_load_balancing_strategies() {
        // Test different load balancing strategies
        let configs = vec![
            ConcurrentLruMapConfig {
                base_config: LruMapConfig { capacity: 16, ..Default::default() },
                shard_count: 4,
                load_balancing: LoadBalancingStrategy::Hash,
            },
            ConcurrentLruMapConfig {
                base_config: LruMapConfig { capacity: 16, ..Default::default() },
                shard_count: 4,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
            ConcurrentLruMapConfig {
                base_config: LruMapConfig { capacity: 16, ..Default::default() },
                shard_count: 4,
                load_balancing: LoadBalancingStrategy::ThreadAffinity,
            },
        ];
        
        for config in configs {
            let cache = ConcurrentLruMap::with_config(config).unwrap();
            
            // Insert items and verify distribution
            for i in 0..16 {
                cache.put(i, format!("value_{}", i)).unwrap();
            }
            
            assert_eq!(cache.len(), 16);
            
            // All items should be retrievable
            for i in 0..16 {
                assert!(cache.get(&i).is_some());
            }
        }
    }
}