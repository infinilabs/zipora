//! Strategy Traits for Unified Hash Map Implementation
//!
//! This module defines the strategy traits that enable the unified ZiporaHashMap
//! to support all existing hash map variants through pluggable algorithms.
//!
//! # Strategy Architecture
//!
//! The strategy pattern allows different algorithms to be combined:
//! - **CollisionResolutionStrategy**: How to handle hash collisions
//! - **StorageLayoutStrategy**: How to organize data in memory
//! - **HashOptimizationStrategy**: Performance optimizations (SIMD, cache, etc.)
//!
//! This enables a single unified implementation to support all use cases that
//! previously required separate implementations.

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::hash_map::cache_locality::{CacheMetrics, Prefetcher};
use crate::hash_map::simd_string_ops::SimdStringOps;
use crate::memory::cache_layout::{CacheOptimizedAllocator, PrefetchHint};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};

/// Collision resolution strategy for hash maps
pub trait CollisionResolutionStrategy<K, V> {
    /// Configuration for this strategy
    type Config: Clone;

    /// Context/state maintained by this strategy
    type Context: Default;

    /// Insert a key-value pair using this collision resolution strategy
    fn insert(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: K,
        value: V,
        hash: u64,
        config: &Self::Config,
    ) -> Result<Option<V>>;

    /// Lookup a key using this collision resolution strategy
    fn get<'a, Q>(
        &self,
        context: &Self::Context,
        buckets: &'a [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Remove a key using this collision resolution strategy
    fn remove<Q>(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Check if resize is needed
    fn needs_resize(&self, context: &Self::Context, capacity: usize, len: usize) -> bool;

    /// Get probe statistics for this strategy
    fn probe_stats(&self, context: &Self::Context) -> ProbeStats;
}

/// Storage layout strategy for memory organization
pub trait StorageLayoutStrategy<K, V> {
    /// Configuration for this strategy
    type Config: Clone;

    /// Storage context/state
    type Storage;

    /// Initialize storage with given capacity
    fn create_storage(capacity: usize, config: &Self::Config) -> Self::Storage;

    /// Resize storage to new capacity
    fn resize_storage(
        storage: &mut Self::Storage,
        new_capacity: usize,
        config: &Self::Config,
    ) -> Result<()>;

    /// Get bucket array from storage
    fn get_buckets_mut(storage: &mut Self::Storage) -> &mut [HashBucket<K, V>];

    /// Get bucket array from storage (immutable)
    fn get_buckets(storage: &Self::Storage) -> &[HashBucket<K, V>];

    /// Get capacity of storage
    fn capacity(storage: &Self::Storage) -> usize;

    /// Estimate memory usage in bytes
    fn memory_usage(storage: &Self::Storage) -> usize;

    /// Perform layout-specific optimizations
    fn optimize_layout(storage: &mut Self::Storage, config: &Self::Config) -> Result<()>;
}

/// Hash optimization strategy for performance features
pub trait HashOptimizationStrategy<K, V> {
    /// Configuration for optimizations
    type Config: Clone;

    /// Optimization context/state
    type Context: Default;

    /// Pre-insert optimization (e.g., prefetching)
    fn pre_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    );

    /// Post-insert optimization (e.g., cache management)
    fn post_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        inserted: bool,
        config: &Self::Config,
    );

    /// Pre-lookup optimization
    fn pre_lookup<Q>(
        &self,
        context: &mut Self::Context,
        key: &Q,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    ) where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Bulk optimization for multiple operations
    fn bulk_optimize(
        &self,
        context: &mut Self::Context,
        operations: &[OptimizationHint],
        config: &Self::Config,
    );

    /// Get optimization metrics
    fn metrics(&self, context: &Self::Context) -> OptimizationMetrics;
}

/// Generic hash bucket for different strategies
#[repr(align(64))] // Cache line alignment
#[derive(Clone)]
pub struct HashBucket<K, V> {
    /// Hash value (cached for performance)
    pub hash: u64,
    /// Key
    pub key: Option<K>,
    /// Value
    pub value: Option<V>,
    /// Probe distance for Robin Hood hashing
    pub probe_distance: u16,
    /// Strategy-specific flags
    pub flags: u16,
    /// Next bucket index for chaining
    pub next: Option<u32>,
    /// Strategy-specific data
    pub strategy_data: u64,
}

impl<K, V> Default for HashBucket<K, V> {
    fn default() -> Self {
        Self {
            hash: 0,
            key: None,
            value: None,
            probe_distance: 0,
            flags: 0,
            next: None,
            strategy_data: 0,
        }
    }
}

impl<K, V> HashBucket<K, V> {
    /// Check if bucket is empty
    pub fn is_empty(&self) -> bool {
        self.key.is_none()
    }

    /// Check if bucket is deleted (tombstone)
    pub fn is_deleted(&self) -> bool {
        self.flags & 0x8000 != 0
    }

    /// Mark bucket as deleted
    pub fn mark_deleted(&mut self) {
        self.flags |= 0x8000;
        self.key = None;
        self.value = None;
    }

    /// Clear bucket
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Probe statistics for collision resolution
#[derive(Debug, Default, Clone)]
pub struct ProbeStats {
    pub average_probe_distance: f64,
    pub max_probe_distance: u16,
    pub total_probes: u64,
    pub collision_count: u64,
    pub variance: f64,
}

/// Optimization metrics
#[derive(Debug, Default, Clone)]
pub struct OptimizationMetrics {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_hits: u64,
    pub simd_operations: u64,
    pub bulk_operations: u64,
}

/// Optimization hints for bulk operations
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Sequential access pattern
    Sequential { start_hash: u64, count: usize },
    /// Random access pattern
    Random { hashes: Vec<u64> },
    /// Bulk insertion
    BulkInsert { count: usize },
    /// Bulk lookup
    BulkLookup { count: usize },
    /// Cache warming
    CacheWarm { bucket_range: std::ops::Range<usize> },
}

// Concrete strategy implementations

/// Robin Hood collision resolution strategy
pub struct RobinHoodStrategy {
    max_probe_distance: u16,
    variance_reduction: bool,
    backward_shift: bool,
}

impl RobinHoodStrategy {
    pub fn new(max_probe_distance: u16, variance_reduction: bool, backward_shift: bool) -> Self {
        Self {
            max_probe_distance,
            variance_reduction,
            backward_shift,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RobinHoodConfig {
    pub max_probe_distance: u16,
    pub variance_reduction: bool,
    pub backward_shift: bool,
}

#[derive(Debug, Default)]
pub struct RobinHoodContext {
    pub total_probe_distance: u64,
    pub max_probe_distance: u16,
    pub collision_count: u64,
    pub eviction_count: u64,
}

impl<K, V> CollisionResolutionStrategy<K, V> for RobinHoodStrategy
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    type Config = RobinHoodConfig;
    type Context = RobinHoodContext;

    fn insert(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: K,
        value: V,
        hash: u64,
        config: &Self::Config,
    ) -> Result<Option<V>> {
        let mask = buckets.len() - 1;
        let mut pos = (hash as usize) & mask;
        let mut probe_distance = 0;
        let mut inserting_key = Some(key);
        let mut inserting_value = Some(value);
        let mut inserting_hash = hash;
        let mut inserting_distance = 0;

        loop {
            if probe_distance > config.max_probe_distance {
                return Err(ZiporaError::invalid_state("Exceeded maximum probe distance"));
            }

            let bucket = &mut buckets[pos];

            // Empty bucket - insert here
            if bucket.is_empty() {
                bucket.hash = inserting_hash;
                bucket.key = inserting_key;
                bucket.value = inserting_value;
                bucket.probe_distance = inserting_distance;
                context.total_probe_distance += inserting_distance as u64;
                context.max_probe_distance = context.max_probe_distance.max(inserting_distance);
                return Ok(None);
            }

            // Key already exists - update value
            if bucket.hash == inserting_hash && bucket.key.as_ref() == inserting_key.as_ref() {
                let old_value = bucket.value.take();
                bucket.value = inserting_value;
                return Ok(old_value);
            }

            // Robin Hood: displace if we've traveled further
            if inserting_distance > bucket.probe_distance {
                // Swap the inserting entry with the bucket entry
                std::mem::swap(&mut bucket.hash, &mut inserting_hash);
                std::mem::swap(&mut bucket.key, &mut inserting_key);
                std::mem::swap(&mut bucket.value, &mut inserting_value);
                std::mem::swap(&mut bucket.probe_distance, &mut inserting_distance);
                context.eviction_count += 1;
            }

            pos = (pos + 1) & mask;
            probe_distance += 1;
            inserting_distance += 1;

            if inserting_distance > 0 {
                context.collision_count += 1;
            }
        }
    }

    fn get<'a, Q>(
        &self,
        context: &Self::Context,
        buckets: &'a [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mask = buckets.len() - 1;
        let mut pos = (hash as usize) & mask;
        let mut probe_distance = 0;

        while probe_distance <= config.max_probe_distance {
            let bucket = &buckets[pos];

            if bucket.is_empty() {
                return None;
            }

            if bucket.hash == hash && bucket.key.as_ref().map(|k| k.borrow()) == Some(key) {
                return bucket.value.as_ref();
            }

            // Robin Hood: if we've probed further than this bucket's distance,
            // the key doesn't exist
            if probe_distance > bucket.probe_distance {
                return None;
            }

            pos = (pos + 1) & mask;
            probe_distance += 1;
        }

        None
    }

    fn remove<Q>(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let mask = buckets.len() - 1;
        let mut pos = (hash as usize) & mask;
        let mut probe_distance = 0;

        // Find the key
        while probe_distance <= config.max_probe_distance {
            let bucket = &mut buckets[pos];

            if bucket.is_empty() {
                return None;
            }

            if bucket.hash == hash && bucket.key.as_ref().map(|k| k.borrow()) == Some(key) {
                let value = bucket.value.take();

                if config.backward_shift {
                    // Backward shift deletion
                    self.backward_shift_delete(buckets, pos);
                } else {
                    // Mark as deleted (tombstone)
                    bucket.mark_deleted();
                }

                return value;
            }

            if probe_distance > bucket.probe_distance {
                return None;
            }

            pos = (pos + 1) & mask;
            probe_distance += 1;
        }

        None
    }

    fn needs_resize(&self, context: &Self::Context, capacity: usize, len: usize) -> bool {
        let load_factor = len as f64 / capacity as f64;
        load_factor > 0.7 || context.max_probe_distance > self.max_probe_distance
    }

    fn probe_stats(&self, context: &Self::Context) -> ProbeStats {
        let avg_probe = if context.collision_count > 0 {
            context.total_probe_distance as f64 / context.collision_count as f64
        } else {
            0.0
        };

        ProbeStats {
            average_probe_distance: avg_probe,
            max_probe_distance: context.max_probe_distance,
            total_probes: context.total_probe_distance,
            collision_count: context.collision_count,
            variance: 0.0, // TODO: Calculate variance
        }
    }
}

impl RobinHoodStrategy {
    fn backward_shift_delete<K, V>(&self, buckets: &mut [HashBucket<K, V>], mut pos: usize)
    where
        K: Clone,
        V: Clone,
    {
        let mask = buckets.len() - 1;
        buckets[pos].clear();

        loop {
            let next_pos = (pos + 1) & mask;
            let next_bucket = &buckets[next_pos];

            // Stop if next bucket is empty or at its ideal position
            if next_bucket.is_empty() || next_bucket.probe_distance == 0 {
                break;
            }

            // Move the next bucket backward
            buckets[pos] = buckets[next_pos].clone();
            buckets[pos].probe_distance -= 1;
            buckets[next_pos].clear();

            pos = next_pos;
        }
    }
}

/// Standard storage layout strategy
pub struct StandardStorageStrategy;

#[derive(Debug, Clone)]
pub struct StandardStorageConfig {
    pub initial_capacity: usize,
    pub growth_factor: f64,
}

impl<K, V> StorageLayoutStrategy<K, V> for StandardStorageStrategy {
    type Config = StandardStorageConfig;
    type Storage = FastVec<HashBucket<K, V>>;

    fn create_storage(capacity: usize, config: &Self::Config) -> Self::Storage {
        // Graceful fallback: try full capacity, then half, then empty vec
        let mut storage = FastVec::with_capacity(capacity)
            .or_else(|_| FastVec::with_capacity(capacity / 2))
            .unwrap_or_else(|_| FastVec::new());

        // Only resize if we got some capacity
        if storage.capacity() > 0 {
            let actual_capacity = storage.capacity();
            let _ = storage.resize_with(actual_capacity, Default::default);
        }
        storage
    }

    fn resize_storage(
        storage: &mut Self::Storage,
        new_capacity: usize,
        config: &Self::Config,
    ) -> Result<()> {
        storage.resize_with(new_capacity, Default::default);
        Ok(())
    }

    fn get_buckets_mut(storage: &mut Self::Storage) -> &mut [HashBucket<K, V>] {
        storage.as_mut_slice()
    }

    fn get_buckets(storage: &Self::Storage) -> &[HashBucket<K, V>] {
        storage.as_slice()
    }

    fn capacity(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn memory_usage(storage: &Self::Storage) -> usize {
        storage.capacity() * std::mem::size_of::<HashBucket<K, V>>()
    }

    fn optimize_layout(storage: &mut Self::Storage, config: &Self::Config) -> Result<()> {
        // No special optimization for standard storage
        Ok(())
    }
}

/// Cache-optimized storage layout strategy
pub struct CacheOptimizedStorageStrategy {
    allocator: CacheOptimizedAllocator,
}

impl CacheOptimizedStorageStrategy {
    pub fn new(allocator: CacheOptimizedAllocator) -> Self {
        Self { allocator }
    }
}

#[derive(Debug, Clone)]
pub struct CacheOptimizedStorageConfig {
    pub cache_line_size: usize,
    pub numa_aware: bool,
    pub prefetch_enabled: bool,
}

// Note: Implementation would use cache-aligned allocation
// For simplicity, using FastVec here but real implementation would use cache allocator
impl<K, V> StorageLayoutStrategy<K, V> for CacheOptimizedStorageStrategy {
    type Config = CacheOptimizedStorageConfig;
    type Storage = FastVec<HashBucket<K, V>>;

    fn create_storage(capacity: usize, config: &Self::Config) -> Self::Storage {
        // TODO: Use cache-aligned allocation
        // Graceful fallback: try full capacity, then half, then empty vec
        let mut storage = FastVec::with_capacity(capacity)
            .or_else(|_| FastVec::with_capacity(capacity / 2))
            .unwrap_or_else(|_| FastVec::new());

        // Only resize if we got some capacity
        if storage.capacity() > 0 {
            let actual_capacity = storage.capacity();
            let _ = storage.resize_with(actual_capacity, Default::default);
        }
        storage
    }

    fn resize_storage(
        storage: &mut Self::Storage,
        new_capacity: usize,
        config: &Self::Config,
    ) -> Result<()> {
        // TODO: Use cache-aligned reallocation
        storage.resize_with(new_capacity, Default::default)?;
        Ok(())
    }

    fn get_buckets_mut(storage: &mut Self::Storage) -> &mut [HashBucket<K, V>] {
        storage.as_mut_slice()
    }

    fn get_buckets(storage: &Self::Storage) -> &[HashBucket<K, V>] {
        storage.as_slice()
    }

    fn capacity(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn memory_usage(storage: &Self::Storage) -> usize {
        storage.capacity() * std::mem::size_of::<HashBucket<K, V>>()
    }

    fn optimize_layout(storage: &mut Self::Storage, config: &Self::Config) -> Result<()> {
        // TODO: Implement cache optimization
        Ok(())
    }
}

/// SIMD optimization strategy
pub struct SimdOptimizationStrategy {
    simd_ops: &'static SimdStringOps,
}

impl SimdOptimizationStrategy {
    pub fn new() -> Self {
        Self {
            simd_ops: crate::hash_map::simd_string_ops::get_global_simd_ops(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimdOptimizationConfig {
    pub enable_string_ops: bool,
    pub enable_bulk_ops: bool,
    pub enable_hash_computation: bool,
}

#[derive(Debug, Default)]
pub struct SimdOptimizationContext {
    pub simd_operations: u64,
    pub bulk_operations: u64,
    pub string_comparisons: u64,
}

impl<K, V> HashOptimizationStrategy<K, V> for SimdOptimizationStrategy {
    type Config = SimdOptimizationConfig;
    type Context = SimdOptimizationContext;

    fn pre_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    ) {
        // TODO: Implement SIMD-accelerated pre-insert optimizations
        context.simd_operations += 1;
    }

    fn post_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        inserted: bool,
        config: &Self::Config,
    ) {
        // TODO: Implement post-insert optimizations
    }

    fn pre_lookup<Q>(
        &self,
        context: &mut Self::Context,
        key: &Q,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    ) where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement SIMD-accelerated lookup optimizations
        context.simd_operations += 1;
    }

    fn bulk_optimize(
        &self,
        context: &mut Self::Context,
        operations: &[OptimizationHint],
        config: &Self::Config,
    ) {
        context.bulk_operations += operations.len() as u64;
        // TODO: Implement bulk SIMD optimizations
    }

    fn metrics(&self, context: &Self::Context) -> OptimizationMetrics {
        OptimizationMetrics {
            simd_operations: context.simd_operations,
            bulk_operations: context.bulk_operations,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robin_hood_strategy() {
        let strategy = RobinHoodStrategy::new(64, true, true);
        let config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        };
        let mut context = RobinHoodContext::default();
        let mut buckets = vec![HashBucket::default(); 16];

        // Test insertion
        let result = strategy.insert(
            &mut context,
            &mut buckets,
            "key1".to_string(),
            "value1".to_string(),
            123456,
            &config,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test lookup
        let found = strategy.get(&context, &buckets, "key1", 123456, &config);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), "value1");
    }

    #[test]
    fn test_standard_storage_strategy() {
        let strategy = StandardStorageStrategy;
        let config = StandardStorageConfig {
            initial_capacity: 16,
            growth_factor: 2.0,
        };

        let mut storage: <StandardStorageStrategy as StorageLayoutStrategy<String, i32>>::Storage =
            StandardStorageStrategy::create_storage(16, &config);
        assert_eq!(StandardStorageStrategy::capacity(&storage), 16);

        let buckets: &mut [HashBucket<String, i32>] = StandardStorageStrategy::get_buckets_mut(&mut storage);
        assert_eq!(buckets.len(), 16);
    }
}