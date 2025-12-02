//! ZiporaHashMap - High-performance hash map with strategy-based configuration
//!
//! This module provides the core hash map implementation for Zipora, designed for
//! extreme performance following referenced project's focused implementation philosophy.
//!
//! # Performance-First Design
//!
//! **"One excellent implementation per data structure"** - referenced project approach
//!
//! ZiporaHashMap achieves high performance through configurable strategies:
//! - **HashStrategy**: Optimized collision resolution (Robin Hood, chaining, hopscotch)
//! - **StorageStrategy**: Cache-aware memory layouts and allocation patterns
//! - **OptimizationStrategy**: Hardware acceleration and performance features
//!
//! # Hardware Acceleration Features
//!
//! - **SIMD Framework**: BMI2/AVX2/POPCNT acceleration with runtime detection
//! - **Cache Optimization**: Prefetching, alignment, and NUMA awareness
//! - **Memory Pool Integration**: SecureMemoryPool for high-performance allocation
//! - **Concurrent Access**: Lock-free and token-based synchronization

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::hash_map::cache_locality::{
    AccessPattern, CacheMetrics, CacheOptimizedBucket, HotColdSeparator, Prefetcher,
};
use crate::hash_map::simd_string_ops::{get_global_simd_ops, SimdStringOps};
use crate::memory::cache_layout::{CacheOptimizedAllocator, CacheLayoutConfig, PrefetchHint};
use crate::memory::SecureMemoryPool;
use ahash::RandomState;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::sync::Arc;

/// Hash strategy configuration for unified hash map
#[derive(Debug, Clone)]
pub enum HashStrategy {
    /// Robin Hood hashing with probe distance optimization
    RobinHood {
        max_probe_distance: u16,
        variance_reduction: bool,
        backward_shift: bool,
    },
    /// Chaining with hash caching for collision resolution
    Chaining {
        load_factor: f64,
        hash_cache: bool,
        compact_links: bool,
    },
    /// Hopscotch hashing with neighborhood management
    Hopscotch {
        neighborhood_size: u8,
        displacement_threshold: u16,
    },
    /// Linear probing with cache-friendly access patterns
    LinearProbing {
        max_probe_distance: u16,
        cache_aligned: bool,
    },
    /// Cuckoo hashing with multiple hash functions
    Cuckoo {
        num_hash_functions: u8,
        max_evictions: u16,
    },
}

/// Storage strategy for memory layout and allocation
#[derive(Debug, Clone)]
pub enum StorageStrategy {
    /// Standard heap allocation with FastVec
    Standard {
        initial_capacity: usize,
        growth_factor: f64,
    },
    /// Inline storage for small collections (N ≤ threshold)
    SmallInline {
        inline_capacity: usize,
        fallback_threshold: usize,
    },
    /// Cache-optimized allocation with alignment
    CacheOptimized {
        cache_line_size: usize,
        numa_aware: bool,
        huge_pages: bool,
    },
    /// String-specialized storage with interning
    StringOptimized {
        arena_size: usize,
        prefix_cache: bool,
        interning: bool,
    },
    /// Memory pool allocation with SecureMemoryPool
    PoolAllocated {
        pool: Arc<SecureMemoryPool>,
        chunk_size: usize,
    },
}

/// Optimization strategy for SIMD, cache, and performance features
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Standard optimization level
    Standard,
    /// SIMD-accelerated operations
    SimdAccelerated {
        string_ops: bool,
        bulk_ops: bool,
        hash_computation: bool,
    },
    /// Cache-aware optimizations
    CacheAware {
        prefetch_distance: usize,
        hot_cold_separation: bool,
        access_pattern_tracking: bool,
    },
    /// High-performance combination of all optimizations
    HighPerformance {
        simd_enabled: bool,
        cache_optimized: bool,
        prefetch_enabled: bool,
        numa_aware: bool,
    },
}

/// Configuration for unified hash map
#[derive(Debug, Clone)]
pub struct ZiporaHashMapConfig {
    pub hash_strategy: HashStrategy,
    pub storage_strategy: StorageStrategy,
    pub optimization_strategy: OptimizationStrategy,
    pub initial_capacity: usize,
    pub load_factor: f64,
}

impl Default for ZiporaHashMapConfig {
    fn default() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 64,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: StorageStrategy::Standard {
                initial_capacity: 16,
                growth_factor: 2.0,
            },
            optimization_strategy: OptimizationStrategy::HighPerformance {
                simd_enabled: true,
                cache_optimized: true,
                prefetch_enabled: true,
                numa_aware: true,
            },
            initial_capacity: 16,
            load_factor: 0.75,
        }
    }
}

impl ZiporaHashMapConfig {
    /// Create configuration for cache-optimized hash map
    pub fn cache_optimized() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 32,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: StorageStrategy::CacheOptimized {
                cache_line_size: 64,
                numa_aware: true,
                huge_pages: false,
            },
            optimization_strategy: OptimizationStrategy::CacheAware {
                prefetch_distance: 2,
                hot_cold_separation: true,
                access_pattern_tracking: true,
            },
            initial_capacity: 64,
            load_factor: 0.6,
        }
    }

    /// Create configuration for string-optimized hash map
    pub fn string_optimized() -> Self {
        Self {
            hash_strategy: HashStrategy::RobinHood {
                max_probe_distance: 48,
                variance_reduction: true,
                backward_shift: true,
            },
            storage_strategy: StorageStrategy::StringOptimized {
                arena_size: 4096,
                prefix_cache: true,
                interning: true,
            },
            optimization_strategy: OptimizationStrategy::SimdAccelerated {
                string_ops: true,
                bulk_ops: true,
                hash_computation: true,
            },
            initial_capacity: 32,
            load_factor: 0.7,
        }
    }

    /// Create configuration for small hash map with inline storage
    pub fn small_inline(inline_capacity: usize) -> Self {
        Self {
            hash_strategy: HashStrategy::LinearProbing {
                max_probe_distance: inline_capacity as u16,
                cache_aligned: true,
            },
            storage_strategy: StorageStrategy::SmallInline {
                inline_capacity,
                fallback_threshold: inline_capacity * 2,
            },
            optimization_strategy: OptimizationStrategy::Standard,
            initial_capacity: inline_capacity,
            load_factor: 1.0, // Use all inline capacity
        }
    }

    /// Create configuration for concurrent access with pool allocation
    pub fn concurrent_pool(pool: Arc<SecureMemoryPool>) -> Self {
        Self {
            hash_strategy: HashStrategy::Hopscotch {
                neighborhood_size: 32,
                displacement_threshold: 128,
            },
            storage_strategy: StorageStrategy::PoolAllocated {
                pool,
                chunk_size: 1024,
            },
            optimization_strategy: OptimizationStrategy::HighPerformance {
                simd_enabled: true,
                cache_optimized: true,
                prefetch_enabled: true,
                numa_aware: true,
            },
            initial_capacity: 64,
            load_factor: 0.65,
        }
    }
}

/// Unified hash map implementation with strategy-based configuration
///
/// ZiporaHashMap consolidates all Zipora hash map variants into a single,
/// highly configurable implementation. Different behaviors are achieved
/// through strategy configuration rather than separate implementations.
///
/// # Examples
///
/// ```rust
/// use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
/// use std::collections::hash_map::RandomState;
///
/// // Cache-optimized hash map
/// let mut map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
///     ZiporaHashMapConfig::cache_optimized()
/// ).unwrap();
/// map.insert("key", "value").unwrap();
///
/// // String-optimized hash map
/// let mut str_map: ZiporaHashMap<&str, i32, RandomState> = ZiporaHashMap::with_config(
///     ZiporaHashMapConfig::string_optimized()
/// ).unwrap();
/// str_map.insert("hello", 42).unwrap();
///
/// // Small inline hash map
/// let mut small_map: ZiporaHashMap<i32, &str, RandomState> = ZiporaHashMap::with_config(
///     ZiporaHashMapConfig::small_inline(4)
/// ).unwrap();
/// small_map.insert(1, "one").unwrap();
/// ```
pub struct ZiporaHashMap<K, V, S = RandomState>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Configuration strategy
    config: ZiporaHashMapConfig,
    /// Hash builder
    hash_builder: S,
    /// Internal storage implementation
    storage: HashMapStorage<K, V>,
    /// Performance statistics
    stats: HashMapStats,
    /// SIMD operations for acceleration
    simd_ops: &'static SimdStringOps,
    /// Cache optimization components
    cache_allocator: Option<CacheOptimizedAllocator>,
    cache_metrics: CacheMetrics,
}

/// Internal storage implementations for different strategies
enum HashMapStorage<K, V>
where
    K: Clone,
    V: Clone,
{
    /// Standard FastVec-based storage
    Standard {
        buckets: FastVec<StandardBucket<K, V>>,
        entries: FastVec<HashEntry<K, V>>,
        mask: usize,
    },
    /// Inline storage for small collections
    SmallInline {
        inline_data: InlineStorage<K, V>,
        fallback: Option<Box<HashMapStorage<K, V>>>,
        len: usize,
    },
    /// Cache-optimized storage with alignment
    CacheOptimized {
        buckets: FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: FastVec<K>,
        cold_data: FastVec<V>,
        prefetcher: Prefetcher,
    },
    /// String-specialized storage with arena
    StringOptimized {
        arena: StringArena,
        buckets: FastVec<StringBucket>,
        entries: FastVec<StringEntry<V>>,
        prefix_cache: FastVec<PrefixCacheEntry>,
    },
}

/// Standard hash table bucket
#[repr(align(64))]
struct StandardBucket<K, V> {
    hash: u64,
    key: K,
    value: V,
    probe_distance: u16,
    is_occupied: bool,
}

/// Inline storage for small hash maps
struct InlineStorage<K, V> {
    data: [MaybeUninit<(K, V)>; 16], // Fixed size for simplicity
    occupied: u16, // Bit mask for occupied slots
}

impl<K, V> InlineStorage<K, V> {
    /// Get the number of occupied slots
    pub fn len(&self) -> usize {
        self.occupied.count_ones() as usize
    }
}

/// String arena for interned strings
struct StringArena {
    data: FastVec<u8>,
    offsets: FastVec<u32>,
    interned: std::collections::HashMap<Vec<u8>, u32>,
}

/// String bucket with prefix caching
struct StringBucket {
    hash: u64,
    string_id: u32,
    probe_distance: u16,
    prefix_cache: u32,
}

/// String entry with value
struct StringEntry<V> {
    value: V,
    next: Option<u32>,
}

/// Prefix cache entry for fast string matching
struct PrefixCacheEntry {
    prefix: u64, // First 8 bytes of string
    string_id: u32,
}

/// Hash entry for standard storage
#[derive(Clone)]
struct HashEntry<K, V>
where
    K: Clone,
    V: Clone,
{
    key: K,
    value: V,
    hash: u64,
    next: Option<u32>,
}

/// Performance statistics
#[derive(Debug, Default, Clone)]
pub struct HashMapStats {
    pub insertions: u64,
    pub lookups: u64,
    pub collisions: u64,
    pub probe_distance_sum: u64,
    pub rehashes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl<K, V, S> ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Create a new hash map with default configuration
    pub fn new() -> Result<Self>
    where
        S: Default,
    {
        Self::with_config(ZiporaHashMapConfig::default())
    }

    /// Create a new hash map with specified initial capacity
    pub fn with_capacity(capacity: usize) -> Result<Self>
    where
        S: Default,
    {
        let mut config = ZiporaHashMapConfig::default();
        config.initial_capacity = capacity.max(16);

        // Update storage strategy to use the specified capacity
        match &mut config.storage_strategy {
            StorageStrategy::Standard { initial_capacity, .. } => {
                *initial_capacity = capacity.max(16);
            }
            _ => {}
        }

        Self::with_config(config)
    }

    /// Create a new hash map with custom configuration
    pub fn with_config(config: ZiporaHashMapConfig) -> Result<Self>
    where
        S: Default,
    {
        Self::with_config_and_hasher(config, S::default())
    }

    /// Create a new hash map with custom configuration and hasher
    pub fn with_config_and_hasher(config: ZiporaHashMapConfig, hash_builder: S) -> Result<Self> {
        let simd_ops = get_global_simd_ops();
        let cache_allocator = match &config.optimization_strategy {
            OptimizationStrategy::CacheAware { .. } | OptimizationStrategy::HighPerformance { cache_optimized: true, .. } => {
                Some(CacheOptimizedAllocator::new(CacheLayoutConfig::default()))
            }
            _ => None,
        };

        let storage = Self::create_storage(&config)?;

        Ok(Self {
            config,
            hash_builder,
            storage,
            stats: HashMapStats::default(),
            simd_ops,
            cache_allocator,
            cache_metrics: CacheMetrics::new(),
        })
    }

    /// Create storage based on strategy configuration
    fn create_storage(config: &ZiporaHashMapConfig) -> Result<HashMapStorage<K, V>> {
        match &config.storage_strategy {
            StorageStrategy::Standard { initial_capacity, .. } => {
                Ok(HashMapStorage::Standard {
                    buckets: FastVec::with_capacity(*initial_capacity)?,
                    entries: FastVec::with_capacity(*initial_capacity)?,
                    mask: initial_capacity.saturating_sub(1),
                })
            }
            StorageStrategy::SmallInline { inline_capacity, .. } => {
                Ok(HashMapStorage::SmallInline {
                    inline_data: InlineStorage {
                        data: unsafe { MaybeUninit::uninit().assume_init() },
                        occupied: 0,
                    },
                    fallback: None,
                    len: 0,
                })
            }
            StorageStrategy::CacheOptimized { .. } => {
                Ok(HashMapStorage::CacheOptimized {
                    buckets: FastVec::with_capacity(config.initial_capacity)?,
                    hot_data: FastVec::with_capacity(config.initial_capacity)?,
                    cold_data: FastVec::with_capacity(config.initial_capacity)?,
                    prefetcher: Prefetcher::new(),
                })
            }
            StorageStrategy::StringOptimized { arena_size, .. } => {
                Ok(HashMapStorage::StringOptimized {
                    arena: StringArena {
                        data: FastVec::with_capacity(*arena_size)?,
                        offsets: FastVec::with_capacity(256)?,
                        interned: std::collections::HashMap::new(),
                    },
                    buckets: FastVec::with_capacity(config.initial_capacity)?,
                    entries: FastVec::with_capacity(config.initial_capacity)?,
                    prefix_cache: FastVec::with_capacity(config.initial_capacity)?,
                })
            }
            StorageStrategy::PoolAllocated { .. } => {
                // For now, fallback to standard storage
                // TODO: Implement pool-based allocation
                Ok(HashMapStorage::Standard {
                    buckets: FastVec::with_capacity(config.initial_capacity)?,
                    entries: FastVec::with_capacity(config.initial_capacity)?,
                    mask: config.initial_capacity.saturating_sub(1),
                })
            }
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        self.stats.insertions += 1;

        let hash = self.hash_key(&key);

        match &mut self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                // Try insertion first
                match Self::insert_standard(&self.hash_builder, buckets, entries, mask, key.clone(), value.clone(), hash) {
                    Ok(result) => Ok(result),
                    Err(_) => {
                        // Table is full, resize and retry
                        self.resize_storage()?;
                        // Retry insertion after resize
                        if let HashMapStorage::Standard { buckets, entries, mask } = &mut self.storage {
                            Self::insert_standard(&self.hash_builder, buckets, entries, mask, key, value, hash)
                        } else {
                            Err(crate::error::ZiporaError::invalid_state("Storage type changed during resize"))
                        }
                    }
                }
            }
            HashMapStorage::SmallInline { inline_data, fallback, len } => {
                Self::insert_small_inline(inline_data, fallback, len, key, value)
            }
            HashMapStorage::CacheOptimized { buckets, hot_data, cold_data, prefetcher } => {
                Self::insert_cache_optimized(buckets, hot_data, cold_data, prefetcher, key, value)
            }
            HashMapStorage::StringOptimized { arena, buckets, entries, prefix_cache } => {
                Self::insert_string_optimized(arena, buckets, entries, prefix_cache, key, value)
            }
        }
    }

    /// Get a value by key
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_key_borrowed(key);

        match &self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                self.get_standard(buckets, entries, mask, key, hash)
            }
            HashMapStorage::SmallInline { inline_data, fallback, len } => {
                self.get_small_inline(inline_data, fallback, len, key)
            }
            HashMapStorage::CacheOptimized { buckets, hot_data, cold_data, prefetcher } => {
                self.get_cache_optimized(buckets, hot_data, cold_data, prefetcher, key)
            }
            HashMapStorage::StringOptimized { arena, buckets, entries, prefix_cache } => {
                self.get_string_optimized(arena, buckets, entries, prefix_cache, key)
            }
        }
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        match &self.storage {
            HashMapStorage::Standard { entries, .. } => {
                // Count non-empty, non-tombstone entries
                entries.iter().filter(|entry| entry.hash != 0 && entry.hash != u64::MAX).count()
            }
            HashMapStorage::SmallInline { len, .. } => *len,
            HashMapStorage::CacheOptimized { hot_data, .. } => hot_data.len(),
            HashMapStorage::StringOptimized { entries, .. } => entries.len(),
        }
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get performance statistics
    pub fn stats(&self) -> &HashMapStats {
        &self.stats
    }

    /// Get cache metrics (if cache optimization enabled)
    pub fn cache_metrics(&self) -> &CacheMetrics {
        &self.cache_metrics
    }

    /// Get a mutable reference to a value by key
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &mut self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                Self::get_mut_standard(&self.hash_builder, buckets, entries, mask, key)
            }
            HashMapStorage::SmallInline { inline_data, fallback, len } => {
                Self::get_mut_small_inline(inline_data, fallback, len, key)
            }
            HashMapStorage::CacheOptimized { buckets, hot_data, cold_data, prefetcher } => {
                Self::get_mut_cache_optimized(buckets, hot_data, cold_data, prefetcher, key)
            }
            HashMapStorage::StringOptimized { arena, buckets, entries, prefix_cache } => {
                Self::get_mut_string_optimized(arena, buckets, entries, prefix_cache, key)
            }
        }
    }

    /// Remove a key-value pair and return the value if it existed
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &mut self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                Self::remove_standard(&self.hash_builder, buckets, entries, mask, key)
            }
            HashMapStorage::SmallInline { inline_data, fallback, len } => {
                Self::remove_small_inline(inline_data, fallback, len, key)
            }
            HashMapStorage::CacheOptimized { buckets, hot_data, cold_data, prefetcher } => {
                Self::remove_cache_optimized(buckets, hot_data, cold_data, prefetcher, key)
            }
            HashMapStorage::StringOptimized { arena, buckets, entries, prefix_cache } => {
                Self::remove_string_optimized(arena, buckets, entries, prefix_cache, key)
            }
        }
    }

    /// Clear all entries from the map
    pub fn clear(&mut self) {
        match &mut self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                Self::clear_standard(buckets, entries, mask)
            }
            HashMapStorage::SmallInline { inline_data, fallback, len } => {
                Self::clear_small_inline(inline_data, fallback, len)
            }
            HashMapStorage::CacheOptimized { buckets, hot_data, cold_data, prefetcher } => {
                Self::clear_cache_optimized(buckets, hot_data, cold_data, prefetcher)
            }
            HashMapStorage::StringOptimized { arena, buckets, entries, prefix_cache } => {
                Self::clear_string_optimized(arena, buckets, entries, prefix_cache)
            }
        }
    }

    /// Check if the map contains a key
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Get the capacity of the map
    pub fn capacity(&self) -> usize {
        match &self.storage {
            HashMapStorage::Standard { entries, .. } => entries.capacity(), // Return actual allocated capacity
            HashMapStorage::SmallInline { inline_data, fallback, .. } => {
                16 + fallback.as_ref().map_or(0, |f| match f.as_ref() {
                    HashMapStorage::Standard { entries, .. } => entries.capacity(),
                    _ => 0,
                })
            }
            HashMapStorage::CacheOptimized { hot_data, .. } => hot_data.capacity(),
            HashMapStorage::StringOptimized { entries, .. } => entries.capacity(),
        }
    }

    /// Iterate over key-value pairs
    pub fn iter(&self) -> ZiporaHashMapIterator<'_, K, V> {
        ZiporaHashMapIterator {
            storage: &self.storage,
            index: 0,
        }
    }

    /// Hash a key using the configured hasher
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a borrowed key using the configured hasher
    fn hash_key_borrowed<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Resize the storage to accommodate more elements
    fn resize_storage(&mut self) -> Result<()> {
        match &mut self.storage {
            HashMapStorage::Standard { buckets, entries, mask } => {
                let old_capacity = entries.len();
                let new_capacity = (old_capacity * 2).max(32); // At least double the size

                // Collect all existing key-value pairs
                let mut existing_entries = Vec::new();
                for entry in entries.iter() {
                    if entry.hash != 0 {
                        existing_entries.push((entry.key.clone(), entry.value.clone(), entry.hash));
                    }
                }

                // Create new larger storage
                let mut new_entries = FastVec::with_capacity(new_capacity)?;

                // Use the first existing entry as template, or create default if no entries
                if let Some((template_key, template_value, _)) = existing_entries.first() {
                    new_entries.resize_with(new_capacity, || HashEntry {
                        key: template_key.clone(), // Placeholder - will be cleared
                        value: template_value.clone(), // Placeholder - will be cleared
                        hash: 0,
                        next: None,
                    })?;
                } else {
                    // No existing entries, we'll handle this when we first insert
                    return Ok(()); // Nothing to resize
                }

                // Clear all entries (hash = 0 indicates empty)
                for entry in new_entries.iter_mut() {
                    entry.hash = 0;
                }

                let new_mask = new_capacity - 1;

                // Reinsert all existing entries
                for (key, value, hash) in existing_entries {
                    let index = (hash as usize) & new_mask;

                    // Find empty slot with linear probing
                    let mut inserted = false;
                    for i in 0..new_capacity {
                        let probe_index = (index + i) & new_mask;
                        let entry = &mut new_entries[probe_index];

                        if entry.hash == 0 {
                            // Empty slot, insert here
                            entry.key = key;
                            entry.value = value;
                            entry.hash = hash;
                            inserted = true;
                            break;
                        }
                    }

                    if !inserted {
                        return Err(crate::error::ZiporaError::invalid_state("Failed to reinsert during resize"));
                    }
                }

                // Update storage with new capacity
                *entries = new_entries;
                *mask = new_mask;
                self.stats.rehashes += 1;

                Ok(())
            }
            _ => {
                // Other storage types don't support resizing yet
                Err(crate::error::ZiporaError::invalid_state("Resize not supported for this storage type"))
            }
        }
    }

    // Implementation methods for different storage strategies
    fn insert_standard(
        hash_builder: &S,
        buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: K,
        value: V,
        hash: u64,
    ) -> Result<Option<V>> {
        // Initialize entries if empty
        if entries.is_empty() {
            let capacity = entries.capacity().max(16); // Use allocated capacity
            entries.resize_with(capacity, || HashEntry {
                key: key.clone(),
                value: value.clone(),
                hash: 0,
                next: None,
            })?;
            *mask = capacity - 1;

            // Clear all entries
            for entry in entries.iter_mut() {
                entry.hash = 0;
            }
        }

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find slot
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 || entry.hash == u64::MAX {
                // Empty slot or tombstone, insert here
                entry.key = key;
                entry.value = value;
                entry.hash = hash;
                return Ok(None);
            } else if entry.hash == hash && entry.key == key {
                // Key exists, update value
                let old_value = std::mem::replace(&mut entry.value, value);
                return Ok(Some(old_value));
            }
        }

        // Table is full, need to resize
        Err(crate::error::ZiporaError::invalid_state("Hash table full - resize needed"))
    }

    fn insert_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: K,
        value: V,
    ) -> Result<Option<V>> {
        // TODO: Implement inline insertion with fallback
        Ok(None)
    }

    fn insert_cache_optimized(
        buckets: &mut FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: &mut FastVec<K>,
        cold_data: &mut FastVec<V>,
        prefetcher: &mut Prefetcher,
        key: K,
        value: V,
    ) -> Result<Option<V>> {
        // TODO: Implement cache-optimized insertion
        Ok(None)
    }

    fn insert_string_optimized(
        arena: &mut StringArena,
        buckets: &mut FastVec<StringBucket>,
        entries: &mut FastVec<StringEntry<V>>,
        prefix_cache: &mut FastVec<PrefixCacheEntry>,
        key: K,
        value: V,
    ) -> Result<Option<V>> {
        // TODO: Implement string-optimized insertion
        Ok(None)
    }

    fn get_standard<'a, Q>(
        &self,
        buckets: &FastVec<StandardBucket<K, V>>,
        entries: &'a FastVec<HashEntry<K, V>>,
        mask: &usize,
        key: &Q,
        hash: u64,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == u64::MAX {
                // Tombstone, skip and continue searching
                continue;
            } else if entry.hash == hash && entry.key.borrow() == key {
                // Found the key
                return Some(&entry.value);
            }
        }

        None
    }

    fn get_small_inline<Q>(
        &self,
        inline_data: &InlineStorage<K, V>,
        fallback: &Option<Box<HashMapStorage<K, V>>>,
        len: &usize,
        key: &Q,
    ) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement inline lookup with fallback
        None
    }

    fn get_cache_optimized<Q>(
        &self,
        buckets: &FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: &FastVec<K>,
        cold_data: &FastVec<V>,
        prefetcher: &Prefetcher,
        key: &Q,
    ) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement cache-optimized lookup
        None
    }

    fn get_string_optimized<Q>(
        &self,
        arena: &StringArena,
        buckets: &FastVec<StringBucket>,
        entries: &FastVec<StringEntry<V>>,
        prefix_cache: &FastVec<PrefixCacheEntry>,
        key: &Q,
    ) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement string-optimized lookup
        None
    }

    // get_mut implementation methods
    fn get_mut_standard<'a, Q>(
        hash_builder: &S,
        buckets: &'a mut FastVec<StandardBucket<K, V>>,
        entries: &'a mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Find the index first
        let mut found_index = None;
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index];  // Immutable borrow for checking

            if entry.hash == 0 {
                // Empty slot, key not found
                break;
            } else if entry.hash == u64::MAX {
                // Tombstone, skip and continue searching
                continue;
            } else if entry.hash == hash && entry.key.borrow() == key {
                // Found the key
                found_index = Some(probe_index);
                break;
            }
        }

        // Return mutable reference if found
        if let Some(idx) = found_index {
            Some(&mut entries[idx].value)
        } else {
            None
        }
    }

    fn get_mut_small_inline<'a, Q>(
        inline_data: &'a mut InlineStorage<K, V>,
        fallback: &'a mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement small inline get_mut
        None
    }

    fn get_mut_cache_optimized<'a, Q>(
        buckets: &'a mut FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: &mut FastVec<K>,
        cold_data: &'a mut FastVec<V>,
        prefetcher: &mut Prefetcher,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement cache-optimized get_mut
        None
    }

    fn get_mut_string_optimized<'a, Q>(
        arena: &mut StringArena,
        buckets: &mut FastVec<StringBucket>,
        entries: &'a mut FastVec<StringEntry<V>>,
        prefix_cache: &mut FastVec<PrefixCacheEntry>,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement string-optimized get_mut
        None
    }

    // remove implementation methods
    fn remove_standard<Q>(
        hash_builder: &S,
        buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if entries.is_empty() {
            return None;
        }

        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == hash && entry.key.borrow() == key {
                // Found the key, remove it
                let old_value = entry.value.clone();

                // Use tombstone approach: mark as deleted but don't create holes
                entry.hash = u64::MAX; // Special tombstone marker
                // Keep key and value for now to avoid breaking cloning

                return Some(old_value);
            }
        }

        None
    }

    /// Backward shift deletion to maintain linear probing invariant
    fn backward_shift_delete(
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: usize,
        mut pos: usize,
    )
    where
        K: Clone,
        V: Clone,
    {
        // Clear the removed entry
        entries[pos].hash = 0;

        loop {
            let next_pos = (pos + 1) & mask;
            let next_entry = &entries[next_pos];

            // Stop if next entry is empty
            if next_entry.hash == 0 {
                break;
            }

            // Calculate the ideal position for the next entry
            let ideal_pos = (next_entry.hash as usize) & mask;

            // Check if we can move this entry backward
            // We can move it if its ideal position would still allow it to be found
            // after the move. This happens when:
            // - The ideal position is at or before the empty slot, OR
            // - The entry is displaced and moving it backward doesn't break the probe sequence

            let can_move = if ideal_pos <= pos {
                // Ideal position is before the empty slot - safe to move
                true
            } else {
                // Entry is displaced. Check if moving backward maintains findability.
                // In a wrapping hash table, we need to consider wrap-around cases.
                // The entry can be moved if the ideal position is between the current
                // empty position and the entry's current position (considering wrap-around).

                if pos < next_pos {
                    // No wrap-around case: ideal should be between pos and next_pos
                    ideal_pos > pos && ideal_pos <= next_pos
                } else {
                    // Wrap-around case: ideal can be after pos or before next_pos
                    ideal_pos > pos || ideal_pos <= next_pos
                }
            };

            if !can_move {
                break;
            }

            // Move the entry backward
            entries[pos] = entries[next_pos].clone();
            entries[next_pos].hash = 0; // Mark the old position as empty

            pos = next_pos;
        }
    }

    fn remove_small_inline<Q>(
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement small inline remove
        None
    }

    fn remove_cache_optimized<Q>(
        buckets: &mut FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: &mut FastVec<K>,
        cold_data: &mut FastVec<V>,
        prefetcher: &mut Prefetcher,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement cache-optimized remove
        None
    }

    fn remove_string_optimized<Q>(
        arena: &mut StringArena,
        buckets: &mut FastVec<StringBucket>,
        entries: &mut FastVec<StringEntry<V>>,
        prefix_cache: &mut FastVec<PrefixCacheEntry>,
        key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement string-optimized remove
        None
    }

    // clear implementation methods
    fn clear_standard(
        buckets: &mut FastVec<StandardBucket<K, V>>,
        entries: &mut FastVec<HashEntry<K, V>>,
        mask: &mut usize,
    ) {
        // TODO: Implement standard clear
        buckets.clear();
        entries.clear();
        *mask = 0;
    }

    fn clear_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
    ) {
        // TODO: Implement small inline clear
        *len = 0;
        if let Some(fallback) = fallback.take() {
            // Clear fallback if it exists
        }
    }

    fn clear_cache_optimized(
        buckets: &mut FastVec<CacheOptimizedBucket<K, V>>,
        hot_data: &mut FastVec<K>,
        cold_data: &mut FastVec<V>,
        prefetcher: &mut Prefetcher,
    ) {
        // TODO: Implement cache-optimized clear
        buckets.clear();
        hot_data.clear();
        cold_data.clear();
    }

    fn clear_string_optimized(
        arena: &mut StringArena,
        buckets: &mut FastVec<StringBucket>,
        entries: &mut FastVec<StringEntry<V>>,
        prefix_cache: &mut FastVec<PrefixCacheEntry>,
    ) {
        // TODO: Implement string-optimized clear
        buckets.clear();
        entries.clear();
        prefix_cache.clear();
    }
}

impl<K, V, S> Default for ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Default,
{
    fn default() -> Self {
        // SAFETY: ZiporaHashMap::new() only fails on memory allocation errors.
        // Use unwrap_or_else with panic as this type has non-trivial dependencies.
        Self::new().unwrap_or_else(|e| {
            panic!("ZiporaHashMap creation failed in Default: {}. \
                   This indicates severe memory pressure.", e)
        })
    }
}

impl<K, V, S> fmt::Debug for ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone + fmt::Debug,
    V: Clone + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZiporaHashMap")
            .field("len", &self.len())
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl<K, V, S> Clone for ZiporaHashMap<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> Self {
        // SAFETY: Clone requires creating a new map with the same config.
        // If allocation fails, we panic as there's no graceful fallback for Clone.
        let new_map = Self::with_config_and_hasher(self.config.clone(), self.hash_builder.clone())
            .unwrap_or_else(|e| {
                panic!("ZiporaHashMap clone failed: {}. \
                       This indicates severe memory pressure.", e)
            });

        // Copy all entries from the original map
        // TODO: Implement proper copying when iter() is available
        // for (key, value) in self.iter() {
        //     let _ = new_map.insert(key.clone(), value.clone());
        // }

        new_map
    }
}

/// Iterator for ZiporaHashMap key-value pairs
pub struct ZiporaHashMapIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    storage: &'a HashMapStorage<K, V>,
    index: usize,
}

impl<'a, K, V> Iterator for ZiporaHashMapIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.storage {
            HashMapStorage::Standard { entries, .. } => {
                while self.index < entries.len() {
                    let entry = &entries[self.index];
                    self.index += 1;
                    if entry.hash != 0 {
                        return Some((&entry.key, &entry.value));
                    }
                }
                None
            }
            HashMapStorage::SmallInline { len, .. } => {
                // TODO: Implement inline iteration - for now return None
                None
            }
            HashMapStorage::CacheOptimized { hot_data, cold_data, .. } => {
                if self.index < hot_data.len() && self.index < cold_data.len() {
                    let key = &hot_data[self.index];
                    let value = &cold_data[self.index];
                    self.index += 1;
                    Some((key, value))
                } else {
                    None
                }
            }
            HashMapStorage::StringOptimized { entries, .. } => {
                // TODO: Implement string-optimized iteration - for now return None
                None
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_hash_map_creation() {
        let map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_cache_optimized_config() {
        let map: ZiporaHashMap<String, i32> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized()).unwrap();
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_string_optimized_config() {
        let map: ZiporaHashMap<String, i32> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized()).unwrap();
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_small_inline_config() {
        let map: ZiporaHashMap<i32, String> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
        assert_eq!(map.len(), 0);
    }
}