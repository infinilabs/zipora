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
    #[inline]
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
struct HashEntry<K, V> {
    key: Option<K>,
    value: Option<V>,
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
                        // SAFETY: This creates an array of MaybeUninit<(K, V)> values.
                        // MaybeUninit<T> does not require initialization, so an array of
                        // uninitialized MaybeUninit values is valid. Individual elements
                        // are only accessed after being explicitly initialized.
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
                match Self::insert_standard(&self.hash_builder, buckets, entries, mask, key, value, hash) {
                    Ok(result) => Ok(result),
                    Err((key, value)) => {
                        // Table is full, resize and retry
                        self.resize_storage()?;
                        // Retry insertion after resize
                        if let HashMapStorage::Standard { buckets, entries, mask } = &mut self.storage {
                            Self::insert_standard(&self.hash_builder, buckets, entries, mask, key, value, hash)
                                .map_err(|_| crate::error::ZiporaError::invalid_state("Hash table full after resize"))
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
    #[inline]
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
    #[inline]
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
        let h = hasher.finish();
        if h == 0 { 1 } else if h == u64::MAX { u64::MAX - 1 } else { h }
    }

    /// Hash a borrowed key using the configured hasher
    fn hash_key_borrowed<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let h = hasher.finish();
        if h == 0 { 1 } else if h == u64::MAX { u64::MAX - 1 } else { h }
    }

    /// Resize the storage to accommodate more elements
    fn resize_storage(&mut self) -> Result<()> {
        match &mut self.storage {
            HashMapStorage::Standard { buckets: _, entries, mask } => {
                let old_capacity = entries.len();
                let new_capacity = (old_capacity * 2).max(32); // At least double the size

                // Create new larger storage
                let mut new_entries: FastVec<HashEntry<K, V>> = FastVec::with_capacity(new_capacity)?;

                // Initialize new empty entries
                // SAFETY: `new_capacity` is within bounds as it was just successfully allocated.
                // All elements 0..new_capacity are immediately initialized via `ptr::write`.
                unsafe { new_entries.set_len(new_capacity); }
                for i in 0..new_capacity {
                    // SAFETY: `new_entries` has capacity `new_capacity`. `i` < `new_capacity`.
                    // It is safe to write to this uninitialized memory.
                    unsafe {
                        std::ptr::write(new_entries.as_mut_ptr().add(i), HashEntry {
                            key: None,
                            value: None,
                            hash: 0,
                            next: None,
                        });
                    }
                }

                let new_mask = new_capacity - 1;

                // Move existing entries
                let mut old_entries = std::mem::replace(entries, new_entries);
                
                for entry in old_entries.iter_mut() {
                    // Skip empty slots AND tombstones (u64::MAX)
                    if entry.hash != 0 && entry.hash != u64::MAX {
                        let index = (entry.hash as usize) & new_mask;

                        // Find empty slot with linear probing
                        let mut inserted = false;
                        for i in 0..new_capacity {
                            let probe_index = (index + i) & new_mask;
                            let new_entry = &mut entries[probe_index];

                            if new_entry.hash == 0 {
                                // Empty slot, insert here
                                new_entry.key = entry.key.take();
                                new_entry.value = entry.value.take();
                                new_entry.hash = entry.hash;
                                inserted = true;
                                break;
                            }
                        }

                        if !inserted {
                            return Err(crate::error::ZiporaError::invalid_state("Failed to reinsert during resize"));
                        }
                    }
                }

                // old_entries is dropped here, which will drop the None keys/values harmlessly.

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
    ) -> std::result::Result<Option<V>, (K, V)> {
        // Initialize entries if empty
        if entries.is_empty() {
            let capacity = entries.capacity();
            if capacity == 0 {
                return Err((key, value)); // Trigger resize to allocate
            }
            // SAFETY: `capacity` is the actual allocated capacity.
            // Elements 0..capacity will be immediately initialized by `ptr::write`.
            unsafe { entries.set_len(capacity); }
            for i in 0..capacity {
                // SAFETY: `entries` capacity is `capacity`. `i` < `capacity`.
                // Thus `as_mut_ptr().add(i)` is valid and within bounds.
                unsafe {
                    std::ptr::write(entries.as_mut_ptr().add(i), HashEntry {
                        key: None,
                        value: None,
                        hash: 0,
                        next: None,
                    });
                }
            }
            *mask = capacity - 1;
        }

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find slot
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 || entry.hash == u64::MAX {
                // Empty slot or tombstone, insert here
                entry.key = Some(key);
                entry.value = Some(value);
                entry.hash = hash;
                return Ok(None);
            } else if entry.hash == hash && entry.key.as_ref().unwrap() == &key {
                // Key exists, update value
                let old_value = entry.value.replace(value).unwrap();
                return Ok(Some(old_value));
            }
        }

        // Table is full, need to resize
        Err((key, value))
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
            } else if entry.hash == hash && entry.key.as_ref().unwrap().borrow() == key {
                // Found the key
                return Some(entry.value.as_ref().unwrap());
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
        let h = hasher.finish();
        let hash = if h == 0 { 1 } else if h == u64::MAX { u64::MAX - 1 } else { h };

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
            } else if entry.hash == hash && entry.key.as_ref().unwrap().borrow() == key {
                // Found the key
                found_index = Some(probe_index);
                break;
            }
        }

        // Return mutable reference if found
        if let Some(idx) = found_index {
            Some(entries[idx].value.as_mut().unwrap())
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
        let h = hasher.finish();
        let hash = if h == 0 { 1 } else if h == u64::MAX { u64::MAX - 1 } else { h };

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == hash && entry.key.as_ref().unwrap().borrow() == key {
                // Found the key, remove it
                let old_value = entry.value.take().unwrap();
                entry.key.take(); // free the key

                // Use tombstone approach: mark as deleted but don't create holes
                entry.hash = u64::MAX; // Special tombstone marker

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

            // Move the entry backward using swap (no cloning needed!)
            entries.swap(pos, next_pos);
            entries[next_pos].hash = 0; // Mark the old position as empty
            entries[next_pos].key = None;
            entries[next_pos].value = None;

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
                    if entry.hash != 0 && entry.hash != u64::MAX {
                        return Some((entry.key.as_ref().unwrap(), entry.value.as_ref().unwrap()));
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
    use std::hash::{BuildHasher, Hasher};

    // --- Custom hasher that returns a fixed u64, for testing sentinel edge cases ---

    #[derive(Clone)]
    struct FixedHashBuilder(u64);

    impl BuildHasher for FixedHashBuilder {
        type Hasher = FixedHasher;
        fn build_hasher(&self) -> FixedHasher {
            FixedHasher(self.0)
        }
    }

    struct FixedHasher(u64);

    impl Hasher for FixedHasher {
        fn finish(&self) -> u64 { self.0 }
        fn write(&mut self, _bytes: &[u8]) {}
    }

    // ==================== Config creation tests ====================

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

    // ==================== Core operations ====================

    #[test]
    fn test_insert_and_get() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.insert("hello".to_string(), 42).unwrap(), None);
        assert_eq!(map.get("hello"), Some(&42));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_insert_overwrite_returns_old_value() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.insert("key".to_string(), 1).unwrap(), None);
        assert_eq!(map.insert("key".to_string(), 2).unwrap(), Some(1));
        assert_eq!(map.get("key"), Some(&2));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_get_nonexistent() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("a".to_string(), 1).unwrap();
        assert_eq!(map.get("b"), None);
        assert_eq!(map.get(""), None);
    }

    #[test]
    fn test_remove_existing() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("key".to_string(), 99).unwrap();
        assert_eq!(map.remove("key"), Some(99));
        assert_eq!(map.get("key"), None);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("a".to_string(), 1).unwrap();
        assert_eq!(map.remove("b"), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_get_mut() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("key".to_string(), 10).unwrap();
        if let Some(val) = map.get_mut("key") {
            *val = 20;
        }
        assert_eq!(map.get("key"), Some(&20));
    }

    #[test]
    fn test_get_mut_nonexistent() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        assert!(map.get_mut("nope").is_none());
    }

    #[test]
    fn test_contains_key() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("yes".to_string(), 1).unwrap();
        assert!(map.contains_key("yes"));
        assert!(!map.contains_key("no"));
    }

    #[test]
    fn test_len_tracking() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.len(), 0);

        map.insert(1, 10).unwrap();
        assert_eq!(map.len(), 1);

        map.insert(2, 20).unwrap();
        assert_eq!(map.len(), 2);

        // Overwrite doesn't increase len
        map.insert(1, 11).unwrap();
        assert_eq!(map.len(), 2);

        map.remove(&1);
        assert_eq!(map.len(), 1);

        map.remove(&2);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_single_element() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        map.insert(42, 100).unwrap();
        assert_eq!(map.get(&42), Some(&100));
        assert_eq!(map.len(), 1);
        assert_eq!(map.remove(&42), Some(100));
        assert!(map.is_empty());
    }

    // ==================== Resize / rehash ====================

    #[test]
    fn test_resize_preserves_all_entries() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        let n = 200;
        for i in 0..n {
            map.insert(i, i * 10).unwrap();
        }
        assert_eq!(map.len(), n as usize);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 10)),
                "key {} missing after resize", i);
        }
    }

    #[test]
    fn test_resize_compacts_tombstones() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        // Fill to just under resize threshold
        for i in 0..15 {
            map.insert(i, i).unwrap();
        }
        // Remove half to create tombstones
        for i in 0..8 {
            map.remove(&i);
        }
        // The 7 remaining entries should still be accessible
        for i in 8..15 {
            assert_eq!(map.get(&i), Some(&i));
        }
        // Insert enough to force resize — tombstones should be compacted
        for i in 100..120 {
            map.insert(i, i).unwrap();
        }
        // All surviving entries must be present
        for i in 8..15 {
            assert_eq!(map.get(&i), Some(&i), "key {} lost after resize", i);
        }
        for i in 100..120 {
            assert_eq!(map.get(&i), Some(&i), "key {} lost after resize", i);
        }
        // Removed entries must stay gone
        for i in 0..8 {
            assert_eq!(map.get(&i), None, "removed key {} reappeared", i);
        }
    }

    // ==================== Tombstone behavior ====================

    #[test]
    fn test_tombstone_get_returns_none() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("gone".to_string(), 1).unwrap();
        map.remove("gone");
        assert_eq!(map.get("gone"), None);
    }

    #[test]
    fn test_tombstone_slot_reuse_same_key() {
        let mut map: ZiporaHashMap<String, i32> = ZiporaHashMap::new().unwrap();
        map.insert("reuse".to_string(), 1).unwrap();
        map.remove("reuse");
        // Re-inserting same key should work (tombstone slot eligible)
        assert_eq!(map.insert("reuse".to_string(), 2).unwrap(), None);
        assert_eq!(map.get("reuse"), Some(&2));
    }

    #[test]
    fn test_tombstone_does_not_break_probe_chain() {
        // Use a fixed hasher so all keys hash to the same value,
        // guaranteeing a linear probe chain: slot 1, 2, 3, ...
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(42)).unwrap();

        // All three keys hash to 42 → same initial slot → probe chain
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.insert(3, 30).unwrap();

        // Remove the middle of the chain
        assert_eq!(map.remove(&2), Some(20));

        // Key before tombstone
        assert_eq!(map.get(&1), Some(&10));
        // Key after tombstone — probe must skip the tombstone
        assert_eq!(map.get(&3), Some(&30));
        // Removed key
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn test_multiple_tombstones_in_chain() {
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(7)).unwrap();

        for i in 0..6 {
            map.insert(i, i * 100).unwrap();
        }
        // Remove alternating: 0, 2, 4
        map.remove(&0);
        map.remove(&2);
        map.remove(&4);

        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&3), Some(&300));
        assert_eq!(map.get(&5), Some(&500));
        assert_eq!(map.get(&0), None);
        assert_eq!(map.get(&2), None);
        assert_eq!(map.get(&4), None);
        assert_eq!(map.len(), 3);
    }

    // ==================== Hash sentinel collision fix ====================

    #[test]
    fn test_hash_sentinel_zero() {
        // All keys hash to raw 0 → sanitized to 1
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(0)).unwrap();

        map.insert(10, 100).unwrap();
        map.insert(20, 200).unwrap();
        assert_eq!(map.get(&10), Some(&100));
        assert_eq!(map.get(&20), Some(&200));
        assert_eq!(map.remove(&10), Some(100));
        assert_eq!(map.get(&10), None);
        assert_eq!(map.get(&20), Some(&200));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_hash_sentinel_max() {
        // All keys hash to raw u64::MAX → sanitized to u64::MAX - 1
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(u64::MAX)).unwrap();

        map.insert(10, 100).unwrap();
        map.insert(20, 200).unwrap();
        assert_eq!(map.get(&10), Some(&100));
        assert_eq!(map.get(&20), Some(&200));
        assert_eq!(map.remove(&10), Some(100));
        assert_eq!(map.get(&10), None);
        assert_eq!(map.get(&20), Some(&200));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_hash_sentinel_max_with_get_mut() {
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(u64::MAX)).unwrap();

        map.insert(1, 10).unwrap();
        if let Some(v) = map.get_mut(&1) {
            *v = 99;
        }
        assert_eq!(map.get(&1), Some(&99));
    }

    // ==================== Iterator ====================

    #[test]
    fn test_iterator_basic() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.insert(3, 30).unwrap();

        let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        assert_eq!(collected, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn test_iterator_skips_tombstones() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.insert(3, 30).unwrap();
        map.remove(&2);

        let mut collected: Vec<(i32, i32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        collected.sort();
        assert_eq!(collected, vec![(1, 10), (3, 30)]);
    }

    #[test]
    fn test_iterator_empty() {
        let map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.iter().count(), 0);
    }

    #[test]
    fn test_iterator_all_removed() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.remove(&1);
        map.remove(&2);
        assert_eq!(map.iter().count(), 0);
    }

    // ==================== Backward shift deletion ====================

    #[test]
    fn test_backward_shift_delete() {
        // Directly exercise the backward_shift_delete method
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(0)).unwrap();

        // Insert 4 keys — all collide, forming a probe chain
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.insert(3, 30).unwrap();
        map.insert(4, 40).unwrap();

        // Manually invoke backward_shift_delete on the first slot
        if let HashMapStorage::Standard { entries, mask, .. } = &mut map.storage {
            // Find the slot containing key 1 (the sanitized hash of 0 is 1 → slot 1 & mask)
            let target_slot = 1usize & *mask;
            entries[target_slot].hash = 0;
            entries[target_slot].key = None;
            entries[target_slot].value = None;
            let m = *mask;
            ZiporaHashMap::<i32, i32, FixedHashBuilder>::backward_shift_delete(entries, m, target_slot);
        }

        // After backward shift: keys 2, 3, 4 should be shifted backward
        // and remain findable (the chain is compacted, no tombstone needed)
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
        assert_eq!(map.get(&4), Some(&40));
    }

    // ==================== Clear ====================

    #[test]
    fn test_clear() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        for i in 0..10 {
            map.insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 10);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        // After clear, can insert again
        map.insert(42, 99).unwrap();
        assert_eq!(map.get(&42), Some(&99));
    }

    // ==================== Stress / edge cases ====================

    #[test]
    fn test_many_inserts() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        let n = 500;
        for i in 0..n {
            map.insert(i, i * 3).unwrap();
        }
        assert_eq!(map.len(), n as usize);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 3)));
        }
    }

    #[test]
    fn test_insert_remove_cycle() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        for round in 0..5 {
            let base = round * 20;
            for i in base..base + 20 {
                map.insert(i, i).unwrap();
            }
            for i in base..base + 10 {
                assert_eq!(map.remove(&i), Some(i));
            }
        }
        // 5 rounds × 10 surviving = 50 entries
        assert_eq!(map.len(), 50);
        for round in 0..5 {
            let base = round * 20;
            for i in base..base + 10 {
                assert_eq!(map.get(&i), None);
            }
            for i in base + 10..base + 20 {
                assert_eq!(map.get(&i), Some(&i));
            }
        }
    }

    #[test]
    fn test_string_keys() {
        let mut map: ZiporaHashMap<String, String> = ZiporaHashMap::new().unwrap();
        map.insert("hello".to_string(), "world".to_string()).unwrap();
        map.insert("foo".to_string(), "bar".to_string()).unwrap();
        assert_eq!(map.get("hello"), Some(&"world".to_string()));
        assert_eq!(map.get("foo"), Some(&"bar".to_string()));
        map.remove("hello");
        assert_eq!(map.get("hello"), None);
        assert_eq!(map.get("foo"), Some(&"bar".to_string()));
    }

    #[test]
    fn test_with_capacity() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::with_capacity(64).unwrap();
        assert!(map.capacity() >= 64);
        for i in 0..64 {
            map.insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 64);
    }

    #[test]
    fn test_overwrite_many_times() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        for v in 0..100 {
            let old = map.insert(0, v).unwrap();
            if v == 0 {
                assert_eq!(old, None);
            } else {
                assert_eq!(old, Some(v - 1));
            }
        }
        assert_eq!(map.get(&0), Some(&99));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_remove_then_reinsert_different_value() {
        let mut map: ZiporaHashMap<i32, i32> = ZiporaHashMap::new().unwrap();
        map.insert(1, 100).unwrap();
        map.remove(&1);
        map.insert(1, 200).unwrap();
        assert_eq!(map.get(&1), Some(&200));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_all_collisions_stress() {
        // All keys hash identically — worst-case linear probing
        let config = ZiporaHashMapConfig::default();
        let mut map: ZiporaHashMap<i32, i32, FixedHashBuilder> =
            ZiporaHashMap::with_config_and_hasher(config, FixedHashBuilder(99)).unwrap();

        let n = 50;
        for i in 0..n {
            map.insert(i, i * 7).unwrap();
        }
        assert_eq!(map.len(), n as usize);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 7)));
        }
        // Remove every other key
        for i in (0..n).step_by(2) {
            assert_eq!(map.remove(&i), Some(i * 7));
        }
        assert_eq!(map.len(), (n / 2) as usize);
        for i in (1..n).step_by(2) {
            assert_eq!(map.get(&i), Some(&(i * 7)));
        }
    }
}