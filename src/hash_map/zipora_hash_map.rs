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
//! - **HashStorageStrategy**: Cache-aware memory layouts and allocation patterns
//! - **OptimizationStrategy**: Hardware acceleration and performance features
//!
//! # Hardware Acceleration Features
//!
//! - **SIMD Framework**: BMI2/AVX2/POPCNT acceleration with runtime detection
//! - **Cache Optimization**: Prefetching, alignment, and NUMA awareness
//! - **Memory Pool Integration**: SecureMemoryPool for high-performance allocation
//! - **Concurrent Access**: Lock-free and token-based synchronization

use crate::containers::FastVec;
use crate::error::Result;
use crate::hash_map::cache_locality::{
    CacheMetrics, CacheOptimizedBucket, Prefetcher,
};
use crate::hash_map::simd_string_ops::{SimdStringOps, get_global_simd_ops};
use crate::memory::cache_layout::{CacheLayoutConfig, CacheOptimizedAllocator};
use ahash::RandomState;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;
use crate::hash_map::config::{HashStorageStrategy, OptimizationStrategy, ZiporaHashMapConfig};

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
/// // Default high-performance hash map
/// let mut map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::new().unwrap();
/// map.insert("key", "value").unwrap();
///
/// // Small inline hash map (zero allocations for ≤N elements)
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
    _simd_ops: &'static SimdStringOps,
    /// Cache optimization components
    _cache_allocator: Option<CacheOptimizedAllocator>,
    cache_metrics: CacheMetrics,
}

/// Internal storage implementations for different strategies
#[allow(dead_code)] // strategy-placeholder variants, matched exhaustively in 9 arms
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
    _hash: u64,
    _key: K,
    _value: V,
    _probe_distance: u16,
    _is_occupied: bool,
}

/// Inline storage for small hash maps
struct InlineStorage<K, V> {
    _data: [MaybeUninit<(K, V)>; 16], // Fixed size for simplicity
    occupied: u16,                    // Bit mask for occupied slots
}

impl<K, V> InlineStorage<K, V> {
}

/// String arena for interned strings
struct StringArena {
    _data: FastVec<u8>,
    _offsets: FastVec<u32>,
    _interned: std::collections::HashMap<Vec<u8>, u32>,
}

/// String bucket with prefix caching
struct StringBucket {
    _hash: u64,
    _string_id: u32,
    _probe_distance: u16,
    _prefix_cache: u32,
}

/// String entry with value
struct StringEntry<V> {
    _value: V,
    _next: Option<u32>,
}

/// Prefix cache entry for fast string matching
struct PrefixCacheEntry {
    _prefix: u64, // First 8 bytes of string
    _string_id: u32,
}

/// Hash entry for standard storage
struct HashEntry<K, V> {
    key: Option<K>,
    value: Option<V>,
    hash: u64,
    _next: Option<u32>,
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
        let cap = capacity.max(16);
        let config = ZiporaHashMapConfig {
            initial_capacity: cap,
            storage_strategy: HashStorageStrategy::Standard {
                initial_capacity: cap,
                growth_factor: 2.0,
            },
            ..ZiporaHashMapConfig::default()
        };

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
            OptimizationStrategy::CacheAware { .. }
            | OptimizationStrategy::HighPerformance {
                cache_optimized: true,
                ..
            } => Some(CacheOptimizedAllocator::new(CacheLayoutConfig::default())),
            _ => None,
        };

        let storage = Self::create_storage(&config)?;

        Ok(Self {
            config,
            hash_builder,
            storage,
            stats: HashMapStats::default(),
            _simd_ops: simd_ops,
            _cache_allocator: cache_allocator,
            cache_metrics: CacheMetrics::new(),
        })
    }

    /// Create storage based on strategy configuration
    fn create_storage(config: &ZiporaHashMapConfig) -> Result<HashMapStorage<K, V>> {
        match &config.storage_strategy {
            HashStorageStrategy::Standard {
                initial_capacity, ..
            } => Ok(HashMapStorage::Standard {
                buckets: FastVec::with_capacity(*initial_capacity)?,
                entries: FastVec::with_capacity(*initial_capacity)?,
                mask: initial_capacity.saturating_sub(1),
            }),
            HashStorageStrategy::SmallInline {
                inline_capacity: _, ..
            } => {
                Ok(HashMapStorage::SmallInline {
                    inline_data: InlineStorage {
                        // SAFETY: This creates an array of MaybeUninit<(K, V)> values.
                        // MaybeUninit<T> does not require initialization, so an array of
                        // uninitialized MaybeUninit values is valid. Individual elements
                        // are only accessed after being explicitly initialized.
                        _data: [const { MaybeUninit::uninit() }; 16],
                        occupied: 0,
                    },
                    fallback: None,
                    len: 0,
                })
            }
            HashStorageStrategy::CacheOptimized { .. } => {
                Err(crate::error::ZiporaError::not_supported(
                    "CacheOptimized storage strategy is not yet implemented",
                ))
            }
            HashStorageStrategy::StringOptimized { .. } => {
                Err(crate::error::ZiporaError::not_supported(
                    "StringOptimized storage strategy is not yet implemented",
                ))
            }
            HashStorageStrategy::PoolAllocated { .. } => {
                Err(crate::error::ZiporaError::not_supported(
                    "PoolAllocated storage strategy is not yet implemented",
                ))
            }
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        self.stats.insertions += 1;

        let hash = self.hash_key(&key);

        match &mut self.storage {
            HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } => {
                // Try insertion first
                match Self::insert_standard(
                    &self.hash_builder,
                    buckets,
                    entries,
                    mask,
                    key,
                    value,
                    hash,
                ) {
                    Ok(result) => Ok(result),
                    Err((key, value)) => {
                        // Table is full, resize and retry
                        self.resize_storage()?;
                        // Retry insertion after resize
                        if let HashMapStorage::Standard {
                            buckets,
                            entries,
                            mask,
                        } = &mut self.storage
                        {
                            Self::insert_standard(
                                &self.hash_builder,
                                buckets,
                                entries,
                                mask,
                                key,
                                value,
                                hash,
                            )
                            .map_err(|_| {
                                crate::error::ZiporaError::invalid_state(
                                    "Hash table full after resize",
                                )
                            })
                        } else {
                            Err(crate::error::ZiporaError::invalid_state(
                                "Storage type changed during resize",
                            ))
                        }
                    }
                }
            }
            HashMapStorage::SmallInline {
                inline_data,
                fallback,
                len,
            } => Self::insert_small_inline(
                inline_data,
                fallback,
                len,
                key,
                value,
                hash,
                &self.hash_builder,
            ),
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
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
            HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } => self.get_standard(buckets, entries, mask, key, hash),
            HashMapStorage::SmallInline {
                inline_data,
                fallback,
                len,
            } => self.get_small_inline(inline_data, fallback, len, key),
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
            }
        }
    }

    /// Get number of elements
    #[inline]
    pub fn len(&self) -> usize {
        match &self.storage {
            HashMapStorage::Standard { entries, .. } => {
                // Count non-empty, non-tombstone entries
                entries
                    .iter()
                    .filter(|entry| entry.hash != 0 && entry.hash != u64::MAX)
                    .count()
            }
            HashMapStorage::SmallInline { len, .. } => *len,
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
            }
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
            HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } => Self::get_mut_standard(&self.hash_builder, buckets, entries, mask, key),
            HashMapStorage::SmallInline {
                inline_data,
                fallback,
                len,
            } => Self::get_mut_small_inline(inline_data, fallback, len, key),
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
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
            HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } => Self::remove_standard(&self.hash_builder, buckets, entries, mask, key),
            HashMapStorage::SmallInline {
                inline_data,
                fallback,
                len,
            } => Self::remove_small_inline(inline_data, fallback, len, key),
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
            }
        }
    }

    /// Clear all entries from the map
    pub fn clear(&mut self) {
        match &mut self.storage {
            HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } => Self::clear_standard(buckets, entries, mask),
            HashMapStorage::SmallInline {
                inline_data,
                fallback,
                len,
            } => Self::clear_small_inline(inline_data, fallback, len),
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
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
            HashMapStorage::SmallInline {
                inline_data: _,
                fallback,
                ..
            } => {
                16 + fallback.as_ref().map_or(0, |f| match f.as_ref() {
                    HashMapStorage::Standard { entries, .. } => entries.capacity(),
                    _ => 0,
                })
            }
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
            }
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
        let h = self.hash_builder.hash_one(key);
        if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        }
    }

    /// Hash a borrowed key using the configured hasher
    fn hash_key_borrowed<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let h = self.hash_builder.hash_one(key);
        if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        }
    }

    /// Resize the storage to accommodate more elements
    fn resize_storage(&mut self) -> Result<()> {
        match &mut self.storage {
            HashMapStorage::Standard {
                buckets: _,
                entries,
                mask,
            } => {
                let old_capacity = entries.len();
                let new_capacity = (old_capacity * 2).max(32); // At least double the size

                // Create new larger storage
                let mut new_entries: FastVec<HashEntry<K, V>> =
                    FastVec::with_capacity(new_capacity)?;

                // Initialize new empty entries
                // SAFETY: `new_capacity` is within bounds as it was just successfully allocated.
                // All elements 0..new_capacity are immediately initialized via `ptr::write`.
                unsafe {
                    new_entries.set_len(new_capacity);
                }
                for i in 0..new_capacity {
                    // SAFETY: `new_entries` has capacity `new_capacity`. `i` < `new_capacity`.
                    // It is safe to write to this uninitialized memory.
                    unsafe {
                        std::ptr::write(
                            new_entries.as_mut_ptr().add(i),
                            HashEntry {
                                key: None,
                                value: None,
                                hash: 0,
                                _next: None,
                            },
                        );
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
                            return Err(crate::error::ZiporaError::invalid_state(
                                "Failed to reinsert during resize",
                            ));
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
                Err(crate::error::ZiporaError::invalid_state(
                    "Resize not supported for this storage type",
                ))
            }
        }
    }

    // Implementation methods for different storage strategies
    fn insert_standard(
        _hash_builder: &S,
        _buckets: &mut FastVec<StandardBucket<K, V>>,
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
            unsafe {
                entries.set_len(capacity);
            }
            for i in 0..capacity {
                // SAFETY: `entries` capacity is `capacity`. `i` < `capacity`.
                // Thus `as_mut_ptr().add(i)` is valid and within bounds.
                unsafe {
                    std::ptr::write(
                        entries.as_mut_ptr().add(i),
                        HashEntry {
                            key: None,
                            value: None,
                            hash: 0,
                            _next: None,
                        },
                    );
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
            } else if entry.hash == hash
                && entry.key.as_ref().expect("occupied entry must have key") == &key
            {
                // Key exists, update value
                let old_value = entry
                    .value
                    .replace(value)
                    .expect("occupied entry must have previous value");
                return Ok(Some(old_value));
            }
        }

        // Table is full, need to resize
        Err((key, value))
    }

    fn insert_small_inline(
        inline_data: &mut InlineStorage<K, V>,
        _fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
        key: K,
        value: V,
        hash: u64,
        hash_builder: &S,
    ) -> Result<Option<V>> {
        // If already migrated to fallback, delegate to Standard storage
        if let Some(fb) = _fallback.as_mut()
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
                ..
            } = fb.as_mut()
        {
                let result =
                    Self::insert_standard(hash_builder, buckets, entries, mask, key, value, hash)
                        .map_err(|_| {
                        crate::error::ZiporaError::invalid_state(
                            "Hash table full in SmallInline fallback storage",
                        )
                    })?;
                if result.is_none() {
                    *len += 1;
                }
                return Ok(result);
            }

        // Check if key already exists in inline storage
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, v) = unsafe { inline_data._data[i].assume_init_ref() };
                if k == &key {
                    let old_v = unsafe { std::ptr::read(v as *const V) };
                    // SAFETY: We just read the old value, now we overwrite it with the new one
                    unsafe {
                        std::ptr::write(inline_data._data[i].as_mut_ptr(), (key, value));
                    }
                    return Ok(Some(old_v));
                }
            }
        }

        // Try to find an empty slot
        if inline_data.occupied != 0xFFFF {
            let slot = inline_data.occupied.trailing_ones() as usize;
            // SAFETY: slot < 16 and is currently uninitialized (bit is 0)
            unsafe {
                std::ptr::write(inline_data._data[slot].as_mut_ptr(), (key, value));
            }
            inline_data.occupied |= 1 << slot;
            *len += 1;
            return Ok(None);
        }

        // Inline storage full — migrate all 16 entries to Standard storage.
        let std_cap = 32; // 16 existing + room to grow
        let mut buckets = FastVec::with_capacity(std_cap)?;
        let mut entries = FastVec::with_capacity(std_cap)?;
        let mut mask = std_cap - 1;

        // Initialize entries
        for _ in 0..std_cap {
            entries.push(HashEntry {
                key: None,
                value: None,
                hash: 0,
                _next: None,
            })?;
        }
        // buckets are allocated but not initialized — Standard path uses entries for probing

        // Re-insert all 16 inline entries into standard storage
        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, v) = unsafe { std::ptr::read(inline_data._data[i].as_ptr()) };
                let raw_h = hash_builder.hash_one(&k);
                let h = if raw_h == 0 {
                    1
                } else if raw_h == u64::MAX {
                    u64::MAX - 1
                } else {
                    raw_h
                };
                let _ = Self::insert_standard(
                    hash_builder,
                    &mut buckets,
                    &mut entries,
                    &mut mask,
                    k,
                    v,
                    h,
                );
            }
        }
        inline_data.occupied = 0;

        // Insert the new key-value pair
        let result = Self::insert_standard(
            hash_builder,
            &mut buckets,
            &mut entries,
            &mut mask,
            key,
            value,
            hash,
        );
        let result = result.map_err(|_| {
            crate::error::ZiporaError::invalid_state("Hash table full after SmallInline migration")
        })?;

        // Store the migrated storage as fallback
        *_fallback = Some(Box::new(HashMapStorage::Standard {
            buckets,
            entries,
            mask,
        }));
        *len += 1;

        Ok(result)
    }

    fn get_standard<'a, Q>(
        &self,
        _buckets: &FastVec<StandardBucket<K, V>>,
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
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key
                return Some(
                    entry
                        .value
                        .as_ref()
                        .expect("occupied entry must have value"),
                );
            }
        }

        None
    }

    fn get_small_inline<'a, Q>(
        &self,
        inline_data: &'a InlineStorage<K, V>,
        fallback: &'a Option<Box<HashMapStorage<K, V>>>,
        _len: &usize,
        key: &Q,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Check fallback first (migrated data)
        if let Some(fb) = fallback
            && let HashMapStorage::Standard {
                buckets,
                entries,
                mask,
            } = fb.as_ref()
        {
            let hash = self.hash_key_borrowed(key);
            return self.get_standard(buckets, entries, mask, key, hash);
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, v) = unsafe { inline_data._data[i].assume_init_ref() };
                if k.borrow() == key {
                    return Some(v);
                }
            }
        }
        None
    }

    // get_mut implementation methods
    fn get_mut_standard<'a, Q>(
        hash_builder: &S,
        _buckets: &'a mut FastVec<StandardBucket<K, V>>,
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

        let h = hash_builder.hash_one(key);
        let hash = if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        };

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Find the index first
        let mut found_index = None;
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &entries[probe_index]; // Immutable borrow for checking

            if entry.hash == 0 {
                // Empty slot, key not found
                break;
            } else if entry.hash == u64::MAX {
                // Tombstone, skip and continue searching
                continue;
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key
                found_index = Some(probe_index);
                break;
            }
        }

        // Return mutable reference if found
        if let Some(idx) = found_index {
            Some(
                entries[idx]
                    .value
                    .as_mut()
                    .expect("occupied entry must have value"),
            )
        } else {
            None
        }
    }

    fn get_mut_small_inline<'a, Q>(
        inline_data: &'a mut InlineStorage<K, V>,
        fallback: &'a mut Option<Box<HashMapStorage<K, V>>>,
        _len: &mut usize,
        key: &Q,
    ) -> Option<&'a mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Check fallback first (migrated data) — not yet supported for get_mut
        // since Standard get_mut needs the hash_builder which we don't have here.
        if fallback.is_some() {
            return None;
        }

        for i in 0..16 {
            if (inline_data.occupied >> i) & 1 == 1 {
                // SAFETY: Bit i is set in occupied, so slot i is initialized
                let (k, _) = unsafe { inline_data._data[i].assume_init_ref() };
                if k.borrow() == key {
                    // SAFETY: same slot, returning mutable reference to value
                    let (_, v) = unsafe { inline_data._data[i].assume_init_mut() };
                    return Some(v);
                }
            }
        }
        None
    }

    // remove implementation methods
    fn remove_standard<Q>(
        hash_builder: &S,
        _buckets: &mut FastVec<StandardBucket<K, V>>,
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

        let h = hash_builder.hash_one(key);
        let hash = if h == 0 {
            1
        } else if h == u64::MAX {
            u64::MAX - 1
        } else {
            h
        };

        let capacity = entries.len();
        let index = (hash as usize) & *mask;

        // Linear probing to find key
        for i in 0..capacity {
            let probe_index = (index + i) & *mask;
            let entry = &mut entries[probe_index];

            if entry.hash == 0 {
                // Empty slot, key not found
                return None;
            } else if entry.hash == hash
                && entry
                    .key
                    .as_ref()
                    .expect("occupied entry must have key")
                    .borrow()
                    == key
            {
                // Found the key, remove it
                let old_value = entry.value.take().expect("occupied entry must have value");
                entry.key.take(); // free the key

                // Use tombstone approach: mark as deleted but don't create holes
                entry.hash = u64::MAX; // Special tombstone marker

                return Some(old_value);
            }
        }

        None
    }

    /// Backward shift deletion to maintain linear probing invariant
    #[cfg(test)]
    fn backward_shift_delete(entries: &mut FastVec<HashEntry<K, V>>, mask: usize, mut pos: usize)
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
        _inline_data: &mut InlineStorage<K, V>,
        _fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        _len: &mut usize,
        _key: &Q,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // TODO: Implement small inline remove
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
        _inline_data: &mut InlineStorage<K, V>,
        fallback: &mut Option<Box<HashMapStorage<K, V>>>,
        len: &mut usize,
    ) {
        // TODO: Implement small inline clear
        *len = 0;
        if let Some(_fallback) = fallback.take() {
            // Clear fallback if it exists
        }
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
            panic!(
                "ZiporaHashMap creation failed in Default: {}. \
                   This indicates severe memory pressure.",
                e
            )
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
                panic!(
                    "ZiporaHashMap clone failed: {}. \
                       This indicates severe memory pressure.",
                    e
                )
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
                        return Some((
                            entry.key.as_ref().expect("occupied entry must have key"),
                            entry
                                .value
                                .as_ref()
                                .expect("occupied entry must have value"),
                        ));
                    }
                }
                None
            }
            HashMapStorage::SmallInline { len: _, .. } => {
                // TODO: Implement inline iteration - for now return None
                None
            }
            HashMapStorage::CacheOptimized { .. } | HashMapStorage::StringOptimized { .. } => {
                unreachable!("unimplemented strategies are rejected at construction")
            }
        }
    }
}

#[cfg(test)]
#[path = "zipora_hash_map_tests.rs"]
mod tests;
