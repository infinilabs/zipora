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
    CacheMetrics,
};
use crate::hash_map::simd_string_ops::{SimdStringOps, get_global_simd_ops};
use crate::memory::cache_layout::{CacheLayoutConfig, CacheOptimizedAllocator};
use ahash::RandomState;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;
use crate::hash_map::config::{HashStorageStrategy, OptimizationStrategy, ZiporaHashMapConfig};
use crate::hash_map::storage::*;

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
    pub(super) config: ZiporaHashMapConfig,
    /// Hash builder
    pub(super) hash_builder: S,
    /// Internal storage implementation
    pub(super) storage: HashMapStorage<K, V>,
    /// Performance statistics
    pub(super) stats: HashMapStats,
    /// SIMD operations for acceleration
    pub(super) _simd_ops: &'static SimdStringOps,
    /// Cache optimization components
    pub(super) _cache_allocator: Option<CacheOptimizedAllocator>,
    pub(super) cache_metrics: CacheMetrics,
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
    pub(super) fn create_storage(config: &ZiporaHashMapConfig) -> Result<HashMapStorage<K, V>> {
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
        let mut new_map = Self::with_config_and_hasher(self.config.clone(), self.hash_builder.clone())
            .unwrap_or_else(|e| {
                panic!(
                    "ZiporaHashMap clone failed: {}. \
                       This indicates severe memory pressure.",
                    e
                )
            });

        // Copy all entries from the original map
        match &self.storage {
            HashMapStorage::Standard { .. } => {
                for (key, value) in self.iter() {
                    new_map.insert(key.clone(), value.clone()).unwrap_or_else(|e| {
                        panic!("ZiporaHashMap clone failed during insertion: {}", e);
                    });
                }
            }
            HashMapStorage::SmallInline { inline_data, fallback, .. } => {
                // Copy inline entries
                for i in 0..16 {
                    if (inline_data.occupied >> i) & 1 == 1 {
                        // SAFETY: Bit i is set in occupied, so slot i is initialized
                        let (k, v) = unsafe { inline_data._data[i].assume_init_ref() };
                        new_map.insert(k.clone(), v.clone()).unwrap_or_else(|e| {
                            panic!("ZiporaHashMap clone failed during inline insertion: {}", e);
                        });
                    }
                }
                // Copy fallback entries if any
                if let Some(fb) = fallback {
                    let fb_iter = ZiporaHashMapIterator {
                        storage: fb.as_ref(),
                        index: 0,
                    };
                    for (key, value) in fb_iter {
                        new_map.insert(key.clone(), value.clone()).unwrap_or_else(|e| {
                            panic!("ZiporaHashMap clone failed during fallback insertion: {}", e);
                        });
                    }
                }
            }
            _ => unreachable!("unimplemented strategies are rejected at construction"),
        }

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
