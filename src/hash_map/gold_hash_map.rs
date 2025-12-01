//! GoldHashMap - High-performance hash table with advanced features
//!
//! A production-grade hash map implementation inspired by Terark's gold_hash_map.hpp,
//! featuring:
//! - Custom link types (u32/u64) for memory efficiency
//! - Fast and safe iteration modes
//! - Optional hash caching to reduce recomputation
//! - Efficient freelist management for deleted slots
//! - Configurable load factor control
//! - Optional automatic garbage collection
//!
//! # Examples
//!
//! ```rust
//! use zipora::hash_map::GoldHashMap;
//!
//! let mut map = GoldHashMap::<String, i32>::new();
//! map.insert("hello".to_string(), 42);
//! assert_eq!(map.get(&"hello".to_string()), Some(&42));
//! ```

use crate::error::{Result, ZiporaError};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

/// Prime numbers used for bucket sizing
const PRIMES: &[usize] = &[
    5, 11, 23, 47, 97, 199, 409, 823, 1741, 3469, 6949, 14033, 28411,
    57557, 116731, 236897, 480881, 976369, 1982627, 4026031,
    8175383, 16601593, 33712729, 68460391, 139022417, 282312799,
    573292817, 1164186217,
];

/// Get next prime number greater than or equal to n
fn next_prime(n: usize) -> usize {
    for &prime in PRIMES {
        if prime >= n {
            return prime;
        }
    }
    // If larger than largest prime, use next power of 2
    n.next_power_of_two()
}

/// Link type trait for customizable index storage
///
/// Allows choosing between u32 (saves memory) and u64 (supports huge maps)
pub trait LinkType: Copy + Eq + PartialEq + Ord + PartialOrd + Default + std::fmt::Debug {
    /// Maximum valid index value
    const MAX: Self;
    /// Special marker for deleted entries
    const DELMARK: Self;
    /// Special marker for end of chain (tail)
    const TAIL: Self;

    /// Convert to usize for indexing
    fn as_usize(self) -> usize;

    /// Try to convert from usize, returns None if out of range
    fn from_usize(val: usize) -> Option<Self>;

    /// Check if this is a valid slot (not deleted or tail)
    fn is_valid(self) -> bool {
        self != Self::DELMARK && self != Self::TAIL
    }
}

impl LinkType for u32 {
    const MAX: Self = u32::MAX - 2;
    const DELMARK: Self = u32::MAX - 1;
    const TAIL: Self = u32::MAX;

    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(val: usize) -> Option<Self> {
        if val <= Self::MAX as usize {
            Some(val as u32)
        } else {
            None
        }
    }
}

impl LinkType for u64 {
    const MAX: Self = u64::MAX - 2;
    const DELMARK: Self = u64::MAX - 1;
    const TAIL: Self = u64::MAX;

    fn as_usize(self) -> usize {
        self as usize
    }

    fn from_usize(val: usize) -> Option<Self> {
        if val as u64 <= Self::MAX {
            Some(val as u64)
        } else {
            None
        }
    }
}

/// Iteration strategy for GoldHashMap
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationStrategy {
    /// Fast mode - direct array access, assumes all entries valid
    /// WARNING: May include deleted entries if not used carefully
    Fast,
    /// Safe mode - skip deleted entries (default)
    Safe,
}

/// Configuration for GoldHashMap
#[derive(Debug, Clone)]
pub struct GoldHashMapConfig {
    /// Initial capacity (will be rounded up to next prime)
    pub initial_capacity: usize,
    /// Load factor (0.0 to 0.999, default 0.7)
    pub load_factor: f32,
    /// Enable hash caching to reduce recomputation
    pub enable_hash_cache: bool,
    /// Enable automatic GC when freelist gets too large
    pub enable_auto_gc: bool,
    /// Enable freelist reuse for better memory efficiency
    pub enable_freelist_reuse: bool,
    /// Default iteration strategy
    pub default_iteration_strategy: IterationStrategy,
}

impl Default for GoldHashMapConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 16,
            load_factor: 0.7,
            enable_hash_cache: false,  // Disabled by default for memory efficiency
            enable_auto_gc: false,
            enable_freelist_reuse: true,
            default_iteration_strategy: IterationStrategy::Safe,
        }
    }
}

impl GoldHashMapConfig {
    /// Create config optimized for small maps (≤100 elements)
    pub fn small() -> Self {
        Self {
            initial_capacity: 16,
            load_factor: 0.7,
            enable_hash_cache: true,  // Hash caching beneficial for small maps
            enable_auto_gc: false,
            enable_freelist_reuse: true,
            default_iteration_strategy: IterationStrategy::Safe,
        }
    }

    /// Create config optimized for large maps (>100k elements)
    pub fn large() -> Self {
        Self {
            initial_capacity: 1024,
            load_factor: 0.7,
            enable_hash_cache: false,  // Save memory on large maps
            enable_auto_gc: true,      // Auto GC helpful for large maps
            enable_freelist_reuse: true,
            default_iteration_strategy: IterationStrategy::Safe,
        }
    }

    /// Create config optimized for high-churn workloads (frequent insert/delete)
    pub fn high_churn() -> Self {
        Self {
            initial_capacity: 64,
            load_factor: 0.6,  // Lower load factor reduces collisions
            enable_hash_cache: false,
            enable_auto_gc: true,  // Important for high churn
            enable_freelist_reuse: true,
            default_iteration_strategy: IterationStrategy::Safe,
        }
    }
}

/// Entry in the hash map
struct Entry<K, V, L: LinkType> {
    key: K,
    value: V,
    /// Link to next entry in collision chain
    link: L,
}

/// GoldHashMap - high-performance hash table
///
/// Generic over key type K, value type V, and link type L (u32 or u64)
pub struct GoldHashMap<K, V, L = u32>
where
    K: Hash + Eq,
    L: LinkType,
{
    /// Bucket array (indices into entries)
    buckets: Vec<L>,
    /// Entry storage (key-value pairs + links)
    entries: Vec<Entry<K, V, L>>,
    /// Optional hash cache (parallel to entries)
    hash_cache: Option<Vec<u64>>,
    /// Number of valid elements (excluding deleted)
    len: usize,
    /// Maximum elements before rehash
    max_load: usize,
    /// Freelist size (count of deleted slots)
    freelist_size: usize,
    /// Configuration
    config: GoldHashMapConfig,
}

impl<K, V, L> GoldHashMap<K, V, L>
where
    K: Hash + Eq + Clone,
    V: Clone,
    L: LinkType,
{
    /// Create new GoldHashMap with default configuration
    pub fn new() -> Self {
        Self::with_config(GoldHashMapConfig::default())
    }

    /// Create GoldHashMap with custom configuration
    pub fn with_config(mut config: GoldHashMapConfig) -> Self {
        // SAFETY FIX: Validate and clamp load_factor to prevent division by zero
        // and other arithmetic issues. Valid range is (0.0, 1.0).
        if config.load_factor <= 0.0 || config.load_factor >= 1.0 || !config.load_factor.is_finite() {
            // Invalid load_factor (≤0, ≥1, NaN, or infinity) - use safe default
            config.load_factor = 0.7;
        }

        let cap = next_prime(config.initial_capacity.max(5));
        let max_load = (cap as f32 * config.load_factor) as usize;

        Self {
            buckets: vec![L::TAIL; cap],
            entries: Vec::with_capacity(cap),
            hash_cache: if config.enable_hash_cache {
                Some(Vec::with_capacity(cap))
            } else {
                None
            },
            len: 0,
            max_load,
            freelist_size: 0,
            config,
        }
    }

    /// Insert key-value pair, returns old value if key existed
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        // Check if rehash needed before insertion
        if self.len >= self.max_load {
            self.rehash(self.buckets.len() + 1)?;
        }

        let hash = self.hash_key(&key);
        let bucket_idx = (hash as usize) % self.buckets.len();

        // Check if key exists in collision chain
        let mut link = self.buckets[bucket_idx];
        while link != L::TAIL {
            let idx = link.as_usize();
            if link.is_valid() && self.entries[idx].key == key {
                // Update existing entry
                let old = self.entries[idx].value.clone();
                self.entries[idx].value = value;
                return Ok(Some(old));
            }
            link = self.entries[idx].link;
        }

        // Insert new entry
        let entry_idx = self.allocate_slot();

        // SAFETY FIX: Check capacity before converting to link type
        if entry_idx > L::MAX.as_usize() {
            return Err(ZiporaError::resource_exhausted(
                format!("HashMap entry index {} exceeds link type capacity (max {}). Consider using GoldHashMap<K, V, u64> for larger maps.",
                        entry_idx, L::MAX.as_usize())
            ));
        }

        // Update entry data - either overwrite existing or push new
        if entry_idx < self.entries.len() {
            // Reusing an existing slot from freelist
            self.entries[entry_idx].key = key;
            self.entries[entry_idx].value = value;
            self.entries[entry_idx].link = self.buckets[bucket_idx];
        } else {
            // Creating a new entry
            self.entries.push(Entry {
                key,
                value,
                link: self.buckets[bucket_idx],
            });
        }

        // Link into bucket chain (insert at head)
        self.buckets[bucket_idx] = L::from_usize(entry_idx)
            .expect("Capacity check above ensures this succeeds");

        // Update hash cache if enabled
        if let Some(ref mut cache) = self.hash_cache {
            cache[entry_idx] = hash;
        }

        self.len += 1;
        Ok(None)
    }

    /// Get reference to value by key
    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = self.hash_key(key);
        let bucket_idx = (hash as usize) % self.buckets.len();

        let mut link = self.buckets[bucket_idx];
        while link != L::TAIL {
            let idx = link.as_usize();
            if link.is_valid() && self.entries[idx].key == *key {
                return Some(&self.entries[idx].value);
            }
            link = self.entries[idx].link;
        }
        None
    }

    /// Get mutable reference to value by key
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let hash = self.hash_key(key);
        let bucket_idx = (hash as usize) % self.buckets.len();

        let mut link = self.buckets[bucket_idx];
        while link != L::TAIL {
            let idx = link.as_usize();
            if link.is_valid() && self.entries[idx].key == *key {
                // Safe: we have unique mutable access to self
                return Some(unsafe { &mut *(&mut self.entries[idx].value as *mut V) });
            }
            link = self.entries[idx].link;
        }
        None
    }

    /// Remove key-value pair, returns value if existed
    pub fn remove(&mut self, key: &K) -> Result<Option<V>> {
        let hash = self.hash_key(key);
        let bucket_idx = (hash as usize) % self.buckets.len();

        let mut prev: Option<usize> = None;
        let mut link = self.buckets[bucket_idx];

        while link != L::TAIL {
            let idx = link.as_usize();
            if link.is_valid() && self.entries[idx].key == *key {
                // Found - remove it
                let old_value = self.entries[idx].value.clone();
                let next_link = self.entries[idx].link;

                // Unlink from chain
                if let Some(prev_idx) = prev {
                    self.entries[prev_idx].link = next_link;
                } else {
                    self.buckets[bucket_idx] = next_link;
                }

                // Mark as deleted and add to freelist
                self.free_slot(idx);
                self.len -= 1;

                // Auto GC if enabled and freelist too large
                if self.config.enable_auto_gc && self.freelist_size > self.len / 2 {
                    self.revoke_deleted()?;
                }

                return Ok(Some(old_value));
            }
            prev = Some(idx);
            link = self.entries[idx].link;
        }
        Ok(None)
    }

    /// Check if map contains key
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Number of elements (excluding deleted)
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get bucket count
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Get current load factor
    pub fn load_factor(&self) -> f32 {
        if self.buckets.is_empty() {
            0.0
        } else {
            self.len as f32 / self.buckets.len() as f32
        }
    }

    /// Get number of deleted entries in freelist
    pub fn deleted_count(&self) -> usize {
        self.freelist_size
    }

    /// Iterate with specific strategy
    pub fn iter_with_strategy(&self, strategy: IterationStrategy) -> GoldHashMapIter<K, V, L> {
        GoldHashMapIter {
            map: self,
            index: 0,
            strategy,
        }
    }

    /// Default safe iteration (skips deleted entries)
    pub fn iter(&self) -> GoldHashMapIter<K, V, L> {
        self.iter_with_strategy(self.config.default_iteration_strategy)
    }

    /// Fast iteration (no deleted entry checks)
    /// WARNING: May include deleted entries, use only if you know the map has no deletions
    pub fn iter_fast(&self) -> GoldHashMapIter<K, V, L> {
        self.iter_with_strategy(IterationStrategy::Fast)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.buckets.fill(L::TAIL);
        self.entries.clear();
        if let Some(ref mut cache) = self.hash_cache {
            cache.clear();
        }
        self.len = 0;
        self.freelist_size = 0;
    }

    /// Compact the map by removing all deleted entries
    /// This will invalidate all indices but improve iteration performance
    pub fn revoke_deleted(&mut self) -> Result<()> {
        if self.freelist_size == 0 {
            return Ok(());
        }

        // Compact entries by removing deleted ones
        let mut write_idx = 0;
        for read_idx in 0..self.entries.len() {
            if self.entries[read_idx].link != L::DELMARK {
                if write_idx != read_idx {
                    self.entries[write_idx] = Entry {
                        key: self.entries[read_idx].key.clone(),
                        value: self.entries[read_idx].value.clone(),
                        link: L::TAIL,  // Will be fixed in relink
                    };
                    if let Some(ref mut cache) = self.hash_cache {
                        cache[write_idx] = cache[read_idx];
                    }
                }
                write_idx += 1;
            }
        }

        // Truncate to compacted size
        self.entries.truncate(write_idx);
        if let Some(ref mut cache) = self.hash_cache {
            cache.truncate(write_idx);
        }

        // Reset freelist
        self.freelist_size = 0;

        // Rebuild bucket links
        self.relink()?;

        Ok(())
    }

    /// Enable or disable hash caching at runtime
    pub fn set_hash_caching(&mut self, enabled: bool) {
        if enabled && self.hash_cache.is_none() {
            // Build cache
            let mut cache = Vec::with_capacity(self.entries.len());
            for entry in &self.entries {
                cache.push(self.hash_key(&entry.key));
            }
            self.hash_cache = Some(cache);
            self.config.enable_hash_cache = true;
        } else if !enabled && self.hash_cache.is_some() {
            self.hash_cache = None;
            self.config.enable_hash_cache = false;
        }
    }

    /// Check if hash caching is enabled
    pub fn is_hash_cached(&self) -> bool {
        self.hash_cache.is_some()
    }

    /// Reserve capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let new_capacity = self.len + additional;
        if new_capacity > self.max_load {
            let new_buckets = next_prime((new_capacity as f32 / self.config.load_factor) as usize);
            self.rehash(new_buckets)?;
        }
        Ok(())
    }

    // Internal: Allocate a slot (reuse from freelist or create new)
    fn allocate_slot(&mut self) -> usize {
        if self.config.enable_freelist_reuse && self.freelist_size > 0 {
            // Scan for a deleted slot to reuse
            for i in 0..self.entries.len() {
                if self.entries[i].link == L::DELMARK {
                    self.freelist_size -= 1;
                    return i;
                }
            }
            // If we didn't find one (shouldn't happen), fall through to create new
        }

        // Append new entry - we'll overwrite it immediately in insert()
        let idx = self.entries.len();

        // Ensure capacity
        if idx >= self.entries.capacity() {
            let new_cap = (idx + 1).next_power_of_two();
            self.entries.reserve(new_cap - self.entries.len());
            if let Some(ref mut cache) = self.hash_cache {
                cache.reserve(new_cap - cache.len());
            }
        }

        if let Some(ref mut cache) = self.hash_cache {
            cache.push(0);
        }

        idx
    }

    // Internal: Free a slot (add to freelist)
    fn free_slot(&mut self, idx: usize) {
        // Mark as deleted - this is critical for safe iteration
        // We MUST always mark with DELMARK so iteration can skip it
        if self.config.enable_freelist_reuse {
            // Add to freelist - we store the next pointer differently
            // To maintain DELMARK in link field, we'll track freelist separately
            // For now, just mark as deleted and increment count
            self.entries[idx].link = L::DELMARK;

            // Store freelist in a different way - we can't override link
            // because we need DELMARK to stay for iteration safety
            // So we build the freelist during allocation by scanning
            self.freelist_size += 1;
        } else {
            self.entries[idx].link = L::DELMARK;
            self.freelist_size += 1;
        }
    }

    // Internal: Rehash to new bucket count
    fn rehash(&mut self, new_size: usize) -> Result<()> {
        let new_size = next_prime(new_size.max(5));
        if new_size == self.buckets.len() {
            return Ok(());
        }

        self.buckets = vec![L::TAIL; new_size];
        self.max_load = (new_size as f32 * self.config.load_factor) as usize;

        // Relink all valid entries
        self.relink()?;

        Ok(())
    }

    // Internal: Rebuild all bucket links
    fn relink(&mut self) -> Result<()> {
        // Clear buckets
        self.buckets.fill(L::TAIL);

        // Relink all valid entries
        for i in 0..self.entries.len() {
            if self.entries[i].link != L::DELMARK {
                let hash = if let Some(ref cache) = self.hash_cache {
                    cache[i]
                } else {
                    self.hash_key(&self.entries[i].key)
                };

                let bucket_idx = (hash as usize) % self.buckets.len();
                self.entries[i].link = self.buckets[bucket_idx];

                // SAFETY FIX: Check capacity before converting to link type
                if i > L::MAX.as_usize() {
                    return Err(ZiporaError::resource_exhausted(
                        format!("HashMap entry index {} exceeds link type capacity (max {}). Consider using GoldHashMap<K, V, u64> for larger maps.",
                                i, L::MAX.as_usize())
                    ));
                }

                self.buckets[bucket_idx] = L::from_usize(i)
                    .expect("Capacity check above ensures this succeeds");
            }
        }

        Ok(())
    }

    // Internal: Hash a key
    fn hash_key(&self, key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl<K, V, L> Default for GoldHashMap<K, V, L>
where
    K: Hash + Eq + Clone,
    V: Clone,
    L: LinkType,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for GoldHashMap
pub struct GoldHashMapIter<'a, K, V, L: LinkType>
where
    K: Hash + Eq,
{
    map: &'a GoldHashMap<K, V, L>,
    index: usize,
    strategy: IterationStrategy,
}

impl<'a, K, V, L> Iterator for GoldHashMapIter<'a, K, V, L>
where
    K: Hash + Eq + Clone,
    V: Clone,
    L: LinkType,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match self.strategy {
            IterationStrategy::Fast => {
                // Fast mode - no validity checks
                if self.index < self.map.entries.len() {
                    let entry = &self.map.entries[self.index];
                    self.index += 1;
                    Some((&entry.key, &entry.value))
                } else {
                    None
                }
            }
            IterationStrategy::Safe => {
                // Safe mode - skip deleted entries
                while self.index < self.map.entries.len() {
                    let entry = &self.map.entries[self.index];
                    self.index += 1;
                    if entry.link != L::DELMARK {
                        return Some((&entry.key, &entry.value));
                    }
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_get() {
        let mut map = GoldHashMap::<String, i32>::new();
        map.insert("hello".to_string(), 42).unwrap();
        assert_eq!(map.get(&"hello".to_string()), Some(&42));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_update_existing() {
        let mut map = GoldHashMap::<String, i32>::new();
        assert_eq!(map.insert("key".to_string(), 1).unwrap(), None);
        assert_eq!(map.insert("key".to_string(), 2).unwrap(), Some(1));
        assert_eq!(map.get(&"key".to_string()), Some(&2));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut map = GoldHashMap::<String, i32>::new();
        map.insert("key".to_string(), 42).unwrap();
        assert_eq!(map.remove(&"key".to_string()).unwrap(), Some(42));
        assert_eq!(map.get(&"key".to_string()), None);
        assert_eq!(map.len(), 0);
        assert_eq!(map.deleted_count(), 1);
    }

    #[test]
    fn test_iteration_safe() {
        let mut map = GoldHashMap::<i32, String>::new();
        map.insert(1, "one".to_string()).unwrap();
        map.insert(2, "two".to_string()).unwrap();
        map.insert(3, "three".to_string()).unwrap();

        let items: Vec<_> = map.iter().map(|(k, v)| (*k, v.clone())).collect();
        assert_eq!(items.len(), 3);
        assert!(items.contains(&(1, "one".to_string())));
        assert!(items.contains(&(2, "two".to_string())));
        assert!(items.contains(&(3, "three".to_string())));
    }

    #[test]
    fn test_iteration_with_deletions() {
        let mut map = GoldHashMap::<i32, String>::new();
        map.insert(1, "one".to_string()).unwrap();
        map.insert(2, "two".to_string()).unwrap();
        map.insert(3, "three".to_string()).unwrap();
        map.remove(&2).unwrap();

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items.len(), 2);
        assert!(items.contains(&1));
        assert!(items.contains(&3));
        assert!(!items.contains(&2));
    }

    #[test]
    fn test_iteration_fast() {
        let mut map = GoldHashMap::<i32, String>::new();
        map.insert(1, "one".to_string()).unwrap();
        map.insert(2, "two".to_string()).unwrap();

        let items: Vec<_> = map.iter_fast().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_hash_caching_enabled() {
        let config = GoldHashMapConfig {
            enable_hash_cache: true,
            ..Default::default()
        };
        let mut map = GoldHashMap::<i32, String>::with_config(config);
        map.insert(1, "one".to_string()).unwrap();
        assert!(map.is_hash_cached());
    }

    #[test]
    fn test_hash_caching_disabled() {
        let config = GoldHashMapConfig {
            enable_hash_cache: false,
            ..Default::default()
        };
        let map = GoldHashMap::<i32, String>::with_config(config);
        assert!(!map.is_hash_cached());
    }

    #[test]
    fn test_link_type_u32() {
        let mut map = GoldHashMap::<i32, String, u32>::new();
        for i in 0..1000 {
            map.insert(i, format!("value{}", i)).unwrap();
        }
        assert_eq!(map.len(), 1000);
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }
    }

    #[test]
    fn test_link_type_u64() {
        let mut map = GoldHashMap::<i32, String, u64>::new();
        for i in 0..1000 {
            map.insert(i, format!("value{}", i)).unwrap();
        }
        assert_eq!(map.len(), 1000);
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }
    }

    #[test]
    fn test_rehash() {
        let mut map = GoldHashMap::<i32, i32>::new();
        let initial_capacity = map.capacity();

        for i in 0..100 {
            map.insert(i, i * 2).unwrap();
        }

        assert!(map.capacity() > initial_capacity);

        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_freelist_reuse() {
        let mut map = GoldHashMap::<i32, i32>::new();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        map.insert(3, 30).unwrap();

        let entries_count = map.entries.len();

        map.remove(&1).unwrap();
        assert_eq!(map.len(), 2);
        assert_eq!(map.deleted_count(), 1);

        map.insert(4, 40).unwrap();  // Should reuse slot from removed entry
        assert_eq!(map.len(), 3);
        assert_eq!(map.deleted_count(), 0);
        assert_eq!(map.entries.len(), entries_count);  // No new allocation
    }

    #[test]
    fn test_large_dataset() {
        let mut map = GoldHashMap::<i32, i32>::new();
        for i in 0..10_000 {
            map.insert(i, i).unwrap();
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn test_contains_key() {
        let mut map = GoldHashMap::<String, i32>::new();
        map.insert("exists".to_string(), 42).unwrap();

        assert!(map.contains_key(&"exists".to_string()));
        assert!(!map.contains_key(&"missing".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut map = GoldHashMap::<i32, i32>::new();
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert_eq!(map.len(), 2);

        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.deleted_count(), 0);
    }

    #[test]
    fn test_revoke_deleted() {
        let mut map = GoldHashMap::<i32, i32>::new();
        for i in 0..10 {
            map.insert(i, i * 10).unwrap();
        }

        // Remove half the elements
        for i in 0..5 {
            map.remove(&i).unwrap();
        }

        assert_eq!(map.len(), 5);
        assert_eq!(map.deleted_count(), 5);

        map.revoke_deleted().unwrap();

        assert_eq!(map.len(), 5);
        assert_eq!(map.deleted_count(), 0);

        // Verify remaining elements still accessible
        for i in 5..10 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_auto_gc() {
        let config = GoldHashMapConfig {
            enable_auto_gc: true,
            ..Default::default()
        };
        let mut map = GoldHashMap::<i32, i32>::with_config(config);

        // Insert many elements
        for i in 0..100 {
            map.insert(i, i).unwrap();
        }

        // Delete most of them to trigger auto GC
        for i in 0..80 {
            map.remove(&i).unwrap();
        }

        assert_eq!(map.len(), 20);
        // Auto GC should have kicked in
        assert!(map.deleted_count() < 40);  // Should be much less than 80
    }

    #[test]
    fn test_get_mut() {
        let mut map = GoldHashMap::<String, i32>::new();
        map.insert("key".to_string(), 42).unwrap();

        if let Some(value) = map.get_mut(&"key".to_string()) {
            *value = 100;
        }

        assert_eq!(map.get(&"key".to_string()), Some(&100));
    }

    #[test]
    fn test_reserve() {
        let mut map = GoldHashMap::<i32, i32>::new();
        let initial_capacity = map.capacity();

        map.reserve(1000).unwrap();
        assert!(map.capacity() > initial_capacity);

        // Should be able to insert without rehashing
        for i in 0..1000 {
            map.insert(i, i).unwrap();
        }
    }

    #[test]
    fn test_runtime_hash_caching_toggle() {
        let mut map = GoldHashMap::<i32, String>::new();
        assert!(!map.is_hash_cached());

        map.insert(1, "one".to_string()).unwrap();
        map.insert(2, "two".to_string()).unwrap();

        map.set_hash_caching(true);
        assert!(map.is_hash_cached());

        map.insert(3, "three".to_string()).unwrap();
        assert_eq!(map.get(&3), Some(&"three".to_string()));

        map.set_hash_caching(false);
        assert!(!map.is_hash_cached());

        // Should still work after disabling
        assert_eq!(map.get(&1), Some(&"one".to_string()));
    }

    #[test]
    fn test_load_factor() {
        let config = GoldHashMapConfig {
            initial_capacity: 100,
            load_factor: 0.5,
            ..Default::default()
        };
        let mut map = GoldHashMap::<i32, i32>::with_config(config);

        for i in 0..40 {
            map.insert(i, i).unwrap();
        }

        let lf = map.load_factor();
        assert!(lf > 0.0 && lf <= 0.5);
    }

    #[test]
    fn test_config_small() {
        let map = GoldHashMap::<i32, i32>::with_config(GoldHashMapConfig::small());
        assert!(map.is_hash_cached());  // Small config enables hash caching
    }

    #[test]
    fn test_config_large() {
        let map = GoldHashMap::<i32, i32>::with_config(GoldHashMapConfig::large());
        assert!(!map.is_hash_cached());  // Large config disables hash caching
        assert!(map.capacity() >= 1024);
    }

    #[test]
    fn test_config_high_churn() {
        let config = GoldHashMapConfig::high_churn();
        assert!(config.load_factor < 0.7);  // Lower load factor
        assert!(config.enable_auto_gc);     // Auto GC enabled
    }
}
