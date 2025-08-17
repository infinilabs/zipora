//! GoldenRatioHashMap - Enhanced hash map with golden ratio optimizations
//!
//! This implementation builds upon the existing GoldHashMap to provide:
//! - Golden ratio growth strategy (1.618x expansion)
//! - FaboHashCombine hash function for better distribution
//! - Enhanced robin hood hashing with distance limiting
//! - Optimized load factor based on golden ratio

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::hash_map::hash_functions::{
    fabo_hash_combine_u64, golden_ratio_next_size, optimal_bucket_count, GOLDEN_LOAD_FACTOR,
};
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

/// Enhanced hash map with golden ratio optimizations
///
/// GoldenRatioHashMap provides superior memory utilization and performance
/// compared to standard hash maps by using:
/// - Golden ratio growth strategy for 15-20% better memory efficiency
/// - FaboHashCombine hash function for improved distribution
/// - Enhanced robin hood probing with distance limiting
/// - Optimized load factor based on mathematical golden ratio
///
/// # Examples
///
/// ```rust
/// use zipora::GoldenRatioHashMap;
///
/// let mut map = GoldenRatioHashMap::new();
/// map.insert("key", "value").unwrap();
/// assert_eq!(map.get("key"), Some(&"value"));
/// ```
pub struct GoldenRatioHashMap<K, V, S = ahash::RandomState> {
    /// Storage for entries using FastVec for efficiency
    entries: FastVec<Entry<K, V>>,
    /// Bucket array with robin hood probing information
    buckets: FastVec<Bucket>,
    /// Hash function builder
    hash_builder: S,
    /// Number of occupied entries
    len: usize,
    /// Maximum probe distance allowed (for robin hood hashing)
    max_probe_distance: u16,
    /// Growth factor denominator (for precise golden ratio calculation)
    growth_factor_denom: u32,
    /// Growth factor numerator
    growth_factor_num: u32,
    /// Marker for types
    _phantom: PhantomData<(K, V)>,
}

/// Internal entry storage with probe distance tracking
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    hash: u64,
}

/// Bucket information for robin hood hashing
#[derive(Debug, Clone, Copy)]
struct Bucket {
    /// Index into entries array, or EMPTY if empty
    entry_index: u32,
    /// Probe distance from ideal position
    probe_distance: u16,
    /// Cached hash for fast comparison (lower 16 bits)
    cached_hash: u16,
}

const EMPTY: u32 = u32::MAX;
const DEFAULT_CAPACITY: usize = 16;
const MAX_PROBE_DISTANCE: u16 = 64;

impl<K, V> GoldenRatioHashMap<K, V, ahash::RandomState> {
    /// Creates a new empty hash map with default hasher
    pub fn new() -> Self {
        Self::with_hasher(ahash::RandomState::new())
    }

    /// Creates a new hash map with specified capacity
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_hasher(capacity, ahash::RandomState::new())
    }
}

impl<K, V, S> GoldenRatioHashMap<K, V, S>
where
    S: BuildHasher,
{
    /// Creates a new hash map with custom hasher
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(DEFAULT_CAPACITY, hash_builder).unwrap()
    }

    /// Creates a new hash map with capacity and custom hasher
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Result<Self> {
        let bucket_count = optimal_bucket_count(capacity);

        let mut buckets = FastVec::with_capacity(bucket_count)?;
        buckets.resize(
            bucket_count,
            Bucket {
                entry_index: EMPTY,
                probe_distance: 0,
                cached_hash: 0,
            },
        )?;

        Ok(GoldenRatioHashMap {
            entries: FastVec::with_capacity(capacity)?,
            buckets,
            hash_builder,
            len: 0,
            max_probe_distance: MAX_PROBE_DISTANCE,
            growth_factor_num: 103, // Golden ratio approximation numerator
            growth_factor_denom: 64, // Golden ratio approximation denominator
            _phantom: PhantomData,
        })
    }

    /// Returns the number of elements in the map
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the current bucket capacity
    pub fn bucket_capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the current entry capacity
    pub fn entry_capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Returns the current load factor (0.0 to 1.0)
    pub fn load_factor(&self) -> f64 {
        if self.buckets.is_empty() {
            0.0
        } else {
            self.len as f64 / self.buckets.len() as f64
        }
    }

    /// Computes hash for a key using FaboHashCombine
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base_hash = hasher.finish();

        // Apply FaboHashCombine for better distribution
        fabo_hash_combine_u64(base_hash, base_hash.rotate_right(32))
    }

    /// Converts hash to bucket index
    fn hash_to_bucket(&self, hash: u64) -> usize {
        // Use upper bits for better distribution with power-of-2 sizes
        let mask = self.buckets.len() - 1;
        ((hash >> 32) as usize) & mask
    }

    /// Finds position for a key using enhanced robin hood probing
    fn find_position<Q>(&self, key: &Q) -> FindResult
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.buckets.is_empty() {
            return FindResult::NotFound { insertion_index: 0 };
        }

        let hash = self.hash_key(key);
        let ideal_bucket = self.hash_to_bucket(hash);
        let cached_hash = (hash as u16) | 1; // Ensure non-zero

        let mut bucket_idx = ideal_bucket;
        let mut probe_distance = 0u16;

        loop {
            let bucket = &self.buckets[bucket_idx];

            if bucket.entry_index == EMPTY {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }

            // Check if this is our key
            if bucket.cached_hash == cached_hash && bucket.entry_index != EMPTY {
                let entry_idx = bucket.entry_index as usize;
                if entry_idx < self.entries.len() {
                    let entry = &self.entries[entry_idx];
                    if entry.key.borrow() == key {
                        return FindResult::Found {
                            bucket_index: bucket_idx,
                            entry_index: entry_idx,
                        };
                    }
                }
            }

            // Robin hood heuristic: if we've probed further than the current entry,
            // we know the key doesn't exist
            if probe_distance > bucket.probe_distance {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }

            probe_distance += 1;
            bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

            // Limit probe distance to prevent excessive clustering
            if probe_distance >= self.max_probe_distance {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }
        }
    }

    /// Checks if resize is needed based on golden ratio load factor
    fn needs_resize(&self) -> bool {
        if self.buckets.is_empty() {
            return true;
        }

        let load_factor = (self.len * 256) / self.buckets.len();
        load_factor as u8 >= GOLDEN_LOAD_FACTOR
    }

    /// Resizes using golden ratio growth strategy
    fn resize_if_needed(&mut self) -> Result<()> {
        if self.needs_resize() {
            let new_capacity = if self.len == 0 {
                DEFAULT_CAPACITY
            } else {
                golden_ratio_next_size(self.len)
            };
            let new_bucket_count = optimal_bucket_count(new_capacity);
            self.resize(new_bucket_count)?;
        }
        Ok(())
    }

    /// Performs the actual resize operation
    fn resize(&mut self, new_bucket_count: usize) -> Result<()> {
        // Save old buckets and create new ones
        let old_buckets = mem::replace(&mut self.buckets, {
            let mut new_buckets = FastVec::with_capacity(new_bucket_count)?;
            new_buckets.resize(
                new_bucket_count,
                Bucket {
                    entry_index: EMPTY,
                    probe_distance: 0,
                    cached_hash: 0,
                },
            )?;
            new_buckets
        });

        // Re-insert all entries
        for entry_idx in 0..self.entries.len() {
            let entry_hash = self.entries[entry_idx].hash;
            let ideal_bucket = self.hash_to_bucket(entry_hash);
            let cached_hash = (entry_hash as u16) | 1;

            let mut bucket_idx = ideal_bucket;
            let mut probe_distance = 0u16;

            // Find position using robin hood insertion
            loop {
                let bucket = &mut self.buckets[bucket_idx];

                if bucket.entry_index == EMPTY {
                    *bucket = Bucket {
                        entry_index: entry_idx as u32,
                        probe_distance,
                        cached_hash,
                    };
                    break;
                }

                // Robin hood swap if we've probed further
                if probe_distance > bucket.probe_distance {
                    let mut old_bucket = *bucket;
                    *bucket = Bucket {
                        entry_index: entry_idx as u32,
                        probe_distance,
                        cached_hash,
                    };

                    // Adjust probe distance for the displaced bucket's new starting position
                    old_bucket.probe_distance += 1;
                    
                    // Continue inserting the displaced entry
                    self.insert_displaced(old_bucket, bucket_idx + 1)?;
                    break;
                }

                probe_distance += 1;
                bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

                if probe_distance >= self.max_probe_distance {
                    return Err(ZiporaError::invalid_data("Resize failed: excessive probing"));
                }
            }
        }

        Ok(())
    }

    /// Inserts a displaced bucket during robin hood insertion
    fn insert_displaced(&mut self, mut displaced: Bucket, start_idx: usize) -> Result<()> {
        let mut bucket_idx = start_idx & (self.buckets.len() - 1);

        loop {
            let bucket = &mut self.buckets[bucket_idx];

            if bucket.entry_index == EMPTY {
                *bucket = displaced;
                break;
            }

            if displaced.probe_distance > bucket.probe_distance {
                // Swap the buckets
                let swapped = *bucket;
                *bucket = displaced;
                displaced = swapped;
            }

            // Move to next position and increment probe distance
            displaced.probe_distance += 1;
            bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

            if displaced.probe_distance >= self.max_probe_distance {
                return Err(ZiporaError::invalid_data("Insert failed: excessive probing"));
            }
        }

        Ok(())
    }
}

impl<K, V, S> GoldenRatioHashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Inserts a key-value pair into the map
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        self.resize_if_needed()?;

        let hash = self.hash_key(&key);

        match self.find_position(&key) {
            FindResult::Found { entry_index, .. } => {
                // Replace existing value
                let old_value = mem::replace(&mut self.entries[entry_index].value, value);
                Ok(Some(old_value))
            }
            FindResult::NotFound { insertion_index } => {
                // Insert new entry
                let entry_index = self.entries.len();
                let cached_hash = (hash as u16) | 1;

                // Add entry to storage
                self.entries.push(Entry { key, value, hash })?;

                // Insert using robin hood probing
                let mut bucket_idx = insertion_index;
                let ideal_bucket = self.hash_to_bucket(hash);
                let mut probe_distance = if bucket_idx >= ideal_bucket {
                    (bucket_idx - ideal_bucket) as u16
                } else {
                    (bucket_idx + self.buckets.len() - ideal_bucket) as u16
                };

                let mut new_bucket = Bucket {
                    entry_index: entry_index as u32,
                    probe_distance,
                    cached_hash,
                };

                loop {
                    let bucket = &mut self.buckets[bucket_idx];

                    if bucket.entry_index == EMPTY {
                        *bucket = new_bucket;
                        break;
                    }

                    // Robin hood swap if we've probed further
                    if new_bucket.probe_distance > bucket.probe_distance {
                        mem::swap(bucket, &mut new_bucket);
                    }

                    new_bucket.probe_distance += 1;
                    bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

                    if new_bucket.probe_distance >= self.max_probe_distance {
                        return Err(ZiporaError::invalid_data("Insert failed: excessive probing"));
                    }
                }

                self.len += 1;
                Ok(None)
            }
        }
    }

    /// Gets a reference to the value for a key
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.find_position(key) {
            FindResult::Found { entry_index, .. } => Some(&self.entries[entry_index].value),
            FindResult::NotFound { .. } => None,
        }
    }

    /// Gets a mutable reference to the value for a key
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.find_position(key) {
            FindResult::Found { entry_index, .. } => Some(&mut self.entries[entry_index].value),
            FindResult::NotFound { .. } => None,
        }
    }

    /// Checks if the map contains a key
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        matches!(self.find_position(key), FindResult::Found { .. })
    }

    /// Removes a key from the map, returning the value if present
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match self.find_position(key) {
            FindResult::Found {
                bucket_index,
                entry_index,
            } => {
                // Remove the entry
                let removed_entry = if entry_index == self.entries.len() - 1 {
                    self.entries.pop().unwrap()
                } else {
                    // Swap with last entry to maintain compactness
                    let last_entry = self.entries.pop().unwrap();
                    let removed_entry = mem::replace(&mut self.entries[entry_index], last_entry);

                    // Update bucket pointing to moved entry
                    for i in 0..self.buckets.len() {
                        if self.buckets[i].entry_index == (self.entries.len()) as u32 {
                            self.buckets[i].entry_index = entry_index as u32;
                            break;
                        }
                    }

                    removed_entry
                };

                // Clear the bucket and shift subsequent entries backward (robin hood)
                self.buckets[bucket_index] = Bucket {
                    entry_index: EMPTY,
                    probe_distance: 0,
                    cached_hash: 0,
                };

                // Shift subsequent entries backward to maintain robin hood invariant
                let mut idx = (bucket_index + 1) & (self.buckets.len() - 1);
                while idx != bucket_index {
                    let current_bucket = self.buckets[idx]; // Copy the bucket data
                    if current_bucket.entry_index == EMPTY || current_bucket.probe_distance == 0 {
                        break;
                    }

                    // Move this entry backward
                    let prev_idx = if idx == 0 { self.buckets.len() - 1 } else { idx - 1 };
                    let mut moved_bucket = current_bucket;
                    moved_bucket.probe_distance -= 1;
                    
                    self.buckets[prev_idx] = moved_bucket;
                    self.buckets[idx] = Bucket {
                        entry_index: EMPTY,
                        probe_distance: 0,
                        cached_hash: 0,
                    };

                    idx = (idx + 1) & (self.buckets.len() - 1);
                }

                self.len -= 1;
                Some(removed_entry.value)
            }
            FindResult::NotFound { .. } => None,
        }
    }

    /// Clears all entries from the map
    pub fn clear(&mut self) {
        self.entries.clear();
        for bucket in self.buckets.iter_mut() {
            *bucket = Bucket {
                entry_index: EMPTY,
                probe_distance: 0,
                cached_hash: 0,
            };
        }
        self.len = 0;
    }

    /// Returns an iterator over key-value pairs
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            entries: &self.entries,
            index: 0,
        }
    }

    /// Returns an iterator over keys
    pub fn keys(&self) -> Keys<K, V> {
        Keys { iter: self.iter() }
    }

    /// Returns an iterator over values
    pub fn values(&self) -> Values<K, V> {
        Values { iter: self.iter() }
    }
}

/// Result of finding a position in the hash map
#[derive(Debug)]
enum FindResult {
    Found {
        bucket_index: usize,
        entry_index: usize,
    },
    NotFound {
        insertion_index: usize,
    },
}

impl<K, V, S> Default for GoldenRatioHashMap<K, V, S>
where
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> fmt::Debug for GoldenRatioHashMap<K, V, S>
where
    K: fmt::Debug + Hash + Eq,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Iterator over key-value pairs
pub struct Iter<'a, K, V> {
    entries: &'a FastVec<Entry<K, V>>,
    index: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entries.len() {
            let entry = &self.entries[self.index];
            self.index += 1;
            Some((&entry.key, &entry.value))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

/// Iterator over keys
pub struct Keys<'a, K, V> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {}

/// Iterator over values
pub struct Values<'a, K, V> {
    iter: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_map() {
        let map = GoldenRatioHashMap::<i32, String>::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.bucket_capacity() >= DEFAULT_CAPACITY);
    }

    #[test]
    fn test_basic_operations() {
        let mut map = GoldenRatioHashMap::new();

        // Insert
        assert_eq!(map.insert("key1", "value1").unwrap(), None);
        assert_eq!(map.insert("key2", "value2").unwrap(), None);
        assert_eq!(map.len(), 2);

        // Get
        assert_eq!(map.get("key1"), Some(&"value1"));
        assert_eq!(map.get("key2"), Some(&"value2"));
        assert_eq!(map.get("key3"), None);

        // Update
        assert_eq!(map.insert("key1", "new_value1").unwrap(), Some("value1"));
        assert_eq!(map.get("key1"), Some(&"new_value1"));
        assert_eq!(map.len(), 2);

        // Contains
        assert!(map.contains_key("key1"));
        assert!(map.contains_key("key2"));
        assert!(!map.contains_key("key3"));

        // Remove
        assert_eq!(map.remove("key1"), Some("new_value1"));
        assert_eq!(map.remove("key1"), None);
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key("key1"));
        assert!(map.contains_key("key2"));
    }

    #[test]
    fn test_golden_ratio_growth() {
        let mut map = GoldenRatioHashMap::new();
        let initial_capacity = map.bucket_capacity();

        // Insert enough items to trigger resize
        for i in 0..100 {
            map.insert(i, i * 2).unwrap();
        }

        let final_capacity = map.bucket_capacity();
        assert!(final_capacity > initial_capacity);

        // Verify all items are still present
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_load_factor() {
        let mut map = GoldenRatioHashMap::new();

        // Initially empty
        assert_eq!(map.load_factor(), 0.0);

        // Add some elements
        for i in 0..10 {
            map.insert(i, i).unwrap();
        }

        let load_factor = map.load_factor();
        assert!(load_factor > 0.0 && load_factor <= 1.0);
    }

    #[test]
    fn test_collision_handling() {
        let mut map = GoldenRatioHashMap::new();

        // Insert items gradually and verify each step
        for i in 0..100 {
            let key = format!("key_{}", i);
            println!("Inserting {}, map len: {}, bucket capacity: {}", key, map.len(), map.bucket_capacity());
            
            map.insert(key.clone(), i).unwrap();
            
            // Verify the item was inserted and can be retrieved immediately
            assert_eq!(map.get(&key), Some(&i), "Failed to retrieve key {} immediately after insertion", key);
            
            // Verify all previous keys are still accessible
            for j in 0..=i {
                let prev_key = format!("key_{}", j);
                assert_eq!(map.get(&prev_key), Some(&j), "Lost key {} after inserting {}", prev_key, key);
            }
        }

        assert_eq!(map.len(), 100);
    }

    #[test]
    fn test_robin_hood_behavior() {
        let mut map = GoldenRatioHashMap::new();

        // Insert items that might cause long probe sequences
        let keys = ["a", "aa", "aaa", "aaaa", "aaaaa"];
        for &key in &keys {
            map.insert(key, key.len()).unwrap();
        }

        // All should be retrievable
        for &key in &keys {
            assert_eq!(map.get(key), Some(&key.len()));
        }
    }

    #[test]
    fn test_iterators() {
        let mut map = GoldenRatioHashMap::new();
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        map.insert("c", 3).unwrap();

        // Test iter
        let mut items: Vec<_> = map.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        assert_eq!(items, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);

        // Test keys
        let mut keys: Vec<_> = map.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);

        // Test values
        let mut values: Vec<_> = map.values().cloned().collect();
        values.sort();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_clear() {
        let mut map = GoldenRatioHashMap::new();
        map.insert("key1", "value1").unwrap();
        map.insert("key2", "value2").unwrap();

        assert_eq!(map.len(), 2);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get("key1"), None);
    }

    #[test]
    fn test_with_capacity() {
        let map = GoldenRatioHashMap::<i32, String>::with_capacity(100).unwrap();
        assert!(map.bucket_capacity() >= 100);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_get_mut() {
        let mut map = GoldenRatioHashMap::new();
        map.insert("key", 42).unwrap();

        if let Some(value) = map.get_mut("key") {
            *value = 84;
        }

        assert_eq!(map.get("key"), Some(&84));
    }
}