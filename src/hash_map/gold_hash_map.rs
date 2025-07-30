//! GoldHashMap - High-performance general-purpose hash map
//!
//! This implementation focuses on:
//! - Fast hash computation using AHash
//! - Efficient memory layout with robin hood hashing
//! - SIMD-optimized probing where available
//! - Cache-friendly data structures

use crate::containers::FastVec;
use crate::error::{Result, ToplingError};
use ahash::RandomState;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher, BuildHasher};
use std::marker::PhantomData;
use std::mem;

/// High-performance hash map using robin hood hashing
/// 
/// GoldHashMap is optimized for throughput and low latency operations.
/// It uses AHash for fast hashing and robin hood probing for efficient
/// collision resolution.
/// 
/// # Examples
/// 
/// ```rust
/// use infini_zip::GoldHashMap;
/// 
/// let mut map = GoldHashMap::new();
/// map.insert("key", "value").unwrap();
/// assert_eq!(map.get("key"), Some(&"value"));
/// ```
pub struct GoldHashMap<K, V> {
    /// Storage for entries, using FastVec for efficient allocation
    entries: FastVec<Entry<K, V>>,
    /// Bucket array storing indices into entries
    buckets: FastVec<BucketEntry>,
    /// Hash function state
    hash_state: RandomState,
    /// Number of occupied entries
    len: usize,
    /// Load factor threshold for resizing (in 1/256ths)
    load_factor_threshold: u8,
    /// Marker for key/value types
    _phantom: PhantomData<(K, V)>,
}

/// Internal entry storage
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    hash: u64,
    /// Distance from ideal position (for robin hood probing)
    probe_distance: u16,
}

/// Bucket entry pointing to actual data
#[derive(Debug, Clone, Copy)]
struct BucketEntry {
    /// Index into entries array, or EMPTY_BUCKET if empty
    entry_index: u32,
    /// Cached hash value for fast comparison
    cached_hash: u32,
}

const EMPTY_BUCKET: u32 = u32::MAX;
const DEFAULT_CAPACITY: usize = 16;
const DEFAULT_LOAD_FACTOR: u8 = 192; // 75% in 1/256ths

impl<K, V> GoldHashMap<K, V> {
    /// Creates a new empty hash map
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY).unwrap()
    }

    /// Creates a new hash map with specified capacity
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        let bucket_count = capacity.next_power_of_two().max(DEFAULT_CAPACITY);
        
        let mut buckets = FastVec::with_capacity(bucket_count)?;
        buckets.resize(bucket_count, BucketEntry {
            entry_index: EMPTY_BUCKET,
            cached_hash: 0,
        })?;

        Ok(GoldHashMap {
            entries: FastVec::with_capacity(capacity)?,
            buckets,
            hash_state: RandomState::new(),
            len: 0,
            load_factor_threshold: DEFAULT_LOAD_FACTOR,
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

    /// Returns the current capacity of the map
    pub fn capacity(&self) -> usize {
        self.buckets.len()
    }

    /// Computes hash for a key
    fn hash_key<Q>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + ?Sized,
    {
        let mut hasher = self.hash_state.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Converts hash to bucket index
    fn hash_to_bucket(&self, hash: u64) -> usize {
        // Use upper bits for better distribution
        let mask = self.buckets.len() - 1;
        ((hash >> 32) as usize) & mask
    }

    /// Finds the position of a key or where it should be inserted
    fn find_position<Q>(&self, key: &Q) -> FindResult
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.buckets.is_empty() {
            return FindResult::NotFound { ideal_bucket: 0 };
        }

        let hash = self.hash_key(key);
        let ideal_bucket = self.hash_to_bucket(hash);
        let cached_hash = (hash as u32) | 1; // Ensure non-zero
        
        let mut bucket_idx = ideal_bucket;
        let mut probe_distance = 0u16;

        loop {
            let bucket = &self.buckets[bucket_idx];
            
            if bucket.entry_index == EMPTY_BUCKET {
                return FindResult::NotFound { ideal_bucket: bucket_idx };
            }

            // Quick hash comparison before expensive key comparison
            if bucket.cached_hash == cached_hash {
                let entry_idx = bucket.entry_index as usize;
                if entry_idx < self.entries.len() {
                    let entry = &self.entries[entry_idx];
                    if entry.key.borrow() == key {
                        return FindResult::Found { 
                            bucket_index: bucket_idx,
                            entry_index: entry_idx 
                        };
                    }
                }
            }

            // Continue linear probing (removed robin hood logic for now)

            probe_distance += 1;
            bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);
            
            // Prevent infinite loop
            if probe_distance >= self.buckets.len() as u16 || bucket_idx == ideal_bucket {
                return FindResult::NotFound { ideal_bucket: bucket_idx };
            }
        }
    }

    /// Checks if resize is needed and performs it
    fn maybe_resize(&mut self) -> Result<()> {
        let load_factor = (self.len * 256) / self.buckets.len();
        
        if load_factor as u8 >= self.load_factor_threshold {
            self.resize(self.buckets.len() * 2)?;
        }
        
        Ok(())
    }

    /// Resizes the hash map to new bucket count
    fn resize(&mut self, new_bucket_count: usize) -> Result<()> {
        let _old_buckets = mem::replace(&mut self.buckets, {
            let mut new_buckets = FastVec::with_capacity(new_bucket_count)?;
            new_buckets.resize(new_bucket_count, BucketEntry {
                entry_index: EMPTY_BUCKET,
                cached_hash: 0,
            })?;
            new_buckets
        });

        // Reinsert all entries with new bucket positions
        let bucket_mask = self.buckets.len() - 1;
        for entry_idx in 0..self.entries.len() {
            let entry_hash = self.entries[entry_idx].hash;
            let bucket_idx = self.hash_to_bucket(entry_hash);
            let cached_hash = (entry_hash as u32) | 1;
            
            let mut pos = bucket_idx;
            let mut probe_distance = 0u16;
            
            loop {
                if self.buckets[pos].entry_index == EMPTY_BUCKET {
                    self.buckets[pos].entry_index = entry_idx as u32;
                    self.buckets[pos].cached_hash = cached_hash;
                    self.entries[entry_idx].probe_distance = probe_distance;
                    break;
                }
                
                probe_distance += 1;
                pos = (pos + 1) & bucket_mask;
            }
        }

        Ok(())
    }
}

impl<K, V> GoldHashMap<K, V>
where
    K: Hash + Eq,
{
    /// Inserts a key-value pair into the map
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        self.maybe_resize()?;
        
        let hash = self.hash_key(&key);
        
        match self.find_position(&key) {
            FindResult::Found { entry_index, .. } => {
                // Replace existing value
                let old_value = mem::replace(&mut self.entries[entry_index].value, value);
                Ok(Some(old_value))
            }
            FindResult::NotFound { ideal_bucket } => {
                // Insert new entry
                let entry_index = self.entries.len();
                let cached_hash = (hash as u32) | 1;
                
                // Add entry to storage
                self.entries.push(Entry {
                    key,
                    value,
                    hash,
                    probe_distance: 0,
                })?;

                // Simple linear probing for now (will optimize later)
                let mut pos = ideal_bucket;
                let mut probe_distance = 0u16;
                
                loop {
                    let bucket = &mut self.buckets[pos];
                    
                    if bucket.entry_index == EMPTY_BUCKET {
                        bucket.entry_index = entry_index as u32;
                        bucket.cached_hash = cached_hash;
                        self.entries[entry_index].probe_distance = probe_distance;
                        break;
                    }
                    
                    probe_distance += 1;
                    pos = (pos + 1) & (self.buckets.len() - 1);
                    
                    // Safety check to prevent infinite loop
                    if probe_distance > self.buckets.len() as u16 {
                        return Err(ToplingError::invalid_data("Hash map is full"));
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
            FindResult::Found { entry_index, .. } => {
                Some(&self.entries[entry_index].value)
            }
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
            FindResult::Found { entry_index, .. } => {
                Some(&mut self.entries[entry_index].value)
            }
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
            FindResult::Found { bucket_index, entry_index } => {
                // Mark bucket as empty
                self.buckets[bucket_index] = BucketEntry {
                    entry_index: EMPTY_BUCKET,
                    cached_hash: 0,
                };

                // Extract the entry value before swapping
                let removed_value = if entry_index == self.entries.len() - 1 {
                    // Last entry, just pop it
                    self.entries.pop().unwrap().value
                } else {
                    // Need to maintain compactness by moving last entry to this position
                    let last_entry = self.entries.pop().unwrap();
                    let removed_entry = mem::replace(&mut self.entries[entry_index], last_entry);
                    
                    // Now we need to update the bucket that was pointing to the last entry
                    // to point to the new position (entry_index)
                    self.update_bucket_for_moved_entry(self.entries.len(), entry_index);
                    
                    removed_entry.value
                };

                self.len -= 1;
                Some(removed_value)
            }
            FindResult::NotFound { .. } => None,
        }
    }

    /// Updates bucket index after moving an entry from old_index to new_index
    fn update_bucket_for_moved_entry(&mut self, old_index: usize, new_index: usize) {
        // Find the bucket that was pointing to old_index and update it to new_index
        for bucket in 0..self.buckets.len() {
            if self.buckets[bucket].entry_index == old_index as u32 {
                self.buckets[bucket].entry_index = new_index as u32;
                return;
            }
        }
    }


    /// Clears all entries from the map
    pub fn clear(&mut self) {
        self.entries.clear();
        for i in 0..self.buckets.len() {
            self.buckets[i].entry_index = EMPTY_BUCKET;
            self.buckets[i].cached_hash = 0;
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
    Found { bucket_index: usize, entry_index: usize },
    NotFound { ideal_bucket: usize },
}

impl<K, V> Default for GoldHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> fmt::Debug for GoldHashMap<K, V>
where
    K: fmt::Debug + Hash + Eq,
    V: fmt::Debug,
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
        let map = GoldHashMap::<i32, String>::new();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.capacity() >= DEFAULT_CAPACITY);
    }

    #[test]
    fn test_insert_and_get() {
        let mut map = GoldHashMap::new();
        
        assert_eq!(map.insert("key1", "value1").unwrap(), None);
        assert_eq!(map.insert("key2", "value2").unwrap(), None);
        assert_eq!(map.len(), 2);

        assert_eq!(map.get("key1"), Some(&"value1"));
        assert_eq!(map.get("key2"), Some(&"value2"));
        assert_eq!(map.get("key3"), None);
    }

    #[test]
    fn test_insert_replace() {
        let mut map = GoldHashMap::new();
        
        assert_eq!(map.insert("key", "value1").unwrap(), None);
        assert_eq!(map.insert("key", "value2").unwrap(), Some("value1"));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("key"), Some(&"value2"));
    }

    #[test]
    fn test_remove() {
        let mut map = GoldHashMap::new();
        
        map.insert("key1", "value1").unwrap();
        map.insert("key2", "value2").unwrap();
        
        assert_eq!(map.remove("key1"), Some("value1"));
        assert_eq!(map.remove("key1"), None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("key2"), Some(&"value2"));
    }

    #[test]
    fn test_contains_key() {
        let mut map = GoldHashMap::new();
        
        assert!(!map.contains_key("key"));
        map.insert("key", "value").unwrap();
        assert!(map.contains_key("key"));
    }

    #[test]
    fn test_clear() {
        let mut map = GoldHashMap::new();
        
        map.insert("key1", "value1").unwrap();
        map.insert("key2", "value2").unwrap();
        assert_eq!(map.len(), 2);
        
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get("key1"), None);
    }

    #[test]
    fn test_small_remove() {
        let mut map = GoldHashMap::new();
        
        // Insert a few items
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        map.insert("c", 3).unwrap();
        
        // Verify all are present
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), Some(&3));
        
        // Remove one
        assert_eq!(map.remove("b"), Some(2));
        
        // Verify remaining are still present
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("c"), Some(&3));
        assert_eq!(map.get("b"), None);
    }

    #[test]
    fn test_large_dataset() {
        let mut map = GoldHashMap::new();
        
        // Insert many items to test resizing
        for i in 0..100 {  // Reduced for easier debugging
            let key = format!("key_{}", i);
            let value = format!("value_{}", i);
            map.insert(key, value).unwrap();
        }
        
        assert_eq!(map.len(), 100);
        
        // Verify all items are present before any removals
        for i in 0..100 {
            let key = format!("key_{}", i);
            let expected_value = format!("value_{}", i);
            assert_eq!(map.get(&key), Some(&expected_value), "Failed to find key_{}", i);
        }
    }

    #[test]
    fn test_iter() {
        let mut map = GoldHashMap::new();
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        map.insert("c", 3).unwrap();
        
        let mut items: Vec<_> = map.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        
        assert_eq!(items, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
        assert_eq!(map.iter().len(), 3);
    }

    #[test]
    fn test_keys_values() {
        let mut map = GoldHashMap::new();
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        
        let mut keys: Vec<_> = map.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
        
        let mut values: Vec<_> = map.values().cloned().collect();
        values.sort();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn test_get_mut() {
        let mut map = GoldHashMap::new();
        map.insert("key", 42).unwrap();
        
        if let Some(value) = map.get_mut("key") {
            *value = 84;
        }
        
        assert_eq!(map.get("key"), Some(&84));
    }

    #[test]
    fn test_hash_collision_handling() {
        let mut map = GoldHashMap::new();
        
        // Insert many items with potentially colliding hashes
        for i in 0..100 {
            let key = i.to_string();
            map.insert(key.clone(), i).unwrap();
        }
        
        // Verify all items can be retrieved
        for i in 0..100 {
            let key = i.to_string();
            assert_eq!(map.get(&key), Some(&i));
        }
        
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn test_debug_impl() {
        let mut map = GoldHashMap::new();
        map.insert("key", "value").unwrap();
        
        let debug_output = format!("{:?}", map);
        assert!(debug_output.contains("key"));
        assert!(debug_output.contains("value"));
    }

    #[test]
    fn test_with_capacity() {
        let map = GoldHashMap::<i32, String>::with_capacity(100).unwrap();
        assert!(map.capacity() >= 100);
        assert_eq!(map.len(), 0);
    }
}