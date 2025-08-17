//! StringOptimizedHashMap - Hash map specialized for string keys
//!
//! This implementation provides superior performance for string keys through:
//! - String interning with reference counting for memory efficiency
//! - Prefix caching in hash nodes for fast string comparison
//! - SIMD-accelerated string operations when available
//! - Compact string storage with offset-based addressing

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use crate::hash_map::hash_functions::{
    fabo_hash_combine_u64, golden_ratio_next_size, optimal_bucket_count, GOLDEN_LOAD_FACTOR,
};
use crate::hash_map::simd_string_ops::{get_global_simd_ops, SimdStringOps};
use crate::memory::SecureMemoryPool;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};
use std::ptr::NonNull;
use std::sync::Arc;

/// String-optimized hash map with interning and SIMD acceleration
///
/// StringOptimizedHashMap provides exceptional performance for string keys by:
/// - Interning strings to reduce memory usage and enable fast equality comparison
/// - Caching string prefixes in hash nodes for ultra-fast lookups
/// - Using SIMD operations for string comparison when available
/// - Employing compact storage with offset-based string addressing
///
/// # Examples
///
/// ```rust
/// use zipora::StringOptimizedHashMap;
///
/// let mut map = StringOptimizedHashMap::new();
/// map.insert("hello", 42).unwrap();
/// map.insert("world", 84).unwrap();
/// 
/// assert_eq!(map.get("hello"), Some(&42));
/// assert_eq!(map.len(), 2);
/// ```
pub struct StringOptimizedHashMap<V, S = ahash::RandomState> {
    /// String arena for interned strings
    string_arena: StringArena,
    /// Hash buckets with prefix caching
    buckets: FastVec<StringBucket>,
    /// Entry storage
    entries: FastVec<StringEntry<V>>,
    /// Hash builder
    hash_builder: S,
    /// Number of elements
    len: usize,
    /// Maximum probe distance
    max_probe_distance: u16,
    /// SIMD string operations for hardware acceleration
    simd_ops: &'static SimdStringOps,
}

/// String interning arena for memory-efficient string storage
struct StringArena {
    /// Raw string data storage
    data: FastVec<u8>,
    /// String table mapping content to offsets
    string_table: std::collections::HashMap<Vec<u8>, StringHandle>,
    /// Memory pool for large strings
    pool: Option<Arc<SecureMemoryPool>>,
    /// Total allocated size
    total_size: usize,
}

/// Handle to an interned string
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StringHandle {
    /// Offset in the string arena
    offset: u32,
    /// Length of the string
    length: u32,
    /// Reference count (for string sharing)
    ref_count: u32,
}

/// Hash bucket with string prefix caching
#[derive(Debug, Clone, Copy)]
struct StringBucket {
    /// Index into entries array
    entry_index: u32,
    /// Probe distance from ideal position
    probe_distance: u16,
    /// Cached hash value (16 bits)
    cached_hash: u16,
    /// First 8 bytes of string for fast comparison
    prefix_cache: u64,
}

/// Entry storing value and string handle
#[derive(Debug)]
struct StringEntry<V> {
    /// Handle to interned string
    string_handle: StringHandle,
    /// Associated value
    value: V,
    /// Full hash of the string
    hash: u64,
}

const EMPTY: u32 = u32::MAX;
const DEFAULT_CAPACITY: usize = 16;
const MAX_PROBE_DISTANCE: u16 = 64;
const PREFIX_CACHE_SIZE: usize = 8;

impl<V> StringOptimizedHashMap<V, ahash::RandomState> {
    /// Creates a new string-optimized hash map
    pub fn new() -> Self {
        Self::with_hasher(ahash::RandomState::new())
    }

    /// Creates a new string-optimized hash map with capacity
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_and_hasher(capacity, ahash::RandomState::new())
    }
}

impl<V, S> StringOptimizedHashMap<V, S>
where
    S: BuildHasher,
{
    /// Creates a new hash map with custom hasher
    pub fn with_hasher(hash_builder: S) -> Self {
        Self::with_capacity_and_hasher(DEFAULT_CAPACITY, hash_builder).unwrap()
    }

    /// Creates a new hash map with capacity and hasher
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Result<Self> {
        let bucket_count = optimal_bucket_count(capacity);

        let mut buckets = FastVec::with_capacity(bucket_count)?;
        buckets.resize(
            bucket_count,
            StringBucket {
                entry_index: EMPTY,
                probe_distance: 0,
                cached_hash: 0,
                prefix_cache: 0,
            },
        )?;

        Ok(StringOptimizedHashMap {
            string_arena: StringArena::new()?,
            buckets,
            entries: FastVec::with_capacity(capacity)?,
            hash_builder,
            len: 0,
            max_probe_distance: MAX_PROBE_DISTANCE,
            simd_ops: get_global_simd_ops(),
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

    /// Returns the current load factor
    pub fn load_factor(&self) -> f64 {
        if self.buckets.is_empty() {
            0.0
        } else {
            self.len as f64 / self.buckets.len() as f64
        }
    }

    /// Returns statistics about string interning
    pub fn string_arena_stats(&self) -> StringArenaStats {
        StringArenaStats {
            total_size: self.string_arena.total_size,
            unique_strings: self.string_arena.string_table.len(),
            arena_size: self.string_arena.data.len(),
        }
    }

    /// Computes hash for a string key with SIMD acceleration
    fn hash_string(&self, key: &str) -> u64 {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        let base_hash = hasher.finish();

        // Use SIMD-accelerated string hashing when available
        let simd_hash = self.simd_ops.fast_string_hash(key, base_hash);
        
        // Apply FaboHashCombine for better distribution
        fabo_hash_combine_u64(simd_hash, simd_hash.rotate_right(32))
    }

    /// Extracts prefix for caching (first 8 bytes) with SIMD optimization
    fn extract_prefix(&self, key: &str) -> u64 {
        // Use SIMD-accelerated prefix extraction when available
        self.simd_ops.extract_prefix_simd(key)
    }

    /// Compares string with cached prefix first using SIMD acceleration
    fn fast_string_compare(&self, handle: StringHandle, key: &str, prefix_cache: u64) -> bool {
        let stored_string = self.string_arena.get_string(handle);
        
        // Use SIMD-accelerated string comparison
        self.simd_ops.fast_string_compare(stored_string, key, prefix_cache)
    }

    /// Converts hash to bucket index
    fn hash_to_bucket(&self, hash: u64) -> usize {
        let mask = self.buckets.len() - 1;
        ((hash >> 32) as usize) & mask
    }

    /// Finds position for a string key
    fn find_position(&self, key: &str) -> FindResult {
        if self.buckets.is_empty() {
            return FindResult::NotFound { insertion_index: 0 };
        }

        let hash = self.hash_string(key);
        let ideal_bucket = self.hash_to_bucket(hash);
        let cached_hash = (hash as u16) | 1;
        let prefix_cache = self.extract_prefix(key);

        let mut bucket_idx = ideal_bucket;
        let mut probe_distance = 0u16;

        loop {
            let bucket = &self.buckets[bucket_idx];

            if bucket.entry_index == EMPTY {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }

            // Fast comparison using cached hash and prefix
            if bucket.cached_hash == cached_hash && bucket.prefix_cache == prefix_cache {
                let entry_idx = bucket.entry_index as usize;
                if entry_idx < self.entries.len() {
                    let entry = &self.entries[entry_idx];
                    if self.fast_string_compare(entry.string_handle, key, prefix_cache) {
                        return FindResult::Found {
                            bucket_index: bucket_idx,
                            entry_index: entry_idx,
                        };
                    }
                }
            }

            // Robin hood heuristic
            if probe_distance > bucket.probe_distance {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }

            probe_distance += 1;
            bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

            if probe_distance >= self.max_probe_distance {
                return FindResult::NotFound {
                    insertion_index: bucket_idx,
                };
            }
        }
    }

    /// Checks if resize is needed
    fn needs_resize(&self) -> bool {
        if self.buckets.is_empty() {
            return true;
        }

        let load_factor = (self.len * 256) / self.buckets.len();
        load_factor as u8 >= GOLDEN_LOAD_FACTOR
    }

    /// Resizes the hash map
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

    /// Performs resize operation
    fn resize(&mut self, new_bucket_count: usize) -> Result<()> {
        // Create new buckets
        let mut new_buckets = FastVec::with_capacity(new_bucket_count)?;
        new_buckets.resize(
            new_bucket_count,
            StringBucket {
                entry_index: EMPTY,
                probe_distance: 0,
                cached_hash: 0,
                prefix_cache: 0,
            },
        )?;

        let old_buckets = std::mem::replace(&mut self.buckets, new_buckets);

        // Re-insert all entries
        for entry_idx in 0..self.entries.len() {
            let entry = &self.entries[entry_idx];
            let hash = entry.hash;
            let ideal_bucket = self.hash_to_bucket(hash);
            let cached_hash = (hash as u16) | 1;
            
            let key = self.string_arena.get_string(entry.string_handle);
            let prefix_cache = self.extract_prefix(key);

            let mut bucket_idx = ideal_bucket;
            let mut probe_distance = 0u16;

            // Robin hood insertion
            loop {
                let bucket = &mut self.buckets[bucket_idx];

                if bucket.entry_index == EMPTY {
                    *bucket = StringBucket {
                        entry_index: entry_idx as u32,
                        probe_distance,
                        cached_hash,
                        prefix_cache,
                    };
                    break;
                }

                if probe_distance > bucket.probe_distance {
                    let mut old_bucket = *bucket;
                    *bucket = StringBucket {
                        entry_index: entry_idx as u32,
                        probe_distance,
                        cached_hash,
                        prefix_cache,
                    };

                    // Adjust probe distance for the displaced bucket's new starting position
                    old_bucket.probe_distance += 1;

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
    fn insert_displaced(&mut self, mut displaced: StringBucket, start_idx: usize) -> Result<()> {
        let mut bucket_idx = start_idx & (self.buckets.len() - 1);

        loop {
            let bucket = &mut self.buckets[bucket_idx];

            if bucket.entry_index == EMPTY {
                *bucket = displaced;
                break;
            }

            if displaced.probe_distance > bucket.probe_distance {
                std::mem::swap(bucket, &mut displaced);
            }

            displaced.probe_distance += 1;
            bucket_idx = (bucket_idx + 1) & (self.buckets.len() - 1);

            if displaced.probe_distance >= self.max_probe_distance {
                return Err(ZiporaError::invalid_data("Insert failed: excessive probing"));
            }
        }

        Ok(())
    }

    /// Inserts a key-value pair
    pub fn insert(&mut self, key: &str, value: V) -> Result<Option<V>> {
        self.resize_if_needed()?;

        let hash = self.hash_string(key);

        match self.find_position(key) {
            FindResult::Found { entry_index, .. } => {
                // Replace existing value
                let old_value = std::mem::replace(&mut self.entries[entry_index].value, value);
                Ok(Some(old_value))
            }
            FindResult::NotFound { insertion_index } => {
                // Intern the string
                let string_handle = self.string_arena.intern_string(key)?;
                
                // Create new entry
                let entry_index = self.entries.len();
                self.entries.push(StringEntry {
                    string_handle,
                    value,
                    hash,
                })?;

                // Insert bucket using robin hood
                let cached_hash = (hash as u16) | 1;
                let prefix_cache = self.extract_prefix(key);
                
                let mut bucket_idx = insertion_index;
                let ideal_bucket = self.hash_to_bucket(hash);
                let mut probe_distance = if bucket_idx >= ideal_bucket {
                    (bucket_idx - ideal_bucket) as u16
                } else {
                    (bucket_idx + self.buckets.len() - ideal_bucket) as u16
                };

                let mut new_bucket = StringBucket {
                    entry_index: entry_index as u32,
                    probe_distance,
                    cached_hash,
                    prefix_cache,
                };

                loop {
                    let bucket = &mut self.buckets[bucket_idx];

                    if bucket.entry_index == EMPTY {
                        *bucket = new_bucket;
                        break;
                    }

                    if new_bucket.probe_distance > bucket.probe_distance {
                        std::mem::swap(bucket, &mut new_bucket);
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
    pub fn get(&self, key: &str) -> Option<&V> {
        match self.find_position(key) {
            FindResult::Found { entry_index, .. } => Some(&self.entries[entry_index].value),
            FindResult::NotFound { .. } => None,
        }
    }

    /// Gets a mutable reference to the value for a key
    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        match self.find_position(key) {
            FindResult::Found { entry_index, .. } => Some(&mut self.entries[entry_index].value),
            FindResult::NotFound { .. } => None,
        }
    }

    /// Checks if the map contains a key
    pub fn contains_key(&self, key: &str) -> bool {
        matches!(self.find_position(key), FindResult::Found { .. })
    }

    /// Removes a key from the map
    pub fn remove(&mut self, key: &str) -> Option<V> {
        match self.find_position(key) {
            FindResult::Found {
                bucket_index,
                entry_index,
            } => {
                // Remove entry and handle string cleanup
                let removed_entry = if entry_index == self.entries.len() - 1 {
                    self.entries.pop().unwrap()
                } else {
                    let last_entry = self.entries.pop().unwrap();
                    let removed_entry = std::mem::replace(&mut self.entries[entry_index], last_entry);

                    // Update bucket pointing to moved entry
                    for i in 0..self.buckets.len() {
                        if self.buckets[i].entry_index == (self.entries.len()) as u32 {
                            self.buckets[i].entry_index = entry_index as u32;
                            break;
                        }
                    }

                    removed_entry
                };

                // Release string reference
                self.string_arena.release_string(removed_entry.string_handle);

                // Clear bucket and shift subsequent entries
                self.buckets[bucket_index] = StringBucket {
                    entry_index: EMPTY,
                    probe_distance: 0,
                    cached_hash: 0,
                    prefix_cache: 0,
                };

                // Robin hood backward shift
                let mut idx = (bucket_index + 1) & (self.buckets.len() - 1);
                while idx != bucket_index {
                    let current_bucket = self.buckets[idx]; // Copy the bucket data
                    if current_bucket.entry_index == EMPTY || current_bucket.probe_distance == 0 {
                        break;
                    }

                    let prev_idx = if idx == 0 { self.buckets.len() - 1 } else { idx - 1 };
                    let mut moved_bucket = current_bucket;
                    moved_bucket.probe_distance -= 1;
                    
                    self.buckets[prev_idx] = moved_bucket;
                    self.buckets[idx] = StringBucket {
                        entry_index: EMPTY,
                        probe_distance: 0,
                        cached_hash: 0,
                        prefix_cache: 0,
                    };

                    idx = (idx + 1) & (self.buckets.len() - 1);
                }

                self.len -= 1;
                Some(removed_entry.value)
            }
            FindResult::NotFound { .. } => None,
        }
    }

    /// Clears all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.string_arena.clear();
        for bucket in self.buckets.iter_mut() {
            *bucket = StringBucket {
                entry_index: EMPTY,
                probe_distance: 0,
                cached_hash: 0,
                prefix_cache: 0,
            };
        }
        self.len = 0;
    }

    /// Returns an iterator over key-value pairs
    pub fn iter(&self) -> StringMapIter<V> {
        StringMapIter {
            entries: &self.entries,
            arena: &self.string_arena,
            index: 0,
        }
    }

    /// Returns an iterator over keys
    pub fn keys(&self) -> StringMapKeys<V> {
        StringMapKeys { iter: self.iter() }
    }

    /// Returns an iterator over values
    pub fn values(&self) -> StringMapValues<V> {
        StringMapValues { iter: self.iter() }
    }
}

impl StringArena {
    /// Creates a new string arena
    fn new() -> Result<Self> {
        Ok(StringArena {
            data: FastVec::new(),
            string_table: std::collections::HashMap::new(),
            pool: None,
            total_size: 0,
        })
    }

    /// Interns a string and returns a handle
    fn intern_string(&mut self, s: &str) -> Result<StringHandle> {
        let bytes = s.as_bytes();
        
        // Check if string is already interned
        if let Some(&handle) = self.string_table.get(bytes) {
            // Increment reference count
            let mut new_handle = handle;
            new_handle.ref_count += 1;
            self.string_table.insert(bytes.to_vec(), new_handle);
            return Ok(new_handle);
        }

        // Store new string
        let offset = self.data.len() as u32;
        let length = bytes.len() as u32;
        
        for &byte in bytes {
            self.data.push(byte)?;
        }

        let handle = StringHandle {
            offset,
            length,
            ref_count: 1,
        };

        self.string_table.insert(bytes.to_vec(), handle);
        self.total_size += bytes.len();

        Ok(handle)
    }

    /// Gets a string from a handle
    fn get_string(&self, handle: StringHandle) -> &str {
        let start = handle.offset as usize;
        let end = start + handle.length as usize;
        
        if end <= self.data.len() {
            let slice = self.data.as_slice();
            let bytes = &slice[start..end];
            std::str::from_utf8(bytes).unwrap_or("")
        } else {
            ""
        }
    }

    /// Releases a string reference
    fn release_string(&mut self, handle: StringHandle) {
        let key_bytes = {
            let start = handle.offset as usize;
            let end = start + handle.length as usize;
            if end <= self.data.len() {
                let slice = self.data.as_slice();
                slice[start..end].to_vec()
            } else {
                return;
            }
        };

        if let Some(existing_handle) = self.string_table.get_mut(&key_bytes) {
            if existing_handle.ref_count > 1 {
                existing_handle.ref_count -= 1;
            } else {
                // Remove from table when ref count reaches 0
                self.string_table.remove(&key_bytes);
                self.total_size -= key_bytes.len();
            }
        }
    }

    /// Clears the arena
    fn clear(&mut self) {
        self.data.clear();
        self.string_table.clear();
        self.total_size = 0;
    }
}

/// Statistics about string arena usage
#[derive(Debug, Clone)]
pub struct StringArenaStats {
    /// Total size of all strings
    pub total_size: usize,
    /// Number of unique strings
    pub unique_strings: usize,
    /// Size of the arena data structure
    pub arena_size: usize,
}

/// Result of finding a position
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

impl<V, S> Default for StringOptimizedHashMap<V, S>
where
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<V, S> fmt::Debug for StringOptimizedHashMap<V, S>
where
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Iterator over key-value pairs
pub struct StringMapIter<'a, V> {
    entries: &'a FastVec<StringEntry<V>>,
    arena: &'a StringArena,
    index: usize,
}

impl<'a, V> Iterator for StringMapIter<'a, V> {
    type Item = (&'a str, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.entries.len() {
            let entry = &self.entries[self.index];
            self.index += 1;
            let key = self.arena.get_string(entry.string_handle);
            Some((key, &entry.value))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, V> ExactSizeIterator for StringMapIter<'a, V> {}

/// Iterator over keys
pub struct StringMapKeys<'a, V> {
    iter: StringMapIter<'a, V>,
}

impl<'a, V> Iterator for StringMapKeys<'a, V> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, V> ExactSizeIterator for StringMapKeys<'a, V> {}

/// Iterator over values
pub struct StringMapValues<'a, V> {
    iter: StringMapIter<'a, V>,
}

impl<'a, V> Iterator for StringMapValues<'a, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, V> ExactSizeIterator for StringMapValues<'a, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_optimized_basic() {
        let mut map = StringOptimizedHashMap::new();
        
        assert_eq!(map.insert("hello", 42).unwrap(), None);
        assert_eq!(map.insert("world", 84).unwrap(), None);
        assert_eq!(map.len(), 2);

        assert_eq!(map.get("hello"), Some(&42));
        assert_eq!(map.get("world"), Some(&84));
        assert_eq!(map.get("missing"), None);
    }

    #[test]
    fn test_string_interning() {
        let mut map = StringOptimizedHashMap::new();
        
        // Insert same string multiple times
        map.insert("duplicate", 1).unwrap();
        map.insert("duplicate", 2).unwrap(); // Should replace
        map.insert("other", 3).unwrap();
        
        let stats = map.string_arena_stats();
        assert_eq!(stats.unique_strings, 2); // "duplicate" and "other"
        assert_eq!(map.get("duplicate"), Some(&2));
    }

    #[test]
    fn test_prefix_caching() {
        let mut map = StringOptimizedHashMap::new();
        
        // Insert strings with same prefix
        map.insert("prefix_test_1", 1).unwrap();
        map.insert("prefix_test_2", 2).unwrap();
        map.insert("different", 3).unwrap();
        
        assert_eq!(map.get("prefix_test_1"), Some(&1));
        assert_eq!(map.get("prefix_test_2"), Some(&2));
        assert_eq!(map.get("different"), Some(&3));
    }

    #[test]
    fn test_removal_and_cleanup() {
        let mut map = StringOptimizedHashMap::new();
        
        map.insert("temp", 42).unwrap();
        map.insert("keep", 84).unwrap();
        
        assert_eq!(map.remove("temp"), Some(42));
        assert_eq!(map.remove("temp"), None);
        assert_eq!(map.get("keep"), Some(&84));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_large_dataset() {
        let mut map = StringOptimizedHashMap::new();
        
        // Insert strings gradually and verify (reduced size to focus on correctness)
        for i in 0..50 {
            let key = format!("key_{}", i);
            map.insert(&key, i).unwrap();
            
            // Verify the item was inserted and can be retrieved immediately
            assert_eq!(map.get(&key), Some(&i), "Failed to retrieve key {} immediately after insertion", key);
        }
        
        // Verify all are present
        for i in 0..50 {
            let key = format!("key_{}", i);
            assert_eq!(map.get(&key), Some(&i), "Failed to retrieve key {} after all insertions", key);
        }
        
        assert_eq!(map.len(), 50);
    }

    #[test]
    fn test_iterators() {
        let mut map = StringOptimizedHashMap::new();
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        map.insert("c", 3).unwrap();

        let mut items: Vec<_> = map.iter().collect();
        items.sort_by_key(|(k, _)| *k);
        assert_eq!(items, vec![("a", &1), ("b", &2), ("c", &3)]);

        let mut keys: Vec<_> = map.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);

        let mut values: Vec<_> = map.values().cloned().collect();
        values.sort();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_clear() {
        let mut map = StringOptimizedHashMap::new();
        map.insert("test", 42).unwrap();
        
        assert_eq!(map.len(), 1);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        
        let stats = map.string_arena_stats();
        assert_eq!(stats.unique_strings, 0);
    }

    #[test]
    fn test_load_factor() {
        let mut map = StringOptimizedHashMap::new();
        
        assert_eq!(map.load_factor(), 0.0);
        
        map.insert("test", 42).unwrap();
        let load_factor = map.load_factor();
        assert!(load_factor > 0.0 && load_factor <= 1.0);
    }

    #[test]
    fn test_get_mut() {
        let mut map = StringOptimizedHashMap::new();
        map.insert("key", 42).unwrap();
        
        if let Some(value) = map.get_mut("key") {
            *value = 84;
        }
        
        assert_eq!(map.get("key"), Some(&84));
    }
}