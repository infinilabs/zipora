//! SmallHashMap - Inline storage hash map for small collections
//!
//! This implementation provides zero-allocation hash maps for small collections:
//! - Inline array storage for elements (typically N ≤ 8)
//! - Linear search optimization for small N
//! - Zero heap allocations until fallback threshold
//! - Automatic fallback to larger hash map when needed

use crate::error::Result;
use crate::hash_map::GoldenRatioHashMap;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;

/// Small hash map with inline storage for performance
///
/// SmallHashMap is optimized for collections with few elements (typically ≤ 8):
/// - Uses inline array storage to avoid heap allocation
/// - Linear search is often faster than hashing for small N
/// - Automatic fallback to full hash map when capacity exceeded
/// - Zero overhead for empty maps
///
/// # Type Parameters
/// - `K`: Key type (must implement Hash + Eq)
/// - `V`: Value type
/// - `N`: Inline capacity (number of elements stored inline)
///
/// # Examples
///
/// ```rust
/// use zipora::SmallHashMap;
///
/// // Hash map with inline storage for up to 4 elements
/// let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
/// map.insert("one", 1).unwrap();
/// map.insert("two", 2).unwrap();
/// 
/// assert_eq!(map.get("one"), Some(&1));
/// assert_eq!(map.len(), 2);
/// assert!(map.is_inline()); // Still using inline storage
/// ```
pub struct SmallHashMap<K, V, const N: usize, S = ahash::RandomState> {
    /// Current storage mode
    storage: Storage<K, V, N, S>,
}

/// Storage modes for the small hash map
enum Storage<K, V, const N: usize, S> {
    /// Inline storage using fixed-size array
    Inline {
        /// Fixed-size array for key-value pairs
        entries: [MaybeUninit<Entry<K, V>>; N],
        /// Number of occupied entries
        len: usize,
        /// Hash builder for when we need to hash keys
        hash_builder: S,
    },
    /// Large storage using full hash map
    Large {
        /// Full hash map for larger collections
        map: Box<GoldenRatioHashMap<K, V, S>>,
    },
}

/// Entry in the inline storage
#[derive(Debug, Clone)]
struct Entry<K, V> {
    key: K,
    value: V,
}

const LINEAR_SEARCH_THRESHOLD: usize = 8;

impl<K, V, const N: usize> SmallHashMap<K, V, N, ahash::RandomState> {
    /// Creates a new small hash map with default hasher
    pub fn new() -> Self {
        Self::with_hasher(ahash::RandomState::new())
    }
}

impl<K, V, const N: usize, S> SmallHashMap<K, V, N, S>
where
    S: BuildHasher + Default,
{
    /// Creates a new small hash map with custom hasher
    pub fn with_hasher(hash_builder: S) -> Self {
        Self {
            storage: Storage::Inline {
                entries: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
                hash_builder,
            },
        }
    }
}

impl<K, V, const N: usize, S> SmallHashMap<K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{
    /// Returns the number of elements in the map
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Inline { len, .. } => *len,
            Storage::Large { map } => map.len(),
        }
    }

    /// Returns true if the map is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the map is using inline storage
    pub fn is_inline(&self) -> bool {
        matches!(self.storage, Storage::Inline { .. })
    }

    /// Returns the current capacity
    pub fn capacity(&self) -> usize {
        match &self.storage {
            Storage::Inline { .. } => N,
            Storage::Large { map } => map.bucket_capacity(),
        }
    }

    /// Finds the position of a key in inline storage
    fn find_inline_position<Q>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if let Storage::Inline { entries, len, .. } = &self.storage {
            Self::find_in_entries(entries, *len, key)
        } else {
            None
        }
    }

    /// Helper function to find key in entries array
    fn find_in_entries<Q>(entries: &[MaybeUninit<Entry<K, V>>; N], len: usize, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        for i in 0..len {
            // Safe because we only access initialized entries
            let entry = unsafe { entries[i].assume_init_ref() };
            if entry.key.borrow() == key {
                return Some(i);
            }
        }
        None
    }

    /// Converts to large storage when inline capacity is exceeded
    fn convert_to_large(&mut self) -> Result<()> {
        if let Storage::Inline { entries, len, hash_builder } = &mut self.storage {
            // Create new hash map with appropriate capacity
            let capacity = (N * 2).max(16); // At least double the inline capacity
            let mut new_map = GoldenRatioHashMap::with_capacity_and_hasher(capacity, hash_builder.clone())?;

            // Move all entries to the new hash map
            for i in 0..*len {
                // Safe because we're only accessing initialized entries
                let entry = unsafe { entries[i].assume_init_read() };
                new_map.insert(entry.key, entry.value)?;
            }

            // Replace storage
            self.storage = Storage::Large {
                map: Box::new(new_map),
            };
        }
        Ok(())
    }

    /// Inserts a key-value pair into the map
    pub fn insert(&mut self, key: K, value: V) -> Result<Option<V>> {
        match &mut self.storage {
            Storage::Inline { entries, len, .. } => {
                // Check if key already exists
                if let Some(pos) = Self::find_in_entries(entries, *len, &key) {
                    // Replace existing value
                    let entry = unsafe { entries[pos].assume_init_mut() };
                    let old_value = std::mem::replace(&mut entry.value, value);
                    return Ok(Some(old_value));
                }

                // Check if we have space for a new entry
                if *len < N {
                    // Add new entry
                    entries[*len] = MaybeUninit::new(Entry { key, value });
                    *len += 1;
                    Ok(None)
                } else {
                    // Need to convert to large storage
                    self.convert_to_large()?;
                    self.insert(key, value)
                }
            }
            Storage::Large { map } => map.insert(key, value),
        }
    }

    /// Gets a reference to the value for a key
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &self.storage {
            Storage::Inline { .. } => {
                if let Some(pos) = self.find_inline_position(key) {
                    if let Storage::Inline { entries, .. } = &self.storage {
                        let entry = unsafe { entries[pos].assume_init_ref() };
                        Some(&entry.value)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Storage::Large { map } => map.get(key),
        }
    }

    /// Gets a mutable reference to the value for a key
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &mut self.storage {
            Storage::Inline { entries, len, .. } => {
                if let Some(pos) = Self::find_in_entries(entries, *len, key) {
                    let entry = unsafe { entries[pos].assume_init_mut() };
                    Some(&mut entry.value)
                } else {
                    None
                }
            }
            Storage::Large { map } => map.get_mut(key),
        }
    }

    /// Checks if the map contains a key
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &self.storage {
            Storage::Inline { .. } => self.find_inline_position(key).is_some(),
            Storage::Large { map } => map.contains_key(key),
        }
    }

    /// Removes a key from the map, returning the value if present
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        match &mut self.storage {
            Storage::Inline { entries, len, .. } => {
                if let Some(pos) = Self::find_in_entries(entries, *len, key) {
                    // Remove entry by moving last entry to this position
                    let removed_entry = unsafe { entries[pos].assume_init_read() };

                    // Move the last entry to fill the gap (if not removing the last entry)
                    if pos < *len - 1 {
                        let last_entry = unsafe { entries[*len - 1].assume_init_read() };
                        entries[pos] = MaybeUninit::new(last_entry);
                    }

                    *len -= 1;
                    Some(removed_entry.value)
                } else {
                    None
                }
            }
            Storage::Large { map } => map.remove(key),
        }
    }

    /// Clears all entries from the map
    pub fn clear(&mut self) {
        match &mut self.storage {
            Storage::Inline { entries, len, .. } => {
                // Drop all initialized entries
                for i in 0..*len {
                    unsafe {
                        entries[i].assume_init_drop();
                    }
                }
                *len = 0;
            }
            Storage::Large { map } => map.clear(),
        }
    }

    /// Returns an iterator over key-value pairs
    pub fn iter(&self) -> SmallMapIter<K, V, N, S> {
        SmallMapIter {
            map: self,
            index: 0,
        }
    }

    /// Returns an iterator over keys
    pub fn keys(&self) -> SmallMapKeys<K, V, N, S> {
        SmallMapKeys { iter: self.iter() }
    }

    /// Returns an iterator over values
    pub fn values(&self) -> SmallMapValues<K, V, N, S> {
        SmallMapValues { iter: self.iter() }
    }

    /// Reserves capacity for at least additional more elements
    pub fn reserve(&mut self, additional: usize) -> Result<()> {
        let needed_capacity = self.len() + additional;
        
        if needed_capacity > N {
            // Convert to large storage if not already
            if matches!(self.storage, Storage::Inline { .. }) {
                self.convert_to_large()?;
            }
            
            // Reserve in the large map (if it supports it)
            // GoldenRatioHashMap doesn't have a reserve method, so this is a no-op for now
        }
        
        Ok(())
    }

    /// Shrinks the capacity as much as possible
    pub fn shrink_to_fit(&mut self) -> Result<()>
    where
        K: Clone,
        V: Clone,
        S: Default,
    {
        // If we're using large storage but could fit in inline storage, convert back
        if let Storage::Large { .. } = &self.storage {
            // Take ownership of the storage to access the map
            let old_storage = std::mem::replace(&mut self.storage, Storage::Inline {
                entries: unsafe { MaybeUninit::uninit().assume_init() },
                len: 0,
                hash_builder: S::default(),
            });
            
            if let Storage::Large { map } = old_storage {
                if map.len() <= N {
                    // Convert back to inline storage
                    let mut entries: [MaybeUninit<Entry<K, V>>; N] = unsafe { MaybeUninit::uninit().assume_init() };
                    let mut len = 0;

                    // Move entries from large map to inline storage
                    for (key, value) in map.iter() {
                        if len < N {
                            entries[len] = MaybeUninit::new(Entry {
                                key: key.clone(),
                                value: value.clone(),
                            });
                            len += 1;
                        }
                    }

                    self.storage = Storage::Inline {
                        entries,
                        len,
                        hash_builder: S::default(),
                    };
                }
            }
        }
        
        Ok(())
    }
}

// Drop implementation to properly clean up inline storage
impl<K, V, const N: usize, S> Drop for SmallHashMap<K, V, N, S> {
    fn drop(&mut self) {
        if let Storage::Inline { entries, len, .. } = &mut self.storage {
            // Drop all initialized entries
            for i in 0..*len {
                unsafe {
                    entries[i].assume_init_drop();
                }
            }
        }
        // Large storage will be dropped automatically via Box
    }
}

impl<K, V, const N: usize, S> Default for SmallHashMap<K, V, N, S>
where
    S: BuildHasher + Default,
{
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, const N: usize, S> fmt::Debug for SmallHashMap<K, V, N, S>
where
    K: fmt::Debug + Hash + Eq,
    V: fmt::Debug,
    S: BuildHasher + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// Iterator over key-value pairs
pub struct SmallMapIter<'a, K, V, const N: usize, S> {
    map: &'a SmallHashMap<K, V, N, S>,
    index: usize,
}

impl<'a, K, V, const N: usize, S> Iterator for SmallMapIter<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        match &self.map.storage {
            Storage::Inline { entries, len, .. } => {
                if self.index < *len {
                    let entry = unsafe { entries[self.index].assume_init_ref() };
                    self.index += 1;
                    Some((&entry.key, &entry.value))
                } else {
                    None
                }
            }
            Storage::Large { map } => {
                // For large storage, we need to use the map's iterator
                // This is a simplified implementation - ideally we'd store the iterator
                let mut iter = map.iter();
                for _ in 0..self.index {
                    iter.next();
                }
                if let Some(item) = iter.next() {
                    self.index += 1;
                    Some(item)
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.map.len() - self.index.min(self.map.len());
        (remaining, Some(remaining))
    }
}

impl<'a, K, V, const N: usize, S> ExactSizeIterator for SmallMapIter<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{}

/// Iterator over keys
pub struct SmallMapKeys<'a, K, V, const N: usize, S> {
    iter: SmallMapIter<'a, K, V, N, S>,
}

impl<'a, K, V, const N: usize, S> Iterator for SmallMapKeys<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V, const N: usize, S> ExactSizeIterator for SmallMapKeys<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{}

/// Iterator over values
pub struct SmallMapValues<'a, K, V, const N: usize, S> {
    iter: SmallMapIter<'a, K, V, N, S>,
}

impl<'a, K, V, const N: usize, S> Iterator for SmallMapValues<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V, const N: usize, S> ExactSizeIterator for SmallMapValues<'a, K, V, N, S>
where
    K: Hash + Eq,
    S: BuildHasher + Clone,
{}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_map_basic() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert!(map.is_inline());

        // Insert within inline capacity
        assert_eq!(map.insert("a", 1).unwrap(), None);
        assert_eq!(map.insert("b", 2).unwrap(), None);
        assert_eq!(map.len(), 2);
        assert!(map.is_inline());

        // Check retrieval
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
        assert_eq!(map.get("c"), None);

        // Update existing
        assert_eq!(map.insert("a", 10).unwrap(), Some(1));
        assert_eq!(map.get("a"), Some(&10));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_inline_to_large_conversion() {
        let mut map: SmallHashMap<i32, i32, 2> = SmallHashMap::new();
        
        // Fill inline capacity
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert!(map.is_inline());
        assert_eq!(map.len(), 2);

        // This should trigger conversion to large storage
        map.insert(3, 30).unwrap();
        assert!(!map.is_inline());
        assert_eq!(map.len(), 3);

        // Verify all values are still accessible
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn test_removal() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();
        map.insert("c", 3).unwrap();

        assert_eq!(map.remove("b"), Some(2));
        assert_eq!(map.remove("b"), None);
        assert_eq!(map.len(), 2);
        
        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("c"), Some(&3));
        assert_eq!(map.get("b"), None);
    }

    #[test]
    fn test_clear() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        
        map.insert("a", 1).unwrap();
        map.insert("b", 2).unwrap();

        assert_eq!(map.len(), 2);
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.get("a"), None);
    }

    #[test]
    fn test_iterators() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
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
    fn test_contains_key() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        
        assert!(!map.contains_key("a"));
        map.insert("a", 1).unwrap();
        assert!(map.contains_key("a"));
        assert!(!map.contains_key("b"));
    }

    #[test]
    fn test_get_mut() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        map.insert("a", 1).unwrap();

        if let Some(value) = map.get_mut("a") {
            *value = 10;
        }

        assert_eq!(map.get("a"), Some(&10));
    }

    #[test]
    fn test_capacity_and_reserve() {
        let mut map: SmallHashMap<i32, i32, 4> = SmallHashMap::new();
        
        assert_eq!(map.capacity(), 4);
        assert!(map.is_inline());

        // Reserve more than inline capacity
        map.reserve(10).unwrap();
        // Note: reserve doesn't guarantee conversion, but large insertions will trigger it

        // Fill beyond inline capacity
        for i in 0..6 {
            map.insert(i, i * 10).unwrap();
        }
        
        assert!(!map.is_inline());
        assert!(map.capacity() > 4);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut map: SmallHashMap<i32, i32, 4> = SmallHashMap::new();
        
        // Force conversion to large storage
        for i in 0..6 {
            map.insert(i, i * 10).unwrap();
        }
        assert!(!map.is_inline());

        // Remove some elements to fit in inline storage
        map.remove(&4).unwrap();
        map.remove(&5).unwrap();
        assert_eq!(map.len(), 4);

        // Shrink should convert back to inline
        map.shrink_to_fit().unwrap();
        assert!(map.is_inline());

        // Verify all remaining elements are accessible
        for i in 0..4 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn test_large_storage_operations() {
        let mut map: SmallHashMap<i32, i32, 2> = SmallHashMap::new();
        
        // Force conversion to large storage
        for i in 0..10 {
            map.insert(i, i * 10).unwrap();
        }
        assert!(!map.is_inline());

        // Test operations on large storage
        assert_eq!(map.len(), 10);
        
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }

        // Test removal in large storage
        assert_eq!(map.remove(&5), Some(50));
        assert_eq!(map.len(), 9);
        assert_eq!(map.get(&5), None);

        // Test clearing large storage
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }
}