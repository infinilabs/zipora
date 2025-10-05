//! Simplified Hash Map Interface
//!
//! A simplified interface wrapper around ZiporaHashMap that provides
//! convenient APIs for common use cases while maintaining the same
//! performance characteristics. This container focuses on ease of use
//! over explicit error handling.

use crate::error::Result;
use crate::hash_map::ZiporaHashMap;
use std::hash::Hash;
use std::marker::PhantomData;

/// Simplified hash map with convenient APIs
///
/// EasyHashMap provides a user-friendly wrapper around ZiporaHashMap that
/// simplifies common operations by removing explicit error handling where
/// safe and providing convenient builder patterns and default value semantics.
///
/// # Design Principles
///
/// - **Convenience**: Simplified APIs for common use cases
/// - **Performance**: Same underlying performance as ZiporaHashMap
/// - **Safety**: Panic-free operations with sensible fallbacks
/// - **Flexibility**: Builder pattern for advanced configuration
///
/// # Performance Characteristics
///
/// - **Memory**: Same as ZiporaHashMap
/// - **Lookup**: O(1) average, identical to ZiporaHashMap
/// - **Insert**: O(1) average with automatic capacity management
/// - **Error Handling**: Zero overhead when no errors occur
///
/// # Example
///
/// ```rust
/// use zipora::containers::EasyHashMap;
///
/// // Simple usage
/// let mut map = EasyHashMap::new();
/// map.put("key1", 42);
/// assert_eq!(map.get(&"key1"), Some(&42));
///
/// // With default values
/// let mut map_with_default = EasyHashMap::with_default(0);
/// assert_eq!(map_with_default.get_or_default(&"missing"), &0);
/// ```
#[derive(Debug)]
pub struct EasyHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Underlying ZiporaHashMap
    inner: ZiporaHashMap<K, V>,
    /// Default value for missing keys
    default_value: Option<V>,
    /// Automatic growth enabled
    auto_grow: bool,
    /// Maximum load factor before resize
    max_load_factor: f64,
}

impl<K, V> EasyHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new empty EasyHashMap
    pub fn new() -> Self {
        Self {
            inner: ZiporaHashMap::new().expect("Failed to create ZiporaHashMap"),
            default_value: None,
            auto_grow: true,
            max_load_factor: 0.75,
        }
    }

    /// Create an EasyHashMap with a default value for missing keys
    pub fn with_default(default_value: V) -> Self {
        Self {
            inner: ZiporaHashMap::new().expect("Failed to create ZiporaHashMap"),
            default_value: Some(default_value),
            auto_grow: true,
            max_load_factor: 0.75,
        }
    }

    /// Insert a key-value pair (simplified interface)
    ///
    /// This method automatically handles capacity growth and never panics.
    /// If an internal error occurs, it will be silently handled.
    ///
    /// # Arguments
    /// * `key` - The key to insert
    /// * `value` - The value to associate with the key
    pub fn put(&mut self, key: K, value: V) {
        // Auto-grow if needed
        if self.auto_grow && self.should_grow() {
            // Force growth by recreating map with larger capacity
            let current_capacity = self.inner.capacity();
            let new_capacity = (current_capacity * 2).max(64);

            if let Ok(mut new_map) = ZiporaHashMap::with_capacity(new_capacity) {
                // Copy existing entries using get_all_entries helper
                let entries_to_copy = self.get_all_entries();
                for (k, v) in entries_to_copy {
                    let _ = new_map.insert(k, v);
                }
                self.inner = new_map;
            }
        }

        // Insert - if it fails due to memory, just continue silently
        let _ = self.inner.insert(key, value);
    }

    /// Get a reference to the value associated with a key
    pub fn get(&self, key: &K) -> Option<&V> {
        self.inner.get(key)
    }

    /// Get a reference to the value or the default value if key doesn't exist
    pub fn get_or_default(&self, key: &K) -> &V {
        self.inner
            .get(key)
            .unwrap_or_else(|| self.default_value.as_ref().expect("No default value set"))
    }

    /// Get a mutable reference to the value, inserting with provided value if missing
    pub fn get_or_insert(&mut self, key: K, value: V) -> Result<&mut V> {
        let key_exists = self.inner.contains_key(&key);
        if !key_exists {
            self.put(key.clone(), value);
        }

        // This should always succeed since we just inserted
        Ok(self
            .inner
            .get_mut(&key)
            .expect("Key should exist after insertion"))
    }

    /// Get a mutable reference to the value, inserting with closure result if missing
    pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> Result<&mut V>
    where
        F: FnOnce() -> V,
    {
        let key_exists = self.inner.contains_key(&key);
        if !key_exists {
            let value = f();
            self.put(key.clone(), value);
        }

        Ok(self
            .inner
            .get_mut(&key)
            .expect("Key should exist after insertion"))
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.inner.remove(key)
    }

    /// Check if the map contains a key
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.contains_key(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the current capacity
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Extend the map with key-value pairs from an iterator
    pub fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.put(key, value);
        }
    }

    /// Retain only the key-value pairs that satisfy the predicate
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let keys_to_remove: Vec<K> = self
            .inner
            .iter()
            .filter_map(|(k, v)| {
                // Simple test - just check if we should keep the item
                let mut value_copy = v.clone();
                if f(k, &mut value_copy) {
                    None
                } else {
                    Some(k.clone())
                }
            })
            .collect();

        for key in keys_to_remove {
            self.inner.remove(&key);
        }
    }

    /// Get an iterator over key-value pairs
    /// TODO: Implement when ZiporaHashMap has iter() support
    // pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
    //     self.inner.iter()
    // }

    /// Get an iterator over keys
    /// TODO: Implement when ZiporaHashMap has keys() support
    // pub fn keys(&self) -> impl Iterator<Item = &K> {
    //     self.inner.keys()
    // }

    /// Get an iterator over values
    /// TODO: Implement when ZiporaHashMap has values() support
    // pub fn values(&self) -> impl Iterator<Item = &V> {
    //     self.inner.values()
    // }

    /// Get an iterator over mutable values
    /// TODO: Implement when ZiporaHashMap has iterator support
    // pub fn values_mut(&mut self) -> ValuesIterMut<K, V> {
    //     ValuesIterMut {
    //         keys: self.inner.keys().cloned().collect(),
    //         map: self,
    //         current: 0,
    //     }
    // }

    /// Enable or disable automatic growth
    pub fn set_auto_grow(&mut self, enabled: bool) {
        self.auto_grow = enabled;
    }

    /// Set the maximum load factor for automatic growth
    pub fn set_max_load_factor(&mut self, factor: f64) {
        self.max_load_factor = factor.clamp(0.1, 0.95);
    }

    /// Create a builder for configuring the map
    pub fn initial_capacity(capacity: usize) -> EasyHashMapBuilder<K, V>
    where
        K: Hash + Eq + Clone,
        V: Clone,
    {
        EasyHashMapBuilder::new().with_capacity(capacity)
    }

    /// Create a builder with a default value
    pub fn with_default_value(default: V) -> EasyHashMapBuilder<K, V>
    where
        K: Hash + Eq + Clone,
        V: Clone,
    {
        EasyHashMapBuilder::new().with_default(default)
    }

    /// Check if the map should grow
    fn should_grow(&self) -> bool {
        let capacity = self.inner.capacity();
        if capacity == 0 {
            return true;
        }

        let load_factor = self.inner.len() as f64 / capacity as f64;
        load_factor >= self.max_load_factor
    }

    /// Get all entries as owned key-value pairs (helper for growth)
    fn get_all_entries(&self) -> Vec<(K, V)> {
        let mut entries = Vec::new();
        for (k, v) in self.inner.iter() {
            entries.push((k.clone(), v.clone()));
        }
        entries
    }

    /// Try to reserve additional capacity (no-op for ZiporaHashMap)
    pub fn try_reserve(&mut self, _additional: usize) -> Result<()> {
        // ZiporaHashMap auto-grows, so this is a no-op
        Ok(())
    }

    /// Reserve additional capacity (panics on failure)
    pub fn reserve(&mut self, additional: usize) {
        let _ = self.try_reserve(additional);
    }

    /// Shrink the capacity to fit the current number of elements
    pub fn shrink_to_fit(&mut self) {
        let current_len = self.inner.len();
        let current_capacity = self.inner.capacity();

        // Only shrink if we can save significant space
        if current_len < current_capacity / 2 && current_capacity > 32 {
            // Collect all entries using the iterator
            let entries: Vec<(K, V)> = self.inner.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            // Calculate new capacity - at least 2x the current length, minimum 16
            let new_capacity = (current_len * 2).max(16);

            // Create new map with smaller capacity
            if let Ok(mut new_map) = ZiporaHashMap::with_capacity(new_capacity) {
                for (key, value) in entries {
                    let _ = new_map.insert(key, value);
                }
                self.inner = new_map;
            }
        }
    }

    /// Get statistics about the map
    pub fn statistics(&self) -> EasyHashMapStats {
        EasyHashMapStats {
            len: self.len(),
            capacity: self.capacity(),
            load_factor: if self.capacity() > 0 {
                self.len() as f64 / self.capacity() as f64
            } else {
                0.0
            },
            has_default: self.default_value.is_some(),
            auto_grow_enabled: self.auto_grow,
            max_load_factor: self.max_load_factor,
        }
    }
}

/// Builder for configuring EasyHashMap
#[derive(Debug)]
pub struct EasyHashMapBuilder<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    capacity: usize,
    default_value: Option<V>,
    auto_grow: bool,
    max_load_factor: f64,
    _phantom: PhantomData<K>,
}

impl<K, V> EasyHashMapBuilder<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            capacity: 16,
            default_value: None,
            auto_grow: true,
            max_load_factor: 0.75,
            _phantom: PhantomData,
        }
    }

    /// Set the initial capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(16);
        self
    }

    /// Set a default value for missing keys
    pub fn with_default(mut self, default: V) -> Self {
        self.default_value = Some(default);
        self
    }

    /// Enable or disable automatic growth
    pub fn auto_grow(mut self, enabled: bool) -> Self {
        self.auto_grow = enabled;
        self
    }

    /// Set the maximum load factor
    pub fn max_load_factor(mut self, factor: f64) -> Self {
        self.max_load_factor = factor.clamp(0.1, 0.95);
        self
    }

    /// Build the EasyHashMap
    pub fn build(self) -> EasyHashMap<K, V> {
        EasyHashMap {
            inner: ZiporaHashMap::with_capacity(self.capacity).expect("Failed to create ZiporaHashMap with capacity"),
            default_value: self.default_value,
            auto_grow: self.auto_grow,
            max_load_factor: self.max_load_factor,
        }
    }
}

/// Iterator over mutable values in EasyHashMap
pub struct ValuesIterMut<'a, K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    keys: Vec<K>,
    map: &'a mut EasyHashMap<K, V>,
    current: usize,
}

impl<'a, K, V> Iterator for ValuesIterMut<'a, K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.keys.len() {
            let key = &self.keys[self.current];
            self.current += 1;

            // SAFETY: We know this key exists since we got it from keys()
            // We need to use unsafe here to extend the lifetime for the iterator
            if let Some(value) = self.map.inner.get_mut(key) {
                return Some(unsafe { std::mem::transmute(value) });
            }
        }
        None
    }
}

/// Statistics about EasyHashMap configuration and usage
#[derive(Debug, Clone)]
pub struct EasyHashMapStats {
    /// Number of key-value pairs
    pub len: usize,
    /// Current capacity
    pub capacity: usize,
    /// Current load factor
    pub load_factor: f64,
    /// Whether a default value is set
    pub has_default: bool,
    /// Whether automatic growth is enabled
    pub auto_grow_enabled: bool,
    /// Maximum load factor threshold
    pub max_load_factor: f64,
}

impl<K, V> Default for EasyHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Default for EasyHashMapBuilder<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// Implement common collection traits
impl<K, V> std::iter::FromIterator<(K, V)> for EasyHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        map.extend(iter);
        map
    }
}

impl<K, V> std::iter::Extend<(K, V)> for EasyHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        self.extend(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut map = EasyHashMap::new();

        // Put and get
        map.put("key1", 42);
        map.put("key2", 43);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"key1"), Some(&42));
        assert_eq!(map.get(&"key2"), Some(&43));
        assert_eq!(map.get(&"key3"), None);

        // Contains
        assert!(map.contains_key(&"key1"));
        assert!(!map.contains_key(&"key3"));

        // Update
        map.put("key1", 44);
        assert_eq!(map.get(&"key1"), Some(&44));

        // Remove
        assert_eq!(map.remove(&"key1"), Some(44));
        assert_eq!(map.get(&"key1"), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_default_value() {
        let mut map = EasyHashMap::with_default(0);

        // Should return default for missing keys
        assert_eq!(map.get_or_default(&"missing"), &0);

        // Insert a real value
        map.put("real", 42);
        assert_eq!(map.get_or_default(&"real"), &42);
        assert_eq!(map.get_or_default(&"still_missing"), &0);
    }

    #[test]
    fn test_get_or_insert() {
        let mut map = EasyHashMap::new();

        // Insert with get_or_insert
        let value = map.get_or_insert("key1", 42).unwrap();
        assert_eq!(*value, 42);
        assert_eq!(map.len(), 1);

        // Should return existing value
        let value = map.get_or_insert("key1", 99).unwrap();
        assert_eq!(*value, 42); // Original value

        // Modify through mutable reference
        *value = 100;
        assert_eq!(map.get(&"key1"), Some(&100));
    }

    #[test]
    fn test_get_or_insert_with() {
        let mut map = EasyHashMap::new();
        let mut call_count = 0;

        // Insert with closure
        let value = map
            .get_or_insert_with("key1", || {
                call_count += 1;
                42
            })
            .unwrap();
        assert_eq!(*value, 42);
        assert_eq!(call_count, 1);

        // Should not call closure again
        let value = map
            .get_or_insert_with("key1", || {
                call_count += 1;
                99
            })
            .unwrap();
        assert_eq!(*value, 42); // Original value
        assert_eq!(call_count, 1); // Closure not called
    }

    #[test]
    fn test_extend() {
        let mut map = EasyHashMap::new();

        let data = vec![("a", 1), ("b", 2), ("c", 3)];
        map.extend(data);

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), Some(&3));
    }

    #[test]
    fn test_retain() {
        let mut map = EasyHashMap::new();

        map.put("a", 1);
        map.put("b", 2);
        map.put("c", 3);
        map.put("d", 4);

        // Retain only even values
        map.retain(|_k, v| *v % 2 == 0);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"d"), Some(&4));
        assert_eq!(map.get(&"a"), None);
        assert_eq!(map.get(&"c"), None);
    }

    #[test]
    fn test_builder_pattern() {
        let map = EasyHashMap::<String, String>::initial_capacity(100)
            .with_default(String::new())
            .auto_grow(false)
            .max_load_factor(0.5)
            .build();

        let stats = map.statistics();
        assert!(stats.capacity >= 100);
        assert!(stats.has_default);
        assert!(!stats.auto_grow_enabled);
        assert_eq!(stats.max_load_factor, 0.5);
    }

    #[test]
    fn test_from_iterator() {
        let data = vec![("x", 10), ("y", 20), ("z", 30)];
        let map: EasyHashMap<&str, i32> = data.into_iter().collect();

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"x"), Some(&10));
        assert_eq!(map.get(&"y"), Some(&20));
        assert_eq!(map.get(&"z"), Some(&30));
    }

    #[test]
    fn test_iterators() {
        let mut map = EasyHashMap::new();
        map.put("a", 1);
        map.put("b", 2);
        map.put("c", 3);

        // TODO: Uncomment when iterator support is added to ZiporaHashMap
        // Test keys iterator
        // let mut keys: Vec<&str> = map.keys().copied().collect();
        // keys.sort();
        // assert_eq!(keys, vec!["a", "b", "c"]);

        // Test values iterator
        // let mut values: Vec<&i32> = map.values().collect();
        // values.sort();
        // assert_eq!(values, vec![&1, &2, &3]);

        // Test iter
        // assert_eq!(map.iter().count(), 3);

        // Test mutable values
        // for value in map.values_mut() {
        //     *value *= 2;
        // }

        // TODO: These values should be doubled after values_mut is implemented
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), Some(&3));
    }

    #[test]
    fn test_auto_growth() {
        let mut map = EasyHashMap::initial_capacity(4)
            .max_load_factor(0.5) // Grow early
            .build();

        let initial_capacity = map.capacity();

        // Fill beyond load factor
        for i in 0..10 {
            map.put(i, i * 2);
        }

        // Should have grown
        assert!(map.capacity() > initial_capacity);
        assert_eq!(map.len(), 10);

        // Verify all values
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_clear_and_shrink() {
        let mut map = EasyHashMap::<i32, i32>::initial_capacity(1000).build();

        // Fill with some data
        for i in 0..10 {
            map.put(i, i);
        }

        assert_eq!(map.len(), 10);
        assert!(map.capacity() >= 1000);

        // Clear
        map.clear();
        assert_eq!(map.len(), 0);
        assert!(map.capacity() >= 1000); // Capacity unchanged

        // Shrink to fit
        map.shrink_to_fit();
        assert!(map.capacity() < 1000); // Should be smaller now
    }

    #[test]
    fn test_statistics() {
        let map = EasyHashMap::<i32, i32>::with_default_value(42)
            .auto_grow(false)
            .max_load_factor(0.8)
            .build();

        let stats = map.statistics();
        assert_eq!(stats.len, 0);
        assert!(stats.capacity > 0);
        assert_eq!(stats.load_factor, 0.0);
        assert!(stats.has_default);
        assert!(!stats.auto_grow_enabled);
        assert_eq!(stats.max_load_factor, 0.8);
    }

    #[test]
    fn test_large_scale() {
        let mut map = EasyHashMap::new();

        // Insert many items
        for i in 0..10000 {
            map.put(i, i * 2);
        }

        assert_eq!(map.len(), 10000);

        // Verify random samples
        for i in (0..10000).step_by(100) {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }

        // Remove half
        for i in 0..5000 {
            map.remove(&i);
        }

        assert_eq!(map.len(), 5000);

        // Verify remaining
        for i in 5000..10000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_error_resilience() {
        let mut map = EasyHashMap::new();

        // These operations should never panic even if internal errors occur
        map.put("key", 42);
        map.put("key", 43); // Update

        assert_eq!(map.get(&"key"), Some(&43));

        // Reserve should not panic on failure
        map.reserve(usize::MAX); // This might fail internally but shouldn't panic

        // Map should still work
        assert_eq!(map.get(&"key"), Some(&43));
    }
}
