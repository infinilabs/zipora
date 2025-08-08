//! Simplified String-Optimized Hash Map
//!
//! A string-optimized hash map that provides basic string interning functionality
//! while maintaining compatibility with the zipora ecosystem.

use crate::error::Result;
use crate::string::FastStr;
use std::collections::HashMap;

/// Simplified string-optimized hash map
///
/// This is a simplified version that provides basic string interning
/// without the complex arena allocation to ensure compilation works.
#[derive(Debug)]
pub struct HashStrMap<V> {
    /// Main hash map using owned strings as keys  
    map: HashMap<String, V>,
    /// Statistics
    total_inserts: usize,
    unique_keys: usize,
}

impl<V> HashStrMap<V> {
    /// Create a new empty HashStrMap
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            total_inserts: 0,
            unique_keys: 0,
        }
    }

    /// Create a new HashStrMap with specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            total_inserts: 0,
            unique_keys: 0,
        }
    }

    /// Insert a key-value pair with string key
    pub fn insert(&mut self, key: &str, value: V) -> Result<Option<V>> {
        let key_string = key.to_string();
        let was_new = !self.map.contains_key(&key_string);
        let result = self.map.insert(key_string, value);
        
        self.total_inserts += 1;
        if was_new {
            self.unique_keys += 1;
        }
        
        Ok(result)
    }

    /// Insert a key-value pair taking ownership of the key string
    pub fn insert_string(&mut self, key: String, value: V) -> Result<Option<V>> {
        let was_new = !self.map.contains_key(&key);
        let result = self.map.insert(key, value);
        
        self.total_inserts += 1;
        if was_new {
            self.unique_keys += 1;
        }
        
        Ok(result)
    }

    /// Get a reference to the value associated with a key
    pub fn get(&self, key: &str) -> Option<&V> {
        self.map.get(key)
    }

    /// Get a mutable reference to the value associated with a key
    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    /// Remove a key-value pair
    pub fn remove(&mut self, key: &str) -> Option<V> {
        self.map.remove(key)
    }

    /// Check if the map contains a key
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Get the number of key-value pairs
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the map is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Check if a string would be considered "interned" (already exists)
    pub fn is_interned(&self, s: &str) -> bool {
        self.map.contains_key(s)
    }

    /// Get the string interning ratio (simulated)
    pub fn interning_ratio(&self) -> f64 {
        if self.total_inserts == 0 {
            0.0
        } else {
            1.0 - (self.unique_keys as f64 / self.total_inserts as f64)
        }
    }

    /// Get total memory used by string storage (approximate)
    pub fn string_memory_usage(&self) -> usize {
        self.map.keys().map(|k| k.len()).sum()
    }

    /// Get value using a FastStr key directly (compatibility)
    pub fn get_by_fast_str(&self, key: &FastStr) -> Option<&V> {
        if let Some(s) = key.as_str() {
            self.get(s)
        } else {
            None
        }
    }

    /// Insert using a FastStr key directly (compatibility)
    pub fn insert_fast_str(&mut self, key: FastStr, value: V) -> Result<Option<V>> {
        if let Some(s) = key.as_str() {
            self.insert(s, value)
        } else {
            self.insert(&String::from_utf8_lossy(key.as_bytes()), value)
        }
    }

    /// Get an iterator over key-value pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &V)> {
        self.map.iter()
    }

    /// Get an iterator over values
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }

    /// Get an iterator over keys
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.map.clear();
        self.total_inserts = 0;
        self.unique_keys = 0;
    }

    /// Clear all entries (alias for clear)
    pub fn clear_all(&mut self) {
        self.clear();
    }

    /// Shrink internal structures to minimize memory usage
    pub fn shrink_to_fit(&mut self) {
        self.map.shrink_to_fit();
    }

    /// Get detailed statistics about the map
    pub fn statistics(&self) -> HashStrMapStats {
        HashStrMapStats {
            entries: self.len(),
            total_strings: self.total_inserts,
            unique_strings: self.unique_keys,
            interning_ratio: self.interning_ratio(),
            string_memory: self.string_memory_usage(),
            map_memory: self.map.capacity() * (std::mem::size_of::<String>() + std::mem::size_of::<V>()),
        }
    }
}

/// Statistics about HashStrMap performance
#[derive(Debug, Clone)]
pub struct HashStrMapStats {
    /// Number of key-value pairs
    pub entries: usize,
    /// Total number of string operations
    pub total_strings: usize,
    /// Number of unique strings stored
    pub unique_strings: usize,
    /// String deduplication ratio (0.0 - 1.0)
    pub interning_ratio: f64,
    /// Memory used by string storage
    pub string_memory: usize,
    /// Memory used by hash map structure
    pub map_memory: usize,
}

impl<V> Default for HashStrMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() -> Result<()> {
        let mut map = HashStrMap::new();
        
        // Insert
        assert_eq!(map.insert("key1", 42)?, None);
        assert_eq!(map.insert("key2", 43)?, None);
        assert_eq!(map.len(), 2);

        // Get
        assert_eq!(map.get("key1"), Some(&42));
        assert_eq!(map.get("key2"), Some(&43));
        assert_eq!(map.get("key3"), None);

        // Contains
        assert!(map.contains_key("key1"));
        assert!(!map.contains_key("key3"));

        // Update
        assert_eq!(map.insert("key1", 44)?, Some(42));
        assert_eq!(map.get("key1"), Some(&44));

        // Remove
        assert_eq!(map.remove("key1"), Some(44));
        assert_eq!(map.get("key1"), None);
        assert_eq!(map.len(), 1);

        Ok(())
    }

    #[test]
    fn test_string_tracking() -> Result<()> {
        let mut map = HashStrMap::new();
        
        // Insert keys
        map.insert("user/john", 1)?;
        map.insert("user/jane", 2)?;
        map.insert("user/john", 3)?; // Duplicate key
        map.insert("admin/root", 4)?;
        
        let stats = map.statistics();
        assert_eq!(stats.entries, 3); // john was updated
        assert_eq!(stats.total_strings, 4); // 4 insert operations
        assert_eq!(stats.unique_strings, 3); // 3 unique keys
        
        Ok(())
    }

    #[test]
    fn test_fast_str_operations() -> Result<()> {
        let mut map = HashStrMap::new();
        
        // Insert using FastStr directly
        let key = FastStr::from_string("fast_key");
        map.insert_fast_str(key, 42)?;
        
        assert_eq!(map.get("fast_key"), Some(&42));
        
        Ok(())
    }

    #[test]
    fn test_iterators() -> Result<()> {
        let mut map = HashStrMap::new();
        
        map.insert("a", 1)?;
        map.insert("b", 2)?;
        map.insert("c", 3)?;
        
        // Test iteration
        let mut keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
        
        let mut values: Vec<&i32> = map.values().collect();
        values.sort();
        assert_eq!(values, vec![&1, &2, &3]);
        
        assert_eq!(map.iter().count(), 3);
        
        Ok(())
    }

    #[test]
    fn test_statistics() -> Result<()> {
        let mut map = HashStrMap::new();
        
        for i in 0..100 {
            let key = format!("key_{}", i % 10); // Reuse some keys
            map.insert(&key, i)?;
        }
        
        let stats = map.statistics();
        assert_eq!(stats.entries, 10); // Only 10 unique keys
        assert_eq!(stats.total_strings, 100); // 100 insert operations
        assert_eq!(stats.unique_strings, 10); // 10 unique keys
        assert!(stats.interning_ratio > 0.8); // High deduplication ratio
        
        Ok(())
    }
}