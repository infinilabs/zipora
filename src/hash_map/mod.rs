//! High-performance hash map implementations
//!
//! This module provides the GoldHashMap, a high-performance general-purpose hash map
//! optimized for speed and memory efficiency. The implementation uses ahash for fast
//! hashing and custom memory management for optimal performance.

mod gold_hash_map;

pub use gold_hash_map::GoldHashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that the module properly exports GoldHashMap
        let _map = GoldHashMap::<i32, String>::new();
    }
}