//! High-performance hash map implementations
//!
//! This module provides specialized hash map implementations optimized for different use cases:
//! - `GoldHashMap`: High-performance general-purpose hash map with AHash
//! - `GoldenRatioHashMap`: Enhanced hash map with golden ratio growth and FaboHashCombine
//! - `StringOptimizedHashMap`: String-specific optimizations with interning and SIMD
//! - `SmallHashMap`: Inline storage for small collections with zero allocations

mod gold_hash_map;
mod hash_functions;
mod golden_ratio_hash_map;
mod small_hash_map;
mod string_optimized_hash_map;
mod simd_string_ops;

// Re-export existing types
pub use gold_hash_map::{GoldHashMap, Iter};

// Export new specialized hash map types  
pub use golden_ratio_hash_map::GoldenRatioHashMap;

// Export iterator types with module-qualified names to avoid conflicts
pub use golden_ratio_hash_map::{
    Iter as GoldenRatioIter,
    Keys as GoldenRatioKeys, 
    Values as GoldenRatioValues
};
pub use string_optimized_hash_map::{StringOptimizedHashMap, StringArenaStats, StringMapIter, StringMapKeys, StringMapValues};
pub use small_hash_map::{SmallHashMap, SmallMapIter, SmallMapKeys, SmallMapValues};

// Export hash function utilities
pub use hash_functions::{
    fabo_hash_combine_u32, fabo_hash_combine_u64, golden_ratio_next_size, optimal_bucket_count,
    advanced_hash_combine, HashFunctionBuilder, CombineStrategy, HashCombinable,
    GOLDEN_RATIO_FRAC_NUM, GOLDEN_RATIO_FRAC_DEN, GOLDEN_LOAD_FACTOR,
};

// Export SIMD string operations
pub use simd_string_ops::{SimdStringOps, SimdTier, get_global_simd_ops};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test existing GoldHashMap
        let _gold_map = GoldHashMap::<i32, String>::new();

        // Test new specialized hash maps
        let _golden_ratio_map = GoldenRatioHashMap::<i32, String>::new();
        let _string_map = StringOptimizedHashMap::<i32>::new();
        let _small_map: SmallHashMap<i32, String, 4> = SmallHashMap::new();

        // Test hash function utilities
        let result = fabo_hash_combine_u32(0x12345678, 0xabcdef00);
        assert_ne!(result, 0);

        let next_size = golden_ratio_next_size(100);
        assert!(next_size > 100);

        let bucket_count = optimal_bucket_count(100);
        assert!(bucket_count.is_power_of_two());
    }

    #[test]
    fn test_golden_ratio_hash_map_basic() {
        let mut map = GoldenRatioHashMap::new();
        assert_eq!(map.insert("test", 42).unwrap(), None);
        assert_eq!(map.get("test"), Some(&42));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_string_optimized_hash_map_basic() {
        let mut map = StringOptimizedHashMap::new();
        assert_eq!(map.insert("hello", 42).unwrap(), None);
        assert_eq!(map.insert("world", 84).unwrap(), None);
        assert_eq!(map.get("hello"), Some(&42));
        assert_eq!(map.len(), 2);

        let stats = map.string_arena_stats();
        assert_eq!(stats.unique_strings, 2);
    }

    #[test]
    fn test_small_hash_map_basic() {
        let mut map: SmallHashMap<&str, i32, 4> = SmallHashMap::new();
        assert!(map.is_inline());

        assert_eq!(map.insert("a", 1).unwrap(), None);
        assert_eq!(map.insert("b", 2).unwrap(), None);
        assert_eq!(map.len(), 2);
        assert!(map.is_inline());

        assert_eq!(map.get("a"), Some(&1));
        assert_eq!(map.get("b"), Some(&2));
    }

    #[test]
    fn test_small_hash_map_conversion() {
        let mut map: SmallHashMap<i32, i32, 2> = SmallHashMap::new();
        
        // Fill inline capacity
        map.insert(1, 10).unwrap();
        map.insert(2, 20).unwrap();
        assert!(map.is_inline());

        // This should trigger conversion to large storage
        map.insert(3, 30).unwrap();
        assert!(!map.is_inline());
        assert_eq!(map.len(), 3);

        // Verify all values are accessible
        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn test_hash_function_utilities() {
        // Test FaboHashCombine
        let hash1 = fabo_hash_combine_u32(0x12345678, 0xabcdef00);
        let hash2 = fabo_hash_combine_u32(0x12345678, 0xabcdef00);
        assert_eq!(hash1, hash2); // Should be deterministic

        let hash3 = fabo_hash_combine_u32(0x12345678, 0xabcdef01);
        assert_ne!(hash1, hash3); // Should be different

        // Test golden ratio utilities
        let sizes = [16, 64, 100, 1000];
        for &size in &sizes {
            let next_size = golden_ratio_next_size(size);
            assert!(next_size > size);
            
            let ratio = next_size as f64 / size as f64;
            assert!(ratio > 1.5 && ratio < 1.7); // Should approximate golden ratio
        }

        // Test optimal bucket count
        for &capacity in &sizes {
            let bucket_count = optimal_bucket_count(capacity);
            assert!(bucket_count.is_power_of_two());
            assert!(bucket_count >= capacity);
        }

        // Test advanced hash combine
        let hashes = [0x123456789abcdef0u64, 0xfedcba9876543210u64];
        let combined = advanced_hash_combine(&hashes);
        assert_ne!(combined, hashes[0]);
        assert_ne!(combined, hashes[1]);
        
        // Should be deterministic
        assert_eq!(combined, advanced_hash_combine(&hashes));
    }

    #[test]
    fn test_hash_function_builder() {
        let builder = HashFunctionBuilder::new()
            .with_strategy(CombineStrategy::Fabo);
        
        let hash_fn = builder.build_u32();
        let result = hash_fn(0x12345678, 0xabcdef00);
        
        // Should be deterministic
        assert_eq!(result, hash_fn(0x12345678, 0xabcdef00));

        // Test different strategies
        let strategies = [
            CombineStrategy::Addition,
            CombineStrategy::Xor,
            CombineStrategy::Fabo,
            CombineStrategy::Advanced,
        ];

        let mut results = Vec::new();
        for strategy in strategies {
            let builder = HashFunctionBuilder::new().with_strategy(strategy);
            let hash_fn = builder.build_u32();
            let result = hash_fn(0x12345678, 0xabcdef00);
            results.push(result);
        }

        // All strategies should produce valid results and be deterministic
        for (i, &result) in results.iter().enumerate() {
            assert_ne!(result, 0, "Strategy {} produced zero result", i);
            
            // Test determinism
            let builder = HashFunctionBuilder::new().with_strategy(strategies[i]);
            let hash_fn = builder.build_u32();
            let result2 = hash_fn(0x12345678, 0xabcdef00);
            assert_eq!(result, result2, "Strategy {} not deterministic", i);
        }
    }
}
