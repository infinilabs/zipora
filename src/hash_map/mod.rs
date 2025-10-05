//! High-performance hash map implementation
//!
//! **ZiporaHashMap**: Single, highly optimized hash map implementation with
//! strategy-based configuration for different use cases. Inspired by referenced project's
//! focused implementation philosophy.
//!
//! # Performance-First Design
//!
//! Following referenced project's approach: **"One excellent implementation per data structure"**
//! instead of maintaining multiple separate implementations.
//!
//! # Examples
//!
//! ```rust
//! use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};
//! use std::collections::hash_map::RandomState;
//!
//! // Default high-performance configuration
//! let mut map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::new().unwrap();
//! map.insert("key", "value").unwrap();
//!
//! // Cache-optimized for NUMA systems
//! let mut cache_map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::cache_optimized()
//! ).unwrap();
//!
//! // String-optimized with interning and SIMD
//! let mut str_map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::string_optimized()
//! ).unwrap();
//!
//! // Small inline storage for ≤N elements
//! let mut small_map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::small_inline(8)
//! ).unwrap();
//!
//! // High-performance concurrent with memory pool
//! let pool = zipora::memory::SecureMemoryPool::new(
//!     zipora::memory::SecurePoolConfig::small_secure()
//! ).expect("Failed to create memory pool");
//! let mut concurrent_map: ZiporaHashMap<&str, &str, RandomState> = ZiporaHashMap::with_config(
//!     ZiporaHashMapConfig::concurrent_pool(pool)
//! ).unwrap();
//! ```

// Core implementation modules
mod zipora_hash_map;
mod strategy_traits;

// Utility modules (keep these as they're used by core implementation)
mod hash_functions;
mod simd_string_ops;
mod cache_locality;

// Core ZiporaHashMap implementation
pub use zipora_hash_map::{
    ZiporaHashMap, ZiporaHashMapConfig, HashMapStats,
    HashStrategy, StorageStrategy, OptimizationStrategy,
};

// Strategy traits for advanced configuration
pub use strategy_traits::{
    CollisionResolutionStrategy, StorageLayoutStrategy, HashOptimizationStrategy,
    HashBucket, ProbeStats, OptimizationMetrics, OptimizationHint,
    RobinHoodStrategy, RobinHoodConfig, RobinHoodContext,
    StandardStorageStrategy, StandardStorageConfig,
    CacheOptimizedStorageStrategy, CacheOptimizedStorageConfig,
    SimdOptimizationStrategy, SimdOptimizationConfig, SimdOptimizationContext,
};

// Export hash function utilities
pub use hash_functions::{
    fabo_hash_combine_u32, fabo_hash_combine_u64, golden_ratio_next_size, optimal_bucket_count,
    advanced_hash_combine, HashFunctionBuilder, CombineStrategy, HashCombinable,
    GOLDEN_RATIO_FRAC_NUM, GOLDEN_RATIO_FRAC_DEN, GOLDEN_LOAD_FACTOR,
    // Export BMI2-specific functions and types for advanced usage
    bmi2_hash_combine_u32, bmi2_hash_combine_u64, extract_hash_bucket_bmi2,
    Bmi2HashDispatcher, HashOptimizationTier, ProbeType,
    bmi2_collision_resolution, fast_string_hash_bmi2, specialized,
    get_global_bmi2_dispatcher, hash_with_bmi2, hash_combine_with_bmi2, extract_bucket_with_bmi2,
};

// Export SIMD string operations
pub use simd_string_ops::{SimdStringOps, SimdTier, get_global_simd_ops};

// Export cache locality optimizations
pub use cache_locality::{
    CacheMetrics, CacheAligned, PrefetchHint, Prefetcher,
    CacheLayoutOptimizer, CacheLevel, AccessPattern,
    HotColdSeparator, NumaAllocator, AccessPatternAnalyzer,
    CacheConsciousResizer, CACHE_LINE_SIZE,
};


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zipora_hash_map_basic() {
        let mut map: ZiporaHashMap<&str, &str> = ZiporaHashMap::new().unwrap();
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        assert_eq!(map.insert("key", "value").unwrap(), None);
        assert_eq!(map.get("key"), Some(&"value"));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_cache_optimized_config() {
        let map: ZiporaHashMap<String, i32> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::cache_optimized()).unwrap();
        assert_eq!(map.len(), 0);
        // Verify cache optimization is enabled
        let _ = map.cache_metrics(); // Ensure cache metrics are accessible
    }

    #[test]
    fn test_string_optimized_config() {
        let map: ZiporaHashMap<String, i32> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::string_optimized()).unwrap();
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_small_inline_config() {
        let map: ZiporaHashMap<i32, String> =
            ZiporaHashMap::with_config(ZiporaHashMapConfig::small_inline(4)).unwrap();
        assert_eq!(map.len(), 0);
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

