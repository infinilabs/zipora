//! Configuration demonstration example.
//!
//! This example shows how to use the unified configuration APIs in Zipora 2.0.

use zipora::fsa::{ZiporaTrieConfig, TrieStrategy, StorageStrategy, CompressionStrategy, RankSelectType};
use zipora::hash_map::{ZiporaHashMapConfig, HashStrategy, OptimizationStrategy};
use zipora::hash_map::StorageStrategy as HashStorageStrategy;
use zipora::memory::{MemoryConfig, PoolConfig};
use zipora::Result;

fn main() -> Result<()> {
    println!("Zipora 2.0 Unified Configuration API Demo");
    println!("==========================================");

    // 1. ZiporaTrie Configuration Examples
    println!("\n1. ZiporaTrie Configurations");

    // Default configuration
    let default_trie_config = ZiporaTrieConfig::default();
    println!("   Default trie config: SIMD={}, concurrency={}",
             default_trie_config.enable_simd, default_trie_config.enable_concurrency);

    // Performance-optimized configuration
    let perf_trie_config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: 1024,
            growth_factor: 2.0,
            free_list_management: true,
            auto_shrink: false,
        },
        storage_strategy: StorageStrategy::CacheOptimized {
            cache_line_size: 64,
            numa_aware: true,
            prefetch_enabled: true,
        },
        compression_strategy: CompressionStrategy::None,
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: true,
        cache_optimization: true,
    };
    println!("   Performance config: strategy=DoubleArray, cache_optimized=true");

    // Memory-optimized configuration
    let memory_trie_config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::CompressedSparse {
            sparse_threshold: 0.3,
            compression_level: 6,
            adaptive_sparse: true,
        },
        storage_strategy: StorageStrategy::Standard {
            initial_capacity: 64,
            growth_factor: 1.2,
        },
        compression_strategy: CompressionStrategy::PathCompression {
            min_path_length: 2,
            max_path_length: 64,
            adaptive_threshold: true,
        },
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: false,
        cache_optimization: false,
    };
    println!("   Memory config: strategy=CompressedSparse, compression=PathCompression");

    // 2. ZiporaHashMap Configuration Examples
    println!("\n2. ZiporaHashMap Configurations");

    // Default hash map configuration
    let default_map_config = ZiporaHashMapConfig::default();
    println!("   Default map config: load_factor={:.2}", default_map_config.load_factor);

    // High-performance configuration
    let perf_map_config = ZiporaHashMapConfig {
        hash_strategy: HashStrategy::RobinHood {
            max_probe_distance: 16,
            variance_reduction: true,
            backward_shift: true,
        },
        storage_strategy: HashStorageStrategy::CacheOptimized {
            cache_line_size: 64,
            numa_aware: true,
            huge_pages: false,
        },
        optimization_strategy: OptimizationStrategy::CacheAware {
            access_pattern_tracking: true,
            hot_cold_separation: true,
            prefetch_distance: 2,
        },
        initial_capacity: 256,
        load_factor: 0.75,
    };
    println!("   Performance config: strategy=RobinHood, cache_optimized=true");

    // 3. Memory Configuration Examples
    println!("\n3. Memory Pool Configurations");

    let memory_config = MemoryConfig {
        use_pools: true,
        use_hugepages: false,
        pool_chunk_size: 64 * 1024, // 64KB
        max_pool_memory: 128 * 1024 * 1024, // 128MB
    };
    println!("   Memory config: pools={}, hugepages={}",
             memory_config.use_pools, memory_config.use_hugepages);

    let pool_config = PoolConfig {
        chunk_size: 1024 * 1024, // 1MB
        max_chunks: 128, // 128MB total
        alignment: 64,
    };
    println!("   Pool config: chunk_size={}MB, max_chunks={}",
             pool_config.chunk_size / (1024 * 1024),
             pool_config.max_chunks);

    // 4. Configuration Pattern Examples
    println!("\n4. Configuration Patterns");

    println!("   Creating specialized configurations...");

    // String-optimized trie
    let string_config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::Patricia {
            max_path_length: 128,
            compression_threshold: 4,
            adaptive_compression: true,
        },
        storage_strategy: StorageStrategy::Standard {
            initial_capacity: 256,
            growth_factor: 1.5,
        },
        compression_strategy: CompressionStrategy::FragmentCompression {
            fragment_size: 8,
            frequency_threshold: 0.3,
            dictionary_size: 256,
        },
        rank_select_type: RankSelectType::Interleaved256,
        enable_simd: true,
        enable_concurrency: false,
        cache_optimization: true,
    };
    println!("   String-optimized: Patricia + FragmentCompression");

    // Small data optimized hash map
    let small_map_config = ZiporaHashMapConfig {
        hash_strategy: HashStrategy::Chaining {
            load_factor: 0.9,
            hash_cache: false,
            compact_links: true,
        },
        storage_strategy: HashStorageStrategy::SmallInline {
            inline_capacity: 4,
            fallback_threshold: 8,
        },
        optimization_strategy: OptimizationStrategy::Standard,
        initial_capacity: 8,
        load_factor: 0.9,
    };
    println!("   Small-data optimized: inline_capacity=4, fallback_threshold=8");
    // 5. Demonstrating configuration usage
    println!("\n5. Using Configurations");

    println!("   Suppressed variables for unused configs to avoid warnings.");
    let _ = default_trie_config;
    let _ = perf_trie_config;
    let _ = memory_trie_config;
    let _ = default_map_config;
    let _ = perf_map_config;
    let _ = memory_config;
    let _ = pool_config;
    let _ = string_config;
    let _ = small_map_config;

    println!("   All configurations created successfully!");
    println!("   In a real application, these would be used to create");
    println!("   ZiporaTrie and ZiporaHashMap instances with:");
    println!("     ZiporaTrie::with_config(config)");
    println!("     ZiporaHashMap::with_config(config)");

    println!("\n6. Configuration Benefits");
    println!("   ✅ Strategy-based configuration for different use cases");
    println!("   ✅ Type-safe configuration with compile-time validation");
    println!("   ✅ Performance vs memory trade-offs clearly expressed");
    println!("   ✅ SIMD and concurrency control per configuration");
    println!("   ✅ Cache optimization and NUMA awareness settings");

    println!("\nUnified Configuration API demonstration completed successfully! ✅");
    Ok(())
}