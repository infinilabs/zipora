//! Comprehensive integration tests for the configuration system.
//! 
//! This module provides thorough testing of configuration functionality including
//! validation, serialization, environment variable parsing, and preset usage.

use super::*;
use crate::config::nest_louds_trie::{CompressionAlgorithm, OptimizationFlags};
use crate::config::memory::{AllocationStrategy, CacheOptimizationLevel};
use crate::error::ZiporaError;
use std::env;
use std::fs;
use tempfile::tempdir;

/// Test all configuration types can be created with default values
#[test]
fn test_all_config_defaults() {
    let nest_config = NestLoudsTrieConfig::default();
    assert!(nest_config.validate().is_ok());
    
    let blob_config = BlobStoreConfig::default();
    assert!(blob_config.validate().is_ok());
    
    let memory_config = MemoryConfig::default();
    assert!(memory_config.validate().is_ok());
    
    let compression_config = CompressionConfig::default();
    assert!(compression_config.validate().is_ok());
    
    let cache_config = CacheConfig::default();
    assert!(cache_config.validate().is_ok());
    
    let simd_config = SIMDConfig::default();
    assert!(simd_config.validate().is_ok());
}

/// Test all preset configurations are valid
#[test]
fn test_all_presets_valid() {
    // Test NestLoudsTrieConfig presets
    assert!(NestLoudsTrieConfig::performance_preset().validate().is_ok());
    assert!(NestLoudsTrieConfig::memory_preset().validate().is_ok());
    assert!(NestLoudsTrieConfig::realtime_preset().validate().is_ok());
    assert!(NestLoudsTrieConfig::balanced_preset().validate().is_ok());
    
    // Test BlobStoreConfig presets
    assert!(BlobStoreConfig::performance_preset().validate().is_ok());
    assert!(BlobStoreConfig::memory_preset().validate().is_ok());
    assert!(BlobStoreConfig::realtime_preset().validate().is_ok());
    assert!(BlobStoreConfig::balanced_preset().validate().is_ok());
    
    // Test MemoryConfig presets
    assert!(MemoryConfig::performance_preset().validate().is_ok());
    assert!(MemoryConfig::memory_preset().validate().is_ok());
    assert!(MemoryConfig::realtime_preset().validate().is_ok());
    assert!(MemoryConfig::balanced_preset().validate().is_ok());
    
    // Test CompressionConfig presets
    assert!(CompressionConfig::performance_preset().validate().is_ok());
    assert!(CompressionConfig::memory_preset().validate().is_ok());
    assert!(CompressionConfig::realtime_preset().validate().is_ok());
    assert!(CompressionConfig::balanced_preset().validate().is_ok());
    
    // Test CacheConfig presets
    assert!(CacheConfig::performance_preset().validate().is_ok());
    assert!(CacheConfig::memory_preset().validate().is_ok());
    assert!(CacheConfig::realtime_preset().validate().is_ok());
    assert!(CacheConfig::balanced_preset().validate().is_ok());
    
    // Test SIMDConfig presets
    assert!(SIMDConfig::performance_preset().validate().is_ok());
    assert!(SIMDConfig::memory_preset().validate().is_ok());
    assert!(SIMDConfig::realtime_preset().validate().is_ok());
    assert!(SIMDConfig::balanced_preset().validate().is_ok());
}

/// Test that different presets have meaningful differences
#[test]
fn test_preset_characteristics() {
    // NestLoudsTrieConfig preset differences
    let perf = NestLoudsTrieConfig::performance_preset();
    let mem = NestLoudsTrieConfig::memory_preset();
    let rt = NestLoudsTrieConfig::realtime_preset();
    
    // Performance preset should favor speed
    assert_eq!(perf.nest_level, 2); // Lower nesting for speed
    assert_eq!(perf.core_str_compression_level, 3); // Fast compression
    assert!(perf.speedup_nest_trie_build);
    assert!(perf.parallel_threads == 0); // Use all cores
    
    // Memory preset should favor compression
    assert_eq!(mem.nest_level, 5); // Higher nesting for compression
    assert_eq!(mem.core_str_compression_level, 15); // High compression
    assert!(mem.enable_queue_compression);
    assert_eq!(mem.parallel_threads, 1); // Single-threaded to save memory
    
    // Real-time preset should be predictable
    assert_eq!(rt.nest_level, 2); // Reduced nesting for predictability
    assert_eq!(rt.core_str_compression_level, 1); // Minimal compression
    assert!(!rt.enable_queue_compression); // Avoid compression overhead
    assert_eq!(rt.parallel_threads, 2); // Limited parallelism
    
    // Memory Config preset differences
    let mem_perf = MemoryConfig::performance_preset();
    let mem_mem = MemoryConfig::memory_preset();
    let mem_rt = MemoryConfig::realtime_preset();
    
    // Performance should use more memory
    assert!(mem_perf.initial_pool_size > mem_mem.initial_pool_size);
    assert!(mem_perf.num_pools > mem_mem.num_pools);
    assert!(mem_perf.huge_page_config.enable_huge_pages);
    
    // Memory preset should minimize usage
    assert_eq!(mem_mem.max_pool_size, 128 * 1024 * 1024); // Fixed limit
    assert!(mem_mem.enable_compaction);
    assert!(!mem_mem.huge_page_config.enable_huge_pages);
    
    // Real-time should have fixed allocation
    assert_eq!(mem_rt.initial_pool_size, mem_rt.max_pool_size); // Fixed size
    assert_eq!(mem_rt.growth_factor, 1.0); // No growth
    assert!(!mem_rt.enable_compaction); // Avoid unpredictable latency
}

/// Test environment variable parsing
#[test]
fn test_environment_variable_parsing() {
    // Set test environment variables
    unsafe { env::set_var("TEST_TRIE_NEST_LEVEL", "5"); }
    unsafe { env::set_var("TEST_TRIE_COMPRESSION_LEVEL", "9"); }
    unsafe { env::set_var("TEST_TRIE_ENABLE_STATISTICS", "true"); }
    unsafe { env::set_var("TEST_BLOB_COMPRESSION_LEVEL", "12"); }
    unsafe { env::set_var("TEST_MEMORY_INITIAL_POOL_SIZE", "134217728"); } // 128MB
    
    // Test NestLoudsTrieConfig environment parsing
    let nest_config = NestLoudsTrieConfig::from_env_with_prefix("TEST_")
        .expect("Failed to parse NestLoudsTrieConfig from environment");
    assert_eq!(nest_config.nest_level, 5);
    assert_eq!(nest_config.core_str_compression_level, 9);
    assert!(nest_config.enable_statistics);
    
    // Test BlobStoreConfig environment parsing
    let blob_config = BlobStoreConfig::from_env_with_prefix("TEST_")
        .expect("Failed to parse BlobStoreConfig from environment");
    assert_eq!(blob_config.compression_level, 12);
    
    // Test MemoryConfig environment parsing
    let memory_config = MemoryConfig::from_env_with_prefix("TEST_")
        .expect("Failed to parse MemoryConfig from environment");
    assert_eq!(memory_config.initial_pool_size, 134217728);
    
    // Clean up environment variables
    unsafe { env::remove_var("TEST_TRIE_NEST_LEVEL"); }
    unsafe { env::remove_var("TEST_TRIE_COMPRESSION_LEVEL"); }
    unsafe { env::remove_var("TEST_TRIE_ENABLE_STATISTICS"); }
    unsafe { env::remove_var("TEST_BLOB_COMPRESSION_LEVEL"); }
    unsafe { env::remove_var("TEST_MEMORY_INITIAL_POOL_SIZE"); }
}

/// Test boolean environment variable parsing
#[test]
fn test_environment_boolean_parsing() {
    // Test various boolean representations
    let test_cases = [
        ("true", true),
        ("TRUE", true),
        ("True", true),
        ("1", true),
        ("yes", true),
        ("YES", true),
        ("on", true),
        ("ON", true),
        ("false", false),
        ("FALSE", false),
        ("False", false),
        ("0", false),
        ("no", false),
        ("NO", false),
        ("off", false),
        ("OFF", false),
        ("invalid", false),
        ("", false),
    ];
    
    for (value, expected) in test_cases.iter() {
        unsafe { env::set_var("TEST_BOOL_VALUE", value); }
        let result = parse_env_bool("TEST_BOOL_VALUE", false);
        assert_eq!(result, *expected, "Failed for value: '{}'", value);
    }
    
    unsafe { env::remove_var("TEST_BOOL_VALUE"); }
}

/// Test file serialization and deserialization
#[test]
fn test_file_serialization() -> Result<()> {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    // Test NestLoudsTrieConfig serialization
    let nest_config = NestLoudsTrieConfig::builder()
        .nest_level(4)
        .compression_level(8)
        .enable_statistics(true)
        .build()?;
    
    let nest_file_path = temp_dir.path().join("nest_config.json");
    nest_config.save_to_file(&nest_file_path)?;
    
    let loaded_nest_config = NestLoudsTrieConfig::load_from_file(&nest_file_path)?;
    assert_eq!(nest_config.nest_level, loaded_nest_config.nest_level);
    assert_eq!(nest_config.core_str_compression_level, loaded_nest_config.core_str_compression_level);
    assert_eq!(nest_config.enable_statistics, loaded_nest_config.enable_statistics);
    
    // Test MemoryConfig serialization
    let memory_config = MemoryConfig::builder()
        .initial_pool_size(256 * 1024 * 1024)
        .enable_numa(true)
        .alignment(128)
        .build()?;
    
    let memory_file_path = temp_dir.path().join("memory_config.json");
    memory_config.save_to_file(&memory_file_path)?;
    
    let loaded_memory_config = MemoryConfig::load_from_file(&memory_file_path)?;
    assert_eq!(memory_config.initial_pool_size, loaded_memory_config.initial_pool_size);
    assert_eq!(memory_config.numa_config.enable_numa_awareness, loaded_memory_config.numa_config.enable_numa_awareness);
    assert_eq!(memory_config.alignment, loaded_memory_config.alignment);
    
    Ok(())
}

/// Test configuration validation edge cases
#[test]
fn test_validation_edge_cases() {
    // Test NestLoudsTrieConfig validation
    let mut nest_config = NestLoudsTrieConfig::default();
    
    // Test invalid nest level
    nest_config.nest_level = 0;
    assert!(nest_config.validate().is_err());
    
    nest_config.nest_level = 17;
    assert!(nest_config.validate().is_err());
    
    // Test invalid compression level
    nest_config = NestLoudsTrieConfig::default();
    nest_config.core_str_compression_level = 23;
    assert!(nest_config.validate().is_err());
    
    // Test invalid load factor
    nest_config = NestLoudsTrieConfig::default();
    nest_config.load_factor = 0.0;
    assert!(nest_config.validate().is_err());
    
    nest_config.load_factor = 1.0;
    assert!(nest_config.validate().is_err());
    
    // Test MemoryConfig validation
    let mut memory_config = MemoryConfig::default();
    
    // Test invalid initial pool size
    memory_config.initial_pool_size = 0;
    assert!(memory_config.validate().is_err());
    
    // Test invalid growth factor
    memory_config = MemoryConfig::default();
    memory_config.growth_factor = 0.5;
    assert!(memory_config.validate().is_err());
    
    memory_config.growth_factor = 5.0;
    assert!(memory_config.validate().is_err());
    
    // Test invalid alignment (not power of 2)
    memory_config = MemoryConfig::default();
    memory_config.alignment = 3;
    assert!(memory_config.validate().is_err());
    
    memory_config.alignment = 0;
    assert!(memory_config.validate().is_err());
}

/// Test builder patterns
#[test]
fn test_builder_patterns() -> Result<()> {
    // Test NestLoudsTrieConfig builder
    let nest_config = NestLoudsTrieConfig::builder()
        .nest_level(4)
        .max_fragment_length(2048)
        .min_fragment_length(16)
        .compression_level(10)
        .compression_algorithm(CompressionAlgorithm::Zstd(15))
        .enable_queue_compression(true)
        .temp_directory("/tmp/zipora")
        .initial_pool_size(128 * 1024 * 1024)
        .enable_statistics(true)
        .enable_profiling(false)
        .optimization_flags(OptimizationFlags::ENABLE_FAST_SEARCH | OptimizationFlags::ENABLE_SIMD_ACCELERATION)
        .parallel_threads(4)
        .build()?;
    
    assert_eq!(nest_config.nest_level, 4);
    assert_eq!(nest_config.max_fragment_length, 2048);
    assert_eq!(nest_config.min_fragment_length, 16);
    assert_eq!(nest_config.core_str_compression_level, 10);
    assert!(matches!(nest_config.compression_algorithm, CompressionAlgorithm::Zstd(15)));
    assert!(nest_config.enable_queue_compression);
    assert_eq!(nest_config.temp_directory, "/tmp/zipora");
    assert_eq!(nest_config.initial_pool_size, 128 * 1024 * 1024);
    assert!(nest_config.enable_statistics);
    assert!(!nest_config.enable_profiling);
    assert!(nest_config.has_optimization_flag(OptimizationFlags::ENABLE_FAST_SEARCH));
    assert!(nest_config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION));
    assert_eq!(nest_config.parallel_threads, 4);
    
    // Test MemoryConfig builder
    let memory_config = MemoryConfig::builder()
        .allocation_strategy(AllocationStrategy::LockFree)
        .initial_pool_size(256 * 1024 * 1024)
        .max_pool_size(1024 * 1024 * 1024)
        .cache_optimization(CacheOptimizationLevel::Maximum)
        .enable_numa(true)
        .enable_huge_pages(true)
        .alignment(64)
        .num_pools(8)
        .enable_protection(true)
        .build()?;
    
    assert_eq!(memory_config.allocation_strategy, AllocationStrategy::LockFree);
    assert_eq!(memory_config.initial_pool_size, 256 * 1024 * 1024);
    assert_eq!(memory_config.max_pool_size, 1024 * 1024 * 1024);
    assert_eq!(memory_config.cache_optimization, CacheOptimizationLevel::Maximum);
    assert!(memory_config.numa_config.enable_numa_awareness);
    assert!(memory_config.huge_page_config.enable_huge_pages);
    assert_eq!(memory_config.alignment, 64);
    assert_eq!(memory_config.num_pools, 8);
    assert!(memory_config.enable_memory_protection);
    
    Ok(())
}

/// Test optimization flags manipulation
#[test]
fn test_optimization_flags() {
    let mut config = NestLoudsTrieConfig::default();
    
    // Test initial flags
    assert!(config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION));
    assert!(config.has_optimization_flag(OptimizationFlags::ENABLE_FAST_SEARCH));
    
    // Test adding a flag
    config.set_optimization_flag(OptimizationFlags::USE_HUGEPAGES, true);
    assert!(config.has_optimization_flag(OptimizationFlags::USE_HUGEPAGES));
    
    // Test removing a flag
    config.set_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION, false);
    assert!(!config.has_optimization_flag(OptimizationFlags::ENABLE_SIMD_ACCELERATION));
    
    // Test flag combinations
    let combined_flags = OptimizationFlags::ENABLE_FAST_SEARCH 
        | OptimizationFlags::USE_MIXED_CORE_LINK 
        | OptimizationFlags::ENABLE_PARALLEL_CONSTRUCTION;
    
    config.optimization_flags = combined_flags;
    assert!(config.has_optimization_flag(OptimizationFlags::ENABLE_FAST_SEARCH));
    assert!(config.has_optimization_flag(OptimizationFlags::USE_MIXED_CORE_LINK));
    assert!(config.has_optimization_flag(OptimizationFlags::ENABLE_PARALLEL_CONSTRUCTION));
    assert!(!config.has_optimization_flag(OptimizationFlags::USE_HUGEPAGES));
}

/// Test memory config effective values
#[test]
fn test_memory_config_effective_values() {
    let config = MemoryConfig::default();
    
    // Test effective cache line size
    let cache_line_size = config.effective_cache_line_size();
    assert!(cache_line_size >= 32 && cache_line_size <= 128);
    
    // Test effective num pools for different strategies
    let mut config = MemoryConfig::default();
    
    config.allocation_strategy = AllocationStrategy::System;
    assert_eq!(config.effective_num_pools(), 1);
    
    config.allocation_strategy = AllocationStrategy::LockFree;
    config.num_pools = 16;
    assert_eq!(config.effective_num_pools(), 16);
    
    config.allocation_strategy = AllocationStrategy::ThreadLocal;
    assert!(config.effective_num_pools() >= 1); // Should be >= number of CPUs
    
    config.allocation_strategy = AllocationStrategy::FixedCapacity;
    assert_eq!(config.effective_num_pools(), 1);
}

/// Test JSON serialization compatibility
#[test]
fn test_json_serialization_compatibility() -> Result<()> {
    // Test NestLoudsTrieConfig JSON serialization
    let nest_config = NestLoudsTrieConfig::performance_preset();
    let json = serde_json::to_string_pretty(&nest_config)?;
    let deserialized: NestLoudsTrieConfig = serde_json::from_str(&json)?;
    
    // Verify critical fields
    assert_eq!(nest_config.nest_level, deserialized.nest_level);
    assert_eq!(nest_config.optimization_flags, deserialized.optimization_flags);
    assert_eq!(nest_config.enable_statistics, deserialized.enable_statistics);
    
    // Test MemoryConfig JSON serialization
    let memory_config = MemoryConfig::memory_preset();
    let json = serde_json::to_string_pretty(&memory_config)?;
    let deserialized: MemoryConfig = serde_json::from_str(&json)?;
    
    // Verify critical fields
    assert_eq!(memory_config.allocation_strategy, deserialized.allocation_strategy);
    assert_eq!(memory_config.numa_config.enable_numa_awareness, deserialized.numa_config.enable_numa_awareness);
    assert_eq!(memory_config.huge_page_config.enable_huge_pages, deserialized.huge_page_config.enable_huge_pages);
    
    Ok(())
}

/// Test error handling in configuration loading
#[test]
fn test_configuration_error_handling() {
    // Test invalid JSON
    let invalid_json = r#"{"nest_level": "invalid"}"#;
    let result = serde_json::from_str::<NestLoudsTrieConfig>(invalid_json);
    assert!(result.is_err());
    
    // Test file not found
    let result = NestLoudsTrieConfig::load_from_file("nonexistent_file.json");
    assert!(result.is_err());
    
    // Test invalid configuration after loading
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("invalid_config.json");
    
    // Create invalid configuration JSON
    let invalid_config_json = r#"{
        "nest_level": 0,
        "max_fragment_length": 1024,
        "optimization_flags": 0
    }"#;
    
    fs::write(&file_path, invalid_config_json).expect("Failed to write test file");
    
    let result = NestLoudsTrieConfig::load_from_file(&file_path);
    assert!(result.is_err());
    
    if let Err(ZiporaError::Configuration { message }) = result {
        assert!(message.contains("nest level must be between 1 and 16"));
    } else {
        panic!("Expected configuration validation error");
    }
}

/// Comprehensive integration test
#[test]
fn test_configuration_integration() -> Result<()> {
    // Create a comprehensive configuration using multiple config types
    let nest_config = NestLoudsTrieConfig::builder()
        .nest_level(3)
        .compression_level(6)
        .enable_statistics(true)
        .parallel_threads(4)
        .build()?;
    
    let memory_config = MemoryConfig::builder()
        .allocation_strategy(AllocationStrategy::SecurePool)
        .initial_pool_size(128 * 1024 * 1024)
        .enable_numa(true)
        .build()?;
    
    let blob_config = BlobStoreConfig::performance_preset();
    let compression_config = CompressionConfig::realtime_preset();
    let cache_config = CacheConfig::memory_preset();
    let simd_config = SIMDConfig::performance_preset();
    
    // Validate all configurations
    assert!(nest_config.validate().is_ok());
    assert!(memory_config.validate().is_ok());
    assert!(blob_config.validate().is_ok());
    assert!(compression_config.validate().is_ok());
    assert!(cache_config.validate().is_ok());
    assert!(simd_config.validate().is_ok());
    
    // Test serialization of all configurations
    let temp_dir = tempdir().expect("Failed to create temp directory");
    
    nest_config.save_to_file(temp_dir.path().join("nest.json"))?;
    memory_config.save_to_file(temp_dir.path().join("memory.json"))?;
    blob_config.save_to_file(temp_dir.path().join("blob.json"))?;
    compression_config.save_to_file(temp_dir.path().join("compression.json"))?;
    cache_config.save_to_file(temp_dir.path().join("cache.json"))?;
    simd_config.save_to_file(temp_dir.path().join("simd.json"))?;
    
    // Test loading and validation
    let loaded_nest = NestLoudsTrieConfig::load_from_file(temp_dir.path().join("nest.json"))?;
    let loaded_memory = MemoryConfig::load_from_file(temp_dir.path().join("memory.json"))?;
    let loaded_blob = BlobStoreConfig::load_from_file(temp_dir.path().join("blob.json"))?;
    let loaded_compression = CompressionConfig::load_from_file(temp_dir.path().join("compression.json"))?;
    let loaded_cache = CacheConfig::load_from_file(temp_dir.path().join("cache.json"))?;
    let loaded_simd = SIMDConfig::load_from_file(temp_dir.path().join("simd.json"))?;
    
    // Verify loaded configurations match originals
    assert_eq!(nest_config.nest_level, loaded_nest.nest_level);
    assert_eq!(memory_config.allocation_strategy, loaded_memory.allocation_strategy);
    assert_eq!(blob_config.compression_level, loaded_blob.compression_level);
    assert_eq!(compression_config.level, loaded_compression.level);
    assert_eq!(cache_config.size, loaded_cache.size);
    assert_eq!(simd_config.enable_simd, loaded_simd.enable_simd);
    
    Ok(())
}

#[cfg(test)]
mod benches {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark configuration creation
    #[test]
    fn bench_config_creation() {
        let iterations = 10000;
        
        // Benchmark default creation
        let start = Instant::now();
        for _ in 0..iterations {
            let _config = NestLoudsTrieConfig::default();
        }
        let default_duration = start.elapsed();
        println!("Default creation: {:.2}μs per config", 
                 default_duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        // Benchmark builder creation
        let start = Instant::now();
        for _ in 0..iterations {
            let _config = NestLoudsTrieConfig::builder()
                .nest_level(3)
                .compression_level(6)
                .build()
                .expect("Failed to build config");
        }
        let builder_duration = start.elapsed();
        println!("Builder creation: {:.2}μs per config", 
                 builder_duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        // Benchmark preset creation
        let start = Instant::now();
        for _ in 0..iterations {
            let _config = NestLoudsTrieConfig::performance_preset();
        }
        let preset_duration = start.elapsed();
        println!("Preset creation: {:.2}μs per config", 
                 preset_duration.as_nanos() as f64 / iterations as f64 / 1000.0);
    }
    
    /// Benchmark configuration validation
    #[test]
    fn bench_config_validation() {
        let config = NestLoudsTrieConfig::default();
        let iterations = 100000;
        
        let start = Instant::now();
        for _ in 0..iterations {
            assert!(config.validate().is_ok());
        }
        let duration = start.elapsed();
        println!("Validation: {:.2}μs per validation", 
                 duration.as_nanos() as f64 / iterations as f64 / 1000.0);
    }
    
    /// Benchmark configuration serialization
    #[test]
    fn bench_config_serialization() -> Result<()> {
        let config = NestLoudsTrieConfig::performance_preset();
        let iterations = 1000;
        
        // Benchmark JSON serialization
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = serde_json::to_string(&config)?;
        }
        let serialize_duration = start.elapsed();
        println!("JSON serialization: {:.2}μs per config", 
                 serialize_duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        // Benchmark JSON deserialization
        let json = serde_json::to_string(&config)?;
        let start = Instant::now();
        for _ in 0..iterations {
            let _config: NestLoudsTrieConfig = serde_json::from_str(&json)?;
        }
        let deserialize_duration = start.elapsed();
        println!("JSON deserialization: {:.2}μs per config", 
                 deserialize_duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        Ok(())
    }
    
    /// Benchmark environment variable parsing
    #[test]
    fn bench_env_parsing() -> Result<()> {
        // Set up environment variables
        unsafe { env::set_var("BENCH_TRIE_NEST_LEVEL", "3"); }
        unsafe { env::set_var("BENCH_TRIE_COMPRESSION_LEVEL", "6"); }
        unsafe { env::set_var("BENCH_TRIE_ENABLE_STATISTICS", "true"); }
        
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _config = NestLoudsTrieConfig::from_env_with_prefix("BENCH_")?;
        }
        let duration = start.elapsed();
        println!("Environment parsing: {:.2}μs per config", 
                 duration.as_nanos() as f64 / iterations as f64 / 1000.0);
        
        // Clean up
        unsafe { env::remove_var("BENCH_TRIE_NEST_LEVEL"); }
        unsafe { env::remove_var("BENCH_TRIE_COMPRESSION_LEVEL"); }
        unsafe { env::remove_var("BENCH_TRIE_ENABLE_STATISTICS"); }
        
        Ok(())
    }
}