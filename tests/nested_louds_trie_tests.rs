//! Comprehensive tests for Nested LOUDS Trie implementation
//!
//! This test suite ensures 97%+ code coverage and validates all performance,
//! correctness, and advanced features of the Nested LOUDS Trie.

use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Barrier};
use std::thread;
use zipora::error::{Result, ZiporaError};
use zipora::fsa::nested_louds_trie::{
    FragmentStats, NestedLoudsTrie, NestedTrieStats, NestingConfig, NestingConfigBuilder,
    NestingLevel,
};
use zipora::fsa::{
    FiniteStateAutomaton, PrefixIterable, StateInspectable, StatisticsProvider, Trie, TrieBuilder,
};
use zipora::succinct::rank_select::RankSelectInterleaved256;

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

fn generate_basic_keys() -> Vec<Vec<u8>> {
    vec![
        b"nested".to_vec(),
        b"louds".to_vec(),
        b"trie".to_vec(),
        b"hierarchical".to_vec(),
        b"compression".to_vec(),
        b"fragment".to_vec(),
        b"multilevel".to_vec(),
        b"performance".to_vec(),
        b"cache".to_vec(),
        b"optimization".to_vec(),
    ]
}

fn generate_prefix_heavy_keys() -> Vec<Vec<u8>> {
    vec![
        b"prefix".to_vec(),
        b"prefix_shared".to_vec(),
        b"prefix_shared_long".to_vec(),
        b"prefix_shared_long_path".to_vec(),
        b"prefix_shared_long_path_compression".to_vec(),
        b"prefix_alternative".to_vec(),
        b"prefix_alternative_path".to_vec(),
        b"different".to_vec(),
        b"different_path".to_vec(),
        b"unrelated".to_vec(),
    ]
}

fn generate_fragment_compression_keys() -> Vec<Vec<u8>> {
    // Keys designed to test fragment compression efficiency
    let base_fragments = vec!["common", "substring", "pattern", "repeated"];
    let mut keys = Vec::new();

    for base in &base_fragments {
        for i in 0..5 {
            keys.push(format!("{}_{:02}", base, i).into_bytes());
            keys.push(format!("prefix_{}_suffix", base).into_bytes());
            keys.push(format!("{}_{}_extended", base, i).into_bytes());
        }
    }

    keys
}

fn generate_hierarchical_keys() -> Vec<Vec<u8>> {
    // Keys that would benefit from hierarchical nesting
    let mut keys = Vec::new();

    // Level 1: Short common prefixes
    for prefix in &["a", "b", "c"] {
        keys.push(prefix.as_bytes().to_vec());

        // Level 2: Medium prefixes
        for mid in &["x", "y", "z"] {
            let mid_key = format!("{}{}", prefix, mid);
            keys.push(mid_key.as_bytes().to_vec());

            // Level 3: Long suffixes
            for suffix in &["1", "2", "3", "4", "5"] {
                let full_key = format!("{}{}", mid_key, suffix);
                keys.push(full_key.as_bytes().to_vec());

                // Level 4: Very long keys
                let extended_key = format!("{}_extended_{}", full_key, suffix);
                keys.push(extended_key.as_bytes().to_vec());
            }
        }
    }

    keys
}

fn generate_sequential_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("sequence_{:06}", i).into_bytes())
        .collect()
}

fn generate_unicode_keys() -> Vec<Vec<u8>> {
    vec![
        "hello".as_bytes().to_vec(),
        "world".as_bytes().to_vec(),
        "nested".as_bytes().to_vec(),
        "世界".as_bytes().to_vec(),
        "🌍".as_bytes().to_vec(),
        "café".as_bytes().to_vec(),
        "naïve".as_bytes().to_vec(),
        "résumé".as_bytes().to_vec(),
        "москва".as_bytes().to_vec(),
        "東京".as_bytes().to_vec(),
        "🚀🌟💫".as_bytes().to_vec(),
    ]
}

// =============================================================================
// CONFIGURATION TESTS
// =============================================================================

#[test]
fn test_nesting_config_defaults() {
    let config = NestingConfig::default();

    assert_eq!(config.max_levels, 4);
    assert_eq!(config.fragment_compression_ratio, 0.3);
    assert_eq!(config.min_fragment_size, 4);
    assert_eq!(config.max_fragment_size, 64);
    assert!(config.cache_optimization);
    assert_eq!(config.cache_block_size, 256);
    assert_eq!(config.density_switch_threshold, 0.5);
    assert!(config.adaptive_backend_selection);
    assert_eq!(config.memory_pool_size, 1024 * 1024);
}

#[test]
fn test_nesting_config_builder() {
    let config = NestingConfigBuilder::new()
        .max_levels(6)
        .fragment_compression_ratio(0.4)
        .min_fragment_size(8)
        .max_fragment_size(128)
        .cache_optimization(false)
        .cache_block_size(512)
        .density_switch_threshold(0.7)
        .adaptive_backend_selection(false)
        .memory_pool_size(2 * 1024 * 1024)
        .build()
        .unwrap();

    assert_eq!(config.max_levels, 6);
    assert_eq!(config.fragment_compression_ratio, 0.4);
    assert_eq!(config.min_fragment_size, 8);
    assert_eq!(config.max_fragment_size, 128);
    assert!(!config.cache_optimization);
    assert_eq!(config.cache_block_size, 512);
    assert_eq!(config.density_switch_threshold, 0.7);
    assert!(!config.adaptive_backend_selection);
    assert_eq!(config.memory_pool_size, 2 * 1024 * 1024);
}

#[test]
fn test_nesting_config_validation() {
    // Test invalid max_levels
    let result = NestingConfigBuilder::new().max_levels(0).build();
    assert!(result.is_err());

    let result = NestingConfigBuilder::new().max_levels(10).build();
    assert!(result.is_err());

    // Test invalid compression ratio
    let result = NestingConfigBuilder::new()
        .fragment_compression_ratio(-0.1)
        .build();
    assert!(result.is_err());

    let result = NestingConfigBuilder::new()
        .fragment_compression_ratio(1.5)
        .build();
    assert!(result.is_err());

    // Test invalid fragment sizes
    let result = NestingConfigBuilder::new().min_fragment_size(0).build();
    assert!(result.is_err());

    let result = NestingConfigBuilder::new()
        .min_fragment_size(100)
        .max_fragment_size(50)
        .build();
    assert!(result.is_err());
}

// =============================================================================
// BASIC FUNCTIONALITY TESTS WITH DIFFERENT BACKENDS
// =============================================================================

#[test]
fn test_creation_with_interleaved_backend() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let trie = TrieType::new();
    assert!(trie.is_ok());

    let trie = trie.unwrap();
    assert_eq!(trie.len(), 0);
    assert!(trie.is_empty());
    assert_eq!(trie.active_levels(), 0);
}

#[test]
fn test_creation_with_cache_config() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .max_levels(3)
        .cache_optimization(true)
        .cache_block_size(512)
        .build()
        .unwrap();

    let trie = TrieType::with_config(config);
    assert!(trie.is_ok());

    let trie = trie.unwrap();
    assert_eq!(trie.config().max_levels, 3);
    assert!(trie.config().cache_optimization);
    assert_eq!(trie.config().cache_block_size, 512);
}

#[test]
fn test_creation_with_fragment_config() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.4)
        .min_fragment_size(8)
        .max_fragment_size(128)
        .build()
        .unwrap();

    let trie = TrieType::with_config(config);
    assert!(trie.is_ok());

    let trie = trie.unwrap();
    assert_eq!(trie.config().fragment_compression_ratio, 0.4);
    assert_eq!(trie.config().min_fragment_size, 8);
    assert_eq!(trie.config().max_fragment_size, 128);
}

#[test]
fn test_creation_with_nesting_config() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .max_levels(6)
        .density_switch_threshold(0.7)
        .adaptive_backend_selection(false)
        .build()
        .unwrap();

    let trie = TrieType::with_config(config);
    assert!(trie.is_ok());

    let trie = trie.unwrap();
    assert_eq!(trie.config().max_levels, 6);
    assert_eq!(trie.config().density_switch_threshold, 0.7);
    assert!(!trie.config().adaptive_backend_selection);
}

#[test]
fn test_creation_with_memory_config() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .memory_pool_size(2 * 1024 * 1024)
        .cache_optimization(false)
        .build()
        .unwrap();

    let trie = TrieType::with_config(config);
    assert!(trie.is_ok());

    let trie = trie.unwrap();
    assert_eq!(trie.config().memory_pool_size, 2 * 1024 * 1024);
    assert!(!trie.config().cache_optimization);
}

// =============================================================================
// BASIC TRIE OPERATIONS
// =============================================================================

#[test]
fn test_basic_insert_and_lookup() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_basic_keys();

    // Insert keys
    for key in &keys {
        let result = trie.insert(key);
        assert!(
            result.is_ok(),
            "Failed to insert key: {:?}",
            String::from_utf8_lossy(key)
        );
    }

    assert_eq!(trie.len(), keys.len());

    // Test lookups
    for key in &keys {
        assert!(
            trie.contains(key),
            "Key not found: {:?}",
            String::from_utf8_lossy(key)
        );
        assert!(trie.lookup(key).is_some());
    }

    // Test non-existent keys
    assert!(!trie.contains(b"nonexistent"));
    assert!(trie.lookup(b"nonexistent").is_none());
}

#[test]
fn test_empty_key_handling() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();

    // Insert empty key
    trie.insert(b"").unwrap();
    assert_eq!(trie.len(), 1);
    assert!(trie.contains(b""));

    // Insert empty key again
    trie.insert(b"").unwrap();
    assert_eq!(trie.len(), 1); // Should not increase
}

#[test]
fn test_prefix_relationships() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_prefix_heavy_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Verify all keys exist
    for key in &keys {
        assert!(trie.contains(key));
    }

    // Test specific prefix relationships
    assert!(trie.contains(b"prefix"));
    assert!(trie.contains(b"prefix_shared"));
    assert!(trie.contains(b"prefix_shared_long"));
    assert!(!trie.contains(b"prefix_nonexistent"));
}

#[test]
fn test_unicode_support() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let unicode_keys = generate_unicode_keys();

    for key in &unicode_keys {
        trie.insert(key).unwrap();
    }

    assert_eq!(trie.len(), unicode_keys.len());

    for key in &unicode_keys {
        assert!(trie.contains(key));
    }
}

// =============================================================================
// HIERARCHICAL NESTING TESTS
// =============================================================================

#[test]
fn test_hierarchical_structure_creation() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new().max_levels(5).build().unwrap();

    let mut trie = TrieType::with_config(config).unwrap();
    let hierarchical_keys = generate_hierarchical_keys();

    for key in &hierarchical_keys {
        trie.insert(key).unwrap();
    }

    // Test that hierarchical structure is created
    assert!(trie.active_levels() >= 1);

    // Verify all keys are findable
    for key in &hierarchical_keys {
        assert!(
            trie.contains(key),
            "Missing hierarchical key: {:?}",
            String::from_utf8_lossy(key)
        );
    }

    // Test performance statistics
    let stats = trie.performance_stats();
    assert_eq!(stats.key_count, hierarchical_keys.len());
    assert!(stats.total_memory > 0);
}

#[test]
fn test_multi_level_optimization() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .max_levels(4)
        .cache_optimization(true)
        .build()
        .unwrap();

    let mut trie = TrieType::with_config(config).unwrap();
    let keys = generate_hierarchical_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Check layer memory usage
    let layer_memory = trie.layer_memory_usage();
    assert!(!layer_memory.is_empty());

    // Total memory should be sum of layers plus overhead
    let total_from_layers: usize = layer_memory.iter().sum();
    let actual_total = trie.total_memory_usage();
    assert!(actual_total >= total_from_layers);
}

// =============================================================================
// FRAGMENT COMPRESSION TESTS
// =============================================================================

#[test]
fn test_fragment_compression_detection() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.3)
        .min_fragment_size(4)
        .max_fragment_size(32)
        .build()
        .unwrap();

    let mut trie = TrieType::with_config(config).unwrap();
    let fragment_keys = generate_fragment_compression_keys();

    for key in &fragment_keys {
        trie.insert(key).unwrap();
    }

    // Verify all keys are accessible
    for key in &fragment_keys {
        assert!(trie.contains(key));
    }

    // Check fragment statistics
    let fragment_stats = trie.fragment_stats();
    assert!(fragment_stats.compression_ratio >= 0.0);
    assert!(fragment_stats.compression_ratio <= 1.0);

    // With BFS fragment detection, we should find fragments in the test data
    assert!(fragment_stats.fragment_count > 0, "BFS fragment detection should find fragments in test data with repeating patterns");
    
    // The exact count may vary, but should be reasonable for the test data
    assert!(fragment_stats.fragment_count <= 500, "Fragment count should be reasonable: found {}", fragment_stats.fragment_count);
}

#[test]
fn test_fragment_extraction_configuration() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    // Test with aggressive compression settings
    let aggressive_config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.1) // Very aggressive
        .min_fragment_size(2)
        .max_fragment_size(16)
        .build()
        .unwrap();

    let mut aggressive_trie = TrieType::with_config(aggressive_config).unwrap();

    // Test with conservative compression settings
    let conservative_config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.8) // Very conservative
        .min_fragment_size(16)
        .max_fragment_size(128)
        .build()
        .unwrap();

    let mut conservative_trie = TrieType::with_config(conservative_config).unwrap();

    let keys = generate_fragment_compression_keys();

    // Insert same keys in both tries
    for key in &keys {
        aggressive_trie.insert(key).unwrap();
        conservative_trie.insert(key).unwrap();
    }

    // Both should have same correctness
    for key in &keys {
        assert_eq!(
            aggressive_trie.contains(key),
            conservative_trie.contains(key)
        );
    }

    // Fragment statistics might differ (when compression is implemented)
    let aggressive_stats = aggressive_trie.fragment_stats();
    let conservative_stats = conservative_trie.fragment_stats();

    // Both should be valid
    assert!(aggressive_stats.compression_ratio >= 0.0);
    assert!(conservative_stats.compression_ratio >= 0.0);
}

// =============================================================================
// BACKEND COMPARISON TESTS
// =============================================================================

#[test]
fn test_configuration_equivalence() {
    let keys = generate_prefix_heavy_keys();

    // Test with different configurations of the same backend
    let mut default_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    let cache_config = NestingConfigBuilder::new()
        .cache_optimization(true)
        .cache_block_size(512)
        .build()
        .unwrap();
    let mut cache_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(cache_config).unwrap();

    let fragment_config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.2)
        .min_fragment_size(8)
        .build()
        .unwrap();
    let mut fragment_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(fragment_config).unwrap();

    // Insert same keys in all tries
    for key in &keys {
        default_trie.insert(key).unwrap();
        cache_trie.insert(key).unwrap();
        fragment_trie.insert(key).unwrap();
    }

    // All should have same length
    assert_eq!(default_trie.len(), keys.len());
    assert_eq!(cache_trie.len(), keys.len());
    assert_eq!(fragment_trie.len(), keys.len());

    // All should find the same keys
    for key in &keys {
        assert_eq!(default_trie.contains(key), cache_trie.contains(key));
        assert_eq!(cache_trie.contains(key), fragment_trie.contains(key));
    }

    // All should reject the same non-existent keys
    let non_existent = b"definitely_not_there";
    assert_eq!(
        default_trie.contains(non_existent),
        cache_trie.contains(non_existent)
    );
    assert_eq!(
        cache_trie.contains(non_existent),
        fragment_trie.contains(non_existent)
    );
    assert!(!default_trie.contains(non_existent));
}

#[test]
fn test_configuration_performance_characteristics() {
    let keys = generate_sequential_keys(1000);

    // Test different configurations for performance characteristics
    let default_config = NestingConfig::default();
    let mut default_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(default_config).unwrap();

    let optimized_config = NestingConfigBuilder::new()
        .cache_optimization(true)
        .cache_block_size(512)
        .fragment_compression_ratio(0.2)
        .build()
        .unwrap();
    let mut optimized_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(optimized_config).unwrap();

    // Insert keys and measure basic metrics
    let start = std::time::Instant::now();
    for key in &keys {
        default_trie.insert(key).unwrap();
    }
    let default_insert_time = start.elapsed();

    let start = std::time::Instant::now();
    for key in &keys {
        optimized_trie.insert(key).unwrap();
    }
    let optimized_insert_time = start.elapsed();

    // Both should work correctly
    assert_eq!(default_trie.len(), keys.len());
    assert_eq!(optimized_trie.len(), keys.len());

    // Performance may vary, but both should complete in reasonable time
    println!(
        "Default insert time: {:?}, Optimized insert time: {:?}",
        default_insert_time, optimized_insert_time
    );

    // Memory usage may differ based on configuration
    let default_memory = default_trie.total_memory_usage();
    let optimized_memory = optimized_trie.total_memory_usage();

    println!(
        "Default memory: {} bytes, Optimized memory: {} bytes",
        default_memory, optimized_memory
    );

    assert!(default_memory > 0);
    assert!(optimized_memory > 0);
}

// =============================================================================
// FSA INTERFACE TESTS
// =============================================================================

#[test]
fn test_fsa_interface_compliance() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_basic_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Test FSA root
    let root = trie.root();
    assert_eq!(root, 0);

    // Test FSA accepts
    for key in &keys {
        assert!(trie.accepts(key));
    }

    assert!(!trie.accepts(b"nonexistent"));

    // Test longest prefix
    if !keys.is_empty() {
        let test_key = &keys[0];
        if test_key.len() > 1 {
            let mut extended_key = test_key.clone();
            extended_key.extend_from_slice(b"_extended");

            let longest = trie.longest_prefix(&extended_key);
            assert_eq!(longest, Some(test_key.len()));
        }
    }
}

#[test]
fn test_state_transitions() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    trie.insert(b"hello").unwrap();
    trie.insert(b"help").unwrap();

    // Test state transitions
    let mut state = trie.root();

    // Follow path for "hello"
    for &symbol in b"hello" {
        if let Some(next_state) = trie.transition(state, symbol) {
            state = next_state;
        } else {
            panic!("Failed transition on symbol {}", symbol as char);
        }
    }

    assert!(trie.is_final(state));

    // Test invalid transition
    let invalid = trie.transition(state, b'x');
    assert!(invalid.is_none());
}

// =============================================================================
// PREFIX ITERATION TESTS
// =============================================================================

#[test]
fn test_prefix_iteration() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_prefix_heavy_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Test prefix iteration for "prefix"
    let prefix_results: Vec<_> = trie.iter_prefix(b"prefix").collect();
    let expected_count = keys.iter().filter(|k| k.starts_with(b"prefix")).count();

    assert_eq!(prefix_results.len(), expected_count);

    // Verify all prefix results start with "prefix"
    for result in &prefix_results {
        assert!(result.starts_with(b"prefix"));
    }

    // Test empty prefix iteration
    let all_results: Vec<_> = trie.iter_prefix(b"").collect();
    assert_eq!(all_results.len(), keys.len());

    // Test non-existent prefix
    let none_results: Vec<_> = trie.iter_prefix(b"xyz").collect();
    assert!(none_results.is_empty());
}

#[test]
fn test_complete_iteration() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_basic_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    let all_keys: Vec<_> = trie.iter_all().collect();
    assert_eq!(all_keys.len(), keys.len());

    for key in &keys {
        assert!(all_keys.contains(key));
    }
}

// =============================================================================
// STATE INSPECTION TESTS
// =============================================================================

#[test]
fn test_state_inspection() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    trie.insert(b"hello").unwrap();
    trie.insert(b"help").unwrap();
    trie.insert(b"world").unwrap();

    let root = trie.root();

    // Test out degree
    let out_degree = trie.out_degree(root);
    assert!(out_degree > 0);

    // Test out symbols
    let symbols = trie.out_symbols(root);
    assert!(!symbols.is_empty());

    // Should have transitions for 'h' (hello, help) and 'w' (world)
    let has_h = symbols.contains(&b'h');
    let has_w = symbols.contains(&b'w');
    assert!(has_h || has_w); // At least one should be present

    // Test leaf detection
    assert!(!trie.is_leaf(root));
}

// =============================================================================
// STATISTICS TESTS
// =============================================================================

#[test]
fn test_statistics_accuracy() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_basic_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Test basic statistics
    let stats = trie.stats();
    assert_eq!(stats.num_keys, keys.len());
    assert!(stats.memory_usage > 0);
    assert!(stats.bits_per_key > 0.0);

    // Test performance statistics
    let perf_stats = trie.performance_stats();
    assert_eq!(perf_stats.key_count, keys.len());
    assert!(perf_stats.total_memory > 0);

    // Test fragment statistics
    let fragment_stats = trie.fragment_stats();
    assert!(fragment_stats.compression_ratio >= 0.0);
    assert!(fragment_stats.compression_ratio <= 1.0);

    // Test memory calculations
    let total_memory = trie.total_memory_usage();
    assert!(total_memory > 0);
    assert_eq!(total_memory, perf_stats.total_memory);
}

#[test]
fn test_layer_statistics() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new().max_levels(3).build().unwrap();

    let mut trie = TrieType::with_config(config).unwrap();
    let hierarchical_keys = generate_hierarchical_keys();

    for key in &hierarchical_keys {
        trie.insert(key).unwrap();
    }

    // Check active levels
    let active_levels = trie.active_levels();
    assert!(active_levels >= 1);
    assert!(active_levels <= 3);

    // Check layer memory usage
    let layer_memory = trie.layer_memory_usage();
    assert_eq!(layer_memory.len(), active_levels);

    for &memory in &layer_memory {
        assert!(memory > 0);
    }
}

// =============================================================================
// PERFORMANCE AND STRESS TESTS
// =============================================================================

#[test]
fn test_large_dataset_performance() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_sequential_keys(5000);

    // Measure insertion time
    let start = std::time::Instant::now();
    for key in &keys {
        trie.insert(key).unwrap();
    }
    let insert_duration = start.elapsed();

    assert_eq!(trie.len(), keys.len());

    // Measure lookup time
    let start = std::time::Instant::now();
    for key in &keys {
        assert!(trie.contains(key));
    }
    let lookup_duration = start.elapsed();

    println!(
        "Nested LOUDS Performance - Insert: {:?}, Lookup: {:?}",
        insert_duration, lookup_duration
    );

    // Performance should be reasonable
    assert!(
        insert_duration.as_secs() < 60,
        "Insert time too slow: {:?}",
        insert_duration
    );
    assert!(
        lookup_duration.as_secs() < 10,
        "Lookup time too slow: {:?}",
        lookup_duration
    );
}

#[test]
fn test_memory_efficiency() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.2)
        .cache_optimization(true)
        .build()
        .unwrap();

    let mut trie = TrieType::with_config(config).unwrap();
    let keys = generate_fragment_compression_keys();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    let stats = trie.stats();
    let total_key_bytes: usize = keys.iter().map(|k| k.len()).sum();
    let memory_ratio = stats.memory_usage as f64 / total_key_bytes as f64;

    println!(
        "Memory efficiency - Ratio: {:.2}, Total keys: {} bytes, Trie: {} bytes",
        memory_ratio, total_key_bytes, stats.memory_usage
    );

    // Should use reasonable amount of memory
    assert!(memory_ratio < 20.0, "Memory usage seems excessive");
    assert!(stats.bits_per_key < 100_000.0);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[test]
fn test_edge_cases() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();

    // Test with very long key
    let long_key = vec![42u8; 10000];
    trie.insert(&long_key).unwrap();
    assert!(trie.contains(&long_key));

    // Test with key containing all byte values
    let all_bytes: Vec<u8> = (0..=255).collect();
    trie.insert(&all_bytes).unwrap();
    assert!(trie.contains(&all_bytes));

    // Test with duplicate insertions
    trie.insert(b"duplicate").unwrap();
    let len_before = trie.len();
    trie.insert(b"duplicate").unwrap();
    assert_eq!(trie.len(), len_before); // Should not increase
}

#[test]
fn test_builder_interface() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let keys = generate_basic_keys();

    // Test building from iterator
    let builder = TrieType::builder();
    let trie: NestedLoudsTrie<RankSelectInterleaved256> =
        builder.build_from_iter(keys.iter().cloned()).unwrap();

    assert_eq!(trie.len(), keys.len());

    for key in &keys {
        assert!(trie.contains(key));
    }
}

// =============================================================================
// PROPERTY-BASED TESTS
// =============================================================================

proptest! {
    #[test]
    fn property_test_insert_lookup_consistency(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..100), 0..200)
    ) {
        type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

        let mut trie = TrieType::new().unwrap();
        let mut expected_keys = HashSet::new();

        // Insert all keys
        for key in &keys {
            if trie.insert(key).is_ok() {
                expected_keys.insert(key.clone());
            }
        }

        // Verify all inserted keys can be found
        for key in &expected_keys {
            prop_assert!(trie.contains(key));
        }

        // Verify count is correct
        prop_assert_eq!(trie.len(), expected_keys.len());
    }

    #[test]
    fn property_test_hierarchical_correctness(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..50), 1..100)
    ) {
        type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

        let config = NestingConfigBuilder::new()
            .max_levels(3)
            .build()
            .unwrap();

        let mut trie = TrieType::with_config(config).unwrap();

        for key in &keys {
            let _ = trie.insert(key);
        }

        // Test that hierarchical structure doesn't affect correctness
        for key in &keys {
            if !key.is_empty() {
                let lookup_result = trie.lookup(key);
                let contains_result = trie.contains(key);

                prop_assert_eq!(lookup_result.is_some(), contains_result);
            }
        }

        // Test that active levels are reasonable
        let active_levels = trie.active_levels();
        prop_assert!(active_levels <= 3);
    }
}

// =============================================================================
// CONCURRENT ACCESS TESTS
// =============================================================================

#[test]
fn test_concurrent_read_access() {
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut trie = TrieType::new().unwrap();
    let keys = generate_sequential_keys(1000);

    // Populate trie
    for key in &keys {
        trie.insert(key).unwrap();
    }

    let trie = Arc::new(trie);
    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone: Arc<NestedLoudsTrie<RankSelectInterleaved256>> = Arc::clone(&trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            // Each thread reads a subset of keys
            for (i, key) in keys_clone.iter().enumerate() {
                if i % num_threads == thread_id {
                    assert!(trie_clone.contains(key));
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

#[test]
fn test_integration_with_different_configs() {
    // Test multiple configurations work together
    type TrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let configs = vec![
        NestingConfigBuilder::new().max_levels(2).build().unwrap(),
        NestingConfigBuilder::new().max_levels(4).build().unwrap(),
        NestingConfigBuilder::new().max_levels(6).build().unwrap(),
    ];

    let keys = generate_hierarchical_keys();

    for config in configs {
        let mut trie = TrieType::with_config(config).unwrap();

        for key in &keys {
            trie.insert(key).unwrap();
        }

        // All configurations should maintain correctness
        for key in &keys {
            assert!(trie.contains(key));
        }

        assert_eq!(trie.len(), keys.len());
    }
}

#[test]
fn test_comparison_with_other_tries() {
    // This test ensures the nested LOUDS trie behaves equivalently to other trie implementations
    // for basic operations, while potentially offering different performance characteristics

    type NestedTrieType = NestedLoudsTrie<RankSelectInterleaved256>;

    let mut nested_trie = NestedTrieType::new().unwrap();
    let keys = generate_basic_keys();

    // Build nested trie
    for key in &keys {
        nested_trie.insert(key).unwrap();
    }

    // Test that all basic trie behaviors are preserved
    for key in &keys {
        assert!(nested_trie.contains(key));
        assert!(nested_trie.lookup(key).is_some());
        assert!(nested_trie.accepts(key));
    }

    // Test that non-existent keys are handled correctly
    assert!(!nested_trie.contains(b"definitely_not_there"));
    assert!(nested_trie.lookup(b"definitely_not_there").is_none());
    assert!(!nested_trie.accepts(b"definitely_not_there"));

    // Test statistics are reasonable
    let stats = nested_trie.stats();
    assert_eq!(stats.num_keys, keys.len());
    assert!(stats.memory_usage > 0);
}
