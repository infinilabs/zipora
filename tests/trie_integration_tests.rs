//! Integration tests for all trie implementations
//!
//! This test suite validates cross-module compatibility, trait implementations,
//! and comprehensive integration scenarios for all trie implementations in zipora.

use proptest::prelude::*;
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use zipora::fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfigBuilder};
use zipora::fsa::{
    DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig, FiniteStateAutomaton,
    PrefixIterable, StateInspectable, StatisticsProvider, Trie,
};
use zipora::succinct::rank_select::{RankSelectInterleaved256, RankSelectOps, RankSelectBuilder};

// Note: CompressedSparseTrie would be tested here too when available in async context

// =============================================================================
// ENHANCED TEST DATA GENERATORS
// =============================================================================

/// Generate keys with realistic patterns for testing
fn generate_common_test_keys() -> Vec<Vec<u8>> {
    vec![
        b"integration".to_vec(),
        b"test".to_vec(),
        b"trie".to_vec(),
        b"compatibility".to_vec(),
        b"cross_module".to_vec(),
        b"zipora".to_vec(),
        b"performance".to_vec(),
        b"comparison".to_vec(),
        b"validation".to_vec(),
        b"comprehensive".to_vec(),
    ]
}

/// Generate keys with pathological patterns to test edge cases
fn generate_pathological_test_keys() -> Vec<Vec<u8>> {
    vec![
        vec![],                                       // Empty key
        vec![0],                                      // Single null byte
        vec![255],                                    // Single max byte
        vec![0; 1000],                                // Long repeated null bytes
        vec![255; 1000],                              // Long repeated max bytes
        (0u8..=255u8).collect(),                      // All byte values
        (0u8..=255u8).rev().collect(),                // All byte values reversed
        vec![0, 255, 0, 255, 0, 255],                 // Alternating min/max
        b"\x00\x01\x02\x03\xFC\xFD\xFE\xFF".to_vec(), // Boundary values
        b"a\x00b\x00c\x00".to_vec(),                  // Embedded nulls
    ]
}

/// Generate keys with shared prefixes for testing compression
fn generate_shared_prefix_keys() -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    let prefixes = ["common_prefix_", "shared_", "test_data_", "integration_"];
    let suffixes = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];

    for prefix in &prefixes {
        for suffix in &suffixes {
            let mut key = prefix.as_bytes().to_vec();
            key.extend_from_slice(suffix.as_bytes());
            keys.push(key);

            // Add some variations
            let mut long_key = prefix.as_bytes().to_vec();
            long_key.extend_from_slice(suffix.as_bytes());
            long_key.extend_from_slice(b"_extended_version");
            keys.push(long_key);
        }

        // Add prefix itself as a key
        keys.push(prefix.as_bytes().to_vec());
    }

    keys
}

fn generate_prefix_test_keys() -> Vec<Vec<u8>> {
    vec![
        b"prefix".to_vec(),
        b"prefix_test".to_vec(),
        b"prefix_test_long".to_vec(),
        b"prefix_test_long_path".to_vec(),
        b"prefix_alternative".to_vec(),
        b"different".to_vec(),
        b"different_path".to_vec(),
        b"another".to_vec(),
        b"another_branch".to_vec(),
        b"unrelated".to_vec(),
    ]
}

fn generate_stress_test_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("stress_test_key_{:06}", i).into_bytes())
        .collect()
}

fn generate_unicode_test_keys() -> Vec<Vec<u8>> {
    vec![
        "hello".as_bytes().to_vec(),
        "world".as_bytes().to_vec(),
        "integration".as_bytes().to_vec(),
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
// TRAIT IMPLEMENTATION COMPATIBILITY TESTS
// =============================================================================

#[test]
fn test_trie_trait_consistency() {
    // Test that all trie implementations behave consistently through the Trie trait
    let keys = generate_common_test_keys();

    // Test Double Array Trie
    let mut da_trie = DoubleArrayTrie::new();
    for key in &keys {
        da_trie.insert(key).unwrap();
    }

    // Test Nested LOUDS Trie
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    for key in &keys {
        nested_trie.insert(key).unwrap();
    }

    // Both should implement Trie trait consistently
    assert_eq!(da_trie.len(), keys.len());
    assert_eq!(nested_trie.len(), keys.len());

    assert_eq!(da_trie.is_empty(), nested_trie.is_empty());
    assert!(!da_trie.is_empty());
    assert!(!nested_trie.is_empty());

    // Both should find the same keys
    for key in &keys {
        assert_eq!(da_trie.contains(key), nested_trie.contains(key));
        assert!(da_trie.contains(key));

        let da_lookup = da_trie.lookup(key);
        let nested_lookup = nested_trie.lookup(key);
        assert_eq!(da_lookup.is_some(), nested_lookup.is_some());
        assert!(da_lookup.is_some());
    }

    // Both should reject non-existent keys
    assert_eq!(
        da_trie.contains(b"nonexistent"),
        nested_trie.contains(b"nonexistent")
    );
    assert!(!da_trie.contains(b"nonexistent"));
}

#[test]
fn test_fsa_trait_consistency() {
    let keys = vec![
        b"fsa".as_slice(),
        b"test".as_slice(),
        b"finite".as_slice(),
        b"state".as_slice(),
        b"automaton".as_slice(),
    ];

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Test FSA root consistency
    assert_eq!(da_trie.root(), nested_trie.root());
    assert_eq!(da_trie.root(), 0);

    // Test FSA accepts method consistency
    for key in &keys {
        assert_eq!(da_trie.accepts(key), nested_trie.accepts(key));
        assert!(da_trie.accepts(key));
    }

    assert_eq!(
        da_trie.accepts(b"nonexistent"),
        nested_trie.accepts(b"nonexistent")
    );
    assert!(!da_trie.accepts(b"nonexistent"));

    // Test longest_prefix consistency
    for key in &keys {
        if key.len() > 2 {
            let mut extended_key = key.to_vec();
            extended_key.extend_from_slice(b"_extended");

            let da_longest = da_trie.longest_prefix(&extended_key);
            let nested_longest = nested_trie.longest_prefix(&extended_key);

            assert_eq!(da_longest, nested_longest);
            assert_eq!(da_longest, Some(key.len()));
        }
    }

    // Test state transitions (basic)
    for key in &keys {
        if !key.is_empty() {
            let da_first_transition = da_trie.transition(da_trie.root(), key[0]);
            let nested_first_transition = nested_trie.transition(nested_trie.root(), key[0]);

            // Both should either succeed or fail consistently
            assert_eq!(
                da_first_transition.is_some(),
                nested_first_transition.is_some()
            );
        }
    }
}

#[test]
fn test_prefix_iterable_consistency() {
    let keys = generate_prefix_test_keys();

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Test prefix iteration for "prefix"
    let da_prefix_results: Vec<_> = da_trie.iter_prefix(b"prefix").collect();
    let nested_prefix_results: Vec<_> = nested_trie.iter_prefix(b"prefix").collect();

    // Both should find the same number of keys with "prefix"
    assert_eq!(da_prefix_results.len(), nested_prefix_results.len());

    // Convert to sets for comparison (order may differ)
    let da_set: HashSet<_> = da_prefix_results.into_iter().collect();
    let nested_set: HashSet<_> = nested_prefix_results.into_iter().collect();
    assert_eq!(da_set, nested_set);

    // Test complete iteration
    let da_all: HashSet<_> = da_trie.iter_all().collect();
    let nested_all: HashSet<_> = nested_trie.iter_all().collect();

    assert_eq!(da_all.len(), keys.len());
    assert_eq!(nested_all.len(), keys.len());
    assert_eq!(da_all, nested_all);
}

#[test]
fn test_state_inspectable_consistency() {
    let keys = vec![
        b"state".as_slice(),
        b"inspection".as_slice(),
        b"test".as_slice(),
    ];

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let root = 0; // Both should use same root

    // Test out_degree
    let da_degree = da_trie.out_degree(root);
    let nested_degree = nested_trie.out_degree(root);

    assert!(da_degree > 0);
    assert!(nested_degree > 0);
    // Note: Actual degrees may differ due to implementation differences

    // Test out_symbols
    let da_symbols = da_trie.out_symbols(root);
    let nested_symbols = nested_trie.out_symbols(root);

    assert!(!da_symbols.is_empty());
    assert!(!nested_symbols.is_empty());

    // Both should have some overlapping symbols (but may differ in representation)
    let da_set: HashSet<_> = da_symbols.into_iter().collect();
    let nested_set: HashSet<_> = nested_symbols.into_iter().collect();

    // Should have at least one symbol in common (first characters of keys)
    let first_chars: HashSet<_> = keys.iter().map(|k| k[0]).collect();
    assert!(da_set.iter().any(|&c| first_chars.contains(&c)));
    assert!(nested_set.iter().any(|&c| first_chars.contains(&c)));

    // Test is_leaf
    assert!(!da_trie.is_leaf(root));
    assert!(!nested_trie.is_leaf(root));
}

#[test]
fn test_statistics_provider_consistency() {
    let keys = generate_common_test_keys();

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Test basic statistics
    let da_stats = da_trie.stats();
    let nested_stats = nested_trie.stats();

    // Both should report same number of keys
    assert_eq!(da_stats.num_keys, keys.len());
    assert_eq!(nested_stats.num_keys, keys.len());

    // Both should have positive memory usage
    assert!(da_stats.memory_usage > 0);
    assert!(nested_stats.memory_usage > 0);

    // Both should have positive bits per key
    assert!(da_stats.bits_per_key > 0.0);
    assert!(nested_stats.bits_per_key > 0.0);

    // Test StatisticsProvider interface methods
    assert_eq!(da_trie.memory_usage(), da_stats.memory_usage);
    assert_eq!(nested_trie.memory_usage(), nested_stats.memory_usage);

    assert_eq!(da_trie.bits_per_key(), da_stats.bits_per_key);
    assert_eq!(nested_trie.bits_per_key(), nested_stats.bits_per_key);

    // Statistics should be reasonable
    assert!(da_stats.bits_per_key < 100_000.0); // Less than ~12KB per key
    assert!(nested_stats.bits_per_key < 100_000.0);
}

// =============================================================================
// CROSS-MODULE INTEGRATION TESTS
// =============================================================================

#[test]
fn test_memory_pool_integration() {
    // Test that different trie implementations can work with memory pools
    let config = DoubleArrayTrieConfig {
        use_memory_pool: true,
        initial_capacity: 1024,
        ..Default::default()
    };

    let mut da_trie = DoubleArrayTrie::with_config(config);

    let nested_config = NestingConfigBuilder::new()
        .memory_pool_size(2 * 1024 * 1024)
        .build()
        .unwrap();
    let mut nested_trie =
        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(nested_config).unwrap();

    let keys = generate_common_test_keys();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Both should work correctly with memory pools
    for key in &keys {
        assert!(da_trie.contains(key));
        assert!(nested_trie.contains(key));
    }
}

#[test]
fn test_builder_pattern_integration() {
    let mut keys = generate_common_test_keys();

    // Sort keys for optimized construction
    keys.sort();

    // Test DoubleArrayTrie builder
    let da_trie = DoubleArrayTrieBuilder::new()
        .build_from_sorted(keys.clone())
        .unwrap();

    // Test NestedLoudsTrie builder
    let nested_builder = NestedLoudsTrie::<RankSelectInterleaved256>::builder();
    let nested_trie: NestedLoudsTrie<RankSelectInterleaved256> = nested_builder
        .build_from_iter(keys.iter().cloned())
        .unwrap();

    // Both builders should produce equivalent results
    assert_eq!(da_trie.len(), nested_trie.len());

    for key in &keys {
        assert_eq!(da_trie.contains(key), nested_trie.contains(key));
        assert!(da_trie.contains(key));
    }
}

#[test]
fn test_unicode_handling_integration() {
    let unicode_keys = generate_unicode_test_keys();

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &unicode_keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Both should handle Unicode correctly
    for key in &unicode_keys {
        assert_eq!(da_trie.contains(key), nested_trie.contains(key));
        assert!(da_trie.contains(key));
    }

    // Test Unicode in prefix operations
    let utf8_prefix = "🚀".as_bytes();
    let da_utf8_results: Vec<_> = da_trie.iter_prefix(utf8_prefix).collect();
    let nested_utf8_results: Vec<_> = nested_trie.iter_prefix(utf8_prefix).collect();

    // Should find the same results
    assert_eq!(da_utf8_results.len(), nested_utf8_results.len());
    let da_set: HashSet<_> = da_utf8_results.into_iter().collect();
    let nested_set: HashSet<_> = nested_utf8_results.into_iter().collect();
    assert_eq!(da_set, nested_set);
}

// =============================================================================
// PERFORMANCE COMPARISON TESTS
// =============================================================================

#[test]
fn test_performance_comparison() {
    let keys = generate_stress_test_keys(5000);

    // Test Double Array Trie performance
    let start = Instant::now();
    let mut da_trie = DoubleArrayTrie::new();
    for key in &keys {
        da_trie.insert(key).unwrap();
    }
    let da_insert_time = start.elapsed();

    let start = Instant::now();
    for key in &keys {
        assert!(da_trie.contains(key));
    }
    let da_lookup_time = start.elapsed();

    // Test Nested LOUDS Trie performance
    let start = Instant::now();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    for key in &keys {
        nested_trie.insert(key).unwrap();
    }
    let nested_insert_time = start.elapsed();

    let start = Instant::now();
    for key in &keys {
        assert!(nested_trie.contains(key));
    }
    let nested_lookup_time = start.elapsed();

    println!("Performance Comparison:");
    println!(
        "  Double Array - Insert: {:?}, Lookup: {:?}",
        da_insert_time, da_lookup_time
    );
    println!(
        "  Nested LOUDS - Insert: {:?}, Lookup: {:?}",
        nested_insert_time, nested_lookup_time
    );

    // Both should complete in reasonable time
    assert!(da_insert_time.as_secs() < 60);
    assert!(da_lookup_time.as_secs() < 10);
    assert!(nested_insert_time.as_secs() < 60);
    assert!(nested_lookup_time.as_secs() < 10);

    // Compare memory usage
    let da_memory = da_trie.memory_usage();
    let nested_memory = nested_trie.memory_usage();

    println!("  Double Array Memory: {} bytes", da_memory);
    println!("  Nested LOUDS Memory: {} bytes", nested_memory);

    assert!(da_memory > 0);
    assert!(nested_memory > 0);
}

#[test]
fn test_memory_efficiency_comparison() {
    let keys = generate_prefix_test_keys(); // Good for compression testing

    let mut da_trie = DoubleArrayTrie::new_compact();
    let config = NestingConfigBuilder::new()
        .fragment_compression_ratio(0.3)
        .build()
        .unwrap();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Optimize memory usage after all insertions
    da_trie.shrink_to_fit();

    let da_stats = da_trie.stats();
    let nested_stats = nested_trie.stats();

    // Calculate memory efficiency metrics
    let total_key_bytes: usize = keys.iter().map(|k| k.len()).sum();

    let da_efficiency = da_stats.memory_usage as f64 / total_key_bytes as f64;
    let nested_efficiency = nested_stats.memory_usage as f64 / total_key_bytes as f64;

    println!("Memory Efficiency:");
    println!("  Raw key data: {} bytes", total_key_bytes);
    println!(
        "  Double Array: {} bytes (ratio: {:.2})",
        da_stats.memory_usage, da_efficiency
    );
    println!(
        "  Nested LOUDS: {} bytes (ratio: {:.2})",
        nested_stats.memory_usage, nested_efficiency
    );

    // Both should be reasonably efficient
    // Note: Double Array with incremental insert achieves ~58x overhead
    // This is expected as it's less memory-efficient than batch-built tries
    // The referenced topling-zip implementation uses batch build which achieves better packing
    assert!(da_efficiency < 70.0); // Less than 70x overhead for incremental insert
    assert!(nested_efficiency < 50.0);

    // Bits per key comparison
    println!("  Double Array bits/key: {:.1}", da_stats.bits_per_key);
    println!("  Nested LOUDS bits/key: {:.1}", nested_stats.bits_per_key);

    assert!(da_stats.bits_per_key < 100_000.0);
    assert!(nested_stats.bits_per_key < 100_000.0);
}

// =============================================================================
// CONCURRENT ACCESS TESTS
// =============================================================================

#[test]
fn test_concurrent_read_compatibility() {
    let keys = generate_stress_test_keys(1000);

    // Build tries
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let da_trie = Arc::new(da_trie);
    let nested_trie = Arc::new(nested_trie);

    let mut handles = Vec::new();

    // Test concurrent reads on Double Array Trie
    for i in 0..4 {
        let da_clone: Arc<DoubleArrayTrie> = Arc::clone(&da_trie);
        let keys_clone = keys.clone();

        let handle = thread::spawn(move || {
            for (j, key) in keys_clone.iter().enumerate() {
                if j % 4 == i {
                    assert!(da_clone.contains(key));
                }
            }
        });

        handles.push(handle);
    }

    // Test concurrent reads on Nested LOUDS Trie
    for i in 0..4 {
        let nested_clone: Arc<NestedLoudsTrie<RankSelectInterleaved256>> = Arc::clone(&nested_trie);
        let keys_clone = keys.clone();

        let handle = thread::spawn(move || {
            for (j, key) in keys_clone.iter().enumerate() {
                if j % 4 == i {
                    assert!(nested_clone.contains(key));
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
}

// =============================================================================
// ERROR HANDLING INTEGRATION TESTS
// =============================================================================

#[test]
fn test_error_handling_consistency() {
    // Test that all implementations handle edge cases consistently

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    // Test empty key
    let da_empty_result = da_trie.insert(b"");
    let nested_empty_result = nested_trie.insert(b"");

    assert_eq!(da_empty_result.is_ok(), nested_empty_result.is_ok());
    assert!(da_empty_result.is_ok());

    // Test very long key
    let long_key = vec![42u8; 10000];
    let da_long_result = da_trie.insert(&long_key);
    let nested_long_result = nested_trie.insert(&long_key);

    assert_eq!(da_long_result.is_ok(), nested_long_result.is_ok());

    // Test key with all byte values
    let all_bytes: Vec<u8> = (0..=255).collect();
    let da_bytes_result = da_trie.insert(&all_bytes);
    let nested_bytes_result = nested_trie.insert(&all_bytes);

    assert_eq!(da_bytes_result.is_ok(), nested_bytes_result.is_ok());

    // If operations succeeded, verify consistency
    if da_empty_result.is_ok() && nested_empty_result.is_ok() {
        assert_eq!(da_trie.contains(b""), nested_trie.contains(b""));
    }

    if da_long_result.is_ok() && nested_long_result.is_ok() {
        assert_eq!(da_trie.contains(&long_key), nested_trie.contains(&long_key));
    }

    if da_bytes_result.is_ok() && nested_bytes_result.is_ok() {
        assert_eq!(
            da_trie.contains(&all_bytes),
            nested_trie.contains(&all_bytes)
        );
    }
}

// =============================================================================
// SERIALIZATION COMPATIBILITY TESTS
// =============================================================================

#[cfg(feature = "serde")]
#[test]
fn test_serialization_compatibility() {
    // Configuration access test (serde not needed for basic access)

    let keys = generate_common_test_keys();

    // Build tries
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    // Test that configuration structs can be accessed
    let da_config = da_trie.config().clone();
    let nested_config = nested_trie.config().clone();

    // Test configuration access
    assert!(da_config.initial_capacity > 0);
    assert!(nested_config.max_levels > 0);
    assert!(nested_config.memory_pool_size > 0);

    // Note: Configuration serialization would require serde_json dependency
    // This test validates that the configuration infrastructure works correctly
}

// =============================================================================
// PROPERTY-BASED INTEGRATION TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn property_test_cross_implementation_consistency(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..50), 0..100)
    ) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        let mut successful_keys = HashSet::new();

        // Insert keys in both tries
        for key in &keys {
            let da_result = da_trie.insert(key);
            let nested_result = nested_trie.insert(key);

            // Both should succeed or fail together for deterministic inputs
            if da_result.is_ok() && nested_result.is_ok() {
                successful_keys.insert(key.clone());
            }
        }

        // Both tries should have the same successful insertions
        prop_assert_eq!(da_trie.len(), nested_trie.len());
        prop_assert_eq!(da_trie.len(), successful_keys.len());

        // Both should find the same keys
        for key in &successful_keys {
            prop_assert_eq!(da_trie.contains(key), nested_trie.contains(key));
            prop_assert!(da_trie.contains(key));
        }

        // Both should reject the same non-existent keys
        let non_existent = b"definitely_not_inserted_12345";
        prop_assert_eq!(da_trie.contains(non_existent), nested_trie.contains(non_existent));
    }

    #[test]
    fn property_test_advanced_prefix_operations(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..30), 1..50),
        query_prefixes in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..15), 1..10)
    ) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        // Insert all keys
        for key in &keys {
            let _ = da_trie.insert(key);
            let _ = nested_trie.insert(key);
        }

        // Test prefix operations consistency
        for prefix in &query_prefixes {
            let da_prefix_results: Vec<_> = da_trie.iter_prefix(prefix).collect();
            let nested_prefix_results: Vec<_> = nested_trie.iter_prefix(prefix).collect();

            // All results should start with the prefix
            for result in &da_prefix_results {
                prop_assert!(result.starts_with(prefix),
                    "DA trie prefix result {:?} doesn't start with prefix {:?}", result, prefix);
            }

            for result in &nested_prefix_results {
                prop_assert!(result.starts_with(prefix),
                    "Nested trie prefix result {:?} doesn't start with prefix {:?}", result, prefix);
            }

            // Convert to sets for comparison (order may differ)
            let da_set: HashSet<_> = da_prefix_results.into_iter().collect();
            let nested_set: HashSet<_> = nested_prefix_results.into_iter().collect();

            prop_assert_eq!(da_set, nested_set,
                "Prefix results differ for prefix {:?}", prefix);
        }
    }

    #[test]
    fn property_test_memory_efficiency_bounds(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..20), 10..100)
    ) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        let mut total_key_bytes = 0;
        let mut unique_keys = 0;

        for key in &keys {
            if da_trie.insert(key).is_ok() {
                total_key_bytes += key.len();
                unique_keys += 1;
            }
            let _ = nested_trie.insert(key);
        }

        if unique_keys > 0 {
            let da_stats = da_trie.stats();
            let nested_stats = nested_trie.stats();

            // Memory efficiency bounds
            let da_overhead_ratio = da_stats.memory_usage as f64 / total_key_bytes as f64;
            let nested_overhead_ratio = nested_stats.memory_usage as f64 / total_key_bytes as f64;

            prop_assert!(da_overhead_ratio < 1000.0,
                "DA trie overhead ratio {} too high for {} keys", da_overhead_ratio, unique_keys);
            prop_assert!(nested_overhead_ratio < 1000.0,
                "Nested trie overhead ratio {} too high for {} keys", nested_overhead_ratio, unique_keys);

            // Bits per key should be reasonable
            prop_assert!(da_stats.bits_per_key < 1_000_000.0,
                "DA trie bits per key {} too high", da_stats.bits_per_key);
            prop_assert!(nested_stats.bits_per_key < 1_000_000.0,
                "Nested trie bits per key {} too high", nested_stats.bits_per_key);
        }
    }

    #[test]
    fn property_test_state_transition_consistency(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..20), 1..30),
        test_symbols in prop::collection::vec(any::<u8>(), 0..10)
    ) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        for key in &keys {
            let _ = da_trie.insert(key);
            let _ = nested_trie.insert(key);
        }

        // Test state transitions from root
        let root = 0;

        for &symbol in &test_symbols {
            let da_transition = da_trie.transition(root, symbol);
            let nested_transition = nested_trie.transition(root, symbol);

            // If both have transitions, they should behave consistently
            if let (Some(da_state), Some(nested_state)) = (da_transition, nested_transition) {
                // Both should be valid states
                prop_assert!(da_state < u32::MAX / 2, "DA trie state too large: {}", da_state);
                prop_assert!(nested_state < u32::MAX / 2, "Nested trie state too large: {}", nested_state);

                // Test out_degree consistency
                let da_degree = da_trie.out_degree(da_state);
                let nested_degree = nested_trie.out_degree(nested_state);

                prop_assert!(da_degree <= 256, "DA trie out_degree too large: {}", da_degree);
                prop_assert!(nested_degree <= 256, "Nested trie out_degree too large: {}", nested_degree);
            }
        }
    }

    #[test]
    fn property_test_fsa_interface_consistency(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..30), 1..50)
    ) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        for key in &keys {
            let _ = da_trie.insert(key);
            let _ = nested_trie.insert(key);
        }

        // Test FSA accepts consistency
        for key in &keys {
            let da_accepts = da_trie.accepts(key);
            let nested_accepts = nested_trie.accepts(key);
            prop_assert_eq!(da_accepts, nested_accepts);
        }

        // Test root consistency
        prop_assert_eq!(da_trie.root(), nested_trie.root());

        // Test that both implement FSA interface correctly
        for key in &keys {
            if da_trie.contains(key) && nested_trie.contains(key) {
                // Both should accept via FSA interface
                prop_assert!(da_trie.accepts(key));
                prop_assert!(nested_trie.accepts(key));

                // Both should have consistent longest_prefix behavior
                let da_longest = da_trie.longest_prefix(key);
                let nested_longest = nested_trie.longest_prefix(key);
                prop_assert_eq!(da_longest, nested_longest);
                prop_assert_eq!(da_longest, Some(key.len()));
            }
        }
    }
}

// =============================================================================
// EDGE CASE AND PATHOLOGICAL TESTS
// =============================================================================

#[test]
fn test_pathological_key_patterns() {
    let pathological_keys = generate_pathological_test_keys();

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &pathological_keys {
        let da_result = da_trie.insert(key);
        let nested_result = nested_trie.insert(key);

        // Both should handle pathological cases gracefully
        assert_eq!(
            da_result.is_ok(),
            nested_result.is_ok(),
            "Inconsistent handling of pathological key: {:?}",
            key
        );

        if da_result.is_ok() {
            assert!(
                da_trie.contains(key),
                "Pathological key not found in DA trie: {:?}",
                key
            );
            assert!(
                nested_trie.contains(key),
                "Pathological key not found in nested trie: {:?}",
                key
            );
        }
    }
}

#[test]
fn test_shared_prefix_compression_efficiency() {
    let shared_prefix_keys = generate_shared_prefix_keys();

    // Test with high compression configuration
    let config = NestingConfigBuilder::new()
        .max_levels(6)
        .fragment_compression_ratio(0.8)
        .min_fragment_size(4)
        .build()
        .unwrap();

    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap();
    let mut da_trie = DoubleArrayTrie::new_compact();

    for key in &shared_prefix_keys {
        nested_trie.insert(key).unwrap();
        da_trie.insert(key).unwrap();
    }

    // Verify all keys are present
    for key in &shared_prefix_keys {
        assert!(
            nested_trie.contains(key),
            "Shared prefix key missing from nested trie: {:?}",
            key
        );
        assert!(
            da_trie.contains(key),
            "Shared prefix key missing from DA trie: {:?}",
            key
        );
    }

    // Check compression effectiveness
    let nested_stats = nested_trie.stats();
    let da_stats = da_trie.stats();

    println!("Shared prefix compression test:");
    println!("  Nested trie memory: {} bytes", nested_stats.memory_usage);
    println!("  DA trie memory: {} bytes", da_stats.memory_usage);
    println!("  Nested trie bits/key: {:.1}", nested_stats.bits_per_key);
    println!("  DA trie bits/key: {:.1}", da_stats.bits_per_key);

    // With shared prefixes, nested trie should potentially be more efficient
    // This is a heuristic check
    assert!(nested_stats.memory_usage > 0);
    assert!(da_stats.memory_usage > 0);
}

#[test]
fn test_extreme_key_lengths() {
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    // Test very long keys
    let very_long_key = vec![42u8; 10000];
    let da_result = da_trie.insert(&very_long_key);
    let nested_result = nested_trie.insert(&very_long_key);

    assert_eq!(da_result.is_ok(), nested_result.is_ok());

    if da_result.is_ok() {
        assert!(da_trie.contains(&very_long_key));
        assert!(nested_trie.contains(&very_long_key));

        // Memory usage should be reasonable even for very long keys
        let da_stats = da_trie.stats();
        let nested_stats = nested_trie.stats();

        assert!(da_stats.memory_usage < 100_000_000); // Less than 100MB
        assert!(nested_stats.memory_usage < 100_000_000);
    }

    // Test many short keys
    for i in 0..256 {
        let short_key = vec![i as u8];
        da_trie.insert(&short_key).unwrap();
        nested_trie.insert(&short_key).unwrap();
    }

    // Should handle many short keys efficiently
    assert_eq!(da_trie.len(), nested_trie.len());
    // With the very long key insertion, total count might exceed 256
    assert!(da_trie.len() > 0);
    assert!(da_trie.len() <= 257); // Due to deduplication + potential very long key
}

#[test]
fn test_rapid_insertion_patterns() {
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    // Test rapid insertion of sequential keys
    for i in 0..5000 {
        let key = format!("sequential_{:06}", i).into_bytes();
        da_trie.insert(&key).unwrap();
        nested_trie.insert(&key).unwrap();

        // Periodic consistency checks
        if i % 1000 == 0 {
            assert_eq!(da_trie.len(), nested_trie.len());

            // Verify some random earlier keys still exist
            for j in (0..i).step_by(500) {
                let check_key = format!("sequential_{:06}", j).into_bytes();
                assert!(
                    da_trie.contains(&check_key),
                    "Missing key at iteration {}: {:?}",
                    i,
                    check_key
                );
                assert!(
                    nested_trie.contains(&check_key),
                    "Missing key at iteration {}: {:?}",
                    i,
                    check_key
                );
            }
        }
    }

    assert_eq!(da_trie.len(), 5000);
    assert_eq!(nested_trie.len(), 5000);
}

// =============================================================================
// COMPREHENSIVE INTEGRATION SCENARIO TESTS
// =============================================================================

#[test]
fn test_comprehensive_integration_scenario() {
    // This test simulates a comprehensive real-world usage scenario
    // where multiple trie implementations are used together

    let dictionary_keys = generate_stress_test_keys(1000);
    let prefix_keys = generate_prefix_test_keys();
    let unicode_keys = generate_unicode_test_keys();

    // Create different trie configurations for different use cases
    let mut primary_trie = DoubleArrayTrie::new(); // Fast lookups
    let config = NestingConfigBuilder::new()
        .max_levels(4)
        .fragment_compression_ratio(0.3)
        .build()
        .unwrap();
    let mut compressed_trie =
        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap(); // Memory efficient

    // Populate primary dictionary
    for key in &dictionary_keys {
        primary_trie.insert(key).unwrap();
    }

    // Populate compressed trie with prefix-heavy data
    for key in &prefix_keys {
        compressed_trie.insert(key).unwrap();
    }
    for key in &unicode_keys {
        compressed_trie.insert(key).unwrap();
    }

    // Verify cross-compatibility

    // 1. Both should handle lookups correctly
    for key in &dictionary_keys {
        assert!(primary_trie.contains(key));
        assert!(primary_trie.lookup(key).is_some());
    }

    for key in &prefix_keys {
        assert!(compressed_trie.contains(key));
        assert!(compressed_trie.lookup(key).is_some());
    }

    // 2. Both should implement statistics correctly
    let primary_stats = primary_trie.stats();
    let compressed_stats = compressed_trie.stats();

    assert_eq!(primary_stats.num_keys, dictionary_keys.len());
    assert_eq!(
        compressed_stats.num_keys,
        prefix_keys.len() + unicode_keys.len()
    );

    assert!(primary_stats.memory_usage > 0);
    assert!(compressed_stats.memory_usage > 0);

    // 3. Both should support prefix operations
    let primary_prefix_results: Vec<_> = primary_trie.iter_prefix(b"stress_test").collect();
    assert_eq!(primary_prefix_results.len(), dictionary_keys.len()); // All start with "stress_test"

    let compressed_prefix_results: Vec<_> = compressed_trie.iter_prefix(b"prefix").collect();
    let expected_prefix_count = prefix_keys
        .iter()
        .filter(|k| k.starts_with(b"prefix"))
        .count();
    assert_eq!(compressed_prefix_results.len(), expected_prefix_count);

    // 4. Performance should be reasonable for both
    let start = Instant::now();
    for key in &dictionary_keys {
        assert!(primary_trie.contains(key));
    }
    let primary_lookup_time = start.elapsed();

    let combined_keys: Vec<_> = prefix_keys.iter().chain(unicode_keys.iter()).collect();
    let start = Instant::now();
    for key in &combined_keys {
        assert!(compressed_trie.contains(key));
    }
    let compressed_lookup_time = start.elapsed();

    println!("Integration scenario performance:");
    println!(
        "  Primary trie ({} keys): {:?}",
        dictionary_keys.len(),
        primary_lookup_time
    );
    println!(
        "  Compressed trie ({} keys): {:?}",
        combined_keys.len(),
        compressed_lookup_time
    );

    assert!(primary_lookup_time.as_secs() < 5);
    assert!(compressed_lookup_time.as_secs() < 5);

    // 5. Memory usage should be reasonable
    println!(
        "  Primary trie memory: {} bytes",
        primary_stats.memory_usage
    );
    println!(
        "  Compressed trie memory: {} bytes",
        compressed_stats.memory_usage
    );

    // Memory usage should be proportional to data size
    let primary_ratio = primary_stats.memory_usage as f64 / dictionary_keys.len() as f64;
    let compressed_ratio = compressed_stats.memory_usage as f64 / combined_keys.len() as f64;

    println!("  Primary bytes per key: {:.1}", primary_ratio);
    println!("  Compressed bytes per key: {:.1}", compressed_ratio);

    assert!(primary_ratio < 10000.0); // Less than 10KB per key
    assert!(compressed_ratio < 10000.0);
}

#[test]
fn test_rank_select_backend_integration() {
    // Test that RankSelectInterleaved256 backend works correctly with trie integration

    let keys = generate_common_test_keys();

    // Test with RankSelectInterleaved256 backend (best performer: 121-302 Mops/s)
    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        trie.insert(key).unwrap();
    }

    // Verify all operations work correctly
    assert_eq!(trie.len(), keys.len());

    for key in &keys {
        assert!(trie.contains(key));
        assert!(trie.lookup(key).is_some());
    }

    // Test prefix operations
    let all_keys: HashSet<_> = trie.iter_all().collect();
    assert_eq!(all_keys.len(), keys.len());

    // Verify statistics are valid
    let stats = trie.stats();
    assert_eq!(stats.num_keys, keys.len());
    assert!(stats.memory_usage > 0);
    assert!(stats.bits_per_key > 0.0);

    println!("RankSelectInterleaved256 backend performance:");
    println!("  Memory usage: {} bytes", stats.memory_usage);
    println!("  Bits per key: {:.2}", stats.bits_per_key);
    println!("  Total keys: {}", stats.num_keys);
}
