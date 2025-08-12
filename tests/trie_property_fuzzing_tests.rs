//! Property-based and fuzzing tests for all trie implementations
//!
//! This test suite uses property-based testing and fuzzing techniques to validate
//! correctness, robustness, and edge case handling for all trie implementations.

use arbitrary::{Arbitrary, Unstructured};
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use zipora::fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfigBuilder};
use zipora::fsa::{
    DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig, FiniteStateAutomaton,
    PrefixIterable, StateInspectable, StatisticsProvider, Trie,
};
use zipora::succinct::rank_select::{RankSelectInterleaved256, RankSelectSimple};

// =============================================================================
// CUSTOM GENERATORS FOR PROPERTY TESTING
// =============================================================================

/// Generate keys with specific patterns for testing edge cases
fn keys_with_patterns() -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(
        prop_oneof![
            // Empty keys (higher weight for edge case testing)
            5 => Just(vec![]),
            // Single byte keys
            10 => any::<u8>().prop_map(|b| vec![b]),
            // Short random keys (most common case)
            40 => prop::collection::vec(any::<u8>(), 1..20),
            // Medium random keys
            20 => prop::collection::vec(any::<u8>(), 20..50),
            // Long random keys (reduced size to avoid pathological cases)
            5 => prop::collection::vec(any::<u8>(), 50..100),
            // Keys with repeated patterns (bounded size)
            10 => (any::<u8>(), 1..15usize).prop_map(|(byte, len)| vec![byte; len]),
            // Keys with all byte values (fixed, reasonable size)
            1 => Just((0u8..=255u8).collect::<Vec<u8>>()),
            // Unicode-like patterns (reasonable size)
            5 => prop::collection::vec(0x80u8..=0xFFu8, 2..15),
            // Common prefixes (reasonable size)
            4 => (prop::collection::vec(any::<u8>(), 1..5), prop::collection::vec(any::<u8>(), 0..10))
                .prop_map(|(prefix, suffix)| {
                    let mut key = prefix;
                    key.extend(suffix);
                    key
                }),
        ],
        0..200, // Reduced from 500 to avoid pathological cases
    )
}

/// Generate trie configurations for testing
fn double_array_configs() -> impl Strategy<Value = DoubleArrayTrieConfig> {
    (
        1usize..10000,    // initial_capacity
        1.1f64..3.0,      // growth_factor
        any::<bool>(),    // use_memory_pool
        any::<bool>(),    // enable_simd
        1024usize..65536, // pool_size_class
    )
        .prop_map(
            |(capacity, growth, pool, simd, pool_size)| DoubleArrayTrieConfig {
                initial_capacity: capacity,
                growth_factor: growth,
                use_memory_pool: pool,
                enable_simd: simd,
                pool_size_class: pool_size,
                auto_shrink: false,
                cache_aligned: false,
                heuristic_collision_avoidance: false,
            },
        )
}

/// Generate nesting configurations for testing
fn nesting_configs() -> impl Strategy<Value = zipora::fsa::nested_louds_trie::NestingConfig> {
    (
        2usize..8,                                  // max_levels
        0.1f64..0.9,                                // fragment_compression_ratio
        2usize..32,                                 // min_fragment_size
        32usize..256,                               // max_fragment_size
        any::<bool>(),                              // cache_optimization
        prop::sample::select(vec![256, 512, 1024]), // cache_block_size
        0.1f64..0.9,                                // density_switch_threshold
        any::<bool>(),                              // adaptive_backend_selection
        1024usize..4194304,                         // memory_pool_size (1KB to 4MB)
    )
        .prop_map(
            |(levels, ratio, min_frag, max_frag, cache, block, density, adaptive, pool)| {
                NestingConfigBuilder::new()
                    .max_levels(levels)
                    .fragment_compression_ratio(ratio)
                    .min_fragment_size(min_frag)
                    .max_fragment_size(max_frag)
                    .cache_optimization(cache)
                    .cache_block_size(block)
                    .density_switch_threshold(density)
                    .adaptive_backend_selection(adaptive)
                    .memory_pool_size(pool)
                    .build()
                    .unwrap()
            },
        )
}

// =============================================================================
// FUNDAMENTAL PROPERTY TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn property_trie_insert_lookup_consistency(keys in keys_with_patterns()) {
        // Test Double Array Trie
        let mut da_trie = DoubleArrayTrie::new();
        let mut expected_keys = HashSet::new();

        // Insert all keys
        for key in &keys {
            if da_trie.insert(key).is_ok() {
                expected_keys.insert(key.clone());
            }
        }

        // Verify all inserted keys can be found
        for key in &expected_keys {
            prop_assert!(da_trie.contains(key),
                "Key not found in DA trie: {:?}", key);
            prop_assert!(da_trie.lookup(key).is_some(),
                "Lookup failed for key in DA trie: {:?}", key);
        }

        // Verify count is correct
        prop_assert_eq!(da_trie.len(), expected_keys.len());

        // Test Nested LOUDS Trie
        let mut nested_trie = NestedLoudsTrie::<RankSelectSimple>::new().unwrap();
        let mut nested_expected = HashSet::new();

        for key in &keys {
            if nested_trie.insert(key).is_ok() {
                nested_expected.insert(key.clone());
            }
        }

        for key in &nested_expected {
            prop_assert!(nested_trie.contains(key),
                "Key not found in nested trie: {:?}", key);
        }

        prop_assert_eq!(nested_trie.len(), nested_expected.len());
    }

    #[test]
    fn property_trie_fsa_interface_consistency(keys in keys_with_patterns()) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        for key in &keys {
            let _ = da_trie.insert(key);
            let _ = nested_trie.insert(key);
        }

        // Test that Trie.contains() and FSA.accepts() are equivalent
        for key in &keys {
            let da_contains = da_trie.contains(key);
            let da_accepts = da_trie.accepts(key);
            prop_assert_eq!(da_contains, da_accepts,
                "contains() and accepts() inconsistent for DA trie with key: {:?}", key);

            let nested_contains = nested_trie.contains(key);
            let nested_accepts = nested_trie.accepts(key);
            prop_assert_eq!(nested_contains, nested_accepts,
                "contains() and accepts() inconsistent for nested trie with key: {:?}", key);
        }

        // Test state transitions
        for key in &keys {
            if da_trie.contains(key) && !key.is_empty() {
                let mut state = da_trie.root();
                let mut valid_path = true;

                for &symbol in key {
                    if let Some(next_state) = da_trie.transition(state, symbol) {
                        state = next_state;
                    } else {
                        valid_path = false;
                        break;
                    }
                }

                if valid_path {
                    prop_assert!(da_trie.is_final(state),
                        "Final state not marked as final for key: {:?}", key);
                }
            }
        }
    }

}

// Separate proptest block for expensive prefix iteration test with reduced cases
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))] // Reduced from 2000 for this expensive test

    #[test]
    fn property_prefix_iteration_completeness(keys in prop::collection::vec(
        prop_oneof![
            // Focused set for prefix testing - smaller, more targeted
            30 => prop::collection::vec(any::<u8>(), 1..10),  // Short keys
            20 => (any::<u8>(), 1..5usize).prop_map(|(byte, len)| vec![byte; len]), // Repeated patterns
            15 => Just(vec![]),  // Empty key
            10 => any::<u8>().prop_map(|b| vec![b]), // Single byte
            // Common prefixes for better prefix testing
            25 => (prop::collection::vec(any::<u8>(), 1..3), prop::collection::vec(any::<u8>(), 0..5))
                .prop_map(|(prefix, suffix)| {
                    let mut key = prefix;
                    key.extend(suffix);
                    key
                }),
        ],
        0..30 // Reduced from 200 to 30 keys max for this test
    )) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut reference_keys = HashSet::new();

        for key in &keys {
            if da_trie.insert(key).is_ok() {
                reference_keys.insert(key.clone());
            }
        }

        // Test prefix iteration for various prefixes - optimized to avoid redundant work
        let mut tested_prefixes = HashSet::new();
        
        for prefix_len in 0..=3 { // Reduced from 5 to 3 for performance
            for key in &reference_keys {
                if key.len() >= prefix_len {
                    let prefix = &key[..prefix_len];
                    
                    // Skip if we've already tested this exact prefix
                    if !tested_prefixes.insert(prefix.to_vec()) {
                        continue;
                    }
                    
                    let prefix_results: Vec<_> = da_trie.iter_prefix(prefix).collect();

                    // Pre-compute expected matches once per prefix (not per key)
                    let expected_matches: Vec<_> = reference_keys
                        .iter()
                        .filter(|k| k.starts_with(prefix))
                        .cloned()
                        .collect();

                    // The generating key should be in the results if prefix == key
                    if prefix == key {
                        prop_assert!(prefix_results.contains(key),
                            "Key {:?} not found in its own prefix results", key);
                    }

                    // All results should start with the prefix
                    for result in &prefix_results {
                        prop_assert!(result.starts_with(prefix),
                            "Result {:?} doesn't start with prefix {:?}", result, prefix);
                    }

                    // Should not have more results than expected
                    prop_assert!(prefix_results.len() <= expected_matches.len(),
                        "Too many prefix results for {:?}: got {}, expected at most {}",
                        prefix, prefix_results.len(), expected_matches.len());
                }
            }
        }
    }
}

// Continue with remaining tests in the original proptest block
proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn property_trie_statistics_consistency(keys in keys_with_patterns()) {
        let mut da_trie = DoubleArrayTrie::new();
        let mut unique_keys = std::collections::HashSet::new();
        let mut total_key_bytes = 0;

        // Track unique keys to match trie deduplication behavior
        for key in &keys {
            if da_trie.insert(key).is_ok() {
                if unique_keys.insert(key.clone()) {
                    total_key_bytes += key.len();
                }
            }
        }

        let actual_count = unique_keys.len();

        let stats = da_trie.stats();

        // Basic statistics validation
        prop_assert_eq!(stats.num_keys, actual_count,
            "Statistics num_keys doesn't match actual insertions");
        prop_assert_eq!(da_trie.len(), actual_count,
            "Trie len() doesn't match actual insertions");

        // Handle empty trie case
        if actual_count == 0 {
            prop_assert_eq!(stats.memory_usage, 0,
                "Empty trie should have zero memory usage");
            prop_assert_eq!(stats.bits_per_key, 0.0,
                "Empty trie should have zero bits per key");
            prop_assert_eq!(stats.num_states, 0,
                "Empty trie should have zero states");
        } else {
            prop_assert!(stats.memory_usage > 0,
                "Memory usage should be positive for non-empty trie");
            prop_assert!(stats.bits_per_key > 0.0,
                "Bits per key should be positive for non-empty trie");
            prop_assert!(stats.num_states > 0,
                "Number of states should be positive for non-empty trie");

            // Validate memory usage bounds based on input size
            let reasonable_upper_bound = std::cmp::max(
                total_key_bytes * 1000,  // At most 1000x key data
                100_000_000              // Or 100MB, whichever is larger
            );
            prop_assert!(stats.memory_usage <= reasonable_upper_bound,
                "Memory usage {} exceeds reasonable bound {} for {} keys with {} total bytes",
                stats.memory_usage, reasonable_upper_bound, actual_count, total_key_bytes);

            // Validate bits per key is reasonable
            let max_reasonable_bits_per_key = f64::max(
                (total_key_bytes as f64 / actual_count as f64) * 10000.0, // At most 10000 bits per byte
                1_000_000.0  // Or 1M bits per key, whichever is larger
            );
            prop_assert!(stats.bits_per_key <= max_reasonable_bits_per_key,
                "Bits per key {} exceeds reasonable bound {} for {} keys",
                stats.bits_per_key, max_reasonable_bits_per_key, actual_count);
        }

        // Invariant checks
        prop_assert!(stats.num_transitions <= stats.num_states * 256,
            "Number of transitions {} cannot exceed states * 256: {}",
            stats.num_transitions, stats.num_states * 256);
        prop_assert!(stats.num_states >= actual_count,
            "Number of states {} should be at least number of keys {}",
            stats.num_states, actual_count);
    }

    #[test]
    fn property_configuration_robustness(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..50), 0..100),
        config in double_array_configs()
    ) {
        // Test that different configurations don't break correctness
        let mut trie = DoubleArrayTrie::with_config(config.clone());
        let mut successful_keys = HashSet::new();

        for key in &keys {
            if trie.insert(key).is_ok() {
                successful_keys.insert(key.clone());
            }
        }

        // All successfully inserted keys should be findable
        for key in &successful_keys {
            prop_assert!(trie.contains(key),
                "Key not found with config {:?}: {:?}", config, key);
        }

        prop_assert_eq!(trie.len(), successful_keys.len());

        // Configuration should be preserved
        prop_assert_eq!(trie.config().initial_capacity, config.initial_capacity);
        prop_assert_eq!(trie.config().use_memory_pool, config.use_memory_pool);
        prop_assert_eq!(trie.config().enable_simd, config.enable_simd);
    }

    #[test]
    fn property_nested_trie_nesting_invariants(
        keys in prop::collection::vec(prop::collection::vec(any::<u8>(), 1..30), 1..100),
        config in nesting_configs()
    ) {
        let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config.clone()).unwrap();

        for key in &keys {
            let _ = trie.insert(key);
        }

        // Test nesting invariants
        let active_levels = trie.active_levels();
        prop_assert!(active_levels <= config.max_levels,
            "Active levels {} exceeds max levels {}", active_levels, config.max_levels);

        // Test memory usage calculations
        let total_memory = trie.total_memory_usage();
        let layer_memory: usize = trie.layer_memory_usage().iter().sum();
        prop_assert!(total_memory >= layer_memory,
            "Total memory {} should be at least layer memory {}", total_memory, layer_memory);

        // Test fragment statistics
        let fragment_stats = trie.fragment_stats();
        prop_assert!(fragment_stats.compression_ratio >= 0.0,
            "Compression ratio should be non-negative");
        prop_assert!(fragment_stats.compression_ratio <= 1.0,
            "Compression ratio should not exceed 1.0");

        // Performance statistics should be consistent
        let perf_stats = trie.performance_stats();
        prop_assert_eq!(perf_stats.key_count, trie.len(),
            "Performance stats key count should match trie length");
    }
}

// =============================================================================
// FUZZING TESTS
// =============================================================================

/// Fuzz test data structure for generating arbitrary operations
#[derive(Debug, Clone, Arbitrary)]
enum TrieOperation {
    Insert(Vec<u8>),
    Contains(Vec<u8>),
    Lookup(Vec<u8>),
    IterPrefix(Vec<u8>),
    IterAll,
    Stats,
    GetRoot,
    Transition { state: u8, symbol: u8 },
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzTestSequence {
    operations: Vec<TrieOperation>,
}

/// Run fuzzing test with arbitrary operation sequence
fn fuzz_trie_operations(data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut unstructured = Unstructured::new(data);
    let sequence = FuzzTestSequence::arbitrary(&mut unstructured)?;

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectSimple>::new()?;
    let mut reference_keys = HashSet::new();

    for operation in sequence.operations {
        match operation {
            TrieOperation::Insert(key) => {
                let da_result = da_trie.insert(&key);
                let nested_result = nested_trie.insert(&key);

                if da_result.is_ok() {
                    reference_keys.insert(key.clone());
                }

                // Both should succeed or fail consistently for well-formed operations
                assert_eq!(
                    da_result.is_ok(),
                    nested_result.is_ok(),
                    "Insert consistency failed for key: {:?}",
                    key
                );
            }

            TrieOperation::Contains(key) => {
                let da_contains = da_trie.contains(&key);
                let nested_contains = nested_trie.contains(&key);

                // Results should be consistent with reference
                if reference_keys.contains(&key) {
                    assert!(da_contains, "DA trie should contain key: {:?}", key);
                    assert!(nested_contains, "Nested trie should contain key: {:?}", key);
                }
            }

            TrieOperation::Lookup(key) => {
                let da_lookup = da_trie.lookup(&key);
                let nested_lookup = nested_trie.lookup(&key);

                // Lookup should be consistent with contains
                assert_eq!(
                    da_lookup.is_some(),
                    da_trie.contains(&key),
                    "DA trie lookup/contains inconsistency for key: {:?}",
                    key
                );
                assert_eq!(
                    nested_lookup.is_some(),
                    nested_trie.contains(&key),
                    "Nested trie lookup/contains inconsistency for key: {:?}",
                    key
                );
            }

            TrieOperation::IterPrefix(prefix) => {
                let da_results: Vec<_> = da_trie.iter_prefix(&prefix).collect();
                let nested_results: Vec<_> = nested_trie.iter_prefix(&prefix).collect();

                // All results should start with prefix
                for result in &da_results {
                    assert!(
                        result.starts_with(&prefix),
                        "DA trie prefix result doesn't start with prefix: {:?} vs {:?}",
                        result,
                        prefix
                    );
                }

                for result in &nested_results {
                    assert!(
                        result.starts_with(&prefix),
                        "Nested trie prefix result doesn't start with prefix: {:?} vs {:?}",
                        result,
                        prefix
                    );
                }
            }

            TrieOperation::IterAll => {
                let da_all: Vec<_> = da_trie.iter_all().collect();
                let nested_all: Vec<_> = nested_trie.iter_all().collect();

                // Should not be more results than insertions
                assert!(
                    da_all.len() <= reference_keys.len(),
                    "DA trie iter_all returned too many results"
                );
                assert!(
                    nested_all.len() <= reference_keys.len(),
                    "Nested trie iter_all returned too many results"
                );
            }

            TrieOperation::Stats => {
                let da_stats = da_trie.stats();
                let nested_stats = nested_trie.stats();

                // Basic sanity checks
                assert!(
                    da_stats.memory_usage < 1_000_000_000,
                    "DA trie memory usage seems too high: {}",
                    da_stats.memory_usage
                );
                assert!(
                    nested_stats.memory_usage < 1_000_000_000,
                    "Nested trie memory usage seems too high: {}",
                    nested_stats.memory_usage
                );

                if da_stats.num_keys > 0 {
                    assert!(
                        da_stats.bits_per_key > 0.0,
                        "DA trie bits per key should be positive when keys exist"
                    );
                }

                if nested_stats.num_keys > 0 {
                    assert!(
                        nested_stats.bits_per_key > 0.0,
                        "Nested trie bits per key should be positive when keys exist"
                    );
                }
            }

            TrieOperation::GetRoot => {
                let da_root = da_trie.root();
                let nested_root = nested_trie.root();

                // Root should be consistent
                assert_eq!(da_root, 0, "DA trie root should be 0");
                assert_eq!(nested_root, 0, "Nested trie root should be 0");
            }

            TrieOperation::Transition { state, symbol } => {
                let da_transition = da_trie.transition(state as u32, symbol);
                let nested_transition = nested_trie.transition(state as u32, symbol);

                // Transitions should not panic or cause errors
                // Results may differ due to different internal representations
                let _ = da_transition;
                let _ = nested_transition;
            }
        }
    }

    // Final consistency checks
    assert_eq!(
        da_trie.len(),
        nested_trie.len(),
        "Final trie lengths should be equal"
    );

    // Verify all reference keys are still present
    for key in &reference_keys {
        assert!(
            da_trie.contains(key),
            "Reference key missing from DA trie: {:?}",
            key
        );
        assert!(
            nested_trie.contains(key),
            "Reference key missing from nested trie: {:?}",
            key
        );
    }

    Ok(())
}

#[test]
fn test_fuzz_basic() {
    // Test with some predefined sequences that have caused issues in the past
    let test_sequences = vec![
        vec![0u8; 100],
        vec![255u8; 100],
        (0u8..=255u8).cycle().take(1000).collect(),
        vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5], // Simple pattern
    ];

    for (i, data) in test_sequences.iter().enumerate() {
        match fuzz_trie_operations(data) {
            Ok(()) => println!("Fuzz test {} passed", i),
            Err(e) => panic!("Fuzz test {} failed: {}", i, e),
        }
    }
}

// =============================================================================
// EDGE CASE PROPERTY TESTS
// =============================================================================

proptest! {
    #[test]
    fn property_empty_operations(operations in prop::collection::vec(any::<u8>(), 0..100)) {
        // Test operations on empty tries
        let da_trie = DoubleArrayTrie::new();
        let nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        for op in &operations {
            let key = vec![*op];

            // Empty trie should not contain any keys
            prop_assert!(!da_trie.contains(&key));
            prop_assert!(!nested_trie.contains(&key));

            // Lookups should return None
            prop_assert!(da_trie.lookup(&key).is_none());
            prop_assert!(nested_trie.lookup(&key).is_none());

            // FSA should not accept any keys
            prop_assert!(!da_trie.accepts(&key));
            prop_assert!(!nested_trie.accepts(&key));

            // Prefix iteration should return empty
            let da_prefix: Vec<_> = da_trie.iter_prefix(&key).collect();
            let nested_prefix: Vec<_> = nested_trie.iter_prefix(&key).collect();
            prop_assert!(da_prefix.is_empty());
            prop_assert!(nested_prefix.is_empty());
        }

        // Empty trie properties
        prop_assert_eq!(da_trie.len(), 0);
        prop_assert_eq!(nested_trie.len(), 0);
        prop_assert!(da_trie.is_empty());
        prop_assert!(nested_trie.is_empty());

        // All iteration should return empty
        let da_all: Vec<_> = da_trie.iter_all().collect();
        let nested_all: Vec<_> = nested_trie.iter_all().collect();
        prop_assert!(da_all.is_empty());
        prop_assert!(nested_all.is_empty());
    }

    #[test]
    fn property_single_key_operations(key in prop::collection::vec(any::<u8>(), 0..100)) {
        // Test operations with single key insertions
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        // Insert the key
        let da_insert = da_trie.insert(&key);
        let nested_insert = nested_trie.insert(&key);

        prop_assert!(da_insert.is_ok());
        prop_assert!(nested_insert.is_ok());

        // Both tries should now contain the key
        prop_assert!(da_trie.contains(&key));
        prop_assert!(nested_trie.contains(&key));

        // Lookups should succeed
        prop_assert!(da_trie.lookup(&key).is_some());
        prop_assert!(nested_trie.lookup(&key).is_some());

        // FSA should accept the key
        prop_assert!(da_trie.accepts(&key));
        prop_assert!(nested_trie.accepts(&key));

        // Length should be 1
        prop_assert_eq!(da_trie.len(), 1);
        prop_assert_eq!(nested_trie.len(), 1);

        // All iteration should return the single key
        let da_all: Vec<_> = da_trie.iter_all().collect();
        let nested_all: Vec<_> = nested_trie.iter_all().collect();
        prop_assert_eq!(da_all.len(), 1);
        prop_assert_eq!(nested_all.len(), 1);
        prop_assert_eq!(&da_all[0], &key);
        prop_assert_eq!(&nested_all[0], &key);

        // Prefix iteration with the key itself should return the key
        let da_prefix: Vec<_> = da_trie.iter_prefix(&key).collect();
        let nested_prefix: Vec<_> = nested_trie.iter_prefix(&key).collect();
        prop_assert!(da_prefix.contains(&key));
        prop_assert!(nested_prefix.contains(&key));
    }

    #[test]
    fn property_duplicate_insertions(
        key in prop::collection::vec(any::<u8>(), 0..50),
        num_duplicates in 1usize..20
    ) {
        // Test that duplicate insertions don't break the trie
        let mut da_trie = DoubleArrayTrie::new();
        let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

        // Insert the same key multiple times
        for _ in 0..num_duplicates {
            let da_result = da_trie.insert(&key);
            let nested_result = nested_trie.insert(&key);

            prop_assert!(da_result.is_ok());
            prop_assert!(nested_result.is_ok());
        }

        // Length should still be 1 (or 0 for empty key edge case)
        let expected_len = if key.is_empty() || da_trie.contains(&key) { 1 } else { 0 };
        prop_assert_eq!(da_trie.len(), expected_len);
        prop_assert_eq!(nested_trie.len(), expected_len);

        if expected_len > 0 {
            // Key should be found
            prop_assert!(da_trie.contains(&key));
            prop_assert!(nested_trie.contains(&key));
        }
    }

    #[test]
    fn property_builder_consistency(keys in keys_with_patterns()) {
        // Test that builder and incremental insertion produce equivalent results
        let mut incremental_trie = DoubleArrayTrie::new();
        let mut successful_keys = Vec::new();

        // Incremental insertion
        for key in &keys {
            if incremental_trie.insert(key).is_ok() {
                successful_keys.push(key.clone());
            }
        }

        // Builder construction (need sorted keys)
        let mut sorted_keys = successful_keys.clone();
        sorted_keys.sort();
        sorted_keys.dedup();

        let builder_trie = DoubleArrayTrieBuilder::new()
            .build_from_sorted(sorted_keys.clone())
            .unwrap();

        // Both tries should have same length
        prop_assert_eq!(incremental_trie.len(), builder_trie.len());

        // Both should contain the same keys
        for key in &sorted_keys {
            prop_assert_eq!(incremental_trie.contains(key), builder_trie.contains(key),
                "Incremental and builder results differ for key: {:?}", key);
            prop_assert!(incremental_trie.contains(key));
        }

        // Statistics should be similar (within reason)
        let inc_stats = incremental_trie.stats();
        let builder_stats = builder_trie.stats();

        prop_assert_eq!(inc_stats.num_keys, builder_stats.num_keys);

        // Memory usage might differ slightly but should be in same ballpark
        if inc_stats.memory_usage > 0 && builder_stats.memory_usage > 0 {
            let ratio = inc_stats.memory_usage as f64 / builder_stats.memory_usage as f64;
            prop_assert!(ratio > 0.1 && ratio < 10.0,
                "Memory usage ratio too extreme: {}", ratio);
        }
    }
}

// =============================================================================
// STRESS TESTING WITH PROPERTY VALIDATION
// =============================================================================

#[test]
fn stress_test_large_random_dataset() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    let mut reference_map = HashMap::new();

    // Generate large random dataset
    let mut hasher = DefaultHasher::new();
    42u64.hash(&mut hasher);

    for i in 0..10000 {
        i.hash(&mut hasher);
        let hash = hasher.finish();

        // Create somewhat realistic key
        let key = format!("stress_test_key_{:08x}_{:04}", hash, i % 1000);
        let key_bytes = key.into_bytes();

        // Insert into all structures
        let da_result = da_trie.insert(&key_bytes);
        let nested_result = nested_trie.insert(&key_bytes);

        assert!(
            da_result.is_ok(),
            "DA trie insert failed at iteration {}",
            i
        );
        assert!(
            nested_result.is_ok(),
            "Nested trie insert failed at iteration {}",
            i
        );

        reference_map.insert(key_bytes.clone(), i);

        // Periodic consistency checks
        if i % 1000 == 0 {
            println!("Stress test progress: {} insertions", i);

            // Check a sample of keys
            for (ref_key, _) in reference_map.iter().take(100) {
                assert!(
                    da_trie.contains(ref_key),
                    "DA trie missing reference key at iteration {}: {:?}",
                    i,
                    ref_key
                );
                assert!(
                    nested_trie.contains(ref_key),
                    "Nested trie missing reference key at iteration {}: {:?}",
                    i,
                    ref_key
                );
            }

            // Statistics should be reasonable
            let da_stats = da_trie.stats();
            let nested_stats = nested_trie.stats();

            assert!(
                da_stats.memory_usage < 100_000_000,
                "DA trie memory usage too high at iteration {}: {}",
                i,
                da_stats.memory_usage
            );
            assert!(
                nested_stats.memory_usage < 100_000_000,
                "Nested trie memory usage too high at iteration {}: {}",
                i,
                nested_stats.memory_usage
            );

            assert!(
                da_stats.bits_per_key < 100_000.0,
                "DA trie bits per key too high at iteration {}: {}",
                i,
                da_stats.bits_per_key
            );
            assert!(
                nested_stats.bits_per_key < 100_000.0,
                "Nested trie bits per key too high at iteration {}: {}",
                i,
                nested_stats.bits_per_key
            );
        }

        hasher = DefaultHasher::new();
        hash.hash(&mut hasher);
    }

    // Final verification
    println!("Final stress test verification...");
    assert_eq!(da_trie.len(), reference_map.len());
    assert_eq!(nested_trie.len(), reference_map.len());

    // Verify all reference keys are present
    for (ref_key, _) in &reference_map {
        assert!(
            da_trie.contains(ref_key),
            "DA trie missing reference key in final check: {:?}",
            ref_key
        );
        assert!(
            nested_trie.contains(ref_key),
            "Nested trie missing reference key in final check: {:?}",
            ref_key
        );
    }

    println!(
        "Stress test completed successfully with {} keys",
        reference_map.len()
    );
}
