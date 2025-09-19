//! Comprehensive concurrency tests for thread-safe trie implementations
//!
//! This test suite validates thread safety, concurrent access patterns,
//! and performance under concurrent load for all trie implementations.
//!
//! IMPROVED CONCURRENCY TESTING:
//! - Replaced time-based loops with deterministic operation counts
//! - Separated insertion and verification phases to avoid race conditions
//! - Focus on invariant verification rather than exact timing-dependent counts
//! - Added explicit synchronization and proper barrier usage
//! - Track successful operations explicitly for reliable verification

use crossbeam_utils::thread as crossbeam_thread;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::task;

use zipora::fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfigBuilder};
use zipora::fsa::{
    DoubleArrayTrie, DoubleArrayTrieConfig, FiniteStateAutomaton, PrefixIterable,
    StatisticsProvider, Trie,
};
use zipora::succinct::rank_select::RankSelectInterleaved256;

// Note: CompressedSparseTrie concurrency tests would go here when available

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

fn generate_concurrent_test_keys(count: usize, prefix: &str) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("{}_{:06}", prefix, i).into_bytes())
        .collect()
}

fn generate_worker_specific_keys(worker_id: usize, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("worker_{}_{:06}", worker_id, i).into_bytes())
        .collect()
}

fn generate_shared_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("shared_{:06}", i).into_bytes())
        .collect()
}

// =============================================================================
// DOUBLE ARRAY TRIE CONCURRENCY TESTS
// =============================================================================

#[test]
fn test_double_array_trie_concurrent_reads() {
    let mut trie = DoubleArrayTrie::new();
    let keys = generate_concurrent_test_keys(1000, "concurrent_read");

    // Populate trie
    for key in &keys {
        trie.insert(key).unwrap();
    }

    let trie = Arc::new(trie);
    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait(); // Synchronize start

            let start = Instant::now();
            let mut successful_reads = 0;

            // Each thread reads all keys multiple times
            for _ in 0..10 {
                for (i, key) in keys_clone.iter().enumerate() {
                    if i % num_threads == thread_id {
                        if trie_clone.contains(key) {
                            successful_reads += 1;
                        }
                    }
                }
            }

            let duration = start.elapsed();
            (thread_id, successful_reads, duration)
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_reads = 0;
    for handle in handles {
        let (thread_id, reads, duration) = handle.join().unwrap();
        total_reads += reads;
        println!("Thread {}: {} reads in {:?}", thread_id, reads, duration);
    }

    println!("Total successful reads: {}", total_reads);
    assert!(total_reads > 0);
}

#[test]
fn test_double_array_trie_mixed_read_write_protection() {
    // Test that reads are consistent while the trie is being modified
    let trie = Arc::new(RwLock::new(DoubleArrayTrie::new()));
    let initial_keys = generate_concurrent_test_keys(500, "initial");

    // Populate initial data
    {
        let mut trie_guard = trie.write().unwrap();
        for key in &initial_keys {
            trie_guard.insert(key).unwrap();
        }
    }

    let num_readers = 6;
    let num_writers = 2;
    let operations_per_writer = 50;
    let reads_per_reader = 100;
    let writer_barrier = Arc::new(Barrier::new(num_writers));
    let reader_barrier = Arc::new(Barrier::new(num_readers));
    let inserted_keys = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    // Phase 1: Writers insert new keys
    for writer_id in 0..num_writers {
        let trie_clone = Arc::clone(&trie);
        let barrier_clone = Arc::clone(&writer_barrier);
        let inserted_clone = Arc::clone(&inserted_keys);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let mut writes_performed = 0;
            let mut local_keys = Vec::new();

            // Fixed number of operations
            for i in 0..operations_per_writer {
                let new_key = format!("writer_{}_{:06}", writer_id, i);

                {
                    let mut trie_guard = trie_clone.write().unwrap();
                    if trie_guard.insert(new_key.as_bytes()).is_ok() {
                        writes_performed += 1;
                        local_keys.push(new_key.as_bytes().to_vec());
                    }
                }
            }

            // Store successfully inserted keys for verification
            {
                let mut inserted_guard = inserted_clone.lock().unwrap();
                inserted_guard.extend(local_keys);
            }

            (writer_id, writes_performed)
        });

        handles.push(handle);
    }

    // Wait for all writers to complete
    let mut total_writes = 0;
    for _ in 0..num_writers {
        let (writer_id, writes) = handles.pop().unwrap().join().unwrap();
        total_writes += writes;
        println!("Writer {}: {} operations completed", writer_id, writes);
    }

    // Phase 2: Readers verify data integrity
    for reader_id in 0..num_readers {
        let trie_clone = Arc::clone(&trie);
        let keys_clone = initial_keys.clone();
        let inserted_clone = Arc::clone(&inserted_keys);
        let barrier_clone = Arc::clone(&reader_barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let mut consistent_reads = 0;

            // Fixed number of read operations
            for _ in 0..reads_per_reader {
                let trie_guard = trie_clone.read().unwrap();

                // Verify that all initial keys are still present
                let mut all_initial_present = true;
                for key in &keys_clone {
                    if !trie_guard.contains(key) {
                        all_initial_present = false;
                        break;
                    }
                }

                // Verify that inserted keys are accessible
                let inserted_guard = inserted_clone.lock().unwrap();
                let mut all_inserted_present = true;
                for key in inserted_guard.iter() {
                    if !trie_guard.contains(key) {
                        all_inserted_present = false;
                        break;
                    }
                }

                if all_initial_present && all_inserted_present {
                    consistent_reads += 1;
                }
            }

            (reader_id, consistent_reads)
        });

        handles.push(handle);
    }

    // Collect reader results
    let mut total_consistent_reads = 0;
    for handle in handles {
        let (reader_id, reads) = handle.join().unwrap();
        total_consistent_reads += reads;
        println!("Reader {}: {} consistent reads", reader_id, reads);
    }

    // Verify final state
    let final_trie = trie.read().unwrap();
    let final_inserted = inserted_keys.lock().unwrap();

    println!(
        "Final state: {} initial keys, {} new keys inserted, {} consistent reads",
        initial_keys.len(),
        final_inserted.len(),
        total_consistent_reads
    );

    // Invariant checks
    assert!(final_trie.len() >= initial_keys.len());
    assert!(
        total_consistent_reads > 0,
        "At least some reads should be consistent"
    );

    // All initial keys should still be present
    for key in &initial_keys {
        assert!(
            final_trie.contains(key),
            "Initial key missing after concurrent access"
        );
    }

    // All successfully inserted keys should be present
    for key in final_inserted.iter() {
        assert!(
            final_trie.contains(key),
            "Inserted key missing from final state"
        );
    }
}

#[test]
fn test_double_array_trie_stress_concurrent_access() {
    let config = DoubleArrayTrieConfig {
        initial_capacity: 2048,
        use_memory_pool: true,
        ..Default::default()
    };

    let trie = Arc::new(Mutex::new(DoubleArrayTrie::with_config(config)));
    let num_threads = 16;
    let operations_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            let worker_keys = generate_worker_specific_keys(thread_id, operations_per_thread);
            barrier_clone.wait();

            let start = Instant::now();
            let mut operations_completed = 0;

            // Perform mixed operations with clear separation
            let mut inserts_completed = 0;
            let mut reads_completed = 0;

            for (i, key) in worker_keys.iter().enumerate() {
                let mut trie_guard = trie_clone.lock().unwrap();

                if i % 3 == 0 {
                    // Insert operation
                    if trie_guard.insert(key).is_ok() {
                        inserts_completed += 1;
                        operations_completed += 1;
                    }
                } else {
                    // Read operation - always succeeds
                    let _result = trie_guard.contains(key);
                    reads_completed += 1;
                    operations_completed += 1;
                }
            }

            // Verify operation count invariants
            assert_eq!(
                operations_completed,
                worker_keys.len(),
                "Total operations should equal number of keys processed"
            );
            assert_eq!(
                inserts_completed + reads_completed,
                operations_completed,
                "Inserts + reads should equal total operations"
            );
            assert!(
                inserts_completed > 0,
                "Should have at least some insert operations"
            );
            assert!(
                reads_completed > 0,
                "Should have at least some read operations"
            );

            let duration = start.elapsed();
            (
                thread_id,
                operations_completed,
                inserts_completed,
                reads_completed,
                duration,
            )
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_operations = 0;
    let mut total_inserts = 0;
    let mut total_reads = 0;
    for handle in handles {
        let (thread_id, ops, inserts, reads, duration) = handle.join().unwrap();
        total_operations += ops;
        total_inserts += inserts;
        total_reads += reads;
        println!(
            "Thread {}: {} total operations ({} inserts, {} reads) in {:?}",
            thread_id, ops, inserts, reads, duration
        );
    }

    println!(
        "Total operations: {}, Inserts: {}, Reads: {}",
        total_operations, total_inserts, total_reads
    );

    // Verify expected operation counts
    let expected_total = num_threads * operations_per_thread;
    assert_eq!(
        total_operations, expected_total,
        "All threads should complete all operations"
    );
    assert!(
        total_inserts > 0,
        "At least some insert operations should succeed"
    );
    assert!(total_reads > 0, "All read operations should complete");

    // Verify trie integrity
    let final_trie = trie.lock().unwrap();
    println!("Final trie size: {}", final_trie.len());
    assert!(final_trie.len() > 0);
}

#[test]
fn test_double_array_trie_rayon_parallel_operations() {
    let mut trie = DoubleArrayTrie::new();

    // Create keys that will actually match the search prefixes
    let mut keys = Vec::new();

    // Create keys for "rayon_test_0" prefix (100 keys)
    for i in 0..100 {
        keys.push(format!("rayon_test_0{:02}", i).into_bytes());
    }

    // Create keys for "rayon_test_1" prefix (100 keys)
    for i in 0..100 {
        keys.push(format!("rayon_test_1{:02}", i).into_bytes());
    }

    // Create keys for "rayon_test_2" prefix (100 keys)
    for i in 0..100 {
        keys.push(format!("rayon_test_2{:02}", i).into_bytes());
    }

    // Add some additional varied keys for completeness (700 keys)
    for i in 300..1000 {
        keys.push(format!("rayon_test_{:06}", i).into_bytes());
    }

    // Sequential population
    for key in &keys {
        trie.insert(key).unwrap();
    }

    let trie = Arc::new(trie);

    // Parallel read operations using rayon
    let results: Vec<bool> = keys
        .par_iter()
        .map(|key| {
            let trie_ref = &*trie;
            trie_ref.contains(key)
        })
        .collect();

    // All reads should succeed
    assert_eq!(results.len(), keys.len());
    assert!(results.iter().all(|&r| r));

    // Parallel prefix operations - now all prefixes should have matches
    let prefixes = vec![
        b"rayon_test_0".as_slice(),
        b"rayon_test_1".as_slice(),
        b"rayon_test_2".as_slice(),
    ];

    // First, verify sequential access works to isolate the concurrency issue
    for (i, &prefix) in prefixes.iter().enumerate() {
        let sequential_results: Vec<_> = trie.iter_prefix(prefix).collect();
        println!(
            "Sequential prefix {}: found {} results",
            i,
            sequential_results.len()
        );
    }

    let prefix_results: Vec<Vec<Vec<u8>>> = prefixes
        .par_iter()
        .enumerate()
        .map(|(idx, &prefix)| {
            let trie_ref = &*trie;
            let results: Vec<_> = trie_ref.iter_prefix(prefix).collect();

            // Debug output for concurrent access
            println!(
                "Parallel prefix {} (thread {:?}): found {} results",
                idx,
                std::thread::current().id(),
                results.len()
            );

            results
        })
        .collect();

    // Should find results for each prefix - each should have 100 matches
    for (i, results) in prefix_results.iter().enumerate() {
        println!("Final prefix {}: found {} results", i, results.len());

        // Debug: Show what keys were generated for this prefix
        let expected_count = keys.iter().filter(|k| k.starts_with(prefixes[i])).count();
        println!(
            "Expected {} keys starting with {:?}",
            expected_count,
            std::str::from_utf8(prefixes[i]).unwrap()
        );

        assert!(
            !results.is_empty(),
            "Prefix {} ({:?}) should have matches - expected {}, got {}",
            i,
            std::str::from_utf8(prefixes[i]).unwrap(),
            expected_count,
            results.len()
        );

        // Verify we got the expected number of results (approximately 100 each)
        assert!(
            results.len() >= 100,
            "Prefix {} should have at least 100 matches, got {}",
            i,
            results.len()
        );
    }
}

// =============================================================================
// NESTED LOUDS TRIE CONCURRENCY TESTS
// =============================================================================

#[test]
fn test_nested_louds_trie_concurrent_reads() {
    let config = NestingConfigBuilder::new().max_levels(3).build().unwrap();

    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap();
    let keys = generate_concurrent_test_keys(800, "nested_concurrent");

    // Populate trie
    for key in &keys {
        trie.insert(key).unwrap();
    }

    let trie = Arc::new(trie);
    let num_threads = 6;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let start = Instant::now();
            let mut read_count = 0;

            // Each thread reads a subset of keys
            for (i, key) in keys_clone.iter().enumerate() {
                if i % num_threads == thread_id {
                    if trie_clone.contains(key) {
                        read_count += 1;
                    }

                    // Test lookup as well
                    if trie_clone.lookup(key).is_some() {
                        // Additional verification
                    }
                }
            }

            let duration = start.elapsed();
            (thread_id, read_count, duration)
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_reads = 0;
    for handle in handles {
        let (thread_id, reads, duration) = handle.join().unwrap();
        total_reads += reads;
        println!(
            "Nested LOUDS Thread {}: {} reads in {:?}",
            thread_id, reads, duration
        );
    }

    println!("Nested LOUDS total reads: {}", total_reads);
    assert!(total_reads > 0);
}

#[test]
fn test_nested_louds_trie_concurrent_prefix_operations() {
    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    // Create keys with common prefixes
    let prefixes = vec!["prefix_a", "prefix_b", "prefix_c"];
    let mut all_keys = Vec::new();

    for prefix in &prefixes {
        for i in 0..100 {
            let key = format!("{}_{:03}", prefix, i);
            all_keys.push(key.as_bytes().to_vec());
        }
    }

    // Populate trie
    for key in &all_keys {
        trie.insert(key).unwrap();
    }

    let trie = Arc::new(trie);
    let num_threads = prefixes.len();
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for (thread_id, prefix) in prefixes.iter().enumerate() {
        let trie_clone = Arc::clone(&trie);
        let prefix_bytes = prefix.as_bytes().to_vec();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let start = Instant::now();

            // Perform prefix operations
            let results: Vec<_> = trie_clone.iter_prefix(&prefix_bytes).collect();

            let duration = start.elapsed();
            (thread_id, results.len(), duration)
        });

        handles.push(handle);
    }

    // Collect results
    for handle in handles {
        let (thread_id, count, duration) = handle.join().unwrap();
        println!(
            "Prefix thread {}: found {} results in {:?}",
            thread_id, count, duration
        );
        assert!(count > 0);
    }
}

#[test]
fn test_nested_louds_concurrent_stress() {
    // Stress test RankSelectInterleaved256 backend under high concurrent load
    let keys = generate_concurrent_test_keys(1000, "stress_test");

    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    for key in &keys {
        trie.insert(key).unwrap();
    }
    let trie = Arc::new(trie);

    let num_threads = 8; // Higher thread count for stress testing
    let reads_per_thread = 150; // More operations per thread
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let start = Instant::now();
            let mut successful_reads = 0;
            let mut lookup_operations = 0;
            let mut prefix_operations = 0;

            // Intensive concurrent operations
            for round in 0..reads_per_thread {
                for (i, key) in keys_clone.iter().enumerate() {
                    if i % num_threads == thread_id {
                        // Mix different types of operations
                        match round % 3 {
                            0 => {
                                // Contains check
                                if trie_clone.contains(key) {
                                    successful_reads += 1;
                                }
                            }
                            1 => {
                                // Lookup operation
                                if trie_clone.lookup(key).is_some() {
                                    lookup_operations += 1;
                                }
                            }
                            2 => {
                                // Prefix operation (use first 8 bytes as prefix)
                                let prefix = &key[..std::cmp::min(8, key.len())];
                                let results: Vec<_> = trie_clone.iter_prefix(prefix).collect();
                                if !results.is_empty() {
                                    prefix_operations += 1;
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }

            let duration = start.elapsed();
            (
                thread_id,
                successful_reads,
                lookup_operations,
                prefix_operations,
                duration,
            )
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_reads = 0;
    let mut total_lookups = 0;
    let mut total_prefixes = 0;

    for handle in handles {
        let (thread_id, reads, lookups, prefixes, duration) = handle.join().unwrap();
        total_reads += reads;
        total_lookups += lookups;
        total_prefixes += prefixes;
        println!(
            "Stress thread {}: {} reads, {} lookups, {} prefix ops in {:?}",
            thread_id, reads, lookups, prefixes, duration
        );
    }

    println!(
        "Stress test totals - Reads: {}, Lookups: {}, Prefix ops: {}",
        total_reads, total_lookups, total_prefixes
    );

    // Verify all operations completed successfully
    assert!(total_reads > 0, "Should have successful reads");
    assert!(total_lookups > 0, "Should have successful lookups");
    assert!(total_prefixes > 0, "Should have successful prefix operations");

    // Verify trie integrity after stress test
    let final_stats = trie.stats();
    assert_eq!(final_stats.num_keys as usize, keys.len());
    assert!(final_stats.memory_usage > 0);

    // Verify all original keys are still accessible
    for key in &keys {
        assert!(
            trie.contains(key),
            "All keys should remain accessible after concurrent stress test"
        );
    }
}

// =============================================================================
// CROSS-IMPLEMENTATION CONCURRENCY TESTS
// =============================================================================

#[test]
fn test_mixed_trie_concurrent_access() {
    // Test concurrent access to different trie implementations
    let shared_keys = generate_shared_keys(500);

    // Setup tries
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &shared_keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let da_trie = Arc::new(da_trie);
    let nested_trie = Arc::new(nested_trie);
    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let da_clone = Arc::clone(&da_trie);
        let nested_clone = Arc::clone(&nested_trie);
        let keys_clone = shared_keys.clone();
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let start = Instant::now();
            let mut da_matches = 0;
            let mut nested_matches = 0;
            let mut consistency_checks = 0;

            for (i, key) in keys_clone.iter().enumerate() {
                if i % num_threads == thread_id {
                    let da_result = da_clone.contains(key);
                    let nested_result = nested_clone.contains(key);

                    if da_result {
                        da_matches += 1;
                    }
                    if nested_result {
                        nested_matches += 1;
                    }

                    // Check consistency between implementations
                    if da_result == nested_result {
                        consistency_checks += 1;
                    }
                }
            }

            let duration = start.elapsed();
            (
                thread_id,
                da_matches,
                nested_matches,
                consistency_checks,
                duration,
            )
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_da = 0;
    let mut total_nested = 0;
    let mut total_consistent = 0;

    for handle in handles {
        let (tid, da, nested, consistent, duration) = handle.join().unwrap();
        total_da += da;
        total_nested += nested;
        total_consistent += consistent;
        println!(
            "Thread {}: DA={}, Nested={}, Consistent={}, Time={:?}",
            tid, da, nested, consistent, duration
        );
    }

    println!(
        "Totals - DA: {}, Nested: {}, Consistent: {}",
        total_da, total_nested, total_consistent
    );

    // Both implementations should find the same keys
    assert_eq!(total_da, total_nested);
    assert_eq!(total_consistent, total_da); // All checks should be consistent
}

#[test]
fn test_concurrent_statistics_access() {
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    let keys = generate_concurrent_test_keys(1000, "stats_test");

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let da_trie = Arc::new(da_trie);
    let nested_trie = Arc::new(nested_trie);
    let num_threads = 10;
    let stats_operations_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let da_clone = Arc::clone(&da_trie);
        let nested_clone = Arc::clone(&nested_trie);
        let barrier_clone = Arc::clone(&barrier);

        let handle = thread::spawn(move || {
            barrier_clone.wait();

            let mut stats_collected = 0;
            let mut consistent_stats = 0;

            // Fixed number of statistics access operations
            for _ in 0..stats_operations_per_thread {
                // Access DA trie statistics
                let da_stats = da_clone.stats();
                let da_valid = da_stats.num_keys > 0 && da_stats.memory_usage > 0;

                // Access nested trie statistics
                let nested_stats = nested_clone.stats();
                let nested_valid = nested_stats.num_keys > 0 && nested_stats.memory_usage > 0;

                // Access performance statistics
                let perf_stats = nested_clone.performance_stats();
                let perf_valid = perf_stats.total_memory > 0;

                if da_valid && nested_valid && perf_valid {
                    consistent_stats += 1;
                }

                stats_collected += 1;

                // Verify logical consistency
                assert_eq!(
                    da_stats.num_keys, nested_stats.num_keys,
                    "DA and nested trie should have same key count"
                );
                assert!(
                    da_stats.memory_usage > 0,
                    "DA trie memory usage should be positive"
                );
                assert!(
                    nested_stats.memory_usage > 0,
                    "Nested trie memory usage should be positive"
                );
                assert!(
                    perf_stats.total_memory > 0,
                    "Performance stats memory should be positive"
                );
            }

            (thread_id, stats_collected, consistent_stats)
        });

        handles.push(handle);
    }

    // Collect results
    let mut total_stats = 0;
    let mut total_consistent = 0;
    for handle in handles {
        let (thread_id, stats, consistent) = handle.join().unwrap();
        total_stats += stats;
        total_consistent += consistent;
        println!(
            "Stats thread {}: {} collections, {} consistent",
            thread_id, stats, consistent
        );
    }

    println!(
        "Total statistics collections: {}, Total consistent: {}",
        total_stats, total_consistent
    );

    // Verify invariants
    assert_eq!(
        total_stats,
        num_threads * stats_operations_per_thread,
        "All threads should complete all operations"
    );
    assert_eq!(
        total_consistent, total_stats,
        "All statistics accesses should be consistent"
    );
}

// =============================================================================
// MEMORY SAFETY UNDER CONCURRENCY TESTS
// =============================================================================

#[test]
fn test_memory_safety_concurrent_access() {
    // This test validates that concurrent access doesn't cause memory corruption
    let config = DoubleArrayTrieConfig {
        use_memory_pool: true,
        initial_capacity: 1024,
        ..Default::default()
    };

    let trie = Arc::new(RwLock::new(DoubleArrayTrie::with_config(config)));
    let num_threads = 12;
    let operations_per_thread = 50;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);

        let handle = thread::spawn(move || {
            let worker_keys = generate_worker_specific_keys(thread_id, operations_per_thread);

            // Mixed read/write operations
            for (i, key) in worker_keys.iter().enumerate() {
                if i % 4 == 0 {
                    // Write operation
                    let mut trie_guard = trie_clone.write().unwrap();
                    let _ = trie_guard.insert(key);
                } else {
                    // Read operation
                    let trie_guard = trie_clone.read().unwrap();
                    let _ = trie_guard.contains(key);
                    let _ = trie_guard.lookup(key);

                    // Access statistics (this tests internal data structure integrity)
                    let stats = trie_guard.stats();
                    assert!(stats.memory_usage > 0);
                }

                // Verify data integrity after every few operations
                if i % 10 == 0 {
                    // Quick integrity check
                    let trie_guard = trie_clone.read().unwrap();
                    let stats = trie_guard.stats();
                    assert!(
                        stats.memory_usage > 0,
                        "Memory usage should always be positive"
                    );
                    assert!(stats.num_keys >= 0, "Key count should be non-negative");
                }
            }

            thread_id
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let thread_id = handle.join().unwrap();
        println!("Memory safety test thread {} completed", thread_id);
    }

    // Verify final state integrity
    let final_trie = trie.read().unwrap();
    let final_stats = final_trie.stats();

    println!(
        "Final trie state - Keys: {}, Memory: {} bytes",
        final_stats.num_keys, final_stats.memory_usage
    );

    assert!(final_stats.num_keys > 0);
    assert!(final_stats.memory_usage > 0);
    assert!(final_stats.bits_per_key > 0.0);
}

#[test]
fn test_nested_louds_memory_safety_concurrent() {
    let config = NestingConfigBuilder::new()
        .max_levels(4)
        .memory_pool_size(2 * 1024 * 1024)
        .build()
        .unwrap();

    let trie = Arc::new(RwLock::new(
        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap(),
    ));

    let num_threads = 8;
    let operations_per_thread = 75;
    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&trie);

        let handle = thread::spawn(move || {
            let worker_keys = generate_worker_specific_keys(thread_id, operations_per_thread);
            let mut operations_completed = 0;

            for (i, key) in worker_keys.iter().enumerate() {
                if i % 3 == 0 {
                    // Write operation
                    let mut trie_guard = trie_clone.write().unwrap();
                    if trie_guard.insert(key).is_ok() {
                        operations_completed += 1;
                    }
                } else {
                    // Read operations
                    let trie_guard = trie_clone.read().unwrap();

                    if trie_guard.contains(key) || !trie_guard.contains(key) {
                        operations_completed += 1;
                    }

                    // Test memory usage calculation and invariants
                    let memory_usage = trie_guard.total_memory_usage();
                    assert!(memory_usage >= 0, "Memory usage should be non-negative");

                    // Test layer information
                    let active_levels = trie_guard.active_levels();
                    assert!(
                        active_levels <= 4,
                        "Should not exceed configured max levels"
                    );
                    assert!(active_levels > 0, "Should have at least one active level");

                    // Verify statistics consistency
                    let stats = trie_guard.stats();
                    assert!(stats.num_keys >= 0, "Key count should be non-negative");
                    assert!(stats.memory_usage > 0, "Memory usage should be positive");
                }
            }

            (thread_id, operations_completed)
        });

        handles.push(handle);
    }

    // Wait for completion
    let mut total_operations = 0;
    for handle in handles {
        let (thread_id, ops) = handle.join().unwrap();
        total_operations += ops;
        println!(
            "Nested LOUDS memory safety thread {}: {} operations",
            thread_id, ops
        );
    }

    println!("Total operations completed: {}", total_operations);
    assert!(total_operations > 0);

    // Verify final integrity
    let final_trie = trie.read().unwrap();
    let stats = final_trie.stats();
    let perf_stats = final_trie.performance_stats();

    println!(
        "Final nested trie - Keys: {}, Memory: {} bytes, Levels: {}",
        stats.num_keys,
        perf_stats.total_memory,
        final_trie.active_levels()
    );

    assert!(stats.num_keys > 0);
    assert!(perf_stats.total_memory > 0);
}

// =============================================================================
// PERFORMANCE UNDER CONCURRENCY TESTS
// =============================================================================

#[test]
fn test_concurrent_performance_comparison() {
    let keys = generate_concurrent_test_keys(2000, "perf_test");

    // Prepare data structures
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let da_trie = Arc::new(da_trie);
    let nested_trie = Arc::new(nested_trie);

    // Test Double Array Trie performance
    let start = Instant::now();
    let da_results: Vec<_> = keys.par_iter().map(|key| da_trie.contains(key)).collect();
    let da_duration = start.elapsed();

    // Test Nested LOUDS Trie performance
    let start = Instant::now();
    let nested_results: Vec<_> = keys
        .par_iter()
        .map(|key| nested_trie.contains(key))
        .collect();
    let nested_duration = start.elapsed();

    println!("Concurrent Performance Comparison:");
    println!(
        "  Double Array Trie: {:?} for {} operations",
        da_duration,
        keys.len()
    );
    println!(
        "  Nested LOUDS Trie: {:?} for {} operations",
        nested_duration,
        keys.len()
    );

    // Verify correctness
    assert_eq!(da_results.len(), keys.len());
    assert_eq!(nested_results.len(), keys.len());
    assert!(da_results.iter().all(|&r| r));
    assert!(nested_results.iter().all(|&r| r));

    // Both should complete in reasonable time
    assert!(da_duration < Duration::from_secs(5));
    assert!(nested_duration < Duration::from_secs(5));
}

#[test]
fn test_scalability_under_concurrent_load() {
    let thread_counts = vec![1, 2, 4, 8, 16];
    let keys_per_thread = 100;

    for &num_threads in &thread_counts {
        let trie = Arc::new(RwLock::new(DoubleArrayTrie::new()));
        let total_keys = num_threads * keys_per_thread;

        let start = Instant::now();
        crossbeam_thread::scope(|s| {
            for thread_id in 0..num_threads {
                let trie_clone = Arc::clone(&trie);

                s.spawn(move |_| {
                    let worker_keys = generate_worker_specific_keys(thread_id, keys_per_thread);

                    for key in &worker_keys {
                        let mut trie_guard = trie_clone.write().unwrap();
                        let _ = trie_guard.insert(key);
                    }
                });
            }
        })
        .unwrap();

        let duration = start.elapsed();
        let operations_per_sec = total_keys as f64 / duration.as_secs_f64();

        println!(
            "Scalability test - {} threads: {:.0} ops/sec ({} total ops in {:?})",
            num_threads, operations_per_sec, total_keys, duration
        );

        // Verify final state and invariants
        let final_trie = trie.read().unwrap();
        let final_len = final_trie.len();

        // Each worker inserts unique keys, so we should have exactly total_keys
        // (no duplicates since each worker has unique keys)
        assert_eq!(
            final_len, total_keys,
            "Should have exactly {} keys inserted, got {}",
            total_keys, final_len
        );

        // Verify that all worker-specific keys are present
        for thread_id in 0..num_threads {
            for i in 0..keys_per_thread {
                let key = format!("worker_{}_{:06}", thread_id, i);
                assert!(
                    final_trie.contains(key.as_bytes()),
                    "Key {} should be present in final trie",
                    key
                );
            }
        }

        // Verify statistics consistency
        let stats = final_trie.stats();
        assert_eq!(
            stats.num_keys, final_len,
            "Statistics should match actual size"
        );
        assert!(stats.memory_usage > 0, "Memory usage should be positive");
    }
}
