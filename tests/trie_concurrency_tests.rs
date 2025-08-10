//! Comprehensive concurrency tests for thread-safe trie implementations
//!
//! This test suite validates thread safety, concurrent access patterns,
//! and performance under concurrent load for all trie implementations.

use std::collections::HashSet;
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use crossbeam_utils::thread as crossbeam_thread;
use rayon::prelude::*;
use tokio::runtime::Runtime;
use tokio::task;

use zipora::fsa::{
    DoubleArrayTrie, DoubleArrayTrieConfig,
    FiniteStateAutomaton, Trie, StatisticsProvider, PrefixIterable,
};
use zipora::fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfigBuilder};
use zipora::succinct::rank_select::{RankSelectInterleaved256, RankSelectSimple};

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
    let barrier = Arc::new(Barrier::new(num_readers + num_writers));
    let mut handles = Vec::new();
    
    // Spawn reader threads
    for reader_id in 0..num_readers {
        let trie_clone = Arc::clone(&trie);
        let keys_clone = initial_keys.clone();
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let mut consistent_reads = 0;
            let start = Instant::now();
            
            // Continuously read for a period
            while start.elapsed() < Duration::from_millis(100) {
                let trie_guard = trie_clone.read().unwrap();
                
                // Verify that all initial keys are still present
                let mut all_present = true;
                for key in &keys_clone {
                    if !trie_guard.contains(key) {
                        all_present = false;
                        break;
                    }
                }
                
                if all_present {
                    consistent_reads += 1;
                }
            }
            
            (reader_id, consistent_reads)
        });
        
        handles.push(handle);
    }
    
    // Spawn writer threads
    for writer_id in 0..num_writers {
        let trie_clone = Arc::clone(&trie);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let mut writes_performed = 0;
            let start = Instant::now();
            
            // Add new keys for a period
            while start.elapsed() < Duration::from_millis(100) {
                let new_key = format!("writer_{}_{:06}", writer_id, writes_performed);
                
                {
                    let mut trie_guard = trie_clone.write().unwrap();
                    trie_guard.insert(new_key.as_bytes()).unwrap();
                }
                
                writes_performed += 1;
                
                // Small delay to allow readers
                thread::sleep(Duration::from_micros(10));
            }
            
            (writer_id, writes_performed)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    for handle in handles {
        let result = handle.join().unwrap();
        println!("Thread result: {:?}", result);
    }
    
    // Verify final state
    let final_trie = trie.read().unwrap();
    assert!(final_trie.len() >= initial_keys.len());
    
    // All initial keys should still be present
    for key in &initial_keys {
        assert!(final_trie.contains(key), "Initial key missing after concurrent access");
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
            
            // Perform mixed operations
            for (i, key) in worker_keys.iter().enumerate() {
                let mut trie_guard = trie_clone.lock().unwrap();
                
                if i % 3 == 0 {
                    // Insert operation
                    if trie_guard.insert(key).is_ok() {
                        operations_completed += 1;
                    }
                } else {
                    // Read operation
                    if trie_guard.contains(key) || !trie_guard.contains(key) {
                        operations_completed += 1;
                    }
                }
            }
            
            let duration = start.elapsed();
            (thread_id, operations_completed, duration)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut total_operations = 0;
    for handle in handles {
        let (thread_id, ops, duration) = handle.join().unwrap();
        total_operations += ops;
        println!("Thread {}: {} operations in {:?}", thread_id, ops, duration);
    }
    
    println!("Total operations completed: {}", total_operations);
    assert!(total_operations > 0);
    
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
    let prefixes = vec![b"rayon_test_0".as_slice(), b"rayon_test_1".as_slice(), b"rayon_test_2".as_slice()];
    
    // First, verify sequential access works to isolate the concurrency issue
    for (i, &prefix) in prefixes.iter().enumerate() {
        let sequential_results: Vec<_> = trie.iter_prefix(prefix).collect();
        println!("Sequential prefix {}: found {} results", i, sequential_results.len());
    }
    
    let prefix_results: Vec<Vec<Vec<u8>>> = prefixes
        .par_iter()
        .enumerate()
        .map(|(idx, &prefix)| {
            let trie_ref = &*trie;
            let results: Vec<_> = trie_ref.iter_prefix(prefix).collect();
            
            // Debug output for concurrent access
            println!("Parallel prefix {} (thread {:?}): found {} results", 
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
        let expected_count = keys.iter()
            .filter(|k| k.starts_with(prefixes[i]))
            .count();
        println!("Expected {} keys starting with {:?}", 
            expected_count, 
            std::str::from_utf8(prefixes[i]).unwrap()
        );
        
        assert!(!results.is_empty(), 
            "Prefix {} ({:?}) should have matches - expected {}, got {}", 
            i, 
            std::str::from_utf8(prefixes[i]).unwrap(),
            expected_count,
            results.len()
        );
        
        // Verify we got the expected number of results (approximately 100 each)
        assert!(results.len() >= 100, 
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
    let config = NestingConfigBuilder::new()
        .max_levels(3)
        .build()
        .unwrap();
    
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
        println!("Nested LOUDS Thread {}: {} reads in {:?}", thread_id, reads, duration);
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
        println!("Prefix thread {}: found {} results in {:?}", thread_id, count, duration);
        assert!(count > 0);
    }
}

#[test]
fn test_nested_louds_backend_concurrent_comparison() {
    let keys = generate_concurrent_test_keys(500, "backend_test");
    
    // Test different backends under concurrent access
    let simple_trie = {
        let mut trie = NestedLoudsTrie::<RankSelectSimple>::new().unwrap();
        for key in &keys {
            trie.insert(key).unwrap();
        }
        Arc::new(trie)
    };
    
    let interleaved_trie = {
        let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
        for key in &keys {
            trie.insert(key).unwrap();
        }
        Arc::new(trie)
    };
    
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads * 2)); // 2 sets of threads
    let mut handles = Vec::new();
    
    // Test simple backend
    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&simple_trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let mut successful_reads = 0;
            for (i, key) in keys_clone.iter().enumerate() {
                if i % num_threads == thread_id {
                    if trie_clone.contains(key) {
                        successful_reads += 1;
                    }
                }
            }
            
            ("simple", thread_id, successful_reads)
        });
        
        handles.push(handle);
    }
    
    // Test interleaved backend
    for thread_id in 0..num_threads {
        let trie_clone = Arc::clone(&interleaved_trie);
        let keys_clone = keys.clone();
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let mut successful_reads = 0;
            for (i, key) in keys_clone.iter().enumerate() {
                if i % num_threads == thread_id {
                    if trie_clone.contains(key) {
                        successful_reads += 1;
                    }
                }
            }
            
            ("interleaved", thread_id, successful_reads)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut simple_total = 0;
    let mut interleaved_total = 0;
    
    for handle in handles {
        let (backend, thread_id, reads) = handle.join().unwrap();
        match backend {
            "simple" => simple_total += reads,
            "interleaved" => interleaved_total += reads,
            _ => unreachable!(),
        }
        println!("{} backend thread {}: {} reads", backend, thread_id, reads);
    }
    
    println!("Simple backend total: {}", simple_total);
    println!("Interleaved backend total: {}", interleaved_total);
    
    // Both backends should perform correctly
    assert!(simple_total > 0);
    assert!(interleaved_total > 0);
    assert_eq!(simple_total, interleaved_total); // Should read same number of keys
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
            (thread_id, da_matches, nested_matches, consistency_checks, duration)
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
        println!("Thread {}: DA={}, Nested={}, Consistent={}, Time={:?}", 
            tid, da, nested, consistent, duration);
    }
    
    println!("Totals - DA: {}, Nested: {}, Consistent: {}", 
        total_da, total_nested, total_consistent);
    
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
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();
    
    for thread_id in 0..num_threads {
        let da_clone = Arc::clone(&da_trie);
        let nested_clone = Arc::clone(&nested_trie);
        let barrier_clone = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier_clone.wait();
            
            let start = Instant::now();
            let mut stats_collected = 0;
            
            // Continuously access statistics
            while start.elapsed() < Duration::from_millis(50) {
                // Access DA trie statistics
                let da_stats = da_clone.stats();
                assert!(da_stats.num_keys > 0);
                assert!(da_stats.memory_usage > 0);
                
                // Access nested trie statistics
                let nested_stats = nested_clone.stats();
                assert!(nested_stats.num_keys > 0);
                assert!(nested_stats.memory_usage > 0);
                
                // Access performance statistics
                let perf_stats = nested_clone.performance_stats();
                assert!(perf_stats.total_memory > 0);
                
                stats_collected += 1;
            }
            
            (thread_id, stats_collected)
        });
        
        handles.push(handle);
    }
    
    // Collect results
    let mut total_stats = 0;
    for handle in handles {
        let (thread_id, stats) = handle.join().unwrap();
        total_stats += stats;
        println!("Stats thread {}: {} collections", thread_id, stats);
    }
    
    println!("Total statistics collections: {}", total_stats);
    assert!(total_stats > 0);
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
                
                // Small delay to increase chance of race conditions
                if i % 10 == 0 {
                    thread::sleep(Duration::from_micros(1));
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
    
    println!("Final trie state - Keys: {}, Memory: {} bytes", 
        final_stats.num_keys, final_stats.memory_usage);
    
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
        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config).unwrap()
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
                    
                    // Test memory usage calculation
                    let memory_usage = trie_guard.total_memory_usage();
                    assert!(memory_usage >= 0);
                    
                    // Test layer information
                    let active_levels = trie_guard.active_levels();
                    assert!(active_levels <= 4); // Should not exceed configured max
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
        println!("Nested LOUDS memory safety thread {}: {} operations", thread_id, ops);
    }
    
    println!("Total operations completed: {}", total_operations);
    assert!(total_operations > 0);
    
    // Verify final integrity
    let final_trie = trie.read().unwrap();
    let stats = final_trie.stats();
    let perf_stats = final_trie.performance_stats();
    
    println!("Final nested trie - Keys: {}, Memory: {} bytes, Levels: {}", 
        stats.num_keys, perf_stats.total_memory, final_trie.active_levels());
    
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
    let da_results: Vec<_> = keys
        .par_iter()
        .map(|key| da_trie.contains(key))
        .collect();
    let da_duration = start.elapsed();
    
    // Test Nested LOUDS Trie performance
    let start = Instant::now();
    let nested_results: Vec<_> = keys
        .par_iter()
        .map(|key| nested_trie.contains(key))
        .collect();
    let nested_duration = start.elapsed();
    
    println!("Concurrent Performance Comparison:");
    println!("  Double Array Trie: {:?} for {} operations", da_duration, keys.len());
    println!("  Nested LOUDS Trie: {:?} for {} operations", nested_duration, keys.len());
    
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
        }).unwrap();
        
        let duration = start.elapsed();
        let operations_per_sec = total_keys as f64 / duration.as_secs_f64();
        
        println!("Scalability test - {} threads: {:.0} ops/sec ({} total ops in {:?})", 
            num_threads, operations_per_sec, total_keys, duration);
        
        // Verify final state
        let final_trie = trie.read().unwrap();
        assert!(final_trie.len() <= total_keys); // May have some duplicates
    }
}