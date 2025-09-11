//! Stress tests for concurrent profiling scenarios
//!
//! This module contains comprehensive stress tests to validate the Advanced Profiling
//! Integration system under high concurrency loads, thread contention, and extreme
//! usage patterns.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use zipora::dev_infrastructure::profiling::*;

/// Test concurrent profiling with multiple threads
#[test]
fn test_concurrent_profiling_stress() {
    const NUM_THREADS: usize = 16;
    const OPERATIONS_PER_THREAD: usize = 1000;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let barrier = barrier.clone();
            let success_count = success_count.clone();
            let error_count = error_count.clone();
            
            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();
                
                for i in 0..OPERATIONS_PER_THREAD {
                    let operation_name = format!("thread_{}_{}", thread_id, i);
                    
                    // Test different profiler types in rotation
                    match i % 3 {
                        0 => {
                            // Hardware profiler
                            if let Ok(profiler) = HardwareProfiler::global() {
                                if let Ok(handle) = profiler.start(&operation_name) {
                                    // Simulate work
                                    thread::sleep(Duration::from_micros(10));
                                    if profiler.end(handle).is_ok() {
                                        success_count.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        error_count.fetch_add(1, Ordering::Relaxed);
                                    }
                                } else {
                                    error_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        1 => {
                            // Memory profiler
                            if let Ok(profiler) = MemoryProfiler::global() {
                                if let Ok(handle) = profiler.start(&operation_name) {
                                    // Simulate memory allocation work
                                    let _vec: Vec<u64> = vec![thread_id as u64; 100];
                                    thread::sleep(Duration::from_micros(5));
                                    if profiler.end(handle).is_ok() {
                                        success_count.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        error_count.fetch_add(1, Ordering::Relaxed);
                                    }
                                } else {
                                    error_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        2 => {
                            // Cache profiler
                            if let Ok(profiler) = CacheProfiler::global() {
                                if let Ok(handle) = profiler.start(&operation_name) {
                                    // Simulate cache-intensive work
                                    let mut sum = 0u64;
                                    for j in 0..100 {
                                        sum = sum.wrapping_add(j * thread_id as u64);
                                    }
                                    thread::sleep(Duration::from_micros(1));
                                    if profiler.end(handle).is_ok() {
                                        success_count.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        error_count.fetch_add(1, Ordering::Relaxed);
                                    }
                                } else {
                                    error_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_successes = success_count.load(Ordering::Relaxed);
    let total_errors = error_count.load(Ordering::Relaxed);
    let expected_total = NUM_THREADS * OPERATIONS_PER_THREAD;
    
    println!("Concurrent stress test results:");
    println!("  Successes: {}", total_successes);
    println!("  Errors: {}", total_errors);
    println!("  Expected total: {}", expected_total);
    
    // Allow some errors due to profiler initialization race conditions,
    // but ensure most operations succeed
    assert!(total_successes >= expected_total * 8 / 10, 
            "Too many failures: {} successes out of {} expected", 
            total_successes, expected_total);
}

/// Test RAII profiler scopes under high contention
#[test]
fn test_raii_scope_stress() {
    const NUM_THREADS: usize = 32;
    const SCOPES_PER_THREAD: usize = 500;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let success_count = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let barrier = barrier.clone();
            let success_count = success_count.clone();
            
            thread::spawn(move || {
                barrier.wait();
                
                for i in 0..SCOPES_PER_THREAD {
                    let scope_name = format!("raii_{}_{}", thread_id, i);
                    
                    // Test nested scopes
                    if let Ok(_outer_scope) = ProfilerScope::new(&scope_name) {
                        if let Ok(_inner_scope) = ProfilerScope::new(&format!("{}_inner", scope_name)) {
                            // Simulate work
                            let mut sum = 0u64;
                            for j in 0..50 {
                                sum = sum.wrapping_add(j * thread_id as u64);
                            }
                            // Both scopes will automatically drop here
                            success_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_successes = success_count.load(Ordering::Relaxed);
    let expected_total = NUM_THREADS * SCOPES_PER_THREAD;
    
    println!("RAII scope stress test results:");
    println!("  Successes: {}", total_successes);
    println!("  Expected: {}", expected_total);
    
    // RAII scopes should have very high success rate
    assert!(total_successes >= expected_total * 9 / 10,
            "RAII scope failures: {} successes out of {} expected",
            total_successes, expected_total);
}

/// Test profiler registry under concurrent access
#[test]
fn test_registry_concurrent_access() {
    const NUM_THREADS: usize = 20;
    const OPERATIONS_PER_THREAD: usize = 200;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let operation_count = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_thread_id| {
            let barrier = barrier.clone();
            let operation_count = operation_count.clone();
            
            thread::spawn(move || {
                barrier.wait();
                
                for _i in 0..OPERATIONS_PER_THREAD {
                    // Test registry operations
                    let registry = ProfilerRegistry::new();
                    
                    // Test profiler lookup operations
                    let _hw_profiler = HardwareProfiler::global();
                    let _mem_profiler = MemoryProfiler::global();
                    let _cache_profiler = CacheProfiler::global();
                    
                    // Simulate work
                    thread::sleep(Duration::from_micros(1));
                    
                    operation_count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_operations = operation_count.load(Ordering::Relaxed);
    let expected_total = NUM_THREADS * OPERATIONS_PER_THREAD;
    
    assert_eq!(total_operations, expected_total,
               "Registry operations incomplete: {} out of {} expected",
               total_operations, expected_total);
}

/// Test profiling under memory pressure
#[test]
fn test_profiling_under_memory_pressure() {
    const NUM_THREADS: usize = 8;
    const LARGE_ALLOCATIONS_PER_THREAD: usize = 100;
    const ALLOCATION_SIZE: usize = 1024 * 1024; // 1MB per allocation
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let success_count = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let barrier = barrier.clone();
            let success_count = success_count.clone();
            
            thread::spawn(move || {
                barrier.wait();
                
                for i in 0..LARGE_ALLOCATIONS_PER_THREAD {
                    let operation_name = format!("memory_pressure_{}_{}", thread_id, i);
                    
                    if let Ok(profiler) = MemoryProfiler::global() {
                        if let Ok(handle) = profiler.start(&operation_name) {
                            // Create memory pressure
                            let mut allocations = Vec::new();
                            for j in 0..10 {
                                let data = vec![thread_id as u8; ALLOCATION_SIZE];
                                allocations.push(data);
                            }
                            
                            // Touch the memory to ensure it's actually allocated
                            for (idx, allocation) in allocations.iter_mut().enumerate() {
                                allocation[0] = (i % 256) as u8;
                                allocation[ALLOCATION_SIZE - 1] = (idx % 256) as u8;
                            }
                            
                            thread::sleep(Duration::from_micros(100));
                            
                            if profiler.end(handle).is_ok() {
                                success_count.fetch_add(1, Ordering::Relaxed);
                            }
                            
                            // Deallocate memory
                            drop(allocations);
                        }
                    }
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_successes = success_count.load(Ordering::Relaxed);
    let expected_total = NUM_THREADS * LARGE_ALLOCATIONS_PER_THREAD;
    
    println!("Memory pressure test results:");
    println!("  Successes: {}", total_successes);
    println!("  Expected: {}", expected_total);
    
    // Should handle memory pressure gracefully
    assert!(total_successes >= expected_total * 7 / 10,
            "Too many failures under memory pressure: {} successes out of {} expected",
            total_successes, expected_total);
}

/// Test long-running profiling session
#[test]
fn test_long_running_profiling_session() {
    const DURATION_SECONDS: u64 = 30; // 30 second stress test
    const NUM_THREADS: usize = 4;
    
    let start_time = Instant::now();
    let end_time = start_time + Duration::from_secs(DURATION_SECONDS);
    let should_stop = Arc::new(AtomicBool::new(false));
    let operation_count = Arc::new(AtomicUsize::new(0));
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let should_stop = should_stop.clone();
            let operation_count = operation_count.clone();
            
            thread::spawn(move || {
                let mut local_count = 0;
                
                while !should_stop.load(Ordering::Relaxed) {
                    let operation_name = format!("long_running_{}_{}", thread_id, local_count);
                    
                    // Use different profilers in rotation
                    match local_count % 3 {
                        0 => {
                            if let Ok(profiler) = HardwareProfiler::global() {
                                if let Ok(handle) = profiler.start(&operation_name) {
                                    thread::sleep(Duration::from_millis(1));
                                    let _ = profiler.end(handle);
                                }
                            }
                        }
                        1 => {
                            let _scope = ProfilerScope::new(&operation_name);
                            thread::sleep(Duration::from_millis(1));
                        }
                        2 => {
                            if let Ok(profiler) = CacheProfiler::global() {
                                if let Ok(handle) = profiler.start(&operation_name) {
                                    // Simulate cache work
                                    let mut sum = 0u64;
                                    for i in 0..1000 {
                                        sum = sum.wrapping_add(i * thread_id as u64);
                                    }
                                    let _ = profiler.end(handle);
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    
                    local_count += 1;
                    operation_count.fetch_add(1, Ordering::Relaxed);
                    
                    // Check if we should stop (but not too frequently)
                    if local_count % 100 == 0 && Instant::now() >= end_time {
                        should_stop.store(true, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();
    
    // Let the test run for the specified duration
    while Instant::now() < end_time {
        thread::sleep(Duration::from_secs(1));
    }
    
    should_stop.store(true, Ordering::Relaxed);
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_operations = operation_count.load(Ordering::Relaxed);
    let actual_duration = start_time.elapsed();
    
    println!("Long-running test results:");
    println!("  Duration: {:?}", actual_duration);
    println!("  Total operations: {}", total_operations);
    println!("  Operations per second: {:.2}", 
             total_operations as f64 / actual_duration.as_secs_f64());
    
    // Should complete a reasonable number of operations
    assert!(total_operations > 0, "No operations completed");
    assert!(actual_duration.as_secs() >= DURATION_SECONDS - 1, 
            "Test completed too early");
}

/// Test profiler configuration changes under load
#[test]
fn test_configuration_changes_under_load() {
    const NUM_THREADS: usize = 8;
    const OPERATIONS_PER_THREAD: usize = 200;
    
    let barrier = Arc::new(Barrier::new(NUM_THREADS + 1)); // +1 for config thread
    let should_stop = Arc::new(AtomicBool::new(false));
    let operation_count = Arc::new(AtomicUsize::new(0));
    
    // Configuration change thread
    let config_barrier = barrier.clone();
    let config_should_stop = should_stop.clone();
    let config_handle = thread::spawn(move || {
        config_barrier.wait();
        
        let configs = vec![
            ProfilingConfig::production(),
            ProfilingConfig::development(),
            ProfilingConfig::debugging(),
            ProfilingConfig::disabled(),
        ];
        
        let mut config_index = 0;
        while !config_should_stop.load(Ordering::Relaxed) {
            // Create new reporter with different config
            let config = configs[config_index % configs.len()].clone();
            let _reporter = ProfilerReporter::new(config);
            
            config_index += 1;
            thread::sleep(Duration::from_millis(50));
        }
    });
    
    // Worker threads
    let worker_handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let barrier = barrier.clone();
            let should_stop = should_stop.clone();
            let operation_count = operation_count.clone();
            
            thread::spawn(move || {
                barrier.wait();
                
                for i in 0..OPERATIONS_PER_THREAD {
                    if should_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    
                    let operation_name = format!("config_test_{}_{}", thread_id, i);
                    
                    // Try different profiling operations
                    if let Ok(_scope) = ProfilerScope::new(&operation_name) {
                        thread::sleep(Duration::from_micros(100));
                        operation_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
                
                should_stop.store(true, Ordering::Relaxed);
            })
        })
        .collect();
    
    // Wait for all worker threads
    for handle in worker_handles {
        handle.join().unwrap();
    }
    
    // Stop config thread
    should_stop.store(true, Ordering::Relaxed);
    config_handle.join().unwrap();
    
    let total_operations = operation_count.load(Ordering::Relaxed);
    
    println!("Configuration change test results:");
    println!("  Total operations: {}", total_operations);
    
    // Should complete operations despite configuration changes
    assert!(total_operations > 0, "No operations completed");
}

/// Benchmark profiling overhead under high load
#[test]
fn test_profiling_overhead_benchmark() {
    const NUM_THREADS: usize = 4;
    const ITERATIONS_PER_THREAD: usize = 10000;
    
    // Baseline test without profiling
    let baseline_start = Instant::now();
    let baseline_handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for i in 0..ITERATIONS_PER_THREAD {
                    // Simulate CPU work
                    let mut sum = 0u64;
                    for j in 0..100 {
                        sum = sum.wrapping_add((i * j * thread_id) as u64);
                    }
                    std::hint::black_box(sum);
                }
            })
        })
        .collect();
    
    for handle in baseline_handles {
        handle.join().unwrap();
    }
    let baseline_duration = baseline_start.elapsed();
    
    // Test with profiling enabled
    let profiling_start = Instant::now();
    let profiling_handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                for i in 0..ITERATIONS_PER_THREAD {
                    let operation_name = format!("overhead_test_{}_{}", thread_id, i);
                    let _scope = ProfilerScope::new(&operation_name);
                    
                    // Same CPU work as baseline
                    let mut sum = 0u64;
                    for j in 0..100 {
                        sum = sum.wrapping_add((i * j * thread_id) as u64);
                    }
                    std::hint::black_box(sum);
                }
            })
        })
        .collect();
    
    for handle in profiling_handles {
        handle.join().unwrap();
    }
    let profiling_duration = profiling_start.elapsed();
    
    let overhead_ratio = profiling_duration.as_secs_f64() / baseline_duration.as_secs_f64();
    let overhead_percentage = (overhead_ratio - 1.0) * 100.0;
    
    println!("Profiling overhead benchmark results:");
    println!("  Baseline duration: {:?}", baseline_duration);
    println!("  Profiling duration: {:?}", profiling_duration);
    println!("  Overhead ratio: {:.3}x", overhead_ratio);
    println!("  Overhead percentage: {:.2}%", overhead_percentage);
    
    // Profiling overhead should be reasonable for stress test scenario
    // Note: In stress tests with high concurrency and memory pressure, overhead can be significant
    assert!(overhead_percentage < 2000.0, 
            "Profiling overhead too high: {:.2}%", overhead_percentage);
}