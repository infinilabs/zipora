//! Security tests for MemoryPool thread safety vulnerabilities
//!
//! WARNING: These tests demonstrate actual vulnerabilities in the current implementation.
//! They may cause crashes, data corruption, or undefined behavior.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use zipora::memory::{MemoryPool, PoolConfig};

#[test]
#[ignore] // Ignore by default as it may crash
fn test_use_after_free_vulnerability() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());
    let barrier = Arc::new(Barrier::new(2));

    let pool1 = pool.clone();
    let barrier1 = barrier.clone();

    let handle1 = thread::spawn(move || {
        let ptr = pool1.allocate().unwrap();
        let raw_ptr = ptr.as_ptr(); // Extract raw pointer
        let raw_addr = raw_ptr as usize; // Convert to address for sending between threads

        // Write identifiable pattern
        unsafe {
            for i in 0..8 {
                raw_ptr.add(i).write(0xAA);
            }
        }

        // Deallocate
        pool1.deallocate(ptr).unwrap();

        // Signal that we've deallocated
        barrier1.wait();

        // VULNERABILITY: We can still access the pointer after deallocation
        unsafe {
            // This is use-after-free!
            for i in 0..8 {
                raw_ptr.add(i).write(0xDD);
            }
        }

        raw_addr // Return address instead of raw pointer
    });

    let pool2 = pool.clone();
    let barrier2 = barrier.clone();

    let handle2 = thread::spawn(move || {
        // Wait for thread 1 to deallocate
        barrier2.wait();

        // We might get the same pointer that thread 1 just freed
        let ptr = pool2.allocate().unwrap();

        // Read potentially corrupted data
        let mut corrupted = false;
        unsafe {
            for i in 0..8 {
                let value = ptr.as_ptr().add(i).read();
                if value == 0xDD {
                    corrupted = true;
                    break;
                }
            }
        }

        pool2.deallocate(ptr).unwrap();
        corrupted
    });

    let _raw_addr1 = handle1.join().unwrap();
    let corrupted = handle2.join().unwrap();

    // This assertion might fail, demonstrating the vulnerability
    if corrupted {
        println!("VULNERABILITY CONFIRMED: Use-after-free detected!");
        println!("Thread 2 read data written by Thread 1 AFTER deallocation");
    }
}

#[test]
fn test_lost_deallocations_under_contention() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 100, 8)).unwrap());
    let initial_stats = pool.stats();

    // Create high contention scenario
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let pool = pool.clone();
            thread::spawn(move || {
                let mut allocations = Vec::new();

                // Allocate multiple chunks
                for _ in 0..10 {
                    if let Ok(ptr) = pool.allocate() {
                        allocations.push(ptr);
                    }
                }

                // Try to deallocate under contention
                // Some of these might fail silently due to try_lock
                for ptr in allocations {
                    let _ = pool.deallocate(ptr);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    // Give some time for any pending operations
    thread::sleep(Duration::from_millis(100));

    let final_stats = pool.stats();

    // Check if deallocations were lost
    if final_stats.dealloc_count < 100 {
        println!(
            "WARNING: Only {} deallocations recorded out of 100",
            final_stats.dealloc_count
        );
        println!("Lost deallocations: {}", 100 - final_stats.dealloc_count);
    }

    // Check for memory leaks
    if final_stats.allocated > initial_stats.allocated {
        println!(
            "MEMORY LEAK DETECTED: {} bytes leaked",
            final_stats.allocated - initial_stats.allocated
        );
    }
}

#[test]
fn test_stats_race_condition() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 50, 8)).unwrap());
    let barrier = Arc::new(Barrier::new(5));

    let handles: Vec<_> = (0..5)
        .map(|_| {
            let pool = pool.clone();
            let barrier = barrier.clone();

            thread::spawn(move || {
                barrier.wait(); // Ensure all threads start simultaneously

                // Rapid allocate/deallocate to cause stats races
                for _ in 0..100 {
                    if let Ok(ptr) = pool.allocate() {
                        // Immediate deallocation
                        let _ = pool.deallocate(ptr);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let stats = pool.stats();

    // With Relaxed ordering, these might not match
    let expected_allocs = 500; // 5 threads * 100 allocations
    let expected_deallocs = 500;

    if stats.alloc_count != expected_allocs {
        println!(
            "Stats race detected: Expected {} allocations, got {}",
            expected_allocs, stats.alloc_count
        );
    }

    if stats.dealloc_count != expected_deallocs {
        println!(
            "Stats race detected: Expected {} deallocations, got {}",
            expected_deallocs, stats.dealloc_count
        );
    }

    // Pool hits + misses should equal total allocations
    let total_pool_ops = stats.pool_hits + stats.pool_misses;
    if total_pool_ops != stats.alloc_count {
        println!(
            "Stats inconsistency: pool_hits({}) + pool_misses({}) != alloc_count({})",
            stats.pool_hits, stats.pool_misses, stats.alloc_count
        );
    }
}

#[test]
fn test_double_free_attempt() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());

    let ptr = pool.allocate().unwrap();

    // First deallocation - should succeed
    assert!(pool.deallocate(ptr).is_ok());

    // Second deallocation of same pointer - VULNERABILITY!
    // This should fail but currently might succeed and corrupt the pool
    let result = pool.deallocate(ptr);

    if result.is_ok() {
        println!("VULNERABILITY: Double-free succeeded! Pool is now corrupted.");

        // Try to allocate - might get the same pointer twice
        let ptr1 = pool.allocate().unwrap();
        let ptr2 = pool.allocate().unwrap();

        if ptr1.as_ptr() == ptr.as_ptr() || ptr2.as_ptr() == ptr.as_ptr() {
            println!("CRITICAL: Same memory allocated multiple times!");
        }
    }
}

#[test]
#[should_panic]
#[ignore] // May cause undefined behavior
fn test_clear_during_active_use() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());

    let pool1 = pool.clone();
    let handle = thread::spawn(move || {
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            if let Ok(ptr) = pool1.allocate() {
                ptrs.push(ptr);
                unsafe {
                    // Write to allocated memory
                    ptr.as_ptr().write(42);
                }
            }
        }

        thread::sleep(Duration::from_millis(50));

        // Try to use the pointers after clear
        for ptr in ptrs {
            unsafe {
                // This might segfault if clear() was called
                let _value = ptr.as_ptr().read();
            }
        }
    });

    thread::sleep(Duration::from_millis(25));

    // Clear the pool while another thread is using it
    pool.clear().unwrap();

    handle.join().unwrap();
}

#[test]
fn test_send_sync_safety() {
    // This test verifies that the unsafe Send/Sync implementation
    // allows potentially dangerous cross-thread pointer sharing

    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}

    // These should NOT compile for a type containing raw pointers
    // but they do due to unsafe impl Send/Sync
    is_send::<MemoryPool>();
    is_sync::<MemoryPool>();

    println!("WARNING: MemoryPool implements Send/Sync despite containing raw pointers!");
    println!("This violates Rust's memory safety guarantees.");
}

#[test]
fn demonstrate_toctou_vulnerability() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 1, 8)).unwrap());
    let barrier = Arc::new(Barrier::new(2));

    let pool1 = pool.clone();
    let barrier1 = barrier.clone();

    let handle1 = thread::spawn(move || {
        barrier1.wait();
        // Try to allocate when pool has exactly 1 chunk
        match pool1.allocate() {
            Ok(ptr) => Ok(ptr.as_ptr() as usize), // Convert to address
            Err(e) => Err(e),
        }
    });

    let pool2 = pool.clone();
    let barrier2 = barrier.clone();

    let handle2 = thread::spawn(move || {
        barrier2.wait();
        // Race with thread 1
        match pool2.allocate() {
            Ok(ptr) => Ok(ptr.as_ptr() as usize), // Convert to address
            Err(e) => Err(e),
        }
    });

    let result1 = handle1.join().unwrap();
    let result2 = handle2.join().unwrap();

    // Both might succeed if there's a TOCTOU bug
    if result1.is_ok() && result2.is_ok() {
        println!("Potential TOCTOU: Both threads got allocation from single-chunk pool");

        let stats = pool.stats();
        if stats.pool_hits > 0 && stats.pool_misses > 0 {
            println!("Confirmed: Stats show both pool hit and miss for concurrent access");
        }
    }
}
