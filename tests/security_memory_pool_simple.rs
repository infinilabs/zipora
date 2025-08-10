//! Simplified security tests for MemoryPool that compile and run

use std::sync::Arc;
use std::thread;
use std::time::Duration;
use zipora::memory::{MemoryPool, PoolConfig};

#[test]
fn test_lost_stats_under_contention() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 100, 8)).unwrap());

    // Create high contention scenario
    let handles: Vec<_> = (0..10)
        .map(|thread_id| {
            let pool = pool.clone();
            thread::spawn(move || {
                let mut local_allocs = 0;
                let mut local_deallocs = 0;

                // Each thread does 100 allocate/deallocate cycles
                for _ in 0..100 {
                    if let Ok(ptr) = pool.allocate() {
                        local_allocs += 1;

                        // Write to memory to ensure it's actually allocated
                        unsafe {
                            ptr.as_ptr().write(thread_id as u8);
                        }

                        // Small delay to increase contention
                        thread::yield_now();

                        if pool.deallocate(ptr).is_ok() {
                            local_deallocs += 1;
                        }
                    }
                }

                (local_allocs, local_deallocs)
            })
        })
        .collect();

    let mut total_local_allocs = 0;
    let mut total_local_deallocs = 0;

    for handle in handles {
        let (allocs, deallocs) = handle.join().unwrap();
        total_local_allocs += allocs;
        total_local_deallocs += deallocs;
    }

    // Give time for any pending operations
    thread::sleep(Duration::from_millis(100));

    let stats = pool.stats();

    println!("=== Lost Stats Test Results ===");
    println!(
        "Local tracking: {} allocs, {} deallocs",
        total_local_allocs, total_local_deallocs
    );
    println!(
        "Pool stats: {} allocs, {} deallocs",
        stats.alloc_count, stats.dealloc_count
    );

    if stats.alloc_count != total_local_allocs as u64 {
        println!(
            "WARNING: Lost {} allocation records",
            total_local_allocs as i64 - stats.alloc_count as i64
        );
    }

    if stats.dealloc_count != total_local_deallocs as u64 {
        println!(
            "WARNING: Lost {} deallocation records",
            total_local_deallocs as i64 - stats.dealloc_count as i64
        );
    }

    // Check consistency of pool operations
    let total_pool_ops = stats.pool_hits + stats.pool_misses;
    if total_pool_ops != stats.alloc_count {
        println!(
            "CRITICAL: Inconsistent stats - pool_hits({}) + pool_misses({}) != alloc_count({})",
            stats.pool_hits, stats.pool_misses, stats.alloc_count
        );
    }
}

#[test]
fn test_double_free_safety() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());

    let ptr = pool.allocate().unwrap();
    let raw_ptr = ptr.as_ptr();

    // Write a canary value
    unsafe {
        raw_ptr.write_bytes(0xAB, 64);
    }

    // First deallocation - should succeed
    assert!(pool.deallocate(ptr).is_ok());

    // Try to deallocate again (simulating double-free)
    // We need to recreate NonNull for the test
    let double_free_ptr = unsafe { std::ptr::NonNull::new_unchecked(raw_ptr) };

    let result = pool.deallocate(double_free_ptr);

    println!("=== Double Free Test ===");
    if result.is_ok() {
        println!("VULNERABILITY CONFIRMED: Double-free succeeded!");
        println!("The pool accepted the same pointer twice.");

        // Check if we can get the same pointer multiple times
        let ptr1 = pool.allocate();
        let ptr2 = pool.allocate();

        if let (Ok(p1), Ok(p2)) = (ptr1, ptr2) {
            if p1.as_ptr() == raw_ptr {
                println!("CRITICAL: Freed pointer reallocated (ptr1)!");
            }
            if p2.as_ptr() == raw_ptr {
                println!("CRITICAL: Freed pointer reallocated (ptr2)!");
            }
            if p1.as_ptr() == p2.as_ptr() {
                println!("CRITICAL: Same memory allocated to two different requests!");
            }
        }
    } else {
        println!("Good: Double-free was rejected");
    }
}

#[test]
fn test_send_sync_implementation() {
    println!("=== Send/Sync Safety Analysis ===");

    // Check if MemoryPool implements Send and Sync
    fn check_send<T: Send>(_: &T) -> bool {
        true
    }
    fn check_sync<T: Sync>(_: &T) -> bool {
        true
    }

    let pool = MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap();

    let is_send = check_send(&pool);
    let is_sync = check_sync(&pool);

    println!("MemoryPool implements Send: {}", is_send);
    println!("MemoryPool implements Sync: {}", is_sync);

    if is_send && is_sync {
        println!("WARNING: MemoryPool manually implements Send and Sync!");
        println!("This is dangerous because it contains raw pointers (*mut u8)");
        println!("Raw pointers are !Send and !Sync by default for safety reasons");
        println!();
        println!("Security implications:");
        println!("1. Pointers could reference thread-local data");
        println!("2. No guarantee of proper synchronization");
        println!("3. Potential for use-after-free across threads");
        println!("4. Violates Rust's memory safety guarantees");
    }
}

#[test]
fn test_clear_safety() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 10, 8)).unwrap());

    // Allocate some chunks
    let mut allocations = Vec::new();
    for i in 0..5 {
        if let Ok(ptr) = pool.allocate() {
            unsafe {
                ptr.as_ptr().write(i as u8);
            }
            allocations.push(ptr);
        }
    }

    // Return some to pool
    for ptr in allocations.iter().take(3) {
        let _ = pool.deallocate(*ptr);
    }

    let stats_before = pool.stats();
    println!("=== Clear Safety Test ===");
    println!("Before clear: {} chunks in pool", stats_before.chunks);

    // Clear the pool
    pool.clear().unwrap();

    let stats_after = pool.stats();
    println!("After clear: {} chunks in pool", stats_after.chunks);

    // Try to use remaining allocations (indices 3 and 4)
    // These were NOT returned to pool, so they should still be valid
    // But if clear() is buggy, it might have freed them anyway

    println!("Testing if non-pooled allocations are still valid...");
    for (i, ptr) in allocations.iter().skip(3).enumerate() {
        unsafe {
            let value = ptr.as_ptr().read();
            if value != (i + 3) as u8 {
                println!("CORRUPTION: Expected {}, got {}", i + 3, value);
            }
        }
    }
}

#[test]
fn test_pool_capacity_overflow() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 2, 8)).unwrap());

    println!("=== Pool Capacity Test ===");
    println!("Pool configured for max 2 chunks");

    // Allocate 3 chunks
    let ptr1 = pool.allocate().unwrap();
    let ptr2 = pool.allocate().unwrap();
    let ptr3 = pool.allocate().unwrap();

    // Return all to pool
    pool.deallocate(ptr1).unwrap();
    pool.deallocate(ptr2).unwrap();
    pool.deallocate(ptr3).unwrap(); // This should exceed capacity

    let stats = pool.stats();
    println!("After returning 3 chunks: {} in pool", stats.chunks);

    if stats.chunks > 2 {
        println!("BUG: Pool exceeded configured capacity!");
        println!("This could lead to unbounded memory growth");
    } else if stats.chunks < 2 {
        println!("Note: Pool has {} chunks, capacity is 2", stats.chunks);
        println!("Third chunk was correctly discarded");
    }
}

#[test]
fn test_memory_ordering_consistency() {
    let pool = Arc::new(MemoryPool::new(PoolConfig::new(64, 50, 8)).unwrap());

    let handles: Vec<_> = (0..5)
        .map(|_| {
            let pool = pool.clone();
            thread::spawn(move || {
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

    println!("=== Memory Ordering Test ===");
    println!("Stats after 500 operations:");
    println!("  Allocations: {}", stats.alloc_count);
    println!("  Deallocations: {}", stats.dealloc_count);
    println!("  Pool hits: {}", stats.pool_hits);
    println!("  Pool misses: {}", stats.pool_misses);

    // With Relaxed ordering, these values might be inconsistent
    if stats.pool_hits + stats.pool_misses != stats.alloc_count {
        println!("WARNING: Relaxed memory ordering caused inconsistent stats!");
        println!(
            "  pool_hits + pool_misses = {}",
            stats.pool_hits + stats.pool_misses
        );
        println!("  alloc_count = {}", stats.alloc_count);
        println!(
            "  Difference: {}",
            (stats.alloc_count as i64) - (stats.pool_hits + stats.pool_misses) as i64
        );
    }
}
