//! Secure Memory Pool Demo
//!
//! This example demonstrates the usage of the production-ready SecureMemoryPool
//! which eliminates the security vulnerabilities found in the original MemoryPool.
//!
//! Key features demonstrated:
//! - Thread-safe allocation without manual Send/Sync
//! - Automatic memory safety with RAII
//! - Use-after-free and double-free prevention
//! - Corruption detection with validation
//! - High-performance thread-local caching
//! - Comprehensive statistics and monitoring

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use zipora::memory::{
    SecureMemoryPool, SecurePoolConfig, get_global_pool_for_size, 
    get_global_secure_pool_stats, size_to_class
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔒 Secure Memory Pool Demo");
    println!("==========================\n");

    // Demo 1: Basic secure allocation
    demo_basic_allocation()?;
    
    // Demo 2: Security features
    demo_security_features()?;
    
    // Demo 3: Thread safety
    demo_thread_safety()?;
    
    // Demo 4: Performance comparison
    demo_performance_comparison()?;
    
    // Demo 5: Global pools
    demo_global_pools()?;
    
    // Demo 6: Statistics and monitoring
    demo_statistics_monitoring()?;

    println!("\n✅ All demos completed successfully!");
    println!("The SecureMemoryPool provides production-ready memory management");
    println!("with comprehensive security guarantees and high performance.");
    
    Ok(())
}

/// Demonstrate basic secure allocation with RAII
fn demo_basic_allocation() -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 Demo 1: Basic Secure Allocation");
    println!("----------------------------------");
    
    // Create a secure memory pool
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config)?;
    
    println!("✓ Created secure memory pool with {} byte chunks", 
             pool.config().chunk_size);
    
    // Allocate memory - returns RAII guard
    let ptr = pool.allocate()?;
    println!("✓ Allocated {} bytes at generation {}", 
             ptr.size(), ptr.generation());
    
    // Validate allocation integrity
    ptr.validate()?;
    println!("✓ Chunk validation passed");
    
    // Memory access through safe interface
    let slice = ptr.as_slice();
    println!("✓ Safe slice access: {} bytes available", slice.len());
    
    // Write some data
    let mut_ptr = unsafe {
        std::slice::from_raw_parts_mut(ptr.as_ptr(), ptr.size())
    };
    mut_ptr[0] = 0x42;
    mut_ptr[ptr.size() - 1] = 0x84;
    println!("✓ Wrote test data to allocation");
    
    // Automatic deallocation on drop - no manual management needed!
    drop(ptr);
    println!("✓ Memory automatically freed with RAII");
    
    // Check statistics
    let stats = pool.stats();
    println!("✓ Pool stats: {} allocs, {} deallocs, {} corruptions detected",
             stats.alloc_count, stats.dealloc_count, stats.corruption_detected);
    
    println!();
    Ok(())
}

/// Demonstrate security features and protections
fn demo_security_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  Demo 2: Security Features");
    println!("-----------------------------");
    
    let config = SecurePoolConfig::medium_secure();
    let pool = SecureMemoryPool::new(config)?;
    
    println!("✓ Created secure pool with security features:");
    println!("  - Zero on free: {}", pool.config().zero_on_free);
    println!("  - Guard pages: {}", pool.config().use_guard_pages);
    println!("  - Generation counters: enabled");
    println!("  - Corruption detection: enabled");
    
    // Demonstrate generation-based safety
    let ptr1 = pool.allocate()?;
    let gen1 = ptr1.generation();
    drop(ptr1);
    
    let ptr2 = pool.allocate()?;
    let gen2 = ptr2.generation();
    drop(ptr2);
    
    println!("✓ Generation counter incremented: {} → {}", gen1, gen2);
    println!("  This prevents use-after-free attacks");
    
    // Show that double-free is structurally impossible
    let ptr = pool.allocate()?;
    println!("✓ Allocated chunk with generation {}", ptr.generation());
    // Cannot manually free - RAII prevents double-free by design
    drop(ptr);
    println!("✓ RAII design makes double-free structurally impossible");
    
    // Demonstrate validation
    let ptr = pool.allocate()?;
    ptr.validate()?;
    println!("✓ Chunk validation detects corruption attempts");
    drop(ptr);
    
    let stats = pool.stats();
    println!("✓ Security stats: {} double-free attempts blocked, {} corruptions detected",
             stats.double_free_detected, stats.corruption_detected);
    
    println!();
    Ok(())
}

/// Demonstrate thread safety without manual Send/Sync
fn demo_thread_safety() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧵 Demo 3: Thread Safety");
    println!("------------------------");
    
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config)?;
    
    println!("✓ Creating pool shared across threads");
    println!("  No manual Send/Sync implementation needed!");
    
    let start = Instant::now();
    let total_allocations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    
    // Spawn multiple threads doing concurrent allocations
    let handles: Vec<_> = (0..8).map(|thread_id| {
        let pool = pool.clone();
        let counter = total_allocations.clone();
        
        thread::spawn(move || {
            println!("  Thread {} starting allocations", thread_id);
            
            for i in 0..1000 {
                // Allocate memory
                let ptr = pool.allocate().unwrap();
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                
                // Validate allocation
                ptr.validate().unwrap();
                
                // Write thread-specific data
                let mut_slice = unsafe {
                    std::slice::from_raw_parts_mut(ptr.as_ptr(), ptr.size())
                };
                mut_slice[0] = thread_id as u8;
                mut_slice[1] = (i % 256) as u8;
                
                // Simulate some work
                if i % 100 == 0 {
                    thread::sleep(Duration::from_micros(10));
                }
                
                // RAII automatically handles deallocation
            }
            
            println!("  Thread {} completed", thread_id);
        })
    }).collect();
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    let duration = start.elapsed();
    let total = total_allocations.load(std::sync::atomic::Ordering::Relaxed);
    
    println!("✓ Completed {} concurrent allocations in {:?}", total, duration);
    println!("  Throughput: {:.0} allocs/sec", total as f64 / duration.as_secs_f64());
    
    let stats = pool.stats();
    println!("✓ Thread safety verified:");
    println!("  - {} total allocations", stats.alloc_count);
    println!("  - {} local cache hits", stats.local_cache_hits);
    println!("  - {} cross-thread steals", stats.cross_thread_steals);
    println!("  - {} corruptions detected", stats.corruption_detected);
    println!("  - {} double-free attempts", stats.double_free_detected);
    
    println!();
    Ok(())
}

/// Compare performance with standard allocator
fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Demo 4: Performance Comparison");
    println!("---------------------------------");
    
    const ITERATIONS: usize = 10_000;
    
    // Benchmark standard allocator
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            panic!("Allocation failed");
        }
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
    let std_duration = start.elapsed();
    
    // Benchmark secure pool
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config)?;
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let ptr = pool.allocate()?;
        drop(ptr);
    }
    let pool_duration = start.elapsed();
    
    println!("Performance comparison ({} iterations):", ITERATIONS);
    println!("  Standard allocator: {:?} ({:.0} allocs/sec)", 
             std_duration, ITERATIONS as f64 / std_duration.as_secs_f64());
    println!("  Secure pool:        {:?} ({:.0} allocs/sec)", 
             pool_duration, ITERATIONS as f64 / pool_duration.as_secs_f64());
    
    let ratio = std_duration.as_nanos() as f64 / pool_duration.as_nanos() as f64;
    if ratio < 1.0 {
        println!("  🚀 Secure pool is {:.1}x FASTER than standard allocator!", 1.0 / ratio);
    } else {
        println!("  📊 Secure pool is {:.1}x slower than standard allocator", ratio);
        println!("     (This includes security overhead for memory safety)");
    }
    
    let stats = pool.stats();
    println!("  Pool efficiency: {:.1}% cache hits", 
             (stats.pool_hits + stats.local_cache_hits) as f64 / stats.alloc_count as f64 * 100.0);
    
    println!();
    Ok(())
}

/// Demonstrate global pool usage for different sizes
fn demo_global_pools() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 Demo 5: Global Pools");
    println!("-----------------------");
    
    // Demonstrate size-based pool selection
    let sizes = [64, 1024, 32768, 1048576];
    
    for size in sizes {
        let size_class = size_to_class(size);
        let pool = get_global_pool_for_size(size);
        
        println!("Size {} → class {} → pool with {} byte chunks", 
                 size, size_class, pool.config().chunk_size);
        
        let ptr = pool.allocate()?;
        println!("  ✓ Allocated from global pool, generation {}", ptr.generation());
        drop(ptr);
    }
    
    // Show global statistics
    let global_stats = get_global_secure_pool_stats();
    println!("\nGlobal pool statistics:");
    println!("  Total allocations: {}", global_stats.alloc_count);
    println!("  Total deallocations: {}", global_stats.dealloc_count);
    println!("  Cache hits: {}", global_stats.pool_hits + global_stats.local_cache_hits);
    println!("  Security events: {} corruptions, {} double-free attempts",
             global_stats.corruption_detected, global_stats.double_free_detected);
    
    println!();
    Ok(())
}

/// Demonstrate statistics and monitoring capabilities
fn demo_statistics_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 6: Statistics & Monitoring");
    println!("----------------------------------");
    
    let config = SecurePoolConfig::small_secure();
    let pool = SecureMemoryPool::new(config)?;
    
    // Generate some activity for interesting statistics
    let mut allocations = Vec::new();
    
    // Create mixed allocation patterns
    for i in 0..100 {
        let ptr = pool.allocate()?;
        
        if i % 10 == 0 {
            // Keep some allocations longer
            allocations.push(ptr);
        } else {
            // Immediate deallocation
            drop(ptr);
        }
    }
    
    // Get detailed statistics
    let stats = pool.stats();
    
    println!("Detailed pool statistics:");
    println!("┌─────────────────────────┬─────────────┐");
    println!("│ Metric                  │ Value       │");
    println!("├─────────────────────────┼─────────────┤");
    println!("│ Total allocations       │ {:>11} │", stats.alloc_count);
    println!("│ Total deallocations     │ {:>11} │", stats.dealloc_count);
    println!("│ Pool hits               │ {:>11} │", stats.pool_hits);
    println!("│ Pool misses             │ {:>11} │", stats.pool_misses);
    println!("│ Local cache hits        │ {:>11} │", stats.local_cache_hits);
    println!("│ Cross-thread steals     │ {:>11} │", stats.cross_thread_steals);
    println!("│ Corruptions detected    │ {:>11} │", stats.corruption_detected);
    println!("│ Double-free attempts    │ {:>11} │", stats.double_free_detected);
    println!("└─────────────────────────┴─────────────┘");
    
    // Calculate derived metrics
    let hit_rate = if stats.alloc_count > 0 {
        (stats.pool_hits + stats.local_cache_hits) as f64 / stats.alloc_count as f64 * 100.0
    } else {
        0.0
    };
    
    let cache_efficiency = if stats.pool_hits + stats.local_cache_hits > 0 {
        stats.local_cache_hits as f64 / (stats.pool_hits + stats.local_cache_hits) as f64 * 100.0
    } else {
        0.0
    };
    
    println!("\nDerived metrics:");
    println!("  Cache hit rate: {:.1}%", hit_rate);
    println!("  Local cache efficiency: {:.1}%", cache_efficiency);
    println!("  Memory safety events: {} total", 
             stats.corruption_detected + stats.double_free_detected);
    
    // Validate pool integrity
    pool.validate()?;
    println!("  ✓ Pool integrity validation passed");
    
    // Clean up remaining allocations
    drop(allocations);
    
    // Final statistics
    let final_stats = pool.stats();
    println!("  Final allocations active: {}", 
             final_stats.alloc_count - final_stats.dealloc_count);
    
    println!();
    Ok(())
}

/// Helper function to format duration nicely
#[allow(dead_code)]
fn format_duration(duration: Duration) -> String {
    let nanos = duration.as_nanos();
    if nanos < 1_000 {
        format!("{}ns", nanos)
    } else if nanos < 1_000_000 {
        format!("{:.1}μs", nanos as f64 / 1_000.0)
    } else if nanos < 1_000_000_000 {
        format!("{:.1}ms", nanos as f64 / 1_000_000.0)
    } else {
        format!("{:.1}s", nanos as f64 / 1_000_000_000.0)
    }
}