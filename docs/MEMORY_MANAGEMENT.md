# Memory Management

Zipora provides sophisticated memory management systems for high-performance applications.

## Secure Memory Management

```rust
use zipora::{SecureMemoryPool, SecurePoolConfig, BumpAllocator, PooledVec};

// Production-ready secure memory pools
let config = SecurePoolConfig::small_secure();
let pool = SecureMemoryPool::new(config).unwrap();

// RAII-based allocation - automatic cleanup, no manual deallocation
let ptr = pool.allocate().unwrap();
println!("Allocated {} bytes safely", ptr.size());

// Use memory through safe interface
let slice = ptr.as_slice();
// ptr automatically freed on drop - no use-after-free possible!

// Global thread-safe pools for common sizes
let small_ptr = zipora::get_global_pool_for_size(1024).allocate().unwrap();

// Bump allocator for sequential allocation
let bump = BumpAllocator::new(1024 * 1024).unwrap();
let ptr = bump.alloc::<u64>().unwrap();

// Pooled containers with automatic pool allocation
let mut pooled_vec = PooledVec::<i32>::new().unwrap();
pooled_vec.push(42).unwrap();

// Linux hugepage support for large datasets
#[cfg(target_os = "linux")]
{
    use zipora::HugePage;
    let hugepage = HugePage::new_2mb(2 * 1024 * 1024).unwrap();
}
```

## Lock-Free Memory Pool

```rust
use zipora::memory::{LockFreeMemoryPool, LockFreePoolConfig, BackoffStrategy};

// High-performance concurrent allocation without locks
let config = LockFreePoolConfig::high_performance();
let pool = LockFreeMemoryPool::new(config).unwrap();

// Concurrent allocation from multiple threads.
// allocate() returns a raw NonNull<u8> — NOT an RAII guard.
let ptr = pool.allocate(1024).unwrap();

// ... use the memory ...

// Explicit deallocation is required (dropping the pointer does NOT free it)
pool.deallocate(ptr, 1024).unwrap();

// Advanced configuration options
let config = LockFreePoolConfig {
    memory_size: 256 * 1024 * 1024, // 256MB backing memory
    enable_stats: true,
    max_cas_retries: 10000,
    backoff_strategy: BackoffStrategy::Exponential { max_delay_us: 100 },
    ..Default::default()
};

// Performance statistics
if let Some(stats) = pool.stats() {
    println!("CAS contention ratio: {:.2}%", stats.contention_ratio() * 100.0);
    println!("Allocation rate: {:.0} allocs/sec", stats.allocation_rate());
}
```

## Thread-Local Memory Pool

```rust
use zipora::memory::{ThreadLocalMemoryPool, ThreadLocalPoolConfig};

// Per-thread allocation caches for zero contention
let config = ThreadLocalPoolConfig::high_performance();
let pool = ThreadLocalMemoryPool::new(config).unwrap();

// Hot area allocation - sequential allocation from thread-local arena
let alloc = pool.allocate(64).unwrap();

// Thread-local free list caching
let cached_alloc = pool.allocate(64).unwrap(); // Likely cache hit

// Configuration for different scenarios
let config = ThreadLocalPoolConfig {
    arena_size: 8 * 1024 * 1024, // 8MB per thread
    max_threads: 1024,
    sync_threshold: 1024 * 1024, // 1MB lazy sync threshold
    use_secure_memory: false, // Disable for max performance
    ..ThreadLocalPoolConfig::default()
};

// Performance monitoring
if let Some(stats) = pool.stats() {
    println!("Cache hit ratio: {:.1}%", stats.hit_ratio() * 100.0);
    println!("Locality score: {:.2}", stats.locality_score());
}
```

## Fixed Capacity Memory Pool

```rust
use zipora::memory::{FixedCapacityMemoryPool, FixedCapacityPoolConfig};

// Bounded memory pool for real-time systems
let config = FixedCapacityPoolConfig::realtime();
let total_blocks = config.total_blocks;
let pool = FixedCapacityMemoryPool::new(config).unwrap();

// Guaranteed allocation within capacity
let alloc = pool.allocate(1024).unwrap();

// Capacity management
println!("Total capacity: {} bytes", pool.total_capacity());
println!("Available: {} bytes", pool.available_capacity());
assert!(pool.has_capacity(2048));

// Configuration for different use cases
let config = FixedCapacityPoolConfig {
    max_block_size: 8192,
    total_blocks: 5000,
    alignment: 64, // Cache line aligned
    enable_stats: false, // Minimize overhead
    eager_allocation: true, // Pre-allocate all memory
    secure_clear: true, // Zero memory on deallocation
};

// Real-time performance monitoring
if let Some(stats) = pool.stats() {
    println!("Utilization: {:.1}%", stats.utilization_percent());
    println!("Success rate: {:.3}", stats.success_rate());
    // is_at_capacity expects a block count (not bytes)
    assert!(!stats.is_at_capacity(total_blocks));
}
```

## Memory-Mapped Vectors

`MmapVec` is gated behind the `mmap` feature (enabled by default).

```rust
use zipora::memory::{MmapVec, MmapVecConfig, MmapVecStats};

// Persistent vector backed by memory-mapped file
let config = MmapVecConfig::large_dataset();
let mut vec = MmapVec::<u64>::create("data.mmap", config).unwrap();

// Standard vector operations with persistence
vec.push(42).unwrap();
vec.push(84).unwrap();
assert_eq!(vec.len(), 2);
assert_eq!(vec.get(0), Some(&42));

// Automatic growth and persistence
vec.reserve(1_000_000).unwrap(); // Reserve for 1M elements
for i in 0..1000 {
    vec.push(i).unwrap();
}

// Cross-process data sharing
vec.sync().unwrap(); // Force sync to disk

// Configuration presets for different use cases
let performance_config = MmapVecConfig::performance_optimized(); // Golden ratio growth
let memory_config = MmapVecConfig::memory_optimized();           // Conservative growth
let realtime_config = MmapVecConfig::realtime();                 // Predictable performance

// Builder pattern for custom configurations
let config = MmapVecConfig::builder()
    .with_initial_capacity(8192)
    .with_growth_factor(1.618)  // Golden ratio growth
    .with_populate_pages(true)  // Pre-load for performance
    .with_sync_on_write(false)  // Async writes for performance
    .build();

// Advanced operations
vec.extend(&[1, 2, 3, 4, 5]).unwrap();
vec.truncate(100).unwrap();
vec.resize(200, 0).unwrap();
vec.shrink_to_fit().unwrap();

// Memory usage statistics
let stats = vec.stats();
println!("Total size: {} bytes", stats.total_size);
println!("Utilization: {:.1}%", stats.utilization * 100.0);
println!("File path: {}", vec.path().display());

// Iterator support
for &value in &vec {
    println!("Value: {}", value);
}
```

## Memory Pool Comparison

| Pool Type | Thread Safety | Overhead | Best Use Case |
|-----------|---------------|----------|---------------|
| **SecureMemoryPool** | Mutex-based | Moderate | Security-critical applications |
| **LockFreeMemoryPool** | Lock-free CAS | Minimal | High-contention multi-threaded |
| **ThreadLocalMemoryPool** | Thread-local | Zero contention | Thread-bound allocations |
| **FixedCapacityMemoryPool** | Configurable | Minimal | Real-time systems |
| **MmapVec** | External sync | I/O bound | Persistent storage |

## Cache Layout Optimization

```rust
use zipora::memory::cache_layout::*;

// Pick a constructor preset: new(), sequential(), random(), write_heavy(), read_heavy()
let mut config = CacheLayoutConfig::sequential();

// Fine-tune via public fields (there are no chainable with_* methods)
config.cache_line_size = 64;
config.prefetch_distance = 128;

let allocator = CacheOptimizedAllocator::new(config.clone());

// Cache-aligned allocation: (size, alignment, is_hot) -> Result<NonNull<u8>>
let ptr = allocator.allocate_aligned(1024, 64, true)?;

// Hot/cold data separation
let mut separator: HotColdSeparator<u64> = HotColdSeparator::new(config);
separator.insert(42, 2000); // access_count >= hot_threshold => hot
separator.insert(7, 3);     // infrequently accessed => cold

let hot = separator.hot_slice();
let cold = separator.cold_slice();

separator.reorganize(); // rebalance items based on recorded access counts
let stats = separator.separation_stats();
println!("hot: {}, cold: {}", stats.hot_items, stats.cold_items);
```

## Performance Characteristics

- **SecureMemoryPool**: RAII safety, automatic cleanup, vulnerability prevention
- **LockFreeMemoryPool**: 10-100K+ allocations/sec per thread
- **ThreadLocalMemoryPool**: Near-zero overhead for thread-local allocations
- **FixedCapacityMemoryPool**: Deterministic allocation time for real-time
- **MmapVec**: Persistence with memory-mapped I/O performance
