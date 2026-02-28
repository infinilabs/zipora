# Concurrency & Synchronization

Zipora provides sophisticated concurrency primitives designed for high-performance multi-threaded applications.

## Five-Level Concurrency Management System

Zipora implements a 5-level concurrency management system that provides graduated concurrency control options for different performance and threading requirements. The system automatically selects the optimal level based on CPU core count, allocation patterns, and workload characteristics.

### The 5 Levels of Concurrency Control

1. **Level 1: No Locking** - Pure single-threaded operation with zero synchronization overhead
2. **Level 2: Mutex-based Locking** - Fine-grained locking with separate mutexes per size class
3. **Level 3: Lock-free Programming** - Atomic compare-and-swap operations for small allocations
4. **Level 4: Thread-local Caching** - Per-thread local memory pools to minimize cross-thread contention
5. **Level 5: Fixed Capacity Variant** - Bounded memory allocation with no expansion

### Key Benefits

- **API Compatibility**: All levels share consistent interfaces
- **Graduated Complexity**: Each level builds sophistication while maintaining simpler fallbacks
- **Hardware Awareness**: Cache alignment, atomic operations, prefetching
- **Adaptive Selection**: Choose appropriate level based on thread count, allocation patterns, and performance requirements
- **Composability**: Different components can use different concurrency levels

### Usage Examples

```rust
use zipora::memory::{
    AdaptiveFiveLevelPool, ConcurrencyLevel, FiveLevelPoolConfig,
    NoLockingPool, MutexBasedPool, LockFreePool, ThreadLocalPool, FixedCapacityPool,
};

// Automatic adaptive selection (recommended)
let config = FiveLevelPoolConfig::performance_optimized();
let mut pool = AdaptiveFiveLevelPool::new(config).unwrap();
let offset = pool.alloc(1024).unwrap();
println!("Selected level: {:?}", pool.current_level());

// Explicit level selection for specific requirements
let pool = AdaptiveFiveLevelPool::with_level(config, ConcurrencyLevel::ThreadLocal).unwrap();

// Direct use of specific levels
let mut single_thread_pool = NoLockingPool::new(config.clone()).unwrap();
let mutex_pool = MutexBasedPool::new(config.clone()).unwrap();
let lockfree_pool = LockFreePool::new(config.clone()).unwrap();
let threadlocal_pool = ThreadLocalPool::new(config.clone()).unwrap();
let mut fixed_pool = FixedCapacityPool::new(config).unwrap();

// Configuration presets for different use cases
let performance_config = FiveLevelPoolConfig::performance_optimized(); // High throughput
let memory_config = FiveLevelPoolConfig::memory_optimized();           // Low memory usage
let realtime_config = FiveLevelPoolConfig::realtime();                 // Predictable latency
```

### Adaptive Selection Logic

The system intelligently selects the optimal concurrency level:

- **Single-threaded**: Level 1 (No Locking) for maximum performance
- **2-4 cores**: Level 2 (Mutex) or Level 3 (Lock-free) based on allocation size
- **5-16 cores**: Level 3 (Lock-free) or Level 4 (Thread-local) based on arena size
- **16+ cores**: Level 4 (Thread-local) for maximum scalability
- **Fixed capacity**: Level 5 for real-time and constrained environments

### Performance Characteristics

| Level | Scalability | Overhead | Use Case |
|-------|-------------|----------|----------|
| **Level 1** | Single-thread | **Minimal** | Single-threaded applications |
| **Level 2** | Good (2-8 threads) | Low | General multi-threaded use |
| **Level 3** | Excellent (8+ threads) | **Minimal** | High-contention scenarios |
| **Level 4** | **Outstanding** | Low | Very high concurrency |
| **Level 5** | Variable | **Minimal** | Real-time/embedded systems |

## Version-Based Synchronization for FSA and Tries

Zipora includes advanced token and version sequence management for safe concurrent access to Finite State Automata and Trie data structures.

### Key Features

- **Graduated Concurrency Control**: Five levels from read-only to full multi-writer scenarios
- **Token-Based Access Control**: Type-safe reader/writer tokens with automatic RAII lifecycle
- **Version Sequence Management**: Atomic version counters with consistency validation
- **Thread-Local Token Caching**: High-performance token reuse with zero allocation overhead
- **Memory Safety**: Zero unsafe operations in public APIs

### Usage Examples

```rust
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, CompressedSparseTrie, ConcurrencyLevel};

// All trie variants use ZiporaTrie with strategy-based config.
// CompressedSparseTrie is a compatibility wrapper for concurrent access patterns.
let trie = CompressedSparseTrie::new(ConcurrencyLevel::OneWriteMultiRead).unwrap();

// Or use ZiporaTrie directly with sparse_optimized config:
let mut trie = ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized());
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Advanced operations with explicit token control
trie.with_writer_token(|trie, token| {
    trie.insert_with_token(b"advanced", 168, token)?;
    Ok(())
}).unwrap();

// Direct token management for fine-grained control
let token_manager = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);

with_reader_token(&token_manager, |token| {
    assert!(token.is_valid());
    Ok(())
}).unwrap();

with_writer_token(&token_manager, |token| {
    assert!(token.is_valid());
    Ok(())
}).unwrap();
```

### Concurrency Levels

| Level | Description | Use Case | Performance |
|-------|-------------|----------|-------------|
| **Level 0** | `NoWriteReadOnly` | Static data, no writers | **Zero overhead** |
| **Level 1** | `SingleThreadStrict` | Single-threaded apps | **Zero overhead** |
| **Level 2** | `SingleThreadShared` | Single-threaded with token validation | **Minimal overhead** |
| **Level 3** | `OneWriteMultiRead` | Read-heavy workloads | **Excellent reader scaling** |
| **Level 4** | `MultiWriteMultiRead` | High-contention scenarios | **Full concurrency** |

### Performance Characteristics

- **Single-threaded overhead**: < 5% compared to no synchronization
- **Multi-reader scaling**: Linear up to 8+ cores
- **Writer throughput**: 90%+ of single-threaded for OneWriteMultiRead
- **Token cache hit rate**: 80%+ for repeated operations
- **Memory overhead**: < 10% additional memory usage

## Low-Level Synchronization Primitives

### Linux Futex Integration

```rust
use zipora::sync::{Futex, FutexWaiter};

let futex = Futex::new(0);

// Wait for condition
futex.wait(0, None).unwrap();

// Wake up waiters
futex.wake(1).unwrap();
```

### Thread-Local Storage

```rust
use zipora::sync::ThreadLocalStorage;

thread_local! {
    static CACHE: ThreadLocalStorage<Vec<u8>> = ThreadLocalStorage::new();
}

CACHE.with(|cache| {
    cache.borrow_mut().push(42);
});
```

### Atomic Operations Framework

```rust
use zipora::sync::{AtomicCounter, AtomicFlag};

let counter = AtomicCounter::new(0);
counter.fetch_add(1);

let flag = AtomicFlag::new();
flag.set();
assert!(flag.is_set());
```
