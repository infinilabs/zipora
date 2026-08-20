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
use zipora::fsa::{TokenManager, ZiporaTrie, ZiporaTrieConfig, with_reader_token, with_writer_token};
use zipora::fsa::version_sync::ConcurrencyLevel;

// All trie variants use ZiporaTrie with strategy-based config.
let mut trie: ZiporaTrie = ZiporaTrie::new(); // default config

// Or use an explicit strategy config:
let mut trie: ZiporaTrie = ZiporaTrie::with_config(ZiporaTrieConfig::sparse_optimized());
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Token-based synchronization is managed through a TokenManager;
// the reader/writer closures are free functions over the manager.
let token_manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);

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

### ConcurrentCsppTrie (Multi-Writer CSPP Trie)

The `ConcurrentCsppTrie` provides true multi-writer/multi-reader concurrent access to a Compressed Sparse Parallel Patricia trie, using epoch-based reclamation (crossbeam-epoch) and optimistic per-node locking.

```rust
use zipora::fsa::cspp_trie_concurrent::ConcurrentCsppTrie;
use crossbeam_epoch as epoch;
use std::sync::Arc;

let trie = Arc::new(ConcurrentCsppTrie::with_capacity(0, 10_000_000));

// Multi-threaded insert (each thread gets its own epoch guard for batching)
let trie_clone = Arc::clone(&trie);
std::thread::spawn(move || {
    let guard = epoch::pin();
    for i in 0..10_000 {
        let key = format!("key_{:06}", i);
        trie_clone.insert_with_guard(key.as_bytes(), &guard);
    }
});

// Concurrent reads are lock-free
assert!(trie.contains(b"key_000000"));
```

**Performance**: 6.9M single-thread insert/sec, 8.0M lookup/sec, 10+ M keys/sec at 16 threads with batched guards.

### Performance Characteristics

- **Single-threaded overhead**: < 5% compared to no synchronization
- **Multi-reader scaling**: Linear up to 8+ cores
- **Writer throughput**: 90%+ of single-threaded for OneWriteMultiRead
- **ConcurrentCsppTrie**: 10+ M keys/sec multi-writer (16 threads, batched guards)
- **Token cache hit rate**: 80%+ for repeated operations
- **Memory overhead**: < 10% additional memory usage

## Low-Level Synchronization Primitives

Low-level primitives live in the `zipora::thread` module.

### Linux Futex Integration (Linux only)

`FutexMutex`, `FutexCondvar`, and `FutexRwLock` are built directly on the `futex(2)` syscall via the `LinuxFutex` backend (`PlatformSync` trait, with a portable fallback on non-Linux targets).

```rust
use zipora::thread::{FutexMutex, FutexCondvar, FutexRwLock};

let mutex = FutexMutex::new();
{
    let _guard = mutex.lock().unwrap();
    // critical section; released on drop
}

let condvar = FutexCondvar::new();
let guard = mutex.lock().unwrap();
let _guard = condvar.wait(guard).unwrap(); // returns the re-acquired guard
condvar.notify_one().unwrap();

let rwlock = FutexRwLock::new();
let r = rwlock.read().unwrap();
drop(r);
let _w = rwlock.write().unwrap();
```

### Instance-Specific Thread-Local Storage

- `InstanceTls<T>` — per-instance TLS: each object gets its own thread-local slot (unlike `thread_local!`, which is per-type)
- `OwnerTls<T, O>` — TLS keyed by an owner object (`get_or_create(&owner)`)
- `TlsPool<T, const POOL_SIZE: usize>` — pool of TLS instances with round-robin access (`get_next()`)

```rust
use zipora::thread::InstanceTls;

let tls: InstanceTls<u64> = InstanceTls::new().unwrap();
tls.set(42);
assert_eq!(tls.get(), 42); // per-thread, per-instance value
```

### Atomic Operations Framework

Extension traits implemented for the standard `std::sync::atomic` integer types:

- `AtomicExt<T>` — `atomic_maximize`/`atomic_minimize`, `cas_weak`/`cas_strong`, `fetch_add_acq_rel`/`fetch_sub_acq_rel`, `update_if`
- `AsAtomic<T>` — safe reinterpretation of primitive integers as their atomic counterparts
- `AtomicBitOps` — `set_bit`/`clear_bit`/`toggle_bit`/`test_bit`/`find_first_set` (unsigned atomics)

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use zipora::thread::{AtomicExt, AtomicBitOps};

let counter = AtomicU64::new(0);
counter.fetch_add_acq_rel(1);
counter.atomic_maximize(42, Ordering::AcqRel);

let bits = AtomicU64::new(0);
bits.set_bit(7);
assert!(bits.test_bit(7));
```
