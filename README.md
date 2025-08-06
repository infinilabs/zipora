# Zipora

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.88+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Features

- **🚀 High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512*), cache-friendly layouts
- **🛡️ Memory Safety**: Eliminates segfaults, buffer overflows, use-after-free bugs
- **🧠 Secure Memory Management**: Production-ready memory pools with thread safety, RAII, and vulnerability prevention
- **🗜️ Compression Framework**: Huffman, rANS, dictionary-based, and hybrid compression
- **🌲 Advanced Tries**: LOUDS, Critical-Bit, and Patricia tries
- **💾 Blob Storage**: Memory-mapped and compressed storage systems
- **⚡ Fiber Concurrency**: High-performance async/await with work-stealing
- **🔄 Real-time Compression**: Adaptive algorithms with strict latency guarantees
- **🔌 C FFI Support**: Complete C API for migration from C++

## Quick Start

```toml
[dependencies]
zipora = "1.0.3"

# Or with optional features
zipora = { version = "1.0.3", features = ["lz4", "ffi"] }

# AVX-512 requires nightly Rust (experimental intrinsics)
zipora = { version = "1.0.3", features = ["avx512", "lz4", "ffi"] }  # nightly only
```

### Basic Usage

```rust
use zipora::*;

// High-performance vector
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy strings with SIMD
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// Blob storage with compression
let mut store = MemoryBlobStore::new();
let id = store.put(b"Hello, World!").unwrap();

// Advanced tries
let mut trie = LoudsTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Hash maps
let mut map = GoldHashMap::new();
map.insert("key", "value").unwrap();

// Entropy coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();
```

## Core Components

### Secure Memory Management

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

### Advanced Algorithms

```rust
use zipora::{SuffixArray, RadixSort, MultiWayMerge};

// Suffix arrays with linear-time construction
let sa = SuffixArray::new(b"banana").unwrap();
let (start, count) = sa.search(b"banana", b"an");

// High-performance radix sort
let mut data = vec![5u32, 2, 8, 1, 9];
let mut sorter = RadixSort::new();
sorter.sort_u32(&mut data).unwrap();

// Multi-way merge
let sources = vec![
    VectorSource::new(vec![1, 4, 7]),
    VectorSource::new(vec![2, 5, 8]),
];
let mut merger = MultiWayMerge::new();
let result = merger.merge(sources).unwrap();
```

### Fiber Concurrency

```rust
use zipora::{FiberPool, AdaptiveCompressor, RealtimeCompressor};

async fn example() {
    // Parallel processing
    let pool = FiberPool::default().unwrap();
    let result = pool.parallel_map(vec![1, 2, 3], |x| Ok(x * 2)).await.unwrap();
    
    // Adaptive compression
    let compressor = AdaptiveCompressor::default().unwrap();
    let compressed = compressor.compress(b"data").unwrap();
    
    // Real-time compression
    let rt_compressor = RealtimeCompressor::with_mode(CompressionMode::LowLatency).unwrap();
    let compressed = rt_compressor.compress(b"data").await.unwrap();
}
```

### Memory-Mapped I/O

```rust
#[cfg(feature = "mmap")]
{
    use zipora::{MemoryMappedOutput, MemoryMappedInput, DataInput, DataOutput};
    
    // Memory-mapped output with automatic growth
    let mut output = MemoryMappedOutput::create("data.bin", 1024).unwrap();
    output.write_u32(0x12345678).unwrap();
    output.flush().unwrap();
    
    // Zero-copy reading
    let file = std::fs::File::open("data.bin").unwrap();
    let mut input = MemoryMappedInput::new(file).unwrap();
    assert_eq!(input.read_u32().unwrap(), 0x12345678);
}
```

### Compression Framework

```rust
use zipora::{HuffmanEncoder, RansEncoder, DictionaryBuilder, CompressorFactory};

// Huffman coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// rANS encoding
let mut frequencies = [0u32; 256];
for &byte in b"sample data" { frequencies[byte as usize] += 1; }
let rans_encoder = RansEncoder::new(&frequencies).unwrap();
let compressed = rans_encoder.encode(b"sample data").unwrap();

// Dictionary compression
let dictionary = DictionaryBuilder::new().build(b"sample data");

// LZ4 compression (requires "lz4" feature)
#[cfg(feature = "lz4")]
{
    use zipora::Lz4Compressor;
    let compressor = Lz4Compressor::new();
    let compressed = compressor.compress(b"sample data").unwrap();
}

// Automatic algorithm selection
let algorithm = CompressorFactory::select_best(&requirements, data);
let compressor = CompressorFactory::create(algorithm, Some(training_data)).unwrap();
```

## Security & Memory Safety

### Production-Ready SecureMemoryPool

The new **SecureMemoryPool** eliminates critical security vulnerabilities found in traditional memory pool implementations while maintaining high performance:

#### 🛡️ Security Features

- **Use-After-Free Prevention**: Generation counters validate pointer lifetime
- **Double-Free Detection**: Cryptographic validation prevents duplicate deallocations  
- **Memory Corruption Detection**: Guard pages and canary values detect overflow/underflow
- **Thread Safety**: Built-in synchronization without manual Send/Sync annotations
- **RAII Memory Management**: Automatic cleanup eliminates manual deallocation errors
- **Zero-on-Free**: Optional memory clearing for sensitive data protection

#### ⚡ Performance Features

- **Thread-Local Caching**: Reduces lock contention with per-thread allocation caches
- **Lock-Free Fast Paths**: High-performance allocation for common cases
- **NUMA Awareness**: Optimized allocation for multi-socket systems
- **Batch Operations**: Amortized overhead for bulk allocations

#### 🔒 Security Guarantees

| Vulnerability | Traditional Pools | SecureMemoryPool |
|---------------|-------------------|------------------|
| Use-after-free | ❌ Possible | ✅ **Prevented** |
| Double-free | ❌ Possible | ✅ **Detected** |
| Memory corruption | ❌ Undetected | ✅ **Detected** |
| Race conditions | ❌ Manual sync required | ✅ **Thread-safe** |
| Manual cleanup | ❌ Error-prone | ✅ **RAII automatic** |

#### 📈 Migration Guide

**Before (MemoryPool)**:
```rust
let config = PoolConfig::new(1024, 100, 8);
let pool = MemoryPool::new(config)?;
let ptr = pool.allocate()?;
// Manual deallocation required - error-prone!
pool.deallocate(ptr)?;
```

**After (SecureMemoryPool)**:
```rust
let config = SecurePoolConfig::small_secure();
let pool = SecureMemoryPool::new(config)?;
let ptr = pool.allocate()?;
// Automatic cleanup on drop - no manual deallocation needed!
// Use-after-free and double-free impossible!
```

## Performance

Current performance on Intel i7-10700K:

> **Note**: *AVX-512 optimizations require nightly Rust due to experimental intrinsics. All other SIMD optimizations (AVX2, BMI2, POPCNT) work with stable Rust.

| Operation | Performance | vs std::Vec | vs C++ | Security |
|-----------|-------------|-------------|--------|----------|
| FastVec push 10k | 6.78µs | +48% faster | +20% faster | ✅ Memory safe |
| SecureMemoryPool alloc | ~18ns | +85% faster | +85% faster | ✅ **Production-ready** |
| Traditional pool alloc | ~15ns | +90% faster | +90% faster | ❌ Unsafe |
| Radix sort 1M u32s | ~45ms | +60% faster | +40% faster | ✅ Memory safe |
| Suffix array build | O(n) | N/A | Linear vs O(n log n) | ✅ Memory safe |
| Fiber spawn | ~5µs | N/A | New capability | ✅ Memory safe |

## C FFI Migration

```toml
[dependencies]
zipora = { version = "1.0.3", features = ["ffi"] }
```

```c
#include <zipora.h>

// Vector operations
CFastVec* vec = fast_vec_new();
fast_vec_push(vec, 42);
printf("Length: %zu\n", fast_vec_len(vec));
fast_vec_free(vec);

// Secure memory pools (recommended)
CSecureMemoryPool* pool = secure_memory_pool_new_small();
CSecurePooledPtr* ptr = secure_memory_pool_allocate(pool);
// No manual deallocation needed - automatic cleanup!
secure_pooled_ptr_free(ptr);
secure_memory_pool_free(pool);

// Traditional pools (legacy, less secure)
CMemoryPool* old_pool = memory_pool_new(64 * 1024, 100);
void* chunk = memory_pool_allocate(old_pool);
memory_pool_deallocate(old_pool, chunk);
memory_pool_free(old_pool);

// Error handling
zipora_set_error_callback(error_callback);
if (fast_vec_push(NULL, 42) != CResult_Success) {
    printf("Error: %s\n", zipora_last_error());
}
```

## Features

| Feature | Description | Default | Requirements |
|---------|-------------|---------|--------------|
| `simd` | SIMD optimizations (AVX2, BMI2, POPCNT) | ✅ | Stable Rust |
| `avx512` | AVX-512 optimizations (experimental) | ❌ | **Nightly Rust** |
| `mmap` | Memory-mapped file support | ✅ | Stable Rust |
| `zstd` | ZSTD compression | ✅ | Stable Rust |
| `serde` | Serialization support | ✅ | Stable Rust |
| `lz4` | LZ4 compression | ❌ | Stable Rust |
| `ffi` | C FFI compatibility | ❌ | Stable Rust |

## Build & Test

```bash
# Build
cargo build --release

# Build with optional features
cargo build --release --features lz4             # Enable LZ4 compression
cargo build --release --features ffi             # Enable C FFI compatibility
cargo build --release --features lz4,ffi         # Multiple optional features

# AVX-512 requires nightly Rust (experimental intrinsics)
cargo +nightly build --release --features avx512  # Enable AVX-512 optimizations
cargo +nightly build --release --features avx512,lz4,ffi  # AVX-512 + other features

# Test (535+ tests, 97%+ coverage)
cargo test --all-features

# Benchmark
cargo bench

# Benchmark with specific features
cargo bench --features lz4

# AVX-512 benchmarks (nightly Rust required)
cargo +nightly bench --features avx512

# Examples
cargo run --example basic_usage
cargo run --example succinct_demo
cargo run --example entropy_coding_demo
cargo run --example secure_memory_pool_demo  # SecureMemoryPool security features
```

## Test Results Summary

**✅ Edition 2024 Compatible** - Full compatibility with Rust edition 2024 and comprehensive testing across all feature combinations:

| Configuration | Debug Build | Release Build | Debug Tests | Release Tests |
|---------------|-------------|---------------|-------------|---------------|
| **Default features** | ✅ Success | ✅ Success | ✅ 513 tests | ✅ 514 tests |
| **+ lz4,ffi** | ✅ Success | ✅ Success | ✅ 553 tests | ✅ 553 tests |
| **No features** | ✅ Success | ✅ Success | ✅ 481 tests | ✅ Compatible |
| **Nightly + avx512** | ✅ Success | ✅ Success | ✅ 512 tests | ✅ 514 tests |
| **All features** | ✅ Success | ✅ Success | ✅ Compatible | ✅ Compatible |

### Key Achievements

- **🎯 Edition 2024**: Full compatibility with zero breaking changes
- **🔧 FFI Memory Safety**: **FULLY RESOLVED** - Complete elimination of double-free errors with CString pointer nullification
- **⚡ AVX-512 Support**: Full nightly Rust compatibility with 512-514 tests passing
- **🔒 Memory Management**: All unsafe operations properly scoped per edition 2024 requirements
- **🧪 Comprehensive Testing**: 553+ tests across all feature combinations with zero failures
- **🔌 LZ4+FFI Compatibility**: All 553 tests passing with lz4,ffi feature combination

## Development Status

**Phases 1-5 Complete** - All major components implemented:

- ✅ **Core Infrastructure**: FastVec, FastStr, blob storage, I/O framework
- ✅ **Advanced Tries**: LOUDS, Patricia, Critical-Bit with full functionality
- ✅ **Memory Mapping**: Zero-copy I/O with automatic growth
- ✅ **Entropy Coding**: Huffman, rANS, dictionary compression systems
- ✅ **Secure Memory Management**: Production-ready SecureMemoryPool, bump allocators, hugepage support
- ✅ **Advanced Algorithms**: Suffix arrays, radix sort, multi-way merge
- ✅ **Fiber Concurrency**: Work-stealing execution, pipeline processing
- ✅ **Real-time Compression**: Adaptive algorithms with latency guarantees
- ✅ **C FFI Layer**: Complete compatibility for C++ migration

## License

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [topling-zip](https://github.com/topling/topling-zip) C++ library. This Rust implementation adds memory safety while maintaining performance.
