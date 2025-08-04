# Zipora

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Features

- **🚀 High Performance**: Zero-copy operations, SIMD optimizations, cache-friendly layouts
- **🛡️ Memory Safety**: Eliminates segfaults, buffer overflows, use-after-free bugs
- **🧠 Advanced Memory Management**: Tiered allocators, memory pools, hugepage support
- **🗜️ Compression Framework**: Huffman, rANS, dictionary-based, and hybrid compression
- **🌲 Advanced Tries**: LOUDS, Critical-Bit, and Patricia tries
- **💾 Blob Storage**: Memory-mapped and compressed storage systems
- **⚡ Fiber Concurrency**: High-performance async/await with work-stealing
- **🔄 Real-time Compression**: Adaptive algorithms with strict latency guarantees
- **🔌 C FFI Support**: Complete C API for migration from C++

## Quick Start

```toml
[dependencies]
zipora = "1.0.2"
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

### Memory Management

```rust
use zipora::{MemoryPool, PoolConfig, BumpAllocator, PooledVec};

// Memory pools for frequent allocations
let config = PoolConfig::new(64 * 1024, 100, 8);
let pool = MemoryPool::new(config).unwrap();
let chunk = pool.allocate().unwrap();

// Bump allocator for sequential allocation
let bump = BumpAllocator::new(1024 * 1024).unwrap();
let ptr = bump.alloc::<u64>().unwrap();

// Pooled containers
let mut pooled_vec = PooledVec::<i32>::new().unwrap();
pooled_vec.push(42).unwrap();

// Linux hugepage support
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

// Automatic algorithm selection
let algorithm = CompressorFactory::select_best(&requirements, data);
let compressor = CompressorFactory::create(algorithm, Some(training_data)).unwrap();
```

## Performance

Current performance on Intel i7-10700K:

| Operation | Performance | vs std::Vec | vs C++ |
|-----------|-------------|-------------|--------|
| FastVec push 10k | 6.78µs | +48% faster | +20% faster |
| Memory pool alloc | ~15ns | +90% faster | +90% faster |
| Radix sort 1M u32s | ~45ms | +60% faster | +40% faster |
| Suffix array build | O(n) | N/A | Linear vs O(n log n) |
| Fiber spawn | ~5µs | N/A | New capability |

## C FFI Migration

```toml
[dependencies]
zipora = { version = "1.0.2", features = ["ffi"] }
```

```c
#include <zipora.h>

// Vector operations
CFastVec* vec = fast_vec_new();
fast_vec_push(vec, 42);
printf("Length: %zu\n", fast_vec_len(vec));
fast_vec_free(vec);

// Memory pools
CMemoryPool* pool = memory_pool_new(64 * 1024, 100);
void* chunk = memory_pool_allocate(pool);
memory_pool_deallocate(pool, chunk);
memory_pool_free(pool);

// Error handling
zipora_set_error_callback(error_callback);
if (fast_vec_push(NULL, 42) != CResult_Success) {
    printf("Error: %s\n", zipora_last_error());
}
```

## Features

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | SIMD optimizations | ✅ |
| `mmap` | Memory-mapped file support | ✅ |
| `zstd` | ZSTD compression | ✅ |
| `serde` | Serialization support | ✅ |
| `lz4` | LZ4 compression | ❌ |
| `ffi` | C FFI compatibility | ❌ |

## Build & Test

```bash
# Build
cargo build --release

# Test (398 tests, 97%+ coverage)
cargo test --all-features

# Benchmark
cargo bench

# Examples
cargo run --example basic_usage
cargo run --example succinct_demo
cargo run --example entropy_coding_demo
```

## Development Status

**Phases 1-5 Complete** - All major components implemented:

- ✅ **Core Infrastructure**: FastVec, FastStr, blob storage, I/O framework
- ✅ **Advanced Tries**: LOUDS, Patricia, Critical-Bit with full functionality
- ✅ **Memory Mapping**: Zero-copy I/O with automatic growth
- ✅ **Entropy Coding**: Huffman, rANS, dictionary compression systems
- ✅ **Memory Management**: Pools, bump allocators, hugepage support
- ✅ **Advanced Algorithms**: Suffix arrays, radix sort, multi-way merge
- ✅ **Fiber Concurrency**: Work-stealing execution, pipeline processing
- ✅ **Real-time Compression**: Adaptive algorithms with latency guarantees
- ✅ **C FFI Layer**: Complete compatibility for C++ migration

## License

Licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [topling-zip](https://github.com/topling/topling-zip) C++ library. This Rust implementation adds memory safety while maintaining performance.
