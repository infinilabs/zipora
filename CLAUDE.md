# CLAUDE.md

## Core Principles
1. **Performance First**: Benchmark changes, exceed C++ performance
2. **Memory Safety**: Use SecureMemoryPool, avoid unsafe in public APIs
3. **Testing**: 97%+ coverage, all tests pass
4. **SIMD**: AVX2/BMI2/POPCNT, AVX-512 on nightly
5. **Production**: Zero compilation errors, robust error handling

## Commands
```bash
cargo build --release && cargo test --all-features && cargo bench
cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --check
```

## Completed Features ✅

### Dynamic SIMD Selection
- Runtime adaptive selection surpassing C++ compile-time approach
- Data-aware algorithm choice based on size, density, access patterns
- Continuous performance monitoring with EMA tracking
- 6-Tier SIMD framework integration (Tier 0-5)
- Micro-benchmarking with degradation detection (<10ns cache hit, <100ns miss)
- LRU-based selection caching, customizable thresholds
- Cross-platform: x86_64 (AVX-512/AVX2/BMI2/POPCNT) + ARM64 (NEON)

### Advanced Prefetching Strategies
- Sophisticated cache prefetching matching C++ implementation patterns
- Adaptive stride detection (Sequential, Strided, Random, PointerChasing)
- Bandwidth-aware and accuracy-based throttling
- Cross-platform: x86_64 (_mm_prefetch) + ARM64 (PRFM inline asm)
- Lookahead prefetching (PREFETCH_DISTANCE=8, +11% improvement)
- Zero monitoring overhead (fixed 29,000% overhead bug)
- Optimized APIs: rank1_optimized(), select1_optimized(), bulk variants

### Multi-Dimensional SIMD Rank/Select
- Template-based patterns with const generic dimensions (1-32)
- Vectorized bulk operations (4-8x speedup)
- Cross-dimensional set operations with AVX2
- Interleaved cache layout (20-30% cache improvement)
- Performance: 4-8x bulk rank, 6-12x bulk select

### Double Array Trie Improvements
- Memory efficiency: 139x → 58x overhead (57.6% improvement)
- Minimal initialization, lazy base allocation, compact base finding
- Critical fixes: terminal bit preservation, root state initialization
- Following C++ implementation's double_array_trie.hpp patterns

### Unified Architecture Transformation
- **ZiporaHashMap**: Single implementation replacing 6+ hash maps
- **ZiporaTrie**: Single implementation replacing 5+ tries
- Strategy-based configuration (HashStrategy, TrieStrategy, etc.)
- Clean module exports, backward-compatible APIs
- Version 2.0.0 with migration guide

### Advanced Multi-Way Merge
- Tournament tree with O(log k) complexity
- 64-byte cache-aligned layout, memory prefetching
- Advanced set operations with bit mask optimization (≤32 ways)
- SIMD-optimized (AVX2/BMI2) with runtime detection

### Advanced Radix Sort Variants
- LSD/MSD algorithms with adaptive hybrid selection
- String-specialized variants
- SIMD acceleration (AVX2/BMI2), parallel work-stealing
- 4-8x faster than comparison sorts

### Advanced Hash Map Ecosystem
- AdvancedHashMap: Robin Hood, chaining, Hopscotch
- CacheOptimizedHashMap: Cache-line aligned, NUMA-aware
- AdvancedStringArena: Offset-based with deduplication
- 64-byte alignment, x86_64 prefetch intrinsics

### Cache Optimization Infrastructure
- Cache-line alignment (64B x86_64, 128B ARM64)
- Hot/cold data separation with access pattern analysis
- Software prefetching (x86_64 _mm_prefetch, ARM64 __builtin_prefetch)
- NUMA-aware allocation with topology detection
- 5 access patterns: Sequential, Random, ReadHeavy, WriteHeavy, Mixed
- Performance: 2-3x memory access, 4-5x sequential, 20-40% NUMA gains

### Cache-Oblivious Algorithms
- Funnel sort: O(1 + N/B * log_{M/B}(N/B)) cache complexity
- AdaptiveAlgorithmSelector: cache-aware vs cache-oblivious
- Van Emde Boas layout with SIMD prefetching
- 2-4x SIMD speedup, automatic cache hierarchy adaptation

### SSE4.2 SIMD String Search
- PCMPESTRI-based hardware acceleration
- Hybrid strategy: pattern length optimization
- Core functions: sse42_strchr, strstr, multi_search, strcmp
- Performance: 2-4x char search, 2-8x pattern search, 3-6x strcmp

### SIMD Memory Operations Integration
- Complete 6-tier SIMD in SecureMemoryPool, LockFreeMemoryPool, MmapVec
- SIMD-accelerated zeroing (4-8x), verification (8-16x)
- Bulk allocation with lookahead (+20-30%)
- Lock-free safety guarantees
- 38 new tests, all performance targets met

### Algorithm Integration
- Radix sort: AVX-512 (16x), AVX2 (8x), BMI2, POPCNT
- SSE4.2 string search: PCMPESTRI operations
- SIMD entropy coding: AVX2+BMI2 Huffman
- Performance: 4-8x radix, 2-8x strings, 5-10x entropy coding

### Enhanced BMI2 Integration
- Systematic BMI2 across bit operations (BEXTR, BZHI, PDEP/PEXT)
- Hash function acceleration (2-5x)
- Compression algorithms (3-10x)
- String processing (2-8x)

### Cache Layout Optimization
- CacheOptimizedAllocator, CacheLayoutConfig
- Cross-platform cache detection (CPUID, /sys)
- HotColdSeparator with access tracking
- SIMD memory operations with prefetch integration
- >95% cache hit rates

### Core Infrastructure
- **Memory**: SecureMemoryPool, LockFreeMemoryPool, MmapVec
- **Concurrency**: Five-Level Management (NoLocking → FixedCapacity)
- **Sync**: Version-based tokens, FutexMutex, InstanceTls, AtomicExt
- **Containers**: 11 specialized (3-4x C++ performance)
- **Storage**: FastVec, IntVec (248+ MB/s bulk), ZipOffsetBlobStore
- **Cache**: LruMap, ConcurrentLruMap, LruPageCache

### Search & Algorithms
- **Rank/Select**: 14 variants (0.3-0.4 Gops/s with BMI2)
- **Tries**: Unified ZiporaTrie (PatriciaTrie, CritBitTrie, DoubleArray, NestedLouds)
- **Hash Maps**: Unified ZiporaHashMap
- **Sorting**: 5 specialized algorithms

### I/O & Serialization
- **Fiber I/O**: FiberAio, StreamBufferedReader, ZeroCopyReader
- **Serialization**: 8 components
- **String**: 3 optimized components

### Compression
- **PA-Zip**: SA-IS suffix arrays, BFS DFA cache, two-level pattern matching
- **Entropy Coding**: Contextual Huffman, 64-bit rANS, FSE with ZSTD
- **Binary Search**: Three-phase optimized search

### Performance Metrics
- **Rank/Select**: 0.3-0.4 Gops/s with BMI2 (2-3x select speedup)
- **Hardware**: PDEP/PEXT/TZCNT, hybrid strategies, prefetch hints
- **Memory**: 50-70% reduction, 96.9% space compression
- **Cache**: >95% hit rates, NUMA-aware, hot/cold separation
- **Radix Sort**: 4-8x faster, near-linear scaling to 8-16 cores
- **Cache-Oblivious**: Optimal complexity, 2-4x SIMD speedup
- **IntVec**: 248+ MB/s bulk construction
- **SIMD Memory**: 4-12x bulk ops, <100ns selection overhead
- **Safety**: Zero unsafe in public APIs
- **Tests**: 1,904+ passing, 97%+ coverage

## SIMD Framework (MANDATORY)

### 6-Tier Hardware Acceleration Architecture

**Hardware Tiers (implement in order):**
- **Tier 5**: AVX-512 (8x parallel, nightly) - `#[cfg(feature = "avx512")]`
- **Tier 4**: AVX2 (4x parallel, stable) - Default
- **Tier 3**: BMI2 (PDEP/PEXT) - Runtime detection
- **Tier 2**: POPCNT - Runtime detection
- **Tier 1**: ARM NEON - `#[cfg(target_arch = "aarch64")]`
- **Tier 0**: Scalar fallback - MANDATORY first

**Required patterns:**
```rust
// Runtime detection with graceful fallbacks
#[cfg(target_arch = "x86_64")]
fn accelerated_operation(data: &[u32]) -> u32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { avx2_implementation(data) }
    } else if is_x86_feature_detected!("sse2") {
        unsafe { sse2_implementation(data) }
    } else {
        scalar_fallback(data)
    }
}
```

**REQUIRED:**
1. Always provide scalar fallback
2. Runtime CPU feature detection (`is_x86_feature_detected!`)
3. Isolate unsafe to SIMD intrinsics only
4. Cross-platform support (x86_64 + ARM64)
5. Comprehensive testing

**Performance targets:**
- Rank/Select: 0.3-0.4 Gops/s with BMI2
- Radix Sort: 4-8x vs comparison sorts
- String Processing: 2-4x UTF-8 validation
- Compression: 5-10x bit manipulation

## Key Types

- **SIMD**: `SimdCapabilities`, `CpuFeatures`, `SimdOperations`, `AdaptiveSimdSelector`
- **Cache**: `CacheOptimizedAllocator`, `CacheLayoutConfig`, `HotColdSeparator`, `CacheAlignedVec`
- **Memory Pools**: `SecureMemoryPool`, `LockFreeMemoryPool`, `MmapVec`
- **Concurrency**: `AdaptiveFiveLevelPool`, `VersionManager`, `TokenManager`
- **Rank/Select**: `RankSelectInterleaved256`, `RankSelectMixed_IL_256`, `AdaptiveRankSelect`
- **Radix Sort**: `AdvancedRadixSort`, `AdvancedRadixSortConfig`, `SortingStrategy`
- **Tries**: `ZiporaTrie`, `ZiporaTrieConfig`
- **Hash Maps**: `ZiporaHashMap`, `AdvancedHashMap`, `CacheOptimizedHashMap`
- **String Search**: `SimdStringSearch`, `sse42_strchr`, `sse42_strstr`
- **I/O**: `FiberAio`, `StreamBufferedReader`, `ZeroCopyReader`
- **Entropy Coding**: `ContextualHuffmanEncoder`, `Rans64Encoder`, `FseEncoder`
- **PA-Zip**: `DictZipBlobStore`, `SuffixArrayDictionary`, `DfaCache`

## Features
- **Default**: `simd`, `mmap`, `zstd`, `serde`
- **Optional**: `lz4`, `ffi`, `avx512` (nightly)

## Patterns (MANDATORY)

### Dynamic SIMD Selection
```rust
use zipora::simd::{AdaptiveSimdSelector, Operation};
let selector = AdaptiveSimdSelector::global();
let impl_type = selector.select_optimal_impl(Operation::Rank, data.len(), Some(0.5));
selector.monitor_performance(Operation::Rank, elapsed, 1);
```

### Cache Optimization
```rust
use zipora::memory::{CacheOptimizedAllocator, CacheLayoutConfig};
let config = CacheLayoutConfig::sequential();
let allocator = CacheOptimizedAllocator::new(config);
```

### Cache-Oblivious Algorithms
```rust
use zipora::algorithms::{CacheObliviousSort, AdaptiveAlgorithmSelector};
let mut sorter = CacheObliviousSort::new(config);
sorter.sort(&mut data)?;
```

## Pattern Reference
- Memory: `SecureMemoryPool::new(config)` (src/memory.rs:45)
- Five-Level: `AdaptiveFiveLevelPool::new(config)` (src/memory/five_level_pool.rs:1200)
- Tries: `ZiporaTrie::with_config()` (src/fsa/mod.rs)
- Hash Maps: `ZiporaHashMap::with_config()` (src/hash_map/mod.rs)
- Rank/Select: `RankSelectMixed_IL_256::new()` (src/succinct/rank_select/mixed_impl.rs)
- Radix Sort: `AdvancedU32RadixSort::new()` (src/algorithms/radix_sort.rs)
- Cache-Oblivious: `CacheObliviousSort::new()` (src/algorithms/cache_oblivious.rs)
- String Search: `SimdStringSearch::new()` (src/string/simd_search.rs)
- PA-Zip: `DictZipBlobStore::new(config)` (src/compression/dict_zip/blob_store.rs)

---
**Status**: Production-ready SIMD acceleration framework
**Performance**: 4-12x memory ops, 0.3-0.4 Gops/s rank/select, 4-8x radix sort, 2-8x string processing
**Cross-Platform**: x86_64 (AVX-512/AVX2/BMI2/POPCNT) + ARM64 (NEON) + scalar fallbacks
**Tests**: 1,904+ passing (100% pass rate)
**Safety**: Zero unsafe in public APIs (MANDATORY)
