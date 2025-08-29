# Cache Locality Optimizations for Hash Maps

## Overview

This document describes the comprehensive cache locality optimizations implemented for the Zipora hash map implementations. These optimizations improve performance through better memory access patterns, data structure layouts, and prefetching strategies.

## Implemented Optimizations

### 1. Cache-Line Aligned Data Structures

- **CacheAligned<T>**: Wrapper type ensuring 64-byte cache line alignment
- **CacheOptimizedBucket**: Bucket structure optimized for cache line size
  - Metadata packed in first 8 bytes
  - Hash values for fast comparison (7 entries fit in cache line)
  - Overflow pointer for chaining

### 2. Software Prefetching

- **Prefetcher Utility**: Hardware-accelerated prefetching with multiple hint levels
  - T0: Prefetch to all cache levels (L1, L2, L3)
  - T1: Prefetch to L2 and L3 only
  - T2: Prefetch to L3 only
  - NTA: Non-temporal (bypass cache)
- **Adaptive Prefetch Distance**: Adjusts based on access patterns
  - Sequential: 2x distance
  - Strided: Match stride
  - Random: Minimal prefetching

### 3. NUMA-Aware Memory Allocation

- **NumaAllocator**: Node-local memory allocation
  - Detects NUMA topology
  - Allocates on preferred node
  - Tracks per-node statistics
- **Cache-line aligned allocation**: Ensures proper alignment for all allocations

### 4. Cache Layout Optimization

- **CacheLayoutOptimizer**: Determines optimal configuration based on working set
  - L1 optimization: 4 entries per bucket, 0.5 load factor
  - L2 optimization: 8 entries per bucket, 0.65 load factor
  - L3 optimization: 16 entries per bucket, 0.75 load factor
  - Memory optimization: Higher load factors for large datasets

### 5. Hot/Cold Data Separation

- **HotColdSeparator**: Separates frequently and infrequently accessed data
  - Hot data stored with cache-line alignment
  - Cold data stored compactly
  - Dynamic rebalancing based on access patterns
  - Configurable hot ratio (default 20%)

### 6. Access Pattern Analysis

- **AccessPatternAnalyzer**: Detects and adapts to access patterns
  - Sequential detection with >70% confidence
  - Strided pattern recognition
  - Random access detection
  - Temporal locality analysis
- **Adaptive Optimization**: Adjusts strategy based on detected patterns

### 7. Cache-Conscious Resizing

- **CacheConsciousResizer**: Incremental resizing to minimize cache thrashing
  - Chunk-based resizing (L3 cache size chunks)
  - Copy-on-write for large tables
  - Minimizes cache invalidations

### 8. Cache Performance Monitoring

- **CacheMetrics**: Comprehensive performance tracking
  - L1/L2/L3 hit/miss rates
  - Prefetch operation count
  - Memory stalls and false sharing detection
  - Cache invalidation tracking
  - Bandwidth estimation

## Performance Characteristics

### Memory Layout
- **Cache Line Size**: 64 bytes
- **Bucket Size**: 7 entries per bucket (fits in 1-2 cache lines)
- **Alignment**: All critical structures are cache-line aligned
- **Padding**: Automatic padding to prevent false sharing

### Access Patterns
- **Random Access**: Minimal prefetching, optimized probe sequences
- **Sequential Access**: Aggressive prefetching, 2x distance
- **Strided Access**: Pattern-matched prefetching
- **Temporal Locality**: Hot/cold separation enabled

### Load Factors
- **L1-optimized**: 50% load factor for minimal collisions
- **L2-optimized**: 65% load factor for balance
- **L3-optimized**: 75% standard load factor
- **Memory-optimized**: 85% for large datasets

## CacheOptimizedHashMap API

```rust
use zipora::hash_map::CacheOptimizedHashMap;

// Create a cache-optimized hash map
let mut map = CacheOptimizedHashMap::new();

// Enable hot/cold separation (20% hot ratio)
map.enable_hot_cold_separation(0.2);

// Enable adaptive optimization
map.set_adaptive_mode(true);

// Insert and access data
map.insert("key", "value").unwrap();
let value = map.get("key");

// Get cache performance metrics
let metrics = map.cache_metrics();
println!("Cache hit ratio: {:.2}%", metrics.hit_ratio() * 100.0);
println!("Estimated bandwidth: {:.2} GB", metrics.estimated_bandwidth_gb());

// Optimize for detected access pattern
map.optimize_for_access_pattern();

// Rebalance hot/cold data
map.rebalance_hot_cold();
```

## Benchmarking

Run the cache locality benchmarks:

```bash
cargo bench --bench cache_locality_bench
```

The benchmarks measure:
- Random vs sequential access patterns
- Cache line operations
- Hot/cold separation effectiveness
- Resize performance
- Cache metrics overhead
- Strided access patterns

## Integration with Existing Hash Maps

The cache optimizations can be integrated with existing hash map implementations:

1. **GoldHashMap**: Already has basic cache optimizations
2. **GoldenRatioHashMap**: Can benefit from prefetching
3. **StringOptimizedHashMap**: String-specific cache optimizations
4. **SmallHashMap**: Inline storage already cache-friendly
5. **AdvancedHashMap**: Collision resolution with cache awareness

## Best Practices

1. **Enable adaptive mode** for workloads with changing patterns
2. **Use hot/cold separation** for skewed access distributions
3. **Monitor cache metrics** to identify optimization opportunities
4. **Choose appropriate initial capacity** to minimize resizing
5. **Consider NUMA topology** for multi-socket systems

## Future Improvements

1. **Hardware Performance Counters**: Direct CPU cache statistics
2. **Huge Pages Support**: Reduce TLB misses
3. **Cache Partitioning**: Intel CAT support
4. **Vectorized Operations**: AVX-512 for bulk operations
5. **Cache-Oblivious Algorithms**: Automatic adaptation to cache hierarchy