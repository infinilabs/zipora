# Algorithms & Data Structures

Zipora provides sophisticated algorithms optimized for performance-critical applications.

## Cache-Oblivious Algorithms

Cache-oblivious algorithms achieve optimal performance across different cache hierarchies without explicit knowledge of cache parameters.

### Key Features

- **Cache-Oblivious Sorting**: Funnel sort with optimal O(1 + N/B * log_{M/B}(N/B)) cache complexity
- **Adaptive Algorithm Selection**: Intelligent choice between cache-aware and cache-oblivious strategies
- **Van Emde Boas Layout**: Cache-optimal data structure layouts with SIMD prefetching
- **SIMD Integration**: Full integration with Zipora's 6-tier SIMD framework
- **Recursive Subdivision**: Optimal cache utilization through divide-and-conquer

### Algorithm Selection Strategy

- **Small data** (< L1 cache): Cache-aware optimized algorithms with insertion sort
- **Medium data** (L1-L3 cache): Cache-oblivious funnel sort for optimal hierarchy utilization
- **Large data** (> L3 cache): Hybrid approach combining cache-oblivious merge with external sorting
- **String data**: Specialized cache-oblivious string algorithms
- **Numeric data**: SIMD-accelerated cache-oblivious variants

```rust
use zipora::algorithms::{CacheObliviousSort, CacheObliviousConfig, AdaptiveAlgorithmSelector, VanEmdeBoas};

// Automatic cache-oblivious sorting with adaptive strategy selection
let mut sorter = CacheObliviousSort::new();
let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
sorter.sort(&mut data).unwrap();
assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

// Custom configuration with SIMD and parallel processing
let config = CacheObliviousConfig {
    use_simd: true,
    use_parallel: true,
    small_threshold: 512,
    ..Default::default()
};
let mut custom_sorter = CacheObliviousSort::with_config(config);

// Adaptive algorithm selector for strategic decision making
let selector = AdaptiveAlgorithmSelector::new(&config);
let strategy = selector.select_strategy(data.len(), &config.cache_hierarchy);
println!("Selected strategy: {:?}", strategy); // CacheAware, CacheOblivious, or Hybrid

// Van Emde Boas layout for cache-optimal data structures
let cache_hierarchy = detect_cache_hierarchy();
let veb_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
let veb = VanEmdeBoas::new(veb_data, cache_hierarchy);
let element = veb.get(3); // Cache-optimal access with SIMD prefetching
```

## Advanced Radix Sort

Zipora provides multiple radix sort implementations with SIMD optimizations and intelligent adaptive selection.

```rust
use zipora::{RadixSort, RadixSortConfig, AdvancedRadixSort};

// Advanced radix sort with intelligent algorithm selection
let mut data = vec![5_000_000u32, 2_500_000u32, 8_750_000u32, 1_250_000u32];
let config = RadixSortConfig::adaptive_optimized();
let mut advanced_sorter = AdvancedRadixSort::with_config(config).unwrap();
advanced_sorter.sort_adaptive(&mut data).unwrap();
println!("Strategy: {:?}", advanced_sorter.stats().selected_strategy);

// Legacy high-performance radix sort
let mut small_data = vec![5u32, 2, 8, 1, 9];
let mut sorter = RadixSort::new();
sorter.sort_u32(&mut small_data).unwrap();
```

### Radix Sort Strategies

- **LSD (Least Significant Digit)**: Best for uniform key distribution
- **MSD (Most Significant Digit)**: Best for variable-length keys
- **Insertion Sort**: For very small arrays (<16 elements)
- **Tim Sort**: For nearly sorted data
- **Adaptive Hybrid**: Automatic selection based on data characteristics

## Multi-Way Merge Algorithms

```rust
use zipora::{EnhancedLoserTree, LoserTreeConfig, MultiWayMerge, SetOperations};

// Enhanced Tournament Tree with O(log k) Complexity
let config = LoserTreeConfig {
    initial_capacity: 64,
    use_secure_memory: true,
    stable_sort: true,
    cache_optimized: true,
    use_simd: true,
    prefetch_distance: 2,
    alignment: 64,
};
let mut enhanced_tree = EnhancedLoserTree::new(config);

// Add sorted input streams with true O(log k) complexity
enhanced_tree.add_way(vec![1, 4, 7, 10].into_iter()).unwrap();
enhanced_tree.add_way(vec![2, 5, 8, 11].into_iter()).unwrap();
enhanced_tree.add_way(vec![3, 6, 9, 12].into_iter()).unwrap();

// Initialize with cache-friendly layout and prefetching
enhanced_tree.initialize().unwrap();

// Merge with O(log k) winner selection
let merged = enhanced_tree.merge_to_vec().unwrap();
assert_eq!(merged, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

// Multi-way merge with vectorized sources
let sources = vec![
    VectorSource::new(vec![1, 4, 7]),
    VectorSource::new(vec![2, 5, 8]),
];
let mut merger = MultiWayMerge::new();
let result = merger.merge(sources).unwrap();
```

## Set Operations

Complete implementation following C++ reference from topling-zip/set_op.hpp.

```rust
use zipora::algorithms::{
    multiset_intersection, multiset_fast_intersection, multiset_union, multiset_difference,
    set_intersection, set_union, set_difference, set_unique, set_unique_default
};

// Multiset Intersection - Preserves Duplicates
let a = vec![1, 2, 2, 3, 4, 5];
let b = vec![2, 2, 3, 6, 7];
let result = multiset_intersection(&a, &b, |x, y| x.cmp(y));
assert_eq!(result, vec![2, 2, 3]);

// Adaptive Fast Intersection - Auto-Selects Best Algorithm
let small = vec![1, 2, 3];
let large: Vec<i32> = (1..=100).collect();
let result = multiset_fast_intersection(&small, &large, |x, y| x.cmp(y), 32);
assert_eq!(result, vec![1, 2, 3]);

// Multiset Union - Combines Both Sequences
let a = vec![1, 2, 3, 4];
let b = vec![3, 4, 5, 6];
let result = multiset_union(&a, &b, |x, y| x.cmp(y));
assert_eq!(result, vec![1, 2, 3, 3, 4, 4, 5, 6]);

// Unique Set Operations
let a = vec![1, 2, 2, 3, 4];
let b = vec![2, 2, 3, 5];

let intersection = set_intersection(&a, &b, |x, y| x.cmp(y));
assert_eq!(intersection, vec![2, 3]); // No duplicates

let union = set_union(&a, &b, |x, y| x.cmp(y));
assert_eq!(union, vec![1, 2, 3, 4, 5]); // No duplicates

// In-Place Deduplication
let mut data = vec![1, 1, 2, 2, 2, 3, 4, 4, 5];
let new_len = set_unique_default(&mut data);
data.truncate(new_len);
assert_eq!(data, vec![1, 2, 3, 4, 5]);
```

### Set Operation Performance

| Operation | Complexity | Best Use Case |
|-----------|-----------|---------------|
| `multiset_intersection` | O(n + m) | Similar-sized sets |
| `multiset_1small_intersection` | O(n * log(m)) | Small first set, large second |
| `multiset_fast_intersection` | **Adaptive** | **Automatic selection** |
| `multiset_union` | O(n + m) | Merge sorted sequences |
| `multiset_difference` | O(n + m) | Set subtraction |
| `set_unique` | O(n) | In-place deduplication |

## Suffix Array Construction

Zipora provides 5 sophisticated suffix array algorithms with adaptive selection.

```rust
use zipora::{SuffixArray, SuffixArrayConfig, SuffixArrayAlgorithm, EnhancedSuffixArray};

let text = b"banana";

// Adaptive algorithm selection (recommended)
let sa = SuffixArray::new(text).unwrap();
let (start, count) = sa.search(text, b"an");
println!("Found 'an' at {} occurrences", count);

// Manual algorithm selection
let config = SuffixArrayConfig {
    algorithm: SuffixArrayAlgorithm::SAIS,     // SA-IS: Linear-time
    use_parallel: true,
    parallel_threshold: 100_000,
    compute_lcp: false,
    ..Default::default()
};
let sa_sais = SuffixArray::with_config(text, &config).unwrap();

// Enhanced suffix array with LCP computation
let enhanced_sa = EnhancedSuffixArray::with_lcp(text).unwrap();
let lcp = enhanced_sa.lcp_array().unwrap();
println!("LCP at position 0: {:?}", lcp.lcp_at(0));

// Suffix array with BWT
let enhanced_sa_bwt = EnhancedSuffixArray::with_bwt(text).unwrap();
if let Some(bwt) = enhanced_sa_bwt.bwt() {
    println!("BWT: {:?}", String::from_utf8_lossy(bwt));
}

// Data characteristics analysis
let characteristics = SuffixArray::analyze_text_characteristics(text);
println!("Text length: {}, Alphabet size: {}",
         characteristics.text_length, characteristics.alphabet_size);
```

### Suffix Array Algorithm Selection Guide

| Algorithm | Time | Best Use Case | Memory |
|-----------|------|---------------|--------|
| **Adaptive** | Varies | **General use (recommended)** | Optimal |
| **SA-IS** | O(n) | Small alphabets | ~8n bytes |
| **DC3** | O(n) | Small inputs, cache locality | ~12n bytes |
| **DivSufSort** | O(n log n) | Large inputs | ~8n bytes |
| **Larsson-Sadakane** | O(n log n) | Highly repetitive data | ~12n bytes |

## Rank/Select Operations

```rust
use zipora::succinct::rank_select::{
    RankSelectInterleaved256, AdaptiveRankSelect, BitVector,
    RankSelectMixed, RankSelectMixed_IL_256, MultiDimRankSelect
};

// Standard rank/select
let mut bv = BitVector::new();
for i in 0..1000 {
    bv.push(i % 3 == 0).unwrap();
}

let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();
let rank = rs.rank1(500);
let selected = rs.select1(10).unwrap();

// Adaptive rank/select - automatic optimization
let adaptive_rs = AdaptiveRankSelect::new(bv).unwrap();
println!("Implementation: {}", adaptive_rs.implementation_name());

// Multi-Dimensional SIMD Rank/Select
let mut dimensions = vec![];
for _ in 0..4 {
    let mut dim_bv = BitVector::new();
    for i in 0..1000 {
        dim_bv.push(i % 3 == 0).unwrap();
    }
    dimensions.push(dim_bv);
}

let multi_rs: MultiDimRankSelect<4> = MultiDimRankSelect::new(dimensions).unwrap();

// Vectorized bulk rank across all dimensions (4-8x faster with SIMD)
let positions = [100, 200, 300, 400];
let ranks = multi_rs.bulk_rank_multidim(&positions);

// Cross-dimensional intersection (AVX2-optimized)
let intersection = multi_rs.intersect_dimensions(0, 1).unwrap();
```

## Performance Characteristics

- **Cache Complexity**: O(1 + N/B * log_{M/B}(N/B)) optimal across all cache levels
- **SIMD Acceleration**: 2-4x speedup with AVX2/BMI2
- **Parallel Scaling**: Near-linear scaling up to 8-16 cores
- **Memory Efficiency**: Minimal overhead with in-place algorithms
- **Rank/Select**: 0.3-0.4 Gops/s with BMI2
