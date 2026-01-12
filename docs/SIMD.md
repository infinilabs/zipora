# SIMD Framework

Zipora implements a 6-tier SIMD framework with runtime detection and adaptive selection.

## 6-Tier Architecture

| Tier | Instruction Set | Description |
|------|-----------------|-------------|
| **Tier 5** | AVX-512 | 512-bit operations (nightly only) |
| **Tier 4** | AVX2 | 256-bit operations (default) |
| **Tier 3** | BMI2 | PDEP/PEXT bit manipulation |
| **Tier 2** | POPCNT | Hardware population count |
| **Tier 1** | ARM NEON | ARM SIMD (cross-platform) |
| **Tier 0** | Scalar | Fallback (MANDATORY) |

## SIMD Dispatch Macros

Use the provided macros to reduce code duplication:

```rust
use zipora::{simd_dispatch, simd_select, simd_feature_check, simd_available};

// Multi-tier dispatch with automatic fallback
fn hash_bytes(data: &[u8]) -> u64 {
    simd_dispatch!(
        avx2 => unsafe { hash_avx2(data) },
        sse2 => unsafe { hash_sse2(data) },
        _ => hash_scalar(data)
    )
}

// Expression selection (no return)
let result = simd_select!(
    avx2 => compute_avx2(data),
    _ => compute_scalar(data)
);

// Single feature check
fn fast_popcount(data: &[u64]) -> u64 {
    simd_feature_check!("popcnt", unsafe { popcount_hw(data) }, popcount_scalar(data))
}

// Check feature availability
if simd_available!("avx2") {
    println!("AVX2 available!");
}
```

### Available Macros

Located in `src/simd/macros.rs`:

- **`simd_dispatch!`** - Multi-tier dispatch (avx512, avx2, avx2_bmi2, sse42, sse2, bmi2, popcnt)
- **`simd_select!`** - Expression selection (no return statement)
- **`simd_feature_check!`** - Single/dual feature check with fallback
- **`simd_available!`** - Check if feature is available (returns bool)

## Runtime Detection

```rust
use zipora::simd::{CpuFeatures, SimdCapabilities, get_cpu_features};

// Detect CPU features at runtime
let features = get_cpu_features();
println!("AVX2: {}", features.has_avx2);
println!("AVX-512: {}", features.has_avx512f);
println!("BMI2: {}", features.has_bmi2);
println!("POPCNT: {}", features.has_popcnt);
println!("ARM NEON: {}", features.has_neon);

// Get SIMD capabilities
let caps = SimdCapabilities::detect();
println!("Optimization tier: {}", caps.optimization_tier);
println!("Max vector width: {} bits", caps.max_vector_width);
```

## Adaptive SIMD Selection

```rust
use zipora::simd::{AdaptiveSimdSelector, Operation};

// Get global selector (cached, lock-free)
let selector = AdaptiveSimdSelector::global();

// Select optimal implementation based on operation and data size
let impl_type = selector.select_optimal_impl(
    Operation::Rank,
    data.len(),
    Some(0.5) // density hint
);

// Use selected implementation
match impl_type {
    SimdImpl::Avx2 => { /* AVX2 path */ }
    SimdImpl::Bmi2 => { /* BMI2 path */ }
    SimdImpl::Scalar => { /* Scalar fallback */ }
    _ => { /* Other tiers */ }
}

// Micro-benchmarking for data-driven selection
let benchmark_result = selector.benchmark_operation(Operation::Select, &data);
println!("Best implementation: {:?}", benchmark_result.best_impl);
println!("Speedup vs scalar: {:.2}x", benchmark_result.speedup);
```

## SIMD Memory Operations

```rust
use zipora::simd::memory::{
    simd_memcpy, simd_memcmp, simd_memset,
    simd_prefetch, PrefetchHint
};

// SIMD-accelerated memory copy (2-3x faster for small, 1.5-2x for medium)
simd_memcpy(dst, src, len);

// SIMD memory comparison
let equal = simd_memcmp(a, b, len);

// SIMD memory set
simd_memset(dst, value, len);

// Software prefetching
simd_prefetch(ptr, PrefetchHint::T0); // L1 cache
simd_prefetch(ptr, PrefetchHint::T1); // L2 cache
simd_prefetch(ptr, PrefetchHint::T2); // L3 cache
simd_prefetch(ptr, PrefetchHint::NTA); // Non-temporal
```

## BMI2 Operations

```rust
use zipora::simd::bmi2::{Bmi2BitOps, Bmi2BlockOps, Bmi2Capabilities};

// Check BMI2 capabilities
let caps = Bmi2Capabilities::get();
println!("BMI2 tier: {}", caps.optimization_tier);

// Ultra-fast select with PDEP/PEXT (5-10x speedup)
let word = 0b1010101010101010u64;
let position = Bmi2BitOps::select1_ultra_fast(word, 3);

// Bulk operations with hardware acceleration
let words = vec![0xAAAAAAAAAAAAAAAAu64; 1000];
let positions = (0..100).step_by(10).collect::<Vec<_>>();
let bulk_ranks = Bmi2BlockOps::bulk_rank1(&words, &positions);

// Parallel bit deposit/extract
let result = Bmi2BitOps::pdep(source, mask);
let extracted = Bmi2BitOps::pext(source, mask);
```

## Multi-Dimensional SIMD

```rust
use zipora::simd::multidim::MultiDimRankSelect;

// Create 4-dimensional rank/select structure
let multi_rs: MultiDimRankSelect<4> = MultiDimRankSelect::new(dimensions).unwrap();

// Vectorized bulk rank across all dimensions (4-8x faster with SIMD)
let positions = [100, 200, 300, 400];
let ranks = multi_rs.bulk_rank_multidim(&positions);

// Cross-dimensional intersection (AVX2-optimized, 4-8x speedup)
let intersection = multi_rs.intersect_dimensions(0, 1).unwrap();

// Cross-dimensional union
let union = multi_rs.union_dimensions(&[0, 1, 2]).unwrap();
```

## AVX-512 Support (Nightly)

```rust
#[cfg(feature = "avx512")]
use zipora::simd::avx512::{Avx512Ops, Avx512MaskOps};

// AVX-512 requires nightly Rust
#[cfg(feature = "avx512")]
{
    // 512-bit operations
    let result = Avx512Ops::process_512bit(data);

    // Mask operations
    let mask = Avx512MaskOps::compare_eq(a, b);
}
```

## Performance Characteristics

| Operation | AVX2 Speedup | BMI2 Speedup | Best Tier |
|-----------|--------------|--------------|-----------|
| **Rank** | 2-4x | 5-10x | BMI2 |
| **Select** | 2-4x | 5-10x | BMI2 |
| **Memcpy (small)** | 2-3x | N/A | AVX2 |
| **Memcpy (large)** | 1.5-2x | N/A | AVX2 |
| **Histogram** | 4-8x | N/A | AVX2 |
| **String compare** | 4-8x | N/A | AVX2 |
| **Population count** | 2-4x | N/A | POPCNT |

## Performance Targets

- **Rank/Select**: 0.3-0.4 Gops/s with BMI2
- **SIMD Memory**: 4-12x bulk operations
- **Histogram**: 4-8x faster frequency counting
- **Cache Hit**: >95% with prefetching

## Cross-Platform Support

```rust
use zipora::simd::CrossPlatform;

// Automatic platform detection
let platform = CrossPlatform::detect();

match platform {
    CrossPlatform::X86_64 { avx2, avx512, bmi2, .. } => {
        println!("x86_64 with AVX2={}, AVX-512={}, BMI2={}", avx2, avx512, bmi2);
    }
    CrossPlatform::Aarch64 { neon, sve, .. } => {
        println!("AArch64 with NEON={}, SVE={}", neon, sve);
    }
    CrossPlatform::Generic => {
        println!("Generic platform, using scalar fallback");
    }
}
```

## Best Practices

1. **Always provide scalar fallback** - Tier 0 is mandatory
2. **Use dispatch macros** - Reduces code duplication
3. **Benchmark on target hardware** - SIMD performance varies
4. **Consider data size** - SIMD overhead may not pay off for small data
5. **Use adaptive selection** - Let the framework choose optimal tier
6. **Enable BMI2 for rank/select** - Provides best speedup (5-10x)
