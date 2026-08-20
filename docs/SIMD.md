# SIMD Framework

Zipora implements a 6-tier SIMD framework with runtime detection and adaptive selection.
This document covers the dispatch macros, CPU feature detection, SIMD memory operations,
multi-dimensional rank/select, and BMI2 acceleration.

## 6-Tier Architecture

| Tier | Instruction Set | Description |
|------|-----------------|-------------|
| **Tier 5** | AVX-512 | 512-bit operations (avx512 feature) |
| **Tier 4** | AVX2 | 256-bit operations (default) |
| **Tier 3** | BMI2 | PDEP/PEXT bit manipulation |
| **Tier 2** | POPCNT | Hardware population count |
| **Tier 1** | ARM NEON | ARM SIMD (cross-platform) |
| **Tier 0** | Scalar | Fallback (MANDATORY) |

Not every module uses all six tiers — each picks the subset that helps its workload
(see the per-module notes below).

## SIMD Dispatch Macros

Located in `src/simd/macros.rs`, exported at the crate root via `#[macro_export]`:

- **`simd_dispatch!`** — Multi-tier dispatch with automatic fallback chain. Tier keywords:
  `avx512`, `avx2`, `avx2_bmi2`, `sse42`, `sse42_bmi2`, `sse2`, `bmi2`, `popcnt`, `neon`,
  plus the mandatory `_ =>` scalar arm.
- **`simd_feature_check!`** — Single or dual feature check with fallback.
- **`simd_select!`** — Expression selection (evaluates to a value, no `return`).
- **`simd_available!`** — Check feature availability, returns `bool` (single or dual feature).

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

// Expression selection (evaluates in place, no return)
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

**Footgun**: the tier arms of `simd_dispatch!` and `simd_feature_check!` expand to
`return $expr;` — they return from the *enclosing function*, not just the macro
expression. The macro must be the tail expression of a function whose return type
matches the arms. Use `simd_select!` (or wrap the dispatch in a helper function) when
you need the result mid-computation.

## Runtime Detection

`get_cpu_features()` (crate root) returns a cached `&'static CpuFeatures`
(`zipora::system::CpuFeatures`) with per-feature booleans, cache sizes, and core counts.
`RuntimeCpuFeatures` (crate root) is the detection driver behind it.

```rust
use zipora::get_cpu_features;

let features = get_cpu_features();
println!("AVX2: {}", features.has_avx2);
println!("AVX-512F: {}", features.has_avx512f);
println!("BMI2: {}", features.has_bmi2);
println!("POPCNT: {}", features.has_popcnt);
println!("ARM NEON: {}", features.has_neon);
println!("L2 cache: {} bytes", features.l2_cache_size);
```

`SimdCapabilities` (crate root, from `succinct::rank_select::simd`) maps the raw features
to an optimization strategy:

```rust
use zipora::SimdCapabilities;

let caps = SimdCapabilities::detect(); // or SimdCapabilities::get() for the cached global
println!("Optimization tier: {}", caps.optimization_tier); // 0=scalar .. 5=AVX-512
println!("Bulk chunk size: {}", caps.chunk_size);
println!("Use prefetch: {}", caps.use_prefetch);
// caps.cpu_features: &'static CpuFeatures
```

## Adaptive SIMD Selection

`AdaptiveSimdSelector` (`src/simd/adaptive.rs`) selects an implementation per operation
based on hardware tier, data size, and optional density, with cached decisions:

```rust
use zipora::simd::{AdaptiveSimdSelector, Operation, SimdImpl};

// Global singleton (cached, runs startup benchmarks once by default)
let selector = AdaptiveSimdSelector::global();

// Select optimal implementation for operation + data characteristics
let impl_type = selector.select_optimal_impl(
    Operation::Rank,
    data.len(),
    Some(0.5), // optional density hint in [0, 1]
);

match impl_type {
    SimdImpl::Avx512 => { /* AVX-512 path */ }
    SimdImpl::Avx2 => { /* AVX2 path */ }
    SimdImpl::Bmi2 => { /* BMI2 path */ }
    SimdImpl::Scalar => { /* Scalar fallback */ }
    _ => { /* Sse42, Sse2, Neon */ }
}
```

## SIMD Memory Operations

`SimdMemOps` (`src/memory/simd_ops.rs`, re-exported at `zipora::memory`) provides
SIMD-accelerated copy, compare, search, and fill over byte slices. All public APIs are
safe wrappers; bounds, overlap, and alignment are validated before any unsafe code runs.

**Tier note**: this module dispatches across **AVX-512 → AVX2 → SSE2 → Scalar** only
(BMI2/POPCNT are not memcpy tiers; on ARM, NEON is used for prefetch hints only, with
scalar `ptr::copy_nonoverlapping`-based fallbacks for the data paths). The AVX-512 paths
require the `avx512` feature.

### Instance API

```rust
use zipora::memory::{SimdMemOps, PrefetchHint};

let ops = SimdMemOps::new(); // or SimdMemOps::with_cache_config(config)

// Core operations
ops.copy_nonoverlapping(&src, &mut dst)?;   // rejects overlap and size mismatch
ops.copy_aligned(&src, &mut dst)?;          // requires 64-byte-aligned src/dst
let ord: i32 = ops.compare(&a, &b);         // memcmp-style ordering
let pos: Option<usize> = ops.find_byte(&haystack, b'x');
ops.fill(&mut buf, 0xFF);

// Cache-optimized operations
ops.prefetch(addr, PrefetchHint::T0);       // addr: *const u8; T0/T1/T2/NTA
ops.prefetch_range(&data);                  // data: &[u8], strided by cache line
ops.copy_cache_optimized(&src, &mut dst)?;  // prefetches ahead on large copies
let ord = ops.compare_cache_optimized(&a, &b);

// Introspection
let tier = ops.tier();                      // SimdTier::{Avx512, Avx2, Sse2, Scalar}
```

### Global convenience functions

```rust
use zipora::memory::{
    fast_copy, fast_compare, fast_find_byte, fast_fill,
    fast_copy_cache_optimized, fast_compare_cache_optimized,
    fast_prefetch, fast_prefetch_range, PrefetchHint,
};

fast_copy(&src, &mut dst)?;                  // uses get_global_simd_ops() internally
let ord = fast_compare(&a, &b);
let pos = fast_find_byte(&haystack, b'S');
fast_fill(&mut buf, 0);

fast_prefetch(&some_value, PrefetchHint::T0); // fast_prefetch<T: ?Sized>(data: &T, hint)
fast_prefetch_range(&data);                   // data: &[u8]
```

`get_global_simd_ops()` returns the cached `&'static SimdMemOps` these wrappers use.
Custom cache behavior comes from `SimdMemOps::with_cache_config(CacheLayoutConfig)`
(see `src/memory/cache_layout.rs`, e.g. `CacheLayoutConfig::sequential()`).

## Multi-Dimensional Rank/Select

`MultiDimRankSelect` (`zipora::succinct::rank_select::multidim_simd`, re-exported at
`zipora::succinct::rank_select`) stores N bit vectors of equal length — one per
dimension — each backed by cache-optimized `RankSelectInterleaved256`, with vectorized
bulk queries and AVX2-accelerated cross-dimensional set operations.

```rust
use zipora::succinct::rank_select::MultiDimRankSelect;
use zipora::BitVector;

// Create 4-dimensional rank/select structure
let mut dimensions = vec![];
for _ in 0..4 {
    let mut bv = BitVector::new();
    for i in 0..1000 {
        bv.push(i % 3 == 0)?;
    }
    dimensions.push(bv);
}
let multi_rs: MultiDimRankSelect<4> = MultiDimRankSelect::new(dimensions)?;

// Vectorized bulk rank: ranks[d] = rank1(positions[d]) in dimension d
let positions = [100, 200, 300, 400];
let ranks: [usize; 4] = multi_rs.bulk_rank_multidim(&positions);

// Vectorized bulk select: positions[d] = select1(ranks[d]) in dimension d
let ranks = [5, 10, 15, 20];
let positions = multi_rs.bulk_select_multidim(&ranks)?;

// Cross-dimensional intersection (bitwise AND) — returns a new BitVector
let intersection = multi_rs.intersect_dimensions(0, 1)?;

// Cross-dimensional union (bitwise OR) over any set of dimensions
let union = multi_rs.union_dimensions(&[0, 1, 2])?;

assert_eq!(multi_rs.num_dimensions(), 4);
```

Constraints and errors:

- Dimensions: 1 to 32 (`new` returns an error outside this range, or when the number of
  bit vectors doesn't match `DIMS`, or when dimensions have different bit lengths).
- `intersect_dimensions`/`union_dimensions` return an error for out-of-range dimension
  indices.
- AVX2 paths cover bulk operations for ≤ 4 dimensions and vectorize AND/OR 256 bits at
  a time; larger dimension counts and other platforms use the scalar/POPCNT paths.

Results compose with the standard rank/select types:

```rust
use zipora::succinct::rank_select::RankSelectInterleaved256;

let intersection = multi_rs.intersect_dimensions(0, 1)?;
let rs = RankSelectInterleaved256::new(intersection)?;
let rank = rs.rank1(100);
```

## BMI2 Acceleration

BMI2 helpers live in `src/succinct/rank_select/bmi2_acceleration.rs`; the main types are
re-exported at the crate root: `Bmi2Capabilities`, `Bmi2BitOps`, `Bmi2BlockOps`,
`Bmi2SelectOps` (plus `Bmi2RankOps`, `Bmi2RangeOps`, and others).

```rust
use zipora::{Bmi2Capabilities, Bmi2SelectOps, Bmi2BitOps, Bmi2BlockOps};

// Capability detection (cached via Bmi2Capabilities::get())
let caps = Bmi2Capabilities::get();
println!("BMI1: {}, BMI2: {}", caps.has_bmi1, caps.has_bmi2);
// caps.simd_caps: SimdCapabilities

// PDEP-based select within a u64 (returns None if k >= popcount)
let word = 0b1010101010101010u64;
let pos: Option<u32> = Bmi2SelectOps::select1_u64(word, 3);
let pos = Bmi2SelectOps::select1_u64_enhanced(word, 3);

// Parallel bit deposit/extract (PDEP/PEXT with scalar fallback)
let deposited = Bmi2BitOps::deposit_bits(source, mask);
let extracted = Bmi2BitOps::extract_bits(source, mask);

// Bulk operations over &[u64] blocks
let words = vec![0xAAAAAAAAAAAAAAAAu64; 1000];
let positions: Vec<usize> = (0..100).map(|i| i * 10).collect();
let ranks: Vec<usize> = Bmi2BlockOps::rank_bulk(&words, &positions);
let selected = Bmi2BlockOps::select_bulk(&words, &[5, 50, 500])?;
```

### AMD PDEP hazard and `select_in_word`

PDEP/PEXT are microcoded (extremely slow) on AMD Zen 1/2. Zipora guards against this
with `has_fast_bmi2()` in `zipora::algorithms::bit_ops`, which combines
`is_x86_feature_detected!("bmi2")` with CPUID family detection (cached in a `OnceLock`)
and returns `false` on Zen 1/2.

The centralized `select_in_word(word, rank)` in `zipora::algorithms::bit_ops` dispatches
across three tiers — PDEP (when `has_fast_bmi2()`) → POPCNT binary search → scalar — and
is used by all rank/select and Elias-Fano hot paths. `popcount_slice(&[u64])` from the
same module provides multi-tier SIMD popcount.

## AVX-512

There is no standalone AVX-512 API surface. AVX-512 code paths exist inside specific
modules (`popcount_slice`, rank/select bulk operations, `SimdMemOps`) and are compiled
behind the `avx512` cargo feature with runtime detection.

## Performance

Verified numbers (see [PERFORMANCE.md](PERFORMANCE.md) for methodology and details):

- **Rank/Select**: ~5.2 Gops/s single-query; bulk SIMD operations 10x scalar; bitwise
  SIMD operations 41x scalar
- **popcount_slice**: 5.2 Gwords/s (POPCNT tier), NEON supported
- **StreamVByte decode**: 8.7x scalar (SSSE3 shuffle-table)

## Best Practices

1. **Always provide a scalar fallback** — Tier 0 is mandatory.
2. **Use the dispatch macros** — reduces duplication; remember the `return` expansion
   footgun for `simd_dispatch!`.
3. **Benchmark on target hardware** — SIMD performance varies by microarchitecture.
4. **Consider data size** — SIMD overhead may not pay off for small inputs; the adaptive
   selector applies size/density thresholds for exactly this reason.
5. **Guard PDEP/PEXT with `has_fast_bmi2()`** — never with raw BMI2 detection alone.
