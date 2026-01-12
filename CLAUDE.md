# CLAUDE.md

## Commands
```bash
cargo build --release && cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## Build Status
- **Build**: Zero errors | **Tests**: 2,300 unit + 207 doctests (100% pass)
- **P0 Critical Issues**: All 6 resolved

## Verified Performance
- Rank/Select: 0.53 Gops/s (BMI2)
- Huffman O1: 2.1-2.6x speedup with fast symbol table
- Radix Sort: 4-8x vs comparison sorts
- SIMD Memory: 4-12x bulk operations

## Remaining Work
- P1: Eliminate unwrap/expect (68/3600 done)
- P1: Document unsafe blocks (~10% done)
- P2: Refactor oversized files

## SIMD Framework
**Tiers**: AVX-512 → AVX2 → BMI2 → POPCNT → NEON → Scalar (mandatory fallback)

```rust
use zipora::{simd_dispatch, simd_feature_check};

simd_dispatch!(avx2 => unsafe { f_avx2(d) }, sse2 => unsafe { f_sse2(d) }, _ => f_scalar(d))
simd_feature_check!("popcnt", unsafe { hw_impl(d) }, scalar_impl(d))
```

## Key Types
| Category | Types |
|----------|-------|
| Memory | `SecureMemoryPool`, `LockFreeMemoryPool`, `MmapVec` |
| Hash | `ZiporaHashMap`, `GoldHashMap`, `CacheOptimizedHashMap` |
| Containers | `UintVecMin0`, `ZipIntVec`, `ValVec32`, `FastVec` |
| Compression | `ContextualHuffmanEncoder`, `FseEncoder`, `Rans64Encoder` |
| Tries | `ZiporaTrie` (Patricia, CritBit, DoubleArray, NestedLouds) |

## Features
- **Default**: `simd`, `mmap`, `zstd`, `serde`
- **Optional**: `lz4`, `ffi`, `avx512` (nightly)
