# Zipora

[中文](README_cn.md)

[![Build Status](https://github.com/infinilabs/zipora/workflows/CI/badge.svg)](https://github.com/infinilabs/zipora/actions)
[![License](https://img.shields.io/badge/license-BDL--1.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.95+-orange.svg)](https://www.rust-lang.org)

High-performance Rust data structures and compression algorithms with memory safety guarantees.

## Key Features

<!-- BEGIN GENERATED KEY FEATURES (scripts/perf_table.py) -->
- **High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512), cache-friendly layouts, SIMD cursor primitives for Block-Max WAND
- **Memory Safety**: Every production `unsafe` site is annotated with a `// SAFETY:` or `# Safety` comment (`scripts/unsafe_audit.py`)
- **Secure Memory Management**: Production-ready memory pools (`SecureMemoryPool`, `LockFreeMemoryPool`, `FixedCapacityPool`, `AdaptiveFiveLevelPool`, `BumpArena`) with thread safety and RAII (`tests/memory_pool_contract.rs`)
- **Blob Storage**: Specialized stores (`MemoryBlobStore`, `PlainBlobStore`, `SimpleZipBlobStore`, `MixedLenBlobStore`, `ZipOffsetBlobStore`, `NestLoudsTrieBlobStore`, `DictZipBlobStore`, `ZstdBlobStore`, `Lz4BlobStore`) with trie-based indexing and compression (`tests/format_spec_golden_tests.rs`)
- **Succinct Data Structures**: `Rank9` (Vigna 2008, `25.00%` overhead), `RankSelectInterleaved256` (`25.00%` rank-only / `31.25%` with select caches), `RankSelectSE256`, `EliasFano` / `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano` with cursor `advance_to_index`, `HybridPostingList` (auto-select encoding), AMD-safe PDEP with `has_fast_bmi2` (`tests/memory_footprint_e6.rs`)
- **BM25 Scoring**: `FieldnormEncoder` (Lucene SmallFloat, `1.0 B/doc` fieldnorms) + `Bm25BatchScorer` (AVX2/AVX-512 SIMD batch, prefetch) (`tests/memory_footprint_e6.rs`, `benches/core_bench.rs`)
- **Specialized Containers**: `FastVec`, `ValVec32`, `IntVec`, `UintVecMin0`, `ZipIntVec`, `FixedLenStrVec`, `SortableStrVec`, `ZoSortedStrVec`, `VecTrbSet`/`VecTrbMap`, `MinimalSso`, `LruMap`, `ConcurrentLruMap`, `FixedCircularQueue`
- **Hash Maps**: SwissTable 16-way SSE2 group-probing `ZiporaHashMap` (`35.65 B/entry` on `1M` `u64`), `GoldHashMap` (`benches/competitor_bench.rs`)
- **Advanced Tries**: Double-Array (`DoubleArrayTrie`, `12.0 B/state`; `DoubleArrayTrieMap<u32>`, `12.0–16.0 B/state`), LOUDS, Critical-Bit (BMI2), Patricia tries, lazy prefix/fuzzy iterators, `CsppTrie` (Compressed Sparse Parallel Patricia, `9.60 B/key` on shared-prefix keys), `ConcurrentCsppTrie` (multi-writer/multi-reader, epoch-based reclamation) (`tests/memory_footprint_e6.rs`)
- **Compression & Integer Codecs**: PA-Zip, Huffman O0/O1/O2, FSE, rANS (64-bit X1/X2/X4/X8), `StreamVByte` (SSSE3), `GroupVarint` (SSSE3), ZSTD & LZ4 integration (`benches/competitor_bench.rs`)
- **C FFI Support**: Complete C11/C++17 API (`--features ffi`, `tests/ffi_c_smoke.rs`)
<!-- END GENERATED KEY FEATURES (scripts/perf_table.py) -->

## Quick Start

```toml
[dependencies]
zipora = "5.0.0"

# With C FFI bindings
zipora = { version = "5.0.0", features = ["ffi"] }

# AVX-512
zipora = { version = "5.0.0", features = ["avx512"] }
```

### Basic Usage

```rust
use zipora::*;

// High-performance vector
let mut vec = FastVec::new();
vec.push(42).unwrap();

// Zero-copy strings with SIMD hashing
let s = FastStr::from_string("hello world");
println!("Hash: {:x}", s.hash_fast());

// Intelligent rank/select with automatic optimization
let mut bv = BitVector::new();
for i in 0..1000 { bv.push(i % 7 == 0).unwrap(); }
let adaptive_rs = AdaptiveRankSelect::new(bv).unwrap();
let rank = adaptive_rs.rank1(500);

// Unified Trie - Strategy-based configuration
use zipora::fsa::{ZiporaTrie, ZiporaTrieConfig, Trie};

let mut trie: ZiporaTrie = ZiporaTrie::new();
trie.insert(b"hello").unwrap();
assert!(trie.contains(b"hello"));

// Unified Hash Map - Strategy-based configuration
use zipora::hash_map::{ZiporaHashMap, ZiporaHashMapConfig};

let mut map: ZiporaHashMap<&str, &str> = ZiporaHashMap::new().unwrap();
map.insert("key", "value").unwrap();

// Blob storage with compression
use zipora::blob_store::{ZipOffsetBlobStoreBuilder, ZipOffsetBlobStoreConfig};

let config = ZipOffsetBlobStoreConfig::performance_optimized();
let mut builder = ZipOffsetBlobStoreBuilder::with_config(config).unwrap();
builder.add_record(b"Compressed data").unwrap();
let store = builder.finish().unwrap();

// Entropy coding
let encoder = HuffmanEncoder::new(b"sample data").unwrap();
let compressed = encoder.encode(b"sample data").unwrap();

// String utilities
use zipora::string::{join_str, hex_encode, hex_decode, words, decimal_strcmp};
let joined = join_str(", ", &["hello", "world"]);
assert_eq!(joined, "hello, world");
```

## Documentation

### Core Components
- **[Containers](docs/CONTAINERS.md)** - Specialized containers (FastVec, ValVec32, IntVec, LruMap, etc.)
- **[Hash Maps](docs/HASH_MAPS.md)** - ZiporaHashMap, GoldHashMap with strategy-based configuration
- **[Blob Storage](docs/BLOB_STORAGE.md)** - Specialized blob store variants with trie indexing and compression
- **[Memory Management](docs/MEMORY_MANAGEMENT.md)** - SecureMemoryPool, MmapVec, five-level pools

### Algorithms & Processing
- **[Algorithms](docs/ALGORITHMS.md)** - Radix sort, suffix arrays, set operations, cache-oblivious algorithms, SIMD popcount, SIMD galloping search, SIMD block filter
- **[Compression](docs/COMPRESSION.md)** - Huffman, FSE, rANS, dictionary compression (PA-Zip), StreamVByte, adaptive selection
- **[String Processing](docs/STRING_PROCESSING.md)** - SIMD string search, join, numeric compare, word boundaries, hex

### System Architecture
- **[Concurrency](docs/CONCURRENCY.md)** - Pipeline processing, work-stealing, parallel trie building
- **[Error Handling](docs/ERROR_HANDLING.md)** - ZiporaError/Result patterns, error classification, bounds checking
- **[Configuration](docs/CONFIGURATION.md)** - Rich configuration APIs, presets, validation
- **[SIMD Framework](docs/SIMD.md)** - 6-tier SIMD with AVX2/BMI2/POPCNT support

### Integration
- **[I/O & Serialization](docs/IO_SERIALIZATION.md)** - Stream processing, endian handling, varint encoding
- **[C FFI](docs/FFI.md)** - C API for interoperability

### Guides
- **[Search Engine Guide](docs/SEARCH_ENGINE_GUIDE.md)** - End-to-end search engine architecture with Zipora
- **[Performance Benchmarks](docs/PERFORMANCE.md)** - Verified benchmarks across all components

### Reference
- **[Binary Format Spec](docs/format_spec.md)** - Wire layouts for persisted containers and blob stores
- **[Porting Status](docs/PORTING_STATUS.md)** - Feature implementation status

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | Yes | SIMD optimizations (AVX2, SSE4.2) |
| `mmap` | Yes | Memory-mapped file support |
| `zstd` | Yes | ZSTD compression |
| `serde` | Yes | Serialization support (serde, serde_json, bincode) |
| `lz4` | Yes | LZ4 compression |
| `async` | Yes | Async runtime (tokio) for concurrency, pipeline, real-time compression |
| `ffi` | No | C FFI bindings |
| `avx512` | Yes | AVX-512 optimizations (stable since 1.89) |

## Build & Test

```bash
# Build (default features)
cargo build --release

# Build with all features including FFI
cargo build --release --all-features

# Test
cargo test --lib

# Sanity check (all feature combinations, debug + release)
make sanity

# Benchmark (release only)
cargo bench

# Lint
cargo clippy --all-targets --all-features -- -D warnings
```

## Verified Performance

See **[Stage 14 Core Benchmarks](docs/review/e1.md)**, **[Stage 15 Competitor Matrix](docs/review/e2_e3.md)**, and **[Stage 16 Optimization Ledger](docs/review/e4.md)** for full methodology and per-benchmark provenance.

<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->
Measured on reference host `bindiego` (AMD EPYC 7B13, 64 logical cores, pinned core 23, `THP-off` via `PR_SET_THP_DISABLE`, `-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32`).
Across all **197** head-to-head comparisons in [`docs/review/e2_e3.md`](docs/review/e2_e3.md) and [`docs/review/e4.md`](docs/review/e4.md), Zipora is **`LEADING` on 68/197** (primary within `-3.0%` and secondary space `<= +10.0%`) and **faster or tied on primary on 107/197 (54.3%)**.

### Head-to-Head Competitor Highlights (`benches/baselines/bindiego/competitor_bench.json` & `e2e_search.json`)

| Family | Zipora Operation | Zipora (`min`) | Space / Secondary | Competitor | Competitor (`min`) | Speedup |
|---|---|---:|---:|---|---:|---:|
| Rank/Select (`1 Gbit`) | `RankSelectInterleaved256::rank1` | `35.23 ns/op` | `131.25% of raw` | `sucds::Rank9Sel` | `42.92 ns/op` | **`1.22x`** |
| Rank/Select (`1 Gbit`) | `RankSelectSE256::select1` | `149.64 ns/op` | `137.50% of raw` | `sucds::Rank9Sel` | `203.58 ns/op` | **`1.36x`** |
| Elias-Fano (`100k`) | `EliasFano::next_geq` (uniform) | `22.17 ns/op` | `8.62 bits/elem` | `sucds::EliasFano` | `47.69 ns/op` | **`2.15x`** |
| Elias-Fano (`100k`) | `OptimalPartitionedEliasFano::next_geq` | `26.55 ns/op` | `5.42 bits/elem` | `vers-vecs::EliasFanoVec` | `61.44 ns/op` | **`2.31x`** |
| Elias-Fano (`100k`) | `ClusteredEliasFano::get` (dense) | `3.39 ns/op` | `2.51 bits/elem` | `sucds::EliasFano` | `8.03 ns/op` | **`2.37x`** |
| Tries (`1M` keys) | `DoubleArrayTrie::contains` (shared prefix) | `247.11 ns/op` | `25.60 B/key` | `qp-trie::Trie` | `521.72 ns/op` | **`2.11x`** |
| Tries (`1M` keys) | `CsppTrie::contains` (shared prefix) | `210.57 ns/op` | `9.60 B/key` | `radix_trie::Trie` | `1027.99 ns/op` | **`4.88x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<[u8; 16], u64>::get` | `83.74 ns/op` | `52.43 B/entry` | `hashbrown::HashMap` (`ahash`) | `191.06 ns/op` | **`2.28x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<u64, u64>::get` | `66.96 ns/op` | `35.65 B/entry` | `std::collections::HashMap` | `85.55 ns/op` | **`1.28x`** |
| Entropy (`64 KiB`) | `HuffmanDecoder::decode` (O0) | `3.50 ns/byte` | `4.01 bits/B` | `huffman-compress` | `9.43 ns/byte` | **`2.69x`** |
| Entropy (`64 KiB`) | `Rans64Decoder<ParallelX4>::decode` | `2.50 ns/byte` | `4.05 bits/B` | `constriction` ANS | `6.15 ns/byte` | **`2.47x`** |
| Integer Codecs (`100k`) | `StreamVByte::encode_deltas` (SSSE3) | `0.48 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `1.29 ns/int` | **`2.68x`** |
| Integer Codecs (`100k`) | `StreamVByte::decode_into` (SSSE3) | `0.19 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `0.25 ns/int` | **`1.36x`** |
| Radix Sort (`10M`) | `RadixSort::sort_u32` (hybrid MSB+LSD) | `5.68 ns/elem` | `8.00 B/elem` | `voracious_radix_sort` | `10.11 ns/elem` | **`1.78x`** |
| String Search (`64 KiB`) | `simd_strchr` (128B AVX2) | `644.84 ns/call` | `0 B aux` | `memchr::memchr` | `636.33 ns/call` | **`0.99x`** |
| String Search (`64 KiB`) | `simd_strstr` (128B AVX2) | `649.14 ns/call` | `0 B aux` | `memchr::memmem` | `1921.17 ns/call` | **`2.96x`** |
| Systemic Search (`100k` docs) | `Zipora BMW` top-10 (`p50 = 21.14 us`) | `19,917.6 QPS` | `1.219 B/posting` (postings) | `tantivy` v0.26.2 (`p50 = 55.04 us`) | `12,036.3 QPS` | **`1.65x`** (`2.34x` vs uncompressed) |

### Additional Core Suite Baselines (`benches/baselines/bindiego/core_bench.json`)

- **BM25 & Fieldnorm Scoring**: `Bm25BatchScorer` AVX2 batch `0.4784 ns/doc`, `FieldnormEncoder::build_score_table` `0.1699 ns/cell` (`1.0 B/doc` fieldnorms).
- **Memory Pools**: `LockFreeMemoryPool` 64B alloc+free `9.106 ns/op` (1T) / `9.110 ns/op` (4T).
- **Rank9 (`1 Gbit`)**: `Rank9::rank1` `43.23 ns/op` at `25.00%` space overhead.
- **Suffix Array (`10M` bytes, SA-IS)**: construction `75.28 ns/byte` (`9.05 B/byte`), `search_range` query `1330.43 ns/op`.
<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->

## Dependencies

Minimal dependency footprint by design:
- **Core**: `bytemuck`, `thiserror`, `log`, `ahash`, `rayon`, `libc`, `raw-cpuid`, `crossbeam-epoch`, `thread_local`, `fastrand`, `bitflags`, `base64`
- **Default**: `memmap2` (mmap), `zstd`, `lz4_flex`, `serde`/`serde_json`/`bincode`, `tokio` (async)
- **Optional**: `cbindgen` (ffi)
- **Removed**: `crossbeam-utils`, `parking_lot`, `uuid`, `num_cpus`, `async-trait`, `futures`, `once_cell`, `pkg-config` (all replaced with std or eliminated)

## Building a Search Engine with Zipora

See **[Search Engine Guide](docs/SEARCH_ENGINE_GUIDE.md)** for the complete guide with code examples covering all 11 components: term dictionaries (DoubleArrayTrie + lazy prefix/fuzzy iterators), posting lists (HybridPostingList + Elias-Fano cursors), SIMD query primitives (simd_gallop_to, simd_block_filter, advance_to_index), BM25 scoring (FieldnormEncoder + Bm25BatchScorer), document storage, compression, multi-threaded indexing, and component selection guide.

## License

The Bindiego License (BDL), Version 1.0 - See [LICENSE](LICENSE) for details.
