# Verified Performance

Every performance and space figure in this document is generated directly from the committed benchmark baselines under [`benches/baselines/bindiego/`](../benches/baselines/bindiego/) by [`scripts/perf_table.py`](../scripts/perf_table.py) and verified in CI / `make sanity` (`python3 scripts/perf_table.py --check`).

For full per-benchmark methodology, dataset specifications, hardware pinning, THP sensitivity analysis, and row-level provenance (`measured_at_commit` and `binary_sha256`), see:
- **[Stage 14 Core Benchmarks & E1.1 Systemic BM25 Search](review/e1.md)** (`benches/core_bench.rs`, `benches/e2e_search_bench.rs`)
- **[Stage 15 Competitor Benchmark Matrix & E3 Standing Table](review/e2_e3.md)** (`benches/competitor_bench.rs`)
- **[Stage 16 Optimization Ledger](review/e4.md)**

## Reproducing Benchmarks

```bash
# Run core benchmark suite and verify against benches/baselines/<host>/core_bench.json
make bench_core

# Run competitor benchmark suite and verify against benches/baselines/<host>/competitor_bench.json
make bench_competitors

# Run end-to-end BM25 search benchmark against tantivy
make bench_e2e_search

# Verify that README.md, README_cn.md, docs/PERFORMANCE.md, CLAUDE.md, and GEMINI.md match baselines
python3 scripts/perf_table.py --check
```

## Baseline Summary

<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->
Measured on reference host `bindiego` (AMD EPYC 7B13, 64 logical cores, pinned core 23, `THP-off` via `PR_SET_THP_DISABLE`, `-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32`).
Across all **197** head-to-head comparisons in [`docs/review/e2_e3.md`](docs/review/e2_e3.md) and [`docs/review/e4.md`](docs/review/e4.md), Zipora is **`LEADING` on 68/197** (primary within `-3.0%` and secondary space `<= +10.0%`) and **faster or tied on primary on 105/197 (53.3%)**.

### Head-to-Head Competitor Highlights (`benches/baselines/bindiego/competitor_bench.json` & `e2e_search.json`)

| Family | Zipora Operation | Zipora (`min`) | Space / Secondary | Competitor | Competitor (`min`) | Speedup |
|---|---|---:|---:|---|---:|---:|
| Rank/Select (`1 Gbit`) | `RankSelectInterleaved256::rank1` | `35.03 ns/op` | `131.25% of raw` | `sucds::Rank9Sel` | `43.33 ns/op` | **`1.24x`** |
| Rank/Select (`1 Gbit`) | `RankSelectSE256::select1` | `145.77 ns/op` | `137.50% of raw` | `sucds::Rank9Sel` | `206.40 ns/op` | **`1.42x`** |
| Elias-Fano (`100k`) | `EliasFano::next_geq` (uniform) | `22.14 ns/op` | `8.62 bits/elem` | `sucds::EliasFano` | `47.53 ns/op` | **`2.15x`** |
| Elias-Fano (`100k`) | `OptimalPartitionedEliasFano::next_geq` | `26.21 ns/op` | `5.42 bits/elem` | `vers-vecs::EliasFanoVec` | `61.26 ns/op` | **`2.34x`** |
| Elias-Fano (`100k`) | `ClusteredEliasFano::get` (dense) | `3.39 ns/op` | `2.51 bits/elem` | `sucds::EliasFano` | `8.05 ns/op` | **`2.37x`** |
| Tries (`1M` keys) | `DoubleArrayTrie::contains` (shared prefix) | `254.87 ns/op` | `25.60 B/key` | `qp-trie::Trie` | `533.08 ns/op` | **`2.09x`** |
| Tries (`1M` keys) | `CsppTrie::contains` (shared prefix) | `255.50 ns/op` | `9.60 B/key` | `radix_trie::Trie` | `977.94 ns/op` | **`3.83x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<[u8; 16], u64>::get` | `87.13 ns/op` | `52.43 B/entry` | `hashbrown::HashMap` (`ahash`) | `184.62 ns/op` | **`2.12x`** |
| Hash Maps (`1M` keys) | `ZiporaHashMap<u64, u64>::get` | `66.12 ns/op` | `35.65 B/entry` | `std::collections::HashMap` | `90.22 ns/op` | **`1.36x`** |
| Entropy (`64 KiB`) | `HuffmanDecoder::decode` (O0) | `3.50 ns/byte` | `4.01 bits/B` | `huffman-compress` | `9.45 ns/byte` | **`2.70x`** |
| Entropy (`64 KiB`) | `Rans64Decoder<ParallelX4>::decode` | `2.51 ns/byte` | `4.05 bits/B` | `constriction` ANS | `6.16 ns/byte` | **`2.46x`** |
| Integer Codecs (`100k`) | `StreamVByte::encode_deltas` (SSSE3) | `0.49 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `1.28 ns/int` | **`2.62x`** |
| Integer Codecs (`100k`) | `StreamVByte::decode_into` (SSSE3) | `0.19 ns/int` | `10.00 bits/int` | `stream-vbyte` SIMD | `0.25 ns/int` | **`1.36x`** |
| Radix Sort (`10M`) | `RadixSort::sort_u32` (hybrid MSB+LSD) | `5.70 ns/elem` | `8.00 B/elem` | `voracious_radix_sort` | `10.10 ns/elem` | **`1.77x`** |
| String Search (`64 KiB`) | `simd_strchr` (128B AVX2) | `644.45 ns/call` | `0 B aux` | `memchr::memchr` | `637.03 ns/call` | **`0.99x`** |
| String Search (`64 KiB`) | `simd_strstr` (128B AVX2) | `649.06 ns/call` | `0 B aux` | `memchr::memmem` | `1957.34 ns/call` | **`3.02x`** |
| Systemic Search (`100k` docs) | `Zipora BMW` top-10 (`p50 = 21.31 us`) | `19,677.1 QPS` | `1.219 B/posting` (postings) | `tantivy` v0.26.2 (`p50 = 53.69 us`) | `12,176.3 QPS` | **`1.62x`** (`2.31x` vs uncompressed) |

### Additional Core Suite Baselines (`benches/baselines/bindiego/core_bench.json`)

- **BM25 & Fieldnorm Scoring**: `Bm25BatchScorer` AVX2 batch `0.4726 ns/doc`, `FieldnormEncoder::build_score_table` `0.1734 ns/cell` (`1.0 B/doc` fieldnorms).
- **Memory Pools**: `LockFreeMemoryPool` 64B alloc+free `9.111 ns/op` (1T) / `9.120 ns/op` (4T).
- **Rank9 (`1 Gbit`)**: `Rank9::rank1` `45.01 ns/op` at `25.00%` space overhead.
- **Suffix Array (`10M` bytes, SA-IS)**: construction `81.68 ns/byte` (`9.05 B/byte`), `search_range` query `1446.15 ns/op`.
<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->
