# Verified Performance

> **Test Machine**: AMD EPYC 7B13 (Zen 3), 64 vCPUs, 117 GB RAM, AVX2/BMI2/POPCNT, rustc 1.91.1, Linux 6.17.
> Results vary across hardware — Intel may differ on BMI2 (native vs microcode), ARM lacks x86 SIMD paths.
> Run `cargo bench` to reproduce on your own hardware.

## Trie / Term Dictionary (DoubleArrayTrie)

| Operation (5000 terms) | Time | Per-op |
|------------------------|------|--------|
| Lookup hit | 103 µs | 20.6 ns/lookup |
| Lookup miss | 19 µs | 3.8 ns/lookup |
| Prefix search (5 queries) | 14 µs | 2.8 µs/query |
| Insert (incremental) | 967 µs | 193 ns/term |

XOR transitions, terminal bit in NInfo, unsafe `get_unchecked` — 3 ops, 1 branch per transition.
Supports arbitrary binary keys including `\x00` bytes.

## BitVector (scatter + popcount)

| Operation (1M bits) | Zipora | Scalar Vec\<u64\> | Ratio |
|---------------------|--------|-------------------|-------|
| Scatter + popcount (20×5K docs) | **1.08 ms** | 1.35 ms | **0.80x (faster)** |
| Allocation (`with_size(1M, false)`) | **155 µs** | 247 µs | **0.63x (faster)** |
| Popcount only (50% density) | 9.25 µs | 9.26 µs | Tied |

`alloc_zeroed` (calloc), zero-copy `from_blocks`, SIMD `popcount_slice` (AVX-512 / POPCNT / AVX2 / NEON).

## popcount_slice (SIMD population count)

| Slice size | Throughput | Rate |
|------------|-----------|------|
| 16 words (128B) | 4.4 ns | 3.7 Gwords/s |
| 781 words (6KB, engine union buffer) | 150 ns | 5.2 Gwords/s |
| 10K words (80KB) | 1.9 µs | 5.4 Gwords/s |

Multi-tier dispatch: AVX-512 VPOPCNTDQ → hardware POPCNT → AVX2 vpshufb → NEON → scalar.
Used internally by `BitVector::count_ones()` and available as `zipora::algorithms::popcount_slice`.

## Succinct Data Structures

| Operation | Zipora | Baseline | Speedup |
|-----------|--------|----------|---------|
| Rank1 query (100K bits) | 192 ns | — | ~5.2 Gops/s |
| Select1 query (100K bits) | 5.4 ms / 100K queries | — | ~18.5 Mops/s |
| Bulk rank (SIMD, 50K) | 8.4 µs | 84.1 µs (individual) | **10x** |
| Bulk bitwise ops (SIMD, 50K) | 3.1 µs | 128.4 µs (individual) | **41x** |
| Range set (SIMD, 50K) | 3.2 µs | 17.9 µs (individual) | **5.6x** |

## Containers vs std

| Operation | Zipora | std | Ratio |
|-----------|--------|-----|-------|
| ValVec32 push (100K) | 119 µs | 120 µs | 1.0x |
| ValVec32 random access (100K) | 706 ns | 729 ns | **0.97x** |
| ValVec32 iteration (10K) | 778 ns | 783 ns | 1.0x |
| ValVec32 bulk extend (100K) | 21.8 µs | 28.7 µs | **0.76x** |
| SmallMap insert+lookup (8 keys) | 444 ns | 805 ns (HashMap) | **1.8x** |
| SmallMap lookup-intensive | 36.9 µs | 141.7 µs (HashMap) | **3.8x** |
| CircularQueue push+pop (100K) | 326 µs | 381 µs (VecDeque) | **0.86x** |
| FixedStr16Vec push (100K) | 755 µs | 5,906 µs (Vec\<String\>) | **7.8x** |
| SortableStrVec sort (5K) | 390 µs | 448 µs (Vec\<String\>) | **1.15x** |

## Entropy Coding (65KB input)

| Algorithm | Entropy 0.5 | Entropy 2.0 | Entropy 6.0 |
|-----------|-------------|-------------|-------------|
| Huffman O0 | 1,124 µs | 1,235 µs | 1,720 µs |
| Huffman O1 (x1 stream) | 188 µs | 173 µs | 188 µs |
| rANS64 | 405 µs | 351 µs | 426 µs |

## Cache (LRU vs HashMap)

| Operation | LruMap | HashMap | Note |
|-----------|--------|---------|------|
| Hot get (cap=64, 10K ops) | 5.7 µs | 152 µs | **26x** faster (hot-set fits in cache) |
| Hot get (cap=1024, 10K ops) | 94.6 µs | 152 µs | **1.6x** faster |
| Insert (cap=64, 10K ops) | 1,897 µs | 1,177 µs | 0.62x (eviction overhead) |

## BM25 Scoring (FieldnormEncoder + Bm25BatchScorer)

| Operation (1M docs) | UintVecMin0 + float | FieldnormEncoder | Improvement |
|---------------------|---------------------|------------------|-------------|
| Memory | 1.13 MB | **1.00 MB** | **2.0x smaller** |
| Random access | 190 µs | **92 µs** | **2.06x faster** |
| BM25 pre-compute (scalar) | 5.13 ms | **2.78 ms** | **1.85x faster** |
| BM25 pre-compute (AVX2 SIMD) | 5.13 ms | **381 µs** | **13.5x faster** |
| Phrase query scoring (1K random) | 3.63 µs | **1.22 µs** | **2.98x faster** |

Lucene SmallFloat encoding (1 byte/doc), 256-entry norm table (zero per-posting division), AVX2 batch scorer (8 scores/iteration).
