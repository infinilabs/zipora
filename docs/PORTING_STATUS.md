# Porting Status: C++ topling-zip → Rust Zipora

**The port is complete.** As of v4.x, Zipora has feature parity with the
reference C++ topling-zip library (succinct data structures, blob stores,
FSA/tries, entropy coding) plus Rust-native additions that have no C++
counterpart (memory-safe concurrent tries, SIMD search-engine primitives,
BM25 scoring, fuzzing/Miri/TSan validation infrastructure).

## Completed subsystems

| Subsystem | Highlights | Documentation |
|-----------|-----------|---------------|
| Succinct structures | 12 rank/select variants, BitVector, Elias-Fano family (EF / PEF / OPEF / Clustered), Rank9, HybridPostingList | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| Tries / FSA | DoubleArrayTrie (XOR transitions), DoubleArrayTrieMap, ZiporaTrie (Patricia / CritBit / LOUDS), CsppTrie + ConcurrentCsppTrie (multi-writer, EBR), NestTrieDawg | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| Blob stores | MixedLen, SimpleZip, ZeroLength, ZipOffset, dictionary/entropy-compressed variants; binary-compatible 80-byte header + 64-byte footer | [BLOB_STORAGE.md](BLOB_STORAGE.md) |
| Compression | Huffman (O0/O1/O2), FSE, rANS (64-bit), dictionary compression (PA-Zip / SA-IS), StreamVByte (SSSE3), adaptive selection | [COMPRESSION.md](COMPRESSION.md) |
| Containers | FastVec, ValVec32, UintVecMin0, ZipIntVec, SortedUintVec, SmallMap, LruMap, circular queues, specialized string vectors | [CONTAINERS.md](CONTAINERS.md) |
| Hash maps | ZiporaHashMap (strategy-based), GoldHashMap | [HASH_MAPS.md](HASH_MAPS.md) |
| Memory | SecureMemoryPool, LockFreeMemoryPool, five-level pool, MmapVec (memmap2 MAP_SHARED), bump/arena allocators | [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) |
| Algorithms | SIMD popcount/select/gallop/filter, LSD radix sort (ping-pong), suffix arrays (SA-IS), cache-oblivious algorithms | [ALGORITHMS.md](ALGORITHMS.md), [SIMD.md](SIMD.md) |
| Scoring | FieldnormEncoder (Lucene SmallFloat), Bm25BatchScorer (AVX2) | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| FFI | Optional C API behind the `ffi` feature, panic-safe entry points | [FFI.md](FFI.md) |

## Current state (v4.0.3)

- 2,734 debug / 2,749 release tests, 100% pass; build and clippy clean
  (`-D warnings`).
- Zero production `.unwrap()`/`.expect()` outside test code; 100% of
  production `unsafe` blocks carry `SAFETY:` comments.
- Adversarial validation: 8 cargo-fuzz targets, Miri (core containers +
  concurrent trie), ThreadSanitizer jobs (see the Makefile:
  `fuzz_smoke`, `fuzz_soak`, `miri_core`, `miri_cspp`, `tsan_cspp`,
  `tsan_pool`).

## Where to look next

- Verified benchmark numbers: [PERFORMANCE.md](PERFORMANCE.md)
- Per-topic guides: the other files in this `docs/` directory
- Historical porting log: git history of this file (pre-v4 revisions)
