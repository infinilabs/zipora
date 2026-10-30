# Porting Status: C++ topling-zip → Rust Zipora

**The port is complete.** As of v5.x, Zipora has feature parity with the
reference C++ topling-zip library (succinct data structures, blob stores,
FSA/tries, entropy coding) plus Rust-native additions that have no C++
counterpart (memory-safe concurrent tries, SIMD search-engine primitives,
BM25 scoring, fuzzing/Miri/TSan validation infrastructure).

## Completed subsystems

| Subsystem | Highlights | Documentation |
|-----------|-----------|---------------|
| Succinct structures | Rank/select variants, `BitVector`, Elias-Fano family (`EliasFano` / `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano`), `Rank9`, `HybridPostingList` | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| Tries / FSA | `DoubleArrayTrie` (XOR transitions), `DoubleArrayTrieMap`, `ZiporaTrie` (Patricia / Critical-Bit / LOUDS), `CsppTrie` + `ConcurrentCsppTrie` (multi-writer, EBR), `NestLoudsTrieBlobStore` | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| Blob stores | `MixedLenBlobStore`, `SimpleZipBlobStore`, `ZeroLengthBlobStore`, `ZipOffsetBlobStore`, dictionary/entropy-compressed variants; binary-compatible 80-byte header + 64-byte footer | [BLOB_STORAGE.md](BLOB_STORAGE.md) |
| Compression | Huffman (O0/O1/O2), FSE, rANS (64-bit), dictionary compression (PA-Zip / SA-IS), `StreamVByte` (SSSE3), adaptive selection | [COMPRESSION.md](COMPRESSION.md) |
| Containers | `FastVec`, `ValVec32`, `UintVecMin0`, `ZipIntVec`, `SortedUintVec`, `SmallMap`, `LruMap`, circular queues, specialized string vectors | [CONTAINERS.md](CONTAINERS.md) |
| Hash maps | `ZiporaHashMap` (strategy-based), `GoldHashMap` | [HASH_MAPS.md](HASH_MAPS.md) |
| Memory | `SecureMemoryPool`, `LockFreeMemoryPool`, five-level pool, `MmapVec` (memmap2 `MAP_SHARED`), bump/arena allocators | [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) |
| Algorithms | SIMD popcount/select/gallop/filter, LSD radix sort (ping-pong), suffix arrays (SA-IS), cache-oblivious algorithms | [ALGORITHMS.md](ALGORITHMS.md), [SIMD.md](SIMD.md) |
| Scoring | `FieldnormEncoder` (Lucene SmallFloat), `Bm25BatchScorer` (AVX2) | [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) |
| FFI | Optional C API behind the `ffi` feature, panic-safe entry points | [FFI.md](FFI.md) |

## Current state (v5.0.0)

- Build, test suite, doctests, and `clippy` (`-D warnings`) pass cleanly (`make sanity`).
- Every production `unsafe` block carries a `// SAFETY:` or `# Safety` comment (`scripts/unsafe_audit.py`).
- Adversarial validation: `cargo-fuzz` targets with committed seed corpora (`make fuzz_seed`, `make fuzz_smoke`, `make fuzz_soak`), Miri suites (`make miri_c12`, `make miri_cspp`, `make miri_pool`, `make miri_weak`), and ThreadSanitizer jobs (`make tsan_concurrency`, `make tsan_cspp`, `make tsan_pool`).

## Where to look next

- Verified benchmark numbers: [PERFORMANCE.md](PERFORMANCE.md)
- Per-topic guides: the other files in this `docs/` directory
- Historical porting log: git history of this file (pre-v4 revisions)
