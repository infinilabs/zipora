# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-10-30

### BREAKING CHANGES

- **Rust Edition & MSRV**:
  Updated crate to Rust `edition = "2024"` with `rust-version = "1.95"`.
- **API Honesty & Surface Cleanup**:
  Removed unimplemented stub types and presets (`NestTrieDawg`, unimplemented trie/hash-map strategy presets), restricted internal `simd::isa` to `pub(crate)`, and standardized `SuffixArray` construction on linear-time SA-IS.
- **Succinct & Trie Memory Layouts**:
  - `DoubleArrayTrie` now uses a `12 B/state` layout incorporating free-list/output links (`DoubleArrayTrieMap<V>` stores values in a dense side table).
  - `RankSelectInterleaved256` unpadded `InterleavedLine` from `64 B` to `40 B` (`25.00%` rank-only overhead, `31.25%` with `select1` + `select0` caches, supporting up to $2^{40}$ bits).
  - `EliasFano` serialization (`EliasFanoWire`) clamps deserialized lengths against `high_bits` popcount and `low_bits` capacity to reject forged wire headers.
- **Optional `zstd` Upgrade**:
  Upgraded optional `zstd` dependency from `0.13` to `0.14.3` (`zstd-safe 8.2.0`, `zstd-sys 2.1.2+zstd.1.5.7`; no `zstd` types are re-exported in Zipora's public API).

### Added

- **Concurrent & Value-Mapped Tries**:
  Added `CsppTrie` (Compressed Sparse Parallel Patricia trie), `ConcurrentCsppTrie` (multi-writer, lock-free readers with `crossbeam-epoch` reclamation), and `DoubleArrayTrieMap<V>`, plus lazy prefix and fuzzy edit-distance iterators.
- **Search-Engine & BM25 Primitives**:
  Added `FieldnormEncoder` (Lucene SmallFloat `1.0 B/doc` fieldnorms), `Bm25BatchScorer` (AVX2/AVX-512 SIMD batch scoring with software prefetch), `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano` cursor `advance_to_index`, and `HybridPostingList` Block-Max WAND cursor primitives.
- **Entropy & Integer Codecs**:
  Added 64-bit `Rans64Encoder` / `Rans64Decoder` (`X1`/`X2`/`X4`/`X8` interleaved streams), SSSE3 `StreamVByte` quad encode/decode with fused delta prefix sums, and SSSE3 `GroupVarint`.
- **Verification & Regression Infrastructure**:
  Added 11 `cargo-fuzz` targets with committed seed corpora and a 1-hour parallel soak (`oom=0, timeout=0, crash=0`), Miri strict-provenance and weak-memory test suites, ThreadSanitizer concurrency/pool/CSPP gates, executable markdown doctests (`89/89` across `README*.md` and `docs/*.md`), and native per-binary benchmark gates (`core_bench`, `competitor_bench`, `e2e_search_bench`).

### Changed

- **`ZiporaHashMap` (SwissTable Group Probing)**:
  Rebuilt `StandardTable` with 16-way SSE2 control-byte probing (`17 B/slot` `MaybeUninit<(K, V)>`) and panic-safe `ClearSlotsGuard`/`ClearInlineGuard` resize/drop paths, eliminating `u64` space overhead relative to `hashbrown`/`std::collections::HashMap`.
- **`LockFreeMemoryPool` (Thread-Local Magazines)**:
  Added an $O(1)$ `SIZE_CLASS_LUT` and 16-slot per-bin thread-local magazines (`ThreadMagazine`) with `Arc`/`Weak` pool-switch, thread-exit, and `Drop` flushing plus double-free tag validation prior to magazine caching.
- **`DoubleArrayTrie::build_from_sorted`**:
  Replaced recursive incremental insertion with a top-down explicit-stack bulk range-partitioning builder using `search_head` free-slot scanning and zero `relocate` calls.
- **Entropy Coding (`Huffman`, `ContextualHuffmanEncoder`, `FSE`, `rANS`)**:
  Added 11-bit (`O0`) and 10-bit (`O1`/`O2`) fast decode LUTs with 64-bit word bit-flushing, bounded `ContextualEncodeCache` dense tree tables, 4-state lockstep 64-bit rANS (`rcp40` + `6 B` decode entries), and `6 B` `FseDecSlot` tables.
- **`AdvancedRadixSort` & `SuffixArray` & SIMD String Search**:
  Added cache-conscious MSB+L2 hybrid in-cache LSD radix sort with panic-safe `(K, V)` ping-pong buffers, packed `WordBitVec` + `Bucket2Table` galloping `SuffixArray::search_range`, and 4x-unrolled 128-byte AVX2 `strchr`/`strstr`.
- **Smart-Pointer & Bulk `Pod` Serialization (`io::smart_ptr`, `io::data_output`)**:
  Replaced recursive pointer-graph serialization/deserialization with iterative traversal bounded on `2 MiB` stacks, single-probe `PtrHasher` ID registration, and bulk little-endian `Pod` slice writes.

### Security & Soundness Fixes

- **Dependency Advisories (`make audit`)**:
  Upgraded `zstd` to `0.14.3` (`zstd-safe 8.2.0`, `zstd-sys 2.1.2+zstd.1.5.7`), resolving 4 vulnerabilities (`RUSTSEC-2026-0391`, `RUSTSEC-2026-0398`, `RUSTSEC-2026-0403`, `RUSTSEC-2026-0410`) and 10 `zstd-safe` `unsound` advisories (`RUSTSEC-2026-0395..0402`, `0404..0406`); upgraded `tokio` to `>= 1.54.1` (`RUSTSEC-2026-0386`).
- **Untrusted Input Bounds (`blob_store` & `entropy`)**:
  Bounded all `SortedUintVec::read_from` and `ZipOffsetBlobStore` header-driven allocations by remaining reader bytes to prevent crafted-header OOM/overflow DoS. Added prefix-free / dyadic Kraft weight validation in `HuffmanTree::deserialize`, single-symbol bit verification in `ContextualHuffmanEncoder`, and non-empty stream checks when `output_size > 0` across `X1`–`X8` and single-threaded decoders.
- **Filesystem & Memory Safety**:
  Hardened `MmapVec` against symlink attacks (`O_NOFOLLOW`) with cleanup on failed map, verified `100%` `// SAFETY:` / `# Safety` documentation across all `1,427` production `unsafe` blocks (`scripts/unsafe_audit.py`), and eliminated unchecked direct slice indexing in production code (`scripts/index_audit.py`).

### Known Limitations / Untested Hardware Paths

- **AVX-512 (`avx512` feature on `x86_64`)**:
  AVX-512 code paths compile cleanly on `x86_64` and are gated at runtime behind `is_x86_feature_detected!` + OS `XCR0` state checks (`src/simd/cpu_features.rs`). However, the reference verification host (`bindiego`, AMD EPYC 7B13 Zen 3) supports `avx`/`avx2` without `avx512f`, and Miri runs with `-C target-cpu=haswell`; the AVX-512 instruction bodies have **never been executed on AVX-512 hardware or an emulator**. To restrict runtime dispatch strictly to hardware-verified AVX2/SSE/scalar paths on AVX-512 machines, depend on Zipora with `default-features = false` and omit `"avx512"` (`features = ["simd", "mmap", "zstd", "serde", "lz4", "async"]`).
- **ARM64 / AArch64 (`aarch64-unknown-linux-gnu`)**:
  Cross-target support for `aarch64-unknown-linux-gnu` is verified via `cargo clippy --target aarch64-unknown-linux-gnu --lib -- -D warnings` (`make miri_weak`); it is **compiled only and has never been executed on ARM hardware or QEMU**.

## [4.2.0] - 2026-09-14

### BREAKING CHANGES

- **ZipOffsetBlobStore Record Size**:
  `ZipOffsetBlobStore::size()` now returns the uncompressed length of the record rather than the compressed zstd frame size. The builder now uses `zstd::bulk::compress` so frame headers reliably include the uncompressed content size.
- **NestLoudsTrieBlobStore Error Propagation**:
  `NestLoudsTrieBlobStore::get_by_prefix` now propagates read errors instead of silently dropping keys.

### Changed

- **LockFreeMemoryPool Out-of-Band Header**:
  Added an 8-byte out-of-band `AtomicU32` block header (`BLOCK_HEADER`, `link_slot()`) to every allocated block. The Treiber free-list next-pointer is stored in this header rather than in user memory, eliminating concurrent read races with caller memory writes. User allocations are offset by 8 bytes (fewer blocks per `memory_size`).
- **Huffman Order-1 Compression**:
  Fixed an inverted priority queue in the Huffman code generator (`Ord` reversal combined with `Reverse` created a max-heap, forcing every tree into a depth-255 chain and causing Order-1 compression to always fall back to uncompressed 8-bit literals). 4.2.0 Order-1 output is now compressed; 4.1.0 emitted a byte copy; streams remain cross-decodable as serialized trees carry explicit code bits.
- **Linear DoubleArrayTrie Construction**:
  Implemented `DaFreeList`, threading unused cells into a doubly-linked list (`base` predecessor, `check` successor) with a high-water mark fallback. Construction of `ZiporaTrie` with DoubleArray strategy is now linear (random 20k keys reduced from 5.9 s to 66 ms, scaling ~2x per doubling up to 160k+ keys).
- **Criterion Benchmark Declarations**:
  Explicitly declared `harness = false` across all 30 benchmark targets in `Cargo.toml` so `cargo bench` correctly executes criterion runners instead of default libtest stubs.
- **Note on Commit `8511e20`**:
  Commit `8511e20` was labelled `[C8]` in its subject; it is tracked in `plan.md` as `[A1.8]` (pre-push finding).

### Fixed

- **Memory Management (`LockFreeMemoryPool` & `SecureMemoryPool`)**:
  - `LockFreeMemoryPool`: Fixed a critical heap corruption bug where fast bins carved blocks at request size but recycled them by size class, resulting in an 8-byte overrun into adjacent live allocations upon reuse.
  - `LockFreeMemoryPool`: Eliminated data race in Treiber stack pop where concurrent user writes overlapped with atomic free-list pointer reads; verified clean under ThreadSanitizer (`make tsan_pool`).
  - `SecureMemoryPool`: Fixed recycled-chunk zeroing (`zero_on_free`) and footer alignment padding (`2707e30`, `6ffcfdb`).
- **Blob Stores (`ZipOffsetBlobStore`, `NestLoudsTrieBlobStore`, `DictZip`)**:
  - `ZipOffsetBlobStore`: Fixed `get` dispatch for `checksum_level = 1` which previously fell through to uncompressed reads and returned raw zstd frames.
  - `NestLoudsTrieBlobStore`: Removed in-memory `pending_blobs` `HashMap` workaround; reads now serve directly from the sealed compressed store after `finalize()`.
  - `ZipOffsetBlobStore`: Fixed builder `finish()` which previously discarded records due to missing offset table serialization and a checksum algorithm mismatch (31-multiply hash vs CRC32C).
  - `DictZip`: Fixed entropy-layer decode (`b4a0848`).
  - `ReorderMap`: Rejects zero-length entries instead of corrupting offsets (`7494a6e`).
- **Radix Sort (`AdvancedRadixSort`)**:
  - Fixed out-of-bounds panic during LSD radix sort on 64-bit keys (`8511e20`, tracked as `[A1.8]`). The AVX2 digit counting kernel now performs native 64-bit vector shifts (`_mm256_srl_epi64` / `_mm256_and_si256`) instead of truncating keys to 32 bits, correctly handling shifts >= 32 and eliminating auxiliary heap vector allocations.
- **Tries (`ZiporaTrie` / `DoubleArrayTrie`)**:
  - `stats()` now accurately reports active states instead of counting empty free-list slots.
  - `shrink_to_fit()` now unlinks and truncates trailing free cells, lowering the high-water mark while preserving insertability.
  - Fixed `state_move` out-of-bounds access on freed states (`3c06c97`).
  - Fixed Patricia trie node recycling on removal to prevent node leaks (`a51cef3`).
- **Succinct Data Structures**:
  - Fixed Elias-Fano boundary handling for `next_geq(u64::MAX)`.
  - Fixed 32-bit overflow in `is_run_heavy` calculation for Clustered Elias-Fano (`76b5347`).
- **Containers**:
  - `FastVec`: Replaced `memcmp`-based `PartialEq` with element-wise comparisons to ensure correct IEEE 754 float semantics (`-0.0` vs `+0.0`, `NaN`) (`7f9f955`).
  - `FastVec`: Implemented `ExactSizeIterator` and `FusedIterator` for `IntoIter` (`9d69809`).
  - `ZiporaHashMap`: Enforced table resize at the configured load factor instead of 100% capacity (`bb0d937`).
  - `LruMap`: Resolved allocate-then-evict race condition under concurrency (`a3d2b1b`).
  - `SortedUintVec`: Fixed multi-word bitfield packing and unpacking for fields wider than 64 bits (`098eb6c`).
  - `ZipIntVec`: Resolved arithmetic overflow when `min == max` (`5b3e002`).
  - `UintVecMin0`: Fixed in-place re-layout on grow (`a949bdf`).
  - `SetOps`: Dedupes with the caller's comparator rather than default equality (`b1a9d1f`).
- **String & SIMD Search**:
  - Fixed SSE4.2 string search where `_SIDD_CMP_EQUAL_ORDERED` was mis-encoded as `_SIDD_CMP_EQUAL_EACH` (0x08 vs 0x0C), restoring full functionality to SSE4.2 `strchr` and `strstr` (`13bdb90`, `11d1f39`).
- **I/O & Variable-Length Integers**:
  - `var_int`: Rejects overflowing 10-byte encodings instead of wrapping (`7c86e6e`).
  - `VectoredIO` & `StreamBuffer`: Handled short reads and short writes correctly (`4f96392`, `42cdc89`).
- **Entropy & Compression**:
  - `FseEncoder`: Added validation for uncompressed size headers to prevent multi-gigabyte allocation denial-of-service from malformed input (`21c8e52`). Reject truncated trailing blocks with explicit error (`721c77a`).
  - `StreamVByte` / `GroupVarint`: Added input-bounded reservation size in raw decode to prevent allocation DoS (`d1fc85b`).
- **Concurrency & System**:
  - `instance_tls`: Unified duplicate global registries, ensuring thread-local IDs are recycled upon drop (`fce9dd2`).
  - `cpu_features`: Corrected XCR0 OS-support detection for AVX/AVX-512 features (`c3b1539`).
