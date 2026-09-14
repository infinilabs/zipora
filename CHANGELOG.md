# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
