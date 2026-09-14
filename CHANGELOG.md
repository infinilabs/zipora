# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.2.0] - 2026-09-14

### BREAKING / FORMAT CHANGES

- **Huffman Compressed Stream Format**:
  Fixed an inverted priority queue in the Huffman code generator (`Ord` reversal combined with `Reverse` created a max-heap, forcing every tree into a depth-255 chain and causing Order-1 compression to always fall back to uncompressed 8-bit literals). Bitstreams generated with v4.2.0 achieve true compression and are incompatible with decoders expecting the malformed v4.1.0 streams.
- **LockFreeMemoryPool Layout**:
  Added an 8-byte out-of-band `AtomicU32` block header (`BLOCK_HEADER`, `link_slot()`) to every allocated block. The Treiber free-list next-pointer is stored in this header rather than in user memory, eliminating concurrent read races with caller memory writes. User allocations are offset by 8 bytes.
- **ZipOffsetBlobStore Record Size**:
  `ZipOffsetBlobStore::size()` now returns the uncompressed length of the record rather than the compressed zstd frame size. The builder now uses `zstd::bulk::compress` so frame headers reliably include the uncompressed content size.

### Added

- **Linear DoubleArrayTrie Construction**:
  Implemented `DaFreeList`, threading unused cells into a doubly-linked list (`base` predecessor, `check` successor) with a high-water mark fallback. Construction of `ZiporaTrie` with DoubleArray strategy is now linear (random 20k keys reduced from 5.9 s to 66 ms, scaling ~2x per doubling up to 160k+ keys).
- **Criterion Benchmark Declarations**:
  Explicitly declared `harness = false` across all 30 benchmark targets in `Cargo.toml` so `cargo bench` correctly executes criterion runners instead of default libtest stubs.

### Fixed

- **Memory Management (`LockFreeMemoryPool`)**:
  - Fixed a critical heap corruption bug where fast bins carved blocks at request size but recycled them by size class, resulting in an 8-byte overrun into adjacent live allocations upon reuse.
  - Eliminated data race in Treiber stack pop where concurrent user writes overlapped with atomic free-list pointer reads; verified clean under ThreadSanitizer (`make tsan_pool`).
- **Blob Stores (`ZipOffsetBlobStore` & `NestLoudsTrieBlobStore`)**:
  - Fixed `ZipOffsetBlobStore::get` dispatch for `checksum_level = 1` which previously fell through to uncompressed reads and returned raw zstd frames.
  - Removed in-memory `pending_blobs` `HashMap` workaround in `NestLoudsTrieBlobStore`; reads now serve directly from the sealed compressed store after `finalize()`.
  - `NestLoudsTrieBlobStore::get_by_prefix` now propagates read errors instead of silently dropping keys.
  - Fixed builder `finish()` which previously discarded records due to missing offset table serialization and a checksum algorithm mismatch (31-multiply hash vs CRC32C).
- **Radix Sort (`AdvancedRadixSort`)**:
  - Fixed out-of-bounds panic during LSD radix sort on 64-bit keys. The AVX2 digit counting kernel now performs native 64-bit vector shifts (`_mm256_srl_epi64` / `_mm256_and_si256`) instead of truncating keys to 32 bits, correctly handling shifts >= 32 and eliminating auxiliary heap vector allocations.
- **Tries (`ZiporaTrie` / `DoubleArrayTrie`)**:
  - `stats()` now accurately reports active states instead of counting empty free-list slots.
  - `shrink_to_fit()` now unlinks and truncates trailing free cells, lowering the high-water mark while preserving insertability.
  - Fixed `state_move` out-of-bounds access on freed states.
  - Fixed Patricia trie node recycling on removal to prevent node leaks.
- **Succinct Data Structures**:
  - Fixed Elias-Fano boundary handling for `next_geq(u64::MAX)`.
  - Fixed 32-bit overflow in `is_run_heavy` calculation for Clustered Elias-Fano.
  - Centralized `select_in_word` implementations with Zen 1/2 AMD PDEP safety checks (`has_fast_bmi2`).
- **Containers**:
  - `FastVec`: Replaced `memcmp`-based `PartialEq` with element-wise comparisons to ensure correct IEEE 754 float semantics (`-0.0` vs `+0.0`, `NaN`).
  - `ZiporaHashMap`: Enforced table resize at the configured load factor instead of 100% capacity.
  - `LruMap`: Resolved allocate-then-evict race condition under concurrency.
  - `SortedUintVec`: Fixed multi-word bitfield packing and unpacking for fields wider than 64 bits.
  - `ZipIntVec`: Resolved arithmetic overflow when `min == max`.
- **String & SIMD Search**:
  - Fixed SSE4.2 string search where `_SIDD_CMP_EQUAL_ORDERED` was mis-encoded as `_SIDD_CMP_EQUAL_EACH` (0x08 vs 0x0C), restoring full functionality to SSE4.2 `strchr` and `strstr`.
- **Entropy & Compression**:
  - `FseEncoder`: Added validation for uncompressed size headers to prevent multi-gigabyte allocation denial-of-service from malformed input. Reject truncated trailing blocks with explicit error.
  - `StreamVByte`: SSSE3 shuffle-table decoding with bounded reservation size.
- **Concurrency & System**:
  - `instance_tls`: Unified duplicate global registries, ensuring thread-local IDs are recycled upon drop.
  - `cpu_features`: Corrected XCR0 OS-support detection for AVX/AVX-512 features.
