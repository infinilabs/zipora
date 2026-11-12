# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.2] - 2026-11-12

### ⚠️ Data-loss advisory
- **`SortedUintVec` / `ZipOffsetBlobStore`, values ≥ 2^`sample_width`**: every release from 1.1.1 (where
  these types were introduced) through 5.0.1 built such vectors without error. `SortedUintVecBuilder::finish`
  kept only the low `sample_width` bits of each block's first value and returned `Ok`, so every value in
  that block read back wrong.
  - **Every configuration is affected:**

    | Config | Sample width | Affected values / store content |
    |---|---|---|
    | `SortedUintVecConfig::memory_optimized()` | 24-bit | ≥ 16 MiB |
    | `SortedUintVecConfig::default()` | 32-bit | ≥ 4 GiB |
    | `SortedUintVecConfig::performance_optimized()` | 40-bit | ≥ 1 TiB |
    | `ZipOffsetBlobStoreConfig::compression_optimized()` | 24-bit | ≥ 16 MiB |
    | `ZipOffsetBlobStoreConfig::default()`, `security_optimized()`, `NestLoudsTrieBlobStore`'s default | 32-bit | ≥ 4 GiB |
    | `ZipOffsetBlobStoreConfig::performance_optimized()` | 40-bit | ≥ 1 TiB |

    `ZipOffsetBlobStore` keeps its record offsets in a `SortedUintVec`, so a store whose content reached
    the size in the table was written with wrong offsets.
  - **Symptoms:** such a store still loads, because the optional file checksum covers the record bytes, not
    the offset index. Reading an affected record either fails (checksum, zstd or bounds error) or silently
    returns bytes from another position. That can happen even with per-record checksums, when the wrong
    offset lands on the start of another record (for example with fixed-size records).
  - **Detection and repair:** 5.0.2 cannot repair these files.
    - A store file smaller than 2^`sample_width` bytes is unaffected.
      `ZipOffsetBlobStore::config().offset_config.sample_width` reports the width after loading.
    - Rebuild larger stores, and any `SortedUintVec` that held values ≥ 2^`sample_width`, from the source data.
  - **New errors:** `SortedUintVecBuilder::finish` now returns `Err(InvalidData)` for such input, and so do
    `ZipOffsetBlobStoreBuilder::finish`, `BatchZipOffsetBlobStoreBuilder::finish` and
    `NestLoudsTrieBlobStore::finalize`. To build larger stores, use a wider `offset_config.sample_width`
    (up to 64), for example `ZipOffsetBlobStoreConfig::performance_optimized()`.
- **`LruPageCache`**: in 5.0.0 and 5.0.1, dropping an `LruPageCache` (or `SingleLruPageCache`) discarded
  data passed to `write` that had not yet been written back by `flush_file`, `close_file` or eviction.
  Files written that way may be missing those writes. 5.0.2 writes the dirty pages of file-backed files
  back on drop.

### Changed
- `memchr` 2.7 is now a regular dependency, used by the bounded substring searches below. It was a
  dev-dependency before; default builds already pulled it in through `serde_json` (`serde` feature).
- `sorted_intersect_adaptive`, `sorted_intersect_simd`, `sorted_intersect_count` and
  `sorted_intersect_galloping` (`zipora::algorithms`) now document their precondition: both inputs must be
  sorted strictly ascending (no duplicates), as `sorted_union_count` already required. Behaviour is
  unchanged. With duplicates or unsorted input the calls still return, but the result is unspecified: the
  SIMD kernels may report a value more than once, and results can differ between CPUs.

### Fixed
- **Silent truncation and overflow**:
  - **`EliasFano::from_sorted` / `from_sorted_u64`** (and the `HybridPostingList` constructors when they pick
    plain Elias-Fano) now panic with `EliasFano: high_len_bits … exceeds the u32 sample range` when the
    upper-bits array would need more than 2^32 bits. This needs at least about 1.43 × 10^9 values.
    - Previously the `u32` select-sample counters overflowed: debug builds panicked with an arithmetic
      overflow, and release builds stored truncated samples or looped forever.
    - These constructors return `Self`, so this is a panic, not an `Err`.
    - Serializing (`serde`) an `EliasFano` with close to 2^32 elements no longer overflows a `u32`
      counter (debug: panic; release: endless loop).
  - **`SortedUintVecBuilder::finish`** returns `Err(InvalidData)` ("block minimum … too large for
    sample_width …") when a block's first value needs more than `sample_width` bits (see the advisory).
  - **Varints:** `MemoryMappedInput`'s `DataInput::read_var_int` and `ZReorderMap`'s run-length varint reader
    (`ZReorderMap::open`, `rewind` and iteration) reject a 10th varint byte greater than `0x01` with
    `Err(InvalidData)`.
    - Previously its high bits were silently dropped and a wrong value was returned. Longer encodings were
      already rejected.
    - During iteration, such an entry ends the iteration, as other malformed entries already did.
  - **`EntropyBitOps::pack_bits(stream: &mut u64, value, offset, width)`** also rejects `offset + width > 64`
    with `Err(InvalidData)`, in addition to `width > 32` and `offset > 64`. Previously such calls panicked
    with an arithmetic overflow in debug builds and returned `Ok(())` without writing anything in release
    builds.
- **`LruPageCache`**:
  - **Drop:** dropping an `LruPageCache` (or `SingleLruPageCache`) writes back the dirty cached pages of
    file-backed files, as `flush_file` does.
    - Errors are only logged (`log::warn!`) and nothing is synced to disk. Call `flush_file` or
      `close_file` to see errors.
    - Pages of virtual files (`register_file(-1)`) are discarded, as before.
  - **Statistics:** a page load no longer also counts as a hit.
    - Misses used to call `record_miss` and also `record_hit(InitialFree)` (plus `record_hit(EvictedOthers)`
      when a page was evicted). So `total_hits`, `hit_ratio` and `miss_ratio` in `stats()` were inflated for
      `read`, `write`, `prefetch` and the methods built on them.
    - A cold read now reports a hit ratio of 0 instead of 0.5. The per-type `hit_counts` are unchanged.
  - **16 TiB limit:** `read`, `write`, `prefetch` and `invalidate_range` (and `read_batch`,
    `read_with_prefetch` and the `SingleLruPageCache` wrappers) return `Err(InvalidData)` for byte ranges
    that start at or extend past 16 TiB (2^32 pages × 4096 bytes), or whose end overflows `u64`.
    - Previously the 32-bit page id wrapped. Such reads returned an empty buffer, writes that crossed
      16 TiB were silently dropped, and writes at or above it panicked (debug) or tried to allocate about
      16 TiB (release).
    - `FileManager::write_page` now returns `Err(OutOfMemory)` instead of aborting when a virtual file
      cannot grow.
- **Containers**:
  - **`impl Extend<T> for FastVec<T>`**, and therefore `FromIterator` (`collect()`), panics with
    `FastVec capacity overflow` when an element cannot be stored (capacity overflow or allocation failure)
    instead of silently dropping it. The inherent `FastVec::extend` still returns `Result`; use it, or the
    fallible `FastVec::reserve`, to handle the failure.
  - **`EasyHashMap`** growth and `shrink_to_fit` replace the table only after every entry was copied.
    `get_or_insert` / `get_or_insert_with` return the insertion error instead of panicking (`key exists in
    map after put call`).
- **Worst-case complexity**:
  - **`CacheObliviousSort::sort`** no longer has quadratic paths.
    - Data larger than L1 and at most L2 is sorted with `slice::sort_unstable`. The old Lomuto quicksort
      with a last-element pivot took n(n−1)/2 comparisons and n levels of recursion on sorted,
      reverse-sorted and all-equal input; n = 20,000 overflowed a 2 MiB stack. This path stays unstable,
      so equal elements may come out in a different order than in 5.0.1.
    - Data that fits in L1 uses insertion sort only up to 16 elements and the stable `slice::sort` above
      that. Output is unchanged.
  - **`AdvancedRadixSort`'s MSD radix sort** (`force_strategy: Some(SortingStrategy::MsdRadix)`) finishes
    buckets whose keys share a prefix longer than 64 bytes with the stable `slice::sort` (O(n log n))
    instead of insertion sort (O(n²)). Output is unchanged.
  - **Substring search** now takes O(n + m) time on adversarial input (for example a haystack of `a`s and
    a needle `a…ab`) instead of O(n·m): about 1.4 s before for a 1 MiB haystack and a 64 KiB needle.
    Results are unchanged.
    - `zipora::string::SimdStringSearch::sse42_strstr`, `zipora::string::sse42_strstr` (every dispatch
      tier) and `FastStr::find` count the bytes spent verifying candidate matches. Once that exceeds
      `2 * haystack.len() + 256`, they finish with `memchr::memmem::find`.
    - `zipora::io::simd_memory` (`SimdStringSearch::find_pattern`, `find_pattern`, `search::sse42_strstr`
      for needles over 32 bytes, and `search::scalar_strstr`) uses `memchr::memmem`.
    - `compression::SimdPatternMatcher::find_pattern_matches` uses `memchr::memmem` for patterns over
      35 bytes, and for every pattern on its scalar tier.
- **Other fixes**:
  - `zipora::memory::align_to_cache_line(size, cache_line_size)` panics with `cache_line_size must be a
    power of two` unless `cache_line_size` is a power of two. Previously `0` underflowed (debug panic;
    release returned 0) and other values returned a misaligned size.
  - `MultiWayMerge::supports_parallel()` returns `false`. It returned `true`, but the merge never ran in
    parallel.
  - `ConcurrentLruMap` picks shards with a per-map, randomly seeded `RandomState` instead of the fixed-key
    `DefaultHasher`, so keys cannot be steered into one shard. Shard assignment, and therefore which
    entries a full shard evicts, now differs between maps and between runs.
  - `ZiporaTrie` (Patricia) `contains` uses the same lookup as `get` (no observable change today).
  - **`PaZipCompressor::compress`:** with `enable_multithreading` (the default), on inputs of at least
    1 MiB (and at least `multithreading_threshold`) it appends to `output` instead of replacing its
    contents, fixing the 5.0.1 known issue. In both this path and the sequential one,
    `stats.bytes_output` (and `compression_ratio`) now count only the bytes written by the call; they used
    to be based on `output.len()`, which included bytes already in `output`.
  - `Rans64Decoder::decode` reserves output space with `try_reserve_exact` and never beyond
    `output_length`. Allocation failures return `Err`.
  - `VersionManager::min_version()` no longer advances past the version of a live reader token. Readers
    and writers are counted as active before they read their version, and `min_version` only advances
    (never decreases) under the token-chain lock.
  - **`ThreadLocalMemoryPool::allocate`**, called from a thread-local destructor on a thread that had not
    yet used any `ThreadLocalMemoryPool`, falls back to a standalone allocation instead of aborting the
    process.
    - This needs `panic = "unwind"` (the default). The fallback catches a panic raised inside the
      `thread_local` crate, and that panic is still reported to the panic hook, once per thread.
    - With `panic = "abort"`, such an allocation still aborts.
  - `FixedCapacityMemoryPool` with `secure_clear` zeroes freed blocks with word-sized volatile writes
    (faster; same result).
  - `FseEncoder` encodes single-worker batches inline, and falls back to inline encoding instead of
    panicking when a worker thread cannot be spawned. Output is unchanged.
  - `NestLoudsTrieBlobStore::put_with_key` applies trie relocations even when the insert fails.
    `key_count`, `total_key_size` and `average_key_length` no longer count repeated puts of the same key.
- **Build and platforms**:
  - Removed from `build.rs`: unused `cargo:rustc-cfg` flags (19 fixed names plus per-feature
    `zipora_has_*` flags), the build-time `rustc` feature probes and the `-lpthread` link directive.
  - Removed dead Windows code (it never compiled, since `winapi` is not a dependency). Added a
    crate-level `compile_error!` on Windows and a "Supported platforms" section to the README.
  - CI runs the library tests natively on `aarch64`.

## [5.0.1] - 2026-11-06

### ⚠️ Data-loss advisory
- **PA-Zip reference mode** (`PaZipCompressorConfig::reference_compliant()` or any config with
  `use_reference_encoding = true`): streams written by 5.0.0 could not be decompressed. 5.0.1
  decodes those streams (they contain literal runs only), and reference-mode configs now emit the
  standard PA-Zip stream. Re-read and verify any data written with these configs.
- **PA-Zip default config, inputs ≥ 1 MiB**: `compress` duplicated earlier blocks in its output
  (multithreaded path). Such 5.0.0 streams decode (with 5.0.0 or 5.0.1) to `B1 | B1B2 | … | B1…Bn`,
  whose last part is the original input. To recover an N-byte input, decompress and keep the
  **last N bytes**. If N was not stored, derive it from the decoded length D:
  N = D − 32768 · n · (n − 1) for the unique n with 65536 · (n − 1) < N ≤ 65536 · n (try n = 16, 17, …).
  `DictZipBlobStore` usually stored such records raw (the duplicated stream was larger than the input).
- **FSE `parallel_blocks: Some(0 | 1)`**: for inputs larger than `2 × block_size`, `FseEncoder::compress`
  in 5.0.0 omitted the 1-byte `0xF5` single-block framing prefix. If `FseDecoder::decompress` rejects
  such a stream with "unknown FSE stream mode byte", prepend `0xF5` and decode again. Streams of
  smaller inputs are already framed and decode as is.
- **NestLoudsTrieBlobStore**: with DoubleArray tries (`memory_optimized()`), ~10% of keys became
  unreachable after later inserts, and `remove(id)` could delete other keys (also with Patricia,
  via duplicate keys). Rebuild such stores from source.

### Security & Soundness (two intentional breaking changes, permitted as soundness fixes)
- `ffi::types::CString` fields are now private (safe code could make `Drop` free a forged pointer).
  Use `as_ptr()`, `len()`, `is_empty()`. The struct is not declared in `include/zipora.h`, so C
  callers are unaffected.
- `CsppTrie`/`ConcurrentCsppTrie::{get_value, set_value}` now require `T: bytemuck::Pod` and
  `size_of::<T>() <= valsize` (safe code could read invalid `bool`/`char`/references or write
  padding; oversized writes corrupted the neighbouring key). Integer and byte-array value types
  (`u8`..`u64`, `[u8; N]`) are unaffected.
- Fixed use-after-free reachable from safe code in `VersionManager` tokens (dropped, moved or
  TLS-cached managers), `ThreadLocalMemoryPool` slabs and `FixedCapacityAllocation`; a safe-code
  heap overflow in `ThreadLocalMemoryPool` (block reused for a larger size class); data races and
  double allocation in `FixedCapacityMemoryPool`; misaligned `SecureMemoryPool` pointers
  (`GoldHashIdx<_, u128>` now works); non-volatile secret zeroing; unsound
  `tokio::runtime::Handle` field read; panic-safety double drops in `SmallMap::clear` and
  `AutoGrowCircularQueue::pop_bulk`; integer-to-pointer provenance loss in `MinimalSso`.

### Changed
- `PaZipCompressor::decompress` now takes `&self` (was `&mut self`). The reference-encoding flags
  (`use_reference_encoding`, `use_suffix_array_local_match`) are now no-ops: every config emits the
  standard PA-Zip stream.
- `FixedCapacityMemoryPool`: a `FixedCapacityAllocation` keeps the pool's arena alive, so dropping it
  after the pool is dropped or moved is safe. `new()` rejects `total_blocks >= u32::MAX`.
- `ThreadLocalMemoryPool`: caches are per pool instance (no longer one thread-local cache shared by all
  pools). Requests above 4096 bytes or above `arena_size / 4` are served as standalone,
  zero-initialized heap allocations (requests above `arena_size / 4` previously panicked or returned
  `Err`; 4097 bytes up to `arena_size / 4` previously came from the arena). Blocks are 16-byte
  aligned. `new()` rejects `arena_size == 0` and `arena_size > u32::MAX`.
- `SecureMemoryPool` honors `config.alignment` (minimum 8 bytes; `small_secure()` now uses 16). Fresh
  chunks are zero-initialized; recycled chunks are zeroed on return when `zero_on_free` is set (the
  default) and on allocation when `zero_on_alloc` is set. With `zero_on_free`, the final free wipes
  the payload with volatile writes.
- `NestLoudsTrieBlobStore::new` returns `Err(NotSupported)` for
  `TrieStrategy::{Louds, CompressedSparse, CriticalBit}`, whose node ids the store cannot resolve
  reliably. `len()` no longer depends on `enable_statistics`, and `remove(id)`
  deletes the key only when `id` is that key's latest record.
- `CsppTrie::{get_value, set_value}` also panic when `valpos` is not a value-slot offset (a multiple of
  `valsize`). The `Pod` bound also applies to `get_value_by_key`, `set_value_by_key` and
  `CsppTrieIterator::value`.
- `TokenManager` no longer caches writer tokens: every writer acquisition counts as a cache miss, so
  writer hit rates are 0. Cached reader tokens are returned only to the manager that issued them.
- `AdaptiveMultiDimensional::dimensions()` returns 1 for instances built with the deprecated `new_dual`.
- The optional `tokio` dependency (`async` feature) now requires `>= 1.49`.

### Known issues
- With `enable_multithreading` (the default), `PaZipCompressor::compress` replaces the contents of
  `output` for inputs ≥ 1 MiB instead of appending to it as it does for smaller inputs. Pass an empty
  buffer; this will be fixed in 5.0.2.

### Fixed
- FSE `parallel_blocks: Some(0|1)` produced undecodable streams; parallel compression spawned one
  OS thread per block; `block_size: 0` panicked.
- rANS x2/x4/x8 and contextual Huffman decoders preallocated a caller-supplied `output_length`.
- `OneWriteMultiRead` could grant concurrent writers; a read-only `TokenManager` could hand out
  a cached writer token.
- `DictZipBlobStore::get` no longer clones the whole compressor per read.

### Deprecated
- `AdaptiveMultiDimensional::new_dual` (indexed only the first bit vector; use
  `RankSelectMixedIL256`).

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
