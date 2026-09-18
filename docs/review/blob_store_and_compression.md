# `blob_store` & `compression/dict_zip` Review (`C2`, `C2.1`, + Stage 2 Carry-overs `S2-F1`, `S2-F2`, `S3-R1`, `S3-R2`) — 2026-09-18

## Findings

### 1. `FseDecoder::decompress` zero-bit / 1-symbol fixed-point hang (`1 exec/s` fuzz DoS) — HIGH (`[C1, S2-F1]`)
- **Location**: `src/entropy/fse.rs:483-569`, `src/entropy/fse.rs:733-855`
- **Expected failure**: A ≤22-byte input claiming a large `original_size` with a 1-symbol header or a zero-bit transition fixed point (`state < freq_0` with `start == 0`) loops `original_size` times without consuming payload bytes, dropping `fuzz_fse_decompress` to `1 exec/s`.
- **Repro test**: `entropy::fse::tests::test_fse_malformed_zero_payload_and_fixed_point_streams_rejected_fast`
- **Evaluation**: **Confirmed** — on base (`19abac2`), the test took 2.14s and returned `Ok` on a 20-byte zero-payload header claiming 100 MiB (`fuzz_fse_decompress` ran at 1 exec/s). Fixed by initializing FSE encoder state at `INITIAL_STATE = 4096` (`TOTFREQ`), validating 1-symbol streams in O(1), enforcing 4-byte block alignment and bit-capacity bounds, and checking mid-stream/end-of-stream exhaustion (`state == INITIAL_STATE && byte_pos == 0`). Fuzz speed increased from **1 exec/s** to **158,942 exec/s**.
- **Fixed**: `d6ad3e3`

### 2. `OptimizedDictionaryCompressor::compress` rolling-hash desync after match skips — MEDIUM (`[C1, S2-F2]`)
- **Location**: `src/entropy/dictionary.rs:791-860`
- **Expected failure**: `RollingHash::roll` was called assuming `pos` advanced by 1 byte even when `pos` jumped forward after emitting a multi-byte match or after a bloom-filter miss, corrupting subsequent hash-chain lookups.
- **Repro test**: `entropy::dictionary::tests::test_optimized_dictionary_rolling_hash_resync_after_skip`
- **Evaluation**: **Confirmed** — tracked `last_hashed_pos: Option<usize>` and called `rolling_hash.init()` whenever `last_hashed_pos != Some(pos - 1)`.
- **Fixed**: `b053623`

### 3. `HuffmanBlobStore::get` returned raw compressed bytes & `RansBlobStore`/`DictionaryBlobStore` skipped compression — HIGH (`[C2, D1]`)
- **Location**: `src/blob_store/entropy.rs:95-498`
- **Expected failure**: After `HuffmanBlobStore::train()` + `put(data)`, `get(id)` returned `self.records[index].clone()` without decoding, silently returning compressed bytes instead of original payload; `size()` returned compressed byte count; `RansBlobStore` and `DictionaryBlobStore` stored uncompressed payloads unconditionally.
- **Repro test**: `blob_store::entropy::tests::test_trained_entropy_blob_stores_roundtrip_and_size`
- **Evaluation**: **Confirmed** — RED-verified on base (`get()` returned compressed bytes != input). Implemented 5-byte frame `[flag: u8][uncompressed_len: u32 LE][payload]` with round-trip decompression and logical uncompressed `size()` across all three entropy blob stores.
- **Fixed**: `53d05d2`

### 4. `MmapVecHeader` native-struct pointer cast & `calculate_file_size` capacity multiplication overflow — HIGH (`[C2.1, C9.1]`)
- **Location**: `src/memory/mmap_vec.rs:285-350`, `src/memory/mmap_vec.rs:942-955`
- **Expected failure**: `MmapVecHeader` was read/written via `*(ptr as *const MmapVecHeader)` without endianness/word-size flags, and `MmapVec::calculate_file_size` computed `capacity * size_of::<T>()` without overflow checking (panicking in debug or wrapping in release to bypass file-length validation).
- **Repro test**: `test_mmap_vec_open_rejects_overflowing_header_capacity` (`tests/mmap_soundness_test.rs`) and `test_golden_mmap_vec_header_v2_le_bytes` (`tests/format_spec_golden_tests.rs`)
- **Evaluation**: **Confirmed** — replaced struct cast with explicit 80-byte little-endian serialization (`to_le_bytes`/`from_le_bytes`), bumped `MMAP_VEC_VERSION` to `2` with `MMAP_VEC_FLAGS = 0x0011`, and used `checked_mul`/`checked_add` in `calculate_file_size`.
- **Fixed**: `62540be`

### 5. `ZReorderMap::read_entry` unbounded sequence length & underflowing decreasing runs — HIGH (`[C2, D10.1]`)
- **Location**: `src/blob_store/reorder_map.rs:240-292`
- **Expected failure**: A crafted `ZReorderMap` file with a huge positive/negative sequence run length in `read_entry` looped billions of times or underflowed `current_val` past `0`.
- **Repro test**: `blob_store::reorder_map::tests::test_zreorder_map_rejects_oversized_and_underflowing_sequences`
- **Evaluation**: **Confirmed** — bounded `seq_len` against `remaining_elements` (`size - mapping.len()`) and verified `first_value + 1 >= seq_len` for decreasing runs.
- **Fixed**: `150c05d`

### 6. `BatchZipOffsetBlobStoreBuilder` concatenated batched records into a single record — HIGH (`[C2]`)
- **Location**: `src/blob_store/zip_offset_builder.rs:580-635`
- **Expected failure**: `BatchZipOffsetBlobStoreBuilder::add_record` appended `0` as a separator into `batch_buffer` and `flush_batch` passed the entire concatenated `batch_buffer` to `self.inner.add_record(&self.batch_buffer)` as a single record.
- **Repro test**: `blob_store::zip_offset_builder::tests::test_batch_builder_functionality`
- **Evaluation**: **Confirmed** — tracked individual record lengths in `batch_lengths: Vec<usize>` and emitted each record separately in `flush_batch`.
- **Fixed**: `150c05d`

### 7. `HybridCompressor` incompressible/short-input round-trip corruption — HIGH (`[C2]`)
- **Location**: `src/compression/mod.rs:265-330`
- **Expected failure**: When no candidate compressor reduced size (`compressed.len() < best_size`, always false for short/incompressible inputs), `HybridCompressor::compress` left `best_algorithm = 0` (`HuffmanCompressor`) with `best_result = data.to_vec()`, so `decompress()` fed raw uncompressed bytes to `HuffmanCompressor::decompress()` and failed with `InvalidData`.
- **Repro test**: `compression::tests::test_hybrid_compressor_incompressible_and_short_input_roundtrip`
- **Evaluation**: **Confirmed** — RED-verified on base (`Huffman compressed data too short`). Added `0xFF` uncompressed pass-through algorithm tag handled by `HybridCompressor::decompress`.
- **Fixed**: `772fcab`

### 8. `PaZipCompressor::decompress` silently ignored truncated instructions and unbounded `Far3Long` lengths — HIGH (`[C2, D10.1]`)
- **Location**: `src/compression/dict_zip/compressor.rs:625-830`
- **Expected failure**: Truncated match instructions (`Global`, `RLE`, `NearShort`, `Far*`) returned `Ok(new_pos)` when `new_pos + hdr_len > input.len()`, causing `decompress()` to re-interpret payload bytes as subsequent instruction opcodes; unknown opcodes silently defaulted to `Literal`; `Far3Long` allowed `u32::MAX` backreference copy loops.
- **Repro test**: `compression::dict_zip::compressor::tests::test_pa_zip_decompress_rejects_truncated_and_oversized_instructions`
- **Evaluation**: **Confirmed** — RED-verified on base (`decompress(&[1, 0, 1, b'X'])` returned `Ok` and emitted `b"X"`). Replaced all instruction reads with checked `.first_chunk::<N>()` returning `Err(ZiporaError::invalid_data(..))` on truncation/unknown opcode/oversized match length.
- **Fixed**: `772fcab`

## Debt Closed in This Pass (`D1`, `D10`, `D10.1`)
- **`D10.1` (`scripts/index_audit.py`)**:
  - `blob_store`: `50` $\rightarrow$ **`0`**
  - `compression`: `133` $\rightarrow$ **`0`**
  - Total direct indexing across all scoped subsystems: `189` $\rightarrow$ **`6`** (only `succinct_load` remains for Stage C5).
- **`D10` (`scripts/unwrap_audit.py`)**:
  - Production `.unwrap()`: **`0`**
  - Production `.expect()`: `154` $\rightarrow$ **`145`** (eliminated all 9 `.expect()` sites in `src/blob_store/`).
- **`D1` (`scripts/api_honesty.py`)**:
  - Total honesty markers: `163` $\rightarrow$ **`123`** (eliminated all 12 markers in `src/blob_store/`, all 25 markers in `src/compression/`, all 1 marker in `src/config/`, and 2 `TODO: Implement` markers in `ZoSortedStrVec` by implementing `ZOSV` v1 binary serialization).
