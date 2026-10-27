# 1-Hour Parallel Fuzz Soak Report (`plan.md` Phase B `B2.1` / `S52-R2` / `S53-R1`)

## 1. Execution Configuration & Exit-Code Gate

- **Target**: `make fuzz_soak` (`Makefile:483-492`)
- **Parameters**: `-max_total_time=3600 -rss_limit_mb=1024 -malloc_limit_mb=1024 -print_final_stats=1` across all 11 libFuzzer targets in `fuzz/fuzz_targets/`.
- **Seed Corpora**: `fuzz/corpus/<target>/seed_0001.bin` generated deterministically by `make fuzz_seed` (`ZIPORA_WRITE_FUZZ_SEEDS=1 cargo test --locked --test format_spec_golden_tests test_b2_1_fuzz_seed_corpora_and_structure_aware_targets`).
- **Exit-Code Enforcement (`S52-R2`)**: `make fuzz_soak` collects the PID of every background fuzzer (`pids="$$pids $$!"`) and checks every child process exit status (`for p in $$pids; do wait $$p || fail=1; done; exit $$fail`) instead of calling a bare `wait` (which returns `0` in POSIX `sh` even when a background job fails).

---

## 2. Defects Found & Fixed (`S52-R2` & `S53-R1`)

During the initial 1-hour parallel soak across all 11 targets (`229,033,298` executions across the 10 targets other than `fuzz_zip_offset_load`, and `360,359,994` total executions when summed with the `131,326,696`-execution `fuzz_zip_offset_load` run), `fuzz_zip_offset_load` uncovered an arithmetic overflow in `SortedUintVec::read_from`, and subsequent `S53-R1` review identified an unbounded header-length allocation when `size`, `index_len`, and `data_len` were mutually consistent on a truncated input:

- **Root Cause (`src/blob_store/sorted_uint_vec.rs:384-474`, `src/blob_store/zip_offset.rs:618-626`)**:
  1. In `SortedUintVec::read_from`, `min_index_len = (num_blocks * config.sample_width as usize).div_ceil(8)` and `min_data_len = (size * config.offset_width as usize).div_ceil(8)` multiplied an untrusted `size` (up to `usize::MAX`) by `offset_width` (`8..=32`) or `num_blocks` by `sample_width` (`16..=64`) without `checked_mul`, panicking with `attempt to multiply with overflow` when a valid `ZipOffsetBlobStore` seed had its element count mutated to a large 64-bit integer (`crash-2950c6e3b040e890a337af10eedb41f09adc3ffd`).
  2. In `SortedUintVec::num_blocks`, `(self.size + self.config.block_mask()) >> self.config.log2_block_units` could overflow when `self.size` was near `usize::MAX`.
  3. In `SortedUintVec::get2` and `SortedUintVec::get_unchecked`, `index + 1` and `block_min + delta as u64` used unchecked addition.
  4. In `ZipOffsetBlobStore::get_record_impl`, `let mut record_len = (end_offset - start_offset) as usize;` was evaluated before checking `end_offset < start_offset`, which would panic on non-monotonic corrupted offset tables.
  5. (`S53-R1`) When a crafted ~300-byte input declared internally consistent huge `size` (`2^43`), `index_len` (`512 GiB`), and `data_len` (`16 TiB`), `SortedUintVec::read_from` and `ZipOffsetBlobStore::load_from_reader` pre-allocated `index_len` / `content_bytes` before reading from the stream, aborting with `memory allocation of 549755813888 bytes failed`.
- **Fixes (`c964b5e` + `6df2780`)**:
  - Replaced the unchecked multiplications and additions in `SortedUintVec::read_from`, `SortedUintVec::num_blocks`, `SortedUintVec::get2`, and `SortedUintVec::get_unchecked` with `checked_mul`, `div_ceil`, `saturating_add`, and `checked_add`, returning `Err(ZiporaError::invalid_data(...))` on overflow.
  - Checked `end_offset < start_offset || start_offset > self.content.len() as u64 || end_offset > self.content.len() as u64` in `ZipOffsetBlobStore::get_record_impl` before computing `end_offset - start_offset`.
  - Bounded every section load in `SortedUintVec::read_from_bounded` and `ZipOffsetBlobStore::load_from_reader` by the bytes actually remaining in the reader via 8 KiB chunk streaming (`SortedUintVec::read_bounded_into_fast_vec`) and outer `offsets_bytes` cross-checking (`6df2780`), and added `-malloc_limit_mb=$(FUZZ_RSS_LIMIT_MB)` (`1024`) to `fuzz_smoke` and `fuzz_soak`.
  - After applying both `c964b5e` and `6df2780`, all 11 fuzz targets (including the 10 targets that had originally run prior to the `SortedUintVec` fixes) were re-built and re-soaked in parallel for a full 1 hour (`3,601s` each).

---

## 3. Post-Fix 1-Hour Re-Soak Summary Across All 11 Targets (`554,862,880` Total Executions, `oom=0, timeout=0, crash=0`)

| Fuzz Target | Status | Duration (s) | Total Executions (`stat::number_of_executed_units`) | Avg `exec/s` | Edge Coverage (`cov`) | Features (`ft`) | Final Corpus (`corp`) | New Units Added | Slowest Unit (s) | Peak RSS (MiB) | Crashes / OOMs / Timeouts |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `fuzz_zip_offset_load` | `DONE` | `3,601` | `153,442,810` | `42,611` | `562` | `1,125` | `160 / 43 Kb` | `520` | `0` | `163` | `oom=0, timeout=0, crash=0` |
| `fuzz_simple_zip_build` | `DONE` | `3,601` | `28,088,174` | `7,800` | `315` | `1,361` | `223 / 63 Kb` | `1,712` | `0` | `534` | `oom=0, timeout=0, crash=0` |
| `fuzz_mixed_len_build` | `DONE` | `3,601` | `62,524,257` | `17,363` | `434` | `1,158` | `133 / 42 Kb` | `1,064` | `0` | `545` | `oom=0, timeout=0, crash=0` |
| `fuzz_huffman_decode` | `DONE` | `3,601` | `104,344,216` | `28,976` | `253` | `577` | `74 / 1,760 b` | `174` | `0` | `432` | `oom=0, timeout=0, crash=0` |
| `fuzz_rans_decode` | `DONE` | `3,601` | `33,273,448` | `9,240` | `93` | `198` | `47 / 1,127 b` | `126` | `0` | `392` | `oom=0, timeout=0, crash=0` |
| `fuzz_fse_decompress` | `DONE` | `3,601` | `3,522,991` | `978` | `230` | `496` | `153 / 16 Kb` | `622` | `0` | `341` | `oom=0, timeout=0, crash=0` |
| `fuzz_double_array_trie` | `DONE` | `3,601` | `21,638,483` | `6,009` | `521` | `2,591` | `600 / 121 Kb` | `5,565` | `0` | `501` | `oom=0, timeout=0, crash=0` |
| `fuzz_uint_vec` | `DONE` | `3,601` | `99,445,851` | `27,616` | `61` | `147` | `31 / 4,828 b` | `77` | `0` | `436` | `oom=0, timeout=0, crash=0` |
| `fuzz_hybrid_posting_list` | `DONE` | `3,601` | `25,924,093` | `7,199` | `1,177` | `5,917` | `878 / 509 Kb` | `5,721` | `0` | `516` | `oom=0, timeout=0, crash=0` |
| `fuzz_double_array_trie_map` | `DONE` | `3,601` | `19,396,450` | `5,386` | `819` | `3,735` | `457 / 154 Kb` | `5,307` | `0` | `519` | `oom=0, timeout=0, crash=0` |
| `fuzz_blob_store_roundtrip` | `DONE` | `3,601` | `3,262,107` | `905` | `1,162` | `4,328` | `417 / 240 Kb` | `2,664` | `0` | `167` | `oom=0, timeout=0, crash=0` |
