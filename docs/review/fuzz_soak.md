# 1-Hour Parallel Fuzz Soak Report (`plan.md` Phase B `B2.1` / `S52-R2`)

## 1. Execution Configuration & Exit-Code Gate

- **Target**: `make fuzz_soak` (`Makefile:482-491`)
- **Parameters**: `-max_total_time=3600 -rss_limit_mb=1024 -print_final_stats=1` across all 11 libFuzzer targets in `fuzz/fuzz_targets/`.
- **Seed Corpora**: `fuzz/corpus/<target>/seed_0001.bin` generated deterministically by `make fuzz_seed` (`ZIPORA_WRITE_FUZZ_SEEDS=1 cargo test --locked --test format_spec_golden_tests test_b2_1_fuzz_seed_corpora_and_structure_aware_targets`).
- **Exit-Code Enforcement (`S52-R2`)**: `make fuzz_soak` collects the PID of every background fuzzer (`pids="$$pids $$!"`) and checks every child process exit status (`for p in $$pids; do wait $$p || fail=1; done; exit $$fail`) instead of calling a bare `wait` (which returns `0` in POSIX `sh` even when a background job fails).

---

## 2. Defect Found & Fixed During the 1-Hour Soak

During the initial 1-hour parallel soak across all 11 targets, 10 targets completed `3,601s` (`DONE`) with zero crashes, OOMs, or timeouts (`229,033,300` total executions), while `fuzz_zip_offset_load` discovered a real arithmetic overflow after `2,484` executions (`crash-2950c6e3b040e890a337af10eedb41f09adc3ffd`):

- **Root Cause (`src/blob_store/sorted_uint_vec.rs:365-370`, `src/blob_store/zip_offset.rs:608-615`)**:
  1. In `SortedUintVec::read_from`, `min_index_len = (num_blocks * config.sample_width as usize).div_ceil(8)` and `min_data_len = (size * config.offset_width as usize).div_ceil(8)` multiplied an untrusted `size` (up to `usize::MAX`) by `offset_width` (`8..=32`) or `num_blocks` by `sample_width` (`16..=64`) without `checked_mul`, panicking with `attempt to multiply with overflow` when a valid `ZipOffsetBlobStore` seed had its element count mutated to a large 64-bit integer.
  2. In `SortedUintVec::num_blocks`, `(self.size + self.config.block_mask()) >> self.config.log2_block_units` could overflow when `self.size` was near `usize::MAX`.
  3. In `SortedUintVec::get2` and `SortedUintVec::get_unchecked`, `index + 1` and `block_min + delta as u64` used unchecked addition.
  4. In `ZipOffsetBlobStore::get_record_impl`, `let mut record_len = (end_offset - start_offset) as usize;` was evaluated before checking `end_offset < start_offset`, which would panic on non-monotonic corrupted offset tables.
- **Fix**:
  - Replaced the unchecked multiplications and additions in `SortedUintVec::read_from`, `SortedUintVec::num_blocks`, `SortedUintVec::get2`, and `SortedUintVec::get_unchecked` with `checked_mul`, `div_ceil`, `saturating_add`, and `checked_add`, returning `Err(ZiporaError::invalid_data(...))` on overflow.
  - Checked `end_offset < start_offset || start_offset > self.content.len() as u64 || end_offset > self.content.len() as u64` in `ZipOffsetBlobStore::get_record_impl` before computing `end_offset - start_offset`.
  - Added regression test `test_s52_r2_fuzz_soak_checks_per_job_exit_codes_and_records_1h_summary` in `tests/format_spec_golden_tests.rs` with the exact 316-byte crash input (`crash-2950c6e3b040e890a337af10eedb41f09adc3ffd`), and re-ran `fuzz_zip_offset_load` for a full 1-hour (`3,600s`) soak.

---

## 3. 1-Hour Soak Summary Across All 11 Targets (`360,359,996` Total Executions, `oom=0, timeout=0, crash=0`)

| Fuzz Target | Status | Duration (s) | Total Executions (`stat::number_of_executed_units`) | Avg `exec/s` | Edge Coverage (`cov`) | Features (`ft`) | Final Corpus (`corp`) | New Units Added | Slowest Unit (s) | Peak RSS (MiB) | Crashes / OOMs / Timeouts |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| `fuzz_zip_offset_load` | `DONE` | `3,601` | `131,326,696` | `36,469` | `681` | `1,359` | `186 / 50 Kb` | `678` | `0` | `163` | `oom=0, timeout=0, crash=0` |
| `fuzz_simple_zip_build` | `DONE` | `3,601` | `10,094,872` | `2,803` | `315` | `1,362` | `218 / 65 Kb` | `1,483` | `0` | `530` | `oom=0, timeout=0, crash=0` |
| `fuzz_mixed_len_build` | `DONE` | `3,601` | `41,949,841` | `11,649` | `434` | `1,152` | `145 / 35 Kb` | `1,077` | `0` | `529` | `oom=0, timeout=0, crash=0` |
| `fuzz_huffman_decode` | `DONE` | `3,601` | `53,052,250` | `14,732` | `253` | `577` | `71 / 1,781 b` | `163` | `0` | `428` | `oom=0, timeout=0, crash=0` |
| `fuzz_rans_decode` | `DONE` | `3,601` | `17,400,424` | `4,832` | `93` | `198` | `44 / 966 b` | `132` | `0` | `390` | `oom=0, timeout=0, crash=0` |
| `fuzz_fse_decompress` | `DONE` | `3,601` | `4,391,452` | `1,219` | `231` | `530` | `158 / 15,797 b` | `664` | `0` | `407` | `oom=0, timeout=0, crash=0` |
| `fuzz_double_array_trie` | `DONE` | `3,601` | `11,897,954` | `3,304` | `516` | `2,634` | `645 / 149 Kb` | `5,148` | `0` | `496` | `oom=0, timeout=0, crash=0` |
| `fuzz_uint_vec` | `DONE` | `3,601` | `68,085,357` | `18,907` | `61` | `147` | `33 / 4,848 b` | `94` | `0` | `431` | `oom=0, timeout=0, crash=0` |
| `fuzz_hybrid_posting_list` | `DONE` | `3,601` | `12,936,192` | `3,592` | `1,177` | `5,944` | `858 / 552 Kb` | `5,377` | `0` | `513` | `oom=0, timeout=0, crash=0` |
| `fuzz_double_array_trie_map` | `DONE` | `3,601` | `7,400,606` | `2,055` | `819` | `3,691` | `454 / 160 Kb` | `3,957` | `0` | `517` | `oom=0, timeout=0, crash=0` |
| `fuzz_blob_store_roundtrip` | `DONE` | `3,601` | `1,824,350` | `506` | `1,247` | `4,320` | `402 / 193 Kb` | `2,293` | `0` | `166` | `oom=0, timeout=0, crash=0` |
