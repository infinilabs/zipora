# Stage 12 Review Ledger — `S30` Carry-Overs (`S30-R1..R3`) + `C11` (`src/ffi/`) + `F3` + `S31` (`S31-R1..R6`)

## 1. Scope & Commit Table (`ea05138..`)

| Commit | Subject | Diffstat (`--shortstat`) |
|---|---|---|
| `23fd5e2` | `test(thread): add Miri-verified unit test and cfg(tokio_unstable) path for tokio_runtime_handle_id [S30-R1]` | `3 files changed, 82 insertions(+), 4 deletions(-)` |
| `52b3d4d` | `fix(concurrency): drop bimodal single-thread metric from bench_s29_r1_executor_timings [S30-R3]` | `1 file changed, 1 insertion(+), 27 deletions(-)` |
| `db159a3` | `fix(ffi): guard all FFI pointers and slices against misalignment and overflow [C11.1]` | `3 files changed, 285 insertions(+), 73 deletions(-)` |
| `3031eac` | `fix(ffi): preserve error details, escape embedded NULs, and avoid callback deadlock [C11.2]` | `5 files changed, 161 insertions(+), 60 deletions(-)` |
| `b231dc8` | `fix(ffi): make C handles opaque, scope cbindgen exports, and split FFI modules [C11.3]` | `8 files changed, 1166 insertions(+), 571 deletions(-)` |
| `d22194a` | `test(ffi,ci): add cbindgen header parity check, C/C++ smoke test, and miri_ffi target [C11.4, F3]` | `5 files changed, 310 insertions(+), 3 deletions(-)` |
| `e91ef17` | `docs(review): fix S30-R2/R3 ledger claims and add Stage 12 FFI review ledger [S30-R2]` | `3 files changed, 114 insertions(+), 15 deletions(-)` |
| `6b05f1a` | `fix(ffi,test): build fresh staticlib per run in CARGO_TARGET_TMPDIR [S31-R1]` | `2 files changed, 41 insertions(+), 22 deletions(-)` |
| `4111877` | `fix(build): revert default crate-type to rlib only [S31-R2]` | `3 files changed, 22 insertions(+), 2 deletions(-)` |
| `af7babf` | `fix(ffi): drop 11 unused types from cbindgen export list and include/zipora.h [S31-R3]` | `5 files changed, 27 insertions(+), 234 deletions(-)` |
| `b057974` | `fix(ffi): guard error callback against recursive error re-entrancy [S31-R4]` | `2 files changed, 52 insertions(+), 7 deletions(-)` |
| `4d2855f` | `fix(ffi,memory): validate chunk layout in memory_pool_new and document max_chunks [S31-R5]` | `5 files changed, 54 insertions(+), 3 deletions(-)` |
| `1cd1bb0` | `test(scripts): add reproducible C11 and S31 RED verification script [S31-R6]` | `1 file changed, 95 insertions(+)` |

---

## 2. `S30` Carry-Over Items (`S30-R1..R3`)

### 2.1 `23fd5e2` — `S30-R1`: `tokio_runtime_handle_id` `cfg(tokio_unstable)` Path & Miri Unit Test
- Added `println!("cargo:rustc-check-cfg=cfg(tokio_unstable)");` in `main` (`1cd1bb0:build.rs:18-34`) so `#[cfg(tokio_unstable)]` compiles without `unexpected_cfgs` warnings.
- Added `#[cfg(all(feature = "async", tokio_unstable))]` implementation of `tokio_runtime_handle_id` (`1cd1bb0:src/thread/mod.rs:71-76`) hashing `handle.id()` via `std::collections::hash_map::DefaultHasher` (`std::hash::{Hash, Hasher}`), while preserving the `#[cfg(all(feature = "async", not(tokio_unstable)))]` word-1 layout read (`1cd1bb0:src/thread/mod.rs:87-100`).
- Added unit test `test_s30_r1_tokio_runtime_handle_id_clone_identity_and_distinctness` (`1cd1bb0:src/thread/mod.rs:107-154`), verifying under `make miri_thread` (`31 passed; 0 failed; 1 ignored` in both SIMD and `MIRI_SCALAR=1` builds) that cloned handles yield identical IDs and distinct `current_thread` and `multi_thread` Tokio runtimes yield distinct IDs.

### 2.2 `S30-R2` — Stage 11 Ledger Fixes (`docs/review/thread_and_concurrency.md`, `docs/review/io.md`)
- Removed unbacked `(323 ns/item 1-stage vs 396 ns/item at 00652d2)` in `docs/review/thread_and_concurrency.md` §7.4.
- Replaced repeated `f7e39ad` numbers under column `1141a79` (`vec_vec_u8_100kx3_de`, `vec_box_u64_100k_de`, `vec_rc_u64_100k_de`, `hashmap_u32_u64_100k_de`) with `—` in `docs/review/io.md` §12.4.
- Credited the `313.8 ns/task` measurement from `bench_s29_r1_executor_timings` explicitly to `f37cf61` in `docs/review/thread_and_concurrency.md` (§1, §7.5, §9.1).

### 2.3 `52b3d4d` — `S30-R3`: Drop Bimodal Single-Thread Metric from `bench_s29_r1_executor_timings`
- Removed the single-thread `1t_100k` section from `bench_s29_r1_executor_timings` (`1cd1bb0:src/concurrency/work_stealing.rs:1803-1858`), leaving `mp8_400k` (`316.6 ns/task` 5-round min at `e91ef17`), and removed the `1t_100k` figures from `docs/review/thread_and_concurrency.md` §11.2.

---

## 3. `C11` (`src/ffi/`) + `F3` (`db159a3..d22194a`)

### 3.1 `db159a3` — `C11.1`: Pointer Alignment, `(NULL, 0)` Slice, and Length Overflow Hardening
- Added `.is_aligned()` checks across all typed FFI pointers in `1cd1bb0:src/ffi/containers.rs:26-211`, `1cd1bb0:src/ffi/blob_store.rs:24-160`, and `1cd1bb0:src/ffi/algorithms.rs:40-156`.
- Enforced `size <= isize::MAX as usize` on byte slices in `blob_store_put` (`1cd1bb0:src/ffi/blob_store.rs:43-86`), `suffix_array_new` (`1cd1bb0:src/ffi/algorithms.rs:15-32`), and `suffix_array_search` (`1cd1bb0:src/ffi/algorithms.rs:77-126`), plus `MAX_U32_LEN = (isize::MAX as usize) / std::mem::size_of::<u32>()` in `radix_sort_u32` (`1cd1bb0:src/ffi/algorithms.rs:134-156`).
- Hardened `CBuffer` (`1cd1bb0:src/ffi/types.rs:340-431`) and `CString` (`1cd1bb0:src/ffi/types.rs:437-509`), and added `test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub` (`1cd1bb0:src/ffi/c_api.rs:513-650`).

### 3.2 `3031eac` — `C11.2` + `D10`: `CResult` Error Fidelity, Embedded NUL Escaping, Re-Entrant Callback Unregister, and `.expect()` Removal
- Added `CResult::from_zipora_error` (`1cd1bb0:src/ffi/mod.rs:49-67`) mapping `ZiporaError::Io(err)` with `ErrorKind::NotFound` to `CResult::NotFound`, other `Io` errors to `CResult::IoError`, `InvalidData`/`InvalidParameter`/`OutOfBounds`/`Misaligned`/`Configuration` to `CResult::InvalidInput`, `OutOfMemory`/`ResourceExhausted` to `CResult::MemoryError`, `NotSupported` to `CResult::UnsupportedOperation`, and remaining variants to `CResult::InternalError`.
- Updated `set_last_error` (`1cd1bb0:src/ffi/c_api.rs:37-73`) to escape interior NUL bytes (`msg.replace('\0', "\\0")`), access `LAST_ERROR` via `.try_with(...)`, and release `ERROR_CALLBACK.lock()` before invoking registered callbacks inside `std::panic::catch_unwind`.
- Eliminated `.expect(...)` in `zipora_version` (`1cd1bb0:src/ffi/c_api.rs:129-133`) by returning `concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char`, lowering `max_expects = 81` (`82 -> 81`, `[D10]`) in `scripts/audit_baseline.toml`.
- Added unit tests `test_c11_2a_cresult_not_found_mapping_and_blob_store_get` (`1cd1bb0:src/ffi/c_api.rs:653-676`), `test_c11_2b_set_last_error_preserves_embedded_nul_bytes` (`1cd1bb0:src/ffi/c_api.rs:679-685`), and `test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock` (`1cd1bb0:src/ffi/c_api.rs:688-711`).

### 3.3 `b231dc8` — `C11.3` + `D6` + `D9`: Opaque C Handles, Scoped `cbindgen` Header, and Module Split
- Removed `#[repr(C)]` from handle wrapper types (`CFastVec`, `CMemoryPool`, `CBlobStore`, `CSuffixArray` in `1cd1bb0:src/ffi/types.rs:11-117`) so `cbindgen` emits strict opaque forward declarations (`typedef struct CFastVec CFastVec;`).
- Removed the 2 `#[allow(dead_code)]` attributes in `build.rs` (`0` remaining in `build.rs` and `0` in `src/ffi/`, `50` remaining outside `src/ffi/` across the repository under `D6`).
- Split `src/ffi/c_api.rs` across submodules (`[D9]`, every FFI source file `< 750` lines):
  - `src/ffi/mod.rs` (`78` lines)
  - `src/ffi/c_api.rs` (`739` lines)
  - `src/ffi/containers.rs` (`309` lines)
  - `src/ffi/blob_store.rs` (`191` lines)
  - `src/ffi/algorithms.rs` (`257` lines)
  - `src/ffi/types.rs` (`600` lines)

---

## 4. Stage 12 Re-Review Remediation (`S31-R1..R6`: `6b05f1a..1cd1bb0`)

### 4.1 `6b05f1a` — `S31-R1`: Build Fresh Static Archive per Run in `CARGO_TARGET_TMPDIR`
- **Root cause**: `d22194a:tests/ffi_c_smoke.rs:20-43` (`find_libzipora_a`) returned `target/debug/libzipora.a` whenever present (allowing broken `radix_sort_u32` code to pass against a stale `.a` archive) and checked `root.join("target/debug/libzipora.a")` unconditionally after `cargo build`, panicking whenever `CARGO_TARGET_DIR` pointed elsewhere.
- **Fix**: Replaced `find_libzipora_a` with `CleanupDir` (`1cd1bb0:tests/ffi_c_smoke.rs:20-26`) and `build_fresh_staticlib` (`1cd1bb0:tests/ffi_c_smoke.rs:28-63`), invoking `cargo rustc --locked --lib --features ffi --crate-type staticlib --target-dir <CARGO_TARGET_TMPDIR>/ffi_staticlib_target` on every run of `test_f3_c11_and_cpp17_header_compile_and_linked_c_smoke_binary_runs` (`1cd1bb0:tests/ffi_c_smoke.rs:153-316`), copying `<build_dir>/debug/libzipora.a` into the temporary C test directory, and deleting `<build_dir>` immediately on drop.
- Verified with `CARGO_TARGET_DIR` pointed at an external directory (`PASS`) and with `radix_sort_u32` broken to a no-op (`FAIL`: `Assertion 'nums[0] == 1 && nums[1] == 7 && nums[2] == 19 && nums[3] == 42 && nums[4] == 99' failed`).

### 4.2 `4111877` — `S31-R2`: Revert Default `[lib] crate-type` to `["rlib"]`
- Reverted `crate-type = ["rlib", "staticlib", "cdylib"]` in `Cargo.toml` to `crate-type = ["rlib"]` so non-FFI and test builds do not emit `libzipora.a` or `libzipora.so` into `target/`, and added `test_s31_r2_cargo_toml_lib_crate_type_is_rlib_only` (`1cd1bb0:tests/ffi_c_smoke.rs:319-336`).

### 4.3 `af7babf` — `S31-R3`: Drop 11 Unused Types from `cbindgen` Export List & `include/zipora.h`
- Removed all 11 unused C FFI struct names (`CRadixSort`, `CMultiWayMerge`, `CMemoryPoolConfig`, `CMemoryPoolStats`, `CRadixSortConfig`, `CAlgorithmStats`, `CBlobStoreStats`, `CMemoryConfig`, `CMemoryStats`, `CBuffer`, `CString`) from `ExportConfig.include` in `generate_ffi_bindings` (`1cd1bb0:build.rs:332-407`), deleted empty placeholder structs `CRadixSort` and `CMultiWayMerge` from `src/ffi/types.rs`, regenerated `include/zipora.h` (`331` lines, exporting only `CResult`, `CFastVec`, `CMemoryPool`, `CBlobStore`, `CSuffixArray`, and the 24 functions), and added `test_s31_r3_header_exports_only_types_used_by_c_functions` (`1cd1bb0:tests/ffi_c_smoke.rs:339-362`).

### 4.4 `b057974` — `S31-R4`: Thread-Local Re-Entrancy Flag on Error Callback Invocation
- Added thread-local `IN_ERROR_CALLBACK: Cell<bool>` and RAII `CallbackReentrancyGuard` (`1cd1bb0:src/ffi/c_api.rs:21-31`) inside `set_last_error` (`1cd1bb0:src/ffi/c_api.rs:37-73`) so any error triggered while a callback is already executing on the current thread updates `LAST_ERROR` and returns immediately without recursing into the callback.
- Added `test_s31_r4_error_callback_triggering_error_does_not_recurse` (`1cd1bb0:src/ffi/c_api.rs:714-738`), verifying both recursion prevention and subsequent callback firing after return.

### 4.5 `4d2855f` — `S31-R5`: Validate `Layout` Upfront in `MemoryPool::new` & Document `max_chunks`
- Added `std::alloc::Layout::from_size_align(config.chunk_size, config.alignment)` check to `MemoryPool::new` (`1cd1bb0:src/memory/pool.rs:133-159`), documented both layout requirements and `max_chunks` free-cache semantics on `memory_pool_new` (`1cd1bb0:src/ffi/containers.rs:101-126`) and in regenerated `include/zipora.h`, and added `test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks` (`1cd1bb0:src/ffi/containers.rs:283-308`).

### 4.6 `1cd1bb0` — `S31-R6`: Reproducible `C11` & `S31` RED Verification Script (`scripts/verify_c11_red.sh`)
- Committed executable splice-and-run script `1cd1bb0:scripts/verify_c11_red.sh:1-95` verifying `C11.1`, `C11.2a..c` on `ea05138` and `S31-R4`, `S31-R5` on `e91ef17` using temporary worktrees under `target/`:

```text
=== Part 1: C11.1 / C11.2a..c RED checks on ea05138 ===
thread 'ffi::c_api::tests::test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub' (4009622) panicked at src/ffi/c_api.rs:552:35:
unsafe precondition(s) violated: slice::from_raw_parts_mut requires the pointer to be aligned and non-null, and the total size of the slice not to exceed `isize::MAX`
RED check test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub on ea05138: exit_code=101
RED check test_c11_2a_cresult_not_found_mapping_and_blob_store_get on ea05138: exit_code=101
RED check test_c11_2b_set_last_error_preserves_embedded_nul_bytes on ea05138: exit_code=101
RED check test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock on ea05138: exit_code=101
=== Part 2: S31-R4 / S31-R5 RED checks on e91ef17 ===
thread 'ffi::c_api::tests::test_s31_r4_error_callback_triggering_error_does_not_recurse' (4012577) has overflowed its stack
fatal runtime error: stack overflow, aborting
RED check test_s31_r4_error_callback_triggering_error_does_not_recurse on e91ef17: exit_code=101
thread 'ffi::containers::tests::test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks' (4012604) panicked at src/ffi/containers.rs:278:13:
memory_pool_new(usize::MAX, 1) must fail upfront on invalid Layout
RED check test_s31_r5_memory_pool_new_rejects_invalid_layout_and_documents_max_chunks on e91ef17: exit_code=101
All 6 RED checks verified cleanly!
```

---

## 5. Verification Summary (`1cd1bb0`)

- **`touch src/lib.rs && make sanity` (`--locked`)**:
  - `lib_debug = 3194`, `lib_release = 3211`, `doctests = 229`, `compile_fail = 8`, `all_targets = 3861`
  - `unsafe`: `4` undocumented (`src/thread/` `69/69 = 100.0%`, `src/concurrency/` `0/0 = 100.0%`, `src/ffi/` `28/28 = 100.0%`)
  - `honesty markers`: `27` (`0` in `src/ffi/` and `src/concurrency/`)
  - `expects`: `81` (`0` in `src/ffi/`, `src/thread/`, and `src/concurrency/`)
- **Sanitizers & Compilers**:
  - `make miri_ffi`: `32 passed; 0 failed; 0 ignored` (SIMD + `MIRI_SCALAR=1`)
  - `make miri_thread`: `31 passed; 0 failed; 1 ignored` (SIMD + `MIRI_SCALAR=1`)
  - `gcc -x c -std=c11 -Wall -Wextra -Wpedantic -Werror -fsyntax-only include/zipora.h`: exit `0`
  - `g++ -x c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -fsyntax-only include/zipora.h`: exit `0`
