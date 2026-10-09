# Stage 12 Review Ledger — `S30` Carry-Overs (`S30-R1..R3`) + `C11` (`src/ffi/`) + `F3` + `Phase D`

## 1. Scope & Commit Table (`ea05138..`)

| Commit | Subject | Diffstat (`--shortstat`) |
|---|---|---|
| `23fd5e2` | `test(thread): add Miri-verified unit test and cfg(tokio_unstable) path for tokio_runtime_handle_id [S30-R1]` | `3 files changed, 82 insertions(+), 4 deletions(-)` |
| `52b3d4d` | `fix(concurrency): drop bimodal single-thread metric from bench_s29_r1_executor_timings [S30-R3]` | `1 file changed, 1 insertion(+), 27 deletions(-)` |
| `db159a3` | `fix(ffi): guard all FFI pointers and slices against misalignment and overflow [C11.1]` | `3 files changed, 285 insertions(+), 73 deletions(-)` |
| `3031eac` | `fix(ffi): preserve error details, escape embedded NULs, and avoid callback deadlock [C11.2]` | `5 files changed, 161 insertions(+), 60 deletions(-)` |
| `b231dc8` | `fix(ffi): make C handles opaque, scope cbindgen exports, and split FFI modules [C11.3]` | `8 files changed, 1166 insertions(+), 571 deletions(-)` |
| `d22194a` | `test(ffi,ci): add cbindgen header parity check, C/C++ smoke test, and miri_ffi target [C11.4, F3]` | `5 files changed, 310 insertions(+), 3 deletions(-)` |

---

## 2. `S30` Carry-Over Items (`S30-R1..R3`)

### 2.1 `23fd5e2` — `S30-R1`: `tokio_runtime_handle_id` `cfg(tokio_unstable)` Path & Miri Unit Test
- Added `println!("cargo:rustc-check-cfg=cfg(tokio_unstable)");` in `main` (`d22194a:build.rs:18-34`) so `#[cfg(tokio_unstable)]` compiles without `unexpected_cfgs` warnings.
- Added `#[cfg(tokio_unstable)]` implementation of `tokio_runtime_handle_id` (`d22194a:src/thread/mod.rs:71-76`) calling `handle.id().to_string()` hashed into a deterministic `usize`, while preserving the `#[cfg(not(tokio_unstable))]` word-1 layout read (`d22194a:src/thread/mod.rs:87-100`).
- Added unit test `test_s30_r1_tokio_runtime_handle_id_clone_identity_and_distinctness` (`d22194a:src/thread/mod.rs:107-154`), verifying under `make miri_thread` (`31 passed; 0 failed; 1 ignored` in both SIMD and `MIRI_SCALAR=1` builds) that cloned handles yield identical IDs and distinct `current_thread` and `multi_thread` Tokio runtimes yield distinct IDs.

### 2.2 `S30-R2` — Stage 11 Ledger Fixes (`docs/review/thread_and_concurrency.md`, `docs/review/io.md`)
- Removed unbacked `(323 ns/item 1-stage vs 396 ns/item at 00652d2)` in `docs/review/thread_and_concurrency.md` §7.4.
- Replaced repeated `f7e39ad` numbers under column `1141a79` (`vec_vec_u8_100kx3_de`, `vec_box_u64_100k_de`, `vec_rc_u64_100k_de`, `hashmap_u32_u64_100k_de`) with `—` in `docs/review/io.md` §12.4.
- Credited the `313.8 ns/task` measurement from `bench_s29_r1_executor_timings` explicitly to `f37cf61` (separated `4929ccc` and `f37cf61` columns in §7.5 and updated §1 `S27-R3` and §9.1 `S27-R3` in `docs/review/thread_and_concurrency.md`).

### 2.3 `52b3d4d` — `S30-R3`: Drop Bimodal Single-Thread Metric from `bench_s29_r1_executor_timings`
- Removed the single-thread `1t_100k` section from `bench_s29_r1_executor_timings` (`d22194a:src/concurrency/work_stealing.rs:1803-1858`), which swung across `456–1,266 ns/task` across runs of the same binary depending on `ThreadId % 8` queue hashing, retaining `mp8_400k` (`329.9 ns/task` 5-round min at `52b3d4d`), and removed the `1t_100k` figures from `docs/review/thread_and_concurrency.md` §11.2.

---

## 3. `C11` (`src/ffi/`) + `F3` + `Phase D` (`B7`, `B8`, `D6`, `D9`, `D10`)

### 3.1 `db159a3` — `C11.1`: Pointer Alignment, `(NULL, 0)` Slice, and Length Overflow Hardening
- **Root cause**:
  1. FFI handle and out-parameter pointers across `fast_vec_*`, `memory_pool_*`, `blob_store_*`, `suffix_array_*`, and `radix_sort_u32` checked only `.is_null()` without `.is_aligned()`, invoking immediate Undefined Behavior under misaligned pointers from C callers.
  2. `radix_sort_u32` (`d22194a:src/ffi/algorithms.rs:134-156`) called `std::slice::from_raw_parts_mut(data, size)` without checking `size <= (isize::MAX as usize) / std::mem::size_of::<u32>()` (`MAX_U32_LEN`), allowing `size * 4 > isize::MAX` UB.
  3. `CBuffer::into_vec`, `CBuffer::as_slice`, `CBuffer::as_mut_slice` (`d22194a:src/ffi/types.rs:389-436`) and `CString::into_string`, `CString::as_str`, `CString::drop` (`d22194a:src/ffi/types.rs:477-520`) lacked `size <= capacity <= isize::MAX as usize` bounds checks and double-free prevention (`std::mem::replace(&mut self.data, std::ptr::null_mut())`).
- **Fix**:
  - Added `.is_aligned()` checks to all typed FFI pointers across `d22194a:src/ffi/containers.rs:26-203`, `d22194a:src/ffi/blob_store.rs:24-160`, and `d22194a:src/ffi/algorithms.rs:40-156`.
  - Enforced `size <= isize::MAX as usize` on byte slices in `blob_store_put` (`d22194a:src/ffi/blob_store.rs:43-86`), `suffix_array_new` (`d22194a:src/ffi/algorithms.rs:15-32`), and `suffix_array_search` (`d22194a:src/ffi/algorithms.rs:77-126`), plus `MAX_U32_LEN = (isize::MAX as usize) / std::mem::size_of::<u32>()` in `radix_sort_u32` (`d22194a:src/ffi/algorithms.rs:134-156`).
  - Hardened `CBuffer` (`d22194a:src/ffi/types.rs:389-436`) and `CString` (`d22194a:src/ffi/types.rs:477-520`), and added `test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub` (`d22194a:src/ffi/c_api.rs:495-632`).

Verbatim Miri RED output of `test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub` spliced onto `52b3d4d`:
```text
running 1 test
test ffi::c_api::tests::test_c11_1_misaligned_and_overflow_pointers_rejected_without_ub ... error: Undefined Behavior: constructing invalid value of type &mut [u32]: encountered an unaligned reference (required 4 byte alignment, found 1)
   --> src/ffi/c_api.rs:552:35
    |
552 |         let data_slice = unsafe { std::slice::from_raw_parts_mut(data, size) };
    |                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Undefined Behavior occurred here
    |
    = help: this indicates a bug in the program: it performed an invalid operation, and caused Undefined Behavior
    = help: see https://doc.rust-lang.org/nightly/reference/behavior-considered-undefined.html for further information
```

### 3.2 `3031eac` — `C11.2` + `D10`: `CResult` Error Fidelity, Embedded NUL Escaping, Re-Entrant Callback Safety, and `.expect()` Removal
- Added `CResult::from_zipora_error` (`d22194a:src/ffi/mod.rs:49-67`) mapping `ZiporaError::Io(err)` with `ErrorKind::NotFound` to `CResult::NotFound`, other `Io` errors to `CResult::IoError`, `InvalidData`/`InvalidParameter`/`OutOfBounds`/`Misaligned`/`Configuration` to `CResult::InvalidInput`, `OutOfMemory`/`ResourceExhausted` to `CResult::MemoryError`, `NotSupported` to `CResult::UnsupportedOperation`, and remaining variants to `CResult::InternalError`.
- Replaced hardcoded `CResult::InvalidInput` error returns in `fast_vec_push` (`d22194a:src/ffi/containers.rs:45-62`), `memory_pool_deallocate` (`d22194a:src/ffi/containers.rs:172-203`), `blob_store_put` (`d22194a:src/ffi/blob_store.rs:43-86`), `blob_store_get` (`d22194a:src/ffi/blob_store.rs:95-160`), and `radix_sort_u32` (`d22194a:src/ffi/algorithms.rs:134-156`) with `CResult::from_zipora_error(&e)`.
- Updated `set_last_error` (`d22194a:src/ffi/c_api.rs:28-55`) to escape interior NUL bytes (`msg.replace('\0', "\\0")`) instead of discarding the message for `"Error message contained null byte"`, access `LAST_ERROR` via `.try_with(...)`, and copy the callback pointer out of `ERROR_CALLBACK.lock()` before invoking it inside `std::panic::catch_unwind` so a callback calling `zipora_set_error_callback(None)` does not self-deadlock.
- Eliminated `.expect(...)` in `zipora_version` (`d22194a:src/ffi/c_api.rs:111-115`) via `const VERSION_CSTR: &CStr` and lowered `max_expects = 81` (`82 -> 81`, `[D10]`) in `scripts/audit_baseline.toml`.
- Added unit tests `test_c11_2a_cresult_not_found_mapping_and_blob_store_get` (`d22194a:src/ffi/c_api.rs:635-658`), `test_c11_2b_set_last_error_preserves_embedded_nul_bytes` (`d22194a:src/ffi/c_api.rs:661-667`), and `test_c11_2c_error_callback_reentrant_unregister_does_not_deadlock` (`d22194a:src/ffi/c_api.rs:670-693`), and updated `test_ffi_blob_store_get_missing` (`d22194a:tests/ffi_comprehensive_tests.rs:240-254`) to assert `CResult::NotFound`.

### 3.3 `b231dc8` — `C11.3` + `D6` + `D9`: Opaque C Handles, Scoped `cbindgen` Header, and Module Split
- Removed `#[repr(C)]` from handle wrapper types (`CFastVec`, `CMemoryPool`, `CBlobStore`, `CSuffixArray`, `CRadixSort`, `CMultiWayMerge` in `d22194a:src/ffi/types.rs:11-127`) so `cbindgen` emits strict opaque forward declarations (`typedef struct CFastVec CFastVec;`), eliminating all incomplete-field errors in ISO C11 / ISO C++17 and all cross-struct pointer casts between handle wrappers and underlying Rust types.
- Removed `#[allow(dead_code)]` attributes (`[D6]`) and scoped `cbindgen::ExportConfig` in `generate_ffi_bindings` (`d22194a:build.rs:332-417`), updated `.gitignore` (`include/*` + `!include/zipora.h`), and committed canonical `include/zipora.h` (`543` lines).
- Split `src/ffi/c_api.rs` (`1,239` lines -> `694` lines) across submodules (`[D9]`, every file `< 700` lines):
  - `src/ffi/mod.rs` (`78` lines)
  - `src/ffi/c_api.rs` (`694` lines)
  - `src/ffi/containers.rs` (`273` lines)
  - `src/ffi/blob_store.rs` (`191` lines)
  - `src/ffi/algorithms.rs` (`257` lines)
  - `src/ffi/types.rs` (`610` lines)

### 3.4 `d22194a` — `C11.4` + `F3`: Crate Artifact Targets, `ffi_c_smoke` Test Suite, `miri_ffi`, and CI Gate
- Added `[lib] crate-type = ["rlib", "staticlib", "cdylib"]` in `Cargo.toml`.
- Added integration test suite `tests/ffi_c_smoke.rs` (`295` lines):
  1. `test_c11_export_and_ffi_guard_parity_and_header_coverage` (`d22194a:tests/ffi_c_smoke.rs:46-114`): verifies all `24` `#[unsafe(no_mangle)] pub unsafe extern "C" fn` exports call `ffi_guard(` (`export_count == ffi_guard_count == 24`) and appear in `include/zipora.h`.
  2. `test_c11_cbindgen_header_matches_committed_include_header` (`d22194a:tests/ffi_c_smoke.rs:117-130`): verifies byte-for-byte parity between `OUT_DIR/zipora.h` and `include/zipora.h`.
  3. `test_f3_c11_and_cpp17_header_compile_and_linked_c_smoke_binary_runs` (`d22194a:tests/ffi_c_smoke.rs:133-195`): compiles `include/zipora.h` under `gcc -x c -std=c11 -Wall -Wextra -Wpedantic -Werror -fsyntax-only` and `g++ -x c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -fsyntax-only`, builds and links a C11 smoke program against `libzipora.a` (`-lpthread -ldl -lm`), and runs the binary end-to-end.
- Added `miri_ffi` target to `Makefile` (`30 passed; 0 failed; 0 ignored` in both SIMD and `MIRI_SCALAR=1` builds) and C/C++ header + C11 smoke gate steps to `.github/workflows/ci.yml`.

---

## 4. Verification Summary

- **`touch src/lib.rs && make sanity` (`--locked`)**:
  - `lib_debug = 3192`, `lib_release = 3209`, `doctests = 229`, `compile_fail = 8`, `all_targets = 3857`
  - `unsafe`: `4` undocumented (`src/thread/` `69/69 = 100.0%`, `src/concurrency/` `0/0 = 100.0%`, `src/ffi/` `28/28 = 100.0%`)
  - `honesty markers`: `27` (`0` in `src/ffi/` and `src/concurrency/`)
  - `expects`: `81` (`0` in `src/ffi/`, `src/thread/`, and `src/concurrency/`)
  - `allow(dead_code)` / `allow(unused)`: `0` across crate and `build.rs`
  - `line_count`: `0` FFI files `>= 700` lines
- **Sanitizers & Header Checks**:
  - `make miri_thread`: `31 passed; 0 failed; 1 ignored` (SIMD + `MIRI_SCALAR=1`)
  - `make miri_ffi`: `30 passed; 0 failed; 0 ignored` (SIMD + `MIRI_SCALAR=1`)
  - `gcc -x c -std=c11 -Wall -Wextra -Wpedantic -Werror -fsyntax-only include/zipora.h`: exit `0` (`0` warnings)
  - `g++ -x c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -fsyntax-only include/zipora.h`: exit `0` (`0` warnings)
