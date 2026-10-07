# Stage 11 Review Ledger: `src/thread/` + `src/concurrency/` + `C10.1` Weak-Memory Ordering Audit + `S25` Remediation

## 1. Scope & Summary

Stage 11 covers:
1. **`S24-R1..R4` + `S25-R8`** carry-over fixes from the Stage 10 re-review (`b8e7f8b..39814be` and `f7e39ad`, documented in `docs/review/io.md` §12).
2. **`C10` (`src/thread/` + `src/concurrency/`)**: deep correctness, safety, panic-resilience, and resource-lifecycle audit across `src/thread/atomic_ext.rs`, `src/thread/linux_futex.rs`, `src/thread/instance_tls.rs`, `src/concurrency/mod.rs`, `src/concurrency/work_stealing.rs`, `src/concurrency/fiber_pool.rs`, `src/concurrency/parallel_trie.rs`, and `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`).
3. **`C10.1` (Crate-Wide Weak-Memory Ordering Audit)**: audit of all `Ordering::Relaxed` / `Release` / `Acquire` / `AcqRel` / `SeqCst` atomic operations across `src/thread/`, `src/concurrency/`, `src/fsa/cspp_trie_concurrent.rs`, `src/memory/lockfree_pool.rs`, `src/containers/specialized/lru_map.rs`, and `src/containers/specialized/concurrent_lru_map.rs`, adding `// ORDERING:` justifications to every `Ordering::Relaxed` site in the audited modules, upgrading weak orderings on shared-data publication/reclamation paths, and replacing all non-atomic byte reads/writes on `SharedPool` in `ConcurrentCsppTrie` with word-atomic helpers.
4. **`S25-R1..R9` Stage 11 Re-Review Remediation (`f7e39ad..48438fe`, §7)**:
   - `S25-R1` (`0932ce7`): Eliminated the 256-byte `read_zpath` over-read in `ConcurrentCsppTrie::lookup`/`contains`/`lookup_with_guard` via zero-copy word-wise `SharedPool::bytes_eq_relaxed(zpath_slot, &key[pos..end])`, cached `raw: u32` header word in `ConcurrentNodeView`, peeled root fast-node step, and inlined leaf `valpos`.
   - `S25-R2` (`48438fe`): Fixed `WorkStealingExecutor::spawn(fiber)` to spawn directly onto the current Tokio runtime (`tokio::spawn`) so it never hangs when a global executor's creation runtime has been dropped or when a fiber awaits a child fiber on a 1-worker executor.
   - `S25-R3`: Corrected all ledger citations, identifier names, and before/after descriptions in `docs/review/thread_and_concurrency.md`, `docs/review/io.md`, and `GEMINI.md`, and upgraded `scripts/audit_ledger.py` with commit-numstat, unbackticked-citation, item-span, full-token, and AST type-member checks.
   - `S25-R4` (`0932ce7`, `b74f6fe`, `48438fe`): Restored `ConcurrentCsppTrie` lookup throughput (`24.8 ns` 8T / `143.6 ns` 1T), `InstanceTls::get` (`2.91 ns` `u64` / `25.7 ns` `String`), `WorkStealingQueue` owner push+pop (`21.3 ns`), and `Pipeline::execute_stream` (`307 ns/item`), and relaxed `LockFreeFreelist::push` / `deallocate_to_fast_bin` / `store_block_width` / `load_block_width` orderings.
   - `S25-R5` (`b74f6fe`): Made `InstanceTls` / `CleanupHandle` `drop` / `TlsMatrix` re-entrant and thread-exit-safe (`TLS_MATRICES.try_with`, borrow-free `T::drop`/`T::clone`/`T::default`, heap-backed lazy row table `Vec<Option<*mut [TlsCell<T>]>>` supporting `ROWS = 4_000_000`, and compile-time size/alignment assertions in `AsAtomic`).
   - `S25-R6` (`48438fe`): Fixed `WorkStealingExecutor::is_idle()` to load `total_queued()` before `active_tasks`, synchronized `submit()` vs `shutdown()` under queue locks, enforced `stage_timeout` on synchronous pipeline stages (`execute_single`, `execute_stream`, `process_batch`), and isolated per-call `local_stage_stats` in `execute_stream`.
   - `S25-R7` & `S25-R9` (`0932ce7`, `b74f6fe`, `48438fe`): Added 65,536+ ID churn tests for `InstanceTls` (after unifying `allocate_id` and `CleanupHandle` `drop` on `global_registry`), asserted exact `expected_id.as_u64()` in `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected`, computed fallback results on `spawn_blocking` failure in `ParallelLoudsTrie::parallel_contains`/`parallel_prefix_search`, and added multi-threaded `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` tests with non-atomic payload writes for `WorkStealingExecutor`, `FiberPool`, `Pipeline`, and `ParallelLoudsTrie`.
5. **Phase D (`B7`, `B8`, `D1`, `D9`, `D10`)**:
   - `src/thread/` undocumented `unsafe`: `2 -> 0` (`56/56 = 100.0%`); `src/concurrency/` `unsafe`: `0/0`; `src/fsa/` `unsafe`: `131 -> 117 -> 99` (`99/99 = 100.0%`); crate-wide `max_undocumented`: `6 -> 4`.
   - `src/concurrency/` honesty markers (`D1`): `3 -> 0`; crate-wide `max_markers`: `30 -> 27`.
   - `src/thread/` `.expect()` (`D10`): `8 -> 0` (`4` in `instance_tls.rs` + `4` in `linux_futex.rs`); `src/concurrency/` `.expect()`: `5 -> 0`; crate-wide `max_expects`: `95 -> 82`.
   - `src/concurrency/pipeline.rs` (`2,183 -> 2,366` lines) split into 4 `< 800`-line submodules (`D9`): `src/concurrency/pipeline/mod.rs` (`618` lines), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`797` lines), and `src/concurrency/pipeline/integration_tests.rs` (`797` lines).
   - Gate baseline ratcheted (`B8`): `lib_debug: 3151 -> 3183`, `lib_release: 3168 -> 3200`, `doctests: 229`, `compile_fail` doctests: `7 -> 8`, `all_targets: 3812 -> 3845`.

---

## 2. Commit Table (`b8e7f8b..48438fe`)

| # | Commit | Parent | Date | Subject | Files | +Ins / −Del |
|---|--------|--------|------|---------|-------|-------------|
| 1 | `b8e7f8b` | `a8d8b39` | 2026-10-06 | fix(io,memory): restrict dense object table to sequential IDs, pack depth state into u64, and tighten S23 tests (S24-R1, S24-R3, S24-R4) | 3 | +380 / −188 |
| 2 | `39814be` | `b8e7f8b` | 2026-10-06 | docs(review): fix Stage 10 io.md identifiers/citations and add scripts/audit_ledger.py (S24-R2) | 2 | +375 / −26 |
| 3 | `05d4245` | `39814be` | 2026-10-06 | fix(thread): fix x86_64 inline asm syntax, AsAtomic mutable provenance, futex wake/wait/ABI bugs, and InstanceTls drop/reentrancy [C10, C10.1, B7, D10] | 3 | +509 / −219 |
| 4 | `a8ac402` | `05d4245` | 2026-10-06 | fix(concurrency): harden work_stealing, fiber_pool, parallel_trie, and pipeline [C10, C10.1, B7, D10] | 5 | +926 / −231 |
| 5 | `d96a162` | `a8ac402` | 2026-10-06 | refactor(concurrency): split pipeline.rs into < 800-line submodules [D9] | 5 | +2382 / −2366 |
| 6 | `a0bcdff` | `d96a162` | 2026-10-06 | fix(sync): audit and upgrade weak-memory atomic orderings across cspp, lockfree_pool, and lru_map [C10.1] | 4 | +161 / −44 |
| 7 | `6571c89` | `a0bcdff` | 2026-10-07 | fix(sync): make SharedPool byte access word-atomic, pin cas_weak_u128 regs, and ratchet Stage 11 gates [C10, C10.1, B7, B8, D10] | 10 | +214 / −270 |
| 8 | `00652d2` | `6571c89` | 2026-10-07 | docs(review): add Stage 11 thread_and_concurrency.md ledger [C10, C10.1] | 1 | +356 / −0 |
| 9 | `f7e39ad` | `00652d2` | 2026-10-07 | fix(io): recover S24-R3 Vec<u64> and Vec<Vec<u32>> decode speed and add s23_serialization_bench [S25-R8, S24-R2] | 4 | +170 / −31 |
| 10 | `0932ce7` | `f7e39ad` | 2026-10-07 | fix(fsa,memory): bound ConcurrentCsppTrie lookup zpath reads to zlen and relax internal freelist/width atomics [S25-R1, S25-R4, S25-R9] | 2 | +300 / −110 |
| 11 | `b74f6fe` | `0932ce7` | 2026-10-07 | fix(thread): make InstanceTls drop/clone/default re-entrant, heap-allocate row table, and assert AsAtomic alignment [S25-R4, S25-R5, S25-R7, S25-R9] | 2 | +488 / −166 |
| 12 | `48438fe` | `b74f6fe` | 2026-10-07 | fix(concurrency): fix WorkStealingExecutor::spawn/is_idle/shutdown, sync stage_timeout, and add multi-threaded tests [S25-R2, S25-R4, S25-R6, S25-R7, S25-R9] | 6 | +566 / −144 |

---

## 3. `C10` Findings & Fixes in `src/thread/` (`05d4245`, `6571c89`, `b74f6fe`)

### 3.1 `src/thread/atomic_ext.rs`
- **`C10.A` (Broken AT&T syntax in Intel-syntax `std::arch::asm!` + `rbx` register-aliasing in `cas_weak_u128`)**:
  - **Before**: `cas_weak_u64_asm` (`39814be:src/thread/atomic_ext.rs:142-159`), `atomic_increment_u64` (`39814be:src/thread/atomic_ext.rs:196-212`), and `atomic_exchange_u64` (`39814be:src/thread/atomic_ext.rs:217-231`) used AT&T addressing syntax `({ptr})` and reversed operand order without `options(att_syntax)`. Furthermore, `cas_weak_u128` (`39814be:src/thread/atomic_ext.rs:165-193`) passed `in("rbx") desired_lo` (rejected by `rustc` because LLVM reserves `rbx`) and had a wrong `#[cfg(target_feature = "cmpxchg16b")]` gate (`39814be:src/thread/atomic_ext.rs:164`). Under `-Zsanitizer=thread` (`--release`), LLVM allocated generic `ptr = in(reg) ptr` (`05d4245:src/thread/atomic_ext.rs:193`) to `%rbx`, so `"xchg rbx, {desired_lo:r}"` overwrote `ptr` with `desired_lo` before `lock cmpxchg16b [{ptr}]`.
  - **After**: Replaced `({ptr})` and reversed operand order with Intel syntax `[{ptr}]` in `cas_weak_u64_asm` (`05d4245:src/thread/atomic_ext.rs:151-169`), `atomic_increment_u64` (`05d4245:src/thread/atomic_ext.rs:211-226`), and `atomic_exchange_u64` (`05d4245:src/thread/atomic_ext.rs:233-248`). In `cas_weak_u128` (`6571c89:src/thread/atomic_ext.rs:176-203`), replaced `#[cfg(target_feature = "cmpxchg16b")]` with `#[target_feature(enable = "cmpxchg16b")]` and pinned `ptr` to `in("rdi")`, `desired_lo` to `inout("rsi")`, and `expected_lo` to `inout("rax")` so LLVM can never allocate `ptr` or `desired_lo` to `rbx` across `xchg rbx, rsi`.
  - **Regression test**: `test_c10_a_x86_64_asm_exchange_increment_and_cas_modify_memory` (`6571c89:src/thread/atomic_ext.rs:619-652`).
- **`C10.B` (`atomic_maximize`, `atomic_minimize`, and `update_if` panicking on `Ordering::Release` / `Ordering::AcqRel`)**:
  - **Before**: `atomic_maximize` (`39814be:src/thread/atomic_ext.rs:41-54`), `atomic_minimize` (`39814be:src/thread/atomic_ext.rs:57-70`), and `update_if` (`39814be:src/thread/atomic_ext.rs:101-114`) passed `order` directly to `self.load(order)` and as the `compare_exchange_weak` failure ordering, panicking at runtime whenever `order` was `Ordering::Release` or `Ordering::AcqRel` (`"there is no such thing as a release load"`).
  - **After**: Added `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) mapping `Release -> Relaxed` and `AcqRel -> Acquire`, and used `load_order = load_ordering(order)` in `atomic_maximize` (`05d4245:src/thread/atomic_ext.rs:50-64`), `atomic_minimize` (`05d4245:src/thread/atomic_ext.rs:67-81`), and `update_if` (`05d4245:src/thread/atomic_ext.rs:112-125`).
  - **Regression test**: `test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel` (`6571c89:src/thread/atomic_ext.rs:655-665`).
- **`C10.C` + `S25-R5` (`AsAtomic::as_atomic(&self)` unsound shared-reference-to-atomic cast + compile-time size/alignment assertions)**:
  - **Before**: `AsAtomic::as_atomic(&self) -> &Self::Atomic` (`39814be:src/thread/atomic_ext.rs:309-363`) transmuted `&T` (shared read-only reference without `UnsafeCell`) into `&Atomic*`, which is immediate Undefined Behavior under Rust's aliasing model (`SharedReadOnly` provenance cannot be used for atomic writes/RMWs) and created data races with concurrent `&T` readers.
  - **After**: Changed `AsAtomic::as_atomic(&mut self) -> &Self::Atomic` (`05d4245:src/thread/atomic_ext.rs:340-396`) across all integer and pointer implementations, added a `compile_fail` doctest on `AsAtomic` (`05d4245:src/thread/atomic_ext.rs:332-339`) proving shared `&u64` cannot call `as_atomic()`, and added `const` size and alignment assertions (`size_of::<T>() == size_of::<Self::Atomic>()` and `align_of::<T>() >= align_of::<Self::Atomic>()`) in `impl_as_atomic!` (`b74f6fe:src/thread/atomic_ext.rs:357-379`) and `impl<T> AsAtomic<*mut T> for *mut T` (`b74f6fe:src/thread/atomic_ext.rs:395-420`).
  - **Regression test**: `test_c10_c_as_atomic_from_mut_is_miri_clean` (`b74f6fe:src/thread/atomic_ext.rs:684-695`).

### 3.2 `src/thread/linux_futex.rs`
- **`C10.D` (`FutexCondvar::notify_all` and `FutexRwLock::unlock_write` waking only 1 waiter due to `u32::MAX as i32 == -1`)**:
  - **Before**: `sys::futex_wake` (`39814be:src/thread/linux_futex.rs:73-85`) and `LinuxFutex::futex_wake` (`39814be:src/thread/linux_futex.rs:113-127`) passed `count` directly to `sys::futex` (`39814be:src/thread/linux_futex.rs:25-45`), which passed `u32::MAX` (`-1` as signed 32-bit `int nr_wake` in Linux `FUTEX_WAKE_PRIVATE`), causing `FutexCondvar` `notify_all` (`39814be:src/thread/linux_futex.rs:283-287`) and `FutexRwLock` `unlock_write` (`39814be:src/thread/linux_futex.rs:384-387`, `futex_wake` at `386`) to wake at most 1 waiter.
  - **After**: Clamped `let clamped = count.min(i32::MAX as u32)` in `sys::futex_wake` (`05d4245:src/thread/linux_futex.rs:82-98`) and passed `i32::MAX as u32` in `FutexCondvar` `notify_all` (`05d4245:src/thread/linux_futex.rs:313-319`) and `FutexRwLock` `unlock_write` (`05d4245:src/thread/linux_futex.rs:418-421`, `futex_wake` at `420`).
  - **Regression test**: `test_c10_d_notify_all_wakes_all_waiters` (`05d4245:src/thread/linux_futex.rs:590-626`).
- **`C10.E` (`sys::futex_wait` `tv_sec` `as i64` wraparound on `Duration::MAX` + `sys::futex` variadic C ABI mismatch under Miri + `FutexMutex`/`FutexCondvar`/`FutexRwLock` ordering & `.expect()` removal)**:
  - **Before**:
    1. `sys::futex_wait` (`39814be:src/thread/linux_futex.rs:49-69`) cast `d.as_secs() as i64`, wrapping `u64::MAX` to `-1` and failing with `EINVAL` (`errno 22`).
    2. `sys::futex` (`39814be:src/thread/linux_futex.rs:25-45`) cast all arguments (`uaddr`, `op`, `val`, `timeout`, `uaddr2`, `val3`) `as usize` when calling `libc::syscall(SYS_futex, ...)`, which Miri rejected as UB (`scalar size mismatch: expected 4 bytes but got 8 bytes`) on 32-bit `op`/`val`/`val3` parameters.
    3. `FutexMutex::lock` (`39814be:src/thread/linux_futex.rs:143-182`) used `Ordering::Relaxed` on the lock handoff CAS and spin-load; `FutexCondvar::wait` (`39814be:src/thread/linux_futex.rs:241-265`) and `notify_one`/`notify_all` used `Ordering::Relaxed` on `sequence`; `FutexRwLock::read` (`39814be:src/thread/linux_futex.rs:323-350`) and `write` (`39814be:src/thread/linux_futex.rs:353-391`) used `Ordering::Relaxed` on lock-state CAS loops; and `FutexMutex`/`FutexRwLock` used 4 production `.expect()` calls (`39814be:src/thread/linux_futex.rs:208, 413, 456, 480`).
  - **After**:
    1. `sys::futex_wait` (`05d4245:src/thread/linux_futex.rs:50-75`) clamps `d.as_secs().min(i64::MAX as u64) as i64` and binds `let ts: timespec;` in the outer frame.
    2. `sys::futex` (`05d4245:src/thread/linux_futex.rs:30-43`) passes `(SYS_futex, uaddr, op, val, timeout, uaddr2, val3)` with their exact C types.
    3. Upgraded `FutexMutex` methods `lock` (`05d4245:src/thread/linux_futex.rs:158-200`) and `unlock` (`05d4245:src/thread/linux_futex.rs:206-212`), `FutexCondvar` methods `wait` (`05d4245:src/thread/linux_futex.rs:259-275`), `wait_timeout` (`05d4245:src/thread/linux_futex.rs:278-302`), `notify_one` (`05d4245:src/thread/linux_futex.rs:304-310`), and `notify_all` (`05d4245:src/thread/linux_futex.rs:313-319`), and `FutexRwLock` methods `read` (`05d4245:src/thread/linux_futex.rs:355-382`) and `write` (`05d4245:src/thread/linux_futex.rs:385-415`) to `Acquire`/`Release`/`AcqRel` with `// ORDERING:` comments on all remaining `Relaxed` spin/counter loads, and replaced all 4 production `.expect()` calls (`D10`).
  - **Regression test**: `test_c10_e_futex_wait_duration_max_does_not_einval` (`05d4245:src/thread/linux_futex.rs:629-638`).

### 3.3 `src/thread/instance_tls.rs`
- **`C10.F1` (`CleanupHandle` `drop` leaving current thread's value alive until ID recycling + `RefCell` borrow panic on re-entrant `T::drop`)**:
  - **Before**: `CleanupHandle` `drop` (`39814be:src/thread/instance_tls.rs:419-439`) only pushed `(self.id, self.generation.wrapping_add(1))` to `free_ids` without clearing the dropping thread's `TlsMatrix` cell, leaving `T` alive on the current thread until another `InstanceTls<T>` reused that slot; furthermore, `InstanceTls::set` (`39814be:src/thread/instance_tls.rs:178-194`) and `InstanceTls::remove` (`39814be:src/thread/instance_tls.rs:197-209`) dropped the old `Option<(u32, T)>` inside `TLS_MATRICES.with(|matrices| { let mut matrices = matrices.borrow_mut(); ... })`, so any `T::drop` touching `InstanceTls` panicked with `RefCell already borrowed`.
  - **After**: In `6571c89` (`6571c89:src/thread/instance_tls.rs:197-230, 410-466`) and further hardened in `b74f6fe` (`S25-R4`/`S25-R5`, §7.3), `InstanceTls::set` (`b74f6fe:src/thread/instance_tls.rs:357-373`), `InstanceTls::remove` (`b74f6fe:src/thread/instance_tls.rs:376-399`), `InstanceTls::get` (`b74f6fe:src/thread/instance_tls.rs:242-302`), `InstanceTls::get_value` (`b74f6fe:src/thread/instance_tls.rs:306-345`), and `CleanupHandle` `drop` (`b74f6fe:src/thread/instance_tls.rs:574-618`) release the `TLS_MATRICES` `RefCell` borrow before running `T::drop`, `T::clone`, or `T::default`, and `CleanupHandle` `drop` uses `TLS_MATRICES.try_with` + `try_borrow_mut` so thread-exit destructor ordering never aborts.
  - **Regression test**: `test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic` (`b74f6fe:src/thread/instance_tls.rs:993-1026`).
- **`C10.F2` (Zero-dimension `ROWS == 0` / `COLS == 0` / `POOL_SIZE == 0` and `ROWS.checked_mul(COLS)` overflow panics + production `.expect()` removal)**:
  - **Before**: `InstanceTls::allocate_id` (`39814be:src/thread/instance_tls.rs:224-260`) computed `ROWS * COLS` and `% COLS` without checking for zero or overflow, `TlsPool` `new` (`39814be:src/thread/instance_tls.rs:531-543`) allowed `POOL_SIZE == 0` (panicking with `% 0` in `get_next`: `39814be:src/thread/instance_tls.rs:546-552`), and `instance_tls.rs` had 4 production `.expect()` calls (`39814be:src/thread/instance_tls.rs:150, 226, 535, 550`).
  - **After**: Added `validate_dimensions` (`6571c89:src/thread/instance_tls.rs:95-110`; `b74f6fe:src/thread/instance_tls.rs:202-217`) using `ROWS.checked_mul(COLS)`, rejecting `ROWS == 0`, `COLS == 0`, or `> u32::MAX as usize` in `InstanceTls::new` (`6571c89:src/thread/instance_tls.rs:113-129`; `b74f6fe:src/thread/instance_tls.rs:220-238`); rejected `POOL_SIZE == 0` in `TlsPool` `new` (`6571c89:src/thread/instance_tls.rs:573-589`; `b74f6fe:src/thread/instance_tls.rs:707-723`); and removed all 4 production `.expect()` calls in `src/thread/instance_tls.rs` (`D10`).
  - **Regression test**: `test_c10_f2_zero_pool_and_overflow_dimensions_return_err` (`b74f6fe:src/thread/instance_tls.rs:1029-1037`).

---

## 4. `C10` Findings & Fixes in `src/concurrency/` (`a8ac402`, `d96a162`, `6571c89`, `48438fe`)

### 4.1 `src/concurrency/work_stealing.rs`
- **`C10.WS1` (`WorkStealingQueue::pop_local` stranded tasks moved into `steal_queue` by `balance()`)**:
  - **Before**: `balance()` (`05d4245:src/concurrency/work_stealing.rs:190-217`) moved half of `local_queue` into `steal_queue`, but `pop_local()` (`05d4245:src/concurrency/work_stealing.rs:148-154`) only popped from `local_queue` and `worker_loop` (`05d4245:src/concurrency/work_stealing.rs:419-453`) / `find_task()` (`05d4245:src/concurrency/work_stealing.rs:481-510`) skipped `worker_id` when stealing. On a 1-worker executor (or when all other workers were idle), tasks moved to `steal_queue` were stranded forever.
  - **After**: `pop_local()` (`a8ac402:src/concurrency/work_stealing.rs:154-186`; `48438fe:src/concurrency/work_stealing.rs:183-204`) falls back to popping from the back of `steal_queue` when `local_queue` is empty.
  - **Regression test**: `test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor` (`a8ac402:src/concurrency/work_stealing.rs:870-888`).
- **`C10.WS2` (`WorkStealingExecutor::shutdown` aborted workers immediately and accepted post-shutdown submissions)**:
  - **Before**: `shutdown()` (`05d4245:src/concurrency/work_stealing.rs:406-417`) called `handle.abort()` on every worker, discarding queued tasks and cancelling in-flight tasks mid-execution, while `submit()` (`05d4245:src/concurrency/work_stealing.rs:343-376`) never checked `self.shutdown`.
  - **After**: Checked `self.shutdown` in `submit()` (`a8ac402:src/concurrency/work_stealing.rs:427-465`; synchronized under queue locks in `48438fe:src/concurrency/work_stealing.rs:476-515`), drained all local, steal, and global queues in `worker_loop` (`a8ac402:src/concurrency/work_stealing.rs:523-620`) and `shutdown()` (`48438fe:src/concurrency/work_stealing.rs:554-602`), and awaited clean worker completion.
  - **Regression test**: `test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit` (`a8ac402:src/concurrency/work_stealing.rs:891-921`).
- **`C10.WS3` (Panicking task leaked `active_tasks` and killed the worker thread)**:
  - **Before**: `worker_loop` (`05d4245:src/concurrency/work_stealing.rs:419-453`) incremented `active_tasks` before `task.execute().await` and decremented it after; a panicking task unwound the worker loop, permanently leaking `active_tasks` and killing the worker.
  - **After**: Added `ActiveTaskGuard` (`a8ac402:src/concurrency/work_stealing.rs:563-576`; `48438fe:src/concurrency/work_stealing.rs:607-621`) and `CatchUnwindFuture` (`a8ac402:src/concurrency/work_stealing.rs:344-362`; `48438fe:src/concurrency/work_stealing.rs:393-411`) around `task.execute()` so panicking tasks always decrement `active_tasks` and record elapsed time without killing the worker.
  - **Regression test**: `test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker` (`a8ac402:src/concurrency/work_stealing.rs:924-957`).
- **`C10.WS5` (`WorkStealingQueue::steal` failed to steal when `local_queue.len() == 1`)**:
  - **Before**: `steal()` (`05d4245:src/concurrency/work_stealing.rs:157-187`) computed `let steal_amount = local.len() / 2;` (`0` when `local.len() == 1`), returning `None` even when a stealable task was waiting in `local_queue`.
  - **After**: `steal()` (`a8ac402:src/concurrency/work_stealing.rs:189-248`; `48438fe:src/concurrency/work_stealing.rs:228-256`) uses `(local.len() / 2).max(1)` so a single queued stealable task can be stolen by an idle worker.
  - **Regression test**: `test_c10_ws5_steal_takes_single_queued_stealable_task` (`a8ac402:src/concurrency/work_stealing.rs:960-974`).
- **`C10.WS6` (`WorkStealingExecutor::spawn` discarded `Fiber::id()` and `new` accepted `queue_capacity == 0`)**:
  - **Before**: `WorkStealingExecutor` `spawn` (`05d4245:src/concurrency/work_stealing.rs:553-558`) ignored `fiber.id()` and allocated a new disconnected ID, and `WorkStealingExecutor` `new` (`05d4245:src/concurrency/work_stealing.rs:283-318`) accepted `queue_capacity == 0`.
  - **After**: Rejected `queue_capacity == 0` in `WorkStealingExecutor` `new` (`a8ac402:src/concurrency/work_stealing.rs:366-404`) and preserved `fiber.id().as_u64()` in `WorkStealingExecutor` `spawn` (`a8ac402:src/concurrency/work_stealing.rs:712-745`; `48438fe:src/concurrency/work_stealing.rs:815-823`).
  - **Regression test**: `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected` (`48438fe:src/concurrency/work_stealing.rs:1057-1073`).

### 4.2 `src/concurrency/fiber_pool.rs`
- **`C10.FP1` (`FiberPool::shutdown` returned while spawned/queued fibers were still running)**:
  - **Before**: `FiberPool::shutdown` (`05d4245:src/concurrency/fiber_pool.rs:282-292`) set `self.shutdown = true` and awaited `worker_handles` (which was always empty because `FiberPool::spawn` spawns onto Tokio directly), returning before active or queued fibers finished.
  - **After**: Added `queue_semaphore` to `FiberPool` (`6571c89:src/concurrency/fiber_pool.rs:116-123`), decremented `active_fibers` via `ActiveFiberGuard` (`6571c89:src/concurrency/fiber_pool.rs:241-253`, `drop` at `6571c89:src/concurrency/fiber_pool.rs:246-252`), rejected post-shutdown `spawn()` (`6571c89:src/concurrency/fiber_pool.rs:204-220`), and acquired all `total_permits` on `queue_semaphore` in `shutdown()` (`6571c89:src/concurrency/fiber_pool.rs:328-338`; chunked in `u32::MAX` steps in `48438fe:src/concurrency/fiber_pool.rs:331-354`).
  - **Regression test**: `test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers` (`6571c89:src/concurrency/fiber_pool.rs:561-592`).
- **`C10.FP2` (`FiberPool::new` accepted `max_fibers == 0` / `initial_workers == 0`, causing `NaN` in `load_factor` / `queue_utilization`)**:
  - **Before**: `FiberPool::new` (`05d4245:src/concurrency/fiber_pool.rs:148-169`) did not validate `config`, so `max_fibers == 0` or `queue_capacity == 0` caused `0.0 / 0.0 == NaN` in `load_factor()` (`05d4245:src/concurrency/fiber_pool.rs:294-298`) and `stats()` (`05d4245:src/concurrency/fiber_pool.rs:250-265`).
  - **After**: Validated `max_fibers > 0`, `initial_workers > 0`, `initial_workers <= max_workers`, and `max_fibers.checked_add(queue_capacity)` in `FiberPool::new` (`6571c89:src/concurrency/fiber_pool.rs:149-195`) and guarded `queue_capacity == 0` in `stats()` (`6571c89:src/concurrency/fiber_pool.rs:289-325`).
  - **Regression test**: `test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan` (`6571c89:src/concurrency/fiber_pool.rs:595-622`).

### 4.3 `src/concurrency/parallel_trie.rs`
- **`C10.PT1` (`chunks(0)` / `tries.remove(0)` panics when `chunk_size == 0`, `max_workers == 0`, or `tries.len() == 1`)**:
  - **Before**: `ParallelTrieBuilder::build_louds_trie` (`05d4245:src/concurrency/parallel_trie.rs:40-75`) called `.chunks(self.chunk_size)` directly (panicking when `chunk_size == 0`) and `Semaphore::new(self.max_workers)` (deadlocking when `max_workers == 0`), while `merge_tries` (`05d4245:src/concurrency/parallel_trie.rs:88, 439`) called `tries.remove(0)` twice when `tries.len() >= 1` (panicking when `tries.len() == 1`).
  - **After**: Validated non-zero `chunk_size` and `max_workers` in `ParallelTrieBuilder::build_louds_trie` (`a8ac402:src/concurrency/parallel_trie.rs:40-54`), replaced `tries.remove(0)` with `tries.into_iter()` in `ParallelTrieBuilder` `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:102-135`) and `ParallelTrieOps` `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:481-505`), and held the write lock across replica updates in `update_replicas_locked` (`a8ac402:src/concurrency/parallel_trie.rs:202-209`).
  - **Regression test**: `test_c10_pt1_zero_chunk_size_or_workers_returns_err_instead_of_panic` (`a8ac402:src/concurrency/parallel_trie.rs:719-737`).
- **`C10.PT3` (`parallel_process`, `parallel_contains`, `parallel_prefix_search`, and `build_louds_trie` used `.expect()` on `JoinHandle` / `Semaphore`, panicking on closure panic)**:
  - **Before**: `parallel_process` (`05d4245:src/concurrency/parallel_trie.rs:324-348`), `parallel_contains` (`05d4245:src/concurrency/parallel_trie.rs:202-227`), `parallel_prefix_search` (`05d4245:src/concurrency/parallel_trie.rs:230-258`), `build_louds_trie` (`05d4245:src/concurrency/parallel_trie.rs:58, 72`), and `merge_tries` (`05d4245:src/concurrency/parallel_trie.rs:99`) called `.expect(...)` on `AcquireError` and `JoinError`, violating `D10` and crashing the caller when a closure panicked.
  - **After**: Wrapped each operation in `std::panic::catch_unwind` and mapped `JoinError` to `Err(ZiporaError::system_error(...))` in `parallel_process` (`a8ac402:src/concurrency/parallel_trie.rs:352-390`), mapped `AcquireError`/`JoinError` to `Err(ZiporaError::configuration(...))` in `build_louds_trie` (`a8ac402:src/concurrency/parallel_trie.rs:40-100`) and `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:102-135`), and replaced `.expect(...)` in `parallel_contains` (`48438fe:src/concurrency/parallel_trie.rs:235-269`) and `parallel_prefix_search` (`48438fe:src/concurrency/parallel_trie.rs:272-307`) with direct fallback evaluation, removing all 5 `.expect()` calls from `src/concurrency/` (`D10`).
  - **Regression test**: `test_c10_pt3_parallel_process_catches_closure_panic_as_err` (`a8ac402:src/concurrency/parallel_trie.rs:740-758`).

### 4.4 `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`)
- **`C10.P1` (Multi-stage `Pipeline::execute_stream` always failed with `"pipeline: channel missing for prior stage"` + dropped stage errors)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:344-478`) stored `(tx, rx)` tuples inside `channels: Vec<Option<(Sender<T>, Receiver<T>)>>` (`05d4245:src/concurrency/pipeline.rs:374-379`), took `channels[0].take()...0` on iteration `i == 0` (`05d4245:src/concurrency/pipeline.rs:418-423`, discarding `.1`), and then tried `channels[i - 1].take()...1` on iteration `i == 1` (`05d4245:src/concurrency/pipeline.rs:403-408`), so every `stages.len() >= 2` pipeline immediately failed with `Err(ZiporaError::invalid_state("pipeline: channel missing for prior stage"))`; additionally, stage processing errors were silently ignored at `05d4245:src/concurrency/pipeline.rs:455-458`.
  - **After**: Rewrote `execute_stream` (`6571c89:src/concurrency/pipeline/mod.rs:262-422`; `48438fe:src/concurrency/pipeline/mod.rs:267-438`) to thread `current_rx` directly from stage `i` to stage `i + 1`, flush partial batches at end-of-stream, and propagate stage errors/timeouts across both single-stage and multi-stage pipelines.
  - **Regression test**: `test_c10_p1_execute_stream_multi_stage_and_error_propagation` (`6571c89:src/concurrency/pipeline/integration_tests.rs:679-725`).
- **`C10.P2` (`items_in_flight` leaked on stage error or timeout in `execute_stream` and `process_batch` and was not tracked in `execute_single`)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:433-464`) and `process_batch` (`05d4245:src/concurrency/pipeline.rs:481-526`) incremented `items_in_flight` before awaiting the stage future and only decremented on success, permanently leaking `items_in_flight` on stage error, timeout, or future cancellation, while `execute_single` (`05d4245:src/concurrency/pipeline.rs:302-341`) did not track `items_in_flight`.
  - **After**: Added `AtomicCountGuard` (`6571c89:src/concurrency/pipeline/mod.rs:183-202`) so `items_in_flight` is unconditionally decremented on scope exit in `execute_single` (`48438fe:src/concurrency/pipeline/mod.rs:219-245`), `execute_stream` (`48438fe:src/concurrency/pipeline/mod.rs:267-438`), and `process_batch` (`48438fe:src/concurrency/pipeline/mod.rs:441-490`).
  - **Regression test**: `test_c10_p2_items_in_flight_decremented_on_timeout_and_error` (`6571c89:src/concurrency/pipeline/integration_tests.rs:728-747`).
- **`C10.P3` (`buffer_size == 0` panicked inside `mpsc::channel(0)`, `max_in_flight == 0` produced `NaN` utilization, and `start_timeout_checker` busy-spun on `Duration::ZERO`)**:
  - **Before**: `mpsc::channel(self.config.buffer_size)` (`05d4245:src/concurrency/pipeline.rs:377`) panicked when `buffer_size == 0`; `stats()` (`05d4245:src/concurrency/pipeline.rs:529-558`, division at line `543`) divided `items_in_flight as f64 / self.config.max_in_flight as f64` (`0.0 / 0.0 == NaN`); and `start_timeout_checker` (`05d4245:src/concurrency/pipeline.rs:240-262`) computed `self.batch_timeout / 4` (`Duration::ZERO` when `batch_timeout < 4ns`), spinning without yielding.
  - **After**: Clamped channel capacities to `.max(1)` in `execute_stream` (`6571c89:src/concurrency/pipeline/mod.rs:262-422`), guarded `max_in_flight == 0` in `stats()` (`6571c89:src/concurrency/pipeline/mod.rs:469-481`), and floored the checker sleep interval to `Duration::from_millis(1)` in `start_timeout_checker` (`6571c89:src/concurrency/pipeline/stages.rs:96-120`).
  - **Regression test**: `test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan` (`6571c89:src/concurrency/pipeline/integration_tests.rs:750-780`).
- **`C10.P5` (`BatchCollector` expired the first item after an idle period immediately)**:
  - **Before**: `BatchCollector` initialized `last_flush` to `Instant::now()` at construction (`05d4245:src/concurrency/pipeline.rs:177`) and after flushes, so when a collector sat idle for `> batch_timeout`, the very first item added had `last_flush.elapsed() >= batch_timeout` and was immediately flushed as a 1-item timeout batch on the next `check_timeout()`.
  - **After**: Reset `*last_flush = Instant::now()` when transitioning `batch` from empty to non-empty in `BatchCollector::add` (`6571c89:src/concurrency/pipeline/stages.rs:33-55`).
  - **Regression test**: `test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout` (`6571c89:src/concurrency/pipeline/integration_tests.rs:783-794`).
- **`D1` & `D9` (Honesty markers removed & `pipeline.rs` split into 4 `< 800`-line submodules)**:
  - Replaced the 3 honesty-marker comments in `BatchMapStage` `process_batch` (`6571c89:src/concurrency/pipeline/stages.rs:184-206`) and split `src/concurrency/pipeline.rs` (`2,366` lines at `a8ac402`) into `src/concurrency/pipeline/mod.rs` (`618` lines at `48438fe`), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`797` lines), and `src/concurrency/pipeline/integration_tests.rs` (`797` lines) (`d96a162`, `6571c89`, `48438fe`).

---

## 5. `C10.1` Crate-Wide Weak-Memory Ordering Audit (`05d4245`, `a8ac402`, `a0bcdff`, `6571c89`, `0932ce7`)

| Module / File | Atomic Ordering / Synchronization Upgrades & `// ORDERING:` Coverage |
|---------------|----------------------------------------------------------------------|
| `src/thread/atomic_ext.rs` | Fixed RMW initial-load/failure orderings via `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) and added `// ORDERING:` comments to `AtomicBitOps` (`05d4245:src/thread/atomic_ext.rs:426-468`). |
| `src/thread/linux_futex.rs` | Upgraded `FutexMutex`, `FutexCondvar`, and `FutexRwLock` state transitions from `Relaxed` to `Acquire`/`Release` (`05d4245:src/thread/linux_futex.rs:158-448`) and documented all remaining `Relaxed` spin/waiter-count loads with `// ORDERING:`. |
| `src/thread/instance_tls.rs` | Documented `next_id` / `next_slot` monotonic `Relaxed` `fetch_add` with `// ORDERING:` (`b74f6fe:src/thread/instance_tls.rs:39, 728`). |
| `src/concurrency/mod.rs`, `work_stealing.rs`, `fiber_pool.rs`, `parallel_trie.rs`, `pipeline/mod.rs` | Upgraded lifecycle and task-accounting atomics (`WorkStealingExecutor::shutdown` at `48438fe:src/concurrency/work_stealing.rs:554-602`, `active_fibers` at `6571c89:src/concurrency/fiber_pool.rs:204-256`, and `FiberPool::shutdown` at `6571c89:src/concurrency/fiber_pool.rs:328-338`) to `Acquire`/`Release`/`AcqRel`, and added `// ORDERING:` comments to all remaining `Relaxed` ID generators and telemetry counters. |
| `src/fsa/cspp_trie_concurrent.rs` | Upgraded `SharedPool::load_meta` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:208-211`) and `SharedPool::load_bytes` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:229-233`) from `load_relaxed` to `load_acquire`, kept `LockFreeFreelist::push` at `(Release, Relaxed)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`) paired with `Acquire` in `pop` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:408-440`), replaced all non-atomic `raw_ptr` / `get_slice` / `copy_nonoverlapping` byte accesses on `SharedPool::data: Box<[AtomicU32]>` with word-atomic `SharedPool::read_bytes_acquire` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:237-249`), `SharedPool::bytes_eq_relaxed` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:258-312`), and `SharedPool::write_bytes_relaxed` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:317-331`), and added `// ORDERING:` comments across the file. Eliminated 32 `unsafe` blocks (`131 -> 117 -> 99` in `src/fsa/`) and made `make tsan_cspp` 15/15 100% ThreadSanitizer-clean. |
| `src/memory/lockfree_pool.rs` | Kept `deallocate_to_fast_bin` Treiber push CAS at `(Release, Relaxed)` (`0932ce7:src/memory/lockfree_pool.rs:689-745`) paired with `(AcqRel, Acquire)` in `pop_fast_bin` (`0932ce7:src/memory/lockfree_pool.rs:595-652`), documented `Relaxed` on `store_block_width` (`0932ce7:src/memory/lockfree_pool.rs:894-912`) and `load_block_width` (`0932ce7:src/memory/lockfree_pool.rs:915-927`) (synchronized by the enclosing skip-list/fast-bin `Release`/`Acquire` CAS), and added `// ORDERING:` comments to all `Relaxed` bump-cursor and statistics operations. |
| `src/containers/specialized/lru_map.rs` & `concurrent_lru_map.rs` | Moved `stats.entry_count.store(...)` inside the exclusive `nodes` write lock in `LruMap` methods `put` (`a0bcdff:src/containers/specialized/lru_map.rs:638-731`) and `evict_lru` (`a0bcdff:src/containers/specialized/lru_map.rs:882-944`) so concurrent operations cannot overwrite `entry_count` with a stale post-unlock read, and added `// ORDERING:` comments to all `Relaxed` list-node and statistics operations. |

---

## 6. Verbatim RED / Miri / TSan Outputs (`C10`)

### 6.1 `src/thread/` RED Outputs on `39814be`
- **`C10.A` compile-RED (`cargo test --locked --lib thread:: -- --test-threads=1`)**:
```text
error: invalid operand for instruction
   --> src/thread/atomic_ext.rs:222:18
    |
222 |                 "xchg ({ptr}), {val:r}",
    |                  ^^^^^^^^^^^^^^^^^^^^^
    |
note: instantiated into assembly here
   --> <inline asm>:2:2
    |
  2 |     xchg (rdx), rsi
    |     ^

error: invalid operand for instruction
   --> src/thread/atomic_ext.rs:202:18
    |
202 |                 "lock xadd ({ptr}), {result:r}",
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
note: instantiated into assembly here
   --> <inline asm>:3:1
    |
  3 | lock xadd (rcx), rdx
    | ^

error: invalid operand for instruction
   --> src/thread/atomic_ext.rs:147:18
    |
147 |                 "lock cmpxchg {desired:r}, ({ptr})",
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
note: instantiated into assembly here
   --> <inline asm>:2:2
    |
  2 |     lock cmpxchg rdx, (rcx)
    |     ^
```
- **`C10.B`, `C10.D`, `C10.E`, `C10.F1`, `C10.F2` runtime-RED on `39814be`**:
```text
---- thread::atomic_ext::tests::test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel stdout ----
thread 'thread::atomic_ext::tests::test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel' (2621510) panicked at /usr/local/google/home/binwu/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/sync/atomic.rs:3907:24:
there is no such thing as a release load

---- thread::instance_tls::tests::test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic stdout ----
thread 'thread::instance_tls::tests::test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic' (2621512) panicked at src/thread/instance_tls.rs:828:9:
assertion `left == right` failed: dropping InstanceTls must immediately drop the current thread's stored value
  left: 2
 right: 1

---- thread::instance_tls::tests::test_c10_f2_zero_pool_and_overflow_dimensions_return_err stdout ----
thread 'thread::instance_tls::tests::test_c10_f2_zero_pool_and_overflow_dimensions_return_err' (2621513) panicked at src/thread/instance_tls.rs:855:9:
TlsPool with POOL_SIZE == 0 must return Err

---- thread::linux_futex::tests::test_c10_d_notify_all_wakes_all_waiters stdout ----
thread 'thread::linux_futex::tests::test_c10_d_notify_all_wakes_all_waiters' (2621528) panicked at src/thread/linux_futex.rs:570:9:
assertion `left == right` failed: FUTEX_WAKE_PRIVATE with u32::MAX must clamp to i32::MAX and wake all 4 waiters, got 1
  left: 1
 right: 4

---- thread::linux_futex::tests::test_c10_e_futex_wait_duration_max_does_not_einval stdout ----
thread 'thread::linux_futex::tests::test_c10_e_futex_wait_duration_max_does_not_einval' (2621535) panicked at src/thread/linux_futex.rs:587:9:
futex_wait with Duration::MAX must clamp tv_sec to i64::MAX instead of failing with EINVAL: Err(SystemError { message: "Futex wait failed: errno 22" })
```
- **Miri RED on `sys::futex` (`39814be:src/thread/linux_futex.rs:35-43`)**:
```text
test thread::linux_futex::tests::test_futex_mutex_basic ... error: Undefined Behavior: scalar size mismatch: expected 4 bytes but got 8 bytes instead
   --> src/thread/linux_futex.rs:35:13
    |
 35 | /             syscall(
 36 | |                 SYS_futex,
 37 | |                 uaddr as usize,
 38 | |                 op as usize,
 39 | |                 val as usize,
 40 | |                 timeout as usize,
 41 | |                 uaddr2 as usize,
 42 | |                 val3 as usize,
 43 | |             ) as c_int
    | |_____________^ Undefined Behavior occurred here
```

### 6.2 `src/concurrency/` RED Outputs on `05d4245`
```text
---- concurrency::fiber_pool::tests::test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers stdout ----
thread 'concurrency::fiber_pool::tests::test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers' (2651143) panicked at src/concurrency/fiber_pool.rs:536:9:
assertion `left == right` failed: FiberPool::shutdown() returned before all spawned/queued fibers completed
  left: 0
 right: 3

---- concurrency::fiber_pool::tests::test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan stdout ----
thread 'concurrency::fiber_pool::tests::test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan' (2651144) panicked at src/concurrency/fiber_pool.rs:549:9:
FiberPool::new with max_fibers == 0 must return Err

---- concurrency::parallel_trie::tests::test_c10_pt1_zero_chunk_size_or_workers_returns_err_instead_of_panic stdout ----
thread 'concurrency::parallel_trie::tests::test_c10_pt1_zero_chunk_size_or_workers_returns_err_instead_of_panic' (2651153) panicked at src/concurrency/parallel_trie.rs:53:14:
chunk size must be non-zero

---- concurrency::parallel_trie::tests::test_c10_pt3_parallel_process_catches_closure_panic_as_err stdout ----
thread '<unnamed>' (2651213) panicked at src/concurrency/parallel_trie.rs:707:17:
intentional closure panic in parallel_process
thread 'concurrency::parallel_trie::tests::test_c10_pt3_parallel_process_catches_closure_panic_as_err' (2651154) panicked at src/concurrency/parallel_trie.rs:346:10:
parallel_process blocking task panicked: JoinError::Panic(Id(16), "intentional closure panic in parallel_process", ...)

---- concurrency::pipeline::tests::test_c10_p1_execute_stream_multi_stage_and_error_propagation stdout ----
thread 'concurrency::pipeline::tests::test_c10_p1_execute_stream_multi_stage_and_error_propagation' (2654388) panicked at src/concurrency/pipeline.rs:2202:14:
2-stage execute_stream must succeed: InvalidState { message: "pipeline: channel missing for prior stage" }

---- concurrency::pipeline::tests::test_c10_p2_items_in_flight_decremented_on_timeout_and_error stdout ----
thread 'concurrency::pipeline::tests::test_c10_p2_items_in_flight_decremented_on_timeout_and_error' (2651274) panicked at src/concurrency/pipeline.rs:2224:9:
assertion `left == right` failed: items_in_flight must be decremented when process_batch fails
  left: 3
 right: 0

---- concurrency::pipeline::tests::test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan stdout ----
thread 'concurrency::pipeline::tests::test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan' (2651275) panicked at src/concurrency/pipeline.rs:2240:9:
Pipeline::stats().utilization must not be NaN when max_in_flight == 0

---- concurrency::pipeline::tests::test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout stdout ----
thread 'concurrency::pipeline::tests::test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout' (2651276) panicked at src/concurrency/pipeline.rs:2273:9:
assertion `left == right` failed: first item after an idle period must start a fresh batch_timeout window
  left: Some([1])
 right: None

---- concurrency::work_stealing::tests::test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor stdout ----
thread 'concurrency::work_stealing::tests::test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor' (2651294) panicked at src/concurrency/work_stealing.rs:703:9:
assertion `left == right` failed: pop_local must drain tasks moved to steal_queue by balance()
  left: 2
 right: 4

---- concurrency::work_stealing::tests::test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit stdout ----
thread 'concurrency::work_stealing::tests::test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit' (2651295) panicked at src/concurrency/work_stealing.rs:730:9:
assertion `left == right` failed: shutdown() must drain and finish all submitted tasks instead of aborting workers
  left: 0
 right: 8

---- concurrency::work_stealing::tests::test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker stdout ----
thread 'concurrency::work_stealing::tests::test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker' (2651296) panicked at src/concurrency/work_stealing.rs:749:21:
intentional task panic
thread 'concurrency::work_stealing::tests::test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker' (2651296) panicked at src/concurrency/work_stealing.rs:755:9:
assertion `left == right` failed: panicking task must not permanently leak active_tasks
  left: 1
 right: 0

---- concurrency::work_stealing::tests::test_c10_ws5_steal_takes_single_queued_stealable_task stdout ----
thread 'concurrency::work_stealing::tests::test_c10_ws5_steal_takes_single_queued_stealable_task' (2651298) panicked at src/concurrency/work_stealing.rs:789:9:
steal() must be able to steal a single queued stealable task

---- concurrency::work_stealing::tests::test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected stdout ----
thread 'concurrency::work_stealing::tests::test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected' (2651299) panicked at src/concurrency/work_stealing.rs:798:9:
WorkStealingExecutor::new with queue_capacity == 0 must return Err
```

---

## 7. Stage 11 Re-Review Remediation (`S25-R1..R9`: `f7e39ad..48438fe`)

### 7.1 `f7e39ad` — `S25-R8` & `S24-R2` (`src/io/data_input.rs`, `src/io/smart_ptr.rs`, `benches/s23_serialization_bench.rs`, `Cargo.toml`)
- Outlined `#[cold] #[inline(never)] fn unexpected_eof_err() -> ZiporaError` (`f7e39ad:src/io/data_input.rs:213-217`), added `DIRTY_OUTER_SER_STATE` (`f7e39ad:src/io/smart_ptr.rs:35`) and `DIRTY_OUTER_DE_STATE` (`f7e39ad:src/io/smart_ptr.rs:36`), and split `Vec<T>::deserialize` (`f7e39ad:src/io/smart_ptr.rs:1210-1237`) into a non-unwinding `0..cap` preallocated push loop plus `#[cold] #[inline(never)] fn deserialize_vec_tail` (`f7e39ad:src/io/smart_ptr.rs:1184-1195`), recovering `Vec<u64>` decode to `0.949 ns/elem` (vs `0.99 ns` at `a8d8b39`) and `Vec<Vec<u32>>` decode to `111.7 ns/inner Vec` (vs `118.5 ns` at `a8d8b39`). Committed `benches/s23_serialization_bench.rs` (`f7e39ad:benches/s23_serialization_bench.rs:1-82`). See `docs/review/io.md` §12.2.

### 7.2 `0932ce7` — `S25-R1`, `S25-R4`, `S25-R9` (`src/fsa/cspp_trie_concurrent.rs`, `src/memory/lockfree_pool.rs`)
- **`S25-R1` (HIGH — `ConcurrentCsppTrie` 256-byte `read_zpath` over-read panic on `lookup` / `contains` near pool end)**:
  - **Root cause**: `6571c89` replaced `ConcurrentNodeView::zpath_slice` with `ConcurrentNodeView::read_zpath(&self, buf: &mut [u8; 256])`, which called `self.pool.read_bytes_acquire(self.curr as usize + self.zpath_offset(), &mut buf[..zlen])` (`zlen = 256` whenever `zpath_len() == 255`), reading 64 `AtomicU32` words regardless of how many `zpath` bytes actually existed in the node (`new_suffix_chain` allocates only `header + zlen.div_ceil(4)` words for tail leaves), panicking with `index out of bounds` whenever a node with `zlen < 255` was allocated within 64 words of `pool.data.len()`.
  - **Fix in `0932ce7:src/fsa/cspp_trie_concurrent.rs`**:
    1. Removed `ConcurrentNodeView::read_zpath` and added zero-copy word-wise `SharedPool::bytes_eq_relaxed(slot, expected)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:258-312`), which bounds `end = pos.saturating_add(zlen).min(key.len())` in `lookup_inner` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1094-1156`) and `contains_with_guard` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1164-1185`) so a reader never touches more than `zlen.div_ceil(4)` words (and returns `false` if `slot + words > self.data.len()`).
    2. Cached the `Acquire`-loaded header word `raw: u32` in `ConcurrentNodeView` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:514-573`), peeled the 256-child root fast-node step (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1100-1115`), and inlined leaf `valpos` extraction in `lookup_inner`.
    3. Relaxed `LockFreeFreelist::push` CAS from `(AcqRel, Acquire)` to `(Release, Relaxed)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`) with `// ORDERING:` comment on `pop` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:408-440`), and relaxed `deallocate_to_fast_bin` (`0932ce7:src/memory/lockfree_pool.rs:689-745`), `store_block_width` (`0932ce7:src/memory/lockfree_pool.rs:894-912`), and `load_block_width` (`0932ce7:src/memory/lockfree_pool.rs:915-927`).
  - **Regression tests**: `test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:3112-3134`) and `test_s25_r1_lookup_multi_insert_near_pool_end` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:3137-3161`).
  - **Verified timings (`taskset -c 62`, 9-round minimum)**: `cspp_lookup_8t = 24.80 ns/lookup` (vs `25.5 ns` at `a8d8b39`, `34.2 ns` at `6571c89`), `cspp_lookup_1t = 143.61 ns/lookup` (vs `213.6 ns` at `6571c89`).

### 7.3 `b74f6fe` — `S25-R4`, `S25-R5`, `S25-R7`, `S25-R9` (`src/thread/atomic_ext.rs`, `src/thread/instance_tls.rs`)
- **`S25-R4`, `S25-R5`, `S25-R7`, `S25-R9` (`InstanceTls` re-entrancy, thread-exit safety, heap row table, ID pool unification, and `O(1)` MRU cache)**:
  - **Fixes in `b74f6fe:src/thread/instance_tls.rs`**:
    1. Replaced inline `[Option<Box<TlsRow<T, COLS>>>; ROWS]` (which stack-overflowed on large `ROWS` such as `ROWS = 4_000_000`) with a heap-backed lazy row pointer table `rows: Vec<Option<*mut [TlsCell<T>]>>` in `TlsMatrix` (`b74f6fe:src/thread/instance_tls.rs:45-47`) and `alloc_row` (`b74f6fe:src/thread/instance_tls.rs:518-529`), `ensure_cell` (`b74f6fe:src/thread/instance_tls.rs:531-541`), and `lookup_cell` (`b74f6fe:src/thread/instance_tls.rs:543-550`).
    2. Added `TlsMatrices` (`b74f6fe:src/thread/instance_tls.rs:57-60`) with a 1-entry MRU `(TypeId, *mut ())` cache in `get_or_init_matrix` (`b74f6fe:src/thread/instance_tls.rs:71-81`) and `lookup_matrix` (`b74f6fe:src/thread/instance_tls.rs:111-125`), re-entrancy-safe `TlsMatrices` `drop` (`b74f6fe:src/thread/instance_tls.rs:129-152`), and `TLS_MATRICES.try_with` + `try_borrow_mut`, moving `T::clone()`, `T::default()`, and `T::drop()` outside the `RefCell` borrow in `get` (`b74f6fe:src/thread/instance_tls.rs:242-302`), `get_value` (`b74f6fe:src/thread/instance_tls.rs:306-345`), `set` (`b74f6fe:src/thread/instance_tls.rs:357-373`), `remove` (`b74f6fe:src/thread/instance_tls.rs:376-399`), `TlsMatrix` `drop` (`b74f6fe:src/thread/instance_tls.rs:554-564`), and `CleanupHandle` `drop` (`b74f6fe:src/thread/instance_tls.rs:574-618`).
    3. Unified `allocate_id` (`b74f6fe:src/thread/instance_tls.rs:414-495`) and `CleanupHandle` `drop` (`b74f6fe:src/thread/instance_tls.rs:574-618`) on `global_registry` (`b74f6fe:src/thread/instance_tls.rs:189-191`) so `test_instance_ids_are_recycled_after_drop` and 65,536+ ID churn tests exercise the same `(id, generation)` free list.
  - **Regression tests**:
    - `test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak` (`b74f6fe:src/thread/instance_tls.rs:1041-1081`)
    - `test_s25_r7_parallel_80k_churn_across_threads` (`b74f6fe:src/thread/instance_tls.rs:1085-1132`)
    - `test_s25_r5_reentrant_drop_at_thread_exit_does_not_abort` (`b74f6fe:src/thread/instance_tls.rs:1136-1175`)
    - `test_s25_r9_reentrant_clone_and_large_rows_do_not_panic_or_overflow_stack` (`b74f6fe:src/thread/instance_tls.rs:1179-1211`)
  - **Verified timings (`taskset -c 62`, 9-round minimum)**: `tls_get_u64 = 2.91 ns/call` (was `25.0 ns` at `6571c89`), `tls_get_string = 25.75 ns/call` (was `42.7 ns` at `6571c89`).

### 7.4 `48438fe` — `S25-R2`, `S25-R4`, `S25-R6`, `S25-R7`, `S25-R9` (`src/concurrency/{work_stealing,fiber_pool,parallel_trie}.rs`, `src/concurrency/pipeline/{mod,tests,integration_tests}.rs`)
- **`S25-R2` (HIGH — `WorkStealingExecutor::spawn(fiber)` hung after global-init runtime drop or nested fiber await on 1 worker)**:
  - Fixed `WorkStealingExecutor::spawn` (`48438fe:src/concurrency/work_stealing.rs:815-823`) to spawn `fiber` directly onto the current Tokio runtime via `tokio::spawn(async move { fiber.await })` while preserving `fiber.id().as_u64()`.
  - **Regression test**: `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` (`48438fe:src/concurrency/work_stealing.rs:1076-1149`).
- **`S25-R4` & `S25-R6` (`WorkStealingQueue` lock-free steal check, `is_idle()` load order, `submit`/`shutdown` synchronization, sync `stage_timeout`, and isolated `local_stage_stats`)**:
  - Added `steal_len: AtomicUsize` (`48438fe:src/concurrency/work_stealing.rs:107`) and $O(1)$ back-check in `insert_by_priority` (`48438fe:src/concurrency/work_stealing.rs:124-135`), recovering `WorkStealingQueue` owner push+pop to `21.34 ns/op` (was `38.6 ns` at `6571c89`).
  - Reordered `WorkStealingExecutor::is_idle()` (`48438fe:src/concurrency/work_stealing.rs:765-767`) to check `self.total_queued() == 0 && self.stats.active_tasks.load(Ordering::Acquire) == 0`, and synchronized `submit()` (`48438fe:src/concurrency/work_stealing.rs:476-515`) with `shutdown()` (`48438fe:src/concurrency/work_stealing.rs:554-602`) via `try_push_local_unless_shutdown` (`48438fe:src/concurrency/work_stealing.rs:161-180`) and a final drain pass in `shutdown()`.
  - Enforced `self.config.stage_timeout` on synchronous stages (`elapsed > stage_timeout`) in `Pipeline::execute_single` (`48438fe:src/concurrency/pipeline/mod.rs:219-245`), `Pipeline::execute_stream` (`48438fe:src/concurrency/pipeline/mod.rs:267-438`), and `Pipeline::process_batch` (`48438fe:src/concurrency/pipeline/mod.rs:441-490`), isolated per-call `local_stage_stats` in `execute_stream`, and added a `try_recv()` batching fast path (`307.4 ns/item` vs `1,136 ns/item` at `6571c89`).
- **`S25-R7` & `S25-R9` (Multi-threaded tests with non-atomic payload writes + `ParallelLoudsTrie` fallback evaluation)**:
  - Computed exact fallback results from a cloned `replicas` snapshot on `spawn_blocking` failure in `ParallelLoudsTrie::parallel_contains` (`48438fe:src/concurrency/parallel_trie.rs:235-269`) and `parallel_prefix_search` (`48438fe:src/concurrency/parallel_trie.rs:272-307`), and asserted `handle.id() == expected_id.as_u64()` in `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected` (`48438fe:src/concurrency/work_stealing.rs:1057-1073`).
  - Added multi-threaded `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` tests with non-atomic payload writes across all 4 concurrency modules:
    - `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` (`48438fe:src/concurrency/work_stealing.rs:1152-1228`)
    - `test_s25_r7_multithreaded_fiber_pool_payload_writes_and_shutdown` (`48438fe:src/concurrency/fiber_pool.rs:626-662`)
    - `test_s25_r7_multithreaded_execute_stream_payload_writes_and_concurrent_stage_stats` (`48438fe:src/concurrency/pipeline/tests.rs:726-796`)
    - `test_s25_r7_multithreaded_parallel_louds_trie_concurrent_inserts_and_reads` (`48438fe:src/concurrency/parallel_trie.rs:776-805`)

### 7.5 Sanitizer & Gate Verification (`S25`)
- `touch src/lib.rs && make sanity`: `lib_debug = 3183`, `lib_release = 3200`, `doctests = 229`, `compile_fail = 8`, `all_targets = 3845`, `max_undocumented = 4` (`thread 56/56 = 100.0%`, `concurrency 0/0 = 100.0%`, `fsa 99/99 = 100.0%`), `max_markers = 27` (`0` in `concurrency`), `max_expects = 82` (`0` in `thread` and `concurrency`), `line_count` (`0` files `> 800` lines in `src/concurrency/pipeline/`).
- `make miri_thread`: `28 passed; 0 failed; 1 ignored` in both SIMD (`-C target-cpu=haswell`) and scalar (`MIRI_SCALAR=1`) builds.
- `make miri_cspp`: `17 passed; 0 failed; 0 ignored` in both SIMD and scalar builds.
- `make tsan_concurrency`: `29 passed; 0 failed` (`thread::`) + `85 passed; 0 failed` (`concurrency::`) under `-Zsanitizer=thread`.
- `make tsan_cspp`: `17 passed; 0 failed` (`fsa::cspp_trie_concurrent`) under `-Zsanitizer=thread`.
- `make tsan_pool`: `33 passed; 0 failed` (`memory::lockfree`) under `-Zsanitizer=thread`.
- `python3 scripts/audit_ledger.py docs/review/io.md docs/review/thread_and_concurrency.md`: `0` errors across `1,098` citations in `docs/review/io.md` and `250` citations in `docs/review/thread_and_concurrency.md` (`15` errors caught across all 8 adversarial test cases).
