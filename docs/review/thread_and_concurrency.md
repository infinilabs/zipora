# Stage 11 Review Ledger: `src/thread/` + `src/concurrency/` + `C10.1` Weak-Memory Ordering Audit

## 1. Scope & Summary

Stage 11 covers:
1. **`S24-R1..R4`** carry-over fixes from the Stage 10 re-review (`b8e7f8b..39814be`, also documented in `docs/review/io.md` §15).
2. **`C10` (`src/thread/` + `src/concurrency/`)**: deep correctness, safety, panic-resilience, and resource-lifecycle audit across `src/thread/atomic_ext.rs`, `src/thread/linux_futex.rs`, `src/thread/instance_tls.rs`, `src/concurrency/mod.rs`, `src/concurrency/work_stealing.rs`, `src/concurrency/fiber_pool.rs`, `src/concurrency/parallel_trie.rs`, and `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`).
3. **`C10.1` (Crate-Wide Weak-Memory Ordering Audit)**: audit of all `Ordering::Relaxed` / `Release` / `Acquire` / `AcqRel` / `SeqCst` atomic operations across `src/thread/`, `src/concurrency/`, `src/fsa/cspp_trie_concurrent.rs`, `src/memory/lockfree_pool.rs`, `src/containers/specialized/lru_map.rs`, and `src/containers/specialized/concurrent_lru_map.rs`, adding `// ORDERING:` justifications to every `Ordering::Relaxed` site in the audited modules, upgrading weak orderings on shared-data publication/reclamation paths, and replacing all non-atomic byte reads/writes on `SharedPool` in `ConcurrentCsppTrie` with word-atomic helpers.
4. **Phase D (`B7`, `B8`, `D1`, `D9`, `D10`)**:
   - `src/thread/` undocumented `unsafe`: `2 -> 0` (`43/43 = 100.0%`); `src/concurrency/` `unsafe`: `0/0`; `src/fsa/` `unsafe`: `117 -> 99` (`99/99 = 100.0%`); crate-wide `max_undocumented`: `6 -> 4`.
   - `src/concurrency/` honesty markers (`D1`): `3 -> 0`; crate-wide `max_markers`: `30 -> 27`.
   - `src/thread/` `.expect()` (`D10`): `8 -> 0`; `src/concurrency/` `.expect()`: `5 -> 0`; crate-wide `max_expects`: `95 -> 82`.
   - `src/concurrency/pipeline.rs` (`2,183 -> 2,366` lines) split into `< 800`-line submodules (`D9`): `src/concurrency/pipeline/mod.rs` (`594` lines), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`723` lines), and `src/concurrency/pipeline/integration_tests.rs` (`795` lines).
   - Gate baseline ratcheted (`B8`): `lib_debug: 3151 -> 3172`, `lib_release: 3168 -> 3189`, `doctests: 229`, `compile_fail` doctests: `7 -> 8`, `all_targets: 3812 -> 3834`.

---

## 2. Commit Table (`b8e7f8b..6571c89`)

| # | Commit | Parent | Date | Subject | Files | +Ins / −Del |
|---|--------|--------|------|---------|-------|-------------|
| 1 | `b8e7f8b` | `a8d8b39` | 2026-10-06 | fix(io,memory): restrict dense object table to sequential IDs, pack depth state into u64, and tighten S23 tests (S24-R1, S24-R3, S24-R4) | 3 | +380 / −188 |
| 2 | `39814be` | `b8e7f8b` | 2026-10-06 | docs(review): fix Stage 10 io.md identifiers/citations and add scripts/audit_ledger.py (S24-R2) | 2 | +375 / −26 |
| 3 | `05d4245` | `39814be` | 2026-10-06 | fix(thread): fix x86_64 inline asm syntax, AsAtomic mutable provenance, futex wake/wait/ABI bugs, and InstanceTls drop/reentrancy [C10, C10.1, B7, D10] | 3 | +509 / −219 |
| 4 | `a8ac402` | `05d4245` | 2026-10-06 | fix(concurrency): harden work_stealing, fiber_pool, parallel_trie, and pipeline [C10, C10.1, B7, D10] | 5 | +926 / −231 |
| 5 | `d96a162` | `a8ac402` | 2026-10-06 | refactor(concurrency): split pipeline.rs into < 800-line submodules [D9] | 5 | +2382 / −2366 |
| 6 | `a0bcdff` | `d96a162` | 2026-10-06 | fix(sync): audit and upgrade weak-memory atomic orderings across cspp, lockfree_pool, and lru_map [C10.1] | 4 | +161 / −44 |
| 7 | `6571c89` | `a0bcdff` | 2026-10-07 | fix(sync): make SharedPool byte access word-atomic, pin cas_weak_u128 regs, and ratchet Stage 11 gates [C10, C10.1, B7, B8, D10] | 10 | +214 / −270 |

---

## 3. `C10` Findings & Fixes in `src/thread/` (`05d4245`, `6571c89`)

### 3.1 `src/thread/atomic_ext.rs`
- **`C10.A` (Broken AT&T syntax in Intel-syntax `std::arch::asm!` + `rbx` register-aliasing in `cas_weak_u128`)**:
  - **Before**: `cas_weak_u64_asm` (`39814be:src/thread/atomic_ext.rs:142-159`), `cas_weak_u128` (`39814be:src/thread/atomic_ext.rs:165-193`), `atomic_increment_u64` (`39814be:src/thread/atomic_ext.rs:196-212`), and `atomic_exchange_u64` (`39814be:src/thread/atomic_ext.rs:217-231`) used AT&T addressing syntax `({ptr})` and reversed operand order without `options(att_syntax)`. Furthermore, `cas_weak_u128` passed `in("rbx") desired_lo` (rejected by `rustc` because LLVM reserves `rbx`) and had a wrong `#[cfg(target_feature = "cmpxchg16b")]` gate (`39814be:src/thread/atomic_ext.rs:164`). Under `-Zsanitizer=thread` (`--release`), LLVM allocated generic `ptr = in(reg) ptr` (`05d4245:src/thread/atomic_ext.rs:193`) to `%rbx`, so `"xchg rbx, {desired_lo:r}"` overwrote `ptr` with `desired_lo` before `lock cmpxchg16b [{ptr}]`.
  - **After**: Replaced `({ptr})` and reversed operand order with Intel syntax `[{ptr}]` in `cas_weak_u64_asm` (`05d4245:src/thread/atomic_ext.rs:151-169`), `atomic_increment_u64` (`05d4245:src/thread/atomic_ext.rs:211-226`), and `atomic_exchange_u64` (`05d4245:src/thread/atomic_ext.rs:233-248`). In `cas_weak_u128` (`6571c89:src/thread/atomic_ext.rs:176-203`), replaced `#[cfg(target_feature = "cmpxchg16b")]` with `#[target_feature(enable = "cmpxchg16b")]` and pinned `ptr` to `in("rdi")`, `desired_lo` to `inout("rsi")`, and `result_u64` to `inout("rax")` so LLVM can never allocate `ptr` or `desired_lo` to `rbx` across `xchg rbx, rsi`.
  - **Regression test**: `test_c10_a_x86_64_asm_exchange_increment_and_cas_modify_memory` (`6571c89:src/thread/atomic_ext.rs:619-652`).
- **`C10.B` (`atomic_maximize`, `atomic_minimize`, and `update_if` panicking on `Ordering::Release` / `Ordering::AcqRel`)**:
  - **Before**: `atomic_maximize` (`39814be:src/thread/atomic_ext.rs:41-54`), `atomic_minimize` (`39814be:src/thread/atomic_ext.rs:57-70`), and `update_if` (`39814be:src/thread/atomic_ext.rs:101-114`) passed `order` directly to `self.load(order)` and as the `compare_exchange_weak` failure ordering, panicking at runtime whenever `order` was `Ordering::Release` or `Ordering::AcqRel` (`"there is no such thing as a release load"`).
  - **After**: Added `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) mapping `Release -> Relaxed` and `AcqRel -> Acquire`, and used `fail_order = load_ordering(order)` in `atomic_maximize` (`05d4245:src/thread/atomic_ext.rs:50-64`), `atomic_minimize` (`05d4245:src/thread/atomic_ext.rs:67-81`), and `update_if` (`05d4245:src/thread/atomic_ext.rs:112-125`).
  - **Regression test**: `test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel` (`6571c89:src/thread/atomic_ext.rs:655-665`).
- **`C10.C` (`AsAtomic::as_atomic(&self)` unsound shared-reference-to-atomic cast — Miri UB)**:
  - **Before**: `AsAtomic::as_atomic(&self) -> &Self::Atomic` (`39814be:src/thread/atomic_ext.rs:309-363`) transmuted `&T` (shared read-only reference without `UnsafeCell`) into `&Atomic*`, which is immediate Undefined Behavior under Rust's aliasing model (`SharedReadOnly` provenance cannot be used for atomic writes/RMWs) and created data races with concurrent `&T` readers.
  - **After**: Changed `AsAtomic::as_atomic(&mut self) -> &Self::Atomic` (`05d4245:src/thread/atomic_ext.rs:340-396`) across all integer and pointer implementations (matching `Atomic*::from_mut`), and added a `compile_fail` doctest on `AsAtomic` (`05d4245:src/thread/atomic_ext.rs:332-339`) proving shared `&u64` cannot call `as_atomic()`.
  - **Regression test**: `test_c10_c_as_atomic_from_mut_is_miri_clean` (`6571c89:src/thread/atomic_ext.rs:668-679`).

### 3.2 `src/thread/linux_futex.rs`
- **`C10.D` (`FutexCondvar::notify_all` and `FutexRwLock::write` unlock waking only 1 waiter due to `u32::MAX as i32 == -1`)**:
  - **Before**: `futex_wake` (`39814be:src/thread/linux_futex.rs:94-103`) passed `num_wake` directly to `sys::futex` (`39814be:src/thread/linux_futex.rs:25-47`), which re-cast `val: u32` as `i32` (`u32::MAX -> -1`), and Linux `FUTEX_WAKE_PRIVATE` clamps negative `val` to `1` (`max(val, 1)`), causing `FutexCondvar::notify_all` (`39814be:src/thread/linux_futex.rs:285-287`) and `FutexRwLock` reader-wake (`39814be:src/thread/linux_futex.rs:413`) to wake at most 1 waiter.
  - **After**: Clamped `num_wake.min(i32::MAX as u32)` in `futex_wake` (`05d4245:src/thread/linux_futex.rs:107-116`) and passed `i32::MAX as u32` in `FutexCondvar::notify_all` (`05d4245:src/thread/linux_futex.rs:313-315`) and `FutexRwLock` drop (`05d4245:src/thread/linux_futex.rs:445`).
  - **Regression test**: `test_c10_d_notify_all_wakes_all_waiters` (`05d4245:src/thread/linux_futex.rs:590-626`).
- **`C10.E` (`futex_wait` `tv_sec` `as i64` wraparound on `Duration::MAX` + `sys::futex` variadic C ABI mismatch under Miri + `FutexCondvar` / `FutexMutex` / `FutexRwLock` ordering & poison restoration)**:
  - **Before**:
    1. `futex_wait` (`39814be:src/thread/linux_futex.rs:54-63`) cast `dur.as_secs() as i64`, wrapping `u64::MAX` to `-1` and failing with `EINVAL` (`errno 22`).
    2. `sys::futex` (`39814be:src/thread/linux_futex.rs:25-47`) declared a custom `unsafe extern "C" fn syscall(num: c_long, ...) -> c_long` and passed `usize` arguments, which Miri rejected as UB (`scalar size mismatch: expected 4 bytes but got 8 bytes`) because `libc::syscall` uses `c_long` variadic arguments.
    3. `FutexMutex::lock` (`39814be:src/thread/linux_futex.rs:145-182`) used `Ordering::Relaxed` on the lock handoff CAS and spin-load; `FutexCondvar::wait` (`39814be:src/thread/linux_futex.rs:241-265`) used `Ordering::Relaxed` for `sequence` and lost the mutex guard if re-locking failed; `FutexRwLock::read`/`write` (`39814be:src/thread/linux_futex.rs:325-391`) used `Ordering::Relaxed` on lock-state CAS loops.
  - **After**:
    1. `futex_wait` (`05d4245:src/thread/linux_futex.rs:59-71`) caps `dur.as_secs().min(libc::time_t::MAX as u64) as libc::time_t`.
    2. `sys::futex` (`05d4245:src/thread/linux_futex.rs:30-50`) calls `libc::syscall(libc::SYS_futex, uaddr, op, val, timeout, uaddr2, val3)` directly with `c_long`-compatible types.
    3. Upgraded `FutexMutex::lock`/`try_lock` (`05d4245:src/thread/linux_futex.rs:158-221`), `FutexCondvar` (`05d4245:src/thread/linux_futex.rs:265-315`), and `FutexRwLock::read`/`write` (`05d4245:src/thread/linux_futex.rs:355-425`) to `Acquire`/`Release`/`AcqRel` with `// ORDERING:` comments on all remaining `Relaxed` spin/counter loads.
  - **Regression test**: `test_c10_e_futex_wait_duration_max_does_not_einval` (`05d4245:src/thread/linux_futex.rs:629-638`).

### 3.3 `src/thread/instance_tls.rs`
- **`C10.F1` (`CleanupHandle::drop` leaving current thread's value alive until ID recycling + `RefCell` borrow panic on re-entrant `T::drop`)**:
  - **Before**: `CleanupHandle::drop` (`39814be:src/thread/instance_tls.rs:416-431`) only recycled the ID into `TLS_ID_POOL` without clearing the dropping thread's `TlsMatrix` cell, leaking `T` on the current thread until another `InstanceTls<T>` reused that slot; furthermore, `InstanceTls::set` (`39814be:src/thread/instance_tls.rs:178-194`) and `InstanceTls::remove` (`39814be:src/thread/instance_tls.rs:197-209`) dropped the old `Option<T>` *inside* `TLS_MATRICES.with(|matrices| { let mut matrices = matrices.borrow_mut(); ... })`, so any `T::drop` touching `InstanceTls` panicked with `RefCell already borrowed`.
  - **After**: `TlsMatrix::set` (`6571c89:src/thread/instance_tls.rs:410-422`) and `TlsMatrix::remove` (`6571c89:src/thread/instance_tls.rs:425-441`) return the evicted values to be dropped *after* the `TLS_MATRICES` `RefCell` borrow ends (`6571c89:src/thread/instance_tls.rs:197-230`), and `CleanupHandle::drop` (`6571c89:src/thread/instance_tls.rs:446-479`) eagerly removes and drops the current thread's cell via `TLS_MATRICES.try_with` + `try_borrow_mut` before returning the `(id, generation)` pair to `TLS_ID_POOL`.
  - **Regression test**: `test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic` (`6571c89:src/thread/instance_tls.rs:859-892`).
- **`C10.F2` (Zero-dimension `ROWS == 0` / `COLS == 0` / `POOL_SIZE == 0` and `ROWS.checked_mul(COLS)` overflow panics + production `.expect()` removal)**:
  - **Before**: `InstanceTls::allocate_id` (`39814be:src/thread/instance_tls.rs:224-260`) computed `ROWS * COLS` and `% COLS` without checking for zero or overflow, `TlsPool::new` (`39814be:src/thread/instance_tls.rs:536-548`) allowed `POOL_SIZE == 0` (panicking with `% 0` in `get_next`: `39814be:src/thread/instance_tls.rs:552`), and `OwnerTls` / `InstanceTls` used 8 production `.expect()` calls.
  - **After**: Added `InstanceTls::validate_dimensions` (`6571c89:src/thread/instance_tls.rs:95-110`) using `ROWS.checked_mul(COLS)`, rejecting `ROWS == 0`, `COLS == 0`, or overflow in `InstanceTls::new` (`6571c89:src/thread/instance_tls.rs:113-130`); rejected `POOL_SIZE == 0` in `TlsPool::new` (`6571c89:src/thread/instance_tls.rs:573-589`); and replaced all 8 production `.expect()` calls in `src/thread/instance_tls.rs` (`D10`).
  - **Regression test**: `test_c10_f2_zero_pool_and_overflow_dimensions_return_err` (`6571c89:src/thread/instance_tls.rs:895-903`).

---

## 4. `C10` Findings & Fixes in `src/concurrency/` (`a8ac402`, `d96a162`, `6571c89`)

### 4.1 `src/concurrency/work_stealing.rs`
- **`C10.WS1` (`WorkStealingQueue::pop_local` stranded tasks moved into `steal_queue` by `balance()`)**:
  - **Before**: `balance()` (`05d4245:src/concurrency/work_stealing.rs:190-217`) moved half of `local_queue` into `steal_queue`, but `pop_local()` (`05d4245:src/concurrency/work_stealing.rs:148-154`) only popped from `local_queue` and `find_task()` (`05d4245:src/concurrency/work_stealing.rs:456-481`) skipped `worker_id` when stealing. On a 1-worker executor (or when all other workers were idle), tasks moved to `steal_queue` were stranded forever.
  - **After**: `pop_local()` (`a8ac402:src/concurrency/work_stealing.rs:154-186`) falls back to popping from the back of `steal_queue` when `local_queue` is empty.
  - **Regression test**: `test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor` (`a8ac402:src/concurrency/work_stealing.rs:870-888`).
- **`C10.WS2` (`WorkStealingExecutor::shutdown` aborted workers immediately and accepted post-shutdown submissions)**:
  - **Before**: `shutdown()` (`05d4245:src/concurrency/work_stealing.rs:406-417`) called `handle.abort()` on every worker, discarding queued tasks and cancelling in-flight tasks mid-execution, while `submit()` (`05d4245:src/concurrency/work_stealing.rs:321-354`) never checked `self.shutdown`.
  - **After**: Added `Notify` wakeups and shutdown checks in `submit()` (`a8ac402:src/concurrency/work_stealing.rs:407-446`), drained all local, steal, and global queues in `worker_loop` (`a8ac402:src/concurrency/work_stealing.rs:523-568`) before exiting, and awaited clean worker completion in `shutdown()` (`a8ac402:src/concurrency/work_stealing.rs:504-520`).
  - **Regression test**: `test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit` (`a8ac402:src/concurrency/work_stealing.rs:891-921`).
- **`C10.WS3` (Panicking task leaked `active_tasks` and killed the worker thread)**:
  - **Before**: `worker_loop` (`05d4245:src/concurrency/work_stealing.rs:429-435`) incremented `active_tasks` before `task.execute().await` and decremented it after; a panicking task unwound the worker loop, permanently leaking `active_tasks` and killing the worker.
  - **After**: Added `ActiveTaskGuard` (`a8ac402:src/concurrency/work_stealing.rs:523-560`) and `CatchUnwindFuture` around `task.execute()` so panicking tasks still decrement `active_tasks` and increment `total_completed` without killing the worker.
  - **Regression test**: `test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker` (`a8ac402:src/concurrency/work_stealing.rs:924-957`).
- **`C10.WS5` (`WorkStealingQueue::steal` failed to steal when `local_queue.len() == 1`)**:
  - **Before**: `steal()` (`05d4245:src/concurrency/work_stealing.rs:157-187`) computed `let steal_count = local.len() / 2;` (`0` when `local.len() == 1`), returning `None` even when a stealable task was waiting in `local_queue`.
  - **After**: `steal()` (`a8ac402:src/concurrency/work_stealing.rs:189-248`) uses `(local.len() / 2).max(1)` so a single queued stealable task can be stolen by an idle worker.
  - **Regression test**: `test_c10_ws5_steal_takes_single_queued_stealable_task` (`a8ac402:src/concurrency/work_stealing.rs:960-974`).
- **`C10.WS6` (`WorkStealingExecutor::spawn` discarded `Fiber::id()` / priority and `new` accepted `queue_capacity == 0`)**:
  - **Before**: `spawn()` (`05d4245:src/concurrency/work_stealing.rs:357-363`) ignored `fiber.id()` and wrapped the fiber in `ClosureTask::new` (allocating a new disconnected ID and resetting priority to `Normal`), and `WorkStealingExecutor::new` (`05d4245:src/concurrency/work_stealing.rs:283-318`) accepted `queue_capacity == 0`.
  - **After**: Rejected `queue_capacity == 0` in `WorkStealingExecutor::new` (`a8ac402:src/concurrency/work_stealing.rs:366-404`) and preserved `fiber.id()` and mapped `fiber.priority()` in `spawn()` (`a8ac402:src/concurrency/work_stealing.rs:449-466`).
  - **Regression test**: `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected` (`a8ac402:src/concurrency/work_stealing.rs:977-993`).

### 4.2 `src/concurrency/fiber_pool.rs`
- **`C10.FP1` (`FiberPool::shutdown` returned while spawned/queued fibers were still running)**:
  - **Before**: `FiberPool::shutdown` (`05d4245:src/concurrency/fiber_pool.rs:282-292`) set `self.shutdown = true` and awaited `worker_handles` (which was always empty because `FiberPool::spawn` spawns onto Tokio directly), returning before active or queued fibers finished.
  - **After**: Tracked `in_flight_fibers` and `idle_notify` in `FiberPool` (`6571c89:src/concurrency/fiber_pool.rs:116-146`), decremented via `ActiveFiberGuard::drop` (`6571c89:src/concurrency/fiber_pool.rs:241-277`), rejected post-shutdown `spawn()` (`6571c89:src/concurrency/fiber_pool.rs:204-216`), and awaited `in_flight_fibers == 0` in `shutdown()` (`6571c89:src/concurrency/fiber_pool.rs:328-338`).
  - **Regression test**: `test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers` (`6571c89:src/concurrency/fiber_pool.rs:561-592`).
- **`C10.FP2` (`FiberPool::new` accepted `max_fibers == 0` / `max_workers == 0`, causing `NaN` in `load_factor` / `utilization`)**:
  - **Before**: `FiberPool::new` (`05d4245:src/concurrency/fiber_pool.rs:148-169`) did not validate `config`, so `max_fibers == 0` or `max_workers == 0` caused `0.0 / 0.0 == NaN` in `load_factor()` (`05d4245:src/concurrency/fiber_pool.rs:294-298`) and `stats()` (`05d4245:src/concurrency/fiber_pool.rs:250-265`).
  - **After**: Validated `max_fibers > 0`, `max_workers > 0`, `initial_workers <= max_workers`, and `fiber_stack_size > 0` in `FiberPool::new` (`6571c89:src/concurrency/fiber_pool.rs:149-181`) and guarded division by zero in `stats()` (`6571c89:src/concurrency/fiber_pool.rs:289-310`) and `load_factor()` (`6571c89:src/concurrency/fiber_pool.rs:341-345`).
  - **Regression test**: `test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan` (`6571c89:src/concurrency/fiber_pool.rs:595-622`).

### 4.3 `src/concurrency/parallel_trie.rs`
- **`C10.PT1` (`chunks(0)` / `% 0` panics when `chunk_size == 0` or `max_workers == 0`)**:
  - **Before**: `ParallelTrieBuilder::build_louds_trie` (`05d4245:src/concurrency/parallel_trie.rs:40-75`), `ParallelLoudsTrie::parallel_contains` (`05d4245:src/concurrency/parallel_trie.rs:202-227`), `ParallelLoudsTrie::parallel_prefix_search` (`05d4245:src/concurrency/parallel_trie.rs:230-258`), and `ParallelLoudsTrie::bulk_insert` (`05d4245:src/concurrency/parallel_trie.rs:299-321`) called `.chunks(self.chunk_size)` directly (panicking when `chunk_size == 0`), while `select_replica` (`05d4245:src/concurrency/parallel_trie.rs:265-269`) computed `% self.read_replicas.len()` (panicking when `read_replicas` was empty).
  - **After**: Validated non-zero `chunk_size` and `max_workers` in `ParallelTrieBuilder::build_louds_trie` (`a8ac402:src/concurrency/parallel_trie.rs:40-54`) and `ParallelLoudsTrie::bulk_insert` (`a8ac402:src/concurrency/parallel_trie.rs:328-349`), clamped chunk sizes to `.max(1)` in `parallel_contains` (`a8ac402:src/concurrency/parallel_trie.rs:235-262`) and `parallel_prefix_search` (`a8ac402:src/concurrency/parallel_trie.rs:265-293`), and used `checked_rem` in `select_replica` (`a8ac402:src/concurrency/parallel_trie.rs:301-305`).
  - **Regression test**: `test_c10_pt1_zero_chunk_size_or_workers_returns_err_instead_of_panic` (`a8ac402:src/concurrency/parallel_trie.rs:719-737`).
- **`C10.PT3` (`parallel_process` and `merge_tries` used `.expect()` on `JoinHandle`, panicking on user closure panic)**:
  - **Before**: `parallel_process` (`05d4245:src/concurrency/parallel_trie.rs:324-348`) and `merge_tries` (`05d4245:src/concurrency/parallel_trie.rs:439-468`) called `.expect(...)` on `AcquireError` and `JoinError`, violating `D10` and crashing the caller when a closure panicked.
  - **After**: Converted `AcquireError` and `JoinError` into `Err(ZiporaError::threading_error(...))` in `parallel_process` (`a8ac402:src/concurrency/parallel_trie.rs:352-391`) and `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:481-511`), removing all 5 `.expect()` calls from `src/concurrency/` (`D10`).
  - **Regression test**: `test_c10_pt3_parallel_process_catches_closure_panic_as_err` (`a8ac402:src/concurrency/parallel_trie.rs:740-758`).

### 4.4 `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`)
- **`C10.P1` (Multi-stage `Pipeline::execute_stream` always failed with `"pipeline: channel missing for prior stage"` + dropped stage errors)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:344-478`) stored an `inter_rx` inside `channels` on iteration `i == 0` and tried `channels.get_mut(i - 1)?.0.take()` on iteration `i == 1`, where `channels[0].0` had already been taken by stage 0, so every `stages.len() >= 2` pipeline immediately returned `Err(InvalidState("pipeline: channel missing for prior stage"))`; additionally, stage processing errors were silently ignored with `if let Ok(output) = ...`.
  - **After**: Rewrote `execute_stream` (`6571c89:src/concurrency/pipeline/mod.rs:262-422`) to thread `current_rx` directly from stage `i` to stage `i + 1`, flush partial batches at end-of-stream (`6571c89:src/concurrency/pipeline/mod.rs:337-362`), and propagate stage errors/timeouts across both single-stage and multi-stage pipelines.
  - **Regression test**: `test_c10_p1_execute_stream_multi_stage_and_error_propagation` (`6571c89:src/concurrency/pipeline/integration_tests.rs:679-725`).
- **`C10.P2` (`items_in_flight` leaked on stage error, timeout, or cancellation in `execute_single` and `process_batch`)**:
  - **Before**: `execute_single` (`05d4245:src/concurrency/pipeline.rs:302-341`) and `process_batch` (`05d4245:src/concurrency/pipeline.rs:481-526`) incremented `items_in_flight` before awaiting the stage future and only decremented on the `Ok(Ok(_))` arm, permanently leaking `items_in_flight` on stage error, timeout, or future cancellation.
  - **After**: Added `AtomicCountGuard` (`6571c89:src/concurrency/pipeline/mod.rs:183-202`) so `items_in_flight` is unconditionally decremented on scope exit in `execute_single`, `execute_stream`, and `process_batch`.
  - **Regression test**: `test_c10_p2_items_in_flight_decremented_on_timeout_and_error` (`6571c89:src/concurrency/pipeline/integration_tests.rs:728-747`).
- **`C10.P3` (`buffer_size == 0` panicked inside `mpsc::channel(0)`, `max_in_flight == 0` produced `NaN` utilization, and `start_timeout_checker` busy-spun on `Duration::ZERO`)**:
  - **Before**: `mpsc::channel(self.config.buffer_size)` (`05d4245:src/concurrency/pipeline.rs:373`) panicked when `buffer_size == 0`; `stats()` (`05d4245:src/concurrency/pipeline.rs:536`) divided `in_flight / max_in_flight` (`0.0 / 0.0 == NaN`); and `start_timeout_checker` (`05d4245:src/concurrency/pipeline.rs:240-262`) computed `self.batch_timeout / 4` (`Duration::ZERO` when `batch_timeout < 4ns`), spinning without yielding.
  - **After**: Clamped channel capacities to `.max(1)` (`6571c89:src/concurrency/pipeline/mod.rs:277,309`), guarded `max_in_flight == 0` in `stats()` (`6571c89:src/concurrency/pipeline/mod.rs:469-481`), and floored the checker sleep interval to `Duration::from_millis(1)` in `start_timeout_checker` (`6571c89:src/concurrency/pipeline/stages.rs:96-120`).
  - **Regression test**: `test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan` (`6571c89:src/concurrency/pipeline/integration_tests.rs:750-780`).
- **`C10.P5` (`BatchCollector` expired the first item after an idle period immediately)**:
  - **Before**: `BatchCollector` initialized `last_flush` to `Instant::now()` at construction (`05d4245:src/concurrency/pipeline.rs:177`) and after flushes, so when a collector sat idle for `> batch_timeout`, the very first item added had `last_flush.elapsed() >= batch_timeout` and was immediately flushed as a 1-item timeout batch on the next `check_timeout()`.
  - **After**: Reset `last_flush = Instant::now()` when transitioning `batch` from empty to non-empty in `BatchCollector::add` (`6571c89:src/concurrency/pipeline/stages.rs:36-55`).
  - **Regression test**: `test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout` (`6571c89:src/concurrency/pipeline/integration_tests.rs:783-794`).
- **`D1` & `D9` (Honesty markers removed & `pipeline.rs` split into `< 800`-line submodules)**:
  - Replaced the 3 honesty-marker comments in `BatchMapStage::process_batch` (`6571c89:src/concurrency/pipeline/stages.rs:184-206`) and split `src/concurrency/pipeline.rs` (`2,366` lines at `a8ac402`) into `src/concurrency/pipeline/mod.rs` (`594` lines), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`723` lines), and `src/concurrency/pipeline/integration_tests.rs` (`795` lines) (`d96a162`, `6571c89`).

---

## 5. `C10.1` Crate-Wide Weak-Memory Ordering Audit (`05d4245`, `a8ac402`, `a0bcdff`, `6571c89`)

| Module / File | Atomic Ordering / Synchronization Upgrades & `// ORDERING:` Coverage |
|---------------|----------------------------------------------------------------------|
| `src/thread/atomic_ext.rs` | Fixed RMW initial-load/failure orderings via `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) and added `// ORDERING:` comments to `AtomicBitOps` (`05d4245:src/thread/atomic_ext.rs:426-468`). |
| `src/thread/linux_futex.rs` | Upgraded `FutexMutex`, `FutexCondvar`, and `FutexRwLock` state transitions from `Relaxed` to `Acquire`/`Release` (`05d4245:src/thread/linux_futex.rs:158-448`) and documented all remaining `Relaxed` spin/waiter-count loads with `// ORDERING:`. |
| `src/thread/instance_tls.rs` | Documented `next_id` monotonic `Relaxed` `fetch_add` with `// ORDERING:` (`6571c89:src/thread/instance_tls.rs:39`). |
| `src/concurrency/mod.rs`, `work_stealing.rs`, `fiber_pool.rs`, `parallel_trie.rs`, `pipeline/mod.rs` | Upgraded lifecycle flags (`WorkStealingExecutor::shutdown` `a8ac402:src/concurrency/work_stealing.rs:408,505,541` and `FiberPool::shutdown` / `in_flight_fibers` `6571c89:src/concurrency/fiber_pool.rs:206,247,329,334`) to `Acquire`/`Release`/`AcqRel`, and added `// ORDERING:` comments to all remaining `Relaxed` ID generators and telemetry counters. |
| `src/fsa/cspp_trie_concurrent.rs` | Upgraded `LockFreeFreelist::push` CAS from `(Release, Relaxed)` to `(AcqRel, Acquire)` (`6571c89:src/fsa/cspp_trie_concurrent.rs:311-319`), upgraded `alloc_val_pos` / `free_val_pos` freelist loads/stores from `Relaxed` to `Acquire`/`Release` (`6571c89:src/fsa/cspp_trie_concurrent.rs:900-935`), replaced all non-atomic `raw_ptr` / `get_slice` / `copy_nonoverlapping` byte accesses on `SharedPool::data: Box<[AtomicU32]>` with word-atomic `SharedPool::read_bytes_acquire` (`6571c89:src/fsa/cspp_trie_concurrent.rs:237-249`) and `SharedPool::write_bytes_relaxed` (`6571c89:src/fsa/cspp_trie_concurrent.rs:254-268`), and added `// ORDERING:` comments across the file. Eliminated 18 `unsafe` blocks (`117 -> 99` in `src/fsa/`) and made `make tsan_cspp` 15/15 100% ThreadSanitizer-clean. |
| `src/memory/lockfree_pool.rs` | Upgraded `deallocate_to_fast_bin` CAS from `(Release, Relaxed)` to `(AcqRel, Relaxed)` (`a0bcdff:src/memory/lockfree_pool.rs:720-726`), upgraded `store_block_width` / `load_block_width` from `Relaxed` to `Release` / `Acquire` (`a0bcdff:src/memory/lockfree_pool.rs:896-915`), and added `// ORDERING:` comments to all `Relaxed` bump-cursor and statistics operations. |
| `src/containers/specialized/lru_map.rs` & `concurrent_lru_map.rs` | Moved `stats.entry_count.store(...)` inside the exclusive `nodes` write lock in `LruMap::insert` (`a0bcdff:src/containers/specialized/lru_map.rs:718-723`) and `LruMap::evict_lru` (`a0bcdff:src/containers/specialized/lru_map.rs:919-924`) so concurrent operations cannot overwrite `entry_count` with a stale post-unlock read, and added `// ORDERING:` comments to all `Relaxed` list-node and statistics operations. |

---

## 6. Verbatim RED / Miri / TSan / Gate Outputs

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
- **Miri RED on `sys::futex` (`39814be:src/thread/linux_futex.rs:39-47`)**:
```text
test thread::linux_futex::tests::test_c10_d_notify_all_wakes_all_waiters ... error: Undefined Behavior: scalar size mismatch: expected 4 bytes but got 8 bytes instead
   --> src/thread/linux_futex.rs:41:13
    |
 41 | /             syscall(
 42 | |                 SYS_futex,
 43 | |                 uaddr as usize,
 44 | |                 op as usize,
...   |
 48 | |                 val3 as usize,
 49 | |             ) as c_int
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

### 6.3 Miri & ThreadSanitizer Verification (`6571c89`)
- **`make miri_thread` (SIMD / haswell & `MIRI_SCALAR=1`)**:
```text
running 25 tests
test thread::atomic_ext::tests::test_as_atomic ... ok
test thread::atomic_ext::tests::test_atomic_bit_ops ... ok
test thread::atomic_ext::tests::test_atomic_maximize ... ok
test thread::atomic_ext::tests::test_atomic_minimize ... ok
test thread::atomic_ext::tests::test_c10_a_x86_64_asm_exchange_increment_and_cas_modify_memory ... ignored, inline x86_64 assembly is not supported by Miri
test thread::atomic_ext::tests::test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel ... ok
test thread::atomic_ext::tests::test_c10_c_as_atomic_from_mut_is_miri_clean ... ok
test thread::atomic_ext::tests::test_update_if ... ok
test thread::instance_tls::tests::test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic ... ok
test thread::instance_tls::tests::test_c10_f2_zero_pool_and_overflow_dimensions_return_err ... ok
test thread::instance_tls::tests::test_instance_ids_are_recycled_after_drop ... ok
test thread::instance_tls::tests::test_instance_tls_basic ... ok
test thread::instance_tls::tests::test_instance_tls_multiple_threads ... ok
test thread::instance_tls::tests::test_instance_tls_set_remove ... ok
test thread::instance_tls::tests::test_matrix_dimensions ... ok
test thread::instance_tls::tests::test_owner_tls ... ok
test thread::instance_tls::tests::test_recycled_id_does_not_expose_the_previous_instances_value ... ok
test thread::instance_tls::tests::test_tls_id_management ... ok
test thread::instance_tls::tests::test_tls_pool ... ok
test thread::linux_futex::tests::test_c10_d_notify_all_wakes_all_waiters ... ok
test thread::linux_futex::tests::test_c10_e_futex_wait_duration_max_does_not_einval ... ok
test thread::linux_futex::tests::test_futex_condvar ... ok
test thread::linux_futex::tests::test_futex_mutex_basic ... ok
test thread::linux_futex::tests::test_futex_mutex_contention ... ok
test thread::linux_futex::tests::test_futex_rwlock ... ok

test result: ok. 24 passed; 0 failed; 1 ignored; 0 measured; 3118 filtered out; finished in 20.12s
```
- **`make miri_cspp`**:
```text
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 3128 filtered out; finished in 58.25s
```
- **`make tsan_cspp`, `make tsan_pool`, `make tsan_concurrency`**:
```text
# make tsan_cspp
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 3152 filtered out; finished in 7.12s

# make tsan_pool
test result: ok. 33 passed; 0 failed; 0 ignored; 0 measured; 3134 filtered out; finished in 0.74s

# make tsan_concurrency (thread:: + concurrency::)
test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 3142 filtered out; finished in 0.10s
test result: ok. 80 passed; 0 failed; 0 ignored; 0 measured; 3087 filtered out; finished in 1.04s
```
