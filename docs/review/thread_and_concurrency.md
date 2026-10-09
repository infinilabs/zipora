# Stage 11 Review Ledger: `src/thread/` + `src/concurrency/` + `C10.1` Weak-Memory Ordering Audit + `S25` / `S26` / `S27` / `S28` Remediation

## 1. Scope & Summary

Stage 11 covers:
1. **`S24-R1..R4`, `S25-R8`, `S26-R4`, and `S27-R5`** carry-over fixes from the Stage 10 re-review (`b8e7f8b..39814be`, `f7e39ad`, `1141a79`, and `4929ccc`, documented in `docs/review/io.md` §12).
2. **`C10` (`src/thread/` + `src/concurrency/`)**: correctness, safety, panic-resilience, and resource-lifecycle audit across `src/thread/atomic_ext.rs`, `src/thread/linux_futex.rs`, `src/thread/instance_tls.rs`, `src/concurrency/mod.rs`, `src/concurrency/work_stealing.rs`, `src/concurrency/fiber_pool.rs`, `src/concurrency/parallel_trie.rs`, and `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`).
3. **`C10.1` (Crate-Wide Weak-Memory Ordering Audit)**: audit of all `Ordering::Relaxed` / `Release` / `Acquire` / `AcqRel` / `SeqCst` atomic operations across `src/thread/`, `src/concurrency/`, `src/fsa/cspp_trie_concurrent.rs`, `src/memory/lockfree_pool.rs`, `src/containers/specialized/lru_map.rs`, and `src/containers/specialized/concurrent_lru_map.rs`, adding `// ORDERING:` justifications to every `Ordering::Relaxed` site in the audited modules, upgrading weak orderings on shared-data publication/reclamation paths, and replacing all non-atomic byte reads/writes on `SharedPool` in `ConcurrentCsppTrie` with word-atomic helpers.
4. **`S25-R1..R9` Stage 11 Re-Review Round 1 Remediation (`f7e39ad..48438fe`, §7)**:
   - `S25-R1` (`0932ce7`): Eliminated the 256-byte `read_zpath` over-read in `ConcurrentCsppTrie::lookup`/`contains`/`lookup_with_guard` via zero-copy word-wise `SharedPool::bytes_eq_relaxed`, cached `raw: u32` header word in `ConcurrentNodeView`, peeled root fast-node step, and inlined leaf `valpos`.
   - `S25-R2` (`48438fe`): Fixed `WorkStealingExecutor::spawn` to spawn directly onto the current Tokio runtime (`tokio::spawn`) so it never hangs when a global executor's creation runtime has been dropped or when a fiber awaits a child fiber on a 1-worker executor.
   - `S25-R3` (`ee8b29a`): Updated `docs/review/thread_and_concurrency.md`, `docs/review/io.md`, and `scripts/audit_ledger.py`.
   - `S25-R4` (`0932ce7`, `b74f6fe`, `48438fe`): Restored `ConcurrentCsppTrie` lookup throughput, sped up `InstanceTls::get` ~9x on `u64`, added lock-free `steal_len` fast path on `WorkStealingQueue` owner push+pop and `try_recv()` batching on `Pipeline::execute_stream`, and relaxed `LockFreeFreelist::push` / `deallocate_to_fast_bin` / `store_block_width` / `load_block_width` orderings back to `a8d8b39`'s set.
   - `S25-R5` (`b74f6fe`): Made `InstanceTls` / `CleanupHandle` `drop` / `TlsMatrix` thread-exit-safe (`TLS_MATRICES.try_with`, heap-backed lazy row table `Vec<Option<*mut [TlsCell<T>]>>` supporting `ROWS = 4_000_000`, and compile-time size/alignment assertions in `AsAtomic`).
   - `S25-R6` (`48438fe`): Fixed `WorkStealingExecutor::is_idle()` to scan queues (`self.total_queued() == 0`) before loading `active_tasks`, synchronized `submit()` vs `shutdown()` under queue locks, checked `stage_timeout` after synchronous pipeline stages (`execute_single`, `execute_stream`, `process_batch`), and isolated per-call `local_stage_stats` in `execute_stream`.
   - `S25-R7` & `S25-R9` (`0932ce7`, `b74f6fe`, `48438fe`): Unified `allocate_id` and `CleanupHandle` `drop` on `global_registry`, asserted `expected_id.as_u64()` in `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected`, computed fallback results on `spawn_blocking` failure in `ParallelLoudsTrie::parallel_contains`/`parallel_prefix_search`, and added multi-threaded `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` tests with non-atomic payload writes for `WorkStealingExecutor`, `FiberPool`, `Pipeline`, and `ParallelLoudsTrie`.
5. **`S26-R1..R5` Stage 11 Re-Review Round 2 Remediation (`1141a79..2050294`, §8)**:
   - `S26-R4` (`1141a79`): Removed the 15 `#[inline]` attributes added by `f7e39ad` on `SerializableType::serialize` methods in `src/io/smart_ptr.rs`, restoring `Vec<Vec<u32>>::serialize` from `100.9 ns/inner Vec` to `73.79 ns/inner Vec` (`docs/review/io.md` §12.3).
   - `S26-R2` (`c87db71`): Fixed `InstanceTls::get` and `InstanceTls::get_value` data loss on re-entrant same-instance access or panicking `T::clone()` by cloning in place through `&T` while holding a `CellReadGuard` (`readers > 0`) on `TlsCell<T>` and deferring same-cell mutations (`set` and `remove`, where `remove()` returns `None` immediately while `readers > 0`) via `TlsMatrix::defer_write` and `TlsMatrix::apply_deferred` until the outermost `clone()` completes.
   - `S26-R3` (`c87db71`, `c64babc`): Rewrote `test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak` to hold `65,536` live `InstanceTls` instances simultaneously (filling all `256 x 256` slots and asserting the `65,537`th `new()` returns `Err`) before dropping and recycling `4,464` more; updated `test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` and `test_s25_r1_lookup_multi_insert_near_pool_end` to use `valsize = 0`; updated `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` to call `WorkStealingExecutor::init(...)` on `rt1` before dropping `rt1`; and updated part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` to use `WorkStealingExecutor::new(4, 8)` with OS-thread submitters.
   - `S26-R5` (`c64babc`): Added `work_notify: Arc<tokio::sync::Notify>` to `WorkStealingExecutor` so idle workers park on `work_notify.notified()` instead of polling with `sleep(1ms)`, added `ensure_workers_running` so `WorkStealingExecutor::global().submit()` respawns workers onto the current Tokio runtime if the `init()` runtime was dropped, deduplicated the two `bytes_eq_relaxed` calls per node in `ConcurrentCsppTrie::lookup_inner`, and narrowed `PipelineConfig::stage_timeout` / `Pipeline::execute_single` docs to state that non-yielding synchronous stages are checked post-hoc upon stage return.
   - `S26-R1` (`2050294`): Removed all whitelisted doc tokens from `scripts/audit_ledger.py`, excluded `scripts/audit_ledger.py` and `scripts/tests/` from the repository index, and committed the Round 1–2 adversarial fixture (`scripts/tests/audit_ledger_adversarial.md`).
6. **`S27-R1..R5` Stage 11 Re-Review Round 3 Remediation (`4929ccc..6e7fee4`, §9)**:
   - `S27-R2` (`4929ccc`): Constructed `ActiveWorkerGuard` before `tokio::spawn` in `WorkStealingExecutor::new` and before `rt_handle.spawn` in `WorkStealingExecutor::ensure_workers_running`, wrapped `WorkerThread::handle` in `Arc<Mutex<Option<JoinHandle<()>>>>`, re-checked `active_workers` in `notify_if_parked`, and expanded `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` to cover unpolled `new_current_thread()` `init()`, plain-thread `submit_closure` rejection with 0 live workers, and immediate `multi_thread` runtime drops.
   - `S27-R3` (`4929ccc`): Added `parked_workers: AtomicUsize` to `ExecutorStatsInner` and `ParkedWorkerGuard` in `worker_loop`, gating `work_notify.notify_waiters()` in `notify_if_parked()` on `parked_workers.load(Ordering::SeqCst) > 0` (`313.8 ns/task` on 8-thread submit+drain measured at `f37cf61` via `bench_s29_r1_executor_timings`).
   - `S27-R4` (`4929ccc`): Rewrote part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` with `WorkStealingExecutor::new(2, 4)` and 8 synchronized OS-thread submitters without `tokio::time::sleep`, failing in ~60% of runs (12/20) when `try_push_local_unless_shutdown`'s shutdown check is reverted.
   - `S27-R5` (`4929ccc`): Documented deferred `set`/`remove` visibility during `T::clone()` on `InstanceTls`, checked `(*cell_ptr).readers.get() == 0` in `InstanceTls::get` before forming `&mut *(*cell_ptr).value.get()`, removed unreachable `defer_write` in `CleanupHandle` `drop`, restored `#[inline]` on smart-pointer `SerializableType::serialize` (`Rc<T>`, `Arc<T>`, `RcWeak<T>`, `ArcWeak<T>`, `Box<T>`), removed `catch_unwind` from both `S25-R1` tests in `src/fsa/cspp_trie_concurrent.rs`, and upgraded all `#[tokio::test]` attributes across `src/concurrency/` to `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]`.
   - `S27-R1`: Hardened `scripts/audit_ledger.py` to index only git-tracked files (`git ls-files` + `git ls-tree`, never gitignored or untracked files such as `GEMINI.md`, `CLAUDE.md`, or `plan.md`), added Round 3 (27 new cases, 49/49 total adversarial checks caught) to `scripts/tests/audit_ledger_adversarial.md`, and replaced the three reconstructed RED blocks in §8 with verbatim `cargo test` output.
7. **`S28-R1..R3` Stage 11 Re-Review Round 4 Remediation (`b18bad3..`, §10)**:
   - `S28-R1` (`b18bad3`): Eliminated the self-deadlock in `WorkStealingExecutor::new` and `WorkStealingExecutor::ensure_workers_running` when Tokio synchronously drops a newly spawned worker future (e.g. `new()` called outside a runtime or `submit()` called on a closed/shutting-down runtime), by replacing the shared `Arc<Mutex<Option<JoinHandle<()>>>>` inside `ActiveWorkerGuard` with lock-free per-worker `state: Arc<AtomicU8>` (`WORKER_DEAD = 0`, `WORKER_SPAWNING = 1`, `WORKER_LIVE = 2`) plus `respawn_lock: Mutex<()>`, `fallback_runtime: Mutex<Option<tokio::runtime::Handle>>`, and `self_weak: Weak<Self>`, never locking `worker.handle` inside `ActiveWorkerGuard::drop` or holding `worker.handle` across `tokio::spawn` / `rt_handle.spawn`.
   - `S28-R3` (`b18bad3`): Eliminated the shutdown-window strand when `ActiveWorkerGuard::drop` runs on a dropping runtime while another runtime's `Handle` was recorded in `fallback_runtime`, documented `# Errors` on `WorkStealingExecutor::submit` and `WorkStealingExecutor::submit_closure`, fixed the 3 stale/imprecise `InstanceTls` comments (`b18bad3:src/thread/instance_tls.rs:39-42, 312-315, 384-386, 720-722`), and disclosed in §8.3 that `test_s25_r1_lookup_multi_insert_near_pool_end` inserts `2` keys (`0..2u32`) in `4929ccc` (reduced from up to `32` keys `0..32u32` in `c64babc`).
   - `S28-R2`: Generalized `scripts/audit_ledger.py` (removing all hardcoded test literals/numbers and non-repo whitelists, indexing only committed git state, and verifying percentage arithmetic, prose decimal measurement backing, panic assertion tokens, and AST/citation semantics), added Round 4 (30 new cases, 79 total adversarial cases across Rounds 1–4) to `scripts/tests/audit_ledger_adversarial.md`, and aligned §8.3 and §9.1 with the exact committed code.
8. **`S29-R1..R3` Stage 11 Final Fix Round Remediation (`c0a56de..fd1ec65`, §11)**:
   - `S29-R1` (`c0a56de`): Keyed the thread-local fallback runtime cache `LAST_RECORDED_FALLBACK` in `src/concurrency/work_stealing.rs` on `Option<(executor_id, runtime_id)>` via `tokio_runtime_handle_id(&rt_handle)` (`c0a56de:src/thread/mod.rs:65-78`), setting the marker only when `tokio::runtime::Handle::try_current()` returns `Ok` and refreshing `fallback_runtime` whenever the active runtime changes for the same executor (`test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime` and `test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime`, plus splice-and-run script `scripts/verify_s29_r1_red.sh`).
   - `S29-R3` (`fd1ec65`): Changed `ActiveWorkerGuard::drop` to `.take()` `fallback_runtime` and capture `Weak<WorkStealingExecutor>` inside the spawned respawn task (`drop(exec)` before `rt_handle.spawn`), cleared `fallback_runtime` in `WorkStealingExecutor::shutdown`, replaced `std::hint::spin_loop()` with `std::thread::yield_now()` in `ensure_workers_running`'s `!progressed` window, and documented the ~50–63% per-run single-line revert detection rate of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` Part 2.
9. **Phase D (`B7`, `B8`, `D1`, `D9`, `D10`)**:
   - `src/thread/` undocumented `unsafe`: `2 -> 0` (`68/68 = 100.0%` at `c0a56de`, `67/67 = 100.0%` at `b18bad3`, `59/59 = 100.0%` at `c87db71`); `src/concurrency/` `unsafe`: `0/0`; `src/fsa/` `unsafe`: `131 -> 99` (`99/99 = 100.0%`); crate-wide `max_undocumented`: `6 -> 4`.
   - `src/concurrency/` honesty markers (`D1`): `3 -> 0`; crate-wide `max_markers`: `30 -> 27`.
   - `src/thread/` `.expect()` (`D10`): `8 -> 0` (all 8 in `src/thread/instance_tls.rs`); `src/concurrency/` `.expect()`: `5 -> 0`; crate-wide `max_expects`: `82`.
   - `src/concurrency/pipeline.rs` (`2,183 -> 2,366` lines) split into 4 `< 800`-line submodules (`D9`): `src/concurrency/pipeline/mod.rs` (`627` lines at `4929ccc`), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`797` lines), and `src/concurrency/pipeline/integration_tests.rs` (`797` lines).
   - Gate baseline ratcheted (`B8`): `lib_debug: 3151 -> 3187`, `lib_release: 3168 -> 3204`, `doctests: 229`, `compile_fail` doctests: `7 -> 8`, `all_targets: 3812 -> 3849`.

---

## 2. Commit Table (`b8e7f8b..fd1ec65`)

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
| 13 | `ee8b29a` | `48438fe` | 2026-10-07 | docs(review): rewrite thread_and_concurrency.md, fix io.md citations, and upgrade audit_ledger.py [S25-R3, S24-R2, B8] | 5 | +656 / −343 |
| 14 | `1141a79` | `ee8b29a` | 2026-10-07 | fix(io): drop #[inline] on primitive SerializableType::serialize methods [S26-R4] | 2 | +13 / −16 |
| 15 | `c87db71` | `1141a79` | 2026-10-07 | fix(thread): preserve InstanceTls value across re-entrant or panicking clone and hold 65k live instances in churn test [S26-R2, S26-R3] | 2 | +377 / −120 |
| 16 | `c64babc` | `c87db71` | 2026-10-07 | fix(concurrency,fsa): park idle workers on Notify, respawn global workers after runtime drop, tighten S25-R1/R2/R6 tests, and deduplicate lookup_inner zpath compare [S26-R3, S26-R5] | 3 | +231 / −74 |
| 17 | `2050294` | `c64babc` | 2026-10-08 | docs(review): fix thread_and_concurrency.md and io.md accuracy and harden audit_ledger.py [S26-R1, B8] | 4 | +539 / −271 |
| 18 | `4929ccc` | `2050294` | 2026-10-08 | fix(concurrency,thread,io,fsa): guard unpolled workers on runtime drop, gate notify on parked_workers, tighten S25-R6 test, and restore smart-ptr inline [S27-R2, S27-R3, S27-R4, S27-R5] | 9 | +334 / −209 |
| 19 | `6e7fee4` | `4929ccc` | 2026-10-08 | docs(review): update thread_and_concurrency.md and io.md with verbatim RED outputs and harden audit_ledger.py [S27-R1, B8] | 4 | +492 / −141 |
| 20 | `b18bad3` | `6e7fee4` | 2026-10-08 | fix(concurrency,thread): avoid worker-handle lock in ActiveWorkerGuard::drop and across spawn [S28-R1, S28-R3] | 2 | +243 / −51 |
| 21 | `f37cf61` | `b18bad3` | 2026-10-08 | docs(review): generalize audit_ledger.py and update Stage 11 review ledger for S28 [S28-R2, S28-R3, B8] | 4 | +1084 / −275 |
| 22 | `c0a56de` | `f37cf61` | 2026-10-09 | fix(concurrency): key thread-local fallback runtime cache on (executor_id, runtime_id) [S29-R1] | 4 | +298 / −10 |
| 23 | `fd1ec65` | `c0a56de` | 2026-10-09 | fix(concurrency): release dead fallback handles, hold Weak in fallback task, and yield on respawn wait [S29-R3] | 1 | +15 / −5 |

Note on `b74f6fe..ee8b29a` (`S26-R1` disclosure): `b74f6fe` added `MatrixEntry` (`b74f6fe:src/thread/instance_tls.rs:50-53`) with `drop_fn: unsafe fn(*mut ())` preceded by a `///` doc comment instead of a `// SAFETY:` line comment, leaving 5 undocumented `unsafe` occurrences crate-wide (1 above the ceiling of 4) at `b74f6fe` and `48438fe`; `ee8b29a` replaced that doc comment with a `// SAFETY:` comment above `MatrixEntry` (`ee8b29a:src/thread/instance_tls.rs:50-53`). Every commit in the `S26`, `S27`, `S28`, and `S29` remediation rounds (`1141a79`, `c87db71`, `c64babc`, `2050294`, `4929ccc`, `6e7fee4`, `b18bad3`, `f37cf61`, `c0a56de`, `fd1ec65`, and the ledger commit) passes the gate individually.

---

## 3. `C10` Findings & Fixes in `src/thread/` (`05d4245`, `6571c89`, `b74f6fe`, `c87db71`)

### 3.1 `src/thread/atomic_ext.rs`
- **C10.A (Broken AT&T syntax in Intel-syntax `std::arch::asm!` + `rbx` register-aliasing in `cas_weak_u128`)**:
  - **Before**: `cas_weak_u64_asm` (`39814be:src/thread/atomic_ext.rs:142-159`), `atomic_increment_u64` (`39814be:src/thread/atomic_ext.rs:196-209`), and `atomic_exchange_u64` (`39814be:src/thread/atomic_ext.rs:217-231`) used AT&T addressing syntax `({ptr})` and reversed operand order without `options(att_syntax)`. Furthermore, `cas_weak_u128` (`39814be:src/thread/atomic_ext.rs:165-188`) was gated behind `#[cfg(all(target_arch = "x86_64", target_feature = "cmpxchg16b"))]` and used `in("rbx") desired_lo` (which LLVM rejects because `rbx` is reserved by LLVM). In `05d4245`, replacing `in("rbx")` with generic `ptr = in(reg) ptr` (`05d4245:src/thread/atomic_ext.rs:176-204`) allowed LLVM under `-Zsanitizer=thread --release` to allocate `ptr` to `%rbx`, so `"xchg rbx, {desired_lo:r}"` overwrote `ptr` with `desired_lo` before `lock cmpxchg16b [{ptr}]`.
  - **After**: Replaced `({ptr})` and reversed operand order with Intel syntax `[{ptr}]` in `cas_weak_u64_asm` (`05d4245:src/thread/atomic_ext.rs:151-169`), `atomic_increment_u64` (`05d4245:src/thread/atomic_ext.rs:211-226`), and `atomic_exchange_u64` (`05d4245:src/thread/atomic_ext.rs:233-248`). In `cas_weak_u128` (`6571c89:src/thread/atomic_ext.rs:176-203`), used `#[target_feature(enable = "cmpxchg16b")]` and pinned `in("rdi") ptr`, `inout("rsi") desired_lo => _`, and `inout("rax") expected_lo` so LLVM can never allocate `ptr` or `desired_lo` to `rbx` across `xchg rbx, rsi`.
  - **Regression test**: `test_c10_a_x86_64_asm_exchange_increment_and_cas_modify_memory` (`6571c89:src/thread/atomic_ext.rs:619-652`).
- **C10.B (`atomic_maximize`, `atomic_minimize`, and `update_if` panicking on `Ordering::Release` / `Ordering::AcqRel`)**:
  - **Before**: `atomic_maximize` (`39814be:src/thread/atomic_ext.rs:41-54`), `atomic_minimize` (`39814be:src/thread/atomic_ext.rs:57-70`), and `update_if` (`39814be:src/thread/atomic_ext.rs:101-114`) passed `order` directly to `self.load(order)` and as the `compare_exchange_weak` failure ordering, panicking at runtime whenever `order` was `Ordering::Release` or `Ordering::AcqRel` (`"there is no such thing as a release load"`).
  - **After**: Added `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) mapping `Release -> Relaxed` and `AcqRel -> Acquire`, and passed `load_ordering(order)` in `atomic_maximize` (`05d4245:src/thread/atomic_ext.rs:50-64`), `atomic_minimize` (`05d4245:src/thread/atomic_ext.rs:67-81`), and `update_if` (`05d4245:src/thread/atomic_ext.rs:112-125`).
  - **Regression test**: `test_c10_b_atomic_maximize_minimize_update_if_accept_release_and_acqrel` (`6571c89:src/thread/atomic_ext.rs:655-665`).
- **C10.C + `S25-R5` (`AsAtomic::as_atomic(&self)` unsound shared-reference-to-atomic cast + compile-time size/alignment assertions)**:
  - **Before**: `AsAtomic` (`39814be:src/thread/atomic_ext.rs:309-317`) declared `fn as_atomic(&self) -> &Self::Atomic`, transmuting `&T` (shared read-only reference without `UnsafeCell`) into `&Atomic*` across `impl_as_atomic!` (`39814be:src/thread/atomic_ext.rs:319-337`), which is immediate Undefined Behavior under Rust's aliasing model (shared read-only provenance cannot be used for atomic writes/RMWs) and created data races with concurrent `&T` readers.
  - **After**: Changed `AsAtomic` (`05d4245:src/thread/atomic_ext.rs:340-348`) to `fn as_atomic(&mut self) -> &Self::Atomic` (with a `compile_fail` doctest on `AsAtomic` at `05d4245:src/thread/atomic_ext.rs:332-348`), and added `const` size and alignment assertions (`size_of::<$t>() == size_of::<$atomic>()` and `align_of::<$t>() >= align_of::<$atomic>()`) in `impl_as_atomic!` (`b74f6fe:src/thread/atomic_ext.rs:351-380`) and `AsAtomic` for `*mut T` (`b74f6fe:src/thread/atomic_ext.rs:395-420`).
  - **Regression test**: `test_c10_c_as_atomic_from_mut_is_miri_clean` (`b74f6fe:src/thread/atomic_ext.rs:684-695`).

### 3.2 `src/thread/linux_futex.rs`
- **C10.D (`FutexCondvar::notify_all` and `FutexRwLock::unlock_write` waking only 1 waiter due to `u32::MAX as i32 == -1`)**:
  - **Before**: `sys::futex_wake` (`39814be:src/thread/linux_futex.rs:73-85`) and `LinuxFutex::futex_wake` (`39814be:src/thread/linux_futex.rs:113-127`) passed `count` directly to `sys::futex` (`39814be:src/thread/linux_futex.rs:25-45`), which passed `u32::MAX` (`-1` as signed 32-bit `int nr_wake` in Linux `FUTEX_WAKE_PRIVATE`), causing `FutexCondvar` `notify_all` (`39814be:src/thread/linux_futex.rs:283-287`) and `FutexRwLock` `unlock_write` (`39814be:src/thread/linux_futex.rs:384-387`) to wake at most 1 waiter.
  - **After**: Clamped `let clamped = count.min(i32::MAX as u32)` in `sys::futex_wake` (`05d4245:src/thread/linux_futex.rs:82-98`) and passed `i32::MAX as u32` in `FutexCondvar` `notify_all` (`05d4245:src/thread/linux_futex.rs:313-319`) and `FutexRwLock` `unlock_write` (`05d4245:src/thread/linux_futex.rs:418-421`).
  - **Regression test**: `test_c10_d_notify_all_wakes_all_waiters` (`05d4245:src/thread/linux_futex.rs:590-626`).
- **C10.E (`sys::futex_wait` `tv_sec` `as i64` wraparound on `Duration::MAX` + `sys::futex` variadic C ABI mismatch under Miri + `FutexCondvar` ordering upgrades)**:
  - **Before**:
    1. `sys::futex_wait` (`39814be:src/thread/linux_futex.rs:49-69`) cast `d.as_secs() as i64`, wrapping `u64::MAX` to `-1` and failing with `EINVAL` (`errno 22`).
    2. `sys::futex` (`39814be:src/thread/linux_futex.rs:25-45`) cast all arguments (`uaddr`, `op`, `val`, `timeout`, `uaddr2`, `val3`) `as usize` when calling `libc::syscall(SYS_futex, ...)`, which Miri rejected as UB (`scalar size mismatch: expected 4 bytes but got 8 bytes`) on 32-bit `op`/`val`/`val3` parameters.
    3. `FutexCondvar` methods `wait` (`39814be:src/thread/linux_futex.rs:240-252`), `wait_timeout` (`39814be:src/thread/linux_futex.rs:255-272`), `notify_one` (`39814be:src/thread/linux_futex.rs:276-280`), and `notify_all` (`39814be:src/thread/linux_futex.rs:283-287`) loaded and incremented `self.sequence` with `Ordering::Relaxed`. (`FutexMutex` and `FutexRwLock` already used `Acquire`/`Release` on lock/unlock transitions in `39814be`.)
  - **After**:
    1. `sys::futex_wait` (`05d4245:src/thread/linux_futex.rs:50-75`) clamps `d.as_secs().min(i64::MAX as u64) as i64` and binds `let ts: timespec;` in the outer frame.
    2. `sys::futex` (`05d4245:src/thread/linux_futex.rs:30-43`) passes `(SYS_futex, uaddr, op, val, timeout, uaddr2, val3)` with their exact C types.
    3. Upgraded `FutexCondvar` `self.sequence` operations in `wait` (`05d4245:src/thread/linux_futex.rs:259-275`), `wait_timeout` (`05d4245:src/thread/linux_futex.rs:278-302`), `notify_one` (`05d4245:src/thread/linux_futex.rs:304-310`), and `notify_all` (`05d4245:src/thread/linux_futex.rs:313-319`) from `Ordering::Relaxed` to `Ordering::Acquire` (loads) and `Ordering::Release` (`fetch_add`), and added `// ORDERING:` justifications to all remaining `Relaxed` spin and waiter-count operations in `FutexMutex`, `FutexCondvar`, and `FutexRwLock`.
  - **Regression test**: `test_c10_e_futex_wait_duration_max_does_not_einval` (`05d4245:src/thread/linux_futex.rs:629-638`).

### 3.3 `src/thread/instance_tls.rs`
- **C10.F1 (`CleanupHandle` `drop` leaving current thread's value alive until ID recycling + `RefCell` borrow panic on re-entrant `T::drop`)**:
  - **Before**: `CleanupHandle` `drop` (`39814be:src/thread/instance_tls.rs:419-439`) only pushed `(self.id, self.generation.wrapping_add(1))` to `free_ids` without clearing the dropping thread's `TlsMatrix` cell, leaving `T` alive on the current thread until another `InstanceTls<T>` reused that slot; furthermore, `InstanceTls::set` (`39814be:src/thread/instance_tls.rs:178-194`) and `InstanceTls::remove` (`39814be:src/thread/instance_tls.rs:197-209`) dropped the old `Option<(u32, T)>` inside `TLS_MATRICES.with(|matrices| { let mut matrices = matrices.borrow_mut(); ... })`, so any `T::drop` touching `InstanceTls` panicked with `RefCell already borrowed`.
  - **After**: In `b74f6fe` (`S25-R4`/`S25-R5`, §7.3) and `c87db71` (`S26-R2`, §8.2), `InstanceTls::set` (`c87db71:src/thread/instance_tls.rs:411-434`), `InstanceTls::remove` (`c87db71:src/thread/instance_tls.rs:437-464`), `InstanceTls::get` (`c87db71:src/thread/instance_tls.rs:312-369`), `InstanceTls::get_value` (`c87db71:src/thread/instance_tls.rs:373-399`), and `CleanupHandle` `drop` (`c87db71:src/thread/instance_tls.rs:686-734`) release the `TLS_MATRICES` `RefCell` borrow before running `T::drop`, `T::clone`, or `T::default`, and `CleanupHandle` `drop` uses `TLS_MATRICES.try_with` + `try_borrow_mut` so thread-exit destructor ordering never aborts.
  - **Regression test**: `test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic` (`c87db71:src/thread/instance_tls.rs:1109-1142`).
- **C10.F2 (Zero-dimension `ROWS == 0` / `COLS == 0` / `POOL_SIZE == 0` and `ROWS.checked_mul(COLS)` overflow panics + production `.expect()` removal)**:
  - **Before**: `InstanceTls::allocate_id` (`39814be:src/thread/instance_tls.rs:224-260`) computed `ROWS * COLS` and `% COLS` without checking for zero or overflow, `TlsPool` `new` (`39814be:src/thread/instance_tls.rs:531-543`) allowed `POOL_SIZE == 0` (panicking with `% 0` in `get_next`: `39814be:src/thread/instance_tls.rs:546-552`), and `src/thread/instance_tls.rs` had 8 production `.expect()` calls (at lines `150`, `180`, `226`, `250`, `355`, `421`, `535`, and `550` of `39814be:src/thread/instance_tls.rs`).
  - **After**: Added `validate_dimensions` (`6571c89:src/thread/instance_tls.rs:95-110`) using `ROWS.checked_mul(COLS)`, rejecting `ROWS == 0`, `COLS == 0`, or `> u32::MAX as usize` in `InstanceTls::new` (`6571c89:src/thread/instance_tls.rs:113-129`); rejected `POOL_SIZE == 0` in `TlsPool` `new` (`6571c89:src/thread/instance_tls.rs:573-589`); and removed all 8 production `.expect()` calls in `src/thread/instance_tls.rs` (`D10`).
  - **Regression test**: `test_c10_f2_zero_pool_and_overflow_dimensions_return_err` (`c87db71:src/thread/instance_tls.rs:1145-1154`).

---

## 4. `C10` Findings & Fixes in `src/concurrency/` (`a8ac402`, `d96a162`, `6571c89`, `48438fe`, `c64babc`)

### 4.1 `src/concurrency/work_stealing.rs`
- **C10.WS1 (`WorkStealingQueue::pop_local` stranded tasks moved into `steal_queue` by `balance()`)**:
  - **Before**: `balance()` (`05d4245:src/concurrency/work_stealing.rs:190-211`) moved half of `local_queue` into `steal_queue`, but `pop_local()` (`05d4245:src/concurrency/work_stealing.rs:148-154`) only popped from `local_queue` and `worker_loop` (`05d4245:src/concurrency/work_stealing.rs:419-453`) / `find_task()` (`05d4245:src/concurrency/work_stealing.rs:481-510`) skipped `worker_id` when stealing. On a 1-worker executor (or when all other workers were idle), tasks moved to `steal_queue` were stranded forever.
  - **After**: `pop_local()` (`a8ac402:src/concurrency/work_stealing.rs:154-168`) falls back to popping from the back of `steal_queue` when `local_queue` is empty.
  - **Regression test**: `test_c10_ws1_balanced_tasks_are_popped_by_owner_and_single_worker_executor` (`a8ac402:src/concurrency/work_stealing.rs:870-888`).
- **C10.WS2 (`WorkStealingExecutor::shutdown` aborted workers immediately and accepted post-shutdown submissions)**:
  - **Before**: `shutdown()` (`05d4245:src/concurrency/work_stealing.rs:406-417`) called `handle.abort()` on every worker, discarding queued tasks and cancelling in-flight tasks mid-execution, while `submit()` (`05d4245:src/concurrency/work_stealing.rs:343-370`) never checked `self.shutdown`.
  - **After**: Checked `self.shutdown` in `submit()` (`a8ac402:src/concurrency/work_stealing.rs:427-465`; synchronized under queue locks in `48438fe:src/concurrency/work_stealing.rs:476-515`), drained all local, steal, and global queues in `worker_loop` (`a8ac402:src/concurrency/work_stealing.rs:523-615`) and `shutdown()` (`48438fe:src/concurrency/work_stealing.rs:554-602`), and awaited clean worker completion.
  - **Regression test**: `test_c10_ws2_shutdown_drains_queued_and_active_tasks_and_rejects_post_shutdown_submit` (`a8ac402:src/concurrency/work_stealing.rs:891-921`).
- **C10.WS3 (Panicking task leaked `active_tasks` and killed the worker thread)**:
  - **Before**: `worker_loop` (`05d4245:src/concurrency/work_stealing.rs:419-453`) incremented `active_tasks` before `task.execute().await` and decremented it after; a panicking task unwound the worker loop, permanently leaking `active_tasks` and killing the worker.
  - **After**: Added `ActiveTaskGuard` (`a8ac402:src/concurrency/work_stealing.rs:563-566`) and `CatchUnwindFuture` (`a8ac402:src/concurrency/work_stealing.rs:344-346`) around `task.execute()` so panicking tasks always decrement `active_tasks` and record elapsed time without killing the worker.
  - **Regression test**: `test_c10_ws3_panicking_task_does_not_leak_active_tasks_or_kill_worker` (`a8ac402:src/concurrency/work_stealing.rs:924-957`).
- **C10.WS5 (`WorkStealingQueue::steal` failed to steal when `local_queue.len() == 1`)**:
  - **Before**: `steal()` (`05d4245:src/concurrency/work_stealing.rs:157-187`) guarded `local_queue` stealing behind `if local_queue.len() > 1`, returning `None` even when a stealable task was waiting in `local_queue`.
  - **After**: `steal()` (`a8ac402:src/concurrency/work_stealing.rs:189-217`) and `steal_and_activate()` (`a8ac402:src/concurrency/work_stealing.rs:220-248`) check `if !local_queue.is_empty()` so a single queued stealable task can be stolen by an idle worker.
  - **Regression test**: `test_c10_ws5_steal_takes_single_queued_stealable_task` (`a8ac402:src/concurrency/work_stealing.rs:960-974`).
- **C10.WS6 (`WorkStealingExecutor::spawn` discarded `Fiber::id()` and `new` accepted `queue_capacity == 0`)**:
  - **Before**: `WorkStealingExecutor` `spawn` (`05d4245:src/concurrency/work_stealing.rs:553-558`) ignored `fiber.id()` and allocated a new disconnected ID, and `WorkStealingExecutor` `new` (`05d4245:src/concurrency/work_stealing.rs:283-318`) accepted `queue_capacity == 0`.
  - **After**: Rejected `queue_capacity == 0` in `WorkStealingExecutor` `new` (`a8ac402:src/concurrency/work_stealing.rs:366-404`) and preserved `fiber.id().as_u64()` in `WorkStealingExecutor` `spawn` (`a8ac402:src/concurrency/work_stealing.rs:712-742`; `48438fe:src/concurrency/work_stealing.rs:815-823`).
  - **Regression test**: `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected` (`48438fe:src/concurrency/work_stealing.rs:1057-1073`).

### 4.2 `src/concurrency/fiber_pool.rs`
- **C10.FP1 (`FiberPool::shutdown` returned while spawned/queued fibers were still running)**:
  - **Before**: `FiberPool::shutdown` (`05d4245:src/concurrency/fiber_pool.rs:282-292`) set `self.shutdown = true` and awaited `worker_handles` (which was always empty because `FiberPool::spawn` spawns onto Tokio directly), returning before active or queued fibers finished.
  - **After**: Added `queue_semaphore` to `FiberPool` (`6571c89:src/concurrency/fiber_pool.rs:116-123`), decremented `active_fibers` via `ActiveFiberGuard` (`6571c89:src/concurrency/fiber_pool.rs:241-244`, `drop` at `6571c89:src/concurrency/fiber_pool.rs:246-252`), rejected post-shutdown `spawn()` (`6571c89:src/concurrency/fiber_pool.rs:204-220`), and acquired all `total_permits` on `queue_semaphore` in `shutdown()` (`6571c89:src/concurrency/fiber_pool.rs:328-338`; chunked in `u32::MAX` steps in `48438fe:src/concurrency/fiber_pool.rs:331-354`).
  - **Regression test**: `test_c10_fp1_shutdown_waits_for_spawned_and_queued_fibers` (`6571c89:src/concurrency/fiber_pool.rs:561-592`).
- **C10.FP2 (`FiberPool::new` accepted `max_fibers == 0` / `initial_workers == 0`, causing `NaN` in `load_factor` / `queue_utilization`)**:
  - **Before**: `FiberPool::new` (`05d4245:src/concurrency/fiber_pool.rs:148-169`) did not validate `config`, so `max_fibers == 0` or `queue_capacity == 0` caused `0.0 / 0.0 == NaN` in `load_factor()` (`05d4245:src/concurrency/fiber_pool.rs:294-298`) and `stats()` (`05d4245:src/concurrency/fiber_pool.rs:250-265`).
  - **After**: Validated `max_fibers > 0`, `initial_workers > 0`, `initial_workers <= max_workers`, and `max_fibers.checked_add(queue_capacity)` in `FiberPool::new` (`6571c89:src/concurrency/fiber_pool.rs:149-195`) and guarded `queue_capacity == 0` in `stats()` (`6571c89:src/concurrency/fiber_pool.rs:289-325`).
  - **Regression test**: `test_c10_fp2_invalid_config_returns_err_instead_of_panic_or_nan` (`6571c89:src/concurrency/fiber_pool.rs:595-622`).

### 4.3 `src/concurrency/parallel_trie.rs`
- **C10.PT1 (`chunks(0)` panic when `chunk_size == 0` and unvalidated `max_workers == 0` in `ParallelTrieBuilder::build_louds_trie`)**:
  - **Before**: `ParallelTrieBuilder::build_louds_trie` (`05d4245:src/concurrency/parallel_trie.rs:40-75`) called `.chunks(self.chunk_size)` directly (panicking when `chunk_size == 0`) and never validated or enforced `self.max_workers`.
  - **After**: Validated non-zero `chunk_size` and `max_workers` and bounded concurrent chunk tasks with `tokio::sync::Semaphore::new(self.max_workers)` in `ParallelTrieBuilder::build_louds_trie` (`a8ac402:src/concurrency/parallel_trie.rs:40-100`), replaced `tries.into_iter().next().expect("non-empty tries")` with `let mut iter = tries.into_iter();` in `ParallelTrieBuilder` `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:102-135`) and `ParallelTrieOps` `merge_tries` (`a8ac402:src/concurrency/parallel_trie.rs:481-505`), and held the write lock across replica updates in `update_replicas_locked` (`a8ac402:src/concurrency/parallel_trie.rs:202-209`).
  - **Regression test**: `test_c10_pt1_zero_chunk_size_or_workers_returns_err_instead_of_panic` (`a8ac402:src/concurrency/parallel_trie.rs:719-737`).
- **C10.PT3 (`parallel_process`, `parallel_contains`, `parallel_prefix_search`, and `merge_tries` used `.expect()`, panicking on closure panic)**:
  - **Before**: `parallel_process` (`05d4245:src/concurrency/parallel_trie.rs:324-348`), `parallel_contains` (`05d4245:src/concurrency/parallel_trie.rs:202-227`), `parallel_prefix_search` (`05d4245:src/concurrency/parallel_trie.rs:230-258`), and both `merge_tries` methods (`05d4245:src/concurrency/parallel_trie.rs:78-115, 439-473`) contained 5 production `.expect(...)` calls, violating `D10` and crashing the caller when a `parallel_process` closure panicked.
  - **After**: Wrapped each operation in `std::panic::catch_unwind` and mapped task join errors to `Err(ZiporaError::system_error(...))` in `parallel_process` (`a8ac402:src/concurrency/parallel_trie.rs:352-390`), and replaced `.expect(...)` in `parallel_contains` (`48438fe:src/concurrency/parallel_trie.rs:235-269`) and `parallel_prefix_search` (`48438fe:src/concurrency/parallel_trie.rs:272-307`) with direct fallback evaluation, removing all 5 `.expect()` calls from `src/concurrency/` (`D10`).
  - **Regression test**: `test_c10_pt3_parallel_process_catches_closure_panic_as_err` (`a8ac402:src/concurrency/parallel_trie.rs:740-758`).

### 4.4 `src/concurrency/pipeline/` (`mod.rs`, `stages.rs`, `tests.rs`, `integration_tests.rs`)
- **C10.P1 (Multi-stage `Pipeline::execute_stream` always failed with `"pipeline: channel missing for prior stage"` + dropped stage errors)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:344-478`) stored `(tx, rx)` tuples inside `channels: Vec<Option<(Sender<T>, Receiver<T>)>>`, took `channels[0].take()...0` on iteration `i == 0` (discarding `.1`), and then tried `channels[i - 1].take()...1` on iteration `i == 1`, so every `stages.len() >= 2` pipeline immediately failed with `Err(ZiporaError::invalid_state("pipeline: channel missing for prior stage"))`; additionally, stage processing errors were silently ignored.
  - **After**: Rewrote `execute_stream` (`6571c89:src/concurrency/pipeline/mod.rs:262-422`; `48438fe:src/concurrency/pipeline/mod.rs:267-438`) to keep `inter_txs` and `inter_rxs` separate (`stage_input_rx` and `stage_output_tx`), flush partial batches at end-of-stream, and propagate stage errors/timeouts across both single-stage and multi-stage pipelines.
  - **Regression test**: `test_c10_p1_execute_stream_multi_stage_and_error_propagation` (`6571c89:src/concurrency/pipeline/integration_tests.rs:679-725`).
- **C10.P2 (`items_in_flight` leaked on stage error or timeout in `execute_stream` and `process_batch` and was not tracked in `execute_single`)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:344-478`) and `process_batch` (`05d4245:src/concurrency/pipeline.rs:481-526`) incremented `items_in_flight` before awaiting the stage future and only decremented on success, permanently leaking `items_in_flight` on stage error, timeout, or future cancellation, while `execute_single` (`05d4245:src/concurrency/pipeline.rs:302-323`) did not track `items_in_flight`.
  - **After**: Added `AtomicCountGuard` (`6571c89:src/concurrency/pipeline/mod.rs:183-186`) so `items_in_flight` is unconditionally decremented on scope exit in `execute_single` (`48438fe:src/concurrency/pipeline/mod.rs:219-245`), `execute_stream` (`48438fe:src/concurrency/pipeline/mod.rs:267-438`), and `process_batch` (`48438fe:src/concurrency/pipeline/mod.rs:441-490`).
  - **Regression test**: `test_c10_p2_items_in_flight_decremented_on_timeout_and_error` (`6571c89:src/concurrency/pipeline/integration_tests.rs:728-747`).
- **C10.P3 (`buffer_size == 0` panicked inside `mpsc::channel(0)`, `max_in_flight == 0` produced `NaN` utilization, and `start_timeout_checker` busy-spun on `Duration::ZERO`)**:
  - **Before**: `execute_stream` (`05d4245:src/concurrency/pipeline.rs:344-478`) passed `self.config.buffer_size` directly to `mpsc::channel`, panicking when `buffer_size == 0`; `stats()` (`05d4245:src/concurrency/pipeline.rs:529-558`) divided `items_in_flight as f64 / self.config.max_in_flight as f64` (`0.0 / 0.0 == NaN`); and `start_timeout_checker` (`05d4245:src/concurrency/pipeline.rs:240-262`) computed `self.batch_timeout / 4` (`Duration::ZERO` when `batch_timeout < 4ns`), spinning without yielding.
  - **After**: Clamped channel capacities to `.max(1)` in `execute_stream` (`6571c89:src/concurrency/pipeline/mod.rs:262-422`), guarded `max_in_flight == 0` in `stats()` (`6571c89:src/concurrency/pipeline/mod.rs:469-481`), and floored the checker sleep interval to `Duration::from_millis(1)` in `start_timeout_checker` (`6571c89:src/concurrency/pipeline/stages.rs:96-120`).
  - **Regression test**: `test_c10_p3_zero_buffer_size_max_in_flight_and_zero_batch_timeout_do_not_panic_or_nan` (`6571c89:src/concurrency/pipeline/integration_tests.rs:750-780`).
- **C10.P5 (`BatchCollector` expired the first item after an idle period immediately)**:
  - **Before**: `BatchCollector` `new` (`05d4245:src/concurrency/pipeline.rs:172-180`) initialized `last_flush` to `Instant::now()` at construction and after flushes, so when a collector sat idle for `> batch_timeout`, the very first item added had `last_flush.elapsed() >= batch_timeout` and was immediately flushed as a 1-item timeout batch on the next `check_timeout()`.
  - **After**: Reset `*last_flush = Instant::now()` when transitioning `batch` from empty to non-empty in `BatchCollector::add` (`6571c89:src/concurrency/pipeline/stages.rs:33-55`).
  - **Regression test**: `test_c10_p5_batch_collector_idle_before_first_item_waits_full_timeout` (`6571c89:src/concurrency/pipeline/integration_tests.rs:783-794`).
- **`D1` & `D9` (Honesty markers removed & `pipeline.rs` split into 4 `< 800`-line submodules)**:
  - Replaced the 3 honesty-marker comments in `BatchMapStage` `process_batch` (`6571c89:src/concurrency/pipeline/stages.rs:184-206`) and split `src/concurrency/pipeline.rs` (`2,366` lines at `a8ac402`) into `src/concurrency/pipeline/mod.rs` (`627` lines at `c64babc`), `src/concurrency/pipeline/stages.rs` (`270` lines), `src/concurrency/pipeline/tests.rs` (`797` lines), and `src/concurrency/pipeline/integration_tests.rs` (`797` lines) (`d96a162`, `6571c89`, `48438fe`, `c64babc`).

---

## 5. `C10.1` Crate-Wide Weak-Memory Ordering Audit (`05d4245`, `a8ac402`, `a0bcdff`, `6571c89`, `0932ce7`)

| Module / File | Atomic Ordering / Synchronization Upgrades & `// ORDERING:` Coverage |
|---------------|----------------------------------------------------------------------|
| `src/thread/atomic_ext.rs` | Fixed RMW initial-load/failure orderings via `load_ordering` (`05d4245:src/thread/atomic_ext.rs:38-44`) and added `// ORDERING:` comments to `AtomicBitOps` impl (`05d4245:src/thread/atomic_ext.rs:423-468`). |
| `src/thread/linux_futex.rs` | Upgraded `FutexCondvar` `sequence` operations in `wait` (`05d4245:src/thread/linux_futex.rs:259-275`), `wait_timeout` (`05d4245:src/thread/linux_futex.rs:278-302`), `notify_one` (`05d4245:src/thread/linux_futex.rs:304-310`), and `notify_all` (`05d4245:src/thread/linux_futex.rs:313-319`) from `Relaxed` to `Acquire`/`Release`, and documented all remaining `Relaxed` spin/waiter-count loads in `FutexMutex`, `FutexCondvar`, and `FutexRwLock` with `// ORDERING:`. |
| `src/thread/instance_tls.rs` | Documented `TlsPool` `get_next` monotonic `next_slot` `Relaxed` `fetch_add` with `// ORDERING:` (`c87db71:src/thread/instance_tls.rs:842-847`). |
| `src/concurrency/mod.rs`, `work_stealing.rs`, `fiber_pool.rs`, `parallel_trie.rs`, `pipeline/mod.rs` | Upgraded lifecycle and task-accounting atomics (`WorkStealingExecutor::shutdown` at `48438fe:src/concurrency/work_stealing.rs:554-602`, `FiberPool::spawn` at `6571c89:src/concurrency/fiber_pool.rs:204-239`, and `FiberPool::shutdown` at `6571c89:src/concurrency/fiber_pool.rs:328-338`) to `Acquire`/`Release`/`AcqRel`, and added `// ORDERING:` comments to all remaining `Relaxed` ID generators and telemetry counters. |
| `src/fsa/cspp_trie_concurrent.rs` | Upgraded `SharedPool::load_meta` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:208-211`) and `SharedPool::load_bytes` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:229-233`) from `load_relaxed` to `load_acquire`, kept `LockFreeFreelist::push` at `(Release, Relaxed)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`) paired with `Acquire` in `pop` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:408-436`), replaced all non-atomic `raw_ptr` / `get_slice` / `copy_nonoverlapping` byte accesses on `SharedPool::data: Box<[AtomicU32]>` with word-atomic `SharedPool::read_bytes_acquire` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:237-249`), `SharedPool::bytes_eq_relaxed` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:258-312`), and `SharedPool::write_bytes_relaxed` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:317-331`), and added `// ORDERING:` comments across the file. Eliminated 32 `unsafe` blocks (`131 -> 99` in `src/fsa/`). |
| `src/memory/lockfree_pool.rs` | Kept `deallocate_to_fast_bin` Treiber push CAS at `(Release, Relaxed)` (`0932ce7:src/memory/lockfree_pool.rs:689-745`) paired with `(AcqRel, Acquire)` in `pop_fast_bin` (`0932ce7:src/memory/lockfree_pool.rs:595-652`), documented `Relaxed` on `store_block_width` (`0932ce7:src/memory/lockfree_pool.rs:894-905`) and `load_block_width` (`0932ce7:src/memory/lockfree_pool.rs:915-927`), and added `// ORDERING:` comments to all `Relaxed` bump-cursor and statistics operations. |
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
- **`C10.B`, `C10.D`, `C10.E`, `C10.F1`, `C10.F2` runtime-RED on `39814be`** (tests spliced onto `39814be:src/thread/instance_tls.rs` at line `820` and `39814be:src/thread/linux_futex.rs` at line `554`; in `05d4245:src/thread/linux_futex.rs`, the production changes above `mod tests` add 38 lines, shifting `570` -> `608` and `587` -> `636`):
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
- **Miri RED on `sys::futex` (`39814be:src/thread/linux_futex.rs:25-45`)**:
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
Tests spliced onto `05d4245:src/concurrency/{fiber_pool,parallel_trie,pipeline,work_stealing}.rs` (before `d96a162` split `src/concurrency/pipeline.rs` into `src/concurrency/pipeline/integration_tests.rs`):
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

## 7. Stage 11 Re-Review Round 1 Remediation (`S25-R1..R9`: `f7e39ad..48438fe`)

### 7.1 `f7e39ad` — `S25-R8` & `S24-R2` (`src/io/data_input.rs`, `src/io/smart_ptr.rs`, `benches/s23_serialization_bench.rs`, `Cargo.toml`)
- Outlined `unexpected_eof_err` (`f7e39ad:src/io/data_input.rs:213-217`), added `DIRTY_OUTER_SER_STATE` (`f7e39ad:src/io/smart_ptr.rs:35`) and `DIRTY_OUTER_DE_STATE` (`f7e39ad:src/io/smart_ptr.rs:36`), and split `Vec<T>::deserialize` (`f7e39ad:src/io/smart_ptr.rs:1210-1237`) into a non-unwinding `0..cap` preallocated push loop plus `deserialize_vec_tail` (`f7e39ad:src/io/smart_ptr.rs:1184-1195`), recovering `Vec<u64>` decode to `0.949 ns/elem` (vs `0.99 ns` at `a8d8b39`) and `Vec<Vec<u32>>` decode to `111.68 ns/inner Vec` (vs `118.5 ns` at `a8d8b39`). Committed `benches/s23_serialization_bench.rs` (`f7e39ad:benches/s23_serialization_bench.rs:1-82`). See `docs/review/io.md` §12.2.

### 7.2 `0932ce7` — `S25-R1`, `S25-R4`, `S25-R9` (`src/fsa/cspp_trie_concurrent.rs`, `src/memory/lockfree_pool.rs`)
- **`S25-R1` (HIGH — `ConcurrentCsppTrie` 256-byte `read_zpath` over-read panic on `lookup` / `contains` near pool end)**:
  - **Root cause**: `6571c89` replaced `ConcurrentNodeView::zpath_slice` with `ConcurrentNodeView::read_zpath(&self, dst: &mut [u8])` (`6571c89:src/fsa/cspp_trie_concurrent.rs:636-645`), and `lookup_inner` (`6571c89:src/fsa/cspp_trie_concurrent.rs:985-1026`) called `let mut zpath = [0u8; 256]; view.read_zpath(&mut zpath);` (`6571c89:src/fsa/cspp_trie_concurrent.rs:993-994`), reading 64 `AtomicU32` words (`256` bytes) into `zpath` regardless of `zlen` (`new_suffix_chain` allocates only `header + zlen.div_ceil(4)` words for tail leaves), panicking with `index out of bounds` whenever a node with `zlen < 255` was allocated within 64 words of `pool.data.len()`.
  - **Fix in `0932ce7:src/fsa/cspp_trie_concurrent.rs`**:
    1. Removed `ConcurrentNodeView::read_zpath` and added zero-copy word-wise `SharedPool::bytes_eq_relaxed(slot, expected)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:258-312`), which bounds `end = pos.saturating_add(zlen).min(key.len())` in `lookup_inner` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1094-1156`) and `contains_with_guard` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1164-1166`) so a reader never reads more than `expected.len().div_ceil(4) <= zlen.div_ceil(4)` words from `&self.data[slot..]`.
    2. Cached the `Acquire`-loaded header word `raw: u32` in `ConcurrentNodeView` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:514-519`), peeled the 256-child root fast-node step in `lookup_inner` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:1094-1156`), and inlined leaf `valpos` extraction.
    3. Relaxed `LockFreeFreelist::push` CAS from `(AcqRel, Relaxed)` (`6571c89:src/fsa/cspp_trie_concurrent.rs:313-338`) back to `(Release, Relaxed)` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`) with `// ORDERING:` comment on `pop` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:408-436`), and relaxed `deallocate_to_fast_bin` (`0932ce7:src/memory/lockfree_pool.rs:689-745`) from `(AcqRel, Relaxed)` to `(Release, Relaxed)` and `store_block_width` (`0932ce7:src/memory/lockfree_pool.rs:894-905`) / `load_block_width` (`0932ce7:src/memory/lockfree_pool.rs:915-927`) from `Release`/`Acquire` to `Relaxed`.
  - **Regression tests**: `test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:3112-3134`; tightened in `c64babc:src/fsa/cspp_trie_concurrent.rs:3110-3143`) and `test_s25_r1_lookup_multi_insert_near_pool_end` (`0932ce7:src/fsa/cspp_trie_concurrent.rs:3137-3161`; tightened in `c64babc:src/fsa/cspp_trie_concurrent.rs:3146-3185`).

### 7.3 `b74f6fe` — `S25-R4`, `S25-R5`, `S25-R7`, `S25-R9` (`src/thread/atomic_ext.rs`, `src/thread/instance_tls.rs`)
- **`S25-R4`, `S25-R5`, `S25-R7`, `S25-R9` (`InstanceTls` thread-exit safety, heap row table, ID pool unification, and `O(1)` MRU cache)**:
  - **Fixes in `b74f6fe:src/thread/instance_tls.rs`**:
    1. Replaced inline `[Option<Box<TlsRow<T, COLS>>>; ROWS]` (which stack-overflowed on large `ROWS` such as `ROWS = 4_000_000`) with a heap-backed lazy row pointer table `rows: Vec<Option<*mut [TlsCell<T>]>>` in `TlsMatrix` (`b74f6fe:src/thread/instance_tls.rs:45-47`) and `alloc_row` (`b74f6fe:src/thread/instance_tls.rs:518-529`), `ensure_cell` (`b74f6fe:src/thread/instance_tls.rs:531-541`), and `lookup_cell` (`b74f6fe:src/thread/instance_tls.rs:543-550`).
    2. Added `TlsMatrices` (`b74f6fe:src/thread/instance_tls.rs:57-60`) with a 1-entry MRU `(TypeId, *mut ())` cache in `get_or_init_matrix` (`b74f6fe:src/thread/instance_tls.rs:71-81`) and `lookup_matrix` (`b74f6fe:src/thread/instance_tls.rs:111-125`), re-entrancy-safe `TlsMatrices` `drop` (`b74f6fe:src/thread/instance_tls.rs:129-152`), and `TLS_MATRICES.try_with` + `try_borrow_mut`, speeding up `InstanceTls::<u64>::get` ~9x.
    3. Unified `allocate_id` (`b74f6fe:src/thread/instance_tls.rs:414-495`) and `CleanupHandle` `drop` (`b74f6fe:src/thread/instance_tls.rs:574-618`) on `global_registry` (`b74f6fe:src/thread/instance_tls.rs:189-191`) so `test_instance_ids_are_recycled_after_drop` and 65,536+ ID churn tests exercise the same `(id, generation)` free list.
  - **Regression tests**:
    - `test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak` (`b74f6fe:src/thread/instance_tls.rs:1041-1081`; tightened in `c87db71:src/thread/instance_tls.rs:1157-1227`)
    - `test_s25_r7_parallel_80k_churn_across_threads` (`b74f6fe:src/thread/instance_tls.rs:1085-1132`)
    - `test_s25_r5_reentrant_drop_at_thread_exit_does_not_abort` (`b74f6fe:src/thread/instance_tls.rs:1136-1175`)
    - `test_s25_r9_reentrant_clone_and_large_rows_do_not_panic_or_overflow_stack` (`b74f6fe:src/thread/instance_tls.rs:1179-1211`)

### 7.4 `48438fe` — `S25-R2`, `S25-R4`, `S25-R6`, `S25-R7`, `S25-R9` (`src/concurrency/{work_stealing,fiber_pool,parallel_trie}.rs`, `src/concurrency/pipeline/{mod,tests,integration_tests}.rs`)
- **`S25-R2` (HIGH — `WorkStealingExecutor::spawn(fiber)` hung after global-init runtime drop or nested fiber await on 1 worker)**:
  - Fixed `WorkStealingExecutor::spawn` (`48438fe:src/concurrency/work_stealing.rs:815-823`) to spawn `fiber` directly onto the current Tokio runtime via `tokio::spawn(async move { fiber.await })` while preserving `fiber.id().as_u64()`.
  - **Regression test**: `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` (`48438fe:src/concurrency/work_stealing.rs:1076-1149`; tightened in `c64babc:src/concurrency/work_stealing.rs:1165-1264`).
- **`S25-R4` & `S25-R6` (`WorkStealingQueue` lock-free steal check, `is_idle()` load order, `submit`/`shutdown` synchronization, sync `stage_timeout`, and isolated `local_stage_stats`)**:
  - Added `steal_len: AtomicUsize` in `WorkStealingQueue` (`48438fe:src/concurrency/work_stealing.rs:103-109`) and an $O(1)$ back-check in `insert_by_priority` (`48438fe:src/concurrency/work_stealing.rs:124-135`) on `WorkStealingQueue` owner push+pop.
  - Reordered `WorkStealingExecutor::is_idle()` (`48438fe:src/concurrency/work_stealing.rs:765-767`) to check `self.total_queued() == 0` before `self.stats.active_tasks.load(Ordering::Acquire) == 0`, and synchronized `submit()` (`48438fe:src/concurrency/work_stealing.rs:476-515`) with `shutdown()` (`48438fe:src/concurrency/work_stealing.rs:554-602`) via `try_push_local_unless_shutdown` (`48438fe:src/concurrency/work_stealing.rs:161-180`) and a final drain pass in `shutdown()`.
  - Checked `self.config.stage_timeout` after synchronous stages (`elapsed > stage_timeout`) in `Pipeline::execute_single` (`48438fe:src/concurrency/pipeline/mod.rs:219-245`), `Pipeline::execute_stream` (`48438fe:src/concurrency/pipeline/mod.rs:267-438`), and `Pipeline::process_batch` (`48438fe:src/concurrency/pipeline/mod.rs:441-490`), isolated per-call `local_stage_stats` in `execute_stream`, and added a `try_recv()` batching fast path.
- **`S25-R7` & `S25-R9` (Multi-threaded tests with non-atomic payload writes + `ParallelLoudsTrie` fallback evaluation)**:
  - Computed exact fallback results from a cloned `replicas` snapshot on `spawn_blocking` failure in `ParallelLoudsTrie::parallel_contains` (`48438fe:src/concurrency/parallel_trie.rs:235-269`) and `parallel_prefix_search` (`48438fe:src/concurrency/parallel_trie.rs:272-307`), and asserted `handle.id() == expected_id.as_u64()` in `test_c10_ws6_spawn_preserves_fiber_id_and_zero_capacity_rejected` (`48438fe:src/concurrency/work_stealing.rs:1057-1073`).
  - Added multi-threaded `#[tokio::test(flavor = "multi_thread", worker_threads = 4)]` tests with non-atomic payload writes across all 4 concurrency modules:
    - `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` (`48438fe:src/concurrency/work_stealing.rs:1152-1228`; tightened in `c64babc:src/concurrency/work_stealing.rs:1267-1353`)
    - `test_s25_r7_multithreaded_fiber_pool_payload_writes_and_shutdown` (`48438fe:src/concurrency/fiber_pool.rs:626-662`)
    - `test_s25_r7_multithreaded_execute_stream_payload_writes_and_concurrent_stage_stats` (`48438fe:src/concurrency/pipeline/tests.rs:726-796`)
    - `test_s25_r7_multithreaded_parallel_louds_trie_concurrent_inserts_and_reads` (`48438fe:src/concurrency/parallel_trie.rs:776-805`)

### 7.5 Stage 11 Performance Summary Table

Commands for the committed benchmark harnesses in this table:
```bash
cargo test --locked --release --lib io::smart_ptr::tests::bench_s24_r3_io_deserialize_timings -- --ignored --nocapture
cargo bench --bench s23_serialization_bench
cargo test --release --locked --lib bench_s29_r1_executor_timings -- --ignored --nocapture
```

| Benchmark | Committed Harness | `a8d8b39` / `00652d2` | `f7e39ad` / `ee8b29a` | `1141a79` / `2050294` | `4929ccc` | `f37cf61` | `fd1ec65` (`S29`) |
|---|---|---|---|---|---|---|---|
| `Vec<u64>::deserialize` (`1,000,000` elems) | `bench_s24_r3_io_deserialize_timings` / `benches/s23_serialization_bench.rs` | `0.99 ns` | `0.949 ns` | `1.048 ns` | — | — | — |
| `Vec<Vec<u32>>::deserialize` (`1,000 × 100`) | `bench_s24_r3_io_deserialize_timings` / `benches/s23_serialization_bench.rs` | `118.5 ns` | `111.68 ns` | `111.83 ns` | — | — | — |
| `Vec<Vec<u32>>::serialize` (`1,000 × 100`) | `bench_s24_r3_io_deserialize_timings` / `benches/s23_serialization_bench.rs` | `84.7 ns` | `100.9 ns` | `73.79 ns` | `82.2 ns` | — | — |
| `WorkStealingExecutor` 8-thread submit+drain (`400,000` tasks) | `bench_s29_r1_executor_timings` (`src/concurrency/work_stealing.rs`) | — | — | — | — | `313.8 ns` | `314.5 ns` |

---

## 8. Stage 11 Re-Review Round 2 Remediation (`S26-R1..R5`: `1141a79..2050294`)

### 8.1 `1141a79` — `S26-R4` (`src/io/smart_ptr.rs`, `benches/s23_serialization_bench.rs`)
- Removed the 15 `#[inline]` attributes added by `f7e39ad` on `SerializableType` `serialize` implementations (`i32` at `1141a79:src/io/smart_ptr.rs:1148-1157`, `u32` at `1141a79:src/io/smart_ptr.rs:1159-1168`, `String` at `1141a79:src/io/smart_ptr.rs:1170-1179`, `Rc<T>` at `1141a79:src/io/smart_ptr.rs:1238-1247`, through `bool` at `1141a79:src/io/smart_ptr.rs:1364-1378`) while keeping `#[inline]` on `deserialize`, and added `vec_vec_u32_1000x100_ser` in `benches/s23_serialization_bench.rs` (`1141a79:benches/s23_serialization_bench.rs:1-85`), restoring `Vec<Vec<u32>>::serialize` (1,000 × 100) from `100.9 ns/inner Vec` at `ee8b29a` to `73.79 ns/inner Vec` (vs `84.7 ns` at `00652d2`). See `docs/review/io.md` §12.3.

### 8.2 `c87db71` — `S26-R2` & `S26-R3` (`src/thread/instance_tls.rs`)
- **`S26-R2` (MEDIUM — `InstanceTls::get` / `get_value` replaced or erased the stored value on re-entrant or panicking `T::clone()`)**:
  - **Root cause**: `b74f6fe` implemented borrow-free `T::clone()` in `InstanceTls::get` (`b74f6fe:src/thread/instance_tls.rs:242-302`) and `InstanceTls::get_value` (`b74f6fe:src/thread/instance_tls.rs:306-345`) by taking `slot.take()` out of the `TlsCell<T>` before calling `val.clone()` and writing `(generation, val)` back into `slot` only if `slot.is_none()` afterwards. Consequently:
    1. If `T::clone()` re-entrantly called `get()` on the **same** `InstanceTls<T>` instance, the inner `get()` saw `None`, initialized the cell with `T::default()`, and the outer `get()` saw `slot.is_some()` and dropped the original `val` (silently replacing `5` with `0`).
    2. If `T::clone()` re-entrantly called `get_value()` or `try_get()` on the same instance, it observed `None` instead of `Some(val)`.
    3. If `T::clone()` panicked while `val` had been taken out of `slot`, the unwind skipped the restore step and permanently erased the stored value (`None`).
  - **Fix in `c87db71:src/thread/instance_tls.rs`**:
    1. Replaced `type TlsCell<T>` with `struct TlsCell<T>` (`c87db71:src/thread/instance_tls.rs:43-46`) holding `readers: std::cell::Cell<u32>` and `value: UnsafeCell<Option<(u32, T)>>`, and added `deferred: UnsafeCell<Vec<DeferredCellWrite<T>>>` (`DeferredCellWrite` at `c87db71:src/thread/instance_tls.rs:58-58`) to `TlsMatrix` (`c87db71:src/thread/instance_tls.rs:62-65`).
    2. Added `CellReadGuard` (`c87db71:src/thread/instance_tls.rs:67-70`, `Drop` at `c87db71:src/thread/instance_tls.rs:72-85`) and `clone_cell_in_place` (`c87db71:src/thread/instance_tls.rs:90-117`). During `InstanceTls::get` (`c87db71:src/thread/instance_tls.rs:312-369`) and `InstanceTls::get_value` (`c87db71:src/thread/instance_tls.rs:373-399`), `clone_cell_in_place` increments `cell.readers` and clones `val` in place via `&T` after releasing the `TLS_MATRICES` `RefCell` borrow.
    3. Because the stored value remains in `cell.value` throughout `T::clone()`, a panicking `T::clone()` leaves the original value untouched while `CellReadGuard::drop` decrements `cell.readers`, and any re-entrant `get()`, `get_value()`, or `try_get()` on the same instance observes the current stored value (`Some(5)`). If `T::clone()` re-entrantly calls `set()` or `remove()` on the same cell while `cell.readers > 0`, `TlsMatrix::defer_write` (`c87db71:src/thread/instance_tls.rs:629-642`) queues the deferred write in `TlsMatrix::deferred` (with `remove()` returning `None` immediately while `readers > 0`), and `CellReadGuard::drop` applies it via `TlsMatrix::apply_deferred` (`c87db71:src/thread/instance_tls.rs:646-661`) when `readers` reaches `0`.
  - **Regression tests**:
    - `test_s26_r2_reentrant_same_instance_get_and_get_value_in_clone` (`c87db71:src/thread/instance_tls.rs:1358-1432`)
    - `test_s26_r2_panicking_clone_preserves_stored_value` (`c87db71:src/thread/instance_tls.rs:1435-1467`)
- **`S26-R3` (MEDIUM — Hold 65,536 live instances simultaneously in `test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak`)**:
  - Updated `test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak` (`c87db71:src/thread/instance_tls.rs:1157-1227`) to keep a `Vec<InstanceTls<CountedA1>>` of `65,536` live instances simultaneously (filling all `256 x 256` slots of the default matrix and asserting that the `65,537`th `InstanceTls::<CountedA1>::new()` returns `Err`), then drop all `65,536` instances and allocate `4,464` recycled instances (`70,000` total creations).

RED output (`test_s26_r2_reentrant_same_instance_get_and_get_value_in_clone` and `test_s26_r2_panicking_clone_preserves_stored_value` spliced onto `ee8b29a:src/thread/instance_tls.rs` before the closing `}` at line `1212`):
```text
---- thread::instance_tls::tests::test_s26_r2_panicking_clone_preserves_stored_value stdout ----
thread 'thread::instance_tls::tests::test_s26_r2_panicking_clone_preserves_stored_value' (2449351) panicked at src/thread/instance_tls.rs:1301:21:
simulated panic in FlakyClone::clone
thread 'thread::instance_tls::tests::test_s26_r2_panicking_clone_preserves_stored_value' (2449351) panicked at src/thread/instance_tls.rs:1314:9:
assertion failed: res_get_val.is_err()

---- thread::instance_tls::tests::test_s26_r2_reentrant_same_instance_get_and_get_value_in_clone stdout ----
thread 'thread::instance_tls::tests::test_s26_r2_reentrant_same_instance_get_and_get_value_in_clone' (2449352) panicked at src/thread/instance_tls.rs:1230:21:
assertion `left == right` failed: re-entrant get() on the same InstanceTls must read the stored value
  left: 0
 right: 5
```

### 8.3 `c64babc` — `S26-R3` & `S26-R5` (`src/fsa/cspp_trie_concurrent.rs`, `src/concurrency/work_stealing.rs`, `src/concurrency/pipeline/mod.rs`)
- **`S26-R3` (MEDIUM — Make `S25-R1`, `S25-R2`, and `S25-R6` regression tests fail on `00652d2`)**:
  1. **`S25-R1` tests (`c64babc:src/fsa/cspp_trie_concurrent.rs:3110-3185`)**: Changed `test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` (`c64babc:src/fsa/cspp_trie_concurrent.rs:3110-3143`) and `test_s25_r1_lookup_multi_insert_near_pool_end` (`c64babc:src/fsa/cspp_trie_concurrent.rs:3146-3185`) from `valsize = 4` (which caused `ConcurrentCsppTrie::with_capacity` to pad `pool_cap` by `capacity / 2 + 1024`, keeping nodes far from `pool.data.len()`) to `valsize = 0` across capacities `258..338` and `262..=330` (wrapping `insert` in `catch_unwind` for capacities `< 269` / `< 277` across `0..32u32`, which in `4929ccc:src/fsa/cspp_trie_concurrent.rs:3110-3169` was narrowed to capacities `269..338` and `277..=330` with `2` keys `0..2u32` in `test_s25_r1_lookup_multi_insert_near_pool_end` so `trie.insert` succeeds without `catch_unwind`). Both tests fail with `index out of bounds` on `00652d2` (verbatim RED output below).
  2. **`S25-R2` test (`c64babc:src/concurrency/work_stealing.rs:1165-1264`)**: Changed `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` from calling `crate::init()` to calling `WorkStealingExecutor::init(ConcurrencyConfig { max_fibers: 2, queue_size: 64 })` on `rt1`, dropping `rt1`, and verifying that both `WorkStealingExecutor::spawn` and `WorkStealingExecutor::global().unwrap().submit_closure(...)` complete on `rt2`.
  3. **`S25-R6` test part 2 (`c64babc:src/concurrency/work_stealing.rs:1267-1353`)**: Updated part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` to use a 4-worker `WorkStealingExecutor::new(4, 8)` with 4 `std::thread::spawn` submitters racing against `exec.shutdown().await` (subsequently tightened in `4929ccc:src/concurrency/work_stealing.rs:1368-1461`, see §9.1).
- **`S26-R5` (LOW — Global executor post-runtime-drop `submit()`, idle worker CPU polling, `lookup_inner` duplicate `bytes_eq_relaxed`, and `stage_timeout` documentation)**:
  1. **Park idle workers on `tokio::sync::Notify` and respawn global workers on runtime drop (`c64babc:src/concurrency/work_stealing.rs`)**: Added `work_notify: Arc<tokio::sync::Notify>` to `WorkStealingExecutor` (`c64babc:src/concurrency/work_stealing.rs:367-375`) so `worker_loop` (`c64babc:src/concurrency/work_stealing.rs:705-795`) parks on `work_notify.notified().await` after `MAX_YIELD = 4` yields instead of polling in a `1 ms` sleep loop, and added `ensure_workers_running` (`c64babc:src/concurrency/work_stealing.rs:489-532`) called by `submit` (`c64babc:src/concurrency/work_stealing.rs:535-580`) to respawn finished worker tasks onto the current Tokio `Handle` (`try_current()`) when a global executor's creation runtime has been dropped.
  2. **Deduplicate `bytes_eq_relaxed` in `ConcurrentCsppTrie::lookup_inner` (`c64babc:src/fsa/cspp_trie_concurrent.rs:1094-1153`)**: Removed the duplicate `if !self.inner.pool.bytes_eq_relaxed(zpath_slot, &key[pos..end])` check inside `lookup_inner` so each node with `zlen > 0` compares its `zpath` slice once.
  3. **Document post-hoc synchronous `stage_timeout` behavior (`c64babc:src/concurrency/pipeline/mod.rs:131-149, 228-255`)**: Updated the doc comments on `PipelineConfig` and `Pipeline::execute_single` to state that futures which yield are cancelled at the `.await` point on timeout, whereas synchronous `PipelineStage::process` implementations that block without yielding cannot be preempted mid-call and have `elapsed > stage_timeout` checked immediately after `process` returns.

RED output (`test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` and `test_s25_r1_lookup_multi_insert_near_pool_end` from `4929ccc` spliced into `mod tests` in `00652d2:src/fsa/cspp_trie_concurrent.rs` before its closing `}` at line `2979`; line `3018` closes top-level `test_concurrent_cspp_trie_deletion_support`):
```text
---- fsa::cspp_trie_concurrent::tests::test_s25_r1_lookup_multi_insert_near_pool_end stdout ----
thread 'fsa::cspp_trie_concurrent::tests::test_s25_r1_lookup_multi_insert_near_pool_end' (2453182) panicked at src/fsa/cspp_trie_concurrent.rs:150:9:
index out of bounds: the len is 277 but the index is 277

---- fsa::cspp_trie_concurrent::tests::test_s25_r1_lookup_zpath_near_pool_end_does_not_overread stdout ----
thread 'fsa::cspp_trie_concurrent::tests::test_s25_r1_lookup_zpath_near_pool_end_does_not_overread' (2453183) panicked at src/fsa/cspp_trie_concurrent.rs:150:9:
index out of bounds: the len is 269 but the index is 269
```

---

## 9. Stage 11 Re-Review Round 3 Remediation (`S27-R1..R5`: `4929ccc..6e7fee4`)

### 9.1 `4929ccc` — `S27-R2`, `S27-R3`, `S27-R4`, `S27-R5` (`src/concurrency/{mod,work_stealing,fiber_pool,parallel_trie}.rs`, `src/concurrency/pipeline/{tests,integration_tests}.rs`, `src/thread/instance_tls.rs`, `src/io/smart_ptr.rs`, `src/fsa/cspp_trie_concurrent.rs`)

- **`S27-R2` (MEDIUM — `WorkStealingExecutor::submit()` returned `Ok(())` for tasks that never ran if worker futures were dropped before first poll)**:
  - **Root cause**: In `c64babc:src/concurrency/work_stealing.rs:713-723`, `ActiveWorkerGuard` was constructed at the start of `worker_loop` rather than before `tokio::spawn` / `rt_handle.spawn` (`c64babc:src/concurrency/work_stealing.rs:451-532`). If a Tokio runtime (`Builder::new_current_thread()` or `Builder::new_multi_thread()`) spawned worker futures in `WorkStealingExecutor::new` or `ensure_workers_running` and was dropped before those futures were polled even once, `worker_loop` was dropped before entering its body—leaving `stats.active_workers` non-zero and `worker.handle` holding a finished `JoinHandle`, so `ensure_workers_running` skipped respawning when called from a plain thread or raced with a concurrently dropping runtime.
  - **Fix in `4929ccc:src/concurrency/work_stealing.rs`**:
    1. Changed `WorkerThread` (`4929ccc:src/concurrency/work_stealing.rs:387-391`) `handle` to `Arc<Mutex<Option<JoinHandle<()>>>>`.
    2. Extended `ActiveWorkerGuard` (`4929ccc:src/concurrency/work_stealing.rs:396-401`, `Drop` at `4929ccc:src/concurrency/work_stealing.rs:404-412`) to hold `handle: Arc<Mutex<Option<JoinHandle<()>>>>`, constructed `worker_guard` **before** `tokio::spawn` in `WorkStealingExecutor::new` (`4929ccc:src/concurrency/work_stealing.rs:452-531`) and **before** `rt_handle.spawn` in `WorkStealingExecutor::ensure_workers_running` (`4929ccc:src/concurrency/work_stealing.rs:535-585`), and moved `worker_guard` into the spawned closure so dropping an unpolled worker future immediately clears `*handle = None` and decrements `active_workers` under the mutex lock.
    3. Updated `notify_if_parked` (`4929ccc:src/concurrency/work_stealing.rs:588-600`) to re-check `self.stats.active_workers.load(Ordering::Acquire) < self.workers.len() && !self.shutdown.load(Ordering::Acquire)` after pushing a task and call `self.ensure_workers_running()` if a concurrent runtime drop terminated a worker during task enqueue.
    4. Updated `Drop` for `WorkStealingExecutor` (`4929ccc:src/concurrency/work_stealing.rs:920-933`) to take `slot.take()` before calling `handle.abort()`.
  - **Regression test**: Expanded `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` (`4929ccc:src/concurrency/work_stealing.rs:1226-1365`) to test: (1) `WorkStealingExecutor::init` on a `new_current_thread()` runtime dropped without polling worker tasks (`active_workers` drops to `0`), (2) `submit_closure` from a plain OS thread with no active Tokio runtime returning `Err` when `active_workers == 0`, and (3) `unpolled_ct_exec` created on a dropped `new_current_thread()` runtime plus `8` executors (`for _ in 0..8`) created on dropped `new_multi_thread()` runtimes before calling `submit_closure` on `rt2` and verifying that `rt2` respawns workers and executes all submitted closures.

RED output (`test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` from `4929ccc` spliced onto `c64babc:src/concurrency/work_stealing.rs` at line `1164`):
```text
---- concurrency::work_stealing::tests::test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers stdout ----
thread 'concurrency::work_stealing::tests::test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers' (2463191) panicked at src/concurrency/work_stealing.rs:1179:9:
assertion `left == right` failed: dropping a runtime whose workers were never polled must decrement active_workers to 0
  left: 2
 right: 0
```

- **`S27-R3` (MEDIUM — 8-thread submit+drain throughput regression from unconditional `work_notify.notify_waiters()`)**:
  - **Root cause**: `c64babc:src/concurrency/work_stealing.rs:562-575` called `self.work_notify.notify_waiters()` unconditionally on every `submit()`, contending on `Notify`'s internal waiter list lock across 8 concurrent submitter threads even when all workers were actively spinning/polling.
  - **Fix in `4929ccc:src/concurrency/work_stealing.rs`**:
    1. Added `parked_workers: AtomicUsize` to `ExecutorStatsInner` (`4929ccc:src/concurrency/work_stealing.rs:377-384`) and `ParkedWorkerGuard` (`4929ccc:src/concurrency/work_stealing.rs:417-419`, `Drop` at `4929ccc:src/concurrency/work_stealing.rs:422-425`).
    2. In `worker_loop` (`4929ccc:src/concurrency/work_stealing.rs:773-856`), the worker increments `stats.parked_workers.fetch_add(1, Ordering::SeqCst)`, creates `ParkedWorkerGuard`, loads `shutdown.load(Ordering::Acquire)`, and calls `Self::find_task(my_queue, &other_queues, &global_queue, &stats, true)`, awaiting `notified.await` only if `find_task` returns `None` and `shutting_down_now` is `false`.
    3. In `notify_if_parked` (`4929ccc:src/concurrency/work_stealing.rs:588-600`), `self.work_notify.notify_waiters()` is called only when `self.stats.parked_workers.load(Ordering::SeqCst) > 0`, recovering 8-thread submit+drain (`400k` tasks) throughput (measured at `f37cf61` as `313.8 ns/task` via `bench_s29_r1_executor_timings`).

- **`S27-R4` (MEDIUM — Part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` passed with the `S25-R6` local-queue shutdown check reverted)**:
  - **Root cause**: In `c64babc:src/concurrency/work_stealing.rs:1296-1352`, part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` slept `40..220 µs` before calling `exec.shutdown().await` on a 4-worker executor with 4 submitters (`512` tasks each), allowing the submitters to finish or hit `Full` before `shutdown()` set `self.shutdown = true`.
  - **Fix in `4929ccc:src/concurrency/work_stealing.rs:1368-1461`**: Rewrote part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` to run `256` iterations (`for it in 0..256usize`) on `WorkStealingExecutor::new(2, 4)` with 8 OS-thread submitters synchronized via `ready: AtomicUsize` + `go: AtomicBool`, waiting until `accepted.load(Ordering::Acquire) < 16 + (it % 64)` and spinning `for _ in 0..((it % 8) * 64)` before calling `exec.shutdown().await` without `tokio::time::sleep` (failing in ~60% of runs when `try_push_local_unless_shutdown`'s shutdown check is reverted).

RED output (`test_s25_r6_is_idle_ordering_and_submit_shutdown_race` on `4929ccc` with the `shutdown.load` check in `try_push_local_unless_shutdown` reverted):
```text
---- concurrency::work_stealing::tests::test_s25_r6_is_idle_ordering_and_submit_shutdown_race stdout ----
thread 'concurrency::work_stealing::tests::test_s25_r6_is_idle_ordering_and_submit_shutdown_race' (2461819) panicked at src/concurrency/work_stealing.rs:1453:13:
assertion `left == right` failed: every task accepted before shutdown() must execute (iter 54)
  left: 86
 right: 83
```

- **`S27-R5` (LOW — `InstanceTls` re-entrant `set`/`remove` docs and `readers == 0` guard, smart-pointer `#[inline]` restoration, `S25-R1` `catch_unwind` removal, and multi-thread `#[tokio::test]` attributes)**:
  1. **`InstanceTls` docs, `get` `readers == 0` check, and `CleanupHandle` `drop` (`4929ccc:src/thread/instance_tls.rs`)**:
     - Documented on `InstanceTls::get` (`4929ccc:src/thread/instance_tls.rs:310-316`), `InstanceTls::get_value` (`4929ccc:src/thread/instance_tls.rs:382-388`), `InstanceTls::set` (`4929ccc:src/thread/instance_tls.rs:425-430`), and `InstanceTls::remove` (`4929ccc:src/thread/instance_tls.rs:455-460`) that `set`/`remove` calls from inside `T::clone()` on the same instance and thread are deferred until the active `clone()` completes (and `remove()` returns `None` immediately while `readers > 0`).
     - Checked `(*cell_ptr).readers.get() == 0` before forming `&mut *(*cell_ptr).value.get()` in the post-`T::default()` store path of `InstanceTls::get` (`4929ccc:src/thread/instance_tls.rs:359-375`) so the code matches its `// SAFETY:` comment even if a custom `T::default()` or `T::clone()` re-entrantly borrows the cell.
     - Removed the unreachable `defer_write` branch from `CleanupHandle` `drop` (`4929ccc:src/thread/instance_tls.rs:709-757`), since `CleanupHandle` `drop` only runs when the last `Arc<CleanupHandle>` (and thus the last `InstanceTls`) is dropped while `get`/`get_value` borrow `&self`.
  2. **Smart-pointer `SerializableType::serialize` `#[inline]` (`4929ccc:src/io/smart_ptr.rs:1238-1300`)**: Restored `#[inline]` on `SerializableType::serialize` for `Rc<T>`, `Arc<T>`, `RcWeak<T>`, `ArcWeak<T>`, and `Box<T>` (see `docs/review/io.md` §12.4).
  3. **`S25-R1` test `catch_unwind` removal (`4929ccc:src/fsa/cspp_trie_concurrent.rs:3110-3169`)**: Started the slot loops at `269..338` in `test_s25_r1_lookup_zpath_near_pool_end_does_not_overread` (`4929ccc:src/fsa/cspp_trie_concurrent.rs:3110-3137`) and `277..=330` in `test_s25_r1_lookup_multi_insert_near_pool_end` (`4929ccc:src/fsa/cspp_trie_concurrent.rs:3140-3169`, inserting `2` keys `0..2u32` per capacity) and unwrapped `trie.insert(...)` directly without `catch_unwind`.
  4. **Multi-thread Tokio runtime across all `src/concurrency/` tests (`4929ccc`)**: Upgraded all `#[tokio::test]` attributes in `src/concurrency/mod.rs` (`4929ccc:src/concurrency/mod.rs:107-135`), `src/concurrency/fiber_pool.rs` (`4929ccc:src/concurrency/fiber_pool.rs:412-669`, synchronizing `test_fiber_pool_queue_capacity_limit` at `4929ccc:src/concurrency/fiber_pool.rs:505-563` with `tokio::sync::Notify`), `src/concurrency/parallel_trie.rs` (`4929ccc:src/concurrency/parallel_trie.rs:514-805`), `src/concurrency/work_stealing.rs` (`4929ccc:src/concurrency/work_stealing.rs:1007-1461`), `src/concurrency/pipeline/tests.rs` (`4929ccc:src/concurrency/pipeline/tests.rs:64-796`), and `src/concurrency/pipeline/integration_tests.rs` (`4929ccc:src/concurrency/pipeline/integration_tests.rs:28-796`) to `#[tokio::test(flavor = "multi_thread", ...)]`.

### 9.2 Sanitizer & Gate Verification (`S27`)
- `touch src/lib.rs && make sanity`: `lib_debug = 3185`, `lib_release = 3202`, `doctests = 229`, `compile_fail = 8`, `all_targets = 3847`, `max_undocumented = 4` (`thread 67/67 = 100.0%` at `b18bad3`, `59/59 = 100.0%` at `c87db71`, `concurrency 0/0 = 100.0%`, `fsa 99/99 = 100.0%`), `max_markers = 27` (`0` in `concurrency`), `max_expects = 82` (`0` in `thread` and `concurrency`), `line_count` (`0` files `> 800` lines in `src/concurrency/pipeline/`).
- `make miri_thread`: `30 passed; 0 failed; 1 ignored` in both SIMD (`-C target-cpu=haswell`) and scalar (`MIRI_SCALAR=1`) builds.
- `make miri_cspp`: `17 passed; 0 failed; 0 ignored` in both SIMD and scalar builds.
- `make tsan_concurrency`: `31 passed; 0 failed` (`thread::`) + `85 passed; 0 failed` (`concurrency::`) under `-Zsanitizer=thread`.
- `make tsan_cspp`: `17 passed; 0 failed` (`fsa::cspp_trie_concurrent`) under `-Zsanitizer=thread`.
- `make tsan_pool`: `33 passed; 0 failed` (`memory::lockfree`) under `-Zsanitizer=thread`.

---

## 10. Stage 11 Re-Review Round 4 Remediation (`S28-R1..R3`: `b18bad3..`)

### 10.1 `b18bad3` — `S28-R1` & `S28-R3` (`src/concurrency/work_stealing.rs`, `src/thread/instance_tls.rs`)

- **`S28-R1` (HIGH — Self-deadlock in `WorkStealingExecutor::new` and `ensure_workers_running` when Tokio synchronously drops the worker future during `spawn`)**:
  - **Root cause**: In `4929ccc:src/concurrency/work_stealing.rs:495-531` (`WorkStealingExecutor::new`) and `4929ccc:src/concurrency/work_stealing.rs:549-583` (`WorkStealingExecutor::ensure_workers_running`), the thread locked `let mut slot = worker.handle.lock()...`, constructed `ActiveWorkerGuard` holding a clone of `Arc<Mutex<Option<JoinHandle<()>>>>`, and called `tokio::spawn` / `rt_handle.spawn` while `slot` was still held. Whenever `tokio::spawn` panicked before creating a task (when called with no active runtime) or `rt_handle.spawn` synchronously dropped the future because the target Tokio runtime had already shut down (or was shutting down while a task's `Drop` called `submit()`), `ActiveWorkerGuard::drop` (`4929ccc:src/concurrency/work_stealing.rs:404-412`) tried to lock `self.handle.lock()` on the same `std::sync::Mutex` already held by the current frame, deadlocking the thread on itself.
  - **Fix in `b18bad3:src/concurrency/work_stealing.rs`**:
    1. Added lock-free per-worker state `state: Arc<AtomicU8>` (`WORKER_DEAD = 0`, `WORKER_SPAWNING = 1`, `WORKER_LIVE = 2` at `b18bad3:src/concurrency/work_stealing.rs:366-368`) to `WorkerThread` (`b18bad3:src/concurrency/work_stealing.rs:401-406`), changed `WorkerThread::handle` back to `Mutex<Option<JoinHandle<()>>>`, and removed `handle` from `ActiveWorkerGuard` (`b18bad3:src/concurrency/work_stealing.rs:412-418`).
    2. `ActiveWorkerGuard::drop` (`b18bad3:src/concurrency/work_stealing.rs:421-445`) never locks `worker.handle` or `respawn_lock`: it atomically decrements `self.stats.active_workers.fetch_sub(1, Ordering::AcqRel)` and swaps `self.state.swap(WORKER_DEAD, Ordering::AcqRel)`.
    3. `WorkStealingExecutor::new` (`b18bad3:src/concurrency/work_stealing.rs:484-582`) and `WorkStealingExecutor::ensure_workers_running` (`b18bad3:src/concurrency/work_stealing.rs:586-678`) never hold `worker.handle.lock()` across `tokio::spawn` or `rt_handle.spawn`: they transition `worker.state` to `WORKER_SPAWNING`, increment `active_workers`, construct `ActiveWorkerGuard`, call `spawn`, and only lock `worker.handle` to store `Some(handle)` if `worker.state.compare_exchange(WORKER_SPAWNING, WORKER_LIVE, Ordering::AcqRel, Ordering::Acquire)` succeeds (or `worker_loop` at `b18bad3:src/concurrency/work_stealing.rs:894-908` already transitioned `state` to `WORKER_LIVE`). If the runtime was already closed and dropped the future synchronously inside `spawn`, `ActiveWorkerGuard::drop` has already set `worker.state = WORKER_DEAD` and decremented `active_workers` without locking `worker.handle`, and `ensure_workers_running` returns `Err(ZiporaError::InvalidState)`.
  - **Regression test**: Expanded `test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` (`b18bad3:src/concurrency/work_stealing.rs:1353-1557`) with a 5-second watchdog thread (`b18bad3:src/concurrency/work_stealing.rs:1404-1460`) testing: (D3) `WorkStealingExecutor::new` outside any Tokio runtime panics cleanly under `catch_unwind` instead of hanging, (D2) `WorkStealingExecutor::new` with a closed runtime's `Handle` entered returns `Ok` with `active_workers == 0`, (D1) `submit_closure` with a closed runtime's `Handle` entered returns `Err` without hanging, and (D4) a `SubmitOnDrop` task whose `Drop` calls `submit_closure` while its `multi_thread` runtime shuts down completes all 8 iterations without hanging.

RED output (`test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers` from `b18bad3` spliced onto `4929ccc:src/concurrency/work_stealing.rs` at line `1226`):
```text
---- concurrency::work_stealing::tests::test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers stdout ----
thread 'concurrency::work_stealing::tests::test_s25_r2_global_init_dropped_runtime_and_nested_await_fibers' (4190742) panicked at src/concurrency/work_stealing.rs:1331:18:
S28-R1 self-deadlock when Tokio synchronously drops worker future during spawn: Timeout
```

- **`S28-R3` (LOW — Shutdown-window strand on concurrent runtime drop, `submit()` `# Errors` docs, `InstanceTls` comments, and `S25-R1` key-count disclosure)**:
  1. **Eliminate shutdown-window strand on concurrent runtime drop (`b18bad3:src/concurrency/work_stealing.rs`)**: Added `fallback_runtime: Mutex<Option<tokio::runtime::Handle>>` and `self_weak: Weak<Self>` to `WorkStealingExecutor` (`b18bad3:src/concurrency/work_stealing.rs:377-389`), recorded the active Tokio `Handle` in `submit` (`b18bad3:src/concurrency/work_stealing.rs:711-721`) via thread-local `LAST_RECORDED_EXECUTOR_ID`, and in `ActiveWorkerGuard::drop` (`b18bad3:src/concurrency/work_stealing.rs:426-442`) spawned a respawn/notify task onto `fallback_runtime` whenever a `WORKER_LIVE` worker is dropped while `!self.shutdown.load(Ordering::Acquire)`.
  2. **Document `# Errors` on `submit` (`b18bad3:src/concurrency/work_stealing.rs:695-705`) and `submit_closure` (`b18bad3:src/concurrency/work_stealing.rs:765-772`)**: Documented the [`ZiporaError::InvalidState`] and [`ZiporaError::Configuration`] error conditions on `WorkStealingExecutor::submit` and `WorkStealingExecutor::submit_closure`.
  3. **Fix 3 stale/imprecise `InstanceTls` comments (`b18bad3:src/thread/instance_tls.rs:39-42, 312-315, 384-386, 720-722`)**: Updated the doc comment on `TlsCell` (`b18bad3:src/thread/instance_tls.rs:39-42`) to state that `value` is only accessed via shared references while `readers > 0` and re-entrant `set`/`remove` updates are recorded in `TlsMatrix::deferred`; updated `InstanceTls::get` (`b18bad3:src/thread/instance_tls.rs:312-315`) and `InstanceTls::get_value` (`b18bad3:src/thread/instance_tls.rs:384-386`) docstrings to say `until the outermost active clone() on the cell completes (readers == 0)`; and updated the `// SAFETY:` comment in `CleanupHandle` `drop` (`b18bad3:src/thread/instance_tls.rs:720-722`) to match the `(*cell_ptr).readers.get() == 0` guard.
  4. **Disclose `test_s25_r1_lookup_multi_insert_near_pool_end` key count (`S28-R3`)**: Documented in §8.3 and §9.1 above that `test_s25_r1_lookup_multi_insert_near_pool_end` (`4929ccc:src/fsa/cspp_trie_concurrent.rs:3140-3169`) inserts `2` keys (`0..2u32`) per capacity across `277..=330` (reduced from up to `32` keys `0..32u32` in `c64babc:src/fsa/cspp_trie_concurrent.rs:3146-3185`).

### 10.2 `S28-R2` — Ledger & `scripts/audit_ledger.py` Generalization (`scripts/audit_ledger.py`, `scripts/tests/audit_ledger_adversarial.md`, `docs/review/thread_and_concurrency.md`, `docs/review/io.md`)
- Removed all hardcoded test literals/numbers and non-repo token whitelists from `scripts/audit_ledger.py`, restricted the repository index strictly to committed git state (never indexing dirty working-tree or staged edits), and implemented general checks for:
  1. Unbackticked, italicized, escaped-underscore (`\_`), bold, and hyphenated code identifiers in prose.
  2. AST `Type::member` and standard-library method validation across prose and ```rust code fences, plus function-body verification when a line pairs `Type::member` with a repository function name.
  3. Dynamic `Makefile` target discovery and test-count validation, historical `gate_baseline.txt` totals, and historical per-module `unsafe` block counts.
  4. Percentage arithmetic verification (`±P% (A unit vs B unit)` within 1.0 percentage point) and prose decimal measurement (`ns`/`µs`/`ms`/`MiB`/`GiB`) backing against tables or fenced benchmark blocks in the ledger.
  5. Panic thread test existence and assertion-message token verification against the cited test function body, plus commit-horizon line bounds on fenced compiler/panic pastes.
  6. Per-clause `Before`/`After` and single-citation `Ordering::*` verification (including `compare_exchange` success ordering and named atomic field ordering), loop iteration bounds (`N iterations`), and `A before B` statement order in cited snippets.
  7. Citation item-span start/end boundary checks, sub-span prose claim checks, standalone commit SHA file/message checks, and un-cited single-file identifier membership checks.
- Added Round 4 (30 new cases, 79 total adversarial cases across Rounds 1–4) to `scripts/tests/audit_ledger_adversarial.md` and verified `0` errors on `docs/review/io.md` and `docs/review/thread_and_concurrency.md` and `102` errors across all `79/79` adversarial cases in `scripts/tests/audit_ledger_adversarial.md`.

---

## 11. Stage 11 Final Fix Round Remediation (`S29-R1..R3`: `c0a56de..fd1ec65`)

### 11.1 `c0a56de` — `S29-R1` (`src/thread/mod.rs`, `src/concurrency/work_stealing.rs`, `scripts/verify_s29_r1_red.sh`, `scripts/gate_baseline.txt`)

- **`S29-R1` (MEDIUM — `WorkStealingExecutor::submit` cached `LAST_RECORDED_EXECUTOR_ID` even when `Handle::try_current()` failed and never refreshed on runtime change)**:
  - **Root cause**: `b18bad3:src/concurrency/work_stealing.rs:711-721` stored a single `executor_id: usize` in thread-local `LAST_RECORDED_EXECUTOR_ID` (`b18bad3:src/concurrency/work_stealing.rs:373`) unconditionally on the first `submit()` to an executor on a thread—even when `tokio::runtime::Handle::try_current()` returned `Err` (mode `b`: first submit from a plain OS thread, subsequent submits from a live Tokio runtime) or when the first submit came from a temporary runtime that was later dropped (mode `c`). When the worker runtime subsequently dropped with tasks queued in the backlog, `ActiveWorkerGuard::drop` found `None` or a dead `Handle` in `fallback_runtime` and left the backlog stranded.
  - **Fix in `c0a56de`**:
    1. Added `pub(crate) fn tokio_runtime_handle_id(handle: &tokio::runtime::Handle) -> usize` in `c0a56de:src/thread/mod.rs:65-78` with compile-time size/alignment assertions and a `// SAFETY:` block.
    2. Replaced `LAST_RECORDED_EXECUTOR_ID: Cell<usize>` with `LAST_RECORDED_FALLBACK: Cell<Option<(usize, usize)>>` (`fd1ec65:src/concurrency/work_stealing.rs:373-374`).
    3. Updated `WorkStealingExecutor::submit` (`fd1ec65:src/concurrency/work_stealing.rs:710-762`) to inspect `tokio::runtime::Handle::try_current()` first, compute `key = (self.executor_id, crate::thread::tokio_runtime_handle_id(&rt_handle))`, perform a single thread-local compare `last.get() != Some(key)`, and set `last.set(Some(key))` + store `Some(rt_handle)` in `self.fallback_runtime` only when a live handle is present and `(executor_id, runtime_id)` changed.
  - **Regression tests & splice-and-run RED script**:
    - `test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime` (`fd1ec65:src/concurrency/work_stealing.rs:1792-1794`)
    - `test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime` (`fd1ec65:src/concurrency/work_stealing.rs:1797-1799`)
    - `bench_s29_r1_executor_timings` (`fd1ec65:src/concurrency/work_stealing.rs:1803-1886`)
    - Committed splice-and-run script `scripts/verify_s29_r1_red.sh` (`c0a56de:scripts/verify_s29_r1_red.sh:1-45`).

Verbatim RED output from `scripts/verify_s29_r1_red.sh` on base commit:
```text
running 2 tests
test concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime ... FAILED
test concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime ... FAILED

failures:

---- concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime stdout ----

thread '<unnamed>' (974822) panicked at src/concurrency/work_stealing.rs:1762:13:
assertion `left == right` failed: S29-R1 mode 'b': queued backlog stranded after worker runtime drop (executed 0/100)
  left: 0
 right: 100
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime stdout ----

thread '<unnamed>' (974886) panicked at src/concurrency/work_stealing.rs:1762:13:
assertion `left == right` failed: S29-R1 mode 'c': queued backlog stranded after worker runtime drop (executed 0/100)
  left: 0
 right: 100


failures:
    concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_b_plain_then_live_runtime
    concurrency::work_stealing::tests::test_s29_r1_fallback_runtime_mode_c_dropped_temp_then_live_runtime

test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 3180 filtered out; finished in 2.04s

error: test failed, to rerun pass `--lib`
exit_code=101
```

### 11.2 `fd1ec65` — `S29-R3` (`src/concurrency/work_stealing.rs`)

- **Fixes in `fd1ec65:src/concurrency/work_stealing.rs`**:
  1. **Release dead/consumed `fallback_runtime` handle (`fd1ec65:src/concurrency/work_stealing.rs:421-452, 814-836`)**: Changed `ActiveWorkerGuard::drop` from `.clone()` to `.take()` on `*self.fallback_runtime.lock()` and cleared `*self.fallback_runtime.lock() = None` in `WorkStealingExecutor::shutdown`, releasing the stored runtime `Handle` (+4 fds) as soon as fallback fires or the executor shuts down.
  2. **Hold `Weak<WorkStealingExecutor>` in the fallback respawn task (`fd1ec65:src/concurrency/work_stealing.rs:433-450`)**: Dropped the upgraded `Arc<WorkStealingExecutor>` before calling `rt_handle.spawn` and captured `weak_exec: Weak<WorkStealingExecutor>` inside the spawned future so an unpolled fallback task queued on an undriven `current_thread` runtime never keeps `WorkStealingExecutor` alive after the caller drops its `Arc`.
  3. **Yield instead of busy-spinning in `ensure_workers_running` (`fd1ec65:src/concurrency/work_stealing.rs:677-680`)**: Replaced `std::hint::spin_loop()` with `std::thread::yield_now()` in the `!progressed` window while holding `respawn_lock`.
  4. **Document `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` Part 2 single-line revert catch rate (`fd1ec65:src/concurrency/work_stealing.rs:1596-1603`)**: Documented that across `256` iterations on a multi-core Linux host, reverting the shutdown check in `try_push_local` fails in ~50–63% of test invocations per run (~60% observed at `4929ccc`).
- **Before/After Executor Timings & Sanitizer Verification**:
```text
BEFORE (f37cf61): S29-R1 executor bench: mp8_400k=313.8 ns/task
AFTER  (fd1ec65): S29-R1 executor bench: mp8_400k=314.5 ns/task
```
  - `make miri_thread`: `30 passed; 0 failed; 1 ignored` in both SIMD (`-C target-cpu=haswell`) and scalar (`MIRI_SCALAR=1`) builds.
  - `make tsan_concurrency`: `31 passed; 0 failed` (`thread::`) + `87 passed; 0 failed; 1 ignored` (`concurrency::`).


