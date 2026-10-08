# Adversarial Ledger Audit Fixtures (S25 + S26 + S27 + S28)

## Round 1 (S25-R9: 9 planted error classes)

1. Wrong-but-in-range citation, no symbol: the RangeReader constructor is at `00652d2:src/io/range_stream.rs:800`.
2. Wrong-but-in-range citation with symbol prefix: `RangeReader::new` (`00652d2:src/io/range_stream.rs:5`).
3. Symbol named, wrong file in range: `decode_leb128_u128_65` (`00652d2:src/io/range_stream.rs:300`).
4. Non-existent identifier: `EndianReader` and `with_simd` and `chunks_Exact_mut` and `swapped_16`.
5. Wrong type/member pairing: `RangeReader::grow_buffer_to` and `MmapVec::encode_leb128_u128`.
6. Totally fabricated member: `RangeReader::frobnicate_xyz`.
7. Substring trick: `RangeReader::get` (`00652d2:src/io/range_stream.rs:100`).
8. Fake commit table row:
| 1 | `6571c89` | `a0bcdff` | 2026-10-07 | fix(sync): make SharedPool byte access word-atomic, pin cas_weak_u128 regs, and ratchet Stage 11 gates [C10, C10.1, B7, B8, D10] | 999 | +1 / −1 |
9. Fabricated number: 4,242 tests pass, TSan 0 races.

## Round 2 (S26-R1: 13 planted error classes)

1a. Wrong item, attached: `pop_local` (`48438fe:src/concurrency/work_stealing.rs:228-256`) falls back to the steal queue.
1b. Wrong item, unattached lowercase prose: the pop_local fallback lives at `48438fe:src/concurrency/work_stealing.rs:228-256`.
2. Right item, wrong commit: `is_idle` (`b74f6fe:src/concurrency/work_stealing.rs:662-664`) checks `total_queued()` before `active_tasks`.
3. False ordering Before/After: **Before**: `FutexMutex::lock` (`39814be:src/thread/linux_futex.rs:145-158`) used `Ordering::Relaxed` on the CAS success path. **After**: `Ordering::Acquire` (`05d4245:src/thread/linux_futex.rs:158-174`).
4. Invented test count: `make miri_thread`: `412 passed; 0 failed`.
5a. Identifier in tree but not in cited file (attached): `steal_len` (`48438fe:src/thread/linux_futex.rs:100-110`).
5b. Identifier in tree but not in named file (no citation): `src/thread/linux_futex.rs` stores `steal_len` on `FutexMutex`.
6a. Member on a different type: `WorkStealingQueue::is_idle`.
6b. Member on a different type: `FiberPool::steal_len`.
7. Fabricated paste line numbers:
```text
error: invalid operand for instruction
   --> src/thread/atomic_ext.rs:9999:18
```
8. Whitelisted fabricated identifier: `steal_amount` and `current_rx` and `load_order`.
9. Identifier only present in scripts/audit_ledger.py: `multi_spans` and `ast_members` field on `FiberPool`.
10. Unbackticked Type::member: WorkStealingQueue::frobnicate is called by shutdown.

## Round 3 (S27-R1: 27 fool cases F01..F27)

F01 invented ident, plain prose: the executor parks idle workers on WorkerParkingLot until shutdown.
F02 invented ident, bold: **spawn_parked_worker** respawns dead workers onto the current runtime.
F03 real ident, wrong line inside same fn: `ensure_workers_running` rejects shut-down executors (`c64babc:src/concurrency/work_stealing.rs:526-529`).
F04 real ident, wrong sha (symbol absent at sha): `ensure_workers_running` (`c87db71:src/concurrency/work_stealing.rs:489-532`) respawns workers.
F05 wrong Ordering in CAS claim: `LockFreeFreelist::push` uses an `(AcqRel, Acquire)` CAS (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`).
F06 wrong Ordering, Ordering:: form: the `pop_fast_bin` CAS uses `Ordering::SeqCst` on success (`0932ce7:src/memory/lockfree_pool.rs:595-652`).
F07 wrong Miri count in range: `make miri_thread`: `44 passed; 0 failed; 1 ignored`.
F08 wrong unsafe count: `src/fsa/` `unsafe`: `131 -> 87` (`87/87 = 100.0%`).
F09 fabricated ns number: `InstanceTls::get` now takes `0.91 ns` on `u64` (was `2.83 ns`).
F10 fabricated test total under threshold: 3,990 tests pass in the debug gate.
F11 real member on wrong type: `TlsMatrices::defer_write` queues writes.
F12 real member on std-listed wrong type: `Mutex::ensure_workers_running` and `Semaphore::apply_deferred` are called by submit.
F13 range crossing item boundary: `clone_cell_in_place` (`c87db71:src/thread/instance_tls.rs:90-140`).
F14 Before/After reversed, no Ordering: **Before**: `worker_loop` parked on `work_notify` (`c64babc:src/concurrency/work_stealing.rs:705-795`). **After**: `worker_loop` slept 1 ms (`c87db71:src/concurrency/work_stealing.rs:639-710`).
F15 Before/After reversed, different orderings: **Before**: `deallocate_to_fast_bin` used `Ordering::Release` (`0932ce7:src/memory/lockfree_pool.rs:689-745`). **After**: `Ordering::AcqRel` (`6571c89:src/memory/lockfree_pool.rs:689-745`).
F16 commit row wrong numstat (control, expected caught):
| 1 | `c64babc` | `c87db71` | 2026-10-07 | fix(concurrency,fsa): park idle workers on Notify, respawn global workers after runtime drop, tighten S25-R1/R2/R6 tests, and deduplicate lookup_inner zpath compare [S26-R3, S26-R5] | 4 | +231 / −74 |
F17 commit row with wrong sha in a 6-column layout (parent column omitted):
| 2 | `c87db72` | 2026-10-07 | fix(thread): preserve InstanceTls value | 2 | +377 / −120 |
F18 invented ident inside a table cell, unbackticked:
| `src/thread/instance_tls.rs` | Added TlsDeferQueue::flush_all for re-entrant writes |
F19 invented ident in a non-paste code fence:
```rust
fn spawn_parked_worker(notify: &WorkerParkingLot) {}
```
F20 non-existent file, no line: the worker loop lives in `src/concurrency/executor.rs`.
F21 fabricated paste message at a valid line:
```text
thread 'x' panicked at src/thread/instance_tls.rs:1244:9:
assertion failed: fabricated message never present in source
```
F22 fabricated paste line beyond file end but within slack:
```text
   --> src/concurrency/work_stealing.rs:1480:13
```
F23 short attached ident at wrong place: `get` (`c87db71:src/thread/instance_tls.rs:629-642`) clones in place.
F24 unbackticked Type::member on std type: Arc::frobnicate_workers is invoked by shutdown.
F25 prose claim, citation, no candidate symbol: the push CAS is SeqCst (`0932ce7:src/fsa/cspp_trie_concurrent.rs:381-405`).
F26 non-existent file with line (control, expected caught): `worker_loop` (`c64babc:src/concurrency/executor.rs:705-795`).
F27 invented Type::member (control, expected caught): `WorkStealingExecutor::park_all_workers`.

## Round 4 (S28-R2: 30 fool cases G01..G30)

G01 hyphenated invented ident: the ensure-workers-alive path respawns dead workers after a runtime drop.
G02 markdown-italic underscore invented ident: workers park via _park_all_idle_workers_ before shutdown.
G03 escaped-underscore invented ident: workers park via park\_all\_idle\_workers before shutdown.
G04 acronym CamelCase invented type: the AWGuard RAII type clears the slot before spawning.
G05 std type with allow-listed but wrong member: `Mutex::upgrade` re-acquires the worker handle after a runtime drop.
G06 std type, real tokio method but code uses another: `notify_if_parked` wakes one worker via `Notify::notify_one`.
G07 Ordering in prose without citation (code uses SeqCst): `ParkedWorkerGuard` decrements `parked_workers` with `Ordering::Relaxed` on drop.
G08 Ordering in prose with citation whose range contains the wrong ordering elsewhere: `notify_if_parked` loads `parked_workers` with `Acquire` (`4929ccc:src/concurrency/work_stealing.rs:588-600`).
G09 fabricated µs number: `notify_if_parked` adds `0.42 µs` per `submit()` on 8 threads.
G10 fabricated MiB number: the 65,536-instance churn test peaks at `11.5 MiB` RSS.
G11 percentage over-computed: 8-thread submit+drain improved by `+38.5%` (`359.9 ns/task` vs `412.0 ns/task` at `c64babc`).
G12 percentage under-computed: `Vec<Vec<u32>>::serialize` regressed by `+12.0%` (`100.9 ns/inner Vec` vs `84.7 ns/inner Vec`).
G13 real sha in prose that never touched the file: `parked_workers` was added to `ExecutorStatsInner` in `1141a79`.
G14 right sha+file+range, wrong STD-ish type named: `Box<T>` regained `#[inline]` at `4929ccc:src/io/smart_ptr.rs:1238-1248`.
G15 real assertion literal from a different test in a RED paste at a valid line:
```text
thread 'concurrency::work_stealing::tests::test_s25_r6_is_idle_ordering_and_submit_shutdown_race' (1) panicked at src/concurrency/work_stealing.rs:1453:13:
assertion `left == right` failed: dropping a runtime whose workers were never polled must decrement active_workers to 0
```
G16 RED paste with a real line/message but wrong file:
```text
thread 'fsa::cspp_trie_concurrent::tests::test_s25_r1_lookup_multi_insert_near_pool_end' (1) panicked at src/concurrency/fiber_pool.rs:150:9:
index out of bounds: the len is 277 but the index is 277
```
G17 panic line beyond file end within slack (fiber_pool.rs max 670 lines):
```text
thread 'concurrency::fiber_pool::tests::x' (1) panicked at src/concurrency/fiber_pool.rs:684:9:
index out of bounds: the len is 4 but the index is 4
```
G18 fabricated make miri_io count in range: `make miri_io`: `250 passed; 0 failed`.
G19 fabricated count for a target missing from the tool table: `make miri_simd`: `9999 passed; 0 failed`.
G20 fabricated tsan_concurrency count in range: `make tsan_concurrency`: `60 passed; 0 failed` (`concurrency::`).
G21 fabricated debug test total under threshold: 3,180 tests pass in the debug gate.
G22 Before/After, ancestry correct, but Before content contradicts the claim: **Before**: `ActiveWorkerGuard` used `Ordering::Relaxed` in `drop` (`c64babc:src/concurrency/work_stealing.rs:713-723`). **After**: `Ordering::AcqRel` (`4929ccc:src/concurrency/work_stealing.rs:403-413`).
G23 truncated test name that prefixes a real test: the regression is covered by `test_s25_r2_global_init_dropped`.
G24 invented ident in a heading:
#### 9.9 `respawn_dead_workers` fast path
G25 invented ident in a list item in parentheses, unbackticked: - parks workers (see park_idle_worker) before shutdown.
G26 real ident, wrong file, no citation: `src/io/smart_ptr.rs` defines `ParkedWorkerGuard` for the worker loop.
G27 real ident attached to a citation in another fn: - `ParkedWorkerGuard` (`4929ccc:src/concurrency/work_stealing.rs:396-401`).
G28 member on wrong type where both exist, with call parens: `WorkStealingQueue::ensure_workers_running()` respawns workers.
G29 test count claim, wrong number of iterations: part 2 of `test_s25_r6_is_idle_ordering_and_submit_shutdown_race` runs `64` iterations (`4929ccc:src/concurrency/work_stealing.rs:1396-1461`).
G30 plural invented type: both WorkerParkSlots are released.
