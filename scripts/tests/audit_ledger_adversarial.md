# Adversarial Ledger Audit Fixtures (S25 + S26)

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
