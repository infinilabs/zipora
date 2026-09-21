# C3 — `src/memory/` review ledger

Stage 4 of the plan (`plan.md` §2, row C3, with the coupled D3 / D8 / D10 rows).

Protocol per `plan.md` §5: RED first (a test that fails for the stated reason, watched
failing), one commit per finding with its regression test, `SAFETY:` on every `unsafe`,
`debug_assert!` is not a guard for `unsafe` reachable from a safe public API.

**Scope.** 16 files, 16,726 lines, 328 `unsafe` sites (the largest remaining subsystem).

| File | Lines | Reviewed |
|---|---|---|
| `bump.rs` | 610 | yes |
| `cache.rs` | 823 | yes |
| `cache_layout.rs` | 920 | yes |
| `five_level_pool.rs` | 1,550 | yes (full) |
| `fixed_capacity_pool.rs` | 938 | yes (full) |
| `hugepage.rs` | 582 | yes |
| `lockfree_pool.rs` | 1,504 | yes (full) |
| `mmap.rs` | 473 | yes |
| `mmap_vec.rs` | 2,318 | partial — C9 owns the zero-copy/mapping invariants |
| `mod.rs` | 210 | yes |
| `pool.rs` | 666 | yes |
| `prefetch.rs` | 925 | partial |
| `secure_pool.rs` | 2,170 | yes (hot paths) |
| `simd_ops.rs` | 1,668 | partial |
| `threadlocal_pool.rs` | 672 | yes (full) |
| `tiered.rs` | 697 | yes |

---

## Findings

### C3.1 — `FixedCapacityMemoryPool::new` overruns its own arena — CRITICAL

**Finding.** `initialize_free_lists` / `initialize_free_lists_internal`
(`fixed_capacity_pool.rs:475`, `:516`) write a 16-byte `BlockHeader` at the start of
every block, at `i * max_block_size` for `i in 0..total_blocks`. Nothing validated
`max_block_size`, so a configuration with blocks narrower than the header made the
write run past the block — and, for the last block, past the end of the arena. This
happens inside `FixedCapacityMemoryPool::new()`, from entirely safe code, before the
caller ever holds a pointer.

Four more unvalidated config fields sat next to it:
`total_blocks == 0` (zero-sized `Layout` handed to `std::alloc::alloc`, which is UB,
plus a divide-by-zero in the statistics path at `:311`/`:346`); a non-power-of-two
`alignment` (invalid `Layout` alignment, and the `& !(alignment - 1)` rounding in
`generate_size_classes` becomes nonsense); an `alignment` below the header's; and a
`max_block_size` that is not a multiple of `alignment` — blocks are carved at
`i * max_block_size`, so the arena's alignment reaches every block only when that
holds, otherwise the pool silently breaks the alignment it was configured with.

**Expected failure.** Heap buffer overflow / dangling-reference UB during construction.

**Repro.**

```rust
FixedCapacityMemoryPool::new(FixedCapacityPoolConfig {
    max_block_size: 8, total_blocks: 2, alignment: 8,
    enable_stats: false, eager_allocation: true, secure_clear: false,
}).unwrap();
```

Under `cargo +nightly miri test` at the parent commit:

```
error: Undefined Behavior: constructing invalid value of type
&mut memory::fixed_capacity_pool::BlockHeader: encountered a dangling reference
(going beyond the bounds of its allocation)
   --> src/memory/fixed_capacity_pool.rs:475:35
    0: FixedCapacityMemoryPool::initialize_free_lists      at :475
    1: FixedCapacityMemoryPool::allocate_backing_memory    at :404
    2: FixedCapacityMemoryPool::new                        at :258
```

**Tests.** `test_config_rejects_block_narrower_than_block_header`,
`test_config_rejects_zero_total_blocks`,
`test_config_rejects_block_size_not_multiple_of_alignment`,
`test_config_rejects_bad_alignment`, `test_all_presets_pass_validation`.

**RED (watched).** With only `Self::validate_config(&config)?;` removed from `new()`:

```
test result: FAILED. 9 passed; 4 failed
  test_config_rejects_block_narrower_than_block_header
      panicked: max_block_size below size_of::<BlockHeader>() must be rejected
  test_config_rejects_zero_total_blocks
      panicked: total_blocks == 0 must be rejected
  test_config_rejects_block_size_not_multiple_of_alignment
      panicked: max_block_size 100 with alignment 64 must be rejected
  test_config_rejects_bad_alignment
      panicked: got: Invalid data: Invalid layout: invalid parameters to
                Layout::from_size_align
```

The fourth is instructive: without the guard a bad alignment surfaced as a generic
`Layout` error from deep inside `allocate_backing_memory`, and `alignment: 2` — below
the header's own alignment — was accepted outright. `total_blocks == 0` also returned
`Ok`, i.e. the zero-sized `alloc` really did happen.

**Fix.** `FixedCapacityMemoryPool::validate_config` rejects all five cases in `new()`
before any memory is touched; the capacity product is `checked_mul`. All five presets
are asserted to pass their own validation.

**Commit.** _pending_

