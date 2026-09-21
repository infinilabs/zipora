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


---

### C3.2 — every `FiveLevelPool` drop is UB (wrong `Layout`) — CRITICAL

**Finding.** `MemoryChunk::new(capacity, alignment)` allocates with
`Layout::from_size_align(capacity, alignment)`, where `alignment` comes from
`FiveLevelPoolConfig::alignment` — 8 by default, 16 for `performance_optimized()`.
`MemoryChunk::drop` deallocated with `Layout::from_size_align_unchecked(capacity,
align_of::<u8>())`, i.e. **alignment 1**. `GlobalAlloc::dealloc` requires the same
layout that was passed to `alloc`; a differing alignment is undefined behaviour.

The `SAFETY:` comment claimed "data allocated with same layout (capacity,
`align_of::<u8>()`)", which was simply false — `new` never used that layout.

This is not an edge case: it fires on every construction and drop of `NoLockingPool`,
`MutexBasedPool`, `LockFreePool`, `ThreadLocalPool`, `FixedCapacityPool` and
`AdaptiveFiveLevelPool`, under every preset, from safe code.

**Expected failure.** Mismatched-layout deallocation.

**RED (watched).** `cargo +nightly miri test --lib
memory::five_level_pool::tests::test_pool_drop_uses_the_allocation_layout` at the
parent commit, on the plain `FiveLevelPoolConfig::default()` case:

```
error: Undefined Behavior: incorrect layout on deallocation:
alloc644677 has size 1048576 and alignment 8, but gave size 1048576 and alignment 1
   --> src/memory/five_level_pool.rs:255:13
    |
255 |             dealloc(self.data.as_ptr(), layout);
    |
    0: <MemoryChunk as Drop>::drop                  at :255
    1: drop_glue::<MemoryChunk>
    2: drop_glue::<NoLockingPool>
    3: mem::drop::<NoLockingPool>
    4: tests::test_pool_drop_uses_the_allocation_layout
```

**Test.** `test_pool_drop_uses_the_allocation_layout` — constructs and drops a pool
under four configurations (default, `performance_optimized`, `memory_optimized`, and
an explicit `alignment: 64`) and asserts the chunk base honours the configured
alignment. The alignment assertion holds either way; the **drop** is the oracle, and
it only speaks under Miri, hence the new `make miri_pool` target.

**Fix.** `MemoryChunk` stores the `Layout` it allocated with and `Drop` uses it.
The `SAFETY:` comment now names the real invariant.

**Commit.** _pending_

---

### C3.3 — the NUMA pool frees every cached block with a layout it never had — CRITICAL

**Finding.** `NumaMemoryPool::deallocate` (`cache.rs:319-340`) filed a freed block into
one of three size-category `Vec<usize>` caches, keyed on `layout.size()` alone, and
stored **only the address** — discarding the block's real size and its alignment.
`Drop` (`cache.rs:354-388`) then freed every cached pointer with a hardcoded `Layout`:
`(1024, 8)` for "small", `(65536, 16)` for "medium", `(1048576, 32)` for "large".
`GlobalAlloc::dealloc` requires the allocating layout; both the size and the alignment
were wrong. With the system allocator an over-aligned block takes a different path
from an 8-aligned one, so this is heap corruption, not a nit.

Two defects rode along. Nothing ever *read* those caches back — `numa_alloc` goes
straight to `alloc` — so the first 100 frees per class per node were retained forever:
a bounded leak whose only effect was to arm the mis-free. And `allocated_bytes` was
never incremented anywhere in the file, so the `fetch_sub` on the overflow path wrapped
a `usize` from 0; `hit_count`/`miss_count` were likewise never incremented, making
`NumaPoolStats::hit_rate()` permanently `0.0`.

Separately, `numa_alloc_aligned(0, ..)` built a valid zero-sized `Layout` and handed it
to `std::alloc::alloc`, whose contract forbids that. The `// SAFETY: layout valid
(size > 0, ...)` comment asserted an invariant nothing enforced.

**Expected failure.** Mismatched-layout deallocation on `clear_numa_pools()`.

**RED (watched).** Reachable from 100 % safe code:

```rust
init_numa_pools().unwrap();
let p = numa_alloc_aligned(64, 64, 0).unwrap();   // Layout(64, 64)
numa_dealloc(p, 64, 64, 0).unwrap();              // cached, not freed
clear_numa_pools().unwrap();                      // Drop -> dealloc(p, Layout(1024, 8))
```

`MIRIFLAGS=-Zmiri-disable-isolation cargo +nightly miri test` at the parent commit:

```
error: Undefined Behavior: ... occurred here
   --> src/memory/cache.rs:360:17
360 |  dealloc(ptr_addr as *mut u8, Layout::from_size_align(1024, 8) ...)
    0: <NumaMemoryPool as Drop>::drop                     at :360
   10: memory::cache::clear_numa_pools                    at :583
   11: tests::scratch_c33_red_probe
```

> [!NOTE]
> An earlier draft of this entry claimed the existing suite already walked this path
> via `test_numa_alloc_dealloc` + `test_numa_pool_stats`. **That is wrong** and I am
> recording the correction: under `--test-threads=1` the alphabetical ordering runs
> `test_numa_alloc_dealloc` *before* `test_numa_pool_initialization`, so the node pool
> does not exist yet and `numa_dealloc` takes the correct direct-free fallback; and
> `test_numa_pool_stats` never frees. A clean Miri run of `memory::cache::tests` at the
> parent commit passes. The probe above is what actually reproduces it.

**Tests.** `test_numa_dealloc_releases_the_block_with_its_own_layout` (asserts the
now-real `allocated_bytes` accounting, and is the `make miri_pool` oracle for the
release path), `test_numa_alloc_rejects_zero_size`.

**Fix.** The pool holds no block cache: the three `Vec<usize>` fields and the `Drop`
are gone, and `deallocate` releases the block immediately with the caller's layout
(now an `unsafe fn` carrying that contract). `numa_alloc` rejects a zero size and
charges `allocated_bytes`; `numa_dealloc` credits it back with a saturating update.
The module doc now describes what the code does instead of claiming a fix that only
covered the allocation half.

**Public API break.** `NumaPoolStats` loses `hit_count`, `miss_count`, `cached_small`,
`cached_medium`, `cached_large` and the `hit_rate()` / `total_cached()` accessors. With
no cache those five numbers are structurally always zero and `hit_rate()` can only
return `0.0`; five lying fields are worse than none. `allocated_bytes` survives and is
now real.

**Commit.** _pending_

---

### C3.4 — `BumpAllocator` overflow rewinds the cursor and hands out aliasing blocks — CRITICAL

**Finding.** `alloc_bytes` computed `aligned_offset + size` with wrapping arithmetic
(`bump.rs:102-103`). `size` is caller-controlled through `alloc_bytes`, `alloc_slice`
and `BumpVec::new_in`, all safe. In release, a request whose end offset overflows
wraps to a *small* `new_offset`, which passes `new_offset > self.capacity`, so the CAS
**stores a cursor below the current one**. The allocator is rewound and re-issues
addresses that are still live: two `&mut [u8]` over the same bytes, obtained with no
`unsafe` in the caller. In debug the same call panics instead of returning `Err`, which
is itself a defect for a `Result`-returning safe API.

`alloc_slice` had the same shape one level up: `size_of::<T>() * count` wrapped, so a
`count` just past `usize::MAX / size_of::<T>()` produced a tiny carve with a huge
element count. `BumpVec::new_in` adopts that count as its `capacity`, and `push` — with
its `len < capacity` guard satisfied — then writes past the arena.

`can_allocate` shared the arithmetic and panicked instead of answering `false`.

The `// SAFETY: aligned_offset is within bounds (checked above)` comment at
`bump.rs:120-122` was false in exactly the state the overflow produces.

**RED (watched), debug.** 3/3 fail on the parent commit:

```
test_alloc_bytes_rejects_overflowing_request_without_rewinding
    panicked at src/memory/bump.rs:103:30: attempt to add with overflow
test_alloc_slice_rejects_overflowing_element_count
    panicked at src/memory/bump.rs:71:20: attempt to multiply with overflow
test_can_allocate_answers_false_on_overflow
    panicked: alignment must be a power of two      (can_allocate(3, 3) -> true)
```

**RED (watched), release — this is the CRITICAL half.** Same 3 tests, `--release`:

```
test_alloc_bytes_rejects_overflowing_request_without_rewinding
    panicked: an allocation whose end offset overflows usize must be rejected
test_alloc_slice_rejects_overflowing_element_count
    panicked: count x size_of::<T>() overflow must be rejected
```

i.e. with overflow checks off the allocator **accepted** a `usize::MAX - 99` byte
request out of a 4 KiB arena, storing `new_offset = 0` as the new cursor. The debug
panic is the benign face of this; release is the aliasing one.

**Tests.** `test_alloc_bytes_rejects_overflowing_request_without_rewinding` (asserts
both the `Err` *and* that the cursor did not move, *and* that the next allocation is a
fresh address), `test_alloc_slice_rejects_overflowing_element_count`,
`test_can_allocate_answers_false_on_overflow`.

**Fix.** `checked_add` on both bump steps and `checked_mul` on the slice size;
`can_allocate` mirrors them and rejects a non-power-of-two alignment. The `# Errors`
docs now say why the overflow case must not be left to wrapping.

**Commit.** _pending_
