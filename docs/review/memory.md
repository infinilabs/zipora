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
| `five_level_pool.rs` | 1,550 | yes (full) — see also the note under C3.6 |
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

**Commit.** `c51ba7d`


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

**Commit.** `e536e54`

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

**Commit.** `f54da91`

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

**Commit.** `eb3a9e2`

---

### C3.5 — `FiveLevelPoolConfig` was entirely unvalidated — HIGH

**Finding.** Every field of `FiveLevelPoolConfig` is `pub` and none of them was
checked. The only validation that existed was incidental: `Layout::from_size_align`
inside `MemoryChunk::new`, which reports `Invalid data: Invalid memory layout` and
names nothing. Four defects sat behind that:

1. **`alloc(0)` / `free(p, 0)` underflow.** `alloc_from_fast_bin` and
   `free_to_fast_bin` compute `bin_index = size / alignment - 1`, and `align_up(0)`
   is `0`. A debug build panics with `attempt to subtract with overflow`; a release
   build wraps to `usize::MAX`, skips the bin, and carves a zero-width block off the
   end of the arena. Both entry points are safe public API on all five levels.
2. **`initial_capacity: 0`.** `Layout::from_size_align(0, 8)` is *valid*, so the
   zero-sized layout reaches `std::alloc::alloc`, which the `GlobalAlloc` contract
   forbids.
3. **`initial_capacity > u32::MAX`.** Offsets are a `u32` (`MemOffset`), and
   `MemOffset::new` guarded the narrowing with a `debug_assert!` only. In release two
   live blocks silently share one offset. Working agreement 4 forbids `debug_assert!`
   as the only guard on a value reachable from safe public API.
4. **`alignment: 2` was accepted** although the fast-bin free list writes a 4-byte
   `u32` link into each freed block.

**RED (watched).** 7/7 fail on the parent commit:

```
test_zero_sized_allocation_is_rejected_not_wrapped
    panicked at src/memory/five_level_pool.rs:331:25: attempt to subtract with overflow
test_zero_capacity_is_rejected_before_allocating
    NoLockingPool::new accepted a config that has a zero initial capacity
test_capacity_beyond_the_offset_space_is_rejected
    NoLockingPool::new accepted a config that exceeds the 32-bit offset space
test_alignment_below_the_link_width_is_rejected
    NoLockingPool::new accepted a config that has an alignment smaller than
    the free-list link
test_fast_block_size_must_be_a_multiple_of_alignment
    NoLockingPool::new accepted a config that has a fast block size that is
    not a multiple of the alignment
test_zero_alignment_is_rejected
test_non_power_of_two_alignment_is_rejected
    error should name the offending field, got: Invalid data: Invalid memory layout
```

> [!NOTE]
> The last two are hardening, not defects, and are labelled as such. I expected a
> divide-by-zero in `num_bins = max_fast_block_size / alignment`, but traced it and
> the division is unreachable: `MemoryChunk::new` runs first in every constructor and
> `Layout::from_size_align` rejects a zero or non-power-of-two alignment before it.
> Those two tests only lock in an explicit, named rejection.

**Tests.** `test_zero_sized_allocation_is_rejected_not_wrapped`,
`test_zero_alignment_is_rejected`, `test_non_power_of_two_alignment_is_rejected`,
`test_alignment_below_the_link_width_is_rejected`,
`test_zero_capacity_is_rejected_before_allocating`,
`test_capacity_beyond_the_offset_space_is_rejected`,
`test_fast_block_size_must_be_a_multiple_of_alignment`, plus an
`expect_config_rejected` helper that drives all four constructors.

**Fix.** `FiveLevelPoolConfig::validate`, called by `NoLockingPool::new`,
`MutexBasedPool::new`, `LockFreePool::new` and `ThreadLocalPool::new`
(`FixedCapacityPool` inherits it through `NoLockingPool`), plus a shared
`reject_zero_size` on all ten public `alloc`/`free` entry points.
`FiveLevelPoolConfig::MAX_ARENA_SIZE` caps `initial_capacity`, `arena_size` and
`fixed_capacity` at `u32::MAX`, which is what makes `MemOffset::new`'s narrowing
correct by construction; its `debug_assert!` is now documented as a backstop with the
real guard named.

**Commit.** `341b846`

---

### C3.6 (D3) — every free above 32 KiB was dropped on the floor — HIGH

**Finding.** Levels 1-3 each shipped a stub for blocks above `max_fast_block_size`:
`NoLockingPool::free_to_skip_list`, `MutexBasedPool::free_to_skip_list` and
`LockFreePool::free_to_huge_mutex` all bound the offset as `_offset` and discarded it
behind a `// TODO: Implement skip list insertion` marker — while still charging
`used_memory -= size` and `fragment_size += size`, so `stats()` reported the lost
bytes as reclaimed. The matching `alloc_from_skip_list` / `alloc_from_huge_mutex`
always bumped the end of the arena. A 1 MiB pool cycling a single 64 KiB block ran out
of memory on the sixteenth iteration.

**Decision on the D3 row.** The row offered "a real skip list with tests against a
BTreeMap model, or drop the tier and rename the pool". **The tier is dropped.**
`lockfree_pool.rs` already solves the identical problem with a plain best-fit `Vec`
behind a mutex, and hand-rolling a concurrent skip list would add new `unsafe` surface
during a pass whose whole purpose is to remove it. The pool keeps its name: "five
level" counts concurrency levels, not skip-list levels. The *methods* are renamed
`alloc_huge` / `free_huge`, and the now-unused `pub max_skip_levels` config field is
removed (breaking, though only `Default` ever named it).

**RED (watched).** 5/5 fail on the parent commit:

```
test_large_blocks_are_reused_after_free_level1
    iteration 15: a 64 KiB allocation failed in a 1 MiB arena after 15
    complete alloc/free cycles: Resource exhausted: Out of memory
test_large_blocks_are_reused_after_free_level2   iteration 16: ... Out of memory
test_large_blocks_are_reused_after_free_level3   iteration 16: ... Out of memory
test_adjacent_large_frees_coalesce
    three adjacent 64 KiB frees must coalesce into one 192 KiB region
    left: 196672   right: 0
test_huge_carve_width_equals_recycle_width
    cycle 0: best fit should reuse the head of the freed region
    left: 262208   right: 0
```

**Tests.** The five above, plus `test_no_two_live_blocks_ever_overlap` — **coverage,
not RED**: it passes on the parent precisely because nothing was ever reused there. It
is the model check guarding the new reuse path, interleaving fast-bin and huge traffic
over 4000 randomized steps and asserting pairwise disjointness of every live block.

**Fix.** `HugeFreeList`: address-ordered, coalescing, best-fit, shared by all three
levels (bare in Level 1, behind a `Mutex` in Levels 2-3). Two invariants are
`debug_assert`ed on every insert — regions stay sorted and never adjacent, and a carve
returns *exactly* the requested width with any remainder handed back to the list. The
second is what keeps the carve width equal to the recycle width, so a region cannot
shrink across a reuse cycle. In Levels 2-3 the huge lock is always released before the
memory lock is taken, so the two are never held together.

> [!NOTE]
> Traced and found safe: the tail-rollback fast path in `NoLockingPool::free` can only
> fire for the topmost live block, so it can never roll the high-water mark back below
> a region that is already in the huge free list. Regions in the list are therefore
> always strictly below `memory.size` and can never be handed out twice.

**Commit.** `e06558c`

---

### C3.7 — the fixed-capacity pool permanently demotes every block it lends small — HIGH

**Finding.** `initialize_free_lists` files *every* block on the largest size class, and
nothing ever moves one down. A small request therefore finds `LIST_TAIL` in its own
class and falls into `allocate_by_splitting`, which does not split —

```rust
// For simplicity, just return the larger block
// Real implementation would split the block
return Ok(ptr);
```

— and hands over a full `max_block_size` block. `FixedCapacityAllocation` then recorded
the **requested** class, so `deallocate_to_free_list` filed that block under the small
class. The block is still physically `max_block_size` wide and still `max_block_size`
away from its neighbours, so nothing overlaps; but the carve width and the recycle
width disagree and the pool has permanently reclassified it. A later full-width request
can never find it, while `available_capacity()` keeps counting it as a whole block.

Nothing can be split here: `initialize_free_lists` lays blocks out at a fixed
`max_block_size` stride, so the size-class vector describes a partition that does not
exist in memory.

**RED (watched).** 3/3 fail on the parent commit:

```
test_small_allocations_do_not_permanently_demote_blocks
    all four 4096-byte blocks were returned and available_capacity() reports
    16384 bytes free, but a full-width request failed:
    Memory allocation failed: requested 0 bytes
test_allocation_reports_the_width_it_actually_reserved
    an 8-byte request reserves a whole 4096-byte block   left: 8   right: 4096
test_alternating_width_cycles_do_not_erode_capacity
    round 2 (4096 bytes), allocation 0: the pool was fully drained and refilled
    on every previous round, so all 8 blocks are free:
    Memory allocation failed: requested 0 bytes
```

> [!NOTE]
> The third test only became RED after being sharpened. A *steady* mixed-width demand
> pattern (`[8, 64, 512, 4096, 8, 64, 512, 4096]` every round) passes on the parent,
> because the classes the blocks are demoted into happen to match the next round's
> requests. Erosion needs demand to change, so the test now alternates all-4096 and
> all-8 rounds and fails on round 2. The first draft of this test was green and would
> have been worthless.

**Tests.** The three above. `test_basic_allocation`'s
`assert_eq!(alloc.size(), 64)` is updated to `1024`: `small_objects()` has
`max_block_size: 1024`, so a 64-byte request always reserved a whole 1024-byte block
and reporting 64 was the lie that caused the misfiling.

**Fix.** `allocate_from_free_list` and `allocate_from_larger_class` (renamed from
`allocate_by_splitting`, which never split) now return `(ptr, actual_class_index)`, and
`allocate` records that class rather than the requested one. `deallocate` therefore
refiles the block under the class it was carved from and, with `secure_clear`, zeroes
the width that was actually reserved. `FixedCapacityAllocation::size()` reports the
reserved width. The out-of-memory error also stops claiming `requested 0 bytes`.

**Commit.** `4a3d8a6`

---

### C3.8 — `BumpArena::scope` could rewind over live allocations — HIGH

**Finding.** `BumpArena::scope` took `&self`, so the arena stayed usable while a scope
was alive, and `BumpScope::drop` unconditionally stores the offset it captured at
creation. Any allocation made through the *arena* during the scope's lifetime was
rewound over, and the next allocation handed the same address straight back out while
the first pointer was still live — two aliasing allocations obtained from entirely safe
code. `BumpAllocator::reset` is `&mut self` for exactly this reason; `scope` was not.

`scope()` had no caller anywhere in the crate: `test_bump_scope` builds a `BumpScope`
by struct literal and never goes through the public entry point, which is how the
receiver survived review.

**RED (watched).** The regression test is a `compile_fail` doctest on `scope`, because
after the fix the defect is a borrow-check error and there is no runtime state left to
assert on. On the parent commit:

```
test src/memory/bump.rs - memory::bump::BumpArena::scope (line 272)
     - compile fail ... FAILED
---- src/memory/bump.rs - memory::bump::BumpArena::scope (line 272) stdout ----
Test compiled successfully, but it's marked `compile_fail`.
```

i.e. the aliasing program built cleanly.

**Tests.** The `compile_fail` doctest, plus
`test_arena_scope_rewinds_only_its_own_allocations` as coverage — the first caller of
`scope()` in the crate — pinning that a scope gives back exactly what it took and
nothing more.

**Fix.** `scope(&mut self)`. Breaking only for external callers, of which there are
none in-tree.

**Commit.** `7b019fe`

---

### C3.9 — `CacheAlignedVec<T>` ignores `align_of::<T>()` and divides by zero on a ZST — HIGH

**Finding.** Two defects in the same type, both reachable from safe public API with
no `unsafe` on the caller's side.

1. *Under-alignment.* Every `Layout` in the type was built with `CACHE_LINE_SIZE` as
   the alignment and never consulted `align_of::<T>()` — in `reallocate` for the new
   layout and again for the old one, and a third time in `Drop`:

   ```rust
   let layout =
       Layout::from_size_align(aligned_capacity * mem::size_of::<T>(), CACHE_LINE_SIZE)
   ```

   `CACHE_LINE_SIZE` is 64. For any `T` aligned more strictly than that — a
   `#[repr(align(128))]` element, an AVX-512 vector wrapper — the allocator is free to
   hand back an address that is 64-byte but not 128-byte aligned, and `push`'s
   `ptr::write` is then undefined behaviour. The `Drop` layout also disagreed with the
   allocation layout whenever `align_of::<T>() > 64`, which is UB in its own right.

2. *ZST divide-by-zero.* `reallocate` computed
   `align_to_cache_line(new_capacity * size_of::<T>()) / size_of::<T>()`. For a
   zero-sized element that is a division by zero, so `CacheAlignedVec::<()>::push`
   panics.

**RED (watched).** 3/3 fail on the parent commit `7b019fe`, with only the tests applied
to `7b019fe`'s production code. The alignment test aborts the process, so the two ZST
tests had to be run in separate invocations:

```
thread 'memory::cache::tests::test_cache_aligned_vec_honours_the_element_alignment'
  panicked at src/memory/cache.rs:147:18:
unsafe precondition(s) violated: slice::from_raw_parts requires the pointer to be
aligned and non-null, and the total size of the slice not to exceed `isize::MAX`
thread caused non-unwinding panic. aborting.
  (signal: 6, SIGABRT: process abort signal)

thread 'memory::cache::tests::test_cache_aligned_vec_supports_zero_sized_elements'
  panicked at src/memory/cache.rs:193:13:
attempt to divide by zero

thread 'memory::cache::tests::test_zero_sized_elements_are_dropped_exactly_once'
  panicked at src/memory/cache.rs:193:13:
attempt to divide by zero
```

> [!NOTE]
> The alignment defect is deterministic in a debug build with no sanitizer at all: the
> standard library's own `slice::from_raw_parts` precondition check fires inside
> `as_slice`. In release that check is compiled out and the misalignment is silent, so
> `make miri_pool` is the release-mode oracle. The test allocates 32 vectors rather than
> one because a single 128-byte-aligned request can get a conforming address by luck.

**Tests.** `test_cache_aligned_vec_honours_the_element_alignment` (32 independent
`CacheAlignedVec<OverAligned>`s, each checked against `align_of::<OverAligned>()`),
`test_cache_aligned_vec_supports_zero_sized_elements` (push/pop/clear on
`CacheAlignedVec<()>`), `test_zero_sized_elements_are_dropped_exactly_once` (a ZST with
a `Drop` counter — the drop count must follow `len`, not a byte count).

**Fix.** A single `CacheAlignedVec::<T>::layout_for(capacity)` associated function is
now the only place a `Layout` is built, using `CACHE_LINE_SIZE.max(align_of::<T>())`
and `checked_mul` for the byte count; `reallocate` uses it for both the new and the old
layout and `Drop` uses it too, so the three can no longer drift apart. `reallocate`
short-circuits for zero-sized elements — capacity becomes `usize::MAX`, nothing is
allocated, and `NonNull::dangling()` is already a valid address for any number of them —
and `Drop` skips the deallocation for the same reason. `reserve` uses `saturating_mul`
on the growth factor so the `usize::MAX` ZST capacity cannot overflow.

**Commit.** `58669d5`

---

### C3.10 — large blocks are lent whole and filed back narrow, so the arena erodes — HIGH

**Finding.** `allocate_from_skip_list` took the best fit from `large_blocks` —
`block.size >= aligned_size`, smallest such — and returned it **whole**:

```rust
if let Some(idx) = best_idx {
    let block = blocks.remove(idx);
    ...
    return self.offset_to_ptr(block.offset);
}
```

`deallocate_to_skip_list` then filed it back under the caller's **request** size:

```rust
blocks.push(FreeBlock { offset, size: aligned_size });
```

Carve width and recycle width disagree, so every time a large block is lent to a
smaller request it is permanently reclassified as that smaller block and the
difference can never be allocated again. Nothing coalesced either, so even the
bytes that were filed correctly could not be rejoined.

Two smaller defects in the same paths: the fast-bin CAS-exhaustion fallback filed the
block under the caller's request size rather than the bin's class width, the same
mismatch one tier down; and a large block could be freed twice, which pushed two
`FreeBlock`s for one region and let the pool hand those bytes to two live callers.

**RED (watched).** 3/3 fail on the parent commit `58669d5`:

```
test_large_blocks_keep_their_width_when_lent_to_smaller_requests
    round 15: the 16384-byte request found nothing to reuse and the arena is
    exhausted, yet every earlier allocation was freed:
    Memory allocation failed: requested 16384 bytes

test_a_block_lent_whole_is_filed_back_whole
    the block was lent to a smaller request and came back narrower, so the
    original width is no longer allocatable
      left: 0x7f5d54002d48   right: 0x7f5d54000d30

test_double_free_of_a_large_block_is_refused
    the second free of 0x7f655c000d30 was accepted
```

> [!NOTE]
> The first draft of the erosion test cycled three *descending* widths
> (16384 / 12288 / 8200) and was a much weaker RED: it converged after three fresh
> carves (49,176 bytes instead of 16,392) and never exhausted the arena. The loss
> from this defect is bounded by the number of *distinct* widths ever requested —
> once one block exists per width, best fit starts hitting exact matches. To drain
> the arena the small request has to grow each round, so that the only block that
> fits is always the big one; the test now does that and fails with a genuine
> out-of-memory at round 15 of 32 in a 256 KiB arena with nothing held.

**Tests.** The three above, plus the existing `test_large_allocations_reuse`,
`test_large_block_best_fit_selection` and `test_large_block_fallback_to_new_allocation`,
which continue to pass.

**Fix.** `large_blocks: Mutex<Vec<FreeBlock>>` becomes
`large_free_list: Mutex<LargeFreeList>`, an address-ordered, coalescing free list whose
documented invariant is that the width a block is carved at is the width it is filed
back under:

* `LargeFreeList::alloc` splits the best fit exactly, leaving the remainder in the
  same slot so the list stays sorted. It only declines to split when the remainder
  could not carry its own `BLOCK_HEADER` and still be `ALIGN_SIZE` wide, and then it
  reports the width the caller actually received.
* `LargeFreeList::free` rejects a region that overlaps one already free — a double
  free, or a foreign pointer — and coalesces with the neighbour below and above,
  where adjacency is `a.offset + a.size + BLOCK_HEADER == b.offset` because every
  block's user memory is preceded by its own header.
* The actual carved width is recorded in the block's own header by
  `store_block_width` when it is handed out and read back by `load_block_width` on
  the way in, so the caller's narrower `deallocate` size can no longer shrink it.
  `load_block_width` validates the header against the deallocated size and the arena
  bound and returns `Err` rather than filing a bogus region. The header is free for
  this use: a block is routed to the fast bins or to the large free list by size, the
  two ranges do not overlap, and only the bins use the header as a free-list link.
* The fast-bin fallback now passes `FAST_BIN_SIZES[bin_index]`, the width the block
  was carved at.

**Commit.** `aceae24`

---

### C3.11 — exponential backoff shifts by the retry count and overflows — MEDIUM

**Finding.** `LockFreeMemoryPool::backoff`:

```rust
BackoffStrategy::Exponential { max_delay_us } => {
    let delay = std::cmp::min(1u64 << retry_count, max_delay_us);
```

`max_cas_retries` defaults to 1000 and `Exponential` is the default strategy, so a
fast bin under sustained contention reaches `retry_count == 64` and the shift
overflows. In a debug build that is a panic — from inside the allocator, on the path
that exists precisely to survive contention. In release the shift wraps to
`1 << (retry_count % 64)`, so the delay collapses back to 1 µs exactly when the bin is
most contended, which is the opposite of what a backoff is for.

**RED (watched).** Fails on the parent commit `aceae24`:

```
thread 'memory::lockfree_pool::tests::test_backoff_does_not_overflow_at_high_retry_counts'
  panicked at src/memory/lockfree_pool.rs:908:43:
attempt to shift left with overflow
```

> [!NOTE]
> This RED is debug-only: release wraps the shift instead of panicking, and the
> wrapped delay is not observable from a test without timing it. The retry loop is
> also not reachable deterministically from the public API — 64 consecutive CAS
> failures on one bin cannot be forced — so the test calls the private `backoff`
> directly with retry counts the loop is permitted to reach, after asserting that
> `max_cas_retries` really does exceed 64 under the default config.

**Tests.** `test_backoff_does_not_overflow_at_high_retry_counts`.

**Fix.** `1u64.checked_shl(retry_count).unwrap_or(u64::MAX).min(max_delay_us)`, so the
delay saturates at `max_delay_us` instead of overflowing.

**Commit.** `07746f9`

---

### D8 (1 of 2) — the two `#[ignore]`d `lockfree_pool` concurrency tests

**Finding.** `test_concurrent_allocation` was disabled "for release mode compatibility"
and `test_pool_exhaustion` "to prevent timeouts in release mode". Neither reason was
about the pool.

* `test_concurrent_allocation`'s only assertion was
  `stats.contention_ratio() < 0.5` — a performance heuristic about how often a CAS
  happened to fail, which depends entirely on how the scheduler interleaves two
  threads and says nothing about whether the pool is correct. It also never wrote to
  the memory it was handed, so it could not have seen the A1.6 free-list-in-user-memory
  defect that a later test did find.
* `test_pool_exhaustion` was wrapped in a five-second wall clock and a
  "Too many allocations — possible infinite loop" escape hatch. That loop was real:
  before the `checked_add` in `allocate_new_block`, the bump cursor wrapped and the
  pool never reported exhaustion. It is fixed, and `test_exhaustion_overflow_safety`
  covers it.

A disabled concurrency test on a lock-free allocator is exactly the kind of gap that
hid the Treiber defect, so neither is left off.

**Tests.** Both rewritten around properties that do not depend on timing, and
un-ignored:

* `test_concurrent_allocation` — 4 threads × 64 mixed-size blocks, twice over. Each
  thread writes its own id into every byte of every block it holds and reads them all
  back, so a block handed to two threads at once is both an assertion failure and a
  data race ThreadSanitizer can see. After the join, no two live blocks may overlap.
  The second round can only pass if the first gave everything back. The arena is sized
  for the worst case in which nothing is ever recycled, so an allocation failure is a
  defect rather than a capacity coincidence, and `BackoffStrategy::None` keeps the run
  free of sleeps. The contention ratio is now printed, not asserted.
* `test_pool_exhaustion` — no clock and no escape hatch. A 1 KiB arena holds exactly
  `(1024 - ALIGN_SIZE) / (64 + BLOCK_HEADER)` 64-byte blocks; the test allocates
  exactly that many, requires the next one to be refused, frees them all and requires
  the whole arena to be allocatable again.

**Verification.** `memory::lockfree_pool` is 32 passed / **0 ignored** in debug and in
release, and `test_concurrent_allocation` was run ten more times in release: 10/10.
Under `make tsan_pool`: clean.

**Not a RED.** Both are coverage: they assert properties the pool already satisfies at
this commit. The defect being fixed is the absence of the tests, not the pool.

**Commit.** _pending_
