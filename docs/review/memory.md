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
  was carved at. (**Superseded by C3.18**: that fallback filed a fast-bin block where
  no fast-bin request could reach it, and no longer exists — the push is unbudgeted.)

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

**Commit.** `687e428`

---

### C3.12 — `MemoryPool::deallocate` takes any pointer, and the same one twice — CRITICAL

**Finding.** `MemoryPool::deallocate` was a *safe* `pub fn` taking a bare
`NonNull<u8>`, and it validated nothing:

```rust
pub fn deallocate(&self, chunk: NonNull<u8>) -> Result<()> {
    self.dealloc_count.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut free_chunks) = self.free_chunks.try_lock()
        && free_chunks.len() < self.config.max_chunks
    {
        free_chunks.push_back(chunk.as_ptr());
```

Two holes, both reachable without the caller writing a single `unsafe` block:

1. **Any pointer.** A pointer this pool never handed out is either parked on the free
   list — from where it is given to the next caller — or passed to the global
   allocator with the pool's own layout.
2. **The same pointer twice.** A double free parks one address twice, so the next two
   allocations both return it: two live callers owning the same chunk. When both are
   freed and the pool is dropped, `clear` passes that address to the global allocator
   twice.

The method's own doc comment described both as "CRITICAL VULNERABILITY" and
"CONFIRMED EXPLOIT" and left them in place. The test that was supposed to cover it,
`test_double_free_attempt`, only `println!`ed *"VULNERABILITY: Double-free succeeded!
Pool is now corrupted."* and passed either way.

The `try_lock` made it worse: a contended free skipped straight to the global
allocator, so even a check placed under that lock would have been skippable by timing.

**RED (watched).** With only the rewritten test applied to `687e428`, the process dies
before any assertion can run:

```
running 1 test
free(): double free detected in tcache 2
  (signal: 6, SIGABRT: process abort signal)
```

That is glibc catching the pool passing one address to `free` twice when the pool is
dropped — the corruption the old test printed a message about.

**Tests.** `test_double_free_is_refused` (replacing `test_double_free_attempt`): the
second free must be an `Err`, and the next two allocations must not be the same
address. `tests/security_memory_pool_simple.rs` turned out to hold a second copy of
the same demonstration, `test_double_free_safety`, which likewise only printed
*"VULNERABILITY CONFIRMED: Double-free succeeded!"*; it now asserts the same two
properties.

**Fix.** `deallocate` becomes an `unsafe fn` with a `# Safety` section: the caller owes
that the chunk came from *this* pool and is live, because no `Result` can express a
foreign pointer. Being freed twice is not the caller's word to take — the pool now
scans its free list under a blocking (poison-recovering) lock and returns
`invalid_data`. The scan is O(`max_chunks`) under a mutex the method already took, and
that is documented on the method. Call sites updated with `SAFETY:` comments:
`PooledVec::drop`, `PooledBuffer::drop`, the two `tiered.rs` routes, the C API's
`memory_pool_deallocate`, and the in-tree tests.

> [!NOTE]
> This closes the hole one level down but not the public chain above it:
> `TieredAllocation::Small(NonNull<u8>, usize)` is a public enum variant, so safe code
> can still forge one and hand it to `TieredMemoryAllocator::deallocate`, which is
> safe. That is tracked separately.

**Commit.** `1529ac1`
---

## D8 (2/2) — twelve `MemoryPool` "security" tests that asserted nothing

**Scope.** `tests/security_memory_pool.rs`, `tests/security_memory_pool_simple.rs`,
`tests/memory_pool_contract.rs` (new), `src/memory/pool.rs` (docs), `Makefile` (comment).

**Finding.** The two files held twelve near-duplicate tests named after the defects
they were supposed to catch — `test_double_free_attempt`, `test_use_after_free_*`,
`test_concurrent_pool_access_race_condition`, and so on. Not one of them asserted the
property in its name. The pattern throughout was:

```rust
println!("VULNERABILITY CONFIRMED: Double-free succeeded!");
Ok(())
```

They passed on a correct pool and on a broken one, which is how the C3.12 double free
survived with a test named after it. Two of them (`test_use_after_free_detection`,
`test_dangling_pointer_access`) were `#[ignore]`d with the note that they are UB by
design.

**Decision on the two ignored tests.** The D8 row asked for them to be moved to a
Miri/ASan negative job rather than deleted. They cannot be: both demonstrate *use after
return to the pool*, and a pool does not hand recycled memory back to the system. The
chunk is still a live global allocation sitting on the free list, so there is no dead
allocation for Miri to flag and no `free()` for ASan to poison. Such a job would pass
unconditionally and would be a third test that asserts nothing. They are deleted, and
the obligation they were gesturing at is now written into `deallocate`'s `# Safety`
section and into the type's doc comment, where a caller will actually read it.

**Tests.** `tests/memory_pool_contract.rs`, six tests, each asserting a property that
can fail:

| Test | Property |
|---|---|
| `test_double_free_is_refused` | second free is `Err`; the next two allocations are distinct addresses |
| `test_counters_are_exact_under_contention` | `pool_hits + pool_misses == alloc_count` exactly, 8 threads |
| `test_concurrent_allocations_are_distinct` | no address is handed to two threads at once |
| `test_pool_caches_at_most_max_chunks` | the cache is bounded by `max_chunks`; the surplus goes back to the global allocator |
| `test_clear_releases_only_pooled_chunks` | `clear` leaves chunks that are out on loan alone |
| `test_memory_pool_is_send_and_sync` | the `unsafe impl`s stay, statically |

Labelled **coverage, not RED**: `1529ac1` already fixed the one live defect here, so
these lock in behaviour rather than fail first. `test_double_free_is_refused` is the
exception — it fails on `687e428` with `free(): double free detected in tcache 2`, and
that RED is recorded under C3.12. The two double-free assertions C3.12 added to the
now-deleted files are that test; nothing that asserted anything was lost.

**Fix.** The `MemoryPool` doc comment was six screens of "CONFIRMED VULNERABILITY",
"Thread Safety Guarantees (VIOLATED)" and a recommendation to use jemalloc instead. It
described a type that no longer exists after C3.12, and it referred the reader to an
absolute path on the author's machine. It is replaced by what the type guarantees
(chunk width and alignment, uninitialized contents, `max_chunks` bounds the cache and
not the live set, no address is live twice, exact counters) and what it does not (it
cannot recognise a foreign pointer, a use-after-free is invisible to it *and* to a
sanitizer, `allocate` uses `try_lock` so `pool_hits` is not deterministic). The same
treatment for `allocate`, `clear` and `stats`. The two `unsafe impl Send/Sync` get a
real four-point `SAFETY:` argument in place of a list of the vulnerabilities they
supposedly caused.

The `miri_pool` comment in the `Makefile` claimed the two ignored tests "live here as
negative tests". They never did; that is corrected to the reasoning above.

**Audit deltas.** `api_honesty.max_markers` and `unsafe_audit.max_undocumented` both
drop; re-ratcheted in the `build(gates):` commit that closes this range.

**Commit.** `4932d7d`
---

## C3.13 — the global-stack arm of `zero_on_alloc` had no test

**Scope.** `src/memory/secure_pool.rs` (tests only).

**Finding.** `2707e30` fixed both zeroing gaps in `SecureMemoryPool` and added
`test_zero_on_free_clears_recycled_chunks` and
`test_zero_on_alloc_clears_recycled_chunks`. Both allocate one chunk, free it, and
allocate again on the same thread, so both only ever take the **thread-local cache**
hit. `allocate_with_hint` has a second, textually separate copy of the same check for
the **global stack** arm (`secure_pool.rs:1024`), and nothing asserted it. The
default `local_cache_size` is 64, so a test has to lower it before the global stack is
even reachable.

This is the shape of gap that C3.7 and C3.10 both turned out to be: a second copy of a
rule, on a path the tests never take.

**Mutation check (in place of a RED).** No production change here, so there is no
commit on which these fail. Instead, with the branch at `secure_pool.rs:1023-1026`
deleted and everything else at `4932d7d`:

```
thread 'memory::secure_pool::tests::test_zero_on_alloc_clears_chunks_taken_off_the_global_stack'
panicked at src/memory/secure_pool.rs:1787:13:
a recycled chunk still carries the previous tenant's bytes

test result: FAILED. 4 passed; 1 failed
```

The other four zeroing tests pass with that branch gone, which is the point: they do
not reach it. The mutation was reverted immediately; the committed tree is unchanged
apart from the tests.

**Tests.** A shared `recycle_through_global_stack` helper sets `local_cache_size` to 2
and drives `local_cache_size + 4` chunks through the pool, so four of them are recycled
via the global stack; it asserts every byte of every chunk in the second round is zero,
and returns the addresses and the `cross_thread_steals` delta so each test can assert
`steals > 0` — otherwise the test would silently degrade into another local-cache test.
Each test also asserts the second round hands out distinct addresses.

* `test_zero_on_free_clears_chunks_routed_through_the_global_stack`
* `test_zero_on_alloc_clears_chunks_taken_off_the_global_stack`

**Fix.** None: the production code is correct. This closes the "`zero_on_free` and
`zero_on_alloc` honoured on cache, stack, and list paths" item of the C3 scope by
covering the one path that was not covered.

**Commit.** `e770b3d`
---

## C3.14 — `MmapAllocation` had no `Drop`, so letting one go out of scope leaked the mapping

**Scope.** `src/memory/mmap.rs`.

**Finding.** `MemoryMappedAllocator::allocate` is a safe public method returning a
public owning handle:

```rust
#[derive(Debug)]
pub struct MmapAllocation {
    ptr: NonNull<u8>,
    size: usize,
    actual_size: usize, // Rounded up to page size
}
```

There was no `Drop`. The region was released only by handing the value back to
`MemoryMappedAllocator::deallocate`; on every other route — an early `return` or `?`
between the allocation and the matching `deallocate`, a panic unwinding past it, or
simply not calling it — the mapping stayed in the address space for the life of the
process. The type is named, `Debug`-printed and accessed like an RAII handle, and the
allocator's own `Drop` unmaps only what is in its *cache*, so a caller reading that
code would reasonably assume the handle cleans up after itself.

This reaches further than `mmap.rs`: `TieredAllocation::Large(MmapAllocation)` is a
public variant, so dropping a `TieredAllocation` leaked too.

**RED (watched).** First attempt used the process-wide total from `/proc/self/statm`:

```
dropping 32 allocations of 4194304 bytes grew the address space by 1950597120 bytes:
the mappings were never unmapped
```

1.95 GB against an expected 134 MB — `cargo test` runs the suite in parallel threads,
so a process-wide number also counts every other test's mappings. The RED was real but
the GREEN would have flaked, so the oracle was replaced with a per-address probe of
`/proc/self/maps`:

```
thread 'memory::mmap::tests::test_dropping_an_allocation_releases_the_mapping'
panicked at src/memory/mmap.rs:339:9:
0x7feb07200000 is still mapped after the allocation was dropped: the mapping leaked
```

**Tests.**

* `test_dropping_an_allocation_releases_the_mapping` — asserts the address *is* mapped
  first, so the test cannot pass vacuously, then drops and asserts it is gone.
* `test_deallocate_does_not_unmap_a_cached_region` — the mirror, and the regression
  test for the fix itself: handing the allocation back must leave the region mapped
  (`munmap_calls == 0`) and the next allocation of that size must get the same address
  back (`cache_hits == 1`). Without the `ManuallyDrop` below, the fix would unmap a
  region it had just cached and the allocator would hand out dead pointers.

**Fix.** `impl Drop for MmapAllocation` unmaps `actual_size` bytes at `ptr` and logs a
warning if `munmap` fails. `MemoryMappedAllocator::deallocate` takes the region over
from the handle, so it now wraps its argument in `std::mem::ManuallyDrop` before
reading the fields — the allocator either caches the region (still mapped) or unmaps it
itself, and the handle's own unmap must not also run. Documented on both: dropping is
correct but always unmaps and never updates the statistics; `deallocate` is the fast
route and the only one that does.

**Commit.** `c28ce55`
---

## C3.15 — safe code could forge a `TieredAllocation`, and free one into the wrong allocator

**Scope.** `src/memory/tiered.rs`, `src/memory/mod.rs`.

**Finding.** C3.12 made `MemoryPool::deallocate` an `unsafe fn` because the pool cannot
recognise a pointer it never handed out. One level up, that contract was unenforceable:

```rust
pub enum TieredAllocation {
    /// Small allocation from memory pool (pointer, size)
    Small(NonNull<u8>, usize),
    /// Medium allocation from size-classed pools (pointer, size)
    Medium(NonNull<u8>, usize),
```

A public enum with public tuple variants. Any safe code could name a pointer and a
length, build a `TieredAllocation::Small`, and hand it to `TieredMemoryAllocator::
deallocate` — a *safe* method — which passed it straight into `unsafe { self.small_pool.
deallocate(ptr) }`. The same forged value passed to `as_mut_slice()` is an arbitrary
write. No `unsafe` block anywhere in the caller.

A second route needs no forging at all. Each `TieredMemoryAllocator` owns its own
`MemoryPool`, so an allocation from allocator *a* handed to allocator *b* is a foreign
pointer as far as *b*'s pool is concerned, and `deallocate` took it. Both are reachable
from entirely safe code, and the global `tiered_deallocate` makes the second one easy to
hit by accident.

**RED (watched), both parts.**

```
thread 'memory::tiered::tests::test_deallocate_rejects_an_allocation_from_another_allocator'
panicked at src/memory/tiered.rs:701:9:
a chunk from allocator a was parked in allocator b's pool
```

```
test src/memory/tiered.rs - memory::tiered::TieredAllocation (line 34) - compile fail ... FAILED
Test compiled successfully, but it's marked `compile_fail`.
```

The second is the C3.8 pattern: the property is "this must not compile", so the test is
a `compile_fail` doctest and the RED is rustdoc reporting that it compiled.

**Tests.**

* the `compile_fail` doctest above — forging from a bare pointer and length;
* a second `compile_fail` doctest on `SmallBlock` — forging via the struct literal now
  that the type exists. **Coverage, not RED**: before the fix there was no `SmallBlock`,
  so it could only have failed to compile for the wrong reason.
* `test_deallocate_rejects_an_allocation_from_another_allocator`.

**Fix.** `Small` and `Medium` now hold `SmallBlock` / `MediumBlock`, whose fields are
private and which have no constructor, so outside this module the only source of an
allocation is `TieredMemoryAllocator::allocate`. `SmallBlock` also carries an
`allocator_id` taken from a process-wide `NEXT_ALLOCATOR_ID` counter, and `deallocate`
returns `invalid_data` when it does not match. The chunk is then leaked rather than
filed into the wrong pool — this allocator has no way to reach the one that owns it —
and that is documented on the method.

`MediumBlock` deliberately has **no** id: the medium pools are thread-local and shared
by every allocator on the thread, and `TieredAllocation` holds a `NonNull` and is
therefore `!Send`, so a medium block cannot reach a thread whose pools it did not come
from. That reasoning is written on the type so a future `unsafe impl Send` has to
confront it.

> [!NOTE]
> The `Small` and `Medium` arms of `as_slice`/`as_mut_slice` still materialise a `&[u8]`
> over **uninitialized** pool memory; only `Large`/`Huge` are kernel-zeroed. Tracked
> separately as C3.16 — this entry is about who can construct and destroy an allocation,
> not what reading one yields.

**Commit.** `cfe2bfe`
---

## C3.16 — `TieredAllocation::as_slice` read uninitialized pool memory, and leaked the previous tenant's bytes

**Scope.** `src/memory/tiered.rs`.

**Finding.** The `Small` and `Medium` arms build a `&[u8]` over memory the allocator
never initialized:

```rust
TieredAllocation::Small(block) => unsafe {
    std::slice::from_raw_parts(block.ptr.as_ptr(), block.size)
},
```

`MemoryPool` hands out `alloc`'d chunks and recycles them without clearing, so for a
fresh chunk this is a read of uninitialized memory — undefined behaviour, reached from a
safe method — and for a recycled one it is the previous tenant's bytes, delivered to the
next caller through an entirely safe API. The `Large` and `Huge` arms have neither
problem, and only by accident: `MAP_ANONYMOUS` pages arrive zeroed from the kernel. That
is also why no existing test caught it — the large-allocation tests would have.

**RED (watched).**

```
---- memory::tiered::tests::test_a_recycled_small_chunk_is_zeroed stdout ----
panicked at src/memory/tiered.rs:808:9:
a recycled small chunk carried 0xAB into the next caller

---- memory::tiered::tests::test_a_recycled_medium_chunk_is_zeroed stdout ----
panicked at src/memory/tiered.rs:815:9:
a recycled medium chunk carried 0xAB into the next caller

test result: FAILED. 11 passed; 2 failed
```

The recycled case is the assertable half of the defect: the uninitialized-read half has
no deterministic observation from a test, but the same write fixes both. Each test first
asserts the second allocation really is the same address, so neither can pass vacuously
if the pool stops recycling.

**Tests.** `test_a_recycled_small_chunk_is_zeroed`,
`test_a_recycled_medium_chunk_is_zeroed`, over a shared `recycled_chunk_is_clean`
helper.

**Fix.** `allocate_small` and `allocate_medium` `write_bytes(.., 0, size)` over the
chunk the pool just handed out, each with a `SAFETY:` comment naming the bound that
makes `size` fit (`size <= SMALL_THRESHOLD`, and the loop guard
`size <= pool.config().chunk_size`). `TieredMemoryAllocator::allocate` now documents
that the first `size` bytes are zero whichever tier serves the request — the kernel does
it for the mapped tiers, the pooled tiers pay a `memset` of `size` bytes — and states
plainly that this is what makes `as_slice` sound.

**Commit.** `d24d8f1`, hardened in `7cf55d2`
---

## C3.17 — the last undocumented `unsafe fn` in `src/memory/`

**Scope.** `src/memory/simd_ops.rs`.

**Finding.** `unsafe_audit.py` listed exactly one undocumented `unsafe` site in this
subsystem: `simd_memcpy_unaligned` at `simd_ops.rs:399`, an `unsafe fn` taking two raw
pointers and a length with no `# Safety` section. Its *body* is fully commented — four
`// SAFETY: caller ensures pointers valid and len within bounds` lines — which is the
inverse of what is wanted: the obligation is stated where it is relied upon and nowhere
where it is imposed.

**Fix.** A `# Safety` section naming both obligations — validity for `len` bytes at each
pointer, and non-overlap, because the SIMD tiers copy in blocks in an order the caller
cannot rely on, so these are `copy_nonoverlapping` semantics and not `memmove` — plus a
line recording that the single caller, `SimdMemOps::copy_nonoverlapping`, rejects a
length mismatch and an overlap with `invalid_data` before reaching it. No behaviour
change.

`unsafe_audit.py`: 34 → 33 undocumented sites, **0 in `src/memory/`**.

**Commit.** `f953869`
---

## C3.18 — a spent CAS retry budget reported OOM with free blocks in the bin

**Scope.** `src/memory/lockfree_pool.rs`.

**Found by.** Miri. `make miri_pool` failed `memory::lockfree_pool::tests::test_pool_exhaustion`
with `second round, block 1: Memory allocation failed: requested 64 bytes` — the test
allocates the whole arena, frees every block, and re-allocates. Miri deliberately fails
`compare_exchange_weak` spuriously, which no x86 run does, so the native suite was green.

**Finding.** `allocate_from_fast_bin` and `deallocate_to_fast_bin` were both
`for retry in 0..self.config.max_cas_retries` loops around `compare_exchange_weak`.
`compare_exchange_weak` is *allowed to fail spuriously* — LL/SC hardware (AArch64) produces
those failures, and Miri models them — and the loops re-read `head` on every iteration, so
the weak form bought nothing in exchange for making a budgeted loop's budget burnable with
no contention at all.

Both fallbacks then gave a wrong answer:

- pop: fell through to `allocate_new_block`, so a pool whose arena was fully carved reported
  `out_of_memory` while recycled blocks of exactly the right class sat in the bin;
- push: fell through to `deallocate_to_skip_list(ptr, FAST_BIN_SIZES[bin])`, filing a
  ≤ 8 KiB block on the *large* free list, which only serves requests above
  `FAST_BIN_THRESHOLD`. The block is unreachable until a neighbour coalesces with it.

Together they are worse than either: the frees strand the arena in the large list and the
next round of allocations then reports exhaustion. That is the Miri failure, and on AArch64
it is reachable from a single thread doing nothing unusual.

`max_cas_retries` is a throughput knob — how long to contend for a recycled block before
carving fresh memory. It was silently also a correctness parameter.

**RED.** `test_cas_budget_exhaustion_does_not_report_oom`, a native and deterministic
reproduction: `max_cas_retries = 0` makes the loop body unreachable, which is what a run of
spurious failures amounts to. Allocate the arena dry, free everything, re-allocate. Before
the fix: `block 0 of 14: Memory allocation failed: requested 64 bytes`. The Miri failure
above is the same defect reached the hard way.

**Fix.**

1. Both CAS sites use the *strong* `compare_exchange`. A budgeted loop must never use the
   weak form.
2. The pop is factored into `pop_fast_bin(bin, budget) -> Result<Option<u32>>`, and
   `allocate_from_fast_bin` now states the invariant it wants: prefer the bin, then carve,
   and **if carving fails, retry the bin unbudgeted** rather than report OOM. That
   terminates: with a strong CAS, every failure means another thread completed a push or a
   pop.
3. The push is unbudgeted outright. A Treiber push always has somewhere to go, so a budget
   could only ever produce a worse answer than waiting; the large-free-list fallback — and
   with it C3.10's documented wart — is gone. `MAX_BACKOFF_RETRY = 16` caps the schedule the
   unbounded loop feeds to `backoff`, so a `Linear` strategy cannot sleep for longer and
   longer forever.

`test_deallocate_cas_exhaustion_falls_back_to_skip_list` (C3.10) asserted the old wart. It is
now `test_deallocate_cas_exhaustion_recycles_into_the_bin` and makes the stronger assertion:
after the free, the bin's `count` is 1.

**Refuted.** The third `compare_exchange_weak`, on `next_offset` in `allocate_new_block`
(`lockfree_pool.rs:763`), is correct as written: its loop is unbounded and re-reads the
observed value from the `Err` arm, which is exactly the idiom the weak form exists for.

**Sweep.** Every `compare_exchange_weak` in `src/` was then checked for the same shape — a
weak CAS whose failure is not simply retried. In `src/memory/` the two fixed here were the
only ones: `bump.rs:229` is a comment, and `fixed_capacity_pool.rs:645`, `:367`, `:720` and
`five_level_pool.rs:961`, `:1015` all sit in unbounded `loop {}`s that re-read the head.

Two sites **outside this subsystem** have the defect and are recorded here so they are not
lost; they belong to whichever phase covers `src/thread/`:

* `src/thread/atomic_ext.rs:107` — `AtomicExt::update_if` is a bare weak CAS with no loop,
  so this public method returns `false` ("the condition did not hold, or someone raced me")
  spuriously, on an uncontended atomic, with the condition true.
* `src/thread/linux_futex.rs:164` — `FutexMutex::try_lock` is a bare weak CAS, so it can
  return `Ok(None)` for an unlocked mutex. (`lock`'s fast path at `:149` is fine: a spurious
  failure only sends it down `lock_slow`, which is correct either way.)

**Commit.** `a215001`
---

## C3.19 — `CacheOptimizedAllocator::allocate_aligned(0)` was UB, `align_to_cache_line` overflowed to 0, two deallocators were safe `pub fn`s, and five leaked test blocks broke `make miri_pool`

**Scope.** `src/memory/cache_layout.rs`, `src/memory/cache.rs`.

**Found by.** Running `make miri_pool` to completion after C3.18 unblocked line 3
(`memory::lockfree_pool`). Line 5 (`memory::cache`) had never executed before because
`make` aborted at line 3; when it finally ran, all 35 tests passed and Miri's leak check
then aborted with five leaked `align: 64` allocations (`cache.rs:832-833` in
`test_numa_pool_stats` and `cache_layout.rs:890-892` in `test_cache_layout_stats`).
Inspecting those two call sites surfaced the remaining defects in the pair.

**Findings.**

1. **`CacheOptimizedAllocator::allocate_aligned(0, ..)` is UB from a safe public API** — the
   exact twin of C3.3's `numa_alloc_aligned(0, ..)`. `align_to_cache_line(0, 64)` evaluates
   `(0 + 63) & !63 == 0`; `Layout::from_size_align(0, 64)` succeeds with `size == 0`; and
   `unsafe { alloc(layout) }` is then called on a zero-sized `Layout`, which the standard
   library's safety contract explicitly forbids.
2. **`align_to_cache_line(size, cache_line_size)` overflows `usize` on large `size`.** In a
   debug build `allocate_aligned(usize::MAX, 64, false)` panics (`attempt to add with
   overflow` at `cache_layout.rs:487`). In a release build `(usize::MAX + 63) & !63` wraps
   to `0`, which then takes path (1) and hands a zero-sized `Layout` to `std::alloc::alloc`.
   A zero or non-power-of-two `cache_line_size` in `CacheLayoutConfig` similarly underflows
   `cache_line_size - 1` or masks with a non-mask.
3. **`numa_dealloc` (`cache.rs:538`) and `CacheOptimizedAllocator::deallocate_aligned`
   (`cache_layout.rs:255`) were safe `pub fn`s whose own `// SAFETY:` comments stated an
   unchecked caller precondition.** Neither allocator records live allocations; both
   reconstruct a `Layout` from caller-supplied `(size, align)` arguments and pass `(ptr,
   layout)` straight to `std::alloc::dealloc`. Calling either from safe code with a dangling
   pointer, a mismatched `(size, align)`, or twice is immediate UB — the same defect class
   as `MemoryPool::deallocate` (C3.12).
4. **Five leaked blocks in two tests (`cache.rs:832-833`, `cache_layout.rs:890-892`)** made
   `make miri_pool` fail its leak detector on line 5.

**RED (watched).**

```
thread 'memory::cache_layout::tests::test_allocate_aligned_rejects_overflowing_size'
  panicked at src/memory/cache_layout.rs:487:6:
attempt to add with overflow

thread 'memory::cache_layout::tests::test_allocate_aligned_rejects_zero_size'
  panicked at src/memory/cache_layout.rs:894:14:
zero-sized cache-aligned allocation must be rejected: 0x5626e0a3b700
```

Plus the five Miri leak errors from `make miri_pool` (`alloc1898988`, `alloc1843296`,
`alloc1843153`, `alloc1898962`, `alloc1898936`), and two `compile_fail` doctests proving
that calling `numa_dealloc` or `CacheOptimizedAllocator::deallocate_aligned` outside `unsafe`
no longer compiles.

**Fix.**

* `CacheOptimizedAllocator::checked_layout` rejects `size == 0`, rejects `cache_line_size ==
  0` or non-power-of-two, and rounds `size` with `checked_add(line - 1)` before calling
  `Layout::from_size_align`. Both `allocate_aligned` and `deallocate_aligned` share it.
* `numa_dealloc` and `CacheOptimizedAllocator::deallocate_aligned` are now `pub unsafe fn`
  with `# Safety` and `# Errors` sections and `compile_fail` doctests (**breaking**).
* `test_numa_pool_stats` and `test_cache_layout_stats` deallocate the five blocks they
  allocate. `MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
  memory::cache -- --test-threads=1`: **37 passed, 0 failed, 0 leaks**.

**Commit.** _pending_
---

# Open findings — read, judged, not fixed

Everything below was found by reading the file and is recorded here so the next pass
starts from a list rather than from scratch. Nothing here is a guess: each entry names
the lines and the concrete bad outcome. None of them has a test yet, which is exactly
why they are still open — the working agreement is RED first, and a RED for several of
these needs machinery this stage did not build (a dereferenceable `five_level_pool`, a
TLS-teardown harness, a 1 GB-hugepage machine).

## `five_level_pool.rs` — the whole file is blocked on one decision

`alloc` returns `MemOffset`, a `#[repr(transparent)] pub struct MemOffset(u32)` whose
field is private, whose `to_usize` and `is_null` are private, and whose only route to an
address, `MemoryChunk::offset_ptr`, is a private method on a private struct. **No caller
outside the module can dereference an allocation.** The only external consumer,
`benches/five_level_pool_bench.rs`, never touches the memory. So this is an address-space
bookkeeper, not an allocator, and the defects below are latent rather than exploitable.
The owner has been asked to choose: make it real (public accessor + the fixes below),
fix the internals without an accessor, or deprecate the module.

| # | Lines | Finding |
|---|---|---|
| F1 | L612, L819, L1010 | The free-list link is written **into user memory**, at the exact address `alloc` returns, on all three levels. `lockfree_pool.rs` fixed the identical defect with an 8-byte `BLOCK_HEADER` (see A1.6); this file never did. Latent only because of the paragraph above. |
| F2 | L388 | `LockFreeFreeListHead.head` is a bare `AtomicU32` with no generation tag: textbook ABA. `lockfree_pool.rs` packs `gen << 32 \| offset` into an `AtomicU64`. |
| F3 | L937 | `alloc_from_fast_bin_lockfree` takes `self.memory.lock()` on **every CAS iteration**. The "lock-free" level is neither lock-free nor fast. |
| F4 | L1108-1240 | `ThreadLocalPool` returns thread-local arena offsets **and** global-pool offsets as the same opaque `MemOffset`, and `free` routes on `offset.to_usize() < cache.arena.len()` (L1230). With the default configuration, local offset 0 and global offset 0 are both live and compare equal. |

Traced and **refuted**, do not re-chase: the tail-rollback fast path in
`NoLockingPool::free` can only fire for the topmost live block, and
`alloc_from_fast_bin_lockfree`'s carve width does equal its recycle class width.

## `tiered.rs`

| # | Lines | Severity | Finding |
|---|---|---|---|
| T3 | L328 | MEDIUM | `total_bytes.fetch_add(size)` runs *before* the allocation can fail and is never decremented on failure, so a failing allocator reports ever-growing usage. |
| T4 | L391, L452, L475 | MEDIUM | `MEDIUM_POOLS.with(..)` panics if reached during TLS teardown, and `deallocate` is the kind of method a `Drop` calls. `try_with` and an `Err` would be honest. A RED needs a TLS-destructor harness that forces MEDIUM_POOLS to be destroyed first (registration order is LIFO), which is why it is not in this range. |
| T5 | L153 | MEDIUM | `expect("memory pool creation")` inside the `thread_local!` initializer. Unreachable as written — the config is a literal — but it is a panic in a TLS initializer. |
| T6 | L211 | LOW | `63 - size.leading_zeros()` hardcodes 64-bit. On a 32-bit target every bucket lands ≥ 32, the `if bucket < 32` guard rejects all of them, and the histogram stays all-zero, so `get_allocation_pattern` silently always answers `Mixed`. Should be `usize::BITS - 1 - lz`. |
| T7 | L415-427, L512-529 | LOW | `optimize_for_pattern()` is a `log::debug!` and `Ok(())`. The two `unsafe impl` SAFETY comments justify a field that does not exist (`bump_allocator`). |

**Refuted**, do not re-chase: medium-pool carve width equals recycle width;
`TieredAllocation` is auto-`!Send`, so the thread-local cross-thread-free hazard is
unreachable; `Huge`/`Large` byte accounting is symmetric, because `MmapAllocation::size()`
returns the requested size and not `actual_size`.

## `mmap.rs` (after C3.14)

| # | Lines | Severity | Finding |
|---|---|---|---|
| M2 | L126-150 | MEDIUM | The cache-hit path returns **dirty** memory while the fresh path returns kernel-zeroed pages, so what a caller sees depends on cache state. C3.16 hit the same split one layer up and closed it by zeroing; this one is still open. |
| M3 | L124 | LOW | `(size + page_size - 1) & !(page_size - 1)` overflows for `size` within a page of `usize::MAX`. |
| M4 | L273 | LOW | `sysconf(_SC_PAGESIZE) as usize` without checking for `-1`; a `-1` becomes `usize::MAX` and every rounding after it is wrong. |
| M5 | L134, L213, L235, L254 | LOW | Counters are incremented before the syscall; `stats()` reads the cache under `try_lock` and reports `cached_regions: 0` when contended; `clear_cache` silently no-ops on a poisoned mutex. |

**Refuted**: carve width equals recycle width; no dealloc-layout mismatch.

## `hugepage.rs`

| # | Lines | Severity | Finding |
|---|---|---|---|
| H1 | L175, L224 | LOW | `(size + page_size - 1)` debug-panics / wraps on overflow. |
| H2 | L184 | MEDIUM | `MAP_HUGETLB` is passed without `MAP_HUGE_1GB`, so a pool configured for 1 GB hugepages silently gets 2 MB ones and the statistics report 1 GB. |
| H3 | — | LOW | The `munmap` result is ignored. |
| H4 | — | LOW | The allocation registry is written and never read. |

**Refuted**: carve width equals recycle width.

## `cache.rs` (after C3.9)

`bind_to_numa_node` (L455-461) is `let _ = (ptr, size, node);`. **Every NUMA guarantee
in this module is unimplemented**, while `numa_node()` reports the requested node back to
the caller as though it had been honoured. The thread-to-node "hash" (L440) is
`format!("{:?}", thread_id).len()`, which is the same value for almost every thread and
heap-allocates a `String` on the `CacheAlignedVec::new()` path.

## `threadlocal_pool.rs`

`deallocate_to_global` (L329) and `deallocate_bypass_cache` (L443) **knowingly
leak**, with `log::warn!("Bypassing secure pool deallocation - potential leak")` standing
in for a fix. `size_to_list_index` puts a recycled pointer back by size alone, with no
ownership or class tag. L606-611 skips the multithreading test "due to Send trait
limitations".

## `secure_pool.rs` (after C3.13)

L987 has the hot/cold separation commented out while `enable_hot_cold_separation` stays
in the public config; L1171 is `let optimal_node = -1; // Simplified: disable NUMA for
now` under `enable_numa_awareness`; **L1192 increments `huge_page_allocs` and then falls
through to the regular allocation**, so that counter reports huge pages that were never
requested from the kernel.

## `pool.rs` (after C3.12 and D8)

`init_global_pools` (L405) validates its argument and then does nothing with it.
`PooledVec::new()` (L454, L461) divides by `size_of::<T>()`, so a ZST panics, and `ptr: chunk.cast()`
ignores `align_of::<T>()`, so an over-aligned `T` is misaligned — the same pair of
defects C3.9 fixed in `CacheAlignedVec`.
