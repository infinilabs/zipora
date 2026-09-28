# Stage 7 (`C6 containers/` + Phase D `D4`) Review & Remediation Ledger (`src/containers/`)

## Scope & Methodology

Stage 7 audited all 22 source files in `src/containers/` (19,311 LOC across `fast_vec/*`, `uint_vec_min0.rs`, `zip_int_vec.rs`, and `specialized/*`) against the `plan.md` Phase C (`C6 containers/`) and Phase D (`D4`) checklists:
1. Safe API soundness and `unsafe` invariant verification (zero UB under Miri across `FastVec`, `ValVec32`, `SmallMap`, `GoldHashIdx`, `MinimalSso`, `AutoGrowCircularQueue`, `FixedCircularQueue`, `VecTrbSet`/`VecTrbMap`, `LruMap`, `ConcurrentLruMap`, and bit-packed integer vectors).
2. Exception safety (`Drop` panic unwind without double-drop or leaked allocations), ZST (`size_of::<T>() == 0`) slice and pointer handling, high-alignment (`align_of::<T>() > 64`) support, and `u32` / `usize` overflow guards.
3. Bit-packed integer vector (`UintVecMin0`, `ZipIntVec`, `IntVec`, `UintVector`) correctness at bit widths `0..=64` (including `59..=64`-bit straddling reads/writes, near-`usize::MAX` domains, `BlockBased` non-zero `sample_min` and wide offsets, and full-sequence sortedness checks before delta compression).
4. Closure of all **6** §8.5 precondition violations in `src/containers/fast_vec/mod.rs` (`max_precondition_violations: 6 -> 0` across the entire crate!), 100% `// SAFETY:` documentation coverage in `src/containers/` (**211/211 = 100.0%**, `max_undocumented: 28 -> 24`), closure of `D4` (`EasyHashMap` iterators), and elimination of all **24** API honesty markers in `src/containers/` (`max_markers: 95 -> 70`, **0 remaining in `src/containers/`**).

---

## Findings & Remediations

### C6.1 — `FastVec::copy_from_slice_fast` and `extend_from_slice_fast` read uninitialized padding bytes as `[u8]` (`UB` on padded types) — CRITICAL

**Finding (`src/containers/fast_vec/mod.rs:845-875, 922-975`).**
- When `T: Copy` had alignment padding (e.g., `(u8, u64)` where bytes `1..8` are uninitialized padding), `copy_from_slice_fast` and `extend_from_slice_fast` cast `slice.as_ptr() as *const u8` and constructed `std::slice::from_raw_parts(..., total_bytes)` to pass to `fast_copy(src_bytes, dst_bytes)`.
- Constructing a `&[u8]` over uninitialized struct padding bytes is immediate Undefined Behavior in Rust (caught deterministically by Miri).

**Test.**
- `containers::fast_vec::tests::test_c6_1_padded_copy_and_extend_fast_no_uninit_ub` in `src/containers/fast_vec/tests.rs`

**Fix (`f40a426`).**
- Replaced the `[u8]` slice cast with `std::ptr::copy_nonoverlapping(slice.as_ptr(), dst_ptr, slice.len())`, which copies `T` elements directly (preserving uninitialized padding bytes without materializing a `&[u8]`) and lowers to `llvm.memcpy` for SIMD/vectorized copy.

---

### C6.2 — `FastVec::extend` trusted `iterator.size_hint().0` without bounds-checking the write loop against `self.capacity` (`heap buffer overflow UB` on dishonest `Iterator`) — CRITICAL

**Finding (`src/containers/fast_vec/mod.rs:483-507`).**
- `FastVec::extend` reserved `lower_bound = iter.size_hint().0` and then wrote every yielded item via `unsafe { ptr::write(self.data.as_ptr().add(self.len), item) }` without checking `self.len == self.capacity`.
- Because `Iterator::size_hint` is an safe trait method that is not `unsafe` to implement inaccurately, an iterator returning `size_hint() == (0, None)` while yielding elements caused an out-of-bounds heap write from 100% safe Rust.

**Test.**
- `containers::fast_vec::tests::test_c6_2_extend_dishonest_exact_size_iterator_no_oob_write` in `src/containers/fast_vec/tests.rs`

**Fix (`f40a426`).**
- Added an in-loop capacity check (`if self.len == self.capacity { self.grow_amortized(1)?; }`) before every `ptr::write` in `FastVec::extend`.

---

### C6.3 — `FastVec::clear`, `FastVec::resize`, `FastVec::resize_with`, and `ValVec32::clear` left `self.len` unchanged while dropping elements (`double-drop UB` on panicking `Drop`) — HIGH

**Finding (`src/containers/fast_vec/mod.rs:367-420, 444-465`, `src/containers/specialized/valvec32.rs:468-478`).**
- `FastVec::clear()`, `FastVec::resize()`, `FastVec::resize_with()`, and `ValVec32::clear()` dropped elements in `0..self.len` (or `new_len..self.len`) while keeping `self.len` at its old value until after the loop completed.
- If any element's `Drop::drop` panicked, stack unwinding ran `Drop` on the container, which iterated `0..self.len` again and double-dropped the already-dropped elements.

**Tests.**
- `containers::fast_vec::tests::test_c6_3_clear_and_resize_panicking_drop_no_double_drop` in `src/containers/fast_vec/tests.rs`
- `containers::specialized::valvec32::tests::test_c6_3_valvec32_clear_panicking_drop_no_double_drop` in `src/containers/specialized/valvec32.rs`

**Fix (`f40a426`).**
- Set `self.len = 0` (or `self.len = new_len`) before calling `ptr::drop_in_place(ptr::slice_from_raw_parts_mut(...))`, so unwinding sees the already-truncated length and never double-drops an element.

---

### C6.4 — `ValVec32` ZST slice overflow in `as_slice`/`as_mut_slice`, `u32` truncation in `from_vec` / `extend_from_slice` / `resize`, and `align > 64` layout panic — HIGH

**Finding (`src/containers/specialized/valvec32.rs`).**
1. For zero-sized types (`size_of::<T>() == 0`), `ValVec32::new()` left `self.data = NonNull::dangling()` and `self.cap = 0`, while `push` incremented `self.len` without growing `self.cap`. However, `as_slice()` and `as_mut_slice()` checked `if self.len == 0 || self.cap == 0 { &[] }` — returning an empty slice `&[]` even when `self.len > 0` for a ZST `ValVec32<()>`! Consequently, `PartialEq`, `Ord`, `Hash`, `Debug`, and `Index` (which delegate to `as_slice()`) saw `len == 0` while `vec.len()` returned `N`.
2. `ValVec32::from_vec`, `extend_from_slice`, and `resize` cast lengths/capacities via `as u32` without checking `u32::MAX`, risking 32-bit truncation of `len` on 64-bit targets.
3. `ValVec32` and `FastVec` assumed `align_of::<T>() <= 64` in their cache-aligned allocator paths.

**Tests.**
- `containers::specialized::valvec32::tests::test_c6_4_valvec32_zst_slice_eq_and_u32_overflow` in `src/containers/specialized/valvec32.rs`
- `containers::specialized::valvec32::tests::test_c6_align_4096_valvec32` in `src/containers/specialized/valvec32.rs`
- `containers::fast_vec::tests::test_c6_align_4096_fast_vec` in `src/containers/fast_vec/tests.rs`

**Fix (`f40a426`).**
- Removed the bogus `|| self.cap == 0` guard in `ValVec32::as_slice` and `as_mut_slice` (`NonNull::dangling()` is always valid for `from_raw_parts` of any ZST length, or `len == 0` for non-ZSTs).
- Added explicit `u32::try_from` bounds checks in `from_vec`, `extend_from_slice`, `resize`, and `reserve`.
- Used `.max(align_of::<T>())` when computing allocation alignment in both `FastVec` and `ValVec32`.

---

### C6.5 — `FastVec::resize` bypassed `Clone::clone()` via `memset` for 1-byte non-`Copy` types, and `copy_from_slice` leaked tail elements when shrinking — HIGH

**Finding (`src/containers/fast_vec/mod.rs:380-405, 830-860`).**
1. `FastVec::resize` used `if size_of::<T>() == 1 { ptr::write_bytes(..., value_byte, count) }` for any 1-byte `T: Clone`, even if `T` was **not** `Copy` and had custom `Clone` / `Drop` logic (e.g., `struct TrackedByte(u8)` that increments a refcount in `Clone` and decrements it in `Drop`), creating bitwise duplicates without calling `Clone::clone()` and causing N drops for 1 construction!
2. `FastVec::copy_from_slice` overwrote `0..src.len()` and set `self.len = src.len()` without dropping existing elements in `src.len()..old_len` when `src.len() < self.len`, and without dropping the old `0..src.len()` elements before `ptr::copy_nonoverlapping`!

**Test.**
- `containers::fast_vec::tests::test_c6_5_resize_1byte_non_copy_calls_clone_and_copy_from_slice_shrink` in `src/containers/fast_vec/tests.rs`

**Fix (`f40a426`).**
- Removed the unsound `size_of::<T>() == 1` `write_bytes` shortcut in `FastVec::resize` (which has only `T: Clone`, not `T: Copy`).
- Updated `FastVec::copy_from_slice` (`T: Copy`) and `truncate` so shrinking cleanly updates `self.len`.
- Closed all **6** §8.5 precondition violations in `src/containers/fast_vec/mod.rs` (`max_precondition_violations: 6 -> 0`).

---

### C6.6 — `UintVecMin0::get2` wrapped `idx + 1` on `usize::MAX`, and `UintVecMin0::fast_get` overflowed `bits * idx` — HIGH

**Finding (`src/containers/uint_vec_min0.rs:344-420`).**
- `UintVecMin0::get2(idx)` checked `assert!(idx + 1 < self.size)`, which wrapped `usize::MAX + 1` to `0` in release mode and passed the assertion whenever `self.size >= 1`, then called `self.get_unchecked(usize::MAX)`.
- `UintVecMin0::fast_get(data, bits, idx)` multiplied `bits * idx` without overflow checking and did not support `bits > 58`.

**Tests.**
- `containers::uint_vec_min0::tests::test_c6_6_get2_usize_max_panics` in `src/containers/uint_vec_min0.rs`
- `containers::uint_vec_min0::tests::test_c6_6_fast_get_overflow_returns_err` in `src/containers/uint_vec_min0.rs`

**Fix (`a1a5429`).**
- Checked `assert!(idx < self.size.saturating_sub(1), ...)` in `UintVecMin0::get2`.
- Used `bits.checked_mul(idx)` and `byte_idx.checked_add(8)` in `UintVecMin0::fast_get`, returning `Err(ZiporaError::invalid_data(...))` on overflow or out-of-bounds access.

---

### C6.7 — `UintVecMin0` panicked with `1u8 << 8` shift overflow on `59..=64`-bit values, rejected `bits > 58` in `get`/`get2`/`fast_get`, and left stale bits on `resize` shrink — CRITICAL

**Finding (`src/containers/uint_vec_min0.rs:280-465, 595-645`).**
1. `UintVecMin0::set_uint_bits` wrote byte chunks of width `chunk_bits = remaining_bits.min(8 - bit_in_byte)` and computed `let mask = ((1u8 << chunk_bits) - 1) << bit_in_byte`. Whenever `bit_in_byte == 0` and `remaining_bits >= 8` (which occurs for every multi-byte slow-path write, i.e., `59..=64` bits), `chunk_bits == 8` and `1u8 << 8` panicked with shift overflow in debug builds!
2. `get`, `get_unchecked`, `get2`, `get2_unchecked`, and `fast_get` asserted `self.bits <= 58` and panicked on `59..=64`-bit vectors even though `with_bits` and `set` allowed `bits <= 64`.
3. `UintVecMin0::resize` shrank `self.size` without zeroing truncated bits in `new_size..old_size`, so subsequent `resize` growth exposed stale non-zero bits instead of `0`.

**Tests.**
- `containers::uint_vec_min0::tests::test_c6_7_wide_bits_59_to_64_roundtrip` in `src/containers/uint_vec_min0.rs`
- `containers::uint_vec_min0::tests::test_c6_7_resize_shrink_and_regrow_zeros_truncated_bits` in `src/containers/uint_vec_min0.rs`

**Fix (`a1a5429`).**
- Fixed `set_uint_bits` mask computation (`if chunk_bits == 8 { 0xFF } else { ((1u8 << chunk_bits) - 1) << bit_in_byte }`).
- Added `get_uint_bits` slow path for `59..=64` bits in `get`, `get_unchecked`, `get2`, `get2_unchecked`, and `fast_get`.
- Zeroed truncated elements in `UintVecMin0::resize` before shrinking `self.size`.

---

### C6.8 — `ZipIntVec::set` and `max_val` overflowed `self.min_val + self.uintmask` near `usize::MAX` — HIGH

**Finding (`src/containers/zip_int_vec.rs:195-210, 290-305`).**
- When `ZipIntVec::new(usize::MAX - 2, usize::MAX)` was constructed, `range = 2`, `bits = 2`, and `uintmask = 3`.
- `ZipIntVec::set` asserted `val <= self.min_val + self.uintmask`, and `max_val()` computed `self.min_val + self.uintmask`, both of which overflowed `(usize::MAX - 2) + 3 = usize::MAX + 1`!

**Test.**
- `containers::zip_int_vec::tests::test_c6_8_near_usize_max_and_full_64bit_range` in `src/containers/zip_int_vec.rs`

**Fix (`a1a5429`).**
- Checked `(val - self.min_val) <= self.uintmask` after verifying `val >= self.min_val` in `ZipIntVec::set`, and used `self.min_val.saturating_add(self.uintmask)` in `ZipIntVec::max_val()`.

---

### C6.9 — `IntVec` sub-sampled sortedness check caused delta underflow on unsorted data, `BlockBased` discarded `sample_min` and truncated block offsets, and 9-byte straddling bit reads truncated high bits — CRITICAL

**Finding (`src/containers/specialized/int_vec.rs`, `src/containers/specialized/uint_vector.rs`).**
1. `IntVec::fast_sorted_check` and `analyze_delta_bulk` only inspected every `(len / 16)`-th sample for `len >= 32`. When an input of `len >= 32` was sorted at the 16 sample points but unsorted between them, `IntVec` selected `CompressionStrategy::Delta` and panicked with unsigned subtraction underflow `(val - prev_val)` inside `compress_delta`.
2. `CompressionStrategy::BlockBased` computed `sample_min` during construction, subtracted `sample_min` from every block sample, and then **discarded `sample_min`** while hardcoding `offset_width: 8, sample_width: 4` in the returned variant! Any `BlockBased` vector with `sample_min != 0` or block bit offsets `>= 256` returned completely corrupted values from `get()`.
3. When reading a `58..=64`-bit value at a non-zero bit offset (`bit_offset + num_bits > 64`), `IntVec::read_bits` loaded at most 8 bytes into a `u64` and shifted right by `bit_offset`, losing the top `bit_offset` bits in the 9th byte.
4. `UintVector::read_bits_fast` and `read_bits_slow` did not guard `bits == 0` (`1u64 << 0` / unallocated buffer) or out-of-bounds bit offsets.

**Tests.**
- `containers::specialized::int_vec::tests::test_c6_9_unsorted_between_sample_points_does_not_underflow` in `src/containers/specialized/int_vec.rs`
- `containers::specialized::int_vec::tests::test_c6_9_block_based_preserves_sample_min_and_wide_offsets` in `src/containers/specialized/int_vec.rs`
- `containers::specialized::int_vec::tests::test_c6_9_nine_byte_straddle_read_bits_roundtrip` in `src/containers/specialized/int_vec.rs`

**Fix (`a1a5429`).**
- Verified full-sequence monotonicity in `fast_sorted_check` and `analyze_delta_bulk`.
- Stored `sample_min` as an 8-byte little-endian prefix in `BlockBased::index_data` and preserved the actual `offset_width` and `sample_width` in `CompressionStrategy::BlockBased`.
- Widened `IntVec::read_bits` to assemble up to 16 bytes into a `u128` before shifting by `bit_offset`, supporting 9-byte straddling reads up to 64 bits.
- Added `bits == 0` and bounds guards in `UintVector::read_bits_fast` and `read_bits_slow`.

---

### C6.10 — `SmallMap` SIMD key search (`u8`, `u32`, `i32`) did not mask inactive lanes, falsely matching key `0` and reading uninitialized `MaybeUninit<V>` (`UB`) — CRITICAL

**Finding (`src/containers/specialized/small_map.rs:830-908`).**
- In `OptimizedSearch::find_optimized` for `u8`, `u32`, and `i32`, keys were copied into a zero-initialized stack array (`[0u8; 8]` or `[0u32; 4]`) of length `len` (`1..8`), loaded into an XMM register, and compared against `_mm_set1_epi8(*self)` / `_mm_set1_epi32(*self)`.
- Because the resulting `_mm_movemask_epi8` (16 bits) or `_mm_movemask_ps` (4 bits) was **not masked to `0..len`**, searching for key `0` when `0` was absent always matched the zero-padded inactive lanes at index `len`!
- `SmallMap<u8, V>::get_fast(&0)` then executed `unsafe { values[index].assume_init_ref() }` on an uninitialized slot `index == len` (immediate UB, or out-of-bounds panic when `len == 8` because bit 8 of `_mm_movemask_epi8` was set!).

**Test.**
- `containers::specialized::small_map::tests::test_c6_10_simd_zero_key_no_false_positive` in `src/containers/specialized/small_map.rs`

**Fix (`2ba0448`).**
- Masked the SIMD comparison bitmask with `(1u32 << active_lanes) - 1` in `u8::find_optimized` and `u32::find_optimized` (which `i32::find_optimized` delegates to).

---

### C6.11 — `MinimalSso::as_str` called `from_utf8_unchecked` on arbitrary bytes from `from_bytes` (`UB`), and `SortableStrVec::push_str` silently truncated strings `> 65,535` bytes — CRITICAL

**Finding (`src/containers/specialized/minimal_sso.rs:230-241`, `src/containers/specialized/sortable_str_vec.rs:195-245, 340-360`).**
1. `MinimalSso::from_bytes(&[u8])` is a safe public constructor accepting arbitrary non-UTF-8 byte slices (e.g., `&[0xFF, 0xFE]`), while `MinimalSso::as_str(&self) -> &str` is a safe public method that called `unsafe { std::str::from_utf8_unchecked(self.as_bytes()) }` — immediate Undefined Behavior from 100% safe Rust.
2. `SortableStrVec::push_str` packed string offset (`40` bits) and length (`16` bits, max `65,535`) into a `CompactEntry` via `as u16` without checking `s.len() <= CompactEntry::MAX_LENGTH`, silently truncating strings `>= 64 KiB`, never updated `self.stats`, and could divide by zero in `block_binary_search` when `self.block_size == 0`.

**Tests.**
- `containers::specialized::minimal_sso::tests::test_c6_11_as_str_rejects_invalid_utf8_without_ub` in `src/containers/specialized/minimal_sso.rs`
- `containers::specialized::sortable_str_vec::tests::test_c6_11_oversized_string_rejected_and_stats_updated` in `src/containers/specialized/sortable_str_vec.rs`

**Fix (`2ba0448`).**
- Replaced `unsafe { std::str::from_utf8_unchecked(...) }` in `MinimalSso::as_str` with `std::str::from_utf8(self.as_bytes())` (panicking cleanly if non-UTF-8 bytes were loaded via `from_bytes`, with zero UB).
- Validated `s.len() <= CompactEntry::MAX_LENGTH` and `offset <= CompactEntry::MAX_OFFSET` in `SortableStrVec::push_str`, updated `self.stats` on push/sort/clear/truncate, and clamped `self.block_size.max(1)` in `block_binary_search`.

---

### C6.12 — `GoldHashIdx` buffer overflow on undersized custom pool & leaked `Drop` values, and `LruMap` / `ConcurrentLruMap` capacity loss on `clear()`, 0-shard div-by-zero, and undocumented `unsafe impl Send/Sync` — CRITICAL

**Finding (`src/containers/specialized/gold_hash_idx.rs`, `lru_map.rs`, `concurrent_lru_map.rs`).**
1. `GoldHashIdx::with_pool(capacity, pool)` accepted any `SecureMemoryPool` without checking that `ptr.size() >= size_of::<V>()` and alignment `>= align_of::<V>()`, writing `V` past the end of the pooled chunk via `ptr::write(ptr.as_ptr() as *mut V, value)` (`heap buffer overflow UB`).
2. `GoldHashIdx::free_value`, `clear`, and `Drop` dropped the `SecurePooledPtr` (returning the raw byte chunk to the pool) without calling `ptr::drop_in_place(ptr.as_ptr() as *mut V)`, leaking every `V`'s `Drop` glue, and `resize_to` re-allocated new value slots without freeing the old ones.
3. `LruMap::clear()` cleared `free_nodes` instead of repopulating `0..self.capacity`, permanently reducing the `LruMap`'s usable capacity to the number of entries that happened to be occupied at `clear()` time! Furthermore, `LruMap::put` had a TOCTOU race between its read-lock duplicate check and write-lock insertion, creating duplicate nodes and leaking the old node index when two threads concurrently `put` the same new key.
4. `ConcurrentLruMap::with_shards(capacity, 0)` panicked with division-by-zero (`capacity / 0`), `capacity < shard_count` truncated `shard_capacity = capacity / shard_count` to `0` (causing every `put` to fail with `"Cannot evict from empty cache"`), and `keys()` / `rebalance()` were unimplemented stubs with `[for now]` honesty markers.
5. `lru_map.rs:876, 884` and `concurrent_lru_map.rs:465, 473` had 4 undocumented `unsafe impl Send/Sync` blocks.

**Tests.**
- `containers::specialized::gold_hash_idx::tests::test_c6_12_undersized_pool_rejected_and_values_dropped` in `src/containers/specialized/gold_hash_idx.rs`
- `containers::specialized::lru_map::tests::test_c6_12_clear_preserves_full_capacity_and_keys` in `src/containers/specialized/lru_map.rs`
- `containers::specialized::concurrent_lru_map::routing_regression_tests::test_c6_12_zero_shards_small_capacity_and_keys` in `src/containers/specialized/concurrent_lru_map.rs`

**Fix (`2ba0448`).**
- Validated `ptr.size() >= size_of::<V>()` and pointer alignment in `GoldHashIdx::allocate_pooled_value`, added `drop_pooled_value` + `Drop for GoldHashIdx`, and updated `resize_to` to rehash existing buckets in-place without duplicating value slots.
- Repopulated `(0..self.capacity).rev()` in `LruMap::clear()`, reset node key/value to `Default` in `LruNode::reset`, re-checked `index.get(&key)` after acquiring the write lock in `LruMap::put`, and added `LruMap::keys()`.
- Guarded `shard_count == 0` in `ConcurrentLruMap`, used `capacity.div_ceil(shard_count).max(1)` for per-shard capacity, implemented `ConcurrentLruMap::keys()` and `rebalance()`, and documented all 4 `unsafe impl Send/Sync` blocks (`max_undocumented: 28 -> 24`).

---

### C6.13 & D4 — `AutoGrowCircularQueue` / `FixedCircularQueue` full-buffer wrap bug (`head == tail`), `VecTrb` fixed 64-depth `PathStack` panic, `ZoSortedStrVec` interior NUL & duplicate `range`, `EasyHashMap` `retain` & iterators (`D4`), and `HashStrMap::remove` stats — HIGH

**Finding (`circular_queue.rs`, `vec_trb.rs`, `zo_sorted_str_vec.rs`, `easy_hash_map.rs`, `hash_str_map.rs`).**
1. In `AutoGrowCircularQueue::clear`, `AutoGrowCircularQueue::fmt`, `AutoGrowCircularQueue::clone`, and `FixedCircularQueue::fmt`, when the ring buffer was completely full (`len == capacity`, reachable via `push_bulk` on `AutoGrowCircularQueue` or `push_back` on `FixedCircularQueue`), `head == tail`. Because those methods branched on `if head <= tail`, a full queue took the contiguous branch `head..tail` (`head..head`, an empty range!), leaking all `len` elements on `clear()`/`Drop`, cloning 0 elements in `clone()`, and formatting as `[]` in `Debug::fmt`!
2. `vec_trb.rs` (`VecTrbSet` / `VecTrbMap`) used a fixed-size `[(u32, bool); 64]` `PathStack` while `remove` performed an un-rebalanced BST splice, so interleaved inserts and removals could push tree depth past 64 and panic with an out-of-bounds array write.
3. `ZoSortedStrVec::from_sorted_strings` accepted strings containing interior `'\0'` bytes even though `get()` uses `'\0'` as the string terminator, and `range(start, end)` used `binary_search` instead of `lower_bound`, skipping duplicate boundary strings.
4. `EasyHashMap::retain` passed `&mut value_copy` to the predicate closure `f(k, &mut value_copy)` and then discarded `value_copy`, silently losing in-place value mutations, while `iter()`, `keys()`, `values()`, and `IntoIterator` were commented out (`D4`).
5. `HashStrMap::remove` removed the entry from the inner `HashMap` without decrementing `self.unique_keys`.

**Tests.**
- `containers::specialized::circular_queue::tests::test_c6_13_full_autogrow_circular_queue_drop_clone_and_debug` in `src/containers/specialized/circular_queue.rs`
- `containers::specialized::vec_trb::tests::test_c6_13_vec_trb_deep_interleaved_insert_remove` in `src/containers/specialized/vec_trb.rs`
- `containers::specialized::zo_sorted_str_vec::tests::test_c6_13_zo_sorted_str_vec_rejects_interior_nul_and_range_duplicates` in `src/containers/specialized/zo_sorted_str_vec.rs`
- `containers::specialized::easy_hash_map::tests::test_iterators` in `src/containers/specialized/easy_hash_map.rs`
- `containers::specialized::hash_str_map::tests::test_c6_13_hash_str_map_remove_updates_unique_keys` in `src/containers/specialized/hash_str_map.rs`

**Fix (`2ba0448`).**
- Changed `if head <= tail` to `if head < tail` (after empty checks) in `AutoGrowCircularQueue::clear`, `fmt`, `clone`, and `FixedCircularQueue::fmt`.
- Added an overflow `Vec<(u32, bool)>` fallback to `PathStack` in `vec_trb.rs` (keeping zero heap allocation for depth `<= 64` while supporting arbitrary depths without panicking) and documented `unsafe fn data` / `data_mut`.
- Rejected interior `'\0'` bytes in `ZoSortedStrVec::from_sorted_strings` and used `lower_bound` in `ZoSortedStrVec::range`.
- Wrote back mutated retained values in `EasyHashMap::retain` and implemented `iter()`, `keys()`, `values()`, and `IntoIterator` on `EasyHashMap` (`D4`).
- Decremented `self.unique_keys` on successful removal in `HashStrMap::remove`.

---

## Audit & Gate Summary (`src/containers/`)

| Metric | Before Stage 7 | After Stage 7 | Status |
|--------|----------------|---------------|--------|
| `src/containers/` `unsafe` documentation coverage | 207 / 211 (98.1%) | **211 / 211 (100.0%)** | PASS |
| Crate-wide undocumented `unsafe` (`max_undocumented`) | 28 | **24** | RATCHETED |
| Crate-wide §8.5 precondition violations (`max_precondition_violations`) | 6 | **0** | RATCHETED |
| `src/containers/` API honesty markers | 24 | **0** | PASS |
| Crate-wide API honesty markers (`max_markers`) | 95 | **70** | RATCHETED |
| Crate-wide `.unwrap()` / `.expect()` (`max_unwraps` / `max_expects`) | 0 / 140 | **0 / 140** | PASS |
| Miri (`MIRIFLAGS="-Zmiri-disable-isolation -Zmiri-tree-borrows -Zmiri-ignore-leaks" cargo +nightly miri test --lib test_c6_`) | — | **26 / 26 pass (0 UB)** | PASS |
