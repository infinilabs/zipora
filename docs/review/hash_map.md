# Stage 8 (`C7 hash_map/` + Phase D `D5`) Review & Remediation Ledger (`src/hash_map/`)

## Scope & Methodology

Stage 8 audited all 11 source files in `src/hash_map/` (`8,342 -> 10,147` LOC across `zipora_hash_map.rs`, `zipora_hash_map_tests.rs`, `probe.rs`, `storage.rs`, `config.rs`, `gold_hash_map.rs`, `strategy_traits.rs`, `cache_locality.rs`, `hash_functions.rs`, `simd_string_ops.rs`, and `mod.rs`) against the `plan.md` Phase C (`C7 hash_map/`) and Phase D (`D5`, `D4` tombstone check) checklists:

| File | LOC (`a69815d` -> `13f06ce`) | `#[test]` (`a69815d` -> `13f06ce`) | Findings & Remediations |
|------|------------------------------|------------------------------------|-------------------------|
| `src/hash_map/zipora_hash_map.rs` | 797 -> 836 | 0 -> 0 | C7.1, C7.2 |
| `src/hash_map/zipora_hash_map_tests.rs` | 983 -> 1,239 | 51 -> 57 (+6) | C7.1, C7.2, D4 tombstone check |
| `src/hash_map/probe.rs` | 741 -> 791 | 0 -> 0 | C7.1, C7.2 |
| `src/hash_map/storage.rs` | 97 -> 53 | 0 -> 0 | C7.2 |
| `src/hash_map/config.rs` | 215 -> 215 | 0 -> 0 | Reviewed clean (carry-overs noted below) |
| `src/hash_map/gold_hash_map.rs` | 1,125 -> 1,242 | 28 -> 31 (+3) | C7.3 |
| `src/hash_map/strategy_traits.rs` | 774 -> 1,848 | 2 -> 14 (+12) | C7.4, D5 |
| `src/hash_map/cache_locality.rs` | 807 -> 882 | 9 -> 16 (+8, -1) | C7.2 (dead bucket types removed), C7.5 |
| `src/hash_map/hash_functions.rs` | 1,893 -> 2,070 | 21 -> 24 (+3) | C7.6 |
| `src/hash_map/simd_string_ops.rs` | 646 -> 709 | 10 -> 11 (+1) | C7.6 |
| `src/hash_map/mod.rs` | 264 -> 262 | 6 -> 6 | C7.4 / D5 re-exports |
| **Total (`src/hash_map/`)** | **8,342 -> 10,147** | **127 -> 159 (+33 new, -1 removed)** | **C7.1–C7.6 + D5** |

Conventions in this ledger:
- **Finding** line numbers refer to the Stage 8 parent commit (`a69815d`).
- **Fix** and **Test** line numbers refer to the final code tree (`13f06ce`). Every file except `src/hash_map/cache_locality.rs` changes in one fix commit only (`cache_locality.rs` loses the dead `CacheOptimizedBucket` / `BucketMetadata` types in `baea270` and receives the `C7.5` fixes in `79e893a`).
- Cargo profiles: the `test` profile compiles with `opt-level = 1`, `debug = true`, `debug-assertions = true`, and `overflow-checks = true`; the `release` profile compiles with `opt-level = 3`, `lto = "thin"`, `codegen-units = 16`, `debug-assertions = false`, and `overflow-checks = false`.

---

## Findings & Remediations

### C7.1 — `ZiporaHashMap` `SmallInline` fallback table never grew past 32 slots, migration left stale `occupied` counts, and `capacity()` added 16 dead inline slots — HIGH

**Finding (`src/hash_map/probe.rs:145-150, 244-272, 305-308`, `src/hash_map/zipora_hash_map.rs:235-245, 442-450`).**
1. When a `SmallInline` map exceeded its 16 inline slots, `insert_small_inline` migrated all 16 entries into a 32-slot `HashMapStorage::Standard` table stored in `fallback` (`probe.rs:305-308`). Subsequent inserts into the migrated map (`probe.rs:254-271`) called `Self::insert_standard` directly without running the load-factor check (`zipora_hash_map.rs:235-245` matched only top-level `HashMapStorage::Standard`), and on `Err((key, value))` returned `Err(ZiporaError::invalid_state("Hash table full in SmallInline fallback storage"))` (`probe.rs:263-267`) without resizing. (`resize_storage` at `probe.rs:145-150` also had a catch-all `_ => Err("Resize not supported for this storage type")`, though `insert` never called `resize_storage` on `SmallInline`.) Any 33rd live entry in a `SmallInline` map failed.
2. `ZiporaHashMap::insert` incremented `self.occupied` on every fresh-key insert (`zipora_hash_map.rs:321-324`), including inline inserts, while inline `remove` freed its slot outright via bitmask clear (`inline_data.occupied &= !(1 << i)`) without decrementing `self.occupied`, and migration to the 32-slot fallback table did not reset `self.occupied` to the 16 migrated entries. Once the fallback table gained a load check, prior inline insert/remove churn left `self.occupied` arbitrarily inflated and forced an immediate spurious resize after migration.
3. `ZiporaHashMap::capacity()` (`zipora_hash_map.rs:442-450`) returned `16 + fb.capacity()` (e.g. `48` right after migration) even though all 16 inline slots were moved into `fallback` on migration and never used again.

**Tests (`src/hash_map/zipora_hash_map_tests.rs`).**
- `hash_map::zipora_hash_map::tests::test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones` (`zipora_hash_map_tests.rs:986-1048`)
- `hash_map::zipora_hash_map::tests::test_c7_1_inline_churn_does_not_force_a_resize_after_migration` (`zipora_hash_map_tests.rs:1051-1072`)
- `hash_map::zipora_hash_map::tests::test_c7_1_migrated_fallback_load_check_counts_the_moved_entries` (`zipora_hash_map_tests.rs:1075-1098`)
- `hash_map::zipora_hash_map::tests::test_c7_iteration_skips_tombstones_in_every_storage_state` (`zipora_hash_map_tests.rs:1199-1239`, `D4` tombstone coverage across `Standard`, pre-migration `SmallInline`, and post-migration `SmallInline`)

**RED on parent (`a69815d`).**
- `test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones`: **FAIL** (debug & release) at `zipora_hash_map_tests.rs:994` — `InvalidState { message: "Hash table full in SmallInline fallback storage" }`.
- `test_c7_1_inline_churn_does_not_force_a_resize_after_migration`: **FAIL** (debug & release) at `zipora_hash_map_tests.rs:1068` — `assertion left == right failed: left: 48, right: 32` (`capacity()` bug fires first on `a69815d`; the `*occupied = *len` migration reset is guarded by mutant `M5`).
- `test_c7_1_migrated_fallback_load_check_counts_the_moved_entries`: **FAIL** (debug & release) at `zipora_hash_map_tests.rs:1091` — `assertion left == right failed: left: 48, right: 32` (`capacity()` bug fires first on `a69815d`; counting the 16 moved entries in `*occupied` is guarded by mutant `M5b`).
- `test_c7_iteration_skips_tombstones_in_every_storage_state`: **FAIL** (debug & release) at `zipora_hash_map_tests.rs:1230` — `InvalidState { message: "Hash table full in SmallInline fallback storage" }` during the `10..60` post-migration inserts (the `Standard` and pre-migration `SmallInline` blocks earlier in the test pass on `a69815d`).

**Fix (`baea270`).**
- Extracted `ZiporaHashMap::resize_standard_entries` (`probe.rs:45-143`; the in-place compaction vs doubling rule at `probe.rs:62` is pre-existing from round 5) and replaced the catch-all in `resize_storage` with explicit arms for every `HashMapStorage` state (`probe.rs:146-187`; note that `insert` only calls `resize_storage` for top-level `Standard` storage, so the `SmallInline` arms of `resize_storage` are defensive and unreachable from the public API).
- In `insert_small_inline` (`probe.rs:289-328`), a migrated `SmallInline` map now runs the same occupancy load check `(capacity as f64 * load_factor) as usize` before `insert_standard` and performs one `resize_standard_entries` + retry if `insert_standard` still returns `Err((retry_key, retry_val))` (needed when `load_factor >= 1.0`, as in `ZiporaHashMapConfig::small_inline`, which sets `load_factor: 1.0`).
- On inline-to-fallback migration (`probe.rs:422-427`), `*occupied` is reset to `*len` (`16`) before `insert` increments both `self.len` and `self.occupied` (`zipora_hash_map.rs:364-367`), so inline churn prior to migration does not inflate `occupied` (`V1` perf design: avoids touching `occupied` on pre-migration inline `remove`).
- `ZiporaHashMap::capacity()` (`zipora_hash_map.rs:484-505`) now reports `entries.capacity()` of the active fallback table after migration and `16` before migration. Documented the `occupied` invariant (`zipora_hash_map.rs:91-105, 286-288`).

---

### C7.2 — `ZiporaHashMap` accepted non-positive / non-finite `load_factor`, overflowed `next_power_of_two()` on large `initial_capacity`, and carried dead storage variants, undocumented `unsafe`, and 8 `.expect()` calls — MEDIUM

**Finding (`src/hash_map/zipora_hash_map.rs:144-187, 241`, `src/hash_map/storage.rs:8-97`, `src/hash_map/probe.rs:216, 229, 277, 402, 411, 496, 512, 541, 594, 599, 698`, `src/hash_map/cache_locality.rs:194-268, 640-649`).**
1. `with_config_and_hasher` did not validate `config.load_factor`. In `insert`, `let limit = (capacity as f64 * self.config.load_factor) as usize` (`zipora_hash_map.rs:241`) evaluated to `0` for `0.0`, negative values (including `-inf`), and `NaN`, so `self.occupied >= limit` (`0 >= 0`) triggered `resize_storage()` on every insert, doubling the table on every insert until allocation failed with `OutOfMemory` (e.g. `8 GiB` on the 27th insert). For `+inf`, `limit` saturated to `usize::MAX`, silently disabling the load-factor check so the table only grew at 100% occupancy.
2. `create_storage` computed `initial_capacity.next_power_of_two()` (`zipora_hash_map.rs:182`). For `initial_capacity > (1usize << (usize::BITS - 1))` (such as `usize::MAX`), `next_power_of_two()` panicked with integer addition overflow in debug/test builds and wrapped to `0` in release builds, silently constructing a 0-capacity table from `with_capacity(usize::MAX)`.
3. `HashMapStorage` (`storage.rs:8-97`) carried `#[allow(dead_code)] // strategy-placeholder variants, matched exhaustively in 9 arms` (`api_honesty` marker) along with never-constructed `CacheOptimized` and `StringOptimized` enum variants, 4 dead types (`StringArena`, `StringBucket`, `StringEntry`, `PrefixCacheEntry`), 9 `unreachable!("unimplemented strategies are rejected at construction")` match arms in `zipora_hash_map.rs` (`:306, 336, 380, 402, 425, 456, 575, 649, 707`), and the never-re-exported `CacheOptimizedBucket` / `BucketMetadata` types with 2 `unsafe impl Send/Sync` in `cache_locality.rs:194-268, 640-649` (note: `HashStorageStrategy::CacheOptimized`, `StringOptimized`, and `PoolAllocated` in `config.rs` were already rejected at construction with `Err(ZiporaError::not_supported(...))` at `zipora_hash_map.rs:205-219`, pre-existing).
4. `probe.rs` had 3 undocumented `unsafe` blocks (`probe.rs:277, 541, 698`) reading `inline_data._data[i].assume_init_ref()` and 8 `.expect(...)` calls (`probe.rs:216, 229, 402, 411, 496, 512, 594, 599`) on `entry.key` / `entry.value`.

**Tests (`src/hash_map/zipora_hash_map_tests.rs`).**
- `hash_map::zipora_hash_map::tests::test_c7_2_load_factor_that_is_not_positive_and_finite_is_replaced` (`zipora_hash_map_tests.rs:1101-1142`)
- `hash_map::zipora_hash_map::tests::test_c7_2_initial_capacity_with_no_power_of_two_is_an_error` (`zipora_hash_map_tests.rs:1145-1165`)
- Removed dead test `hash_map::cache_locality::tests::test_cache_optimized_bucket` (`cache_locality.rs:664-691` on `a69815d`) along with `CacheOptimizedBucket`.

**RED on parent (`a69815d`).**
- `test_c7_2_load_factor_that_is_not_positive_and_finite_is_replaced`: **FAIL** (debug & release) at `zipora_hash_map_tests.rs:1113` — `OutOfMemory { size: 8589934592 }` on `bad_lf = 0.0`.
- `test_c7_2_initial_capacity_with_no_power_of_two_is_an_error`: **FAIL** — in debug (`test` profile) panics at `core/src/num/mod.rs:1448` with `attempt to add with overflow`; in `--release` panics at `zipora_hash_map_tests.rs:1158` with `initial_capacity 18446744073709551615 must be rejected` because `usize::MAX.next_power_of_two()` wraps to `0` and returns `Ok`.

**Fix (`baea270`).**
- `with_config_and_hasher` (`zipora_hash_map.rs:167-184`) replaces any `config.load_factor` where `!(config.load_factor.is_finite() && config.load_factor > 0.0)` with `0.75` (values `>= 1.0` are kept so `small_inline`'s `1.0` continues to use all slots before growing).
- `create_storage` (`zipora_hash_map.rs:217-234`) uses `initial_capacity.checked_next_power_of_two()`, returning `Err(ZiporaError::out_of_memory(...))` when no power of two fits in `usize`.
- Added `# Hash flooding` (`zipora_hash_map.rs:43-52`) and `with_capacity` rounding (`zipora_hash_map.rs:137-141`) rustdoc sections.
- Deleted the never-constructed `HashMapStorage::CacheOptimized` and `StringOptimized` variants, their 4 placeholder structs and `#[allow(dead_code)]` marker in `storage.rs` (`97 -> 53` LOC), the 9 `unreachable!` arms in `zipora_hash_map.rs`, and the unexported `CacheOptimizedBucket` / `BucketMetadata` types + 2 `unsafe impl`s in `cache_locality.rs`.
- Added `// SAFETY:` comments to all 3 inline-slot `assume_init_ref()` blocks (`probe.rs:333, 591, 747`) and replaced all 8 `.expect(...)` calls in `probe.rs` with pattern matching / `Option` propagation (note: because `entry.key.is_some()` implies `entry.value.is_some()`, the `None` fallback paths on `entry.value` are unreachable defensive conversions).

---

### C7.3 — `GoldHashMap` fast iteration yielded deleted entries, `reserve` overflowed `self.len + additional`, and failed `insert` leaked `hash_cache` entries — HIGH

**Finding (`src/hash_map/gold_hash_map.rs:343-361, 397-404, 408-433, 494, 543-577, 703-712`).**
1. `GoldHashMap::iter()` was documented as `/// Default safe iteration (skips deleted entries)` (`gold_hash_map.rs:352`), yet delegated to `self.iter_with_strategy(self.config.default_iteration_strategy)`. When `default_iteration_strategy` was `IterationStrategy::Fast` (or when calling `iter_fast()`, whose doc warned `WARNING: May include deleted entries`, `gold_hash_map.rs:357-358`), `GoldHashMapIter::next` (`gold_hash_map.rs:704-712`) walked `self.map.entries` linearly without checking `entry.link != L::DELMARK` and yielded removed `(key, value)` pairs.
2. `GoldHashMap::reserve(&mut self, additional: usize)` (`gold_hash_map.rs:397-404`) computed `let new_capacity = self.len + additional;` without overflow checking (panicking in debug/test builds when `self.len > 0` and `additional == usize::MAX`, and wrapping to a silent no-op `Ok(())` in release builds), did not check `new_capacity` against `L::MAX`, and computed `(new_capacity as f32 / self.config.load_factor) as usize` in `f32` without guarding non-finite or `>= isize::MAX` values before `next_prime` / `vec![L::TAIL; new_buckets]`.
3. `GoldHashMap::allocate_slot` (`gold_hash_map.rs:408-433`) pushed `0` to `self.hash_cache` (`:429-431`) before `insert` checked `if entry_idx > L::MAX.as_usize()` (`:546-552`). Whenever an insert failed the `L::MAX` check, `self.entries` remained unchanged while `self.hash_cache` grew by 1 stale element per failed insert. In addition, `relink` (`:494`) and `insert` (`:575`) used `.expect("Capacity check above ensures this succeeds")`.

**Tests (`src/hash_map/gold_hash_map.rs`).**
- `hash_map::gold_hash_map::tests::test_c7_3_fast_iteration_skips_removed_entries` (`gold_hash_map.rs:1179-1200`)
- `hash_map::gold_hash_map::tests::test_c7_3_reserve_overflow_is_an_error` (`gold_hash_map.rs:1203-1217`)
- `hash_map::gold_hash_map::tests::test_c7_3_link_capacity_failure_keeps_hash_cache_in_step` (`gold_hash_map.rs:1220-1241`)

**RED on parent (`baea270`).**
- `test_c7_3_fast_iteration_skips_removed_entries`: **FAIL** (debug & release) at `gold_hash_map.rs:1165` (injected line; `1195` in `13f06ce`) — `assertion left == right failed: left: 3, right: 2` on the first loop iteration (`map.iter_fast()`; note that the second loop iteration `map.iter()` with `default_iteration_strategy: IterationStrategy::Fast` was not reached before the panic).
- `test_c7_3_reserve_overflow_is_an_error`: **FAIL** — in debug (`test` profile) panics at `gold_hash_map.rs:398` with `attempt to add with overflow`; in `--release` panics at `gold_hash_map.rs:1179` (`1209` in `13f06ce`) with `assertion failed: map.reserve(usize::MAX).is_err()` because `2 + usize::MAX` wraps to `1 <= self.max_load` and returns `Ok(())`.
- `test_c7_3_link_capacity_failure_keeps_hash_cache_in_step`: **FAIL** (debug & release) at `gold_hash_map.rs:1203` (`1233` in `13f06ce`) — `assertion left == right failed: left: 6, right: 4`.

**Fix (`664c3b5`).**
- `GoldHashMap::iter_with_strategy` (`gold_hash_map.rs:343-360`) downgrades `IterationStrategy::Fast` to `IterationStrategy::Safe` whenever `self.freelist_size > 0` (`deleted_count() > 0`), so neither `iter()` nor `iter_fast()` ever yields a deleted entry (`free_slot` at `:474-484` always marks `L::DELMARK` and increments `freelist_size`; updated docs at `:343-348, 367-376, 732-733`).
- `GoldHashMap::reserve` (`gold_hash_map.rs:411-433`) uses `self.len.checked_add(additional)`, rejects `new_capacity > L::MAX.as_usize().saturating_add(1)`, and computes `let scaled = (new_capacity as f64 / self.config.load_factor as f64).ceil()`, returning `Err(ZiporaError::resource_exhausted(...))` if `!scaled.is_finite() || scaled >= (isize::MAX as f64)` (note: the `f64`/`ceil` rounding improvement and the `!scaled.is_finite() || scaled >= isize::MAX` guard are defensive and not directly exercised by a dedicated test, and `rehash` still computes `self.max_load` in `f32`, so we do not claim a general no-rehash capacity guarantee).
- `allocate_slot` (`gold_hash_map.rs:437-470`) now returns `Result<usize>` and checks `idx > L::MAX.as_usize()` before reserving or pushing to `self.hash_cache`. Replaced both `.expect(...)` calls in `relink` (`:519-527`) and `insert` (`:578-585`) with `let Some(new_head) = L::from_usize(...) else { return Err(...) }` (unreachable defensive conversions after the `L::MAX` check).

---

### C7.4 & D5 — `RobinHoodStrategy` lost entries behind tombstones, duplicated keys on tombstone reuse, corrupted the table on `max_probe_distance` `Err`, underflowed on empty slices, never rehashed in `StandardStorageStrategy::resize_storage`, and carried 2 stub strategies (`D5`) — CRITICAL

**Finding (`src/hash_map/strategy_traits.rs:196-211, 298-498, 514-723`).**
1. **Tombstone / vacant conflation (`strategy_traits.rs:197-211, 325-333, 381-383, 420-431`).** `HashBucket::is_empty()` was `self.key.is_none()` and `mark_deleted()` set `self.flags |= 0x8000; self.key = None; self.value = None;`. Consequently:
   - `get` (`:381`) and `remove` (`:420`) stopped at the first tombstone because `bucket.is_empty()` returned `true`, making every live entry whose probe chain passed through a tombstone unreachable.
   - `insert` (`:325-333`) took the first tombstone as an empty slot without clearing `0x8000` (`flags` remained marked deleted!) and without continuing the probe scan to check whether the key already existed further down the cluster (creating duplicate keys), or respecting the tombstone's Robin Hood probe-distance invariant.
2. **Destructive `Err` after Robin Hood swaps & rejected updates (`strategy_traits.rs:316-350`).** `insert` checked `if probe_distance > config.max_probe_distance { return Err(...) }` (`:316`) before comparing keys (`:336`) and inside the swap loop (`:343-350`). Updating an existing key whose probe distance exceeded `config.max_probe_distance` failed with `Err`, and when a new insert exceeded `max_probe_distance` after displacing rich entries via `std::mem::swap`, returning `Err` dropped the currently displaced `(inserting_key, inserting_value)` on the floor, corrupting the table.
3. **Empty and non-power-of-two slice hazards (`strategy_traits.rs:307, 374, 414, 479`).** `let mask = buckets.len() - 1;` underflowed on an empty slice (`buckets.len() == 0`) — panicking in debug/test builds and indexing `buckets[pos]` out of bounds in release builds — and on non-power-of-two slice lengths (e.g. length `5`, `mask = 4`) `pos = (pos + 1) & mask` visited only a subset of slots (`0` and `4`).
4. **`backward_shift_delete` and `probe_stats` bugs (`strategy_traits.rs:456-498`).** `backward_shift_delete` (`:474-498`, which also required `K: Clone, V: Clone` and moved entries via `.clone()` + `clear()`) stopped at tombstones because `next_bucket.is_empty()` was `true` for tombstones, leaving later displaced entries unshifted. `probe_stats` (`:456-470`) divided `total_probe_distance` by `collision_count` instead of live entry count and hardcoded `variance: 0.0, // TODO: Calculate variance` (`:468`).
5. **`StandardStorageStrategy` bugs (`strategy_traits.rs:514-556`).** `create_storage` (`:514-526`) did not round capacities to a power of two (e.g. `3`, `5`, `17`), producing non-power-of-two bucket slices that broke mask probing; `resize_storage` (`:528-535`) only called `let _ = storage.resize_with(new_capacity, Default::default);` without rehashing existing entries (making every entry whose home slot `(hash as usize) & (new_len - 1)` changed unreachable, while ignoring resize errors); and `optimize_layout` (`:553-556`) was a no-op that never compacted tombstones.
6. **Phase D `D5` strategy stubs (`strategy_traits.rs:559-723`).** `CacheOptimizedStorageStrategy` (`:559-630`, with 3 `TODO`s at `:586, 605, 627`) and `SimdOptimizationStrategy` (`:632-723`, with 4 `TODO: Implement ...` no-op methods at `:677, 688, 702, 713`) were unimplemented stubs carrying 5 `api_honesty` markers (`:627, 677, 688, 702, 713`).

**Tests (`src/hash_map/strategy_traits.rs`).**
- `hash_map::strategy_traits::tests::test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`strategy_traits.rs:1019-1169`)
- `hash_map::strategy_traits::tests::test_c7_4_update_beyond_max_probe_distance_and_full_table` (`strategy_traits.rs:1172-1331`)
- `hash_map::strategy_traits::tests::test_c7_4_tombstone_reuse_keeps_later_entries_reachable` (`strategy_traits.rs:1334-1377`)
- `hash_map::strategy_traits::tests::test_c7_4_backward_shift_moves_tombstones_and_later_entries` (`strategy_traits.rs:1380-1412`)
- `hash_map::strategy_traits::tests::test_c7_4_probe_stats_match_live_layout` (`strategy_traits.rs:1415-1475`)
- `hash_map::strategy_traits::tests::test_c7_4_create_and_resize_storage_use_power_of_two_lengths` (`strategy_traits.rs:1478-1589`)
- `hash_map::strategy_traits::tests::test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`strategy_traits.rs:1618-1626`)
- `hash_map::strategy_traits::tests::test_c7_4_rehash_rejects_u16_probe_distance_overflow` (`strategy_traits.rs:1629-1679`)
- `hash_map::strategy_traits::tests::test_c7_4_rehash_over_65536_entries_keeps_every_entry` (`strategy_traits.rs:1682-1715`)
- `hash_map::strategy_traits::tests::test_max_linear_probe_distance_matches_robin_hood_simulation` (`strategy_traits.rs:1720-1766`, white-box unit test for the new `max_linear_probe_distance` helper; cannot compile on parent)
- `hash_map::strategy_traits::tests::test_c7_4_optimize_layout_drops_tombstones` (`strategy_traits.rs:1769-1823`)
- `hash_map::strategy_traits::tests::test_c7_4_non_power_of_two_slices_are_rejected` (`strategy_traits.rs:1826-1847`)

**RED on parent (`664c3b5`).**
- All 11 runnable `test_c7_4_*` tests **FAIL** on `664c3b5` in both debug (`test` profile) and `--release`:
  1. `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance`: debug panics at `strategy_traits.rs:307` (`attempt to subtract with overflow` on empty slice); release panics at `strategy_traits.rs:322` (`index out of bounds: the len is 0 but the index is 1`).
  2. `test_c7_4_update_beyond_max_probe_distance_and_full_table`: panics at `Exceeded maximum probe distance`.
  3. `test_c7_4_tombstone_reuse_keeps_later_entries_reachable`: panics with `left: None, right: Some(50)`.
  4. `test_c7_4_backward_shift_moves_tombstones_and_later_entries`: panics with `left: None, right: Some(13)`.
  5. `test_c7_4_probe_stats_match_live_layout`: panics with `left: None, right: Some(0)`.
  6. `test_c7_4_create_and_resize_storage_use_power_of_two_lengths`: panics with `left: [0, 1, 2, 3, 5, 17], right: [0, 1, 2, 4, 8, 32]`.
  7. `test_c7_4_robin_hood_matches_std_hashmap_under_collisions`: panics with `slot 5: stored probe distance left: 1, right: 9`.
  8. `test_c7_4_rehash_rejects_u16_probe_distance_overflow`: panics with `a rehash that needs a probe distance above u16::MAX must fail`.
  9. `test_c7_4_rehash_over_65536_entries_keeps_every_entry`: panics with `slot 1: stored probe distance left: 0, right: 131072`.
  10. `test_c7_4_optimize_layout_drops_tombstones`: panics with `optimize_layout must drop tombstones`.
  11. `test_c7_4_non_power_of_two_slices_are_rejected`: panics with `insert into a length-5 slice must fail`.

**Fix (`2944cf0`).**
- **Tombstone phantom-distance invariant (`strategy_traits.rs:263-277, 398-470`).** Added internal `is_vacant(bucket)` (`bucket.key.is_none() && (bucket.flags & DELETED_FLAG) == 0`) and `is_live(bucket)` (`bucket.key.is_some()`), while preserving the public `HashBucket::is_empty()` method signature and body (`self.key.is_none()`). When `remove` marks a bucket deleted (`mark_deleted()`), the bucket keeps its `probe_distance` as a phantom distance. Lookups (`rh_find` / `rh_find_from`, `:401-453`) stop on a truly vacant bucket or when `probe_distance > bucket.probe_distance` (valid across both live buckets and tombstones), and skip tombstones while continuing the probe chain.
- **Two-phase non-destructive `insert` (`strategy_traits.rs:479-647`).** Rejects empty or non-power-of-two `buckets` slices up front (`!buckets.len().is_power_of_two()`). If the key is already present (`rh_find_from` continuation from the first displacement/tombstone point), updates its value in place regardless of `config.max_probe_distance`. Otherwise, performs a read-only dry run of the Robin Hood displacement walk to verify that every displaced entry (and the new entry) fits within `config.max_probe_distance.min(u16::MAX as usize)` and that a vacant slot or reusable tombstone (`inserting_distance >= bucket.probe_distance`) exists before mutating any bucket. Only then executes the displacement walk (`rh_place` clears `DELETED_FLAG`). Relaxed `impl<K, V> CollisionResolutionStrategy<K, V> for RobinHoodStrategy` from `K: Hash + Eq + Clone, V: Clone` to `K: Hash + Eq` (`:472-474`).
- **`backward_shift_delete` and live-layout `probe_stats` (`strategy_traits.rs:362-396, 665-759`).** `backward_shift_delete` (`:725-759`) now shifts both live buckets and tombstones with `probe_distance > 0` backward via `std::mem::take` (no `Clone` bound) and updates `RobinHoodContext` counters (`live_entries`, `total_probe_distance`, `sum_squared_probe_distance`, `max_probe_distance`) via `#[inline]` helpers `record_placed`, `record_moved`, and `record_removed`. `probe_stats` (`:704-721`) computes mean and population variance $\mathbb{E}[d^2] - \mathbb{E}[d]^2$ over `context.live_entries`.
- **`StandardStorageStrategy` power-of-two sizing, rehashing `resize_storage`, and `optimize_layout` (`strategy_traits.rs:762-966`).** `create_storage` rounds non-zero capacities to a power of two (falling back to `prev_power_of_two` on allocation failure); `resize_storage` (`:934-940`) and `optimize_layout` (`:958-965`) rehash live entries via `rehash_bucket_storage` (`:819-889`), which pre-checks in $O(\text{new\_cap})$ time via `max_linear_probe_distance` (`:777-814`, run when `live > u16::MAX as usize + 1`) that no entry's Robin Hood probe distance in the new layout can exceed `u16::MAX`, leaving `storage` untouched on `Err`.
- **Phase D `D5` (`strategy_traits.rs`, `src/hash_map/mod.rs:64-71`).** Deleted the unimplemented `CacheOptimizedStorageStrategy` (`CacheOptimizedStorageConfig`) and `SimdOptimizationStrategy` (`SimdOptimizationConfig`, `SimdOptimizationContext`) stubs and removed their re-exports from `src/hash_map/mod.rs` (**5** `api_honesty` markers eliminated; **0** remaining in `strategy_traits.rs`).

---

### C7.5 — `NumaAllocator` zero-size `alloc_zeroed` UB & overflowing-layout panics, `HotColdSeparator` OOB `access` / underflowing `new` / no-op `rebalance`, `CacheLayoutOptimizer` zero bucket size, `CacheConsciousResizer` zero-chunk livelock, and `AccessPatternAnalyzer` zero-history panic — HIGH

**Finding (`src/hash_map/cache_locality.rs:311-372, 437-439, 487-521, 537, 585-596` at `a69815d`; lines shifted by `-74` in parent `2944cf0` after removing `CacheOptimizedBucket`).**
1. **`NumaAllocator::alloc_cache_aligned` / `alloc_on_node` / `dealloc` (`cache_locality.rs:311-372` at `a69815d`; `:237-298` at `2944cf0`).** `alloc_on_node`'s `# Safety` doc (`:311`) required `layout` to have non-zero size, but `alloc_cache_aligned<T>(count)`'s `# Safety` doc (`:331-332`) only required that `size_of::<T>() * count` not overflow `isize` and forwarded `count == 0` or ZST `T` (`size_of::<T>() == 0`) directly to `alloc_on_node(layout, ...)` -> `std::alloc::alloc_zeroed(layout)` (`:319`) with a zero-sized `Layout`, which is immediate Undefined Behavior in Rust (flagged by Miri: `creating allocation with size 0`). In addition, `size_of::<T>() * count` panicked on multiplication overflow in debug builds and hit `.expect("Layout invariant violated...")` (`:346, 368`) when `total_bytes > isize::MAX as usize` in release builds (widening `alloc_cache_aligned` to return null on overflow removes both panics and the 2 `.expect()` calls).
2. **`HotColdSeparator` (`cache_locality.rs:487-521` at `a69815d`; `:413-447` at `2944cf0`).**
   - `HotColdSeparator::new(capacity, hot_ratio)` computed `let hot_capacity = ((capacity as f64) * hot_ratio) as usize; let cold_capacity = capacity - hot_capacity;` (`:489-490`), panicking with subtraction underflow in debug builds (and `RawVec` capacity overflow in release builds) when `hot_ratio > 1.0`.
   - `HotColdSeparator::access(&self, index: usize)` (`:501-509`) executed `self.access_counts[index].fetch_add(1, Ordering::Relaxed);` before checking bounds (`new()` only calls `Vec::with_capacity`, so `access_counts.len()` is `0` on every freshly constructed separator — `access(0)` always panicked).
   - `HotColdSeparator::rebalance(&mut self)` (`:513-520`) only reset `access_counts` to `0` without ever swapping hot and cold elements.
3. **`CacheLayoutOptimizer::optimal_bucket_size` (`cache_locality.rs:437-439` at `a69815d`).** Computed `let cache_line_capacity = CACHE_LINE_SIZE / entry_size;`, which was `0` whenever `size_of::<(K, V)>() + 4 > 64`, returning an optimal bucket size of `0` for every cache level.
4. **`CacheConsciousResizer::new` (`cache_locality.rs:537` at `a69815d`).** Computed `let chunk_size = (L3_CACHE_SIZE / CACHE_LINE_SIZE).min(target_size / 16);`, which was `0` whenever `target_size < 16`, causing `resize_step` to loop forever without advancing `current_size`.
5. **`AccessPatternAnalyzer::new` (`cache_locality.rs:585-596` at `a69815d`).** `new(0)` allocated `access_history: vec![0; 0]`, causing the first `record_access` call (`self.access_history[self.history_pos] = address; self.history_pos = (self.history_pos + 1) % self.access_history.len();`) to panic with index out of bounds.

**Tests (`src/hash_map/cache_locality.rs`).**
- `hash_map::cache_locality::tests::test_c7_5_hot_cold_access_out_of_range_is_none` (`cache_locality.rs:782-791`)
- `hash_map::cache_locality::tests::test_c7_5_hot_cold_rebalance_promotes_a_frequently_accessed_cold_element` (`cache_locality.rs:794-815`)
- `hash_map::cache_locality::tests::test_c7_5_hot_cold_ratio_outside_unit_interval_is_clamped` (`cache_locality.rs:818-824`)
- `hash_map::cache_locality::tests::test_c7_5_optimal_bucket_size_is_at_least_one_for_entries_over_a_cache_line` (`cache_locality.rs:827-830`)
- `hash_map::cache_locality::tests::test_c7_5_resizer_with_target_below_16_makes_progress` (`cache_locality.rs:833-842`)
- `hash_map::cache_locality::tests::test_c7_5_access_pattern_analyzer_with_zero_history_records_accesses` (`cache_locality.rs:845-850`)
- `hash_map::cache_locality::tests::test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator` (`cache_locality.rs:853-869`)
- `hash_map::cache_locality::tests::test_c7_5_numa_allocator_overflowing_request_returns_null` (`cache_locality.rs:872-881`)

**RED on parent (`2944cf0`).**
- 7 of the 8 `test_c7_5_*` tests **FAIL** natively in both debug (`test` profile) and `--release` on `2944cf0`, and the 8th (`test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator`) **FAILs under Miri** on `2944cf0` (natively the system allocator returns a non-null pointer for size 0 rather than trapping):
  1. `test_c7_5_hot_cold_access_out_of_range_is_none`: panics at `cache_locality.rs:426` (`index out of bounds: the len is 0 but the index is 0`).
  2. `test_c7_5_hot_cold_rebalance_promotes_a_frequently_accessed_cold_element`: panics at `left: Some(10), right: Some(20)`.
  3. `test_c7_5_hot_cold_ratio_outside_unit_interval_is_clamped`: debug panics at `cache_locality.rs:414` (`attempt to subtract with overflow`); release panics in `alloc::raw_vec` (`capacity overflow`).
  4. `test_c7_5_optimal_bucket_size_is_at_least_one_for_entries_over_a_cache_line`: panics at `left: 0, right: 1`.
  5. `test_c7_5_resizer_with_target_below_16_makes_progress`: panics at `CacheConsciousResizer failed to make progress`.
  6. `test_c7_5_access_pattern_analyzer_with_zero_history_records_accesses`: panics at `cache_locality.rs:518` (`index out of bounds: the len is 0 but the index is 0`).
  7. `test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator`: **PASS** natively (glibc `calloc` returns a non-null pointer for size 0); **FAIL under Miri** at `cache_locality.rs:245` (`319` at `a69815d`) with `Undefined Behavior: creating allocation with size 0`.
  8. `test_c7_5_numa_allocator_overflowing_request_returns_null`: debug panics at `cache_locality.rs:267` (`attempt to multiply with overflow`); release panics at `cache_locality.rs:270` (`Layout invariant violated: alignment is always valid power of 2: LayoutError`).

**Fix (`79e893a`).**
- `NumaAllocator::alloc_on_node` (`cache_locality.rs:244-258`) and `alloc_cache_aligned` (`:274-287`) return `std::ptr::without_provenance_mut(align)` when `layout.size() == 0` / `total_bytes == 0`, and `dealloc` (`:294-313`) returns early when `ptr.is_null()` or `total_bytes == 0` (note: the `ptr.is_null()` early return in `dealloc` and the `!self.node_stats.is_empty()` guard in `alloc_on_node` at `:253` are defensive hygiene — `detect_numa_nodes()` returns `1`, so `node_stats` is non-empty). `alloc_cache_aligned` uses `size_of::<T>().checked_mul(count)` and `Layout::from_size_align(total_bytes, align)`, returning `std::ptr::null_mut()` on overflow (widening the `# Safety` contract and removing both `.expect()` calls). Note: the 2 `// For now` honesty markers in `NumaAllocator` (`cache_locality.rs:226, 251`) remain in place because `NumaAllocator` still uses `alloc_zeroed` rather than `libnuma`, and this is now explicitly documented on `NumaAllocator` (`:193-197`).
- `HotColdSeparator::new` (`cache_locality.rs:439-450`) clamps `hot_ratio` to `[0.0, 1.0]` and uses `capacity.saturating_sub(hot_capacity)`. `access` (`:455-468`) bounds-checks `hot` and `cold` before incrementing `access_counts.get(index)`. `rebalance` (`:476-506`) swaps any cold item with `access_counts >= migration_threshold` (`10`) with the coldest item in `hot` when `min_hot_accesses < cold_accesses`, then resets counts. Documented on `HotColdSeparator` (`:417-422`) that no public insertion method exists yet.
- `CacheLayoutOptimizer::optimal_bucket_size` (`cache_locality.rs:380-382`) clamps `(CACHE_LINE_SIZE / entry_size).max(1)`.
- `CacheConsciousResizer::new` (`cache_locality.rs:525`) uses `(target_size / 16).max(1)` so `chunk_size >= 1`, `resize_step` (`:551-554`) uses `saturating_add` (defensive), and its rustdoc (`:536-541`) discloses that `_rehash_fn` is not invoked.
- `AccessPatternAnalyzer::new` (`cache_locality.rs:583-591`) clamps `history_size.max(1)`.

---

### C7.6 — `Bmi2HashDispatcher` collapsed non-UTF-8 byte slices to `0`, growth helpers overflowed or dropped bit 63 via `BZHI`, scalar `bmi2_hash_combine_*` returned `0`, SIMD string hashes disagreed with scalar on `>= 8`-byte tails, and `TZCNT`/`BEXTR`/`LZCNT` had mismatched target-feature gates — HIGH

**Finding (`src/hash_map/hash_functions.rs:97-123, 238-260, 299-329, 659, 688, 737-875, 1062, 1141`, `src/hash_map/simd_string_ops.rs:188-380, 413`).**
1. **`Bmi2HashDispatcher::hash_with_acceleration` non-UTF-8 collision to `0` (`hash_functions.rs:1060-1066`).** On `HashOptimizationTier::Bmi2Avx2 | HashOptimizationTier::Bmi2`, slices with `bytes.len() >= 8` were hashed via `fast_string_hash_bmi2(std::str::from_utf8(bytes).unwrap_or(""), 0)`. Every non-UTF-8 byte slice of length `>= 8` (such as binary keys or serialized integers) fell back to `""` and hashed to `0` (`base_hash`).
2. **`golden_ratio_next_size` and `optimal_bucket_count` overflow & `BZHI` truncation (`hash_functions.rs:238-260, 299-329`).**
   - In `bmi2_golden_ratio_hardware` (`:255`) and `bmi2_bucket_count_hardware` (`:317`), `_bzhi_u64(size_64 * GOLDEN_RATIO_FRAC_NUM, 63)` and `_bzhi_u64(capacity_64 * 256, 63)` multiplied in `u64` before `BZHI` (wrapping on overflow for `current_size > u64::MAX / 103` or `desired_capacity >= 2^56`) and cleared bit 63 when the product lay in `[2^63, 2^64)` — so on any BMI2 CPU, `optimal_bucket_count(1 << 55)` had `scaled_capacity = _bzhi_u64(1 << 63, 63) = 0`, returning `16` buckets for $2^{55}$ elements, and `golden_ratio_next_size(1 << 57)` shrank to `(39 << 51) + 1` instead of `(103 << 51) + 1`.
   - In the scalar fallbacks (`:240, 302-303`), `size_64 * GOLDEN_RATIO_FRAC_NUM`, `desired_capacity as u64 * 256`, and `.next_power_of_two()` overflowed (panicking in debug builds and wrapping in release builds).
   - `bmi2_bucket_count_hardware` (`:326`) called `_lzcnt_u64(required_buckets - 1)` inside a `#[target_feature(enable = "bmi2")]` function gated only on `caps.has_bmi2`, even though `LZCNT` is an ABM/`lzcnt` instruction separate from BMI2.
3. **Scalar `bmi2_hash_combine_u32` / `bmi2_hash_combine_u64` returned `0` (`hash_functions.rs:97-123`).** The BMI2 hardware implementations remapped `result == 0` to `0x9e3779b9` / `0x9e3779b97f4a7c15` (`:151, 186`), whereas the scalar fallbacks returned `0` for `(0, 0)`.
4. **`SimdStringOps::fast_string_hash` tier mismatch on `>= 8`-byte tails (`simd_string_ops.rs:241-245, 308-312, 376-380`).** `avx2_string_hash` (32-byte chunks), `sse42_string_hash` (16-byte chunks), and `avx512_string_hash` (64-byte chunks) hashed their remaining tail bytes one byte at a time (`hash = hash.rotate_left(5).wrapping_add(byte as u64)`), whereas `scalar_string_hash` (`:407-423`) mixed 8-byte `u64` words first and only hashed `len % 8` bytes byte-by-byte. Any string with a tail of `8..15` (SSE4.2), `8..31` (AVX2, e.g. length `40`), or `8..63` (AVX-512) produced a different hash on SIMD tiers vs `SimdTier::Scalar`.
5. **Target-feature gates, undocumented `unsafe fn`s, and chunk `.expect()`s (`hash_functions.rs:659, 688, 737-875, 1141`, `simd_string_ops.rs:187-360, 413`).** `bmi2_linear_probe_hardware` (`_tzcnt_u64`), `bmi2_quadratic_probe_hardware` (`_bextr_u64`), and `bmi2_double_hash_probe_hardware` (`_bextr_u64`) used BMI1 instructions (`TZCNT`, `BEXTR`) while gated only on `caps.has_bmi2` and `#[target_feature(enable = "bmi2")]`. `sse42_string_compare` (`simd_string_ops.rs:255`) and `avx512_string_compare` (`:323`) were flagged by `scripts/unsafe_audit.py` as undocumented `unsafe fn`s (and none of the 6 SIMD compare/hash `unsafe fn`s had a `# Safety` doc section). Four 8-byte chunk loads used `chunk_bytes.try_into().expect("chunk is 8 bytes")`.

**Tests (`src/hash_map/hash_functions.rs`, `src/hash_map/simd_string_ops.rs`).**
- `hash_map::hash_functions::tests::test_c7_6_bmi2_dispatcher_non_utf8_bytes_do_not_collide_to_zero` (`hash_functions.rs:2017-2034`)
- `hash_map::hash_functions::tests::test_c7_6_growth_helpers_are_exact_on_every_tier` (`hash_functions.rs:2037-2061`)
- `hash_map::hash_functions::tests::test_c7_6_hash_combine_of_zero_and_zero_is_nonzero_on_every_tier` (`hash_functions.rs:2064-2069`)
- `hash_map::simd_string_ops::tests::test_c7_6_simd_string_hash_matches_scalar_tier_at_every_length` (`simd_string_ops.rs:687-708`)

**RED on parent (`79e893a`).**
- On this host (`bmi1`, `bmi2`, `avx2`, `sse4.2` present; no `avx512f`), 3 of the 4 `test_c7_6_*` tests **FAIL** natively in both debug (`test` profile) and `--release` on `79e893a`, and the 4th (`test_c7_6_hash_combine_of_zero_and_zero_is_nonzero_on_every_tier`) **FAILs under Miri** on `79e893a` (where `Bmi2Capabilities` selects the scalar fallback):
  1. `test_c7_6_bmi2_dispatcher_non_utf8_bytes_do_not_collide_to_zero`: **FAIL** natively (debug & release) at `hash_functions.rs:1906` (`2029` in `13f06ce`) — `assertion left != right failed: non-UTF-8 slice must not hash to 0: left: 0, right: 0` (**PASS** under Miri because Miri uses `HashOptimizationTier::Scalar`).
  2. `test_c7_6_growth_helpers_are_exact_on_every_tier`: **FAIL** natively (debug & release) at `hash_functions.rs:1919` (`2042` in `13f06ce`) — `assertion left == right failed: left: 16, right: 72057594037927936` (`optimal_bucket_count(1 << 55)`).
  3. `test_c7_6_hash_combine_of_zero_and_zero_is_nonzero_on_every_tier`: **PASS** natively (host takes the BMI2 hardware path); **FAIL under Miri** at `hash_functions.rs:1950` (`2067` in `13f06ce`) — `assertion left != right failed: left: 0, right: 0`.
  4. `test_c7_6_simd_string_hash_matches_scalar_tier_at_every_length`: **FAIL** natively (debug & release) at `simd_string_ops.rs:664` (`702` in `13f06ce`) — `SIMD string hash mismatch vs scalar tier at length 40: left: 3036131181060257989, right: 3937610358858823715` (**PASS** under Miri because both `active_ops` and `scalar_ops` use `SimdTier::Scalar`).

**Fix (`13f06ce`).**
- Added `#[inline] pub fn fast_bytes_hash_bmi2(bytes: &[u8], base_hash: u64) -> u64` (`hash_functions.rs:722-743`, with `fast_string_hash_bmi2` delegating to `fast_bytes_hash_bmi2(s.as_bytes(), base_hash)`) and wired `Bmi2HashDispatcher::hash_with_acceleration` (`:1182`) to call `fast_bytes_hash_bmi2(bytes, 0)` directly without UTF-8 conversion.
- Added overflow-free scalar references `scalar_golden_ratio_next_size` (`hash_functions.rs:266-271`) and `scalar_optimal_bucket_count` (`:360-372`, using `u64` division/remainder by constant `158` widened to `u128` to avoid `__udivti3` software division calls), delegated the BMI2 hardware functions to the scalar references above `MAX_EXACT` (`:291-294, 390-393`), and replaced `unsafe { _lzcnt_u64(required_buckets - 1) }` with `(required_buckets - 1).leading_zeros()` (`:409`).
- Remapped `result == 0` in scalar `bmi2_hash_combine_u32` (`:101-102`) and `bmi2_hash_combine_u64` (`:124-129`) to `0x9e3779b9` and `0x9e3779b97f4a7c15`.
- Gated `bmi2_linear_probe`, `bmi2_quadratic_probe`, and `bmi2_double_hash_probe` on `caps.has_bmi1 && caps.has_bmi2` (`hash_functions.rs:848, 897, 954`) and annotated their hardware functions with `#[target_feature(enable = "bmi1,bmi2")]` (`:870, 920, 981`), removing 4 redundant inner `unsafe` blocks around `_tzcnt_u64` / `_bextr_u64` (5 total in `hash_functions.rs` with `_lzcnt_u64`; note: on CPUs that have both BMI1 and BMI2, this gate change has no runtime behaviour difference and is not testable on this host).
- Delegated the tail of `avx2_string_hash` (`simd_string_ops.rs:260`), `sse42_string_hash` (`:338`), and `avx512_string_hash` (`:417`) to `self.scalar_string_hash(&bytes[remaining_start..], hash)`. Added `# Safety` doc sections to all 6 SIMD compare/hash `unsafe fn`s (`simd_string_ops.rs:193-196, 233-235, 272-275, 311-313, 351-354, 389-391`, noting that `sse42_string_compare` uses SSE2 `_mm_cmpeq_epi8` under an `sse4.2` gate and that the compare functions require `bytes1.len() == bytes2.len()`). Replaced all 4 `.try_into().expect("chunk is 8 bytes")` calls with `copy_from_slice` (`hash_functions.rs:762-764, 793-795, 1261-1263`, `simd_string_ops.rs:450-452`).

---

## Per-Commit RED Verification Matrix

Every new test was injected into its fix commit's parent in `.claude/worktrees/head` (`$S/commit_checks.sh` for native `test` and `release` profiles; `$S/red_miri_parents.sh` for Miri under `MIRIFLAGS="-Zmiri-tree-borrows"`):

| Commit | Parent | Test (`hash_map::...`) | Parent `test` profile | Parent `release` profile | Parent Miri (`tree-borrows`) |
|--------|--------|------------------------|-----------------------|--------------------------|------------------------------|
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones` | **FAIL** (`Hash table full in SmallInline fallback storage`) | **FAIL** (same) | — |
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_1_inline_churn_does_not_force_a_resize_after_migration` | **FAIL** (`48 != 32` at first `capacity()` check; `occupied` logic guarded by mutant `M5`) | **FAIL** (same) | — |
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_1_migrated_fallback_load_check_counts_the_moved_entries` | **FAIL** (`48 != 32` at first `capacity()` check; `occupied` logic guarded by mutant `M5b`) | **FAIL** (same) | — |
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_2_load_factor_that_is_not_positive_and_finite_is_replaced` | **FAIL** (`OutOfMemory { size: 8589934592 }`) | **FAIL** (same) | — |
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_2_initial_capacity_with_no_power_of_two_is_an_error` | **FAIL** (`attempt to add with overflow`) | **FAIL** (`initial_capacity ... must be rejected`) | — |
| `baea270` (C7.1, C7.2) | `a69815d` | `zipora_hash_map::tests::test_c7_iteration_skips_tombstones_in_every_storage_state` | **FAIL** (`Hash table full in SmallInline fallback storage` on `10..60`) | **FAIL** (same) | — |
| `664c3b5` (C7.3) | `baea270` | `gold_hash_map::tests::test_c7_3_fast_iteration_skips_removed_entries` | **FAIL** (`3 != 2` on `iter_fast()`) | **FAIL** (same) | — |
| `664c3b5` (C7.3) | `baea270` | `gold_hash_map::tests::test_c7_3_reserve_overflow_is_an_error` | **FAIL** (`attempt to add with overflow`) | **FAIL** (`map.reserve(usize::MAX).is_err()`) | — |
| `664c3b5` (C7.3) | `baea270` | `gold_hash_map::tests::test_c7_3_link_capacity_failure_keeps_hash_cache_in_step` | **FAIL** (`6 != 4`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` | **FAIL** (`attempt to subtract with overflow` at `strategy_traits.rs:307`) | **FAIL** (`index out of bounds: len 0, index 1` at `:322`) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_update_beyond_max_probe_distance_and_full_table` | **FAIL** (`Exceeded maximum probe distance`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_tombstone_reuse_keeps_later_entries_reachable` | **FAIL** (`None != Some(50)`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_backward_shift_moves_tombstones_and_later_entries` | **FAIL** (`None != Some(13)`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_probe_stats_match_live_layout` | **FAIL** (`None != Some(0)`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_create_and_resize_storage_use_power_of_two_lengths` | **FAIL** (`[0, 1, 2, 3, 5, 17] != [0, 1, 2, 4, 8, 32]`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_robin_hood_matches_std_hashmap_under_collisions` | **FAIL** (`slot 5: stored probe distance 1 != 9`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_rehash_rejects_u16_probe_distance_overflow` | **FAIL** (`rehash that needs a probe distance above u16::MAX must fail`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_rehash_over_65536_entries_keeps_every_entry` | **FAIL** (`slot 1: stored probe distance 0 != 131072`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_optimize_layout_drops_tombstones` | **FAIL** (`optimize_layout must drop tombstones`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_c7_4_non_power_of_two_slices_are_rejected` | **FAIL** (`insert into a length-5 slice must fail`) | **FAIL** (same) | — |
| `2944cf0` (C7.4, D5) | `664c3b5` | `strategy_traits::tests::test_max_linear_probe_distance_matches_robin_hood_simulation` | **compile-RED** (tests new helper `max_linear_probe_distance`) | **compile-RED** | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_hot_cold_access_out_of_range_is_none` | **FAIL** (`index out of bounds: len 0, index 0`) | **FAIL** (same) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_hot_cold_rebalance_promotes_a_frequently_accessed_cold_element` | **FAIL** (`Some(10) != Some(20)`) | **FAIL** (same) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_hot_cold_ratio_outside_unit_interval_is_clamped` | **FAIL** (`attempt to subtract with overflow`) | **FAIL** (`capacity overflow`) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_optimal_bucket_size_is_at_least_one_for_entries_over_a_cache_line` | **FAIL** (`0 != 1`) | **FAIL** (same) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_resizer_with_target_below_16_makes_progress` | **FAIL** (`CacheConsciousResizer failed to make progress`) | **FAIL** (same) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_access_pattern_analyzer_with_zero_history_records_accesses` | **FAIL** (`index out of bounds: len 0, index 0`) | **FAIL** (same) | — |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator` | PASS natively (glibc `calloc` returns non-null for size 0) | PASS natively | **FAIL (UB)**: `creating allocation with size 0` |
| `79e893a` (C7.5) | `2944cf0` | `cache_locality::tests::test_c7_5_numa_allocator_overflowing_request_returns_null` | **FAIL** (`attempt to multiply with overflow`) | **FAIL** (`Layout invariant violated: ... LayoutError`) | — |
| `13f06ce` (C7.6) | `79e893a` | `hash_functions::tests::test_c7_6_bmi2_dispatcher_non_utf8_bytes_do_not_collide_to_zero` | **FAIL** (`0 != 0`) | **FAIL** (same) | PASS under Miri (scalar tier) |
| `13f06ce` (C7.6) | `79e893a` | `hash_functions::tests::test_c7_6_growth_helpers_are_exact_on_every_tier` | **FAIL** (`16 != 72057594037927936`) | **FAIL** (same) | — |
| `13f06ce` (C7.6) | `79e893a` | `hash_functions::tests::test_c7_6_hash_combine_of_zero_and_zero_is_nonzero_on_every_tier` | PASS natively (BMI2 hardware tier) | PASS natively | **FAIL**: `left: 0, right: 0` (scalar tier) |
| `13f06ce` (C7.6) | `79e893a` | `simd_string_ops::tests::test_c7_6_simd_string_hash_matches_scalar_tier_at_every_length` | **FAIL** (`length 40: 3036131181060257989 != 3937610358858823715`) | **FAIL** (same) | PASS under Miri (both scalar tier) |

---

## Mutation Testing Verification (`C7.1` & `C7.4`)

Because a single regression test stops at its first failing assertion on the parent commit, 19 targeted source mutants (`M1a`–`M14` on `RobinHoodStrategy` / `StandardStorageStrategy` and `M5`, `M5b`, `M6`, `M6b` on `SmallInline`) were run against the final test suite:

| Mutant | Mutation | Verdict | Killing Tests (panic line in `13f06ce`) |
|--------|----------|---------|------------------------------------------|
| `M1a` | Walk reuses any tombstone regardless of `inserting_distance >= bucket.probe_distance` | **KILLED** | `test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`:1323`), `test_c7_4_tombstone_reuse_keeps_later_entries_reachable` (`:1369`) |
| `M1b` | Both dry run and walk reuse any tombstone regardless of `inserting_distance >= bucket.probe_distance` | **EQUIVALENT** | Equivalent on every valid Robin Hood layout: before the first tombstone `T` at slot `s`, every slot on the probe chain holds a live entry with `d_entry >= d_new`, so by the Robin Hood invariant `T.probe_distance <= d_new` already holds at the first tombstone encountered (verified across 64 seeds × both deletion modes with 0 divergences). |
| `M2` | `backward_shift_delete` stops at tombstones (`next_bucket.is_empty()` instead of `is_vacant(next_bucket)`) | **KILLED** | `test_c7_4_backward_shift_moves_tombstones_and_later_entries` (`:1406`), `test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`:1323`) |
| `M3a` | Lookup (`rh_find` / `rh_find_from`) stops at tombstones | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1153`) |
| `M3b` | `insert` does not look past the first displacement / tombstone for an existing key | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1153`), `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1252`) |
| `M4` | `StandardStorageStrategy::resize_storage` does not rehash (`resize_with` only) | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1092`), `test_c7_4_tombstone_reuse_keeps_later_entries_reachable` (`:1357`), `test_c7_4_backward_shift_moves_tombstones_and_later_entries` (`:1406`), `test_c7_4_probe_stats_match_live_layout` (`:1446`), `test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`:1607`), `test_c7_4_rehash_rejects_u16_probe_distance_overflow` (`:1674`), `test_c7_4_rehash_over_65536_entries_keeps_every_entry` (`:1710`) |
| `M4b` | `rh_place` forgets to clear `DELETED_FLAG` when reusing a tombstone | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1097`), `test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`:1323`), `test_c7_4_tombstone_reuse_keeps_later_entries_reachable` (`:1369`) |
| `M7` | `record_moved` does not update `sum_squared_probe_distance` | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1278`), `test_c7_4_robin_hood_matches_std_hashmap_under_collisions` (`:1309`), `test_c7_4_probe_stats_match_live_layout` (`:1440`) |
| `M8` | `create_storage` does not round capacity to a power of two | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1097`), `test_c7_4_create_and_resize_storage_use_power_of_two_lengths` (`:1567`) |
| `M9` | Dry run skips `sim_distance > max_distance` check on displaced entries | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1210`) |
| `M10` | Dry run skips `sim_distance > max_distance` check before placing into vacant/tombstone slot | **KILLED** | `test_c7_4_robin_hood_tombstones_resize_rehash_and_variance` (`:1097`), `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1210`), `test_c7_4_create_and_resize_storage_use_power_of_two_lengths` (`:1567`) |
| `M11` | Existing-key check stops at `max_probe_distance` | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1196`) |
| `M12` | `collision_count` omits steps before the first displacement (`first_action_distance`) | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1278`) |
| `M13` | `collision_count` omits steps during the displacement walk | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1277`) |
| `M14` | `eviction_count` not incremented on Robin Hood swap | **KILLED** | `test_c7_4_update_beyond_max_probe_distance_and_full_table` (`:1278`), `test_c7_4_probe_stats_match_live_layout` (`:1438`) |
| `M5` | Migration omits `*occupied = *len` (keeps stale inline `occupied`) | **KILLED** | `test_c7_1_inline_churn_does_not_force_a_resize_after_migration` (`zipora_hash_map_tests.rs:1067`) |
| `M5b` | Migration sets `*occupied = 0` instead of `*occupied = *len` | **KILLED** | `test_c7_1_migrated_fallback_load_check_counts_the_moved_entries` (`zipora_hash_map_tests.rs:1093`) |
| `M6` | Migrated `SmallInline` fallback skips the load-factor check | **KILLED** | `test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones` (`zipora_hash_map_tests.rs:1026`), `test_c7_1_migrated_fallback_load_check_counts_the_moved_entries` (`:1093`) |
| `M6b` | Migrated `SmallInline` fallback skips resize-on-`Err` when `load_factor >= 1.0` | **KILLED** | `test_c7_1_small_inline_fallback_grows_past_32_and_compacts_tombstones` (`zipora_hash_map_tests.rs:1042`) |

---

## `plan.md` Row `C7` & `D5` Exit-Criteria Mapping

| Exit Criterion (`plan.md` C7 / D4 / D5) | Resolution |
|----------------------------------------|------------|
| Load-factor resize invariant | `ZiporaHashMap` validates `config.load_factor` (`!(is_finite() && > 0.0) -> 0.75`, `zipora_hash_map.rs:182-184`) and enforces the occupancy load check both on `Standard` storage (`zipora_hash_map.rs:289-299`) and on migrated `SmallInline` fallback storage (`probe.rs:290-323`), with `occupied` reset to `len` (`16`) at migration (`probe.rs:427`). |
| Tombstone handling | `ZiporaHashMap` compacts tombstones via in-place rehash when `live < load_limit / 2` (`probe.rs:62-66, 301-303`); `GoldHashMap` downgrades `Fast` iteration to `Safe` while `deleted_count() > 0` (`gold_hash_map.rs:349-360`); `RobinHoodStrategy` preserves phantom `probe_distance` on tombstones, probes past them, reuses them soundly, shifts across them in `backward_shift_delete`, and compacts them in `resize_storage` / `optimize_layout` (`strategy_traits.rs:263-965`). |
| `with_capacity` rounding | `ZiporaHashMap::create_storage` (`zipora_hash_map.rs:225-234`) uses `checked_next_power_of_two()` and returns `Err(OutOfMemory)` when no power of two fits in `usize`; documented on `with_capacity` (`zipora_hash_map.rs:137-141`). `StandardStorageStrategy::create_storage` and `resize_storage` round to powers of two (`strategy_traits.rs:911-940`). |
| Hash-flooding resistance statement (`ahash` keys) | Documented on `ZiporaHashMap` (`zipora_hash_map.rs:43-52`): default `S = ahash::RandomState` uses per-process OS-RNG keys (`runtime-rng` feature on `ahash 0.8.12`) varied per map instance, whereas `GoldHashMap` uses fixed-seed `ahash::AHasher::default()` (`gold_hash_map.rs:202-204, 535-537`, pre-existing note) and is not intended for attacker-controlled keys. |
| Iteration under mutation rules & `D4` tombstone check | Shared/mutable borrowing prevents mutation of `ZiporaHashMap` or `GoldHashMap` while any iterator (`iter`, `iter_mut`, `keys`, `values`, `values_mut`) is alive (`zipora_hash_map_tests.rs:1200-1202`). `test_c7_iteration_skips_tombstones_in_every_storage_state` (`zipora_hash_map_tests.rs:1199-1239`) verifies all 5 iterators across `Standard`, pre-migration `SmallInline`, and post-migration `SmallInline` with tombstones. |
| Remove or implement the strategy stubs (`D5`) | Implemented `RobinHoodStrategy::probe_stats` variance (`strategy_traits.rs:704-721`) and `StandardStorageStrategy::resize_storage` / `optimize_layout` (`:934-965`); deleted the unimplemented `CacheOptimizedStorageStrategy` and `SimdOptimizationStrategy` stubs and their configs/contexts/re-exports (`2944cf0`, **5** `TODO: Implement` honesty markers removed). |

---

## S18 Follow-Up Recap (`2055395..a69815d`, Completed Prior to C7)

Before starting `C7`, the two commits `2055395` (`S18-R1, R2, R4-R7`) and `a69815d` (`S18-R3`) closed all seven Stage 7 review follow-ups:
- **`S18-R1` (`2055395`, `src/containers/fast_vec/traits.rs:95-145`)**: Restored `FastVec::extend` vectorization by bounding the inner write loop to `(&mut iter).take(spare)` with a `SetLenOnDrop` guard and moving the dishonest-iterator reallocation fallback to `#[cold] #[inline(never)] fn extend_overflow_slow`.
- **`S18-R2` (`2055395`, `src/containers/uint_vec_min0.rs:265-330`)**: Extracted the `59..=64`-bit straddling read/write paths into `#[cold] #[inline(never)]` helpers (`get_wide_slow`, `get2_wide_slow`, `fast_get_wide_slow`), restoring `UintVecMin0::get` sequential speed for `<= 58`-bit widths.
- **`S18-R3` (`a69815d`, `src/containers/specialized/vec_trb.rs`)**: Implemented full threaded Red-Black tree deletion rebalancing (`delete_fixup` with threaded left/right rotations) in `VecTrbSet` / `VecTrbMap`, verified by `test_s18_r3_vec_trb_remove_rebalances_and_maintains_log_height`.
- **`S18-R4` (`2055395`, `src/containers/fast_vec/mod.rs`, `valvec32.rs`, `circular_queue.rs`)**: Added `DeallocGuard` in `Drop` for `FastVec`, `ValVec32`, and `AutoGrowCircularQueue` so a panicking element `Drop` still deallocates the container's backing buffer, verified under Miri by `test_s18_r4_container_drop_panicking_element_deallocates_buffer`.
- **`S18-R5` (`2055395`, `src/succinct/elias_fano/clustered.rs`)**: Restored `ClusteredEliasFano` dense build throughput by skipping the per-element EF cost pass when the chunk is already classified as a `Run` or dense `Bitmap`.
- **`S18-R6` (`2055395`, `src/containers/specialized/fixed_len_str_vec.rs`, `easy_hash_map.rs`, `src/hash_map/zipora_hash_map.rs`)**: Wired real SIMD search/prefix-count into `FixedLenStrVec::find_exact_simd` and `count_prefix_simd` (`test_s18_r6_find_exact_simd_and_count_prefix_simd`) and implemented `iter_mut()`, `values_mut()`, and `IntoIterator for &mut` on both `EasyHashMap` and `ZiporaHashMap` (`D4`).
- **`S18-R7` (`2055395`, `docs/review/containers.md`)**: Corrected all inaccurate claims in `docs/review/containers.md`.

---

## Breaking & Behaviour Changes (`src/hash_map/`)

1. **Removed unimplemented strategy re-exports (`2944cf0`, `D5`, BREAKING)**: `CacheOptimizedStorageStrategy`, `CacheOptimizedStorageConfig`, `SimdOptimizationStrategy`, `SimdOptimizationConfig`, and `SimdOptimizationContext` were deleted from `src/hash_map/strategy_traits.rs` and are no longer re-exported from `crate::hash_map`.
2. **`RobinHoodContext` public fields (`2944cf0`, BREAKING for exhaustive struct literals)**: `RobinHoodContext` gained two `pub` fields (`pub live_entries: usize` and `pub sum_squared_probe_distance: u128`); callers constructing it via `RobinHoodContext::default()` or `RobinHoodStrategy::create_context()` are unaffected.
3. **`RobinHoodStrategy` trait bounds relaxed (`2944cf0`)**: `CollisionResolutionStrategy<K, V> for RobinHoodStrategy` now requires `K: Hash + Eq` instead of `K: Hash + Eq + Clone, V: Clone`.
4. **Added public function (`13f06ce`)**: `crate::hash_map::hash_functions::fast_bytes_hash_bmi2(bytes: &[u8], base_hash: u64) -> u64`.
5. **Behavioural changes on edge-case inputs**:
   - `ZiporaHashMap::with_config` / `with_config_and_hasher` replaces non-positive or non-finite `load_factor` (`<= 0.0`, `NaN`, `±inf`) with `0.75`, and returns `Err(OutOfMemory)` when `initial_capacity` has no power of two in `usize`.
   - `ZiporaHashMap::capacity()` on a migrated `SmallInline` map returns the fallback table's capacity (`32`, `64`, ...) instead of `16 + fallback.capacity()`.
   - `GoldHashMap::iter_fast()` and `iter_with_strategy(IterationStrategy::Fast)` downgrade to `IterationStrategy::Safe` while `deleted_count() > 0`.
   - `NumaAllocator::alloc_on_node` and `alloc_cache_aligned` return a dangling aligned pointer (`without_provenance_mut`) for zero-byte requests and `null_mut()` on layout overflow.

---

## Performance Verification (A/B Benchmark, `a69815d` vs `13f06ce`)

Measured with the standalone benchmark binary (`$S/hmbench`, `release`, `codegen-units = 1`, 10 interleaved A/B rounds × best of 7 reps per round on 64-core x86-64 host with `sse4.2`, `popcnt`, `bmi1`, `bmi2`, `avx2`):

| Benchmark | `13f06ce` vs `a69815d` (min / median) | Notes |
|-----------|---------------------------------------|-------|
| `dispatcher_hash_64k_x1000` | **-26.1% / -26.4%** (10/10 wins) | `Bmi2HashDispatcher` calls `fast_bytes_hash_bmi2(bytes, 0)` directly without `std::str::from_utf8(bytes)` validation on every call. |
| `robin_hood_insert_45k_into_64k` | **+61.1% / +54.9%** (0/10) | Expected cost of the read-only dry run before mutation (required so a failed `insert` after Robin Hood swaps never corrupts the table) plus `u128` squared-distance variance accounting (`#[inline]` on `record_*` recovered ~23% from `1.14 ms` to `0.88 ms` vs `0.76 ms` on `a69815d`; magnitude is harness-dependent). |
| `robin_hood_insert_45k_into_64k_prealloc` | **+68.5% / +67.9%** | Same dry-run + variance accounting cost on pre-allocated slice. |
| `robin_hood_get_45k_x10` | **+2.2% / +1.1%** | Lookup path checks `is_vacant` (`key.is_none() && (flags & 0x8000) == 0`) instead of `key.is_none()`. |
| `growth_helpers_x10m` | **+15.4% / +14.5%** (0/10) | `MAX_EXACT` branch in `bmi2_golden_ratio_hardware` / `bmi2_bucket_count_hardware`. |
| `zhm_small_inline_churn_1m` | **+1.7% / +1.4%** | `V1` design resets `*occupied = *len` once at migration instead of decrementing on every inline `remove` (`-2.5% / -3.0%` faster than per-remove decrement, 10/10 wins). |
| `zhm_insert` / `zhm_get_hit` / `zhm_remove_reinsert` / `gold_*` (no deletions) / `simd_string_hash` / `fast_string_hash_bmi2` | **Neutral (within noise / placement)** | `zhm_get_hit` (`+3.3% / +3.9%`) and `fast_string_hash_bmi2` short (`+2.0% / +2.5%`) have identical machine code in the hot loop (verified via `objdump`; differences are code-alignment placement artifacts). |
| `gold_iter_fast_1m_25pct_removed` | **+35.6% / +59.2%** | Expected when `deleted_count() > 0`: `iter_fast()` now checks `entry.link != L::DELMARK` instead of yielding deleted entries (when `deleted_count() == 0`, `iter_fast()` takes the unchanged `Fast` arm). |

---

## Carry-Over Items & Explicit Disclosures

### Disclosures on Tests & Defensive Code
1. **First-failure-only RED runs**: In multi-assertion tests (`test_c7_1_inline_churn_...`, `test_c7_1_migrated_fallback_...`, `test_c7_3_fast_iteration_...`, `test_c7_4_*`), native RED execution halts at the first panicking assertion. Specifically:
   - `test_c7_1_inline_churn_does_not_force_a_resize_after_migration` and `test_c7_1_migrated_fallback_load_check_counts_the_moved_entries` fail on `a69815d` at `assert_eq!(map.capacity(), 32)` (`48 != 32`) before testing `occupied`; their `occupied` logic is verified by mutants `M5` and `M5b`.
   - `test_c7_iteration_skips_tombstones_in_every_storage_state` passes its `Standard` and pre-migration `SmallInline` blocks on `a69815d` and fails on `10..60` via `C7.1`.
   - `test_c7_3_fast_iteration_skips_removed_entries` fails on `baea270` on `map.iter_fast()` (whose `a69815d` doc warned it could yield deleted entries) before reaching the `map.iter()` assertion (where `default_iteration_strategy: Fast` violated `iter()`'s doc).
2. **Tier- and Miri-specific REDs**:
   - `test_c7_5_numa_allocator_zero_byte_requests_skip_the_global_allocator` and `test_c7_6_hash_combine_of_zero_and_zero_is_nonzero_on_every_tier` are **RED under Miri only** and pass natively on this x86-64 BMI2 host.
   - `test_c7_6_bmi2_dispatcher_non_utf8_bytes_do_not_collide_to_zero` and `test_c7_6_simd_string_hash_matches_scalar_tier_at_every_length` are **RED natively** and pass under Miri (where all tiers fall back to scalar).
   - `test_max_linear_probe_distance_matches_robin_hood_simulation` tests the new private helper `max_linear_probe_distance` added in `2944cf0` and is therefore compile-RED on `664c3b5`.
3. **Contract widening in `NumaAllocator::alloc_cache_aligned`**: On `a69815d`, non-overflow of `size_of::<T>() * count` was a documented `# Safety` precondition of `alloc_cache_aligned`; `test_c7_5_numa_allocator_overflowing_request_returns_null` tests the widened contract (returning `null_mut()` instead of panicking).
4. **Untested defensive guards & unreachable conversions**:
   - `ZiporaHashMap::resize_storage` `SmallInline` arms (`probe.rs:158-185`): unreachable from the public API because `insert` only calls `resize_storage` when `self.storage` is `Standard`.
   - `probe.rs` value `Option` propagations (replacing the 8 `.expect()`s) and `gold_hash_map.rs` `L::from_usize` `let-else` branches (`:520, 580`, replacing the 2 `.expect()`s): the `None` arms are unreachable because `entry.key.is_some()` implies `entry.value.is_some()` and `idx <= L::MAX.as_usize()` is checked first.
   - `GoldHashMap::reserve` `f64`/`ceil` bucket calculation and `!scaled.is_finite() || scaled >= isize::MAX` guard (`gold_hash_map.rs:423-428`).
   - `NumaAllocator::dealloc` null-pointer guard (`cache_locality.rs:295`), `alloc_on_node` `!self.node_stats.is_empty()` guard (`:253`), and `CacheConsciousResizer::resize_step` `saturating_add` (`:551-554`).
   - `caps.has_bmi1 && caps.has_bmi2` runtime gates (`hash_functions.rs:848, 897, 954`): untestable on a host CPU that has both BMI1 and BMI2.

### Carry-Over Notes (Out of Scope / Documented)
- `src/hash_map/cache_locality.rs:226, 251`: The 2 `// For now` honesty markers in `NumaAllocator` remain because `NumaAllocator` still delegates to `alloc_zeroed` rather than OS NUMA bindings (disclosed in rustdoc at `:193-197`). `HotColdSeparator` has no public insertion API (`:417-422`), `CacheConsciousResizer::resize_step` does not invoke `_rehash_fn` (`:536-541`), and several types in `hash_map::cache_locality` duplicate `memory::cache_layout`.
- `src/hash_map/hash_functions.rs:1199-1211`: `Bmi2HashDispatcher::hash_combine_optimal` `Bmi1` and `Scalar` arms can still return `0` for `(0, 0)`, and the `#[cfg(test)]` helper `backward_shift_delete` in `src/hash_map/probe.rs` is test-only.
- `src/hash_map/gold_hash_map.rs`: `rehash` uses infallible `vec![L::TAIL; new_buckets]` and computes `self.max_load` in `f32`.
- `src/hash_map/config.rs` / `zipora_hash_map.rs` / `strategy_traits.rs`: several configuration fields (`hash_strategy`, `growth_factor`, `inline_capacity`, `fallback_threshold`, top-level `initial_capacity`, `optimization_strategy`, `RobinHoodConfig::variance_reduction`, and the `variance_reduction`/`backward_shift` constructor arguments on `RobinHoodStrategy::new`) are not read by the runtime paths (documented on `RobinHoodStrategy` at `strategy_traits.rs:307-309` and `StandardStorageConfig` at `:899-900`). `HashBucket::is_empty(&self)` still means `self.key.is_none()` (true for tombstones; documented at `strategy_traits.rs:278-285`). Pre-migration `SmallInline` does not maintain `self.occupied` until migration resets `*occupied = *len`, and migration's `let _ = Self::insert_standard(...)` relies on `std_cap (32) > 16`.

---

## Audit & Gate Summary (`src/hash_map/`)

| Metric | Before Stage 8 (`a69815d`) | After Stage 8 (`13f06ce`) | Status |
|--------|----------------------------|---------------------------|--------|
| `src/hash_map/` `unsafe` sites (documented / total) | 73 / 78 (93.6%, 5 undocumented) | **71 / 71 (100.0%, 0 undocumented)** | PASS |
| Crate-wide `unsafe` sites (documented / total) | 1,308 / 1,332 (98.2%) | **1,306 / 1,325 (98.6%)** | PASS |
| Crate-wide undocumented `unsafe` (`max_undocumented`) | 24 | **19** | RATCHETED (`24 -> 19`) |
| Crate-wide §8.5 precondition violations (`max_precondition_violations`) | 0 | **0** | PASS |
| `src/hash_map/` API honesty markers | 8 (`storage.rs` 1, `strategy_traits.rs` 5, `cache_locality.rs` 2) | **2** (`cache_locality.rs:226, 251` `For now`) | PASS (-6) |
| Crate-wide API honesty markers (`max_markers`) | 70 | **64** | RATCHETED (`70 -> 64`) |
| `src/hash_map/` production `.expect()` calls | 16 (`probe.rs` 8, `gold_hash_map.rs` 2, `cache_locality.rs` 2, `hash_functions.rs` 3, `simd_string_ops.rs` 1) | **0** | PASS (-16) |
| Crate-wide `.unwrap()` / `.expect()` (`max_unwraps` / `max_expects`) | 0 / 135 (ceiling 140) | **0 / 119** (ceiling **119**) | RATCHETED (`140 -> 119`) |
| Decoder indexing (`max_total` / `max_in_bounds`) | 0 / 132 | **0 / 132** | PASS |
| `src/hash_map/` `#[test]` count | 127 | **159** (+33 new, -1 removed) | PASS |
| Gate test counts (`lib_debug` / `lib_release` / `doctests` / `all_targets`) | 2,959 / 2,976 / 228 / 3,614 (at `d8ad641`; +3 in `S18`, +32 net in `C7`) | **2,994 / 3,011 / 228 / 3,649** (+6 `compile_fail` doctests) | RATCHETED |
| Miri (`MIRIFLAGS="-Zmiri-tree-borrows" cargo +nightly miri test --lib hash_map`) | — | **171 passed, 0 failed, 2 ignored** (0 UB, 0 leaks; 159 `hash_map::` + 14 `easy_hash_map`) | PASS |

---

## Verification Summary

- **Commits (`a69815d..13f06ce`)**:
  1. `baea270` — `fix(hash_map): grow the SmallInline fallback table and validate ZiporaHashMap load factor and capacity (C7.1, C7.2)`
  2. `664c3b5` — `fix(hash_map): skip removed entries in GoldHashMap fast iteration, reject overflowing reserve, and check link capacity before allocating (C7.3)`
  3. `2944cf0` — `fix(hash_map): keep RobinHoodStrategy entries reachable across tombstones, rehash on resize, and delete the strategy stubs (C7.4, D5)`
  4. `79e893a` — `fix(hash_map): make zero-size and overflowing NumaAllocator requests sound and fix cache_locality edge cases (C7.5)`
  5. `13f06ce` — `fix(hash_map): hash non-UTF-8 bytes in the BMI2 dispatcher, make growth helpers exact, and gate TZCNT/BEXTR on BMI1 (C7.6)`
- **Sanity Gate (`make sanity`)**: **PASS** (`exit 0`; 0 clippy warnings across `--all-targets --all-features` and `--no-default-features`; all 4 audit scripts and `check_gate_drift.py` pass at the ratcheted baselines).
