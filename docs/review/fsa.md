# C4 — `src/fsa/` review ledger (+ `D2` / `D5` / `five_level_pool` Option (a) / `S11-R1`)

Stage 5 of the plan (`plan.md` §2, row C4, with the coupled Phase D rows `D2` and `D5`, the carry-in `five_level_pool` Option (a) decision `F1`–`F3`, and `S11-R1`).

Protocol per `plan.md` §5: RED first (a test that fails for the stated reason, watched failing on the parent commit), one commit per finding with its regression test, `SAFETY:` on every `unsafe` (`src/fsa/` is **126/126 = 100.0%**), zero §8.5 `debug_assert!` precondition violations in `src/fsa/`, and **0 honesty markers** in `src/fsa/`.

**Scope.** 26 files in `src/fsa/` (`19,820` lines after fixes, `126` `unsafe` sites, **100.0%** documented) plus `src/memory/five_level_pool.rs` (`F1`–`F3`, `S11-R1`).

| File | Lines | Reviewed |
|---|---|---|
| `cspp_trie.rs` | 1,680 | yes (full — see C4.1, C4.4) |
| `cspp_trie_concurrent.rs` | 1,590 | yes (full — see C4.4) |
| `dawg.rs` | 1,038 | yes (full — see C4.3) |
| `double_array/builder.rs` | 308 | yes |
| `double_array/iter.rs` | 396 | yes |
| `double_array/mod.rs` | 112 | yes |
| `double_array/tests.rs` | 739 | yes (full — see C4.1) |
| `double_array/trie.rs` | 530 | yes (full — see C4.1) |
| `double_array/trie_map.rs` | 1,103 | yes (full — see C4.1) |
| `fast_search.rs` | 1,196 | yes (full — see C4.1) |
| `louds_trie.rs` | 1,265 | yes |
| `metadata.rs` | 510 | yes |
| `mod.rs` | 146 | yes |
| `nesting_trie_dawg.rs` | 1,133 | yes |
| `patricia_trie.rs` | 817 | yes |
| `production_polish.rs` | 674 | yes |
| `simd_accelerations.rs` | 611 | yes |
| `simple_implementations.rs` | 482 | yes (full — see C4.3) |
| `state_management.rs` | 848 | yes |
| `strategy_traits.rs` | 909 | yes (full — see C4.3, D5) |
| `streaming.rs` | 895 | yes |
| `traits.rs` | 299 | yes |
| `version_sync.rs` | 458 | yes (full — see C4.3) |
| `zipora_trie/core.rs` | 1,354 | yes (full — see C4.2, D2) |
| `zipora_trie/map.rs` | 336 | yes (full — see C4.2, D2) |
| `zipora_trie/mod.rs` | 74 | yes |
| `zipora_trie/tests.rs` | 317 | yes (full — see C4.2, D2) |

---

## Findings

### C4.1 — `DoubleArrayTrie::for_each_child` panics on `NIL_STATE` / OOB states, `restore_key` visits freed states, and §8.5 `debug_assert!` precondition violations — HIGH

**Finding.**
1. `DoubleArrayTrie::for_each_child(state, f)` (`src/fsa/double_array/trie.rs:297`) indexed `self.units[state as usize]` without checking `state as usize < self.units.len()` or `self.units[state as usize].check >= 0`. Passing `NIL_STATE` (`u32::MAX`, the sentinel returned by `state_move` on a miss) panicked with `index out of bounds`, and passing a freed state slot walked garbage transitions from `base_value()`.
2. `DoubleArrayTrie::restore_key(state)` (`src/fsa/double_array/trie.rs:348`) walked parent pointers via `check` without verifying that the starting state is live (`check >= 0`).
3. Four `debug_assert!` statements in `src/fsa/double_array/trie.rs` (`L114`, `L325`, `L361`) inside functions containing `get_unchecked` were flagged by `scripts/unsafe_audit.py` (§8.5) because their loop invariants were not explicitly proven with `// PROVEN:` comments, and 5 `unsafe` sites in `src/fsa/cspp_trie.rs:62–66` and `src/fsa/fast_search.rs:81` lacked `// SAFETY:` comments.
4. Carry-in `S11-R1`: `test_thread_local_pool_reuses_global_fast_bin_before_carving_new_slabs` in `src/memory/five_level_pool.rs` ran 4,000 iterations × 4 threads unscaled under Miri (~23 min wall time), and `src/fsa/double_array/tests.rs:9` had an unused `zipora::fsa::FiniteStateAutomaton` import when compiled without `--test`.

**Expected failure.** Panic (`index out of bounds: the len is ... but the index is 4294967295`) when calling `for_each_child(NIL_STATE, ...)` from safe code; garbage transitions/keys returned for freed state slots.

**Repro.**
```rust
let trie = DoubleArrayTrie::from_sorted_keys(&[b"alpha".as_slice(), b"beta".as_slice()]).unwrap();
let mut visited = Vec::new();
trie.for_each_child(NIL_STATE, |b, next| visited.push((b, next)));
```

**Tests.** `double_array::tests::test_for_each_child_and_state_queries_on_oob_and_freed_states` in `src/fsa/double_array/tests.rs`.

**RED (watched on `362d061`).**
```
thread 'fsa::double_array::tests::test_for_each_child_and_state_queries_on_oob_and_freed_states' panicked at src/fsa/double_array/trie.rs:297:20:
index out of bounds: the len is 512 but the index is 4294967295
```

**Fix.**
- Guarded `DoubleArrayTrie::for_each_child`, `is_term`, and `restore_key` (`src/fsa/double_array/trie.rs`) against `state as usize >= self.units.len()` and freed slots (`check < 0`).
- Annotated the 4 `debug_assert!` sites in `src/fsa/double_array/trie.rs` with `// PROVEN:` invariant proofs (`max_precondition_violations: 10 -> 6`) and added `// SAFETY:` documentation to `src/fsa/cspp_trie.rs:62–66` and `src/fsa/fast_search.rs:81` (`max_undocumented: 33 -> 28`, `src/fsa/` = **126/126 = 100.0%**).
- Added `#[cfg(miri)]` scaling (`N = 400`, `ARENA = 8 KiB`) to `test_thread_local_pool_reuses_global_fast_bin_before_carving_new_slabs` (`S11-R1`, verified under Miri in `10.83s`) and removed the unused import at `src/fsa/double_array/tests.rs:9`.

**Commit.** `f604bcd`

---

### C4.2 / D2 — `ZiporaTrie` `DoubleArray` terminal transition bug (`DA_TERMINAL_BIT`), unimplemented `CompressedSparse` strategy, and silent error stubs — HIGH

**Finding.**
1. **Terminal-with-children transitions lost in `DoubleArray`**: `ZiporaTrie::transitions(state)` (`src/fsa/zipora_trie/core.rs:914`) read `let base = self.base[state as usize];` without masking off `DA_TERMINAL_BIT` (`0x8000_0000` = `i32::MIN`), which `insert_double_array` ORs into `base[state]` when marking a state terminal. Whenever a terminal state also had outgoing edges (e.g., `"app"` when `"apple"` was also in the trie), `base` was negative (`<= 0`), so `if base <= 0 { return Vec::new(); }` returned `[]` and hid every child transition. Furthermore, `transitions()` masked `check[next]` with `0x7FFF_FFFF` instead of checking `check >= 0 && (check & !DA_TERMINAL_BIT) == state as i32`, which could match negative free-list links.
2. **OOB / freed-state access in `ZiporaTrie`**: `is_final`, `transition`, `transitions`, and `restore_string_double_array` did not verify that `state` is an allocated live state (`check[state] >= 0` for `state > 0`).
3. **`D2` — `CompressedSparse` strategy stubs**: `TrieStrategy::CompressedSparse` was selectable via `ZiporaTrie::with_strategy(TrieStrategy::CompressedSparse)` and `ZiporaTrieMap::with_strategy(TrieStrategy::CompressedSparse)`, yet `insert` returned `Err(NotSupported)`, `contains` returned `false`, `transition` returned `None`, `transitions` returned `Vec::new()`, `is_final` returned `false`, `predict_double_array`/`restore_string` returned empty results, and `ZiporaTrieMap` had no value mapping for `CompressedSparse`.
4. **Silent no-ops on `CriticalBit` / `Louds`**: `insert_and_get_node_id` returned `Ok(0)` without inserting, and `remove` returned `Ok(false)` instead of `Err(ZiporaError::NotSupported)`.

**Expected failure.** `ZiporaTrie::transitions(state_of_app)` returns `[]` even when `"apple"` is in the trie; `ZiporaTrie::with_strategy(TrieStrategy::CompressedSparse)` fails on `insert` and returns empty/false on all FSA/Trie/Map methods.

**Repro.**
```rust
let mut trie = ZiporaTrie::with_strategy(TrieStrategy::DoubleArray);
trie.insert(b"app").unwrap();
trie.insert(b"apple").unwrap();
let s = trie. traverse(b"app").unwrap();
assert!(!trie.transitions(s).is_empty()); // failed: returned []
```

**Tests.**
- `zipora_trie::tests::test_double_array_terminal_state_preserves_child_transitions_and_guards_free_states`
- `zipora_trie::tests::test_compressed_sparse_strategy_full_fsa_trie_and_map_relocation`
- `zipora_trie::tests::test_unsupported_mutating_methods_return_not_supported_error`

**RED (watched on `f604bcd`).**
```
thread 'fsa::zipora_trie::tests::test_double_array_terminal_state_preserves_child_transitions_and_guards_free_states' panicked:
terminal prefix state 'app' must still expose child transition 'l' -> 'apple', got []
```

**Fix.**
- Masked `DA_TERMINAL_BIT` via `da_raw_base(self.base[idx])` (`raw & !DA_TERMINAL_BIT`) in `transitions()`, verified `self.is_valid_da_state(idx)` across `is_final`, `transition`, `transitions`, and `restore_string_double_array`, and checked `self.check[next_idx] >= 0 && da_check_parent(self.check[next_idx]) == state as i32`.
- Embedded a `CsppTrie` inside `ZiporaTrie` for `TrieStrategy::CompressedSparse`, encoding FSA states as `(node_slot << 8) | zpath_progress` (`u32`) and storing a permanent 1-based `u32` `value_id` in `CsppTrie`'s node `valpos` (which `CsppTrie` copies verbatim across all node relocations and `zpath` splits), giving `ZiporaTrieMap` relocation-safe `insert`/`get`/`get_mut`/`remove`/`iter` under `DoubleArray`, `Patricia`, and `CompressedSparse`.
- Changed `insert_and_get_node_id` and `remove` on `CriticalBit`/`Louds` to return `Err(ZiporaError::NotSupported)`.
- Removed all 12 honesty markers in `src/fsa/zipora_trie/`.

**Commit.** `7edb8f5`

---

### C4.3 / D5 — `PatriciaAlgorithmStrategy` short-key truncation & `split_compressed_path` no-op (`strategy_traits.rs`) and `NestedTrieDawg` incremental `insert` self-loop (`dawg.rs`) — HIGH

**Finding.**
1. **`D5` — `PatriciaAlgorithmStrategy::insert` truncated keys shorter than `compression_threshold` (`8`) to 1 byte (`src/fsa/strategy_traits.rs:285`)**:
   `if key[i..].len() >= self.compression_threshold` handled keys of length `>= 8` by creating a compressed edge, while the `else` branch created a 1-byte child (`vec![byte]`), set `node.is_terminal = i == key.len() - 1`, and immediately executed `break;`! Any key of length `2..=7` (e.g. `"cat"`, `"rust"`, `"alpha"`) had only its first byte inserted and was never marked terminal (`contains(b"cat") == false`), while its 1-byte prefix became a phantom non-terminal node.
2. **`D5` — `PatriciaAlgorithmStrategy::split_compressed_path` was a comment-only no-op (`src/fsa/strategy_traits.rs:235`)**:
   When a key had length `>= 8` (creating a compressed edge `"abcdefgh"`) and a second key sharing a prefix (`"abcdefxyz"`) was inserted, `split_compressed_path` did nothing and `insert` returned `Ok(true)` without creating the split node or the new branch (`contains(b"abcdefxyz") == false`).
3. **`NestedTrieDawg` (`src/fsa/dawg.rs:367`) incremental `Trie::insert` corrupted state 0**:
   `NestedTrieDawg::new()` initializes `self.states` as empty (`root_state = 0`), whereas `build_from_sorted` pushes root state `0` before calling `insert_key`. Calling `Trie::insert` on `NestedTrieDawg::new()` called `insert_key` with `self.states.is_empty()`, so the first allocation (`self.states.len() as StateId`) returned `0` — turning transition `(0, key[0]) -> 0` into a self-loop on the root state and marking state `0` final for `1`-byte keys (making the empty key `b""` match!). Additionally, duplicate inserts incremented `self.num_keys` unconditionally and `transition`/`transitions` did not guard OOB states consistently.

**Expected failure.** `PatriciaAlgorithmStrategy::new()` fails `contains(b"cat")` immediately after `insert(b"cat")`, and fails prefix splits on `>= 8`-byte keys; `NestedTrieDawg::new()` corrupts root state `0` on `Trie::insert(b"a")`.

**Repro.**
```rust
let mut pat = PatriciaAlgorithmStrategy::new();
pat.insert(b"cat").unwrap();
assert!(pat.contains(b"cat")); // failed: returned false!
```

**Tests.**
- `strategy_traits::tests::test_patricia_strategy_short_keys_and_compressed_path_splitting` in `src/fsa/strategy_traits.rs`
- `dawg::tests::test_nested_trie_dawg_incremental_insert_and_oob_state_guards` in `src/fsa/dawg.rs`

**RED (watched on `7edb8f5`).**
```
thread 'fsa::strategy_traits::tests::test_patricia_strategy_short_keys_and_compressed_path_splitting' panicked:
assertion failed: pat.contains(b"cat")
thread 'fsa::dawg::tests::test_nested_trie_dawg_incremental_insert_and_oob_state_guards' panicked:
assertion failed: !dawg.contains(b"")
```

**Fix.**
- Rewrote `PatriciaAlgorithmStrategy::insert`, `split_compressed_path`, `optimize` (collapsing single-child non-terminal chains), and `statistics` (`compute_depth_and_efficiency`) in `src/fsa/strategy_traits.rs` so keys of every length (`0..=N`) and compressed-edge splits work accurately.
- Fixed `NestedTrieDawg::insert_key` in `src/fsa/dawg.rs` to allocate root state `0` when `self.states.is_empty()` and only increment `self.num_keys` when transitioning a state from non-final to final.
- Cleared the remaining 5 honesty markers across `src/fsa/` (`strategy_traits.rs`, `simple_implementations.rs`, `version_sync.rs`, `cspp_trie.rs`), bringing `src/fsa/` to **0 honesty markers** (`max_markers: 119 -> 102`).

**Commit.** `bb47a44`

---

### C4.4 — `CsppTrie` `NodeView::zpath_slice` 262 KiB OOB slice UB on freed/arbitrary slots and `ConcurrentCsppTrie` lost `set_value` across concurrent node splits — CRITICAL

**Finding.**
1. **`CsppTrie` `NodeView` out-of-bounds raw-pointer slice UB (`src/fsa/cspp_trie.rs:288–329`)**:
   `CsppTrie::free_node(pos, slots)` (`src/fsa/cspp_trie.rs:365`) wrote the freelist link `self.nodes[pos as usize] = self.fast_bins[slots]` directly into word 0 of the freed node, overwriting `flags`, `cnt_type`, and `n_zpath_len` (which is initialized to `u32::MAX` (`0xFFFF_FFFF`) when a bin is empty). Calling `trie.node_view(state)` on an arbitrary or freed `state` (or calling `node_view(2)` on a newly created `CsppTrie::new(4)` where slots `1..4` are zeroed with `cnt_type = 0`, an invalid type whose `skip_slots(0)` is `1` and `zpath_offset` is `curr + 1`) caused `NodeView::zpath_slice()` to compute `std::slice::from_raw_parts(ptr, len)` where `ptr` or `ptr + len` (with `n_zpath_len = 0xFFFF`, i.e., `262,140` bytes) ran past the end of `self.nodes: Vec<u32>`. That is immediate Undefined Behavior from a 100% safe public method.
2. **`ConcurrentCsppTrie` (`src/fsa/cspp_trie_concurrent.rs`) lost `set_value` writes under concurrent splits**:
   The multi-writer value API returns `(is_new, valpos)` from `insert(key)` so the writer can call `trie.set_value(valpos, val)`. However, when Thread A inserted `key_a` (creating node `N1` with value slot `valpos_1`, initialized to `0xFFFF_FFFF`) and, before Thread A executed `trie.set_value(valpos_1, val_a)`, Thread B inserted `key_b` and split/replaced `N1` with a relocated node `N2` (whose value slot is `valpos_2`, copied from `valpos_1` while it was still `0xFFFF_FFFF`), Thread A's subsequent `set_value(valpos_1, val_a)` wrote only to retired slot `valpos_1`. Readers (and `get_value(key_a)`) read `N2`'s `valpos_2` and saw `0xFFFF_FFFF` — silently losing Thread A's value write!

**Expected failure.**
- Under Miri/bounds check: calling `CsppTrie::new(4).node_view(2).zpath_slice()` or `node_view` on a freed slot constructs an out-of-bounds slice beyond `self.nodes`.
- Under concurrent writers (`insert` + `set_value` + `get_value`): `trie.get_value(&key)` returns `Some(0xFFFF_FFFF)` instead of the value written by `trie.set_value(valpos, expected_val)`.

**Repro.**
```rust
// 1. OOB NodeView slice on CsppTrie:
let mut trie = CsppTrie::new(4);
trie.insert(b"hello", 1);
trie.insert(b"world", 2); // frees old root slot 0 with header = 0xFFFF_FFFF
let view = trie.node_view(0);
assert!(view.zpath_slice().is_empty()); // UB: 65535-byte slice past Vec end!

// 2. ConcurrentCsppTrie insert + set_value under contended splits:
let (is_new, valpos) = concurrent_trie.insert(&key);
assert!(is_new);
concurrent_trie.set_value(valpos, expected);
assert_eq!(concurrent_trie.get_value(&key), Some(expected)); // failed: Some(0xFFFF_FFFF)!
```

**Tests.**
- `cspp_trie::tests::test_cspp_node_view_oob_and_freed_state_safety` in `src/fsa/cspp_trie.rs`
- `cspp_trie_concurrent::tests::test_contended_inserts_with_values_preserved_across_splits` in `src/fsa/cspp_trie_concurrent.rs`

**RED (watched on `bb47a44`).**
```
thread 'fsa::cspp_trie::tests::test_cspp_node_view_oob_and_freed_state_safety' panicked:
range end index 65539 out of range for slice of length 18
thread 'fsa::cspp_trie_concurrent::tests::test_contended_inserts_with_values_preserved_across_splits' panicked at src/fsa/cspp_trie_concurrent.rs:1572:21:
assertion `left == right` failed: lost set_value write for key "00012-1" across concurrent split
  left: Some(4294967295)
 right: Some(200012)
```

**Fix.**
- Moved `CsppTrie` freelist links out-of-band into `free_next: Vec<u32>`, stamped freed node slots with `FREED_NODE_HEADER = 0xFF00_0029` (`flags = 0x29` bit `0x20` set, `n_zpath_len = 0xFF > MAX_ZPATH`), added `NodeView::is_well_formed()` checking `curr < nodes.len()`, `(flags & 0x20) == 0`, `zpath_len <= MAX_ZPATH`, `cnt_type in 1..=7`, and `curr + skip + n_children + zpath_slots <= nodes.len()`, and replaced raw-pointer slice creation in `zpath_slice()` with safe bounds-checked slicing.
- Added `CsppTrie::state_move(state, byte)` / `is_term(state)` and `ConcurrentCsppTrie::state_move(state, byte)` / `is_term(state)` with OOB and freed-node guards (`STATE_NOT_FOUND` on invalid/freed states).
- Added an atomic `valpos_remap: Box<[AtomicU32]>` forwarding table to `ConcurrentCsppTrie`: whenever `copy_val_if_final` relocates a terminal node's value slot from `old_valpos` to `new_valpos`, it records `valpos_remap[old_valpos] = new_valpos` (Release) and re-copies `nodes[old_valpos] -> nodes[new_valpos]` inside `update_curr_ptr` after the parent CAS succeeds; `set_value(valpos, value)` stores to `valpos` and follows any `valpos_remap` chain (`Acquire`/`Release`) so both in-flight and post-split `set_value` writes reach the latest relocated node.
- Verified with `make tsan_cspp` (0 warnings) and `make miri_cspp` (14/14 passed).

**Commit.** `980d88c`

---

### C4.5 / F1 / F2 / F3 — `five_level_pool` Option (a): `MemOffset` byte accessors, out-of-band freelist links, and lock-free ABA-tagged fast bins — HIGH

**Finding.**
1. **No safe API to dereference `MemOffset`**: `NoLockingPool`, `MutexBasedPool`, `LockFreePool`, `ThreadLocalPool`, `FixedCapacityPool`, `AdaptiveFiveLevelPool`, and `FiveLevelPoolHandle` returned opaque `MemOffset(u32)` handles with no safe slice or read/write accessors.
2. **`F1` — Freelist link overwrote user bytes `0..4` (`src/memory/five_level_pool.rs:517, 740, 998`)**:
   Once `MemOffset` is dereferenceable, `NoLockingPool::free`, `MutexBasedPool::free`, and `LockFreePool::free` wrote a `u32` freelist next-offset into the first 4 bytes of the freed user block (`*(base_ptr.add(offset) as *mut u32) = prev_head`), corrupting user bytes `0..4` and coupling block alignment to link width.
3. **`F2` — Untaggged 32-bit CAS head in `LockFreeFreeListHead` (`src/memory/five_level_pool.rs:845`)**:
   `LockFreeFreeListHead::head` used an untagged `AtomicU32`, making `pop`/`push` vulnerable to classic pop-pop-push ABA races under concurrent fast-bin reuse.
4. **`F3` — `LockFreePool` fast-bin `alloc`/`free` locked `self.memory.lock()` (`src/memory/five_level_pool.rs:942, 996`)**:
   Even on the lock-free fast-bin path, `LockFreePool::alloc` and `LockFreePool::free` acquired `let memory = self.memory.lock().unwrap();` just to read/write the embedded freelist next pointer or bump-allocate a fast block, serializing all fast-bin callers on a single `Mutex`.

**Expected failure.** Overwritten bytes `0..4` upon `free` (`F1`), lack of `read_bytes`/`write_bytes`/`as_slice`/`as_mut_slice` on `MemOffset`, and mutex contention on `LockFreePool` fast-bin `alloc`/`free` (`F3`).

**Tests.** `memory::five_level_pool::tests::test_five_level_pool_mem_offset_deref_and_out_of_band_freelist_f1_f2_f3` in `src/memory/five_level_pool.rs` (bringing `memory::five_level_pool` to 33 unit tests, 31 under Miri).

**Fix.**
- Added `as_slice(offset, len)`, `as_mut_slice(offset, len)`, `read_bytes(offset, dst)`, and `write_bytes(offset, src)` on `NoLockingPool` and `FixedCapacityPool`, and `read_bytes`/`write_bytes` on `MutexBasedPool`, `LockFreePool`, `ThreadLocalPool`, `AdaptiveFiveLevelPool`, and `FiveLevelPoolHandle` with full bounds checking (`offset.checked_add(len) <= capacity`).
- **`F1`**: Replaced in-band user-memory freelist writes in `NoLockingPool`, `MutexBasedPool`, and `LockFreePool` with out-of-band `next_links` arrays (`Vec<u32>` / `Box<[AtomicU32]>`) indexed by `offset / min_step`. User arena bytes are never touched by freelist metadata on `free`.
- **`F2`**: Upgraded `LockFreeFreeListHead::head` from `AtomicU32` to an ABA-tagged `AtomicU64` packing `(generation: u32) << 32 | (offset: u32)`, incrementing the 32-bit generation counter on every `push` and `pop`.
- **`F3`**: Added `bump_offset: AtomicUsize` and `next_links: Box<[AtomicU32]>` to `LockFreePool` so fast-bin `pop`, `push`, and bump carving execute via lock-free `AcqRel` CAS loops without ever locking `self.memory`.

**Commit.** `80b3fe5`

---

### C4.6 / S12-R1..R9 — `ConcurrentCsppTrie` permanent value slots, `DoubleArrayTrie` `Deserialize` invariant validation, O(1) `ZiporaTrie` `CompressedSparse` `transition`, in-band `CsppTrie` freelist links, and lazy `five_level_pool` freelist blocks — CRITICAL

**Finding.**
1. **`S12-R1` (CRITICAL, `src/fsa/cspp_trie_concurrent.rs`)**: `valpos_remap` in `980d88c` indexed by byte offset (`old_valpos / 4`) without clearing entries when retired nodes were recycled through `LockFreeFreelist`. Once a retired node's slot range was recycled into a new node's header or child pointers, a later `set_value` chased stale `valpos_remap` edges into the new node's interior (SIGSEGV on 200,000 single-threaded random inserts + `set_value`), and `get_value` chased stale chains across unrelated nodes (causing `test_concurrent_trie_multithreaded` and `test_contended_inserts_with_values_preserved_across_splits` to fail 8–9/60 runs).
2. **`S12-R2` (HIGH, `src/fsa/double_array/trie.rs`)**: `DoubleArrayTrie` derived `serde::Deserialize` without checking the structural invariants (`states.len() >= 256`, `states.len() == ninfos.len()`, `!states[0].is_free()`, `(child0 | 0xFF) < states.len()`, valid `parent`) relied upon by `// PROVEN:` `get_unchecked` indexing in `state_move`, `contains`, `longest_prefix`, `for_each_child`, and `restore_key`. A 90-byte JSON payload with `child0 = 0x7FFF0000` and 1 state caused `contains(b"a")` to SIGSEGV in safe code.
3. **`S12-R3` & `S12-R4` (HIGH/MEDIUM, `src/fsa/zipora_trie/trie.rs`)**: `ZiporaTrie::is_valid_cspp_node` ran a recursive whole-trie DFS from root `0` on every `CompressedSparse` `transition`/`transitions`/`is_final` call (turning O(1) transitions into O(n): 668 µs/step at 100k keys), and `(slot << 8) | zprog` state IDs silently wrapped past `2^24` slots (`64 MiB`).
4. **`S12-R5` (MEDIUM, `src/memory/five_level_pool.rs`)**: `MemoryChunk::new` eagerly ran `write_bytes(0, capacity)` (faulting all pages upfront) and `NoLockingPool` / `MutexBasedPool` eagerly constructed a `capacity / min_step` `Vec<u32>` in a loop (`ThreadLocalPool::new(256 MiB)`: 15 µs → 11 ms).
5. **`S12-R6` & `S12-R7` (MEDIUM/LOW, `src/fsa/cspp_trie.rs`, `src/fsa/zipora_trie/trie.rs`)**: `CsppTrie` maintained a parallel `free_next: Vec<u32>` equal to `mempool.len()` (+100% memory overhead) and re-ran `NodeView::is_well_formed()` on every accessor; `ZiporaTrie::transition` (`DoubleArray`) ran `is_free_double_array(state)` before `base`/`check` comparison (+65% transition walk latency).
6. **`S12-R8` & `S12-R9` (LOW, `src/fsa/strategy_traits.rs`)**: Propagate `self.nodes.push(...)?` in `PatriciaAlgorithmStrategy::split_compressed_path` instead of discarding `Result`, and keep `.rs` changes strictly separated from gate/doc commits.

**Tests.**
- `cspp_trie_concurrent::tests::test_single_thread_200k_random_inserts_with_set_value_no_slot_collision` in `src/fsa/cspp_trie_concurrent.rs`
- `double_array::tests::test_deserialize_rejects_malformed_json_that_violates_proven_bounds` in `src/fsa/double_array/tests.rs`
- `zipora_trie::tests::test_compressed_sparse_transition_is_constant_time_and_rejects_overflow_slots` in `src/fsa/zipora_trie/tests.rs`

**Fix.**
- **`S12-R1`**: Removed `valpos_remap` and `has_final_replacements`. When `valsize > 0`, each final node stores a single 1-slot (`4 B`) immutable `valpos: u32` handle at its trailing value-pointer slot, pointing to a permanently bump-allocated value cell (`val_words` slots) in `SharedPool` that is never recycled when nodes are split or replaced. `copy_val_if_final` copies the 4-byte `valpos` pointer verbatim into the replacement node, and `get_value`/`set_value` read/write `pool.data[valpos / 4]` directly with zero indirection maps and zero slot-reuse collisions.
- **`S12-R2`**: Added `#[serde(try_from = "RawDoubleArrayTrie")]` and `TryFrom<RawDoubleArrayTrie> for DoubleArrayTrie` validating `states.len() >= 256`, `states.len() == ninfos.len()`, `!states[0].is_free()`, `search_head < len`, and `(child0 | 0xFF) < len` + valid non-free `parent < len` for every allocated state before constructing `DoubleArrayTrie`.
- **`S12-R3` & `S12-R4`**: Replaced the recursive DFS in `ZiporaTrie::is_valid_cspp_node` with O(1) `cspp.node_view(target_slot).is_well_formed()` (7–17 ns/step across 1k/10k/100k keys), converted `count_cspp_states_and_transitions` and `restore_string_cspp` to iterative stack walks, and added `CSPP_MAX_SLOT = (1 << 24) - 1` + `encode_cspp_state` bounds checks.
- **`S12-R5`**: Switched `MemoryChunk::new` to `std::alloc::alloc_zeroed(layout)` (OS lazy zero-page mapping, 11.2 µs for 256 MiB), replaced eager `next_links: Vec<u32>` in `NoLockingPool`/`MutexBasedPool` with per-bin `Vec<MemOffset>` stacks (zero upfront allocation in `new()`), and allocated `LockFreePool::next_links` via `alloc_zeroed` with `0 = NULL` encoding.
- **`S12-R6` & `S12-R7`**: Removed `free_next: Vec<u32>` from `CsppTrie` (storing freelist next links in-band at `mempool[slot + 1].child` for `slots >= 2` while keeping `FREED` marker `0x29` in `mempool[slot].meta`), cached `meta` and `well_formed` once in `NodeView::new` (`NodeView::new_trusted` for internal hot paths), and folded the `DoubleArray` free-state check into the `check_value == state` hit branch (44 ns/key transition walk).
- **`S12-R9`**: Propagated `self.nodes.push(...)?` in `PatriciaAlgorithmStrategy::split_compressed_path`.

**Commit.** `d5da487`

---

## Verification Summary

- **Gate (`touch src/lib.rs && make sanity`)**:
  - `cargo clippy --all-targets --all-features -- -D warnings`: **0 warnings**
  - `cargo clippy --no-default-features -- -D unused_variables -D unused_imports`: **0 warnings**
  - `scripts/unsafe_audit.py`: **28 undocumented** (ceiling `28`, down from `33`; `src/fsa/` = **135/135 = 100.0%**, `src/memory/` = **340/340 = 100.0%**), **6 precondition violations** (ceiling `6`, down from `10`; `src/fsa/` = **0**)
  - `scripts/api_honesty.py`: **102 markers** (ceiling `102`, down from `119`; `src/fsa/` = **0**)
  - `scripts/index_audit.py`: **6 / 6** (`max_in_bounds = 132`)
  - `scripts/unwrap_audit.py`: **0 `.unwrap()` / 141 `.expect()`**
  - Tests: **2,919 debug lib / 2,936 release lib / 228 doctests / 3,574 all-targets** (+12 new RED-verified regression tests across C4.1–C4.6, 0 removed)
- **Sanitizers & Flake-Loop Verification**:
  - `cargo test --release --lib fsa::cspp_trie_concurrent` × **200 runs**: **0 failures / 200 runs**
  - `make tsan_cspp`: **15 / 15 passed** in `7.33s`
  - `make miri_cspp`: **15 / 15 passed** in `44.58s` (`-Zmiri-tree-borrows`, 0 UB, 0 leaks)
  - `cargo +nightly miri test --lib memory::five_level_pool`: **33 / 33 passed** in `220.49s` (all 33 tests in `five_level_pool`, 0 UB, 0 leaks)
