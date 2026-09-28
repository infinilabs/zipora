# Stage 6 (`C5 succinct/` + Phase D Coupling) Review & Remediation Ledger (`src/succinct/`)

## Scope & Methodology

Stage 6 audited all 27 source files in `src/succinct/` (18,939 LOC across `bit_vector.rs`, `elias_fano/*`, and `rank_select/*`) against the `plan.md` Phase C (`C5 succinct/`) and Phase D checklists:
1. Safe API soundness and `unsafe` invariant verification (zero UB on untrusted serialized bytes, empty inputs, `u64::MAX`, non-multiple-of-64/256/512 bit lengths, out-of-bounds indices/positions/ranks).
2. `rank1` / `rank0` / `select1` / `select0` and Elias-Fano `get` / `next_geq` / `advance_to_geq` / `advance_to_index` / `intersect_count` mathematical correctness across all tiers (AVX-512, AVX2, BMI2, POPCNT, NEON, scalar).
3. Elimination of narrow `as u32` casts, unchecked slice indexing (`succinct_load` in `scripts/index_audit.py`: `6 -> 0`), production `.expect()` (`141 -> 140`), and API honesty markers in `src/succinct/` (`102 -> 95`, **0 remaining in `src/succinct/`**).
4. 100% `// SAFETY:` documentation coverage in `src/succinct/` (**119/119 = 100.0%**) and **0** §8.5 precondition violations in `src/succinct/`.

---

## Findings & Remediations

### C5.1 — `BitVector::bulk_bitwise_op_simd_avx2` underflowed on empty ranges and corrupted boundary bits outside `[start, end)` — CRITICAL

**Finding (`src/succinct/bit_vector.rs:948-1002`).**
- `BitVector::bulk_bitwise_op_simd_avx2` computed `let end_word = (end - 1) / BITS_PER_BLOCK`, which panicked with integer underflow in debug builds (or wrapped to `usize::MAX / 64` in release builds) when `start == 0 && end == 0`.
- When `end_word >= start_word + 3`, the AVX2 4-word loop loaded and stored entire 256-bit chunks (`4 * u64`) starting at `start_word` without applying `start_bit` or `end_bit` masks, corrupting existing bits before `start` in `start_word` and after `end` in `end_word`.
- Furthermore, when `end == self.num_bits` was not a multiple of 64 and `other.num_bits > self.num_bits` had set bits in `(self.num_bits % 64)..64`, `Or` / `Xor` left dirty trailing bits in the final `u64` block, violating the `clear_unused_bits()` invariant and corrupting `count_ones()`, `rank1()`, and `rank0()`.

**Test.**
- `succinct::bit_vector::tests::test_bulk_bitwise_op_simd_and_from_blocks_invariants` in `src/succinct/bit_vector.rs`

**Fix (`690e218`).**
- Added an early `if start >= end { return Ok(()); }` guard in `bulk_bitwise_op_simd`.
- Restricted the unmasked 4-word AVX2 loop to strictly interior full words (`full_start..=full_end`), handling partial boundary words (`start_bit != 0` or `end_bit != 0`) with exact bitmasks and calling `self.clear_unused_bits()` after the operation.

---

### C5.2 — `BitVector::from_blocks` and `from_raw_bits` only checked `blocks.len()` via `debug_assert!` and left dirty trailing bits in the final block — HIGH

**Finding (`src/succinct/bit_vector.rs:146-172`).**
- `BitVector::from_blocks(blocks, num_bits)` and `BitVector::from_raw_bits(blocks, num_bits)` used `debug_assert!(blocks.len() >= required_blocks)` (§8.5 precondition violation), allowing release builds to construct a `BitVector` with fewer blocks than `ceil(num_bits / 64)` and panic later on in-bounds queries.
- Neither constructor cleared unused high bits (`num_bits % 64..64`) in the final block after `blocks.truncate(required_blocks)`, so passing a final word with dirty upper bits caused `count_ones()`, `rank1()`, and `rank0()` to miscount.

**Tests.**
- `succinct::bit_vector::tests::test_bulk_bitwise_op_simd_and_from_blocks_invariants` in `src/succinct/bit_vector.rs`
- `succinct::bit_vector::tests::test_from_blocks_insufficient_panics_in_all_profiles` in `src/succinct/bit_vector.rs`

**Fix (`690e218`).**
- Replaced `debug_assert!` with an unconditional `assert!` in `from_blocks` (to which `from_raw_bits` now delegates) and called `bv.clear_unused_bits()` before returning.

---

### C5.3 — `BitVector::rank1` and `rank1_bulk` did not clamp `pos > num_bits` consistently, and `blocks_mut` safety contract was undocumented — MEDIUM

**Finding (`src/succinct/bit_vector.rs:212-220, 421-462`).**
- `BitVector::rank1(pos)` returned `0` when `pos > self.num_bits`, whereas `rank1_bulk` did not clamp `pos` to `self.num_bits` and could read past `self.blocks` when `pos >= self.blocks.len() * 64`.
- `BitVector::blocks_mut` lacked explicit documentation of the caller's obligation to preserve `self.blocks.len() == (self.num_bits + 63) / 64` and keep trailing bits beyond `self.num_bits` zeroed.

**Test.**
- `succinct::bit_vector::tests::test_bulk_bitwise_op_simd_and_from_blocks_invariants` in `src/succinct/bit_vector.rs`

**Fix (`690e218`).**
- Clamped `pos` to `self.num_bits` in both `rank1` and `rank1_bulk`, and documented the `blocks_mut` invariant and all `// SAFETY:` blocks in `src/succinct/bit_vector.rs`.

---

### C5.4 — `ClusteredEliasFano` and `HybridPostingList` misclassified non-strictly-increasing monotone sequences with duplicates as `Run` or `Bitmap`, silently corrupting values — CRITICAL

**Finding (`src/succinct/elias_fano/clustered.rs:148-195`, `src/succinct/elias_fano/hybrid.rs:95-115`).**
- `ClusteredEliasFano::from_sorted_u64` classified any chunk with `span + 1 == count as u64` as `ChunkKind::Run` and any chunk with `span < BITMAP_DOMAIN` (`4096`) as `ChunkKind::Bitmap`. However, `Run` (`base_val + idx`) and `Bitmap` (one bit per distinct value in `[0, 4096)`) can only represent **strictly increasing** chunks (`w[0] < w[1]`).
- When given a monotone non-decreasing sequence containing duplicates:
  - A chunk of length 128 with `base_val = 0`, `max_val = 127`, and duplicate values (e.g., `[0, 0, 1, 2, ..., 125, 127]`) satisfied `span + 1 == 128` and was encoded as `ChunkKind::Run`, silently replacing the duplicate `0` with `1..126` and synthesizing the missing `126`!
  - Any chunk with duplicates and `span < 4096` (e.g., `[42; 200]`) was encoded as `ChunkKind::Bitmap`, collapsing all duplicate values into a single set bit so `get(1..200)` returned `None`!
- Similarly, `HybridPostingList::is_run_heavy` did not check `w[0] < w[1]` before routing to `ClusteredEliasFano`.

**Test.**
- `succinct::elias_fano::tests::tests::test_monotone_duplicates_and_next_geq_first_index_all_variants` in `src/succinct/elias_fano/tests.rs`

**Fix (`c8aaa6e`).**
- Checked `let is_strictly_increasing = chunk.windows(2).all(|w| w[0] < w[1]);` in `ClusteredEliasFano::from_sorted_u64`, restricting `ChunkKind::Run` and `ChunkKind::Bitmap` to strictly increasing chunks and encoding chunks with duplicates via `ChunkKind::EliasFano`.
- Updated `HybridPostingList::from_sorted_u64` / `is_run_heavy` to require `w[0] < w[1]` for `Run` runs.

---

### C5.5 — `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`, and `HybridPostingList` `next_geq` returned a later chunk's index when duplicate values spanned chunk boundaries — HIGH

**Finding (`src/succinct/elias_fano/partitioned.rs`, `optimal.rs`, `clustered.rs`, `hybrid.rs`).**
- `next_geq(target)` binary-searched chunk metadata using `if meta[mid].max_val < target` without breaking ties toward the lowest chunk index when multiple consecutive chunks had `max_val == target` (e.g., a constant run `[42; 300]` spanning multiple chunks). Binary search landed on an arbitrary middle chunk with `max_val == target` and returned an index `>= 128` instead of index `0` (the first index `i` with `seq[i] >= target`).

**Test.**
- `succinct::elias_fano::tests::tests::test_monotone_duplicates_and_next_geq_first_index_all_variants` in `src/succinct/elias_fano/tests.rs`

**Fix (`c8aaa6e`).**
- Replaced the manual binary search in `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`, and `HybridPostingList` (`next_geq` and cursor `advance_to_geq`) with `partition_point(|m| m.max_val < target)`, which always finds the **first** chunk whose `max_val >= target`.

---

### C5.6 — Deserialized `EliasFano`, `PartitionedEliasFano`, and `OptimalPartitionedEliasFano` did not guard against corrupt `len` / `high_bits`, and `partitioned.rs` / `optimal.rs` had 6 direct slice index sites (`succinct_load`) — HIGH

**Finding (`src/succinct/elias_fano/basic.rs`, `partitioned.rs`, `optimal.rs`, `chunk.rs`).**
1. When `EliasFano`, `PartitionedEliasFano`, or `OptimalPartitionedEliasFano` was deserialized from untrusted input (`#[cfg(feature = "serde")]`) with a `len` or `meta[i].count` larger than the set bits in `high_bits`, or with truncated `low_bits`, calling `get(index)` or `PartitionedEliasFanoBatchCursor::advance` could underflow `bit_pos - index`, shift by `low_width >= 64`, or read out-of-bounds in `extract_low_bits` (which requires `word_idx + 1 < low_bits.len()` for its branchless `u128` load).
2. `partitioned.rs:344-345, 360` and `optimal.rs:265-266, 281` used 6 direct slice index operations (`succinct_load: 6` in `scripts/index_audit.py`), and `optimal.rs` used `.expect(...)` in production code.

**Tests.**
- `succinct::elias_fano::tests::tests::test_corrupt_serde_get_and_pef_batch_cursor_no_ub_or_panic` in `src/succinct/elias_fano/tests.rs`
- `succinct::elias_fano::tests::tests::test_unsorted_input_panics_cleanly` in `src/succinct/elias_fano/tests.rs`

**Fix (`c8aaa6e`).**
- Hardened `extract_low_bits`, `EliasFano::get`, `PartitionedEliasFano`, `PartitionedEliasFanoBatchCursor`, `OptimalPartitionedEliasFano`, and `ClusteredEliasFano` against corrupt `low_width >= 64`, truncated `low_bits`/`high_bits`, and `select1` failure / `bit_pos < index` underflow, returning `None` cleanly without panicking or triggering UB.
- Replaced all 6 direct indexing sites in `partitioned.rs` and `optimal.rs` (`succinct_load: 6 -> 0` in `scripts/index_audit.py`), eliminated the production `.expect()` in `optimal.rs` (`max_expects: 141 -> 140`), and documented all `// SAFETY:` blocks across `src/succinct/elias_fano/`.

---

### C5.7 — `simd::bulk_rank1_*` returned `0` for `pos > bit_len`, and `bulk_select1_bmi2` / `bulk_select1_popcnt` returned wrong positions and panicked on `1u64 << 64` — CRITICAL

**Finding (`src/succinct/rank_select/simd.rs:280-548`).**
1. `bulk_rank1_avx512`, `bulk_rank1_avx2`, `bulk_rank1_popcnt`, `bulk_rank1_neon`, and `bulk_rank1_scalar` checked `if word_index >= bit_data.len() { results.push(0); continue; }`, returning `0` instead of `total_ones` whenever `pos >= bit_data.len() * 64`.
2. `bulk_select1_bmi2` computed `_pdep_u64((1u64 << remaining_rank) - 1, word)` and `deposit.trailing_zeros()`:
   - `_pdep_u64((1 << r) - 1, word)` deposits the lowest `r` bits into the lowest `r` set bits of `word`, whose `trailing_zeros()` is **always the 0-th set bit of `word`**, not the `r`-th set bit!
   - When `remaining_rank == 64` (selecting the 64th set bit in an all-ones word `u64::MAX`), `1u64 << 64` panicked with shift overflow in debug mode (or wrapped to `0` in release mode, returning `64` out of word bounds).
   - It recomputed `bulk_popcount_simd(bit_data)` inside the `for &k in indices` loop in $O(M \cdot N)$ time.
3. `bulk_select1_popcnt` ignored `bit_data` after computing `word_popcounts` and delegated to `bulk_select1_scalar` in $O(M \cdot N)$ time (with an honesty marker `// For now, delegate to scalar with POPCNT acceleration`).

**Test.**
- `rank_select::tests::test_bulk_select1_simd_and_bulk_rank1_simd_correctness` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Clamped `complete_words = word_index.min(bit_data.len())` in all 5 `bulk_rank1_*` tiers so `pos >= bit_data.len() * 64` returns `total_ones`.
- Rewrote `bulk_select1_bmi2` and `bulk_select1_popcnt` to precompute prefix popcounts once in $O(N)$, binary-search the prefix array in $O(\log N)$ per query, and call `crate::algorithms::select_in_word(bit_data[word_idx], rank_in_word)`.
- Added `#[target_feature(enable = "...")]` and `// SAFETY:` comments to all SIMD inner functions in `simd.rs`, removing the honesty marker.

---

### C5.8 — `bmi2_acceleration` shift overflow (`1u64 << 64`) in `one_seq_revlen`, underflow/overflow in `Bmi2BlockOps::select_bulk`, division by zero in `select1_hybrid_cache`, and hardware/scalar mismatch on `>= 64` bit widths — HIGH

**Finding (`src/succinct/rank_select/bmi2_acceleration.rs`).**
1. `Bmi2SequenceOps::one_seq_revlen` and `count_leading_ones_in_mask` computed `(1u64 << end_bit) - 1` where `end_bit` was `64` at word boundaries (`end_pos % 64 == 0` or when scanning preceding full words), panicking on `1u64 << 64`.
2. `Bmi2BlockOps::select_bulk` searched `cumulative_counts.binary_search(&(k + 1))` and subtracted `1` from `Err(pos)` (`pos - 1`), which:
   - Overflowed `k + 1` when `k == usize::MAX`.
   - Underflowed `pos - 1` when block 0 had popcount `0` and `k == 0` (`cumulative_counts[0] == 0`, `binary_search(&1)` returned `Ok(pos)` for a later block or `Err(0)` if the first block had 0 bits — wait: when block 0 had popcount 0 and block 1 had popcount 1, `cumulative_counts` was `[0, 1]`; `binary_search(&1)` returned `Ok(1)`, selecting block `1`, and `remaining = 0 - cumulative_counts[0] = 0`, which happened to work only if `Ok` hit the exact boundary, whereas if block 1 had popcount 2, `cumulative_counts` was `[0, 2]`, `binary_search(&1)` returned `Err(1)`, `block_idx = 0`, and it searched block 0 which had **zero** set bits!).
   - Used `.expect("cumulative_counts non-empty")` in production code.
3. `Bmi2SelectOps::select1_hybrid_cache` divided by `cache_density` without guarding `cache_density == 0`.
4. `Bmi2BextrOps`, `Bmi2BzhiOps`, and `Bmi2Dispatcher` had mismatched boundary behavior between hardware `_bextr_u64`/`_bzhi_u64` and scalar fallbacks when `start >= 64` or `length >= 64`.

**Test.**
- `rank_select::tests::test_bmi2_one_seq_revlen_and_block_select_bulk_and_extract_bits` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Replaced `(1u64 << end_bit) - 1` with `if end_bit >= 64 { u64::MAX } else { (1u64 << end_bit) - 1 }` in `one_seq_revlen` and `count_leading_ones_in_mask`.
- Rewrote `Bmi2BlockOps::select_bulk` to prefix-sum `[0, b0, b0+b1, ...]`, check `k >= total_ones` first, and locate `block_idx = cumulative_counts.partition_point(|&c| c <= k) - 1` (eliminating both zero-block misselection and `.expect(...)`).
- Guarded `cache_density == 0` in `select1_hybrid_cache` and normalized `start >= 64` / `length >= 64` across `Bmi2BextrOps`, `Bmi2BzhiOps`, and `Bmi2Dispatcher`.
- Removed the honesty marker at `bmi2_acceleration.rs:252`.

---

### C5.9 — `RankSelectMixedIL256`, `RankSelectMixedXL256`, and `RankSelectMixedSE512` panicked on `rank1_dim(dim, size)` when `size % 256 == 0`, and returned `Err("not implemented")` on `select0` — HIGH

**Finding (`src/succinct/rank_select/mixed_il_256.rs`, `mixed_xl_256.rs`, `mixed_se_512.rs`).**
1. In `RankSelectMixedIL256::rank1_dim` and `RankSelectMixedXL256::rank1_dim`: `nlines = max_size.div_ceil(256)` allocated `nlines * arity` blocks (no sentinel block). When `size[dim]` was a non-zero multiple of 256 (e.g., 256) and `pos == size[dim]`, `block_idx = (256 / 256) * arity + dim = arity + dim >= self.blocks.len()`, panicking with an index out-of-bounds error!
2. In all three mixed-dimension types (`MixedDimView`, `MixedXL256DimView`, `MixedSE512DimView`), `RankSelectOps::select0` was a stub returning `Err(ZiporaError::invalid_data("select0 not implemented for ..."))`, and `space_overhead_percent()` returned `0.0`.

**Test.**
- `rank_select::tests::test_mixed_il_256_and_xl_256_and_se_512_boundary_and_select0` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Added early return `if pos >= self.size[dim] { return self.max_rank1[dim]; }` in `rank1_dim` and clamped `let pos = pos.min(self.size[dim]);` in `rank0_dim` across `RankSelectMixedIL256`, `RankSelectMixedXL256`, and `RankSelectMixedSE512`.
- Implemented `select0_dim` and `RankSelectOps::select0` (plus `space_overhead_percent()`) across all three mixed-dimension structures.

---

### C5.10 — `RankSelectInterleaved256` hung infinitely when `select_sample_rate == 0`, overflowed in `prefetch_ahead`, and used an $O(N)$ bit-by-bit scan when `select_cache` was enabled — HIGH

**Finding (`src/succinct/rank_select/interleaved.rs`).**
1. `RankSelectInterleaved256::with_options` accepted `select_sample_rate == 0`, causing `build_select_cache` to loop forever (`target_ones = ones_seen + 0 == ones_seen`) and `select1_cache_optimized` / `prefetch_select1` to divide by zero (`k / self.select_sample_rate`).
2. When `select_cache` was enabled (`Some`), `select1_cache_optimized` called `select1_linear_search`, which scanned bit-by-bit from `start_pos` in $O(\text{sample\_rate})$ instead of using the 256-bit line directory (`binary_search_lines_range` + `select1_within_line`).
3. `prefetch_ahead(base_pos, count, strategy)` computed `(start_line + count).min(self.lines.len())`, which overflowed `usize` when `count` was large (e.g., `usize::MAX`), wrapping around to a small `end_line`.

**Test.**
- `rank_select::tests::test_interleaved256_zero_sample_rate_and_hint_and_prefetch` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Validated `select_sample_rate > 0` in `RankSelectInterleaved256::with_options` (and guarded `select_sample_rate == 0` in `prefetch_select1`).
- Replaced `select1_linear_search` with sample-bracketed line binary search (`binary_search_lines_range`) + `select1_within_line`, and exposed `pub fn select1_from_hint` using the same $O(\log L)$ line search + PDEP word select (removing `#[allow(dead_code)]`).
- Used `start_line.saturating_add(count).min(self.lines.len())` and bounds-checked `start_line < self.lines.len()` in `prefetch_ahead`.
- Removed the 2 honesty markers in `interleaved.rs`.

---

### C5.11 — `RankSelectSE256`, `RankSelectSE512`, `RankSelectSimple`, `RankSelectFewOne`, `RankSelectFewZero`, `RankSelectAllZero`, `RankSelectAllOne`, and `MultiDimRankSelect` panicked or returned wrong counts on `pos > len()` — MEDIUM

**Finding (`separated.rs`, `separated_512.rs`, `simple.rs`, `few.rs`, `trivial.rs`, `multidim_simd.rs`).**
- While `Rank9` and `RankSelectInterleaved256` clamped `pos.min(self.len())` in `rank1` and `rank0`, `RankSelectSE256`, `RankSelectSE512`, `RankSelectSimple`, `RankSelectFewOne`, `RankSelectFewZero`, `RankSelectAllZero`, and `RankSelectAllOne` panicked via `assert!(pos <= self.size)` on `pos > self.size` and did not clamp `pos` in `rank0(pos)` (`pos - self.rank1(pos)`).
- `MultiDimRankSelect::bulk_rank_multidim` returned `0` instead of `count_ones()` when `positions[dim] > self.total_bits`.
- `RankSelectFewOne::get` and `RankSelectFewZero::get` cast `index as u32` before binary searching `self.positions`.

**Test.**
- `rank_select::tests::test_differential_all_rank_select_variants_and_dirty_trailing_words` and `test_interleaved256_zero_sample_rate_and_hint_and_prefetch` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Clamped `pos.min(self.size)` in `rank1` and `rank0` across `RankSelectSE256`, `RankSelectSE512`, `RankSelectSimple`, `RankSelectFewOne`, `RankSelectFewZero`, `RankSelectAllZero`, and `RankSelectAllOne`, and delegated all positions in `MultiDimRankSelect::bulk_rank_multidim` to `self.dimensions[dim].rank1(positions[dim])`.
- Replaced `binary_search(&(index as u32))` in `few.rs` with `binary_search_by_key(&index, |&p| p as usize)`.

---

### C5.12 — `builder::analyze_bit_vector`, `utils::benchmark_variants`, `AdaptiveRankSelect::select_implementation`, and `SeparatedStorageConfig` stubs & zero-division guards — LOW

**Finding (`builder.rs`, `mod.rs`, `adaptive.rs`, `config.rs`).**
1. `builder::analyze_bit_vector` was a `// TODO: Implement analysis logic` stub returning `BuilderOptions::default()`, and `RankSelectFactory::create_optimal` ignored its `opts: BuilderOptions` parameter.
2. `utils::benchmark_variants` had `// For now, return placeholder data` and returned hardcoded constants without touching `bit_vector` or `iterations`.
3. `utils::estimate_memory_overhead` panicked on `block_size == 0` or `select_sample_rate == 0` and returned `NaN` on `total_bits == 0`.
4. `SeparatedStorageConfig::validate` did not reject `select_sample_rate == 0`, and `analyze_and_optimize` computed `0.0 / 0.0 = NaN` on an empty `BitVector`.
5. `AdaptiveRankSelect::select_implementation` had a `(simplified to use best performer)` comment and ignored `criteria.prefer_space` and `profile.ones_count`.

**Test.**
- `rank_select::tests::test_interleaved256_zero_sample_rate_and_hint_and_prefetch` in `src/succinct/rank_select/mod.rs`

**Fix (`07536d7`).**
- Implemented `builder::analyze_bit_vector`, wired `opts` in `RankSelectFactory::create_optimal`, implemented real timing/overhead measurement in `utils::benchmark_variants`, wired `criteria.prefer_space` and `optimal_select_sample_rate` in `AdaptiveRankSelect::select_implementation`, guarded zero inputs in `estimate_memory_overhead` and `SeparatedStorageConfig::analyze_and_optimize`, and validated `select_sample_rate > 0` in `SeparatedStorageConfig::validate`.

---

### S17-R1..R4 — `HybridPostingList::intersect_count` progress guard on corrupt input, chunked Elias-Fano `Deserialize` structural validation, `#[ignore]` on >4 GiB CSPP test, and folded sortedness check — MEDIUM / LOW

**Finding.**
1. **`S17-R1` (MEDIUM, `src/succinct/elias_fano/hybrid.rs:199-214`)**: `HybridPostingList::intersect_count` looped forever on a corrupted deserialized input (`/tmp/s17_hang_mutant.json`) whose `chunk_upper_bounds` were non-monotone (`[312658, 8992919731896089034, 945839, 1133629]`), causing `next_geq(probe)` to return `va < probe`.
2. **`S17-R2` (LOW–MEDIUM, `src/succinct/elias_fano/{partitioned,optimal,clustered,hybrid}.rs`)**: Mutated serialized inputs for `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`, and `HybridPostingList` were accepted without structural validation and could panic on out-of-bounds slice/chunk indexing during queries.
3. **`S17-R3` (LOW, `src/fsa/cspp_trie.rs:2748-2779`)**: Part 3 of `test_value_arena_4gib_scaling_and_valsize_zero_and_mem_stat` allocated >4 GiB (~42 s) in both default gate lanes (`make sanity`).
4. **`S17-R4` (LOW, `src/succinct/elias_fano/{basic,partitioned,optimal,clustered,hybrid}.rs`)**: Running a separate `values.windows(2).all(|w| w[0] <= w[1])` pass before building `EliasFano` added ~16% overhead to `EliasFano::from_sorted`.

**Test.**
- `succinct::elias_fano::tests::test_corrupt_serde_get_and_pef_batch_cursor_no_ub_or_panic` in `src/succinct/elias_fano/tests.rs`

**Fix.**
- **`S17-R1`**: Added monotonic progress checks (`if va < probe { break; }` and `Some((_, vb)) if vb > va => probe = vb`) in `HybridPostingList::intersect_count`.
- **`S17-R2`**: Added `#[serde(try_from = "Raw...")]` structural validation on `Deserialize` for `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`, and `HybridPostingList` (`Dense`), rejecting mismatched chunk counts, out-of-bounds offsets/words, `low_bit_width > 63`, and non-monotone chunk bounds at deserialize time.
- **`S17-R3`**: Split the >4 GiB CSPP value-arena scaling test into `#[test] #[ignore] fn test_value_arena_4gib_scaling_beyond_u32_max()`.
- **`S17-R4`**: Folded the non-decreasing sortedness `assert!` into `from_sorted_impl` across `EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, and `ClusteredEliasFano`, and removed the redundant outer scan in `HybridPostingList`.

---

## Verification Summary

- **Gate (`touch src/lib.rs && make sanity`)**:
  - `cargo clippy --all-targets --all-features -- -D warnings`: **0 warnings**
  - `cargo clippy --no-default-features -- -D unused_variables -D unused_imports`: **0 warnings**
  - `scripts/unsafe_audit.py`: **28 undocumented** (unchanged ceiling `28`; `src/succinct/` was already 100% documented at `102/102` and is now **`119/119 = 100.0%`**, `src/fsa/` = **131/131 = 100.0%**, `src/memory/` = **338/338 = 100.0%**), **6 precondition violations** (unchanged ceiling `6`; `src/succinct/` = **0**)
  - `scripts/api_honesty.py`: **95 markers** (ceiling `95`, down from `102`; `src/succinct/` = **0**)
  - `scripts/index_audit.py`: **0 / 0** (ceiling `0`, down from `6`; `succinct_load: 6 -> 0`, `max_total: 6 -> 0`, `max_in_bounds = 132`)
  - `scripts/unwrap_audit.py`: **0 `.unwrap()` / 140 `.expect()`** (ceiling `140`, down from `141`)
  - Tests: **2,933 debug lib / 2,950 release lib / 228 doctests / 3,588 all-targets** (+11 net new RED-verified regression tests across `bit_vector`, `elias_fano`, `rank_select`, and `fsa`, plus 1 `#[ignore]`d >4 GiB test; 0 removed)
- **Miri Verification**:
  - `cargo +nightly miri test --lib succinct::bit_vector::tests`: **41/41 pass, 0 UB**
  - `cargo +nightly miri test --lib fsa::cspp_trie::tests`: **35/35 pass, 0 UB**
  - New `succinct::elias_fano` and `succinct::rank_select` regression tests (9/9): **0 UB**

