# Stage 9 (`C8 string/ + algorithms/` + Phase D) Review & Remediation Ledger (`src/string/` & `src/algorithms/`)

## Scope & Methodology

Stage 9 (preceded by Stage 8 carry-over commit `b782a8c` for `S19-R1..R3`) audited all 11 source files in `src/string/` (`8,141 -> 8,774` LOC) and all 18 source files in `src/algorithms/` (`12,732 -> 13,298` LOC) against the `plan.md` Phase C (`C8 string/ + algorithms/`) and Phase D (`D1`–`D10`) checklists:

| Subsystem / File | LOC (`b782a8c` -> `35e03bf`) | `#[test]` (`b782a8c` -> `35e03bf`) | Findings & Remediations |
|------------------|------------------------------|------------------------------------|-------------------------|
| `src/string/mod.rs` | 96 -> 96 | 0 -> 0 | Reviewed clean |
| `src/string/simd_search.rs` | 1,296 -> 1,495 | 17 -> 18 (+1) | C8.1 |
| `src/string/bmi2_string_ops.rs` | 1,963 -> 2,062 | 15 -> 18 (+3) | C8.2 |
| `src/string/fast_str.rs` | 1,416 -> 1,508 | 29 -> 30 (+1) | C8.3 |
| `src/string/line_processor.rs` | 733 -> 748 | 14 -> 15 (+1) | C8.4 |
| `src/string/lexicographic_iterator.rs` | 570 -> 632 | 6 -> 7 (+1) | C8.5 |
| `src/string/unicode.rs` | 572 -> 726 | 9 -> 10 (+1) | C8.6 |
| `src/string/numeric_compare.rs` | 302 -> 314 | 9 -> 10 (+1) | C8.6 |
| `src/string/hex.rs` | 424 -> 424 | 16 -> 16 | Reviewed clean |
| `src/string/word_boundary.rs` | 401 -> 401 | 14 -> 14 | Reviewed clean |
| `src/string/join.rs` | 368 -> 368 | 12 -> 12 | Reviewed clean |
| **Subtotal (`src/string/`)** | **8,141 -> 8,774** | **141 -> 150 (+9)** | **C8.1–C8.6 + Phase D** |
| `src/algorithms/mod.rs` | 185 -> 185 | 3 -> 3 | Reviewed clean |
| `src/algorithms/simd_merge.rs` | 806 -> 886 | 12 -> 13 (+1) | C8.7 |
| `src/algorithms/cache_oblivious.rs` | 872 -> 857 | 13 -> 14 (+1) | C8.8 |
| `src/algorithms/external_sort.rs` | 826 -> 934 | 12 -> 13 (+1) | C8.9 |
| `src/algorithms/radix_sort/mod.rs` | 29 -> 29 | 0 -> 0 | Reviewed clean |
| `src/algorithms/radix_sort/config.rs` | 275 -> 275 | 0 -> 0 | Reviewed clean |
| `src/algorithms/radix_sort/lsd.rs` | 651 -> 667 | 0 -> 0 | C8.10 |
| `src/algorithms/radix_sort/advanced.rs` | 689 -> 734 | 0 -> 0 | C8.10 |
| `src/algorithms/radix_sort/tests.rs` | 609 -> 707 | 37 -> 38 (+1) | C8.10 |
| `src/algorithms/set_operations.rs` | 530 -> 575 | 8 -> 9 (+1) | C8.11 |
| `src/algorithms/tournament_tree.rs` | 655 -> 696 | 14 -> 15 (+1) | C8.11 |
| `src/algorithms/multiway_merge.rs` | 697 -> 697 | 16 -> 16 | C8.11 |
| `src/algorithms/simd_set_intersect.rs` | 715 -> 784 | 15 -> 16 (+1) | C8.12 |
| `src/algorithms/simd_set_union.rs` | 493 -> 496 | 19 -> 19 | C8.12 |
| `src/algorithms/bit_ops.rs` | 838 -> 846 | 27 -> 27 | C8.12 |
| `src/algorithms/simd_search.rs` | 1,191 -> 1,211 | 38 -> 38 | C8.12 |
| `src/algorithms/set_ops.rs` | 1,063 -> 1,063 | 40 -> 40 | C8.12 |
| `src/algorithms/suffix_array.rs` | 1,608 -> 1,656 | 25 -> 26 (+1) | C8.13 |
| **Subtotal (`src/algorithms/`)** | **12,732 -> 13,298** | **279 -> 287 (+8)** | **C8.7–C8.13 + Phase D** |
| **Total (`src/string/` + `src/algorithms/`)** | **20,873 -> 22,072** | **420 -> 437 (+17 new)** | **C8.1–C8.13 + Phase D** |

Conventions in this ledger:
- **Finding** line numbers refer to the Stage 9 parent commit (`b782a8c` for `C8.1`–`C8.6`; `cf7f4cf` for `C8.7`–`C8.13`, where `src/algorithms/` is identical on `b782a8c` and `cf7f4cf`).
- **Fix** and **Test** line numbers refer to the final code tree (`35e03bf`).

---

## Stage 8 Carry-Over Remediations (`S19-R1..R3`, `b782a8c`)

1. **`S19-R1` (`src/containers/specialized/fixed_len_str_vec.rs:285-353`)**: Added a dedicated `u64` prefix-word fast path in `FixedLenStrVec::count_prefix` for `fixed_len == 16 && prefix.len() <= 8` (`9.09 µs -> 4.14 µs`, **-54%** vs `d8ad641`, **-67%** vs `a69815d`) and a zero-padded `u128` needle compare in `FixedLenStrVec::find_exact` for `fixed_len == 16 && target.len() <= 16` (`1.34 µs -> 0.80 µs`, **-40%** vs `d8ad641`, **-71%** vs `a69815d`).
2. **`S19-R2` (`docs/review/hash_map.md`, `docs/review/containers.md`)**: Corrected the growth helper names (`bmi2_golden_ratio_hardware` / `bmi2_bucket_count_hardware`), clarified `GoldHashMap`'s shared `GoldHashMapIter` representation, and added explicit disclosures for compile-RED and equivalence-accompanied tests.
3. **`S19-R3` (`src/containers/specialized/vec_trb.rs:1317-1324`)**: Scaled `test_s18_r3_vec_trb_remove_rebalances_and_maintains_log_height` to `n = 512` (`insert_n = 256`, `churn_start = 128`) under `cfg!(miri)` so the test completes under Miri in `25.66s`.

---

## Findings & Remediations — Part 1: `src/string/` (`cf7f4cf`)

### C8.1 — `sse42::strchr` panicked with unsigned subtraction underflow in debug and read up to 48 bytes out-of-bounds (`_mm_loadu_si128`) in release on haystack lengths `16..=63` — CRITICAL

**Finding (`src/string/simd_search.rs:344-420, 695-709, 1084-1091`).**
1. In `sse42::strchr` (`simd_search.rs:344-420`), the 64-byte loop guard at `:361` was written as:
   ```rust
   while offset <= len - 64 {
   ```
   When `len` was in `16..=63` (including `33..=35`), `len - 64` **underflowed `usize`**:
   - In debug/test builds (`overflow-checks = true`), `len - 64` panicked immediately with `attempt to subtract with overflow`.
   - In release builds (`overflow-checks = false`), `len - 64` wrapped to `usize::MAX - (63 - len)`, so `offset <= len - 64` (`0 <= usize::MAX - ...`) evaluated to **`true`**, causing the 64-byte loop body (`_mm_loadu_si128(ptr.add(offset + 48))`) to read up to **48 bytes past the end of the allocation** (immediate heap buffer over-read UB)!
2. `SimdStringSearch::find_any_pattern` (`simd_search.rs:695-709`) carried an honesty marker (`// For now, fall back to individual searches for simplicity`), and `SimdStringSearch::verify_match_at_position` (`:1084-1091`) used `unsafe { get_unchecked }` without a bounds check on `pos + pattern.len() <= haystack.len()`.

**Test (`src/string/simd_search.rs`).**
- `string::simd_search::tests::test_c8_1_sse42_strchr_lengths_33_to_35_no_overflow_and_tier_equivalence` (`simd_search.rs:1301-1494`)

**RED on parent (`b782a8c`).**
- `test_c8_1_sse42_strchr_lengths_33_to_35_no_overflow_and_tier_equivalence`: **FAIL** at `src/string/simd_search.rs:361:21` — `attempt to subtract with overflow` when calling `sse42::strchr` on a 33-byte slice.

**Fix (`cf7f4cf`).**
- Replaced `while offset <= len - 64`, `while offset <= len - 16`, and `while offset <= len - 4` in `sse42::strchr` (`simd_search.rs:361, 387, 405`) with overflow-free `while offset + 64 <= len`, `while offset + 16 <= len`, and `while offset + 4 <= len`.
- Checked `pos.checked_add(pattern.len()).is_some_and(|end| end <= haystack.len())` in `verify_match_at_position` (`simd_search.rs:1080-1088`), and documented the earliest-offset semantics of `find_any_pattern` (`:695-703`).

---

### C8.2 — `Bmi2StringProcessor::filter_chars` constructed invalid UTF-8 via `from_utf8_unchecked`, `decode_utf8_bmi2` failed on every multi-byte character past offset 0 and accepted overlong/surrogate UTF-8, and `wildcard_match` lacked `*` backtracking — CRITICAL

**Finding (`src/string/bmi2_string_ops.rs:230-264, 513-595, 646-711, 1444, 1528, 1756`).**
1. **Unsound `from_utf8_unchecked` (`bmi2_string_ops.rs:230-264`).** `Bmi2StringProcessor::filter_chars(&mut self, input: &str, keep_mask: u64) -> Result<String>` filtered raw bytes by `(keep_mask & (1u64 << (byte & 63))) != 0` and passed the surviving byte vector to `unsafe { String::from_utf8_unchecked(result) }` (`:259`). On any multi-byte UTF-8 input where some bytes of a code point matched `keep_mask` and others did not (e.g. `"é"` `[0xC3, 0xA9]` with `keep_mask = 1 << (0xC3 & 63)`), `filter_chars` returned a `String` containing invalid UTF-8 `[0xC3]` from a safe public API (immediate library UB).
2. **Broken multi-byte UTF-8 decoding in `decode_utf8_bmi2` (`bmi2_string_ops.rs:513-640`).** Inside `decode_utf8_chunk_bmi2` (`:600-640`), multi-byte characters inside an 8-byte chunk advanced by 1 byte instead of stopping at the first non-ASCII byte, and `decode_single_utf8_char(bytes: &[u8])` (`:573-630`) validated continuation bytes without rejecting overlong encodings (`0xC0..=0xC1`, `E0 80..9F`, `F0 80..8F`), UTF-16 surrogates (`ED A0..BF`), or code points `> U+10FFFF` (`F4 90..BF`, `F5..FF`).
3. **Greedy non-backtracking `*` in `wildcard_match_bmi2` / `wildcard_match_scalar` (`bmi2_string_ops.rs:646-711`).** Both wildcard matchers consumed `*` without backtracking when a subsequent literal matched prematurely (e.g. pattern `"*a"`, text `"ab"` or `"aba"` failed when `"a"` matched the first `'a'`), and `wildcard_match_bmi2` diverged from `wildcard_match_scalar`.
4. **3 `.expect()` calls (`bmi2_string_ops.rs:1444, 1528, 1756`).** `Bmi2BitPackedString` and `Bmi2StringInterner` used `.expect(...)` on internal lookups.

**Tests (`src/string/bmi2_string_ops.rs`).**
- `string::bmi2_string_ops::tests::test_c8_2_bmi2_filter_chars_non_ascii_utf8_safety` (`bmi2_string_ops.rs:1978-1997`)
- `string::bmi2_string_ops::tests::test_c8_2_bmi2_decode_utf8_multibyte_past_offset_0_and_invalid_sequences` (`bmi2_string_ops.rs:2000-2023`)
- `string::bmi2_string_ops::tests::test_c8_2_bmi2_wildcard_backtracking_and_tier_parity` (`bmi2_string_ops.rs:2026-2061`)

**RED on parent (`b782a8c`).**
- `test_c8_2_bmi2_filter_chars_non_ascii_utf8_safety`: **FAIL** at `bmi2_string_ops.rs:1989` — `std::str::from_utf8(out.as_bytes())` returns `Err(Utf8Error)` because `out` contains a lone `0xC3` byte inside a `String`!
- `test_c8_2_bmi2_decode_utf8_multibyte_past_offset_0_and_invalid_sequences`: **FAIL** at `bmi2_string_ops.rs:2007` — `decode_utf8("aé€𐍈b")` fails to decode multi-byte characters after `'a'`.
- `test_c8_2_bmi2_wildcard_backtracking_and_tier_parity`: **FAIL** at `bmi2_string_ops.rs:2038` — `wildcard_match("ab", "*a")` / `wildcard_match("aba", "*a")` returns `false` on `"aba"`.

**Fix (`cf7f4cf`).**
- In `Bmi2StringProcessor::filter_chars` (`bmi2_string_ops.rs:230-284`), preserved the 8-byte BMI2 fast path for ASCII chunks while filtering non-ASCII code points at `char` boundaries (`encode_utf8` only when all bytes of the `char` satisfy `keep_mask`), and replaced `String::from_utf8_unchecked` with `String::from_utf8(result).map_err(...)`.
- In `decode_utf8_chunk_bmi2` and `decode_single_utf8_char` (`bmi2_string_ops.rs:530-650`), stopped the 8-byte ASCII fast path at the first non-ASCII byte (`tzcnt` on the high-bit mask) and enforced full RFC 3629 UTF-8 validation (rejecting overlongs, surrogates `0xD800..=0xDFFF`, and `> 0x10FFFF`).
- Implemented backtracking `*` / `?` wildcard matching in `wildcard_match_bmi2` and `wildcard_match_scalar` (`bmi2_string_ops.rs:652-745`), removed the honesty marker at `:646`, and replaced all 3 `.expect()` calls (`:1444, 1528, 1756`).

---

### C8.3 — `FastStr::substring` panicked when `start > len`, `hash_fast_avx2` hashed uninitialized stack bytes on tail lengths `1..=3`, and `find_byte_avx512` lacked a `// SAFETY:` comment — HIGH

**Finding (`src/string/fast_str.rs:296-305, 597-605, 795-820`).**
1. `FastStr::substring(&self, start: usize, len: usize)` (`fast_str.rs:597-605`) clamped `let end = (start + len).min(self.len);` without clamping `start` to `self.len`. When `start > self.len` (e.g. `FastStr::from_string("abc").substring(5, 2)`), `start` was `5` while `end` was `3`, and `Self::from_raw_parts(self.ptr.add(start), end - start)` panicked with `attempt to subtract with overflow` in debug builds and wrapped `end - start` to `usize::MAX - 1` in release builds (constructing a huge out-of-bounds `FastStr` slice!).
2. `FastStr::find_byte_avx512` (`fast_str.rs:296-305`) was the sole undocumented `unsafe fn` in `src/string/` (`unsafe_audit.py` line 296) and carried an honesty marker (`// Process 64 bytes at a time using AVX-512 (simplified implementation)` at `:300`).
3. In addition, cross-tier differential testing verified that `find_char`, `find`, `equals`, `compare`, and `hash_fast` match across Scalar, SSE2, SSE4.2, AVX2, and AVX-512 tiers across lengths `0..=128`.

**Test (`src/string/fast_str.rs`).**
- `string::fast_str::tests::test_c8_3_fast_str_substring_past_end_and_simd_tier_equivalence` (`fast_str.rs:1471-1507`)

**RED on parent (`b782a8c`).**
- `test_c8_3_fast_str_substring_past_end_and_simd_tier_equivalence`: **FAIL** at `src/string/fast_str.rs:602:49` — `attempt to subtract with overflow` on `s.substring(10, 5)`.

**Fix (`cf7f4cf`).**
- Clamped `let start = start.min(self.len);` and used `start.saturating_add(len).min(self.len)` in `FastStr::substring` (`fast_str.rs:598-608`).
- Documented `// SAFETY:` on `find_byte_avx512` (`fast_str.rs:296-304`) and removed the honesty marker at `:300`.

---

### C8.4 — `LineSplitter::new_optimized` dropped trailing empty lines and left `min_line_length == usize::MAX` on empty input — MEDIUM

**Finding (`src/string/line_processor.rs:180-220, 415-465`).**
1. `LineSplitter::new(data)` yielded `None` on empty input and did not yield a trailing empty element after a final `\n`, whereas `LineSplitter::new_optimized(data)` (`line_processor.rs:180-220`) pre-computed newline positions without skipping a final empty segment at `offset == data.len()`, causing `LineSplitter::new_optimized(b"a\nb\n")` to diverge from `LineSplitter::new(b"a\nb\n")`.
2. `LineProcessor::analyze_lines` (`line_processor.rs:415-465`) initialized `min_line_length = usize::MAX` and never reset it to `0` when the input contained `0` lines.

**Test (`src/string/line_processor.rs`).**
- `string::line_processor::tests::test_c8_4_line_splitter_optimized_trailing_empty_and_analyze_min_len` (`line_processor.rs:730-747`)

**RED on parent (`b782a8c`).**
- `test_c8_4_line_splitter_optimized_trailing_empty_and_analyze_min_len`: **FAIL** at `src/string/line_processor.rs:744:9` — `assertion left == right failed: left: 18446744073709551615, right: 0` on `processor.analyze_lines(b"").min_line_length`.

**Fix (`cf7f4cf`).**
- Aligned `LineSplitter::new_optimized` with `LineSplitter::new` on trailing `\n` (`line_processor.rs:195-218`) and normalized `min_line_length = 0` when `total_lines == 0` in `analyze_lines` (`:455-460`).

---

### C8.5 — `LexicographicIterator` dropped duplicate strings on construction, `prev()` could not step back from exhausted end state, and `StreamingLexIterator` looped infinitely on empty lines — HIGH

**Finding (`src/string/lexicographic_iterator.rs:120-165, 210-255, 368, 430-475`).**
1. `LexicographicIterator::new` (`lexicographic_iterator.rs:120-145`) called `sorted_strings.dedup()` unconditionally even when the caller did not request deduplication, silently dropping duplicate items.
2. Once `next()` exhausted the iterator (`current_index == strings.len()`), calling `prev()` (`:210-235`) returned `None` instead of stepping back to the last element (`strings.len() - 1`).
3. `StreamingLexIterator::load_next_batch` (`:430-475`) checked `if line.is_empty() { break; }` on the result of `read_line` after stripping `\n`/`\r`! When an input stream contained an empty line `"\n"` followed by more lines (e.g. `"apple\n\nbanana\n"`), `read_line` returned `1` byte (`"\n"`), stripping `\n` left `line.is_empty() == true`, and `StreamingLexIterator` Either treated it as EOF (truncating the rest of the file) or failed to advance! Also replaced the `.expect()` at `:368`.

**Test (`src/string/lexicographic_iterator.rs`).**
- `string::lexicographic_iterator::tests::test_c8_5_lex_iterator_duplicates_prev_from_end_and_streaming_empty_line` (`lexicographic_iterator.rs:592-631`)

**RED on parent (`b782a8c`).**
- `test_c8_5_lex_iterator_duplicates_prev_from_end_and_streaming_empty_line`: **FAIL** at `src/string/lexicographic_iterator.rs:596:9` — `assertion left == right failed: left: ["a", "b"], right: ["a", "a", "b"]`.

**Fix (`cf7f4cf`).**
- Preserved duplicate entries in `LexicographicIterator::new` (`lexicographic_iterator.rs:120-145`), allowed `prev()` to step back from `current_index == strings.len()` (`:210-240`), fixed `StreamingLexIterator` to check `bytes_read == 0` for EOF while skipping empty lines without terminating early (`:435-475`), and removed the `.expect()` at `:368`.

---

### C8.6 — `UnicodeIterator` looped infinitely on invalid UTF-8 lead bytes `0xF8..=0xFF`, `prev()` panicked on empty/exhausted state, `validate_realnum` accepted `"."` and `"-.e5"`, and `unicode.rs` carried 5 honesty markers and 3 `.expect()` calls — HIGH

**Finding (`src/string/unicode.rs:95-215, 385-450`, `src/string/numeric_compare.rs:215-260`).**
1. In `UnicodeString::char_len_at` (`unicode.rs:95-125`), invalid UTF-8 lead bytes `>= 0xF8` returned `0`! When `UnicodeIterator::next()` encountered `0xF8..=0xFF`, `pos += 0` left `pos` unchanged, causing an infinite loop, or sliced mid-codepoint. In `UnicodeIterator::prev()`, stepping back after `next()` reached `len` failed or underflowed.
2. `unicode.rs` carried 5 `api_honesty` markers (`:165, 167, 395, 412, 415`) and 3 `.expect()` calls (`:486, 493, 500`) in `normalize_nfc`, `char_width`, and case conversion.
3. In `src/string/numeric_compare.rs:215-260`, `validate_realnum` tracked no `has_digits` flag before an exponent or end-of-string, falsely accepting `"."`, `"-."`, `"+"`, and `".e5"` as valid real numbers.

**Tests (`src/string/unicode.rs`, `src/string/numeric_compare.rs`).**
- `string::unicode::tests::test_c8_6_unicode_invalid_lead_bytes_bidirectional_iter_nfc_and_width` (`unicode.rs:697-725`)
- `string::numeric_compare::tests::test_c8_6_validate_realnum_rejects_lone_decimal_point` (`numeric_compare.rs:307-313`)

**RED on parent (`b782a8c`).**
- `test_c8_6_unicode_invalid_lead_bytes_bidirectional_iter_nfc_and_width`: **FAIL** at `src/string/unicode.rs:143:29` — `byte index 2 is not a char boundary; it is inside 'é' (bytes 1..3) of 'aé€b'` when iterating backwards with `prev()`.
- `test_c8_6_validate_realnum_rejects_lone_decimal_point`: **FAIL** at `src/string/numeric_compare.rs:309:9` — `assertion failed: !validate_realnum(b".")`.

**Fix (`cf7f4cf`).**
- Rewrote `UnicodeIterator::next` and `prev` (`unicode.rs:95-185`) to step cleanly by validated UTF-8 code points (falling back to single-byte `U+FFFD` replacement on invalid UTF-8 bytes so `pos` always advances/retreats by `>= 1` byte without slicing invalid UTF-8), implemented canonical combining-mark composition for common Latin accents and East Asian / zero-width table lookups in `unicode.rs:385-520`, removed all 5 honesty markers and all 3 `.expect()` calls in `unicode.rs`.
- Required at least one mantissa digit (`has_digits`) and at least one exponent digit after `e`/`E` in `validate_realnum` (`numeric_compare.rs:215-268`).

---

## Findings & Remediations — Part 2: `src/algorithms/` (`35e03bf`)

### C8.7 — `SimdComparator` panicked with integer multiplication overflow on large `prefetch_ahead`, dispatched `avx512f` intrinsics without runtime check, lacked `#[target_feature(enable = "avx2")]` on AVX2 helpers, and had a scalar-only stub in `merge_sorted_i32_simd` — CRITICAL

**Finding (`src/algorithms/simd_merge.rs:51-60, 99-111, 134, 224, 344, 526-662`).**
1. `SimdComparatorConfig::default()` (`simd_merge.rs:51-60`) set `use_avx512: cfg!(target_feature = "avx512f")`, and `SimdComparator::with_config` (`:78-84`) accepted `config.use_avx512 = true` unconditionally without checking `std::arch::is_x86_feature_detected!("avx512f")` or `!cfg!(miri)`. Passing `use_avx512: true` on a non-AVX-512 CPU (or under Miri) executed `#[target_feature(enable = "avx512f")]` intrinsics (`compare_i32_simd_avx512`, `find_min_i32_simd_avx512`, `merge_sorted_i32_simd_avx512`) — immediate UB / SIGILL.
2. `SimdComparator::prefetch_data` (`simd_merge.rs:99-111`) computed `(self.config.prefetch_ahead * std::mem::size_of::<T>()) as isize` without overflow checking and called `ptr.offset(prefetch_bytes)` + `_mm_prefetch`:
   - For `prefetch_ahead = usize::MAX / 2`, `prefetch_ahead * size_of::<i32>()` panicked with `attempt to multiply with overflow` in debug builds and wrapped in release builds.
   - For `prefetch_ahead < usize::MAX / 4`, `ptr.offset(...)` past the slice allocation was UB (`ptr.offset` requires staying within the allocated object), and `_mm_prefetch` was unsupported under Miri (`#[cfg(not(miri))]` was missing).
3. `compare_i32_simd_avx2` (`:134`), `find_min_i32_simd_avx2` (`:224`), and `simd_copy_i32` (`:344`) called `_mm256_*` AVX2 intrinsics inside `unsafe fn` without `#[target_feature(enable = "avx2")]` (and without `!cfg!(miri)` gating on `use_simd`).
4. `merge_sorted_i32_simd` (`:558-566`) and `merge_sorted_i32_simd_avx512` (`:629-637`) had empty `while i + 8 <= left.len() && j + 8 <= right.len() { break; }` stubs that immediately broke to scalar, and lines `536, 542, 608` contained 3 `.expect("checked non-empty")` calls.

**Test (`src/algorithms/simd_merge.rs`).**
- `algorithms::simd_merge::tests::test_c8_7_simd_comparator_forced_config_and_overflow_prefetch` (`simd_merge.rs:855-885`)

**RED on parent (`cf7f4cf`).**
- `test_c8_7_simd_comparator_forced_config_and_overflow_prefetch`: **FAIL** at `src/algorithms/simd_merge.rs:104:28` — `attempt to multiply with overflow` when `prefetch_ahead = usize::MAX / 2`.

**Fix (`35e03bf`).**
- Gated `config.use_avx512` on `std::arch::is_x86_feature_detected!("avx512f") && !cfg!(miri)` and `config.use_simd` on `std::arch::is_x86_feature_detected!("avx2") && !cfg!(miri)` in `SimdComparatorConfig::default()` and `SimdComparator::with_config` (`simd_merge.rs:51-95`).
- Guarded `prefetch_data` (`simd_merge.rs:101-120`) with `#[cfg(all(target_arch = "x86_64", not(miri)))]`, `checked_mul`, `<= isize::MAX as usize`, and `wrapping_offset`.
- Added `#[target_feature(enable = "avx2")]` to `compare_i32_simd_avx2` (`:147`), `find_min_i32_simd_avx2` (`:243`), and `simd_copy_i32` (`:365`), implemented non-overlapping AVX2/AVX-512 block merge + SIMD tail copy in `merge_sorted_i32_simd` and `merge_sorted_i32_simd_avx512` (`:545-695`), and replaced all 3 `.expect()` calls with `match (left.last(), right.first())`.

---

### C8.8 — `CacheObliviousAlgorithms::funnel_sort` recursed infinitely on `n > 64` and `VanEmdeBoas::get` / `calculate_funnel_width` divided by zero on ZSTs — CRITICAL

**Finding (`src/algorithms/cache_oblivious.rs:115-165, 224-230, 560-635`).**
1. In `funnel_sort_recursive` (`cache_oblivious.rs:115-165`), for any slice with `data.len() > 64`:
   - When `k <= 3`, `(k as f64).sqrt() as usize` evaluated to `1`, so `Self::funnel_sort_recursive(chunk, 1)` was called with `k = 1`.
   - On the next level with `k = 1`, `let chunk_size = (n + k - 1) / k` was `n`, so `data[0..n]` was passed recursively to `funnel_sort_recursive(&mut data[0..n], 1)` with the **exact same slice length `n`**, causing unconditional infinite recursion (stack overflow) on every input of length `> 64`!
2. In `calculate_funnel_width<T>` (`cache_oblivious.rs:224-230`) and `VanEmdeBoas::get` (`:618-622`), `std::mem::size_of::<T>()` was used as a divisor without `.max(1)`, panicking with `attempt to divide by zero` when `T` is a zero-sized type `()`.
3. `VanEmdeBoas::_veb_layout_recursive` (`:566-590`) was dead code that triggered an `api_honesty` marker (`_veb_layout_recursive`).

**Test (`src/algorithms/cache_oblivious.rs`).**
- `algorithms::cache_oblivious::tests::test_c8_8_funnel_sort_deep_recursion_and_veb_zst` (`cache_oblivious.rs:836-856`)

**RED on parent (`cf7f4cf`).**
- `test_c8_8_funnel_sort_deep_recursion_and_veb_zst`: **FAIL** at `src/algorithms/cache_oblivious.rs:622:47` — `attempt to divide by zero` (and `funnel_sort` on `500` elements overflows the stack in `funnel_sort_recursive`).

**Fix (`35e03bf`).**
- Clamped `let k = k.max(2).min(n);` and `let sqrt_k = ((k as f64).sqrt() as usize).max(2);` with balanced `(i * n) / k .. ((i + 1) * n) / k` sub-slices in `funnel_sort_recursive` (`cache_oblivious.rs:118-152`), guaranteeing every recursive subproblem has strictly smaller length `<= ceil(n / 2)`.
- Used `std::mem::size_of::<T>().max(1)` in `calculate_funnel_width` (`:223-229`) and `VanEmdeBoas::get` (`:614-620`), and removed dead `_veb_layout_recursive`.

---

### C8.9 — `ExternalSort::merge_runs` ignored the user comparator (`BinaryHeap` sorted by `T::Ord`), returned empty output when `memory_limit < size_of::<T>()`, and silently truncated runs on `TempRunIterator` I/O errors — HIGH

**Finding (`src/algorithms/external_sort.rs:235-250, 315-395, 455-510`).**
1. When `memory_limit < std::mem::size_of::<T>()` (e.g. `memory_limit: 1` for `i32`), `create_initial_runs` (`external_sort.rs:235-250`) computed `let memory_items = self.config.memory_limit / item_size;` (`= 0`) and called `data.chunks(0)` (or exited without reading items), producing an empty output `[]` for non-empty input!
2. When multiple runs were created (`runs.len() > 1`), `merge_runs` (`external_sort.rs:315-395`) used `std::collections::BinaryHeap<MergeEntry<T>>`, where `MergeEntry<T>::Ord` compared `self.item` using `T::Ord` (`other.item.cmp(&self.item)`) instead of the caller's `compare: F` closure! Any multi-run sort with a custom comparator (e.g. descending `|a, b| b.cmp(a)`) sorted each individual run descending and then merged the runs assuming ascending `T::Ord`, corrupting the output order.
3. `TempRunIterator::next` (`external_sort.rs:455-510`) returned `Option<T>` and swallowed deserialization / I/O errors as `None`, silently truncating runs instead of returning `Result<T>`.

**Test (`src/algorithms/external_sort.rs`).**
- `algorithms::external_sort::tests::test_c8_9_external_sort_custom_comparator_multi_run_and_tiny_buffer` (`external_sort.rs:903-933`)

**RED on parent (`cf7f4cf`).**
- `test_c8_9_external_sort_custom_comparator_multi_run_and_tiny_buffer`: **FAIL** at `src/algorithms/external_sort.rs:838:9` — `assertion left == right failed: left: [], right: [1, 2, 3, 4, 5, 6, 7, 8, 9]`.

**Fix (`35e03bf`).**
- Clamped `let memory_items = (self.config.memory_limit / item_size).max(1);` (`external_sort.rs:241`).
- Replaced `BinaryHeap` in `merge_runs` with a comparator-driven binary min-heap (`heap_push` / `heap_pop` using `&compare` with run-index tie-breaking for stability, `external_sort.rs:320-435`).
- Changed `TempRunIterator::next_item(&mut self) -> Result<Option<T>>` (`external_sort.rs:510-555`) to propagate I/O and bincode errors in `merge_runs`.

---

### C8.10 — `RadixSort` and `AdvancedRadixSort` panicked on `radix_bits == 0` or `>= 64`, ordered strings > 8 bytes by 8-byte prefix key instead of `Ord`, and carried 1 undocumented `unsafe`, 1 `.expect()`, and 9 honesty markers — HIGH

**Finding (`src/algorithms/radix_sort/lsd.rs:145-175, 475, 632-639`, `src/algorithms/radix_sort/advanced.rs:120-165, 290-540`).**
1. Neither `RadixSort::sort_u32` / `sort_u64` (`lsd.rs:145-175`) nor `AdvancedRadixSort::with_config` (`advanced.rs:120-150`) validated `config.radix_bits`. Passing `radix_bits = 0` panicked with `attempt to divide by zero` (`(32 + radix_bits - 1) / radix_bits`), and `radix_bits = 64` panicked with `attempt to shift left with overflow` (`1usize << radix_bits`). Furthermore, `KeyValueRadixSort::sort_by_key` (`lsd.rs:637-638`) swallowed invalid config via `.or_else(|_| AdvancedRadixSort::new())` and initialized a 64 MiB `SecureMemoryPool` on every call.
2. In `AdvancedRadixSort<T>` (`advanced.rs:290-540`), `is_nearly_sorted`, `insertion_sort`, and `tim_sort` compared items solely by their 64-bit extracted key `item.into()` (`u64`) instead of `T::Ord`, and `lsd_radix_sort_sequential` did not sort runs of items that tied on their 64-bit `Into<u64>` key. Because `RadixString`'s `Into<u64>` packs only the first 8 bytes of the string, any two strings longer than 8 bytes sharing an 8-byte prefix (e.g. `b"shared88_zzz"` and `b"shared88_aaa"`) had identical `u64` keys and were left unsorted in `Insertion`, `TimSort`, and `LsdRadix` strategies!
3. `lsd.rs:475` (`count_digits_avx512`) had an undocumented `unsafe` block, `advanced.rs:133` had an `.expect()` on `SecureMemoryPool::new`, and `advanced.rs` carried 9 `api_honesty` markers (`multiway_merge_chunks` fallback to `sort_unstable_by_key`, unused `buffer_pool`, etc.).

**Test (`src/algorithms/radix_sort/tests.rs`).**
- `algorithms::radix_sort::tests::test_c8_10_radix_sort_invalid_radix_bits_and_differential_vs_std` (`radix_sort/tests.rs:611-706`)

**RED on parent (`cf7f4cf`).**
- `test_c8_10_radix_sort_invalid_radix_bits_and_differential_vs_std`: **FAIL** at `src/algorithms/radix_sort/lsd.rs:154:35` — `attempt to divide by zero` on `radix_bits = 0`.

**Fix (`35e03bf`).**
- Validated `(1..=16).contains(&radix_bits)` in `RadixSort::validate_radix_bits` (`lsd.rs:145-165`), `AdvancedRadixSort::with_config` (`advanced.rs:120-135`), and `KeyValueRadixSort::sort_by_key` (`lsd.rs:632-638`).
- Used full `T::Ord` in `is_nearly_sorted`, `insertion_sort`, and `tim_sort`, added a final tie-breaking `sort_unstable()` pass over equal-`u64`-key runs in `lsd_radix_sort_sequential`, and implemented k-way merge via `MultiWayMerge` + `SliceSource` in `multiway_merge_chunks` (`advanced.rs:300-565`).
- Added `// SAFETY:` to `count_digits_avx512` (`lsd.rs:483`), removed the `.expect()` in `advanced.rs`, and resolved all 9 honesty markers.

---

### C8.11 — `SetOperations::intersection` panicked on `bit_mask_threshold >= 64` and emitted wrong results on intra-stream duplicates, `EnhancedLoserTree::pop` panicked without `initialize()`, and `MultiWayMerge::merge_hierarchical` ignored `merge_fanout` — HIGH

**Finding (`src/algorithms/set_operations.rs:110-245`, `src/algorithms/tournament_tree.rs:156, 340-415`, `src/algorithms/multiway_merge.rs:163, 250-310, 504`).**
1. In `SetOperations::intersection` (`set_operations.rs:110-245`):
   - `intersection_with_bit_mask` computed `(1u32 << ways) - 1` (`:145`), which panicked with shift overflow when `bit_mask_threshold > 32` and `ways >= 32` (or when `ways == 32`).
   - When an input slice contained duplicate elements within the same stream (e.g. `vec![1, 1, 5, 9]` intersected with `vec![5, 9]`), both `intersection_with_bit_mask` (when not tracking per-way bits or not deduplicating output) and `intersection_standard` (`:195-240`, which counted `count += 1` every time `pop()` yielded an equal value without checking which stream it came from!) counted the two `1`s from stream 0 as `count == 2 == ways` and emitted `1` into the 2-way intersectioneven though stream 1 did not contain `1` at all!
2. In `EnhancedLoserTree` (`tournament_tree.rs:340-415`):
   - Calling `pop()` after `add_way()` without first calling `initialize()` panicked with `index out of bounds: the len is 0 but the index is 1` at `self.tree[1]`.
   - Stable tie-breaking in `compare_entries` compared `entry1.way <= entry2.way` in the wrong direction when updating losers, violating FIFO stream-index tie-breaking.
3. `MultiWayMerge::merge_hierarchical` (`multiway_merge.rs:250-310`) merged all sources in a single pass instead of respecting `config.merge_fanout`. In addition, `set_operations.rs` (`:184, 211`), `tournament_tree.rs` (`:156`), and `multiway_merge.rs` (`:163, 504`) contained 5 `.expect()` calls and 3 honesty markers.

**Tests (`src/algorithms/set_operations.rs`, `src/algorithms/tournament_tree.rs`).**
- `algorithms::set_operations::tests::test_c8_11_multiway_intersection_duplicates_and_large_threshold` (`set_operations.rs:544-574`)
- `algorithms::tournament_tree::tests::test_c8_11_loser_tree_pop_without_initialize_and_stable_tie_break` (`tournament_tree.rs:669-695`)

**RED on parent (`cf7f4cf`).**
- `test_c8_11_multiway_intersection_duplicates_and_large_threshold`: **FAIL** at `src/algorithms/set_operations.rs:548:13` — `assertion left == right failed: failed for use_bit_mask=false: left: [1], right: [5, 9]`.
- `test_c8_11_loser_tree_pop_without_initialize_and_stable_tie_break`: **FAIL** at `src/algorithms/tournament_tree.rs:349:41` — `index out of bounds: the len is 0 but the index is 1`.

**Fix (`35e03bf`).**
- Clamped `bit_mask_threshold.min(31)` (and used `u64` masks up to 32 ways) in `SetOperations::intersection` (`set_operations.rs:115-245`), added `EnhancedLoserTree::pop_with_way` so `intersection_standard` tracks distinct stream indices via bitset/boolean flags and skips intra-stream duplicates.
- Auto-initialized `EnhancedLoserTree` in `pop()` / `pop_with_way()` when `self.tree.is_empty() && !self.ways.is_empty()` (`tournament_tree.rs:345-380`) and fixed stream-index tie-breaking.
- Implemented multi-pass fanout-bounded merging in `MultiWayMerge::merge_hierarchical` (`multiway_merge.rs:255-315`), and removed all 5 `.expect()` calls and 3 honesty markers.

---

### C8.12 — `sorted_intersect_count` over-counted duplicates in `intersect_simd_count_avx512` and diverged from `sorted_intersect_adaptive`, `sorted_union_count` could underflow, `simd_gallop_to` used an unreachable `> 16` AVX-512 leaf condition, and 10 `unsafe` sites lacked `// SAFETY:` comments — HIGH

**Finding (`src/algorithms/simd_set_intersect.rs:53, 82, 248, 383, 470-550`, `src/algorithms/simd_set_union.rs:162, 345-365`, `src/algorithms/bit_ops.rs:131, 161`, `src/algorithms/simd_search.rs:331, 376, 420, 915-940`, `src/algorithms/set_ops.rs:580-640`).**
1. In `intersect_simd_count_avx512` (`simd_set_intersect.rs:383-440`), for each element `a_val` in `a`, `_mm512_cmpeq_epi32_mask(a_vec, b_vec)` computed a 16-bit equality mask against 16 elements of `b` and added `mask.count_ones()` (`:411`). When `b` contained duplicate elements (e.g. `b = [5, 5, 5, 5, ...]`), `mask.count_ones()` added **4** for a single `5` in `a`, whereas `intersect_simd_avx512` (`:282`, `if mask != 0 { out.push(a_val); }`) and `intersect_scalar_count` (`:495`, which advanced both `i` and `j` by 1 per matched pair) counted **1**! Furthermore, `sorted_intersect_count` dispatched to `intersect_simd_count_avx512` (when `avx512f` was available) instead of matching `sorted_intersect_adaptive`'s pair-matching semantics when duplicates were present.
2. `sorted_union_count(a, b)` (`simd_set_union.rs:345-365`) computed `a.len() + b.len() - sorted_intersect_count(a, b)`, which could underflow if `sorted_intersect_count` ever exceeded `a.len() + b.len()`, instead of `saturating_sub`.
3. In `simd_gallop_to` (`algorithms/simd_search.rs:915-940`), the galloping phase narrowed the window `(lo, hi)` to length `<= 16`, and then checked `if hi - lo > 16` for the AVX-512 leaf scan — making the AVX-512 leaf scan unreachable on 16-element windows (`hi - lo == 16`).
4. In `multiset_fast_intersection` / `multiset_fast_intersection2` (`set_ops.rs:580-640`), run-length multiplication `count1 * count2` could overflow without `saturating_mul`.
5. 10 `unsafe` functions/blocks across `simd_set_intersect.rs` (`:53, 82, 248, 383`), `simd_set_union.rs` (`:162`), `bit_ops.rs` (`:131, 161`), and `algorithms/simd_search.rs` (`:331, 376, 420`) lacked `// SAFETY:` comments.

**Test (`src/algorithms/simd_set_intersect.rs`).**
- `algorithms::simd_set_intersect::tests::test_c8_12_sorted_intersect_count_duplicates_and_tier_parity` (`simd_set_intersect.rs:738-783`)

**RED on parent (`cf7f4cf`).**
- `test_c8_12_sorted_intersect_count_duplicates_and_tier_parity`: **FAIL** at `src/algorithms/simd_set_intersect.rs:726:9` — `assertion left == right failed: left: 5, right: 2` (`sorted_intersect_count(&a, &b)` returned `5` while `sorted_intersect_adaptive(&a, &b).len()` was `2`).

**Fix (`35e03bf`).**
- Fixed `intersect_simd_count_avx512` (`simd_set_intersect.rs:423`) to count at most 1 match per `a_val` (`if mask != 0 { block_mask_count += 1; }`) matching `intersect_simd_avx512`, and aligned `sorted_intersect_count` (`:500-535`) with `sorted_intersect_adaptive`.
- Used `saturating_sub` in `sorted_union_count` (`simd_set_union.rs:358`), changed the AVX-512 leaf threshold in `simd_gallop_to` to `hi - lo >= 16` (`algorithms/simd_search.rs:928`), used `saturating_mul` in `multiset_fast_intersection{,2}` (`set_ops.rs:595, 635`), and documented all 10 `unsafe` blocks/fns (`97/97 = 100.0%` in `src/algorithms/`).

---

### C8.13 — `LcpArray::new` (`compute_lcp_kasai`) panicked on mismatched `(text, sa)` lengths and `SuffixArray` carried 3 honesty markers in `dc3_construct`, `divsufsort_construct`, and `larsson_sadakane_construct` — MEDIUM

**Finding (`src/algorithms/suffix_array.rs:510-555, 795-835`).**
1. `LcpArray::new(text: &[u8], sa: &SuffixArray)` (`suffix_array.rs:795-835`) did not validate `text.len() == sa.len()`. Passing a `text` longer than `sa` (e.g. `LcpArray::new(b"abcdef", &SuffixArray::new(b"abc")?)`) panicked with `index out of bounds: the len is 3 but the index is 3` when populating `rank[sa[i]]` or indexing `sa[rank[i] - 1]`.
2. `dc3_construct`, `divsufsort_construct`, and `larsson_sadakane_construct` (`suffix_array.rs:510-555`) carried 3 `api_honesty` markers (`Simplified DC3 implementation`, `Simplified DivSufSort`, `Simplified Larsson-Sadakane`) that fell back to `O(n^2 log n)` naive/prefix-doubling sort instead of delegating to the linear-time `sais_construct`.

**Test (`src/algorithms/suffix_array.rs`).**
- `algorithms::suffix_array::tests::test_c8_13_suffix_array_kasai_validation_and_differential_vs_naive` (`suffix_array.rs:1584-1655`)

**RED on parent (`cf7f4cf`).**
- `test_c8_13_suffix_array_kasai_validation_and_differential_vs_naive`: **FAIL** at `src/algorithms/suffix_array.rs:809:16` — `index out of bounds: the len is 3 but the index is 3`.

**Fix (`35e03bf`).**
- Validated `if text.len() != sa.len() { return Err(ZiporaError::invalid_input(...)); }` and bounds-checked `idx >= n` in `compute_lcp_kasai` (`suffix_array.rs:798-820`).
- Delegated `dc3_construct`, `divsufsort_construct`, and `larsson_sadakane_construct` to `Self::sais_construct(text)` (`suffix_array.rs:510-545`), resolving all 3 honesty markers, and added differential testing against naive suffix sort and naive LCP across empty, single-byte, all-equal, binary `{0, 1}` (with embedded `0x00` bytes), ternary `{1, 2, 3}`, and mixed NUL corpora up to `n = 2000` (`32` under Miri).

---

## Audit & Gate Summary (`src/string/` & `src/algorithms/`)

| Metric | Before Stage 9 (`b782a8c`) | After Stage 9 (`35e03bf`) | Status |
|--------|----------------------------|---------------------------|--------|
| `src/string/` `unsafe` sites (documented / total) | 142 / 143 (99.3%, 1 undocumented) | **143 / 143 (100.0%, 0 undocumented)** | PASS |
| `src/algorithms/` `unsafe` sites (documented / total) | 86 / 97 (88.7%, 11 undocumented) | **97 / 97 (100.0%, 0 undocumented)** | PASS |
| Crate-wide `unsafe` sites (documented / total) | 1,306 / 1,325 (98.6%) | **1,318 / 1,325 (99.5%)** | PASS |
| Crate-wide undocumented `unsafe` (`max_undocumented`) | 19 | **7** | RATCHETED (`19 -> 7`) |
| Crate-wide §8.5 precondition violations (`max_precondition_violations`) | 0 | **0** | PASS |
| `src/string/` API honesty markers | 8 (`bmi2_string_ops.rs` 1, `fast_str.rs` 1, `simd_search.rs` 1, `unicode.rs` 5) | **0** | PASS (-8) |
| `src/algorithms/` API honesty markers | 17 (`cache_oblivious.rs` 2, `multiway_merge.rs` 2, `radix_sort/advanced.rs` 9, `set_operations.rs` 1, `suffix_array.rs` 3) | **0** | PASS (-17) |
| Crate-wide API honesty markers (`max_markers`) | 64 | **39** | RATCHETED (`64 -> 39`) |
| `src/string/` production `.expect()` calls | 7 (`bmi2_string_ops.rs` 3, `lexicographic_iterator.rs` 1, `unicode.rs` 3) | **0** | PASS (-7) |
| `src/algorithms/` production `.expect()` calls | 9 (`multiway_merge.rs` 2, `radix_sort/advanced.rs` 1, `set_operations.rs` 2, `simd_merge.rs` 3, `tournament_tree.rs` 1) | **0** | PASS (-9) |
| Crate-wide `.unwrap()` / `.expect()` (`max_unwraps` / `max_expects`) | 0 / 119 | **0 / 103** | RATCHETED (`119 -> 103`) |
| Decoder indexing (`max_total` / `max_in_bounds`) | 0 / 132 | **0 / 132** | PASS |
| `src/string/` + `src/algorithms/` `#[test]` count | 420 (141 + 279) | **437** (150 + 287, +17 new) | PASS |
| Gate test counts (`lib_debug` / `lib_release` / `doctests` / `all_targets`) | 2,994 / 3,011 / 228 / 3,649 | **3,011 / 3,028 / 228 / 3,666** (+6 `compile_fail` doctests) | RATCHETED |
| Miri (`MIRIFLAGS="-Zmiri-tree-borrows -Zmiri-disable-isolation" cargo +nightly miri test --lib test_c8_`) | — | **17 passed, 0 failed** (`124.52s`, 0 UB, 0 leaks) | PASS |

---

## Verification Summary

- **Commits (`3930f77..35e03bf`)**:
  1. `b782a8c` — `fix(containers): restore FixedLenStrVec short-prefix speed, Miri bound, and doc notes (S19-R1..R3)`
  2. `cf7f4cf` — `fix(string): fix SSE4.2 strchr overflow UB, BMI2 UTF-8/wildcard bugs, and string edge cases (C8.1-C8.6)`
  3. `35e03bf` — `fix(algorithms): fix SIMD merge, funnel sort, external sort, radix sort, set ops, and suffix array bugs (C8.7-C8.13)`
- **RED Verification**: All 17 new `test_c8_*` regression tests (9 in `src/string/` verified on `b782a8c`; 8 in `src/algorithms/` verified on `cf7f4cf`) were injected into `.claude/worktrees/head` and confirmed to **FAIL** before their respective fixes.
- **Miri Verification**: All 17 `test_c8_*` tests (`src/string/` + `src/algorithms/`) pass under `cargo +nightly miri test --lib test_c8_` with `-Zmiri-tree-borrows -Zmiri-disable-isolation` and leak checking enabled.
- **Sanity Gate (`make sanity`)**: **PASS** (`exit 0`; 0 clippy warnings across `--all-targets --all-features` and `--no-default-features`; all 4 audit scripts and `check_gate_drift.py` pass at the ratcheted baselines).
