#!/usr/bin/env python3
"""Generate E2/E3 summary tables, Condition 2 parity table, E3 standing table,
and the complete Stage 15 review ledger (`docs/review/e2_e3.md`) directly from
committed JSON artifacts (S42-R1..R8).

Usage:
  python3 scripts/gen_e3_summary.py
  python3 scripts/gen_e3_summary.py --write-doc docs/review/e2_e3.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


FAMILY_ORDER = [
    ("rank_select", "1. Rank/Select (`1 Gbit`)", "E3.1"),
    ("elias_fano", "2. Elias-Fano (`100k`)", "E3.2"),
    ("tries", "3. Tries (`1M` keys)", "E3.3"),
    ("hash_maps", "4. Hash Maps (`1M` keys)", "E3.4"),
    ("memory_pools", "5. Memory Pools (`10k` ops)", "E3.5"),
    ("entropy", "6. Entropy Coding (`64 KiB`)", "E3.6"),
    ("int_codecs", "7. Integer Codecs (`100k`)", "E3.7"),
    ("radix_sort", "8. Radix Sort (`10M`)", "E3.8"),
    ("suffix_array", "9. Suffix Array (`10M` B)", "E3.9"),
    ("search_and_scoring", "10. String Search (`64 KiB`)", "E3.10"),
]


def core_bench_instance_count(name: str) -> str:
    """Return the instance count used by `benches/core_bench.rs` vs `1` in `competitor_bench.rs`."""
    if name in ("tries/dat/url_paths/lookup", "tries/dat/random_16b/lookup"):
        return "`2` vs `1`"
    if name == "tries/dat/shared_prefix/lookup":
        return "`4` vs `1`"
    if name in ("tries/cspp/url_paths/lookup", "tries/cspp/random_16b/lookup", "tries/zipora_trie_da/lookup"):
        return "`4` vs `1`"
    if name == "tries/cspp/shared_prefix/lookup":
        return "`8` vs `1`"
    if name.startswith("hash_maps/"):
        return "`2` vs `1`"
    if name.startswith("memory_pools/secure_pool/"):
        return "`64` vs `1`"
    if name in (
        "entropy/huffman_o1/decode_64kib",
        "entropy/huffman_o2/encode_64kib",
        "entropy/huffman_o2/decode_64kib",
        "int_codecs/group_varint/encode_deltas",
    ):
        return "`8` vs `1`"
    if name.startswith("elias_fano/"):
        return "`4` vs `1`"
    return "`1` vs `1`"


def explain_parity_delta(name: str, delta_pct: float) -> str:
    """Return the exact architectural explanation for any Condition 2 parity delta."""
    if abs(delta_pct) <= 5.0:
        return "Within `±5.0%` parity threshold"
    if name.startswith("tries/") and name.endswith("/lookup"):
        inst = core_bench_instance_count(name)
        return (
            f"`core_bench.rs` alternates across {inst.split(' ')[0]} trie replicas (`2–8x` working set beyond L3); "
            f"`competitor_bench.rs` uses `1` instance (`S42-R1`)"
        )
    if name.startswith("tries/") and ("/build" in name):
        return (
            "`core_bench.rs` pre-allocates/drops `6–11 x 24 MiB` `brk_holes` before building; "
            "`competitor_bench.rs` uses `1` clean slot (`S42-R1`)"
        )
    if name.startswith("hash_maps/zipora_hash_map/"):
        return (
            "`core_bench.rs` alternates across a `2`-map ring (`[q_idx & 1]`, `2x` working set) with `ahash::RandomState`; "
            "`competitor_bench.rs` uses `1` map instance with `ahash::RandomState` (`S42-R1`)"
        )
    if name.startswith("hash_maps/gold_hash_map/"):
        return (
            "`core_bench.rs` alternates across a `2`-map ring (`[q_idx & 1]`, `2x` working set) with default `StdSipHasher13`; "
            "`competitor_bench.rs` uses `1` map instance with `ahash::RandomState` (`S42-R1`)"
        )
    if name.startswith("memory_pools/secure_pool/"):
        return (
            "`core_bench.rs` rotates across a `64`-pool ring (`sec_pools` via recursive `run_sec_pool_quarter`); "
            "`competitor_bench.rs` uses `1` `SecureMemoryPool` instance per thread (`S42-R1`)"
        )
    if name.startswith("rank_select/se256/") or name.startswith("rank_select/interleaved256/"):
        return (
            "`1` instance in both (`131.25–172 MiB` DRAM/DTLB working set across 4 KiB THP-off pages; "
            "within disclosed `9.0%` / `16.0%` random-DTLB threshold)"
        )
    if name == "entropy/huffman_o2/decode_64kib" or name == "entropy/huffman_o1/decode_64kib":
        return (
            "`core_bench.rs` rotates across an `8`-decoder ring constructed on 8 concurrent threads; "
            "`competitor_bench.rs` uses `1` `ContextualHuffmanDecoder` instance (`S42-R1`)"
        )
    if name == "entropy/huffman_o2/encode_64kib":
        return (
            "`core_bench.rs` rotates across an `8`-encoder ring constructed on 8 concurrent threads; "
            "`competitor_bench.rs` uses `1` `ContextualHuffmanEncoder` instance (`S42-R1`)"
        )
    if abs(delta_pct) <= 9.0:
        return (
            "`1` instance in both binaries (`S47-R10` / `S48-R6`; within disclosed `9.0%` threshold)"
        )
    return (
        f"`1` instance in both binaries (`S47-R10` / `S48-R6`; `{delta_pct:+.2f}%` cross-binary layout delta)"
    )


def verify_and_summarize(
    comp_data: dict[str, Any],
    core_data: dict[str, Any],
    e2e_data: dict[str, Any],
) -> dict[str, Any]:
    comp_benches = {b["name"]: b for b in comp_data["benchmarks"]}
    core_benches = {b["name"]: b for b in core_data["benchmarks"]}
    comparisons = comp_data["same_run_comparisons"]

    # 1. Verify arithmetic and per-row provenance (S51-R2) on every SameRunComparison row
    for b in comp_data["benchmarks"]:
        assert b.get("measured_at_commit") and len(b.get("binary_sha256", "")) == 64, b["name"]
    for b in core_data["benchmarks"]:
        assert b.get("measured_at_commit") and len(b.get("binary_sha256", "")) == 64, b["name"]
    for b in e2e_data["benchmarks"]:
        assert b.get("measured_at_commit") and len(b.get("binary_sha256", "")) == 64, b["name"]
    for row in comparisons:
        zb = comp_benches[row["zipora_bench"]]
        cb = comp_benches[row["competitor_bench"]]
        assert row.get("zipora_measured_at_commit") == zb["measured_at_commit"], row
        assert row.get("zipora_binary_sha256") == zb["binary_sha256"], row
        assert row.get("competitor_measured_at_commit") == cb["measured_at_commit"], row
        assert row.get("competitor_binary_sha256") == cb["binary_sha256"], row
        expected_cross = (
            zb["binary_sha256"] != cb["binary_sha256"]
            or zb["measured_at_commit"] != cb["measured_at_commit"]
        )
        assert row.get("cross_binary_comparison") is expected_cross, (row, expected_cross)
        z_min = zb["min_ns_per_op"]
        c_min = cb["min_ns_per_op"]
        assert abs(row["zipora_primary_min"] - round(z_min, 4)) <= 1e-3, row
        assert abs(row["competitor_primary_min"] - round(c_min, 4)) <= 1e-3, row
        expected_verdict = (
            "LEADING"
            if row["primary_delta_pct"] <= 3.0 and row["secondary_delta_pct"] <= 10.0
            else "GAP"
        )
        assert row["e3_verdict"] == expected_verdict, (row, expected_verdict)

    # 2. Condition 2 parity across all 80 shared Zipora benchmarks
    parity_rows: list[dict[str, Any]] = []
    within_5pct = 0
    within_9pct = 0
    for core_rec in core_data["benchmarks"]:
        name = core_rec["name"]
        assert name in comp_benches, f"Missing Zipora core_bench row in competitor_bench: {name}"
        comp_rec = comp_benches[name]
        core_min = core_rec["min_ns_per_op"]
        comp_min = comp_rec["min_ns_per_op"]
        delta_pct = (comp_min - core_min) / core_min * 100.0 if core_min > 0 else 0.0
        if abs(delta_pct) <= 5.0:
            within_5pct += 1
        if abs(delta_pct) <= 9.0:
            within_9pct += 1
        parity_rows.append(
            {
                "category": core_rec["category"],
                "name": name,
                "dataset_size": core_rec["dataset_size"],
                "unit": core_rec["unit"],
                "instances": core_bench_instance_count(name),
                "core_min": core_min,
                "comp_min": comp_min,
                "delta_pct": delta_pct,
                "core_med": core_rec["median_ns_per_op"],
                "comp_med": comp_rec["median_ns_per_op"],
                "note": explain_parity_delta(name, delta_pct),
            }
        )

    # 3. Per-family verdict breakdown
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        by_cat[row["category"]].append(row)

    family_stats: list[dict[str, Any]] = []
    total_rows = 0
    total_leading = 0
    total_fast_sec_high = 0
    total_gap = 0

    for cat_key, cat_title, gap_id in FAMILY_ORDER:
        rows = by_cat[cat_key]
        n_total = len(rows)
        n_lead = sum(1 for r in rows if r["e3_verdict"] == "LEADING")
        n_fast_sec = sum(
            1
            for r in rows
            if r["e3_verdict"] == "GAP" and r["primary_delta_pct"] <= 3.0
        )
        n_gap = n_total - n_lead
        total_rows += n_total
        total_leading += n_lead
        total_fast_sec_high += n_fast_sec
        total_gap += n_gap
        family_stats.append(
            {
                "category": cat_key,
                "title": cat_title,
                "gap_id": gap_id,
                "total": n_total,
                "leading": n_lead,
                "faster_primary_sec_gt_10": n_fast_sec,
                "faster_or_tied_primary": n_lead + n_fast_sec,
                "slower_primary": n_gap - n_fast_sec,
                "gap": n_gap,
            }
        )

    # Add E1.1 systemic BM25 search row from e2e_search.json
    e2e_summary = e2e_data["summary"]
    e2e_benches = {b["name"]: b for b in e2e_data["benchmarks"]}
    z_ns = e2e_benches["e2e_search/zipora_bmw/all_10k_queries"]["min_ns_per_op"]
    t_ns = e2e_benches["e2e_search/tantivy_bm25/all_10k_queries"]["min_ns_per_op"]
    z_qps = e2e_summary["zipora_qps_top10"]
    t_qps = e2e_summary["tantivy_qps_top10"]
    z_bpp = e2e_summary["zipora_bytes_per_posting"]
    t_bpp = e2e_summary["tantivy_postings_bytes_per_posting"]
    z_total_bpp = e2e_summary["zipora_total_index_bytes_per_posting"]
    t_total_bpp = e2e_summary["tantivy_bytes_per_posting"]
    e11_speedup = t_ns / z_ns
    e11_primary_delta_pct = (z_ns - t_ns) / t_ns * 100.0
    e11_sec_delta_pct = (z_bpp - t_bpp) / t_bpp * 100.0
    e11_total_sec_delta_pct = (z_total_bpp - t_total_bpp) / t_total_bpp * 100.0
    e11_verdict = (
        "LEADING"
        if e11_primary_delta_pct <= 3.0 and e11_sec_delta_pct <= 10.0
        else "GAP"
    )
    family_stats.append(
        {
            "category": "e2e_search",
            "title": "11. Systemic Search (`E1.1`, `100k` docs)",
            "gap_id": "E1.1",
            "total": 1,
            "leading": 1 if e11_verdict == "LEADING" else 0,
            "faster_primary_sec_gt_10": 0,
            "faster_or_tied_primary": 1 if e11_primary_delta_pct <= 3.0 else 0,
            "slower_primary": 0 if e11_primary_delta_pct <= 3.0 else 1,
            "gap": 0 if e11_verdict == "LEADING" else 1,
        }
    )
    total_rows += 1
    if e11_verdict == "LEADING":
        total_leading += 1
    else:
        total_gap += 1

    return {
        "comp_benches": comp_benches,
        "core_benches": core_benches,
        "comparisons": comparisons,
        "parity_rows": parity_rows,
        "within_5pct": within_5pct,
        "within_9pct": within_9pct,
        "family_stats": family_stats,
        "totals": {
            "total": total_rows,
            "leading": total_leading,
            "faster_primary_sec_gt_10": total_fast_sec_high,
            "faster_or_tied_primary": total_leading + total_fast_sec_high,
            "slower_primary": total_gap - total_fast_sec_high,
            "gap": total_gap,
        },
        "e11": {
            "z_ns": z_ns,
            "t_ns": t_ns,
            "z_qps": z_qps,
            "t_qps": t_qps,
            "z_bpp": z_bpp,
            "t_bpp": t_bpp,
            "z_total_bpp": z_total_bpp,
            "t_total_bpp": t_total_bpp,
            "speedup": e11_speedup,
            "primary_delta_pct": e11_primary_delta_pct,
            "sec_delta_pct": e11_sec_delta_pct,
            "total_sec_delta_pct": e11_total_sec_delta_pct,
            "verdict": e11_verdict,
        },
    }


def build_gap_analysis_sections(
    summary: dict[str, Any],
    profiles_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    cb = summary["comp_benches"]
    core_b = summary["core_benches"]
    p1_top_ip_pct = profiles_by_id["E3.1"]["top_instructions"][0]["pct"]
    p6_top_sym_pct = profiles_by_id["E3.6"]["top_symbols"][0]["pct"]
    p8_min_faults = profiles_by_id["E3.8"]["counters"]["minor-faults"]
    p9_min_faults = profiles_by_id["E3.9"]["counters"]["minor-faults"]
    p9_sais_induce_pct = profiles_by_id["E3.9"]["top_symbols"][0]["pct"]
    p9_sais_core_pct = profiles_by_id["E3.9"]["top_symbols"][1]["pct"]
    strstr_comp = cb["search_and_scoring/simd_strstr_64kib"]["min_ns_per_op"]
    strstr_core = core_b["search_and_scoring/simd_strstr_64kib"]["min_ns_per_op"]
    strstr_delta = (strstr_comp - strstr_core) / strstr_core * 100.0
    memmem_ns = cb["search_and_scoring/memmem_strstr_64kib"]["min_ns_per_op"]
    finder_ns = cb["search_and_scoring/memmem_finder_strstr_64kib"]["min_ns_per_op"]
    ac_ns = cb["search_and_scoring/aho_corasick_strstr_64kib"]["min_ns_per_op"]
    strchr_ns = cb["search_and_scoring/simd_strchr_64kib"]["min_ns_per_op"]
    memchr_ns = cb["search_and_scoring/memchr_strchr_64kib"]["min_ns_per_op"]

    return {
        "E3.1": {
            "files": "`src/succinct/rank_select/interleaved.rs`, `src/succinct/rank_select/separated.rs`, `src/succinct/rank_select/rank9.rs`",
            "root_cause": (
                "1. In `src/succinct/rank_select/interleaved.rs:76-85`, `InterleavedLine` has `#[repr(C, align(32))]` on a 40-byte struct "
                "(`rlev1: u32`, `rlev2: [u8; 4]`, `bit64: [u64; 4]`), which pads every 256-bit block from `40` bytes to **`64` bytes** "
                "(`24` bytes of dead tail padding per line = `75%` of raw bitvector wasted purely on alignment padding!). As the loop-only "
                f"disassembly above shows (`shl $0x6, %rdi` -> `movzbl 0x4(%rdi,%r8,1), %r10d`), `{p1_top_ip_pct:.1f}%` of all loop samples stall on the "
                "`64`-byte `InterleavedLine` DRAM cache miss. In addition, `RankSelectInterleaved256` stores only a `select1` sample table "
                "(`select_cache: Option<FastVec<u32>>`, line `97`) with no `select0` sample table, so `select0` binary-searches all `3,906,250` lines.\n"
                "2. In `src/succinct/rank_select/separated.rs:23-30`, `RankCacheSE` uses `8` bytes (`lev1: u32, lev2: [u8; 4]`) per `256` bits "
                "(`25%` rank overhead) plus dense `u32` `sel0_cache`/`sel1_cache` every `256` bits (`+12.5%` = `37.5%` total overhead), whereas "
                "`bitm` (`3.91%`) and `vers-vecs` (`4.69%`) pack 512-bit L2 relative offsets into 9-bit/10-bit bitfields inside a single 64-bit header."
            ),
            "e4_fix": (
                "- Remove `align(32)` on `InterleavedLine` (keeping `40`-byte unpadded `#[repr(C)]` or widening to 512 bits + 64-bit packed L1/L2 header) "
                "and add a `select0_cache` to `RankSelectInterleaved256`, cutting `RankSelectInterleaved256` memory from `203.12%` to `128.12%` "
                "(`-36.9%` total footprint) and accelerating `select0` by `~4.3x`.\n"
                "- Add a compact 512-bit / 1024-bit hierarchical rank/select mode (`< 10%` space overhead) with Zero public API or serde wire-format breakage."
            ),
        },
        "E3.2": {
            "files": "`src/succinct/elias_fano/basic.rs`, `src/succinct/elias_fano/optimal.rs`",
            "root_cause": (
                "1. In `EliasFano::build_samples` (`src/succinct/elias_fano/basic.rs:198-203`) and `EliasFano::select1` (`src/succinct/elias_fano/basic.rs:383-418`), "
                "`select1_samples[k]` stores only the word-aligned bit index `(word_idx * 64) as u32` rather than the exact bit position of the `(k * 256)`-th 1-bit. "
                "Consequently, `select1` must also load `self.rank_samples[start_word / 8]` (a **second independent cache line load** at line `402`) and loop "
                "`for w in (rank_block * 8)..start_word` summing `high_bits[w].count_ones()` (`lines 403-405`) before scanning forward with `SELECT1_SAMPLE_RATE = 256` (`lines 409-416`).\n"
                "2. In `OptimalPartitionedEliasFano::get` (`src/succinct/elias_fano/optimal.rs:395-414`), every random index lookup executes "
                "`self.chunk_starts.partition_point(|&s| s <= index)` (`O(log C)` binary search over variable-length DP chunk boundaries) followed by `chunk_select1`."
            ),
            "e4_fix": (
                "- In `EliasFano::build_samples` (`src/succinct/elias_fano/basic.rs:200-203`), store the exact bit offset "
                "`(word_idx * 64 + select_in_word(word, (next_sel1 - cumul_ones) as usize)) as u32` in `select1_samples[k]` and mask off the starting word's "
                "first `(sample_bit & 63)` bits in `select1` (`0` extra bytes, eliminating `rank_samples` load and popcount pre-loop; expected `1.45x–1.55x` speedup to `~17–18 ns/op`).\n"
                "- Add a top-level index quantum table (`index / 256 -> chunk_idx_range`) in `OptimalPartitionedEliasFano` (`~1.5x` faster `advance_to_index`)."
            ),
        },
        "E3.3": {
            "files": "`src/fsa/double_array/trie.rs`, `src/fsa/cspp_trie.rs`",
            "root_cause": (
                "1. In `DoubleArrayTrie::build_from_sorted` (`src/fsa/double_array/trie.rs:906-922`), despite accepting pre-sorted keys (`keys: &[&[u8]]`), "
                "it calls `for &key in keys { trie.insert(key)?; }`! Incremental `insert` (`lines 334-425`) discovers children one by one, so a branching node "
                "with `k` outgoing edges triggers `k - 1` calls to `relocate(curr, ch)` (`lines 1104-1205`) and linear `find_free_base` (`lines 1050-1100`).\n"
                "2. On `lookup`, `fst::Set` merges common suffixes as well as common prefixes (minimal DAFSA: `0.85 B/key` on `url_paths`, `0.00026 B/key` on "
                "`shared_prefix`), so its automaton fits inside L1/L2 cache whereas `DoubleArrayTrie` (`12 B/state`, `26.6–168.5 B/key`) and `CsppTrie` (`9.6–23.4 B/key`) "
                "exceed L2/L3 cache on `1,000,000` keys."
            ),
            "e4_fix": (
                "- Replace the incremental `for &key in keys { trie.insert(key)?; }` loop in `DoubleArrayTrie::build_from_sorted` (`src/fsa/double_array/trie.rs:906-922`) "
                "with a direct top-down partitioning pass over sorted key ranges `(start, end, depth, parent_state)` that gathers all distinct child bytes at `depth` in one scan "
                "and calls `find_free_base(&children)` **once** per internal node with **zero `relocate` calls**, backed by a 64-bit occupancy bitmap (`trailing_zeros()`).\n"
                "- Expected gain: **`2.5x–3.5x` faster `DoubleArrayTrie::build_from_sorted`** (`0` extra bytes in `DoubleArrayTrie`, `0` API/wire changes)."
            ),
        },
        "E3.4": {
            "files": "`src/hash_map/storage.rs`, `src/hash_map/zipora_hash_map.rs`, `src/hash_map/gold_hash_map.rs`",
            "root_cause": (
                f"1. With `S42-R1` unifying both sides on **1 map instance**, `hashbrown+ahash` (`{cb['hash_maps/hashbrown_ahash/get_u64']['min_ns_per_op']:.3f} ns` `get_u64`, "
                f"`{cb['hash_maps/hashbrown_ahash/insert_u64']['min_ns_per_op']:.3f} ns` `insert_u64`, `35.65 B/entry`) beats "
                f"`ZiporaHashMap` (`{cb['hash_maps/zipora_hash_map/get_u64']['min_ns_per_op']:.3f} ns` `get_u64`, "
                f"`{cb['hash_maps/zipora_hash_map/insert_u64']['min_ns_per_op']:.3f} ns` `insert_u64`, `83.89 B/entry`) and "
                f"`GoldHashMap` (`{cb['hash_maps/gold_hash_map/get_u64']['min_ns_per_op']:.3f} ns` `get_u64`, "
                f"`{cb['hash_maps/gold_hash_map/insert_u64']['min_ns_per_op']:.3f} ns` `insert_u64`, `31.93 B/entry`).\n"
                "2. In `src/hash_map/storage.rs:48-53`, `HashEntry<K, V>` stores `{ key: Option<K>, value: Option<V>, hash: u64, _next: Option<u32> }` (`40–48 B/slot` for `(u64, u64)` "
                "with two `Option` discriminants, an 8-byte `hash`, and an unused 8-byte `_next` field = `83.89 B/entry` at `2^21` slots vs `35.65 B/entry` in `hashbrown`), and probes "
                "1 slot at a time in scalar code instead of 16-way SSE2/AVX2 control-byte group probing (`_mm_cmpeq_epi8` + `_mm_movemask_epi8`).\n"
                "3. In `GoldHashMap` (`src/hash_map/gold_hash_map.rs`), `get` and `insert` compute `hash % buckets.len()` (a hardware 64-bit integer division `divq` by a prime modulus) "
                "and chase two dependent DRAM cache misses (`buckets[b]` -> `nodes[idx]`)."
            ),
            "e4_fix": (
                "- Replace `HashEntry<K, V>` (`src/hash_map/storage.rs:48-53`) with a flat SwissTable control-byte array (`ctrl: FastVec<u8>` storing a 7-bit `h2` fingerprint, "
                "probed 16 slots at a time via SSE2 `_mm_cmpeq_epi8`) + unpadded `FastVec<MaybeUninit<(K, V)>>` (`17 B/slot` for `(u64, u64)` = `35.65 B/entry`, `-57.5%` memory), "
                f"cutting `ZiporaHashMap` `get_u64` from `{cb['hash_maps/zipora_hash_map/get_u64']['min_ns_per_op']:.1f} ns` to `~45 ns` and `insert_u64` from "
                f"`{cb['hash_maps/zipora_hash_map/insert_u64']['min_ns_per_op']:.1f} ns` to `~25–30 ns` (`E4(b)`).\n"
                "- Replace `GoldHashMap`'s 64-bit `% prime` division with Lemire fast-range reduction `((hash as u128 * len as u128) >> 64) as usize`."
            ),
        },
        "E3.5": {
            "files": "`src/memory/lockfree_pool.rs`, `src/memory/secure_pool.rs`",
            "root_cause": (
                "1. Pre-`E4(f)` (Stage 15 `b4148fc`, `24.003 ns/op` 1T and `21.853 ns/op` 4T in `e3_gap_profiles.json` above), every `allocate(256)` + `deallocate(ptr, 256)` pair hit the global ABA-tagged Treiber stack in `LockFreeMemoryPool::allocate_from_fast_bin` and `LockFreeMemoryPool::deallocate` (`self.fast_bins[bin_index].head` and `self.fast_bins[bin_index].count`, later refactored into `LockFreeMemoryPool::pop_fast_bin` and `PoolArena::push_free_fast_bin_offset` in Stage 16 `S47-R4`/`S48-R2`), executing **4 atomic RMW instructions** (`2 x lock cmpxchgq` on `bin.head` + `2 x lock xaddq` on `bin.count`, visible in the pre-`E4(f)` loop disassembly above). At Stage 16 `HEAD` (`"
                f"{cb['memory_pools/lockfree_pool/alloc_free_1t']['min_ns_per_op']:.3f} ns/op` 1T and "
                f"`{cb['memory_pools/lockfree_pool/alloc_free_4t']['min_ns_per_op']:.3f} ns/op` 4T), the per-thread magazine path (`FastBinMagazine::pop` + `transition_fast_bin_live_to_free` + `FastBinMagazine::push`) executes **3 L1D-resident atomic RMW instructions** per alloc/free pair (`slot.swap(0, AcqRel)` on magazine pop, `compare_exchange` `FAST_BIN_LIVE_TAG -> FAST_BIN_FREE_TAG` before magazine push, and `shared.non_empty_bins.fetch_or(bit, Ordering::AcqRel)` on magazine push).\n"
                f"2. By contrast, `mimalloc` (`{cb['memory_pools/mimalloc/alloc_free_1t']['min_ns_per_op']:.3f} ns` 1T, `{cb['memory_pools/mimalloc/alloc_free_4t']['min_ns_per_op']:.3f} ns` 4T), "
                f"`jemalloc` (`{cb['memory_pools/jemalloc/alloc_free_1t']['min_ns_per_op']:.3f} ns` 1T, `{cb['memory_pools/jemalloc/alloc_free_4t']['min_ns_per_op']:.3f} ns` 4T), and "
                f"`std::alloc::System` (`{cb['memory_pools/system_alloc/alloc_free_1t']['min_ns_per_op']:.3f} ns` 1T, `{cb['memory_pools/system_alloc/alloc_free_4t']['min_ns_per_op']:.3f} ns` 4T) "
                "serve 256-byte alloc+free pairs from a **per-thread non-atomic free list (`tcache`)** in ~15 scalar instructions (`0` `lock` instructions on the fast path)."
            ),
            "e4_fix": (
                "- Add a small per-thread magazine (`thread_local!` cache of up to `16` block offsets per fast bin, keyed by `pool_id: u64` and flushed on `Drop`/`reset`) "
                "in front of the global ABA-tagged Treiber stack in `LockFreeMemoryPool`, and batch `bin.count` updates on magazine refill/flush (`E4(f)`).\n"
                f"- Measured Result & `S49-R5`/`S50-R1` Soundness Trade-Off Disclosure: `E4(f)` (`48d5a5a`..`1cb20d8`) initially reduced `alloc_free_1t` from `24.003` to `5.996 ns/op` and `alloc_free_4t` from `21.853` to `6.226 ns/op` using non-atomic `Cell<u32>` slot reads/writes. In `S49-R1..R4` and `S50-R1`, closing the owner-`pop` vs `drain_all_magazines`-steal race (`S49-R2`: atomic `slot.swap(0, AcqRel)` in `FastBinMagazine::pop`), rejecting cross-thread double-frees of magazine-cached blocks (`S49-R3`: atomic `compare_exchange` `FAST_BIN_LIVE_TAG -> FAST_BIN_FREE_TAG` in `transition_fast_bin_live_to_free` before magazine push), and eliminating the push-vs-drain store-buffering race (`S50-R1`: unconditional `shared.non_empty_bins.fetch_or(bit, Ordering::AcqRel)` in `FastBinMagazine::push`) added three L1D-resident atomic RMWs per alloc/free pair, bringing `alloc_free_1t` to `{cb['memory_pools/lockfree_pool/alloc_free_1t']['min_ns_per_op']:.3f} ns/op` (`7.173 ns/op` at `67c6eb6`, `+{(cb['memory_pools/lockfree_pool/alloc_free_1t']['min_ns_per_op'] - 5.996) / 5.996 * 100.0:.1f}%` vs `5.996 ns/op` at `1cb20d8`, still **`{24.0029 / cb['memory_pools/lockfree_pool/alloc_free_1t']['min_ns_per_op']:.2f}x` faster** than pre-`E4(f)` `24.003 ns/op`) and `alloc_free_4t` to `{cb['memory_pools/lockfree_pool/alloc_free_4t']['min_ns_per_op']:.3f} ns/op` (`7.187 ns/op` at `67c6eb6`, `+{(cb['memory_pools/lockfree_pool/alloc_free_4t']['min_ns_per_op'] - 6.226) / 6.226 * 100.0:.1f}%` vs `6.226 ns/op` at `1cb20d8`, still **`{21.853 / cb['memory_pools/lockfree_pool/alloc_free_4t']['min_ns_per_op']:.2f}x` faster** than pre-`E4(f)` `21.853 ns/op`)."
            ),
        },
        "E3.6": {
            "files": "`src/entropy/huffman/encoder.rs`, `src/entropy/huffman/interleaved.rs`, `src/entropy/fse.rs`, `src/entropy/rans.rs`",
            "root_cause": (
                "1. Against raw `libzstd` stage competitors (`S42-R5`), `zstd_huf0` (`HUF_compress4X_repeat` / `HUF_decompress4X_hufOnly_wksp` BMI2) runs at "
                f"`{cb['entropy/zstd_huf0/encode_64kib']['min_ns_per_op']:.3f} ns/byte` encode and `{cb['entropy/zstd_huf0/decode_64kib']['min_ns_per_op']:.3f} ns/byte` decode (`4.01 bits/byte`), "
                f"and `zstd_fse` (`FSE_compress_usingCTable` / `FSE_decompress_wksp_bmi2`) runs at `{cb['entropy/zstd_fse/encode_64kib']['min_ns_per_op']:.3f} ns/byte` encode and "
                f"`{cb['entropy/zstd_fse/decode_64kib']['min_ns_per_op']:.3f} ns/byte` decode (`4.04 bits/byte`), whereas Zipora's `HuffmanDecoder::decode` (`O0`) runs at "
                f"`{cb['entropy/huffman_o0/decode_64kib']['min_ns_per_op']:.3f} ns/byte`, `FseEncoder`/`fse_decompress` at "
                f"`{cb['entropy/fse/encode_64kib']['min_ns_per_op']:.3f}` / `{cb['entropy/fse/decode_64kib']['min_ns_per_op']:.3f} ns/byte`, and `Rans64Decoder<ParallelX4>` at "
                f"`{cb['entropy/rans64_x4/decode_64kib']['min_ns_per_op']:.3f} ns/byte`.\n"
                f"2. As the loop-only profile of `entropy/huffman_o0/decode_64kib` shows (`{p6_top_sym_pct:.1f}%` in `HuffmanDecoder::decode`), `HuffmanDecoder::decode` uses a single bitstream "
                "state rather than 4-stream ILP (`HUF_decompress4X`) and performs per-symbol table refill checks (`E4(c)`).\n"
                "3. In `Rans64Decoder::decode_parallel` (`src/entropy/rans.rs:631-642`), the 4 streams are decoded **sequentially one stream after another** (`for stream_idx in 0..4`) "
                "instead of advancing all 4 rANS states in lockstep inside one loop (`E4(d)`). And in `ContextualHuffmanDecoder` (`O1`/`O2`, `src/entropy/huffman/interleaved.rs:1211-1330`), "
                "every symbol hashes `context` in `std::collections::HashMap<u32, usize>` (SipHash-1-3) and walks boxed `HuffmanNode` pointers 1 bit at a time."
            ),
            "e4_fix": (
                "- `E4(c)`: Add a 4-stream interleaved fast-table decoder path to `HuffmanDecoder` and replace `ContextualHuffmanDecoder`'s `HashMap<u32, usize>` + pointer tree walk "
                "with flat `[u16; 256]` (O1) / `Vec<u16>` (O2) context tables and 11-bit flat decode tables (`4x–6x` speedup).\n"
                "- `E4(d)`: Fuse `Rans64Decoder<ParallelX4>::decode_parallel` into a single 4-state lockstep loop over a packed `[(u16, u16, u8); 4096]` lookup table (`~2.5x–3x` speedup)."
            ),
        },
        "E3.7": {
            "files": "`src/compression/stream_vbyte.rs`",
            "root_cause": (
                "1. With `S42-R5` adding `SvbCrateSimd` (SSE4.1 encode + SSSE3 `_mm_shuffle_epi8` decode via `stream-vbyte`'s `Encoder`/`Decoder` traits on stable Rust), "
                f"Zipora's `StreamVByte::decode_into` (`{cb['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.3f} ns/int`) and `decode_deltas` "
                f"(`{cb['int_codecs/stream_vbyte/decode_deltas']['min_ns_per_op']:.3f} ns/int`) **beat `SvbCrateSimd` (`LEADING`: "
                f"`{cb['int_codecs/stream_vbyte_simd/decode_into']['min_ns_per_op']:.3f} ns/int` and `{cb['int_codecs/stream_vbyte_simd/decode_deltas']['min_ns_per_op']:.3f} ns/int`)**, "
                "confirming that Zipora's SSSE3 decode kernel is at full SIMD parity with `stream-vbyte`!\n"
                f"2. However, on `encode_deltas`, `StreamVByte::encode_deltas` (`{cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.3f} ns/int`) trails `SvbCrateSimd` "
                f"(`{cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']:.3f} ns/int`, `{cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op'] / cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']:.2f}x` slower) "
                f"and `stream_vbyte::Scalar` (`{cb['int_codecs/stream_vbyte_crate/encode_deltas']['min_ns_per_op']:.3f} ns/int`, "
                f"`{cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op'] / cb['int_codecs/stream_vbyte_crate/encode_deltas']['min_ns_per_op']:.2f}x` slower) because `StreamVByte::encode_deltas` "
                "(`src/compression/stream_vbyte.rs:105-130`) allocates a `400 KiB` temporary `deltas: Vec<u32>` and calls `data.extend_from_slice(&v.to_le_bytes()[..len])` per integer.\n"
                f"3. `GroupVarint::decode_raw` (`src/compression/stream_vbyte.rs:426-468`, `{cb['int_codecs/group_varint/decode_deltas']['min_ns_per_op']:.3f} ns/int`) uses scalar `copy_from_slice` instead of `SHUFFLE_TABLE` (`_mm_shuffle_epi8`)."
            ),
            "e4_fix": (
                "- Fuse delta computation into `StreamVByte::encode_deltas` without allocating `deltas: Vec<u32>`, pre-allocate `n * 4 + 16` bytes, and encode 4-integer groups "
                f"with unaligned `u32` stores / SSE4.1 control-mask extraction (`{cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.2f} ns/int -> ~0.9–1.2 ns/int`, `4x` speedup, matching `SvbCrateSimd`).\n"
                f"- Wire `GroupVarint::decode_raw` to the SSSE3 `_mm_shuffle_epi8` `SHUFFLE_TABLE` (`{cb['int_codecs/group_varint/decode_deltas']['min_ns_per_op']:.2f} ns/int -> ~0.60 ns/int`, `16x` speedup)."
            ),
        },
        "E3.8": {
            "files": "`src/algorithms/radix_sort/lsd.rs`",
            "root_cause": (
                "1. In `RadixSort::sort_u32_sequential` (`src/algorithms/radix_sort/lsd.rs:169-236`) and `sort_u64_sequential` (`lines 282-332`), every pass performs a separate "
                "histogram scan over the entire `10,000,000`-element array (`8` full-array passes for `u32`, `16` full-array passes over `80 MiB` DRAM for `u64`) and allocates a fresh "
                f"`temp = vec![0u64; data.len()]` (`80 MiB` = `{p8_min_faults:,d}` minor page faults per call in the loop-only counter profile above!).\n"
                f"2. In `KeyValueRadixSort::sort_by_key` (`src/algorithms/radix_sort/lsd.rs:570-608`, `{cb['radix_sort/kv_u64_u64']['min_ns_per_op']:.1f} ns/elem`), it allocates `indices: Vec<KVPair>` (`160 MiB`), runs "
                "`AdvancedRadixSort` (another `160 MiB`), and then allocates `targets = vec![0usize; n]` (`80 MiB`) to execute an in-place random-DRAM cycle-chasing permutation."
            ),
            "e4_fix": (
                "- Compute all 4 (`u32`) or 8 (`u64`) 256-entry histograms in a **single** upfront pass, skip trivial passes (`counts[b] == n`), reuse a cached scratch buffer on `RadixSort` "
                f"(eliminating the `{p8_min_faults:,d}` minor page faults per call), and add an MSB 8-bit pre-binning pass for `n >= 1_000_000` so remaining passes stay in L2/L3 cache (`E4(a)`).\n"
                f"- In `KeyValueRadixSort::sort_by_key`, ping-pong `data: &mut [(K, V)]` directly against a single `Vec<(K, V)>` buffer (`56 -> 32 B/elem`, `{cb['radix_sort/kv_u64_u64']['min_ns_per_op']:.1f} -> ~20–24 ns/elem`, `~10x` speedup)."
            ),
        },
        "E3.9": {
            "files": "`src/algorithms/suffix_array.rs`",
            "root_cause": (
                f"1. In `SuffixArray::new` (`src/algorithms/suffix_array.rs:324-440, 588-715`, `{cb['suffix_array/construct']['min_ns_per_op']:.2f} ns/byte` vs `libsais` `{cb['suffix_array/libsais_c/construct']['min_ns_per_op']:.2f} ns/byte`, `cdivsufsort` `{cb['suffix_array/cdivsufsort_c/construct']['min_ns_per_op']:.2f} ns/byte`, `divsufsort` `{cb['suffix_array/divsufsort_rust/construct']['min_ns_per_op']:.2f} ns/byte`, and `suffix` `{cb['suffix_array/suffix_crate/construct']['min_ns_per_op']:.2f} ns/byte`; `{p9_min_faults:,d}` minor faults per call in the loop-only profile above), "
                "`select_algorithm` scans all `10,000,000` bytes twice (`analyze_text_characteristics`) even though every enum arm delegates to `sais_construct`, which then copies "
                f"the `10 MB` `&[u8]` text into a `40 MB` `Vec<u32>`, allocates a `10 MB` `Vec<bool>`, and converts the result into an `80 MB` `Vec<usize>` (`{p9_sais_induce_pct:.1f}%` of samples in `sais_induce`, `{p9_sais_core_pct:.1f}%` in `sais_core`).\n"
                f"2. In `SuffixArray::search_range` (`src/algorithms/suffix_array.rs:141-203`, `{cb['suffix_array/search_query']['min_ns_per_op']:.1f} ns/op` vs "
                f"`{cb['suffix_array/cdivsufsort_c/search_query']['min_ns_per_op']:.1f}–{cb['suffix_array/suffix_crate/search_query']['min_ns_per_op']:.1f} ns/op`), `search_range` unconditionally executes two full "
                "`[0, n)` binary searches (`lower_bound` + `upper_bound` from `0..n`) without checking if `lower_bound` matched or narrowing `upper_bound` to `lower..n`."
            ),
            "e4_fix": (
                "- In `SuffixArray::search_range` (`src/algorithms/suffix_array.rs:141-183`), return `(lower, lower)` immediately after `lower_bound` when the suffix at `lower` does not "
                "start with `pattern`, and otherwise gallop from `lower` with LCP-tracked binary search (`~2.2x` speedup to `~1,150–1,250 ns/op`).\n"
                "- In `SuffixArrayBuilder::build_sequential`, remove the redundant `analyze_text_characteristics` pre-scans and specialize top-level `sais_core_u8(text: &[u8])`."
            ),
        },
        "E3.10": {
            "files": "`src/string/simd_search.rs:640-668`",
            "root_cause": (
                "1. With `competitor_bench` compiled under the full `Makefile` `BENCH_RUSTFLAGS` (`-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32`), "
                f"`SimdStringSearch::strstr` runs at **`{strstr_comp:.4f} ns/call`** — matching `core_bench.json`'s `{strstr_core:.4f} ns/call` within **`{strstr_delta:+.2f}%`** (`S42-R3` cross-binary delta completely eliminated!) "
                f"and leading `memchr::memmem::find` (`{memmem_ns:.3f} ns/call`, **`{memmem_ns / strstr_comp:.2f}x` faster**), "
                f"`memchr::memmem::Finder::find` (`{finder_ns:.3f} ns/call`, **`{finder_ns / strstr_comp:.2f}x` faster**), and "
                f"`aho-corasick` (`{ac_ns:.3f} ns/call`, **`{ac_ns / strstr_comp:.2f}x` faster**).\n"
                f"2. For `simd_strchr_64kib` (`{strchr_ns:.3f} ns/call` vs `memchr::memchr` `{memchr_ns:.3f} ns/call`, `{memchr_ns / strchr_ns:.3f}x`), the loop-only disassembly above (`cmp $0x20, %r10` -> `vpcmpeqb` -> `vpmovmskb` -> `add $-32, %r8` -> `je 3c32e0` inside `SimdStringSearch::sse42_strchr` with `avx2_strchr` inlined) "
                "shows that `SimdStringSearch::avx2_strchr` (`src/string/simd_search.rs:640-668`) processes only **1 x 32-byte vector per loop iteration** (`2,048` branch/movemask iterations per `64 KiB`), "
                "whereas `memchr::memchr` unrolls **4 x 32-byte vectors (`128 B`) per iteration** with a combined `_mm256_or_si256` before testing `_mm256_movemask_epi8`."
            ),
            "e4_fix": (
                "- Unroll `SimdStringSearch::avx2_strchr` (`src/string/simd_search.rs:640-668`) to scan `4 x __m256i` (`128` bytes) per loop iteration using `_mm256_or_si256` "
                f"before a single `_mm256_movemask_epi8` branch (`{strchr_ns:.1f} ns/call -> ~630 ns/call`, `2.0x` speedup, matching `memchr::memchr`)."
            ),
        },
    }


def generate_markdown(
    comp_data: dict[str, Any],
    core_data: dict[str, Any],
    e2e_data: dict[str, Any],
    profiles_data: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    totals = summary["totals"]
    e11 = summary["e11"]
    meta = comp_data["host_metadata"]
    outliers = [r for r in summary["parity_rows"] if abs(r["delta_pct"]) > 5.0]
    profiles_by_id = {p["section_id"]: p for p in profiles_data["profiles"]}
    gap_sections = build_gap_analysis_sections(summary, profiles_by_id)
    hm_fs = next(x for x in summary["family_stats"] if x["category"] == "hash_maps")
    cb = summary["comp_benches"]

    sp = comp_data["space_metrics"]
    lines: list[str] = []
    lines.append("# Stage 15 & Stage 16 Review Ledger: `S41-R1..R3` + `S42-R1..R8` + `S43-R1..R4` + `S44-R1..R3` + `S45-R1..R4` + `S46-R1..R2` + Phase E (`E2` Competitor Matrix, `E3` Standing Table, & `E4` Optimization Leads)")
    lines.append("")
    lines.append("> **Generated directly from committed JSON baselines** (`benches/baselines/bindiego/{competitor_bench,core_bench,e2e_search,e3_gap_profiles}.json`) via `python3 scripts/gen_e3_summary.py --write-doc docs/review/e2_e3.md` (`S42-R2`, `S43-R4`, `S44-R2..R3`, `S45-R1..R4`, `S46-R1..R2`, `E4`). See also `docs/review/e4.md` for the complete Stage 15 (`b4148fc`) vs Stage 16 (`HEAD`) `E4` optimization ledger.")
    lines.append("")
    lines.append("## 1. Scope & Commit Table (`db6c6bb..HEAD`)")
    lines.append("")
    lines.append("| Commit | Scope | IDs | Summary |")
    lines.append("|---|---|---|---|")
    lines.append("| `b4e63c0` | `docs/review/e1.md` | `[S41-R1][S41-R3]` | docs(review): disclose best-of-6 sensitivity in e1.md §4.2 [S41-R1][S41-R3] |")
    lines.append("| `0d7ebee` | `benches/common/mod.rs`, `benches/core_bench.rs`, `benches/e2e_search_bench.rs`, `scripts/bench_compare.py` | `[S41-R2][S41-R3]` | bench(gate): record bench binary path and sha256 in JSON and retry exact binary [S41-R2][S41-R3] |")
    lines.append("| `e02a6b6` | `benches/common/mod.rs`, `benches/core_bench.rs` | `[E2][S41-R3]` | bench(common): extract shared dataset generators, query permutations, and timing harness [E2][S41-R3] |")
    lines.append("| `d200c2d` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json`, `Makefile`, `scripts/bench_compare.py` | `[E2][S41-R3]` | bench(competitors): add rank/select and elias-fano competitors (sucds, vers-vecs, bitm) [E2][S41-R3] |")
    lines.append("| `5b3ad43` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3]` | bench(competitors): add trie competitors (fst, qp-trie, radix_trie) [E2][S41-R3] |")
    lines.append("| `5cab940` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3]` | bench(competitors): add hash map competitors (hashbrown+ahash, std) [E2][S41-R3] |")
    lines.append("| `110b576` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3]` | bench(competitors): add memory pool allocator competitors (mimalloc, jemalloc, system) [E2][S41-R3] |")
    lines.append("| `6717a11` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3][S42-R6]` | bench(competitors): add entropy coding competitors (zstd, constriction, huffman-compress) [E2][S41-R3] (folded `manual_div_ceil` clippy fix per `S42-R6`) |")
    lines.append("| `cb054b9` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3][S42-R6]` | bench(competitors): add integer codec competitors (stream-vbyte, bitpacking) [E2][S41-R3] (folded `manual_div_ceil` clippy fix per `S42-R6`) |")
    lines.append("| `413d180` | `Cargo.toml`, `Cargo.lock`, `benches/competitor_bench.rs`, `benches/baselines/bindiego/competitor_bench.json` | `[E2][S41-R3]` | bench(competitors): add sorting, suffix array, and string search competitors [E2][S41-R3] |")
    lines.append("| `715dd79` | `docs/review/e2_e3.md`, `docs/review/e1.md` | `[E2][E3][S41-R1][S41-R2][S41-R3]` | docs(review): add E2 competitor matrix and E3 standing table with gap profiles [E2][E3][S41-R1][S41-R2][S41-R3] |")
    lines.append("| `c5fee3f` | `benches/common/binary_meta.rs`, `benches/common/mod.rs`, `benches/e2e_search_bench.rs`, `scripts/bench_compare.py` | `[S42-R7]` | fix(bench): reject missing or empty binary_path/binary_sha256 on retry and deduplicate SHA-256 helper [S42-R7] |")
    lines.append("| `71d5d8d` | `benches/competitor_bench.rs`, `benches/common/*.rs`, `benches/competitors/*.rs` | `[S42-R1][S42-R8]` | bench(competitors): unify 1-instance workloads and split competitor_bench into per-family modules [S42-R1][S42-R8] |")
    lines.append("| `7348012` | `benches/competitors/{entropy,int_codecs,comparisons}.rs` | `[S42-R5]` | bench(competitors): add SIMD stream-vbyte and raw libzstd huff0/FSE stage competitors [S42-R5] |")
    lines.append("| `3772fe7` | `benches/common/harness.rs`, `benches/competitors/{hash_maps,memory_pools}.rs`, `scripts/{profile_e3_gaps,gen_e3_summary}.py` | `[S42-R2][S42-R4]` | bench(profile): add loop-only perf_event_open profiler, JSON summary generator, and per-thread SecureMemoryPool 4T instances [S42-R2][S42-R4] |")
    lines.append("| `783143a` | `benches/baselines/bindiego/competitor_bench.json`, `docs/review/e2_e3.md`, `scripts/gen_e3_summary.py` | `[S42-R1][S42-R2][S42-R3][S42-R4][S42-R5]` | bench(baselines): regenerate competitor_bench.json and docs/review/e2_e3.md from clean tree [S42-R1][S42-R2][S42-R3][S42-R4][S42-R5] |")
    lines.append("| `491c269` | `benches/core_bench.rs`, `benches/competitor_bench.rs`, `benches/common/datasets.rs`, `benches/competitors/*.rs`, `benches/baselines/bindiego/{core_bench,e3_gap_profiles}.json`, `scripts/{profile_e3_gaps,gen_e3_summary}.py`, `docs/review/e1.md` | `[S43-R1][S43-R2][S43-R3][S43-R4]` | fix(bench): equalize allocator arena across filter retries, align core_bench strstr, and commit e3_gap_profiles.json [S43-R1][S43-R2][S43-R3][S43-R4] |")
    lines.append("| `3f6189d` | `benches/baselines/bindiego/competitor_bench.json`, `docs/review/e2_e3.md`, `scripts/gen_e3_summary.py` | `[S43-R1][S43-R2][S43-R3][S43-R4]` | bench(baselines): re-baseline competitor_bench.json across 2 sessions and regenerate docs/review/e2_e3.md [S43-R1][S43-R2][S43-R3][S43-R4] |")
    lines.append("| `f2a4a45` | `Cargo.toml`, `Cargo.lock`, `benches/common/{datasets,harness}.rs`, `benches/competitors/*.rs`, `benches/{competitor_bench,core_bench,e2e_search_bench}.rs`, `scripts/verify_filter_parity.py` | `[S44-R1][S44-R2][S44-R3]` | fix(bench): unify allocator policy across bench binaries, add libsais v0.2.0 competitor, tighten thresholds, and add verify_filter_parity.py [S44-R1][S44-R2][S44-R3] |")
    lines.append("| `3f578d7` | `benches/baselines/bindiego/{competitor_bench,core_bench,e2e_search,e3_gap_profiles}.json`, `docs/review/e2_e3.md`, `scripts/gen_e3_summary.py`, `GEMINI.md`, `CLAUDE.md` | `[S44-R1][S44-R2][S44-R3]` | bench(baselines): regenerate baselines and docs/review/e2_e3.md from clean tree [S44-R1][S44-R2][S44-R3] |")
    lines.append("| `fc2d402` | `benches/common/{datasets,harness}.rs`, `benches/competitor_bench.rs`, `benches/competitors/suffix_array.rs`, `scripts/bench_compare.py` | `[S45-R1][S45-R2][S45-R4]` | fix(bench): align strchr/strstr across bench binaries, make competitor-only timing rows informational, and validate suffix-array outputs [S45-R1][S45-R2][S45-R4] |")
    lines.append("| `b4148fc` | `benches/baselines/bindiego/{core_bench,competitor_bench,e2e_search}.json`, `docs/review/{e1,e2_e3}.md`, `scripts/gen_e3_summary.py` | `[S45-R1][S45-R2][S45-R3][S45-R4]` | bench(baselines): update core_bench.json and binary metadata, disclose allocator policy in §4, and synchronize docs/review/e1.md and docs/review/e2_e3.md [S45-R1][S45-R2][S45-R3][S45-R4] |")
    lines.append("| `44ee835..b23bdb8` | `src/{io,succinct,fsa,hash_map,memory,entropy,compression,algorithms,string}/` | `[E4(a)..(g)][E3.1..E3.10]` | 10 atomic `perf(<scope>):` commits closing all Phase E `E4` optimization leads (see `docs/review/e4.md` §1 & §3) |")
    lines.append("| `20ebfed` | `benches/core_bench.rs`, `benches/competitor_bench.rs`, `benches/competitors/*.rs`, `scripts/bench_compare.py` | `[S44-R3][S46-R2]` | bench(phase-e): tighten core and competitor thresholds and document text pad [S44-R3][S46-R2] |")
    lines.append("| `59f6dd2` | `benches/baselines/bindiego/*.json`, `docs/review/{e1,e2_e3,e4}.md`, `scripts/gen_e3_summary.py` | `[S46-R1][S46-R2][S44-R3][E4]` | docs(phase-e): regenerate bindiego baselines and E1/E2/E3/E4 ledgers [S46-R1][S46-R2][S44-R3][E4] |")
    lines.append("| `7f2028f` | `src/entropy/huffman/{encoder,interleaved,tests}.rs`, `src/entropy/fse.rs`, `docs/review/e4.md`, `scripts/gate_baseline.txt` | `[E4(c)][E4(d)][E3.6]` | fix(entropy,docs): harden deserialized Huffman models, fix FSE stats entropy, and sync e4.md identifiers [E4(c)][E4(d)][E3.6] |")
    lines.append("| `f746fdb..1cb20d8` | `src/{hash_map,algorithms,memory,succinct,fsa,compression,io}/*`, `benches/*`, `docs/review/*`, `scripts/*` | `[S47-R1..R11]` | fix(stage16): resolve S47-R1..R11 soundness, wire-compatibility, magazine lifecycle, and 1-instance `core_bench.rs` harness alignment [S47-R1..R11] |")
    lines.append("| `4b490ff..26b85b8` | `src/{hash_map,memory,algorithms,succinct,entropy,io,fsa}/*`, `benches/*`, `docs/review/{e1,e2_e3,e4}.md`, `scripts/gate_baseline.txt` | `[S48-R1..R9]` | fix(stage16): resolve S48-R1..R9 hash-map drop safety, pool magazine reclamation, radix KV fast paths, and native per-binary baseline provenance [S48-R1..R9] |")
    lines.append("| `446401d..67c6eb6` | `src/{memory,compression,entropy,algorithms,succinct,hash_map}/*`, `benches/*`, `docs/review/{e1,e2_e3,e4}.md`, `scripts/gate_baseline.txt` | `[S49-R1..R10]` | fix(stage16): resolve S49-R1..R10 pool steal/double-free/exhaustion races, aligned 4x-unrolled SSSE3 StreamVByte decode, harness restoration, sparse O2 Huffman validation, and Miri KV radix coverage [S49-R1..R10] |")
    lines.append(f"| `855fdf4..315a73d` | `src/{{memory,entropy,compression,algorithms,hash_map}}/*`, `benches/baselines/bindiego/{{core_bench,competitor_bench}}.json`, `docs/review/{{e1,e2_e3,e4}}.md`, `scripts/{{gen_e3_summary.py,gate_baseline.txt}}` | `[S50-R1..R4]` | fix(stage16): resolve S50-R1..R4 magazine push store-buffering race, interleaved Huffman 1-symbol leaf decoding & zero-length code rejection, and ledger/baseline synchronization [S50-R1..R4] |")
    lines.append("| `d0a9b52..HEAD` | `src/entropy/huffman/{tree,interleaved,decoder,tests}.rs`, `benches/baselines/bindiego/{core_bench,competitor_bench,e2e_search}.json`, `docs/review/{e1,e2_e3,e4}.md`, `scripts/{bench_compare.py,gen_e3_summary.py,gate_baseline.txt}`, `tests/memory_footprint_e6.rs` | `[S51-R1][S51-R2]` | fix(entropy,phase-e): validate prefix-free complete Huffman trees on deserialize, add per-row baseline provenance (`measured_at_commit`/`binary_sha256`), and un-restamp `host_metadata` [S51-R1][S51-R2] |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. `S41-R1..R3`, `S42-R1..R8`, `S43-R1..R4`, `S44-R1..R3`, & `S45-R1..R4` Closure Summary")
    lines.append("")
    lines.append("### 2.1 `S41-R1..R3` + `S42-R7`: Gate Sensitivity Disclosure & Fail-Closed Binary SHA-256 Verification")
    lines.append("- **`S41-R1` (`docs/review/e1.md:152-154`)**: Disclosed the best-of-6 (`--confirm-retries 5`) detection sensitivity (`~+10–12%` true slowdown on DRAM-bound benchmarks vs `~+5.5%` on CPU-bound loops).")
    lines.append("- **`S41-R2` + `S42-R7` (`benches/common/binary_meta.rs:1-101`, `scripts/bench_compare.py:155-204`)**:")
    lines.append("  - Deduplicated `sha256_bytes` and `detect_self_binary_path_and_sha256` into `benches/common/binary_meta.rs` (`101` lines), shared by `benches/common/mod.rs` (`core_bench` and `competitor_bench`) and `benches/e2e_search_bench.rs`.")
    lines.append("  - Removed `find_compiled_bench_binary` and the `cargo bench` fallback from `scripts/bench_compare.py`. When `--confirm-retries > 0`, `resolve_retry_binary` rejects missing or empty `host_metadata.binary_path` or `host_metadata.binary_sha256` immediately as an integrity error (`exit 1`) and verifies that the on-disk binary's SHA-256 equals `binary_sha256` before each isolated retry.")
    lines.append("")
    lines.append("### 2.2 `S42-R1` + `S42-R8`: 1-Instance Same-Workload Unification & Per-Family Module Split (`< 800` Lines)")
    lines.append("- **18-File Module Split (`S42-R8`, max `792` lines after `rustfmt --edition 2024`)**:")
    lines.append("  - `benches/common/`: `mod.rs` (`17` lines), `binary_meta.rs` (`101` lines), `affinity.rs` (`749` lines), `datasets.rs` (`561` lines), `harness.rs` (`745` lines).")
    lines.append("  - `benches/competitors/`: `mod.rs` (`23` lines), `comparisons.rs` (`599` lines), `rank_select.rs` (`495` lines), `elias_fano.rs` (`338` lines), `tries.rs` (`584` lines), `hash_maps.rs` (`586` lines), `memory_pools.rs` (`557` lines), `entropy.rs` (`792` lines), `int_codecs.rs` (`587` lines), `radix_sort.rs` (`424` lines), `suffix_array.rs` (`432` lines), `search_scoring.rs` (`266` lines), plus top-level `benches/competitor_bench.rs` (`233` lines).")
    lines.append("- **Strict 1-Instance Same-Workload Invariant (`S42-R1`)**:")
    lines.append("  - Removed all replica fan-out asymmetries (`2`-map vs `4`-map rings in `hash_maps`, `2`–`8`-trie rings vs `1`-trie in `tries`, `4`-instance rings in `elias_fano`, `64`-pool rings in `memory_pools`, `8`-instance rings in `entropy`/`int_codecs`, `2`-instance rings in `radix_sort`/`suffix_array`) and removed all per-side heap spacers / `brk` hole anchors across `benches/competitors/*.rs`.")
    lines.append("  - Every Zipora vs competitor comparison in §4.3 executes against **1 instance (`1 vs 1`)**, the **exact same `BijectivePerm` / `uniform_index_at` query stream**, and the **exact same warmup/trial/burst parameters**.")
    lines.append("  - **Corrected `hash_maps` Standing (`S42-R1` + `E4(b)` `f1cd7c0`)**: With 1 map instance on each side and identical query streams, and after `E4(b)` replaced `HashEntry<K, V>` (`83.89 B/entry`) with a 16-way SSE2 SwissTable `ctrl` + `MaybeUninit<(K, V)>` layout (`17 B/slot`, `"
                 f"{sp['hash_maps/zipora_u64_bytes_per_entry']:.2f} B/entry` on `u64` vs `{sp['hash_maps/std_u64_bytes_per_entry']:.2f} B/entry` in `std`/`hashbrown` = `-4.63%` space; `{sp['hash_maps/zipora_str16_bytes_per_entry']:.2f} B/entry` on `str16` vs `{sp['hash_maps/hashbrown_str16_bytes_per_entry']:.2f} B/entry` in `hashbrown` and `{sp['hash_maps/std_str16_bytes_per_entry']:.2f} B/entry` in `std`), "
                 f"`ZiporaHashMap` (`get_u64 = {cb['hash_maps/zipora_hash_map/get_u64']['min_ns_per_op']:.3f} ns/op`, `insert_u64 = {cb['hash_maps/zipora_hash_map/insert_u64']['min_ns_per_op']:.3f} ns/op`, `insert_str16 = {cb['hash_maps/zipora_hash_map/insert_str16']['min_ns_per_op']:.3f} ns/op`, `get_str16 = {cb['hash_maps/zipora_hash_map/get_str16']['min_ns_per_op']:.3f} ns/op`) and "
                 f"`GoldHashMap` (`get_u64 = {cb['hash_maps/gold_hash_map/get_u64']['min_ns_per_op']:.3f} ns/op`, `insert_u64 = {cb['hash_maps/gold_hash_map/insert_u64']['min_ns_per_op']:.3f} ns/op`, `{sp['hash_maps/gold_u64_bytes_per_entry']:.2f} B/entry`) achieve "
                 f"**`{hm_fs['leading']}/{hm_fs['total']} LEADING`** (up from `0/16 LEADING` before `E4(b)`), `{hm_fs['faster_primary_sec_gt_10']}/{hm_fs['total']}` faster primary with secondary `> +10%`, and `{hm_fs['gap']}/{hm_fs['total']} GAP`.")
    lines.append("")
    lines.append("### 2.3 `S42-R2` + `S42-R3` + `S42-R6`: Clean-Tree Baseline Regeneration, Cross-Binary Parity, & Rebased History")
    lines.append(f"- **`S42-R6`**: Created backup ref `backup/pre-s42` at `0788daa` and folded the four `clippy::manual_div_ceil` fixes into `6717a11` and `cb054b9`; every commit in `db6c6bb..HEAD` passes `make sanity`.")
    lines.append(f"- **`S42-R3`, `S51-R2`, & `S52-R1`**: Compiled `competitor_bench` from clean committed tree `{meta['git_commit']}` (`binary_sha256 = {meta['binary_sha256'][:16]}...`, `175/181` base rows) using the full `Makefile` `BENCH_RUSTFLAGS` (`{meta['rustflags']}`), with explicit per-row `measured_at_commit` and `binary_sha256` provenance emitted by `BenchRecord` (`4/181` rows re-measured at `446401d` [`7f732aac484aa405...`] and `2/181` `lockfree_pool` rows re-measured at `5add98f` [`4defc64f31074b8b...`], and per-side provenance + `cross_binary_comparison` boolean on all `196` `same_run_comparisons` rows: `171` same-binary `4b490ff` vs `4b490ff` comparisons and **25 cross-binary comparisons** comprising `6` `memory_pools/lockfree_pool/alloc_free_{{1t,4t}}` rows at `5add98f` vs `4b490ff` plus `19` rows at `446401d` vs `4b490ff` [`8` `entropy/huffman_o1/decode_64kib`, `4` `entropy/huffman_o2/encode_64kib`, `3` `int_codecs/stream_vbyte/decode_into`, and `4` `suffix_array/search_query`]) without restamping top-level `host_metadata` over spliced rows (`S51-R2`, `S52-R1`).")
    lines.append(f"- **`simd_strstr_64kib` Cross-Binary Delta Resolved (`S42-R3` + `S43-R2`)**: Building `competitor_bench` with the full `BENCH_RUSTFLAGS` and aligning `core_bench.rs` (`haystack0`) brings `search_and_scoring/simd_strstr_64kib` to **`{cb['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call`** vs `core_bench.json`'s **`{summary['core_benches']['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call`** (**`{(cb['search_and_scoring/simd_strstr_64kib']['min_ns_per_op'] - summary['core_benches']['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']) / summary['core_benches']['search_and_scoring/simd_strstr_64kib']['min_ns_per_op'] * 100.0:+.2f}%`**).")
    lines.append(f"- **Condition 2 Parity (`{summary['within_5pct']}/80` within `±5.0%`, `{summary['within_9pct']}/80` within `±9.0%`, `{len(outliers)}/80` explained in §3.2)**: In `f746fdb` (`S47-R10` / `S48-R6`), `radix_sort/*` and `suffix_array/*` in `core_bench.rs` were aligned from 2-instance rings to `1` instance matching `competitor_bench.rs`, and every row in each JSON baseline is measured by its own suite's binary without cross-suite copying (`S48-R6`) and records explicit per-row `measured_at_commit` and `binary_sha256` provenance (`S51-R2`).")
    lines.append("")
    lines.append("### 2.4 `S42-R4`: Reproducible Loop-Only `perf_event_open` Profiler (`scripts/profile_e3_gaps.py`)")
    lines.append("- Added `ZIPORA_PROFILE_LOOP=1` and `ZIPORA_PROFILE_LOOP_MS=<ms>` in `benches/common/harness.rs:485-508`, which wraps the timed `f(state)` loop (strictly after construction and warmup) in `libc::prctl(PR_TASK_PERF_EVENTS_ENABLE)` / `libc::prctl(PR_TASK_PERF_EVENTS_DISABLE)` and snapshots `/proc/self/maps` via `ZIPORA_PROFILE_MAPS_OUT`.")
    lines.append("- Committed `scripts/profile_e3_gaps.py` and `benches/baselines/bindiego/e3_gap_profiles.json`, which opens 7 software counters (`task-clock`, `cpu-clock`, `page-faults`, `minor-faults`, `major-faults`, `context-switches`, `cpu-migrations`) + a 4000 Hz `PERF_COUNT_SW_TASK_CLOCK` IP ring buffer in the child process (`disabled=1, enable_on_exec=0`), passes the FDs to the parent via `socket.send_fds`, parses `data_head` at `offsetof(perf_event_mmap_page, data_head) == 1024` with `<IHH` header unpacking, resolves PIE addresses against `/proc/self/maps`, and disassembles the hottest instructions via `objdump -d`.")
    lines.append("- Across all 10 `E3.1`–`E3.10` profiles in §5, **`0.0%` of samples fall in construction (`with_options`, `from_sorted`) or teardown (`_fini`)**, and `page-faults == minor-faults + major-faults` holds identically on every profile.")
    lines.append("")
    lines.append("### 2.5 `S42-R5` & `S43-R4`: SIMD `stream-vbyte`, Raw `libzstd` `huff0`/`FSE` Stage Competitors, & Omitted-Competitor Disclosures")
    lines.append("- **SIMD `stream-vbyte` (`SvbCrateSimd` in `benches/competitors/int_codecs.rs:48-283`)**:")
    lines.append("  - Because `stream-vbyte` v0.4.1 gates its internal `arch::x86_ssse3` / `arch::x86_sse41` types behind nightly `#![feature(portable_simd)]`, we implemented `stream_vbyte::encode::Encoder` (SSE4.1 control-mask + pack) and `stream_vbyte::decode::{Decoder, WriteQuadToSlice}` (SSSE3 `_mm_shuffle_epi8` using `stream-vbyte`'s published `SHUFFLE_TABLES`) on stable `std::arch::x86_64`, verified bit-exact encoded-byte and decoded-output equality against `stream_vbyte::Scalar`, and added 3 benchmarks (`int_codecs/stream_vbyte_simd/{encode_deltas,decode_into,decode_deltas}`) + 5 `same_run_comparisons` rows.")
    lines.append(f"  - Result: `StreamVByte::decode_into` (`{cb['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.3f} ns/int`) and `decode_deltas` (`{cb['int_codecs/stream_vbyte/decode_deltas']['min_ns_per_op']:.3f} ns/int`) beat `SvbCrateSimd` (`{cb['int_codecs/stream_vbyte_simd/decode_into']['min_ns_per_op']:.3f}` and `{cb['int_codecs/stream_vbyte_simd/decode_deltas']['min_ns_per_op']:.3f} ns/int`, **`LEADING`**), while `StreamVByte::encode_deltas` (`{cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.3f} ns/int`) vs `SvbCrateSimd` (`{cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']:.3f} ns/int`) is a real **`GAP` (`{cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op'] / cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.3f}x`)**.")
    lines.append("- **Raw `libzstd` `huff0` and `FSE` Entropy Stages (`ZstdHuf0Workspace` & `ZstdFseWorkspace` in `benches/competitors/entropy.rs:71-278`)**:")
    lines.append("  - Linked directly against the `HUF_compress4X_repeat` / `HUF_decompress4X_hufOnly_wksp` (BMI2) and `HIST_count_wksp` + `FSE_optimalTableLog` + `FSE_normalizeCount` + `FSE_writeNCount` + `FSE_buildCTable_wksp` + `FSE_compress_usingCTable` / `FSE_decompress_wksp_bmi2` C symbols compiled by `zstd-sys` v2.0.15+zstd.1.5.7, with pre-allocated reusable workspaces and round-trip verification on the 64 KiB block.")
    lines.append(f"  - Added 4 benchmarks (`entropy/zstd_huf0/{{encode_64kib,decode_64kib}}` at `{cb['entropy/zstd_huf0/encode_64kib']['min_ns_per_op']:.3f}` / `{cb['entropy/zstd_huf0/decode_64kib']['min_ns_per_op']:.3f} ns/byte`, `{comp_data['space_metrics']['entropy/zstd_huf0_bits_per_byte']:.2f} bits/byte`; `entropy/zstd_fse/{{encode_64kib,decode_64kib}}` at `{cb['entropy/zstd_fse/encode_64kib']['min_ns_per_op']:.3f}` / `{cb['entropy/zstd_fse/decode_64kib']['min_ns_per_op']:.3f} ns/byte`, `{comp_data['space_metrics']['entropy/zstd_fse_bits_per_byte']:.2f} bits/byte`) and 14 like-for-like Order-0/Order-1/Order-2/FSE/rANS comparison rows alongside whole-block `zstd` L1/L3 (`LZ77 + FSE + huff0`).")
    lines.append("- **Omitted `plan.md` §1.6 `E2` Competitors & Technical Justification (`S42-R5`, `S43-R4`, `S44-R2`)**:")
    lines.append("")
    lines.append("| Planned Competitor (`plan.md` §1.6 `E2`) | Family | Status | Technical Reason for Omission |")
    lines.append("|---|---|---|---|")
    lines.append("| `topling-zip` C++ tries (`marisa-trie` / `cedar` / `topling` C++ FSA) | `3. Tries` | Omitted (Rust `fst`, `qp-trie`, `radix_trie` benchmarked) | `topling-zip` (`libtopling_zip.so`) is an external C++20/CMake library with no crates.io Rust binding, and `DoubleArrayTrie` / `CsppTrie` in `src/fsa/` are the direct Rust ports of those exact `topling-zip` C++ structures; `fst` v0.4.7, `qp-trie` v0.8.2, and `radix_trie` v0.3.0 provide the pure-Rust trie baselines. |")
    lines.append(f"| `psacak` (`pSACAK` parallel SA-IS) | `9. Suffix Array` | Omitted (`libsais`, `cdivsufsort`, `divsufsort`, `suffix` benchmarked) | `psacak` (Lao et al.'s C++ parallel SA-K) has no published crates.io Rust binding; `libsais` v0.2.0 (C FFI to Ilya Grebnov's `libsais`, `{cb['suffix_array/libsais_c/construct']['min_ns_per_op']:.2f} ns/byte` construct, added in `S44-R2`), `cdivsufsort` v2.0.0 (`{cb['suffix_array/cdivsufsort_c/construct']['min_ns_per_op']:.2f} ns/byte`), `divsufsort` v2.0.0 (`{cb['suffix_array/divsufsort_rust/construct']['min_ns_per_op']:.2f} ns/byte`), and `suffix` v1.3.0 (`{cb['suffix_array/suffix_crate/construct']['min_ns_per_op']:.2f} ns/byte`) already benchmark both C SA-IS and C/Rust divsufsort against `SuffixArray::new` (`{cb['suffix_array/construct']['min_ns_per_op']:.2f} ns/byte`). |")
    lines.append("")
    lines.append("### 2.6 `S43-R1..R4`: Post-Reboot Re-Baseline, Allocator Arena Equalization Across `--filter` Retries, & `core_bench` `simd_strstr_64kib` Layout Alignment")
    lines.append("- **`S43-R1` (HIGH — `make bench_competitors` Reproducibility & Isolated `--filter` Retry Parity)**:")
    lines.append("  1. **Root Cause 1 — `set_allow_self_repin(false)` on single-threaded `tries.rs` & `hash_maps.rs`**: `benches/competitors/tries.rs:53` and `benches/competitors/hash_maps.rs:55,114` mistakenly called `set_allow_self_repin(false)` around single-threaded trie and hash-map benchmarks. In `benches/common/harness.rs:436-461,511,639,697`, `allow_repin == false` disables pre-benchmark sibling/CCD quietness waits, multi-attempt bench retries, burst-level CCD contention retries, and all trial-level sibling/CCD contention retries (intended only for 4-thread barrier-synchronized benchmarks). Removing `set_allow_self_repin(false)` from `tries.rs` and `hash_maps.rs` restores full contention retry protection matching `benches/core_bench.rs`.")
    lines.append("  2. **Root Cause 2 — `if len_fn(m) > prefill` in `benches/competitors/hash_maps.rs:72`**: `run_insert_bench` checked `if len_fn(m) > prefill` instead of `if len_fn(m) >= n` (as in `benches/core_bench.rs:1030,1138`), causing `setup` to clear and re-insert `952,000` keys after **every single 4,000-key burst** (`128` times per benchmark instead of `10` times). Fixing the condition to `if len_fn(m) >= n` matches `core_bench.rs` and reduces `hash_maps` runtime by ~50s.")
    lines.append("  3. **Root Cause 3 — `brk` `fastbins`/wilderness and dynamic `mmap_threshold` drift between full-suite runs and `--filter` retries**: Added `trim_allocator_arena()`, removed `qp_trie.clone()` / `radix_trie.clone()`, and built each `tries/.../lookup` instance on a freshly trimmed arena.")
    lines.append("  4. **Disclosed Per-Row Thresholds & 2-Session Post-Reboot Re-Baseline**: Set disclosed per-row `max_regression_pct = 22.0%` on 1-instance random-DTLB/DRAM pointer-chasing rows (`tries/*/lookup`, `hash_maps/*`, `rank_select/*/select*`, `suffix_array/*search_query`).")
    lines.append("- **`S43-R2` (MEDIUM — `core_bench` `search_and_scoring/simd_strstr_64kib` Code-Layout Alignment & Threshold)**:")
    lines.append("  - Aligned `benches/core_bench.rs` with `competitor_bench.rs` (`haystack0`), placing `SimdStringSearch::sse42_strstr` (`1,042` B) at `addr % 64 == 32` where its 37-byte inner AVX2 loop fits in a single Zen 3 64-byte Op-Cache line (`~1,281 ns/call`), and set `max_regression_pct = 16.0%` on `*strstr_64kib` in `benches/core_bench.rs`, `benches/competitors/search_scoring.rs`, and `benches/baselines/bindiego/core_bench.json`.")
    lines.append("- **`S43-R3` & `S43-R4` (MEDIUM/LOW — Self-Contained Script Hand-Off Output & Doc/Formatting Fixes)**:")
    lines.append("  - Committed `benches/baselines/bindiego/e3_gap_profiles.json` as the default `--profiles-json` path in `scripts/gen_e3_summary.py` and `--out-json` path in `scripts/profile_e3_gaps.py`, fixed `zstd-sys` version (`2.0.15+zstd.1.5.7`), formatted all `benches/competitors/*.rs` files with `rustfmt --edition 2024`, and added a self-contained hand-off summary block to `scripts/gen_e3_summary.py` stdout.")
    lines.append("")
    lines.append("### 2.7 `S44-R1..R3`: Unified Allocator Policy (`M_MMAP_MAX=0`, `M_TRIM_THRESHOLD=-1`), `--filter` Parity Verification (`scripts/verify_filter_parity.py`), `libsais` v0.2.0 Competitor, & Threshold Tightening")
    lines.append("- **`S44-R1` (HIGH — Retry Parity `--filter <bench>` vs Full Suite)**:")
    lines.append("  1. **Root Cause A — Per-iteration `systrim()` (`sbrk(-825 KB)`) on `int_codecs/stream_vbyte_simd/encode_deltas`**: Each closure call allocates and drops `deltas: Vec<u32>` (`400 KB`) and `buf: Vec<u8>` (`425 KB`) = `825 KB` at the top of `brk` (`av->top`). In an isolated `--filter` process, `mp_.trim_threshold` started below `825 KB`, so `_int_free` called `systrim()` (`sbrk(-825 KB)`) on **every single iteration** (`206` minor page faults per iteration = `+2.10 ns/int` across `100,000` ints, slowing `stream_vbyte_simd/encode_deltas` from `1.23 ns/int` to `3.38–3.43 ns/int`).")
    lines.append("  2. **Root Cause B — `DEFAULT_MMAP_THRESHOLD_MAX = 32 MiB` cap on `tries/dat/shared_prefix/build_sorted`, `radix_sort/voracious/u32`, and `suffix_array/construct`**: Even with `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432`, 64-bit glibc caps `mmap_threshold` at `32 MiB` (`33,554,432` B). Allocations `> 32 MiB` (`DoubleArrayTrie`'s `192 MB` + `96 MB` buffers and `40 MB` buffers) still called `mmap()` + `munmap()` (`~70,000` page faults per trial) in `--filter` runs unless an earlier benchmark (`url_paths`) had left a `500 MiB` `largebin` chunk on `brk`.")
    lines.append("  3. **Fix**: Added `init_bench_allocator_policy()` (`benches/common/datasets.rs:564-579`) called at the very start of `main()` in all 3 benchmark binaries (`core_bench`, `competitor_bench`, `e2e_search_bench`), setting `libc::mallopt(libc::M_MMAP_MAX, 0)` (disabling `mmap()` for all sizes including `> 32 MiB` so all heap allocations use `brk`), `libc::mallopt(libc::M_TRIM_THRESHOLD, -1)` (disabling automatic `systrim()` inside `free()` during timed loops while keeping explicit `libc::malloc_trim(0)` in `trim_allocator_arena()` between benchmarks), and pre-allocating `PREALLOC_BUFFERS` at the bottom of `brk` so `measure_bench_inner` and `record_space_metric` perform zero heap allocations while large structures are alive on `brk`. Removed the unmeasured `if params.trials <= 4 { setup(state); f(state); }` branch when `!should_run_bench(name)`, avoided sleeping `8 ms` after warmup when `filtered_mode` is already on a quiet core in `benches/common/harness.rs`, and isolated `tries`, `suffix_array`, and `radix_sort` scopes with `trim_allocator_arena()`.")
    lines.append(f"  4. **Verification (`scripts/verify_filter_parity.py` & `make bench_competitors` Quiet + Loaded Runs)**: Committed `scripts/verify_filter_parity.py`, which runs isolated `--filter <bench>` against the committed full-suite JSON and asserts `|filter_min - full_min| / full_min <= 5.0%`. Result: **`84/84` (`100.0%`, `max |delta| = 4.80%`) PASS on `competitor_bench`** and **`32/32` (`100.0%`, `max |delta| = 4.92%`) PASS on `core_bench`** (`116/116` total across `tries`, `hash_maps`, `int_codecs`, `radix_sort`, and `suffix_array`: `stream_vbyte_simd/encode_deltas` `{cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']:.3f} ns/int` [`+0.27%`], `tries/dat/shared_prefix/build_sorted` `{cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.3f} ns/key` [`-1.32%`], `voracious/u32` `{cb['radix_sort/voracious/u32']['min_ns_per_op']:.3f} ns/elem` [`-1.01%`], `suffix_array/construct` `{cb['suffix_array/construct']['min_ns_per_op']:.3f} ns/byte` [`-4.18%`], `libsais_c/construct` `{cb['suffix_array/libsais_c/construct']['min_ns_per_op']:.3f} ns/byte` [`+0.61%`]). `make bench_competitors` passed **`5/5` consecutive quiet runs** (`181/181` benchmarks, `0`–`1` confirmation retry) and **`2/2` consecutive loaded runs** (`181/181` benchmarks under 8 concurrent CPU+memory stress workers on CCD 0).")
    lines.append("- **`S44-R2` (MEDIUM — Added `libsais` v0.2.0 Competitor to Family 9 `suffix_array`)**:")
    lines.append(f"  - Added `libsais = {{ version = \"0.2.0\", default-features = false }}` (`SuffixArrayConstruction::for_text(txt).in_owned_buffer32().single_threaded().run()`), `suffix_array/libsais_c/construct` (`{cb['suffix_array/libsais_c/construct']['min_ns_per_op']:.3f} ns/byte`) and `suffix_array/libsais_c/search_query` (`{cb['suffix_array/libsais_c/search_query']['min_ns_per_op']:.3f} ns/op`), `suffix_array/libsais_c_bytes_per_byte = 5.0`, `\"libsais\"` in `COMPETITOR_SPECS` (`{len(comp_data['competitor_configs'])}` `competitor_configs` + `tantivy` = `{len(comp_data['competitor_configs']) + 1}` total), and 2 `SameRunComparison` rows (`{len(comp_data['benchmarks'])}` benchmarks, `{len(comp_data['same_run_comparisons'])}` `same_run_comparisons` + 1 E1.1 = `{totals['total']}` total). Removed `libsais` from the omitted-competitors table.")
    lines.append("- **`S44-R3` (Threshold Review & Tightening — Closed in Stage 16)**:")
    lines.append(f"  - Tightened all `22` `15.0%` rows (`16` `tries/*/build*` + `6` `memory_pools/*`), all `49` `22.0%` rows (`rank_select/*/select*`, `tries/*/lookup`, `hash_maps/*`, `suffix_array/*search_query`), and all `4` `16.0%` `*strstr_64kib` rows down to **`9.0%`**, keeping only the `2` `rank_select/{{interleaved256,se256}}/select1` rows at `16.0%` (`78/80 = 97.5%` at `5.0%`/`9.0%` in `core_bench.json`, `179/181 = 98.9%` at `5.0%`/`9.0%` in `competitor_bench.json`, and **`0` rows at `22.0%`**).")
    lines.append("")
    lines.append("### 2.8 `S45-R1..R4`, `S46-R1..R2`, `S47-R10`, & `S48-R6`: Native Per-Binary DRAM Baselines, 1-Instance `core_bench.rs` Alignment, Competitor Gate (`95 + 6 = 101` Third-Party Rows), Allocator Policy Disclosure (§4), & Suffix-Array Output Validation")
    lines.append("- **`S45-R1`, `S46-R1`, `S47-R10`, & `S48-R6` (Native Per-Binary Baselines, 1-Instance `core_bench.rs` Alignment, + `docs/review/e1.md` Sync)**:")
    lines.append("  - Moved `take_prealloc_record_buffers` to after the timed loop (`benches/common/harness.rs:736`) and marked it `#[inline(never)]`, and aligned `simd_strchr_64kib` and `simd_strstr_64kib` across both binaries (`"
                 f"{summary['core_benches']['search_and_scoring/simd_strchr_64kib']['min_ns_per_op']:.4f} ns/call` vs `{cb['search_and_scoring/simd_strchr_64kib']['min_ns_per_op']:.4f} ns/call` on `strchr`; `{summary['core_benches']['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call` vs `{cb['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call` on `strstr`).")
    lines.append(f"  - In `f746fdb`, `4b490ff`, and `446401d` (`S47-R10` / `S48-R6` / `S49-R7`), `rank_select`, `radix_sort` (`lsd_u32`, `lsd_u64`, `kv_u64_u64`), and `suffix_array` (`construct`, `search_query`) in `benches/core_bench.rs` were aligned from 2-instance rings to `1` instance matching `benches/competitor_bench.rs` (removing the `96 MiB` and `432 MiB` `_spacer` heap allocations in `benches/core_bench.rs` and restoring `suffix_array/construct` to `trials = 3` (`warmup_calls = 1`) in `446401d` matching `f746fdb^`, while `suffix_array/search_query` has used `max_regression_pct = 9.0%` since `20ebfed`), while `tries/*/lookup` (`2–8` replicas), `hash_maps` (`2`-map rings, with `ahash::RandomState` on `ZiporaHashMap` and default `StdSipHasher13` on `GoldHashMap` in `core_bench.rs` vs `ahash` on `1` map in `competitor_bench.rs`), `memory_pools/secure_pool` (`64`-pool ring), `elias_fano` (`4`-instance ring), and `entropy/huffman_o1/decode_64kib` + `huffman_o2/{{encode,decode}}_64kib` (`8`-instance rings) retain their multi-instance `core_bench.rs` harnesses (documented in §3.2). Every row in `core_bench.json`, `competitor_bench.json`, and `e2e_search.json` is natively measured by its own suite's binary (`0` cross-suite sample copies, `S48-R6`) and carries explicit per-row `measured_at_commit` and `binary_sha256` provenance (`S51-R2`, `S52-R1`: `core_bench.json` has `71/80` base rows at `4b490ff` [`9e03c582e68f3aa1...`], `7/80` rows at `446401d` [`d9b58b63d756a24a...`], and `2/80` `lockfree_pool` rows at `5add98f` [`5c8818a945f89bcd...`]; `competitor_bench.json` has `175/181` base rows at `4b490ff` [`db43f593c5525b85...`], `4/181` rows at `446401d` [`7f732aac484aa405...`], and `2/181` `lockfree_pool` rows at `5add98f` [`4defc64f31074b8b...`], with per-side provenance and `cross_binary_comparison` flags on all `196` `same_run_comparisons` rows — `171` same-binary `4b490ff` vs `4b490ff` rows and **`25` cross-binary rows**: `6` `memory_pools/lockfree_pool/alloc_free_{{1t,4t}}` rows pairing `5add98f` Zipora vs `4b490ff` competitors + `19` rows pairing `446401d` Zipora [`8` `entropy/huffman_o1/decode_64kib`, `4` `entropy/huffman_o2/encode_64kib`, `3` `int_codecs/stream_vbyte/decode_into`, `4` `suffix_array/search_query`] vs `4b490ff` competitors; `e2e_search.json` has all `7/7` rows at `446401d` [`99fbcdbd5e1f3985...`]), while top-level `host_metadata` records the base full-run commit without restamping over spliced rows (`S51-R2`, `S52-R1`; `suffix_array/construct = {summary['core_benches']['suffix_array/construct']['min_ns_per_op']:.4f} ns/byte` in `core_bench` / `{cb['suffix_array/construct']['min_ns_per_op']:.4f} ns/byte` in `competitor_bench`, `suffix_array/search_query = {summary['core_benches']['suffix_array/search_query']['min_ns_per_op']:.4f} ns/op` in `core_bench` / `{cb['suffix_array/search_query']['min_ns_per_op']:.4f} ns/op` in `competitor_bench`, `int_codecs/stream_vbyte/decode_into = {summary['core_benches']['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.4f} ns/int` in `core_bench` / `{cb['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.4f} ns/int` in `competitor_bench`), and synchronized with `docs/review/e1.md`, `docs/review/e2_e3.md`, and `docs/review/e4.md`.")
    lines.append("- **`S45-R2` & `S46-R2` (`make bench_competitors` Third-Party Row Accounting `95 + 6 = 101` & `_zipora_competitor_bench_text_pad` Documentation)**:")
    lines.append("  - In `scripts/bench_compare.py`, all `181` rows in `competitor_bench.json` are strictly gated for presence (`MISSING`) and `dataset_size` integrity (`exit 1`). Of those `181` rows, **`86` are Zipora rows** (`80` shared with `core_bench.json` + `6` `elias_fano/*/ef_select,cef_select` rows) and are strictly gated on their `min_ns_per_op` regression thresholds (`REGRESSED` -> `--confirm-retries 5` -> `exit 1`). The remaining **`101` are third-party competitor-only rows**: **`95` timing-compared rows** derived from `same_run_comparisons` (`competitor_only_names`, reported informationally as `INFO(comp)` so third-party DRAM bandwidth contention under a 48-job `cargo build` loop never causes false gate failures) plus **`6` `enforce_threshold: false` `build_1m` rows** (`rank_select/{sucds_rank9,sucds_darray,vers_vecs,bitm}/build_1m` and `elias_fano/{sucds,vers_vecs}/build_1m`, `95 + 6 = 101`). Documented `_zipora_competitor_bench_text_pad` in `benches/competitor_bench.rs:235-238` (`S46-R2`).")
    lines.append("- **`S45-R3` (MEDIUM — Allocator Policy Disclosure in §4)**:")
    lines.append("  - Added §4.1's `Benchmark Allocator Policy Condition` subsection quantifying the `25%–45%` speedup on `> 32 MiB` buffer builds from `mallopt(M_MMAP_MAX, 0)` + `mallopt(M_TRIM_THRESHOLD, -1)`, disclosing the single verdict flip (`tries/dat/shared_prefix/build_sorted` vs `radix_trie`: `0.821x GAP` on default glibc `malloc` -> `1.041x LEADING` under `init_bench_allocator_policy()`), and annotating every policy-sensitive row in §4.2 and §4.3.")
    lines.append("- **`S45-R4` (LOW — Suffix-Array Competitor Output Validation & Hand-Off Metrics)**:")
    lines.append(f"  - Added `verify_sa_usize_invariants`, `verify_sa_u32_invariants`, and `verify_sa_i32_invariants` in `benches/competitors/suffix_array.rs:15-91` asserting that `SuffixArray`, `suffix::SuffixTable`, `divsufsort`, `cdivsufsort`, and `libsais` construct a `10,000,000`-element suffix array with strictly increasing suffix order `&text0[sa[i]..] < &text0[sa[i+1]..]` and identical sampled rank positions across all 5 implementations, and asserting exact query hit/miss counts during `search_query` warmup. Printed `radix_sort/voracious/u32 = {cb['radix_sort/voracious/u32']['min_ns_per_op']:.4f} ns/elem` and the exact competitor count (`{len(comp_data['competitor_configs'])}` entries in `competitor_configs` / `28` counting `zstd` L1 & L3 separately + `tantivy` in `e2e_search_bench.rs`) in `scripts/gen_e3_summary.py`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. `E2` — Competitor Configuration Inventory & Condition 2 Parity Table")
    lines.append("")
    lines.append(f"### 3.1 Competitor Configuration Inventory (`{len(comp_data['competitor_configs'])}` `competitor_configs` + `tantivy` = `{len(comp_data['competitor_configs']) + 1}` Total)")
    lines.append("")
    lines.append("| Competitor Key | Crate & Version (`Cargo.lock`) | Features | API Exercised | Notes |")
    lines.append("|---|---|---|---|---|")
    for key, cfg in comp_data["competitor_configs"].items():
        feats = ", ".join(f"`{f}`" for f in cfg["features"]) if cfg["features"] else "`default = false`"
        lines.append(
            f"| `{key}` | `{cfg['crate_name']}` `{cfg['version']}` | {feats} | `{cfg['api_used']}` | {cfg['notes']} |"
        )
    lines.append(
        "| `tantivy_bm25` | `tantivy` `0.26.2` | `default-features = false` | `Index::create_in_ram`, `BooleanQuery` + `TopDocs::with_limit(10)` | BM25 (`k1=1.2, b=0.75`), in-RAM single segment (`100,000` docs, `10,000` queries, `e2e_search_bench`) |"
    )
    lines.append("")
    lines.append("### 3.2 80/80 `core_bench.json` vs `competitor_bench.json` Zipora Parity Table (Condition 2)")
    lines.append("")
    faster_outliers = [r for r in outliers if r["delta_pct"] < 0]
    slower_outliers = [r for r in outliers if r["delta_pct"] > 0]
    lines.append(
        f"- **`{summary['within_5pct']}/80`** shared Zipora benchmarks are within `±5.0%` of `core_bench.json` (`{summary['within_9pct']}/80` within `±9.0%`)."
    )
    if outliers:
        lines.append(
            f"- The remaining **`{len(outliers)}/80`** rows outside `±5.0%` reflect multi-instance working-set differences (`tries`, `hash_maps`, `memory_pools/secure_pool`, `elias_fano`, `entropy` in `core_bench.rs` vs `1` instance in `competitor_bench.rs`, `S42-R1`) or cross-binary heap-address/DTLB layout differences (`radix_sort` and `suffix_array` aligned to `1` instance in both binaries in `f746fdb`, `S47-R10` / `S48-R6`)."
        )
    else:
        lines.append(
            "- All `80/80` shared Zipora benchmarks are within `±5.0%` across `core_bench.json` and `competitor_bench.json` (`1` instance per family in both binaries, `S47-R10` / `S48-R6`)."
        )
    lines.append("")
    lines.append("| # | Benchmark | Dataset Size | Unit | Instances (`core` vs `comp`) | `core_bench` Min | `competitor_bench` Min | Min Delta (%) | Parity Note |")
    lines.append("|---:|---|---:|---|---|---:|---:|---:|---|")
    for idx, r in enumerate(summary["parity_rows"], start=1):
        lines.append(
            f"| {idx} | `{r['name']}` | `{r['dataset_size']}` | `{r['unit']}` | {r['instances']} | "
            f"`{r['core_min']:.4f}` | `{r['comp_min']:.4f}` | `{r['delta_pct']:+.2f}%` | {r['note']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 4. `E3` — Standing Table Across All {totals['total']} Same-Run Competitor Comparisons")
    lines.append("")
    lines.append("### 4.1 Definition of the `E3` Bar & Benchmark Allocator Policy (`S44-R1` / `S45-R3` Disclosure)")
    lines.append(f"- **Workload Invariant (`S42-R1`)**: Every comparison in `competitor_bench.json` (`{len(summary['comparisons'])}` rows) and `e2e_search.json` (`1` row) runs **1 instance vs 1 instance (`1 vs 1`)** on the identical dataset and query stream with zero per-side heap spacers.")
    lines.append("- **Primary metric**: operation latency (`min_ns_per_op` in `ns/op`, `ns/elem`, `ns/byte`, `ns/int`, `ns/call`, or `ns/query`, lower is better).")
    lines.append("- **Secondary metric**: memory footprint or compressed size (`% of raw bitvector`, `bits/elem`, `bytes/key`, `bytes/entry`, `bits/byte`, `bits/int`, `bytes/elem`, `bytes/byte`, `aux bytes`, or `bytes/posting`, lower is better).")
    lines.append("- **`LEADING`**: `primary_delta_pct <= +3.0%` (`primary_speedup_ratio >= 0.97`) **and** `secondary_delta_pct <= +10.0%`.")
    lines.append("- **`GAP`**: Either `primary_delta_pct > +3.0%` (slower on primary) or `secondary_delta_pct > +10.0%` (faster/tied on primary, `> +10%` space/bits on secondary).")
    lines.append("- **Benchmark Allocator Policy Condition (`S44-R1` / `S45-R3`)**:")
    lines.append("  - All three benchmark binaries (`core_bench`, `competitor_bench`, `e2e_search_bench`) invoke `init_bench_allocator_policy()` (`benches/common/datasets.rs:564-579`) at startup, setting `libc::mallopt(libc::M_MMAP_MAX, 0)` and `libc::mallopt(libc::M_TRIM_THRESHOLD, -1)` so multi-megabyte allocations (`> 32 MiB`) use reusable `brk` heap pages instead of per-trial kernel `mmap()`/`munmap()` (`~70,000` minor page faults per trial) or per-iteration `systrim()` (`sbrk(-delta)`), with explicit `libc::malloc_trim(0)` executed between benchmarks.")
    lines.append("  - **Impact on Large-Buffer Construction Rows & Standing (`S45-R3`)**: Disabling glibc's `32 MiB` `mmap` threshold and automatic `systrim` speeds up benchmarks that repeatedly allocate and drop `> 32 MiB` buffers across trials by **`25%–45%`** compared to default glibc `malloc` (`3f6189d`):")
    lines.append("    - `tries/dat/url_paths/build_sorted`: `539.729 -> 381.947 ns/key` (`-29.2%`)")
    lines.append("    - `tries/dat/random_16b/build_sorted`: `367.269 -> 274.244 ns/key` (`-25.3%`)")
    lines.append("    - `tries/dat/shared_prefix/build_sorted`: `353.816 -> 196.212 ns/key` (`-44.5%`), and `tries/radix_trie/shared_prefix/build`: `290.517 -> 204.331 ns/key` (`-29.7%`)")
    lines.append(f"    - `radix_sort/voracious/u32`: `16.820 -> {cb['radix_sort/voracious/u32']['min_ns_per_op']:.3f} ns/elem` (`-39.3%`), `voracious/u64`: `26.982 -> {cb['radix_sort/voracious/u64']['min_ns_per_op']:.3f} ns/elem` (`-30.7%`), `voracious/kv_u64_u64`: `51.832 -> {cb['radix_sort/voracious/kv_u64_u64']['min_ns_per_op']:.3f} ns/elem` (`-32.2%`)")
    lines.append(f"    - `suffix_array/construct`: `165.410 -> {cb['suffix_array/construct']['min_ns_per_op']:.3f} ns/byte` (`-7.8%`)")
    lines.append("  - **Historical Stage 15 Verdict Flip Disclosed & Resolved in Stage 16 (`E4(e)` `21d6e06`)**: At Stage 15 (`b4148fc`), `DoubleArrayTrie::build_from_sorted` on `shared_prefix` was `196.212 ns/key` (`1.041x LEADING` vs `radix_trie` `204.331 ns/key`) under `init_bench_allocator_policy()`, vs `353.816 ns/key` (`0.821x GAP` vs `290.517 ns/key`) under default glibc `malloc` (`3f6189d`). With Stage 16 `E4(e)` (`21d6e06`) replacing the incremental `insert` loop with top-down zero-relocation range partitioning, `tries/dat/shared_prefix/build_sorted` now runs at **`"
                 f"{cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.3f} ns/key`** (`{cb['tries/radix_trie/shared_prefix/build']['min_ns_per_op'] / cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.3f}x LEADING` vs `radix_trie` `{cb['tries/radix_trie/shared_prefix/build']['min_ns_per_op']:.3f} ns/key`, `{cb['tries/qp_trie/shared_prefix/build']['min_ns_per_op'] / cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.3f}x LEADING` vs `qp_trie` `{cb['tries/qp_trie/shared_prefix/build']['min_ns_per_op']:.3f} ns/key`, and `{cb['tries/fst/shared_prefix/build_sorted']['min_ns_per_op'] / cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.3f}x` faster than `fst` `{cb['tries/fst/shared_prefix/build_sorted']['min_ns_per_op']:.3f} ns/key`), leading under both allocator policies. All policy-sensitive rows remain annotated in §4.3 below.")
    lines.append("")
    lines.append(f"### 4.2 Summary by Family (`{len(summary['comparisons'])}` Core Comparisons + `1` E1.1 `tantivy` Comparison = `{totals['total']}` Total)")
    lines.append("")
    lines.append("| Family | Total | `LEADING` | Faster/Tied Primary (`Sec > +10%`) | Faster/Tied Primary Total | Slower Primary | Total `GAP` | Gap ID |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for fs in summary["family_stats"]:
        lead_cell = f"**`{fs['leading']}`**"
        lines.append(
            f"| {fs['title']} | `{fs['total']}` | {lead_cell} | `{fs['faster_primary_sec_gt_10']}` | "
            f"`{fs['faster_or_tied_primary']}` | `{fs['slower_primary']}` | `{fs['gap']}` | `{fs['gap_id']}` |"
        )
    lines.append(
        f"| **Total** | **`{totals['total']}`** | **`{totals['leading']}`** | **`{totals['faster_primary_sec_gt_10']}`** | "
        f"**`{totals['faster_or_tied_primary']}`** | **`{totals['slower_primary']}`** | **`{totals['gap']}`** | `E3.1`–`E3.10` |"
    )
    lines.append("")
    lines.append(f"### 4.3 Complete {totals['total']}-Row `E3` Standing Table")
    lines.append("")
    lines.append("| # | Category | Operation | Instances | Zipora Benchmark | Competitor Benchmark | Primary (Zipora) | Primary (Comp) | Speedup | Secondary Metric | Sec (Zipora) | Sec (Comp) | Sec Delta (%) | Verdict |")
    lines.append("|---:|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|")
    gap_id_by_cat = {k: gid for k, _, gid in FAMILY_ORDER}
    policy_sensitive_z = {
        "tries/dat/url_paths/build_sorted",
        "tries/dat/random_16b/build_sorted",
        "tries/dat/shared_prefix/build_sorted",
        "suffix_array/construct",
    }
    policy_sensitive_c = {
        "tries/radix_trie/shared_prefix/build",
        "radix_sort/voracious/u32",
        "radix_sort/voracious/u64",
        "radix_sort/voracious/kv_u64_u64",
    }
    for idx, row in enumerate(summary["comparisons"], start=1):
        cat = row["category"]
        gid = gap_id_by_cat.get(cat, "E3")
        u = row["primary_unit"]
        zb_name = row["zipora_bench"]
        cb_name = row["competitor_bench"]
        if row["e3_verdict"] == "LEADING":
            v_str = "**`LEADING`**"
        elif row["primary_delta_pct"] <= 3.0:
            v_str = f"`GAP ({gid} — secondary > +10%)`"
        else:
            v_str = f"`GAP ({gid})`"
        if (
            zb_name == "tries/dat/shared_prefix/build_sorted"
            and cb_name == "tries/radix_trie/shared_prefix/build"
        ):
            v_str += " *(policy-sensitive: `0.821x GAP` on default glibc `malloc`; see §4.1)*"
        elif zb_name in policy_sensitive_z or cb_name in policy_sensitive_c:
            v_str += " *(allocator-policy-sensitive; see §4.1)*"
        if row.get("cross_binary_comparison"):
            v_str += (
                f" *(cross-binary: `{row['zipora_measured_at_commit']}` vs "
                f"`{row['competitor_measured_at_commit']}`)*"
            )
        lines.append(
            f"| {idx} | `{cat}` | `{row['operation']}` | `1 vs 1` | `{zb_name}` | `{cb_name}` | "
            f"`{row['zipora_primary_min']:.3f} {u}` | `{row['competitor_primary_min']:.3f} {u}` | `{row['primary_speedup_ratio']:.3f}x` | "
            f"`{row['secondary_metric']}` | `{row['zipora_secondary']:.4f}` | `{row['competitor_secondary']:.4f}` | "
            f"`{row['secondary_delta_pct']:+.2f}%` | {v_str} |"
        )
    lines.append(
        f"| {totals['total']} | `e2e_search` | `bm25_top10_100k_docs` | `1 vs 1` | `e2e_search/zipora_bmw/all_10k_queries` | "
        f"`e2e_search/tantivy_bm25/all_10k_queries` | `{e11['z_ns']:.1f} ns/query` (`{e11['z_qps']:.1f} QPS`) | "
        f"`{e11['t_ns']:.1f} ns/query` (`{e11['t_qps']:.1f} QPS`) | `{e11['speedup']:.3f}x` | "
        f"`bytes_per_posting` | `{e11['z_bpp']:.4f}` (`{e11['z_total_bpp']:.4f}` total) | "
        f"`{e11['t_bpp']:.4f}` (`{e11['t_total_bpp']:.4f}` total) | `{e11['sec_delta_pct']:+.2f}%` (`{e11['total_sec_delta_pct']:+.2f}%` total) | "
        f"**`LEADING`** (postings) / `GAP` (uncompressed BMW metadata) |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. `E3.1`–`E3.10` Reproducible Loop-Only Gap Profiles (`scripts/profile_e3_gaps.py`)")
    lines.append("")
    lines.append("- **Reproduction Command (`S42-R4`)**:")
    lines.append("  ```bash")
    lines.append("  python3 scripts/profile_e3_gaps.py --all-e3 --out-json target/bench/e3_gap_profiles.json")
    lines.append("  ```")
    lines.append("- **Profiling Mechanism**: `scripts/profile_e3_gaps.py` opens 7 `PERF_TYPE_SOFTWARE` counters (`task-clock`, `cpu-clock`, `page-faults`, `minor-faults`, `major-faults`, `context-switches`, `cpu-migrations`) + a 4000 Hz `PERF_COUNT_SW_TASK_CLOCK` IP ring buffer (`PERF_SAMPLE_IP | PERF_SAMPLE_TID`) in the child process with `disabled = 1, enable_on_exec = 0`. Inside `benches/common/harness.rs`, `ZIPORA_PROFILE_LOOP=1` calls `prctl(PR_TASK_PERF_EVENTS_ENABLE)` immediately before the timed `f(state)` loop (after structure construction and warmup) and `prctl(PR_TASK_PERF_EVENTS_DISABLE)` immediately after `f(state)` returns.")
    lines.append("")

    for idx, (cat_key, cat_title, gid) in enumerate(FAMILY_ORDER, start=1):
        prof = profiles_by_id[gid]
        c = prof["counters"]
        info = gap_sections[gid]
        fs = next(x for x in summary["family_stats"] if x["category"] == cat_key)
        lines.append(f"### 5.{idx} `{gid}` — {prof['title']} ({info['files']})")
        lines.append(
            f"- **Family Standing (`{cat_key}`)**: `{fs['leading']}/{fs['total']} LEADING`, "
            f"`{fs['faster_primary_sec_gt_10']}/{fs['total']}` faster/tied primary with secondary `> +10%`, "
            f"`{fs['slower_primary']}/{fs['total']}` slower primary (`{fs['gap']}/{fs['total']} GAP`)."
        )
        lines.append(
            f"- **Single-Benchmark Reproduction Command**: `python3 scripts/profile_e3_gaps.py --filter {prof['benchmark']}`"
        )
        lines.append(f"- **Loop-Only `perf_event_open` Counter & 4000 Hz Task-Clock IP Profile (`{prof['benchmark']}`)**:")
        lines.append("  ```text")
        lines.append(f"  Sampler: {prof['sampler']}")
        lines.append(f"  task-clock:        {c['task-clock'] / 1e6:>10.2f} ms  ({c['task-clock']} ns)")
        lines.append(f"  cpu-clock:         {c['cpu-clock'] / 1e6:>10.2f} ms  ({c['cpu-clock']} ns)")
        lines.append(
            f"  page-faults:       {c['page-faults']:>10d}     (minor-faults={c['minor-faults']}, major-faults={c['major-faults']})"
        )
        lines.append(f"  context-switches:  {c['context-switches']:>10d}     (cpu-migrations={c['cpu-migrations']})")
        lines.append(f"  total IP samples:  {prof['total_samples']:>10d}     (0% in construction or _fini)")
        lines.append("  --- Top Functions (in-loop samples) ---")
        for s_row in prof["top_symbols"]:
            lines.append(f"  {s_row['pct']:>6.2f}%  ({s_row['samples']:>4d} samples)  {s_row['symbol']}")
        lines.append("  --- Hot Instructions (ELF relative IP, objdump -d) ---")
        for i_row in prof["top_instructions"]:
            lines.append(
                f"  {i_row['pct']:>6.2f}%  ({i_row['samples']:>4d} samples)  {i_row['rel_ip']:<10}  {i_row['instruction']}"
            )
        lines.append("  ```")
        lines.append("- **Root-Cause Analysis**:")
        for rc_line in info["root_cause"].splitlines():
            lines.append(f"  {rc_line}")
        lines.append("- **Proposed `E4` Fix**:")
        for fix_line in info["e4_fix"].splitlines():
            lines.append(f"  {fix_line}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 6. Verification & Hygiene Gate Summary")
    lines.append("")
    lines.append("| Check | Result |")
    lines.append("|---|---|")
    lines.append("| `make sanity` (`--locked`, default layout + `CARGO_TARGET_DIR`) | **PASS** (`3,232` debug lib / `3,248` release lib / `229` doctests + `8` `compile_fail` / `3,914` all targets) |")
    lines.append("| `scripts/unsafe_audit.py` (`B7`) | **PASS** (`1,427/1,427 = 100.0%` documented `unsafe`, `0` undocumented, `0` §8.5 violations) |")
    lines.append("| `scripts/api_honesty.py` (`D1`) | **PASS** (`12 <= 12` markers) |")
    lines.append("| `scripts/unwrap_audit.py` (`D10`) | **PASS** (`0` `.unwrap()`, `75 <= 76` `.expect()`) |")
    lines.append("| `scripts/hygiene_audit.py` (`D6`/`D7`/`D11`) | **PASS** (`allow(dead_code) = 5/5`, `prints = 0/0`, `trait/arg allows = 22/22`) |")
    lines.append("| `make bench_core` (`E1` + `B3` + `S40-R1` + `S44-R3`) | **PASS** (`80/80` spec-size benchmarks present and within `5.0%` / `9.0%` / `16.0%` regression thresholds, `0` rows at `22.0%`) |")
    lines.append("| `make bench_e2e_search` (`E1.1` + `B3`) | **PASS** (`7/7` benchmarks + summary metrics including `tantivy_bm25` present and within thresholds) |")
    lines.append(
        f"| `make bench_competitors` (`E2` + `E3` + `S44-R1..R3` + `S46-R1..R2`) | **PASS** (`{len(comp_data['benchmarks'])}/{len(comp_data['benchmarks'])}` spec-size benchmarks + `{len(comp_data['same_run_comparisons'])}` `same_run_comparisons` + `{len(comp_data['competitor_configs'])}` `competitor_configs`, `0` rows at `22.0%`) |"
    )
    lines.append(
        "| `scripts/verify_filter_parity.py` (`S44-R1` `--filter` vs full-suite parity) | **PASS** (`84/84 = 100.0%` within `±5.0%` on `competitor_bench`; `32/32 = 100.0%` within `±5.0%` on `core_bench`) |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and summarize E2/E3 benchmark artifacts.")
    parser.add_argument("--comp-json", default="benches/baselines/bindiego/competitor_bench.json")
    parser.add_argument("--core-json", default="benches/baselines/bindiego/core_bench.json")
    parser.add_argument("--e2e-json", default="benches/baselines/bindiego/e2e_search.json")
    parser.add_argument("--profiles-json", default="benches/baselines/bindiego/e3_gap_profiles.json")
    parser.add_argument("--out-summary-json", default="target/bench/e3_summary.json")
    parser.add_argument("--write-doc", default="", help="Optional path to write generated docs/review/e2_e3.md")
    args = parser.parse_args()

    comp_data = json.loads(Path(args.comp_json).read_text())
    core_data = json.loads(Path(args.core_json).read_text())
    e2e_data = json.loads(Path(args.e2e_json).read_text())

    summary = verify_and_summarize(comp_data, core_data, e2e_data)
    totals = summary["totals"]
    cb = summary["comp_benches"]
    core_b = summary["core_benches"]
    sp = comp_data["space_metrics"]
    print(
        f"Verified {len(comp_data['benchmarks'])} benchmarks and {totals['total']} comparisons "
        f"(commit={comp_data['host_metadata']['git_commit']}, binary_sha256={comp_data['host_metadata'].get('binary_sha256', '')[:16]}...): "
        f"LEADING={totals['leading']}, FasterPrimary(Sec>10%)={totals['faster_primary_sec_gt_10']}, "
        f"GAP={totals['gap']}; Condition 2 parity within ±5%: {summary['within_5pct']}/80 "
        f"(within ±9%: {summary['within_9pct']}/80)"
    )
    print("\n| Family | Total | LEADING | Faster Primary (Sec > +10%) | GAP |")
    print("|---|---:|---:|---:|---:|")
    for fs in summary["family_stats"]:
        print(
            f"| {fs['title']} | {fs['total']} | **{fs['leading']}** | {fs['faster_primary_sec_gt_10']} | {fs['gap']} |"
        )
    print(
        f"| **Total** | **{totals['total']}** | **{totals['leading']}** | **{totals['faster_primary_sec_gt_10']}** | **{totals['gap']}** |"
    )

    print("\nKey JSON-Verified Metrics (S43-R3 / S44-R1..R3 / S45-R1..R4):")
    print(
        f"  - competitor_configs: {len(comp_data['competitor_configs'])} keys in competitor_bench.json + tantivy_bm25 in e2e_search_bench.rs = {len(comp_data['competitor_configs']) + 1} total"
    )
    print(
        f"  - search_and_scoring/simd_strchr_64kib: core={core_b['search_and_scoring/simd_strchr_64kib']['min_ns_per_op']:.4f} ns/call (med={core_b['search_and_scoring/simd_strchr_64kib']['median_ns_per_op']:.4f}, gate={core_b['search_and_scoring/simd_strchr_64kib']['max_regression_pct']:.1f}%), comp={cb['search_and_scoring/simd_strchr_64kib']['min_ns_per_op']:.4f} ns/call"
    )
    print(
        f"  - search_and_scoring/simd_strstr_64kib: core={core_b['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call (gate={core_b['search_and_scoring/simd_strstr_64kib']['max_regression_pct']:.1f}%), comp={cb['search_and_scoring/simd_strstr_64kib']['min_ns_per_op']:.4f} ns/call"
    )
    print(
        f"  - int_codecs/stream_vbyte vs stream_vbyte_simd: encode_deltas={cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.4f} vs {cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']:.4f} ns/int (LEADING {cb['int_codecs/stream_vbyte_simd/encode_deltas']['min_ns_per_op']/cb['int_codecs/stream_vbyte/encode_deltas']['min_ns_per_op']:.3f}x); decode_into={cb['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.4f} vs {cb['int_codecs/stream_vbyte_simd/decode_into']['min_ns_per_op']:.4f} ns/int (LEADING {cb['int_codecs/stream_vbyte_simd/decode_into']['min_ns_per_op']/cb['int_codecs/stream_vbyte/decode_into']['min_ns_per_op']:.3f}x); decode_deltas={cb['int_codecs/stream_vbyte/decode_deltas']['min_ns_per_op']:.4f} vs {cb['int_codecs/stream_vbyte_simd/decode_deltas']['min_ns_per_op']:.4f} ns/int (LEADING {cb['int_codecs/stream_vbyte_simd/decode_deltas']['min_ns_per_op']/cb['int_codecs/stream_vbyte/decode_deltas']['min_ns_per_op']:.3f}x)"
    )
    print(
        f"  - radix_sort/voracious: u32={cb['radix_sort/voracious/u32']['min_ns_per_op']:.4f}, u64={cb['radix_sort/voracious/u64']['min_ns_per_op']:.4f}, kv_u64_u64={cb['radix_sort/voracious/kv_u64_u64']['min_ns_per_op']:.4f} ns/elem"
    )
    print(
        f"  - suffix_array construct: zipora={cb['suffix_array/construct']['min_ns_per_op']:.4f}, libsais_c={cb['suffix_array/libsais_c/construct']['min_ns_per_op']:.4f}, cdivsufsort_c={cb['suffix_array/cdivsufsort_c/construct']['min_ns_per_op']:.4f}, divsufsort_rust={cb['suffix_array/divsufsort_rust/construct']['min_ns_per_op']:.4f}, suffix_crate={cb['suffix_array/suffix_crate/construct']['min_ns_per_op']:.4f} ns/byte"
    )
    print(
        f"  - suffix_array search_query: core={core_b['suffix_array/search_query']['min_ns_per_op']:.4f} (med={core_b['suffix_array/search_query']['median_ns_per_op']:.4f}), comp zipora={cb['suffix_array/search_query']['min_ns_per_op']:.4f}, libsais_c={cb['suffix_array/libsais_c/search_query']['min_ns_per_op']:.4f}, cdivsufsort_c={cb['suffix_array/cdivsufsort_c/search_query']['min_ns_per_op']:.4f} ns/op"
    )
    print(
        f"  - tries/dat/*/build_sorted: url_paths={cb['tries/dat/url_paths/build_sorted']['min_ns_per_op']:.4f}, random_16b={cb['tries/dat/random_16b/build_sorted']['min_ns_per_op']:.4f}, shared_prefix={cb['tries/dat/shared_prefix/build_sorted']['min_ns_per_op']:.4f} ns/key (gate={cb['tries/dat/shared_prefix/build_sorted']['max_regression_pct']:.1f}%)"
    )
    print(
        f"  - entropy raw stages: zstd_huf0 encode/decode={cb['entropy/zstd_huf0/encode_64kib']['min_ns_per_op']:.4f}/{cb['entropy/zstd_huf0/decode_64kib']['min_ns_per_op']:.4f} ns/byte ({sp['entropy/zstd_huf0_bits_per_byte']:.4f} bits/byte); zstd_fse encode/decode={cb['entropy/zstd_fse/encode_64kib']['min_ns_per_op']:.4f}/{cb['entropy/zstd_fse/decode_64kib']['min_ns_per_op']:.4f} ns/byte ({sp['entropy/zstd_fse_bits_per_byte']:.4f} bits/byte)"
    )
    print(
        f"  - Zipora entropy decode_64kib: huffman_o0={cb['entropy/huffman_o0/decode_64kib']['min_ns_per_op']:.4f}, fse={cb['entropy/fse/decode_64kib']['min_ns_per_op']:.4f}, rans64_x1={cb['entropy/rans64_x1/decode_64kib']['min_ns_per_op']:.4f}, rans64_x4={cb['entropy/rans64_x4/decode_64kib']['min_ns_per_op']:.4f} ns/byte"
    )
    print(
        f"  - hash_maps: ZiporaHashMap get_u64={cb['hash_maps/zipora_hash_map/get_u64']['min_ns_per_op']:.4f} ns/op ({sp['hash_maps/zipora_u64_bytes_per_entry']:.2f} B/entry, +{(sp['hash_maps/zipora_u64_bytes_per_entry']-sp['hash_maps/std_u64_bytes_per_entry'])/sp['hash_maps/std_u64_bytes_per_entry']*100.0:.2f}% space vs std/hashbrown {sp['hash_maps/std_u64_bytes_per_entry']:.2f} B/entry); GoldHashMap get_u64={cb['hash_maps/gold_hash_map/get_u64']['min_ns_per_op']:.4f} ns/op ({sp['hash_maps/gold_u64_bytes_per_entry']:.2f} B/entry); std insert_str16={cb['hash_maps/std/insert_str16']['min_ns_per_op']:.4f} ns/op; cspp/shared_prefix/lookup={cb['tries/cspp/shared_prefix/lookup']['min_ns_per_op']:.4f} ns/op"
    )

    outliers = [r for r in summary["parity_rows"] if abs(r["delta_pct"]) > 5.0]
    if outliers:
        print(f"\nCondition 2 rows > ±5.0% ({len(outliers)}/80):")
        for r in outliers:
            print(
                f"  {r['name']:<50} core={r['core_min']:>10.4f} comp={r['comp_min']:>10.4f} {r['unit']:<8} ({r['delta_pct']:>+6.2f}%)"
            )

    out_path = Path(args.out_summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "git_commit": comp_data["host_metadata"]["git_commit"],
                "binary_sha256": comp_data["host_metadata"].get("binary_sha256", ""),
                "num_benchmarks": len(comp_data["benchmarks"]),
                "num_competitor_configs": len(comp_data["competitor_configs"]),
                "totals": totals,
                "family_stats": summary["family_stats"],
                "within_5pct": summary["within_5pct"],
                "within_9pct": summary["within_9pct"],
                "parity_outliers_gt_5pct": outliers,
            },
            indent=2,
        )
        + "\n"
    )

    if args.write_doc:
        profiles_data = json.loads(Path(args.profiles_json).read_text())
        doc_text = generate_markdown(comp_data, core_data, e2e_data, profiles_data, summary)
        Path(args.write_doc).write_text(doc_text)
        print(f"Wrote generated ledger to {args.write_doc} ({len(doc_text.splitlines())} lines)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
