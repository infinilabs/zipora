#!/usr/bin/env python3
"""Generate Verified Performance tables and Key Features sections from committed
baselines (`E7`) and audit `README.md`, `README_cn.md`, and `docs/PERFORMANCE.md`
numeric and capability claims against tests/baselines (`F2` / `S52-R5`).

Usage:
  python3 scripts/perf_table.py
  python3 scripts/perf_table.py --update
  python3 scripts/perf_table.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "benches" / "baselines" / "bindiego"

BEGIN_MARKER = "<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->"
END_MARKER = "<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->"
BEGIN_KF_MARKER = "<!-- BEGIN GENERATED KEY FEATURES (scripts/perf_table.py) -->"
END_KF_MARKER = "<!-- END GENERATED KEY FEATURES (scripts/perf_table.py) -->"

ALLOWED_HTML_COMMENTS = {
    BEGIN_MARKER,
    END_MARKER,
    BEGIN_KF_MARKER,
    END_KF_MARKER,
}

# Verified structural/E6 constants asserted in tests/memory_footprint_e6.rs
E6_DAT_MAP_BYTES_PER_STATE = "12.0–16.0 B/state"
E6_FIELDNORM_BYTES_PER_DOC = "1.0 B/doc"

EXTERNAL_ALLOWED_TYPES = {
    "Vec",
    "String",
    "Arc",
    "Rc",
    "Box",
    "Mutex",
    "RwLock",
    "HashMap",
    "BTreeMap",
    "VecDeque",
    "Option",
    "Result",
    "Rank9Sel",
    "EliasFanoVec",
    "Trie",
    "ParallelX1",
    "ParallelX2",
    "ParallelX4",
    "ParallelX8",
    "System",
    "SmallFloat",
    "VarInt",
    "SwissTable",
    "KiB",
    "MiB",
    "GiB",
    "Gbit",
}


def load_baselines() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with open(BASELINE_DIR / "core_bench.json", encoding="utf-8") as f:
        core = json.load(f)
    with open(BASELINE_DIR / "competitor_bench.json", encoding="utf-8") as f:
        comp = json.load(f)
    with open(BASELINE_DIR / "e2e_search.json", encoding="utf-8") as f:
        e2e = json.load(f)
    return core, comp, e2e


def format_secondary(metric: str, val: float) -> str:
    if metric == "total_space_pct_of_raw":
        return f"{val:.2f}% of raw"
    if metric == "bits_per_elem":
        return f"{val:.2f} bits/elem"
    if metric == "bytes_per_key":
        return f"{val:.2f} B/key"
    if metric == "bytes_per_entry":
        return f"{val:.2f} B/entry"
    if metric == "bits_per_byte":
        return f"{val:.2f} bits/B"
    if metric == "bits_per_int":
        return f"{val:.2f} bits/int"
    if metric == "bytes_per_elem":
        return f"{val:.2f} B/elem"
    if metric == "aux_bytes":
        return f"{val:.0f} B aux"
    return f"{val:.2f}"


def generate_key_features_en(
    core: dict[str, Any],
    comp: dict[str, Any],
) -> str:
    space = core["space_metrics"]
    pair_map = {
        (c["zipora_bench"], c["competitor_bench"]): c
        for c in comp["same_run_comparisons"]
    }
    r9_ovh = space["rank_select/rank9_overhead_pct"]
    il_sel_ovh = (
        pair_map[
            ("rank_select/interleaved256/rank1", "rank_select/sucds_rank9sel/rank1")
        ]["zipora_secondary"]
        - 100.0
    )
    hm_u64_b = pair_map[
        ("hash_maps/zipora_hash_map/get_u64", "hash_maps/std/get_u64")
    ]["zipora_secondary"]
    dat_bps = space["tries/dat_shared_prefix_bytes_per_state"]
    cspp_bpk = space["tries/cspp_shared_prefix_bytes_per_key"]

    lines = [
        BEGIN_KF_MARKER,
        "- **High Performance**: Zero-copy operations, SIMD optimizations (AVX2, AVX-512), cache-friendly layouts, SIMD cursor primitives for Block-Max WAND",
        "- **Memory Safety**: Every production `unsafe` site is annotated with a `// SAFETY:` or `# Safety` comment (`scripts/unsafe_audit.py`)",
        "- **Secure Memory Management**: Production-ready memory pools (`SecureMemoryPool`, `LockFreeMemoryPool`, `FixedCapacityPool`, `AdaptiveFiveLevelPool`, `BumpArena`) with thread safety and RAII (`tests/memory_pool_contract.rs`)",
        "- **Blob Storage**: Specialized stores (`MemoryBlobStore`, `PlainBlobStore`, `SimpleZipBlobStore`, `MixedLenBlobStore`, `ZipOffsetBlobStore`, `NestLoudsTrieBlobStore`, `DictZipBlobStore`, `ZstdBlobStore`, `Lz4BlobStore`) with trie-based indexing and compression (`tests/format_spec_golden_tests.rs`)",
        f"- **Succinct Data Structures**: `Rank9` (Vigna 2008, `{r9_ovh:.2f}%` overhead), `RankSelectInterleaved256` (`{r9_ovh:.2f}%` rank-only / `{il_sel_ovh:.2f}%` with select caches), `RankSelectSE256`, `EliasFano` / `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano` with cursor `advance_to_index`, `HybridPostingList` (auto-select encoding), AMD-safe PDEP with `has_fast_bmi2` (`tests/memory_footprint_e6.rs`)",
        f"- **BM25 Scoring**: `FieldnormEncoder` (Lucene SmallFloat, `{E6_FIELDNORM_BYTES_PER_DOC}` fieldnorms) + `Bm25BatchScorer` (AVX2/AVX-512 SIMD batch, prefetch) (`tests/memory_footprint_e6.rs`, `benches/core_bench.rs`)",
        "- **Specialized Containers**: `FastVec`, `ValVec32`, `IntVec`, `UintVecMin0`, `ZipIntVec`, `FixedLenStrVec`, `SortableStrVec`, `ZoSortedStrVec`, `VecTrbSet`/`VecTrbMap`, `MinimalSso`, `LruMap`, `ConcurrentLruMap`, `FixedCircularQueue`",
        f"- **Hash Maps**: SwissTable 16-way SSE2 group-probing `ZiporaHashMap` (`{hm_u64_b:.2f} B/entry` on `1M` `u64`), `GoldHashMap` (`benches/competitor_bench.rs`)",
        f"- **Advanced Tries**: Double-Array (`DoubleArrayTrie`, `{dat_bps:.1f} B/state`; `DoubleArrayTrieMap<u32>`, `{E6_DAT_MAP_BYTES_PER_STATE}`), LOUDS, Critical-Bit (BMI2), Patricia tries, lazy prefix/fuzzy iterators, `CsppTrie` (Compressed Sparse Parallel Patricia, `{cspp_bpk:.2f} B/key` on shared-prefix keys), `ConcurrentCsppTrie` (multi-writer/multi-reader, epoch-based reclamation) (`tests/memory_footprint_e6.rs`)",
        "- **Compression & Integer Codecs**: PA-Zip, Huffman O0/O1/O2, FSE, rANS (64-bit X1/X2/X4/X8), `StreamVByte` (SSSE3), `GroupVarint` (SSSE3), ZSTD & LZ4 integration (`benches/competitor_bench.rs`)",
        "- **C FFI Support**: Complete C11/C++17 API (`--features ffi`, `tests/ffi_c_smoke.rs`)",
        END_KF_MARKER,
    ]
    return "\n".join(lines)


def generate_key_features_cn(
    core: dict[str, Any],
    comp: dict[str, Any],
) -> str:
    space = core["space_metrics"]
    pair_map = {
        (c["zipora_bench"], c["competitor_bench"]): c
        for c in comp["same_run_comparisons"]
    }
    r9_ovh = space["rank_select/rank9_overhead_pct"]
    il_sel_ovh = (
        pair_map[
            ("rank_select/interleaved256/rank1", "rank_select/sucds_rank9sel/rank1")
        ]["zipora_secondary"]
        - 100.0
    )
    hm_u64_b = pair_map[
        ("hash_maps/zipora_hash_map/get_u64", "hash_maps/std/get_u64")
    ]["zipora_secondary"]
    dat_bps = space["tries/dat_shared_prefix_bytes_per_state"]
    cspp_bpk = space["tries/cspp_shared_prefix_bytes_per_key"]

    lines = [
        BEGIN_KF_MARKER,
        "- **高性能**：零拷贝操作、SIMD 优化（AVX2、AVX-512）、缓存友好的内存布局、Block-Max WAND SIMD 游标原语",
        "- **内存安全**：所有生产环境 `unsafe` 位置均附有 `// SAFETY:` 或 `# Safety` 注释（`scripts/unsafe_audit.py`）",
        "- **安全内存管理**：线程安全的生产级内存池（`SecureMemoryPool`、`LockFreeMemoryPool`、`FixedCapacityPool`、`AdaptiveFiveLevelPool`、`BumpArena`），支持 RAII（`tests/memory_pool_contract.rs`）",
        "- **Blob 存储**：专用存储引擎（`MemoryBlobStore`、`PlainBlobStore`、`SimpleZipBlobStore`、`MixedLenBlobStore`、`ZipOffsetBlobStore`、`NestLoudsTrieBlobStore`、`DictZipBlobStore`、`ZstdBlobStore`、`Lz4BlobStore`），支持 Trie 索引和压缩（`tests/format_spec_golden_tests.rs`）",
        f"- **简洁数据结构**：`Rank9`（Vigna 2008，`{r9_ovh:.2f}%` 额外开销）、`RankSelectInterleaved256`（仅 rank `{r9_ovh:.2f}%` / 含 select 缓存 `{il_sel_ovh:.2f}%`）、`RankSelectSE256`、`EliasFano` / `PartitionedEliasFano` / `OptimalPartitionedEliasFano` / `ClusteredEliasFano`（支持游标 `advance_to_index`）、`HybridPostingList`（自动选择编码策略）、AMD 安全 PDEP（`has_fast_bmi2` 检测）（`tests/memory_footprint_e6.rs`）",
        f"- **BM25 评分**：`FieldnormEncoder`（Lucene SmallFloat，`{E6_FIELDNORM_BYTES_PER_DOC}` 字段长度） + `Bm25BatchScorer`（AVX2/AVX-512 SIMD 批量评分、预取）（`tests/memory_footprint_e6.rs`、`benches/core_bench.rs`）",
        "- **专用容器**：`FastVec`、`ValVec32`、`IntVec`、`UintVecMin0`、`ZipIntVec`、`FixedLenStrVec`、`SortableStrVec`、`ZoSortedStrVec`、`VecTrbSet`/`VecTrbMap`、`MinimalSso`、`LruMap`、`ConcurrentLruMap`、`FixedCircularQueue`",
        f"- **哈希表**：SwissTable 16 路 SSE2 组探测 `ZiporaHashMap`（`1M` `u64` 下 `{hm_u64_b:.2f} B/entry`）、`GoldHashMap`（`benches/competitor_bench.rs`）",
        f"- **高级 Trie**：双数组（`DoubleArrayTrie`，`{dat_bps:.1f} B/state`；`DoubleArrayTrieMap<u32>`，`{E6_DAT_MAP_BYTES_PER_STATE}`）、LOUDS、Critical-Bit（BMI2）、Patricia Trie、惰性前缀/模糊迭代器、`CsppTrie`（压缩稀疏并行 Patricia，共享前缀键 `{cspp_bpk:.2f} B/key`）、`ConcurrentCsppTrie`（多写多读，基于 epoch 的回收）（`tests/memory_footprint_e6.rs`）",
        "- **压缩与整数编码**：PA-Zip、Huffman O0/O1/O2、FSE、rANS（64 位 X1/X2/X4/X8）、`StreamVByte`（SSSE3）、`GroupVarint`（SSSE3）、ZSTD 与 LZ4 集成（`benches/competitor_bench.rs`）",
        "- **C FFI 支持**：完整的 C11/C++17 API（`--features ffi`，`tests/ffi_c_smoke.rs`）",
        END_KF_MARKER,
    ]
    return "\n".join(lines)


def generate_perf_section(
    core: dict[str, Any],
    comp: dict[str, Any],
    e2e: dict[str, Any],
) -> str:
    core_map = {b["name"]: b for b in core["benchmarks"]}
    pair_map = {
        (c["zipora_bench"], c["competitor_bench"]): c
        for c in comp["same_run_comparisons"]
    }

    leading_count = sum(
        1 for c in comp["same_run_comparisons"] if c["e3_verdict"] == "LEADING"
    )
    faster_primary_count = sum(
        1
        for c in comp["same_run_comparisons"]
        if c["primary_delta_pct"] <= 3.0
    )

    e2e_sum = e2e["summary"]
    z_qps = e2e_sum["zipora_qps_top10"]
    t_qps = e2e_sum["tantivy_qps_top10"]
    u_qps = e2e_sum["baseline_qps_top10"]
    if z_qps >= t_qps * 0.97:
        leading_count += 1
        faster_primary_count += 1

    total_pairs = len(comp["same_run_comparisons"]) + 1

    curated_pairs = [
        (
            "Rank/Select (`1 Gbit`)",
            "rank_select/interleaved256/rank1",
            "rank_select/sucds_rank9sel/rank1",
            "`RankSelectInterleaved256::rank1`",
            "`sucds::Rank9Sel`",
        ),
        (
            "Rank/Select (`1 Gbit`)",
            "rank_select/se256/select1",
            "rank_select/sucds_rank9sel/select1",
            "`RankSelectSE256::select1`",
            "`sucds::Rank9Sel`",
        ),
        (
            "Elias-Fano (`100k`)",
            "elias_fano/uniform/ef_next_geq",
            "elias_fano/uniform/sucds_ef_next_geq",
            "`EliasFano::next_geq` (uniform)",
            "`sucds::EliasFano`",
        ),
        (
            "Elias-Fano (`100k`)",
            "elias_fano/zipf_clustered/opef_next_geq",
            "elias_fano/zipf_clustered/vers_ef_next_geq",
            "`OptimalPartitionedEliasFano::next_geq`",
            "`vers-vecs::EliasFanoVec`",
        ),
        (
            "Elias-Fano (`100k`)",
            "elias_fano/dense/cef_select",
            "elias_fano/dense/sucds_ef_select",
            "`ClusteredEliasFano::get` (dense)",
            "`sucds::EliasFano`",
        ),
        (
            "Tries (`1M` keys)",
            "tries/dat/shared_prefix/lookup",
            "tries/qp_trie/shared_prefix/lookup",
            "`DoubleArrayTrie::contains` (shared prefix)",
            "`qp-trie::Trie`",
        ),
        (
            "Tries (`1M` keys)",
            "tries/cspp/shared_prefix/lookup",
            "tries/radix_trie/shared_prefix/lookup",
            "`CsppTrie::contains` (shared prefix)",
            "`radix_trie::Trie`",
        ),
        (
            "Hash Maps (`1M` keys)",
            "hash_maps/zipora_hash_map/get_str16",
            "hash_maps/hashbrown_ahash/get_str16",
            "`ZiporaHashMap<[u8; 16], u64>::get`",
            "`hashbrown::HashMap` (`ahash`)",
        ),
        (
            "Hash Maps (`1M` keys)",
            "hash_maps/zipora_hash_map/get_u64",
            "hash_maps/std/get_u64",
            "`ZiporaHashMap<u64, u64>::get`",
            "`std::collections::HashMap`",
        ),
        (
            "Entropy (`64 KiB`)",
            "entropy/huffman_o0/decode_64kib",
            "entropy/huffman_compress/decode_64kib",
            "`HuffmanDecoder::decode` (O0)",
            "`huffman-compress`",
        ),
        (
            "Entropy (`64 KiB`)",
            "entropy/rans64_x4/decode_64kib",
            "entropy/constriction_ans/decode_64kib",
            "`Rans64Decoder<ParallelX4>::decode`",
            "`constriction` ANS",
        ),
        (
            "Integer Codecs (`100k`)",
            "int_codecs/stream_vbyte/encode_deltas",
            "int_codecs/stream_vbyte_simd/encode_deltas",
            "`StreamVByte::encode_deltas` (SSSE3)",
            "`stream-vbyte` SIMD",
        ),
        (
            "Integer Codecs (`100k`)",
            "int_codecs/stream_vbyte/decode_into",
            "int_codecs/stream_vbyte_simd/decode_into",
            "`StreamVByte::decode_into` (SSSE3)",
            "`stream-vbyte` SIMD",
        ),
        (
            "Radix Sort (`10M`)",
            "radix_sort/lsd_u32",
            "radix_sort/voracious/u32",
            "`RadixSort::sort_u32` (hybrid MSB+LSD)",
            "`voracious_radix_sort`",
        ),
        (
            "String Search (`64 KiB`)",
            "search_and_scoring/simd_strchr_64kib",
            "search_and_scoring/memchr_strchr_64kib",
            "`simd_strchr` (128B AVX2)",
            "`memchr::memchr`",
        ),
        (
            "String Search (`64 KiB`)",
            "search_and_scoring/simd_strstr_64kib",
            "search_and_scoring/memmem_strstr_64kib",
            "`simd_strstr` (128B AVX2)",
            "`memchr::memmem`",
        ),
    ]

    host_meta = core["host_metadata"]
    lines = [
        BEGIN_MARKER,
        f"Measured on reference host `{host_meta['host']}` ({host_meta['cpu_model']}, "
        f"{host_meta['logical_cores']} logical cores, pinned core {host_meta['pinned_core']}, "
        f"`THP-off` via `PR_SET_THP_DISABLE`, `{host_meta['rustflags']}`).",
        f"Across all **{total_pairs}** head-to-head comparisons in [`docs/review/e2_e3.md`](docs/review/e2_e3.md) "
        f"and [`docs/review/e4.md`](docs/review/e4.md), Zipora is **`LEADING` on {leading_count}/{total_pairs}** "
        f"(primary within `-3.0%` and secondary space `<= +10.0%`) and **faster or tied on primary on "
        f"{faster_primary_count}/{total_pairs} ({faster_primary_count * 100.0 / total_pairs:.1f}%)**.",
        "",
        "### Head-to-Head Competitor Highlights (`benches/baselines/bindiego/competitor_bench.json` & `e2e_search.json`)",
        "",
        "| Family | Zipora Operation | Zipora (`min`) | Space / Secondary | Competitor | Competitor (`min`) | Speedup |",
        "|---|---|---:|---:|---|---:|---:|",
    ]

    for (
        family_label,
        z_key,
        c_key,
        z_label,
        c_label,
    ) in curated_pairs:
        pair = pair_map[(z_key, c_key)]
        assert pair["e3_verdict"] == "LEADING", (
            f"G7 violation: curated row {z_key} vs {c_key} is {pair['e3_verdict']}, not LEADING"
        )
        unit = pair["primary_unit"]
        z_min = pair["zipora_primary_min"]
        c_min = pair["competitor_primary_min"]
        speedup = pair["primary_speedup_ratio"]
        sec_str = format_secondary(pair["secondary_metric"], pair["zipora_secondary"])
        lines.append(
            f"| {family_label} | {z_label} | `{z_min:.2f} {unit}` | `{sec_str}` | "
            f"{c_label} | `{c_min:.2f} {unit}` | **`{speedup:.2f}x`** |"
        )

    qps_ratio_tantivy = z_qps / t_qps
    qps_ratio_uncomp = z_qps / u_qps
    lines.append(
        f"| Systemic Search (`100k` docs) | `Zipora BMW` top-10 (`p50 = {e2e_sum['zipora_p50_us']:.2f} us`) | "
        f"`{z_qps:,.1f} QPS` | `{e2e_sum['zipora_bytes_per_posting']:.3f} B/posting` (postings) | "
        f"`tantivy` v0.26.2 (`p50 = {e2e_sum['tantivy_p50_us']:.2f} us`) | `{t_qps:,.1f} QPS` | "
        f"**`{qps_ratio_tantivy:.2f}x`** (`{qps_ratio_uncomp:.2f}x` vs uncompressed) |"
    )

    bm25_avx2 = core_map["search_and_scoring/bm25_batch_score_simd"]
    fieldnorm_tbl = core_map["search_and_scoring/fieldnorm_build_score_table"]
    lf_1t = core_map["memory_pools/lockfree_pool/alloc_free_1t"]
    lf_4t = core_map["memory_pools/lockfree_pool/alloc_free_4t"]
    sa_con = core_map["suffix_array/construct"]
    sa_qry = core_map["suffix_array/search_query"]
    rank9_r1 = core_map["rank_select/rank9/rank1"]
    rank9_ovh = core["space_metrics"]["rank_select/rank9_overhead_pct"]

    lines.extend(
        [
            "",
            "### Additional Core Suite Baselines (`benches/baselines/bindiego/core_bench.json`)",
            "",
            f"- **BM25 & Fieldnorm Scoring**: `Bm25BatchScorer` AVX2 batch `{bm25_avx2['min_ns_per_op']:.4f} ns/doc`, `FieldnormEncoder::build_score_table` `{fieldnorm_tbl['min_ns_per_op']:.4f} ns/cell` (`{E6_FIELDNORM_BYTES_PER_DOC}` fieldnorms).",
            f"- **Memory Pools**: `LockFreeMemoryPool` 64B alloc+free `{lf_1t['min_ns_per_op']:.3f} ns/op` (1T) / `{lf_4t['min_ns_per_op']:.3f} ns/op` (4T).",
            f"- **Rank9 (`1 Gbit`)**: `Rank9::rank1` `{rank9_r1['min_ns_per_op']:.2f} ns/op` at `{rank9_ovh:.2f}%` space overhead.",
            f"- **Suffix Array (`10M` bytes, SA-IS)**: construction `{sa_con['min_ns_per_op']:.2f} ns/byte` (`9.05 B/byte`), `search_range` query `{sa_qry['min_ns_per_op']:.2f} ns/op`.",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def replace_between_markers(
    content: str,
    generated: str,
    begin_marker: str = BEGIN_MARKER,
    end_marker: str = END_MARKER,
) -> str:
    if begin_marker in content and end_marker in content:
        pattern = re.compile(
            re.escape(begin_marker) + r".*?" + re.escape(end_marker),
            re.DOTALL,
        )
        return pattern.sub(lambda _: generated, content, count=1)
    raise ValueError(f"Markers {begin_marker} .. {end_marker} not found")


def collect_crate_public_types() -> set[str]:
    pub_types: set[str] = set()
    decl_re = re.compile(
        r"\bpub(?:\s*\([^)]*\))?\s+(?:struct|enum|trait|type)\s+([A-Z][A-Za-z0-9_]*)"
    )
    reexport_re = re.compile(r"\bpub\s+use\s+[^;]+;")
    ident_re = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")
    for rs_path in (ROOT / "src").rglob("*.rs"):
        src = rs_path.read_text(encoding="utf-8")
        for m in decl_re.finditer(src):
            pub_types.add(m.group(1))
        for m in reexport_re.finditer(src):
            for id_m in ident_re.finditer(m.group(0)):
                pub_types.add(id_m.group(1))
    return pub_types


def strip_generated_and_code_blocks(text: str, has_key_features: bool = True) -> str:
    cleaned = text
    markers = [(BEGIN_MARKER, END_MARKER)]
    if has_key_features:
        markers.append((BEGIN_KF_MARKER, END_KF_MARKER))
    for b_m, e_m in markers:
        b_idx = cleaned.find(b_m)
        e_idx = cleaned.find(e_m)
        if b_idx != -1 and e_idx != -1 and e_idx >= b_idx:
            cleaned = cleaned[:b_idx] + cleaned[e_idx + len(e_m) :]
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    return cleaned


def verify_no_html_comments_or_outside_numbers(
    label: str, text: str, has_key_features: bool = True
) -> list[str]:
    errors: list[str] = []

    for m in re.finditer(r"<!--[\s\S]*?-->", text):
        comment = m.group(0).strip()
        if comment not in ALLOWED_HTML_COMMENTS:
            errors.append(
                f"{label}: unauthorized HTML comment `{comment}` (cannot hide prose/numbers in comments)"
            )

    outside = strip_generated_and_code_blocks(text, has_key_features=has_key_features)
    outside_no_urls = re.sub(r"\(https?://[^)]+\)", "()", outside)
    outside_no_urls = re.sub(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)", "", outside_no_urls)

    outside_patterns = [
        (
            re.compile(
                r"(?:\b\d+(?:\.\d+)?\s*(?:x|×)\b|(?:\b\d+(?:\.\d+)?)\s*×|快\s*\d+(?:\.\d+)?\s*倍|\d+(?:\.\d+)?\s*倍)",
                re.IGNORECASE,
            ),
            "multiplier/speedup claim outside generated block",
        ),
        (
            re.compile(
                r"(?:~?\d[\d,]*(?:\.\d+)?(?:–\d+(?:\.\d+)?|-\d+(?:\.\d+)?)?\s*"
                r"(?:ns/op|ns/byte|ns/int|ns/elem|ns/call|ns/doc|ns/cell|ns\b|"
                r"us\b|µs|μs|ms\b|"
                r"QPS|qps|Gops/s|Mops/s|Gwords/s|ops/sec|ops/s|"
                r"GB/s|GiB/s|MB/s|MiB/s|"
                r"B/entry|B/state|B/key|B/doc|B/posting|B/byte|B/elem|B\s+aux|"
                r"bits/elem|bits/B|bits/int|of\s+raw|%|"
                r"specialized\s+containers|containers\b|个专用容器|个容器))"
            ),
            "numeric metric/count outside generated block",
        ),
    ]

    for regex, desc in outside_patterns:
        for m in regex.finditer(outside_no_urls):
            errors.append(
                f"{label}: {desc} `{m.group(0)}` (all numeric claims must be inside generated blocks)"
            )

    return errors


def verify_public_types_in_doc(
    label: str, text: str, pub_types: set[str]
) -> list[str]:
    errors: list[str] = []
    without_fenced = re.sub(r"```[\s\S]*?```", "", text)
    pascal_re = re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9_]*)\b")
    for id_m in pascal_re.finditer(without_fenced):
        tname = id_m.group(1)
        if tname not in pub_types and tname not in EXTERNAL_ALLOWED_TYPES:
            errors.append(
                f"{label}: references non-existent public type `{tname}`"
            )
    return errors


def verify_files(
    readme_override: str | None = None,
    readme_cn_override: str | None = None,
    perf_md_override: str | None = None,
) -> list[str]:
    core, comp, e2e = load_baselines()
    expected_kf_en = generate_key_features_en(core, comp)
    expected_kf_cn = generate_key_features_cn(core, comp)
    expected_perf = generate_perf_section(core, comp, e2e)
    pub_types = collect_crate_public_types()

    errors: list[str] = []

    docs_to_check = [
        ("README.md", readme_override, expected_perf, expected_kf_en, True),
        ("README_cn.md", readme_cn_override, expected_perf, expected_kf_cn, True),
        ("docs/PERFORMANCE.md", perf_md_override, expected_perf, None, False),
    ]

    for rel_path, override_text, exp_perf, exp_kf, has_kf in docs_to_check:
        content = (
            override_text
            if override_text is not None
            else (ROOT / rel_path).read_text(encoding="utf-8")
        )

        if exp_perf not in content:
            errors.append(
                f"{rel_path}: Verified Performance table drifted from baselines (run `python3 scripts/perf_table.py --update`)"
            )

        if has_kf and exp_kf not in content:
            errors.append(
                f"{rel_path}: Key Features section drifted from baselines (run `python3 scripts/perf_table.py --update`)"
            )

        errors.extend(
            verify_no_html_comments_or_outside_numbers(
                rel_path, content, has_key_features=has_kf
            )
        )
        errors.extend(verify_public_types_in_doc(rel_path, content, pub_types))

    return errors


def audit_readme_claims(
    readme_text: str,
    _core: dict[str, Any] | None = None,
    _comp: dict[str, Any] | None = None,
    _e2e: dict[str, Any] | None = None,
) -> list[str]:
    return verify_files(readme_override=readme_text)


def run_mutation_self_test() -> list[str]:
    """Run all 30 adversarial mutations from s53/rev_infra/adv/mut.py against in-memory docs."""
    _, _, e2e = load_baselines()
    qps_str = f"{e2e['summary']['zipora_qps_top10']:,.1f} QPS"
    orig_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    orig_cn = (ROOT / "README_cn.md").read_text(encoding="utf-8")
    orig_perf = (ROOT / "docs/PERFORMANCE.md").read_text(encoding="utf-8")

    base_errs = verify_files(orig_readme, orig_cn, orig_perf)
    if base_errs:
        return [f"unmutated_baseline_failed: {base_errs}"]

    cases = [
        ("table_qps", "README.md", orig_readme.replace(qps_str, "99,999.9 QPS")),
        ("prose_bentry", "README.md", orig_readme.replace("35.65 B/entry", "12.34 B/entry")),
        ("gops_readme", "README.md", orig_readme + "\nAchieves 5.2 Gops/s on rank.\n"),
        ("gops_cn", "README_cn.md", orig_cn + "\n吞吐量达到 5.2 Gops/s。\n"),
        ("gops_perf", "docs/PERFORMANCE.md", orig_perf + "\nAchieves 5.2 Gops/s.\n"),
        ("x26_readme", "README.md", orig_readme + "\nUp to 26x faster.\n"),
        ("x26_cn", "README_cn.md", orig_cn + "\n性能提升 26x。\n"),
        ("x26_perf", "docs/PERFORMANCE.md", orig_perf + "\nUp to 26x faster.\n"),
        ("x26_unicode", "README.md", orig_readme + "\nUp to 26× faster than baseline.\n"),
        ("x_vs", "README.md", orig_readme + "\n Achieves 3.8x vs std::collections::HashMap.\n"),
        ("x_speedup", "README.md", orig_readme + "\n Delivers a 2.4x speedup on lookups.\n"),
        ("x_higher_qps", "README.md", orig_readme + "\n Delivers 5.5x higher QPS in production.\n"),
        ("us_ms", "README.md", orig_readme + "\nQuery latency is 173-188 µs (down from 5.13 ms).\n"),
        ("ns_bare", "README.md", orig_readme + "\nLookup takes only 12 ns on average.\n"),
        ("gbs", "README.md", orig_readme + "\nDecoding at 4.5 GB/s.\n"),
        ("gwords_spaced", "README.md", orig_readme + "\nThroughput of 5.2 Gwords/s.\n"),
        ("allowed_100pct", "README.md", orig_readme + "\n achieves 100.0% compression ratio.\n"),
        ("allowed_bdoc", "README.md", orig_readme + "\n full index uses only 1.0 B/doc total.\n"),
        ("unitless_containers", "README.md", orig_readme + "\nProvides 47 specialized containers.\n"),
        ("html_comment", "README.md", orig_readme + "\n<!-- 999999 QPS 100x faster -->\n"),
        (
            "html_comment_table",
            "README.md",
            orig_readme.replace(
                "### Additional Core Suite Baselines",
                "<!-- 999x -->\n### Additional Core Suite Baselines",
            ),
        ),
        (
            "foobar_kf",
            "README.md",
            orig_readme.replace("`ZiporaHashMap`", "`ZiporaHashMap`, `FooBarMap`", 1),
        ),
        (
            "foobar_kf_path",
            "README.md",
            orig_readme.replace("`ZiporaHashMap`", "`ZiporaHashMap`, `zipora::FooBarMap`", 1),
        ),
        (
            "foobar_kf_method",
            "README.md",
            orig_readme.replace("`ZiporaHashMap`", "`ZiporaHashMap`, `FooBarMap::new`", 1),
        ),
        (
            "foobar_kf_bare",
            "README.md",
            orig_readme.replace("`ZiporaHashMap`", "`ZiporaHashMap`, FooBarMap", 1),
        ),
        (
            "foobar_cn",
            "README_cn.md",
            orig_cn.replace("`ZiporaHashMap`", "`ZiporaHashMap`、`FooBarMap`", 1),
        ),
        (
            "foobar_perf",
            "docs/PERFORMANCE.md",
            orig_perf.replace("`Rank9::rank1`", "`FooBarMap::rank1`", 1),
        ),
        ("foobar_readme_other_section", "README.md", orig_readme + "\nUses `FooBarMap` internally.\n"),
        ("cn_bei", "README_cn.md", orig_cn + "\n比标准库快 26 倍。\n"),
        ("cn_bei_nospace", "README_cn.md", orig_cn + "\n比标准库快26倍。\n"),
    ]

    uncaught: list[str] = []
    for name, target_file, mutated_text in cases:
        r_en = mutated_text if target_file == "README.md" else orig_readme
        r_cn = mutated_text if target_file == "README_cn.md" else orig_cn
        r_pf = mutated_text if target_file == "docs/PERFORMANCE.md" else orig_perf
        errs = verify_files(r_en, r_cn, r_pf)
        if not errs:
            uncaught.append(name)
    return uncaught


def update_files() -> None:
    core, comp, e2e = load_baselines()
    kf_en = generate_key_features_en(core, comp)
    kf_cn = generate_key_features_cn(core, comp)
    perf_block = generate_perf_section(core, comp, e2e)

    for rel_path, kf_block in [
        ("README.md", kf_en),
        ("README_cn.md", kf_cn),
        ("docs/PERFORMANCE.md", None),
        ("CLAUDE.md", None),
        ("GEMINI.md", None),
    ]:
        p = ROOT / rel_path
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        if kf_block is not None:
            txt = replace_between_markers(
                txt, kf_block, BEGIN_KF_MARKER, END_KF_MARKER
            )
        if BEGIN_MARKER in txt and END_MARKER in txt:
            txt = replace_between_markers(
                txt, perf_block, BEGIN_MARKER, END_MARKER
            )
        p.write_text(txt, encoding="utf-8")
        print(f"Updated {rel_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.update:
        update_files()

    errors = verify_files()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    uncaught = run_mutation_self_test()
    if uncaught:
        print(
            f"ERROR: mutation self-test failed to catch: {uncaught}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(
            "=== Performance Table & Documentation Parity Check (Rule E7 / F2 / S52-R5): PASS ==="
        )
    else:
        core, comp, e2e = load_baselines()
        print(generate_perf_section(core, comp, e2e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
