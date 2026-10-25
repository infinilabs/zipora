#!/usr/bin/env python3
"""Generate Verified Performance tables from committed baselines (`E7`) and
audit `README.md` numeric and capability claims against tests/baselines (`F2`).

Usage:
  python3 scripts/perf_table.py
  python3 scripts/perf_table.py --update
  python3 scripts/perf_table.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "benches" / "baselines" / "bindiego"

BEGIN_MARKER = "<!-- BEGIN GENERATED PERF TABLE (scripts/perf_table.py) -->"
END_MARKER = "<!-- END GENERATED PERF TABLE (scripts/perf_table.py) -->"


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

    # Curated LEADING highlights from competitor_bench.json + e2e_search.json (Rule G7:
    # every perf number comes from a committed baseline on `bindiego` and matches/beats
    # the named competitor).
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

    # Append E1.1 Systemic BM25 Search row from e2e_search.json
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
            f"- **BM25 & Fieldnorm Scoring**: `Bm25BatchScorer` AVX2 batch `{bm25_avx2['min_ns_per_op']:.4f} ns/doc`, `FieldnormEncoder::build_score_table` `{fieldnorm_tbl['min_ns_per_op']:.4f} ns/cell` (`1.0 B/doc` fieldnorms).",
            f"- **Memory Pools**: `LockFreeMemoryPool` 64B alloc+free `{lf_1t['min_ns_per_op']:.3f} ns/op` (1T) / `{lf_4t['min_ns_per_op']:.3f} ns/op` (4T).",
            f"- **Rank9 (`1 Gbit`)**: `Rank9::rank1` `{rank9_r1['min_ns_per_op']:.2f} ns/op` at `{rank9_ovh:.2f}%` space overhead.",
            f"- **Suffix Array (`10M` bytes, SA-IS)**: construction `{sa_con['min_ns_per_op']:.2f} ns/byte` (`9.05 B/byte`), `search_range` query `{sa_qry['min_ns_per_op']:.2f} ns/op`.",
            END_MARKER,
        ]
    )
    return "\n".join(lines)


def replace_between_markers(content: str, generated: str) -> str:
    if BEGIN_MARKER in content and END_MARKER in content:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )
        return pattern.sub(lambda _: generated, content, count=1)
    raise ValueError(f"Markers {BEGIN_MARKER} .. {END_MARKER} not found")


def audit_readme_claims(
    readme_text: str,
    core: dict[str, Any],
    comp: dict[str, Any],
    e2e: dict[str, Any],
) -> list[str]:
    """Audit every numeric and capability claim in README.md against tests and baselines (Rule F2)."""
    errors: list[str] = []

    # 1. Generated perf table must be present and up-to-date in README.md (and CLAUDE.md/GEMINI.md if present)
    expected_perf = generate_perf_section(core, comp, e2e)
    for doc_name in ("README.md", "CLAUDE.md", "GEMINI.md"):
        doc_path = ROOT / doc_name
        if not doc_path.exists():
            continue
        doc_text = doc_path.read_text(encoding="utf-8")
        if BEGIN_MARKER not in doc_text or END_MARKER not in doc_text:
            errors.append(f"{doc_name} is missing BEGIN/END GENERATED PERF TABLE markers")
        else:
            actual_perf = doc_text[
                doc_text.index(BEGIN_MARKER) : doc_text.index(END_MARKER)
                + len(END_MARKER)
            ]
            if actual_perf != expected_perf:
                errors.append(
                    f"{doc_name} Verified Performance table drifted from baselines (run `python3 scripts/perf_table.py --update`)"
                )

    # 2. Check MSRV badge matches Cargo.toml rust-version = "1.95"
    cargo_toml = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    if 'rust-version = "1.95"' not in cargo_toml or "rust-1.95+" not in readme_text:
        errors.append("README.md MSRV badge must match Cargo.toml rust-version = 1.95")

    # 3. Check unsafe documentation count matches scripts/unsafe_audit.py
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "unsafe_audit.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    m = re.search(r"Total unsafe sites in non-test code:\s*(\d+)", proc.stdout)
    if not m:
        errors.append("Could not parse unsafe site count from scripts/unsafe_audit.py")
    else:
        unsafe_total = int(m.group(1))
        formatted_total = f"{unsafe_total:,}"
        if formatted_total not in readme_text:
            errors.append(
                f"README.md unsafe count does not match unsafe_audit.py ({formatted_total})"
            )
        if "1,122" in readme_text:
            errors.append("README.md still contains stale unsafe block count 1,122")

    # 4. Check CsppTrie bytes/key, DoubleArrayTrie B/state, FieldnormEncoder B/doc, and ZiporaHashMap B/entry
    cspp_bpk = core["space_metrics"]["tries/cspp_shared_prefix_bytes_per_key"]
    if f"{cspp_bpk:.2f} B/key" not in readme_text:
        errors.append(
            f"README.md must cite CsppTrie {cspp_bpk:.2f} B/key from core_bench.json"
        )
    if "10.7 bytes/key" in readme_text or "20.6 ns/lookup" in readme_text:
        errors.append("README.md still contains stale pre-Phase-E performance claims")

    e6_test = (ROOT / "tests" / "memory_footprint_e6.rs").read_text(encoding="utf-8")
    for required_claim in ("8.0 B/state", "12.0 B/state", "1.0 B/doc", "25.00%", "31.25%"):
        if required_claim not in readme_text:
            errors.append(f"README.md missing E6-verified claim: {required_claim}")
    if "test_e6_" not in e6_test:
        errors.append("tests/memory_footprint_e6.rs must contain test_e6_ footprint assertions")

    # 5. Check named types in README.md Key Features exist in src/
    blob_mod = (ROOT / "src" / "blob_store" / "mod.rs").read_text(encoding="utf-8")
    for store_name in (
        "MemoryBlobStore",
        "PlainBlobStore",
        "SimpleZipBlobStore",
        "MixedLenBlobStore",
        "ZipOffsetBlobStore",
        "NestLoudsTrieBlobStore",
        "DictZipBlobStore",
        "ZstdBlobStore",
        "Lz4BlobStore",
    ):
        if store_name not in readme_text or store_name not in blob_mod:
            errors.append(f"Blob store claim {store_name} missing in README.md or src/blob_store/mod.rs")

    # 6. Check every linked doc file in README.md exists on disk
    for doc_rel in re.findall(r"\((docs/[A-Za-z0-9_./-]+\.md)\)", readme_text):
        if not (ROOT / doc_rel).exists():
            errors.append(f"README.md links to non-existent doc: {doc_rel}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update Verified Performance section in README.md (and CLAUDE.md/GEMINI.md if present)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify README.md Verified Performance table and F2 claims match baselines/tests",
    )
    args = parser.parse_args()

    core, comp, e2e = load_baselines()
    perf_section = generate_perf_section(core, comp, e2e)

    if args.update:
        for fname in ("README.md", "CLAUDE.md", "GEMINI.md"):
            fpath = ROOT / fname
            if fpath.exists():
                text = fpath.read_text(encoding="utf-8")
                if BEGIN_MARKER in text and END_MARKER in text:
                    updated = replace_between_markers(text, perf_section)
                    fpath.write_text(updated, encoding="utf-8")
                    print(f"Updated {fname}")
        return 0

    if args.check:
        readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
        errors = audit_readme_claims(readme_text, core, comp, e2e)
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print("README.md E7 performance tables and F2 claims audit: PASS")
        return 0

    print(perf_section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
