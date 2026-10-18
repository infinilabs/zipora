// Competitor configurations and same-run E3 comparison table construction.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::common::{BenchRecord, HostMetadata};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorConfigMeta {
    pub crate_name: String,
    pub version: String,
    pub features: Vec<String>,
    pub api_used: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SameRunComparison {
    pub category: String,
    pub operation: String,
    pub zipora_bench: String,
    pub zipora_variant: String,
    pub competitor_bench: String,
    pub competitor_name: String,
    pub primary_unit: String,
    pub zipora_primary_min: f64,
    pub competitor_primary_min: f64,
    pub primary_speedup_ratio: f64,
    pub primary_delta_pct: f64,
    pub secondary_metric: String,
    pub zipora_secondary: f64,
    pub competitor_secondary: f64,
    pub secondary_delta_pct: f64,
    pub e3_verdict: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorBenchReport {
    pub suite: String,
    pub tier: String,
    pub seed: u64,
    pub entropy_block_source: String,
    pub host_metadata: HostMetadata,
    pub competitor_configs: BTreeMap<String, CompetitorConfigMeta>,
    pub dataset_sizes: BTreeMap<String, u64>,
    pub space_metrics: BTreeMap<String, f64>,
    pub same_run_comparisons: Vec<SameRunComparison>,
    pub benchmarks: Vec<BenchRecord>,
}

type CompetitorSpec = (
    &'static str,
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static str,
    &'static str,
);

const COMPETITOR_SPECS: &[CompetitorSpec] = &[
    (
        "sucds",
        "sucds",
        "0.10.0",
        &["std", "intrinsics"],
        "bit_vectors::Rank9Sel::new(bv).select1_hints().select0_hints() & mii_sequences::EliasFanoBuilder::build().enable_rank()",
        "Broadword bit operations accelerated via hardware intrinsics feature",
    ),
    (
        "vers-vecs",
        "vers-vecs",
        "1.10.2",
        &["simd"],
        "RsVec::from_bit_vec(BitVec::from_vec) & EliasFanoVec::from_slice",
        "Hardware POPCNT/PDEP/TZCNT + SIMD feature enabled",
    ),
    (
        "bitm",
        "bitm",
        "0.5.2",
        &[],
        "RankSelect101111::<CombinedSampling, CombinedSampling>::build(Box<[u64]>)",
        "3-level interleaved rank index (3.125% rank overhead + CombinedSampling select)",
    ),
    (
        "fst",
        "fst",
        "0.4.7",
        &["default"],
        "fst::Set::from_iter(sorted_keys) & Set::contains(&[u8])",
        "Finite-state transducer minimal acyclic deterministic automaton over sorted keys",
    ),
    (
        "qp-trie",
        "qp-trie",
        "0.8.2",
        &["default"],
        "qp_trie::Trie::<&[u8], ()>::insert & Trie::contains_key(&[u8])",
        "Quad-bit popcount patricia trie borrowing key slices (&[u8]); bytes/key includes node heap + key backing bytes",
    ),
    (
        "radix_trie",
        "radix_trie",
        "0.3.0",
        &[],
        "radix_trie::Trie::<&[u8], ()>::insert & Trie::get(&[u8])",
        "16-way nibble-compressed radix tree with boxed child pointers and Nibblet vectors; bytes/key includes node heap + key backing bytes",
    ),
    (
        "hashbrown+ahash",
        "hashbrown",
        "0.16.1",
        &["inline-more", "raw-entry", "equivalent"],
        "hashbrown::HashMap::<K, u64, ahash::RandomState>::with_capacity_and_hasher & insert / get",
        "Google SwissTable (AVX2 16-way control byte group probing) with ahash v0.8.12",
    ),
    (
        "std::collections::HashMap",
        "std",
        "1.97.1",
        &["SipHash13"],
        "std::collections::HashMap::<K, u64>::with_capacity & insert / get",
        "Standard library SwissTable (hashbrown backend) with default SipHash-1-3 RandomState",
    ),
    (
        "mimalloc",
        "mimalloc",
        "0.1.52",
        &[],
        "<mimalloc::MiMalloc as GlobalAlloc>::alloc / dealloc",
        "Microsoft mimalloc thread-local free-list allocator invoked directly via GlobalAlloc",
    ),
    (
        "tikv-jemallocator",
        "tikv-jemallocator",
        "0.6.1",
        &["default"],
        "<tikv_jemallocator::Jemalloc as GlobalAlloc>::alloc / dealloc",
        "jemalloc thread-cached arena allocator invoked directly via GlobalAlloc",
    ),
    (
        "std::alloc::System",
        "std",
        "1.97.1",
        &["glibc_ptmalloc"],
        "<std::alloc::System as GlobalAlloc>::alloc / dealloc",
        "OS system allocator (glibc ptmalloc2 with per-thread tcache)",
    ),
    (
        "zstd",
        "zstd",
        "0.13.3",
        &["default"],
        "zstd::bulk::Compressor::new(level).compress(&[u8]) & zstd::bulk::Decompressor::new().decompress(&[u8], capacity)",
        "Whole-block LZ77 + FSE + 4-stream Huffman compressor at levels 1 and 3 with reused ZSTD_CCtx / ZSTD_DCtx (not an order-0 coder; raw huff0/FSE stages are benchmarked separately under zstd_huf0 and zstd_fse)",
    ),
    (
        "zstd_huf0",
        "zstd-sys",
        "2.0.15+zstd.1.5.7",
        &["bmi2"],
        "HUF_compress4X_repeat & HUF_decompress4X_hufOnly_wksp (HUF_flags_bmi2 = 1)",
        "Raw libzstd huff0 4-stream Order-0 Huffman entropy stage (table build + header + 4X stream encode/decode, no LZ77)",
    ),
    (
        "zstd_fse",
        "zstd-sys",
        "2.0.15+zstd.1.5.7",
        &["bmi2"],
        "HIST_count_wksp + FSE_normalizeCount + FSE_writeNCount + FSE_buildCTable_wksp + FSE_compress_usingCTable & FSE_decompress_wksp_bmi2",
        "Raw libzstd FSE (Finite State Entropy / tANS) Order-0 entropy stage (table build + header + stream encode/decode, no LZ77)",
    ),
    (
        "constriction",
        "constriction",
        "0.5.0",
        &["std"],
        "DefaultAnsCoder (AnsCoder<u32, u64>) + SmallContiguousCategoricalEntropyModel (PRECISION=12) + SmallContiguousLookupDecoderModel (PRECISION=12)",
        "64-bit state / 32-bit word rANS coder with 12-bit (4096-entry) O(1) lookup decoder table",
    ),
    (
        "huffman-compress",
        "huffman-compress",
        "0.7.0",
        &[],
        "CodeBuilder::from_iter(weights).finish() -> Book::<u8>::encode(&mut BitVec, &u8) & Tree::<u8>::decoder(&BitVec, len)",
        "Static Order-0 Huffman codebook (BTreeMap<u8, BitVec>) + arena binary tree decoder over bit-vec v0.8.0",
    ),
    (
        "stream-vbyte",
        "stream-vbyte",
        "0.4.1",
        &[],
        "stream_vbyte::encode::encode::<Scalar> & stream_vbyte::decode::decode::<Scalar>",
        "stream-vbyte v0.4.1 Scalar backend + delta pass (baseline reference; see stream-vbyte-simd for SSSE3/SSE4.1)",
    ),
    (
        "stream-vbyte-simd",
        "stream-vbyte",
        "0.4.1",
        &["ssse3", "sse4.1"],
        "stream_vbyte::encode::encode::<SvbCrateSimd> & stream_vbyte::decode::decode::<SvbCrateSimd>",
        "stream-vbyte v0.4.1 Encoder/Decoder trait implementation using stable-Rust SSE4.1/SSSE3 intrinsics matching upstream x86::Sse41 / x86::Ssse3",
    ),
    (
        "bitpacking",
        "bitpacking",
        "0.9.3",
        &["bitpacker1x", "bitpacker4x", "bitpacker8x"],
        "bitpacking::BitPacker4x::{num_bits_sorted, compress_sorted, decompress, decompress_sorted}",
        "Quickwit/Tantivy SIMD bit-packing (128-int blocks, SSE3 4-lane BitPacker4x with SIMD delta integration)",
    ),
    (
        "voracious_radix_sort",
        "voracious_radix_sort",
        "1.2.0",
        &[],
        "<[T] as voracious_radix_sort::RadixSort>::voracious_sort",
        "Single-threaded hybrid MSD/Ska/LSD radix sort over u32, u64, and Radixable<u64> (u64, u64) pairs",
    ),
    (
        "rdst",
        "rdst",
        "0.20.11",
        &[],
        "slice.radix_sort_builder().with_single_threaded_tuner().sort()",
        "Single-threaded adaptive LSB/Ska radix sort with RadixKey for u32, u64, and (u64, u64) pairs",
    ),
    (
        "std::slice::sort_unstable",
        "std",
        "1.97.1",
        &["ipnsort"],
        "<[T]>::sort_unstable & <[T]>::sort_unstable_by_key",
        "Standard library in-place pattern-defeating quicksort (ipnsort)",
    ),
    (
        "suffix",
        "suffix",
        "1.3.0",
        &[],
        "suffix::SuffixTable::new(&str) & SuffixTable::positions(&str)",
        "BurntSushi pure-Rust SA-IS suffix array construction and binary search",
    ),
    (
        "divsufsort",
        "divsufsort",
        "2.0.0",
        &[],
        "divsufsort::sort(&[u8]) & sacabase::StringIndex::longest_substring_match(&[u8])",
        "Pure-Rust port of Yuta Mori's libdivsufsort induced-sorting suffix array construction",
    ),
    (
        "cdivsufsort",
        "cdivsufsort",
        "2.0.0",
        &[],
        "cdivsufsort::sort(&[u8]) & sacabase::StringIndex::longest_substring_match(&[u8])",
        "C FFI binding to Yuta Mori's libdivsufsort reference implementation",
    ),
    (
        "memchr",
        "memchr",
        "2.7.5",
        &["std"],
        "memchr::memchr(u8, &[u8]), memchr::memmem::find(&[u8], &[u8]), & memchr::memmem::Finder::find(&[u8])",
        "BurntSushi AVX2/SSE2 vectorized byte and substring search",
    ),
    (
        "aho-corasick",
        "aho-corasick",
        "1.1.3",
        &["std", "perf-literal"],
        "aho_corasick::AhoCorasick::new([needle]).unwrap().find(&[u8])",
        "BurntSushi Aho-Corasick automaton with SIMD prefilter",
    ),
];

pub fn build_competitor_configs() -> BTreeMap<String, CompetitorConfigMeta> {
    let mut m = BTreeMap::new();
    for &(key, crate_name, version, features, api_used, notes) in COMPETITOR_SPECS {
        m.insert(
            key.to_string(),
            CompetitorConfigMeta {
                crate_name: crate_name.to_string(),
                version: version.to_string(),
                features: features.iter().map(|s| (*s).to_string()).collect(),
                api_used: api_used.to_string(),
                notes: notes.to_string(),
            },
        );
    }
    m
}

fn find_min(records: &[BenchRecord], name: &str) -> Option<f64> {
    records
        .iter()
        .find(|r| r.name == name)
        .map(|r| r.min_ns_per_op)
}

#[allow(clippy::too_many_arguments)]
fn push_comparison(
    out: &mut Vec<SameRunComparison>,
    records: &[BenchRecord],
    category: &str,
    operation: &str,
    zipora_bench: &str,
    zipora_variant: &str,
    competitor_bench: &str,
    competitor_name: &str,
    primary_unit: &str,
    secondary_metric: &str,
    zipora_secondary: f64,
    competitor_secondary: f64,
) {
    let (Some(z_min), Some(c_min)) = (
        find_min(records, zipora_bench),
        find_min(records, competitor_bench),
    ) else {
        return;
    };
    let primary_speedup_ratio = if z_min > 0.0 { c_min / z_min } else { 1.0 };
    let primary_delta_pct = if c_min > 0.0 {
        (z_min - c_min) / c_min * 100.0
    } else {
        0.0
    };
    let secondary_delta_pct = if competitor_secondary > 0.0 {
        (zipora_secondary - competitor_secondary) / competitor_secondary * 100.0
    } else {
        0.0
    };
    let e3_verdict = if primary_delta_pct <= 3.0 && secondary_delta_pct <= 10.0 {
        "LEADING"
    } else {
        "GAP"
    };
    out.push(SameRunComparison {
        category: category.to_string(),
        operation: operation.to_string(),
        zipora_bench: zipora_bench.to_string(),
        zipora_variant: zipora_variant.to_string(),
        competitor_bench: competitor_bench.to_string(),
        competitor_name: competitor_name.to_string(),
        primary_unit: primary_unit.to_string(),
        zipora_primary_min: (z_min * 1e4).round() / 1e4,
        competitor_primary_min: (c_min * 1e4).round() / 1e4,
        primary_speedup_ratio: (primary_speedup_ratio * 1e4).round() / 1e4,
        primary_delta_pct: (primary_delta_pct * 1e2).round() / 1e2,
        secondary_metric: secondary_metric.to_string(),
        zipora_secondary: (zipora_secondary * 1e4).round() / 1e4,
        competitor_secondary: (competitor_secondary * 1e4).round() / 1e4,
        secondary_delta_pct: (secondary_delta_pct * 1e2).round() / 1e2,
        e3_verdict: e3_verdict.to_string(),
    });
}

pub fn build_same_run_comparisons(
    records: &[BenchRecord],
    space: &BTreeMap<String, f64>,
) -> Vec<SameRunComparison> {
    let mut out = Vec::new();
    let sp = |key: &str, default: f64| space.get(key).copied().unwrap_or(default);

    // 1. Rank/Select
    let z_se_total_pct = 100.0 + sp("rank_select/se256_overhead_pct", 0.0);
    let z_rsi_total_pct = 100.0 + sp("rank_select/interleaved256_overhead_pct", 0.0);
    let sucds_total_pct = 100.0 + sp("rank_select/sucds_rank9sel_overhead_pct", 0.0);
    let vers_total_pct = 100.0 + sp("rank_select/vers_vecs_rsvec_overhead_pct", 0.0);
    let bitm_total_pct = 100.0 + sp("rank_select/bitm_rs101111_overhead_pct", 0.0);

    for (comp_prefix, comp_label, comp_space) in [
        ("rank_select/sucds_rank9sel", "sucds::Rank9Sel", sucds_total_pct),
        ("rank_select/vers_vecs_rsvec", "vers_vecs::RsVec", vers_total_pct),
        ("rank_select/bitm_rs101111", "bitm::RankSelect101111", bitm_total_pct),
    ] {
        for (op, z_bench, z_var, z_sp, c_op) in [
            ("rank1 (1 Gbit)", "rank_select/interleaved256/rank1", "RankSelectInterleaved256", z_rsi_total_pct, "rank1"),
            ("rank1 (1 Gbit, SE256)", "rank_select/se256/rank1", "RankSelectSE256", z_se_total_pct, "rank1"),
            ("select1 (1 Gbit)", "rank_select/se256/select1", "RankSelectSE256", z_se_total_pct, "select1"),
            ("select0 (1 Gbit)", "rank_select/se256/select0", "RankSelectSE256", z_se_total_pct, "select0"),
        ] {
            push_comparison(&mut out, records, "rank_select", op, z_bench, z_var, &format!("{comp_prefix}/{c_op}"), comp_label, "ns/op", "total_space_pct_of_raw", z_sp, comp_space);
        }
    }

    // 2. Elias-Fano (100k elements across uniform, zipf_clustered, dense)
    for dist_name in ["uniform", "zipf_clustered", "dense"] {
        let z_ef_bpe = sp(&format!("elias_fano/{dist_name}/ef_bits_per_elem"), 0.0);
        let z_opef_bpe = sp(&format!("elias_fano/{dist_name}/opef_bits_per_elem"), 0.0);
        let z_cef_bpe = sp(&format!("elias_fano/{dist_name}/cef_bits_per_elem"), 0.0);
        let sucds_bpe = sp(&format!("elias_fano/{dist_name}/sucds_ef_bits_per_elem"), 0.0);
        let vers_bpe = sp(&format!("elias_fano/{dist_name}/vers_ef_bits_per_elem"), 0.0);

        let (best_z_bench, best_z_var, best_z_bpe) = match dist_name {
            "uniform" => (format!("elias_fano/{dist_name}/ef_next_geq"), "EliasFano", z_ef_bpe),
            "zipf_clustered" => (format!("elias_fano/{dist_name}/opef_next_geq"), "OptimalPartitionedEliasFano", z_opef_bpe),
            _ => (format!("elias_fano/{dist_name}/cef_next_geq"), "ClusteredEliasFano", z_cef_bpe),
        };

        for (comp_suffix, comp_label, comp_bpe) in [
            ("sucds_ef", "sucds::EliasFano", sucds_bpe),
            ("vers_ef", "vers_vecs::EliasFanoVec", vers_bpe),
        ] {
            push_comparison(&mut out, records, "elias_fano", &format!("next_geq ({dist_name}, EF)"), &format!("elias_fano/{dist_name}/ef_next_geq"), "EliasFano", &format!("elias_fano/{dist_name}/{comp_suffix}_next_geq"), comp_label, "ns/op", "bits_per_elem", z_ef_bpe, comp_bpe);
            if best_z_var != "EliasFano" {
                push_comparison(&mut out, records, "elias_fano", &format!("next_geq ({dist_name}, best variant)"), &best_z_bench, best_z_var, &format!("elias_fano/{dist_name}/{comp_suffix}_next_geq"), comp_label, "ns/op", "bits_per_elem", best_z_bpe, comp_bpe);
            }
            for (op_title, z_suf, z_var, z_bpe) in [
                (format!("select / get ({dist_name}, EF)"), "ef_select", "EliasFano", z_ef_bpe),
                (format!("select / get ({dist_name}, CEF)"), "cef_select", "ClusteredEliasFano", z_cef_bpe),
                (format!("advance_to_index ({dist_name}, OPEF)"), "opef_cursor_advance_to_index", "OptimalPartitionedEliasFano", z_opef_bpe),
            ] {
                push_comparison(&mut out, records, "elias_fano", &op_title, &format!("elias_fano/{dist_name}/{z_suf}"), z_var, &format!("elias_fano/{dist_name}/{comp_suffix}_select"), comp_label, "ns/op", "bits_per_elem", z_bpe, comp_bpe);
            }
        }
    }

    // 3. Tries (1,000,000 keys across url_paths, random_16b, shared_prefix)
    for dist_name in ["url_paths", "random_16b", "shared_prefix"] {
        let dat_bpk = sp(&format!("tries/dat_{dist_name}_bytes_per_key"), 0.0);
        let cspp_bpk = sp(&format!("tries/cspp_{dist_name}_bytes_per_key"), 0.0);
        let fst_bpk = sp(&format!("tries/fst_{dist_name}_bytes_per_key"), 0.0);
        let qp_bpk = sp(&format!("tries/qp_trie_{dist_name}_bytes_per_key"), 0.0);
        let radix_bpk = sp(&format!("tries/radix_trie_{dist_name}_bytes_per_key"), 0.0);

        for (comp_build, comp_lookup, comp_label, comp_bpk) in [
            (format!("tries/fst/{dist_name}/build_sorted"), format!("tries/fst/{dist_name}/lookup"), "fst::Set", fst_bpk),
            (format!("tries/qp_trie/{dist_name}/build"), format!("tries/qp_trie/{dist_name}/lookup"), "qp_trie::Trie", qp_bpk),
            (format!("tries/radix_trie/{dist_name}/build"), format!("tries/radix_trie/{dist_name}/lookup"), "radix_trie::Trie", radix_bpk),
        ] {
            push_comparison(&mut out, records, "tries", &format!("build ({dist_name}, DAT)"), &format!("tries/dat/{dist_name}/build_sorted"), "DoubleArrayTrie", &comp_build, comp_label, "ns/key", "bytes_per_key", dat_bpk, comp_bpk);
            push_comparison(&mut out, records, "tries", &format!("build ({dist_name}, CSPP)"), &format!("tries/cspp/{dist_name}/build"), "CsppTrie", &comp_build, comp_label, "ns/key", "bytes_per_key", cspp_bpk, comp_bpk);
            push_comparison(&mut out, records, "tries", &format!("lookup ({dist_name}, DAT)"), &format!("tries/dat/{dist_name}/lookup"), "DoubleArrayTrie", &comp_lookup, comp_label, "ns/op", "bytes_per_key", dat_bpk, comp_bpk);
            push_comparison(&mut out, records, "tries", &format!("lookup ({dist_name}, CSPP)"), &format!("tries/cspp/{dist_name}/lookup"), "CsppTrie", &comp_lookup, comp_label, "ns/op", "bytes_per_key", cspp_bpk, comp_bpk);
        }
    }

    // 4. Hash Maps (1,000,000 u64 and [u8; 16] keys)
    for (key_kind, z_bpe_key, g_bpe_key, hb_bpe_key, std_bpe_key) in [
        ("u64", "hash_maps/zipora_u64_bytes_per_entry", "hash_maps/gold_u64_bytes_per_entry", "hash_maps/hashbrown_u64_bytes_per_entry", "hash_maps/std_u64_bytes_per_entry"),
        ("str16", "hash_maps/zipora_str16_bytes_per_entry", "hash_maps/gold_str16_bytes_per_entry", "hash_maps/hashbrown_str16_bytes_per_entry", "hash_maps/std_str16_bytes_per_entry"),
    ] {
        let (z_bpe, g_bpe, hb_bpe, std_bpe) = (sp(z_bpe_key, 0.0), sp(g_bpe_key, 0.0), sp(hb_bpe_key, 0.0), sp(std_bpe_key, 0.0));
        for (comp_prefix, comp_label, comp_bpe) in [
            ("hash_maps/hashbrown_ahash", "hashbrown+ahash", hb_bpe),
            ("hash_maps/std", "std::collections::HashMap", std_bpe),
        ] {
            for op_name in ["insert", "get"] {
                push_comparison(&mut out, records, "hash_maps", &format!("{op_name}_{key_kind} (ZiporaHashMap)"), &format!("hash_maps/zipora_hash_map/{op_name}_{key_kind}"), "ZiporaHashMap", &format!("{comp_prefix}/{op_name}_{key_kind}"), comp_label, "ns/op", "bytes_per_entry", z_bpe, comp_bpe);
                push_comparison(&mut out, records, "hash_maps", &format!("{op_name}_{key_kind} (GoldHashMap)"), &format!("hash_maps/gold_hash_map/{op_name}_{key_kind}"), "GoldHashMap", &format!("{comp_prefix}/{op_name}_{key_kind}"), comp_label, "ns/op", "bytes_per_entry", g_bpe, comp_bpe);
            }
        }
    }

    // 5. Memory Pools (10,000 ops of 256B alloc+free, 1T & 4T)
    let lf_slot = sp("memory_pools/lockfree_bytes_per_256b_slot", 256.0);
    let sec_slot = sp("memory_pools/secure_bytes_per_256b_slot", 384.0);
    let mi_slot = sp("memory_pools/mimalloc_bytes_per_256b_slot", 256.0);
    let je_slot = sp("memory_pools/jemalloc_bytes_per_256b_slot", 256.0);
    let sys_slot = sp("memory_pools/system_bytes_per_256b_slot", 272.0);

    for (comp_prefix, comp_label, comp_slot) in [
        ("memory_pools/mimalloc", "mimalloc::MiMalloc", mi_slot),
        ("memory_pools/jemalloc", "tikv_jemallocator::Jemalloc", je_slot),
        ("memory_pools/system_alloc", "std::alloc::System", sys_slot),
    ] {
        for threads in ["1t", "4t"] {
            push_comparison(&mut out, records, "memory_pools", &format!("alloc_free_{threads} (LockFreeMemoryPool)"), &format!("memory_pools/lockfree_pool/alloc_free_{threads}"), "LockFreeMemoryPool", &format!("{comp_prefix}/alloc_free_{threads}"), comp_label, "ns/op", "bytes_per_256b_slot", lf_slot, comp_slot);
            push_comparison(&mut out, records, "memory_pools", &format!("alloc_free_{threads} (SecureMemoryPool)"), &format!("memory_pools/secure_pool/alloc_free_{threads}"), "SecureMemoryPool", &format!("{comp_prefix}/alloc_free_{threads}"), comp_label, "ns/op", "bytes_per_256b_slot", sec_slot, comp_slot);
        }
    }

    // 6. Entropy Coding (64 KiB synthetic enwik8-like block)
    let (huff_o0_bpb, huff_o1_bpb, huff_o2_bpb) = (
        sp("entropy/huffman_o0_bits_per_byte", 0.0),
        sp("entropy/huffman_o1_bits_per_byte", 0.0),
        sp("entropy/huffman_o2_bits_per_byte", 0.0),
    );
    let (fse_bpb, rans_x1_bpb, rans_x4_bpb) = (
        sp("entropy/fse_bits_per_byte", 0.0),
        sp("entropy/rans64_x1_bits_per_byte", 0.0),
        sp("entropy/rans64_x4_bits_per_byte", 0.0),
    );
    let (hc_bpb, cons_bpb, zstd_huf0_bpb, zstd_fse_bpb, zstd_l1_bpb, zstd_l3_bpb) = (
        sp("entropy/huffman_compress_bits_per_byte", 0.0),
        sp("entropy/constriction_ans_bits_per_byte", 0.0),
        sp("entropy/zstd_huf0_bits_per_byte", 0.0),
        sp("entropy/zstd_fse_bits_per_byte", 0.0),
        sp("entropy/zstd_l1_bits_per_byte", 0.0),
        sp("entropy/zstd_l3_bits_per_byte", 0.0),
    );

    for (z_enc, z_dec, z_var, z_bpb) in [
        ("entropy/huffman_o0/encode_64kib", "entropy/huffman_o0/decode_64kib", "HuffmanEncoder (O0)", huff_o0_bpb),
        ("entropy/huffman_o1/encode_x1_64kib", "entropy/huffman_o1/decode_64kib", "ContextualHuffmanEncoder (O1 x1)", huff_o1_bpb),
        ("entropy/huffman_o1/encode_x4_64kib", "entropy/huffman_o1/decode_64kib", "ContextualHuffmanEncoder (O1 x4)", huff_o1_bpb),
        ("entropy/huffman_o2/encode_64kib", "entropy/huffman_o2/decode_64kib", "ContextualHuffmanEncoder (O2)", huff_o2_bpb),
    ] {
        for (c_prefix, c_label, c_bpb) in [
            ("entropy/huffman_compress", "huffman-compress", hc_bpb),
            ("entropy/zstd_huf0", "libzstd::huff0 (4X)", zstd_huf0_bpb),
            ("entropy/zstd_l1", "zstd (level 1, LZ77+entropy)", zstd_l1_bpb),
            ("entropy/zstd_l3", "zstd (level 3, LZ77+entropy)", zstd_l3_bpb),
        ] {
            push_comparison(&mut out, records, "entropy", &format!("encode_64kib ({z_var})"), z_enc, z_var, &format!("{c_prefix}/encode_64kib"), c_label, "ns/byte", "bits_per_byte", z_bpb, c_bpb);
            push_comparison(&mut out, records, "entropy", &format!("decode_64kib ({z_var})"), z_dec, z_var, &format!("{c_prefix}/decode_64kib"), c_label, "ns/byte", "bits_per_byte", z_bpb, c_bpb);
        }
    }

    for (z_enc, z_dec, z_var, z_bpb) in [
        ("entropy/fse/encode_64kib", "entropy/fse/decode_64kib", "FseEncoder", fse_bpb),
        ("entropy/rans64_x1/encode_64kib", "entropy/rans64_x1/decode_64kib", "Rans64Encoder<ParallelX1>", rans_x1_bpb),
        ("entropy/rans64_x4/encode_64kib", "entropy/rans64_x4/decode_64kib", "Rans64Encoder<ParallelX4>", rans_x4_bpb),
    ] {
        for (c_prefix, c_label, c_bpb) in [
            ("entropy/constriction_ans", "constriction::DefaultAnsCoder", cons_bpb),
            ("entropy/zstd_fse", "libzstd::FSE (tANS)", zstd_fse_bpb),
            ("entropy/zstd_l1", "zstd (level 1, LZ77+entropy)", zstd_l1_bpb),
            ("entropy/zstd_l3", "zstd (level 3, LZ77+entropy)", zstd_l3_bpb),
        ] {
            push_comparison(&mut out, records, "entropy", &format!("encode_64kib ({z_var})"), z_enc, z_var, &format!("{c_prefix}/encode_64kib"), c_label, "ns/byte", "bits_per_byte", z_bpb, c_bpb);
            push_comparison(&mut out, records, "entropy", &format!("decode_64kib ({z_var})"), z_dec, z_var, &format!("{c_prefix}/decode_64kib"), c_label, "ns/byte", "bits_per_byte", z_bpb, c_bpb);
        }
    }

    // 7. Integer Codecs (100,000 sorted u32 integers)
    let svb_bpi = sp("int_codecs/stream_vbyte_bits_per_int", 0.0);
    let gv_bpi = sp("int_codecs/group_varint_bits_per_int", 0.0);
    let svb_crate_bpi = sp("int_codecs/stream_vbyte_crate_bits_per_int", 0.0);
    let svb_simd_bpi = sp("int_codecs/stream_vbyte_simd_bits_per_int", svb_crate_bpi);
    let bp_bpi = sp("int_codecs/bitpacking_4x_bits_per_int", 0.0);

    for (comp_prefix, comp_label, comp_bpi) in [
        ("int_codecs/stream_vbyte_crate", "stream_vbyte::Scalar", svb_crate_bpi),
        ("int_codecs/stream_vbyte_simd", "stream_vbyte::Ssse3/Sse41", svb_simd_bpi),
        ("int_codecs/bitpacking_4x", "bitpacking::BitPacker4x", bp_bpi),
    ] {
        for op in ["encode_deltas", "decode_into", "decode_deltas"] {
            push_comparison(&mut out, records, "int_codecs", &format!("{op} (StreamVByte)"), &format!("int_codecs/stream_vbyte/{op}"), "StreamVByte", &format!("{comp_prefix}/{op}"), comp_label, "ns/int", "bits_per_int", svb_bpi, comp_bpi);
        }
        for op in ["encode_deltas", "decode_deltas"] {
            push_comparison(&mut out, records, "int_codecs", &format!("{op} (GroupVarint)"), &format!("int_codecs/group_varint/{op}"), "GroupVarint", &format!("{comp_prefix}/{op}"), comp_label, "ns/int", "bits_per_int", gv_bpi, comp_bpi);
        }
    }

    // 8. Radix Sort (10,000,000 elements: u32, u64, (u64, u64))
    for (kind, z_bench, z_var, z_space_key, comp_suffix, vor_space_key, rdst_space_key, std_space_key) in [
        ("u32", "radix_sort/lsd_u32", "RadixSort::sort_u32", "radix_sort/zipora_u32_bytes_per_elem", "u32", "radix_sort/voracious_u32_bytes_per_elem", "radix_sort/rdst_u32_bytes_per_elem", "radix_sort/std_u32_bytes_per_elem"),
        ("u64", "radix_sort/lsd_u64", "RadixSort::sort_u64", "radix_sort/zipora_u64_bytes_per_elem", "u64", "radix_sort/voracious_u64_bytes_per_elem", "radix_sort/rdst_u64_bytes_per_elem", "radix_sort/std_u64_bytes_per_elem"),
        ("(u64, u64)", "radix_sort/kv_u64_u64", "KeyValueRadixSort<u64, u64>", "radix_sort/zipora_kv_bytes_per_elem", "kv_u64_u64", "radix_sort/voracious_kv_bytes_per_elem", "radix_sort/rdst_kv_bytes_per_elem", "radix_sort/std_kv_bytes_per_elem"),
    ] {
        let (z_bpe, vor_bpe, rdst_bpe, std_bpe) = (sp(z_space_key, 0.0), sp(vor_space_key, 0.0), sp(rdst_space_key, 0.0), sp(std_space_key, 0.0));
        for (c_prefix, c_label, c_bpe) in [
            ("radix_sort/voracious", "voracious_radix_sort", vor_bpe),
            ("radix_sort/rdst", "rdst (single-threaded)", rdst_bpe),
            ("radix_sort/std_unstable", "[T]::sort_unstable", std_bpe),
        ] {
            push_comparison(&mut out, records, "radix_sort", &format!("sort_{kind} ({z_var})"), z_bench, z_var, &format!("{c_prefix}/{comp_suffix}"), c_label, "ns/elem", "bytes_per_elem", z_bpe, c_bpe);
        }
    }

    // 9. Suffix Array (10,000,000 bytes construct + 100,000 16B substring queries)
    let sa_z_bpb = sp("suffix_array/zipora_bytes_per_byte", 5.0);
    for (c_prefix, c_label, c_space_key) in [
        ("suffix_array/suffix_crate", "suffix::SuffixTable", "suffix_array/suffix_crate_bytes_per_byte"),
        ("suffix_array/divsufsort_rust", "divsufsort (Rust)", "suffix_array/divsufsort_rust_bytes_per_byte"),
        ("suffix_array/cdivsufsort_c", "cdivsufsort (C)", "suffix_array/cdivsufsort_c_bytes_per_byte"),
    ] {
        let c_bpb = sp(c_space_key, 5.0);
        push_comparison(&mut out, records, "suffix_array", "construct (10M bytes)", "suffix_array/construct", "SuffixArray", &format!("{c_prefix}/construct"), c_label, "ns/byte", "bytes_per_byte", sa_z_bpb, c_bpb);
        push_comparison(&mut out, records, "suffix_array", "search_query (10M bytes, 16B pattern)", "suffix_array/search_query", "SuffixArray", &format!("{c_prefix}/search_query"), c_label, "ns/op", "bytes_per_byte", sa_z_bpb, c_bpb);
    }

    // 10. String Search (64 KiB strchr & strstr)
    push_comparison(&mut out, records, "search_and_scoring", "strchr_64kib (SimdStringSearch)", "search_and_scoring/simd_strchr_64kib", "SimdStringSearch::sse42_strchr", "search_and_scoring/memchr_strchr_64kib", "memchr::memchr", "ns/call", "aux_bytes", 0.0, 0.0);
    for (c_bench, c_label) in [
        ("search_and_scoring/memmem_strstr_64kib", "memchr::memmem::find"),
        ("search_and_scoring/memmem_finder_strstr_64kib", "memchr::memmem::Finder"),
        ("search_and_scoring/aho_corasick_strstr_64kib", "aho_corasick::AhoCorasick"),
    ] {
        push_comparison(&mut out, records, "search_and_scoring", "strstr_64kib (SimdStringSearch)", "search_and_scoring/simd_strstr_64kib", "SimdStringSearch::sse42_strstr", c_bench, c_label, "ns/call", "aux_bytes", 0.0, 0.0);
    }

    out
}
