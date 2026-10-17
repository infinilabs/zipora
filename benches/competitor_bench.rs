//! Canonical Phase E Competitor Benchmark Matrix (`plan.md` §1.6 `E2` + `E3` + `B3`).
//!
//! Runs Zipora and each external Rust competitor crate in the **same process**, on the
//! **exact same deterministic fixed-seed datasets and uniform query streams** (`benches/common/mod.rs`),
//! under the same self-only CPU pinning and quiet-CCD discipline as `benches/core_bench.rs`.
//!
//! Emits structured JSON to `target/bench/competitor_bench.json` (recording per-benchmark
//! latency distributions, space metrics, competitor crate versions/configurations, and
//! same-run Zipora-vs-competitor primary/secondary ratios + `E3` verdicts), and writes
//! `benches/baselines/<host>/competitor_bench.json` when `UPDATE_BASELINE=1`,
//! `ZIPORA_BENCH_SAVE_BASELINE=1`, or `--save-baseline` is explicitly provided.

use std::alloc::{GlobalAlloc, Layout, System as SystemAlloc};
use std::collections::hash_map::DefaultHasher as StdSipHasher13;
use std::collections::{BTreeMap, HashMap as StdHashMap};
use std::fs;
use std::hash::BuildHasherDefault;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

type DeterministicStdHashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<StdSipHasher13>>;

use aho_corasick::AhoCorasick;
use bit_vec::BitVec as HuffmanBitVec;
use bitm::{
    CombinedSampling, Rank as BitmRank, RankSelect101111, Select as BitmSelect,
    Select0 as BitmSelect0,
};
use bitpacking::{BitPacker, BitPacker4x};
use constriction::stream::{
    Decode as ConstrictionDecode,
    model::SmallContiguousCategoricalEntropyModel,
    stack::DefaultAnsCoder,
};
use dyn_size_of::GetSize;
use fst::Set as FstSet;
use hashbrown::HashMap as HashbrownMap;
use huffman_compress::CodeBuilder as HuffmanCompressBuilder;
use memchr::{memchr as crate_memchr, memmem as crate_memmem};
use mimalloc::MiMalloc;
use qp_trie::Trie as QpTrie;
use radix_trie::{Trie as RadixTrie, TrieCommon as _};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rdst::{RadixKey as RdstRadixKey, RadixSort as RdstRadixSort};
use sacabase::StringIndex as SacabaseStringIndex;
use serde::{Deserialize, Serialize};
use stream_vbyte::{
    decode::decode as svb_crate_decode, encode::encode as svb_crate_encode,
    scalar::Scalar as SvbCrateScalar,
};
use sucds::Serializable as SucdsSerializable;
use sucds::bit_vectors::{
    BitVector as SucdsBitVector, Rank as SucdsRank, Rank9Sel as SucdsRank9Sel,
    Select as SucdsSelect,
};
use sucds::mii_sequences::{
    EliasFano as SucdsEliasFano, EliasFanoBuilder as SucdsEliasFanoBuilder,
};
use suffix::SuffixTable;
use tikv_jemallocator::Jemalloc;
use vers_vecs::{BitVec as VersBitVec, EliasFanoVec as VersEliasFanoVec, RsVec as VersRsVec};
use voracious_radix_sort::{RadixSort as VoraciousRadixSort, Radixable as VoraciousRadixable};

use zipora::algorithms::radix_sort::{KeyValueRadixSort, RadixSort, RadixSortConfig};
use zipora::algorithms::suffix_array::SuffixArray;
use zipora::compression::stream_vbyte::{GroupVarint, StreamVByte};
use zipora::entropy::fse::{FseConfig, FseEncoder, fse_decompress};
use zipora::entropy::huffman::{
    ContextualHuffmanDecoder, ContextualHuffmanEncoder, HuffmanDecoder, HuffmanEncoder,
    HuffmanOrder,
};
use zipora::entropy::rans::{ParallelX1, ParallelX4, Rans64Decoder, Rans64Encoder};
use zipora::fsa::cspp_trie::CsppTrie;
use zipora::fsa::{DoubleArrayTrie, TrieStrategy, ZiporaTrie, ZiporaTrieConfig};
use zipora::hash_map::{
    GoldHashMap, GoldHashMapConfig, HashStorageStrategy, ZiporaHashMap, ZiporaHashMapConfig,
};
use zipora::memory::{
    LockFreeMemoryPool, LockFreePoolConfig, SecureMemoryPool, SecurePoolConfig,
};
use zipora::scoring::bm25::Bm25BatchScorer;
use zipora::scoring::fieldnorm::FieldnormEncoder;
use zipora::string::SimdStringSearch;
use zipora::succinct::BitVector;
use zipora::succinct::elias_fano::{
    ClusteredEliasFano, EliasFano, HybridPostingList, OptimalPartitionedEliasFano,
    PartitionedEliasFano,
};
use zipora::succinct::rank_select::{
    Rank9, RankSelectInterleaved256, RankSelectOps, RankSelectSE256,
};

mod common;
use common::*;

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

fn build_competitor_configs() -> BTreeMap<String, CompetitorConfigMeta> {
    let mut m = BTreeMap::new();
    m.insert(
        "sucds".to_string(),
        CompetitorConfigMeta {
            crate_name: "sucds".to_string(),
            version: "0.10.0".to_string(),
            features: vec!["std".to_string(), "intrinsics".to_string()],
            api_used: "bit_vectors::Rank9Sel::new(bv).select1_hints().select0_hints() & mii_sequences::EliasFanoBuilder::build().enable_rank()".to_string(),
            notes: "Broadword bit operations accelerated via hardware intrinsics feature".to_string(),
        },
    );
    m.insert(
        "vers-vecs".to_string(),
        CompetitorConfigMeta {
            crate_name: "vers-vecs".to_string(),
            version: "1.10.2".to_string(),
            features: vec!["simd".to_string()],
            api_used: "RsVec::from_bit_vec(BitVec::from_vec) & EliasFanoVec::from_slice"
                .to_string(),
            notes: "Hardware POPCNT/PDEP/TZCNT + SIMD feature enabled".to_string(),
        },
    );
    m.insert(
        "bitm".to_string(),
        CompetitorConfigMeta {
            crate_name: "bitm".to_string(),
            version: "0.5.2".to_string(),
            features: vec![],
            api_used: "RankSelect101111::<CombinedSampling, CombinedSampling>::build(Box<[u64]>)"
                .to_string(),
            notes:
                "3-level interleaved rank index (3.125% rank overhead + CombinedSampling select)"
                    .to_string(),
        },
    );
    m.insert(
        "fst".to_string(),
        CompetitorConfigMeta {
            crate_name: "fst".to_string(),
            version: "0.4.7".to_string(),
            features: vec!["default".to_string()],
            api_used: "fst::Set::from_iter(sorted_keys) & Set::contains(&[u8])".to_string(),
            notes: "Finite-state transducer minimal acyclic deterministic automaton over sorted keys"
                .to_string(),
        },
    );
    m.insert(
        "qp-trie".to_string(),
        CompetitorConfigMeta {
            crate_name: "qp-trie".to_string(),
            version: "0.8.2".to_string(),
            features: vec!["default".to_string()],
            api_used: "qp_trie::Trie::<&[u8], ()>::insert & Trie::contains_key(&[u8])".to_string(),
            notes: "Quad-bit popcount patricia trie borrowing key slices (&[u8]); bytes/key includes node heap + key backing bytes"
                .to_string(),
        },
    );
    m.insert(
        "radix_trie".to_string(),
        CompetitorConfigMeta {
            crate_name: "radix_trie".to_string(),
            version: "0.3.0".to_string(),
            features: vec![],
            api_used: "radix_trie::Trie::<&[u8], ()>::insert & Trie::get(&[u8])".to_string(),
            notes: "16-way nibble-compressed radix tree with boxed child pointers and Nibblet vectors; bytes/key includes node heap + key backing bytes"
                .to_string(),
        },
    );
    m.insert(
        "hashbrown+ahash".to_string(),
        CompetitorConfigMeta {
            crate_name: "hashbrown".to_string(),
            version: "0.16.1".to_string(),
            features: vec![
                "inline-more".to_string(),
                "raw-entry".to_string(),
                "equivalent".to_string(),
            ],
            api_used: "hashbrown::HashMap::<K, u64, ahash::RandomState>::with_capacity_and_hasher & insert / get".to_string(),
            notes: "Google SwissTable (AVX2 16-way control byte group probing) with ahash v0.8.12".to_string(),
        },
    );
    m.insert(
        "std::collections::HashMap".to_string(),
        CompetitorConfigMeta {
            crate_name: "std".to_string(),
            version: "1.97.1".to_string(),
            features: vec!["SipHash13".to_string()],
            api_used: "std::collections::HashMap::<K, u64>::with_capacity & insert / get".to_string(),
            notes: "Standard library SwissTable (hashbrown backend) with default SipHash-1-3 RandomState".to_string(),
        },
    );
    m.insert(
        "mimalloc".to_string(),
        CompetitorConfigMeta {
            crate_name: "mimalloc".to_string(),
            version: "0.1.52".to_string(),
            features: vec![],
            api_used: "<mimalloc::MiMalloc as GlobalAlloc>::alloc / dealloc".to_string(),
            notes: "Microsoft mimalloc thread-local free-list allocator invoked directly via GlobalAlloc".to_string(),
        },
    );
    m.insert(
        "tikv-jemallocator".to_string(),
        CompetitorConfigMeta {
            crate_name: "tikv-jemallocator".to_string(),
            version: "0.6.1".to_string(),
            features: vec!["default".to_string()],
            api_used: "<tikv_jemallocator::Jemalloc as GlobalAlloc>::alloc / dealloc".to_string(),
            notes: "jemalloc thread-cached arena allocator invoked directly via GlobalAlloc".to_string(),
        },
    );
    m.insert(
        "std::alloc::System".to_string(),
        CompetitorConfigMeta {
            crate_name: "std".to_string(),
            version: "1.97.1".to_string(),
            features: vec!["glibc_ptmalloc".to_string()],
            api_used: "<std::alloc::System as GlobalAlloc>::alloc / dealloc".to_string(),
            notes: "OS system allocator (glibc ptmalloc2 with per-thread tcache)".to_string(),
        },
    );
    m.insert(
        "zstd".to_string(),
        CompetitorConfigMeta {
            crate_name: "zstd".to_string(),
            version: "0.13.3".to_string(),
            features: vec!["default".to_string()],
            api_used: "zstd::bulk::Compressor::new(level).compress(&[u8]) & zstd::bulk::Decompressor::new().decompress(&[u8], capacity)".to_string(),
            notes: "Official libzstd C binding at levels 1 and 3 with reused ZSTD_CCtx / ZSTD_DCtx (LZ77 + FSE + 4-stream Huffman whole-block compressor)".to_string(),
        },
    );
    m.insert(
        "constriction".to_string(),
        CompetitorConfigMeta {
            crate_name: "constriction".to_string(),
            version: "0.5.0".to_string(),
            features: vec!["std".to_string()],
            api_used: "DefaultAnsCoder (AnsCoder<u32, u64>) + SmallContiguousCategoricalEntropyModel (PRECISION=12) + SmallContiguousLookupDecoderModel (PRECISION=12)".to_string(),
            notes: "64-bit state / 32-bit word rANS coder with 12-bit (4096-entry) O(1) lookup decoder table".to_string(),
        },
    );
    m.insert(
        "huffman-compress".to_string(),
        CompetitorConfigMeta {
            crate_name: "huffman-compress".to_string(),
            version: "0.7.0".to_string(),
            features: vec![],
            api_used: "CodeBuilder::from_iter(weights).finish() -> Book::<u8>::encode(&mut BitVec, &u8) & Tree::<u8>::decoder(&BitVec, len)".to_string(),
            notes: "Static Order-0 Huffman codebook (BTreeMap<u8, BitVec>) + arena binary tree decoder over bit-vec v0.8.0".to_string(),
        },
    );
    m.insert(
        "stream-vbyte".to_string(),
        CompetitorConfigMeta {
            crate_name: "stream-vbyte".to_string(),
            version: "0.4.1".to_string(),
            features: vec![],
            api_used: "stream_vbyte::encode::encode::<Scalar> & stream_vbyte::decode::decode::<Scalar>".to_string(),
            notes: "stream-vbyte v0.4.1 gates x86_ssse3/x86_sse41 behind unstable #![feature(portable_simd)]; benchmarked on stable rustc with Scalar + delta pass".to_string(),
        },
    );
    m.insert(
        "bitpacking".to_string(),
        CompetitorConfigMeta {
            crate_name: "bitpacking".to_string(),
            version: "0.9.3".to_string(),
            features: vec![
                "bitpacker1x".to_string(),
                "bitpacker4x".to_string(),
                "bitpacker8x".to_string(),
            ],
            api_used: "bitpacking::BitPacker4x::{num_bits_sorted, compress_sorted, decompress, decompress_sorted}".to_string(),
            notes: "Quickwit/Tantivy SIMD bit-packing (128-int blocks, SSE3 4-lane BitPacker4x with SIMD delta integration)".to_string(),
        },
    );
    m.insert(
        "voracious_radix_sort".to_string(),
        CompetitorConfigMeta {
            crate_name: "voracious_radix_sort".to_string(),
            version: "1.2.0".to_string(),
            features: vec![],
            api_used: "<[T] as voracious_radix_sort::RadixSort>::voracious_sort".to_string(),
            notes: "Single-threaded hybrid MSD/Ska/LSD radix sort over u32, u64, and Radixable<u64> (u64, u64) pairs".to_string(),
        },
    );
    m.insert(
        "rdst".to_string(),
        CompetitorConfigMeta {
            crate_name: "rdst".to_string(),
            version: "0.20.11".to_string(),
            features: vec![],
            api_used: "slice.radix_sort_builder().with_single_threaded_tuner().sort()".to_string(),
            notes: "Single-threaded adaptive LSB/Ska radix sort with RadixKey for u32, u64, and (u64, u64) pairs".to_string(),
        },
    );
    m.insert(
        "std::slice::sort_unstable".to_string(),
        CompetitorConfigMeta {
            crate_name: "std".to_string(),
            version: "1.97.1".to_string(),
            features: vec!["ipnsort".to_string()],
            api_used: "<[T]>::sort_unstable & <[T]>::sort_unstable_by_key".to_string(),
            notes: "Standard library in-place pattern-defeating quicksort (ipnsort)".to_string(),
        },
    );
    m.insert(
        "suffix".to_string(),
        CompetitorConfigMeta {
            crate_name: "suffix".to_string(),
            version: "1.3.0".to_string(),
            features: vec![],
            api_used: "suffix::SuffixTable::new(&str) & SuffixTable::positions(&str)".to_string(),
            notes: "BurntSushi pure-Rust SA-IS suffix array construction and binary search".to_string(),
        },
    );
    m.insert(
        "divsufsort".to_string(),
        CompetitorConfigMeta {
            crate_name: "divsufsort".to_string(),
            version: "2.0.0".to_string(),
            features: vec![],
            api_used: "divsufsort::sort(&[u8]) & sacabase::StringIndex::longest_substring_match(&[u8])".to_string(),
            notes: "Pure-Rust port of Yuta Mori's libdivsufsort induced-sorting suffix array construction".to_string(),
        },
    );
    m.insert(
        "cdivsufsort".to_string(),
        CompetitorConfigMeta {
            crate_name: "cdivsufsort".to_string(),
            version: "2.0.0".to_string(),
            features: vec![],
            api_used: "cdivsufsort::sort(&[u8]) & sacabase::StringIndex::longest_substring_match(&[u8])".to_string(),
            notes: "C FFI binding to Yuta Mori's libdivsufsort reference implementation".to_string(),
        },
    );
    m.insert(
        "memchr".to_string(),
        CompetitorConfigMeta {
            crate_name: "memchr".to_string(),
            version: "2.7.5".to_string(),
            features: vec!["std".to_string()],
            api_used: "memchr::memchr(u8, &[u8]), memchr::memmem::find(&[u8], &[u8]), & memchr::memmem::Finder::find(&[u8])".to_string(),
            notes: "BurntSushi AVX2/SSE2 vectorized byte and substring search".to_string(),
        },
    );
    m.insert(
        "aho-corasick".to_string(),
        CompetitorConfigMeta {
            crate_name: "aho-corasick".to_string(),
            version: "1.1.3".to_string(),
            features: vec!["std".to_string(), "perf-literal".to_string()],
            api_used: "aho_corasick::AhoCorasick::new([needle]).unwrap().find(&[u8])".to_string(),
            notes: "BurntSushi Aho-Corasick automaton with SIMD prefilter".to_string(),
        },
    );
    m
}

// ---------------------------------------------------------------------------
// 1. Rank/Select (1 Gbit = 1,000,000,000 bits, 1,000,000 uniform random queries)
//    Zipora: Rank9, RankSelectInterleaved256, RankSelectSE256
//    Competitors: sucds::Rank9Sel, vers_vecs::RsVec, bitm::RankSelect101111
// ---------------------------------------------------------------------------

fn bench_rank_select(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let num_bits = sizes.rank_select_bits;
    let num_words = num_bits / 64;
    let num_queries = sizes.rank_select_queries;
    let query_batch = (num_queries / 100).max(1_000).min(num_queries.max(1));
    let select_batch = (num_queries / 140).max(1_000).min(num_queries.max(1));
    let select0_interleaved_batch = (num_queries / 160).max(1_000).min(num_queries.max(1));

    let mut rank_params = BenchParams::heavy(num_bits as u64, query_batch as u64, 12);
    rank_params.warmup_calls = 8;
    rank_params.bursts_per_trial = 8;
    let mut select1_params = BenchParams::heavy(num_bits as u64, select_batch as u64, 10);
    select1_params.warmup_calls = 6;
    select1_params.bursts_per_trial = 14;
    let mut select0_interleaved_params =
        BenchParams::heavy(num_bits as u64, select0_interleaved_batch as u64, 10);
    select0_interleaved_params.warmup_calls = 6;
    select0_interleaved_params.bursts_per_trial = 16;

    let rank_seed = SEED_BASE ^ 0x0101;
    let select1_seed = SEED_BASE ^ 0x0102;
    let select0_seed = SEED_BASE ^ 0x0103;
    let base_words = gen_rank_words(num_words);
    let raw_bytes = (num_words * 8) as f64;
    let total_ones: usize = base_words.iter().map(|w| w.count_ones() as usize).sum();
    let max_ones = total_ones.max(1);
    let max_zeros = num_bits.saturating_sub(total_ones).max(1);

    // 1a. Zipora Rank9
    if should_run_bench("rank_select/rank9/rank1") {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        let r9 = Rank9::new(BitVector::from_blocks(words, num_bits)).unwrap();
        warm_rank_pages(&r9, num_bits);
        record_space_metric(
            space,
            "rank_select/rank9_overhead_pct",
            r9.space_overhead_percent(),
        );
        measure_bench(
            records,
            "rank_select",
            "rank_select/rank9/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(r9.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );
    }

    // 1b. Zipora RankSelectInterleaved256
    if should_run_any(&[
        "rank_select/interleaved256/rank1",
        "rank_select/interleaved256/select1",
        "rank_select/interleaved256/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        let rsi = RankSelectInterleaved256::new(BitVector::from_blocks(words, num_bits)).unwrap();
        record_space_metric(
            space,
            "rank_select/interleaved256_overhead_pct",
            rsi.space_overhead_percent(),
        );
        warm_rank_pages(&rsi, num_bits);
        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(rsi.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );

        warm_select1_pages(&rsi, max_ones);
        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/select1",
            "ns/op",
            select1_params,
            || {
                let base = current_burst_index() * select_batch;
                let mut acc = 0usize;
                for idx in base..base + select_batch {
                    let q = uniform_index_at(select1_seed, idx, max_ones);
                    acc = acc.wrapping_add(rsi.select1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        warm_select0_pages(&rsi, max_zeros);
        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/select0",
            "ns/op",
            select0_interleaved_params,
            || {
                let base = current_burst_index() * select0_interleaved_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_interleaved_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(rsi.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1c. Zipora RankSelectSE256
    if should_run_any(&[
        "rank_select/se256/rank1",
        "rank_select/se256/select1",
        "rank_select/se256/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        let rs_se = RankSelectSE256::new(BitVector::from_blocks(words, num_bits)).unwrap();
        warm_rank_pages(&rs_se, num_bits);
        record_space_metric(
            space,
            "rank_select/se256_overhead_pct",
            rs_se.space_overhead_percent(),
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(rs_se.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );

        warm_select1_pages(&rs_se, max_ones);
        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/select1",
            "ns/op",
            select1_params,
            || {
                let base = current_burst_index() * select_batch;
                let mut acc = 0usize;
                for idx in base..base + select_batch {
                    let q = uniform_index_at(select1_seed, idx, max_ones);
                    acc = acc.wrapping_add(rs_se.select1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        warm_select0_pages(&rs_se, max_zeros);
        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/select0",
            "ns/op",
            select0_interleaved_params,
            || {
                let base = current_burst_index() * select0_interleaved_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_interleaved_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(rs_se.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1d. Competitor: sucds::bit_vectors::Rank9Sel (with select1_hints + select0_hints)
    if should_run_any(&[
        "rank_select/sucds_rank9sel/rank1",
        "rank_select/sucds_rank9sel/select1",
        "rank_select/sucds_rank9sel/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let mut sbv = SucdsBitVector::with_capacity(num_bits);
        for &w in &base_words {
            sbv.push_bits(w, 64).unwrap();
        }
        let sucds_rs = SucdsRank9Sel::new(sbv).select1_hints().select0_hints();
        let sucds_overhead_pct =
            ((sucds_rs.size_in_bytes() as f64) - raw_bytes).max(0.0) / raw_bytes * 100.0;
        record_space_metric(
            space,
            "rank_select/sucds_rank9sel_overhead_pct",
            sucds_overhead_pct,
        );

        let mut warm = 0usize;
        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0190, idx, num_bits);
            warm = warm.wrapping_add(sucds_rs.rank1(q).unwrap_or(0));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/sucds_rank9sel/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(sucds_rs.rank1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0191, idx, max_ones);
            warm = warm.wrapping_add(sucds_rs.select1(q).unwrap_or(0));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/sucds_rank9sel/select1",
            "ns/op",
            select1_params,
            || {
                let base = current_burst_index() * select_batch;
                let mut acc = 0usize;
                for idx in base..base + select_batch {
                    let q = uniform_index_at(select1_seed, idx, max_ones);
                    acc = acc.wrapping_add(sucds_rs.select1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0192, idx, max_zeros);
            warm = warm.wrapping_add(sucds_rs.select0(q).unwrap_or(0));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/sucds_rank9sel/select0",
            "ns/op",
            select0_interleaved_params,
            || {
                let base = current_burst_index() * select0_interleaved_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_interleaved_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(sucds_rs.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1e. Competitor: vers_vecs::RsVec
    if should_run_any(&[
        "rank_select/vers_vecs_rsvec/rank1",
        "rank_select/vers_vecs_rsvec/select1",
        "rank_select/vers_vecs_rsvec/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        let vers_rs = VersRsVec::from_bit_vec(VersBitVec::from_vec(words));
        let vers_overhead_pct =
            ((vers_rs.heap_size() as f64) - raw_bytes).max(0.0) / raw_bytes * 100.0;
        record_space_metric(
            space,
            "rank_select/vers_vecs_rsvec_overhead_pct",
            vers_overhead_pct,
        );

        let mut warm = 0usize;
        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0190, idx, num_bits);
            warm = warm.wrapping_add(vers_rs.rank1(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/vers_vecs_rsvec/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(vers_rs.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0191, idx, max_ones);
            warm = warm.wrapping_add(vers_rs.select1(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/vers_vecs_rsvec/select1",
            "ns/op",
            select1_params,
            || {
                let base = current_burst_index() * select_batch;
                let mut acc = 0usize;
                for idx in base..base + select_batch {
                    let q = uniform_index_at(select1_seed, idx, max_ones);
                    acc = acc.wrapping_add(vers_rs.select1(black_box(q)));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0192, idx, max_zeros);
            warm = warm.wrapping_add(vers_rs.select0(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/vers_vecs_rsvec/select0",
            "ns/op",
            select0_interleaved_params,
            || {
                let base = current_burst_index() * select0_interleaved_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_interleaved_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(vers_rs.select0(black_box(q)));
                }
                black_box(acc);
            },
        );
    }

    // 1f. Competitor: bitm::RankSelect101111<CombinedSampling, CombinedSampling>
    if should_run_any(&[
        "rank_select/bitm_rs101111/rank1",
        "rank_select/bitm_rs101111/select1",
        "rank_select/bitm_rs101111/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words;
        let (bitm_rs, _) =
            RankSelect101111::<CombinedSampling, CombinedSampling>::build(words.into_boxed_slice());
        let bitm_overhead_pct =
            ((bitm_rs.size_bytes_dyn() as f64) - raw_bytes).max(0.0) / raw_bytes * 100.0;
        record_space_metric(
            space,
            "rank_select/bitm_rs101111_overhead_pct",
            bitm_overhead_pct,
        );

        let mut warm = 0usize;
        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0190, idx, num_bits);
            warm = warm.wrapping_add(bitm_rs.rank(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/bitm_rs101111/rank1",
            "ns/op",
            rank_params,
            || {
                let base = current_burst_index() * query_batch;
                let mut acc = 0usize;
                for idx in base..base + query_batch {
                    let q = uniform_index_at(rank_seed, idx, num_bits);
                    acc = acc.wrapping_add(bitm_rs.rank(black_box(q)));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0191, idx, max_ones);
            warm = warm.wrapping_add(bitm_rs.select(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/bitm_rs101111/select1",
            "ns/op",
            select1_params,
            || {
                let base = current_burst_index() * select_batch;
                let mut acc = 0usize;
                for idx in base..base + select_batch {
                    let q = uniform_index_at(select1_seed, idx, max_ones);
                    acc = acc.wrapping_add(bitm_rs.select(black_box(q)));
                }
                black_box(acc);
            },
        );

        for idx in 0..100_000usize {
            let q = uniform_index_at(SEED_BASE ^ 0x0192, idx, max_zeros);
            warm = warm.wrapping_add(bitm_rs.select0(q));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/bitm_rs101111/select0",
            "ns/op",
            select0_interleaved_params,
            || {
                let base = current_burst_index() * select0_interleaved_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_interleaved_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(bitm_rs.select0(black_box(q)));
                }
                black_box(acc);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Elias-Fano Posting Lists (100,000 elements, 100,000 distinct queries)
//    Zipora: EliasFano, PartitionedEliasFano, OptimalPartitionedEliasFano, ClusteredEliasFano
//    Competitors: sucds::mii_sequences::EliasFano, vers_vecs::EliasFanoVec
// ---------------------------------------------------------------------------

fn bench_elias_fano(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.elias_fano_elements;
    let num_queries = n;
    let mut params = BenchParams::medium(n as u64, num_queries as u64);
    params.warmup_calls = 2;
    params.trials = 7;
    params.bursts_per_trial = 3;

    let uniform = gen_uniform_sparse(n, 50, SEED_BASE ^ 0x02);
    let clustered = gen_zipf_clustered(n, SEED_BASE ^ 0x03);
    let dense: Vec<u32> = (0..n as u32).collect();

    for (dist_name, data) in [
        ("uniform", &uniform),
        ("zipf_clustered", &clustered),
        ("dense", &dense),
    ] {
        if !should_run_prefix(&format!("elias_fano/{dist_name}/")) {
            continue;
        }
        let mut ef_spacers: Vec<Vec<u8>> = Vec::with_capacity(24);
        let ef0 = EliasFano::from_sorted(data);
        let efs: [EliasFano; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            ef0.clone()
        });
        let pef0 = PartitionedEliasFano::from_sorted(data);
        let pefs: [PartitionedEliasFano; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            pef0.clone()
        });
        let opef0 = OptimalPartitionedEliasFano::from_sorted(data);
        let opefs: [OptimalPartitionedEliasFano; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            opef0.clone()
        });
        let cef0 = ClusteredEliasFano::from_sorted(data);
        let cefs: [ClusteredEliasFano; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            cef0.clone()
        });
        let hybrid = HybridPostingList::from_sorted(data);

        let universe = *data.last().unwrap() as u64 + 1;
        let data_u64: Vec<u64> = data.iter().map(|&v| v as u64).collect();
        let mut sucds_builder = SucdsEliasFanoBuilder::new(universe, n).unwrap();
        sucds_builder.extend(data_u64.iter().copied()).unwrap();
        let sucds_ef0 = sucds_builder.build().enable_rank();
        let sucds_efs: [SucdsEliasFano; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            sucds_ef0.clone()
        });

        let vers_ef0 = VersEliasFanoVec::from_slice(&data_u64);
        let vers_efs: [VersEliasFanoVec; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 4_672]);
            vers_ef0.clone()
        });

        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/ef_bits_per_elem"),
            efs[0].bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/pef_bits_per_elem"),
            pefs[0].bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/opef_bits_per_elem"),
            opefs[0].bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/cef_bits_per_elem"),
            cefs[0].bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/hybrid_bits_per_elem"),
            hybrid.bits_per_element(),
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/sucds_ef_bits_per_elem"),
            (sucds_efs[0].size_in_bytes() * 8) as f64 / n as f64,
        );
        record_space_metric(
            space,
            &format!("elias_fano/{dist_name}/vers_ef_bits_per_elem"),
            (vers_efs[0].heap_size() * 8) as f64 / n as f64,
        );

        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x04);
        let targets: Vec<u64> = (0..num_queries)
            .map(|_| rng.random_range(0..universe))
            .collect();
        let targets_ring: [Vec<u64>; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 136 * 1024 + 4_672]);
            targets.clone()
        });
        drop(targets);

        let stream_len = 1_000.min(num_queries.max(1));
        let num_streams = (num_queries / stream_len).max(1);
        let cursor_streams: Vec<Vec<u64>> = (0..num_streams)
            .map(|_| {
                let mut stream: Vec<u64> = (0..stream_len)
                    .map(|_| rng.random_range(0..universe))
                    .collect();
                stream.sort_unstable();
                stream
            })
            .collect();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut ef_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/ef_next_geq"),
            "ns/op",
            params,
            &mut ef_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &efs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some((_, v)) = ef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut pef_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_next_geq"),
            "ns/op",
            params,
            &mut pef_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let pef = &pefs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some((_, v)) = pef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut opef_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_next_geq"),
            "ns/op",
            params,
            &mut opef_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let opef = &opefs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some((_, v)) = opef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut cef_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/cef_next_geq"),
            "ns/op",
            params,
            &mut cef_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let cef = &cefs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some((_, v)) = cef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut cursor_params = BenchParams::medium(n as u64, (num_streams * stream_len) as u64);
        cursor_params.warmup_calls = 2;
        cursor_params.trials = 7;
        cursor_params.bursts_per_trial = 3;
        let mut pef_cur_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_cursor_advance_to_geq"),
            "ns/op",
            cursor_params,
            &mut pef_cur_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let pef = &pefs[*ring];
                let mut acc = 0u64;
                for stream in &cursor_streams {
                    let mut cursor = pef.cursor();
                    for &t in stream {
                        if cursor.advance_to_geq(black_box(t))
                            && let Some(v) = cursor.current()
                        {
                            acc = acc.wrapping_add(v);
                        }
                    }
                }
                black_box(acc);
            },
        );

        let mut opef_cur_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_cursor_advance_to_index"),
            "ns/op",
            params,
            &mut opef_cur_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let opef = &opefs[*ring];
                let mut acc = 0u64;
                let mut cursor = opef.cursor();
                for &idx in &indices {
                    if cursor.advance_to_index(black_box(idx))
                        && let Some(v) = cursor.current()
                    {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut ef_sel_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/ef_select"),
            "ns/op",
            params,
            &mut ef_sel_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &efs[*ring];
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = ef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut cef_sel_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/cef_select"),
            "ns/op",
            params,
            &mut cef_sel_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let cef = &cefs[*ring];
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = cef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        // Competitor: sucds::mii_sequences::EliasFano successor (next_geq) & select
        let mut sucds_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/sucds_ef_next_geq"),
            "ns/op",
            params,
            &mut sucds_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &sucds_efs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some(v) = ef.successor(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut sucds_sel_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/sucds_ef_select"),
            "ns/op",
            params,
            &mut sucds_sel_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &sucds_efs[*ring];
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = ef.select(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        // Competitor: vers_vecs::EliasFanoVec successor (next_geq) & get (select)
        let mut vers_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/vers_ef_next_geq"),
            "ns/op",
            params,
            &mut vers_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &vers_efs[*ring];
                let t_slice = &targets_ring[*ring];
                let mut acc = 0u64;
                for &t in t_slice {
                    if let Some(v) = ef.successor(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        let mut vers_sel_ring = 0usize;
        measure_bench_with_setup(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/vers_ef_select"),
            "ns/op",
            params,
            &mut vers_sel_ring,
            |ring| {
                *ring = (*ring + 1) & 3;
            },
            |ring| {
                let ef = &vers_efs[*ring];
                let mut acc = 0u64;
                for &idx in &indices {
                    if let Some(v) = ef.get(black_box(idx)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );
        drop(ef_spacers);
    }
}

fn mallinfo_used_bytes() -> usize {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `libc::mallinfo2` is thread-safe and reads glibc allocator statistics by value.
        let mi = unsafe { libc::mallinfo2() };
        mi.uordblks.saturating_add(mi.hblkhd)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        0
    }
}

/// Pre-faults `total_mib` MiB in glibc's main `brk` arena using `16 KiB` allocations
/// (below `M_MMAP_THRESHOLD`) and returns a top-of-heap anchor so glibc does not `systrim`
/// the arena back to the kernel between benchmark trials, ensuring identical heap state
/// between full-suite runs and isolated `--filter` confirmation retries.
fn warm_brk_arena_with_anchor(total_mib: usize) -> Vec<u8> {
    let chunks = (total_mib * 1024) / 16;
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        blocks.push(black_box(vec![1u8; 16 * 1024]));
    }
    let anchor = black_box(vec![1u8; 16 * 1024]);
    drop(blocks);
    anchor
}

// ---------------------------------------------------------------------------
// 3. Tries (1,000,000 keys across 3 distributions, lookup on all 1,000,000 keys)
//    Zipora: DoubleArrayTrie, CsppTrie
//    Competitors: fst::Set, qp_trie::Trie, radix_trie::Trie
// ---------------------------------------------------------------------------

fn bench_tries(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.trie_keys;
    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    let lookup_batch = (n / 250).max(1_000).min(n.max(1));

    for dist_idx in 0..3usize {
        let dist_label = match dist_idx {
            0 => "url_paths",
            1 => "random_16b",
            _ => "shared_prefix",
        };
        let run_dat = should_run_any(&[
            &format!("tries/dat/{dist_label}/build_sorted"),
            &format!("tries/dat/{dist_label}/lookup"),
        ]);
        let run_cspp = should_run_any(&[
            &format!("tries/cspp/{dist_label}/build"),
            &format!("tries/cspp/{dist_label}/lookup"),
        ]);
        let run_fst = should_run_any(&[
            &format!("tries/fst/{dist_label}/build_sorted"),
            &format!("tries/fst/{dist_label}/lookup"),
        ]);
        let run_qp = should_run_any(&[
            &format!("tries/qp_trie/{dist_label}/build"),
            &format!("tries/qp_trie/{dist_label}/lookup"),
        ]);
        let run_radix = should_run_any(&[
            &format!("tries/radix_trie/{dist_label}/build"),
            &format!("tries/radix_trie/{dist_label}/lookup"),
        ]);
        if !run_dat && !run_cspp && !run_fst && !run_qp && !run_radix {
            continue;
        }
        maybe_repin_self_within_ccd();
        let perm = BijectivePerm::new(n, SEED_BASE ^ 0x0500 ^ (dist_idx as u64));
        let num_keys = n as u64;
        let mut lookup_params = BenchParams::heavy(num_keys, lookup_batch as u64, 14);
        lookup_params.warmup_calls = 10;
        lookup_params.bursts_per_trial = 18;

        let mut dat_build_params = BenchParams::heavy(num_keys, num_keys, 3);
        dat_build_params.warmup_calls = usize::from(dist_idx == 2);
        dat_build_params.bursts_per_trial = 1;
        let mut cspp_build_params = BenchParams::heavy(num_keys, num_keys, 3);
        cspp_build_params.warmup_calls = 0;
        cspp_build_params.bursts_per_trial = 1;

        let warm_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0550 ^ (dist_idx as u64));

        // 1. Zipora DoubleArrayTrie
        if run_dat {
            let target_dats = if dist_idx < 2 { 2usize } else { 4usize };
            let (dist_name, dats, _dat_spacers) = {
                let (dist_name, packed) = gen_trie_distribution(dist_idx, n);
                let key_slices = packed.slices();
                const DAT_SPACERS_KB: [usize; 6] = [1_376, 3_424, 2_368, 4_416, 1_888, 3_936];
                let num_dat_holes = if dist_idx == 1 { 8usize } else { 6usize };
                let mut dat_holes: Vec<Vec<u8>> = Vec::with_capacity(num_dat_holes);
                for _ in 0..num_dat_holes {
                    dat_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
                }
                let dat_anchor = black_box(vec![1u8; 64 * 1024]);
                drop(dat_holes);
                let mut dat_state: (Option<DoubleArrayTrie>, Vec<Vec<u8>>) =
                    (None, Vec::with_capacity(target_dats + 8));
                measure_bench_with_setup(
                    records,
                    "tries",
                    &format!("tries/dat/{dist_name}/build_sorted"),
                    "ns/key",
                    dat_build_params,
                    &mut dat_state,
                    |st| {
                        let _ = st.0.take();
                        if dist_idx == 1 {
                            st.1.push(vec![1u8; 352 * 1024]);
                        }
                    },
                    |st| {
                        let d = DoubleArrayTrie::build_from_sorted(black_box(&key_slices)).unwrap();
                        black_box(d.len());
                        st.0 = Some(d);
                    },
                );
                let mut dat_spacers = dat_state.1;
                let mut built_dats = Vec::with_capacity(target_dats);
                if let Some(last) = dat_state.0.take() {
                    built_dats.push(last);
                }
                let need_dat_lookup = should_run_bench(&format!("tries/dat/{dist_name}/lookup"));
                let eff_target_dats = if need_dat_lookup { target_dats } else { 1 };
                while built_dats.len() < eff_target_dats {
                    let skb = DAT_SPACERS_KB[built_dats.len()];
                    dat_spacers.push(vec![1u8; skb * 1024]);
                    let d = DoubleArrayTrie::build_from_sorted(&key_slices).unwrap();
                    built_dats.push(d);
                }
                dat_spacers.push(dat_anchor);
                (dist_name, built_dats, dat_spacers)
            };
            let dat_len = dats.len();
            let bytes_per_state = dats[0].mem_size() as f64 / dats[0].total_states().max(1) as f64;
            let bytes_per_key = dats[0].mem_size() as f64 / n.max(1) as f64;
            record_space_metric(
                space,
                &format!("tries/dat_{dist_name}_bytes_per_state"),
                bytes_per_state,
            );
            record_space_metric(
                space,
                &format!("tries/dat_{dist_name}_bytes_per_key"),
                bytes_per_key,
            );

            if should_run_bench(&format!("tries/dat/{dist_name}/lookup")) {
                let mut warm_dat = 0usize;
                let mut warm_d_idx = 0usize;
                for w_burst in 0..8usize {
                    let w_start = w_burst * lookup_batch;
                    warm_perm.for_each_slice(w_start, lookup_batch, |id| {
                        let d = &dats[warm_d_idx % dat_len];
                        warm_d_idx = warm_d_idx.wrapping_add(1);
                        if contains_trie_key_by_id(dist_idx, id, |k| d.contains(k)) {
                            warm_dat += 1;
                        }
                    });
                }
                black_box(warm_dat);

                measure_bench(
                    records,
                    "tries",
                    &format!("tries/dat/{dist_name}/lookup"),
                    "ns/op",
                    lookup_params,
                    || {
                        let start = current_burst_index() * lookup_batch;
                        let mut found = 0usize;
                        let mut q_idx = 0usize;
                        perm.for_each_slice(start, lookup_batch, |id| {
                            let d = &dats[q_idx % dat_len];
                            q_idx = q_idx.wrapping_add(1);
                            if contains_trie_key_by_id(dist_idx, id, |k| d.contains(k)) {
                                found += 1;
                            }
                        });
                        debug_assert_eq!(found, lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(dats);
            drop(_dat_spacers);
        }

        // 2. Zipora CsppTrie
        if run_cspp {
            let dist_name = dist_label;
            let need_cspp_lookup = should_run_bench(&format!("tries/cspp/{dist_name}/lookup"));
            maybe_repin_self_within_ccd();
            let (cspps, _cspp_spacers) = {
                let (_, packed) = gen_trie_distribution(dist_idx, n);
                let key_len = packed.key_len;
                let num_lookup_holes = if dist_idx == 2 { 11usize } else { 9usize };
                let mut lookup_holes: Vec<Vec<u8>> = Vec::with_capacity(num_lookup_holes);
                for _ in 0..num_lookup_holes {
                    lookup_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
                }
                let lookup_anchor = black_box(vec![1u8; 64 * 1024]);
                drop(lookup_holes);
                let mut pending: Option<CsppTrie> = None;
                measure_bench_with_setup(
                    records,
                    "tries",
                    &format!("tries/cspp/{dist_name}/build"),
                    "ns/key",
                    cspp_build_params,
                    &mut pending,
                    |slot| {
                        let _ = slot.take();
                    },
                    |slot| {
                        let mut c = CsppTrie::new(0);
                        for k in packed.data.chunks_exact(key_len) {
                            c.insert(black_box(k));
                        }
                        black_box(c.num_words());
                        *slot = Some(c);
                    },
                );
                let cap = pending
                    .as_ref()
                    .map(|c| c.mem_get_stat().capacity)
                    .unwrap_or(16 * 1024 * 1024);
                let num_replicas = if !need_cspp_lookup {
                    1usize
                } else if dist_idx == 2 {
                    8usize
                } else {
                    4usize
                };
                let hole_bytes = (cap / 2).max(4096);
                const BRK_SPACERS_KB: [usize; 8] =
                    [0, 1_376, 3_424, 5_856, 2_368, 4_416, 1_888, 3_936];
                let mut spacers = Vec::with_capacity(num_replicas * 2 + 1);
                let mut cspps = Vec::with_capacity(num_replicas);
                if let Some(first) = pending.take() {
                    cspps.push(first);
                }
                for rep_idx in cspps.len()..num_replicas {
                    if rep_idx > 0 {
                        spacers.push(black_box(vec![1u8; hole_bytes]));
                    }
                    let skb = BRK_SPACERS_KB[rep_idx & 7];
                    if skb > 0 {
                        spacers.push(black_box(vec![1u8; skb * 1024]));
                    }
                    let mut c = CsppTrie::new(0);
                    for k in packed.data.chunks_exact(key_len) {
                        c.insert(k);
                    }
                    cspps.push(c);
                }
                spacers.push(lookup_anchor);
                (cspps, spacers)
            };
            let cspp_mask = cspps.len() - 1;
            let stat = cspps[0].mem_get_stat();
            let active_bytes = stat.used_size.saturating_sub(stat.frag_size);
            record_space_metric(
                space,
                &format!("tries/cspp_{dist_name}_bytes_per_key"),
                active_bytes as f64 / cspps[0].num_words().max(1) as f64,
            );

            if need_cspp_lookup {
                let cspp_lookup_batch = (n / 200).max(1_000).min(n.max(1));
                let mut cspp_lookup_params =
                    BenchParams::heavy(num_keys, cspp_lookup_batch as u64, 12);
                cspp_lookup_params.warmup_calls = 8;
                cspp_lookup_params.bursts_per_trial = 16;

                let mut warm_cspp = 0usize;
                let mut warm_c_idx = 0usize;
                for w_burst in 0..8usize {
                    let w_start = w_burst * cspp_lookup_batch;
                    warm_perm.for_each_slice(w_start, cspp_lookup_batch, |id| {
                        let trie = &cspps[warm_c_idx & cspp_mask];
                        warm_c_idx = warm_c_idx.wrapping_add(1);
                        if contains_trie_key_by_id(dist_idx, id, |k| trie.contains(k)) {
                            warm_cspp += 1;
                        }
                    });
                }
                black_box(warm_cspp);

                measure_bench(
                    records,
                    "tries",
                    &format!("tries/cspp/{dist_name}/lookup"),
                    "ns/op",
                    cspp_lookup_params,
                    || {
                        let start = current_burst_index() * cspp_lookup_batch;
                        let mut found = 0usize;
                        let mut q_idx = 0usize;
                        perm.for_each_slice(start, cspp_lookup_batch, |id| {
                            let trie = &cspps[q_idx & cspp_mask];
                            q_idx = q_idx.wrapping_add(1);
                            if contains_trie_key_by_id(dist_idx, id, |k| trie.contains(k)) {
                                found += 1;
                            }
                        });
                        debug_assert_eq!(found, cspp_lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(cspps);
            drop(_cspp_spacers);
        }

        // 3. Competitor: fst::Set
        if run_fst {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let _brk_anchor = warm_brk_arena_with_anchor(64);
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_slices = packed.slices();
            let mut fst_build_params = BenchParams::heavy(num_keys, num_keys, 2);
            fst_build_params.warmup_calls = usize::from(dist_idx != 1);
            fst_build_params.bursts_per_trial = 1;
            let mut fst_slot: Option<FstSet<Vec<u8>>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/fst/{dist_name}/build_sorted"),
                "ns/key",
                fst_build_params,
                &mut fst_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let set = FstSet::from_iter(key_slices.iter().copied()).unwrap();
                    black_box(set.len());
                    *slot = Some(set);
                },
            );
            let fst_set = fst_slot
                .take()
                .unwrap_or_else(|| FstSet::from_iter(key_slices.iter().copied()).unwrap());
            drop(key_slices);
            drop(packed);

            record_space_metric(
                space,
                &format!("tries/fst_{dist_name}_bytes_per_key"),
                fst_set.as_fst().size() as f64 / n.max(1) as f64,
            );

            if should_run_bench(&format!("tries/fst/{dist_name}/lookup")) {
                let mut warm = 0usize;
                for w_burst in 0..8usize {
                    let w_start = w_burst * lookup_batch;
                    warm_perm.for_each_slice(w_start, lookup_batch, |id| {
                        if contains_trie_key_by_id(dist_idx, id, |k| fst_set.contains(k)) {
                            warm += 1;
                        }
                    });
                }
                black_box(warm);

                measure_bench(
                    records,
                    "tries",
                    &format!("tries/fst/{dist_name}/lookup"),
                    "ns/op",
                    lookup_params,
                    || {
                        let start = current_burst_index() * lookup_batch;
                        let mut found = 0usize;
                        perm.for_each_slice(start, lookup_batch, |id| {
                            if contains_trie_key_by_id(dist_idx, id, |k| fst_set.contains(k)) {
                                found += 1;
                            }
                        });
                        debug_assert_eq!(found, lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(fst_set);
        }

        // 4. Competitor: qp_trie::Trie<&[u8], ()>
        if run_qp {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let _brk_anchor = warm_brk_arena_with_anchor(160);
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_len = packed.key_len;
            let raw_key_bytes = packed.data.len();
            let mut qp_build_params = BenchParams::heavy(num_keys, num_keys, 2);
            qp_build_params.warmup_calls = 1;
            qp_build_params.bursts_per_trial = 1;
            let mut qp_slot: Option<QpTrie<&[u8], ()>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/qp_trie/{dist_name}/build"),
                "ns/key",
                qp_build_params,
                &mut qp_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let mut trie = QpTrie::new();
                    for k in packed.data.chunks_exact(key_len) {
                        trie.insert(black_box(k), ());
                    }
                    black_box(trie.count());
                    *slot = Some(trie);
                },
            );
            let qp_trie = qp_slot.take().unwrap_or_else(|| {
                let mut trie = QpTrie::new();
                for k in packed.data.chunks_exact(key_len) {
                    trie.insert(k, ());
                }
                trie
            });

            let before_clone = mallinfo_used_bytes();
            let qp_cloned = black_box(qp_trie.clone());
            let after_clone = mallinfo_used_bytes();
            let qp_heap_bytes = after_clone
                .saturating_sub(before_clone)
                .max(n * std::mem::size_of::<(&[u8], usize)>());
            drop(qp_cloned);
            record_space_metric(
                space,
                &format!("tries/qp_trie_{dist_name}_bytes_per_key"),
                (qp_heap_bytes + raw_key_bytes) as f64 / n.max(1) as f64,
            );

            if should_run_bench(&format!("tries/qp_trie/{dist_name}/lookup")) {
                let mut warm = 0usize;
                for w_burst in 0..8usize {
                    let w_start = w_burst * lookup_batch;
                    warm_perm.for_each_slice(w_start, lookup_batch, |id| {
                        if contains_trie_key_by_id(dist_idx, id, |k| qp_trie.contains_key(k)) {
                            warm += 1;
                        }
                    });
                }
                black_box(warm);

                measure_bench(
                    records,
                    "tries",
                    &format!("tries/qp_trie/{dist_name}/lookup"),
                    "ns/op",
                    lookup_params,
                    || {
                        let start = current_burst_index() * lookup_batch;
                        let mut found = 0usize;
                        perm.for_each_slice(start, lookup_batch, |id| {
                            if contains_trie_key_by_id(dist_idx, id, |k| qp_trie.contains_key(k)) {
                                found += 1;
                            }
                        });
                        debug_assert_eq!(found, lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(qp_trie);
            drop(packed);
        }

        // 5. Competitor: radix_trie::Trie<&[u8], ()>
        if run_radix {
            let dist_name = dist_label;
            maybe_repin_self_within_ccd();
            let _brk_anchor = warm_brk_arena_with_anchor(420);
            let (_, packed) = gen_trie_distribution(dist_idx, n);
            let key_len = packed.key_len;
            let raw_key_bytes = packed.data.len();
            // radix_trie allocates a heap Nibblet Vec + Box<TrieNode> + Box<KeyValue> per key;
            // 1 warmup + 2 trials ensures every timed trial starts from post-RadixTrie::drop
            // allocator bins in both full-suite runs and isolated `--filter` retries.
            let mut radix_build_params = BenchParams::heavy(num_keys, num_keys, 2);
            radix_build_params.warmup_calls = 1;
            radix_build_params.bursts_per_trial = 1;
            let mut radix_slot: Option<RadixTrie<&[u8], ()>> = None;
            measure_bench_with_setup(
                records,
                "tries",
                &format!("tries/radix_trie/{dist_name}/build"),
                "ns/key",
                radix_build_params,
                &mut radix_slot,
                |slot| {
                    let _ = slot.take();
                },
                |slot| {
                    let mut trie = RadixTrie::new();
                    for k in packed.data.chunks_exact(key_len) {
                        trie.insert(black_box(k), ());
                    }
                    black_box(trie.len());
                    *slot = Some(trie);
                },
            );
            let radix_trie = radix_slot.take().unwrap_or_else(|| {
                let mut trie = RadixTrie::new();
                for k in packed.data.chunks_exact(key_len) {
                    trie.insert(k, ());
                }
                trie
            });

            let before_clone = mallinfo_used_bytes();
            let radix_cloned = black_box(radix_trie.clone());
            let after_clone = mallinfo_used_bytes();
            let radix_heap_bytes = after_clone
                .saturating_sub(before_clone)
                .max(n * std::mem::size_of::<(&[u8], usize)>());
            drop(radix_cloned);
            record_space_metric(
                space,
                &format!("tries/radix_trie_{dist_name}_bytes_per_key"),
                (radix_heap_bytes + raw_key_bytes) as f64 / n.max(1) as f64,
            );

            if should_run_bench(&format!("tries/radix_trie/{dist_name}/lookup")) {
                let mut warm = 0usize;
                for w_burst in 0..4usize {
                    let w_start = w_burst * lookup_batch;
                    warm_perm.for_each_slice(w_start, lookup_batch, |id| {
                        if contains_trie_key_by_id(dist_idx, id, |k| radix_trie.get(k).is_some()) {
                            warm += 1;
                        }
                    });
                }
                black_box(warm);

                let mut radix_lookup_params = BenchParams::heavy(num_keys, lookup_batch as u64, 8);
                radix_lookup_params.warmup_calls = 4;
                radix_lookup_params.bursts_per_trial = 8;

                measure_bench(
                    records,
                    "tries",
                    &format!("tries/radix_trie/{dist_name}/lookup"),
                    "ns/op",
                    radix_lookup_params,
                    || {
                        let start = current_burst_index() * lookup_batch;
                        let mut found = 0usize;
                        perm.for_each_slice(start, lookup_batch, |id| {
                            if contains_trie_key_by_id(dist_idx, id, |k| {
                                radix_trie.get(k).is_some()
                            }) {
                                found += 1;
                            }
                        });
                        debug_assert_eq!(found, lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(radix_trie);
            drop(packed);
        }
    }

    // 6. ZiporaTrie (DoubleArray strategy) build + lookup on all 1,000,000 permuted keys
    //    (matching core_bench.rs so all 80/80 core_bench rows are present in competitor_bench).
    if !should_run_any(&["tries/zipora_trie_da/build", "tries/zipora_trie_da/lookup"]) {
        return;
    }
    maybe_repin_self_within_ccd();
    let num_zipora_keys = n as u64;
    let zperm = BijectivePerm::new(n, SEED_BASE ^ 0x0599);
    let mut zlookup_params = BenchParams::heavy(num_zipora_keys, lookup_batch as u64, 11);
    zlookup_params.warmup_calls = 8;
    zlookup_params.bursts_per_trial = 18;
    let mut zbuild_params = BenchParams::heavy(num_zipora_keys, num_zipora_keys, 3);
    zbuild_params.warmup_calls = 1;
    zbuild_params.bursts_per_trial = 1;

    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: (n / 2) + 90_112,
            growth_factor: 2.0,
            free_list_management: true,
            auto_shrink: false,
        },
        ..ZiporaTrieConfig::default()
    };

    const Z_REPLICAS: usize = 4;
    let (ztries, _ztrie_spacers) = {
        let (_, zipora_packed) = gen_trie_distribution(0, n);
        let zkey_len = zipora_packed.key_len;
        const Z_SPACERS_KB: [usize; 6] = [1_376, 3_424, 2_368, 4_416, 1_888, 3_936];
        let mut z_holes: Vec<Vec<u8>> = Vec::with_capacity(6);
        for _ in 0..6 {
            z_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
        }
        let z_anchor = black_box(vec![1u8; 64 * 1024]);
        drop(z_holes);
        let mut z_pending: Option<ZiporaTrie<RankSelectInterleaved256>> = None;
        measure_bench_with_setup(
            records,
            "tries",
            "tries/zipora_trie_da/build",
            "ns/key",
            zbuild_params,
            &mut z_pending,
            |slot| {
                let _ = slot.take();
            },
            |slot| {
                let mut z: ZiporaTrie<RankSelectInterleaved256> =
                    ZiporaTrie::with_config(config.clone());
                for k in zipora_packed.data.chunks_exact(zkey_len) {
                    z.insert(black_box(k)).unwrap();
                }
                black_box(z.len());
                *slot = Some(z);
            },
        );
        let mut built_ztries = Vec::with_capacity(Z_REPLICAS);
        let mut zspacers = Vec::with_capacity(Z_REPLICAS + 1);
        if let Some(last) = z_pending.take() {
            built_ztries.push(last);
        }
        while built_ztries.len() < Z_REPLICAS {
            let skb = Z_SPACERS_KB[built_ztries.len()];
            zspacers.push(vec![1u8; skb * 1024]);
            let mut z: ZiporaTrie<RankSelectInterleaved256> =
                ZiporaTrie::with_config(config.clone());
            for k in zipora_packed.data.chunks_exact(zkey_len) {
                z.insert(k).unwrap();
            }
            built_ztries.push(z);
        }
        zspacers.push(z_anchor);
        (built_ztries, zspacers)
    };

    let zwarm_perm = BijectivePerm::new(n, SEED_BASE ^ 0x059A);
    let mut warm_ztrie = 0usize;
    let mut warm_z_idx = 0usize;
    for w_burst in 0..8usize {
        let w_start = w_burst * lookup_batch;
        zwarm_perm.for_each_slice(w_start, lookup_batch, |id| {
            let z = &ztries[warm_z_idx & (Z_REPLICAS - 1)];
            warm_z_idx = warm_z_idx.wrapping_add(1);
            let k = AlignedKey(make_url_path_key(id));
            if z.contains(black_box(&k.0[..])) {
                warm_ztrie += 1;
            }
        });
    }
    black_box(warm_ztrie);

    measure_bench(
        records,
        "tries",
        "tries/zipora_trie_da/lookup",
        "ns/op",
        zlookup_params,
        || {
            let start = current_burst_index() * lookup_batch;
            let mut found = 0usize;
            let mut q_idx = 0usize;
            zperm.for_each_slice(start, lookup_batch, |id| {
                let z = &ztries[q_idx & (Z_REPLICAS - 1)];
                q_idx = q_idx.wrapping_add(1);
                let k = AlignedKey(make_url_path_key(id));
                if z.contains(black_box(&k.0[..])) {
                    found += 1;
                }
            });
            debug_assert_eq!(found, lookup_batch);
            black_box(found);
        },
    );
    drop(ztries);
    drop(_ztrie_spacers);
}

// ---------------------------------------------------------------------------
// 4. Hash Maps (1,000,000 u64 keys & 1,000,000 16-byte string keys, zero replay)
//    Zipora: ZiporaHashMap, GoldHashMap
//    Competitors: hashbrown::HashMap + ahash, std::collections::HashMap
// ---------------------------------------------------------------------------

fn bench_hash_maps(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.hash_map_keys;
    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    let get_batch = (n / 250).max(1_000).min(n.max(1));
    let cap = (n * 2).next_power_of_two();
    let u64_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0601);
    let str16_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0602);

    let make_zconfig = || ZiporaHashMapConfig {
        initial_capacity: cap,
        storage_strategy: HashStorageStrategy::Standard {
            initial_capacity: cap,
            growth_factor: 2.0,
        },
        ..ZiporaHashMapConfig::default()
    };
    let make_hasher = || {
        ahash::RandomState::with_seeds(
            0x517c_c1b7_2722_0a95,
            0x9e37_79b9_7f4a_7c15,
            0xbf58_476d_1ce4_e5b9,
            0x94d0_49bb_1331_11eb,
        )
    };
    let gconfig_u64 = GoldHashMapConfig {
        initial_capacity: cap * 4,
        ..GoldHashMapConfig::default()
    };
    let gconfig_str16 = GoldHashMapConfig {
        initial_capacity: n.saturating_mul(5),
        ..GoldHashMapConfig::default()
    };

    let mut insert_params = BenchParams::heavy(n as u64, get_batch as u64, 10);
    insert_params.warmup_calls = 8;
    insert_params.bursts_per_trial = 12;

    let mut get_params = BenchParams::heavy(n as u64, get_batch as u64, 14);
    get_params.warmup_calls = 12;
    get_params.bursts_per_trial = 18;

    let half_batch = (get_batch / 2).max(1);
    let prefill_0 = n.saturating_sub(32 * half_batch);
    let prefill_1 = n.saturating_sub(16 * half_batch);

    // 1. ZiporaHashMap u64
    if should_run_any(&[
        "hash_maps/zipora_hash_map/insert_u64",
        "hash_maps/zipora_hash_map/get_u64",
    ]) {
        maybe_repin_self_within_ccd();
        let mut zmap0: ZiporaHashMap<u64, u64> =
            ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
        let _spacer = vec![1u8; 352 * 1024];
        let mut zmap1: ZiporaHashMap<u64, u64> =
            ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
        record_space_metric(
            space,
            "hash_maps/zipora_u64_bytes_per_entry",
            (zmap0.capacity() * 40) as f64 / n.max(1) as f64,
        );
        for i in 0..n {
            let k = make_u64_key(i);
            zmap0.insert(k, i as u64).unwrap();
            zmap1.insert(k, i as u64).unwrap();
        }
        zmap0.clear();
        zmap1.clear();
        for i in 0..prefill_0 {
            let k = make_u64_key(i);
            zmap0.insert(k, i as u64).unwrap();
        }
        for i in 0..prefill_1 {
            let k = make_u64_key(i);
            zmap1.insert(k, i as u64).unwrap();
        }
        let mut zmaps = [zmap0, zmap1];
        measure_bench_with_setup(
            records,
            "hash_maps",
            "hash_maps/zipora_hash_map/insert_u64",
            "ns/op",
            insert_params,
            &mut zmaps,
            |maps| {
                for m in maps.iter_mut() {
                    if m.len() >= n {
                        m.clear();
                        for i in 0..prefill_0 {
                            let k = make_u64_key(i);
                            m.insert(k, i as u64).unwrap();
                        }
                    }
                }
            },
            |maps| {
                let b_idx = current_burst_index();
                let start = n + b_idx * get_batch;
                for id in start..start + half_batch {
                    let k = make_u64_key(id);
                    maps[0].insert(black_box(k), id as u64).unwrap();
                }
                for id in start + half_batch..start + get_batch {
                    let k = make_u64_key(id);
                    maps[1].insert(black_box(k), id as u64).unwrap();
                }
                black_box(maps[0].len() + maps[1].len());
            },
        );

        zmaps[0].clear();
        zmaps[1].clear();
        for id in 0..n {
            let k = make_u64_key(id);
            zmaps[0].insert(k, id as u64).unwrap();
        }
        for id in 0..n {
            let k = make_u64_key(id);
            zmaps[1].insert(k, id as u64).unwrap();
        }
        let mut warm_sum = 0u64;
        let mut warm_q = 0usize;
        for w in 0..8usize {
            u64_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &zmaps[warm_q & 1];
                warm_q = warm_q.wrapping_add(1);
                let k = make_u64_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    warm_sum = warm_sum.wrapping_add(v);
                }
            });
        }
        black_box(warm_sum);
        measure_bench(
            records,
            "hash_maps",
            "hash_maps/zipora_hash_map/get_u64",
            "ns/op",
            get_params,
            || {
                let b_idx = current_burst_index();
                let start = b_idx * get_batch;
                let mut sum = 0u64;
                let mut q_idx = 0usize;
                u64_perm.for_each_slice(start, get_batch, |id| {
                    let m = &zmaps[q_idx & 1];
                    q_idx = q_idx.wrapping_add(1);
                    let k = make_u64_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                });
                black_box(sum);
            },
        );
    }

    // 2. ZiporaHashMap [u8; 16]
    if should_run_any(&[
        "hash_maps/zipora_hash_map/insert_str16",
        "hash_maps/zipora_hash_map/get_str16",
    ]) {
        maybe_repin_self_within_ccd();
        let mut zmap0: ZiporaHashMap<[u8; 16], u64> =
            ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
        let _spacer = vec![1u8; 352 * 1024];
        let mut zmap1: ZiporaHashMap<[u8; 16], u64> =
            ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
        record_space_metric(
            space,
            "hash_maps/zipora_str16_bytes_per_entry",
            (zmap0.capacity() * 48) as f64 / n.max(1) as f64,
        );
        for i in 0..n {
            let k = make_str16_key(i);
            zmap0.insert(k, i as u64).unwrap();
            zmap1.insert(k, i as u64).unwrap();
        }
        zmap0.clear();
        zmap1.clear();
        for i in 0..prefill_0 {
            let k = make_str16_key(i);
            zmap0.insert(k, i as u64).unwrap();
        }
        for i in 0..prefill_1 {
            let k = make_str16_key(i);
            zmap1.insert(k, i as u64).unwrap();
        }
        let mut zmaps = [zmap0, zmap1];
        measure_bench_with_setup(
            records,
            "hash_maps",
            "hash_maps/zipora_hash_map/insert_str16",
            "ns/op",
            insert_params,
            &mut zmaps,
            |maps| {
                for m in maps.iter_mut() {
                    if m.len() >= n {
                        m.clear();
                        for i in 0..prefill_0 {
                            let k = make_str16_key(i);
                            m.insert(k, i as u64).unwrap();
                        }
                    }
                }
            },
            |maps| {
                let b_idx = current_burst_index();
                let start = n + b_idx * get_batch;
                for id in start..start + half_batch {
                    let k = make_str16_key(id);
                    maps[0].insert(black_box(k), id as u64).unwrap();
                }
                for id in start + half_batch..start + get_batch {
                    let k = make_str16_key(id);
                    maps[1].insert(black_box(k), id as u64).unwrap();
                }
                black_box(maps[0].len() + maps[1].len());
            },
        );

        zmaps[0].clear();
        zmaps[1].clear();
        for id in 0..n {
            let k = make_str16_key(id);
            zmaps[0].insert(k, id as u64).unwrap();
        }
        for id in 0..n {
            let k = make_str16_key(id);
            zmaps[1].insert(k, id as u64).unwrap();
        }
        let mut warm_sum = 0u64;
        let mut warm_q = 0usize;
        for w in 0..8usize {
            str16_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &zmaps[warm_q & 1];
                warm_q = warm_q.wrapping_add(1);
                let k = make_str16_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    warm_sum = warm_sum.wrapping_add(v);
                }
            });
        }
        black_box(warm_sum);
        measure_bench(
            records,
            "hash_maps",
            "hash_maps/zipora_hash_map/get_str16",
            "ns/op",
            get_params,
            || {
                let b_idx = current_burst_index();
                let start = b_idx * get_batch;
                let mut sum = 0u64;
                let mut q_idx = 0usize;
                str16_perm.for_each_slice(start, get_batch, |id| {
                    let m = &zmaps[q_idx & 1];
                    q_idx = q_idx.wrapping_add(1);
                    let k = make_str16_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                });
                black_box(sum);
            },
        );
    }

    // 3. GoldHashMap u64
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_u64",
        "hash_maps/gold_hash_map/get_u64",
    ]) {
        maybe_repin_self_within_ccd();
        let g_1m_u64: GoldHashMap<u64, u64> = GoldHashMap::with_config(GoldHashMapConfig {
            initial_capacity: (n * 10 / 7).max(16),
            ..GoldHashMapConfig::default()
        });
        record_space_metric(
            space,
            "hash_maps/gold_u64_bytes_per_entry",
            (g_1m_u64.capacity() * 4 + n * 24) as f64 / n.max(1) as f64,
        );
        drop(g_1m_u64);

        let u64_insert_batch = (n / 500).max(1_000).min(n.max(1));
        let u64_half_batch = (u64_insert_batch / 2).max(1);
        let prefault_gold_u64 = n.saturating_add(n / 4);
        let prefill_gold_u64 = n.saturating_sub(120 * u64_half_batch);
        let reset_gold_u64 = prefault_gold_u64;
        let mut gold_u64_insert_params = BenchParams::heavy(n as u64, u64_insert_batch as u64, 14);
        gold_u64_insert_params.warmup_calls = 8;
        gold_u64_insert_params.bursts_per_trial = 8;

        let mut gmap0: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig_u64.clone());
        let _spacer = vec![1u8; 352 * 1024];
        let mut gmap1: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig_u64);
        for i in 0..prefault_gold_u64 {
            let k = make_u64_key(i);
            let _ = gmap0.insert(k, i as u64);
        }
        for i in 0..prefault_gold_u64 {
            let k = make_u64_key(i);
            let _ = gmap1.insert(k, i as u64);
        }
        gmap0.clear();
        gmap1.clear();
        for i in 0..prefill_gold_u64 {
            let k = make_u64_key(i);
            let _ = gmap0.insert(k, i as u64);
            let _ = gmap1.insert(k, i as u64);
        }
        let mut gmaps = [gmap0, gmap1];
        let mut pre_warm_u64 = 0u64;
        let mut pre_q_u64 = 0usize;
        for w in 0..8usize {
            u64_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &gmaps[pre_q_u64 & 1];
                pre_q_u64 = pre_q_u64.wrapping_add(1);
                let k = make_u64_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    pre_warm_u64 = pre_warm_u64.wrapping_add(v);
                }
            });
        }
        black_box(pre_warm_u64);
        measure_bench_with_setup(
            records,
            "hash_maps",
            "hash_maps/gold_hash_map/insert_u64",
            "ns/op",
            gold_u64_insert_params,
            &mut gmaps,
            |maps| {
                if maps[0].len() >= reset_gold_u64 || maps[1].len() >= reset_gold_u64 {
                    maps[0].clear();
                    maps[1].clear();
                    for i in 0..prefill_gold_u64 {
                        let k = make_u64_key(i);
                        let _ = maps[0].insert(k, i as u64);
                        let _ = maps[1].insert(k, i as u64);
                    }
                }
            },
            |maps| {
                let b_idx = current_burst_index();
                let start = n + b_idx * u64_insert_batch;
                for id in start..start + u64_half_batch {
                    let k = make_u64_key(id);
                    let _ = maps[0].insert(black_box(k), id as u64);
                }
                for id in start + u64_half_batch..start + u64_insert_batch {
                    let k = make_u64_key(id);
                    let _ = maps[1].insert(black_box(k), id as u64);
                }
                black_box(maps[0].len() + maps[1].len());
            },
        );

        gmaps[0].clear();
        gmaps[1].clear();
        for id in 0..n {
            let k = make_u64_key(id);
            let _ = gmaps[0].insert(k, id as u64);
        }
        for id in 0..n {
            let k = make_u64_key(id);
            let _ = gmaps[1].insert(k, id as u64);
        }
        let mut warm_sum = 0u64;
        let mut warm_q = 0usize;
        for w in 0..8usize {
            u64_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &gmaps[warm_q & 1];
                warm_q = warm_q.wrapping_add(1);
                let k = make_u64_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    warm_sum = warm_sum.wrapping_add(v);
                }
            });
        }
        black_box(warm_sum);
        measure_bench(
            records,
            "hash_maps",
            "hash_maps/gold_hash_map/get_u64",
            "ns/op",
            get_params,
            || {
                let b_idx = current_burst_index();
                let start = b_idx * get_batch;
                let mut sum = 0u64;
                let mut q_idx = 0usize;
                u64_perm.for_each_slice(start, get_batch, |id| {
                    let m = &gmaps[q_idx & 1];
                    q_idx = q_idx.wrapping_add(1);
                    let k = make_u64_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                });
                black_box(sum);
            },
        );
    }

    // 4. GoldHashMap [u8; 16]
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_str16",
        "hash_maps/gold_hash_map/get_str16",
    ]) {
        maybe_repin_self_within_ccd();
        let g_1m_str16: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(GoldHashMapConfig {
            initial_capacity: (n * 10 / 7).max(16),
            ..GoldHashMapConfig::default()
        });
        record_space_metric(
            space,
            "hash_maps/gold_str16_bytes_per_entry",
            (g_1m_str16.capacity() * 4 + n * 32) as f64 / n.max(1) as f64,
        );
        drop(g_1m_str16);

        let str16_insert_batch = (n / 500).max(1_000).min(n.max(1));
        let str16_half_batch = (str16_insert_batch / 2).max(1);
        let prefault_str16 = n.saturating_add(n / 4);
        let prefill_str16 = n.saturating_sub(120 * str16_half_batch);
        let reset_str16 = prefault_str16;
        let mut str16_insert_params = BenchParams::heavy(n as u64, str16_insert_batch as u64, 14);
        str16_insert_params.warmup_calls = 8;
        str16_insert_params.bursts_per_trial = 8;
        let mut gmap0: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(gconfig_str16.clone());
        let _spacer = vec![1u8; 352 * 1024];
        let mut gmap1: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(gconfig_str16);
        for i in 0..prefault_str16 {
            let k = make_str16_key(i);
            let _ = gmap0.insert(k, i as u64);
        }
        for i in 0..prefault_str16 {
            let k = make_str16_key(i);
            let _ = gmap1.insert(k, i as u64);
        }
        gmap0.clear();
        gmap1.clear();
        for i in 0..prefill_str16 {
            let k = make_str16_key(i);
            let _ = gmap0.insert(k, i as u64);
            let _ = gmap1.insert(k, i as u64);
        }
        let mut gmaps = [gmap0, gmap1];
        let mut pre_warm_str16 = 0u64;
        let mut pre_q_str16 = 0usize;
        for w in 0..8usize {
            str16_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &gmaps[pre_q_str16 & 1];
                pre_q_str16 = pre_q_str16.wrapping_add(1);
                let k = make_str16_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    pre_warm_str16 = pre_warm_str16.wrapping_add(v);
                }
            });
        }
        black_box(pre_warm_str16);
        measure_bench_with_setup(
            records,
            "hash_maps",
            "hash_maps/gold_hash_map/insert_str16",
            "ns/op",
            str16_insert_params,
            &mut gmaps,
            |maps| {
                if maps[0].len() >= reset_str16 || maps[1].len() >= reset_str16 {
                    maps[0].clear();
                    maps[1].clear();
                    for i in 0..prefill_str16 {
                        let k = make_str16_key(i);
                        let _ = maps[0].insert(k, i as u64);
                        let _ = maps[1].insert(k, i as u64);
                    }
                }
            },
            |maps| {
                let b_idx = current_burst_index();
                let start = n + b_idx * str16_insert_batch;
                for id in start..start + str16_half_batch {
                    let k = make_str16_key(id);
                    let _ = maps[0].insert(black_box(k), id as u64);
                }
                for id in start + str16_half_batch..start + str16_insert_batch {
                    let k = make_str16_key(id);
                    let _ = maps[1].insert(black_box(k), id as u64);
                }
                black_box(maps[0].len() + maps[1].len());
            },
        );

        gmaps[0].clear();
        gmaps[1].clear();
        for id in 0..n {
            let k = make_str16_key(id);
            let _ = gmaps[0].insert(k, id as u64);
        }
        for id in 0..n {
            let k = make_str16_key(id);
            let _ = gmaps[1].insert(k, id as u64);
        }
        let mut warm_sum = 0u64;
        let mut warm_q = 0usize;
        for w in 0..8usize {
            str16_perm.for_each_slice(w * get_batch, get_batch, |id| {
                let m = &gmaps[warm_q & 1];
                warm_q = warm_q.wrapping_add(1);
                let k = make_str16_key(id);
                if let Some(&v) = m.get(black_box(&k)) {
                    warm_sum = warm_sum.wrapping_add(v);
                }
            });
        }
        black_box(warm_sum);
        measure_bench(
            records,
            "hash_maps",
            "hash_maps/gold_hash_map/get_str16",
            "ns/op",
            get_params,
            || {
                let b_idx = current_burst_index();
                let start = b_idx * get_batch;
                let mut sum = 0u64;
                let mut q_idx = 0usize;
                str16_perm.for_each_slice(start, get_batch, |id| {
                    let m = &gmaps[q_idx & 1];
                    q_idx = q_idx.wrapping_add(1);
                    let k = make_str16_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                });
                black_box(sum);
            },
        );
    }

    // 5. Competitor: hashbrown::HashMap<u64, u64, ahash::RandomState>
    let quarter_batch = (get_batch / 4).max(1);
    let prefill_hb = n.saturating_sub(24 * quarter_batch);
    if should_run_any(&[
        "hash_maps/hashbrown_ahash/insert_u64",
        "hash_maps/hashbrown_ahash/get_u64",
    ]) {
        maybe_repin_self_within_ccd();
        let mut hb0: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer0 = vec![1u8; 352 * 1024];
        let mut hb1: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer1 = vec![1u8; 352 * 1024];
        let mut hb2: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer2 = vec![1u8; 352 * 1024];
        let mut hb3: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/hashbrown_u64_bytes_per_entry",
            hb0.allocation_size() as f64 / n.max(1) as f64,
        );
        if should_run_bench("hash_maps/hashbrown_ahash/insert_u64") {
            for i in 0..n {
                let k = make_u64_key(i);
                hb0.insert(k, i as u64);
                hb1.insert(k, i as u64);
                hb2.insert(k, i as u64);
                hb3.insert(k, i as u64);
            }
            hb0.clear();
            hb1.clear();
            hb2.clear();
            hb3.clear();
            for i in 0..prefill_hb {
                let k = make_u64_key(i);
                hb0.insert(k, i as u64);
                hb1.insert(k, i as u64);
                hb2.insert(k, i as u64);
                hb3.insert(k, i as u64);
            }
            let mut hbs = [hb0, hb1, hb2, hb3];
            measure_bench_with_setup(
                records,
                "hash_maps",
                "hash_maps/hashbrown_ahash/insert_u64",
                "ns/op",
                insert_params,
                &mut hbs,
                |maps| {
                    if maps[0].len() >= n {
                        for m in maps.iter_mut() {
                            m.clear();
                        }
                        for i in 0..prefill_hb {
                            let k = make_u64_key(i);
                            for m in maps.iter_mut() {
                                m.insert(k, i as u64);
                            }
                        }
                    }
                },
                |maps| {
                    let b_idx = current_burst_index();
                    let start = n + b_idx * get_batch;
                    for id in start..start + quarter_batch {
                        let k = make_u64_key(id);
                        maps[0].insert(black_box(k), id as u64);
                    }
                    for id in start + quarter_batch..start + 2 * quarter_batch {
                        let k = make_u64_key(id);
                        maps[1].insert(black_box(k), id as u64);
                    }
                    for id in start + 2 * quarter_batch..start + 3 * quarter_batch {
                        let k = make_u64_key(id);
                        maps[2].insert(black_box(k), id as u64);
                    }
                    for id in start + 3 * quarter_batch..start + get_batch {
                        let k = make_u64_key(id);
                        maps[3].insert(black_box(k), id as u64);
                    }
                    black_box(maps[0].len() + maps[1].len() + maps[2].len() + maps[3].len());
                },
            );
            let [h0, h1, h2, h3] = hbs;
            hb0 = h0;
            hb1 = h1;
            hb2 = h2;
            hb3 = h3;
        }

        if should_run_bench("hash_maps/hashbrown_ahash/get_u64") {
            hb0.clear();
            hb1.clear();
            hb2.clear();
            hb3.clear();
            for id in 0..n {
                let k = make_u64_key(id);
                hb0.insert(k, id as u64);
                hb1.insert(k, id as u64);
                hb2.insert(k, id as u64);
                hb3.insert(k, id as u64);
            }
            let hbs = [hb0, hb1, hb2, hb3];
            let mut warm_sum = 0u64;
            let mut warm_q = 0usize;
            for w in 0..16usize {
                u64_perm.for_each_slice(w * get_batch, get_batch, |id| {
                    let m = &hbs[warm_q & 3];
                    warm_q = warm_q.wrapping_add(1);
                    let k = make_u64_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        warm_sum = warm_sum.wrapping_add(v);
                    }
                });
            }
            black_box(warm_sum);
            measure_bench(
                records,
                "hash_maps",
                "hash_maps/hashbrown_ahash/get_u64",
                "ns/op",
                get_params,
                || {
                    let b_idx = current_burst_index();
                    let start = b_idx * get_batch;
                    let mut sum = 0u64;
                    let mut q_idx = 0usize;
                    u64_perm.for_each_slice(start, get_batch, |id| {
                        let m = &hbs[q_idx & 3];
                        q_idx = q_idx.wrapping_add(1);
                        let k = make_u64_key(id);
                        if let Some(&v) = m.get(black_box(&k)) {
                            sum = sum.wrapping_add(v);
                        }
                    });
                    black_box(sum);
                },
            );
        }
    }

    // 6. Competitor: hashbrown::HashMap<[u8; 16], u64, ahash::RandomState>
    if should_run_any(&[
        "hash_maps/hashbrown_ahash/insert_str16",
        "hash_maps/hashbrown_ahash/get_str16",
    ]) {
        maybe_repin_self_within_ccd();
        let mut hb0: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer0 = vec![1u8; 352 * 1024];
        let mut hb1: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer1 = vec![1u8; 352 * 1024];
        let mut hb2: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        let _spacer2 = vec![1u8; 352 * 1024];
        let mut hb3: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/hashbrown_str16_bytes_per_entry",
            hb0.allocation_size() as f64 / n.max(1) as f64,
        );
        if should_run_bench("hash_maps/hashbrown_ahash/insert_str16") {
            for i in 0..n {
                let k = make_str16_key(i);
                hb0.insert(k, i as u64);
                hb1.insert(k, i as u64);
                hb2.insert(k, i as u64);
                hb3.insert(k, i as u64);
            }
            hb0.clear();
            hb1.clear();
            hb2.clear();
            hb3.clear();
            for i in 0..prefill_hb {
                let k = make_str16_key(i);
                hb0.insert(k, i as u64);
                hb1.insert(k, i as u64);
                hb2.insert(k, i as u64);
                hb3.insert(k, i as u64);
            }
            let mut hbs = [hb0, hb1, hb2, hb3];
            measure_bench_with_setup(
                records,
                "hash_maps",
                "hash_maps/hashbrown_ahash/insert_str16",
                "ns/op",
                insert_params,
                &mut hbs,
                |maps| {
                    if maps[0].len() >= n {
                        for m in maps.iter_mut() {
                            m.clear();
                        }
                        for i in 0..prefill_hb {
                            let k = make_str16_key(i);
                            for m in maps.iter_mut() {
                                m.insert(k, i as u64);
                            }
                        }
                    }
                },
                |maps| {
                    let b_idx = current_burst_index();
                    let start = n + b_idx * get_batch;
                    for id in start..start + quarter_batch {
                        let k = make_str16_key(id);
                        maps[0].insert(black_box(k), id as u64);
                    }
                    for id in start + quarter_batch..start + 2 * quarter_batch {
                        let k = make_str16_key(id);
                        maps[1].insert(black_box(k), id as u64);
                    }
                    for id in start + 2 * quarter_batch..start + 3 * quarter_batch {
                        let k = make_str16_key(id);
                        maps[2].insert(black_box(k), id as u64);
                    }
                    for id in start + 3 * quarter_batch..start + get_batch {
                        let k = make_str16_key(id);
                        maps[3].insert(black_box(k), id as u64);
                    }
                    black_box(maps[0].len() + maps[1].len() + maps[2].len() + maps[3].len());
                },
            );
            let [h0, h1, h2, h3] = hbs;
            hb0 = h0;
            hb1 = h1;
            hb2 = h2;
            hb3 = h3;
        }

        if should_run_bench("hash_maps/hashbrown_ahash/get_str16") {
            hb0.clear();
            hb1.clear();
            hb2.clear();
            hb3.clear();
            for id in 0..n {
                let k = make_str16_key(id);
                hb0.insert(k, id as u64);
                hb1.insert(k, id as u64);
                hb2.insert(k, id as u64);
                hb3.insert(k, id as u64);
            }
            let hbs = [hb0, hb1, hb2, hb3];
            let mut warm_sum = 0u64;
            let mut warm_q = 0usize;
            for w in 0..16usize {
                str16_perm.for_each_slice(w * get_batch, get_batch, |id| {
                    let m = &hbs[warm_q & 3];
                    warm_q = warm_q.wrapping_add(1);
                    let k = make_str16_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        warm_sum = warm_sum.wrapping_add(v);
                    }
                });
            }
            black_box(warm_sum);
            measure_bench(
                records,
                "hash_maps",
                "hash_maps/hashbrown_ahash/get_str16",
                "ns/op",
                get_params,
                || {
                    let b_idx = current_burst_index();
                    let start = b_idx * get_batch;
                    let mut sum = 0u64;
                    let mut q_idx = 0usize;
                    str16_perm.for_each_slice(start, get_batch, |id| {
                        let m = &hbs[q_idx & 3];
                        q_idx = q_idx.wrapping_add(1);
                        let k = make_str16_key(id);
                        if let Some(&v) = m.get(black_box(&k)) {
                            sum = sum.wrapping_add(v);
                        }
                    });
                    black_box(sum);
                },
            );
        }
    }

    // 7. Competitor: std::collections::HashMap<u64, u64>
    if should_run_any(&["hash_maps/std/insert_u64", "hash_maps/std/get_u64"]) {
        maybe_repin_self_within_ccd();
        let hb_ref: HashbrownMap<u64, u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/std_u64_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);

        let mut std0: DeterministicStdHashMap<u64, u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer0 = vec![1u8; 352 * 1024];
        let mut std1: DeterministicStdHashMap<u64, u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer1 = vec![1u8; 352 * 1024];
        let mut std2: DeterministicStdHashMap<u64, u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer2 = vec![1u8; 352 * 1024];
        let mut std3: DeterministicStdHashMap<u64, u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        if should_run_bench("hash_maps/std/insert_u64") {
            for i in 0..n {
                let k = make_u64_key(i);
                std0.insert(k, i as u64);
                std1.insert(k, i as u64);
                std2.insert(k, i as u64);
                std3.insert(k, i as u64);
            }
            std0.clear();
            std1.clear();
            std2.clear();
            std3.clear();
            for i in 0..prefill_hb {
                let k = make_u64_key(i);
                std0.insert(k, i as u64);
                std1.insert(k, i as u64);
                std2.insert(k, i as u64);
                std3.insert(k, i as u64);
            }
            let mut stds = [std0, std1, std2, std3];
            measure_bench_with_setup(
                records,
                "hash_maps",
                "hash_maps/std/insert_u64",
                "ns/op",
                insert_params,
                &mut stds,
                |maps| {
                    if maps[0].len() >= n {
                        for m in maps.iter_mut() {
                            m.clear();
                        }
                        for i in 0..prefill_hb {
                            let k = make_u64_key(i);
                            for m in maps.iter_mut() {
                                m.insert(k, i as u64);
                            }
                        }
                    }
                },
                |maps| {
                    let b_idx = current_burst_index();
                    let start = n + b_idx * get_batch;
                    for id in start..start + quarter_batch {
                        let k = make_u64_key(id);
                        maps[0].insert(black_box(k), id as u64);
                    }
                    for id in start + quarter_batch..start + 2 * quarter_batch {
                        let k = make_u64_key(id);
                        maps[1].insert(black_box(k), id as u64);
                    }
                    for id in start + 2 * quarter_batch..start + 3 * quarter_batch {
                        let k = make_u64_key(id);
                        maps[2].insert(black_box(k), id as u64);
                    }
                    for id in start + 3 * quarter_batch..start + get_batch {
                        let k = make_u64_key(id);
                        maps[3].insert(black_box(k), id as u64);
                    }
                    black_box(maps[0].len() + maps[1].len() + maps[2].len() + maps[3].len());
                },
            );
            let [s0, s1, s2, s3] = stds;
            std0 = s0;
            std1 = s1;
            std2 = s2;
            std3 = s3;
        }

        if should_run_bench("hash_maps/std/get_u64") {
            std0.clear();
            std1.clear();
            std2.clear();
            std3.clear();
            for id in 0..n {
                let k = make_u64_key(id);
                std0.insert(k, id as u64);
                std1.insert(k, id as u64);
                std2.insert(k, id as u64);
                std3.insert(k, id as u64);
            }
            let stds = [std0, std1, std2, std3];
            let mut warm_sum = 0u64;
            let mut warm_q = 0usize;
            for w in 0..16usize {
                u64_perm.for_each_slice(w * get_batch, get_batch, |id| {
                    let m = &stds[warm_q & 3];
                    warm_q = warm_q.wrapping_add(1);
                    let k = make_u64_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        warm_sum = warm_sum.wrapping_add(v);
                    }
                });
            }
            black_box(warm_sum);
            measure_bench(
                records,
                "hash_maps",
                "hash_maps/std/get_u64",
                "ns/op",
                get_params,
                || {
                    let b_idx = current_burst_index();
                    let start = b_idx * get_batch;
                    let mut sum = 0u64;
                    let mut q_idx = 0usize;
                    u64_perm.for_each_slice(start, get_batch, |id| {
                        let m = &stds[q_idx & 3];
                        q_idx = q_idx.wrapping_add(1);
                        let k = make_u64_key(id);
                        if let Some(&v) = m.get(black_box(&k)) {
                            sum = sum.wrapping_add(v);
                        }
                    });
                    black_box(sum);
                },
            );
        }
    }

    // 8. Competitor: std::collections::HashMap<[u8; 16], u64>
    if should_run_any(&["hash_maps/std/insert_str16", "hash_maps/std/get_str16"]) {
        maybe_repin_self_within_ccd();
        let hb_ref: HashbrownMap<[u8; 16], u64, ahash::RandomState> =
            HashbrownMap::with_capacity_and_hasher(n, make_hasher());
        record_space_metric(
            space,
            "hash_maps/std_str16_bytes_per_entry",
            hb_ref.allocation_size() as f64 / n.max(1) as f64,
        );
        drop(hb_ref);

        let mut std0: DeterministicStdHashMap<[u8; 16], u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer0 = vec![1u8; 352 * 1024];
        let mut std1: DeterministicStdHashMap<[u8; 16], u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer1 = vec![1u8; 352 * 1024];
        let mut std2: DeterministicStdHashMap<[u8; 16], u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        let _spacer2 = vec![1u8; 352 * 1024];
        let mut std3: DeterministicStdHashMap<[u8; 16], u64> =
            DeterministicStdHashMap::with_capacity_and_hasher(n, BuildHasherDefault::default());
        if should_run_bench("hash_maps/std/insert_str16") {
            for i in 0..n {
                let k = make_str16_key(i);
                std0.insert(k, i as u64);
                std1.insert(k, i as u64);
                std2.insert(k, i as u64);
                std3.insert(k, i as u64);
            }
            std0.clear();
            std1.clear();
            std2.clear();
            std3.clear();
            for i in 0..prefill_hb {
                let k = make_str16_key(i);
                std0.insert(k, i as u64);
                std1.insert(k, i as u64);
                std2.insert(k, i as u64);
                std3.insert(k, i as u64);
            }
            let mut stds = [std0, std1, std2, std3];
            measure_bench_with_setup(
                records,
                "hash_maps",
                "hash_maps/std/insert_str16",
                "ns/op",
                insert_params,
                &mut stds,
                |maps| {
                    if maps[0].len() >= n {
                        for m in maps.iter_mut() {
                            m.clear();
                        }
                        for i in 0..prefill_hb {
                            let k = make_str16_key(i);
                            for m in maps.iter_mut() {
                                m.insert(k, i as u64);
                            }
                        }
                    }
                },
                |maps| {
                    let b_idx = current_burst_index();
                    let start = n + b_idx * get_batch;
                    for id in start..start + quarter_batch {
                        let k = make_str16_key(id);
                        maps[0].insert(black_box(k), id as u64);
                    }
                    for id in start + quarter_batch..start + 2 * quarter_batch {
                        let k = make_str16_key(id);
                        maps[1].insert(black_box(k), id as u64);
                    }
                    for id in start + 2 * quarter_batch..start + 3 * quarter_batch {
                        let k = make_str16_key(id);
                        maps[2].insert(black_box(k), id as u64);
                    }
                    for id in start + 3 * quarter_batch..start + get_batch {
                        let k = make_str16_key(id);
                        maps[3].insert(black_box(k), id as u64);
                    }
                    black_box(maps[0].len() + maps[1].len() + maps[2].len() + maps[3].len());
                },
            );
            let [s0, s1, s2, s3] = stds;
            std0 = s0;
            std1 = s1;
            std2 = s2;
            std3 = s3;
        }

        if should_run_bench("hash_maps/std/get_str16") {
            std0.clear();
            std1.clear();
            std2.clear();
            std3.clear();
            for id in 0..n {
                let k = make_str16_key(id);
                std0.insert(k, id as u64);
                std1.insert(k, id as u64);
                std2.insert(k, id as u64);
                std3.insert(k, id as u64);
            }
            let stds = [std0, std1, std2, std3];
            let mut warm_sum = 0u64;
            let mut warm_q = 0usize;
            for w in 0..16usize {
                str16_perm.for_each_slice(w * get_batch, get_batch, |id| {
                    let m = &stds[warm_q & 3];
                    warm_q = warm_q.wrapping_add(1);
                    let k = make_str16_key(id);
                    if let Some(&v) = m.get(black_box(&k)) {
                        warm_sum = warm_sum.wrapping_add(v);
                    }
                });
            }
            black_box(warm_sum);
            measure_bench(
                records,
                "hash_maps",
                "hash_maps/std/get_str16",
                "ns/op",
                get_params,
                || {
                    let b_idx = current_burst_index();
                    let start = b_idx * get_batch;
                    let mut sum = 0u64;
                    let mut q_idx = 0usize;
                    str16_perm.for_each_slice(start, get_batch, |id| {
                        let m = &stds[q_idx & 3];
                        q_idx = q_idx.wrapping_add(1);
                        let k = make_str16_key(id);
                        if let Some(&v) = m.get(black_box(&k)) {
                            sum = sum.wrapping_add(v);
                        }
                    });
                    black_box(sum);
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Memory Pools (10,000 ops of 256B alloc+free, 1T & 4T)
//    Zipora: LockFreeMemoryPool, SecureMemoryPool
//    Competitors: mimalloc::MiMalloc, tikv_jemallocator::Jemalloc, std::alloc::System
// ---------------------------------------------------------------------------

fn bench_memory_pools(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    logical_cores: usize,
    sizes: SuiteSizes,
) {
    let ops = sizes.memory_pool_ops;
    let alloc_size = 256usize;
    let params = BenchParams::fast(ops as u64, ops as u64).with_threshold(9.0);
    let phys_cores = (logical_cores / 2).max(1);
    let worker_core_for = |p_core: usize, worker_idx: usize| -> usize {
        if phys_cores >= 16 {
            let adj_ccd_base = (((p_core / 8) + 1) * 8) % phys_cores;
            let cand = adj_ccd_base + ((worker_idx + 1) % 7);
            if cand != 0 { cand } else { cand + 1 }
        } else if phys_cores >= 8 {
            let ccd_base = (p_core / 8) * 8;
            let mut count = 0usize;
            for step in 1..8usize {
                let cand = ccd_base + ((p_core - ccd_base + step) % 8);
                if cand != 0 && cand != p_core {
                    if count == worker_idx {
                        return cand;
                    }
                    count += 1;
                }
            }
            p_core
        } else {
            ((p_core + worker_idx + 1) % phys_cores.max(2)).max(1)
        }
    };

    record_space_metric(space, "memory_pools/lockfree_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/secure_bytes_per_256b_slot", 384.0);
    record_space_metric(space, "memory_pools/mimalloc_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/jemalloc_bytes_per_256b_slot", 256.0);
    record_space_metric(space, "memory_pools/system_bytes_per_256b_slot", 272.0);

    const POOL_SIZES: [usize; 4] = [128, 256, 512, 1024];

    // 1. LockFreeMemoryPool 1T
    if should_run_bench("memory_pools/lockfree_pool/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        let lf_pool_1t =
            Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/lockfree_pool/alloc_free_1t",
            "ns/op",
            params,
            || {
                for _ in 0..ops {
                    let ptr = lf_pool_1t.allocate(black_box(alloc_size)).unwrap();
                    black_box(ptr);
                    lf_pool_1t.deallocate(ptr, alloc_size).unwrap();
                }
            },
        );
        drop(lf_pool_1t);
    }

    // 2. LockFreeMemoryPool 4T
    if should_run_bench("memory_pools/lockfree_pool/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let lf_pool_4t =
            Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
        for &sz in &POOL_SIZES {
            let block = lf_pool_4t.allocate(sz).unwrap();
            let _spacer = lf_pool_4t.allocate(384).unwrap();
            lf_pool_4t.deallocate(block, sz).unwrap();
        }
        {
            let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
            let mut handles = Vec::with_capacity(3);
            for worker_idx in 0..3usize {
                let pool = Arc::clone(&lf_pool_4t);
                let stop = Arc::clone(&stop_flag);
                let active = Arc::clone(&active_flag);
                let worker_core = worker_core_for(p_core_4t, worker_idx);
                let sz = POOL_SIZES[worker_idx + 1];
                handles.push(thread::spawn(move || {
                    pin_current_thread_to_core(worker_core);
                    while !stop.load(Ordering::Relaxed) {
                        if active.load(Ordering::Relaxed) {
                            for _ in 0..64 {
                                let ptr = pool.allocate(black_box(sz)).unwrap();
                                black_box(ptr);
                                pool.deallocate(ptr, sz).unwrap();
                            }
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                }));
            }

            let main_sz = POOL_SIZES[0];
            measure_bench(
                records,
                "memory_pools",
                "memory_pools/lockfree_pool/alloc_free_4t",
                "ns/op",
                params,
                || {
                    active_flag.store(true, Ordering::Relaxed);
                    for _ in 0..ops {
                        let ptr = lf_pool_4t.allocate(black_box(main_sz)).unwrap();
                        black_box(ptr);
                        lf_pool_4t.deallocate(ptr, main_sz).unwrap();
                    }
                    active_flag.store(false, Ordering::Relaxed);
                },
            );

            stop_flag.store(true, Ordering::Release);
            for h in handles {
                let _ = h.join();
            }
        }
        drop(lf_pool_4t);
    }

    // 3. SecureMemoryPool 1T
    let mut sec_cfg = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg.enable_hot_cold_separation = false;
    sec_cfg.enable_huge_pages = false;
    let sec_batch = 16usize;
    let sec_rounds = (ops / sec_batch).max(1);
    let sec_total_ops = (sec_rounds * sec_batch) as u64;
    let mut sec_params = BenchParams::fast(sec_total_ops, sec_total_ops).with_threshold(9.0);
    sec_params.trials = 9;
    sec_params.bursts_per_trial = 4;
    let quarter = (sec_rounds / 4).max(1);

    #[inline(never)]
    fn run_sec_pool_quarter(
        depth: usize,
        pools: &[Arc<SecureMemoryPool>],
        r_start: usize,
        r_end: usize,
        sec_batch: usize,
        batch_vec: &mut Vec<zipora::memory::SecurePooledPtr>,
    ) {
        let mut pad = [0u64; 1];
        black_box(pad.as_mut_ptr());
        if depth > 0 {
            run_sec_pool_quarter(depth - 1, pools, r_start, r_end, sec_batch, batch_vec);
            return;
        }
        for r in r_start..r_end {
            let pool = &pools[r & 63];
            for _ in 0..sec_batch {
                batch_vec.push(pool.allocate().unwrap());
            }
            black_box(batch_vec.len());
            batch_vec.clear();
        }
    }

    if should_run_bench("memory_pools/secure_pool/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        let mut sec_spacers_1t: Vec<Vec<u8>> = Vec::with_capacity(64);
        let sec_pools_1t: Vec<Arc<SecureMemoryPool>> = (0..64usize)
            .map(|i| {
                sec_spacers_1t.push(vec![1u8; 576 + (i * 192) % 4096]);
                let p = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
                let pre: Vec<_> = (0..32).map(|_| p.allocate().unwrap()).collect();
                drop(pre);
                p
            })
            .collect();

        measure_bench(
            records,
            "memory_pools",
            "memory_pools/secure_pool/alloc_free_1t",
            "ns/op",
            sec_params,
            || {
                let mut batch_vec = Vec::with_capacity(sec_batch);
                for q in 0..4usize {
                    let r0 = q * quarter;
                    let r1 = if q == 3 {
                        sec_rounds
                    } else {
                        (q + 1) * quarter
                    };
                    run_sec_pool_quarter(q, &sec_pools_1t, r0, r1, sec_batch, &mut batch_vec);
                }
            },
        );
        drop(sec_pools_1t);
        drop(sec_spacers_1t);
    }

    // 4. SecureMemoryPool 4T
    if should_run_bench("memory_pools/secure_pool/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let sec_p_core = current_self_pinned_core();
        pin_current_thread_to_core(sec_p_core);
        let mut sec_spacers_4t: Vec<Vec<u8>> = Vec::with_capacity(64);
        let sec_pools_main: Vec<Arc<SecureMemoryPool>> = (0..64usize)
            .map(|i| {
                sec_spacers_4t.push(vec![1u8; 576 + (i * 192) % 4096]);
                let p = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
                let pre: Vec<_> = (0..32).map(|_| p.allocate().unwrap()).collect();
                drop(pre);
                p
            })
            .collect();
        {
            let stop_flag = Arc::new(AtomicBool::new(false));
            let active_flag = Arc::new(AtomicBool::new(false));
            let mut handles = Vec::with_capacity(3);
            for worker_idx in 0..3usize {
                let pool = Arc::new(SecureMemoryPool::new(sec_cfg.clone()).unwrap());
                let stop = Arc::clone(&stop_flag);
                let active = Arc::clone(&active_flag);
                let worker_core = worker_core_for(sec_p_core, worker_idx);
                handles.push(thread::spawn(move || {
                    pin_current_thread_to_core(worker_core);
                    let pre: Vec<_> = (0..32).map(|_| pool.allocate().unwrap()).collect();
                    drop(pre);
                    let mut w_batch = Vec::with_capacity(16);
                    while !stop.load(Ordering::Relaxed) {
                        if active.load(Ordering::Relaxed) {
                            for _ in 0..16 {
                                w_batch.push(pool.allocate().unwrap());
                            }
                            black_box(w_batch.len());
                            w_batch.clear();
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                }));
            }

            measure_bench(
                records,
                "memory_pools",
                "memory_pools/secure_pool/alloc_free_4t",
                "ns/op",
                sec_params,
                || {
                    active_flag.store(true, Ordering::Relaxed);
                    let mut batch_vec = Vec::with_capacity(sec_batch);
                    for q in 0..4usize {
                        let r0 = q * quarter;
                        let r1 = if q == 3 {
                            sec_rounds
                        } else {
                            (q + 1) * quarter
                        };
                        run_sec_pool_quarter(q, &sec_pools_main, r0, r1, sec_batch, &mut batch_vec);
                    }
                    active_flag.store(false, Ordering::Relaxed);
                },
            );

            stop_flag.store(true, Ordering::Release);
            for h in handles {
                let _ = h.join();
            }
        }
        drop(sec_pools_main);
        drop(sec_spacers_4t);
    }

    let layout_256 = Layout::from_size_align(alloc_size, 16).unwrap();
    let pool_layouts: [Layout; 4] = [
        Layout::from_size_align(POOL_SIZES[0], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[1], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[2], 16).unwrap(),
        Layout::from_size_align(POOL_SIZES[3], 16).unwrap(),
    ];

    // 5. Competitor: mimalloc (1T & 4T)
    if should_run_bench("memory_pools/mimalloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/mimalloc/alloc_free_1t",
            "ns/op",
            params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned),
                    // and `ptr` is immediately deallocated with the identical `lay`.
                    unsafe {
                        let ptr = MiMalloc.alloc(lay);
                        black_box(ptr);
                        MiMalloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/mimalloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = MiMalloc.alloc(l);
                                black_box(ptr);
                                MiMalloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/mimalloc/alloc_free_4t",
            "ns/op",
            params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = MiMalloc.alloc(l);
                        black_box(ptr);
                        MiMalloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    // 6. Competitor: tikv-jemallocator (1T & 4T)
    if should_run_bench("memory_pools/jemalloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/jemalloc/alloc_free_1t",
            "ns/op",
            params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned).
                    unsafe {
                        let ptr = Jemalloc.alloc(lay);
                        black_box(ptr);
                        Jemalloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/jemalloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = Jemalloc.alloc(l);
                                black_box(ptr);
                                Jemalloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/jemalloc/alloc_free_4t",
            "ns/op",
            params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = Jemalloc.alloc(l);
                        black_box(ptr);
                        Jemalloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    // 7. Competitor: std::alloc::System (1T & 4T)
    if should_run_bench("memory_pools/system_alloc/alloc_free_1t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/system_alloc/alloc_free_1t",
            "ns/op",
            params,
            || {
                for _ in 0..ops {
                    let lay = black_box(layout_256);
                    // SAFETY: `lay` is a valid non-zero `Layout` (256 bytes, 16-byte aligned).
                    unsafe {
                        let ptr = SystemAlloc.alloc(lay);
                        black_box(ptr);
                        SystemAlloc.dealloc(ptr, lay);
                    }
                }
            },
        );
    }

    if should_run_bench("memory_pools/system_alloc/alloc_free_4t") {
        set_allow_self_repin(true);
        maybe_repin_self_within_ccd();
        wait_for_quiet_sibling();
        set_allow_self_repin(false);
        let p_core_4t = current_self_pinned_core();
        pin_current_thread_to_core(p_core_4t);
        let stop_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let active_flag = Arc::new(zipora::memory::CachePadded::new(AtomicBool::new(false)));
        let mut handles = Vec::with_capacity(3);
        for worker_idx in 0..3usize {
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = worker_core_for(p_core_4t, worker_idx);
            let lay = pool_layouts[worker_idx + 1];
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..64 {
                            let l = black_box(lay);
                            // SAFETY: `l` is a valid non-zero `Layout`.
                            unsafe {
                                let ptr = SystemAlloc.alloc(l);
                                black_box(ptr);
                                SystemAlloc.dealloc(ptr, l);
                            }
                        }
                    } else {
                        std::hint::spin_loop();
                    }
                }
            }));
        }

        let main_lay = pool_layouts[0];
        measure_bench(
            records,
            "memory_pools",
            "memory_pools/system_alloc/alloc_free_4t",
            "ns/op",
            params,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops {
                    let l = black_box(main_lay);
                    // SAFETY: `l` is a valid non-zero `Layout`.
                    unsafe {
                        let ptr = SystemAlloc.alloc(l);
                        black_box(ptr);
                        SystemAlloc.dealloc(ptr, l);
                    }
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }

    set_allow_self_repin(true);
    pin_current_thread_to_core(current_self_pinned_core());
}

// ---------------------------------------------------------------------------
// 6. Entropy Coding (65,536 bytes = 64 KiB synthetic enwik8-like block)
//    Zipora: HuffmanEncoder O0, ContextualHuffmanEncoder O1/O2, FseEncoder,
//            Rans64Encoder<ParallelX1>, Rans64Encoder<ParallelX4>
//    Competitors: huffman-compress, constriction (ANS), zstd (levels 1 & 3)
// ---------------------------------------------------------------------------

fn bench_entropy(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let mut brk_holes: Vec<Vec<u8>> = Vec::with_capacity(6);
    for _ in 0..6 {
        brk_holes.push(black_box(vec![1u8; 4 * 1024 * 1024]));
    }
    let _brk_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(brk_holes);

    let block_size = sizes.entropy_block_bytes;
    let block = gen_enwik_like_block(block_size, SEED_BASE ^ 0x07);
    let mut params = BenchParams::fast(block_size as u64, block_size as u64).with_threshold(9.0);
    params.warmup_calls = 8;
    params.trials = 11;
    params.bursts_per_trial = 8;

    // 1. Zipora Huffman O0 (64 KiB)
    let huff_o0_enc = HuffmanEncoder::new(&block).unwrap();
    let huff_o0_compressed = huff_o0_enc.encode(&block).unwrap();
    let huff_o0_dec = HuffmanDecoder::new(huff_o0_enc.tree().clone());
    record_space_metric(
        space,
        "entropy/huffman_o0_bits_per_byte",
        (huff_o0_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o0/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o0_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o0/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o0_dec
                .decode(black_box(&huff_o0_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // 2. Zipora Huffman O1 (full 64 KiB block for encode_x1, encode_x4, and decode)
    let huff_o1_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap();
    let huff_o1_comp_sample = huff_o1_enc.encode(&block).unwrap();
    record_space_metric(
        space,
        "entropy/huffman_o1_bits_per_byte",
        (huff_o1_comp_sample.len() as f64 * 8.0) / (block_size as f64),
    );
    drop(huff_o1_comp_sample);

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/encode_x1_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o1_enc.encode_x1(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/encode_x4_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o1_enc.encode_x4(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );
    drop(huff_o1_enc);

    type ContextualBuiltItem = (
        Option<ContextualHuffmanEncoder>,
        Option<ContextualHuffmanDecoder>,
        Vec<u8>,
    );
    type ContextualBuildFn = fn(Vec<u8>, Vec<u8>) -> ContextualBuiltItem;

    let build_on_concurrent_threads = |f: ContextualBuildFn| {
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8usize)
            .map(|i| {
                let blk = block.clone();
                let bar = Arc::clone(&barrier);
                thread::spawn(move || {
                    let spacer = vec![1u8; (i * 4_672) + 1_376];
                    bar.wait();
                    let res = f(blk, spacer);
                    bar.wait();
                    res
                })
            })
            .collect();
        let mut out = Vec::with_capacity(8);
        for h in handles {
            out.push(h.join().unwrap());
        }
        pin_current_thread_to_core(current_self_pinned_core());
        out
    };

    let mut ctx_ring_params = params;
    ctx_ring_params.trials = 9;
    ctx_ring_params.bursts_per_trial = 8;
    {
        let mut built = build_on_concurrent_threads(|blk, spacer| {
            black_box(spacer.as_ptr());
            let enc = ContextualHuffmanEncoder::new(&blk, HuffmanOrder::Order1).unwrap();
            let comp = enc.encode(&blk).unwrap();
            drop(spacer);
            (None, Some(ContextualHuffmanDecoder::new(enc)), comp)
        });
        let mut huff_o1_decs: (usize, [(ContextualHuffmanDecoder, Vec<u8>); 8]) = (
            0,
            std::array::from_fn(|_| {
                let (_, dec, comp) = built.pop().unwrap();
                (dec.unwrap(), comp)
            }),
        );
        measure_bench_with_setup(
            records,
            "entropy",
            "entropy/huffman_o1/decode_64kib",
            "ns/byte",
            ctx_ring_params,
            &mut huff_o1_decs,
            |state| {
                state.0 = (state.0 + 1) & 7;
            },
            |state| {
                let (dec, comp) = &state.1[state.0];
                let decoded = dec.decode(black_box(comp), block_size).unwrap();
                black_box(decoded.len());
            },
        );
    }

    // 3. Zipora Huffman O2 (full 64 KiB block for encode and decode)
    {
        type ContextualO2EncPair = (ContextualHuffmanEncoder, Vec<u8>);
        type ContextualO2EncRing = (usize, [ContextualO2EncPair; 8], [Vec<u8>; 2]);

        let mut built = build_on_concurrent_threads(|blk, spacer| {
            black_box(spacer.as_ptr());
            let enc = ContextualHuffmanEncoder::new(&blk, HuffmanOrder::Order2).unwrap();
            drop(spacer);
            (Some(enc), None, blk)
        });
        let huff_o2_comp_sample = built[0].0.as_ref().unwrap().encode(&block).unwrap();
        record_space_metric(
            space,
            "entropy/huffman_o2_bits_per_byte",
            (huff_o2_comp_sample.len() as f64 * 8.0) / (block_size as f64),
        );
        drop(huff_o2_comp_sample);

        let mut huff_o2_encs: ContextualO2EncRing = (
            0,
            std::array::from_fn(|_| {
                let (enc, _, blk) = built.pop().unwrap();
                (enc.unwrap(), blk)
            }),
            [Vec::new(), Vec::new()],
        );
        measure_bench_with_setup(
            records,
            "entropy",
            "entropy/huffman_o2/encode_64kib",
            "ns/byte",
            ctx_ring_params,
            &mut huff_o2_encs,
            |state| {
                state.0 = (state.0 + 1) & 7;
            },
            |state| {
                let (enc, blk) = &state.1[state.0];
                let encoded = enc.encode(black_box(blk)).unwrap();
                black_box(encoded.len());
                state.2[state.0 & 1] = encoded;
            },
        );
    }

    {
        let mut built = build_on_concurrent_threads(|blk, spacer| {
            black_box(spacer.as_ptr());
            let enc = ContextualHuffmanEncoder::new(&blk, HuffmanOrder::Order2).unwrap();
            let comp = enc.encode(&blk).unwrap();
            drop(spacer);
            (None, Some(ContextualHuffmanDecoder::new(enc)), comp)
        });
        let mut huff_o2_decs: (usize, [(ContextualHuffmanDecoder, Vec<u8>); 8]) = (
            0,
            std::array::from_fn(|_| {
                let (_, dec, comp) = built.pop().unwrap();
                (dec.unwrap(), comp)
            }),
        );
        measure_bench_with_setup(
            records,
            "entropy",
            "entropy/huffman_o2/decode_64kib",
            "ns/byte",
            ctx_ring_params,
            &mut huff_o2_decs,
            |state| {
                state.0 = (state.0 + 1) & 7;
            },
            |state| {
                let (dec, comp) = &state.1[state.0];
                let decoded = dec.decode(black_box(comp), block_size).unwrap();
                black_box(decoded.len());
            },
        );
    }

    // 4. Zipora FSE (64 KiB)
    let mut fse_enc = FseEncoder::new(FseConfig::default()).unwrap();
    let fse_compressed = fse_enc.compress(&block).unwrap();
    record_space_metric(
        space,
        "entropy/fse_bits_per_byte",
        (fse_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/fse/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = fse_enc.compress(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/fse/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = fse_decompress(black_box(&fse_compressed)).unwrap();
            black_box(decoded.len());
        },
    );

    // 5. Zipora rANS64 X1 and X4 (64 KiB)
    let mut freqs = [1u32; 256];
    for &b in &block {
        freqs[b as usize] += 1;
    }
    let rans_x1_enc = Rans64Encoder::<ParallelX1>::new(&freqs).unwrap();
    let rans_x1_compressed = rans_x1_enc.encode(&block).unwrap();
    let rans_x1_dec = Rans64Decoder::<ParallelX1>::new(&rans_x1_enc);
    record_space_metric(
        space,
        "entropy/rans64_x1_bits_per_byte",
        (rans_x1_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    let rans_x4_enc = Rans64Encoder::<ParallelX4>::new(&freqs).unwrap();
    let rans_x4_compressed = rans_x4_enc.encode(&block).unwrap();
    let rans_x4_dec = Rans64Decoder::<ParallelX4>::new(&rans_x4_enc);
    record_space_metric(
        space,
        "entropy/rans64_x4_bits_per_byte",
        (rans_x4_compressed.len() as f64 * 8.0) / (block_size as f64),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x1/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = rans_x1_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x1/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = rans_x1_dec
                .decode(black_box(&rans_x1_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x4/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = rans_x4_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/rans64_x4/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = rans_x4_dec
                .decode(black_box(&rans_x4_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // 6. Competitor: huffman-compress (static Order-0 Huffman on 64 KiB block)
    {
        let mut huff_weights: BTreeMap<u8, u32> = BTreeMap::new();
        for &b in &block {
            *huff_weights.entry(b).or_insert(0) += 1;
        }
        let (hc_book, hc_tree) = HuffmanCompressBuilder::from_iter(huff_weights).finish();
        let mut hc_compressed = HuffmanBitVec::with_capacity(block_size * 8);
        for &b in &block {
            hc_book.encode(&mut hc_compressed, &b).unwrap();
        }
        let hc_bytes = hc_compressed.len().div_ceil(8);
        record_space_metric(
            space,
            "entropy/huffman_compress_bits_per_byte",
            (hc_bytes as f64 * 8.0) / (block_size as f64),
        );

        let mut hc_params = params;
        hc_params.warmup_calls = 2;
        hc_params.trials = 7;
        hc_params.bursts_per_trial = 2;

        measure_bench(
            records,
            "entropy",
            "entropy/huffman_compress/encode_64kib",
            "ns/byte",
            hc_params,
            || {
                let mut buf = HuffmanBitVec::with_capacity(block_size * 8);
                for b in black_box(&block) {
                    hc_book.encode(&mut buf, b).unwrap();
                }
                black_box(buf.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/huffman_compress/decode_64kib",
            "ns/byte",
            hc_params,
            || {
                let mut out = Vec::with_capacity(block_size);
                out.extend(hc_tree.decoder(black_box(&hc_compressed), block_size));
                black_box(out.len());
            },
        );
    }

    // 7. Competitor: constriction ANS (DefaultAnsCoder + 12-bit categorical / lookup model)
    {
        let probs_f64: Vec<f64> = freqs.iter().map(|&c| c as f64).collect();
        let constriction_enc_model =
            SmallContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                &probs_f64,
            )
            .unwrap();
        let constriction_dec_model = constriction_enc_model.to_lookup_decoder_model();

        let mut init_coder = DefaultAnsCoder::with_write_backend(Vec::with_capacity(block_size / 4));
        init_coder
            .encode_iid_symbols_reverse(
                block.iter().map(|&b| b as usize),
                constriction_enc_model.as_view(),
            )
            .unwrap();
        let constriction_compressed: Vec<u32> = init_coder.into_compressed().unwrap();
        record_space_metric(
            space,
            "entropy/constriction_ans_bits_per_byte",
            ((constriction_compressed.len() * 4) as f64 * 8.0) / (block_size as f64),
        );

        measure_bench(
            records,
            "entropy",
            "entropy/constriction_ans/encode_64kib",
            "ns/byte",
            params,
            || {
                let mut coder =
                    DefaultAnsCoder::with_write_backend(Vec::with_capacity(block_size / 4));
                coder
                    .encode_iid_symbols_reverse(
                        black_box(&block).iter().map(|&b| b as usize),
                        constriction_enc_model.as_view(),
                    )
                    .unwrap();
                let words = coder.into_compressed().unwrap();
                black_box(words.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/constriction_ans/decode_64kib",
            "ns/byte",
            params,
            || {
                let mut coder =
                    DefaultAnsCoder::from_compressed_slice(black_box(&constriction_compressed))
                        .unwrap();
                let mut out = Vec::with_capacity(block_size);
                for sym in coder.decode_iid_symbols(block_size, &constriction_dec_model) {
                    out.push(sym.unwrap() as u8);
                }
                black_box(out.len());
            },
        );
    }

    // 8. Competitor: zstd levels 1 and 3 (64 KiB whole-block compress / decompress with reused CCtx/DCtx)
    {
        let mut zstd_l1_comp = zstd::bulk::Compressor::new(1).unwrap();
        let mut zstd_l3_comp = zstd::bulk::Compressor::new(3).unwrap();
        let mut zstd_decomp = zstd::bulk::Decompressor::new().unwrap();
        let zstd_l1_compressed = zstd_l1_comp.compress(&block).unwrap();
        let zstd_l3_compressed = zstd_l3_comp.compress(&block).unwrap();
        record_space_metric(
            space,
            "entropy/zstd_l1_bits_per_byte",
            (zstd_l1_compressed.len() as f64 * 8.0) / (block_size as f64),
        );
        record_space_metric(
            space,
            "entropy/zstd_l3_bits_per_byte",
            (zstd_l3_compressed.len() as f64 * 8.0) / (block_size as f64),
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l1/encode_64kib",
            "ns/byte",
            params,
            || {
                let encoded = zstd_l1_comp.compress(black_box(&block)).unwrap();
                black_box(encoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l1/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded = zstd_decomp
                    .decompress(black_box(&zstd_l1_compressed), block_size)
                    .unwrap();
                black_box(decoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l3/encode_64kib",
            "ns/byte",
            params,
            || {
                let encoded = zstd_l3_comp.compress(black_box(&block)).unwrap();
                black_box(encoded.len());
            },
        );

        measure_bench(
            records,
            "entropy",
            "entropy/zstd_l3/decode_64kib",
            "ns/byte",
            params,
            || {
                let decoded = zstd_decomp
                    .decompress(black_box(&zstd_l3_compressed), block_size)
                    .unwrap();
                black_box(decoded.len());
            },
        );
    }

    drop(_brk_anchor);
}

// ---------------------------------------------------------------------------
// 7. Integer Codecs (100,000 contiguous sorted u32 integers by default)
//    Zipora: StreamVByte, GroupVarint
//    Competitors: stream-vbyte (Scalar), bitpacking (BitPacker4x)
// ---------------------------------------------------------------------------

fn bench_int_codecs(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let mut brk_holes: Vec<Vec<u8>> = Vec::with_capacity(6);
    for _ in 0..6 {
        brk_holes.push(black_box(vec![1u8; 4 * 1024 * 1024]));
    }
    let _brk_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(brk_holes);

    let n = sizes.int_codec_elements;
    let sorted_u32 = gen_uniform_sparse(n, 120, SEED_BASE ^ 0x08);
    let mut params = BenchParams::fast(n as u64, n as u64).with_threshold(9.0);
    params.trials = 11;
    params.bursts_per_trial = 8;

    let mut svb_into_params = BenchParams::fast(n as u64, (n * 32) as u64).with_threshold(9.0);
    svb_into_params.warmup_calls = 2;
    svb_into_params.trials = 11;
    svb_into_params.bursts_per_trial = 4;

    let mut svb_deltas_params = BenchParams::fast(n as u64, (n * 16) as u64).with_threshold(9.0);
    svb_deltas_params.warmup_calls = 2;
    svb_deltas_params.trials = 11;
    svb_deltas_params.bursts_per_trial = 4;

    // 1. Zipora StreamVByte
    let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
    let mut svb_out = vec![0u32; n];
    record_space_metric(
        space,
        "int_codecs/stream_vbyte_bits_per_int",
        ((svb_stream.controls.len() + svb_stream.data.len()) as f64 * 8.0) / (n as f64),
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/encode_deltas",
        "ns/int",
        params,
        || {
            let encoded = StreamVByte::encode_deltas(black_box(&sorted_u32));
            black_box(encoded.data.len());
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_into",
        "ns/int",
        svb_into_params,
        || {
            for _ in 0..32usize {
                let decoded = StreamVByte::decode_into(black_box(&svb_stream), n, &mut svb_out);
                black_box(decoded);
            }
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_deltas",
        "ns/int",
        svb_deltas_params,
        || {
            for _ in 0..16usize {
                let decoded = StreamVByte::decode_deltas(black_box(&svb_stream), n);
                black_box(decoded.len());
            }
        },
    );

    // 2. Zipora GroupVarint
    let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);
    record_space_metric(
        space,
        "int_codecs/group_varint_bits_per_int",
        (gv_encoded.len() as f64 * 8.0) / (n as f64),
    );
    let mut gv_spacers = Vec::with_capacity(8);
    let gv_inputs: [Vec<u32>; 8] = std::array::from_fn(|_| {
        gv_spacers.push(vec![1u8; 4_672]);
        sorted_u32.clone()
    });
    let mut gv_ring = 0usize;

    measure_bench_with_setup(
        records,
        "int_codecs",
        "int_codecs/group_varint/encode_deltas",
        "ns/int",
        params,
        &mut gv_ring,
        |ring| {
            *ring = (*ring + 1) & 7;
        },
        |ring| {
            let encoded = GroupVarint::encode_deltas(black_box(&gv_inputs[*ring]));
            black_box(encoded.len());
        },
    );
    drop(gv_spacers);

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/decode_deltas",
        "ns/int",
        params,
        || {
            let decoded = GroupVarint::decode_deltas(black_box(&gv_encoded), n);
            black_box(decoded.len());
        },
    );

    // 3. Competitor: stream-vbyte crate (v0.4.1 Scalar + delta pass)
    {
        let mut init_deltas = Vec::with_capacity(n);
        let mut prev = 0u32;
        for &v in &sorted_u32 {
            init_deltas.push(v.wrapping_sub(prev));
            prev = v;
        }
        let mut svb_crate_encoded = vec![0u8; n.div_ceil(4) + n * 4];
        let enc_len = svb_crate_encode::<SvbCrateScalar>(&init_deltas, &mut svb_crate_encoded);
        svb_crate_encoded.truncate(enc_len);
        record_space_metric(
            space,
            "int_codecs/stream_vbyte_crate_bits_per_int",
            (svb_crate_encoded.len() as f64 * 8.0) / (n as f64),
        );

        let mut svb_crate_out = vec![0u32; n];

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte_crate/encode_deltas",
            "ns/int",
            params,
            || {
                let src = black_box(&sorted_u32);
                let mut deltas = Vec::with_capacity(n);
                let mut p = 0u32;
                for &v in src {
                    deltas.push(v.wrapping_sub(p));
                    p = v;
                }
                let mut buf = vec![0u8; n.div_ceil(4) + n * 4];
                let written = svb_crate_encode::<SvbCrateScalar>(&deltas, &mut buf);
                buf.truncate(written);
                black_box(buf.len());
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte_crate/decode_into",
            "ns/int",
            params,
            || {
                let consumed = svb_crate_decode::<SvbCrateScalar>(
                    black_box(&svb_crate_encoded),
                    n,
                    &mut svb_crate_out,
                );
                black_box(consumed);
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/stream_vbyte_crate/decode_deltas",
            "ns/int",
            params,
            || {
                let mut out = vec![0u32; n];
                let consumed = svb_crate_decode::<SvbCrateScalar>(
                    black_box(&svb_crate_encoded),
                    n,
                    &mut out,
                );
                let mut acc = 0u32;
                for v in &mut out {
                    acc = acc.wrapping_add(*v);
                    *v = acc;
                }
                black_box(consumed + out.len());
            },
        );
    }

    // 4. Competitor: bitpacking crate (v0.9.3 BitPacker4x, 128-element SIMD blocks)
    {
        let bitpacker = BitPacker4x::new();
        const BP_BLOCK: usize = BitPacker4x::BLOCK_LEN; // 128
        let num_blocks = n.div_ceil(BP_BLOCK);
        let padded_len = num_blocks * BP_BLOCK;
        let mut padded_sorted = sorted_u32.clone();
        let last_val = *sorted_u32.last().unwrap_or(&0);
        padded_sorted.resize(padded_len, last_val);

        let mut bp_num_bits = Vec::with_capacity(num_blocks);
        let mut bp_compressed = vec![0u8; padded_len * 4];
        let mut bp_len = 0usize;
        let mut init_val = 0u32;
        for b_idx in 0..num_blocks {
            let chunk = &padded_sorted[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
            let nb = bitpacker.num_bits_sorted(init_val, chunk);
            bp_num_bits.push(nb);
            bp_len += bitpacker.compress_sorted(init_val, chunk, &mut bp_compressed[bp_len..], nb);
            init_val = chunk[BP_BLOCK - 1];
        }
        bp_compressed.truncate(bp_len);
        record_space_metric(
            space,
            "int_codecs/bitpacking_4x_bits_per_int",
            ((bp_compressed.len() + bp_num_bits.len()) as f64 * 8.0) / (n as f64),
        );

        let mut bp_out = vec![0u32; padded_len];

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/encode_deltas",
            "ns/int",
            params,
            || {
                let src = black_box(&padded_sorted);
                let mut num_bits_vec = Vec::with_capacity(num_blocks);
                let mut out_buf = vec![0u8; padded_len * 4];
                let mut pos = 0usize;
                let mut initial = 0u32;
                for b_idx in 0..num_blocks {
                    let chunk = &src[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                    let nb = bitpacker.num_bits_sorted(initial, chunk);
                    num_bits_vec.push(nb);
                    pos += bitpacker.compress_sorted(initial, chunk, &mut out_buf[pos..], nb);
                    initial = chunk[BP_BLOCK - 1];
                }
                out_buf.truncate(pos);
                black_box(out_buf.len() + num_bits_vec.len());
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/decode_into",
            "ns/int",
            svb_into_params,
            || {
                for _ in 0..32usize {
                    let comp = black_box(&bp_compressed);
                    let nbs = black_box(&bp_num_bits);
                    let mut pos = 0usize;
                    for (b_idx, &nb) in nbs.iter().enumerate() {
                        let dst = &mut bp_out[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                        pos += bitpacker.decompress(&comp[pos..], dst, nb);
                    }
                    black_box(pos);
                }
            },
        );

        measure_bench(
            records,
            "int_codecs",
            "int_codecs/bitpacking_4x/decode_deltas",
            "ns/int",
            svb_deltas_params,
            || {
                for _ in 0..16usize {
                    let comp = black_box(&bp_compressed);
                    let nbs = black_box(&bp_num_bits);
                    let mut out = vec![0u32; padded_len];
                    let mut pos = 0usize;
                    let mut initial = 0u32;
                    for (b_idx, &nb) in nbs.iter().enumerate() {
                        let dst = &mut out[b_idx * BP_BLOCK..(b_idx + 1) * BP_BLOCK];
                        pos += bitpacker.decompress_sorted(initial, &comp[pos..], dst, nb);
                        initial = dst[BP_BLOCK - 1];
                    }
                    out.truncate(n);
                    black_box(pos + out.len());
                }
            },
        );
    }

    drop(_brk_anchor);
}

// ---------------------------------------------------------------------------
// 8. Radix Sort (10,000,000 = 10^7 elements for u32, u64, (u64, u64) by default)
//    Zipora: RadixSort (lsd_u32, lsd_u64), KeyValueRadixSort<u64, u64> (kv_u64_u64)
//    Competitors: voracious_radix_sort, rdst (single-threaded), [T]::sort_unstable
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct RadixKvPair {
    key: u64,
    val: u64,
}

impl PartialEq for RadixKvPair {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for RadixKvPair {}

impl PartialOrd for RadixKvPair {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RadixKvPair {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl VoraciousRadixable<u64> for RadixKvPair {
    type Key = u64;
    #[inline]
    fn key(&self) -> Self::Key {
        self.key
    }
}

impl RdstRadixKey for RadixKvPair {
    const LEVELS: usize = 8;
    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.key >> (level * 8)) as u8
    }
}

fn bench_radix_sort_lsd(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.radix_sort_elements;
    let mut sorter = RadixSort::with_config(RadixSortConfig {
        use_parallel: false,
        use_counting_sort_threshold: 0,
        ..RadixSortConfig::default()
    });
    let mut lsd_u32_params = BenchParams::heavy(n as u64, n as u64, 3);
    lsd_u32_params.warmup_calls = 1;
    lsd_u32_params.bursts_per_trial = 2;
    let mut lsd_u64_params = BenchParams::heavy(n as u64, n as u64, 3);
    lsd_u64_params.warmup_calls = 1;
    lsd_u64_params.bursts_per_trial = 1;

    record_space_metric(space, "radix_sort/zipora_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/voracious_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/rdst_u32_bytes_per_elem", 8.0);
    record_space_metric(space, "radix_sort/std_u32_bytes_per_elem", 4.0);
    record_space_metric(space, "radix_sort/zipora_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/voracious_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/rdst_u64_bytes_per_elem", 16.0);
    record_space_metric(space, "radix_sort/std_u64_bytes_per_elem", 8.0);

    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    let mut lsd_holes: Vec<Vec<u8>> = Vec::with_capacity(4);
    for _ in 0..4 {
        lsd_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
    }
    let _lsd_top_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(lsd_holes);

    // 1. u32 sorting (10^7 elements): Zipora LSD, voracious_radix_sort, rdst, std::sort_unstable
    {
        let template_u32 = gen_radix_u32_template(n);
        if should_run_bench("radix_sort/lsd_u32") {
            let sorter0 = RadixSort::with_config(RadixSortConfig {
                use_parallel: false,
                use_counting_sort_threshold: 0,
                ..RadixSortConfig::default()
            });
            let work0 = vec![0u32; n];
            let _spacer = vec![1u8; 352 * 1024];
            let sorter1 = RadixSort::with_config(RadixSortConfig {
                use_parallel: false,
                use_counting_sort_threshold: 0,
                ..RadixSortConfig::default()
            });
            let work1 = vec![0u32; n];
            let mut u32_state = (
                0usize,
                Vec::<Vec<u8>>::with_capacity(32),
                [(sorter0, work0), (sorter1, work1)],
            );
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/lsd_u32",
                "ns/elem",
                lsd_u32_params,
                &mut u32_state,
                |st| {
                    st.0 ^= 1;
                    st.1.push(vec![1u8; 352 * 1024]);
                    st.2[st.0].1.copy_from_slice(&template_u32);
                },
                |st| {
                    let (srt, buf) = &mut st.2[st.0];
                    srt.sort_u32(black_box(buf.as_mut_slice())).unwrap();
                    black_box(buf[0]);
                },
            );
        }

        let mut work_u32 = vec![0u32; n];
        if should_run_bench("radix_sort/voracious/u32") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/u32",
                "ns/elem",
                lsd_u64_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0]);
                },
            );
        }

        if should_run_bench("radix_sort/rdst/u32") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/u32",
                "ns/elem",
                lsd_u64_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0]);
                },
            );
        }

        if should_run_bench("radix_sort/std_unstable/u32") {
            let mut std_u32_params = BenchParams::heavy(n as u64, n as u64, 3);
            std_u32_params.warmup_calls = 0;
            std_u32_params.bursts_per_trial = 1;
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/u32",
                "ns/elem",
                std_u32_params,
                &mut work_u32,
                |buf| {
                    buf.copy_from_slice(&template_u32);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable();
                    black_box(buf[0]);
                },
            );
        }
    }

    // 2. u64 sorting (10^7 elements): Zipora LSD, voracious_radix_sort, rdst, std::sort_unstable
    {
        let template_u64 = gen_radix_u64_template(n);
        let mut work_u64 = (Vec::<Vec<u8>>::with_capacity(16), vec![0u64; n]);
        if should_run_bench("radix_sort/lsd_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/lsd_u64",
                "ns/elem",
                lsd_u64_params,
                &mut work_u64,
                |st| {
                    st.0.push(vec![1u8; 352 * 1024]);
                    st.1.copy_from_slice(&template_u64);
                },
                |st| {
                    sorter.sort_u64(black_box(st.1.as_mut_slice())).unwrap();
                    black_box(st.1[0]);
                },
            );
        }

        if should_run_bench("radix_sort/voracious/u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/u64",
                "ns/elem",
                lsd_u64_params,
                &mut work_u64.1,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0]);
                },
            );
        }

        if should_run_bench("radix_sort/rdst/u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/u64",
                "ns/elem",
                lsd_u64_params,
                &mut work_u64.1,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0]);
                },
            );
        }

        if should_run_bench("radix_sort/std_unstable/u64") {
            let mut std_u64_params = BenchParams::heavy(n as u64, n as u64, 3);
            std_u64_params.warmup_calls = 0;
            std_u64_params.bursts_per_trial = 1;
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/u64",
                "ns/elem",
                std_u64_params,
                &mut work_u64.1,
                |buf| {
                    buf.copy_from_slice(&template_u64);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable();
                    black_box(buf[0]);
                },
            );
        }
    }
    drop(_lsd_top_anchor);
}

fn bench_radix_sort_kv(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    primary_core: usize,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    let n = sizes.radix_sort_elements;
    record_space_metric(space, "radix_sort/zipora_kv_bytes_per_elem", 56.0);
    record_space_metric(space, "radix_sort/voracious_kv_bytes_per_elem", 32.0);
    record_space_metric(space, "radix_sort/rdst_kv_bytes_per_elem", 32.0);
    record_space_metric(space, "radix_sort/std_kv_bytes_per_elem", 16.0);

    let p_core_init = current_self_pinned_core().min(primary_core);
    let _ = p_core_init;
    let rayon_core_for = |p_core: usize| -> usize {
        let ccd_base = (p_core / 8) * 8;
        if p_core != ccd_base + 2 {
            ccd_base + 2
        } else {
            ccd_base + 3
        }
    };
    let init_rayon_core = rayon_core_for(current_self_pinned_core());
    set_allow_self_repin(false);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .start_handler(move |_| {
            pin_current_thread_to_core(init_rayon_core);
        })
        .build_global();
    rayon::broadcast(|_| {});

    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    let template_kv = gen_radix_kv_template(n);
    let mut work_kv = vec![(0u64, 0u64); n];
    let mut kv_holes: Vec<Vec<u8>> = Vec::with_capacity(18);
    for _ in 0..18 {
        kv_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
    }
    let _kv_top_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(kv_holes);

    let kv_sorter = KeyValueRadixSort::<u64, u64>::new();
    let mut kv_params = BenchParams::heavy(n as u64, n as u64, 3);
    kv_params.warmup_calls = 1;
    kv_params.bursts_per_trial = 1;
    if should_run_bench("radix_sort/kv_u64_u64") {
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/kv_u64_u64",
            "ns/elem",
            kv_params,
            &mut work_kv,
            |buf| {
                buf.copy_from_slice(&template_kv);
            },
            |buf| {
                kv_sorter
                    .sort_by_key(black_box(buf.as_mut_slice()))
                    .unwrap();
                black_box(buf[0].0);
            },
        );
    }

    if should_run_bench("radix_sort/voracious/kv_u64_u64")
        || should_run_bench("radix_sort/rdst/kv_u64_u64")
        || should_run_bench("radix_sort/std_unstable/kv_u64_u64")
    {
        let template_pairs: Vec<RadixKvPair> = template_kv
            .iter()
            .map(|&(key, val)| RadixKvPair { key, val })
            .collect();
        let mut work_pairs = vec![RadixKvPair::default(); n];

        if should_run_bench("radix_sort/voracious/kv_u64_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/voracious/kv_u64_u64",
                "ns/elem",
                kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).voracious_sort();
                    black_box(buf[0].key);
                },
            );
        }

        if should_run_bench("radix_sort/rdst/kv_u64_u64") {
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/rdst/kv_u64_u64",
                "ns/elem",
                kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice())
                        .radix_sort_builder()
                        .with_single_threaded_tuner()
                        .sort();
                    black_box(buf[0].key);
                },
            );
        }

        if should_run_bench("radix_sort/std_unstable/kv_u64_u64") {
            let mut std_kv_params = BenchParams::heavy(n as u64, n as u64, 3);
            std_kv_params.warmup_calls = 0;
            std_kv_params.bursts_per_trial = 1;
            measure_bench_with_setup(
                records,
                "radix_sort",
                "radix_sort/std_unstable/kv_u64_u64",
                "ns/elem",
                std_kv_params,
                &mut work_pairs,
                |buf| {
                    buf.copy_from_slice(&template_pairs);
                },
                |buf| {
                    black_box(buf.as_mut_slice()).sort_unstable_by_key(|p| p.key);
                    black_box(buf[0].key);
                },
            );
        }
    }

    drop(_kv_top_anchor);
    set_allow_self_repin(true);
}

// ---------------------------------------------------------------------------
// 9. Suffix Array (10,000,000 = 10^7 bytes + 100,000 uniform substring queries)
//    Zipora: SuffixArray
//    Competitors: suffix::SuffixTable, divsufsort, cdivsufsort
// ---------------------------------------------------------------------------

fn bench_suffix_array(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    let n = sizes.suffix_array_bytes;
    record_space_metric(space, "suffix_array/zipora_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/suffix_crate_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/divsufsort_rust_bytes_per_byte", 5.0);
    record_space_metric(space, "suffix_array/cdivsufsort_c_bytes_per_byte", 5.0);

    let text0 = gen_enwik_like_block(n, SEED_BASE ^ 0x0A);
    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    let mut sa_holes: Vec<Vec<u8>> = Vec::with_capacity(18);
    for _ in 0..18 {
        sa_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
    }
    let _brk_top_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(sa_holes);

    let num_queries = sizes.suffix_array_queries;
    let sa_batch = (num_queries / 100).max(500).min(num_queries.max(1));
    let pat_len = 16usize;
    let max_start = n - pat_len;
    let sa_query_seed = SEED_BASE ^ 0x0A01;
    let sa_perm_len = ((n / 10) / 1000 * 1000).max(1_000);
    let sa_window = (max_start / sa_perm_len).max(1);
    let sa_perm = BijectivePerm::new(sa_perm_len, SEED_BASE ^ 0x0A02);
    let mut sa_search_params = BenchParams::heavy(n as u64, sa_batch as u64, 12);
    sa_search_params.warmup_calls = 8;
    sa_search_params.bursts_per_trial = 12;

    if should_run_any(&["suffix_array/construct", "suffix_array/search_query"]) {
        let mut sa_pending: Option<SuffixArray> = None;
        let mut sa_build_params = BenchParams::heavy(n as u64, n as u64, 4);
        sa_build_params.warmup_calls = 1;
        sa_build_params.bursts_per_trial = 1;
        measure_bench_with_setup(
            records,
            "suffix_array",
            "suffix_array/construct",
            "ns/byte",
            sa_build_params,
            &mut sa_pending,
            |slot| {
                let _ = slot.take();
            },
            |slot| {
                let built = SuffixArray::new(black_box(&text0)).unwrap();
                black_box(built.as_slice().len());
                *slot = Some(built);
            },
        );
        let sa0 = sa_pending
            .take()
            .unwrap_or_else(|| SuffixArray::new(&text0).unwrap());

        if should_run_bench("suffix_array/search_query") {
            let _text_spacer = vec![1u8; 1_376 * 1024];
            let text1 = text0.clone();
            let _sa_spacer = vec![1u8; 3_424 * 1024];
            let sa1 = SuffixArray::new(&text1).unwrap();
            let sas = [sa0, sa1];
            let texts = [text0.clone(), text1];

            let mut warm_sa_hits = 0usize;
            for idx in 0..(sa_batch * 24) {
                let slot = idx & 1;
                let txt = &texts[slot];
                let sa = &sas[slot];
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&txt[pos..pos + pat_len]);
                if idx % 10 == 9 {
                    pat[pat_len - 1] = b'~';
                }
                let (_, count) = sa.search(black_box(txt), black_box(&pat[..]));
                warm_sa_hits = warm_sa_hits.wrapping_add(count);
            }
            black_box(warm_sa_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let slot = q_idx & 1;
                        let txt = &texts[slot];
                        let sa = &sas[slot];
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&txt[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let (_, count) = sa.search(black_box(txt), black_box(&pat[..]));
                        hits = hits.wrapping_add(count);
                    });
                    black_box(hits);
                },
            );
            drop(sas);
            drop(texts);
            drop(_sa_spacer);
            drop(_text_spacer);
        } else {
            drop(sa0);
        }
    }

    // 2. Competitor: suffix::SuffixTable (pure-Rust SA-IS)
    if should_run_bench("suffix_array/suffix_crate/construct")
        || should_run_bench("suffix_array/suffix_crate/search_query")
    {
        let text_str0 = std::str::from_utf8(&text0).unwrap();
        let mut st_pending: Option<SuffixTable<'_, '_>> = None;
        let mut comp_sa_build_params = BenchParams::heavy(n as u64, n as u64, 3);
        comp_sa_build_params.warmup_calls = 0;
        comp_sa_build_params.bursts_per_trial = 1;
        measure_bench_with_setup(
            records,
            "suffix_array",
            "suffix_array/suffix_crate/construct",
            "ns/byte",
            comp_sa_build_params,
            &mut st_pending,
            |slot| {
                let _ = slot.take();
            },
            |slot| {
                let built = SuffixTable::new(black_box(text_str0));
                black_box(built.table().len());
                *slot = Some(built);
            },
        );
        if should_run_bench("suffix_array/suffix_crate/search_query") {
            let st0 = st_pending
                .take()
                .unwrap_or_else(|| SuffixTable::new(text_str0));
            let _text_spacer = vec![1u8; 1_376 * 1024];
            let text1 = text0.clone();
            let text_str1 = std::str::from_utf8(&text1).unwrap();
            let _sa_spacer = vec![1u8; 3_424 * 1024];
            let st1 = SuffixTable::new(text_str1);
            let sts = [st0, st1];
            let texts = [&text0, &text1];

            let mut warm_hits = 0usize;
            for idx in 0..(sa_batch * 24) {
                let slot = idx & 1;
                let txt = texts[slot];
                let st = &sts[slot];
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&txt[pos..pos + pat_len]);
                if idx % 10 == 9 {
                    pat[pat_len - 1] = b'~';
                }
                let pat_str = std::str::from_utf8(&pat).unwrap();
                warm_hits = warm_hits.wrapping_add(st.positions(black_box(pat_str)).len());
            }
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/suffix_crate/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let slot = q_idx & 1;
                        let txt = texts[slot];
                        let st = &sts[slot];
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&txt[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let pat_str = std::str::from_utf8(&pat).unwrap();
                        let positions = st.positions(black_box(pat_str));
                        hits = hits.wrapping_add(positions.len());
                    });
                    black_box(hits);
                },
            );
            drop(sts);
            drop(text1);
            drop(_sa_spacer);
            drop(_text_spacer);
        } else {
            drop(st_pending);
        }
    }

    // 3. Competitor: divsufsort (pure-Rust divsufsort)
    if should_run_bench("suffix_array/divsufsort_rust/construct")
        || should_run_bench("suffix_array/divsufsort_rust/search_query")
    {
        let mut dss_pending: Option<sacabase::SuffixArray<'_, i32>> = None;
        let mut comp_sa_build_params = BenchParams::heavy(n as u64, n as u64, 3);
        comp_sa_build_params.warmup_calls = 0;
        comp_sa_build_params.bursts_per_trial = 1;
        measure_bench_with_setup(
            records,
            "suffix_array",
            "suffix_array/divsufsort_rust/construct",
            "ns/byte",
            comp_sa_build_params,
            &mut dss_pending,
            |slot| {
                let _ = slot.take();
            },
            |slot| {
                let built = divsufsort::sort(black_box(&text0));
                black_box(built.text().len());
                *slot = Some(built);
            },
        );
        if should_run_bench("suffix_array/divsufsort_rust/search_query") {
            let dss0 = dss_pending
                .take()
                .unwrap_or_else(|| divsufsort::sort(&text0));
            let _text_spacer = vec![1u8; 1_376 * 1024];
            let text1 = text0.clone();
            let _sa_spacer = vec![1u8; 3_424 * 1024];
            let dss1 = divsufsort::sort(&text1);
            let dsss = [dss0, dss1];
            let texts = [&text0, &text1];

            let mut warm_hits = 0usize;
            for idx in 0..(sa_batch * 24) {
                let slot = idx & 1;
                let txt = texts[slot];
                let dss = &dsss[slot];
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&txt[pos..pos + pat_len]);
                if idx % 10 == 9 {
                    pat[pat_len - 1] = b'~';
                }
                let m = dss.longest_substring_match(black_box(&pat[..]));
                if m.len == pat_len {
                    warm_hits = warm_hits.wrapping_add(1);
                }
            }
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/divsufsort_rust/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let slot = q_idx & 1;
                        let txt = texts[slot];
                        let dss = &dsss[slot];
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&txt[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let m = dss.longest_substring_match(black_box(&pat[..]));
                        if m.len == pat_len {
                            hits = hits.wrapping_add(1);
                        }
                    });
                    black_box(hits);
                },
            );
            drop(dsss);
            drop(text1);
            drop(_sa_spacer);
            drop(_text_spacer);
        } else {
            drop(dss_pending);
        }
    }

    // 4. Competitor: cdivsufsort (C libdivsufsort binding)
    if should_run_bench("suffix_array/cdivsufsort_c/construct")
        || should_run_bench("suffix_array/cdivsufsort_c/search_query")
    {
        let mut cdss_pending: Option<sacabase::SuffixArray<'_, i32>> = None;
        let mut comp_sa_build_params = BenchParams::heavy(n as u64, n as u64, 3);
        comp_sa_build_params.warmup_calls = 0;
        comp_sa_build_params.bursts_per_trial = 1;
        measure_bench_with_setup(
            records,
            "suffix_array",
            "suffix_array/cdivsufsort_c/construct",
            "ns/byte",
            comp_sa_build_params,
            &mut cdss_pending,
            |slot| {
                let _ = slot.take();
            },
            |slot| {
                let built = cdivsufsort::sort(black_box(&text0));
                black_box(built.text().len());
                *slot = Some(built);
            },
        );
        if should_run_bench("suffix_array/cdivsufsort_c/search_query") {
            let cdss0 = cdss_pending
                .take()
                .unwrap_or_else(|| cdivsufsort::sort(&text0));
            let _text_spacer = vec![1u8; 1_376 * 1024];
            let text1 = text0.clone();
            let _sa_spacer = vec![1u8; 3_424 * 1024];
            let cdss1 = cdivsufsort::sort(&text1);
            let cdsss = [cdss0, cdss1];
            let texts = [&text0, &text1];

            let mut warm_hits = 0usize;
            for idx in 0..(sa_batch * 24) {
                let slot = idx & 1;
                let txt = texts[slot];
                let cdss = &cdsss[slot];
                let pos = uniform_index_at(sa_query_seed ^ 0x55, idx, max_start);
                let mut pat = [0u8; 16];
                pat.copy_from_slice(&txt[pos..pos + pat_len]);
                if idx % 10 == 9 {
                    pat[pat_len - 1] = b'~';
                }
                let m = cdss.longest_substring_match(black_box(&pat[..]));
                if m.len == pat_len {
                    warm_hits = warm_hits.wrapping_add(1);
                }
            }
            black_box(warm_hits);

            measure_bench(
                records,
                "suffix_array",
                "suffix_array/cdivsufsort_c/search_query",
                "ns/op",
                sa_search_params,
                || {
                    let base = current_burst_index() * sa_batch;
                    let mut hits = 0usize;
                    let mut q_idx = 0usize;
                    sa_perm.for_each_slice(base, sa_batch, |id| {
                        let slot = q_idx & 1;
                        let txt = texts[slot];
                        let cdss = &cdsss[slot];
                        let pos = (id * sa_window + uniform_index_at(sa_query_seed, id, sa_window))
                            .min(max_start.saturating_sub(1));
                        let mut pat = [0u8; 16];
                        pat.copy_from_slice(&txt[pos..pos + pat_len]);
                        if q_idx % 10 == 9 {
                            pat[pat_len - 1] = b'~';
                        }
                        q_idx = q_idx.wrapping_add(1);
                        let m = cdss.longest_substring_match(black_box(&pat[..]));
                        if m.len == pat_len {
                            hits = hits.wrapping_add(1);
                        }
                    });
                    black_box(hits);
                },
            );
            drop(cdsss);
            drop(text1);
            drop(_sa_spacer);
            drop(_text_spacer);
        } else {
            drop(cdss_pending);
        }
    }

    drop(_brk_top_anchor);
}

// ---------------------------------------------------------------------------
// 10. String Search & BM25 Scoring (64 KiB strchr & strstr + 100,000-doc BM25)
//     Zipora: SimdStringSearch (sse42_strchr, sse42_strstr), Bm25BatchScorer, FieldnormEncoder
//     Competitors: memchr::memchr, memchr::memmem::{find, Finder}, aho_corasick::AhoCorasick
// ---------------------------------------------------------------------------

fn bench_search_and_scoring(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    maybe_repin_self_within_ccd();
    record_space_metric(space, "search_and_scoring/strchr_strstr_aux_bytes", 0.0);
    {
        let haystack_len = 65_536usize;
        let haystacks = &SEARCH_HAYSTACKS.0;
        let needle = &SEARCH_NEEDLE.0[..13];

        let searcher = SimdStringSearch::new();
        let search_iters = 128u64;
        let mut str_params =
            BenchParams::fast(haystack_len as u64, search_iters).with_threshold(9.0);
        str_params.warmup_calls = 64;
        str_params.trials = 15;
        str_params.bursts_per_trial = 10;

        wait_for_quiet_sibling();
        let haystack0 = &haystacks[0][..];
        let mut pre_acc = 0usize;
        for _ in 0..4_096usize {
            if let Some(pos) = searcher.sse42_strchr(black_box(haystack0), black_box(b'Z')) {
                pre_acc = pre_acc.wrapping_add(pos);
            }
        }
        black_box(pre_acc);

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/simd_strchr_64kib",
            "ns/call",
            str_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = searcher.sse42_strchr(black_box(haystack0), black_box(b'Z'))
                    {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        for idx in 0..2_048usize {
            let haystack = &haystacks[idx & 3][..];
            if let Some(pos) = searcher.sse42_strstr(black_box(haystack), black_box(needle)) {
                pre_acc = pre_acc.wrapping_add(pos);
            }
        }
        black_box(pre_acc);

        let mut strstr_params = str_params;
        strstr_params.warmup_calls = 64;
        strstr_params.bursts_per_trial = 8;
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/simd_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for idx in 0..search_iters {
                    let haystack = &haystacks[(idx as usize) & 3][..];
                    if let Some(pos) = searcher.sse42_strstr(black_box(haystack), black_box(needle))
                    {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        // Competitors for strchr & strstr on the exact same haystacks and 13-byte needle
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memchr_strchr_64kib",
            "ns/call",
            str_params,
            || {
                let mut acc = 0usize;
                for _ in 0..search_iters {
                    if let Some(pos) = crate_memchr(black_box(b'Z'), black_box(haystack0)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memmem_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for idx in 0..search_iters {
                    let haystack = &haystacks[(idx as usize) & 3][..];
                    if let Some(pos) = crate_memmem::find(black_box(haystack), black_box(needle)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        let finder = crate_memmem::Finder::new(needle);
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/memmem_finder_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for idx in 0..search_iters {
                    let haystack = &haystacks[(idx as usize) & 3][..];
                    if let Some(pos) = finder.find(black_box(haystack)) {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );

        let ac = AhoCorasick::new([needle]).unwrap();
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/aho_corasick_strstr_64kib",
            "ns/call",
            strstr_params,
            || {
                let mut acc = 0usize;
                for idx in 0..search_iters {
                    let haystack = &haystacks[(idx as usize) & 3][..];
                    if let Some(m) = ac.find(black_box(haystack)) {
                        acc = acc.wrapping_add(m.start());
                    }
                }
                black_box(acc);
            },
        );
    }

    // BM25 hot paths (100,000 docs by default) + S36-R2 build_score_table
    let k1 = 1.2f32;
    let b = 0.75f32;
    let avg_dl = 120.0f32;
    let idf = 3.25f32;

    {
        let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
        let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

        let num_docs = sizes.bm25_docs;
        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x0B);
        let doc_ids: Vec<u32> = (0..num_docs as u32).map(|i| i * 3).collect();
        let tfs: Vec<u16> = (0..num_docs).map(|_| rng.random_range(1..=16)).collect();
        let norms: Vec<u8> = (0..num_docs * 3)
            .map(|_| rng.random_range(1..=200))
            .collect();
        let contiguous_norms: Vec<u8> = doc_ids.iter().map(|&d| norms[d as usize]).collect();
        let mut scores_out = vec![0.0f32; num_docs];

        let bm25_params = BenchParams::fast(num_docs as u64, num_docs as u64);
        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_score_single",
            "ns/doc",
            bm25_params,
            || {
                let mut sum = 0.0f32;
                for i in 0..num_docs {
                    sum += scorer.score(black_box(norms[doc_ids[i] as usize]), black_box(tfs[i]));
                }
                black_box(sum);
            },
        );

        measure_bench(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_batch_score_with_prefetch",
            "ns/doc",
            bm25_params.with_threshold(9.0),
            || {
                scorer.batch_score_with_prefetch(
                    black_box(&norms),
                    black_box(&doc_ids),
                    black_box(&tfs),
                    black_box(&mut scores_out),
                );
                black_box(scores_out[0]);
            },
        );

        let mut simd_spacers: Vec<Vec<u8>> = Vec::with_capacity(8);
        let mut simd_ring_bufs: [(Vec<u8>, Vec<u16>, Vec<f32>); 8] = std::array::from_fn(|_| {
            simd_spacers.push(vec![1u8; 4_672]);
            (
                contiguous_norms.clone(),
                tfs.clone(),
                vec![0.0f32; num_docs],
            )
        });
        let mut simd_params = bm25_params;
        simd_params.bursts_per_trial = 8;
        let mut simd_ring = 0usize;
        measure_bench_with_setup(
            records,
            "search_and_scoring",
            "search_and_scoring/bm25_batch_score_simd",
            "ns/doc",
            simd_params,
            &mut simd_ring,
            |ring| {
                *ring = (*ring + 1) & 7;
            },
            |ring| {
                let (cn, tf_buf, out_buf) = &mut simd_ring_bufs[*ring];
                scorer.batch_score(
                    black_box(cn.as_slice()),
                    black_box(tf_buf.as_slice()),
                    black_box(out_buf.as_mut_slice()),
                );
                black_box(out_buf[0]);
            },
        );
        drop(simd_spacers);
    }

    // S36-R2: FieldnormEncoder::build_score_table (256 * 256 = 65,536 cells = 256 KiB per call)
    let table_rounds = 500u64;
    let max_tf = 255u16;
    let cells_per_call = (max_tf as u64 + 1) * 256u64;
    let mut table_params =
        BenchParams::fast(cells_per_call, table_rounds * cells_per_call).with_threshold(9.0);
    table_params.warmup_calls = 4;
    table_params.trials = 7;
    table_params.bursts_per_trial = 3;
    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/fieldnorm_build_score_table",
        "ns/cell",
        table_params,
        || {
            for r in 0..table_rounds {
                let table = FieldnormEncoder::build_score_table(
                    black_box(avg_dl),
                    black_box(k1),
                    black_box(b),
                    black_box(idf + (r as f32) * 1e-6),
                    black_box(max_tf),
                );
                black_box(table[1][1]);
            }
        },
    );
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
    // E3 rule: "leading" means at or above the best competitor within ±3% on the primary
    // metric (i.e. latency delta <= +3.0%) and no more than 10% worse on the secondary
    // metric (secondary_delta_pct <= +10.0%).
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

fn build_same_run_comparisons(
    records: &[BenchRecord],
    space: &BTreeMap<String, f64>,
) -> Vec<SameRunComparison> {
    let mut out = Vec::new();

    // 1. Rank/Select: compare Zipora's fastest compact Rank/Select (RankSelectSE256 & Interleaved256)
    //    against sucds::Rank9Sel, vers_vecs::RsVec, and bitm::RankSelect101111.
    //    Use total bits/bit (100% + overhead_pct) as the secondary space metric so 3.5% vs 3.5%
    //    is compared on total footprint rather than dividing near-zero overhead percentages.
    let z_se_total_pct = 100.0
        + space
            .get("rank_select/se256_overhead_pct")
            .copied()
            .unwrap_or(0.0);
    let z_rsi_total_pct = 100.0
        + space
            .get("rank_select/interleaved256_overhead_pct")
            .copied()
            .unwrap_or(0.0);
    let sucds_total_pct = 100.0
        + space
            .get("rank_select/sucds_rank9sel_overhead_pct")
            .copied()
            .unwrap_or(0.0);
    let vers_total_pct = 100.0
        + space
            .get("rank_select/vers_vecs_rsvec_overhead_pct")
            .copied()
            .unwrap_or(0.0);
    let bitm_total_pct = 100.0
        + space
            .get("rank_select/bitm_rs101111_overhead_pct")
            .copied()
            .unwrap_or(0.0);

    for (comp_prefix, comp_label, comp_space) in [
        (
            "rank_select/sucds_rank9sel",
            "sucds::Rank9Sel",
            sucds_total_pct,
        ),
        (
            "rank_select/vers_vecs_rsvec",
            "vers_vecs::RsVec",
            vers_total_pct,
        ),
        (
            "rank_select/bitm_rs101111",
            "bitm::RankSelect101111",
            bitm_total_pct,
        ),
    ] {
        push_comparison(
            &mut out,
            records,
            "rank_select",
            "rank1 (1 Gbit)",
            "rank_select/interleaved256/rank1",
            "RankSelectInterleaved256",
            &format!("{comp_prefix}/rank1"),
            comp_label,
            "ns/op",
            "total_space_pct_of_raw",
            z_rsi_total_pct,
            comp_space,
        );
        push_comparison(
            &mut out,
            records,
            "rank_select",
            "rank1 (1 Gbit, SE256)",
            "rank_select/se256/rank1",
            "RankSelectSE256",
            &format!("{comp_prefix}/rank1"),
            comp_label,
            "ns/op",
            "total_space_pct_of_raw",
            z_se_total_pct,
            comp_space,
        );
        push_comparison(
            &mut out,
            records,
            "rank_select",
            "select1 (1 Gbit)",
            "rank_select/se256/select1",
            "RankSelectSE256",
            &format!("{comp_prefix}/select1"),
            comp_label,
            "ns/op",
            "total_space_pct_of_raw",
            z_se_total_pct,
            comp_space,
        );
        push_comparison(
            &mut out,
            records,
            "rank_select",
            "select0 (1 Gbit)",
            "rank_select/se256/select0",
            "RankSelectSE256",
            &format!("{comp_prefix}/select0"),
            comp_label,
            "ns/op",
            "total_space_pct_of_raw",
            z_se_total_pct,
            comp_space,
        );
    }

    // 2. Elias-Fano (100k elements across uniform, zipf_clustered, dense)
    for dist_name in ["uniform", "zipf_clustered", "dense"] {
        let z_ef_bpe = space
            .get(&format!("elias_fano/{dist_name}/ef_bits_per_elem"))
            .copied()
            .unwrap_or(0.0);
        let z_opef_bpe = space
            .get(&format!("elias_fano/{dist_name}/opef_bits_per_elem"))
            .copied()
            .unwrap_or(0.0);
        let z_cef_bpe = space
            .get(&format!("elias_fano/{dist_name}/cef_bits_per_elem"))
            .copied()
            .unwrap_or(0.0);
        let sucds_bpe = space
            .get(&format!("elias_fano/{dist_name}/sucds_ef_bits_per_elem"))
            .copied()
            .unwrap_or(0.0);
        let vers_bpe = space
            .get(&format!("elias_fano/{dist_name}/vers_ef_bits_per_elem"))
            .copied()
            .unwrap_or(0.0);

        let (best_z_bench, best_z_var, best_z_bpe) = match dist_name {
            "uniform" => (
                format!("elias_fano/{dist_name}/ef_next_geq"),
                "EliasFano",
                z_ef_bpe,
            ),
            "zipf_clustered" => (
                format!("elias_fano/{dist_name}/opef_next_geq"),
                "OptimalPartitionedEliasFano",
                z_opef_bpe,
            ),
            _ => (
                format!("elias_fano/{dist_name}/cef_next_geq"),
                "ClusteredEliasFano",
                z_cef_bpe,
            ),
        };

        for (comp_suffix, comp_label, comp_bpe) in [
            ("sucds_ef", "sucds::EliasFano", sucds_bpe),
            ("vers_ef", "vers_vecs::EliasFanoVec", vers_bpe),
        ] {
            push_comparison(
                &mut out,
                records,
                "elias_fano",
                &format!("next_geq ({dist_name}, EF)"),
                &format!("elias_fano/{dist_name}/ef_next_geq"),
                "EliasFano",
                &format!("elias_fano/{dist_name}/{comp_suffix}_next_geq"),
                comp_label,
                "ns/op",
                "bits_per_elem",
                z_ef_bpe,
                comp_bpe,
            );
            if best_z_var != "EliasFano" {
                push_comparison(
                    &mut out,
                    records,
                    "elias_fano",
                    &format!("next_geq ({dist_name}, best variant)"),
                    &best_z_bench,
                    best_z_var,
                    &format!("elias_fano/{dist_name}/{comp_suffix}_next_geq"),
                    comp_label,
                    "ns/op",
                    "bits_per_elem",
                    best_z_bpe,
                    comp_bpe,
                );
            }
            push_comparison(
                &mut out,
                records,
                "elias_fano",
                &format!("select / get ({dist_name}, EF)"),
                &format!("elias_fano/{dist_name}/ef_select"),
                "EliasFano",
                &format!("elias_fano/{dist_name}/{comp_suffix}_select"),
                comp_label,
                "ns/op",
                "bits_per_elem",
                z_ef_bpe,
                comp_bpe,
            );
            push_comparison(
                &mut out,
                records,
                "elias_fano",
                &format!("select / get ({dist_name}, CEF)"),
                &format!("elias_fano/{dist_name}/cef_select"),
                "ClusteredEliasFano",
                &format!("elias_fano/{dist_name}/{comp_suffix}_select"),
                comp_label,
                "ns/op",
                "bits_per_elem",
                z_cef_bpe,
                comp_bpe,
            );
            push_comparison(
                &mut out,
                records,
                "elias_fano",
                &format!("advance_to_index ({dist_name}, OPEF)"),
                &format!("elias_fano/{dist_name}/opef_cursor_advance_to_index"),
                "OptimalPartitionedEliasFano",
                &format!("elias_fano/{dist_name}/{comp_suffix}_select"),
                comp_label,
                "ns/op",
                "bits_per_elem",
                z_opef_bpe,
                comp_bpe,
            );
        }
    }

    // 3. Tries (1,000,000 keys across url_paths, random_16b, shared_prefix)
    for dist_name in ["url_paths", "random_16b", "shared_prefix"] {
        let dat_bpk = space
            .get(&format!("tries/dat_{dist_name}_bytes_per_key"))
            .copied()
            .unwrap_or(0.0);
        let cspp_bpk = space
            .get(&format!("tries/cspp_{dist_name}_bytes_per_key"))
            .copied()
            .unwrap_or(0.0);
        let fst_bpk = space
            .get(&format!("tries/fst_{dist_name}_bytes_per_key"))
            .copied()
            .unwrap_or(0.0);
        let qp_bpk = space
            .get(&format!("tries/qp_trie_{dist_name}_bytes_per_key"))
            .copied()
            .unwrap_or(0.0);
        let radix_bpk = space
            .get(&format!("tries/radix_trie_{dist_name}_bytes_per_key"))
            .copied()
            .unwrap_or(0.0);

        for (comp_build, comp_lookup, comp_label, comp_bpk) in [
            (
                format!("tries/fst/{dist_name}/build_sorted"),
                format!("tries/fst/{dist_name}/lookup"),
                "fst::Set",
                fst_bpk,
            ),
            (
                format!("tries/qp_trie/{dist_name}/build"),
                format!("tries/qp_trie/{dist_name}/lookup"),
                "qp_trie::Trie",
                qp_bpk,
            ),
            (
                format!("tries/radix_trie/{dist_name}/build"),
                format!("tries/radix_trie/{dist_name}/lookup"),
                "radix_trie::Trie",
                radix_bpk,
            ),
        ] {
            push_comparison(
                &mut out,
                records,
                "tries",
                &format!("build ({dist_name}, DAT)"),
                &format!("tries/dat/{dist_name}/build_sorted"),
                "DoubleArrayTrie",
                &comp_build,
                comp_label,
                "ns/key",
                "bytes_per_key",
                dat_bpk,
                comp_bpk,
            );
            push_comparison(
                &mut out,
                records,
                "tries",
                &format!("build ({dist_name}, CSPP)"),
                &format!("tries/cspp/{dist_name}/build"),
                "CsppTrie",
                &comp_build,
                comp_label,
                "ns/key",
                "bytes_per_key",
                cspp_bpk,
                comp_bpk,
            );
            push_comparison(
                &mut out,
                records,
                "tries",
                &format!("lookup ({dist_name}, DAT)"),
                &format!("tries/dat/{dist_name}/lookup"),
                "DoubleArrayTrie",
                &comp_lookup,
                comp_label,
                "ns/op",
                "bytes_per_key",
                dat_bpk,
                comp_bpk,
            );
            push_comparison(
                &mut out,
                records,
                "tries",
                &format!("lookup ({dist_name}, CSPP)"),
                &format!("tries/cspp/{dist_name}/lookup"),
                "CsppTrie",
                &comp_lookup,
                comp_label,
                "ns/op",
                "bytes_per_key",
                cspp_bpk,
                comp_bpk,
            );
        }
    }

    // 4. Hash Maps (1,000,000 u64 and [u8; 16] keys)
    for (key_kind, z_bpe_key, g_bpe_key, hb_bpe_key, std_bpe_key) in [
        (
            "u64",
            "hash_maps/zipora_u64_bytes_per_entry",
            "hash_maps/gold_u64_bytes_per_entry",
            "hash_maps/hashbrown_u64_bytes_per_entry",
            "hash_maps/std_u64_bytes_per_entry",
        ),
        (
            "str16",
            "hash_maps/zipora_str16_bytes_per_entry",
            "hash_maps/gold_str16_bytes_per_entry",
            "hash_maps/hashbrown_str16_bytes_per_entry",
            "hash_maps/std_str16_bytes_per_entry",
        ),
    ] {
        let z_bpe = space.get(z_bpe_key).copied().unwrap_or(0.0);
        let g_bpe = space.get(g_bpe_key).copied().unwrap_or(0.0);
        let hb_bpe = space.get(hb_bpe_key).copied().unwrap_or(0.0);
        let std_bpe = space.get(std_bpe_key).copied().unwrap_or(0.0);

        for (comp_prefix, comp_label, comp_bpe) in [
            ("hash_maps/hashbrown_ahash", "hashbrown+ahash", hb_bpe),
            ("hash_maps/std", "std::collections::HashMap", std_bpe),
        ] {
            for (op_name, op_suffix) in [("insert", "insert"), ("get", "get")] {
                push_comparison(
                    &mut out,
                    records,
                    "hash_maps",
                    &format!("{op_name}_{key_kind} (ZiporaHashMap)"),
                    &format!("hash_maps/zipora_hash_map/{op_suffix}_{key_kind}"),
                    "ZiporaHashMap",
                    &format!("{comp_prefix}/{op_suffix}_{key_kind}"),
                    comp_label,
                    "ns/op",
                    "bytes_per_entry",
                    z_bpe,
                    comp_bpe,
                );
                push_comparison(
                    &mut out,
                    records,
                    "hash_maps",
                    &format!("{op_name}_{key_kind} (GoldHashMap)"),
                    &format!("hash_maps/gold_hash_map/{op_suffix}_{key_kind}"),
                    "GoldHashMap",
                    &format!("{comp_prefix}/{op_suffix}_{key_kind}"),
                    comp_label,
                    "ns/op",
                    "bytes_per_entry",
                    g_bpe,
                    comp_bpe,
                );
            }
        }
    }

    // 5. Memory Pools (10,000 ops of 256B alloc+free, 1T & 4T)
    let lf_slot = space
        .get("memory_pools/lockfree_bytes_per_256b_slot")
        .copied()
        .unwrap_or(256.0);
    let sec_slot = space
        .get("memory_pools/secure_bytes_per_256b_slot")
        .copied()
        .unwrap_or(384.0);
    let mi_slot = space
        .get("memory_pools/mimalloc_bytes_per_256b_slot")
        .copied()
        .unwrap_or(256.0);
    let je_slot = space
        .get("memory_pools/jemalloc_bytes_per_256b_slot")
        .copied()
        .unwrap_or(256.0);
    let sys_slot = space
        .get("memory_pools/system_bytes_per_256b_slot")
        .copied()
        .unwrap_or(272.0);

    for (comp_prefix, comp_label, comp_slot) in [
        ("memory_pools/mimalloc", "mimalloc::MiMalloc", mi_slot),
        ("memory_pools/jemalloc", "tikv_jemallocator::Jemalloc", je_slot),
        ("memory_pools/system_alloc", "std::alloc::System", sys_slot),
    ] {
        for threads in ["1t", "4t"] {
            push_comparison(
                &mut out,
                records,
                "memory_pools",
                &format!("alloc_free_{threads} (LockFreeMemoryPool)"),
                &format!("memory_pools/lockfree_pool/alloc_free_{threads}"),
                "LockFreeMemoryPool",
                &format!("{comp_prefix}/alloc_free_{threads}"),
                comp_label,
                "ns/op",
                "bytes_per_256b_slot",
                lf_slot,
                comp_slot,
            );
            push_comparison(
                &mut out,
                records,
                "memory_pools",
                &format!("alloc_free_{threads} (SecureMemoryPool)"),
                &format!("memory_pools/secure_pool/alloc_free_{threads}"),
                "SecureMemoryPool",
                &format!("{comp_prefix}/alloc_free_{threads}"),
                comp_label,
                "ns/op",
                "bytes_per_256b_slot",
                sec_slot,
                comp_slot,
            );
        }
    }

    // 6. Entropy Coding (64 KiB synthetic enwik8-like block)
    let huff_o0_bpb = space
        .get("entropy/huffman_o0_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let huff_o1_bpb = space
        .get("entropy/huffman_o1_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let huff_o2_bpb = space
        .get("entropy/huffman_o2_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let fse_bpb = space
        .get("entropy/fse_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let rans_x1_bpb = space
        .get("entropy/rans64_x1_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let rans_x4_bpb = space
        .get("entropy/rans64_x4_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let hc_bpb = space
        .get("entropy/huffman_compress_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let cons_bpb = space
        .get("entropy/constriction_ans_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let zstd_l1_bpb = space
        .get("entropy/zstd_l1_bits_per_byte")
        .copied()
        .unwrap_or(0.0);
    let zstd_l3_bpb = space
        .get("entropy/zstd_l3_bits_per_byte")
        .copied()
        .unwrap_or(0.0);

    // 6a. Huffman O0/O1/O2 vs huffman-compress and zstd
    for (z_enc, z_dec, z_var, z_bpb) in [
        (
            "entropy/huffman_o0/encode_64kib",
            "entropy/huffman_o0/decode_64kib",
            "HuffmanEncoder (O0)",
            huff_o0_bpb,
        ),
        (
            "entropy/huffman_o1/encode_x1_64kib",
            "entropy/huffman_o1/decode_64kib",
            "ContextualHuffmanEncoder (O1 x1)",
            huff_o1_bpb,
        ),
        (
            "entropy/huffman_o1/encode_x4_64kib",
            "entropy/huffman_o1/decode_64kib",
            "ContextualHuffmanEncoder (O1 x4)",
            huff_o1_bpb,
        ),
        (
            "entropy/huffman_o2/encode_64kib",
            "entropy/huffman_o2/decode_64kib",
            "ContextualHuffmanEncoder (O2)",
            huff_o2_bpb,
        ),
    ] {
        for (c_prefix, c_label, c_bpb) in [
            ("entropy/huffman_compress", "huffman-compress", hc_bpb),
            ("entropy/zstd_l1", "zstd (level 1)", zstd_l1_bpb),
            ("entropy/zstd_l3", "zstd (level 3)", zstd_l3_bpb),
        ] {
            push_comparison(
                &mut out,
                records,
                "entropy",
                &format!("encode_64kib ({z_var})"),
                z_enc,
                z_var,
                &format!("{c_prefix}/encode_64kib"),
                c_label,
                "ns/byte",
                "bits_per_byte",
                z_bpb,
                c_bpb,
            );
            push_comparison(
                &mut out,
                records,
                "entropy",
                &format!("decode_64kib ({z_var})"),
                z_dec,
                z_var,
                &format!("{c_prefix}/decode_64kib"),
                c_label,
                "ns/byte",
                "bits_per_byte",
                z_bpb,
                c_bpb,
            );
        }
    }

    // 6b. FSE & rANS64 (X1 & X4) vs constriction ANS and zstd (levels 1 & 3)
    for (z_enc, z_dec, z_var, z_bpb) in [
        (
            "entropy/fse/encode_64kib",
            "entropy/fse/decode_64kib",
            "FseEncoder",
            fse_bpb,
        ),
        (
            "entropy/rans64_x1/encode_64kib",
            "entropy/rans64_x1/decode_64kib",
            "Rans64Encoder<ParallelX1>",
            rans_x1_bpb,
        ),
        (
            "entropy/rans64_x4/encode_64kib",
            "entropy/rans64_x4/decode_64kib",
            "Rans64Encoder<ParallelX4>",
            rans_x4_bpb,
        ),
    ] {
        for (c_prefix, c_label, c_bpb) in [
            ("entropy/constriction_ans", "constriction::DefaultAnsCoder", cons_bpb),
            ("entropy/zstd_l1", "zstd (level 1)", zstd_l1_bpb),
            ("entropy/zstd_l3", "zstd (level 3)", zstd_l3_bpb),
        ] {
            push_comparison(
                &mut out,
                records,
                "entropy",
                &format!("encode_64kib ({z_var})"),
                z_enc,
                z_var,
                &format!("{c_prefix}/encode_64kib"),
                c_label,
                "ns/byte",
                "bits_per_byte",
                z_bpb,
                c_bpb,
            );
            push_comparison(
                &mut out,
                records,
                "entropy",
                &format!("decode_64kib ({z_var})"),
                z_dec,
                z_var,
                &format!("{c_prefix}/decode_64kib"),
                c_label,
                "ns/byte",
                "bits_per_byte",
                z_bpb,
                c_bpb,
            );
        }
    }

    // 7. Integer Codecs (100,000 sorted u32 integers)
    let svb_bpi = space
        .get("int_codecs/stream_vbyte_bits_per_int")
        .copied()
        .unwrap_or(0.0);
    let gv_bpi = space
        .get("int_codecs/group_varint_bits_per_int")
        .copied()
        .unwrap_or(0.0);
    let svb_crate_bpi = space
        .get("int_codecs/stream_vbyte_crate_bits_per_int")
        .copied()
        .unwrap_or(0.0);
    let bp_bpi = space
        .get("int_codecs/bitpacking_4x_bits_per_int")
        .copied()
        .unwrap_or(0.0);

    for (comp_prefix, comp_label, comp_bpi) in [
        ("int_codecs/stream_vbyte_crate", "stream_vbyte::Scalar", svb_crate_bpi),
        ("int_codecs/bitpacking_4x", "bitpacking::BitPacker4x", bp_bpi),
    ] {
        for op in ["encode_deltas", "decode_into", "decode_deltas"] {
            push_comparison(
                &mut out,
                records,
                "int_codecs",
                &format!("{op} (StreamVByte)"),
                &format!("int_codecs/stream_vbyte/{op}"),
                "StreamVByte",
                &format!("{comp_prefix}/{op}"),
                comp_label,
                "ns/int",
                "bits_per_int",
                svb_bpi,
                comp_bpi,
            );
        }
        for op in ["encode_deltas", "decode_deltas"] {
            push_comparison(
                &mut out,
                records,
                "int_codecs",
                &format!("{op} (GroupVarint)"),
                &format!("int_codecs/group_varint/{op}"),
                "GroupVarint",
                &format!("{comp_prefix}/{op}"),
                comp_label,
                "ns/int",
                "bits_per_int",
                gv_bpi,
                comp_bpi,
            );
        }
    }

    // 8. Radix Sort (10,000,000 elements: u32, u64, (u64, u64))
    for (kind, z_bench, z_var, z_space_key, comp_suffix, vor_space_key, rdst_space_key, std_space_key) in [
        (
            "u32",
            "radix_sort/lsd_u32",
            "RadixSort::sort_u32",
            "radix_sort/zipora_u32_bytes_per_elem",
            "u32",
            "radix_sort/voracious_u32_bytes_per_elem",
            "radix_sort/rdst_u32_bytes_per_elem",
            "radix_sort/std_u32_bytes_per_elem",
        ),
        (
            "u64",
            "radix_sort/lsd_u64",
            "RadixSort::sort_u64",
            "radix_sort/zipora_u64_bytes_per_elem",
            "u64",
            "radix_sort/voracious_u64_bytes_per_elem",
            "radix_sort/rdst_u64_bytes_per_elem",
            "radix_sort/std_u64_bytes_per_elem",
        ),
        (
            "(u64, u64)",
            "radix_sort/kv_u64_u64",
            "KeyValueRadixSort<u64, u64>",
            "radix_sort/zipora_kv_bytes_per_elem",
            "kv_u64_u64",
            "radix_sort/voracious_kv_bytes_per_elem",
            "radix_sort/rdst_kv_bytes_per_elem",
            "radix_sort/std_kv_bytes_per_elem",
        ),
    ] {
        let z_bpe = space.get(z_space_key).copied().unwrap_or(0.0);
        let vor_bpe = space.get(vor_space_key).copied().unwrap_or(0.0);
        let rdst_bpe = space.get(rdst_space_key).copied().unwrap_or(0.0);
        let std_bpe = space.get(std_space_key).copied().unwrap_or(0.0);
        for (c_prefix, c_label, c_bpe) in [
            ("radix_sort/voracious", "voracious_radix_sort", vor_bpe),
            ("radix_sort/rdst", "rdst (single-threaded)", rdst_bpe),
            ("radix_sort/std_unstable", "[T]::sort_unstable", std_bpe),
        ] {
            push_comparison(
                &mut out,
                records,
                "radix_sort",
                &format!("sort_{kind} ({z_var})"),
                z_bench,
                z_var,
                &format!("{c_prefix}/{comp_suffix}"),
                c_label,
                "ns/elem",
                "bytes_per_elem",
                z_bpe,
                c_bpe,
            );
        }
    }

    // 9. Suffix Array (10,000,000 bytes construct + 100,000 16B substring queries)
    let sa_z_bpb = space
        .get("suffix_array/zipora_bytes_per_byte")
        .copied()
        .unwrap_or(5.0);
    for (c_prefix, c_label, c_space_key) in [
        (
            "suffix_array/suffix_crate",
            "suffix::SuffixTable",
            "suffix_array/suffix_crate_bytes_per_byte",
        ),
        (
            "suffix_array/divsufsort_rust",
            "divsufsort (Rust)",
            "suffix_array/divsufsort_rust_bytes_per_byte",
        ),
        (
            "suffix_array/cdivsufsort_c",
            "cdivsufsort (C)",
            "suffix_array/cdivsufsort_c_bytes_per_byte",
        ),
    ] {
        let c_bpb = space.get(c_space_key).copied().unwrap_or(5.0);
        push_comparison(
            &mut out,
            records,
            "suffix_array",
            "construct (10M bytes)",
            "suffix_array/construct",
            "SuffixArray",
            &format!("{c_prefix}/construct"),
            c_label,
            "ns/byte",
            "bytes_per_byte",
            sa_z_bpb,
            c_bpb,
        );
        push_comparison(
            &mut out,
            records,
            "suffix_array",
            "search_query (10M bytes, 16B pattern)",
            "suffix_array/search_query",
            "SuffixArray",
            &format!("{c_prefix}/search_query"),
            c_label,
            "ns/op",
            "bytes_per_byte",
            sa_z_bpb,
            c_bpb,
        );
    }

    // 10. String Search (64 KiB strchr & strstr)
    push_comparison(
        &mut out,
        records,
        "search_and_scoring",
        "strchr_64kib (SimdStringSearch)",
        "search_and_scoring/simd_strchr_64kib",
        "SimdStringSearch::sse42_strchr",
        "search_and_scoring/memchr_strchr_64kib",
        "memchr::memchr",
        "ns/call",
        "aux_bytes",
        0.0,
        0.0,
    );
    for (c_bench, c_label) in [
        ("search_and_scoring/memmem_strstr_64kib", "memchr::memmem::find"),
        (
            "search_and_scoring/memmem_finder_strstr_64kib",
            "memchr::memmem::Finder",
        ),
        (
            "search_and_scoring/aho_corasick_strstr_64kib",
            "aho_corasick::AhoCorasick",
        ),
    ] {
        push_comparison(
            &mut out,
            records,
            "search_and_scoring",
            "strstr_64kib (SimdStringSearch)",
            "search_and_scoring/simd_strstr_64kib",
            "SimdStringSearch::sse42_strstr",
            c_bench,
            c_label,
            "ns/call",
            "aux_bytes",
            0.0,
            0.0,
        );
    }

    out
}

fn write_json(path: &Path, report: &CompetitorBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote competitor benchmark report to {}", path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let smoke_mode = args.iter().any(|a| a == "--smoke")
        || std::env::var("ZIPORA_BENCH_SMOKE").as_deref() == Ok("1");
    let save_baseline = args.iter().any(|a| a == "--save-baseline")
        || std::env::var("ZIPORA_BENCH_SAVE_BASELINE").as_deref() == Ok("1")
        || std::env::var("UPDATE_BASELINE").as_deref() == Ok("1");

    let custom_out: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--output" || w[0] == "--json")
        .map(|w| PathBuf::from(&w[1]))
        .or_else(|| std::env::var("ZIPORA_BENCH_OUT").ok().map(PathBuf::from));

    let cli_filter: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--filter" || w[0] == "--only")
        .map(|w| w[1].clone())
        .or_else(|| {
            args.iter()
                .find_map(|a| a.strip_prefix("--filter=").map(str::to_string))
        });
    init_bench_filter(cli_filter);

    let thp_on = args.iter().any(|a| a == "--thp-on")
        || matches!(
            std::env::var("ZIPORA_BENCH_THP").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        );
    if !thp_on {
        disable_transparent_hugepages_for_self();
    }
    let logical_cores = hardware_logical_cores();
    let primary_core = select_primary_core(logical_cores);
    pin_current_thread_to_core(primary_core);
    init_self_pin_tracker(primary_core, logical_cores);
    if has_active_filter() {
        wait_for_quiet_host_and_repin(12_000, true);
    }

    let mut records = Vec::with_capacity(128);
    let mut space_metrics = BTreeMap::new();

    let sizes = if smoke_mode {
        SuiteSizes::smoke()
    } else {
        SuiteSizes::spec()
    };
    let tier = if smoke_mode { "smoke" } else { "spec" };
    let cat_filter = std::env::var("ZIPORA_BENCH_CAT").ok();
    let run_cat = |cat: &str| -> bool { should_run_category(cat, cat_filter.as_deref()) };

    let suite_start = Instant::now();

    if run_cat("rank_select") {
        bench_rank_select(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("elias_fano") {
        bench_elias_fano(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("tries") {
        bench_tries(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("hash_maps") {
        bench_hash_maps(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("memory_pools") {
        bench_memory_pools(&mut records, &mut space_metrics, logical_cores, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("entropy") {
        bench_entropy(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("int_codecs") {
        bench_int_codecs(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("radix_sort") {
        bench_radix_sort_lsd(&mut records, &mut space_metrics, sizes);
        bench_radix_sort_kv(&mut records, &mut space_metrics, primary_core, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("suffix_array") {
        bench_suffix_array(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("search_and_scoring") {
        bench_search_and_scoring(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }

    let elapsed = suite_start.elapsed();
    let mut host_metadata = collect_host_metadata(primary_core);
    host_metadata.runner_command = format!(
        "RUSTFLAGS=\"{}\" cargo bench --locked --bench competitor_bench && python3 scripts/bench_compare.py --suite competitor_bench --confirm-retries 5",
        host_metadata.rustflags
    );
    let same_run_comparisons = build_same_run_comparisons(&records, &space_metrics);

    eprintln!(
        "=== Zipora E2 Competitor Benchmark Suite (tier={}, host={}, cpu={}, cores={}, pinned_core={}, commit={}) ===",
        tier,
        host_metadata.host,
        host_metadata.cpu_model,
        host_metadata.logical_cores,
        primary_core,
        host_metadata.git_commit
    );

    for rec in &records {
        eprintln!(
            "  [{:<18}] {:<48} N={:<10} min={:>8.3} {:<7} med={:>8.3} {:<7} p95={:>8.3} (spread={:>5.2}%)",
            rec.category,
            rec.name,
            rec.dataset_size,
            rec.min_ns_per_op,
            rec.unit,
            rec.median_ns_per_op,
            rec.unit,
            rec.p95_ns_per_op,
            rec.spread_pct
        );
    }

    eprintln!(
        "=== Completed {} competitor benchmarks ({} comparisons, tier={}) in {:.2}s ===",
        records.len(),
        same_run_comparisons.len(),
        tier,
        elapsed.as_secs_f64()
    );

    let report = CompetitorBenchReport {
        suite: "competitor_bench".to_string(),
        tier: tier.to_string(),
        seed: SEED_BASE,
        entropy_block_source: ENTROPY_BLOCK_SOURCE.to_string(),
        host_metadata: host_metadata.clone(),
        competitor_configs: build_competitor_configs(),
        dataset_sizes: sizes.to_map(),
        space_metrics,
        same_run_comparisons,
        benchmarks: records,
    };

    let default_out = if smoke_mode {
        PathBuf::from("target/bench/competitor_smoke_bench.json")
    } else if has_active_filter() || cat_filter.is_some() {
        PathBuf::from("target/bench/competitor_bench_filtered.json")
    } else {
        PathBuf::from("target/bench/competitor_bench.json")
    };
    let latest_out = custom_out.unwrap_or(default_out);
    write_json(&latest_out, &report);

    if save_baseline && !smoke_mode && !has_active_filter() && cat_filter.is_none() {
        let baseline_path = PathBuf::from(format!(
            "benches/baselines/{}/competitor_bench.json",
            host_metadata.host
        ));
        write_json(&baseline_path, &report);
    }
}
