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

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bitm::{
    CombinedSampling, Rank as BitmRank, RankSelect101111, Select as BitmSelect,
    Select0 as BitmSelect0,
};
use dyn_size_of::GetSize;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sucds::Serializable as SucdsSerializable;
use sucds::bit_vectors::{
    BitVector as SucdsBitVector, Rank as SucdsRank, Rank9Sel as SucdsRank9Sel,
    Select as SucdsSelect,
};
use sucds::mii_sequences::{
    EliasFano as SucdsEliasFano, EliasFanoBuilder as SucdsEliasFanoBuilder,
};
use vers_vecs::{BitVec as VersBitVec, EliasFanoVec as VersEliasFanoVec, RsVec as VersRsVec};

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

    let elapsed = suite_start.elapsed();
    let host_metadata = collect_host_metadata(primary_core);
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
