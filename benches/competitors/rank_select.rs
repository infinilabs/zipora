//! Family 1: Rank / Select (1 Gbit = 1,000,000,000 bits, 1,000,000 uniform random queries).
//!
//! Zipora: `Rank9`, `RankSelectInterleaved256`, `RankSelectSE256`
//! Competitors: `sucds::Rank9Sel`, `vers_vecs::RsVec`, `bitm::RankSelect101111`
//!
//! Fairness invariant (`S42-R1`): every implementation is benchmarked on 1 instance over
//! the exact same 1 Gbit bitvector, query batch size, and `uniform_index_at` query stream.

use std::collections::BTreeMap;
use std::hint::black_box;

use bitm::{
    CombinedSampling, Rank as BitmRank, RankSelect101111, Select as BitmSelect,
    Select0 as BitmSelect0,
};
use dyn_size_of::GetSize;
use sucds::Serializable as SucdsSerializable;
use sucds::bit_vectors::{
    BitVector as SucdsBitVector, Rank as SucdsRank, Rank9Sel as SucdsRank9Sel,
    Select as SucdsSelect,
};
use vers_vecs::{BitVec as VersBitVec, RsVec as VersRsVec};
use zipora::succinct::BitVector;
use zipora::succinct::rank_select::{
    Rank9, RankSelectInterleaved256, RankSelectOps, RankSelectSE256,
};

use crate::common::*;

pub fn bench_rank_select(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let num_bits = sizes.rank_select_bits;
    let num_words = num_bits / 64;
    let num_queries = sizes.rank_select_queries;
    let query_batch = (num_queries / 100).max(1_000).min(num_queries.max(1));
    let select_batch = (num_queries / 140).max(1_000).min(num_queries.max(1));
    let select0_batch = (num_queries / 160).max(1_000).min(num_queries.max(1));

    let mut rank_params = BenchParams::heavy(num_bits as u64, query_batch as u64, 12);
    rank_params.warmup_calls = 8;
    rank_params.bursts_per_trial = 8;
    let mut select1_params = BenchParams::heavy(num_bits as u64, select_batch as u64, 10);
    select1_params.warmup_calls = 6;
    select1_params.bursts_per_trial = 14;
    let mut select0_params = BenchParams::heavy(num_bits as u64, select0_batch as u64, 10);
    select0_params.warmup_calls = 6;
    select0_params.bursts_per_trial = 16;

    let rank_seed = SEED_BASE ^ 0x0101;
    let select1_seed = SEED_BASE ^ 0x0102;
    let select0_seed = SEED_BASE ^ 0x0103;
    let base_words = gen_rank_words(num_words);
    let raw_bytes = (num_words * 8) as f64;
    let total_ones: usize = base_words.iter().map(|w| w.count_ones() as usize).sum();
    let max_ones = total_ones.max(1);
    let max_zeros = num_bits.saturating_sub(total_ones).max(1);

    // 1a. Zipora Rank9 (1 instance)
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

    // 1b. Zipora RankSelectInterleaved256 (1 instance)
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
            select0_params,
            || {
                let base = current_burst_index() * select0_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(rsi.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1c. Zipora RankSelectSE256 (1 instance)
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
            select0_params,
            || {
                let base = current_burst_index() * select0_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(rs_se.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1d. Competitor: sucds::bit_vectors::Rank9Sel (1 instance)
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
        for bit in (0..num_bits).step_by(4096) {
            warm = warm.wrapping_add(sucds_rs.rank1(bit).unwrap_or(0));
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

        for k in (0..max_ones).step_by(4096) {
            warm = warm.wrapping_add(sucds_rs.select1(k).unwrap_or(0));
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

        for k in (0..max_zeros).step_by(4096) {
            warm = warm.wrapping_add(sucds_rs.select0(k).unwrap_or(0));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/sucds_rank9sel/select0",
            "ns/op",
            select0_params,
            || {
                let base = current_burst_index() * select0_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(sucds_rs.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }

    // 1e. Competitor: vers_vecs::RsVec (1 instance)
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
        for bit in (0..num_bits).step_by(4096) {
            warm = warm.wrapping_add(vers_rs.rank1(bit));
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

        for k in (0..max_ones).step_by(4096) {
            warm = warm.wrapping_add(vers_rs.select1(k));
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

        for k in (0..max_zeros).step_by(4096) {
            warm = warm.wrapping_add(vers_rs.select0(k));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/vers_vecs_rsvec/select0",
            "ns/op",
            select0_params,
            || {
                let base = current_burst_index() * select0_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(vers_rs.select0(black_box(q)));
                }
                black_box(acc);
            },
        );
    }

    // 1f. Competitor: bitm::RankSelect101111<CombinedSampling, CombinedSampling> (1 instance)
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
        for bit in (0..num_bits).step_by(4096) {
            warm = warm.wrapping_add(bitm_rs.rank(bit));
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

        for k in (0..max_ones).step_by(4096) {
            warm = warm.wrapping_add(bitm_rs.select(k));
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

        for k in (0..max_zeros).step_by(4096) {
            warm = warm.wrapping_add(bitm_rs.select0(k));
        }
        black_box(warm);

        measure_bench(
            records,
            "rank_select",
            "rank_select/bitm_rs101111/select0",
            "ns/op",
            select0_params,
            || {
                let base = current_burst_index() * select0_batch;
                let mut acc = 0usize;
                for idx in base..base + select0_batch {
                    let q = uniform_index_at(select0_seed, idx, max_zeros);
                    acc = acc.wrapping_add(bitm_rs.select0(black_box(q)));
                }
                black_box(acc);
            },
        );
    }
}
