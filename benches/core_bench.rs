//! Canonical Phase E Core Benchmark Suite (`plan.md` §1.6 `E1` + `S36-R2` + `S37-R1` + `S38-R1..R4`).
//!
//! Runs a deterministic, fixed-seed benchmark suite at canonical specification sizes
//! across all 10 core subsystems using production allocator settings, self-only CPU
//! affinity, and uniform full-structure query distributions (zero sample replay):
//! 1. Rank/Select (`1 Gbit` = `1,000,000,000` bits: `Rank9`, `RankSelectInterleaved256`,
//!    `RankSelectSE256` `rank1`/`select1`/`select0` over `1,000,000` uniform random queries)
//! 2. Elias-Fano (`100,000` elements across uniform, Zipf-clustered, and dense distributions:
//!    `EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`
//!    `next_geq`, `advance_to_geq`, `advance_to_index` over `100,000` distinct queries, and
//!    `bits_per_element` incl. `HybridPostingList`)
//! 3. Tries (`1,000,000` keys across 3 distributions `url_paths`, `random_16b`, `shared_prefix`:
//!    `DoubleArrayTrie` `build_from_sorted` + `lookup`, `CsppTrie` `build` + `lookup`,
//!    and `ZiporaTrie` DoubleArray `build` + `lookup` across all `1,000,000` shuffled keys)
//! 4. Hash Maps (`1,000,000` `u64` keys and `1,000,000` 16-byte string keys:
//!    `ZiporaHashMap` and `GoldHashMap` `insert_u64`, `insert_str16`, `get_u64`, `get_str16`
//!    across all `1,000,000` keys)
//! 5. Memory Pools (`10,000` ops: `LockFreeMemoryPool` and `SecureMemoryPool` alloc+free on 1 and 4 threads)
//! 6. Entropy Coding (`65,536` bytes = `64 KiB` deterministic synthetic `enwik8`-like block:
//!    `HuffmanEncoder` O0, `ContextualHuffmanEncoder` O1/O2, `FseEncoder`, `Rans64Encoder` X1/X4
//!    encode+decode)
//! 7. Integer Codecs (`100,000` contiguous integers: `StreamVByte` and `GroupVarint` encode/decode deltas)
//! 8. Radix Sort (`10,000,000` = `10^7` elements: `RadixSort` `u32`, `u64`, and
//!    `KeyValueRadixSort<u64, u64>` `(u64, u64)`)
//! 9. Suffix Array (`10,000,000` = `10^7` bytes: `SuffixArray` construction + `100,000` uniform substring queries)
//! 10. String Search & BM25 Scoring (`64 KiB` `SimdStringSearch` `strchr`/`strstr`, `100,000`-doc
//!     `Bm25BatchScorer` `score`/`batch_score`/`batch_score_with_prefetch`, and
//!     `FieldnormEncoder::build_score_table` for `S36-R2`).
//!
//! A separate smaller smoke tier is available via `--smoke` or `ZIPORA_BENCH_SMOKE=1`
//! (`make bench_smoke`), which writes to `target/bench/smoke_bench.json` and never
//! overwrites `core_bench.json` or any committed baseline.
//!
//! Emits structured JSON to `target/bench/core_bench.json`, and writes
//! `benches/baselines/<host>/core_bench.json` **only** when `UPDATE_BASELINE=1`,
//! `ZIPORA_BENCH_SAVE_BASELINE=1`, or `--save-baseline` is explicitly provided.

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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
use zipora::memory::{LockFreeMemoryPool, LockFreePoolConfig, SecureMemoryPool, SecurePoolConfig};
use zipora::scoring::{Bm25BatchScorer, FieldnormEncoder};
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

fn bench_rank_select(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let num_bits = sizes.rank_select_bits;
    let num_words = num_bits / 64;
    let num_queries = sizes.rank_select_queries;
    // Step continuously through `batch = 10,000` (`rank1`), `7,000` (`select1`),
    // or `6,250` (`select0`) distinct uniform queries per burst across
    // `trials = 10..12, bursts_per_trial = 8..16` (`>= 1,000,000` distinct queries total across
    // `0..num_bits` / `0..max_ones` / `0..max_zeros`, with zero replayed samples).
    let query_batch = (num_queries / 100).max(1_000).min(num_queries.max(1));
    let select_batch = (num_queries / 140).max(1_000).min(num_queries.max(1));
    let select0_interleaved_batch = (num_queries / 160).max(1_000).min(num_queries.max(1));

    let mut rank_params = BenchParams::heavy(num_bits as u64, query_batch as u64, 12);
    rank_params.warmup_calls = 8;
    rank_params.bursts_per_trial = 8;
    let mut select1_params =
        BenchParams::heavy(num_bits as u64, select_batch as u64, 10).with_threshold(16.0);
    select1_params.warmup_calls = 6;
    select1_params.bursts_per_trial = 14;
    let mut select0_interleaved_params =
        BenchParams::heavy(num_bits as u64, select0_interleaved_batch as u64, 10)
            .with_threshold(9.0);
    select0_interleaved_params.warmup_calls = 6;
    select0_interleaved_params.bursts_per_trial = 16;

    let rank_seed = SEED_BASE ^ 0x0101;
    let select1_seed = SEED_BASE ^ 0x0102;
    let select0_seed = SEED_BASE ^ 0x0103;
    let base_words = gen_rank_words(num_words);
    let total_ones: usize = base_words.iter().map(|w| w.count_ones() as usize).sum();
    let max_ones = total_ones.max(1);
    let max_zeros = num_bits.saturating_sub(total_ones).max(1);

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

    if should_run_any(&[
        "rank_select/interleaved256/rank1",
        "rank_select/interleaved256/select1",
        "rank_select/interleaved256/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        let rsi = RankSelectInterleaved256::new(BitVector::from_blocks(words, num_bits)).unwrap();
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

    if should_run_any(&[
        "rank_select/se256/rank1",
        "rank_select/se256/select1",
        "rank_select/se256/select0",
    ]) {
        maybe_repin_self_within_ccd();
        let words = base_words.clone();
        drop(base_words);
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
}

// ---------------------------------------------------------------------------
// 2. Elias-Fano Posting Lists (100,000 elements, 100,000 distinct queries)
// ---------------------------------------------------------------------------

fn bench_elias_fano(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.elias_fano_elements;
    let num_queries = n; // 100,000 distinct queries (== structure size, zero replay)
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
        // Construct a 4-instance ring for each Elias-Fano variant separated by 4,672-byte
        // spacers (`4096 + 576`, coprime to 64B cache lines and 4KiB pages) so binary-search
        // / select tables step across cache-line and page offsets across the bursts.
        let mut ef_spacers: Vec<Vec<u8>> = Vec::with_capacity(16);
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

        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x04);
        let universe = *data.last().unwrap() as u64 + 1;
        // 100,000 distinct random targets across 0..universe (zero replay), replicated
        // across a 4-buffer ring separated by non-power-of-two spacers so `targets` does
        // not suffer a systematic 512 KiB L2 set conflict with any single EF replica.
        let targets: Vec<u64> = (0..num_queries)
            .map(|_| rng.random_range(0..universe))
            .collect();
        let targets_ring: [Vec<u64>; 4] = std::array::from_fn(|_| {
            ef_spacers.push(vec![1u8; 136 * 1024 + 4_672]);
            targets.clone()
        });
        drop(targets);
        // 100 independent sorted streams of 1,000 distinct targets = 100,000 distinct
        // cursor `advance_to_geq` queries across the entire posting list (zero replay).
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

        // All `n` indices shuffled into a uniform random permutation (zero replay)
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
        drop(ef_spacers);
    }
}

// ---------------------------------------------------------------------------
// 3. Tries (1,000,000 keys across 3 distributions, lookup on all 1,000,000 keys)
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
        if !run_dat && !run_cspp {
            continue;
        }
        maybe_repin_self_within_ccd();
        let perm = BijectivePerm::new(n, SEED_BASE ^ 0x0500 ^ (dist_idx as u64));
        let num_keys = n as u64;
        // Step continuously through `lookup_batch = 4,000` permuted keys per burst
        // across `warmup_calls = 10, trials = 14, bursts_per_trial = 18` (`1,048,000 >= 1,000,000`
        // queries visiting all `1,000,000` keys in the trie across `0..n`, with zero replayed samples).
        let mut lookup_params = BenchParams::heavy(num_keys, lookup_batch as u64, 14);
        lookup_params.warmup_calls = 10;
        lookup_params.bursts_per_trial = 18;

        let mut dat_build_params = BenchParams::heavy(num_keys, num_keys, 3);
        dat_build_params.warmup_calls = 1;
        dat_build_params.bursts_per_trial = 1;
        let mut cspp_build_params = BenchParams::heavy(num_keys, num_keys, 3);
        cspp_build_params.warmup_calls = 1;
        cspp_build_params.bursts_per_trial = 1;

        let warm_perm = BijectivePerm::new(n, SEED_BASE ^ 0x0550 ^ (dist_idx as u64));

        // 1. DoubleArrayTrie (build_from_sorted + lookup on all 1,000,000 permuted keys).
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
            record_space_metric(
                space,
                &format!("tries/dat_{dist_name}_bytes_per_state"),
                bytes_per_state,
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

        // 2. CsppTrie (build + lookup on all 1,000,000 permuted keys).
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
                        debug_assert_eq!(found, lookup_batch);
                        black_box(found);
                    },
                );
            }
            drop(cspps);
            drop(_cspp_spacers);
        }
    }

    // 3. ZiporaTrie (DoubleArray strategy) build + lookup on all 1,000,000 permuted keys.
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
        let need_z_lookup = should_run_bench("tries/zipora_trie_da/lookup");
        let eff_z_replicas = if need_z_lookup { Z_REPLICAS } else { 0 };
        let mut built_ztries = Vec::with_capacity(eff_z_replicas);
        let mut zspacers = Vec::with_capacity(eff_z_replicas + 1);
        if need_z_lookup && let Some(last) = z_pending.take() {
            built_ztries.push(last);
        }
        while built_ztries.len() < eff_z_replicas {
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

    if should_run_bench("tries/zipora_trie_da/lookup") {
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
    }
    drop(ztries);
    drop(_ztrie_spacers);
}

// ---------------------------------------------------------------------------
// 4. Hash Maps (1,000,000 u64 keys & 1,000,000 16-byte string keys, zero replay)
// ---------------------------------------------------------------------------

fn bench_hash_maps(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
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
    // Size GoldHashMap's bucket directory:
    // - `u64`: `cap * 4` -> `next_prime = 16,601,593` u32 buckets = 63.33 MiB + 24 MiB nodes = 87.3 MiB per map
    // - `[u8; 16]`: `n * 5` -> `next_prime = 8,175,383` u32 buckets = 31.19 MiB + 32 MiB nodes = 63.2 MiB per map
    // so both tables insert `>= 1,000,000` distinct keys without rehashing.
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

    // 1. ZiporaHashMap u64 (insert >= 1M distinct keys across a 180-degree staggered 2-map ring,
    //    then get all 1M permuted keys across the 2-map ring)
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

    // 2. ZiporaHashMap [u8; 16] (insert >= 1M distinct keys across a 180-degree staggered 2-map ring,
    //    then get all 1M permuted keys across the 2-map ring)
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

    // 3. GoldHashMap u64 (63.33 MiB buckets + 24 MiB nodes = 87.3 MiB > 32 MiB L3)
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_u64",
        "hash_maps/gold_hash_map/get_u64",
    ]) {
        maybe_repin_self_within_ccd();
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

    // 4. GoldHashMap [u8; 16] (31.19 MiB buckets + 32 MiB nodes = 63.19 MiB per map, > 126 MiB across 2-map ring)
    if should_run_any(&[
        "hash_maps/gold_hash_map/insert_str16",
        "hash_maps/gold_hash_map/get_str16",
    ]) {
        maybe_repin_self_within_ccd();
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
}

// ---------------------------------------------------------------------------
// 5. Memory Pools (LockFreeMemoryPool & SecureMemoryPool, 1T & 4T)
// ---------------------------------------------------------------------------

fn bench_memory_pools(records: &mut Vec<BenchRecord>, logical_cores: usize, sizes: SuiteSizes) {
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

    // LockFreeMemoryPool 1T
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

    // LockFreeMemoryPool 4T
    maybe_repin_self_within_ccd();
    set_allow_self_repin(false);
    let p_core_4t = current_self_pinned_core();
    pin_current_thread_to_core(p_core_4t);
    const POOL_SIZES: [usize; 4] = [128, 256, 512, 1024];
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

    // SecureMemoryPool 1T
    set_allow_self_repin(true);
    maybe_repin_self_within_ccd();
    let mut sec_cfg = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg.enable_hot_cold_separation = false;
    sec_cfg.enable_huge_pages = false;
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
    let sec_batch = 16usize;
    let sec_rounds = (ops / sec_batch).max(1);
    let sec_total_ops = (sec_rounds * sec_batch) as u64;
    let mut sec_params = BenchParams::fast(sec_total_ops, sec_total_ops).with_threshold(9.0);
    sec_params.trials = 9;
    sec_params.bursts_per_trial = 4;

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

    let quarter = (sec_rounds / 4).max(1);
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

    // SecureMemoryPool 4T
    maybe_repin_self_within_ccd();
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
    set_allow_self_repin(true);
}

// ---------------------------------------------------------------------------
// 6. Entropy Coding (Huffman O0/O1/O2, FSE, rANS on 64 KiB synthetic enwik8-like block)
// ---------------------------------------------------------------------------

/// Generates a deterministic 64 KiB synthetic `enwik8`-like block composed of
fn bench_entropy(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
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

    // Huffman O0 (64 KiB)
    let huff_o0_enc = HuffmanEncoder::new(&block).unwrap();
    let huff_o0_compressed = huff_o0_enc.encode(&block).unwrap();
    let huff_o0_dec = HuffmanDecoder::new(huff_o0_enc.tree().clone());

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

    // Huffman O1 (full 64 KiB block for encode_x1, encode_x4, and decode)
    // Construct each of the 8 `ContextualHuffmanDecoder` instances on its own thread so
    // `std::collections::hash_map::RandomState` (which initializes `thread_local!(KEYS)`
    // once per thread from `getrandom`) draws 8 genuinely independent `(k0, k1)` seeds and
    // heap layouts.
    let huff_o1_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap();

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

    // Huffman O2 (full 64 KiB block for encode and decode)
    // Construct all 8 `ContextualHuffmanEncoder` and `ContextualHuffmanDecoder` instances
    // on 8 concurrently alive OS threads (Barrier-synchronized) so each draws an independent
    // per-thread `RandomState` `(k0, k1)` seed and distinct glibc per-thread `malloc` arena.
    {
        type ContextualO2EncPair = (ContextualHuffmanEncoder, Vec<u8>);
        type ContextualO2EncRing = (usize, [ContextualO2EncPair; 8], [Vec<u8>; 2]);

        let mut built = build_on_concurrent_threads(|blk, spacer| {
            black_box(spacer.as_ptr());
            let enc = ContextualHuffmanEncoder::new(&blk, HuffmanOrder::Order2).unwrap();
            drop(spacer);
            (Some(enc), None, blk)
        });
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

    // FSE (64 KiB)
    let mut fse_enc = FseEncoder::new(FseConfig::default()).unwrap();
    let fse_compressed = fse_enc.compress(&block).unwrap();

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

    // rANS64 X1 and X4 (64 KiB)
    let mut freqs = [1u32; 256];
    for &b in &block {
        freqs[b as usize] += 1;
    }
    let rans_x1_enc = Rans64Encoder::<ParallelX1>::new(&freqs).unwrap();
    let rans_x1_compressed = rans_x1_enc.encode(&block).unwrap();
    let rans_x1_dec = Rans64Decoder::<ParallelX1>::new(&rans_x1_enc);

    let rans_x4_enc = Rans64Encoder::<ParallelX4>::new(&freqs).unwrap();
    let rans_x4_compressed = rans_x4_enc.encode(&block).unwrap();
    let rans_x4_dec = Rans64Decoder::<ParallelX4>::new(&rans_x4_enc);

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
    drop(_brk_anchor);
}

// ---------------------------------------------------------------------------
// 7. Integer Codecs (100,000 contiguous integers by default)
// ---------------------------------------------------------------------------

fn bench_int_codecs(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
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

    let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
    let mut svb_out = vec![0u32; n];

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

    let mut svb_into_params = BenchParams::fast(n as u64, (n * 32) as u64).with_threshold(9.0);
    svb_into_params.warmup_calls = 2;
    svb_into_params.trials = 11;
    svb_into_params.bursts_per_trial = 4;
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

    let mut svb_deltas_params = BenchParams::fast(n as u64, (n * 16) as u64).with_threshold(9.0);
    svb_deltas_params.warmup_calls = 2;
    svb_deltas_params.trials = 11;
    svb_deltas_params.bursts_per_trial = 4;
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

    let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);
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
    drop(_brk_anchor);
}

// ---------------------------------------------------------------------------
// 8. Radix Sort (10,000,000 = 10^7 elements for u32, u64, (u64, u64) by default)
// ---------------------------------------------------------------------------

fn bench_radix_sort_lsd(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
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

    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    // Pre-fault a contiguous `96 MiB` (`4 x 24 MiB`) `brk` hole below `_lsd_top_anchor`
    // so `RadixSort::sort_u32`'s `40 MiB` buffer and `RadixSort::sort_u64`'s `80 MiB` buffer
    // are served from pre-faulted user-space `brk` memory without `mmap`/`munmap` page faults.
    let mut lsd_holes: Vec<Vec<u8>> = Vec::with_capacity(4);
    for _ in 0..4 {
        lsd_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
    }
    let _lsd_top_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(lsd_holes);

    // 1. LSD u32 (10^7 elements across a 2-buffer / 2-sorter ring with 352 KiB spacer)
    {
        let template_u32 = gen_radix_u32_template(n);
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

    // 2. LSD u64 (10^7 elements)
    {
        let template_u64 = gen_radix_u64_template(n);
        let mut work_u64 = (Vec::<Vec<u8>>::with_capacity(16), vec![0u64; n]);
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
    drop(_lsd_top_anchor);
}

fn bench_radix_sort_kv(records: &mut Vec<BenchRecord>, primary_core: usize, sizes: SuiteSizes) {
    maybe_repin_self_within_ccd();
    let n = sizes.radix_sort_elements;
    // 3. KeyValueRadixSort<u64, u64> (10^7 (u64, u64) elements)
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
    // Pre-fault a contiguous `432 MiB` (`18 x 24 MiB`) `brk` hole below `_kv_top_anchor`
    // so `KeyValueRadixSort::sort_by_key`'s `160 MiB` `indices`, `160 MiB` `buffer`, and
    // `80 MiB` `targets` allocations are served from pre-faulted user-space `brk` memory
    // without `mmap`/`munmap` page-fault churn.
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
    drop(_kv_top_anchor);
    set_allow_self_repin(true);
}

// ---------------------------------------------------------------------------
// 9. Suffix Array (10,000,000 = 10^7 bytes + 100,000 uniform substring queries)
// ---------------------------------------------------------------------------

fn bench_suffix_array(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    maybe_repin_self_within_ccd();
    let n = sizes.suffix_array_bytes;
    let text0 = gen_enwik_like_block(n, SEED_BASE ^ 0x0A);
    drop(black_box(vec![1u8; 31 * 1024 * 1024]));
    // Pre-fault a contiguous `432 MiB` (`18 x 24 MiB`) `brk` hole below `_brk_top_anchor`
    // so `SuffixArray::new`'s DC3 recursive working vectors (`s_padded`, `s12`, `sa12`, `s0`,
    // `sa0`, `ranks`, `sa` = `40 MiB` each) are served from pre-faulted user-space `brk`
    // memory without `mmap`/`munmap` page-fault churn across trials.
    let mut sa_holes: Vec<Vec<u8>> = Vec::with_capacity(18);
    for _ in 0..18 {
        sa_holes.push(black_box(vec![1u8; 24 * 1024 * 1024]));
    }
    let _brk_top_anchor = black_box(vec![1u8; 64 * 1024]);
    drop(sa_holes);

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
    if should_run_bench("suffix_array/search_query") {
        let sa0 = sa_pending
            .take()
            .unwrap_or_else(|| SuffixArray::new(&text0).unwrap());
        let _text_spacer = vec![1u8; 1_376 * 1024];
        let text1 = text0.clone();
        let _sa_spacer = vec![1u8; 3_424 * 1024];
        let sa1 = SuffixArray::new(&text1).unwrap();
        let sas = [sa0, sa1];
        let texts = [text0, text1];

        // Sample distinct 16-byte substrings uniformly across the entire 10,000,000-byte
        // text via a stratified bijection (`sa_perm` over `1,000,000` 10-byte windows at
        // spec size, with 10% perturbed non-matching queries), stepping continuously through
        // `sa_batch = 1,000` distinct queries per burst across `warmup_calls = 8, trials = 12,
        // bursts_per_trial = 12` (`152,000 >= 100,000` distinct queries total, zero replay).
        let num_queries = sizes.suffix_array_queries;
        let sa_batch = (num_queries / 100).max(500).min(num_queries.max(1));
        let pat_len = 16usize;
        let max_start = n - pat_len;
        let sa_query_seed = SEED_BASE ^ 0x0A01;
        let sa_perm_len = ((n / 10) / 1000 * 1000).max(1_000);
        let sa_window = (max_start / sa_perm_len).max(1);
        let sa_perm = BijectivePerm::new(sa_perm_len, SEED_BASE ^ 0x0A02);

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

        let mut sa_search_params = BenchParams::heavy(n as u64, sa_batch as u64, 12);
        sa_search_params.warmup_calls = 8;
        sa_search_params.bursts_per_trial = 12;

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
    }
    drop(_brk_top_anchor);
}

// ---------------------------------------------------------------------------
// 10. String Search & BM25 Scoring (including S36-R2 build_score_table)
// ---------------------------------------------------------------------------

fn bench_search_and_scoring(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    maybe_repin_self_within_ccd();
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

        for _ in 0..2_048usize {
            if let Some(pos) = searcher.sse42_strstr(black_box(haystack0), black_box(needle)) {
                pre_acc = pre_acc.wrapping_add(pos);
            }
        }
        black_box(pre_acc);

        let mut strstr_params = str_params.with_threshold(9.0);
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
                for _ in 0..search_iters {
                    if let Some(pos) =
                        searcher.sse42_strstr(black_box(haystack0), black_box(needle))
                    {
                        acc = acc.wrapping_add(pos);
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

#[inline(never)]
fn _zipora_core_bench_text_pad(x: u64) -> u64 {
    let a = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let b = a.rotate_left(17).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    b.wrapping_add(0x94D0_49BB_1331_11EB)
}

fn write_json(path: &Path, report: &CoreBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote benchmark report to {}", path.display());
}

fn main() {
    init_bench_allocator_policy();
    black_box(_zipora_core_bench_text_pad(black_box(0)));
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

    let mut search_records = Vec::with_capacity(8);
    if run_cat("search_and_scoring") {
        trim_allocator_arena();
        bench_search_and_scoring(&mut search_records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("rank_select") {
        trim_allocator_arena();
        bench_rank_select(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("elias_fano") {
        trim_allocator_arena();
        bench_elias_fano(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("tries") {
        trim_allocator_arena();
        bench_tries(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("hash_maps") {
        trim_allocator_arena();
        bench_hash_maps(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("memory_pools") {
        trim_allocator_arena();
        bench_memory_pools(&mut records, logical_cores, sizes);
        set_allow_self_repin(true);
        pin_current_thread_to_core(current_self_pinned_core());
        maybe_repin_self_within_ccd();
    }
    if run_cat("entropy") {
        trim_allocator_arena();
        bench_entropy(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("int_codecs") {
        trim_allocator_arena();
        bench_int_codecs(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("radix_sort") {
        if should_run_any(&["radix_sort/lsd_u32", "radix_sort/lsd_u64"]) {
            trim_allocator_arena();
            bench_radix_sort_lsd(&mut records, sizes);
            maybe_repin_self_within_ccd();
        }
        if should_run_bench("radix_sort/kv_u64_u64") {
            trim_allocator_arena();
            bench_radix_sort_kv(&mut records, current_self_pinned_core(), sizes);
            maybe_repin_self_within_ccd();
        }
    }
    if run_cat("suffix_array") {
        trim_allocator_arena();
        bench_suffix_array(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    records.append(&mut search_records);
    flush_prealloc_space_metrics(&mut space_metrics);

    let elapsed = suite_start.elapsed();
    let host_metadata = collect_host_metadata(primary_core);

    eprintln!(
        "=== Zipora E1 Core Benchmark Suite (tier={}, host={}, cpu={}, cores={}, pinned_core={}, commit={}) ===",
        tier,
        host_metadata.host,
        host_metadata.cpu_model,
        host_metadata.logical_cores,
        primary_core,
        host_metadata.git_commit
    );

    for rec in &records {
        eprintln!(
            "  [{:<18}] {:<44} N={:<10} min={:>8.3} {:<7} med={:>8.3} {:<7} p95={:>8.3} (spread={:>5.2}%)",
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
        "=== Completed {} core benchmarks (tier={}) in {:.2}s ===",
        records.len(),
        tier,
        elapsed.as_secs_f64()
    );

    let suite_name = if smoke_mode {
        "smoke_bench"
    } else {
        "core_bench"
    };
    let report = CoreBenchReport {
        suite: suite_name.to_string(),
        tier: tier.to_string(),
        seed: SEED_BASE,
        entropy_block_source: ENTROPY_BLOCK_SOURCE.to_string(),
        host_metadata: host_metadata.clone(),
        dataset_sizes: sizes.to_map(),
        space_metrics,
        benchmarks: records,
    };

    let default_out = if smoke_mode {
        PathBuf::from("target/bench/smoke_bench.json")
    } else if has_active_filter() || cat_filter.is_some() {
        PathBuf::from("target/bench/core_bench_filtered.json")
    } else {
        PathBuf::from("target/bench/core_bench.json")
    };
    let latest_out = custom_out.unwrap_or(default_out);
    write_json(&latest_out, &report);

    if save_baseline && !smoke_mode && !has_active_filter() && cat_filter.is_none() {
        let baseline_path = PathBuf::from(format!(
            "benches/baselines/{}/core_bench.json",
            host_metadata.host
        ));
        write_json(&baseline_path, &report);
    }
}
