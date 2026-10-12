//! Canonical Phase E Core Benchmark Suite (`plan.md` §1.6 `E1` + `S36-R2` + `S37-R1`).
//!
//! Runs a deterministic, fixed-seed benchmark suite at canonical specification sizes
//! across all 10 core subsystems:
//! 1. Rank/Select (`1 Gbit` = `1,073,741,824` bits: `Rank9`, `RankSelectInterleaved256`,
//!    `RankSelectSE256` `rank1`/`select1`/`select0`)
//! 2. Elias-Fano (`100,000` elements across uniform, Zipf-clustered, and dense distributions:
//!    `EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`, `ClusteredEliasFano`
//!    `next_geq`, `advance_to_geq`, `advance_to_index`, and `bits_per_element` incl. `HybridPostingList`)
//! 3. Tries (`1,000,000` keys across 3 distributions `url_paths`, `random_16b`, `shared_prefix`:
//!    `DoubleArrayTrie` `build_from_sorted` + `lookup`, `CsppTrie` `build` + `lookup`,
//!    and `ZiporaTrie` DoubleArray `build` + `lookup`)
//! 4. Hash Maps (`1,000,000` `u64` keys and `1,000,000` 16-byte string keys:
//!    `ZiporaHashMap` and `GoldHashMap` `insert_u64`, `insert_str16`, `get_u64`, `get_str16`)
//! 5. Memory Pools (`10,000` ops: `LockFreeMemoryPool` and `SecureMemoryPool` alloc+free on 1 and 4 threads)
//! 6. Entropy Coding (`65,536` bytes = `64 KiB` deterministic synthetic `enwik8`-like block:
//!    `HuffmanEncoder` O0, `ContextualHuffmanEncoder` O1/O2, `FseEncoder`, `Rans64Encoder` X1/X4
//!    encode+decode)
//! 7. Integer Codecs (`100,000` contiguous integers: `StreamVByte` and `GroupVarint` encode/decode deltas)
//! 8. Radix Sort (`10,000,000` = `10^7` elements: `RadixSort` `u32`, `u64`, and
//!    `KeyValueRadixSort<u64, u64>` `(u64, u64)`)
//! 9. Suffix Array (`10,000,000` = `10^7` bytes: `SuffixArray` construction + substring search)
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
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

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

const SEED_BASE: u64 = 0xE1C0_5EBE_4C00_2026;
const ENTROPY_BLOCK_SOURCE: &str = "synthetic_enwik8_like_64kib (deterministic Zipf-skewed English/Wiki markup token stream, seed = 0xE1C0_5EBE_4C00_2021)";

#[cfg(target_os = "linux")]
fn pin_current_thread_to_core(core_id: usize) {
    // SAFETY: zeroed cpu_set_t initialized with CPU_ZERO/CPU_SET and passed to sched_setaffinity
    // for the calling thread (pid = 0). Errors are ignored on restricted containers.
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(core_id, &mut cpuset);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_core(_core_id: usize) {}

fn select_primary_core(logical_cores: usize) -> usize {
    if let Ok(val) = std::env::var("ZIPORA_BENCH_PIN_CORE")
        && let Ok(parsed) = val.parse::<usize>()
        && parsed < logical_cores
    {
        return parsed;
    }
    if logical_cores > 62 {
        62
    } else {
        logical_cores.saturating_sub(1)
    }
}

#[derive(Debug, Clone, Copy)]
struct SuiteSizes {
    rank_select_bits: usize,
    elias_fano_elements: usize,
    trie_keys: usize,
    hash_map_keys: usize,
    memory_pool_ops: usize,
    entropy_block_bytes: usize,
    int_codec_elements: usize,
    radix_sort_elements: usize,
    suffix_array_bytes: usize,
    bm25_docs: usize,
}

impl SuiteSizes {
    fn spec() -> Self {
        Self {
            rank_select_bits: std::env::var("ZIPORA_BENCH_RANK_BITS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_073_741_824), // 1 Gbit (2^30 bits)
            elias_fano_elements: std::env::var("ZIPORA_BENCH_EF_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100_000),
            trie_keys: std::env::var("ZIPORA_BENCH_TRIE_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000), // 1 M keys
            hash_map_keys: std::env::var("ZIPORA_BENCH_MAP_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000), // 1 M keys
            memory_pool_ops: 10_000,
            entropy_block_bytes: 65_536, // 64 KiB
            int_codec_elements: 100_000,
            radix_sort_elements: std::env::var("ZIPORA_BENCH_SORT_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000_000), // 10^7 elements
            suffix_array_bytes: std::env::var("ZIPORA_BENCH_SA_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_000_000), // 10^7 bytes
            bm25_docs: 100_000,
        }
    }

    fn smoke() -> Self {
        Self {
            rank_select_bits: 1 << 20, // 1 Mbit
            elias_fano_elements: 10_000,
            trie_keys: 25_000,
            hash_map_keys: 25_000,
            memory_pool_ops: 2_000,
            entropy_block_bytes: 65_536,
            int_codec_elements: 10_000,
            radix_sort_elements: 100_000,
            suffix_array_bytes: 100_000,
            bm25_docs: 10_000,
        }
    }

    fn to_map(self) -> BTreeMap<String, u64> {
        let mut map = BTreeMap::new();
        map.insert("rank_select_bits".to_string(), self.rank_select_bits as u64);
        map.insert(
            "elias_fano_elements".to_string(),
            self.elias_fano_elements as u64,
        );
        map.insert("trie_keys".to_string(), self.trie_keys as u64);
        map.insert("hash_map_keys".to_string(), self.hash_map_keys as u64);
        map.insert("memory_pool_ops".to_string(), self.memory_pool_ops as u64);
        map.insert(
            "entropy_block_bytes".to_string(),
            self.entropy_block_bytes as u64,
        );
        map.insert(
            "int_codec_elements".to_string(),
            self.int_codec_elements as u64,
        );
        map.insert(
            "radix_sort_elements".to_string(),
            self.radix_sort_elements as u64,
        );
        map.insert(
            "suffix_array_bytes".to_string(),
            self.suffix_array_bytes as u64,
        );
        map.insert("bm25_docs".to_string(), self.bm25_docs as u64);
        map
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostMetadata {
    host: String,
    arch: String,
    cpu_model: String,
    cpu_mhz: String,
    cache_size: String,
    logical_cores: usize,
    pinned_core: usize,
    cpu_governor: String,
    git_commit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRecord {
    name: String,
    category: String,
    unit: String,
    dataset_size: u64,
    ops_per_iter: u64,
    trials: usize,
    min_ns_per_op: f64,
    median_ns_per_op: f64,
    p95_ns_per_op: f64,
    spread_pct: f64,
    max_regression_pct: f64,
    samples: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoreBenchReport {
    suite: String,
    tier: String,
    seed: u64,
    entropy_block_source: String,
    host_metadata: HostMetadata,
    dataset_sizes: BTreeMap<String, u64>,
    space_metrics: BTreeMap<String, f64>,
    benchmarks: Vec<BenchRecord>,
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn collect_host_metadata(pinned_core: usize) -> HostMetadata {
    let host_raw = read_trimmed("/etc/hostname")
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let host = host_raw
        .split('.')
        .next()
        .unwrap_or("unknown")
        .trim()
        .to_string();

    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let extract_field = |key: &str| -> String {
        for line in cpuinfo.lines() {
            if let Some((k, v)) = line.split_once(':')
                && k.trim() == key
            {
                return v.trim().to_string();
            }
        }
        "unknown".to_string()
    };

    let cpu_model = extract_field("model name");
    let cpu_mhz = extract_field("cpu MHz");
    let cache_size = extract_field("cache size");
    let logical_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cpu_governor = read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|| "unavailable (virtualized/no-cpufreq)".to_string());

    let git_commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    HostMetadata {
        host,
        arch: std::env::consts::ARCH.to_string(),
        cpu_model,
        cpu_mhz,
        cache_size,
        logical_cores,
        pinned_core,
        cpu_governor,
        git_commit,
    }
}

#[derive(Debug, Clone, Copy)]
struct BenchParams {
    dataset_size: u64,
    ops_per_iter: u64,
    warmup_calls: usize,
    trials: usize,
    bursts_per_trial: usize,
    max_regression_pct: f64,
}

impl BenchParams {
    const fn fast(dataset_size: u64, ops_per_iter: u64) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 4,
            trials: 5,
            bursts_per_trial: 4,
            max_regression_pct: 5.0,
        }
    }

    const fn medium(dataset_size: u64, ops_per_iter: u64) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 4,
            trials: 5,
            bursts_per_trial: 3,
            max_regression_pct: 5.0,
        }
    }

    const fn heavy(dataset_size: u64, ops_per_iter: u64, trials: usize) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 1,
            trials,
            bursts_per_trial: 1,
            max_regression_pct: 5.0,
        }
    }
}

fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn record_samples(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    mut samples: Vec<f64>,
) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total_trials = samples.len();
    let min_ns = samples[0];
    let median_ns = samples[total_trials / 2];
    let p95_idx = ((total_trials as f64 * 0.95).floor() as usize).min(total_trials - 1);
    let p95_ns = samples[p95_idx];
    let spread_pct = if min_ns > 0.0 {
        ((p95_ns - min_ns) / min_ns) * 100.0
    } else {
        0.0
    };
    let rounded_samples = samples.into_iter().map(round4).collect();

    records.push(BenchRecord {
        name: name.to_string(),
        category: category.to_string(),
        unit: unit.to_string(),
        dataset_size: params.dataset_size,
        ops_per_iter: params.ops_per_iter,
        trials: total_trials,
        min_ns_per_op: round4(min_ns),
        median_ns_per_op: round4(median_ns),
        p95_ns_per_op: round4(p95_ns),
        spread_pct: round2(spread_pct),
        max_regression_pct: params.max_regression_pct,
        samples: rounded_samples,
    });
}

fn measure_bench<F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    mut f: F,
) where
    F: FnMut(),
{
    measure_bench_with_setup(
        records,
        category,
        name,
        unit,
        params,
        &mut (),
        |()| {},
        |()| f(),
    );
}

#[allow(clippy::too_many_arguments)]
fn measure_bench_with_setup<T, S, F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    state: &mut T,
    mut setup: S,
    mut f: F,
) where
    S: FnMut(&mut T),
    F: FnMut(&mut T),
{
    assert!(
        params.ops_per_iter > 0,
        "ops_per_iter must be > 0 for {name}"
    );
    assert!(params.trials > 0, "trials must be > 0 for {name}");
    assert!(
        params.bursts_per_trial > 0,
        "bursts_per_trial must be > 0 for {name}"
    );

    for _ in 0..params.warmup_calls {
        setup(state);
        f(state);
    }

    let mut samples = Vec::with_capacity(params.trials);
    for _ in 0..params.trials {
        let mut best_burst_ns = f64::MAX;
        for _ in 0..params.bursts_per_trial {
            setup(state);
            let start = Instant::now();
            f(state);
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            if elapsed_ns < best_burst_ns {
                best_burst_ns = elapsed_ns;
            }
        }
        samples.push(best_burst_ns / (params.ops_per_iter as f64));
    }

    record_samples(records, category, name, unit, params, samples);
}

// ---------------------------------------------------------------------------
// 1. Rank / Select (1 Gbit = 1,073,741,824 bits by default)
// ---------------------------------------------------------------------------

fn bench_rank_select(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let num_bits = sizes.rank_select_bits;
    let num_words = num_bits / 64;

    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
    let words: Vec<u64> = (0..num_words).map(|_| rng.random::<u64>()).collect();
    let bv = BitVector::from_blocks(words, num_bits);

    let num_queries = 65_536usize;
    let rank_queries: Vec<usize> = (0..num_queries)
        .map(|_| rng.random_range(0..num_bits))
        .collect();
    let params = BenchParams::medium(num_bits as u64, num_queries as u64);

    // Build and benchmark one 1-Gbit Rank/Select structure at a time so peak RSS stays bounded.
    {
        let r9 = Rank9::new(bv.clone()).unwrap();
        space.insert(
            "rank_select/rank9_overhead_pct".to_string(),
            r9.space_overhead_percent(),
        );
        measure_bench(
            records,
            "rank_select",
            "rank_select/rank9/rank1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &rank_queries {
                    acc = acc.wrapping_add(r9.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );
    }

    let (select1_queries, select0_queries) = {
        let rsi = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let max_ones = rsi.count_ones().max(1);
        let max_zeros = rsi.count_zeros().max(1);
        let s1: Vec<usize> = (0..num_queries)
            .map(|_| rng.random_range(0..max_ones))
            .collect();
        let s0: Vec<usize> = (0..num_queries)
            .map(|_| rng.random_range(0..max_zeros))
            .collect();

        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/rank1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &rank_queries {
                    acc = acc.wrapping_add(rsi.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/select1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &s1 {
                    acc = acc.wrapping_add(rsi.select1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/select0",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &s0 {
                    acc = acc.wrapping_add(rsi.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        (s1, s0)
    };

    {
        let rs_se = RankSelectSE256::new(bv).unwrap();
        space.insert(
            "rank_select/se256_overhead_pct".to_string(),
            rs_se.space_overhead_percent(),
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/rank1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &rank_queries {
                    acc = acc.wrapping_add(rs_se.rank1(black_box(q)));
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/select1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &select1_queries {
                    acc = acc.wrapping_add(rs_se.select1(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "rank_select",
            "rank_select/se256/select0",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for &q in &select0_queries {
                    acc = acc.wrapping_add(rs_se.select0(black_box(q)).unwrap_or(0));
                }
                black_box(acc);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Elias-Fano Posting Lists (100,000 elements by default)
// ---------------------------------------------------------------------------

fn gen_uniform_sparse(n: usize, avg_gap: u32, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut cur: u32 = 0;
    for _ in 0..n {
        cur = cur.saturating_add(rng.random_range(1..=2 * avg_gap));
        out.push(cur);
    }
    out
}

fn gen_zipf_clustered(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut cur: u32 = 0;
    while out.len() < n {
        let run = rng.random_range(50..=200).min(n - out.len());
        for _ in 0..run {
            out.push(cur);
            cur = cur.saturating_add(1);
        }
        cur = cur.saturating_add(rng.random_range(5_000..=50_000));
    }
    out
}

fn bench_elias_fano(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.elias_fano_elements;
    let num_queries = 4_096usize;
    let params = BenchParams::fast(n as u64, num_queries as u64);

    let uniform = gen_uniform_sparse(n, 50, SEED_BASE ^ 0x02);
    let clustered = gen_zipf_clustered(n, SEED_BASE ^ 0x03);
    let dense: Vec<u32> = (0..n as u32).collect();

    for (dist_name, data) in [
        ("uniform", &uniform),
        ("zipf_clustered", &clustered),
        ("dense", &dense),
    ] {
        let ef = EliasFano::from_sorted(data);
        let pef = PartitionedEliasFano::from_sorted(data);
        let opef = OptimalPartitionedEliasFano::from_sorted(data);
        let cef = ClusteredEliasFano::from_sorted(data);
        let hybrid = HybridPostingList::from_sorted(data);

        space.insert(
            format!("elias_fano/{dist_name}/ef_bits_per_elem"),
            ef.bits_per_element(),
        );
        space.insert(
            format!("elias_fano/{dist_name}/pef_bits_per_elem"),
            pef.bits_per_element(),
        );
        space.insert(
            format!("elias_fano/{dist_name}/opef_bits_per_elem"),
            opef.bits_per_element(),
        );
        space.insert(
            format!("elias_fano/{dist_name}/cef_bits_per_elem"),
            cef.bits_per_element(),
        );
        space.insert(
            format!("elias_fano/{dist_name}/hybrid_bits_per_elem"),
            hybrid.bits_per_element(),
        );

        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x04);
        let universe = *data.last().unwrap() as u64 + 1;
        let targets: Vec<u64> = (0..num_queries)
            .map(|_| rng.random_range(0..universe))
            .collect();
        let mut sorted_targets = targets.clone();
        sorted_targets.sort_unstable();

        let indices: Vec<usize> = (0..num_queries).map(|_| rng.random_range(0..n)).collect();

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/ef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = ef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = pef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = opef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/cef_next_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for &t in &targets {
                    if let Some((_, v)) = cef.next_geq(black_box(t)) {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/pef_cursor_advance_to_geq"),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                let mut cursor = pef.cursor();
                for &t in &sorted_targets {
                    if cursor.advance_to_geq(black_box(t))
                        && let Some(v) = cursor.current()
                    {
                        acc = acc.wrapping_add(v);
                    }
                }
                black_box(acc);
            },
        );

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_cursor_advance_to_index"),
            "ns/op",
            params,
            || {
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
    }
}

// ---------------------------------------------------------------------------
// 3. Tries (1,000,000 keys across 3 distributions by default)
// ---------------------------------------------------------------------------

fn gen_trie_distributions(n: usize, seed: u64) -> Vec<(&'static str, Vec<Vec<u8>>)> {
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Hierarchical URL/path keys
    let mut paths: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            format!(
                "org/infinilabs/mod_{:02}/sub_{:03}/item_{:07}",
                i % 32,
                (i / 32) % 256,
                i
            )
            .into_bytes()
        })
        .collect();
    paths.sort();
    paths.dedup();

    // 2. Random 16-byte ASCII keys (append unique index suffix to guarantee exactly n unique keys)
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut random_16b: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let mut key = Vec::with_capacity(16);
            for _ in 0..10 {
                key.push(alphabet[rng.random_range(0..alphabet.len())]);
            }
            let suffix = format!("{:06x}", i);
            key.extend_from_slice(suffix.as_bytes());
            key
        })
        .collect();
    random_16b.sort();
    random_16b.dedup();

    // 3. Shared-prefix keys
    let mut shared_prefix: Vec<Vec<u8>> = (0..n)
        .map(|i| format!("common_prefix_header_v1_{:08x}", i).into_bytes())
        .collect();
    shared_prefix.sort();
    shared_prefix.dedup();

    vec![
        ("url_paths", paths),
        ("random_16b", random_16b),
        ("shared_prefix", shared_prefix),
    ]
}

fn bench_tries(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    let n = sizes.trie_keys;
    let dists = gen_trie_distributions(n, SEED_BASE ^ 0x05);

    for (dist_name, keys) in &dists {
        let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let num_keys = key_slices.len() as u64;

        // 1. DoubleArrayTrie build_from_sorted on all 1M keys + lookup on all 1M keys
        let mut built_dat: Option<DoubleArrayTrie> = None;
        measure_bench_with_setup(
            records,
            "tries",
            &format!("tries/dat/{dist_name}/build_sorted"),
            "ns/key",
            BenchParams::heavy(num_keys, num_keys, 3),
            &mut built_dat,
            |slot| {
                *slot = None;
            },
            |slot| {
                let dat = DoubleArrayTrie::build_from_sorted(black_box(&key_slices)).unwrap();
                black_box(dat.len());
                *slot = Some(dat);
            },
        );

        let mut dat = built_dat.unwrap();
        dat.shrink_to_fit();
        let bytes_per_state = dat.mem_size() as f64 / dat.total_states().max(1) as f64;
        space.insert(
            format!("tries/dat_{dist_name}_bytes_per_state"),
            bytes_per_state,
        );

        let lookup_step = (key_slices.len() / 200_000).max(1);
        let lookup_sample: Vec<&[u8]> = key_slices.iter().copied().step_by(lookup_step).collect();
        let lookup_params = BenchParams::medium(num_keys, lookup_sample.len() as u64);

        measure_bench(
            records,
            "tries",
            &format!("tries/dat/{dist_name}/lookup"),
            "ns/op",
            lookup_params,
            || {
                let mut found = 0usize;
                for &k in &lookup_sample {
                    if dat.contains(black_box(k)) {
                        found += 1;
                    }
                }
                black_box(found);
            },
        );
        drop(dat);

        // 2. CsppTrie build on all 1M keys + lookup on the 1M-key trie
        let mut built_cspp: Option<CsppTrie> = None;
        measure_bench_with_setup(
            records,
            "tries",
            &format!("tries/cspp/{dist_name}/build"),
            "ns/key",
            BenchParams::heavy(num_keys, num_keys, 3),
            &mut built_cspp,
            |slot| {
                *slot = None;
            },
            |slot| {
                let mut cspp = CsppTrie::new(0);
                for &k in &key_slices {
                    cspp.insert(black_box(k));
                }
                black_box(cspp.num_words());
                *slot = Some(cspp);
            },
        );

        let cspp = built_cspp.unwrap();
        let stat = cspp.mem_get_stat();
        let active_bytes = stat.used_size.saturating_sub(stat.frag_size);
        space.insert(
            format!("tries/cspp_{dist_name}_bytes_per_key"),
            active_bytes as f64 / cspp.num_words().max(1) as f64,
        );

        measure_bench(
            records,
            "tries",
            &format!("tries/cspp/{dist_name}/lookup"),
            "ns/op",
            lookup_params,
            || {
                let mut found = 0usize;
                for &k in &lookup_sample {
                    if cspp.contains(black_box(k)) {
                        found += 1;
                    }
                }
                black_box(found);
            },
        );
    }

    // 3. ZiporaTrie (DoubleArray strategy) build + lookup on 1M keys
    let zipora_keys = &dists[0].1;
    let num_zipora_keys = zipora_keys.len() as u64;
    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: (zipora_keys.len() * 4).next_power_of_two(),
            growth_factor: 2.0,
            free_list_management: true,
            auto_shrink: false,
        },
        ..ZiporaTrieConfig::default()
    };

    let mut built_ztrie: Option<ZiporaTrie<RankSelectInterleaved256>> = None;
    measure_bench_with_setup(
        records,
        "tries",
        "tries/zipora_trie_da/build",
        "ns/key",
        BenchParams::heavy(num_zipora_keys, num_zipora_keys, 3),
        &mut built_ztrie,
        |slot| {
            *slot = None;
        },
        |slot| {
            let mut ztrie: ZiporaTrie<RankSelectInterleaved256> =
                ZiporaTrie::with_config(config.clone());
            for k in zipora_keys {
                ztrie.insert(black_box(k)).unwrap();
            }
            black_box(ztrie.len());
            *slot = Some(ztrie);
        },
    );

    let ztrie = built_ztrie.unwrap();
    let zlookup_step = (zipora_keys.len() / 200_000).max(1);
    let zlookup_sample: Vec<&[u8]> = zipora_keys
        .iter()
        .map(|k| k.as_slice())
        .step_by(zlookup_step)
        .collect();
    measure_bench(
        records,
        "tries",
        "tries/zipora_trie_da/lookup",
        "ns/op",
        BenchParams::medium(num_zipora_keys, zlookup_sample.len() as u64),
        || {
            let mut found = 0usize;
            for &k in &zlookup_sample {
                if ztrie.contains(black_box(k)) {
                    found += 1;
                }
            }
            black_box(found);
        },
    );
}

// ---------------------------------------------------------------------------
// 4. Hash Maps (1,000,000 u64 keys & 1,000,000 16-byte string keys by default)
// ---------------------------------------------------------------------------

fn bench_hash_maps(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    let n = sizes.hash_map_keys;
    let cap = (n * 2).next_power_of_two();
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x06);

    let u64_keys: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
    let str16_keys: Vec<[u8; 16]> = (0..n)
        .map(|i| {
            let mut buf = [0u8; 16];
            let s = format!("key_{:012x}", i);
            buf.copy_from_slice(s.as_bytes());
            buf
        })
        .collect();

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
    // GoldHashMap rounds initial_capacity up to the next prime (for n = 1,000,000,
    // (n * 3) / 2 = 1,500,000 -> prime 1,982,627 buckets, max_load = 1,387,838 > 1,000,000).
    let gconfig = GoldHashMapConfig {
        initial_capacity: (n * 3) / 2,
        ..GoldHashMapConfig::default()
    };

    let insert_params = BenchParams::heavy(n as u64, n as u64, 5);
    let get_params = BenchParams::medium(n as u64, n as u64);

    // 1. ZiporaHashMap<u64, u64> insert + get (1M keys)
    let mut built_zmap_u64: Option<ZiporaHashMap<u64, u64>> = None;
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/insert_u64",
        "ns/op",
        insert_params,
        &mut built_zmap_u64,
        |slot| {
            *slot = None;
            *slot =
                Some(ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap());
        },
        |slot| {
            let m = slot.as_mut().unwrap();
            for (i, &k) in u64_keys.iter().enumerate() {
                m.insert(black_box(k), i as u64).unwrap();
            }
            black_box(m.len());
        },
    );

    let zmap_u64 = built_zmap_u64.unwrap();
    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_u64",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for &k in &u64_keys {
                if let Some(&v) = zmap_u64.get(black_box(&k)) {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum);
        },
    );
    drop(zmap_u64);

    // 2. ZiporaHashMap<[u8; 16], u64> insert + get (1M keys)
    let mut built_zmap_str16: Option<ZiporaHashMap<[u8; 16], u64>> = None;
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/insert_str16",
        "ns/op",
        insert_params,
        &mut built_zmap_str16,
        |slot| {
            *slot = None;
            *slot =
                Some(ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap());
        },
        |slot| {
            let m = slot.as_mut().unwrap();
            for (i, k) in str16_keys.iter().enumerate() {
                m.insert(black_box(*k), i as u64).unwrap();
            }
            black_box(m.len());
        },
    );

    let zmap_str16 = built_zmap_str16.unwrap();
    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_str16",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for k in &str16_keys {
                if let Some(&v) = zmap_str16.get(black_box(k)) {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum);
        },
    );
    drop(zmap_str16);

    // 3. GoldHashMap<u64, u64> insert + get (1M keys)
    let mut built_gmap_u64: Option<GoldHashMap<u64, u64>> = None;
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/insert_u64",
        "ns/op",
        insert_params,
        &mut built_gmap_u64,
        |slot| {
            *slot = None;
            *slot = Some(GoldHashMap::with_config(gconfig.clone()));
        },
        |slot| {
            let m = slot.as_mut().unwrap();
            for (i, &k) in u64_keys.iter().enumerate() {
                let _ = m.insert(black_box(k), i as u64);
            }
            black_box(m.len());
        },
    );

    let gmap_u64 = built_gmap_u64.unwrap();
    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_u64",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for &k in &u64_keys {
                if let Some(&v) = gmap_u64.get(black_box(&k)) {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum);
        },
    );
    drop(gmap_u64);

    // 4. GoldHashMap<[u8; 16], u64> insert + get (1M keys)
    let mut built_gmap_str16: Option<GoldHashMap<[u8; 16], u64>> = None;
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/insert_str16",
        "ns/op",
        insert_params,
        &mut built_gmap_str16,
        |slot| {
            *slot = None;
            *slot = Some(GoldHashMap::with_config(gconfig.clone()));
        },
        |slot| {
            let m = slot.as_mut().unwrap();
            for (i, k) in str16_keys.iter().enumerate() {
                let _ = m.insert(black_box(*k), i as u64);
            }
            black_box(m.len());
        },
    );

    let gmap_str16 = built_gmap_str16.unwrap();
    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_str16",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for k in &str16_keys {
                if let Some(&v) = gmap_str16.get(black_box(k)) {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum);
        },
    );
}

// ---------------------------------------------------------------------------
// 5. Memory Pools (LockFreeMemoryPool & SecureMemoryPool, 1T & 4T)
// ---------------------------------------------------------------------------

fn bench_memory_pools(records: &mut Vec<BenchRecord>, logical_cores: usize, sizes: SuiteSizes) {
    let ops = sizes.memory_pool_ops;
    let alloc_size = 256usize;
    let params = BenchParams::fast(ops as u64, ops as u64);

    // LockFreeMemoryPool 1T
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

    // LockFreeMemoryPool 4T
    const POOL_SIZES: [usize; 4] = [128, 256, 512, 1024];
    let lf_pool_4t =
        Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
    for &sz in &POOL_SIZES {
        let block = lf_pool_4t.allocate(sz).unwrap();
        let _spacer = lf_pool_4t.allocate(384).unwrap();
        lf_pool_4t.deallocate(block, sz).unwrap();
    }
    {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let active_flag = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(3);
        let ccx_stride = (logical_cores / 8).max(2);
        for worker_idx in 0..3usize {
            let pool = Arc::clone(&lf_pool_4t);
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = ((worker_idx + 1) * ccx_stride) % logical_cores.max(1);
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

    // SecureMemoryPool 1T (pre-populate 32 chunks in local cache and rotate in batches of 16
    // so active_allocations HashMap averages across 16 distinct chunk addresses instead of a
    // single ASLR address / RandomState bucket)
    let mut sec_cfg = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg.enable_hot_cold_separation = false;
    sec_cfg.enable_huge_pages = false;
    let sec_pool_1t = Arc::new(SecureMemoryPool::new(sec_cfg.clone()).unwrap());
    {
        let pre: Vec<_> = (0..32).map(|_| sec_pool_1t.allocate().unwrap()).collect();
        drop(pre);
    }
    let sec_batch = 16usize;
    let sec_rounds = (ops / sec_batch).max(1);
    let sec_total_ops = (sec_rounds * sec_batch) as u64;
    let sec_params = BenchParams::fast(sec_total_ops, sec_total_ops);

    measure_bench(
        records,
        "memory_pools",
        "memory_pools/secure_pool/alloc_free_1t",
        "ns/op",
        sec_params,
        || {
            let mut batch_vec = Vec::with_capacity(sec_batch);
            for _ in 0..sec_rounds {
                for _ in 0..sec_batch {
                    batch_vec.push(sec_pool_1t.allocate().unwrap());
                }
                black_box(batch_vec.len());
                batch_vec.clear();
            }
        },
    );

    // SecureMemoryPool 4T
    let sec_pool_main = Arc::new(SecureMemoryPool::new(sec_cfg.clone()).unwrap());
    {
        let pre: Vec<_> = (0..32).map(|_| sec_pool_main.allocate().unwrap()).collect();
        drop(pre);
    }
    {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let active_flag = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(3);
        let ccx_stride = (logical_cores / 8).max(2);
        for worker_idx in 0..3usize {
            let pool = Arc::new(SecureMemoryPool::new(sec_cfg.clone()).unwrap());
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = ((worker_idx + 1) * ccx_stride) % logical_cores.max(1);
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
                for _ in 0..sec_rounds {
                    for _ in 0..sec_batch {
                        batch_vec.push(sec_pool_main.allocate().unwrap());
                    }
                    black_box(batch_vec.len());
                    batch_vec.clear();
                }
                active_flag.store(false, Ordering::Relaxed);
            },
        );

        stop_flag.store(true, Ordering::Release);
        for h in handles {
            let _ = h.join();
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Entropy Coding (Huffman O0/O1/O2, FSE, rANS on 64 KiB synthetic enwik8-like block)
// ---------------------------------------------------------------------------

/// Generates a deterministic 64 KiB synthetic `enwik8`-like block composed of
/// Zipf-skewed English words and MediaWiki markup tokens.
fn gen_enwik_like_block(size: usize, seed: u64) -> Vec<u8> {
    const TOKENS: &[&[u8]] = &[
        b"the ",
        b"of ",
        b"and ",
        b"in ",
        b"to ",
        b"a ",
        b"is ",
        b"for ",
        b"that ",
        b"with ",
        b"on ",
        b"as ",
        b"are ",
        b"by ",
        b"this ",
        b"be ",
        b"from ",
        b"or ",
        b"an ",
        b"at ",
        b"[[Category:",
        b"Wikipedia",
        b"http://",
        b"archive",
        b"reference",
        b"section",
        b"compression",
        b"succinct",
        b"automaton",
        b"structure",
        b"algorithm",
        b"performance ",
    ];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(size + 64);
    while out.len() < size {
        let r = rng.random_range(0..100usize);
        let idx = if r < 55 {
            rng.random_range(0..8)
        } else if r < 85 {
            rng.random_range(8..20)
        } else {
            rng.random_range(20..TOKENS.len())
        };
        out.extend_from_slice(TOKENS[idx]);
    }
    out.truncate(size);
    out
}

fn bench_entropy(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    let block_size = sizes.entropy_block_bytes;
    let block = gen_enwik_like_block(block_size, SEED_BASE ^ 0x07);
    let params = BenchParams::fast(block_size as u64, block_size as u64);

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
    let huff_o1_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap();
    let huff_o1_compressed = huff_o1_enc.encode(&block).unwrap();
    let huff_o1_dec = ContextualHuffmanDecoder::new(
        ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap(),
    );

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

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o1_dec
                .decode(black_box(&huff_o1_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // Huffman O2 (full 64 KiB block for encode and decode)
    let huff_o2_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order2).unwrap();
    let huff_o2_compressed = huff_o2_enc.encode(&block).unwrap();
    let huff_o2_dec = ContextualHuffmanDecoder::new(
        ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order2).unwrap(),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/encode_64kib",
        "ns/byte",
        params,
        || {
            let encoded = huff_o2_enc.encode(black_box(&block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/decode_64kib",
        "ns/byte",
        params,
        || {
            let decoded = huff_o2_dec
                .decode(black_box(&huff_o2_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

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
}

// ---------------------------------------------------------------------------
// 7. Integer Codecs (100,000 contiguous integers by default)
// ---------------------------------------------------------------------------

fn bench_int_codecs(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    let n = sizes.int_codec_elements;
    let sorted_u32 = gen_uniform_sparse(n, 120, SEED_BASE ^ 0x08);
    let params = BenchParams::fast(n as u64, n as u64);

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

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_into",
        "ns/int",
        params,
        || {
            let decoded = StreamVByte::decode_into(black_box(&svb_stream), n, &mut svb_out);
            black_box(decoded);
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_deltas",
        "ns/int",
        params,
        || {
            let decoded = StreamVByte::decode_deltas(black_box(&svb_stream), n);
            black_box(decoded.len());
        },
    );

    let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/encode_deltas",
        "ns/int",
        params,
        || {
            let encoded = GroupVarint::encode_deltas(black_box(&sorted_u32));
            black_box(encoded.len());
        },
    );

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
}

// ---------------------------------------------------------------------------
// 8. Radix Sort (10,000,000 = 10^7 elements for u32, u64, (u64, u64) by default)
// ---------------------------------------------------------------------------

fn bench_radix_sort(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    let n = sizes.radix_sort_elements;
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x09);
    let mut sorter = RadixSort::with_config(RadixSortConfig {
        use_parallel: false,
        use_counting_sort_threshold: 0,
        ..RadixSortConfig::default()
    });

    // 1. LSD u32 (10^7 elements)
    {
        let orig_u32: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
        let mut work_u32 = vec![0u32; n];
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/lsd_u32",
            "ns/elem",
            BenchParams::heavy(n as u64, n as u64, 5),
            &mut work_u32,
            |buf| {
                buf.copy_from_slice(&orig_u32);
            },
            |buf| {
                sorter.sort_u32(black_box(buf.as_mut_slice())).unwrap();
                black_box(buf[0]);
            },
        );
    }

    // 2. LSD u64 (10^7 elements)
    {
        let orig_u64: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
        let mut work_u64 = vec![0u64; n];
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/lsd_u64",
            "ns/elem",
            BenchParams::heavy(n as u64, n as u64, 5),
            &mut work_u64,
            |buf| {
                buf.copy_from_slice(&orig_u64);
            },
            |buf| {
                sorter.sort_u64(black_box(buf.as_mut_slice())).unwrap();
                black_box(buf[0]);
            },
        );
    }

    // 3. KeyValueRadixSort<u64, u64> (10^7 (u64, u64) elements)
    {
        let orig_kv: Vec<(u64, u64)> = (0..n).map(|i| (rng.random::<u64>(), i as u64)).collect();
        let mut work_kv = vec![(0u64, 0u64); n];
        let kv_sorter = KeyValueRadixSort::<u64, u64>::new();
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/kv_u64_u64",
            "ns/elem",
            BenchParams::heavy(n as u64, n as u64, 3),
            &mut work_kv,
            |buf| {
                buf.copy_from_slice(&orig_kv);
            },
            |buf| {
                kv_sorter
                    .sort_by_key(black_box(buf.as_mut_slice()))
                    .unwrap();
                black_box(buf[0].0);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Suffix Array (10,000,000 = 10^7 bytes by default)
// ---------------------------------------------------------------------------

fn bench_suffix_array(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    let n = sizes.suffix_array_bytes;
    let text = gen_enwik_like_block(n, SEED_BASE ^ 0x0A);

    let mut built_sa: Option<SuffixArray> = None;
    measure_bench_with_setup(
        records,
        "suffix_array",
        "suffix_array/construct",
        "ns/byte",
        BenchParams::heavy(n as u64, n as u64, 3),
        &mut built_sa,
        |slot| {
            *slot = None;
        },
        |slot| {
            let sa = SuffixArray::new(black_box(&text)).unwrap();
            black_box(sa.as_slice().len());
            *slot = Some(sa);
        },
    );

    let sa = built_sa.unwrap();
    let patterns: &[&[u8]] = &[
        b"Wikipedia",
        b"compression",
        b"succinct",
        b"automaton",
        b"performance",
        b"nonexistent_pattern_xyz",
    ];
    let query_loops = 500usize;
    let total_searches = (query_loops * patterns.len()) as u64;
    measure_bench(
        records,
        "suffix_array",
        "suffix_array/search_query",
        "ns/op",
        BenchParams::fast(n as u64, total_searches),
        || {
            let mut hits = 0usize;
            for _ in 0..query_loops {
                for &pat in patterns {
                    let (_, count) = sa.search(black_box(&text), black_box(pat));
                    hits = hits.wrapping_add(count);
                }
            }
            black_box(hits);
        },
    );
}

// ---------------------------------------------------------------------------
// 10. String Search & BM25 Scoring (including S36-R2 build_score_table)
// ---------------------------------------------------------------------------

fn bench_search_and_scoring(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    #[repr(C, align(64))]
    struct AlignedHaystack([u8; 65_536]);

    let haystack_len = 65_536usize;
    let mut aligned = Box::new(AlignedHaystack([b'a'; 65_536]));
    aligned.0[haystack_len - 32..haystack_len - 19].copy_from_slice(b"needle_target");
    aligned.0[haystack_len - 5] = b'Z';
    let haystack = &aligned.0[..];

    let searcher = SimdStringSearch::new();
    let search_iters = 200u64;
    let str_params = BenchParams::fast(haystack_len as u64, search_iters);

    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/simd_strchr_64kib",
        "ns/call",
        str_params,
        || {
            let mut acc = 0usize;
            for _ in 0..search_iters {
                if let Some(pos) = searcher.sse42_strchr(black_box(haystack), black_box(b'Z')) {
                    acc = acc.wrapping_add(pos);
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/simd_strstr_64kib",
        "ns/call",
        str_params,
        || {
            let mut acc = 0usize;
            for _ in 0..search_iters {
                if let Some(pos) =
                    searcher.sse42_strstr(black_box(haystack), black_box(b"needle_target"))
                {
                    acc = acc.wrapping_add(pos);
                }
            }
            black_box(acc);
        },
    );

    // BM25 hot paths (100,000 docs by default) + S36-R2 build_score_table
    let k1 = 1.2f32;
    let b = 0.75f32;
    let avg_dl = 120.0f32;
    let idf = 3.25f32;
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

    let bm25_rounds = 10u64;
    let bm25_params = BenchParams::fast(num_docs as u64, bm25_rounds * num_docs as u64);
    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/bm25_score_single",
        "ns/doc",
        bm25_params,
        || {
            let mut sum = 0.0f32;
            for _ in 0..bm25_rounds {
                for i in 0..num_docs {
                    sum += scorer.score(black_box(norms[doc_ids[i] as usize]), black_box(tfs[i]));
                }
            }
            black_box(sum);
        },
    );

    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/bm25_batch_score_with_prefetch",
        "ns/doc",
        bm25_params,
        || {
            for _ in 0..bm25_rounds {
                scorer.batch_score_with_prefetch(
                    black_box(&norms),
                    black_box(&doc_ids),
                    black_box(&tfs),
                    black_box(&mut scores_out),
                );
                black_box(scores_out[0]);
            }
        },
    );

    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/bm25_batch_score_simd",
        "ns/doc",
        bm25_params,
        || {
            for _ in 0..bm25_rounds {
                scorer.batch_score(
                    black_box(&contiguous_norms),
                    black_box(&tfs),
                    black_box(&mut scores_out),
                );
                black_box(scores_out[0]);
            }
        },
    );

    // S36-R2: FieldnormEncoder::build_score_table (256 * 256 = 65,536 cells per call)
    let table_rounds = 500u64;
    let max_tf = 255u16;
    let cells_per_call = (max_tf as u64 + 1) * 256u64;
    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/fieldnorm_build_score_table",
        "ns/cell",
        BenchParams::fast(cells_per_call, table_rounds * cells_per_call),
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

fn write_json(path: &Path, report: &CoreBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote benchmark report to {}", path.display());
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
        .find(|w| w[0] == "--output")
        .map(|w| PathBuf::from(&w[1]))
        .or_else(|| std::env::var("ZIPORA_BENCH_OUT").ok().map(PathBuf::from));

    // Pin Rayon's global thread pool to 1 thread so single-core benchmarks (including
    // KeyValueRadixSort::sort_by_key) run deterministically on the pinned core without
    // spawning 128 threads that contend on a single core's affinity mask.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    let logical_cores = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let primary_core = select_primary_core(logical_cores);
    let host_metadata = collect_host_metadata(primary_core);
    pin_current_thread_to_core(primary_core);
    let sizes = if smoke_mode {
        SuiteSizes::smoke()
    } else {
        SuiteSizes::spec()
    };
    let tier = if smoke_mode { "smoke" } else { "spec" };

    {
        let spin_start = Instant::now();
        let mut spin_acc = 0x1234_5678_9ABC_DEF0u64;
        while spin_start.elapsed().as_millis() < 200 {
            for _ in 0..10_000 {
                spin_acc = spin_acc.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
            black_box(spin_acc);
        }
    }

    eprintln!(
        "=== Zipora E1 Core Benchmark Suite (tier={}, host={}, cpu={}, cores={}, pinned_core={}, commit={}) ===",
        tier,
        host_metadata.host,
        host_metadata.cpu_model,
        host_metadata.logical_cores,
        primary_core,
        host_metadata.git_commit
    );

    let suite_start = Instant::now();
    let mut records = Vec::new();
    let mut space_metrics = BTreeMap::new();

    bench_rank_select(&mut records, &mut space_metrics, sizes);
    bench_elias_fano(&mut records, &mut space_metrics, sizes);
    bench_tries(&mut records, &mut space_metrics, sizes);
    bench_hash_maps(&mut records, sizes);
    bench_memory_pools(&mut records, host_metadata.logical_cores, sizes);
    bench_entropy(&mut records, sizes);
    bench_int_codecs(&mut records, sizes);
    bench_radix_sort(&mut records, sizes);
    bench_suffix_array(&mut records, sizes);
    bench_search_and_scoring(&mut records, sizes);

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

    let elapsed = suite_start.elapsed();
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
    } else {
        PathBuf::from("target/bench/core_bench.json")
    };
    let latest_out = custom_out.unwrap_or(default_out);
    write_json(&latest_out, &report);

    if save_baseline && !smoke_mode {
        let baseline_path = PathBuf::from(format!(
            "benches/baselines/{}/core_bench.json",
            host_metadata.host
        ));
        write_json(&baseline_path, &report);
    }
}
