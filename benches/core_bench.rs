//! Canonical Phase E Core Benchmark Suite (`plan.md` §1.6 `E1` + `S36-R2`).
//!
//! Runs a deterministic, fixed-seed benchmark suite covering all 10 core subsystems:
//! 1. Rank/Select (`Rank9`, `RankSelectInterleaved256`, `RankSelectSE256` `rank1`/`select1`/`select0`)
//! 2. Elias-Fano (`EliasFano`, `PartitionedEliasFano`, `OptimalPartitionedEliasFano`,
//!    `ClusteredEliasFano` `next_geq`, `advance_to_geq`, `advance_to_index`, and `bits_per_element`)
//! 3. Tries (`DoubleArrayTrie` lookup/insert across 3 distributions, `CsppTrie` lookup/insert,
//!    `ZiporaTrie` DoubleArray build/lookup)
//! 4. Hash Maps (`ZiporaHashMap` and `GoldHashMap` get/insert on `u64` and 16-byte string keys)
//! 5. Memory Pools (`LockFreeMemoryPool` and `SecureMemoryPool` alloc+free on 1 and 4 threads)
//! 6. Entropy Coding (`HuffmanEncoder` O0, `ContextualHuffmanEncoder` O1/O2, `FseEncoder`,
//!    `Rans64Encoder` X1/X4 encode+decode on deterministic 64 KiB blocks)
//! 7. Integer Codecs (`StreamVByte` and `GroupVarint` encode/decode deltas)
//! 8. Radix Sort (`RadixSort` `u32`, `u64`, and `KeyValueRadixSort<u64, u64>` `(u64, u64)`)
//! 9. Suffix Array (`SuffixArray` construction + substring search)
//! 10. String Search & BM25 Scoring (`SimdStringSearch` `strchr`/`strstr`, `Bm25BatchScorer`
//!     `score`/`batch_score`/`batch_score_with_prefetch`, and `FieldnormEncoder::build_score_table`
//!     for `S36-R2`).
//!
//! Emits structured JSON to `target/bench/core_bench.json` (and optionally
//! `benches/baselines/<host>/core_bench.json` when `--save-baseline` or
//! `ZIPORA_BENCH_SAVE_BASELINE=1` is set, or when the host baseline does not yet exist).

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
use zipora::memory::{
    LockFreeMemoryPool, LockFreePoolConfig, SecureMemoryPool, SecurePoolConfig,
};
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
const DEFAULT_TRIALS: usize = 3;
const SUITE_PASSES: usize = 3;

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

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn configure_allocator_for_benchmarks() {
    const M_TRIM_THRESHOLD: libc::c_int = -1;
    const M_MMAP_THRESHOLD: libc::c_int = -3;
    // SAFETY: mallopt adjusts process-wide glibc malloc thresholds so temporary
    // allocations up to 64 MiB stay in user-space heap instead of hitting kernel
    // mmap/munmap page faults on every benchmark iteration.
    unsafe {
        libc::mallopt(M_TRIM_THRESHOLD, -1);
        libc::mallopt(M_MMAP_THRESHOLD, 64 * 1024 * 1024);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn configure_allocator_for_benchmarks() {}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostMetadata {
    host: String,
    arch: String,
    cpu_model: String,
    cpu_mhz: String,
    cache_size: String,
    logical_cores: usize,
    cpu_governor: String,
    git_commit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRecord {
    name: String,
    category: String,
    unit: String,
    ops_per_iter: u64,
    trials: usize,
    min_ns_per_op: f64,
    median_ns_per_op: f64,
    p95_ns_per_op: f64,
    #[serde(skip)]
    samples: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoreBenchReport {
    suite: String,
    seed: u64,
    host_metadata: HostMetadata,
    space_metrics: BTreeMap<String, f64>,
    benchmarks: Vec<BenchRecord>,
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn collect_host_metadata() -> HostMetadata {
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
        cpu_governor,
        git_commit,
    }
}

fn measure_bench<F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    ops_per_iter: u64,
    trials: usize,
    mut f: F,
) where
    F: FnMut(),
{
    assert!(ops_per_iter > 0, "ops_per_iter must be > 0 for {name}");
    assert!(trials > 0, "trials must be > 0 for {name}");

    // Warmup for at least 5 ms and at least 6 calls per pass so CPU boost clock, branch
    // predictors, L1/L2 caches, and allocator bins reach steady state without perturbation.
    let warmup_start = Instant::now();
    let mut warmup_calls = 0usize;
    while warmup_calls < 6 || warmup_start.elapsed().as_nanos() < 5_000_000 {
        f();
        warmup_calls += 1;
        if warmup_calls >= 32 {
            break;
        }
    }

    // Choose batch_calls deterministically from ops_per_iter (never from runtime timings).
    let batch_calls: usize = if ops_per_iter <= 16_384 { 4 } else { 1 };
    let total_ops_per_batch = (ops_per_iter as f64) * (batch_calls as f64);

    let mut pass_samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        // Take the best of 8 uninterrupted bursts per trial.
        let mut best_burst_ns = f64::MAX;
        for _ in 0..8 {
            let start = Instant::now();
            for _ in 0..batch_calls {
                f();
            }
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            if elapsed_ns < best_burst_ns {
                best_burst_ns = elapsed_ns;
            }
        }
        pass_samples.push(best_burst_ns / total_ops_per_batch);
    }

    if let Some(existing) = records.iter_mut().find(|r| r.name == name) {
        existing.samples.extend_from_slice(&pass_samples);
        existing
            .samples
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let total_trials = existing.samples.len();
        let min_ns = existing.samples[0];
        let median_ns = existing.samples[total_trials / 2];
        let p95_idx = ((total_trials as f64 * 0.95).floor() as usize).min(total_trials - 1);
        let p95_ns = existing.samples[p95_idx];
        existing.trials = total_trials;
        existing.min_ns_per_op = (min_ns * 10_000.0).round() / 10_000.0;
        existing.median_ns_per_op = (median_ns * 10_000.0).round() / 10_000.0;
        existing.p95_ns_per_op = (p95_ns * 10_000.0).round() / 10_000.0;
    } else {
        pass_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_ns = pass_samples[0];
        let median_ns = pass_samples[trials / 2];
        let p95_idx = ((trials as f64 * 0.95).floor() as usize).min(trials - 1);
        let p95_ns = pass_samples[p95_idx];
        records.push(BenchRecord {
            name: name.to_string(),
            category: category.to_string(),
            unit: unit.to_string(),
            ops_per_iter,
            trials,
            min_ns_per_op: (min_ns * 10_000.0).round() / 10_000.0,
            median_ns_per_op: (median_ns * 10_000.0).round() / 10_000.0,
            p95_ns_per_op: (p95_ns * 10_000.0).round() / 10_000.0,
            samples: pass_samples,
        });
    }
}

// ---------------------------------------------------------------------------
// 1. Rank / Select
// ---------------------------------------------------------------------------

fn bench_rank_select(records: &mut Vec<BenchRecord>, space: &mut BTreeMap<String, f64>) {
    // Default 128 Kbit (16 KiB bitvector, ~24 KiB with rank/select directory) + 8 KiB query
    // slice so the entire working set fits inside the 32 KiB L1D cache and 64-entry L1 DTLB;
    // setting ZIPORA_BENCH_RANK_BITS=1073741824 scales to 1 Gbit.
    let num_bits: usize = std::env::var("ZIPORA_BENCH_RANK_BITS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128 * 1024);
    let num_words = num_bits / 64;

    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
    let words: Vec<u64> = (0..num_words).map(|_| rng.random::<u64>()).collect();
    let bv = BitVector::from_blocks(words, num_bits);

    let r9 = Rank9::new(bv.clone()).unwrap();
    let rsi = RankSelectInterleaved256::new(bv.clone()).unwrap();
    let rs_se = RankSelectSE256::new(bv).unwrap();

    space.insert(
        "rank_select/rank9_overhead_pct".to_string(),
        r9.space_overhead_percent(),
    );
    space.insert(
        "rank_select/se256_overhead_pct".to_string(),
        rs_se.space_overhead_percent(),
    );

    let num_queries = 1_024usize;
    let query_passes = 4usize;
    let total_queries = (num_queries * query_passes) as u64;
    let rank_queries: Vec<usize> = (0..num_queries)
        .map(|_| rng.random_range(0..num_bits))
        .collect();
    let max_ones = rsi.count_ones().max(1);
    let max_zeros = rsi.count_zeros().max(1);
    let select1_queries: Vec<usize> = (0..num_queries)
        .map(|_| rng.random_range(0..max_ones))
        .collect();
    let select0_queries: Vec<usize> = (0..num_queries)
        .map(|_| rng.random_range(0..max_zeros))
        .collect();

    measure_bench(
        records,
        "rank_select",
        "rank_select/rank9/rank1",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &rank_queries {
                    acc = acc.wrapping_add(r9.rank1(black_box(q)));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/interleaved256/rank1",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &rank_queries {
                    acc = acc.wrapping_add(rsi.rank1(black_box(q)));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/interleaved256/select1",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &select1_queries {
                    acc = acc.wrapping_add(rsi.select1(black_box(q)).unwrap_or(0));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/interleaved256/select0",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &select0_queries {
                    acc = acc.wrapping_add(rsi.select0(black_box(q)).unwrap_or(0));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/se256/rank1",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &rank_queries {
                    acc = acc.wrapping_add(rs_se.rank1(black_box(q)));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/se256/select1",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &select1_queries {
                    acc = acc.wrapping_add(rs_se.select1(black_box(q)).unwrap_or(0));
                }
            }
            black_box(acc);
        },
    );

    measure_bench(
        records,
        "rank_select",
        "rank_select/se256/select0",
        "ns/op",
        total_queries,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0usize;
            for _ in 0..query_passes {
                for &q in &select0_queries {
                    acc = acc.wrapping_add(rs_se.select0(black_box(q)).unwrap_or(0));
                }
            }
            black_box(acc);
        },
    );
}

// ---------------------------------------------------------------------------
// 2. Elias-Fano Posting Lists
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

fn bench_elias_fano(records: &mut Vec<BenchRecord>, space: &mut BTreeMap<String, f64>) {
    let n: usize = std::env::var("ZIPORA_BENCH_EF_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8_192);
    let num_queries = 1_024usize;
    let query_passes = 4usize;
    let total_queries = (num_queries * query_passes) as u64;

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
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    for &t in &targets {
                        if let Some((_, v)) = ef.next_geq(black_box(t)) {
                            acc = acc.wrapping_add(v);
                        }
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
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    for &t in &targets {
                        if let Some((_, v)) = pef.next_geq(black_box(t)) {
                            acc = acc.wrapping_add(v);
                        }
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
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    for &t in &targets {
                        if let Some((_, v)) = opef.next_geq(black_box(t)) {
                            acc = acc.wrapping_add(v);
                        }
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
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    for &t in &targets {
                        if let Some((_, v)) = cef.next_geq(black_box(t)) {
                            acc = acc.wrapping_add(v);
                        }
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
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    let mut cursor = pef.cursor();
                    for &t in &sorted_targets {
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

        measure_bench(
            records,
            "elias_fano",
            &format!("elias_fano/{dist_name}/opef_cursor_advance_to_index"),
            "ns/op",
            total_queries,
            DEFAULT_TRIALS,
            || {
                let mut acc = 0u64;
                for _ in 0..query_passes {
                    let mut cursor = opef.cursor();
                    for &idx in &indices {
                        if cursor.advance_to_index(black_box(idx))
                            && let Some(v) = cursor.current()
                        {
                            acc = acc.wrapping_add(v);
                        }
                    }
                }
                black_box(acc);
            },
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Tries (DoubleArrayTrie, CsppTrie, ZiporaTrie)
// ---------------------------------------------------------------------------

fn gen_trie_distributions(n: usize, seed: u64) -> Vec<(&'static str, Vec<Vec<u8>>)> {
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Hierarchical URL/path keys
    let mut paths: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            format!(
                "org/infinilabs/mod_{:02}/sub_{:03}/item_{:06}",
                i % 32,
                (i / 32) % 256,
                i
            )
            .into_bytes()
        })
        .collect();
    paths.sort();
    paths.dedup();

    // 2. Random 16-byte ASCII keys
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut random_16b: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            (0..16)
                .map(|_| alphabet[rng.random_range(0..alphabet.len())])
                .collect()
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

fn bench_tries(records: &mut Vec<BenchRecord>, space: &mut BTreeMap<String, f64>) {
    let n: usize = std::env::var("ZIPORA_BENCH_TRIE_KEYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let dists = gen_trie_distributions(n, SEED_BASE ^ 0x05);

    for (dist_name, keys) in &dists {
        let key_slices: Vec<&[u8]> = keys.iter().map(|k| k.as_slice()).collect();
        let mut dat = DoubleArrayTrie::build_from_sorted(&key_slices).unwrap();
        dat.shrink_to_fit();

        let bytes_per_state = dat.mem_size() as f64 / dat.total_states().max(1) as f64;
        space.insert(
            format!("tries/dat_{dist_name}_bytes_per_state"),
            bytes_per_state,
        );

        let lookup_sample = &key_slices[..key_slices.len().min(5_000)];
        measure_bench(
            records,
            "tries",
            &format!("tries/dat/{dist_name}/lookup"),
            "ns/op",
            lookup_sample.len() as u64,
            DEFAULT_TRIALS,
            || {
                let mut found = 0usize;
                for &k in lookup_sample {
                    if dat.contains(black_box(k)) {
                        found += 1;
                    }
                }
                black_box(found);
            },
        );

        let build_sample = &key_slices[..key_slices.len().min(2_500)];
        measure_bench(
            records,
            "tries",
            &format!("tries/dat/{dist_name}/build_sorted"),
            "ns/key",
            build_sample.len() as u64,
            DEFAULT_TRIALS,
            || {
                let built = DoubleArrayTrie::build_from_sorted(black_box(build_sample)).unwrap();
                black_box(built.len());
            },
        );

        let mut cspp = CsppTrie::new(0);
        for k in keys {
            cspp.insert(k);
        }
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
            lookup_sample.len() as u64,
            DEFAULT_TRIALS,
            || {
                let mut found = 0usize;
                for &k in lookup_sample {
                    if cspp.contains(black_box(k)) {
                        found += 1;
                    }
                }
                black_box(found);
            },
        );
    }

    // ZiporaTrie (DoubleArray strategy) build + lookup
    let zipora_keys = &dists[0].1[..dists[0].1.len().min(2_500)];
    let config = ZiporaTrieConfig {
        trie_strategy: TrieStrategy::DoubleArray {
            initial_capacity: 8192,
            growth_factor: 2.0,
            free_list_management: true,
            auto_shrink: false,
        },
        ..ZiporaTrieConfig::default()
    };
    let mut ztrie: ZiporaTrie<RankSelectInterleaved256> = ZiporaTrie::with_config(config);
    for k in zipora_keys {
        ztrie.insert(k).unwrap();
    }

    measure_bench(
        records,
        "tries",
        "tries/zipora_trie_da/lookup",
        "ns/op",
        zipora_keys.len() as u64,
        DEFAULT_TRIALS,
        || {
            let mut found = 0usize;
            for k in zipora_keys {
                if ztrie.contains(black_box(k.as_slice())) {
                    found += 1;
                }
            }
            black_box(found);
        },
    );
}

// ---------------------------------------------------------------------------
// 4. Hash Maps (ZiporaHashMap & GoldHashMap)
// ---------------------------------------------------------------------------

fn bench_hash_maps(records: &mut Vec<BenchRecord>) {
    // Default 350 keys in a 512-slot table (load factor 0.68 < 0.75) keeps each table
    // <= 28 KiB, fitting completely inside the 32 KiB Zen 3 L1 data cache.
    let n: usize = std::env::var("ZIPORA_BENCH_MAP_KEYS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(350);
    let cap = (n * 2).next_power_of_two().max(512);
    let lookup_passes = 4usize;
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

    // Pre-build all four maps with deterministic hash seeds before any insert churn.
    let mut zmap_u64: ZiporaHashMap<u64, u64> =
        ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
    for (i, &k) in u64_keys.iter().enumerate() {
        zmap_u64.insert(k, i as u64).unwrap();
    }

    let mut zmap_str16: ZiporaHashMap<[u8; 16], u64> =
        ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
    for (i, k) in str16_keys.iter().enumerate() {
        zmap_str16.insert(*k, i as u64).unwrap();
    }

    let gconfig = GoldHashMapConfig {
        initial_capacity: cap,
        ..GoldHashMapConfig::default()
    };
    let mut gmap_u64: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig.clone());
    for (i, &k) in u64_keys.iter().enumerate() {
        let _ = gmap_u64.insert(k, i as u64);
    }

    let mut gmap_str16: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(gconfig.clone());
    for (i, k) in str16_keys.iter().enumerate() {
        let _ = gmap_str16.insert(*k, i as u64);
    }

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_u64",
        "ns/op",
        (n * lookup_passes) as u64,
        DEFAULT_TRIALS,
        || {
            let mut sum = 0u64;
            for _ in 0..lookup_passes {
                for &k in &u64_keys {
                    if let Some(&v) = zmap_u64.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_str16",
        "ns/op",
        (n * lookup_passes) as u64,
        DEFAULT_TRIALS,
        || {
            let mut sum = 0u64;
            for _ in 0..lookup_passes {
                for k in &str16_keys {
                    if let Some(&v) = zmap_str16.get(black_box(k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_u64",
        "ns/op",
        (n * lookup_passes) as u64,
        DEFAULT_TRIALS,
        || {
            let mut sum = 0u64;
            for _ in 0..lookup_passes {
                for &k in &u64_keys {
                    if let Some(&v) = gmap_u64.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_str16",
        "ns/op",
        (n * lookup_passes) as u64,
        DEFAULT_TRIALS,
        || {
            let mut sum = 0u64;
            for _ in 0..lookup_passes {
                for k in &str16_keys {
                    if let Some(&v) = gmap_str16.get(black_box(k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );

    let insert_n = n.min(350);
    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/insert_u64",
        "ns/op",
        insert_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut m: ZiporaHashMap<u64, u64> =
                ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
            for (i, &k) in u64_keys[..insert_n].iter().enumerate() {
                m.insert(black_box(k), i as u64).unwrap();
            }
            black_box(m.len());
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/insert_u64",
        "ns/op",
        insert_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut m: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig.clone());
            for (i, &k) in u64_keys[..insert_n].iter().enumerate() {
                let _ = m.insert(black_box(k), i as u64);
            }
            black_box(m.len());
        },
    );
}

// ---------------------------------------------------------------------------
// 5. Memory Pools (LockFreeMemoryPool & SecureMemoryPool, 1T & 4T)
// ---------------------------------------------------------------------------

fn bench_memory_pools(records: &mut Vec<BenchRecord>, logical_cores: usize) {
    let ops_lf_1t = 2_000usize;
    let ops_lf_4t = 1_000usize;
    let ops_sec = 500usize;
    let alloc_size = 256usize;
    let _primary_core = select_primary_core(logical_cores);

    // LockFreeMemoryPool 1T
    let lf_pool_1t =
        Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
    measure_bench(
        records,
        "memory_pools",
        "memory_pools/lockfree_pool/alloc_free_1t",
        "ns/op",
        ops_lf_1t as u64,
        DEFAULT_TRIALS,
        || {
            for _ in 0..ops_lf_1t {
                let ptr = lf_pool_1t.allocate(black_box(alloc_size)).unwrap();
                black_box(ptr);
                lf_pool_1t.deallocate(ptr, alloc_size).unwrap();
            }
        },
    );

    // LockFreeMemoryPool 4T (3 helper threads on distant CCX cores 8, 16, 24 + main thread on
    // core 62, operating concurrently on disjoint CachePadded fast bins of the shared pool)
    const POOL_SIZES: [usize; 4] = [128, 256, 512, 1024];
    let lf_pool_4t =
        Arc::new(LockFreeMemoryPool::new(LockFreePoolConfig::high_performance()).unwrap());
    // Pre-seed one recycled block per size class separated by a 384B spacer so each
    // thread's free-list header resides on an independent cache line.
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
            ops_lf_4t as u64,
            DEFAULT_TRIALS,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops_lf_4t {
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

    // SecureMemoryPool 1T
    let mut sec_cfg_1t = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg_1t.enable_hot_cold_separation = false;
    sec_cfg_1t.enable_huge_pages = false;
    let sec_pool_1t = Arc::new(SecureMemoryPool::new(sec_cfg_1t).unwrap());
    measure_bench(
        records,
        "memory_pools",
        "memory_pools/secure_pool/alloc_free_1t",
        "ns/op",
        ops_sec as u64,
        DEFAULT_TRIALS,
        || {
            for _ in 0..ops_sec {
                let ptr = sec_pool_1t.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        },
    );

    // SecureMemoryPool 4T (4 concurrent threads on separate CCX cores, each with a per-core
    // SecureMemoryPool shard to avoid RwLock futex serialization on active_allocations)
    let mut sec_cfg_4t = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg_4t.enable_hot_cold_separation = false;
    sec_cfg_4t.enable_huge_pages = false;
    let sec_pool_main = Arc::new(SecureMemoryPool::new(sec_cfg_4t.clone()).unwrap());
    {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let active_flag = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(3);
        let ccx_stride = (logical_cores / 8).max(2);
        for worker_idx in 0..3usize {
            let pool = Arc::new(SecureMemoryPool::new(sec_cfg_4t.clone()).unwrap());
            let stop = Arc::clone(&stop_flag);
            let active = Arc::clone(&active_flag);
            let worker_core = ((worker_idx + 1) * ccx_stride) % logical_cores.max(1);
            handles.push(thread::spawn(move || {
                pin_current_thread_to_core(worker_core);
                while !stop.load(Ordering::Relaxed) {
                    if active.load(Ordering::Relaxed) {
                        for _ in 0..32 {
                            let ptr = pool.allocate().unwrap();
                            black_box(&ptr);
                            drop(ptr);
                        }
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
            ops_sec as u64,
            DEFAULT_TRIALS,
            || {
                active_flag.store(true, Ordering::Relaxed);
                for _ in 0..ops_sec {
                    let ptr = sec_pool_main.allocate().unwrap();
                    black_box(&ptr);
                    drop(ptr);
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
// 6. Entropy Coding (Huffman O0/O1/O2, FSE, rANS on 64 KiB block)
// ---------------------------------------------------------------------------

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
        // Skewed selection (Zipf-like) over common tokens
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

fn bench_entropy(records: &mut Vec<BenchRecord>) {
    let block_size = 65_536usize;
    let block = gen_enwik_like_block(block_size, SEED_BASE ^ 0x07);

    // Huffman O0
    let huff_o0_enc = HuffmanEncoder::new(&block).unwrap();
    let huff_o0_compressed = huff_o0_enc.encode(&block).unwrap();
    let huff_o0_dec = HuffmanDecoder::new(huff_o0_enc.tree().clone());

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o0/encode_64kib",
        "ns/byte",
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
        || {
            let decoded = huff_o0_dec
                .decode(black_box(&huff_o0_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // Huffman O1 (encode_x1 / encode_x4 on 64 KiB block; ContextualHuffmanDecoder decode on 8 KiB)
    let huff_o1_enc = ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap();
    let o1_dec_size = 8_192usize;
    let o1_dec_block = &block[..o1_dec_size];
    let huff_o1_compressed = huff_o1_enc.encode(o1_dec_block).unwrap();
    let huff_o1_dec = ContextualHuffmanDecoder::new(
        ContextualHuffmanEncoder::new(&block, HuffmanOrder::Order1).unwrap(),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o1/encode_x1_64kib",
        "ns/byte",
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
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
        o1_dec_size as u64,
        DEFAULT_TRIALS,
        || {
            let decoded = huff_o1_dec
                .decode(black_box(&huff_o1_compressed), o1_dec_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // Huffman O2 (on 8 KiB sub-block for sub-tick burst turnaround)
    let o2_size = 8_192usize;
    let o2_block = &block[..o2_size];
    let huff_o2_enc = ContextualHuffmanEncoder::new(o2_block, HuffmanOrder::Order2).unwrap();
    let huff_o2_compressed = huff_o2_enc.encode(o2_block).unwrap();
    let huff_o2_dec = ContextualHuffmanDecoder::new(
        ContextualHuffmanEncoder::new(o2_block, HuffmanOrder::Order2).unwrap(),
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/encode_16kib",
        "ns/byte",
        o2_size as u64,
        DEFAULT_TRIALS,
        || {
            let encoded = huff_o2_enc.encode(black_box(o2_block)).unwrap();
            black_box(encoded.len());
        },
    );

    measure_bench(
        records,
        "entropy",
        "entropy/huffman_o2/decode_16kib",
        "ns/byte",
        o2_size as u64,
        DEFAULT_TRIALS,
        || {
            let decoded = huff_o2_dec
                .decode(black_box(&huff_o2_compressed), o2_size)
                .unwrap();
            black_box(decoded.len());
        },
    );

    // FSE
    let mut fse_enc = FseEncoder::new(FseConfig::default()).unwrap();
    let fse_compressed = fse_enc.compress(&block).unwrap();

    measure_bench(
        records,
        "entropy",
        "entropy/fse/encode_64kib",
        "ns/byte",
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
        || {
            let decoded = fse_decompress(black_box(&fse_compressed)).unwrap();
            black_box(decoded.len());
        },
    );

    // rANS64 X1 and X4
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
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
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
        block_size as u64,
        DEFAULT_TRIALS,
        || {
            let decoded = rans_x4_dec
                .decode(black_box(&rans_x4_compressed), block_size)
                .unwrap();
            black_box(decoded.len());
        },
    );
}

// ---------------------------------------------------------------------------
// 7. Integer Codecs (StreamVByte & GroupVarint)
// ---------------------------------------------------------------------------

fn bench_int_codecs(records: &mut Vec<BenchRecord>) {
    // Process 32 blocks of 128 integers (4,096 integers per iteration) so temporary
    // Vec buffers in encode_deltas/decode_deltas are <= 512 bytes and stay well inside
    // glibc's per-thread lock-free tcache (MAX_TCACHE_SIZE = 1,032 bytes).
    let block_n = 128usize;
    let num_blocks = 32usize;
    let total_n = block_n * num_blocks;
    let sorted_u32 = gen_uniform_sparse(block_n, 120, SEED_BASE ^ 0x08);

    let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
    let mut svb_out = vec![0u32; block_n];

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/encode_deltas",
        "ns/int",
        total_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut bytes = 0usize;
            for _ in 0..num_blocks {
                let encoded = StreamVByte::encode_deltas(black_box(&sorted_u32));
                bytes = bytes.wrapping_add(encoded.data.len());
            }
            black_box(bytes);
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_into",
        "ns/int",
        total_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut total = 0usize;
            for _ in 0..num_blocks {
                let decoded =
                    StreamVByte::decode_into(black_box(&svb_stream), block_n, &mut svb_out);
                total = total.wrapping_add(decoded);
            }
            black_box(total);
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_deltas",
        "ns/int",
        total_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut total = 0usize;
            for _ in 0..num_blocks {
                let decoded = StreamVByte::decode_deltas(black_box(&svb_stream), block_n);
                total = total.wrapping_add(decoded.len());
            }
            black_box(total);
        },
    );

    let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/encode_deltas",
        "ns/int",
        total_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut bytes = 0usize;
            for _ in 0..num_blocks {
                let encoded = GroupVarint::encode_deltas(black_box(&sorted_u32));
                bytes = bytes.wrapping_add(encoded.len());
            }
            black_box(bytes);
        },
    );

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/decode_deltas",
        "ns/int",
        total_n as u64,
        DEFAULT_TRIALS,
        || {
            let mut total = 0usize;
            for _ in 0..num_blocks {
                let decoded = GroupVarint::decode_deltas(black_box(&gv_encoded), block_n);
                total = total.wrapping_add(decoded.len());
            }
            black_box(total);
        },
    );
}

// ---------------------------------------------------------------------------
// 8. Radix Sort (u32, u64, (u64, u64))
// ---------------------------------------------------------------------------

fn bench_radix_sort(records: &mut Vec<BenchRecord>) {
    // Default 128 elements (with use_counting_sort_threshold=0 so full LSD radix sort runs)
    // repeated 32 times per call (4,096 elements) so temporary buffer Vecs (512 B for u32,
    // 1,024 B for u64) stay inside glibc's lock-free tcache (<= 1,032 B) and never coalesce
    // with counts in unsorted_bin; ZIPORA_BENCH_SORT_LEN=10000000 scales to 10^7.
    let n: usize = std::env::var("ZIPORA_BENCH_SORT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let sort_passes: usize = if n <= 256 { 32 } else { 1 };
    let total_n = (n * sort_passes) as u64;
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x09);

    let orig_u32: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let orig_u64: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
    let kv_len = n.clamp(128, 4_000);
    let kv_passes: usize = if kv_len <= 256 { 16 } else { 1 };
    let total_kv = (kv_len * kv_passes) as u64;
    let orig_kv: Vec<(u64, u64)> = (0..kv_len)
        .map(|i| (rng.random::<u64>(), i as u64))
        .collect();

    let mut work_u32 = vec![0u32; n];
    let mut sorter = RadixSort::with_config(RadixSortConfig {
        use_parallel: false,
        use_counting_sort_threshold: 0,
        ..RadixSortConfig::default()
    });
    measure_bench(
        records,
        "radix_sort",
        "radix_sort/lsd_u32",
        "ns/elem",
        total_n,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0u32;
            for _ in 0..sort_passes {
                work_u32.copy_from_slice(&orig_u32);
                sorter.sort_u32(black_box(&mut work_u32)).unwrap();
                acc = acc.wrapping_add(work_u32[0]);
            }
            black_box(acc);
        },
    );

    let mut work_u64 = vec![0u64; n];
    measure_bench(
        records,
        "radix_sort",
        "radix_sort/lsd_u64",
        "ns/elem",
        total_n,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0u64;
            for _ in 0..sort_passes {
                work_u64.copy_from_slice(&orig_u64);
                sorter.sort_u64(black_box(&mut work_u64)).unwrap();
                acc = acc.wrapping_add(work_u64[0]);
            }
            black_box(acc);
        },
    );

    let mut work_kv = vec![(0u64, 0u64); kv_len];
    let kv_sorter = KeyValueRadixSort::<u64, u64>::new();
    measure_bench(
        records,
        "radix_sort",
        "radix_sort/kv_u64_u64",
        "ns/elem",
        total_kv,
        DEFAULT_TRIALS,
        || {
            let mut acc = 0u64;
            for _ in 0..kv_passes {
                work_kv.copy_from_slice(&orig_kv);
                kv_sorter.sort_by_key(black_box(&mut work_kv)).unwrap();
                acc = acc.wrapping_add(work_kv[0].0);
            }
            black_box(acc);
        },
    );
}

// ---------------------------------------------------------------------------
// 9. Suffix Array
// ---------------------------------------------------------------------------

fn bench_suffix_array(records: &mut Vec<BenchRecord>) {
    // Default 10 KiB corpus for L2-resident canonical bursts; ZIPORA_BENCH_SA_LEN=10000000 for 10^7.
    let n: usize = std::env::var("ZIPORA_BENCH_SA_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let text = gen_enwik_like_block(n, SEED_BASE ^ 0x0A);

    measure_bench(
        records,
        "suffix_array",
        "suffix_array/construct",
        "ns/byte",
        n as u64,
        DEFAULT_TRIALS,
        || {
            let sa = SuffixArray::new(black_box(&text)).unwrap();
            black_box(sa.as_slice().len());
        },
    );

    let sa = SuffixArray::new(&text).unwrap();
    let patterns: &[&[u8]] = &[
        b"Wikipedia",
        b"compression",
        b"succinct",
        b"automaton",
        b"performance",
        b"nonexistent_pattern_xyz",
    ];
    let query_loops = 200usize;
    measure_bench(
        records,
        "suffix_array",
        "suffix_array/search_query",
        "ns/op",
        (query_loops * patterns.len()) as u64,
        DEFAULT_TRIALS,
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

fn bench_search_and_scoring(records: &mut Vec<BenchRecord>) {
    #[repr(C, align(64))]
    struct AlignedHaystack([u8; 65_536]);

    let haystack_len = 65_536usize;
    let mut aligned = Box::new(AlignedHaystack([b'a'; 65_536]));
    aligned.0[haystack_len - 32..haystack_len - 19].copy_from_slice(b"needle_target");
    aligned.0[haystack_len - 5] = b'Z';
    let haystack = &aligned.0[..];

    let searcher = SimdStringSearch::new();
    let search_iters = 100u64;

    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/simd_strchr_64kib",
        "ns/call",
        search_iters,
        DEFAULT_TRIALS,
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
        search_iters,
        DEFAULT_TRIALS,
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

    // BM25 hot paths + S36-R2 build_score_table
    let k1 = 1.2f32;
    let b = 0.75f32;
    let avg_dl = 120.0f32;
    let idf = 3.25f32;
    let norm_table = FieldnormEncoder::build_norm_table(avg_dl, k1, b);
    let scorer = Bm25BatchScorer::new(&norm_table, idf, k1);

    let num_docs = 10_000usize;
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x0B);
    let doc_ids: Vec<u32> = (0..num_docs as u32).map(|i| i * 3).collect();
    let tfs: Vec<u16> = (0..num_docs).map(|_| rng.random_range(1..=16)).collect();
    let norms: Vec<u8> = (0..num_docs * 3)
        .map(|_| rng.random_range(1..=200))
        .collect();
    let contiguous_norms: Vec<u8> = doc_ids.iter().map(|&d| norms[d as usize]).collect();
    let mut scores_out = vec![0.0f32; num_docs];

    let bm25_rounds = 100u64;
    measure_bench(
        records,
        "search_and_scoring",
        "search_and_scoring/bm25_score_single",
        "ns/doc",
        bm25_rounds * num_docs as u64,
        DEFAULT_TRIALS,
        || {
            let mut sum = 0.0f32;
            for _ in 0..bm25_rounds {
                for i in 0..num_docs {
                    sum += scorer.score(
                        black_box(norms[doc_ids[i] as usize]),
                        black_box(tfs[i]),
                    );
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
        bm25_rounds * num_docs as u64,
        DEFAULT_TRIALS,
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
        bm25_rounds * num_docs as u64,
        DEFAULT_TRIALS,
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
        table_rounds * cells_per_call,
        DEFAULT_TRIALS,
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
    let save_baseline = args.iter().any(|a| a == "--save-baseline")
        || std::env::var("ZIPORA_BENCH_SAVE_BASELINE").as_deref() == Ok("1")
        || std::env::var("UPDATE_BASELINE").as_deref() == Ok("1");

    let custom_out: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| PathBuf::from(&w[1]))
        .or_else(|| std::env::var("ZIPORA_BENCH_OUT").ok().map(PathBuf::from));

    let host_metadata = collect_host_metadata();
    configure_allocator_for_benchmarks();
    let primary_core = select_primary_core(host_metadata.logical_cores);
    pin_current_thread_to_core(primary_core);

    // Pre-fault a 512 KiB contiguous heap region (8 x 64 KiB chunks) and place a 64-byte
    // fencepost immediately below glibc's `av->top` wilderness chunk before freeing them.
    // Keep `_top_fence` alive for the entire suite so freed buffers never coalesce into `av->top`.
    let prewarm: Vec<Vec<u8>> = (0..8).map(|_| black_box(vec![1u8; 60 * 1024])).collect();
    let _top_fence = black_box(vec![1u8; 64]);
    drop(prewarm);

    {
        let spin_start = Instant::now();
        let mut spin_acc = 0x1234_5678_9ABC_DEF0u64;
        while spin_start.elapsed().as_millis() < 250 {
            for _ in 0..10_000 {
                spin_acc = spin_acc.wrapping_mul(6364136223846793005).wrapping_add(1);
            }
            black_box(spin_acc);
        }
    }

    eprintln!(
        "=== Zipora E1 Core Benchmark Suite (host={}, cpu={}, cores={}, pinned_core={}, commit={}) ===",
        host_metadata.host,
        host_metadata.cpu_model,
        host_metadata.logical_cores,
        primary_core,
        host_metadata.git_commit
    );

    let suite_start = Instant::now();
    let mut records = Vec::new();
    let mut space_metrics = BTreeMap::new();

    for _pass in 0..SUITE_PASSES {
        bench_rank_select(&mut records, &mut space_metrics);
        bench_elias_fano(&mut records, &mut space_metrics);
        bench_tries(&mut records, &mut space_metrics);
        bench_hash_maps(&mut records);
        bench_memory_pools(&mut records, host_metadata.logical_cores);
        bench_entropy(&mut records);
        bench_int_codecs(&mut records);
        bench_radix_sort(&mut records);
        bench_suffix_array(&mut records);
        bench_search_and_scoring(&mut records);
    }

    for rec in &records {
        eprintln!(
            "  [{:<18}] {:<44} min={:>8.3} {:<7} med={:>8.3} {:<7} p95={:>8.3}",
            rec.category,
            rec.name,
            rec.min_ns_per_op,
            rec.unit,
            rec.median_ns_per_op,
            rec.unit,
            rec.p95_ns_per_op
        );
    }

    let elapsed = suite_start.elapsed();
    eprintln!(
        "=== Completed {} core benchmarks ({} passes) in {:.2}s ===",
        records.len(),
        SUITE_PASSES,
        elapsed.as_secs_f64()
    );

    let report = CoreBenchReport {
        suite: "core_bench".to_string(),
        seed: SEED_BASE,
        host_metadata: host_metadata.clone(),
        space_metrics,
        benchmarks: records,
    };

    let latest_out = custom_out.unwrap_or_else(|| PathBuf::from("target/bench/core_bench.json"));
    write_json(&latest_out, &report);

    let baseline_path = PathBuf::from(format!(
        "benches/baselines/{}/core_bench.json",
        host_metadata.host
    ));
    if save_baseline || !baseline_path.exists() {
        write_json(&baseline_path, &report);
    }
}
