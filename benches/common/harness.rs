//! Suite size presets, host metadata collection, and true-min benchmark timing harness.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::affinity::*;
use super::binary_meta::detect_self_binary_path_and_sha256;

#[derive(Debug, Clone, Copy)]
pub struct SuiteSizes {
    pub rank_select_bits: usize,
    pub rank_select_queries: usize,
    pub elias_fano_elements: usize,
    pub trie_keys: usize,
    pub hash_map_keys: usize,
    pub memory_pool_ops: usize,
    pub entropy_block_bytes: usize,
    pub int_codec_elements: usize,
    pub radix_sort_elements: usize,
    pub suffix_array_bytes: usize,
    pub suffix_array_queries: usize,
    pub bm25_docs: usize,
}

impl SuiteSizes {
    pub fn spec() -> Self {
        Self {
            rank_select_bits: std::env::var("ZIPORA_BENCH_RANK_BITS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000_000), // 1 Gbit (10^9 bits)
            rank_select_queries: std::env::var("ZIPORA_BENCH_RANK_QUERIES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000), // 1M distinct uniform queries (zero replay)
            elias_fano_elements: std::env::var("ZIPORA_BENCH_EF_LEN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100_000),
            trie_keys: std::env::var("ZIPORA_BENCH_TRIE_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000), // 1M keys
            hash_map_keys: std::env::var("ZIPORA_BENCH_MAP_KEYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000), // 1M keys
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
            suffix_array_queries: 100_000, // 100k distinct uniform substring queries
            bm25_docs: 100_000,
        }
    }

    pub fn smoke() -> Self {
        Self {
            rank_select_bits: 1 << 20, // 1 Mbit
            rank_select_queries: 10_000,
            elias_fano_elements: 10_000,
            trie_keys: 25_000,
            hash_map_keys: 25_000,
            memory_pool_ops: 2_000,
            entropy_block_bytes: 65_536,
            int_codec_elements: 10_000,
            radix_sort_elements: 100_000,
            suffix_array_bytes: 100_000,
            suffix_array_queries: 5_000,
            bm25_docs: 10_000,
        }
    }

    pub fn to_map(self) -> BTreeMap<String, u64> {
        let mut map = BTreeMap::new();
        map.insert("rank_select_bits".to_string(), self.rank_select_bits as u64);
        map.insert(
            "rank_select_queries".to_string(),
            self.rank_select_queries as u64,
        );
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
        map.insert(
            "suffix_array_queries".to_string(),
            self.suffix_array_queries as u64,
        );
        map.insert("bm25_docs".to_string(), self.bm25_docs as u64);
        map
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostMetadata {
    pub host: String,
    pub arch: String,
    pub cpu_model: String,
    pub cpu_mhz: String,
    pub cache_size: String,
    pub logical_cores: usize,
    pub pinned_core: usize,
    pub cpu_governor: String,
    pub rustflags: String,
    pub runner_command: String,
    pub git_commit: String,
    #[serde(default)]
    pub binary_path: String,
    #[serde(default)]
    pub binary_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRecord {
    pub name: String,
    pub category: String,
    pub unit: String,
    pub dataset_size: u64,
    pub ops_per_iter: u64,
    pub trials: usize,
    pub min_ns_per_op: f64,
    pub median_ns_per_op: f64,
    pub p95_ns_per_op: f64,
    pub spread_pct: f64,
    pub max_regression_pct: f64,
    pub samples: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreBenchReport {
    pub suite: String,
    pub tier: String,
    pub seed: u64,
    pub entropy_block_source: String,
    pub host_metadata: HostMetadata,
    pub dataset_sizes: BTreeMap<String, u64>,
    pub space_metrics: BTreeMap<String, f64>,
    pub benchmarks: Vec<BenchRecord>,
}

pub fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

pub fn collect_host_metadata(pinned_core: usize) -> HostMetadata {
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
    let logical_cores = hardware_logical_cores();
    let cpu_governor = read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|| "unavailable (virtualized/no-cpufreq)".to_string());

    let rustflags = std::env::var("RUSTFLAGS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| option_env!("ZIPORA_EFFECTIVE_RUSTFLAGS").map(str::to_string))
        .unwrap_or_else(|| {
            "-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32"
                .to_string()
        });
    let runner_command = format!(
        "RUSTFLAGS=\"{rustflags}\" cargo bench --locked --bench core_bench && python3 scripts/bench_compare.py --suite core_bench --confirm-retries 5"
    );

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

    let (binary_path, binary_sha256) = detect_self_binary_path_and_sha256();

    HostMetadata {
        host,
        arch: std::env::consts::ARCH.to_string(),
        cpu_model,
        cpu_mhz,
        cache_size,
        logical_cores,
        pinned_core,
        cpu_governor,
        rustflags,
        runner_command,
        git_commit,
        binary_path,
        binary_sha256,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BenchParams {
    pub dataset_size: u64,
    pub ops_per_iter: u64,
    pub warmup_calls: usize,
    pub trials: usize,
    pub bursts_per_trial: usize,
    pub max_regression_pct: f64,
}

impl BenchParams {
    pub const fn fast(dataset_size: u64, ops_per_iter: u64) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 2,
            trials: 11,
            bursts_per_trial: 6,
            max_regression_pct: 5.0,
        }
    }

    pub const fn medium(dataset_size: u64, ops_per_iter: u64) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 2,
            trials: 7,
            bursts_per_trial: 3,
            max_regression_pct: 9.0,
        }
    }

    /// Multi-megabyte DRAM / OS-allocator benchmark parameter preset.
    /// Uses a 9.0% regression threshold (strictly `< 9.5%`, so any `+9.5%` or `+10%`
    /// slowdown fails) to account for multi-CCD socket memory-controller contention and
    /// Linux page-fault variance on 16-320 MiB working sets on a shared host.
    pub const fn heavy(dataset_size: u64, ops_per_iter: u64, trials: usize) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 1,
            trials,
            bursts_per_trial: 1,
            max_regression_pct: 9.0,
        }
    }

    pub const fn with_threshold(mut self, max_regression_pct: f64) -> Self {
        self.max_regression_pct = max_regression_pct;
        self
    }
}

pub fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

pub fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

pub fn record_space_metric(space: &mut BTreeMap<String, f64>, key: &str, val: f64) {
    super::datasets::record_prealloc_space_metric(space, key, val);
}

pub fn measure_bench<F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    mut f: F,
) where
    F: FnMut(),
{
    measure_bench_inner::<(), _, _, false>(
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
pub fn measure_bench_with_setup<T, S, F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    state: &mut T,
    setup: S,
    f: F,
) where
    S: FnMut(&mut T),
    F: FnMut(&mut T),
{
    measure_bench_inner::<T, S, F, true>(records, category, name, unit, params, state, setup, f);
}

#[allow(clippy::too_many_arguments)]
pub fn measure_bench_inner<T, S, F, const HAS_SETUP: bool>(
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
    assert!(
        params.trials > 0 && params.trials <= 16,
        "trials must be in 1..=16 for {name}"
    );
    assert!(
        params.bursts_per_trial > 0,
        "bursts_per_trial must be > 0 for {name}"
    );

    if !should_run_bench(name) {
        return;
    }

    let filtered_mode = has_active_filter();
    let check_preempt = true;
    let check_sibling = true;
    let allow_repin = is_self_repin_allowed();
    if allow_repin {
        if filtered_mode {
            wait_for_quiet_host_and_repin(8_000, false);
        } else {
            maybe_repin_self_within_ccd();
        }
        let (_cur_ns0, sib_ns0, _sib_pc0, sib_j0, ccd_ns0) = read_current_and_sibling_raw();
        let spin_start = Instant::now();
        let cpu0 = current_thread_cpu_ns();
        let mut acc = 1u64;
        while spin_start.elapsed() < Duration::from_micros(500) {
            for _ in 0..256 {
                acc = black_box(acc.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
            }
        }
        black_box(acc);
        let wall_ns = spin_start.elapsed().as_nanos() as u64;
        let cpu1 = current_thread_cpu_ns();
        let (_cur_ns1, sib_ns1, _sib_pc1, sib_j1, ccd_ns1) = read_current_and_sibling_raw();
        let stolen = wall_ns.saturating_sub(cpu1.saturating_sub(cpu0));
        let sib_d_ns = sib_ns1.saturating_sub(sib_ns0);
        let sib_d_j = sib_j1.saturating_sub(sib_j0);
        let ccd_d_ns = ccd_ns1.saturating_sub(ccd_ns0);
        if stolen >= 15_000 || sib_d_ns >= 15_000 || sib_d_j > 0 || ccd_d_ns >= 250_000 {
            wait_for_quiet_sibling();
        }
    } else if read_nr_running() > 6 {
        wait_for_quiet_host_and_repin(if filtered_mode { 8_000 } else { 2_500 }, false);
    }
    let mut samples = [0.0f64; 16];
    CURRENT_BURST_IDX.store(0, Ordering::Relaxed);
    for _ in 0..params.warmup_calls {
        setup(state);
        f(state);
    }
    let profile_loop_ms: u64 = std::env::var("ZIPORA_PROFILE_LOOP_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_else(|| {
            if matches!(
                std::env::var("ZIPORA_PROFILE_LOOP").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            ) || std::env::args().any(|a| a == "--profile-loop")
            {
                400
            } else {
                0
            }
        });
    if profile_loop_ms > 0 {
        if let Ok(maps_path) = std::env::var("ZIPORA_PROFILE_MAPS_OUT") {
            let _ = std::fs::copy("/proc/self/maps", maps_path);
        }
        let target_ns = profile_loop_ms.saturating_mul(1_000_000);
        let mut accum_ns = 0u64;
        while accum_ns < target_ns {
            setup(state);
            #[cfg(target_os = "linux")]
            // SAFETY: `PR_TASK_PERF_EVENTS_ENABLE` enables perf events attached to the calling task.
            unsafe {
                let _ = libc::prctl(libc::PR_TASK_PERF_EVENTS_ENABLE, 0, 0, 0, 0);
            }
            let t0 = Instant::now();
            f(state);
            let dt = t0.elapsed().as_nanos() as u64;
            #[cfg(target_os = "linux")]
            // SAFETY: `PR_TASK_PERF_EVENTS_DISABLE` disables perf events immediately after `f(state)`.
            unsafe {
                let _ = libc::prctl(libc::PR_TASK_PERF_EVENTS_DISABLE, 0, 0, 0, 0);
            }
            accum_ns = accum_ns.saturating_add(dt.max(1));
        }
    }
    let uses_burst_idx = CURRENT_BURST_IDX.load(Ordering::Relaxed) > 0;
    let max_bench_attempts =
        if check_sibling && allow_repin && params.bursts_per_trial >= 2 && !uses_burst_idx {
            5usize
        } else {
            1usize
        };
    let retry_mult = if filtered_mode && read_nr_running() > 16 {
        2usize
    } else {
        1usize
    };
    for bench_attempt in 0..max_bench_attempts {
        let bench_start = Instant::now();
        let per_trial_burst_retries = (if params.bursts_per_trial == 1 {
            3usize
        } else if params.bursts_per_trial == 2 {
            4usize
        } else {
            6usize
        }) * retry_mult;
        let mut total_burst_retries = (if params.bursts_per_trial == 1 {
            6usize
        } else {
            (per_trial_burst_retries * params.trials).min(32)
        }) * retry_mult;
        let mut bench_trial_retries = if check_sibling {
            (if params.bursts_per_trial == 1 {
                0usize
            } else if params.bursts_per_trial == 2 {
                4usize
            } else {
                6usize
            }) * retry_mult
        } else {
            0usize
        };
        let (mut t_sib_ns0, mut _t_sib_pc0, mut _t_sib_j0, mut t_ccd_ns0) = if check_sibling {
            read_current_sibling_activity()
        } else {
            (0, 0, 0, 0)
        };
        let bench_sib_ns0 = t_sib_ns0;
        let bench_sib_j0 = _t_sib_j0;
        let bench_ccd_ns0 = t_ccd_ns0;
        let mut b_sib_ns0 = t_sib_ns0;
        let mut b_ccd_ns0 = t_ccd_ns0;
        for sample_slot in samples.iter_mut().take(params.trials) {
            let eff_bursts_per_trial = if filtered_mode
                && read_nr_running() > 16
                && params.bursts_per_trial == 1
                && params.trials <= 4
            {
                2usize
            } else {
                params.bursts_per_trial
            };
            if filtered_mode && read_nr_running() > 16 {
                wait_for_quiet_host_and_repin(12, allow_repin);
                let (ns_after, pc_after, j_after, ccd_after) = read_current_sibling_activity();
                t_sib_ns0 = ns_after;
                b_sib_ns0 = ns_after;
                b_ccd_ns0 = ccd_after;
                _t_sib_pc0 = pc_after;
                _t_sib_j0 = j_after;
                t_ccd_ns0 = ccd_after;
            }
            let mut best_burst_ns = f64::MAX;
            let mut fallback_burst_ns = f64::MAX;
            loop {
                let mut trial_burst_retries = per_trial_burst_retries.min(total_burst_retries);
                let mut trial_wall_ns = 0u64;
                let mut b = 0usize;
                while b < eff_bursts_per_trial {
                    if HAS_SETUP {
                        let setup_start = Instant::now();
                        setup(state);
                        let setup_ns = setup_start.elapsed().as_nanos() as u64;
                        if check_sibling && setup_ns >= 100_000 {
                            let (s_after_setup, _, _, c_after_setup) =
                                read_current_sibling_activity();
                            let setup_sib = s_after_setup.saturating_sub(b_sib_ns0);
                            let setup_ccd = c_after_setup.saturating_sub(b_ccd_ns0);
                            t_sib_ns0 = t_sib_ns0.saturating_add(setup_sib);
                            t_ccd_ns0 = t_ccd_ns0.saturating_add(setup_ccd);
                            b_sib_ns0 = s_after_setup;
                            b_ccd_ns0 = c_after_setup;
                        }
                    } else {
                        setup(state);
                    }
                    let cpu0 = if check_preempt {
                        current_thread_cpu_ns()
                    } else {
                        0
                    };
                    let start = Instant::now();
                    f(state);
                    let wall_ns = start.elapsed().as_nanos() as u64;
                    trial_wall_ns = trial_wall_ns.saturating_add(wall_ns);
                    let cpu1 = if check_preempt {
                        current_thread_cpu_ns()
                    } else {
                        0
                    };
                    let d_cpu = cpu1.saturating_sub(cpu0);
                    let stolen = wall_ns.saturating_sub(d_cpu);
                    let elapsed_ns = if allow_repin && wall_ns >= 25_000_000 && d_cpu > 0 {
                        d_cpu.min(wall_ns) as f64
                    } else {
                        wall_ns as f64
                    };
                    if elapsed_ns < fallback_burst_ns {
                        fallback_burst_ns = elapsed_ns;
                    }
                    let (b_sib_busy, b_ccd_busy) = if check_sibling && wall_ns >= 250_000 {
                        let (s_ns1, _, _, c_ns1) = read_current_sibling_activity();
                        let sib_delta = s_ns1.saturating_sub(b_sib_ns0);
                        let ccd_delta = c_ns1.saturating_sub(b_ccd_ns0);
                        b_sib_ns0 = s_ns1;
                        b_ccd_ns0 = c_ns1;
                        (sib_delta, ccd_delta)
                    } else {
                        (0, 0)
                    };
                    if check_preempt && trial_burst_retries > 0 && total_burst_retries > 0 {
                        let is_long_burst = allow_repin && wall_ns >= 100_000_000;
                        let preempted = if is_long_burst {
                            stolen.saturating_mul(50) >= wall_ns
                        } else {
                            (stolen >= 20_000 && stolen.saturating_mul(25) >= wall_ns)
                                || (allow_repin
                                    && stolen >= 2_500_000
                                    && stolen.saturating_mul(50) >= wall_ns)
                        };
                        let sib_contended = if is_long_burst {
                            b_sib_busy.saturating_mul(14) >= wall_ns
                        } else {
                            (b_sib_busy >= 60_000 && b_sib_busy.saturating_mul(20) >= wall_ns)
                                || (allow_repin
                                    && b_sib_busy >= 3_000_000
                                    && b_sib_busy.saturating_mul(40) >= wall_ns)
                        };
                        let nr_now = read_nr_running();
                        let ccd_contended = allow_repin
                            && (!is_long_burst || nr_now <= 6)
                            && wall_ns >= 2_000_000
                            && b_ccd_busy >= wall_ns.saturating_mul(3)
                            && nr_now <= 16;
                        if preempted || sib_contended || ccd_contended {
                            if allow_repin
                                && (stolen >= 500_000 || b_sib_busy >= 150_000 || ccd_contended)
                            {
                                wait_for_quiet_sibling();
                                let (ns_after, pc_after, j_after, ccd_after) =
                                    read_current_sibling_activity();
                                t_sib_ns0 = ns_after;
                                b_sib_ns0 = ns_after;
                                b_ccd_ns0 = ccd_after;
                                _t_sib_pc0 = pc_after;
                                _t_sib_j0 = j_after;
                                t_ccd_ns0 = ccd_after;
                            } else if !allow_repin {
                                let wake_start = Instant::now();
                                let mut acc = 1u64;
                                while wake_start.elapsed() < Duration::from_micros(15_000) {
                                    for _ in 0..256 {
                                        acc = black_box(
                                            acc.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1),
                                        );
                                    }
                                }
                                black_box(acc);
                                let (ns_after, pc_after, j_after, ccd_after) =
                                    read_current_sibling_activity();
                                t_sib_ns0 = ns_after;
                                b_sib_ns0 = ns_after;
                                b_ccd_ns0 = ccd_after;
                                _t_sib_pc0 = pc_after;
                                _t_sib_j0 = j_after;
                                t_ccd_ns0 = ccd_after;
                            }
                            trial_burst_retries -= 1;
                            total_burst_retries -= 1;
                            continue;
                        }
                    }
                    if elapsed_ns < best_burst_ns {
                        best_burst_ns = elapsed_ns;
                    }
                    b += 1;
                }
                if check_sibling {
                    let (t_sib_ns1, t_sib_pc1, t_sib_j1, t_ccd_ns1) =
                        read_current_sibling_activity();
                    let sib_busy_ns = t_sib_ns1.saturating_sub(t_sib_ns0);
                    let ccd_busy_ns = t_ccd_ns1.saturating_sub(t_ccd_ns0);
                    t_sib_ns0 = t_sib_ns1;
                    b_sib_ns0 = t_sib_ns1;
                    b_ccd_ns0 = t_ccd_ns1;
                    _t_sib_pc0 = t_sib_pc1;
                    _t_sib_j0 = t_sib_j1;
                    t_ccd_ns0 = t_ccd_ns1;
                    if allow_repin
                        && bench_trial_retries > 0
                        && (params.bursts_per_trial <= 2 || best_burst_ns == f64::MAX)
                        && ((sib_busy_ns >= 150_000
                            && sib_busy_ns.saturating_mul(16) >= trial_wall_ns)
                            || (sib_busy_ns >= 8_000_000
                                && sib_busy_ns.saturating_mul(25) >= trial_wall_ns)
                            || ((2_000_000..100_000_000).contains(&trial_wall_ns)
                                && ccd_busy_ns >= trial_wall_ns.saturating_mul(3)
                                && read_nr_running() <= 16))
                    {
                        wait_for_quiet_sibling();
                        let (ns_after, pc_after, j_after, ccd_after) =
                            read_current_sibling_activity();
                        t_sib_ns0 = ns_after;
                        b_sib_ns0 = ns_after;
                        b_ccd_ns0 = ccd_after;
                        _t_sib_pc0 = pc_after;
                        _t_sib_j0 = j_after;
                        t_ccd_ns0 = ccd_after;
                        bench_trial_retries -= 1;
                        continue;
                    }
                }
                break;
            }
            let chosen_ns = if best_burst_ns < f64::MAX {
                best_burst_ns
            } else {
                fallback_burst_ns
            };
            *sample_slot = chosen_ns / (params.ops_per_iter as f64);
        }
        if bench_attempt + 1 < max_bench_attempts
            && bench_start.elapsed() < Duration::from_millis(25)
        {
            let mut acc = 1u64;
            while bench_start.elapsed() < Duration::from_micros(5_000) {
                for _ in 0..256 {
                    acc = black_box(acc.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));
                }
            }
            black_box(acc);
            let total_wall_ns = bench_start.elapsed().as_nanos() as u64;
            let (end_sib_ns, _, end_sib_j, end_ccd_ns) = read_current_sibling_activity();
            let total_sib_ns = end_sib_ns.saturating_sub(bench_sib_ns0);
            let total_ccd_ns = end_ccd_ns.saturating_sub(bench_ccd_ns0);
            if (total_sib_ns >= 350_000 && total_sib_ns.saturating_mul(10) >= total_wall_ns)
                || (total_wall_ns <= 25_000_000 && end_sib_j > bench_sib_j0)
                || (total_wall_ns <= 25_000_000
                    && total_ccd_ns >= 2_000_000
                    && total_ccd_ns.saturating_mul(2) >= total_wall_ns)
            {
                wait_for_quiet_sibling();
                continue;
            }
        }
        break;
    }

    let s_slice = &mut samples[..params.trials];
    s_slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let total_trials = s_slice.len();
    let min_ns = s_slice[0];
    let median_ns = s_slice[total_trials / 2];
    let p95_idx = ((total_trials as f64 * 0.95).floor() as usize).min(total_trials - 1);
    let p95_ns = s_slice[p95_idx];
    let spread_pct = if min_ns > 0.0 {
        ((p95_ns - min_ns) / min_ns) * 100.0
    } else {
        0.0
    };
    let (name_owned, category_owned, unit_owned, mut rounded_samples) =
        super::datasets::take_prealloc_record_buffers(name, category, unit);
    for &s in s_slice.iter() {
        rounded_samples.push(round4(s));
    }

    records.push(BenchRecord {
        name: name_owned,
        category: category_owned,
        unit: unit_owned,
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
