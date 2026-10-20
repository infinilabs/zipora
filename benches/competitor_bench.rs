//! Canonical Phase E Competitor Benchmark Matrix (`plan.md` §1.6 `E2` + `E3` + `B3`).
//!
//! Runs Zipora and each external Rust competitor crate in the **same process**, on the
//! **exact same deterministic fixed-seed datasets, instance counts, and uniform query streams**
//! (`benches/common/`), under the same self-only CPU pinning and quiet-CCD discipline as
//! `benches/core_bench.rs`.
//!
//! Emits structured JSON to `target/bench/competitor_bench.json` (recording per-benchmark
//! latency distributions, space metrics, competitor crate versions/configurations, and
//! same-run Zipora-vs-competitor primary/secondary ratios + `E3` verdicts), and writes
//! `benches/baselines/<host>/competitor_bench.json` when `UPDATE_BASELINE=1`,
//! `ZIPORA_BENCH_SAVE_BASELINE=1`, or `--save-baseline` is explicitly provided.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod common;
mod competitors;

use common::*;
use competitors::*;

fn write_json(path: &Path, report: &CompetitorBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote competitor benchmark report to {}", path.display());
}

fn main() {
    init_bench_allocator_policy();
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

    let mut records = Vec::with_capacity(256);
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
        bench_hash_maps(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("memory_pools") {
        trim_allocator_arena();
        bench_memory_pools(&mut records, &mut space_metrics, logical_cores, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("entropy") {
        trim_allocator_arena();
        bench_entropy(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("int_codecs") {
        trim_allocator_arena();
        bench_int_codecs(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("radix_sort") {
        if should_run_any(&[
            "radix_sort/lsd_u32",
            "radix_sort/voracious/u32",
            "radix_sort/rdst/u32",
            "radix_sort/std_unstable/u32",
            "radix_sort/lsd_u64",
            "radix_sort/voracious/u64",
            "radix_sort/rdst/u64",
            "radix_sort/std_unstable/u64",
        ]) {
            trim_allocator_arena();
            bench_radix_sort_lsd(&mut records, &mut space_metrics, sizes);
            maybe_repin_self_within_ccd();
        }
        if should_run_any(&[
            "radix_sort/kv_u64_u64",
            "radix_sort/voracious/kv_u64_u64",
            "radix_sort/rdst/kv_u64_u64",
            "radix_sort/std_unstable/kv_u64_u64",
        ]) {
            trim_allocator_arena();
            bench_radix_sort_kv(&mut records, &mut space_metrics, primary_core, sizes);
            maybe_repin_self_within_ccd();
        }
    }
    if run_cat("suffix_array") {
        trim_allocator_arena();
        bench_suffix_array(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("search_and_scoring") {
        trim_allocator_arena();
        bench_search_and_scoring(&mut records, &mut space_metrics, sizes);
        maybe_repin_self_within_ccd();
    }

    flush_prealloc_space_metrics(&mut space_metrics);
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

#[unsafe(no_mangle)]
pub extern "C" fn _zipora_competitor_bench_text_pad(x: u64) -> u64 {
    let a = x.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    a.rotate_left(17)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
        .wrapping_add(x)
}

#[used]
static _COMP_TEXT_PAD_REF: extern "C" fn(u64) -> u64 = _zipora_competitor_bench_text_pad;
