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
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
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
    // for the calling thread only (pid = 0). Never modifies any other process.
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(core_id, &mut cpuset);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_core(_core_id: usize) {}

/// Disables Transparent Huge Pages (`PR_SET_THP_DISABLE`) for the current benchmark
/// process only (`self`), unless `--thp-on` or `ZIPORA_BENCH_THP=on` (`1`) is set,
/// while keeping 100% default production glibc `malloc` settings.
///
/// Disclosure & Justification (`S38-R2` / `S39-R2`): `bindiego` is a KVM virtual machine where
/// `/sys/kernel/mm/transparent_hugepage/enabled` is `[always]`, `defrag` is `[madvise]`,
/// and `/proc/buddyinfo` has fluctuating free `order 9` (`2 MiB`) blocks in `Zone Normal`.
/// Without `PR_SET_THP_DISABLE`, 16-160 MiB structures (`Rank9`, `ZiporaTrie`, `GoldHashMap`,
/// `ZiporaHashMap`, `KeyValueRadixSort`, `SuffixArray`) non-deterministically receive variable
/// fractions of `2 MiB` hugepages across runs (measured at up to 35-55% faster random lookups
/// and 15-35% faster builds when backed by 2 MiB THP vs 4 KiB pages due to L2 dTLB coverage;
/// see `docs/review/e1.md` §1.2 for the full `--thp-on` vs default `THP-off` comparison table).
/// Setting `PR_SET_THP_DISABLE` by default ensures deterministic `4 KiB` page backing for regression
/// gating; pass `--thp-on` (or `ZIPORA_BENCH_THP=on`) to measure with OS transparent hugepages enabled.
#[cfg(target_os = "linux")]
fn disable_transparent_hugepages_for_self() {
    // SAFETY: `prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)` affects only the calling process.
    unsafe {
        let _ = libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }
}

#[cfg(not(target_os = "linux"))]
fn disable_transparent_hugepages_for_self() {}

fn parse_cpu_list(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in spec.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo_s, hi_s)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo_s.parse::<usize>(), hi_s.parse::<usize>()) {
                for c in lo..=hi {
                    out.push(c);
                }
            }
        } else if let Ok(c) = part.parse::<usize>() {
            out.push(c);
        }
    }
    out
}

fn hardware_logical_cores() -> usize {
    if let Ok(online) = fs::read_to_string("/sys/devices/system/cpu/online") {
        let cpus = parse_cpu_list(&online);
        if !cpus.is_empty() {
            return cpus.len();
        }
    }
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let count = cpuinfo
            .lines()
            .filter(|l| l.starts_with("processor"))
            .count();
        if count > 0 {
            return count;
        }
    }
    thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn explicit_pin_core_override(logical_cores: usize) -> Option<usize> {
    for env_key in ["ZIPORA_BENCH_PIN_CORE", "ZIPORA_BENCH_CORE"] {
        if let Ok(val) = std::env::var(env_key)
            && let Ok(parsed) = val.parse::<usize>()
            && parsed < logical_cores
        {
            return Some(parsed);
        }
    }
    None
}

/// Selects the initial physical core on interior L3 CCD 2 (`default_core = 23` on `bindiego`)
/// for the calling thread (`pid = 0` only; never modifies any other process).
fn select_primary_core(logical_cores: usize) -> usize {
    if let Some(parsed) = explicit_pin_core_override(logical_cores) {
        return parsed;
    }
    let phys_cores = (logical_cores / 2).max(1);
    if phys_cores >= 24 {
        23
    } else {
        phys_cores.saturating_sub(1)
    }
}

static BENCH_FILTER: OnceLock<Option<Vec<String>>> = OnceLock::new();
static HOST_WAIT_BUDGET_MS: AtomicU64 = AtomicU64::new(20_000);

fn init_bench_filter(filter_spec: Option<String>) {
    let parsed = filter_spec.and_then(|s| {
        let items: Vec<String> = s
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect();
        if items.is_empty() { None } else { Some(items) }
    });
    if parsed.is_some() {
        HOST_WAIT_BUDGET_MS.store(60_000, Ordering::Relaxed);
    }
    let _ = BENCH_FILTER.set(parsed);
}

fn has_active_filter() -> bool {
    BENCH_FILTER
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|v| !v.is_empty())
}

fn should_run_bench(name: &str) -> bool {
    let Some(Some(filters)) = BENCH_FILTER.get() else {
        return true;
    };
    filters
        .iter()
        .any(|f| name == f.as_str() || name.contains(f.as_str()))
}

fn should_run_any(names: &[&str]) -> bool {
    names.iter().any(|n| should_run_bench(n))
}

fn should_run_prefix(prefix: &str) -> bool {
    let Some(Some(filters)) = BENCH_FILTER.get() else {
        return true;
    };
    filters.iter().any(|f| {
        !f.contains('/')
            || f.starts_with(prefix)
            || prefix.starts_with(f.as_str())
            || f.contains(prefix)
    })
}

fn should_run_category(cat: &str, cat_filter: Option<&str>) -> bool {
    if let Some(cf) = cat_filter
        && !cf.is_empty()
        && !cf.split(',').any(|p| p.trim() == cat)
    {
        return false;
    }
    let cat_prefix = format!("{cat}/");
    should_run_prefix(&cat_prefix)
}

struct SelfPinTracker {
    primary_core: usize,
    current_core: usize,
    sibling_core: usize,
    logical_cores: usize,
    allow_repin: bool,
    user_pinned: bool,
    schedstat_file: Option<fs::File>,
    procstat_file: Option<fs::File>,
    loadavg_file: Option<fs::File>,
}

static SELF_PIN_TRACKER: Mutex<SelfPinTracker> = Mutex::new(SelfPinTracker {
    primary_core: 23,
    current_core: 23,
    sibling_core: 55,
    logical_cores: 64,
    allow_repin: true,
    user_pinned: false,
    schedstat_file: None,
    procstat_file: None,
    loadavg_file: None,
});

static CURRENT_BURST_IDX: AtomicUsize = AtomicUsize::new(0);

#[inline(always)]
fn current_burst_index() -> usize {
    CURRENT_BURST_IDX.fetch_add(1, Ordering::Relaxed)
}

#[cfg(target_os = "linux")]
fn current_thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `CLOCK_THREAD_CPUTIME_ID` is valid for the calling thread and `ts` is a
    // valid mutable stack `timespec`.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc == 0 {
        (ts.tv_sec as u64)
            .saturating_mul(1_000_000_000)
            .saturating_add(ts.tv_nsec as u64)
    } else {
        0
    }
}

#[cfg(not(target_os = "linux"))]
fn current_thread_cpu_ns() -> u64 {
    0
}

fn read_file_into_stack_buf(file_opt: &mut Option<fs::File>, buf: &mut [u8]) -> usize {
    let Some(file) = file_opt.as_mut() else {
        return 0;
    };
    if file.seek(SeekFrom::Start(0)).is_err() {
        return 0;
    }
    let mut total = 0usize;
    while total < buf.len() {
        match file.read(&mut buf[total..]) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    total
}

fn parse_loadavg_nr_running(buf: &[u8]) -> usize {
    let mut idx = 0usize;
    for _ in 0..3 {
        while idx < buf.len() && (buf[idx] == b' ' || buf[idx] == b'\t') {
            idx += 1;
        }
        while idx < buf.len() && buf[idx] != b' ' && buf[idx] != b'\t' && buf[idx] != b'\n' {
            idx += 1;
        }
    }
    while idx < buf.len() && (buf[idx] == b' ' || buf[idx] == b'\t') {
        idx += 1;
    }
    let mut nr = 0usize;
    let mut has_digit = false;
    while idx < buf.len() && buf[idx].is_ascii_digit() {
        nr = nr
            .saturating_mul(10)
            .saturating_add((buf[idx] - b'0') as usize);
        idx += 1;
        has_digit = true;
    }
    if has_digit { nr } else { 1 }
}

fn read_nr_running() -> usize {
    let Ok(mut guard) = SELF_PIN_TRACKER.lock() else {
        return 1;
    };
    let mut buf = [0u8; 128];
    let len = read_file_into_stack_buf(&mut guard.loadavg_file, &mut buf);
    if len > 0 {
        parse_loadavg_nr_running(&buf[..len])
    } else {
        1
    }
}

fn format_cpu_prefix(cpu_id: usize, out: &mut [u8; 12]) -> usize {
    out[0] = b'c';
    out[1] = b'p';
    out[2] = b'u';
    let mut n = cpu_id;
    let mut digits = [0u8; 8];
    let mut dlen = 0usize;
    if n == 0 {
        digits[0] = b'0';
        dlen = 1;
    } else {
        while n > 0 {
            digits[dlen] = b'0' + (n % 10) as u8;
            dlen += 1;
            n /= 10;
        }
    }
    for i in 0..dlen {
        out[3 + i] = digits[dlen - 1 - i];
    }
    out[3 + dlen] = b' ';
    4 + dlen
}

fn parse_cpu_line_tokens(buf: &[u8], cpu_id: usize, tokens: &mut [u64; 10]) -> usize {
    let mut prefix = [0u8; 12];
    let plen = format_cpu_prefix(cpu_id, &mut prefix);
    for line in buf.split(|&b| b == b'\n') {
        if line.starts_with(&prefix[..plen]) {
            let mut count = 0usize;
            let mut idx = plen;
            let rest = line;
            while idx < rest.len() && count < tokens.len() {
                while idx < rest.len()
                    && (rest[idx] == b' ' || rest[idx] == b'\t' || rest[idx] == b'\r')
                {
                    idx += 1;
                }
                if idx >= rest.len() {
                    break;
                }
                let mut val = 0u64;
                let mut has_digit = false;
                while idx < rest.len() && rest[idx].is_ascii_digit() {
                    val = val
                        .saturating_mul(10)
                        .saturating_add((rest[idx] - b'0') as u64);
                    idx += 1;
                    has_digit = true;
                }
                if has_digit {
                    tokens[count] = val;
                    count += 1;
                } else {
                    break;
                }
            }
            return count;
        }
    }
    0
}

fn parse_all_schedstat_ns(sched_buf: &[u8], out_ns: &mut [u64; 64], out_pc: &mut [u64; 64]) {
    for line in sched_buf.split(|&b| b == b'\n') {
        if line.len() < 5
            || line[0] != b'c'
            || line[1] != b'p'
            || line[2] != b'u'
            || !line[3].is_ascii_digit()
        {
            continue;
        }
        let mut idx = 3usize;
        let mut cid = 0usize;
        while idx < line.len() && line[idx].is_ascii_digit() {
            cid = cid
                .saturating_mul(10)
                .saturating_add((line[idx] - b'0') as usize);
            idx += 1;
        }
        if cid >= 64 || idx >= line.len() || line[idx] != b' ' {
            continue;
        }
        let mut tokens = [0u64; 10];
        let mut count = 0usize;
        while idx < line.len() && count < tokens.len() {
            while idx < line.len()
                && (line[idx] == b' ' || line[idx] == b'\t' || line[idx] == b'\r')
            {
                idx += 1;
            }
            if idx >= line.len() {
                break;
            }
            let mut val = 0u64;
            let mut has_digit = false;
            while idx < line.len() && line[idx].is_ascii_digit() {
                val = val
                    .saturating_mul(10)
                    .saturating_add((line[idx] - b'0') as u64);
                idx += 1;
                has_digit = true;
            }
            if has_digit {
                tokens[count] = val;
                count += 1;
            } else {
                break;
            }
        }
        if count >= 9 {
            out_ns[cid] = tokens[6];
            out_pc[cid] = tokens[8];
        }
    }
}

fn parse_ccd_schedstat(sched_buf: &[u8], cur: usize, sib: usize) -> (u64, u64, u64, u64) {
    let mut out_ns = [0u64; 64];
    let mut out_pc = [0u64; 64];
    parse_all_schedstat_ns(sched_buf, &mut out_ns, &mut out_pc);
    let ccd_start = (cur % 32) & !7;
    let ccd_end = ccd_start + 8;
    let mut ccd_other_ns = 0u64;
    for cid in 0..64usize {
        if cid == cur || cid == sib {
            continue;
        }
        let in_ccd =
            (cid >= ccd_start && cid < ccd_end) || (cid >= ccd_start + 32 && cid < ccd_end + 32);
        if in_ccd {
            ccd_other_ns = ccd_other_ns.saturating_add(out_ns[cid]);
        }
    }
    let cur_ns = if cur < 64 { out_ns[cur] } else { 0 };
    let sib_ns = if sib < 64 { out_ns[sib] } else { 0 };
    let sib_pc = if sib < 64 { out_pc[sib] } else { 0 };
    (cur_ns, sib_ns, sib_pc, ccd_other_ns)
}

fn read_all_schedstat_snapshot(out_ns: &mut [u64; 64]) -> (usize, usize, bool, bool) {
    let Ok(mut guard) = SELF_PIN_TRACKER.lock() else {
        return (23, 64, false, false);
    };
    let mut sched_buf = [0u8; 32768];
    let sched_len = read_file_into_stack_buf(&mut guard.schedstat_file, &mut sched_buf);
    let mut out_pc = [0u64; 64];
    if sched_len > 0 {
        parse_all_schedstat_ns(&sched_buf[..sched_len], out_ns, &mut out_pc);
    }
    (
        guard.primary_core,
        guard.logical_cores,
        guard.allow_repin,
        guard.user_pinned,
    )
}

fn choose_quietest_physical_core(
    delta_ns: &[u64; 64],
    primary_core: usize,
    logical_cores: usize,
) -> usize {
    let phys = (logical_cores / 2).clamp(1, 32);
    if phys <= 1 {
        return 0;
    }
    let pair_busy = |p: usize| -> u64 {
        let sib = if p + phys < 64 { p + phys } else { p };
        delta_ns[p].saturating_add(delta_ns[sib])
    };
    let ccd_busy = |ccd: usize| -> u64 {
        let start = (ccd * 8).min(phys);
        let end = ((ccd + 1) * 8).min(phys);
        let mut sum = 0u64;
        for q in start..end {
            sum = sum.saturating_add(pair_busy(q));
        }
        sum
    };
    let prim = primary_core.min(phys - 1).max(1);
    if pair_busy(prim) <= 15_000 && ccd_busy(prim / 8) <= 800_000 {
        return prim;
    }
    let ccd_pref = |ccd: usize| -> usize {
        match ccd {
            2 => 0,
            1 => 1,
            3 => 2,
            _ => 3,
        }
    };
    let mut best_p = prim;
    let mut best_key = (u64::MAX, u64::MAX, usize::MAX, usize::MAX);
    for p in 1..phys {
        let key = (
            pair_busy(p) / 15_000,
            ccd_busy(p / 8) / 250_000,
            ccd_pref(p / 8),
            usize::from(p != prim),
        );
        if key < best_key {
            best_key = key;
            best_p = p;
        }
    }
    best_p
}

fn apply_self_core_selection(best_core: usize) {
    if let Ok(mut guard) = SELF_PIN_TRACKER.lock() {
        let phys = (guard.logical_cores / 2).max(1);
        let sib = if best_core + phys < guard.logical_cores {
            best_core + phys
        } else {
            best_core
        };
        guard.current_core = best_core;
        guard.sibling_core = sib;
    }
    pin_current_thread_to_core(best_core);
}

fn consume_wait_budget_ms(step_ms: u64) -> bool {
    loop {
        let cur = HOST_WAIT_BUDGET_MS.load(Ordering::Relaxed);
        if cur == 0 {
            return false;
        }
        let next = cur.saturating_sub(step_ms);
        if HOST_WAIT_BUDGET_MS
            .compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return true;
        }
    }
}

/// Waits briefly if a concurrent all-core workload (such as a `rustmq` build where
/// `/proc/loadavg` `nr_running > 7` or `> 6` other CPU cores are active) is saturating
/// socket DRAM bandwidth, and pins the calling thread (`pid = 0` only) to the quietest
/// physical core pair on the quietest L3 CCD (defaulting to `primary_core = 23` when
/// uncontended). Never touches any other process.
fn wait_for_quiet_host_and_repin(max_wait_ms: u64, force_sample_repin: bool) {
    let nr0 = read_nr_running();
    if nr0 <= 6 && !force_sample_repin {
        pin_current_thread_to_core(current_self_pinned_core());
        return;
    }
    let mut waited_ms = 0u64;
    let mut last_delta = [0u64; 64];
    let (mut prim, mut lcores, mut allow_repin, mut user_pinned) = (23usize, 64usize, true, false);
    let mut sampled = false;
    loop {
        let step_ms = if read_nr_running() > 6 && waited_ms < max_wait_ms {
            40u64
        } else {
            8u64
        };
        if step_ms >= 40 && !consume_wait_budget_ms(step_ms) {
            break;
        }
        let mut ns0 = [0u64; 64];
        (prim, lcores, allow_repin, user_pinned) = read_all_schedstat_snapshot(&mut ns0);
        let t0 = Instant::now();
        thread::sleep(Duration::from_millis(step_ms));
        let wall_ns = t0.elapsed().as_nanos() as u64;
        let mut ns1 = [0u64; 64];
        let _ = read_all_schedstat_snapshot(&mut ns1);
        let mut total_other_ns = 0u64;
        for c in 0..lcores.min(64) {
            last_delta[c] = ns1[c].saturating_sub(ns0[c]);
            total_other_ns = total_other_ns.saturating_add(last_delta[c]);
        }
        sampled = true;
        waited_ms = waited_ms.saturating_add(step_ms);
        let active_other_cpus = (total_other_ns as f64) / (wall_ns.max(1) as f64);
        let nr = read_nr_running();
        if (nr <= 6 && active_other_cpus <= 5.5) || waited_ms >= max_wait_ms {
            break;
        }
    }
    if allow_repin && !user_pinned && sampled {
        let best = choose_quietest_physical_core(&last_delta, prim, lcores);
        apply_self_core_selection(best);
    } else {
        pin_current_thread_to_core(current_self_pinned_core());
    }
}

/// Reads `(rq_cpu_time_ns, pcount, busy_jiffies, ccd_other_sched_ns)` for the SMT sibling
/// hardware thread (`sibling_core = current_core + 32`, e.g., `cpu55` for `cpu23`) and
/// the other 14 logical cores on the same L3 CCD from pre-opened `/proc/schedstat` and
/// `/proc/stat` using stack buffers only (zero heap allocations).
fn read_current_sibling_activity() -> (u64, u64, u64, u64) {
    let Ok(mut guard) = SELF_PIN_TRACKER.lock() else {
        return (0, 0, 0, 0);
    };
    let cur = guard.current_core;
    let sib = guard.sibling_core;
    let mut sched_buf = [0u8; 32768];
    let sched_len = read_file_into_stack_buf(&mut guard.schedstat_file, &mut sched_buf);
    let mut stat_buf = [0u8; 8192];
    let stat_len = read_file_into_stack_buf(&mut guard.procstat_file, &mut stat_buf);
    let mut tokens = [0u64; 10];
    let (_cur_ns, sched_ns, pcount, ccd_other_ns) = if sched_len > 0 {
        parse_ccd_schedstat(&sched_buf[..sched_len], cur, sib)
    } else {
        (0, 0, 0, 0)
    };
    let busy_jiffies =
        if stat_len > 0 && parse_cpu_line_tokens(&stat_buf[..stat_len], sib, &mut tokens) >= 8 {
            tokens[0]
                .saturating_add(tokens[1])
                .saturating_add(tokens[2])
                .saturating_add(tokens[5])
                .saturating_add(tokens[6])
                .saturating_add(tokens[7])
        } else {
            0
        };
    (sched_ns, pcount, busy_jiffies, ccd_other_ns)
}

/// Reads `(cur_sched_ns, sib_sched_ns, sib_pcount, sib_busy_jiffies, ccd_other_sched_ns)`
/// for `current_core`, `sibling_core`, and the other 14 logical cores on the same L3 CCD.
fn read_current_and_sibling_raw() -> (u64, u64, u64, u64, u64) {
    let Ok(mut guard) = SELF_PIN_TRACKER.lock() else {
        return (0, 0, 0, 0, 0);
    };
    let cur = guard.current_core;
    let sib = guard.sibling_core;
    let mut sched_buf = [0u8; 32768];
    let sched_len = read_file_into_stack_buf(&mut guard.schedstat_file, &mut sched_buf);
    let mut stat_buf = [0u8; 8192];
    let stat_len = read_file_into_stack_buf(&mut guard.procstat_file, &mut stat_buf);
    let mut tokens = [0u64; 10];
    let (cur_ns, sib_ns, sib_pc, ccd_other_ns) = if sched_len > 0 {
        parse_ccd_schedstat(&sched_buf[..sched_len], cur, sib)
    } else {
        (0, 0, 0, 0)
    };
    let sib_j =
        if stat_len > 0 && parse_cpu_line_tokens(&stat_buf[..stat_len], sib, &mut tokens) >= 8 {
            tokens[0]
                .saturating_add(tokens[1])
                .saturating_add(tokens[2])
                .saturating_add(tokens[5])
                .saturating_add(tokens[6])
                .saturating_add(tokens[7])
        } else {
            0
        };
    (cur_ns, sib_ns, sib_pc, sib_j, ccd_other_ns)
}

/// Spins briefly on `current_core` in `500 us` (`0.5 ms`) windows when transient
/// primary-core preemption, SMT-sibling contention, or CCD contention is detected;
/// if contention persists beyond 4 windows (`2 ms`) or the host is under heavy load,
/// waits out the burst and hops `self` (`pid = 0` only) to the quietest physical core
/// pair on the quietest CCD. Never touches any other process.
fn wait_for_quiet_sibling() {
    if read_nr_running() > 7 {
        let max_ms = if has_active_filter() { 4_000 } else { 1_200 };
        wait_for_quiet_host_and_repin(max_ms, true);
        return;
    }
    for iter in 0..16 {
        let (_cur_ns0, sib_ns0, _sib_pc0, sib_j0, ccd_ns0) = read_current_and_sibling_raw();
        let cpu0 = current_thread_cpu_ns();
        let spin_start = Instant::now();
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
        if stolen < 15_000 && sib_d_ns < 15_000 && sib_d_j == 0 && ccd_d_ns < 250_000 {
            return;
        }
        if iter == 3 && is_self_repin_allowed() {
            let max_ms = if has_active_filter() { 3_000 } else { 800 };
            wait_for_quiet_host_and_repin(max_ms, true);
            return;
        }
    }
}

fn init_self_pin_tracker(primary_core: usize, logical_cores: usize) {
    let phys_cores = (logical_cores / 2).max(1);
    let sibling_core = if primary_core + phys_cores < logical_cores {
        primary_core + phys_cores
    } else {
        primary_core
    };
    let user_pinned = explicit_pin_core_override(logical_cores).is_some();
    let schedstat_file = fs::File::open("/proc/schedstat").ok();
    let procstat_file = fs::File::open("/proc/stat").ok();
    let loadavg_file = fs::File::open("/proc/loadavg").ok();
    if let Ok(mut guard) = SELF_PIN_TRACKER.lock() {
        guard.primary_core = primary_core;
        guard.current_core = primary_core;
        guard.sibling_core = sibling_core;
        guard.logical_cores = logical_cores;
        guard.allow_repin = true;
        guard.user_pinned = user_pinned;
        guard.schedstat_file = schedstat_file;
        guard.procstat_file = procstat_file;
        guard.loadavg_file = loadavg_file;
    }
}

fn set_allow_self_repin(allow: bool) {
    if let Ok(mut guard) = SELF_PIN_TRACKER.lock() {
        guard.allow_repin = allow;
    }
}

fn is_self_repin_allowed() -> bool {
    SELF_PIN_TRACKER
        .lock()
        .map(|g| g.allow_repin)
        .unwrap_or(false)
}

fn current_self_pinned_core() -> usize {
    SELF_PIN_TRACKER
        .lock()
        .map(|g| g.current_core)
        .unwrap_or(23)
}

/// Ensures the calling benchmark thread (`pid = 0` only; never modifies any other
/// process) is pinned to `primary_core` (`23` on CCD 2) when uncontended, or dynamically
/// selects the quietest physical core pair on the quietest CCD if `primary_core` / CCD 2
/// or the host is contended.
fn maybe_repin_self_within_ccd() {
    let max_ms = if has_active_filter() { 6_000 } else { 2_500 };
    wait_for_quiet_host_and_repin(max_ms, false);
}

#[derive(Debug, Clone, Copy)]
struct SuiteSizes {
    rank_select_bits: usize,
    rank_select_queries: usize,
    elias_fano_elements: usize,
    trie_keys: usize,
    hash_map_keys: usize,
    memory_pool_ops: usize,
    entropy_block_bytes: usize,
    int_codec_elements: usize,
    radix_sort_elements: usize,
    suffix_array_bytes: usize,
    suffix_array_queries: usize,
    bm25_docs: usize,
}

impl SuiteSizes {
    fn spec() -> Self {
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

    fn smoke() -> Self {
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

    fn to_map(self) -> BTreeMap<String, u64> {
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
struct HostMetadata {
    host: String,
    arch: String,
    cpu_model: String,
    cpu_mhz: String,
    cache_size: String,
    logical_cores: usize,
    pinned_core: usize,
    cpu_governor: String,
    rustflags: String,
    runner_command: String,
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
            warmup_calls: 2,
            trials: 11,
            bursts_per_trial: 6,
            max_regression_pct: 5.0,
        }
    }

    const fn medium(dataset_size: u64, ops_per_iter: u64) -> Self {
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
    const fn heavy(dataset_size: u64, ops_per_iter: u64, trials: usize) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 1,
            trials,
            bursts_per_trial: 1,
            max_regression_pct: 9.0,
        }
    }

    const fn with_threshold(mut self, max_regression_pct: f64) -> Self {
        self.max_regression_pct = max_regression_pct;
        self
    }
}

fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn record_space_metric(space: &mut BTreeMap<String, f64>, key: &str, val: f64) {
    space.insert(key.to_string(), val);
}

fn record_samples(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    params: BenchParams,
    samples: &mut [f64],
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
    let rounded_samples: Vec<f64> = samples.iter().copied().map(round4).collect();

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
fn measure_bench_with_setup<T, S, F>(
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
fn measure_bench_inner<T, S, F, const HAS_SETUP: bool>(
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
        if params.trials <= 4 {
            CURRENT_BURST_IDX.store(0, Ordering::Relaxed);
            setup(state);
            f(state);
        }
        return;
    }

    let filtered_mode = has_active_filter();
    let check_preempt = true;
    let check_sibling = true;
    let allow_repin = is_self_repin_allowed();
    if allow_repin {
        if filtered_mode {
            wait_for_quiet_host_and_repin(8_000, true);
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
    let uses_burst_idx = CURRENT_BURST_IDX.load(Ordering::Relaxed) > 0;
    let max_bench_attempts =
        if check_sibling && allow_repin && params.bursts_per_trial >= 2 && !uses_burst_idx {
            5usize
        } else {
            1usize
        };
    let retry_mult = if filtered_mode { 2usize } else { 1usize };
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
            if filtered_mode && read_nr_running() > 6 {
                wait_for_quiet_host_and_repin(2_000, allow_repin);
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
                while b < params.bursts_per_trial {
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
                        let ccd_contended = allow_repin
                            && wall_ns >= 2_000_000
                            && (b_ccd_busy.saturating_mul(2) >= wall_ns
                                || (wall_ns >= 20_000_000 && read_nr_running() > 7));
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
                        && ((sib_busy_ns >= 150_000
                            && sib_busy_ns.saturating_mul(16) >= trial_wall_ns)
                            || (sib_busy_ns >= 8_000_000
                                && sib_busy_ns.saturating_mul(25) >= trial_wall_ns)
                            || (trial_wall_ns >= 2_000_000
                                && ccd_busy_ns.saturating_mul(2) >= trial_wall_ns))
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

    record_samples(
        records,
        category,
        name,
        unit,
        params,
        &mut samples[..params.trials],
    );
}

// ---------------------------------------------------------------------------
// 1. Rank / Select (1 Gbit = 1,000,000,000 bits, 1,000,000 uniform queries)
// ---------------------------------------------------------------------------

#[inline(always)]
fn splitmix64_at(seed: u64, idx: usize) -> u64 {
    let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
fn uniform_index_at(seed: u64, idx: usize, range: usize) -> usize {
    ((splitmix64_at(seed, idx) as u128 * range as u128) >> 64) as usize
}

/// Exact 2-round Feistel-like bijection over `0..n` (when `n` is a multiple of `1,000`)
/// backed by L1-resident permutation tables (`<= 16 KiB` total).
///
/// Visits all `n` (`1,000,000` at spec size) keys in the structure in a uniform
/// pseudo-random order with zero replayed samples, while stratifying every `m`-key
/// sub-slice across all `m` (`1,000`) outer buckets `q2 in 0..m`.
struct BijectivePerm {
    p1: [u32; 1000],
    p2: Vec<u32>,
    p3: [u32; 1000],
    p4: Vec<u32>,
}

impl BijectivePerm {
    fn new(n: usize, seed: u64) -> Self {
        assert!(
            n.is_multiple_of(1000) && n > 0,
            "n must be a positive multiple of 1000"
        );
        let m = n / 1000;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p1 = [0u32; 1000];
        for (i, slot) in p1.iter_mut().enumerate() {
            *slot = i as u32;
        }
        p1.shuffle(&mut rng);

        let mut p2: Vec<u32> = (0..m as u32).collect();
        p2.shuffle(&mut rng);

        let mut p3 = [0u32; 1000];
        for slot in &mut p3 {
            *slot = rng.random_range(0..m as u32);
        }

        let mut p4: Vec<u32> = (0..m).map(|_| rng.random_range(0..1000u32)).collect();
        p4.shuffle(&mut rng);

        Self { p1, p2, p3, p4 }
    }

    #[inline(always)]
    fn for_each_slice<F: FnMut(usize)>(&self, start: usize, count: usize, mut f: F) {
        let m = self.p2.len();
        let r_steps = (count / m).max(1);
        let mut r = (start / m) % 1000;
        for _ in 0..r_steps {
            let mut q_idx = self.p3[r] as usize;
            for _ in 0..m {
                let q2 = self.p2[q_idx] as usize;
                let mut r_idx = r + self.p4[q2] as usize;
                if r_idx >= 1000 {
                    r_idx -= 1000;
                }
                let r2 = self.p1[r_idx] as usize;
                if m == 1000 {
                    f(r2 * 1000 + q2);
                } else {
                    f(q2 * 1000 + r2);
                }
                q_idx += 1;
                if q_idx == m {
                    q_idx = 0;
                }
            }
            r += 1;
            if r == 1000 {
                r = 0;
            }
        }
    }
}

fn gen_rank_words(num_words: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
    (0..num_words).map(|_| rng.random::<u64>()).collect()
}

fn warm_rank_pages<R: RankSelectOps>(rs: &R, num_bits: usize) {
    let mut acc = 0usize;
    for bit in (0..num_bits).step_by(4096) {
        acc = acc.wrapping_add(rs.rank1(bit));
    }
    black_box(acc);
}

fn warm_select1_pages<R: RankSelectOps>(rs: &R, max_ones: usize) {
    let mut acc = 0usize;
    for k in (0..max_ones).step_by(4096) {
        acc = acc.wrapping_add(rs.select1(k).unwrap_or(0));
    }
    black_box(acc);
}

fn warm_select0_pages<R: RankSelectOps>(rs: &R, max_zeros: usize) {
    let mut acc = 0usize;
    for k in (0..max_zeros).step_by(4096) {
        acc = acc.wrapping_add(rs.select0(k).unwrap_or(0));
    }
    black_box(acc);
}

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

const TRIE_RAND_SEED: u64 = SEED_BASE ^ 0x05;

#[inline(always)]
fn make_url_path_key(id: usize) -> [u8; 42] {
    // Exact byte representation of:
    // format!("org/infinilabs/mod_{:02}/sub_{:03}/item_{:07}", id % 32, (id / 32) % 256, id)
    let mut out = *b"org/infinilabs/mod_00/sub_000/item_0000000";
    let m = (id & 31) as u32;
    out[19] = b'0' + (m / 10) as u8;
    out[20] = b'0' + (m % 10) as u8;
    let s = ((id >> 5) & 255) as u32;
    out[26] = b'0' + (s / 100) as u8;
    out[27] = b'0' + ((s / 10) % 10) as u8;
    out[28] = b'0' + (s % 10) as u8;
    let mut v = id as u32;
    out[41] = b'0' + (v % 10) as u8;
    v /= 10;
    out[40] = b'0' + (v % 10) as u8;
    v /= 10;
    out[39] = b'0' + (v % 10) as u8;
    v /= 10;
    out[38] = b'0' + (v % 10) as u8;
    v /= 10;
    out[37] = b'0' + (v % 10) as u8;
    v /= 10;
    out[36] = b'0' + (v % 10) as u8;
    v /= 10;
    out[35] = b'0' + (v % 10) as u8;
    out
}

#[inline(always)]
fn make_random_16b_key(id: usize) -> [u8; 16] {
    // 10 deterministic pseudo-random alphanumeric bytes + 6-hex-digit unique index suffix
    const ALPHABET: &[u8; 36] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut z0 = TRIE_RAND_SEED ^ (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z0 = (z0 ^ (z0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z0 ^= z0 >> 27;
    let mut z1 = z0.wrapping_add(0x94D0_49BB_1331_11EB);
    z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z1 ^= z1 >> 27;

    let mut key = [0u8; 16];
    for i in 0..5 {
        key[i] = ALPHABET[((z0 >> (i * 12)) as usize) % 36];
        key[i + 5] = ALPHABET[((z1 >> (i * 12)) as usize) % 36];
    }
    let v = id as u32;
    key[10] = HEX[((v >> 20) & 0xf) as usize];
    key[11] = HEX[((v >> 16) & 0xf) as usize];
    key[12] = HEX[((v >> 12) & 0xf) as usize];
    key[13] = HEX[((v >> 8) & 0xf) as usize];
    key[14] = HEX[((v >> 4) & 0xf) as usize];
    key[15] = HEX[(v & 0xf) as usize];
    key
}

#[inline(always)]
fn make_shared_prefix_key(id: usize) -> [u8; 32] {
    // Exact byte representation of: format!("common_prefix_header_v1_{:08x}", id)
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = *b"common_prefix_header_v1_00000000";
    let v = id as u32;
    key[24] = HEX[((v >> 28) & 0xf) as usize];
    key[25] = HEX[((v >> 24) & 0xf) as usize];
    key[26] = HEX[((v >> 20) & 0xf) as usize];
    key[27] = HEX[((v >> 16) & 0xf) as usize];
    key[28] = HEX[((v >> 12) & 0xf) as usize];
    key[29] = HEX[((v >> 8) & 0xf) as usize];
    key[30] = HEX[((v >> 4) & 0xf) as usize];
    key[31] = HEX[(v & 0xf) as usize];
    key
}

struct PackedKeySet {
    data: Vec<u8>,
    key_len: usize,
}

impl PackedKeySet {
    fn from_sorted_fixed<const N: usize>(keys: &[[u8; N]]) -> Self {
        let mut data = Vec::with_capacity(keys.len() * N);
        for k in keys {
            data.extend_from_slice(k);
        }
        Self { data, key_len: N }
    }

    fn slices(&self) -> Vec<&[u8]> {
        self.data.chunks_exact(self.key_len).collect()
    }
}

fn gen_trie_distribution(dist_idx: usize, n: usize) -> (&'static str, PackedKeySet) {
    match dist_idx {
        0 => {
            let mut paths: Vec<[u8; 42]> = (0..n).map(make_url_path_key).collect();
            paths.sort_unstable();
            paths.dedup();
            ("url_paths", PackedKeySet::from_sorted_fixed(&paths))
        }
        1 => {
            let mut random_16b: Vec<[u8; 16]> = (0..n).map(make_random_16b_key).collect();
            random_16b.sort_unstable();
            random_16b.dedup();
            ("random_16b", PackedKeySet::from_sorted_fixed(&random_16b))
        }
        _ => {
            let mut shared_prefix: Vec<[u8; 32]> = (0..n).map(make_shared_prefix_key).collect();
            shared_prefix.sort_unstable();
            shared_prefix.dedup();
            (
                "shared_prefix",
                PackedKeySet::from_sorted_fixed(&shared_prefix),
            )
        }
    }
}

#[repr(C, align(64))]
struct AlignedKey<const N: usize>([u8; N]);

#[inline(always)]
fn contains_trie_key_by_id<F: FnMut(&[u8]) -> bool>(
    dist_idx: usize,
    id: usize,
    mut contains: F,
) -> bool {
    match dist_idx {
        0 => {
            let k = AlignedKey(make_url_path_key(id));
            contains(black_box(&k.0[..]))
        }
        1 => {
            let k = AlignedKey(make_random_16b_key(id));
            contains(black_box(&k.0[..]))
        }
        _ => {
            let k = AlignedKey(make_shared_prefix_key(id));
            contains(black_box(&k.0[..]))
        }
    }
}

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
        dat_build_params.warmup_calls = usize::from(dist_idx == 2);
        dat_build_params.bursts_per_trial = 1;
        let mut cspp_build_params = BenchParams::heavy(num_keys, num_keys, 3);
        cspp_build_params.warmup_calls = 0;
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
// ---------------------------------------------------------------------------

#[inline(always)]
fn make_u64_key(id: usize) -> u64 {
    splitmix64_at(SEED_BASE ^ 0x06, id)
}

#[inline(always)]
fn make_str16_key(id: usize) -> [u8; 16] {
    let lo = splitmix64_at(SEED_BASE ^ 0x0610, id) | 0x2020_2020_2020_2020;
    let hi = (id as u64) ^ 0x6b65_795f_0000_0000;
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&lo.to_le_bytes());
    buf[8..].copy_from_slice(&hi.to_le_bytes());
    buf
}

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
        let mut template_u32 = vec![0u32; n];
        let mut s = SEED_BASE ^ 0x0901;
        for slot in template_u32.iter_mut() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            *slot = (z ^ (z >> 29)) as u32;
        }
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
        let mut template_u64 = vec![0u64; n];
        let mut s = SEED_BASE ^ 0x0902;
        for slot in template_u64.iter_mut() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            *slot = z ^ (z >> 27);
        }
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
    let mut template_kv = vec![(0u64, 0u64); n];
    {
        let mut s = SEED_BASE ^ 0x0903;
        for (i, slot) in template_kv.iter_mut().enumerate() {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            *slot = (z ^ (z >> 27), i as u64);
        }
    }
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
    let sa0 = sa_pending.take().unwrap();
    if should_run_bench("suffix_array/search_query") {
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

#[repr(C, align(4096))]
struct AlignedHaystacks([[u8; 65_536]; 4]);

#[repr(C, align(64))]
struct AlignedNeedle([u8; 16]);

static SEARCH_HAYSTACKS: AlignedHaystacks = {
    let mut buf = [b'a'; 65_536];
    let needle = *b"needle_target";
    let start = 65_536 - 32;
    let mut i = 0;
    while i < 13 {
        buf[start + i] = needle[i];
        i += 1;
    }
    buf[65_536 - 5] = b'Z';
    AlignedHaystacks([buf, buf, buf, buf])
};

static SEARCH_NEEDLE: AlignedNeedle = AlignedNeedle(*b"needle_target\0\0\0");

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
        bench_search_and_scoring(&mut search_records, sizes);
        maybe_repin_self_within_ccd();
    }
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
        bench_hash_maps(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("memory_pools") {
        bench_memory_pools(&mut records, logical_cores, sizes);
        set_allow_self_repin(true);
        pin_current_thread_to_core(current_self_pinned_core());
        maybe_repin_self_within_ccd();
    }
    if run_cat("entropy") {
        bench_entropy(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("int_codecs") {
        bench_int_codecs(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    if run_cat("radix_sort") {
        if should_run_any(&["radix_sort/lsd_u32", "radix_sort/lsd_u64"]) {
            bench_radix_sort_lsd(&mut records, sizes);
            maybe_repin_self_within_ccd();
        }
        if should_run_bench("radix_sort/kv_u64_u64") {
            bench_radix_sort_kv(&mut records, current_self_pinned_core(), sizes);
            maybe_repin_self_within_ccd();
        }
    }
    if run_cat("suffix_array") {
        bench_suffix_array(&mut records, sizes);
        maybe_repin_self_within_ccd();
    }
    records.append(&mut search_records);

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
