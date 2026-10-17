//! Shared dataset generators, seeds, sizes, bijective uniform query permutations,
//! self-only CPU affinity/repinning, and timing harness for `core_bench` (E1) and
//! `competitor_bench` (E2).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::io::{Read, Seek, SeekFrom};
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use zipora::succinct::rank_select::RankSelectOps;

pub const SEED_BASE: u64 = 0xE1C0_5EBE_4C00_2026;
pub const ENTROPY_BLOCK_SOURCE: &str = "synthetic_enwik8_like_64kib (deterministic Zipf-skewed English/Wiki markup token stream, seed = 0xE1C0_5EBE_4C00_2021)";

#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_core(core_id: usize) {
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
pub fn pin_current_thread_to_core(_core_id: usize) {}

/// Disables Transparent Huge Pages (`PR_SET_THP_DISABLE`) for the current benchmark
/// process only (`self`), unless `--thp-on` or `ZIPORA_BENCH_THP=on` (`1`) is set,
/// while keeping 100% default production glibc `malloc` settings.
///
/// Disclosure & Justification (`S38-R2` / `S39-R2` / `S40-R2`): `bindiego` is a KVM virtual machine
/// where `/sys/kernel/mm/transparent_hugepage/enabled` is `[always]`, `defrag` is `[madvise]`,
/// and `/proc/buddyinfo` (`Node 0, zone Normal`) has fluctuating free `order 9` (`2 MiB`) and
/// `order 10` (`4 MiB`) blocks (ranging from `10` free `order 9` blocks = `20 MiB` under heavy
/// build fragmentation to `374-426` free `order 9` blocks = `820-924 MiB` when unfragmented).
/// Without `PR_SET_THP_DISABLE`, 16-400+ MiB structures (`Rank9`, `ZiporaTrie`, `GoldHashMap`,
/// `ZiporaHashMap`, `KeyValueRadixSort`, `SuffixArray`) non-deterministically receive variable
/// fractions of `2 MiB` hugepages across runs: when unfragmented (`374-426` free `2 MiB` blocks),
/// `--thp-on` is measured at `20-24%` faster on hash-map lookups (`20-26%` on inserts), `10-26%`
/// faster on trie lookups (`26-44%` on DAT `build_sorted`), `12-23%` faster on `1 Gbit` rank/select,
/// `28-34%` faster on `10M` radix sort, and `26-28%` faster on `10M` suffix array construct/search,
/// whereas when `order 9` free blocks drop to `~10` (`20 MiB`), multi-hundred-MiB allocations fall
/// back to `4 KiB` pages (`~0%` gain; see `docs/review/e1.md` §4.3 for the 3-run min/median and
/// best-case `--thp-on` vs default `THP-off` comparison table and `/proc/buddyinfo` counts).
/// Setting `PR_SET_THP_DISABLE` by default ensures deterministic `4 KiB` page backing for regression
/// gating; pass `--thp-on` (or `ZIPORA_BENCH_THP=on`) to measure with OS transparent hugepages enabled.
#[cfg(target_os = "linux")]
pub fn disable_transparent_hugepages_for_self() {
    // SAFETY: `prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)` affects only the calling process.
    unsafe {
        let _ = libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn disable_transparent_hugepages_for_self() {}

pub fn parse_cpu_list(spec: &str) -> Vec<usize> {
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

pub fn hardware_logical_cores() -> usize {
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

pub fn explicit_pin_core_override(logical_cores: usize) -> Option<usize> {
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
pub fn select_primary_core(logical_cores: usize) -> usize {
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

pub static BENCH_FILTER: OnceLock<Option<Vec<String>>> = OnceLock::new();
pub static HOST_WAIT_BUDGET_MS: AtomicU64 = AtomicU64::new(20_000);

pub fn init_bench_filter(filter_spec: Option<String>) {
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

pub fn has_active_filter() -> bool {
    BENCH_FILTER
        .get()
        .and_then(Option::as_ref)
        .is_some_and(|v| !v.is_empty())
}

pub fn should_run_bench(name: &str) -> bool {
    let Some(Some(filters)) = BENCH_FILTER.get() else {
        return true;
    };
    filters
        .iter()
        .any(|f| name == f.as_str() || name.contains(f.as_str()))
}

pub fn should_run_any(names: &[&str]) -> bool {
    names.iter().any(|n| should_run_bench(n))
}

pub fn should_run_prefix(prefix: &str) -> bool {
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

pub fn should_run_category(cat: &str, cat_filter: Option<&str>) -> bool {
    if let Some(cf) = cat_filter
        && !cf.is_empty()
        && !cf.split(',').any(|p| p.trim() == cat)
    {
        return false;
    }
    let cat_prefix = format!("{cat}/");
    should_run_prefix(&cat_prefix)
}

pub struct SelfPinTracker {
    pub primary_core: usize,
    pub current_core: usize,
    pub sibling_core: usize,
    pub logical_cores: usize,
    pub allow_repin: bool,
    pub user_pinned: bool,
    pub schedstat_file: Option<fs::File>,
    pub procstat_file: Option<fs::File>,
    pub loadavg_file: Option<fs::File>,
}

pub static SELF_PIN_TRACKER: Mutex<SelfPinTracker> = Mutex::new(SelfPinTracker {
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

pub static CURRENT_BURST_IDX: AtomicUsize = AtomicUsize::new(0);

#[inline(always)]
pub fn current_burst_index() -> usize {
    CURRENT_BURST_IDX.fetch_add(1, Ordering::Relaxed)
}

#[cfg(target_os = "linux")]
pub fn current_thread_cpu_ns() -> u64 {
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
pub fn current_thread_cpu_ns() -> u64 {
    0
}

pub fn read_file_into_stack_buf(file_opt: &mut Option<fs::File>, buf: &mut [u8]) -> usize {
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

pub fn parse_loadavg_nr_running(buf: &[u8]) -> usize {
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

pub fn read_nr_running() -> usize {
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

pub fn format_cpu_prefix(cpu_id: usize, out: &mut [u8; 12]) -> usize {
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

pub fn parse_cpu_line_tokens(buf: &[u8], cpu_id: usize, tokens: &mut [u64; 10]) -> usize {
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

pub fn parse_all_schedstat_ns(sched_buf: &[u8], out_ns: &mut [u64; 64], out_pc: &mut [u64; 64]) {
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

pub fn parse_ccd_schedstat(sched_buf: &[u8], cur: usize, sib: usize) -> (u64, u64, u64, u64) {
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

pub fn read_all_schedstat_snapshot(out_ns: &mut [u64; 64]) -> (usize, usize, bool, bool) {
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

pub fn choose_quietest_physical_core(
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

pub fn apply_self_core_selection(best_core: usize) {
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

pub fn consume_wait_budget_ms(step_ms: u64) -> bool {
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
pub fn wait_for_quiet_host_and_repin(max_wait_ms: u64, force_sample_repin: bool) {
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
pub fn read_current_sibling_activity() -> (u64, u64, u64, u64) {
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
pub fn read_current_and_sibling_raw() -> (u64, u64, u64, u64, u64) {
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
pub fn wait_for_quiet_sibling() {
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

pub fn init_self_pin_tracker(primary_core: usize, logical_cores: usize) {
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

pub fn set_allow_self_repin(allow: bool) {
    if let Ok(mut guard) = SELF_PIN_TRACKER.lock() {
        guard.allow_repin = allow;
    }
}

pub fn is_self_repin_allowed() -> bool {
    SELF_PIN_TRACKER
        .lock()
        .map(|g| g.allow_repin)
        .unwrap_or(false)
}

pub fn current_self_pinned_core() -> usize {
    SELF_PIN_TRACKER
        .lock()
        .map(|g| g.current_core)
        .unwrap_or(23)
}

/// Ensures the calling benchmark thread (`pid = 0` only; never modifies any other
/// process) is pinned to `primary_core` (`23` on CCD 2) when uncontended, or dynamically
/// selects the quietest physical core pair on the quietest CCD if `primary_core` / CCD 2
/// or the host is contended.
pub fn maybe_repin_self_within_ccd() {
    let max_ms = if has_active_filter() { 6_000 } else { 2_500 };
    wait_for_quiet_host_and_repin(max_ms, false);
}

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

pub fn sha256_bytes(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{word:08x}");
    }
    out
}

pub fn detect_self_binary_path_and_sha256() -> (String, String) {
    let Ok(exe) = std::env::current_exe() else {
        return (String::new(), String::new());
    };
    let canonical = fs::canonicalize(&exe).unwrap_or(exe);
    let path_str = canonical.to_string_lossy().into_owned();
    let sha = fs::read(&canonical)
        .map(|bytes| sha256_bytes(&bytes))
        .unwrap_or_default();
    (path_str, sha)
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
    space.insert(key.to_string(), val);
}

pub fn record_samples(
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
pub fn splitmix64_at(seed: u64, idx: usize) -> u64 {
    let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline(always)]
pub fn uniform_index_at(seed: u64, idx: usize, range: usize) -> usize {
    ((splitmix64_at(seed, idx) as u128 * range as u128) >> 64) as usize
}

/// Exact 2-round Feistel-like bijection over `0..n` (when `n` is a multiple of `1,000`)
/// backed by L1-resident permutation tables (`<= 16 KiB` total).
///
/// Visits all `n` (`1,000,000` at spec size) keys in the structure in a uniform
/// pseudo-random order with zero replayed samples, while stratifying every `m`-key
/// sub-slice across all `m` (`1,000`) outer buckets `q2 in 0..m`.
pub struct BijectivePerm {
    pub p1: [u32; 1000],
    pub p2: Vec<u32>,
    pub p3: [u32; 1000],
    pub p4: Vec<u32>,
}

impl BijectivePerm {
    pub fn new(n: usize, seed: u64) -> Self {
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
    pub fn for_each_slice<F: FnMut(usize)>(&self, start: usize, count: usize, mut f: F) {
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

pub fn gen_rank_words(num_words: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
    (0..num_words).map(|_| rng.random::<u64>()).collect()
}

pub fn warm_rank_pages<R: RankSelectOps>(rs: &R, num_bits: usize) {
    let mut acc = 0usize;
    for bit in (0..num_bits).step_by(4096) {
        acc = acc.wrapping_add(rs.rank1(bit));
    }
    black_box(acc);
}

pub fn warm_select1_pages<R: RankSelectOps>(rs: &R, max_ones: usize) {
    let mut acc = 0usize;
    for k in (0..max_ones).step_by(4096) {
        acc = acc.wrapping_add(rs.select1(k).unwrap_or(0));
    }
    black_box(acc);
}

pub fn warm_select0_pages<R: RankSelectOps>(rs: &R, max_zeros: usize) {
    let mut acc = 0usize;
    for k in (0..max_zeros).step_by(4096) {
        acc = acc.wrapping_add(rs.select0(k).unwrap_or(0));
    }
    black_box(acc);
}

pub fn gen_uniform_sparse(n: usize, avg_gap: u32, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    let mut cur: u32 = 0;
    for _ in 0..n {
        cur = cur.saturating_add(rng.random_range(1..=2 * avg_gap));
        out.push(cur);
    }
    out
}

pub fn gen_zipf_clustered(n: usize, seed: u64) -> Vec<u32> {
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

pub const TRIE_RAND_SEED: u64 = SEED_BASE ^ 0x05;

#[inline(always)]
pub fn make_url_path_key(id: usize) -> [u8; 42] {
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
pub fn make_random_16b_key(id: usize) -> [u8; 16] {
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
pub fn make_shared_prefix_key(id: usize) -> [u8; 32] {
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

pub struct PackedKeySet {
    pub data: Vec<u8>,
    pub key_len: usize,
}

impl PackedKeySet {
    pub fn from_sorted_fixed<const N: usize>(keys: &[[u8; N]]) -> Self {
        let mut data = Vec::with_capacity(keys.len() * N);
        for k in keys {
            data.extend_from_slice(k);
        }
        Self { data, key_len: N }
    }

    pub fn slices(&self) -> Vec<&[u8]> {
        self.data.chunks_exact(self.key_len).collect()
    }
}

pub fn gen_trie_distribution(dist_idx: usize, n: usize) -> (&'static str, PackedKeySet) {
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
pub struct AlignedKey<const N: usize>(pub [u8; N]);

#[inline(always)]
pub fn contains_trie_key_by_id<F: FnMut(&[u8]) -> bool>(
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

#[inline(always)]
pub fn make_u64_key(id: usize) -> u64 {
    splitmix64_at(SEED_BASE ^ 0x06, id)
}

#[inline(always)]
pub fn make_str16_key(id: usize) -> [u8; 16] {
    let lo = splitmix64_at(SEED_BASE ^ 0x0610, id) | 0x2020_2020_2020_2020;
    let hi = (id as u64) ^ 0x6b65_795f_0000_0000;
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&lo.to_le_bytes());
    buf[8..].copy_from_slice(&hi.to_le_bytes());
    buf
}

/// Zipf-skewed English words and MediaWiki markup tokens.
pub fn gen_enwik_like_block(size: usize, seed: u64) -> Vec<u8> {
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

#[repr(C, align(4096))]
pub struct AlignedHaystacks(pub [[u8; 65_536]; 4]);

#[repr(C, align(64))]
pub struct AlignedNeedle(pub [u8; 16]);

pub static SEARCH_HAYSTACKS: AlignedHaystacks = {
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

pub static SEARCH_NEEDLE: AlignedNeedle = AlignedNeedle(*b"needle_target\0\0\0");

pub fn gen_radix_u32_template(n: usize) -> Vec<u32> {
    let mut template_u32 = vec![0u32; n];
    let mut s = SEED_BASE ^ 0x0901;
    for slot in template_u32.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = (z ^ (z >> 29)) as u32;
    }
    template_u32
}

pub fn gen_radix_u64_template(n: usize) -> Vec<u64> {
    let mut template_u64 = vec![0u64; n];
    let mut s = SEED_BASE ^ 0x0902;
    for slot in template_u64.iter_mut() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = z ^ (z >> 27);
    }
    template_u64
}

pub fn gen_radix_kv_template(n: usize) -> Vec<(u64, u64)> {
    let mut template_kv = vec![(0u64, 0u64); n];
    let mut s = SEED_BASE ^ 0x0903;
    for (i, slot) in template_kv.iter_mut().enumerate() {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *slot = (z ^ (z >> 27), i as u64);
    }
    template_kv
}
