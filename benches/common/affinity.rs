//! Self-only CPU affinity, THP control, benchmark filter state, and `/proc/schedstat`
//! CCD quietness tracking for `core_bench` and `competitor_bench`.

#![allow(dead_code)]

use std::fs;
use std::hint::black_box;
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

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
    let nr = read_nr_running();
    if pair_busy(prim) <= 15_000 && (nr <= 16 || ccd_busy(prim / 8) <= 800_000) {
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
    let cur_core = current_self_pinned_core().min(phys - 1).max(1);
    let cur_ccd = cur_core / 8;
    let ccd_start = (cur_ccd * 8).max(1).min(phys);
    let ccd_end = ((cur_ccd + 1) * 8).min(phys);
    let mut best_same_ccd_p = cur_core;
    let mut best_same_ccd_busy = u64::MAX;
    for p in ccd_start..ccd_end {
        let pb = pair_busy(p);
        if pb < best_same_ccd_busy || (pb == best_same_ccd_busy && p == prim) {
            best_same_ccd_busy = pb;
            best_same_ccd_p = p;
        }
    }
    if nr <= 16 || (best_same_ccd_busy <= 150_000 && ccd_busy(cur_ccd) <= 4_000_000) {
        return best_same_ccd_p;
    }
    let mut best_p = prim;
    let mut best_key = (u64::MAX, usize::MAX, u64::MAX, usize::MAX, usize::MAX);
    for p in 1..phys {
        let key = (
            pair_busy(p) / 15_000,
            usize::from(p / 8 != cur_ccd && best_same_ccd_busy <= 600_000),
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
    let eff_max_wait_ms = if nr0 > 16 {
        max_wait_ms.min(200)
    } else {
        max_wait_ms
    };
    let mut waited_ms = 0u64;
    let mut last_delta = [0u64; 64];
    let (mut prim, mut lcores, mut allow_repin, mut user_pinned) = (23usize, 64usize, true, false);
    let mut sampled = false;
    loop {
        let mut step_ms =
            if read_nr_running() > 6 && waited_ms < eff_max_wait_ms && eff_max_wait_ms >= 40 {
                40u64
            } else {
                8u64
            };
        if step_ms >= 40 && !consume_wait_budget_ms(step_ms) {
            if sampled {
                break;
            }
            step_ms = 8u64;
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
        if (nr <= 6 && active_other_cpus <= 5.5) || waited_ms >= eff_max_wait_ms {
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
    if read_nr_running() > 16 {
        wait_for_quiet_host_and_repin(12, true);
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
        if stolen < 15_000 && sib_d_ns < 15_000 && sib_d_j == 0 && ccd_d_ns < 1_500_000 {
            return;
        }
        if iter == 3 && is_self_repin_allowed() && read_nr_running() > 16 {
            wait_for_quiet_host_and_repin(16, true);
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
