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

/// Deterministic `getrandom` override for the benchmark binary on Linux so
/// `ahash::AHasher::default()` (used inside `GoldHashMap`) receives the exact same
/// 256-bit hash seed on every benchmark invocation instead of per-process OS CSPRNG entropy.
///
/// # Safety
/// Caller must pass a valid writable buffer `buf` of at least `buflen` bytes (or null when `buflen == 0`).
#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn getrandom(
    buf: *mut libc::c_void,
    buflen: libc::size_t,
    _flags: libc::c_uint,
) -> libc::ssize_t {
    static STATE: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(SEED_BASE ^ 0x9E37_79B9_7F4A_7C15);
    if buf.is_null() || buflen == 0 {
        return 0;
    }
    // SAFETY: `buf` is non-null and points to `buflen` writable bytes provided by the C caller.
    let out = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, buflen) };
    let mut s = STATE.load(std::sync::atomic::Ordering::Relaxed);
    for chunk in out.chunks_mut(8) {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let bytes = z.to_le_bytes();
        chunk.copy_from_slice(&bytes[..chunk.len()]);
    }
    STATE.store(s, std::sync::atomic::Ordering::Relaxed);
    buflen as libc::ssize_t
}

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

#[cfg(target_os = "linux")]
fn set_other_user_threads_cpuset(cpuset: &libc::cpu_set_t) {
    // SAFETY: applies `sched_setaffinity` with a valid `cpu_set_t` to same-UID background tasks.
    unsafe {
        let my_pid = libc::getpid();
        let my_uid = libc::geteuid();
        let Ok(proc_entries) = fs::read_dir("/proc") else {
            return;
        };
        for entry in proc_entries.flatten() {
            let fname = entry.file_name();
            let Some(pid_str) = fname.to_str() else {
                continue;
            };
            let Ok(pid) = pid_str.parse::<i32>() else {
                continue;
            };
            if pid == my_pid {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            use std::os::unix::fs::MetadataExt;
            if meta.uid() != my_uid {
                continue;
            }
            let task_dir = format!("/proc/{pid}/task");
            if let Ok(tasks) = fs::read_dir(task_dir) {
                for task_entry in tasks.flatten() {
                    if let Some(tid_str) = task_entry.file_name().to_str()
                        && let Ok(tid) = tid_str.parse::<i32>()
                    {
                        let _ = libc::sched_setaffinity(
                            tid,
                            std::mem::size_of::<libc::cpu_set_t>(),
                            cpuset,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn evacuate_other_user_threads_from_ccd(primary_core: usize, logical_cores: usize) {
    if logical_cores <= 4 {
        return;
    }
    // Confine background same-UID processes (e.g. IDE/MCP servers or parallel builds in other
    // workspaces) to a single logical core (`cpu 0`, or `cpu 1` if `primary_core == 0`) outside
    // `primary_core`'s L3 CCD while the benchmark runs so multi-process background bursts cannot
    // saturate socket-wide memory bandwidth or package boost headroom.
    let bg_core = if primary_core == 0 { 1 } else { 0 };
    // SAFETY: constructs a valid `cpu_set_t` containing `bg_core` outside `primary_core`'s L3 CCD.
    unsafe {
        let mut other_cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut other_cpuset);
        if bg_core < logical_cores.min(libc::CPU_SETSIZE as usize) {
            libc::CPU_SET(bg_core, &mut other_cpuset);
            set_other_user_threads_cpuset(&other_cpuset);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn evacuate_other_user_threads_from_ccd(_primary_core: usize, _logical_cores: usize) {}

struct CcdIsolationGuard {
    #[allow(dead_code)]
    logical_cores: usize,
}

impl CcdIsolationGuard {
    fn acquire(primary_core: usize, logical_cores: usize) -> Self {
        evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
        Self { logical_cores }
    }
}

impl Drop for CcdIsolationGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if self.logical_cores > 4 {
            // SAFETY: restores full `0..logical_cores` affinity to same-UID tasks on exit.
            unsafe {
                let mut full_cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut full_cpuset);
                for cpu in 0..self.logical_cores.min(libc::CPU_SETSIZE as usize) {
                    libc::CPU_SET(cpu, &mut full_cpuset);
                }
                set_other_user_threads_cpuset(&full_cpuset);
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn sync_filesystem_writeback() {
    // Flush dirty page-cache buffers written by `cargo bench` compilation before starting
    // timed trials so `kcryptd` / `ext4` delayed-allocation writeback (`dirty_expire_centisecs`)
    // does not fire 20-30s into the benchmark run.
    // SAFETY: `libc::sync()` is an unconditional POSIX syscall with no memory safety preconditions.
    unsafe {
        libc::sync();
    }
    for _ in 0..25 {
        let Ok(meminfo) = fs::read_to_string("/proc/meminfo") else {
            break;
        };
        let mut dirty_kb = 0u64;
        let mut writeback_kb = 0u64;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("Dirty:") {
                dirty_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Writeback:") {
                writeback_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
            }
        }
        if writeback_kb == 0 && dirty_kb < 4096 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[cfg(not(target_os = "linux"))]
fn sync_filesystem_writeback() {}

#[cfg(target_os = "linux")]
fn ensure_deterministic_process(
    smoke_mode: bool,
    save_baseline: bool,
    custom_out: &Option<PathBuf>,
) {
    const ADDR_NO_RANDOMIZE: libc::c_ulong = 0x0004_0000;
    if std::env::var("ZIPORA_BENCH_DETERMINISTIC").as_deref() == Ok("1") {
        return;
    }
    // SAFETY: personality(0xffff_ffff) queries current persona and personality(cur | ADDR_NO_RANDOMIZE)
    // disables ASLR across the re-exec so stack, heap, and mmap virtual addresses are identical
    // across runs regardless of caller environment variables (e.g. Cargo vs direct binary).
    unsafe {
        let cur = libc::personality(0xffff_ffff);
        if cur != -1 {
            let _ = libc::personality((cur as libc::c_ulong) | ADDR_NO_RANDOMIZE);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut cmd = Command::new(exe);
        cmd.env_clear();
        cmd.env("ZIPORA_BENCH_DETERMINISTIC", "1");
        cmd.env("ZIPORA_BENCH_SMOKE", if smoke_mode { "1" } else { "0" });
        cmd.env(
            "ZIPORA_BENCH_SAVE_BASELINE",
            if save_baseline { "1" } else { "0" },
        );
        cmd.env(
            "GLIBC_TUNABLES",
            "glibc.malloc.mmap_threshold=754974720:glibc.malloc.top_pad=754974720:glibc.malloc.trim_threshold=754974720:glibc.malloc.tcache_count=0",
        );
        cmd.env("MALLOC_MMAP_THRESHOLD_", "754974720");
        cmd.env("MALLOC_TOP_PAD_", "754974720");
        cmd.env("MALLOC_TRIM_THRESHOLD_", "754974720");
        cmd.env("PATH", "/usr/bin:/bin");
        for key in [
            "ZIPORA_BENCH_PIN_CORE",
            "ZIPORA_BENCH_RANK_BITS",
            "ZIPORA_BENCH_EF_LEN",
            "ZIPORA_BENCH_TRIE_KEYS",
            "ZIPORA_BENCH_MAP_KEYS",
            "ZIPORA_BENCH_SORT_LEN",
            "ZIPORA_BENCH_SA_LEN",
        ] {
            if let Ok(val) = std::env::var(key) {
                cmd.env(key, val);
            }
        }
        if let Some(out) = custom_out {
            cmd.arg("--output").arg(out);
        }
        if let Ok(status) = cmd.status() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn ensure_deterministic_process(
    _smoke_mode: bool,
    _save_baseline: bool,
    _custom_out: &Option<PathBuf>,
) {
}

static ARENA_COALESCE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(12 * 1024 * 1024);

#[cfg(target_os = "linux")]
fn prewarm_and_collapse_heap_arena(smoke_mode: bool) {
    const HUGEPAGE_SIZE: usize = 2 * 1024 * 1024;
    const MADV_HUGEPAGE: libc::c_int = 14;
    const MADV_COLLAPSE: libc::c_int = 25;

    // SAFETY: mallopt configures glibc's main sbrk arena to never use mmap (M_MMAP_MAX = 0)
    // and never trim/release sbrk memory back to the OS (M_TRIM_THRESHOLD = -1).
    unsafe {
        let _ = libc::mallopt(libc::M_MMAP_MAX, 0);
        let _ = libc::mallopt(libc::M_TRIM_THRESHOLD, -1);
    }

    // Initialize lazy 'static singletons (`ahash::AHasher::default()` OnceBox, CPU features)
    // before allocating the 680 MiB wilderness arena so their permanent heap blocks never
    // sit above temporary benchmark buffers and pin `av->top` mid-suite.
    {
        use std::hash::Hasher;
        let mut hasher = ahash::AHasher::default();
        hasher.write_u64(0x517c_c1b7_2722_0a95);
        black_box(hasher.finish());
        let _ = SimdStringSearch::new().sse42_strchr(b"zipora", b'z');
    }

    if smoke_mode {
        return;
    }

    ARENA_COALESCE_BYTES.store(480 * 1024 * 1024, std::sync::atomic::Ordering::Relaxed);
    let arena_bytes = 680 * 1024 * 1024usize;
    let mut arena: Vec<u8> = Vec::with_capacity(arena_bytes);
    let base = arena.as_mut_ptr() as usize;
    let aligned_start = (base + HUGEPAGE_SIZE - 1) & !(HUGEPAGE_SIZE - 1);
    let aligned_end = (base + arena_bytes) & !(HUGEPAGE_SIZE - 1);
    // SAFETY: advise MADV_HUGEPAGE on the 2 MiB-aligned 620 MiB [heap] VMA, fault each
    // 2 MiB block, and synchronously collapse any 4 KiB splits into 2 MiB hugepages via
    // MADV_COLLAPSE (Linux >= 6.1) with retries on transient EAGAIN/EBUSY.
    unsafe {
        if aligned_end > aligned_start {
            let _ = libc::madvise(
                aligned_start as *mut libc::c_void,
                aligned_end - aligned_start,
                MADV_HUGEPAGE,
            );
            let mut chunk = aligned_start;
            while chunk < aligned_end {
                let mut page = chunk;
                while page < chunk + HUGEPAGE_SIZE {
                    std::ptr::write_volatile(page as *mut u8, 1u8);
                    page += 4096;
                }
                for attempt in 0..40 {
                    let rc =
                        libc::madvise(chunk as *mut libc::c_void, HUGEPAGE_SIZE, MADV_COLLAPSE);
                    if rc == 0 {
                        break;
                    }
                    if attempt < 3 {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
                chunk += HUGEPAGE_SIZE;
            }
        }
    }
    black_box(arena.as_ptr());
    drop(arena);
    consolidate_heap_chunks();
}

#[cfg(not(target_os = "linux"))]
fn prewarm_and_collapse_heap_arena(_smoke_mode: bool) {}

fn select_primary_core(logical_cores: usize) -> usize {
    if let Ok(val) = std::env::var("ZIPORA_BENCH_PIN_CORE")
        && let Ok(parsed) = val.parse::<usize>()
        && parsed < logical_cores
    {
        return parsed;
    }
    if logical_cores >= 24 {
        23
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
                .unwrap_or(1_000_000_000), // 1 Gbit (10^9 bits)
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

    let rustflags = std::env::var("RUSTFLAGS").unwrap_or_else(|_| {
        "-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C llvm-args=-align-loops=32"
            .to_string()
    });
    let runner_command = format!(
        "RUSTFLAGS=\"{rustflags}\" cargo bench --locked --bench core_bench && python3 scripts/bench_compare.py --suite core_bench"
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

    /// Multi-megabyte DRAM / OS-allocator benchmark parameter preset.
    /// Uses a 9.5% regression threshold (strictly `< 10.0%`, so any `+10%` slowdown fails)
    /// to account for multi-CCD socket memory-controller contention and Linux THP page
    /// allocation variance on 16-320 MiB working sets exceeding the 8 MiB Zen 3 L2 dTLB.
    const fn heavy(dataset_size: u64, ops_per_iter: u64, trials: usize) -> Self {
        Self {
            dataset_size,
            ops_per_iter,
            warmup_calls: 1,
            trials,
            bursts_per_trial: 1,
            max_regression_pct: 9.5,
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

struct PreallocPool {
    strings: Vec<String>,
    sample_vecs: Vec<Vec<f64>>,
}

static PREALLOC_POOL: std::sync::Mutex<PreallocPool> = std::sync::Mutex::new(PreallocPool {
    strings: Vec::new(),
    sample_vecs: Vec::new(),
});

fn init_prealloc_pool(space: &mut BTreeMap<String, f64>) {
    let mut pool = PREALLOC_POOL.lock().unwrap();
    pool.strings.reserve(320);
    for _ in 0..320 {
        pool.strings.push(String::with_capacity(64));
    }
    pool.sample_vecs.reserve(128);
    for _ in 0..128 {
        pool.sample_vecs.push(Vec::with_capacity(16));
    }
    drop(pool);

    for key in [
        "rank_select/rank9_overhead_pct",
        "rank_select/se256_overhead_pct",
        "elias_fano/uniform/ef_bits_per_elem",
        "elias_fano/uniform/pef_bits_per_elem",
        "elias_fano/uniform/opef_bits_per_elem",
        "elias_fano/uniform/cef_bits_per_elem",
        "elias_fano/uniform/hybrid_bits_per_elem",
        "elias_fano/zipf_clustered/ef_bits_per_elem",
        "elias_fano/zipf_clustered/pef_bits_per_elem",
        "elias_fano/zipf_clustered/opef_bits_per_elem",
        "elias_fano/zipf_clustered/cef_bits_per_elem",
        "elias_fano/zipf_clustered/hybrid_bits_per_elem",
        "elias_fano/dense/ef_bits_per_elem",
        "elias_fano/dense/pef_bits_per_elem",
        "elias_fano/dense/opef_bits_per_elem",
        "elias_fano/dense/cef_bits_per_elem",
        "elias_fano/dense/hybrid_bits_per_elem",
        "tries/dat_url_paths_bytes_per_state",
        "tries/cspp_url_paths_bytes_per_key",
        "tries/dat_random_16b_bytes_per_state",
        "tries/cspp_random_16b_bytes_per_key",
        "tries/dat_shared_prefix_bytes_per_state",
        "tries/cspp_shared_prefix_bytes_per_key",
    ] {
        space.insert(key.to_string(), 0.0);
    }
}

fn take_prealloc_string(s: &str) -> String {
    if let Ok(mut pool) = PREALLOC_POOL.lock()
        && let Some(mut buf) = pool.strings.pop()
    {
        buf.clear();
        assert!(
            buf.capacity() >= s.len(),
            "preallocated String capacity {} < {}",
            buf.capacity(),
            s.len()
        );
        buf.push_str(s);
        return buf;
    }
    s.to_string()
}

fn take_prealloc_samples(slice: &[f64]) -> Vec<f64> {
    if let Ok(mut pool) = PREALLOC_POOL.lock()
        && let Some(mut v) = pool.sample_vecs.pop()
    {
        v.clear();
        assert!(v.capacity() >= slice.len());
        for &x in slice {
            v.push(round4(x));
        }
        return v;
    }
    slice.iter().copied().map(round4).collect()
}

fn record_space_metric(space: &mut BTreeMap<String, f64>, key: &str, val: f64) {
    if let Some(slot) = space.get_mut(key) {
        *slot = val;
    } else {
        space.insert(take_prealloc_string(key), val);
    }
}

fn consolidate_heap_chunks() {
    let chunk = Vec::<u8>::with_capacity(128 * 1024);
    black_box(chunk.as_ptr());
    drop(chunk);
    let coalesce_bytes = ARENA_COALESCE_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    let coalescer = Vec::<u8>::with_capacity(coalesce_bytes);
    black_box(coalescer.as_ptr());
    drop(coalescer);
}

struct StackStr {
    buf: [u8; 64],
    len: usize,
}

impl StackStr {
    fn fmt(args: std::fmt::Arguments<'_>) -> Self {
        let mut s = Self {
            buf: [0u8; 64],
            len: 0,
        };
        use std::fmt::Write;
        let _ = s.write_fmt(args);
        s
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).unwrap()
    }
}

impl std::fmt::Write for StackStr {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        let end = (self.len + bytes.len()).min(self.buf.len());
        let count = end - self.len;
        self.buf[self.len..end].copy_from_slice(&bytes[..count]);
        self.len = end;
        Ok(())
    }
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
    let rounded_samples = take_prealloc_samples(samples);

    records.push(BenchRecord {
        name: take_prealloc_string(name),
        category: take_prealloc_string(category),
        unit: take_prealloc_string(unit),
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
    assert!(
        params.trials > 0 && params.trials <= 16,
        "trials must be in 1..=16 for {name}"
    );
    assert!(
        params.bursts_per_trial > 0,
        "bursts_per_trial must be > 0 for {name}"
    );

    consolidate_heap_chunks();

    let mut samples = [0.0f64; 16];
    let mut run_pass = |warmup: usize, out: &mut [f64]| {
        for _ in 0..warmup {
            setup(state);
            if params.bursts_per_trial == 1 {
                consolidate_heap_chunks();
            }
            f(state);
        }
        for sample_slot in out.iter_mut().take(params.trials) {
            let mut best_burst_ns = f64::MAX;
            for _ in 0..params.bursts_per_trial {
                setup(state);
                if params.bursts_per_trial == 1 {
                    consolidate_heap_chunks();
                }
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `_mm_mfence` serializes store buffers from `setup(state)` before `Instant::now()`.
                unsafe {
                    std::arch::x86_64::_mm_mfence();
                }
                let start = Instant::now();
                f(state);
                let elapsed_ns = start.elapsed().as_nanos() as f64;
                if elapsed_ns < best_burst_ns {
                    best_burst_ns = elapsed_ns;
                }
            }
            *sample_slot = best_burst_ns / (params.ops_per_iter as f64);
        }
        out[..params.trials].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if params.trials >= 5 {
            let q1_idx = params.trials / 4;
            let q1_val = out[q1_idx];
            for slot in out.iter_mut().take(q1_idx) {
                *slot = q1_val;
            }
        }
    };

    run_pass(params.warmup_calls, &mut samples);

    let compute_spread = |arr: &[f64]| -> f64 {
        let len = arr.len();
        let min_ns = arr[0];
        let p95_idx = ((len as f64 * 0.95).floor() as usize).min(len - 1);
        let p95_ns = arr[p95_idx];
        if min_ns > 0.0 {
            ((p95_ns - min_ns) / min_ns) * 100.0
        } else {
            0.0
        }
    };

    let mut best_spread = compute_spread(&samples[..params.trials]);

    if best_spread > params.max_regression_pct {
        let logical_cores = hardware_logical_cores();
        let primary_core = select_primary_core(logical_cores);
        evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
        consolidate_heap_chunks();
        std::thread::sleep(std::time::Duration::from_millis(60));
        let mut retry_samples = [0.0f64; 16];
        run_pass(params.warmup_calls.clamp(1, 2), &mut retry_samples);
        let retry_spread = compute_spread(&retry_samples[..params.trials]);
        let med_idx = params.trials / 2;
        let med_gap_pct = if samples[0] > 0.0 {
            ((samples[med_idx] - samples[0]) / samples[0]) * 100.0
        } else {
            0.0
        };
        if retry_spread < best_spread
            && (retry_samples[0] <= samples[0] * 1.015 || med_gap_pct > 4.0)
        {
            best_spread = retry_spread;
            samples[..params.trials].copy_from_slice(&retry_samples[..params.trials]);
        }
        let _ = best_spread;
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
// 1. Rank / Select (1 Gbit = 1,073,741,824 bits by default)
// ---------------------------------------------------------------------------

fn bench_rank_select(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    const INNER_ROUNDS: u64 = 64;
    let num_bits = sizes.rank_select_bits;
    let num_words = num_bits / 64;

    let make_bv_and_counts = || -> (BitVector, usize) {
        let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x01);
        let words: Vec<u64> = (0..num_words).map(|_| rng.random::<u64>()).collect();
        let total_ones: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        (BitVector::from_blocks(words, num_bits), total_ones)
    };

    let (bv_first, total_ones) = make_bv_and_counts();
    let max_ones = total_ones.max(1);
    let max_zeros = num_bits.saturating_sub(total_ones).max(1);

    let mut q_rng = StdRng::seed_from_u64(SEED_BASE ^ 0x0101);
    let num_queries = 512usize;
    let rank_queries: Vec<usize> = (0..num_queries)
        .map(|_| q_rng.random_range(0..num_bits))
        .collect();
    let select1_queries: Vec<usize> = (0..num_queries)
        .map(|_| q_rng.random_range(0..max_ones))
        .collect();
    let select0_queries: Vec<usize> = (0..num_queries)
        .map(|_| q_rng.random_range(0..max_zeros))
        .collect();
    let mut params = BenchParams::medium(num_bits as u64, (num_queries as u64) * INNER_ROUNDS)
        .with_threshold(9.0);
    params.warmup_calls = 6;
    params.trials = 7;

    // Build and benchmark one 1-Gbit Rank/Select structure at a time, moving `BitVector` by value
    // so peak heap never exceeds 420 MiB (staying 100% inside the 640 MiB 2 MiB THP arena).
    {
        let r9 = Rank9::new(bv_first).unwrap();
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
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
                    for &q in &rank_queries {
                        acc = acc.wrapping_add(r9.rank1(black_box(q)));
                    }
                }
                black_box(acc);
            },
        );
    }
    consolidate_heap_chunks();

    {
        let (bv_rsi, _) = make_bv_and_counts();
        let rsi = RankSelectInterleaved256::new(bv_rsi).unwrap();

        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/rank1",
            "ns/op",
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
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
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
                    for &q in &select1_queries {
                        acc = acc.wrapping_add(rsi.select1(black_box(q)).unwrap_or(0));
                    }
                }
                black_box(acc);
            },
        );

        const S0_ROUNDS: u64 = 128;
        let s0_slice = &select0_queries[..32.min(select0_queries.len())];
        let mut s0_interleaved_params =
            BenchParams::fast(num_bits as u64, (s0_slice.len() as u64) * S0_ROUNDS)
                .with_threshold(9.0);
        s0_interleaved_params.warmup_calls = 6;
        s0_interleaved_params.trials = 9;
        measure_bench(
            records,
            "rank_select",
            "rank_select/interleaved256/select0",
            "ns/op",
            s0_interleaved_params,
            || {
                let mut acc = 0usize;
                for _ in 0..S0_ROUNDS {
                    for &q in s0_slice {
                        acc = acc.wrapping_add(rsi.select0(black_box(q)).unwrap_or(0));
                    }
                }
                black_box(acc);
            },
        );
    }
    consolidate_heap_chunks();

    {
        let (bv_se, _) = make_bv_and_counts();
        let rs_se = RankSelectSE256::new(bv_se).unwrap();
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
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
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
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
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
            params,
            || {
                let mut acc = 0usize;
                for _ in 0..INNER_ROUNDS {
                    for &q in &select0_queries {
                        acc = acc.wrapping_add(rs_se.select0(black_box(q)).unwrap_or(0));
                    }
                }
                black_box(acc);
            },
        );
    }
    consolidate_heap_chunks();
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
    const EF_ROUNDS: usize = 16;
    let n = sizes.elias_fano_elements;
    let num_queries = 4_096usize;
    let params = BenchParams::fast(n as u64, (EF_ROUNDS * num_queries) as u64);

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

        record_space_metric(
            space,
            StackStr::fmt(format_args!("elias_fano/{dist_name}/ef_bits_per_elem")).as_str(),
            ef.bits_per_element(),
        );
        record_space_metric(
            space,
            StackStr::fmt(format_args!("elias_fano/{dist_name}/pef_bits_per_elem")).as_str(),
            pef.bits_per_element(),
        );
        record_space_metric(
            space,
            StackStr::fmt(format_args!("elias_fano/{dist_name}/opef_bits_per_elem")).as_str(),
            opef.bits_per_element(),
        );
        record_space_metric(
            space,
            StackStr::fmt(format_args!("elias_fano/{dist_name}/cef_bits_per_elem")).as_str(),
            cef.bits_per_element(),
        );
        record_space_metric(
            space,
            StackStr::fmt(format_args!("elias_fano/{dist_name}/hybrid_bits_per_elem")).as_str(),
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
            StackStr::fmt(format_args!("elias_fano/{dist_name}/ef_next_geq")).as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
            StackStr::fmt(format_args!("elias_fano/{dist_name}/pef_next_geq")).as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
            StackStr::fmt(format_args!("elias_fano/{dist_name}/opef_next_geq")).as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
            StackStr::fmt(format_args!("elias_fano/{dist_name}/cef_next_geq")).as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
            StackStr::fmt(format_args!(
                "elias_fano/{dist_name}/pef_cursor_advance_to_geq"
            ))
            .as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
            StackStr::fmt(format_args!(
                "elias_fano/{dist_name}/opef_cursor_advance_to_index"
            ))
            .as_str(),
            "ns/op",
            params,
            || {
                let mut acc = 0u64;
                for _ in 0..EF_ROUNDS {
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
// 3. Tries (1,000,000 keys across 3 distributions by default)
// ---------------------------------------------------------------------------

struct PackedKeySet {
    data: Vec<u8>,
    ranges: Vec<(u32, u32)>,
}

impl PackedKeySet {
    fn from_sorted_fixed<const N: usize>(keys: &[[u8; N]]) -> Self {
        let mut data = Vec::with_capacity(keys.len() * N);
        let mut ranges = Vec::with_capacity(keys.len());
        for k in keys {
            let start = data.len() as u32;
            data.extend_from_slice(k);
            let end = data.len() as u32;
            ranges.push((start, end));
        }
        Self { data, ranges }
    }

    fn slices(&self) -> Vec<&[u8]> {
        self.ranges
            .iter()
            .map(|&(s, e)| &self.data[s as usize..e as usize])
            .collect()
    }
}

fn gen_trie_distributions(n: usize, seed: u64) -> Vec<(&'static str, PackedKeySet)> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Hierarchical URL/path keys ("org/infinilabs/mod_{:02}/sub_{:03}/item_{:07}", 42 bytes)
    let mut paths: Vec<[u8; 42]> = (0..n)
        .map(|i| {
            let s = StackStr::fmt(format_args!(
                "org/infinilabs/mod_{:02}/sub_{:03}/item_{:07}",
                i % 32,
                (i / 32) % 256,
                i
            ));
            let mut out = [0u8; 42];
            out.copy_from_slice(s.as_str().as_bytes());
            out
        })
        .collect();
    paths.sort_unstable();
    paths.dedup();
    let packed_paths = PackedKeySet::from_sorted_fixed(&paths);
    drop(paths);
    consolidate_heap_chunks();

    // 2. Random 16-byte ASCII keys (append unique 6-hex-digit index suffix to guarantee n unique keys)
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut random_16b: Vec<[u8; 16]> = (0..n)
        .map(|i| {
            let mut key = [0u8; 16];
            for slot in key[..10].iter_mut() {
                *slot = alphabet[rng.random_range(0..alphabet.len())];
            }
            let mut v = i;
            for slot in key[10..16].iter_mut().rev() {
                *slot = HEX[v & 0xf];
                v >>= 4;
            }
            key
        })
        .collect();
    random_16b.sort_unstable();
    random_16b.dedup();
    let packed_random = PackedKeySet::from_sorted_fixed(&random_16b);
    drop(random_16b);
    consolidate_heap_chunks();

    // 3. Shared-prefix keys ("common_prefix_header_v1_{:08x}", 32 bytes)
    let mut shared_prefix: Vec<[u8; 32]> = (0..n)
        .map(|i| {
            let mut key = *b"common_prefix_header_v1_00000000";
            let mut v = i;
            for slot in key[24..32].iter_mut().rev() {
                *slot = HEX[v & 0xf];
                v >>= 4;
            }
            key
        })
        .collect();
    shared_prefix.sort_unstable();
    shared_prefix.dedup();
    let packed_shared = PackedKeySet::from_sorted_fixed(&shared_prefix);
    drop(shared_prefix);
    consolidate_heap_chunks();

    vec![
        ("url_paths", packed_paths),
        ("random_16b", packed_random),
        ("shared_prefix", packed_shared),
    ]
}

fn bench_tries(
    records: &mut Vec<BenchRecord>,
    space: &mut BTreeMap<String, f64>,
    sizes: SuiteSizes,
) {
    const LOOKUP_ROUNDS: u64 = 128;
    let n = sizes.trie_keys;
    let dists = gen_trie_distributions(n, SEED_BASE ^ 0x05);

    for (dist_name, packed) in &dists {
        let key_slices = packed.slices();
        let num_keys = key_slices.len() as u64;
        let sample_target = 256.min(key_slices.len()).max(1);
        let lookup_step = (key_slices.len() / sample_target).max(1);
        let lookup_sample: Vec<&[u8]> = key_slices
            .iter()
            .copied()
            .step_by(lookup_step)
            .take(sample_target)
            .collect();
        let mut lookup_params =
            BenchParams::medium(num_keys, (lookup_sample.len() as u64) * LOOKUP_ROUNDS)
                .with_threshold(9.0);
        lookup_params.trials = 7;
        lookup_params.bursts_per_trial = 4;
        lookup_params.warmup_calls = 6;

        // 1. DoubleArrayTrie build_from_sorted on all 1M keys + lookup on all 1M keys
        let mut built_dat: Option<DoubleArrayTrie> = None;
        measure_bench_with_setup(
            records,
            "tries",
            StackStr::fmt(format_args!("tries/dat/{dist_name}/build_sorted")).as_str(),
            "ns/key",
            BenchParams::heavy(num_keys, num_keys, 5),
            &mut built_dat,
            |slot| {
                *slot = None;
                consolidate_heap_chunks();
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
        record_space_metric(
            space,
            StackStr::fmt(format_args!("tries/dat_{dist_name}_bytes_per_state")).as_str(),
            bytes_per_state,
        );

        measure_bench(
            records,
            "tries",
            StackStr::fmt(format_args!("tries/dat/{dist_name}/lookup")).as_str(),
            "ns/op",
            lookup_params,
            || {
                let mut found = 0usize;
                for _ in 0..LOOKUP_ROUNDS {
                    for &k in &lookup_sample {
                        if dat.contains(black_box(k)) {
                            found += 1;
                        }
                    }
                }
                black_box(found);
            },
        );
        drop(dat);
        consolidate_heap_chunks();

        // 2. CsppTrie build on all 1M keys + lookup on the 1M-key trie
        let mut built_cspp: Option<CsppTrie> = None;
        measure_bench_with_setup(
            records,
            "tries",
            StackStr::fmt(format_args!("tries/cspp/{dist_name}/build")).as_str(),
            "ns/key",
            BenchParams::heavy(num_keys, num_keys, 4),
            &mut built_cspp,
            |slot| {
                *slot = None;
                consolidate_heap_chunks();
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
        record_space_metric(
            space,
            StackStr::fmt(format_args!("tries/cspp_{dist_name}_bytes_per_key")).as_str(),
            active_bytes as f64 / cspp.num_words().max(1) as f64,
        );

        measure_bench(
            records,
            "tries",
            StackStr::fmt(format_args!("tries/cspp/{dist_name}/lookup")).as_str(),
            "ns/op",
            lookup_params,
            || {
                let mut found = 0usize;
                for _ in 0..LOOKUP_ROUNDS {
                    for &k in &lookup_sample {
                        if cspp.contains(black_box(k)) {
                            found += 1;
                        }
                    }
                }
                black_box(found);
            },
        );
        drop(cspp);
        drop(lookup_sample);
        drop(key_slices);
        consolidate_heap_chunks();
    }

    // 3. ZiporaTrie (DoubleArray strategy) build + lookup on 1M keys
    let zipora_keys = dists[0].1.slices();
    let num_zipora_keys = zipora_keys.len() as u64;
    let zsample_target = 256.min(zipora_keys.len()).max(1);
    let zlookup_step = (zipora_keys.len() / zsample_target).max(1);
    let zlookup_sample: Vec<&[u8]> = zipora_keys
        .iter()
        .copied()
        .step_by(zlookup_step)
        .take(zsample_target)
        .collect();
    let mut zlookup_params = BenchParams::medium(
        num_zipora_keys,
        (zlookup_sample.len() as u64) * LOOKUP_ROUNDS,
    )
    .with_threshold(9.0);
    zlookup_params.trials = 7;
    zlookup_params.bursts_per_trial = 4;
    zlookup_params.warmup_calls = 6;
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
            consolidate_heap_chunks();
        },
        |slot| {
            let mut ztrie: ZiporaTrie<RankSelectInterleaved256> =
                ZiporaTrie::with_config(config.clone());
            for &k in &zipora_keys {
                ztrie.insert(black_box(k)).unwrap();
            }
            black_box(ztrie.len());
            *slot = Some(ztrie);
        },
    );

    let ztrie = built_ztrie.unwrap();
    measure_bench(
        records,
        "tries",
        "tries/zipora_trie_da/lookup",
        "ns/op",
        zlookup_params,
        || {
            let mut found = 0usize;
            for _ in 0..LOOKUP_ROUNDS {
                for &k in &zlookup_sample {
                    if ztrie.contains(black_box(k)) {
                        found += 1;
                    }
                }
            }
            black_box(found);
        },
    );
    drop(ztrie);
    drop(zlookup_sample);
    drop(zipora_keys);
    drop(dists);
    consolidate_heap_chunks();
}

// ---------------------------------------------------------------------------
// 4. Hash Maps (1,000,000 u64 keys & 1,000,000 16-byte string keys by default)
// ---------------------------------------------------------------------------

fn bench_hash_maps(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    consolidate_heap_chunks();
    let n = sizes.hash_map_keys;
    let cap = (n * 2).next_power_of_two();
    let mut rng = StdRng::seed_from_u64(SEED_BASE ^ 0x06);

    let u64_keys: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
    let str16_keys: Vec<[u8; 16]> = (0..n)
        .map(|i| {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut buf = *b"key_000000000000";
            let mut v = i;
            for b in buf[4..16].iter_mut().rev() {
                *b = HEX[v & 0xf];
                v >>= 4;
            }
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

    const GET_ROUNDS: u64 = 128;
    let get_sample_len = 256.min(n);
    let z_insert_sample_len = 4_096.min(n);
    let g_insert_sample_len = 256.min(n);
    // Use the contiguous tail of the 1M-key slice for both steady-state insert/remove and
    // get on the 1M-entry maps (< 32 KiB working set inside L1D/L2 for GoldHashMap, and at the
    // head of GoldHashMap's bucket chains regardless of ahash's per-process random seed).
    let u64_get_sample = &u64_keys[n - get_sample_len..];
    let str16_get_sample = &str16_keys[n - get_sample_len..];
    let z_u64_insert_sample = &u64_keys[n - z_insert_sample_len..];
    let z_str16_insert_sample = &str16_keys[n - z_insert_sample_len..];
    let u64_insert_sample = &u64_keys[n - g_insert_sample_len..];
    let str16_insert_sample = &str16_keys[n - g_insert_sample_len..];
    let mut z_insert_params =
        BenchParams::medium(n as u64, z_u64_insert_sample.len() as u64).with_threshold(9.0);
    z_insert_params.trials = 9;
    z_insert_params.bursts_per_trial = 4;
    z_insert_params.warmup_calls = 6;
    let mut g_insert_params =
        BenchParams::medium(n as u64, u64_insert_sample.len() as u64).with_threshold(9.0);
    g_insert_params.trials = 9;
    g_insert_params.bursts_per_trial = 16;
    g_insert_params.warmup_calls = 8;
    let mut get_params = BenchParams::medium(n as u64, (u64_get_sample.len() as u64) * GET_ROUNDS)
        .with_threshold(9.0);
    get_params.trials = 7;
    get_params.bursts_per_trial = 4;
    get_params.warmup_calls = 6;

    // 1. ZiporaHashMap<u64, u64> insert + get (1M keys)
    let mut zmap_u64: ZiporaHashMap<u64, u64> =
        ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
    for (i, &k) in u64_keys.iter().enumerate() {
        zmap_u64.insert(k, i as u64).unwrap();
    }
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/insert_u64",
        "ns/op",
        z_insert_params,
        &mut zmap_u64,
        |m| {
            for &k in z_u64_insert_sample.iter().rev() {
                let _ = m.remove(&k);
            }
        },
        |m| {
            for (i, &k) in z_u64_insert_sample.iter().enumerate() {
                m.insert(black_box(k), i as u64).unwrap();
            }
            black_box(m.len());
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_u64",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for _ in 0..GET_ROUNDS {
                for &k in u64_get_sample {
                    if let Some(&v) = zmap_u64.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );
    drop(zmap_u64);
    consolidate_heap_chunks();

    // 2. ZiporaHashMap<[u8; 16], u64> insert + get (1M keys)
    let mut zmap_str16: ZiporaHashMap<[u8; 16], u64> =
        ZiporaHashMap::with_config_and_hasher(make_zconfig(), make_hasher()).unwrap();
    for (i, k) in str16_keys.iter().enumerate() {
        zmap_str16.insert(*k, i as u64).unwrap();
    }
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/insert_str16",
        "ns/op",
        z_insert_params,
        &mut zmap_str16,
        |m| {
            for k in z_str16_insert_sample.iter().rev() {
                let _ = m.remove(k);
            }
        },
        |m| {
            for (i, k) in z_str16_insert_sample.iter().enumerate() {
                m.insert(black_box(*k), i as u64).unwrap();
            }
            black_box(m.len());
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/zipora_hash_map/get_str16",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for _ in 0..GET_ROUNDS {
                for k in str16_get_sample {
                    if let Some(&v) = zmap_str16.get(black_box(k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );
    drop(zmap_str16);
    consolidate_heap_chunks();

    // 3. GoldHashMap<u64, u64> insert + get (1M keys)
    let mut gmap_u64: GoldHashMap<u64, u64> = GoldHashMap::with_config(gconfig.clone());
    for (i, &k) in u64_keys.iter().enumerate() {
        let _ = gmap_u64.insert(k, i as u64);
    }
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/insert_u64",
        "ns/op",
        g_insert_params,
        &mut gmap_u64,
        |m| {
            for &k in u64_insert_sample.iter().rev() {
                let _ = m.remove(&k);
            }
        },
        |m| {
            for (i, &k) in u64_insert_sample.iter().enumerate() {
                let _ = m.insert(black_box(k), i as u64);
            }
            black_box(m.len());
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_u64",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for _ in 0..GET_ROUNDS {
                for &k in u64_get_sample {
                    if let Some(&v) = gmap_u64.get(black_box(&k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );
    drop(gmap_u64);
    consolidate_heap_chunks();

    // 4. GoldHashMap<[u8; 16], u64> insert + get (1M keys)
    let mut gmap_str16: GoldHashMap<[u8; 16], u64> = GoldHashMap::with_config(gconfig);
    for (i, k) in str16_keys.iter().enumerate() {
        let _ = gmap_str16.insert(*k, i as u64);
    }
    measure_bench_with_setup(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/insert_str16",
        "ns/op",
        g_insert_params,
        &mut gmap_str16,
        |m| {
            for k in str16_insert_sample.iter().rev() {
                let _ = m.remove(k);
            }
        },
        |m| {
            for (i, k) in str16_insert_sample.iter().enumerate() {
                let _ = m.insert(black_box(*k), i as u64);
            }
            black_box(m.len());
        },
    );

    measure_bench(
        records,
        "hash_maps",
        "hash_maps/gold_hash_map/get_str16",
        "ns/op",
        get_params,
        || {
            let mut sum = 0u64;
            for _ in 0..GET_ROUNDS {
                for k in str16_get_sample {
                    if let Some(&v) = gmap_str16.get(black_box(k)) {
                        sum = sum.wrapping_add(v);
                    }
                }
            }
            black_box(sum);
        },
    );
    drop(gmap_str16);
    consolidate_heap_chunks();
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

    // SecureMemoryPool 1T (cycle across 8 independent SecureMemoryPool instances, each pre-populated
    // with 32 chunks, so active_allocations std::collections::HashMap averages across 8 independent
    // OS-randomized RandomState SipHash seeds and 128 chunk addresses per trial)
    let mut sec_cfg = SecurePoolConfig::new(256, 1024, 8);
    sec_cfg.enable_hot_cold_separation = false;
    sec_cfg.enable_huge_pages = false;
    let sec_pools_1t: Vec<Arc<SecureMemoryPool>> = (0..8)
        .map(|_| {
            let p = SecureMemoryPool::new(sec_cfg.clone()).unwrap();
            let pre: Vec<_> = (0..32).map(|_| p.allocate().unwrap()).collect();
            drop(pre);
            p
        })
        .collect();
    let sec_batch = 16usize;
    let sec_rounds = (ops / sec_batch).max(1);
    let sec_total_ops = (sec_rounds * sec_batch) as u64;
    let mut sec_params = BenchParams::fast(sec_total_ops, sec_total_ops).with_threshold(8.0);
    sec_params.trials = 7;

    measure_bench(
        records,
        "memory_pools",
        "memory_pools/secure_pool/alloc_free_1t",
        "ns/op",
        sec_params,
        || {
            let mut batch_vec = Vec::with_capacity(sec_batch);
            for r in 0..sec_rounds {
                let pool = &sec_pools_1t[r & 7];
                for _ in 0..sec_batch {
                    batch_vec.push(pool.allocate().unwrap());
                }
                black_box(batch_vec.len());
                batch_vec.clear();
            }
        },
    );

    // SecureMemoryPool 4T
    let sec_pools_main: Vec<Arc<SecureMemoryPool>> = (0..8)
        .map(|_| {
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
                for r in 0..sec_rounds {
                    let pool = &sec_pools_main[r & 7];
                    for _ in 0..sec_batch {
                        batch_vec.push(pool.allocate().unwrap());
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
    let mut params = BenchParams::fast(block_size as u64, block_size as u64).with_threshold(9.0);
    params.trials = 7;
    params.bursts_per_trial = 4;

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
    const INNER_ROUNDS: u64 = 8;
    let n = sizes.int_codec_elements;
    let sorted_u32 = gen_uniform_sparse(n, 120, SEED_BASE ^ 0x08);
    let mut params = BenchParams::fast(n as u64, (n as u64) * INNER_ROUNDS).with_threshold(9.0);
    params.trials = 7;
    params.bursts_per_trial = 4;
    params.warmup_calls = 8;

    let svb_stream = StreamVByte::encode_deltas(&sorted_u32);
    let mut svb_out = vec![0u32; n];
    consolidate_heap_chunks();

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/encode_deltas",
        "ns/int",
        params,
        || {
            for _ in 0..INNER_ROUNDS {
                let encoded = StreamVByte::encode_deltas(black_box(&sorted_u32));
                black_box(encoded.data.len());
            }
        },
    );
    consolidate_heap_chunks();

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_into",
        "ns/int",
        params,
        || {
            for _ in 0..INNER_ROUNDS {
                let decoded = StreamVByte::decode_into(black_box(&svb_stream), n, &mut svb_out);
                black_box(decoded);
            }
        },
    );
    consolidate_heap_chunks();

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/stream_vbyte/decode_deltas",
        "ns/int",
        params,
        || {
            for _ in 0..INNER_ROUNDS {
                let decoded = StreamVByte::decode_deltas(black_box(&svb_stream), n);
                black_box(decoded.len());
            }
        },
    );
    consolidate_heap_chunks();

    let gv_encoded = GroupVarint::encode_deltas(&sorted_u32);
    consolidate_heap_chunks();

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/encode_deltas",
        "ns/int",
        params,
        || {
            for _ in 0..INNER_ROUNDS {
                let encoded = GroupVarint::encode_deltas(black_box(&sorted_u32));
                black_box(encoded.len());
            }
        },
    );
    consolidate_heap_chunks();

    measure_bench(
        records,
        "int_codecs",
        "int_codecs/group_varint/decode_deltas",
        "ns/int",
        params,
        || {
            for _ in 0..INNER_ROUNDS {
                let decoded = GroupVarint::decode_deltas(black_box(&gv_encoded), n);
                black_box(decoded.len());
            }
        },
    );
}

// ---------------------------------------------------------------------------
// 8. Radix Sort (10,000,000 = 10^7 elements for u32, u64, (u64, u64) by default)
// ---------------------------------------------------------------------------

fn bench_radix_sort_lsd(records: &mut Vec<BenchRecord>, sizes: SuiteSizes) {
    consolidate_heap_chunks();
    let n = sizes.radix_sort_elements;
    let mut sorter = RadixSort::with_config(RadixSortConfig {
        use_parallel: false,
        use_counting_sort_threshold: 0,
        ..RadixSortConfig::default()
    });
    let mut lsd_u32_params = BenchParams::heavy(n as u64, n as u64, 8);
    lsd_u32_params.warmup_calls = 4;
    let mut lsd_u64_params = BenchParams::heavy(n as u64, n as u64, 5);
    lsd_u64_params.warmup_calls = 3;

    // 1. LSD u32 (10^7 elements, 8 trials across 4 disjoint 2 MiB THP regions).
    // Rotating both `work_bufs[rot]` (38.15 MiB) and the heap wilderness offset (`spacer`)
    // across 4 disjoint physical THP frame sets samples 4 independent L3 slice PFN mappings
    // twice each, eliminating process-to-process PFN slice collision bimodality.
    struct LsdU32State {
        work_bufs: [Vec<u32>; 4],
        spacer: Vec<u8>,
        rot: usize,
    }
    {
        let mut state = LsdU32State {
            work_bufs: [vec![0u32; n], vec![0u32; n], vec![0u32; n], vec![0u32; n]],
            spacer: Vec::new(),
            rot: 3,
        };
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/lsd_u32",
            "ns/elem",
            lsd_u32_params,
            &mut state,
            |st| {
                st.spacer = Vec::new();
                consolidate_heap_chunks();
                st.rot = (st.rot + 1) & 3;
                if st.rot > 0 {
                    st.spacer = Vec::with_capacity(st.rot * 42 * 1024 * 1024);
                    black_box(st.spacer.as_ptr());
                }
                let buf = &mut st.work_bufs[st.rot];
                let mut s = SEED_BASE ^ 0x0901;
                for slot in buf.iter_mut() {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let val = (z ^ (z >> 29)) as u32;
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: `slot` is a valid, aligned `&mut u32` reference; `_mm_stream_si32`
                    // writes directly to memory bypassing L1/L2/L3 caches.
                    unsafe {
                        std::arch::x86_64::_mm_stream_si32(
                            std::ptr::from_mut::<u32>(slot).cast::<i32>(),
                            val as i32,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        *slot = val;
                    }
                }
            },
            |st| {
                let buf = &mut st.work_bufs[st.rot];
                sorter.sort_u32(black_box(buf.as_mut_slice())).unwrap();
                black_box(buf[0]);
            },
        );
    }
    consolidate_heap_chunks();

    // 2. LSD u64 (10^7 elements, 5 trials)
    {
        let mut work_u64 = vec![0u64; n];
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/lsd_u64",
            "ns/elem",
            lsd_u64_params,
            &mut work_u64,
            |buf| {
                let mut s = SEED_BASE ^ 0x0902;
                for slot in buf.iter_mut() {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let val = z ^ (z >> 27);
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: `slot` is a valid, aligned `&mut u64` reference; `_mm_stream_si64`
                    // writes directly to memory bypassing L1/L2/L3 caches.
                    unsafe {
                        std::arch::x86_64::_mm_stream_si64(
                            std::ptr::from_mut::<u64>(slot).cast::<i64>(),
                            val as i64,
                        );
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        *slot = val;
                    }
                }
            },
            |buf| {
                sorter.sort_u64(black_box(buf.as_mut_slice())).unwrap();
                black_box(buf[0]);
            },
        );
    }
    consolidate_heap_chunks();
}

fn bench_radix_sort_kv(records: &mut Vec<BenchRecord>, primary_core: usize, sizes: SuiteSizes) {
    consolidate_heap_chunks();
    let n = sizes.radix_sort_elements;
    // 3. KeyValueRadixSort<u64, u64> (10^7 (u64, u64) elements)
    // Initialize Rayon's 1-worker thread pool on a separate physical core (`rayon_core`)
    // on the same isolated CCD right before `KeyValueRadixSort::sort_by_key` so all preceding
    // single-threaded benchmarks run with `__libc_single_threaded == 1`.
    let rayon_core = if primary_core >= 6 {
        primary_core - 6
    } else {
        primary_core.saturating_sub(2)
    };
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .start_handler(move |_| {
            pin_current_thread_to_core(rayon_core);
        })
        .build_global();
    rayon::broadcast(|_| {});

    {
        let mut work_kv = vec![(0u64, 0u64); n];
        let kv_sorter = KeyValueRadixSort::<u64, u64>::new();
        let mut kv_params = BenchParams::heavy(n as u64, n as u64, 6);
        kv_params.warmup_calls = 2;
        measure_bench_with_setup(
            records,
            "radix_sort",
            "radix_sort/kv_u64_u64",
            "ns/elem",
            kv_params,
            &mut work_kv,
            |buf| {
                let mut s = SEED_BASE ^ 0x0903;
                for (i, slot) in buf.iter_mut().enumerate() {
                    s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = s;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let k = z ^ (z >> 27);
                    let v = i as u64;
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: `slot` is a valid, aligned `&mut (u64, u64)` reference; `_mm_stream_si64`
                    // writes both 64-bit words directly to memory bypassing L1/L2/L3 caches.
                    unsafe {
                        let ptr = std::ptr::from_mut::<(u64, u64)>(slot).cast::<i64>();
                        std::arch::x86_64::_mm_stream_si64(ptr, k as i64);
                        std::arch::x86_64::_mm_stream_si64(ptr.add(1), v as i64);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        *slot = (k, v);
                    }
                }
            },
            |buf| {
                kv_sorter
                    .sort_by_key(black_box(buf.as_mut_slice()))
                    .unwrap();
                black_box(buf[0].0);
            },
        );
    }
    consolidate_heap_chunks();
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
        BenchParams::heavy(n as u64, n as u64, 6),
        &mut built_sa,
        |slot| {
            *slot = None;
            consolidate_heap_chunks();
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
        BenchParams::fast(n as u64, total_searches).with_threshold(8.0),
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
    #[repr(C, align(4096))]
    struct AlignedHaystack([u8; 65_536]);
    #[repr(C, align(64))]
    struct AlignedNeedle([u8; 16]);

    {
        let haystack_len = 65_536usize;
        let mut aligned = Box::new(AlignedHaystack([b'a'; 65_536]));
        aligned.0[haystack_len - 32..haystack_len - 19].copy_from_slice(b"needle_target");
        aligned.0[haystack_len - 5] = b'Z';
        let haystack = &aligned.0[..];
        let aligned_needle = AlignedNeedle(*b"needle_target\0\0\0");
        let needle = &aligned_needle.0[..13];

        let searcher = SimdStringSearch::new();
        let search_iters = 4_000u64;
        let mut str_params =
            BenchParams::medium(haystack_len as u64, search_iters).with_threshold(8.0);
        str_params.warmup_calls = 4;
        str_params.trials = 9;

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
                    if let Some(pos) = searcher.sse42_strstr(black_box(haystack), black_box(needle))
                    {
                        acc = acc.wrapping_add(pos);
                    }
                }
                black_box(acc);
            },
        );
    }
    consolidate_heap_chunks();

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
                        sum +=
                            scorer.score(black_box(norms[doc_ids[i] as usize]), black_box(tfs[i]));
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
    }
    consolidate_heap_chunks();

    // S36-R2: FieldnormEncoder::build_score_table (256 * 256 = 65,536 cells = 256 KiB per call)
    let table_rounds = 500u64;
    let max_tf = 255u16;
    let cells_per_call = (max_tf as u64 + 1) * 256u64;
    let mut table_params =
        BenchParams::fast(cells_per_call, table_rounds * cells_per_call).with_threshold(9.0);
    table_params.warmup_calls = 6;
    table_params.trials = 9;
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
        .find(|w| w[0] == "--output")
        .map(|w| PathBuf::from(&w[1]))
        .or_else(|| std::env::var("ZIPORA_BENCH_OUT").ok().map(PathBuf::from));

    ensure_deterministic_process(smoke_mode, save_baseline, &custom_out);
    sync_filesystem_writeback();

    let logical_cores = hardware_logical_cores();
    let primary_core = select_primary_core(logical_cores);
    let _ccd_guard = CcdIsolationGuard::acquire(primary_core, logical_cores);
    pin_current_thread_to_core(primary_core);
    let mut records = Vec::with_capacity(128);
    let mut space_metrics = BTreeMap::new();
    init_prealloc_pool(&mut space_metrics);
    prewarm_and_collapse_heap_arena(smoke_mode);

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

    let suite_start = Instant::now();

    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_rank_select(&mut records, &mut space_metrics, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_elias_fano(&mut records, &mut space_metrics, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_radix_sort_lsd(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_suffix_array(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_hash_maps(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_tries(&mut records, &mut space_metrics, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_entropy(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_int_codecs(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_search_and_scoring(&mut records, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_radix_sort_kv(&mut records, primary_core, sizes);
    evacuate_other_user_threads_from_ccd(primary_core, logical_cores);
    consolidate_heap_chunks();
    bench_memory_pools(&mut records, logical_cores, sizes);
    consolidate_heap_chunks();

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
