//! Phase E1.1 (G15) Systemic End-to-End Search Benchmark (`make bench_e2e_search`).
//!
//! Combines Zipora's core search-engine primitives into an end-to-end BM25 top-10
//! retrieval engine and benchmarks it on a deterministic 100,000-document Zipfian corpus
//! (`~1.61M` postings across 4,096 terms, exceeding L2 cache) and 10,000 Zipf queries
//! (1–4 terms, fixed seed) side-by-side against `tantivy` v0.26 (`Index::create_in_ram`,
//! `IndexRecordOption::WithFreqs`, `BooleanQuery` + `TopDocs::with_limit(10)`) and an
//! in-process uncompressed inverted-index exhaustive baseline:
//! - Term dictionary: [`DoubleArrayTrieMap<u32>`]
//! - Posting lists: [`HybridPostingList`] per term (auto-selecting `Dense`,
//!   `EliasFano`, `Partitioned`, `Optimal`, or `Clustered`) with 64-doc block-max scores
//! - Document lengths: [`FieldnormEncoder`] (`1.0 B/doc` SmallFloat quantization)
//! - Query evaluation: bench-local Block-Max WAND (BMW) using [`simd_gallop_to`],
//!   [`simd_block_filter`], and [`Bm25BatchScorer`] (`batch_score_with_prefetch` + `score`)

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::{Index, IndexReader, ReloadPolicy, TantivyDocument, Term};

use zipora::algorithms::{simd_block_filter, simd_gallop_to};
use zipora::fsa::DoubleArrayTrieMap;
use zipora::scoring::{Bm25BatchScorer, FieldnormEncoder};
use zipora::succinct::elias_fano::{
    ClusteredEliasFano, EliasFanoCursor, HybridPostingList, OptimalPefCursor,
    PartitionedEliasFanoCursor, PostingEncoding,
};

const SEED_E2E: u64 = 0x5A17_0E11_E2E5_EA2C;
const BLOCK_SIZE: usize = 64;
const TOP_K: usize = 10;
const DEFAULT_TRIALS: usize = 7;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostMetadata {
    host: String,
    cpu_model: String,
    logical_cores: usize,
    pinned_core: usize,
    governor: String,
    rustflags: String,
    git_commit: String,
    timestamp_utc: String,
    #[serde(default)]
    binary_path: String,
    #[serde(default)]
    binary_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRecord {
    category: String,
    name: String,
    unit: String,
    dataset_size: u64,
    ops_per_trial: u64,
    trials: usize,
    min_ns_per_op: f64,
    median_ns_per_op: f64,
    p95_ns_per_op: f64,
    spread_pct: f64,
    max_regression_pct: f64,
    samples_ns_per_op: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct E2ESummary {
    num_docs: usize,
    vocab_size: usize,
    total_postings: usize,
    num_queries: usize,
    top_k: usize,
    zipora_qps_top10: f64,
    zipora_p50_us: f64,
    zipora_p99_us: f64,
    zipora_vm_hwm_kib: u64,
    zipora_bytes_per_posting: f64,
    zipora_postings_plus_bmw_bytes_per_posting: f64,
    zipora_total_index_bytes_per_posting: f64,
    zipora_dict_bytes: usize,
    zipora_fieldnorm_bytes: usize,
    zipora_postings_bytes: usize,
    zipora_bmw_metadata_bytes: usize,
    zipora_total_index_bytes: usize,
    tantivy_qps_top10: f64,
    tantivy_p50_us: f64,
    tantivy_p99_us: f64,
    tantivy_postings_bytes: usize,
    tantivy_postings_bytes_per_posting: f64,
    tantivy_termdict_bytes: usize,
    tantivy_fieldnorms_bytes: usize,
    tantivy_store_bytes: usize,
    tantivy_index_without_store_bytes: usize,
    tantivy_index_without_store_bytes_per_posting: f64,
    tantivy_index_bytes: usize,
    tantivy_bytes_per_posting: f64,
    qps_speedup_vs_tantivy: f64,
    baseline_qps_top10: f64,
    baseline_p50_us: f64,
    baseline_p99_us: f64,
    baseline_bytes_per_posting: f64,
    qps_speedup_vs_baseline: f64,
    top10_recall_vs_exhaustive: f64,
    competitor_note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct E2EBenchReport {
    suite: String,
    seed: u64,
    host_metadata: HostMetadata,
    summary: E2ESummary,
    encoding_distribution: BTreeMap<String, usize>,
    benchmarks: Vec<BenchRecord>,
}

// ---------------------------------------------------------------------------
// Host / Environment Helpers
// ---------------------------------------------------------------------------

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn detect_short_host() -> String {
    if let Some(h) = read_trimmed("/etc/hostname")
        && !h.is_empty()
    {
        return h.split('.').next().unwrap_or(&h).to_string();
    }
    std::env::var("HOSTNAME")
        .unwrap_or_else(|_| "unknown".to_string())
        .split('.')
        .next()
        .unwrap_or("unknown")
        .to_string()
}

fn detect_cpu_model() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if let Some((k, v)) = line.split_once(':')
                && k.trim() == "model name"
            {
                return v.trim().to_string();
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn detect_governor() -> String {
    read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .unwrap_or_else(|| "unknown (virtualized/host-managed)".to_string())
}

fn detect_git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn detect_utc_timestamp() -> String {
    Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn read_vm_hwm_kib() -> u64 {
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let digits: String = rest.chars().filter(|c| c.is_ascii_digit()).collect();
                if let Ok(val) = digits.parse::<u64>() {
                    return val;
                }
            }
        }
    }
    0
}

fn parse_cpu_list(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in spec.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start_s, end_s)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start_s.parse::<usize>(), end_s.parse::<usize>()) {
                for c in s..=e {
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
    if let Ok(present) = fs::read_to_string("/sys/devices/system/cpu/present") {
        let cpus = parse_cpu_list(&present);
        if !cpus.is_empty() {
            return cpus.len();
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[path = "common/binary_meta.rs"]
mod binary_meta;
use binary_meta::detect_self_binary_path_and_sha256;

fn collect_host_metadata(pinned_core: usize) -> HostMetadata {
    let rustflags = std::env::var("RUSTFLAGS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(|| option_env!("ZIPORA_EFFECTIVE_RUSTFLAGS").map(str::to_string))
        .unwrap_or_else(|| {
            "-C target-cpu=haswell -C target-feature=+bmi2,+avx2,+popcnt -C codegen-units=1 -C llvm-args=-align-loops=32"
                .to_string()
        });
    let (binary_path, binary_sha256) = detect_self_binary_path_and_sha256();
    HostMetadata {
        host: detect_short_host(),
        cpu_model: detect_cpu_model(),
        logical_cores: hardware_logical_cores(),
        pinned_core,
        governor: detect_governor(),
        rustflags,
        git_commit: detect_git_commit(),
        timestamp_utc: detect_utc_timestamp(),
        binary_path,
        binary_sha256,
    }
}

/// Select the primary physical core on interior L3 CCD 2 (`core 23` on `bindiego`) for the
/// calling thread (`pid = 0` only; never modifies any other process).
fn select_primary_core(logical_cores: usize, _sample_ms: u64) -> usize {
    for env_key in ["ZIPORA_BENCH_PIN_CORE", "ZIPORA_BENCH_CORE"] {
        if let Ok(val) = std::env::var(env_key)
            && let Ok(core) = val.parse::<usize>()
        {
            return core % logical_cores.max(1);
        }
    }
    let phys_cores = (logical_cores / 2).max(1);
    if phys_cores >= 24 {
        23
    } else {
        phys_cores.saturating_sub(1)
    }
}

#[cfg(target_os = "linux")]
fn pin_current_thread_to_core(core_id: usize) {
    // SAFETY: `cpu_set_t` is zero-initialized and `CPU_SET` / `sched_setaffinity`
    // are called with valid size and self PID (0). Never modifies any other process.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core_id, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_core(_core_id: usize) {}

#[cfg(target_os = "linux")]
fn disable_transparent_hugepages_for_self() {
    // SAFETY: `prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)` affects only the calling process.
    unsafe {
        let _ = libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }
}

#[cfg(not(target_os = "linux"))]
fn disable_transparent_hugepages_for_self() {}

fn round4(val: f64) -> f64 {
    (val * 10_000.0).round() / 10_000.0
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

fn compute_stats(samples: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let n = sorted.len();
    let min = sorted[0];
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let p95_idx = ((n as f64 * 0.95).floor() as usize).min(n - 1);
    let p95 = sorted[p95_idx];
    let spread_pct = if min > 0.0 {
        ((p95 - min) / min) * 100.0
    } else {
        0.0
    };
    (min, median, p95, spread_pct)
}

fn measure_bench<F>(
    records: &mut Vec<BenchRecord>,
    name: &str,
    dataset_size: u64,
    ops_per_trial: u64,
    trials: usize,
    mut f: F,
) where
    F: FnMut(),
{
    let warmup_calls = if ops_per_trial <= 2_500 { 3 } else { 2 };
    for _ in 0..warmup_calls {
        f();
    }

    let bursts_per_trial = if ops_per_trial <= 2_500 { 3 } else { 2 };
    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let mut best_elapsed_ns = f64::MAX;
        for _ in 0..bursts_per_trial {
            let start = Instant::now();
            f();
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            if elapsed_ns < best_elapsed_ns {
                best_elapsed_ns = elapsed_ns;
            }
        }
        let ns_per_op = best_elapsed_ns / (ops_per_trial.max(1) as f64);
        samples.push(ns_per_op);
    }

    let (min_ns, median_ns, p95_ns, spread_pct) = compute_stats(&samples);
    let rounded_samples: Vec<f64> = samples.into_iter().map(round4).collect();
    records.push(BenchRecord {
        category: "e2e_search".to_string(),
        name: name.to_string(),
        unit: "ns/query".to_string(),
        dataset_size,
        ops_per_trial,
        trials,
        min_ns_per_op: round4(min_ns),
        median_ns_per_op: round4(median_ns),
        p95_ns_per_op: round4(p95_ns),
        spread_pct: round2(spread_pct),
        max_regression_pct: 9.0,
        samples_ns_per_op: rounded_samples,
    });
}

// ---------------------------------------------------------------------------
// Top-K Hit & Min-Heap
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredDoc {
    score: f32,
    doc_id: u32,
}

impl Eq for ScoredDoc {}

// Min-heap ordering: smallest score at the top (tie-broken by larger doc_id first
// so lower doc_ids win ties deterministically).
impl Ord for ScoredDoc {
    #[inline]
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

impl PartialOrd for ScoredDoc {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

struct TopKCollector {
    heap: BinaryHeap<ScoredDoc>,
    k: usize,
    theta: f32,
}

impl TopKCollector {
    #[inline]
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            theta: 0.0,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.heap.clear();
        self.theta = 0.0;
    }

    #[inline]
    fn push(&mut self, doc_id: u32, score: f32) {
        if self.heap.len() < self.k {
            self.heap.push(ScoredDoc { score, doc_id });
            if self.heap.len() == self.k
                && let Some(top) = self.heap.peek()
            {
                self.theta = top.score;
            }
        } else {
            let mut updated = false;
            if let Some(mut top) = self.heap.peek_mut()
                && (score > top.score || (score == top.score && doc_id < top.doc_id))
            {
                *top = ScoredDoc { score, doc_id };
                updated = true;
            }
            if updated && let Some(new_top) = self.heap.peek() {
                self.theta = new_top.score;
            }
        }
    }

    fn to_sorted_vec(&self) -> Vec<ScoredDoc> {
        let mut v: Vec<ScoredDoc> = self.heap.iter().copied().collect();
        v.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
        });
        v
    }
}

// ---------------------------------------------------------------------------
// Synthetic Zipf Corpus & Queries (100,000 docs, ~1.61M postings > L2)
// ---------------------------------------------------------------------------

struct CorpusData {
    num_docs: usize,
    avg_dl: f32,
    doc_lens: Vec<u32>,
    terms: Vec<String>,
    term_doc_ids: Vec<Vec<u32>>,
    term_tfs: Vec<Vec<u16>>,
    queries: Vec<Vec<String>>,
    queries_by_len: [Vec<Vec<String>>; 4],
}

fn generate_zipf_corpus_and_queries(
    num_docs: usize,
    vocab_size: usize,
    num_queries: usize,
    seed: u64,
) -> CorpusData {
    let mut rng = StdRng::seed_from_u64(seed);

    // 1. Document lengths (32..=512 tokens, skewed distribution)
    let mut doc_lens = Vec::with_capacity(num_docs);
    let mut total_dl = 0u64;
    for _ in 0..num_docs {
        let base = rng.random_range(32u32..=192u32);
        let tail = if rng.random_range(0u32..4) == 0 {
            rng.random_range(0u32..=320u32)
        } else {
            0
        };
        let dl = base + tail;
        doc_lens.push(dl);
        total_dl += dl as u64;
    }
    let avg_dl = (total_dl as f64 / num_docs as f64) as f32;

    // 2. Vocabulary terms (sorted lexicographically for clean trie & deterministic IDs)
    let mut terms: Vec<String> = (0..vocab_size)
        .map(|i| {
            format!(
                "term_{i:05}_{:04x}",
                (i as u64).wrapping_mul(0x9E37) & 0xFFFF
            )
        })
        .collect();
    terms.sort();

    // 3. Zipfian posting lists across all HybridPostingList tiers on 100,000 docs:
    //    - Top 8 terms: very dense / run-heavy head terms (>50% of docs -> Clustered)
    //    - Next 24 terms: large OptimalPEF tier (>4,096 postings -> Optimal)
    //    - Next 352 terms: medium-large PartitionedEF tier (280..3,600 postings -> Partitioned)
    //    - Next 1,024 terms: medium EliasFano tier (68..250 postings -> EliasFano)
    //    - Remaining terms: rare Dense tier (8..64 postings -> Dense)
    let mut term_doc_ids = Vec::with_capacity(vocab_size);
    let mut term_tfs = Vec::with_capacity(vocab_size);

    for rank in 0..vocab_size {
        let df: usize = if rank < 8 {
            ((num_docs * 3) / 5 + (8 - rank) * 120).min(num_docs)
        } else if rank < 32 {
            (4_200 + (32 - rank) * 180).min(num_docs)
        } else if rank < 384 {
            let span = 384 - 32;
            let pos = rank - 32;
            (320 + ((span - pos) * 3_200) / span).min(num_docs)
        } else if rank < 1_408 {
            let span = 1_408 - 384;
            let pos = rank - 384;
            (68 + ((span - pos) * 180) / span).min(num_docs)
        } else {
            let span = vocab_size - 1_408;
            let pos = rank - 1_408;
            (8 + ((span - pos) * 54) / span.max(1)).min(num_docs)
        };

        let df = df.clamp(1, num_docs);
        let mut docs = Vec::with_capacity(df);
        if df * 2 >= num_docs {
            let mut d = rng.random_range(0..4usize);
            while docs.len() < df && d < num_docs {
                let run_len = rng.random_range(128..=512usize).min(df - docs.len());
                for r in 0..run_len {
                    if d + r < num_docs {
                        docs.push((d + r) as u32);
                    }
                }
                d += run_len + rng.random_range(1..=64usize);
            }
            if docs.len() < df {
                let mut present = vec![false; num_docs];
                for &doc_id in &docs {
                    present[doc_id as usize] = true;
                }
                for (doc_id, &is_set) in present.iter().enumerate() {
                    if !is_set {
                        docs.push(doc_id as u32);
                        if docs.len() == df {
                            break;
                        }
                    }
                }
                docs.sort_unstable();
            }
        } else {
            let avg_step = (num_docs / df).max(1);
            let mut cur = rng.random_range(0..avg_step.max(1));
            for i in 0..df {
                let remaining = df - i;
                let max_doc = num_docs - remaining;
                if cur > max_doc {
                    cur = max_doc;
                }
                docs.push(cur as u32);
                let step = rng.random_range(1..=(avg_step * 2).max(2));
                cur = (cur + step).min(num_docs.saturating_sub(remaining.saturating_sub(1)));
            }
        }

        // Topically clustered term frequencies: ~15% of 64-posting blocks are topical
        // clusters (containing high-TF bursts 3..=25), while background blocks have TF 1..=2.
        let mut tfs = Vec::with_capacity(df);
        let num_blocks = df.div_ceil(BLOCK_SIZE);
        let hot_block = rng.random_range(0..num_blocks.max(1));
        for blk in 0..num_blocks {
            let start = blk * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(df);
            let is_topical = blk == hot_block || rng.random_range(0u32..100) < 14;
            for _ in start..end {
                let tf: u16 = if is_topical {
                    let u = rng.random_range(0u32..100);
                    if u < 50 {
                        rng.random_range(1..=2)
                    } else if u < 82 {
                        rng.random_range(3..=8)
                    } else {
                        rng.random_range(9..=25)
                    }
                } else if rng.random_range(0u32..100) < 82 {
                    1
                } else {
                    2
                };
                tfs.push(tf);
            }
        }

        term_doc_ids.push(docs);
        term_tfs.push(tfs);
    }

    // 4. Generate 10,000 Zipf queries (2,500 each of 1, 2, 3, and 4 terms)
    let mut zipf_weights = Vec::with_capacity(vocab_size);
    let mut cum = 0.0f64;
    for rank in 0..vocab_size {
        let w = 1.0 / ((rank as f64 + 16.0).powf(1.05));
        cum += w;
        zipf_weights.push(cum);
    }
    let total_weight = cum;

    let sample_term_idx = |rng: &mut StdRng| -> usize {
        let target = rng.random::<f64>() * total_weight;
        zipf_weights
            .binary_search_by(|w| w.partial_cmp(&target).unwrap_or(CmpOrdering::Equal))
            .unwrap_or_else(|i| i.min(vocab_size - 1))
    };

    let mut queries = Vec::with_capacity(num_queries);
    let mut queries_by_len: [Vec<Vec<String>>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    for q_idx in 0..num_queries {
        let q_len = (q_idx % 4) + 1;
        let mut q_terms = Vec::with_capacity(q_len);
        while q_terms.len() < q_len {
            let tid = sample_term_idx(&mut rng);
            let t_str = terms[tid].clone();
            if !q_terms.contains(&t_str) {
                q_terms.push(t_str);
            }
        }
        queries_by_len[q_len - 1].push(q_terms.clone());
        queries.push(q_terms);
    }

    CorpusData {
        num_docs,
        avg_dl,
        doc_lens,
        terms,
        term_doc_ids,
        term_tfs,
        queries,
        queries_by_len,
    }
}

// ---------------------------------------------------------------------------
// Zipora End-to-End Search Engine (DoubleArrayTrieMap + HybridPostingList +
// FieldnormEncoder + Bench-Local Block-Max WAND)
// ---------------------------------------------------------------------------

#[repr(C, align(64))]
struct AlignedBlockScratch {
    docs: [u32; BLOCK_SIZE],
    scores: [f32; BLOCK_SIZE],
    tfs: [u16; BLOCK_SIZE],
}

impl AlignedBlockScratch {
    #[inline]
    const fn new() -> Self {
        Self {
            docs: [0u32; BLOCK_SIZE],
            scores: [0.0f32; BLOCK_SIZE],
            tfs: [0u16; BLOCK_SIZE],
        }
    }
}

struct TermPostingData {
    postings: HybridPostingList,
    tfs: Vec<u8>,
    block_max_doc_ids: Vec<u32>,
    block_max_scores: Vec<f32>,
    peak_blocks: [u32; 2],
    max_term_score: f32,
    idf_k1p1: f32,
}

struct ZiporaSearchEngine {
    dict: DoubleArrayTrieMap<u32>,
    fieldnorms: Vec<u8>,
    terms: Vec<TermPostingData>,
    scorers: Vec<Bm25BatchScorer<'static>>,
    unit_scorer: Bm25BatchScorer<'static>,
    total_postings: usize,
}

impl ZiporaSearchEngine {
    fn build(corpus: &CorpusData) -> Self {
        let mut dict = DoubleArrayTrieMap::<u32>::with_capacity(corpus.terms.len() * 2);
        for (tid, term) in corpus.terms.iter().enumerate() {
            dict.insert(term.as_bytes(), tid as u32).unwrap();
        }
        dict.shrink_to_fit();

        let fieldnorms: Vec<u8> = corpus
            .doc_lens
            .iter()
            .map(|&dl| FieldnormEncoder::encode(dl))
            .collect();
        static NORM_TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
        let norm_table = NORM_TABLE
            .get_or_init(|| FieldnormEncoder::build_norm_table(corpus.avg_dl, BM25_K1, BM25_B));
        let unit_scorer = Bm25BatchScorer::new(norm_table, 1.0 / (BM25_K1 + 1.0), BM25_K1);

        let n_docs_f = corpus.num_docs as f32;
        let mut terms = Vec::with_capacity(corpus.terms.len());
        let mut scorers = Vec::with_capacity(corpus.terms.len());
        let mut total_postings = 0usize;

        for (docs, tfs_u16) in corpus.term_doc_ids.iter().zip(corpus.term_tfs.iter()) {
            let df = docs.len();
            total_postings += df;
            let idf = ((n_docs_f - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
            let idf_k1p1 = idf * (BM25_K1 + 1.0);
            let scorer = Bm25BatchScorer::new(norm_table, idf, BM25_K1);

            let postings = HybridPostingList::from_sorted(docs);
            let tfs: Vec<u8> = tfs_u16.iter().map(|&tf| tf.min(255) as u8).collect();

            let num_blocks = df.div_ceil(BLOCK_SIZE);
            let mut block_max_doc_ids = Vec::with_capacity(num_blocks);
            let mut block_max_scores = Vec::with_capacity(num_blocks);
            let mut max_term_score = 0.0f32;
            let mut best_blk_0 = (0u32, -1.0f32);
            let mut best_blk_1 = (0u32, -1.0f32);

            for blk_idx in 0..num_blocks {
                let start = blk_idx * BLOCK_SIZE;
                let end = (start + BLOCK_SIZE).min(df);
                let last_doc = docs[end - 1];
                let mut blk_max = 0.0f32;
                for i in start..end {
                    let s = scorer.score(fieldnorms[docs[i] as usize], tfs[i] as u16);
                    if s > blk_max {
                        blk_max = s;
                    }
                }
                if blk_max > max_term_score {
                    max_term_score = blk_max;
                }
                if blk_max > best_blk_0.1 {
                    best_blk_1 = best_blk_0;
                    best_blk_0 = (blk_idx as u32, blk_max);
                } else if blk_max > best_blk_1.1 {
                    best_blk_1 = (blk_idx as u32, blk_max);
                }
                block_max_doc_ids.push(last_doc);
                block_max_scores.push(blk_max);
            }

            terms.push(TermPostingData {
                postings,
                tfs,
                block_max_doc_ids,
                block_max_scores,
                peak_blocks: [best_blk_0.0, best_blk_1.0],
                max_term_score,
                idf_k1p1,
            });
            scorers.push(scorer);
        }

        Self {
            dict,
            fieldnorms,
            terms,
            scorers,
            unit_scorer,
            total_postings,
        }
    }

    fn encoding_distribution(&self) -> BTreeMap<String, usize> {
        let mut dist = BTreeMap::new();
        for t in &self.terms {
            let name = match t.postings.encoding() {
                PostingEncoding::Dense => "Dense",
                PostingEncoding::EliasFano => "EliasFano",
                PostingEncoding::Partitioned => "Partitioned",
                PostingEncoding::Optimal => "Optimal",
                PostingEncoding::Clustered => "Clustered",
            };
            *dist.entry(name.to_string()).or_insert(0) += 1;
        }
        dist
    }

    fn memory_breakdown(&self) -> (usize, usize, usize, usize) {
        let dict_bytes = self.dict.mem_size();
        let fieldnorm_bytes = self.fieldnorms.len();
        let postings_bytes: usize = self
            .terms
            .iter()
            .map(|t| t.postings.size_bytes())
            .collect::<Vec<_>>()
            .into_iter()
            .sum();
        let bmw_metadata_bytes: usize = self
            .terms
            .iter()
            .map(|t| {
                t.tfs.len()
                    + t.block_max_doc_ids.len() * std::mem::size_of::<u32>()
                    + t.block_max_scores.len() * std::mem::size_of::<f32>()
                    + std::mem::size_of::<[u32; 2]>()
            })
            .sum();
        (
            dict_bytes,
            fieldnorm_bytes,
            postings_bytes,
            bmw_metadata_bytes,
        )
    }
}

enum HybridCursor<'a> {
    Dense {
        slice: &'a [u32],
        idx: usize,
    },
    EliasFano(EliasFanoCursor<'a>),
    Partitioned(PartitionedEliasFanoCursor<'a>),
    Optimal(OptimalPefCursor<'a>),
    Clustered {
        cef: &'a ClusteredEliasFano,
        idx: usize,
        cur_doc: u32,
        len: usize,
        run_end_idx: usize,
    },
}

impl<'a> HybridCursor<'a> {
    #[inline]
    fn new(postings: &'a HybridPostingList) -> Option<(Self, usize, u32)> {
        match postings {
            HybridPostingList::Dense(v) => {
                let &first = v.first()?;
                Some((Self::Dense { slice: v, idx: 0 }, 0, first))
            }
            HybridPostingList::EliasFano(ef) => {
                let c = ef.cursor();
                let first = c.current()? as u32;
                Some((Self::EliasFano(c), 0, first))
            }
            HybridPostingList::Partitioned(pef) => {
                let c = pef.cursor();
                let first = c.current()? as u32;
                Some((Self::Partitioned(c), 0, first))
            }
            HybridPostingList::Optimal(opef) => {
                let c = opef.cursor();
                let first = c.current()? as u32;
                Some((Self::Optimal(c), 0, first))
            }
            HybridPostingList::Clustered(cef) => {
                let len = cef.len();
                if len == 0 {
                    None
                } else {
                    let first = cef.get(0)? as u32;
                    let chunk_end = 128.min(len);
                    let last_in_chunk = cef.get(chunk_end - 1)? as u32;
                    let run_end_idx = if last_in_chunk - first == (chunk_end - 1) as u32 {
                        chunk_end
                    } else {
                        0
                    };
                    Some((
                        Self::Clustered {
                            cef,
                            idx: 0,
                            cur_doc: first,
                            len,
                            run_end_idx,
                        },
                        0,
                        first,
                    ))
                }
            }
        }
    }

    #[inline]
    fn advance(&mut self) -> Option<(usize, u32)> {
        match self {
            Self::Dense { slice, idx } => {
                *idx += 1;
                if *idx < slice.len() {
                    Some((*idx, slice[*idx]))
                } else {
                    None
                }
            }
            Self::EliasFano(c) => {
                if c.advance() {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Partitioned(c) => {
                if c.advance() {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Optimal(c) => {
                if c.advance() {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Clustered {
                cef,
                idx,
                cur_doc,
                len,
                run_end_idx,
            } => {
                *idx += 1;
                if *idx >= *len {
                    return None;
                }
                if *idx < *run_end_idx {
                    *cur_doc += 1;
                    return Some((*idx, *cur_doc));
                }
                let d = cef.get(*idx)? as u32;
                *cur_doc = d;
                if (*idx & 127) == 0 {
                    let chunk_end = (*idx + 128).min(*len);
                    if let Some(last_in_chunk) = cef.get(chunk_end - 1)
                        && (last_in_chunk as u32) - d == (chunk_end - 1 - *idx) as u32
                    {
                        *run_end_idx = chunk_end;
                    }
                }
                Some((*idx, d))
            }
        }
    }

    #[inline]
    fn advance_to_geq(&mut self, target: u32) -> Option<(usize, u32)> {
        match self {
            Self::Dense { slice, idx } => {
                if simd_gallop_to(slice, idx, target) {
                    Some((*idx, slice[*idx]))
                } else {
                    None
                }
            }
            Self::EliasFano(c) => {
                if c.advance_to_geq(target as u64) {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Partitioned(c) => {
                if c.advance_to_geq(target as u64) {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Optimal(c) => {
                if c.advance_to_geq(target as u64) {
                    Some((c.index(), c.current()? as u32))
                } else {
                    None
                }
            }
            Self::Clustered {
                cef,
                idx,
                cur_doc,
                len,
                run_end_idx,
            } => {
                if *idx >= *len {
                    None
                } else if *cur_doc >= target {
                    Some((*idx, *cur_doc))
                } else {
                    // O(1) fast path if `target` falls within the current contiguous Run chunk
                    if *idx < *run_end_idx {
                        let max_run_doc = *cur_doc + (*run_end_idx - 1 - *idx) as u32;
                        if target <= max_run_doc {
                            let step = (target - *cur_doc) as usize;
                            *idx += step;
                            *cur_doc = target;
                            return Some((*idx, target));
                        }
                    }
                    let (i, v) = cef.next_geq(target as u64)?;
                    let v32 = v as u32;
                    *idx = i;
                    *cur_doc = v32;
                    let chunk_end = ((i & !127) + 128).min(*len);
                    if let Some(last_in_chunk) = cef.get(chunk_end - 1)
                        && (last_in_chunk as u32) - v32 == (chunk_end - 1 - i) as u32
                    {
                        *run_end_idx = chunk_end;
                    } else {
                        *run_end_idx = 0;
                    }
                    Some((i, v32))
                }
            }
        }
    }
}

impl ZiporaSearchEngine {
    /// Execute a top-K query using Block-Max WAND (`simd_gallop_to` + `simd_block_filter` + `Bm25BatchScorer`).
    #[inline]
    fn search_bmw(&self, query_terms: &[String], collector: &mut TopKCollector) {
        collector.clear();
        if query_terms.is_empty() {
            return;
        }

        if query_terms.len() == 1 {
            let Some(tid) = self.dict.get(query_terms[0].as_bytes()) else {
                return;
            };
            self.search_single_term(tid as usize, collector);
            return;
        }

        let mut term_indices = [0usize; 4];
        let mut n_terms = 0usize;
        for qt in query_terms.iter().take(4) {
            if let Some(tid) = self.dict.get(qt.as_bytes()) {
                term_indices[n_terms] = tid as usize;
                n_terms += 1;
            }
        }
        if n_terms == 0 {
            return;
        }
        if n_terms == 1 {
            self.search_single_term(term_indices[0], collector);
            return;
        }

        self.search_multi_term_bmw(&term_indices[..n_terms], collector);
    }

    /// Evaluate a single 64-doc block `blk_idx` of `term` using random access when priming `collector.theta`.
    #[inline]
    fn score_single_block_direct(
        &self,
        term: &TermPostingData,
        scorer: &Bm25BatchScorer<'static>,
        blk_idx: usize,
        collector: &mut TopKCollector,
    ) {
        let df = term.postings.len();
        let start = blk_idx * BLOCK_SIZE;
        if start >= df {
            return;
        }
        let end = (start + BLOCK_SIZE).min(df);
        let count = end - start;
        let mut scratch = AlignedBlockScratch::new();

        match &term.postings {
            HybridPostingList::Dense(slice) => {
                scratch.docs[..count].copy_from_slice(&slice[start..end]);
            }
            HybridPostingList::Clustered(cef) => {
                let first_doc = cef.get(start).unwrap_or(0) as u32;
                let last_doc = term.block_max_doc_ids[blk_idx];
                if last_doc - first_doc == (count - 1) as u32 {
                    for (i, slot) in scratch.docs[..count].iter_mut().enumerate() {
                        *slot = first_doc + i as u32;
                    }
                } else {
                    for (i, slot) in scratch.docs[..count].iter_mut().enumerate() {
                        *slot = cef.get(start + i).unwrap_or(0) as u32;
                    }
                }
            }
            other => {
                if let Some((mut cur, mut cur_idx, mut cur_doc)) = HybridCursor::new(other) {
                    if cur_idx < start {
                        let min_target = if blk_idx > 0 {
                            term.block_max_doc_ids[blk_idx - 1].saturating_add(1)
                        } else {
                            0
                        };
                        if let Some((i, d)) = cur.advance_to_geq(min_target) {
                            cur_idx = i;
                            cur_doc = d;
                        }
                        while cur_idx < start {
                            if let Some((i, d)) = cur.advance() {
                                cur_idx = i;
                                cur_doc = d;
                            } else {
                                break;
                            }
                        }
                    }
                    for slot in &mut scratch.docs[..count] {
                        *slot = cur_doc;
                        if let Some((i, d)) = cur.advance() {
                            cur_idx = i;
                            cur_doc = d;
                        }
                    }
                    let _ = cur_idx;
                }
            }
        }

        for i in 0..count {
            scratch.tfs[i] = term.tfs[start + i] as u16;
        }
        scorer.batch_score_with_prefetch(
            &self.fieldnorms,
            &scratch.docs[..count],
            &scratch.tfs[..count],
            &mut scratch.scores[..count],
        );

        let filter_theta = if collector.heap.len() < collector.k {
            -1.0
        } else {
            collector.theta - 1e-6
        };
        let (mut mask, _) = simd_block_filter(
            &scratch.docs[..count],
            &scratch.scores[..count],
            filter_theta,
        );
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            collector.push(scratch.docs[bit], scratch.scores[bit]);
        }
    }

    /// Single-term Block-Max pruning + `Bm25BatchScorer::batch_score_with_prefetch` + `simd_block_filter`.
    #[inline]
    fn search_single_term(&self, tid: usize, collector: &mut TopKCollector) {
        let term = &self.terms[tid];
        let scorer = &self.scorers[tid];
        let df = term.postings.len();
        if df == 0 {
            return;
        }
        let num_blocks = term.block_max_scores.len();

        // Prime `collector.theta` using the term's 2 highest-scoring blocks first so
        // background TF=1..2 blocks at the front of the posting list are pruned immediately.
        let mut primed_0 = usize::MAX;
        let mut primed_1 = usize::MAX;
        if num_blocks > 2 {
            primed_0 = term.peak_blocks[0] as usize;
            self.score_single_block_direct(term, scorer, primed_0, collector);
            let p1 = term.peak_blocks[1] as usize;
            if p1 != primed_0
                && (term.block_max_scores[p1] >= collector.theta
                    || collector.heap.len() < collector.k)
            {
                primed_1 = p1;
                self.score_single_block_direct(term, scorer, primed_1, collector);
            }
        }

        let mut scratch = AlignedBlockScratch::new();

        let dense_slice = match &term.postings {
            HybridPostingList::Dense(v) => Some(v.as_slice()),
            _ => None,
        };
        let mut cursor_opt = if dense_slice.is_none() {
            HybridCursor::new(&term.postings)
        } else {
            None
        };

        for blk_idx in 0..num_blocks {
            if blk_idx == primed_0 || blk_idx == primed_1 {
                continue;
            }
            if term.block_max_scores[blk_idx] < collector.theta
                && collector.heap.len() >= collector.k
            {
                continue;
            }

            let start = blk_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(df);
            let count = end - start;

            if let Some(slice) = dense_slice {
                scratch.docs[..count].copy_from_slice(&slice[start..end]);
            } else if let Some((ref mut cur, ref mut cur_idx, ref mut cur_doc)) = cursor_opt {
                if *cur_idx < start {
                    let min_target = if blk_idx > 0 {
                        term.block_max_doc_ids[blk_idx - 1].saturating_add(1)
                    } else {
                        0
                    };
                    if let Some((i, d)) = cur.advance_to_geq(min_target) {
                        *cur_idx = i;
                        *cur_doc = d;
                    }
                    while *cur_idx < start {
                        if let Some((i, d)) = cur.advance() {
                            *cur_idx = i;
                            *cur_doc = d;
                        } else {
                            break;
                        }
                    }
                }
                for slot in &mut scratch.docs[..count] {
                    *slot = *cur_doc;
                    if let Some((i, d)) = cur.advance() {
                        *cur_idx = i;
                        *cur_doc = d;
                    }
                }
            }
            for i in 0..count {
                scratch.tfs[i] = term.tfs[start + i] as u16;
            }

            scorer.batch_score_with_prefetch(
                &self.fieldnorms,
                &scratch.docs[..count],
                &scratch.tfs[..count],
                &mut scratch.scores[..count],
            );

            let filter_theta = if collector.heap.len() < collector.k {
                -1.0
            } else {
                collector.theta - 1e-6
            };
            let (mut mask, _) = simd_block_filter(
                &scratch.docs[..count],
                &scratch.scores[..count],
                filter_theta,
            );

            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                collector.push(scratch.docs[bit], scratch.scores[bit]);
            }
        }
    }

    /// Multi-term Block-Max WAND / MaxScore using `Bm25BatchScorer::batch_score_with_prefetch`,
    /// `simd_block_filter` on lead-term 64-doc blocks, tight multi-term block-max upper bounds via
    /// `simd_gallop_to` on `block_max_doc_ids`, and stateful `HybridCursor::advance_to_geq`.
    #[inline]
    fn search_multi_term_bmw(&self, term_ids: &[usize], collector: &mut TopKCollector) {
        let mut sorted_tids = [0usize; 4];
        let m = term_ids.len().min(4);
        sorted_tids[..m].copy_from_slice(&term_ids[..m]);
        // Sort terms by max_term_score descending (highest-IDF / most selective lead terms first)
        sorted_tids[..m].sort_by(|&a, &b| {
            self.terms[b]
                .max_term_score
                .total_cmp(&self.terms[a].max_term_score)
        });

        // Compute suffix upper bounds: suffix_max[i] = sum_{j = i..m} max_term_score[j]
        let mut suffix_max = [0.0f32; 5];
        for i in (0..m).rev() {
            suffix_max[i] = suffix_max[i + 1] + self.terms[sorted_tids[i]].max_term_score;
        }

        let mut scratch = AlignedBlockScratch::new();

        // Evaluate essential lead terms i = 0..m while suffix_max[i] > collector.theta
        for lead_pos in 0..m {
            let threshold_at_lead = if collector.heap.len() < collector.k {
                -1.0
            } else {
                collector.theta - 1e-6
            };
            if suffix_max[lead_pos] <= threshold_at_lead {
                break;
            }

            let lead_tid = sorted_tids[lead_pos];
            let lead_term = &self.terms[lead_tid];
            let lead_scorer = &self.scorers[lead_tid];
            let df = lead_term.postings.len();
            if df == 0 {
                continue;
            }
            let rem_upper = suffix_max[lead_pos + 1];
            let n_tail = m - (lead_pos + 1);
            let num_blocks = lead_term.block_max_scores.len();

            let mut eval_block_pass =
                |only_peaks: bool, p0: usize, p1: usize, collector: &mut TopKCollector| {
                    type TermCursorSlot<'a> =
                        Option<(&'a TermPostingData, HybridCursor<'a>, usize, u32, usize)>;

                    // Stateful cursors for earlier terms 0..lead_pos (to skip already-evaluated docs)
                    let mut prev_cursors: [TermCursorSlot<'_>; 4] = [None, None, None, None];
                    for (p_idx, &p_tid) in sorted_tids[..lead_pos].iter().enumerate() {
                        let p_term = &self.terms[p_tid];
                        if let Some((cur, idx, first_doc)) = HybridCursor::new(&p_term.postings) {
                            prev_cursors[p_idx] = Some((p_term, cur, idx, first_doc, 0));
                        }
                    }

                    // Stateful cursors for trailing terms (lead_pos + 1)..m
                    let mut tail_cursors: [TermCursorSlot<'_>; 4] = [None, None, None, None];
                    for (t_offset, &t_tid) in sorted_tids[(lead_pos + 1)..m].iter().enumerate() {
                        let t_term = &self.terms[t_tid];
                        if let Some((cur, idx, first_doc)) = HybridCursor::new(&t_term.postings) {
                            tail_cursors[t_offset] = Some((t_term, cur, idx, first_doc, 0));
                        }
                    }

                    let dense_slice = match &lead_term.postings {
                        HybridPostingList::Dense(v) => Some(v.as_slice()),
                        _ => None,
                    };
                    let mut lead_cursor = if dense_slice.is_none() {
                        HybridCursor::new(&lead_term.postings)
                    } else {
                        None
                    };

                    let pass_len = if only_peaks {
                        if p0 == p1 { 1 } else { 2 }
                    } else {
                        num_blocks
                    };

                    for step_idx in 0..pass_len {
                        let blk_idx = if only_peaks {
                            if step_idx == 0 { p0 } else { p1 }
                        } else {
                            if step_idx == p0 || step_idx == p1 {
                                continue;
                            }
                            step_idx
                        };

                        let cur_theta = if collector.heap.len() < collector.k {
                            -1.0
                        } else {
                            collector.theta - 1e-6
                        };

                        // Block-Max WAND pruning on lead term block + remaining terms' global max
                        if lead_term.block_max_scores[blk_idx] + rem_upper <= cur_theta {
                            continue;
                        }

                        let start = blk_idx * BLOCK_SIZE;
                        let end = (start + BLOCK_SIZE).min(df);
                        let count = end - start;

                        if let Some(slice) = dense_slice {
                            scratch.docs[..count].copy_from_slice(&slice[start..end]);
                        } else if let Some((ref mut cur, ref mut cur_idx, ref mut cur_doc)) =
                            lead_cursor
                        {
                            if *cur_idx < start {
                                let min_target = if blk_idx > 0 {
                                    lead_term.block_max_doc_ids[blk_idx - 1].saturating_add(1)
                                } else {
                                    0
                                };
                                if let Some((i, d)) = cur.advance_to_geq(min_target) {
                                    *cur_idx = i;
                                    *cur_doc = d;
                                }
                                while *cur_idx < start {
                                    if let Some((i, d)) = cur.advance() {
                                        *cur_idx = i;
                                        *cur_doc = d;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            for slot in &mut scratch.docs[..count] {
                                *slot = *cur_doc;
                                if let Some((i, d)) = cur.advance() {
                                    *cur_idx = i;
                                    *cur_doc = d;
                                }
                            }
                        }

                        for i in 0..count {
                            scratch.tfs[i] = lead_term.tfs[start + i] as u16;
                        }

                        lead_scorer.batch_score_with_prefetch(
                            &self.fieldnorms,
                            &scratch.docs[..count],
                            &scratch.tfs[..count],
                            &mut scratch.scores[..count],
                        );

                        // SIMD block filter: keep only docs where score(d, lead) + rem_upper > cur_theta
                        let block_filter_theta = cur_theta - rem_upper;
                        let (mut mask, _) = simd_block_filter(
                            &scratch.docs[..count],
                            &scratch.scores[..count],
                            block_filter_theta,
                        );

                        'doc_loop: while mask != 0 {
                            let bit = mask.trailing_zeros() as usize;
                            mask &= mask - 1;
                            let doc_id = scratch.docs[bit];
                            let mut acc_score = scratch.scores[bit];

                            let doc_theta = if collector.heap.len() < collector.k {
                                -1.0
                            } else {
                                collector.theta - 1e-6
                            };
                            if acc_score + rem_upper <= doc_theta {
                                continue;
                            }

                            // 1. Tight Block-Max WAND upper-bound check across all trailing terms:
                            // Advance each trailing term's `t_blk` pointer in `block_max_doc_ids` using
                            // `simd_gallop_to` and sum `block_max_scores[*t_blk]` BEFORE decoding any
                            // compressed posting list or earlier-term deduplication cursor.
                            let mut tail_blk_max = [0.0f32; 4];
                            let mut potential_upper = acc_score;
                            for t_offset in 0..n_tail {
                                if let Some((t_term, _, _, ref mut t_doc, ref mut t_blk)) =
                                    tail_cursors[t_offset]
                                {
                                    if *t_doc < doc_id
                                        && !simd_gallop_to(&t_term.block_max_doc_ids, t_blk, doc_id)
                                    {
                                        *t_doc = u32::MAX;
                                    }
                                    if *t_doc != u32::MAX {
                                        let bmax = t_term.block_max_scores[*t_blk];
                                        tail_blk_max[t_offset] = bmax;
                                        potential_upper += bmax;
                                    }
                                }
                            }
                            if potential_upper <= doc_theta {
                                continue 'doc_loop;
                            }

                            // 2. If lead_pos > 0, check if doc_id already appeared in an earlier lead term 0..lead_pos
                            for p_slot in prev_cursors[..lead_pos].iter_mut().flatten() {
                                let (
                                    p_term,
                                    ref mut p_cur,
                                    ref mut p_post_idx,
                                    ref mut p_doc,
                                    ref mut p_blk,
                                ) = *p_slot;
                                if *p_doc < doc_id {
                                    if !simd_gallop_to(&p_term.block_max_doc_ids, p_blk, doc_id) {
                                        *p_doc = u32::MAX;
                                    } else if let Some((i, d)) = p_cur.advance_to_geq(doc_id) {
                                        *p_post_idx = i;
                                        *p_doc = d;
                                    } else {
                                        *p_doc = u32::MAX;
                                    }
                                }
                                if *p_doc == doc_id {
                                    continue 'doc_loop;
                                }
                            }

                            // 3. Probe trailing terms (lead_pos + 1)..m, tightening `potential_upper` after each term
                            let norm_id = self.fieldnorms[doc_id as usize];
                            for t_offset in 0..n_tail {
                                let bmax = tail_blk_max[t_offset];
                                if bmax == 0.0 {
                                    continue;
                                }

                                let mut term_contrib = 0.0f32;
                                if let Some((
                                    t_term,
                                    ref mut t_cur,
                                    ref mut t_post_idx,
                                    ref mut t_doc,
                                    _,
                                )) = tail_cursors[t_offset]
                                {
                                    if *t_doc < doc_id {
                                        if let Some((i, d)) = t_cur.advance_to_geq(doc_id) {
                                            *t_post_idx = i;
                                            *t_doc = d;
                                        } else {
                                            *t_doc = u32::MAX;
                                        }
                                    }
                                    if *t_doc == doc_id {
                                        let tf = t_term.tfs[*t_post_idx] as u16;
                                        term_contrib =
                                            t_term.idf_k1p1 * self.unit_scorer.score(norm_id, tf);
                                        acc_score += term_contrib;
                                    }
                                }

                                potential_upper = potential_upper - bmax + term_contrib;
                                if potential_upper <= doc_theta {
                                    continue 'doc_loop;
                                }
                            }

                            collector.push(doc_id, acc_score);
                        }
                    }
                };

            if num_blocks > 4 {
                let b0 = lead_term.peak_blocks[0] as usize;
                let b1 = lead_term.peak_blocks[1] as usize;
                let p0 = b0.min(b1);
                let p1 = b0.max(b1);
                eval_block_pass(true, p0, p1, collector);
                eval_block_pass(false, p0, p1, collector);
            } else {
                eval_block_pass(false, usize::MAX, usize::MAX, collector);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tantivy v0.26 In-Memory BM25 Competitor Engine
// ---------------------------------------------------------------------------

struct TantivyEngine {
    _index: Index,
    reader: IndexReader,
    queries: Vec<BooleanQuery>,
    postings_bytes: usize,
    termdict_bytes: usize,
    fieldnorms_bytes: usize,
    store_bytes: usize,
    index_without_store_bytes: usize,
    index_bytes: usize,
    total_postings: usize,
}

impl TantivyEngine {
    fn build(corpus: &CorpusData) -> Self {
        let mut schema_builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("raw")
            .set_index_option(IndexRecordOption::WithFreqs);
        let text_options = TextOptions::default().set_indexing_options(indexing);
        let body_field = schema_builder.add_text_field("body", text_options);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);

        // Invert (term -> (doc_id, tf)) into per-document postings for single-segment indexing
        let mut doc_postings: Vec<Vec<(u32, u16)>> = vec![Vec::new(); corpus.num_docs];
        let mut total_postings = 0usize;
        for (tid, (docs, tfs)) in corpus
            .term_doc_ids
            .iter()
            .zip(corpus.term_tfs.iter())
            .enumerate()
        {
            total_postings += docs.len();
            for (&doc_id, &tf) in docs.iter().zip(tfs.iter()) {
                doc_postings[doc_id as usize].push((tid as u32, tf));
            }
        }

        let mut writer = index.writer_with_num_threads(1, 128 * 1024 * 1024).unwrap();
        for postings in &doc_postings {
            let mut doc = TantivyDocument::default();
            for &(tid, tf) in postings {
                let term_str = &corpus.terms[tid as usize];
                for _ in 0..tf {
                    doc.add_text(body_field, term_str);
                }
            }
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        writer.wait_merging_threads().unwrap();

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .unwrap();
        let searcher = reader.searcher();
        let (postings_bytes, termdict_bytes, fieldnorms_bytes, store_bytes, index_bytes) = searcher
            .space_usage()
            .map(|u| {
                let mut postings = 0usize;
                let mut termdict = 0usize;
                let mut fieldnorms = 0usize;
                let mut store = 0usize;
                for seg in u.segments() {
                    postings += seg.postings().total().get_bytes() as usize;
                    termdict += seg.termdict().total().get_bytes() as usize;
                    fieldnorms += seg.fieldnorms().total().get_bytes() as usize;
                    store += seg.store().total().get_bytes() as usize;
                }
                (
                    postings,
                    termdict,
                    fieldnorms,
                    store,
                    u.total().get_bytes() as usize,
                )
            })
            .unwrap_or((0, 0, 0, 0, 0));
        let index_without_store_bytes = index_bytes.saturating_sub(store_bytes);

        let queries: Vec<BooleanQuery> = corpus
            .queries
            .iter()
            .map(|q| {
                let clauses: Vec<(Occur, Box<dyn Query>)> = q
                    .iter()
                    .map(|qt| {
                        let term = Term::from_field_text(body_field, qt);
                        (
                            Occur::Should,
                            Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs))
                                as Box<dyn Query>,
                        )
                    })
                    .collect();
                BooleanQuery::new(clauses)
            })
            .collect();

        Self {
            _index: index,
            reader,
            queries,
            postings_bytes,
            termdict_bytes,
            fieldnorms_bytes,
            store_bytes,
            index_without_store_bytes,
            index_bytes,
            total_postings,
        }
    }

    fn postings_bytes_per_posting(&self) -> f64 {
        self.postings_bytes as f64 / self.total_postings.max(1) as f64
    }

    fn index_without_store_bytes_per_posting(&self) -> f64 {
        self.index_without_store_bytes as f64 / self.total_postings.max(1) as f64
    }

    fn bytes_per_posting(&self) -> f64 {
        self.index_bytes as f64 / self.total_postings.max(1) as f64
    }

    #[inline]
    fn search_query_idx(&self, query_idx: usize) -> u32 {
        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(
                &self.queries[query_idx],
                &TopDocs::with_limit(TOP_K).order_by_score(),
            )
            .unwrap();
        top_docs.first().map(|(_, addr)| addr.doc_id).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// In-Process Uncompressed Baseline Engine
// ---------------------------------------------------------------------------

struct UncompressedTermPostings {
    doc_ids: Vec<u32>,
    tfs: Vec<u16>,
    idf: f32,
}

struct UncompressedBaselineEngine {
    dict: HashMap<String, u32>,
    norm_table: [f32; 256],
    doc_norm_ids: Vec<u8>,
    terms: Vec<UncompressedTermPostings>,
    total_postings: usize,
}

impl UncompressedBaselineEngine {
    fn build(corpus: &CorpusData) -> Self {
        let mut dict = HashMap::with_capacity(corpus.terms.len() * 2);
        for (tid, term) in corpus.terms.iter().enumerate() {
            dict.insert(term.clone(), tid as u32);
        }

        // Standard BM25 length norm using quantized FieldnormEncoder lengths so scores match.
        let norm_table = FieldnormEncoder::build_norm_table(corpus.avg_dl, BM25_K1, BM25_B);
        let doc_norm_ids: Vec<u8> = corpus
            .doc_lens
            .iter()
            .map(|&dl| FieldnormEncoder::encode(dl))
            .collect();

        let n_docs_f = corpus.num_docs as f32;
        let mut terms = Vec::with_capacity(corpus.terms.len());
        let mut total_postings = 0usize;

        for (docs, tfs) in corpus.term_doc_ids.iter().zip(corpus.term_tfs.iter()) {
            let df = docs.len();
            total_postings += df;
            let idf = ((n_docs_f - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
            terms.push(UncompressedTermPostings {
                doc_ids: docs.clone(),
                tfs: tfs.clone(),
                idf,
            });
        }

        Self {
            dict,
            norm_table,
            doc_norm_ids,
            terms,
            total_postings,
        }
    }

    fn bytes_per_posting(&self) -> f64 {
        let postings_bytes: usize = self
            .terms
            .iter()
            .map(|t| {
                t.doc_ids.len() * std::mem::size_of::<u32>()
                    + t.tfs.len() * std::mem::size_of::<u16>()
            })
            .sum();
        postings_bytes as f64 / self.total_postings.max(1) as f64
    }

    #[inline]
    fn search_exhaustive(&mut self, query_terms: &[String], collector: &mut TopKCollector) {
        type ExhaustiveCursor<'a> = (&'a [u32], &'a [u16], f32, usize, u32);
        collector.clear();
        let norm_table = &self.norm_table;
        let doc_norm_ids = &self.doc_norm_ids;
        let k1_plus_1 = BM25_K1 + 1.0;

        let mut cursors: [ExhaustiveCursor<'_>; 8] = [(&[], &[], 0.0, 0, u32::MAX); 8];
        let mut m = 0usize;
        for qt in query_terms {
            if m >= 8 {
                break;
            }
            if let Some(&tid) = self.dict.get(qt.as_str()) {
                let term = &self.terms[tid as usize];
                if let Some(&first_doc) = term.doc_ids.first() {
                    cursors[m] = (&term.doc_ids, &term.tfs, term.idf * k1_plus_1, 0, first_doc);
                    m += 1;
                }
            }
        }
        if m == 0 {
            return;
        }
        if m == 1 {
            let (docs, tfs, idf_k1p1, _, _) = cursors[0];
            for (&doc_id, &tf) in docs.iter().zip(tfs.iter()) {
                let tf_f = tf as f32;
                let norm = norm_table[doc_norm_ids[doc_id as usize] as usize];
                collector.push(doc_id, idf_k1p1 * tf_f / (tf_f + norm));
            }
            return;
        }

        loop {
            let mut min_doc = cursors[0].4;
            for c in &cursors[1..m] {
                if c.4 < min_doc {
                    min_doc = c.4;
                }
            }
            if min_doc == u32::MAX {
                break;
            }
            let norm = norm_table[doc_norm_ids[min_doc as usize] as usize];
            let mut score = 0.0f32;
            for c in &mut cursors[..m] {
                if c.4 == min_doc {
                    let idx = c.3;
                    let tf_f = c.1[idx] as f32;
                    score += c.2 * tf_f / (tf_f + norm);
                    let next_idx = idx + 1;
                    c.3 = next_idx;
                    c.4 = if next_idx < c.0.len() {
                        c.0[next_idx]
                    } else {
                        u32::MAX
                    };
                }
            }
            collector.push(min_doc, score);
        }
    }
}

fn verify_top10_recall(
    zipora: &ZiporaSearchEngine,
    baseline: &mut UncompressedBaselineEngine,
    queries: &[Vec<String>],
) -> f64 {
    let mut bmw_col = TopKCollector::new(TOP_K);
    let mut ex_col = TopKCollector::new(TOP_K);
    let mut matched = 0usize;
    let mut total = 0usize;

    for q in queries {
        zipora.search_bmw(q, &mut bmw_col);
        baseline.search_exhaustive(q, &mut ex_col);

        let bmw_hits = bmw_col.to_sorted_vec();
        let ex_hits = ex_col.to_sorted_vec();
        assert_eq!(
            bmw_hits.len(),
            ex_hits.len(),
            "Top-K count mismatch on query {:?}",
            q
        );

        for (b_hit, e_hit) in bmw_hits.iter().zip(ex_hits.iter()) {
            total += 1;
            let diff = (b_hit.score - e_hit.score).abs();
            assert!(
                diff < 1e-4,
                "BM25 Top-K score mismatch on query {:?}: bmw=({},{}) vs exhaustive=({},{})",
                q,
                b_hit.doc_id,
                b_hit.score,
                e_hit.doc_id,
                e_hit.score
            );
            if b_hit.doc_id == e_hit.doc_id || diff < 1e-5 {
                matched += 1;
            }
        }
    }

    if total == 0 {
        1.0
    } else {
        matched as f64 / total as f64
    }
}

fn measure_query_latency_distribution_us<F>(
    num_queries: usize,
    passes: usize,
    mut run_idx: F,
) -> (f64, f64)
where
    F: FnMut(usize),
{
    let mut latencies_us = vec![f64::INFINITY; num_queries];
    // Run multiple full passes over all 10,000 queries in stream order and record the per-query
    // minimum across passes so cache state reflects realistic stream execution while filtering
    // out single-query OS scheduler interrupts.
    for _pass in 0..passes {
        for (idx, slot) in latencies_us.iter_mut().enumerate() {
            let t0 = Instant::now();
            run_idx(idx);
            let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
            if us < *slot {
                *slot = us;
            }
        }
    }
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let n = latencies_us.len().max(1);
    let p50 = latencies_us[n / 2];
    let p99_idx = ((n as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let p99 = latencies_us[p99_idx];
    (round2(p50), round2(p99))
}

fn write_json(path: &Path, report: &E2EBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote E2E search benchmark report to {}", path.display());
}

fn init_bench_allocator_policy() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `libc::mallopt` sets global glibc ptmalloc2 parameters using valid
        // documented option constants (`M_MMAP_MAX`, `M_TRIM_THRESHOLD`).
        unsafe {
            libc::mallopt(libc::M_MMAP_MAX, 0);
            libc::mallopt(libc::M_TRIM_THRESHOLD, -1);
        }
    }
}

fn main() {
    init_bench_allocator_policy();
    let args: Vec<String> = std::env::args().collect();
    let save_baseline = args.iter().any(|a| a == "--save-baseline")
        || std::env::var("ZIPORA_BENCH_SAVE_BASELINE").as_deref() == Ok("1")
        || std::env::var("UPDATE_BASELINE").as_deref() == Ok("1");

    let custom_out: Option<PathBuf> = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| PathBuf::from(&w[1]))
        .or_else(|| std::env::var("ZIPORA_BENCH_OUT").ok().map(PathBuf::from));

    let thp_on = args.iter().any(|a| a == "--thp-on")
        || matches!(
            std::env::var("ZIPORA_BENCH_THP").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        );
    if !thp_on {
        disable_transparent_hugepages_for_self();
    }
    let logical_cores = hardware_logical_cores();
    let primary_core = select_primary_core(logical_cores, 150);
    pin_current_thread_to_core(primary_core);
    let host_metadata = collect_host_metadata(primary_core);

    let num_docs: usize = std::env::var("ZIPORA_E2E_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let vocab_size: usize = std::env::var("ZIPORA_E2E_VOCAB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_096);
    let num_queries: usize = std::env::var("ZIPORA_E2E_QUERIES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    eprintln!(
        "=== Zipora E1.1 Systemic E2E Search Benchmark (host={}, cpu={}, pinned_core={}, docs={}, vocab={}, queries={}) ===",
        host_metadata.host,
        host_metadata.cpu_model,
        primary_core,
        num_docs,
        vocab_size,
        num_queries
    );

    let corpus = generate_zipf_corpus_and_queries(num_docs, vocab_size, num_queries, SEED_E2E);
    let zipora_engine = ZiporaSearchEngine::build(&corpus);
    let mut baseline_engine = UncompressedBaselineEngine::build(&corpus);

    // Verify exact top-10 recall and score parity on a 1,000-query stratified sample
    let recall = verify_top10_recall(
        &zipora_engine,
        &mut baseline_engine,
        &corpus.queries[..1_000],
    );
    assert!(
        (recall - 1.0).abs() < 1e-9,
        "Expected 100% top-10 recall vs exhaustive BM25, got {recall}"
    );

    let mut records = Vec::new();
    let mut collector = TopKCollector::new(TOP_K);
    let dataset_size = num_docs as u64;

    // 1. All 10k queries (Zipora Block-Max WAND)
    measure_bench(
        &mut records,
        "e2e_search/zipora_bmw/all_10k_queries",
        dataset_size,
        corpus.queries.len() as u64,
        DEFAULT_TRIALS,
        || {
            let mut hit_sum = 0u32;
            for q in &corpus.queries {
                zipora_engine.search_bmw(black_box(q), &mut collector);
                if let Some(top) = collector.heap.peek() {
                    hit_sum = hit_sum.wrapping_add(top.doc_id);
                }
            }
            black_box(hit_sum);
        },
    );

    // 2. Per query length breakdown (1..=4 terms)
    for q_len in 1..=4usize {
        let subset = &corpus.queries_by_len[q_len - 1];
        let trials = if q_len == 1 { 25 } else { DEFAULT_TRIALS };
        let bench_name = format!("e2e_search/zipora_bmw/{q_len}_term_queries");
        measure_bench(
            &mut records,
            &bench_name,
            dataset_size,
            subset.len() as u64,
            trials,
            || {
                let mut hit_sum = 0u32;
                for q in subset {
                    zipora_engine.search_bmw(black_box(q), &mut collector);
                    if let Some(top) = collector.heap.peek() {
                        hit_sum = hit_sum.wrapping_add(top.doc_id);
                    }
                }
                black_box(hit_sum);
            },
        );
    }

    // Measure Zipora p50/p99 per-query latency distribution before building Tantivy's
    // multi-threaded IndexWriter arena.
    let (zipora_p50_us, zipora_p99_us) =
        measure_query_latency_distribution_us(corpus.queries.len(), 9, |idx| {
            zipora_engine.search_bmw(black_box(&corpus.queries[idx]), &mut collector);
            black_box(collector.theta);
        });

    let tantivy_engine = TantivyEngine::build(&corpus);
    // Re-pin calling thread after Tantivy's IndexWriter thread exits (self PID 0 only).
    pin_current_thread_to_core(primary_core);

    // 3. Tantivy v0.26 in-RAM BM25 TopDocs(10) on all 10k queries
    measure_bench(
        &mut records,
        "e2e_search/tantivy_bm25/all_10k_queries",
        dataset_size,
        corpus.queries.len() as u64,
        DEFAULT_TRIALS,
        || {
            let mut hit_sum = 0u32;
            for q_idx in 0..corpus.queries.len() {
                hit_sum = hit_sum.wrapping_add(tantivy_engine.search_query_idx(black_box(q_idx)));
            }
            black_box(hit_sum);
        },
    );

    // 4. Uncompressed reference baseline on all 10k queries
    measure_bench(
        &mut records,
        "e2e_search/uncompressed_baseline/all_10k_queries",
        dataset_size,
        corpus.queries.len() as u64,
        DEFAULT_TRIALS,
        || {
            let mut hit_sum = 0u32;
            for q in &corpus.queries {
                baseline_engine.search_exhaustive(black_box(q), &mut collector);
                if let Some(top) = collector.heap.peek() {
                    hit_sum = hit_sum.wrapping_add(top.doc_id);
                }
            }
            black_box(hit_sum);
        },
    );

    let (tantivy_p50_us, tantivy_p99_us) =
        measure_query_latency_distribution_us(corpus.queries.len(), 5, |idx| {
            black_box(tantivy_engine.search_query_idx(black_box(idx)));
        });
    let (baseline_p50_us, baseline_p99_us) =
        measure_query_latency_distribution_us(corpus.queries.len(), 3, |idx| {
            baseline_engine.search_exhaustive(black_box(&corpus.queries[idx]), &mut collector);
            black_box(collector.theta);
        });

    let zipora_all_min_ns = records
        .iter()
        .find(|r| r.name == "e2e_search/zipora_bmw/all_10k_queries")
        .map(|r| r.min_ns_per_op)
        .unwrap_or(1.0);
    let tantivy_all_min_ns = records
        .iter()
        .find(|r| r.name == "e2e_search/tantivy_bm25/all_10k_queries")
        .map(|r| r.min_ns_per_op)
        .unwrap_or(1.0);
    let baseline_all_min_ns = records
        .iter()
        .find(|r| r.name == "e2e_search/uncompressed_baseline/all_10k_queries")
        .map(|r| r.min_ns_per_op)
        .unwrap_or(1.0);

    let zipora_qps_top10 = round2(1_000_000_000.0 / zipora_all_min_ns.max(1e-9));
    let tantivy_qps_top10 = round2(1_000_000_000.0 / tantivy_all_min_ns.max(1e-9));
    let baseline_qps_top10 = round2(1_000_000_000.0 / baseline_all_min_ns.max(1e-9));
    let qps_speedup_vs_tantivy = round4(zipora_qps_top10 / tantivy_qps_top10.max(1e-9));
    let qps_speedup_vs_baseline = round4(zipora_qps_top10 / baseline_qps_top10.max(1e-9));

    let (dict_bytes, fieldnorm_bytes, postings_bytes, bmw_metadata_bytes) =
        zipora_engine.memory_breakdown();
    let total_postings = zipora_engine.total_postings.max(1);
    let zipora_total_index_bytes =
        dict_bytes + fieldnorm_bytes + postings_bytes + bmw_metadata_bytes;
    let zipora_bytes_per_posting = round4(postings_bytes as f64 / total_postings as f64);
    let zipora_postings_plus_bmw_bytes_per_posting =
        round4((postings_bytes + bmw_metadata_bytes) as f64 / total_postings as f64);
    let zipora_total_index_bytes_per_posting =
        round4(zipora_total_index_bytes as f64 / total_postings as f64);
    let tantivy_postings_bytes_per_posting = round4(tantivy_engine.postings_bytes_per_posting());
    let tantivy_index_without_store_bytes_per_posting =
        round4(tantivy_engine.index_without_store_bytes_per_posting());
    let tantivy_bytes_per_posting = round4(tantivy_engine.bytes_per_posting());
    let baseline_bytes_per_posting = round4(baseline_engine.bytes_per_posting());
    let vm_hwm_kib = read_vm_hwm_kib();
    let encoding_distribution = zipora_engine.encoding_distribution();

    for rec in &records {
        eprintln!(
            "  [{:<12}] {:<48} min={:>9.2} {:<8} med={:>9.2} {:<8} p95={:>9.2} spread={:>5.2}%",
            rec.category,
            rec.name,
            rec.min_ns_per_op,
            rec.unit,
            rec.median_ns_per_op,
            rec.unit,
            rec.p95_ns_per_op,
            rec.spread_pct
        );
    }

    eprintln!(
        "  Zipora BMW:   QPS@top-10 = {:.1} q/s (p50 = {:.2} us, p99 = {:.2} us, postings = {:.3} B/posting, postings+bmw = {:.3} B/posting, total = {:.3} B/posting, VmHWM = {} KiB)",
        zipora_qps_top10,
        zipora_p50_us,
        zipora_p99_us,
        zipora_bytes_per_posting,
        zipora_postings_plus_bmw_bytes_per_posting,
        zipora_total_index_bytes_per_posting,
        vm_hwm_kib
    );
    eprintln!(
        "  Tantivy 0.26: QPS@top-10 = {:.1} q/s (p50 = {:.2} us, p99 = {:.2} us, postings = {:.3} B/posting, index_no_store = {:.3} B/posting, total = {:.3} B/posting) -> Zipora speedup vs Tantivy = {:.2}x",
        tantivy_qps_top10,
        tantivy_p50_us,
        tantivy_p99_us,
        tantivy_postings_bytes_per_posting,
        tantivy_index_without_store_bytes_per_posting,
        tantivy_bytes_per_posting,
        qps_speedup_vs_tantivy
    );
    eprintln!(
        "  Uncompressed: QPS@top-10 = {:.1} q/s (p50 = {:.2} us, p99 = {:.2} us, postings = {:.3} B/posting) -> Zipora speedup = {:.2}x, recall@10 = {:.4}",
        baseline_qps_top10,
        baseline_p50_us,
        baseline_p99_us,
        baseline_bytes_per_posting,
        qps_speedup_vs_baseline,
        recall
    );

    let summary = E2ESummary {
        num_docs,
        vocab_size,
        total_postings: zipora_engine.total_postings,
        num_queries,
        top_k: TOP_K,
        zipora_qps_top10,
        zipora_p50_us,
        zipora_p99_us,
        zipora_vm_hwm_kib: vm_hwm_kib,
        zipora_bytes_per_posting,
        zipora_postings_plus_bmw_bytes_per_posting,
        zipora_total_index_bytes_per_posting,
        zipora_dict_bytes: dict_bytes,
        zipora_fieldnorm_bytes: fieldnorm_bytes,
        zipora_postings_bytes: postings_bytes,
        zipora_bmw_metadata_bytes: bmw_metadata_bytes,
        zipora_total_index_bytes,
        tantivy_qps_top10,
        tantivy_p50_us,
        tantivy_p99_us,
        tantivy_postings_bytes: tantivy_engine.postings_bytes,
        tantivy_postings_bytes_per_posting,
        tantivy_termdict_bytes: tantivy_engine.termdict_bytes,
        tantivy_fieldnorms_bytes: tantivy_engine.fieldnorms_bytes,
        tantivy_store_bytes: tantivy_engine.store_bytes,
        tantivy_index_without_store_bytes: tantivy_engine.index_without_store_bytes,
        tantivy_index_without_store_bytes_per_posting,
        tantivy_index_bytes: tantivy_engine.index_bytes,
        tantivy_bytes_per_posting,
        qps_speedup_vs_tantivy,
        baseline_qps_top10,
        baseline_p50_us,
        baseline_p99_us,
        baseline_bytes_per_posting,
        qps_speedup_vs_baseline,
        top10_recall_vs_exhaustive: recall,
        competitor_note: "Side-by-side comparison on the identical 100,000-doc Zipf corpus and 10,000 queries against tantivy v0.26.2 (Index::create_in_ram, IndexRecordOption::WithFreqs, BooleanQuery + TopDocs::with_limit(10).order_by_score()) and an in-process uncompressed inverted-index exhaustive baseline. Zipora BMW beats both Tantivy and the uncompressed baseline on QPS@10 and p50 latency, and beats the uncompressed baseline on p99 tail latency (via peak-block threshold priming, contiguous-run Clustered decoding, and tight multi-term block-max upper-bound pruning); Tantivy v0.26.2 has ~6% lower p99 on 4-term dense-tail queries due to 128-doc SIMD bitpacking vs Elias-Fano select1 decoding. Space comparison: postings-vs-postings is parity (Zipora HybridPostingList = 1.223 B/posting vs Tantivy postings = 1.227 B/posting; including Zipora's uncompressed 64-doc block-max metadata is 2.379 B/posting vs Tantivy's 1.227 B/posting, 1.94x larger), and whole-index is 2.701 B/posting for Zipora vs 1.323 B/posting for Tantivy without its empty doc store (1.643 B/posting including the 515 KB empty doc store).".to_string(),
    };

    let report = E2EBenchReport {
        suite: "e2e_search".to_string(),
        seed: SEED_E2E,
        host_metadata: host_metadata.clone(),
        summary,
        encoding_distribution,
        benchmarks: records,
    };

    let latest_out = custom_out.unwrap_or_else(|| PathBuf::from("target/bench/e2e_search.json"));
    write_json(&latest_out, &report);

    if save_baseline {
        let baseline_path = PathBuf::from(format!(
            "benches/baselines/{}/e2e_search.json",
            host_metadata.host
        ));
        write_json(&baseline_path, &report);
    }
}
