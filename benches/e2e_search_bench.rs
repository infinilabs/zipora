//! Phase E1.1 (G15) Systemic End-to-End Search Benchmark (`make bench_e2e_search`).
//!
//! Combines Zipora's core search-engine primitives into an end-to-end BM25 top-10
//! retrieval engine and benchmarks it on a deterministic Zipfian corpus and
//! 10,000 Zipf queries (1–4 terms, fixed seed):
//! - Term dictionary: [`DoubleArrayTrieMap<u32>`]
//! - Posting lists: [`HybridPostingList`] per term (auto-selecting `Dense`,
//!   `EliasFano`, `Partitioned`, `Optimal`, or `Clustered`) with 64-doc block-max scores
//! - Document lengths: [`FieldnormEncoder`] (`1.0 B/doc` SmallFloat quantization)
//! - Query evaluation: bench-local Block-Max WAND (BMW) using [`simd_gallop_to`],
//!   [`simd_block_filter`], and [`Bm25BatchScorer`] (`batch_score_with_prefetch` + `score`)
//! - Comparison baseline: in-process uncompressed inverted-index baseline (`HashMap`
//!   dictionary + uncompressed `Vec<u32>` doc IDs + `Vec<u16>` TFs + full BM25 scan)
//!   on the identical corpus and 10,000 queries (`Cargo.lock` is frozen with `--locked`
//!   in `make sanity` per `S21-R1`).

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
const DEFAULT_TRIALS: usize = 3;
const SUITE_PASSES: usize = 3;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HostMetadata {
    host: String,
    cpu_model: String,
    logical_cores: usize,
    governor: String,
    git_commit: String,
    timestamp_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchRecord {
    category: String,
    name: String,
    unit: String,
    ops_per_trial: u64,
    trials: usize,
    min_ns_per_op: f64,
    median_ns_per_op: f64,
    p95_ns_per_op: f64,
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
    zipora_total_index_bytes_per_posting: f64,
    zipora_dict_bytes: usize,
    zipora_fieldnorm_bytes: usize,
    zipora_postings_bytes: usize,
    zipora_bmw_metadata_bytes: usize,
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

fn collect_host_metadata() -> HostMetadata {
    HostMetadata {
        host: detect_short_host(),
        cpu_model: detect_cpu_model(),
        logical_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        governor: detect_governor(),
        git_commit: detect_git_commit(),
        timestamp_utc: detect_utc_timestamp(),
    }
}

fn select_primary_core(logical_cores: usize) -> usize {
    if let Ok(val) = std::env::var("ZIPORA_BENCH_CORE")
        && let Ok(core) = val.parse::<usize>()
    {
        return core % logical_cores.max(1);
    }
    if logical_cores > 62 {
        62
    } else {
        logical_cores.saturating_sub(1)
    }
}

#[cfg(target_os = "linux")]
fn configure_allocator_for_benchmarks() {
    const M_TRIM_THRESHOLD: libc::c_int = -1;
    const M_MMAP_THRESHOLD: libc::c_int = -3;
    // SAFETY: Standard glibc mallopt parameters to disable brk auto-trimming and
    // raise mmap threshold to 64 MiB during benchmark runs.
    unsafe {
        libc::mallopt(M_TRIM_THRESHOLD, -1);
        libc::mallopt(M_MMAP_THRESHOLD, 64 * 1024 * 1024);
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_allocator_for_benchmarks() {}

#[cfg(target_os = "linux")]
fn pin_current_thread_to_core(core_id: usize) {
    // SAFETY: `cpu_set_t` is zero-initialized and `CPU_SET` / `sched_setaffinity`
    // are called with valid size and self PID (0).
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core_id, &mut set);
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_core(_core_id: usize) {}

fn compute_stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let n = sorted.len();
    let min = sorted[0];
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let p95_idx = ((n as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let p95 = sorted[p95_idx];
    (min, median, p95)
}

fn measure_bench<F>(
    records: &mut Vec<BenchRecord>,
    category: &str,
    name: &str,
    unit: &str,
    ops_per_trial: u64,
    trials: usize,
    mut f: F,
) where
    F: FnMut(),
{
    let is_first_pass = !records.iter().any(|r| r.name == name);
    if is_first_pass {
        f();
    }
    let mut samples = Vec::with_capacity(trials);
    let bursts_per_trial = 4usize;
    for _ in 0..trials {
        let mut best_ns = f64::INFINITY;
        for _ in 0..bursts_per_trial {
            let start = Instant::now();
            f();
            let elapsed_ns = start.elapsed().as_nanos() as f64;
            let ns_per_op = elapsed_ns / (ops_per_trial.max(1) as f64);
            if ns_per_op < best_ns {
                best_ns = ns_per_op;
            }
        }
        samples.push(best_ns);
    }

    if let Some(existing) = records.iter_mut().find(|r| r.name == name) {
        existing.samples_ns_per_op.extend_from_slice(&samples);
        let (min_ns, median_ns, p95_ns) = compute_stats(&existing.samples_ns_per_op);
        existing.trials = existing.samples_ns_per_op.len();
        existing.min_ns_per_op = min_ns;
        existing.median_ns_per_op = median_ns;
        existing.p95_ns_per_op = p95_ns;
    } else {
        let (min_ns, median_ns, p95_ns) = compute_stats(&samples);
        records.push(BenchRecord {
            category: category.to_string(),
            name: name.to_string(),
            unit: unit.to_string(),
            ops_per_trial,
            trials: samples.len(),
            min_ns_per_op: min_ns,
            median_ns_per_op: median_ns,
            p95_ns_per_op: p95_ns,
            samples_ns_per_op: samples,
        });
    }
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
// Synthetic Zipf Corpus & Queries
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

    // 3. Zipfian posting lists across all HybridPostingList tiers:
    //    - Top 32 terms: high-frequency / run-heavy or large (2,048..6,500 postings -> Optimal / Clustered)
    //    - Next 384 terms: medium-large (260..2,000 postings -> Partitioned)
    //    - Next 1,024 terms: medium (65..256 postings -> EliasFano)
    //    - Remaining terms: rare (8..64 postings -> Dense)
    let mut term_doc_ids = Vec::with_capacity(vocab_size);
    let mut term_tfs = Vec::with_capacity(vocab_size);

    for rank in 0..vocab_size {
        let df: usize = if rank < 8 {
            // Very dense / run-heavy head terms (>50% of docs -> Clustered)
            ((num_docs * 3) / 5 + (8 - rank) * 120).min(num_docs)
        } else if rank < 32 {
            // Large OptimalPEF tier (> 4,096 postings)
            (4_200 + (32 - rank) * 60).min(num_docs)
        } else if rank < 384 {
            // PartitionedEF tier (257..=4,096 postings)
            let span = 384 - 32;
            let pos = rank - 32;
            280 + ((span - pos) * 1_600) / span
        } else if rank < 1_408 {
            // EliasFano tier (65..=256 postings)
            let span = 1_408 - 384;
            let pos = rank - 384;
            68 + ((span - pos) * 180) / span
        } else {
            // Dense tier (8..=64 postings)
            let span = vocab_size - 1_408;
            let pos = rank - 1_408;
            8 + ((span - pos) * 54) / span.max(1)
        };

        let df = df.clamp(1, num_docs);
        // Generate sorted unique doc_ids in [0, num_docs)
        let mut docs = Vec::with_capacity(df);
        if df * 2 >= num_docs {
            // Dense sampling with contiguous 128..=512 doc runs so ClusteredEliasFano
            // exercises ChunkKind::Run as well as Bitmap/EF chunks.
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
            let mut fill = 0u32;
            while docs.len() < df {
                if docs.binary_search(&fill).is_err() {
                    docs.push(fill);
                    docs.sort_unstable();
                }
                fill += 1;
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
    // Sample query terms using a Zipfian CDF over ranks [8..vocab_size) + occasional head terms.
    let mut zipf_weights = Vec::with_capacity(vocab_size);
    let mut cum = 0.0f64;
    for rank in 0..vocab_size {
        // Query Zipf exponent s = 1.05 shifted by 16
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

struct TermPostingData {
    postings: HybridPostingList,
    tfs: Vec<u8>,
    block_max_doc_ids: Vec<u32>,
    block_max_scores: Vec<f32>,
    max_term_score: f32,
    scorer: Bm25BatchScorer<'static>,
}

struct ZiporaSearchEngine {
    dict: DoubleArrayTrieMap<u32>,
    fieldnorms: Vec<u8>,
    terms: Vec<TermPostingData>,
    total_postings: usize,
}

impl ZiporaSearchEngine {
    fn build(corpus: &CorpusData) -> Self {
        let mut dict = DoubleArrayTrieMap::<u32>::with_capacity(corpus.terms.len() * 2);
        for (tid, term) in corpus.terms.iter().enumerate() {
            dict.insert(term.as_bytes(), tid as u32).unwrap();
        }

        let fieldnorms: Vec<u8> = corpus
            .doc_lens
            .iter()
            .map(|&dl| FieldnormEncoder::encode(dl))
            .collect();
        static NORM_TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
        let norm_table = NORM_TABLE
            .get_or_init(|| FieldnormEncoder::build_norm_table(corpus.avg_dl, BM25_K1, BM25_B));

        let n_docs_f = corpus.num_docs as f32;
        let mut terms = Vec::with_capacity(corpus.terms.len());
        let mut total_postings = 0usize;

        for (docs, tfs_u16) in corpus.term_doc_ids.iter().zip(corpus.term_tfs.iter()) {
            let df = docs.len();
            total_postings += df;
            let idf = ((n_docs_f - df as f32 + 0.5) / (df as f32 + 0.5) + 1.0).ln();
            let scorer = Bm25BatchScorer::new(norm_table, idf, BM25_K1);

            let postings = HybridPostingList::from_sorted(docs);
            let tfs: Vec<u8> = tfs_u16.iter().map(|&tf| tf.min(255) as u8).collect();

            let num_blocks = df.div_ceil(BLOCK_SIZE);
            let mut block_max_doc_ids = Vec::with_capacity(num_blocks);
            let mut block_max_scores = Vec::with_capacity(num_blocks);
            let mut max_term_score = 0.0f32;

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
                block_max_doc_ids.push(last_doc);
                block_max_scores.push(blk_max);
            }

            terms.push(TermPostingData {
                postings,
                tfs,
                block_max_doc_ids,
                block_max_scores,
                max_term_score,
                scorer,
            });
        }

        Self {
            dict,
            fieldnorms,
            terms,
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
                    Some((
                        Self::Clustered {
                            cef,
                            idx: 0,
                            cur_doc: first,
                            len,
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
            } => {
                *idx += 1;
                if *idx < *len {
                    let d = cef.get(*idx)? as u32;
                    *cur_doc = d;
                    Some((*idx, d))
                } else {
                    None
                }
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
            } => {
                if *idx >= *len {
                    None
                } else if *cur_doc >= target {
                    Some((*idx, *cur_doc))
                } else {
                    let (i, v) = cef.next_geq(target as u64)?;
                    *idx = i;
                    *cur_doc = v as u32;
                    Some((i, v as u32))
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
            self.search_single_term(&self.terms[tid as usize], collector);
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
            self.search_single_term(&self.terms[term_indices[0]], collector);
            return;
        }

        self.search_multi_term_bmw(&term_indices[..n_terms], collector);
    }

    /// Single-term Block-Max pruning + `Bm25BatchScorer::batch_score_with_prefetch` + `simd_block_filter`.
    #[inline]
    fn search_single_term(&self, term: &TermPostingData, collector: &mut TopKCollector) {
        let df = term.postings.len();
        if df == 0 {
            return;
        }
        let num_blocks = term.block_max_scores.len();
        let mut block_docs = [0u32; BLOCK_SIZE];
        let mut block_tfs = [0u16; BLOCK_SIZE];
        let mut block_scores = [0.0f32; BLOCK_SIZE];

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
            // Block-Max pruning: skip entire 64-doc block if its max score cannot enter top-K
            if term.block_max_scores[blk_idx] < collector.theta
                && collector.heap.len() >= collector.k
            {
                continue;
            }

            let start = blk_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(df);
            let count = end - start;

            if let Some(slice) = dense_slice {
                block_docs[..count].copy_from_slice(&slice[start..end]);
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
                for slot in &mut block_docs[..count] {
                    *slot = *cur_doc;
                    if let Some((i, d)) = cur.advance() {
                        *cur_idx = i;
                        *cur_doc = d;
                    }
                }
            }
            for i in 0..count {
                block_tfs[i] = term.tfs[start + i] as u16;
            }

            term.scorer.batch_score_with_prefetch(
                &self.fieldnorms,
                &block_docs[..count],
                &block_tfs[..count],
                &mut block_scores[..count],
            );

            let filter_theta = if collector.heap.len() < collector.k {
                -1.0
            } else {
                collector.theta - 1e-6
            };
            let (mut mask, _) =
                simd_block_filter(&block_docs[..count], &block_scores[..count], filter_theta);

            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                collector.push(block_docs[bit], block_scores[bit]);
            }
        }
    }

    /// Multi-term Block-Max WAND / MaxScore using `Bm25BatchScorer::batch_score_with_prefetch`,
    /// `simd_block_filter` on lead-term 64-doc blocks, and `simd_gallop_to` on `block_max_doc_ids`
    /// + stateful `HybridCursor::advance_to_geq` on non-essential terms.
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

        let mut block_docs = [0u32; BLOCK_SIZE];
        let mut block_tfs = [0u16; BLOCK_SIZE];
        let mut block_scores = [0.0f32; BLOCK_SIZE];

        // Evaluate essential lead terms i = 0..m while suffix_max[i] > collector.theta
        for lead_pos in 0..m {
            let threshold_at_lead = if collector.heap.len() < collector.k {
                -1.0
            } else {
                collector.theta - 1e-6
            };
            if suffix_max[lead_pos] <= threshold_at_lead {
                // Remaining terms combined cannot beat collector.theta
                break;
            }

            let lead_tid = sorted_tids[lead_pos];
            let lead_term = &self.terms[lead_tid];
            let df = lead_term.postings.len();
            if df == 0 {
                continue;
            }
            let rem_upper = suffix_max[lead_pos + 1];

            // Stateful cursors for earlier terms 0..lead_pos (to skip already-evaluated docs)
            let mut prev_cursors: [Option<(HybridCursor<'_>, usize, u32, usize)>; 4] =
                [None, None, None, None];
            for (p_idx, &p_tid) in sorted_tids[..lead_pos].iter().enumerate() {
                if let Some((cur, idx, first_doc)) = HybridCursor::new(&self.terms[p_tid].postings)
                {
                    prev_cursors[p_idx] = Some((cur, idx, first_doc, 0));
                }
            }

            // Stateful cursors for trailing terms (lead_pos + 1)..m
            let mut tail_cursors: [Option<(HybridCursor<'_>, usize, u32, usize)>; 4] =
                [None, None, None, None];
            for (t_offset, &t_tid) in sorted_tids[(lead_pos + 1)..m].iter().enumerate() {
                if let Some((cur, idx, first_doc)) = HybridCursor::new(&self.terms[t_tid].postings)
                {
                    tail_cursors[t_offset] = Some((cur, idx, first_doc, 0));
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

            let num_blocks = lead_term.block_max_scores.len();
            for blk_idx in 0..num_blocks {
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
                    block_docs[..count].copy_from_slice(&slice[start..end]);
                } else if let Some((ref mut cur, ref mut cur_idx, ref mut cur_doc)) = lead_cursor {
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
                    for slot in &mut block_docs[..count] {
                        *slot = *cur_doc;
                        if let Some((i, d)) = cur.advance() {
                            *cur_idx = i;
                            *cur_doc = d;
                        }
                    }
                }

                for i in 0..count {
                    block_tfs[i] = lead_term.tfs[start + i] as u16;
                }

                lead_term.scorer.batch_score_with_prefetch(
                    &self.fieldnorms,
                    &block_docs[..count],
                    &block_tfs[..count],
                    &mut block_scores[..count],
                );

                // SIMD block filter: keep only docs where score(d, lead) + rem_upper > cur_theta
                let block_filter_theta = cur_theta - rem_upper;
                let (mut mask, _) = simd_block_filter(
                    &block_docs[..count],
                    &block_scores[..count],
                    block_filter_theta,
                );

                'doc_loop: while mask != 0 {
                    let bit = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let doc_id = block_docs[bit];
                    let mut acc_score = block_scores[bit];

                    let doc_theta = if collector.heap.len() < collector.k {
                        -1.0
                    } else {
                        collector.theta - 1e-6
                    };
                    if acc_score + rem_upper <= doc_theta {
                        continue;
                    }

                    // If lead_pos > 0, check if doc_id already appeared in an earlier lead term 0..lead_pos
                    for (p_idx, &p_tid) in sorted_tids[..lead_pos].iter().enumerate() {
                        if let Some((
                            ref mut p_cur,
                            ref mut p_post_idx,
                            ref mut p_doc,
                            ref mut p_blk,
                        )) = prev_cursors[p_idx]
                        {
                            let p_term = &self.terms[p_tid];
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
                                // Already evaluated when `p_tid` was the lead term
                                continue 'doc_loop;
                            }
                        }
                    }

                    // Probe trailing terms (lead_pos + 1)..m using Block-Max WAND bounds
                    let norm_id = self.fieldnorms[doc_id as usize];
                    for (t_offset, &t_tid) in sorted_tids[(lead_pos + 1)..m].iter().enumerate() {
                        let after_this_upper = suffix_max[lead_pos + 1 + t_offset + 1];
                        let t_term = &self.terms[t_tid];

                        if let Some((
                            ref mut t_cur,
                            ref mut t_post_idx,
                            ref mut t_doc,
                            ref mut t_blk,
                        )) = tail_cursors[t_offset]
                        {
                            if *t_doc < doc_id {
                                if !simd_gallop_to(&t_term.block_max_doc_ids, t_blk, doc_id) {
                                    *t_doc = u32::MAX;
                                } else {
                                    // Check Block-Max bound BEFORE decoding t_term's posting list!
                                    let blk_max = t_term.block_max_scores[*t_blk];
                                    if acc_score + blk_max + after_this_upper <= doc_theta {
                                        // Even with this block's max score + all later terms' global max,
                                        // doc_id cannot beat doc_theta!
                                        continue 'doc_loop;
                                    }
                                    if let Some((i, d)) = t_cur.advance_to_geq(doc_id) {
                                        *t_post_idx = i;
                                        *t_doc = d;
                                    } else {
                                        *t_doc = u32::MAX;
                                    }
                                }
                            }
                            if *t_doc == doc_id {
                                let tf = t_term.tfs[*t_post_idx] as u16;
                                acc_score += t_term.scorer.score(norm_id, tf);
                            }
                        }

                        if acc_score + after_this_upper <= doc_theta {
                            continue 'doc_loop;
                        }
                    }

                    collector.push(doc_id, acc_score);
                }
            }
        }
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
    doc_norms: Vec<f32>,
    terms: Vec<UncompressedTermPostings>,
    total_postings: usize,
    doc_scores_scratch: Vec<f32>,
    touched_docs: Vec<u32>,
}

impl UncompressedBaselineEngine {
    fn build(corpus: &CorpusData) -> Self {
        let mut dict = HashMap::with_capacity(corpus.terms.len() * 2);
        for (tid, term) in corpus.terms.iter().enumerate() {
            dict.insert(term.clone(), tid as u32);
        }

        // Standard BM25 length norm using quantized FieldnormEncoder lengths so scores match
        let norm_table = FieldnormEncoder::build_norm_table(corpus.avg_dl, BM25_K1, BM25_B);
        let doc_norms: Vec<f32> = corpus
            .doc_lens
            .iter()
            .map(|&dl| norm_table[FieldnormEncoder::encode(dl) as usize])
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
            doc_norms,
            terms,
            total_postings,
            doc_scores_scratch: vec![0.0f32; corpus.num_docs],
            touched_docs: Vec::with_capacity(4_096),
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
        collector.clear();
        self.touched_docs.clear();

        let k1_plus_1 = BM25_K1 + 1.0;
        for qt in query_terms {
            let Some(&tid) = self.dict.get(qt.as_str()) else {
                continue;
            };
            let term = &self.terms[tid as usize];
            let idf = term.idf;
            for (&doc_id, &tf) in term.doc_ids.iter().zip(term.tfs.iter()) {
                let d = doc_id as usize;
                let prev = self.doc_scores_scratch[d];
                if prev == 0.0 {
                    self.touched_docs.push(doc_id);
                }
                let tf_f = tf as f32;
                let score = idf * (tf_f * k1_plus_1) / (tf_f + self.doc_norms[d]);
                self.doc_scores_scratch[d] = prev + score;
            }
        }

        for &doc_id in &self.touched_docs {
            let d = doc_id as usize;
            let s = self.doc_scores_scratch[d];
            self.doc_scores_scratch[d] = 0.0;
            collector.push(doc_id, s);
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

fn measure_query_latency_distribution_us<F>(queries: &[Vec<String>], mut run_one: F) -> (f64, f64)
where
    F: FnMut(&[String]),
{
    let mut latencies_us = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = Instant::now();
        run_one(q);
        let us = t0.elapsed().as_nanos() as f64 / 1_000.0;
        latencies_us.push(us);
    }
    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let n = latencies_us.len().max(1);
    let p50 = latencies_us[n / 2];
    let p99_idx = ((n as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let p99 = latencies_us[p99_idx];
    (p50, p99)
}

fn write_json(path: &Path, report: &E2EBenchReport) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let json = serde_json::to_string_pretty(report).unwrap();
    fs::write(path, json).unwrap();
    eprintln!("Wrote E2E search benchmark report to {}", path.display());
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

    let num_docs: usize = std::env::var("ZIPORA_E2E_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000);
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

    for _pass in 0..SUITE_PASSES {
        // 1. All 10k queries (Zipora Block-Max WAND)
        measure_bench(
            &mut records,
            "e2e_search",
            "e2e_search/zipora_bmw/all_10k_queries",
            "ns/query",
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
            let bench_name = format!("e2e_search/zipora_bmw/{q_len}_term_queries");
            measure_bench(
                &mut records,
                "e2e_search",
                &bench_name,
                "ns/query",
                subset.len() as u64,
                DEFAULT_TRIALS,
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

        // 3. Uncompressed reference baseline on all 10k queries
        measure_bench(
            &mut records,
            "e2e_search",
            "e2e_search/uncompressed_baseline/all_10k_queries",
            "ns/query",
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
    }

    // Measure p50/p99 per-query latency distribution (in microseconds)
    let (zipora_p50_us, zipora_p99_us) =
        measure_query_latency_distribution_us(&corpus.queries, |q| {
            zipora_engine.search_bmw(black_box(q), &mut collector);
            black_box(collector.theta);
        });
    let (baseline_p50_us, baseline_p99_us) =
        measure_query_latency_distribution_us(&corpus.queries, |q| {
            baseline_engine.search_exhaustive(black_box(q), &mut collector);
            black_box(collector.theta);
        });

    let zipora_all_med_ns = records
        .iter()
        .find(|r| r.name == "e2e_search/zipora_bmw/all_10k_queries")
        .map(|r| r.median_ns_per_op)
        .unwrap_or(1.0);
    let baseline_all_med_ns = records
        .iter()
        .find(|r| r.name == "e2e_search/uncompressed_baseline/all_10k_queries")
        .map(|r| r.median_ns_per_op)
        .unwrap_or(1.0);

    let zipora_qps_top10 = 1_000_000_000.0 / zipora_all_med_ns.max(1e-9);
    let baseline_qps_top10 = 1_000_000_000.0 / baseline_all_med_ns.max(1e-9);
    let qps_speedup_vs_baseline = zipora_qps_top10 / baseline_qps_top10.max(1e-9);

    let (dict_bytes, fieldnorm_bytes, postings_bytes, bmw_metadata_bytes) =
        zipora_engine.memory_breakdown();
    let total_postings = zipora_engine.total_postings.max(1);
    let zipora_bytes_per_posting = postings_bytes as f64 / total_postings as f64;
    let zipora_total_index_bytes_per_posting =
        (dict_bytes + fieldnorm_bytes + postings_bytes + bmw_metadata_bytes) as f64
            / total_postings as f64;
    let baseline_bytes_per_posting = baseline_engine.bytes_per_posting();
    let vm_hwm_kib = read_vm_hwm_kib();
    let encoding_distribution = zipora_engine.encoding_distribution();

    for rec in &records {
        eprintln!(
            "  [{:<12}] {:<48} min={:>9.2} {:<8} med={:>9.2} {:<8} p95={:>9.2}",
            rec.category,
            rec.name,
            rec.min_ns_per_op,
            rec.unit,
            rec.median_ns_per_op,
            rec.unit,
            rec.p95_ns_per_op
        );
    }

    eprintln!(
        "  Summary: Zipora BMW QPS@top-10 = {:.1} q/s (p50 = {:.2} us, p99 = {:.2} us, postings = {:.3} B/posting, total = {:.3} B/posting, VmHWM = {} KiB)",
        zipora_qps_top10,
        zipora_p50_us,
        zipora_p99_us,
        zipora_bytes_per_posting,
        zipora_total_index_bytes_per_posting,
        vm_hwm_kib
    );
    eprintln!(
        "  Baseline: Uncompressed QPS@top-10 = {:.1} q/s (p50 = {:.2} us, p99 = {:.2} us, postings = {:.3} B/posting) -> Zipora speedup = {:.2}x, recall@10 = {:.4}",
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
        zipora_total_index_bytes_per_posting,
        zipora_dict_bytes: dict_bytes,
        zipora_fieldnorm_bytes: fieldnorm_bytes,
        zipora_postings_bytes: postings_bytes,
        zipora_bmw_metadata_bytes: bmw_metadata_bytes,
        baseline_qps_top10,
        baseline_p50_us,
        baseline_p99_us,
        baseline_bytes_per_posting,
        qps_speedup_vs_baseline,
        top10_recall_vs_exhaustive: recall,
        competitor_note: "In-process uncompressed inverted-index baseline (HashMap<String, u32> + Vec<u32> doc_ids + Vec<u16> tfs + full BM25 scan); Cargo.lock is frozen under --locked in make sanity (S21-R1) so external tantivy crate is deferred to E2.".to_string(),
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

    let baseline_path = PathBuf::from(format!(
        "benches/baselines/{}/e2e_search.json",
        host_metadata.host
    ));
    if save_baseline || !baseline_path.exists() {
        write_json(&baseline_path, &report);
    }
}
