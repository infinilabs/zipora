//! Elias-Fano / posting-list benchmark harness (code_review.md §5.2, Phase 0).
//!
//! Establishes the baseline this repo lacks: `next_geq` (skip-heavy), construction,
//! sequential `iter`, 2-list intersection, and space (bits/elem) across the existing
//! encodings (EF / PEF / OPEF) on three data distributions:
//!   - **uniform-sparse**  — large random gaps (EF's home turf)
//!   - **bursty-clustered** — dense runs separated by big gaps (the case ClusteredEF targets)
//!   - **fully-dense**      — consecutive integers (run-length's home turf)
//!
//! Run: `cargo bench --bench elias_fano_bench`
//! Space numbers (bits/elem) are printed to stderr once at startup.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use zipora::succinct::elias_fano::{
    ClusteredEliasFano, EliasFano, OptimalPartitionedEliasFano, PartitionedEliasFano,
};

const N: usize = 100_000;
const NUM_TARGETS: usize = 2_000;
const SEED: u64 = 0xE1A5_FA00;

// ---------------------------------------------------------------------------
// Data generators — all return strictly increasing Vec<u32>.
// ---------------------------------------------------------------------------

/// Uniform-sparse: cumulative sum of random gaps in [1, 2*avg_gap].
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

/// Bursty-clustered: dense consecutive runs (50..=200) separated by big gaps
/// (5_000..=50_000). Models per-tenant / per-crawl doc-id locality.
fn gen_bursty_clustered(n: usize, seed: u64) -> Vec<u32> {
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

/// Fully-dense: consecutive integers 0..n.
fn gen_fully_dense(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// Random targets in [0, universe) for skip-heavy next_geq.
fn gen_targets(values: &[u32], count: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let universe = *values.last().unwrap() as u64 + 1;
    (0..count).map(|_| rng.random_range(0..universe)).collect()
}

// ---------------------------------------------------------------------------
// Intersection oracle (leapfrog via next_geq) — counts common elements.
// Works against any type exposing next_geq(u64) -> Option<(usize, u64)>.
// ---------------------------------------------------------------------------

macro_rules! intersect_count {
    ($a:expr, $b:expr) => {{
        let (a, b) = ($a, $b);
        let mut count = 0usize;
        let mut probe = 0u64;
        loop {
            match a.next_geq(probe) {
                Some((_, va)) => match b.next_geq(va) {
                    Some((_, vb)) => {
                        if vb == va {
                            count += 1;
                            probe = va + 1;
                        } else {
                            probe = vb;
                        }
                    }
                    None => break,
                },
                None => break,
            }
        }
        count
    }};
}

// ---------------------------------------------------------------------------
// Space report (printed once, not timed).
// ---------------------------------------------------------------------------

fn report_space() {
    let dists: [(&str, Vec<u32>); 3] = [
        ("uniform-sparse", gen_uniform_sparse(N, 50, SEED)),
        ("bursty-clustered", gen_bursty_clustered(N, SEED)),
        ("fully-dense", gen_fully_dense(N)),
    ];
    eprintln!("\n=== Elias-Fano space baseline (N={N}, bits/elem) ===");
    eprintln!(
        "{:<18} {:>10} {:>10} {:>10} {:>10}",
        "distribution", "EF", "PEF", "OPEF", "CEF"
    );
    for (name, v) in &dists {
        let ef = EliasFano::from_sorted(v);
        let pef = PartitionedEliasFano::from_sorted(v);
        let opef = OptimalPartitionedEliasFano::from_sorted(v);
        let cef = ClusteredEliasFano::from_sorted(v);
        eprintln!(
            "{:<18} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            name,
            ef.bits_per_element(),
            pef.bits_per_element(),
            opef.bits_per_element(),
            cef.bits_per_element(),
        );
    }
    eprintln!();
}

// ---------------------------------------------------------------------------
// Benches.
// ---------------------------------------------------------------------------

fn bench_next_geq(c: &mut Criterion) {
    let mut group = c.benchmark_group("next_geq");
    let dists: [(&str, Vec<u32>); 3] = [
        ("sparse", gen_uniform_sparse(N, 50, SEED)),
        ("clustered", gen_bursty_clustered(N, SEED)),
        ("dense", gen_fully_dense(N)),
    ];
    for (name, v) in &dists {
        let targets = gen_targets(v, NUM_TARGETS, SEED ^ 0x9E37);
        let ef = EliasFano::from_sorted(v);
        let pef = PartitionedEliasFano::from_sorted(v);
        let opef = OptimalPartitionedEliasFano::from_sorted(v);
        let cef = ClusteredEliasFano::from_sorted(v);

        group.bench_function(format!("EF/{name}"), |bch| {
            bch.iter(|| {
                for &t in &targets {
                    black_box(ef.next_geq(black_box(t)));
                }
            })
        });
        group.bench_function(format!("PEF/{name}"), |bch| {
            bch.iter(|| {
                for &t in &targets {
                    black_box(pef.next_geq(black_box(t)));
                }
            })
        });
        group.bench_function(format!("OPEF/{name}"), |bch| {
            bch.iter(|| {
                for &t in &targets {
                    black_box(opef.next_geq(black_box(t)));
                }
            })
        });
        group.bench_function(format!("CEF/{name}"), |bch| {
            bch.iter(|| {
                for &t in &targets {
                    black_box(cef.next_geq(black_box(t)));
                }
            })
        });
    }
    group.finish();
}

fn bench_construct(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct");
    let dists: [(&str, Vec<u32>); 3] = [
        ("sparse", gen_uniform_sparse(N, 50, SEED)),
        ("clustered", gen_bursty_clustered(N, SEED)),
        ("dense", gen_fully_dense(N)),
    ];
    for (name, v) in &dists {
        group.bench_function(format!("EF/{name}"), |b| {
            b.iter(|| black_box(EliasFano::from_sorted(black_box(v))))
        });
        group.bench_function(format!("PEF/{name}"), |b| {
            b.iter(|| black_box(PartitionedEliasFano::from_sorted(black_box(v))))
        });
        group.bench_function(format!("OPEF/{name}"), |b| {
            b.iter(|| black_box(OptimalPartitionedEliasFano::from_sorted(black_box(v))))
        });
    }
    group.finish();
}

fn bench_intersect(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect");
    // Two independently-seeded lists per distribution.
    let dists: [(&str, Vec<u32>, Vec<u32>); 3] = [
        (
            "sparse",
            gen_uniform_sparse(N, 50, SEED),
            gen_uniform_sparse(N, 50, SEED ^ 0xABCD),
        ),
        (
            "clustered",
            gen_bursty_clustered(N, SEED),
            gen_bursty_clustered(N, SEED ^ 0xABCD),
        ),
        (
            "dense",
            gen_fully_dense(N),
            gen_fully_dense(N),
        ),
    ];
    for (name, a, b) in &dists {
        let ef_a = EliasFano::from_sorted(a);
        let ef_b = EliasFano::from_sorted(b);
        let pef_a = PartitionedEliasFano::from_sorted(a);
        let pef_b = PartitionedEliasFano::from_sorted(b);
        let opef_a = OptimalPartitionedEliasFano::from_sorted(a);
        let opef_b = OptimalPartitionedEliasFano::from_sorted(b);
        let cef_a = ClusteredEliasFano::from_sorted(a);
        let cef_b = ClusteredEliasFano::from_sorted(b);

        group.bench_function(format!("EF/{name}"), |bch| {
            bch.iter(|| black_box(intersect_count!(&ef_a, &ef_b)))
        });
        group.bench_function(format!("PEF/{name}"), |bch| {
            bch.iter(|| black_box(intersect_count!(&pef_a, &pef_b)))
        });
        group.bench_function(format!("OPEF/{name}"), |bch| {
            bch.iter(|| black_box(intersect_count!(&opef_a, &opef_b)))
        });
        group.bench_function(format!("CEF/{name}"), |bch| {
            bch.iter(|| black_box(intersect_count!(&cef_a, &cef_b)))
        });
        // Block-level intersection (the leapfrog-unreachable fast path).
        group.bench_function(format!("CEF-block/{name}"), |bch| {
            bch.iter(|| black_box(cef_a.intersect_count(&cef_b)))
        });
    }
    group.finish();
}

fn benches(c: &mut Criterion) {
    report_space();
    bench_next_geq(c);
    bench_construct(c);
    bench_intersect(c);
}

criterion_group!(elias_fano, benches);
criterion_main!(elias_fano);
