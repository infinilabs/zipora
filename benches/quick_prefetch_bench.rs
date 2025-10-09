//! Quick benchmark for prefetching + adaptive SIMD integration
//! Fast benchmark to verify performance improvements

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zipora::succinct::rank_select::interleaved::RankSelectInterleaved256;
use zipora::succinct::BitVector;
use zipora::succinct::rank_select::RankSelectOps;
use zipora::RankSelectPerformanceOps;

fn create_test_data(size: usize) -> RankSelectInterleaved256 {
    let mut bv = BitVector::new();
    for i in 0..size {
        bv.push(i % 3 == 0).unwrap();
    }
    RankSelectInterleaved256::new(bv).unwrap()
}

fn bench_rank1_comparison(c: &mut Criterion) {
    let rs = create_test_data(1_000_000);
    let test_pos = 500_000;

    let mut group = c.benchmark_group("rank1_quick");

    // Baseline: rank1 (standard)
    group.bench_function("base", |b| {
        b.iter(|| black_box(rs.rank1(test_pos)))
    });

    // Optimized: rank1_optimized (prefetch + adaptive)
    group.bench_function("optimized", |b| {
        b.iter(|| black_box(rs.rank1_optimized(test_pos)))
    });

    // Adaptive only
    group.bench_function("adaptive", |b| {
        b.iter(|| black_box(rs.rank1_adaptive(test_pos)))
    });

    group.finish();
}

fn bench_bulk_comparison(c: &mut Criterion) {
    let rs = create_test_data(1_000_000);
    let positions: Vec<usize> = (0..1000).map(|i| i * 1000).collect();

    let mut group = c.benchmark_group("bulk_quick");

    // Individual calls
    group.bench_function("individual", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(positions.len());
            for &pos in &positions {
                results.push(rs.rank1(pos));
            }
            black_box(results)
        })
    });

    // Bulk with lookahead prefetching
    group.bench_function("bulk_optimized", |b| {
        b.iter(|| black_box(rs.rank1_bulk_optimized(&positions)))
    });

    group.finish();
}

criterion_group!(benches, bench_rank1_comparison, bench_bulk_comparison);
criterion_main!(benches);
