//! Comprehensive benchmarks for Prefetching + Adaptive SIMD Integration
//!
//! Benchmarks the systematic integration of:
//! - Advanced prefetching strategies (prefetch_rank1, prefetch_select1, lookahead)
//! - Runtime adaptive SIMD selection
//! - Performance monitoring hooks
//!
//! Compares:
//! - Base methods vs optimized methods (with prefetching)
//! - Single operations vs bulk operations (with lookahead prefetching)
//! - Cache-optimized vs adaptive SIMD vs fully optimized

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zipora::succinct::rank_select::interleaved::RankSelectInterleaved256;
use zipora::succinct::BitVector;
use zipora::succinct::rank_select::RankSelectOps;
use zipora::RankSelectPerformanceOps;

/// Create test bitvector with controlled density
fn create_test_bitvector(size: usize, density: f64) -> BitVector {
    let mut bv = BitVector::new();
    let threshold = (density * 1000.0) as usize;
    for i in 0..size {
        bv.push((i * 31) % 1000 < threshold).unwrap();
    }
    bv
}

/// Benchmark: Base rank1 vs optimized rank1 (with prefetching + adaptive SIMD)
fn bench_rank1_base_vs_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_base_vs_optimized");

    let sizes = vec![10_000, 100_000, 1_000_000];
    let densities = vec![0.1, 0.5, 0.9];

    for &size in &sizes {
        for &density in &densities {
            let bv = create_test_bitvector(size, density);
            let rs = RankSelectInterleaved256::new(bv).unwrap();
            let test_pos = size / 2;

            let bench_name = format!("size_{}_density_{}", size, (density * 100.0) as u32);

            group.throughput(Throughput::Elements(1));

            // Baseline: rank1 (standard method without explicit prefetch optimization)
            group.bench_with_input(
                BenchmarkId::new("base_rank1", &bench_name),
                &(&rs, test_pos),
                |b, (rs, pos)| {
                    b.iter(|| {
                        black_box(rs.rank1(*pos))
                    })
                },
            );

            // Optimized: rank1_optimized (with prefetch + adaptive SIMD)
            group.bench_with_input(
                BenchmarkId::new("optimized", &bench_name),
                &(&rs, test_pos),
                |b, (rs, pos)| {
                    b.iter(|| {
                        black_box(rs.rank1_optimized(*pos))
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Bulk operations with/without lookahead prefetching
fn bench_bulk_operations_lookahead(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_operations_lookahead");

    let sizes = vec![10_000, 100_000, 1_000_000];
    let densities = vec![0.1, 0.5, 0.9];
    let batch_sizes = vec![100, 1000, 10000];

    for &size in &sizes {
        for &density in &densities {
            for &batch_size in &batch_sizes {
                if batch_size > size / 10 {
                    continue; // Skip unreasonably large batches
                }

                let bv = create_test_bitvector(size, density);
                let rs = RankSelectInterleaved256::new(bv).unwrap();

                // Create test positions for bulk operations
                let positions: Vec<usize> = (0..batch_size)
                    .map(|i| (i * size / batch_size).min(size - 1))
                    .collect();

                let bench_name = format!(
                    "size_{}_density_{}_batch_{}",
                    size,
                    (density * 100.0) as u32,
                    batch_size
                );

                group.throughput(Throughput::Elements(batch_size as u64));

                // Baseline: Individual rank1 calls (no bulk optimization)
                group.bench_with_input(
                    BenchmarkId::new("individual_calls", &bench_name),
                    &(&rs, &positions),
                    |b, (rs, positions)| {
                        b.iter(|| {
                            let mut results = Vec::with_capacity(positions.len());
                            for &pos in *positions {
                                results.push(rs.rank1(pos));
                            }
                            black_box(results)
                        })
                    },
                );

                // Optimized: rank1_bulk_optimized (with PREFETCH_DISTANCE=8 lookahead)
                group.bench_with_input(
                    BenchmarkId::new("bulk_with_lookahead", &bench_name),
                    &(&rs, &positions),
                    |b, (rs, positions)| {
                        b.iter(|| {
                            black_box(rs.rank1_bulk_optimized(positions))
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark: Select operations with/without prefetching
fn bench_select1_base_vs_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_base_vs_optimized");

    let sizes = vec![10_000, 100_000, 1_000_000];
    let densities = vec![0.1, 0.5, 0.9];

    for &size in &sizes {
        for &density in &densities {
            let bv = create_test_bitvector(size, density);
            let rs = RankSelectInterleaved256::new(bv).unwrap();

            let ones_count = rs.count_ones();
            if ones_count == 0 {
                continue; // Skip empty structures
            }

            let test_id = ones_count / 2;

            let bench_name = format!("size_{}_density_{}", size, (density * 100.0) as u32);

            group.throughput(Throughput::Elements(1));

            // Baseline: select1 (no prefetch, no adaptive)
            group.bench_with_input(
                BenchmarkId::new("base_select1", &bench_name),
                &(&rs, test_id),
                |b, (rs, id)| {
                    b.iter(|| {
                        black_box(rs.select1(*id).ok())
                    })
                },
            );

            // Optimized: select1_optimized (with prefetch + adaptive SIMD)
            group.bench_with_input(
                BenchmarkId::new("optimized", &bench_name),
                &(&rs, test_id),
                |b, (rs, id)| {
                    b.iter(|| {
                        black_box(rs.select1_optimized(*id).ok())
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: Bulk select with lookahead prefetching
fn bench_select_bulk_lookahead(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_bulk_lookahead");

    let sizes = vec![10_000, 100_000, 1_000_000];
    let densities = vec![0.1, 0.5, 0.9];
    let batch_sizes = vec![100, 1000];

    for &size in &sizes {
        for &density in &densities {
            for &batch_size in &batch_sizes {
                let bv = create_test_bitvector(size, density);
                let rs = RankSelectInterleaved256::new(bv).unwrap();

                let ones_count = rs.count_ones();
                if ones_count < batch_size {
                    continue; // Skip if not enough ones
                }

                // Create test IDs for bulk operations
                let ids: Vec<usize> = (0..batch_size)
                    .map(|i| (i * ones_count / batch_size).min(ones_count - 1))
                    .collect();

                let bench_name = format!(
                    "size_{}_density_{}_batch_{}",
                    size,
                    (density * 100.0) as u32,
                    batch_size
                );

                group.throughput(Throughput::Elements(batch_size as u64));

                // Baseline: Individual select1 calls
                group.bench_with_input(
                    BenchmarkId::new("individual_calls", &bench_name),
                    &(&rs, &ids),
                    |b, (rs, ids)| {
                        b.iter(|| {
                            let mut results = Vec::with_capacity(ids.len());
                            for &id in *ids {
                                results.push(rs.select1(id).ok());
                            }
                            black_box(results)
                        })
                    },
                );

                // Optimized: select1_bulk_optimized (with lookahead prefetching)
                group.bench_with_input(
                    BenchmarkId::new("bulk_with_lookahead", &bench_name),
                    &(&rs, &ids),
                    |b, (rs, ids)| {
                        b.iter(|| {
                            black_box(rs.select1_bulk_optimized(ids).ok())
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark: Adaptive SIMD selection overhead
fn bench_adaptive_simd_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_simd_overhead");

    let size = 1_000_000;
    let density = 0.5;

    let bv = create_test_bitvector(size, density);
    let rs = RankSelectInterleaved256::new(bv).unwrap();
    let test_pos = size / 2;

    group.throughput(Throughput::Elements(1));

    // Baseline: Direct rank1 (no adaptive overhead)
    group.bench_function("no_adaptive", |b| {
        b.iter(|| {
            black_box(rs.rank1(test_pos))
        })
    });

    // With adaptive: rank1_adaptive (includes selection + monitoring)
    group.bench_function("with_adaptive", |b| {
        b.iter(|| {
            black_box(rs.rank1_adaptive(test_pos))
        })
    });

    // Fully optimized: rank1_optimized (prefetch + adaptive)
    group.bench_function("fully_optimized", |b| {
        b.iter(|| {
            black_box(rs.rank1_optimized(test_pos))
        })
    });

    group.finish();
}

/// Benchmark: Prefetch-only impact (isolate prefetch benefit)
fn bench_prefetch_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefetch_impact");

    let size = 1_000_000;
    let densities = vec![0.1, 0.5, 0.9];

    for &density in &densities {
        let bv = create_test_bitvector(size, density);
        let rs = RankSelectInterleaved256::new(bv).unwrap();

        // Sequential access pattern (prefetch helps)
        let positions: Vec<usize> = (0..1000).map(|i| i * (size / 1000)).collect();

        let bench_name = format!("density_{}", (density * 100.0) as u32);

        group.throughput(Throughput::Elements(positions.len() as u64));

        // Without explicit prefetch (relies on hardware prefetcher)
        group.bench_with_input(
            BenchmarkId::new("no_prefetch", &bench_name),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += rs.rank1(pos);
                    }
                    black_box(sum)
                })
            },
        );

        // With explicit prefetch (PREFETCH_DISTANCE=8 lookahead)
        group.bench_with_input(
            BenchmarkId::new("with_prefetch", &bench_name),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    black_box(rs.rank1_bulk_optimized(positions))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rank1_base_vs_optimized,
    bench_bulk_operations_lookahead,
    bench_select1_base_vs_optimized,
    bench_select_bulk_lookahead,
    bench_adaptive_simd_overhead,
    bench_prefetch_impact
);

criterion_main!(benches);
