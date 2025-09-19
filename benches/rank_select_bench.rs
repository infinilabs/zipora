//! Comprehensive benchmarks for rank/select variants
//!
//! This module provides performance benchmarks for all rank/select implementations,
//! comparing them against each other and validating performance against C++ baseline.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;
use zipora::{
    BitVector, RankSelect256,
    succinct::rank_select::{
        RankSelectInterleaved256, RankSelectOps, SimdCapabilities, bulk_popcount_simd, bulk_rank1_simd,
        bulk_select1_simd,
    },
};

// Test data configurations
const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 100_000;
const LARGE_SIZE: usize = 10_000_000;
const WARMUP_TIME: Duration = Duration::from_millis(100);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
const SAMPLE_SIZE: usize = 100;

/// Create test bit vectors with different patterns
fn create_test_data(size: usize, pattern: &str) -> BitVector {
    let mut bv = BitVector::new();

    match pattern {
        "alternating" => {
            for i in 0..size {
                bv.push(i % 2 == 0).unwrap();
            }
        }
        "sparse" => {
            for i in 0..size {
                bv.push(i % 100 == 0).unwrap(); // 1% density
            }
        }
        "dense" => {
            for i in 0..size {
                bv.push(i % 4 != 3).unwrap(); // 75% density
            }
        }
        "random" => {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            for i in 0..size {
                let mut hasher = DefaultHasher::new();
                i.hash(&mut hasher);
                bv.push(hasher.finish() % 2 == 0).unwrap();
            }
        }
        _ => panic!("Unknown pattern: {}", pattern),
    }

    bv
}

/// Generate test positions for rank operations
fn generate_rank_positions(size: usize, count: usize) -> Vec<usize> {
    (0..count).map(|i| (i * size) / count).collect()
}

/// Generate test indices for select operations  
fn generate_select_indices(total_ones: usize, count: usize) -> Vec<usize> {
    if total_ones == 0 {
        return vec![];
    }
    (0..count.min(total_ones))
        .map(|i| (i * total_ones) / count)
        .collect()
}

/// Benchmark rank operations for all variants
fn benchmark_rank_operations(c: &mut Criterion) {
    let sizes = vec![SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    let patterns = vec!["alternating", "sparse", "dense", "random"];

    for &size in &sizes {
        for pattern in &patterns {
            let mut group = c.benchmark_group(format!("rank_{}_{}", size, pattern));
            group.warm_up_time(WARMUP_TIME);
            group.measurement_time(MEASUREMENT_TIME);
            group.sample_size(SAMPLE_SIZE);
            group.throughput(Throughput::Elements(size as u64));

            let bv = create_test_data(size, pattern);
            let positions = generate_rank_positions(size, 1000);

            // Benchmark all variants
            let simple = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("simple", |b| {
                b.iter(|| {
                    for &pos in &positions {
                        black_box(simple.rank1(pos));
                    }
                })
            });

            let separated256 = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("separated256", |b| {
                b.iter(|| {
                    for &pos in &positions {
                        black_box(separated256.rank1(pos));
                    }
                })
            });

            let separated512 = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("separated512", |b| {
                b.iter(|| {
                    for &pos in &positions {
                        black_box(separated512.rank1(pos));
                    }
                })
            });

            let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
            group.bench_function("interleaved256", |b| {
                b.iter(|| {
                    for &pos in &positions {
                        black_box(interleaved.rank1(pos));
                    }
                })
            });

            // Only benchmark sparse variant on sparse data
            if *pattern == "sparse" {
                if let Ok(sparse) = RankSelect256::new(bv.clone()) {
                    group.bench_function("sparse", |b| {
                        b.iter(|| {
                            for &pos in &positions {
                                black_box(sparse.rank1(pos));
                            }
                        })
                    });
                }
            }

            group.finish();
        }
    }
}

/// Benchmark select operations for all variants
fn benchmark_select_operations(c: &mut Criterion) {
    let sizes = vec![SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    let patterns = vec!["alternating", "sparse", "dense", "random"];

    for &size in &sizes {
        for pattern in &patterns {
            let mut group = c.benchmark_group(format!("select_{}_{}", size, pattern));
            group.warm_up_time(WARMUP_TIME);
            group.measurement_time(MEASUREMENT_TIME);
            group.sample_size(SAMPLE_SIZE);

            let bv = create_test_data(size, pattern);
            let total_ones = bv.count_ones();

            if total_ones == 0 {
                continue; // Skip if no set bits
            }

            group.throughput(Throughput::Elements(total_ones as u64));
            let indices = generate_select_indices(total_ones, 100);

            // Benchmark all variants
            let simple = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("simple", |b| {
                b.iter(|| {
                    for &idx in &indices {
                        if let Ok(pos) = simple.select1(idx) {
                            black_box(pos);
                        }
                    }
                })
            });

            let separated256 = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("separated256", |b| {
                b.iter(|| {
                    for &idx in &indices {
                        if let Ok(pos) = separated256.select1(idx) {
                            black_box(pos);
                        }
                    }
                })
            });

            let separated512 = RankSelect256::new(bv.clone()).unwrap();
            group.bench_function("separated512", |b| {
                b.iter(|| {
                    for &idx in &indices {
                        if let Ok(pos) = separated512.select1(idx) {
                            black_box(pos);
                        }
                    }
                })
            });

            let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
            group.bench_function("interleaved256", |b| {
                b.iter(|| {
                    for &idx in &indices {
                        if let Ok(pos) = interleaved.select1(idx) {
                            black_box(pos);
                        }
                    }
                })
            });

            // Only benchmark sparse variant on sparse data
            if *pattern == "sparse" {
                if let Ok(sparse) = RankSelect256::new(bv.clone()) {
                    group.bench_function("sparse", |b| {
                        b.iter(|| {
                            for &idx in &indices {
                                if let Ok(pos) = sparse.select1(idx) {
                                    black_box(pos);
                                }
                            }
                        })
                    });
                }
            }

            group.finish();
        }
    }
}

// /// Benchmark multi-dimensional variants (DISABLED - implementations removed)
// fn benchmark_multi_dimensional_operations(c: &mut Criterion) {
//     // Multi-dimensional rank-select implementations have been consolidated
//     // This benchmark is disabled until new multi-dimensional support is added
// }

/// Benchmark SIMD bulk operations
fn benchmark_simd_operations(c: &mut Criterion) {
    let sizes = vec![MEDIUM_SIZE, LARGE_SIZE];
    let patterns = vec!["alternating", "dense", "random"];

    for &size in &sizes {
        for pattern in &patterns {
            let mut group = c.benchmark_group(format!("simd_{}_{}", size, pattern));
            group.warm_up_time(WARMUP_TIME);
            group.measurement_time(MEASUREMENT_TIME);
            group.sample_size(SAMPLE_SIZE);
            group.throughput(Throughput::Elements(size as u64));

            let bv = create_test_data(size, pattern);
            let bit_data: Vec<u64> = bv.blocks().to_vec();
            let positions = generate_rank_positions(size, 1000);
            let total_ones = bv.count_ones();
            let indices = generate_select_indices(total_ones, 100);

            // Benchmark bulk popcount
            group.bench_function("bulk_popcount", |b| {
                b.iter(|| {
                    black_box(bulk_popcount_simd(&bit_data));
                })
            });

            // Benchmark bulk rank
            group.bench_function("bulk_rank", |b| {
                b.iter(|| {
                    black_box(bulk_rank1_simd(&bit_data, &positions));
                })
            });

            // Benchmark bulk select (only if we have set bits)
            if !indices.is_empty() {
                group.bench_function("bulk_select", |b| {
                    b.iter(|| {
                        if let Ok(result) = bulk_select1_simd(&bit_data, &indices) {
                            black_box(result);
                        }
                    })
                });
            }

            group.finish();
        }
    }
}

/// Benchmark space overhead
fn benchmark_space_overhead(c: &mut Criterion) {
    let sizes = vec![SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    let patterns = vec!["alternating", "sparse", "dense"];

    for &size in &sizes {
        for pattern in &patterns {
            let mut group = c.benchmark_group(format!("space_{}_{}", size, pattern));

            let bv = create_test_data(size, pattern);
            let original_bytes = bv.len() / 8 + if bv.len() % 8 > 0 { 1 } else { 0 };

            // Test all variants
            let simple = RankSelect256::new(bv.clone()).unwrap();
            let overhead_simple = simple.space_overhead_percent();

            let separated256 = RankSelect256::new(bv.clone()).unwrap();
            let overhead_separated256 = separated256.space_overhead_percent();

            let separated512 = RankSelect256::new(bv.clone()).unwrap();
            let overhead_separated512 = separated512.space_overhead_percent();

            let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
            let overhead_interleaved = interleaved.space_overhead_percent();

            eprintln!(
                "Size: {}, Pattern: {}, Original: {} bytes",
                size, pattern, original_bytes
            );
            eprintln!("  Simple:       {:.2}% overhead", overhead_simple);
            eprintln!("  Separated256: {:.2}% overhead", overhead_separated256);
            eprintln!("  Separated512: {:.2}% overhead", overhead_separated512);
            eprintln!("  Interleaved:  {:.2}% overhead", overhead_interleaved);

            // Test sparse variant on sparse data
            if *pattern == "sparse" {
                if let Ok(sparse) = RankSelect256::new(bv.clone()) {
                    let space_overhead = sparse.space_overhead_percent();
                    eprintln!(
                        "  Sparse:       {:.2}% space overhead",
                        space_overhead
                    );
                }
            }

            group.finish();
        }
    }
}

/// Benchmark construction time
fn benchmark_construction(c: &mut Criterion) {
    let sizes = vec![SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];
    let patterns = vec!["alternating", "sparse", "dense"];

    for &size in &sizes {
        for pattern in &patterns {
            let mut group = c.benchmark_group(format!("construction_{}_{}", size, pattern));
            group.warm_up_time(WARMUP_TIME);
            group.measurement_time(MEASUREMENT_TIME);
            group.sample_size(SAMPLE_SIZE);
            group.throughput(Throughput::Elements(size as u64));

            let bv = create_test_data(size, pattern);

            group.bench_function("simple", |b| {
                b.iter(|| {
                    black_box(RankSelect256::new(bv.clone()).unwrap());
                })
            });

            group.bench_function("separated256", |b| {
                b.iter(|| {
                    black_box(RankSelect256::new(bv.clone()).unwrap());
                })
            });

            group.bench_function("separated512", |b| {
                b.iter(|| {
                    black_box(RankSelect256::new(bv.clone()).unwrap());
                })
            });

            group.bench_function("interleaved256", |b| {
                b.iter(|| {
                    black_box(RankSelectInterleaved256::new(bv.clone()).unwrap());
                })
            });

            // Only benchmark sparse construction on appropriate data
            if *pattern == "sparse" || *pattern == "dense" {
                group.bench_function("sparse", |b| {
                    b.iter(|| {
                        if let Ok(sparse) = RankSelect256::new(bv.clone()) {
                            black_box(sparse);
                        }
                    })
                });
            }

            group.finish();
        }
    }
}

/// Print SIMD capabilities and benchmark configuration
fn print_benchmark_info() {
    let caps = SimdCapabilities::get();
    eprintln!("=== Rank/Select Benchmark Configuration ===");
    eprintln!("SIMD Optimization Tier: {}", caps.optimization_tier);
    eprintln!("Chunk Size: {} bytes", caps.chunk_size);
    eprintln!("Prefetch Enabled: {}", caps.use_prefetch);
    eprintln!("CPU Features:");
    eprintln!("  POPCNT: {}", caps.cpu_features.has_popcnt);
    eprintln!("  BMI2: {}", caps.cpu_features.has_bmi2);
    eprintln!("  AVX2: {}", caps.cpu_features.has_avx2);

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    eprintln!("  AVX-512: {}", caps.cpu_features.has_avx512vpopcntdq);

    eprintln!("Test Sizes: {:?}", [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE]);
    eprintln!("Measurement Time: {:?}", MEASUREMENT_TIME);
    eprintln!("Sample Size: {}", SAMPLE_SIZE);
    eprintln!("==========================================");
}

criterion_group!(
    benches,
    benchmark_rank_operations,
    benchmark_select_operations,
    benchmark_simd_operations,
    benchmark_space_overhead,
    benchmark_construction
);

criterion_main!(benches);
