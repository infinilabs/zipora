//! Comprehensive benchmarks for sparse rank-select implementations
//!
//! This benchmark suite tests all sparse optimizations including:
//! - Enhanced RankSelectFew with topling-zip optimizations
//! - BMI2 hardware acceleration with PDEP/PEXT/TZCNT
//! - SortedUintVec with block-based compression

use criterion::{black_box, BenchmarkId, Criterion, criterion_group, criterion_main, Throughput};
use zipora::{
    BitVector, RankSelectOps, RankSelectSimple, RankSelectSeparated256, RankSelectInterleaved256,
    RankSelectFew,
};

/// Benchmark data patterns for comprehensive testing
#[derive(Debug, Clone)]
struct BenchmarkPattern {
    name: &'static str,
    size: usize,
    generator: fn(usize) -> bool,
    expected_density: f64,
    description: &'static str,
}

impl BenchmarkPattern {
    fn generate_bitvector(&self) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..self.size {
            bv.push((self.generator)(i)).unwrap();
        }
        bv
    }
}

/// Standard test patterns based on real-world sparse data characteristics
fn get_test_patterns() -> Vec<BenchmarkPattern> {
    vec![
        BenchmarkPattern {
            name: "ultra_sparse",
            size: 100_000,
            generator: |i| i % 1000 == 0,
            expected_density: 0.001,
            description: "0.1% density - very sparse data like error indicators",
        },
        BenchmarkPattern {
            name: "very_sparse",
            size: 100_000,
            generator: |i| i % 100 == 0,
            expected_density: 0.01,
            description: "1% density - sparse data like rare events",
        },
        BenchmarkPattern {
            name: "sparse",
            size: 100_000,
            generator: |i| i % 20 == 0,
            expected_density: 0.05,
            description: "5% density - moderately sparse data",
        },
        BenchmarkPattern {
            name: "medium_sparse",
            size: 100_000,
            generator: |i| i % 10 == 0,
            expected_density: 0.1,
            description: "10% density - threshold between sparse and dense",
        },
        BenchmarkPattern {
            name: "clustered_sparse",
            size: 100_000,
            generator: |i| (i % 1000 < 10),
            expected_density: 0.01,
            description: "1% density - clustered sparse bits",
        },
        BenchmarkPattern {
            name: "random_sparse",
            size: 100_000,
            generator: |i| (i.wrapping_mul(31337) + 17) % 100 == 0,
            expected_density: 0.01,
            description: "1% density - randomly distributed sparse bits",
        },
    ]
}

/// Benchmark rank operations across different implementations
fn benchmark_sparse_rank_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Rank Operations");
    group.throughput(Throughput::Elements(1000));
    
    let patterns = get_test_patterns();
    
    for pattern in patterns {
        let bv = pattern.generate_bitvector();
        let query_positions: Vec<usize> = (0..1000)
            .map(|i| (i * pattern.size / 1000) % pattern.size)
            .collect();
        
        // Test different implementations
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let few: RankSelectFew<true, 256> = RankSelectFew::from_bit_vector(bv.clone()).unwrap();
        
        // Benchmark each implementation
        group.bench_with_input(
            BenchmarkId::new("simple", pattern.name),
            &(&simple, &query_positions),
            |b, (rs, positions): &(&RankSelectSimple, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("separated256", pattern.name),
            &(&separated, &query_positions),
            |b, (rs, positions): &(&RankSelectSeparated256, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("interleaved256", pattern.name),
            &(&interleaved, &query_positions),
            |b, (rs, positions): &(&RankSelectInterleaved256, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("few", pattern.name),
            &(&few, &query_positions),
            |b, (rs, positions): &(&RankSelectFew<true, 256>, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark select operations for sparse data
fn benchmark_sparse_select_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Select Operations");
    group.throughput(Throughput::Elements(100));
    
    let patterns = get_test_patterns();
    
    for pattern in patterns {
        let bv = pattern.generate_bitvector();
        let ones_count = bv.count_ones();
        
        if ones_count == 0 {
            continue; // Skip patterns with no ones
        }
        
        let query_ranks: Vec<usize> = (0..100)
            .map(|i| (i * ones_count / 100).min(ones_count - 1))
            .collect();
        
        // Test different implementations
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        let few: RankSelectFew<true, 256> = RankSelectFew::from_bit_vector(bv.clone()).unwrap();
        
        // Benchmark each implementation
        group.bench_with_input(
            BenchmarkId::new("simple", pattern.name),
            &(&simple, &query_ranks),
            |b, (rs, ranks): &(&RankSelectSimple, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &rank in *ranks {
                        if let Ok(pos) = rs.select1(black_box(rank)) {
                            sum += pos;
                        }
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("separated256", pattern.name),
            &(&separated, &query_ranks),
            |b, (rs, ranks): &(&RankSelectSeparated256, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &rank in *ranks {
                        if let Ok(pos) = rs.select1(black_box(rank)) {
                            sum += pos;
                        }
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("few", pattern.name),
            &(&few, &query_ranks),
            |b, (rs, ranks): &(&RankSelectFew<true, 256>, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &rank in *ranks {
                        if let Ok(pos) = rs.select1(black_box(rank)) {
                            sum += pos;
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark construction time for different implementations
fn benchmark_sparse_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Construction");
    
    let patterns = get_test_patterns();
    
    for pattern in patterns {
        let bv = pattern.generate_bitvector();
        
        group.bench_with_input(
            BenchmarkId::new("simple", pattern.name),
            &bv,
            |b, bv: &BitVector| {
                b.iter(|| {
                    let rs = RankSelectSimple::new(bv.clone()).unwrap();
                    black_box(rs)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("separated256", pattern.name),
            &bv,
            |b, bv: &BitVector| {
                b.iter(|| {
                    let rs = RankSelectSeparated256::new(bv.clone()).unwrap();
                    black_box(rs)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("few", pattern.name),
            &bv,
            |b, bv: &BitVector| {
                b.iter(|| {
                    let rs: RankSelectFew<true, 256> = RankSelectFew::from_bit_vector(bv.clone()).unwrap();
                    black_box(rs)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Efficiency");
    
    let patterns = get_test_patterns();
    
    for pattern in patterns {
        let bv = pattern.generate_bitvector();
        
        // Create all implementations and measure their size overhead
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        let few: RankSelectFew<true, 256> = RankSelectFew::from_bit_vector(bv.clone()).unwrap();
        
        // We can't directly measure memory usage in benchmarks, but we can
        // benchmark operations that stress memory hierarchy
        let positions: Vec<usize> = (0..1000)
            .map(|i| (i * pattern.size / 1000) % pattern.size)
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("simple_memory_stress", pattern.name),
            &(&simple, &positions),
            |b, (rs, positions): &(&RankSelectSimple, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    // Stress memory with random access pattern
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                        sum += rs.rank0(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("few_memory_stress", pattern.name),
            &(&few, &positions),
            |b, (rs, positions): &(&RankSelectFew<true, 256>, &Vec<usize>)| {
                b.iter(|| {
                    let mut sum = 0;
                    // Stress memory with random access pattern
                    for &pos in *positions {
                        sum += rs.rank1(black_box(pos));
                        sum += rs.rank0(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_sparse_rank_operations,
    benchmark_sparse_select_operations,
    benchmark_sparse_construction,
    benchmark_memory_efficiency,
);

criterion_main!(benches);