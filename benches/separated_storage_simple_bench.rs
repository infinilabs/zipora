//! Simplified Benchmarks for Separated Storage Configuration System
//!
//! This benchmark validates that the configuration system delivers expected
//! performance improvements across different settings.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, black_box
};
use std::time::Duration;
use zipora::{
    BitVector,
    succinct::rank_select::{
        RankSelectOps, RankSelectSeparated256, RankSelectSimple,
        SeparatedStorageConfig,
    },
};

// Benchmark configuration constants
const WARMUP_TIME: Duration = Duration::from_millis(300);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
const SAMPLE_SIZE: usize = 50;

// Test data sizes
const SIZES: &[usize] = &[10_000, 100_000];

/// Generate bit vector with specific characteristics
fn create_bit_vector(size: usize, pattern: DataPattern) -> BitVector {
    let mut bv = BitVector::new();
    
    match pattern {
        DataPattern::Uniform(density) => {
            // Use deterministic "random" for reproducible benchmarks
            let threshold = (density * 1000.0) as usize;
            for i in 0..size {
                bv.push((i * 31 + 17) % 1000 < threshold).unwrap();
            }
        }
        DataPattern::Sparse => {
            for i in 0..size {
                bv.push(i % 100 == 0).unwrap(); // 1% density
            }
        }
        DataPattern::Dense => {
            for i in 0..size {
                bv.push(i % 10 != 0).unwrap(); // 90% density
            }
        }
    }
    
    bv
}

/// Data patterns for testing different algorithmic behaviors
#[derive(Debug, Clone, Copy)]
enum DataPattern {
    Uniform(f64),  // Uniform random with specified density
    Sparse,        // Very few 1s
    Dense,         // Mostly 1s
}

impl DataPattern {
    fn name(&self) -> &'static str {
        match self {
            DataPattern::Uniform(d) => if *d < 0.1 { "uniform_sparse" } 
                                      else if *d > 0.9 { "uniform_dense" } 
                                      else { "uniform_balanced" },
            DataPattern::Sparse => "sparse",
            DataPattern::Dense => "dense", 
        }
    }
}

/// Benchmark different configuration strategies
fn bench_configuration_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("Configuration_Strategies");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    let test_patterns = vec![
        DataPattern::Uniform(0.1),
        DataPattern::Uniform(0.5), 
        DataPattern::Dense,
    ];

    for &size in SIZES {
        for pattern in &test_patterns {
            let bv = create_bit_vector(size, *pattern);
            
            // Test different configuration strategies
            let configs = vec![
                ("default", SeparatedStorageConfig::default()),
                ("high_performance", SeparatedStorageConfig::high_performance().build()),
                ("low_memory", SeparatedStorageConfig::low_memory().build()),
            ];
            
            for (strategy_name, config) in configs {
                let rs = RankSelectSeparated256::with_config(bv.clone(), config).unwrap();
                let test_positions: Vec<usize> = (0..size).step_by((size / 100).max(1)).collect();
                
                group.throughput(Throughput::Elements(test_positions.len() as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("rank_{}_{}_size_{}", strategy_name, pattern.name(), size),
                        size
                    ),
                    &(&rs, &test_positions),
                    |b, (rs, positions)| {
                        b.iter(|| {
                            let mut sum = 0;
                            for &pos in *positions {
                                sum += black_box(rs.rank1(pos));
                            }
                            black_box(sum)
                        })
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark memory overhead vs performance trade-offs
fn bench_memory_overhead_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory_Overhead_Analysis");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 100_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Test different block sizes and their impact
    let block_configs = vec![
        ("block_256", SeparatedStorageConfig::new().block_size(256).build()),
        ("block_512", SeparatedStorageConfig::new().block_size(512).build()),
    ];
    
    for (config_name, config) in block_configs {
        // Print memory overhead for analysis
        println!("Config {}: estimated overhead = {:.2}%", 
                config_name, config.estimated_memory_overhead());
        
        let rs = RankSelectSeparated256::with_config(bv.clone(), config).unwrap();
        let actual_overhead = rs.space_overhead_percent();
        println!("Config {}: actual overhead = {:.2}%", config_name, actual_overhead);
        
        let test_positions: Vec<usize> = (0..size).step_by(size / 1000).collect();
        
        group.throughput(Throughput::Elements(test_positions.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("rank_{}", config_name), size),
            &(&rs, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += black_box(rs.rank1(pos));
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark scalability across different data sizes
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scalability");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let optimal_config = SeparatedStorageConfig::high_performance().build();
    
    for &size in SIZES {
        let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
        let rs = RankSelectSeparated256::with_config(bv, optimal_config.clone()).unwrap();
        
        // Test with fixed number of operations to measure per-operation performance
        let num_ops = 1000;
        let test_positions: Vec<usize> = (0..num_ops)
            .map(|i| (i * size) / num_ops)
            .collect();
        
        group.throughput(Throughput::Elements(num_ops as u64));
        group.bench_with_input(
            BenchmarkId::new("rank_scalability", size),
            &(&rs, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += black_box(rs.rank1(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark select scalability
        let ones_count = rs.count_ones();
        if ones_count > num_ops {
            let select_indices: Vec<usize> = (0..num_ops)
                .map(|i| (i * ones_count) / num_ops)
                .collect();
            
            group.throughput(Throughput::Elements(num_ops as u64));
            group.bench_with_input(
                BenchmarkId::new("select_scalability", size),
                &(&rs, &select_indices),
                |b, (rs, indices)| {
                    b.iter(|| {
                        let mut sum = 0;
                        for &idx in *indices {
                            if let Ok(pos) = rs.select1(idx) {
                                sum += black_box(pos);
                            }
                        }
                        black_box(sum)
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark baseline comparison with simple implementation
fn bench_baseline_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Baseline_Comparison");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 50_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Compare different implementations
    let simple_rs = RankSelectSimple::new(bv.clone()).unwrap();
    let separated_rs = RankSelectSeparated256::with_config(
        bv.clone(), SeparatedStorageConfig::high_performance().build()
    ).unwrap();
    
    let test_positions: Vec<usize> = (0..size).step_by(size / 1000).collect();
    
    let implementations = vec![
        ("simple", &simple_rs as &dyn RankSelectOps),
        ("separated_256", &separated_rs as &dyn RankSelectOps),
    ];
    
    for (impl_name, rs) in implementations {
        group.throughput(Throughput::Elements(test_positions.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("rank_{}", impl_name), size),
            &(rs, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        sum += black_box(rs.rank1(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Compare space overhead
        println!("{} space overhead: {:.2}%", impl_name, rs.space_overhead_percent());
    }
    
    group.finish();
}

/// Print benchmark configuration information
fn print_benchmark_info() {
    println!("=== Separated Storage Benchmark Configuration ===");
    println!("Test Sizes: {:?}", SIZES);
    println!("Measurement Time: {:?}", MEASUREMENT_TIME);
    println!("Sample Size: {}", SAMPLE_SIZE);
    println!("==============================================");
}

criterion_group!(
    benches,
    bench_configuration_strategies,
    bench_memory_overhead_analysis,
    bench_scalability,
    bench_baseline_comparison,
);

criterion_main!(benches);