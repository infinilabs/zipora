//! Comprehensive Benchmarks for Separated Storage Configuration System
//!
//! This benchmark suite validates that the configuration system delivers expected
//! performance improvements across different data patterns, sizes, and hardware
//! configurations. Based on research and optimization patterns.
//!
//! # Benchmark Categories
//!
//! 1. **Configuration Strategy Comparison**: Different memory strategies and layouts
//! 2. **Hardware Acceleration Validation**: BMI2, POPCNT, SIMD impact
//! 3. **Multi-dimensional Performance**: Cache sharing vs separated strategies  
//! 4. **Memory Overhead Analysis**: Space/performance trade-offs
//! 5. **Adaptive Selection Validation**: Automatic configuration vs manual tuning
//! 6. **Scale Testing**: Performance across different data sizes
//! 7. **Access Pattern Analysis**: Random vs sequential vs sparse access patterns

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, black_box
};
use std::time::Duration;
use zipora::{
    BitVector,
    succinct::rank_select::{
        RankSelectOps, RankSelectSeparated256, RankSelectSeparated512,
        RankSelectMixedIL256, RankSelectMixedSE512, RankSelectMixedXL256,
        SeparatedStorageConfig, FeatureDetection, HardwareOptimizations,
        RankSelectSimple, // for baseline comparison
    },
    CpuFeatures,
};

// Benchmark configuration constants
const WARMUP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(3);
const SAMPLE_SIZE: usize = 100;

// Test data sizes
const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000, 10_000_000];
const SMALL_SIZES: &[usize] = &[1_000, 10_000, 100_000]; // For expensive tests
const LARGE_SIZES: &[usize] = &[1_000_000, 10_000_000]; // For scale tests

// Data density patterns
const DENSITIES: &[f64] = &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];

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
        DataPattern::Alternating => {
            for i in 0..size {
                bv.push(i % 2 == 0).unwrap();
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
        DataPattern::Clustered => {
            // Clusters of 1s and 0s
            for i in 0..size {
                let cluster = (i / 64) % 4;
                bv.push(cluster < 3).unwrap(); // 75% density with clustering
            }
        }
        DataPattern::Binary => {
            for i in 0..size {
                bv.push(i < size / 2).unwrap(); // First half 1s, second half 0s
            }
        }
    }
    
    bv
}

/// Data patterns for testing different algorithmic behaviors
#[derive(Debug, Clone, Copy)]
enum DataPattern {
    Uniform(f64),  // Uniform random with specified density
    Alternating,   // 010101...
    Sparse,        // Very few 1s
    Dense,         // Mostly 1s
    Clustered,     // Grouped patterns
    Binary,        // Half 0s, half 1s
}

impl DataPattern {
    fn name(&self) -> &'static str {
        match self {
            DataPattern::Uniform(d) => if *d < 0.1 { "uniform_sparse" } 
                                      else if *d > 0.9 { "uniform_dense" } 
                                      else { "uniform_balanced" },
            DataPattern::Alternating => "alternating",
            DataPattern::Sparse => "sparse",
            DataPattern::Dense => "dense", 
            DataPattern::Clustered => "clustered",
            DataPattern::Binary => "binary",
        }
    }
}

/// Benchmark different configuration strategies
fn bench_configuration_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("Configuration Strategies");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    let test_patterns = vec![
        DataPattern::Uniform(0.1),
        DataPattern::Uniform(0.5), 
        DataPattern::Uniform(0.9),
    ];

    for &size in SMALL_SIZES {
        for pattern in &test_patterns {
            let bv = create_bit_vector(size, *pattern);
            
            // Test different configuration strategies
            let configs = vec![
                ("default", SeparatedStorageConfig::default()),
                ("high_performance", SeparatedStorageConfig::high_performance().build()),
                ("low_memory", SeparatedStorageConfig::low_memory().build()),
                ("adaptive", SeparatedStorageConfig::analyze_and_optimize(&bv).build()),
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

/// Benchmark hardware acceleration impact
fn bench_hardware_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hardware Acceleration");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    // Detect available CPU features
    let cpu_features = CpuFeatures::get();
    println!("CPU Features: POPCNT={}, BMI2={}, AVX2={}", 
             cpu_features.has_popcnt, cpu_features.has_bmi2, cpu_features.has_avx2);
    
    let size = 100_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Test with different hardware acceleration settings
    let hw_configs = vec![
        ("disabled", SeparatedStorageConfig::new()
            .hardware_optimizations(HardwareOptimizations {
                enable_bmi2: false,
                enable_bmi1: false,
                enable_simd: false,
                enable_avx512: false,
                enable_prefetch: false,
                feature_detection: FeatureDetection::Disable,
            })
            .build()),
        ("bmi1_only", SeparatedStorageConfig::new()
            .hardware_optimizations(HardwareOptimizations {
                enable_bmi2: false,
                enable_bmi1: true,
                enable_simd: false,
                enable_avx512: false,
                enable_prefetch: false,
                feature_detection: FeatureDetection::Runtime,
            })
            .build()),
        ("full_accel", SeparatedStorageConfig::new()
            .enable_hardware_acceleration(true)
            .build()),
    ];
    
    for (hw_name, config) in hw_configs {
        let rs = RankSelectSeparated256::with_config(bv.clone(), config).unwrap();
        let test_positions: Vec<usize> = (0..size).step_by(size / 1000).collect();
        
        group.throughput(Throughput::Elements(test_positions.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("rank_{}", hw_name), size),
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
        
        // Benchmark select operations (more sensitive to hardware acceleration)
        let ones_count = rs.count_ones();
        if ones_count > 1000 {
            let select_indices: Vec<usize> = (0..ones_count).step_by(ones_count / 100).collect();
            
            group.throughput(Throughput::Elements(select_indices.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("select_{}", hw_name), size),
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

/// Benchmark multi-dimensional configurations 
fn bench_multi_dimensional_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multi-Dimensional Configs");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 50_000; // Smaller for multi-dimensional tests
    
    // Create correlated bit vectors for multi-dimensional testing
    let bv1 = create_bit_vector(size, DataPattern::Uniform(0.3));
    let bv2 = create_bit_vector(size, DataPattern::Uniform(0.7));
    let bv3 = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Test different multi-dimensional strategies
    let rs_il = RankSelectMixedIL256::new([bv1.clone(), bv2.clone()]).unwrap();
    let rs_se = RankSelectMixedSE512::new([bv1.clone(), bv2.clone()]).unwrap();
    
    let _multi_configs = vec![
        ("interleaved_default", &rs_il as &dyn RankSelectOps),
        ("separated_default", &rs_se as &dyn RankSelectOps),
    ];
    
    let test_positions: Vec<usize> = (0..size).step_by((size / 100).max(1)).collect();
    
    // Benchmark interleaved variant
    group.throughput(Throughput::Elements(test_positions.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("rank_interleaved_dim0", size),
        &(&rs_il, &test_positions),
        |b, (rs, positions)| {
            b.iter(|| {
                let mut sum = 0;
                for &pos in *positions {
                    sum += black_box(rs.rank1_dimension(pos, 0));
                }
                black_box(sum)
            })
        },
    );
    
    group.bench_with_input(
        BenchmarkId::new("rank_interleaved_dim1", size),
        &(&rs_il, &test_positions),
        |b, (rs, positions)| {
            b.iter(|| {
                let mut sum = 0;
                for &pos in *positions {
                    sum += black_box(rs.rank1_dimension(pos, 1));
                }
                black_box(sum)
            })
        },
    );
    
    // Benchmark separated variant  
    group.bench_with_input(
        BenchmarkId::new("rank_separated_dim0", size),
        &(&rs_se, &test_positions),
        |b, (rs, positions)| {
            b.iter(|| {
                let mut sum = 0;
                for &pos in *positions {
                    sum += black_box(rs.rank1_dimension(pos, 0));
                }
                black_box(sum)
            })
        },
    );
    
    group.bench_with_input(
        BenchmarkId::new("rank_separated_dim1", size),
        &(&rs_se, &test_positions),
        |b, (rs, positions)| {
            b.iter(|| {
                let mut sum = 0;
                for &pos in *positions {
                    sum += black_box(rs.rank1_dimension(pos, 1));
                }
                black_box(sum)
            })
        },
    );
    
    // Test 3-dimensional configuration with RankSelectMixedXL256
    let rs_3d = RankSelectMixedXL256::<3>::new([bv1, bv2, bv3]).unwrap();
    let test_positions: Vec<usize> = (0..size).step_by((size / 100).max(1)).collect();
    
    for dim in 0..3 {
        group.throughput(Throughput::Elements(test_positions.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("rank_3d_dim{}", dim), size),
            &(&rs_3d, &test_positions, dim),
            |b, (rs, positions, dimension)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in *positions {
                        match dimension {
                            0 => sum += black_box(rs.rank1_dimension::<0>(pos)),
                            1 => sum += black_box(rs.rank1_dimension::<1>(pos)),
                            2 => sum += black_box(rs.rank1_dimension::<2>(pos)),
                            _ => unreachable!(),
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory overhead vs performance trade-offs
fn bench_memory_overhead_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Overhead Analysis");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 100_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Test different block sizes and their impact
    let block_configs = vec![
        ("block_256", SeparatedStorageConfig::new().block_size(256).build()),
        ("block_512", SeparatedStorageConfig::new().block_size(512).build()),
        ("block_1024", SeparatedStorageConfig::new().block_size(1024).build()),
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
    
    // Test select cache density impact
    let select_configs = vec![
        ("no_select", SeparatedStorageConfig::new().enable_select_acceleration(false).build()),
        ("select_1024", SeparatedStorageConfig::new().select_sample_rate(1024).build()),
        ("select_512", SeparatedStorageConfig::new().select_sample_rate(512).build()),
        ("select_256", SeparatedStorageConfig::new().select_sample_rate(256).build()),
    ];
    
    for (config_name, config) in select_configs {
        let rs = RankSelectSeparated256::with_config(bv.clone(), config).unwrap();
        let ones_count = rs.count_ones();
        
        if ones_count > 100 {
            let select_indices: Vec<usize> = (0..ones_count).step_by((ones_count / 100).max(1)).collect();
            
            group.throughput(Throughput::Elements(select_indices.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("select_{}", config_name), size),
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

/// Benchmark adaptive configuration selection
fn bench_adaptive_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive Selection");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let test_cases = vec![
        (1_000, DataPattern::Sparse),      // Small sparse
        (1_000, DataPattern::Dense),       // Small dense  
        (100_000, DataPattern::Uniform(0.5)), // Medium balanced
        (1_000_000, DataPattern::Alternating), // Large regular pattern
    ];
    
    for (size, pattern) in test_cases {
        let bv = create_bit_vector(size, pattern);
        
        // Compare manual configuration vs adaptive
        let manual_config = SeparatedStorageConfig::high_performance().build();
        let adaptive_config = SeparatedStorageConfig::analyze_and_optimize(&bv).build();
        
        let configs = vec![
            ("manual", manual_config),
            ("adaptive", adaptive_config),
        ];
        
        for (config_type, config) in configs {
            let rs = RankSelectSeparated256::with_config(bv.clone(), config).unwrap();
            let test_positions: Vec<usize> = (0..size).step_by((size / 100).max(1)).collect();
            
            group.throughput(Throughput::Elements(test_positions.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("rank_{}_{}_size_{}", config_type, pattern.name(), size),
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

/// Benchmark different access patterns
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Access Patterns");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 100_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    let rs = RankSelectSeparated256::with_config(
        bv, SeparatedStorageConfig::high_performance().build()
    ).unwrap();
    
    // Different access patterns
    let access_patterns = vec![
        ("sequential", (0..size).step_by(size / 1000).collect::<Vec<_>>()),
        ("random", {
            let mut positions = Vec::new();
            for i in 0..1000 {
                positions.push((i * 31 + 17) % size);
            }
            positions
        }),
        ("clustered", {
            let mut positions = Vec::new();
            for cluster in 0..10 {
                let base = (cluster * size) / 10;
                for i in 0..100 {
                    positions.push(base + i);
                }
            }
            positions
        }),
        ("stride", (0..1000).map(|i| (i * 97) % size).collect()),
    ];
    
    for (pattern_name, positions) in access_patterns {
        group.throughput(Throughput::Elements(positions.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("rank_{}", pattern_name), size),
            &(&rs, &positions),
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

/// Benchmark baseline comparison with simple implementation
fn bench_baseline_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Baseline Comparison");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    
    let size = 100_000;
    let bv = create_bit_vector(size, DataPattern::Uniform(0.5));
    
    // Compare different implementations
    let simple_rs = RankSelectSimple::new(bv.clone()).unwrap();
    let separated_rs = RankSelectSeparated256::with_config(
        bv.clone(), SeparatedStorageConfig::high_performance().build()
    ).unwrap();
    let separated_512_rs = RankSelectSeparated512::new(bv).unwrap();
    
    let test_positions: Vec<usize> = (0..size).step_by(size / 1000).collect();
    
    let implementations = vec![
        ("simple", &simple_rs as &dyn RankSelectOps),
        ("separated_256", &separated_rs as &dyn RankSelectOps),
        ("separated_512", &separated_512_rs as &dyn RankSelectOps),
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


criterion_group!(
    benches,
    bench_configuration_strategies,
    bench_hardware_acceleration,
    bench_multi_dimensional_configs,
    bench_memory_overhead_analysis,
    bench_adaptive_selection,
    bench_scalability,
    bench_access_patterns,
    bench_baseline_comparison,
);

criterion_main!(benches);