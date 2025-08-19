//! Comprehensive benchmarks for sparse rank-select implementations
//!
//! This benchmark suite tests all sparse optimizations including:
//! - Enhanced RankSelectFew with topling-zip optimizations
//! - BMI2 hardware acceleration with PDEP/PEXT/TZCNT
//! - AdaptiveRankSelect with automatic threshold tuning
//! - SortedUintVec with block-based compression
//! - Pattern analysis and adaptive strategy selection

use criterion::{black_box, BenchmarkId, Criterion, criterion_group, criterion_main, Throughput};
use zipora::{
    BitVector, RankSelectOps, RankSelectSimple, RankSelectSeparated256, RankSelectInterleaved256,
    RankSelectFew, AdaptiveRankSelect, SortedUintVec, SortedUintVecBuilder, SortedUintVecConfig,
    Bmi2Capabilities, Bmi2BitOps, Bmi2BlockOps, Bmi2SequenceOps,
    DataProfile, SelectionCriteria, AccessPattern, OptimizationStrategy,
};
use std::time::Instant;

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
            generator: |i| i % 200 == 0,
            expected_density: 0.005,
            description: "0.5% density - sparse data like outlier flags",
        },
        BenchmarkPattern {
            name: "sparse",
            size: 100_000,
            generator: |i| i % 50 == 0,
            expected_density: 0.02,
            description: "2% density - moderately sparse data",
        },
        BenchmarkPattern {
            name: "low_density",
            size: 100_000,
            generator: |i| i % 20 == 0,
            expected_density: 0.05,
            description: "5% density - threshold of sparse optimization",
        },
        BenchmarkPattern {
            name: "medium_density",
            size: 100_000,
            generator: |i| i % 4 == 0,
            expected_density: 0.25,
            description: "25% density - medium density data",
        },
        BenchmarkPattern {
            name: "clustered_sparse",
            size: 100_000,
            generator: |i| (i / 1000) % 2 == 0 && i % 100 == 0,
            expected_density: 0.005,
            description: "Clustered sparse pattern with spatial locality",
        },
        BenchmarkPattern {
            name: "random_sparse",
            size: 100_000,
            generator: |i| (i.wrapping_mul(31).wrapping_add(17)) % 200 == 0,
            expected_density: 0.005,
            description: "Random sparse pattern without locality",
        },
        BenchmarkPattern {
            name: "alternating_blocks",
            size: 100_000,
            generator: |i| (i / 64) % 16 < 2 && i % 8 == 0,
            expected_density: 0.025,
            description: "Block-alternating pattern for cache testing",
        },
    ]
}

/// Comprehensive rank operation benchmarks
fn benchmark_sparse_rank_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Rank Operations");
    group.throughput(Throughput::Elements(100));
    
    for pattern in get_test_patterns() {
        let bv = pattern.generate_bitvector();
        let test_positions: Vec<usize> = (0..pattern.size).step_by(pattern.size / 100).collect();
        
        // Traditional implementations for comparison
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        
        // Enhanced sparse implementations
        let sparse_ones = RankSelectFew::<true, 64>::from_bit_vector(bv.clone()).unwrap();
        let sparse_zeros = RankSelectFew::<false, 64>::from_bit_vector(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        // Benchmark simple implementation (baseline)
        group.bench_with_input(
            BenchmarkId::new("simple", &pattern.name),
            &(&simple, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark separated implementation
        group.bench_with_input(
            BenchmarkId::new("separated256", &pattern.name),
            &(&separated, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark sparse implementation (storing 1s)
        group.bench_with_input(
            BenchmarkId::new("sparse_ones", &pattern.name),
            &(&sparse_ones, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark sparse implementation (storing 0s)
        group.bench_with_input(
            BenchmarkId::new("sparse_zeros", &pattern.name),
            &(&sparse_zeros, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark adaptive implementation
        group.bench_with_input(
            BenchmarkId::new("adaptive", &pattern.name),
            &(&adaptive, &test_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Comprehensive select operation benchmarks
fn benchmark_sparse_select_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Select Operations");
    group.throughput(Throughput::Elements(20));
    
    for pattern in get_test_patterns() {
        let bv = pattern.generate_bitvector();
        
        // Traditional implementations
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        
        // Enhanced sparse implementations
        let sparse_ones = RankSelectFew::<true, 64>::from_bit_vector(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        let ones_count = simple.count_ones();
        if ones_count == 0 {
            continue;
        }
        
        let test_ks: Vec<usize> = (0..ones_count).step_by((ones_count / 20).max(1)).collect();
        
        // Benchmark simple select
        group.bench_with_input(
            BenchmarkId::new("simple", &pattern.name),
            &(&simple, &test_ks),
            |b, (rs, ks)| {
                b.iter(|| {
                    let mut positions = Vec::new();
                    for &k in black_box(*ks) {
                        if let Ok(pos) = rs.select1(black_box(k)) {
                            positions.push(pos);
                        }
                    }
                    black_box(positions)
                })
            },
        );
        
        // Benchmark separated select
        group.bench_with_input(
            BenchmarkId::new("separated256", &pattern.name),
            &(&separated, &test_ks),
            |b, (rs, ks)| {
                b.iter(|| {
                    let mut positions = Vec::new();
                    for &k in black_box(*ks) {
                        if let Ok(pos) = rs.select1(black_box(k)) {
                            positions.push(pos);
                        }
                    }
                    black_box(positions)
                })
            },
        );
        
        // Benchmark sparse select (optimized for sparse data)
        group.bench_with_input(
            BenchmarkId::new("sparse_ones", &pattern.name),
            &(&sparse_ones, &test_ks),
            |b, (rs, ks)| {
                b.iter(|| {
                    let mut positions = Vec::new();
                    for &k in black_box(*ks) {
                        if let Ok(pos) = rs.select1(black_box(k)) {
                            positions.push(pos);
                        }
                    }
                    black_box(positions)
                })
            },
        );
        
        // Benchmark adaptive select
        group.bench_with_input(
            BenchmarkId::new("adaptive", &pattern.name),
            &(&adaptive, &test_ks),
            |b, (rs, ks)| {
                b.iter(|| {
                    let mut positions = Vec::new();
                    for &k in black_box(*ks) {
                        if let Ok(pos) = rs.select1(black_box(k)) {
                            positions.push(pos);
                        }
                    }
                    black_box(positions)
                })
            },
        );
    }
    
    group.finish();
}

/// Construction time benchmarks for different implementations
fn benchmark_sparse_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse Construction Performance");
    
    for pattern in get_test_patterns() {
        let bv = pattern.generate_bitvector();
        
        // Benchmark simple construction
        group.bench_with_input(
            BenchmarkId::new("simple", &pattern.name),
            &bv,
            |b, bv| b.iter(|| RankSelectSimple::new(black_box(bv.clone())).unwrap()),
        );
        
        // Benchmark separated construction
        group.bench_with_input(
            BenchmarkId::new("separated256", &pattern.name),
            &bv,
            |b, bv| b.iter(|| RankSelectSeparated256::new(black_box(bv.clone())).unwrap()),
        );
        
        // Benchmark sparse construction (ones)
        group.bench_with_input(
            BenchmarkId::new("sparse_ones", &pattern.name),
            &bv,
            |b, bv| b.iter(|| RankSelectFew::<true, 64>::from_bit_vector(black_box(bv.clone())).unwrap()),
        );
        
        // Benchmark sparse construction (zeros)
        group.bench_with_input(
            BenchmarkId::new("sparse_zeros", &pattern.name),
            &bv,
            |b, bv| b.iter(|| RankSelectFew::<false, 64>::from_bit_vector(black_box(bv.clone())).unwrap()),
        );
        
        // Benchmark adaptive construction (includes analysis overhead)
        group.bench_with_input(
            BenchmarkId::new("adaptive", &pattern.name),
            &bv,
            |b, bv| b.iter(|| AdaptiveRankSelect::new(black_box(bv.clone())).unwrap()),
        );
    }
    
    group.finish();
}

/// Memory efficiency benchmarks
fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Efficiency");
    
    for pattern in get_test_patterns() {
        let bv = pattern.generate_bitvector();
        
        // Create all variants for memory comparison
        let simple = RankSelectSimple::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        let sparse_ones = RankSelectFew::<true, 64>::from_bit_vector(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        // Benchmark memory usage and compression ratios
        group.bench_function(
            BenchmarkId::new("memory_analysis", &pattern.name),
            |b| b.iter(|| {
                let original_bits = bv.len();
                let original_bytes = (original_bits + 7) / 8;
                
                let simple_overhead = simple.space_overhead_percent();
                let separated_overhead = separated.space_overhead_percent();
                let sparse_compression = sparse_ones.compression_ratio();
                let adaptive_overhead = adaptive.space_overhead_percent();
                
                // Return metrics for analysis
                black_box((
                    original_bytes,
                    simple_overhead,
                    separated_overhead, 
                    sparse_compression,
                    adaptive_overhead,
                ))
            })
        );
    }
    
    group.finish();
}

/// BMI2 hardware acceleration benchmarks
fn benchmark_bmi2_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("BMI2 Hardware Acceleration");
    
    let caps = Bmi2Capabilities::get();
    println!("BMI2 Capabilities: tier={}, BMI1={}, BMI2={}, POPCNT={}, AVX2={}", 
             caps.optimization_tier, caps.has_bmi1, caps.has_bmi2, caps.has_popcnt, caps.has_avx2);
    
    let test_words = vec![
        0x0000000000000000u64,
        0xFFFFFFFFFFFFFFFFu64,
        0xAAAAAAAAAAAAAAAAu64,
        0x5555555555555555u64,
        0x1248102448102448u64,
        0x8421084210842108u64,
        0x123456789ABCDEFu64,
        0xFEDCBA9876543210u64,
    ];
    
    // Benchmark individual BMI2 operations
    group.bench_function("bmi2_select1_ultra_fast", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for &word in &test_words {
                for rank in 1..=word.count_ones() as usize {
                    if let Some(pos) = Bmi2BitOps::select1_ultra_fast(black_box(word), black_box(rank)) {
                        results.push(pos);
                    }
                }
            }
            black_box(results)
        })
    });
    
    group.bench_function("bmi2_rank1_optimized", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for &word in &test_words {
                for pos in [0, 16, 32, 48, 63] {
                    let rank = Bmi2BitOps::rank1_optimized(black_box(word), black_box(pos));
                    results.push(rank);
                }
            }
            black_box(results)
        })
    });
    
    group.bench_function("bmi2_extract_bits_pext", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for &word in &test_words {
                let masks = [0xFF00FF00FF00FF00u64, 0xF0F0F0F0F0F0F0F0u64, 0xAAAAAAAAAAAAAAAAu64];
                for &mask in &masks {
                    let extracted = Bmi2BitOps::extract_bits_pext(black_box(word), black_box(mask));
                    results.push(extracted);
                }
            }
            black_box(results)
        })
    });
    
    // Benchmark bulk operations with different chunk sizes
    let words = vec![0xAAAAAAAAAAAAAAAAu64; 1000];
    let positions = (0..1000).step_by(10).collect::<Vec<_>>();
    let ranks = (1..=500).step_by(25).collect::<Vec<_>>();
    
    group.bench_function("bmi2_bulk_rank1", |b| {
        b.iter(|| {
            let results = Bmi2BlockOps::bulk_rank1(black_box(&words), black_box(&positions));
            black_box(results)
        })
    });
    
    group.bench_function("bmi2_bulk_select1", |b| {
        b.iter(|| {
            if let Ok(results) = Bmi2BlockOps::bulk_select1(black_box(&words), black_box(&ranks)) {
                black_box(results)
            }
        })
    });
    
    group.finish();
}

/// SortedUintVec compression benchmarks
fn benchmark_sorted_uint_vec_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("SortedUintVec Compression");
    
    // Different compression scenarios
    let scenarios = vec![
        ("small_deltas", 0..10000u64, "Small deltas from base value"),
        ("medium_deltas", (0..10000).map(|i| i * 100), "Medium-sized deltas"),
        ("large_deltas", (0..10000).map(|i| i * 10000), "Large deltas"),
        ("clustered", (0..10000).map(|i| if i < 5000 { i } else { i + 1000000 }), "Clustered values"),
    ];
    
    for (name, values, _desc) in scenarios {
        let values: Vec<u64> = values.collect();
        
        // Test different configurations
        let configs = vec![
            ("default", SortedUintVecConfig::default()),
            ("performance", SortedUintVecConfig::performance_optimized()),
            ("memory", SortedUintVecConfig::memory_optimized()),
        ];
        
        for (config_name, config) in configs {
            // Benchmark construction
            group.bench_with_input(
                BenchmarkId::new(format!("construct_{}_{}", name, config_name), values.len()),
                &(&values, &config),
                |b, (values, config)| {
                    b.iter(|| {
                        let mut builder = SortedUintVecBuilder::with_config(*config);
                        for &value in black_box(*values) {
                            builder.push(black_box(value)).unwrap();
                        }
                        black_box(builder.finish().unwrap())
                    })
                },
            );
            
            // Benchmark random access
            let mut builder = SortedUintVecBuilder::with_config(config);
            for &value in &values {
                builder.push(value).unwrap();
            }
            let vec = builder.finish().unwrap();
            let access_indices: Vec<usize> = (0..values.len()).step_by(values.len() / 100).collect();
            
            group.bench_with_input(
                BenchmarkId::new(format!("access_{}_{}", name, config_name), values.len()),
                &(&vec, &access_indices),
                |b, (vec, indices)| {
                    b.iter(|| {
                        let mut sum = 0;
                        for &idx in black_box(*indices) {
                            sum += vec.get(black_box(idx)).unwrap();
                        }
                        black_box(sum)
                    })
                },
            );
            
            // Benchmark block access
            if vec.num_blocks() > 0 {
                group.bench_with_input(
                    BenchmarkId::new(format!("block_{}_{}", name, config_name), values.len()),
                    &vec,
                    |b, vec| {
                        b.iter(|| {
                            let mut block_data = vec![0u64; vec.config().block_size()];
                            for block_idx in 0..vec.num_blocks().min(10) {
                                vec.get_block(black_box(block_idx), &mut block_data).unwrap();
                                black_box(&block_data);
                            }
                        })
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Adaptive strategy selection benchmarks
fn benchmark_adaptive_strategy_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive Strategy Selection");
    
    for pattern in get_test_patterns() {
        let bv = pattern.generate_bitvector();
        
        // Benchmark data analysis phase
        group.bench_with_input(
            BenchmarkId::new("data_analysis", &pattern.name),
            &bv,
            |b, bv| {
                let criteria = SelectionCriteria::default();
                b.iter(|| {
                    let profile = AdaptiveRankSelect::analyze_data(black_box(bv), black_box(&criteria));
                    black_box(profile)
                })
            },
        );
        
        // Benchmark pattern analysis
        let criteria = SelectionCriteria::default();
        let profile = AdaptiveRankSelect::analyze_data(&bv, &criteria);
        
        group.bench_with_input(
            BenchmarkId::new("pattern_analysis", &pattern.name),
            &bv,
            |b, bv| {
                b.iter(|| {
                    let run_stats = AdaptiveRankSelect::analyze_run_lengths(black_box(bv));
                    let complexity = AdaptiveRankSelect::calculate_pattern_complexity(black_box(bv), &run_stats);
                    let clustering = AdaptiveRankSelect::calculate_clustering_coefficient(black_box(bv));
                    let entropy = AdaptiveRankSelect::calculate_entropy(black_box(bv));
                    black_box((run_stats, complexity, clustering, entropy))
                })
            },
        );
        
        // Benchmark threshold tuning
        group.bench_with_input(
            BenchmarkId::new("threshold_tuning", &pattern.name),
            &(&profile, &criteria),
            |b, (profile, criteria)| {
                b.iter(|| {
                    let thresholds = AdaptiveRankSelect::tune_thresholds(black_box(profile), black_box(criteria));
                    black_box(thresholds)
                })
            },
        );
        
        // Benchmark full adaptive construction
        group.bench_with_input(
            BenchmarkId::new("full_adaptive_construction", &pattern.name),
            &bv,
            |b, bv| {
                b.iter(|| {
                    let adaptive = AdaptiveRankSelect::new(black_box(bv.clone())).unwrap();
                    black_box(adaptive)
                })
            },
        );
    }
    
    group.finish();
}

/// Pattern-specific performance analysis
fn benchmark_pattern_specific_optimizations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pattern-Specific Optimizations");
    
    // Create patterns that benefit from specific optimizations
    let special_patterns = vec![
        ("sequential_locality", 100_000, |i: usize| (i / 64) % 16 == 0, "Sequential access pattern"),
        ("random_access", 100_000, |i: usize| (i.wrapping_mul(31337) + 17) % 100 == 0, "Random access pattern"),  
        ("clustered_small", 100_000, |i: usize| i < 1000 || (i >= 50000 && i < 51000), "Small clusters"),
        ("clustered_large", 100_000, |i: usize| (i / 10000) % 2 == 0 && i % 100 == 0, "Large clusters"),
        ("alternating_dense", 100_000, |i: usize| (i / 1000) % 2 == 0, "Alternating dense/sparse"),
        ("power_of_two", 100_000, |i: usize| i > 0 && (i & (i - 1)) == 0, "Power-of-two positions"),
    ];
    
    for (name, size, generator, _desc) in special_patterns {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(generator(i)).unwrap();
        }
        
        // Test different access patterns
        let sequential_positions: Vec<usize> = (0..size).step_by(size / 100).collect();
        let random_positions: Vec<usize> = (0..size).step_by(size / 100)
            .map(|i| (i.wrapping_mul(31337) + 17) % size)
            .collect();
        
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        let separated = RankSelectSeparated256::new(bv.clone()).unwrap();
        
        // Benchmark sequential access
        group.bench_with_input(
            BenchmarkId::new(format!("adaptive_sequential_{}", name), size),
            &(&adaptive, &sequential_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new(format!("separated_sequential_{}", name), size),
            &(&separated, &sequential_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        // Benchmark random access
        group.bench_with_input(
            BenchmarkId::new(format!("adaptive_random_{}", name), size),
            &(&adaptive, &random_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new(format!("separated_random_{}", name), size),
            &(&separated, &random_positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0;
                    for &pos in black_box(*positions) {
                        sum += rs.rank1(black_box(pos));
                    }
                    black_box(sum)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark sequence analysis for optimization hints
fn benchmark_sequence_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sequence Analysis");
    
    // Generate different types of sequence data for SortedUintVec
    let sequences = vec![
        ("uniform_small", (0..10000u64).collect::<Vec<_>>()),
        ("uniform_large", (0..10000).map(|i| i * 1000).collect::<Vec<_>>()),
        ("exponential", (0..10000).map(|i| 1u64 << (i % 60)).collect::<Vec<_>>()),
        ("fibonacci_mod", {
            let mut fib = vec![1u64, 1];
            for i in 2..10000 {
                let next = (fib[i-1] + fib[i-2]) % 1000000;
                fib.push(next);
            }
            fib.sort();
            fib
        }),
        ("clustered_gaps", {
            let mut values = Vec::new();
            let mut current = 0u64;
            for cluster in 0..100 {
                let cluster_size = 100;
                for i in 0..cluster_size {
                    values.push(current + i);
                }
                current += cluster_size * 10; // Gap between clusters
            }
            values
        }),
    ];
    
    for (name, values) in sequences {
        let words: Vec<u64> = values.chunks(64)
            .map(|chunk| {
                let mut word = 0u64;
                for (i, &val) in chunk.iter().enumerate() {
                    if (val % 2) == 1 {
                        word |= 1u64 << i;
                    }
                }
                word
            })
            .collect();
        
        // Benchmark BMI2 sequence analysis
        group.bench_with_input(
            BenchmarkId::new("bmi2_sequence_analysis", name),
            &words,
            |b, words| {
                b.iter(|| {
                    let analysis = Bmi2SequenceOps::analyze_bit_patterns(black_box(words));
                    black_box(analysis)
                })
            },
        );
        
        // Benchmark SortedUintVec sequence analysis
        let mut builder = SortedUintVecBuilder::new();
        for &value in &values {
            builder.push(value).unwrap();
        }
        let sorted_vec = builder.finish().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("sorted_uint_vec_analysis", name),
            &sorted_vec,
            |b, vec| {
                b.iter(|| {
                    let stats = vec.compression_stats();
                    let analysis = vec.analyze_sequence_patterns();
                    black_box((stats, analysis))
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
    benchmark_bmi2_acceleration,
    benchmark_sorted_uint_vec_compression,
    benchmark_adaptive_strategy_selection,
    benchmark_pattern_specific_optimizations,
    benchmark_sequence_analysis
);

criterion_main!(benches);