//! SIMD Rank-Select Benchmarks
//!
//! Comprehensive benchmarks comparing different optimization levels:
//! - Lookup table optimizations
//! - Hardware-accelerated POPCNT instructions
//! - Hardware-accelerated BMI2 PDEP/PEXT instructions
//! - Adaptive implementations
//! - SIMD bulk operations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use zipora::{BitVector, RankSelect256, BitwiseOp, CpuFeatures};

fn create_benchmark_data(size: usize, density: f64) -> BitVector {
    let mut bv = BitVector::new();
    let threshold = (density * 1000.0) as usize;
    
    for i in 0..size {
        bv.push((i * 31) % 1000 < threshold).unwrap();
    }
    
    bv
}

fn bench_rank_operations(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000];
    let densities = [0.1, 0.5, 0.9]; // 10%, 50%, 90% density
    
    for &size in &sizes {
        for &density in &densities {
            let test_data = create_benchmark_data(size, density);
            let rs = RankSelect256::new(test_data).unwrap();
            let test_positions: Vec<usize> = (0..size).step_by(size / 100).collect();
            
            let group_name = format!("rank_size_{}_density_{}", size, (density * 100.0) as usize);
            let mut group = c.benchmark_group(&group_name);
            
            // Benchmark lookup table implementation
            group.bench_function("lookup_tables", |b| {
                b.iter(|| {
                    for &pos in &test_positions {
                        black_box(rs.rank1_optimized(black_box(pos)));
                    }
                });
            });
            
            // Benchmark hardware-accelerated implementation
            group.bench_function("hardware_accelerated", |b| {
                b.iter(|| {
                    for &pos in &test_positions {
                        black_box(rs.rank1_hardware_accelerated(black_box(pos)));
                    }
                });
            });
            
            // Benchmark adaptive implementation
            group.bench_function("adaptive", |b| {
                b.iter(|| {
                    for &pos in &test_positions {
                        black_box(rs.rank1_adaptive(black_box(pos)));
                    }
                });
            });
            
            group.finish();
        }
    }
}

fn bench_select_operations(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000];
    let densities = [0.1, 0.5, 0.9];
    
    for &size in &sizes {
        for &density in &densities {
            let test_data = create_benchmark_data(size, density);
            let rs = RankSelect256::new(test_data).unwrap();
            let ones_count = rs.count_ones();
            
            if ones_count == 0 {
                continue;
            }
            
            let test_ks: Vec<usize> = (0..ones_count).step_by(ones_count.max(1) / 50).collect();
            
            let group_name = format!("select_size_{}_density_{}", size, (density * 100.0) as usize);
            let mut group = c.benchmark_group(&group_name);
            
            // Benchmark lookup table implementation
            group.bench_function("lookup_tables", |b| {
                b.iter(|| {
                    for &k in &test_ks {
                        black_box(rs.select1_optimized(black_box(k)).unwrap());
                    }
                });
            });
            
            // Benchmark hardware-accelerated implementation
            group.bench_function("hardware_accelerated", |b| {
                b.iter(|| {
                    for &k in &test_ks {
                        black_box(rs.select1_hardware_accelerated(black_box(k)).unwrap());
                    }
                });
            });
            
            // Benchmark adaptive implementation
            group.bench_function("adaptive", |b| {
                b.iter(|| {
                    for &k in &test_ks {
                        black_box(rs.select1_adaptive(black_box(k)).unwrap());
                    }
                });
            });
            
            group.finish();
        }
    }
}

fn bench_simd_bulk_operations(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 50_000];
    
    for &size in &sizes {
        let test_data = create_benchmark_data(size, 0.3); // 30% density
        
        // Benchmark bulk rank operations
        let positions: Vec<usize> = (0..size).step_by(size / 100).collect();
        
        let mut group = c.benchmark_group(&format!("bulk_rank_size_{}", size));
        
        group.bench_function("simd_bulk", |b| {
            b.iter(|| {
                black_box(test_data.rank1_bulk_simd(black_box(&positions)));
            });
        });
        
        group.bench_function("individual", |b| {
            b.iter(|| {
                let results: Vec<usize> = positions.iter()
                    .map(|&pos| test_data.rank1(black_box(pos)))
                    .collect();
                black_box(results);
            });
        });
        
        group.finish();
        
        // Benchmark range setting operations
        let mut group = c.benchmark_group(&format!("range_set_size_{}", size));
        let range_size = size / 10;
        
        group.bench_function("simd_range", |b| {
            b.iter(|| {
                let mut bv = test_data.clone();
                bv.set_range_simd(black_box(0), black_box(range_size), black_box(true)).unwrap();
                black_box(bv);
            });
        });
        
        group.bench_function("individual", |b| {
            b.iter(|| {
                let mut bv = test_data.clone();
                for i in 0..range_size {
                    bv.set(black_box(i), black_box(true)).unwrap();
                }
                black_box(bv);
            });
        });
        
        group.finish();
        
        // Benchmark bulk bitwise operations
        let other_data = create_benchmark_data(size, 0.4); // Different density
        let mut group = c.benchmark_group(&format!("bulk_bitwise_size_{}", size));
        let op_range = size / 2;
        
        group.bench_function("simd_bitwise", |b| {
            b.iter(|| {
                let mut bv = test_data.clone();
                bv.bulk_bitwise_op_simd(black_box(&other_data), black_box(BitwiseOp::And), black_box(0), black_box(op_range)).unwrap();
                black_box(bv);
            });
        });
        
        group.bench_function("individual", |b| {
            b.iter(|| {
                let mut bv = test_data.clone();
                for i in 0..op_range {
                    let self_bit = bv.get(black_box(i)).unwrap_or(false);
                    let other_bit = other_data.get(black_box(i)).unwrap_or(false);
                    bv.set(black_box(i), black_box(self_bit & other_bit)).unwrap();
                }
                black_box(bv);
            });
        });
        
        group.finish();
    }
}

fn bench_hardware_instructions(c: &mut Criterion) {
    // Benchmark the core hardware instruction wrappers using bit vectors
    let test_values = [
        0x0000000000000000u64,
        0xFFFFFFFFFFFFFFFFu64,
        0xAAAAAAAAAAAAAAAAu64,
        0x5555555555555555u64,
        0x123456789ABCDEFu64,
        0x8000000000000001u64,
        0xF0F0F0F0F0F0F0F0u64,
    ];
    
    let mut group = c.benchmark_group("hardware_instructions");
    
    // Create bit vectors from test values to use public API
    let mut bit_vectors = Vec::new();
    for &val in &test_values {
        let mut bv = BitVector::new();
        for i in 0..64 {
            bv.push((val >> i) & 1 == 1).unwrap();
        }
        bit_vectors.push(bv);
    }
    
    // Benchmark popcount through rank operations
    group.bench_function("popcount_via_rank", |b| {
        b.iter(|| {
            for bv in &bit_vectors {
                black_box(bv.rank1(64));
            }
        });
    });
    
    // Benchmark select operations through RankSelect256
    for (idx, &val) in test_values.iter().enumerate() {
        let popcount = val.count_ones() as usize;
        if popcount > 0 {
            let rs = RankSelect256::new(bit_vectors[idx].clone()).unwrap();
            let test_ks: Vec<usize> = (0..popcount).step_by(popcount.max(1) / 8 + 1).collect();
            
            group.bench_with_input(
                BenchmarkId::new("select_optimized", format!("0x{:016x}", val)),
                &(&rs, &test_ks),
                |b, &(rs, ks)| {
                    b.iter(|| {
                        for &k in ks {
                            black_box(rs.select1(black_box(k)).unwrap());
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_features");
    
    group.bench_function("feature_detection", |b| {
        b.iter(|| {
            black_box(CpuFeatures::detect());
        });
    });
    
    group.bench_function("cached_features", |b| {
        b.iter(|| {
            black_box(CpuFeatures::get());
        });
    });
    
    group.finish();
}

// Custom benchmark configuration for detailed analysis
criterion_group!(
    name = simd_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10))
        .warm_up_time(std::time::Duration::from_secs(3));
    targets = 
        bench_rank_operations,
        bench_select_operations,
        bench_simd_bulk_operations,
        bench_hardware_instructions,
        bench_cpu_feature_detection
);

criterion_main!(simd_benches);