//! Comprehensive rank-select hardware acceleration benchmarks

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use infini_zip::{BitVector, RankSelect256, CpuFeatures};

fn create_test_bitvector(size: usize, density: f64) -> BitVector {
    let mut bv = BitVector::new();
    let threshold = (density * 1000.0) as usize;
    for i in 0..size {
        bv.push((i * 31) % 1000 < threshold).unwrap();
    }
    bv
}

fn benchmark_rank_operations(c: &mut Criterion) {
    // Detect CPU features
    let features = CpuFeatures::get();
    println!("CPU Features - POPCNT: {}, BMI2: {}, AVX2: {}", 
             features.has_popcnt, features.has_bmi2, features.has_avx2);
    
    let mut group = c.benchmark_group("Rank Operations");
    
    let sizes = vec![1_000, 10_000, 100_000];
    let densities = vec![0.1, 0.5, 0.9];
    
    for &size in &sizes {
        for &density in &densities {
            let bv = create_test_bitvector(size, density);
            let rs = RankSelect256::new(bv).unwrap();
            
            let test_positions: Vec<usize> = (0..size).step_by((size / 100).max(1)).collect();
            
            // Benchmark optimized rank
            group.bench_with_input(
                BenchmarkId::new(format!("optimized_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_positions),
                |b, (rs, positions)| {
                    b.iter(|| {
                        let mut sum = 0;
                        for &pos in *positions {
                            sum += rs.rank1_optimized(pos);
                        }
                        sum
                    })
                },
            );
            
            // Benchmark hardware-accelerated rank
            group.bench_with_input(
                BenchmarkId::new(format!("hardware_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_positions),
                |b, (rs, positions)| {
                    b.iter(|| {
                        let mut sum = 0;
                        for &pos in *positions {
                            sum += rs.rank1_hardware_accelerated(pos);
                        }
                        sum
                    })
                },
            );
            
            // Benchmark adaptive rank
            group.bench_with_input(
                BenchmarkId::new(format!("adaptive_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_positions),
                |b, (rs, positions)| {
                    b.iter(|| {
                        let mut sum = 0;
                        for &pos in *positions {
                            sum += rs.rank1_adaptive(pos);
                        }
                        sum
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_select_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Select Operations");
    
    let sizes = vec![1_000, 10_000, 100_000];
    let densities = vec![0.1, 0.5, 0.9];
    
    for &size in &sizes {
        for &density in &densities {
            let bv = create_test_bitvector(size, density);
            let rs = RankSelect256::new(bv).unwrap();
            
            let ones_count = rs.count_ones();
            if ones_count == 0 { continue; }
            
            let test_ks: Vec<usize> = (0..ones_count).step_by((ones_count / 20).max(1)).collect();
            
            // Benchmark optimized select
            group.bench_with_input(
                BenchmarkId::new(format!("optimized_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_ks),
                |b, (rs, ks)| {
                    b.iter(|| {
                        let mut positions = Vec::new();
                        for &k in *ks {
                            positions.push(rs.select1_optimized(k).unwrap());
                        }
                        positions
                    })
                },
            );
            
            // Benchmark hardware-accelerated select
            group.bench_with_input(
                BenchmarkId::new(format!("hardware_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_ks),
                |b, (rs, ks)| {
                    b.iter(|| {
                        let mut positions = Vec::new();
                        for &k in *ks {
                            positions.push(rs.select1_hardware_accelerated(k).unwrap());
                        }
                        positions
                    })
                },
            );
            
            // Benchmark adaptive select
            group.bench_with_input(
                BenchmarkId::new(format!("adaptive_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &(&rs, &test_ks),
                |b, (rs, ks)| {
                    b.iter(|| {
                        let mut positions = Vec::new();
                        for &k in *ks {
                            positions.push(rs.select1_adaptive(k).unwrap());
                        }
                        positions
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Construction Performance");
    
    let sizes = vec![1_000, 10_000, 100_000];
    let densities = vec![0.1, 0.5, 0.9];
    
    for &size in &sizes {
        for &density in &densities {
            let bv = create_test_bitvector(size, density);
            
            group.bench_with_input(
                BenchmarkId::new(format!("construction_size_{}_density_{}", size, (density * 100.0) as u32), size),
                &bv,
                |b, bv| {
                    b.iter(|| {
                        RankSelect256::new(bv.clone()).unwrap()
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_bit_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit-level Operations");
    
    let test_values = [
        0x0000000000000000u64,
        0xFFFFFFFFFFFFFFFFu64,
        0xAAAAAAAAAAAAAAAAu64,
        0x5555555555555555u64,
        0x123456789ABCDEFu64,
    ];
    
    group.bench_function("bitvector_rank1_64", |b| {
        b.iter(|| {
            let mut total_ones = 0u64;
            for &val in &test_values {
                let mut bv = BitVector::new();
                for i in 0..64 {
                    bv.push((val >> i) & 1 == 1).unwrap();
                }
                total_ones += bv.rank1(64) as u64;
            }
            total_ones
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_rank_operations,
    benchmark_select_operations,
    benchmark_construction,
    benchmark_bit_operations
);
criterion_main!(benches);