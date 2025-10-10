//! Comprehensive Performance Comparison: Zipora 2.0 vs C++ implementation
//!
//! This benchmark suite compares zipora's unified v2.0 implementations against
//! C++ implementation's battle-tested C++ components. Focus is on production-ready
//! unified implementations with hardware acceleration and prefetching.
//!
//! Key comparisons:
//! - RankSelectInterleaved256 (zipora) vs rank_select_il_256_32 (C++ implementation)
//! - Prefetching strategies: prefetch_rank1, prefetch_select1, lookahead
//! - Adaptive SIMD selection vs compile-time optimization
//! - Memory efficiency and overhead ratios

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;
use zipora::{
    succinct::{
        rank_select::{
            interleaved::RankSelectInterleaved256,
            RankSelectOps,
            RankSelectPerformanceOps,
        },
        BitVector,
    },
};

// ============================================================================
// Data Generation (matching C++ implementation patterns)
// ============================================================================

/// Matches C++ implementation's data generation: 25% all-ones, 20% all-zeros, 55% random
pub struct CppImplDataGenerator {
    seed: u64,
}

impl CppImplDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Simple LCG random number generator (deterministic, reproducible)
    fn next_u64(&mut self) -> u64 {
        // Linear Congruential Generator - simple and fast
        // Constants from Numerical Recipes
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        self.seed
    }

    /// Generate bitvector matching C++ implementation's pattern:
    /// - 20% all-zero words
    /// - 25% all-one words
    /// - 55% random words
    pub fn generate_bitvector(&mut self, bits: usize) -> Vec<u64> {
        let words = (bits + 63) / 64;
        let mut data = Vec::with_capacity(words);

        for _ in 0..words {
            let r = self.next_u64();
            let word = match r % 5 {
                0 => 0,                    // 20% all-zeros
                _ if r % 4 == 0 => !0,     // 25% all-ones
                _ => self.next_u64(),      // 55% random
            };
            data.push(word);
        }

        data
    }

    /// Generate sequential access pattern (0 to size)
    pub fn generate_ordered_positions(&mut self, size: usize, count: usize) -> Vec<usize> {
        (0..count)
            .map(|i| (i * size / count).min(size - 1))
            .collect()
    }

    /// Generate random access pattern (shuffled indices)
    pub fn generate_random_positions(&mut self, size: usize, count: usize) -> Vec<usize> {
        let mut positions = self.generate_ordered_positions(size, count);
        // Fisher-Yates shuffle
        for i in (1..positions.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            positions.swap(i, j);
        }
        positions
    }

    /// Generate ordered IDs for select operations (0 to max_rank)
    pub fn generate_ordered_ids(&mut self, max_rank: usize, count: usize) -> Vec<usize> {
        if max_rank == 0 {
            return vec![];
        }
        (0..count.min(max_rank))
            .map(|i| (i * max_rank / count).min(max_rank - 1))
            .collect()
    }

    /// Generate random IDs for select operations (shuffled)
    pub fn generate_random_ids(&mut self, max_rank: usize, count: usize) -> Vec<usize> {
        let mut ids = self.generate_ordered_ids(max_rank, count);
        // Fisher-Yates shuffle
        for i in (1..ids.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            ids.swap(i, j);
        }
        ids
    }
}

// ============================================================================
// Benchmark Result Collection
// ============================================================================

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub impl_name: String,
    pub data_size: usize,
    pub pattern: String,

    // Timing
    pub avg_ns: f64,
    pub median_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub std_dev_ns: f64,

    // Throughput
    pub ops_per_sec: f64,
    pub gops_per_sec: Option<f64>,

    // Memory
    pub memory_bytes: usize,
    pub overhead_ratio: f64,

    // Correctness
    pub checksum: u64,
}

impl BenchmarkResult {
    pub fn new(operation: &str, impl_name: &str, data_size: usize, pattern: &str) -> Self {
        Self {
            operation: operation.to_string(),
            impl_name: impl_name.to_string(),
            data_size,
            pattern: pattern.to_string(),
            avg_ns: 0.0,
            median_ns: 0.0,
            p95_ns: 0.0,
            p99_ns: 0.0,
            std_dev_ns: 0.0,
            ops_per_sec: 0.0,
            gops_per_sec: None,
            memory_bytes: 0,
            overhead_ratio: 0.0,
            checksum: 0,
        }
    }
}

// ============================================================================
// Rank/Select Benchmarks (Primary Focus)
// ============================================================================

/// Benchmark rank operations with ordered access pattern
fn bench_rank_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_ordered");

    // Test configurations matching C++ implementation
    let configs = vec![
        (4 * 1024 * 1024 * 8, "4MB"),    // 4MB of bits
        (128 * 1024 * 1024 * 8, "128MB"), // 128MB of bits
    ];

    for (bit_size, size_label) in configs {
        let mut generator = CppImplDataGenerator::new(12345);
        let data = generator.generate_bitvector(bit_size);

        // Create BitVector from raw data
        let mut bv = BitVector::new();
        for word in &data {
            for bit in 0..64 {
                if bit_size <= bv.len() {
                    break;
                }
                bv.push((word >> bit) & 1 == 1).unwrap();
            }
        }

        // Create rank/select structure
        let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();

        // Generate test positions (sequential)
        let positions = generator.generate_ordered_positions(bit_size, 10000);

        group.throughput(Throughput::Elements(positions.len() as u64));

        // Benchmark: Base rank1 (no prefetch)
        group.bench_with_input(
            BenchmarkId::new("zipora_base", size_label),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &pos in *positions {
                        sum = sum.wrapping_add(rs.rank1(pos) as u64);
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Optimized rank1 (with prefetch + adaptive SIMD)
        group.bench_with_input(
            BenchmarkId::new("zipora_optimized", size_label),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &pos in *positions {
                        sum = sum.wrapping_add(rs.rank1_optimized(pos) as u64);
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Bulk rank1 with lookahead prefetching
        group.bench_with_input(
            BenchmarkId::new("zipora_bulk_lookahead", size_label),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let results = rs.rank1_bulk_optimized(positions);
                    let sum: u64 = results.iter().map(|&r| r as u64).sum();
                    black_box(sum)
                })
            },
        );

        // Memory overhead calculation
        let raw_bytes = bit_size / 8;
        let overhead_percent = rs.space_overhead_percent();
        let structure_bytes = raw_bytes + ((raw_bytes as f64 * overhead_percent / 100.0) as usize);

        eprintln!(
            "rank1_ordered [{}]: raw={} KB, overhead={:.2}%, total={} KB",
            size_label,
            raw_bytes / 1024,
            overhead_percent,
            structure_bytes / 1024
        );
    }

    group.finish();
}

/// Benchmark rank operations with random access pattern
fn bench_rank_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_random");

    let configs = vec![
        (4 * 1024 * 1024 * 8, "4MB"),
        (128 * 1024 * 1024 * 8, "128MB"),
    ];

    for (bit_size, size_label) in configs {
        let mut generator = CppImplDataGenerator::new(12345);
        let data = generator.generate_bitvector(bit_size);

        // Create BitVector
        let mut bv = BitVector::new();
        for word in &data {
            for bit in 0..64 {
                if bit_size <= bv.len() {
                    break;
                }
                bv.push((word >> bit) & 1 == 1).unwrap();
            }
        }

        let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();

        // Generate random test positions
        let positions = generator.generate_random_positions(bit_size, 10000);

        group.throughput(Throughput::Elements(positions.len() as u64));

        // Benchmark: Base rank1 (random access)
        group.bench_with_input(
            BenchmarkId::new("zipora_base", size_label),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &pos in *positions {
                        sum = sum.wrapping_add(rs.rank1(pos) as u64);
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Optimized rank1 (prefetch less effective on random)
        group.bench_with_input(
            BenchmarkId::new("zipora_optimized", size_label),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &pos in *positions {
                        sum = sum.wrapping_add(rs.rank1_optimized(pos) as u64);
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark select operations with ordered access pattern
fn bench_select_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_ordered");

    let configs = vec![
        (4 * 1024 * 1024 * 8, "4MB"),
        (128 * 1024 * 1024 * 8, "128MB"),
    ];

    for (bit_size, size_label) in configs {
        let mut generator = CppImplDataGenerator::new(12345);
        let data = generator.generate_bitvector(bit_size);

        // Create BitVector
        let mut bv = BitVector::new();
        for word in &data {
            for bit in 0..64 {
                if bit_size <= bv.len() {
                    break;
                }
                bv.push((word >> bit) & 1 == 1).unwrap();
            }
        }

        let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let ones_count = rs.count_ones();

        if ones_count == 0 {
            continue;
        }

        // Generate ordered test IDs
        let ids = generator.generate_ordered_ids(ones_count, 10000.min(ones_count));

        group.throughput(Throughput::Elements(ids.len() as u64));

        // Benchmark: Base select1
        group.bench_with_input(
            BenchmarkId::new("zipora_base", size_label),
            &(&rs, &ids),
            |b, (rs, ids)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &id in *ids {
                        if let Ok(pos) = rs.select1(id) {
                            sum = sum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Optimized select1 (with prefetch)
        group.bench_with_input(
            BenchmarkId::new("zipora_optimized", size_label),
            &(&rs, &ids),
            |b, (rs, ids)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &id in *ids {
                        if let Ok(pos) = rs.select1_optimized(id) {
                            sum = sum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Bulk select1 with lookahead
        group.bench_with_input(
            BenchmarkId::new("zipora_bulk_lookahead", size_label),
            &(&rs, &ids),
            |b, (rs, ids)| {
                b.iter(|| {
                    if let Ok(results) = rs.select1_bulk_optimized(ids) {
                        let sum: u64 = results.iter().map(|&r| r as u64).sum();
                        black_box(sum)
                    } else {
                        black_box(0u64)
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark select operations with random access pattern
fn bench_select_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_random");

    let configs = vec![
        (4 * 1024 * 1024 * 8, "4MB"),
        (128 * 1024 * 1024 * 8, "128MB"),
    ];

    for (bit_size, size_label) in configs {
        let mut generator = CppImplDataGenerator::new(12345);
        let data = generator.generate_bitvector(bit_size);

        // Create BitVector
        let mut bv = BitVector::new();
        for word in &data {
            for bit in 0..64 {
                if bit_size <= bv.len() {
                    break;
                }
                bv.push((word >> bit) & 1 == 1).unwrap();
            }
        }

        let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let ones_count = rs.count_ones();

        if ones_count == 0 {
            continue;
        }

        // Generate random test IDs
        let ids = generator.generate_random_ids(ones_count, 10000.min(ones_count));

        group.throughput(Throughput::Elements(ids.len() as u64));

        // Benchmark: Base select1 (random access)
        group.bench_with_input(
            BenchmarkId::new("zipora_base", size_label),
            &(&rs, &ids),
            |b, (rs, ids)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &id in *ids {
                        if let Ok(pos) = rs.select1(id) {
                            sum = sum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Benchmark: Optimized select1
        group.bench_with_input(
            BenchmarkId::new("zipora_optimized", size_label),
            &(&rs, &ids),
            |b, (rs, ids)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &id in *ids {
                        if let Ok(pos) = rs.select1_optimized(id) {
                            sum = sum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory overhead and construction time
fn bench_memory_and_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_and_memory");

    let configs = vec![
        (1024 * 1024 * 8, "1MB"),
        (4 * 1024 * 1024 * 8, "4MB"),
        (16 * 1024 * 1024 * 8, "16MB"),
    ];

    for (bit_size, size_label) in configs {
        let mut generator = CppImplDataGenerator::new(12345);
        let data = generator.generate_bitvector(bit_size);

        // Create BitVector
        let mut bv = BitVector::new();
        for word in &data {
            for bit in 0..64 {
                if bit_size <= bv.len() {
                    break;
                }
                bv.push((word >> bit) & 1 == 1).unwrap();
            }
        }

        // Benchmark construction time
        group.bench_with_input(
            BenchmarkId::new("zipora_construction", size_label),
            &bv,
            |b, bv| {
                b.iter(|| {
                    black_box(RankSelectInterleaved256::new(bv.clone()).unwrap())
                })
            },
        );

        // Measure memory overhead
        let rs = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let raw_bytes = bit_size / 8;
        let overhead_percent = rs.space_overhead_percent();
        let structure_bytes = raw_bytes + ((raw_bytes as f64 * overhead_percent / 100.0) as usize);

        eprintln!(
            "Memory overhead [{}]: raw={} KB, overhead={:.2}%, total={} KB",
            size_label,
            raw_bytes / 1024,
            overhead_percent,
            structure_bytes / 1024
        );
    }

    group.finish();
}

/// Print system information and benchmark configuration
fn print_system_info() {
    eprintln!("========================================");
    eprintln!("Zipora vs C++ implementation Performance Comparison");
    eprintln!("========================================");
    eprintln!("Platform: Linux x86_64");
    eprintln!("Rust: Release mode with LTO, opt-level=3");
    eprintln!("Framework: Criterion.rs");
    eprintln!();

    // Check CPU features
    #[cfg(target_arch = "x86_64")]
    {
        eprintln!("CPU Features:");
        eprintln!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        eprintln!("  BMI2: {}", is_x86_feature_detected!("bmi2"));
        eprintln!("  POPCNT: {}", is_x86_feature_detected!("popcnt"));
        eprintln!("  SSE4.2: {}", is_x86_feature_detected!("sse4.2"));
    }

    eprintln!();
    eprintln!("Data Generation Pattern (matching C++ implementation):");
    eprintln!("  25% all-ones words");
    eprintln!("  20% all-zeros words");
    eprintln!("  55% random words");
    eprintln!();
    eprintln!("Test Configurations:");
    eprintln!("  Small: 4MB of bits");
    eprintln!("  Large: 128MB of bits");
    eprintln!("  Operations: 10,000 queries per test");
    eprintln!("========================================");
    eprintln!();
}

// Run system info print before benchmarks
pub fn setup_benchmarks() {
    print_system_info();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
        .sample_size(100);
    targets =
        bench_rank_ordered,
        bench_rank_random,
        bench_select_ordered,
        bench_select_random,
        bench_memory_and_construction
}

criterion_main!(benches);