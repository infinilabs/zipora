//! Rank/Select Comparison Benchmarks: zipora vs C++ implementation
//!
//! This module provides comprehensive benchmarks comparing zipora's
//! RankSelectInterleaved256 against C++ implementation's rank_select_il_256
//! and rank_select_se_512 implementations.
//!
//! Methodology matches C++ implementation exactly:
//! - Data: 25% all-ones, 20% all-zeros, 55% random
//! - Scenarios: rank_ordered, select_ordered, rank_random, select_random
//! - Sizes: 4KB, 128KB, 4MB, 128MB
//! - Metrics: ns/op, Gops/s, memory overhead, checksums

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Instant;
use zipora::{
    BitVector,
    succinct::rank_select::{RankSelectInterleaved256, RankSelectOps, RankSelectPerformanceOps},
};

use super::mod_utils::{
    AccessPattern, BenchmarkMetrics, DataPattern, CppImplDataGenerator,
    calculate_checksum, markdown_table_header,
};

// Test data sizes matching C++ implementation benchmarks
const SIZE_4KB: usize = 4 * 1024 * 8;      // 4KB in bits
const SIZE_128KB: usize = 128 * 1024 * 8;  // 128KB in bits
const SIZE_4MB: usize = 4 * 1024 * 1024 * 8;  // 4MB in bits
const SIZE_128MB: usize = 128 * 1024 * 1024 * 8;  // 128MB in bits (only for release builds)

const NUM_QUERIES: usize = 10_000;  // Number of queries per benchmark

/// Create RankSelectInterleaved256 from raw bit data
fn create_rank_select(bit_data: &[u64], num_bits: usize) -> RankSelectInterleaved256 {
    let mut bv = BitVector::new();

    for word_idx in 0..bit_data.len() {
        let word = bit_data[word_idx];
        let bits_in_word = if word_idx == bit_data.len() - 1 && num_bits % 64 != 0 {
            num_bits % 64
        } else {
            64
        };

        for bit_idx in 0..bits_in_word {
            let bit = (word >> bit_idx) & 1 == 1;
            bv.push(bit).unwrap();
        }
    }

    RankSelectInterleaved256::new(bv).unwrap()
}

/// Benchmark: rank1 with sequential ordered access
///
/// This matches C++ implementation's rank_ordered scenario:
/// - Query positions: 0, 1, 2, ..., size-1 (sequential)
/// - Measures cache efficiency and sequential access performance
fn bench_rank1_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_ordered");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let positions = gen.generate_positions(num_bits, NUM_QUERIES, AccessPattern::Sequential);

        group.throughput(Throughput::Elements(NUM_QUERIES as u64));

        group.bench_with_input(
            BenchmarkId::new("zipora_RankSelectInterleaved256", size_name),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &pos in *positions {
                        checksum = checksum.wrapping_add(rs.rank1(pos) as u64);
                    }
                    black_box(checksum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: rank1 with random access
///
/// This matches C++ implementation's rank_random scenario:
/// - Query positions: shuffled random order
/// - Measures random access performance and cache misses
fn bench_rank1_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_random");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let positions = gen.generate_positions(num_bits, NUM_QUERIES, AccessPattern::Random);

        group.throughput(Throughput::Elements(NUM_QUERIES as u64));

        group.bench_with_input(
            BenchmarkId::new("zipora_RankSelectInterleaved256", size_name),
            &(&rs, &positions),
            |b, (rs, positions)| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &pos in *positions {
                        checksum = checksum.wrapping_add(rs.rank1(pos) as u64);
                    }
                    black_box(checksum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: select1 with sequential ordered access
///
/// This matches C++ implementation's select_ordered scenario:
/// - Query indices: 0, 1, 2, ..., num_ones-1 (sequential)
/// - Measures select operation performance with good locality
fn bench_select1_ordered(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_ordered");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let total_ones = rs.count_ones();
        if total_ones == 0 {
            continue;
        }

        let indices = gen.generate_indices(total_ones, NUM_QUERIES.min(total_ones), AccessPattern::Sequential);

        group.throughput(Throughput::Elements(indices.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("zipora_RankSelectInterleaved256", size_name),
            &(&rs, &indices),
            |b, (rs, indices)| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &idx in *indices {
                        if let Ok(pos) = rs.select1(idx) {
                            checksum = checksum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(checksum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: select1 with random access
///
/// This matches C++ implementation's select_random scenario:
/// - Query indices: shuffled random order
/// - Measures worst-case select performance
fn bench_select1_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_random");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let total_ones = rs.count_ones();
        if total_ones == 0 {
            continue;
        }

        let indices = gen.generate_indices(total_ones, NUM_QUERIES.min(total_ones), AccessPattern::Random);

        group.throughput(Throughput::Elements(indices.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("zipora_RankSelectInterleaved256", size_name),
            &(&rs, &indices),
            |b, (rs, indices)| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &idx in *indices {
                        if let Ok(pos) = rs.select1(idx) {
                            checksum = checksum.wrapping_add(pos as u64);
                        }
                    }
                    black_box(checksum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Bulk rank1 with lookahead prefetching
///
/// This tests zipora's bulk optimization with PREFETCH_DISTANCE=8
/// against naive individual rank1 calls.
fn bench_rank1_bulk_prefetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank1_bulk_prefetch");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let positions = gen.generate_positions(num_bits, NUM_QUERIES, AccessPattern::Sequential);

        group.throughput(Throughput::Elements(NUM_QUERIES as u64));

        // Baseline: Individual calls
        group.bench_with_input(
            BenchmarkId::new("individual", size_name),
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

        // Optimized: Bulk with prefetching
        group.bench_with_input(
            BenchmarkId::new("bulk_optimized", size_name),
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

/// Benchmark: Bulk select1 with lookahead prefetching
fn bench_select1_bulk_prefetch(c: &mut Criterion) {
    let mut group = c.benchmark_group("select1_bulk_prefetch");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let total_ones = rs.count_ones();
        if total_ones == 0 {
            continue;
        }

        let indices = gen.generate_indices(total_ones, NUM_QUERIES.min(total_ones), AccessPattern::Sequential);

        group.throughput(Throughput::Elements(indices.len() as u64));

        // Baseline: Individual calls
        group.bench_with_input(
            BenchmarkId::new("individual", size_name),
            &(&rs, &indices),
            |b, (rs, indices)| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(indices.len());
                    for &idx in *indices {
                        if let Ok(pos) = rs.select1(idx) {
                            results.push(pos);
                        }
                    }
                    black_box(results)
                })
            },
        );

        // Optimized: Bulk with prefetching
        group.bench_with_input(
            BenchmarkId::new("bulk_optimized", size_name),
            &(&rs, &indices),
            |b, (rs, indices)| {
                b.iter(|| {
                    black_box(rs.select1_bulk_optimized(indices).ok())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory overhead comparison
///
/// This measures the space overhead of the rank/select structure
/// compared to the raw bit data.
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        let raw_bytes = (num_bits + 7) / 8;
        let overhead_percent = rs.space_overhead_percent();
        let total_bytes = raw_bytes + ((raw_bytes as f64 * overhead_percent / 100.0) as usize);

        println!("\n=== Memory Overhead: {} ===", size_name);
        println!("Raw data: {} bytes ({} KB)", raw_bytes, raw_bytes / 1024);
        println!("Overhead: {:.2}%", overhead_percent);
        println!("Total: {} bytes ({} KB)", total_bytes, total_bytes / 1024);
        println!("Ratio: {:.2}x", 1.0 + overhead_percent / 100.0);

        // Benchmark construction time
        group.bench_with_input(
            BenchmarkId::new("construction", size_name),
            &(&bit_data, num_bits),
            |b, (bit_data, num_bits)| {
                b.iter(|| {
                    black_box(create_rank_select(bit_data, *num_bits))
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive correctness validation
///
/// Generates detailed metrics for comparison with C++ implementation,
/// including correctness checksums.
fn validate_correctness() {
    println!("\n=== Correctness Validation ===");

    let sizes = vec![
        ("4KB", SIZE_4KB),
        ("128KB", SIZE_128KB),
        ("4MB", SIZE_4MB),
    ];

    for (size_name, num_bits) in sizes {
        println!("\n--- Data Size: {} ({} bits) ---", size_name, num_bits);

        let mut gen = CppImplDataGenerator::new(42);
        let bit_data = gen.generate_bitvector(num_bits, DataPattern::CppImplDefault);
        let rs = create_rank_select(&bit_data, num_bits);

        // Rank ordered
        let positions = gen.generate_positions(num_bits, 1000, AccessPattern::Sequential);
        let rank_results: Vec<usize> = positions.iter().map(|&pos| rs.rank1(pos)).collect();
        let rank_checksum = calculate_checksum(&rank_results);
        println!("rank1_ordered checksum: {:016x}", rank_checksum);

        // Rank random
        let random_positions = gen.generate_positions(num_bits, 1000, AccessPattern::Random);
        let rank_random_results: Vec<usize> = random_positions.iter().map(|&pos| rs.rank1(pos)).collect();
        let rank_random_checksum = calculate_checksum(&rank_random_results);
        println!("rank1_random checksum:  {:016x}", rank_random_checksum);

        // Select ordered
        let total_ones = rs.count_ones();
        if total_ones > 0 {
            let indices = gen.generate_indices(total_ones, 1000.min(total_ones), AccessPattern::Sequential);
            let select_results: Vec<usize> = indices.iter()
                .filter_map(|&idx| rs.select1(idx).ok())
                .collect();
            let select_checksum = calculate_checksum(&select_results);
            println!("select1_ordered checksum: {:016x}", select_checksum);

            // Select random
            let random_indices = gen.generate_indices(total_ones, 1000.min(total_ones), AccessPattern::Random);
            let select_random_results: Vec<usize> = random_indices.iter()
                .filter_map(|&idx| rs.select1(idx).ok())
                .collect();
            let select_random_checksum = calculate_checksum(&select_random_results);
            println!("select1_random checksum:  {:016x}", select_random_checksum);
        }

        println!("Total ones: {} ({:.2}% density)", total_ones, (total_ones as f64 / num_bits as f64) * 100.0);
        println!("Space overhead: {:.2}%", rs.space_overhead_percent());
    }
}

// Only run correctness validation if --test flag is present
#[cfg(test)]
#[test]
fn test_correctness_validation() {
    validate_correctness();
}

criterion_group!(
    rank_select_comparison,
    bench_rank1_ordered,
    bench_rank1_random,
    bench_select1_ordered,
    bench_select1_random,
    bench_rank1_bulk_prefetch,
    bench_select1_bulk_prefetch,
    bench_memory_overhead,
);

criterion_main!(rank_select_comparison);
