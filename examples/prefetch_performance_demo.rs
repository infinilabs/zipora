//! Performance demonstration for Prefetching + Adaptive SIMD Integration
//!
//! Measures and reports the performance improvements from:
//! - Advanced prefetching strategies
//! - Runtime adaptive SIMD selection
//! - Lookahead prefetching in bulk operations

use std::time::Instant;
use zipora::succinct::rank_select::interleaved::RankSelectInterleaved256;
use zipora::succinct::BitVector;
use zipora::succinct::rank_select::RankSelectOps;
use zipora::RankSelectPerformanceOps;

fn create_test_data(size: usize, density: f64) -> RankSelectInterleaved256 {
    let mut bv = BitVector::new();
    let threshold = (density * 1000.0) as usize;
    for i in 0..size {
        bv.push((i * 31) % 1000 < threshold).unwrap();
    }
    RankSelectInterleaved256::new(bv).unwrap()
}

fn bench_rank1_single_operation() {
    println!("\n=== Benchmark: Single rank1 Operation ===");
    let rs = create_test_data(1_000_000, 0.5);
    let iterations = 1_000_000;
    let test_pos = 500_000;

    // Baseline: standard rank1
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(rs.rank1(test_pos));
    }
    let base_duration = start.elapsed();
    let base_ops_per_sec = iterations as f64 / base_duration.as_secs_f64();

    // Optimized: rank1_optimized (prefetch + adaptive SIMD)
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(rs.rank1_optimized(test_pos));
    }
    let opt_duration = start.elapsed();
    let opt_ops_per_sec = iterations as f64 / opt_duration.as_secs_f64();

    // Adaptive only: rank1_adaptive
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(rs.rank1_adaptive(test_pos));
    }
    let adaptive_duration = start.elapsed();
    let adaptive_ops_per_sec = iterations as f64 / adaptive_duration.as_secs_f64();

    println!("Iterations: {}", iterations);
    println!("\nBaseline rank1:");
    println!("  Time: {:?}", base_duration);
    println!("  Throughput: {:.2} M ops/sec", base_ops_per_sec / 1_000_000.0);

    println!("\nrank1_adaptive (adaptive SIMD only):");
    println!("  Time: {:?}", adaptive_duration);
    println!("  Throughput: {:.2} M ops/sec", adaptive_ops_per_sec / 1_000_000.0);
    println!("  Speedup: {:.3}x", adaptive_ops_per_sec / base_ops_per_sec);
    println!("  Overhead: {:.1}%", ((adaptive_duration.as_nanos() as f64 / base_duration.as_nanos() as f64) - 1.0) * 100.0);

    println!("\nrank1_optimized (prefetch + adaptive SIMD):");
    println!("  Time: {:?}", opt_duration);
    println!("  Throughput: {:.2} M ops/sec", opt_ops_per_sec / 1_000_000.0);
    println!("  Speedup: {:.3}x", opt_ops_per_sec / base_ops_per_sec);
    println!("  Improvement: {:.1}%", ((opt_ops_per_sec / base_ops_per_sec) - 1.0) * 100.0);
}

fn bench_bulk_operations() {
    println!("\n=== Benchmark: Bulk Operations with Lookahead Prefetching ===");
    let rs = create_test_data(1_000_000, 0.5);
    let batch_sizes = [100, 1000];

    for &batch_size in &batch_sizes {
        let positions: Vec<usize> = (0..batch_size)
            .map(|i| (i * 1_000_000 / batch_size).min(999_999))
            .collect();

        let iterations = 100;

        // Individual calls
        let start = Instant::now();
        for _ in 0..iterations {
            let mut results = Vec::with_capacity(positions.len());
            for &pos in &positions {
                results.push(rs.rank1(pos));
            }
            std::hint::black_box(results);
        }
        let individual_duration = start.elapsed();
        let individual_ops_per_sec = (iterations * batch_size) as f64 / individual_duration.as_secs_f64();

        // Bulk optimized (with PREFETCH_DISTANCE=8 lookahead)
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(rs.rank1_bulk_optimized(&positions));
        }
        let bulk_duration = start.elapsed();
        let bulk_ops_per_sec = (iterations * batch_size) as f64 / bulk_duration.as_secs_f64();

        println!("\nBatch size: {}", batch_size);
        println!("  Individual calls:");
        println!("    Time: {:?}", individual_duration);
        println!("    Throughput: {:.2} M ops/sec", individual_ops_per_sec / 1_000_000.0);
        println!("  Bulk with lookahead prefetch (PREFETCH_DISTANCE=8):");
        println!("    Time: {:?}", bulk_duration);
        println!("    Throughput: {:.2} M ops/sec", bulk_ops_per_sec / 1_000_000.0);
        println!("    Speedup: {:.3}x", bulk_ops_per_sec / individual_ops_per_sec);
        println!("    Improvement: {:.1}%", ((bulk_ops_per_sec / individual_ops_per_sec) - 1.0) * 100.0);
    }
}

fn bench_select_operations() {
    println!("\n=== Benchmark: Select Operations with Prefetching ===");
    let rs = create_test_data(1_000_000, 0.5);
    let ones_count = rs.count_ones();
    println!("Total ones: {}", ones_count);

    if ones_count < 1000 {
        println!("Skipping select benchmark - not enough ones");
        return;
    }

    let iterations = 100_000;
    let test_id = ones_count / 2;

    // Baseline select1
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(rs.select1(test_id).ok());
    }
    let base_duration = start.elapsed();
    let base_ops_per_sec = iterations as f64 / base_duration.as_secs_f64();

    // Optimized select1_optimized
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(rs.select1_optimized(test_id).ok());
    }
    let opt_duration = start.elapsed();
    let opt_ops_per_sec = iterations as f64 / opt_duration.as_secs_f64();

    println!("\nBaseline select1:");
    println!("  Time: {:?}", base_duration);
    println!("  Throughput: {:.2} M ops/sec", base_ops_per_sec / 1_000_000.0);

    println!("\nselect1_optimized (prefetch + adaptive):");
    println!("  Time: {:?}", opt_duration);
    println!("  Throughput: {:.2} M ops/sec", opt_ops_per_sec / 1_000_000.0);
    println!("  Speedup: {:.3}x", opt_ops_per_sec / base_ops_per_sec);
    println!("  Improvement: {:.1}%", ((opt_ops_per_sec / base_ops_per_sec) - 1.0) * 100.0);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Prefetching + Adaptive SIMD Integration Performance Demonstration  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    println!("\nThis benchmark demonstrates the performance improvements from:");
    println!("  1. Advanced prefetching strategies (prefetch_rank1, prefetch_select1)");
    println!("  2. Lookahead prefetching in bulk operations (PREFETCH_DISTANCE=8)");
    println!("  3. Runtime adaptive SIMD selection");
    println!("  4. Performance monitoring hooks");

    bench_rank1_single_operation();
    bench_bulk_operations();
    bench_select_operations();

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  Summary: Prefetching + Adaptive SIMD Integration Complete ✅        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
}
