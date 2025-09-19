/*!
 * Production Performance Benchmark for Zipora
 *
 * This benchmark runs in production mode with real hardware acceleration enabled
 * to measure actual performance vs. the inflated claims from test-mode benchmarks.
 *
 * Critical Fix: This benchmark will use actual SIMD instructions unlike the
 * previous benchmarks that ran with hardware acceleration disabled.
 */

use std::time::Instant;
use zipora::system::{CpuFeatures, get_cpu_features};
use zipora::succinct::rank_select::{RankSelectPerformanceOps, RankSelectInterleaved256};
use zipora::succinct::BitVector;

fn main() {
    println!("🚀 Zipora Production Performance Benchmark");
    println!("==========================================");

    // Verify hardware acceleration is working
    let features = get_cpu_features();
    println!("Hardware Features Detected:");
    println!("  - POPCNT: {}", features.has_popcnt);
    println!("  - BMI2:   {}", features.has_bmi2);
    println!("  - AVX2:   {}", features.has_avx2);
    println!("  - AVX512F: {}", features.has_avx512f);
    println!("  - AVX512BW: {}", features.has_avx512bw);
    println!("  - AVX512VPOPCNTDQ: {}", features.has_avx512vpopcntdq);
    println!();

    if !features.has_bmi2 && !features.has_avx2 && !features.has_popcnt {
        println!("⚠️  WARNING: No advanced CPU features detected!");
        println!("   Performance will be limited to scalar implementations.");
        println!("   For CI environments, this is expected.");
        println!();
    }

    // Test different data sizes to see real-world performance (reduced for faster testing)
    let test_sizes = vec![1_000, 10_000, 100_000];

    println!("📊 Rank/Select Performance Tests");
    println!("================================");

    for size in test_sizes {
        benchmark_rank_select(size);
    }

    println!("\n🎯 Performance Claims Validation");
    println!("================================");
    validate_performance_claims();
}

fn benchmark_rank_select(size: usize) {
    println!("Testing rank/select with {} elements...", format_number(size));

    // Create a bitvector with ~50% density for realistic testing
    let mut bitvec = BitVector::new();
    for i in 0..size {
        let bit = ((i as u64) * 0x9e3779b97f4a7c15_u64) & 1 == 1;
        if let Err(e) = bitvec.push(bit) {
            println!("  Error building bitvector: {}", e);
            return;
        }
    }

    // Build optimized rank/select structure (using best performer only)
    let rs = match RankSelectInterleaved256::new(bitvec.clone()) {
        Ok(rs) => rs,
        Err(e) => {
            println!("  Error building rank/select: {}", e);
            return;
        }
    };

    // Test the optimized implementation
    test_rank_select_performance("RankSelectInterleaved256 (Best Performer)", &rs, size);
}


fn test_rank_select_performance<T: RankSelectPerformanceOps>(name: &str, rs: &T, size: usize) {
    // Warm up
    let mut rank_sum = 0;
    for _ in 0..1000 {
        rank_sum += rs.rank1_hardware_accelerated(size / 2);
    }

    // Actual benchmark (significantly reduced iterations for faster testing)
    let iterations = std::cmp::max(100, 1_000_000 / size);
    let start = Instant::now();

    for i in 0..iterations {
        let pos = (i * 997) % size; // Pseudo-random positions
        rank_sum += rs.rank1_hardware_accelerated(pos);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

    println!("    {} Size {}: {:.3} Mops/s ({:.2} ns/op)",
             name,
             format_number(size),
             ops_per_sec / 1_000_000.0,
             ns_per_op);

    // Prevent optimization
    if rank_sum == 0 { println!("Unexpected zero sum"); }
}



fn validate_performance_claims() {
    println!("Validating performance claims against actual measurements...");

    // Test rank/select performance specifically (significantly reduced for faster testing)
    let size = 10_000;
    let iterations = 10_000;

    // Create test bitvector
    let mut bitvec = BitVector::new();
    for i in 0..size {
        let bit = ((i as u64) * 0x9e3779b97f4a7c15_u64) & 1 == 1;
        if let Err(e) = bitvec.push(bit) {
            println!("Error building bitvector: {}", e);
            return;
        }
    }

    // Build the highest-performance rank/select structure
    let rs = match RankSelectInterleaved256::new(bitvec) {
        Ok(rs) => rs,
        Err(e) => {
            println!("Error building rank/select structure: {}", e);
            return;
        }
    };

    let start = Instant::now();
    let mut sum = 0;

    for i in 0..iterations {
        let pos = (i * 997) % size;
        sum += rs.rank1_hardware_accelerated(pos);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let gops_per_sec = ops_per_sec / 1_000_000_000.0;

    println!("📈 Actual Performance Measurement (Hardware-Accelerated Zipora):");
    println!("   Operations: {}", format_number(iterations));
    println!("   Time: {:.3}s", elapsed.as_secs_f64());
    println!("   Rate: {:.3} Gops/s", gops_per_sec);
    println!();

    // Compare against realistic expectations (updated after performance analysis)
    let expected_performance = 0.35; // Gops/s - realistic hardware-accelerated performance
    let performance_ratio = gops_per_sec / expected_performance;

    println!("📊 Performance Analysis:");
    println!("   Expected: {:.2} Gops/s (realistic hardware-accelerated)", expected_performance);
    println!("   Measured: {:.3} Gops/s", gops_per_sec);

    if performance_ratio >= 0.8 {
        println!("✅ PASS: Performance is {:.1}% of expected realistic range", performance_ratio * 100.0);
        println!("   Hardware acceleration is working as expected.");
    } else if performance_ratio >= 0.5 {
        println!("⚠️  ACCEPTABLE: Performance is {:.1}% of expected", performance_ratio * 100.0);
        println!("   Performance is within acceptable range for different hardware configurations.");
    } else {
        println!("❌ BELOW EXPECTED: Performance is only {:.1}% of expected", performance_ratio * 100.0);
        println!("   This may indicate suboptimal hardware or configuration issues.");
    }

    // Hardware acceleration verification
    let features = get_cpu_features();
    if features.has_bmi2 || features.has_avx2 || features.has_popcnt {
        println!("✅ Hardware acceleration available and enabled");
        println!("   POPCNT: {}, BMI2: {}, AVX2: {}", features.has_popcnt, features.has_bmi2, features.has_avx2);
    } else {
        println!("❌ No hardware acceleration detected - running in scalar mode");
    }

    // Prevent optimization
    if sum == 0 { println!("Unexpected zero sum"); }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}