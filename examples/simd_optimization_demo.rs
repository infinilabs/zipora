//! SIMD Optimization Demo
//!
//! This example demonstrates the advanced SIMD optimizations including:
//! - BMI2 POPCNT and BMI2 PDEP/PEXT hardware instructions
//! - Runtime CPU feature detection and adaptive algorithm selection
//! - Multiple optimization tiers based on available hardware capabilities
//! - Performance comparisons between different optimization levels
//!
//! Run with: cargo run --example simd_optimization_demo --features simd

use infini_zip::{BitVector, RankSelect256, CpuFeatures, BitwiseOp};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SIMD Optimization Demo - Advanced Hardware Acceleration");
    println!("================================================================\n");

    // 1. CPU Feature Detection
    demonstrate_cpu_features();

    // 2. Create test data
    let test_data = create_test_data(100_000);
    println!("📊 Created test dataset with {} bits", test_data.len());
    println!("   Set bits: {}", test_data.count_ones());
    println!("   Density: {:.2}%\n", (test_data.count_ones() as f64 / test_data.len() as f64) * 100.0);

    // 3. Build RankSelect256 structure
    let rs = RankSelect256::new(test_data.clone())?;
    println!("🏗️  Built RankSelect256 structure");
    println!("   Space overhead: {:.2}%\n", rs.space_overhead_percent());

    // 4. Performance comparison of rank operations
    demonstrate_rank_performance(&rs);

    // 5. Performance comparison of select operations
    demonstrate_select_performance(&rs);

    // 6. SIMD bulk operations on BitVector
    demonstrate_simd_bulk_operations(test_data);

    println!("✅ SIMD Optimization Demo completed successfully!");
    Ok(())
}

fn demonstrate_cpu_features() {
    println!("🔍 CPU Feature Detection");
    println!("========================");
    
    let features = CpuFeatures::detect();
    println!("Detected CPU features:");
    println!("  POPCNT: {}", if features.has_popcnt { "✅ Available" } else { "❌ Not available" });
    println!("  BMI2:   {}", if features.has_bmi2 { "✅ Available" } else { "❌ Not available" });
    println!("  AVX2:   {}", if features.has_avx2 { "✅ Available" } else { "❌ Not available" });
    
    println!("\nOptimization levels available:");
    if features.has_popcnt {
        println!("  📈 Hardware POPCNT: 2-5x faster rank operations");
    }
    if features.has_bmi2 {
        println!("  🚀 Hardware BMI2: 5-10x faster select operations");
    }
    if features.has_avx2 {
        println!("  ⚡ SIMD AVX2: Vectorized bulk operations");
    }
    if !features.has_popcnt && !features.has_bmi2 && !features.has_avx2 {
        println!("  📚 Lookup tables: 10-100x faster than naive implementation");
    }
    println!();
}

fn create_test_data(size: usize) -> BitVector {
    let mut bv = BitVector::new();
    
    // Create varied density patterns to test different scenarios
    for i in 0..size {
        let bit = match i % 10000 {
            0..=1000 => i % 7 == 0,      // Dense region (14.3% density)
            1001..=5000 => i % 137 == 0, // Sparse region (0.7% density)
            5001..=8000 => i % 3 == 0,   // Medium density (33.3% density)
            _ => i % 1013 == 0,          // Very sparse (0.1% density)
        };
        bv.push(bit).unwrap();
    }
    
    bv
}

fn demonstrate_rank_performance(rs: &RankSelect256) {
    println!("⚡ Rank Operation Performance Comparison");
    println!("========================================");

    // Test positions covering the entire range
    let test_positions: Vec<usize> = (0..rs.len()).step_by(rs.len() / 1000).collect();
    let iterations = 1000;

    println!("Testing {} rank operations across {} iterations...", test_positions.len(), iterations);

    // Baseline: Optimized lookup table implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &pos in &test_positions {
            let _ = rs.rank1_optimized(pos);
        }
    }
    let lookup_time = start.elapsed();

    // Hardware-accelerated implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &pos in &test_positions {
            let _ = rs.rank1_hardware_accelerated(pos);
        }
    }
    let hardware_time = start.elapsed();

    // Adaptive implementation (chooses best available)
    let start = Instant::now();
    for _ in 0..iterations {
        for &pos in &test_positions {
            let _ = rs.rank1_adaptive(pos);
        }
    }
    let adaptive_time = start.elapsed();

    println!("Results:");
    println!("  Lookup tables:      {:8.2} ms", lookup_time.as_secs_f64() * 1000.0);
    println!("  Hardware-accel:     {:8.2} ms", hardware_time.as_secs_f64() * 1000.0);
    println!("  Adaptive (best):    {:8.2} ms", adaptive_time.as_secs_f64() * 1000.0);

    if hardware_time < lookup_time {
        let speedup = lookup_time.as_secs_f64() / hardware_time.as_secs_f64();
        println!("  🚀 Hardware speedup: {:.1}x faster", speedup);
    } else {
        println!("  📚 Lookup tables are optimal on this CPU");
    }
    println!();
}

fn demonstrate_select_performance(rs: &RankSelect256) {
    println!("🎯 Select Operation Performance Comparison");
    println!("===========================================");

    let ones_count = rs.count_ones();
    if ones_count == 0 {
        println!("No set bits in test data, skipping select performance test.\n");
        return;
    }

    // Test select operations across the full range
    let test_ks: Vec<usize> = (0..ones_count).step_by(ones_count.max(1) / 500).collect();
    let iterations = 500;

    println!("Testing {} select operations across {} iterations...", test_ks.len(), iterations);

    // Baseline: Optimized lookup table implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &k in &test_ks {
            let _ = rs.select1_optimized(k);
        }
    }
    let lookup_time = start.elapsed();

    // Hardware-accelerated implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &k in &test_ks {
            let _ = rs.select1_hardware_accelerated(k);
        }
    }
    let hardware_time = start.elapsed();

    // Adaptive implementation
    let start = Instant::now();
    for _ in 0..iterations {
        for &k in &test_ks {
            let _ = rs.select1_adaptive(k);
        }
    }
    let adaptive_time = start.elapsed();

    println!("Results:");
    println!("  Lookup tables:      {:8.2} ms", lookup_time.as_secs_f64() * 1000.0);
    println!("  Hardware-accel:     {:8.2} ms", hardware_time.as_secs_f64() * 1000.0);
    println!("  Adaptive (best):    {:8.2} ms", adaptive_time.as_secs_f64() * 1000.0);

    if hardware_time < lookup_time {
        let speedup = lookup_time.as_secs_f64() / hardware_time.as_secs_f64();
        println!("  🚀 Hardware speedup: {:.1}x faster", speedup);
    } else {
        println!("  📚 Lookup tables are optimal on this CPU");
    }
    println!();
}

fn demonstrate_simd_bulk_operations(mut test_data: BitVector) {
    println!("🔄 SIMD Bulk Operations Demo");
    println!("============================");

    // 1. Bulk rank operations
    let positions: Vec<usize> = (0..test_data.len()).step_by(1000).collect();
    println!("Testing bulk rank operations on {} positions...", positions.len());

    let start = Instant::now();
    let bulk_ranks = test_data.rank1_bulk_simd(&positions);
    let bulk_time = start.elapsed();

    let start = Instant::now();
    let individual_ranks: Vec<usize> = positions.iter().map(|&pos| test_data.rank1(pos)).collect();
    let individual_time = start.elapsed();

    println!("  Bulk SIMD:     {:8.2} ms", bulk_time.as_secs_f64() * 1000.0);
    println!("  Individual:    {:8.2} ms", individual_time.as_secs_f64() * 1000.0);
    
    // Verify results match
    assert_eq!(bulk_ranks, individual_ranks, "Bulk rank results must match individual results");
    println!("  ✅ Results verified");

    if individual_time > bulk_time {
        let speedup = individual_time.as_secs_f64() / bulk_time.as_secs_f64();
        println!("  🚀 Bulk speedup: {:.1}x faster", speedup);
    }

    // 2. Range setting operations
    println!("\nTesting SIMD range setting...");
    let mut bv_copy = test_data.clone();
    
    let start = Instant::now();
    bv_copy.set_range_simd(1000, 5000, true).unwrap();
    let simd_time = start.elapsed();

    let mut bv_individual = test_data.clone();
    let start = Instant::now();
    for i in 1000..5000 {
        bv_individual.set(i, true).unwrap();
    }
    let individual_time = start.elapsed();

    println!("  SIMD range:    {:8.2} ms", simd_time.as_secs_f64() * 1000.0);
    println!("  Individual:    {:8.2} ms", individual_time.as_secs_f64() * 1000.0);

    // Verify results match
    for i in 1000..5000 {
        assert_eq!(bv_copy.get(i), bv_individual.get(i), "Range setting results must match at position {}", i);
    }
    println!("  ✅ Results verified");

    if individual_time > simd_time {
        let speedup = individual_time.as_secs_f64() / simd_time.as_secs_f64();
        println!("  🚀 SIMD speedup: {:.1}x faster", speedup);
    }

    // 3. Bulk bitwise operations
    println!("\nTesting SIMD bulk bitwise operations...");
    let mut other_bv = BitVector::new();
    for i in 0..test_data.len() {
        other_bv.push(i % 5 == 0).unwrap();
    }

    let mut bv_simd = test_data.clone();
    let start = Instant::now();
    bv_simd.bulk_bitwise_op_simd(&other_bv, BitwiseOp::And, 0, 10000).unwrap();
    let simd_time = start.elapsed();

    let mut bv_individual = test_data.clone();
    let start = Instant::now();
    for i in 0..10000 {
        let self_bit = bv_individual.get(i).unwrap_or(false);
        let other_bit = other_bv.get(i).unwrap_or(false);
        bv_individual.set(i, self_bit & other_bit).unwrap();
    }
    let individual_time = start.elapsed();

    println!("  SIMD bitwise:  {:8.2} ms", simd_time.as_secs_f64() * 1000.0);
    println!("  Individual:    {:8.2} ms", individual_time.as_secs_f64() * 1000.0);

    // Verify results match
    for i in 0..10000 {
        assert_eq!(bv_simd.get(i), bv_individual.get(i), "Bitwise operation results must match at position {}", i);
    }
    println!("  ✅ Results verified");

    if individual_time > simd_time {
        let speedup = individual_time.as_secs_f64() / simd_time.as_secs_f64();
        println!("  🚀 SIMD speedup: {:.1}x faster", speedup);
    }
    println!();
}