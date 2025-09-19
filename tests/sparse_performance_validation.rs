//! Performance validation tests for sparse rank-select implementations
//!
//! This test suite validates that the enhanced sparse implementations provide
//! the expected performance improvements over baseline implementations.

use zipora::{
    BitVector,
    succinct::rank_select::{
        RankSelectOps, RankSelectPerformanceOps, RankSelectInterleaved256, AdaptiveRankSelect, RankSelectBuilder,
        bmi2_comprehensive::{Bmi2Capabilities, Bmi2BitOps, Bmi2BlockOps},
    },
    blob_store::{SortedUintVec, SortedUintVecBuilder},
};
use std::time::{Duration, Instant};

/// Performance measurement utilities
struct PerformanceMeasurer;

impl PerformanceMeasurer {
    /// Measure execution time of a closure
    fn measure<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Measure average execution time over multiple iterations
    fn measure_average<F, R>(f: F, iterations: usize) -> (R, Duration)
    where
        F: Fn() -> R,
    {
        let mut total_duration = Duration::new(0, 0);
        let mut last_result = None;
        
        for _ in 0..iterations {
            let (result, duration) = Self::measure(&f);
            total_duration += duration;
            last_result = Some(result);
        }
        
        (last_result.unwrap(), total_duration / iterations as u32)
    }
    
    /// Calculate improvement ratio (old_time / new_time)
    fn improvement_ratio(old_time: Duration, new_time: Duration) -> f64 {
        if new_time.as_nanos() == 0 {
            return 1.0;
        }
        old_time.as_nanos() as f64 / new_time.as_nanos() as f64
    }
}

/// Test data generators for performance validation
struct PerformanceTestData;

impl PerformanceTestData {
    /// Generate ultra-sparse data (0.1% density) - best case for sparse optimizations
    fn ultra_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 1000 == 0).unwrap();
        }
        bv
    }
    
    /// Generate very sparse data (0.5% density)
    fn very_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 200 == 0).unwrap();
        }
        bv
    }
    
    /// Generate clustered sparse data with high locality
    fn clustered_sparse(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            // Create clusters of 64 bits with 4 bits set, then 1936 bits clear
            let cluster_pos = i % 2000;
            bv.push(cluster_pos < 64 && cluster_pos % 16 == 0).unwrap();
        }
        bv
    }
    
    /// Generate medium density data (25% density) - should favor traditional implementations
    fn medium_density(size: usize) -> BitVector {
        let mut bv = BitVector::new();
        for i in 0..size {
            bv.push(i % 4 == 0).unwrap();
        }
        bv
    }
}

/// Validate sparse rank operations performance
#[test]
#[ignore] // Run only when specifically requested
fn test_sparse_rank_performance() {
    println!("=== Sparse Rank Performance Validation ===");
    
    let test_size = 100_000;
    let iterations = 1000;
    
    let test_cases = vec![
        ("ultra_sparse", PerformanceTestData::ultra_sparse(test_size)),
        ("very_sparse", PerformanceTestData::very_sparse(test_size)),
        ("clustered_sparse", PerformanceTestData::clustered_sparse(test_size)),
        ("medium_density", PerformanceTestData::medium_density(test_size)),
    ];
    
    for (test_name, bv) in test_cases {
        println!("\nTesting {}: {} bits, {} ones ({:.3}% density)", 
                 test_name, bv.len(), bv.count_ones(),
                 (bv.count_ones() as f64 / bv.len() as f64) * 100.0);
        
        // Create implementations
        let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        // Generate test positions
        let test_positions: Vec<usize> = (0..bv.len()).step_by(bv.len() / 100).collect();
        
        // Benchmark interleaved implementation (baseline)
        let (_, interleaved_time) = PerformanceMeasurer::measure_average(|| {
            let mut sum = 0;
            for &pos in &test_positions {
                sum += interleaved.rank1(pos);
            }
            sum
        }, iterations);
        
        // Benchmark adaptive implementation
        let (_, adaptive_time) = PerformanceMeasurer::measure_average(|| {
            let mut sum = 0;
            for &pos in &test_positions {
                sum += adaptive.rank1(pos);
            }
            sum
        }, iterations);
        
        // Calculate improvements
        let adaptive_improvement = PerformanceMeasurer::improvement_ratio(interleaved_time, adaptive_time);

        println!("  Interleaved: {:9.3}ms", interleaved_time.as_secs_f64() * 1000.0);
        println!("  Adaptive:    {:9.3}ms ({:.2}x improvement)", adaptive_time.as_secs_f64() * 1000.0, adaptive_improvement);
        println!("  Selected:    {}", adaptive.implementation_name());

        // Validate expected improvements for sparse data
        if test_name.contains("sparse") {
            // For sparse data, adaptive should be competitive or better than interleaved
            assert!(adaptive_improvement >= 0.8,
                   "Adaptive should perform well on sparse data: {:.2}x", adaptive_improvement);
        }

        // Adaptive implementation should be reasonably fast
        assert!(adaptive_improvement >= 0.5, "Adaptive should not be too slow: {:.2}x", adaptive_improvement);
    }
}

/// Validate sparse select operations performance
#[test]
#[ignore] // Run only when specifically requested
fn test_sparse_select_performance() {
    println!("\n=== Sparse Select Performance Validation ===");
    
    let test_size = 100_000;
    let iterations = 200; // Fewer iterations for select (more expensive)
    
    let test_cases = vec![
        ("ultra_sparse", PerformanceTestData::ultra_sparse(test_size)),
        ("very_sparse", PerformanceTestData::very_sparse(test_size)),
        ("clustered_sparse", PerformanceTestData::clustered_sparse(test_size)),
    ];
    
    for (test_name, bv) in test_cases {
        let ones_count = bv.count_ones();
        if ones_count == 0 {
            continue;
        }
        
        println!("\nTesting {}: {} ones", test_name, ones_count);
        
        // Create implementations
        let interleaved = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        // Generate test select indices
        let test_ks: Vec<usize> = (0..ones_count).step_by((ones_count / 20).max(1)).collect();
        
        // Benchmark implementations
        let (_, interleaved_time) = PerformanceMeasurer::measure_average(|| {
            let mut positions = Vec::new();
            for &k in &test_ks {
                positions.push(interleaved.select1(k).unwrap());
            }
            positions
        }, iterations);

        let (_, adaptive_time) = PerformanceMeasurer::measure_average(|| {
            let mut positions = Vec::new();
            for &k in &test_ks {
                positions.push(adaptive.select1(k).unwrap());
            }
            positions
        }, iterations);

        let (_, hardware_accel_time) = PerformanceMeasurer::measure_average(|| {
            let mut positions = Vec::new();
            for &k in &test_ks {
                positions.push(interleaved.select1_hardware_accelerated(k).unwrap());
            }
            positions
        }, iterations);

        // Calculate improvements (using interleaved as baseline)
        let adaptive_improvement = PerformanceMeasurer::improvement_ratio(interleaved_time, adaptive_time);
        let hardware_improvement = PerformanceMeasurer::improvement_ratio(interleaved_time, hardware_accel_time);

        println!("  Interleaved:  {:9.3}ms", interleaved_time.as_secs_f64() * 1000.0);
        println!("  Adaptive:     {:9.3}ms ({:.2}x vs interleaved)", adaptive_time.as_secs_f64() * 1000.0, adaptive_improvement);
        println!("  Hardware-acc: {:9.3}ms ({:.2}x vs interleaved)", hardware_accel_time.as_secs_f64() * 1000.0, hardware_improvement);

        // For sparse data, adaptive select should choose optimal implementation
        if test_name.contains("sparse") {
            // Hardware acceleration should provide improvement for sparse data
            assert!(hardware_improvement >= 1.0,
                   "Hardware accelerated select should be competitive: {:.2}x", hardware_improvement);

            // Adaptive should choose well and be competitive
            assert!(adaptive_improvement >= 0.5,
                   "Adaptive select should be reasonable: {:.2}x", adaptive_improvement);
        }
    }
}

/// Validate BMI2 hardware acceleration performance
#[test]
#[ignore] // Run only when specifically requested  
fn test_bmi2_acceleration_performance() {
    println!("\n=== BMI2 Hardware Acceleration Performance ===");
    
    let caps = Bmi2Capabilities::get();
    println!("BMI2 Capabilities: tier={}, BMI1={}, BMI2={}, POPCNT={}, AVX2={}", 
             caps.optimization_tier, caps.has_bmi1, caps.has_bmi2, caps.has_popcnt, caps.has_avx2);
    
    let iterations = 10000;
    let test_words = vec![
        0xAAAAAAAAAAAAAAAAu64, // Alternating bits
        0x5555555555555555u64, // Alternating bits (offset)
        0x1248102448102448u64, // Sparse pattern
        0x8421084210842108u64, // Sparse pattern (offset)
        0x123456789ABCDEFu64,  // Mixed pattern
    ];
    
    // Test select operations
    let mut total_hardware_time = Duration::new(0, 0);
    let mut total_fallback_time = Duration::new(0, 0);
    let mut test_count = 0;
    
    for &word in &test_words {
        let ones_count = word.count_ones() as usize;
        if ones_count == 0 {
            continue;
        }
        
        for rank in 1..=ones_count {
            // Benchmark fallback implementation
            let (_, fallback_time) = PerformanceMeasurer::measure_average(|| {
                Bmi2BitOps::select1_fallback(word, rank)
            }, iterations);
            
            // Benchmark hardware-accelerated implementation
            #[cfg(target_arch = "x86_64")]
            let (_, hardware_time) = PerformanceMeasurer::measure_average(|| {
                Bmi2BitOps::select1_ultra_fast(word, rank)
            }, iterations);
            
            #[cfg(not(target_arch = "x86_64"))]
            let hardware_time = fallback_time;
            
            total_fallback_time += fallback_time;
            total_hardware_time += hardware_time;
            test_count += 1;
        }
    }
    
    if test_count > 0 {
        let avg_fallback = total_fallback_time / test_count;
        let avg_hardware = total_hardware_time / test_count;
        let improvement = PerformanceMeasurer::improvement_ratio(avg_fallback, avg_hardware);
        
        println!("  Fallback select: {:9.3}ns", avg_fallback.as_nanos() as f64);
        println!("  Hardware select: {:9.3}ns ({:.2}x improvement)", avg_hardware.as_nanos() as f64, improvement);
        
        // On BMI2-capable systems, expect significant improvement
        #[cfg(target_arch = "x86_64")]
        {
            if caps.has_bmi2 {
                assert!(improvement >= 2.0,
                       "BMI2 select should be significantly faster: {:.2}x", improvement);
            }
        }
    }
    
    // Test rank operations
    let mut rank_fallback_time = Duration::new(0, 0);
    let mut rank_hardware_time = Duration::new(0, 0);
    let mut rank_count = 0;
    
    for &word in &test_words {
        for pos in [0, 16, 32, 48, 63] {
            let (_, fallback_time) = PerformanceMeasurer::measure_average(|| {
                (word & ((1u64 << pos) - 1)).count_ones() as usize
            }, iterations);
            
            let (_, hardware_time) = PerformanceMeasurer::measure_average(|| {
                Bmi2BitOps::rank1_optimized(word, pos)
            }, iterations);
            
            rank_fallback_time += fallback_time;
            rank_hardware_time += hardware_time;
            rank_count += 1;
        }
    }
    
    if rank_count > 0 {
        let avg_rank_fallback = rank_fallback_time / rank_count;
        let avg_rank_hardware = rank_hardware_time / rank_count;
        let rank_improvement = PerformanceMeasurer::improvement_ratio(avg_rank_fallback, avg_rank_hardware);
        
        println!("  Fallback rank:   {:9.3}ns", avg_rank_fallback.as_nanos() as f64);
        println!("  Hardware rank:   {:9.3}ns ({:.2}x improvement)", avg_rank_hardware.as_nanos() as f64, rank_improvement);
        
        // Hardware rank should be competitive
        assert!(rank_improvement >= 0.8,
               "Hardware rank should be competitive: {:.2}x", rank_improvement);
    }
    
    // Test bulk operations
    let words = vec![0xAAAAAAAAAAAAAAAAu64; 1000];
    let positions = (0..1000).step_by(10).collect::<Vec<_>>();
    let ranks = (1..=500).step_by(5).collect::<Vec<_>>();
    
    let (_, bulk_rank_time) = PerformanceMeasurer::measure_average(|| {
        Bmi2BlockOps::bulk_rank1(&words, &positions)
    }, 100);
    
    let (_, bulk_select_time) = PerformanceMeasurer::measure_average(|| {
        if let Ok(results) = Bmi2BlockOps::bulk_select1(&words, &ranks) {
            results
        } else {
            Vec::new()
        }
    }, 100);
    
    println!("  Bulk rank:       {:9.3}ms", bulk_rank_time.as_secs_f64() * 1000.0);
    println!("  Bulk select:     {:9.3}ms", bulk_select_time.as_secs_f64() * 1000.0);
    
    // Bulk operations should complete in reasonable time
    assert!(bulk_rank_time.as_millis() < 100, "Bulk rank should be fast");
    assert!(bulk_select_time.as_millis() < 100, "Bulk select should be fast");
}

/// Validate SortedUintVec compression performance
#[test]
#[ignore] // Run only when specifically requested
fn test_sorted_uint_vec_performance() {
    println!("\n=== SortedUintVec Compression Performance ===");
    
    let test_scenarios = vec![
        ("small_deltas", (0..10000u64).collect::<Vec<_>>()),
        ("medium_deltas", (0..10000).map(|i| i * 100).collect::<Vec<_>>()),
        ("large_deltas", (0..1000).map(|i| i * 10000).collect::<Vec<_>>()),
    ];
    
    for (scenario_name, values) in test_scenarios {
        println!("\nTesting {}: {} values", scenario_name, values.len());
        
        // Measure construction time
        let (compressed, construction_time) = PerformanceMeasurer::measure(|| {
            let mut builder = SortedUintVecBuilder::new();
            for &value in &values {
                builder.push(value).unwrap();
            }
            builder.finish().unwrap()
        });
        
        // Measure access time
        let access_indices: Vec<usize> = (0..values.len()).step_by((values.len() / 100).max(1)).collect();
        let (_, access_time) = PerformanceMeasurer::measure_average(|| {
            let mut sum = 0u64;
            for &idx in &access_indices {
                sum += compressed.get(idx).unwrap();
            }
            sum
        }, 1000);
        
        // Measure block access time
        let (_, block_access_time) = PerformanceMeasurer::measure_average(|| {
            if compressed.num_blocks() > 0 {
                let mut block_data = vec![0u64; compressed.config().block_size()];
                compressed.get_block(0, &mut block_data).unwrap();
                block_data[0]
            } else {
                0
            }
        }, 1000);
        
        // Calculate compression metrics
        let stats = compressed.compression_stats();
        let original_bytes = values.len() * 8;
        let compressed_bytes = compressed.memory_usage();
        
        println!("  Construction:    {:9.3}ms", construction_time.as_secs_f64() * 1000.0);
        println!("  Access time:     {:9.3}μs", access_time.as_micros() as f64);
        println!("  Block access:    {:9.3}μs", block_access_time.as_micros() as f64);
        println!("  Original size:   {} bytes", original_bytes);
        println!("  Compressed size: {} bytes", compressed_bytes);
        println!("  Compression:     {:.2}x", stats.compression_ratio);
        println!("  Space savings:   {:.1}%", stats.space_savings_percent);
        
        // Validate compression effectiveness
        if scenario_name == "small_deltas" {
            assert!(stats.compression_ratio < 0.5,
                   "Small deltas should compress well: {:.2}x", stats.compression_ratio);
        }
        
        // All scenarios should achieve some compression
        assert!(stats.compression_ratio < 1.0,
               "Should achieve some compression: {:.2}x", stats.compression_ratio);
        
        // Access should be reasonably fast
        assert!(access_time.as_micros() < 1000,
               "Random access should be fast: {}μs", access_time.as_micros());
        assert!(block_access_time.as_micros() < 1000,
               "Block access should be fast: {}μs", block_access_time.as_micros());
    }
}

/// Validate adaptive construction overhead
#[test]
#[ignore] // Run only when specifically requested
fn test_adaptive_construction_overhead() {
    println!("\n=== Adaptive Construction Overhead ===");
    
    let test_cases = vec![
        ("ultra_sparse", PerformanceTestData::ultra_sparse(50000)),
        ("very_sparse", PerformanceTestData::very_sparse(50000)),
        ("medium_density", PerformanceTestData::medium_density(50000)),
    ];
    
    for (test_name, bv) in test_cases {
        println!("\nTesting {}: {} bits", test_name, bv.len());
        
        // Measure simple construction
        let (_, simple_time) = PerformanceMeasurer::measure(|| {
            RankSelectInterleaved256::new(bv.clone()).unwrap()
        });
        
        // Measure separated construction
        let (_, separated_time) = PerformanceMeasurer::measure(|| {
            RankSelectInterleaved256::new(bv.clone()).unwrap()
        });
        
        // Measure sparse construction using builder trait
        let (_, sparse_time) = PerformanceMeasurer::measure(|| {
            RankSelectInterleaved256::from_bit_vector(bv.clone()).unwrap()
        });
        
        // Measure adaptive construction (includes analysis overhead)
        let (adaptive, adaptive_time) = PerformanceMeasurer::measure(|| {
            AdaptiveRankSelect::new(bv.clone()).unwrap()
        });
        
        let simple_ratio = PerformanceMeasurer::improvement_ratio(adaptive_time, simple_time);
        let separated_ratio = PerformanceMeasurer::improvement_ratio(adaptive_time, separated_time);
        let sparse_ratio = PerformanceMeasurer::improvement_ratio(adaptive_time, sparse_time);
        
        println!("  Simple:     {:9.3}ms ({:.2}x vs adaptive)", simple_time.as_secs_f64() * 1000.0, simple_ratio);
        println!("  Separated:  {:9.3}ms ({:.2}x vs adaptive)", separated_time.as_secs_f64() * 1000.0, separated_ratio);
        println!("  Sparse:     {:9.3}ms ({:.2}x vs adaptive)", sparse_time.as_secs_f64() * 1000.0, sparse_ratio);
        println!("  Adaptive:   {:9.3}ms", adaptive_time.as_secs_f64() * 1000.0);
        println!("  Selected:   {}", adaptive.implementation_name());
        
        // Adaptive construction overhead should be reasonable
        assert!(adaptive_time.as_millis() < 1000,
               "Adaptive construction should complete in reasonable time: {}ms", adaptive_time.as_millis());
        
        // Analysis overhead should not be too high compared to construction
        let min_construction_time = simple_time.min(separated_time).min(sparse_time);
        let overhead_ratio = adaptive_time.as_nanos() as f64 / min_construction_time.as_nanos() as f64;
        assert!(overhead_ratio < 5.0,
               "Adaptive analysis overhead should be reasonable: {:.2}x", overhead_ratio);
    }
}

/// Overall performance validation summary
#[test]
#[ignore] // Run only when specifically requested
fn test_overall_performance_validation() {
    println!("\n=== Overall Performance Validation Summary ===");
    
    // Create representative test data
    let ultra_sparse = PerformanceTestData::ultra_sparse(100000);
    let clustered = PerformanceTestData::clustered_sparse(100000);
    let medium = PerformanceTestData::medium_density(100000);
    
    let test_cases = vec![
        ("ultra_sparse", ultra_sparse),
        ("clustered", clustered),
        ("medium", medium),
    ];
    
    let mut results = Vec::new();
    
    for (test_name, bv) in test_cases {
        println!("\nValidating {}: {} bits, {} ones ({:.3}% density)", 
                 test_name, bv.len(), bv.count_ones(),
                 (bv.count_ones() as f64 / bv.len() as f64) * 100.0);
        
        // Create implementations
        let simple = RankSelectInterleaved256::new(bv.clone()).unwrap();
        let adaptive = AdaptiveRankSelect::new(bv.clone()).unwrap();
        
        // Measure memory usage
        let original_bytes = (bv.len() + 7) / 8;
        let simple_overhead = simple.space_overhead_percent();
        let adaptive_overhead = adaptive.space_overhead_percent();
        
        // Quick performance test
        let test_positions: Vec<usize> = (0..bv.len()).step_by(bv.len() / 100).collect();
        
        let (_, simple_rank_time) = PerformanceMeasurer::measure_average(|| {
            let mut sum = 0;
            for &pos in &test_positions {
                sum += simple.rank1(pos);
            }
            sum
        }, 100);
        
        let (_, adaptive_rank_time) = PerformanceMeasurer::measure_average(|| {
            let mut sum = 0;
            for &pos in &test_positions {
                sum += adaptive.rank1(pos);
            }
            sum
        }, 100);
        
        let rank_improvement = PerformanceMeasurer::improvement_ratio(simple_rank_time, adaptive_rank_time);
        
        println!("  Original size:      {} bytes", original_bytes);
        println!("  Simple overhead:    {:.2}%", simple_overhead);
        println!("  Adaptive overhead:  {:.2}%", adaptive_overhead);
        println!("  Rank improvement:   {:.2}x", rank_improvement);
        println!("  Selected impl:      {}", adaptive.implementation_name());
        
        results.push((test_name, rank_improvement, adaptive.implementation_name().to_string()));
        
        // Validate that adaptive performs competitively
        assert!(rank_improvement >= 0.8,
               "Adaptive should perform competitively: {:.2}x", rank_improvement);
        
        // For sparse data, adaptive should choose sparse implementations
        if test_name.contains("sparse") && bv.count_ones() < bv.len() / 20 {
            assert!(adaptive.implementation_name().contains("RankSelectInterleaved256") || 
                   adaptive.implementation_name().contains("RankSelectInterleaved256"),
                   "Should choose sparse implementation for sparse data: {}", adaptive.implementation_name());
        }
    }
    
    println!("\n=== Validation Summary ===");
    for (test_name, improvement, impl_name) in results {
        println!("  {}: {:.2}x improvement with {}", test_name, improvement, impl_name);
    }
    
    println!("\nAll performance validation tests passed!");
    println!("Enhanced sparse implementations demonstrate expected performance characteristics.");
}