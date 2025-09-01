//! Comprehensive SIMD Memory Operations Performance Benchmarks
//!
//! This benchmark suite measures the performance of SIMD memory operations
//! against the targets specified in Phase 1.2:
//! - Small copies (≤64 bytes): 2-3x faster than memcpy
//! - Medium copies (64-4096 bytes): 1.5-2x faster with prefetching
//! - Large copies (>4KB): Match or exceed system memcpy
//!
//! The benchmarks test:
//! 1. Memory copy operations across different size ranges
//! 2. Memory comparison with early termination
//! 3. Memory search operations
//! 4. Memory fill operations
//! 5. Comparison against standard library implementations
//! 6. Different SIMD tiers when available

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::alloc::{alloc, dealloc, Layout};
use zipora::memory::simd_ops::{SimdMemOps, SimdTier};

//==============================================================================
// TEST DATA GENERATION
//==============================================================================

/// Generate test data with a reproducible pattern
fn generate_test_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 17 + 13) % 256) as u8).collect()
}

/// Generate aligned buffer for testing aligned operations
fn generate_aligned_buffer(size: usize, alignment: usize) -> (*mut u8, Layout) {
    let layout = Layout::from_size_align(size, alignment)
        .expect("Failed to create layout");
    let ptr = unsafe { alloc(layout) };
    assert!(!ptr.is_null(), "Failed to allocate aligned memory");
    (ptr, layout)
}

//==============================================================================
// MEMORY COPY BENCHMARKS
//==============================================================================

/// Benchmark memory copy operations across different size categories
fn bench_memory_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Memory Copy");
    
    // Test different size categories as per Phase 1.2 targets
    let test_sizes = vec![
        // Small copies (≤64 bytes) - Target: 2-3x faster
        ("tiny_8B", 8),
        ("small_16B", 16),
        ("small_32B", 32),
        ("small_64B", 64),
        
        // Medium copies (64-4096 bytes) - Target: 1.5-2x faster
        ("medium_128B", 128),
        ("medium_256B", 256),
        ("medium_512B", 512),
        ("medium_1KB", 1024),
        ("medium_2KB", 2048),
        ("medium_4KB", 4096),
        
        // Large copies (>4KB) - Target: Match or exceed system memcpy
        ("large_8KB", 8192),
        ("large_16KB", 16384),
        ("large_32KB", 32768),
        ("large_64KB", 65536),
        ("large_128KB", 131072),
        ("large_256KB", 262144),
        ("large_512KB", 524288),
        ("large_1MB", 1048576),
    ];
    
    let simd_ops = SimdMemOps::new();
    
    for (name, size) in test_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark SIMD implementation
        group.bench_function(BenchmarkId::new("SIMD", name), |b| {
            let src = generate_test_data(size);
            let mut dst = vec![0u8; size];
            b.iter(|| {
                simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                    .expect("SIMD copy failed");
            });
        });
        
        // Benchmark standard library copy_from_slice
        group.bench_function(BenchmarkId::new("std_copy_from_slice", name), |b| {
            let src = generate_test_data(size);
            let mut dst = vec![0u8; size];
            b.iter(|| {
                black_box(&mut dst).copy_from_slice(black_box(&src));
            });
        });
        
        // Benchmark unsafe ptr::copy_nonoverlapping
        group.bench_function(BenchmarkId::new("ptr_copy_nonoverlapping", name), |b| {
            let src = generate_test_data(size);
            let mut dst = vec![0u8; size];
            b.iter(|| unsafe {
                std::ptr::copy_nonoverlapping(
                    black_box(src.as_ptr()),
                    black_box(dst.as_mut_ptr()),
                    size,
                );
            });
        });
    }
    
    group.finish();
}

/// Benchmark aligned memory copy operations
fn bench_aligned_memory_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Aligned Memory Copy");
    
    let test_sizes = vec![
        ("aligned_64B", 64),
        ("aligned_128B", 128),
        ("aligned_256B", 256),
        ("aligned_512B", 512),
        ("aligned_1KB", 1024),
        ("aligned_4KB", 4096),
        ("aligned_16KB", 16384),
        ("aligned_64KB", 65536),
    ];
    
    let simd_ops = SimdMemOps::new();
    const CACHE_LINE_SIZE: usize = 64;
    
    for (name, size) in test_sizes {
        // Ensure size is a multiple of cache line size for aligned operations
        let aligned_size = (size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;
        group.throughput(Throughput::Bytes(aligned_size as u64));
        
        // Benchmark aligned SIMD copy
        group.bench_function(BenchmarkId::new("SIMD_aligned", name), |b| {
            let (src_ptr, src_layout) = generate_aligned_buffer(aligned_size, CACHE_LINE_SIZE);
            let (dst_ptr, dst_layout) = generate_aligned_buffer(aligned_size, CACHE_LINE_SIZE);
            
            // Initialize source with test data
            unsafe {
                for i in 0..aligned_size {
                    *src_ptr.add(i) = ((i * 17 + 13) % 256) as u8;
                }
            }
            
            b.iter(|| unsafe {
                let src_slice = std::slice::from_raw_parts(src_ptr, aligned_size);
                let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, aligned_size);
                simd_ops.copy_aligned(black_box(src_slice), black_box(dst_slice))
                    .expect("Aligned copy failed");
            });
            
            unsafe {
                dealloc(src_ptr, src_layout);
                dealloc(dst_ptr, dst_layout);
            }
        });
        
        // Benchmark unaligned SIMD copy for comparison
        group.bench_function(BenchmarkId::new("SIMD_unaligned", name), |b| {
            let src = generate_test_data(aligned_size);
            let mut dst = vec![0u8; aligned_size];
            b.iter(|| {
                simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                    .expect("SIMD copy failed");
            });
        });
    }
    
    group.finish();
}

//==============================================================================
// MEMORY COMPARISON BENCHMARKS
//==============================================================================

/// Benchmark memory comparison operations with early termination
fn bench_memory_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Memory Compare");
    
    let test_sizes = vec![
        ("small_32B", 32),
        ("small_64B", 64),
        ("medium_256B", 256),
        ("medium_1KB", 1024),
        ("medium_4KB", 4096),
        ("large_16KB", 16384),
        ("large_64KB", 65536),
    ];
    
    let simd_ops = SimdMemOps::new();
    
    for (name, size) in test_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Test equal buffers (worst case - no early termination)
        group.bench_function(BenchmarkId::new("SIMD_equal", name), |b| {
            let data = generate_test_data(size);
            let data_copy = data.clone();
            b.iter(|| {
                black_box(simd_ops.compare(black_box(&data), black_box(&data_copy)));
            });
        });
        
        // Test different buffers with early termination (difference at 25%)
        group.bench_function(BenchmarkId::new("SIMD_diff_25pct", name), |b| {
            let data1 = generate_test_data(size);
            let mut data2 = data1.clone();
            if size > 4 {
                data2[size / 4] = data2[size / 4].wrapping_add(1);
            }
            b.iter(|| {
                black_box(simd_ops.compare(black_box(&data1), black_box(&data2)));
            });
        });
        
        // Compare with standard library
        group.bench_function(BenchmarkId::new("std_cmp", name), |b| {
            let data = generate_test_data(size);
            let data_copy = data.clone();
            b.iter(|| {
                black_box(black_box(&data) == black_box(&data_copy));
            });
        });
        
        // Compare with iterator-based comparison
        group.bench_function(BenchmarkId::new("iter_cmp", name), |b| {
            let data = generate_test_data(size);
            let data_copy = data.clone();
            b.iter(|| {
                black_box(
                    black_box(&data)
                        .iter()
                        .zip(black_box(&data_copy).iter())
                        .all(|(a, b)| a == b)
                );
            });
        });
    }
    
    group.finish();
}

//==============================================================================
// MEMORY SEARCH BENCHMARKS
//==============================================================================

/// Benchmark memory search operations
fn bench_memory_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Memory Search");
    
    let test_sizes = vec![
        ("small_64B", 64),
        ("medium_256B", 256),
        ("medium_1KB", 1024),
        ("medium_4KB", 4096),
        ("large_16KB", 16384),
        ("large_64KB", 65536),
        ("large_256KB", 262144),
    ];
    
    let simd_ops = SimdMemOps::new();
    
    for (name, size) in test_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Search for byte at beginning (best case)
        group.bench_function(BenchmarkId::new("SIMD_find_first", name), |b| {
            let mut data = generate_test_data(size);
            let needle = 0xAA;
            data[0] = needle;
            b.iter(|| {
                black_box(simd_ops.find_byte(black_box(&data), needle));
            });
        });
        
        // Search for byte at middle
        group.bench_function(BenchmarkId::new("SIMD_find_middle", name), |b| {
            let mut data = generate_test_data(size);
            let needle = 0xBB;
            data[size / 2] = needle;
            b.iter(|| {
                black_box(simd_ops.find_byte(black_box(&data), needle));
            });
        });
        
        // Search for byte not present (worst case)
        group.bench_function(BenchmarkId::new("SIMD_find_none", name), |b| {
            let data = generate_test_data(size);
            let needle = 0xFF; // Unlikely to be in generated data
            b.iter(|| {
                black_box(simd_ops.find_byte(black_box(&data), needle));
            });
        });
        
        // Compare with standard library memchr
        group.bench_function(BenchmarkId::new("std_position", name), |b| {
            let mut data = generate_test_data(size);
            let needle = 0xBB;
            data[size / 2] = needle;
            b.iter(|| {
                black_box(black_box(&data).iter().position(|&x| x == needle));
            });
        });
    }
    
    group.finish();
}

//==============================================================================
// MEMORY FILL BENCHMARKS
//==============================================================================

/// Benchmark memory fill operations
fn bench_memory_fill(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Memory Fill");
    
    let test_sizes = vec![
        ("small_64B", 64),
        ("medium_256B", 256),
        ("medium_1KB", 1024),
        ("medium_4KB", 4096),
        ("large_16KB", 16384),
        ("large_64KB", 65536),
        ("large_256KB", 262144),
    ];
    
    let simd_ops = SimdMemOps::new();
    
    for (name, size) in test_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark SIMD fill
        group.bench_function(BenchmarkId::new("SIMD", name), |b| {
            let mut buffer = vec![0u8; size];
            let value = 0x42;
            b.iter(|| {
                simd_ops.fill(black_box(&mut buffer), value);
            });
        });
        
        // Benchmark standard library fill
        group.bench_function(BenchmarkId::new("std_fill", name), |b| {
            let mut buffer = vec![0u8; size];
            let value = 0x42;
            b.iter(|| {
                black_box(&mut buffer).fill(value);
            });
        });
        
        // Benchmark unsafe ptr::write_bytes
        group.bench_function(BenchmarkId::new("ptr_write_bytes", name), |b| {
            let mut buffer = vec![0u8; size];
            let value = 0x42;
            b.iter(|| unsafe {
                std::ptr::write_bytes(black_box(buffer.as_mut_ptr()), value, size);
            });
        });
    }
    
    group.finish();
}

//==============================================================================
// SIMD TIER COMPARISON BENCHMARKS
//==============================================================================

/// Benchmark different SIMD tiers to show performance scaling
fn bench_simd_tiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Tier Comparison");
    
    // Only run if we can detect different tiers
    let simd_ops = SimdMemOps::new();
    let detected_tier = simd_ops.tier();
    
    println!("Detected SIMD tier: {:?}", detected_tier);
    println!("CPU Features: {:?}", simd_ops.cpu_features());
    
    // Test a range of sizes to see where different tiers excel
    let test_cases = vec![
        ("small_32B", 32),
        ("medium_256B", 256),
        ("medium_1KB", 1024),
        ("large_16KB", 16384),
        ("large_64KB", 65536),
    ];
    
    for (name, size) in test_cases {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark current tier for copy
        group.bench_function(
            BenchmarkId::new(format!("Copy_{:?}", detected_tier), name),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                        .expect("SIMD copy failed");
                });
            },
        );
        
        // Benchmark current tier for comparison
        group.bench_function(
            BenchmarkId::new(format!("Compare_{:?}", detected_tier), name),
            |b| {
                let data1 = generate_test_data(size);
                let data2 = data1.clone();
                b.iter(|| {
                    black_box(simd_ops.compare(black_box(&data1), black_box(&data2)));
                });
            },
        );
        
        // Benchmark current tier for search
        group.bench_function(
            BenchmarkId::new(format!("Search_{:?}", detected_tier), name),
            |b| {
                let data = generate_test_data(size);
                let needle = 0xFF; // Not present
                b.iter(|| {
                    black_box(simd_ops.find_byte(black_box(&data), needle));
                });
            },
        );
    }
    
    group.finish();
}

//==============================================================================
// MIXED WORKLOAD BENCHMARKS
//==============================================================================

/// Benchmark mixed workloads that simulate real-world usage patterns
fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Mixed Workload");
    
    let simd_ops = SimdMemOps::new();
    
    // Simulate a typical data processing pipeline
    group.bench_function("process_buffer_chain", |b| {
        let size = 4096;
        let src = generate_test_data(size);
        let mut temp1 = vec![0u8; size];
        let mut temp2 = vec![0u8; size];
        let pattern = 0x00;
        
        b.iter(|| {
            // Copy data to temp buffer
            simd_ops.copy_nonoverlapping(&src, &mut temp1).unwrap();
            
            // Search for pattern
            let pos = simd_ops.find_byte(&temp1, pattern);
            
            // Fill second buffer
            simd_ops.fill(&mut temp2, 0xFF);
            
            // Compare buffers
            let cmp = simd_ops.compare(&temp1, &temp2);
            
            black_box((pos, cmp));
        });
    });
    
    // Simulate memory manipulation with conditional operations
    group.bench_function("conditional_memory_ops", |b| {
        let size = 1024;
        let src = generate_test_data(size);
        let mut dst = vec![0u8; size];
        
        b.iter(|| {
            // Check if buffers are different
            if simd_ops.compare(&src, &dst) != 0 {
                // Copy if different
                simd_ops.copy_nonoverlapping(&src, &mut dst).unwrap();
            }
            
            // Search for a specific byte
            if let Some(pos) = simd_ops.find_byte(&dst, 0x42) {
                // Fill from that position
                if pos < size {
                    simd_ops.fill(&mut dst[pos..], 0x00);
                }
            }
            
            black_box(&dst);
        });
    });
    
    group.finish();
}

//==============================================================================
// PERFORMANCE RATIO ANALYSIS
//==============================================================================

/// Special benchmark to measure and report performance ratios against targets
fn bench_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance Target Analysis");
    
    // This group specifically measures against Phase 1.2 targets
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(10));
    
    let simd_ops = SimdMemOps::new();
    
    println!("\n=== SIMD Memory Operations Performance Target Analysis ===");
    println!("CPU Features: {:?}", simd_ops.cpu_features());
    println!("Selected SIMD Tier: {:?}", simd_ops.tier());
    
    // Small copies (≤64 bytes) - Target: 2-3x faster
    let small_sizes = vec![8, 16, 32, 64];
    for size in small_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_function(
            BenchmarkId::new("Target_Small_SIMD", format!("{}B", size)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                        .expect("SIMD copy failed");
                });
            },
        );
        
        group.bench_function(
            BenchmarkId::new("Target_Small_Std", format!("{}B", size)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    black_box(&mut dst).copy_from_slice(black_box(&src));
                });
            },
        );
    }
    
    // Medium copies (64-4096 bytes) - Target: 1.5-2x faster
    let medium_sizes = vec![128, 256, 512, 1024, 2048, 4096];
    for size in medium_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_function(
            BenchmarkId::new("Target_Medium_SIMD", format!("{}B", size)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                        .expect("SIMD copy failed");
                });
            },
        );
        
        group.bench_function(
            BenchmarkId::new("Target_Medium_Std", format!("{}B", size)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    black_box(&mut dst).copy_from_slice(black_box(&src));
                });
            },
        );
    }
    
    // Large copies (>4KB) - Target: Match or exceed system memcpy
    let large_sizes = vec![8192, 16384, 32768, 65536, 131072];
    for size in large_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_function(
            BenchmarkId::new("Target_Large_SIMD", format!("{}KB", size / 1024)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    simd_ops.copy_nonoverlapping(black_box(&src), black_box(&mut dst))
                        .expect("SIMD copy failed");
                });
            },
        );
        
        group.bench_function(
            BenchmarkId::new("Target_Large_Std", format!("{}KB", size / 1024)),
            |b| {
                let src = generate_test_data(size);
                let mut dst = vec![0u8; size];
                b.iter(|| {
                    black_box(&mut dst).copy_from_slice(black_box(&src));
                });
            },
        );
    }
    
    group.finish();
    
    println!("\n=== Target Performance Summary ===");
    println!("Small copies (≤64B): Target 2-3x faster than memcpy");
    println!("Medium copies (64-4096B): Target 1.5-2x faster with prefetching");
    println!("Large copies (>4KB): Target match or exceed system memcpy");
    println!("Run 'cargo bench --bench simd_memory_bench' to see detailed results");
    println!("HTML report available at target/criterion/report/index.html");
}

//==============================================================================
// BENCHMARK GROUPS
//==============================================================================

criterion_group!(
    memory_ops,
    bench_memory_copy,
    bench_aligned_memory_copy,
    bench_memory_compare,
    bench_memory_search,
    bench_memory_fill,
);

criterion_group!(
    tier_analysis,
    bench_simd_tiers,
    bench_mixed_workload,
);

criterion_group!(
    name = performance_targets;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(10));
    targets = bench_performance_targets
);

criterion_main!(memory_ops, tier_analysis, performance_targets);