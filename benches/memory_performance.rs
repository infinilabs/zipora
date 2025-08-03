//! Comprehensive memory allocation performance benchmarks
//!
//! This benchmark suite tests the performance of the advanced tiered memory
//! allocator against standard allocation and compares with C++ performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zipora::memory::{
    tiered_allocate, tiered_deallocate, TieredMemoryAllocator, TieredConfig,
    MemoryMappedAllocator, MemoryPool, PoolConfig,
};
use std::alloc::{alloc, dealloc, Layout};

/// Test allocation patterns that match the C++ comparison
fn benchmark_allocation_patterns_detailed(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Allocation Performance");

    // Test the specific allocation patterns from PERF_VS_CPP.md
    let test_cases = vec![
        ("Small_100x64B", 100, 64),
        ("Medium_100x1KB", 100, 1024),
        ("Large_100x16KB", 100, 16 * 1024),
        ("XLarge_100x64KB", 100, 64 * 1024),
        ("Huge_10x1MB", 10, 1024 * 1024),
        ("Huge_10x4MB", 10, 4 * 1024 * 1024),
    ];

    for (name, count, size) in test_cases {
        group.throughput(Throughput::Elements(count as u64));

        // Test standard Rust allocation
        group.bench_function(BenchmarkId::new("Rust_Standard", name), |b| {
            b.iter(|| {
                let mut allocations = Vec::new();
                for _ in 0..count {
                    let layout = Layout::from_size_align(size, 8).unwrap();
                    let ptr = unsafe { alloc(layout) };
                    if !ptr.is_null() {
                        allocations.push((ptr, layout));
                        // Touch memory to ensure allocation
                        unsafe { ptr.write(42); }
                    }
                }
                
                // Deallocate all
                for (ptr, layout) in allocations {
                    unsafe { dealloc(ptr, layout); }
                }
                black_box(());
            });
        });

        // Test tiered allocator
        group.bench_function(BenchmarkId::new("Rust_Tiered", name), |b| {
            b.iter(|| {
                let mut allocations = Vec::new();
                for _ in 0..count {
                    if let Ok(allocation) = tiered_allocate(size) {
                        // Touch memory to ensure allocation
                        let slice = allocation.as_slice();
                        if !slice.is_empty() {
                            unsafe { *(slice.as_ptr() as *mut u8) = 42; }
                        }
                        allocations.push(allocation);
                    }
                }
                
                // Deallocate all
                for allocation in allocations {
                    let _ = tiered_deallocate(allocation);
                }
                black_box(());
            });
        });

        // Test custom tiered allocator instance for this size
        group.bench_function(BenchmarkId::new("Rust_Optimized", name), |b| {
            let config = TieredConfig {
                enable_small_pools: size <= 1024,
                enable_medium_pools: size > 1024 && size <= 16 * 1024,
                enable_mmap_large: size > 16 * 1024,
                enable_hugepages: size > 2 * 1024 * 1024,
                mmap_threshold: 16 * 1024,
                hugepage_threshold: 2 * 1024 * 1024,
            };
            let allocator = TieredMemoryAllocator::new(config).unwrap();
            
            b.iter(|| {
                let mut allocations = Vec::new();
                for _ in 0..count {
                    if let Ok(allocation) = allocator.allocate(size) {
                        // Touch memory to ensure allocation
                        let slice = allocation.as_slice();
                        if !slice.is_empty() {
                            unsafe { *(slice.as_ptr() as *mut u8) = 42; }
                        }
                        allocations.push(allocation);
                    }
                }
                
                // Deallocate all
                for allocation in allocations {
                    let _ = allocator.deallocate(allocation);
                }
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark memory-mapped allocator performance specifically
fn benchmark_mmap_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Mapped Allocation");

    let allocator = MemoryMappedAllocator::default();

    let sizes = vec![
        16 * 1024,    // 16KB
        64 * 1024,    // 64KB
        256 * 1024,   // 256KB
        1024 * 1024,  // 1MB
        4 * 1024 * 1024, // 4MB
    ];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("mmap_allocate", size), |b| {
            b.iter(|| {
                let allocation = allocator.allocate(size).unwrap();
                // Touch first and last page to ensure mapping
                let slice = allocation.as_slice();
                unsafe {
                    *(slice.as_ptr() as *mut u8) = 42;
                    *((slice.as_ptr() as *mut u8).add(size - 1)) = 84;
                }
                allocator.deallocate(allocation).unwrap();
                black_box(());
            });
        });

        // Test allocation reuse (cache efficiency)
        group.bench_function(BenchmarkId::new("mmap_reuse", size), |b| {
            b.iter(|| {
                let allocation1 = allocator.allocate(size).unwrap();
                allocator.deallocate(allocation1).unwrap();
                
                let allocation2 = allocator.allocate(size).unwrap();
                allocator.deallocate(allocation2).unwrap();
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark pool-based allocation for medium sizes
fn benchmark_pool_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Pool Allocation");

    // Create pools for different size classes
    let pool_configs = vec![
        ("1KB", PoolConfig::new(1024, 100, 8)),
        ("4KB", PoolConfig::new(4096, 50, 16)),
        ("16KB", PoolConfig::new(16384, 20, 32)),
    ];

    for (name, config) in pool_configs {
        let pool = MemoryPool::new(config).unwrap();
        let chunk_size = pool.config().chunk_size;
        
        group.throughput(Throughput::Bytes(chunk_size as u64));

        // Test allocation/deallocation performance
        group.bench_function(BenchmarkId::new("pool_alloc_dealloc", name), |b| {
            b.iter(|| {
                let chunk = pool.allocate().unwrap();
                // Touch memory
                unsafe { chunk.as_ptr().write(42); }
                pool.deallocate(chunk).unwrap();
                black_box(());
            });
        });

        // Test bulk allocation
        group.bench_function(BenchmarkId::new("pool_bulk_alloc", name), |b| {
            b.iter(|| {
                let mut chunks = Vec::new();
                for _ in 0..10 {
                    let chunk = pool.allocate().unwrap();
                    chunks.push(chunk);
                }
                
                for chunk in chunks {
                    pool.deallocate(chunk).unwrap();
                }
                black_box(());
            });
        });

        // Test pool warmup effect
        group.bench_function(BenchmarkId::new("pool_warm", name), |b| {
            // Warm up the pool
            let mut warmup_chunks = Vec::new();
            for _ in 0..50 {
                if let Ok(chunk) = pool.allocate() {
                    warmup_chunks.push(chunk);
                }
            }
            for chunk in warmup_chunks {
                let _ = pool.deallocate(chunk);
            }
            
            b.iter(|| {
                let chunk = pool.allocate().unwrap();
                pool.deallocate(chunk).unwrap();
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark allocation pattern adaptation
fn benchmark_adaptive_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Adaptive Allocation Patterns");

    let allocator = TieredMemoryAllocator::default().unwrap();

    // Test different allocation patterns
    let patterns = vec![
        ("small_dominated", vec![128, 256, 512, 256, 128, 512, 256, 128]),
        ("medium_dominated", vec![2048, 4096, 8192, 4096, 2048, 8192, 4096]),
        ("large_dominated", vec![32768, 65536, 131072, 65536, 32768]),
        ("mixed_pattern", vec![128, 4096, 65536, 256, 8192, 131072, 512]),
    ];

    for (pattern_name, sizes) in patterns {
        group.throughput(Throughput::Elements(sizes.len() as u64));

        group.bench_function(BenchmarkId::new("adaptive", pattern_name), |b| {
            b.iter(|| {
                let mut allocations = Vec::new();
                
                // Allocate according to pattern
                for &size in &sizes {
                    if let Ok(allocation) = allocator.allocate(size) {
                        allocations.push(allocation);
                    }
                }
                
                // Deallocate all
                for allocation in allocations {
                    let _ = allocator.deallocate(allocation);
                }
                
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark concurrent allocation performance
fn benchmark_concurrent_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Allocation");

    let allocator = TieredMemoryAllocator::default().unwrap();
    let allocator = std::sync::Arc::new(allocator);

    let sizes = vec![1024, 4096, 16384, 65536];

    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("concurrent", size), |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..4).map(|_| {
                    let allocator = allocator.clone();
                    std::thread::spawn(move || {
                        let mut allocations = Vec::new();
                        for _ in 0..10 {
                            if let Ok(allocation) = allocator.allocate(size) {
                                allocations.push(allocation);
                            }
                        }
                        
                        for allocation in allocations {
                            let _ = allocator.deallocate(allocation);
                        }
                    })
                }).collect();

                for handle in handles {
                    handle.join().unwrap();
                }
                
                black_box(());
            });
        });
    }

    group.finish();
}

/// Test memory allocation latency distribution
fn benchmark_allocation_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocation Latency");
    group.sample_size(1000); // More samples for latency analysis

    let allocator = TieredMemoryAllocator::default().unwrap();

    let sizes = vec![
        ("small", 512),
        ("medium", 4096),
        ("large", 65536),
        ("huge", 1048576),
    ];

    for (name, size) in sizes {
        group.bench_function(BenchmarkId::new("latency", name), |b| {
            b.iter(|| {
                let allocation = allocator.allocate(size).unwrap();
                let _ = allocator.deallocate(allocation);
                black_box(());
            });
        });
    }

    group.finish();
}

/// Performance regression test against the old system
fn benchmark_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance Regression Test");

    // Create old-style allocator (basic pools only)
    let old_style_config = TieredConfig {
        enable_small_pools: true,
        enable_medium_pools: false, // Disabled
        enable_mmap_large: false,   // Disabled
        enable_hugepages: false,    // Disabled
        mmap_threshold: usize::MAX,
        hugepage_threshold: usize::MAX,
    };
    let old_allocator = TieredMemoryAllocator::new(old_style_config).unwrap();

    // New optimized allocator
    let new_allocator = TieredMemoryAllocator::default().unwrap();

    let test_sizes = vec![1024, 16384, 65536, 1048576];

    for size in test_sizes {
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("old_system", size), |b| {
            b.iter(|| {
                let allocation = old_allocator.allocate(size).unwrap();
                let _ = old_allocator.deallocate(allocation);
                black_box(());
            });
        });

        group.bench_function(BenchmarkId::new("new_system", size), |b| {
            b.iter(|| {
                let allocation = new_allocator.allocate(size).unwrap();
                let _ = new_allocator.deallocate(allocation);
                black_box(());
            });
        });
    }

    group.finish();
}

criterion_group!(
    memory_benches,
    benchmark_allocation_patterns_detailed,
    benchmark_mmap_performance,
    benchmark_pool_performance,
    benchmark_adaptive_allocation,
    benchmark_concurrent_allocation,
    benchmark_allocation_latency,
    benchmark_performance_regression
);

criterion_main!(memory_benches);