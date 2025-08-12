//! Benchmark for comparing different memory pool implementations
//!
//! This benchmark compares the performance of all the new memory pool variants:
//! - Lock-Free Memory Pool
//! - Thread-Local Memory Pool
//! - Fixed Capacity Memory Pool
//! - Memory-Mapped Vectors
//! - Existing Secure Memory Pool (baseline)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use zipora::memory::*;

const ALLOCATION_SIZES: &[usize] = &[64, 256, 1024, 4096, 16384];
const ALLOCATION_COUNT: usize = 1000;

/// Benchmark secure memory pool (baseline)
fn bench_secure_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("secure_pool");
    
    for &size in ALLOCATION_SIZES {
        group.throughput(Throughput::Elements(ALLOCATION_COUNT as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            &size,
            |b, &size| {
                let config = SecurePoolConfig::new(size.max(1024), 1000, 8);
                let pool = SecureMemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    let mut allocations = Vec::new();
                    for _ in 0..ALLOCATION_COUNT {
                        let ptr = pool.allocate().unwrap();
                        allocations.push(ptr);
                    }
                    // Allocations automatically freed on drop
                    black_box(allocations);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark lock-free memory pool
fn bench_lockfree_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("lockfree_pool");
    
    for &size in ALLOCATION_SIZES {
        group.throughput(Throughput::Elements(ALLOCATION_COUNT as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            &size,
            |b, &size| {
                let config = LockFreePoolConfig::high_performance();
                let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
                
                b.iter(|| {
                    let mut allocations = Vec::new();
                    for _ in 0..ALLOCATION_COUNT {
                        if let Ok(ptr) = pool.allocate(size) {
                            let alloc = LockFreeAllocation::new(ptr, size, Arc::clone(&pool));
                            allocations.push(alloc);
                        }
                    }
                    // Allocations automatically freed on drop
                    black_box(allocations);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark thread-local memory pool
fn bench_threadlocal_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("threadlocal_pool");
    
    for &size in ALLOCATION_SIZES {
        group.throughput(Throughput::Elements(ALLOCATION_COUNT as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            &size,
            |b, &size| {
                let config = ThreadLocalPoolConfig::high_performance();
                let pool = ThreadLocalMemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    let mut allocations = Vec::new();
                    for _ in 0..ALLOCATION_COUNT {
                        if let Ok(alloc) = pool.allocate(size) {
                            allocations.push(alloc);
                        }
                    }
                    // Allocations automatically freed on drop
                    black_box(allocations);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark fixed capacity memory pool
fn bench_fixed_capacity_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_capacity_pool");
    
    for &size in ALLOCATION_SIZES {
        group.throughput(Throughput::Elements(ALLOCATION_COUNT as u64));
        group.bench_with_input(
            BenchmarkId::new("allocate_deallocate", size),
            &size,
            |b, &size| {
                let config = FixedCapacityPoolConfig {
                    max_block_size: size.max(4096),
                    total_blocks: ALLOCATION_COUNT * 2, // Ensure enough capacity
                    enable_stats: false, // Disable for max performance
                    ..FixedCapacityPoolConfig::default()
                };
                let pool = FixedCapacityMemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    let mut allocations = Vec::new();
                    for _ in 0..ALLOCATION_COUNT {
                        if let Ok(alloc) = pool.allocate(size) {
                            allocations.push(alloc);
                        }
                    }
                    // Allocations automatically freed on drop
                    black_box(allocations);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark memory-mapped vectors
fn bench_mmap_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmap_vec");
    
    // Test with different element counts for vectors
    let element_counts = [100, 500, 1000, 2000];
    
    for &count in &element_counts {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("push_pop", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    // Create temporary file path
                    let temp_path = format!("/tmp/zipora_bench_{}.mmap", 
                                          std::process::id());
                    
                    {
                        let config = MmapVecConfig::large_dataset();
                        let mut vec = MmapVec::<u64>::create(&temp_path, config).unwrap();
                        
                        // Push elements
                        for i in 0..count {
                            vec.push(i as u64).unwrap();
                        }
                        
                        // Pop elements
                        for _ in 0..count {
                            black_box(vec.pop());
                        }
                    }
                    
                    // Clean up
                    let _ = std::fs::remove_file(&temp_path);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory pool allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");
    
    // Test mixed allocation sizes (realistic workload)
    group.throughput(Throughput::Elements(ALLOCATION_COUNT as u64));
    group.bench_function("mixed_sizes_lockfree", |b| {
        let config = LockFreePoolConfig::high_performance();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for i in 0..ALLOCATION_COUNT {
                let size = ALLOCATION_SIZES[i % ALLOCATION_SIZES.len()];
                if let Ok(ptr) = pool.allocate(size) {
                    let alloc = LockFreeAllocation::new(ptr, size, Arc::clone(&pool));
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    group.bench_function("mixed_sizes_threadlocal", |b| {
        let config = ThreadLocalPoolConfig::high_performance();
        let pool = ThreadLocalMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for i in 0..ALLOCATION_COUNT {
                let size = ALLOCATION_SIZES[i % ALLOCATION_SIZES.len()];
                if let Ok(alloc) = pool.allocate(size) {
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    group.finish();
}

/// Benchmark contention scenarios
fn bench_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention");
    
    // Test lock-free pool under contention
    group.bench_function("lockfree_single_thread", |b| {
        let config = LockFreePoolConfig::high_performance();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for _ in 0..100 {
                if let Ok(ptr) = pool.allocate(1024) {
                    let alloc = LockFreeAllocation::new(ptr, 1024, Arc::clone(&pool));
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    // Test fixed capacity pool performance
    group.bench_function("fixed_capacity", |b| {
        let config = FixedCapacityPoolConfig {
            max_block_size: 4096,
            total_blocks: 1000,
            enable_stats: false,
            ..FixedCapacityPoolConfig::default()
        };
        let pool = FixedCapacityMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for _ in 0..100 {
                if let Ok(alloc) = pool.allocate(1024) {
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    group.finish();
}

/// Benchmark memory usage efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test small allocation efficiency
    group.bench_function("small_allocs_lockfree", |b| {
        let config = LockFreePoolConfig::compact();
        let pool = Arc::new(LockFreeMemoryPool::new(config).unwrap());
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for _ in 0..1000 {
                if let Ok(ptr) = pool.allocate(32) {
                    let alloc = LockFreeAllocation::new(ptr, 32, Arc::clone(&pool));
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    group.bench_function("small_allocs_fixed_capacity", |b| {
        let config = FixedCapacityPoolConfig::small_objects();
        let pool = FixedCapacityMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let mut allocations = Vec::new();
            for _ in 0..1000 {
                if let Ok(alloc) = pool.allocate(32) {
                    allocations.push(alloc);
                }
            }
            black_box(allocations);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_secure_pool,
    bench_lockfree_pool,
    bench_threadlocal_pool,
    bench_fixed_capacity_pool,
    bench_mmap_vec,
    bench_allocation_patterns,
    bench_contention,
    bench_memory_efficiency
);
criterion_main!(benches);