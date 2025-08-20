//! Comprehensive benchmarks for the 5-level concurrency management system
//!
//! This benchmark suite validates the performance characteristics of all 5 levels
//! and measures the effectiveness of the adaptive selection mechanism.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use zipora::memory::{
    AdaptiveFiveLevelPool, ConcurrencyLevel, FiveLevelPoolConfig,
    NoLockingPool, MutexBasedPool, LockFreePool, ThreadLocalPool, FixedCapacityPool,
};

/// Benchmark configuration for consistent testing
struct BenchConfig {
    iterations: usize,
    allocation_size: usize,
    thread_count: usize,
}

impl BenchConfig {
    fn small() -> Self {
        Self {
            iterations: 1000,
            allocation_size: 64,
            thread_count: 1,
        }
    }

    fn medium() -> Self {
        Self {
            iterations: 10000,
            allocation_size: 512,
            thread_count: 4,
        }
    }

    fn large() -> Self {
        Self {
            iterations: 100000,
            allocation_size: 4096,
            thread_count: 8,
        }
    }
}

/// Benchmark Level 1: No Locking (single-threaded)
fn bench_no_locking_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_1_no_locking");
    
    for config in [BenchConfig::small(), BenchConfig::medium(), BenchConfig::large()] {
        group.throughput(Throughput::Elements(config.iterations as u64));
        
        group.bench_with_input(
            BenchmarkId::new("alloc_free", format!("{}x{}", config.iterations, config.allocation_size)),
            &config,
            |b, config| {
                b.iter(|| {
                    let pool_config = FiveLevelPoolConfig::performance_optimized();
                    let mut pool = NoLockingPool::new(pool_config).unwrap();
                    
                    let mut offsets = Vec::with_capacity(config.iterations);
                    
                    // Allocation phase
                    for _ in 0..config.iterations {
                        let offset = pool.alloc(black_box(config.allocation_size)).unwrap();
                        offsets.push(offset);
                    }
                    
                    // Deallocation phase
                    for offset in offsets {
                        pool.free(offset, config.allocation_size).unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Level 2: Mutex-based locking
fn bench_mutex_based_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_2_mutex_based");
    
    for config in [BenchConfig::small(), BenchConfig::medium()] {
        group.throughput(Throughput::Elements((config.iterations * config.thread_count) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_alloc_free", 
                format!("{}threads_{}x{}", config.thread_count, config.iterations, config.allocation_size)),
            &config,
            |b, config| {
                b.iter(|| {
                    let pool_config = FiveLevelPoolConfig::performance_optimized();
                    let pool = Arc::new(MutexBasedPool::new(pool_config).unwrap());
                    
                    let handles: Vec<_> = (0..config.thread_count)
                        .map(|_| {
                            let pool = Arc::clone(&pool);
                            let iterations = config.iterations;
                            let allocation_size = config.allocation_size;
                            
                            thread::spawn(move || {
                                let mut offsets = Vec::with_capacity(iterations);
                                
                                // Allocation phase
                                for _ in 0..iterations {
                                    let offset = pool.alloc(black_box(allocation_size)).unwrap();
                                    offsets.push(offset);
                                }
                                
                                // Deallocation phase
                                for offset in offsets {
                                    pool.free(offset, allocation_size).unwrap();
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Level 3: Lock-free programming
fn bench_lock_free_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_3_lock_free");
    
    for config in [BenchConfig::small(), BenchConfig::medium(), BenchConfig::large()] {
        group.throughput(Throughput::Elements((config.iterations * config.thread_count) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_alloc_free", 
                format!("{}threads_{}x{}", config.thread_count, config.iterations, config.allocation_size)),
            &config,
            |b, config| {
                b.iter(|| {
                    let pool_config = FiveLevelPoolConfig::performance_optimized();
                    let pool = Arc::new(LockFreePool::new(pool_config).unwrap());
                    
                    let handles: Vec<_> = (0..config.thread_count)
                        .map(|_| {
                            let pool = Arc::clone(&pool);
                            let iterations = config.iterations;
                            let allocation_size = config.allocation_size;
                            
                            thread::spawn(move || {
                                let mut offsets = Vec::with_capacity(iterations);
                                
                                // Allocation phase
                                for _ in 0..iterations {
                                    let offset = pool.alloc(black_box(allocation_size)).unwrap();
                                    offsets.push(offset);
                                }
                                
                                // Deallocation phase
                                for offset in offsets {
                                    pool.free(offset, allocation_size).unwrap();
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Level 4: Thread-local caching
fn bench_thread_local_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_4_thread_local");
    
    for config in [BenchConfig::small(), BenchConfig::medium()] {
        group.throughput(Throughput::Elements((config.iterations * config.thread_count) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_alloc_free", 
                format!("{}threads_{}x{}", config.thread_count, config.iterations, config.allocation_size)),
            &config,
            |b, config| {
                b.iter(|| {
                    let pool_config = FiveLevelPoolConfig::performance_optimized();
                    let pool = Arc::new(ThreadLocalPool::new(pool_config).unwrap());
                    
                    let handles: Vec<_> = (0..config.thread_count)
                        .map(|_| {
                            let pool = Arc::clone(&pool);
                            let iterations = config.iterations;
                            let allocation_size = config.allocation_size;
                            
                            thread::spawn(move || {
                                let mut offsets = Vec::with_capacity(iterations);
                                
                                // Allocation phase
                                for _ in 0..iterations {
                                    let offset = pool.alloc(black_box(allocation_size)).unwrap();
                                    offsets.push(offset);
                                }
                                
                                // Deallocation phase
                                for offset in offsets {
                                    pool.free(offset, allocation_size).unwrap();
                                }
                            })
                        })
                        .collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark Level 5: Fixed capacity
fn bench_fixed_capacity_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_5_fixed_capacity");
    
    for config in [BenchConfig::small(), BenchConfig::medium()] {
        group.throughput(Throughput::Elements(config.iterations as u64));
        
        group.bench_with_input(
            BenchmarkId::new("bounded_alloc_free", format!("{}x{}", config.iterations, config.allocation_size)),
            &config,
            |b, config| {
                b.iter(|| {
                    let mut pool_config = FiveLevelPoolConfig::performance_optimized();
                    pool_config.fixed_capacity = Some(16 * 1024 * 1024); // 16MB limit
                    let mut pool = FixedCapacityPool::new(pool_config).unwrap();
                    
                    let mut offsets = Vec::with_capacity(config.iterations);
                    
                    // Allocation phase
                    for _ in 0..config.iterations {
                        let offset = pool.alloc(black_box(config.allocation_size)).unwrap();
                        offsets.push(offset);
                    }
                    
                    // Deallocation phase
                    for offset in offsets {
                        pool.free(offset, config.allocation_size).unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark adaptive selection mechanism
fn bench_adaptive_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_selection");
    
    // Test different scenarios where adaptive selection should choose different levels
    let scenarios = [
        ("single_thread_small", BenchConfig::small()),
        ("multi_thread_medium", BenchConfig::medium()),
        ("high_concurrency_large", BenchConfig::large()),
    ];
    
    for (scenario_name, config) in scenarios {
        group.throughput(Throughput::Elements((config.iterations * config.thread_count) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("adaptive_alloc_free", scenario_name),
            &config,
            |b, config| {
                b.iter(|| {
                    let pool_config = FiveLevelPoolConfig::performance_optimized();
                    let mut pool = AdaptiveFiveLevelPool::new(pool_config).unwrap();
                    
                    if config.thread_count == 1 {
                        // Single-threaded benchmark
                        let mut offsets = Vec::with_capacity(config.iterations);
                        
                        for _ in 0..config.iterations {
                            let offset = pool.alloc(black_box(config.allocation_size)).unwrap();
                            offsets.push(offset);
                        }
                        
                        for offset in offsets {
                            pool.free(offset, config.allocation_size).unwrap();
                        }
                    } else {
                        // Multi-threaded benchmark
                        let handle = pool.get_handle().unwrap();
                        let handles: Vec<_> = (0..config.thread_count)
                            .map(|_| {
                                let handle = handle.clone();
                                let iterations = config.iterations;
                                let allocation_size = config.allocation_size;
                                
                                thread::spawn(move || {
                                    let mut offsets = Vec::with_capacity(iterations);
                                    
                                    for _ in 0..iterations {
                                        let offset = handle.alloc(black_box(allocation_size)).unwrap();
                                        offsets.push(offset);
                                    }
                                    
                                    for offset in offsets {
                                        handle.free(offset, allocation_size).unwrap();
                                    }
                                })
                            })
                            .collect();
                        
                        for handle in handles {
                            handle.join().unwrap();
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark level comparison
fn bench_level_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_comparison");
    group.throughput(Throughput::Elements(10000));
    
    let config = BenchConfig::medium();
    
    // Single-threaded comparison
    group.bench_function("level1_no_locking", |b| {
        b.iter(|| {
            let pool_config = FiveLevelPoolConfig::performance_optimized();
            let mut pool = AdaptiveFiveLevelPool::with_level(pool_config, ConcurrencyLevel::SingleThread).unwrap();
            
            let mut offsets = Vec::with_capacity(config.iterations);
            
            for _ in 0..config.iterations {
                let offset = pool.alloc(black_box(config.allocation_size)).unwrap();
                offsets.push(offset);
            }
            
            for offset in offsets {
                pool.free(offset, config.allocation_size).unwrap();
            }
        });
    });
    
    // Multi-threaded comparison (4 threads)
    let thread_count = 4;
    
    for (level_name, level) in [
        ("level2_mutex", ConcurrencyLevel::MultiThreadMutex),
        ("level3_lockfree", ConcurrencyLevel::MultiThreadLockFree),
        ("level4_threadlocal", ConcurrencyLevel::ThreadLocal),
    ] {
        group.bench_function(level_name, |b| {
            b.iter(|| {
                let pool_config = FiveLevelPoolConfig::performance_optimized();
                let pool = AdaptiveFiveLevelPool::with_level(pool_config, level).unwrap();
                let handle = pool.get_handle().unwrap();
                
                let handles: Vec<_> = (0..thread_count)
                    .map(|_| {
                        let handle = handle.clone();
                        let iterations = config.iterations / thread_count;
                        let allocation_size = config.allocation_size;
                        
                        thread::spawn(move || {
                            let mut offsets = Vec::with_capacity(iterations);
                            
                            for _ in 0..iterations {
                                let offset = handle.alloc(black_box(allocation_size)).unwrap();
                                offsets.push(offset);
                            }
                            
                            for offset in offsets {
                                handle.free(offset, allocation_size).unwrap();
                            }
                        })
                    })
                    .collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory fragmentation behavior
fn bench_fragmentation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("fragmentation_patterns");
    
    // Test different allocation patterns that can cause fragmentation
    let patterns = [
        ("sequential", false),
        ("interleaved", true),
    ];
    
    for (pattern_name, interleaved) in patterns {
        group.bench_function(pattern_name, |b| {
            b.iter(|| {
                let pool_config = FiveLevelPoolConfig::memory_optimized();
                let mut pool = NoLockingPool::new(pool_config).unwrap();
                
                let iterations = 1000;
                let mut offsets = Vec::with_capacity(iterations);
                
                if interleaved {
                    // Interleaved allocation/deallocation pattern
                    for i in 0..iterations {
                        let size = if i % 3 == 0 { 64 } else if i % 3 == 1 { 128 } else { 256 };
                        let offset = pool.alloc(black_box(size)).unwrap();
                        offsets.push((offset, size));
                        
                        // Free every 4th allocation to create fragmentation
                        if i % 4 == 3 && !offsets.is_empty() {
                            let (free_offset, free_size) = offsets.remove(offsets.len() / 2);
                            pool.free(free_offset, free_size).unwrap();
                        }
                    }
                } else {
                    // Sequential allocation pattern
                    for i in 0..iterations {
                        let size = 64 + (i % 192); // Variable sizes 64-255
                        let offset = pool.alloc(black_box(size)).unwrap();
                        offsets.push((offset, size));
                    }
                }
                
                // Free remaining allocations
                for (offset, size) in offsets {
                    pool.free(offset, size).unwrap();
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory alignment behavior
fn bench_alignment_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("alignment_patterns");
    
    let alignments = [8, 16, 32, 64];
    
    for alignment in alignments {
        group.bench_function(format!("align_{}", alignment), |b| {
            b.iter(|| {
                let mut pool_config = FiveLevelPoolConfig::performance_optimized();
                pool_config.alignment = alignment;
                let mut pool = NoLockingPool::new(pool_config).unwrap();
                
                let iterations = 1000;
                let mut offsets = Vec::with_capacity(iterations);
                
                for _ in 0..iterations {
                    let offset = pool.alloc(black_box(64)).unwrap();
                    offsets.push(offset);
                }
                
                for offset in offsets {
                    pool.free(offset, 64).unwrap();
                }
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_no_locking_pool,
    bench_mutex_based_pool,
    bench_lock_free_pool,
    bench_thread_local_pool,
    bench_fixed_capacity_pool,
    bench_adaptive_selection,
    bench_level_comparison,
    bench_fragmentation_patterns,
    bench_alignment_patterns
);

criterion_main!(benches);