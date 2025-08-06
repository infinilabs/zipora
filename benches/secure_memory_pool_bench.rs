//! Comprehensive benchmarks for SecureMemoryPool performance validation
//!
//! This benchmark suite validates that the SecureMemoryPool maintains high performance
//! while providing security guarantees, comparing against the original MemoryPool
//! and standard allocators.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use zipora::memory::{
    MemoryPool, PoolConfig, SecureMemoryPool, SecurePoolConfig,
    get_global_pool_for_size, get_global_secure_pool_stats
};

/// Benchmark single-threaded allocation/deallocation throughput
fn bench_single_threaded_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_threaded_throughput");
    group.throughput(Throughput::Elements(10000));
    
    // Standard allocator baseline
    group.bench_function("std_alloc", |b| {
        let layout = Layout::from_size_align(1024, 8).unwrap();
        b.iter(|| {
            for _ in 0..10000 {
                let ptr = unsafe { alloc(layout) };
                black_box(ptr);
                unsafe { dealloc(ptr, layout) };
            }
        });
    });
    
    // Original MemoryPool (unsafe)
    group.bench_function("original_pool", |b| {
        let config = PoolConfig::small();
        let pool = MemoryPool::new(config).unwrap();
        b.iter(|| {
            for _ in 0..10000 {
                let ptr = pool.allocate().unwrap();
                black_box(ptr);
                pool.deallocate(ptr).unwrap();
            }
        });
    });
    
    // SecureMemoryPool
    group.bench_function("secure_pool", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        b.iter(|| {
            for _ in 0..10000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    // Global secure pool
    group.bench_function("global_secure_pool", |b| {
        let pool = get_global_pool_for_size(1024);
        b.iter(|| {
            for _ in 0..10000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.finish();
}

/// Benchmark multi-threaded allocation/deallocation
fn bench_multi_threaded_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_threaded_throughput");
    
    for thread_count in [2, 4, 8, 16].iter() {
        group.throughput(Throughput::Elements(*thread_count * 1000));
        
        // Original MemoryPool (unsafe)
        group.bench_with_input(
            BenchmarkId::new("original_pool", thread_count),
            thread_count,
            |b, &thread_count| {
                let config = PoolConfig::small();
                let pool = Arc::new(MemoryPool::new(config).unwrap());
                
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let pool = pool.clone();
                            thread::spawn(move || {
                                for _ in 0..1000 {
                                    let ptr = pool.allocate().unwrap();
                                    black_box(ptr);
                                    pool.deallocate(ptr).unwrap();
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
        
        // SecureMemoryPool
        group.bench_with_input(
            BenchmarkId::new("secure_pool", thread_count),
            thread_count,
            |b, &thread_count| {
                let config = SecurePoolConfig::small_secure();
                let pool = SecureMemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let pool = pool.clone();
                            thread::spawn(move || {
                                for _ in 0..1000 {
                                    let ptr = pool.allocate().unwrap();
                                    black_box(&ptr);
                                    drop(ptr);
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

/// Benchmark allocation patterns (burst vs steady)
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");
    
    // Burst allocation (allocate many, then deallocate all)
    group.bench_function("secure_pool_burst", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let mut ptrs = Vec::new();
            
            // Burst allocate
            for _ in 0..1000 {
                ptrs.push(pool.allocate().unwrap());
            }
            black_box(&ptrs);
            
            // Burst deallocate
            drop(ptrs);
        });
    });
    
    // Steady allocation (allocate and deallocate immediately)
    group.bench_function("secure_pool_steady", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..1000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    // Mixed pattern (some long-lived, some short-lived)
    group.bench_function("secure_pool_mixed", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let mut long_lived = Vec::new();
            
            for i in 0..1000 {
                let ptr = pool.allocate().unwrap();
                
                if i % 10 == 0 {
                    // Keep every 10th allocation as long-lived
                    long_lived.push(ptr);
                } else {
                    // Deallocate immediately
                    drop(ptr);
                }
            }
            
            black_box(&long_lived);
            drop(long_lived);
        });
    });
    
    group.finish();
}

/// Benchmark different size classes
fn bench_size_classes(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_classes");
    group.throughput(Throughput::Elements(5000));
    
    // Small allocations (1KB)
    group.bench_function("small_1kb", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..5000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    // Medium allocations (64KB)
    group.bench_function("medium_64kb", |b| {
        let config = SecurePoolConfig::medium_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..5000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    // Large allocations (1MB)
    group.bench_function("large_1mb", |b| {
        let config = SecurePoolConfig::large_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..1000 { // Fewer iterations for large allocations
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.finish();
}

/// Benchmark contention scenarios
fn bench_contention_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("contention_scenarios");
    group.measurement_time(Duration::from_secs(10));
    
    // High contention (many threads, small pool)
    group.bench_function("high_contention", |b| {
        let mut config = SecurePoolConfig::small_secure();
        config.max_chunks = 10; // Force contention
        config.local_cache_size = 2; // Small cache
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    let pool = pool.clone();
                    thread::spawn(move || {
                        for _ in 0..100 {
                            let ptr = pool.allocate().unwrap();
                            // Hold briefly to increase contention
                            thread::sleep(Duration::from_micros(1));
                            drop(ptr);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    // Low contention (few threads, large pool)
    group.bench_function("low_contention", |b| {
        let mut config = SecurePoolConfig::small_secure();
        config.max_chunks = 1000; // Large pool
        config.local_cache_size = 64; // Large cache
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let pool = pool.clone();
                    thread::spawn(move || {
                        for _ in 0..500 {
                            let ptr = pool.allocate().unwrap();
                            drop(ptr);
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
    
    group.finish();
}

/// Benchmark validation overhead
fn bench_validation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_overhead");
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("with_validation", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..1000 {
                let ptr = pool.allocate().unwrap();
                // Explicitly validate
                ptr.validate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.bench_function("without_explicit_validation", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..1000 {
                let ptr = pool.allocate().unwrap();
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");
    group.throughput(Throughput::Bytes(1024 * 100));
    
    group.bench_function("sequential_write", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..100 {
                let mut ptr = pool.allocate().unwrap();
                let slice = ptr.as_mut_slice();
                
                // Sequential write
                for (i, byte) in slice.iter_mut().enumerate() {
                    *byte = (i % 256) as u8;
                }
                
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.bench_function("random_access", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..100 {
                let mut ptr = pool.allocate().unwrap();
                let slice = ptr.as_mut_slice();
                
                // Random access pattern
                for _ in 0..100 {
                    let idx = fastrand::usize(..slice.len());
                    slice[idx] = fastrand::u8(..);
                }
                
                black_box(&ptr);
                drop(ptr);
            }
        });
    });
    
    group.finish();
}

/// Benchmark pool statistics collection
fn bench_statistics_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics_collection");
    
    group.bench_function("pool_stats", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        // Create some activity
        let _ptrs: Vec<_> = (0..100).map(|_| pool.allocate().unwrap()).collect();
        
        b.iter(|| {
            let stats = pool.stats();
            black_box(stats);
        });
    });
    
    group.bench_function("global_stats", |b| {
        // Create some activity in global pools
        let _ptrs: Vec<_> = (0..100).map(|_| {
            get_global_pool_for_size(1024).allocate().unwrap()
        }).collect();
        
        b.iter(|| {
            let stats = get_global_secure_pool_stats();
            black_box(stats);
        });
    });
    
    group.finish();
}

/// Benchmark cache efficiency
fn bench_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");
    
    // Test different cache sizes
    for cache_size in [4, 16, 64, 256].iter() {
        group.bench_with_input(
            BenchmarkId::new("cache_size", cache_size),
            cache_size,
            |b, &cache_size| {
                let mut config = SecurePoolConfig::small_secure();
                config.local_cache_size = cache_size;
                let pool = SecureMemoryPool::new(config).unwrap();
                
                b.iter(|| {
                    // Pattern that should benefit from caching
                    for _ in 0..cache_size * 2 {
                        let ptr = pool.allocate().unwrap();
                        drop(ptr);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark generation counter overhead
fn bench_generation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation_overhead");
    group.throughput(Throughput::Elements(10000));
    
    group.bench_function("secure_pool_with_generations", |b| {
        let config = SecurePoolConfig::small_secure();
        let pool = SecureMemoryPool::new(config).unwrap();
        
        b.iter(|| {
            for _ in 0..10000 {
                let ptr = pool.allocate().unwrap();
                // Access generation counter
                black_box(ptr.generation());
                drop(ptr);
            }
        });
    });
    
    group.finish();
}

/// Comprehensive performance comparison
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    group.throughput(Throughput::Elements(5000));
    group.measurement_time(Duration::from_secs(15));
    
    // Realistic workload: mix of allocation sizes and lifetimes
    group.bench_function("std_allocator_realistic", |b| {
        b.iter(|| {
            let mut long_lived = Vec::new();
            let layouts = [
                Layout::from_size_align(64, 8).unwrap(),
                Layout::from_size_align(256, 8).unwrap(),
                Layout::from_size_align(1024, 8).unwrap(),
                Layout::from_size_align(4096, 8).unwrap(),
            ];
            
            for i in 0..5000 {
                let layout = layouts[i % layouts.len()];
                let ptr = unsafe { alloc(layout) };
                
                if i % 20 == 0 {
                    long_lived.push((ptr, layout));
                } else {
                    unsafe { dealloc(ptr, layout) };
                }
            }
            
            // Cleanup long-lived allocations
            for (ptr, layout) in long_lived {
                unsafe { dealloc(ptr, layout) };
            }
        });
    });
    
    group.bench_function("secure_pool_realistic", |b| {
        let pools = [
            get_global_pool_for_size(64),
            get_global_pool_for_size(256), 
            get_global_pool_for_size(1024),
            get_global_pool_for_size(4096),
        ];
        
        b.iter(|| {
            let mut long_lived = Vec::new();
            
            for i in 0..5000 {
                let pool = pools[i % pools.len()];
                let ptr = pool.allocate().unwrap();
                
                if i % 20 == 0 {
                    long_lived.push(ptr);
                } else {
                    drop(ptr);
                }
            }
            
            // Cleanup is automatic with RAII
            drop(long_lived);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_single_threaded_throughput,
    bench_multi_threaded_throughput,
    bench_allocation_patterns,
    bench_size_classes,
    bench_contention_scenarios,
    bench_validation_overhead,
    bench_memory_access,
    bench_statistics_collection,
    bench_cache_efficiency,
    bench_generation_overhead,
    bench_comprehensive_comparison
);

criterion_main!(benches);