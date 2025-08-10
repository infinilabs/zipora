//! Benchmarks for cache-conscious data structures
//!
//! These benchmarks compare cache-aligned data structures against standard ones
//! to measure the performance impact of cache optimization.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashMap;
use zipora::{
    CacheAlignedVec, FastVec, get_numa_stats, get_optimal_numa_node,
    init_numa_pools, set_current_numa_node,
};

/// Benchmark cache-aligned vector vs standard vector for bulk operations
fn bench_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    for size in [1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Standard Vec
        group.bench_with_input(BenchmarkId::new("std_vec_push", size), size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::new();
                for i in 0..size {
                    vec.push(black_box(i));
                }
                black_box(vec)
            })
        });

        // FastVec (existing optimized vector)
        group.bench_with_input(BenchmarkId::new("fast_vec_push", size), size, |b, &size| {
            b.iter(|| {
                let mut vec = FastVec::new();
                for i in 0..size {
                    vec.push(black_box(i)).unwrap();
                }
                black_box(vec)
            })
        });

        // CacheAlignedVec
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_vec_push", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = CacheAlignedVec::new();
                    for i in 0..size {
                        vec.push(black_box(i)).unwrap();
                    }
                    black_box(vec)
                })
            },
        );

        // CacheAlignedVec with capacity pre-allocated
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_vec_with_capacity", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = CacheAlignedVec::with_capacity(size).unwrap();
                    for i in 0..size {
                        vec.push(black_box(i)).unwrap();
                    }
                    black_box(vec)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark random access patterns to test cache efficiency
fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access");

    const SIZE: usize = 100000;
    const ACCESSES: usize = 10000;

    // Generate random indices
    let indices: Vec<usize> = (0..ACCESSES).map(|i| (i * 7919) % SIZE).collect();

    group.throughput(Throughput::Elements(ACCESSES as u64));

    // Standard Vec
    group.bench_function("std_vec_random_access", |b| {
        let vec: Vec<i32> = (0..SIZE as i32).collect();
        b.iter(|| {
            let mut sum = 0i32;
            for &idx in &indices {
                sum = sum.wrapping_add(black_box(vec[idx]));
            }
            black_box(sum)
        })
    });

    // FastVec
    group.bench_function("fast_vec_random_access", |b| {
        let mut vec = FastVec::new();
        for i in 0..SIZE as i32 {
            vec.push(i).unwrap();
        }
        b.iter(|| {
            let mut sum = 0i32;
            for &idx in &indices {
                sum = sum.wrapping_add(black_box(*vec.get(idx).unwrap()));
            }
            black_box(sum)
        })
    });

    // CacheAlignedVec
    group.bench_function("cache_aligned_vec_random_access", |b| {
        let mut vec = CacheAlignedVec::with_capacity(SIZE).unwrap();
        for i in 0..SIZE as i32 {
            vec.push(i).unwrap();
        }
        b.iter(|| {
            let mut sum = 0i32;
            for &idx in &indices {
                sum = sum.wrapping_add(black_box(*vec.get(idx).unwrap()));
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Benchmark sequential access patterns (cache-friendly)
fn bench_sequential_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_access");

    for size in [10000, 100000, 1000000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Standard Vec
        group.bench_with_input(
            BenchmarkId::new("std_vec_sequential", size),
            size,
            |b, &size| {
                let vec: Vec<i32> = (0..size as i32).collect();
                b.iter(|| {
                    let mut sum = 0i32;
                    for val in &vec {
                        sum = sum.wrapping_add(black_box(*val));
                    }
                    black_box(sum)
                })
            },
        );

        // CacheAlignedVec
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_vec_sequential", size),
            size,
            |b, &size| {
                let mut vec = CacheAlignedVec::with_capacity(size).unwrap();
                for i in 0..size as i32 {
                    vec.push(i).unwrap();
                }
                b.iter(|| {
                    let mut sum = 0i32;
                    for i in 0..vec.len() {
                        sum = sum.wrapping_add(black_box(*vec.get(i).unwrap()));
                    }
                    black_box(sum)
                })
            },
        );

        // CacheAlignedVec using slice
        group.bench_with_input(
            BenchmarkId::new("cache_aligned_vec_slice", size),
            size,
            |b, &size| {
                let mut vec = CacheAlignedVec::with_capacity(size).unwrap();
                for i in 0..size as i32 {
                    vec.push(i).unwrap();
                }
                b.iter(|| {
                    let mut sum = 0i32;
                    for val in vec.as_slice() {
                        sum = sum.wrapping_add(black_box(*val));
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark NUMA-aware allocations
fn bench_numa_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_operations");

    // Initialize NUMA pools
    let _ = init_numa_pools();

    let numa_stats = get_numa_stats();
    if numa_stats.node_count > 1 {
        group.bench_function("numa_aware_allocation", |b| {
            b.iter(|| {
                let node = get_optimal_numa_node();
                let vec = CacheAlignedVec::<u64>::with_numa_node(node);
                black_box(vec)
            })
        });

        group.bench_function("numa_cross_node_allocation", |b| {
            b.iter(|| {
                // Allocate on different nodes to test cross-NUMA performance
                let node = black_box(0); // Force node 0
                let vec = CacheAlignedVec::<u64>::with_numa_node(node);
                black_box(vec)
            })
        });
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");

    const ITERATIONS: usize = 1000;

    group.throughput(Throughput::Elements(ITERATIONS as u64));

    // Standard allocation/deallocation
    group.bench_function("std_vec_alloc_dealloc", |b| {
        b.iter(|| {
            for i in 0..ITERATIONS {
                let mut vec = Vec::with_capacity(black_box(i + 100));
                for j in 0..100 {
                    vec.push(black_box(j));
                }
                black_box(vec);
                // Vec drops here
            }
        })
    });

    // Cache-aligned allocation/deallocation
    group.bench_function("cache_aligned_alloc_dealloc", |b| {
        b.iter(|| {
            for i in 0..ITERATIONS {
                let mut vec = CacheAlignedVec::with_capacity(black_box(i + 100)).unwrap();
                for j in 0..100 {
                    vec.push(black_box(j)).unwrap();
                }
                black_box(vec);
                // CacheAlignedVec drops here
            }
        })
    });

    group.finish();
}

/// Benchmark cache line utilization with different data layouts
fn bench_cache_line_utilization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_line_utilization");

    const SIZE: usize = 10000;

    // Structure that fits exactly in one cache line (64 bytes)
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CacheLineFriendly {
        data: [u8; 64],
    }

    // Structure that spans multiple cache lines
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CacheLineSpanning {
        data: [u8; 100],
    }

    group.throughput(Throughput::Elements(SIZE as u64));

    // Cache-friendly structure
    group.bench_function("cache_friendly_struct", |b| {
        let mut vec = CacheAlignedVec::with_capacity(SIZE).unwrap();
        for _ in 0..SIZE {
            vec.push(CacheLineFriendly { data: [42; 64] }).unwrap();
        }

        b.iter(|| {
            let mut sum = 0u8;
            for item in vec.as_slice() {
                sum = sum.wrapping_add(black_box(item.data[0]));
            }
            black_box(sum)
        })
    });

    // Cache-spanning structure
    group.bench_function("cache_spanning_struct", |b| {
        let mut vec = CacheAlignedVec::with_capacity(SIZE).unwrap();
        for _ in 0..SIZE {
            vec.push(CacheLineSpanning { data: [42; 100] }).unwrap();
        }

        b.iter(|| {
            let mut sum = 0u8;
            for item in vec.as_slice() {
                sum = sum.wrapping_add(black_box(item.data[0]));
            }
            black_box(sum)
        })
    });

    group.finish();
}

/// Benchmark NUMA statistics collection
fn bench_numa_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_stats");

    // Initialize pools with some allocations
    let _ = init_numa_pools();
    let _vec1 = CacheAlignedVec::<u64>::with_numa_node(0);
    let _vec2 = CacheAlignedVec::<u32>::with_numa_node(get_optimal_numa_node());

    group.bench_function("get_numa_stats", |b| {
        b.iter(|| {
            let stats = get_numa_stats();
            black_box(stats)
        })
    });

    group.finish();
}

/// Benchmark comparing performance with different NUMA configurations
fn bench_numa_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_configurations");

    let numa_stats = get_numa_stats();
    if numa_stats.node_count > 1 {
        const SIZE: usize = 10000;

        // Benchmark local NUMA node access
        group.bench_function("local_numa_access", |b| {
            let current_node = get_optimal_numa_node();
            let _ = set_current_numa_node(current_node);

            b.iter(|| {
                let mut vec = CacheAlignedVec::with_numa_node(current_node);
                for i in 0..SIZE {
                    vec.push(black_box(i)).unwrap();
                }

                let mut sum = 0usize;
                for i in 0..vec.len() {
                    sum = sum.wrapping_add(black_box(*vec.get(i).unwrap()));
                }
                black_box(sum)
            })
        });

        // Benchmark remote NUMA node access
        group.bench_function("remote_numa_access", |b| {
            let current_node = get_optimal_numa_node();
            let remote_node = (current_node + 1) % numa_stats.node_count;

            b.iter(|| {
                let mut vec = CacheAlignedVec::with_numa_node(remote_node);
                for i in 0..SIZE {
                    vec.push(black_box(i)).unwrap();
                }

                let mut sum = 0usize;
                for i in 0..vec.len() {
                    sum = sum.wrapping_add(black_box(*vec.get(i).unwrap()));
                }
                black_box(sum)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_operations,
    bench_random_access,
    bench_sequential_access,
    bench_numa_operations,
    bench_allocation_patterns,
    bench_cache_line_utilization,
    bench_numa_stats,
    bench_numa_configurations
);

criterion_main!(benches);
