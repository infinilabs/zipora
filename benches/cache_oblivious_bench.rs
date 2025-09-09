//! Comprehensive benchmarks for cache-oblivious algorithms
//!
//! These benchmarks compare cache-oblivious algorithms against cache-aware
//! and standard algorithms to demonstrate performance characteristics across
//! different data sizes and cache hierarchies.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;
use zipora::algorithms::{
    CacheObliviousSort, CacheObliviousConfig, CacheObliviousSortingStrategy,
    RadixSort, RadixSortConfig,
};
use zipora::memory::cache_layout::{CacheHierarchy, detect_cache_hierarchy};
use zipora::system::cpu_features::get_cpu_features;

/// Benchmark cache-oblivious sort vs standard algorithms
fn bench_sorting_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_algorithms");
    group.measurement_time(Duration::from_secs(10));
    
    let cache_hierarchy = detect_cache_hierarchy();
    let cpu_features = get_cpu_features().clone();

    // Test different data sizes: L1, L2, L3, and beyond cache sizes
    let data_sizes = [
        ("L1_fit", cache_hierarchy.l1_size / 4),           // Fits in L1 cache
        ("L2_fit", cache_hierarchy.l2_size / 4),           // Fits in L2 cache  
        ("L3_fit", cache_hierarchy.l3_size / 4),           // Fits in L3 cache
        ("Beyond_L3", cache_hierarchy.l3_size * 2),        // Exceeds L3 cache
        ("Large", cache_hierarchy.l3_size * 8),            // Very large dataset
    ];

    for (size_name, element_count) in data_sizes.iter() {
        if *element_count > 1_000_000 {
            continue; // Skip very large tests for CI
        }
        
        group.throughput(Throughput::Elements(*element_count as u64));

        // Standard Rust sort
        group.bench_with_input(
            BenchmarkId::new("std_sort", size_name), 
            element_count, 
            |b, &size| {
                b.iter_with_setup(
                    || create_test_data(size),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    }
                )
            }
        );

        // Radix sort for comparison
        group.bench_with_input(
            BenchmarkId::new("radix_sort", size_name),
            element_count,
            |b, &size| {
                let config = RadixSortConfig::default();
                b.iter_with_setup(
                    || create_test_data_u32(size),
                    |mut data| {
                        let mut sorter = RadixSort::with_config(config.clone());
                        sorter.sort_u32(&mut data).unwrap();
                        black_box(data)
                    }
                )
            }
        );

        // Cache-oblivious sort with adaptive strategy
        group.bench_with_input(
            BenchmarkId::new("cache_oblivious_adaptive", size_name),
            element_count,
            |b, &size| {
                let config = CacheObliviousConfig {
                    cache_hierarchy: cache_hierarchy.clone(),
                    use_simd: true,
                    use_parallel: true,
                    small_threshold: 1024,
                    memory_pool: None,
                    cpu_features: cpu_features.clone(),
                };
                b.iter_with_setup(
                    || create_test_data(size),
                    |mut data| {
                        let mut sorter = CacheObliviousSort::with_config(config.clone());
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    }
                )
            }
        );

        // Cache-oblivious sort forced strategy
        group.bench_with_input(
            BenchmarkId::new("cache_oblivious_forced", size_name),
            element_count,
            |b, &size| {
                let mut config = CacheObliviousConfig {
                    cache_hierarchy: cache_hierarchy.clone(),
                    use_simd: true,
                    use_parallel: true,
                    small_threshold: 1024,
                    memory_pool: None,
                    cpu_features: cpu_features.clone(),
                };
                b.iter_with_setup(
                    || create_test_data(size),
                    |mut data| {
                        let mut sorter = CacheObliviousSort::with_config(config.clone());
                        // Force cache-oblivious strategy
                        if let Ok(()) = sorter.cache_oblivious_sort(&mut data) {
                            black_box(data)
                        } else {
                            black_box(data)
                        }
                    }
                )
            }
        );
    }

    group.finish();
}

/// Benchmark cache-oblivious algorithms with different access patterns
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");
    group.measurement_time(Duration::from_secs(8));
    
    let cache_hierarchy = detect_cache_hierarchy();
    let cpu_features = get_cpu_features().clone();
    let test_size = cache_hierarchy.l2_size / 4; // Medium-sized test

    group.throughput(Throughput::Elements(test_size as u64));

    // Random data
    group.bench_function("random_data", |b| {
        let config = CacheObliviousConfig {
            cache_hierarchy: cache_hierarchy.clone(),
            use_simd: true,
            use_parallel: true,
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: cpu_features.clone(),
        };
        b.iter_with_setup(
            || create_random_data(test_size),
            |mut data| {
                let mut sorter = CacheObliviousSort::with_config(config.clone());
                sorter.sort(&mut data).unwrap();
                black_box(data)
            }
        )
    });

    // Nearly sorted data
    group.bench_function("nearly_sorted_data", |b| {
        let config = CacheObliviousConfig {
            cache_hierarchy: cache_hierarchy.clone(),
            use_simd: true,
            use_parallel: true,
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: cpu_features.clone(),
        };
        b.iter_with_setup(
            || create_nearly_sorted_data(test_size),
            |mut data| {
                let mut sorter = CacheObliviousSort::with_config(config.clone());
                sorter.sort(&mut data).unwrap();
                black_box(data)
            }
        )
    });

    // Reverse sorted data
    group.bench_function("reverse_sorted_data", |b| {
        let config = CacheObliviousConfig {
            cache_hierarchy: cache_hierarchy.clone(),
            use_simd: true,
            use_parallel: true,
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: cpu_features.clone(),
        };
        b.iter_with_setup(
            || create_reverse_sorted_data(test_size),
            |mut data| {
                let mut sorter = CacheObliviousSort::with_config(config.clone());
                sorter.sort(&mut data).unwrap();
                black_box(data)
            }
        )
    });

    group.finish();
}

/// Benchmark SIMD vs non-SIMD cache-oblivious algorithms
fn bench_simd_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_impact");
    group.measurement_time(Duration::from_secs(6));
    
    let cache_hierarchy = detect_cache_hierarchy();
    let cpu_features = get_cpu_features().clone();
    
    // Test with medium-sized data where SIMD should make a difference
    let test_size = cache_hierarchy.l2_size / 8;
    group.throughput(Throughput::Elements(test_size as u64));

    // Cache-oblivious sort with SIMD enabled
    group.bench_function("simd_enabled", |b| {
        let config = CacheObliviousConfig {
            cache_hierarchy: cache_hierarchy.clone(),
            use_simd: true,
            use_parallel: false, // Focus on SIMD impact
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: cpu_features.clone(),
        };
        b.iter_with_setup(
            || create_test_data(test_size),
            |mut data| {
                let mut sorter = CacheObliviousSort::with_config(config.clone());
                sorter.sort(&mut data).unwrap();
                black_box(data)
            }
        )
    });

    // Cache-oblivious sort with SIMD disabled
    group.bench_function("simd_disabled", |b| {
        let config = CacheObliviousConfig {
            cache_hierarchy: cache_hierarchy.clone(),
            use_simd: false,
            use_parallel: false, // Focus on SIMD impact
            small_threshold: 1024,
            memory_pool: None,
            cpu_features: cpu_features.clone(),
        };
        b.iter_with_setup(
            || create_test_data(test_size),
            |mut data| {
                let mut sorter = CacheObliviousSort::with_config(config.clone());
                sorter.sort(&mut data).unwrap();
                black_box(data)
            }
        )
    });

    group.finish();
}

/// Benchmark cache-oblivious algorithms with different cache hierarchy sizes
fn bench_cache_hierarchy_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_hierarchy_adaptation");
    group.measurement_time(Duration::from_secs(6));
    
    let cpu_features = get_cpu_features().clone();
    
    // Test different simulated cache hierarchies
    let cache_configs = [
        ("small_cache", CacheHierarchy {
            l1_line_size: 64,
            l1_size: 16 * 1024,      // 16KB L1
            l2_line_size: 64,
            l2_size: 128 * 1024,     // 128KB L2
            l3_line_size: 64,
            l3_size: 2 * 1024 * 1024, // 2MB L3
            levels: 3,
            associativity: 8,
        }),
        ("large_cache", CacheHierarchy {
            l1_line_size: 64,
            l1_size: 64 * 1024,      // 64KB L1
            l2_line_size: 64,
            l2_size: 512 * 1024,     // 512KB L2
            l3_line_size: 64,
            l3_size: 16 * 1024 * 1024, // 16MB L3
            levels: 3,
            associativity: 16,
        }),
    ];

    for (cache_name, cache_hierarchy) in cache_configs.iter() {
        let test_size = cache_hierarchy.l2_size / 4;
        group.throughput(Throughput::Elements(test_size as u64));

        group.bench_with_input(
            BenchmarkId::new("cache_adaptive", cache_name),
            &test_size,
            |b, &size| {
                let config = CacheObliviousConfig {
                    cache_hierarchy: cache_hierarchy.clone(),
                    use_simd: true,
                    use_parallel: true,
                    small_threshold: 1024,
                    memory_pool: None,
                    cpu_features: cpu_features.clone(),
                };
                b.iter_with_setup(
                    || create_test_data(size),
                    |mut data| {
                        let mut sorter = CacheObliviousSort::with_config(config.clone());
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    }
                )
            }
        );
    }

    group.finish();
}

// Helper functions for creating test data

fn create_test_data(size: usize) -> Vec<i32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(0..1000000)).collect()
}

fn create_test_data_u32(size: usize) -> Vec<u32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(0..1000000)).collect()
}

fn create_random_data(size: usize) -> Vec<i32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen()).collect()
}

fn create_nearly_sorted_data(size: usize) -> Vec<i32> {
    use rand::prelude::*;
    let mut data: Vec<i32> = (0..size as i32).collect();
    let mut rng = thread_rng();
    
    // Introduce some disorder (swap ~5% of elements)
    let swaps = size / 20;
    for _ in 0..swaps {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        data.swap(i, j);
    }
    data
}

fn create_reverse_sorted_data(size: usize) -> Vec<i32> {
    (0..size as i32).rev().collect()
}

criterion_group!(
    benches,
    bench_sorting_algorithms,
    bench_access_patterns,
    bench_simd_impact,
    bench_cache_hierarchy_adaptation
);
criterion_main!(benches);