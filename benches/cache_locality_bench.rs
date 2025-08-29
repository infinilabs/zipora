//! Benchmarks for cache locality optimizations in hash maps

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::prelude::*;
use std::collections::HashMap;
use zipora::hash_map::{
    CacheOptimizedHashMap, GoldHashMap,
};

/// Benchmark configuration
const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];
const ITERATIONS: usize = 1000;

/// Generate random keys for testing
fn generate_keys(count: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..count).map(|_| rng.r#gen::<u64>()).collect()
}

/// Generate sequential keys for testing
fn generate_sequential_keys(count: usize) -> Vec<u64> {
    (0..count as u64).collect()
}

/// Generate strided keys for testing
fn generate_strided_keys(count: usize, stride: usize) -> Vec<u64> {
    (0..count).map(|i| (i * stride) as u64).collect()
}

/// Benchmark random access pattern
fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access");
    
    for &size in SIZES {
        let keys = generate_keys(size);
        let values: Vec<u64> = (0..size as u64).collect();
        
        // Standard HashMap baseline
        group.bench_function(BenchmarkId::new("std_hashmap", size), |b| {
            let mut map = HashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(k, v);
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for _ in 0..ITERATIONS {
                    let key = &keys[rand::random::<usize>() % size];
                    if let Some(v) = map.get(key) {
                        sum += **v;
                    }
                }
                black_box(sum)
            });
        });
        
        // GoldHashMap
        group.bench_function(BenchmarkId::new("gold_hashmap", size), |b| {
            let mut map = GoldHashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(k, v).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for _ in 0..ITERATIONS {
                    let key = &keys[rand::random::<usize>() % size];
                    if let Some(v) = map.get(key) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
        
        // CacheOptimizedHashMap
        group.bench_function(BenchmarkId::new("cache_optimized", size), |b| {
            let mut map = CacheOptimizedHashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(*k, *v).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for _ in 0..ITERATIONS {
                    let key = keys[rand::random::<usize>() % size];
                    if let Some(v) = map.get(&key) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
    }
    
    group.finish();
}

/// Benchmark sequential access pattern
fn bench_sequential_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_access");
    
    for &size in SIZES {
        let keys = generate_sequential_keys(size);
        let values: Vec<u64> = (0..size as u64).collect();
        
        // Standard HashMap baseline
        group.bench_function(BenchmarkId::new("std_hashmap", size), |b| {
            let mut map = HashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(k, v);
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for key in &keys {
                    if let Some(v) = map.get(key) {
                        sum += **v;
                    }
                }
                black_box(sum)
            });
        });
        
        // GoldHashMap
        group.bench_function(BenchmarkId::new("gold_hashmap", size), |b| {
            let mut map = GoldHashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(k, v).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for key in &keys {
                    if let Some(v) = map.get(key) {
                        sum += *v;
                    }
                }
                black_box(sum)
            });
        });
        
        // CacheOptimizedHashMap
        group.bench_function(BenchmarkId::new("cache_optimized", size), |b| {
            let mut map = CacheOptimizedHashMap::new();
            for (k, v) in keys.iter().zip(values.iter()) {
                map.insert(*k, *v).unwrap();
            }
            
            // Optimize for sequential access
            map.optimize_for_access_pattern();
            
            b.iter(|| {
                let mut sum = 0u64;
                for key in &keys {
                    if let Some(v) = map.get(key) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
    }
    
    group.finish();
}

/// Benchmark cache-line aware operations
fn bench_cache_line_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_line_ops");
    
    // Test different bucket sizes aligned to cache lines
    for &size in &[1000, 10000] {
        let keys = generate_keys(size);
        
        // Benchmark with prefetching enabled
        group.bench_function(BenchmarkId::new("with_prefetch", size), |b| {
            let mut map = CacheOptimizedHashMap::new();
            for (i, k) in keys.iter().enumerate() {
                map.insert(*k, i as u64).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for k in keys.iter().take(100) {
                    if let Some(v) = map.get(k) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
        
        // Benchmark without adaptive optimization
        group.bench_function(BenchmarkId::new("no_adaptive", size), |b| {
            let mut map = CacheOptimizedHashMap::new();
            map.set_adaptive_mode(false);
            
            for (i, k) in keys.iter().enumerate() {
                map.insert(*k, i as u64).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for k in keys.iter().take(100) {
                    if let Some(v) = map.get(k) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
    }
    
    group.finish();
}

/// Benchmark hot/cold data separation
fn bench_hot_cold_separation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_cold_separation");
    
    let size = 10000;
    let hot_size = 100; // 1% of data is hot
    let keys = generate_keys(size);
    let hot_keys: Vec<_> = keys.iter().take(hot_size).cloned().collect();
    
    // Without hot/cold separation
    group.bench_function("without_separation", |b| {
        let mut map = CacheOptimizedHashMap::new();
        for (i, k) in keys.iter().enumerate() {
            map.insert(*k, i as u64).unwrap();
        }
        
        b.iter(|| {
            let mut sum = 0u64;
            // 90% hot key access, 10% cold key access
            for _ in 0..900 {
                let k = hot_keys[rand::random::<usize>() % hot_size];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            for _ in 0..100 {
                let k = keys[hot_size + rand::random::<usize>() % (size - hot_size)];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    });
    
    // With hot/cold separation
    group.bench_function("with_separation", |b| {
        let mut map = CacheOptimizedHashMap::new();
        map.enable_hot_cold_separation(0.02); // 2% hot ratio
        
        for (i, k) in keys.iter().enumerate() {
            map.insert(*k, i as u64).unwrap();
        }
        
        // Warm up hot/cold separation
        for _ in 0..10 {
            for k in &hot_keys {
                map.get(k);
            }
        }
        map.rebalance_hot_cold();
        
        b.iter(|| {
            let mut sum = 0u64;
            // 90% hot key access, 10% cold key access
            for _ in 0..900 {
                let k = hot_keys[rand::random::<usize>() % hot_size];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            for _ in 0..100 {
                let k = keys[hot_size + rand::random::<usize>() % (size - hot_size)];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    });
    
    group.finish();
}

/// Benchmark resize operations
fn bench_resize_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_operations");
    
    // Standard resize
    group.bench_function("standard_resize", |b| {
        b.iter_batched(
            || {
                let mut map = GoldHashMap::with_capacity(100).unwrap();
                for i in 0..100 {
                    map.insert(i, i * 2).unwrap();
                }
                map
            },
            |mut map| {
                // Trigger resize by inserting beyond capacity
                for i in 100..1000 {
                    map.insert(i, i * 2).unwrap();
                }
                black_box(map)
            },
            BatchSize::SmallInput,
        );
    });
    
    // Cache-conscious resize
    group.bench_function("cache_conscious_resize", |b| {
        b.iter_batched(
            || {
                let mut map = CacheOptimizedHashMap::with_capacity(100).unwrap();
                for i in 0..100 {
                    map.insert(i, i * 2).unwrap();
                }
                map
            },
            |mut map| {
                // Trigger incremental resize
                for i in 100..1000 {
                    map.insert(i, i * 2).unwrap();
                }
                black_box(map)
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Benchmark cache metrics collection
fn bench_cache_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_metrics");
    
    let size = 10000;
    let keys = generate_keys(size);
    
    group.bench_function("with_metrics", |b| {
        let mut map = CacheOptimizedHashMap::new();
        for (i, k) in keys.iter().enumerate() {
            map.insert(*k, i as u64).unwrap();
        }
        
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..100 {
                let k = keys[rand::random::<usize>() % size];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            
            // Access metrics
            let metrics = map.cache_metrics();
            black_box(metrics.hit_ratio());
            black_box(sum)
        });
    });
    
    group.bench_function("without_metrics", |b| {
        let mut map = CacheOptimizedHashMap::new();
        map.reset_cache_metrics(); // Reset to avoid overhead
        
        for (i, k) in keys.iter().enumerate() {
            map.insert(*k, i as u64).unwrap();
        }
        
        b.iter(|| {
            let mut sum = 0u64;
            for _ in 0..100 {
                let k = keys[rand::random::<usize>() % size];
                if let Some(v) = map.get(&k) {
                    sum += v;
                }
            }
            black_box(sum)
        });
    });
    
    group.finish();
}

/// Benchmark strided access patterns
fn bench_strided_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("strided_access");
    
    for &stride in &[1, 4, 16, 64] {
        let size = 10000;
        let keys = generate_strided_keys(size, stride);
        
        group.bench_function(BenchmarkId::new("cache_optimized", stride), |b| {
            let mut map = CacheOptimizedHashMap::new();
            for (i, k) in keys.iter().enumerate() {
                map.insert(*k, i as u64).unwrap();
            }
            
            b.iter(|| {
                let mut sum = 0u64;
                for k in &keys {
                    if let Some(v) = map.get(k) {
                        sum += v;
                    }
                }
                black_box(sum)
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_random_access,
    bench_sequential_access,
    bench_cache_line_operations,
    bench_hot_cold_separation,
    bench_resize_operations,
    bench_cache_metrics,
    bench_strided_access
);

criterion_main!(benches);