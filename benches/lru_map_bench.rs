//! LRU Map performance benchmarks
//!
//! These benchmarks compare the performance of zipora's LRU map implementations
//! against standard HashMap and other cache implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zipora::containers::{LruMap, ConcurrentLruMap, LruMapConfig, ConcurrentLruMapConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const CAPACITIES: &[usize] = &[64, 256, 1024, 4096];
const WORKLOAD_SIZES: &[usize] = &[100, 1000, 10000];

/// Benchmark LRU map operations vs HashMap
fn bench_lru_vs_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_vs_hashmap");
    
    for &capacity in CAPACITIES {
        for &workload_size in WORKLOAD_SIZES {
            group.throughput(Throughput::Elements(workload_size as u64));
            
            // LRU Map - Insert
            group.bench_with_input(
                BenchmarkId::new("lru_insert", format!("cap_{}_work_{}", capacity, workload_size)),
                &(capacity, workload_size),
                |b, &(cap, work)| {
                    b.iter(|| {
                        let lru = LruMap::new(cap).unwrap();
                        for i in 0..work {
                            black_box(lru.put(i, format!("value_{}", i)).unwrap());
                        }
                        lru
                    });
                },
            );
            
            // HashMap - Insert
            group.bench_with_input(
                BenchmarkId::new("hashmap_insert", format!("cap_{}_work_{}", capacity, workload_size)),
                &(capacity, workload_size),
                |b, &(_cap, work)| {
                    b.iter(|| {
                        let mut map = HashMap::new();
                        for i in 0..work {
                            black_box(map.insert(i, format!("value_{}", i)));
                        }
                        map
                    });
                },
            );
            
            // LRU Map - Get (hot)
            group.bench_with_input(
                BenchmarkId::new("lru_get_hot", format!("cap_{}_work_{}", capacity, workload_size)),
                &(capacity, workload_size),
                |b, &(cap, work)| {
                    let lru = LruMap::new(cap).unwrap();
                    for i in 0..std::cmp::min(cap, work) {
                        lru.put(i, format!("value_{}", i)).unwrap();
                    }
                    
                    b.iter(|| {
                        for i in 0..std::cmp::min(cap, work) {
                            black_box(lru.get(&i));
                        }
                    });
                },
            );
            
            // HashMap - Get
            group.bench_with_input(
                BenchmarkId::new("hashmap_get", format!("cap_{}_work_{}", capacity, workload_size)),
                &(capacity, workload_size),
                |b, &(_cap, work)| {
                    let mut map = HashMap::new();
                    for i in 0..work {
                        map.insert(i, format!("value_{}", i));
                    }
                    
                    b.iter(|| {
                        for i in 0..work {
                            black_box(map.get(&i));
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark different LRU map configurations
fn bench_lru_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("lru_configurations");
    group.sample_size(50);
    
    let workload_size = 1000;
    let capacity = 256;
    
    group.throughput(Throughput::Elements(workload_size as u64));
    
    // Performance optimized configuration
    group.bench_function("performance_optimized", |b| {
        let config = LruMapConfig::performance_optimized();
        let config = LruMapConfig { capacity, ..config };
        let lru = LruMap::with_config(config).unwrap();
        
        b.iter(|| {
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            for i in 0..capacity {
                black_box(lru.get(&i));
            }
        });
    });
    
    // Memory optimized configuration
    group.bench_function("memory_optimized", |b| {
        let config = LruMapConfig::memory_optimized();
        let config = LruMapConfig { capacity, ..config };
        let lru = LruMap::with_config(config).unwrap();
        
        b.iter(|| {
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            for i in 0..capacity {
                black_box(lru.get(&i));
            }
        });
    });
    
    // Security optimized configuration
    group.bench_function("security_optimized", |b| {
        let config = LruMapConfig::security_optimized();
        let config = LruMapConfig { capacity, ..config };
        let lru = LruMap::with_config(config).unwrap();
        
        b.iter(|| {
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            for i in 0..capacity {
                black_box(lru.get(&i));
            }
        });
    });
    
    group.finish();
}

/// Benchmark concurrent LRU map vs mutex-protected HashMap
fn bench_concurrent_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_performance");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    
    let capacity = 1024;
    let ops_per_thread = 1000;
    let thread_counts = &[1, 2, 4, 8];
    
    for &thread_count in thread_counts {
        group.throughput(Throughput::Elements((thread_count * ops_per_thread) as u64));
        
        // Concurrent LRU Map
        group.bench_with_input(
            BenchmarkId::new("concurrent_lru", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let lru = Arc::new(ConcurrentLruMap::new(capacity, 8).unwrap());
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let lru = lru.clone();
                            thread::spawn(move || {
                                for i in 0..ops_per_thread {
                                    let key = thread_id * ops_per_thread + i;
                                    black_box(lru.put(key, format!("value_{}", key)).unwrap());
                                    black_box(lru.get(&key));
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
        
        // Mutex-protected HashMap
        group.bench_with_input(
            BenchmarkId::new("mutex_hashmap", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter(|| {
                    let map = Arc::new(Mutex::new(HashMap::new()));
                    
                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let map = map.clone();
                            thread::spawn(move || {
                                for i in 0..ops_per_thread {
                                    let key = thread_id * ops_per_thread + i;
                                    {
                                        let mut guard = map.lock().unwrap();
                                        black_box(guard.insert(key, format!("value_{}", key)));
                                    }
                                    {
                                        let guard = map.lock().unwrap();
                                        black_box(guard.get(&key));
                                    }
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

/// Benchmark different access patterns
fn bench_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("access_patterns");
    
    let capacity = 512;
    let workload_size = 2000; // Larger than capacity to trigger evictions
    
    group.throughput(Throughput::Elements(workload_size as u64));
    
    // Sequential access pattern
    group.bench_function("sequential_access", |b| {
        let lru = LruMap::new(capacity).unwrap();
        
        b.iter(|| {
            // Fill cache
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            
            // Sequential access
            for i in 0..workload_size {
                black_box(lru.get(&i));
            }
        });
    });
    
    // Random access pattern
    group.bench_function("random_access", |b| {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let indices: Vec<usize> = (0..workload_size).collect();
        
        let lru = LruMap::new(capacity).unwrap();
        
        b.iter(|| {
            // Fill cache
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            
            // Random access
            for &i in indices.choose_multiple(&mut rng, workload_size) {
                black_box(lru.get(&i));
            }
        });
    });
    
    // Hot-spot access pattern (80/20 rule)
    group.bench_function("hotspot_access", |b| {
        let lru = LruMap::new(capacity).unwrap();
        let hot_keys = workload_size / 5; // 20% of keys
        
        b.iter(|| {
            // Fill cache
            for i in 0..workload_size {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
            
            // 80% of accesses to 20% of keys
            for _ in 0..(workload_size * 4 / 5) {
                let key = fastrand::usize(..hot_keys);
                black_box(lru.get(&key));
            }
            
            // 20% of accesses to remaining 80% of keys
            for _ in 0..(workload_size / 5) {
                let key = hot_keys + fastrand::usize(..(workload_size - hot_keys));
                black_box(lru.get(&key));
            }
        });
    });
    
    group.finish();
}

/// Benchmark cache hit ratios under different scenarios
fn bench_hit_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("hit_ratios");
    
    let capacity = 256;
    let access_count = 1000;
    
    // Test different working set sizes relative to capacity
    let working_set_ratios = &[0.5, 0.75, 1.0, 1.5, 2.0];
    
    for &ratio in working_set_ratios {
        let working_set_size = (capacity as f64 * ratio) as usize;
        
        group.bench_with_input(
            BenchmarkId::new("hit_ratio", format!("ratio_{:.1}", ratio)),
            &working_set_size,
            |b, &work_size| {
                let lru = LruMap::new(capacity).unwrap();
                
                // Pre-populate cache
                for i in 0..capacity {
                    lru.put(i, format!("value_{}", i)).unwrap();
                }
                
                b.iter(|| {
                    for _ in 0..access_count {
                        let key = fastrand::usize(..work_size);
                        black_box(lru.get(&key));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark eviction performance
fn bench_eviction_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction_performance");
    
    let capacity = 100;
    let overflow_factor = 5; // Insert 5x capacity to trigger many evictions
    
    group.throughput(Throughput::Elements((capacity * overflow_factor) as u64));
    
    // Without eviction callback
    group.bench_function("no_callback", |b| {
        b.iter(|| {
            let lru = LruMap::new(capacity).unwrap();
            for i in 0..(capacity * overflow_factor) {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
        });
    });
    
    // With eviction callback
    group.bench_function("with_callback", |b| {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use zipora::containers::EvictionCallback;
        
        struct CountingCallback {
            count: AtomicUsize,
        }
        
        impl EvictionCallback<usize, String> for CountingCallback {
            fn on_evict(&self, _key: &usize, _value: &String) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        b.iter(|| {
            let callback = CountingCallback { count: AtomicUsize::new(0) };
            let lru = LruMap::with_eviction_callback(capacity, callback).unwrap();
            for i in 0..(capacity * overflow_factor) {
                black_box(lru.put(i, format!("value_{}", i)).unwrap());
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let capacities = &[64, 256, 1024];
    
    for &capacity in capacities {
        group.bench_with_input(
            BenchmarkId::new("string_values", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    let lru = LruMap::new(cap).unwrap();
                    
                    // Fill with increasingly large string values
                    for i in 0..cap {
                        let value = "x".repeat(i + 1);
                        black_box(lru.put(i, value).unwrap());
                    }
                    
                    // Access all entries
                    for i in 0..cap {
                        black_box(lru.get(&i));
                    }
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("integer_values", capacity),
            &capacity,
            |b, &cap| {
                b.iter(|| {
                    let lru = LruMap::new(cap).unwrap();
                    
                    // Fill with integer values
                    for i in 0..cap {
                        black_box(lru.put(i, i * 2).unwrap());
                    }
                    
                    // Access all entries
                    for i in 0..cap {
                        black_box(lru.get(&i));
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_lru_vs_hashmap,
    bench_lru_configurations,
    bench_concurrent_performance,
    bench_access_patterns,
    bench_hit_ratios,
    bench_eviction_performance,
    bench_memory_usage
);

criterion_main!(benches);