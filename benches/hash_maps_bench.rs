//! Comprehensive benchmarks for specialized hash map implementations
//!
//! This module provides detailed performance comparisons between:
//! - GoldHashMap (existing high-performance map)  
//! - GoldenRatioHashMap (enhanced with golden ratio growth)
//! - StringOptimizedHashMap (optimized for string keys)
//! - SmallHashMap (inline storage for small collections)
//! - std::HashMap (baseline comparison)

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
    measurement::WallTime, BenchmarkGroup
};
use std::collections::HashMap;
use std::time::Duration;

use zipora::hash_map::{
    GoldHashMap, GoldenRatioHashMap, StringOptimizedHashMap, SmallHashMap
};

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

const SMALL_SIZE: usize = 100;
const MEDIUM_SIZE: usize = 1_000;
const LARGE_SIZE: usize = 10_000;
const SIZES: &[usize] = &[SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];

const SMALL_MAP_SIZES: &[usize] = &[2, 4, 8, 16]; // For SmallHashMap inline testing

// =============================================================================
// INTEGER KEY BENCHMARKS
// =============================================================================

fn bench_integer_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_insertion");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // std::HashMap baseline
        group.bench_with_input(BenchmarkId::new("std::HashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = HashMap::with_capacity(size);
                for i in 0..size {
                    map.insert(black_box(i), black_box(i * 2));
                }
                black_box(map)
            });
        });

        // GoldHashMap (existing)
        group.bench_with_input(BenchmarkId::new("GoldHashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = GoldHashMap::with_capacity(size * 2).unwrap(); // Extra capacity to avoid full map
                for i in 0..size {
                    map.insert(black_box(i), black_box(i * 2)).unwrap();
                }
                black_box(map)
            });
        });

        // GoldenRatioHashMap (new)
        group.bench_with_input(BenchmarkId::new("GoldenRatioHashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = GoldenRatioHashMap::with_capacity(size).unwrap();
                for i in 0..size {
                    map.insert(black_box(i), black_box(i * 2)).unwrap();
                }
                black_box(map)
            });
        });

        // SmallHashMap (for small sizes only)
        if size <= 16 {
            group.bench_with_input(BenchmarkId::new("SmallHashMap<64>", size), &size, |b, &size| {
                b.iter(|| {
                    let mut map: SmallHashMap<usize, usize, 64> = SmallHashMap::new();
                    for i in 0..size {
                        map.insert(black_box(i), black_box(i * 2)).unwrap();
                    }
                    black_box(map)
                });
            });
        }
    }

    group.finish();
}

fn bench_integer_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("integer_lookup");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Prepare data
        let keys: Vec<usize> = (0..size).collect();
        
        // std::HashMap
        let mut std_map = HashMap::with_capacity(size);
        for &key in &keys {
            std_map.insert(key, key * 2);
        }
        
        // GoldHashMap
        let mut gold_map = GoldHashMap::with_capacity(size * 2).unwrap();
        for &key in &keys {
            gold_map.insert(key, key * 2).unwrap();
        }
        
        // GoldenRatioHashMap
        let mut golden_map = GoldenRatioHashMap::with_capacity(size).unwrap();
        for &key in &keys {
            golden_map.insert(key, key * 2).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("std::HashMap", size), &size, |b, _| {
            b.iter(|| {
                for &key in &keys {
                    black_box(std_map.get(&black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("GoldHashMap", size), &size, |b, _| {
            b.iter(|| {
                for &key in &keys {
                    black_box(gold_map.get(&black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("GoldenRatioHashMap", size), &size, |b, _| {
            b.iter(|| {
                for &key in &keys {
                    black_box(golden_map.get(&black_box(key)));
                }
            });
        });

        // SmallHashMap for small sizes
        if size <= 16 {
            let mut small_map: SmallHashMap<usize, usize, 64> = SmallHashMap::new();
            for &key in &keys {
                small_map.insert(key, key * 2).unwrap();
            }

            group.bench_with_input(BenchmarkId::new("SmallHashMap<64>", size), &size, |b, _| {
                b.iter(|| {
                    for &key in &keys {
                        black_box(small_map.get(&black_box(key)));
                    }
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// STRING KEY BENCHMARKS  
// =============================================================================

fn bench_string_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_insertion");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Generate string keys
        let keys: Vec<String> = (0..size)
            .map(|i| format!("key_{:06}", i))
            .collect();

        // std::HashMap baseline
        group.bench_with_input(BenchmarkId::new("std::HashMap", size), &size, |b, _| {
            b.iter(|| {
                let mut map = HashMap::with_capacity(size);
                for (i, key) in keys.iter().enumerate() {
                    map.insert(black_box(key.clone()), black_box(i));
                }
                black_box(map)
            });
        });

        // GoldHashMap
        group.bench_with_input(BenchmarkId::new("GoldHashMap", size), &size, |b, _| {
            b.iter(|| {
                let mut map = GoldHashMap::with_capacity(size * 2).unwrap();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(black_box(key.clone()), black_box(i)).unwrap();
                }
                black_box(map)
            });
        });

        // GoldenRatioHashMap
        group.bench_with_input(BenchmarkId::new("GoldenRatioHashMap", size), &size, |b, _| {
            b.iter(|| {
                let mut map = GoldenRatioHashMap::with_capacity(size).unwrap();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(black_box(key.clone()), black_box(i)).unwrap();
                }
                black_box(map)
            });
        });

        // StringOptimizedHashMap (should excel here)
        group.bench_with_input(BenchmarkId::new("StringOptimizedHashMap", size), &size, |b, _| {
            b.iter(|| {
                let mut map = StringOptimizedHashMap::with_capacity(size).unwrap();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(black_box(key.as_str()), black_box(i)).unwrap();
                }
                black_box(map)
            });
        });
    }

    group.finish();
}

fn bench_string_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_lookup");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Generate string keys
        let keys: Vec<String> = (0..size)
            .map(|i| format!("key_{:06}", i))
            .collect();

        // Prepare maps
        let mut std_map = HashMap::with_capacity(size);
        let mut gold_map = GoldHashMap::with_capacity(size * 2).unwrap();
        let mut golden_map = GoldenRatioHashMap::with_capacity(size).unwrap();
        let mut string_map = StringOptimizedHashMap::with_capacity(size).unwrap();

        for (i, key) in keys.iter().enumerate() {
            std_map.insert(key.clone(), i);
            gold_map.insert(key.clone(), i).unwrap();
            golden_map.insert(key.clone(), i).unwrap();
            string_map.insert(key.as_str(), i).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("std::HashMap", size), &size, |b, _| {
            b.iter(|| {
                for key in &keys {
                    black_box(std_map.get(black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("GoldHashMap", size), &size, |b, _| {
            b.iter(|| {
                for key in &keys {
                    black_box(gold_map.get(black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("GoldenRatioHashMap", size), &size, |b, _| {
            b.iter(|| {
                for key in &keys {
                    black_box(golden_map.get(black_box(key)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("StringOptimizedHashMap", size), &size, |b, _| {
            b.iter(|| {
                for key in &keys {
                    black_box(string_map.get(black_box(key.as_str())));
                }
            });
        });
    }

    group.finish();
}

// =============================================================================
// SMALL MAP SPECIALIZED BENCHMARKS
// =============================================================================

fn bench_small_map_inline_vs_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_map_inline_vs_heap");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &size in SMALL_MAP_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // SmallHashMap with different inline capacities
        group.bench_with_input(BenchmarkId::new("SmallHashMap<4>", size), &size, |b, &size| {
            b.iter(|| {
                let mut map: SmallHashMap<i32, i32, 4> = SmallHashMap::new();
                for i in 0..size as i32 {
                    map.insert(black_box(i), black_box(i * 2)).unwrap();
                }
                // Verify inline/heap status
                let is_inline = map.is_inline();
                black_box((map, is_inline))
            });
        });

        group.bench_with_input(BenchmarkId::new("SmallHashMap<8>", size), &size, |b, &size| {
            b.iter(|| {
                let mut map: SmallHashMap<i32, i32, 8> = SmallHashMap::new();
                for i in 0..size as i32 {
                    map.insert(black_box(i), black_box(i * 2)).unwrap();
                }
                let is_inline = map.is_inline();
                black_box((map, is_inline))
            });
        });

        group.bench_with_input(BenchmarkId::new("SmallHashMap<16>", size), &size, |b, &size| {
            b.iter(|| {
                let mut map: SmallHashMap<i32, i32, 16> = SmallHashMap::new();
                for i in 0..size as i32 {
                    map.insert(black_box(i), black_box(i * 2)).unwrap();
                }
                let is_inline = map.is_inline();
                black_box((map, is_inline))
            });
        });

        // std::HashMap for comparison
        group.bench_with_input(BenchmarkId::new("std::HashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = HashMap::with_capacity(size);
                for i in 0..size as i32 {
                    map.insert(black_box(i), black_box(i * 2));
                }
                black_box(map)
            });
        });
    }

    group.finish();
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for &size in &[1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));

        // Test golden ratio growth efficiency
        group.bench_with_input(BenchmarkId::new("GoldenRatioHashMap_growth", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = GoldenRatioHashMap::new();
                let mut load_factors = Vec::new();
                
                for i in 0..size {
                    map.insert(black_box(i), black_box(i)).unwrap();
                    
                    // Sample load factor periodically
                    if i % (size / 10).max(1) == 0 {
                        load_factors.push(map.load_factor());
                    }
                }
                
                black_box((map, load_factors))
            });
        });

        // Test string arena efficiency
        group.bench_with_input(BenchmarkId::new("StringOptimizedHashMap_arena", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = StringOptimizedHashMap::new();
                
                // Test with duplicate strings to verify interning
                let base_strings = ["prefix_", "common_", "shared_"];
                
                for i in 0..size {
                    let key = format!("{}{}", 
                        base_strings[i % base_strings.len()], 
                        i / base_strings.len()
                    );
                    map.insert(&key, black_box(i)).unwrap();
                }
                
                let stats = map.string_arena_stats();
                black_box((map, stats))
            });
        });
    }

    group.finish();
}

// =============================================================================
// COLLISION RESISTANCE BENCHMARKS
// =============================================================================

fn bench_collision_resistance(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_resistance");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let size = 10000;
    group.throughput(Throughput::Elements(size as u64));

    // Test with pathological keys designed to cause collisions
    let pathological_keys: Vec<String> = (0..size)
        .map(|i| {
            // Create keys that might hash to similar values
            let base = i % 100;  // Many keys will have same base
            format!("collision_key_{}_{}", base, i / 100)
        })
        .collect();

    group.bench_function("GoldenRatioHashMap_collisions", |b| {
        b.iter(|| {
            let mut map = GoldenRatioHashMap::new();
            for (i, key) in pathological_keys.iter().enumerate() {
                map.insert(black_box(key.clone()), black_box(i)).unwrap();
            }
            
            // Verify all keys can be found
            for (i, key) in pathological_keys.iter().enumerate() {
                assert_eq!(map.get(key), Some(&i));
            }
            
            black_box(map)
        });
    });

    group.bench_function("StringOptimizedHashMap_collisions", |b| {
        b.iter(|| {
            let mut map = StringOptimizedHashMap::new();
            for (i, key) in pathological_keys.iter().enumerate() {
                map.insert(black_box(key.as_str()), black_box(i)).unwrap();
            }
            
            // Verify all keys can be found
            for (i, key) in pathological_keys.iter().enumerate() {
                assert_eq!(map.get(key.as_str()), Some(&i));
            }
            
            black_box(map)
        });
    });

    group.bench_function("std::HashMap_collisions", |b| {
        b.iter(|| {
            let mut map = HashMap::with_capacity(size);
            for (i, key) in pathological_keys.iter().enumerate() {
                map.insert(black_box(key.clone()), black_box(i));
            }
            
            // Verify all keys can be found
            for (i, key) in pathological_keys.iter().enumerate() {
                assert_eq!(map.get(key), Some(&i));
            }
            
            black_box(map)
        });
    });

    group.finish();
}

// =============================================================================
// RANDOM ACCESS PATTERNS
// =============================================================================

fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let size = 10000;
    group.throughput(Throughput::Elements(size as u64));

    // Generate pseudo-random access pattern
    let mut keys = Vec::new();
    let mut hasher = DefaultHasher::new();
    for i in 0..size {
        i.hash(&mut hasher);
        let pseudo_random = (hasher.finish() as usize) % size;
        keys.push(pseudo_random);
    }

    // Prepare maps
    let mut golden_map = GoldenRatioHashMap::new();
    let mut std_map = HashMap::new();
    
    for i in 0..size {
        golden_map.insert(i, i * 2).unwrap();
        std_map.insert(i, i * 2);
    }

    group.bench_function("GoldenRatioHashMap_random", |b| {
        b.iter(|| {
            for &key in &keys {
                black_box(golden_map.get(&black_box(key)));
            }
        });
    });

    group.bench_function("std::HashMap_random", |b| {
        b.iter(|| {
            for &key in &keys {
                black_box(std_map.get(&black_box(key)));
            }
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    hash_map_benches,
    bench_integer_insertion,
    bench_integer_lookup,
    bench_string_insertion,
    bench_string_lookup,
    bench_small_map_inline_vs_heap,
    bench_memory_efficiency,
    bench_collision_resistance,
    bench_random_access
);

criterion_main!(hash_map_benches);