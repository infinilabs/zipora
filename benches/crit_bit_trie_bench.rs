//! Performance benchmarks for Critical Bit Trie implementations
//!
//! This benchmark suite provides comprehensive performance analysis for:
//! - Standard Critical Bit Trie
//! - Space-Optimized Critical Bit Trie with BMI2 acceleration
//! - Memory usage patterns and cache performance
//! - Hardware acceleration effectiveness

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use std::time::Duration;
use zipora::fsa::{CritBitTrie, SpaceOptimizedCritBitTrie, PrefixIterable, StatisticsProvider, Trie};
use zipora::memory::{SecurePoolConfig};

// =============================================================================
// BENCHMARK DATA GENERATORS
// =============================================================================

fn generate_sequential_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("key_{:08}", i).into_bytes())
        .collect()
}

fn generate_random_keys(count: usize, seed: u64) -> Vec<Vec<u8>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut keys = Vec::with_capacity(count);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    for i in 0..count {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let key = format!("random_{:016x}_{:04}", hash, i % 10000);
        keys.push(key.into_bytes());
        hasher = DefaultHasher::new();
        hash.hash(&mut hasher);
    }

    keys.sort();
    keys.dedup();
    keys
}

fn generate_prefix_heavy_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    let prefixes = vec![
        "app", "application", "compress", "data", "engine", "fast", "graph",
        "index", "journal", "kernel", "library", "memory", "network", "object",
    ];

    for (i, prefix) in (0..count).zip(prefixes.iter().cycle()) {
        keys.push(format!("{}_{:06}", prefix, i).into_bytes());
    }

    keys
}

fn generate_variable_length_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();

    for i in 0..count {
        let length = (i % 50) + 5; // 5 to 54 characters
        let key = format!("var_len_key_{:04}_{}", i, "x".repeat(length));
        keys.push(key.into_bytes());
    }

    keys
}

fn generate_sparse_keys(count: usize) -> Vec<Vec<u8>> {
    // Generate keys with large bit differences (sparse critical bits)
    let mut keys = Vec::new();
    let mut value = 1u64;
    
    for i in 0..count {
        let key = format!("{:064b}_{:04}", value, i);
        keys.push(key.into_bytes());
        value = value.wrapping_mul(3).wrapping_add(7); // Create sparse distribution
    }
    
    keys
}

fn generate_dense_keys(count: usize) -> Vec<Vec<u8>> {
    // Generate keys with minimal bit differences (dense critical bits)
    (0..count)
        .map(|i| format!("dense_{:020}", i).into_bytes())
        .collect()
}

// =============================================================================
// STANDARD CRIT BIT TRIE BENCHMARKS
// =============================================================================

fn bench_crit_bit_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("crit_bit_insertion");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[100, 1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("sequential", size), &keys, |b, keys| {
            b.iter(|| {
                let mut trie = CritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });

        let random_keys = generate_random_keys(size, 42);
        group.bench_with_input(BenchmarkId::new("random", size), &random_keys, |b, keys| {
            b.iter(|| {
                let mut trie = CritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });

        let prefix_keys = generate_prefix_heavy_keys(size);
        group.bench_with_input(BenchmarkId::new("prefix_heavy", size), &prefix_keys, |b, keys| {
            b.iter(|| {
                let mut trie = CritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });
    }

    group.finish();
}

fn bench_crit_bit_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("crit_bit_lookup");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);
        let mut trie = CritBitTrie::new();
        for key in &keys {
            trie.insert(key).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("existing_keys", size),
            &(&trie, &keys),
            |b, (trie, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(trie.contains(key));
                    }
                });
            },
        );

        let non_existent_keys: Vec<_> = (0..size)
            .map(|i| format!("nonexistent_{:08}", i).into_bytes())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("non_existent", size),
            &(&trie, &non_existent_keys),
            |b, (trie, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(trie.contains(key));
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SPACE-OPTIMIZED CRIT BIT TRIE BENCHMARKS
// =============================================================================

fn bench_space_optimized_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_optimized_insertion");
    group.measurement_time(Duration::from_secs(10));

    for &size in &[100, 1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("sequential", size), &keys, |b, keys| {
            b.iter(|| {
                let mut trie = SpaceOptimizedCritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });

        let random_keys = generate_random_keys(size, 42);
        group.bench_with_input(BenchmarkId::new("random", size), &random_keys, |b, keys| {
            b.iter(|| {
                let mut trie = SpaceOptimizedCritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });

        // Test with secure memory pool
        group.bench_with_input(BenchmarkId::new("with_pool", size), &keys, |b, keys| {
            b.iter(|| {
                let config = SecurePoolConfig::small_secure();
                let mut trie = SpaceOptimizedCritBitTrie::with_secure_pool(config).unwrap();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });
    }

    group.finish();
}

fn bench_space_optimized_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("space_optimized_lookup");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);
        let mut trie = SpaceOptimizedCritBitTrie::new();
        for key in &keys {
            trie.insert(key).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("existing_keys", size),
            &(&trie, &keys),
            |b, (trie, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(trie.contains(key));
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// BMI2 ACCELERATION BENCHMARKS
// =============================================================================

fn bench_bmi2_acceleration_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmi2_acceleration");
    group.measurement_time(Duration::from_secs(3));

    // Test with different workloads to measure BMI2 impact
    let test_cases = vec![
        ("short_keys", generate_sequential_keys(1000).into_iter()
            .map(|mut k| { k.truncate(8); k })
            .collect::<Vec<_>>()),
        ("medium_keys", generate_sequential_keys(1000).into_iter()
            .map(|mut k| { k.truncate(32); k })
            .collect::<Vec<_>>()),
        ("long_keys", generate_sequential_keys(1000).into_iter()
            .map(|mut k| { k.extend_from_slice(b"_extended_key_data"); k })
            .collect::<Vec<_>>()),
        ("sparse_keys", generate_sparse_keys(1000)),
        ("dense_keys", generate_dense_keys(1000)),
    ];

    for (name, keys) in test_cases {
        // Standard CritBitTrie (no BMI2)
        group.bench_with_input(
            BenchmarkId::new("standard", name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = CritBitTrie::new();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    // Perform some lookups to test critical bit operations
                    for key in keys.iter().take(100) {
                        black_box(trie.contains(key));
                    }
                    black_box(trie)
                });
            },
        );

        // Space-optimized with BMI2 (if available)
        group.bench_with_input(
            BenchmarkId::new("bmi2_optimized", name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = SpaceOptimizedCritBitTrie::new();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    // Perform some lookups to test critical bit operations
                    for key in keys.iter().take(100) {
                        black_box(trie.contains(key));
                    }
                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[1000, 10000, 50000] {
        let keys = generate_prefix_heavy_keys(size);

        group.bench_with_input(
            BenchmarkId::new("standard_memory", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = CritBitTrie::new();
                    for key in keys {
                        trie.insert(key).unwrap();
                    }
                    let stats = trie.stats();
                    black_box((stats.memory_usage, stats.bits_per_key))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("space_optimized_memory", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = SpaceOptimizedCritBitTrie::new();
                    for key in keys {
                        trie.insert(key).unwrap();
                    }
                    let stats = trie.stats();
                    black_box((stats.memory_usage, stats.bits_per_key))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// PREFIX ITERATION BENCHMARKS
// =============================================================================

fn bench_prefix_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_iteration");
    
    let keys = generate_prefix_heavy_keys(10000);
    
    let mut standard_trie = CritBitTrie::new();
    let mut optimized_trie = SpaceOptimizedCritBitTrie::new();
    
    for key in &keys {
        standard_trie.insert(key).unwrap();
        optimized_trie.insert(key).unwrap();
    }

    let prefixes = vec![
        b"app".as_slice(),
        b"compress".as_slice(),
        b"data".as_slice(),
    ];

    for prefix in prefixes {
        group.bench_with_input(
            BenchmarkId::new("standard", String::from_utf8_lossy(prefix)),
            &(&standard_trie, prefix),
            |b, (trie, prefix)| {
                b.iter(|| {
                    let results: Vec<_> = trie.iter_prefix(prefix).collect();
                    black_box(results)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("space_optimized", String::from_utf8_lossy(prefix)),
            &(&optimized_trie, prefix),
            |b, (trie, prefix)| {
                b.iter(|| {
                    let results: Vec<_> = trie.iter_prefix(prefix).collect();
                    black_box(results)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// CACHE PERFORMANCE BENCHMARKS
// =============================================================================

fn bench_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    group.measurement_time(Duration::from_secs(5));

    // Test with different access patterns
    let size = 10000;
    let keys = generate_random_keys(size, 42);
    
    let mut standard_trie = CritBitTrie::new();
    let mut optimized_trie = SpaceOptimizedCritBitTrie::new();
    
    for key in &keys {
        standard_trie.insert(key).unwrap();
        optimized_trie.insert(key).unwrap();
    }

    // Sequential access pattern
    group.bench_with_input(
        BenchmarkId::new("standard_sequential", size),
        &(&standard_trie, &keys),
        |b, (trie, keys)| {
            b.iter(|| {
                for key in *keys {
                    black_box(trie.contains(key));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("optimized_sequential", size),
        &(&optimized_trie, &keys),
        |b, (trie, keys)| {
            b.iter(|| {
                for key in *keys {
                    black_box(trie.contains(key));
                }
            });
        },
    );

    // Random access pattern
    let mut random_indices: Vec<usize> = (0..keys.len()).collect();
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    random_indices.shuffle(&mut rng);

    group.bench_with_input(
        BenchmarkId::new("standard_random", size),
        &(&standard_trie, &keys, &random_indices),
        |b, (trie, keys, indices)| {
            b.iter(|| {
                for &i in *indices {
                    black_box(trie.contains(&keys[i]));
                }
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("optimized_random", size),
        &(&optimized_trie, &keys, &random_indices),
        |b, (trie, keys, indices)| {
            b.iter(|| {
                for &i in *indices {
                    black_box(trie.contains(&keys[i]));
                }
            });
        },
    );

    group.finish();
}

// =============================================================================
// COMPARISON WITH STANDARD COLLECTIONS
// =============================================================================

fn bench_comparison_with_hashmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("comparison");
    group.measurement_time(Duration::from_secs(5));

    for &size in &[1000, 10000, 50000] {
        let keys = generate_random_keys(size, 42);

        // HashMap insertion
        group.bench_with_input(
            BenchmarkId::new("hashmap_insert", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = HashMap::new();
                    for key in keys {
                        black_box(map.insert(key.clone(), true));
                    }
                    black_box(map)
                });
            },
        );

        // CritBitTrie insertion
        group.bench_with_input(
            BenchmarkId::new("critbit_insert", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = CritBitTrie::new();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        // SpaceOptimizedCritBitTrie insertion
        group.bench_with_input(
            BenchmarkId::new("space_optimized_insert", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = SpaceOptimizedCritBitTrie::new();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        // Prepare for lookup benchmarks
        let mut map = HashMap::new();
        let mut standard_trie = CritBitTrie::new();
        let mut optimized_trie = SpaceOptimizedCritBitTrie::new();
        
        for key in &keys {
            map.insert(key.clone(), true);
            standard_trie.insert(key).unwrap();
            optimized_trie.insert(key).unwrap();
        }

        // HashMap lookup
        group.bench_with_input(
            BenchmarkId::new("hashmap_lookup", size),
            &(&map, &keys),
            |b, (map, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(map.get(key));
                    }
                });
            },
        );

        // CritBitTrie lookup
        group.bench_with_input(
            BenchmarkId::new("critbit_lookup", size),
            &(&standard_trie, &keys),
            |b, (trie, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(trie.contains(key));
                    }
                });
            },
        );

        // SpaceOptimizedCritBitTrie lookup
        group.bench_with_input(
            BenchmarkId::new("space_optimized_lookup", size),
            &(&optimized_trie, &keys),
            |b, (trie, keys)| {
                b.iter(|| {
                    for key in *keys {
                        black_box(trie.contains(key));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_crit_bit_insertion,
    bench_crit_bit_lookup,
    bench_space_optimized_insertion,
    bench_space_optimized_lookup,
    bench_bmi2_acceleration_impact,
    bench_memory_efficiency,
    bench_prefix_iteration,
    bench_cache_performance,
    bench_comparison_with_hashmap,
);

criterion_main!(benches);