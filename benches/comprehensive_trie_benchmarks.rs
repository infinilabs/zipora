//! Comprehensive performance benchmarks for all trie implementations
//!
//! This benchmark suite provides detailed performance analysis for:
//! - Double Array Trie
//! - Compressed Sparse Trie
//! - Nested LOUDS Trie (with various backends)
//!
//! Benchmarks cover insertion, lookup, memory usage, and scalability scenarios.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashMap;
use std::time::Duration;

use zipora::fsa::nested_louds_trie::{NestedLoudsTrie, NestingConfigBuilder};
use zipora::fsa::{
    DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig, PrefixIterable,
    StatisticsProvider, Trie,
};
use zipora::succinct::rank_select::{
    RankSelectInterleaved256,
};

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
        "app",
        "application",
        "compress",
        "data",
        "engine",
        "fast",
        "graph",
    ];

    for (i, prefix) in (0..count).zip(prefixes.iter().cycle()) {
        keys.push(format!("{}_{:06}", prefix, i).into_bytes());
    }

    keys
}

fn generate_unicode_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    let unicode_bases = vec!["café", "naïve", "résumé", "москва", "東京", "🌍", "🚀"];

    for (i, base) in (0..count).zip(unicode_bases.iter().cycle()) {
        keys.push(format!("{}_{:04}", base, i).into_bytes());
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

// =============================================================================
// DOUBLE ARRAY TRIE BENCHMARKS
// =============================================================================

fn bench_double_array_trie_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_array_trie_insertion");

    for &size in &[100, 1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("sequential", size), &keys, |b, keys| {
            b.iter(|| {
                let mut trie = DoubleArrayTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });

        let random_keys = generate_random_keys(size, 42);
        group.bench_with_input(BenchmarkId::new("random", size), &random_keys, |b, keys| {
            b.iter(|| {
                let mut trie = DoubleArrayTrie::new();
                for key in keys {
                    black_box(trie.insert(key).unwrap());
                }
                black_box(trie)
            });
        });
    }

    group.finish();
}

fn bench_double_array_trie_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_array_trie_lookup");

    for &size in &[1000, 10000, 50000] {
        let keys = generate_sequential_keys(size);
        let mut trie = DoubleArrayTrie::new();
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
            BenchmarkId::new("non_existent_keys", size),
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

fn bench_double_array_trie_prefix_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_array_trie_prefix");

    let keys = generate_prefix_heavy_keys(10000);
    let mut trie = DoubleArrayTrie::new();
    for key in &keys {
        trie.insert(key).unwrap();
    }

    let prefixes = vec![
        b"app".as_slice(),
        b"compress".as_slice(),
        b"data".as_slice(),
    ];

    for prefix in prefixes {
        group.bench_with_input(
            BenchmarkId::new("prefix_iteration", String::from_utf8_lossy(prefix)),
            &(&trie, prefix),
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

fn bench_double_array_trie_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("double_array_trie_config");

    let keys = generate_sequential_keys(5000);

    // Standard configuration
    group.bench_function("standard_config", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::new();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // SIMD enabled
    let simd_config = DoubleArrayTrieConfig {
        enable_simd: true,
        ..Default::default()
    };
    group.bench_function("simd_enabled", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::with_config(simd_config.clone());
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // Memory pool enabled
    let pool_config = DoubleArrayTrieConfig {
        use_memory_pool: true,
        pool_size_class: 16384,
        ..Default::default()
    };
    group.bench_function("memory_pool", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::with_config(pool_config.clone());
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.finish();
}

// =============================================================================
// NESTED LOUDS TRIE BENCHMARKS
// =============================================================================

fn bench_nested_louds_trie_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_louds_backends");

    let keys = generate_sequential_keys(5000);

    // RankSelectInterleaved256 backend
    group.bench_function("simple_backend", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // RankSelectInterleaved256 backend
    group.bench_function("interleaved256_backend", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // RankSelectInterleaved256 backend
    group.bench_function("separated512_backend", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // RankSelectFew backend (sparse implementation)
    group.bench_function("sparse_backend", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.finish();
}

fn bench_nested_louds_trie_nesting_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_louds_nesting");

    let keys = generate_prefix_heavy_keys(3000);

    for &levels in &[2, 3, 4, 5] {
        let config = NestingConfigBuilder::new()
            .max_levels(levels)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("nesting_levels", levels),
            &(config, &keys),
            |b, (config, keys)| {
                b.iter(|| {
                    let mut trie =
                        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config.clone())
                            .unwrap();
                    for key in *keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

fn bench_nested_louds_trie_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_louds_compression");

    let keys = generate_prefix_heavy_keys(5000);

    for &ratio in &[0.1, 0.3, 0.5, 0.8] {
        let config = NestingConfigBuilder::new()
            .fragment_compression_ratio(ratio)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("compression_ratio", (ratio * 100.0) as u32),
            &(config, &keys),
            |b, (config, keys)| {
                b.iter(|| {
                    let mut trie =
                        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config.clone())
                            .unwrap();
                    for key in *keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

fn bench_nested_louds_trie_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("nested_louds_lookup");

    for &size in &[1000, 5000, 10000] {
        let keys = generate_sequential_keys(size);
        let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
        for key in &keys {
            trie.insert(key).unwrap();
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("lookup", size),
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
// CROSS-IMPLEMENTATION COMPARISON BENCHMARKS
// =============================================================================

fn bench_trie_comparison_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("trie_comparison_insertion");

    let keys = generate_sequential_keys(5000);

    group.bench_function("double_array_trie", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::new();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_simple", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_interleaved", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("hashmap_baseline", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for (i, key) in keys.iter().enumerate() {
                black_box(map.insert(key.clone(), i));
            }
            black_box(map)
        });
    });

    group.finish();
}

fn bench_trie_comparison_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("trie_comparison_lookup");

    let keys = generate_sequential_keys(10000);

    // Build all data structures
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_simple = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    let mut nested_interleaved = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    let mut hashmap = HashMap::new();

    for (i, key) in keys.iter().enumerate() {
        da_trie.insert(key).unwrap();
        nested_simple.insert(key).unwrap();
        nested_interleaved.insert(key).unwrap();
        hashmap.insert(key.clone(), i);
    }

    group.throughput(Throughput::Elements(keys.len() as u64));

    group.bench_function("double_array_trie", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(da_trie.contains(key));
            }
        });
    });

    group.bench_function("nested_louds_simple", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(nested_simple.contains(key));
            }
        });
    });

    group.bench_function("nested_louds_interleaved", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(nested_interleaved.contains(key));
            }
        });
    });

    group.bench_function("hashmap_baseline", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(hashmap.contains_key(key));
            }
        });
    });

    group.finish();
}

fn bench_trie_comparison_prefix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("trie_comparison_prefix");

    let keys = generate_prefix_heavy_keys(5000);

    // Build tries (HashMap doesn't support prefix operations natively)
    let mut da_trie = DoubleArrayTrie::new();
    let mut nested_trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();

    for key in &keys {
        da_trie.insert(key).unwrap();
        nested_trie.insert(key).unwrap();
    }

    let test_prefix = b"app";

    group.bench_function("double_array_trie_prefix", |b| {
        b.iter(|| {
            let results: Vec<_> = da_trie.iter_prefix(test_prefix).collect();
            black_box(results)
        });
    });

    group.bench_function("nested_louds_trie_prefix", |b| {
        b.iter(|| {
            let results: Vec<_> = nested_trie.iter_prefix(test_prefix).collect();
            black_box(results)
        });
    });

    group.finish();
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

fn bench_memory_usage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for &size in &[1000, 5000, 10000] {
        let keys = generate_sequential_keys(size);

        group.bench_with_input(
            BenchmarkId::new("double_array_memory", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = DoubleArrayTrie::new();
                    for key in keys {
                        trie.insert(key).unwrap();
                    }
                    let stats = trie.stats();
                    black_box((trie, stats.memory_usage))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("nested_louds_memory", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
                    for key in keys {
                        trie.insert(key).unwrap();
                    }
                    let memory = trie.total_memory_usage();
                    black_box((trie, memory))
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SCALABILITY BENCHMARKS
// =============================================================================

fn bench_scalability_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_insertion");
    group.sample_size(20); // Reduce sample size for large datasets
    group.measurement_time(Duration::from_secs(30)); // Increase measurement time

    for &size in &[1000, 5000, 10000, 25000] {
        let keys = generate_sequential_keys(size);

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("double_array_scalability", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = DoubleArrayTrie::new();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("nested_louds_scalability", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
                    for key in keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// SPECIAL SCENARIO BENCHMARKS
// =============================================================================

fn bench_unicode_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_performance");

    let unicode_keys = generate_unicode_keys(2000);

    group.bench_function("double_array_unicode", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::new();
            for key in &unicode_keys {
                black_box(trie.insert(key).unwrap());
            }
            // Test lookups
            for key in &unicode_keys {
                black_box(trie.contains(key));
            }
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_unicode", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &unicode_keys {
                black_box(trie.insert(key).unwrap());
            }
            // Test lookups
            for key in &unicode_keys {
                black_box(trie.contains(key));
            }
            black_box(trie)
        });
    });

    group.finish();
}

fn bench_variable_length_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("variable_length_keys");

    let var_keys = generate_variable_length_keys(3000);

    group.bench_function("double_array_variable", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::new();
            for key in &var_keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_variable", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &var_keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.finish();
}

fn bench_builder_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder_patterns");

    let keys = generate_sequential_keys(5000);

    group.bench_function("double_array_incremental", |b| {
        b.iter(|| {
            let mut trie = DoubleArrayTrie::new();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("double_array_builder_sorted", |b| {
        b.iter(|| {
            let trie = DoubleArrayTrieBuilder::new()
                .build_from_sorted(keys.clone())
                .unwrap();
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_incremental", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
            for key in &keys {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    group.bench_function("nested_louds_builder", |b| {
        b.iter(|| {
            let builder = NestedLoudsTrie::<RankSelectInterleaved256>::builder();
            let trie: NestedLoudsTrie<RankSelectInterleaved256> =
                builder.build_from_iter(keys.iter().cloned()).unwrap();
            black_box(trie)
        });
    });

    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    double_array_benches,
    bench_double_array_trie_insertion,
    bench_double_array_trie_lookup,
    bench_double_array_trie_prefix_iteration,
    bench_double_array_trie_configurations
);

criterion_group!(
    nested_louds_benches,
    bench_nested_louds_trie_backends,
    bench_nested_louds_trie_nesting_levels,
    bench_nested_louds_trie_compression,
    bench_nested_louds_trie_lookup
);

criterion_group!(
    comparison_benches,
    bench_trie_comparison_insertion,
    bench_trie_comparison_lookup,
    bench_trie_comparison_prefix_operations,
    bench_memory_usage_comparison
);

criterion_group!(scalability_benches, bench_scalability_insertion);

criterion_group!(
    special_scenario_benches,
    bench_unicode_performance,
    bench_variable_length_keys,
    bench_builder_patterns
);

criterion_main!(
    double_array_benches,
    nested_louds_benches,
    comparison_benches,
    scalability_benches,
    special_scenario_benches
);
