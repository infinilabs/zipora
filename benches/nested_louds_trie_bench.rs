//! Benchmarks for Nested LOUDS Trie implementations
//!
//! This benchmark suite compares different nested LOUDS trie configurations
//! and rank/select backends to demonstrate the performance characteristics
//! and optimization opportunities.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use zipora::{
    fsa::{FiniteStateAutomaton, NestedLoudsTrie, NestingConfig, PrefixIterable, Trie},
    RankSelect256,
    succinct::rank_select::RankSelectInterleaved256,
};

/// Generate test data with different characteristics
fn generate_test_data(size: usize, pattern: &str) -> Vec<Vec<u8>> {
    match pattern {
        "random" => (0..size)
            .map(|i| format!("random_key_{:08}", i).into_bytes())
            .collect(),
        "prefixed" => {
            let prefixes = ["http://", "https://", "ftp://", "file://"];
            (0..size)
                .map(|i| {
                    let prefix = prefixes[i % prefixes.len()];
                    format!("{}example{:05}.com", prefix, i).into_bytes()
                })
                .collect()
        }
        "hierarchical" => (0..size)
            .map(|i| format!("level_{}/sublevel_{}/item_{:04}", i / 100, i / 10, i).into_bytes())
            .collect(),
        "sequential" => (0..size)
            .map(|i| format!("{:010}", i).into_bytes())
            .collect(),
        _ => vec![b"default".to_vec(); size],
    }
}

/// Benchmark insertion performance with different backends
fn bench_insertion_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_backends");
    let sizes = vec![100, 500, 1000];

    for size in sizes {
        let data = generate_test_data(size, "random");

        // Benchmark RankSelect256 backend
        group.bench_with_input(
            BenchmarkId::new("RankSelect256", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelect256>::new().unwrap();
                    for key in data {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        // Benchmark RankSelect256 backend
        group.bench_with_input(
            BenchmarkId::new("RankSelect256", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelect256>::new().unwrap();
                    for key in data {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        // Benchmark RankSelectInterleaved256 backend
        group.bench_with_input(
            BenchmarkId::new("RankSelectInterleaved256", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
                    for key in data {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );

        // Benchmark RankSelect256 backend
        group.bench_with_input(
            BenchmarkId::new("RankSelect256", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut trie = NestedLoudsTrie::<RankSelect256>::new().unwrap();
                    for key in data {
                        black_box(trie.insert(key).unwrap());
                    }
                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark lookup performance with different backends
fn bench_lookup_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_backends");
    let size = 1000;
    let data = generate_test_data(size, "random");

    // Pre-build tries with different backends
    let mut trie_simple = NestedLoudsTrie::<RankSelect256>::new().unwrap();
    let mut trie_sep256 = NestedLoudsTrie::<RankSelect256>::new().unwrap();
    let mut trie_int256 = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    let mut trie_sep512 = NestedLoudsTrie::<RankSelect256>::new().unwrap();

    for key in &data {
        trie_simple.insert(key).unwrap();
        trie_sep256.insert(key).unwrap();
        trie_int256.insert(key).unwrap();
        trie_sep512.insert(key).unwrap();
    }

    // Benchmark lookups
    group.bench_function("RankSelect256", |b| {
        b.iter(|| {
            for key in &data {
                black_box(trie_simple.contains(key));
            }
        });
    });

    group.bench_function("RankSelect256", |b| {
        b.iter(|| {
            for key in &data {
                black_box(trie_sep256.contains(key));
            }
        });
    });

    group.bench_function("RankSelectInterleaved256", |b| {
        b.iter(|| {
            for key in &data {
                black_box(trie_int256.contains(key));
            }
        });
    });

    group.bench_function("RankSelect256", |b| {
        b.iter(|| {
            for key in &data {
                black_box(trie_sep512.contains(key));
            }
        });
    });

    group.finish();
}

/// Benchmark different nesting levels
fn bench_nesting_levels(c: &mut Criterion) {
    let mut group = c.benchmark_group("nesting_levels");
    let data = generate_test_data(500, "hierarchical");

    for levels in [1, 2, 3, 4, 5, 6] {
        let config = NestingConfig::builder()
            .max_levels(levels)
            .fragment_compression_ratio(0.3)
            .build()
            .unwrap();

        group.bench_with_input(BenchmarkId::new("levels", levels), &data, |b, data| {
            b.iter(|| {
                let mut trie =
                    NestedLoudsTrie::<RankSelect256>::with_config(config.clone()).unwrap();
                for key in data {
                    black_box(trie.insert(key).unwrap());
                }

                // Test some lookups
                for key in data.iter().step_by(10) {
                    black_box(trie.contains(key));
                }

                black_box(trie)
            });
        });
    }

    group.finish();
}

/// Benchmark fragment compression ratios
fn bench_compression_ratios(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_ratios");
    let data = generate_test_data(300, "prefixed"); // Use prefixed data for better compression

    for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let config = NestingConfig::builder()
            .max_levels(4)
            .fragment_compression_ratio(ratio)
            .cache_optimization(true)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("ratio", (ratio * 100.0) as u32),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut trie =
                        NestedLoudsTrie::<RankSelectInterleaved256>::with_config(config.clone())
                            .unwrap();
                    for key in data {
                        black_box(trie.insert(key).unwrap());
                    }

                    // Trigger optimization
                    black_box(trie.optimize().unwrap());

                    // Test lookups after optimization
                    for key in data.iter().step_by(5) {
                        black_box(trie.contains(key));
                    }

                    black_box(trie)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different data patterns
fn bench_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_patterns");
    let size = 500;
    let patterns = ["random", "prefixed", "hierarchical", "sequential"];

    for pattern in patterns {
        let data = generate_test_data(size, pattern);
        let config = NestingConfig::builder()
            .max_levels(4)
            .fragment_compression_ratio(0.3)
            .adaptive_backend_selection(true)
            .build()
            .unwrap();

        group.bench_with_input(BenchmarkId::new("pattern", pattern), &data, |b, data| {
            b.iter(|| {
                let mut trie =
                    NestedLoudsTrie::<RankSelect256>::with_config(config.clone()).unwrap();
                for key in data {
                    black_box(trie.insert(key).unwrap());
                }

                // Test various operations
                for key in data.iter().step_by(7) {
                    black_box(trie.contains(key));
                    if let Some(prefix_len) = trie.longest_prefix(key) {
                        black_box(prefix_len);
                    }
                }

                black_box(trie)
            });
        });
    }

    group.finish();
}

/// Benchmark prefix operations
fn bench_prefix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_operations");
    let data = generate_test_data(200, "prefixed");

    // Build trie
    let mut trie = NestedLoudsTrie::<RankSelectInterleaved256>::new().unwrap();
    for key in &data {
        trie.insert(key).unwrap();
    }

    // Benchmark longest prefix
    group.bench_function("longest_prefix", |b| {
        b.iter(|| {
            for key in &data {
                black_box(trie.longest_prefix(key));
            }
        });
    });

    // Benchmark prefix iteration
    let prefixes = ["http", "https", "ftp"];
    group.bench_function("prefix_iteration", |b| {
        b.iter(|| {
            for prefix in &prefixes {
                let results: Vec<_> = trie.iter_prefix(prefix.as_bytes()).collect();
                black_box(results);
            }
        });
    });

    group.finish();
}

/// Benchmark memory usage and efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    let data = generate_test_data(1000, "prefixed");

    // Compare different configurations for memory efficiency
    let configs = [
        ("default", NestingConfig::default()),
        (
            "compressed",
            NestingConfig::builder()
                .max_levels(5)
                .fragment_compression_ratio(0.4)
                .cache_optimization(true)
                .build()
                .unwrap(),
        ),
        (
            "minimal",
            NestingConfig::builder()
                .max_levels(2)
                .fragment_compression_ratio(0.1)
                .cache_optimization(false)
                .build()
                .unwrap(),
        ),
    ];

    for (name, config) in configs {
        group.bench_with_input(BenchmarkId::new("config", name), &data, |b, data| {
            b.iter(|| {
                let mut trie =
                    NestedLoudsTrie::<RankSelect256>::with_config(config.clone()).unwrap();
                for key in data {
                    trie.insert(key).unwrap();
                }

                // Measure memory usage
                let memory_usage = trie.total_memory_usage();
                let stats = trie.performance_stats().clone(); // Clone to avoid borrowing issues

                black_box((trie, memory_usage, stats))
            });
        });
    }

    group.finish();
}

/// Benchmark builder patterns
fn bench_builder_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder_patterns");
    let data = generate_test_data(500, "random");
    let mut sorted_data = data.clone();
    sorted_data.sort();

    // Benchmark incremental insertion
    group.bench_function("incremental", |b| {
        b.iter(|| {
            let mut trie = NestedLoudsTrie::<RankSelect256>::new().unwrap();
            for key in &data {
                black_box(trie.insert(key).unwrap());
            }
            black_box(trie)
        });
    });

    // Benchmark sorted builder
    group.bench_function("sorted_builder", |b| {
        b.iter(|| {
            let trie =
                NestedLoudsTrie::<RankSelect256>::build_from_sorted(sorted_data.clone())
                    .unwrap();
            black_box(trie)
        });
    });

    // Benchmark unsorted builder
    group.bench_function("unsorted_builder", |b| {
        b.iter(|| {
            let trie = NestedLoudsTrie::<RankSelect256>::build_from_unsorted(data.clone())
                .unwrap();
            black_box(trie)
        });
    });

    // Benchmark optimized builder
    group.bench_function("optimized_builder", |b| {
        let config = NestingConfig::builder()
            .max_levels(4)
            .fragment_compression_ratio(0.3)
            .build()
            .unwrap();

        b.iter(|| {
            let trie = NestedLoudsTrie::<RankSelect256>::build_optimized(
                sorted_data.clone(),
                config.clone(),
            )
            .unwrap();
            black_box(trie)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insertion_backends,
    bench_lookup_backends,
    bench_nesting_levels,
    bench_compression_ratios,
    bench_data_patterns,
    bench_prefix_operations,
    bench_memory_efficiency,
    bench_builder_patterns
);

criterion_main!(benches);
