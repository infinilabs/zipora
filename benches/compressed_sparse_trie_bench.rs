//! Comprehensive benchmarks for Compressed Sparse Patricia Trie
//!
//! This benchmark suite evaluates:
//! - Single-threaded vs multi-threaded performance
//! - Different concurrency levels
//! - Memory efficiency vs standard tries
//! - Lock-free operation performance
//! - Sparse vs dense data performance

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use zipora::fsa::{CompressedSparseTrie, ConcurrencyLevel, LoudsTrie, PatriciaTrie, Trie, StatisticsProvider};

use std::collections::HashSet;

/// Generate test data with controlled sparsity
fn generate_sparse_keys(count: usize, sparsity: f64) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    let base_chars = b"abcdefghijklmnopqrstuvwxyz0123456789";

    for i in 0..count {
        let mut key = Vec::new();

        // Create sparse structure by using limited character set
        let char_set_size = (base_chars.len() as f64 * sparsity) as usize;
        let char_set = &base_chars[..char_set_size.max(1)];

        // Variable length keys (4-20 bytes)
        let key_len = 4 + (i % 17);

        for j in 0..key_len {
            let char_index = (i * 7 + j * 3) % char_set.len();
            key.push(char_set[char_index]);
        }

        keys.push(key);
    }

    // Remove duplicates and sort
    let mut unique_keys: Vec<Vec<u8>> = keys
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique_keys.sort();
    unique_keys
}

/// Generate dense test data (high branching factor)
fn generate_dense_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();

    for i in 0..count {
        let key = format!("key_{:08x}_{:04x}", i, i % 1000);
        keys.push(key.into_bytes());
    }

    keys.sort();
    keys.dedup();
    keys
}

/// Benchmark CSP Trie insertion performance across concurrency levels
fn bench_csp_insertion(c: &mut Criterion) {
    let key_sizes = [1_000, 10_000, 100_000];
    let concurrency_levels = [
        ConcurrencyLevel::SingleThreadStrict,
        ConcurrencyLevel::SingleThreadShared,
        ConcurrencyLevel::OneWriteMultiRead,
        ConcurrencyLevel::MultiWriteMultiRead,
    ];

    for &key_count in &key_sizes {
        for &level in &concurrency_levels {
            let keys = generate_sparse_keys(key_count, 0.3); // 30% sparsity

            c.bench_with_input(
                BenchmarkId::new("csp_insertion", format!("{}_{:?}", key_count, level)),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || {
                            let trie = CompressedSparseTrie::new(level).unwrap();
                            (trie, keys.clone())
                        },
                        |(mut trie, keys)| {
                            for key in &keys {
                                black_box(trie.insert(key).unwrap());
                            }
                            trie
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
}

/// Benchmark CSP Trie lookup performance with different access patterns
fn bench_csp_lookup(c: &mut Criterion) {
    let key_count = 100_000;
    let keys = generate_sparse_keys(key_count, 0.3);

    // Pre-populate tries for different concurrency levels
    let levels = [
        ConcurrencyLevel::SingleThreadShared,
        ConcurrencyLevel::OneWriteMultiRead,
        ConcurrencyLevel::MultiWriteMultiRead,
    ];

    for &level in &levels {
        let mut group = c.benchmark_group(format!("csp_lookup_{:?}", level));
        group.throughput(Throughput::Elements(keys.len() as u64));

        group.bench_function("sequential", |b| {
            b.iter_batched(
                || {
                    let mut trie = CompressedSparseTrie::new(level).unwrap();
                    for key in &keys {
                        trie.insert(key).unwrap();
                    }
                    (trie, keys.clone())
                },
                |(trie, keys)| {
                    for key in &keys {
                        black_box(trie.contains(key));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

/// Benchmark concurrent operations with multiple threads
fn bench_concurrent_operations(c: &mut Criterion) {
    let key_count = 50_000;
    let thread_counts = [2, 4, 8];

    for &thread_count in &thread_counts {
        let keys = generate_sparse_keys(key_count, 0.3);
        let keys_per_thread = keys.len() / thread_count;

        c.bench_with_input(
            BenchmarkId::new("concurrent_insertion", thread_count),
            &keys,
            |b, keys| {
                b.iter_batched(
                    || {
                        let trie = CompressedSparseTrie::new(ConcurrencyLevel::MultiWriteMultiRead)
                            .unwrap();
                        (trie, keys.clone())
                    },
                    |(mut trie, keys)| {
                        // Simulate concurrent access by inserting in chunks
                        for chunk in keys.chunks(keys_per_thread) {
                            for key in chunk {
                                black_box(trie.insert(key).unwrap());
                            }
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

/// Compare CSP Trie vs standard Patricia Trie vs LOUDS Trie
fn bench_trie_comparison(c: &mut Criterion) {
    let key_count = 10_000; // Reduced for faster benchmarking
    let sparse_keys = generate_sparse_keys(key_count, 0.2); // Very sparse
    let dense_keys = generate_dense_keys(key_count);

    let datasets = [("sparse", sparse_keys), ("dense", dense_keys)];

    for (dataset_name, keys) in datasets {
        let mut group = c.benchmark_group(format!("trie_comparison_{}", dataset_name));
        group.throughput(Throughput::Elements(keys.len() as u64));

        // CSP Trie benchmark
        group.bench_function("csp_trie", |b| {
            b.iter_batched(
                || keys.clone(),
                |keys| {
                    let mut trie =
                        CompressedSparseTrie::new(ConcurrencyLevel::SingleThreadStrict).unwrap();
                    for key in &keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    trie
                },
                BatchSize::SmallInput,
            );
        });

        // Patricia Trie benchmark
        group.bench_function("patricia_trie", |b| {
            b.iter_batched(
                || keys.clone(),
                |keys| {
                    let mut trie = PatriciaTrie::new();
                    for key in &keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    trie
                },
                BatchSize::SmallInput,
            );
        });

        // LOUDS Trie benchmark (build from sorted)
        group.bench_function("louds_trie", |b| {
            b.iter_batched(
                || {
                    let mut sorted_keys = keys.clone();
                    sorted_keys.sort();
                    sorted_keys.dedup();
                    sorted_keys
                },
                |keys| black_box(LoudsTrie::build_from_sorted(keys).unwrap()),
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let key_counts = [10_000, 50_000, 100_000];
    let sparsity_levels = [0.1, 0.3, 0.5, 0.8]; // 10% to 80% sparsity

    for &key_count in &key_counts {
        for &sparsity in &sparsity_levels {
            let keys = generate_sparse_keys(key_count, sparsity);

            c.bench_with_input(
                BenchmarkId::new(
                    "memory_efficiency",
                    format!("{}keys_{}sparse", key_count, (sparsity * 100.0) as u32),
                ),
                &keys,
                |b, keys| {
                    b.iter_with_large_drop(|| {
                        let mut csp_trie =
                            CompressedSparseTrie::new(ConcurrencyLevel::SingleThreadStrict)
                                .unwrap();
                        let mut patricia_trie = PatriciaTrie::new();

                        for key in keys {
                            csp_trie.insert(key).unwrap();
                            patricia_trie.insert(key).unwrap();
                        }

                        let csp_memory = csp_trie.memory_usage();
                        let patricia_memory = patricia_trie.memory_usage();

                        // Return both tries to measure actual memory footprint
                        black_box((csp_trie, patricia_trie, csp_memory, patricia_memory))
                    });
                },
            );
        }
    }
}

/// Benchmark lock-free operations performance
fn bench_lock_free_operations(c: &mut Criterion) {
    let key_count = 10_000; // Reduced for faster benchmarking
    let keys = generate_sparse_keys(key_count, 0.3);

    let mut group = c.benchmark_group("lock_free_operations");
    group.throughput(Throughput::Elements(keys.len() as u64));

    // Benchmark with different levels of contention
    let contention_levels = [1, 2, 4];

    for &readers in &contention_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", readers),
            &keys,
            |b, keys| {
                b.iter_batched(
                    || {
                        let mut trie =
                            CompressedSparseTrie::new(ConcurrencyLevel::OneWriteMultiRead).unwrap();
                        for key in keys {
                            trie.insert(key).unwrap();
                        }
                        (trie, keys.clone())
                    },
                    |(trie, keys)| {
                        let keys_per_reader = keys.len() / readers;
                        // Simulate concurrent reads by processing in chunks
                        for chunk in keys.chunks(keys_per_reader) {
                            for key in chunk {
                                black_box(trie.contains(key));
                            }
                        }
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark different key patterns and their impact on performance
fn bench_key_patterns(c: &mut Criterion) {
    let key_count = 5_000; // Reduced for faster benchmarking

    // Different key patterns
    let patterns = [
        (
            "short_keys",
            (0..key_count)
                .map(|i| format!("k{}", i).into_bytes())
                .collect::<Vec<_>>(),
        ),
        (
            "long_keys",
            (0..key_count)
                .map(|i| format!("very_long_key_with_lots_of_characters_{:08x}", i).into_bytes())
                .collect::<Vec<_>>(),
        ),
        (
            "common_prefix",
            (0..key_count)
                .map(|i| format!("common_prefix_for_compression_test_{:06x}", i).into_bytes())
                .collect::<Vec<_>>(),
        ),
    ];

    for (pattern_name, keys) in patterns {
        let mut group = c.benchmark_group(format!("key_patterns_{}", pattern_name));
        group.throughput(Throughput::Elements(keys.len() as u64));

        group.bench_function("csp_trie_insert", |b| {
            b.iter_batched(
                || keys.clone(),
                |keys| {
                    let mut trie =
                        CompressedSparseTrie::new(ConcurrencyLevel::SingleThreadStrict).unwrap();
                    for key in &keys {
                        black_box(trie.insert(key).unwrap());
                    }
                    trie
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("csp_trie_lookup", |b| {
            b.iter_batched(
                || {
                    let mut trie =
                        CompressedSparseTrie::new(ConcurrencyLevel::SingleThreadStrict).unwrap();
                    for key in &keys {
                        trie.insert(key).unwrap();
                    }
                    (trie, keys.clone())
                },
                |(trie, keys)| {
                    for key in &keys {
                        black_box(trie.contains(key));
                    }
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    bench_csp_insertion,
    bench_csp_lookup,
    bench_concurrent_operations,
    bench_trie_comparison,
    bench_memory_efficiency,
    bench_lock_free_operations,
    bench_key_patterns
);

criterion_main!(benches);
