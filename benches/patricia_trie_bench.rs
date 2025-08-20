//! Comprehensive benchmarks for Patricia Trie implementation
//!
//! This benchmark suite compares the sophisticated Patricia Trie performance against:
//! - HashMap (std::collections::HashMap)
//! - BTreeMap (std::collections::BTreeMap)
//! - Other trie implementations in zipora
//!
//! Performance targets based on topling-zip patterns:
//! - 3-4x faster than HashMap for dense key sets with path compression
//! - 50-70% memory reduction through advanced optimizations
//! - Hardware-accelerated operations with BMI2 and SIMD
//! - Sub-microsecond concurrent operations with token-based access

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use zipora::fsa::{
    PatriciaTrie, PrefixIterable, StatisticsProvider, Trie,
    PatriciaConfig, ConcurrencyLevel,
};

// =============================================================================
// BENCHMARK DATA GENERATORS
// =============================================================================

/// Generate sequential keys that benefit from path compression
fn generate_sequential_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("seq_key_{:08}", i).into_bytes())
        .collect()
}

/// Generate keys with common prefixes to test path compression effectiveness
fn generate_prefix_heavy_keys(count: usize) -> Vec<Vec<u8>> {
    let prefixes = [
        "application",
        "applications",
        "apply",
        "approve",
        "approximation",
        "banana", 
        "band",
        "bandana",
        "cat",
        "category",
        "catastrophe",
        "catalog",
    ];
    
    let mut keys = Vec::new();
    for i in 0..count {
        let prefix = prefixes[i % prefixes.len()];
        let key = format!("{}_item_{:06}", prefix, i / prefixes.len());
        keys.push(key.into_bytes());
    }
    
    keys.sort();
    keys.dedup();
    keys
}

/// Generate sparse keys to test worst-case path compression scenarios
fn generate_sparse_keys(count: usize) -> Vec<Vec<u8>> {
    let mut keys = Vec::new();
    let mut _step = 1;
    
    for i in 0..count {
        // Create varied patterns that don't share prefixes well
        let key = if i % 3 == 0 {
            format!("sparse_a_{:08x}", i * 17)
        } else if i % 3 == 1 {
            format!("different_b_{:08x}", i * 37)
        } else {
            format!("unique_c_{:08x}", i * 97)
        };
        keys.push(key.into_bytes());
        _step += 1;
    }
    
    keys.sort();
    keys
}

/// Generate random ASCII keys for realistic mixed patterns
fn generate_random_ascii_keys(count: usize, seed: u64) -> Vec<Vec<u8>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut keys = Vec::with_capacity(count);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    for i in 0..count {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Generate varied length strings
        let len = 8 + (hash % 40) as usize; // 8-47 characters
        let mut key = Vec::with_capacity(len);
        
        let mut working_hash = hash;
        for _ in 0..len {
            let ch = ((working_hash % 94) + 33) as u8; // Printable ASCII
            key.push(ch);
            working_hash = working_hash.wrapping_mul(1103515245).wrapping_add(12345);
        }
        
        keys.push(key);
        hasher = DefaultHasher::new();
        hash.hash(&mut hasher);
    }

    keys.sort();
    keys.dedup();
    keys
}

/// Generate binary keys to test general byte patterns
fn generate_binary_keys(count: usize, seed: u64) -> Vec<Vec<u8>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut keys = Vec::with_capacity(count);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    for i in 0..count {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        
        let len = 4 + (hash % 32) as usize; // 4-35 bytes
        let mut key = Vec::with_capacity(len);
        
        let mut working_hash = hash;
        for _ in 0..len {
            key.push((working_hash % 256) as u8);
            working_hash = working_hash.wrapping_mul(1103515245).wrapping_add(12345);
        }
        
        keys.push(key);
        hasher = DefaultHasher::new();
        hash.hash(&mut hasher);
    }

    keys.sort();
    keys.dedup();
    keys
}

// =============================================================================
// PATRICIA TRIE CONFIGURATION BENCHMARKS
// =============================================================================

fn bench_patricia_configurations(c: &mut Criterion) {
    let keys = generate_prefix_heavy_keys(10000);
    
    let configs = vec![
        ("default", PatriciaConfig::default()),
        ("performance", PatriciaConfig::performance_optimized()),
        ("memory", PatriciaConfig::memory_optimized()),
    ];
    
    let mut group = c.benchmark_group("patricia_configurations");
    group.throughput(Throughput::Elements(keys.len() as u64));
    
    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("insert", name),
            &keys,
            |b, keys| {
                b.iter_batched(
                    || PatriciaTrie::with_config(config.clone()),
                    |mut trie| {
                        for key in keys {
                            black_box(trie.insert(key).unwrap());
                        }
                        trie
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("lookup", name),
            &keys,
            |b, keys| {
                let mut trie = PatriciaTrie::with_config(config.clone());
                for key in keys {
                    trie.insert(key).unwrap();
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(trie.contains(key));
                    }
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// CONCURRENCY BENCHMARKS
// =============================================================================

fn bench_patricia_concurrency(c: &mut Criterion) {
    let keys = generate_sequential_keys(5000);
    
    let levels = vec![
        ("single_thread", ConcurrencyLevel::SingleThread),
        ("one_write_multi_read", ConcurrencyLevel::OneWriteMultiRead),
        ("multi_write_multi_read", ConcurrencyLevel::MultiWriteMultiRead),
    ];
    
    let mut group = c.benchmark_group("patricia_concurrency");
    group.throughput(Throughput::Elements(keys.len() as u64));
    
    for (name, level) in levels {
        if level == ConcurrencyLevel::SingleThread {
            group.bench_with_input(
                BenchmarkId::new("insert", name),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || PatriciaTrie::with_concurrency_level(level),
                        |mut trie| {
                            for key in keys {
                                black_box(trie.insert(key).unwrap());
                            }
                            trie
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
        } else {
            group.bench_with_input(
                BenchmarkId::new("token_insert", name),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || PatriciaTrie::with_concurrency_level(level),
                        |mut trie| {
                            let write_token = trie.acquire_write_token();
                            for key in keys {
                                black_box(trie.insert_with_token(key, &write_token).unwrap());
                            }
                            trie
                        },
                        BatchSize::SmallInput,
                    )
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new("token_lookup", name),
                &keys,
                |b, keys| {
                    let mut trie = PatriciaTrie::with_concurrency_level(level);
                    for key in keys {
                        trie.insert(key).unwrap();
                    }
                    
                    b.iter(|| {
                        let read_token = trie.acquire_read_token();
                        for key in keys {
                            black_box(trie.lookup_with_token(key, &read_token));
                        }
                    })
                },
            );
        }
    }
    
    group.finish();
}

// =============================================================================
// INSERTION PERFORMANCE BENCHMARKS
// =============================================================================

fn bench_insertion_performance(c: &mut Criterion) {
    let test_cases = vec![
        ("sequential_1k", generate_sequential_keys(1000)),
        ("sequential_10k", generate_sequential_keys(10000)),
        ("prefix_heavy_1k", generate_prefix_heavy_keys(1000)),
        ("prefix_heavy_10k", generate_prefix_heavy_keys(10000)),
        ("sparse_1k", generate_sparse_keys(1000)),
        ("sparse_10k", generate_sparse_keys(10000)),
        ("random_ascii_1k", generate_random_ascii_keys(1000, 42)),
        ("random_ascii_10k", generate_random_ascii_keys(10000, 42)),
        ("binary_1k", generate_binary_keys(1000, 123)),
        ("binary_10k", generate_binary_keys(10000, 123)),
    ];
    
    let mut group = c.benchmark_group("insertion_performance");
    
    for (name, keys) in &test_cases {
        group.throughput(Throughput::Elements(keys.len() as u64));
        
        // Patricia Trie
        group.bench_with_input(
            BenchmarkId::new("patricia_trie", name),
            keys,
            |b, keys| {
                b.iter_batched(
                    || PatriciaTrie::new(),
                    |mut trie| {
                        for key in keys {
                            black_box(trie.insert(key).unwrap());
                        }
                        trie
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        // HashMap comparison
        group.bench_with_input(
            BenchmarkId::new("hashmap", name),
            keys,
            |b, keys| {
                b.iter_batched(
                    || HashMap::new(),
                    |mut map| {
                        for (i, key) in keys.iter().enumerate() {
                            black_box(map.insert(key.clone(), i));
                        }
                        map
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        // BTreeMap comparison  
        group.bench_with_input(
            BenchmarkId::new("btreemap", name),
            keys,
            |b, keys| {
                b.iter_batched(
                    || BTreeMap::new(),
                    |mut map| {
                        for (i, key) in keys.iter().enumerate() {
                            black_box(map.insert(key.clone(), i));
                        }
                        map
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// LOOKUP PERFORMANCE BENCHMARKS  
// =============================================================================

fn bench_lookup_performance(c: &mut Criterion) {
    let test_cases = vec![
        ("sequential_10k", generate_sequential_keys(10000)),
        ("prefix_heavy_10k", generate_prefix_heavy_keys(10000)),
        ("sparse_10k", generate_sparse_keys(10000)),
        ("random_ascii_10k", generate_random_ascii_keys(10000, 42)),
        ("binary_10k", generate_binary_keys(10000, 123)),
    ];
    
    let mut group = c.benchmark_group("lookup_performance");
    
    for (name, keys) in &test_cases {
        group.throughput(Throughput::Elements(keys.len() as u64));
        
        // Patricia Trie
        group.bench_with_input(
            BenchmarkId::new("patricia_trie", name),
            keys,
            |b, keys| {
                let mut trie = PatriciaTrie::new();
                for key in keys {
                    trie.insert(key).unwrap();
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(trie.contains(key));
                    }
                })
            },
        );
        
        // HashMap comparison
        group.bench_with_input(
            BenchmarkId::new("hashmap", name),
            keys,
            |b, keys| {
                let mut map = HashMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i);
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(map.contains_key(key));
                    }
                })
            },
        );
        
        // BTreeMap comparison
        group.bench_with_input(
            BenchmarkId::new("btreemap", name),
            keys,
            |b, keys| {
                let mut map = BTreeMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i);
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(map.contains_key(key));
                    }
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// PREFIX ITERATION BENCHMARKS
// =============================================================================

fn bench_prefix_iteration(c: &mut Criterion) {
    let keys = generate_prefix_heavy_keys(10000);
    let prefixes = [b"app", b"ban", b"cat"];
    
    let mut group = c.benchmark_group("prefix_iteration");
    
    // Build Patricia Trie
    let mut patricia_trie = PatriciaTrie::new();
    for key in &keys {
        patricia_trie.insert(key).unwrap();
    }
    
    for prefix in &prefixes {
        group.bench_with_input(
            BenchmarkId::new("patricia_trie", std::str::from_utf8(*prefix).unwrap()),
            prefix,
            |b, &prefix| {
                b.iter(|| {
                    let results: Vec<Vec<u8>> = patricia_trie.iter_prefix(prefix).collect();
                    black_box(results)
                })
            },
        );
        
        // HashMap comparison (manual filtering)
        group.bench_with_input(
            BenchmarkId::new("hashmap_filter", std::str::from_utf8(*prefix).unwrap()),
            prefix,
            |b, &prefix| {
                b.iter(|| {
                    let results: Vec<Vec<u8>> = keys
                        .iter()
                        .filter(|key| key.starts_with(prefix))
                        .cloned()
                        .collect();
                    black_box(results)
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

fn bench_memory_usage(c: &mut Criterion) {
    let test_cases = vec![
        ("sequential_10k", generate_sequential_keys(10000)),
        ("prefix_heavy_10k", generate_prefix_heavy_keys(10000)),
        ("sparse_10k", generate_sparse_keys(10000)),
    ];
    
    let mut group = c.benchmark_group("memory_usage");
    
    for (name, keys) in &test_cases {
        group.bench_with_input(
            BenchmarkId::new("patricia_memory_stats", name),
            keys,
            |b, keys| {
                b.iter_batched(
                    || {
                        let mut trie = PatriciaTrie::new();
                        for key in keys {
                            trie.insert(key).unwrap();
                        }
                        trie
                    },
                    |trie| {
                        let stats = trie.stats();
                        black_box((stats.memory_usage, stats.bits_per_key, stats.num_states))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// PATH COMPRESSION EFFECTIVENESS BENCHMARKS
// =============================================================================

fn bench_path_compression_effectiveness(c: &mut Criterion) {
    let highly_compressible = generate_prefix_heavy_keys(5000);
    let poorly_compressible = generate_sparse_keys(5000);
    
    let mut group = c.benchmark_group("path_compression");
    group.throughput(Throughput::Elements(5000));
    
    // Highly compressible data
    group.bench_function("highly_compressible_insert", |b| {
        b.iter_batched(
            || PatriciaTrie::new(),
            |mut trie| {
                for key in &highly_compressible {
                    black_box(trie.insert(key).unwrap());
                }
                let stats = trie.stats();
                black_box((stats.memory_usage, stats.num_states, stats.num_keys))
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("highly_compressible_lookup", |b| {
        let mut trie = PatriciaTrie::new();
        for key in &highly_compressible {
            trie.insert(key).unwrap();
        }
        
        b.iter(|| {
            for key in &highly_compressible {
                black_box(trie.contains(key));
            }
        })
    });
    
    // Poorly compressible data
    group.bench_function("poorly_compressible_insert", |b| {
        b.iter_batched(
            || PatriciaTrie::new(),
            |mut trie| {
                for key in &poorly_compressible {
                    black_box(trie.insert(key).unwrap());
                }
                let stats = trie.stats();
                black_box((stats.memory_usage, stats.num_states, stats.num_keys))
            },
            BatchSize::SmallInput,
        )
    });
    
    group.bench_function("poorly_compressible_lookup", |b| {
        let mut trie = PatriciaTrie::new();
        for key in &poorly_compressible {
            trie.insert(key).unwrap();
        }
        
        b.iter(|| {
            for key in &poorly_compressible {
                black_box(trie.contains(key));
            }
        })
    });
    
    group.finish();
}

// =============================================================================
// SCALABILITY BENCHMARKS
// =============================================================================

fn bench_scalability(c: &mut Criterion) {
    let sizes = vec![100, 500, 1000, 5000, 10000, 25000];
    
    let mut group = c.benchmark_group("scalability");
    
    for size in sizes {
        let keys = generate_prefix_heavy_keys(size);
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("patricia_insert", size),
            &keys,
            |b, keys| {
                b.iter_batched(
                    || PatriciaTrie::new(),
                    |mut trie| {
                        for key in keys {
                            black_box(trie.insert(key).unwrap());
                        }
                        trie
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("patricia_lookup", size),
            &keys,
            |b, keys| {
                let mut trie = PatriciaTrie::new();
                for key in keys {
                    trie.insert(key).unwrap();
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(trie.contains(key));
                    }
                })
            },
        );
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    name = patricia_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(20);
    targets = 
        bench_patricia_configurations,
        bench_patricia_concurrency,
        bench_insertion_performance,
        bench_lookup_performance,
        bench_prefix_iteration,
        bench_memory_usage,
        bench_path_compression_effectiveness,
        bench_scalability
);

criterion_main!(patricia_benches);