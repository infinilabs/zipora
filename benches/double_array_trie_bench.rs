//! Benchmarks for Double Array Trie implementation
//!
//! This benchmark suite compares the Double Array Trie performance against:
//! - HashMap (std::collections::HashMap)
//! - BTreeMap (std::collections::BTreeMap)  
//! - Existing trie implementations (LOUDS, CritBit)
//!
//! Performance targets:
//! - 2-3x faster than HashMap for dense key sets
//! - Comparable or better memory efficiency
//! - Constant-time state transitions

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use zipora::fsa::{
    CritBitTrie, DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieConfig,
    LoudsTrie, Trie, FiniteStateAutomaton,
};

// Benchmark data generators
fn generate_dense_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| format!("key_{:06}", i).into_bytes())
        .collect()
}

fn generate_sparse_keys(count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .step_by(37) // Sparse pattern
        .map(|i| format!("sparse_key_{:08}", i).into_bytes())
        .collect()
}

fn generate_prefixed_keys(count: usize) -> Vec<Vec<u8>> {
    let prefixes = ["app", "application", "apply", "banana", "band", "cat", "dog"];
    let mut keys = Vec::new();
    
    for i in 0..count {
        let prefix = prefixes[i % prefixes.len()];
        let key = format!("{}_{:04}", prefix, i);
        keys.push(key.into_bytes());
    }
    
    keys.sort();
    keys.dedup();
    keys
}

fn generate_realistic_keys(count: usize) -> Vec<Vec<u8>> {
    // Simulate realistic string patterns (URLs, file paths, etc.)
    let patterns = [
        "/api/v1/users",
        "/api/v1/posts", 
        "/static/js/app",
        "/static/css/style",
        "/images/thumbnails",
        "/data/cache",
        "com.example.package",
        "org.apache.commons",
    ];
    
    let mut keys = Vec::new();
    for i in 0..count {
        let pattern = patterns[i % patterns.len()];
        let key = format!("{}/{:04}", pattern, i);
        keys.push(key.into_bytes());
    }
    
    keys.sort();
    keys.dedup();
    keys
}

fn generate_unicode_keys(count: usize) -> Vec<Vec<u8>> {
    let patterns = ["hello", "世界", "🌍", "café", "москва", "東京"];
    let mut keys = Vec::new();
    
    for i in 0..count {
        let pattern = patterns[i % patterns.len()];
        let key = format!("{}_{:04}", pattern, i);
        keys.push(key.into_bytes());
    }
    
    keys.sort();
    keys.dedup();
    keys
}

// Construction benchmarks
fn bench_construction_double_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_double_array");
    
    for size in [100, 1000, 10000].iter() {
        let keys = generate_dense_keys(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            format!("incremental_{}", size),
            size,
            |b, &_size| {
                b.iter_batched(
                    || keys.clone(),
                    |keys| {
                        let mut trie = DoubleArrayTrie::new();
                        for key in keys {
                            black_box(trie.insert(&key).unwrap());
                        }
                        trie
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        group.bench_with_input(
            format!("builder_sorted_{}", size),
            size,
            |b, &_size| {
                b.iter_batched(
                    || keys.clone(),
                    |keys| {
                        black_box(
                            DoubleArrayTrieBuilder::new()
                                .build_from_sorted(keys)
                                .unwrap()
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
        
        group.bench_with_input(
            format!("builder_unsorted_{}", size),
            size,
            |b, &_size| {
                b.iter_batched(
                    || {
                        let mut unsorted = keys.clone();
                        unsorted.reverse();
                        unsorted
                    },
                    |keys| {
                        black_box(
                            DoubleArrayTrieBuilder::new()
                                .build_from_unsorted(keys)
                                .unwrap()
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

fn bench_construction_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_comparison");
    group.measurement_time(Duration::from_secs(10));
    
    let size = 1000;
    let keys = generate_dense_keys(size);
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Double Array Trie
    group.bench_function("double_array_trie", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut trie = DoubleArrayTrie::new();
                for key in keys {
                    black_box(trie.insert(&key).unwrap());
                }
                trie
            },
            BatchSize::SmallInput,
        )
    });
    
    // LOUDS Trie
    group.bench_function("louds_trie", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut trie = LoudsTrie::new();
                for key in keys {
                    black_box(trie.insert(&key).unwrap());
                }
                trie
            },
            BatchSize::SmallInput,
        )
    });
    
    // CritBit Trie
    group.bench_function("critbit_trie", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut trie = CritBitTrie::new();
                for key in keys {
                    black_box(trie.insert(&key).unwrap());
                }
                trie
            },
            BatchSize::SmallInput,
        )
    });
    
    // HashMap
    group.bench_function("hashmap", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut map = HashMap::new();
                for (i, key) in keys.into_iter().enumerate() {
                    black_box(map.insert(key, i));
                }
                map
            },
            BatchSize::SmallInput,
        )
    });
    
    // BTreeMap
    group.bench_function("btreemap", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut map = BTreeMap::new();
                for (i, key) in keys.into_iter().enumerate() {
                    black_box(map.insert(key, i));
                }
                map
            },
            BatchSize::SmallInput,
        )
    });
    
    group.finish();
}

// Lookup benchmarks
fn bench_lookup_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_performance");
    
    for size in [100, 1000, 10000].iter() {
        let keys = generate_dense_keys(*size);
        
        // Build structures once
        let mut double_array = DoubleArrayTrie::new();
        let mut louds = LoudsTrie::new();
        let mut critbit = CritBitTrie::new();
        let mut hashmap = HashMap::new();
        let mut btreemap = BTreeMap::new();
        
        for (i, key) in keys.iter().enumerate() {
            double_array.insert(key).unwrap();
            louds.insert(key).unwrap();
            critbit.insert(key).unwrap();
            hashmap.insert(key.clone(), i);
            btreemap.insert(key.clone(), i);
        }
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // Double Array Trie lookup
        group.bench_with_input(
            format!("double_array_{}", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        black_box(double_array.contains(key));
                    }
                })
            },
        );
        
        // LOUDS Trie lookup
        group.bench_with_input(
            format!("louds_{}", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        black_box(louds.contains(key));
                    }
                })
            },
        );
        
        // CritBit Trie lookup
        group.bench_with_input(
            format!("critbit_{}", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        black_box(critbit.contains(key));
                    }
                })
            },
        );
        
        // HashMap lookup
        group.bench_with_input(
            format!("hashmap_{}", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        black_box(hashmap.contains_key(key));
                    }
                })
            },
        );
        
        // BTreeMap lookup
        group.bench_with_input(
            format!("btreemap_{}", size),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        black_box(btreemap.contains_key(key));
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_prefix_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_operations");
    
    let keys = generate_prefixed_keys(1000);
    let mut trie = DoubleArrayTrie::new();
    
    for key in &keys {
        trie.insert(key).unwrap();
    }
    
    let prefixes = [b"app".as_slice(), b"application".as_slice(), b"banana".as_slice()];
    
    group.bench_function("prefix_iteration", |b| {
        b.iter(|| {
            for prefix in &prefixes {
                let results: Vec<_> = trie.iter_prefix(prefix).collect();
                black_box(results);
            }
        })
    });
    
    group.bench_function("longest_prefix", |b| {
        let test_strings = [
            b"application_extended".as_slice(),
            b"app_test".as_slice(),
            b"banana_split".as_slice(),
            b"unknown".as_slice(),
        ];
        
        b.iter(|| {
            for test_str in &test_strings {
                black_box(trie.longest_prefix(test_str));
            }
        })
    });
    
    group.finish();
}

fn bench_state_transitions(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_transitions");
    
    let keys = generate_dense_keys(1000);
    let mut trie = DoubleArrayTrie::new();
    
    for key in &keys {
        trie.insert(key).unwrap();
    }
    
    group.bench_function("manual_transitions", |b| {
        b.iter(|| {
            for key in &keys {
                let mut state = trie.root();
                for &symbol in key {
                    if let Some(next_state) = trie.transition(state, symbol) {
                        state = next_state;
                        black_box(state);
                    } else {
                        break;
                    }
                }
                black_box(trie.is_final(state));
            }
        })
    });
    
    group.bench_function("accepts_method", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(trie.accepts(key));
            }
        })
    });
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10); // Fewer samples for memory tests
    
    let sizes = [100, 1000, 5000];
    
    for &size in &sizes {
        let keys = generate_dense_keys(size);
        
        group.bench_with_input(
            format!("memory_usage_{}", size),
            &size,
            |b, &_size| {
                b.iter_batched(
                    || keys.clone(),
                    |keys| {
                        let mut trie = DoubleArrayTrie::new();
                        for key in keys {
                            trie.insert(&key).unwrap();
                        }
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

fn bench_simd_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_optimization");
    
    // Create long keys that will benefit from SIMD processing
    let long_keys: Vec<Vec<u8>> = (0..100)
        .map(|i| format!("very_long_key_for_simd_processing_benchmark_{:06}", i).into_bytes())
        .collect();
    
    // SIMD enabled configuration
    let simd_config = DoubleArrayTrieConfig {
        enable_simd: true,
        ..Default::default()
    };
    
    // SIMD disabled configuration
    let no_simd_config = DoubleArrayTrieConfig {
        enable_simd: false,
        ..Default::default()
    };
    
    let mut simd_trie = DoubleArrayTrie::with_config(simd_config);
    let mut no_simd_trie = DoubleArrayTrie::with_config(no_simd_config);
    
    for key in &long_keys {
        simd_trie.insert(key).unwrap();
        no_simd_trie.insert(key).unwrap();
    }
    
    group.bench_function("simd_enabled", |b| {
        b.iter(|| {
            for key in &long_keys {
                black_box(simd_trie.contains(key));
            }
        })
    });
    
    group.bench_function("simd_disabled", |b| {
        b.iter(|| {
            for key in &long_keys {
                black_box(no_simd_trie.contains(key));
            }
        })
    });
    
    group.finish();
}

fn bench_realistic_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("realistic_workloads");
    
    // URL routing simulation
    let url_keys = generate_realistic_keys(500);
    let mut url_trie = DoubleArrayTrie::new();
    let mut url_hashmap = HashMap::new();
    
    for (i, key) in url_keys.iter().enumerate() {
        url_trie.insert(key).unwrap();
        url_hashmap.insert(key.clone(), i);
    }
    
    group.bench_function("url_routing_trie", |b| {
        b.iter(|| {
            for key in &url_keys {
                black_box(url_trie.contains(key));
            }
        })
    });
    
    group.bench_function("url_routing_hashmap", |b| {
        b.iter(|| {
            for key in &url_keys {
                black_box(url_hashmap.contains_key(key));
            }
        })
    });
    
    // Dictionary/autocomplete simulation
    let dict_keys = generate_prefixed_keys(1000);
    let mut dict_trie = DoubleArrayTrie::new();
    
    for key in &dict_keys {
        dict_trie.insert(key).unwrap();
    }
    
    group.bench_function("autocomplete_prefix_search", |b| {
        let prefixes = [b"app".as_slice(), b"ban".as_slice(), b"cat".as_slice()];
        b.iter(|| {
            for prefix in &prefixes {
                let results: Vec<_> = dict_trie.iter_prefix(prefix).take(10).collect();
                black_box(results);
            }
        })
    });
    
    // Unicode text processing
    let unicode_keys = generate_unicode_keys(500);
    let mut unicode_trie = DoubleArrayTrie::new();
    
    for key in &unicode_keys {
        unicode_trie.insert(key).unwrap();
    }
    
    group.bench_function("unicode_processing", |b| {
        b.iter(|| {
            for key in &unicode_keys {
                black_box(unicode_trie.contains(key));
            }
        })
    });
    
    group.finish();
}

fn bench_scaling_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_performance");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);
    
    for &size in &[1000, 5000, 10000, 25000] {
        let keys = generate_dense_keys(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            format!("lookup_scaling_{}", size),
            &keys,
            |b, keys| {
                let mut trie = DoubleArrayTrie::new();
                for key in keys {
                    trie.insert(key).unwrap();
                }
                
                b.iter(|| {
                    // Test random subset for large datasets
                    let step = std::cmp::max(1, keys.len() / 100);
                    for key in keys.iter().step_by(step) {
                        black_box(trie.contains(key));
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_config_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_variants");
    
    let keys = generate_dense_keys(1000);
    
    let configs = [
        ("default", DoubleArrayTrieConfig::default()),
        ("large_capacity", DoubleArrayTrieConfig {
            initial_capacity: 4096,
            ..Default::default()
        }),
        ("no_memory_pool", DoubleArrayTrieConfig {
            use_memory_pool: false,
            ..Default::default()
        }),
        ("fast_growth", DoubleArrayTrieConfig {
            growth_factor: 2.0,
            ..Default::default()
        }),
    ];
    
    for (name, config) in &configs {
        group.bench_with_input(
            format!("construction_{}", name),
            config,
            |b, config| {
                b.iter_batched(
                    || keys.clone(),
                    |keys| {
                        let mut trie = DoubleArrayTrie::with_config(config.clone());
                        for key in keys {
                            black_box(trie.insert(&key).unwrap());
                        }
                        trie
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    construction_benches,
    bench_construction_double_array,
    bench_construction_comparison,
    bench_config_variants
);

criterion_group!(
    lookup_benches,
    bench_lookup_performance,
    bench_state_transitions,
    bench_scaling_performance
);

criterion_group!(
    feature_benches,
    bench_prefix_operations,
    bench_simd_optimization,
    bench_memory_efficiency
);

criterion_group!(
    workload_benches,
    bench_realistic_workloads
);

criterion_main!(
    construction_benches,
    lookup_benches,
    feature_benches,
    workload_benches
);