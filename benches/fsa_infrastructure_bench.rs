//! Benchmark suite for FSA infrastructure components
//!
//! This benchmark suite validates the performance of the FSA infrastructure
//! components implemented in Phase 8A including caching, DAWG construction,
//! graph walking, and fast search algorithms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zipora::fsa::simple_implementations::*;
use std::time::Duration;

/// Benchmark FSA cache operations
fn bench_fsa_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("fsa_cache");
    
    for cache_size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("insert", cache_size),
            cache_size,
            |b, &size| {
                b.iter_with_setup(
                    || SimpleFsaCache::new(size),
                    |mut cache| {
                        for i in 0..1000 {
                            black_box(cache.insert(i as u32, i as u32 * 2).unwrap());
                        }
                    }
                );
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("lookup", cache_size),
            cache_size,
            |b, &size| {
                let mut cache = SimpleFsaCache::new(size);
                // Populate cache
                for i in 0..size.min(1000) {
                    cache.insert(i as u32, i as u32 * 2).unwrap();
                }
                
                b.iter(|| {
                    for i in 0..100 {
                        black_box(cache.get(i as u32));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark DAWG construction and operations
fn bench_simple_dawg(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_dawg");
    
    // Test data sets of increasing size
    let test_keys = [
        vec![b"cat".as_slice(), b"car".as_slice(), b"card".as_slice(), b"care".as_slice(), b"careful".as_slice()],
        vec![b"apple".as_slice(), b"application".as_slice(), b"apply".as_slice(), b"approach".as_slice(), b"appropriate".as_slice(), 
             b"approve".as_slice(), b"approximately".as_slice(), b"april".as_slice(), b"area".as_slice(), b"argue".as_slice()],
        generate_test_keys(100),
        generate_test_keys(1000),
    ];
    
    let sizes = [5, 10, 100, 1000];
    
    for (i, keys) in test_keys.iter().enumerate() {
        let size = sizes[i];
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("construction", size),
            keys,
            |b, keys| {
                b.iter(|| {
                    let mut dawg = SimpleDawg::new();
                    for key in keys {
                        black_box(dawg.insert(key).unwrap());
                    }
                    black_box(dawg)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("lookup", size),
            keys,
            |b, keys| {
                let mut dawg = SimpleDawg::new();
                for key in keys {
                    dawg.insert(key).unwrap();
                }
                
                b.iter(|| {
                    for key in keys {
                        black_box(dawg.contains(key));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark graph walker performance
fn bench_graph_walker(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_walker");
    
    for graph_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("bfs_traversal", graph_size),
            graph_size,
            |b, &size| {
                b.iter(|| {
                    let mut walker = SimpleGraphWalker::new();
                    
                    // Simple connected graph simulation
                    let graph_fn = |node: u32| -> zipora::error::Result<Vec<u32>> {
                        if node >= size {
                            Ok(vec![])
                        } else {
                            let neighbors = match node % 3 {
                                0 => vec![node + 1, node + 2],
                                1 => vec![node + 1],
                                _ => vec![],
                            };
                            Ok(neighbors.into_iter().filter(|&n| n < size).collect())
                        }
                    };
                    
                    black_box(walker.walk_bfs(0, graph_fn).unwrap());
                    black_box(walker.visited_count())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark fast search algorithms
fn bench_fast_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_search");
    
    // Generate test data of different sizes
    let data_sizes = [100, 1000, 10000, 100000];
    
    for &size in data_sizes.iter() {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("search_byte", size),
            &data,
            |b, data| {
                let search = SimpleFastSearch::new();
                b.iter(|| {
                    black_box(search.search_byte(data, b'a'));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("find_first", size),
            &data,
            |b, data| {
                let search = SimpleFastSearch::new();
                b.iter(|| {
                    black_box(search.find_first(data, b'a'));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("count", size),
            &data,
            |b, data| {
                let search = SimpleFastSearch::new();
                b.iter(|| {
                    black_box(search.count(data, b'a'));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("pattern_search", size),
            &data,
            |b, data| {
                let search = SimpleFastSearch::new();
                b.iter(|| {
                    black_box(search.search_pattern(data, b"test"));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark combined FSA operations (integration test)
fn bench_fsa_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("fsa_integration");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("combined_operations", |b| {
        b.iter(|| {
            // Create and populate DAWG
            let mut dawg = SimpleDawg::new();
            let keys = [b"computer".as_slice(), b"computation".as_slice(), b"compute".as_slice(), b"computing".as_slice()];
            for key in &keys {
                dawg.insert(key).unwrap();
            }
            
            // Create and use cache
            let mut cache = SimpleFsaCache::new(1000);
            for i in 0..100 {
                cache.insert(i, i * 2).unwrap();
            }
            
            // Perform searches
            let search = SimpleFastSearch::new();
            let data = b"computer science computation complexity";
            let positions = search.search_byte(data, b'c');
            
            // Graph traversal simulation
            let mut walker = SimpleGraphWalker::new();
            let graph_fn = |node: u32| -> zipora::error::Result<Vec<u32>> {
                Ok(if node < 10 { vec![node + 1] } else { vec![] })
            };
            walker.walk_bfs(0, graph_fn).unwrap();
            
            black_box((dawg.num_keys(), cache.len(), positions.len(), walker.visited_count()))
        });
    });
    
    group.finish();
}

/// Benchmark memory efficiency of FSA components
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    group.bench_function("dawg_memory_usage", |b| {
        b.iter_with_setup(
            || {
                let mut dawg = SimpleDawg::new();
                let keys = generate_test_keys(1000);
                for key in &keys {
                    dawg.insert(key).unwrap();
                }
                dawg
            },
            |dawg| {
                black_box(dawg.memory_usage());
            }
        );
    });
    
    group.bench_function("cache_memory_efficiency", |b| {
        b.iter_with_setup(
            || {
                let mut cache = SimpleFsaCache::new(10000);
                for i in 0..5000 {
                    cache.insert(i, i * 2).unwrap();
                }
                cache
            },
            |cache| {
                // Simulate memory pressure by accessing random entries
                for i in (0..1000).step_by(7) {
                    black_box(cache.get(i));
                }
            }
        );
    });
    
    group.finish();
}

/// Generate test keys for benchmarking
fn generate_test_keys(count: usize) -> Vec<&'static [u8]> {
    // Pre-computed test keys to avoid allocation overhead in benchmarks
    let base_keys: &[&[u8]] = &[
        b"computer", b"computation", b"compute", b"computing", b"compiler", 
        b"complete", b"complex", b"component", b"composite", b"compress",
        b"algorithm", b"analysis", b"application", b"architecture", b"artificial",
        b"automatic", b"available", b"background", b"bandwidth", b"benchmark",
        b"database", b"dataflow", b"debugging", b"declaration", b"decomposition",
        b"directory", b"distributed", b"documentation", b"dynamic", b"efficiency",
        b"element", b"encapsulation", b"environment", b"evaluation", b"exception",
        b"execution", b"expression", b"extension", b"framework", b"function",
        b"generation", b"graphics", b"hardware", b"hierarchy", b"implementation",
        b"information", b"inheritance", b"initialization", b"instruction", b"interface",
        b"language", b"library", b"machine", b"management", b"memory",
        b"network", b"object", b"operation", b"optimization", b"organization",
        b"parameter", b"performance", b"platform", b"pointer", b"procedure",
        b"process", b"program", b"protocol", b"quality", b"query",
        b"reference", b"register", b"representation", b"resource", b"response",
        b"security", b"software", b"structure", b"system", b"technology",
        b"template", b"thread", b"transaction", b"transformation", b"variable",
        b"vector", b"verification", b"version", b"virtual", b"visualization",
    ];
    
    let mut keys = Vec::new();
    for i in 0..count {
        keys.push(base_keys[i % base_keys.len()]);
    }
    keys
}

/// Generate test data for search benchmarks
fn generate_test_data(size: usize) -> Vec<u8> {
    let pattern = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut data = Vec::with_capacity(size);
    
    for i in 0..size {
        data.push(pattern[i % pattern.len()]);
    }
    
    // Insert some test patterns
    let test_patterns: &[&[u8]] = &[b"test", b"benchmark", b"performance", b"zipora"];
    for (i, pattern) in test_patterns.iter().enumerate() {
        let pos = (i + 1) * size / (test_patterns.len() + 1);
        if pos + pattern.len() < size {
            for (j, &byte) in pattern.iter().enumerate() {
                data[pos + j] = byte;
            }
        }
    }
    
    data
}

criterion_group!(
    fsa_infrastructure_benches,
    bench_fsa_cache,
    bench_simple_dawg,
    bench_graph_walker,
    bench_fast_search,
    bench_fsa_integration,
    bench_memory_efficiency
);

criterion_main!(fsa_infrastructure_benches);