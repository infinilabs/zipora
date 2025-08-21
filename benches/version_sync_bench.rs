use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use zipora::fsa::version_sync::{ConcurrencyLevel, VersionManager};
use zipora::fsa::token::{TokenManager, with_reader_token, with_writer_token};

fn version_manager_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("version_manager");

    // Benchmark single-threaded token acquisition
    group.bench_function("single_thread_reader_token", |b| {
        let manager = VersionManager::new(ConcurrencyLevel::SingleThreadStrict);
        b.iter(|| {
            let token = black_box(manager.acquire_reader_token().unwrap());
            drop(token);
        });
    });

    group.bench_function("single_thread_writer_token", |b| {
        let manager = VersionManager::new(ConcurrencyLevel::SingleThreadStrict);
        b.iter(|| {
            let token = black_box(manager.acquire_writer_token().unwrap());
            drop(token);
        });
    });

    // Benchmark multi-threaded token acquisition
    group.bench_function("multi_thread_reader_tokens", |b| {
        let manager = Arc::new(VersionManager::new(ConcurrencyLevel::OneWriteMultiRead));
        b.iter(|| {
            let manager_clone = Arc::clone(&manager);
            let handle = thread::spawn(move || {
                let token = manager_clone.acquire_reader_token().unwrap();
                drop(token);
            });
            handle.join().unwrap();
        });
    });

    group.finish();
}

fn token_manager_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_manager");

    // Benchmark token caching performance
    group.bench_function("cached_reader_token", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        
        // Prime the cache
        let token = manager.acquire_reader_token().unwrap();
        manager.return_reader_token(token);

        b.iter(|| {
            let token = black_box(manager.acquire_reader_token().unwrap());
            manager.return_reader_token(token);
        });
    });

    group.bench_function("cached_writer_token", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        
        // Prime the cache
        let token = manager.acquire_writer_token().unwrap();
        manager.return_writer_token(token);

        b.iter(|| {
            let token = black_box(manager.acquire_writer_token().unwrap());
            manager.return_writer_token(token);
        });
    });

    // Benchmark convenience functions
    group.bench_function("with_reader_token", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        b.iter(|| {
            with_reader_token(&manager, |token| {
                black_box(token.is_valid());
                Ok(())
            }).unwrap();
        });
    });

    group.bench_function("with_writer_token", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        b.iter(|| {
            with_writer_token(&manager, |token| {
                black_box(token.is_valid());
                Ok(())
            }).unwrap();
        });
    });

    group.finish();
}

fn concurrency_level_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrency_levels");

    // Benchmark different concurrency levels
    let levels = vec![
        ConcurrencyLevel::NoWriteReadOnly,
        ConcurrencyLevel::SingleThreadStrict,
        ConcurrencyLevel::SingleThreadShared,
        ConcurrencyLevel::OneWriteMultiRead,
        ConcurrencyLevel::MultiWriteMultiRead,
    ];

    for level in levels {
        group.bench_with_input(
            BenchmarkId::new("reader_token_acquisition", format!("{:?}", level)),
            &level,
            |b, &level| {
                let manager = TokenManager::new(level);
                b.iter(|| {
                    if level == ConcurrencyLevel::NoWriteReadOnly {
                        // Special case for read-only
                        let token = black_box(manager.acquire_reader_token().unwrap());
                        drop(token);
                    } else {
                        let token = black_box(manager.acquire_reader_token().unwrap());
                        manager.return_reader_token(token);
                    }
                });
            },
        );
    }

    group.finish();
}

fn threading_overhead_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("threading_overhead");
    group.throughput(Throughput::Elements(1000));

    // Benchmark single-threaded vs multi-threaded overhead
    group.bench_function("single_threaded_operations", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::SingleThreadStrict);
        b.iter(|| {
            for _ in 0..1000 {
                with_reader_token(&manager, |_| Ok(())).unwrap();
            }
        });
    });

    group.bench_function("multi_threaded_operations", |b| {
        let manager = Arc::new(TokenManager::new(ConcurrencyLevel::OneWriteMultiRead));
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let manager_clone = Arc::clone(&manager);
                    thread::spawn(move || {
                        for _ in 0..250 {
                            with_reader_token(&manager_clone, |_| Ok(())).unwrap();
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

fn cache_performance_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");

    // Benchmark cache hit vs cache miss performance
    group.bench_function("cache_hit_reader", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        
        // Prime the cache
        let token = manager.acquire_reader_token().unwrap();
        manager.return_reader_token(token);

        b.iter(|| {
            // This should be a cache hit
            let token = manager.acquire_reader_token().unwrap();
            manager.return_reader_token(token);
        });
    });

    group.bench_function("cache_miss_reader", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::OneWriteMultiRead);
        b.iter(|| {
            // Clear cache to force miss
            manager.clear_thread_cache();
            let token = manager.acquire_reader_token().unwrap();
            manager.return_reader_token(token);
        });
    });

    group.finish();
}

fn memory_overhead_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_overhead");

    // Benchmark memory usage of different components
    group.bench_function("version_manager_creation", |b| {
        b.iter(|| {
            let manager = black_box(VersionManager::new(ConcurrencyLevel::OneWriteMultiRead));
            drop(manager);
        });
    });

    group.bench_function("token_manager_creation", |b| {
        b.iter(|| {
            let manager = black_box(TokenManager::new(ConcurrencyLevel::OneWriteMultiRead));
            drop(manager);
        });
    });

    // Test with many concurrent tokens
    group.bench_function("many_concurrent_tokens", |b| {
        let manager = TokenManager::new(ConcurrencyLevel::MultiWriteMultiRead);
        b.iter(|| {
            let mut tokens = Vec::new();
            for _ in 0..100 {
                tokens.push(manager.acquire_reader_token().unwrap());
            }
            for token in tokens {
                manager.return_reader_token(token);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    version_manager_benchmarks,
    token_manager_benchmarks,
    concurrency_level_benchmarks,
    threading_overhead_benchmarks,
    cache_performance_benchmarks,
    memory_overhead_benchmarks
);
criterion_main!(benches);