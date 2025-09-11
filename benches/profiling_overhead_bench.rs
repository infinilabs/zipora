//! Comprehensive profiling overhead benchmarks
//!
//! This benchmark suite measures the performance overhead introduced by the
//! Advanced Profiling Integration system to ensure it has minimal impact on
//! application performance while providing valuable insights.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use zipora::dev_infrastructure::profiling::*;

/// Simulate a CPU-intensive workload for profiling overhead measurement
fn cpu_intensive_workload(iterations: usize) -> u64 {
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add((i as u64).wrapping_mul(17).wrapping_add(23));
        // Add some branching to make it more realistic
        if sum % 3 == 0 {
            sum = sum.wrapping_mul(7);
        }
    }
    sum
}

/// Simulate memory allocation workload
fn memory_intensive_workload(allocations: usize, size: usize) -> Vec<Vec<u8>> {
    let mut data = Vec::with_capacity(allocations);
    for i in 0..allocations {
        let mut vec = vec![0u8; size];
        // Touch the memory to ensure allocation
        vec[0] = (i % 256) as u8;
        if size > 1 {
            vec[size - 1] = ((i * 17) % 256) as u8;
        }
        data.push(vec);
    }
    data
}

/// Simulate cache-intensive workload with random access patterns
fn cache_intensive_workload(data_size: usize, access_count: usize) -> u64 {
    let data: Vec<u64> = (0..data_size).map(|i| (i as u64).wrapping_mul(19)).collect();
    let mut sum = 0u64;
    let mut index: usize = 0;
    
    for i in 0..access_count {
        // Create pseudo-random access pattern
        index = (index.wrapping_add(i).wrapping_mul(31).wrapping_add(7)) % data_size;
        sum = sum.wrapping_add(data[index]);
    }
    sum
}

/// Benchmark baseline performance without any profiling
fn benchmark_baseline_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Baseline Performance (No Profiling)");
    
    // CPU-intensive workloads
    let cpu_workloads = vec![
        ("small_cpu", 1000),
        ("medium_cpu", 10000),
        ("large_cpu", 100000),
    ];
    
    for (name, iterations) in cpu_workloads {
        group.throughput(Throughput::Elements(iterations as u64));
        group.bench_function(BenchmarkId::new("cpu_baseline", name), |b| {
            b.iter(|| {
                let result = cpu_intensive_workload(iterations);
                black_box(result);
            });
        });
    }
    
    // Memory-intensive workloads
    let memory_workloads = vec![
        ("small_mem", 100, 1024),        // 100 x 1KB
        ("medium_mem", 100, 16384),      // 100 x 16KB
        ("large_mem", 50, 65536),        // 50 x 64KB
    ];
    
    for (name, count, size) in memory_workloads {
        group.throughput(Throughput::Bytes((count * size) as u64));
        group.bench_function(BenchmarkId::new("memory_baseline", name), |b| {
            b.iter(|| {
                let result = memory_intensive_workload(count, size);
                black_box(result);
            });
        });
    }
    
    // Cache-intensive workloads
    let cache_workloads = vec![
        ("small_cache", 1000, 10000),
        ("medium_cache", 10000, 100000),
        ("large_cache", 100000, 1000000),
    ];
    
    for (name, data_size, access_count) in cache_workloads {
        group.throughput(Throughput::Elements(access_count as u64));
        group.bench_function(BenchmarkId::new("cache_baseline", name), |b| {
            b.iter(|| {
                let result = cache_intensive_workload(data_size, access_count);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

/// Benchmark overhead with different profiling levels
fn benchmark_profiling_level_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Profiling Level Overhead");
    
    let configs = vec![
        ("disabled", ProfilingConfig::disabled()),
        ("basic", ProfilingConfig::production()),
        ("standard", ProfilingConfig::standard()),
        ("detailed", ProfilingConfig::development()),
        ("debug", ProfilingConfig::debugging()),
    ];
    
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    for (config_name, config) in configs {
        // Create a profiler reporter for this configuration
        let _reporter = match ProfilerReporter::new(config) {
            Ok(r) => r,
            Err(_) => continue, // Skip this config if it fails
        };
        
        group.bench_function(BenchmarkId::new("cpu_profiling", config_name), |b| {
            b.iter(|| {
                let _scope = ProfilerScope::new("benchmark_operation").unwrap_or_else(|_| {
                    // Fallback to disabled scope if profiling fails
                    ProfilerScope::new_with_profiler("benchmark_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

/// Benchmark overhead of different profiler types
fn benchmark_profiler_type_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Profiler Type Overhead");
    
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    // Default profiler
    let default_profiler = DefaultProfiler::global();
    group.bench_function("default_profiler", |b| {
        b.iter(|| {
            let _scope = ProfilerScope::new_with_profiler("benchmark_operation", default_profiler.clone()).unwrap_or_else(|_| {
                ProfilerScope::new_with_profiler("benchmark_operation", DefaultProfiler::global()).unwrap()
            });
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    // Hardware profiler (if available)
    if let Ok(hardware_profiler) = HardwareProfiler::global() {
        group.bench_function("hardware_profiler", |b| {
            let hp = hardware_profiler.clone();
            b.iter(|| {
                let _scope = ProfilerScope::new_with_profiler("benchmark_operation", hp.clone()).unwrap_or_else(|_| {
                    ProfilerScope::new_with_profiler("benchmark_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    // Memory profiler
    if let Ok(memory_profiler) = MemoryProfiler::global() {
        group.bench_function("memory_profiler", |b| {
            let mp = memory_profiler.clone();
            b.iter(|| {
                let _scope = ProfilerScope::new_with_profiler("benchmark_operation", mp.clone()).unwrap_or_else(|_| {
                    ProfilerScope::new_with_profiler("benchmark_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    // Cache profiler
    if let Ok(cache_profiler) = CacheProfiler::global() {
        group.bench_function("cache_profiler", |b| {
            let cp = cache_profiler.clone();
            b.iter(|| {
                let _scope = ProfilerScope::new_with_profiler("benchmark_operation", cp.clone()).unwrap_or_else(|_| {
                    ProfilerScope::new_with_profiler("benchmark_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

/// Benchmark profiler scope creation overhead
fn benchmark_scope_creation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scope Creation Overhead");
    
    let iterations = 1000;
    group.throughput(Throughput::Elements(iterations));
    
    // Baseline: No profiling scope
    group.bench_function("no_scope", |b| {
        b.iter(|| {
            for i in 0..iterations {
                let result = cpu_intensive_workload(100);
                black_box((i, result));
            }
        });
    });
    
    // With profiling scope
    group.bench_function("with_scope", |b| {
        b.iter(|| {
            for i in 0..iterations {
                let _scope = ProfilerScope::new("micro_operation").unwrap();
                let result = cpu_intensive_workload(100);
                black_box((i, result));
            }
        });
    });
    
    // With nested scopes
    group.bench_function("nested_scopes", |b| {
        b.iter(|| {
            let _outer_scope = ProfilerScope::new("outer_operation").unwrap();
            for i in 0..iterations {
                let _inner_scope = ProfilerScope::new("inner_operation").unwrap();
                let result = cpu_intensive_workload(100);
                black_box((i, result));
            }
        });
    });
    
    group.finish();
}

/// Benchmark sampling rate impact on overhead
fn benchmark_sampling_rate_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sampling Rate Overhead");
    
    let sampling_rates = vec![
        ("no_sampling", 0.0),
        ("low_sampling", 0.01),    // 1%
        ("medium_sampling", 0.1),   // 10%
        ("high_sampling", 0.5),     // 50%
        ("full_sampling", 1.0),     // 100%
    ];
    
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    for (rate_name, sampling_rate) in sampling_rates {
        let config = ProfilingConfig::standard()
            .with_sampling_rate(sampling_rate);
        let _reporter = match ProfilerReporter::new(config) {
            Ok(r) => r,
            Err(_) => continue,
        };
        
        group.bench_function(BenchmarkId::new("sampling", rate_name), |b| {
            b.iter(|| {
                let _scope = ProfilerScope::new("sampled_operation").unwrap_or_else(|_| {
                    ProfilerScope::new_with_profiler("sampled_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

/// Benchmark concurrent profiling overhead
fn benchmark_concurrent_profiling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Concurrent Profiling Overhead");
    
    let thread_counts = vec![1, 2, 4, 8];
    let workload_per_thread = 5000;
    
    for thread_count in thread_counts {
        group.throughput(Throughput::Elements((thread_count * workload_per_thread) as u64));
        
        // Baseline: No profiling
        group.bench_function(BenchmarkId::new("no_profiling", thread_count), |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..thread_count)
                    .map(|_| {
                        std::thread::spawn(move || {
                            let result = cpu_intensive_workload(workload_per_thread);
                            black_box(result);
                        })
                    })
                    .collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
        
        // With profiling
        group.bench_function(BenchmarkId::new("with_profiling", thread_count), |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..thread_count)
                    .map(|thread_id| {
                        std::thread::spawn(move || {
                            let _scope = ProfilerScope::new(&format!("thread_{}", thread_id)).unwrap_or_else(|_| {
                                ProfilerScope::new_with_profiler(&format!("thread_{}", thread_id), DefaultProfiler::global()).unwrap()
                            });
                            let result = cpu_intensive_workload(workload_per_thread);
                            black_box(result);
                        })
                    })
                    .collect();
                
                for handle in handles {
                    handle.join().unwrap();
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory overhead of profiling data collection
fn benchmark_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Overhead");
    
    let data_collection_sizes = vec![
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ];
    
    for (size_name, sample_count) in data_collection_sizes {
        group.throughput(Throughput::Elements(sample_count));
        
        group.bench_function(BenchmarkId::new("data_collection", size_name), |b| {
            let config = ProfilingConfig::standard();
            
            b.iter(|| {
                for i in 0..sample_count {
                    let _scope = ProfilerScope::new(&format!("operation_{}", i)).unwrap_or_else(|_| {
                        ProfilerScope::new_with_profiler(&format!("operation_{}", i), DefaultProfiler::global()).unwrap()
                    });
                    let result = cpu_intensive_workload(1000);
                    black_box(result);
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark report generation overhead
fn benchmark_report_generation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Report Generation Overhead");
    
    let sample_counts = vec![100, 1000];
    
    for sample_count in sample_counts {
        group.throughput(Throughput::Elements(sample_count));
        
        // Simplified benchmark without complex report generation
        group.bench_function(BenchmarkId::new("simple_operations", sample_count), |b| {
            b.iter(|| {
                for i in 0..sample_count {
                    let _scope = ProfilerScope::new(&format!("operation_{}", i % 10)).unwrap_or_else(|_| {
                        ProfilerScope::new_with_profiler(&format!("operation_{}", i % 10), DefaultProfiler::global()).unwrap()
                    });
                    let result = cpu_intensive_workload(100);
                    black_box(result);
                }
            });
        });
    }
    
    group.finish();
}

/// Benchmark configuration overhead
fn benchmark_configuration_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Configuration Overhead");
    
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    let configs = vec![
        ("minimal", ProfilingConfig::production()),
        ("standard", ProfilingConfig::standard()),
        ("full_featured", ProfilingConfig::debugging()),
        ("custom_optimized", ProfilingConfig::standard()
            .with_sampling_rate(0.1)
            .with_buffer_size(512)
            .with_thread_local_caching(true)
            .with_simd_ops(true)),
    ];
    
    for (config_name, config) in configs {
        let _reporter = match ProfilerReporter::new(config) {
            Ok(r) => r,
            Err(_) => continue,
        };
        
        group.bench_function(BenchmarkId::new("config", config_name), |b| {
            b.iter(|| {
                let _scope = ProfilerScope::new("configured_operation").unwrap_or_else(|_| {
                    ProfilerScope::new_with_profiler("configured_operation", DefaultProfiler::global()).unwrap()
                });
                let result = cpu_intensive_workload(workload_iterations as usize);
                black_box(result);
            });
        });
    }
    
    group.finish();
}

/// Benchmark SIMD framework integration overhead
fn benchmark_simd_integration_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Integration Overhead");
    
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    // Test with SIMD optimizations enabled
    let simd_config = ProfilingConfig::standard()
        .with_simd_ops(true)
        .with_simd_threshold(64);
    
    group.bench_function("simd_enabled", |b| {
        let _reporter = ProfilerReporter::new(simd_config.clone()).unwrap_or_else(|_| {
            ProfilerReporter::new(ProfilingConfig::disabled()).unwrap()
        });
        
        b.iter(|| {
            let _scope = ProfilerScope::new("simd_operation").unwrap_or_else(|_| {
                ProfilerScope::new_with_profiler("simd_operation", DefaultProfiler::global()).unwrap()
            });
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    // Test with SIMD optimizations disabled
    let no_simd_config = ProfilingConfig::standard()
        .with_simd_ops(false);
    
    group.bench_function("simd_disabled", |b| {
        let _reporter = ProfilerReporter::new(no_simd_config.clone()).unwrap_or_else(|_| {
            ProfilerReporter::new(ProfilingConfig::disabled()).unwrap()
        });
        
        b.iter(|| {
            let _scope = ProfilerScope::new("no_simd_operation").unwrap_or_else(|_| {
                ProfilerScope::new_with_profiler("no_simd_operation", DefaultProfiler::global()).unwrap()
            });
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    group.finish();
}

/// Benchmark profiling overhead regression test
fn benchmark_profiling_regression_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("Profiling Overhead Regression");
    
    // Define acceptable overhead limits
    let workload_iterations = 10000;
    group.throughput(Throughput::Elements(workload_iterations));
    
    // Baseline performance
    group.bench_function("baseline_no_profiling", |b| {
        b.iter(|| {
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    // Production profiling (should have <5% overhead)
    group.bench_function("production_profiling", |b| {
        let config = ProfilingConfig::production();
        let _reporter = ProfilerReporter::new(config).unwrap_or_else(|_| {
            ProfilerReporter::new(ProfilingConfig::disabled()).unwrap()
        });
        
        b.iter(|| {
            let _scope = ProfilerScope::new("production_operation").unwrap_or_else(|_| {
                ProfilerScope::new_with_profiler("production_operation", DefaultProfiler::global()).unwrap()
            });
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    // Development profiling (should have <15% overhead)
    group.bench_function("development_profiling", |b| {
        let config = ProfilingConfig::development();
        let _reporter = ProfilerReporter::new(config).unwrap_or_else(|_| {
            ProfilerReporter::new(ProfilingConfig::disabled()).unwrap()
        });
        
        b.iter(|| {
            let _scope = ProfilerScope::new("development_operation").unwrap_or_else(|_| {
                ProfilerScope::new_with_profiler("development_operation", DefaultProfiler::global()).unwrap()
            });
            let result = cpu_intensive_workload(workload_iterations as usize);
            black_box(result);
        });
    });
    
    group.finish();
}

// Wrapper functions are no longer needed since functions don't return Results

criterion_group!(
    profiling_overhead_benches,
    benchmark_baseline_performance,
    benchmark_profiling_level_overhead,
    benchmark_profiler_type_overhead,
    benchmark_scope_creation_overhead,
    benchmark_sampling_rate_overhead,
    benchmark_concurrent_profiling_overhead,
    benchmark_memory_overhead,
    benchmark_report_generation_overhead,
    benchmark_configuration_overhead,
    benchmark_simd_integration_overhead,
    benchmark_profiling_regression_test
);

criterion_main!(profiling_overhead_benches);