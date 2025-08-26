//! Comprehensive benchmarks for Enhanced SAIS Suffix Array implementation
//!
//! This benchmark suite compares the enhanced suffix array implementation
//! against the baseline and measures performance across different text types,
//! sizes, and configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zipora::algorithms::suffix_array::{SuffixArray as BaseSuffixArray, SuffixArrayBuilder as BaseBuilder};
use zipora::compression::suffix_array::{SuffixArrayCompressor, SuffixArrayConfig, EnhancedSuffixArray};
use std::time::Duration;

/// Generate test data of various types for benchmarking
fn generate_test_data(size: usize, data_type: &str) -> Vec<u8> {
    match data_type {
        "random" => {
            // Random data - worst case for compression
            (0..size).map(|i| ((i * 7 + 13) % 256) as u8).collect()
        }
        "repetitive" => {
            // Highly repetitive data - best case for compression
            (0..size).map(|i| ((i / 100) % 4) as u8).collect()
        }
        "dna" => {
            // DNA-like data (4-character alphabet)
            (0..size).map(|i| match i % 4 {
                0 => b'A',
                1 => b'C',
                2 => b'G',
                _ => b'T',
            }).collect()
        }
        "text" => {
            // English-like text
            let alphabet = b"abcdefghijklmnopqrstuvwxyz ";
            (0..size).map(|i| alphabet[(i * 17 + 7) % alphabet.len()]).collect()
        }
        "sorted" => {
            // Sorted data - good for delta compression
            (0..size).map(|i| (i / 100) as u8).collect()
        }
        _ => panic!("Unknown data type: {}", data_type),
    }
}

/// Benchmark suffix array construction
fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array_construction");
    
    let sizes = vec![1_000, 10_000, 100_000];
    let data_types = vec!["random", "repetitive", "dna", "text"];
    
    for size in &sizes {
        for data_type in &data_types {
            let mut data = generate_test_data(*size, data_type);
            data.push(0); // Add sentinel
            
            group.throughput(Throughput::Bytes(data.len() as u64));
            
            // Benchmark baseline implementation
            group.bench_with_input(
                BenchmarkId::new(format!("baseline_{}", data_type), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let sa = BaseSuffixArray::new(black_box(data)).unwrap();
                        black_box(sa.as_slice().len())
                    })
                },
            );
            
            // Benchmark enhanced implementation (default config)
            group.bench_with_input(
                BenchmarkId::new(format!("enhanced_default_{}", data_type), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let compressor = SuffixArrayCompressor::default();
                        let sa = compressor.build_suffix_array(black_box(data)).unwrap();
                        black_box(sa.len())
                    })
                },
            );
            
            // Benchmark enhanced implementation (dictionary config)
            group.bench_with_input(
                BenchmarkId::new(format!("enhanced_dict_{}", data_type), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let config = SuffixArrayConfig::for_dictionary_compression();
                        let compressor = SuffixArrayCompressor::new(config).unwrap();
                        let sa = compressor.build_suffix_array(black_box(data)).unwrap();
                        black_box(sa.len())
                    })
                },
            );
            
            // Benchmark enhanced implementation (realtime config)
            group.bench_with_input(
                BenchmarkId::new(format!("enhanced_realtime_{}", data_type), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let config = SuffixArrayConfig::for_realtime();
                        let compressor = SuffixArrayCompressor::new(config).unwrap();
                        let sa = compressor.build_suffix_array(black_box(data)).unwrap();
                        black_box(sa.len())
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark pattern searching
fn bench_pattern_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_search");
    group.measurement_time(Duration::from_secs(10));
    
    // Create test data
    let size = 100_000;
    let mut text = generate_test_data(size, "text");
    text.push(0); // Add sentinel
    
    // Build suffix arrays once
    let base_sa = BaseSuffixArray::new(&text).unwrap();
    let enhanced_sa = SuffixArrayCompressor::default()
        .build_suffix_array(&text)
        .unwrap();
    
    let patterns = vec![b"abc", b"the", b"tion", b"xyz"];
    
    for pattern in &patterns {
        group.throughput(Throughput::Elements(text.len() as u64));
        
        // Benchmark baseline search
        group.bench_with_input(
            BenchmarkId::new("baseline", pattern.len()),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    let (start, count) = base_sa.search(black_box(&text), black_box(pattern));
                    black_box(start + count)
                })
            },
        );
        
        // Benchmark enhanced search
        group.bench_with_input(
            BenchmarkId::new("enhanced", pattern.len()),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    let occurrences = enhanced_sa.find_pattern(black_box(&text), black_box(pattern));
                    black_box(occurrences.len())
                })
            },
        );
        
        // Benchmark enhanced count (optimized for counting)
        group.bench_with_input(
            BenchmarkId::new("enhanced_count", pattern.len()),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    let count = enhanced_sa.count_pattern(black_box(&text), black_box(pattern));
                    black_box(count)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");
    
    let size = 50_000;
    let mut text = generate_test_data(size, "text");
    text.push(0);
    
    // Build suffix arrays
    let base_sa = BaseSuffixArray::new(&text).unwrap();
    let enhanced_sa = SuffixArrayCompressor::default()
        .build_suffix_array(&text)
        .unwrap();
    
    // Random access pattern
    let indices: Vec<usize> = (0..1000).map(|i| (i * 7 + 13) % text.len()).collect();
    
    group.bench_function("baseline_random_access", |b| {
        b.iter(|| {
            let mut sum = 0;
            for &idx in &indices {
                if let Some(suffix_pos) = base_sa.suffix_at_rank(idx) {
                    sum += suffix_pos;
                }
            }
            black_box(sum)
        })
    });
    
    group.bench_function("enhanced_random_access", |b| {
        b.iter(|| {
            let mut sum = 0;
            for &idx in &indices {
                if let Some(suffix_pos) = enhanced_sa.suffix_at_rank(idx) {
                    sum += suffix_pos;
                }
            }
            black_box(sum)
        })
    });
    
    // Sequential access pattern
    group.bench_function("baseline_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0;
            for i in 0..1000 {
                if let Some(suffix_pos) = base_sa.suffix_at_rank(i) {
                    sum += suffix_pos;
                }
            }
            black_box(sum)
        })
    });
    
    group.bench_function("enhanced_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0;
            for i in 0..1000 {
                if let Some(suffix_pos) = enhanced_sa.suffix_at_rank(i) {
                    sum += suffix_pos;
                }
            }
            black_box(sum)
        })
    });
    
    group.finish();
}

/// Benchmark memory usage and compression
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    let sizes = vec![10_000, 50_000, 100_000];
    let data_types = vec!["random", "repetitive", "sorted"];
    
    for size in &sizes {
        for data_type in &data_types {
            let mut data = generate_test_data(*size, data_type);
            data.push(0);
            
            group.bench_with_input(
                BenchmarkId::new(format!("memory_usage_{}", data_type), size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let enhanced_sa = SuffixArrayCompressor::default()
                            .build_suffix_array(black_box(data))
                            .unwrap();
                        
                        let memory_usage = enhanced_sa.memory_usage();
                        let compression_ratio = enhanced_sa.compression_ratio();
                        
                        black_box((memory_usage, compression_ratio))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark different configurations
fn bench_configurations(c: &mut Criterion) {
    let mut group = c.benchmark_group("configurations");
    
    let size = 50_000;
    let mut data = generate_test_data(size, "text");
    data.push(0);
    
    let configs = vec![
        ("default", SuffixArrayConfig::default()),
        ("dictionary", SuffixArrayConfig::for_dictionary_compression()),
        ("large_text", SuffixArrayConfig::for_large_text()),
        ("realtime", SuffixArrayConfig::for_realtime()),
    ];
    
    for (name, config) in configs {
        group.bench_with_input(
            BenchmarkId::new("config", name),
            &config,
            |b, config| {
                b.iter(|| {
                    let compressor = SuffixArrayCompressor::new(config.clone()).unwrap();
                    let sa = compressor.build_suffix_array(black_box(&data)).unwrap();
                    black_box(sa.len())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark large-scale performance
fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10); // Fewer samples for large tests
    group.measurement_time(Duration::from_secs(30));
    
    let sizes = vec![500_000, 1_000_000];
    
    for size in &sizes {
        let mut data = generate_test_data(*size, "text");
        data.push(0);
        
        group.throughput(Throughput::Bytes(data.len() as u64));
        
        group.bench_with_input(
            BenchmarkId::new("large_text_construction", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let config = SuffixArrayConfig::for_large_text();
                    let compressor = SuffixArrayCompressor::new(config).unwrap();
                    let sa = compressor.build_suffix_array(black_box(data)).unwrap();
                    black_box(sa.len())
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_pattern_search,
    bench_memory_access,
    bench_memory_efficiency,
    bench_configurations,
    bench_large_scale
);

criterion_main!(benches);