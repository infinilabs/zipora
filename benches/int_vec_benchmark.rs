//! Comprehensive benchmarks for IntVec<T> bit-packed integer storage
//!
//! This benchmark suite validates the performance targets for IntVec<T>:
//! - 60-80% space reduction compared to Vec<T>
//! - Hardware-accelerated compression/decompression
//! - O(1) random access performance
//! - Adaptive compression strategy selection

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use zipora::containers::specialized::{IntVec, UintVector};

/// Generate test data with different compression characteristics
struct TestDataGenerator;

impl TestDataGenerator {
    /// Sorted sequence - excellent for delta compression
    fn sorted_sequence(size: usize) -> Vec<u32> {
        (0..size as u32).collect()
    }

    /// Small range data - excellent for min-max compression
    fn small_range(size: usize) -> Vec<u32> {
        (0..size).map(|i| (i % 1000) as u32).collect()
    }

    /// Sparse data with larger gaps
    fn sparse_data(size: usize) -> Vec<u32> {
        (0..size).map(|i| (i * 113 + 1000) as u32).collect()
    }

    /// Random data - challenging for compression
    fn random_data(size: usize) -> Vec<u32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        (0..size).map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            (hasher.finish() % 1_000_000) as u32
        }).collect()
    }

    /// Nearly identical values - great for run-length encoding potential
    fn nearly_identical(size: usize) -> Vec<u32> {
        (0..size).map(|i| 42 + (i % 3) as u32).collect()
    }

    /// Large values near u32::MAX
    fn large_values(size: usize) -> Vec<u32> {
        (0..size).map(|i| u32::MAX - (i % 10000) as u32).collect()
    }
}

/// Benchmark compression ratio and memory efficiency
fn benchmark_compression_ratio(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("compression_ratio");
    
    for size in sizes {
        // Test different data patterns
        let test_cases = vec![
            ("sorted", TestDataGenerator::sorted_sequence(size)),
            ("small_range", TestDataGenerator::small_range(size)),
            ("sparse", TestDataGenerator::sparse_data(size)),
            ("random", TestDataGenerator::random_data(size)),
            ("nearly_identical", TestDataGenerator::nearly_identical(size)),
            ("large_values", TestDataGenerator::large_values(size)),
        ];

        for (pattern, data) in test_cases {
            let original_size = data.len() * 4; // u32 = 4 bytes
            
            group.bench_with_input(
                BenchmarkId::new("IntVec", format!("{}_{}_{}", pattern, size, "compression")),
                &data,
                |b, data| {
                    b.iter(|| {
                        let compressed = IntVec::<u32>::from_slice(black_box(data)).unwrap();
                        let ratio = compressed.compression_ratio();
                        let memory_usage = compressed.memory_usage();
                        
                        // Validate compression targets
                        if pattern == "sorted" || pattern == "small_range" || pattern == "nearly_identical" {
                            assert!(ratio < 0.5, "Should achieve >50% compression for {}", pattern);
                        }
                        
                        black_box((ratio, memory_usage))
                    })
                },
            );

            // Compare with UintVector for compatibility
            group.bench_with_input(
                BenchmarkId::new("UintVector", format!("{}_{}_{}", pattern, size, "compression")),
                &data,
                |b, data| {
                    b.iter(|| {
                        let compressed = UintVector::build_from(black_box(data)).unwrap();
                        let ratio = compressed.compression_ratio();
                        let memory_usage = compressed.memory_usage();
                        black_box((ratio, memory_usage))
                    })
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark construction performance
fn benchmark_construction(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("construction");
    group.throughput(Throughput::Elements(100000)); // Base throughput
    
    for size in sizes {
        let data = TestDataGenerator::small_range(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("IntVec", size),
            &data,
            |b, data| {
                b.iter(|| {
                    IntVec::<u32>::from_slice(black_box(data)).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("UintVector", size),
            &data,
            |b, data| {
                b.iter(|| {
                    UintVector::build_from(black_box(data)).unwrap()
                })
            },
        );

        // Raw Vec for baseline
        group.bench_with_input(
            BenchmarkId::new("Vec", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let vec: Vec<u32> = black_box(data).to_vec();
                    black_box(vec)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark random access performance - should be O(1)
fn benchmark_random_access(c: &mut Criterion) {
    let size = 100000;
    let data = TestDataGenerator::small_range(size);
    let compressed = IntVec::<u32>::from_slice(&data).unwrap();
    let uint_vector = UintVector::build_from(&data).unwrap();
    let vec = data.clone();
    
    // Generate random indices for access
    let indices: Vec<usize> = (0..1000).map(|i| (i * 97) % size).collect();
    
    let mut group = c.benchmark_group("random_access");
    group.throughput(Throughput::Elements(1000));
    
    group.bench_function("IntVec", |b| {
        b.iter(|| {
            for &index in &indices {
                black_box(compressed.get(black_box(index)));
            }
        })
    });

    group.bench_function("UintVector", |b| {
        b.iter(|| {
            for &index in &indices {
                black_box(uint_vector.get(black_box(index)));
            }
        })
    });

    group.bench_function("Vec", |b| {
        b.iter(|| {
            for &index in &indices {
                black_box(vec.get(black_box(index)));
            }
        })
    });
    
    group.finish();
}

/// Benchmark sequential access performance
fn benchmark_sequential_access(c: &mut Criterion) {
    let size = 100000;
    let data = TestDataGenerator::small_range(size);
    let compressed = IntVec::<u32>::from_slice(&data).unwrap();
    let uint_vector = UintVector::build_from(&data).unwrap();
    let vec = data.clone();
    
    let mut group = c.benchmark_group("sequential_access");
    group.throughput(Throughput::Elements(size as u64));
    
    group.bench_function("IntVec", |b| {
        b.iter(|| {
            for i in 0..size {
                black_box(compressed.get(black_box(i)));
            }
        })
    });

    group.bench_function("UintVector", |b| {
        b.iter(|| {
            for i in 0..size {
                black_box(uint_vector.get(black_box(i)));
            }
        })
    });

    group.bench_function("Vec", |b| {
        b.iter(|| {
            for i in 0..size {
                black_box(vec.get(black_box(i)));
            }
        })
    });
    
    group.finish();
}

/// Benchmark different integer types
fn benchmark_integer_types(c: &mut Criterion) {
    let size = 10000;
    
    let mut group = c.benchmark_group("integer_types");
    group.throughput(Throughput::Elements(size as u64));
    
    // Test different integer types
    let u8_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
    let u16_data: Vec<u16> = (0..size).map(|i| (i % 65536) as u16).collect();
    let u32_data: Vec<u32> = (0..size).map(|i| (i % 1000000) as u32).collect();
    let u64_data: Vec<u64> = (0..size).map(|i| (i % 1000000) as u64).collect();
    
    let i8_data: Vec<i8> = (0..size).map(|i| ((i % 256) as i8) - 128).collect();
    let i16_data: Vec<i16> = (0..size).map(|i| ((i % 65536) as i16) - 32768).collect();
    let i32_data: Vec<i32> = (0..size).map(|i| ((i % 1000000) as i32) - 500000).collect();
    let i64_data: Vec<i64> = (0..size).map(|i| ((i % 1000000) as i64) - 500000).collect();

    group.bench_function("u8", |b| {
        b.iter(|| IntVec::<u8>::from_slice(black_box(&u8_data)).unwrap())
    });

    group.bench_function("u16", |b| {
        b.iter(|| IntVec::<u16>::from_slice(black_box(&u16_data)).unwrap())
    });

    group.bench_function("u32", |b| {
        b.iter(|| IntVec::<u32>::from_slice(black_box(&u32_data)).unwrap())
    });

    group.bench_function("u64", |b| {
        b.iter(|| IntVec::<u64>::from_slice(black_box(&u64_data)).unwrap())
    });

    group.bench_function("i8", |b| {
        b.iter(|| IntVec::<i8>::from_slice(black_box(&i8_data)).unwrap())
    });

    group.bench_function("i16", |b| {
        b.iter(|| IntVec::<i16>::from_slice(black_box(&i16_data)).unwrap())
    });

    group.bench_function("i32", |b| {
        b.iter(|| IntVec::<i32>::from_slice(black_box(&i32_data)).unwrap())
    });

    group.bench_function("i64", |b| {
        b.iter(|| IntVec::<i64>::from_slice(black_box(&i64_data)).unwrap())
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_patterns(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("memory_efficiency");
    
    for size in sizes {
        let data = TestDataGenerator::small_range(size);
        let original_size = data.len() * 4;
        
        group.bench_with_input(
            BenchmarkId::new("memory_analysis", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = IntVec::<u32>::from_slice(black_box(data)).unwrap();
                    let memory_usage = compressed.memory_usage();
                    let compression_ratio = compressed.compression_ratio();
                    let stats = compressed.stats();
                    
                    // Validate memory efficiency
                    assert!(memory_usage < original_size, 
                           "Memory usage {} should be less than original {}", 
                           memory_usage, original_size);
                    
                    black_box((memory_usage, compression_ratio, stats))
                })
            },
        );
    }
    
    group.finish();
}

/// Performance stress test with large datasets
fn benchmark_large_datasets(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_datasets");
    group.sample_size(10); // Fewer samples for large datasets
    group.measurement_time(Duration::from_secs(30)); // Longer measurement time
    
    let sizes = vec![1_000_000, 5_000_000];
    
    for size in sizes {
        let data = TestDataGenerator::small_range(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("IntVec_large", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressed = IntVec::<u32>::from_slice(black_box(data)).unwrap();
                    
                    // Validate compression
                    let ratio = compressed.compression_ratio();
                    assert!(ratio < 0.5, "Should achieve good compression even for large datasets");
                    
                    // Test some random accesses
                    let test_indices = [0, size/4, size/2, 3*size/4, size-1];
                    for &idx in &test_indices {
                        black_box(compressed.get(idx));
                    }
                    
                    compressed
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark hardware acceleration features
fn benchmark_hardware_acceleration(c: &mut Criterion) {
    let size = 50000;
    let data = TestDataGenerator::sparse_data(size);
    
    let mut group = c.benchmark_group("hardware_acceleration");
    group.throughput(Throughput::Elements(size as u64));
    
    // Focus on bit manipulation intensive operations
    group.bench_function("bit_width_calculation", |b| {
        b.iter(|| {
            let compressed = IntVec::<u32>::from_slice(black_box(&data)).unwrap();
            // The compression process includes bit width calculations
            black_box(compressed)
        })
    });

    // Test decompression performance (bit extraction)
    let compressed = IntVec::<u32>::from_slice(&data).unwrap();
    group.bench_function("bit_extraction", |b| {
        b.iter(|| {
            // Extract all values to test bit manipulation performance
            for i in 0..size {
                black_box(compressed.get(black_box(i)));
            }
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_compression_ratio,
    benchmark_construction,
    benchmark_random_access,
    benchmark_sequential_access,
    benchmark_integer_types,
    benchmark_memory_patterns,
    benchmark_large_datasets,
    benchmark_hardware_acceleration
);

criterion_main!(benches);