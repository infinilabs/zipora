//! Comprehensive benchmarks for Advanced Radix Sort Variants
//!
//! This module provides detailed performance benchmarks for all radix sort strategies,
//! demonstrating performance improvements and optimization benefits.
//!
//! ## Benchmark Categories:
//! 1. **Strategy Comparison** - LSD, MSD, Insertion, Tim, Hybrid approaches
//! 2. **Data Size Performance** - Scalability across different input sizes
//! 3. **Data Pattern Performance** - Performance on different data characteristics
//! 4. **SIMD Acceleration** - Benefits of SIMD optimizations
//! 5. **Parallel vs Sequential** - Parallel execution performance
//! 6. **Memory Pool Performance** - Memory management optimization benefits
//! 7. **CPU Feature Detection** - Performance across different CPU capabilities

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
    BatchSize, PlotConfiguration, AxisScale
};
use std::time::Duration;
use std::sync::Arc;

use zipora::algorithms::radix_sort::{
    AdvancedRadixSortConfig, AdvancedU32RadixSort, AdvancedU64RadixSort,
    AdvancedStringRadixSort, RadixSort, RadixString, SortingStrategy,
    CpuFeatures, DataCharacteristics
};
use zipora::memory::{SecureMemoryPool, SecurePoolConfig};

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

const TINY_SIZE: usize = 100;
const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 10_000;
const LARGE_SIZE: usize = 100_000;
const XLARGE_SIZE: usize = 1_000_000;

const DATA_SIZES: &[usize] = &[TINY_SIZE, SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, XLARGE_SIZE];
const SAMPLE_SIZES_FOR_BENCH: &[usize] = &[SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE]; // Exclude largest for faster benchmarks

const WARMUP_TIME: Duration = Duration::from_millis(100);
const MEASUREMENT_TIME: Duration = Duration::from_secs(3);
const SAMPLE_SIZE: usize = 50;

// =============================================================================
// DATA GENERATION UTILITIES
// =============================================================================

/// Generate test data with different patterns for consistent benchmarks
struct DataGenerator {
    seed: u64,
}

impl DataGenerator {
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Simple LCG for deterministic random numbers
    fn next_u64(&mut self) -> u64 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        self.seed
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Generate random data
    fn generate_random_u32(&mut self, size: usize) -> Vec<u32> {
        (0..size).map(|_| self.next_u32()).collect()
    }

    fn generate_random_u64(&mut self, size: usize) -> Vec<u64> {
        (0..size).map(|_| self.next_u64()).collect()
    }

    /// Generate nearly sorted data (90% sorted)
    fn generate_nearly_sorted_u32(&mut self, size: usize) -> Vec<u32> {
        let mut data: Vec<u32> = (0..size).map(|i| i as u32).collect();
        
        // Randomly swap 10% of elements
        let swaps = size / 10;
        for _ in 0..swaps {
            let i = (self.next_u32() as usize) % size;
            let j = (self.next_u32() as usize) % size;
            data.swap(i, j);
        }
        
        data
    }

    fn generate_nearly_sorted_u64(&mut self, size: usize) -> Vec<u64> {
        let mut data: Vec<u64> = (0..size).map(|i| i as u64).collect();
        
        let swaps = size / 10;
        for _ in 0..swaps {
            let i = (self.next_u32() as usize) % size;
            let j = (self.next_u32() as usize) % size;
            data.swap(i, j);
        }
        
        data
    }

    /// Generate reverse sorted data
    fn generate_reverse_sorted_u32(&mut self, size: usize) -> Vec<u32> {
        (0..size).rev().map(|i| i as u32).collect()
    }

    fn generate_reverse_sorted_u64(&mut self, size: usize) -> Vec<u64> {
        (0..size).rev().map(|i| i as u64).collect()
    }

    /// Generate data with many duplicates
    fn generate_many_duplicates_u32(&mut self, size: usize) -> Vec<u32> {
        let unique_values = (size / 100).max(1); // 1% unique values
        (0..size).map(|i| (i % unique_values) as u32).collect()
    }

    fn generate_many_duplicates_u64(&mut self, size: usize) -> Vec<u64> {
        let unique_values = (size / 100).max(1);
        (0..size).map(|i| (i % unique_values) as u64).collect()
    }

    /// Generate string data with different characteristics
    fn generate_random_strings(&mut self, count: usize, avg_length: usize) -> Vec<Vec<u8>> {
        let chars = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        (0..count).map(|_| {
            let length = (avg_length / 2) + ((self.next_u32() as usize) % avg_length);
            (0..length).map(|_| {
                chars[(self.next_u32() as usize) % chars.len()]
            }).collect()
        }).collect()
    }

    fn generate_sorted_strings(&mut self, count: usize, avg_length: usize) -> Vec<Vec<u8>> {
        let mut strings = self.generate_random_strings(count, avg_length);
        strings.sort();
        strings
    }

    fn generate_prefix_strings(&mut self, count: usize, _prefix_length: usize) -> Vec<Vec<u8>> {
        let prefix = b"common_prefix_";
        (0..count).map(|i| {
            let mut s = prefix.to_vec();
            let suffix = format!("{:010}", i);
            s.extend_from_slice(suffix.as_bytes());
            s
        }).collect()
    }
}

// =============================================================================
// STRATEGY COMPARISON BENCHMARKS
// =============================================================================

fn benchmark_strategy_comparison_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_comparison_u32");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in SAMPLE_SIZES_FOR_BENCH {
        group.throughput(Throughput::Elements(size as u64));

        // Generate test data once
        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u32(size);

        // LSD Radix Sort
        group.bench_with_input(
            BenchmarkId::new("LSD_Radix", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                use_parallel: false,
                                use_simd: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // MSD Radix Sort
        group.bench_with_input(
            BenchmarkId::new("MSD_Radix", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::MsdRadix),
                                use_parallel: false,
                                use_simd: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Insertion Sort
        group.bench_with_input(
            BenchmarkId::new("Insertion", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::Insertion),
                                use_parallel: false,
                                use_simd: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Tim Sort
        group.bench_with_input(
            BenchmarkId::new("TimSort", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::TimSort),
                                use_parallel: false,
                                use_simd: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Adaptive Strategy
        group.bench_with_input(
            BenchmarkId::new("Adaptive", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                adaptive_strategy: true,
                                force_strategy: None,
                                use_parallel: false,
                                use_simd: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Standard library for comparison
        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn benchmark_strategy_comparison_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_comparison_u64");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    for &size in SAMPLE_SIZES_FOR_BENCH {
        group.throughput(Throughput::Elements(size as u64));

        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u64(size);

        // LSD Radix Sort
        group.bench_with_input(
            BenchmarkId::new("LSD_Radix", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU64RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                use_parallel: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Adaptive Strategy
        group.bench_with_input(
            BenchmarkId::new("Adaptive", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU64RadixSort::new().unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Standard library
        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// DATA SIZE PERFORMANCE BENCHMARKS
// =============================================================================

fn benchmark_data_size_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_size_performance");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &size in DATA_SIZES {
        group.throughput(Throughput::Elements(size as u64));

        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u32(size);

        group.bench_with_input(
            BenchmarkId::new("AdvancedRadixSort", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::new().unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("BasicRadixSort", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = RadixSort::new();
                        sorter.sort_u32(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// DATA PATTERN PERFORMANCE BENCHMARKS
// =============================================================================

fn benchmark_data_pattern_performance_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_pattern_performance_u32");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    let size = MEDIUM_SIZE;
    group.throughput(Throughput::Elements(size as u64));

    let mut generator = DataGenerator::new(42);
    
    let patterns = vec![
        ("random", generator.generate_random_u32(size)),
        ("nearly_sorted", generator.generate_nearly_sorted_u32(size)),
        ("reverse_sorted", generator.generate_reverse_sorted_u32(size)),
        ("many_duplicates", generator.generate_many_duplicates_u32(size)),
    ];

    for (pattern_name, test_data) in patterns {
        // Advanced Radix Sort with adaptive strategy
        group.bench_with_input(
            BenchmarkId::new("AdvancedRadix_Adaptive", pattern_name),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::new().unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Advanced Radix Sort with forced LSD strategy
        group.bench_with_input(
            BenchmarkId::new("AdvancedRadix_LSD", pattern_name),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Standard library
        group.bench_with_input(
            BenchmarkId::new("std_sort_unstable", pattern_name),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

fn benchmark_string_data_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_data_patterns");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    let count = SMALL_SIZE; // Use smaller size for string benchmarks
    group.throughput(Throughput::Elements(count as u64));

    let mut generator = DataGenerator::new(42);
    
    let string_patterns = vec![
        ("random_strings", generator.generate_random_strings(count, 10)),
        ("sorted_strings", generator.generate_sorted_strings(count, 10)),
        ("prefix_strings", generator.generate_prefix_strings(count, 5)),
    ];

    for (pattern_name, test_strings) in string_patterns {
        // Convert to RadixString for testing
        let radix_strings: Vec<RadixString> = test_strings.iter()
            .map(|s| RadixString::new(s.as_slice()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("AdvancedRadix_MSD", pattern_name),
            &radix_strings,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedStringRadixSort::with_config(
                            AdvancedRadixSortConfig {
                                force_strategy: Some(SortingStrategy::MsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("AdvancedRadix_Adaptive", pattern_name),
            &radix_strings,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedStringRadixSort::new().unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Compare with standard sort on byte vectors
        group.bench_with_input(
            BenchmarkId::new("std_sort", pattern_name),
            &test_strings,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        data.sort_unstable();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// SIMD ACCELERATION BENCHMARKS
// =============================================================================

fn benchmark_simd_acceleration(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_acceleration");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    for &size in SAMPLE_SIZES_FOR_BENCH {
        group.throughput(Throughput::Elements(size as u64));

        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u32(size);

        // With SIMD enabled
        group.bench_with_input(
            BenchmarkId::new("SIMD_Enabled", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_simd: true,
                                use_parallel: false,
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // With SIMD disabled
        group.bench_with_input(
            BenchmarkId::new("SIMD_Disabled", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_simd: false,
                                use_parallel: false,
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// PARALLEL vs SEQUENTIAL BENCHMARKS
// =============================================================================

fn benchmark_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    // Use larger sizes where parallel benefits are more apparent
    let parallel_sizes = &[LARGE_SIZE, XLARGE_SIZE];

    for &size in parallel_sizes {
        group.throughput(Throughput::Elements(size as u64));

        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u32(size);

        // Parallel execution
        group.bench_with_input(
            BenchmarkId::new("Parallel", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_parallel: true,
                                parallel_threshold: size / 4, // Force parallel execution
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Sequential execution
        group.bench_with_input(
            BenchmarkId::new("Sequential", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_parallel: false,
                                force_strategy: Some(SortingStrategy::LsdRadix),
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// MEMORY POOL PERFORMANCE BENCHMARKS
// =============================================================================

fn benchmark_memory_pool_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pool_performance");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    for &size in SAMPLE_SIZES_FOR_BENCH {
        group.throughput(Throughput::Elements(size as u64));

        let mut generator = DataGenerator::new(42);
        let test_data = generator.generate_random_u32(size);

        // With secure memory pool enabled (using default config)
        group.bench_with_input(
            BenchmarkId::new("SecureMemoryEnabled", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_secure_memory: true,
                                use_parallel: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        // Without secure memory pool (standard allocation)
        group.bench_with_input(
            BenchmarkId::new("StandardAlloc", size),
            &test_data,
            |b, data| {
                b.iter_batched(
                    || data.clone(),
                    |mut data| {
                        let mut sorter = AdvancedU32RadixSort::with_config(
                            AdvancedRadixSortConfig {
                                use_secure_memory: false,
                                use_parallel: false,
                                ..Default::default()
                            }
                        ).unwrap();
                        sorter.sort(&mut data).unwrap();
                        black_box(data)
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

// =============================================================================
// CPU FEATURE DETECTION BENCHMARKS
// =============================================================================

fn benchmark_cpu_feature_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_feature_detection");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);

    let size = MEDIUM_SIZE;
    group.throughput(Throughput::Elements(size as u64));

    let mut generator = DataGenerator::new(42);
    let test_data = generator.generate_random_u32(size);

    // Detect actual CPU features
    let cpu_features = CpuFeatures::detect();

    // Test different feature combinations
    group.bench_function("cpu_feature_detection_overhead", |b| {
        b.iter(|| {
            black_box(CpuFeatures::detect())
        })
    });

    // Test with detected features
    group.bench_with_input(
        BenchmarkId::new("WithDetectedFeatures", "optimal"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::with_config(
                        AdvancedRadixSortConfig {
                            use_simd: cpu_features.has_advanced_simd(),
                            use_parallel: true,
                            ..Default::default()
                        }
                    ).unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    // Test with conservative features (no SIMD)
    group.bench_with_input(
        BenchmarkId::new("ConservativeFeatures", "no_simd"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::with_config(
                        AdvancedRadixSortConfig {
                            use_simd: false,
                            use_parallel: false,
                            ..Default::default()
                        }
                    ).unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

// =============================================================================
// COMPREHENSIVE PERFORMANCE COMPARISON
// =============================================================================

fn benchmark_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(SAMPLE_SIZE);
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let size = LARGE_SIZE;
    group.throughput(Throughput::Elements(size as u64));

    let mut generator = DataGenerator::new(42);
    let test_data = generator.generate_random_u32(size);

    // Advanced Radix Sort - Full optimization
    group.bench_with_input(
        BenchmarkId::new("AdvancedRadix", "FullyOptimized"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::new().unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    // Advanced Radix Sort - SIMD only
    group.bench_with_input(
        BenchmarkId::new("AdvancedRadix", "SIMDOnly"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::with_config(
                        AdvancedRadixSortConfig {
                            use_simd: true,
                            use_parallel: false,
                            ..Default::default()
                        }
                    ).unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    // Advanced Radix Sort - Parallel only
    group.bench_with_input(
        BenchmarkId::new("AdvancedRadix", "ParallelOnly"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::with_config(
                        AdvancedRadixSortConfig {
                            use_simd: false,
                            use_parallel: true,
                            ..Default::default()
                        }
                    ).unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    // Basic Radix Sort
    group.bench_with_input(
        BenchmarkId::new("BasicRadix", "Default"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = RadixSort::new();
                    sorter.sort_u32(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    // Standard library sorts
    group.bench_with_input(
        BenchmarkId::new("std", "sort_unstable"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    data.sort_unstable();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("std", "sort"),
        &test_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    data.sort();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

// =============================================================================
// ALGORITHM ANALYSIS AND STATISTICS
// =============================================================================

fn benchmark_algorithm_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_statistics");
    group.warm_up_time(WARMUP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.sample_size(10); // Smaller sample for detailed analysis

    let size = MEDIUM_SIZE;
    group.throughput(Throughput::Elements(size as u64));

    let mut generator = DataGenerator::new(42);
    
    // Test data characteristics analysis
    let random_data = generator.generate_random_u32(size);
    let nearly_sorted_data = generator.generate_nearly_sorted_u32(size);

    group.bench_function("data_characteristics_analysis", |b| {
        b.iter(|| {
            let chars = DataCharacteristics::analyze_integers(&random_data);
            black_box(chars);
        })
    });

    // Test adaptive strategy selection with full sorting (includes strategy selection overhead)
    group.bench_with_input(
        BenchmarkId::new("adaptive_strategy", "random"),
        &random_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::new().unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.bench_with_input(
        BenchmarkId::new("adaptive_strategy", "nearly_sorted"),
        &nearly_sorted_data,
        |b, data| {
            b.iter_batched(
                || data.clone(),
                |mut data| {
                    let mut sorter = AdvancedU32RadixSort::new().unwrap();
                    sorter.sort(&mut data).unwrap();
                    black_box(data)
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    benches,
    benchmark_strategy_comparison_u32,
    benchmark_strategy_comparison_u64,
    benchmark_data_size_performance,
    benchmark_data_pattern_performance_u32,
    benchmark_string_data_patterns,
    benchmark_simd_acceleration,
    benchmark_parallel_vs_sequential,
    benchmark_memory_pool_performance,
    benchmark_cpu_feature_detection,
    benchmark_comprehensive_comparison,
    benchmark_algorithm_statistics
);

criterion_main!(benches);