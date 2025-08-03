use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use zipora::{DictionaryBuilder, DictionaryCompressor, OptimizedDictionaryCompressor};

fn benchmark_dictionary_compression_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dictionary Compression Performance");

    // Create test data with varying levels of repetition
    let test_cases = vec![
        ("Small Repeated", generate_repeated_data(b"hello world", 50)),
        ("Medium Repeated", generate_repeated_data(b"this is a longer pattern for testing", 100)),
        ("Large Repeated", generate_repeated_data(b"abcdefghijklmnopqrstuvwxyz0123456789", 200)),
        ("Biased Data", generate_biased_data(10000)),
        ("Random Data", generate_random_data(5000)),
    ];

    for (name, data) in test_cases {
        // Benchmark original implementation
        group.bench_with_input(
            BenchmarkId::new("Original Dictionary", name),
            &data,
            |b, data| {
                b.iter(|| {
                    let builder = DictionaryBuilder::new();
                    let dict = builder.build(black_box(data));
                    let compressor = DictionaryCompressor::new(dict);
                    let compressed = compressor.compress(black_box(data)).unwrap();
                    black_box(compressed)
                });
            },
        );

        // Benchmark optimized implementation
        group.bench_with_input(
            BenchmarkId::new("Optimized Dictionary", name),
            &data,
            |b, data| {
                b.iter(|| {
                    let compressor = OptimizedDictionaryCompressor::new(black_box(data)).unwrap();
                    let compressed = compressor.compress(black_box(data)).unwrap();
                    black_box(compressed)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_dictionary_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dictionary Construction Performance");

    let data = generate_repeated_data(b"pattern", 1000);

    group.bench_function("Original Dictionary Build", |b| {
        b.iter(|| {
            let builder = DictionaryBuilder::new();
            let dict = builder.build(black_box(&data));
            black_box(dict)
        });
    });

    group.bench_function("Optimized Dictionary Build", |b| {
        b.iter(|| {
            let compressor = OptimizedDictionaryCompressor::new(black_box(&data)).unwrap();
            black_box(compressor)
        });
    });

    group.finish();
}

fn benchmark_compression_ratio_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression Ratio Analysis");

    let highly_compressible = generate_repeated_data(
        b"this is a very long pattern that should compress extremely well when repeated many times",
        100,
    );

    // Test compression ratio estimation performance
    group.bench_function("Original Compression Ratio", |b| {
        let builder = DictionaryBuilder::new();
        let dict = builder.build(&highly_compressible);
        let compressor = DictionaryCompressor::new(dict);
        
        b.iter(|| {
            let ratio = compressor.estimate_compression_ratio(black_box(&highly_compressible));
            black_box(ratio)
        });
    });

    group.bench_function("Optimized Compression Ratio", |b| {
        let compressor = OptimizedDictionaryCompressor::new(&highly_compressible).unwrap();
        
        b.iter(|| {
            let ratio = compressor.estimate_compression_ratio(black_box(&highly_compressible));
            black_box(ratio)
        });
    });

    group.finish();
}

fn benchmark_decompression_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decompression Performance");

    let data = generate_repeated_data(b"decompression test pattern", 200);

    // Pre-compress with both methods
    let builder = DictionaryBuilder::new();
    let dict = builder.build(&data);
    let original_compressor = DictionaryCompressor::new(dict);
    let original_compressed = original_compressor.compress(&data).unwrap();

    let optimized_compressor = OptimizedDictionaryCompressor::new(&data).unwrap();
    let optimized_compressed = optimized_compressor.compress(&data).unwrap();

    group.bench_function("Original Decompression", |b| {
        b.iter(|| {
            let decompressed = original_compressor.decompress(black_box(&original_compressed)).unwrap();
            black_box(decompressed)
        });
    });

    group.bench_function("Optimized Decompression", |b| {
        b.iter(|| {
            let decompressed = optimized_compressor.decompress(black_box(&optimized_compressed)).unwrap();
            black_box(decompressed)
        });
    });

    group.finish();
}

// Helper functions to generate test data
fn generate_repeated_data(pattern: &[u8], repetitions: usize) -> Vec<u8> {
    let mut data = Vec::new();
    for _ in 0..repetitions {
        data.extend_from_slice(pattern);
    }
    data
}

fn generate_biased_data(size: usize) -> Vec<u8> {
    // Generate data heavily biased towards certain characters
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let byte = match i % 10 {
            0..=6 => b'A', // 70% A's
            7..=8 => b'B', // 20% B's  
            _ => b'C',     // 10% C's
        };
        data.push(byte);
    }
    data
}

fn generate_random_data(size: usize) -> Vec<u8> {
    // Generate pseudo-random data that's harder to compress
    let mut data = Vec::with_capacity(size);
    let mut seed = 12345u32;
    
    for _ in 0..size {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((seed >> 24) as u8);
    }
    data
}

criterion_group!(
    benches,
    benchmark_dictionary_compression_comparison,
    benchmark_dictionary_construction,
    benchmark_compression_ratio_comparison,
    benchmark_decompression_performance
);
criterion_main!(benches);