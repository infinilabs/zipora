use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use zipora::entropy::*;
use zipora::error::Result;

fn generate_test_data(size: usize, entropy_level: f64) -> Vec<u8> {
    let mut data = Vec::with_capacity(size);
    
    if entropy_level < 1.0 {
        // Low entropy - mostly repeated bytes
        let pattern = (entropy_level * 256.0) as u8;
        for _ in 0..size {
            data.push(pattern);
        }
    } else if entropy_level < 4.0 {
        // Medium entropy - some patterns
        let pattern_size = (8.0 / entropy_level) as usize;
        let pattern: Vec<u8> = (0..pattern_size).map(|i| i as u8).collect();
        for i in 0..size {
            data.push(pattern[i % pattern.len()]);
        }
    } else {
        // High entropy - more randomized
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        for i in 0..size {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            entropy_level.to_bits().hash(&mut hasher);
            data.push((hasher.finish() % 256) as u8);
        }
    }
    
    data
}

fn bench_huffman_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("huffman_algorithms");
    
    let sizes = vec![1024, 8192, 65536];
    let entropy_levels = vec![0.5, 2.0, 6.0]; // Low, medium, high entropy
    
    for &size in &sizes {
        for &entropy in &entropy_levels {
            let data = generate_test_data(size, entropy);
            
            // Basic Huffman
            group.bench_with_input(
                BenchmarkId::new("basic_huffman", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = HuffmanEncoder::new(data).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Contextual Huffman Order-1
            group.bench_with_input(
                BenchmarkId::new("contextual_huffman_order1", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order1).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Contextual Huffman Order-2
            group.bench_with_input(
                BenchmarkId::new("contextual_huffman_order2", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = ContextualHuffmanEncoder::new(data, HuffmanOrder::Order2).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_rans_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans_algorithms");
    
    let sizes = vec![1024, 8192, 65536];
    let entropy_levels = vec![2.0, 6.0, 7.5]; // Medium to high entropy (rANS works best here)
    
    for &size in &sizes {
        for &entropy in &entropy_levels {
            let data = generate_test_data(size, entropy);
            
            // Calculate frequencies for rANS
            let mut frequencies = [1u32; 256]; // Start with 1 to avoid zeros
            for &byte in &data {
                frequencies[byte as usize] += 1;
            }
            
            // Enhanced rANS 64-bit
            group.bench_with_input(
                BenchmarkId::new("rans64_encoder", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = Rans64Encoder::<ParallelX1>::new(&frequencies).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Parallel rANS x2
            group.bench_with_input(
                BenchmarkId::new("rans64_parallel_x2", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = Rans64Encoder::<ParallelX2>::new(&frequencies).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Parallel rANS x4
            group.bench_with_input(
                BenchmarkId::new("rans64_parallel_x4", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let encoder = Rans64Encoder::<ParallelX4>::new(&frequencies).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_fse_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("fse_algorithms");
    
    let sizes = vec![1024, 8192, 65536];
    let entropy_levels = vec![1.0, 4.0, 6.0]; // Various entropy levels
    
    for &size in &sizes {
        for &entropy in &entropy_levels {
            let data = generate_test_data(size, entropy);
            
            // Enhanced FSE default config
            group.bench_with_input(
                BenchmarkId::new("enhanced_fse_default", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut encoder = EnhancedFseEncoder::new(EnhancedFseConfig::default()).unwrap();
                        let compressed = encoder.compress(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Enhanced FSE fast compression
            group.bench_with_input(
                BenchmarkId::new("enhanced_fse_fast", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut encoder = EnhancedFseEncoder::new(EnhancedFseConfig::fast_compression()).unwrap();
                        let compressed = encoder.compress(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Enhanced FSE high compression
            group.bench_with_input(
                BenchmarkId::new("enhanced_fse_high", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut encoder = EnhancedFseEncoder::new(EnhancedFseConfig::high_compression()).unwrap();
                        let compressed = encoder.compress(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_parallel_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_encoding");
    
    let sizes = vec![8192, 65536, 262144]; // Larger sizes for parallel processing
    let entropy_levels = vec![2.0, 6.0]; // Medium and high entropy
    
    for &size in &sizes {
        for &entropy in &entropy_levels {
            let data = generate_test_data(size, entropy);
            
            // Parallel Huffman x2
            group.bench_with_input(
                BenchmarkId::new("parallel_huffman_x2", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let config = ParallelConfig::default();
                        let mut encoder = ParallelHuffmanEncoder::<ParallelX2Variant>::new(config).unwrap();
                        encoder.train(data).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Parallel Huffman x4
            group.bench_with_input(
                BenchmarkId::new("parallel_huffman_x4", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let config = ParallelConfig::default();
                        let mut encoder = ParallelHuffmanEncoder::<ParallelX4Variant>::new(config).unwrap();
                        encoder.train(data).unwrap();
                        let compressed = encoder.encode(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
            
            // Adaptive Parallel Encoder
            group.bench_with_input(
                BenchmarkId::new("adaptive_parallel", format!("{}_{}", size, entropy)),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut encoder = AdaptiveParallelEncoder::new().unwrap();
                        let compressed = encoder.encode_adaptive(data).unwrap();
                        black_box(compressed);
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_bit_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_operations");
    
    let bit_ops = BitOps::new();
    let test_values: Vec<u64> = (0..1000).map(|i| i * 0x123456789ABCDEF).collect();
    
    group.bench_function("popcount_builtin", |b| {
        b.iter(|| {
            for &value in &test_values {
                black_box(bit_ops.popcount64(value));
            }
        });
    });
    
    group.bench_function("reverse_bits32", |b| {
        b.iter(|| {
            for &value in &test_values {
                black_box(bit_ops.reverse_bits32(value as u32));
            }
        });
    });
    
    group.bench_function("reverse_bits64", |b| {
        b.iter(|| {
            for &value in &test_values {
                black_box(bit_ops.reverse_bits64(value));
            }
        });
    });
    
    // Test PDEP/PEXT if available
    if bit_ops.has_bmi2() {
        group.bench_function("pdep_u64", |b| {
            b.iter(|| {
                for &value in &test_values {
                    black_box(bit_ops.pdep_u64(value, 0xAAAAAAAAAAAAAAAA));
                }
            });
        });
        
        group.bench_function("pext_u64", |b| {
            b.iter(|| {
                for &value in &test_values {
                    black_box(bit_ops.pext_u64(value, 0xAAAAAAAAAAAAAAAA));
                }
            });
        });
    }
    
    group.finish();
}

fn bench_entropy_context(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_context");
    
    let sizes = vec![1024, 8192, 65536];
    
    for &size in &sizes {
        let config = EntropyContextConfig::default();
        let mut context = EntropyContext::with_config(config);
        
        group.bench_with_input(
            BenchmarkId::new("context_get_buffer", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buffer = context.get_buffer(size).unwrap();
                    black_box(buffer);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("context_get_temp_buffer", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buffer = context.get_temp_buffer(size).unwrap();
                    black_box(buffer);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    entropy_benches,
    bench_huffman_algorithms,
    bench_rans_algorithms,
    bench_fse_algorithms,
    bench_parallel_encoding,
    bench_bit_operations,
    bench_entropy_context
);

criterion_main!(entropy_benches);