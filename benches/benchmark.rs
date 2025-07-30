use criterion::{black_box, criterion_group, criterion_main, Criterion};
use infini_zip::{
    FastVec, FastStr, BitVector, RankSelect256, GoldHashMap,
    EntropyStats, HuffmanEncoder, HuffmanTree, RansEncoder, DictionaryBuilder, DictionaryCompressor,
    HuffmanBlobStore, MemoryBlobStore, BlobStore
};
use std::collections::HashMap;

#[cfg(feature = "mmap")]
use infini_zip::{MemoryMappedInput, MemoryMappedOutput, DataInput, DataOutput};

use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

fn benchmark_fast_vec_push(c: &mut Criterion) {
    c.bench_function("FastVec push 100k elements", |b| {
        b.iter(|| {
            let mut vec = FastVec::new();
            for i in 0..100_000 {
                vec.push(black_box(i)).unwrap();
            }
            vec
        });
    });
}

fn benchmark_fast_vec_vs_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Comparison");
    
    group.bench_function("FastVec", |b| {
        b.iter(|| {
            let mut vec = FastVec::new();
            for i in 0..10_000 {
                vec.push(black_box(i)).unwrap();
            }
            vec
        });
    });
    
    group.bench_function("std::Vec", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..10_000 {
                vec.push(black_box(i));
            }
            vec
        });
    });
    
    group.finish();
}

fn benchmark_fast_str_hash(c: &mut Criterion) {
    let data = "The quick brown fox jumps over the lazy dog".repeat(100);
    let fast_str = FastStr::from_string(&data);
    
    c.bench_function("FastStr hash", |b| {
        b.iter(|| black_box(fast_str.hash_fast()));
    });
}

fn benchmark_fast_str_operations(c: &mut Criterion) {
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit";
    let fast_str = FastStr::from_string(text);
    let needle = FastStr::from_string("dolor");
    
    let mut group = c.benchmark_group("FastStr Operations");
    
    group.bench_function("find", |b| {
        b.iter(|| black_box(fast_str.find(needle)));
    });
    
    group.bench_function("starts_with", |b| {
        let prefix = FastStr::from_string("Lorem");
        b.iter(|| black_box(fast_str.starts_with(prefix)));
    });
    
    group.bench_function("substring", |b| {
        b.iter(|| black_box(fast_str.substring(6, 5)));
    });
    
    group.finish();
}

fn benchmark_succinct_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("Succinct Data Structures");
    
    // Create a large bit vector with known pattern
    let mut bv = BitVector::new();
    for i in 0..100_000 {
        bv.push(i % 7 == 0).unwrap(); // Every 7th bit is set
    }
    
    group.bench_function("BitVector creation", |b| {
        b.iter(|| {
            let mut bv = BitVector::new();
            for i in 0..10_000 {
                bv.push(black_box(i % 7 == 0)).unwrap();
            }
            bv
        });
    });
    
    group.bench_function("RankSelect256 construction", |b| {
        b.iter(|| {
            let rs = RankSelect256::new(black_box(bv.clone())).unwrap();
            rs
        });
    });
    
    let rs = RankSelect256::new(bv.clone()).unwrap();
    
    group.bench_function("rank1 operation", |b| {
        b.iter(|| {
            rs.rank1(black_box(50_000))
        });
    });
    
    group.bench_function("select1 operation", |b| {
        b.iter(|| {
            rs.select1(black_box(5_000)).unwrap_or(0)
        });
    });
    
    group.finish();
}

fn benchmark_hash_map_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashMap Comparison");
    
    // Benchmark insertion performance
    group.bench_function("GoldHashMap insert 10k", |b| {
        b.iter(|| {
            let mut map = GoldHashMap::new();
            for i in 0..10_000 {
                let key = format!("key_{}", i);
                map.insert(black_box(key), black_box(i)).unwrap();
            }
            map
        });
    });
    
    group.bench_function("std::HashMap insert 10k", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for i in 0..10_000 {
                let key = format!("key_{}", i);
                map.insert(black_box(key), black_box(i));
            }
            map
        });
    });
    
    // Create pre-populated maps for lookup benchmarks
    let mut gold_map = GoldHashMap::new();
    let mut std_map = HashMap::new();
    
    for i in 0..10_000 {
        let key = format!("key_{}", i);
        gold_map.insert(key.clone(), i).unwrap();
        std_map.insert(key, i);
    }
    
    // Benchmark lookup performance
    group.bench_function("GoldHashMap lookup", |b| {
        b.iter(|| {
            for i in 0..1_000 {
                let key = format!("key_{}", black_box(i));
                black_box(gold_map.get(&key));
            }
        });
    });
    
    group.bench_function("std::HashMap lookup", |b| {
        b.iter(|| {
            for i in 0..1_000 {
                let key = format!("key_{}", black_box(i));
                black_box(std_map.get(&key));
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory mapping performance (Phase 2.5.4)
#[cfg(feature = "mmap")]
fn benchmark_memory_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Mapping Performance");
    
    // Create test data files of different sizes
    let small_data = vec![42u8; 1024]; // 1KB
    let medium_data = vec![42u8; 1024 * 1024]; // 1MB
    let large_data = vec![42u8; 10 * 1024 * 1024]; // 10MB
    
    // Benchmark memory mapped input vs regular file I/O
    for (size_name, data) in [
        ("1KB", &small_data),
        ("1MB", &medium_data), 
        ("10MB", &large_data)
    ].iter() {
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(data).unwrap();
        temp_file.flush().unwrap();
        let file_path = temp_file.path();
        
        // Benchmark memory mapped reading
        group.bench_function(&format!("MemoryMappedInput read {}", size_name), |b| {
            b.iter(|| {
                let file = File::open(file_path).unwrap();
                let mut mmap_input = MemoryMappedInput::new(file).unwrap();
                let mut buffer = vec![0u8; data.len()];
                let mut pos = 0;
                while pos < data.len() {
                    let chunk = std::cmp::min(1024, data.len() - pos);
                    mmap_input.read_bytes(&mut buffer[pos..pos + chunk]).unwrap();
                    pos += chunk;
                }
                black_box(buffer)
            });
        });
        
        // Benchmark regular file I/O for comparison
        group.bench_function(&format!("Regular File read {}", size_name), |b| {
            b.iter(|| {
                use std::io::Read;
                let mut file = File::open(file_path).unwrap();
                let mut buffer = vec![0u8; data.len()];
                file.read_exact(&mut buffer).unwrap();
                black_box(buffer)
            });
        });
    }
    
    // Benchmark memory mapped output
    group.bench_function("MemoryMappedOutput write 1MB", |b| {
        let data = vec![42u8; 1024 * 1024];
        b.iter(|| {
            let temp_file = NamedTempFile::new().unwrap();
            let mut mmap_output = MemoryMappedOutput::create(temp_file.path(), data.len()).unwrap();
            
            for chunk in data.chunks(1024) {
                mmap_output.write_bytes(chunk).unwrap();
            }
            
            black_box(mmap_output)
        });
    });
    
    group.finish();
}

/// Fallback memory mapping benchmark for when mmap feature is disabled
#[cfg(not(feature = "mmap"))]
fn benchmark_memory_mapping(_c: &mut Criterion) {
    // No-op when mmap feature is disabled
}

/// Benchmark entropy coding performance (Phase 3.6)
fn benchmark_entropy_coding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Coding Performance");
    
    // Test data with different entropy characteristics
    let random_data = (0..10000).map(|i| (i * 17 + 13) as u8).collect::<Vec<_>>();
    let biased_data = "hello world! ".repeat(1000).into_bytes();
    let repeated_data = "the quick brown fox jumps over the lazy dog. ".repeat(200).into_bytes();
    
    let test_datasets = [
        ("Random", &random_data),
        ("Biased", &biased_data),
        ("Repeated", &repeated_data),
    ];
    
    for (name, data) in test_datasets.iter() {
        // Benchmark entropy calculation
        group.bench_function(&format!("Entropy calculation {}", name), |b| {
            b.iter(|| {
                let entropy = EntropyStats::calculate_entropy(black_box(data));
                black_box(entropy)
            });
        });
        
        // Benchmark Huffman encoding
        group.bench_function(&format!("Huffman tree construction {}", name), |b| {
            b.iter(|| {
                let tree = HuffmanTree::from_data(black_box(data)).unwrap();
                black_box(tree)
            });
        });
        
        group.bench_function(&format!("Huffman encoding {}", name), |b| {
            let encoder = HuffmanEncoder::new(data).unwrap();
            b.iter(|| {
                let encoded = encoder.encode(black_box(data)).unwrap();
                black_box(encoded)
            });
        });
        
        // Benchmark rANS encoding
        group.bench_function(&format!("rANS encoder creation {}", name), |b| {
            b.iter(|| {
                let mut frequencies = [0u32; 256];
                for &byte in data.iter() {
                    frequencies[byte as usize] += 1;
                }
                let encoder = RansEncoder::new(black_box(&frequencies)).unwrap();
                black_box(encoder)
            });
        });
        
        // Benchmark dictionary compression
        group.bench_function(&format!("Dictionary construction {}", name), |b| {
            b.iter(|| {
                let builder = DictionaryBuilder::new()
                    .min_match_length(3)
                    .max_match_length(20)
                    .max_entries(100);
                let dictionary = builder.build(black_box(data));
                black_box(dictionary)
            });
        });
        
        group.bench_function(&format!("Dictionary compression {}", name), |b| {
            let builder = DictionaryBuilder::new();
            let dictionary = builder.build(data);
            let compressor = DictionaryCompressor::new(dictionary);
            
            b.iter(|| {
                let ratio = compressor.estimate_compression_ratio(black_box(data));
                black_box(ratio)
            });
        });
    }
    
    group.finish();
}

/// Benchmark entropy coding blob store integration
fn benchmark_entropy_blob_store(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Blob Store Performance");
    
    let test_data = "hello world! this is test data for entropy blob store. ".repeat(100).into_bytes();
    
    // Benchmark Huffman blob store
    group.bench_function("HuffmanBlobStore setup", |b| {
        b.iter(|| {
            let inner = MemoryBlobStore::new();
            let mut huffman_store = HuffmanBlobStore::new(inner);
            huffman_store.add_training_data(&test_data);
            huffman_store.build_tree().unwrap();
            black_box(huffman_store)
        });
    });
    
    group.bench_function("HuffmanBlobStore put operations", |b| {
        let inner = MemoryBlobStore::new();
        let mut huffman_store = HuffmanBlobStore::new(inner);
        huffman_store.add_training_data(&test_data);
        huffman_store.build_tree().unwrap();
        
        b.iter(|| {
            let data = b"test data for blob store";
            let id = huffman_store.put(black_box(data)).unwrap();
            black_box(id)
        });
    });
    
    // Benchmark compression effectiveness
    group.bench_function("Compression ratio analysis", |b| {
        b.iter(|| {
            // Test multiple data types
            let datasets = [
                ("Random", (0..1000).map(|i| (i * 17) as u8).collect::<Vec<_>>()),
                ("Text", "the quick brown fox jumps over the lazy dog. ".repeat(50).into_bytes()),
                ("Structured", "{\"key\": \"value\", \"number\": 42}".repeat(100).into_bytes()),
            ];
            
            let mut results = Vec::new();
            for (name, data) in datasets.iter() {
                let entropy = EntropyStats::calculate_entropy(data);
                let theoretical_limit = (1.0 - entropy / 8.0) * 100.0;
                
                if let Ok(encoder) = HuffmanEncoder::new(data) {
                    let ratio = encoder.estimate_compression_ratio(data);
                    let actual_savings = (1.0 - ratio) * 100.0;
                    results.push((name.to_string(), theoretical_limit, actual_savings));
                }
            }
            
            black_box(results)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_fast_vec_push,
    benchmark_fast_vec_vs_vec,
    benchmark_fast_str_hash,
    benchmark_fast_str_operations,
    benchmark_succinct_data_structures,
    benchmark_hash_map_comparison,
    benchmark_memory_mapping,
    benchmark_entropy_coding,
    benchmark_entropy_blob_store
);
criterion_main!(benches);