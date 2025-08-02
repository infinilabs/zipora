use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use infini_zip::{
    FastVec, FastStr, BitVector, RankSelect256, GoldHashMap,
    EntropyStats, HuffmanEncoder, HuffmanTree
};
use std::collections::HashMap;
use std::ffi::c_void;

#[cfg(feature = "mmap")]
use infini_zip::{MemoryMappedInput, DataInput};

use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

// External C functions from the C++ wrapper
#[link(name = "topling_zip_wrapper")]
extern "C" {
    // Vector operations
    fn cpp_valvec_create() -> *mut c_void;
    fn cpp_valvec_destroy(vec: *mut c_void);
    fn cpp_valvec_push(vec: *mut c_void, value: i32);
    fn cpp_valvec_size(vec: *mut c_void) -> usize;
    fn cpp_valvec_capacity(vec: *mut c_void) -> usize;
    fn cpp_valvec_get(vec: *mut c_void, index: usize) -> i32;
    fn cpp_valvec_reserve(vec: *mut c_void, capacity: usize);
    
    // String operations
    fn cpp_fstring_create(data: *const u8, len: usize) -> *mut c_void;
    fn cpp_fstring_destroy(fstr: *mut c_void);
    fn cpp_fstring_hash(fstr: *mut c_void) -> u64;
    fn cpp_fstring_find(fstr: *mut c_void, needle: *const u8, needle_len: usize) -> i64;
    fn cpp_fstring_substring(fstr: *mut c_void, start: usize, len: usize) -> *mut c_void;
    fn cpp_fstring_length(fstr: *mut c_void) -> usize;
    fn cpp_fstring_data(fstr: *mut c_void) -> *const u8;
    
    // Rank-select operations
    fn cpp_rank_select_create(bits: *const u64, bit_count: usize) -> *mut c_void;
    fn cpp_rank_select_destroy(rs: *mut c_void);
    fn cpp_rank_select_rank1(rs: *mut c_void, pos: usize) -> usize;
    fn cpp_rank_select_select1(rs: *mut c_void, k: usize) -> usize;
    
    // Performance measurement
    fn cpp_get_memory_usage() -> u64;
    fn cpp_get_allocation_count() -> u64;
    fn cpp_reset_counters();
    fn cpp_warmup_caches();
    fn cpp_measure_allocation_speed(count: usize, size: usize) -> f64;
    fn cpp_measure_hash_speed(data: *const u8, len: usize, iterations: usize) -> f64;
}

/// Safe wrapper for C++ valvec operations
struct CppValVec {
    ptr: *mut c_void,
}

impl CppValVec {
    fn new() -> Self {
        unsafe {
            Self {
                ptr: cpp_valvec_create(),
            }
        }
    }
    
    fn push(&mut self, value: i32) {
        unsafe {
            cpp_valvec_push(self.ptr, value);
        }
    }
    
    fn size(&self) -> usize {
        unsafe {
            cpp_valvec_size(self.ptr)
        }
    }
    
    fn capacity(&self) -> usize {
        unsafe {
            cpp_valvec_capacity(self.ptr)
        }
    }
    
    fn get(&self, index: usize) -> i32 {
        unsafe {
            cpp_valvec_get(self.ptr, index)
        }
    }
    
    fn reserve(&mut self, capacity: usize) {
        unsafe {
            cpp_valvec_reserve(self.ptr, capacity);
        }
    }
}

impl Drop for CppValVec {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cpp_valvec_destroy(self.ptr);
            }
        }
    }
}

/// Safe wrapper for C++ fstring operations
struct CppFString {
    ptr: *mut c_void,
}

impl CppFString {
    fn new(data: &[u8]) -> Self {
        unsafe {
            Self {
                ptr: cpp_fstring_create(data.as_ptr(), data.len()),
            }
        }
    }
    
    fn hash(&self) -> u64 {
        unsafe {
            cpp_fstring_hash(self.ptr)
        }
    }
    
    fn find(&self, needle: &[u8]) -> Option<usize> {
        unsafe {
            let result = cpp_fstring_find(self.ptr, needle.as_ptr(), needle.len());
            if result >= 0 {
                Some(result as usize)
            } else {
                None
            }
        }
    }
    
    fn substring(&self, start: usize, len: usize) -> Option<CppFString> {
        unsafe {
            let ptr = cpp_fstring_substring(self.ptr, start, len);
            if ptr.is_null() {
                None
            } else {
                Some(CppFString { ptr })
            }
        }
    }
    
    fn length(&self) -> usize {
        unsafe {
            cpp_fstring_length(self.ptr)
        }
    }
}

impl Drop for CppFString {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cpp_fstring_destroy(self.ptr);
            }
        }
    }
}

/// Safe wrapper for C++ rank-select operations
struct CppRankSelect {
    ptr: *mut c_void,
}

impl CppRankSelect {
    fn new(bits: &[u64], bit_count: usize) -> Self {
        unsafe {
            Self {
                ptr: cpp_rank_select_create(bits.as_ptr(), bit_count),
            }
        }
    }
    
    fn rank1(&self, pos: usize) -> usize {
        unsafe {
            cpp_rank_select_rank1(self.ptr, pos)
        }
    }
    
    fn select1(&self, k: usize) -> usize {
        unsafe {
            cpp_rank_select_select1(self.ptr, k)
        }
    }
}

impl Drop for CppRankSelect {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cpp_rank_select_destroy(self.ptr);
            }
        }
    }
}

/// Memory tracking utilities
struct MemoryStats {
    initial_usage: u64,
    initial_allocations: u64,
}

impl MemoryStats {
    fn new() -> Self {
        unsafe {
            Self {
                initial_usage: cpp_get_memory_usage(),
                initial_allocations: cpp_get_allocation_count(),
            }
        }
    }
    
    fn current_usage(&self) -> u64 {
        unsafe {
            cpp_get_memory_usage() - self.initial_usage
        }
    }
    
    fn current_allocations(&self) -> u64 {
        unsafe {
            cpp_get_allocation_count() - self.initial_allocations
        }
    }
    
    fn reset() {
        unsafe {
            cpp_reset_counters();
        }
    }
}

/// Comprehensive vector performance comparison
fn benchmark_vector_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Operations Comparison");
    
    // Test different vector sizes for scaling analysis
    let sizes = vec![1_000, 10_000, 100_000];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Rust FastVec push operations
        group.bench_with_input(
            BenchmarkId::new("Rust FastVec push", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = FastVec::new();
                    for i in 0..size {
                        vec.push(black_box(i as i32)).unwrap();
                    }
                    vec
                });
            },
        );
        
        // C++ valvec push operations
        group.bench_with_input(
            BenchmarkId::new("C++ valvec push", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = CppValVec::new();
                    for i in 0..size {
                        vec.push(black_box(i as i32));
                    }
                    vec
                });
            },
        );
        
        // Pre-reserved vector comparison
        group.bench_with_input(
            BenchmarkId::new("Rust FastVec push (reserved)", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = FastVec::with_capacity(size).unwrap();
                    for i in 0..size {
                        vec.push(black_box(i as i32)).unwrap();
                    }
                    vec
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("C++ valvec push (reserved)", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = CppValVec::new();
                    vec.reserve(size);
                    for i in 0..size {
                        vec.push(black_box(i as i32));
                    }
                    vec
                });
            },
        );
    }
    
    group.finish();
}

/// Memory usage and allocation count comparison for vectors
fn benchmark_vector_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Memory Usage");
    
    group.bench_function("Rust FastVec memory efficiency", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let mut vec = FastVec::new();
                for i in 0..10_000 {
                    vec.push(black_box(i as i32)).unwrap();
                }
                // Vector automatically dropped
            }
            
            start.elapsed()
        });
    });
    
    group.bench_function("C++ valvec memory efficiency", |b| {
        b.iter_custom(|iters| {
            MemoryStats::reset();
            let start = std::time::Instant::now();
            
            for _ in 0..iters {
                let mut vec = CppValVec::new();
                for i in 0..10_000 {
                    vec.push(black_box(i as i32));
                }
                // Vector automatically dropped
            }
            
            start.elapsed()
        });
    });
    
    group.finish();
}

/// String operations performance comparison
fn benchmark_string_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("String Operations Comparison");
    
    let medium_text = "The quick brown fox jumps over the lazy dog. ".repeat(10);
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100);
    let test_strings = vec![
        ("Short", "Hello World!"),
        ("Medium", medium_text.as_str()),
        ("Long", long_text.as_str()),
    ];
    
    for (size_name, text) in test_strings {
        let text_bytes = text.as_bytes();
        
        // Hash performance comparison
        group.bench_function(&format!("Rust FastStr hash {}", size_name), |b| {
            let fast_str = FastStr::from_string(text);
            b.iter(|| black_box(fast_str.hash_fast()));
        });
        
        group.bench_function(&format!("C++ fstring hash {}", size_name), |b| {
            let cpp_str = CppFString::new(text_bytes);
            b.iter(|| black_box(cpp_str.hash()));
        });
        
        // Find performance comparison
        let needle = if text.len() > 20 { &text[10..15] } else { "o" };
        let needle_bytes = needle.as_bytes();
        
        group.bench_function(&format!("Rust FastStr find {}", size_name), |b| {
            let fast_str = FastStr::from_string(text);
            let needle_str = FastStr::from_string(needle);
            b.iter(|| black_box(fast_str.find(needle_str)));
        });
        
        group.bench_function(&format!("C++ fstring find {}", size_name), |b| {
            let cpp_str = CppFString::new(text_bytes);
            b.iter(|| black_box(cpp_str.find(needle_bytes)));
        });
        
        // Substring performance comparison (if text is long enough)
        if text.len() > 20 {
            group.bench_function(&format!("Rust FastStr substring {}", size_name), |b| {
                let fast_str = FastStr::from_string(text);
                b.iter(|| black_box(fast_str.substring(5, 10)));
            });
            
            group.bench_function(&format!("C++ fstring substring {}", size_name), |b| {
                let cpp_str = CppFString::new(text_bytes);
                b.iter(|| black_box(cpp_str.substring(5, 10)));
            });
        }
    }
    
    group.finish();
}

/// String creation and memory overhead comparison
fn benchmark_string_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("String Memory Comparison");
    
    let test_data = "Hello, world! This is a test string.";
    let iterations = 1000;
    
    group.throughput(Throughput::Elements(iterations));
    
    group.bench_function("Rust FastStr creation", |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let fast_str = FastStr::from_string(test_data);
                black_box(fast_str);
            }
        });
    });
    
    group.bench_function("C++ fstring creation", |b| {
        b.iter(|| {
            for _ in 0..iterations {
                let cpp_str = CppFString::new(test_data.as_bytes());
                black_box(cpp_str);
            }
        });
    });
    
    group.finish();
}

/// Succinct data structures comparison
fn benchmark_succinct_structures_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Succinct Data Structures Comparison");
    
    // Create test bit patterns
    let sizes = vec![10_000, 100_000];
    
    for size in sizes {
        // Create bit vector with known pattern (every 7th bit set)
        let mut rust_bv = BitVector::new();
        let mut cpp_bits = Vec::new();
        let mut current_word = 0u64;
        let mut bit_count = 0;
        
        for i in 0..size {
            let bit = i % 7 == 0;
            rust_bv.push(bit).unwrap();
            
            if bit {
                current_word |= 1u64 << (bit_count % 64);
            }
            bit_count += 1;
            
            if bit_count % 64 == 0 {
                cpp_bits.push(current_word);
                current_word = 0;
            }
        }
        
        if bit_count % 64 != 0 {
            cpp_bits.push(current_word);
        }
        
        // BitVector creation comparison
        group.bench_function(&format!("Rust BitVector creation {}", size), |b| {
            b.iter(|| {
                let mut bv = BitVector::new();
                for i in 0..size {
                    bv.push(black_box(i % 7 == 0)).unwrap();
                }
                bv
            });
        });
        
        // RankSelect construction comparison
        group.bench_function(&format!("Rust RankSelect256 construction {}", size), |b| {
            b.iter(|| {
                let rs = RankSelect256::new(black_box(rust_bv.clone())).unwrap();
                rs
            });
        });
        
        group.bench_function(&format!("C++ RankSelect construction {}", size), |b| {
            b.iter(|| {
                let rs = CppRankSelect::new(black_box(&cpp_bits), size);
                rs
            });
        });
        
        // Query performance comparison
        let rust_rs = RankSelect256::new(rust_bv.clone()).unwrap();
        let cpp_rs = CppRankSelect::new(&cpp_bits, size);
        
        group.bench_function(&format!("Rust rank1 queries {}", size), |b| {
            b.iter(|| {
                for i in (0..size).step_by(100) {
                    black_box(rust_rs.rank1(black_box(i)));
                }
            });
        });
        
        group.bench_function(&format!("C++ rank1 queries {}", size), |b| {
            b.iter(|| {
                for i in (0..size).step_by(100) {
                    black_box(cpp_rs.rank1(black_box(i)));
                }
            });
        });
        
        group.bench_function(&format!("Rust select1 queries {}", size), |b| {
            let max_rank = rust_rs.rank1(size - 1);
            b.iter(|| {
                for i in (0..max_rank).step_by(max_rank / 100 + 1) {
                    black_box(rust_rs.select1(black_box(i)));
                }
            });
        });
        
        group.bench_function(&format!("C++ select1 queries {}", size), |b| {
            b.iter(|| {
                for i in (0..size/7).step_by(size/700 + 1) {
                    black_box(cpp_rs.select1(black_box(i)));
                }
            });
        });
    }
    
    group.finish();
}

/// Hash map operations comparison
fn benchmark_hashmap_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("HashMap Operations Comparison");
    
    let sizes = vec![1_000, 10_000];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Insertion performance
        group.bench_with_input(
            BenchmarkId::new("Rust GoldHashMap insert", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map = GoldHashMap::new();
                    for i in 0..size {
                        let key = format!("key_{}", i);
                        map.insert(black_box(key), black_box(i)).unwrap();
                    }
                    map
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("std::HashMap insert", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map = HashMap::new();
                    for i in 0..size {
                        let key = format!("key_{}", i);
                        map.insert(black_box(key), black_box(i));
                    }
                    map
                });
            },
        );
        
        // Lookup performance
        let mut gold_map = GoldHashMap::new();
        let mut std_map = HashMap::new();
        
        for i in 0..size {
            let key = format!("key_{}", i);
            gold_map.insert(key.clone(), i).unwrap();
            std_map.insert(key, i);
        }
        
        group.bench_with_input(
            BenchmarkId::new("Rust GoldHashMap lookup", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    for i in 0..size / 10 {
                        let key = format!("key_{}", black_box(i));
                        black_box(gold_map.get(&key));
                    }
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("std::HashMap lookup", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    for i in 0..size / 10 {
                        let key = format!("key_{}", black_box(i));
                        black_box(std_map.get(&key));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Memory mapping performance comparison (if available)
#[cfg(feature = "mmap")]
fn benchmark_memory_mapping_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Mapping Comparison");
    
    let data_sizes = vec![
        ("1KB", 1024),
        ("1MB", 1024 * 1024),
        ("10MB", 10 * 1024 * 1024),
    ];
    
    for (size_name, size) in data_sizes {
        let test_data = vec![42u8; size];
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&test_data).unwrap();
        temp_file.flush().unwrap();
        let file_path = temp_file.path();
        
        group.throughput(Throughput::Bytes(size as u64));
        
        // Memory mapped reading performance
        group.bench_function(&format!("Rust MemoryMappedInput {}", size_name), |b| {
            b.iter(|| {
                let file = File::open(file_path).unwrap();
                let mut mmap_input = MemoryMappedInput::new(file).unwrap();
                let mut buffer = vec![0u8; size];
                let mut pos = 0;
                while pos < size {
                    let chunk = std::cmp::min(1024, size - pos);
                    mmap_input.read_bytes(&mut buffer[pos..pos + chunk]).unwrap();
                    pos += chunk;
                }
                black_box(buffer)
            });
        });
        
        // Standard file I/O for comparison
        group.bench_function(&format!("Standard File I/O {}", size_name), |b| {
            b.iter(|| {
                use std::io::Read;
                let mut file = File::open(file_path).unwrap();
                let mut buffer = vec![0u8; size];
                file.read_exact(&mut buffer).unwrap();
                black_box(buffer)
            });
        });
    }
    
    group.finish();
}

#[cfg(not(feature = "mmap"))]
fn benchmark_memory_mapping_comparison(_c: &mut Criterion) {
    // No-op when mmap feature is disabled
}

/// Entropy coding comparison (Rust only since C++ wrapper doesn't include entropy coding)
fn benchmark_entropy_coding_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Entropy Coding Performance");
    
    let test_datasets = vec![
        ("Random", (0..10000).map(|i| (i * 17 + 13) as u8).collect::<Vec<_>>()),
        ("Biased", "hello world! ".repeat(1000).into_bytes()),
        ("Text", "the quick brown fox jumps over the lazy dog. ".repeat(200).into_bytes()),
    ];
    
    for (name, data) in test_datasets {
        group.throughput(Throughput::Bytes(data.len() as u64));
        
        // Entropy calculation
        group.bench_function(&format!("Entropy calculation {}", name), |b| {
            b.iter(|| {
                let entropy = EntropyStats::calculate_entropy(black_box(&data));
                black_box(entropy)
            });
        });
        
        // Huffman tree construction
        group.bench_function(&format!("Huffman tree construction {}", name), |b| {
            b.iter(|| {
                let tree = HuffmanTree::from_data(black_box(&data)).unwrap();
                black_box(tree)
            });
        });
        
        // Huffman encoding
        group.bench_function(&format!("Huffman encoding {}", name), |b| {
            let encoder = HuffmanEncoder::new(&data).unwrap();
            b.iter(|| {
                let encoded = encoder.encode(black_box(&data)).unwrap();
                black_box(encoded)
            });
        });
    }
    
    group.finish();
}

/// Cache efficiency and low-level performance analysis
fn benchmark_cache_efficiency_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Cache Efficiency Analysis");
    
    // Warm up caches before starting
    unsafe {
        cpp_warmup_caches();
    }
    
    // Sequential vs random access patterns
    let sizes = vec![1024, 16384, 262144]; // L1, L2, L3 cache sizes approximately
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Sequential access pattern
        group.bench_with_input(
            BenchmarkId::new("Rust FastVec sequential", size),
            &size,
            |b, &size| {
                let mut vec = FastVec::with_capacity(size).unwrap();
                for i in 0..size {
                    vec.push(i as i32).unwrap();
                }
                
                b.iter(|| {
                    let mut sum = 0i64;
                    for i in 0..size {
                        sum += *vec.get(i).unwrap() as i64;
                    }
                    black_box(sum)
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("C++ valvec sequential", size),
            &size,
            |b, &size| {
                let mut vec = CppValVec::new();
                vec.reserve(size);
                for i in 0..size {
                    vec.push(i as i32);
                }
                
                b.iter(|| {
                    let mut sum = 0i64;
                    for i in 0..size {
                        sum += vec.get(i) as i64;
                    }
                    black_box(sum)
                });
            },
        );
    }
    
    group.finish();
}

/// Allocation pattern analysis
fn benchmark_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("Allocation Pattern Analysis");
    
    // Test different allocation sizes and patterns
    let allocation_sizes = vec![64, 1024, 16384];
    let allocation_counts = vec![100, 1000];
    
    for size in allocation_sizes {
        for count in &allocation_counts {
            group.throughput(Throughput::Elements(*count as u64));
            
            // Rust allocation pattern
            group.bench_function(&format!("Rust allocation {}x{}", count, size), |b| {
                b.iter(|| {
                    let mut vecs = Vec::new();
                    for _ in 0..*count {
                        let mut vec = FastVec::with_capacity(size / 4).unwrap(); // 4 bytes per i32
                        for i in 0..size/4 {
                            vec.push(black_box(i as i32)).unwrap();
                        }
                        vecs.push(vec);
                    }
                    black_box(vecs)
                });
            });
            
            // C++ allocation pattern measurement
            group.bench_function(&format!("C++ allocation {}x{}", count, size), |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    
                    for _ in 0..iters {
                        unsafe {
                            let time = cpp_measure_allocation_speed(*count, size);
                            black_box(time);
                        }
                    }
                    
                    start.elapsed()
                });
            });
        }
    }
    
    group.finish();
}

/// Hash function performance comparison
fn benchmark_hash_functions_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hash Function Performance");
    
    let text_512b = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(10);
    let text_4kb = "Large text data for hash performance testing. ".repeat(100);
    let test_strings = vec![
        ("8B", "password"),
        ("64B", "The quick brown fox jumps over the lazy dog, again and again!"),
        ("512B", text_512b.as_str()),
        ("4KB", text_4kb.as_str()),
    ];
    
    for (size_name, text) in test_strings {
        let text_bytes = text.as_bytes();
        
        group.throughput(Throughput::Bytes(text_bytes.len() as u64));
        
        // Rust hash performance
        group.bench_function(&format!("Rust FastStr hash {}", size_name), |b| {
            let fast_str = FastStr::from_string(text);
            b.iter(|| black_box(fast_str.hash_fast()));
        });
        
        // C++ hash performance
        group.bench_function(&format!("C++ fstring hash {}", size_name), |b| {
            b.iter_custom(|iters| {
                let start = std::time::Instant::now();
                unsafe {
                    let time = cpp_measure_hash_speed(text_bytes.as_ptr(), text_bytes.len(), iters as usize);
                    black_box(time);
                }
                start.elapsed()
            });
        });
    }
    
    group.finish();
}

/// Comprehensive benchmark suite combining all comparisons
fn benchmark_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comprehensive Performance Summary");
    group.measurement_time(std::time::Duration::from_secs(30));
    
    // Create a representative workload combining multiple operations
    group.bench_function("Rust comprehensive workload", |b| {
        b.iter(|| {
            // Vector operations
            let mut vec = FastVec::new();
            for i in 0..1000 {
                vec.push(black_box(i)).unwrap();
            }
            
            // String operations
            let text = "The quick brown fox jumps over the lazy dog";
            let fast_str = FastStr::from_string(text);
            let hash = fast_str.hash_fast();
            
            // Hash map operations
            let mut map = GoldHashMap::new();
            for i in 0..100 {
                let key = format!("key_{}", i);
                map.insert(key, i).unwrap();
            }
            
            // Succinct operations
            let mut bv = BitVector::new();
            for i in 0..1000 {
                bv.push(i % 7 == 0).unwrap();
            }
            let rs = RankSelect256::new(bv).unwrap();
            let rank = rs.rank1(500);
            
            black_box((vec, hash, map, rank))
        });
    });
    
    group.bench_function("C++ comprehensive workload", |b| {
        b.iter(|| {
            // Vector operations
            let mut vec = CppValVec::new();
            for i in 0..1000 {
                vec.push(black_box(i));
            }
            
            // String operations
            let text = "The quick brown fox jumps over the lazy dog";
            let cpp_str = CppFString::new(text.as_bytes());
            let hash = cpp_str.hash();
            
            // Note: No C++ hash map in wrapper, skip that part
            
            // Succinct operations (using stubs)
            let bits = vec![0u64; 16]; // 1000 bits worth
            let rs = CppRankSelect::new(&bits, 1000);
            let rank = rs.rank1(500);
            
            black_box((vec, hash, rank))
        });
    });
    
    group.finish();
}

criterion_group!(
    cpp_comparison_benches,
    benchmark_vector_operations_comparison,
    benchmark_vector_memory_comparison,
    benchmark_string_operations_comparison,
    benchmark_string_memory_comparison,
    benchmark_succinct_structures_comparison,
    benchmark_hashmap_comparison,
    benchmark_memory_mapping_comparison,
    benchmark_entropy_coding_performance,
    benchmark_cache_efficiency_comparison,
    benchmark_allocation_patterns,
    benchmark_hash_functions_comparison,
    benchmark_comprehensive_comparison
);

criterion_main!(cpp_comparison_benches);