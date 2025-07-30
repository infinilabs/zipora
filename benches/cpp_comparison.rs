//! Benchmark comparison between Rust infini-zip and C++ topling-zip
//!
//! This benchmark suite provides direct performance comparisons between the Rust
//! implementation and the original C++ topling-zip library. It tests equivalent
//! operations to measure relative performance characteristics.
//!
//! Prerequisites:
//! 1. Build the C++ topling-zip library with optimizations
//! 2. Create C bindings for the operations we want to benchmark
//! 3. Link against the C++ library for comparison tests

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infini_zip::{FastVec, FastStr, BitVector, RankSelect256};

// C++ FFI declarations for comparison benchmarks
// These would link to a wrapper around the original C++ topling-zip
extern "C" {
    // Vector operations
    fn cpp_valvec_create() -> *mut std::ffi::c_void;
    fn cpp_valvec_destroy(vec: *mut std::ffi::c_void);
    fn cpp_valvec_push(vec: *mut std::ffi::c_void, value: i32);
    fn cpp_valvec_size(vec: *mut std::ffi::c_void) -> usize;
    fn cpp_valvec_capacity(vec: *mut std::ffi::c_void) -> usize;
    
    // String operations
    fn cpp_fstring_create(data: *const u8, len: usize) -> *mut std::ffi::c_void;
    fn cpp_fstring_destroy(fstr: *mut std::ffi::c_void);
    fn cpp_fstring_hash(fstr: *mut std::ffi::c_void) -> u64;
    fn cpp_fstring_find(fstr: *mut std::ffi::c_void, needle: *const u8, needle_len: usize) -> i64;
    fn cpp_fstring_substring(fstr: *mut std::ffi::c_void, start: usize, len: usize) -> *mut std::ffi::c_void;
    
    // Rank-select operations (if available in C++ version)
    fn cpp_rank_select_create(bits: *const u64, bit_count: usize) -> *mut std::ffi::c_void;
    fn cpp_rank_select_destroy(rs: *mut std::ffi::c_void);
    fn cpp_rank_select_rank1(rs: *mut std::ffi::c_void, pos: usize) -> usize;
    fn cpp_rank_select_select1(rs: *mut std::ffi::c_void, k: usize) -> usize;
}

/// Wrapper for C++ valvec to make it safe to use in benchmarks
struct CppValvec {
    ptr: *mut std::ffi::c_void,
}

impl CppValvec {
    fn new() -> Self {
        Self {
            ptr: unsafe { cpp_valvec_create() },
        }
    }
    
    fn push(&mut self, value: i32) {
        unsafe { cpp_valvec_push(self.ptr, value) };
    }
    
    fn len(&self) -> usize {
        unsafe { cpp_valvec_size(self.ptr) }
    }
    
    fn capacity(&self) -> usize {
        unsafe { cpp_valvec_capacity(self.ptr) }
    }
}

impl Drop for CppValvec {
    fn drop(&mut self) {
        unsafe { cpp_valvec_destroy(self.ptr) };
    }
}

/// Wrapper for C++ fstring
struct CppFstring {
    ptr: *mut std::ffi::c_void,
}

impl CppFstring {
    fn new(data: &[u8]) -> Self {
        Self {
            ptr: unsafe { cpp_fstring_create(data.as_ptr(), data.len()) },
        }
    }
    
    fn hash(&self) -> u64 {
        unsafe { cpp_fstring_hash(self.ptr) }
    }
    
    fn find(&self, needle: &[u8]) -> Option<usize> {
        let result = unsafe { cpp_fstring_find(self.ptr, needle.as_ptr(), needle.len()) };
        if result >= 0 {
            Some(result as usize)
        } else {
            None
        }
    }
}

impl Drop for CppFstring {
    fn drop(&mut self) {
        unsafe { cpp_fstring_destroy(self.ptr) };
    }
}

/// Vector comparison benchmarks
fn benchmark_vector_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Operations Comparison");
    
    // Test different sizes to see scaling behavior
    for size in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("Rust FastVec push", size),
            size,
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
        
        group.bench_with_input(
            BenchmarkId::new("C++ valvec push", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = CppValvec::new();
                    for i in 0..size {
                        vec.push(black_box(i as i32));
                    }
                    vec
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("std::Vec push", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::new();
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

/// String operations comparison
fn benchmark_string_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("String Operations Comparison");
    
    // Test data of varying sizes
    let long_string = "long string ".repeat(100);
    let very_long_string = "very long string ".repeat(1000);
    let test_strings = vec![
        "short",
        "medium length string for testing",
        &long_string,
        &very_long_string,
    ];
    
    for (i, test_str) in test_strings.iter().enumerate() {
        let data = test_str.as_bytes();
        
        // Hash benchmarks
        group.bench_with_input(
            BenchmarkId::new("Rust FastStr hash", i),
            &data,
            |b, data| {
                let fast_str = FastStr::new(data);
                b.iter(|| black_box(fast_str.hash_fast()));
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("C++ fstring hash", i),
            &data,
            |b, data| {
                let cpp_str = CppFstring::new(data);
                b.iter(|| black_box(cpp_str.hash()));
            },
        );
        
        // Find operations
        let needle = b"string";
        if data.len() > needle.len() {
            group.bench_with_input(
                BenchmarkId::new("Rust FastStr find", i),
                &(data, needle),
                |b, (data, needle)| {
                    let fast_str = FastStr::new(data);
                    let needle_str = FastStr::new(*needle);
                    b.iter(|| black_box(fast_str.find(needle_str)));
                },
            );
            
            group.bench_with_input(
                BenchmarkId::new("C++ fstring find", i),
                &(data, needle),
                |b, (data, needle)| {
                    let cpp_str = CppFstring::new(data);
                    b.iter(|| black_box(cpp_str.find(*needle)));
                },
            );
        }
    }
    
    group.finish();
}

/// Memory usage comparison
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage Comparison");
    
    // Test memory allocation patterns
    group.bench_function("Rust FastVec memory pattern", |b| {
        b.iter(|| {
            let mut vecs = Vec::new();
            for size in [10, 100, 1000, 10000] {
                let mut vec = FastVec::with_capacity(size).unwrap();
                for i in 0..size {
                    vec.push(black_box(i as i32)).unwrap();
                }
                vecs.push(vec);
            }
            vecs
        });
    });
    
    group.bench_function("C++ valvec memory pattern", |b| {
        b.iter(|| {
            let mut vecs = Vec::new();
            for size in [10, 100, 1000, 10000] {
                let mut vec = CppValvec::new();
                for i in 0..size {
                    vec.push(black_box(i as i32));
                }
                vecs.push(vec);
            }
            vecs
        });
    });
    
    group.finish();
}

/// Complex workload simulation
fn benchmark_real_world_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("Real World Workload");
    
    // Simulate a text processing workload
    let text_data = include_str!("../README.md").as_bytes();
    let words: Vec<&[u8]> = text_data.split(|&b| b == b' ' || b == b'\n').collect();
    
    group.bench_function("Rust text processing", |b| {
        b.iter(|| {
            let mut word_vec = FastVec::new();
            let mut hash_sum = 0u64;
            
            for word in &words {
                if word.len() > 3 {
                    let fast_str = FastStr::new(word);
                    hash_sum = hash_sum.wrapping_add(fast_str.hash_fast());
                    word_vec.push(fast_str).unwrap();
                }
            }
            
            // Find operations
            let target = FastStr::new(b"performance");
            let mut found_count = 0;
            for word_str in word_vec.iter() {
                if word_str.find(target).is_some() {
                    found_count += 1;
                }
            }
            
            black_box((hash_sum, found_count))
        });
    });
    
    group.bench_function("C++ text processing", |b| {
        b.iter(|| {
            let mut word_vec = Vec::new();
            let mut hash_sum = 0u64;
            
            for word in &words {
                if word.len() > 3 {
                    let cpp_str = CppFstring::new(word);
                    hash_sum = hash_sum.wrapping_add(cpp_str.hash());
                    word_vec.push(cpp_str);
                }
            }
            
            // Find operations
            let target = b"performance";
            let mut found_count = 0;
            for cpp_str in &word_vec {
                if cpp_str.find(target).is_some() {
                    found_count += 1;
                }
            }
            
            black_box((hash_sum, found_count))
        });
    });
    
    group.finish();
}

/// Benchmark compilation and build times (simulated)
fn benchmark_build_characteristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("Build Characteristics");
    
    // Simulate the relative complexity by measuring type construction overhead
    group.bench_function("Rust type construction overhead", |b| {
        b.iter(|| {
            // Measure the overhead of Rust's type system vs C++
            let vec = FastVec::<i32>::new();
            let str_slice = FastStr::new(b"test");
            let bit_vec = BitVector::new();
            black_box((vec, str_slice, bit_vec))
        });
    });
    
    group.finish();
}

/// Performance regression detection
fn benchmark_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("Performance Regression Detection");
    
    // Define baseline expectations based on C++ performance
    let _baseline_expectations = vec![
        ("vector_push_10k", 15.0), // microseconds
        ("string_hash_4kb", 2.0),  // microseconds  
        ("rank_select_query", 0.1), // microseconds
    ];
    
    // Vector performance baseline
    group.bench_function("vector_push_10k", |b| {
        b.iter(|| {
            let mut vec = FastVec::new();
            for i in 0..10_000 {
                vec.push(black_box(i)).unwrap();
            }
            vec
        });
    });
    
    // String hash baseline
    let large_string = "x".repeat(4096);
    group.bench_function("string_hash_4kb", |b| {
        let fast_str = FastStr::from_string(&large_string);
        b.iter(|| black_box(fast_str.hash_fast()));
    });
    
    // Rank-select baseline
    let mut bit_vec = BitVector::new();
    for i in 0..10000 {
        bit_vec.push(i % 7 == 0).unwrap();
    }
    let rs = RankSelect256::new(bit_vec).unwrap();
    
    group.bench_function("rank_select_query", |b| {
        b.iter(|| {
            let pos = black_box(5000);
            black_box(rs.rank1(pos))
        });
    });
    
    group.finish();
}

criterion_group!(
    cpp_comparison,
    benchmark_vector_comparison,
    benchmark_string_comparison,
    benchmark_memory_usage,
    benchmark_real_world_workload,
    benchmark_build_characteristics,
    benchmark_performance_regression
);

criterion_main!(cpp_comparison);