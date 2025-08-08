//! Criterion-based benchmarks for specialized containers
//!
//! This module provides high-precision benchmarking using the Criterion framework
//! to validate performance claims and detect regressions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use zipora::containers::specialized::{
    ValVec32, SmallMap, FixedCircularQueue, AutoGrowCircularQueue, UintVector,
    FixedStr8Vec, FixedStr16Vec, SortableStrVec
};

// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================

const SMALL_SIZE: usize = 1_000;
const MEDIUM_SIZE: usize = 10_000;
const LARGE_SIZE: usize = 100_000;
const SIZES: &[usize] = &[SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE];

// =============================================================================
// VALVEC32 BENCHMARKS
// =============================================================================

fn bench_valvec32_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_push");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("ValVec32", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = ValVec32::with_capacity(size.try_into().unwrap()).unwrap();
                for i in 0..size {
                    vec.push(black_box(i as u64)).unwrap();
                }
                black_box(vec)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(size);
                for i in 0..size {
                    vec.push(black_box(i as u64));
                }
                black_box(vec)
            });
        });
    }
    
    group.finish();
}

fn bench_valvec32_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_random_access");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(1000)); // 1000 random accesses
        
        // Setup data
        let mut valvec = ValVec32::with_capacity(size.try_into().unwrap()).unwrap();
        let mut stdvec = Vec::with_capacity(size);
        for i in 0..size {
            valvec.push(i as u64).unwrap();
            stdvec.push(i as u64);
        }
        
        group.bench_with_input(BenchmarkId::new("ValVec32", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..1000 {
                    let index = black_box((i * 73) % size);
                    let value = valvec[index as u32];
                    black_box(value);
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..1000 {
                    let index = black_box((i * 73) % size);
                    let value = stdvec[index];
                    black_box(value);
                }
            });
        });
    }
    
    group.finish();
}

fn bench_valvec32_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_iteration");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        // Setup data
        let mut valvec = ValVec32::with_capacity(size.try_into().unwrap()).unwrap();
        let mut stdvec = Vec::with_capacity(size);
        for i in 0..size {
            valvec.push(i as u64).unwrap();
            stdvec.push(i as u64);
        }
        
        group.bench_with_input(BenchmarkId::new("ValVec32", size), &size, |b, &_size| {
            b.iter(|| {
                let sum: u64 = valvec.iter().sum();
                black_box(sum)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &_size| {
            b.iter(|| {
                let sum: u64 = stdvec.iter().sum();
                black_box(sum)
            });
        });
    }
    
    group.finish();
}

// =============================================================================
// SMALLMAP BENCHMARKS  
// =============================================================================

fn bench_small_map_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_map_operations");
    
    let small_sizes = [4, 8, 16, 32]; // Test sizes where SmallMap should excel
    
    for &size in &small_sizes {
        group.throughput(Throughput::Elements(size as u64 * 2)); // insert + lookup
        
        group.bench_with_input(BenchmarkId::new("SmallMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = SmallMap::new();
                
                // Insert phase
                for i in 0..size {
                    map.insert(black_box(i as i32), black_box(format!("value{}", i))).unwrap();
                }
                
                // Lookup phase
                for i in 0..size {
                    let value = map.get(&black_box(i as i32));
                    black_box(value);
                }
                
                black_box(map)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = HashMap::new();
                
                // Insert phase
                for i in 0..size {
                    map.insert(black_box(i as i32), black_box(format!("value{}", i)));
                }
                
                // Lookup phase
                for i in 0..size {
                    let value = map.get(&black_box(i as i32));
                    black_box(value);
                }
                
                black_box(map)
            });
        });
    }
    
    group.finish();
}

fn bench_small_map_lookup_intensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_map_lookup_intensive");
    
    let size = 8; // Sweet spot for SmallMap
    group.throughput(Throughput::Elements(10000)); // 10k lookups
    
    // Setup maps
    let mut small_map = SmallMap::new();
    let mut hash_map = HashMap::new();
    for i in 0..size {
        small_map.insert(i as u8, i * 100).unwrap();
        hash_map.insert(i as u8, i * 100);
    }
    
    group.bench_function("SmallMap_lookup_intensive", |b| {
        b.iter(|| {
            for i in 0..10000 {
                let key = black_box((i % size) as u8);
                let value = small_map.get(&key);
                black_box(value);
            }
        });
    });
    
    group.bench_function("HashMap_lookup_intensive", |b| {
        b.iter(|| {
            for i in 0..10000 {
                let key = black_box((i % size) as u8);
                let value = hash_map.get(&key);
                black_box(value);
            }
        });
    });
    
    group.finish();
}

// =============================================================================
// CIRCULAR QUEUE BENCHMARKS
// =============================================================================

fn bench_circular_queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("circular_queue_operations");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("AutoGrowCircularQueue", size), &size, |b, &size| {
            b.iter(|| {
                let mut queue = AutoGrowCircularQueue::new();
                
                // Fill and drain pattern
                for i in 0..size {
                    queue.push(black_box(i as i32)).unwrap();
                    if i % 3 == 0 && !queue.is_empty() {
                        let value = queue.pop();
                        black_box(value);
                    }
                }
                
                // Drain remaining
                while !queue.is_empty() {
                    let value = queue.pop();
                    black_box(value);
                }
                
                black_box(queue)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("VecDeque", size), &size, |b, &size| {
            b.iter(|| {
                let mut queue = VecDeque::new();
                
                // Fill and drain pattern
                for i in 0..size {
                    queue.push_back(black_box(i as i32));
                    if i % 3 == 0 && !queue.is_empty() {
                        let value = queue.pop_front();
                        black_box(value);
                    }
                }
                
                // Drain remaining
                while !queue.is_empty() {
                    let value = queue.pop_front();
                    black_box(value);
                }
                
                black_box(queue)
            });
        });
    }
    
    group.finish();
}

fn bench_fixed_circular_queue(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_circular_queue");
    
    group.throughput(Throughput::Elements(100000));
    
    group.bench_function("FixedCircularQueue_ring_buffer", |b| {
        b.iter(|| {
            let mut queue: FixedCircularQueue<i32, 1024> = FixedCircularQueue::new();
            
            // Fill to capacity
            for i in 0..1024 {
                queue.push(black_box(i)).unwrap();
            }
            
            // Ring buffer operations
            for i in 1024..100000 {
                let value = queue.pop();
                black_box(value);
                queue.push(black_box(i)).unwrap();
            }
            
            black_box(queue)
        });
    });
    
    group.finish();
}

// =============================================================================
// UINT VECTOR BENCHMARKS
// =============================================================================

fn bench_uint_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("uint_vector_operations");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        // Use data that should compress well
        let test_data: Vec<u32> = (0..size).map(|i| (i % 1000) as u32).collect();
        
        group.bench_with_input(BenchmarkId::new("UintVector", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = UintVector::new();
                for &value in &test_data {
                    vec.push(black_box(value)).unwrap();
                }
                black_box(vec)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("std::Vec", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = Vec::new();
                for &value in &test_data {
                    vec.push(black_box(value));
                }
                black_box(vec)
            });
        });
    }
    
    group.finish();
}

fn bench_uint_vector_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("uint_vector_access");
    
    let size = MEDIUM_SIZE;
    group.throughput(Throughput::Elements(1000));
    
    // Setup data
    let mut uint_vec = UintVector::new();
    let mut std_vec = Vec::new();
    for i in 0..size {
        let value = (i % 10000) as u32;
        uint_vec.push(value).unwrap();
        std_vec.push(value);
    }
    
    group.bench_function("UintVector_random_access", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let index = black_box((i * 73) % size);
                let value = uint_vec.get(index);
                black_box(value);
            }
        });
    });
    
    group.bench_function("std::Vec_random_access", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let index = black_box((i * 73) % size);
                let value = std_vec.get(index);
                black_box(value);
            }
        });
    });
    
    group.finish();
}

// =============================================================================
// STRING CONTAINER BENCHMARKS
// =============================================================================

fn bench_fixed_str_vec_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_str_vec_operations");
    
    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate test strings that fit in 16 characters
        let test_strings: Vec<String> = (0..size)
            .map(|i| format!("test{:011}", i))
            .collect();
        
        group.bench_with_input(BenchmarkId::new("FixedStr16Vec", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = FixedStr16Vec::with_capacity(size);
                for s in &test_strings {
                    vec.push(black_box(s)).unwrap();
                }
                black_box(vec)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("Vec<String>", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = Vec::with_capacity(size);
                for s in &test_strings {
                    vec.push(black_box(s.clone()));
                }
                black_box(vec)
            });
        });
    }
    
    group.finish();
}

fn bench_fixed_str_vec_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_str_vec_access");
    
    let size = MEDIUM_SIZE;
    group.throughput(Throughput::Elements(1000));
    
    // Setup data
    let mut fixed_vec = FixedStr8Vec::with_capacity(size);
    let mut string_vec = Vec::with_capacity(size);
    for i in 0..size {
        let s = format!("{:07}", i);
        fixed_vec.push(&s).unwrap();
        string_vec.push(s);
    }
    
    group.bench_function("FixedStr8Vec_access", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let index = black_box((i * 73) % size);
                let value = fixed_vec.get(index);
                black_box(value);
            }
        });
    });
    
    group.bench_function("Vec<String>_access", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let index = black_box((i * 73) % size);
                let value = string_vec.get(index);
                black_box(value);
            }
        });
    });
    
    group.finish();
}

fn bench_sortable_str_vec_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortable_str_vec_sorting");
    
    let sizes = [100, 1000, 5000]; // Reasonable sizes for sorting benchmarks
    
    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Generate random strings for sorting
        let test_strings: Vec<String> = (0..size)
            .map(|i| format!("string{:05}_{}", (i * 37) % 10000, i))
            .collect();
        
        group.bench_with_input(BenchmarkId::new("SortableStrVec", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = SortableStrVec::with_capacity(size);
                for s in &test_strings {
                    vec.push_str(black_box(s)).unwrap();
                }
                vec.sort().unwrap();
                black_box(vec)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("Vec<String>", size), &size, |b, &_size| {
            b.iter(|| {
                let mut vec = test_strings.clone();
                vec.sort();
                black_box(vec)
            });
        });
    }
    
    group.finish();
}

// =============================================================================
// MEMORY EFFICIENCY BENCHMARKS
// =============================================================================

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(10)); // Longer measurement for memory analysis
    
    let size = MEDIUM_SIZE;
    
    // ValVec32 memory efficiency
    group.bench_function("ValVec32_memory", |b| {
        b.iter(|| {
            let mut vec = ValVec32::with_capacity(black_box(size.try_into().unwrap())).unwrap();
            for i in 0..size {
                vec.push(black_box(i as u64)).unwrap();
            }
            // Access some elements to prevent optimization
            for i in (0..size).step_by(100) {
                let _ = black_box(vec[i as u32]);
            }
            black_box(vec)
        });
    });
    
    group.bench_function("std::Vec_memory", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(black_box(size));
            for i in 0..size {
                vec.push(black_box(i as u64));
            }
            // Access some elements to prevent optimization
            for i in (0..size).step_by(100) {
                let _ = black_box(vec[i]);
            }
            black_box(vec)
        });
    });
    
    // UintVector compression efficiency
    group.bench_function("UintVector_compression", |b| {
        b.iter(|| {
            let mut vec = UintVector::new();
            for i in 0..size {
                vec.push(black_box((i % 1000) as u32)).unwrap();
            }
            // Access to verify compression works
            for i in (0..size).step_by(100) {
                let _ = black_box(vec.get(i));
            }
            black_box(vec)
        });
    });
    
    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    valvec32_benches,
    bench_valvec32_push,
    bench_valvec32_random_access,
    bench_valvec32_iteration
);

criterion_group!(
    small_map_benches,
    bench_small_map_operations,
    bench_small_map_lookup_intensive
);

criterion_group!(
    circular_queue_benches,
    bench_circular_queue_operations,
    bench_fixed_circular_queue
);

criterion_group!(
    uint_vector_benches,
    bench_uint_vector_operations,
    bench_uint_vector_access
);

criterion_group!(
    string_container_benches,
    bench_fixed_str_vec_operations,
    bench_fixed_str_vec_access,
    bench_sortable_str_vec_sorting
);

criterion_group!(
    memory_benches,
    bench_memory_efficiency
);

criterion_main!(
    valvec32_benches,
    small_map_benches,
    circular_queue_benches,
    uint_vector_benches,
    string_container_benches,
    memory_benches
);