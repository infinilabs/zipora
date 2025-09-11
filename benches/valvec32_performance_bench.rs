use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use zipora::containers::specialized::ValVec32;

fn bench_push_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_push");
    
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let size = *size;
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark ValVec32::push_panic (optimized for benchmarking)
        group.bench_with_input(
            BenchmarkId::new("ValVec32::push_panic", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = ValVec32::new();
                    for i in 0..size {
                        vec.push_panic(black_box(i as u64));
                    }
                    vec
                });
            },
        );
        
        // Benchmark std::Vec::push for comparison
        group.bench_with_input(
            BenchmarkId::new("std::Vec::push", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::new();
                    for i in 0..size {
                        vec.push(black_box(i as u64));
                    }
                    vec
                });
            },
        );
        
        // Benchmark ValVec32::push_panic with pre-allocation
        group.bench_with_input(
            BenchmarkId::new("ValVec32::push_panic_preallocated", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = ValVec32::with_capacity(size).unwrap();
                    for i in 0..size {
                        vec.push_panic(black_box(i as u64));
                    }
                    vec
                });
            },
        );
        
        // Benchmark std::Vec::push with pre-allocation
        group.bench_with_input(
            BenchmarkId::new("std::Vec::push_preallocated", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::with_capacity(size as usize);
                    for i in 0..size {
                        vec.push(black_box(i as u64));
                    }
                    vec
                });
            },
        );
        
        // Benchmark unchecked_push for maximum performance
        group.bench_with_input(
            BenchmarkId::new("ValVec32::unchecked_push", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = ValVec32::with_capacity(size).unwrap();
                    for i in 0..size {
                        unsafe {
                            vec.unchecked_push(black_box(i as u64));
                        }
                    }
                    vec
                });
            },
        );
    }
    
    group.finish();
}

fn bench_bulk_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_bulk");
    
    let sizes = [100, 1_000, 10_000, 100_000];
    
    for size in sizes.iter() {
        let size = *size;
        let data: Vec<u64> = (0..size as u64).collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark ValVec32::extend_from_slice_copy
        group.bench_with_input(
            BenchmarkId::new("ValVec32::extend_from_slice_copy", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut vec = ValVec32::new();
                    vec.extend_from_slice_copy(black_box(data)).unwrap();
                    vec
                });
            },
        );
        
        // Benchmark std::Vec::extend_from_slice
        group.bench_with_input(
            BenchmarkId::new("std::Vec::extend_from_slice", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut vec = Vec::new();
                    vec.extend_from_slice(black_box(data));
                    vec
                });
            },
        );
        
        // Benchmark ValVec32::push_n_copy
        group.bench_with_input(
            BenchmarkId::new("ValVec32::push_n_copy", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = ValVec32::new();
                    vec.push_n_copy(size, black_box(42u64)).unwrap();
                    vec
                });
            },
        );
        
        // Benchmark std::Vec resize for comparison
        group.bench_with_input(
            BenchmarkId::new("std::Vec::resize", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut vec = Vec::new();
                    vec.resize(size as usize, black_box(42u64));
                    vec
                });
            },
        );
    }
    
    group.finish();
}

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_iteration");
    
    for size in [1_000, 10_000, 100_000].iter() {
        let size = *size;
        
        let mut valvec = ValVec32::with_capacity(size).unwrap();
        let mut stdvec = Vec::with_capacity(size as usize);
        
        for i in 0..size {
            valvec.push_panic(i as u64);
            stdvec.push(i as u64);
        }
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark ValVec32 iteration
        group.bench_with_input(
            BenchmarkId::new("ValVec32::iter", size),
            &valvec,
            |b, vec| {
                b.iter(|| {
                    let sum: u64 = vec.iter().sum();
                    black_box(sum)
                });
            },
        );
        
        // Benchmark std::Vec iteration
        group.bench_with_input(
            BenchmarkId::new("std::Vec::iter", size),
            &stdvec,
            |b, vec| {
                b.iter(|| {
                    let sum: u64 = vec.iter().sum();
                    black_box(sum)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_random_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("valvec32_random_access");
    
    for size in [1_000, 10_000, 100_000].iter() {
        let size = *size;
        
        let mut valvec = ValVec32::with_capacity(size).unwrap();
        let mut stdvec = Vec::with_capacity(size as usize);
        
        for i in 0..size {
            valvec.push_panic(i as u64);
            stdvec.push(i as u64);
        }
        
        // Generate random indices
        let indices: Vec<u32> = (0..1000).map(|i| (i * 997) % size).collect();
        let indices_usize: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
        
        group.throughput(Throughput::Elements(1000));
        
        // Benchmark ValVec32 random access
        group.bench_with_input(
            BenchmarkId::new("ValVec32::index", size),
            &(valvec, indices),
            |b, (vec, indices)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum += vec[idx];
                    }
                    black_box(sum)
                });
            },
        );
        
        // Benchmark std::Vec random access
        group.bench_with_input(
            BenchmarkId::new("std::Vec::index", size),
            &(stdvec, indices_usize),
            |b, (vec, indices)| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &idx in indices {
                        sum += vec[idx];
                    }
                    black_box(sum)
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_push_operations,
    bench_bulk_operations,
    bench_iteration,
    bench_random_access
);
criterion_main!(benches);