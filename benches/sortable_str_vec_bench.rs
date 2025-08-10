use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use zipora::SortableStrVec;
use std::time::Duration;

fn bench_sortable_str_vec_vs_vec_string(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortable_str_vec_vs_vec_string");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different dataset sizes
    for size in [100, 1000, 5000, 10000].iter() {
        let size = *size;
        
        // Generate test data
        let test_strings: Vec<String> = (0..size)
            .map(|i| format!("test_string_{:08}_{}", i * 7919 % size, i))
            .collect();
        
        // Benchmark SortableStrVec sorting
        group.bench_with_input(
            BenchmarkId::new("SortableStrVec::sort", size),
            &test_strings,
            |b, strings| {
                b.iter_batched(
                    || {
                        let mut vec = SortableStrVec::new();
                        for s in strings {
                            vec.push_str(s).unwrap();
                        }
                        vec
                    },
                    |mut vec| {
                        vec.sort().unwrap();
                        black_box(vec.get_sorted(0).map(|s| s.to_owned()));
                        vec
                    },
                    criterion::BatchSize::SmallInput
                );
            },
        );
        
        // Benchmark Vec<String> sorting
        group.bench_with_input(
            BenchmarkId::new("Vec<String>::sort", size),
            &test_strings,
            |b, strings| {
                b.iter_batched(
                    || strings.clone(),
                    |mut vec| {
                        vec.sort();
                        black_box(vec[0].clone());
                        vec
                    },
                    criterion::BatchSize::SmallInput
                );
            },
        );
        
        // Benchmark radix sort specifically for longer strings
        if size <= 1000 {
            let long_strings: Vec<String> = (0..size)
                .map(|i| format!("this_is_a_much_longer_test_string_to_trigger_radix_sort_{:08}_{}", i * 7919 % size, i))
                .collect();
            
            group.bench_with_input(
                BenchmarkId::new("SortableStrVec::radix_sort", size),
                &long_strings,
                |b, strings| {
                    b.iter_batched(
                        || {
                            let mut vec = SortableStrVec::new();
                            for s in strings {
                                vec.push_str(s).unwrap();
                            }
                            vec
                        },
                        |mut vec| {
                            vec.radix_sort().unwrap();
                            black_box(vec.get_sorted(0).map(|s| s.to_owned()));
                            vec
                        },
                        criterion::BatchSize::SmallInput
                    );
                },
            );
        }
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Pre-generate test data once
    let test_strings: Vec<String> = (0..10000)
        .map(|i| format!("string_{:06}", i))
        .collect();
    
    group.bench_function("SortableStrVec_10k_strings", |b| {
        b.iter_batched(
            || {
                let mut vec = SortableStrVec::new();
                for s in &test_strings {
                    vec.push_str(s).unwrap();
                }
                vec
            },
            |vec| {
                let (vec_size, our_size, ratio) = vec.memory_savings_vs_vec_string();
                black_box((vec_size, our_size, ratio))
            },
            criterion::BatchSize::SmallInput
        );
    });
    
    group.bench_function("Vec<String>_10k_strings", |b| {
        b.iter_batched(
            || test_strings.clone(),
            |vec| {
                let size = vec.capacity() * std::mem::size_of::<String>() + 
                          vec.iter().map(|s| s.capacity()).sum::<usize>();
                black_box(size)
            },
            criterion::BatchSize::SmallInput
        );
    });
    
    group.finish();
}

fn bench_binary_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_search");
    
    // Prepare sorted data
    let mut sortable = SortableStrVec::new();
    let mut vec_string = Vec::new();
    
    for i in 0..10000 {
        let s = format!("string_{:08}", i);
        sortable.push_str(&s).unwrap();
        vec_string.push(s);
    }
    
    sortable.sort().unwrap();
    vec_string.sort();
    
    group.bench_function("SortableStrVec::binary_search", |b| {
        b.iter(|| {
            let result = sortable.binary_search("string_00005000");
            black_box(result);
        });
    });
    
    group.bench_function("Vec<String>::binary_search", |b| {
        b.iter(|| {
            let result = vec_string.binary_search(&"string_00005000".to_string());
            black_box(result);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_sortable_str_vec_vs_vec_string,
    bench_memory_efficiency,
    bench_binary_search
);
criterion_main!(benches);