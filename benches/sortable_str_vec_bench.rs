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
                b.iter(|| {
                    let mut vec = SortableStrVec::new();
                    for s in strings {
                        vec.push_str(s).unwrap();
                    }
                    vec.sort().unwrap();
                    black_box(vec.get_sorted(0));
                });
            },
        );
        
        // Benchmark Vec<String> sorting
        group.bench_with_input(
            BenchmarkId::new("Vec<String>::sort", size),
            &test_strings,
            |b, strings| {
                b.iter(|| {
                    let mut vec: Vec<String> = strings.clone();
                    vec.sort();
                    black_box(&vec[0]);
                });
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
                    b.iter(|| {
                        let mut vec = SortableStrVec::new();
                        for s in strings {
                            vec.push_str(s).unwrap();
                        }
                        vec.radix_sort().unwrap();
                        black_box(vec.get_sorted(0));
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    group.bench_function("SortableStrVec_10k_strings", |b| {
        b.iter(|| {
            let mut vec = SortableStrVec::new();
            for i in 0..10000 {
                vec.push_str(&format!("string_{:06}", i)).unwrap();
            }
            let (vec_size, our_size, ratio) = vec.memory_savings_vs_vec_string();
            black_box((vec_size, our_size, ratio));
        });
    });
    
    group.bench_function("Vec<String>_10k_strings", |b| {
        b.iter(|| {
            let mut vec = Vec::new();
            for i in 0..10000 {
                vec.push(format!("string_{:06}", i));
            }
            let size = vec.capacity() * std::mem::size_of::<String>() + 
                      vec.iter().map(|s| s.capacity()).sum::<usize>();
            black_box(size);
        });
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