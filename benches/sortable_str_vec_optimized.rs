use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;
use zipora::containers::specialized::SortableStrVec;

fn benchmark_sortable_str_vec_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortable_str_vec_sorting");
    group.measurement_time(Duration::from_secs(10));

    // Test different dataset sizes to show the optimization impact
    for size in &[100, 500, 1000, 5000, 10000] {
        // Generate test strings
        let strings: Vec<String> = (0..*size)
            .map(|i| format!("test_string_{:08}_{}", i * 7919 % size, i))
            .collect();

        // Benchmark SortableStrVec sorting
        group.bench_with_input(BenchmarkId::new("sortable_str_vec", size), size, |b, _| {
            b.iter_batched(
                || {
                    let mut vec = SortableStrVec::new();
                    for s in &strings {
                        vec.push_str(s).unwrap();
                    }
                    vec
                },
                |mut vec| {
                    vec.sort_lexicographic().unwrap();
                    black_box(vec)
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // Benchmark Vec<String> sorting for comparison
        group.bench_with_input(BenchmarkId::new("vec_string", size), size, |b, _| {
            b.iter_batched(
                || strings.clone(),
                |mut vec| {
                    vec.sort_unstable();
                    black_box(vec)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_memory_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("sortable_memory_vs_performance");

    // Medium dataset (1000 strings)
    let strings: Vec<String> = (0..1000)
        .map(|i| format!("string_{:04}_{}", i * 7919 % 1000, i))
        .collect();

    group.bench_function("sortable_1000_with_precomputed", |b| {
        b.iter_batched(
            || {
                let mut vec = SortableStrVec::new();
                for s in &strings {
                    vec.push_str(s).unwrap();
                }
                vec
            },
            |mut vec| {
                vec.sort_lexicographic().unwrap();
                black_box(vec)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Large dataset (5000 strings)
    let strings_large: Vec<String> = (0..5000)
        .map(|i| format!("string_{:05}_{}", i * 7919 % 5000, i))
        .collect();

    group.bench_function("sortable_5000_with_precomputed", |b| {
        b.iter_batched(
            || {
                let mut vec = SortableStrVec::new();
                for s in &strings_large {
                    vec.push_str(s).unwrap();
                }
                vec
            },
            |mut vec| {
                vec.sort_lexicographic().unwrap();
                black_box(vec)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_sortable_str_vec_sorting,
    benchmark_memory_impact
);
criterion_main!(benches);
