use criterion::{black_box, criterion_group, criterion_main, Criterion};
use infini_zip::{FastVec, FastStr, BitVector, RankSelect256};

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
    let fast_str = FastStr::from_str(&data);
    
    c.bench_function("FastStr hash", |b| {
        b.iter(|| black_box(fast_str.hash_fast()));
    });
}

fn benchmark_fast_str_operations(c: &mut Criterion) {
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit";
    let fast_str = FastStr::from_str(text);
    let needle = FastStr::from_str("dolor");
    
    let mut group = c.benchmark_group("FastStr Operations");
    
    group.bench_function("find", |b| {
        b.iter(|| black_box(fast_str.find(needle)));
    });
    
    group.bench_function("starts_with", |b| {
        let prefix = FastStr::from_str("Lorem");
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

criterion_group!(
    benches,
    benchmark_fast_vec_push,
    benchmark_fast_vec_vs_vec,
    benchmark_fast_str_hash,
    benchmark_fast_str_operations,
    benchmark_succinct_data_structures
);
criterion_main!(benches);