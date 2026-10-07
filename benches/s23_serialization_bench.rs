use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::collections::HashMap;
use std::hint::black_box;
use std::rc::Rc;
use zipora::io::{SerializableType, SliceDataInput, VecDataOutput};

fn serialize_bytes<T: SerializableType>(value: &T) -> Vec<u8> {
    let mut output = VecDataOutput::new();
    value.serialize(&mut output).unwrap();
    output.into_vec()
}

fn bench_s23_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("s23_serialization");

    let vec_u64: Vec<u64> = (0..1_000_000u64).collect();
    let bytes_u64 = serialize_bytes(&vec_u64);
    group.throughput(Throughput::Elements(1_000_000));
    group.bench_function("vec_u64_1M_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_u64));
            black_box(Vec::<u64>::deserialize(&mut input).unwrap())
        });
    });

    let vec_rc: Vec<Rc<u64>> = (0..100_000u64).map(Rc::new).collect();
    let bytes_rc = serialize_bytes(&vec_rc);
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("vec_rc_u64_100k_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_rc));
            black_box(Vec::<Rc<u64>>::deserialize(&mut input).unwrap())
        });
    });

    let vec_vec_u32: Vec<Vec<u32>> = (0..1_000u32)
        .map(|i| (0..100u32).map(|j| i * 100 + j).collect())
        .collect();
    let bytes_vv32 = serialize_bytes(&vec_vec_u32);
    group.throughput(Throughput::Elements(1_000));
    group.bench_function("vec_vec_u32_1000x100_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_vv32));
            black_box(Vec::<Vec<u32>>::deserialize(&mut input).unwrap())
        });
    });
    group.bench_function("vec_vec_u32_1000x100_ser", |b| {
        b.iter(|| {
            let mut output = VecDataOutput::with_capacity(4 + 1_000 * 404);
            black_box(&vec_vec_u32).serialize(&mut output).unwrap();
            black_box(output)
        });
    });

    let vec_vec_u8: Vec<Vec<u8>> = (0..100_000u32).map(|i| vec![i as u8; 3]).collect();
    let bytes_vv8 = serialize_bytes(&vec_vec_u8);
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("vec_vec_u8_100kx3_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_vv8));
            black_box(Vec::<Vec<u8>>::deserialize(&mut input).unwrap())
        });
    });

    let hm: HashMap<u32, u64> = (0..100_000u32).map(|i| (i, i as u64)).collect();
    let bytes_hm = serialize_bytes(&hm);
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("hashmap_u32_u64_100k_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_hm));
            black_box(HashMap::<u32, u64>::deserialize(&mut input).unwrap())
        });
    });

    let vec_box: Vec<Box<u64>> = (0..100_000u64).map(Box::new).collect();
    let bytes_box = serialize_bytes(&vec_box);
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("vec_box_u64_100k_de", |b| {
        b.iter(|| {
            let mut input = SliceDataInput::new(black_box(&bytes_box));
            black_box(Vec::<Box<u64>>::deserialize(&mut input).unwrap())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_s23_serialization);
criterion_main!(benches);
