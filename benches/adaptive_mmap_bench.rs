use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use zipora::{DataInput, MemoryMappedInput};
use zipora::io::AccessPattern;
use std::fs::File;
use std::io::{Write as StdWrite, BufReader, Read as StdRead};
use tempfile::NamedTempFile;

fn create_test_file(size: usize) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().unwrap();
    let data = (0..size).map(|i| (i % 256) as u8).collect::<Vec<u8>>();
    temp_file.write_all(&data).unwrap();
    temp_file.flush().unwrap();
    temp_file
}

fn benchmark_small_file_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Small File Performance (2KB)");
    
    // Test with 2KB file - should use buffered I/O strategy
    let temp_file = create_test_file(2048);
    let file_path = temp_file.path();

    // Benchmark adaptive memory mapping (should use buffered I/O)
    group.bench_function("Adaptive MemoryMappedInput", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new(file).unwrap();
            let mut sum = 0u64;
            while input.remaining() > 0 {
                sum += black_box(input.read_u8().unwrap()) as u64;
            }
            black_box(sum)
        });
    });

    // Benchmark standard BufReader for comparison
    group.bench_function("BufReader", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut reader = BufReader::new(file);
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            let mut sum = 0u64;
            for &byte in &buffer {
                sum += black_box(byte) as u64;
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn benchmark_medium_file_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Medium File Performance (64KB)");
    
    // Test with 64KB file - should use standard memory mapping
    let temp_file = create_test_file(64 * 1024);
    let file_path = temp_file.path();

    // Benchmark adaptive memory mapping (should use standard mmap)
    group.bench_function("Adaptive MemoryMappedInput", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new(file).unwrap();
            let mut sum = 0u64;
            while input.remaining() > 0 {
                sum += black_box(input.read_u8().unwrap()) as u64;
            }
            black_box(sum)
        });
    });

    // Benchmark standard BufReader for comparison
    group.bench_function("BufReader", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut reader = BufReader::new(file);
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer).unwrap();
            let mut sum = 0u64;
            for &byte in &buffer {
                sum += black_box(byte) as u64;
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn benchmark_access_pattern_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Access Pattern Optimization (16KB)");
    
    let temp_file = create_test_file(16 * 1024);
    let file_path = temp_file.path();

    // Test sequential access pattern
    group.bench_function("Sequential Access", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new_with_pattern(file, AccessPattern::Sequential).unwrap();
            let mut sum = 0u64;
            // Sequential read
            while input.remaining() > 0 {
                sum += black_box(input.read_u8().unwrap()) as u64;
            }
            black_box(sum)
        });
    });

    // Test random access pattern
    group.bench_function("Random Access", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new_with_pattern(file, AccessPattern::Random).unwrap();
            let mut sum = 0u64;
            
            // Random access pattern - seek to different positions
            let positions = [100, 8000, 2000, 15000, 500, 12000, 1000, 10000];
            for &pos in &positions {
                if pos < input.len() {
                    input.seek(pos).unwrap();
                    sum += black_box(input.read_u8().unwrap()) as u64;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

fn benchmark_threshold_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("Threshold Boundary Performance");
    
    let test_sizes = vec![
        (4095, "Just below 4KB"),    // Should use buffered I/O
        (4096, "Exactly 4KB"),       // Should use buffered I/O 
        (4097, "Just above 4KB"),    // Should use standard mmap
        (1024 * 1024 - 1, "Just below 1MB"), // Should use standard mmap
    ];

    for (size, label) in test_sizes {
        let temp_file = create_test_file(size);
        let file_path = temp_file.path();

        group.bench_with_input(BenchmarkId::new("adaptive", label), &size, |b, _| {
            b.iter(|| {
                let file = File::open(file_path).unwrap();
                let mut input = MemoryMappedInput::new(file).unwrap();
                
                // Read first 100 bytes to test strategy efficiency
                let mut sum = 0u64;
                for _ in 0..std::cmp::min(100, input.len()) {
                    sum += black_box(input.read_u8().unwrap()) as u64;
                }
                black_box(sum)
            });
        });
    }

    group.finish();
}

fn benchmark_zero_copy_vs_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("Zero-Copy vs Copy Performance (32KB)");
    
    let temp_file = create_test_file(32 * 1024);
    let file_path = temp_file.path();

    // Test zero-copy read (should work for mmap strategy)
    group.bench_function("Zero-Copy Read", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new(file).unwrap();
            let mut sum = 0u64;
            
            // Read in 1KB chunks using zero-copy
            while input.remaining() >= 1024 {
                let slice = input.read_slice_zero_copy(1024).unwrap();
                for &byte in slice {
                    sum += black_box(byte) as u64;
                }
            }
            black_box(sum)
        });
    });

    // Test copying read 
    group.bench_function("Copying Read", |b| {
        b.iter(|| {
            let file = File::open(file_path).unwrap();
            let mut input = MemoryMappedInput::new(file).unwrap();
            let mut sum = 0u64;
            
            // Read in 1KB chunks using copying
            while input.remaining() >= 1024 {
                let data = input.read_slice(1024).unwrap();
                for byte in data {
                    sum += black_box(byte) as u64;
                }
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_small_file_performance,
    benchmark_medium_file_performance,
    benchmark_access_pattern_optimization,
    benchmark_threshold_boundaries,
    benchmark_zero_copy_vs_copy
);
criterion_main!(benches);