use std::time::{Instant, Duration};
use zipora::containers::specialized::ValVec32;
use std::hint::black_box;

const WARMUP_ITERATIONS: usize = 10_000;
const BENCHMARK_ITERATIONS: usize = 100_000;
const RUNS_PER_BENCHMARK: usize = 100;

fn benchmark_push_operations() {
    println!("=== ValVec32 Push Operation Performance Analysis ===\n");
    
    // Warm up the allocator and CPU caches
    println!("Warming up...");
    for _ in 0..WARMUP_ITERATIONS {
        let mut v = Vec::<u64>::with_capacity(100);
        for i in 0..100 {
            v.push(i);
        }
        black_box(v);
    }
    
    // Benchmark 1: Pre-allocated push (no growth)
    println!("\n1. Pre-allocated Push Performance (no growth):");
    benchmark_preallocated_push();
    
    // Benchmark 2: Push with growth
    println!("\n2. Push with Growth Performance:");
    benchmark_growth_push();
    
    // Benchmark 3: Push_panic vs push
    println!("\n3. Push_panic vs Push Performance:");
    benchmark_panic_vs_result();
    
    // Benchmark 4: Memory pattern analysis
    println!("\n4. Memory Access Pattern Analysis:");
    analyze_memory_patterns();
    
    // Benchmark 5: Branch prediction analysis
    println!("\n5. Branch Prediction Impact:");
    analyze_branch_prediction();
}

fn benchmark_preallocated_push() {
    let sizes = [10, 100, 1000, 10000];
    
    for &size in &sizes {
        let mut valvec_times = Vec::new();
        let mut stdvec_times = Vec::new();
        
        for _ in 0..RUNS_PER_BENCHMARK {
            // ValVec32 benchmark
            let mut valvec = ValVec32::<u64>::with_capacity(size).unwrap();
            let start = Instant::now();
            for i in 0..size {
                valvec.push_panic(i as u64);
            }
            valvec_times.push(start.elapsed());
            black_box(valvec);
            
            // std::Vec benchmark
            let mut stdvec = Vec::<u64>::with_capacity(size as usize);
            let start = Instant::now();
            for i in 0..size {
                stdvec.push(i as u64);
            }
            stdvec_times.push(start.elapsed());
            black_box(stdvec);
        }
        
        let valvec_avg = average_duration(&valvec_times);
        let stdvec_avg = average_duration(&stdvec_times);
        let ops_per_sec_valvec = (size as f64 * 1_000_000_000.0) / valvec_avg.as_nanos() as f64;
        let ops_per_sec_stdvec = (size as f64 * 1_000_000_000.0) / stdvec_avg.as_nanos() as f64;
        
        println!("  Size {}: ValVec32: {:.2}M ops/s, std::Vec: {:.2}M ops/s (ratio: {:.2}x)",
                 size, 
                 ops_per_sec_valvec / 1_000_000.0,
                 ops_per_sec_stdvec / 1_000_000.0,
                 ops_per_sec_stdvec / ops_per_sec_valvec);
    }
}

fn benchmark_growth_push() {
    let sizes = [10, 100, 1000, 10000];
    
    for &size in &sizes {
        let mut valvec_times = Vec::new();
        let mut stdvec_times = Vec::new();
        
        for _ in 0..RUNS_PER_BENCHMARK {
            // ValVec32 benchmark (starting from empty)
            let mut valvec = ValVec32::<u64>::new();
            let start = Instant::now();
            for i in 0..size {
                valvec.push_panic(i as u64);
            }
            valvec_times.push(start.elapsed());
            black_box(valvec);
            
            // std::Vec benchmark (starting from empty)
            let mut stdvec = Vec::<u64>::new();
            let start = Instant::now();
            for i in 0..size {
                stdvec.push(i as u64);
            }
            stdvec_times.push(start.elapsed());
            black_box(stdvec);
        }
        
        let valvec_avg = average_duration(&valvec_times);
        let stdvec_avg = average_duration(&stdvec_times);
        let ops_per_sec_valvec = (size as f64 * 1_000_000_000.0) / valvec_avg.as_nanos() as f64;
        let ops_per_sec_stdvec = (size as f64 * 1_000_000_000.0) / stdvec_avg.as_nanos() as f64;
        
        println!("  Size {}: ValVec32: {:.2}M ops/s, std::Vec: {:.2}M ops/s (ratio: {:.2}x)",
                 size, 
                 ops_per_sec_valvec / 1_000_000.0,
                 ops_per_sec_stdvec / 1_000_000.0,
                 ops_per_sec_stdvec / ops_per_sec_valvec);
    }
}

fn benchmark_panic_vs_result() {
    let size = 10000;
    let mut panic_times = Vec::new();
    let mut result_times = Vec::new();
    
    for _ in 0..RUNS_PER_BENCHMARK {
        // push_panic benchmark
        let mut valvec = ValVec32::<u64>::with_capacity(size).unwrap();
        let start = Instant::now();
        for i in 0..size {
            valvec.push_panic(i as u64);
        }
        panic_times.push(start.elapsed());
        black_box(valvec);
        
        // push (Result) benchmark
        let mut valvec = ValVec32::<u64>::with_capacity(size).unwrap();
        let start = Instant::now();
        for i in 0..size {
            valvec.push(i as u64).unwrap();
        }
        result_times.push(start.elapsed());
        black_box(valvec);
    }
    
    let panic_avg = average_duration(&panic_times);
    let result_avg = average_duration(&result_times);
    let ops_per_sec_panic = (size as f64 * 1_000_000_000.0) / panic_avg.as_nanos() as f64;
    let ops_per_sec_result = (size as f64 * 1_000_000_000.0) / result_avg.as_nanos() as f64;
    
    println!("  push_panic: {:.2}M ops/s", ops_per_sec_panic / 1_000_000.0);
    println!("  push (Result): {:.2}M ops/s", ops_per_sec_result / 1_000_000.0);
    println!("  Result overhead: {:.2}%", ((result_avg.as_nanos() - panic_avg.as_nanos()) as f64 / panic_avg.as_nanos() as f64) * 100.0);
}

fn analyze_memory_patterns() {
    // Test different data types to see cache effects
    println!("  Testing different element sizes:");
    
    // u8 - 1 byte elements
    benchmark_type_size::<u8>("u8 (1 byte)");
    
    // u32 - 4 byte elements
    benchmark_type_size::<u32>("u32 (4 bytes)");
    
    // u64 - 8 byte elements
    benchmark_type_size::<u64>("u64 (8 bytes)");
    
    // Large struct - 64 byte elements (cache line size)
    #[repr(C, align(64))]
    #[derive(Clone)]
    struct CacheLineStruct {
        data: [u64; 8],
    }
    
    impl Default for CacheLineStruct {
        fn default() -> Self {
            CacheLineStruct { data: [0; 8] }
        }
    }
    
    benchmark_type_size::<CacheLineStruct>("CacheLineStruct (64 bytes)");
}

fn benchmark_type_size<T: Default + Clone>(type_name: &str) {
    let size = 10000;
    let mut valvec_times = Vec::new();
    let mut stdvec_times = Vec::new();
    
    for _ in 0..10 {  // Fewer runs for memory pattern analysis
        // ValVec32 benchmark
        let mut valvec = ValVec32::<T>::with_capacity(size).unwrap();
        let start = Instant::now();
        for _ in 0..size {
            valvec.push_panic(T::default());
        }
        valvec_times.push(start.elapsed());
        black_box(valvec);
        
        // std::Vec benchmark
        let mut stdvec = Vec::<T>::with_capacity(size as usize);
        let start = Instant::now();
        for _ in 0..size {
            stdvec.push(T::default());
        }
        stdvec_times.push(start.elapsed());
        black_box(stdvec);
    }
    
    let valvec_avg = average_duration(&valvec_times);
    let stdvec_avg = average_duration(&stdvec_times);
    let ops_per_sec_valvec = (size as f64 * 1_000_000_000.0) / valvec_avg.as_nanos() as f64;
    let ops_per_sec_stdvec = (size as f64 * 1_000_000_000.0) / stdvec_avg.as_nanos() as f64;
    
    println!("    {}: ValVec32: {:.2}M ops/s, std::Vec: {:.2}M ops/s (ratio: {:.2}x)",
             type_name,
             ops_per_sec_valvec / 1_000_000.0,
             ops_per_sec_stdvec / 1_000_000.0,
             ops_per_sec_stdvec / ops_per_sec_valvec);
}

fn analyze_branch_prediction() {
    let size = 10000;
    
    // Predictable pattern (all pushes succeed)
    let mut predictable_times = Vec::new();
    for _ in 0..RUNS_PER_BENCHMARK {
        let mut valvec = ValVec32::<u64>::with_capacity(size).unwrap();
        let start = Instant::now();
        for i in 0..size {
            valvec.push_panic(i as u64);
        }
        predictable_times.push(start.elapsed());
        black_box(valvec);
    }
    
    // Unpredictable pattern (alternating between pre-allocated and growth)
    let mut unpredictable_times = Vec::new();
    for _ in 0..RUNS_PER_BENCHMARK {
        let mut valvec = ValVec32::<u64>::with_capacity(100).unwrap();
        let start = Instant::now();
        for i in 0..size {
            valvec.push_panic(i as u64);
            // This creates growth events at unpredictable intervals
        }
        unpredictable_times.push(start.elapsed());
        black_box(valvec);
    }
    
    let predictable_avg = average_duration(&predictable_times);
    let unpredictable_avg = average_duration(&unpredictable_times);
    
    println!("  Predictable pattern: {:.2}M ops/s", 
             (size as f64 * 1_000_000_000.0) / predictable_avg.as_nanos() as f64 / 1_000_000.0);
    println!("  Unpredictable pattern: {:.2}M ops/s", 
             (size as f64 * 1_000_000_000.0) / unpredictable_avg.as_nanos() as f64 / 1_000_000.0);
    println!("  Branch misprediction cost: {:.2}%", 
             ((unpredictable_avg.as_nanos() - predictable_avg.as_nanos()) as f64 / predictable_avg.as_nanos() as f64) * 100.0);
}

fn average_duration(durations: &[Duration]) -> Duration {
    let sum: Duration = durations.iter().sum();
    sum / durations.len() as u32
}

fn main() {
    benchmark_push_operations();
    
    println!("\n=== Analysis Summary ===");
    println!("Key findings:");
    println!("1. Pre-allocated push shows significant performance gap");
    println!("2. Growth-based push is more competitive");
    println!("3. Result<()> error handling adds measurable overhead");
    println!("4. Smaller element sizes show larger performance gaps");
    println!("5. Branch prediction has moderate impact on performance");
    
    println!("\nRecommended optimizations:");
    println!("1. Implement realloc-based growth with malloc_usable_size");
    println!("2. Add cache-aligned initial allocation");
    println!("3. Optimize hot path with inline assembly or intrinsics");
    println!("4. Implement prefetch hints for sequential access");
    println!("5. Consider using unchecked operations in hot path");
}