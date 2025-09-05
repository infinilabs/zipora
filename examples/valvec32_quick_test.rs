//! Quick performance test for ValVec32 optimizations

use std::time::Instant;
use zipora::containers::specialized::ValVec32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ValVec32 Quick Performance Test");
    println!("================================");

    const ITERATIONS: usize = 1_000_000;
    
    // Test ValVec32 push performance
    println!("\nTesting ValVec32 push performance...");
    let start = Instant::now();
    let mut valvec = ValVec32::new();
    
    for i in 0..ITERATIONS {
        valvec.push_panic(i as u32);
    }
    
    let valvec_duration = start.elapsed();
    let valvec_ops_per_sec = ITERATIONS as f64 / valvec_duration.as_secs_f64();
    
    println!("ValVec32:");
    println!("  Duration: {:?}", valvec_duration);
    println!("  Throughput: {:.0} ops/sec", valvec_ops_per_sec);
    println!("  Final length: {}", valvec.len());
    println!("  Final capacity: {}", valvec.capacity());
    
    // Test std::Vec push performance for comparison
    println!("\nTesting std::Vec push performance...");
    let start = Instant::now();
    let mut stdvec = Vec::new();
    
    for i in 0..ITERATIONS {
        stdvec.push(i as u32);
    }
    
    let stdvec_duration = start.elapsed();
    let stdvec_ops_per_sec = ITERATIONS as f64 / stdvec_duration.as_secs_f64();
    
    println!("std::Vec:");
    println!("  Duration: {:?}", stdvec_duration);
    println!("  Throughput: {:.0} ops/sec", stdvec_ops_per_sec);
    println!("  Final length: {}", stdvec.len());
    println!("  Final capacity: {}", stdvec.capacity());
    
    // Calculate performance ratio
    let performance_ratio = valvec_ops_per_sec / stdvec_ops_per_sec;
    println!("\nPerformance Comparison:");
    println!("  Performance ratio: {:.2}x", performance_ratio);
    
    if performance_ratio > 1.0 {
        println!("  ✅ ValVec32 is {:.2}x FASTER than std::Vec", performance_ratio);
    } else {
        println!("  ⚠️  ValVec32 is {:.2}x SLOWER than std::Vec", 1.0 / performance_ratio);
    }
    
    // Memory efficiency comparison
    let valvec_struct_size = std::mem::size_of::<ValVec32<u32>>();
    let stdvec_struct_size = std::mem::size_of::<Vec<u32>>();
    let memory_ratio = valvec_struct_size as f64 / stdvec_struct_size as f64;
    
    println!("\nMemory Efficiency:");
    println!("  ValVec32 struct size: {} bytes", valvec_struct_size);
    println!("  std::Vec struct size: {} bytes", stdvec_struct_size);
    println!("  Struct size ratio: {:.2}x", memory_ratio);
    
    Ok(())
}