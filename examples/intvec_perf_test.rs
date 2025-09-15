use zipora::IntVec;
use std::time::Instant;

fn main() -> Result<(), zipora::ZiporaError> {
    // Test dataset: 100,000 u32 elements
    let test_data: Vec<u32> = (0..100_000).collect();
    let data_size_mb = (test_data.len() * 4) as f64 / 1_024_000.0;
    
    println!("Testing IntVec performance with {} elements ({:.2} MB)", 
             test_data.len(), data_size_mb);
    println!("{}", "=".repeat(60));
    
    // Warm up
    for _ in 0..3 {
        let _ = IntVec::from_slice(&test_data)?;
        let _ = IntVec::from_slice_bulk(&test_data)?;
    }
    
    // Test regular from_slice
    let mut regular_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _vec = IntVec::from_slice(&test_data)?;
        let elapsed = start.elapsed();
        regular_times.push(elapsed);
    }
    
    // Test bulk from_slice_bulk
    let mut bulk_times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _vec = IntVec::from_slice_bulk(&test_data)?;
        let elapsed = start.elapsed();
        bulk_times.push(elapsed);
    }
    
    // Calculate statistics
    let regular_avg = regular_times.iter().sum::<std::time::Duration>() / regular_times.len() as u32;
    let bulk_avg = bulk_times.iter().sum::<std::time::Duration>() / bulk_times.len() as u32;
    
    let regular_throughput = data_size_mb / regular_avg.as_secs_f64();
    let bulk_throughput = data_size_mb / bulk_avg.as_secs_f64();
    
    let speedup = bulk_throughput / regular_throughput;
    
    println!("\nResults (average of 10 runs):");
    println!("Regular from_slice:");
    println!("  Time: {:.3} ms", regular_avg.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} MB/s", regular_throughput);
    
    println!("\nBulk from_slice_bulk:");
    println!("  Time: {:.3} ms", bulk_avg.as_secs_f64() * 1000.0);
    println!("  Throughput: {:.1} MB/s", bulk_throughput);
    
    println!("\nSpeedup: {:.2}x", speedup);
    
    if speedup >= 1.0 {
        println!("✅ PASS: Bulk constructor is faster!");
    } else {
        println!("❌ FAIL: Bulk constructor is slower (target: ≥1.0x)");
    }
    
    Ok(())
}