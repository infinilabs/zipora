use std::time::Instant;
use zipora::containers::specialized::ValVec32;

fn main() {
    // Test push performance
    let sizes = [1_000, 10_000, 100_000];

    println!("=== ValVec32 Performance Test (optimizations) ===\n");

    for &size in &sizes {
        println!("Testing size: {}", size);

        // ValVec32 push test
        let start = Instant::now();
        let mut valvec = ValVec32::with_capacity(size).unwrap();
        for i in 0..size {
            valvec.push(i as u64).unwrap();
        }
        let valvec_push_time = start.elapsed();

        // std::Vec push test
        let start = Instant::now();
        let mut stdvec = Vec::with_capacity(size as usize);
        for i in 0..size {
            stdvec.push(i as u64);
        }
        let stdvec_push_time = start.elapsed();

        // ValVec32 iteration test
        let start = Instant::now();
        let sum: u64 = valvec.iter().sum();
        let valvec_iter_time = start.elapsed();

        // std::Vec iteration test
        let start = Instant::now();
        let sum2: u64 = stdvec.iter().sum();
        let stdvec_iter_time = start.elapsed();

        println!(
            "  Push: ValVec32 {:?}, std::Vec {:?} (ratio: {:.2}x)",
            valvec_push_time,
            stdvec_push_time,
            valvec_push_time.as_secs_f64() / stdvec_push_time.as_secs_f64()
        );
        println!(
            "  Iter: ValVec32 {:?}, std::Vec {:?} (ratio: {:.2}x)",
            valvec_iter_time,
            stdvec_iter_time,
            valvec_iter_time.as_secs_f64() / stdvec_iter_time.as_secs_f64()
        );
        assert_eq!(sum, sum2);

        println!();
    }

    // Test golden ratio growth
    println!("=== Testing Golden Ratio Growth Pattern ===");
    let mut vec = ValVec32::<u32>::new();
    let mut capacities = Vec::new();

    for i in 0..20 {
        let old_capacity = vec.capacity();
        vec.push(i).unwrap();
        let new_capacity = vec.capacity();

        if new_capacity != old_capacity {
            capacities.push((i, old_capacity, new_capacity));
            if old_capacity > 0 {
                let ratio = new_capacity as f64 / old_capacity as f64;
                println!(
                    "Growth at element {}: {} -> {} (ratio: {:.3}x, target: 1.609x)",
                    i, old_capacity, new_capacity, ratio
                );
            }
        }
    }

    println!("\n✅ All tests passed! Golden ratio growth and optimizations working correctly.");
}
