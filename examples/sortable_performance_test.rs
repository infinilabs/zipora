use std::time::Instant;
use zipora::SortableStrVec;

fn main() {
    println!("SortableStrVec Performance Test");
    println!("================================\n");
    
    // Test different sizes
    for size in [100, 1000, 5000, 10000] {
        println!("Testing with {} strings:", size);
        
        // Generate test data
        let test_strings: Vec<String> = (0..size)
            .map(|i| format!("test_string_{:08}_{}", i * 7919 % size, i))
            .collect();
        
        // Benchmark SortableStrVec
        let start = Instant::now();
        let mut sortable = SortableStrVec::new();
        for s in &test_strings {
            sortable.push_str(s).unwrap();
        }
        sortable.sort().unwrap();
        let sortable_time = start.elapsed();
        
        // Benchmark Vec<String>
        let start = Instant::now();
        let mut vec_string = test_strings.clone();
        vec_string.sort();
        let vec_time = start.elapsed();
        
        // Calculate performance ratio
        let ratio = vec_time.as_secs_f64() / sortable_time.as_secs_f64();
        
        println!("  SortableStrVec: {:?}", sortable_time);
        println!("  Vec<String>:    {:?}", vec_time);
        println!("  Performance:    {:.2}x {}", 
                 ratio,
                 if ratio > 1.0 { "faster ✓" } else { "slower ✗" });
        
        // Check memory efficiency
        let (vec_mem, our_mem, savings) = sortable.memory_savings_vs_vec_string();
        println!("  Memory savings: {:.1}% ({} vs {} bytes)\n", 
                 savings * 100.0, our_mem, vec_mem);
    }
}