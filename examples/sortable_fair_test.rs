use std::time::Instant;
use zipora::SortableStrVec;

fn main() {
    println!("SortableStrVec Fair Performance Test (Sorting Only)");
    println!("====================================================\n");
    
    // Test different sizes
    for size in [100, 1000, 5000, 10000] {
        println!("Testing with {} strings:", size);
        
        // Generate test data
        let test_strings: Vec<String> = (0..size)
            .map(|i| format!("test_string_{:08}_{}", i * 7919 % size, i))
            .collect();
        
        // Pre-populate SortableStrVec (not timed)
        let mut sortable = SortableStrVec::new();
        for s in &test_strings {
            sortable.push_str(s).unwrap();
        }
        
        // Pre-clone Vec<String> (not timed)
        let vec_string = test_strings.clone();
        
        // Benchmark SortableStrVec sorting ONLY
        let start = Instant::now();
        sortable.sort().unwrap();
        let sortable_time = start.elapsed();
        
        // Benchmark Vec<String> sorting ONLY (with clone overhead for fair comparison)
        let start = Instant::now();
        let mut vec_clone = vec_string.clone();
        vec_clone.sort();
        let vec_time = start.elapsed();
        
        // Calculate performance ratio
        let ratio = vec_time.as_secs_f64() / sortable_time.as_secs_f64();
        
        println!("  SortableStrVec sort: {:?}", sortable_time);
        println!("  Vec<String> clone+sort: {:?}", vec_time);
        println!("  Performance: {:.2}x {}", 
                 ratio,
                 if ratio > 1.15 { "faster ✓✓" } 
                 else if ratio > 1.0 { "faster ✓" } 
                 else { "slower ✗" });
        
        // Now test without clone overhead
        let mut vec_for_sort = test_strings.clone();
        let start = Instant::now();
        vec_for_sort.sort();
        let vec_sort_only = start.elapsed();
        
        let ratio_pure = vec_sort_only.as_secs_f64() / sortable_time.as_secs_f64();
        println!("  Vec<String> sort only: {:?}", vec_sort_only);
        println!("  Pure sort ratio: {:.2}x {}", 
                 ratio_pure,
                 if ratio_pure > 1.0 { "faster ✓" } else { "slower ✗" });
        
        // Check memory efficiency
        let (vec_mem, our_mem, savings) = sortable.memory_savings_vs_vec_string();
        println!("  Memory savings: {:.1}% ({} vs {} bytes)\n", 
                 savings * 100.0, our_mem, vec_mem);
    }
    
    println!("\nTesting with longer strings (triggers different algorithm):");
    for size in [100, 500, 1000] {
        println!("\nTesting {} long strings:", size);
        
        let long_strings: Vec<String> = (0..size)
            .map(|i| format!("this_is_a_much_longer_test_string_to_see_performance_{:08}_{}", i * 7919 % size, i))
            .collect();
        
        // Pre-populate
        let mut sortable = SortableStrVec::new();
        for s in &long_strings {
            sortable.push_str(s).unwrap();
        }
        
        // Benchmark sorting
        let start = Instant::now();
        sortable.sort().unwrap();
        let sortable_time = start.elapsed();
        
        let mut vec_clone = long_strings.clone();
        let start = Instant::now();
        vec_clone.sort();
        let vec_time = start.elapsed();
        
        let ratio = vec_time.as_secs_f64() / sortable_time.as_secs_f64();
        
        println!("  SortableStrVec: {:?}", sortable_time);
        println!("  Vec<String>: {:?}", vec_time);
        println!("  Performance: {:.2}x {}", 
                 ratio,
                 if ratio > 1.15 { "faster ✓✓" } 
                 else if ratio > 1.0 { "faster ✓" } 
                 else { "slower ✗" });
    }
}