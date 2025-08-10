use std::env;
use std::time::Instant;
use zipora::containers::specialized::sortable_str_vec::SortableStrVec;

fn main() {
    // Enable debug output
    unsafe {
        env::set_var("SORTABLE_DEBUG", "1");
    }

    println!("\n=== Testing Topling-zip Algorithm Selection Optimization ===\n");

    // Test 1: Short strings (should use comparison sort by default)
    {
        println!("Test 1: Short strings (10 chars each, should use comparison sort)");
        let mut vec = SortableStrVec::new();
        for i in 0..1000 {
            vec.push_str(&format!("str_{:06}", i)).unwrap();
        }

        let start = Instant::now();
        vec.sort_lexicographic().unwrap();
        let elapsed = start.elapsed();
        println!("  Sort time: {:?}\n", elapsed);
    }

    // Test 2: Long strings (might use radix if threshold is low)
    {
        println!("Test 2: Long strings (100 chars each, should still use comparison with default)");
        let mut vec = SortableStrVec::new();
        for i in 0..1000 {
            let s = format!(
                "this_is_a_very_long_string_for_testing_radix_sort_algorithm_selection_{:030}",
                i
            );
            vec.push_str(&s).unwrap();
        }

        let start = Instant::now();
        vec.sort_lexicographic().unwrap();
        let elapsed = start.elapsed();
        println!("  Sort time: {:?}\n", elapsed);
    }

    // Test 3: Performance comparison - SortableStrVec vs Vec<String>
    {
        println!("Test 3: Performance comparison with 10,000 strings");

        // Prepare test data
        let test_strings: Vec<String> = (0..10000)
            .map(|i| format!("test_string_{:08}", i))
            .collect();

        // Test SortableStrVec with new algorithm selection
        {
            println!("SortableStrVec (intelligent algorithm selection):");
            let mut vec = SortableStrVec::new();

            // Add strings
            let start = Instant::now();
            for s in &test_strings {
                vec.push_str(s).unwrap();
            }
            let add_time = start.elapsed();

            // Sort
            let start = Instant::now();
            vec.sort_lexicographic().unwrap();
            let sort_time = start.elapsed();

            println!("  Add time:  {:?}", add_time);
            println!("  Sort time: {:?}", sort_time);

            // Memory usage
            let (vec_size, our_size, savings) = vec.memory_savings_vs_vec_string();
            println!(
                "  Memory: {} bytes (vs Vec<String>: {} bytes)",
                our_size, vec_size
            );
            println!("  Savings: {:.1}%\n", savings * 100.0);
        }

        // Test Vec<String> baseline
        {
            println!("Vec<String> (baseline):");
            let mut vec = Vec::new();

            // Add strings
            let start = Instant::now();
            for s in &test_strings {
                vec.push(s.clone());
            }
            let add_time = start.elapsed();

            // Sort
            let start = Instant::now();
            vec.sort();
            let sort_time = start.elapsed();

            println!("  Add time:  {:?}", add_time);
            println!("  Sort time: {:?}", sort_time);

            // Memory usage estimate
            let mem_size = vec.len() * (std::mem::size_of::<String>() + 20); // ~20 bytes per string
            println!("  Memory: ~{} bytes\n", mem_size);
        }
    }

    // Test 4: Force radix sort with environment variable
    {
        println!("Test 4: Force radix sort with low threshold (should be much slower)");
        unsafe {
            env::set_var("SORTABLE_STRVEC_MIN_RADIX_LEN", "5");
        } // Very low threshold

        let mut vec = SortableStrVec::new();
        for i in 0..1000 {
            vec.push_str(&format!("str_{:06}", i)).unwrap();
        }

        let start = Instant::now();
        vec.sort_lexicographic().unwrap();
        let elapsed = start.elapsed();
        println!("  Sort time: {:?}\n", elapsed);

        // Reset env var
        unsafe {
            env::remove_var("SORTABLE_STRVEC_MIN_RADIX_LEN");
        }
    }

    println!("=== Algorithm Selection Test Complete ===");
}
