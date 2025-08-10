//! Specialized Containers Demo
//!
//! Demonstrates the usage and performance characteristics of the specialized containers:
//! - UintVector: Compressed integer storage
//! - FixedLenStrVec: Fixed-length string storage with SIMD optimizations
//! - SortableStrVec: String vector with specialized sorting capabilities

use std::time::Instant;
use zipora::containers::{FixedStr8Vec, FixedStr16Vec, SortableStrVec, UintVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Specialized Containers Demo ===\n");

    demonstrate_uint_vector()?;
    demonstrate_fixed_len_str_vec()?;
    demonstrate_sortable_str_vec()?;

    Ok(())
}

fn demonstrate_uint_vector() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 UintVector - Compressed Integer Storage");
    println!("==========================================");

    let mut vec = UintVector::new();

    // Add sequential data
    let start = Instant::now();
    for i in 0..1000 {
        vec.push(i)?;
    }
    let insert_time = start.elapsed();

    // Test retrieval
    let start = Instant::now();
    let mut sum = 0u64;
    for i in 0..1000 {
        if let Some(value) = vec.get(i) {
            sum += value as u64;
        }
    }
    let retrieval_time = start.elapsed();

    let (original_size, compressed_size, ratio) = vec.stats();

    println!("✅ Inserted 1000 integers in {:?}", insert_time);
    println!("✅ Retrieved all values in {:?}", retrieval_time);
    println!("✅ Sum verification: {}", sum);
    println!("📊 Original size: {} bytes", original_size);
    println!("📊 Compressed size: {} bytes", compressed_size);
    println!("📊 Compression ratio: {:.2}", ratio);
    println!(
        "💾 Memory efficiency: {:.1}% of original size\n",
        ratio * 100.0
    );

    Ok(())
}

fn demonstrate_fixed_len_str_vec() -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 FixedLenStrVec - Fixed-Length String Storage");
    println!("===============================================");

    // Demonstrate with 8-byte strings
    let mut vec8: FixedStr8Vec = FixedStr8Vec::new();

    let test_strings = vec![
        "apple", "banana", "cherry", "date", "elderb", "fig", "grape",
    ];

    let start = Instant::now();
    for s in &test_strings {
        vec8.push(s)?;
    }
    let insert_time = start.elapsed();

    // Test SIMD-optimized search
    let start = Instant::now();
    let found_banana = vec8.find_exact("banana");
    let found_missing = vec8.find_exact("kiwi");
    let search_time = start.elapsed();

    // Test prefix counting
    let start = Instant::now();
    let prefix_count = vec8.count_prefix("gr");
    let prefix_time = start.elapsed();

    let (vec_string_size, our_size, savings_ratio) = vec8.memory_savings_vs_vec_string();

    println!(
        "✅ Inserted {} strings in {:?}",
        test_strings.len(),
        insert_time
    );
    println!(
        "✅ Found 'banana' at index: {:?} in {:?}",
        found_banana, search_time
    );
    println!("✅ 'kiwi' not found: {:?}", found_missing);
    println!(
        "✅ Strings with prefix 'gr': {} in {:?}",
        prefix_count, prefix_time
    );
    println!("📊 Vec<String> equivalent: {} bytes", vec_string_size);
    println!("📊 FixedLenStrVec size: {} bytes", our_size);
    println!("💾 Memory savings: {:.1}%\n", savings_ratio * 100.0);

    // Demonstrate with 16-byte strings
    println!("📝 FixedStr16Vec - Longer String Support");
    println!("----------------------------------------");

    let mut vec16: FixedStr16Vec = FixedStr16Vec::new();
    let longer_strings = vec![
        "programming",
        "algorithms",
        "data_structures",
        "performance",
        "optimization",
        "benchmarking",
        "profiling",
        "memory_layout",
    ];

    for s in &longer_strings {
        vec16.push(s)?;
    }

    let prefix_opt = vec16.count_prefix("opt");
    println!("✅ Strings with prefix 'opt': {}", prefix_opt);
    println!("✅ Total strings stored: {}", vec16.len());

    Ok(())
}

fn demonstrate_sortable_str_vec() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔤 SortableStrVec - Specialized String Sorting");
    println!("==============================================");

    let mut vec = SortableStrVec::new();

    let test_data = vec![
        "zebra",
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
    ];

    // Insert strings
    let start = Instant::now();
    for s in &test_data {
        vec.push_str(s)?;
    }
    let insert_time = start.elapsed();

    println!(
        "✅ Inserted {} strings in {:?}",
        test_data.len(),
        insert_time
    );

    // Demonstrate lexicographic sorting
    let start = Instant::now();
    vec.sort_lexicographic()?;
    let sort_time = start.elapsed();

    println!("✅ Lexicographic sort completed in {:?}", sort_time);
    println!("📋 Sorted order:");
    for i in 0..vec.len().min(6) {
        if let Some(s) = vec.get_sorted(i) {
            println!("   {}. {}", i + 1, s);
        }
    }

    // Test binary search
    let start = Instant::now();
    let banana_pos = vec.binary_search("banana");
    let missing_pos = vec.binary_search("pineapple");
    let search_time = start.elapsed();

    println!("✅ Binary search completed in {:?}", search_time);
    println!("🔍 'banana' found at sorted position: {:?}", banana_pos);
    println!("🔍 'pineapple' not found: {:?}", missing_pos);

    // Demonstrate sorting by length
    vec.sort_by_length()?;
    println!("\n📏 Sorted by length:");
    for i in 0..vec.len().min(6) {
        if let Some(s) = vec.get_sorted(i) {
            println!("   {} chars: {}", s.len(), s);
        }
    }

    // Performance statistics
    let (total_strings, utilization, last_sort_time, memory_ratio) = vec.stats();
    println!("\n📊 Performance Statistics:");
    println!("   Total strings: {}", total_strings);
    println!("   Storage utilization: {:.1}%", utilization * 100.0);
    println!("   Last sort time: {} μs", last_sort_time);
    println!("   Memory efficiency: {:.1}%", memory_ratio * 100.0);

    // Demonstrate custom sorting
    println!("\n🔤 Custom Sort - Case Insensitive:");
    vec.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()))?;
    for i in 0..vec.len().min(6) {
        if let Some(s) = vec.get_sorted(i) {
            println!("   {}", s);
        }
    }

    Ok(())
}
