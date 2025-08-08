//! Debug test for single space string issue

use zipora::containers::specialized::SortableStrVec;

fn main() {
    println!("=== Debugging single space string issue ===");
    
    let mut vec = SortableStrVec::new();
    vec.push(" ".to_string()).unwrap();
    
    println!("After push:");
    println!("  len(): {}", vec.len());
    println!("  is_empty(): {}", vec.is_empty());
    println!("  get(0): {:?}", vec.get(0));
    
    println!("Before sort - checking internal state...");
    
    // Call sort()
    vec.sort().unwrap();
    
    println!("After sort:");
    println!("  get_sorted(0): {:?}", vec.get_sorted(0));
    
    // Test iterator
    let sorted_result: Vec<_> = vec.iter_sorted().collect();
    println!("  iter_sorted result: {:?}", sorted_result);
    println!("  iter_sorted length: {}", sorted_result.len());
    
    // Verify reference
    let mut reference = vec![" ".to_string()];
    reference.sort();
    println!("  reference sorted: {:?}", reference);
    
    // Test if the issue is with iterator vs collect
    println!("Manual iteration:");
    let mut iter = vec.iter_sorted();
    let mut count = 0;
    while let Some(s) = iter.next() {
        println!("    [{}]: {:?}", count, s);
        count += 1;
    }
}