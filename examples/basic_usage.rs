use infini_zip::{FastVec, FastStr, Result};

fn main() -> Result<()> {
    println!("=== Infini-Zip Rust Demo ===\n");

    // Demonstrate FastVec
    println!("1. FastVec Performance:");
    let mut vec = FastVec::new();
    
    // Add some elements
    for i in 0..1000 {
        vec.push(i)?;
    }
    
    println!("   Created FastVec with {} elements", vec.len());
    println!("   Capacity: {}", vec.capacity());
    println!("   First 10 elements: {:?}", &vec.as_slice()[0..10]);
    
    // Demonstrate realloc optimization
    vec.reserve(10000)?;
    println!("   After reserve(10000), capacity: {}", vec.capacity());
    
    // Demonstrate FastStr
    println!("\n2. FastStr Zero-Copy Operations:");
    let text = "The quick brown fox jumps over the lazy dog";
    let s = FastStr::from_string(text);
    
    println!("   Original: '{}'", s);
    println!("   Length: {} bytes", s.len());
    
    // Zero-copy substring operations
    let words: Vec<_> = s.split(b' ').collect();
    println!("   Split into {} words:", words.len());
    for (i, word) in words.iter().enumerate() {
        println!("     {}: '{}'", i + 1, word);
    }
    
    // Demonstrate find operations
    let fox = FastStr::from_string("fox");
    if let Some(pos) = s.find(fox) {
        println!("   Found 'fox' at position: {}", pos);
    }
    
    // Demonstrate prefix/suffix operations
    println!("   Starts with 'The': {}", s.starts_with(FastStr::from_string("The")));
    println!("   Ends with 'dog': {}", s.ends_with(FastStr::from_string("dog")));
    
    // Demonstrate high-performance hashing
    let hash = s.hash_fast();
    println!("   Fast hash: 0x{:016x}", hash);
    
    // Compare two strings
    let s1 = FastStr::from_string("abc");
    let s2 = FastStr::from_string("abd");
    println!("\n3. String Comparison:");
    println!("   '{}' < '{}': {}", s1, s2, s1 < s2);
    println!("   Common prefix length: {}", s1.common_prefix_len(s2));
    
    // Demonstrate substring operations
    println!("\n4. Substring Operations:");
    let original = FastStr::from_string("Hello, World!");
    println!("   Original: '{}'", original);
    println!("   Prefix(5): '{}'", original.prefix(5));
    println!("   Suffix(6): '{}'", original.suffix(6));
    println!("   Substring(7,5): '{}'", original.substring(7, 5));
    
    // Memory usage comparison
    println!("\n5. Memory Efficiency:");
    println!("   FastStr uses zero-copy: {} bytes overhead per string", 
             std::mem::size_of::<FastStr>());
    println!("   FastVec<i32> overhead: {} bytes", 
             std::mem::size_of::<FastVec<i32>>());
    
    println!("\n=== Demo Complete ===");
    Ok(())
}